# ==============================================================================
# UVR Headless Runner - Windows Installation Script (PowerShell)
# ==============================================================================
# This script installs native-style CLI wrappers for Windows.
# By default, it pulls pre-built images from Docker Hub for fast installation.
#
# Usage:
#   .\docker\install.ps1                     # Quick install (pulls from Docker Hub)
#   .\docker\install.ps1 -Cpu                # Force CPU-only installation
#   .\docker\install.ps1 -Gpu                # Force GPU installation (CUDA 12.4)
#   .\docker\install.ps1 -Cuda cu121         # GPU with CUDA 12.1 (driver 530+)
#   .\docker\install.ps1 -Cuda cu124         # GPU with CUDA 12.4 (driver 550+, default)
#   .\docker\install.ps1 -Cuda cu128         # GPU with CUDA 12.8 (driver 560+)
#   .\docker\install.ps1 -Build              # Force local build (slower)
#   .\docker\install.ps1 -Uninstall          # Remove installed wrappers
#
# Image Source:
#   Default: Pulls pre-built images from Docker Hub (fast, ~2-5 min)
#   -Build:  Builds locally from source (slower, ~10-30 min)
#
# CUDA Version Options:
#   cu121 - CUDA 12.1, requires NVIDIA driver 530+
#   cu124 - CUDA 12.4, requires NVIDIA driver 550+ (default, recommended)
#   cu128 - CUDA 12.8, requires NVIDIA driver 560+
#
# ==============================================================================

param(
    [switch]$Cpu,
    [switch]$Gpu,
    [ValidateSet("cu121", "cu124", "cu128")]
    [string]$Cuda = "",
    [switch]$Build,        # Force local build instead of pulling from Docker Hub
    [switch]$Uninstall,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$InstallDir = if ($env:UVR_INSTALL_DIR) { $env:UVR_INSTALL_DIR } else { "$env:LOCALAPPDATA\UVR" }
$ModelsDir = if ($env:UVR_MODELS_DIR) { $env:UVR_MODELS_DIR } else { "$env:USERPROFILE\.uvr_models" }
$ImageName = "uvr-headless-runner"
# Docker Hub image for pre-built images (much faster!)
$DockerHubImage = "chyinan/uvr-headless-runner"
$DefaultCudaVersion = if ($env:UVR_CUDA_VERSION) { $env:UVR_CUDA_VERSION } else { "cu124" }
$ForceBuild = if ($env:UVR_FORCE_BUILD) { $true } else { $false }

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
function Write-Info { param($Message) Write-Host "[INFO] $Message" -ForegroundColor Blue }
function Write-Success { param($Message) Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
function Write-Warn { param($Message) Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }

function Show-Banner {
    Write-Host ""
    Write-Host "=========================================================" -ForegroundColor Cyan
    Write-Host "       UVR Headless Runner - Windows Installation        " -ForegroundColor Cyan
    Write-Host "=========================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Get-CudaDriverRequirement {
    param([string]$CudaVersion)
    switch ($CudaVersion) {
        "cu121" { return "530+" }
        "cu124" { return "550+" }
        "cu128" { return "560+" }
        default { return "unknown" }
    }
}

function Test-Docker {
    try {
        $null = docker info 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $false
        }
        return $true
    } catch {
        return $false
    }
}

function Test-GpuSupport {
    try {
        # First check if nvidia-smi exists
        $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if (-not $nvidiaSmi) {
            Write-Info "nvidia-smi not found - GPU support unavailable"
            return $false
        }
        
        $null = nvidia-smi 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Info "nvidia-smi failed - GPU may not be available"
            return $false
        }
        
        # Try running a GPU container
        Write-Info "Testing Docker GPU support..."
        $result = docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        
        Write-Warn "Docker GPU test failed - falling back to CPU mode"
        return $false
    } catch {
        Write-Warn "GPU detection error: $_"
        return $false
    }
}

# Convert Windows path to Docker-compatible path
# Docker Desktop on Windows accepts paths like /c/Users/... or //c/Users/...
function Convert-ToDockerPath {
    param([string]$WindowsPath)
    
    if ([string]::IsNullOrEmpty($WindowsPath)) {
        return $WindowsPath
    }
    
    # Get absolute path
    try {
        $absPath = (Resolve-Path $WindowsPath -ErrorAction Stop).Path
    } catch {
        # Path doesn't exist yet, try to make it absolute anyway
        $absPath = [System.IO.Path]::GetFullPath($WindowsPath)
    }
    
    # Convert backslashes to forward slashes
    $unixPath = $absPath -replace '\\', '/'
    
    # Convert drive letter: C:\... -> /c/...
    # Docker Desktop on Windows expects lowercase drive letter
    if ($unixPath -match '^([A-Za-z]):(.*)$') {
        $driveLetter = $Matches[1].ToLower()
        $pathRest = $Matches[2]
        $unixPath = "/$driveLetter$pathRest"
    }
    
    return $unixPath
}

# ------------------------------------------------------------------------------
# Pull or Build Docker Image
# ------------------------------------------------------------------------------
function Pull-Image {
    param(
        [string]$Target,
        [string]$LocalTag
    )
    
    # Map target to Docker Hub tag
    if ($Target -eq "gpu") {
        $HubTag = "${DockerHubImage}:latest"
    } else {
        $HubTag = "${DockerHubImage}:latest-cpu"
    }
    
    Write-Info "Pulling pre-built image from Docker Hub: $HubTag"
    Write-Info "This is much faster than building locally!"
    
    docker pull $HubTag
    if ($LASTEXITCODE -eq 0) {
        # Tag the pulled image with our local tag for consistency
        docker tag $HubTag $LocalTag
        Write-Success "Image pulled and tagged: $LocalTag"
        return $true
    } else {
        Write-Warn "Failed to pull from Docker Hub"
        return $false
    }
}

function Build-Image {
    param(
        [string]$Target,
        [string]$CudaVersion
    )
    
    if ($Target -eq "gpu") {
        $Tag = "${ImageName}:gpu-${CudaVersion}"
    } else {
        $Tag = "${ImageName}:${Target}"
    }
    
    # Try pulling from Docker Hub first (unless force build is set)
    if (-not $ForceBuild -and -not $Build) {
        if (Pull-Image -Target $Target -LocalTag $Tag) {
            return
        }
        Write-Info "Falling back to local build..."
        Write-Host ""
    }
    
    Write-Info "Building Docker image locally: $Tag"
    if ($Target -eq "gpu") {
        Write-Info "CUDA version: $CudaVersion (requires driver $(Get-CudaDriverRequirement $CudaVersion))"
    }
    Write-Info "This may take 10-30 minutes on first build..."
    
    Push-Location $ProjectRoot
    try {
        # Check if .dockerignore exists, if not create it
        $dockerignorePath = Join-Path $ProjectRoot "docker\.dockerignore"
        if (-not (Test-Path $dockerignorePath)) {
            Write-Warn ".dockerignore not found - build context may be large"
        }
        
        $buildArgs = @()
        if ($Target -eq "gpu") {
            $buildArgs += "--build-arg"
            $buildArgs += "CUDA_VERSION=$CudaVersion"
        }
        
        docker build -t $Tag -f docker/Dockerfile --target $Target @buildArgs .
        if ($LASTEXITCODE -ne 0) {
            throw "Docker build failed with exit code $LASTEXITCODE"
        }
        Write-Success "Image built successfully: $Tag"
    } finally {
        Pop-Location
    }
}

# ------------------------------------------------------------------------------
# Create CLI Wrapper Scripts
# ------------------------------------------------------------------------------
function New-Wrapper {
    param(
        [string]$CmdName,
        [string]$RunnerScript,
        [string]$Target,
        [string]$CudaVersion
    )
    
    $WrapperPath = Join-Path $InstallDir "$CmdName.cmd"
    $PsWrapperPath = Join-Path $InstallDir "$CmdName.ps1"
    
    # Determine image tag
    if ($Target -eq "gpu") {
        $ImageTag = "${ImageName}:gpu-${CudaVersion}"
    } else {
        $ImageTag = "${ImageName}:${Target}"
    }
    
    Write-Info "Creating wrapper: $CmdName (image: $ImageTag)"
    
    # GPU flags
    $GpuFlags = if ($Target -eq "gpu") { "--gpus all" } else { "" }
    
    # Convert ModelsDir to Docker path
    $ModelsDockerPath = Convert-ToDockerPath $ModelsDir
    
    # Create CMD wrapper (simple version - delegates to PowerShell for complex path handling)
    $CmdContent = @"
@echo off
REM ==============================================================================
REM $CmdName - UVR Headless Runner CLI Wrapper
REM ==============================================================================
REM For complex paths with spaces or special characters, use the PowerShell wrapper:
REM   powershell -ExecutionPolicy Bypass -File "%~dp0$CmdName.ps1" %*
REM ==============================================================================
setlocal enabledelayedexpansion

set IMAGE=$ImageTag
set MODELS_DIR=%UVR_MODELS_DIR%
if "%MODELS_DIR%"=="" set MODELS_DIR=$ModelsDir

REM Ensure models directory exists
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

REM Build proxy arguments if set (auto-passthrough from host environment)
set PROXY_ARGS=
if defined HTTP_PROXY set PROXY_ARGS=%PROXY_ARGS% -e HTTP_PROXY=%HTTP_PROXY%
if defined HTTPS_PROXY set PROXY_ARGS=%PROXY_ARGS% -e HTTPS_PROXY=%HTTPS_PROXY%
if defined NO_PROXY set PROXY_ARGS=%PROXY_ARGS% -e NO_PROXY=%NO_PROXY%
if defined http_proxy set PROXY_ARGS=%PROXY_ARGS% -e http_proxy=%http_proxy%
if defined https_proxy set PROXY_ARGS=%PROXY_ARGS% -e https_proxy=%https_proxy%
if defined no_proxy set PROXY_ARGS=%PROXY_ARGS% -e no_proxy=%no_proxy%

REM Simple case: run container with basic arguments
REM For complex paths, use PowerShell wrapper
docker run --rm -it $GpuFlags -v "%MODELS_DIR%:/models" -e "UVR_MODELS_DIR=/models" %PROXY_ARGS% %IMAGE% $RunnerScript %*
"@
    
    # Create PowerShell wrapper (handles complex paths properly)
    $PsContent = @"
# ==============================================================================
# $CmdName - UVR Headless Runner CLI Wrapper (PowerShell)
# ==============================================================================
# This wrapper handles Windows paths correctly for Docker
# ==============================================================================

`$ErrorActionPreference = "Continue"

`$Image = "$ImageTag"
`$ModelsDir = if (`$env:UVR_MODELS_DIR) { `$env:UVR_MODELS_DIR } else { "$ModelsDir" }

# Function to convert Windows path to Docker path
function Convert-ToDockerPath {
    param([string]`$WindowsPath)
    
    if ([string]::IsNullOrEmpty(`$WindowsPath)) {
        return `$WindowsPath
    }
    
    # Get absolute path
    try {
        if (Test-Path `$WindowsPath) {
            `$absPath = (Resolve-Path `$WindowsPath).Path
        } else {
            `$absPath = [System.IO.Path]::GetFullPath(`$WindowsPath)
        }
    } catch {
        `$absPath = `$WindowsPath
    }
    
    # Convert backslashes to forward slashes
    `$unixPath = `$absPath -replace '\\\\', '/'
    
    # Convert drive letter: C:\... -> /c/...
    if (`$unixPath -match '^([A-Za-z]):(.*)$') {
        `$driveLetter = `$Matches[1].ToLower()
        `$pathRest = `$Matches[2]
        `$unixPath = "/`$driveLetter`$pathRest"
    }
    
    return `$unixPath
}

# Ensure models directory exists
if (-not (Test-Path `$ModelsDir)) {
    New-Item -ItemType Directory -Path `$ModelsDir -Force | Out-Null
}

# Convert models directory to Docker path
`$ModelsDockerPath = Convert-ToDockerPath `$ModelsDir

# Process arguments for path mounting
`$MountArgs = @()
`$ProcessedArgs = @()
`$MountedDirs = @{}

for (`$i = 0; `$i -lt `$args.Count; `$i++) {
    `$arg = `$args[`$i]
    
    if (`$arg -eq "-i" -or `$arg -eq "--input") {
        `$ProcessedArgs += `$arg
        `$i++
        if (`$i -lt `$args.Count) {
            `$path = `$args[`$i]
            
            # Resolve and convert path
            if (Test-Path `$path) {
                `$absPath = (Resolve-Path `$path).Path
                `$dir = Split-Path `$absPath -Parent
                `$dockerPath = Convert-ToDockerPath `$absPath
                `$dockerDir = Convert-ToDockerPath `$dir
                
                if (-not `$MountedDirs.ContainsKey(`$dir)) {
                    `$MountedDirs[`$dir] = `$true
                    `$MountArgs += "-v"
                    `$MountArgs += "`"`$(`$dir):`$(`$dockerDir):ro`""
                }
                `$ProcessedArgs += `$dockerPath
            } else {
                Write-Warning "Input file not found: `$path"
                `$ProcessedArgs += `$path
            }
        }
    }
    elseif (`$arg -eq "-o" -or `$arg -eq "--output") {
        `$ProcessedArgs += `$arg
        `$i++
        if (`$i -lt `$args.Count) {
            `$path = `$args[`$i]
            
            # Create output directory if it doesn't exist
            if (-not (Test-Path `$path)) {
                try {
                    New-Item -ItemType Directory -Path `$path -Force | Out-Null
                } catch {
                    Write-Warning "Cannot create output directory: `$path"
                }
            }
            
            if (Test-Path `$path) {
                `$absPath = (Resolve-Path `$path).Path
                `$dockerPath = Convert-ToDockerPath `$absPath
                
                if (-not `$MountedDirs.ContainsKey(`$absPath)) {
                    `$MountedDirs[`$absPath] = `$true
                    `$MountArgs += "-v"
                    `$MountArgs += "`"`$(`$absPath):`$(`$dockerPath):rw`""
                }
                `$ProcessedArgs += `$dockerPath
            } else {
                `$ProcessedArgs += `$path
            }
        }
    }
    else {
        `$ProcessedArgs += `$arg
    }
}

# Build docker command
`$DockerArgs = @("run", "--rm")

# Add -it only if running interactively in a terminal
if ([Environment]::UserInteractive -and `$Host.Name -eq 'ConsoleHost') {
    `$DockerArgs += "-it"
}

# Add GPU flags if needed
`$GpuEnabled = "$($Target -eq 'gpu')"
if (`$GpuEnabled -eq "True") {
    `$DockerArgs += "--gpus"
    `$DockerArgs += "all"
}

# Add models volume
`$DockerArgs += "-v"
`$DockerArgs += "`"`$(`$ModelsDir):`$(`$ModelsDockerPath)`""

# Add mounted directories
`$DockerArgs += `$MountArgs

# Add environment variable
`$DockerArgs += "-e"
`$DockerArgs += "UVR_MODELS_DIR=/models"

# HTTP/HTTPS Proxy passthrough
# Automatically passes proxy settings from host to container if set
# SECURITY: Values are passed but not logged (may contain credentials)
if (`$env:HTTP_PROXY) { `$DockerArgs += "-e"; `$DockerArgs += "HTTP_PROXY=`$(`$env:HTTP_PROXY)" }
if (`$env:HTTPS_PROXY) { `$DockerArgs += "-e"; `$DockerArgs += "HTTPS_PROXY=`$(`$env:HTTPS_PROXY)" }
if (`$env:NO_PROXY) { `$DockerArgs += "-e"; `$DockerArgs += "NO_PROXY=`$(`$env:NO_PROXY)" }
if (`$env:http_proxy) { `$DockerArgs += "-e"; `$DockerArgs += "http_proxy=`$(`$env:http_proxy)" }
if (`$env:https_proxy) { `$DockerArgs += "-e"; `$DockerArgs += "https_proxy=`$(`$env:https_proxy)" }
if (`$env:no_proxy) { `$DockerArgs += "-e"; `$DockerArgs += "no_proxy=`$(`$env:no_proxy)" }

# Add image and command
`$DockerArgs += `$Image
`$DockerArgs += "$RunnerScript"
`$DockerArgs += `$ProcessedArgs

# Show command in verbose mode (proxy vars intentionally excluded for security)
if (`$env:UVR_DEBUG) {
    Write-Host "Docker command: docker `$(`$DockerArgs -join ' ')" -ForegroundColor Gray
}

# Execute docker
& docker @DockerArgs
exit `$LASTEXITCODE
"@
    
    # Write wrappers with correct encoding
    [System.IO.File]::WriteAllText($WrapperPath, $CmdContent, [System.Text.Encoding]::ASCII)
    [System.IO.File]::WriteAllText($PsWrapperPath, $PsContent, [System.Text.UTF8Encoding]::new($false))
    
    Write-Success "Installed: $WrapperPath"
}

# ------------------------------------------------------------------------------
# Uninstall
# ------------------------------------------------------------------------------
function Uninstall-Wrappers {
    Write-Info "Uninstalling UVR CLI wrappers..."
    
    $Wrappers = @("uvr", "uvr-mdx", "uvr-demucs", "uvr-vr")
    
    foreach ($wrapper in $Wrappers) {
        $cmdPath = Join-Path $InstallDir "$wrapper.cmd"
        $psPath = Join-Path $InstallDir "$wrapper.ps1"
        
        if (Test-Path $cmdPath) {
            Remove-Item $cmdPath -Force
            Write-Success "Removed: $cmdPath"
        }
        if (Test-Path $psPath) {
            Remove-Item $psPath -Force
            Write-Success "Removed: $psPath"
        }
    }
    
    Write-Info "Uninstallation complete."
    Write-Host ""
    Write-Host "Note: Docker images and model cache were not removed."
    Write-Host ""
    Write-Host "To remove Docker images:"
    Write-Host "  docker rmi ${ImageName}:gpu-cu124 ${ImageName}:cpu"
    Write-Host ""
    Write-Host "To remove model cache:"
    Write-Host "  Remove-Item -Recurse -Force `"$ModelsDir`""
}

# ------------------------------------------------------------------------------
# Main Installation
# ------------------------------------------------------------------------------
function Install-UVR {
    param(
        [string]$Target,
        [string]$CudaVersion
    )
    
    Show-Banner
    
    # Check Docker
    if (-not (Test-Docker)) {
        Write-Error "Docker is not installed or not running."
        Write-Host ""
        Write-Host "Please install Docker Desktop:" -ForegroundColor Yellow
        Write-Host "  https://docs.docker.com/desktop/install/windows-install/"
        Write-Host ""
        Write-Host "After installation, make sure Docker Desktop is running."
        exit 1
    }
    
    Write-Success "Docker is available"
    
    # Auto-detect GPU
    if (-not $Target) {
        Write-Info "Auto-detecting GPU support..."
        if (Test-GpuSupport) {
            $Target = "gpu"
            Write-Success "GPU support detected!"
        } else {
            $Target = "cpu"
            Write-Info "Using CPU mode (no GPU support found)"
        }
    }
    
    Write-Host ""
    Write-Host "Installation mode: $Target" -ForegroundColor Cyan
    if ($Target -eq "gpu") {
        Write-Host "CUDA version: $CudaVersion (requires driver $(Get-CudaDriverRequirement $CudaVersion))" -ForegroundColor Cyan
    }
    Write-Host ""
    
    # Create directories
    Write-Info "Creating directories..."
    try {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
        New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null
        New-Item -ItemType Directory -Path "$ModelsDir\VR_Models" -Force | Out-Null
        New-Item -ItemType Directory -Path "$ModelsDir\MDX_Net_Models" -Force | Out-Null
        New-Item -ItemType Directory -Path "$ModelsDir\Demucs_Models" -Force | Out-Null
        Write-Success "Directories created"
    } catch {
        Write-Error "Failed to create directories: $_"
        exit 1
    }
    
    # Build Docker image
    Build-Image -Target $Target -CudaVersion $CudaVersion
    
    # Create wrappers
    Write-Info "Installing CLI wrappers to $InstallDir..."
    
    New-Wrapper -CmdName "uvr-mdx" -RunnerScript "uvr-mdx" -Target $Target -CudaVersion $CudaVersion
    New-Wrapper -CmdName "uvr-demucs" -RunnerScript "uvr-demucs" -Target $Target -CudaVersion $CudaVersion
    New-Wrapper -CmdName "uvr-vr" -RunnerScript "uvr-vr" -Target $Target -CudaVersion $CudaVersion
    New-Wrapper -CmdName "uvr" -RunnerScript "uvr" -Target $Target -CudaVersion $CudaVersion
    
    # Add to PATH if not already
    $CurrentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($CurrentPath -notlike "*$InstallDir*") {
        Write-Info "Adding $InstallDir to PATH..."
        try {
            [Environment]::SetEnvironmentVariable("Path", "$CurrentPath;$InstallDir", "User")
            $env:Path = "$env:Path;$InstallDir"
            Write-Success "PATH updated"
        } catch {
            Write-Warn "Could not update PATH automatically. Please add manually:"
            Write-Host "  $InstallDir"
        }
    }
    
    # Success message
    Write-Host ""
    Write-Host "=========================================================" -ForegroundColor Green
    Write-Host "            Installation Complete!                       " -ForegroundColor Green
    Write-Host "=========================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now use these commands:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host '  uvr-mdx -m "UVR-MDX-NET Inst HQ 3" -i song.wav -o output/'
    Write-Host '  uvr-demucs -m htdemucs -i song.wav -o output/'
    Write-Host '  uvr-vr -m "UVR-De-Echo-Normal" -i song.wav -o output/'
    Write-Host ""
    Write-Host "  uvr mdx --list          # List MDX models"
    Write-Host "  uvr demucs --list       # List Demucs models"
    Write-Host "  uvr vr --list           # List VR models"
    Write-Host ""
    Write-Host "Models will be cached in: $ModelsDir" -ForegroundColor Cyan
    Write-Host ""
    
    if ($Target -eq "gpu") {
        Write-Host "GPU acceleration is enabled! (CUDA $CudaVersion)" -ForegroundColor Green
        Write-Host ""
        Write-Host "CUDA compatibility:" -ForegroundColor Cyan
        switch ($CudaVersion) {
            "cu121" { Write-Host "  CUDA 12.1 - requires NVIDIA driver 530+" }
            "cu124" { Write-Host "  CUDA 12.4 - requires NVIDIA driver 550+" }
            "cu128" { Write-Host "  CUDA 12.8 - requires NVIDIA driver 560+" }
        }
    } else {
        Write-Host "Running in CPU mode." -ForegroundColor Yellow
        Write-Host "For GPU support, ensure NVIDIA drivers and Docker GPU support are configured."
    }
    
    Write-Host ""
    Write-Host "NOTE: You may need to restart your terminal for PATH changes to take effect." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "For complex paths with spaces, use the PowerShell wrappers:" -ForegroundColor Gray
    Write-Host "  powershell -File `"$InstallDir\uvr-mdx.ps1`" ..." -ForegroundColor Gray
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if ($Help) {
    Show-Banner
    Write-Host "Usage: .\install.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Cpu           Force CPU-only installation"
    Write-Host "  -Gpu           Force GPU installation (uses default CUDA version)"
    Write-Host "  -Cuda VERSION  GPU installation with specific CUDA version"
    Write-Host "                 VERSION: cu121, cu124 (default), cu128"
    Write-Host "  -Build         Force local build instead of pulling from Docker Hub"
    Write-Host "  -Uninstall     Remove installed CLI wrappers"
    Write-Host "  -Help          Show this help message"
    Write-Host ""
    Write-Host "Image Source:"
    Write-Host "  By default, the script pulls pre-built images from Docker Hub (fast!)"
    Write-Host "  Use -Build to force local building (slower, but uses latest code)"
    Write-Host ""
    Write-Host "CUDA Versions:"
    Write-Host "  cu121 - CUDA 12.1, requires NVIDIA driver 530+"
    Write-Host "  cu124 - CUDA 12.4, requires NVIDIA driver 550+ (default)"
    Write-Host "  cu128 - CUDA 12.8, requires NVIDIA driver 560+"
    Write-Host ""
    Write-Host "Environment Variables:"
    Write-Host "  UVR_INSTALL_DIR    Installation directory (default: %LOCALAPPDATA%\UVR)"
    Write-Host "  UVR_MODELS_DIR     Model cache directory (default: %USERPROFILE%\.uvr_models)"
    Write-Host "  UVR_CUDA_VERSION   CUDA version (default: cu124)"
    Write-Host "  UVR_FORCE_BUILD    Set to 1 to force local build"
    Write-Host "  UVR_DEBUG          Set to 1 to show debug output"
    Write-Host ""
    Write-Host "Proxy Support (auto-passthrough if set):"
    Write-Host "  HTTP_PROXY         HTTP proxy URL (e.g., http://proxy:8080)"
    Write-Host "  HTTPS_PROXY        HTTPS proxy URL"
    Write-Host "  NO_PROXY           Comma-separated list of hosts to bypass proxy"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  # Quick install (pulls from Docker Hub)"
    Write-Host "  .\install.ps1"
    Write-Host ""
    Write-Host "  # Install GPU with CUDA 12.1 (for older drivers)"
    Write-Host "  .\install.ps1 -Cuda cu121"
    Write-Host ""
    Write-Host "  # Force local build with GPU"
    Write-Host "  .\install.ps1 -Gpu -Build"
    exit 0
}

if ($Uninstall) {
    Uninstall-Wrappers
    exit 0
}

# Determine target and CUDA version
$Target = $null
$CudaVersion = $DefaultCudaVersion

if ($Cpu) { 
    $Target = "cpu" 
}
if ($Gpu) { 
    $Target = "gpu" 
}
if ($Cuda) {
    $Target = "gpu"
    $CudaVersion = $Cuda
}

Install-UVR -Target $Target -CudaVersion $CudaVersion
