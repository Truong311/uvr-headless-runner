#!/bin/bash
# ==============================================================================
# UVR Headless Runner - Installation Script
# ==============================================================================
# This script installs native-style CLI wrappers so you can run:
#   uvr-mdx -m "UVR-MDX-NET Inst HQ 3" -i song.wav -o output/
#   uvr-demucs -m htdemucs -i song.wav -o output/
#   uvr-vr -m "UVR-De-Echo-Normal" -i song.wav -o output/
#
# Without needing to type `docker run` commands!
#
# Usage:
#   ./docker/install.sh              # Install with auto-detected GPU support (CUDA 12.4)
#   ./docker/install.sh --cpu        # Force CPU-only installation
#   ./docker/install.sh --gpu        # Force GPU installation (CUDA 12.4)
#   ./docker/install.sh --cuda cu121 # GPU with specific CUDA version
#   ./docker/install.sh --cuda cu124 # GPU with CUDA 12.4 (default)
#   ./docker/install.sh --cuda cu128 # GPU with CUDA 12.8
#   ./docker/install.sh --uninstall  # Remove installed wrappers
#
# CUDA Version Options:
#   cu121 - CUDA 12.1, requires NVIDIA driver 530+
#   cu124 - CUDA 12.4, requires NVIDIA driver 550+ (default, recommended)
#   cu128 - CUDA 12.8, requires NVIDIA driver 560+
#
# ==============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_DIR="${UVR_INSTALL_DIR:-/usr/local/bin}"
MODELS_DIR="${UVR_MODELS_DIR:-${HOME}/.uvr_models}"
IMAGE_NAME="uvr-headless"
# CUDA version for GPU builds (cu121, cu124, cu128)
CUDA_VERSION="${UVR_CUDA_VERSION:-cu124}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo -e "${CYAN}"
    echo "========================================================="
    echo "       UVR Headless Runner - Installation Script         "
    echo "========================================================="
    echo -e "${NC}"
}

# Portable in-place file editing (works on both macOS BSD sed and Linux GNU sed)
# Uses perl which is available on all Unix-like systems
sed_inplace() {
    local pattern="$1"
    local file="$2"
    if command -v perl &> /dev/null; then
        perl -pi -e "$pattern" "$file"
    else
        # Fallback to sed with temp file (works everywhere)
        local tmp="${file}.tmp.$$"
        sed "$pattern" "$file" > "$tmp" && mv "$tmp" "$file"
    fi
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        echo "  https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running or you don't have permission."
        echo ""
        echo "Try one of the following:"
        echo "  - Start Docker: sudo systemctl start docker"
        echo "  - Add user to docker group: sudo usermod -aG docker \$USER"
        echo "  - Then log out and back in"
        exit 1
    fi
    
    log_success "Docker is available"
}

detect_gpu() {
    # Check for NVIDIA GPU and Docker GPU support
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected"
        
        # Check if nvidia-container-toolkit is installed
        if docker info 2>/dev/null | grep -q "nvidia"; then
            log_success "Docker GPU support (nvidia runtime) detected"
            echo "gpu"
            return
        fi
        
        # Try running a GPU container
        log_info "Testing Docker GPU support..."
        if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            log_success "Docker GPU test passed"
            echo "gpu"
            return
        else
            log_warn "Docker GPU test failed - nvidia-container-toolkit may not be installed"
            log_warn "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        fi
    else
        log_info "No NVIDIA GPU detected"
    fi
    
    echo "cpu"
}

# Check if we can write to installation directory
check_install_permissions() {
    local dir="$1"
    
    if [ -w "${dir}" ]; then
        echo "direct"
    elif command -v sudo &> /dev/null; then
        # Check if sudo is available and user can use it
        if sudo -n true 2>/dev/null; then
            echo "sudo"
        else
            log_warn "Installation to ${dir} requires sudo access"
            echo "sudo"
        fi
    else
        echo "none"
    fi
}

# ------------------------------------------------------------------------------
# Build Docker Image
# ------------------------------------------------------------------------------
build_image() {
    local target="$1"
    local cuda_ver="$2"
    local tag
    
    if [ "${target}" = "gpu" ]; then
        tag="${IMAGE_NAME}:gpu-${cuda_ver}"
    else
        tag="${IMAGE_NAME}:${target}"
    fi
    
    log_info "Building Docker image: ${tag}"
    if [ "${target}" = "gpu" ]; then
        log_info "CUDA version: ${cuda_ver}"
    fi
    log_info "This may take several minutes on first build..."
    
    cd "${PROJECT_ROOT}"
    
    # Check for .dockerignore
    if [ ! -f "docker/.dockerignore" ]; then
        log_warn ".dockerignore not found - build context may be large"
    fi
    
    local build_args=""
    if [ "${target}" = "gpu" ]; then
        build_args="--build-arg CUDA_VERSION=${cuda_ver}"
    fi
    
    if docker build -t "${tag}" -f docker/Dockerfile --target "${target}" ${build_args} .; then
        log_success "Image built successfully: ${tag}"
    else
        log_error "Failed to build image"
        exit 1
    fi
}

# ------------------------------------------------------------------------------
# Create CLI Wrapper Scripts
# ------------------------------------------------------------------------------
create_wrapper() {
    local cmd_name="$1"
    local runner_script="$2"
    local target="$3"
    local perm_mode="$4"
    local cuda_ver="$5"
    local wrapper_path="${INSTALL_DIR}/${cmd_name}"
    local image_tag
    
    if [ "${target}" = "gpu" ]; then
        image_tag="${IMAGE_NAME}:gpu-${cuda_ver}"
    else
        image_tag="${IMAGE_NAME}:${target}"
    fi
    
    log_info "Creating wrapper: ${cmd_name} (image: ${image_tag})"
    
    # Create wrapper in temp location first
    cat > "/tmp/${cmd_name}" << 'WRAPPER_EOF'
#!/bin/bash
# ==============================================================================
# WRAPPER_CMD_NAME - UVR Headless Runner CLI Wrapper
# ==============================================================================
# Auto-generated by install.sh
# Image: WRAPPER_IMAGE
# ==============================================================================

set -e

# Configuration
IMAGE="WRAPPER_IMAGE"
MODELS_DIR="${UVR_MODELS_DIR:-WRAPPER_MODELS_DIR}"

# Ensure models directory exists
mkdir -p "${MODELS_DIR}"

# Process arguments to handle file paths
DOCKER_ARGS=()
MOUNT_ARGS=()
PROCESSED_ARGS=()

# Track mounted directories to avoid duplicates
declare -A MOUNTED_DIRS 2>/dev/null || {
    # Bash 3 fallback (macOS) - use a simple approach
    MOUNTED_DIRS=""
}

# Function to check if directory is already mounted (Bash 3 compatible)
is_mounted() {
    local dir="$1"
    if [ -n "${MOUNTED_DIRS}" ]; then
        echo "${MOUNTED_DIRS}" | grep -q "|${dir}|" && return 0
    fi
    return 1
}

add_mount() {
    local dir="$1"
    MOUNTED_DIRS="${MOUNTED_DIRS}|${dir}|"
}

process_path() {
    local path="$1"
    local mode="$2"  # "ro" for input, "rw" for output
    
    # Skip if not a path
    if [[ ! "$path" =~ ^[./~] ]] && [[ ! "$path" =~ ^/ ]]; then
        echo "$path"
        return
    fi
    
    # Expand path
    local abs_path
    if [[ "$path" = /* ]]; then
        abs_path="$path"
    elif [[ "$path" = ~* ]]; then
        abs_path="${path/#\~/$HOME}"
    else
        abs_path="$(cd "$(dirname "$path")" 2>/dev/null && pwd)/$(basename "$path")" || abs_path="$path"
    fi
    
    # Get directory
    local dir
    if [ -d "$abs_path" ]; then
        dir="$abs_path"
    else
        dir="$(dirname "$abs_path")"
    fi
    
    # Create directory if output
    if [ "$mode" = "rw" ] && [ ! -d "$dir" ]; then
        mkdir -p "$dir" 2>/dev/null || true
    fi
    
    # Add mount if not already mounted and directory exists
    if ! is_mounted "$dir" && [ -d "$dir" ]; then
        add_mount "$dir"
        MOUNT_ARGS+=("-v" "${dir}:${dir}:${mode}")
    fi
    
    echo "$abs_path"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)
            PROCESSED_ARGS+=("$1")
            shift
            if [[ $# -gt 0 ]]; then
                PROCESSED_ARGS+=("$(process_path "$1" "ro")")
                shift
            fi
            ;;
        -o|--output)
            PROCESSED_ARGS+=("$1")
            shift
            if [[ $# -gt 0 ]]; then
                PROCESSED_ARGS+=("$(process_path "$1" "rw")")
                shift
            fi
            ;;
        *)
            PROCESSED_ARGS+=("$1")
            shift
            ;;
    esac
done

# Build docker run command
# Use -it only if running in a terminal
if [ -t 0 ] && [ -t 1 ]; then
    DOCKER_CMD=(docker run --rm -it)
else
    DOCKER_CMD=(docker run --rm)
fi
WRAPPER_EOF

    # Add GPU flags conditionally
    if [ "${target}" = "gpu" ]; then
        cat >> "/tmp/${cmd_name}" << 'GPU_EOF'
DOCKER_CMD+=(--gpus all)
GPU_EOF
    fi

    cat >> "/tmp/${cmd_name}" << 'END_EOF'
DOCKER_CMD+=(-v "${MODELS_DIR}:/models")
DOCKER_CMD+=("${MOUNT_ARGS[@]}")
DOCKER_CMD+=(-e "UVR_MODELS_DIR=/models")

# HTTP/HTTPS Proxy passthrough
# Automatically passes proxy settings from host to container if set
# SECURITY: Values are passed but not logged (may contain credentials)
[ -n "${HTTP_PROXY:-}" ] && DOCKER_CMD+=(-e "HTTP_PROXY=${HTTP_PROXY}")
[ -n "${HTTPS_PROXY:-}" ] && DOCKER_CMD+=(-e "HTTPS_PROXY=${HTTPS_PROXY}")
[ -n "${NO_PROXY:-}" ] && DOCKER_CMD+=(-e "NO_PROXY=${NO_PROXY}")
[ -n "${http_proxy:-}" ] && DOCKER_CMD+=(-e "http_proxy=${http_proxy}")
[ -n "${https_proxy:-}" ] && DOCKER_CMD+=(-e "https_proxy=${https_proxy}")
[ -n "${no_proxy:-}" ] && DOCKER_CMD+=(-e "no_proxy=${no_proxy}")

DOCKER_CMD+=("${IMAGE}")
DOCKER_CMD+=(WRAPPER_RUNNER)
DOCKER_CMD+=("${PROCESSED_ARGS[@]}")

# Debug mode (proxy vars are intentionally excluded from debug output for security)
if [ -n "${UVR_DEBUG}" ]; then
    echo "Docker command: ${DOCKER_CMD[*]}" >&2
fi

# Run container
exec "${DOCKER_CMD[@]}"
END_EOF

    # Replace placeholders using portable sed function
    sed_inplace "s|WRAPPER_CMD_NAME|${cmd_name}|g" "/tmp/${cmd_name}"
    sed_inplace "s|WRAPPER_IMAGE|${image_tag}|g" "/tmp/${cmd_name}"
    sed_inplace "s|WRAPPER_MODELS_DIR|${MODELS_DIR}|g" "/tmp/${cmd_name}"
    sed_inplace "s|WRAPPER_RUNNER|${runner_script}|g" "/tmp/${cmd_name}"

    # Install wrapper based on permission mode
    case "$perm_mode" in
        direct)
            mv "/tmp/${cmd_name}" "${wrapper_path}"
            chmod +x "${wrapper_path}"
            ;;
        sudo)
            sudo mv "/tmp/${cmd_name}" "${wrapper_path}"
            sudo chmod +x "${wrapper_path}"
            ;;
        *)
            log_error "Cannot install to ${INSTALL_DIR} - no write permission and sudo not available"
            log_info "Try setting UVR_INSTALL_DIR to a writable directory:"
            log_info "  UVR_INSTALL_DIR=\$HOME/.local/bin ./docker/install.sh"
            rm -f "/tmp/${cmd_name}"
            exit 1
            ;;
    esac
    
    log_success "Installed: ${wrapper_path}"
}

# ------------------------------------------------------------------------------
# Uninstall
# ------------------------------------------------------------------------------
uninstall() {
    log_info "Uninstalling UVR CLI wrappers..."
    
    local wrappers=("uvr" "uvr-mdx" "uvr-demucs" "uvr-vr")
    local perm_mode=$(check_install_permissions "${INSTALL_DIR}")
    
    for wrapper in "${wrappers[@]}"; do
        local path="${INSTALL_DIR}/${wrapper}"
        if [ -f "${path}" ]; then
            case "$perm_mode" in
                direct)
                    rm -f "${path}"
                    ;;
                sudo)
                    sudo rm -f "${path}"
                    ;;
                *)
                    log_warn "Cannot remove ${path} - no permission"
                    continue
                    ;;
            esac
            log_success "Removed: ${path}"
        fi
    done
    
    log_info "Uninstallation complete."
    log_info "Note: Docker images and model cache were not removed."
    echo ""
    echo "To remove Docker images:"
    echo "  docker rmi ${IMAGE_NAME}:gpu ${IMAGE_NAME}:cpu"
    echo ""
    echo "To remove model cache:"
    echo "  rm -rf ${MODELS_DIR}"
}

# ------------------------------------------------------------------------------
# Main Installation
# ------------------------------------------------------------------------------
install() {
    local target="$1"
    local cuda_ver="${CUDA_VERSION}"
    
    print_banner
    
    # Check prerequisites
    check_docker
    
    # Validate CUDA version
    case "${cuda_ver}" in
        cu121|cu124|cu128)
            ;;
        *)
            log_error "Invalid CUDA version: ${cuda_ver}"
            log_info "Valid options: cu121, cu124, cu128"
            exit 1
            ;;
    esac
    
    # Check installation permissions
    log_info "Checking installation permissions..."
    local perm_mode=$(check_install_permissions "${INSTALL_DIR}")
    
    if [ "$perm_mode" = "none" ]; then
        log_error "Cannot write to ${INSTALL_DIR} and sudo is not available"
        log_info "Option 1: Run with sudo: sudo ./docker/install.sh"
        log_info "Option 2: Set custom install directory:"
        log_info "  UVR_INSTALL_DIR=\$HOME/.local/bin ./docker/install.sh"
        exit 1
    fi
    
    if [ "$perm_mode" = "sudo" ]; then
        log_info "Will use sudo for installation to ${INSTALL_DIR}"
    fi
    
    # Auto-detect GPU if not specified
    if [ -z "${target}" ]; then
        log_info "Auto-detecting GPU support..."
        target=$(detect_gpu)
    fi
    
    echo ""
    log_info "Installation mode: ${target}"
    if [ "${target}" = "gpu" ]; then
        log_info "CUDA version: ${cuda_ver}"
        case "${cuda_ver}" in
            cu121) log_info "Requires NVIDIA driver 530+" ;;
            cu124) log_info "Requires NVIDIA driver 550+" ;;
            cu128) log_info "Requires NVIDIA driver 560+" ;;
        esac
    fi
    echo ""
    
    # Create models directory
    log_info "Creating models directory: ${MODELS_DIR}"
    mkdir -p "${MODELS_DIR}"
    mkdir -p "${MODELS_DIR}/VR_Models"
    mkdir -p "${MODELS_DIR}/MDX_Net_Models"
    mkdir -p "${MODELS_DIR}/Demucs_Models"
    log_success "Models directory created"
    
    # Build Docker image
    build_image "${target}" "${cuda_ver}"
    
    # Create wrapper scripts
    log_info "Installing CLI wrappers to ${INSTALL_DIR}..."
    
    create_wrapper "uvr-mdx" "uvr-mdx" "${target}" "$perm_mode" "${cuda_ver}"
    create_wrapper "uvr-demucs" "uvr-demucs" "${target}" "$perm_mode" "${cuda_ver}"
    create_wrapper "uvr-vr" "uvr-vr" "${target}" "$perm_mode" "${cuda_ver}"
    create_wrapper "uvr" "uvr" "${target}" "$perm_mode" "${cuda_ver}"
    
    # Print success message
    echo ""
    echo -e "${GREEN}=========================================================${NC}"
    echo -e "${GREEN}            Installation Complete!                       ${NC}"
    echo -e "${GREEN}=========================================================${NC}"
    echo ""
    echo -e "${CYAN}You can now use these commands:${NC}"
    echo ""
    echo "  uvr-mdx -m \"UVR-MDX-NET Inst HQ 3\" -i song.wav -o output/"
    echo "  uvr-demucs -m htdemucs -i song.wav -o output/"
    echo "  uvr-vr -m \"UVR-De-Echo-Normal\" -i song.wav -o output/"
    echo ""
    echo "  uvr mdx --list          # List MDX models"
    echo "  uvr demucs --list       # List Demucs models"
    echo "  uvr vr --list           # List VR models"
    echo "  uvr info                # Show system info"
    echo ""
    echo -e "${CYAN}Models will be cached in: ${MODELS_DIR}${NC}"
    echo ""
    
    if [ "${target}" = "gpu" ]; then
        echo -e "${GREEN}GPU acceleration is enabled! (CUDA ${cuda_ver})${NC}"
        echo ""
        echo "CUDA compatibility:"
        case "${cuda_ver}" in
            cu121) echo "  CUDA 12.1 - requires NVIDIA driver 530+" ;;
            cu124) echo "  CUDA 12.4 - requires NVIDIA driver 550+" ;;
            cu128) echo "  CUDA 12.8 - requires NVIDIA driver 560+" ;;
        esac
    else
        echo -e "${YELLOW}Running in CPU mode.${NC}"
        echo "For GPU support, ensure NVIDIA drivers and nvidia-container-toolkit are installed."
        echo "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
    
    echo ""
    echo "Debug mode: Set UVR_DEBUG=1 to see docker commands"
}

# ------------------------------------------------------------------------------
# Parse Arguments
# ------------------------------------------------------------------------------
TARGET=""
ACTION="install"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            TARGET="cpu"
            shift
            ;;
        --gpu)
            TARGET="gpu"
            shift
            ;;
        --cuda)
            TARGET="gpu"
            shift
            if [[ $# -gt 0 ]] && [[ "$1" != --* ]]; then
                CUDA_VERSION="$1"
                shift
            else
                log_error "--cuda requires a version argument (cu121, cu124, cu128)"
                exit 1
            fi
            ;;
        --uninstall|uninstall)
            ACTION="uninstall"
            shift
            ;;
        --help|-h)
            print_banner
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cpu           Force CPU-only installation"
            echo "  --gpu           Force GPU installation (uses default CUDA version)"
            echo "  --cuda VERSION  GPU installation with specific CUDA version"
            echo "                  VERSION: cu121, cu124 (default), cu128"
            echo "  --uninstall     Remove installed CLI wrappers"
            echo "  --help          Show this help message"
            echo ""
            echo "CUDA Versions:"
            echo "  cu121 - CUDA 12.1, requires NVIDIA driver 530+"
            echo "  cu124 - CUDA 12.4, requires NVIDIA driver 550+ (default)"
            echo "  cu128 - CUDA 12.8, requires NVIDIA driver 560+"
            echo ""
            echo "Environment Variables:"
            echo "  UVR_INSTALL_DIR    Installation directory (default: /usr/local/bin)"
            echo "  UVR_MODELS_DIR     Model cache directory (default: ~/.uvr_models)"
            echo "  UVR_CUDA_VERSION   CUDA version (default: cu124)"
            echo "  UVR_DEBUG          Set to 1 to show debug output"
            echo ""
            echo "Proxy Support (auto-passthrough if set):"
            echo "  HTTP_PROXY         HTTP proxy URL (e.g., http://proxy:8080)"
            echo "  HTTPS_PROXY        HTTPS proxy URL"
            echo "  NO_PROXY           Comma-separated list of hosts to bypass proxy"
            echo ""
            echo "Examples:"
            echo "  # Install to user directory (no sudo needed)"
            echo "  UVR_INSTALL_DIR=\$HOME/.local/bin ./docker/install.sh"
            echo ""
            echo "  # Install GPU with CUDA 12.1 (for older drivers)"
            echo "  ./docker/install.sh --cuda cu121"
            echo ""
            echo "  # Install with custom model directory"
            echo "  UVR_MODELS_DIR=/data/models ./docker/install.sh"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Execute
case "${ACTION}" in
    install)
        install "${TARGET}"
        ;;
    uninstall)
        uninstall
        ;;
esac
