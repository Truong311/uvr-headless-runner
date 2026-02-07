#!/bin/bash
# ==============================================================================
# UVR Headless Runner - Container Entrypoint
# ==============================================================================
# Handles:
# - GPU auto-detection and fallback
# - Model directory initialization
# - HTTP/HTTPS proxy passthrough
# - CLI routing
# ==============================================================================

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ------------------------------------------------------------------------------
# Proxy Environment Variable Normalization
# ------------------------------------------------------------------------------
# Handles both lowercase and uppercase variants automatically.
# This ensures Python/pip/curl/wget all see the proxy settings correctly.
# SECURITY: Proxy URLs may contain credentials - never log them directly!
# ------------------------------------------------------------------------------
normalize_proxy_vars() {
    # HTTP_PROXY: Prefer uppercase, fallback to lowercase
    if [ -n "${HTTP_PROXY:-}" ]; then
        export http_proxy="${HTTP_PROXY}"
    elif [ -n "${http_proxy:-}" ]; then
        export HTTP_PROXY="${http_proxy}"
    fi
    
    # HTTPS_PROXY: Prefer uppercase, fallback to lowercase
    if [ -n "${HTTPS_PROXY:-}" ]; then
        export https_proxy="${HTTPS_PROXY}"
    elif [ -n "${https_proxy:-}" ]; then
        export HTTPS_PROXY="${https_proxy}"
    fi
    
    # NO_PROXY: Prefer uppercase, fallback to lowercase
    if [ -n "${NO_PROXY:-}" ]; then
        export no_proxy="${NO_PROXY}"
    elif [ -n "${no_proxy:-}" ]; then
        export NO_PROXY="${no_proxy}"
    fi
    
    # ALL_PROXY (used by some tools): Prefer uppercase, fallback to lowercase
    if [ -n "${ALL_PROXY:-}" ]; then
        export all_proxy="${ALL_PROXY}"
    elif [ -n "${all_proxy:-}" ]; then
        export ALL_PROXY="${all_proxy}"
    fi
}

# Check if proxy is configured (for status reporting)
# SECURITY: Only reports presence, never the actual URL (may contain credentials)
is_proxy_configured() {
    [ -n "${HTTP_PROXY:-}" ] || [ -n "${http_proxy:-}" ] || \
    [ -n "${HTTPS_PROXY:-}" ] || [ -n "${https_proxy:-}" ] || \
    [ -n "${ALL_PROXY:-}" ] || [ -n "${all_proxy:-}" ]
}

# Get safe proxy status string (no credentials)
get_proxy_status() {
    if is_proxy_configured; then
        echo "configured"
    else
        echo "not configured"
    fi
}

# ------------------------------------------------------------------------------
# Signal Handling for Graceful Shutdown
# ------------------------------------------------------------------------------
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo -e "${YELLOW}[INFO]${NC} Received shutdown signal, cleaning up..."
    
    # Clean up numba cache temp files
    if [ -d "${NUMBA_CACHE_DIR:-/tmp/numba_cache}" ]; then
        find "${NUMBA_CACHE_DIR:-/tmp/numba_cache}" -type f -name "*.tmp" -delete 2>/dev/null || true
    fi
    
    # Clean up any partial model downloads
    find "${UVR_MODELS_DIR:-/models}" -name "*.tmp" -mmin +5 -delete 2>/dev/null || true
    
    exit "$exit_code"
}

# Trap signals for graceful shutdown
trap 'cleanup_and_exit 130' INT   # Ctrl+C
trap 'cleanup_and_exit 143' TERM  # docker stop

# ------------------------------------------------------------------------------
# Logging Functions (with timestamps for debugging)
# ------------------------------------------------------------------------------
get_timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

log_info() {
    echo -e "${BLUE}[$(get_timestamp)] [INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(get_timestamp)] [OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(get_timestamp)] [WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(get_timestamp)] [ERROR]${NC} $1"
}

# ------------------------------------------------------------------------------
# GPU Detection
# ------------------------------------------------------------------------------
detect_gpu() {
    # If explicitly set to cpu, respect that
    if [ "${UVR_DEVICE}" = "cpu" ]; then
        echo "cpu"
        return
    fi
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            # Verify PyTorch can see CUDA
            if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
                echo "cuda"
                return
            else
                log_warn "nvidia-smi works but PyTorch cannot access CUDA"
            fi
        fi
    fi
    
    echo "cpu"
}

# ------------------------------------------------------------------------------
# Cleanup Old Cache Files
# ------------------------------------------------------------------------------
cleanup_old_cache() {
    # Clean up numba cache files older than 7 days
    if [ -d "${NUMBA_CACHE_DIR:-/tmp/numba_cache}" ]; then
        find "${NUMBA_CACHE_DIR:-/tmp/numba_cache}" -type f -mtime +7 -delete 2>/dev/null || true
    fi
    
    # Clean up orphaned .tmp download files (older than 1 day)
    if [ -d "${UVR_MODELS_DIR:-/models}" ]; then
        find "${UVR_MODELS_DIR:-/models}" -name "*.tmp" -mtime +1 -delete 2>/dev/null || true
    fi
}

# ------------------------------------------------------------------------------
# Model Directory Setup
# ------------------------------------------------------------------------------
setup_model_dirs() {
    local models_dir="${UVR_MODELS_DIR:-/models}"
    
    # Clean up old cache on startup
    cleanup_old_cache
    
    # Ensure /uvr_models mount point exists (for custom local model auto-mount)
    # The CLI wrappers may mount user-specified model paths here.
    if [ ! -d "/uvr_models" ]; then
        mkdir -p /uvr_models 2>/dev/null || true
    fi
    
    # Check if volume is mounted and writable
    if [ ! -d "${models_dir}" ]; then
        log_error "Models directory does not exist: ${models_dir}"
        log_error "Please mount a volume: docker run -v /path/to/models:/models ..."
        return 1
    fi
    
    # Test write permission
    if ! touch "${models_dir}/.write_test" 2>/dev/null; then
        log_warn "Models directory is read-only: ${models_dir}"
        log_warn "Model downloads will fail. Mount with :rw for write access."
    else
        rm -f "${models_dir}/.write_test"
    fi
    
    # Create model subdirectories if they don't exist (in mounted volume)
    local subdirs=(
        "VR_Models/model_data"
        "MDX_Net_Models/model_data/mdx_c_configs"
        "Demucs_Models/v3_v4_repo"
        "Demucs_Models/model_data"
        "Apollo_Models/model_configs"
    )
    
    local mkdir_failed=0
    for subdir in "${subdirs[@]}"; do
        if [ ! -d "${models_dir}/${subdir}" ]; then
            if ! mkdir -p "${models_dir}/${subdir}" 2>/dev/null; then
                if [ $mkdir_failed -eq 0 ]; then
                    log_warn "Cannot create model subdirectory: ${models_dir}/${subdir}"
                    log_warn "This may be due to permission issues or read-only volume."
                    log_warn "Some model types may not work correctly."
                    mkdir_failed=1
                fi
            fi
        fi
    done
    
    # Copy model metadata from app to models volume (first run only)
    # Source files are in /app/models_data (copied during build, not symlinked)
    if [ -d /app/models_data ]; then
        # VR model data
        if [ -d /app/models_data/VR_Models/model_data ] && \
           [ ! -f "${models_dir}/VR_Models/model_data/model_data.json" ]; then
            cp -r /app/models_data/VR_Models/model_data/* "${models_dir}/VR_Models/model_data/" 2>/dev/null || true
            log_info "Initialized VR model metadata"
        fi
        
        # MDX model data
        if [ -d /app/models_data/MDX_Net_Models/model_data ] && \
           [ ! -f "${models_dir}/MDX_Net_Models/model_data/model_data.json" ]; then
            cp -r /app/models_data/MDX_Net_Models/model_data/* "${models_dir}/MDX_Net_Models/model_data/" 2>/dev/null || true
            log_info "Initialized MDX model metadata"
        fi
        
        # Demucs model data
        if [ -d /app/models_data/Demucs_Models/model_data ] && \
           [ ! -f "${models_dir}/Demucs_Models/model_data/model_name_mapper.json" ]; then
            cp -r /app/models_data/Demucs_Models/model_data/* "${models_dir}/Demucs_Models/model_data/" 2>/dev/null || true
            log_info "Initialized Demucs model metadata"
        fi
        
        # Apollo model data
        if [ -d /app/models_data/Apollo_Models ]; then
            cp -r /app/models_data/Apollo_Models/* "${models_dir}/Apollo_Models/" 2>/dev/null || true
        fi
    fi
    
    return 0
}

# ------------------------------------------------------------------------------
# Print Startup Info
# ------------------------------------------------------------------------------
print_startup_info() {
    local device="$1"
    
    echo -e "${CYAN}"
    echo "========================================================="
    echo "       UVR Headless Runner - Container Started           "
    echo "========================================================="
    echo -e "${NC}"
    
    echo -e "Device: ${GREEN}${device}${NC}"
    
    if [ "${device}" = "cuda" ]; then
        local gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
        local cuda_ver=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "Unknown")
        local vram=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')" 2>/dev/null || echo "Unknown")
        echo -e "GPU: ${GREEN}${gpu_name}${NC}"
        echo -e "CUDA: ${GREEN}${cuda_ver}${NC}"
        echo -e "VRAM: ${GREEN}${vram}${NC}"
    fi
    
    echo -e "Models: ${GREEN}${UVR_MODELS_DIR:-/models}${NC}"
    
    # Show custom models mount status
    if [ -d "/uvr_models" ] && [ "$(ls -A /uvr_models 2>/dev/null)" ]; then
        echo -e "Custom Models: ${GREEN}/uvr_models (mounted)${NC}"
    else
        echo -e "Custom Models: ${YELLOW}/uvr_models (not mounted)${NC}"
    fi
    
    # Show proxy status (without revealing credentials)
    local proxy_status=$(get_proxy_status)
    if [ "${proxy_status}" = "configured" ]; then
        echo -e "Proxy: ${GREEN}${proxy_status}${NC}"
    fi
    
    echo ""
}

# ------------------------------------------------------------------------------
# Print Help
# ------------------------------------------------------------------------------
print_help() {
    print_startup_info "$1"
    
    echo "Usage:"
    echo "  uvr-mdx    - MDX-Net/Roformer separation"
    echo "  uvr-demucs - Demucs separation"
    echo "  uvr-vr     - VR Architecture separation"
    echo "  uvr        - Unified CLI"
    echo ""
    echo "Examples:"
    echo "  uvr-mdx -m \"UVR-MDX-NET Inst HQ 3\" -i /input/song.wav -o /output/"
    echo "  uvr-demucs -m htdemucs -i /input/song.wav -o /output/"
    echo "  uvr-vr -m \"UVR-De-Echo-Normal\" -i /input/song.wav -o /output/"
    echo ""
    echo "Model Management:"
    echo "  uvr-mdx --list          # List MDX models"
    echo "  uvr-demucs --list       # List Demucs models"
    echo "  uvr-vr --list           # List VR models"
    echo "  uvr info                # Show system info"
    echo ""
    echo "For more help:"
    echo "  uvr-mdx --help"
    echo "  uvr-demucs --help"
    echo "  uvr-vr --help"
}

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
main() {
    # Normalize proxy environment variables first (before any network access)
    normalize_proxy_vars
    
    # Setup model directories
    setup_model_dirs || {
        log_error "Failed to setup model directories"
        exit 1
    }
    
    # Detect device
    local detected_device=$(detect_gpu)
    
    # Use detected device if UVR_DEVICE not set, or if UVR_DEVICE is set to cuda but GPU not available
    if [ -z "${UVR_DEVICE}" ]; then
        export UVR_DEVICE="${detected_device}"
    elif [ "${UVR_DEVICE}" = "cuda" ] && [ "${detected_device}" = "cpu" ]; then
        log_warn "UVR_DEVICE=cuda but GPU not available, falling back to CPU"
        export UVR_DEVICE="cpu"
    fi
    
    # Ensure UVR_CUSTOM_MODELS_DIR is set for the Python runners
    export UVR_CUSTOM_MODELS_DIR="${UVR_CUSTOM_MODELS_DIR:-/uvr_models}"
    
    # Log if custom models are mounted
    if [ -d "/uvr_models" ] && [ "$(ls -A /uvr_models 2>/dev/null)" ]; then
        log_info "Custom models mount detected at /uvr_models"
    fi
    
    # Handle no arguments - show help
    if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        print_help "${UVR_DEVICE}"
        exit 0
    fi
    
    # Route to appropriate command
    case "$1" in
        uvr-mdx|mdx)
            shift
            exec /usr/local/bin/uvr-mdx "$@"
            ;;
        uvr-demucs|demucs)
            shift
            exec /usr/local/bin/uvr-demucs "$@"
            ;;
        uvr-vr|vr)
            shift
            exec /usr/local/bin/uvr-vr "$@"
            ;;
        uvr)
            shift
            exec /usr/local/bin/uvr "$@"
            ;;
        python|python3)
            # Allow direct Python execution
            exec "$@"
            ;;
        bash|sh)
            # Allow shell access
            exec "$@"
            ;;
        info|--info)
            # Show system info
            print_startup_info "${UVR_DEVICE}"
            python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU device:', torch.cuda.get_device_name(0))
    print('VRAM:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
            exit 0
            ;;
        *)
            # Default: try to execute as command
            exec "$@"
            ;;
    esac
}

main "$@"
