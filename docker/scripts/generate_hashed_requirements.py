#!/usr/bin/env python3
"""
Generate pip requirements with SHA256 hashes from poetry.lock.

This script extracts package versions and hashes from poetry.lock to create
pip requirements files with --require-hashes support for supply-chain security.

Security Rationale:
- Hash verification ensures packages haven't been tampered with (MITM, CDN compromise)
- Reproducible builds: same hashes = same bytes = same behavior
- Fails loudly on mismatch: pip refuses to install if hash doesn't match

Usage:
    python generate_hashed_requirements.py [--output FILE] [--packages PKG1,PKG2]
    
Example:
    python generate_hashed_requirements.py --output requirements-hashed.txt
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Packages to EXCLUDE from requirements (handled separately or not needed for Docker)
EXCLUDE_PACKAGES = {
    # PyTorch ecosystem - installed from pytorch wheel index with specific CUDA version
    'torch', 'torchvision', 'torchaudio',
    # NVIDIA CUDA packages - bundled with torch or not needed for CPU
    'nvidia-cublas-cu12', 'nvidia-cuda-cupti-cu12', 'nvidia-cuda-nvrtc-cu12',
    'nvidia-cuda-runtime-cu12', 'nvidia-cudnn-cu12', 'nvidia-cufft-cu12',
    'nvidia-cufile-cu12', 'nvidia-curand-cu12', 'nvidia-cusolver-cu12',
    'nvidia-cusparse-cu12', 'nvidia-cusparselt-cu12', 'nvidia-nccl-cu12',
    'nvidia-nvjitlink-cu12', 'nvidia-nvtx-cu12',
    # triton - CUDA-specific
    'triton',
}

# Target Python version (Docker uses 3.9)
TARGET_PYTHON = "3.9"

# Target platform for Docker (Linux x86_64)
TARGET_PLATFORMS = [
    'manylinux', 'linux', 'any',  # Linux wheels
]

# Packages we explicitly want (from requirements.txt) + all transitive dependencies
# When using --require-hashes, ALL dependencies must have hashes
REQUIRED_PACKAGES = {
    # === Direct dependencies (from requirements.txt) ===
    'numpy', 'scipy', 'librosa', 'soundfile', 'audioread', 'resampy',
    'julius', 'pydub', 'pyrubberband', 'samplerate', 'pytorch-lightning',
    'einops', 'ml-collections', 'omegaconf', 'onnx', 'onnx2pytorch',
    'onnxruntime',  # CPU version - GPU uses onnxruntime-gpu
    'rotary-embedding-torch', 'beartype', 'pyyaml', 'natsort', 'tqdm',
    'requests', 'aiohttp', 'diffq',
    # Note: 'rich' is fetched from PyPI directly (not in poetry.lock)
    
    # === Transitive dependencies (required for --require-hashes) ===
    # aiohttp deps
    'aiohappyeyeballs', 'aiosignal', 'async-timeout', 'attrs', 'frozenlist',
    'multidict', 'propcache', 'yarl',
    # requests deps  
    'certifi', 'charset-normalizer', 'idna', 'urllib3',
    # librosa deps
    'decorator', 'joblib', 'llvmlite', 'numba', 'packaging', 'pooch',
    'platformdirs', 'scikit-learn', 'threadpoolctl', 'appdirs', 'msgpack',
    # soundfile deps
    'cffi', 'pycparser',
    # pytorch-lightning deps
    'fsspec', 'lightning-utilities', 'torchmetrics', 'typing-extensions',
    # omegaconf deps
    'antlr4-python3-runtime',
    # onnx deps
    'protobuf',
    # ml-collections deps
    'absl-py', 'contextlib2',
    # misc
    'coloredlogs', 'humanfriendly', 'flatbuffers', 'sympy', 'mpmath',
    'filelock', 'jinja2', 'markupsafe', 'networkx',
    # setuptools for some packages
    'setuptools',
}

# Packages fetched from PyPI (not in poetry.lock)
# These will be added with hashes fetched at generation time
PYPI_EXTRA_PACKAGES = {
    # rich and its dependencies (for CLI progress display)
    'rich': '13.7.0',
    'markdown-it-py': '3.0.0',
    'mdurl': '0.1.2',
    'pygments': '2.18.0',
}


def parse_poetry_lock(lock_path: Path) -> Dict[str, dict]:
    """
    Parse poetry.lock and extract package info with hashes.
    
    Returns dict: {package_name: {version, files: [(filename, hash), ...]}}
    """
    content = lock_path.read_text(encoding='utf-8')
    packages = {}
    
    # Split into package blocks
    # Each package starts with [[package]]
    blocks = re.split(r'\n\[\[package\]\]\n', content)
    
    for block in blocks[1:]:  # Skip header
        # Extract name
        name_match = re.search(r'^name\s*=\s*"([^"]+)"', block, re.MULTILINE)
        if not name_match:
            continue
        name = name_match.group(1).lower()
        
        # Extract version
        version_match = re.search(r'^version\s*=\s*"([^"]+)"', block, re.MULTILINE)
        if not version_match:
            continue
        version = version_match.group(1)
        
        # Check for python version markers
        markers_match = re.search(r'^markers\s*=\s*"([^"]+)"', block, re.MULTILINE)
        python_marker = None
        if markers_match:
            python_marker = markers_match.group(1)
            # Skip packages not for our Python version
            if 'python_version' in python_marker:
                # Check if it's compatible with 3.9
                if '== "3.10"' in python_marker or '== "3.11"' in python_marker:
                    continue
                if '>= "3.10"' in python_marker or '> "3.9"' in python_marker:
                    continue
        
        # Extract files with hashes
        files_match = re.search(r'^files\s*=\s*\[(.*?)\]', block, re.MULTILINE | re.DOTALL)
        if not files_match:
            continue
            
        files_content = files_match.group(1)
        files = []
        
        # Parse each file entry
        for file_match in re.finditer(
            r'\{file\s*=\s*"([^"]+)",\s*hash\s*=\s*"sha256:([a-f0-9]+)"\}',
            files_content
        ):
            filename = file_match.group(1)
            hash_value = file_match.group(2)
            files.append((filename, hash_value))
        
        if files:
            # If package already exists, check version (prefer 3.9 compatible)
            if name in packages:
                existing_version = packages[name]['version']
                # Keep existing if it's the same or if new has restrictive markers
                if existing_version == version or python_marker:
                    continue
            
            packages[name] = {
                'version': version,
                'files': files,
                'markers': python_marker,
            }
    
    return packages


def filter_wheels_for_platform(
    files: List[Tuple[str, str]], 
    python_version: str = "39"
) -> List[Tuple[str, str]]:
    """
    Filter wheels to only include those compatible with target platform.
    
    Priority:
    1. Source distributions (.tar.gz) - always work
    2. Pure Python wheels (py3-none-any)
    3. Linux x86_64 wheels for cp39
    """
    source_dists = []
    pure_python = []
    linux_wheels = []
    
    for filename, hash_value in files:
        if filename.endswith('.tar.gz') or filename.endswith('.zip'):
            source_dists.append((filename, hash_value))
        elif '-py3-none-any.whl' in filename or '-py2.py3-none-any.whl' in filename:
            pure_python.append((filename, hash_value))
        elif f'-cp{python_version}-' in filename or f'-cp{python_version[0]}{python_version[1]}-' in filename:
            # Check for Linux/manylinux
            if any(p in filename.lower() for p in TARGET_PLATFORMS):
                linux_wheels.append((filename, hash_value))
    
    # Return all applicable wheels (pip will choose the best one)
    # For hash verification, we need to include all possible matches
    result = []
    result.extend(linux_wheels)
    result.extend(pure_python)
    result.extend(source_dists)
    
    return result


def generate_requirements(
    packages: Dict[str, dict],
    required: Optional[set] = None,
    exclude: Optional[set] = None,
    include_deps: bool = True
) -> List[str]:
    """
    Generate requirements lines with hashes.
    
    Format:
        package==version \
            --hash=sha256:abc123 \
            --hash=sha256:def456
    """
    exclude = exclude or EXCLUDE_PACKAGES
    required = required or set(packages.keys())
    
    lines = []
    processed = set()
    
    def add_package(name: str):
        if name in processed or name in exclude:
            return
        if name not in packages:
            return
            
        processed.add(name)
        pkg = packages[name]
        version = pkg['version']
        files = pkg['files']
        
        # Filter to platform-compatible wheels
        compatible_files = filter_wheels_for_platform(files)
        if not compatible_files:
            # Fall back to all files if no specific match
            compatible_files = files
        
        # Generate requirement line with hashes
        # SECURITY: Include all valid hashes - pip verifies against ANY match
        hashes = [f"    --hash=sha256:{h}" for _, h in compatible_files]
        
        if hashes:
            line = f"{name}=={version} \\\n" + " \\\n".join(hashes)
            lines.append(line)
    
    # Process required packages first
    for name in sorted(required):
        add_package(name.lower().replace('_', '-'))
    
    return lines


def fetch_pypi_hashes(package: str, version: str) -> List[Tuple[str, str]]:
    """
    Fetch wheel hashes from PyPI for packages not in poetry.lock.
    """
    import urllib.request
    import json
    
    url = f"https://pypi.org/pypi/{package}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"  Warning: Could not fetch {package}=={version}: {e}", file=sys.stderr)
        return []
    
    hashes = []
    for file_info in data.get('urls', []):
        filename = file_info.get('filename', '')
        digests = file_info.get('digests', {})
        sha256 = digests.get('sha256')
        
        if sha256:
            # Filter to Linux-compatible wheels
            if filename.endswith('.tar.gz'):
                hashes.append((filename, sha256))
            elif '-py3-none-any.whl' in filename or '-py2.py3-none-any.whl' in filename:
                hashes.append((filename, sha256))
            elif '-cp39-' in filename and ('manylinux' in filename or 'linux' in filename):
                hashes.append((filename, sha256))
    
    return hashes


def main():
    parser = argparse.ArgumentParser(
        description='Generate pip requirements with SHA256 hashes from poetry.lock'
    )
    parser.add_argument(
        '--lock', '-l',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'poetry.lock',
        help='Path to poetry.lock file'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output requirements file path'
    )
    parser.add_argument(
        '--packages', '-p',
        type=str,
        help='Comma-separated list of packages to include (default: all required)'
    )
    parser.add_argument(
        '--include-pypi-extra',
        action='store_true',
        default=True,
        help='Include extra packages from PyPI (rich, etc.)'
    )
    
    args = parser.parse_args()
    
    # Parse poetry.lock
    if not args.lock.exists():
        print(f"Error: poetry.lock not found at {args.lock}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Parsing {args.lock}...", file=sys.stderr)
    packages = parse_poetry_lock(args.lock)
    print(f"Found {len(packages)} packages in poetry.lock", file=sys.stderr)
    
    # Fetch extra packages from PyPI
    if args.include_pypi_extra:
        print("Fetching extra packages from PyPI...", file=sys.stderr)
        for pkg_name, pkg_version in PYPI_EXTRA_PACKAGES.items():
            hashes = fetch_pypi_hashes(pkg_name, pkg_version)
            if hashes:
                packages[pkg_name] = {
                    'version': pkg_version,
                    'files': hashes,
                    'markers': None,
                }
                print(f"  {pkg_name}=={pkg_version}: {len(hashes)} hashes", file=sys.stderr)
    
    # Determine which packages to include
    required = REQUIRED_PACKAGES | set(PYPI_EXTRA_PACKAGES.keys())
    if args.packages:
        required = set(p.strip() for p in args.packages.split(','))
    
    # Generate requirements
    lines = generate_requirements(packages, required=required)
    
    # Build output
    header = """\
# =============================================================================
# UVR Headless Runner - Hashed Requirements
# =============================================================================
# AUTO-GENERATED from poetry.lock - DO NOT EDIT MANUALLY
#
# Security: SHA256 hashes ensure package integrity and prevent supply-chain attacks.
# If a package is tampered with (MITM, CDN compromise, malicious maintainer),
# pip will refuse to install it because the hash won't match.
#
# Regenerate with:
#   python docker/scripts/generate_hashed_requirements.py -o docker/requirements-hashed.txt
#
# Usage in Dockerfile:
#   pip install --require-hashes -r requirements-hashed.txt
#
# NOTE: PyTorch packages are NOT included here - they are installed separately
# from pytorch wheel index with specific CUDA versions.
# =============================================================================

"""
    
    output = header + "\n\n".join(lines) + "\n"
    
    if args.output:
        args.output.write_text(output, encoding='utf-8')
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
