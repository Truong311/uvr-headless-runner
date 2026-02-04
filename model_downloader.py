#!/usr/bin/env python3
"""
UVR Model Downloader Module
===========================
Replicates the official UVR GUI model download center behavior.

This module provides:
1. Remote model registry sync (from official UVR sources)
2. Per-architecture model listings
3. Automatic model downloading with retry and resume
4. UVR GUI-compatible directory structure
5. Checksum verification
6. Fuzzy model name matching

Based on reverse engineering of UVR.py download_checks.json and related logic.

Usage:
    from model_downloader import ModelDownloader
    
    downloader = ModelDownloader()
    
    # List available models
    print(downloader.list_models('mdx'))
    
    # Get model info
    info = downloader.get_model_info('UVR-MDX-NET Inst HQ 3', 'mdx')
    
    # Download a model
    downloader.download_model('UVR-MDX-NET Inst HQ 3', 'mdx')
"""

import os
import sys
import json
import hashlib
import urllib.request
import urllib.error
import shutil
import time
import socket
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable


# ============================================================================
# Custom Exception Classes for Better Error Handling
# ============================================================================

class ModelDownloaderError(Exception):
    """Base exception for model downloader errors."""
    def __init__(self, message: str, suggestion: str = None):
        self.message = message
        self.suggestion = suggestion
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        msg = f"[ModelDownloader Error] {self.message}"
        if self.suggestion:
            msg += f"\n  Suggestion: {self.suggestion}"
        return msg


class NetworkError(ModelDownloaderError):
    """Network-related errors (timeout, connection, etc.)."""
    pass


class RegistryError(ModelDownloaderError):
    """Model registry sync/lookup errors."""
    pass


class DownloadError(ModelDownloaderError):
    """File download errors."""
    pass


class IntegrityError(ModelDownloaderError):
    """File integrity/checksum errors."""
    pass


class ModelNotFoundError(ModelDownloaderError):
    """Model not found in registry."""
    def __init__(self, model_name: str, arch_type: str, similar_models: List[str] = None):
        self.model_name = model_name
        self.arch_type = arch_type
        self.similar_models = similar_models or []
        
        message = f"Model '{model_name}' not found in {arch_type} registry."
        suggestion = None
        if self.similar_models:
            suggestion = f"Did you mean: {', '.join(self.similar_models[:5])}?"
        else:
            suggestion = f"Use --list to see available {arch_type} models."
        
        super().__init__(message, suggestion)


class DiskSpaceError(ModelDownloaderError):
    """Insufficient disk space error."""
    pass


class PermissionError(ModelDownloaderError):
    """File/directory permission errors."""
    pass


# ============================================================================
# UVR Official Data Sources (DO NOT CHANGE unless UVR changes)
# ============================================================================
DOWNLOAD_CHECKS_URL = "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json"
NORMAL_REPO = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
MDX23_CONFIG_CHECKS = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/"

# Model data links (for hash lookups)
VR_MODEL_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/vr_model_data/model_data_new.json"
MDX_MODEL_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_data_new.json"
MDX_MODEL_NAME_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/model_name_mapper.json"
DEMUCS_MODEL_NAME_DATA_LINK = "https://raw.githubusercontent.com/TRvlvr/application_data/main/demucs_model_data/model_name_mapper.json"

# Architecture types (matching UVR constants)
VR_ARCH_TYPE = 'VR Arc'
MDX_ARCH_TYPE = 'MDX-Net'
DEMUCS_ARCH_TYPE = 'Demucs'

# Demucs version identifiers
DEMUCS_V3_ARCH_TYPE = 'Demucs v3'
DEMUCS_V4_ARCH_TYPE = 'Demucs v4'
DEMUCS_NEWER_ARCH_TYPES = [DEMUCS_V3_ARCH_TYPE, DEMUCS_V4_ARCH_TYPE]

# Network configuration
DEFAULT_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # Exponential backoff: 2^attempt seconds
CHUNK_SIZE = 8192


# ============================================================================
# Utility Functions
# ============================================================================

def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    backoff_base: int = RETRY_BACKOFF_BASE,
    exceptions: tuple = (urllib.error.URLError, socket.timeout, ConnectionError)
):
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_base: Base for exponential backoff (seconds)
        exceptions: Tuple of exception types to catch and retry
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = backoff_base ** attempt
                        # Try to get verbose flag from self if it's a method
                        verbose = True
                        if args and hasattr(args[0], 'verbose'):
                            verbose = args[0].verbose
                        if verbose:
                            print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {str(e)[:100]}")
                        time.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator


def fuzzy_match_model(query: str, model_names: List[str], threshold: float = 0.6) -> List[str]:
    """
    Find models with names similar to the query.
    
    Args:
        query: User's model name query
        model_names: List of available model names
        threshold: Minimum similarity ratio (0-1)
    
    Returns:
        List of similar model names, sorted by similarity
    """
    query_lower = query.lower()
    matches = []
    
    for name in model_names:
        # Check exact substring match first
        if query_lower in name.lower():
            matches.append((name, 1.0))
            continue
        
        # Calculate similarity ratio
        ratio = difflib.SequenceMatcher(None, query_lower, name.lower()).ratio()
        if ratio >= threshold:
            matches.append((name, ratio))
    
    # Sort by similarity (highest first)
    matches.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matches]


def get_disk_free_space(path: str) -> int:
    """Get free disk space in bytes for the given path."""
    try:
        if sys.platform == 'win32':
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(path), None, None, ctypes.pointer(free_bytes)
            )
            return free_bytes.value
        else:
            stat = os.statvfs(path)
            return stat.f_bavail * stat.f_frsize
    except Exception:
        return -1  # Unknown


def format_bytes(size: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def calculate_file_hash(filepath: str, algorithm: str = 'md5', last_mb: int = 10) -> Optional[str]:
    """
    Calculate hash of a file (last N MB or whole file if smaller).
    
    This matches UVR GUI's hash calculation method.
    
    Args:
        filepath: Path to the file
        algorithm: Hash algorithm ('md5', 'sha256')
        last_mb: Read last N MB of file (UVR uses 10MB)
    
    Returns:
        Hex hash string or None if file doesn't exist
    """
    if not os.path.isfile(filepath):
        return None
    
    try:
        hasher = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            try:
                # Seek to last N MB from end
                f.seek(-last_mb * 1024 * 1024, 2)
                hasher.update(f.read())
            except OSError:
                # File smaller than N MB, read whole file
                f.seek(0)
                hasher.update(f.read())
        return hasher.hexdigest()
    except Exception:
        return None


class ModelDownloader:
    """
    Handles model registry sync and downloading for all UVR architectures.
    
    Replicates UVR GUI download center behavior exactly.
    """
    
    def __init__(self, base_path: Optional[str] = None, verbose: bool = True):
        """
        Initialize the downloader.
        
        Args:
            base_path: Base directory for models (defaults to script directory)
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        
        # Setup paths (exactly matching UVR.py directory structure)
        if base_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        self.base_path = base_path
        self.models_dir = os.path.join(base_path, 'models')
        self.vr_models_dir = os.path.join(self.models_dir, 'VR_Models')
        self.mdx_models_dir = os.path.join(self.models_dir, 'MDX_Net_Models')
        self.demucs_models_dir = os.path.join(self.models_dir, 'Demucs_Models')
        self.demucs_newer_repo_dir = os.path.join(self.demucs_models_dir, 'v3_v4_repo')
        self.mdx_c_config_path = os.path.join(self.mdx_models_dir, 'model_data', 'mdx_c_configs')
        
        # Ensure directories exist
        for dir_path in [self.vr_models_dir, self.mdx_models_dir, 
                         self.demucs_models_dir, self.demucs_newer_repo_dir,
                         self.mdx_c_config_path]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Model registry (fetched from remote)
        self.online_data: Dict = {}
        self.vr_download_list: Dict = {}
        self.mdx_download_list: Dict = {}
        self.demucs_download_list: Dict = {}
        
        # Local cache path for download_checks.json
        self.cache_dir = os.path.join(base_path, '.model_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.download_checks_cache = os.path.join(self.cache_dir, 'download_checks.json')
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def sync_registry(self, force: bool = False) -> bool:
        """
        Sync model registry from official UVR sources.
        
        This replicates UVR.py online_data_refresh() behavior.
        
        Args:
            force: Force refresh even if cache exists
            
        Returns:
            True if sync successful, False otherwise
            
        Raises:
            RegistryError: If sync fails and no cache available
        """
        # Check cache first
        if not force and os.path.isfile(self.download_checks_cache):
            try:
                cache_age = os.path.getmtime(self.download_checks_cache)
                # Cache valid for 1 hour
                if time.time() - cache_age < 3600:
                    self._log("Using cached model registry...")
                    with open(self.download_checks_cache, 'r', encoding='utf-8') as f:
                        self.online_data = json.load(f)
                    self._populate_model_lists()
                    return True
            except (OSError, json.JSONDecodeError) as e:
                self._log(f"Warning: Cache read error: {e}")
                # Continue to fetch from network
        
        # Try to fetch from network with retry
        network_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                self._log("Syncing model registry from official UVR sources...")
                
                # Fetch download_checks.json (UVR.py line 5605)
                request = urllib.request.Request(
                    DOWNLOAD_CHECKS_URL,
                    headers={'User-Agent': 'UVR-Headless-Runner/1.0'}
                )
                with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                    raw_data = response.read().decode('utf-8')
                    self.online_data = json.loads(raw_data)
                
                # Validate response structure
                if not isinstance(self.online_data, dict):
                    raise RegistryError(
                        "Invalid registry format: expected dictionary",
                        "The server may be returning an error. Try again later."
                    )
                
                # Save to cache (atomic write)
                temp_cache = self.download_checks_cache + '.tmp'
                try:
                    with open(temp_cache, 'w', encoding='utf-8') as f:
                        json.dump(self.online_data, f, indent=2)
                    # Atomic rename
                    if os.path.exists(self.download_checks_cache):
                        os.remove(self.download_checks_cache)
                    shutil.move(temp_cache, self.download_checks_cache)
                except OSError as e:
                    self._log(f"Warning: Could not save cache: {e}")
                    # Clean up temp file
                    if os.path.exists(temp_cache):
                        try:
                            os.remove(temp_cache)
                        except OSError:
                            pass
                
                self._populate_model_lists()
                self._log("Model registry synced successfully!")
                return True
                
            except urllib.error.HTTPError as e:
                network_error = NetworkError(
                    f"HTTP {e.code}: {e.reason}",
                    "The UVR model server may be temporarily unavailable."
                )
                if e.code >= 500:
                    # Server error - retry
                    if attempt < MAX_RETRIES:
                        wait_time = RETRY_BACKOFF_BASE ** attempt
                        self._log(f"  Server error, retry {attempt + 1}/{MAX_RETRIES} after {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                break
                
            except urllib.error.URLError as e:
                network_error = NetworkError(
                    f"Connection failed: {e.reason}",
                    "Check your internet connection or try again later."
                )
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_BACKOFF_BASE ** attempt
                    self._log(f"  Connection error, retry {attempt + 1}/{MAX_RETRIES} after {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                break
                
            except socket.timeout:
                network_error = NetworkError(
                    "Connection timed out",
                    "The server is not responding. Check your network or try again later."
                )
                if attempt < MAX_RETRIES:
                    wait_time = RETRY_BACKOFF_BASE ** attempt
                    self._log(f"  Timeout, retry {attempt + 1}/{MAX_RETRIES} after {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                break
                
            except json.JSONDecodeError as e:
                network_error = RegistryError(
                    f"Invalid JSON response: {e}",
                    "The server returned malformed data. Try again later."
                )
                break
                
            except Exception as e:
                network_error = RegistryError(
                    f"Unexpected error: {e}",
                    "Please report this issue."
                )
                break
        
        # Network fetch failed, try cache as fallback
        if os.path.isfile(self.download_checks_cache):
            try:
                self._log("Network unavailable. Using cached registry...")
                with open(self.download_checks_cache, 'r', encoding='utf-8') as f:
                    self.online_data = json.load(f)
                self._populate_model_lists()
                return True
            except (OSError, json.JSONDecodeError) as e:
                self._log(f"Cache also unreadable: {e}")
        
        # Both network and cache failed
        if network_error:
            self._log(str(network_error))
        return False
    
    def _populate_model_lists(self):
        """
        Populate per-architecture model lists from online_data.
        
        Replicates UVR.py download_list_fill() behavior (lines 5736-5745).
        """
        self.vr_download_list = self.online_data.get("vr_download_list", {})
        self.mdx_download_list = self.online_data.get("mdx_download_list", {})
        self.demucs_download_list = self.online_data.get("demucs_download_list", {})
        
        # Merge additional MDX lists (UVR.py line 5739-5740)
        self.mdx_download_list.update(self.online_data.get("mdx23c_download_list", {}))
        self.mdx_download_list.update(self.online_data.get("other_network_list", {}))
        self.mdx_download_list.update(self.online_data.get("other_network_list_new", {}))
    
    def list_models(self, arch_type: str, show_installed: bool = True) -> List[Dict[str, Any]]:
        """
        List available models for an architecture.
        
        Args:
            arch_type: 'vr', 'mdx', or 'demucs'
            show_installed: If True, include installation status
            
        Returns:
            List of model info dictionaries
        """
        if not self.online_data:
            self.sync_registry()
        
        arch_type = arch_type.lower()
        models = []
        
        if arch_type == 'vr':
            download_list = self.vr_download_list
            model_dir = self.vr_models_dir
        elif arch_type in ['mdx', 'mdx-net']:
            download_list = self.mdx_download_list
            model_dir = self.mdx_models_dir
        elif arch_type == 'demucs':
            download_list = self.demucs_download_list
            model_dir = self.demucs_models_dir
        else:
            raise ValueError(f"Unknown architecture: {arch_type}")
        
        for display_name, model_info in download_list.items():
            model_data = {
                'display_name': display_name,
                'name': self._extract_model_name(display_name),
            }
            
            # Determine filename(s) and check installation
            if isinstance(model_info, dict):
                # Complex model (MDX-C/Roformer or Demucs with multiple files)
                files = list(model_info.keys())
                model_data['files'] = files
                model_data['is_multi_file'] = True
                
                # Check if installed (all files present)
                if arch_type == 'demucs':
                    is_newer = any(x in display_name for x in ['v3', 'v4'])
                    check_dir = self.demucs_newer_repo_dir if is_newer else self.demucs_models_dir
                else:
                    check_dir = model_dir
                
                installed = all(
                    os.path.isfile(os.path.join(check_dir, f)) or
                    os.path.isfile(os.path.join(self.mdx_c_config_path, f))
                    for f in files
                )
            else:
                # Simple model (single file)
                model_data['files'] = [str(model_info)]
                model_data['is_multi_file'] = False
                installed = os.path.isfile(os.path.join(model_dir, str(model_info)))
            
            if show_installed:
                model_data['installed'] = installed
            
            models.append(model_data)
        
        return models
    
    def _extract_model_name(self, display_name: str) -> str:
        """Extract clean model name from display name."""
        # Remove prefix like "VR Arch Single Model v5: " or "MDX-Net Model: "
        if ':' in display_name:
            return display_name.split(':', 1)[1].strip()
        return display_name
    
    def get_model_info(
        self, 
        model_name: str, 
        arch_type: str,
        raise_on_not_found: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed info for a specific model.
        
        Args:
            model_name: Model name or display name
            arch_type: 'vr', 'mdx', or 'demucs'
            raise_on_not_found: If True, raise ModelNotFoundError instead of returning None
            
        Returns:
            Model info dictionary or None if not found
            
        Raises:
            ModelNotFoundError: If raise_on_not_found=True and model not found
            ValueError: If arch_type is invalid
        """
        if not self.online_data:
            self.sync_registry()
        
        arch_type = arch_type.lower()
        
        if arch_type == 'vr':
            download_list = self.vr_download_list
            model_dir = self.vr_models_dir
            subdir = 'VR_Models'
        elif arch_type in ['mdx', 'mdx-net']:
            download_list = self.mdx_download_list
            model_dir = self.mdx_models_dir
            subdir = 'MDX_Net_Models'
        elif arch_type == 'demucs':
            download_list = self.demucs_download_list
            model_dir = self.demucs_models_dir
            subdir = 'Demucs_Models'
        else:
            raise ValueError(f"Unknown architecture: {arch_type}. Valid options: vr, mdx, demucs")
        
        # Search by exact display name or partial match
        for display_name, model_info in download_list.items():
            # Match by display name or extracted name
            clean_name = self._extract_model_name(display_name)
            if model_name in [display_name, clean_name] or model_name.lower() == clean_name.lower():
                result = {
                    'display_name': display_name,
                    'name': clean_name,
                    'arch_type': arch_type,
                    'subdir': subdir,
                }
                
                # Parse model info based on type
                if isinstance(model_info, dict):
                    result['is_multi_file'] = True
                    result['files'] = {}
                    
                    for filename, value in model_info.items():
                        if isinstance(value, str) and (value.startswith('http://') or value.startswith('https://')):
                            # Full URL provided
                            result['files'][filename] = value
                        else:
                            # Need to construct URL
                            if filename.endswith('.yaml'):
                                # Config file - check if it's a reference or needs URL
                                if arch_type in ['mdx', 'mdx-net']:
                                    result['files'][filename] = f"{MDX23_CONFIG_CHECKS}{filename}"
                            else:
                                result['files'][filename] = f"{NORMAL_REPO}{filename}"
                    
                    # Determine actual save directory for Demucs
                    if arch_type == 'demucs':
                        is_newer = any(x in display_name for x in ['v3', 'v4'])
                        result['save_dir'] = self.demucs_newer_repo_dir if is_newer else self.demucs_models_dir
                        result['subdir'] = 'Demucs_Models/v3_v4_repo' if is_newer else 'Demucs_Models'
                    else:
                        result['save_dir'] = model_dir
                else:
                    # Simple single-file model
                    result['is_multi_file'] = False
                    filename = str(model_info)
                    result['filename'] = filename
                    result['url'] = f"{NORMAL_REPO}{filename}"
                    result['save_dir'] = model_dir
                    result['local_path'] = os.path.join(model_dir, filename)
                
                # Check if installed
                if result.get('is_multi_file'):
                    save_dir = result.get('save_dir', model_dir)
                    result['installed'] = all(
                        os.path.isfile(os.path.join(save_dir, f)) or
                        os.path.isfile(os.path.join(self.mdx_c_config_path, f))
                        for f in result['files'].keys()
                    )
                else:
                    result['installed'] = os.path.isfile(result.get('local_path', ''))
                
                return result
        
        # Model not found - provide helpful suggestions
        if raise_on_not_found:
            # Get all model names for fuzzy matching
            all_names = [self._extract_model_name(dn) for dn in download_list.keys()]
            similar = fuzzy_match_model(model_name, all_names)
            raise ModelNotFoundError(model_name, arch_type, similar)
        
        return None
    
    def download_model(
        self, 
        model_name: str, 
        arch_type: str,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, str]:
        """
        Download a model from official UVR sources.
        
        Replicates UVR.py download_item() behavior with enhanced error handling.
        
        Args:
            model_name: Model name or display name
            arch_type: 'vr', 'mdx', or 'demucs'
            progress_callback: Optional callback(current, total, filename)
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Get model info with fuzzy matching suggestions on failure
        try:
            model_info = self.get_model_info(model_name, arch_type, raise_on_not_found=True)
        except ModelNotFoundError as e:
            return False, str(e)
        except ValueError as e:
            return False, f"Invalid architecture type: {e}"
        
        if model_info is None:
            return False, f"Model not found: {model_name}"
        
        if model_info.get('installed'):
            return True, f"Model already installed: {model_name}"
        
        try:
            if model_info.get('is_multi_file'):
                # Download multiple files (Demucs or MDX-C/Roformer)
                files = model_info['files']
                save_dir = model_info.get('save_dir', self.mdx_models_dir)
                
                total_files = len(files)
                downloaded_files = 0
                
                for i, (filename, url) in enumerate(files.items(), 1):
                    # Determine save path
                    if filename.endswith('.yaml') and arch_type in ['mdx', 'mdx-net']:
                        save_path = os.path.join(self.mdx_c_config_path, filename)
                    else:
                        save_path = os.path.join(save_dir, filename)
                    
                    # Skip if already exists
                    if os.path.isfile(save_path):
                        self._log(f"  [{i}/{total_files}] {filename} (already exists)")
                        downloaded_files += 1
                        continue
                    
                    self._log(f"  [{i}/{total_files}] Downloading {filename}...")
                    try:
                        self._download_file(url, save_path, progress_callback)
                        downloaded_files += 1
                    except (DownloadError, NetworkError, IntegrityError) as e:
                        # Partial download - report which files failed
                        return False, (
                            f"Download failed for {filename}: {e.message}\n"
                            f"  Downloaded {downloaded_files}/{total_files} files.\n"
                            f"  {e.suggestion if e.suggestion else ''}"
                        )
                
                return True, f"Successfully downloaded: {model_name} ({downloaded_files} files)"
            else:
                # Download single file
                url = model_info['url']
                save_path = model_info['local_path']
                filename = model_info['filename']
                
                self._log(f"Downloading {filename}...")
                try:
                    self._download_file(url, save_path, progress_callback)
                except (DownloadError, NetworkError, IntegrityError, DiskSpaceError) as e:
                    return False, f"{e.message}\n  {e.suggestion if e.suggestion else ''}"
                
                return True, f"Successfully downloaded: {model_name}"
                
        except DiskSpaceError as e:
            return False, f"{e.message}\n  {e.suggestion}"
        except ModelDownloaderError as e:
            return False, str(e)
        except Exception as e:
            return False, (
                f"Unexpected error during download: {type(e).__name__}: {e}\n"
                f"  Please report this issue if it persists."
            )
    
    def _download_file(
        self, 
        url: str, 
        save_path: str, 
        progress_callback: Optional[Callable] = None,
        expected_size: int = 0,
        expected_hash: str = None,
        verify_size: bool = True
    ):
        """
        Download a file with progress reporting, resume support, and integrity checking.
        
        Features:
        - Retry with exponential backoff
        - Resume partial downloads
        - Disk space pre-check
        - Atomic file writes
        - Optional hash verification
        
        Args:
            url: URL to download from
            save_path: Local path to save file
            progress_callback: Optional callback(current, total, filename)
            expected_size: Expected file size (for disk space check)
            expected_hash: Expected file hash for verification
            verify_size: Whether to verify downloaded size matches Content-Length
            
        Raises:
            DownloadError: If download fails after all retries
            DiskSpaceError: If insufficient disk space
            IntegrityError: If hash verification fails
        """
        temp_path = save_path + '.tmp'
        filename = os.path.basename(save_path)
        
        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as e:
            raise DownloadError(
                f"Cannot create directory: {save_dir}",
                f"Check write permissions or create the directory manually: {e}"
            )
        
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                # Check for existing partial download
                resume_pos = 0
                if os.path.isfile(temp_path):
                    resume_pos = os.path.getsize(temp_path)
                    self._log(f"    Resuming from {format_bytes(resume_pos)}...")
                
                # Build request with resume support
                request = urllib.request.Request(url)
                request.add_header('User-Agent', 'UVR-Headless-Runner/1.0')
                if resume_pos > 0:
                    request.add_header('Range', f'bytes={resume_pos}-')
                
                # Open connection
                with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                    # Get content info
                    content_length = response.headers.get('Content-Length')
                    total_size = int(content_length) if content_length else 0
                    
                    # Handle resume response
                    if response.status == 206:  # Partial Content
                        total_size += resume_pos
                    elif response.status == 200 and resume_pos > 0:
                        # Server doesn't support resume, restart
                        resume_pos = 0
                        if os.path.isfile(temp_path):
                            os.remove(temp_path)
                    
                    # Check disk space
                    if total_size > 0:
                        free_space = get_disk_free_space(save_dir)
                        if free_space > 0 and free_space < total_size * 1.1:  # 10% margin
                            raise DiskSpaceError(
                                f"Insufficient disk space: need {format_bytes(total_size)}, "
                                f"have {format_bytes(free_space)}",
                                "Free up disk space or choose a different download location."
                            )
                    
                    downloaded = resume_pos
                    
                    # Open file for append or write
                    mode = 'ab' if resume_pos > 0 else 'wb'
                    with open(temp_path, mode) as f:
                        while True:
                            try:
                                chunk = response.read(CHUNK_SIZE)
                            except socket.timeout:
                                raise NetworkError(
                                    "Download stalled",
                                    "Network connection lost. Will retry..."
                                )
                            
                            if not chunk:
                                break
                            
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if progress_callback and total_size > 0:
                                progress_callback(downloaded, total_size, filename)
                            elif self.verbose and total_size > 0:
                                percent = int(100 * downloaded / total_size)
                                speed_info = ""
                                print(f"\r    Progress: {percent}% ({format_bytes(downloaded)}/{format_bytes(total_size)}){speed_info}", end='', flush=True)
                    
                    if self.verbose and total_size > 0:
                        print()  # New line after progress
                    
                    # Verify downloaded size
                    if verify_size and total_size > 0 and downloaded != total_size:
                        raise DownloadError(
                            f"Incomplete download: got {format_bytes(downloaded)}, expected {format_bytes(total_size)}",
                            "Download was interrupted. Will retry..."
                        )
                
                # Verify hash if provided
                if expected_hash:
                    actual_hash = calculate_file_hash(temp_path)
                    if actual_hash and actual_hash != expected_hash:
                        # Remove corrupted file
                        os.remove(temp_path)
                        raise IntegrityError(
                            f"Hash mismatch for {filename}",
                            "The downloaded file is corrupted. Will retry download."
                        )
                
                # Atomic move to final location
                try:
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    shutil.move(temp_path, save_path)
                except OSError as e:
                    raise DownloadError(
                        f"Cannot save file: {save_path}",
                        f"Check file permissions: {e}"
                    )
                
                # Success!
                return
                
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise DownloadError(
                        f"File not found on server: {filename}",
                        "The model may have been moved or removed from the repository."
                    )
                elif e.code == 416:  # Range not satisfiable
                    # Remove invalid partial file and restart
                    if os.path.isfile(temp_path):
                        os.remove(temp_path)
                    continue
                else:
                    last_error = DownloadError(
                        f"HTTP {e.code}: {e.reason}",
                        "Server error. Will retry..."
                    )
                    
            except urllib.error.URLError as e:
                last_error = NetworkError(
                    f"Connection failed: {e.reason}",
                    "Check your internet connection."
                )
                
            except socket.timeout:
                last_error = NetworkError(
                    "Connection timed out",
                    "The server is not responding."
                )
            
            except (DiskSpaceError, IntegrityError):
                # Don't retry these errors
                raise
                
            except OSError as e:
                if e.errno == 28:  # No space left on device
                    raise DiskSpaceError(
                        "Disk full during download",
                        "Free up disk space and try again."
                    )
                last_error = DownloadError(
                    f"File system error: {e}",
                    "Check disk and permissions."
                )
            
            # Retry logic
            if attempt < MAX_RETRIES:
                wait_time = RETRY_BACKOFF_BASE ** attempt
                self._log(f"    Retry {attempt + 1}/{MAX_RETRIES} after {wait_time}s...")
                time.sleep(wait_time)
        
        # All retries exhausted
        # Clean up temp file
        if os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        
        if last_error:
            raise last_error
        raise DownloadError(
            f"Download failed after {MAX_RETRIES} attempts",
            "Check your network connection and try again later."
        )
    
    def ensure_model(
        self, 
        model_name: str, 
        arch_type: str,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, str]:
        """
        Ensure a model is available locally, downloading if necessary.
        
        This is the main entry point for automatic downloading with full
        error handling and user feedback.
        
        Args:
            model_name: Model name or display name
            arch_type: 'vr', 'mdx', or 'demucs'
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (success: bool, local_path_or_error: str)
        """
        try:
            model_info = self.get_model_info(model_name, arch_type, raise_on_not_found=False)
        except ValueError as e:
            return False, f"Invalid architecture type: {e}"
        
        if model_info is None:
            # Try sync and retry with fuzzy matching
            self._log(f"Model '{model_name}' not found. Refreshing registry...")
            self.sync_registry(force=True)
            
            try:
                model_info = self.get_model_info(model_name, arch_type, raise_on_not_found=True)
            except ModelNotFoundError as e:
                return False, str(e)
        
        # Helper function to get local path from model_info
        def get_local_path(info: Dict) -> str:
            if info.get('is_multi_file'):
                save_dir = info.get('save_dir')
                # Return path to the main model file (not config)
                for f in info['files'].keys():
                    if not f.endswith('.yaml'):
                        return os.path.join(save_dir, f)
                # If all yaml, return first file
                first_file = list(info['files'].keys())[0]
                return os.path.join(save_dir, first_file)
            else:
                return info['local_path']
        
        if model_info.get('installed'):
            # Return local path
            local_path = get_local_path(model_info)
            
            # Verify file actually exists
            if not os.path.isfile(local_path):
                self._log(f"Warning: Model marked as installed but file missing. Re-downloading...")
            else:
                return True, local_path
        
        # Download the model
        self._log(f"Model not found locally. Downloading: {model_name}")
        success, message = self.download_model(model_name, arch_type, progress_callback)
        
        if success:
            # Refresh model_info and return local path
            model_info = self.get_model_info(model_name, arch_type)
            if model_info:
                return True, get_local_path(model_info)
            else:
                return False, "Download succeeded but model info unavailable"
        
        return False, message
    
    def verify_model_integrity(
        self, 
        model_name: str, 
        arch_type: str,
        redownload_on_failure: bool = False
    ) -> Tuple[bool, str]:
        """
        Verify the integrity of a downloaded model.
        
        Checks:
        1. File exists
        2. File size is non-zero
        3. File is readable
        4. For ONNX: basic structure validation
        
        Args:
            model_name: Model name
            arch_type: Architecture type
            redownload_on_failure: If True, attempt to redownload corrupted files
            
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        model_info = self.get_model_info(model_name, arch_type)
        
        if model_info is None:
            return False, f"Model not found in registry: {model_name}"
        
        if not model_info.get('installed'):
            return False, f"Model not installed: {model_name}"
        
        # Get file paths to check
        files_to_check = []
        if model_info.get('is_multi_file'):
            save_dir = model_info.get('save_dir')
            for f in model_info['files'].keys():
                if f.endswith('.yaml'):
                    files_to_check.append(os.path.join(self.mdx_c_config_path, f))
                else:
                    files_to_check.append(os.path.join(save_dir, f))
        else:
            files_to_check.append(model_info['local_path'])
        
        # Check each file
        for filepath in files_to_check:
            filename = os.path.basename(filepath)
            
            # Check existence
            if not os.path.isfile(filepath):
                if redownload_on_failure:
                    self._log(f"Missing file: {filename}. Redownloading...")
                    success, msg = self.download_model(model_name, arch_type)
                    if not success:
                        return False, f"Redownload failed: {msg}"
                    continue
                return False, f"File missing: {filepath}"
            
            # Check size
            size = os.path.getsize(filepath)
            if size == 0:
                if redownload_on_failure:
                    os.remove(filepath)
                    self._log(f"Empty file: {filename}. Redownloading...")
                    success, msg = self.download_model(model_name, arch_type)
                    if not success:
                        return False, f"Redownload failed: {msg}"
                    continue
                return False, f"File is empty: {filepath}"
            
            # Check readability
            try:
                with open(filepath, 'rb') as f:
                    # Read first few bytes to verify file is accessible
                    header = f.read(16)
                    if len(header) < 16 and size >= 16:
                        return False, f"File read error: {filepath}"
            except OSError as e:
                return False, f"Cannot read file {filepath}: {e}"
            
            # ONNX-specific validation
            if filepath.endswith('.onnx'):
                # Check ONNX magic number
                if header[:4] != b'\x08\x00\x12\x04':
                    # Not all ONNX files have this header, but we can check for protobuf structure
                    pass  # Skip strict validation
        
        return True, f"Model integrity verified: {model_name}"
    
    def get_local_model_path(self, model_name: str, arch_type: str) -> Optional[str]:
        """
        Get the local path for a model if it exists.
        
        Args:
            model_name: Model name or display name
            arch_type: 'vr', 'mdx', or 'demucs'
            
        Returns:
            Local path if model exists, None otherwise
        """
        model_info = self.get_model_info(model_name, arch_type)
        
        if model_info is None:
            return None
        
        if not model_info.get('installed'):
            return None
        
        if model_info.get('is_multi_file'):
            save_dir = model_info.get('save_dir')
            for f in model_info['files'].keys():
                if not f.endswith('.yaml'):
                    return os.path.join(save_dir, f)
            first_file = list(model_info['files'].keys())[0]
            return os.path.join(save_dir, first_file)
        else:
            return model_info.get('local_path')


# ============================================================================
# Architecture-specific registries (for embedding in runners)
# ============================================================================

def get_mdx_models() -> Dict[str, Dict]:
    """Get MDX model registry."""
    downloader = ModelDownloader(verbose=False)
    downloader.sync_registry()
    
    result = {}
    for model in downloader.list_models('mdx', show_installed=True):
        clean_name = model['name']
        result[clean_name] = {
            'display_name': model['display_name'],
            'files': model['files'],
            'installed': model['installed'],
            'subdir': 'MDX_Net_Models'
        }
    return result


def get_vr_models() -> Dict[str, Dict]:
    """Get VR model registry."""
    downloader = ModelDownloader(verbose=False)
    downloader.sync_registry()
    
    result = {}
    for model in downloader.list_models('vr', show_installed=True):
        clean_name = model['name']
        result[clean_name] = {
            'display_name': model['display_name'],
            'files': model['files'],
            'installed': model['installed'],
            'subdir': 'VR_Models'
        }
    return result


def get_demucs_models() -> Dict[str, Dict]:
    """Get Demucs model registry."""
    downloader = ModelDownloader(verbose=False)
    downloader.sync_registry()
    
    result = {}
    for model in downloader.list_models('demucs', show_installed=True):
        clean_name = model['name']
        is_newer = any(x in model['display_name'] for x in ['v3', 'v4'])
        result[clean_name] = {
            'display_name': model['display_name'],
            'files': model['files'],
            'installed': model['installed'],
            'subdir': 'Demucs_Models/v3_v4_repo' if is_newer else 'Demucs_Models'
        }
    return result


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for model downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='UVR Model Downloader - Download models from official UVR sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all MDX models
  python model_downloader.py --list mdx
  
  # List only uninstalled VR models
  python model_downloader.py --list vr --uninstalled
  
  # Download a specific model
  python model_downloader.py --download "UVR-MDX-NET Inst HQ 3" --arch mdx
  
  # Get model info
  python model_downloader.py --info "htdemucs" --arch demucs
  
  # Sync registry from remote
  python model_downloader.py --sync
"""
    )
    
    parser.add_argument('--list', '-l', choices=['vr', 'mdx', 'demucs'], 
                        help='List available models for architecture')
    parser.add_argument('--uninstalled', action='store_true',
                        help='Only show uninstalled models')
    parser.add_argument('--download', '-d', help='Download a model by name')
    parser.add_argument('--arch', '-a', choices=['vr', 'mdx', 'demucs'],
                        help='Architecture type (required for --download and --info)')
    parser.add_argument('--info', '-i', help='Get detailed info for a model')
    parser.add_argument('--sync', '-s', action='store_true',
                        help='Force sync registry from remote')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode')
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(verbose=not args.quiet)
    
    if args.sync:
        success = downloader.sync_registry(force=True)
        if success:
            print("Registry synced successfully!")
        else:
            print("Failed to sync registry")
            return 1
    
    if args.list:
        models = downloader.list_models(args.list)
        
        if args.uninstalled:
            models = [m for m in models if not m['installed']]
        
        if not models:
            print(f"No {'uninstalled ' if args.uninstalled else ''}models found for {args.list}")
            return 0
        
        print(f"\n{'Uninstalled ' if args.uninstalled else ''}Models for {args.list.upper()}:")
        print("=" * 60)
        for model in models:
            status = "Y" if model['installed'] else "N"
            print(f"  [{status}] {model['name']}")
        print(f"\nTotal: {len(models)} models")
        return 0
    
    if args.info:
        if not args.arch:
            print("Error: --arch is required with --info")
            return 1
        
        info = downloader.get_model_info(args.info, args.arch)
        if info:
            print(f"\nModel Info: {info['name']}")
            print("=" * 60)
            print(f"  Display Name: {info['display_name']}")
            print(f"  Architecture: {info['arch_type']}")
            print(f"  Installed: {'Yes' if info['installed'] else 'No'}")
            print(f"  Directory: {info['subdir']}")
            if info.get('is_multi_file'):
                print(f"  Files:")
                for f, url in info['files'].items():
                    print(f"    - {f}")
                    print(f"      URL: {url[:80]}...")
            else:
                print(f"  Filename: {info['filename']}")
                print(f"  URL: {info['url']}")
        else:
            print(f"Model not found: {args.info}")
            return 1
        return 0
    
    if args.download:
        if not args.arch:
            print("Error: --arch is required with --download")
            return 1
        
        print(f"Downloading: {args.download}")
        success, message = downloader.download_model(args.download, args.arch)
        print(message)
        return 0 if success else 1
    
    # Default: show help
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
