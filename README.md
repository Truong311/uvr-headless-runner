# UVR Headless Runner

[中文版](README_CN.md) | **English**

## Design Philosophy

> **This project is a headless automation layer for Ultimate Vocal Remover (UVR).**
> 
> It does **not** reimplement any separation logic.
> 
> It **exactly replicates** UVR GUI behavior, including model loading, parameter fallback, and auto-detection logic.
> 
> **If a model works in the UVR GUI, it is expected to work here — without additional configuration.**

---

## Overview

This project provides two headless runners for audio source separation without the GUI:

- **`mdx_headless_runner.py`** - Supports MDX-Net, MDX-C, Roformer, SCNet models
- **`demucs_headless_runner.py`** - Supports Demucs models (v1/v2/v3/v4)

> ⚠️ **VR Architecture** is intentionally not supported — no current demand from the developer.

---

## Requirements

- **Python**: 3.9.x (3.10+ not fully tested)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **OS**: Windows / Linux / macOS

---

## Installation

### Option 1: Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/uvr-headless-runner.git
cd uvr-headless-runner

# Install dependencies
poetry install

# For GPU support, install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ONNX Runtime GPU (optional, for ONNX models)
pip install onnxruntime-gpu
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/uvr-headless-runner.git
cd uvr-headless-runner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For GPU support, install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install ONNX Runtime GPU (optional)
pip install onnxruntime-gpu
```

### Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### AMD GPU Support (DirectML)

For AMD GPU users, DirectML acceleration is supported:

```bash
# Install torch-directml
pip install torch-directml
```

The code will automatically detect and use DirectML if available. No code changes needed!

> ⚠️ DirectML support is experimental. NVIDIA CUDA is recommended for best performance.

---

## Models

Download models from [UVR Model Database](https://github.com/TRvlvr/model_repo/releases) or use models from an existing UVR installation:

- **MDX models**: `C:\Users\{user}\AppData\Local\Programs\Ultimate Vocal Remover\models\MDX_Net_Models\`
- **Demucs models**: Downloaded automatically on first use

---

## MDX-Net Headless Runner

### Quick Start

```powershell
# Basic usage
poetry run python mdx_headless_runner.py -m "model.ckpt" -i "input.wav" -o "output/" --gpu

# Vocals only
poetry run python mdx_headless_runner.py -m "model.ckpt" -i "input.wav" -o "output/" --gpu --vocals-only
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--model` | `-m` | **Required** | Model file path (.ckpt) |
| `--input` | `-i` | **Required** | Input audio file path |
| `--output` | `-o` | **Required** | Output directory path |
| `--name` | `-n` | Auto | Output file base name |
| `--json` | - | - | Model JSON config file path |
| `--gpu` | - | Auto-detect | Use GPU (NVIDIA CUDA) |
| `--cpu` | - | - | Force CPU |
| `--directml` | - | - | Use DirectML (AMD GPU) |
| `--device` | `-d` | `0` | GPU device ID |
| `--segment-size` | - | `256` | Segment size |
| `--overlap` | - | `0.25` | MDX overlap (0.25-0.99) |
| `--overlap-mdxc` | - | `2` | MDX-C/Roformer overlap (2-50) |
| `--batch-size` | - | `1` | Batch size |
| `--wav-type` | - | `PCM_24` | Output bit depth: PCM_16, PCM_24, PCM_32, FLOAT (32-bit float), DOUBLE (64-bit float) |
| `--quiet` | `-q` | - | Quiet mode |

#### Output Control

| Argument | Description |
|----------|-------------|
| `--primary-only` | Save primary stem only |
| `--secondary-only` | Save secondary stem only |
| `--vocals-only` | Save vocals only (= --primary-only) |
| `--instrumental-only` | Save instrumental only (= --secondary-only) |
| `--dry-only` | Save Dry only (= --primary-only) |
| `--no-dry-only` | Save No Dry only (= --secondary-only) |
| `--stem` | Select stem (MDX-C only: all/vocals/drums/bass/other) |

### Examples

```powershell
# Roformer model with custom overlap
poetry run python mdx_headless_runner.py \
    -m "MDX23C-8KFFT-InstVoc_HQ.ckpt" \
    -i "song.flac" \
    -o "output/" \
    --gpu \
    --overlap-mdxc 8

# Instrumental only
poetry run python mdx_headless_runner.py \
    -m "model.ckpt" \
    -i "song.flac" \
    -o "output/" \
    --gpu \
    --instrumental-only
```

---

## Demucs Headless Runner

### Quick Start

```powershell
# Output all stems (equivalent to GUI "All Stems")
poetry run python demucs_headless_runner.py --model htdemucs --input "song.flac" --output "output/" --gpu

# Vocals only (equivalent to GUI "Vocals" + "Primary Stem Only")
poetry run python demucs_headless_runner.py --model htdemucs --input "song.flac" --output "output/" --gpu --stem Vocals --primary-only
```

### Supported Models

| Model | Version | Stems | Description |
|-------|---------|-------|-------------|
| `htdemucs` | v4 | 4 | Drums, Bass, Other, Vocals |
| `htdemucs_ft` | v4 | 4 | Fine-tuned, better quality |
| `htdemucs_6s` | v4 | 6 | Drums, Bass, Other, Vocals, Guitar, Piano |
| `hdemucs_mmi` | v4 | 4 | Standard version |
| `mdx_extra_q` | v3 | 4 | v3 version |

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--model` | `-m` | **Required** | Model name or path |
| `--input` | `-i` | **Required** | Input audio file path |
| `--output` | `-o` | **Required** | Output directory path |
| `--name` | `-n` | Auto | Output file base name |
| `--gpu` | - | Auto-detect | Use GPU (NVIDIA CUDA) |
| `--cpu` | - | - | Force CPU |
| `--directml` | - | - | Use DirectML (AMD GPU) |
| `--device` | `-d` | `0` | GPU device ID |
| `--segment` | - | `Default` | Segment size (Default/1-100+, supports custom values) |
| `--shifts` | - | `2` | Number of time shifts |
| `--overlap` | - | `0.25` | Overlap ratio |
| `--stem` | - | - | Select stem (Vocals/Other/Bass/Drums/Guitar/Piano) |
| `--wav-type` | - | `PCM_24` | Output bit depth: PCM_16, PCM_24, PCM_32, FLOAT (32-bit float), DOUBLE (64-bit float) |
| `--primary-only` | - | - | Output primary stem only |
| `--secondary-only` | - | - | Output secondary stem only |
| `--quiet` | `-q` | - | Quiet mode |

### Examples

```powershell
# Output all 4 stems
poetry run python demucs_headless_runner.py \
    --model htdemucs \
    --input "song.flac" \
    --output "output/" \
    --gpu

# Vocals only (1 file)
poetry run python demucs_headless_runner.py \
    --model htdemucs \
    --input "song.flac" \
    --output "output/" \
    --gpu \
    --stem Vocals \
    --primary-only

# 6-stem model
poetry run python demucs_headless_runner.py \
    --model htdemucs_6s \
    --input "song.flac" \
    --output "output/" \
    --gpu
```

### Stem Selection Guide

| GUI Action | CLI Equivalent |
|------------|----------------|
| All Stems | Don't specify `--stem` |
| Vocals | `--stem Vocals` |
| Vocals + Primary Only | `--stem Vocals --primary-only` |
| Vocals + Secondary Only | `--stem Vocals --secondary-only` |

---

## Output Files

### MDX-Net Models

```
output/
├── {filename}_(Vocals).wav      # Primary stem
└── {filename}_(Instrumental).wav # Secondary stem
```

### Demucs 4-stem Models

```
output/
├── {filename}_(Drums).wav
├── {filename}_(Bass).wav
├── {filename}_(Other).wav
└── {filename}_(Vocals).wav
```

### Demucs 6-stem Models

```
output/
├── {filename}_(Drums).wav
├── {filename}_(Bass).wav
├── {filename}_(Other).wav
├── {filename}_(Vocals).wav
├── {filename}_(Guitar).wav
└── {filename}_(Piano).wav
```

---

## Python API

### MDX Runner

```python
from mdx_headless_runner import run_mdx_headless

output_files = run_mdx_headless(
    model_path='model.ckpt',
    audio_file='input.wav',
    export_path='output',
    use_gpu=True,
    mdx_segment_size=256,
    overlap_mdx=0.25,
    overlap_mdx23=2,
    verbose=True
)
```

### Demucs Runner

```python
from demucs_headless_runner import run_demucs_headless

output_files = run_demucs_headless(
    model_path='htdemucs',
    audio_file='input.wav',
    export_path='output',
    use_gpu=True,
    demucs_stems='Vocals',
    primary_only=True,
    verbose=True
)
```

---

## Troubleshooting

### GPU Not Working

```powershell
# Check CUDA availability
poetry run python -c "import torch; print(torch.cuda.is_available())"

# Check GPU device
poetry run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Model Not Found

- MDX models default location: `C:\Users\{user}\AppData\Local\Programs\Ultimate Vocal Remover\models\MDX_Net_Models\`
- Demucs models default location: `...\Demucs_Models\v3_v4_repo\`

### Poor Output Quality

- Ensure using correct model config (`--json` argument)
- Try increasing `--overlap` or `--overlap-mdxc`
- For Demucs, try increasing `--segment`

---

## Acknowledgments

This project is based on [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) by:
- [Anjok07](https://github.com/anjok07) - Core Developer
- [aufr33](https://github.com/aufr33) - Core Developer

Special thanks to:
- [ZFTurbo](https://github.com/ZFTurbo) - MDX23C models & SCNet implementation
- [Adefossez & Facebook Research](https://github.com/facebookresearch/demucs) - Demucs AI code
- [Kuielab & Woosung Choi](https://github.com/kuielab) - MDX-Net AI code

### New Features in This Project

- `mdx_headless_runner.py` - MDX/Roformer/SCNet headless runner
- `demucs_headless_runner.py` - Demucs headless runner
- Command-line interface support
- GPU optimization

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2022 Anjok07 (Ultimate Vocal Remover)
Copyright (c) 2026 UVR Headless Runner Contributors
```

See [LICENSE](LICENSE) for full details.

### Third-Party Licenses

| Project | License | Link |
|---------|---------|------|
| Ultimate Vocal Remover GUI | MIT | [GitHub](https://github.com/Anjok07/ultimatevocalremovergui) |
| Demucs | MIT | [GitHub](https://github.com/facebookresearch/demucs) |
| MDX-Net | MIT | [GitHub](https://github.com/kuielab/mdx-net) |
