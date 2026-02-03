# UVR Headless Runner 使用说明

**中文** | [English](README.md)

## 设计理念

> **本项目是 Ultimate Vocal Remover (UVR) 的无头自动化层。**
> 
> 它**不会**重新实现任何分离逻辑。
> 
> 它**完全复制** UVR GUI 的行为，包括模型加载、参数回退和自动检测逻辑。
> 
> **如果一个模型在 UVR GUI 中能正常工作，那么在这里也应该能正常工作——无需额外配置。**

---

## 概述

本项目提供两个无头（headless）运行器，允许您在没有 GUI 的情况下进行音频分离：

- **`mdx_headless_runner.py`** - 支持 MDX-Net、MDX-C、Roformer、SCNet 模型
- **`demucs_headless_runner.py`** - 支持 Demucs 模型（v1/v2/v3/v4）

> ⚠️ **VR Architecture** 架构对于开发者来说暂时没有需求，暂不做支持考虑。

---

## 系统要求

- **Python**: 3.9.x（3.10+ 未完全测试）
- **GPU**: 支持 CUDA 的 NVIDIA 显卡（可选但推荐）
- **系统**: Windows / Linux / macOS

---

## 安装

### 方式一：使用 Poetry（推荐）

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/uvr-headless-runner.git
cd uvr-headless-runner

# 安装依赖
poetry install

# GPU 支持：安装带 CUDA 的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装 ONNX Runtime GPU（可选，用于 ONNX 模型）
pip install onnxruntime-gpu
```

### 方式二：使用 pip

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/uvr-headless-runner.git
cd uvr-headless-runner

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或: venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# GPU 支持：安装带 CUDA 的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装 ONNX Runtime GPU（可选）
pip install onnxruntime-gpu
```

### 验证安装

```bash
# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 检查 GPU
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### AMD 显卡支持（DirectML）

AMD 显卡用户可以使用 DirectML 加速：

```bash
# 安装 torch-directml
pip install torch-directml
```

代码会自动检测并使用 DirectML，无需修改代码！

> ⚠️ DirectML 支持为实验性功能。推荐使用 NVIDIA CUDA 以获得最佳性能。

---

## 模型

从 [UVR 模型库](https://github.com/TRvlvr/model_repo/releases) 下载模型，或使用已安装 UVR 中的模型：

- **MDX 模型**: `C:\Users\{user}\AppData\Local\Programs\Ultimate Vocal Remover\models\MDX_Net_Models\`
- **Demucs 模型**: 首次使用时自动下载

---

## MDX-Net Headless Runner

### 快速开始

```powershell
# 基本用法
poetry run python mdx_headless_runner.py -m "model.ckpt" -i "input.wav" -o "output/" --gpu

# 只输出人声
poetry run python mdx_headless_runner.py -m "model.ckpt" -i "input.wav" -o "output/" --gpu --vocals-only
```

### 命令行参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | `-m` | **必需** | 模型文件路径 (.ckpt) |
| `--input` | `-i` | **必需** | 输入音频文件路径 |
| `--output` | `-o` | **必需** | 输出目录路径 |
| `--name` | `-n` | 自动 | 输出文件基础名称 |
| `--json` | - | - | 模型 JSON 配置文件路径 |
| `--gpu` | - | 自动检测 | 使用 GPU (NVIDIA CUDA) |
| `--cpu` | - | - | 强制使用 CPU |
| `--directml` | - | - | 使用 DirectML (AMD GPU) |
| `--device` | `-d` | `0` | GPU 设备 ID |
| `--segment-size` | - | `256` | 段大小 |
| `--overlap` | - | `0.25` | MDX 重叠率 (0.25-0.99) |
| `--overlap-mdxc` | - | `2` | MDX-C/Roformer 重叠 (2-50) |
| `--batch-size` | - | `1` | 批次大小 |
| `--wav-type` | - | `PCM_24` | 输出位深度: PCM_16, PCM_24, PCM_32, FLOAT (32-bit 浮点), DOUBLE (64-bit 浮点) |
| `--quiet` | `-q` | - | 静默模式 |

#### 输出控制参数

| 参数 | 说明 |
|------|------|
| `--primary-only` | 仅保存 primary stem |
| `--secondary-only` | 仅保存 secondary stem |
| `--vocals-only` | 仅保存人声（= --primary-only） |
| `--instrumental-only` | 仅保存伴奏（= --secondary-only） |
| `--dry-only` | 仅保存 Dry（= --primary-only） |
| `--no-dry-only` | 仅保存 No Dry（= --secondary-only） |
| `--stem` | 选择 stem（仅 MDX-C：all/vocals/drums/bass/other） |

### 使用示例

```powershell
# Roformer 模型 - 使用自定义 overlap
poetry run python mdx_headless_runner.py \
    -m "MDX23C-8KFFT-InstVoc_HQ.ckpt" \
    -i "song.flac" \
    -o "output/" \
    --gpu \
    --overlap-mdxc 8

# 只输出伴奏
poetry run python mdx_headless_runner.py \
    -m "model.ckpt" \
    -i "song.flac" \
    -o "output/" \
    --gpu \
    --instrumental-only
```

---

## Demucs Headless Runner

### 快速开始

```powershell
# 输出所有 stems（等同于 GUI "All Stems"）
poetry run python demucs_headless_runner.py --model htdemucs --input "song.flac" --output "output/" --gpu

# 只输出人声（等同于 GUI "Vocals" + "Primary Stem Only"）
poetry run python demucs_headless_runner.py --model htdemucs --input "song.flac" --output "output/" --gpu --stem Vocals --primary-only
```

### 支持的模型

| 模型 | 版本 | Stems | 说明 |
|------|------|-------|------|
| `htdemucs` | v4 | 4 | Drums, Bass, Other, Vocals |
| `htdemucs_ft` | v4 | 4 | Fine-tuned 版本，质量更好 |
| `htdemucs_6s` | v4 | 6 | Drums, Bass, Other, Vocals, Guitar, Piano |
| `hdemucs_mmi` | v4 | 4 | 标准版本 |
| `mdx_extra_q` | v3 | 4 | v3 版本 |

### 命令行参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | `-m` | **必需** | 模型名称或路径 |
| `--input` | `-i` | **必需** | 输入音频文件路径 |
| `--output` | `-o` | **必需** | 输出目录路径 |
| `--name` | `-n` | 自动 | 输出文件基础名称 |
| `--gpu` | - | 自动检测 | 使用 GPU (NVIDIA CUDA) |
| `--cpu` | - | - | 强制使用 CPU |
| `--directml` | - | - | 使用 DirectML (AMD GPU) |
| `--device` | `-d` | `0` | GPU 设备 ID |
| `--segment` | - | `Default` | 分段大小 (Default/1-100+，支持自定义值) |
| `--shifts` | - | `2` | 时间偏移次数 |
| `--overlap` | - | `0.25` | 重叠率 |
| `--stem` | - | - | 选择 stem (Vocals/Other/Bass/Drums/Guitar/Piano) |
| `--wav-type` | - | `PCM_24` | 输出位深度: PCM_16, PCM_24, PCM_32, FLOAT (32-bit 浮点), DOUBLE (64-bit 浮点) |
| `--primary-only` | - | - | 只输出 primary stem |
| `--secondary-only` | - | - | 只输出 secondary stem |
| `--quiet` | `-q` | - | 静默模式 |

### 使用示例

```powershell
# 输出所有 4 stems
poetry run python demucs_headless_runner.py \
    --model htdemucs \
    --input "song.flac" \
    --output "output/" \
    --gpu

# 只输出人声（1个文件）
poetry run python demucs_headless_runner.py \
    --model htdemucs \
    --input "song.flac" \
    --output "output/" \
    --gpu \
    --stem Vocals \
    --primary-only

# 只输出伴奏（选择 Vocals 但输出 secondary = Instrumental）
poetry run python demucs_headless_runner.py \
    --model htdemucs \
    --input "song.flac" \
    --output "output/" \
    --gpu \
    --stem Vocals \
    --secondary-only

# 6-stem 模型 - 输出所有 6 stems
poetry run python demucs_headless_runner.py \
    --model htdemucs_6s \
    --input "song.flac" \
    --output "output/" \
    --gpu

# 自定义 segment（提高质量）
poetry run python demucs_headless_runner.py \
    --model htdemucs_ft \
    --input "song.flac" \
    --output "output/" \
    --gpu \
    --segment 85
```

### Stem 选择说明

| GUI 操作 | CLI 等价命令 |
|---------|-------------|
| All Stems | 不指定 `--stem` |
| Vocals | `--stem Vocals` |
| Vocals + Primary Only | `--stem Vocals --primary-only` |
| Vocals + Secondary Only | `--stem Vocals --secondary-only` |

---

## 输出文件

### MDX-Net 模型

```
output/
├── {filename}_(Vocals).wav      # Primary stem
└── {filename}_(Instrumental).wav # Secondary stem
```

### Demucs 4-stem 模型

```
output/
├── {filename}_(Drums).wav
├── {filename}_(Bass).wav
├── {filename}_(Other).wav
└── {filename}_(Vocals).wav
```

### Demucs 6-stem 模型

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
    overlap_mdx23=2,  # MDX-C/Roformer
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
    demucs_stems='Vocals',  # 或 ALL_STEMS
    primary_only=True,
    verbose=True
)
```

---

## 故障排除

### GPU 不工作

```powershell
# 检查 CUDA 是否可用
poetry run python -c "import torch; print(torch.cuda.is_available())"

# 检查 GPU 设备
poetry run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 模型找不到

- MDX 模型默认位置：`C:\Users\{user}\AppData\Local\Programs\Ultimate Vocal Remover\models\MDX_Net_Models\`
- Demucs 模型默认位置：`...\Demucs_Models\v3_v4_repo\`

### 输出质量不佳

- 确保使用正确的模型配置（`--json` 参数）
- 尝试增加 `--overlap` 或 `--overlap-mdxc`
- 对于 Demucs，尝试增加 `--segment`

---

## 致谢

本项目基于 [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) 开发。

核心开发者：
- [Anjok07](https://github.com/anjok07)
- [aufr33](https://github.com/aufr33)

特别感谢：
- [ZFTurbo](https://github.com/ZFTurbo) - MDX23C 模型 & SCNet 实现
- [Adefossez & Facebook Research](https://github.com/facebookresearch/demucs) - Demucs AI 代码
- [Kuielab & Woosung Choi](https://github.com/kuielab) - MDX-Net AI 代码

### 本项目新增功能

- `mdx_headless_runner.py` - MDX/Roformer/SCNet 无头运行器
- `demucs_headless_runner.py` - Demucs 无头运行器
- 命令行接口支持
- GPU 优化

---

## 许可证

本项目使用 **MIT 许可证**。

```
MIT License

Copyright (c) 2022 Anjok07 (Ultimate Vocal Remover)
Copyright (c) 2026 UVR Headless Runner Contributors
```

详情请查看 [LICENSE](LICENSE) 文件。

### 第三方许可证

| 项目 | 许可证 | 链接 |
|------|--------|------|
| Ultimate Vocal Remover GUI | MIT | [GitHub](https://github.com/Anjok07/ultimatevocalremovergui) |
| Demucs | MIT | [GitHub](https://github.com/facebookresearch/demucs) |
| MDX-Net | MIT | [GitHub](https://github.com/kuielab/mdx-net) |
