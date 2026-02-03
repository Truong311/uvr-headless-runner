# UVR Headless Runner 开发进度总结

## 📋 项目概述

本项目为 Ultimate Vocal Remover (UVR) GUI 创建无头（headless）运行器，支持通过命令行进行音频分离。

### 支持的架构

| 架构 | Runner | 支持状态 |
|------|--------|---------|
| MDX-Net | `mdx_headless_runner.py` | ✅ 完成 |
| MDX-C (Roformer) | `mdx_headless_runner.py` | ✅ 完成 |
| SCNet | `mdx_headless_runner.py` | ✅ 完成 |
| Demucs (v1-v4) | `demucs_headless_runner.py` | ✅ 完成 |

---

## ✅ 已完成的工作

### 1. MDX Headless Runner (2026-02-03)

#### 1.1 核心功能
- ✅ 完整的 CLI 参数支持
- ✅ GPU/CUDA 支持（已验证 RTX 4060）
- ✅ 模型哈希计算与 UVR GUI 完全一致
- ✅ 配置加载回退链（JSON → hash lookup → auto-detect → defaults）
- ✅ MDX-C/Roformer 模型自动识别

#### 1.2 参数支持
- ✅ `--segment-size` - 段大小（默认 256）
- ✅ `--overlap` - MDX 重叠率（默认 0.25）
- ✅ `--overlap-mdxc` - MDX-C/Roformer 重叠（默认 2，范围 2-50）
- ✅ `--batch-size` - 批次大小
- ✅ 输出控制：`--primary-only`, `--secondary-only`, `--vocals-only`, `--instrumental-only`

#### 1.3 支持的模型类型
- ✅ MDX-Net 模型 (.ckpt/.onnx)
- ✅ MDX-C/Roformer 模型 (如 `MDX23C-8KFFT-InstVoc_HQ.ckpt`)
- ✅ SCNet 模型

### 2. Demucs Headless Runner (2026-02-03)

#### 2.1 核心功能
- ✅ 支持 Demucs v1/v2/v3/v4 所有版本
- ✅ 支持 4-stem 和 6-stem 模型
- ✅ 严格按照 GUI 行为（All Stems 或单选）
- ✅ GPU/CUDA 支持（已验证 RTX 4060）

#### 2.2 参数支持
- ✅ `--segment` - 分段大小（Default/1-100+，支持自定义值）
- ✅ `--shifts` - 时间偏移次数（默认 2）
- ✅ `--overlap` - 重叠率（默认 0.25）
- ✅ `--stem` - 选择 stem（Vocals/Other/Bass/Drums/Guitar/Piano）
- ✅ `--primary-only` / `--secondary-only` - 输出控制

#### 2.3 已测试的模型
- ✅ `htdemucs` (v4, 4-stem)
- ✅ `htdemucs_ft` (v4, 4-stem, fine-tuned)
- ✅ `htdemucs_6s` (v4, 6-stem)

### 3. GPU 优化 (2026-02-03)

- ✅ 修复了 Demucs 核显/独显混用问题
  - 问题：`torch.tensor()` 默认在 CPU 创建，导致频繁的 CPU-GPU 数据传输
  - 解决：直接在目标设备创建张量 `torch.tensor(..., device=self.device)`
- ✅ 验证 CUDA 设备检测正确（只检测到 RTX 4060）

### 4. 代码质量

- ✅ 简化了 Demucs 的文件删除逻辑（移除复杂的多选删除）
- ✅ 严格按照 GUI 行为：All Stems 或单选
- ✅ 修复了 PyTorch 2.6 的 `weights_only=True` 兼容性问题

---

## 📊 完成度

| 模块 | 完成度 |
|------|--------|
| MDX-Net Runner | 100% ✅ |
| Demucs Runner | 100% ✅ |
| GPU 支持 | 100% ✅ |
| 文档 | 100% ✅ |

**总体完成度**: 100% 🎉

---

## 🔧 默认参数对照表

### MDX Runner

| 参数 | CLI | 默认值 | GUI 默认 |
|------|-----|--------|----------|
| Segment Size | `--segment-size` | 256 | 256 |
| Overlap (MDX) | `--overlap` | 0.25 | Default |
| Overlap (MDX-C) | `--overlap-mdxc` | 2 | 2 |
| Batch Size | `--batch-size` | 1 | 1 |

### Demucs Runner

| 参数 | CLI | 默认值 | GUI 默认 |
|------|-----|--------|----------|
| Segment | `--segment` | Default | Default |
| Shifts | `--shifts` | 2 | 2 |
| Overlap | `--overlap` | 0.25 | 0.25 |

---

## 📁 关键文件

```
ultimatevocalremovergui-5.6.0_roformer_add-directml/
├── mdx_headless_runner.py      # MDX/Roformer/SCNet Runner
├── demucs_headless_runner.py   # Demucs Runner
├── separate.py                 # 核心分离逻辑（已修改）
├── demucs/
│   └── states.py               # 已修改 weights_only=False
├── HEADLESS_RUNNER_README.md   # 用户文档
└── PROGRESS.md                 # 本文档
```

---

## 🚀 使用示例

### MDX Runner

```powershell
# Roformer 模型
poetry run python mdx_headless_runner.py \
    -m "MDX23C-8KFFT-InstVoc_HQ.ckpt" \
    -i "song.flac" \
    -o "output/" \
    --gpu

# 自定义 MDX-C overlap
poetry run python mdx_headless_runner.py \
    -m "model.ckpt" \
    -i "song.flac" \
    -o "output/" \
    --gpu \
    --overlap-mdxc 8
```

### Demucs Runner

```powershell
# 输出所有 stems
poetry run python demucs_headless_runner.py \
    --model htdemucs \
    --input "song.flac" \
    --output "output/" \
    --gpu

# 只输出人声
poetry run python demucs_headless_runner.py \
    --model htdemucs \
    --input "song.flac" \
    --output "output/" \
    --gpu \
    --stem Vocals \
    --primary-only

# 6-stem 模型
poetry run python demucs_headless_runner.py \
    --model htdemucs_6s \
    --input "song.flac" \
    --output "output/" \
    --gpu
```

---

## 📝 更新日志

### 2026-02-03 (最新)

**Demucs Runner**
- ✅ 新增 `demucs_headless_runner.py`
- ✅ 支持所有 Demucs 版本（v1/v2/v3/v4）
- ✅ 支持 4-stem 和 6-stem 模型
- ✅ 严格按照 GUI 行为（All Stems 或单选）
- ✅ 修复 PyTorch 2.6 兼容性（weights_only）

**GPU 优化**
- ✅ 修复 Demucs 核显问题（张量直接在 GPU 创建）
- ✅ 验证 RTX 4060 独显正常工作

**MDX Runner**
- ✅ 新增 `--overlap-mdxc` 参数
- ✅ MDX-C 默认 overlap 改为 2（与 GUI 一致）

**代码质量**
- ✅ 简化 Demucs 文件输出逻辑
- ✅ 移除复杂的多选删除机制

---

---

## 📜 许可证合规

本项目遵循 MIT 许可证，与 UVR 原项目一致。

- ✅ `LICENSE` - MIT 许可证（包含原始 UVR 版权声明）
- ✅ `HEADLESS_RUNNER_README.md` - 包含致谢和第三方许可证说明
- ✅ 保留原始 `README.md`（UVR 官方文档）

---

**最后更新**: 2026-02-03  
**状态**: 已完成 ✅
