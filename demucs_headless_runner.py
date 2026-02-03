#!/usr/bin/env python3
"""
Demucs Headless Runner
用于在没有 GUI 的情况下运行 Demucs 模型分离

使用方法:
    # 使用 v4 htdemucs 模型
    python demucs_headless_runner.py --model htdemucs --input input.wav --output output/
    
    # 使用 6-stem 模型
    python demucs_headless_runner.py --model htdemucs_6s --input input.wav --output output/
    
    # 指定模型目录
    python demucs_headless_runner.py --model htdemucs --model-dir /path/to/v3_v4_repo --input input.wav --output output/
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from types import SimpleNamespace

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入必需的模块
from separate import SeperateDemucs, prepare_mix
from gui_data.constants import (
    DEMUCS_ARCH_TYPE,
    DEMUCS_V1, DEMUCS_V2, DEMUCS_V3, DEMUCS_V4,
    DEMUCS_V1_TAG, DEMUCS_V2_TAG, DEMUCS_V3_TAG, DEMUCS_V4_TAG,
    DEMUCS_VERSION_MAPPER,
    DEMUCS_2_SOURCE, DEMUCS_4_SOURCE,
    DEMUCS_2_SOURCE_MAPPER, DEMUCS_4_SOURCE_MAPPER, DEMUCS_6_SOURCE_MAPPER,
    DEMUCS_4_SOURCE_LIST, DEMUCS_6_SOURCE_LIST,
    DEMUCS_UVR_MODEL,
    VOCAL_STEM, INST_STEM, DRUM_STEM, BASS_STEM, OTHER_STEM, GUITAR_STEM, PIANO_STEM,
    DEFAULT, CUDA_DEVICE, CPU, ALL_STEMS,
    secondary_stem,
    PRIMARY_STEM
)

# 设备检测
mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
cuda_available = torch.cuda.is_available()
cpu = torch.device('cpu')

# 默认路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEMUCS_DIR = os.path.join(SCRIPT_DIR, 'models', 'Demucs_Models')
DEFAULT_DEMUCS_V3_V4_DIR = os.path.join(DEFAULT_DEMUCS_DIR, 'v3_v4_repo')


def get_demucs_version(model_name):
    """根据模型名称确定 Demucs 版本"""
    for version, tag in DEMUCS_VERSION_MAPPER.items():
        if tag.strip(' | ') in model_name or tag in model_name:
            return version
    
    # 根据模型名称特征判断
    if 'htdemucs' in model_name.lower():
        return DEMUCS_V4
    elif 'hdemucs' in model_name.lower():
        return DEMUCS_V3
    elif model_name.endswith('.gz') or 'demucs' in model_name.lower():
        return DEMUCS_V2
    
    # 默认 v4
    return DEMUCS_V4


def get_demucs_sources(model_name):
    """根据模型名称确定源配置"""
    if DEMUCS_UVR_MODEL in model_name or '2stem' in model_name.lower():
        return DEMUCS_2_SOURCE, DEMUCS_2_SOURCE_MAPPER, 2
    elif '6s' in model_name.lower() or 'htdemucs_6s' in model_name.lower():
        return DEMUCS_6_SOURCE_LIST, DEMUCS_6_SOURCE_MAPPER, 6
    else:
        return DEMUCS_4_SOURCE, DEMUCS_4_SOURCE_MAPPER, 4


def find_demucs_model_path(model_name, model_dir=None):
    """查找 Demucs 模型路径"""
    search_dirs = []
    
    if model_dir:
        search_dirs.append(model_dir)
    
    # 默认搜索路径
    search_dirs.extend([
        DEFAULT_DEMUCS_V3_V4_DIR,
        DEFAULT_DEMUCS_DIR,
        os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Programs',
                     'Ultimate Vocal Remover', 'models', 'Demucs_Models', 'v3_v4_repo'),
        os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Programs',
                     'Ultimate Vocal Remover', 'models', 'Demucs_Models'),
    ])
    
    # 如果是完整路径
    if os.path.isfile(model_name):
        return model_name
    
    # 搜索模型文件
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        
        # 尝试 YAML 配置文件 (v3/v4)
        yaml_path = os.path.join(search_dir, f'{model_name}.yaml')
        if os.path.isfile(yaml_path):
            return yaml_path
        
        # 尝试 .th 文件
        th_path = os.path.join(search_dir, f'{model_name}.th')
        if os.path.isfile(th_path):
            return th_path
        
        # 尝试 .gz 文件 (v1)
        gz_path = os.path.join(search_dir, f'{model_name}.gz')
        if os.path.isfile(gz_path):
            return gz_path
        
        # 尝试 .pth 文件 (v2)
        pth_path = os.path.join(search_dir, f'{model_name}.pth')
        if os.path.isfile(pth_path):
            return pth_path
    
    return None


# ============================================================================
# IMPORTANT:
# This logic MUST stay behavior-identical to UVR GUI.
# Do NOT refactor, "optimize", or reinterpret unless UVR itself changes.
# ============================================================================
def create_demucs_model_data(model_path, **kwargs):
    """创建 Demucs 模型的 ModelData 对象 - 严格复制 UVR GUI 的 ModelData 结构"""
    model_data = SimpleNamespace()
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # ========== 基本信息 ==========
    model_data.model_path = model_path
    model_data.model_name = model_name
    model_data.model_basename = model_name
    model_data.process_method = DEMUCS_ARCH_TYPE
    
    # ========== Demucs 特定参数 ==========
    demucs_version = kwargs.get('demucs_version')
    model_data.demucs_version = demucs_version if demucs_version else get_demucs_version(model_name)
    
    source_list, source_map, stem_count = get_demucs_sources(model_name)
    model_data.demucs_source_list = kwargs.get('demucs_source_list', source_list)
    model_data.demucs_source_map = kwargs.get('demucs_source_map', source_map)
    model_data.demucs_stem_count = kwargs.get('demucs_stem_count', stem_count)
    
    model_data.demucs_stems = kwargs.get('demucs_stems', ALL_STEMS)
    model_data.is_chunk_demucs = kwargs.get('is_chunk_demucs', False)
    model_data.segment = kwargs.get('segment', 'Default')
    model_data.shifts = kwargs.get('shifts', 2)
    model_data.is_split_mode = kwargs.get('is_split_mode', True)
    model_data.is_demucs_combine_stems = kwargs.get('is_demucs_combine_stems', True)
    
    # ========== 设备设置 ==========
    use_gpu = kwargs.get('use_gpu', cuda_available)
    model_data.is_gpu_conversion = 0 if use_gpu else -1
    model_data.device_set = kwargs.get('device_set', '0')
    model_data.is_use_opencl = False
    model_data.is_use_directml = kwargs.get('is_use_directml', False)  # For AMD GPUs
    
    # ========== Stem 设置 ==========
    primary_only = kwargs.get('primary_only', False)
    secondary_only = kwargs.get('secondary_only', False)
    
    if primary_only and secondary_only:
        secondary_only = False
    
    model_data.is_primary_stem_only = primary_only
    model_data.is_secondary_stem_only = secondary_only
    
    # 设置 primary/secondary stem
    if model_data.demucs_stems == ALL_STEMS:
        model_data.primary_stem = PRIMARY_STEM
    else:
        model_data.primary_stem = model_data.demucs_stems
    model_data.secondary_stem = secondary_stem(model_data.primary_stem)
    model_data.primary_stem_native = model_data.primary_stem
    
    # ========== 输出设置 ==========
    model_data.wav_type_set = kwargs.get('wav_type_set', 'PCM_24')  # 默认 24-bit
    model_data.save_format = kwargs.get('save_format', 'WAV')
    model_data.mp3_bit_set = kwargs.get('mp3_bit_set', None)
    model_data.is_normalization = kwargs.get('is_normalization', True)
    
    # ========== 二级模型 ==========
    model_data.is_secondary_model_activated = False
    model_data.is_secondary_model = False
    model_data.secondary_model = None
    model_data.secondary_model_scale = None
    model_data.primary_model_primary_stem = None
    model_data.is_pre_proc_model = False
    model_data.pre_proc_model = None
    model_data.secondary_model_4_stem = [None, None, None, None]
    model_data.secondary_model_4_stem_scale = [None, None, None, None]
    
    # ========== Vocal Split ==========
    model_data.vocal_split_model = None
    model_data.is_vocal_split_model = False
    model_data.is_save_inst_vocal_splitter = False
    model_data.is_inst_only_voc_splitter = False
    model_data.is_save_vocal_only = False
    
    # ========== Denoise/Deverb ==========
    model_data.is_denoise = False
    model_data.is_denoise_model = False
    model_data.DENOISER_MODEL = None
    model_data.DEVERBER_MODEL = None
    model_data.is_deverb_vocals = False
    model_data.deverb_vocal_opt = None
    
    # ========== Pitch ==========
    model_data.is_pitch_change = False
    model_data.semitone_shift = 0.0
    model_data.is_match_frequency_pitch = False
    
    # ========== Ensemble ==========
    model_data.is_ensemble_mode = False
    model_data.ensemble_primary_stem = None
    model_data.ensemble_secondary_stem = None
    model_data.is_multi_stem_ensemble = False
    model_data.is_4_stem_ensemble = False
    
    # ========== 其他标志 ==========
    model_data.mixer_path = None
    model_data.model_samplerate = 44100
    model_data.is_invert_spec = kwargs.get('is_invert_spec', False)
    model_data.is_mixer_mode = False
    model_data.is_karaoke = False
    model_data.is_bv_model = False
    model_data.bv_model_rebalance = 0
    model_data.is_sec_bv_rebalance = False
    model_data.is_demucs_pre_proc_model_inst_mix = False
    model_data.overlap = kwargs.get('overlap', 0.25)
    model_data.overlap_mdx = kwargs.get('overlap_mdx', 0.25)
    model_data.overlap_mdx23 = kwargs.get('overlap_mdx23', 8)
    
    # ========== MDX 相关（Demucs 不使用，但 SeperateAttributes 需要） ==========
    model_data.is_mdx_combine_stems = False
    model_data.is_mdx_c = False
    model_data.mdx_c_configs = None
    model_data.mdxnet_stem_select = None
    model_data.model_capacity = (32, 128)
    model_data.is_vr_51_model = False
    model_data.is_target_instrument = False
    model_data.is_roformer = False
    
    return model_data


def create_process_data(audio_file, export_path, audio_file_base=None, **kwargs):
    """创建 process_data 字典"""
    if audio_file_base is None:
        audio_file_base = os.path.splitext(os.path.basename(audio_file))[0]
    
    def noop_progress(step=0, inference_iterations=0):
        pass
    
    return {
        'audio_file': audio_file,
        'export_path': export_path,
        'audio_file_base': audio_file_base,
        'set_progress_bar': noop_progress,
        'write_to_console': lambda text, base_text='': print(text.strip()),
        'process_iteration': lambda: None,
        'cached_source_callback': lambda *args, **kw: (None, None),
        'cached_model_source_holder': lambda *args, **kw: None,
        'list_all_models': [],
        'is_ensemble_master': False,
        'is_4_stem_ensemble': False,
        'is_multi_stem_ensemble': False,
    }


def run_demucs_headless(
    model_path,
    audio_file,
    export_path,
    audio_file_base=None,
    use_gpu=None,
    device_set='0',
    is_use_directml=False,
    demucs_version=None,
    segment='Default',
    shifts=2,
    overlap=0.25,
    wav_type_set='PCM_24',
    demucs_stems=ALL_STEMS,  # ALL_STEMS 或单个 stem 名称 (Vocals/Other/Bass/Drums/Guitar/Piano)
    primary_only=False,
    secondary_only=False,
    verbose=True
):
    """
    运行 Demucs 分离的主函数（严格按照 GUI 行为）
    
    Args:
        model_path: 模型文件路径或模型名称
        audio_file: 输入音频文件路径
        export_path: 输出目录路径
        audio_file_base: 输出文件基名（可选）
        use_gpu: 是否使用 GPU（默认自动检测）
        device_set: GPU 设备 ID
        demucs_version: Demucs 版本 (v1/v2/v3/v4)
        segment: 分段大小
        shifts: 时间偏移次数
        overlap: 重叠率
        demucs_stems: ALL_STEMS（输出所有）或单个 stem 名称（只输出该 stem）
        primary_only: 只输出 primary stem
        secondary_only: 只输出 secondary stem
        verbose: 是否显示详细输出
    
    Returns:
        输出文件路径字典
    """
    import glob
    
    # 确保输出目录存在
    os.makedirs(export_path, exist_ok=True)
    
    # 处理音频文件基名
    if audio_file_base is None:
        audio_file_base = os.path.splitext(os.path.basename(audio_file))[0]
    
    # 创建 model_data
    # 转换 wav_type 名称为 soundfile 格式
    wav_type_map = {
        'PCM_U8': 'PCM_U8',
        'PCM_16': 'PCM_16',
        'PCM_24': 'PCM_24',
        'PCM_32': 'PCM_32',
        'FLOAT': 'FLOAT',
        'DOUBLE': 'DOUBLE',
        '32-bit Float': 'FLOAT',
        '64-bit Float': 'DOUBLE'
    }
    wav_type = wav_type_map.get(wav_type_set, 'PCM_24')
    
    model_data = create_demucs_model_data(
        model_path,
        use_gpu=use_gpu if use_gpu is not None else cuda_available,
        device_set=device_set,
        is_use_directml=is_use_directml,
        demucs_version=demucs_version,
        segment=segment,
        shifts=shifts,
        overlap=overlap,
        wav_type_set=wav_type,
        demucs_stems=demucs_stems,
        primary_only=primary_only,
        secondary_only=secondary_only,
        verbose=verbose
    )
    
    # 创建 process_data
    process_data = create_process_data(audio_file, export_path, audio_file_base, verbose=verbose)
    
    if verbose:
        print("=" * 50)
        print("Demucs Headless Runner")
        print("=" * 50)
        print(f"模型: {model_path}")
        print(f"输入: {audio_file}")
        print(f"输出目录: {export_path}")
        print(f"设备: {'GPU' if model_data.is_gpu_conversion >= 0 else 'CPU'}")
        print(f"Device Set: {model_data.device_set}")
        # 显示实际的 PyTorch 设备
        import torch
        if torch.cuda.is_available():
            print(f"CUDA 设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"当前默认 CUDA 设备: {torch.cuda.current_device()}")
        print(f"Demucs 版本: {model_data.demucs_version}")
        print(f"Stems: {model_data.demucs_stem_count}")
        print(f"Shifts: {shifts}")
        print(f"输出格式: {model_data.wav_type_set}")
        print("=" * 50)
    
    # 运行分离（严格按照 GUI 行为，由 separate.py 控制输出）
    separator = SeperateDemucs(model_data, process_data)
    separator.seperate()
    
    if verbose:
        print(f"\n处理完成!")
        print(f"输出目录: {export_path}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='Demucs Headless Runner - 使用 Demucs 模型进行音频分离（严格按照 GUI 行为）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 输出所有 stems（等同于 GUI "All Stems"）
    python demucs_headless_runner.py --model htdemucs --input song.wav --output ./output --gpu
    
    # 选择 Vocals stem（输出 Vocals + Instrumental，等同于 GUI 选择 "Vocals"）
    python demucs_headless_runner.py --model htdemucs --input song.wav --output ./output --gpu --stem Vocals
    
    # 只输出 Vocals 一个文件（等同于 GUI 选择 "Vocals" + 勾选 "Primary Stem Only"）
    python demucs_headless_runner.py --model htdemucs --input song.wav --output ./output --gpu --stem Vocals --primary-only
    
    # 只输出伴奏（选择 Vocals 但只要 secondary = Instrumental）
    python demucs_headless_runner.py --model htdemucs --input song.wav --output ./output --gpu --stem Vocals --secondary-only

可用的 stem 选项:
    4-stem 模型: Vocals, Other, Bass, Drums
    6-stem 模型: Vocals, Other, Bass, Drums, Guitar, Piano

说明:
    --stem X              选择 stem X 为 primary，输出 X 和对应的 secondary
    --stem X --primary-only   只输出 stem X
    --stem X --secondary-only 只输出 stem X 的 secondary（互补）
"""
    )
    
    parser.add_argument('--model', '-m', required=True, help='模型名称或路径')
    parser.add_argument('--model-dir', help='模型目录路径')
    parser.add_argument('--input', '-i', required=True, help='输入音频文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出目录路径')
    parser.add_argument('--name', '-n', help='输出文件基名')
    
    parser.add_argument('--gpu', action='store_true', help='使用 GPU')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU')
    parser.add_argument('--directml', action='store_true', help='使用 DirectML (AMD GPU)')
    parser.add_argument('--device', '-d', default='0', help='GPU 设备 ID (默认: 0)')
    
    parser.add_argument('--shifts', type=int, default=2, help='时间偏移次数 (默认: 2)')
    parser.add_argument('--overlap', type=float, default=0.25, help='重叠率 (默认: 0.25)')
    parser.add_argument('--segment', default='Default', help='分段大小 (默认: Default)')
    parser.add_argument('--wav-type', default='PCM_24',
                        choices=['PCM_U8', 'PCM_16', 'PCM_24', 'PCM_32', 'FLOAT', 'DOUBLE'],
                        help='输出音频位深度 (默认: PCM_24)')
    
    parser.add_argument('--stem', help='只输出指定 stem (Vocals/Other/Bass/Drums，6-stem 模型还有 Guitar/Piano)，不指定则输出全部')
    parser.add_argument('--primary-only', action='store_true', help='只输出 primary stem')
    parser.add_argument('--secondary-only', action='store_true', help='只输出 secondary stem')
    
    parser.add_argument('--quiet', '-q', action='store_true', help='安静模式')
    
    args = parser.parse_args()
    
    # 查找模型路径
    model_path = find_demucs_model_path(args.model, args.model_dir)
    if model_path is None:
        print(f"错误: 找不到模型 '{args.model}'", file=sys.stderr)
        print(f"搜索的目录:", file=sys.stderr)
        print(f"  - {DEFAULT_DEMUCS_V3_V4_DIR}", file=sys.stderr)
        print(f"  - {DEFAULT_DEMUCS_DIR}", file=sys.stderr)
        return 1
    
    # 确定 GPU 使用
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = cuda_available
    
    # 确定 stems 选择（严格按照 GUI：ALL_STEMS 或单个 stem）
    demucs_stems = ALL_STEMS
    if args.stem:
        demucs_stems = args.stem
    
    try:
        run_demucs_headless(
            model_path=model_path,
            audio_file=args.input,
            export_path=args.output,
            audio_file_base=args.name,
            use_gpu=use_gpu,
            device_set=args.device,
            is_use_directml=args.directml,
            shifts=args.shifts,
            overlap=args.overlap,
            segment=args.segment,
            wav_type_set=args.wav_type,
            demucs_stems=demucs_stems,
            primary_only=args.primary_only,
            secondary_only=args.secondary_only,
            verbose=not args.quiet
        )
        
        return 0
    except Exception as e:
        import traceback
        print(f"错误: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
