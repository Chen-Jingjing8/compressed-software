"""
image_compress.py (auto-pipeline)
=================================

在“增强版”的基础上，新增**自动批处理管线**：
- 直接运行本文件（不带任何命令行参数）即可：
  1) 扫描 ./data/ 下的图片；
  2) 对每张图片在 **灰度 L** 与 **RGB（三通道分别熵编码）** 模式下，
     分别用 Huffman / Shannon / SFE 编码压缩与解压；
  3) 保存“灰度中间图”、直方图（CSV/PNG）、压缩文件 .icomp、重建图；
  4) 汇总统计并**绘制对比图**（平均码长 vs 熵、压缩率）；
  5) 输出到 ./out, ./hists, ./results/figures, ./results/tables。
- 若提供命令行参数，仍可像之前那样 encode/decode（保留兼容）。

依赖：
  - Pillow: pip install pillow
  - matplotlib: pip install matplotlib
  - numpy: pip install numpy  （matplotlib 依赖）
"""

import argparse
import json
import math
import os
import sys
import hashlib
from typing import Dict, Iterable, List, Tuple, Optional

# ------------------------ 库依赖（绘图） ------------------------
try:
    import numpy as np
except Exception:
    np = None  # 允许在无 numpy 环境下运行不生成图
try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    plt = None

# ---------------------------------------------------------------------
# 兼容性导入：优先从 src/ 目录导入（标准项目结构）
# ---------------------------------------------------------------------
# 获取项目根目录（image_compress.py可能在experiments子目录中）
_script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_script_dir) == 'experiments':
    _project_root = os.path.dirname(_script_dir)
else:
    _project_root = _script_dir

# 优先从src目录导入
_src_path = os.path.join(_project_root, 'src')
if os.path.exists(_src_path):
    sys.path.insert(0, _src_path)

try:
    from encoder import HuffmanEncoder, ShannonEncoder, ShannonFanoEliasEncoder
    try:
        from utils import get_frequency_distribution  # 可选
    except Exception:
        get_frequency_distribution = None  # type: ignore
except Exception:
    # 回退：尝试从项目根目录导入
    sys.path.insert(0, _project_root)
    try:
        from encoder import HuffmanEncoder, ShannonEncoder, ShannonFanoEliasEncoder
        try:
            from utils import get_frequency_distribution  # 可选
        except Exception:
            get_frequency_distribution = None  # type: ignore
    except Exception:
        raise ImportError("无法导入encoder模块，请确保src/encoder.py存在")

from PIL import Image
from collections import Counter

# ---------------------------------------------------------------------
# 阅读指引（仅增强注释，不改逻辑）
# - 基础工具：位串转换、概率统计、熵与平均码长
# - 核心流程：encode_image / decode_image
# - 实验流程：auto_pipeline（批量压缩、解压、汇总、绘图）
# - 展示层：plot_gray_figures / plot_rgb_figures
# - 入口：build_cli / main
# ---------------------------------------------------------------------


# ============================ 基础工具 ============================
# 将 '0'/'1' 位串打包为字节，返回 (字节数据, 末尾补零位数)。
def bits_to_bytes(bits: str) -> Tuple[bytes, int]:
    """'0'/'1'字符串 -> 字节序列（返回 bytes 与 padding_bits 个数）"""
    padding = (8 - len(bits) % 8) % 8
    if padding:
        bits = bits + '0' * padding
    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int(bits[i:i+8], 2))
    return bytes(out), padding


# 将字节数据展开为位串，保留每个字节前导 0（固定 8 位）。
def bytes_to_bits(data: bytes) -> str:
    """字节序列 -> '0'/'1'字符串（保留前导 0）"""
    return ''.join(f'{b:08b}' for b in data)


# 统计频数与概率分布；若项目提供 utils 实现则优先复用。
def freq_prob_from_iterable(seq: Iterable[int]) -> Tuple[Dict[int, int], Dict[int, float]]:
    """统计频率与概率分布；若项目提供 utils.get_frequency_distribution 则优先使用。"""
    if get_frequency_distribution is not None:
        try:
            freq, probs = get_frequency_distribution(list(seq))
            freq = {int(k): int(v) for k, v in freq.items()}
            probs = {int(k): float(v) for k, v in probs.items()}
            return freq, probs
        except Exception:
            pass
    c = Counter(seq)
    total = sum(c.values())
    if total == 0:
        return {}, {}
    probs = {k: v / total for k, v in c.items()}
    return dict(c), probs


# 根据信号概率分布计算信息熵 H(X)。
def entropy_from_probs(probs: Dict[int, float]) -> float:
    """H(X) = -sum p log2 p"""
    H = 0.0
    for p in probs.values():
        if p > 0:
            H -= p * math.log2(p)
    return H


# 计算平均码长 L_avg = sum p(x)*|code(x)|。
def average_code_length(codes: Dict[int, str], probs: Dict[int, float]) -> float:
    """L_avg = sum_x p(x) * |code(x)|"""
    return sum(probs[sym] * len(codes[sym]) for sym in probs if sym in codes)


# 对原始像素字节做 SHA-256，用于校验重建一致性。
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================ 编码器工厂 ============================
# 根据 method 构造具体编码器实例（Huffman / Shannon / SFE）。
def make_encoder(method: str, probs: Dict[int, float]):
    m = method.lower()
    if m == 'huffman':
        return HuffmanEncoder(probs)
    elif m == 'shannon':
        return ShannonEncoder(probs)
    elif m in ('sfe', 'shannon-fano-elias', 'shannon_fano_elias'):
        return ShannonFanoEliasEncoder(probs)
    else:
        raise ValueError(f"不支持的编码方法：{method}")


# ============================ 直方图导出 ============================
# 导出单通道直方图：CSV（数值/计数/概率）+ PNG（若 matplotlib 可用）。
def export_histogram(prefix: str, channel_name: str, values: List[int]) -> None:
    """
    导出某一通道的直方图到 CSV 与（若可用）PNG。
    CSV 字段：value,count,prob
    PNG：柱状图（若 matplotlib 可用）。
    """
    import csv
    counts = [0]*256
    for v in values:
        if 0 <= v <= 255:
            counts[v] += 1
    total = len(values) if len(values) > 0 else 1
    probs = [c/total for c in counts]

    csv_path = f"{prefix}_{channel_name}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['value', 'count', 'prob'])
        for v in range(256):
            w.writerow([v, counts[v], f"{probs[v]:.10f}"])
    print(f"[信息] 已导出直方图 CSV：{csv_path}")

    if plt is not None:
        try:
            plt.figure()
            if np is not None:
                xs = np.arange(256)
                ys = np.array(counts)
            else:
                xs = list(range(256))
                ys = counts
            plt.bar(xs, ys)
            plt.title(f'Histogram - {channel_name}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            plt.tight_layout()
            png_path = f"{prefix}_{channel_name}.png"
            plt.savefig(png_path, dpi=160)
            plt.close()
            print(f"[信息] 已导出直方图 PNG：{png_path}")
        except Exception as e:
            print(f"[警告] 直方图 PNG 导出失败：{e}")


# ============================ 核心：编码/解码 ============================
# 压缩主流程：
# 1) 读取图片并转为目标模式（L 或 RGB）
# 2) 按通道统计概率并熵编码
# 3) 写入 icomp-v2 容器（元数据 + 分段负载）
# 4) 返回统计行，供批处理汇总和绘图
def encode_image(input_path: str,
                 output_path: str,
                 method: str = 'huffman',
                 mode: str = 'L',
                 save_gray: Optional[str] = None,
                 export_hist_prefix: Optional[str] = None) -> Dict[str, float]:
    """
    压缩图片：
      - mode=L：将输入转换为灰度，按像素统计并熵编码；
      - mode=RGB：将输入转换为 RGB，分别对 R/G/B 三通道统计并熵编码（**三段负载**）。
    返回一个“统计字典”用于后续汇总与绘图。
    """
    img = Image.open(input_path).convert(mode)
    width, height = img.size

    # 像素字节的哈希（用于无失真校验）
    pixel_bytes = img.tobytes()
    pixel_hash = sha256_bytes(pixel_bytes)

    if mode == 'L':
        values = list(img.getdata())
        channels = {'L': values}
        n_channels = 1
        bits_per_pixel_raw = 8
        if save_gray:
            os.makedirs(os.path.dirname(save_gray), exist_ok=True)
            try:
                img.save(save_gray)
            except Exception as e:
                print(f"[警告] 保存灰度图失败：{e}")
    elif mode == 'RGB':
        r, g, b = img.split()
        channels = {'R': list(r.getdata()), 'G': list(g.getdata()), 'B': list(b.getdata())}
        n_channels = 3
        bits_per_pixel_raw = 24
    else:
        raise ValueError("暂仅支持 L 与 RGB 模式。")

    # 可选直方图
    if export_hist_prefix:
        for cname, vals in channels.items():
            export_histogram(export_hist_prefix, cname, vals)

    # 每通道编码
    codes_all: Dict[str, Dict[int, str]] = {}
    paddings: Dict[str, int] = {}
    payloads: Dict[str, bytes] = {}
    stats_per_ch = {}

    for cname, vals in channels.items():
        _, probs = freq_prob_from_iterable(vals)
        if not probs:
            raise ValueError(f"通道 {cname} 概率分布为空。")

        enc = make_encoder(method, probs)
        codes = getattr(enc, 'codes', None) or getattr(enc, 'codebook', None)
        if codes is None:
            raise RuntimeError("无法从编码器取得码本（期待 .codes 或 .codebook）。")

        bits = ''.join(codes[v] for v in vals)
        byts, pad = bits_to_bytes(bits)
        codes_all[cname] = codes  # type: ignore[assignment]
        paddings[cname] = pad
        payloads[cname] = byts

        H = getattr(enc, 'entropy', None)
        if H is None:
            H = entropy_from_probs(probs)
        if hasattr(enc, 'get_average_code_length'):
            try:
                Lavg = float(enc.get_average_code_length())
            except Exception:
                Lavg = average_code_length(codes, probs)  # type: ignore[arg-type]
        else:
            Lavg = average_code_length(codes, probs)  # type: ignore[arg-type]

        eff = None
        if hasattr(enc, 'get_efficiency'):
            try:
                eff = float(enc.get_efficiency())
            except Exception:
                eff = None
        if eff is None and Lavg > 0:
            eff = float(H) / float(Lavg)

        stats_per_ch[cname] = {'H': float(H), 'Lavg': float(Lavg), 'eff': float(eff) if eff else None}

    seg_names = list(payloads.keys())
    seg_lengths = [len(payloads[n]) for n in seg_names]

    H_total = sum(stats_per_ch[n]['H'] for n in seg_names)
    Lavg_total = sum(stats_per_ch[n]['Lavg'] for n in seg_names)
    effective_bits = sum(int(len(channels[n]) * stats_per_ch[n]['Lavg']) for n in seg_names)
    original_bits = len(next(iter(channels.values()))) * bits_per_pixel_raw
    ratio = effective_bits / original_bits if original_bits > 0 else 1.0

    metadata = {
        'format': 'icomp-v2',
        'method': method,
        'mode': mode,
        'width': width,
        'height': height,
        'num_pixels': width * height,
        'n_channels': n_channels,
        'channel_order': ''.join(seg_names),  # 'L' 或 'RGB'
        'pixel_sha256': pixel_hash,
        'codes': {ch: {str(k): v for k, v in codes_all[ch].items()} for ch in seg_names},
        'padding_bits': paddings,
        'segments': seg_names,
        'segment_lengths_bytes': seg_lengths,
        'stats_per_channel': stats_per_ch,
        'H_total': float(H_total),
        'Lavg_total': float(Lavg_total),
        'effective_bits': int(effective_bits),
        'original_bits': int(original_bits),
        'ratio_bits_over_raw': float(ratio),
    }

    # 写容器（4B meta_len + meta_json + 各段负载）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        meta_bytes = json.dumps(metadata, ensure_ascii=False).encode('utf-8')
        f.write(len(meta_bytes).to_bytes(4, 'big'))
        f.write(meta_bytes)
        for n in seg_names:
            f.write(payloads[n])

    file_size_bytes = os.path.getsize(output_path)

    print(f"[完成] {os.path.basename(input_path)} | mode={mode} method={method} | "
          f"H_sum={H_total:.4f} L_sum={Lavg_total:.4f} ratio={ratio:.4f} file={file_size_bytes}B")

    # 返回供汇总的统计
    row = {
        'image': os.path.basename(input_path),
        'mode': mode,
        'method': method,
        'width': width, 'height': height, 'pixels': width*height,
        'channels': ''.join(seg_names),
        'H_total': float(H_total),
        'Lavg_total': float(Lavg_total),
        'ratio_bits_over_raw': float(ratio),
        'file_bytes': file_size_bytes
    }
    # 展开通道
    for ch in seg_names:
        row[f'H_{ch}'] = stats_per_ch[ch]['H']
        row[f'Lavg_{ch}'] = stats_per_ch[ch]['Lavg']
        if stats_per_ch[ch]['eff'] is not None:
            row[f'eta_{ch}'] = stats_per_ch[ch]['eff']
    # 也返回中间灰度图路径（便于后续 PSNR）
    if save_gray:
        row['gray_path'] = save_gray
    return row


# 解压主流程：读取容器 -> 按码本逐通道解码 -> 重建并保存图片。
def decode_image(input_path: str,
                 output_path: str) -> Image.Image:
    """读取容器 -> 解码 -> 返回 PIL 图像对象（并保存到 output_path）"""
    with open(input_path, 'rb') as f:
        meta_len = int.from_bytes(f.read(4), 'big')
        meta = json.loads(f.read(meta_len).decode('utf-8'))
        payload = f.read()

    mode = meta.get('mode', 'L')
    width = int(meta['width']); height = int(meta['height'])
    num_pixels = int(meta['num_pixels'])

    segments = meta.get('segments')
    seg_lengths = meta.get('segment_lengths_bytes')

    codes_meta = meta['codes']
    if isinstance(codes_meta, dict) and all(k in codes_meta for k in ('R','G','B')):
        codes = {'R': {int(k): v for k, v in codes_meta['R'].items()},
                 'G': {int(k): v for k, v in codes_meta['G'].items()},
                 'B': {int(k): v for k, v in codes_meta['B'].items()}}
    elif isinstance(codes_meta, dict) and 'L' in codes_meta:
        codes = {'L': {int(k): v for k, v in codes_meta['L'].items()}}
    else:
        codes = {'L': {int(k): v for k, v in codes_meta.items()}}

    reverse = {ch: {code: sym for sym, code in codes[ch].items()} for ch in codes}

    payloads: Dict[str, bytes] = {}
    if segments and seg_lengths:
        offset = 0
        for ch_name, nbytes in zip(segments, seg_lengths):
            nbytes = int(nbytes)
            payloads[ch_name] = payload[offset: offset + nbytes]
            offset += nbytes
    else:
        payloads['L'] = payload

    decoded_channels: Dict[str, List[int]] = {}
    for ch_name, byts in payloads.items():
        bits = bytes_to_bits(byts)
        rev = reverse[ch_name]
        vals: List[int] = []
        cur = ''
        for b in bits:
            cur += b
            if cur in rev:
                vals.append(rev[cur])
                cur = ''
                if len(vals) == num_pixels:
                    break
        decoded_channels[ch_name] = vals

    if mode == 'RGB':
        r_vals = decoded_channels.get('R'); g_vals = decoded_channels.get('G'); b_vals = decoded_channels.get('B')
        if not (r_vals and g_vals and b_vals):
            raise RuntimeError("RGB 解码缺少某个通道。")
        data = list(zip(r_vals, g_vals, b_vals))
        img = Image.new('RGB', (width, height))
        img.putdata(data[:width*height])
    else:
        vals = decoded_channels.get('L') or next(iter(decoded_channels.values()))
        img = Image.new('L', (width, height))
        img.putdata((vals or [])[:width*height])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    return img


# ============================ PSNR ============================
# 计算 PSNR（8-bit），用于评估重建图与参考图差异。
def compute_psnr(img_ref: Image.Image, img_test: Image.Image) -> float:
    """计算 L 或 RGB 图像的 PSNR（8-bit 量化）"""
    if img_ref.mode != img_test.mode:
        img_test = img_test.convert(img_ref.mode)

    if img_ref.mode == 'L':
        ref = list(img_ref.getdata())
        tst = list(img_test.getdata())
        mse = mse_list(ref, tst)
    elif img_ref.mode == 'RGB':
        r1, g1, b1 = img_ref.split()
        r2, g2, b2 = img_test.split()
        mse_r = mse_list(list(r1.getdata()), list(r2.getdata()))
        mse_g = mse_list(list(g1.getdata()), list(g2.getdata()))
        mse_b = mse_list(list(b1.getdata()), list(b2.getdata()))
        mse = (mse_r + mse_g + mse_b) / 3.0
    else:
        raise ValueError("PSNR 仅支持 L / RGB 模式。")

    if mse == 0:
        return float('inf')
    MAXI = 255.0
    return 10.0 * math.log10((MAXI * MAXI) / mse)


# 计算两个序列的均方误差 MSE（长度不一致时按较短长度）。
def mse_list(a: List[int], b: List[int]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    err = 0.0
    for i in range(n):
        d = float(a[i]) - float(b[i])
        err += d * d
    return err / n


# ============================ 自动管线 ============================
# 扫描目录中的常见图片格式并返回完整路径列表。
def list_images(data_dir: str) -> List[str]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    if not os.path.isdir(data_dir):
        return []
    files = []
    for name in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            files.append(os.path.join(data_dir, name))
    return files


# 批处理入口：
# - 每张图分别做 L 与 RGB 两种模式
# - 每种模式跑 huffman / shannon / sfe
# - 输出压缩文件、重建图、直方图、统计表与对比图
def auto_pipeline(data_dir: str = 'data/images') -> None:
    """
    全自动批处理：扫描 data/images/，对每张图在 L 与 RGB 模式下用三种编码方法压缩、解压、统计与绘图。
    输出：
      - out/*.icomp, out/*_recon.png, out/*_gray.png（仅 L）
      - hists/*_{L|R|G|B}.csv/.png
      - results/tables/{gray|rgb}_summary.csv
      - results/figures/*.png（若安装了 matplotlib）
    """
    # 获取项目根目录（image_compress.py可能在experiments子目录中）
    script_dir = os.path.abspath(os.path.dirname(__file__))
    # 如果脚本在experiments目录中，需要回到项目根目录
    if os.path.basename(script_dir) == 'experiments':
        base_dir = os.path.dirname(script_dir)
    else:
        base_dir = script_dir
    
    data_dir = os.path.join(base_dir, data_dir)
    out_dir = os.path.join(base_dir, 'out')
    hist_dir = os.path.join(base_dir, 'hists')
    fig_dir = os.path.join(base_dir, 'results', 'figures')
    tbl_dir = os.path.join(base_dir, 'results', 'tables')
    for d in (out_dir, hist_dir, fig_dir, tbl_dir):
        os.makedirs(d, exist_ok=True)

    imgs = list_images(data_dir)
    if not imgs:
        print(f"[提示] 未在 {data_dir} 发现图片。请将待测图片放入该目录再运行。")
        return

    methods = ['huffman', 'shannon', 'sfe']

    rows_gray: List[Dict[str, float]] = []
    rows_rgb: List[Dict[str, float]] = []

    print("="*70)
    print(f"自动批处理开始：共 {len(imgs)} 张图片")
    print("="*70)

    for img_path in imgs:
        stem = os.path.splitext(os.path.basename(img_path))[0]

        # ---------- L 模式 ----------
        gray_save = os.path.join(out_dir, f"{stem}_gray.png")
        hist_prefix = os.path.join(hist_dir, stem)  # L 模式下只会输出 _L
        for i, m in enumerate(methods):
            export_hist = hist_prefix if i == 0 else None  # 只在第一种方法时输出直方图
            icomp = os.path.join(out_dir, f"{stem}_{m}_L.icomp")
            row = encode_image(img_path, icomp, method=m, mode='L',
                               save_gray=gray_save, export_hist_prefix=export_hist)
            recon = os.path.join(out_dir, f"{stem}_{m}_L_recon.png")
            img_recon = decode_image(icomp, recon)
            # PSNR（参考灰度图）
            ref = Image.open(gray_save).convert('L')
            psnr_db = compute_psnr(ref, img_recon)
            row['psnr_db'] = float('inf') if math.isinf(psnr_db) else float(psnr_db)
            rows_gray.append(row)

        # ---------- RGB 模式 ----------
        hist_prefix = os.path.join(hist_dir, stem)  # RGB 下会输出 _R/_G/_B
        for i, m in enumerate(methods):
            export_hist = hist_prefix if i == 0 else None
            icomp = os.path.join(out_dir, f"{stem}_{m}_RGB.icomp")
            row = encode_image(img_path, icomp, method=m, mode='RGB',
                               export_hist_prefix=export_hist)
            recon = os.path.join(out_dir, f"{stem}_{m}_RGB_recon.png")
            img_recon = decode_image(icomp, recon)
            # PSNR（参考 = 原图转 RGB）
            ref = Image.open(img_path).convert('RGB')
            psnr_db = compute_psnr(ref, img_recon)
            row['psnr_db'] = float('inf') if math.isinf(psnr_db) else float(psnr_db)
            rows_rgb.append(row)

    # ---------- 汇总表 ----------
    import csv
    gray_csv = os.path.join(tbl_dir, 'gray_summary.csv')
    rgb_csv = os.path.join(tbl_dir, 'rgb_summary.csv')

    if rows_gray:
        keys = sorted({k for r in rows_gray for k in r.keys()})
        with open(gray_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows_gray:
                w.writerow(r)
        print(f"[信息] 已写入：{gray_csv}")

    if rows_rgb:
        keys = sorted({k for r in rows_rgb for k in r.keys()})
        with open(rgb_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows_rgb:
                w.writerow(r)
        print(f"[信息] 已写入：{rgb_csv}")

    # ---------- 生成对比图（若可用） ----------
    if plt is not None and np is not None:
        try:
            plot_gray_figures(rows_gray, fig_dir)
            plot_rgb_figures(rows_rgb, fig_dir)
        except Exception as e:
            print(f"[警告] 绘图失败：{e}")
    else:
        print("[提示] 未安装 matplotlib 或 numpy，跳过绘图。")

    print("="*70)
    print("自动批处理结束。产物位于：")
    print(f" - 压缩文件/中间图/重建图：{out_dir}")
    print(f" - 直方图 CSV/PNG：{hist_dir}")
    print(f" - 汇总表：{tbl_dir}")
    print(f" - 结果图：{fig_dir}")
    print("="*70)


# ============================ 绘图（参考组员风格） ============================
# 将结果按 image 分组，便于同图不同方法横向比较。
def _group_by_image(rows: List[Dict[str, float]]):
    """将 rows 按 image 分组，返回 {image: {method: row}}"""
    by_img: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in rows:
        img = r['image']
        by_img.setdefault(img, {})[r['method']] = r
    return by_img


# 绘制灰度模式结果：熵 vs 平均码长、压缩率。
def plot_gray_figures(rows_gray: List[Dict[str, float]], fig_dir: str) -> None:
    if not rows_gray:
        return
    by_img = _group_by_image(rows_gray)
    images = sorted(by_img.keys())
    x = np.arange(len(images))
    width = 0.2

    # 1) H vs Lavg（灰度总量）
    H = [by_img[im]['huffman']['H_total'] for im in images]  # 任取一种方法的 H_total 均相同
    L_h = [by_img[im]['huffman']['Lavg_total'] for im in images]
    L_s = [by_img[im]['shannon']['Lavg_total'] for im in images]
    L_e = [by_img[im]['sfe']['Lavg_total'] for im in images]

    plt.figure(figsize=(10, 6))
    plt.bar(x - width*1.5, H, width, label='理论熵 H(X)', alpha=0.85)
    plt.bar(x - width*0.5, L_h, width, label='哈夫曼 L_avg', alpha=0.85)
    plt.bar(x + width*0.5, L_s, width, label='香农 L_avg', alpha=0.85)
    plt.bar(x + width*1.5, L_e, width, label='SFE L_avg', alpha=0.85)
    plt.xticks(x, images, rotation=20, ha='right')
    plt.ylabel('bits / 像素')
    plt.title('灰度图：理论熵与平均码长对比（每像素）')
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, 'exp_img_gray_avglen.png')
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"[图] {path}")

    # 2) 压缩率（有效比特 / 原始比特）
    R_h = [by_img[im]['huffman']['ratio_bits_over_raw'] for im in images]
    R_s = [by_img[im]['shannon']['ratio_bits_over_raw'] for im in images]
    R_e = [by_img[im]['sfe']['ratio_bits_over_raw'] for im in images]

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, R_h, width, label='Huffman ρ')
    plt.bar(x,         R_s, width, label='Shannon ρ')
    plt.bar(x + width, R_e, width, label='SFE ρ')
    plt.xticks(x, images, rotation=20, ha='right')
    plt.ylabel('压缩率 ρ')
    plt.title('灰度图：压缩率对比（越低越好）')
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = os.path.join(fig_dir, 'exp_img_gray_ratio.png')
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"[图] {path}")


# 绘制 RGB 模式结果：总量对比、压缩率、R/G/B 通道分图对比。
def plot_rgb_figures(rows_rgb: List[Dict[str, float]], fig_dir: str) -> None:
    if not rows_rgb:
        return
    by_img = _group_by_image(rows_rgb)
    images = sorted(by_img.keys())
    x = np.arange(len(images))
    width = 0.2

    # 1) H_total vs Lavg_total（RGB 总量）
    H = [by_img[im]['huffman']['H_total'] for im in images]  # 对同一图片，三方法 H_total 相同
    L_h = [by_img[im]['huffman']['Lavg_total'] for im in images]
    L_s = [by_img[im]['shannon']['Lavg_total'] for im in images]
    L_e = [by_img[im]['sfe']['Lavg_total'] for im in images]

    plt.figure(figsize=(10, 6))
    plt.bar(x - width*1.5, H, width, label='理论熵 H_sum', alpha=0.85)
    plt.bar(x - width*0.5, L_h, width, label='哈夫曼 L_sum', alpha=0.85)
    plt.bar(x + width*0.5, L_s, width, label='香农 L_sum', alpha=0.85)
    plt.bar(x + width*1.5, L_e, width, label='SFE L_sum', alpha=0.85)
    plt.xticks(x, images, rotation=20, ha='right')
    plt.ylabel('bits / 像素（R+G+B）')
    plt.title('RGB：理论熵与平均码长对比（按像素合计）')
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, 'exp_img_rgb_avglen.png')
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"[图] {path}")

    # 2) 压缩率（有效比特 / 原始比特，原始 = 24N）
    R_h = [by_img[im]['huffman']['ratio_bits_over_raw'] for im in images]
    R_s = [by_img[im]['shannon']['ratio_bits_over_raw'] for im in images]
    R_e = [by_img[im]['sfe']['ratio_bits_over_raw'] for im in images]

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, R_h, width, label='Huffman ρ')
    plt.bar(x,         R_s, width, label='Shannon ρ')
    plt.bar(x + width, R_e, width, label='SFE ρ')
    plt.xticks(x, images, rotation=20, ha='right')
    plt.ylabel('压缩率 ρ')
    plt.title('RGB：压缩率对比（越低越好）')
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = os.path.join(fig_dir, 'exp_img_rgb_ratio.png')
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
    print(f"[图] {path}")

    # 3) 每通道：H 与 L_avg（R/G/B 分别一张图）
    for ch in ['R','G','B']:
        Hc  = [by_img[im]['huffman'].get(f'H_{ch}', float('nan')) for im in images]
        Lhc = [by_img[im]['huffman'].get(f'Lavg_{ch}', float('nan')) for im in images]
        Lsc = [by_img[im]['shannon'].get(f'Lavg_{ch}', float('nan')) for im in images]
        Lec = [by_img[im]['sfe'].get(f'Lavg_{ch}', float('nan')) for im in images]
        plt.figure(figsize=(10,6))
        plt.bar(x - width*1.5, Hc, width, label=f'H_{ch}', alpha=0.85)
        plt.bar(x - width*0.5, Lhc, width, label=f'Huffman L_{ch}', alpha=0.85)
        plt.bar(x + width*0.5, Lsc, width, label=f'Shannon L_{ch}', alpha=0.85)
        plt.bar(x + width*1.5, Lec, width, label=f'SFE L_{ch}', alpha=0.85)
        plt.xticks(x, images, rotation=20, ha='right')
        plt.ylabel('bits / 像素')
        plt.title(f'RGB 通道 {ch}：理论熵与平均码长对比')
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = os.path.join(fig_dir, f'exp_img_rgb_{ch}_avglen.png')
        plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()
        print(f"[图] {path}")


# ============================ CLI & 入口 ============================
# 构建命令行接口：encode / decode / auto。
def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="图片压缩工具（L / RGB；三通道分别熵编码；直方图导出；PSNR 校验；自动批处理）")
    sub = parser.add_subparsers(dest='command', required=True)

    # 编码
    p_enc = sub.add_parser('encode', help='压缩图片')
    p_enc.add_argument('input', help='输入图片路径（RGB/L；会按 --mode 转换）')
    p_enc.add_argument('output', help='输出压缩文件路径（*.icomp）')
    p_enc.add_argument('--method', choices=['huffman', 'shannon', 'sfe'],
                       default='huffman', help='编码方法（默认 huffman）')
    p_enc.add_argument('--mode', choices=['L', 'RGB'], default='L',
                       help='图像模式：L=灰度；RGB=三通道分别熵编码')
    p_enc.add_argument('--save-gray', default=None,
                       help='若 --mode L，则另存灰度中间图路径（可选）')
    p_enc.add_argument('--export-hist', default=None,
                       help='导出直方图的前缀（将生成 *_L 或 *_R/_G/_B 的 csv/png）')

    # 解码
    p_dec = sub.add_parser('decode', help='解压图片')
    p_dec.add_argument('input', help='压缩文件路径（*.icomp）')
    p_dec.add_argument('output', help='输出图片路径（*.png/*.bmp 等）')

    # 自动（可显式调用，也支持默认无参时触发）
    p_auto = sub.add_parser('auto', help='自动批处理 ./data 目录并绘图')
    p_auto.add_argument('--data', default='data', help='数据目录（默认 ./data）')

    return parser


# 程序入口：无参数执行自动批处理；有参数按子命令执行。
def main(argv: Optional[List[str]] = None) -> None:
    # 若无参数，直接运行自动管线
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        auto_pipeline('data')
        return

    # 否则走命令行子命令
    parser = build_cli()
    args = parser.parse_args(argv)

    if args.command == 'encode':
        encode_image(args.input, args.output,
                     method=args.method,
                     mode=args.mode,
                     save_gray=args.save_gray,
                     export_hist_prefix=args.export_hist)
    elif args.command == 'decode':
        img = decode_image(args.input, args.output)
        print(f"[完成] 解压并保存：{args.output}  模式={img.mode} 尺寸={img.size}")
    elif args.command == 'auto':
        auto_pipeline(args.data)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
