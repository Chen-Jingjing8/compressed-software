"""
实验2：真实文本数据的压缩效果
测试不同编码方法在真实文本上的表现
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from encoder import HuffmanEncoder, ShannonEncoder, ShannonFanoEliasEncoder
from utils import get_frequency_distribution, calculate_entropy

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_sample_files():
    """创建示例文本文件"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'samples')
    os.makedirs(data_dir, exist_ok=True)
    
    # 英文文本示例
    english_text = """
    The quick brown fox jumps over the lazy dog. This is a classic pangram that contains every letter of the English alphabet at least once. 
    Information theory is a branch of mathematics that deals with the quantification, storage, and communication of information.
    Data compression is the process of encoding information using fewer bits than the original representation.
    """
    
    # 中文文本示例
    chinese_text = """
    信息论是应用数学的一个分支，主要研究信息的量化、存储和通信。香农在1948年发表了《通信的数学理论》，
    奠定了信息论的基础。信源编码定理表明，对于任意离散无记忆信源，存在编码方法使得平均码长可以任意接近熵。
    哈夫曼编码是一种最优前缀码，在平均码长意义下达到最优。
    """
    
    # 代码示例
    code_text = """
    def huffman_encode(text):
        frequencies = count_frequencies(text)
        tree = build_huffman_tree(frequencies)
        codes = generate_codes(tree)
        return encode_text(text, codes)
    
    class Node:
        def __init__(self, symbol, frequency):
            self.symbol = symbol
            self.frequency = frequency
            self.left = None
            self.right = None
    """
    
    files = {
        'english.txt': english_text,
        'chinese.txt': chinese_text,
        'code.txt': code_text
    }
    
    for filename, content in files.items():
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    return files


def experiment_2():
    """运行实验2"""
    print("=" * 60)
    print("实验2：真实文本数据的压缩效果")
    print("=" * 60)
    
    # 创建示例文件
    sample_files = create_sample_files()
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'samples')
    
    results = []
    
    for filename, _ in sample_files.items():
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"警告：文件不存在 {filepath}")
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if len(text) == 0:
            continue
        
        print(f"\n处理文件: {filename}")
        print(f"  文件大小: {len(text)} 字符")
        
        # 统计分布
        _, probabilities = get_frequency_distribution(text)
        entropy = calculate_entropy(probabilities)
        
        # 测试编码
        huffman = HuffmanEncoder(probabilities)
        shannon = ShannonEncoder(probabilities)
        sfe = ShannonFanoEliasEncoder(probabilities)
        
        huffman_avg = huffman.get_average_code_length()
        shannon_avg = shannon.get_average_code_length()
        sfe_avg = sfe.get_average_code_length()
        
        # 计算压缩率（相对于8位/字符）
        original_bits_per_char = 8
        huffman_ratio = huffman_avg / original_bits_per_char
        shannon_ratio = shannon_avg / original_bits_per_char
        sfe_ratio = sfe_avg / original_bits_per_char
        
        results.append({
            'file': filename.replace('.txt', ''),
            'entropy': entropy,
            'huffman': huffman_avg,
            'shannon': shannon_avg,
            'sfe': sfe_avg,
            'huffman_ratio': huffman_ratio,
            'shannon_ratio': shannon_ratio,
            'sfe_ratio': sfe_ratio
        })
        
        print(f"  理论熵: {entropy:.4f} bits/符号")
        print(f"  哈夫曼平均码长: {huffman_avg:.4f} bits/符号 (压缩率: {huffman_ratio:.4f})")
        print(f"  香农平均码长: {shannon_avg:.4f} bits/符号 (压缩率: {shannon_ratio:.4f})")
        print(f"  SFE平均码长: {sfe_avg:.4f} bits/符号 (压缩率: {sfe_ratio:.4f})")
    
    # 绘制结果
    plot_results(results)
    
    return results


def plot_results(results):
    """绘制实验结果"""
    files = [r['file'] for r in results]
    entropy = [r['entropy'] for r in results]
    huffman = [r['huffman'] for r in results]
    shannon = [r['shannon'] for r in results]
    sfe = [r['sfe'] for r in results]
    
    # 图1：平均码长比较
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(files))
    width = 0.2
    
    ax1.bar(x - width * 1.5, entropy, width, label='理论熵 H(X)', alpha=0.8)
    ax1.bar(x - width * 0.5, huffman, width, label='哈夫曼码', alpha=0.8)
    ax1.bar(x + width * 0.5, shannon, width, label='香农码', alpha=0.8)
    ax1.bar(x + width * 1.5, sfe, width, label='SFE码', alpha=0.8)
    
    ax1.set_xlabel('文件类型', fontsize=12)
    ax1.set_ylabel('平均码长 (bits/符号)', fontsize=12)
    ax1.set_title('不同编码方法的平均码长', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(files)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2：熵 vs 平均码长散点图
    ax2.scatter(entropy, huffman, label='哈夫曼码', s=100, alpha=0.7)
    ax2.scatter(entropy, shannon, label='香农码', s=100, alpha=0.7)
    ax2.scatter(entropy, sfe, label='SFE码', s=100, alpha=0.7)
    
    # 绘制理想线 y = x 和 y = x + 1
    min_entropy = min(entropy)
    max_entropy = max(entropy)
    x_line = np.linspace(min_entropy, max_entropy, 100)
    ax2.plot(x_line, x_line, 'k--', label='理想线 H(X)', linewidth=2)
    ax2.plot(x_line, x_line + 1, 'r--', label='上界 H(X)+1', linewidth=2, alpha=0.5)
    
    ax2.set_xlabel('理论熵 H(X)', fontsize=12)
    ax2.set_ylabel('平均码长 L_avg', fontsize=12)
    ax2.set_title('熵 vs 平均码长', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('../results/figures', exist_ok=True)
    plt.savefig('../results/figures/exp2_real_text.png', dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: results/figures/exp2_real_text.png")
    
    plt.show()


if __name__ == '__main__':
    experiment_2()

