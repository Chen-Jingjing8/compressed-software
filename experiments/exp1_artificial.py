"""
实验1：人工信源验证信源编码定理
构造不同概率分布，验证平均码长与熵的关系
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from encoder import HuffmanEncoder, ShannonEncoder, ShannonFanoEliasEncoder
from utils import calculate_entropy, get_frequency_distribution

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_artificial_source(distribution_type: str, num_symbols: int = 10, size: int = 10000) -> str:
    """
    生成人工信源数据
    
    Args:
        distribution_type: 分布类型 ('uniform', 'skewed', 'moderate')
        num_symbols: 符号数量
        size: 生成序列长度
        
    Returns:
        生成的文本字符串
    """
    symbols = [chr(ord('A') + i) for i in range(num_symbols)]
    
    if distribution_type == 'uniform':
        # 均匀分布（熵最大）
        probs = np.ones(num_symbols) / num_symbols
    elif distribution_type == 'skewed':
        # 极度偏斜分布
        probs = np.zeros(num_symbols)
        probs[0] = 0.9
        probs[1:] = 0.1 / (num_symbols - 1)
    elif distribution_type == 'moderate':
        # 中等偏斜分布（几何分布）
        p = 0.3
        probs = np.array([p * (1-p)**i for i in range(num_symbols)])
        probs = probs / probs.sum()
    else:
        raise ValueError(f"未知分布类型: {distribution_type}")
    
    # 生成序列
    sequence = np.random.choice(symbols, size=size, p=probs)
    return ''.join(sequence)


def experiment_1():
    """运行实验1"""
    print("=" * 60)
    print("实验1：人工信源验证信源编码定理")
    print("=" * 60)
    
    distribution_types = ['uniform', 'moderate', 'skewed']
    results = []
    
    for dist_type in distribution_types:
        print(f"\n测试分布类型: {dist_type}")
        
        # 生成数据
        text = generate_artificial_source(dist_type, num_symbols=10, size=10000)
        _, probabilities = get_frequency_distribution(text)
        
        # 计算理论熵
        entropy = calculate_entropy(probabilities)
        
        # 测试不同编码方法
        huffman = HuffmanEncoder(probabilities)
        shannon = ShannonEncoder(probabilities)
        sfe = ShannonFanoEliasEncoder(probabilities)
        
        huffman_avg = huffman.get_average_code_length()
        shannon_avg = shannon.get_average_code_length()
        sfe_avg = sfe.get_average_code_length()
        
        results.append({
            'type': dist_type,
            'entropy': entropy,
            'huffman': huffman_avg,
            'shannon': shannon_avg,
            'sfe': sfe_avg
        })
        
        print(f"  理论熵 H(X): {entropy:.4f}")
        print(f"  哈夫曼平均码长: {huffman_avg:.4f}")
        print(f"  香农平均码长: {shannon_avg:.4f}")
        print(f"  SFE平均码长: {sfe_avg:.4f}")
        print(f"  哈夫曼效率: {huffman.get_efficiency():.4f}")
    
    # 绘制结果
    plot_results(results)
    
    return results


def plot_results(results):
    """绘制实验结果"""
    types = [r['type'] for r in results]
    entropy = [r['entropy'] for r in results]
    huffman = [r['huffman'] for r in results]
    shannon = [r['shannon'] for r in results]
    sfe = [r['sfe'] for r in results]
    
    x = np.arange(len(types))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width * 1.5, entropy, width, label='理论熵 H(X)', alpha=0.8)
    ax.bar(x - width * 0.5, huffman, width, label='哈夫曼码', alpha=0.8)
    ax.bar(x + width * 0.5, shannon, width, label='香农码', alpha=0.8)
    ax.bar(x + width * 1.5, sfe, width, label='SFE码', alpha=0.8)
    
    ax.set_xlabel('分布类型', fontsize=12)
    ax.set_ylabel('平均码长 (bits/符号)', fontsize=12)
    ax.set_title('实验1：不同编码方法的平均码长比较', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['均匀分布', '中等偏斜', '极度偏斜'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('../results/figures', exist_ok=True)
    plt.savefig('../results/figures/exp1_artificial.png', dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: results/figures/exp1_artificial.png")
    
    plt.show()


if __name__ == '__main__':
    experiment_1()

