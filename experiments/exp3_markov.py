"""
实验3：马尔可夫信源与熵率
比较零阶熵与熵率，验证利用相关性可以降低平均码长
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from encoder import HuffmanEncoder
from utils import calculate_entropy, get_frequency_distribution

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_markov_chain(p: float, q: float, size: int = 10000) -> str:
    """
    生成一阶马尔可夫链（二元信源）
    
    转移概率：
    P(0|0) = 1-p, P(1|0) = p
    P(0|1) = q, P(1|1) = 1-q
    
    Args:
        p: 从0转移到1的概率
        q: 从1转移到0的概率
        size: 序列长度
        
    Returns:
        生成的二进制字符串
    """
    sequence = ['0']  # 从0开始
    current = 0
    
    for _ in range(size - 1):
        if current == 0:
            # 当前为0
            next_bit = '1' if np.random.random() < p else '0'
        else:
            # 当前为1
            next_bit = '0' if np.random.random() < q else '1'
        
        sequence.append(next_bit)
        current = int(next_bit)
    
    return ''.join(sequence)


def calculate_markov_entropy_rate(p: float, q: float) -> float:
    """
    计算马尔可夫链的理论熵率
    
    H = π_0 * H(X_1|X_0=0) + π_1 * H(X_1|X_0=1)
    
    其中 π_0, π_1 是平稳分布
    """
    # 计算平稳分布
    # π_0 = q/(p+q), π_1 = p/(p+q)
    if p + q == 0:
        return 0.0
    
    pi_0 = q / (p + q)
    pi_1 = p / (p + q)
    
    # 条件熵
    if p == 0 or p == 1:
        H_0 = 0
    else:
        H_0 = -p * np.log2(p) - (1-p) * np.log2(1-p)
    
    if q == 0 or q == 1:
        H_1 = 0
    else:
        H_1 = -q * np.log2(q) - (1-q) * np.log2(1-q)
    
    # 熵率
    H_rate = pi_0 * H_0 + pi_1 * H_1
    return H_rate


def experiment_3():
    """运行实验3"""
    print("=" * 60)
    print("实验3：马尔可夫信源与熵率")
    print("=" * 60)
    
    # 测试不同的转移概率
    # 注意：p=q=0.1时，状态很难跳转，相关性高；p=q=0.5时，状态容易跳转，相关性低
    test_cases = [
        (0.1, 0.1, "高相关性"),  # p=q=0.1，状态难跳转，相关性高
        (0.3, 0.3, "中等相关性"),
        (0.5, 0.5, "低相关性"),  # p=q=0.5，状态容易跳转，相关性低
        (0.7, 0.2, "非对称"),
    ]
    
    results = []
    
    for p, q, name in test_cases:
        print(f"\n测试案例: {name} (p={p}, q={q})")
        
        # 生成马尔可夫序列
        text = generate_markov_chain(p, q, size=10000)
        
        # 计算零阶熵（单符号分布）
        _, prob_single = get_frequency_distribution(text)
        H_0 = calculate_entropy(prob_single)
        
        # 计算理论熵率
        H_rate_theory = calculate_markov_entropy_rate(p, q)
        
        # 方法1：单符号哈夫曼编码
        encoder_single = HuffmanEncoder(prob_single)
        L_single = encoder_single.get_average_code_length()
        
        # 方法2：2-bit块哈夫曼编码（利用相关性）
        # 将序列分成2-bit块（不重叠，确保是偶数长度）
        # 注意：要确保pairs是完整的2-bit块，所以长度应该是偶数
        text_for_pairs = text[:len(text)//2*2]  # 确保长度是偶数
        pairs = [text_for_pairs[i:i+2] for i in range(0, len(text_for_pairs), 2)]
        
        # 统计每个pair的频率（作为独立的"符号"）
        from collections import Counter
        pair_counter = Counter(pairs)
        total_pairs = len(pairs)
        prob_pairs = {pair: count / total_pairs for pair, count in pair_counter.items()}
        
        # 打印调试信息（用于验证）
        if p == 0.1 and q == 0.1:
            print(f"\n  调试信息 (p=q=0.1):")
            print(f"    总块数: {total_pairs}")
            print(f"    块的概率分布:")
            for pair in ['00', '01', '10', '11']:
                if pair in prob_pairs:
                    print(f"      P({pair}) = {prob_pairs[pair]:.4f}")
            # 计算块的熵
            H_pair = calculate_entropy(prob_pairs)
            print(f"    块的熵 H(pair) = {H_pair:.4f} bits/块")
            print(f"    理论下界 (每符号) = {H_pair/2:.4f} bits/符号")
        
        encoder_pairs = HuffmanEncoder(prob_pairs)
        L_pairs_per_block = encoder_pairs.get_average_code_length()  # 每块的平均码长
        L_pairs = L_pairs_per_block / 2  # 归一化到每符号
        
        # 打印更多调试信息
        if p == 0.1 and q == 0.1:
            print(f"    每块平均码长 L_pair = {L_pairs_per_block:.4f} bits/块")
            print(f"    每符号平均码长 L_per_symbol = {L_pairs:.4f} bits/符号")
        
        results.append({
            'name': name,
            'p': p,
            'q': q,
            'H_0': H_0,
            'H_rate': H_rate_theory,
            'L_single': L_single,
            'L_pairs': L_pairs
        })
        
        print(f"  零阶熵 H_0: {H_0:.4f}")
        print(f"  理论熵率 H: {H_rate_theory:.4f}")
        print(f"  单符号编码平均码长: {L_single:.4f}")
        print(f"  2-bit块编码平均码长: {L_pairs:.4f}")
        print(f"  熵率与零阶熵差值: {H_0 - H_rate_theory:.4f}")
    
    # 绘制结果
    plot_results(results)
    
    return results


def plot_results(results):
    """绘制实验结果"""
    names = [r['name'] for r in results]
    H_0 = [r['H_0'] for r in results]
    H_rate = [r['H_rate'] for r in results]
    L_single = [r['L_single'] for r in results]
    L_pairs = [r['L_pairs'] for r in results]
    
    x = np.arange(len(names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width * 1.5, H_0, width, label='零阶熵 H_0', alpha=0.8)
    ax.bar(x - width * 0.5, H_rate, width, label='理论熵率 H', alpha=0.8)
    ax.bar(x + width * 0.5, L_single, width, label='单符号编码 L_avg', alpha=0.8)
    ax.bar(x + width * 1.5, L_pairs, width, label='2-bit块编码 L_avg', alpha=0.8)
    
    ax.set_xlabel('测试案例', fontsize=12)
    ax.set_ylabel('熵/平均码长 (bits/符号)', fontsize=12)
    ax.set_title('实验3：零阶熵 vs 熵率 vs 编码性能', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('../results/figures', exist_ok=True)
    plt.savefig('../results/figures/exp3_markov.png', dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: results/figures/exp3_markov.png")
    
    plt.show()


if __name__ == '__main__':
    experiment_3()

