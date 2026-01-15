"""
工具函数：熵计算、频率统计等
"""
import numpy as np
from collections import Counter
from typing import Dict, Tuple


def calculate_entropy(probabilities: Dict[str, float]) -> float:
    """
    计算离散信源的熵
    
    Args:
        probabilities: 符号概率分布字典
        
    Returns:
        熵值 H(X) = -Σ p(x) * log2(p(x))
    """
    entropy = 0.0
    for prob in probabilities.values():
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return entropy


def calculate_entropy_rate(text: str, order: int = 1) -> float:
    """
    计算文本的熵率（基于n-gram）
    
    Args:
        text: 输入文本
        order: n-gram的阶数（1表示单字符，2表示字符对等）
        
    Returns:
        熵率估计值
    """
    if order == 1:
        # 零阶熵
        counter = Counter(text)
        total = len(text)
        probs = {char: count / total for char, count in counter.items()}
        return calculate_entropy(probs)
    else:
        # n-gram熵率
        ngrams = [text[i:i+order] for i in range(len(text) - order + 1)]
        counter = Counter(ngrams)
        total = len(ngrams)
        probs = {ngram: count / total for ngram, count in counter.items()}
        return calculate_entropy(probs) / order


def get_frequency_distribution(text: str) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    统计文本中字符的频率分布
    
    Args:
        text: 输入文本
        
    Returns:
        (频率字典, 概率字典)
    """
    counter = Counter(text)
    total = len(text)
    frequencies = dict(counter)
    probabilities = {char: count / total for char, count in counter.items()}
    return frequencies, probabilities


def average_code_length(codes: Dict[str, str], probabilities: Dict[str, float]) -> float:
    """
    计算平均码长
    
    Args:
        codes: 编码字典 {符号: 码字}
        probabilities: 概率分布字典
        
    Returns:
        平均码长 L_avg = Σ p(x) * l(x)
    """
    avg_length = 0.0
    for symbol, prob in probabilities.items():
        if symbol in codes:
            avg_length += prob * len(codes[symbol])
    return avg_length


def compression_ratio(original_size: int, compressed_size: int) -> float:
    """
    计算压缩率
    
    Args:
        original_size: 原始文件大小（字节）
        compressed_size: 压缩后大小（字节）
        
    Returns:
        压缩率 = compressed_size / original_size
    """
    if original_size == 0:
        return 0.0
    return compressed_size / original_size

