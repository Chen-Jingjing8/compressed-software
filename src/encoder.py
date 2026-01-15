"""
编码器实现：哈夫曼编码、香农编码、Shannon-Fano-Elias编码
"""
import heapq
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np
try:
    from .utils import calculate_entropy, average_code_length
except ImportError:
    from utils import calculate_entropy, average_code_length


class BaseEncoder:
    """编码器基类"""
    
    def __init__(self, probabilities: Dict[str, float]):
        """
        初始化编码器
        
        Args:
            probabilities: 符号概率分布字典
        """
        self.probabilities = probabilities
        self.codes: Dict[str, str] = {}
        self.entropy = calculate_entropy(probabilities)
        
    def encode(self, text: str) -> str:
        """
        编码文本
        
        Args:
            text: 输入文本
            
        Returns:
            编码后的二进制字符串
        """
        return ''.join(self.codes.get(char, '') for char in text)
    
    def decode(self, encoded: str) -> str:
        """
        解码二进制字符串
        
        Args:
            encoded: 编码后的二进制字符串
            
        Returns:
            解码后的文本
        """
        # 需要反向查找，构建反向字典
        reverse_codes = {code: symbol for symbol, code in self.codes.items()}
        
        result = []
        current = ''
        for bit in encoded:
            current += bit
            if current in reverse_codes:
                result.append(reverse_codes[current])
                current = ''
        
        return ''.join(result)
    
    def get_average_code_length(self) -> float:
        """获取平均码长"""
        return average_code_length(self.codes, self.probabilities)
    
    def get_efficiency(self) -> float:
        """获取编码效率 = H(X) / L_avg"""
        avg_len = self.get_average_code_length()
        if avg_len == 0:
            return 0.0
        return self.entropy / avg_len


class HuffmanEncoder(BaseEncoder):
    """哈夫曼编码器"""
    
    class Node:
        """哈夫曼树节点"""
        def __init__(self, symbol: Optional[str], prob: float):
            self.symbol = symbol
            self.prob = prob
            self.left = None
            self.right = None
        
        def __lt__(self, other):
            return self.prob < other.prob
    
    def __init__(self, probabilities: Dict[str, float]):
        super().__init__(probabilities)
        self._build_codes()
    
    def _build_codes(self):
        """构建哈夫曼编码"""
        if len(self.probabilities) == 0:
            return
        
        if len(self.probabilities) == 1:
            # 只有一个符号，编码为'0'
            symbol = list(self.probabilities.keys())[0]
            self.codes[symbol] = '0'
            return
        
        # 构建优先队列
        heap = []
        for symbol, prob in self.probabilities.items():
            node = self.Node(symbol, prob)
            heapq.heappush(heap, node)
        
        # 构建哈夫曼树
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = self.Node(None, left.prob + right.prob)
            merged.left = left
            merged.right = right
            heapq.heappush(heap, merged)
        
        root = heap[0]
        self._assign_codes(root, '')
    
    def _assign_codes(self, node: Node, code: str):
        """递归分配编码"""
        if node.symbol is not None:
            self.codes[node.symbol] = code if code else '0'
        else:
            if node.left:
                self._assign_codes(node.left, code + '0')
            if node.right:
                self._assign_codes(node.right, code + '1')


class ShannonEncoder(BaseEncoder):
    """香农编码器"""
    
    def __init__(self, probabilities: Dict[str, float]):
        super().__init__(probabilities)
        self._build_codes()
    
    def _build_codes(self):
        """构建香农编码"""
        # 按概率降序排序
        sorted_symbols = sorted(self.probabilities.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # 计算累积分布函数
        cumulative = 0.0
        for symbol, prob in sorted_symbols:
            # 码长 l_i = ceil(-log2(p_i))
            if prob > 0:
                code_length = int(np.ceil(-np.log2(prob)))
            else:
                code_length = 1
            
            # 将累积概率转换为二进制，取前code_length位
            binary = self._decimal_to_binary(cumulative, code_length)
            self.codes[symbol] = binary
            
            cumulative += prob
    
    def _decimal_to_binary(self, decimal: float, length: int) -> str:
        """将小数转换为二进制字符串"""
        result = ''
        value = decimal
        for _ in range(length):
            value *= 2
            if value >= 1:
                result += '1'
                value -= 1
            else:
                result += '0'
        return result


class ShannonFanoEliasEncoder(BaseEncoder):
    """Shannon-Fano-Elias编码器"""
    
    def __init__(self, probabilities: Dict[str, float]):
        super().__init__(probabilities)
        self._build_codes()
    
    def _build_codes(self):
        """构建Shannon-Fano-Elias编码"""
        # 按符号顺序排序（保持一致性）
        sorted_symbols = sorted(self.probabilities.items())
        
        # 计算累积分布函数
        cumulative = 0.0
        for symbol, prob in sorted_symbols:
            # F_bar(x) = F(x) + p(x)/2
            F_bar = cumulative + prob / 2.0
            
            # 码长 l(x) = ceil(-log2(p(x))) + 1
            if prob > 0:
                code_length = int(np.ceil(-np.log2(prob))) + 1
            else:
                code_length = 2
            
            # 将F_bar转换为二进制
            binary = self._decimal_to_binary(F_bar, code_length)
            self.codes[symbol] = binary
            
            cumulative += prob
    
    def _decimal_to_binary(self, decimal: float, length: int) -> str:
        """将小数转换为二进制字符串"""
        result = ''
        value = decimal
        for _ in range(length):
            value *= 2
            if value >= 1:
                result += '1'
                value -= 1
            else:
                result += '0'
        return result

