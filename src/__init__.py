"""
信息论压缩软件包
"""
from .encoder import HuffmanEncoder, ShannonEncoder, ShannonFanoEliasEncoder
from .utils import calculate_entropy, get_frequency_distribution, average_code_length

__version__ = '1.0.0'

