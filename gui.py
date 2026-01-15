"""
基于PySide6和qfluentwidgets的压缩软件GUI界面
"""
import sys
import os
import json
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QTabWidget
from PySide6.QtCore import Qt, QThread, Signal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib中文字体为五号宋体
# 五号字体 = 10.5pt ≈ 14px
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', '宋体', 'SimHei', 'Arial Unicode MS']
plt.rcParams['font.size'] = 10.5  # 五号字体大小
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入qfluentwidgets，如果失败则使用标准组件
try:
    from qfluentwidgets import (
        CardWidget, PushButton, ComboBox, LineEdit, TextEdit, 
        ProgressBar, TitleLabel, BodyLabel, FluentIcon, 
        PrimaryPushButton, InfoBar, ScrollArea, setTheme, Theme
    )
    QFLUENT_AVAILABLE = True
except ImportError:
    # 如果qfluentwidgets不可用，使用标准PySide6组件
    from PySide6.QtWidgets import (
        QPushButton, QComboBox, QLineEdit, QTextEdit,
        QProgressBar, QLabel, QScrollArea, QFrame
    )
    from PySide6.QtGui import QIcon
    QFLUENT_AVAILABLE = False
    print("警告: qfluentwidgets未安装，将使用标准PySide6组件")
    print("建议安装: pip install qfluentwidgets")
    
    # 创建兼容性包装类
    class CardWidget(QFrame):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setStyleSheet("""
                QFrame {
                    background-color: white;
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)
    
    class PushButton(QPushButton):
        pass
    
    class PrimaryPushButton(QPushButton):
        def __init__(self, text, parent=None):
            super().__init__(text, parent)
            self.setStyleSheet("""
                QPushButton {
                    background-color: #0078d4;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #106ebe;
                }
                QPushButton:pressed {
                    background-color: #005a9e;
                }
            """)
    
    class ComboBox(QComboBox):
        pass
    
    class LineEdit(QLineEdit):
        pass
    
    class TextEdit(QTextEdit):
        pass
    
    class ProgressBar(QProgressBar):
        pass
    
    class TitleLabel(QLabel):
        def __init__(self, text, parent=None):
            super().__init__(text, parent)
            font = self.font()
            font.setPointSize(16)
            font.setBold(True)
            self.setFont(font)
    
    class BodyLabel(QLabel):
        pass
    
    class FluentIcon:
        """兼容性图标类，所有属性都返回None"""
        def __getattr__(self, name):
            return None
    
    class InfoBar:
        @staticmethod
        def success(title, content, parent=None, duration=3000):
            from PySide6.QtWidgets import QMessageBox
            msg = QMessageBox(parent)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle(title)
            msg.setText(content)
            msg.exec()
        
        @staticmethod
        def error(title, content, parent=None, duration=3000):
            from PySide6.QtWidgets import QMessageBox
            msg = QMessageBox(parent)
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle(title)
            msg.setText(content)
            msg.exec()
        
        @staticmethod
        def warning(title, content, parent=None, duration=3000):
            from PySide6.QtWidgets import QMessageBox
            msg = QMessageBox(parent)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle(title)
            msg.setText(content)
            msg.exec()
    
    class ScrollArea(QScrollArea):
        pass
    
    def setTheme(theme):
        pass
    
    class Theme:
        AUTO = "auto"
        LIGHT = "light"
        DARK = "dark"

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from encoder import HuffmanEncoder, ShannonEncoder, ShannonFanoEliasEncoder
from utils import get_frequency_distribution, compression_ratio


def bits_to_bytes(bits: str) -> bytes:
    """将二进制字符串转换为字节"""
    padding = (8 - len(bits) % 8) % 8
    bits += '0' * padding
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = int(bits[i:i+8], 2)
        result.append(byte)
    return bytes(result)


def bytes_to_bits(data: bytes) -> str:
    """将字节转换为二进制字符串"""
    bits = ''
    for byte in data:
        bits += format(byte, '08b')
    return bits


class CompressionThread(QThread):
    """压缩/解压工作线程"""
    finished = Signal(dict)  # 传递结果字典
    error = Signal(str)  # 传递错误信息
    progress = Signal(int)  # 进度信号
    
    def __init__(self, operation: str, input_path: str, output_path: str = None, method: str = 'huffman'):
        super().__init__()
        self.operation = operation  # 'encode' 或 'decode'
        self.input_path = input_path
        self.output_path = output_path
        self.method = method
    
    def run(self):
        try:
            if self.operation == 'encode':
                self._encode()
            elif self.operation == 'decode':
                self._decode()
        except Exception as e:
            self.error.emit(str(e))
    
    def _encode(self):
        """压缩文件"""
        # 读取文件
        with open(self.input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if len(text) == 0:
            self.error.emit("错误：文件为空")
            return
        
        self.progress.emit(20)
        
        # 统计频率分布
        frequencies, probabilities = get_frequency_distribution(text)
        self.progress.emit(40)
        
        # 选择编码器
        if self.method == 'huffman':
            encoder = HuffmanEncoder(probabilities)
        elif self.method == 'shannon':
            encoder = ShannonEncoder(probabilities)
        elif self.method == 'sfe':
            encoder = ShannonFanoEliasEncoder(probabilities)
        else:
            self.error.emit(f"不支持的编码方法: {self.method}")
            return
        
        self.progress.emit(60)
        
        # 编码文本
        encoded_bits = encoder.encode(text)
        self.progress.emit(80)
        
        # 保存压缩文件
        # 只保存必需信息：codes（解码必需）和original_size（验证用）
        # method可选保存，用于显示；统计信息可以重新计算，不保存以减小文件
        metadata = {
            'codes': encoder.codes,  # 必需：解码时必须
            'original_size': len(text),  # 推荐：用于验证解码结果
            'method': self.method  # 可选：用于显示编码方法
        }
        
        encoded_bytes = bits_to_bytes(encoded_bits)
        
        with open(self.output_path, 'wb') as f:
            metadata_json = json.dumps(metadata, ensure_ascii=False)
            metadata_bytes = metadata_json.encode('utf-8')
            f.write(len(metadata_bytes).to_bytes(4, 'big'))
            f.write(metadata_bytes)
            f.write(encoded_bytes)
        
        self.progress.emit(100)
        
        # 返回结果
        original_size_bits = len(text) * 8
        compressed_size_bits = len(encoded_bits)
        compressed_size_bytes = os.path.getsize(self.output_path)
        
        result = {
            'success': True,
            'method': self.method,
            'original_size': len(text),
            'original_size_bits': original_size_bits,
            'compressed_size_bytes': compressed_size_bytes,
            'compressed_size_bits': compressed_size_bits,
            'compression_ratio': compression_ratio(original_size_bits, compressed_size_bits),
            'entropy': encoder.entropy,
            'avg_code_length': encoder.get_average_code_length(),
            'efficiency': encoder.get_efficiency()
        }
        
        self.finished.emit(result)
    
    def _decode(self):
        """解压文件"""
        with open(self.input_path, 'rb') as f:
            metadata_len = int.from_bytes(f.read(4), 'big')
            metadata_json = f.read(metadata_len).decode('utf-8')
            metadata = json.loads(metadata_json)
            encoded_bytes = f.read()
        
        self.progress.emit(50)
        
        encoded_bits = bytes_to_bits(encoded_bytes)
        codes = metadata['codes']
        reverse_codes = {code: symbol for symbol, code in codes.items()}
        
        result = []
        current = ''
        for bit in encoded_bits:
            current += bit
            if current in reverse_codes:
                result.append(reverse_codes[current])
                current = ''
        
        decoded_text = ''.join(result)
        self.progress.emit(80)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(decoded_text)
        
        self.progress.emit(100)
        
        result_dict = {
            'success': True,
            'original_size': metadata.get('original_size', len(decoded_text)),
            'decoded_size': len(decoded_text),
            'method': metadata.get('method', 'unknown')
        }
        
        self.finished.emit(result_dict)

# ImageCompressThread - 插入到 gui.py 中与其它 QThread 定义相邻处（紧跟 CompressionThread 之后）
import importlib.util
from pathlib import Path
import os
from PySide6.QtCore import QThread, Signal

class ImageCompressThread(QThread):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, input_path: str, output_path: str, method: str = 'huffman', mode: str = 'L', save_gray: str = None):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.method = method
        self.mode = mode
        self.save_gray = save_gray

    def _load_image_compress(self):
        try:
            import image_compress as imgcomp
            return imgcomp
        except Exception:
            # 尝试多个可能的位置
            possible_paths = [
                Path(__file__).parent / 'image_compress.py',  # 主目录
                Path(__file__).parent / 'experiments' / 'image_compress.py',  # experiments目录
                Path(__file__).parent.parent / 'image_compress.py',  # 父目录
            ]
            
            for p in possible_paths:
                if p.exists():
                    spec = importlib.util.spec_from_file_location('image_compress', str(p))
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
            
            raise ImportError("无法找到 image_compress.py 模块，请确保它在项目根目录或experiments目录中。")
            spec = importlib.util.spec_from_file_location('image_compress', str(p))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    def run(self):
        try:
            imgcomp = self._load_image_compress()
            out_path = self.output_path
            if os.path.isdir(out_path):
                stem = Path(self.input_path).stem
                out_path = os.path.join(out_path, f"{stem}_{self.method}_{self.mode}.icomp")
            if not out_path.lower().endswith('.icomp'):
                out_path = out_path + '.icomp'
            row = imgcomp.encode_image(self.input_path, out_path,
                                       method=self.method, mode=self.mode,
                                       save_gray=self.save_gray,
                                       export_hist_prefix=None)
            result = {
                'type': 'image_compress',
                'method': self.method,
                'mode': self.mode,
                'input': self.input_path,
                'output': out_path,
                'stats': row
            }
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

import importlib.util
from pathlib import Path
import os
from PySide6.QtCore import QThread, Signal

class ImageCompressThread(QThread):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, input_path: str, output_path: str, method: str = 'huffman', mode: str = 'L', save_gray: str = None):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.method = method
        self.mode = mode
        self.save_gray = save_gray

    def _load_image_compress(self):
        try:
            import image_compress as imgcomp
            return imgcomp
        except Exception:
            # 尝试多个可能的位置
            possible_paths = [
                Path(__file__).parent / 'image_compress.py',  # 主目录
                Path(__file__).parent / 'experiments' / 'image_compress.py',  # experiments目录
                Path(__file__).parent.parent / 'image_compress.py',  # 父目录
            ]
            
            for p in possible_paths:
                if p.exists():
                    spec = importlib.util.spec_from_file_location('image_compress', str(p))
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
            
            raise ImportError("无法找到 image_compress.py 模块，请确保它在项目根目录或experiments目录中。")
            spec = importlib.util.spec_from_file_location('image_compress', str(p))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    def run(self):
        try:
            imgcomp = self._load_image_compress()
            out_path = self.output_path
            if os.path.isdir(out_path):
                stem = Path(self.input_path).stem
                out_path = os.path.join(out_path, f"{stem}_{self.method}_{self.mode}.icomp")
            if not out_path.lower().endswith('.icomp'):
                out_path = out_path + '.icomp'
            row = imgcomp.encode_image(self.input_path, out_path,
                                       method=self.method, mode=self.mode,
                                       save_gray=self.save_gray,
                                       export_hist_prefix=None)
            result = {
                'type': 'image_compress',
                'method': self.method,
                'mode': self.mode,
                'input': self.input_path,
                'output': out_path,
                'stats': row
            }
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# ImageDecompressThread - 插入到 gui.py 中与其它 QThread 定义相邻处（紧跟 ImageCompressThread 之后）
import importlib.util
from pathlib import Path
import os
from PySide6.QtCore import QThread, Signal

class ImageDecompressThread(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, input_path: str, output_path: str):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path

    def _load_image_compress(self):
        try:
            import image_compress as imgcomp
            return imgcomp
        except Exception:
            # 尝试多个可能的位置
            possible_paths = [
                Path(__file__).parent / 'image_compress.py',  # 主目录
                Path(__file__).parent / 'experiments' / 'image_compress.py',  # experiments目录
                Path(__file__).parent.parent / 'image_compress.py',  # 父目录
            ]
            
            for p in possible_paths:
                if p.exists():
                    spec = importlib.util.spec_from_file_location('image_compress', str(p))
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
            
            raise ImportError("无法找到 image_compress.py 模块，请确保它在项目根目录或experiments目录中。")
            spec = importlib.util.spec_from_file_location('image_compress', str(p))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    def run(self):
        try:
            imgcomp = self._load_image_compress()
            out_path = self.output_path
            if os.path.isdir(out_path):
                stem = Path(self.input_path).stem
                out_path = os.path.join(out_path, f"{stem}_recon.png")
            img = imgcomp.decode_image(self.input_path, out_path)
            result = {
                'type': 'image_decompress',
                'input': self.input_path,
                'output': out_path,
                'mode': img.mode,
                'size': img.size
            }
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class CompressionCard(CardWidget):
    """压缩功能卡片"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()
        self.thread: Optional[CompressionThread] = None
    
    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title = TitleLabel("文件压缩", self)
        layout.addWidget(title)
        
        # 文件选择
        file_layout = QHBoxLayout()
        self.file_input = LineEdit(self)
        self.file_input.setPlaceholderText("选择要压缩的文件...")
        self.file_input.setReadOnly(True)
        file_btn = PushButton("选择文件", self)
        if QFLUENT_AVAILABLE:
            try:
                icon = getattr(FluentIcon, 'FOLDER', None)
                if icon:
                    file_btn.setIcon(icon)
            except (AttributeError, TypeError):
                pass  # 图标不可用时忽略
        file_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(file_btn)
        layout.addLayout(file_layout)
        
        # 编码方式选择
        method_layout = QHBoxLayout()
        method_label = BodyLabel("编码方式:", self)
        self.method_combo = ComboBox(self)
        self.method_combo.addItems(["哈夫曼编码 (Huffman)", "香农编码 (Shannon)", "Shannon-Fano-Elias编码"])
        self.method_combo.setCurrentIndex(0)
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        layout.addLayout(method_layout)
        
        # 输出文件路径
        output_layout = QHBoxLayout()
        self.output_input = LineEdit(self)
        self.output_input.setPlaceholderText("压缩文件保存路径（可选，默认在原文件目录）...")
        self.output_input.setReadOnly(True)
        output_btn = PushButton("选择保存位置", self)
        if QFLUENT_AVAILABLE:
            try:
                icon = getattr(FluentIcon, 'SAVE', None)
                if icon:
                    output_btn.setIcon(icon)
            except (AttributeError, TypeError):
                pass  # 图标不可用时忽略
        output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(output_btn)
        layout.addLayout(output_layout)
        
        # 压缩按钮
        self.compress_btn = PrimaryPushButton("开始压缩", self)
        if QFLUENT_AVAILABLE:
            try:
                # 尝试使用常见的压缩相关图标
                for icon_name in ['ZIP_FOLDER', 'ARCHIVE', 'FOLDER_ZIP']:
                    icon = getattr(FluentIcon, icon_name, None)
                    if icon:
                        self.compress_btn.setIcon(icon)
                        break
            except (AttributeError, TypeError):
                pass  # 图标不可用时忽略
        self.compress_btn.clicked.connect(self.start_compress)
        layout.addWidget(self.compress_btn)
        
        # 进度条
        self.progress_bar = ProgressBar(self)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 结果显示区域
        self.result_text = TextEdit(self)
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(200)
        self.result_text.setPlaceholderText("压缩结果将显示在这里...")
        layout.addWidget(self.result_text)
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择要压缩的文件", "", "文本文件 (*.txt);;所有文件 (*.*)"
        )
        if file_path:
            self.file_input.setText(file_path)
            # 自动设置输出路径
            if not self.output_input.text():
                base_path = Path(file_path)
                output_path = base_path.parent / f"{base_path.stem}.compressed"
                self.output_input.setText(str(output_path))
    
    def select_output(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存压缩文件", "", "压缩文件 (*.compressed);;所有文件 (*.*)"
        )
        if file_path:
            self.output_input.setText(file_path)
    
    def start_compress(self):
        input_path = self.file_input.text()
        if not input_path or not os.path.exists(input_path):
            InfoBar.warning("警告", "请先选择要压缩的文件", parent=self, duration=3000)
            return
        
        output_path = self.output_input.text()
        if not output_path:
            base_path = Path(input_path)
            output_path = str(base_path.parent / f"{base_path.stem}.compressed")
            self.output_input.setText(output_path)
        
        # 获取编码方式
        method_map = {
            0: 'huffman',
            1: 'shannon',
            2: 'sfe'
        }
        method = method_map[self.method_combo.currentIndex()]
        
        # 禁用按钮
        self.compress_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.result_text.clear()
        
        # 启动工作线程
        self.thread = CompressionThread('encode', input_path, output_path, method)
        self.thread.finished.connect(self.on_compress_finished)
        self.thread.error.connect(self.on_error)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.start()
    
    def on_compress_finished(self, result: dict):
        self.compress_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if result['success']:
            info = f"""压缩完成！

编码方式: {result['method']}
原始大小: {result['original_size']} 字符 ({result['original_size_bits']} bits)
压缩后大小: {result['compressed_size_bytes']} 字节 ({result['compressed_size_bits']} bits)
压缩率: {result['compression_ratio']:.4f}
理论熵 H(X): {result['entropy']:.4f} bits/符号
平均码长 L_avg: {result['avg_code_length']:.4f} bits/符号
编码效率: {result['efficiency']:.4f}

文件已保存到: {self.output_input.text()}
"""
            self.result_text.setText(info)
            InfoBar.success("成功", "文件压缩完成！", parent=self, duration=3000)
        else:
            InfoBar.error("错误", "压缩失败", parent=self, duration=3000)
    
    def on_error(self, error_msg: str):
        self.compress_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        InfoBar.error("错误", error_msg, parent=self, duration=5000)
        self.result_text.setText(f"错误: {error_msg}")

import os
from pathlib import Path
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QFileDialog
from PySide6.QtCore import Qt

class ImageCompressionCard(CardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        title = TitleLabel("图片压缩/解压", self)
        title.setAlignment(Qt.AlignLeft)
        title.setStyleSheet("font-weight:700; font-size:14pt;")
        layout.addWidget(title)

        desc = BodyLabel("使用哈夫曼 / 香农 / SFE 对图片进行熵编码（支持 L 与 RGB）。", self)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        h1 = QHBoxLayout()
        self.input_input = LineEdit(self)
        self.input_input.setPlaceholderText("选择待压缩的图片 或 选择待解压的 .icomp")
        btn_sel = PushButton("选择文件", self)
        btn_sel.clicked.connect(self.select_file)
        h1.addWidget(self.input_input)
        h1.addWidget(btn_sel)
        layout.addLayout(h1)

        h2 = QHBoxLayout()
        self.output_input = LineEdit(self)
        self.output_input.setPlaceholderText("选择输出路径（文件或目录）")
        btn_out = PushButton("选择保存位置", self)
        btn_out.clicked.connect(self.select_output)
        h2.addWidget(self.output_input)
        h2.addWidget(btn_out)
        layout.addLayout(h2)

        h3 = QHBoxLayout()
        self.method_box = ComboBox(self)
        self.method_box.addItems(['huffman', 'shannon', 'sfe'])
        self.mode_box = ComboBox(self)
        self.mode_box.addItems(['L', 'RGB'])
        h3.addWidget(QLabel("方法：", self))
        h3.addWidget(self.method_box)
        h3.addSpacing(10)
        h3.addWidget(QLabel("模式：", self))
        h3.addWidget(self.mode_box)
        layout.addLayout(h3)

        h4 = QHBoxLayout()
        self.compress_btn = PrimaryPushButton("压缩图片", self)
        self.compress_btn.clicked.connect(self.start_compress)
        self.decompress_btn = PushButton("解压 .icomp", self)
        self.decompress_btn.clicked.connect(self.start_decompress)
        h4.addWidget(self.compress_btn)
        h4.addWidget(self.decompress_btn)
        layout.addLayout(h4)

        self.result_text = TextEdit(self)
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(140)
        layout.addWidget(self.result_text)

    def select_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择文件", str(Path.home()), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;IComp (*.icomp);;All Files (*)")
        if f:
            self.input_input.setText(f)

    def select_output(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", str(Path.home()))
        if d:
            self.output_input.setText(d)

    def start_compress(self):
        inp = self.input_input.text().strip()
        out = self.output_input.text().strip()
        if not inp:
            InfoBar.error("错误", "请选择输入图片文件。", parent=self, duration=3000); return
        if not out:
            InfoBar.error("错误", "请选择输出目录或文件。", parent=self, duration=3000); return
        method = self.method_box.currentText()
        mode = self.mode_box.currentText()
        save_gray = None
        if mode == 'L':
            save_gray = os.path.join(out if os.path.isdir(out) else os.path.dirname(out), Path(inp).stem + "_gray.png")
        self.compress_btn.setEnabled(False)
        self.decompress_btn.setEnabled(False)
        self.result_text.clear()
        self.thread = ImageCompressThread(inp, out, method=method, mode=mode, save_gray=save_gray)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def start_decompress(self):
        inp = self.input_input.text().strip()
        out = self.output_input.text().strip()
        if not inp:
            InfoBar.error("错误", "请选择输入 .icomp 文件。", parent=self, duration=3000); return
        if not out:
            InfoBar.error("错误", "请选择输出目录。", parent=self, duration=3000); return
        self.compress_btn.setEnabled(False)
        self.decompress_btn.setEnabled(False)
        self.result_text.clear()
        self.thread = ImageDecompressThread(inp, out)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_finished(self, result: dict):
        self.compress_btn.setEnabled(True)
        self.decompress_btn.setEnabled(True)
        if result.get('type') == 'image_compress':
            stats = result.get('stats', {})
            info = f"已压缩到: {result.get('output')}\n方法: {result.get('method')} 模式: {result.get('mode')}\n图片: {Path(result.get('input')).name}\n文件大小: {stats.get('file_bytes', 'N/A')} bytes\nratio: {stats.get('ratio_bits_over_raw', 'N/A')}"
            self.result_text.setText(info)
            InfoBar.success("完成", "图片压缩完成", parent=self, duration=3000)
        else:
            info = f"已解压到: {result.get('output')}\n模式: {result.get('mode')} 尺寸: {result.get('size')}"
            self.result_text.setText(info)
            InfoBar.success("完成", "图片解压完成", parent=self, duration=3000)

    def on_error(self, error_msg: str):
        self.compress_btn.setEnabled(True)
        self.decompress_btn.setEnabled(True)
        InfoBar.error("错误", error_msg, parent=self, duration=5000)
        self.result_text.setText(f"错误: {error_msg}")

class DecompressionCard(CardWidget):
    """解压功能卡片"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()
        self.thread: Optional[CompressionThread] = None
    
    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title = TitleLabel("文件解压", self)
        layout.addWidget(title)
        
        # 文件选择
        file_layout = QHBoxLayout()
        self.file_input = LineEdit(self)
        self.file_input.setPlaceholderText("选择要解压的文件...")
        self.file_input.setReadOnly(True)
        file_btn = PushButton("选择文件", self)
        if QFLUENT_AVAILABLE:
            try:
                icon = getattr(FluentIcon, 'FOLDER', None)
                if icon:
                    file_btn.setIcon(icon)
            except (AttributeError, TypeError):
                pass  # 图标不可用时忽略
        file_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(file_btn)
        layout.addLayout(file_layout)
        
        # 输出文件路径
        output_layout = QHBoxLayout()
        self.output_input = LineEdit(self)
        self.output_input.setPlaceholderText("解压文件保存路径（可选，默认在原文件目录）...")
        self.output_input.setReadOnly(True)
        output_btn = PushButton("选择保存位置", self)
        if QFLUENT_AVAILABLE:
            try:
                icon = getattr(FluentIcon, 'SAVE', None)
                if icon:
                    output_btn.setIcon(icon)
            except (AttributeError, TypeError):
                pass  # 图标不可用时忽略
        output_btn.clicked.connect(self.select_output)
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(output_btn)
        layout.addLayout(output_layout)
        
        # 解压按钮
        self.decompress_btn = PrimaryPushButton("开始解压", self)
        if QFLUENT_AVAILABLE:
            try:
                # 尝试使用常见的下载/解压相关图标
                for icon_name in ['DOWNLOAD', 'EXTRACT', 'UNZIP']:
                    icon = getattr(FluentIcon, icon_name, None)
                    if icon:
                        self.decompress_btn.setIcon(icon)
                        break
            except (AttributeError, TypeError):
                pass  # 图标不可用时忽略
        self.decompress_btn.clicked.connect(self.start_decompress)
        layout.addWidget(self.decompress_btn)
        
        # 进度条
        self.progress_bar = ProgressBar(self)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 结果显示区域
        self.result_text = TextEdit(self)
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        self.result_text.setPlaceholderText("解压结果将显示在这里...")
        layout.addWidget(self.result_text)
    
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择要解压的文件", "", "压缩文件 (*.compressed);;所有文件 (*.*)"
        )
        if file_path:
            self.file_input.setText(file_path)
            # 自动设置输出路径
            if not self.output_input.text():
                base_path = Path(file_path)
                output_path = base_path.parent / f"{base_path.stem}_decompressed.txt"
                self.output_input.setText(str(output_path))
    
    def select_output(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存解压文件", "", "文本文件 (*.txt);;所有文件 (*.*)"
        )
        if file_path:
            self.output_input.setText(file_path)
    
    def start_decompress(self):
        input_path = self.file_input.text()
        if not input_path or not os.path.exists(input_path):
            InfoBar.warning("警告", "请先选择要解压的文件", parent=self, duration=3000)
            return
        
        output_path = self.output_input.text()
        if not output_path:
            base_path = Path(input_path)
            output_path = str(base_path.parent / f"{base_path.stem}_decompressed.txt")
            self.output_input.setText(output_path)
        
        # 禁用按钮
        self.decompress_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.result_text.clear()
        
        # 启动工作线程
        self.thread = CompressionThread('decode', input_path, output_path)
        self.thread.finished.connect(self.on_decompress_finished)
        self.thread.error.connect(self.on_error)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.start()
    
    def on_decompress_finished(self, result: dict):
        self.decompress_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if result['success']:
            info = f"""解压完成！

原始大小: {result['original_size']} 字符
解压后大小: {result['decoded_size']} 字符
编码方式: {result['method']}

文件已保存到: {self.output_input.text()}
"""
            if result['original_size'] == result['decoded_size']:
                info += "\n✓ 解码验证通过"
            else:
                info += "\n✗ 警告：解码后大小不匹配"
            
            self.result_text.setText(info)
            InfoBar.success("成功", "文件解压完成！", parent=self, duration=3000)
        else:
            InfoBar.error("错误", "解压失败", parent=self, duration=3000)
    
    def on_error(self, error_msg: str):
        self.decompress_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        InfoBar.error("错误", error_msg, parent=self, duration=5000)
        self.result_text.setText(f"错误: {error_msg}")


class ExperimentThread(QThread):
    """实验工作线程"""
    finished = Signal(object)  # 传递实验结果
    error = Signal(str)
    progress = Signal(int)
    
    def __init__(self, exp_type: str):
        super().__init__()
        self.exp_type = exp_type  # 'exp1', 'exp2', 'exp3'
    
    def run(self):
        try:
            if self.exp_type == 'exp1':
                result = self._run_exp1()
            elif self.exp_type == 'exp2':
                result = self._run_exp2()
            elif self.exp_type == 'exp3':
                result = self._run_exp3()
            elif self.exp_type == 'exp4':
                result = self._run_exp4()
            else:
                self.error.emit(f"未知实验类型: {self.exp_type}")
                return
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def _run_exp1(self):
        """实验1：人工信源验证"""
        # 确保路径正确
        base_dir = os.path.dirname(__file__)
        sys.path.insert(0, os.path.join(base_dir, 'src'))
        sys.path.insert(0, os.path.join(base_dir, 'experiments'))
        
        # 导入实验函数
        import exp1_artificial
        from encoder import HuffmanEncoder, ShannonEncoder, ShannonFanoEliasEncoder
        from utils import calculate_entropy, get_frequency_distribution
        
        generate_artificial_source = exp1_artificial.generate_artificial_source
        
        distribution_types = ['uniform', 'moderate', 'skewed']
        results = []
        
        for i, dist_type in enumerate(distribution_types):
            self.progress.emit(int((i + 1) * 33))
            text = generate_artificial_source(dist_type, num_symbols=10, size=10000)
            _, probabilities = get_frequency_distribution(text)
            entropy = calculate_entropy(probabilities)
            
            huffman = HuffmanEncoder(probabilities)
            shannon = ShannonEncoder(probabilities)
            sfe = ShannonFanoEliasEncoder(probabilities)
            
            results.append({
                'type': dist_type,
                'entropy': entropy,
                'huffman': huffman.get_average_code_length(),
                'shannon': shannon.get_average_code_length(),
                'sfe': sfe.get_average_code_length()
            })
        
        return {'type': 'exp1', 'results': results}
    
    def _run_exp2(self):
        """实验2：真实文本压缩"""
        base_dir = os.path.dirname(__file__)
        sys.path.insert(0, os.path.join(base_dir, 'src'))
        sys.path.insert(0, os.path.join(base_dir, 'experiments'))
        
        import exp2_real_text
        from encoder import HuffmanEncoder, ShannonEncoder, ShannonFanoEliasEncoder
        from utils import get_frequency_distribution, calculate_entropy
        
        create_sample_files = exp2_real_text.create_sample_files
        
        create_sample_files()
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, 'data', 'samples')
        sample_files = ['english.txt', 'chinese.txt', 'code.txt']
        results = []
        
        for i, filename in enumerate(sample_files):
            self.progress.emit(int((i + 1) * 33))
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                continue
            
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            _, probabilities = get_frequency_distribution(text)
            entropy = calculate_entropy(probabilities)
            
            huffman = HuffmanEncoder(probabilities)
            shannon = ShannonEncoder(probabilities)
            sfe = ShannonFanoEliasEncoder(probabilities)
            
            results.append({
                'file': filename.replace('.txt', ''),
                'entropy': entropy,
                'huffman': huffman.get_average_code_length(),
                'shannon': shannon.get_average_code_length(),
                'sfe': sfe.get_average_code_length()
            })
        
        return {'type': 'exp2', 'results': results}
    
    def _run_exp3(self):
        """实验3：马尔可夫信源"""
        base_dir = os.path.dirname(__file__)
        sys.path.insert(0, os.path.join(base_dir, 'src'))
        sys.path.insert(0, os.path.join(base_dir, 'experiments'))
        
        import exp3_markov
        from encoder import HuffmanEncoder
        from utils import calculate_entropy, get_frequency_distribution
        from collections import Counter
        
        generate_markov_chain = exp3_markov.generate_markov_chain
        calculate_markov_entropy_rate = exp3_markov.calculate_markov_entropy_rate
        
        test_cases = [
            (0.1, 0.1, "高相关性"),
            (0.3, 0.3, "中等相关性"),
            (0.5, 0.5, "低相关性"),
            (0.7, 0.2, "非对称"),
        ]
        results = []
        
        for i, (p, q, name) in enumerate(test_cases):
            self.progress.emit(int((i + 1) * 25))
            text = generate_markov_chain(p, q, size=10000)
            _, prob_single = get_frequency_distribution(text)
            H_0 = calculate_entropy(prob_single)
            H_rate = calculate_markov_entropy_rate(p, q)
            
            encoder_single = HuffmanEncoder(prob_single)
            L_single = encoder_single.get_average_code_length()
            
            # 2-bit块编码
            text_for_pairs = text[:len(text)//2*2]
            pairs = [text_for_pairs[i:i+2] for i in range(0, len(text_for_pairs), 2)]
            pair_counter = Counter(pairs)
            total_pairs = len(pairs)
            prob_pairs = {pair: count / total_pairs for pair, count in pair_counter.items()}
            encoder_pairs = HuffmanEncoder(prob_pairs)
            L_pairs = encoder_pairs.get_average_code_length() / 2
            
            results.append({
                'name': name,
                'p': p,
                'q': q,
                'H_0': H_0,
                'H_rate': H_rate,
                'L_single': L_single,
                'L_pairs': L_pairs
            })
        
        return {'type': 'exp3', 'results': results}
    
    def _run_exp4(self):
        """实验4：KL散度与冗余的关系"""
        try:
            base_dir = os.path.dirname(__file__)
            sys.path.insert(0, os.path.join(base_dir, 'src'))
            sys.path.insert(0, os.path.join(base_dir, 'experiments'))
            
            print("开始运行实验4...")
            import exp4_kl
            ExtendedKLExperiment = exp4_kl.ExtendedKLExperiment
            
            # 设置分布参数
            P = [0.7, 0.3]  # 真实分布
            
            # 三个主要的Q分布（用于对比）
            main_q_dists = [
                [0.5, 0.5],  # 均匀分布
                [0.9, 0.1],  # 极端分布
                [0.8, 0.2]  # 中度偏离分布
            ]
            
            # 创建实验实例
            print("创建实验实例...")
            experiment = ExtendedKLExperiment()
            
            # 运行主实验
            print("运行主实验...")
            self.progress.emit(30)
            results = experiment.run_main_experiment(P, main_q_dists, sequence_length=100000)
            print(f"主实验完成，结果类型: {type(results)}")
            
            # 生成多个Q分布
            print("生成多个Q分布...")
            self.progress.emit(60)
            many_q_dists = experiment.generate_many_q_distributions(P, num_points=20)
            print(f"生成了 {len(many_q_dists)} 个Q分布")
            
            # 运行扩展实验
            print("运行扩展实验...")
            self.progress.emit(90)
            extended_results = experiment.run_extended_experiment(P, many_q_dists, sequence_length=100000)
            print(f"扩展实验完成，结果数量: {len(extended_results)}")
            
            # 准备返回数据（不传递实验对象，因为Qt信号无法传递复杂对象）
            result_dict = {
                'type': 'exp4',
                'results': results,
                'extended_results': extended_results,
                'p_dist': P  # 传递分布参数，用于绘图
            }
            print("实验4完成，准备返回结果")
            self.progress.emit(100)
            return result_dict
        except Exception as e:
            import traceback
            error_msg = f"实验4运行错误: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            raise Exception(error_msg)


class ExperimentTab(QWidget):
    """实验选项卡基类"""
    
    def __init__(self, exp_type: str, title: str, description: str, parent=None):
        super().__init__(parent)
        self.exp_type = exp_type
        self.title = title
        self.description = description
        self.thread: Optional[ExperimentThread] = None
        self.setupUI()
    
    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题和描述
        title_label = TitleLabel(self.title, self)
        layout.addWidget(title_label)
        
        desc_label = BodyLabel(self.description, self)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # 运行按钮
        self.run_btn = PrimaryPushButton("运行实验", self)
        self.run_btn.clicked.connect(self.run_experiment)
        layout.addWidget(self.run_btn)
        
        # 进度条
        self.progress_bar = ProgressBar(self)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 图表显示区域
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # 结果显示
        self.result_text = TextEdit(self)
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        layout.addWidget(self.result_text)
    
    def save_figure(self, filename: str):
        """保存当前figure到results/figures文件夹"""
        try:
            # 获取项目根目录
            base_dir = os.path.dirname(os.path.abspath(__file__))
            fig_dir = os.path.join(base_dir, 'results', 'figures')
            
            # 确保文件夹存在
            os.makedirs(fig_dir, exist_ok=True)
            
            # 保存图片
            filepath = os.path.join(fig_dir, filename)
            self.figure.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"图片已保存: {filepath}")
            return filepath
        except Exception as e:
            print(f"保存图片失败: {e}")
            return None
    
    def run_experiment(self):
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.result_text.clear()
        self.figure.clear()
        
        self.thread = ExperimentThread(self.exp_type)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.start()
    
    def on_finished(self, result: dict):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if result['type'] == 'exp1':
            self.plot_exp1(result['results'])
        elif result['type'] == 'exp2':
            self.plot_exp2(result['results'])
        elif result['type'] == 'exp3':
            self.plot_exp3(result['results'])
        elif result['type'] == 'exp4':
            try:
                self.plot_exp4(result)
                self.canvas.draw()
                InfoBar.success("成功", "实验完成！", parent=self, duration=3000)
            except Exception as e:
                import traceback
                error_msg = f"实验4绘图失败: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                self.result_text.setText(error_msg)
                InfoBar.error("错误", f"绘图失败: {str(e)}", parent=self, duration=5000)
        else:
            self.canvas.draw()
            InfoBar.success("成功", "实验完成！", parent=self, duration=3000)
    
    def on_error(self, error_msg: str):
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        import traceback
        error_detail = f"错误: {error_msg}\n\n详细错误信息:\n{traceback.format_exc()}"
        InfoBar.error("错误", error_msg, parent=self, duration=5000)
        self.result_text.setText(error_detail)
        print(f"实验错误: {error_msg}")  # 控制台输出便于调试
        print(traceback.format_exc())
    
    def plot_exp1(self, results):
        """绘制实验1结果 - 美化版"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        types = [r['type'] for r in results]
        entropy = [r['entropy'] for r in results]
        huffman = [r['huffman'] for r in results]
        shannon = [r['shannon'] for r in results]
        sfe = [r['sfe'] for r in results]
        
        # 美观的颜色方案
        colors = {
            'entropy': '#2E86AB',      # 深蓝色
            'huffman': '#A23B72',      # 紫红色
            'shannon': '#F18F01',     # 橙色
            'sfe': '#C73E1D'          # 深红色
        }
        
        x = np.arange(len(types))
        width = 0.18
        spacing = 0.02
        
        # 绘制柱状图，添加边框和阴影效果
        bars1 = ax.bar(x - width * 1.5 - spacing, entropy, width, 
                      label='理论熵 H(X)', color=colors['entropy'], 
                      alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
        bars2 = ax.bar(x - width * 0.5, huffman, width, 
                      label='哈夫曼码', color=colors['huffman'], 
                      alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
        bars3 = ax.bar(x + width * 0.5 + spacing, shannon, width, 
                      label='香农码', color=colors['shannon'], 
                      alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
        bars4 = ax.bar(x + width * 1.5 + spacing * 2, sfe, width, 
                      label='SFE码', color=colors['sfe'], 
                      alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
        
        # 添加数值标签（五号宋体）
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10.5, 
                       family='SimSun', weight='normal')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        add_value_labels(bars4)
        
        # 计算y轴上限，为图例和数值标签留出足够空间
        max_value = max(max(entropy), max(huffman), max(shannon), max(sfe))
        y_max = max_value * 1.5  # 增加50%的空间
        ax.set_ylim(0, y_max)
        
        # 美化坐标轴（五号宋体）
        ax.set_xlabel('分布类型', fontsize=10.5, family='SimSun', 
                     weight='normal', color='#333333')
        ax.set_ylabel('平均码长 (bits/符号)', fontsize=10.5, family='SimSun', 
                     weight='normal', color='#333333')
        ax.set_title('实验1：不同编码方法的平均码长比较', 
                    fontsize=10.5, family='SimSun', weight='normal', 
                    pad=20, color='#1a1a1a')
        
        # 设置x轴标签（五号宋体）
        ax.set_xticks(x)
        ax.set_xticklabels(['均匀分布', '中等偏斜', '极度偏斜'], 
                          fontsize=10.5, family='SimSun', weight='normal')
        
        # 美化图例（五号宋体）- 横向排列放在右上角
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                          shadow=True, fontsize=10.5, prop={'family': 'SimSun', 'size': 10.5}, 
                          framealpha=0.95, edgecolor='#cccccc', facecolor='white',
                          ncol=4)  # 横向4列排列
        legend.get_frame().set_linewidth(1.5)
        
        # 设置坐标轴刻度标签为五号宋体
        ax.tick_params(axis='both', labelsize=10.5)
        for label in ax.get_xticklabels():
            label.set_fontfamily('SimSun')
        for label in ax.get_yticklabels():
            label.set_fontfamily('SimSun')
        
        # 美化网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='#888888', zorder=0)
        ax.set_axisbelow(True)
        
        # 设置背景色
        ax.set_facecolor('#fafafa')
        self.figure.patch.set_facecolor('white')
        
        # 美化边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')
            spine.set_linewidth(1.2)
        
        # 调整布局
        self.figure.tight_layout(pad=2.0)
        
        # 保存图片
        saved_path = self.save_figure('exp1_artificial.png')
        
        # 显示结果文本
        text = "📊 实验结果详情：\n\n"
        type_names = {'uniform': '均匀分布', 'moderate': '中等偏斜', 'skewed': '极度偏斜'}
        for r in results:
            type_name = type_names.get(r['type'], r['type'])
            text += f"🔹 {type_name}:\n"
            text += f"   理论熵: {r['entropy']:.4f} bits/符号\n"
            text += f"   哈夫曼码: {r['huffman']:.4f} bits/符号\n"
            text += f"   香农码: {r['shannon']:.4f} bits/符号\n"
            text += f"   SFE码: {r['sfe']:.4f} bits/符号\n\n"
        if saved_path:
            text += f"\n💾 图片已保存: results/figures/exp1_artificial.png\n"
        self.result_text.setText(text)
    
    def plot_exp2(self, results):
        """绘制实验2结果 - 美化版"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        files = [r['file'] for r in results]
        entropy = [r['entropy'] for r in results]
        huffman = [r['huffman'] for r in results]
        shannon = [r['shannon'] for r in results]
        sfe = [r['sfe'] for r in results]
        
        # 美观的颜色方案
        colors = {
            'entropy': '#2E86AB',      # 深蓝色
            'huffman': '#A23B72',      # 紫红色
            'shannon': '#F18F01',     # 橙色
            'sfe': '#C73E1D'          # 深红色
        }
        
        x = np.arange(len(files))
        width = 0.18
        spacing = 0.02
        
        # 绘制柱状图
        bars1 = ax.bar(x - width * 1.5 - spacing, entropy, width, 
                      label='理论熵 H(X)', color=colors['entropy'], 
                      alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
        bars2 = ax.bar(x - width * 0.5, huffman, width, 
                      label='哈夫曼码', color=colors['huffman'], 
                      alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
        bars3 = ax.bar(x + width * 0.5 + spacing, shannon, width, 
                      label='香农码', color=colors['shannon'], 
                      alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
        bars4 = ax.bar(x + width * 1.5 + spacing * 2, sfe, width, 
                      label='SFE码', color=colors['sfe'], 
                      alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
        
        # 添加数值标签（五号宋体）
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10.5, 
                       family='SimSun', weight='normal')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        add_value_labels(bars4)
        
        # 计算y轴上限，为图例和数值标签留出足够空间
        max_value = max(max(entropy), max(huffman), max(shannon), max(sfe))
        y_max = max_value * 1.5  # 增加50%的空间
        ax.set_ylim(0, y_max)
        
        # 美化坐标轴（五号宋体）
        ax.set_xlabel('文件类型', fontsize=10.5, family='SimSun', 
                     weight='normal', color='#333333')
        ax.set_ylabel('平均码长 (bits/符号)', fontsize=10.5, family='SimSun', 
                     weight='normal', color='#333333')
        ax.set_title('实验2：真实文本数据的压缩效果对比', 
                    fontsize=10.5, family='SimSun', weight='normal', 
                    pad=20, color='#1a1a1a')
        
        # 设置x轴标签（五号宋体）
        file_labels = {'english': '英文文本', 'chinese': '中文文本', 'code': '代码文本'}
        labels = [file_labels.get(f, f) for f in files]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10.5, family='SimSun', weight='normal')
        
        # 美化图例（五号宋体）- 横向排列放在右上角
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                          shadow=True, fontsize=10.5, prop={'family': 'SimSun', 'size': 10.5}, 
                          framealpha=0.95, edgecolor='#cccccc', facecolor='white',
                          ncol=4)  # 横向4列排列
        legend.get_frame().set_linewidth(1.5)
        
        # 设置坐标轴刻度标签为五号宋体
        ax.tick_params(axis='both', labelsize=10.5)
        for label in ax.get_xticklabels():
            label.set_fontfamily('SimSun')
        for label in ax.get_yticklabels():
            label.set_fontfamily('SimSun')
        
        # 美化网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='#888888', zorder=0)
        ax.set_axisbelow(True)
        
        # 设置背景色
        ax.set_facecolor('#fafafa')
        self.figure.patch.set_facecolor('white')
        
        # 美化边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')
            spine.set_linewidth(1.2)
        
        # 调整布局
        self.figure.tight_layout(pad=2.0)
        
        # 保存图片
        saved_path = self.save_figure('exp2_real_text.png')
        
        # 显示结果文本
        text = "📊 实验结果详情：\n\n"
        file_names = {'english': '英文文本', 'chinese': '中文文本', 'code': '代码文本'}
        for r in results:
            file_name = file_names.get(r['file'], r['file'])
            text += f"🔹 {file_name}:\n"
            text += f"   理论熵: {r['entropy']:.4f} bits/符号\n"
            text += f"   哈夫曼码: {r['huffman']:.4f} bits/符号\n"
            text += f"   香农码: {r['shannon']:.4f} bits/符号\n"
            text += f"   SFE码: {r['sfe']:.4f} bits/符号\n\n"
        if saved_path:
            text += f"\n💾 图片已保存: results/figures/exp2_real_text.png\n"
        self.result_text.setText(text)
    
    def plot_exp3(self, results):
        """绘制实验3结果 - 美化版"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        names = [r['name'] for r in results]
        H_0 = [r['H_0'] for r in results]
        H_rate = [r['H_rate'] for r in results]
        L_single = [r['L_single'] for r in results]
        L_pairs = [r['L_pairs'] for r in results]
        
        # 美观的颜色方案
        colors = {
            'H_0': '#6A4C93',         # 紫色
            'H_rate': '#1982C4',      # 蓝色
            'L_single': '#FF6B35',    # 橙红色
            'L_pairs': '#06A77D'      # 青绿色
        }
        
        x = np.arange(len(names))
        width = 0.18
        spacing = 0.02
        
        # 绘制柱状图
        bars1 = ax.bar(x - width * 1.5 - spacing, H_0, width, 
                      label='零阶熵 H_0', color=colors['H_0'], 
                      alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
        bars2 = ax.bar(x - width * 0.5, H_rate, width, 
                      label='理论熵率 H', color=colors['H_rate'], 
                      alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
        bars3 = ax.bar(x + width * 0.5 + spacing, L_single, width, 
                      label='单符号编码 L_avg', color=colors['L_single'], 
                      alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
        bars4 = ax.bar(x + width * 1.5 + spacing * 2, L_pairs, width, 
                      label='2-bit块编码 L_avg', color=colors['L_pairs'], 
                      alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
        
        # 添加数值标签（五号宋体）
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10.5, 
                       family='SimSun', weight='normal')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        add_value_labels(bars4)
        
        # 计算y轴上限，为图例和数值标签留出足够空间
        max_value = max(max(H_0), max(H_rate), max(L_single), max(L_pairs))
        y_max = max_value * 1.5  # 增加50%的空间
        ax.set_ylim(0, y_max)
        
        # 美化坐标轴（五号宋体）
        ax.set_xlabel('测试案例', fontsize=10.5, family='SimSun', 
                     weight='normal', color='#333333')
        ax.set_ylabel('熵/平均码长 (bits/符号)', fontsize=10.5, family='SimSun', 
                     weight='normal', color='#333333')
        ax.set_title('实验3：零阶熵 vs 熵率 vs 编码性能对比', 
                    fontsize=10.5, family='SimSun', weight='normal', 
                    pad=20, color='#1a1a1a')
        
        # 设置x轴标签（五号宋体）
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right', 
                          fontsize=10.5, family='SimSun', weight='normal')
        
        # 美化图例（五号宋体）- 横向排列放在右上角
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                          shadow=True, fontsize=10.5, prop={'family': 'SimSun', 'size': 10.5}, 
                          framealpha=0.95, edgecolor='#cccccc', facecolor='white',
                          ncol=4)  # 横向4列排列
        legend.get_frame().set_linewidth(1.5)
        
        # 设置坐标轴刻度标签为五号宋体
        ax.tick_params(axis='both', labelsize=10.5)
        for label in ax.get_xticklabels():
            label.set_fontfamily('SimSun')
        for label in ax.get_yticklabels():
            label.set_fontfamily('SimSun')
        
        # 美化网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='#888888', zorder=0)
        ax.set_axisbelow(True)
        
        # 设置背景色
        ax.set_facecolor('#fafafa')
        self.figure.patch.set_facecolor('white')
        
        # 美化边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')
            spine.set_linewidth(1.2)
        
        # 调整布局
        self.figure.tight_layout(pad=2.0)
        
        # 保存图片
        saved_path = self.save_figure('exp3_markov.png')
        
        # 显示结果文本
        text = "📊 实验结果详情：\n\n"
        for r in results:
            text += f"🔹 {r['name']} (p={r['p']}, q={r['q']}):\n"
            text += f"   零阶熵 H₀: {r['H_0']:.4f} bits/符号\n"
            text += f"   理论熵率 H: {r['H_rate']:.4f} bits/符号\n"
            text += f"   单符号编码: {r['L_single']:.4f} bits/符号\n"
            text += f"   2-bit块编码: {r['L_pairs']:.4f} bits/符号\n"
            text += f"   效率提升: {((r['L_single'] - r['L_pairs']) / r['L_single'] * 100):.2f}%\n\n"
        if saved_path:
            text += f"\n💾 图片已保存: results/figures/exp3_markov.png\n"
        self.result_text.setText(text)
    
    def plot_exp4(self, result_data):
        """绘制实验4结果：KL散度与冗余的关系"""
        try:
            self.figure.clear()
            
            # 获取数据（不传递实验对象，因为Qt信号无法传递复杂对象）
            if not isinstance(result_data, dict):
                raise ValueError(f"result_data应该是字典，但得到: {type(result_data)}")
            
            results = result_data.get('results')
            extended_results = result_data.get('extended_results')
            p_dist = result_data.get('p_dist', [0.7, 0.3])
            
            if results is None:
                raise ValueError("缺少'results'数据")
            if extended_results is None:
                raise ValueError("缺少'extended_results'数据")
            
            print(f"plot_exp4: results类型={type(results)}, keys={results.keys() if isinstance(results, dict) else 'N/A'}")
            print(f"plot_exp4: extended_results数量={len(extended_results) if extended_results else 0}")
            
            # 设置figure大小
            self.figure.set_size_inches(12, 10)
            
            # 创建2x2子图布局
            axes = self.figure.subplots(2, 2)
            ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
            
            # 1. 主实验：三个主要分布的结果
            categories1 = []
            values1 = []
            
            categories1.extend(['H(P)', 'L_P(理想)'])
            values1.extend([results['H_P'], results['L_P_ideal']])
            
            for i, q_result in enumerate(results['q_results']):
                categories1.extend([
                    f'H(P)+D(P∥Q{i + 1})',
                    f'L_Q{i + 1}(理想)'
                ])
                values1.extend([
                    q_result['L_Q_theoretical'],
                    q_result['L_Q_ideal']
                ])
            
            x_pos1 = np.arange(len(categories1))
            bars1 = ax1.bar(x_pos1, values1, width=0.5,
                            color=['#2E86AB', '#A23B72'] * (len(categories1) // 2))
            
            ax1.set_xticks(x_pos1)
            ax1.set_xticklabels(categories1, rotation=25, ha='right', fontsize=10.5)
            ax1.set_ylabel('比特数', fontsize=10.5, family='SimSun')
            ax1.set_title('主实验：理想码长下理论值与实验值比较', 
                         fontsize=10.5, family='SimSun', pad=15)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=results['H_P'], color='red', linestyle='--', 
                       alpha=0.5, label='H(P)', linewidth=1.5)
            
            for bar, value in zip(bars1, values1):
                height = bar.get_height()
                va_pos = 'bottom' if height < max(values1) * 0.4 else 'top'
                y_offset = 0.02 if height < max(values1) * 0.4 else -0.02
                ax1.text(bar.get_x() + bar.get_width() / 2., height + y_offset,
                        f'{value:.3f}', ha='center', va=va_pos, fontsize=9.5,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                 edgecolor='none', alpha=0.8))
            ax1.legend(fontsize=10.5, prop={'family': 'SimSun'})
            
            # 2. 扩展实验：冗余与KL散度的关系
            if not extended_results or len(extended_results) == 0:
                raise ValueError("extended_results为空，无法绘制扩展实验图表")
            
            kl_values = [r['D_PQ'] for r in extended_results]
            redundancy_values = [r['redundancy'] for r in extended_results]
            q1_values = [r['q1_value'] for r in extended_results]
            
            if len(kl_values) == 0:
                raise ValueError("无法从extended_results中提取KL散度值")
            
            scatter = ax2.scatter(kl_values, redundancy_values,
                                 c=q1_values, cmap='viridis', s=60, alpha=0.8,
                                 edgecolors='black', linewidth=0.5)
            
            cbar = self.figure.colorbar(scatter, ax=ax2, pad=0.02)
            cbar.set_label('Q的第一个分量值', fontsize=10.5, family='SimSun')
            cbar.ax.tick_params(labelsize=10)
            
            # 绘制理想对角线
            min_val = min(min(kl_values), min(redundancy_values))
            max_val = max(max(kl_values), max(redundancy_values))
            padding = (max_val - min_val) * 0.05
            ax2.plot([min_val - padding, max_val + padding],
                    [min_val - padding, max_val + padding],
                    'r--', alpha=0.8, linewidth=2, label='理想线: 冗余 = KL散度')
            
            # 线性回归
            if len(kl_values) > 1:
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(kl_values, redundancy_values)
                regression_line = [slope * x + intercept for x in kl_values]
                ax2.plot(kl_values, regression_line, 'g-', alpha=0.7,
                        linewidth=2, label=f'回归线: y={slope:.4f}x+{intercept:.4f}')
                r_squared = r_value ** 2
                ax2.text(0.98, 0.98, f'$R^2$ = {r_squared:.6f}', transform=ax2.transAxes,
                        fontsize=10.5, verticalalignment='top', horizontalalignment='right')
            
            # 标记三个主要实验点
            for i, q_result in enumerate(results['q_results']):
                label_key = f'Q{i + 1}'
                ax2.scatter(q_result['D_PQ'], q_result['redundancy_ideal'],
                           color='red', s=80, marker='s', edgecolors='black',
                           linewidth=1.5, zorder=5, label=f'主实验点{label_key}' if i == 0 else None)
                ax2.text(q_result['D_PQ'], q_result['redundancy_ideal'] + 0.015,
                        label_key, fontsize=10.5, ha='left', va='bottom',
                        fontweight='bold', bbox=dict(boxstyle='round,pad=0.2',
                                                    facecolor='white', edgecolor='none', alpha=0.9))
            
            ax2.set_xlabel('KL散度 D(P∥Q)', fontsize=10.5, family='SimSun')
            ax2.set_ylabel('理想冗余 (L_Q_ideal - H(P))', fontsize=10.5, family='SimSun')
            ax2.set_title('扩展实验：冗余与KL散度的关系（20个数据点）', 
                         fontsize=10.5, family='SimSun', pad=15)
            ax2.legend(fontsize=10.5, prop={'family': 'SimSun'}, loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # 3. 误差分布图
            differences = [abs(r['redundancy'] - r['D_PQ']) for r in extended_results]
            kl_bins = np.linspace(min(kl_values), max(kl_values), 8)
            bin_indices = np.digitize(kl_values, kl_bins)
            bin_avg_differences = []
            bin_centers = []
            for i in range(1, len(kl_bins)):
                mask = (bin_indices == i)
                if np.any(mask):
                    bin_avg_differences.append(np.mean(np.array(differences)[mask]))
                    bin_centers.append((kl_bins[i - 1] + kl_bins[i]) / 2)
            
            bars3 = ax3.bar(bin_centers, bin_avg_differences,
                           width=(kl_bins[1] - kl_bins[0]) * 0.6,
                           color='skyblue', edgecolor='black', alpha=0.7)
            
            for bar, value in zip(bars3, bin_avg_differences):
                height = bar.get_height()
                if height > max(bin_avg_differences) * 0.1:
                    ax3.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{value:.4f}', ha='center', va='bottom',
                            fontsize=9.5, rotation=0)
            
            ax3.set_xlabel('KL散度 D(P∥Q)', fontsize=10.5, family='SimSun')
            ax3.set_ylabel('平均绝对差异', fontsize=10.5, family='SimSun')
            ax3.set_title('冗余与KL散度差异的分布', fontsize=10.5, family='SimSun', pad=15)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # 4. Q分布参数与KL散度的关系
            sorted_indices = np.argsort(q1_values)
            sorted_q1 = np.array(q1_values)[sorted_indices]
            sorted_kl = np.array(kl_values)[sorted_indices]
            
            ax4.plot(sorted_q1, sorted_kl, 'b-', linewidth=2, alpha=0.7)
            
            key_indices = [0, len(sorted_q1) // 2, len(sorted_q1) - 1]
            for idx in key_indices:
                x, y = sorted_q1[idx], sorted_kl[idx]
                ha_pos = 'left' if idx == 0 else ('center' if idx == len(sorted_q1) // 2 else 'right')
                va_pos = 'bottom' if idx == 0 else ('top' if idx == len(sorted_q1) // 2 else 'bottom')
                ax4.text(x, y, f'({x:.2f},{y:.3f})', fontsize=8.5,
                        ha=ha_pos, va=va_pos,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            ax4.set_xlabel('Q的第一个分量值', fontsize=10.5, family='SimSun')
            ax4.set_ylabel('KL散度 D(P∥Q)', fontsize=10.5, family='SimSun')
            ax4.set_title('Q分布参数与KL散度的关系', fontsize=10.5, family='SimSun', pad=15)
            ax4.grid(True, alpha=0.3)
            
            p1 = p_dist[0]  # 使用传递的p_dist参数
            ax4.axvline(x=p1, color='red', linestyle='--', alpha=0.7,
                       linewidth=1.5, label=f'P的第一个分量={p1}')
            ax4.legend(fontsize=10.5, prop={'family': 'SimSun'}, loc='upper right')
            
            # 设置所有坐标轴的刻度字体大小
            for ax in [ax1, ax2, ax3, ax4]:
                ax.tick_params(axis='both', which='major', labelsize=10)
                for label in ax.get_xticklabels():
                    label.set_fontfamily('SimSun')
                for label in ax.get_yticklabels():
                    label.set_fontfamily('SimSun')
            
            # 调整布局
            self.figure.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)
            
            # 保存图片
            saved_path = self.save_figure('exp4_kl.png')
            
            # 显示结果文本
            text = "📊 实验4：冗余 = KL散度\n\n"
            text += f"真实分布P: [{p_dist[0]:.2f}, {p_dist[1]:.2f}]\n"
            text += f"熵 H(P): {results['H_P']:.4f} bits\n"
            text += f"理想码长 L_P(理想): {results['L_P_ideal']:.4f} bits\n\n"
            
            for i, q_result in enumerate(results['q_results']):
                text += f"🔹 Q{i+1} = [{q_result['q_dist'][0]:.2f}, {q_result['q_dist'][1]:.2f}]:\n"
                text += f"   KL散度 D(P∥Q{i+1}): {q_result['D_PQ']:.4f}\n"
                text += f"   理想码长 L_Q{i+1}(理想): {q_result['L_Q_ideal']:.4f} bits\n"
                text += f"   冗余: {q_result['redundancy_ideal']:.4f} bits\n"
                text += f"   理论预测: H(P)+D(P∥Q{i+1}) = {q_result['L_Q_theoretical']:.4f} bits\n\n"
            
            if len(kl_values) > 1:
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(kl_values, redundancy_values)
                r_squared = r_value ** 2
                text += f"📈 线性回归分析:\n"
                text += f"   斜率: {slope:.6f} (理论应为1.0)\n"
                text += f"   截距: {intercept:.6f} (理论应为0.0)\n"
                text += f"   决定系数R²: {r_squared:.6f}\n"
                text += f"   相关系数R: {r_value:.6f}\n"
            
            if saved_path:
                text += f"\n💾 图片已保存: results/figures/exp4_kl.png\n"
            
            self.result_text.setText(text)
        except Exception as e:
            import traceback
            error_msg = f"绘图错误: {str(e)}\n\n详细错误:\n{traceback.format_exc()}"
            print(error_msg)
            self.result_text.setText(error_msg)
            # 绘制一个简单的错误提示图
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'绘图错误:\n{str(e)}', 
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes, color='red')
            self.canvas.draw()


class MainWindow(QWidget):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("信息论压缩软件 - 基于哈夫曼与香农编码")
        self.resize(1000, 800)
        self.setupUI()
    
    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title = TitleLabel("信息论压缩软件", self)
        title.setAlignment(Qt.AlignCenter)
        subtitle = BodyLabel("基于哈夫曼编码、香农编码和Shannon-Fano-Elias编码", self)
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        # 创建选项卡
        self.tabs = QTabWidget(self)
        
        # 选项卡1：压缩/解压
        compress_tab = QWidget()
        compress_layout = QVBoxLayout(compress_tab)
        compress_layout.setSpacing(20)
        compress_layout.setContentsMargins(20, 20, 20, 20)
        
        scroll = ScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        self.compress_card = CompressionCard()
        self.decompress_card = DecompressionCard()
        scroll_layout.addWidget(self.compress_card)
        scroll_layout.addWidget(self.decompress_card)
        self.image_card = ImageCompressionCard()
        scroll_layout.addWidget(self.image_card)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        compress_layout.addWidget(scroll)
        
        self.tabs.addTab(compress_tab, "压缩/解压")
        
        # 选项卡2：实验1
        exp1_tab = ExperimentTab(
            'exp1',
            "实验1：人工信源验证信源编码定理",
            "构造不同概率分布（均匀、中等偏斜、极度偏斜），验证平均码长与熵的关系，比较哈夫曼、香农和SFE编码的性能。"
        )
        self.tabs.addTab(exp1_tab, "实验1：人工信源")
        
        # 选项卡3：实验2
        exp2_tab = ExperimentTab(
            'exp2',
            "实验2：真实文本数据的压缩效果",
            "测试不同编码方法在真实文本（英文、中文、代码）上的表现，比较压缩率和平均码长。"
        )
        self.tabs.addTab(exp2_tab, "实验2：真实文本")
        
        # 选项卡4：实验3
        exp3_tab = ExperimentTab(
            'exp3',
            "实验3：马尔可夫信源与熵率",
            "比较零阶熵与熵率，验证利用符号相关性可以降低平均码长，接近熵率。"
        )
        self.tabs.addTab(exp3_tab, "实验3：马尔可夫信源")
        
        # 选项卡5：实验4
        exp4_tab = ExperimentTab(
            'exp4',
            "实验4：冗余与KL散度的关系",
            "验证当使用错误分布Q进行编码时，冗余等于KL散度D(P∥Q)，即 L_Q - H(P) = D(P∥Q)。"
        )
        self.tabs.addTab(exp4_tab, "实验4：KL散度")
        
        layout.addWidget(self.tabs)


def main():
    # 创建应用
    app = QApplication(sys.argv)
    
    # 设置主题（如果qfluentwidgets可用）
    if QFLUENT_AVAILABLE:
        setTheme(Theme.AUTO)
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
