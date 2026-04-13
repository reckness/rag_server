"""
Mini DeepDoc Parser - 模仿 RAGFlow DeepDoc 的精简版文档解析器
功能：OCR识别 + 版面分析 + 表格识别 + 文本合并
"""

import os
import re
import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import tempfile

import cv2
import numpy as np
from PIL import Image

# 安装依赖：
# pip install paddlepaddle paddleocr opencv-python-headless pdf2image numpy Pillow

try:
    from paddleocr import PaddleOCR, PPStructureV3 as PPStructure
except ImportError:
    print("请先安装 paddleocr: pip install paddleocr")
    exit(1)

try:
    from pdf2image import convert_from_bytes, convert_from_path
except ImportError:
    print("请先安装 pdf2image: pip install pdf2image")
    print("Linux 还需要: apt-get install poppler-utils")
    print("Mac 还需要: brew install poppler")
    exit(1)


@dataclass
class BBox:
    """边界框数据结构，模仿 DeepDoc 的 bbox"""
    page_number: int
    x0: float
    x1: float
    top: float
    bottom: float
    text: str = ""
    layout_type: str = "text"  # text, title, table, figure, equation
    image: Optional[Image.Image] = None
    positions: List[List] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "page_number": self.page_number,
            "x0": self.x0,
            "x1": self.x1,
            "top": self.top,
            "bottom": self.bottom,
            "text": self.text,
            "layout_type": self.layout_type,
            "positions": self.positions,
        }


class MiniDeepDocParser:
    """
    精简版 DeepDoc 解析器
    
    核心功能：
    1. PDF/图片 OCR 文字识别
    2. 版面分析（识别标题、段落、表格、图片）
    3. 表格结构识别
    4. 文本合并和阅读顺序恢复
    """
    
    def __init__(
        self,
        lang: str = "ch",
        enable_layout: bool = True,
        enable_table: bool = False,  # 禁用表格识别，避免崩溃
        dpi: int = 200,
        model_dir: str = None,
    ):
        """
        初始化解析器
        
        Args:
            lang: 语言 ('ch', 'en', 'ch_en' 等)
            enable_layout: 是否启用版面分析
            enable_table: 是否启用表格识别
            dpi: PDF 转图片的分辨率
            model_dir: 模型文件目录路径
        """
        self.lang = lang
        self.enable_layout = enable_layout
        self.enable_table = enable_table
        self.dpi = dpi
        self.model_dir = model_dir
        
        # 初始化 OCR 引擎
        print("正在初始化 OCR 引擎...")
        
        # 尝试使用 PaddleOCR 2.x 版本的 API 来加载旧格式的模型
        try:
            # 尝试使用旧版本的 API
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                det_model_dir=None,
                rec_model_dir=None,
                cls_model_dir=None,
            )
            print("使用默认模型")
        except Exception as e:
            print(f"初始化 OCR 引擎失败: {e}")
            print("尝试使用备用方法...")
            # 备用方法：使用 PaddleOCR 的默认模型
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
            )
            print("使用默认模型")
        print("OCR 引擎初始化完成")
        
        # 初始化版面分析器
        if enable_layout:
            print("正在初始化版面分析器...")
            self.layout_engine = PPStructure()
            print("版面分析器初始化完成")
        
        # 表格识别模型（可选）
        self.table_engine = None
        if enable_table:
            try:
                from paddleocr import PPStructureV3 as TableEngine
                self.table_engine = TableEngine()
                print("表格识别引擎初始化完成")
            except Exception as e:
                print(f"表格识别引擎初始化失败: {e}")
                self.enable_table = False
        
        self.page_cum_height = [0]  # 累计高度，用于跨页坐标
        self.boxes: List[BBox] = []
        
    def pdf_to_images(self, pdf_data):
        """
        PDF 转图片
        
        Args:
            pdf_data: PDF 二进制数据
            
        Returns:
            图片列表 (numpy array 格式)
        """
        import tempfile
        import fitz  # PyMuPDF
        from PIL import Image
        import os
        
        images = []
        tmp_file = None
        doc = None
        try:
            # 创建临时文件，设置 delete=False，这样文件在关闭后不会被删除
            tmp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            tmp_file.write(pdf_data)
            tmp_file.flush()
            tmp_file.close()  # 关闭文件，以便 PyMuPDF 可以打开它
            
            # 使用 PyMuPDF 打开 PDF 文件
            doc = fitz.open(tmp_file.name)
            
            # 遍历每一页并转换为图像
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=self.dpi)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(np.array(img))
        finally:
            # 确保 PyMuPDF 已经关闭了文件
            if doc:
                doc.close()
            # 手动删除临时文件
            if tmp_file and os.path.exists(tmp_file.name):
                # 尝试删除文件，如果失败，等待一段时间后再尝试
                import time
                for i in range(3):
                    try:
                        os.unlink(tmp_file.name)
                        break
                    except PermissionError:
                        time.sleep(0.1)
        
        return images
    
    def image_to_boxes(
        self,
        image: np.ndarray,
        page_num: int,
        page_height: int,
    ) -> List[BBox]:
        """
        对单张图片进行 OCR 和版面分析
        
        Args:
            image: 图片 (numpy array)
            page_num: 页码
            page_height: 页面高度
            
        Returns:
            BBox 列表
        """
        boxes = []
        
        # 步骤1：版面分析（找出各区域的位置和类型）
        regions = []
        if self.enable_layout:
            try:
                # PPStructure 返回格式: [{'type': 'text', 'bbox': [x1,y1,x2,y2], ...}]
                layout_result = self.layout_engine(image)
                if layout_result:
                    for item in layout_result:
                        bbox = item.get('bbox', [0, 0, 0, 0])
                        region_type = item.get('type', 'text')
                        regions.append({
                            'bbox': bbox,
                            'type': region_type,
                        })
            except Exception as e:
                print(f"版面分析失败 (Page {page_num}): {e}")
        
        # 如果没有版面分析结果，对整个页面进行 OCR
        if not regions:
            # 整页作为一个区域
            h, w = image.shape[:2]
            regions = [{'bbox': [0, 0, w, h], 'type': 'text'}]
        
        # 步骤2：对每个区域进行 OCR
        for region in regions:
            bbox = region['bbox']
            region_type = region['type']
            
            # 确保 bbox 坐标有效
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            if x1 >= x2 or y1 >= y2:
                continue
            
            # 裁剪区域图像
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            
            # OCR 识别
            try:
                ocr_result = self.ocr.ocr(cropped, cls=True)
                text = self._extract_text_from_ocr(ocr_result)
            except Exception as e:
                print(f"OCR 失败 (Page {page_num}): {e}")
                text = ""
            
            if not text.strip():
                continue
            
            # 创建 BBox
            box = BBox(
                page_number=page_num,
                x0=float(x1),
                x1=float(x2),
                top=float(y1),
                bottom=float(y2),
                text=text,
                layout_type=self._map_layout_type(region_type),
            )
            
            # 如果是表格，进行表格识别
            if region_type == 'table' and self.enable_table and self.table_engine:
                table_html = self._recognize_table(cropped)
                if table_html:
                    box.text = table_html
                    box.layout_type = 'table'
            
            boxes.append(box)
        
        return boxes
    
    def _extract_text_from_ocr(self, ocr_result) -> str:
        """
        从 OCR 结果中提取文本
        
        Args:
            ocr_result: PaddleOCR 的识别结果
            
        Returns:
            提取的文本
        """
        if not ocr_result or not ocr_result[0]:
            return ""
        
        texts = []
        for line in ocr_result[0]:
            if line and len(line) >= 2:
                text = line[1][0] if isinstance(line[1], (list, tuple)) else line[1]
                if text:
                    texts.append(text)
        
        return "\n".join(texts)
    
    def _recognize_table(self, image: np.ndarray) -> Optional[str]:
        """
        识别表格并返回 HTML 格式
        
        Args:
            image: 表格区域的图片
            
        Returns:
            HTML 格式的表格，或 None
        """
        try:
            # 使用 PaddleOCR 的表格识别
            result = self.table_engine(image)
            if result and len(result) > 0:
                # 提取表格 HTML
                if 'html' in result[0]:
                    return result[0]['html']
                elif 'res' in result[0] and 'html' in result[0]['res']:
                    return result[0]['res']['html']
        except Exception as e:
            print(f"表格识别失败: {e}")
        
        return None
    
    def _map_layout_type(self, region_type: str) -> str:
        """
        映射版面类型到 DeepDoc 兼容的格式
        
        Args:
            region_type: 原始类型
            
        Returns:
            映射后的类型
        """
        type_map = {
            'title': 'title',
            'text': 'text',
            'table': 'table',
            'figure': 'figure',
            'picture': 'figure',
            'image': 'figure',
            'header': 'header',
            'footer': 'footer',
            'caption': 'caption',
            'reference': 'reference',
            'equation': 'equation',
        }
        return type_map.get(region_type.lower(), 'text')
    
    def _merge_horizontal_boxes(self, boxes: List[BBox]) -> List[BBox]:
        """
        水平合并相邻的文本框（同一行）
        
        Args:
            boxes: BBox 列表
            
        Returns:
            合并后的 BBox 列表
        """
        if len(boxes) <= 1:
            return boxes
        
        merged = []
        i = 0
        while i < len(boxes):
            current = boxes[i]
            j = i + 1
            
            # 查找同行且相邻的框
            while j < len(boxes):
                next_box = boxes[j]
                
                # 不同页面，不合并
                if current.page_number != next_box.page_number:
                    break
                
                # 垂直距离判断（是否在同一行）
                y_overlap = min(current.bottom, next_box.bottom) - max(current.top, next_box.top)
                if y_overlap < 0:
                    # 不在同一行，停止合并
                    break
                
                # 水平距离判断（是否相邻）
                x_gap = next_box.x0 - current.x1
                avg_height = (current.bottom - current.top + next_box.bottom - next_box.top) / 2
                
                if x_gap <= avg_height * 0.5:
                    # 合并
                    current.x1 = max(current.x1, next_box.x1)
                    current.text += " " + next_box.text
                    j += 1
                else:
                    break
            
            merged.append(current)
            i = j
        
        return merged
    
    def _merge_vertical_boxes(self, boxes: List[BBox]) -> List[BBox]:
        """
        垂直合并上下相关的文本框（同一段落）
        
        Args:
            boxes: BBox 列表
            
        Returns:
            合并后的 BBox 列表
        """
        if len(boxes) <= 1:
            return boxes
        
        merged = []
        i = 0
        
        while i < len(boxes):
            current = boxes[i]
            j = i + 1
            
            while j < len(boxes):
                next_box = boxes[j]
                
                # 不同页面，不合并
                if current.page_number != next_box.page_number:
                    break
                
                # 不同类型，不合并
                if current.layout_type != next_box.layout_type:
                    break
                
                # 计算垂直距离
                v_gap = next_box.top - current.bottom
                avg_height = (current.bottom - current.top + next_box.bottom - next_box.top) / 2
                
                # 水平重叠度判断
                x_overlap = min(current.x1, next_box.x1) - max(current.x0, next_box.x0)
                x_overlap_ratio = x_overlap / min(current.x1 - current.x0, next_box.x1 - next_box.x0) if x_overlap > 0 else 0
                
                # 合并条件：
                # 1. 垂直距离小于 2 倍行高
                # 2. 水平重叠度大于 30%
                if v_gap <= avg_height * 2 and x_overlap_ratio > 0.3:
                    # 合并
                    current.bottom = next_box.bottom
                    current.x0 = min(current.x0, next_box.x0)
                    current.x1 = max(current.x1, next_box.x1)
                    
                    # 判断是否需要加空格
                    if current.text and next_box.text:
                        if re.match(r'[a-zA-Z0-9]$', current.text[-1]) and re.match(r'[a-zA-Z0-9]', next_box.text[0]):
                            current.text += " "
                    current.text += next_box.text
                    j += 1
                else:
                    break
            
            merged.append(current)
            i = j
        
        return merged
    
    def _sort_by_reading_order(self, boxes: List[BBox]) -> List[BBox]:
        """
        按阅读顺序排序（先按列，再按行）
        
        Args:
            boxes: BBox 列表
            
        Returns:
            排序后的 BBox 列表
        """
        # 按页码分组
        pages = {}
        for box in boxes:
            if box.page_number not in pages:
                pages[box.page_number] = []
            pages[box.page_number].append(box)
        
        # 对每页进行列检测和排序
        sorted_boxes = []
        
        for page_num in sorted(pages.keys()):
            page_boxes = pages[page_num]
            
            # 简单的列检测：按 x0 坐标聚类
            x0s = [box.x0 for box in page_boxes]
            if len(x0s) > 1:
                # 使用简单的阈值判断列数
                x0s_sorted = sorted(x0s)
                gaps = [x0s_sorted[i+1] - x0s_sorted[i] for i in range(len(x0s_sorted)-1)]
                avg_gap = sum(gaps) / len(gaps) if gaps else 0
                
                # 如果有明显的列间隙，认为是多列布局
                col_threshold = avg_gap * 2 if avg_gap > 0 else float('inf')
                
                # 分配列 ID
                for box in page_boxes:
                    # 找到所属列
                    col_id = 0
                    for i, x in enumerate(x0s_sorted):
                        if abs(box.x0 - x) < col_threshold:
                            col_id = i
                            break
                    box.col_id = col_id
            else:
                for box in page_boxes:
                    box.col_id = 0
            
            # 按列排序，每列内按 top 排序
            page_boxes.sort(key=lambda b: (b.col_id, b.top, b.x0))
            sorted_boxes.extend(page_boxes)
        
        return sorted_boxes
    
    def parse(
        self,
        pdf_data: bytes,
        callback: Optional[callable] = None,
    ) -> Tuple[List[BBox], str]:
        """
        解析 PDF 文档
        
        Args:
            pdf_data: PDF 文件的二进制数据
            callback: 进度回调函数 callback(progress, message)
            
        Returns:
            (BBox列表, Markdown格式文本)
        """
        if callback:
            callback(0.05, "正在转换 PDF 为图片...")
        
        # 步骤1：PDF 转图片
        images = self.pdf_to_images(pdf_data)
        total_pages = len(images)
        
        if callback:
            callback(0.15, f"已转换 {total_pages} 页图片")
        
        # 步骤2：逐页 OCR 和版面分析
        self.boxes = []
        page_heights = []
        cumulative_height = 0
        
        for idx, img in enumerate(images):
            page_num = idx + 1
            page_height = img.shape[0]
            page_heights.append(page_height)
            
            if callback:
                progress = 0.15 + (idx / total_pages) * 0.6
                callback(progress, f"正在处理第 {page_num}/{total_pages} 页...")
            
            # OCR 和版面分析
            page_boxes = self.image_to_boxes(img, page_num, page_height)
            
            # 调整坐标（加入累计高度）
            for box in page_boxes:
                box.top += cumulative_height
                box.bottom += cumulative_height
                self.boxes.append(box)
            
            cumulative_height += page_height
        
        if callback:
            callback(0.75, "正在合并文本块...")
        
        # 步骤3：文本合并
        self.boxes = self._merge_horizontal_boxes(self.boxes)
        self.boxes = self._merge_vertical_boxes(self.boxes)
        
        if callback:
            callback(0.85, "正在排序...")
        
        # 步骤4：按阅读顺序排序
        self.boxes = self._sort_by_reading_order(self.boxes)
        
        if callback:
            callback(0.95, "正在生成 Markdown...")
        
        # 步骤5：生成 Markdown
        markdown = self._to_markdown()
        
        if callback:
            callback(1.0, "解析完成！")
        
        return self.boxes, markdown
    
    def _to_markdown(self) -> str:
        """
        将解析结果转换为 Markdown 格式
        
        Returns:
            Markdown 文本
        """
        md_lines = []
        
        for box in self.boxes:
            if not box.text:
                continue
            
            text = box.text.strip()
            
            if box.layout_type == 'title':
                # 标题：添加 ## 前缀
                md_lines.append(f"\n## {text}\n")
            elif box.layout_type == 'table':
                # 表格：已经是 HTML 或 Markdown 格式
                md_lines.append(text)
                md_lines.append("")
            elif box.layout_type == 'figure':
                # 图片描述
                md_lines.append(f"\n![Image]({text})\n")
            else:
                # 普通文本
                md_lines.append(text)
        
        return "\n".join(md_lines)
    
    def get_boxes_json(self) -> List[Dict]:
        """
        获取 JSON 格式的解析结果
        """
        return [box.to_dict() for box in self.boxes]


def main():
    """
    使用示例
    """
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python mini_deepdoc.py <pdf_file_path>")
        print("示例: python mini_deepdoc.py document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # 读取 PDF 文件
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    
    # 创建解析器
    parser = MiniDeepDocParser(
        lang='ch',          # 中文
        use_gpu=False,      # 使用 CPU（如果有 GPU 可改为 True）
        enable_layout=True, # 启用版面分析
        enable_table=True,  # 启用表格识别
        dpi=150,            # 图片分辨率
    )
    
    # 定义进度回调
    def on_progress(progress, message):
        print(f"[{progress*100:.1f}%] {message}")
    
    print(f"开始解析: {pdf_path}")
    
    # 执行解析
    boxes, markdown = parser.parse(pdf_data, callback=on_progress)
    
    print(f"\n解析完成！共识别 {len(boxes)} 个文本块")
    
    # 打印前 10 个块的信息
    print("\n前 10 个文本块:")
    for i, box in enumerate(boxes[:10]):
        print(f"{i+1}. [Page {box.page_number}, {box.layout_type}] {box.text[:50]}...")
    
    # 输出 Markdown 到文件
    output_path = pdf_path.replace('.pdf', '_output.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    print(f"\nMarkdown 输出已保存到: {output_path}")
    
    # 输出 JSON 到文件
    import json
    json_path = pdf_path.replace('.pdf', '_output.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(parser.get_boxes_json(), f, ensure_ascii=False, indent=2)
    print(f"JSON 输出已保存到: {json_path}")


if __name__ == "__main__":
    main()