"""
文件格式转换服务：将 Word / TXT / JSON / PPT / Excel 转换为 PDF
依赖 LibreOffice（headless 模式）
"""
import os
import subprocess
import tempfile
import shutil


# 支持的源格式 → 分组
SUPPORTED_EXTENSIONS = {
    ".doc", ".docx",       # Word
    ".txt", ".json",       # 文本
    ".ppt", ".pptx",       # PPT
    ".xls", ".xlsx",       # Excel
}


def is_supported(filename: str) -> bool:
    """判断文件扩展名是否支持转换"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def convert_to_pdf(source_path: str, output_dir: str = None) -> str:
    """
    将文件转换为 PDF。

    参数:
        source_path: 源文件路径
        output_dir:  PDF 输出目录（默认与源文件同目录）

    返回:
        str: 生成的 PDF 文件路径

    异常:
        ValueError: 不支持的文件格式
        RuntimeError: LibreOffice 转换失败
    """
    ext = os.path.splitext(source_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式: {ext}")

    if output_dir is None:
        output_dir = os.path.dirname(source_path)

    os.makedirs(output_dir, exist_ok=True)

    # 使用 LibreOffice headless 模式转换
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to", "pdf",
        "--outdir", output_dir,
        source_path,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 分钟超时
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"LibreOffice 转换失败 (exit={result.returncode}): "
            f"{result.stderr or result.stdout}"
        )

    # LibreOffice 输出的 PDF 文件名 = 源文件名（去扩展名）+ .pdf
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")

    if not os.path.exists(pdf_path):
        raise RuntimeError(
            f"转换完成但未找到输出文件: {pdf_path}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    return pdf_path
