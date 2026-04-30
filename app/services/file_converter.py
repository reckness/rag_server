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


def _is_pdf_valid(pdf_path: str) -> bool:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()
        return page_count > 0
    except Exception:
        return False


def _run_libreoffice_convert(source_path: str, output_dir: str, target_format: str) -> subprocess.CompletedProcess:
    profile_dir = tempfile.mkdtemp(prefix="lo_profile_")
    profile_uri = f"file://{profile_dir}"
    cmd = [
        "libreoffice",
        "--headless",
        "--nologo",
        "--nolockcheck",
        "--nodefault",
        "--nofirststartwizard",
        f"-env:UserInstallation={profile_uri}",
        "--convert-to", target_format,
        "--outdir", output_dir,
        source_path,
    ]
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)


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

    result = _run_libreoffice_convert(
        source_path=source_path,
        output_dir=output_dir,
        target_format="pdf:writer_pdf_Export",
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

    if _is_pdf_valid(pdf_path):
        return pdf_path

    if ext == ".doc":
        fallback_dir = tempfile.mkdtemp(prefix="lo_doc_fallback_")
        try:
            docx_result = _run_libreoffice_convert(
                source_path=source_path,
                output_dir=fallback_dir,
                target_format="docx",
            )
            fallback_docx_path = os.path.join(
                fallback_dir,
                f"{os.path.splitext(os.path.basename(source_path))[0]}.docx",
            )
            if docx_result.returncode == 0 and os.path.exists(fallback_docx_path):
                retry_result = _run_libreoffice_convert(
                    source_path=fallback_docx_path,
                    output_dir=output_dir,
                    target_format="pdf:writer_pdf_Export",
                )
                if retry_result.returncode == 0 and os.path.exists(pdf_path) and _is_pdf_valid(pdf_path):
                    return pdf_path
        finally:
            shutil.rmtree(fallback_dir, ignore_errors=True)

    raise RuntimeError(
        f"生成的 PDF 异常（无法正常打开或页数为 0）: {pdf_path}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
