"""
文件格式转换服务：将 Word / TXT / JSON / PPT / Excel 转换为 PDF
依赖 LibreOffice（headless 模式）
"""
import os
import subprocess
import tempfile
import shutil
import base64
import re
import urllib.parse

import requests

from common.config import (
    LIBREOFFICE_CONVERT_TIMEOUT,
    ONLINE_PREVIEW_BASE_URL,
    SUPPORTED_CONVERT_EXTENSIONS,
)

# 支持的源格式 → 分组
SUPPORTED_EXTENSIONS = SUPPORTED_CONVERT_EXTENSIONS


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
            timeout=LIBREOFFICE_CONVERT_TIMEOUT,
        )
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)


def _convert_to_pdf_via_online_preview(source_url: str, output_dir: str, output_filename: str = None) -> str:
    encoded_source_url = base64.b64encode(source_url.encode("utf-8")).decode("utf-8")
    preview_url = f"{ONLINE_PREVIEW_BASE_URL.rstrip('/')}/onlinePreview?url={encoded_source_url}"
    preview_resp = requests.get(preview_url, timeout=120)
    preview_resp.raise_for_status()

    content_type = preview_resp.headers.get("content-type", "")
    if "application/pdf" in content_type or preview_resp.content.startswith(b"%PDF-"):
        pdf_path = os.path.join(output_dir, output_filename or "online_preview.pdf")
        with open(pdf_path, "wb") as f:
            f.write(preview_resp.content)
        if _is_pdf_valid(pdf_path):
            return pdf_path
        raise RuntimeError(f"在线预览服务返回的 PDF 异常: {pdf_path}")

    match = re.search(r"var\s+url\s*=\s*'([^']+\.pdf)'", preview_resp.text)
    if not match:
        raise RuntimeError("在线预览服务未返回可下载的 PDF 地址")

    pdf_url = match.group(1)
    encoded_pdf_url = base64.b64encode(pdf_url.encode("utf-8")).decode("utf-8")
    download_url = (
        f"{ONLINE_PREVIEW_BASE_URL.rstrip('/')}/getCorsFile?"
        f"urlPath={urllib.parse.quote(encoded_pdf_url)}&key=false"
    )
    download_resp = requests.get(download_url, timeout=120)
    download_resp.raise_for_status()
    if not download_resp.content.startswith(b"%PDF-"):
        raise RuntimeError("在线预览服务下载结果不是 PDF")

    pdf_path = os.path.join(output_dir, output_filename or os.path.basename(urllib.parse.urlparse(pdf_url).path))
    with open(pdf_path, "wb") as f:
        f.write(download_resp.content)

    if _is_pdf_valid(pdf_path):
        return pdf_path
    raise RuntimeError(f"在线预览服务生成的 PDF 异常: {pdf_path}")


def convert_to_pdf(source_path: str, output_dir: str = None, source_url: str = None) -> str:
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

    base_name = os.path.splitext(os.path.basename(source_path))[0]
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")

    if source_url:
        try:
            return _convert_to_pdf_via_online_preview(
                source_url=source_url,
                output_dir=output_dir,
                output_filename=f"{base_name}.pdf",
            )
        except Exception as e:
            print(f"在线预览服务转换失败，回退 LibreOffice: {e}")

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
