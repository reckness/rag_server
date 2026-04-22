import sys
import os
from markitdown import MarkItDown
import pymupdf


def pdf_to_md_markitdown(pdf_path, output_path=None):
    """方式1: 使用 MarkItDown 将PDF转换为Markdown"""
    md = MarkItDown()
    result = md.convert(pdf_path)
    text = result.text_content

    if output_path is None:
        base = os.path.splitext(pdf_path)[0]
        output_path = base + "_markitdown.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[SUCCESS] MarkItDown 输出已保存至: {output_path}")
    print(f"内容长度: {len(text)} 字符")
    print(f"\n===== 前 500 字符预览 =====\n{text[:500]}")


def pdf_to_md_pymupdf(pdf_path, output_path=None):
    """方式2: 使用 PyMuPDF 逐页提取PDF文本"""
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        page_text = page.get_text("text")
        pages.append(page_text)
    doc.close()

    # 组合为 Markdown：每页用分隔符标记
    md_parts = []
    for i, text in enumerate(pages):
        md_parts.append(f"<!-- Page {i+1} -->\n{text.strip()}")
    full_text = "\n\n---\n\n".join(md_parts)

    if output_path is None:
        base = os.path.splitext(pdf_path)[0]
        output_path = base + "_pymupdf.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"[SUCCESS] PyMuPDF 输出已保存至: {output_path}")
    print(f"总页数: {len(pages)}")
    print(f"内容长度: {len(full_text)} 字符")
    print(f"\n===== 前 500 字符预览 =====\n{full_text[:500]}")


def main():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "./pdf/珠三角电子信息产业集群创新网络演化及其机理研究_王炜.pdf"

    if not os.path.isfile(pdf_path):
        print(f"[ERROR] 文件不存在: {pdf_path}")
        return

    print(f"PDF 文件: {pdf_path}")
    print("请选择转换方式:")
    print("  1 - MarkItDown")
    print("  2 - PyMuPDF")
    print("  3 - 两种都运行")

    choice = input("输入选择 (1/2/3): ").strip()

    if choice == "1":
        pdf_to_md_markitdown(pdf_path)
    elif choice == "2":
        pdf_to_md_pymupdf(pdf_path)
    elif choice == "3":
        pdf_to_md_markitdown(pdf_path)
        print("\n" + "=" * 60 + "\n")
        pdf_to_md_pymupdf(pdf_path)
    else:
        print(f"[ERROR] 无效选择: {choice}")


if __name__ == "__main__":
    main()
