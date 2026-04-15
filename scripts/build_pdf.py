#!/usr/bin/env python3
"""
将 docs/ 和 interview/ 目录下的 Markdown 文件转换为 PDF。

使用方法（两种方案）:

方案 A（推荐）—— 使用 md2pdf:
    pip install md2pdf
    python scripts/build_pdf.py

方案 B —— 使用 weasyprint:
    pip install markdown weasyprint
    python scripts/build_pdf.py --engine weasyprint

输出:
    dist/pdf/ 目录下生成所有 PDF 文件
"""

import argparse
import os
import sys

# Windows 上 weasyprint 需要显式加载 GTK DLL 目录
if sys.platform == "win32":
    _gtk_bin = r"E:\GTK3-Runtime Win64\bin"
    if os.path.isdir(_gtk_bin):
        os.add_dll_directory(_gtk_bin)
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
INTERVIEW_DIR = ROOT / "interview"
OUTPUT_DIR = ROOT / "dist" / "pdf"

CSS = """
@page { size: A4; margin: 2cm; }
body {
    font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #1e293b;
}
h1 { font-size: 20pt; color: #4f46e5; border-bottom: 2px solid #4f46e5; padding-bottom: 6pt; margin-top: 24pt; }
h2 { font-size: 16pt; color: #334155; border-left: 4px solid #4f46e5; padding-left: 8pt; margin-top: 20pt; }
h3 { font-size: 13pt; margin-top: 16pt; }
code { background: #f1f5f9; padding: 1pt 4pt; border-radius: 3pt; font-size: 9.5pt; }
pre { background: #1e293b; color: #e2e8f0; padding: 12pt; border-radius: 6pt; font-size: 9pt; line-height: 1.4; overflow-wrap: break-word; white-space: pre-wrap; }
pre code { background: none; color: inherit; padding: 0; }
table { width: 100%; border-collapse: collapse; font-size: 10pt; margin: 8pt 0; }
th, td { border: 1px solid #e2e8f0; padding: 5pt 8pt; text-align: left; }
th { background: #eef2ff; }
blockquote { border-left: 3px solid #4f46e5; background: #eef2ff; padding: 8pt 12pt; margin: 8pt 0; }
img { max-width: 100%; height: auto; }
strong { color: #4f46e5; }
"""


def build_with_weasyprint():
    try:
        import markdown
        from markdown.extensions.fenced_code import FencedCodeExtension
        from markdown.extensions.tables import TableExtension
        from weasyprint import HTML as WHTML
    except ImportError:
        print("请安装依赖: pip install markdown weasyprint")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    for source_dir in [DOCS_DIR, INTERVIEW_DIR]:
        if not source_dir.exists():
            continue
        for md_path in sorted(source_dir.glob("*.md")):
            md_text = md_path.read_text(encoding="utf-8")
            html_body = markdown.markdown(
                md_text,
                extensions=[FencedCodeExtension(), TableExtension()],
            )
            full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body></html>"""

            pdf_path = OUTPUT_DIR / (md_path.stem + ".pdf")
            WHTML(string=full_html, base_url=str(ROOT)).write_pdf(str(pdf_path))
            count += 1
            print(f"  [ok] {pdf_path.name}")

    print(f"\n完成：已生成 {count} 个 PDF 文件到 {OUTPUT_DIR}")


def build_with_md2pdf():
    try:
        from md2pdf.core import md2pdf as convert
    except ImportError:
        print("请安装依赖: pip install md2pdf")
        print("或使用 weasyprint: python scripts/build_pdf.py --engine weasyprint")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    for source_dir in [DOCS_DIR, INTERVIEW_DIR]:
        if not source_dir.exists():
            continue
        for md_path in sorted(source_dir.glob("*.md")):
            pdf_path = OUTPUT_DIR / (md_path.stem + ".pdf")
            try:
                convert(
                    str(pdf_path),
                    md_file_path=str(md_path),
                    css_file_path=None,
                    base_url=str(ROOT),
                )
                count += 1
                print(f"  [ok] {pdf_path.name}")
            except Exception as e:
                print(f"  [fail] {md_path.name}: {e}")

    print(f"\n完成：已生成 {count} 个 PDF 文件到 {OUTPUT_DIR}")


def build_combined_pdf():
    """生成一个合并的完整 PDF（所有课程 + 面试题）"""
    try:
        import markdown
        from markdown.extensions.fenced_code import FencedCodeExtension
        from markdown.extensions.tables import TableExtension
        from weasyprint import HTML as WHTML
    except ImportError:
        print("合并 PDF 需要 weasyprint: pip install markdown weasyprint")
        return

    all_html = []
    for source_dir in [DOCS_DIR, INTERVIEW_DIR]:
        if not source_dir.exists():
            continue
        for md_path in sorted(source_dir.glob("*.md")):
            md_text = md_path.read_text(encoding="utf-8")
            html = markdown.markdown(
                md_text,
                extensions=[FencedCodeExtension(), TableExtension()],
            )
            all_html.append(f'<div style="page-break-before: always;">{html}</div>')

    combined = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
<div style="text-align:center; padding: 200pt 0;">
<h1 style="font-size: 32pt; border: none;">MiniMind 面试学习指南</h1>
<p style="font-size: 14pt; color: #64748b; margin-top: 20pt;">从零开始学习 64M 参数大语言模型</p>
<p style="font-size: 12pt; color: #94a3b8; margin-top: 40pt;">面试导向 · 小白友好 · 全链路覆盖</p>
</div>
{"".join(all_html)}
</body></html>"""

    pdf_path = OUTPUT_DIR / "MiniMind面试学习指南-完整版.pdf"
    WHTML(string=combined, base_url=str(ROOT)).write_pdf(str(pdf_path))
    print(f"完整版 PDF: {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Markdown to PDF converter")
    parser.add_argument(
        "--engine",
        choices=["md2pdf", "weasyprint"],
        default="weasyprint",
        help="PDF 生成引擎 (默认: weasyprint)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="同时生成合并版完整 PDF",
    )
    args = parser.parse_args()

    if args.engine == "weasyprint":
        build_with_weasyprint()
    else:
        build_with_md2pdf()

    if args.combined:
        build_combined_pdf()
