#!/usr/bin/env python3
"""
将 docs/ 和 interview/ 目录下的 Markdown 文件转换为带样式的 HTML 页面。

使用方法:
    pip install markdown pygments
    python scripts/build_html.py

输出:
    dist/html/ 目录下生成所有 HTML 文件和 index.html 首页
"""

import os
import re
import shutil
from pathlib import Path

try:
    import markdown
    from markdown.extensions.codehilite import CodeHiliteExtension
    from markdown.extensions.fenced_code import FencedCodeExtension
    from markdown.extensions.tables import TableExtension
    from markdown.extensions.toc import TocExtension
except ImportError:
    print("请先安装依赖: pip install markdown pygments")
    exit(1)

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
INTERVIEW_DIR = ROOT / "interview"
ASSETS_DIR = ROOT / "assets"
OUTPUT_DIR = ROOT / "dist" / "html"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - MiniMind 面试学习指南</title>
    <style>
        :root {{
            --primary: #4f46e5;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-secondary: #64748b;
            --border: #e2e8f0;
            --code-bg: #f1f5f9;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.8;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1.5rem;
        }}
        nav {{
            background: var(--primary);
            color: white;
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        nav a {{ color: white; text-decoration: none; font-weight: 600; font-size: 1.1rem; }}
        nav a:hover {{ opacity: 0.85; }}
        nav .back {{ margin-right: 1rem; opacity: 0.8; }}
        h1 {{ font-size: 2rem; margin: 1.5rem 0 1rem; color: var(--primary); border-bottom: 3px solid var(--primary); padding-bottom: 0.5rem; }}
        h2 {{ font-size: 1.5rem; margin: 2rem 0 0.8rem; color: var(--text); border-left: 4px solid var(--primary); padding-left: 0.8rem; }}
        h3 {{ font-size: 1.2rem; margin: 1.5rem 0 0.5rem; }}
        p {{ margin: 0.8rem 0; }}
        blockquote {{
            border-left: 4px solid var(--primary);
            background: #eef2ff;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }}
        code {{
            background: var(--code-bg);
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.9em;
            font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
        }}
        pre {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 1.2rem;
            border-radius: 8px;
            overflow-x: auto;
            margin: 1rem 0;
            line-height: 1.5;
        }}
        pre code {{ background: none; padding: 0; color: inherit; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.95rem;
        }}
        th, td {{
            border: 1px solid var(--border);
            padding: 0.6rem 1rem;
            text-align: left;
        }}
        th {{ background: #eef2ff; font-weight: 600; }}
        tr:hover {{ background: #f8fafc; }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        hr {{ border: none; border-top: 2px solid var(--border); margin: 2rem 0; }}
        ul, ol {{ padding-left: 1.5rem; margin: 0.5rem 0; }}
        li {{ margin: 0.3rem 0; }}
        strong {{ color: var(--primary); }}
        .footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
            border-top: 1px solid var(--border);
            margin-top: 3rem;
        }}
        @media (max-width: 768px) {{
            .container {{ padding: 1rem; }}
            h1 {{ font-size: 1.5rem; }}
        }}
    </style>
</head>
<body>
    <nav>
        <a class="back" href="index.html">← 首页</a>
        <a href="index.html">MiniMind 面试学习指南</a>
    </nav>
    <div class="container">
        {content}
    </div>
    <div class="footer">
        <p>MiniMind 面试学习指南 | 基于 <a href="https://github.com/jingyaogong/minimind" style="color:var(--primary)">MiniMind</a> 项目</p>
    </div>
</body>
</html>"""

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MiniMind 面试学习指南</title>
    <style>
        :root {{ --primary: #4f46e5; --bg: #f8fafc; --card-bg: #ffffff; --text: #1e293b; --text-secondary: #64748b; --border: #e2e8f0; }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif; background: var(--bg); color: var(--text); }}
        .hero {{ background: linear-gradient(135deg, #4f46e5, #7c3aed); color: white; padding: 4rem 2rem; text-align: center; }}
        .hero h1 {{ font-size: 2.5rem; margin-bottom: 1rem; }}
        .hero p {{ font-size: 1.2rem; opacity: 0.9; max-width: 600px; margin: 0 auto; }}
        .container {{ max-width: 1000px; margin: 0 auto; padding: 2rem; }}
        .section-title {{ font-size: 1.5rem; margin: 2rem 0 1rem; color: var(--primary); border-left: 4px solid var(--primary); padding-left: 1rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            transition: all 0.2s;
            text-decoration: none;
            color: var(--text);
            display: block;
        }}
        .card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.1); border-color: var(--primary); }}
        .card h3 {{ color: var(--primary); margin-bottom: 0.5rem; font-size: 1.1rem; }}
        .card p {{ color: var(--text-secondary); font-size: 0.9rem; line-height: 1.5; }}
        .badge {{ display: inline-block; background: #eef2ff; color: var(--primary); padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-bottom: 0.5rem; }}
        .footer {{ text-align: center; padding: 2rem; color: var(--text-secondary); font-size: 0.9rem; }}
    </style>
</head>
<body>
    <div class="hero">
        <h1>MiniMind 面试学习指南</h1>
        <p>从零开始学习 64M 参数大语言模型的完整训练链路，面试导向，小白友好</p>
    </div>
    <div class="container">
        <h2 class="section-title">课程体系（24 课）</h2>
        <div class="grid">{docs_cards}</div>
        <h2 class="section-title">面试宝库（10 篇）</h2>
        <div class="grid">{interview_cards}</div>
    </div>
    <div class="footer">
        <p>MiniMind 面试学习指南 | <a href="https://github.com/bcefghj/learn-minimind" style="color:var(--primary)">GitHub</a> | <a href="https://github.com/jingyaogong/minimind" style="color:var(--primary)">MiniMind 原项目</a></p>
    </div>
</body>
</html>"""


def md_to_html(md_text: str) -> str:
    extensions = [
        FencedCodeExtension(),
        CodeHiliteExtension(css_class="highlight", linenums=False),
        TableExtension(),
        TocExtension(permalink=False),
        "markdown.extensions.nl2br",
    ]
    return markdown.markdown(md_text, extensions=extensions)


def get_title(md_text: str) -> str:
    for line in md_text.splitlines():
        if line.startswith("# "):
            return line.lstrip("# ").strip()
    return "Untitled"


def fix_image_paths(html: str, source_dir: str) -> str:
    """Fix relative image paths for HTML output."""
    html = re.sub(
        r'src="\.\./(assets/[^"]+)"',
        lambda m: f'src="{m.group(1)}"',
        html,
    )
    return html


def build():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if ASSETS_DIR.exists():
        dst_assets = OUTPUT_DIR / "assets"
        if dst_assets.exists():
            shutil.rmtree(dst_assets)
        shutil.copytree(ASSETS_DIR, dst_assets)

    docs_cards = []
    interview_cards = []

    for source_dir, cards_list, badge in [
        (DOCS_DIR, docs_cards, "课程"),
        (INTERVIEW_DIR, interview_cards, "面试"),
    ]:
        if not source_dir.exists():
            continue
        md_files = sorted(source_dir.glob("*.md"))
        for md_path in md_files:
            md_text = md_path.read_text(encoding="utf-8")
            title = get_title(md_text)
            html_content = md_to_html(md_text)
            html_content = fix_image_paths(html_content, str(source_dir))

            html_filename = md_path.stem + ".html"
            html_page = HTML_TEMPLATE.format(title=title, content=html_content)
            (OUTPUT_DIR / html_filename).write_text(html_page, encoding="utf-8")

            desc = ""
            for line in md_text.splitlines():
                if line.startswith(">"):
                    desc = line.lstrip("> ").strip()
                    break

            cards_list.append(
                f'<a class="card" href="{html_filename}">'
                f'<span class="badge">{badge}</span>'
                f"<h3>{title}</h3>"
                f"<p>{desc[:80]}</p>"
                f"</a>"
            )

    index_html = INDEX_TEMPLATE.format(
        docs_cards="\n".join(docs_cards),
        interview_cards="\n".join(interview_cards),
    )
    (OUTPUT_DIR / "index.html").write_text(index_html, encoding="utf-8")

    total = len(docs_cards) + len(interview_cards)
    print(f"✅ 已生成 {total} 个 HTML 文件到 {OUTPUT_DIR}")
    print(f"   首页: {OUTPUT_DIR / 'index.html'}")


if __name__ == "__main__":
    build()
