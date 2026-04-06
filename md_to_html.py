#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 Markdown 转为与 RL_PO.html 同主题的静态 HTML（MathJax + 浅蓝配色）。
支持：标题/段落/列表/链接/代码块/引用、GFM 表格；无表格时按文章排版。
转换前会抽离 $...$ / $$...$$，避免 Markdown 把下划线 _ 当成强调而弄坏 LaTeX。

用法:
  python md_to_html.py --input KL.md
  python md_to_html.py --input RL_PO.md -o out.html
  python md_to_html.py --batch . --open-browser
  python md_to_html.py --batch ./notes --nav-out index.html --site-title "课程笔记"
  python md_to_html.py --batch ./notes --flat   # 仅根目录 .md，不递归子文件夹

批量递归时：根目录为 --nav-out 指定的总目录；每个含内容的子文件夹内会生成 index.html，只索引该文件夹下的页面与下一级子文件夹。

依赖: pip install markdown beautifulsoup4
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import webbrowser
from pathlib import Path

try:
    import markdown
except ImportError:
    print("请先安装: pip install markdown beautifulsoup4", file=sys.stderr)
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("请先安装: pip install beautifulsoup4", file=sys.stderr)
    sys.exit(1)


def extract_title_from_md(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip()
    return None


def strip_first_h1_html(fragment: str) -> str:
    soup = BeautifulSoup(fragment, "html.parser")
    h1 = soup.find("h1")
    if h1:
        h1.decompose()
    # 若只剩空白包装，保留结构
    return str(soup)


# Markdown 会把 `_..._` 当成强调，破坏 \mathbb{E}_{p(x)} 等；先整体替换公式再 convert
_MATH_PH = "MJXPH{:06d}END"


def protect_math(md_text: str) -> tuple[str, list[tuple[str, str]]]:
    """先抽离 $$...$$ 再抽离 $...$，返回 (改写后的 md, [('d'|'i', tex), ...])。"""
    vault: list[tuple[str, str]] = []

    def repl_display(m) -> str:
        vault.append(("d", m.group(1)))
        return f"\n\n{_MATH_PH.format(len(vault) - 1)}\n\n"

    s = re.sub(r"\$\$([\s\S]*?)\$\$", repl_display, md_text)

    def repl_inline(m) -> str:
        vault.append(("i", m.group(1)))
        return _MATH_PH.format(len(vault) - 1)

    # 允许 $...$ 内换行（单行续行），但不跨空行；避免 [^\$\n] 拆坏多行公式
    _inline = r"(?<!\$)\$(?!\$)([^$\n]*(?:\n[^$\n]*)*?)\$(?!\$)"
    s = re.sub(_inline, repl_inline, s)
    return s, vault


def _esc_tex_for_html_body(tex: str) -> str:
    """公式插回 HTML 前转义 & < >，避免 BeautifulSoup/浏览器把 0<...<G 等当成标签。"""
    return (
        tex.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def restore_math(html: str, vault: list[tuple[str, str]]) -> str:
    for i, (kind, tex) in enumerate(vault):
        ph = _MATH_PH.format(i)
        safe = _esc_tex_for_html_body(tex)
        if kind == "d":
            rep = "$$" + safe + "$$"
        else:
            rep = "$" + safe + "$"
        html = html.replace(ph, rep)
    return html


def wrap_markdown_tables(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        parent = table.parent
        if parent and parent.name == "div" and "tbl-wrap-md" in (parent.get("class") or []):
            continue
        div = soup.new_tag("div", attrs={"class": "tbl-wrap tbl-wrap-md"})
        table.replace_with(div)
        div.append(table)
        table["class"] = table.get("class", []) + ["md-table"]
        for cell in table.find_all(["td", "th"]):
            cls = cell.get("class") or []
            if "formula-cell" not in cls:
                cell["class"] = cls + ["formula-cell"]
    return str(soup)


THEME_CSS = r"""
    :root {
      --bg:       #e8f2fc;
      --surface:  #e4eef9;
      --surface2: #d7e8f8;
      --border:   #a8c8ef;
      --c1: #2563eb;
      --c5: #0284c7;
      --text:  #1e3a5f;
      --muted: #4a6fa5;
      --radius: 14px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background:
        radial-gradient(ellipse 120% 80% at 50% -20%, rgba(147,197,253,.35), transparent 55%),
        var(--bg);
      color: var(--text);
      font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
      line-height: 1.7;
      padding: 2.5rem 1.5rem 4rem;
    }
    header {
      text-align: center;
      margin-bottom: 2.2rem;
      max-width: min(1920px, 98vw);
      margin-left: auto;
      margin-right: auto;
    }
    header h1 {
      font-size: 1.85rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--c1) 0%, var(--c5) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      letter-spacing: .02em;
    }
    header .subtitle {
      color: var(--muted);
      margin-top: .45rem;
      font-size: .9rem;
    }

    /* ── 正文（无表格或表格前后混排）── */
    main.md-content {
      max-width: min(1920px, 98vw);
      margin: 0 auto;
      padding: 1.2rem 0 2rem;
      font-size: .95rem;
      line-height: 1.75;
      color: var(--text);
    }
    .md-content > *:first-child { margin-top: 0; }
    .md-content h1 {
      font-size: 1.55rem;
      margin: 1.4em 0 0.55em;
      color: #1e3a8a;
      border-bottom: 2px solid var(--border);
      padding-bottom: .35rem;
    }
    .md-content h2 {
      font-size: 1.28rem;
      margin: 1.25em 0 0.5em;
      color: #1d4ed8;
    }
    .md-content h3 {
      font-size: 1.1rem;
      margin: 1.1em 0 0.45em;
      color: #2563eb;
    }
    .md-content h4, .md-content h5, .md-content h6 {
      margin: 1em 0 0.4em;
      color: var(--text);
    }
    .md-content p { margin: 0.65em 0; }
    .md-content ul, .md-content ol {
      margin: 0.6em 0 0.6em 1.35rem;
    }
    .md-content li { margin: 0.35em 0; }
    .md-content li::marker { color: var(--c5); }
    .md-content a {
      color: #0369a1;
      text-decoration: none;
      border-bottom: 1px solid rgba(3,105,161,.35);
    }
    .md-content a:hover {
      color: #0c4a6e;
      border-bottom-color: #0c4a6e;
    }
    .md-content blockquote {
      margin: 1rem 0;
      padding: .65rem 1rem .65rem 1rem;
      border-left: 4px solid var(--c1);
      background: color-mix(in srgb, var(--c1) 6%, var(--surface));
      border-radius: 0 var(--radius) var(--radius) 0;
      color: #2c4a6e;
    }
    .md-content blockquote p { margin: 0.4em 0; }
    .md-content code {
      font-family: ui-monospace, 'Cascadia Code', Consolas, monospace;
      font-size: .88em;
      background: var(--surface2);
      padding: .12em .4em;
      border-radius: 4px;
      border: 1px solid var(--border);
    }
    .md-content pre {
      margin: 1rem 0;
      padding: 1rem 1.1rem;
      overflow-x: auto;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: 0 2px 8px rgba(37,99,235,.06);
    }
    .md-content pre code {
      background: none;
      border: none;
      padding: 0;
      font-size: .84rem;
      line-height: 1.55;
    }
    .md-content hr {
      border: none;
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--border), transparent);
      margin: 2rem 0;
    }
    .md-content img {
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .md-content mjx-container[display="true"] {
      margin: 0.65em 0 !important;
    }

    /* ── Markdown 表格（GFM）── */
    .tbl-wrap {
      margin: 1.25rem auto;
      overflow-x: auto;
      border-radius: var(--radius);
      border: 1px solid var(--border);
      box-shadow:
        0 4px 6px -1px rgba(30,58,95,.06),
        0 12px 28px -4px rgba(37,99,235,.12);
    }
    .tbl-wrap-md table.md-table {
      width: 100%;
      min-width: min(100%, max-content);
      border-collapse: collapse;
      background: var(--surface);
      font-size: .88rem;
    }
    .tbl-wrap-md thead tr { background: var(--surface2); }
    .tbl-wrap-md th {
      padding: .55rem .75rem;
      text-align: left;
      font-weight: 700;
      font-size: .78rem;
      letter-spacing: .04em;
      color: var(--muted);
      border-bottom: 2px solid var(--border);
      vertical-align: top;
    }
    .tbl-wrap-md td {
      padding: .5rem .75rem;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
      line-height: 1.55;
    }
    .tbl-wrap-md tbody tr {
      transition: background .12s;
    }
    .tbl-wrap-md tbody tr:hover {
      background: rgba(219,234,254,.4);
    }
    .tbl-wrap-md tbody tr:last-child td { border-bottom: none; }
    .tbl-wrap-md .formula-cell { overflow-x: auto; }
    .tbl-wrap-md .formula-cell mjx-container[display="true"] {
      margin: 0.2em 0 !important;
    }

    /* 文末脚注块（若用 blockquote 模拟）*/
    .md-content > blockquote:last-of-type {
      margin-top: 2rem;
    }

    /* ── 子页返回目录 ── */
    nav.site-back {
      max-width: min(1920px, 98vw);
      margin: -0.8rem auto 1.25rem;
      padding: 0 0.15rem;
      font-size: .88rem;
    }
    nav.site-back a {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      color: #0369a1;
      text-decoration: none;
      font-weight: 600;
      border-bottom: 1px solid rgba(3,105,161,.3);
      padding-bottom: 0.1rem;
    }
    nav.site-back a:hover {
      color: #0c4a6e;
      border-bottom-color: #0c4a6e;
    }

    /* ── 批量目录页 ── */
    .catalog-main { padding-top: 0.5rem; }
    .catalog-hint {
      color: var(--muted);
      font-size: .88rem;
      margin-bottom: 1.1rem !important;
    }
    ul.catalog-list {
      list-style: none;
      margin: 0 !important;
      padding: 0 !important;
      display: flex;
      flex-direction: column;
      gap: 0.55rem;
    }
    ul.catalog-list li {
      margin: 0 !important;
    }
    ul.catalog-list a {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.5rem 1rem;
      padding: 0.75rem 1rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: 0 2px 6px rgba(37,99,235,.05);
      text-decoration: none !important;
      border-bottom: 1px solid var(--border) !important;
      transition: background .15s, box-shadow .15s, border-color .15s;
    }
    ul.catalog-list a:hover {
      background: #dbeafe;
      border-color: #93c5fd !important;
      box-shadow: 0 4px 14px rgba(37,99,235,.1);
    }
    ul.catalog-list .cat-t {
      font-weight: 600;
      font-size: 1.02rem;
      color: var(--text);
    }
    ul.catalog-list a:hover .cat-t { color: #1d4ed8; }
    ul.catalog-list .cat-f {
      font-size: .78rem;
      color: var(--muted);
      font-family: ui-monospace, Consolas, monospace;
      background: var(--surface2);
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      border: 1px solid var(--border);
    }
"""


# 与 RL_PO.html 等页 MathJax 一致；显式启用 textmacros，使 \text{} 内 \_ 等按文本模式解析为下划线
MATHJAX_MACROS_BLOCK = """
        macros: {
          piold: '\\\\pi_{\\\\theta_\\\\mathrm{old}}',
          pinew: '\\\\pi_{\\\\theta}'
        }"""


def build_html(
    title: str,
    subtitle: str | None,
    body_html: str,
    include_tex_macros: bool = True,
    nav_back_href: str | None = None,
    nav_back_label: str = "← 文档目录",
    main_extra_class: str = "",
) -> str:
    macros_tex = "," + MATHJAX_MACROS_BLOCK if include_tex_macros else ""

    mj_config = f"""
    MathJax = {{
      loader: {{
        load: ['[tex]/textmacros']
      }},
      tex: {{
        packages: {{'[+]': ['textmacros']}},
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        tags: 'none'{macros_tex}
      }},
      options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }}
    }};
""".strip()

    sub = f'<p class="subtitle">{subtitle}</p>' if subtitle else ""
    nav = ""
    if nav_back_href:
        nav = (
            f'  <nav class="site-back"><a href="{_esc_html(nav_back_href)}">'
            f"{_esc_html(nav_back_label)}</a></nav>\n"
        )

    main_cls = "md-content"
    if main_extra_class.strip():
        main_cls = f"{main_cls} {main_extra_class.strip()}"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{_esc_html(title)}</title>
  <script>
    {mj_config}
  </script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
{THEME_CSS}
  </style>
</head>
<body>
<header>
  <h1>{_esc_html(title)}</h1>
  {sub}
</header>
{nav}<main class="{main_cls}">
{body_html}
</main>
</body>
</html>
"""


def _esc_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def convert_md_to_body(md_text: str) -> str:
    protected, vault = protect_math(md_text)
    ext = [
        "markdown.extensions.tables",
        "markdown.extensions.fenced_code",
        "markdown.extensions.sane_lists",
    ]
    md = markdown.Markdown(extensions=ext)
    html = md.convert(protected)
    html = restore_math(html, vault)
    html = wrap_markdown_tables(html)
    return html


def render_page_from_markdown(
    md_path: Path,
    *,
    title_override: str | None = None,
    subtitle: str | None = None,
    keep_first_h1: bool = False,
    no_strip_h1: bool = False,
    nav_back_href: str | None = None,
    nav_back_label: str = "← 文档目录",
) -> tuple[str, str]:
    """读取 .md，返回 (页面标题, 完整 HTML 文档)。"""
    md_text = md_path.read_text(encoding="utf-8")
    title = title_override or extract_title_from_md(md_text) or md_path.stem
    body_html = convert_md_to_body(md_text)
    if not no_strip_h1 and not keep_first_h1:
        extracted = extract_title_from_md(md_text)
        if extracted and (title_override is None or title_override == extracted):
            body_html = strip_first_h1_html(body_html)
    full = build_html(
        title,
        subtitle,
        body_html,
        nav_back_href=nav_back_href,
        nav_back_label=nav_back_label,
    )
    return title, full


def _nav_href_from_page(nav_file: Path, page_html: Path) -> str:
    """从生成的子页面 HTML 到目录页文件的相对链接（POSIX 路径）。"""
    nav_file = nav_file.resolve()
    page_html = page_html.resolve()
    return Path(os.path.relpath(nav_file, start=page_html.parent)).as_posix()


def _dir_tree_has_output(sub_root: Path, all_out: set[Path]) -> bool:
    """sub_root 目录（含其子树）内是否存在已生成的 HTML。"""
    sub_root = sub_root.resolve()
    for o in all_out:
        try:
            o.relative_to(sub_root)
            return True
        except ValueError:
            pass
    return False


def _catalog_li(title: str, href: str) -> str:
    return (
        "<li><a "
        f'href="{_esc_html(href)}">'
        f'<span class="cat-t">{_esc_html(title)}</span>'
        f'<code class="cat-f">{_esc_html(href)}</code></a></li>'
    )


def build_folder_catalog_inner_html(
    *,
    subfolder_entries: list[tuple[str, str]],
    page_entries: list[tuple[str, str]],
    hint: str,
) -> str:
    """子文件夹条目 (标题, href)、页面条目 (标题, href)；href 均相对于当前目录页所在文件夹。"""
    blocks: list[str] = [f'<p class="catalog-hint">{_esc_html(hint)}</p>']
    if subfolder_entries:
        joined = "\n    ".join(_catalog_li(t, h) for t, h in subfolder_entries)
        blocks.append("<p><strong>子文件夹</strong></p>")
        blocks.append(f'<ul class="catalog-list">\n    {joined}\n</ul>')
    if page_entries:
        joined = "\n    ".join(_catalog_li(t, h) for t, h in page_entries)
        blocks.append("<p><strong>页面</strong></p>")
        blocks.append(f'<ul class="catalog-list">\n    {joined}\n</ul>')
    return "\n".join(blocks)


def run_batch(
    directory: Path,
    *,
    nav_out: str = "index.html",
    site_title: str = "文档目录",
    site_subtitle: str | None = None,
    add_nav_link: bool = True,
    open_browser: bool = False,
    keep_first_h1: bool = False,
    no_strip_h1: bool = False,
    recursive: bool = True,
) -> None:
    directory = directory.resolve()
    if not directory.is_dir():
        print(f"不是目录: {directory}", file=sys.stderr)
        sys.exit(1)

    nav_path = (directory / nav_out).resolve()

    it = directory.rglob("*.md") if recursive else directory.glob("*.md")
    md_files = sorted(
        it,
        key=lambda p: (str(p.relative_to(directory)).lower(), p.name.lower()),
    )
    if not md_files:
        hint = "（含子目录）" if recursive else ""
        print(f"目录{hint}中无 .md 文件: {directory}", file=sys.stderr)
        sys.exit(1)

    dresolved = directory.resolve()
    built: list[tuple[Path, str, str]] = []

    for md in md_files:
        out_file = md.with_suffix(".html")
        if out_file.resolve() == nav_path:
            out_file = md.parent / f"{md.stem}_page.html"
            print(
                f"注意: {md.relative_to(directory)} 输出为 "
                f"{out_file.relative_to(directory).as_posix()}（避免覆盖目录页 {nav_out}）",
                file=sys.stderr,
            )

        out_abs = out_file.resolve()
        md_parent = md.parent.resolve()
        if md_parent == dresolved:
            folder_index = nav_path
            back_label = "← 文档目录"
        else:
            folder_index = (md.parent / "index.html").resolve()
            back_label = "← 本文件夹目录"

        back = (
            _nav_href_from_page(folder_index, out_abs) if add_nav_link else None
        )
        title, html = render_page_from_markdown(
            md,
            keep_first_h1=keep_first_h1,
            no_strip_h1=no_strip_h1,
            nav_back_href=back,
            nav_back_label=back_label,
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(html, encoding="utf-8")
        print(f"已写入: {out_file.relative_to(directory)}")
        built.append((out_abs, title, html))

    all_out: set[Path] = {b[0] for b in built}

    dirs_with_index: set[Path] = set()
    for out_abs, _, _ in built:
        p = out_abs.parent
        while True:
            dirs_with_index.add(p)
            if p == dresolved:
                break
            p = p.parent

    for folder in sorted(dirs_with_index, key=lambda p: (len(p.parts), str(p))):
        folder = folder.resolve()

        if folder == dresolved:
            catalog_path = nav_path
            cat_title = site_title
            cat_sub = site_subtitle
            parent_nav: str | None = None
            parent_lbl = "← 文档目录"
            hint_txt = (
                "根目录索引：下列为子文件夹与本目录下的页面；子文件夹内另有各自的 index.html。"
            )
        else:
            catalog_path = folder / "index.html"
            rel = folder.relative_to(dresolved).as_posix()
            cat_title = folder.name
            cat_sub = f"子目录 · {rel}"
            if folder.parent.resolve() == dresolved:
                parent_cat = nav_path
            else:
                parent_cat = folder.parent / "index.html"
            parent_nav = Path(
                os.path.relpath(parent_cat.resolve(), start=catalog_path.parent)
            ).as_posix()
            parent_lbl = "← 上一级"
            hint_txt = "本文件夹内的页面与子文件夹。"

        catalog_dir = catalog_path.parent.resolve()

        subfolder_entries: list[tuple[str, str]] = []
        try:
            children = sorted(folder.iterdir(), key=lambda x: x.name.lower())
        except OSError:
            children = []
        for child in children:
            if not child.is_dir():
                continue
            try:
                child.resolve().relative_to(dresolved)
            except ValueError:
                continue
            if not _dir_tree_has_output(child.resolve(), all_out):
                continue
            child_index = (child / "index.html").resolve()
            href = Path(os.path.relpath(child_index, start=catalog_dir)).as_posix()
            subfolder_entries.append((f"{child.name} /", href))

        page_entries: list[tuple[str, str]] = []
        for out_abs, title, _ in built:
            if out_abs.parent != folder:
                continue
            href = Path(os.path.relpath(out_abs, start=catalog_dir)).as_posix()
            page_entries.append((title, href))

        page_entries.sort(key=lambda x: (x[0].lower(), x[1]))
        subfolder_entries.sort(key=lambda x: (x[0].lower(), x[1]))

        inner = build_folder_catalog_inner_html(
            subfolder_entries=subfolder_entries,
            page_entries=page_entries,
            hint=hint_txt,
        )
        catalog_doc = build_html(
            cat_title,
            cat_sub,
            inner,
            nav_back_href=parent_nav,
            nav_back_label=parent_lbl,
            main_extra_class="catalog-main",
        )
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        catalog_path.write_text(catalog_doc, encoding="utf-8")
        print(f"目录页: {catalog_path.relative_to(dresolved)}")

    if open_browser:
        webbrowser.open(nav_path.as_uri())


def main() -> None:
    ap = argparse.ArgumentParser(description="Markdown → 浅蓝主题 HTML（MathJax）")
    ap.add_argument(
        "input_pos",
        nargs="?",
        type=Path,
        help="输入 .md 文件（单文件模式，与 --input 二选一）",
    )
    ap.add_argument("--input", type=Path, help="输入 .md 文件")
    ap.add_argument("-o", "--output", type=Path, help="输出 .html（默认同名）")
    ap.add_argument("-t", "--title", help="页面标题（默认取首个 # 标题或文件名）")
    ap.add_argument("--subtitle", help="副标题（显示在 h1 下方）")
    ap.add_argument(
        "--batch",
        type=Path,
        metavar="DIR",
        help="批量模式：递归扫描 .md，HTML 与 .md 同目录；根目录生成 --nav-out，各子文件夹内生成 index.html",
    )
    ap.add_argument(
        "--flat",
        action="store_true",
        help="批量模式：仅处理指定目录根下的 .md，不递归子文件夹",
    )
    ap.add_argument(
        "--nav-out",
        default="index.html",
        help="批量模式：目录页文件名（相对于批处理目录，默认 index.html）",
    )
    ap.add_argument(
        "--site-title",
        default="文档目录",
        help="批量模式：目录页的站点标题",
    )
    ap.add_argument(
        "--site-subtitle",
        default=None,
        help="批量模式：目录页副标题",
    )
    ap.add_argument(
        "--no-nav-link",
        action="store_true",
        help="批量模式：不在各子页面顶部显示「返回目录」链接",
    )
    ap.add_argument(
        "--open-browser",
        action="store_true",
        help="批量模式结束后用系统默认浏览器打开目录页",
    )
    ap.add_argument(
        "--keep-first-h1",
        action="store_true",
        help="保留正文中的第一个 h1（默认若与标题重复则去掉）",
    )
    ap.add_argument(
        "--no-strip-h1",
        action="store_true",
        help="不去掉正文第一个 h1（可能与 header 重复）",
    )
    args = ap.parse_args()

    if args.batch is not None:
        run_batch(
            args.batch,
            nav_out=args.nav_out,
            site_title=args.site_title,
            site_subtitle=args.site_subtitle,
            add_nav_link=not args.no_nav_link,
            open_browser=args.open_browser,
            keep_first_h1=args.keep_first_h1,
            no_strip_h1=args.no_strip_h1,
            recursive=not args.flat,
        )
        return

    path: Path | None = args.input or args.input_pos
    if path is None:
        ap.print_help()
        print("\n请指定 --input FILE.md、位置参数 FILE.md，或 --batch DIR", file=sys.stderr)
        sys.exit(2)

    if not path.is_file():
        print(f"找不到文件: {path}", file=sys.stderr)
        sys.exit(1)

    md_text = path.read_text(encoding="utf-8")

    title = args.title or extract_title_from_md(md_text) or path.stem
    body_html = convert_md_to_body(md_text)

    if not args.no_strip_h1 and not args.keep_first_h1:
        extracted = extract_title_from_md(md_text)
        if extracted and (args.title is None or args.title == extracted):
            body_html = strip_first_h1_html(body_html)

    out = args.output or path.with_suffix(".html")

    full = build_html(
        title=title,
        subtitle=args.subtitle,
        body_html=body_html,
        include_tex_macros=True,
    )
    out.write_text(full, encoding="utf-8")
    print(f"已写入: {out.resolve()}")


if __name__ == "__main__":
    main()
