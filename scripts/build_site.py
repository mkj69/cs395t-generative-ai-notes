#!/usr/bin/env python3
"""Render public note Markdown into the existing dependency-free site."""

from __future__ import annotations

import html
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from markdown import Markdown


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INDEX_PATH = DOCS / "data" / "notes.json"
PUBLIC_TYPES = {"learning-note", "research-note"}


def split_front_matter(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        return {}, raw

    end = raw.find("\n---\n", 4)
    if end == -1:
        raise ValueError(f"Unclosed YAML front matter: {path.relative_to(ROOT)}")

    metadata = yaml.safe_load(raw[4:end]) or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Front matter must be a mapping: {path.relative_to(ROOT)}")
    return metadata, raw[end + 5 :]


def plain(value: Any) -> str:
    return html.escape(str(value), quote=True)


def normalized_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def status_copy(status: str) -> tuple[str, str]:
    return {
        "capturing": ("Manual reconstruction in progress", "NOT YET CHECKED"),
        "reconstructing": ("Manual reconstruction in progress", "SOURCE CHECK PENDING"),
        "checked": ("Source-checked reconstruction", "CHECKED"),
    }.get(status, ("Work in progress", status.upper()))


def make_markdown() -> Markdown:
    return Markdown(
        extensions=["fenced_code", "sane_lists", "tables", "toc"],
        extension_configs={"toc": {"toc_depth": "2-3"}},
        output_format="html5",
    )


def discover_notes() -> list[tuple[Path, dict[str, Any], str]]:
    notes: list[tuple[Path, dict[str, Any], str]] = []
    for source_root in (ROOT / "course-notes", ROOT / "notes"):
        for path in sorted(source_root.rglob("*.md")):
            metadata, body = split_front_matter(path)
            if metadata.get("type") not in PUBLIC_TYPES or metadata.get("public") is not True:
                continue
            notes.append((path, metadata, body))
    return notes


def render_note(path: Path, metadata: dict[str, Any], body: str) -> dict[str, Any]:
    title = str(metadata.get("title") or path.stem)
    slug = str(metadata.get("slug") or path.stem)
    note_type = str(metadata["type"])
    status = str(metadata.get("status") or "draft").lower()
    summary = str(metadata.get("summary") or "A public note in progress.")
    tags = [str(tag) for tag in metadata.get("tags") or []]
    note_date = normalized_date(metadata.get("date"))
    date_long = note_date.strftime("%d %b %Y").upper() if note_date else "DATE NOT SET"
    date_short = note_date.strftime("%b %Y") if note_date else "Undated"
    banner_copy, check_copy = status_copy(status)

    if note_type == "learning-note":
        output_dir = DOCS / "course-notes"
        url = f"course-notes/{slug}.html"
        type_label = "Learning note"
        type_value = "learning"
        banner_class = " learning-banner"
        footer_label = "Learning note"
        disclaimer = "Manual · checked" if status == "checked" else "Manual · in progress"
    else:
        output_dir = DOCS / "notes"
        url = f"notes/{slug}.html"
        type_label = "Research note"
        type_value = "research-question"
        banner_class = ""
        footer_label = "Research note"
        disclaimer = "Research · reviewed" if status == "checked" else "Research · in progress"

    md = make_markdown()
    article_html = md.convert(body)
    toc_html = md.toc
    tags_html = "".join(f"<li>{plain(tag)}</li>" for tag in tags)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{plain(summary)}">
  <title>{plain(title)} — CS 395T</title>
  <link rel="stylesheet" href="../assets/styles.css">
  <script src="../assets/app.js" defer></script>
</head>
<body>
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="site-header">
    <a class="wordmark" href="../index.html"><span>CS 395T</span><small>Learning &amp; research notebook</small></a>
    <button class="menu-button" type="button" aria-expanded="false" aria-controls="site-nav">Menu</button>
    <nav id="site-nav" class="site-nav" aria-label="Primary navigation">
      <a href="../structure.html">Structure</a>
      <a href="../index.html#notes">Index</a>
      <a href="../method.html">Writing guide</a>
      <a href="../index.html#about">About</a>
    </nav>
  </header>

  <main id="main" class="note-shell">
    <header class="note-hero">
      <div class="template-banner{banner_class}">
        <strong>{plain(type_label.upper())} · {plain(status.upper())}</strong>
        <span>{plain(banner_copy)}</span>
      </div>
      <p class="eyebrow">{plain(date_long)} · {plain(check_copy)}</p>
      <h1>{plain(title)}</h1>
      <p class="note-abstract">{plain(summary)}</p>
      <ul class="tag-list" aria-label="Topics">{tags_html}</ul>
    </header>

    <div class="note-layout">
      <aside class="note-toc" aria-label="On this page">
        <p class="aside-label">ON THIS PAGE</p>
        {toc_html}
      </aside>
      <article class="prose note-prose markdown-body">
        {article_html}
      </article>
    </div>
  </main>

  <footer class="site-footer section-shell">
    <div><strong>CS 395T</strong><span>{plain(footer_label)}</span></div>
    <p>Markdown source · {plain(status)}</p>
    <a href="../index.html#notes">Return to index ←</a>
  </footer>
</body>
</html>
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{slug}.html").write_text(page, encoding="utf-8")

    return {
        "title": title,
        "question": summary,
        "type": type_value,
        "typeLabel": type_label,
        "status": status.title(),
        "date": date_short,
        "tags": tags,
        "url": url,
        "disclaimer": disclaimer,
    }


def main() -> None:
    existing = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    static_entries = [entry for entry in existing if entry.get("type") == "template"]
    discovered = discover_notes()
    discovered.sort(
        key=lambda item: normalized_date(item[1].get("date")) or date.min,
        reverse=True,
    )
    generated_entries = [render_note(*note) for note in discovered]
    INDEX_PATH.write_text(
        json.dumps(generated_entries + static_entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Rendered {len(generated_entries)} public note(s).")


if __name__ == "__main__":
    main()
