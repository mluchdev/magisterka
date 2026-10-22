#!/usr/bin/env python3
"""Convert paper PDFs to Markdown (pymupdf4llm).

Usage: pdf2md.py [pdf ...]   (default: every PDF in articles/pdfs)
Output: articles/md/<name>.md
"""
import sys
from pathlib import Path

import pymupdf4llm

ROOT = Path(__file__).parent
OUT = ROOT / "md"


def convert(pdf: Path) -> Path:
    OUT.mkdir(exist_ok=True)
    out = OUT / f"{pdf.stem}.md"
    out.write_text(pymupdf4llm.to_markdown(str(pdf)), encoding="utf-8")
    return out


if __name__ == "__main__":
    pdfs = [Path(a) for a in sys.argv[1:]] or sorted((ROOT / "pdfs").glob("*.pdf"))
    for p in pdfs:
        print(f"{p.name} -> {convert(p)}")
