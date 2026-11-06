#!/usr/bin/env python3
"""Convert paper PDFs to Markdown (pymupdf4llm).

Usage: pdf2md.py [pdf ...]   (default: every PDF in articles/pdfs without an .md yet)
Output: articles/md/<name>.md
"""
import subprocess
import sys
from pathlib import Path

import pymupdf4llm

ROOT = Path(__file__).parent
OUT = ROOT / "md"


def convert(pdf: Path) -> Path:
    OUT.mkdir(exist_ok=True)
    out = OUT / f"{pdf.stem}.md"
    text = pymupdf4llm.to_markdown(str(pdf), use_ocr=False)
    if not text.strip():  # image-only first page etc.: plain text instead
        text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
    out.write_text(text, encoding="utf-8")
    return out


if __name__ == "__main__":
    todo = [Path(a) for a in sys.argv[1:]] or [
        p for p in sorted((ROOT / "pdfs").glob("*.pdf")) if not (OUT / f"{p.stem}.md").exists()
    ]
    for p in todo:
        try:
            print(f"{p.name} -> {convert(p)}")
        except Exception as e:
            print(f"{p.name} FAILED: {e}")
    if todo:  # refresh work/bibliografia.bib
        subprocess.run([sys.executable, str(ROOT / "md2bib.py")])
