#!/usr/bin/env python3
"""Build work/bibliografia.bib (overwritten) from the papers in articles/md (title -> lookup).

Usage: md2bib.py [name ...]   (default: every .md)
Cite key = first author surname + year, e.g. kirchenbauer2023.
Papers in manual.bib (no online match; details copied from the PDFs) are appended as they are.
Lookup: Crossref first, then arXiv (preprints).
Papers Crossref can't match are listed at the end; add those by hand.
"""
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).parent
API = "https://api.crossref.org/works"
ATOM = "{http://www.w3.org/2005/Atom}"
SKIP = re.compile(r"^(published|accepted|preprint|under review|acl|findings|proceedings|arxiv)", re.I)

# titles the .md heading garbles (OCR/typesetting); name -> real title
TITLES = {
    "lu_sco": "Large Language Models can be Guided to Evade AI-Generated Text Detection",
    "remark_llm": "REMARK-LLM: A Robust and Efficient Watermarking Framework for Generative Large Language Models",
    "upv": "An Unforgeable Publicly Verifiable Watermark for Large Language Models",
}


MANUAL = {"Ai-detection-tools-accuracy", "c2pa", "kulkarni_grover"}  # covered by manual.bib


def get(url: str) -> str:
    for wait in (1, 5, 15):  # be polite, retry on 429
        time.sleep(wait if wait > 1 else 0.5)
        try:
            return urllib.request.urlopen(url, timeout=20).read().decode()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
    raise RuntimeError("rate limited")


def title_of(md: Path) -> str:
    for line in md.read_text(encoding="utf-8").splitlines():
        t = re.sub(r"[#*_]", "", line).strip()
        if len(t) > 10 and not SKIP.match(t):
            return t
    return md.stem


def similar(a: str, b: str) -> bool:
    norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
    return SequenceMatcher(None, norm(a), norm(b)).ratio() >= 0.97


def arxiv(title: str) -> str | None:
    q = urllib.parse.urlencode({"search_query": f'ti:"{title}"', "max_results": 5})
    entries = ET.fromstring(get(f"https://export.arxiv.org/api/query?{q}")).findall(ATOM + "entry")
    e = next((e for e in entries if similar(title, e.findtext(ATOM + "title"))), None)
    if e is None:
        return None
    authors = " and ".join(a.findtext(ATOM + "name") for a in e.findall(ATOM + "author"))
    aid = e.findtext(ATOM + "id").rsplit("/abs/", 1)[1].split("v")[0]
    return (f"@misc{{x,\n  title = {{{' '.join(e.findtext(ATOM + 'title').split())}}},\n"
            f"  author = {{{authors}}},\n  year = {{{e.findtext(ATOM + 'published')[:4]}}},\n"
            f"  eprint = {{{aid}}},\n  archiveprefix = {{arXiv}},\n  url = {{https://arxiv.org/abs/{aid}}},\n}}")


def bibtex(title: str) -> str | None:
    q = urllib.parse.urlencode({"query.title": title, "rows": 1, "select": "DOI,title"})
    items = json.loads(get(f"{API}?{q}"))["message"]["items"]
    if not items or not similar(title, items[0]["title"][0]):
        return arxiv(title)
    entry = get(f"{API}/{items[0]['DOI']}/transform/application/x-bibtex")
    return entry.strip()


def field(entry: str, name: str) -> str:
    m = re.search(rf"\b{name}\s*=\s*[{{\"]+([^}}\"]*)", entry)
    return m.group(1) if m else ""


def key_of(entry: str) -> str:
    first = field(entry, "author").split(" and ")[0]
    surname = first.split(",")[0] if "," in first else first.split()[-1]
    surname = unicodedata.normalize("NFKD", surname).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", surname.lower()) + field(entry, "year")


if __name__ == "__main__":
    names = sys.argv[1:] or [p.stem for p in sorted((ROOT / "md").glob("*.md"))]
    found, missing, used = [], [], {}
    for n in names:
        title = TITLES.get(n) or title_of(ROOT / "md" / f"{n}.md")
        try:
            entry = bibtex(title)
        except Exception as e:
            entry, title = None, f"{title} ({e})"
        if not entry:
            missing.append(f"{n}: {title}")
            continue
        entry = re.sub(r"month\s*=\s*([A-Za-z]{3})[a-z]*", lambda m: "month=" + m.group(1).lower(), entry)
        key = key_of(entry)
        used[key] = used.get(key, 0) + 1
        if used[key] > 1:  # same author + year: kirchenbauer2023b
            key += chr(ord("a") + used[key] - 1)
        found.append(re.sub(r"^(@\w+)\{[^,]*,", rf"\1{{{key},", entry, count=1))
    manual = (ROOT / "manual.bib").read_text(encoding="utf-8")
    found.append(manual.strip())
    missing = [m for m in missing if m.split(":")[0] not in MANUAL]
    out = ROOT.parent / "work" / "bibliografia.bib"
    out.write_text("\n\n".join(found) + "\n", encoding="utf-8")
    print(f"{sum(e.count(chr(10)+chr(64))+1 if e.startswith(chr(64)) else 0 for e in found)} entries -> {out}")
    print("No match:\n  " + "\n  ".join(missing) if missing else "")
