"""Scrape the human-written HuffPost articles of OpenTuringBench (the HF repo ships only machine texts).

URL list and body extraction are the authors' (scripts/human_urls.json, scripts/fetch_human_data.py). Unlike theirs,
each text also gets the header the machine texts have: "<title>\\n\\n<Month D, YYYY>\\n\\n<body>" (title from h1.headline,
date from article:published_time in US Eastern). Resumable; appends to data/openturingbench/human_<split>.jsonl
({"url", "content", "model": "human"} per line). Failures (HTTP != 200, no body or title/date) go to
human_<split>_failed.txt and are skipped on re-run.
Run: uv run python src/data_generation/fetch_otb_human.py [split ...]   (test | val | train, default all three in that order)
"""

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent))  # flat modules in src/
from data_generation.openturingbench import DATASET_ID, HUMAN_DIR

WORKERS = 4  # each worker sleeps 1 s after a request, so ~4 requests/s at most


def clean_text(text):
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_article(html):
    """'<title>\\n\\n<date>\\n\\n<paragraphs>' or '' if the body, title or date is missing."""
    soup = BeautifulSoup(html, "lxml")
    section = soup.find("section", class_="entry__content-list js-entry-content js-cet-subunit")
    h1, meta = soup.find("h1", class_="headline"), soup.find("meta", property="article:published_time")
    if not (section and h1 and meta):
        return ""
    paragraphs = [p for div in section.find_all("div", class_="primary-cli cli cli-text") for p in div.find_all("p")]
    body = "\n\n".join(clean_text(p.get_text()) for p in paragraphs if clean_text(p.get_text()) != "Also on HuffPost:")
    when = datetime.fromisoformat(meta["content"].replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York"))
    if not body or not clean_text(h1.get_text()):
        return ""
    return f"{clean_text(h1.get_text())}\n\n{when:%B} {when.day}, {when.year}\n\n{body}"


def fetch(url):
    """(key, text, status); text is '' on failure, status None on a network error (retried on the next run)."""
    key = Path(urlparse(url).path).name  # identifier shared with the machine rows
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException as e:
        print(f"{key}: {e}", flush=True)
        return key, "", None
    time.sleep(1)  # do not hammer the server
    return key, (extract_article(r.text) if r.status_code == 200 else ""), r.status_code


def main(split):
    urls = json.load(open(hf_hub_download(DATASET_ID, "scripts/human_urls.json", repo_type="dataset")))[split]
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    out, failed = HUMAN_DIR / f"human_{split}.jsonl", HUMAN_DIR / f"human_{split}_failed.txt"
    done = set()
    if out.exists():
        done |= {json.loads(l)["url"] for l in out.read_text().splitlines() if l}
    if failed.exists():
        done |= {l.split("\t")[0] for l in failed.read_text().splitlines() if l}
    todo = [u for u in urls if Path(urlparse(u).path).name not in done]
    print(f"{split}: {len(urls)} urls, {len(done)} already done, {len(todo)} to fetch", flush=True)

    with open(out, "a") as fo, open(failed, "a") as ff, ThreadPoolExecutor(WORKERS) as pool:
        for i, (key, text, status) in enumerate(pool.map(fetch, todo), 1):
            if text:
                fo.write(json.dumps({"url": key, "content": text, "model": "human"}) + "\n")
                fo.flush()
            elif status is not None:
                ff.write(f"{key}\t{status if status != 200 else 'empty'}\n")
                ff.flush()
            if i % 200 == 0:
                print(f"{split} {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    for split in sys.argv[1:] or ["test", "val", "train"]:
        main(split)
