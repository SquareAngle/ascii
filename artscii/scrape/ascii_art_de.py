"""Scrape www.ascii-art.de: alphabetical index pages link to plain .txt files, one per
topic, with pieces separated by blank lines (split during cleaning).

    python -m artscii.scrape.ascii_art_de
"""
import argparse
import re

from .common import Fetcher, Writer, record

BASE = "https://www.ascii-art.de/ascii/"


def scrape(limit=None):
    fetch, out = Fetcher("ascii_art_de", delay=0.5), Writer("ascii_art_de")
    root = fetch.get(BASE) or ""
    pages = sorted(set(re.findall(r'href="(index[^"]*\.shtml)"', root)))
    files = set(re.findall(r'href="([a-z]+/[^"]+\.txt)"', root))
    for page in pages:
        files |= set(re.findall(r'href="([a-z]+/[^"]+\.txt)"', fetch.get(BASE + page) or ""))
    files = sorted(files)[:limit]
    for i, f in enumerate(files):
        text = fetch.get(BASE + f)
        if text:
            topic = f.split("/")[-1][:-4].replace("_", " ")
            out.write(record("ascii_art_de", BASE + f, topic, text))
        if i % 100 == 0:
            print(f"{i}/{len(files)} files, {out.n} records", flush=True)
    out.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    scrape(ap.parse_args().limit)
