"""Scrape ascii.co.uk: ~2.5k topic pages under /art/<topic>. Only the first <pre> on a
page is art (it can hold many pieces; splitting happens in cleaning) — the other four
are generated FIGlet banners of the topic name and a character-code table.

    python -m artscii.scrape.ascii_co_uk
"""
import argparse
import re

from bs4 import BeautifulSoup

from .common import Fetcher, Writer, record

BASE = "https://ascii.co.uk"


def scrape(limit=None):
    fetch, out = Fetcher("ascii_co_uk", delay=0.5), Writer("ascii_co_uk")
    index = fetch.get(BASE + "/art") or ""
    topics = sorted(set(re.findall(r'href="/art/([A-Za-z0-9_-]+)"', index)))[:limit]
    for i, topic in enumerate(topics):
        url = f"{BASE}/art/{topic}"
        html = fetch.get(url)
        if html is None:
            continue
        pre = BeautifulSoup(html, "lxml").find("pre")
        if pre:
            out.write(record("ascii_co_uk", url, topic.replace("_", " "), pre.get_text()))
        if i % 100 == 0:
            print(f"{i}/{len(topics)} topics, {out.n} blocks", flush=True)
    out.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    scrape(ap.parse_args().limit)
