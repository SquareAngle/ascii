"""Scrape www.asciiart.eu: the curated gallery, and optionally the Usenet archive.

Gallery: each category page lists every piece as a `div.art-card` (title/artist in
data attributes, art in `div.art-card__ascii`); subcategories are links below the
current path. Usenet: one post body per thread page; the thread subject is the title.

    python -m artscii.scrape.asciiart_eu            # gallery
    python -m artscii.scrape.asciiart_eu --usenet   # Usenet archive (~38k pages)
"""
import argparse
import re

from bs4 import BeautifulSoup

from .common import Fetcher, Writer, record

BASE = "https://www.asciiart.eu"
# Top-level gallery sections that hold drawings (skips tools, info pages, borders,
# animations, one-liners and generated patterns).
SECTIONS = """animals art-and-design books buildings-and-places cartoons
clothing-and-accessories comics computers electronics food-and-drinks
holiday-and-events logos miscellaneous movies music mythology nature people plants
religion space sports-and-outdoors television toys vehicles video-games weapons""".split()


def scrape_gallery(limit=None):
    fetch, out = Fetcher("asciiart_eu", delay=0.5), Writer("asciiart_eu")
    queue, seen = [f"/{s}" for s in SECTIONS], set()
    while queue and (limit is None or len(seen) < limit):
        path = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        html = fetch.get(BASE + path)
        if html is None:
            continue
        soup = BeautifulSoup(html, "lxml")
        category = path.strip("/").replace("/", " > ").replace("-", " ")
        for card in soup.select("div.art-card"):
            art = card.select_one("div.art-card__ascii")
            if art:
                out.write(record("asciiart_eu", BASE + path, category, art.get_text(),
                                 card.get("data-title", ""), card.get("data-artist", ""),
                                 idx=card.get("data-id", "")))
        for a in soup.find_all("a", href=True):
            href = a["href"].split("?")[0].split("#")[0]
            if href.startswith(path + "/") and href.count("/") <= 4:
                queue.append(href)
        print(f"{len(seen)} pages, {out.n} pieces  {path}", flush=True)
    out.close()


def scrape_usenet(limit=None):
    fetch, out = Fetcher("asciiart_eu_usenet", delay=0.5), Writer("asciiart_eu_usenet")
    sitemap = fetch.get(BASE + "/sitemap-usenet.xml") or ""
    threads = re.findall(r"<loc>([^<]+/usenet/thread/[^<]+)</loc>", sitemap)[:limit]
    for i, url in enumerate(threads):
        html = fetch.get(url)
        if html is None:
            continue
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.get_text() if soup.title else ""
        subject, _, rest = title.partition(" - ")
        group = rest.partition(" - ")[0]
        for j, body in enumerate(soup.select("pre.usenet-body")):
            out.write(record("asciiart_eu_usenet", url, f"usenet > {group}",
                             body.get_text(), subject, idx=j))
        if i % 200 == 0:
            print(f"{i}/{len(threads)} threads, {out.n} posts", flush=True)
    out.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--usenet", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    (scrape_usenet if args.usenet else scrape_gallery)(args.limit)
