"""Shared scraping helpers: a polite cached fetcher and a JSONL record writer.

Every raw record has the same schema so cleaning is source-agnostic:
    {id, source, url, category, title, artist, art}
"""
import hashlib
import json
import time
from pathlib import Path

import requests

DATA_ROOT = Path.home() / "artscii_data"
USER_AGENT = "artscii-research-scraper/0.1 (university course project)"


class Fetcher:
    """GET with an on-disk cache (re-runs are free) and a fixed delay between network hits."""

    def __init__(self, source: str, delay: float = 1.0):
        self.cache = DATA_ROOT / "cache" / source
        self.cache.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last = 0.0

    def get(self, url: str) -> str | None:
        path = self.cache / hashlib.sha1(url.encode()).hexdigest()
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        for attempt in range(3):
            time.sleep(max(0.0, self.delay - (time.time() - self._last)))
            self._last = time.time()
            try:
                r = self.session.get(url, timeout=30)
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            if r.ok:
                r.encoding = r.encoding or "utf-8"
                path.write_text(r.text, encoding="utf-8")
                return r.text
            time.sleep(5 * (attempt + 1))
        return None


def record(source, url, category, art, title="", artist="", idx=0) -> dict:
    rid = hashlib.sha1(f"{url}#{idx}".encode()).hexdigest()[:16]
    return dict(id=rid, source=source, url=url, category=category,
                title=title, artist=artist, art=art)


class Writer:
    """Append-only JSONL writer for raw/<source>.jsonl."""

    def __init__(self, source: str):
        out = DATA_ROOT / "raw"
        out.mkdir(parents=True, exist_ok=True)
        self.f = open(out / f"{source}.jsonl", "w")
        self.n = 0

    def write(self, rec: dict):
        self.f.write(json.dumps(rec) + "\n")
        self.n += 1
        if self.n % 500 == 0:
            self.f.flush()

    def close(self):
        self.f.close()
