"""Turn raw scraped records into a filtered, deduplicated dataset of sketch-style art.

Pipeline per raw record: normalize -> split into pieces -> trim -> filter -> dedup.
Filters target the three things we don't want:
  * image conversions (dense brightness-ramp fills, e.g. `@#%8MWB` or letter soup),
  * text (prose, banners, signatures-only snippets),
  * fragments too small or too sparse to be a drawing.

    python -m artscii.data.clean            # writes ~/artscii_data/processed/
"""
import argparse
import hashlib
import html
import json
import random
import re
from collections import Counter
from pathlib import Path

from artscii.scrape.common import DATA_ROOT

# Higher priority sources first: on duplicates we keep the better-labelled copy.
SOURCES = ["asciiart_eu", "ascii_art_de", "ascii_co_uk", "asciiart_eu_usenet"]
SPLIT_SOURCES = {"ascii_art_de", "ascii_co_uk", "asciiart_eu_usenet"}
HEAVY = set("@#%&$8MWBNQ0")  # typical "dark" end of brightness ramps
LINE = set("/\\|_()<>[]{}`'-^~")  # strokes: what outline (sketch) drawings are made of
HTML_TAG_LINE = re.compile(r"^\s*</?[A-Za-z][^>]*>\s*$", re.M)
WORD = re.compile(r"[A-Za-z]{3,}")
EXCLUDED_CATEGORIES = re.compile(r"sexual|nude|porn|adult", re.I)  # class project: keep it SFW
MAX_SIDE = 100  # keep anything up to this; the dataset drops what doesn't fit its canvas


def normalize(text: str) -> str | None:
    """Unify newlines/tabs; drop records that aren't (nearly) plain printable ASCII."""
    text = html.unescape(text).replace("\r\n", "\n").replace("\r", "\n").expandtabs(8)
    text = HTML_TAG_LINE.sub("", text)  # stray <P>, <br> etc. left in some .txt files
    bad = sum(1 for c in text if c != "\n" and not (32 <= ord(c) < 127))
    if bad > 0.01 * max(1, len(text)):
        return None
    return "".join(c if c == "\n" or 32 <= ord(c) < 127 else " " for c in text)


CHATTER = re.compile(r"^\s*(>|From:|Subject:|Newsgroups:|Date:|In article|.*wrote:\s*$|_{3}\s*$|-- ?$)"
                     r"|\S+@\S+\.\S+|https?://|www\.")  # also signature delimiter, emails, URLs


def strip_chatter(text: str) -> str:
    """Usenet posts: blank out headers, quoted replies and prose lines (mostly words),
    so the drawings between them separate cleanly."""
    out = []
    for line in text.split("\n"):
        ink = len(line.replace(" ", ""))
        words = WORD.findall(line)
        prose = len(words) >= 4 and sum(map(len, words)) > 0.6 * ink
        out.append("" if CHATTER.search(line) or prose else line)
    return "\n".join(out)


def split_pieces(text: str) -> list[str]:
    """Several pieces in one block are separated by 2+ blank lines."""
    return re.split(r"\n[ ]*\n[ ]*\n", text)


def trim(piece: str) -> list[str]:
    """Strip trailing spaces, blank edge rows and the common left indent."""
    lines = [l.rstrip() for l in piece.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        return []
    if all(l.startswith(":") for l in lines if l):  # ": "-quoted reply: drop the quote column
        return trim("\n".join(l[1:] for l in lines))
    indent = min(len(l) - len(l.lstrip()) for l in lines if l)
    return [l[indent:] for l in lines]


def features(lines: list[str]) -> dict:
    h, w = len(lines), max(len(l) for l in lines)
    ink = [c for l in lines for c in l if c != " "]
    n = max(1, len(ink))
    words = sum(len(m) for l in lines for m in WORD.findall(l))
    return dict(h=h, w=w, ink=len(ink), density=len(ink) / (h * w),
                alnum=sum(c.isalnum() for c in ink) / n, words=words / n,
                heavy=sum(c in HEAVY for c in ink) / n, line=sum(c in LINE for c in ink) / n)


def reject_reason(f: dict) -> str | None:
    """First failed filter, or None if the piece is kept. Thresholds tuned by eye on report.html."""
    if f["h"] < 3 or f["w"] < 6 or f["ink"] < 15:
        return "too_small"
    if f["h"] > MAX_SIDE or f["w"] > MAX_SIDE:
        return "too_big"
    if f["density"] < 0.04:
        return "too_sparse"
    # Dense pieces are kept only if they are still mostly strokes (boxes, keyboards);
    # dense fills of texture characters are shaded or image-converted art.
    if f["density"] > 0.75 or (f["density"] > 0.4 and f["line"] < 0.35):
        return "too_dense"
    if f["heavy"] > 0.35:
        return "ramp_chars"
    if f["words"] > 0.3 or f["alnum"] > 0.5:
        return "text"
    return None


def dedup_key(lines: list[str]) -> str:
    """Ignore all whitespace so re-indented or re-wrapped copies collide."""
    return hashlib.sha1("".join(lines).replace(" ", "").encode()).hexdigest()


def description(rec: dict) -> str:
    """Placeholder conditioning text: category path, plus the title if it adds information."""
    title = rec["title"].strip()
    if title and title.lower() not in rec["category"].lower():
        return f"{rec['category']}: {title}"
    return rec["category"]


def main(val_frac=0.03):
    out_dir = DATA_ROOT / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    seen, kept, rejected = set(), [], {}
    stats = Counter()
    for source in SOURCES:
        path = DATA_ROOT / "raw" / f"{source}.jsonl"
        if not path.exists():
            continue
        for line in open(path):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # partially written last line while scraping
                continue
            if EXCLUDED_CATEGORIES.search(rec["category"] + " " + (rec.get("title") or "")):
                continue
            text = normalize(rec["art"])
            if text is None:
                stats[f"{source}/non_ascii"] += 1
                continue
            if source == "asciiart_eu_usenet":
                text = strip_chatter(text)
            for piece in (split_pieces(text) if source in SPLIT_SOURCES else [text]):
                lines = trim(piece)
                if not lines:
                    continue
                f = features(lines)
                reason = reject_reason(f)
                key = dedup_key(lines)
                if reason is None and key in seen:
                    reason = "duplicate"
                stats[f"{source}/{reason or 'kept'}"] += 1
                if reason:
                    rejected.setdefault(reason, []).append("\n".join(lines))
                    continue
                seen.add(key)
                kept.append(dict(art="\n".join(lines), h=f["h"], w=f["w"], source=source,
                                 category=rec["category"], description=description(rec),
                                 title=rec["title"], artist=rec["artist"], url=rec["url"],
                                 split="val" if int(key, 16) % 1000 < val_frac * 1000
                                 else "train"))
    for split in ("train", "val"):
        with open(out_dir / f"{split}.jsonl", "w") as fh:
            for r in kept:
                if r["split"] == split:
                    fh.write(json.dumps(r) + "\n")
    canvases = [(32, 32), (48, 24), (48, 48), (64, 32), (64, 48), (64, 64), (80, 40), (80, 64)]
    survival = {f"{w}x{h}": sum(r["w"] <= w and r["h"] <= h for r in kept) for w, h in canvases}
    summary = dict(total_kept=len(kept), by_reason=dict(sorted(stats.items())),
                   survival_WxH=survival)
    (out_dir / "stats.json").write_text(json.dumps(summary, indent=2))
    write_report(out_dir / "report.html", kept, rejected)
    print(json.dumps(summary, indent=2))


def write_report(path: Path, kept: list, rejected: dict, n=40):
    """Static HTML page of random kept/rejected samples, for tuning thresholds by eye."""
    rng = random.Random(0)
    sections = [("kept", [r["art"] for r in kept])] + sorted(rejected.items())
    parts = ["<html><body style='font-family:sans-serif'><h1>clean.py report</h1>"]
    for name, arts in sections:
        parts.append(f"<h2>{name} ({len(arts)})</h2><div style='display:flex;flex-wrap:wrap'>")
        for a in rng.sample(arts, min(n, len(arts))):
            parts.append("<pre style='border:1px solid #ccc;margin:4px;padding:4px;"
                         f"font-size:10px'>{html.escape(a)}</pre>")
        parts.append("</div>")
    path.write_text("".join(parts) + "</body></html>")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_frac", type=float, default=0.03)
    main(ap.parse_args().val_frac)
