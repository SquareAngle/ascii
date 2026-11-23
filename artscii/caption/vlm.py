"""Caption ASCII art with a vision-language model: render -> prompt -> parse.

Each piece is rendered to a PNG in a monospace font (VLMs read drawn strokes far
better than raw character soup) and the VLM returns three captions at different
levels of detail, or `unclear` when it can't tell what the drawing shows:

    {"short": "a sitting cat",
     "medium": "a cat sitting upright with its tail curled around its paws",
     "long": "..."}

This module has no GPU or VLM dependency; modal_caption.py runs the model.
"""
import base64
import hashlib
import html
import io
import json
import re

from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"]
LEVELS = ("short", "medium", "long")


def piece_key(art: str) -> str:
    """Stable id of a cleaned piece; equals clean.dedup_key(art.split("\\n"))."""
    return hashlib.sha1(art.replace("\n", "").replace(" ", "").encode()).hexdigest()


def _font(size):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise OSError(f"no monospace font found in {FONT_PATHS}")


def render_png(art: str, font_size: int = 18, pad: int = 16, min_side: int = 384) -> Image.Image:
    """Black-on-white render with exact character-cell alignment. Small pieces are
    scaled up so their strokes stay legible after the VLM's own resizing."""
    font = _font(font_size)
    lines = art.split("\n")
    cw = font.getlength("M")
    lh = round(font_size * 1.25)
    w = round(cw * max(len(l) for l in lines)) + 2 * pad
    h = lh * len(lines) + 2 * pad
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):  # per cell, so glyph widths can't drift the grid
            if ch != " ":
                draw.text((pad + c * cw, pad + r * lh), ch, font=font, fill=0)
    if max(w, h) < min_side:
        s = min_side / max(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return img


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


PROMPT = """\
The image shows a piece of ASCII art: a picture drawn with text characters, rendered in a \
monospace font. Captions you write will train a model that generates such art from text, \
so describe what the picture depicts the way a person would ask for it.

Hints from the website it came from (often vague, wrong or unrelated; trust the image): \
{hints}

Rules:
- Describe the subject, not the medium: never say "ASCII", "characters", "text art", \
"drawing of" or "line art".
- Ignore artist signatures or initials (e.g. "jgs", "hjw", "-Row") and stray text, unless \
the lettering itself is the subject.
- Mention pose, orientation (e.g. facing left) and distinctive parts only when visible.
- Do not guess beyond what is drawn. If you cannot tell what it depicts, answer \
{{"unclear": true}}.

Answer with JSON only, no other text:
{{"short": "<2-5 word noun phrase, e.g. a sitting cat>",
  "medium": "<one sentence, at most 15 words: subject, pose or action, notable parts>",
  "long": "<one or two sentences: subject, pose, key details and composition>"}}"""


def hints(piece: dict) -> str:
    title = (piece.get("title") or "").strip()
    parts = [f'category "{piece["category"]}"'] if piece.get("category") else []
    if title and title.lower() not in piece.get("category", "").lower():
        parts.append(f'title "{title}"')
    return "; ".join(parts) or "none"


def build_messages(piece: dict, img_b64: str) -> list[dict]:
    """OpenAI-style chat messages (what vLLM's LLM.chat and most VLM APIs accept)."""
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        {"type": "text", "text": PROMPT.format(hints=hints(piece))},
    ]}]


def parse_response(text: str) -> dict | None:
    """{"short", "medium", "long"} or {"unclear": True}; None if the output is unusable."""
    m = re.search(r"\{.*\}", text, re.S)  # tolerate ```json fences and chatter
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("unclear") is True:
        return {"unclear": True}
    caps = {k: obj.get(k) for k in LEVELS}
    if not all(isinstance(v, str) and v.strip() for v in caps.values()):
        return None
    return {k: " ".join(v.split()) for k, v in caps.items()}


def write_report(path, rows: list[dict]):
    """Review page for a pilot: rendered image, source hints and captions side by side."""
    n = len(rows)
    unclear = sum(bool(r.get("unclear")) for r in rows)
    failed = sum(r.get("failed", False) for r in rows)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Caption pilot</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#111}"
        ".card{display:flex;gap:16px;border-top:1px solid #ddd;padding:12px 0;flex-wrap:wrap}"
        ".card img{max-width:min(560px,100%);border:1px solid #eee}"
        ".meta{flex:1;min-width:260px}.k{color:#666;font-size:12px}"
        ".bad{color:#b00}</style></head><body>",
        f"<h1>Caption pilot</h1><p>{n} pieces · {n - unclear - failed} captioned · "
        f"{unclear} unclear · {failed} unparseable</p>",
    ]
    for r in rows:
        if r.get("failed"):
            body = f"<p class='bad'>unparseable output:</p><pre>{html.escape(r.get('raw', ''))}</pre>"
        elif r.get("unclear"):
            body = "<p class='bad'>unclear</p>"
        else:
            body = "".join(f"<p><span class='k'>{k}</span><br>{html.escape(r[k])}</p>"
                           for k in LEVELS)
        parts.append(
            f"<div class='card'><img src='data:image/png;base64,{r['png']}' alt=''>"
            f"<div class='meta'><p class='k'>{html.escape(r['source'])} · "
            f"{html.escape(hints(r))}</p>{body}</div></div>")
    path.write_text("".join(parts) + "</body></html>")
