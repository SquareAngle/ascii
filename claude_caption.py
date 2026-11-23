"""Caption pieces with Claude (vision) instead of a local VLM, for comparison with
modal_caption.py. Same render, prompt, parser and output rows; runs locally.

Needs ANTHROPIC_API_KEY. Caption the exact pieces of an earlier pilot:
    python claude_caption.py --data-dir <dir with train/val.jsonl> \\
        --keys captions/pilot.jsonl --out claude_pilot
Writes <out>.jsonl and <out>.html (same review page as the Modal pilot).
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from artscii.caption.vlm import (PROMPT, hints, parse_response, piece_key, png_b64,
                                 render_png, write_report)

MODEL = "claude-opus-5-5"


def caption_one(client, piece, model, effort):
    b64 = png_b64(render_png(piece["art"]))
    resp = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # re-run on another model if this one declines
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": b64}},
            {"type": "text", "text": PROMPT.format(hints=hints(piece))},
        ]}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    parsed = parse_response(raw) if resp.stop_reason != "refusal" else None
    row = dict(key=piece_key(piece["art"]), source=piece["source"],
               category=piece["category"], title=piece["title"], split=piece["split"],
               h=piece["h"], w=piece["w"])
    row.update(parsed if parsed else dict(failed=True, raw=raw or f"[{resp.stop_reason}]"))
    usage = dict(input=resp.usage.input_tokens, output=resp.usage.output_tokens)
    return row, b64, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--keys", required=True, help="jsonl whose `key` fields pick the pieces")
    ap.add_argument("--out", default="claude_pilot")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    order = [json.loads(l)["key"] for l in open(args.keys)]
    wanted = set(order)
    found = {}
    for split in ("train", "val"):
        for line in open(Path(args.data_dir) / f"{split}.jsonl"):
            p = json.loads(line)
            k = piece_key(p["art"])
            if k in wanted:
                found[k] = p
    pieces = [found[k] for k in order if k in found]
    print(f"{len(pieces)}/{len(order)} pieces found", flush=True)

    client = anthropic.Anthropic()
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda p: caption_one(client, p, args.model, args.effort), pieces))
    dt = time.time() - t0

    counts = {"captioned": 0, "unclear": 0, "failed": 0}
    tok_in = tok_out = 0
    with open(f"{args.out}.jsonl", "w") as fh:
        for row, _, u in results:
            counts["failed" if row.get("failed") else "unclear" if row.get("unclear")
                   else "captioned"] += 1
            tok_in += u["input"]
            tok_out += u["output"]
            fh.write(json.dumps(row) + "\n")
    write_report(Path(f"{args.out}.html"), [dict(r, png=b) for r, b, _ in results])
    print(f"{counts} in {dt:.1f}s; tokens in={tok_in} out={tok_out} "
          f"(per piece {tok_in / len(results):.0f} / {tok_out / len(results):.0f})")


if __name__ == "__main__":
    main()
