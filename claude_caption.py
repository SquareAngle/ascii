"""Caption pieces with Claude (vision) instead of a local VLM, for comparison with
modal_caption.py. Same render, prompt, parser and output rows; runs locally.

Needs an API key in ARTSCII_ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY, which Claude Code
cloud environments reserve for their own login). Caption the exact pieces of an earlier pilot:
    python claude_caption.py --data-dir <dir with train/val.jsonl> \\
        --keys captions/pilot.jsonl --out claude_pilot
Writes <out>.jsonl and <out>.html (same review page as the Modal pilot).

Differences from the Qwen rows: captioned rows add "confidence" (0 to 1), and
unclear or unparseable rows add "reasoning", the model's summarized reasoning.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from artscii.caption.vlm import (PROMPT, hints, parse_response, piece_key, png_b64,
                                 render_png, write_report)

MODEL = "claude-opus-5-5"


def _claude_prompt(prompt):
    """The shared (Qwen) prompt with Claude-only changes: a natural-wording rule, a shorter
    long caption, and a confidence rule and field. The changes go inside the rules and the
    JSON template: a confidence request appended after the template made the model write the
    JSON and then a second object or a note, or leave the field out."""
    rules = (
        "- Write like a person describing the picture in plain everyday words. Name the "
        "subject and at most two or three details that stand out, not every part.\n"
        "- Describe only what is in the picture: no comparisons to anything outside it (a "
        "story, film or famous scene) and no hedges such as \"likely\", \"possibly\" or "
        "\"appears to be\". Put any doubt in the confidence number, never in the captions.\n"
        "- Also avoid the words \"drawn\" and \"dotted\".\n"
        "- Rate your confidence that the captions are right as a number from 0 to 1: about "
        "0.9 or more if you are sure what the picture shows, about 0.5 to 0.8 if the subject "
        "is clear but a detail or the exact identity is a guess, below 0.5 if the whole "
        "subject is a plausible guess.\n\nAnswer with JSON only")
    old_long = '"long": "<one or two sentences: subject, pose, key details and composition>"}}'
    new_long = ('"long": "<one or two sentences, at most 30 words: the subject, what it is '
                'doing and the details that stand out>",\n  "confidence": <number from 0 to 1>}}')
    assert prompt.count("\n\nAnswer with JSON only") == 1 and prompt.endswith(old_long)
    return prompt.replace("\n\nAnswer with JSON only", "\n" + rules).removesuffix(
        old_long) + new_long


CLAUDE_PROMPT = _claude_prompt(PROMPT)


def request_params(piece, b64, model, effort):
    """Messages API parameters for one piece (shared with the Batch API in claude_batch.py)."""
    return dict(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive", "display": "summarized"},  # summary kept for unclear rows
        output_config={"effort": effort},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": b64}},
            {"type": "text", "text": CLAUDE_PROMPT.format(hints=hints(piece))},
        ]}],
    )


def make_row(piece, msg):
    """Output row for one piece from a Message: captions, unclear or failed."""
    raw = "".join(b.text for b in msg.content if b.type == "text")
    reasoning = " ".join(" ".join(b.thinking.split()) for b in msg.content
                         if b.type == "thinking" and b.thinking)
    parsed = parse_response(raw) if msg.stop_reason != "refusal" else None
    row = dict(key=piece_key(piece["art"]), source=piece["source"],
               category=piece["category"], title=piece["title"], split=piece["split"],
               h=piece["h"], w=piece["w"])
    row.update(parsed if parsed else dict(failed=True, raw=raw or f"[{msg.stop_reason}]"))
    if (not parsed or parsed.get("unclear")) and reasoning:
        row["reasoning"] = reasoning
    return row


def caption_one(client, piece, model, effort):
    b64 = png_b64(render_png(piece["art"]))
    resp = client.beta.messages.create(
        **request_params(piece, b64, model, effort),
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # re-run on another model if this one declines
    )
    usage = dict(input=resp.usage.input_tokens, output=resp.usage.output_tokens)
    return make_row(piece, resp), b64, usage


def load_pieces(data_dir, keys=None):
    """Pieces from train.jsonl and val.jsonl, in file order, or in `keys` order if given."""
    wanted = set(keys) if keys is not None else None
    found = {}
    for split in ("train", "val"):
        for line in open(Path(data_dir) / f"{split}.jsonl"):
            p = json.loads(line)
            k = piece_key(p["art"])
            if wanted is None or k in wanted:
                found[k] = p
    return [found[k] for k in keys if k in found] if keys is not None else list(found.values())


def make_client():
    key = os.environ.get("ARTSCII_ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


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
    pieces = load_pieces(args.data_dir, order)
    print(f"{len(pieces)}/{len(order)} pieces found", flush=True)

    client = make_client()
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda p: caption_one(client, p, args.model, args.effort), pieces))
    dt = time.time() - t0

    counts = {"captioned": 0, "unclear": 0, "failed": 0}
    conf = {">=0.9": 0, "0.5-0.9": 0, "<0.5": 0, "missing": 0}
    tok_in = tok_out = 0
    with open(f"{args.out}.jsonl", "w") as fh:
        for row, _, u in results:
            counts["failed" if row.get("failed") else "unclear" if row.get("unclear")
                   else "captioned"] += 1
            if not row.get("failed") and not row.get("unclear"):
                c = row.get("confidence")
                conf["missing" if c is None else ">=0.9" if c >= 0.9 else "0.5-0.9"
                     if c >= 0.5 else "<0.5"] += 1
            tok_in += u["input"]
            tok_out += u["output"]
            fh.write(json.dumps(row) + "\n")
    write_report(Path(f"{args.out}.html"), [dict(r, png=b) for r, b, _ in results])
    print(f"{counts} confidence={conf} in {dt:.1f}s; tokens in={tok_in} out={tok_out} "
          f"(per piece {tok_in / len(results):.0f} / {tok_out / len(results):.0f})")


if __name__ == "__main__":
    main()
