"""Caption pieces with Claude through the Message Batches API (half price, asynchronous).

Same prompt, settings and output rows as claude_caption.py, except that batch requests
can't use the server-side refusal fallback: refused, errored or expired pieces are
re-run through the normal API (full price, with the fallback) by `collect --retry`.

    # submit every piece in train.jsonl + val.jsonl (or only --keys); resumable
    python claude_batch.py submit --data-dir <dir> --state claude_all.state.json
    python claude_batch.py status --state claude_all.state.json
    # write rows as batches end; --wait polls until all have ended
    python claude_batch.py collect --data-dir <dir> --state claude_all.state.json \\
        --out claude_all.jsonl --wait --retry

The state file holds the batch ids and which pieces each one covers; keep it. Results
stay downloadable from the API for 29 days after a batch is created.
"""
import argparse
import json
import time
from pathlib import Path

from artscii.caption.vlm import piece_key, png_b64, render_png
from claude_caption import MODEL, caption_one, load_pieces, make_client, make_row, request_params

MAX_REQUESTS = 100_000
BATCH_PRICES = {"claude-opus-5-5": (2.0, 10.0)}  # $ per million tokens in / out, batch rate


def load_state(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {"batches": []}


def save_state(path, state):
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def done_keys(out):
    return {json.loads(l)["key"] for l in open(out)} if out and Path(out).exists() else set()


def submit(args):
    state = load_state(args.state)
    state.setdefault("model", args.model)
    state.setdefault("effort", args.effort)
    if (state["model"], state["effort"]) != (args.model, args.effort):
        raise SystemExit(f"state was started with {state['model']} / {state['effort']}")
    keys = [json.loads(l)["key"] for l in open(args.keys)] if args.keys else None
    pieces = load_pieces(args.data_dir, keys)
    skip = done_keys(args.out) | {k for b in state["batches"] for k in b["keys"]}
    todo = [p for p in pieces if piece_key(p["art"]) not in skip]
    print(f"{len(pieces)} pieces, {len(pieces) - len(todo)} already submitted or captioned, "
          f"{len(todo)} to submit", flush=True)

    client = make_client()
    limit = args.max_mb * 2**20
    reqs, size = [], 0

    def flush():
        nonlocal reqs, size
        if not reqs:
            return
        batch = client.messages.batches.create(requests=reqs)
        state["batches"].append(dict(id=batch.id, keys=[r["custom_id"] for r in reqs],
                                     created=batch.created_at.isoformat()))
        save_state(args.state, state)
        print(f"submitted {batch.id}: {len(reqs)} requests, {size / 2**20:.0f} MB", flush=True)
        reqs, size = [], 0

    for i, p in enumerate(todo, 1):
        req = dict(custom_id=piece_key(p["art"]),
                   params=request_params(p, png_b64(render_png(p["art"])), args.model, args.effort))
        n = len(json.dumps(req)) + 2
        if reqs and (size + n > limit or len(reqs) >= MAX_REQUESTS):
            flush()
        reqs.append(req)
        size += n
        if i % 5000 == 0:
            print(f"  built {i}/{len(todo)}", flush=True)
    flush()


def status(args):
    client = make_client()
    for b in load_state(args.state)["batches"]:
        r = client.messages.batches.retrieve(b["id"])
        c = r.request_counts
        print(f"{b['id']} {r.processing_status:11} processing={c.processing} "
              f"succeeded={c.succeeded} errored={c.errored} expired={c.expired} "
              f"canceled={c.canceled} {'collected' if b.get('collected') else ''}")


def summarize(out, state):
    counts = {"captioned": 0, "unclear": 0, "failed": 0}
    for row in map(json.loads, open(out)):
        counts["failed" if row.get("failed") else "unclear" if row.get("unclear")
               else "captioned"] += 1
    tin = sum(b.get("tokens_in", 0) for b in state["batches"])
    tout = sum(b.get("tokens_out", 0) for b in state["batches"])
    price = BATCH_PRICES.get(state.get("model"))
    cost = f", about ${(tin * price[0] + tout * price[1]) / 1e6:.2f} at batch prices" if price else ""
    print(f"{out}: {counts}; batch tokens in={tin} out={tout}{cost}", flush=True)


def collect(args):
    state = load_state(args.state)
    client = make_client()
    pieces = {piece_key(p["art"]): p for p in load_pieces(args.data_dir)}
    have = done_keys(args.out)
    while True:
        pending = 0
        for b in state["batches"]:
            if b.get("collected"):
                continue
            if client.messages.batches.retrieve(b["id"]).processing_status != "ended":
                pending += 1
                continue
            tin = tout = 0
            retry = []
            with open(args.out, "a") as fh:
                for res in client.messages.batches.results(b["id"]):
                    k = res.custom_id
                    if k in have:
                        continue
                    if res.result.type != "succeeded":
                        retry.append(k)
                        continue
                    msg = res.result.message
                    tin += msg.usage.input_tokens
                    tout += msg.usage.output_tokens
                    row = make_row(pieces[k], msg)
                    if row.get("failed"):
                        retry.append(k)
                    fh.write(json.dumps(row) + "\n")
                    have.add(k)
            b.update(collected=True, tokens_in=tin, tokens_out=tout, retry=retry)
            save_state(args.state, state)
            print(f"collected {b['id']}: {len(b['keys']) - len(retry)} ok, {len(retry)} to retry",
                  flush=True)
        if not pending or not args.wait:
            break
        print(f"{pending} batch(es) still processing; checking again in {args.poll}s", flush=True)
        time.sleep(args.poll)

    retry = [k for b in state["batches"] for k in b.get("retry", [])]
    if pending:
        print(f"{pending} batch(es) still processing; run collect again later")
    elif args.retry and retry:
        print(f"re-running {len(retry)} pieces through the normal API (full price)", flush=True)
        rows = {json.loads(l)["key"]: json.loads(l) for l in open(args.out)}
        for k in retry:
            try:
                rows[k], _, _ = caption_one(client, pieces[k], state["model"], state["effort"])
            except Exception as e:  # keep going; the piece stays failed or missing
                print(f"  {k}: {type(e).__name__}: {e}", flush=True)
        tmp = Path(f"{args.out}.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
        tmp.replace(args.out)
        for b in state["batches"]:
            b["retry"] = [k for k in b.get("retry", []) if rows.get(k, {}).get("failed", True)]
        save_state(args.state, state)
    summarize(args.out, state)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--data-dir", required=True)
    s.add_argument("--state", required=True)
    s.add_argument("--keys", help="jsonl whose `key` fields pick the pieces (default: all)")
    s.add_argument("--out", help="captions already in this jsonl are skipped")
    s.add_argument("--model", default=MODEL)
    s.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    s.add_argument("--max-mb", type=int, default=200, help="per batch; the API limit is 256")
    t = sub.add_parser("status")
    t.add_argument("--state", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--data-dir", required=True)
    c.add_argument("--state", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--wait", action="store_true", help="poll until every batch has ended")
    c.add_argument("--poll", type=int, default=120, help="seconds between checks with --wait")
    c.add_argument("--retry", action="store_true",
                   help="re-run refused, errored or expired pieces through the normal API")
    args = ap.parse_args()
    {"submit": submit, "status": status, "collect": collect}[args.cmd](args)


if __name__ == "__main__":
    main()
