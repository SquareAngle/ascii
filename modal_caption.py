"""Caption the cleaned dataset with a VLM on Modal (vLLM + Qwen3-VL by default).

Reads processed/{train,val}.jsonl from the `ascii` volume (see modal_train.py for the
upload) and writes captions/<out>.jsonl there, one row per piece:
    {key, source, category, title, split, h, w, short, medium, long}  (captioned)
    {..., unclear: true}                                               (VLM couldn't tell)
    {..., failed: true, raw: "..."}                                    (unparseable output)
`key` is artscii.caption.vlm.piece_key(art), the same hash clean.py deduplicates on.

Pilot: a sample spread evenly over the four sources, plus an HTML review page:
    modal run modal_caption.py --limit 20
    modal volume get ascii captions/pilot.html .

Full run (resumes: pieces already in the output file are skipped):
    modal run --detach modal_caption.py --limit 0 --out captions

By default only pieces that fit the 64x32 training canvas are captioned; pass
--max-h 0 --max-w 0 for all of them.
"""
import os

import modal

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
GPU = "L40S"  # 48 GB: the 8B model in bf16 plus a large KV cache for batching

app = modal.App("artscii-caption")
VOLUME = os.environ.get("ARTSCII_VOLUME", "ascii")  # dataset + captions volume
data_vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)  # model weights
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("fonts-dejavu-core")
    .pip_install("vllm==0.11.2", "pillow")
    .env({"HF_HOME": "/hf"})
    .add_local_python_source("artscii")
)


def load_pieces(max_h, max_w):
    import json

    rows = []
    for split in ("train", "val"):
        with open(f"/vol/processed/{split}.jsonl") as f:
            rows += [json.loads(l) for l in f]
    return [r for r in rows if (not max_h or r["h"] <= max_h) and (not max_w or r["w"] <= max_w)]


def stratified(pieces, n, seed):
    """n pieces, as evenly as possible across sources (small sources contribute all they have)."""
    import random

    rng = random.Random(seed)
    by_source = {}
    for p in pieces:
        by_source.setdefault(p["source"], []).append(p)
    for group in by_source.values():
        rng.shuffle(group)
    out, groups = [], sorted(by_source.values(), key=len)
    for i, group in enumerate(groups):
        take = (n - len(out)) // (len(groups) - i)
        out += group[:take]
    rng.shuffle(out)
    return out


@app.function(image=image, gpu=GPU, volumes={"/vol": data_vol, "/hf": hf_vol},
              timeout=24 * 3600, memory=65536)
def caption(limit: int, out: str, model: str, max_h: int, max_w: int, seed: int,
            chunk: int = 512):
    import json
    from pathlib import Path

    from vllm import LLM, SamplingParams

    from artscii.caption.vlm import (build_messages, parse_response, piece_key, png_b64,
                                     render_png, write_report)

    data_vol.reload()
    pieces = load_pieces(max_h, max_w)
    pilot = limit > 0
    if pilot:
        pieces = stratified(pieces, limit, seed)
    out_dir = Path("/vol/captions")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{out}.jsonl"
    done = set()
    if out_path.exists() and not pilot:
        done = {json.loads(l)["key"] for l in open(out_path)}
    todo = [p for p in pieces if piece_key(p["art"]) not in done]
    print(f"{len(pieces)} pieces selected, {len(done)} already captioned, {len(todo)} to go",
          flush=True)

    llm = LLM(model=model, max_model_len=8192, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.92)
    params = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=300, seed=seed)
    report_rows, counts = [], {"captioned": 0, "unclear": 0, "failed": 0}
    with open(out_path, "w" if pilot else "a") as fh:
        for start in range(0, len(todo), chunk):
            batch = todo[start:start + chunk]
            pngs = [png_b64(render_png(p["art"])) for p in batch]
            outputs = llm.chat([build_messages(p, b) for p, b in zip(batch, pngs)], params,
                               use_tqdm=False)
            for p, b, o in zip(batch, pngs, outputs):
                raw = o.outputs[0].text
                parsed = parse_response(raw)
                row = dict(key=piece_key(p["art"]), source=p["source"], category=p["category"],
                           title=p["title"], split=p["split"], h=p["h"], w=p["w"])
                row.update(parsed if parsed else dict(failed=True, raw=raw))
                counts["failed" if not parsed else "unclear" if parsed.get("unclear")
                       else "captioned"] += 1
                fh.write(json.dumps(row) + "\n")
                if pilot:
                    report_rows.append(dict(row, png=b))
            fh.flush()
            data_vol.commit()
            print(f"{start + len(batch)}/{len(todo)} {counts}", flush=True)
    if pilot:
        write_report(out_dir / f"{out}.html", report_rows)
    data_vol.commit()
    return counts


@app.local_entrypoint()
def main(limit: int = 20, out: str = "", model: str = MODEL, max_h: int = 32,
         max_w: int = 64, seed: int = 0):
    out = out or ("pilot" if limit > 0 else "captions")
    print(caption.remote(limit, out, model, max_h, max_w, seed))
    print(f"modal volume get {VOLUME} captions/{out}.jsonl .")
    if limit > 0:
        print(f"modal volume get {VOLUME} captions/{out}.html .")
