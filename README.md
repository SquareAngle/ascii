# ascii art generation
Text-to-ASCII art generation using masked diffusion

## Training on Modal
`modal_train.py`, `modal_eval.py` and `modal_caption.py` run training, evaluation and VLM captioning on Modal GPUs, with data and checkpoints kept on a Modal Volume named `artscii`. Setup and usage are in each file's docstring:

```bash
modal volume put artscii ~/artscii_data/processed /processed
modal run --detach modal_train.py --overrides "name=uncond_small_modal"
modal run modal_eval.py --runs uncond_small_modal
modal volume get artscii runs/uncond_small_modal ./runs/
modal run modal_caption.py --limit 20   # caption pilot -> captions/pilot.html
```

## Text conditioning
Setting `captions` in the config (see `configs/cond_small.yaml`) trains a text-conditioned model:
- A frozen FLAN-T5-large encoder (`artscii/model/text.py`) embeds each piece's caption every step. Each time a piece is loaded, one of its short, medium or long captions is picked at random.
- Every transformer block cross-attends to the caption.
- 10% of captions, plus pieces without a usable caption, are replaced by a learned null condition. That is what makes classifier-free guidance possible at sampling time.

```bash
modal volume put artscii captions.jsonl /captions/captions.jsonl
modal run --detach modal_train.py --config configs/cond_small.yaml
modal run modal_eval.py --runs cond_small --prompts "a cat;a sailboat on the water" --guidance 2
```

Tests: `python -m tests.test_cond`.
