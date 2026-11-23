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
