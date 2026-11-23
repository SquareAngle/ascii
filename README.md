# ascii art generation
Text-to-ASCII art generation using masked diffusion

## Training on Modal
`modal_train.py` and `modal_eval.py` run training and evaluation on Modal GPUs, with data and checkpoints kept on a Modal Volume named `artscii`. Setup and usage are in each file's docstring:

```bash
modal volume put artscii ~/artscii_data/processed /processed
modal run --detach modal_train.py --overrides "name=uncond_small_modal"
modal run modal_eval.py --runs uncond_small_modal
modal volume get artscii runs/uncond_small_modal ./runs/
```
