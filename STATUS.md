# Status

Live page: https://claude.ai/artifact/SZq8b7t27YAFr5btBxMVKD (updated at each milestone).

- [x] uv env (`scripts/setup_env.sh`), torch 2.7.1 cu126, V100 checked
- [x] Scrapers: asciiart.eu gallery + Usenet, ascii.co.uk, ascii-art.de (`artscii/scrape/`)
- [x] Scraping: all sources done (Usenet: 38,499 threads, 122,805 posts)
- [x] Cleaning: 78,781 pieces kept, 50,676 fit 64×32 (Usenet 41,297, ascii-art.de 5,653, asciiart.eu 3,409, ascii.co.uk 317); synced to /scratch
- [x] Model / diffusion / training loop (`artscii/train.py`), unit checks (`python -m tests.test_core`)
- [x] Overfit test (64 pieces, loss → 0.002)
- [x] Small run `uncond_small_v1` (job 61700): val 0.162; samples best at temperature 1.0
- [x] Scaling sweep (15k steps): val tiny 0.1785, small 0.1703, base 0.1635, large 0.1592; plot runs/scaling.png; samples and inpainting in runs/sweep_*
