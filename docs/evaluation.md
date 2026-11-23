# Evaluation and reproduction

**Visual fidelity to the paper remains unverified.** The pipeline follows the paper's matching and constrained-deformation framework. It retains recognizable shapes and negative space on several clean inputs and makes selected strokes more coherent. Results also show broken loops, distracting glyphs, missing small features, and optimization stalls on clean vectors. The published church and portrait results have more aligned outlines than the runs available here. Different glyph banks, a different vectorization, and uncertain annealing details prevent an exact benchmark comparison.

[Gallery](../evaluation/GALLERY.md) · [Method](method.md) · [Inputs and licenses](datasets.md)

## Full runs

All 45 published fixture runs reached the 5,000-proposal patience rule. The separate author-raster run also reached it. Here, “patience” is a stopping reason. The exhaustive move check below found improvements remaining in two states that stopped.

| Group | Runs | Best score lower than initial | Proposals per run |
| --- | ---: | ---: | ---: |
| Controlled diagnostics | 4 | 3 | 5,000–65,486 |
| Paper analogues | 2 | 1 | 5,000–35,893 |
| Frozen Quick, Draw! | 24 | 20 | 5,000–93,690 |
| Width and seed sweep | 8 | 6 | 5,000–55,341 |
| Single-setting experiments | 7 | 2 | 5,000–126,705 |
| Author's Figure 12 raster | 1 | 0 | 5,000 |

The line fixture has the strongest improvement. Its long line becomes a coherent row of underscores. The ring keeps an open center while glyph substitutions leave visible breaks. The kite traps the search at its initial rendering and substitutes two U's along a central stroke. The raster mug preserves its bowl and negative space. Its handle interior and steam are noisy. The train analogue reads as a locomotive with a chimney, cab, and ground line, while the wheel rims merge into the body and the rail.

On Quick, Draw!, the better examples retain the boat hull and sail, the apple silhouette and stem, or the house roof, walls and door. The challenging bicycle with two small wheels and several overlapping frame strokes turns those details into punctuation and letters. Cat eyes and whiskers lose definition. The unrecognized house source is face-like, and its result retains the oval and interior marks, so the class prompt alone would mis-score the output. The gallery keeps each selected input, including the four runs with no accepted moves.

## Checks on matching, deformation and search

- **Raster topology.** The mug has 7,846 thresholded pixels and 1,093 skeleton pixels. The ink mask and skeleton both have three 8-connected components. The traced outline retains the handle holes, body, and separate steam strokes. The traced output contains 16 polylines because junctions split chains. Analytical fixtures cover a junction, a closed contour, isolated ink and a one-pixel gap. Near strokes joined by thresholding can remain joined in the traced topology. [Mug preprocessing](../evaluation/results/main/mug/preprocessing.png)
- **Figure 7 glyph comparisons.** Five manually specified masks probe similar geometries. The long horizontal at row 9 matches a hyphen, and moving it down two pixels changes the best glyph to `=`. The approximated corner, left vertical and top line with spur match `_`, `h` and a double quote, respectively, while the paper reports `L`, `}` and `~`. Exact query pixels and reference font would let us tell descriptor discrepancies from alignment and glyph differences. [Masks and top-five predictions](../evaluation/results/probes.json)
- **First moves.** Exhaustive search found 11 improving moves out of 36,226 valid trials on the kite and 30 out of 92,574 on the chapel. Only 0.03% of uniformly sampled vertex-offset pairs improve either initial state. Removing accessibility gives the kite 42 and the chapel 103 improving moves. The chapel has 1,086 accessibility links, including one only 0.63 matching pixels long. The short links and wide proposal disk both deserve attention. [Probe results](../evaluation/results/probes.json)
- **Width and seed.** At 12 and 24 columns the chapel improves with both seeds, while the 22-column baseline stalls. At 24 columns there is room for the arched doorway and windows. The ring changes little in the 12/24-column sweep despite the clear contour in its initial geometry. [Sweep results](../evaluation/results/sweep.json)
- **Ambiguity controls.** In the 22-column chapel, the printed relative-length term produced a lower within-run best score, starting from a higher identity penalty. Direct-distance accessibility did not improve the result. Using only the printed temperature or reversed acceptance accepted thousands of moves while retaining the initial best. Local-only improved the best objective from 1.938 to 1.833. The geometric schedule and continuous radius proposals accepted more moves than baseline, with no lower best in this run. These are one-case tests that hold the other settings fixed. [Single-setting experiment results](../evaluation/results/ambiguities.json)
- **Available original.** The standalone Figure 12 queen raster at 28 columns yielded 13 rows under the bundled font, compared with the paper's 21 Shift-JIS rows at that width. It has 483 fitted vertices and 471 segments. Annealing accepted four moves and kept the initial state at 2.2854. Crown texture and portions of the face break into small marks while the outer portrait remains visible. The author raster, panels, and text are retained locally. [Numeric run and settings](../evaluation/results/author-source-run.json) · [Author's comparison](https://ttwong12.github.io/papers/asciiart/queen/queen.htm)

A controlled follow-up would use the queen input from the supplement, hold the extracted vectors fixed, match the published 28 × 21 cell geometry and glyph set, and compare the initial matches before tuning the search. The Figure 7 query tiles or source glyph outlines, if obtainable, can check the descriptor independently. With a verified matcher, use an identical proposal stream to compare fixed-link and recomputed-visibility models, and measure window drift and seam continuity as well as Eq. 6.

## Reproduction commands

The recorded runs use macOS 26.5.2 arm64, Python 3.14.7, the bundled Liberation Mono 2.1.5, and the dependency versions in [requirements-repro.txt](../requirements-repro.txt). Run times are wall clock and include any cold start incurred by each worker. The results grids use LF-terminated ASCII rows and keep all trailing spaces.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install --no-deps -e .
python -m pytest -q

# The 24 selected source records and six fixtures are already in the checkout.
python -m evaluation.run diagnostics
python -m evaluation.run paper
python -m evaluation.run real --workers 3
python -m evaluation.run sweep
python -m evaluation.run ambiguities
python -m evaluation.probe
```

The runner writes each run's input checksum, vectors, initial and final exact text, and five-panel comparison. Each `run.json` records configuration, iteration count, accepted count, history every 100 proposals and at the last proposal, and the best iteration. Append `--output /path/to/results` to run in a separate directory. To exercise the optional budget, use `--max-proposals 1000` and a separate output directory. The tests cover fixed-seed equality, patience, and the `budget` stopping reason. Primary runs used no proposal cap.

```sh
# Recreate the fixed inputs from source, only if needed.
python -m evaluation.make_fixtures
python -m evaluation.select_quickdraw

# Recreate the local-only author comparison.
mkdir -p evaluation/local-references
curl --fail --location \
  https://ttwong12.github.io/papers/asciiart/queen/queen.3.gif \
  --output evaluation/local-references/queen.3.gif
python -m evaluation.run source --workers 1 \
  --output evaluation/local-references/results
```

The test suite includes a direct per-pixel AISS implementation, exhaustive candidate matching, analytic angle and length penalties, first-hit visibility and link movement, segment-cell weights, topology preservation, randomized full-versus-incremental scoring, fixed seeds, acceptance, best retention, and process-pool error isolation. Its recorded result was **27 passed**. [Tests](../tests)

Verification checked all 31 input checksums and recomputed the scores and exact ASCII from the full initial and final geometry of all 46 runs. The installed CLI also produced a 19 × 9 PNG and exact text for the mug with a 1,000-proposal budget and reported `budget`.
