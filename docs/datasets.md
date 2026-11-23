# Sketch data for evaluation

## ImageNet-Sketch

The release uses category-specific Google Image queries and manual cleanup, with some rotations and flips. The publisher describes roughly 50,000 raster images across 1,000 ImageNet classes and provides a full archive through Drive and Kaggle. The repository links to a Hugging Face loader. The archive is several gigabytes. The tested canonical row endpoint refused the request, so inspection used the authors' public preview and six individually downloaded rows from a small mirror. [Publisher repository](https://github.com/HaohanWang/ImageNet-Sketch) · [Small mirror](https://huggingface.co/datasets/nathancday/imagenet_sketch_mini)

| Small-mirror row | Class | Visual inspection |
| --- | --- | --- |
| 0 | Mantis | Readable outline with a source watermark |
| 1 | Plow | Watermarked technical sketch with heavy hatching |
| 2 | Microphone | Readable outline with hatching and a prominent watermark |
| 3 | Vine snake | Shaded snake on branches |
| 4 | Crib | Crib with a child and a toy |
| 5 | Miniskirt | Product-style grayscale image with text |

The mirror schema has `image` and `label`. The classes are generally relevant in these six rows and the main subject is usually identifiable. The annotations supply class names, with no image-specific captions, prompts, original stroke data, or paired ASCII in the inspected schema. The publisher's preview also shows filled silhouettes, outlines and shaded drawings. This is a useful later stress test for edge extraction on varied raster inputs. Its defaults would turn shading, watermarks, and filled regions into spurious centerlines. [Publisher preview](https://github.com/HaohanWang/ImageNet-Sketch/blob/master/imagenet_sketch.jpg) · [Six-row response](https://datasets-server.huggingface.co/rows?dataset=nathancday%2Fimagenet_sketch_mini&config=default&split=train&offset=0&length=6)

The scripts repository displays an MIT license, the large archive mirror says “other,” and the small mirror uses an Apache-2.0 tag. The sampled watermarks make third-party image rights material to redistribution. This repository keeps sample row identifiers and observations. [Script license](https://github.com/HaohanWang/ImageNet-Sketch/blob/master/LICENSE) · [Archive mirror](https://huggingface.co/datasets/nateraw/imagenet-sketch-data) · [Small mirror](https://huggingface.co/datasets/nathancday/imagenet_sketch_mini)

## SVE-ASCII

The supplied Hugging Face ID is published as a model repository with JSON data files. At revision `e77ec942990d323d768370eda9ab6b8cff2dea1f`, the seed file has 1,165 `instruction`, `category`, `art` records. The train card lists 13,668 chat records. Inspection read the whole 376 kB seed file and 98 complete training rows from the first 64 KiB. The first training rows present ASCII in the user message and a description in the assistant message. The inspected seed and train records contain no sketch or photograph fields, image ID, or raster/ASCII pairing. Instructions and descriptions vary from object names to more specific scenes. The baby on stairs and candle samples were intelligible, while other small glyph arrangements are less unambiguous. [Pinned release](https://huggingface.co/SVE-ASCII/SVE-ASCII-Dataset/tree/e77ec942990d323d768370eda9ab6b8cff2dea1f) · [Project](https://sve-ascii.github.io/)

Of the 1,165 seed records, 57 have non-ASCII code points, including box drawing, Braille, and curly punctuation. 1,104 have unequal row lengths after splitting newlines, and the file includes CR line endings in artwork strings. The schema carries no font or display-width metadata. Monospace rendering helps align printable ASCII examples, but Unicode widths and layout need explicit handling. The data can inform aesthetics or support future language-conditioned work once reuse terms are resolved. The pinned release lists no data license or provenance terms and has no license file. The samples and artwork remain outside Git. [Dataset files and metadata](https://huggingface.co/api/models/SVE-ASCII/SVE-ASCII-Dataset)

## Alternatives and selection

- **Quick, Draw!** supplies timestamped raw vectors, simplified x/y strokes, and low-resolution raster renders. The `word` is the game's drawing prompt. `recognized` is the game's classifier flag. Neither is a caption or a guarantee of the depicted subject. The native stroke paths isolate the optimizer and include incomplete or poorly recognized marks. CC BY 4.0 provides clear attribution and redistribution terms. [Google dataset documentation](https://github.com/googlecreativelab/quickdraw-dataset) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **TU Berlin Sketch** offers 20,000 human drawings of 250 categories, with PNG and SVG distributions. It is a candidate for raster/vector comparisons on more detailed contour drawings. The annotations are categories, with no ASCII references. Archive size and the straightforward bounded Quick, Draw! sampling determined the choice for this milestone. [Project](https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) · [Primary paper](https://cybertron.cg.tu-berlin.de/eitz/pdf/2012_siggraph_classifysketch.pdf)
- **The Sketchy Database** associates crowd sketches with specific photographs across 125 categories. The associations are useful for evaluating subject structure against a source photograph. They provide no ASCII reference. The project mixes source photographs and multiple sketch renders, whereas the present experiment needs a small, directly licensed vector set. [Project paper](https://faculty.cc.gatech.edu/~hays/tmp/sketchy-database.pdf) · [Dataset documentation](https://github.com/CDOTAD/SketchyDatabase/blob/master/SketchyDataset_README.md)

## Frozen set and attribution

The selection was frozen before viewing the ASCII results. Categories are apple, bicycle, cat, fish, house, and sailboat. Within the first 64 KiB of each simplified NDJSON file, the sampler took the first three records marked recognized and the first record marked unrecognized, keeping source order. It preserves all 24 selected source records byte-for-byte, including their IDs, timestamp, country code and prompt. Coverage runs from one stroke to 15 strokes and from 22 to 75 source points. All selected records remain in the result set, including the face-like unrecognized `house` and drawings with disconnected outlines.

[Quick, Draw! manifest](../evaluation/quickdraw-manifest.json) pins source URLs and SHA-256 digests. [Diagnostic and paper-analogue manifest](../evaluation/fixtures-manifest.json) records the six original fixtures. Prompt, category and original caption fields remain distinct. The captured Quick, Draw! records carry no captions. Conversion derives a canvas from the maximum x and y, fits and subdivides the original strokes, then rasterizes, matches, and deforms them. Gallery originals are renders of the preserved vector records.

The Quick, Draw! records and their rendered adaptations are attributed to players of *Quick, Draw!* and Google, Inc., which made them available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The gallery identifies the inputs and adds the project’s preprocessed vectors, deformations, rendering, and ASCII interpretations. The author's queen input is recorded separately for a local paper comparison. It comes from the public supplement and has no standalone reuse license on that page. [Queen supplement](https://ttwong12.github.io/papers/asciiart/queen/queen.htm)
