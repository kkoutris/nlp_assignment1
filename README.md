# IMDb sentiment classification

Binary sentiment classification on [`mteb/imdb`](https://huggingface.co/datasets/mteb/imdb) for the NLP1 projects,
evaluated with cross-validation and a Bayesian analysis against a region of practical equivalence (ROPE).

| Notebook | Project |
|---|---|
| **`imdb_sentiment.ipynb`** | Project 1: sparse text representations (token indicator, bag-of-words, TF-IDF) with log-linear and MLP classifiers |
| **`imdb_sentiment_part2.ipynb`** | Project 2: pretrained word embeddings, embed-and-pool and recurrent encoders, on the same data, folds and protocol |

Each notebook holds its whole pipeline inline: data, tokeniser, models, tuning, cross-validation, Bayesian comparison,
test evaluation and error analysis.

## Running

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab imdb_sentiment.ipynb          # or imdb_sentiment_part2.ipynb
```

Slow stages save to `results/` (Project 1) or `results_part2/` (Project 2) and skip work already done, so a run from
the saved results takes a few minutes plus the dataset download. To redo everything, set `RECOMPUTE = True` in the
setup cell. That writes to `results_recomputed/` or `results_part2_recomputed/`. Project 1 takes about 1.5 hours on a
laptop CPU. Project 2 ran in under an hour on one A100 GPU and takes several hours on a laptop.

Project 2 needs Project 1's `results/` (same folds, comparisons, error analysis). On its first run it downloads the
GloVe and fastText tables (2.7 GB). It keeps only the rows for IMDb words in `cache.nosync/`, which is not in git.

## Layout

| Path | Contents |
|---|---|
| `imdb_sentiment.ipynb` | The full pipeline, with all project code inline |
| `nlp1_tutorial/` | Tutorial modules used by the notebook, copied unchanged (see Credits) |
| `results/vocab_sweep.csv`, `results/tokenizers/` | Tokeniser candidates and their dev scores |
| `results/tuning.csv`, `results/cv_scores.csv` | Dev tuning and 5-fold × 3-seed CV scores for the nine systems (three architectures × three representations) |
| `results/cv_predictions/` | Validation predictions for every fold, for error analysis |
| `results/test_results.json`, `results/test_predictions.npz` | The single test evaluation |
| `results/figures/` | Figures written by the notebook |
| `results/logs/` | Logs from the original runs |
| `part2/` | Tutorial 2 modules used by the Part 2 notebook (`clslib.py`, `analysis.py`, ...), and the assignment |
| `results_part2/dev_runs.csv`, `results_part2/dev_histories.jsonl` | Every dev-set run of the Part 2 search, with per-epoch losses |
| `results_part2/cv_scores.csv`, `results_part2/cv_predictions/` | CV scores and validation predictions of the two Part 2 finalists |
| `results_part2/test_results.json`, `results_part2/test_predictions.npz` | The single Part 2 test evaluation |
| `results_part2/coverage.csv`, `bda_posterior.npz`, `reproducibility.json`, `figures/` | Embedding coverage, error-analysis posterior, refit check, figures |
| `AI_USAGE.md` | Record of AI assistance |

## Results, Project 1

Mean CV accuracy over 15 folds:

| | Token indicator | Bag-of-words | TF-IDF |
|---|---|---|---|
| Log-linear | 0.8714 | 0.8723 | 0.8813 |
| MLP, 1 hidden layer | 0.8846 | **0.8885** | 0.8913 |
| MLP, 2 hidden layers | 0.8839 | 0.8874 | 0.8911 |

- **Selected system:** at ROPE 0.005, the one-hidden-layer MLP with bag-of-words. The selected representation changes
  with the ROPE: TF-IDF at 0.0025, token indicator at 0.01.
- **Architectures:** the MLP beats log-linear on every representation (by 0.010 to 0.016); a second hidden layer
  ties with one.
- **Test:** single models average 0.8803, and a majority vote of the 15 fold models reaches 0.8853.

## Results, Project 2

Mean CV accuracy over the same 15 folds:

| System | CV accuracy |
|---|---|
| Embed-and-pool: fastText, mean pooling, fine-tuned embeddings | **0.8961** |
| GRU, max readout, frozen fastText, first 400 tokens | 0.8936 |
| Project 1 best: MLP on TF-IDF | 0.8913 |
| Project 1 selected: MLP on bag-of-words | 0.8885 |

- **Embedding:** fastText wiki-news 300d beat GloVe 2024 300d and 50d on the task (0.842 against 0.818 and 0.73
  frozen), despite lacking `n't` and ratings such as `7/10`.
- **Embed-and-pool:** training the embeddings gains 4.3 points; random embeddings trained from scratch do almost as
  well. Max pooling is poor.
- **Recurrent:** max readout matters most. Fine-tuning, bidirectionality and longer inputs don't help. Pretrained
  embeddings add 5 points with 1,120 training reviews but nothing with all 22,402.
- **Selected system:** embed-and-pool. It ties with the GRU (P(tie) 0.63), beats Project 1's selected system
  (P = 0.81) and ties with its best (+0.47 points).
- **Test:** a majority vote of the 15 fold models reaches 0.8915, against 0.8853 for Project 1 (exact McNemar
  p = 0.0003). Single models average 0.8829.
- **Errors:** most come from reviews whose opinion words point the other way. Word order helps with negation; the GRU
  is hurt by contrast words and truncation.

## Credits

`nlp1_tutorial/` contains `analysis.py`, `clslib.py`, `experiment.py`, `loglin.py` and `smlp.py`, copied without
changes from the NLP1 2026 tutorial repository (`part1/`). They are under the Apache License 2.0 (`nlp1_tutorial/LICENSE`).
`part2/` holds the tutorial's `part2/` modules (`analysis.py`, `clslib.py`, `experiment.py`, `utils.py`), also unchanged.
Embeddings: GloVe 2024 Wikipedia + Gigaword (Stanford NLP) and fastText `wiki-news-300d-1M` (Mikolov et al., 2018).
Opinion lexicon: Hu and Liu (2004), through NLTK.
Data: Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*, through the `mteb/imdb` copy on Hugging Face.
