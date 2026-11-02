# Sentiment Analysis of IMDB Movie Reviews Using a Bidirectional LSTM

GenAI practical assignment — binary sentiment classification of 50,000 IMDB movie
reviews with a 2-layer Bidirectional LSTM trained in PyTorch and compared against
two baselines.

| Field | Detail |
|---|---|
| **Student** | Swapnil Santosh Akhade |
| **PRN** | 202401110002 |
| **Department** | AIML |
| **Course** | GenAI — Practical 2 |
| **Environment** | Google Colab (GPU runtime recommended) |
| **Framework** | PyTorch + Hugging Face Datasets + scikit-learn |

## Headline results (25,000-review test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **BiLSTM (2 layers)** | **0.8737** | **0.8926** | 0.8496 | 0.8706 | **0.9479** |
| LSTM (1 layer, unidirectional) | 0.8428 | 0.8157 | 0.8858 | 0.8493 | 0.9207 |
| TF-IDF + Logistic Regression | 0.8866 | 0.8871 | 0.8860 | 0.8865 | 0.9551 |

Confusion matrix (BiLSTM): 11,222 TN / 1,278 FP / 1,880 FN / 10,620 TP — 3,158 of
25,000 reviews misclassified (12.6%). The full write-up is in [`Lab_Report.md`](Lab_Report.md).

## Repository layout

```
.
├── notebooks/
│   └── GenAI_Practical_2_IMDB_Sentiment_Analysis.ipynb   # the assignment (14 sections, run top-to-bottom)
├── results/                       # artifacts produced by the executed notebook
│   ├── eda_overview.png           # class balance, review-length distribution
│   ├── training_curves.png        # loss / accuracy over 8 epochs
│   ├── evaluation.png             # confusion matrices, ROC curve
│   ├── comparison.png             # baseline comparison chart
│   ├── embeddings.png             # PCA of learned word embeddings
│   ├── classification_report.txt  # per-class precision / recall / F1
│   ├── error_analysis.txt         # sample misclassified reviews
│   ├── embedding_neighbours.txt   # cosine-similarity nearest neighbours
│   ├── test_metrics.csv           # BiLSTM metrics
│   ├── model_comparison.csv       # all three models, full precision
│   ├── bilstm_model.pt            # trained weights (state dict, ~12.3 MB)
│   └── vocab.pt                   # stoi / itos / MAX_LEN
├── predict.py                     # standalone inference using the saved checkpoint
├── Lab_Report.md                  # aim / methodology / results / conclusion
├── requirements.txt               # pinned environment
└── submission/                    # local only — Word/PDF report (git-ignored)
```

`submission/` is intentionally excluded from Git, so your college-format `.docx` /
`.pdf` never reaches GitHub. Everything else is committed, including the `.pt`
checkpoints, so the repo is self-contained and `predict.py` runs without retraining.

## Quick start

### Option A — Google Colab (recommended)

1. Open [`notebooks/GenAI_Practical_2_IMDB_Sentiment_Analysis.ipynb`](notebooks/GenAI_Practical_2_IMDB_Sentiment_Analysis.ipynb) and upload it to Colab.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Run all cells top-to-bottom. Dependencies and the Stanford IMDB dataset are
   installed/downloaded automatically; artifacts land in `/content/results/` and are
   zipped as `results.zip` in the final cell. Training took ≈17 s/epoch on a T4 GPU
   (8 epochs ≈ 2.5 min for the BiLSTM).

### Option B — locally

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
jupyter lab notebooks/GenAI_Practical_2_IMDB_Sentiment_Analysis.ipynb
```

Without a GPU the notebook falls back to CPU automatically, but training the two
recurrent models takes considerably longer.

## Try the trained model

No training needed — the checkpoint is committed:

```bash
python predict.py "The plot was sharp and the acting was wonderful."
python predict.py -f my_reviews.txt --topk 3      # one review per line
python predict.py --interactive                   # type reviews at the prompt
python predict.py --model results/bilstm_model.pt --device cpu
```

The script rebuilds the architecture from the checkpoint shapes, so it also loads
the unidirectional-LSTM state dict, and it warns when an input's out-of-vocabulary
rate exceeds 20%.

## What the notebook covers

| § | Section | Highlights |
|---|---|---|
| 1–2 | Setup, seeds, dataset loading | `SEED = 2026`, Hugging Face `stanfordnlp/imdb` |
| 3 | Cleaning & EDA | lowercase + `<br />` strip + regex tokenizer; length stats |
| 4 | Split, vocabulary, encoding | 90/10 stratified train/val, 20k vocab, pad to 300 |
| 5 | BiLSTM architecture | Embedding(20000,128) → 2-layer BiLSTM(128) → dropout 0.4 → linear; 3,219,713 params |
| 6–7 | Training & curves | Adam, lr 1e-3, batch 64, 8 epochs, grad-norm clip 1.0, early stopping (patience 3) with best-state restore |
| 8 | Evaluation | accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| 9 | Baseline comparison | unidirectional LSTM and TF-IDF + Logistic Regression |
| 10 | Error analysis | qualitative look at confidently wrong predictions |
| 11 | Embedding analysis | cosine neighbours + PCA projection |
| 12–14 | Results, checkpoints, export | discussion, `bilstm_model.pt`, `vocab.pt`, `results.zip` |

## Notes & limitations

- Reviews are truncated at 300 tokens; about 23% of training reviews are longer, so
  some context is lost.
- The TF-IDF + Logistic Regression baseline edges out the BiLSTM on accuracy, recall,
  F1 and ROC-AUC — the neural model wins on precision. Both are reported honestly.
- Mixed-sentiment and sarcastic reviews drive most errors (see `results/error_analysis.txt`).

## Reproducibility

`SEED = 2026` is fixed for Python, NumPy and PyTorch, and the train/val split uses
`random_state=SEED`, so metrics are stable across runs. Small deviations on CUDA are
normal because cuDNN kernels are non-deterministic by default.
