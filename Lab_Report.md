# Lab Report — Sentiment Analysis of IMDB Movie Reviews Using a Bidirectional LSTM

**Student Name:** Swapnil Santosh Akhade
**PRN:** 202401110002
**Department:** AIML
**Course:** GenAI — Practical 2
**Submission copy:** the Word/PDF version of this report is kept locally in `submission/`
(which is Git-ignored); this Markdown file is the repository copy and the source for it.

---

## 1. Aim

To develop and evaluate a Bidirectional LSTM model for binary sentiment
classification of movie reviews.

## 2. Objectives

1. Load and explore the IMDB movie-review dataset.
2. Preprocess text data and convert reviews into numerical sequences.
3. Implement and train a Bidirectional LSTM classifier.
4. Evaluate the classifier using standard classification metrics.
5. Compare the BiLSTM model with a unidirectional LSTM and a TF-IDF-based
   Logistic Regression model.
6. Analyse classification errors and visualize learned word embeddings.

## 3. Dataset and Tools

| Item | Detail |
|---|---|
| Dataset | Stanford IMDB movie-review dataset, loaded via Hugging Face `datasets` (`stanfordnlp/imdb`) |
| Size | 50,000 reviews — 25,000 train / 25,000 test, balanced (12,500 per class) |
| Labels | Binary: `0 = negative`, `1 = positive` |
| Language / framework | Python 3, PyTorch |
| Supporting libraries | NumPy, Pandas, scikit-learn, Matplotlib |
| Environment | Google Colab, T4 GPU runtime (auto-falls back to CPU) |
| Random seed | `SEED = 2026` (Python, NumPy, PyTorch, and the split's `random_state`) |

No external dataset upload is required: the notebook downloads the corpus itself.

## 4. Methodology

### 4.1 Text preprocessing

`clean()` lowercases each review, replaces `<br />` tags with spaces, and tokenizes
with the regex `[a-z0-9']+|[!?]`, which retains negations (`n't`, `don't`) and
sentence-level punctuation (`?`, `!`) that carry sentiment signal.

### 4.2 Exploratory data analysis

Three diagnostics are plotted to `results/eda_overview.png`: the class distribution
(exactly balanced, so accuracy is a fair primary metric), the token-length
histogram, and length-by-class boxplots. Training reviews average **235 tokens**
(median 176), which sits under the `MAX_LEN = 300` cap — yet roughly **23% of reviews
are longer than 300 tokens** and get truncated, which is a stated limitation later.

### 4.3 Splitting, vocabulary and encoding

- The 25,000-review training partition is split **90/10** into train/validation with
  `stratify` on the label, leaving 22,500 train / 2,500 validation; the official
  25,000-review test partition is untouched until final evaluation.
- Vocabulary = the 20,000 most frequent tokens from `Counter`, prefixed by
  `<pad>` (id 0) and `<unk>` (id 1).
- Each review is mapped to ids (out-of-vocabulary → `<unk>`), truncated to 300, and
  right-padded into a dense `TensorDataset`. True lengths are carried alongside so
  `pack_padded_sequence` can skip padding. Batches of 64 (train, shuffled) and 256
  (validation).

### 4.4 Model architecture

```
Embedding(20000, 128, padding_idx=0)
  → Dropout(0.4)
  → LSTM(128 → 128, num_layers=2, bidirectional=True, dropout=0.4)  [packed input]
  → concat(final fwd, final bwd) = 256-d
  → Dropout(0.4)
  → Linear(256 → 1)   # single logit
```

**3,219,713 trainable parameters.** The embedding matrix is learned end-to-end rather
than initialized from pre-trained vectors. Bidirectionality lets the final hidden state
summarize context in both directions, which matters for constructions such as
*"not bad at all"*.

### 4.5 Training setup

| Hyperparameter | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` (logit + sigmoid fused for stability) |
| Optimizer | Adam, learning rate `1e-3` |
| Epochs | 8 (max) |
| Batch size | 64 |
| Dropout | 0.4 |
| Gradient clipping | `clip_grad_norm_`, max-norm 1.0 |
| Early stopping | patience 3 on validation loss; best state dict restored |

### 4.6 Training behaviour

Validation loss dropped quickly, then became non-monotonic — it rose at epochs 4 and 5
(0.4117, 0.4960) while training loss kept falling, the signature of mild over-fitting,
before recovering for the last three epochs. Early stopping (patience 3) was therefore
never triggered and all 8 epochs ran; epoch 8 finished with the best validation loss,
so the restored best state is the final model.

| Epoch | Train loss | Train acc | Val loss | Val acc | Best |
|---:|---:|---:|---:|---:|:--:|
| 1 | 0.6278 | 0.6356 | 0.6334 | 0.6832 | yes |
| 2 | 0.5397 | 0.7282 | 0.4780 | 0.7908 | yes |
| 3 | 0.4272 | 0.8066 | 0.3826 | 0.8420 | yes |
| 4 | 0.3689 | 0.8431 | 0.4117 | 0.8532 | — |
| 5 | 0.3173 | 0.8677 | 0.4960 | 0.8184 | — |
| 6 | 0.2920 | 0.8771 | 0.3224 | 0.8812 | yes |
| 7 | 0.2460 | 0.9013 | 0.3026 | 0.8900 | yes |
| 8 | 0.2278 | 0.9086 | **0.2756** | **0.8912** | yes |

Training took ~17–19 s per epoch (138 s total) on a T4 GPU. Curves:
`results/training_curves.png`.

## 5. Results

### 5.1 BiLSTM on the held-out test set (25,000 reviews)

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| negative | 0.8565 | 0.8978 | 0.8767 | 12,500 |
| positive | 0.8926 | 0.8496 | 0.8706 | 12,500 |
| **accuracy** | | | **0.8737** | 25,000 |
| macro avg | 0.8745 | 0.8737 | 0.8736 | 25,000 |

ROC-AUC = **0.9479**. Confusion matrix: **11,222 TN, 1,278 FP, 1,880 FN, 10,620 TP**
→ 3,158 errors (12.6%). Full report: `results/classification_report.txt`;
plots: `results/evaluation.png`.

Positive-class precision (0.8926) exceeds its recall (0.8496): when the model labels a
review positive it is usually right, but it misses proportionally more positive reviews.

### 5.2 Comparison with baselines

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **BiLSTM (2 layers)** | 0.8737 | **0.8926** | 0.8496 | 0.8706 | 0.9479 |
| LSTM (1 layer, unidirectional) | 0.8428 | 0.8157 | **0.8858** | 0.8493 | 0.9207 |
| TF-IDF + Logistic Regression | **0.8866** | 0.8871 | 0.8860 | **0.8865** | **0.9551** |

Two findings:

1. **BiLSTM > unidirectional LSTM** on accuracy (+3.09 pts), precision, F1 and ROC-AUC,
   confirming that bidirectional context helps; the unidirectional model only wins on
   recall.
2. **TF-IDF + Logistic Regression beats the neural model** on accuracy, recall, F1 and
   ROC-AUC. This is reported rather than hidden: on IMDB, bag-of-words features with a
   linear classifier are a famously strong baseline, and a recurrent network is not the
   best model on every metric. The BiLSTM retains the highest precision.

Chart: `results/comparison.png`, raw numbers `results/model_comparison.csv`.

### 5.3 Error analysis

Representative failures in `results/error_analysis.txt` are confidently wrong
(P(positive) = 1.000) and share one pattern — locally positive wording with an overall
negative verdict:

- *"I found 'Strange Fruit' to be an excellent movie… but for a low-budget movie that is
  to be expected!"* → labelled negative.
- *"This movie was pure genius. John Waters is brilliant…"* → labelled negative
  (sarcastic / satirical rating).
- *"Masterpiece. Carrot Top blows the screen away… 10/10"* → labelled negative (sarcasm).

So the residual errors are driven by sarcasm, contrastive "yes, but" structure and
reviewer irony, not by token-level noise. Truncation of the 23% of long reviews removes
the closing sentences where the final verdict often appears, compounding this.

### 5.4 Embedding analysis

Cosine-similarity neighbours of seed words (`results/embedding_neighbours.txt`) show the
layer is only partially semantic — as expected for an embedding trained from scratch on
20k tokens with a modest amount of signal:

- `excellent → magnificent, spellbinding, pet, jerome, inexperienced`
- `waste → disappointing, disappointment, pretentious, nada`
- `brilliant → uber, aviation, serrault, perfectly, king`
- `boring → grades, awful, cautionary, unrealistic`

Correct sentiment clusters appear (`waste`/`boring` gather negative terms), but
frequency-driven artefacts (`pet`, `jerome`, `grades`) intrude. The PCA projection of
normalized embeddings (`results/embeddings.png`) gives a visible but overlapping
positive/negative separation; it is exploratory evidence, not proof of a clean split.

## 6. Conclusion

A binary text-classification model was implemented with a trainable Embedding layer and
a 2-layer Bidirectional LSTM, trained on the IMDB training partition with early stopping
and evaluated on the separate 25,000-review test partition using accuracy, precision,
recall, F1, ROC-AUC, a classification report and confusion matrices. The model attained
**87.37% test accuracy** and **0.9479 ROC-AUC**, and outperformed a unidirectional LSTM
on most metrics while being outperformed by a TF-IDF + Logistic Regression baseline — a
comparison that illustrates why multiple metrics and a traditional baseline are needed
when assessing a classifier. All six stated objectives are satisfied, so the practical
requirement of building and evaluating an Embedding–LSTM text classifier is met.

## 7. Artifacts and Reproduction

| Output | File |
|---|---|
| Trained weights (state dict) | `results/bilstm_model.pt` |
| Vocabulary + `MAX_LEN` | `results/vocab.pt` |
| EDA / training / evaluation / comparison / embedding plots | `results/*.png` |
| Metrics tables | `results/test_metrics.csv`, `results/model_comparison.csv` |
| Text reports | `results/classification_report.txt`, `results/error_analysis.txt`, `results/embedding_neighbours.txt` |

Reproduce end-to-end by running `notebooks/GenAI_Practical_2_IMDB_Sentiment_Analysis.ipynb`
top-to-bottom on a Colab GPU runtime (~17 s/epoch). To classify new text without
retraining:

```bash
python predict.py "The plot was sharp and the acting was wonderful."
```

## 8. Viva / defence notes

- **Why `BCEWithLogitsLoss` instead of `Sigmoid` + `BCELoss`:** numerical stability via
  log-sum-exp, and `logits > 0` is a cheap threshold at evaluation time.
- **Why `pack_padded_sequence`:** prevents padded positions from contributing LSTM state;
  `enforce_sorted=False` lets shuffled batches stay shuffled.
- **Why `padding_idx=0`:** keeps the `<pad>` embedding at zero so it never receives
  gradient-driven meaning.
- **Why stratified 90/10:** the validation set must keep the class balance for validation
  loss/accuracy to be comparable to test.
- **Why best-state restore matters here:** validation loss was non-monotonic (epochs 4–5),
  so a fixed-epoch stop could have returned an over-fitted state; in this run the best
  validation loss happened to land on the final epoch.
- **Why the TF-IDF baseline matters:** it keeps the report honest about whether a BiLSTM
  is actually the right tool for this dataset.
