#!/usr/bin/env python3
"""Standalone inference for the IMDB BiLSTM sentiment classifier.

Loads the artifacts written by the notebook (``results/bilstm_model.pt`` and
``results/vocab.pt``) and predicts positive/negative sentiment for any review
text. The architecture is rebuilt from the checkpoint shapes, so the same
script also works with the unidirectional-LSTM baseline if you point it at
that state dict.

Usage
-----
    python predict.py "The plot was sharp and the acting was wonderful."
    python predict.py -f reviews.txt --topk 3
    python predict.py --interactive
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pack_padded_sequence

LABEL_NAMES = {0: "negative", 1: "positive"}


# --------------------------------------------------------------------------
# Preprocessing — kept byte-for-byte consistent with notebook CELL 3 / CELL 4
# --------------------------------------------------------------------------
def clean(text: str) -> list[str]:
    """Lowercase, drop <br /> tags, keep word tokens plus ? and !."""
    text = re.sub(r"<br\s*/?>", " ", text.lower())
    return re.findall(r"[a-z0-9']+|[!?]", text)


def encode(text: str, stoi: dict[str, int], max_len: int) -> tuple[torch.Tensor, int]:
    """Map a raw review to padded token ids, returning ids and true length."""
    ids = [stoi.get(w, 1) for w in clean(text)][:max_len]
    length = max(len(ids), 1)
    padded = ids + [0] * (max_len - len(ids))
    return torch.tensor(padded, dtype=torch.long), length


# --------------------------------------------------------------------------
# Model — identical definition to notebook CELL 5
# --------------------------------------------------------------------------
class LSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=128, layers=2,
                 bidirectional=True, dropout=0.4):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(emb_dim, hidden, num_layers=layers, batch_first=True,
                                  bidirectional=bidirectional,
                                  dropout=dropout if layers > 1 else 0.0)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden * (2 if bidirectional else 1), 1)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.fc(self.dropout(h)).squeeze(1)


def infer_architecture(state_dict: dict) -> dict:
    """Recover emb_dim / hidden / layers / bidirectional from weight shapes."""
    vocab_size, emb_dim = state_dict["embedding.weight"].shape
    hidden = state_dict["lstm.weight_hh_l0"].shape[1]
    # Layer indices live in keys like 'lstm.weight_ih_l0' / '..._l0_reverse';
    # parse the trailing digit rather than splitting, or '_reverse' leaks in.
    layer_ids = {int(m.group(1)) for k in state_dict
                 if (m := re.match(r"lstm\.weight_ih_l(\d+)(?:_reverse)?$", k))}
    if not layer_ids:
        raise ValueError("checkpoint does not look like an LSTMClassifier state dict")
    layers = max(layer_ids) + 1
    bidirectional = any(k.endswith("_reverse") for k in state_dict)
    directions = 2 if bidirectional else 1
    if state_dict["fc.weight"].shape[1] != hidden * directions:
        raise ValueError(f"fc input {state_dict['fc.weight'].shape[1]} does not match "
                         f"hidden={hidden} x directions={directions}")
    return {"emb_dim": emb_dim, "hidden": hidden, "layers": layers,
            "bidirectional": bidirectional, "vocab_size": vocab_size}


# --------------------------------------------------------------------------
# Loading & prediction
# --------------------------------------------------------------------------
def torch_load(path: Path, device: torch.device):
    """torch.load wrapper that works across the weights_only default change."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # torch < 2.4 has no weights_only argument
        return torch.load(path, map_location=device)


def load_predictor(model_path: Path, vocab_path: Path, device: torch.device):
    if not model_path.exists():
        sys.exit(f"[error] model checkpoint not found: {model_path}\n"
                 f"        run the notebook first, or pass --model.")
    if not vocab_path.exists():
        sys.exit(f"[error] vocabulary not found: {vocab_path}\n"
                 f"        run the notebook first, or pass --vocab.")

    vocab = torch_load(vocab_path, device)
    state_dict = torch_load(model_path, device)
    cfg = infer_architecture(state_dict)
    model = LSTMClassifier(cfg["vocab_size"], cfg["emb_dim"], cfg["hidden"],
                           cfg["layers"], cfg["bidirectional"])
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, vocab["stoi"], vocab["itos"], int(vocab["MAX_LEN"])


@torch.no_grad()
def predict_proba(model, texts, stoi, itos, max_len, device, batch_size=64):
    """Return P(positive) for each review; OOV words map to <unk> (id 1)."""
    model.eval()
    probs = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        encoded = [encode(t, stoi, max_len) for t in chunk]
        x = torch.stack([e[0] for e in encoded]).to(device)
        lengths = torch.tensor([e[1] for e in encoded]).to(device)
        probs.extend(torch.sigmoid(model(x, lengths)).cpu().tolist())
    return probs


def oov_report(texts, stoi) -> tuple[int, int]:
    """Count tokens falling back to <unk>, useful for domain shift."""
    vocab = set(stoi)
    total = sum(len(clean(t)) for t in texts)
    unknown = sum(1 for t in texts for w in clean(t) if w not in vocab)
    return unknown, total


def format_row(text: str, p_pos: float, topk: int = 0,
               stoi=None, itos=None) -> str:
    label = LABEL_NAMES[int(p_pos >= 0.5)]
    conf = p_pos if p_pos >= 0.5 else 1.0 - p_pos
    lines = [f"{label:<8} p(positive)={p_pos:.4f}  confidence={conf:.4f}",
             f"  review: {text.strip()[:110]}{'...' if len(text.strip()) > 110 else ''}"]
    if topk and stoi and itos:
        vec = model_embedding_norms(stoi, itos, topk, clean(text))
        if vec:
            lines.append("  top tokens: " + ", ".join(vec))
    return "\n".join(lines)


def model_embedding_norms(stoi, itos, topk, tokens) -> list[str]:
    """Surface the highest-id (rarest kept) in-vocab tokens for transparency."""
    ranked = sorted({t for t in tokens if t in stoi and stoi[t] > 1},
                    key=lambda w: stoi[w], reverse=True)
    return [f"{w}#{stoi[w]}" for w in ranked[:topk]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reviews", nargs="*", help="one or more review texts")
    ap.add_argument("-f", "--file", type=Path,
                    help="text file with one review per line")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="read reviews from stdin, one per line")
    ap.add_argument("--model", type=Path, default=None,
                    help="path to state dict (default: results/bilstm_model.pt)")
    ap.add_argument("--vocab", type=Path, default=None,
                    help="path to vocab.pt (default: results/vocab.pt)")
    ap.add_argument("--results-dir", type=Path, default=Path("results"),
                    help="directory holding the notebook artifacts")
    ap.add_argument("--topk", type=int, default=0,
                    help="also echo the N rarest in-vocabulary tokens")
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (auto-detected)")
    args = ap.parse_args(argv)

    texts = list(args.reviews)
    if args.file:
        texts += [ln for ln in args.file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.interactive:
        texts += [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]
    if not texts:
        ap.error("no reviews given: pass them as arguments, via --file, or --interactive")

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = args.model or args.results_dir / "bilstm_model.pt"
    vocab_path = args.vocab or args.results_dir / "vocab.pt"

    model, stoi, itos, max_len = load_predictor(model_path, vocab_path, device)
    cfg = infer_architecture(model.state_dict())
    direction = "BiLSTM" if cfg["bidirectional"] else "LSTM"
    print(f"loaded {direction} ({cfg['layers']} layer, hidden={cfg['hidden']}, "
          f"emb={cfg['emb_dim']}) on {device} | vocab={len(itos)}, max_len={max_len}\n")

    probs = predict_proba(model, texts, stoi, itos, max_len, device)
    for text, p in zip(texts, probs):
        print(format_row(text, p, args.topk, stoi, itos))
        print()

    unknown, total = oov_report(texts, stoi)
    pos = sum(1 for p in probs if p >= 0.5)
    print(f"summary: {pos} positive / {len(probs) - pos} negative "
          f"| OOV rate {unknown}/{total} tokens ({100 * unknown / max(total, 1):.1f}%)")
    if unknown / max(total, 1) > 0.2:
        print("warning: high OOV rate - predictions are less reliable for this input")
    return 0


if __name__ == "__main__":
    sys.exit(main())
