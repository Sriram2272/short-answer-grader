"""Modern models: SBERT similarity and a fine-tuned cross-encoder, same question-grouped folds.

Needs Hugging Face access to download the models (run on Google Colab with a T4 GPU):
    python -m src.transformer --data data/raw/mohler.csv
Writes reports/transformer_results.csv, to set beside reports/results.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import folds_by_question, load
from .metrics import report

SBERT = "sentence-transformers/all-MiniLM-L6-v2"
CROSS = "cross-encoder/stsb-roberta-base"


def sbert_scores(df: pd.DataFrame) -> np.ndarray:
    """Cosine of SBERT embeddings, rescaled per fold by a linear fit on the training questions."""
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LinearRegression

    m = SentenceTransformer(SBERT)
    ref = m.encode(df["reference_answer"].tolist(), normalize_embeddings=True, batch_size=64)
    ans = m.encode(df["student_answer"].tolist(), normalize_embeddings=True, batch_size=64)
    cos = (ref * ans).sum(1).reshape(-1, 1)
    out = np.zeros(len(df))
    for tr, te in folds_by_question(df):
        out[te] = LinearRegression().fit(cos[tr], df["score"].iloc[tr]).predict(cos[te])
    return np.clip(out, 0, 5)


def cross_encoder_scores(df: pd.DataFrame, epochs: int = 3) -> np.ndarray:
    """A cross-encoder fine-tuned on (reference, answer) -> score/5, one model per fold."""
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader

    out = np.zeros(len(df))
    for i, (tr, te) in enumerate(folds_by_question(df)):
        train = [InputExample(texts=[r.reference_answer, r.student_answer], label=r.score / 5)
                 for r in df.iloc[tr].itertuples()]
        model = CrossEncoder(CROSS, num_labels=1)
        model.fit(train_dataloader=DataLoader(train, shuffle=True, batch_size=16),
                  epochs=epochs, warmup_steps=100, show_progress_bar=False)
        pairs = list(zip(df.iloc[te]["reference_answer"], df.iloc[te]["student_answer"]))
        out[te] = np.clip(model.predict(pairs) * 5, 0, 5)
        print(f"fold {i + 1}/5 done")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/raw/mohler.csv")
    p.add_argument("--epochs", type=int, default=3)
    a = p.parse_args()
    df = load(a.data)
    rows = [{"model": "SBERT cosine (MiniLM)", **report(df["score"], sbert_scores(df))},
            {"model": "Cross-encoder fine-tuned (RoBERTa)", **report(df["score"], cross_encoder_scores(df, a.epochs))}]
    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("reports/transformer_results.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
