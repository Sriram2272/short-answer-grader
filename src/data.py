"""Loading short-answer grading data and splitting it by question."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

REQUIRED = ["question_id", "question", "reference_answer", "student_answer", "score"]

# Mohler et al. (2011) column names -> ours. score_avg is the mean of the two graders.
MOHLER = {"id": "question_id", "desired_answer": "reference_answer", "score_avg": "score"}


def load(path: str) -> pd.DataFrame:
    """Read a CSV in our schema or Mohler's; refuse one that lacks a required column."""
    df = pd.read_csv(path).rename(columns=MOHLER)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["question_id"] = df["question_id"].astype(str)
    for c in ("question", "reference_answer", "student_answer"):
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["score"] = df["score"].astype(float)
    return df[REQUIRED + [c for c in ("score_me", "score_other") if c in df.columns]].reset_index(drop=True)


def split_by_question(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """One train/test split in which no question appears on both sides.

    Grading unseen questions is the real use case; a random row split lets the
    model learn each question's answers and reports inflated scores.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df["question_id"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def folds_by_question(df: pd.DataFrame, k: int = 5):
    """k (train_idx, test_idx) pairs, each question in exactly one test fold."""
    return list(GroupKFold(n_splits=k).split(df, groups=df["question_id"]))
