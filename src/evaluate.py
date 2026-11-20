"""5-fold cross-validation grouped by question, a comparison table and an error analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .data import folds_by_question, load
from .features import FEATURE_NAMES, missing_concepts
from .metrics import report
from .models import all_graders, best_grader


def cross_validate(df: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Out-of-fold predictions for every grader; each question is predicted by a model that never saw it."""
    preds = {g.name: np.zeros(len(df)) for g in all_graders()}
    for tr, te in folds_by_question(df, k):
        for g in all_graders():
            preds[g.name][te] = g.fit(df.iloc[tr]).predict(df.iloc[te])
    rows = [{"model": name, **report(df["score"], p)} for name, p in preds.items()]
    return pd.DataFrame(rows), preds


def human_agreement(df: pd.DataFrame) -> dict | None:
    """How well the two human graders agree with each other: the ceiling to compare against."""
    if {"score_me", "score_other"} <= set(df.columns):
        return {"model": "Grader A vs grader B (humans)", **report(df["score_me"], df["score_other"])}
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/raw/mohler.csv")
    p.add_argument("--out", default="reports")
    a = p.parse_args()
    df = load(a.data)
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    table, preds = cross_validate(df)
    human = human_agreement(df)
    if human:
        table = pd.concat([table, pd.DataFrame([human])], ignore_index=True)

    best = best_grader()
    df = df.assign(predicted=np.round(preds[best.name], 2))
    df["error"] = (df["predicted"] - df["score"]).round(2)

    # Permutation importance on a held-out fold, so it measures generalisation, not memory.
    tr, te = folds_by_question(df)[0]
    best.fit(df.iloc[tr])
    imp = permutation_importance(best.regressor, best.features(df.iloc[te]), df.iloc[te]["score"],
                                 n_repeats=10, random_state=0, scoring="neg_root_mean_squared_error")
    importance = sorted(zip(FEATURE_NAMES, imp.importances_mean), key=lambda x: -x[1])

    worst = df.reindex(df["error"].abs().sort_values(ascending=False).index).head(8)
    per_q = (df.groupby("question_id")
             .apply(lambda g: report(g["score"], g["predicted"])["rmse"], include_groups=False)
             .sort_values(ascending=False))

    table.to_csv(out / "results.csv", index=False)
    (out / "results.json").write_text(json.dumps({
        "dataset": a.data, "answers": len(df), "questions": int(df["question_id"].nunique()),
        "results": table.to_dict("records"), "feature_importance": dict(importance),
    }, indent=2))
    lines = [
        "# Results", "",
        f"Mohler et al. (2011): {len(df)} answers to {df['question_id'].nunique()} questions, "
        "scores 0-5 (mean of two graders). 5-fold cross-validation **grouped by question**: "
        "every score below is for questions the model never saw in training.", "",
        table.to_markdown(index=False), "",
        "## What the best model relies on (permutation importance, RMSE increase)", "",
        *[f"- `{n}`: {v:.3f}" for n, v in importance], "",
        "## Hardest questions (highest RMSE)", "",
        *[f"- Q{q}: RMSE {v:.2f}, \"{df[df.question_id == q].question.iloc[0]}\"" for q, v in per_q.head(5).items()], "",
        "## Largest errors", "",
    ]
    for _, r in worst.iterrows():
        lines += [f"**Q{r.question_id}** human {r.score}, model {r.predicted} ({r.error:+.2f})", "",
                  f"- Question: {r.question}", f"- Reference: {r.reference_answer}",
                  f"- Student: {r.student_answer}",
                  f"- Missing concepts: {', '.join(missing_concepts(r.question, r.reference_answer, r.student_answer)[:8]) or 'none'}", ""]
    (out / "results.md").write_text("\n".join(lines))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
