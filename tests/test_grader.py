import numpy as np
import pandas as pd

from src.data import folds_by_question, load, split_by_question
from src.features import FeatureBuilder, concepts, missing_concepts
from src.metrics import report, to_half_points
from src.models import best_grader

DATA = "data/raw/mohler.csv"


def toy():
    rows = []
    for q, ref in [("What is a stack?", "A stack is last in first out; push and pop at the top."),
                   ("What is a queue?", "A queue is first in first out; enqueue at the rear, dequeue at the front."),
                   ("What is a constructor?", "A constructor initializes a new object when it is created.")]:
        for ans, s in [(ref, 5), ("It is a data structure.", 2), ("I do not know.", 0)]:
            rows.append({"question_id": q, "question": q, "reference_answer": ref, "student_answer": ans, "score": s})
    return pd.DataFrame(rows)


def test_folds_never_share_a_question():
    df = load(DATA)
    for tr, te in folds_by_question(df):
        assert not set(df.iloc[tr].question_id) & set(df.iloc[te].question_id)
    tr, te = split_by_question(df)
    assert not set(tr.question_id) & set(te.question_id)


def test_mohler_loads_with_our_schema():
    df = load(DATA)
    assert len(df) == 2273 and df.question_id.nunique() == 81
    assert df.score.between(0, 5).all()


def test_the_reference_scores_above_a_wrong_answer():
    df = toy()
    g = best_grader().fit(pd.concat([df] * 10, ignore_index=True))
    good, bad = g.predict(df.iloc[[0, 2]])
    assert good > bad


def test_features_are_one_row_per_answer_and_finite():
    df = toy()
    X = FeatureBuilder().fit_transform(df)
    assert X.shape[0] == len(df) and np.isfinite(X).all()


def test_question_words_are_not_counted_as_missing():
    assert "stack" not in missing_concepts("What is a stack?", "A stack is LIFO", "It is LIFO")
    assert "lifo" in missing_concepts("What is a stack?", "A stack is LIFO", "a pile")


def test_concepts_stem_and_keep_negation():
    assert concepts("Nodes are not linked") == {"node", "not", "link"}


def test_metrics_on_perfect_and_rounding():
    assert report([0, 2.5, 5], [0, 2.5, 5]) == {"pearson": 1.0, "rmse": 0.0, "mae": 0.0, "qwk": 1.0}
    assert list(to_half_points([-1, 2.26, 7])) == [0, 2.5, 5]
