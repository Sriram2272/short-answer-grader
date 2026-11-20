"""Hand-built similarity features between a reference answer and a student answer."""
from __future__ import annotations

import re

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

_TOKEN = re.compile(r"[a-z0-9]+(?:\([a-z0-9 ]*\))?")
_NEG = {"not", "no", "never", "cannot", "cant", "dont", "doesnt", "isnt", "without"}
_stem = PorterStemmer().stem
STOP = ENGLISH_STOP_WORDS - _NEG

FEATURE_NAMES = [
    "word_tfidf_cos", "char_tfidf_cos", "lsa_cos",
    "ref_coverage", "ref_coverage_minus_question", "answer_precision",
    "length_ratio", "log_answer_words", "negation_mismatch",
]


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def concepts(text: str) -> set[str]:
    """Stemmed content words of a text, stop words removed (negations kept)."""
    return {_stem(t) for t in tokens(text) if t not in STOP and len(t) > 1}


def _cos(a, b) -> np.ndarray:
    num = np.asarray(a.multiply(b).sum(axis=1)).ravel() if hasattr(a, "multiply") else (a * b).sum(1)
    if hasattr(a, "multiply"):
        return num  # TF-IDF rows are L2-normalised already
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


class FeatureBuilder:
    """Fits its vocabularies on training text only; transform is per row and deterministic."""

    def __init__(self, lsa_dims: int = 100, seed: int = 0) -> None:
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, preprocessor=self._norm)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.lsa_dims, self.seed = lsa_dims, seed

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(_stem(t) for t in tokens(text))

    def fit(self, df) -> "FeatureBuilder":
        corpus = list(df["reference_answer"]) + list(df["student_answer"]) + list(df["question"])
        self.word.fit(corpus)
        self.char.fit(corpus)
        dims = max(2, min(self.lsa_dims, len(self.word.vocabulary_) - 1))
        self.svd = TruncatedSVD(n_components=dims, random_state=self.seed).fit(self.word.transform(corpus))
        return self

    def transform(self, questions, refs, answers) -> np.ndarray:
        questions, refs, answers = list(questions), list(refs), list(answers)
        wr, wa = self.word.transform(refs), self.word.transform(answers)
        cr, ca = self.char.transform(refs), self.char.transform(answers)
        word_cos, char_cos = _cos(wr, wa), _cos(cr, ca)
        lsa_cos = _cos(self.svd.transform(wr), self.svd.transform(wa))
        rows = []
        for i, (q, ref, ans) in enumerate(zip(questions, refs, answers)):
            rc, ac, qc = concepts(ref), concepts(ans), concepts(q)
            rc_new = rc - qc  # credit only for concepts the question did not already give away
            n_ans, n_ref = len(tokens(ans)), max(len(tokens(ref)), 1)
            rows.append([
                word_cos[i], char_cos[i], lsa_cos[i],
                len(rc & ac) / len(rc) if rc else 0.0,
                len(rc_new & ac) / len(rc_new) if rc_new else 0.0,
                len(rc & ac) / len(ac) if ac else 0.0,
                min(n_ans / n_ref, 3.0),
                np.log1p(n_ans),
                float(bool(_NEG & set(tokens(ref))) != bool(_NEG & set(tokens(ans)))),
            ])
        return np.array(rows, dtype=float)

    def fit_transform(self, df) -> np.ndarray:
        return self.fit(df).transform(df["question"], df["reference_answer"], df["student_answer"])


def missing_concepts(question: str, ref: str, answer: str) -> list[str]:
    """Reference words (as written) whose stem the answer lacks, ignoring words from the question."""
    have = concepts(answer) | concepts(question)
    seen, out = set(), []
    for t in tokens(ref):
        s = _stem(t)
        if t not in STOP and len(t) > 1 and s not in have and s not in seen:
            seen.add(s)
            out.append(t)
    return out
