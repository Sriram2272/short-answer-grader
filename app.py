"""Streamlit grading portal: one answer at a time, or a CSV of a whole class."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data import load
from src.features import missing_concepts
from src.models import best_grader


@st.cache_resource
def grader():
    return best_grader().fit(load("data/raw/mohler.csv"))


st.set_page_config(page_title="Short-Answer Grader", layout="wide")
st.title("Short-Answer Grading Assistant")
st.caption("Suggests a 0-5 score against the teacher's model answer. A teacher makes the final decision.")

one, batch = st.tabs(["Grade one answer", "Grade a class (CSV)"])

with one:
    q = st.text_input("Question", "What is a stack?")
    ref = st.text_area("Model answer", "A stack is a linear data structure where elements are added and removed "
                       "at the top, so the last element pushed is the first one popped (LIFO).")
    ans = st.text_area("Student answer", "It is a LIFO structure; you push and pop at the top.")
    if st.button("Grade", type="primary"):
        row = pd.DataFrame([{"question": q, "reference_answer": ref, "student_answer": ans}])
        score = float(grader().predict(row)[0])
        st.metric("Suggested score", f"{round(score * 2) / 2:.1f} / 5")
        missing = missing_concepts(q, ref, ans)
        st.write("**Concepts from the model answer not found:** " + (", ".join(missing) if missing else "none"))

with batch:
    st.write("Upload a CSV with columns `question`, `reference_answer`, `student_answer`.")
    f = st.file_uploader("CSV file", type="csv")
    if f is not None:
        df = pd.read_csv(f).fillna("")
        need = {"question", "reference_answer", "student_answer"}
        if not need <= set(df.columns):
            st.error(f"Missing columns: {', '.join(sorted(need - set(df.columns)))}")
        else:
            df["suggested_score"] = (grader().predict(df) * 2).round() / 2
            df["missing_concepts"] = [", ".join(missing_concepts(r.question, r.reference_answer, r.student_answer))
                                      for r in df.itertuples()]
            st.dataframe(df, use_container_width=True)
            st.download_button("Download graded CSV", df.to_csv(index=False), "graded.csv")
