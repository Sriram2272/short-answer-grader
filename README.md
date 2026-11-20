# Short-Answer Grading Assistant

**CSB306 Natural Language Processing, project NLP-21 · Dama Sri Ram**

A system that suggests a 0–5 score for a student's short written answer by comparing it with the
teacher's model answer, and lists which concepts of the model answer the student left out. It is an
assistant: a teacher makes the final decision.

## Results

Dataset: Mohler et al. (2011), 2,273 student answers to 81 Data Structures questions, each scored
0–5 by two human graders (we predict their average). Evaluation is **5-fold cross-validation grouped
by question**, so every number is for questions the model never saw in training. A random row split
would let the model learn each question's answers and report much better, meaningless numbers.

| Model | Pearson ↑ | RMSE ↓ | MAE ↓ | QWK ↑ |
|---|---:|---:|---:|---:|
| Predict the mean (floor) | −0.124 | 1.112 | 0.905 | 0.000 |
| TF-IDF cosine only (classical baseline) | 0.453 | 0.996 | 0.750 | 0.247 |
| All features + Ridge | 0.544 | 0.930 | 0.692 | 0.431 |
| **All features + Gradient Boosting** | **0.554** | **0.925** | **0.658** | **0.478** |
| Human grader A vs human grader B | 0.597 | 1.311 | 0.748 | 0.491 |

The best model's correlation with the graders' average comes within 0.04 of how well the two humans
agree with each other, and its RMSE is lower than one human's against the other. QWK is quadratic
weighted kappa on half-point bands.

Transformer models (SBERT similarity and a fine-tuned RoBERTa cross-encoder) are in
`src/transformer.py`; they need Hugging Face downloads and a GPU, so run them on Google Colab
(below) and add their row to the table.

Full report with error analysis: [`reports/results.md`](reports/results.md).

## How it works

For each (question, model answer, student answer) the model computes nine features
(`src/features.py`):

| Feature | What it measures |
|---|---|
| `word_tfidf_cos` | Word and bigram TF-IDF cosine (stemmed) |
| `char_tfidf_cos` | Character 3–5-gram TF-IDF cosine (robust to spelling) |
| `lsa_cos` | Cosine in a 100-dimensional LSA space (synonyms and related words) |
| `ref_coverage` | Share of the model answer's concepts the student used |
| `ref_coverage_minus_question` | The same, not counting words the question already gives away |
| `answer_precision` | Share of the student's concepts that are in the model answer |
| `length_ratio`, `log_answer_words` | Answer length |
| `negation_mismatch` | One answer negates and the other does not |

A gradient-boosted regressor maps the features to a score. Permutation importance on a held-out
fold shows `ref_coverage_minus_question` matters most: students often repeat the question, and
crediting that inflates scores.

## Error analysis (from `reports/results.md`)

1. **Very short model answers.** Q2.5 "How many constructors can be created for a class?", model
   answer "Unlimited number". The student wrote "one", a human gave 0, the model gave 3.5. With two
   reference words there is almost nothing to compare, and every short answer looks alike.
2. **Correct answers in other words.** Paraphrases without shared vocabulary ("sets up the object's
   values" for "initializes the object") score low. LSA helps a little; this is where sentence
   embeddings should win.
3. **Keyword stuffing.** An answer that lists the right terms with the wrong meaning gets coverage
   credit. Only a model that reads the relationship between words (a cross-encoder) can catch it.
4. **Negation.** "A queue is not LIFO" and "A queue is LIFO" share all their words; the
   `negation_mismatch` feature catches the simplest cases only.
5. **The graders disagree too.** Grader-to-grader QWK is 0.491, so a perfect model cannot score far
   above that against either one.

## Run it

```bash
pip install -r requirements.txt
bash scripts/get_data.sh          # downloads the Mohler dataset to data/raw/
python -m src.evaluate            # cross-validation table + reports/results.md (about a minute)
streamlit run app.py              # the grading portal
pytest -q                         # tests
```

The app grades one answer at a time, or a whole class from a CSV with columns `question`,
`reference_answer`, `student_answer`, and lets you download the graded file.

### Transformer models on Google Colab

```bash
!git clone https://github.com/Sriram2272/short-answer-grader && cd short-answer-grader
!pip install -r requirements.txt && bash scripts/get_data.sh
!python -m src.transformer        # writes reports/transformer_results.csv
```

## Layout

```
src/data.py         loading, question-grouped splits
src/features.py     the nine similarity features, missing-concept list
src/models.py       mean floor, TF-IDF baseline, Ridge, Gradient Boosting
src/metrics.py      Pearson, RMSE, MAE, QWK
src/evaluate.py     cross-validation, importance, error analysis -> reports/
src/transformer.py  SBERT and fine-tuned cross-encoder (Colab)
app.py              Streamlit portal
tests/              unit tests
```

## Future scope

- Fine-tuned cross-encoder results on Colab, and a combination of it with the feature model.
- Several model answers per question, and the best match taken.
- Per-question calibration with a few teacher-graded answers.
- SciEntsBank / Beetle datasets to check the model beyond one course.

## Reference

Mohler, M., Bunescu, R., & Mihalcea, R. (2011). Learning to Grade Short Answer Questions using
Semantic Similarity Measures and Dependency Graph Alignments. *ACL 2011*.
