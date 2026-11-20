# Results

Mohler et al. (2011): 2273 answers to 81 questions, scores 0-5 (mean of two graders). 5-fold cross-validation **grouped by question**: every score below is for questions the model never saw in training.

| model                            |   pearson |   rmse |   mae |   qwk |
|:---------------------------------|----------:|-------:|------:|------:|
| mean (floor)                     |    -0.124 |  1.112 | 0.905 | 0     |
| TF-IDF cosine only               |     0.453 |  0.996 | 0.75  | 0.247 |
| All features + Ridge             |     0.544 |  0.93  | 0.692 | 0.431 |
| All features + Gradient Boosting |     0.554 |  0.925 | 0.658 | 0.478 |
| Grader A vs grader B (humans)    |     0.597 |  1.311 | 0.748 | 0.491 |

## What the best model relies on (permutation importance, RMSE increase)

- `ref_coverage_minus_question`: 0.102
- `word_tfidf_cos`: 0.049
- `lsa_cos`: 0.024
- `char_tfidf_cos`: 0.024
- `ref_coverage`: 0.016
- `negation_mismatch`: 0.010
- `answer_precision`: 0.006
- `length_ratio`: 0.001
- `log_answer_words`: -0.008

## Hardest questions (highest RMSE)

- Q2.5: RMSE 2.26, "How many constructors can be created for a class?"
- Q11.3: RMSE 1.74, "How are objects initialized when they are created?"
- Q1.2: RMSE 1.45, "What stages in the software life cycle are influenced by the testing stage?"
- Q12.5: RMSE 1.43, "What is the advantage of linked lists over arrays?"
- Q10.7: RMSE 1.40, "How many comparisons does it take to find an element in a binary search tree?"

## Largest errors

**Q2.5** human 0.0, model 3.53 (+3.53)

- Question: How many constructors can be created for a class?
- Reference: Unlimited number.
- Student: one
- Missing concepts: unlimited, number

**Q2.5** human 0.0, model 3.53 (+3.53)

- Question: How many constructors can be created for a class?
- Reference: Unlimited number.
- Student: One
- Missing concepts: unlimited, number

**Q2.5** human 0.0, model 3.53 (+3.53)

- Question: How many constructors can be created for a class?
- Reference: Unlimited number.
- Student: One
- Missing concepts: unlimited, number

**Q2.5** human 0.0, model 3.53 (+3.53)

- Question: How many constructors can be created for a class?
- Reference: Unlimited number.
- Student: one
- Missing concepts: unlimited, number

**Q2.5** human 0.0, model 3.53 (+3.53)

- Question: How many constructors can be created for a class?
- Reference: Unlimited number.
- Student: one
- Missing concepts: unlimited, number

**Q2.5** human 0.0, model 3.48 (+3.48)

- Question: How many constructors can be created for a class?
- Reference: Unlimited number.
- Student: 1
- Missing concepts: unlimited, number

**Q4.3** human 5.0, model 1.53 (-3.47)

- Question: How are arrays passed to functions?
- Reference: by reference.
- Student: #NAME?
- Missing concepts: reference

**Q10.6** human 1.5, model 4.92 (+3.42)

- Question: What is the inorder traversal of a binary tree?
- Reference: Traverse the left subtree, then the root, then the right subtree.
- Student: in order starts with the root then does right child then left child recursively
- Missing concepts: subtree
