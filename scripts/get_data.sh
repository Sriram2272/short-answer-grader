#!/usr/bin/env bash
# Downloads the Mohler et al. (2011) short-answer grading dataset (Texas, CS Data Structures course):
# 81 questions, 2,273 student answers, each scored 0-5 by two human graders.
set -euo pipefail
mkdir -p data/raw
curl -fsSL -o data/raw/mohler.csv \
  https://raw.githubusercontent.com/gsasikiran/Comparative-Evaluation-of-Pretrained-Transfer-Learning-Models-on-ASAG/master/comparative_evaluation_on_mohler_dataset/dataset/mohler_dataset_edited.csv
echo "saved data/raw/mohler.csv ($(wc -l < data/raw/mohler.csv) lines)"
