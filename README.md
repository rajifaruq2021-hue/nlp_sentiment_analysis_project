# Sentiment Analysis on Movie Reviews (NLP Project)
Course: DLBAIPNLP01 - Project: Natural Language Processing
Degree Programme: BSc Computer Science / Artificial Intelligence
IU International University of Applied Sciences

Author: Faruq Ishola Raji
Matriculation Number: 4253139

---

## 1. Project Overview

This project implements an end-to-end Natural Language Processing (NLP) pipeline for binary sentiment analysis on movie reviews. The primary objective is to evaluate how classical machine learning algorithms perform when categorizing unstructured natural language text as either positive or negative, and to observe how performance scales across different data sample sizes.

The system tests three supervised learning algorithms:
- Multinomial Naive Bayes (Probabilistic baseline)
- Logistic Regression (Linear model with L2 regularization)
- Linear Support Vector Classifier (Maximum margin linear separator)

All models are systematically evaluated across three progressive dataset scenarios:
- **Scenario A (Small Scale):** 500 reviews
- **Scenario B (Medium Scale):** 1,000 reviews
- **Scenario C (Full Benchmark):** 2,000 reviews

---

## 2. Repository Structure

```
├── data/
│   └── movie_reviews_corpus.csv    # Benchmark movie reviews dataset (2,000 balanced reviews)
├── outputs/
│   ├── figures/                    # Generated charts and confusion matrices
│   │   ├── figure1_accuracy_progression.png
│   │   ├── figure2_full_benchmark_metrics.png
│   │   └── figure3_confusion_matrix_linear_svc.png
│   ├── progressive_results.csv     # Tabulated performance metrics across all scenarios
│   └── confusion_matrices.json     # Raw classification matrices
├── src/
│   ├── preprocess.py               # Text cleaning, stopword filtering, and lemmatization
│   ├── train_evaluate.py           # Training, evaluation, and chart generation script
│   └── main.py                     # Interactive CLI inference application
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 3. Installation and Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed
- Virtual environment recommended

### Installation Steps

1. Clone or download the repository to your local machine:
```bash
git clone https://github.com/rajifaruq2021-hue/nlp_sentiment_analysis_project.git
cd nlp_sentiment_analysis_project
```

2. Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. How to Run the Project

### A. Run Progressive Experiments
To train all three models across Small, Medium, and Full data scenarios, run:
```bash
python src/train_evaluate.py
```
This script will:
- Clean and tokenize the dataset.
- Train models across all 3 scenarios.
- Output detailed accuracy, precision, recall, and F1-scores.
- Save summary figures into `outputs/figures/`.

### B. Run the Interactive Inference Tool
To test the trained model on sample movie reviews or custom user input:
```bash
python src/main.py
```

---

## 5. Summary of Experimental Results

| Scenario | Sample Size | Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| **Scenario A** | 500 reviews | Multinomial Naive Bayes | 80.00% | 0.7778 | 0.8400 | 0.8077 |
| **Scenario A** | 500 reviews | Logistic Regression | 80.00% | 0.8000 | 0.8000 | 0.8000 |
| **Scenario A** | 500 reviews | Linear Support Vector Classifier | **81.00%** | 0.8039 | 0.8200 | 0.8119 |
| **Scenario B** | 1,000 reviews | Multinomial Naive Bayes | 74.00% | 0.7609 | 0.7000 | 0.7292 |
| **Scenario B** | 1,000 reviews | Logistic Regression | 78.00% | 0.7642 | 0.8100 | 0.7864 |
| **Scenario B** | 1,000 reviews | Linear Support Vector Classifier | **80.50%** | 0.7798 | 0.8500 | 0.8134 |
| **Scenario C** | 2,000 reviews | Multinomial Naive Bayes | 81.25% | 0.8205 | 0.8000 | 0.8101 |
| **Scenario C** | 2,000 reviews | Logistic Regression | 83.75% | 0.8261 | 0.8550 | 0.8403 |
| **Scenario C** | 2,000 reviews | Linear Support Vector Classifier | **85.25%** | 0.8373 | 0.8750 | **0.8557** |

### Key Findings
- Linear Support Vector Classifier (Linear SVC) achieved the highest overall performance across all data scenarios, reaching an optimal accuracy of **85.25%** and an F1-score of **0.8557** on the full benchmark.
- Data scaling visibly enhanced model generalizability and stability, reducing variance and improving boundary separation in the high-dimensional TF-IDF feature space.

