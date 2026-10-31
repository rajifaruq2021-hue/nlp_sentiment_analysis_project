"""
Model Training and Progressive Evaluation Module
Course: DLBAIPNLP01 - Project: NLP
Author: Faruq Ishola Raji (4253139)

Trains and evaluates supervised machine learning models (Multinomial Naive Bayes,
Logistic Regression, and Linear Support Vector Classifier) across progressive dataset scales
(Small = 500 reviews, Medium = 1,000 reviews, Full = 2,000 reviews).
"""

import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocess import TextPreprocessor


def run_progressive_experiments(data_path: str, output_dir: str):
    """
    Executes progressive scaling experiments as specified in IU Project Report guidelines.
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'figures')
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading dataset from:", data_path)
    df = pd.read_csv(data_path)
    
    # Preprocess text
    print("Preprocessing text data...")
    preprocessor = TextPreprocessor()
    df['cleaned_review'] = preprocessor.preprocess_corpus(df['review'].tolist())
    df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})

    # Define experimental scenarios
    scenarios = [
        {'name': 'Scenario A: Small Scale', 'sample_size': 500, 'random_state': 42},
        {'name': 'Scenario B: Medium Scale', 'sample_size': 1000, 'random_state': 42},
        {'name': 'Scenario C: Full Benchmark', 'sample_size': len(df), 'random_state': 42}
    ]

    models_to_test = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear Support Vector Classifier': LinearSVC(random_state=42, dual=False)
    }

    all_results = []
    confusion_matrices = {}

    for scenario in scenarios:
        s_name = scenario['name']
        size = scenario['sample_size']
        print(f"\n--- Running {s_name} (Sample Size = {size}) ---")

        # Balanced subsample
        if size < len(df):
            pos_samples = df[df['label'] == 1].sample(n=size // 2, random_state=scenario['random_state'])
            neg_samples = df[df['label'] == 0].sample(n=size // 2, random_state=scenario['random_state'])
            sub_df = pd.concat([pos_samples, neg_samples]).sample(frac=1.0, random_state=scenario['random_state']).reset_index(drop=True)
        else:
            sub_df = df.copy()

        # Split 80/20 train/test
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            sub_df['cleaned_review'],
            sub_df['label'],
            test_size=0.20,
            random_state=scenario['random_state'],
            stratify=sub_df['label']
        )

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)

        for model_name, model in models_to_test.items():
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            record = {
                'Scenario': s_name,
                'Sample_Size': size,
                'Train_Size': X_train.shape[0],
                'Test_Size': X_test.shape[0],
                'Model': model_name,
                'Accuracy': round(acc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1_Score': round(f1, 4),
                'Training_Time_Sec': round(train_time, 4)
            }
            all_results.append(record)
            confusion_matrices[f"{s_name}_{model_name}"] = cm.tolist()

            print(f"[{model_name}] Accuracy: {acc:.4f} | F1: {f1:.4f} | Time: {train_time:.4f}s")

    # Save results to CSV & JSON
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(output_dir, 'progressive_results.csv')
    results_df.to_csv(results_csv, index=False)
    print("\nSaved experiment results to:", results_csv)

    cm_json = os.path.join(output_dir, 'confusion_matrices.json')
    with open(cm_json, 'w') as f:
        json.dump(confusion_matrices, f, indent=4)

    # Generate Figures for Project Report
    generate_figures(results_df, confusion_matrices, plots_dir)

    return results_df


def generate_figures(results_df: pd.DataFrame, confusion_matrices: dict, plots_dir: str):
    """
    Creates charts and visual figures for inclusion in the project report.
    """
    sns.set_theme(style='whitegrid')

    # Figure 1: Accuracy Progression across Scenarios
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=results_df,
        x='Scenario',
        y='Accuracy',
        hue='Model',
        palette='Blues_d'
    )
    plt.title('Figure 1: Model Accuracy across Progressive Data Scenarios', fontsize=12, fontweight='bold', pad=12)
    plt.ylabel('Test Accuracy', fontsize=10)
    plt.xlabel('Data Scale Scenario', fontsize=10)
    plt.ylim(0.60, 0.95)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2%}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 3),
                        textcoords='offset points')
    plt.legend(title='Model', loc='lower right', frameon=True)
    plt.tight_layout()
    fig1_path = os.path.join(plots_dir, 'figure1_accuracy_progression.png')
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print("Saved Figure 1:", fig1_path)

    # Figure 2: F1-Score Comparison in Full Benchmark
    full_df = results_df[results_df['Scenario'].str.contains('Full Benchmark')].copy()
    plt.figure(figsize=(8, 4.5))
    metrics_melted = pd.melt(
        full_df,
        id_vars=['Model'],
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1_Score'],
        var_name='Metric',
        value_name='Score'
    )
    ax2 = sns.barplot(
        data=metrics_melted,
        x='Metric',
        y='Score',
        hue='Model',
        palette='crest'
    )
    plt.title('Figure 2: Performance Metrics on Full Benchmark Dataset (N = 2,000)', fontsize=12, fontweight='bold', pad=12)
    plt.ylabel('Score (0 - 1.0)', fontsize=10)
    plt.ylim(0.65, 0.95)
    plt.legend(title='Model', loc='lower right', frameon=True)
    plt.tight_layout()
    fig2_path = os.path.join(plots_dir, 'figure2_full_benchmark_metrics.png')
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print("Saved Figure 2:", fig2_path)

    # Figure 3: Confusion Matrix for Best Model (Linear SVC on Full Benchmark)
    best_cm_key = 'Scenario C: Full Benchmark_Linear Support Vector Classifier'
    if best_cm_key in confusion_matrices:
        cm_array = np.array(confusion_matrices[best_cm_key])
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_array,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.title('Figure 3: Confusion Matrix (Linear SVC on Full Benchmark)', fontsize=12, fontweight='bold', pad=12)
        plt.ylabel('True Class', fontsize=10)
        plt.xlabel('Predicted Class', fontsize=10)
        plt.tight_layout()
        fig3_path = os.path.join(plots_dir, 'figure3_confusion_matrix_linear_svc.png')
        plt.savefig(fig3_path, dpi=300)
        plt.close()
        print("Saved Figure 3:", fig3_path)


if __name__ == '__main__':
    data_file = r'D:\IU INTERNATIONAL UNIVERSITY\2 Semester\Project- NLP (DLBAIPNLP)\Project Report - Sentiment Analysis\data\movie_reviews_corpus.csv'
    out_dir = r'D:\IU INTERNATIONAL UNIVERSITY\2 Semester\Project- NLP (DLBAIPNLP)\Project Report - Sentiment Analysis\outputs'
    run_progressive_experiments(data_file, out_dir)
