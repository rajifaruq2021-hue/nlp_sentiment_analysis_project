"""
train_evaluate.py

I am the model training and progressive evaluation module for my NLP Sentiment Analysis project.
Course: DLBAIPNLP01 - Project: NLP
Author: Faruq Ishola Raji
Matriculation Number: 4253139

What I do in this file:
1. I load the movie reviews dataset.
2. I clean all reviews using my custom TextPreprocessor class.
3. I test three machine learning models:
   - Multinomial Naive Bayes (MNB)
   - Logistic Regression (LR)
   - Linear Support Vector Classifier (LinearSVC)
4. I evaluate performance across three progressive dataset sizes:
   - Scenario A: Small scale (500 reviews)
   - Scenario B: Medium scale (1,000 reviews)
   - Scenario C: Full benchmark (2,000 reviews)
5. I calculate Accuracy, Precision, Recall, F1-Score, and Training Time for every experiment.
6. I save the numerical metrics to CSV/JSON and generate three high-resolution figures for my report:
   - Figure 1: Accuracy progression across data sizes
   - Figure 2: Detailed performance metrics on the full benchmark
   - Figure 3: Confusion matrix for my top-performing Linear SVC model
"""

# I import standard Python tools for folders, JSON, timing, and dataframes
import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# I import scikit-learn modules for data splitting, TF-IDF feature extraction, and models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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

# I import my custom text preprocessor class from preprocess.py
from preprocess import TextPreprocessor


def run_progressive_experiments(data_path: str, output_dir: str):
    """
    I run the full series of experiments across progressive data scales as required by the course brief.
    """
    # I create the output and figures folders if they do not exist
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'figures')
    os.makedirs(plots_dir, exist_ok=True)

    print("I am loading the movie reviews dataset from:", data_path)
    df = pd.read_csv(data_path)

    # I clean all review texts using my preprocessor
    print("I am cleaning and preprocessing all review texts.")
    preprocessor = TextPreprocessor()
    df['cleaned_review'] = preprocessor.preprocess_corpus(df['review'].tolist())

    # I encode sentiment labels as integers: negative -> 0, positive -> 1
    df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})

    # I define the three progressive data scale scenarios required by the project brief
    scenarios = [
        {'name': 'Scenario A: Small Scale', 'sample_size': 500, 'random_state': 42},
        {'name': 'Scenario B: Medium Scale', 'sample_size': 1000, 'random_state': 42},
        {'name': 'Scenario C: Full Benchmark', 'sample_size': len(df), 'random_state': 42}
    ]

    # I set up the three classifiers that I am benchmarking
    models_to_test = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear Support Vector Classifier': LinearSVC(random_state=42, dual=False)
    }

    all_results = []
    confusion_matrices = {}

    # I loop through each data size scenario
    for scenario in scenarios:
        s_name = scenario['name']
        size = scenario['sample_size']
        print(f"\n--- I am running {s_name} (Sample Size = {size}) ---")

        # I create a balanced subset of positive and negative reviews
        if size < len(df):
            pos_samples = df[df['label'] == 1].sample(n=size // 2, random_state=scenario['random_state'])
            neg_samples = df[df['label'] == 0].sample(n=size // 2, random_state=scenario['random_state'])
            sub_df = pd.concat([pos_samples, neg_samples]).sample(frac=1.0, random_state=scenario['random_state']).reset_index(drop=True)
        else:
            sub_df = df.copy()

        # I split the data into 80% training and 20% testing with balanced classes
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            sub_df['cleaned_review'],
            sub_df['label'],
            test_size=0.20,
            random_state=scenario['random_state'],
            stratify=sub_df['label']
        )

        # I convert the text reviews into TF-IDF feature matrices (using unigrams and bigrams up to 5,000 features)
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)

        # I train and evaluate each model on this data scale
        for model_name, model in models_to_test.items():
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # I make predictions on the unseen test set
            y_pred = model.predict(X_test)

            # I compute quantitative evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            # I record the metrics in my results list
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

    # I export the complete experimental results to a CSV file
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(output_dir, 'progressive_results.csv')
    results_df.to_csv(results_csv, index=False)
    print("\nI saved the progressive experimental results to:", results_csv)

    # I save all confusion matrices into a JSON file
    cm_json = os.path.join(output_dir, 'confusion_matrices.json')
    with open(cm_json, 'w') as f:
        json.dump(confusion_matrices, f, indent=4)

    # I generate and save all required figures for my report
    generate_figures(results_df, confusion_matrices, plots_dir)

    return results_df


def generate_figures(results_df: pd.DataFrame, confusion_matrices: dict, plots_dir: str):
    """
    I generate and save the three visual figures for my report.
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

    # I annotate each bar with its exact percentage
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
    print("I saved Figure 1 (Accuracy Progression):", fig1_path)

    # Figure 2: Detailed Performance Metrics on the Full Benchmark (N = 2,000)
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
    print("I saved Figure 2 (Full Benchmark Metrics):", fig2_path)

    # Figure 3: Confusion Matrix for my best-performing model (Linear SVC on Full Benchmark)
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
        print("I saved Figure 3 (Confusion Matrix):", fig3_path)


# I run the experiments when executing this file directly
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, 'data', 'movie_reviews_corpus.csv')
    out_dir = os.path.join(base_dir, 'outputs')
    run_progressive_experiments(data_file, out_dir)
