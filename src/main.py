"""
Interactive Sentiment Analysis CLI Inference Tool
Course: DLBAIPNLP01 - Project: NLP
Author: Faruq Ishola Raji (4253139)

Allows a user or evaluator to enter a custom movie review and receive
an instant binary classification (Positive or Negative) along with confidence scoring.
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from preprocess import TextPreprocessor


class SentimentClassifierApp:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = CalibratedClassifierCV(LinearSVC(random_state=42, dual=False))
        self._is_trained = False

    def train(self):
        print("Training production model on complete movie reviews corpus...")
        df = pd.read_csv(self.data_path)
        cleaned = self.preprocessor.preprocess_corpus(df['review'].tolist())
        labels = df['sentiment'].map({'negative': 0, 'positive': 1}).values

        X = self.vectorizer.fit_transform(cleaned)
        self.model.fit(X, labels)
        self._is_trained = True
        print("Training complete! Model is ready for inference.")

    def predict(self, raw_review: str) -> dict:
        if not self._is_trained:
            raise RuntimeError("Model must be trained prior to inference.")

        cleaned = self.preprocessor.clean_text(raw_review)
        if not cleaned.strip():
            return {
                "sentiment": "Neutral/Unknown",
                "confidence": 0.50,
                "cleaned_text": ""
            }

        vec = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(vec)[0]
        pos_prob = probs[1]
        neg_prob = probs[0]

        sentiment = "Positive" if pos_prob >= 0.5 else "Negative"
        confidence = pos_prob if sentiment == "Positive" else neg_prob

        return {
            "sentiment": sentiment,
            "confidence": round(float(confidence), 4),
            "positive_probability": round(float(pos_prob), 4),
            "negative_probability": round(float(neg_prob), 4),
            "cleaned_text": cleaned
        }


def main():
    data_file = r'D:\IU INTERNATIONAL UNIVERSITY\2 Semester\Project- NLP (DLBAIPNLP)\Project Report - Sentiment Analysis\data\movie_reviews_corpus.csv'
    app = SentimentClassifierApp(data_file)
    app.train()

    test_samples = [
        "This film was an absolute masterpiece with breathtaking cinematography and incredible acting!",
        "The plot was painfully slow, disjointed, and the characters were utterly unconvincing. A total waste of time.",
        "A charming romantic comedy that kept our entire family smiling from start to finish.",
        "Terrible script, poor sound design, and flat performances across the board."
    ]

    print("\n================ TEST INFERENCE DEMONSTRATION ================")
    for sample in test_samples:
        res = app.predict(sample)
        print(f"\nRaw Review: \"{sample}\"")
        print(f"Cleaned Text: \"{res['cleaned_text']}\"")
        print(f"Outcome: {res['sentiment']} (Confidence: {res['confidence']:.2%})")
    print("==============================================================\n")


if __name__ == '__main__':
    main()
