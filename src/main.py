"""
main.py

I am the interactive sentiment analysis CLI inference tool for my project.
Course: DLBAIPNLP01 - Project: NLP
Author: Faruq Ishola Raji
Matriculation Number: 4253139

What I do in this script:
1. I load the movie reviews dataset.
2. I train my best-performing Linear Support Vector Classifier (LinearSVC) using TF-IDF features.
3. I wrap the classifier with probability calibration so I can report confidence scores.
4. I test real movie review sentences and classify them as Positive or Negative with confidence percentages.
"""

# I import standard libraries for file paths, tables, and machine learning
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# I import my custom text preprocessor class from preprocess.py
from preprocess import TextPreprocessor


class SentimentClassifierApp:
    """
    I manage the end-to-end sentiment classification workflow:
    from raw text input, through preprocessing and vectorization, to the final sentiment prediction.
    """

    def __init__(self, data_path: str):
        # I store the path to the training CSV dataset
        self.data_path = data_path

        # I create an instance of my custom TextPreprocessor
        self.preprocessor = TextPreprocessor()

        # I set up a TF-IDF vectorizer that extracts unigrams and bigrams (up to 5,000 top features)
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

        # I use a Linear Support Vector Classifier wrapped with probability calibration
        # This allows me to compute confidence percentages for each prediction
        self.model = CalibratedClassifierCV(LinearSVC(random_state=42, dual=False))

        # I track whether the model has finished training
        self._is_trained = False

    def train(self):
        """
        I train the classifier on the complete movie reviews corpus.
        """
        print("I am loading and training my model on the movie reviews corpus.")
        df = pd.read_csv(self.data_path)

        # I clean all review texts using my preprocessor
        cleaned_reviews = self.preprocessor.preprocess_corpus(df['review'].tolist())

        # I map text labels into binary numbers: negative -> 0, positive -> 1
        labels = df['sentiment'].map({'negative': 0, 'positive': 1}).values

        # I convert the cleaned text strings into numerical TF-IDF feature matrices
        X = self.vectorizer.fit_transform(cleaned_reviews)

        # I fit the calibrated support vector classifier
        self.model.fit(X, labels)
        self._is_trained = True
        print("Training complete! The sentiment classifier is ready for predictions.")

    def predict(self, raw_review: str) -> dict:
        """
        I predict the sentiment of any raw text review entered by the user.
        """
        # I ensure the model has been trained before accepting inputs
        if not self._is_trained:
            raise RuntimeError("Model must be trained prior to making predictions.")

        # I clean the incoming text review
        cleaned = self.preprocessor.clean_text(raw_review)

        # If the input contains no valid words, I return neutral
        if not cleaned.strip():
            return {
                "sentiment": "Neutral/Unknown",
                "confidence": 0.50,
                "cleaned_text": ""
            }

        # I transform the cleaned sentence into the learned TF-IDF space
        vec = self.vectorizer.transform([cleaned])

        # I obtain probability estimates for both classes [negative_prob, positive_prob]
        probs = self.model.predict_proba(vec)[0]
        pos_prob = probs[1]
        neg_prob = probs[0]

        # I determine the final sentiment based on which class has the higher probability
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
    """
    I demonstrate my sentiment classifier on sample movie reviews.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, 'data', 'movie_reviews_corpus.csv')

    # I initialize and train the sentiment application
    app = SentimentClassifierApp(data_file)
    app.train()

    # I evaluate test reviews representing both strongly positive and strongly negative sentiments
    test_samples = [
        "This film was an absolute masterpiece with breathtaking cinematography and incredible acting!",
        "The plot was painfully slow, disjointed, and the characters were utterly unconvincing. A total waste of time.",
        "A charming romantic comedy that kept our entire family smiling from start to finish.",
        "Terrible script, poor sound design, and flat performances across the board."
    ]

    print("\n================ SENTIMENT ANALYSIS INFERENCE DEMONSTRATION ================")
    print("Author: Faruq Ishola Raji (4253139) | Course: DLBAIPNLP01")
    print("=============================================================================")

    for sample in test_samples:
        res = app.predict(sample)
        print(f"\nRaw Review  : \"{sample}\"")
        print(f"Cleaned Text: \"{res['cleaned_text']}\"")
        print(f"Prediction  : {res['sentiment']} (Confidence: {res['confidence']:.2%})")

    print("\n=============================================================================")
    print("Inference demonstration completed successfully!\n")


# I run the demonstration when this script is executed directly
if __name__ == '__main__':
    main()
