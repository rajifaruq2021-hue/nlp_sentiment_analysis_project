"""
preprocess.py

I am the text preprocessing module for my NLP Sentiment Analysis project.
Course: DLBAIPNLP01 - Project: NLP
Author: Faruq Ishola Raji
Matriculation Number: 4253139

What I do in this file:
1. I take raw, messy movie review texts from the dataset.
2. I remove HTML tags, web formatting, numbers, and special symbols.
3. I convert all text into clean lowercase words.
4. I filter out common English stop words (like 'the', 'is', 'at') while carefully keeping negation words
   (like 'not', 'no', 'never') because they are essential for detecting negative sentiment.
5. I lemmatize words back to their dictionary root form (for example, 'running' becomes 'run').
"""

# I import regular expressions and string tools for cleaning text
import re
import string
from typing import List, Optional

# I import NLTK (Natural Language Toolkit) tools for stop words and word lemmatization
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """
    I prepare and clean raw text reviews so that machine learning algorithms can learn from them.
    """

    def __init__(self, language: str = 'english', remove_stopwords: bool = True, lemmatize: bool = True):
        """
        I configure my preprocessor settings.
        """
        # I record whether the user wants to remove stop words and apply lemmatization
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize

        # I load the list of standard English stop words from NLTK
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words(language))

            # I make sure I keep negation words!
            # If I remove "not" from "not good", the computer might think it is positive.
            # So I explicitly keep negation words in my vocabulary.
            self.negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none'}
            self.stop_words = self.stop_words - self.negation_words
        else:
            self.stop_words = set()

        # I set up the WordNet lemmatizer to reduce words to their base dictionary roots
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None

    def clean_text(self, text: str) -> str:
        """
        I take a single raw review string and return a clean, normalized string.
        """
        # If the input is empty or not a string, I safely return an empty string
        if not isinstance(text, str):
            return ""

        # 1. I remove HTML break tags like <br /> and other web markup commonly found in scraped reviews
        text = re.sub(r'<.*?>', ' ', text)

        # 2. I convert all characters to lowercase so that "Good", "good", and "GOOD" are treated as the same word
        text = text.lower()

        # 3. I strip out numbers, punctuation, and symbols, keeping only letters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)

        # 4. I split the text string into individual word tokens
        tokens = text.split()

        # 5. I filter out stop words and single-letter words
        if self.remove_stopwords:
            tokens = [t for t in tokens if (t not in self.stop_words and len(t) > 1)]

        # 6. I lemmatize each word back to its base form (e.g., "movies" -> "movie")
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        # 7. I join the cleaned tokens back into a single clean sentence
        return ' '.join(tokens)

    def preprocess_corpus(self, texts: List[str]) -> List[str]:
        """
        I take a whole list of text reviews and clean each one using my clean_text method.
        """
        return [self.clean_text(t) for t in texts]


# I run a small demonstration to verify the cleaning process when this script is executed directly
if __name__ == '__main__':
    sample_text = "The movie was NOT good! It was totally boring and a waste of 2 hours. <br />"
    preprocessor = TextPreprocessor()

    print("--- Text Preprocessor Verification ---")
    print("Original Text :", sample_text)
    print("Cleaned Text  :", preprocessor.clean_text(sample_text))
    print("Verification complete!")
