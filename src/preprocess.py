"""
Text Preprocessing Module for Sentiment Analysis
Course: DLBAIPNLP01 - Project: NLP
Author: Faruq Ishola Raji (4253139)
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """
    Cleans raw movie review text for natural language processing models.
    Applies lowercasing, punctuation stripping, stop word filtering, and lemmatization.
    """

    def __init__(self, language: str = 'english', remove_stopwords: bool = True, lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Load stop words from NLTK
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words(language))
            # Preserve critical negation words that shift sentiment
            self.negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none'}
            self.stop_words = self.stop_words - self.negation_words
        else:
            self.stop_words = set()

        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None

    def clean_text(self, text: str) -> str:
        """
        Cleans a single string of text.
        """
        if not isinstance(text, str):
            return ""

        # Remove HTML break tags and formatting often present in scraped reviews
        text = re.sub(r'<.*?>', ' ', text)
        
        # Lowercase conversion
        text = text.lower()
        
        # Remove numbers and special characters, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Tokenize by splitting whitespace
        tokens = text.split()
        
        # Filter stop words and short tokens
        if self.remove_stopwords:
            tokens = [t for t in tokens if (t not in self.stop_words and len(t) > 1)]
            
        # Lemmatize tokens
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            
        return ' '.join(tokens)

    def preprocess_corpus(self, texts: List[str]) -> List[str]:
        """
        Processes a list of text reviews.
        """
        return [self.clean_text(t) for t in texts]


if __name__ == '__main__':
    sample = "The movie was NOT good! It was totally boring and a waste of 2 hours. <br />"
    prep = TextPreprocessor()
    print("Original:", sample)
    print("Cleaned: ", prep.clean_text(sample))
