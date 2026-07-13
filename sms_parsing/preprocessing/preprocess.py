"""
Preprocessing module for SMS text data.
Handles text cleaning, tokenization, and sequence padding.
"""

import re
from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SMSPreprocessor:
    """
    Preprocessor for SMS text data with tokenization and sequence padding.
    """

    def __init__(self, max_vocab_size: int = 5000, max_seq_length: int = 100):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.tokenizer: Tokenizer = Tokenizer(
            num_words=max_vocab_size,
            oov_token="<OOV>",
            filters=""
        )
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.classes_: List[str] = []

    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"[^a-z0-9₹%./\-\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit(self, texts: List[str], labels: List[str]) -> None:
        cleaned_texts = [self.clean_text(text) for text in texts]
        self.tokenizer.fit_on_texts(cleaned_texts)
        self.label_encoder.fit(labels)
        self.classes_ = self.label_encoder.classes_.tolist()

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.tokenizer.texts_to_sequences(cleaned_texts)

    def pad_sequences(self, sequences: List[List[int]]) -> np.ndarray:
        return pad_sequences(
            sequences,
            maxlen=self.max_seq_length,
            padding="post",
            truncating="post",
            value=0,
        )

    def transform(self, texts: List[str], labels: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        if not self.tokenizer.word_index:
            raise ValueError("Preprocessor must be fitted first using fit()")

        sequences = self.texts_to_sequences(texts)
        padded_sequences = self.pad_sequences(sequences)

        labels_array = None
        if labels is not None:
            labels_array = self.label_encoder.transform(labels)

        return padded_sequences.astype(np.int32), labels_array

    def fit_transform(self, texts: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(texts, labels)
        return self.transform(texts, labels)

    def get_config(self) -> Dict[str, Any]:
        vocab_size = min(self.max_vocab_size, len(self.tokenizer.word_index) + 1)
        return {
            "max_vocab_size": self.max_vocab_size,
            "max_seq_length": self.max_seq_length,
            "vocab_size": vocab_size,
            "num_classes": len(self.classes_),
            "classes": self.classes_,
        }
