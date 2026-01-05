from typing import Dict, List, Optional, Union

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


class SentimentAnalyzer:
    """Basic sentiment analyzer - participants need to implement ML models."""

    def __init__(self) -> None:
        """Initialize the sentiment analyzer."""
        # ML pipeline (vectorizer + classifier)
        self.pipeline: Optional[Pipeline] = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),        # bi-grams improve accuracy
                min_df=1,
                max_features=20000,
                sublinear_tf=True
            )),
            ("clf", LogisticRegression(
                max_iter=400,
                n_jobs=-1
            ))
        ])

        self.is_trained: bool = False
        self.train_accuracy: float = 0.0
        self.n_samples: int = 0

    # --------------------------- TRAIN --------------------------- #
    def train(self, texts: List[str], labels: List[str]) -> None:
        if not texts or not labels or len(texts) != len(labels):
            raise ValueError("Texts and labels must be non-empty and equal length")

        self.pipeline.fit(texts, labels)

        preds = self.pipeline.predict(texts)
        self.train_accuracy = float(accuracy_score(labels, preds))
        self.n_samples = len(texts)
        self.is_trained = True

    # --------------------------- PREDICT --------------------------- #
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        if not self.is_trained or not self.pipeline:
            return {"sentiment": "neutral", "confidence": 0.5}

        if not text.strip():
            return {"sentiment": "neutral", "confidence": 0.0}

        pred = self.pipeline.predict([text])[0]

        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            confidence = float(np.max(self.pipeline.predict_proba([text])))
        else:
            confidence = 0.7

        return {"sentiment": str(pred), "confidence": confidence}

    # ------------------------- BATCH PREDICT ----------------------- #
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        if not self.is_trained or not self.pipeline:
            return [{"sentiment": "neutral", "confidence": 0.5} for _ in texts]

        preds = self.pipeline.predict(texts)

        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            confs = np.max(self.pipeline.predict_proba(texts), axis=1)
        else:
            confs = np.full(len(texts), 0.7)

        return [
            {"sentiment": str(s), "confidence": float(c)}
            for s, c in zip(preds, confs)
        ]

    # --------------------------- META INFO -------------------------- #
    def get_model_info(self) -> Dict[str, Union[str, float]]:
        return {
            "model_type": "LogisticRegression + TF-IDF",
            "status": "trained" if self.is_trained else "untrained",
            "train_accuracy": self.train_accuracy,
            "n_samples": self.n_samples,
        }
