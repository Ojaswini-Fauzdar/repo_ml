from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib


class LogisticSentimentModel:
    """
    Logistic Regression based sentiment analysis model
    with TF-IDF features.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.model = LogisticRegression(max_iter=1000)

    def train(self, texts, labels):
        """
        Train the sentiment model.
        """
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, texts):
        """
        Predict sentiment labels.
        """
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_with_confidence(self, texts):
        """
        Predict sentiment with confidence scores.
        """
        X = self.vectorizer.transform(texts)
        probs = self.model.predict_proba(X)

        predictions = []
        for i, prob in enumerate(probs):
            label = self.model.classes_[prob.argmax()]
            confidence = prob.max()
            predictions.append({
                "label": label,
                "confidence": round(float(confidence), 4)
            })

        return predictions

    def save(self, path):
        """
        Save model and vectorizer.
        """
        joblib.dump(
            {"model": self.model, "vectorizer": self.vectorizer},
            path
        )

    def load(self, path):
        """
        Load model and vectorizer.
        """
        data = joblib.load(path)
        self.model = data["model"]
        self.vectorizer = data["vectorizer"]

