import os
import tempfile
from src.models.logistic_sentiment import LogisticSentimentModel


def test_training_and_prediction():
    texts = [
        "I love this product",
        "This is amazing",
        "I hate this",
        "This is terrible"
    ]
    labels = ["positive", "positive", "negative", "negative"]

    model = LogisticSentimentModel()
    model.train(texts, labels)

    preds = model.predict(["I love it", "I hate it"])
    assert len(preds) == 2


def test_prediction_with_confidence():
    texts = [
        "good experience",
        "bad experience"
    ]
    labels = ["positive", "negative"]

    model = LogisticSentimentModel()
    model.train(texts, labels)

    results = model.predict_with_confidence(["good", "bad"])

    assert "label" in results[0]
    assert "confidence" in results[0]
    assert 0 <= results[0]["confidence"] <= 1


def test_model_save_and_load():
    texts = ["nice", "awful"]
    labels = ["positive", "negative"]

    model = LogisticSentimentModel()
    model.train(texts, labels)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.joblib")
        model.save(path)

        new_model = LogisticSentimentModel()
        new_model.load(path)

        preds = new_model.predict(["nice"])
        assert len(preds) == 1
