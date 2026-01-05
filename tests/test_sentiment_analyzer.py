from src.models.sentiment_analyzer import SentimentAnalyzer


def test_training_and_single_prediction():
    texts = [
        "I love this product",
        "This is amazing",
        "I hate this",
        "This is terrible"
    ]
    labels = ["positive", "positive", "negative", "negative"]

    model = SentimentAnalyzer()
    model.train(texts, labels)

    result = model.predict("I really love this phone")

    assert "sentiment" in result
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0


def test_batch_prediction():
    texts = ["good movie", "bad movie"]
    labels = ["positive", "negative"]

    model = SentimentAnalyzer()
    model.train(texts, labels)

    results = model.predict_batch(["good", "bad", "okay"])

    assert len(results) == 3
    for r in results:
        assert "sentiment" in r
        assert "confidence" in r
        assert 0.0 <= r["confidence"] <= 1.0


def test_untrained_model_returns_neutral():
    model = SentimentAnalyzer()

    result = model.predict("Hello world")

    assert result["sentiment"] == "neutral"
    assert result["confidence"] == 0.5


def test_empty_input_returns_neutral_zero_confidence():
    model = SentimentAnalyzer()
    model.is_trained = True  # simulate trained model

    result = model.predict("   ")

    assert result["sentiment"] == "neutral"
    assert result["confidence"] == 0.0


def test_model_info_metadata():
    model = SentimentAnalyzer()

    info = model.get_model_info()

    assert "model_type" in info
    assert "status" in info
    assert "train_accuracy" in info
    assert "n_samples" in info
