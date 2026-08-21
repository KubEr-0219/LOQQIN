from src.model import analyze_question_metrics, predict_question


class FakeVectorizer:
    def transform(self, questions):
        return questions


class FakeModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value]


def test_prediction_uses_model_output_without_keyword_override():
    prediction, score = predict_question(FakeModel(5.25), FakeVectorizer(), "design a secure system")
    assert prediction == 1
    assert score == 5.25


def test_score_is_clamped_to_product_range():
    _, score = predict_question(FakeModel(14), FakeVectorizer(), "analyze this")
    assert score == 10.0


def test_bloom_metrics_are_deterministic():
    clarity, specificity, bloom = analyze_question_metrics("Compare database architectures")
    assert clarity == "High"
    assert specificity == "High"
    assert bloom == "Analyze"
