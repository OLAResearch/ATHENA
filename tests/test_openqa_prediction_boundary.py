from scripts.eval_openqa import truncate_openqa_prediction


def test_truncate_openqa_prediction_stops_repeated_question():
    assert truncate_openqa_prediction(
        "W. Edwards Deming ; Question: who developed kaizen"
    ) == "W. Edwards Deming"


def test_truncate_openqa_prediction_preserves_normal_short_answer():
    assert truncate_openqa_prediction("James I") == "James I"
