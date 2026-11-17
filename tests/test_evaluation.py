from app.evaluation import evaluate
from tests.conftest import GOLDEN


def test_golden_set_meets_quality_thresholds(assistant):
    report = evaluate(assistant, GOLDEN)
    assert report["passed"], report["failures"]
