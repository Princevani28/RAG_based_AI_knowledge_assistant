"""Run the AI-assurance evaluation on the golden set.  Usage: python scripts/evaluate.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.assistant import get_assistant  # noqa: E402
from app.evaluation import evaluate  # noqa: E402

if __name__ == "__main__":
    report = evaluate(get_assistant(), ROOT / "tests" / "golden" / "golden_qa.json")
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "eval_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["metrics"], indent=2))
    print("PASSED" if report["passed"] else f"FAILED: {report['failures']}")
    sys.exit(0 if report["passed"] else 1)
