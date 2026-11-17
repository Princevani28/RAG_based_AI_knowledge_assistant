"""Usage: python scripts/ingest.py [path ...]   (defaults to data/documents)"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.assistant import KnowledgeAssistant  # noqa: E402
from app.config import setup_logging  # noqa: E402

if __name__ == "__main__":
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args()
    print(json.dumps(KnowledgeAssistant().ingest(args.paths or None), indent=2))
