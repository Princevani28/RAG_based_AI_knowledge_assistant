"""AI assurance: retrieval quality, answer relevance, groundedness and out-of-scope refusal metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.assistant import KnowledgeAssistant
from app.graph.workflow import NO_ANSWER

THRESHOLDS = {"hit_rate": 0.85, "mrr": 0.7, "keyword_coverage": 0.85,
              "groundedness_rate": 0.95, "refusal_accuracy": 1.0}


def evaluate(assistant: KnowledgeAssistant, golden_path: Path | str) -> dict[str, Any]:
    cases = json.loads(Path(golden_path).read_text())
    k = assistant.settings.top_k
    hits, rr, kw_scores, grounded, refusals, rows = [], [], [], [], [], []
    for i, case in enumerate(cases):
        q = case["question"]
        in_scope = case.get("expected_source") is not None
        res = assistant.ask(q, session_id=f"eval-{i}")
        row = {"question": q, "in_scope": in_scope, "answer": res["answer"], "steps": res["steps"]}
        if in_scope:
            retrieved = [d.metadata.get("source") for d, _ in assistant.store.similarity_search_with_score(q, k=k)]
            rank = retrieved.index(case["expected_source"]) + 1 if case["expected_source"] in retrieved else None
            hits.append(1.0 if rank else 0.0)
            rr.append(1.0 / rank if rank else 0.0)
            kws = case.get("expected_keywords", [])
            cov = sum(kw.lower() in res["answer"].lower() for kw in kws) / len(kws) if kws else 1.0
            kw_scores.append(cov)
            grounded.append(1.0 if res["grounded"] else 0.0)
            row.update(rank=rank, keyword_coverage=cov, grounded=res["grounded"])
        else:
            refused = res["answer"].startswith(NO_ANSWER[:30]) or "don't know" in res["answer"].lower()
            refusals.append(1.0 if refused else 0.0)
            row["refused"] = refused
        rows.append(row)

    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else 1.0
    metrics = {"hit_rate": mean(hits), "mrr": mean(rr), "keyword_coverage": mean(kw_scores),
               "groundedness_rate": mean(grounded), "refusal_accuracy": mean(refusals)}
    failures = {m: v for m, v in metrics.items() if v < THRESHOLDS[m]}
    return {"metrics": metrics, "thresholds": THRESHOLDS, "failures": failures,
            "passed": not failures, "cases": rows}
