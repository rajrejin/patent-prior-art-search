"""NLI entailment threshold, chosen on validation reports and stored in results/."""

from __future__ import annotations

import json
from pathlib import Path

from patentrag.config import ROOT


def threshold_path() -> Path:
    return ROOT / "results" / "nli_threshold.json"


def load_threshold(default: float = 0.5) -> float:
    p = threshold_path()
    if p.exists():
        return float(json.loads(p.read_text(encoding="utf-8"))["threshold"])
    return default
