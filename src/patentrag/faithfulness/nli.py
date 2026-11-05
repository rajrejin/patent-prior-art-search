"""NLI cross-encoder judge: entailment probability of a claim given the source text."""

from __future__ import annotations

from typing import Any


def passages(text: str, words: int = 110, stride: int = 80, limit: int = 8) -> list[str]:
    toks = text.split()
    if not toks:
        return []
    out = []
    for i in range(0, len(toks), stride):
        out.append(" ".join(toks[i : i + words]))
        if i + words >= len(toks) or len(out) >= limit:
            break
    return out


class NliJudge:
    def __init__(self, model_name: str):
        import torch
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model: Any = CrossEncoder(
            model_name, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        id2label = self.model.model.config.id2label
        self.entail_idx = next(
            int(i) for i, lab in id2label.items() if str(lab).lower().startswith("entail")
        )

    def entailment(self, premise: str, claim: str) -> float:
        """Max entailment probability of `claim` over windows of `premise`."""
        wins = passages(premise)
        if not wins:
            return 0.0
        probs = self.model.predict(
            [(w, claim) for w in wins], apply_softmax=True, show_progress_bar=False
        )
        return float(max(p[self.entail_idx] for p in probs))
