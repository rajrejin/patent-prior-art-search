"""Claim-level support judging with two independent judges and agreement statistics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from patentrag.faithfulness.claims import AtomicClaim, atomic_claims
from patentrag.llm.client import LLMClient, generate_structured
from patentrag.llm.schemas import JudgeVerdict, PriorArtReport

JUDGE_SYSTEM = (
    "You check whether a claim is supported by a source text. Use only the source text. "
    'Answer "yes" if the source text clearly states or directly implies the claim, "no" if the '
    'source text contradicts it, and "unsupported" if the source text does not say enough.'
)


def llm_judge(llm: LLMClient, source: str, claim: str) -> bool:
    prompt = f"SOURCE TEXT:\n{source[:3500]}\n\nCLAIM:\n{claim}\n\nIs the claim supported?"
    verdict, _ = generate_structured(llm, JudgeVerdict, prompt, system=JUDGE_SYSTEM, max_tokens=20)
    return verdict.verdict == "yes"


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa for two binary raters. Returns 1.0 for perfect agreement on constant data."""
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


@dataclass
class ClaimVerdict:
    claim: AtomicClaim
    llm_supported: bool | None
    nli_prob: float | None
    nli_supported: bool | None
    supported: bool


@dataclass
class SupportResult:
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.verdicts)

    @property
    def unsupported(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if not v.supported]

    def rate(self, which: str) -> float:
        vals = [
            getattr(v, f"{which}_supported")
            for v in self.verdicts
            if getattr(v, f"{which}_supported") is not None
        ]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def combined_rate(self) -> float:
        return sum(v.supported for v in self.verdicts) / self.n if self.n else 1.0


def check_support(
    report: PriorArtReport,
    sources: dict[str, str],
    query_text: str,
    *,
    llm_fn: Callable[[str, str], bool] | None,
    nli_fn: Callable[[str, str], float] | None,
    nli_threshold: float,
    flag_if_either: bool = True,
) -> SupportResult:
    """Judge every atomic claim against its source (cited patent text, or the query)."""
    out = SupportResult()
    for c in atomic_claims(report):
        src = c.premise or (query_text if c.source == "query" else sources.get(c.source, ""))
        llm_ok = llm_fn(src, c.text) if llm_fn and src else (False if llm_fn else None)
        prob = nli_fn(src, c.text) if nli_fn and src else (0.0 if nli_fn else None)
        nli_ok = None if prob is None else prob >= nli_threshold
        votes = [v for v in (llm_ok, nli_ok) if v is not None]
        if not votes:
            supported = False
        elif flag_if_either:
            supported = all(votes)
        else:
            supported = any(votes)
        out.verdicts.append(ClaimVerdict(c, llm_ok, prob, nli_ok, supported))
    return out
