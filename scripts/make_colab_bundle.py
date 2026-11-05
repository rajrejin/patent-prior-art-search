"""Zip the code, configs and corpus for a Colab run. Never includes .env, raw OPS data or stores.

Usage: python scripts/make_colab_bundle.py   ->  colab/patentrag_bundle.zip
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ["src", "api", "app", "configs", "scripts", "tests"]
INCLUDE_FILES = [
    "pyproject.toml",
    "data/corpus/patents_full.jsonl",
    "data/corpus/manifest_full.json",
    "data/corpus/selection_full.json",
    "data/corpus/closure_full.json",
    "results/testset_manifest_full.json",
    "results/corpus_stats_full.json",
    "results/graph_load_full.json",
    "results/environment.json",
]
OPTIONAL_FILES = [
    "results/tuned_params_full.json",
    "results/eval_full_leak_demo.json",
    "results/eval_full_leak_demo_default.json",
    "results/ablations_full_test.json",
    "results/robustness_full.json",
]
SKIP = {"__pycache__", ".pyc"}


def main() -> None:
    out = ROOT / "colab" / "patentrag_bundle.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE_DIRS:
            for f in sorted((ROOT / d).rglob("*")):
                if f.is_file() and not any(s in str(f) for s in SKIP):
                    z.write(f, f.relative_to(ROOT).as_posix())
        for rel in INCLUDE_FILES:
            z.write(ROOT / rel, rel)
        # tuned parameters and part-1 results, when present, so `pipeline.sh agent` can resume
        for rel in OPTIONAL_FILES:
            if (ROOT / rel).exists():
                z.write(ROOT / rel, rel)
    assert not any(n.endswith(".env") for n in zipfile.ZipFile(out).namelist())
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
