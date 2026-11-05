"""Print result files compactly for copying out of a remote notebook (no file download).

Each file is printed as `FILE <name> <length> <crc32>`, then its compact content on one line,
then `ENDFILE`. JSON is reduced before printing: per-query score tables are dropped and floats
are rounded to 4 decimals. `scripts/import_results.py` checks length and CRC on the other side.
Usage: python scripts/export_results.py <file> [<file> ...]   (paths relative to results/)
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parents[1] / "results"
DROP = {"per_query"}


def reduce(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: reduce(v) for k, v in x.items() if k not in DROP}
    if isinstance(x, list):
        return [reduce(v) for v in x]
    return x


def compact(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.dumps(reduce(json.loads(text)), separators=(",", ":"), ensure_ascii=True)
    return json.dumps(text, ensure_ascii=True)  # one line, newlines escaped


def main(names: list[str]) -> None:
    for name in names:
        body = compact(RESULTS / name)
        raw = body.encode("ascii")
        print(f"FILE {name} {len(raw)} {zlib.crc32(raw)}")
        print(body)
        print("ENDFILE")


if __name__ == "__main__":
    main(sys.argv[1:])
