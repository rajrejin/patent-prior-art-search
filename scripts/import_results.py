"""Write files printed by `scripts/export_results.py` into results/, after checking each one.

Reads the copied text (one or more FILE blocks) from a file, verifies length and CRC32 of every
block, and writes only the blocks that match. JSON is pretty-printed; other files are restored
from their escaped one-line form.
Usage: python scripts/import_results.py <copied.txt>
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main(src: str) -> int:
    lines = Path(src).read_text(encoding="utf-8").splitlines()
    bad = 0
    i = 0
    while i < len(lines):
        if not lines[i].startswith("FILE "):
            i += 1
            continue
        _, name, size, crc = lines[i].split()
        body = lines[i + 1].strip()
        raw = body.encode("ascii")
        ok = len(raw) == int(size) and zlib.crc32(raw) == int(crc)
        print(f"{name}: {'ok' if ok else 'MISMATCH'} ({len(raw)} bytes)")
        if ok:
            dest = RESULTS / name
            if name.endswith(".json"):
                dest.write_text(json.dumps(json.loads(body), indent=1) + "\n", encoding="utf-8")
            else:
                dest.write_text(json.loads(body), encoding="utf-8")
        else:
            bad += 1
        i += 3
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
