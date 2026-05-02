"""Regenerate every demo GIF under demo/.

The README links into demo/ for inline GIFs; this script reruns each
example with --demo so all of them stay in sync after code changes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEMOS = [
    "example/defect_free_assembly_benchmark.py",
    "example/inversion_ripa_aod.py",
    "example/inversion_6x6_benchmark.py",
    "example/random_routing_tones_demo.py",
]


def main() -> None:
    for rel in DEMOS:
        cmd = [sys.executable, str(ROOT / rel), "--demo"]
        print(f"\n$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
