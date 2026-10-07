#!/usr/bin/env python3
"""Generate colab/*.ipynb from notebooks/*.py (jupytext py:percent).

The .py files are the source of truth. The Colab notebooks are generated, plus a
bootstrap cell that clones the repo and installs deps — Colab starts with no repo.

Usage: python scripts/build_colab.py    (needs jupytext)
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks"
OUT = ROOT / "colab"

BOOTSTRAP = """# @title Setup (chạy ô này trước)
# Colab bắt đầu với một máy trống — clone repo và cài dependency.
import os, subprocess, sys

REPO = "https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-Lab.git"
if not os.path.exists("Day21-Track3-Finetuning-Lab"):
    subprocess.run(["git", "clone", "-q", REPO], check=True)
os.chdir("Day21-Track3-Finetuning-Lab")
sys.path.insert(0, "src")

# Install from requirements.txt, NOT a copied list. The copied list is how the
# torchao>=0.16 pin reached requirements.txt and this bootstrap on different days --
# and a bootstrap missing a pin does not fail here, it fails 10 minutes later inside
# get_peft_model(). One source of truth. torch is preinstalled on Colab and
# requirements.txt pins it compatibly, so that line is a no-op.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
               check=True)

os.environ.setdefault("COMPUTE_TIER", "T4")
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — Runtime > Change runtime type > T4 GPU")
"""


def _stamp_cell_ids(raw: dict) -> None:
    """Give every cell an id derived from its position and content.

    nbformat mints a RANDOM id per cell, so regenerating unchanged notebooks produced a
    ~90-line diff of nothing but id churn. That is not cosmetic: a diff that is always
    noise is a diff nobody reads, which is how the stale-bootstrap bug (F-18) survived
    a review. Now `make colab` on unchanged sources is a genuinely empty diff, and any
    line that does move is a line that means something.
    """
    for i, cell in enumerate(raw.get("cells", [])):
        body = "".join(cell.get("source", []))
        cell["id"] = hashlib.sha1(f"{i}\x00{body}".encode()).hexdigest()[:8]


def main() -> int:
    try:
        import jupytext
    except ImportError:
        print("jupytext not installed:  pip install jupytext", file=sys.stderr)
        return 1

    OUT.mkdir(exist_ok=True)
    made = []
    for src in sorted(SRC.glob("*.py")):
        nb = jupytext.read(src, fmt="py:percent")
        import nbformat
        nb.cells.insert(0, nbformat.v4.new_code_cell(BOOTSTRAP))
        dest = OUT / f"Lab21_{src.stem}.ipynb"
        # ensure_ascii=True: Vietnamese content must survive tooling that assumes ASCII
        jupytext.write(nb, dest, fmt="ipynb")
        raw = json.loads(dest.read_text(encoding="utf-8"))
        _stamp_cell_ids(raw)
        dest.write_text(json.dumps(raw, ensure_ascii=True, indent=1), encoding="utf-8")
        made.append(dest.name)
    print(f"wrote {len(made)} notebooks to colab/:")
    for m in made:
        print("  ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
