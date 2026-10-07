#!/usr/bin/env python3
"""Generate colab/*.ipynb from notebooks/*.py (jupytext py:percent).

The .py files are the source of truth. The notebooks are generated with a bootstrap
cell that clones the fork and installs dependencies on Kaggle or Colab.

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

BOOTSTRAP = """# @title Setup (Kaggle / Colab)
# Clone the fork into the writable workspace, then install the shared lab dependencies.
from pathlib import Path
import os, subprocess, sys

REPO = "https://github.com/MinMinhMin/Day21-Track3-Finetuning-Lab"
if (Path.cwd() / "scripts" / "colab_run.py").is_file():
    REPO_DIR = Path.cwd()
    if (REPO_DIR / ".git").exists():
        subprocess.run(["git", "-C", str(REPO_DIR), "remote", "set-url", "origin", REPO], check=False)
else:
    if Path("/kaggle/working").is_dir():
        WORK_DIR = Path("/kaggle/working")
    elif Path("/content").is_dir():
        WORK_DIR = Path("/content")
    else:
        WORK_DIR = Path.cwd()
    REPO_DIR = WORK_DIR / "Day21-Track3-Finetuning-Lab"
    if REPO_DIR.exists():
        if not (REPO_DIR / ".git").exists():
            raise RuntimeError(f"Existing directory is not a git checkout: {REPO_DIR}")
        subprocess.run(["git", "-C", str(REPO_DIR), "remote", "set-url", "origin", REPO], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
    else:
        subprocess.run(["git", "clone", "-q", REPO, str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR / "src"))

# Install from requirements.txt, NOT a copied list. The copied list is how the
# torchao>=0.16 pin reached requirements.txt and this bootstrap on different days --
# and a bootstrap missing a pin does not fail here, it fails 10 minutes later inside
# get_peft_model(). One source of truth. Kaggle/Colab provide CUDA-enabled torch;
# requirements.txt keeps the training stack consistent across both environments.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
               check=True)

os.environ.setdefault("COMPUTE_TIER", "T4")
import torch
gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("Repo:", REPO_DIR)
print("GPU(s):", gpu_names if gpu_names else "NONE — enable a GPU in Notebook Settings")
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
