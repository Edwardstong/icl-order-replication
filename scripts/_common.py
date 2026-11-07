"""Shared argument handling for the run scripts."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def add_model_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", required=True, help="Hugging Face model id, e.g. Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dataset", required=True, help="ag_news | dbpedia | mmlu | gsm8k")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--dtype", default=None, help="bfloat16 | float16 | float32 (default: bf16 on GPU/MPS, fp32 on CPU)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256, help="generation tasks only")
    ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. smoke")


def env_info() -> dict:
    import torch
    import transformers
    return {
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "platform": platform.platform(),
    }


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
