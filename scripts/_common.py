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
    ap.add_argument("--backend", choices=("hf", "openai"), default="hf",
                    help="hf: local transformers with prefix cache (default); openai: chat completions API, "
                         "key read from OPENAI_API_KEY, e.g. --backend openai --model gpt-5-nano")
    ap.add_argument("--workers", type=int, default=8, help="openai backend: concurrent requests")
    ap.add_argument("--reasoning-effort", default="minimal",
                    help="openai backend: reasoning_effort for GPT-5 models (minimal | low | medium | high); 'none' omits it")
    ap.add_argument("--max-completion-tokens", type=int, default=64,
                    help="openai backend, classification: completion budget per request (hidden reasoning tokens count)")


def make_scorer(args):
    """The scorer for args.backend; both expose label_token_ids / encode_prefix / predict / generate."""
    if args.backend == "openai":
        from iclorder.openai_scorer import OpenAIScorer
        effort = None if args.reasoning_effort in ("none", "") else args.reasoning_effort
        return OpenAIScorer(args.model, workers=args.workers, reasoning_effort=effort,
                            max_completion_tokens=args.max_completion_tokens, cache_dir=RESULTS_DIR / "openai_cache")
    from iclorder.scorer import CachedScorer
    return CachedScorer(args.model, device=args.device, dtype=args.dtype, batch_size=args.batch_size)


def scorer_extra(args, scorer) -> dict:
    """Backend fields for summary.json; API usage and cost when the scorer tracks them."""
    out = {"backend": args.backend}
    if hasattr(scorer, "usage_summary"):
        out["api"] = scorer.usage_summary()
    return out


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
