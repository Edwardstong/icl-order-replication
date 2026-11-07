"""Dataset registry.

Every dataset the paper uses is listed, even the ones we cannot run, so the
scope is visible in one place. `render` turns a raw Hugging Face row into
(input_text, target). For classification `target` is the integer label and the
word shown to the model is `verbalizers[target]`. For generation `target` is
the reference answer text as it should appear in a demonstration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

Kind = Literal["classification", "generation"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: Kind
    hf_id: str | None
    hf_config: str | None
    demo_split: str
    eval_split: str
    answer_key: str
    render: Callable[[dict], tuple[str, object]] | None
    verbalizers: tuple[str, ...] = ()
    paper_name: str = ""
    runnable: bool = True
    notes: str = ""

    @property
    def n_labels(self) -> int:
        return len(self.verbalizers)

    @property
    def k(self) -> int:
        """Demonstrations per prompt. Paper: k = 2|C| for classification, 8 for generation."""
        return 2 * self.n_labels if self.kind == "classification" else 8


def _ag_news(ex: dict) -> tuple[str, int]:
    return f"Article: {ex['text'].strip()}", int(ex["label"])


def _dbpedia(ex: dict) -> tuple[str, int]:
    return f"Text: {ex['content'].strip()}", int(ex["label"])


def _mmlu(ex: dict) -> tuple[str, int]:
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", ex["choices"]))
    return f"Question: {ex['question'].strip()}\n{options}", int(ex["answer"])


def _gsm8k(ex: dict) -> tuple[str, str]:
    # The reference answer keeps its rationale and the final "#### N" line, so a
    # demonstration shows the model the format it must reproduce.
    return f"Question: {ex['question'].strip()}", ex["answer"].strip()


SPECS: dict[str, DatasetSpec] = {
    "ag_news": DatasetSpec(
        name="ag_news", kind="classification", hf_id="fancyzhx/ag_news", hf_config=None,
        demo_split="train", eval_split="test", answer_key="Topic", render=_ag_news,
        verbalizers=("World", "Sports", "Business", "Technology"),
        paper_name="ag_news",
        notes="Original label names: World, Sports, Business, Sci/Tech. 'Technology' chosen as a one-word verbalizer.",
    ),
    "dbpedia": DatasetSpec(
        name="dbpedia", kind="classification", hf_id="fancyzhx/dbpedia_14", hf_config=None,
        demo_split="train", eval_split="test", answer_key="Category", render=_dbpedia,
        verbalizers=("Company", "School", "Artist", "Athlete", "Politician", "Vehicle", "Building",
                     "Nature", "Village", "Animal", "Plant", "Album", "Film", "Book"),
        paper_name="dbpedia",
        notes="DBPedia-14. One-word verbalizers replace EducationalInstitution, OfficeHolder, MeanOfTransportation, NaturalPlace, WrittenWork.",
    ),
    "mmlu": DatasetSpec(
        name="mmlu", kind="classification", hf_id="cais/mmlu", hf_config="all",
        demo_split="validation", eval_split="test", answer_key="Answer", render=_mmlu,
        verbalizers=("A", "B", "C", "D"),
        paper_name="mmlu",
        notes="Paper lists MMLU as a 4-label classification task. We read the four labels as the answer letters A-D.",
    ),
    "gsm8k": DatasetSpec(
        name="gsm8k", kind="generation", hf_id="openai/gsm8k", hf_config="main",
        demo_split="train", eval_split="test", answer_key="Answer", render=_gsm8k,
        paper_name="gsm8k",
        notes="Generation task. Needs batched decoding; run on GPU time. Test split has 1,319 rows, so dev is drawn from train.",
    ),
    # Listed for completeness. Not runnable in this scaffold.
    "nyt_topics": DatasetSpec(
        name="nyt_topics", kind="classification", hf_id=None, hf_config=None, demo_split="", eval_split="",
        answer_key="Topic", render=None, verbalizers=tuple(f"label{i}" for i in range(9)),
        paper_name="nyt-topics", runnable=False,
        notes="Built by the authors from the NYT Annotated Corpus (LDC2008T19), a licensed dataset. Construction not described.",
    ),
    "nyt_locations": DatasetSpec(
        name="nyt_locations", kind="classification", hf_id=None, hf_config=None, demo_split="", eval_split="",
        answer_key="Location", render=None, verbalizers=tuple(f"label{i}" for i in range(10)),
        paper_name="nyt-locations", runnable=False,
        notes="Same source as nyt_topics. Licensed.",
    ),
    "math": DatasetSpec(
        name="math", kind="generation", hf_id="EleutherAI/hendrycks_math", hf_config=None,
        demo_split="train", eval_split="test", answer_key="Answer", render=None,
        paper_name="math", runnable=False,
        notes="Deferred. Needs a LaTeX-aware answer normalizer before 'exact match' means anything.",
    ),
    "mmlu_pro": DatasetSpec(
        name="mmlu_pro", kind="generation", hf_id="TIGER-Lab/MMLU-Pro", hf_config=None,
        demo_split="validation", eval_split="test", answer_key="Answer", render=None,
        paper_name="mmlu-pro", runnable=False,
        notes="Deferred. Paper lists it but it is missing from its own Tables 3 and 4.",
    ),
}


def get_spec(name: str) -> DatasetSpec:
    try:
        spec = SPECS[name]
    except KeyError:
        raise SystemExit(f"Unknown dataset {name!r}. Known: {', '.join(SPECS)}") from None
    if not spec.runnable:
        raise SystemExit(f"Dataset {name!r} is not runnable in this scaffold: {spec.notes}")
    return spec
