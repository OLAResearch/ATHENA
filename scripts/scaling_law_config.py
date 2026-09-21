"""Configuration for the paper-aligned ATHENA scaling-law sweep.

The reference sweep follows the four GPT-2 sizes and the two corpora used by
the MLP-Memory scaling figure.  The backbone is selected by the paper point;
all ATHENA capacity knobs are scaled with the same point rather than keeping
the Mistral-sized Engram/generator/router fixed.

This module is intentionally dependency-free so it can be used both locally
to generate a submission manifest and inside the LUMI container to validate a
point's metadata.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable


PAPER_MODELS = {
    "small": {
        "target_model": "gpt2",
        "backbone_parameters": 124_000_000,
        "num_layers": 12,
        "hidden_size": 768,
    },
    "medium": {
        "target_model": "gpt2-medium",
        "backbone_parameters": 345_000_000,
        "num_layers": 24,
        "hidden_size": 1024,
    },
    "large": {
        "target_model": "gpt2-large",
        "backbone_parameters": 774_000_000,
        "num_layers": 36,
        "hidden_size": 1280,
    },
    "xl": {
        "target_model": "gpt2-xl",
        "backbone_parameters": 1_500_000_000,
        "num_layers": 48,
        "hidden_size": 1600,
    },
}


def _multiple(value: float, base: int) -> int:
    return max(base, int(round(value / base)) * base)


def scaled_capacity(model_key: str) -> dict:
    """Return the scaled non-backbone capacity for one GPT-2 size.

    Memory table parameters are scaled approximately linearly with backbone
    parameters.  Generator/router widths use the square-root law because
    their dominant projections are quadratic in width.  Latents, layers and
    low-rank adapters also grow monotonically with the model point.
    """

    model = PAPER_MODELS[model_key]
    ratio = model["backbone_parameters"] / PAPER_MODELS["small"]["backbone_parameters"]
    width_ratio = math.sqrt(ratio)

    max_ngram = 3
    heads_per_order = 4
    total_heads = (max_ngram - 1) * heads_per_order
    d_head = _multiple(64 * width_ratio, 8)
    target_memory_parameters = int(round(33_554_432 * ratio))
    table_size = _multiple(
        target_memory_parameters / (total_heads * d_head),
        1024,
    )

    generator_hidden_size = _multiple(256 * width_ratio, 8)
    if generator_hidden_size < 512:
        generator_heads = 4
    elif generator_hidden_size < 800:
        generator_heads = 8
    else:
        generator_heads = 16
    # MultiheadAttention requires an exact divisibility relation.
    generator_hidden_size = _multiple(256 * width_ratio, generator_heads)
    generator_layers = {"small": 2, "medium": 2, "large": 3, "xl": 4}[model_key]
    generator_num_latents = {"small": 4, "medium": 6, "large": 8, "xl": 12}[model_key]
    router_hidden_size = _multiple(64 * width_ratio, 8)
    router_semantic_size = router_hidden_size
    source_adapter_rank = _multiple(16 * width_ratio, 8)

    return {
        "max_ngram": max_ngram,
        "heads_per_order": heads_per_order,
        "table_size": table_size,
        "d_head": d_head,
        "memory_parameters_target": target_memory_parameters,
        "generator_num_latents": generator_num_latents,
        "generator_hidden_size": generator_hidden_size,
        "generator_layers": generator_layers,
        "generator_heads": generator_heads,
        "generator_cue_window": 3,
        "generator_router_hidden_size": router_hidden_size,
        "generator_router_semantic_size": router_semantic_size,
        "generator_source_adapter_rank": source_adapter_rank,
        "adaptor_branches": 1,
        "injection_layers": [model["num_layers"] // 3],
        "architecture": "generative",
        "reader_type": "cross_attention",
        "generator_cue_source": "hybrid",
        "generator_fusion_type": "tri_reader",
    }


def build_points() -> list[dict]:
    """Build panels (a), (b), and (c) from the paper's scaling protocol."""

    points: list[dict] = []
    for panel, corpus, max_tokens in (
        ("a", "wikitext", 100_000_000),
        ("b", "general-mixed", 600_000_000),
    ):
        for model_key in PAPER_MODELS:
            points.append(
                make_point(
                    panel=panel,
                    model_key=model_key,
                    corpus=corpus,
                    max_tokens=max_tokens,
                )
            )

    # Figure 5(c)-style compute scaling: hold GPT-2-XL and the mixed Web
    # corpus fixed while varying the per-stage token budget.
    for max_tokens in (10_000_000, 30_000_000, 100_000_000, 300_000_000):
        points.append(
            make_point(
                panel="c",
                model_key="xl",
                corpus="general-mixed",
                max_tokens=max_tokens,
                compute_label=f"{max_tokens // 1_000_000}M",
            )
        )
    return points


def make_point(
    *,
    panel: str,
    model_key: str,
    corpus: str,
    max_tokens: int,
    compute_label: str | None = None,
) -> dict:
    model = PAPER_MODELS[model_key]
    point_id = f"panel-{panel}-{model_key}"
    if compute_label is not None:
        point_id = f"panel-{panel}-xl-{compute_label.lower()}"
    return {
        "point_id": point_id,
        "panel": panel,
        "model_key": model_key,
        "target_model": model["target_model"],
        "backbone_parameters": model["backbone_parameters"],
        "backbone_hidden_size": model["hidden_size"],
        "backbone_layers": model["num_layers"],
        "corpus": corpus,
        "router_corpus": corpus,
        "dataset_protocol": {
            "wikitext": "Salesforce/wikitext:wikitext-103-raw-v1",
            "general-mixed": (
                "WikiText-103 + amazon_polarity + cc_news + imdb; "
                "equal token mixture"
            ),
        }[corpus],
        "max_tokens_per_stage": max_tokens,
        "early_stopping_patience": 3,
        "compute_label": compute_label,
        **scaled_capacity(model_key),
    }


def write_manifest(path: str | Path) -> list[dict]:
    points = build_points()
    output = {
        "protocol": "MLP-Memory-Figure-5-aligned-ATHENA-scaling-v1",
        "backbone_fixed_by_point": True,
        "non_backbone_capacity_scaled": True,
        "early_stopping": {
            "patience": 3,
            "selection_metric": "validation perplexity",
            "budget_is_per_stage": True,
        },
        "points": points,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    return points


def iter_points(panel: str | None = None) -> Iterable[dict]:
    for point in build_points():
        if panel is None or point["panel"] == panel:
            yield point


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        print(json.dumps({"points": build_points()}, indent=2))
    else:
        write_manifest(args.output)
