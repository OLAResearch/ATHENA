from pathlib import Path

from scripts.scaling_law_config import build_points


def test_scaling_points_keep_router_corpus_aligned_with_training_corpus():
    points = build_points()

    assert all(point["router_corpus"] == point["corpus"] for point in points)
    assert all(
        point["router_corpus"] == "wikitext"
        for point in points
        if point["panel"] == "a"
    )


def test_scaling_slurm_does_not_replace_wikitext_router_corpus():
    script = Path("run/lumi_scaling_law_point_20260918.slurm").read_text()

    assert 'ROUTER_CORPUS="wikipedia-2021"' not in script
    assert '--corpus "$ATHENA_ROUTER_CORPUS"' in script
    assert 'POINT_ROOT=${ATHENA_POINT_ROOT:-$DEFAULT_POINT_ROOT}' in script
    assert 'SOURCE_OUT=${ATHENA_SOURCE_OUT:-$POINT_ROOT/source_memory}' in script
    assert 'COMPLETION_MARKER=${ATHENA_COMPLETION_MARKER:-$POINT_ROOT/COMPLETED}' in script
