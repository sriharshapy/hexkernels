"""Path anchors must resolve inside THIS repository.

Flattening the package moved every module up one directory. A `parents[N]`
that is off by one does not raise -- it silently points at a sibling of the
repo, and the miner then reports "no coverage cache" instead of "wrong path".
"""
import inspect
import pathlib

from hexkernels.forge import mine, mined

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_mine_repo_is_this_repository():
    assert mine.REPO == REPO_ROOT


def test_coverage_cache_is_inside_the_repo():
    assert mine.COVERAGE_CACHE.is_relative_to(REPO_ROOT)
    assert mine.COVERAGE_CACHE.parent.name == "benchmark"


def test_selection_is_inside_the_repo():
    assert mined.SELECTION.is_relative_to(REPO_ROOT)
    assert mined.SELECTION.parent.name == "benchmark"


def test_no_module_points_outside_the_repo():
    """The bug this pins: parents[2] resolved to the repo's PARENT."""
    for p in (mine.REPO, mine.COVERAGE_CACHE, mined.SELECTION, mine.DEFAULT_POOL):
        assert REPO_ROOT in (p, *p.parents), f"{p} escapes {REPO_ROOT}"


def test_mine_json_default_is_inside_the_repo():
    """`--json` is the flag the next task's miner invocation writes through.

    A default that re-hardcodes `run_artifacts/forge2/...` would write
    outside this repo (or fail) instead of into benchmark/.
    """
    assert mine.DEFAULT_POOL.is_relative_to(REPO_ROOT)
    assert mine.DEFAULT_POOL.parent.name == "benchmark"


def test_mine_cli_json_default_reuses_the_module_constant():
    """The `--json` default in `main()` must read `DEFAULT_POOL`, not a
    second, independently hardcoded path string -- one source of truth."""
    src = inspect.getsource(mine.main)
    assert "DEFAULT_POOL" in src, (
        "main()'s --json default should reuse mine.DEFAULT_POOL instead of "
        "hardcoding its own path")
