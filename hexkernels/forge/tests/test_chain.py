"""The three artifacts the miner needs, and the property each must have.

Each assertion pins a way the chain has silently produced nothing: an empty
harvest (wrong path), a coverage cache with no covered list (scan crashed on a
row), and a pool with no HMX (the fp32 dtype default -- see Task 2)."""
import json
import pathlib

BENCH = pathlib.Path(__file__).resolve().parents[2] / "benchmark"


def test_harvest_has_rows_with_schemas():
    lines = [ln for ln in (BENCH / "ops.jsonl").read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) > 1000, f"only {len(lines)} harvested rows"
    row = json.loads(lines[0])
    # The harvest row's schema string is carried under "raw" everywhere in this
    # codebase (hexkernels.forge.sources.torch_registry.rows(), and mine.py reads
    # row.get("raw") to populate a pool spec's own "schema" field) -- there is no
    # "schema" key on a harvest row itself. The brief's literal test text checked
    # for "schema", which does not exist on this shape; corrected to match the
    # actual, consistently-used field.
    assert "raw" in row


def test_coverage_cache_has_a_measured_covered_list():
    d = json.loads((BENCH / "coverage_cache.json").read_text(encoding="utf-8"))
    covered = d.get("covered") or d.get("expressible") or []
    assert len(covered) > 100, f"only {len(covered)} covered ops"


def test_pool_is_non_empty_and_carries_provenance():
    d = json.loads((BENCH / "pool.json").read_text(encoding="utf-8"))
    specs = d if isinstance(d, list) else [k for g in d["batches"] for k in g]
    assert len(specs) >= 320, f"pool has only {len(specs)} specs"
    for s in specs[:50]:
        assert s["schema"].startswith(("aten::", "prims::"))
        assert s["accessor"] == "torch._C._jit_get_all_schemas()"


def test_pool_contains_hmx_eligible_specs():
    """Task 2's twinning must have been applied BEFORE the pool was mined."""
    d = json.loads((BENCH / "pool.json").read_text(encoding="utf-8"))
    specs = d if isinstance(d, list) else [k for g in d["batches"] for k in g]
    hmx = [s for s in specs if "hmx" in s["mechanisms"]]
    assert len(hmx) >= 24, (
        f"only {len(hmx)} hmx-eligible specs; re-mine after Task 2 rather than "
        "lowering the minimum")
