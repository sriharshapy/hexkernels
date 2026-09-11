"""Depth is metadata, not an axis -- which only works if it is recorded.

Rejecting a depth AXIS (see the design doc) was justified by depth being
measurable after the fact. That justification fails silently if the field is
absent, and the audit's confound check reads it.

Arithmetic intensity is recorded for the same reason (design doc, spec section
9): it is not yet part of entitlement, but recording it now means the set can
be re-labelled once the silicon measurement lands, without re-selecting.
"""
import json
import pathlib

BENCH = pathlib.Path(__file__).resolve().parents[2] / "benchmark"


def test_every_spec_records_its_depth():
    with open(BENCH / "selection.json", encoding="utf-8") as f:
        specs = json.load(f)["specs"]
    missing = [s["op"] for s in specs if s.get("n_primitives") is None]
    assert not missing, f"{len(missing)} specs lack n_primitives: {missing[:5]}"


def test_depth_is_a_positive_int():
    with open(BENCH / "selection.json", encoding="utf-8") as f:
        specs = json.load(f)["specs"]
    for s in specs:
        assert isinstance(s["n_primitives"], int) and s["n_primitives"] >= 1
        assert isinstance(s["n_plumbing"], int) and s["n_plumbing"] >= 0


def test_intensity_is_recorded():
    with open(BENCH / "selection.json", encoding="utf-8") as f:
        specs = json.load(f)["specs"]
    assert all("ops_per_element" in s for s in specs)
