"""Coverage constraints on the 320-task set and the 128-task core.

Every assertion here is a defect observed in the previous holdout: l2fetch
had ONE task, some tasks declared no mechanism and scored genuine vacuously
via all([]) == True, and HMX arrived only from hand-built batches.
"""
import pytest

from hexkernels.forge import testset


def _spec(op, tier, mechs, dtype="float32", dtype_bytes=4):
    return {"op": op, "overload": "", "sig": "binary", "dtype": dtype,
            "dtype_bytes": dtype_bytes, "shape": [64, 64], "tier": tier,
            "mechanisms": list(mechs), "working_set": 16384,
            "schema": f"aten::{op}(Tensor self) -> Tensor",
            "accessor": "torch._C._jit_get_all_schemas()"}


def _pool():
    """A pool rich enough to satisfy every constraint, plus filler."""
    pool = []
    mech_for = {"T0": ["hvx"], "T1": ["hvx", "l2fetch"],
                "T2": ["hvx", "l2fetch", "vtcm", "dma"],
                "T3": ["hvx", "l2fetch", "vtcm", "dma"]}
    for tier in ("T0", "T1", "T2", "T3"):
        for name in sum(testset.REQUIRED_OPS.values(), ()):
            pool.append(_spec(name, tier, mech_for[tier]))
        for i in range(200):
            pool.append(_spec(f"filler_{tier}_{i}", tier, mech_for[tier]))
        for i in range(10):  # hmx-eligible, fp16
            pool.append(_spec(f"contract_{tier}_{i}", tier,
                              mech_for[tier] + ["hmx"], "float16", 2))
    return pool


def test_selects_exactly_per_tier():
    s = testset.select_test_set(_pool())
    assert len(s) == 320
    for tier in ("T0", "T1", "T2", "T3"):
        assert sum(1 for k in s if k["tier"] == tier) == testset.PER_TIER


def test_mechanism_minimums_met():
    s = testset.select_test_set(_pool())
    for mech, minimum in testset.MECHANISM_MINIMUMS.items():
        got = sum(1 for k in s if mech in k["mechanisms"])
        assert got >= minimum, f"{mech}: {got} < {minimum}"


def test_no_task_has_an_empty_mechanism_set():
    """all([]) == True would score genuine vacuously."""
    s = testset.select_test_set(_pool())
    assert all(k["mechanisms"] for k in s)
    assert all("hvx" in k["mechanisms"] for k in s)


def test_required_ops_all_present():
    s = testset.select_test_set(_pool())
    names = {k["op"] for k in s}
    for cls, ops in testset.REQUIRED_OPS.items():
        missing = [o for o in ops if o not in names]
        assert not missing, f"{cls} missing {missing}"


def test_raises_when_pool_cannot_meet_a_minimum():
    """Fail loudly. A silently under-covered set is the defect we are fixing."""
    thin = [k for k in _pool() if "hmx" not in k["mechanisms"]]
    with pytest.raises(ValueError, match="hmx"):
        testset.select_test_set(thin)


def test_core_is_32_per_tier_and_a_subset():
    s = testset.select_test_set(_pool())
    core = testset.designate_core(s)
    assert len(core) == 128
    ids = {testset.task_id(k) for k in s}
    assert set(core) <= ids
    by_tier = {t: 0 for t in ("T0", "T1", "T2", "T3")}
    for k in s:
        if testset.task_id(k) in set(core):
            by_tier[k["tier"]] += 1
    assert all(v == testset.CORE_PER_TIER for v in by_tier.values())


def test_core_is_deterministic():
    s = testset.select_test_set(_pool())
    assert testset.designate_core(s, seed=0) == testset.designate_core(s, seed=0)


def test_core_oversamples_hmx_for_a_usable_denominator():
    """The brief's original name for this test ("preserves the hmx proportion")
    was the opposite of what the code does: `designate_core` deliberately takes
    all HMX-bearing tasks first per tier, inflating the core's HMX share well
    above the full set's. Renamed so the name matches the behaviour; see
    `test_core_strata_weights_permit_correction` for the correction this makes
    necessary."""
    s = testset.select_test_set(_pool())
    core = set(testset.designate_core(s))
    hmx = sum(1 for k in s if testset.task_id(k) in core
              and "hmx" in k["mechanisms"])
    assert hmx >= 8, f"core has only {hmx} hmx tasks"


def test_core_strata_weights_permit_correction():
    """The core's HMX share exceeds the set's. That is intended, and it is
    only sound if the correction factor is recorded."""
    s = testset.select_test_set(_pool())
    core = testset.designate_core(s)
    strata = testset.core_strata(s, core)

    assert set(strata) == set(core)

    hmx_weights = [v["weight"] for v in strata.values() if v["stratum"] == "hmx"]
    non_hmx_weights = [v["weight"] for v in strata.values()
                        if v["stratum"] == "non_hmx"]
    assert hmx_weights, "no hmx-stratum entries in the core"
    assert all(w < 1.0 for w in hmx_weights), hmx_weights
    assert non_hmx_weights, "no non-hmx-stratum entries in the core"
    # A tier whose full set already has zero hmx tasks correctly gets weight
    # 1.0 for its (only) non-hmx stratum -- there is nothing to oversample.
    # Every tier that DOES have an hmx stratum must show its non-hmx weight
    # pulled above 1.0 to compensate.
    assert all(w >= 1.0 for w in non_hmx_weights), non_hmx_weights
    assert any(w > 1.0 for w in non_hmx_weights), non_hmx_weights

    full_hmx_share = sum(1 for k in s if "hmx" in k["mechanisms"]) / len(s)
    core_ids = set(core)
    reweighted = sum(v["weight"] for v in strata.values()
                      if v["stratum"] == "hmx") / len(core)
    assert reweighted == pytest.approx(full_hmx_share, abs=1e-9)


def test_stage_aware_task_id_no_collision_with_bare_first_stage():
    """A fused spec's `op` field holds only its FIRST stage; the composition
    lives in `stages`. Without appending the later stages, a fused spec and
    its bare first-stage single op collide on task id -- and worse, two fused
    specs sharing a first stage but differing in their epilogue collide with
    each other."""
    bare = _spec("all", "T0", ["hvx"])
    fused = dict(bare)
    fused["stages"] = [["all", ""], ["amax", ""]]
    fused_other = dict(bare)
    fused_other["stages"] = [["all", ""], ["amin", ""]]

    ids = {testset.task_id(bare), testset.task_id(fused),
           testset.task_id(fused_other)}
    assert len(ids) == 3
