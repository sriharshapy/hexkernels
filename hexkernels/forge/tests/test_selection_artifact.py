"""The committed artifacts ARE the test set. A diff to them is a diff to the
benchmark, which is why they are asserted rather than trusted."""
import json
import pathlib

from hexkernels.forge import kernels, mined, testset

BENCH = pathlib.Path(__file__).resolve().parents[2] / "benchmark"


def _load(name):
    with open(BENCH / name, encoding="utf-8") as f:
        return json.load(f)


def test_selection_has_320_at_80_per_tier():
    d = _load("selection.json")
    assert d["n"] == len(d["specs"]) == 320
    for tier in testset.TIERS:
        assert sum(1 for s in d["specs"] if s["tier"] == tier) == 80


def test_selection_meets_every_minimum():
    specs = _load("selection.json")["specs"]
    for mech, minimum in testset.MECHANISM_MINIMUMS.items():
        got = sum(1 for s in specs if mech in s["mechanisms"])
        assert got >= minimum, f"{mech}: {got} < {minimum}"


def _harvested_op_names():
    names = set()
    with open(BENCH / "ops.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            names.add(json.loads(line)["op"])
    return names


def test_every_spec_carries_its_provenance():
    """Every spec's provenance must trace back to the harvest -- but a FUSED
    spec's schema is `fused: aten::X then aten::Y` and its accessor is `""`
    (mine_fused leaves it empty; see AUDIT.md), so the brief's single check
    (schema starts `aten::`/`prims::`, accessor is the registry accessor)
    fails on every one of the ~100 fused specs this set contains. Checking
    each component op against the harvest is a STRONGER provenance claim than
    the string prefix it replaces: it proves both halves came from the
    registry rather than that one string was formatted a certain way.
    """
    harvested = _harvested_op_names()
    for s in _load("selection.json")["specs"]:
        stages = s.get("stages")
        if not stages:
            assert s["schema"].startswith(("aten::", "prims::")), s["schema"]
            assert s["accessor"] == "torch._C._jit_get_all_schemas()"
            continue
        assert len(stages) == 2, (testset.task_id(s), stages)
        first_op, second_op = stages[0][0], stages[1][0]
        assert s["schema"] == f"fused: aten::{first_op} then aten::{second_op}", \
            s["schema"]
        assert first_op in harvested, f"{first_op} not in the harvested registry"
        assert second_op in harvested, f"{second_op} not in the harvested registry"


def test_task_ids_are_unique():
    specs = _load("selection.json")["specs"]
    ids = [testset.task_id(s) for s in specs]
    assert len(set(ids)) == len(ids)


def test_core_is_128_and_a_subset_of_the_selection():
    specs = _load("selection.json")["specs"]
    core = _load("eval_core.json")
    assert core["n"] == len(core["task_ids"]) == 128
    assert set(core["task_ids"]) <= {testset.task_id(s) for s in specs}


def test_core_strata_cover_every_core_task_id():
    """Amendment 2: `eval_core.json` gains `strata`, the per-task sampling
    weight the core's deliberate HMX oversampling requires for a corrected
    aggregate. It must cover every id in `task_ids`."""
    core = _load("eval_core.json")
    strata = core["strata"]
    assert set(strata) == set(core["task_ids"])
    for tid, entry in strata.items():
        assert entry["stratum"] in ("hmx", "non_hmx")
        assert entry["tier"] in testset.TIERS
        assert isinstance(entry["weight"], (int, float))


def test_fp32_contractions_are_present_and_refused_hmx():
    """The negative controls. Without them, 'we grant HMX correctly' is
    indistinguishable from a rule that grants it always."""
    specs = _load("selection.json")["specs"]
    contractions = set(testset.REQUIRED_OPS["contraction"])
    controls = [s for s in specs
                if s["op"] in contractions and s["dtype"] == "float32"]
    assert controls, "no fp32 contraction negative controls in the set"
    for s in controls:
        assert "hmx" not in s["mechanisms"], f"{testset.task_id(s)} got hmx at fp32"


KNOWN_ABSENT = frozenset({"gelu", "amax", "amin"})


def test_mined_batches_are_disjoint_from_handwritten():
    """`kernels.batch(n)` resolves HANDWRITTEN_BATCHES before MINED_BATCHES
    (see `kernels.batch`'s docstring: "Batches 1-15 are hand-written here;
    16+ are MINED"). If a mined batch number collided with a hand-written
    one, the mined batch would be silently unreachable through the only
    public accessor even though `all_batches()` still counted it -- exactly
    the defect `selection.json`'s `first_batch` regressed to once already.
    """
    assert set(mined.MINED_BATCHES) & set(kernels.HANDWRITTEN_BATCHES) == set()


def test_mined_batches_start_after_the_handwritten_range():
    """Disjointness alone would also pass if the mined range were, say,
    -64..-1 -- it is the ORDERING relative to the hand-written range that
    `kernels.batch`'s resolution order depends on, so assert that directly,
    derived from HANDWRITTEN_BATCHES rather than a hardcoded 15."""
    highest_handwritten = max(kernels.HANDWRITTEN_BATCHES)
    assert mined.MINED_BATCHES, "no mined batches loaded"
    assert min(mined.MINED_BATCHES) > highest_handwritten


def test_kernels_batch_resolves_mined_batches_to_the_mined_spec():
    """The assertion that actually pins the bug. Set-disjointness and
    ordering can both hold while `kernels.batch(n)` still resolves a mined
    number to a hand-written spec if resolution order itself is broken --
    this checks the round trip through the real accessor returns the exact
    mined tuple, not merely *a* tuple.

    Sampled across the mined range (first, a middle spread, and last) rather
    than every one of the 64 batches, since the accessor's resolution logic
    does not vary per batch number.
    """
    numbers = sorted(mined.MINED_BATCHES)
    sample = sorted(set(numbers[:2] + numbers[len(numbers) // 2:
                                           len(numbers) // 2 + 2] + numbers[-2:]))
    for n in sample:
        assert n not in kernels.HANDWRITTEN_BATCHES, (
            f"mined batch {n} collides with a hand-written batch number")
        resolved = kernels.batch(n)
        assert resolved is mined.MINED_BATCHES[n], (
            f"kernels.batch({n}) did not resolve to the mined spec")


def test_required_ops_present_or_explicitly_known_absent():
    """A required op that is silently missing is the failure this pins.

    `_take()` takes zero rather than raising when an op is not in the pool, so
    the constraint the module declares is not one it enforces. Ops in
    KNOWN_ABSENT are excluded from the pool upstream and are documented in
    AUDIT.md; anything else missing is a regression.
    """
    names = {s["op"] for s in _load("selection.json")["specs"]}
    required = sum(testset.REQUIRED_OPS.values(), ())
    missing = [o for o in required if o not in names]
    unexpected = [o for o in missing if o not in KNOWN_ABSENT]
    assert not unexpected, f"required ops missing without a waiver: {unexpected}"
    # The other half: a waiver that no longer applies is a regression hiding
    # behind a stale exemption, so KNOWN_ABSENT must still be genuinely absent.
    stale = [o for o in KNOWN_ABSENT if o in names]
    assert not stale, f"KNOWN_ABSENT ops now present -- remove the waiver: {stale}"
