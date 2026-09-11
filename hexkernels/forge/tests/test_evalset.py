"""The eval core must resolve to real, uniquely-keyed pipeline positions.

Every one of these is a property of the FROZEN artifacts, so a failure here means
either the artifacts moved or the mapping rule drifted -- both of which invalidate
any results already collected against them.
"""
from hexkernels.forge import evalset, mined


def test_every_core_id_resolves_to_a_spec():
    tasks, report = evalset.core_tasks()
    assert report["missing"] == [], report["missing"]
    assert report["requested"] == 128
    assert report["resolved"] == 128
    assert len(tasks) == 128


def test_core_is_32_per_tier():
    tasks, _ = evalset.core_tasks()
    for tier in ("T0", "T1", "T2", "T3"):
        assert sum(1 for t in tasks if t.tier == tier) == 32


def test_keys_are_unique_even_where_names_are_not():
    """The reason `(batch, index)` is the key rather than the kernel name.

    Names collide across tiers by construction (`mined._load` puts no tier in the
    name), so a results file keyed on the name would overwrite a sibling task.
    """
    tasks, _ = evalset.core_tasks()
    assert len({t.key for t in tasks}) == 128
    assert len({t.task_id for t in tasks}) == 128


def test_names_match_mined():
    """`evalset._kernel_name` duplicates `mined._load`'s rule; assert they agree.

    Checked over the whole 320 rather than the core, because the duplication is
    what makes resolving ids cheap (no torch, no golden draw) and a divergence
    would point every artifact path at a directory that does not exist.
    """
    for ref in evalset.all_tasks():
        specs = mined.MINED_BATCHES[ref.batch]
        assert specs[ref.index].name == ref.name, ref.key


def test_positions_land_on_the_expected_tier():
    """A TaskRef's tier must match the spec the position actually resolves to."""
    for ref in evalset.all_tasks():
        spec = mined.MINED_BATCHES[ref.batch][ref.index]
        assert spec.expect_tier == ref.tier, ref.key


def test_reference_status_is_honest_about_unbuilt_batches():
    """An unbuilt reference is `verified: False` with a reason, never a guess."""
    tasks, _ = evalset.core_tasks()
    for ref in tasks:
        st = evalset.reference_status(ref)
        assert isinstance(st["verified"], bool)
        if not st["verified"]:
            assert st["reason"], ref.key
        else:
            assert st["harness"], ref.key
