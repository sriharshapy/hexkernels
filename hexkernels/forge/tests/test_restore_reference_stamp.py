"""The restore script must never be able to invent a stage-(g) pass.

It exists because `benchmark/witness_build/` was lost and re-running stage (g) costs
days, so the temptation it has to resist is writing `correct: true` for a reference
nothing ever verified. That would make every candidate graded against it a statement
about an unchecked harness.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

import restore_reference_stamp as rrs  # noqa: E402

from hexkernels.forge import evalset  # noqa: E402


@pytest.fixture(scope="module")
def ref():
    tasks, _ = evalset.core_tasks()
    return tasks[0]


def test_it_refuses_a_reference_no_record_ever_verified(ref):
    with pytest.raises(LookupError) as exc:
        rrs.restored_verdict({}, ref)
    assert "no recorded stage-(g) pass" in str(exc.value)
    assert f"--batch {ref.batch}" in str(exc.value), "must say how to build it"


def test_a_recorded_stamp_restores_a_pass_that_is_labelled_as_restored(ref):
    got = rrs.restored_verdict({ref.key: True}, ref)
    assert got["correct"] is True
    assert got["restored"] is True, "never mistakable for a fresh execution"
    assert got["restored_evidence"]
    assert got["insns"] is None and got["pcycles"] is None, \
        "the reference's own counts were never recorded; do not invent them"


def test_stamps_come_from_the_committed_record(tmp_path):
    """Read the merged jsonl, not the per-attempt files -- only the jsonl is in git."""
    import json
    d = tmp_path / "rung0" / "m"
    d.mkdir(parents=True)
    (d / "attempts.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"task": {"key": "k_pass"}, "grade": {"reference_verified": True}},
        {"task": {"key": "k_fail"}, "grade": {"reference_verified": False}},
        {"task": {"key": "k_none"}, "grade": {}},
    ]) + "\n", encoding="utf-8")
    got = rrs.stamps_for("m", root=tmp_path)
    assert got == {"k_pass": True}
