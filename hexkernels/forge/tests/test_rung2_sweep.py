"""`sweep --all-turns` reads the SUPERSEDED candidates, so it must find them all.

Rung 1's open item: mechanism engagement can be counted on the FINAL candidate or on
ANY turn, and rung 1 was the first rung where the two differed (0 finally, 1 at some
point). Reporting only the final count would have erased the single most informative
attempt in the study.

At rung 2, waves rewrite candidates in volume, so this stops being a footnote. The
scan is POST-HOC on purpose -- it reads `candidate_turn<N>.cpp` off disk, so it can
run at any time and cannot be invalidated by wave ordering, which is the one thing
threading it into `turn` would have cost.
"""
import json

import pytest

from hexkernels.forge import rung0


class _Ref:
    key, name, batch, index, tier = "b1i0_x", "x", 1, 0, "T0"
    task_id, dtype, shape, mechanisms, entry = "t", "fp32", (4,), (), {}


def _attempt(tmp_path, files):
    adir = tmp_path / "rung2" / "m" / "attempts" / "b1i0_x" / "seed1"
    adir.mkdir(parents=True)
    (adir / "attempt.json").write_text(json.dumps({"rung": 2}), encoding="utf-8")
    for name, text in files.items():
        (adir / name).write_text(text, encoding="utf-8")
    return adir


@pytest.fixture(autouse=True)
def _no_toolchain(monkeypatch):
    """These tests are about PLUMBING -- which files, in which order, under which
    turn number -- so the two SDK-reading helpers are stubbed. Leaving them live
    would make the test fail on a machine without the vendor headers for a reason
    that has nothing to do with what it checks."""
    from hexkernels.gym import intrinsic_repair, metrics
    monkeypatch.setattr(intrinsic_repair, "valid_intrinsics",
                        lambda *a, **k: frozenset())
    monkeypatch.setattr(metrics, "hallucinated_intrinsics", lambda *a, **k: set())


def test_it_scans_every_saved_turn_in_order(tmp_path, monkeypatch):
    _attempt(tmp_path, {
        "candidate_turn1.cpp": "// turn 1\nvoid k(){}\n",
        "candidate_turn2.cpp": "// turn 2\nvoid k(){}\n",
        "candidate.cpp": "// final\nvoid k(){}\n",
    })
    monkeypatch.setattr(rung0, "attempt_dir",
                        lambda *a, **k: tmp_path / "rung2" / "m" / "attempts"
                        / "b1i0_x" / "seed1")
    # No toolchain in this test: stub the scanner and assert the PLUMBING, which is
    # what has a bug in it (which files, in which order, under which turn number).
    seen = []

    def fake_mechanisms_of(src, **kw):
        seen.append(src.splitlines()[0])
        return {"scanned": True, "compiled": True,
                "mechanisms": {"hvx_compute": "turn 2" in src}, "error_text": ""}

    from hexkernels.forge import verify
    monkeypatch.setattr(verify, "mechanisms_of", fake_mechanisms_of)

    rec = rung0.sweep_turns_one(_Ref(), "m", 1, root=str(tmp_path), rung=2)
    scans = rec["mechanism_scan_turns"]
    assert [s["turn"] for s in scans] == [1, 2, 3]
    assert [s["is_final"] for s in scans] == [False, False, True]
    assert seen == ["// turn 1", "// turn 2", "// final"]
    assert scans[1]["mechanisms"]["hvx_compute"] is True
    assert scans[2]["mechanisms"]["hvx_compute"] is False


def test_a_single_turn_attempt_yields_one_scan_marked_final(tmp_path, monkeypatch):
    _attempt(tmp_path, {"candidate.cpp": "void k(){}\n"})
    monkeypatch.setattr(rung0, "attempt_dir",
                        lambda *a, **k: tmp_path / "rung2" / "m" / "attempts"
                        / "b1i0_x" / "seed1")
    from hexkernels.forge import verify
    monkeypatch.setattr(verify, "mechanisms_of",
                        lambda src, **kw: {"scanned": True, "compiled": True,
                                           "mechanisms": {}, "error_text": ""})
    rec = rung0.sweep_turns_one(_Ref(), "m", 1, root=str(tmp_path), rung=2)
    assert [s["turn"] for s in rec["mechanism_scan_turns"]] == [1]
    assert rec["mechanism_scan_turns"][0]["is_final"] is True


def test_it_does_not_touch_the_final_mechanism_scan(tmp_path, monkeypatch):
    """`mechanism_scan` is the headline field and the aggregator reads it. A turn
    scan must be additive, or a re-run of this command would silently redefine the
    rung's own result."""
    adir = _attempt(tmp_path, {"candidate.cpp": "void k(){}\n"})
    rec0 = json.loads((adir / "attempt.json").read_text(encoding="utf-8"))
    rec0["mechanism_scan"] = {"scanned": True, "mechanisms_fired": ["hvx"]}
    (adir / "attempt.json").write_text(json.dumps(rec0), encoding="utf-8")
    monkeypatch.setattr(rung0, "attempt_dir", lambda *a, **k: adir)
    from hexkernels.forge import verify
    monkeypatch.setattr(verify, "mechanisms_of",
                        lambda src, **kw: {"scanned": True, "compiled": True,
                                           "mechanisms": {}, "error_text": ""})
    rec = rung0.sweep_turns_one(_Ref(), "m", 1, root=str(tmp_path), rung=2)
    assert rec["mechanism_scan"]["mechanisms_fired"] == ["hvx"]


class _Args:
    """A stand-in for argparse's Namespace, carrying only what `cmd_sweep` reads."""
    def __init__(self, root, all_turns, redo=False):
        self.model = "m"
        self.seeds = 1
        self.root = str(root)
        self.rung = 2
        self.only = None
        self.tier = None
        self.redo = redo
        self.all_turns = all_turns


def _set_attempt_json(adir, **fields):
    rec = json.loads((adir / "attempt.json").read_text(encoding="utf-8"))
    rec.update(fields)
    (adir / "attempt.json").write_text(json.dumps(rec), encoding="utf-8")


def test_all_turns_processes_an_attempt_that_has_only_the_final_scan(
        tmp_path, monkeypatch):
    """THE REGRESSION THAT WOULD HAVE MADE THE FLAG A NO-OP. The run script sweeps
    once before grading, again after the retry waves, and only then runs
    `--all-turns` -- so by the time `--all-turns` runs, every attempt already
    carries `mechanism_scan`. A pre-filter keyed on that field would skip all of
    them, silently, and the feature would never scan anything in the one sequence
    it exists to serve."""
    adir = _attempt(tmp_path, {"candidate.cpp": "void k(){}\n"})
    _set_attempt_json(adir, mechanism_scan={"scanned": True,
                                            "mechanisms_fired": ["hvx"]})
    monkeypatch.setattr(rung0, "select_tasks", lambda tasks, args: [_Ref()])
    calls = []
    monkeypatch.setattr(rung0, "sweep_turns_one",
                        lambda ref, model, s, **kw: calls.append((ref.key, s)) or
                        {"mechanism_scan_turns": [{"scanned": True,
                                                   "mechanisms_fired": []}]})
    rc = rung0.cmd_sweep(_Args(tmp_path, all_turns=True))
    assert rc == 0
    assert calls == [("b1i0_x", 1)]


def test_all_turns_skips_an_attempt_already_turn_scanned_unless_redo(
        tmp_path, monkeypatch):
    adir = _attempt(tmp_path, {"candidate.cpp": "void k(){}\n"})
    _set_attempt_json(adir, mechanism_scan_turns=[{"scanned": True,
                                                    "mechanisms_fired": []}])
    monkeypatch.setattr(rung0, "select_tasks", lambda tasks, args: [_Ref()])
    calls = []
    monkeypatch.setattr(rung0, "sweep_turns_one",
                        lambda ref, model, s, **kw: calls.append((ref.key, s)) or
                        {"mechanism_scan_turns": [{"scanned": True,
                                                   "mechanisms_fired": []}]})

    rung0.cmd_sweep(_Args(tmp_path, all_turns=True, redo=False))
    assert calls == []

    rung0.cmd_sweep(_Args(tmp_path, all_turns=True, redo=True))
    assert calls == [("b1i0_x", 1)]


def test_the_ordinary_path_still_skips_an_attempt_with_a_final_scan(
        tmp_path, monkeypatch):
    """The fix must not change the no-flag behaviour: it still keys on
    `mechanism_scan`, unchanged from before this fix."""
    adir = _attempt(tmp_path, {"candidate.cpp": "void k(){}\n"})
    _set_attempt_json(adir, mechanism_scan={"scanned": True,
                                            "mechanisms_fired": ["hvx"]})
    monkeypatch.setattr(rung0, "select_tasks", lambda tasks, args: [_Ref()])
    calls = []
    monkeypatch.setattr(rung0, "sweep_one",
                        lambda ref, model, s, **kw: calls.append((ref.key, s)) or
                        {"mechanism_scan": {"scanned": True,
                                            "mechanisms_fired": ["hvx"]}})
    rung0.cmd_sweep(_Args(tmp_path, all_turns=False))
    assert calls == []
