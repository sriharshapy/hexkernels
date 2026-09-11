"""The driver's job is to lose nothing: a killed run must cost only what was in
flight, and a failed call must be recorded rather than dropped.

No network and no simulator here -- `complete` is stubbed. What is under test is
the bookkeeping around it, which is where a 640-attempt run silently loses data.
"""
import json

import pytest

from hexkernels.forge import evalset, mined, rung0, run_batch


@pytest.fixture(scope="module")
def task():
    tasks, _ = evalset.core_tasks()
    ref = tasks[0]
    art = run_batch.build(mined.MINED_BATCHES[ref.batch][ref.index])
    assert art["failed_stage"] is None, art["error"]
    return ref, art


@pytest.fixture
def verified_reference(tmp_path, monkeypatch):
    """Make `reference_status` report a usable reference, whatever is on disk.

    These tests are about `grade_one`'s BOOKKEEPING, not about the reference build.
    They used to read the real `benchmark/witness_build/`, so they broke the moment
    that regenerable directory went missing -- a test suite that depends on hours of
    local simulator output is a test suite that cannot be trusted to run anywhere.
    """
    harness = tmp_path / "harness.cpp"
    harness.write_text("int main(){return 0;}\n")
    monkeypatch.setattr(
        "hexkernels.forge.evalset.reference_status",
        lambda ref, witness_root=None: {"verified": True, "reason": "",
                                        "dir": str(tmp_path),
                                        "harness": str(harness),
                                        "results_json": None, "timing": False})
    return harness


def _stub(text, *, usage=None, error=None, finish="stop"):
    def fake(model, prompt, key, **kw):
        return {"text": text, "usage": usage or {"prompt_tokens": 10,
                                                 "completion_tokens": 20},
                "finish_reason": finish, "seed_honored": True, "error": error,
                "latency_s": 0.1, "http_attempts": 1, "response_model": model}
    return fake


def test_it_records_a_completion_and_writes_the_candidate(task, tmp_path,
                                                          monkeypatch):
    ref, art = task
    monkeypatch.setattr(rung0, "complete", _stub('extern "C" void x() {}\n'))
    rec = rung0.generate_one(ref, art, "m", 1, "key", max_tokens=100,
                             price_in=1.0, price_out=2.0, seed_base=7,
                             root=tmp_path)
    adir = rung0.attempt_dir("m", ref, 1, tmp_path)
    assert (adir / "candidate.cpp").exists()
    assert (adir / "response.txt").exists()
    assert json.loads((adir / "attempt.json").read_text())["task"]["key"] == ref.key
    assert rec["model"]["seed"] == 8            # seed_base + seed_index
    assert rec["usage"]["cost_usd"] == round(10 / 1e6 * 1 + 20 / 1e6 * 2, 6)
    assert rec["prompt"]["bytes"] > 0
    assert rec["grade"] == {}


def test_a_failed_call_is_recorded_and_writes_no_candidate(task, tmp_path,
                                                           monkeypatch):
    """An API failure is data about the run, and must not look like a bad kernel."""
    ref, art = task
    monkeypatch.setattr(rung0, "complete",
                        _stub("", error="HTTP 429: slow down", finish=None))
    rec = rung0.generate_one(ref, art, "m", 1, "key", max_tokens=100,
                             price_in=None, price_out=None, seed_base=None,
                             root=tmp_path)
    adir = rung0.attempt_dir("m", ref, 1, tmp_path)
    assert rec["generation_error"] == "HTTP 429: slow down"
    assert not (adir / "candidate.cpp").exists()
    assert rec["candidate"] == {}
    assert rec["usage"]["cost_usd"] is None


def test_cost_is_none_without_a_price_rather_than_zero():
    """A fabricated zero would read as a measured free run."""
    assert rung0.cost_usd({"prompt_tokens": 1e6}, None, None) is None
    assert rung0.cost_usd({}, 1.0, 2.0) is None
    assert rung0.cost_usd({"prompt_tokens": 1_000_000,
                           "completion_tokens": 0}, 1.25, 10.0) == 1.25


def test_a_fenced_answer_is_unfenced_and_flagged(task, tmp_path, monkeypatch):
    ref, art = task
    monkeypatch.setattr(rung0, "complete",
                        _stub('Here you go:\n```cpp\nint x = 1;\n```\n'))
    rec = rung0.generate_one(ref, art, "m", 1, "key", max_tokens=100,
                             price_in=None, price_out=None, seed_base=None,
                             root=tmp_path)
    src = (rung0.attempt_dir("m", ref, 1, tmp_path) / "candidate.cpp").read_text()
    assert src.strip() == "int x = 1;"
    assert rec["candidate"]["fenced"] is True


def test_records_are_written_atomically(task, tmp_path, monkeypatch):
    """No `.tmp` left behind, and the record parses -- a half-written record would
    be indistinguishable from a missing one on resume."""
    ref, art = task
    monkeypatch.setattr(rung0, "complete", _stub("int x;\n"))
    rung0.generate_one(ref, art, "m", 1, "key", max_tokens=100, price_in=None,
                       price_out=None, seed_base=None, root=tmp_path)
    leftovers = list(tmp_path.rglob("*.tmp"))
    assert leftovers == []
    for path in tmp_path.rglob("attempt.json"):
        json.loads(path.read_text())


def test_a_task_that_fails_to_build_is_not_charged_to_the_model(task, tmp_path,
                                                                monkeypatch):
    ref, _art = task
    broken = {"failed_stage": "emit", "error": "KeyError: nope\ntraceback..."}
    called = []
    monkeypatch.setattr(rung0, "complete",
                        lambda *a, **k: called.append(1) or _stub("x")(*a, **k))
    rec = rung0.generate_one(ref, broken, "m", 1, "key", max_tokens=100,
                             price_in=None, price_out=None, seed_base=None,
                             root=tmp_path)
    assert called == [], "must not spend an API call on an unbuildable task"
    assert "task build failed at stage emit" in rec["generation_error"]
    assert rec["grade"] == {}


def test_grade_refuses_to_judge_against_an_unverified_reference(tmp_path,
                                                               monkeypatch):
    """A T3 task's reference is unbuilt, so its attempts must be SKIPPED.

    Grading against a harness the reference never passed measures the harness.
    """
    tasks, _ = evalset.core_tasks()
    ref = next(t for t in tasks
               if not evalset.reference_status(t)["verified"])
    adir = rung0.attempt_dir("m", ref, 1, tmp_path)
    adir.mkdir(parents=True)
    rec = rung0._attempt_skeleton(ref, "m", 1)
    (adir / "attempt.json").write_text(json.dumps(rec))
    (adir / "candidate.cpp").write_text('extern "C" void candidate_kernel() {}\n')

    out = rung0.grade_one(ref, "m", 1, root=tmp_path)
    assert out["grade"]["graded"] is False
    assert out["grade"]["reference_verified"] is False
    assert "reference not usable" in out["grade"]["skip_reason"]


def test_a_compile_failure_with_no_output_is_not_recorded_as_a_verdict(
        task, tmp_path, monkeypatch, verified_reference):
    """The corruption that faked a finding: 126 of 145 T2 attempts were written as
    "did not compile" in one second when the toolchain was killed mid-pass, and the
    result looked like a compile rate falling from 100% at T1 to 13% at T2.

    `verify` always records the compiler's output or "compile timed out" on a real
    failure, so empty-and-failed means the compiler never ran. That must go back in
    the queue, not into the numbers.
    """
    ref, art = task
    adir = rung0.attempt_dir("m", ref, 1, tmp_path)
    adir.mkdir(parents=True)
    (adir / "attempt.json").write_text(
        json.dumps(rung0._attempt_skeleton(ref, "m", 1)))
    (adir / "candidate.cpp").write_text(f'extern "C" {art["signature"]} {{}}\n')
    monkeypatch.setattr(
        "hexkernels.forge.verify.verify",
        lambda *a, **k: {"compiled": False, "ran": False, "correct": False,
                         "mechanisms": {}, "error_text": "", "stdout": ""})

    out = rung0.grade_one(ref, "m", 1, root=tmp_path,
                          facts=rung0._facts_from_art(art))
    assert out["grade"]["graded"] is False
    assert "no compiler output" in out["grade"]["skip_reason"]
    assert "compiled" not in out["grade"], "must record no verdict at all"


def test_a_real_compile_failure_is_still_a_verdict(task, tmp_path, monkeypatch, verified_reference):
    """The other half: a failure WITH compiler output is a genuine result."""
    ref, art = task
    adir = rung0.attempt_dir("m", ref, 1, tmp_path)
    adir.mkdir(parents=True)
    (adir / "attempt.json").write_text(
        json.dumps(rung0._attempt_skeleton(ref, "m", 1)))
    (adir / "candidate.cpp").write_text(f'extern "C" {art["signature"]} {{}}\n')
    monkeypatch.setattr(
        "hexkernels.forge.verify.verify",
        lambda *a, **k: {"compiled": False, "ran": False, "correct": False,
                         "mechanisms": {}, "stdout": "",
                         "error_text": "error: use of undeclared identifier 'M_PI'"})

    out = rung0.grade_one(ref, "m", 1, root=tmp_path,
                          facts=rung0._facts_from_art(art))
    assert out["grade"]["graded"] is True
    assert out["grade"]["compiled"] is False
    assert "M_PI" in out["grade"]["error_text"]


def test_grade_skips_an_attempt_with_no_candidate(tmp_path):
    tasks, _ = evalset.core_tasks()
    ref = tasks[0]
    adir = rung0.attempt_dir("m", ref, 1, tmp_path)
    adir.mkdir(parents=True)
    (adir / "attempt.json").write_text(
        json.dumps(rung0._attempt_skeleton(ref, "m", 1)))
    out = rung0.grade_one(ref, "m", 1, root=tmp_path)
    assert out["grade"]["graded"] is False
    assert "no candidate" in out["grade"]["skip_reason"]


def test_grade_uses_a_supplied_artifact_and_never_traces_in_the_worker(
        task, tmp_path, monkeypatch, verified_reference):
    """The segfault regression: tracing with torch.fx off the main thread crashes.

    `grade --jobs 3` died with SIGSEGV because `grade_one` traced the module
    itself. The artifact is now built by `cmd_grade` up front and passed in, so
    this asserts the passed-in one is honoured -- `_task_artifacts` raising here
    stands in for the crash.
    """
    ref, art = task
    monkeypatch.setattr(rung0, "_task_artifacts",
                        lambda r: (_ for _ in ()).throw(
                            RuntimeError("traced in a worker thread")))
    adir = rung0.attempt_dir("m", ref, 1, tmp_path)
    adir.mkdir(parents=True)
    (adir / "attempt.json").write_text(
        json.dumps(rung0._attempt_skeleton(ref, "m", 1)))
    # A signature the lint gate blocks, so the simulator is never reached and the
    # test stays fast -- the artifact is still needed, for the lint rules.
    (adir / "candidate.cpp").write_text('extern "C" void wrong_name() {}\n')

    out = rung0.grade_one(ref, "m", 1, root=tmp_path,
                          facts=rung0._facts_from_art(art))
    assert out["grade"]["graded"] is True
    assert out["grade"]["compiled"] is False
    assert out["grade"]["lint"]["errors"], "expected the signature rule to block"


def test_lint_facts_round_trip_through_the_cache(task, tmp_path):
    """The cache must carry everything `lint` needs, and survive JSON.

    It exists because the reaper killed a grading run during the tracing phase, so
    the phase must not have to happen twice. If a fact were lost in the round trip
    the lint gate would silently weaken.
    """
    _ref, art = task
    facts = rung0._facts_from_art(art)
    assert set(facts) == set(rung0.LINT_FACT_KEYS)
    assert facts["signature"] and facts["tier"]
    rung0.save_lint_facts({"k": facts}, tmp_path)
    assert rung0.load_lint_facts(tmp_path)["k"] == facts


def test_load_lint_facts_tolerates_a_corrupt_cache(tmp_path):
    """A cache truncated by a kill must degrade to a re-trace, not a crash."""
    rung0.lint_facts_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    rung0.lint_facts_path(tmp_path).write_text("{not json")
    assert rung0.load_lint_facts(tmp_path) == {}


def test_concurrent_writers_to_one_path_do_not_collide(tmp_path):
    """The WinError 32 regression, which cost a 640-attempt run after 6 attempts.

    Five seeds of one task run in parallel and all five write that task's prompt
    file. A temp name derived only from the target put two threads in the same
    file, and `os.replace` then failed -- aborting the whole pool, not just the
    attempt.
    """
    import concurrent.futures

    target = tmp_path / "nested" / "prompt.md"
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: rung0._write_atomic(target, f"body {i}\n"),
                      range(40)))
    assert target.read_text().startswith("body ")
    assert list(tmp_path.rglob("*.tmp")) == []


def test_grade_refuses_to_run_more_than_one_simulator(capsys):
    """Standing instruction: one hexagon-sim at a time on Windows, never parallel.

    Refused rather than clamped, so a caller who asks for 6 learns they did not get
    6 instead of believing they did.
    """
    rc = rung0.main(["grade", "--model", "m", "--jobs", "4"])
    assert rc == 2
    assert "refused" in capsys.readouterr().err


def test_grade_defaults_to_one_job():
    ap_rc = rung0.main(["grade", "--model", "_nonexistent_model_", "--seeds", "1"])
    # Reaches the work (and finds nothing to do) rather than being refused, which is
    # only possible if the default --jobs is 1.
    assert ap_rc == 0


def test_key_file_beats_a_stale_environment_variable(tmp_path, monkeypatch):
    """Measured on this machine: the env var is a dead key and the file works."""
    monkeypatch.setenv("OPENAI_API_KEY", "stale")
    kf = tmp_path / "tok"
    kf.write_text("sk-good\n")
    assert rung0.load_key(str(kf)) == "sk-good"


def test_a_mechanism_scan_is_never_recorded_as_a_verdict(task, tmp_path,
                                                         monkeypatch):
    """The scan answers "did it use the accelerator", never "is it correct".

    They are different claims and cost three orders of magnitude apart (~0.4 s of
    compile-and-objdump against ~12 minutes of simulation), so the record keeps
    them in separate fields. A scan that leaked into `grade` would report an
    unexecuted kernel as having passed golden vectors.
    """
    ref, art = task
    adir = rung0.attempt_dir("m", ref, 1, tmp_path)
    adir.mkdir(parents=True)
    (adir / "attempt.json").write_text(
        json.dumps(rung0._attempt_skeleton(ref, "m", 1)))
    (adir / "candidate.cpp").write_text(f'extern "C" {art["signature"]} {{}}\n')
    monkeypatch.setattr(
        "hexkernels.forge.verify.mechanisms_of",
        lambda *a, **k: {"scanned": True, "compiled": True, "error_text": "",
                         "mechanisms": {"hvx": False, "hvx_compute": False,
                                        "hmx": False, "dma": False,
                                        "l2fetch": False, "vtcm": False}})

    out = rung0.sweep_one(ref, "m", 1, root=tmp_path)
    assert out["grade"] == {}, "a scan must not write a verdict"
    ms = out["mechanism_scan"]
    assert ms["scanned"] is True
    assert ms["mechanisms_fired"] == []
    # The one conclusion a scan can reach alone: genuine = correct AND fired, so
    # nothing firing makes it False whatever correctness turns out to be.
    assert ms["genuine_ruled_out"] is True
    assert ms["genuine_possible"] is False


def test_a_scan_that_fires_does_not_claim_genuine(task, tmp_path, monkeypatch):
    """The converse is not settled: firing says nothing about correctness."""
    ref, art = task
    adir = rung0.attempt_dir("m", ref, 1, tmp_path)
    adir.mkdir(parents=True)
    (adir / "attempt.json").write_text(
        json.dumps(rung0._attempt_skeleton(ref, "m", 1)))
    (adir / "candidate.cpp").write_text("Q6_Vw_vadd_VwVw\n")
    monkeypatch.setattr(
        "hexkernels.forge.verify.mechanisms_of",
        lambda *a, **k: {"scanned": True, "compiled": True, "error_text": "",
                         "mechanisms": {"hvx": True, "hvx_compute": True,
                                        "hmx": False, "dma": False,
                                        "l2fetch": False, "vtcm": False}})

    ms = rung0.sweep_one(ref, "m", 1, root=tmp_path)["mechanism_scan"]
    assert "hvx" in ms["mechanisms_fired"]
    assert ms["genuine_ruled_out"] is False
    assert ms["genuine_possible"] is True
    assert "genuine" not in ms, "a scan cannot conclude genuine on its own"


def test_an_unscannable_candidate_makes_no_mechanism_claim(task, tmp_path,
                                                           monkeypatch):
    """Fail closed: no object, no claims -- and never 'used nothing'.

    Crediting a kernel that failed to build with having honestly declined the
    accelerator would flatter exactly the failure this benchmark measures.
    """
    ref, art = task
    adir = rung0.attempt_dir("m", ref, 1, tmp_path)
    adir.mkdir(parents=True)
    (adir / "attempt.json").write_text(
        json.dumps(rung0._attempt_skeleton(ref, "m", 1)))
    (adir / "candidate.cpp").write_text("this is not c++\n")
    monkeypatch.setattr(
        "hexkernels.forge.verify.mechanisms_of",
        lambda *a, **k: {"scanned": False, "compiled": False,
                         "mechanisms": {}, "error_text": "error: expected ';'"})

    ms = rung0.sweep_one(ref, "m", 1, root=tmp_path)["mechanism_scan"]
    assert ms["scanned"] is False
    assert ms["mechanisms_fired"] == []
    assert ms["genuine_ruled_out"] is False, "unscannable is not 'used nothing'"
    assert "expected ';'" in ms["error_text"]


def test_a_simulator_timeout_is_not_recorded_as_a_wrong_answer(
        task, tmp_path, monkeypatch, verified_reference):
    """The measured corruption: 19 of 27 rung-0 failures were the 900 s simulator
    budget running out, written into the record as `correct: false`.

    A timeout is a budget exhausted, not a disagreement with the golden vectors --
    and the tasks it hit are exactly the slow tail (T2/T3 contractions whose median
    is ~12 min), so scoring them wrong biases correctness DOWNWARD. Same rule as the
    empty-compiler-output guard above: it goes back in the queue.
    """
    ref, art = task
    adir = rung0.attempt_dir("m", ref, 1, tmp_path)
    adir.mkdir(parents=True)
    (adir / "attempt.json").write_text(
        json.dumps(rung0._attempt_skeleton(ref, "m", 1)))
    (adir / "candidate.cpp").write_text(f'extern "C" {art["signature"]} {{}}\n')
    monkeypatch.setattr(
        "hexkernels.forge.verify.verify",
        lambda *a, **k: {"compiled": True, "ran": False, "correct": False,
                         "timed_out": True, "mechanisms": {}, "stdout": "",
                         "error_text": "simulator timed out after 900s"})

    out = rung0.grade_one(ref, "m", 1, root=tmp_path,
                          facts=rung0._facts_from_art(art))
    assert out["grade"]["graded"] is False
    assert "timed out" in out["grade"]["skip_reason"]
    assert "correct" not in out["grade"], "must record no verdict at all"


def test_a_wrong_answer_from_a_completed_run_is_still_a_verdict(
        task, tmp_path, monkeypatch, verified_reference):
    """The boundary the timeout guard must not cross. A simulator that RAN and
    disagreed with the golden vectors is the measurement this benchmark exists to
    take -- if the guard swallowed this too, correctness would only ever count
    passes.
    """
    ref, art = task
    adir = rung0.attempt_dir("m", ref, 1, tmp_path)
    adir.mkdir(parents=True)
    (adir / "attempt.json").write_text(
        json.dumps(rung0._attempt_skeleton(ref, "m", 1)))
    (adir / "candidate.cpp").write_text(f'extern "C" {art["signature"]} {{}}\n')
    monkeypatch.setattr(
        "hexkernels.forge.verify.verify",
        lambda *a, **k: {"compiled": True, "ran": True, "correct": False,
                         "mechanisms": {}, "stdout": "HVXENV_INCORRECT errors=7",
                         "error_text": "HVXENV_INCORRECT errors=7 n=1024"})

    out = rung0.grade_one(ref, "m", 1, root=tmp_path,
                          facts=rung0._facts_from_art(art))
    assert out["grade"]["graded"] is True
    assert out["grade"]["correct"] is False
    assert out["grade"]["skip_reason"] == ""


def test_verify_flags_a_simulator_timeout_rather_than_only_describing_it(
        monkeypatch, tmp_path):
    """`grade_one` must not have to parse prose to tell a timeout from a wrong
    answer. `verify` is the only place that knows which one happened, so it says so
    in a field -- string-matching "timed out" in `error_text` would break the moment
    the message is reworded.
    """
    import subprocess

    from hexkernels.forge import verify as _v

    def boom(cmd, env, timeout):
        if "hexagon-sim" in " ".join(cmd):
            raise subprocess.TimeoutExpired(cmd, timeout)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(_v, "_run", boom)
    monkeypatch.setattr(_v, "_disasm", lambda *a, **k: "")
    out = _v.verify('extern "C" void candidate_kernel() {}\n',
                    "int main(){return 0;}\n", name="t", timeout=7)
    assert out["timed_out"] is True
    assert out["ran"] is False


def _write_attempt(tmp_path, ref, grade):
    adir = rung0.attempt_dir("m", ref, 1, tmp_path)
    adir.mkdir(parents=True, exist_ok=True)
    rec = rung0._attempt_skeleton(ref, "m", 1)
    rec["response"] = {"text": "kept"}
    rec["grade"] = grade
    (adir / "attempt.json").write_text(json.dumps(rec, indent=1))
    return adir / "attempt.json"


def test_invalidate_timed_out_reclassifies_rather_than_blanking_the_reason(tmp_path):
    """Clearing these to `{}` would lose WHY they hold no verdict, and the summary
    could then only report them as an unexplained gap. A corrected experiment has to
    be able to say what it discarded, so the reason is written where
    `aggregate.summarise` already reads one.
    """
    tasks, _ = evalset.core_tasks()
    ref = tasks[0]
    path = _write_attempt(tmp_path, ref, {
        "graded": True, "compiled": True, "ran": False, "correct": False,
        "error_text": "simulator timed out after 900s", "graded_at": "2026-08-18"})

    rc = rung0.main(["invalidate", "--model", "m", "--seeds", "1",
                     "--timed-out", "--root", str(tmp_path),
                     "--only", ref.key])
    assert rc == 0

    got = json.loads(path.read_text())
    assert got["grade"]["graded"] is False
    assert "timed out" in got["grade"]["skip_reason"]
    assert "correct" not in got["grade"], "the void verdict must be gone"
    assert got["response"]["text"] == "kept", "generation is never touched"


def test_invalidate_timed_out_leaves_a_real_wrong_answer_alone(tmp_path):
    """The selector must not be able to quietly delete failures it did not cause."""
    tasks, _ = evalset.core_tasks()
    ref = tasks[0]
    path = _write_attempt(tmp_path, ref, {
        "graded": True, "compiled": True, "ran": True, "correct": False,
        "error_text": "HVXENV_INCORRECT errors=7", "graded_at": "2026-08-18"})

    rung0.main(["invalidate", "--model", "m", "--seeds", "1", "--timed-out",
                "--root", str(tmp_path), "--only", ref.key])

    got = json.loads(path.read_text())
    assert got["grade"]["graded"] is True
    assert got["grade"]["correct"] is False
