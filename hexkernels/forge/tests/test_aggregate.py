"""The aggregator's refusals matter more than its arithmetic.

Every number in the paper comes through `summarise`, so the cases asserted here are
the ones that would produce a plausible-looking wrong number: pooling two eval
sets, counting an ungraded attempt as a failure, or crediting a correct scalar
kernel as having engaged the accelerator.
"""
import pytest

from hexkernels.forge import aggregate, rung0


def _rec(*, key="b16i0_x", tier="T0", seed=1, graded=True, compiled=True,
         correct=True, fired=(), entitled=("hvx",), eval_set="core128", rung=0,
         skip_reason="", halluc=(), similarity=None):
    grade = {"graded": graded, "skip_reason": skip_reason}
    if graded:
        grade.update({"compiled": compiled, "correct": correct,
                      "mechanisms_fired": list(fired),
                      "genuine": bool(correct) and bool(fired),
                      "hallucinated_intrinsics": list(halluc),
                      "scalar_similarity": similarity})
    return {"schema_version": 1, "eval_set": eval_set, "rung": rung,
            "seed_index": seed,
            "task": {"key": key, "task_id": key, "name": key, "batch": 16,
                     "index": 0, "tier": tier, "dtype": "float16",
                     "shape": [32, 64], "entitled_mechanisms": list(entitled)},
            "model": {"provider": "openai", "requested": "m", "resolved": "m"},
            "usage": {}, "generation_error": None, "grade": grade}


def test_refuses_to_pool_two_eval_sets():
    with pytest.raises(ValueError, match="eval_set"):
        aggregate.summarise([_rec(eval_set="core128"),
                             _rec(eval_set="silicon64")])


def test_refuses_to_pool_two_rungs():
    with pytest.raises(ValueError, match="rung"):
        aggregate.summarise([_rec(rung=0), _rec(rung=1)])


def test_ungraded_attempts_are_excluded_not_counted_as_failures():
    """An unbuilt reference is the reference build's state, not the model's."""
    recs = [_rec(key="a", correct=True, fired=("hvx",)),
            _rec(key="b", graded=False, skip_reason="reference not usable: x")]
    s = aggregate.summarise(recs)
    assert s["graded"] == 1
    assert s["ungraded"] == 1
    assert s["pooled"]["correct_rate"] == 1.0     # 1/1 graded, not 1/2
    assert "reference not usable: x" in "".join(s["ungraded_reasons"])


def test_correct_but_scalar_is_not_genuine():
    """The whole thesis: correctness cannot detect accelerator use."""
    s = aggregate.summarise([_rec(correct=True, fired=())])
    assert s["pooled"]["correct_rate"] == 1.0
    assert s["pooled"]["genuine_rate"] == 0.0


def test_genuine_needs_both_halves():
    incorrect_but_fired = aggregate.summarise([_rec(correct=False, fired=("hvx",))])
    assert incorrect_but_fired["pooled"]["genuine_rate"] == 0.0


def test_mechanism_denominator_is_entitlement():
    """A task too small to justify DMA cannot fail to use DMA."""
    recs = [_rec(key="a", entitled=("hvx", "hmx"), fired=("hvx",)),
            _rec(key="b", entitled=("hvx",), fired=("hvx",))]
    s = aggregate.summarise(recs)
    assert s["per_mechanism"]["hvx"]["entitled_attempts"] == 2
    assert s["per_mechanism"]["hvx"]["fired_rate"] == 1.0
    assert s["per_mechanism"]["hmx"]["entitled_attempts"] == 1
    assert s["per_mechanism"]["hmx"]["fired_rate"] == 0.0
    assert "dma" not in s["per_mechanism"]


def test_seed_spread_is_reported():
    """PLAN.md section 2: a single-seed number is not a result."""
    recs = [_rec(key="a", seed=1, correct=True, fired=("hvx",)),
            _rec(key="a", seed=2, correct=False),
            _rec(key="b", seed=1, correct=True, fired=("hvx",)),
            _rec(key="b", seed=2, correct=True, fired=("hvx",))]
    s = aggregate.summarise(recs)
    assert s["per_seed"][1]["correct"] == 2
    assert s["per_seed"][2]["correct"] == 1
    assert s["seed_spread"]["correct"] == {"min": 1, "max": 2, "range": 1,
                                           "mean": 1.5}


def test_per_task_separates_ever_from_always():
    recs = [_rec(key="a", seed=1, correct=True, fired=("hvx",)),
            _rec(key="a", seed=2, correct=False),
            _rec(key="b", seed=1, correct=True, fired=("hvx",)),
            _rec(key="b", seed=2, correct=True, fired=("hvx",))]
    s = aggregate.summarise(recs)
    assert s["per_task"] == {"n": 2, "ever_correct": 2, "always_correct": 1,
                             "ever_genuine": 2}


def test_render_produces_markdown_without_crashing_on_empty_fields():
    s = aggregate.summarise([_rec(similarity=0.5, halluc=["Q6_made_up"])])
    text = aggregate.render(s, model="m")
    assert text.startswith("# Rung 0")
    assert "genuine" in text
    assert "Q6" not in text or "distinct" in text
    assert aggregate.render({"n_attempts": 0}) == "# No attempts\n"


def test_attempt_skeleton_carries_the_documented_interface():
    """Fields rungs 1-3 and the aggregator depend on must exist from the start."""
    tasks, _ = __import__("hexkernels.forge.evalset", fromlist=["x"]).core_tasks()
    rec = rung0._attempt_skeleton(tasks[0], "m", 1)
    for field in ("schema_version", "eval_set", "rung", "seed_index", "task",
                  "model", "prompt", "response", "candidate", "usage",
                  "turns_used", "generation_error", "grade"):
        assert field in rec, field
    for field in ("key", "task_id", "name", "batch", "index", "tier", "dtype",
                  "shape", "entitled_mechanisms"):
        assert field in rec["task"], field
    assert rec["eval_set"] == "core128"
    assert rec["rung"] == 0
    assert rec["turns_used"] == 1


def test_hvx_entitlement_maps_to_the_compute_flag_not_the_bare_flag():
    """Load-and-store HVX is the false positive the anti-cheat exists to remove."""
    assert rung0.MECHANISM_FLAG["hvx"] == "hvx_compute"
    for mech in ("hmx", "dma", "vtcm", "l2fetch"):
        assert rung0.MECHANISM_FLAG[mech] == mech


def _scan(*, scanned=True, fired=(), q6=False):
    return {"scanned": scanned, "compiled": scanned,
            "mechanisms": {}, "mechanisms_fired": list(fired),
            "genuine_ruled_out": scanned and not fired,
            "genuine_possible": bool(fired),
            "uses_any_q6_intrinsic": q6, "hallucinated_intrinsics": []}


def test_scan_covers_attempts_the_simulator_never_reached():
    """The point of the sweep: mechanism coverage without a correctness verdict."""
    recs = [_rec(key="a", graded=False, skip_reason="reference not usable"),
            _rec(key="b", graded=False, skip_reason="reference not usable")]
    for r in recs:
        r["mechanism_scan"] = _scan()
    s = aggregate.summarise(recs)
    assert s["graded"] == 0                      # nothing executed
    assert s["mechanism_scan"]["scanned"] == 2   # yet both are measured
    assert s["mechanism_scan"]["genuine_ruled_out"] == 2


def test_an_unscannable_attempt_is_not_counted_as_using_nothing():
    recs = [_rec(key="a", graded=False)]
    recs[0]["mechanism_scan"] = _scan(scanned=False)
    s = aggregate.summarise(recs)
    assert s["mechanism_scan"] == {} or s["mechanism_scan"]["scanned"] == 0


def test_firing_does_not_rule_genuine_in():
    recs = [_rec(key="a", graded=False)]
    recs[0]["mechanism_scan"] = _scan(fired=("hvx",))
    s = aggregate.summarise(recs)
    assert s["mechanism_scan"]["fired"] == 1
    assert s["mechanism_scan"]["genuine_ruled_out"] == 0


def test_scan_and_verdict_must_agree_where_both_exist():
    """Both read the same static detectors, so a disagreement is a bug worth seeing."""
    agree = _rec(key="a", correct=True, fired=("hvx",))
    agree["mechanism_scan"] = _scan(fired=("hvx",))
    clash = _rec(key="b", correct=True, fired=("hvx",))
    clash["mechanism_scan"] = _scan(fired=())
    s = aggregate.summarise([agree, clash])
    dis = s["scan_verdict_disagreements"]
    assert len(dis) == 1 and dis[0]["task"] == "b"
    assert dis[0]["graded"] == ["hvx"] and dis[0]["scanned"] == []


def test_autovectorisation_is_separated_from_model_written_intrinsics():
    """Who put the vector code there: the model, or hexagon-clang?

    The anti-cheat reads the binary, so it cannot tell. Rung 0 measured 3 attempts
    firing `hvx_compute` with zero `Q6_` names in their source -- the compiler
    vectorised a scalar loop. Reporting that as the model engaging the accelerator
    would credit the compiler's work to the model.
    """
    auto = _rec(key="a", graded=False)
    auto["mechanism_scan"] = _scan(fired=("hvx",), q6=False)
    asked = _rec(key="b", graded=False)
    asked["mechanism_scan"] = _scan(fired=("hvx",), q6=True)
    ms = aggregate.summarise([auto, asked])["mechanism_scan"]
    assert ms["fired"] == 2
    assert ms["fired_with_model_written_intrinsics"] == 1
    assert ms["fired_by_compiler_autovectorisation"] == 1
