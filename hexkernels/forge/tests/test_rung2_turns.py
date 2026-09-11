"""Rung 2's retry turns say what Forge says, and the boundary is a SEAM.

`verify` glues `diagnose`'s inference onto `error_text` at `rung1.DIAGNOSE_SEAM`.
Rung 1 cuts there and throws the second half away; rung 2 keeps both halves and adds
`prompt.retry_facts`. Same seam, opposite side -- that is the entire difference
between the two rungs' retry channels, so it is pinned here rather than left to
inspection.

Rung 2 does NOT get its own feedback prose. It adapts the stored grade record into
the verdict dict `model_client.feedback` already consumes, so "rung 2 is Forge" is
true of the words and not just of the pipeline.
"""
import pytest

from hexkernels.forge import prompt, rung1, rung2


def _grade(**kw):
    g = {"graded": True, "compiled": True, "ran": True, "correct": False,
         "error_text": "", "stdout": "", "skip_reason": "", "rd_leak": [],
         "lint": {"errors": [], "warnings": []}}
    g.update(kw)
    return g


# ------------------------------------------------------------------- the adapter

def test_it_splits_error_text_at_the_seam():
    g = _grade(error_text="CHECK errors=3 first=[7]"
                          + rung1.DIAGNOSE_SEAM
                          + " a widening deal left lanes interleaved")
    v = rung2.verdict_from_grade(g)
    assert v["error_text"] == "CHECK errors=3 first=[7]"
    assert v["diagnosis"] == ["a widening deal left lanes interleaved"]


def test_error_text_with_no_seam_yields_no_diagnosis():
    v = rung2.verdict_from_grade(_grade(error_text="plain compiler noise"))
    assert v["error_text"] == "plain compiler noise"
    assert v["diagnosis"] == []


def test_it_lifts_lint_errors_out_of_their_nesting():
    g = _grade(compiled=False, lint={"errors": ["fabricated: Q6_V_ld"],
                                     "warnings": ["w"]})
    assert rung2.verdict_from_grade(g)["lint_errors"] == ["fabricated: Q6_V_ld"]


def test_the_harness_line_survives_when_only_error_text_has_it():
    """`model_client.feedback` renders a wrong answer from `stdout`, and `verify`
    does not guarantee both fields carry the same failure. A turn that says "your
    numbers are wrong" and then shows nothing is the one thing a retry may not do."""
    v = rung2.verdict_from_grade(_grade(error_text="CHECK errors=7 first=[3]",
                                        stdout=""))
    assert v["stdout"] == "CHECK errors=7 first=[3]"


# ------------------------------------------------------ what a turn actually says

def test_a_wrong_answer_gets_the_diagnosis_and_the_isa_facts():
    """The whole point of the rung. Rung 1 gets neither."""
    g = _grade(error_text="CHECK errors=3 first=[7]" + rung1.DIAGNOSE_SEAM
               + " a widening deal left lanes interleaved",
               stdout="CHECK_INCORRECT errors=3")
    fb = rung2.feedback(g)
    assert "WRONG VALUES" in fb
    assert "a widening deal left lanes interleaved" in fb
    assert "MAGIC-CONSTANT ROUNDING DOES NOT WORK" in fb, "ISA_FACTS must be here"
    assert prompt.ISA_FACTS in fb


def test_a_lint_rejection_says_it_never_compiled_and_carries_the_facts():
    fb = rung2.feedback(_grade(compiled=False,
                               lint={"errors": ["fabricated intrinsic: Q6_V_ld"]}))
    assert "REJECTED WITHOUT COMPILING" in fb
    assert "Q6_V_ld" in fb
    assert prompt.ISA_FACTS in fb


def test_a_compile_failure_hands_back_the_compilers_own_words():
    fb = rung2.feedback(_grade(
        compiled=False, ran=False,
        error_text="kernel.cpp:7:3: error: no matching function 'Q6_V_st'"))
    assert "did not compile" in fb
    assert "no matching function 'Q6_V_st'" in fb


def test_a_provenance_leak_outranks_everything_else():
    fb = rung2.feedback(_grade(compiled=False, rd_leak=["hmx_helpers.h"],
                               lint={"errors": ["something else"]}))
    assert "REJECTED WITHOUT COMPILING" in fb
    assert "hmx_helpers.h" in fb
    assert "something else" not in fb


def test_a_correct_attempt_gets_nothing():
    assert rung2.feedback(_grade(correct=True)) == ""


def test_the_harness_token_is_not_neutralised_at_this_rung():
    """Rung 1 redacts HVXENV because "HVX" is a leak there. At rung 2 the mechanism
    budget names hvx in the FIRST prompt, so redacting it in the feedback while
    printing it in the prompt would be theatre -- and would corrupt the toolchain's
    own text for nothing."""
    fb = rung2.feedback(_grade(error_text="HVXENV_INCORRECT errors=7 first=[3]"))
    assert "HVXENV_INCORRECT" in fb
    assert "errors=7" in fb


# ------------------------------------------------------------- the taxonomy field

@pytest.mark.parametrize("grade,kind", [
    (_grade(rd_leak=["x"], compiled=False), "rd_leak"),
    (_grade(compiled=False, lint={"errors": ["e"]}), "lint"),
    (_grade(compiled=False), "compile"),
    (_grade(ran=False), "ran"),
    (_grade(), "wrong"),
    (_grade(correct=True), ""),
])
def test_every_turn_is_classified(grade, kind):
    """PLAN.md section 5.B wants a per-stage failure taxonomy, and rung 2 is the
    first rung with more than one stage in play. Entitlement rejections land in
    `lint`, which is why they can be separated from correctness failures in the
    report -- otherwise "rung 2 needed more turns" reads as a capability statement
    when part of it is a policy statement."""
    assert rung2.feedback_kind(grade) == kind


# --------------------------------------------------------------- the turn budget

def test_only_a_graded_incorrect_attempt_takes_another_turn():
    assert rung2.needs_another_turn({"grade": _grade(), "turns_used": 1})


def test_a_correct_attempt_is_finished():
    assert not rung2.needs_another_turn({"grade": _grade(correct=True)})


def test_an_attempt_with_no_verdict_does_not_take_a_turn():
    """The 70 attempts whose reference never passed stage (g) carry graded: false.
    Retrying them would spend money on something nothing can grade."""
    assert not rung2.needs_another_turn(
        {"grade": {"graded": False, "skip_reason": "reference not usable"}})


def test_the_turn_budget_is_respected():
    assert not rung2.needs_another_turn({"grade": _grade(), "turns_used": 5})
    assert rung2.needs_another_turn({"grade": _grade(), "turns_used": 4})


def test_a_followup_shows_the_model_its_own_kernel_and_the_complaint():
    out = rung2.followup_prompt("ORIGINAL ASK", "void k() {}",
                                _grade(compiled=False, error_text="boom"))
    assert "ORIGINAL ASK" in out
    assert "void k() {}" in out
    assert "boom" in out
    assert "translation unit" in out


# ---------------------------------------------------------------- the turn record

def test_a_turn_record_keeps_the_verdict_that_provoked_it():
    rec = {"turns_used": 1}
    grade_before = _grade(compiled=False, error_text="boom")
    out = rung2._turn_record(rec, 2, "PROMPT", {"usage": {"prompt_tokens": 11,
                                                          "completion_tokens": 3},
                                                 "finish_reason": "stop"},
                             "void k() {}", grade_before)
    turn = out["turns"][-1]
    assert turn["turn"] == 2
    assert turn["feedback_kind"] == "compile"
    assert "boom" in turn["feedback_given"]
    assert turn["verdict_before"]["compiled"] is False
    assert turn["usage"]["prompt_tokens"] == 11
    assert out["turns_used"] == 2
    assert out["grade"] == {}, "the stale verdict must be dropped, not updated"


def test_a_turn_record_folds_its_usage_into_the_top_level_total():
    """Rung 1's defect 2: per-turn usage lived only inside turns[], so anything
    summing `usage` under-reported the rung. Harmless at 12 turns, material here."""
    rec = {"turns_used": 1, "usage": {"prompt_tokens": 100,
                                      "completion_tokens": 40,
                                      "prompt_tokens_details": {"cached_tokens": 50}}}
    out = rung2._turn_record(rec, 2, "P",
                             {"usage": {"prompt_tokens": 7, "completion_tokens": 5,
                                        "prompt_tokens_details":
                                            {"cached_tokens": 4}}},
                             "void k() {}", _grade())
    assert out["usage"]["prompt_tokens"] == 107
    assert out["usage"]["completion_tokens"] == 45
    assert out["usage"]["prompt_tokens_details"]["cached_tokens"] == 54
    assert out["usage"]["turns_counted"] == 2


def test_a_turn_record_folds_every_nested_usage_detail_not_just_one():
    """`_add_usage` used to special-case exactly one nested dict,
    `prompt_tokens_details`, and silently dropped every other one. Real records
    also carry `completion_tokens_details.reasoning_tokens` -- e.g. 1024 against a
    `completion_tokens` of 1224 -- so a folded multi-turn record kept turn 1's
    reasoning count beside an N-turn completion total. Both nested details must
    accumulate across turns now."""
    rec = {"turns_used": 1,
          "usage": {"prompt_tokens": 100, "completion_tokens": 1224,
                    "prompt_tokens_details": {"cached_tokens": 50},
                    "completion_tokens_details": {"reasoning_tokens": 1024}}}
    out = rung2._turn_record(rec, 2, "P",
                             {"usage": {"prompt_tokens": 7, "completion_tokens": 610,
                                        "prompt_tokens_details":
                                            {"cached_tokens": 4},
                                        "completion_tokens_details":
                                            {"reasoning_tokens": 512}}},
                             "void k() {}", _grade())
    assert out["usage"]["prompt_tokens_details"]["cached_tokens"] == 54
    assert out["usage"]["completion_tokens_details"]["reasoning_tokens"] == 1536


def test_a_turn_record_reprices_the_folded_total_not_just_turn_one():
    """The other half of rung 1's defect 2. `generate_one` prices turn 1 and writes
    `price_per_1m` beside it; `_turn_record` folds turn 2's tokens into the same
    `usage` dict, and without a reprice the record would carry two turns of tokens
    against one turn of dollars while `aggregate` sums exactly this field."""
    rec = {"turns_used": 1,
          "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
                    "cost_usd": 5.0,
                    "price_per_1m": {"in": 1.0, "out": 4.0}}}
    out = rung2._turn_record(rec, 2, "P",
                             {"usage": {"prompt_tokens": 1_000_000,
                                        "completion_tokens": 1_000_000}},
                             "void k() {}", _grade())
    assert out["usage"]["prompt_tokens"] == 2_000_000
    assert out["usage"]["cost_usd"] == 10.0, \
        "both turns priced, not just the first"


def test_a_turn_record_with_no_recorded_rates_stays_unpriced():
    """No `price_per_1m` on the record means the attempt was never priced. A
    reprice must not invent a rate to fill the gap."""
    rec = {"turns_used": 1, "usage": {"prompt_tokens": 100, "completion_tokens": 40}}
    out = rung2._turn_record(rec, 2, "P",
                             {"usage": {"prompt_tokens": 7, "completion_tokens": 5}},
                             "void k() {}", _grade())
    assert out["usage"].get("cost_usd") is None


def test_a_turn_record_honours_the_cached_rate_across_turns():
    """The cached rate recorded at turn 1 must still apply once turn 2's tokens --
    cached or not -- are folded in and the total is repriced."""
    rec = {"turns_used": 1,
          "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0,
                    "prompt_tokens_details": {"cached_tokens": 500_000},
                    "cost_usd": 0.55,
                    "price_per_1m": {"in": 1.0, "out": 4.0, "cached_in": 0.1}}}
    out = rung2._turn_record(rec, 2, "P",
                             {"usage": {"prompt_tokens": 1_000_000,
                                        "completion_tokens": 0,
                                        "prompt_tokens_details":
                                            {"cached_tokens": 500_000}}},
                             "void k() {}", _grade())
    assert out["usage"]["prompt_tokens"] == 2_000_000
    assert out["usage"]["prompt_tokens_details"]["cached_tokens"] == 1_000_000
    assert out["usage"]["cost_usd"] == 1.1, "1M at $1 + 1M at $0.10, both turns"
