"""Rung 1's retry turns may carry the toolchain's complaint and nothing of ours.

The boundary, decided 2026-08-20: a turn tells the model what the COMPILER or the
HARNESS said about the model's own code. It never adds knowledge of the hardware --
no intrinsic names (`prompt.retry_facts`, which rung 2's `model_client.feedback`
uses), and not `verify.diagnose`'s inference about which elements are wrong.

That distinction matters in a way worth stating: echoing a compiler diagnostic about
the model's OWN code leaks nothing, because the model wrote the code that provoked it.
`diagnose` is different -- it is this repository reasoning about vector-shaped failure
modes, and handing it over would start describing the mechanism.

The trap this pins: `verify` already APPENDS `diagnose`'s output to `error_text`
("\n  likely cause: ..."), so a feedback builder that passed `error_text` through
verbatim would ship rung-2 reasoning while looking correct.
"""
import json

import pytest

from hexkernels.forge import rung1


def _grade(**kw):
    g = {"graded": True, "compiled": True, "ran": True, "correct": False,
         "error_text": "", "stdout": "", "skip_reason": "", "lint": {"errors": []}}
    g.update(kw)
    return g


def test_a_compile_failure_hands_back_the_compilers_own_words():
    fb = rung1.feedback(_grade(
        compiled=False, ran=False,
        error_text="kernel.cpp:7:3: error: use of undeclared identifier 'M_PI'"))
    assert "did not compile" in fb
    assert "use of undeclared identifier 'M_PI'" in fb
    assert "translation unit" in fb, "must say what shape the answer takes"


def test_a_wrong_answer_hands_back_the_harness_line():
    fb = rung1.feedback(_grade(
        error_text="HVXENV_INCORRECT errors=7 n=1024 first=[3] got=1.5 want=2.0"))
    assert "wrong" in fb.lower()
    assert "errors=7" in fb
    assert "first=[3]" in fb


def test_it_strips_diagnose_reasoning_from_the_error_text():
    """The trap. `verify` glues its own inference onto `error_text`; rung 1 must cut
    it off at the seam rather than trust the field to be clean."""
    fb = rung1.feedback(_grade(
        error_text="HVXENV_INCORRECT errors=7 n=1024\n"
                   "  likely cause: tail elements untouched; lane order reversed"))
    assert "errors=7" in fb
    assert "likely cause" not in fb
    assert "lane order" not in fb


def test_it_never_adds_isa_facts_of_its_own():
    """Whatever the verdict says, the feedback must not carry `retry_facts`."""
    from hexkernels.forge import prompt

    isa = prompt.retry_facts()
    for g in (_grade(compiled=False, ran=False, error_text="error: boom"),
              _grade(error_text="HVXENV_INCORRECT errors=1 n=8")):
        fb = rung1.feedback(g)
        assert isa not in fb
        # A sample of what ISA_FACTS names must not appear from OUR side.
        assert "Q6_" not in fb


def test_it_preserves_a_compiler_diagnostic_that_names_an_intrinsic():
    """NOT redacted, deliberately. If the model reached for an intrinsic and the
    compiler rejected it, that text is about the model's own code -- withholding it
    would make the turn useless and would hide the model's own attempt from it."""
    fb = rung1.feedback(_grade(
        compiled=False, ran=False,
        error_text="error: use of undeclared identifier 'Q6_Vsf_vadd_VsfVsf'"))
    assert "Q6_Vsf_vadd_VsfVsf" in fb


def test_a_lint_rejection_is_reported_as_the_correctness_problem_it_is():
    fb = rung1.feedback(_grade(
        compiled=False, ran=False,
        error_text="entry point must be exactly: extern \"C\" void "
                   "candidate_kernel(const float *, float *)",
        lint={"errors": ["entry point mismatch"]}))
    assert "entry point" in fb


# ------------------------------------------------------------- which attempts retry

def _rec(grade, turns_used=1):
    return {"rung": 1, "turns_used": turns_used, "grade": grade,
            "candidate": {"bytes": 10}}


def test_only_a_graded_incorrect_attempt_takes_another_turn():
    assert rung1.needs_another_turn(_rec(_grade(correct=False)), max_turns=5)


def test_a_correct_attempt_is_finished():
    assert not rung1.needs_another_turn(_rec(_grade(correct=True)), max_turns=5)


def test_an_attempt_with_no_verdict_does_not_take_a_turn():
    """The 70 attempts whose reference never passed stage (g) are ungraded, not
    failing. Retrying them would spend API money on a task nothing can grade, and
    would look like a model failure in the turn histogram."""
    ungraded = {"graded": False, "skip_reason": "reference not usable", }
    assert not rung1.needs_another_turn(_rec(ungraded), max_turns=5)


def test_the_turn_budget_is_respected():
    assert not rung1.needs_another_turn(_rec(_grade(correct=False), turns_used=5),
                                        max_turns=5)
    assert rung1.needs_another_turn(_rec(_grade(correct=False), turns_used=4),
                                    max_turns=5)


def test_a_followup_shows_the_model_its_own_kernel_and_the_complaint():
    first = "Write a kernel named `k` ...\n"
    prior = 'extern "C" void candidate_kernel(const float *a, float *b) {}\n'
    out = rung1.followup_prompt(first, prior, _grade(
        error_text="HVXENV_INCORRECT errors=7 n=1024"))
    assert first.strip() in out, "the original ask is restated, not assumed"
    assert prior.strip() in out, "the model must see what it actually wrote"
    assert "errors=7" in out
    for mech in ("hvx", "hmx", "vtcm", "l2fetch", "dma"):
        assert mech not in out.lower(), mech


def test_the_harness_verdict_token_is_neutralised_but_its_numbers_survive():
    """`HVXENV_INCORRECT` contains "HVX". This is the leak that hides in plain sight:
    it arrives on every wrong-answer turn, in our own words, at the moment the model is
    choosing what to try next -- and `test_rung0.py` bans "hvx" from a prompt for
    exactly that reason, so the feedback channel must not readmit it.

    Neutralised rather than dropped, because the numbers are the whole point of a
    correctness turn.
    """
    fb = rung1.feedback(_grade(
        error_text="HVXENV_INCORRECT errors=7 n=1024 first=[3] got=1.5 want=2.0"))
    assert "HVX" not in fb and "hvx" not in fb.lower()
    assert "CHECK_INCORRECT" in fb, "recognisable, just not named after the hardware"
    for kept in ("errors=7", "n=1024", "first=[3]", "got=1.5", "want=2.0"):
        assert kept in fb, kept
