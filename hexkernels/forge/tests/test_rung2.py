"""Rung 2 hands over the whole apparatus, so this file asserts PRESENCE.

`test_rung0.py` and `test_rung1.py` are guards against leaks; at rung 2 the
scaffolding IS the treatment, so the failure mode inverts: a rung-2 prompt missing
its hardware facts, its budget or its schedule would be a THIRD measurement of rung
1 reported under a rung-2 heading, and nothing downstream could detect it.

One thing is still withheld, and it is the only thing: INTRINSIC NAMES. `prompt.py`
cites `test_prompt_does_not_hand_over_the_answer` as the guard on that, and PLAN.md
section 5.A left the old suites behind -- the name occurs nowhere in this repository
but in two comments. So the guard is written here, and it covers `build()` too,
because `run_batch` sends that prompt to build the corpus.
"""
import pytest

from hexkernels.forge import evalset, prompt, run_batch

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def rung2():
    tasks, _ = evalset.core_tasks()
    ref = tasks[0]
    spec = __import__("hexkernels.forge.mined", fromlist=["x"]).MINED_BATCHES[
        ref.batch][ref.index]
    art = run_batch.build(spec)
    assert art["failed_stage"] is None, art["error"]
    text = prompt.build_rung2(ref.name, art["signature"], art["graph"], ref.entry,
                              art["kernel_c"], art["plan"],
                              linalg_ir=art["linalg_ir"])
    return {"ref": ref, "art": art, "text": text}


# ------------------------------------------------- everything rungs 0 and 1 carried

def _norm(text: str) -> str:
    """Collapse whitespace so a line-wrapping difference cannot fail a match."""
    return " ".join(text.split())


def test_rung0_contract_and_rung2_buffer_guarantee_say_the_same_thing():
    """`RUNG0_CONTRACT` states the layout/aliasing licence inline; rung 2 carries
    it as `prompt.BUFFER_GUARANTEE` instead, because `CONTRACT` has no equivalent
    of its own. This is what keeps the two copies from drifting apart -- the fact
    a model needs to know a contiguous vector store is legal must say the same
    thing at both rungs, word-for-word once wrapping is normalised."""
    assert _norm(prompt.BUFFER_GUARANTEE) in _norm(prompt.RUNG0_CONTRACT)


def test_it_is_a_superset_of_rung1(rung2):
    """Monotonicity. `prompt.build()` omits the operator identity and the buffer
    table because a corpus author already knows them; used as-is, rung 2 would
    WITHHOLD what rung 1 gave, at the one rung the argument depends on.

    A SPOT-CHECK VERSION OF THIS TEST FAILED TO CATCH A REAL DROPPED GUARANTEE.
    The previous body checked the signature, `extern "C"`, one input's shape and
    name, the reference body, and a bare length comparison -- and none of those
    would have noticed that rung 2's `CONTRACT` silently withheld the "every
    buffer is dense and row-major ... no two of them overlap" licence that
    `RUNG0_CONTRACT` states, because that sentence is not any of the things the
    old checks looked for. So this asserts every semantic part rung 1 carries is
    present in rung 2's text, explicitly, one at a time.
    """
    ref, art = rung2["ref"], rung2["art"]
    r1 = prompt.build_rung1(ref.name, art["signature"], art["graph"], ref.entry,
                            art["kernel_c"])
    text = rung2["text"]
    norm_text = _norm(text)

    # The signature and the "extern C" requirement it must be wrapped in.
    assert art["signature"] in text
    assert 'extern "C"' in text

    # The operator identity -- what rung 1 asks the kernel to compute.
    assert _norm(prompt.task_statement(ref.entry)) in norm_text

    # The full buffer table, not just one input's shape and name.
    assert _norm(prompt._buffer_table(art["graph"])) in norm_text

    # The layout and aliasing licence rung 1 grants (see the test above).
    assert _norm(prompt.BUFFER_GUARANTEE) in norm_text

    # The complete reference body, verbatim.
    assert art["kernel_c"].strip() in text

    # Cheap backstop: a superset of non-trivial added content cannot be shorter.
    assert len(r1) < len(text)


# ------------------------------------------------------------- what rung 2 adds

def test_it_states_the_hardware_facts(rung2):
    from hexkernels.core import target
    t = target.current()
    text = rung2["text"]
    assert str(t.hvx_bytes) in text
    assert str(t.l2_bytes) in text
    assert str(t.vtcm_bytes) in text
    assert f"{t.vtcm_base:#010x}" in text


def test_it_states_the_mechanism_budget(rung2):
    text, art = rung2["text"], rung2["art"]
    assert art["tier"] in text
    assert str(art["plan"].working_set_bytes) in text
    for mech in art["mechanisms"]:
        assert mech in text, mech


def test_it_states_the_schedule(rung2):
    """`build_rung2`'s own explanatory prose hardcodes the word
    "vectorizable_loop", so a bare substring check on that word passes whether or
    not the schedule section is actually present. Assert the real rendered
    section instead."""
    text, art = rung2["text"], rung2["art"]
    assert prompt._schedule_section(art["graph"]) in text
    assert "iterator_types" in text


def test_it_carries_the_linalg_when_there_is_any(rung2):
    text, art = rung2["text"], rung2["art"]
    if art["linalg_ir"]:
        assert art["linalg_ir"].rstrip() in text
    else:
        pytest.skip(f"no linalg for this task: {art['linalg_error']}")


def test_it_enumerates_the_accelerator_headers(rung2):
    """Rung 0 deliberately did not: naming the two headers names the two
    mechanisms. At rung 2 that is the point."""
    text = rung2["text"]
    assert "hvx_hexagon_protos.h" in text
    assert "hmx_hexagon_protos.h" in text


def test_it_permits_deriving_from_the_reference(rung2):
    assert "Derive this kernel from the reference" in rung2["text"]


# ------------------------------- the one thing still withheld, guarded for real

def _first_prompts(art, ref):
    """Every prompt this project sends as a FIRST ask, at any rung."""
    return {
        "rung0": prompt.build_rung0(ref.name, art["signature"], art["graph"],
                                    ref.entry),
        "rung1": prompt.build_rung1(ref.name, art["signature"], art["graph"],
                                    ref.entry, art["kernel_c"]),
        "rung2": prompt.build_rung2(ref.name, art["signature"], art["graph"],
                                    ref.entry, art["kernel_c"], art["plan"],
                                    linalg_ir=art["linalg_ir"]),
        "batch_forge": prompt.build(ref.name, art["signature"], art["kernel_c"],
                                    art["graph"], art["plan"],
                                    linalg_ir=art["linalg_ir"]),
    }


def test_the_first_prompt_names_no_intrinsic(rung2):
    """The guard `prompt.retry_facts` claims exists. It did not.

    `retry_facts`' own docstring says adding ISA_FACTS to `build()` "fails
    `test_prompt_does_not_hand_over_the_answer`" -- a test that was never ported.
    Without it nothing stops an intrinsic name reaching a first ask, and the
    mechanism headline at every rung becomes unfalsifiable: a model told
    `Q6_Vsf_vmax_VsfVsf` exists is being tested on retrieval.
    """
    for label, text in _first_prompts(rung2["art"], rung2["ref"]).items():
        for token in ("Q6_", "V6_"):
            assert token not in text, f"{label} names an intrinsic ({token})"


def test_isa_facts_reach_no_first_prompt(rung2):
    """The same rule stated on the payload rather than on the token, because
    ISA_FACTS could be reworded and still be the leak."""
    probe = "MAGIC-CONSTANT ROUNDING DOES NOT WORK"
    assert probe in prompt.ISA_FACTS, "probe drifted; pick another ISA_FACTS line"
    for label, text in _first_prompts(rung2["art"], rung2["ref"]).items():
        assert probe not in text, label


def test_it_is_deterministic(rung2):
    ref, art = rung2["ref"], rung2["art"]
    again = prompt.build_rung2(ref.name, art["signature"], art["graph"], ref.entry,
                               art["kernel_c"], art["plan"],
                               linalg_ir=art["linalg_ir"])
    assert again == rung2["text"]


# ------------------------------------------------------- the driver's rung switch

def test_the_driver_builds_the_rung2_prompt(rung2):
    from hexkernels.forge import rung0
    ref, art = rung2["ref"], rung2["art"]
    assert rung0.first_prompt(2, ref, art) == rung2["text"]


def test_the_driver_still_refuses_a_rung_it_has_no_prompt_for(rung2):
    from hexkernels.forge import rung0
    with pytest.raises(ValueError) as exc:
        rung0.first_prompt(3, rung2["ref"], rung2["art"])
    assert "rung 3" in str(exc.value)
