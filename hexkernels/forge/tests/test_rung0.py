"""Rung 0 is the CONTROL, so what it withholds is as much a spec as what it says.

PLAN.md section 2 makes rung 0 "bare prompt, 1 shot", and the argument it carries is
that a model given no help does not reach the accelerator on its own. That argument
only holds if the prompt really is bare -- one leaked mechanism name and rung 0
becomes a second, weaker measurement of rung 2. These tests are the guard.

They run the real pipeline on one real task rather than a fixture, because the leak
this is defending against would arrive through `build()`'s own artifacts.
"""
import pytest

from hexkernels.forge import evalset, prompt, run_batch
from hexkernels.forge.frontend.emit import cname

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def rung0():
    """The bare prompt for the first eval-core task, plus its inputs."""
    tasks, _ = evalset.core_tasks()
    ref = tasks[0]
    spec = __import__("hexkernels.forge.mined", fromlist=["x"]).MINED_BATCHES[
        ref.batch][ref.index]
    art = run_batch.build(spec)
    assert art["failed_stage"] is None, art["error"]
    text = prompt.build_rung0(ref.name, art["signature"], art["graph"], ref.entry)
    return {"ref": ref, "art": art, "text": text}


def test_it_carries_what_makes_the_task_answerable(rung0):
    text, art, ref = rung0["text"], rung0["art"], rung0["ref"]
    assert art["signature"] in text
    assert 'extern "C"' in text
    assert ref.entry["op"] in text
    # Shapes: without them there is no loop bound. Checked as the rendered tuple
    # of the first input, which is what `_buffer_table` promises.
    first_in = next(n for n in art["graph"].inputs if n.shape)
    assert str(tuple(first_in.shape)) in text
    assert cname(first_in.name) in text


def test_it_withholds_the_scalar_reference(rung0):
    """The reference body is rung 1's help and rung 2's specification."""
    text, art = rung0["text"], rung0["art"]
    body = [ln.strip() for ln in art["kernel_c"].splitlines()
            if ln.strip().startswith("for (")]
    assert body, "expected the emitted reference to contain loops"
    for line in body:
        assert line not in text
    assert "candidate_kernel(const" not in text.replace(
        art["signature"], "")     # only the signature, never the definition


def test_it_withholds_the_schedule_and_the_linalg(rung0):
    text = rung0["text"]
    for leak in ("iterator_types", "vectorizable_loop", "linalg.", "#map",
                 "affine_map", "parallel", "reduction"):
        assert leak not in text, leak


def test_it_withholds_the_mechanism_budget_and_the_hardware_facts(rung0):
    text = rung0["text"]
    for leak in ("Mechanisms this size justifies", "Working set",
                 "L1 data cache", "L2 cache", "HVX vector width", "VTCM",
                 "scratchpad", "tier"):
        assert leak not in text, leak


def test_it_names_no_mechanism_and_no_intrinsic(rung0):
    """The sharpest leak: naming an accelerator, or a header that names one.

    `CONTRACT` enumerates `<hvx_hexagon_protos.h>` and `<hmx_hexagon_protos.h>`,
    which would tell the model both mechanisms the benchmark is about. Rung 0 says
    only that the SDK's headers are available.
    """
    lowered = rung0["text"].lower()
    for mech in ("hvx", "hmx", "vtcm", "l2fetch", "dma", "intrinsic"):
        assert mech not in lowered, mech
    assert "Q6_" not in rung0["text"]
    assert "V6_" not in rung0["text"]


def test_it_states_the_fixed_non_tensor_arguments():
    """A synth row's chosen values decide the golden, so they must be stated.

    Without them the prompt asks for one function and grades another -- and the
    failure would read as a wrong kernel rather than an underspecified task.
    """
    from hexkernels.forge import mined

    tasks, _ = evalset.core_tasks()
    ref = next((t for t in tasks if t.entry.get("synth_plan")
                and any(k == "v" for k, _v in t.entry["synth_plan"])), None)
    if ref is None:
        pytest.skip("no eval-core task fixes a non-tensor argument")
    art = run_batch.build(mined.MINED_BATCHES[ref.batch][ref.index])
    assert art["failed_stage"] is None, art["error"]
    text = prompt.build_rung0(ref.name, art["signature"], art["graph"], ref.entry)
    for kind, value in ref.entry["synth_plan"]:
        if kind != "t":
            assert repr(value) in text, (ref.key, value)


def test_it_is_deterministic(rung0):
    """Pure: the same task must give a byte-identical prompt every time.

    The prompt's hash is recorded with every attempt, so a prompt that drifted
    would silently split one condition into two.
    """
    ref, art = rung0["ref"], rung0["art"]
    again = prompt.build_rung0(ref.name, art["signature"], art["graph"],
                               ref.entry)
    assert again == rung0["text"]


def test_it_is_much_shorter_than_the_rung2_prompt(rung0):
    """A sanity check on the whole premise: bare should be visibly smaller."""
    art = rung0["art"]
    assert len(rung0["text"]) < len(art["prompt"]) / 2
