"""Rung 1 adds the reference and NOTHING ELSE, so both halves are a spec.

PLAN.md section 2 makes rung 1 "+ PyTorch reference + scalar code, up to 5 turns",
asking "does help with *correctness* produce *mechanism*?" That question only has an
answer if the help really is confined to correctness: one leaked hardware fact,
schedule, or intrinsic name and rung 1 becomes rung 2, collapsing the gap PLAN.md
section 6 calls the spine of the paper.

So these tests are the mirror image of `test_rung0.py`. The scalar reference, which
rung 0 must NOT contain, rung 1 MUST contain -- and everything rung 0 withholds,
rung 1 still withholds.

They run the real pipeline on a real task, because the leak being defended against
would arrive through `build()`'s own artifacts -- including through the emitted
reference body itself, which is new text rung 0 never had to vet.
"""
import pytest

from hexkernels.forge import evalset, prompt, run_batch
from hexkernels.forge.frontend.emit import cname

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def rung1():
    tasks, _ = evalset.core_tasks()
    ref = tasks[0]
    spec = __import__("hexkernels.forge.mined", fromlist=["x"]).MINED_BATCHES[
        ref.batch][ref.index]
    art = run_batch.build(spec)
    assert art["failed_stage"] is None, art["error"]
    text = prompt.build_rung1(ref.name, art["signature"], art["graph"], ref.entry,
                              art["kernel_c"])
    return {"ref": ref, "art": art, "text": text}


def test_it_carries_everything_rung0_carried(rung1):
    """Rung 1 is rung 0 PLUS the reference, not a different ask."""
    text, art, ref = rung1["text"], rung1["art"], rung1["ref"]
    assert art["signature"] in text
    assert 'extern "C"' in text
    assert ref.entry["op"] in text
    first_in = next(n for n in art["graph"].inputs if n.shape)
    assert str(tuple(first_in.shape)) in text
    assert cname(first_in.name) in text


def test_it_hands_over_the_scalar_reference(rung1):
    """The one thing rung 1 adds. Asserted on the reference's own loop bodies, the
    same lines `test_rung0.py` asserts are ABSENT there."""
    text, art = rung1["text"], rung1["art"]
    body = [ln.strip() for ln in art["kernel_c"].splitlines()
            if ln.strip().startswith("for (")]
    assert body, "expected the emitted reference to contain loops"
    for line in body:
        assert line in text, line
    assert art["kernel_c"].strip() in text


def test_it_permits_deriving_from_that_reference(rung1):
    """Rung 0's provenance rule says to write the kernel yourself and not to pattern
    it on any existing solution. Carried over verbatim it would forbid using the very
    reference this rung hands over -- a contradiction the model would have to guess
    its way out of. Rung 1 states the permission explicitly.
    """
    text = rung1["text"]
    assert "Derive this kernel from the reference" in text
    assert "Write this kernel yourself" not in text


def test_it_still_withholds_the_schedule_and_the_linalg(rung1):
    for leak in ("iterator_types", "vectorizable_loop", "linalg.", "#map",
                 "affine_map"):
        assert leak not in rung1["text"], leak


def test_it_still_withholds_the_mechanism_budget_and_hardware_facts(rung1):
    for leak in ("Mechanisms this size justifies", "Working set",
                 "L1 data cache", "L2 cache", "HVX vector width", "VTCM",
                 "scratchpad", "tier"):
        assert leak not in rung1["text"], leak


def test_it_names_no_mechanism_and_no_intrinsic(rung1):
    """The sharpest leak, and rung 1 has a new way to spring it: the reference body
    is text rung 0 never included, so the emitted C++ itself must be clean."""
    lowered = rung1["text"].lower()
    for mech in ("hvx", "hmx", "vtcm", "l2fetch", "dma", "intrinsic"):
        assert mech not in lowered, mech
    assert "Q6_" not in rung1["text"]
    assert "V6_" not in rung1["text"]


def test_no_emitted_reference_in_the_eval_core_leaks_a_mechanism():
    """The test above vets ONE task's reference. This vets every one that is on disk.

    Rung 1 pastes `kernel.cpp` into 128 prompts, so a single mechanism word in any
    emitted reference leaks the answer for that task. Reads the regenerated
    `witness_build` tree rather than building 128 artifacts, so it costs milliseconds;
    skips when the tree is absent, since it is a regenerable cache.
    """
    import pathlib
    roots = sorted(pathlib.Path(evalset.WITNESS).glob("batch*/*/kernel.cpp"))
    if not roots:
        pytest.skip("witness_build not present (regenerable cache)")
    bad = []
    for p in roots:
        low = p.read_text(encoding="utf-8").lower()
        for mech in ("hvx", "hmx", "vtcm", "l2fetch", "intrinsic", "q6_", "v6_"):
            if mech in low:
                bad.append((p.parent.name, mech))
    assert not bad, f"emitted references naming a mechanism: {bad[:10]}"


def test_it_is_deterministic(rung1):
    ref, art = rung1["ref"], rung1["art"]
    again = prompt.build_rung1(ref.name, art["signature"], art["graph"],
                               ref.entry, art["kernel_c"])
    assert again == rung1["text"]


def test_it_sits_between_rung0_and_rung2_in_size(rung1):
    """A cheap invariant that catches the two ways this rung can be built wrong:
    forgetting the reference, or accidentally reusing the rung-2 prompt."""
    ref, art = rung1["ref"], rung1["art"]
    r0 = prompt.build_rung0(ref.name, art["signature"], art["graph"], ref.entry)
    r2 = prompt.build_rung2(ref.name, art["signature"], art["graph"], ref.entry,
                            art["kernel_c"], art["plan"],
                            linalg_ir=art["linalg_ir"])
    assert len(r0) < len(rung1["text"]) < len(r2)


# ------------------------------------------------------- the driver's rung switch

def test_the_driver_builds_the_right_prompt_for_each_rung(rung1):
    """One switch, so a rung-1 run cannot silently generate rung-0 prompts.

    This is the failure that would be hardest to notice after the fact: the run
    completes, the records say `rung: 1`, and the numbers are a second rung-0
    measurement with a different label.
    """
    from hexkernels.forge import rung0

    ref, art = rung1["ref"], rung1["art"]
    assert rung0.first_prompt(0, ref, art) == prompt.build_rung0(
        ref.name, art["signature"], art["graph"], ref.entry)
    assert rung0.first_prompt(1, ref, art) == rung1["text"]


def test_the_driver_refuses_a_rung_it_has_no_prompt_for(rung1):
    """Rung 3 has its own scaffolding (HexagonGym) and is not reachable by passing
    --rung 3 to this driver. Refused loudly rather than falling back to rung 0,
    which would mislabel the measurement."""
    from hexkernels.forge import rung0

    ref, art = rung1["ref"], rung1["art"]
    for bad in (3, -1):
        with pytest.raises(ValueError, match="rung"):
            rung0.first_prompt(bad, ref, art)
