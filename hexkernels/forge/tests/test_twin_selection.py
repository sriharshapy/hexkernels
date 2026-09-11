"""Selection must EMIT the fp16 twin, not merely be able to detect it.

Task 2 added hmx_twin_dtype and nothing called it, so the pool still had zero
HMX-eligible specs. This pins the wiring, which is the part that was missing.
"""
import inspect

import torch

from hexkernels.forge import mine
from hexkernels.forge.mechanism import HMX, Plan


def test_selection_calls_the_twin_path():
    """The emit loop must consult a narrow dtype. Guards against the detector
    existing while nothing invokes it -- exactly the gap this task closes."""
    src = inspect.getsource(mine)
    body = src[src.index("picked.append("):]
    assert "float16" in body, (
        "the emit loop never mentions float16; the twin is not wired in")


def test_twin_entry_shape_is_solved_not_copied():
    """Decision 2: the fp16 entry gets its own size_for_tier result."""
    src = inspect.getsource(mine)
    body = src[src.index("picked.append("):]
    assert "size_for_tier" in body, (
        "the twin reuses the fp32 shape instead of solving its own")


def test_emitted_twin_carries_hmx_and_fp16():
    """End-to-end on a hand-built contraction pick.

    Uses the module's own sizing path so this fails if the twin is emitted
    without hmx, or emitted at the wrong dtype.
    """
    picked = mine.mine_twin_probe() if hasattr(mine, "mine_twin_probe") else None
    if picked is None:
        import pytest
        pytest.skip("no probe hook; covered by the pool test in Task 1b")
    twins = [e for e in picked if e["dtype"] == "float16"]
    assert twins, "no fp16 twin emitted"
    for e in twins:
        assert HMX in e["mechanisms"]
        assert e["dtype_bytes"] == 2


def test_fp32_parent_is_retained_as_the_negative_control():
    src = inspect.getsource(mine)
    body = src[src.index("picked.append("):]
    # The fp32 append must not be inside an else-branch of the twin check.
    assert body.count("picked.append(") >= 2, (
        "expected two append sites: the fp32 pick and its twin")


# ---- Fix-round-1 regression: the twin's dedup key must be dtype-qualified --------
#
# Round 1 of this task wired the twin in but keyed its dedup check on the bare
# `graph_signature` result, same as the fp32 parent. `graph_signature` is
# dtype-BLIND for a float contraction (fp32-vs-fp16 decomposes identically --
# unlike an integer operand, which inserts an extra `_to_copy`), so the twin's
# signature was byte-for-byte the parent's, added to `seen_sigs` two lines
# earlier -- so `_tsig not in seen_sigs` was False on every single contraction
# and the twin was silently dropped. A 320-spec mined pool came back with zero
# HMX-eligible specs even though this block ran on every candidate.
#
# `inspect.getsource` string checks (the tests above) cannot catch this: the
# fixed and the broken code both mention `float16` and `size_for_tier`. This
# has to run `mine()` for real and look at what it actually emits.
#
# `mine()` needs a coverage cache and a harvest index that do not exist in
# this checkout (see the task report), so `load_coverage`/`_harvest_index`/
# `corpus_targets`/`corpus_signatures` are monkeypatched to a tiny, real fixture
# instead of the on-disk artifacts -- everything downstream, in particular the
# exact `seen_sigs`/dedup-key control flow under test, is `mine()`'s own,
# unmodified code. `size_for_tier` and `graph_signature` are also
# monkeypatched: they are pure sizing/signature functions with no bearing on
# the bug (the bug is entirely in how their results are used, not in what they
# compute), and replacing them lets the test reproduce the exact dtype-blind
# collision -- same signature at fp32 and fp16 -- without needing a real
# traceable contraction shaped just right for the ladder.
def _kernel_row(op):
    return {
        "op": op, "overload": "", "namespace": "aten", "klass": "kernel",
        "has_tensor_in": True, "has_tensor_out": True,
        "args": [
            {"is_tensor": True, "type": "Tensor", "is_list": False,
             "is_mutable": False, "default": None, "kwonly": False},
            {"is_tensor": True, "type": "Tensor", "is_list": False,
             "is_mutable": False, "default": None, "kwonly": False},
        ],
        "returns": [{"is_tensor": True, "type": "Tensor"}],
        "raw": f"aten::{op}(Tensor self, Tensor mat2) -> Tensor",
        "accessor": f"torch.{op}",
    }


def _patch_mine_io(monkeypatch, keys, hidx):
    """Swap `mine()`'s on-disk data sources for an in-memory fixture, leaving
    every bit of its own control flow (the loop, `seen_sigs`, the twin block
    under test) untouched."""
    monkeypatch.setattr(mine, "load_coverage", lambda path=None: {"covered": keys})
    monkeypatch.setattr(mine, "_harvest_index", lambda: hidx)
    monkeypatch.setattr(mine, "corpus_targets", lambda: set())
    monkeypatch.setattr(mine, "corpus_signatures", lambda: set())


def test_twin_survives_dtype_blind_signature_collision(monkeypatch):
    """Pins the fix-round-1 defect directly: even when the twin's
    `graph_signature` is IDENTICAL to its fp32 parent's (the real, dtype-blind
    case for a float contraction), the twin must still be emitted.
    """
    hidx = {("mm", ""): _kernel_row("mm")}
    _patch_mine_io(monkeypatch, ["mm.default"], hidx)

    def fake_size_for_tier(op, overload, sig, dtype, target, hidx, aplan=None,
                           ladder=None):
        if dtype is torch.float32:
            return ((8, 8), Plan(tier="T0", working_set_bytes=256,
                                 mechanisms=frozenset({"hvx"}), reasons=()), None)
        if dtype is torch.float16:
            return ((8, 8), Plan(tier="T0", working_set_bytes=128,
                                 mechanisms=frozenset({"hvx", HMX}), reasons=()), None)
        return None

    same_sig = ("aten.mm.default",)

    def fake_graph_signature(op, overload, sig, hidx, dtype=torch.float32, plan=None):
        # The real bug: identical signature at fp32 and fp16 for a float
        # contraction, because graph_signature does not look at dtype at all.
        return same_sig

    monkeypatch.setattr(mine, "size_for_tier", fake_size_for_tier)
    monkeypatch.setattr(mine, "graph_signature", fake_graph_signature)

    batches = mine.mine(2, 0, per_batch=1)
    picked = [e for b in batches for e in b] + mine.mine.remainder

    dtypes = sorted(e["dtype"] for e in picked)
    assert dtypes == ["float16", "float32"], (
        f"expected both the fp32 parent and its fp16 twin, got {dtypes!r} -- "
        "the twin was dropped as a duplicate of its own parent")
    twin = next(e for e in picked if e["dtype"] == "float16")
    assert HMX in twin["mechanisms"]


def test_twin_dedups_against_another_twin_not_its_parent(monkeypatch):
    """Decision from the fix: two fp16 twins of the SAME decomposition still
    collapse into one entry -- the fix must not simply stop deduping twins
    altogether, it must dedup them against each other and not their parents.
    """
    hidx = {("mm", ""): _kernel_row("mm"), ("bmm", ""): _kernel_row("bmm")}
    _patch_mine_io(monkeypatch, ["mm.default", "bmm.default"], hidx)

    def fake_size_for_tier(op, overload, sig, dtype, target, hidx, aplan=None,
                           ladder=None):
        if dtype is torch.float32:
            return ((8, 8), Plan(tier="T0", working_set_bytes=256,
                                 mechanisms=frozenset({"hvx"}), reasons=()), None)
        if dtype is torch.float16:
            return ((8, 8), Plan(tier="T0", working_set_bytes=128,
                                 mechanisms=frozenset({"hvx", HMX}), reasons=()), None)
        return None

    def fake_graph_signature(op, overload, sig, hidx, dtype=torch.float32, plan=None):
        if dtype is torch.float16:
            # Both ops' twins decompose to the SAME thing -- a genuine alias.
            return ("SAME_TWIN_DECOMPOSITION",)
        # Distinct fp32 parents, so only the twin collision is under test.
        return (f"{op}_fp32",)

    monkeypatch.setattr(mine, "size_for_tier", fake_size_for_tier)
    monkeypatch.setattr(mine, "graph_signature", fake_graph_signature)

    batches = mine.mine(4, 0, per_batch=1)
    picked = [e for b in batches for e in b] + mine.mine.remainder

    twins = [e for e in picked if e["dtype"] == "float16"]
    assert len(twins) == 1, (
        f"expected the two same-decomposition twins to dedup to one, got "
        f"{len(twins)}")
    parents = [e for e in picked if e["dtype"] == "float32"]
    assert len(parents) == 2, "both fp32 parents (mm, bmm) should still be kept"
