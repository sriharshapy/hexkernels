"""Invariants of the assembled corpus.

The admission rule is the whole premise of this repo: a library entry is a
kernel that reaches the accelerator. These tests fail if `kernels/` ever drifts
from that -- including if a future assembler run quietly lets a scalar kernel in.
"""

import json
import os
import re

import pytest

from hexkernels.library import find, load, root, summary

HVX_RE = re.compile(r"HVX_Vector|Q6_V[a-zA-Z0-9_]*\(|Q6_W[a-zA-Z0-9_]*\(")
HMX_RE = re.compile(r"\bHMX\b|mxmem|Q6_mx|hmx_", re.IGNORECASE)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(root(), "index.json")),
    reason="kernels/ not assembled; run tools/assemble_library.py",
)


def test_the_corpus_loads_and_matches_its_own_index():
    kernels = load()
    assert len(kernels) == summary()["total"]
    counts = summary()["counts"]
    for origin, n in counts.items():
        assert len(find(origin=origin)) == n


def test_every_entry_actually_reaches_the_accelerator():
    """The admission rule. A scalar kernel is never a library entry."""
    scalar = [k.name for k in load()
              if not (HVX_RE.search(k.source) or HMX_RE.search(k.source))]
    assert scalar == [], f"scalar kernels admitted to the library: {scalar}"


def test_no_reference_is_itself_accelerated():
    """`reference.c` is the scalar ground truth -- if it vectorised, it is not
    a reference any more and the comparison it anchors means nothing."""
    bad = []
    for k in load():
        ref = k.reference
        if ref is None:
            continue
        # Strip comments first: several references legitimately *mention* an
        # intrinsic in prose while remaining entirely scalar.
        code = re.sub(r"/\*.*?\*/", "", ref, flags=re.S)
        code = re.sub(r"//.*", "", code)
        if HVX_RE.search(code):
            bad.append(k.name)
    assert bad == [], f"references containing vector code: {bad}"


def test_every_bundle_marked_complete_really_is():
    for k in find(buildable=True):
        for f in ("kernel.c", "reference.c", "harness.c", "spec.json"):
            assert os.path.exists(os.path.join(k.path, f)), f"{k.name} missing {f}"


def test_elf_confirmation_is_tri_state_and_never_collapsed():
    """`None` (never scanned) must stay distinct from `False` (scanned, absent)."""
    states = {k.elf_confirmed for k in load()}
    assert states <= {True, False, None}
    # `find` must not treat None as a filter value.
    assert len(find(elf_confirmed=None)) == len(load())
    unscanned = find(elf_confirmed="unscanned")
    assert all(k.elf_confirmed is None for k in unscanned)
    assert (len(find(elf_confirmed=True)) + len(find(elf_confirmed=False))
            + len(unscanned)) == len(load())


def test_mined_kernels_carry_their_tier_mismatch_warning():
    """Most mined kernels were authored for a different tier than they are filed
    under. That is retained deliberately, so the flag must survive assembly."""
    mined = find(origin="mined")
    assert mined
    assert all("tier_match" in k.spec["provenance"] for k in mined)
    mismatched = [k for k in mined if k.spec["provenance"]["tier_match"] is False]
    assert mismatched, "the tier mismatch is a documented property of this corpus"


def test_specs_are_valid_json_with_the_required_keys():
    required = {"name", "origin", "entry", "bundle", "files",
                "kernel_sha256", "mechanisms_in_source", "verified", "provenance"}
    for k in load():
        with open(os.path.join(k.path, "spec.json"), encoding="utf-8") as fh:
            spec = json.load(fh)
        assert required <= set(spec), f"{k.name} missing {required - set(spec)}"


def test_expert_speedups_are_measured_not_asserted():
    experts = [k for k in find(origin="expert") if k.speedup is not None]
    assert len(experts) > 300
    assert all(k.speedup > 0 for k in experts)


def test_oversized_harnesses_say_how_to_regenerate():
    """A harness too large to ship must leave instructions, not a silent gap."""
    regen = [k for k in load() if k.spec["bundle"] == "harness-regenerable"]
    assert regen, "the harness cap is part of this corpus's design"
    for k in regen:
        assert os.path.exists(os.path.join(k.path, "HARNESS.md")), k.name
        assert not os.path.exists(os.path.join(k.path, "harness.c")), k.name
        # The non-regenerable half must still be there.
        assert k.source and k.reference
        assert k.spec["harness_bytes"] > 0


def test_bundle_states_are_exhaustive():
    states = {k.spec["bundle"] for k in load()}
    assert states <= {"complete", "harness-regenerable", "kernel-only"}
    assert (summary()["bundle_complete"]
            + summary()["bundle_harness_regenerable"]
            + summary()["bundle_kernel_only"]) == summary()["total"]
