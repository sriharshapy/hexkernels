"""Where a kernel came from, traced back to PyTorch, or it does not ship.

THE RULE
--------
A kernel is admissible only if **every primitive in its traced graph resolves to a
row in the harvested PyTorch operator registry**. No resolution, no kernel. This is
not a labelling exercise: a kernel whose op cannot be pointed back at
`torch._C._jit_get_all_schemas()` has no provenance chain, and this pipeline's only
claim is that its corpus derives from public sources. An unprovable kernel is
useless however correct and however fast, so it is destroyed rather than kept with
a caveat.

WHAT THE CHAIN IS
-----------------
Four links, each citable to something a reader can re-run:

    1. ORIGIN     torch._C._jit_get_all_schemas() at a pinned torch version
                  -> the schema string, e.g. `aten::mm(Tensor self, Tensor mat2)
                     -> Tensor`
    2. SELECTION  `mechanism.mechanism_eligible` on that row -- tensor in, tensor
                  out, real arithmetic. Carries the harvest's own `reason`.
    3. SIZE       the traced shapes -> working-set bytes -> a tier, by comparison
                  against the PROBED memory hierarchy (not a judgement)
    4. MECHANISM  the tier -> which hardware mechanisms the size entitles, each
                  with the arithmetic that justifies it

Link 1 is the one this module adds and the one that was missing. Links 2-4 already
existed in `mechanism.py`; what did not exist was any check that the op at the
bottom of them had ever been harvested. Batch 3's ops all had been -- but that was
established by hand afterwards, which is exactly the difference between a provenance
chain and a plausible story.

WHY THE OVERLOAD MATTERS
------------------------
A traced target is `aten.sum.dim_IntList`, not `aten.sum`: the overload selects
which schema, and different overloads have different argument lists and different
nests. Resolving only the base name would let `aten.sum.SomethingElse` inherit the
provenance of a schema it does not implement. So the overload is matched too, with
FX's `default` mapping to the registry's empty overload string.
"""
import json
import os

from hexkernels.forge.mechanism import (mechanism_eligible,
                                       primitive_admissible)

DEFAULT_HARVEST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmark", "ops.jsonl")


class UnprovenKernel(ValueError):
    """A primitive in this kernel does not resolve to a harvested PyTorch schema.

    Raised, not warned. The whole point is that such a kernel cannot be kept.
    """


class RDLeak(ValueError):
    """A forge v2 candidate reached into the R&D side of the repo."""


# R&D-ONLY, MUST NOT ENTER A FORGE V2 KERNEL.
#
# `hmx_helpers.h` is research-and-development material: by its own docstring it was
# *lifted from* expert solutions. Forge v2's claim is that its kernels derive from
# the reference, the schedule and the VENDOR headers only, so a forge v2 kernel that
# calls it -- or is written having read it -- has a different and weaker provenance
# story than the one this pipeline makes. That is a provenance failure, not a
# shortcut, and it does not become acceptable because the kernel is correct.
#
# The channel was wide open and nobody had noticed: `hmx_helpers.h` lives in
# `hexbench/env/harness/`, which is exactly the directory `verify.py` passes to
# `-I`. Any candidate could have included it and it would have compiled. Three
# things close it now, because one is not enough:
#
#   1. this source scan, which also catches a COPY of the helper body (an include
#      guard would not);
#   2. `-DFORGE2_BUILD=1` on every forge v2 compile, with an `#error` in the header,
#      so an include is a hard failure with a message rather than a silent success;
#   3. the scan runs before compiling, so a leaking candidate is never even built.
# EVERY header under `hexbench/env/harness/` is R&D-only, not just the HMX one.
# `harness_common.h` was the wider channel of the two: seven candidates across
# batches 2 and 3 included it, for the VTCM aperture base, the 128-byte alignment
# attribute and the fp16 crouton offset -- all facts a kernel must state for
# itself, from the target and the vendor documentation, if its chain is to mean
# anything. The include path is gone from the compile (`forge2.verify`), so these
# are unreachable rather than merely forbidden; this list is what catches a
# COPIED BODY, which no include path can.
RD_ONLY_HEADERS = ("hmx_helpers.h", "harness_common.h")
RD_ONLY_SYMBOLS = ("hmx_tile_matmul_i8_field", "hmx_tile_matmul_fp16_field",
                   "hmx_tile_matmul_i8", "hmx_tile_matmul_fp16",
                   "hvx_hmx_i8_act_off", "hvx_hmx_i8_wgt_off",
                   "hvx_hmx_i8_out_off",
                   # harness_common.h's exports. The names, not the values: a
                   # kernel is free -- required, in fact -- to derive the same
                   # 0xd9000000 aperture or the same (r/2)*64+c*2+(r&1) crouton
                   # offset and say where it got it. What it may not do is carry
                   # the R&D spelling, because that is the evidence of having
                   # read the R&D file rather than the documentation.
                   "HVX_VTCM_BASE", "HVX_ALIGN", "hvx_crouton_off",
                   "hvx_hmx_enable", "hvx_close_f32", "hvx_close_f16bits",
                   "hvx_report", "hvx_report_u16", "hvx_hmx_requant_0x40")


def rd_leaks(source: str) -> list:
    """Every R&D reference in a candidate. Empty list means clean.

    Matches MENTIONS, comments included, deliberately. A comment saying "I did not
    use the helper because ..." still evidences that the author read R&D material,
    and a rule that has to judge intent is a rule nobody can enforce. The findings
    such a comment records are worth keeping -- in the batch report or the SDK doc
    index, which is where a fact about the benchmark belongs, not inside a forge v2
    kernel.
    """
    return sorted({tok for tok in RD_ONLY_HEADERS + RD_ONLY_SYMBOLS
                   if tok in source})


def assert_no_rd_leak(source: str, name="candidate") -> None:
    hits = rd_leaks(source)
    if hits:
        raise RDLeak(
            f"{name}: references R&D-only material {hits}. "
            "Nothing under hexbench/env/harness/ may appear in a forge v2 kernel -- "
            "neither `hmx_helpers.h` and its tile-matmul helpers nor "
            "`harness_common.h` and its aperture/alignment/crouton macros. A forge "
            "v2 kernel must derive only from the "
            "reference, the schedule and the vendor's intrinsic headers "
            "(<hexagon_types.h>, <hexagon_protos.h>, <hvx_hexagon_protos.h>, "
            "<hmx_hexagon_protos.h>). This is a provenance failure and is not "
            "excused by the kernel being correct.")


def load_harvest(path=DEFAULT_HARVEST) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def harvest_index(rows) -> dict:
    """(namespace, op, overload) -> row, for rows ADMISSIBLE AS A PRIMITIVE.

    Keyed on the full triple because the overload is part of the identity.

    THIS USED TO INDEX ONLY SELECTION-ELIGIBLE ROWS, and that was the conflation.
    Its docstring said "a kernel built on an op the selection filter rejects (a
    metadata-only `view`) has no more standing than one built on an op that was never
    harvested" -- which is true of the op a kernel IS, and false of the ops it
    decomposes THROUGH. Because the index was the only lookup, an unselected
    primitive could not even be found, so `chain` marked it unresolved and
    `assert_proven` destroyed the kernel.

    Measured cost: 66 of the expressible ops, including
    `scaled_dot_product_attention` (17 primitives, 2 of them `permute` and
    `full_like`), `rnn_relu_cell` and `rnn_tanh_cell`.

    The claim this module enforces is unchanged: the kernel is a PyTorch op and every
    step inside it is a PyTorch op. What is gone is the extra demand that each step
    also be independently kernel-worthy, which was never a statement about origin.
    See `mechanism.primitive_admissible` for which harvest classes qualify and why
    three are still refused.
    """
    return {(r.get("namespace"), r.get("op"), r.get("overload") or ""): r
            for r in rows if primitive_admissible(r)}


def split_target(target: str):
    """`aten.sum.dim_IntList` -> ("aten", "sum", "dim_IntList").

    FX spells the no-overload case `default`; the registry spells it "". Anything
    that is not exactly namespace.op.overload returns None so the caller reports it
    unresolved rather than guessing at a split.
    """
    parts = target.split(".")
    if len(parts) != 3:
        return None
    ns, op, ov = parts
    return ns, op, ("" if ov == "default" else ov)


def resolve(target: str, index: dict):
    """The harvested row this traced primitive came from, or None."""
    key = split_target(target)
    return None if key is None else index.get(key)


def chain(art, index, harvest_path=DEFAULT_HARVEST) -> dict:
    """The full origin-to-mechanism record for one built kernel.

    Takes a `run_batch.build` artifact. Returns a dict that is written verbatim to
    `provenance.json` beside the kernel, so the claim travels with the artifact
    rather than living in a report someone has to correlate.
    """
    prims = art.get("primitives") or []
    links, unresolved = [], []
    for target in prims:
        row = resolve(target, index)
        if row is None:
            unresolved.append(target)
            links.append({"primitive": target, "resolved": False})
            continue
        links.append({
            "primitive": target,
            "resolved": True,
            # Link 1: the origin, as a string a reader can grep for in the registry.
            "schema": row.get("raw"),
            "accessor": row.get("accessor"),
            "source": row.get("source"),
            "torch_version": row.get("torch_version"),
            # Link 2: why the harvest kept it.
            "klass": row.get("klass"),
            "selection_reason": row.get("reason"),
            # SAFEGUARD 2 of the split: is this step ALSO kernel-worthy on its own,
            # or is it plumbing the selected op decomposes through? Recorded per
            # primitive so "attention is 17 primitives of which 2 are plumbing" is a
            # fact on disk rather than something a reader has to work out.
            "kernel_worthy": bool(mechanism_eligible(row)),
        })
    plan = art.get("plan")
    return {
        "kernel": art.get("name"),
        "harvest": os.path.relpath(harvest_path, os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "primitives": links,
        "unresolved": unresolved,
        "proven": bool(prims) and not unresolved,
        # The plumbing ratio, so a corpus filling up with shape ops is visible in
        # every record and in REPORT.md rather than discovered later.
        "n_primitives": len(links),
        "n_plumbing": sum(1 for l in links
                          if l.get("resolved") and not l.get("kernel_worthy")),
        # VACUOUS = every primitive is plumbing, so the kernel computes nothing.
        # This is the part of the old merged rule that was worth keeping, at the
        # right granularity: the KERNEL must do arithmetic; its individual STEPS
        # need not each be arithmetic. See `assert_proven`.
        "vacuous": bool(links) and all(
            l.get("resolved") and not l.get("kernel_worthy") for l in links),
        # Links 3 and 4, copied from the plan so the record is self-contained.
        "working_set_bytes": art.get("working_set_bytes"),
        "tier": art.get("tier"),
        "mechanisms": art.get("mechanisms"),
        "mechanism_reasons": ([[m, why] for m, why in plan.reasons]
                              if plan is not None else []),
    }


def assert_proven(art, index, harvest_path=DEFAULT_HARVEST) -> dict:
    """`chain`, but raising if the kernel has no origin.

    The enforcement point. A kernel that reaches here unproven does not get a
    caveat in a report -- it stops the batch for that kernel.
    """
    rec = chain(art, index, harvest_path)

    # SAFEGUARD 1, ENFORCED HERE RATHER THAN ONLY AT MINING TIME.
    #
    # Relaxing the per-primitive rule to `primitive_admissible` is what makes
    # attention buildable, and it has one edge that the relaxation alone does not
    # cover: a kernel whose graph is ENTIRELY plumbing. `x.t().contiguous()` traces to
    # `permute` + `clone`, both admissible, both harvested -- so it built cleanly and
    # reported `proven: True` with `n_plumbing 2/2`. It computes nothing. There is no
    # accelerator use to measure and nothing for a candidate to accelerate.
    #
    # `mine`'s non-vacuous filter already refused these when SELECTING (an op granted
    # no compute mechanism), but hand-written specs never pass through it, so the
    # guarantee was only as good as who wrote the spec. It belongs at the build stage,
    # where every kernel goes.
    #
    # This is the half of the old merged rule that was worth keeping, moved to the
    # right granularity: the KERNEL must do arithmetic, its individual STEPS need not.
    if rec.get("vacuous"):
        raise UnprovenKernel(
            f"{art.get('name')}: every one of its {rec['n_primitives']} primitives is "
            "plumbing (shape, metadata or constant-materialising), so the kernel "
            "computes nothing and there is no accelerator use to measure. A primitive "
            "may be plumbing -- attention transposes K -- but a KERNEL may not be "
            "entirely plumbing. Select it on an op that does arithmetic.")

    if not rec["proven"]:
        why = ("no primitives at all" if not rec["primitives"]
               else f"unresolved primitives: {', '.join(rec['unresolved'])}")
        raise UnprovenKernel(
            f"{art.get('name')}: {why}. Every primitive must resolve to a "
            f"harvested PyTorch schema in {harvest_path} that is admissible as a "
            "primitive (see mechanism.primitive_admissible: an aten/prims row of "
            "class kernel, plumbing or no_tensor). A kernel without that chain has "
            "no provenance and cannot be kept, however correct it is.")
    return rec


def render(rec) -> str:
    """Human-readable chain, for the report and for a reader checking the claim."""
    head = [f"# Provenance — {rec['kernel']}", "",
            f"Harvest: `{rec['harvest']}`", ""]
    if not rec["proven"]:
        head += [f"**UNPROVEN — {rec['unresolved']}**", ""]
    head += ["| primitive | PyTorch schema | harvested via | kept because |",
             "|---|---|---|---|"]
    for l in rec["primitives"]:
        if l["resolved"]:
            head.append(f"| `{l['primitive']}` | `{l['schema']}` | "
                        f"`{l['accessor']}` (torch {l['torch_version']}) | "
                        f"{l['selection_reason']} |")
        else:
            head.append(f"| `{l['primitive']}` | **UNRESOLVED** | — | — |")
    head += ["", f"Working set {rec['working_set_bytes']} B -> tier {rec['tier']}; "
                 f"mechanisms {', '.join(rec['mechanisms'] or []) or 'none'}.", ""]
    for m, why in rec["mechanism_reasons"]:
        head.append(f"- **{m}**: {why}")
    return "\n".join(head) + "\n"
