"""Batches 16+ , built from the MINED selection rather than written by hand.

The selection lives in `run_artifacts/forge2/mined_selection.json`, produced by
`forge2.mine`. This module turns it into `KernelSpec`s at import time, so the
committed artifact is the SELECTION and not generated code -- there is no file
anyone has to remember to regenerate, and a diff to the selection is a diff to the
batches.

WHY THESE OPS AND NOT OTHERS
----------------------------
Nothing here was chosen by a person. `mine` walks `coverage.scan`'s measured
`covered` list in the harvest's own row order and keeps a row only if it

  * is not already in the corpus (by op AND by decomposition -- see below),
  * survives `mechanism_eligible` for every primitive it decomposes to, so the
    provenance stage cannot destroy it,
  * is granted a COMPUTE mechanism, so there is something to accelerate, and
  * can be sized to hit its slot's target tier.

Two filters were added after the first mining run produced a list that counted
fifty kernels and did not contain fifty tasks:

  * **ALIAS COLLAPSE.** The registry carries many spellings of one operation.
    Harvest order gave `absolute` beside `abs`, `arccos` beside `acos`, and
    `arcsin`/`arctan`/`divide`/`fix` beside their canonical names. Ops that
    decompose to the SAME primitive graph are one task here however they are
    spelled, so the signature is the traced graph and duplicates are dropped. A
    name blocklist would have been a judgement call and would go stale.
  * **NON-VACUOUS.** `atleast_1d`, `atleast_2d`, `fliplr` and `conj_physical` all
    planned to `l2fetch` alone with no compute mechanism at all -- at rank 2
    `atleast_2d` is the identity. A task with nothing to accelerate cannot report
    whether an accelerator was used.

WHAT THIS BATCH SERIES IS, HONESTLY
-----------------------------------
It is the elementwise and transcendental TAIL of the operator registry, because
that is what mining the covered list in registry order yields: unary and binary
maps, comparison predicates, a few losses and reductions. It is not a curated tour
of interesting nests -- batches 1-15 were that, and the interesting nests were
found by hand. What this series adds is BREADTH with a mechanical selection
argument, which is the part a reader can check.

The tier pattern T0/T1/T1/T2/T3 per batch is the one deliberate choice, and it is a
correction rather than a preference: the corpus-wide EDA found 62% of the first 55
kernels in T1, with 3 in T0 and 5 in T3.
"""
import json
import pathlib

import torch

from hexkernels.forge.kernels import KernelSpec

SELECTION = (pathlib.Path(__file__).resolve().parents[1]
             / "benchmark" / "selection.json")

_DTYPES = {"float32": torch.float32, "float16": torch.float16,
           "int32": torch.int32, "int8": torch.int8, "uint8": torch.uint8,
           "bool": torch.bool}


class FusedCall(torch.nn.Module):
    """A CHAIN of registry calls: `out = g(f(x))`.

    A KERNEL IS A COMPUTATION, NOT AN OPERATOR. `conv`, `conv + layernorm` and
    `conv + softmax` are three different kernels -- three schedules, three fusion
    opportunities, three different right answers about what a candidate should do --
    even though they compose the same registry entries. Selecting only single ops
    treated them as one, which is why the corpus appeared to exhaust the registry at
    250 kernels while the composition space was untouched.

    PROVENANCE IS UNCHANGED AND THAT IS THE POINT. Every primitive in the fused graph
    still resolves to a harvested PyTorch schema, so `assert_proven` passes exactly as
    it does for a single op -- the claim "this kernel derives from public PyTorch" is
    as true of a composition as of its parts. What the composition is NOT is a new
    operator: it is a new TASK built from harvested ones, and the selection records
    both stages so a reader can check that.

    Measured before this was built: of 120 randomly chosen ordered pairs of unary
    expressible ops, 114 traced and 113 also emitted, scheduled and passed provenance
    -- a 94% viable rate. The composition space is not a long tail; it is the bulk of
    what this pipeline can express.

    Chained on the FIRST tensor position, which is where an elementwise or reduction op
    takes its operand. A stage needing more than one tensor is not composed here; that
    is a wider synthesiser and a separate question.
    """

    def __init__(self, stages):
        super().__init__()
        self.fns = []
        for op, overload in stages:
            packet = getattr(torch.ops.aten, op)
            self.fns.append(getattr(packet, overload or "default"))
        self.stages = list(stages)

    def forward(self, *xs):
        # THE FIRST STAGE MAY TAKE SEVERAL TENSORS -- `add`, `mul`, `matmul`, `conv`.
        # That is where the valuable fusions live (`linear then gelu`,
        # `matmul then softmax`), and restricting the chain to unary first stages
        # exhausted the composed space at ~100 kernels while leaving those untouched.
        # Later stages are epilogues and take the running value alone.
        x = self.fns[0](*xs)
        for fn in self.fns[1:]:
            x = fn(x)
        return x


class MinedCall(torch.nn.Module):
    """A module whose forward is exactly one registry call.

    One op per module is what keeps the result attributable: every primitive in the
    traced graph came from this op's decomposition and nothing else, which is the
    same discipline `coverage._Call` uses when it proves the op expressible in the
    first place.
    """

    def __init__(self, op: str, overload: str, synth_plan=None):
        super().__init__()
        packet = getattr(torch.ops.aten, op)
        self.fn = getattr(packet, overload or "default")
        self.op = op
        self.overload = overload or "default"
        # The per-position argument plan for a SYNTH row: positions marked "t" take
        # the tensors the caller passes, and positions marked "v" carry a value this
        # pipeline chose. Baked in here rather than passed as arguments so only the
        # TENSORS become graph placeholders -- which is what makes the emitted
        # kernel's signature the tensor list and nothing else.
        self.synth_plan = synth_plan

    def forward(self, *args):
        if self.synth_plan is None:
            return self.fn(*args)
        it = iter(args)
        full = tuple(next(it) if kind == "t" else v for kind, v in self.synth_plan)
        return self.fn(*full)


def _args(entry, seed=0):
    """Deterministic example arguments for one mined entry.

    Seeded per kernel so the golden is reproducible: the harness embeds the actual
    values, and an unseeded draw would make every rebuild a different task.

    Floats are drawn from [0.5, 1.5) rather than N(0,1) -- the same choice
    `coverage` makes when it probes. Away from 0 and 1, `log`, `rsqrt`, `acosh`,
    `atanh` and the rest are defined and non-degenerate, and a NaN in the golden
    would fail a kernel for a reason that has nothing to do with the kernel.
    `acosh` in particular needs its argument >= 1, which this range does not
    guarantee -- see `_domain_shift`.
    """
    torch.manual_seed(seed)
    dtype = _DTYPES[entry["dtype"]]
    shape = tuple(entry["shape"])
    plan = entry.get("synth_plan")
    n = (sum(1 for kind, _v in plan if kind == "t") if plan
         else (1 if entry["sig"] == "unary" else 2))
    # PER-POSITION SHAPES. A plan entry may carry its tensor's own shape -- a
    # convolution WEIGHT is (Cout, Cin, K...) and is NOT the input's shape. This is the
    # fourth module that constructs the call (after `coverage.probe`,
    # `mine._args_for` and `mine.graph_signature`) and the last one to be taught it.
    #
    # It is safe ONLY because `mine()` now stores the plan `size_for_tier` actually
    # used, re-derived at the chosen shape's extent and channel count. Reading the
    # stored shapes while the miner recorded probe-time ones is what broke batch 48.
    tshapes = ([tuple(sh) if sh else shape for kind, sh in plan if kind == "t"]
               if plan else [shape] * n)
    if dtype is torch.bool:
        return tuple(torch.randint(0, 2, s_).bool() for s_ in tshapes)
    if dtype.is_floating_point:
        return tuple(_domain_shift(entry["op"],
                                   torch.rand(s_, dtype=dtype) + 0.5)
                     for s_ in tshapes)
    return tuple(torch.randint(1, 5, s_, dtype=dtype) for s_ in tshapes)


#: Ops whose DOMAIN excludes part of the default [0.5, 1.5) draw.
#:
#: MEASURED, not anticipated: the first reference build lost `acos` and `acosh`
#: because [0.5, 1.5) leaves their domains -- acos is undefined above 1, acosh
#: below it -- so torch produced a golden full of NaN. A NaN golden is worse than a
#: wrong one: `nan != nan`, so every element fails and the failure looks like a
#: kernel bug rather than a badly chosen input.
#:
#: Stated per op with a NAMED range rather than clamped globally. A global clamp
#: would silently change what the task is, and squeezing every op into one safe
#: interval would make the whole series test the same narrow band.
_UNIT = "unit"      # |x| < 1     : the inverse circular / atanh family
_GE1 = "ge1"        # x >= 1      : acosh
_PROB = "prob"      # 0 < x < 1   : logit, and anything taking a PROBABILITY
_DOMAIN = {
    "acos": _UNIT, "arccos": _UNIT,
    "asin": _UNIT, "arcsin": _UNIT,
    "atanh": _UNIT, "arctanh": _UNIT,
    "erfinv": _UNIT,
    "acosh": _GE1, "arccosh": _GE1,
    # `logit(p) = log(p/(1-p))` is defined only on (0, 1). On the default [0.5, 1.5)
    # draw more than half the tensor has p > 1, so `1 - p` is negative and its log is
    # NaN -- measured, 3,124 of 6,144 elements, and since `nan != nan` EVERY one of
    # them fails the harness. The reference was reported incorrect for a reason that
    # has nothing to do with the reference.
    "logit": _PROB, "special_logit": _PROB,
    # torch refuses outright rather than returning NaN: "all elements of target
    # should be between 0 and 1". Both operands are probabilities here.
    "binary_cross_entropy": _PROB,
    "kl_div": _PROB,
}


def _domain_shift(op, t):
    """Move `t` (drawn from [0.5, 1.5)) into `op`'s domain, or leave it alone.

    `_UNIT` maps to [-0.9, 0.9]: strictly inside the open interval, and SIGNED, so
    the negative half of the function is exercised rather than only the positive.
    0.9 rather than 1.0 because `atanh` diverges at the endpoint and a golden with
    an infinity has the same problem a NaN does.
    """
    kind = _DOMAIN.get(op)
    if kind == _UNIT:
        return (t - 1.0) * 1.8            # [0.5, 1.5) -> [-0.9, 0.9)
    if kind == _GE1:
        return t + 1.0                    # [0.5, 1.5) -> [1.5, 2.5)
    if kind == _PROB:
        # [0.5, 1.5) -> [0.05, 0.95): strictly inside (0, 1), and away from both
        # endpoints because `logit` diverges at each and a golden with an infinity
        # has the same problem a NaN does.
        return (t - 0.5) * 0.9 + 0.05
    return t


def _note(entry):
    """The per-kernel note. Says what was DERIVED, since nothing here was chosen."""
    if entry.get("stages"):
        return _fused_note(entry)
    ov = entry["overload"] or "default"
    return (f"MINED, not chosen: `aten::{entry['op']}.{ov}` came from "
            f"coverage's measured `covered` list in harvest order, survived the "
            f"alias and non-vacuous filters, and was sized to {tuple(entry['shape'])} "
            f"because that is the smallest shape on the ladder whose working set "
            f"({entry['working_set']:,} B) lands in {entry['tier']}. Mechanisms "
            f"{','.join(entry['mechanisms'])} follow from that size. Schema: "
            f"{entry['schema'].split('(')[0].strip() or 'aten::' + entry['op']}")


def _fused_note(entry):
    """The note for a COMPOSED kernel, and the marking is the point.

    A fused kernel must never read as a harvested operator. `aten::relu` is something
    PyTorch defines; `relu then softmax` is something THIS PIPELINE chose to chain.
    Both are legitimate benchmark tasks and only one of them is an operator, so the
    note says which -- leading with COMPOSED, naming both stages and their schemas,
    and stating exactly where the provenance claim holds and where it stops.

    WHAT IS INHERITED FROM THE HARVEST: every primitive in the traced graph resolves
    to a harvested PyTorch schema, so `assert_proven` passes for a composition exactly
    as for a single op. Each STEP is public PyTorch.

    WHAT IS OURS: the decision to chain these two steps. PyTorch does not define this
    operator -- there is no `aten::relu_softmax` -- and a reader must not be able to
    infer one from the corpus. That is why the name carries a double underscore and
    this note leads with the word COMPOSED.
    """
    a, b = entry["stages"][0], entry["stages"][1]
    an = f"aten::{a[0]}" + (f".{a[1]}" if a[1] else "")
    bn = f"aten::{b[0]}" + (f".{b[1]}" if b[1] else "")
    nl = chr(10) * 2
    return (
        f"COMPOSED, NOT A PYTORCH OPERATOR. This task is `{bn}({an}(x))` -- two "
        f"harvested operators chained by this pipeline. PyTorch defines each STEP and "
        f"does NOT define their composition: there is no single registry entry for it "
        f"and none should be inferred from this corpus." + nl +
        f"PROVENANCE, precisely: every primitive in the traced graph resolves to a "
        f"harvested schema, so the origin claim holds for `{an}` and `{bn}` "
        f"individually and `assert_proven` passes as it would for either alone. What "
        f"is THIS PIPELINE'S is the choice to chain them -- which is why the kernel is "
        f"named for the chain and this note leads with COMPOSED." + nl +
        f"WHY COMPOSE: a kernel is a computation, not an operator. `conv`, "
        f"`conv + layernorm` and `conv + softmax` are three kernels -- three schedules, "
        f"three fusion opportunities, three different right answers for a candidate -- "
        f"though they compose the same registry entries." + nl +
        f"RELEVANT, not exhaustive: the second stage is an EPILOGUE (an activation, "
        f"normalisation or reduction), which is the composition real networks and real "
        f"fusion passes actually produce. Arbitrary pairs like `abs then acosh` are "
        f"excluded -- they would inflate the corpus without adding a schedule anyone "
        f"needs. Pairs are walked in harvest row order, deduplicated by traced-graph "
        f"signature, and filtered so the composition computes something new and the "
        f"stages are type-compatible." + nl +
        f"Sized to {tuple(entry['shape'])} because that working set "
        f"({entry['working_set']:,} B) lands in {entry['tier']}; mechanisms "
        f"{','.join(entry['mechanisms'])} follow from that size.")


def _load():
    if not SELECTION.exists():
        return {}
    doc = json.loads(SELECTION.read_text(encoding="utf-8"))
    first, per = doc["first_batch"], doc["per_batch"]
    out = {}
    for bi, chunk in enumerate(doc["batches"]):
        specs = []
        for ki, entry in enumerate(chunk):
            ov = entry["overload"]
            pre = entry['dtype'].replace('float', 'fp').replace('int', 'i')
            # A FUSED entry carries `stages` instead of a single op: `out = g(f(x))`.
            # Named for the chain so the corpus reads as tasks rather than operators --
            # `fp32_relu__softmax` is a different kernel from either part.
            if entry.get("stages"):
                stages = [(o, v) for o, v in entry["stages"]]
                name = pre + "_" + "__".join(o for o, _v in stages)
                module = FusedCall(stages)
            else:
                name = f"{pre}_{entry['op']}" + (f"_{ov}" if ov else "")
                module = MinedCall(entry["op"], ov, entry.get("synth_plan"))
            specs.append(KernelSpec(
                name=name,
                module=module,
                args=_args(entry, seed=1000 * (first + bi) + ki),
                dtype_bytes=entry["dtype_bytes"],
                expect_tier=entry["tier"],
                expect_mechanisms=frozenset(entry["mechanisms"]),
                note=_note(entry),
            ))
        out[first + bi] = tuple(specs)
    return out


MINED_BATCHES = _load()


def batch(n: int):
    if n not in MINED_BATCHES:
        raise ValueError(f"no mined batch {n}; have {sorted(MINED_BATCHES)}")
    return MINED_BATCHES[n]
