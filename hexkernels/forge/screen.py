"""Screen candidate ops before committing a batch to them.

WHY THIS EXISTS
---------------
`coverage.py` answers "can the pipeline express this op" -- trace, schedule, emit.
It does NOT answer the two questions that actually decide whether a kernel can go
in a batch:

  * **Does torch-mlir legalize it?** Both IRs are required pipeline inputs, and
    torch-mlir refuses ops it has no lowering for. Batch 4 lost `linalg_vecdot`
    that way and batch 5 lost `erfc` and `hypot`, each discovered only after the
    spec was written and a reference had been built.
  * **What tier and mechanisms does the size give it, and are they EARNED?**
    `ABLATION.md` measured that dma/vtcm and l2fetch pay off only above ~3 vector
    ops per element. A kernel granted staging at intensity 1 is a kernel whose
    mechanism claim the measurement contradicts.

So this screens a shortlist cheaply -- no simulator, no candidate -- and reports
which ops are viable and at what size. Cost is a trace, a plan and a torch-mlir
invocation per op, a few seconds each.

    python -m hexkernels.forge.screen                # the built-in shortlist
"""
import argparse
import sys

import torch

from hexkernels.forge.frontend.emit import emit_c
from hexkernels.forge.frontend.trace import trace
from hexkernels.forge import linalg as _linalg
from hexkernels.forge import mechanism as _mech

# Vector operations per element, counted from the reference's primitive chain. Used
# only to say whether a size's mechanism grant is EARNED (>= 3) or merely allowed.
STAGING_THRESHOLD = 3


def screen(name: str, module, args, dtype_bytes: int) -> dict:
    """trace -> plan -> emit -> Linalg, reporting the first thing that fails."""
    out = {"name": name, "tier": None, "mechanisms": None, "prims": None,
           "linalg": None, "error": None, "working_set": None}
    try:
        g = trace(module, args, name)
    except Exception as e:                                # noqa: BLE001
        out["error"] = f"trace: {type(e).__name__}: {str(e)[:120]}"
        return out
    out["prims"] = len(g.nodes)
    try:
        plan = _mech.plan_for(g, dtype_bytes)
        out["tier"] = plan.tier
        out["mechanisms"] = sorted(plan.mechanisms)
        out["working_set"] = plan.working_set_bytes
    except Exception as e:                                # noqa: BLE001
        out["error"] = f"plan: {type(e).__name__}: {str(e)[:120]}"
        return out
    try:
        emit_c(g)
    except Exception as e:                                # noqa: BLE001
        out["error"] = f"emit: {type(e).__name__}: {str(e)[:120]}"
        return out
    try:
        ir = _linalg.linalg_ir(module, args)
        out["linalg"] = bool(ir)
        if not ir:
            out["error"] = "linalg: produced nothing"
    except Exception as e:                                # noqa: BLE001
        out["linalg"] = False
        msg = str(e)
        # the message that matters is the legalization failure, buried in MLIR noise
        key = "failed to legalize operation"
        if key in msg:
            frag = msg[msg.index(key):msg.index(key) + 200]
            out["error"] = f"linalg: {frag.splitlines()[0][:160]}"
        else:
            out["error"] = f"linalg: {type(e).__name__}: {msg[:120]}"
    return out


def render(rows) -> str:
    lines = [f"{'op':26s} {'tier':5s} {'wset':>10s} {'prims':>5s} {'linalg':6s} "
             f"mechanisms / error"]
    for r in rows:
        ws = f"{r['working_set']:,}" if r["working_set"] else "-"
        mech = ",".join(r["mechanisms"] or []) or "-"
        ok = "ok" if r["linalg"] else ("FAIL" if r["linalg"] is False else "-")
        tail = mech if not r["error"] else r["error"]
        lines.append(f"{r['name']:26s} {str(r['tier'] or '-'):5s} {ws:>10s} "
                     f"{str(r['prims'] or '-'):>5s} {ok:6s} {tail}")
    return "\n".join(lines)


# ---- a shortlist to screen. Each entry is (name, module, args, dtype_bytes).
def shortlist():
    R = torch.randn

    class Unary(torch.nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x):
            return self.fn(x)

    class Binary(torch.nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, a, b):
            return self.fn(a, b)

    f32 = (R(256, 256),)
    f32b = (R(256, 256), R(256, 256))
    f16 = (R(256, 256, dtype=torch.float16),)
    return [
        ("erfc_f32", Unary(torch.erfc), f32, 4),
        ("hypot_f32", Binary(torch.hypot), f32b, 4),
        ("atan2_f32", Binary(torch.atan2), f32b, 4),
        ("tan_f32", Unary(torch.tan), f32, 4),
        ("sinh_f32", Unary(torch.sinh), f32, 4),
        ("cosh_f32", Unary(torch.cosh), f32, 4),
        ("asinh_f32", Unary(torch.asinh), f32, 4),
        ("acos_f32", Unary(torch.acos), f32, 4),
        ("atan_f32", Unary(torch.atan), f32, 4),
        ("isnan_f32", Unary(torch.isnan), f32, 4),
        ("logaddexp_f32", Binary(torch.logaddexp), f32b, 4),
        ("log1p_f32", Unary(torch.log1p), f32, 4),
        ("expm1_f32", Unary(torch.expm1), f32, 4),
        ("frac_f32", Unary(torch.frac), f32, 4),
        ("round_f32", Unary(torch.round), f32, 4),
        ("prod_all_f32", Unary(torch.prod), f32, 4),
        ("amax_all_f32", Unary(torch.amax), f32, 4),
        ("mean_all_f16", Unary(lambda t: t.mean()), f16, 2),
        ("norm_rows_f32", Unary(lambda t: torch.linalg.vector_norm(t, dim=-1)), f32, 4),
        ("sigmoid_f16", Unary(torch.sigmoid), f16, 2),
        ("softplus_f32", Unary(torch.nn.functional.softplus), f32, 4),
        ("mish_f32", Unary(torch.nn.functional.mish), f32, 4),
        ("elu_f32", Unary(torch.nn.functional.elu), f32, 4),
        ("selu_f32", Unary(torch.selu), f32, 4),
        ("logsigmoid_f32", Unary(torch.nn.functional.logsigmoid), f32, 4),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="comma-separated names to screen")
    args = ap.parse_args(argv)
    items = shortlist()
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        items = [i for i in items if i[0] in want]
    rows = [screen(*i) for i in items]
    print(render(rows))
    ok = [r for r in rows if r["linalg"] and not r["error"]]
    print(f"\nviable: {len(ok)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
