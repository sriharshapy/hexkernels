"""Check one accelerated candidate against its kernel's generated harness.

    python -m hexkernels.forge.check --batch 1 --name softmax_fp32 --file cand.cpp

The tight loop for whoever is writing the accelerated kernel, human or model.
It rebuilds the harness from the batch definition rather than reading a cached
copy, so a candidate is always judged against the golden values for the kernel
it claims to implement -- a stale harness would silently judge it against
another shape.

Prints the verdict and exits non-zero unless the candidate compiled, ran and
was correct. Mechanism flags are printed but never gate: a correct kernel that
did not touch the accelerator is a result worth seeing, not an error.
"""
import argparse
import sys

from hexkernels.forge import lint as _lint
from hexkernels.forge import provenance as _prov
from hexkernels.forge.kernels import batch as _batch
from hexkernels.forge.run_batch import build
from hexkernels.forge.verify import verify


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--name", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args(argv)

    spec = next((s for s in _batch(args.batch) if s.name == args.name), None)
    if spec is None:
        print(f"no kernel {args.name!r} in batch {args.batch}", file=sys.stderr)
        return 2

    art = build(spec)
    if art["failed_stage"]:
        print(f"cannot judge: reference build failed at {art['failed_stage']}\n"
              f"{art['error']}", file=sys.stderr)
        return 2

    with open(args.file, encoding="utf-8") as f:
        src = f.read()

    # Provenance before correctness, and before compiling. If the author has
    # reached into the R&D side, there is no verdict worth producing -- a passing
    # one would only make the kernel harder to discard. Reported here rather than
    # only in the batch driver so the tight loop tells you immediately.
    leaks = _prov.rd_leaks(src)
    if leaks:
        print(f"REJECTED: {args.name} references R&D-only material {leaks}",
              file=sys.stderr)
        print("Nothing under hexbench/env/harness/ is available to a forge v2 kernel: "
              "hmx_helpers.h (tile-matmul helpers lifted from expert solutions) and "
              "harness_common.h (VTCM aperture, alignment, crouton offset) are both "
              "R&D-only, and their directory is not on the compile's include path. "
              "Derive from the reference, the schedule, and the vendor headers "
              "(<hexagon_types.h>, <hexagon_protos.h>, <hvx_hexagon_protos.h>, "
              "<hmx_hexagon_protos.h>).", file=sys.stderr)
        return 3

    # THE STATIC GATE BELONGS ON THIS PATH ABOVE ALL OTHERS. It was wired into
    # `run_batch.verify_candidates` first and not here -- which is backwards: this
    # is the loop an author actually iterates in, so every rule was bypassed
    # exactly where it would have saved the most time, and only applied at batch
    # time once the mistake had already been made several times.
    #
    # A guard that is not on the path people use is not a guard.
    findings = _lint.lint(src, signature=f'extern "C" {art["signature"]}',
                          tier=art.get("tier"),
                          mechanisms=art.get("mechanisms") or (),
                          nelem=art.get("nelem"), dtype=art.get("dtype"))
    for f in findings:
        if f.severity == "warn":
            print(f"warn: {f.rule}: {f.message}"
                  + (f"  [{f.evidence}]" if f.evidence else ""), file=sys.stderr)
    blocking = _lint.errors(findings)
    if blocking:
        print(f"REJECTED by the static gate, not compiled: {args.name}",
              file=sys.stderr)
        print(_lint.format_findings(blocking), file=sys.stderr)
        print("\nEach of these has produced a crash or silently wrong data in "
              "this corpus before; see hexbench/forge2/lint.py for which.",
              file=sys.stderr)
        return 3

    v = verify(src, art["harness_c"], name=f"{args.name}_cand", timing=args.timing)

    print(f"signature expected : extern \"C\" {art['signature']}")
    print(f"compiled : {v['compiled']}")
    print(f"ran      : {v['ran']}")
    print(f"correct  : {v['correct']}")
    used = sorted(k for k, on in (v["mechanisms"] or {}).items() if on)
    print(f"mechanisms used     : {', '.join(used) or 'none'}")
    print(f"mechanisms entitled : {', '.join(sorted(art['mechanisms'])) or 'none'}")
    if v["error_text"]:
        print(f"\nerror:\n{v['error_text'][:4000]}")
    return 0 if v["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
