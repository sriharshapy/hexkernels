"""Drive the encoder kernel set through forge2's pipeline.

    python -m hexkernels.forge.run_hexlib_encoder --skip-verify   # no toolchain
    python -m hexkernels.forge.run_hexlib_encoder                 # + simulator

A separate driver rather than a numbered batch, for the same reason
`hexlib_encoder.py` is a separate module: these kernels are for a downstream
project and must not land in this benchmark's corpus, its `results.json`, or its
reference cache. Artifacts go to `run_artifacts/hexlib_encoder/`, which no
corpus tool reads.

Stage (g), verify_reference, is the gate that matters. If the emitted scalar
reference cannot pass the harness generated next to it, then the emitter and the
golden disagree, and a candidate judged against that harness tells you about the
harness rather than about the candidate. A kernel whose reference fails is
reported failed and gets no prompt.
"""
import argparse
import json
import os

from hexkernels.forge.hexlib_encoder import encoder_batch
from hexkernels.forge.run_batch import (
    _write_artifacts,
    build,
    load_reference_cache,
    save_reference_cache,
    verify_reference,
)

DEFAULT_OUT = os.path.join("benchmark", "hexlib_encoder")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--skip-verify", action="store_true",
                    help="build artifacts only; no toolchain needed")
    ap.add_argument("--only", default="",
                    help="comma-separated spec names, for iterating on one")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--reference-timeout", type=int, default=900,
                    help="a scalar reference at these shapes is slow by "
                         "design; 3072x3072 is 9.4M MACs per output row")
    args = ap.parse_args(argv)

    specs = encoder_batch()
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = want - {s.name for s in specs}
        if unknown:
            print(f"unknown spec names: {sorted(unknown)}")
            return 2
        specs = tuple(s for s in specs if s.name in want)

    os.makedirs(args.out, exist_ok=True)
    cache = None if args.skip_verify else load_reference_cache(args.out)

    results, failed = {}, []
    for spec in specs:
        art = build(spec)
        stage = art.get("failed_stage")
        if stage:
            # Reported, never swallowed: a spec that cannot be traced or emitted
            # is a spec to fix or to hand-write, and either way the name of the
            # stage that broke is the whole diagnostic.
            err = (art.get("error") or "").splitlines()
            print(f"  {spec.name:34s} FAILED at {stage}: "
                  f"{err[0] if err else '(no message)'}", flush=True)
            failed.append((spec.name, stage, "\n".join(err)))
            results[spec.name] = {"failed_stage": stage}
            continue

        _write_artifacts(art, args.out)
        line = (f"  {spec.name:34s} tier={art.get('tier', '-'):3s} "
                f"prims={len(art.get('primitives') or []):2d}")

        if args.skip_verify:
            results[spec.name] = {"tier": art.get("tier"), "emitted": True}
            print(line + "  emitted", flush=True)
            continue

        ref = verify_reference(art, timing=args.timing, cache=cache,
                               timeout=args.reference_timeout)
        if cache is not None:
            save_reference_cache(args.out, cache)
        results[spec.name] = {"tier": art.get("tier"), **ref}
        print(line + f"  ref_correct={ref.get('correct')}"
                     f"{' (cached)' if ref.get('from_cache') else ''}",
              flush=True)

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    ok = sum(1 for r in results.values() if r.get("correct"))
    emitted = sum(1 for r in results.values() if not r.get("failed_stage"))
    print(f"\n{emitted}/{len(specs)} emitted; {ok} references verified correct")
    if failed:
        print("failed:")
        for name, stage, _ in failed:
            print(f"  {name} at {stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
