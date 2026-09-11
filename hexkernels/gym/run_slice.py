"""Box runner: run an agent over a mechanism slice with a real checkpoint, print the
per-mechanism scoreboard (reliable) + a box-gated speed note. Heavy imports live in main()."""
import argparse
import json
import os
import tempfile

from hexkernels.gym import agents as A
from hexkernels.gym import scoreboard as SB
from hexkernels.gym.env import HexagonKernelEnv, load_task_from_disk


def slice_task_ids(target_mechs, root="data/v6/tasks"):
    ids = []
    for tid in sorted(os.listdir(root)):
        p = os.path.join(root, tid, "spec.json")
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            mechs = (json.load(f).get("mechanisms") or [])
        if any(m in mechs for m in target_mechs):
            ids.append(tid)
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--agent", choices=["single_shot", "multiturn", "best_of_n"], default="single_shot")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mechs", nargs="+", default=["dma", "vtcm"])
    ap.add_argument("--sdk-root", default=os.environ.get("HEXAGON_SDK_ROOT"))
    ap.add_argument("--timing", dest="timing", action="store_true", default=True)
    ap.add_argument("--no-timing", dest="timing", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    # Heavy imports here so `import hexkernels.gym.run_slice` stays offline-safe.
    from hexkernels.core.generator import VLLMGenerator
    from hexkernels.core.fleet import probe_fleet_size
    from hexkernels.core import evaluate as _ev
    from hexkernels.core.toolchain import DEFAULT_SDK_ROOT

    sdk_root = args.sdk_root or DEFAULT_SDK_ROOT

    gen = VLLMGenerator(model_id=args.model, adapter=args.adapter) \
        if args.adapter else VLLMGenerator(model_id=args.model)
    call_fn = lambda msgs: gen.chat(msgs)

    width = probe_fleet_size()  # fleet_width=0 semantics: let the probe choose
    def gate_fn(task_id, src):
        fd, path = tempfile.mkstemp(suffix=".c")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(src or "")
            return _ev.evaluate(path, task_id=task_id, sdk_root=sdk_root, timing=args.timing)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    env = HexagonKernelEnv(gate_fn=gate_fn, task_loader=load_task_from_disk,
                           max_turns=3, on_box=True)

    records = []
    ids = slice_task_ids(args.mechs)
    if args.limit:
        ids = ids[:args.limit]
    for tid in ids:
        _, spec = load_task_from_disk(tid)
        if args.agent == "single_shot":
            rec = A.single_shot(env, tid, call_fn)
        elif args.agent == "multiturn":
            rec = A.pick_best(A.multiturn_repair(env, tid, call_fn))
        else:
            rec = A.best_of_n(env, tid, call_fn, args.n, spec)
        records.append({"task_id": tid, "spec": spec, "feedback": rec["feedback"]})

    sb = SB.scoreboard(records)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"agent": args.agent, "scoreboard": sb, "fleet_width": width}, f, indent=2)
    print(json.dumps(sb, indent=2))


if __name__ == "__main__":
    main()
