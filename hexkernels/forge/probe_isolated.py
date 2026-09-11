"""Probe the harvest with each row's crash attributed to that row.

    python -m hexkernels.forge.probe_isolated --out run_artifacts/forge2/coverage_synth.json

WHY THIS EXISTS
---------------
`coverage.scan` measures which harvested ops this pipeline can express, and its
biggest unreached bucket is one it already has code for. Of 824 `unprobeable` rows,
**676 are "required arguments are not 1-2 plain Tensors"** -- exactly what
`coverage.synth_signature` was written to handle. It is off by default
(`ALLOW_SYNTH_DEFAULT = False`) for one reason, recorded in that module:

    "measuring it produced a SEGFAULT. The rows it newly reaches include
     `_ctc_loss`, `_cholesky_solve_helper`, `_convert_weight_to_int4pack` and
     `_dyn_quant_matmul_4bit`"

A segfault in an in-process scan is not a failed row, it is a lost scan: the
interpreter dies and every verdict already computed dies with it. So the extension
that would unlock the largest part of the registry cannot be run.

THE DESIGN, AND WHY IT IS NOT ONE SUBPROCESS PER ROW
---------------------------------------------------
`coverage.py`'s own note proposes "running each probe in its own process". That
works and costs 1,170 torch imports -- roughly four seconds each, so over an hour of
pure startup. Instead a child probes a RUN of rows and appends each verdict to a
file, flushed, before starting the next one. Then:

  * a child that exits cleanly has reported every row it was given;
  * a child that dies reported everything up to the row it died on, and THAT ROW is
    the crash -- it is the first one with no verdict on disk;
  * the parent records the crash against that row and respawns from the next one.

So crash attribution is exact and the common case pays one import per run rather
than one per row. This is the same argument as a resumable batch driver: the unit of
loss must be a row, not the scan.

FLUSHING IS LOAD-BEARING. Without a flush per verdict the child's buffer is lost on
a segfault, which would attribute the crash to the first row of the run instead of
the guilty one -- and every subsequent run would start in the wrong place. That is
the difference between this module working and it producing confident nonsense.
"""
import argparse
import json
import os
import subprocess
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Run inside the child. Probes the rows it is given, one JSON object per line,
#: FLUSHED after each -- see the module docstring on why.
_CHILD = textwrap.dedent('''
    import json, sys
    sys.path.insert(0, r"{repo}")
    from hexkernels.forge import coverage as C

    rows = json.loads(open(r"{rows}", encoding="utf-8").read())
    lo = int(sys.argv[1])
    out = open(r"{sink}", "a", encoding="utf-8")
    for i in range(lo, len(rows)):
        row = rows[i]
        key = "{{}}.{{}}".format(row["op"], row.get("overload") or "default")
        try:
            status, detail = C.probe(row, allow_synth={allow_synth})
        except Exception as exc:
            status, detail = "unprobeable", "{{}}: {{}}".format(
                type(exc).__name__, str(exc)[:200])
        out.write(json.dumps({{"i": i, "key": key, "status": status,
                              "detail": detail}}) + "\\n")
        out.flush()
    out.close()
''')


def eligible_rows(harvest_path=None):
    """The rows `coverage.scan` would probe, in its order, deduplicated by op."""
    from hexkernels.forge.coverage import _overload
    from hexkernels.forge.mechanism import mechanism_eligible
    from hexkernels.forge.provenance import DEFAULT_HARVEST, load_harvest
    rows = load_harvest(harvest_path or DEFAULT_HARVEST)
    seen, uniq = set(), []
    for r in rows:
        if not mechanism_eligible(r):
            continue
        key = (r["op"], _overload(r))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


def scan_isolated(rows, *, workdir, python_exe=None, allow_synth=True,
                  timeout=600, max_respawn=None, log=print, child_src=None):
    """Probe `rows`, surviving a crash in any of them.

    Returns `(verdicts, crashes)` where `verdicts` maps row index to
    `{"key", "status", "detail"}` and `crashes` is the list of indices whose probe
    killed the interpreter.
    """
    python_exe = python_exe or sys.executable
    os.makedirs(workdir, exist_ok=True)
    rows_path = os.path.join(workdir, "rows.json")
    sink = os.path.join(workdir, "verdicts.jsonl")
    with open(rows_path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    # Fresh sink: a stale one would be read as this scan's own progress and the
    # parent would skip rows it never probed.
    open(sink, "w", encoding="utf-8").close()

    # `child_src` is an injection point for the TESTS, and it exists because the
    # crash-attribution deduction is the only thing in this module that can be
    # wrong in a way that produces confident nonsense -- it has to be exercised
    # against a child that really dies, which no real probe can be made to do on
    # demand. Production callers never pass it.
    src = child_src or _CHILD.format(repo=REPO, rows=rows_path, sink=sink,
                                     allow_synth=bool(allow_synth))
    if child_src:
        src = src.format(rows=rows_path, sink=sink)
    child_path = os.path.join(workdir, "child.py")
    with open(child_path, "w", encoding="utf-8") as f:
        f.write(src)

    verdicts, crashes = {}, []
    lo, spawns = 0, 0
    cap = max_respawn if max_respawn is not None else len(rows) + 8
    while lo < len(rows) and spawns <= cap:
        spawns += 1
        try:
            subprocess.run([python_exe, child_path, str(lo)],
                           capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            pass                      # handled below like any other early exit
        # Read everything reported so far, whatever the child's fate.
        with open(sink, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue          # a partial line from a crash mid-write
                verdicts[rec["i"]] = {k: rec[k] for k in ("key", "status", "detail")}
        nxt = lo
        while nxt in verdicts:
            nxt += 1
        if nxt >= len(rows):
            lo = nxt
            break
        if nxt == lo and lo in verdicts:
            nxt = lo + 1              # defensive: never fail to advance
        if nxt not in verdicts:
            # THE CRASH IS THE FIRST UNREPORTED ROW. The child wrote and flushed a
            # verdict for everything before it, so this is the one that took the
            # interpreter down -- not a guess, a deduction from what is on disk.
            r = rows[nxt]
            key = f"{r['op']}.{r.get('overload') or 'default'}"
            verdicts[nxt] = {"key": key, "status": "crashed",
                             "detail": "probing this row killed the interpreter "
                                       "(segfault, abort or timeout)"}
            crashes.append(nxt)
            log(f"  CRASH at row {nxt}: {key}")
            nxt += 1
        lo = nxt
    return verdicts, crashes


def summarise(rows, verdicts):
    """Fold per-row verdicts into `coverage.scan`'s result shape."""
    out = {"covered": {}, "blocked": {}, "unprobeable": {}, "out_of_scope": {},
           "covered_synth": {}, "blocked_synth": {}, "crashed": {},
           "total": len(rows)}
    for i, v in sorted(verdicts.items()):
        bucket = v["status"] if v["status"] in out else "unprobeable"
        out[bucket][v["key"]] = v["detail"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="where to write the result JSON")
    ap.add_argument("--workdir", default=None,
                    help="scratch for the child script and the verdict log")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-synth", action="store_true",
                    help="probe without invented argument values, as scan() does "
                         "by default -- for comparing the two")
    ap.add_argument("--timeout", type=int, default=600,
                    help="seconds per child; a child probes many rows, so this "
                         "bounds a HANG rather than a single probe")
    args = ap.parse_args(argv)

    rows = eligible_rows()
    if args.limit:
        rows = rows[:args.limit]
    workdir = args.workdir or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "_probe_isolated")

    print(f"probing {len(rows)} eligible rows, synth="
          f"{not args.no_synth}, crash-isolated")
    verdicts, crashes = scan_isolated(
        rows, workdir=workdir, allow_synth=not args.no_synth,
        timeout=args.timeout)
    res = summarise(rows, verdicts)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True)

    print()
    for k in ("covered", "covered_synth", "blocked", "blocked_synth",
              "unprobeable", "out_of_scope", "crashed"):
        print(f"  {k:16s} {len(res[k])}")
    print(f"  {'total':16s} {res['total']}")
    if crashes:
        print(f"\n{len(crashes)} row(s) crashed the interpreter and are attributed "
              "individually -- that is the whole point of this module, and in an "
              "in-process scan every one of them would have lost the entire run.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
