#!/usr/bin/env bash
# Build + validate the remaining scalar references (batches 48-79, T2/T3).
#
# RUN THIS OUTSIDE CLAUDE CODE -- in a plain terminal. Long jobs get reaped at
# ~15-30 min inside the harness.
#
#   bash build_refs_parallel.sh          # SERIAL: one hexagon-sim at a time
#
# JOBS IS NOW 1 AND THE FILENAME IS A LEFTOVER. Standing instruction from the user
# (2026-08-17): never parallelise hexagon-sim on Windows; one sim thread only. The
# earlier "6-way, parallelism across batches is the entire win" note is superseded.
#
# Two things went wrong while this ran 5-way, both from the concurrency rather than
# the length: it starved everything else on the box, and when the job was killed the
# `run_one` background subshells survived and kept launching FRESH batches, so
# hexagon-sim.exe processes reappeared after every taskkill. Serial has neither.
#
# Resumable: any batch with results.json is skipped, so re-running after an
# interruption costs only the batches that were in flight.
#
# No --timing: stage (g) is a correctness gate against PyTorch goldens, and
# cycle-accurate mode multiplies its cost ~5x for a sim-cycle number PLAN.md
# section 3 says is measured on silicon anyway.

set -u
cd /c/Users/user/NEU/shlabs/hexbench || exit 1

JOBS=1                      # do not raise; see the header
LOGDIR=benchmark/witness_build
mkdir -p "$LOGDIR"

run_one() {
  n=$1
  out="$LOGDIR/batch$n"
  if [ -f "$out/results.json" ]; then
    echo "SKIP  batch$n"
    return 0
  fi
  start=$(date +%s)
  python -m hexforge.run_batch --batch "$n" --out "$out" \
    > "$LOGDIR/batch$n.log" 2>&1
  rc=$?
  echo "DONE  batch$n rc=$rc $(( ($(date +%s)-start)/60 ))min"
}
export -f run_one
export LOGDIR

# SERIAL, and deliberately not backgrounded. The `run_one &` version left orphan
# subshells that outlived the job and relaunched batches behind our back; a plain
# loop dies when the shell dies, which is the behaviour we want.
for n in $(seq 48 79); do
  run_one "$n"
done

echo
echo "=== batches with results.json: $(ls -d $LOGDIR/batch*/results.json 2>/dev/null | wc -l) / 64"
python - <<'PY'
import json, glob, os
ok = tot = 0; bad = []
for f in sorted(glob.glob('benchmark/witness_build/batch*/results.json')):
    r = json.load(open(f))
    for k, e in r['reference'].items():
        tot += 1
        if e.get('correct'): ok += 1
        else: bad.append((os.path.basename(os.path.dirname(f)), k,
                          (e.get('error_text') or '')[:70]))
print(f"scalar references: {tot} built, {ok} pass their PyTorch goldens, {len(bad)} fail")
for b in bad[:20]: print("  FAIL", b)
PY
