#!/usr/bin/env bash
# Grade a rung-0 run on the simulator, ONE SIMULATOR AT A TIME.
#
#   bash scripts/grade_rung0_parallel.sh gpt-5.6-luna
#   PASSES=8 bash scripts/grade_rung0_parallel.sh gpt-5.6-luna
#
# THERE IS NO JOBS KNOB, and the name of this file is now a lie kept only so the
# handoff notes still point somewhere. Standing instruction from the user
# (2026-08-17): never parallelise hexagon-sim on Windows; use one sim thread. So
# grading is serial and `hexforge.rung0 grade` REFUSES --jobs above 1.
#
# What this buys, beyond following the instruction: the two failure modes seen while
# grading was parallel both came from concurrency, not from length. Several
# simulators alongside the reference build starved the box, and orphaned
# hexagon-sim.exe processes outlived every kill while a surviving shell loop kept
# launching more.
#
# Why a loop is still here: every attempt persists its verdict the moment it lands,
# so a killed pass costs only the attempt in flight and re-running skips everything
# already graded. Serial grading is slow, so being interrupted is expected.
#
# Run `python -m hexforge.rung0 facts` FIRST if the lint-facts cache is cold --
# otherwise each pass spends its first minutes tracing instead of grading, and that
# is the phase the reaper kept killing.
#
# No --timing: PLAN.md section 3 measures cycles on silicon, so simulation here is a
# correctness-and-mechanism gate. Cycle-accurate mode costs ~5x for a number that is
# not the one being reported.
#
# Attempts whose task has no verified reference are SKIPPED, not failed. Re-run after
# the T3 references finish building and they get graded then.

set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="${1:-gpt-5.6-luna}"
SEEDS="${SEEDS:-5}"
PASSES="${PASSES:-8}"
LOGDIR="runs/rung0/${MODEL}/logs"
mkdir -p "$LOGDIR"

# One sim at a time means ANY other simulator is a violation, not just contention.
running=$(tasklist 2>/dev/null | grep -ci hexagon-sim || true)
if [ "${running:-0}" -gt 0 ]; then
  echo "REFUSING: $running hexagon-sim process(es) are already running."
  echo "  Stop them first:  taskkill //IM hexagon-sim.exe //F"
  echo "  (Check for a surviving build_refs_parallel.sh loop too -- it relaunches them.)"
  exit 1
fi

for pass in $(seq 1 "$PASSES"); do
  log="$LOGDIR/grade-pass$pass.log"
  echo "=== pass $pass/$PASSES (serial) -> $log"
  python -u -m hexforge.rung0 grade --model "$MODEL" --seeds "$SEEDS" \
    --jobs 1 > "$log" 2>&1
  rc=$?
  grep -E "^(graded|[0-9]+ attempts)" "$log" | tail -2
  if head -1 "$log" | grep -q '^0 attempts'; then
    echo "converged on pass $pass (rc=$rc)"
    break
  fi
done

echo
python -u -m hexforge.rung0 report --model "$MODEL" --seeds "$SEEDS"
