#!/usr/bin/env bash
# Finish the rung-0 run: grade T2, build the last T3 references, grade T3, report.
#
#   bash scripts/finish_rung0.sh                 # default model gpt-5.6-luna
#   bash scripts/finish_rung0.sh gpt-5.6-luna
#
# RUN THIS IN A PLAIN TERMINAL. Everything launched from inside Claude Code gets
# reaped within minutes. Nothing is lost when that happens -- each attempt and each
# reference batch persists the moment it completes -- but progress is much faster
# when the job is left alone.
#
# ONE hexagon-sim AT A TIME, THROUGHOUT. Standing instruction: never parallelise the
# simulator on Windows. That is why this is a sequence and not a fan-out, and why the
# wall-clock below is what it is.
#
# EXPECT DAYS, NOT HOURS. Measured on this box: a T0 attempt is seconds, a T2 attempt
# is ~10-20 minutes, and a T3 reference build is slower still (T3 exceeds VTCM, so the
# naive reference streams a working set of >8 MB through a scalar loop nest). Roughly
# ~290 attempts and 6 reference batches remain. It is resumable, so stopping and
# restarting costs only the item in flight.
#
# ORDER, and why this one:
#   1. T2 grading      -- 29 of 32 T2 tasks already have verified references, so this
#                         is the largest block of work that is not blocked on anything.
#   2. T3 references   -- 6 batches (66-70, 72). Unblocks the rest of T3.
#   3. T3 grading      -- everything the step above made gradable.
#   4. Report          -- regenerates SUMMARY.md and attempts.jsonl from the records.
# Steps 1 and 3 are re-run after step 2 anyway, because `grade` skips what is done and
# picks up whatever became gradable in the meantime.

set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="${1:-gpt-5.6-luna}"
SEEDS="${SEEDS:-5}"
LOGDIR="runs/rung0/${MODEL}/logs"
mkdir -p "$LOGDIR"

guard_sims() {
  local n
  n=$(tasklist 2>/dev/null | grep -ci hexagon-sim || true)
  if [ "${n:-0}" -gt 0 ]; then
    echo "REFUSING: $n hexagon-sim process(es) already running."
    echo "  taskkill //IM hexagon-sim.exe //F"
    echo "  Also check for a surviving shell loop; killing the pythons alone is not enough."
    exit 1
  fi
}

grade_tier() {
  local tier=$1 pass
  for pass in 1 2 3 4 5 6 7 8; do
    local log="$LOGDIR/grade-${tier}-pass${pass}.log"
    echo "--- grade $tier pass $pass -> $log"
    python -u -m hexforge.rung0 grade --model "$MODEL" --seeds "$SEEDS" \
      --tier "$tier" > "$log" 2>&1
    grep -E "^(graded|[0-9]+ attempts)" "$log" | tail -2
    if head -1 "$log" | grep -q '^0 attempts'; then
      echo "    $tier converged"
      return 0
    fi
  done
  echo "    $tier still has work after 8 passes; re-run this script"
}

guard_sims

# The facts cache must be warm or every pass spends its first minutes tracing --
# that is the phase the reaper kept killing. Cheap and idempotent.
echo "=== 0. lint facts"
python -u -m hexforge.rung0 facts > "$LOGDIR/facts.log" 2>&1
tail -1 "$LOGDIR/facts.log"

echo "=== 1. grade T2"
grade_tier T2

echo "=== 2. build the last T3 references (serial)"
bash scripts/build_refs_parallel.sh > "$LOGDIR/refs.log" 2>&1
tail -3 "$LOGDIR/refs.log"

echo "=== 3. facts for whatever became gradable, then grade T3"
python -u -m hexforge.rung0 facts >> "$LOGDIR/facts.log" 2>&1
grade_tier T3

echo "=== 4. sweep any tier that gained a reference late"
grade_tier T2

echo "=== 5. report"
python -u -m hexforge.rung0 report --model "$MODEL" --seeds "$SEEDS"
