#!/usr/bin/env bash
# Finish a rung-1 run: grade, run the retry waves, sweep, report.
#
# ORDERING IS NOT ARBITRARY. Three constraints shape it:
#
#   1. ONE SIMULATOR, SERIAL. Standing instruction, 2026-08-17. Every `grade` here is
#      `--jobs 1` by default and they are sequenced, never backgrounded together.
#   2. T0,T1 FIRST. A T0 attempt costs ~1 s and a T3 median ~2.5 min, so grading in
#      arrival order puts hours of big kernels in front of a reportable baseline that is
#      otherwise minutes away.
#   3. THE SWEEP RUNS LAST. It reads each attempt's CURRENT candidate.cpp, and a retry
#      wave rewrites that file. Sweeping before the waves would report mechanism flags
#      for candidates that no longer exist.
#
# Waves are bounded by --max-turns (PLAN.md section 2 allows 5). `rung1 turn` is a no-op
# once every graded attempt is either correct or out of budget, so the loop is safe to
# run to its bound: it costs one cheap pass over the records, not API calls.
set -uo pipefail

MODEL="${1:-gpt-5.6-luna}"
SEEDS="${2:-5}"
MAX_TURNS="${3:-5}"
# Per-attempt simulator budget. 9000 s, not `grade`'s 900 s default: the heaviest T3
# attempt measured 9.98 BILLION instructions, which is ~64 min at the measured 2.6 M
# instructions/s -- so even 3600 s cuts it close, and a budget that expires produces an
# UNGRADED attempt rather than a result.
SIM_TIMEOUT="${4:-9000}"
cd "$(dirname "$0")/.."
LOG=runs/rung1

say() { echo; echo "=== $* ==="; }

# A wave, then re-grade whatever it rewrote. Scoped by tier so the cheap tiers can
# converge before the expensive ones start.
wave_until_settled() {
  local tier_arg="$1" label="$2"
  for turn in $(seq 2 "$MAX_TURNS"); do
    say "$label turn $turn: re-asking failed attempts"
    python -m hexforge.rung1 turn --model "$MODEL" --seeds "$SEEDS" \
      --max-turns "$MAX_TURNS" $tier_arg >> "${LOG}_turn.log" 2>&1
    if tail -5 "${LOG}_turn.log" | grep -q "nothing to retry"; then
      echo "settled before turn $turn"; break
    fi
    say "$label turn $turn: re-grading"
    # THE SAME BUDGET AS THE MAIN PASS. This inherited `grade`'s 900 s default once,
    # which is below what the heaviest T3 attempts need (`b74i0__euclidean_dist__all`
    # ran 7.8-10.0 billion instructions at rung 0, ~50-64 min at the measured 2.6 M
    # instructions/s). Two attempts hit the cap and came out UNGRADED -- correct
    # behaviour by `grade_one`'s timeout guard, and still a hole in the numbers,
    # because a wave grades exactly the attempts a shorter budget already failed.
    python -m hexforge.rung0 grade --rung 1 --model "$MODEL" --seeds "$SEEDS" \
      --jobs 1 --timeout "$SIM_TIMEOUT" $tier_arg >> "${LOG}_grade.log" 2>&1
  done
}

say "grade T0,T1 (cheap tiers first)"
python -m hexforge.rung0 grade --rung 1 --model "$MODEL" --seeds "$SEEDS" \
  --jobs 1 --tier T0,T1 > "${LOG}_grade.log" 2>&1

wave_until_settled "--tier T0,T1" "T0,T1"

say "grade T2,T3 (the long pole: T2 median ~25 s, T3 ~2.5 min, tail ~1 h)"
python -m hexforge.rung0 grade --rung 1 --model "$MODEL" --seeds "$SEEDS" \
  --jobs 1 --tier T2,T3 --timeout "$SIM_TIMEOUT" >> "${LOG}_grade.log" 2>&1

wave_until_settled "" "all tiers"

say "sweep (no simulator; must follow the waves, see header)"
python -m hexforge.rung0 sweep --rung 1 --model "$MODEL" --seeds "$SEEDS" \
  > "${LOG}_sweep.log" 2>&1

say "report"
python -m hexforge.rung0 report --rung 1 --model "$MODEL" --seeds "$SEEDS" \
  > "${LOG}_report.log" 2>&1
python -m hexforge.rung1 histogram --model "$MODEL" --seeds "$SEEDS" \
  > "${LOG}_histogram.json" 2>&1

say "RUNG 1 DONE"
tail -32 "runs/rung1/${MODEL}/SUMMARY.md" 2>/dev/null
echo
echo "turn histogram:"; cat "${LOG}_histogram.json"
