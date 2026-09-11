#!/usr/bin/env bash
# Finish a rung-2 run: grade, run the retry waves, sweep, report.
#
# ORDERING IS NOT ARBITRARY, and it is `finish_rung1.sh`'s ordering plus one addition:
#
#   1. ONE SIMULATOR, SERIAL. Standing instruction, 2026-08-17. Every `grade` here is
#      `--jobs 1` and they are sequenced, never backgrounded together.
#   2. T0,T1 FIRST. A T0 attempt costs ~1 s and a T3 median ~2.5 min, so grading in
#      arrival order puts hours of big kernels in front of a reportable baseline.
#   3. THE FINAL SWEEP RUNS LAST. It reads each attempt's CURRENT candidate.cpp, and a
#      retry wave rewrites that file.
#   4. NEW AT THIS RUNG: the mechanism sweep is ALSO taken before any grading, because
#      it needs no simulator (~150 s for 640 attempts) and it is the headline number.
#      Rung 1 needed 12 retry turns in 640 attempts, so nothing was at stake there;
#      rung 2 invites the model to write intrinsics, and the simulator cost of the
#      waves cannot be priced in advance. Taking the mechanism reading first means the
#      rung's central result exists before any of that is committed.
#   5. `sweep --all-turns` runs at the very end. It is post-hoc over the saved
#      `candidate_turn<N>.cpp` files, so it has no ordering constraint of its own.
set -uo pipefail

MODEL="${1:-gpt-5.6-luna}"
SEEDS="${2:-5}"
MAX_TURNS="${3:-5}"
# Per-attempt simulator budget. 9000 s, not `grade`'s 900 s default: the heaviest T3
# attempt measured 9.98 BILLION instructions, ~64 min at the measured 2.6 M
# instructions/s. A budget that expires produces an UNGRADED attempt, not a result --
# and a wave grades exactly the attempts a shorter budget already failed, which is how
# rung 1 lost two T3 verdicts to `grade`'s default.
SIM_TIMEOUT="${4:-9000}"
cd "$(dirname "$0")/.."
LOG=runs/rung2

say() { echo; echo "=== $* ==="; }

wave_until_settled() {
  local tier_arg="$1" label="$2"
  for turn in $(seq 2 "$MAX_TURNS"); do
    say "$label turn $turn: re-asking failed attempts"
    python -m hexforge.rung2 turn --model "$MODEL" --seeds "$SEEDS" \
      --max-turns "$MAX_TURNS" $tier_arg >> "${LOG}_turn.log" 2>&1
    if tail -5 "${LOG}_turn.log" | grep -q "nothing to retry"; then
      echo "settled before turn $turn"; break
    fi
    say "$label turn $turn: re-grading"
    python -m hexforge.rung0 grade --rung 2 --model "$MODEL" --seeds "$SEEDS" \
      --jobs 1 --timeout "$SIM_TIMEOUT" $tier_arg >> "${LOG}_grade.log" 2>&1
  done
}

say "mechanism sweep BEFORE grading: the headline, no simulator, ~150 s"
python -m hexforge.rung0 sweep --rung 2 --model "$MODEL" --seeds "$SEEDS" \
  > "${LOG}_sweep_turn1.log" 2>&1
tail -20 "${LOG}_sweep_turn1.log"

say "grade T0,T1 (cheap tiers first)"
python -m hexforge.rung0 grade --rung 2 --model "$MODEL" --seeds "$SEEDS" \
  --jobs 1 --tier T0,T1 --timeout "$SIM_TIMEOUT" > "${LOG}_grade.log" 2>&1

wave_until_settled "--tier T0,T1" "T0,T1"

say "grade T2,T3 (the long pole: T2 median ~25 s, T3 ~2.5 min, tail ~1 h)"
python -m hexforge.rung0 grade --rung 2 --model "$MODEL" --seeds "$SEEDS" \
  --jobs 1 --tier T2,T3 --timeout "$SIM_TIMEOUT" >> "${LOG}_grade.log" 2>&1

wave_until_settled "" "all tiers"

say "final sweep (no simulator; must follow the waves, see header)"
# --redo IS MANDATORY HERE, unlike the early sweep above. `cmd_sweep` skips any
# attempt that already carries `mechanism_scan`, and the early sweep above already
# wrote that field for every attempt before the waves ran. Without --redo this sweep
# would scan nothing and the published mechanism_scan would still describe the
# pre-retry candidates while the correctness verdict describes the post-retry ones.
python -m hexforge.rung0 sweep --rung 2 --model "$MODEL" --seeds "$SEEDS" --redo \
  > "${LOG}_sweep.log" 2>&1

say "per-turn mechanism curve (post-hoc, reads candidate_turn<N>.cpp)"
python -m hexforge.rung0 sweep --rung 2 --model "$MODEL" --seeds "$SEEDS" \
  --all-turns > "${LOG}_sweep_turns.log" 2>&1

say "report"
python -m hexforge.rung0 report --rung 2 --model "$MODEL" --seeds "$SEEDS" \
  > "${LOG}_report.log" 2>&1
python -m hexforge.rung2 histogram --model "$MODEL" --seeds "$SEEDS" \
  > "${LOG}_histogram.json" 2>&1

say "RUNG 2 DONE"
tail -32 "runs/rung2/${MODEL}/SUMMARY.md" 2>/dev/null
echo
echo "turns and failure stages:"; cat "${LOG}_histogram.json"
