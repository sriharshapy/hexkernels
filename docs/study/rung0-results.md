# Rung 0 baseline — measured results

**Status: COMPLETE for every attempt that can be graded** (2026-08-20). The
**mechanism** half covers all 128 eval-core tasks and all four tiers. The
**correctness** half covers all four tiers too: 570 of 640 attempts hold a verdict,
and the 70 that do not are blocked on their *reference*, not on simulator time —
their reference never passed stage (g), so there is nothing to grade them against.

- Model `gpt-5.6-luna` · eval set `core128` · schema v1 · 5 seeds · rung 0
- Generated 2026-08-17 (128/128 tasks, 640 attempts, 0 failures)
- Live numbers: `runs/rung0/gpt-5.6-luna/SUMMARY.md`; raw records `attempts.jsonl`
- Design, prompt policy, enforced withheld set: [`rung0.md`](rung0.md)

**Which ladder "T0–T3" means here.** Every tier below is a **task tier** — the size
of the working set (T0 fits L1D, T1 reaches L2, T2 exceeds L2 but fits VTCM, T3
exceeds VTCM). The README also uses T0–T3 for an unrelated **outcome ladder**
(doesn't compile / compiles-wrong / correct-but-scalar / correct-and-using-the-
mechanism). The two have been mistaken for each other; always say which.

## The mechanism result — complete, and it is the headline

From static disassembly of a `-c` compile per attempt (~0.4 s each; 636 attempts in
148 s). No simulator, so this half was never gated on the serial-simulation cost.

| | count |
|---|---|
| attempts scanned (128 tasks, all 4 tiers) | 636 |
| **attempts naming any `Q6_` intrinsic** | **0** |
| attempts where an entitled mechanism fired | 3 |
| — of those, model-written intrinsics | **0** |
| — of those, compiler auto-vectorisation | **3** |
| `genuine` ruled out without executing anything | 633 |
| could not be compiled, so make no mechanism claim | 4 |

| tier | scanned | mechanism fired | named any `Q6_` |
|---|---|---|---|
| T0 | 159 | 0 | 0 |
| T1 | 160 | 0 | 0 |
| T2 | 159 | 0 | 0 |
| T3 | 158 | 3 | 0 |

**Unaided, the model never once asks for the accelerator.** Zero of 636 attempts,
across every task in the eval core, contain a single Hexagon vector intrinsic. That
is the strongest form the rung-0 claim could take: the zero is not the compiler
declining to select an instruction, and not the detector being strict — nothing was
attempted.

### The auto-vectorisation confound, and why it is reported separately

Three T3 attempts (`fp32_pdist` seeds 2 and 3, `fp32_tensordot` seed 3) fired
`hvx_compute` and were correct, so they score `genuine`. **None of them contains an
intrinsic.** hexagon-clang auto-vectorised a scalar loop.

The anti-cheat is behaving correctly — the binary really does contain HVX compute,
which is exactly why it reads the ELF rather than the source. But it cannot tell
*who* caused the instruction to exist, and the paper's claim is about the model. So
the count is split rather than merged: crediting the compiler's work to the model
would overstate the thesis in the one direction a reviewer will probe.

Both readings are supported by the data (`uses_any_q6_intrinsic` is on every
record). **Open item: decide whether the paper counts auto-vectorised kernels as
genuine.** Recommended: report `genuine` as measured, and report the
intrinsic-authored subset beside it, because "the model engaged the hardware" and
"the model's scalar code happened to vectorise" are different claims.

## The correctness result — all four tiers

| tier | graded | ungraded | compiled | correct | genuine |
|---|---|---|---|---|---|
| T0 | 160/160 | 0 | 99.4% | 99.4% | 0.0% |
| T1 | 160/160 | 0 | 100.0% | 100.0% | 0.0% |
| T2 | 145/160 | 15 | 100.0% | 97.9% | 0.0% |
| T3 | 105/160 | 55 | 98.1% | 96.2% | 2.9% |
| **pooled** | **570/640** | 70 | 567 (99.5%) | **562 (98.6%)** | 3 (0.5%) |

Every ungraded attempt is one of the 70 whose reference never passed stage (g). No
attempt is ungraded for want of simulator time.

**562 of 570 correct (98.6%), 3 genuine (0.5%).** Correctness sees near-total success
on the same attempts where mechanism engagement sees nothing. That gap is the
instrument's entire claim, and it holds at every tier: T0/T1 are 319 of 320 correct
with 0 genuine, and T2/T3 stay above 96% correct while remaining at 0 model-written
intrinsics.

Supporting properties:

1. **Not a seed artifact.** Per-seed correct counts range 110–114 of 114, a spread of
   4; T0/T1 alone are 63–64 of 64. 106 of 113 tasks are solved on every seed and 113
   of 113 on at least one. PLAN.md §2 warns re-runs have moved correctness −7/+3
   tasks; nothing like that appears here.
2. **Not a copying artifact.** Median similarity to the naive scalar reference is
   0.136 (max 0.630), and rung 0 never shows that reference. The model wrote its own
   scalar code. This is the recorded answer to the "you showed it scalar code"
   objection.
3. **The failures are the model's.** The one T0/T1 non-compile used `M_PI`, which is
   not a standard C++ identifier.

## pass@k, and why correctness pass@k is the wrong headline

Unbiased estimator, pass@k = 1 - C(n-c, k)/C(n, k), averaged over the 114 tasks that
have a verdict, n = 5 samples each. Ungraded attempts are excluded, never counted as
failures.

| k | correct | genuine |
|---|---|---|
| 1 | 98.6% | 0.5% |
| 2 | 99.9% | 1.0% |
| 3 | **100.0%** | 1.3% |
| 4 | 100.0% | 1.6% |
| 5 | 100.0% | 1.8% |

**Correctness pass@k saturates at k=3.** A benchmark reporting correctness alone would
show this hardware as solved, unaided, at rung 0. That single row is the argument for
`genuine` as the headline metric.

**The genuine column is not a capability curve.** Its rise from 0.5% to 1.8% is three
compiler auto-vectorisations spread over two tasks (`fp32_pdist`, `fp32_tensordot`).
Nothing the model does improves with k -- more samples buy more chances for
hexagon-clang to vectorise a scalar loop it was handed. Do not read it as evidence of
latent ability.

## Token cost, per task and per tier

640 attempts, 0 failures: **304,825 in / 834,466 out**, of which 684,205 (82.0%) are
reasoning tokens. Nothing was cached -- every rung-0 prompt is under the ~1024-token
floor that triggers automatic prompt caching (median 463, max 609).

| | input | output |
|---|---|---|
| per task (5 seeds) | 2,381 | 6,519 |
| per attempt | 476 | 1,304 |
| per-attempt median | 463 (409-609) | 1,110 (167-4,179) |

| tier | in/task | out/task |
|---|---|---|
| T0 | 2,366 | 5,198 |
| T1 | 2,359 | 5,440 |
| T2 | 2,422 | 7,248 |
| T3 | 2,379 | 8,191 |

Input is flat across tiers (the prompt barely grows with the task) while output climbs
**58%** from T0 to T3. The model does not read more; it thinks more.

## What these numbers do NOT say

- **0 hallucinated intrinsics is not evidence of ISA knowledge.** It follows
  mechanically from never naming an intrinsic. The metric only becomes informative
  at rungs 1–3.
- **No cycle claim is made, by design.** Grading runs without `--timing`: the
  simulator is a correctness and mechanism gate, and timings are measured on silicon
  (§3, ratified 2026-08-17). `pcycles` is recorded but is not a cycle number.
- **The mechanism flags are valid without `--timing`, including vtcm**, because this
  path uses the STATIC detector (`hexkernels/forge/verify.py` → `anticheat._disasm_has_vtcm`),
  not the runtime PMU counter the README warns about. That static-vs-PMU asymmetry is
  the PLAN.md §5.F disclosure.
- **One model, one rung.** Only an OpenAI key is available, so the four-model ladder
  is one model. There is no rung-1/2/3 delta yet: this is a floor, not a comparison.
- **Ungraded is not failed.** 70 attempts have no verdict because their reference
  never passed stage (g), and `aggregate.summarise` refuses to fold them into a rate.
  Nothing is ungraded for want of simulator time any more, and no timeout is counted
  as a wrong answer (see correction 4 below).

## Reproducing it

```bash
python -m hexkernels.forge.rung0 generate --model gpt-5.6-luna --seeds 5 --jobs 8
python -m hexkernels.forge.rung0 sweep    --model gpt-5.6-luna   # mechanisms, ~3 min total
python -m hexkernels.forge.rung0 facts                           # pre-trace once
python -m hexkernels.forge.rung0 grade    --model gpt-5.6-luna --tier T0,T1   # serial
python -m hexkernels.forge.rung0 report   --model gpt-5.6-luna
python -m hexkernels.forge.rung0 grade    --model gpt-5.6-luna --tier T2,T3   # serial, hours
python -m hexkernels.forge.rung0 report   --model gpt-5.6-luna
```

If `benchmark/witness_build/` is missing, restore it first — seconds per batch, not a
rebuild. See "the reference builds were lost" below.

```bash
python -m hexkernels.forge.run_batch --batch N --out benchmark/witness_build/batchN --skip-verify
python scripts/restore_reference_stamp.py --model gpt-5.6-luna --only <task keys> --apply
```

Generation cost 304,825 prompt + 834,466 completion tokens (684,205 reasoning) for
640 attempts, 0 failures. Dollar cost is `null` unless `--price-in`/`--price-out` are
given — deliberately, so no stale price table can look like a measured number.

## What correctness actually cost, corrected

Cost is driven by **arithmetic intensity, not footprint**, and simulation is serial by
standing instruction (one `hexagon-sim`).

**The earlier version of this section overstated T2 by ~35×, and the error is worth
recording.** It quoted a T2 median of 2,176,061,750 instructions and "~12 min per
attempt" — but at the time only 19 of 145 T2 attempts had run, and those 19 were
precisely the ones that had **exceeded the 900 s timeout**. The sample WAS the slow
tail. Over all 570 graded attempts:

| tier | median instructions | median wall-clock | slowest attempt |
|---|---|---|---|
| T0 | 50,884 | ~1 s | 4,526,502 |
| T1 | 1,966,226 | ~5 s | 35,521,438 |
| T2 | 61,657,758 | ~25 s | 3,549,133,371 |
| T3 | 390,024,431 | ~2.5 min | 9,978,727,512 |

Throughput measured on the re-graded tail: 1,328,767,535 instructions in ~510 s, so
**~2.6 M instructions/s**. Wall-clock above is derived from that rate.

So the distribution is heavily skewed, not uniformly expensive: a median T2 attempt
costs 25 seconds, while the worst T3 attempt costs about an hour. The 19 attempts the
900 s cap failed had a median of 3.5 billion instructions — 57× the T2 median — which
is why raising the budget to 3600 s was doing real work and not merely relabelling:
one of them needed 18.7 min and would have overrun 900 s even uncontended.

`--timing` would multiply all of this ~5×.

**Sizes cannot be cut to make this cheaper.** The tier *is* the size —
`mechanism.py` grants `dma`/`vtcm` on `ws > L2`, and `size_for_tier` already takes
the first ladder rung that crosses the threshold. Measured: T2 sits at a median
1.13× its floor (minimum 1.00×, four bytes over), T3 at 1.19×. Shrinking a T2 task
below 1 MB makes it T1 and *removes* the entitlement, which is the property the tier
exists to create.

**QDC would not unblock this either.** Device execution is ~1000× faster, but the
silicon path is unbuilt (five unchecked gates in §5.B, and 8 of the first 9 jobs
failed), the 2000-minute budget is non-renewable with only 100 minutes allocated to
correctness spot-checks, and per-job overhead makes batching mandatory. QDC is the
instrument for cycles, not for correctness the simulator does for free.

The sweep is what actually removed the bottleneck: it answers the mechanism question
for the whole eval set in minutes, and `genuine = correct ∧ fired` means a scan
finding nothing settles `genuine` for 633 attempts with no execution at all.

**And the reference tree was never the multi-day obstacle it looked like** — see
below. Its expensive half (the stage-(g) pass) is recorded in `runs/`, and its other
half regenerates in ~14 s per batch on the host.

## Verdicts discarded, and why

A corrected experiment must be able to say what it threw away. All four corrections
cleared **verdicts only** (`hexkernels.forge.rung0 invalidate`), never generations, so
nothing was re-paid to the API.

1. **352 verdicts taken while four simulators ran concurrently were voided.** Under
   contention a run can exceed its timeout and be recorded as a wrong answer, biasing
   correctness downward. Re-taken serially.
2. **19 of those had been failed by the entry-point lint rule on whitespace.** It
   collapsed runs of whitespace but not whitespace *added* next to punctuation, so
   `candidate_kernel( const _Float16 *x)` was rejected against
   `candidate_kernel(const _Float16 *x)` — which links perfectly. Fixed in
   `lint.py::_canon_sig`, pinned by `tests/test_lint_signature.py`: formatting
   forgiven, renamed parameters / changed types / reordered arguments / missing
   `extern "C"` still rejected.
3. **126 T2 verdicts with no compiler output were voided.** Killing a grading pass
   mid-flight made every remaining attempt fail instantly with empty output, and 126
   of 145 T2 attempts were written as "did not compile" inside one second. It even
   looked like a finding — a compile rate falling from 100% at T1 to 13% at T2.
   `grade_one` now refuses to record a verdict when `compiled` is false and both
   output streams are empty, because an infrastructure failure must never be able to
   masquerade as a measurement.
4. **19 verdicts that were only the simulator's budget running out** (corrected
   2026-08-19, `invalidate --timed-out`). They carried
   `error_text: "simulator timed out after 900s"` with `ran: false` — nothing was ever
   compared against the golden vectors — yet were written as `correct: false`. All 19
   sat in T2/T3, whose median attempt runs ~12 min against a 900 s cap, so the cap
   caught **the slow tail** and correctness read 95.3% where the evidence supports
   98.6%. The error ran in the direction that flatters the thesis, which is the
   direction that must never be silent.

   Same defect as (3) and fixed the same way: `verify` now sets `timed_out`, and
   `grade_one` refuses to record a verdict when it is set. The flag rather than the
   message, so rewording cannot turn an exhausted budget back into a failed
   measurement; pinned by `test_rung0_driver.py`, which also pins the boundary — a run
   that *completed* and disagreed is still a verdict. Unlike (1)–(3) these were
   **reclassified, not blanked**: the timeout is itself the reason, and
   `ungraded_reasons` should keep saying so rather than showing an unexplained gap.
   Compile timeouts now set the same flag, though none occurred in this run.

## The reference builds were lost on 2026-08-18

`benchmark/witness_build/` — every built reference, 50+ batches, days of serial
simulator time — is **gone from disk**. Cause unknown and not guessed at here: there
are no git hooks, `run_batch.destroy_artifacts` only ever removes a single kernel
directory, and no delete was issued against that path. `benchmark/` shows an mtime
of 2026-08-18 23:24.

**What survived, and it is the expensive part.** All 640 generated candidates, all
640 attempt records, every prompt and raw response — `runs/` is intact. The frozen
corpus (`selection.json`, `eval_core.json`, `candidates/`, `AUDIT.md`) is intact.
Nothing that cost API money was lost, and nothing that defines the benchmark was
lost.

**What it costs — corrected 2026-08-19, and it is far less than first written.** The
earlier estimate here ("rebuilding those references first, the original multi-day
serial cost") priced the wrong thing. The directory held two artifacts with wildly
different costs, and only one of them is needed to grade:

| artifact | what grading needs it for | cost to restore |
|---|---|---|
| `harness.cpp` | the golden vectors a candidate is compared against | **~14 s per batch**, host-only |
| `results.json` | the record that the reference PASSED stage (g) | the multi-day part |

`harness.cpp` is **generated by PyTorch on the host** — `run_batch` stages `golden`
and `harness` (`oracle.golden` → `oracle.harness_c`), both before any simulator runs.
There is already a flag for it: `run_batch --skip-verify`, *"build artifacts only; no
toolchain needed."* Measured: 13.7 s for a whole five-kernel batch. Stage (g), the
expensive `verify(kernel_c, harness_c)` that follows, produces nothing grading
consumes — it only **proves** the emitted reference reproduces those goldens.

And the proof does not need re-taking for a reference that already passed: every
attempt record carries `grade.reference_verified`, stamped when it was graded against
a reference `reference_status` had reported verified. `runs/` survived, so that fact
survived. `scripts/restore_reference_stamp.py` copies it back into the file
`reference_status` reads, marked `restored: true` with its evidence so it can never be
mistaken for a fresh execution, and **refuses** where no record carries a stamp.

**The stamp only transfers if the harness came back identical, so that was measured,
not assumed.** On batch 27, five recorded candidates re-run against the *regenerated*
harness reproduced their recorded instruction counts exactly — 5/5 (14356, 14430,
14359, 60195, 60195). The generator supports this: `hexkernels/forge/frontend/`, `mined.py`,
`kernels.py` and `coverage.py` are untouched since 2026-08-11 (before the run), torch
is the same 2.7.1+cu128 recorded in `benchmark/environment.json`, one toolchain
(19.0.04, unchanged since May), and two regenerations produce byte-identical
`harness.cpp`. Across batches 48/57/69/72/74 the rung-0 prompt sha256 reproduces for
**all 25 tasks**, batch 48 included — so the shape-plan bug `mined.py` records for that
batch does not affect these specs.

> **`insns` depends on the ELF filename.** `verify(name=...)` names the binary, the
> simulator is invoked with that path, and startup walks argv — roughly 18
> instructions per character. A constant −396 across five candidates was the check
> using a shorter name than the grader's `f"{ref.key}_s{seed}"`, not a harness
> difference. Any comparison against a recorded count must use the grader's own name.

So the loss costs a batch regeneration (seconds) plus a stamp restoration, not a
rebuild — and the mechanism result was never affected at all, since the sweep needs
only the candidate source and the compiler.

**Why it is not committed as insurance.** `harness.cpp` embeds its golden vectors as
base64, so a T3 task with a 10 MB working set carries roughly 14 MB of source. Across
320 tasks that is gigabytes — which is why the directory is git-ignored in the first
place. That is now the right trade rather than a regret: the directory is
**regenerable in seconds per batch**, so it is a cache, not an asset. What is worth
keeping is `results.json` alone (a few KB per batch), which is the only part that
records days of simulator time.

## Open items

1. **Auto-vectorised kernels: genuine or not?** See above. Affects 3 attempts today,
   and will affect more at rungs 1–3.
2. **Shapes are given in the bare prompt**, against the earlier "no shape hint" note.
   Unratified; rationale in [`rung0.md`](rung0.md).
3. **Which GPT-5.6 variant.** No plain `gpt-5.6` exists; a 3-task pilot chose `luna`
   over `sol`/`terra` (3/3 vs 2/3 vs 1/3 correct, fewest tokens, lowest latency).
   Weak evidence, one flag to change.
4. **70 attempts can never be graded as things stand** — 3 T2 tasks and 11 T3 tasks
   whose references failed stage (g). Full 32-per-tier coverage needs that debugged;
   it is not more simulation.

   Narrowed 2026-08-19: this is **not** the `mined.py` shape-plan bug, as previously
   suspected. Every spec in batches 48/57/69/72/74 — batch 48 included — regenerates
   byte-for-byte, checked by comparing `prompt.build_rung0`'s sha256 against the hash
   recorded in all 25 attempt records. So the references fail stage (g) for some other
   reason, and diagnosing it starts by rebuilding one of them for real
   (`run_batch --batch N` without `--skip-verify`) and reading why it disagrees with
   its own golden vectors.
5. **Rung 1 is unbuilt** (prompt template, multi-turn loop). Rungs 2/3 exist as
   pipelines but have no ladder driver.
