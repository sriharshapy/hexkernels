# Provenance of `kernels/mined/`

## These 171 kernels are NOT 171 witnesses -- read this first

This directory holds 171 imported kernels, but a name match against this
repo's task list is not a task match. A candidate `.cpp` was written
against one specific *shape and tier* in the source (Forge v2) pipeline.
This repo's mined corpus was assembled independently, and which tier an
operator lands in depends on its position in the mining walk -- so the
same operator name very often sits at a different tier here than it did
in the source repo.

Comparing, for each of the 171 imported kernels, this repo's task tier
(from `benchmark/selection.json`, via `hexkernels.forge.mined.MINED_BATCHES`)
against the tier recorded in the source repo's own
`run_artifacts/forge2/batch*/<kernel>/provenance.json` (field `tier`):
**45 of the 171 share their task's tier here; 126 do not.** Both numbers,
and the per-kernel tier pair, are recorded in `MANIFEST.json` under
`task_tier` / `source_tier` / `tier_match`, computed by the same method
`hexkernels/forge/tests/test_candidates.py` checks for internal consistency.
(An earlier informal pass over this same comparison had estimated
"43 match, 128 differ" -- close, but this file and `MANIFEST.json` report
the exact, reproducible count above rather than that estimate, precisely
so this document cannot drift from what the manifest actually says.)

A mismatched tier is not a cosmetic detail: this benchmark's harness
enforces a static entitlement gate per tier, and a kernel written for a
tier other than the one its task lands at here will typically be
**rejected at that gate** before it ever runs (e.g. a kernel written
assuming a tier that grants `l2fetch` will fail with something like
`T0 does not grant l2fetch (granted: ['hvx']) [Q6_l2fetch_AP]` if the
task's actual tier here is `T0`). A live compile-and-run pass over the
first 160 of this repo's 320 tasks confirms this: of those 160, 58 had a
candidate present in this import; only **17 compiled and were correct**.
**37** of the remaining 41 were rejected by the static entitlement gate
with exactly this class of error, and **4** more compiled but returned
wrong results (`HVXENV_INCORRECT`) -- a shape mismatch within a tier that
otherwise matched. That puts the measured, full-set mechanism-witness
rate at **18 of 160 tasks checked so far (11%)**, not 171 of 320.

The 128 (now measured as 126) tier-mismatched kernels are **deliberately
retained here, not deleted**. A kernel that is wrong for this repo's
current tier assignment becomes a valid witness again if a future re-mine
of the corpus assigns its task a different tier -- tier assignment is a
property of the mining walk, not of the kernel, and re-mining is expected
to happen. Deleting an auditable, sha256-pinned artifact just to make a
headline "witness count" look better is exactly the kind of quiet
overstatement this benchmark's own audit machinery exists to catch; this
import does not do that to itself.

## What these are

The kernels in this directory are **achievability witnesses**, not a
performance baseline. Each one proves that a correct implementation exists
for its task that actually trips the hardware mechanism the task is
entitled to (HVX, HMX, DMA, VTCM, l2fetch, as recorded per-task in
`hexkernels.forge.mined.MINED_BATCHES`). The reference implementations generated
elsewhere in this repo are naive scalar C by design and never touch the
accelerator -- a failing model on a task would otherwise be
indistinguishable from an impossible task. These candidates close that gap.

They were imported from the Forge v2 pipeline artifacts in the
`HVX-clean` repository
(`run_artifacts/forge2/batch*/candidates/*.cpp`), which this project was
ported from. 171 of 295 distinct task names in this repo's frozen mined set
had a matching candidate; the rest have no witness yet. See
`MANIFEST.json` for the exact source path and sha256 of every imported
file. Content was copied byte-for-byte; kernel source was not edited.

Two task names (`fp32_trace`, `fp32_absolute__any`) had a candidate present
in two different source batches. In both cases the file kept is the one
whose batch `results.json` records `hvx: true` and `hvx_compute: true`
(and, for the tie between two such candidates, the lower `pcycles`) --
since HVX is the mechanism this benchmark centers on, that is the
more informative achievability witness. This is recorded in git history
for anyone who wants to re-derive the choice; the losing duplicate was not
imported.

## Derivation rule enforced by the source project

The source (Forge v2) pipeline enforced a strict derivation rule on every
candidate: it could derive only from the generated reference kernel, the
schedule annotation, and **vendor** intrinsic headers --
`hexagon_types.h`, `hexagon_protos.h`, `hvx_hexagon_protos.h`, and
`hmx_hexagon_protos.h`. R&D helper headers (`hmx_helpers.h`,
`harness_common.h`) were forbidden, made unreachable to the model that
authored candidates, and scanned for by symbol name after the fact. This
import re-ran that same symbol scan (see `hexkernels/forge/tests/test_candidates.py`
and the leak-check step recorded below) against the final set of 171
imported files: no hits.

## Open disclosure item (not settled here)

The Forge corpus is model-authored. `PLAN.md` §F requires disclosing which
model authored the corpus, and any vendor overlap between that model and
the models evaluated at rungs 0-3 of this benchmark. **That disclosure is
outstanding** -- it is not resolved by this import and should be filled in
before these witnesses are used in any comparison-sensitive context.

The controller's reasoning for importing anyway: an existence proof is not
contaminated by its author. Whether a task is achievable -- whether some
implementation exists that is both correct and trips the required hardware
mechanism -- is a fact about the hardware and the task, not a fact about
who wrote the witness. A witness authored by model X is exactly as valid
a proof of achievability as one authored by a human, provided it is
verified correct and verified to trip the mechanism (which the source
project's `correct` and `mechanisms` fields in each batch's `results.json`
already attest to, independent of authorship).

A distinct risk remains, and is explicitly **not** resolved by that
reasoning: using these same kernels as the `expert_kernel_cycles`
denominator in a speedup comparison against whichever ladder model
authored the Forge corpus would be a comparison contamination -- the
"expert" and the "compared-against model" would be the same author. That
is a separate decision, gated on the still-outstanding disclosure above,
and this import does not make it.
