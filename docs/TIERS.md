# Two ladders, both called T0–T3

`T0`–`T3` names **two unrelated things** in this project. They have been confused for
each other in conversation more than once. Always say which ladder a tier belongs to.

## 1. The outcome ladder — a score read off the binary, per attempt

| | |
|---|---|
| **T0** | doesn't compile |
| **T1** | compiles, wrong answer |
| **T2** | correct, but scalar |
| **T3** | correct **and** using the mechanism its prompt demanded |

**T2 is the interesting failure**, and the reason this project exists. A working
scalar loop passes a correctness test and often *beats* a mediocre vectorised kernel
on cycles — so a speed-based score actively rewards not using the accelerator. Only
the disassembly separates T2 from T3. See [ANTI-CHEAT.md](ANTI-CHEAT.md).

## 2. The task tiers — the size of a task's working set

Fixed per task in `benchmark/selection.json`, recorded as `task.tier` in every attempt
record and in each kernel's `spec.json`. This is what "32 tasks per tier" and the
per-tier rows of a results table mean.

| task tier | working set | mechanisms the size entitles |
|---|---|---|
| **T0** | fits L1D (≤16 KB) | `hvx` only — nothing for a prefetch to hide |
| **T1** | past L1D, up to L2 (≤1 MB) | `hvx` + `l2fetch` |
| **T2** | past L2, inside VTCM | `hvx` + `l2fetch` + `dma` + `vtcm` |
| **T3** | exceeds VTCM (>8 MB) | same grant as T2, but cannot hold the working set in the scratchpad, so it must stream tiles |

**The mechanism column is derived from the size.** That is the corpus's central claim,
not a convention: which accelerator features a kernel is entitled to reach for follows
from how much data it has to move, and nothing else. `hexkernels/forge/kernels.py`
holds the controlled demonstration — the `fp32_rnorm_t0..t3` ladder, one module at
five sizes, where only the size varies.

## The entitlement gate, and why a mined kernel may be rejected

The harness enforces the table above statically. A kernel that issues `l2fetch` in a
task whose tier is T0 is rejected before it runs:

```
T0 does not grant l2fetch (granted: ['hvx']) [Q6_l2fetch_AP]
```

This matters when using `kernels/mined/`. Those kernels were authored against a
different mining walk, so **126 of 171 sit at a different tier here than the one they
were written for** and will trip this gate. `spec.json` records
`provenance.tier_match` for every one of them. They are retained rather than deleted
because tier assignment is a property of the mining walk, not of the kernel — a
re-mine can make a rejected kernel valid again. See
[study/MINED-PROVENANCE.md](study/MINED-PROVENANCE.md).

```python
from hexkernels.library import find

# mined kernels that match the tier they are filed under, and so should pass the gate
[k for k in find(origin="mined") if k.spec["provenance"]["tier_match"]]
```
