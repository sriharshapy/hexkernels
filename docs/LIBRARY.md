# The kernel library

538 kernels, each of which reaches HVX or HMX. The admission rule has no exceptions:
**if the accelerated source contains no vector or matrix intrinsic, it is not a library
entry.** Scalar C ships as `reference.c`, which is a different role.

```python
from hexkernels.library import find, load, summary

summary()                                  # the counts, as built
find(hmx=True)                             # 73 kernels that reach the matrix engine
find(origin="expert", dtype="fp16")        # 32 hand-written fp16 kernels
find(tier="T2", buildable=True)            # complete bundles at one working-set tier
find(elf_confirmed=True)                   # the disassembler saw the mechanism
find(elf_confirmed="unscanned")            # never scanned -- a third state, not a no
```

## One schema, whatever the origin

```
kernels/<origin>/<name>/
    kernel.c       the accelerated kernel          ← the library entry
    reference.c    scalar ground truth
    harness.c      correctness + cycle harness
    kernel_api.h   entry-point declaration          (expert only)
    nearmiss_*.c   plausible WRONG kernels the harness must reject
    PROMPT.md      the ask that produced it
    spec.json      normalised metadata
    HARNESS.md     how to regenerate the harness, when it is too large to ship
```

Every entry has the same shape, so a consumer never has to branch on `origin`. The
entry point is always `candidate_kernel`.

`spec.json.bundle` is one field answering "can I build this?":

| bundle | n | meaning |
|---|---:|---|
| `complete` | 467 | reference and harness both present |
| `harness-regenerable` | 71 | everything non-regenerable is here; the harness is not |

A harness embeds its golden vectors as base64, so its size tracks the task's **working
set**, not its complexity — the largest is 93 MB, and 71 of 538 would account for
675 MB of an otherwise 16 MB corpus. Those are a build cache, not a result:
`run_batch` regenerates a whole batch in about 14 seconds, and each affected entry's
`HARNESS.md` carries the exact command. Nothing that cannot be reproduced from the
task definition was dropped.

`kernels/index.json` is the whole corpus in one file. Rebuild everything with:

```bash
python tools/assemble_library.py --hexbench ../hexbench --v6 ../HVX-clean/data/v6
```

## The three origins, and what each actually attests

### `expert` — 350 kernels

Hand-written HVX/HMX kernels. Each carries a **measured** `expert_kernel_cycles`
against its own scalar baseline, from a simulator run at corpus build time: median
speedup **3.4×**, best **124×**. 68 of them reach HMX.

`spec.json` also carries `edge_cases` — the specific inputs the harness uses to catch a
kernel that got the easy path right, e.g. for sigmoid: *"x=0 gives 0.5, not 0 or 1, so
relu and identity both fail"*.

Every one of the 350 ships its **near-miss kernels** too — 421 in total, plausible
wrong implementations (`nearmiss_identity.c`, `nearmiss_relu.c`, …) that the harness
must reject. They are what makes a passing verdict mean something: a harness that no
wrong answer can fail is not testing anything, and these are the wrong answers it was
checked against.

What they do *not* carry: an ELF scan. These are attested by cycles, not by the
disassembly detector, so `verified.elf_confirmed` is `null` for all of them.

### `mined` — 171 kernels

Written against operators mined from the PyTorch registry — the op came from a measured
coverage list in harvest order, the size was solved for a tier, and the mechanisms
follow from that size. What is hand-written is the arithmetic and the memory plan.

**Read `provenance.tier_match` before using one.** 126 of 171 were authored against a
different tier than the task they are filed under here, and the entitlement gate will
reject them (see [TIERS.md](TIERS.md)). They are retained rather than deleted because
tier assignment is a property of the mining walk, not of the kernel: a re-mine can make
a rejected kernel valid again. Deleting auditable, hash-pinned artifacts to improve a
headline count is exactly the quiet overstatement this project's audit machinery exists
to catch.

A live pass over 160 tasks found 58 with a candidate present, of which 17 compiled and
were correct; 37 of the rest were rejected by the entitlement gate and 4 more returned
wrong results. Full numbers in [study/MINED-PROVENANCE.md](study/MINED-PROVENANCE.md).

### `model` — 17 kernels

Kernels a language model wrote that reached for HVX — the rare survivors of 1,920
attempts. One entry per task; where several seeds produced an accelerated kernel, the
kept one is correct-and-confirmed first, then lowest seed, so the choice is
deterministic and re-runnable. Each `spec.json` records which rung, seed and turn count
it came from.

These are the only kernels in the library with a real ELF verdict, and it is worth
reading:

| | n |
|---|---:|
| ELF confirms the mechanism | 9 |
| ELF **refutes** it | 3 |
| never scanned | 5 |

All 17 have intrinsics in the source. Three of the twelve scanned did not have them in
the binary. That gap is the entire argument for [ANTI-CHEAT.md](ANTI-CHEAT.md).

## `elf_confirmed` is tri-state, deliberately

`true`, `false`, and `null` for never-scanned. Collapsing `null` into `false` overstates
what is known; collapsing it into `true` overstates what was found. `find()` matches the
state exactly — `find(elf_confirmed=None)` means "do not filter on this", and the string
`"unscanned"` selects the third state explicitly rather than letting it hide behind a
falsy value.

## Building a kernel

Building, simulating, or scanning needs the Hexagon SDK; `hexkernels/core/toolchain.py`
is the one place that knows the SDK paths and the pinned bus parameters. The harness
headers every build includes are in `env/harness/`.

Loading and querying the corpus needs none of that — the metadata, sources, and prompts
are all plain files.
