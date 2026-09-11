# hexkernels

**538 Hexagon NSP kernels that actually use the accelerator** — every one carrying
HVX or HMX intrinsics, a scalar reference to check it against, and a harness to run
both. Plus the machinery that produced them: an agentic kernel-building pipeline, a
simulator and on-silicon measurement path, and an anti-cheat detector that decides
whether a kernel *really* used the hardware by disassembling the ELF instead of
trusting the source.

A scalar C file is never a library entry here. It ships as `reference.c`, which is a
different job.

```python
from hexkernels.library import find

find(hmx=True)                  # 73 kernels that reach the matrix engine
find(origin="expert")           # 350 hand-written, with measured speedups
find(elf_confirmed=True)        # 9 the disassembler confirms, not just the source
```

## The library

| origin | n | what it attests |
|---|---:|---|
| `expert` | 350 | Hand-written HVX/HMX kernels, each with a measured `expert_kernel_cycles` against its own scalar baseline. Median speedup **3.4×**, best **124×**. |
| `mined` | 171 | Written against operators mined from the PyTorch registry. **Read `tier_match` before using one** — 126 of 171 were authored for a different working-set tier than the task they are filed under, and the entitlement gate will reject them. Retained deliberately; see [MINED-PROVENANCE](docs/study/MINED-PROVENANCE.md). |
| `model` | 17 | Kernels a language model wrote that reached for HVX. The disassembler confirms **9**, refutes **3**, and 5 were never scanned. |

73 of the 538 reach HMX, the matrix engine. Every one ships its accelerated source,
its scalar reference, its prompt and its metadata. 467 also ship their harness; for the
other 71 the harness would be 1 MB to 93 MB of base64 golden vectors, so a `HARNESS.md`
records the one command that regenerates it instead.

Every entry has the same shape, whatever its origin:

```
kernels/<origin>/<name>/
    kernel.c       the accelerated kernel          ← the library entry
    reference.c    scalar ground truth
    harness.c      correctness + cycle harness
    kernel_api.h   entry-point declaration          (expert only)
    nearmiss_*.c   plausible WRONG kernels the harness must reject
    PROMPT.md      the ask that produced it
    spec.json      normalised metadata
```

`spec.json.bundle` says which of `complete` (467) or `harness-regenerable` (71) an
entry is, so "can I build this?" is one field rather than a directory listing.

`kernels/index.json` is the whole corpus in one file. Rebuild it with
`python tools/assemble_library.py`.

## The three things worth reading

### 1. Anti-cheat — `hexkernels/anticheat/`

The central problem: **a working scalar loop passes a correctness test and often
beats a mediocre vectorised kernel on cycles.** A score based on speed actively
rewards not using the accelerator. So correctness and speed both fail as evidence,
and the only thing that settles it is the disassembly.

Four detectors are static and architecture-independent; `used_vtcm` also exists as a
runtime PMU counter. The two disagree in a way that is [documented, not
hidden](docs/ANTI-CHEAT.md) — a kernel is not called "genuine" unless the detector
that produced the verdict is named alongside it.

This is not hypothetical. All 17 model-written kernels in this library have HVX
intrinsics *in the source*. Twelve of them were scanned: the ELF confirms the
mechanism in 9 and refutes it in 3. Writing the intrinsic is not the same as the
binary containing it.

### 2. The agentic flow — `hexkernels/forge/`

PyTorch operator → Linalg IR → MLIR fusion and bufferisation → affine loops →
portable scalar C. That scalar C is **the question, not the answer**: a model is then
asked to reconstruct it as accelerated Hexagon C, and the result is judged against a
golden harness generated beside the reference.

The pipeline refuses to skip a stage. A kernel without an FX-derived primitive graph
or without real Linalg IR is not a product of this pipeline, and a batch whose
Linalg coverage is under 100% is not shippable. [Details](docs/AGENTIC-FLOW.md).

`hexkernels/gym/` closes the loop: the model writes a kernel, a profiler names the
bottleneck, and it is told which mechanism is missing.

### 3. Simulator and silicon — `hexkernels/core/`, `hexkernels/device/qdc/`

`core/` holds the toolchain wrapper, target detection, the simulator fleet, and the
reward function. `device/qdc/` runs the same kernels on real hardware through
Qualcomm Device Cloud — which is an adb-server port forward, not a shell, and bills a
session for its full timeout if you do not release it.

The measurement rules that are easy to get wrong — and that cost this project 352
discarded verdicts when they were — are in [MEASUREMENT.md](docs/MEASUREMENT.md).
The first one: **never parallelise `hexagon-sim`.**

## Install

```bash
pip install -e .
```

The library, its metadata, and the loader work with no special hardware. Anything
that compiles, simulates, or measures needs the Hexagon SDK;
`hexkernels/core/toolchain.py` is the one place that knows the SDK paths and the
pinned bus parameters.

## Two ladders both called T0–T3

`T0`–`T3` names two unrelated things, and they have been confused for each other
before. Always say which ladder you mean.

**The outcome ladder**, scored per attempt: T0 doesn't compile → T1 compiles but
wrong → T2 correct but scalar → T3 correct *and* using the mechanism.

**The task tiers**, fixed per task by working-set size: T0 fits L1D → T1 reaches L2 →
T2 fits VTCM → T3 exceeds it and must stream tiles. The mechanisms a task is
entitled to are *derived from its size*, which is the corpus's central claim.

Full table in [TIERS.md](docs/TIERS.md).

## The study

The library came out of a question: can a model be got to use the accelerator at all?
Three rungs, 1,920 attempts, one model.

| rung | the ask | reached a mechanism |
|---|---|---:|
| 0 | bare, unaided | 3 / 640 |
| 1 | + a turn loop that reports failures | **0 / 640** |
| 2 | + the mechanism named explicitly | 20 / 640 in source, 2 confirmed in ELF |

Rung 1 taking correctness to 100% while mechanism use went to *zero* is the result
the anti-cheat detector exists to make visible. Write-ups, methodology and caveats are
in [docs/study/](docs/study/). The 17 kernels that survived it are in the library under
`origin: model`; the raw per-attempt artifacts are not published.

## Layout

| path | what it is |
|---|---|
| `kernels/` | The library. One directory per kernel, one schema, plus `index.json`. |
| `hexkernels/library/` | Load and query the corpus. |
| `hexkernels/anticheat/` | The ELF disassembly detectors. |
| `hexkernels/forge/` | The agentic pipeline, including `frontend/` (trace, graph, emit, schedule, oracle). |
| `hexkernels/gym/` | Profile-and-tune loop. |
| `hexkernels/core/` | Toolchain, target detection, simulator fleet, reward. |
| `hexkernels/device/qdc/` | On-silicon measurement via Qualcomm Device Cloud. |
| `benchmark/` | The frozen corpus definition: task selection, eval core, operator pool. |
| `env/harness/` | Harness headers every build includes. |
| `tools/` | The script that assembles `kernels/`. |

## License

Apache-2.0. See [LICENSE](LICENSE).
