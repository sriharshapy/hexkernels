# Anti-cheat: deciding whether a kernel really used the accelerator

## Why correctness and speed both fail as evidence

A kernel that is *correct* proves only that the arithmetic is right. A scalar loop is
correct. A kernel that is *fast* proves only that it was faster than whatever it was
compared against — and here is the trap that motivates this whole module:

> **A working scalar loop often beats a mediocre vectorised kernel on cycles.**

So a score based on speed does not merely fail to reward accelerator use, it actively
punishes it: the model that writes a tight scalar loop outscores the model that
writes a clumsy but genuine HVX kernel. Optimise that score and you get a benchmark
that trains models away from the hardware it exists to measure.

The only thing that settles the question is the binary. Not the source — the source
can contain an intrinsic the compiler never emitted — and not the runtime, which
reports cycles and says nothing about which units retired them.

## What the detectors read

`hexkernels/anticheat/` disassembles the built ELF and looks for the instruction
classes that correspond to each mechanism:

| mechanism | what confirms it |
|---|---|
| `hvx` | vector instructions in the disassembly |
| `hvx_compute` | vector instructions that compute, not merely move — a kernel that only loads and stores vectors has not vectorised its arithmetic |
| `hmx` | matrix-engine instructions |
| `l2fetch` | the prefetch instruction, with a non-degenerate descriptor |
| `dma` | user-DMA start/wait |
| `vtcm` | addresses in the scratchpad's range |

Four of these are static and architecture-independent: they need the ELF and nothing
else — no simulator, no timing, no device.

## The `used_vtcm` asymmetry — disclose which detector you used

`used_vtcm` exists twice, and the two do not always agree:

- **Static** (`anticheat._disasm_has_vtcm`) reads the disassembly. Needs no timing.
  This is what `hexkernels/forge/verify.py` uses.
- **Runtime** is a PMU counter. It needs timing enabled and **silently reads 0
  otherwise** — which looks exactly like a kernel that did not use VTCM.

A number from one is not a number from the other. Any reported VTCM figure must name
the detector that produced it. This asymmetry is a disclosure item, not an
implementation detail.

## `hvx_compute` and why it exists separately from `hvx`

The C runtime and the harness pull in library routines that themselves contain vector
instructions. A detector that only asked "are there vector instructions in this ELF?"
would answer yes for a kernel that is entirely scalar, because `memcpy` is vectorised.
`hvx_compute` is the narrower question — did the *kernel's own arithmetic* vectorise —
and it is the one that matters.

## The evidence in this repo

Of the 17 model-written kernels in `kernels/model/`, every one contains HVX intrinsics
in its source. Twelve were scanned:

| ELF verdict | n |
|---|---:|
| confirmed | 9 |
| refuted | 3 |
| never scanned | 5 |

The three refuted kernels are `fp16__convolution`, `fp16_inner` and
`fp16_rnn_relu_cell` — all fp16, all compiled, and none of them reached the vector
units in the binary despite asking for them in the source.

Note the three states. `spec.json`'s `verified.elf_confirmed` is tri-state on purpose:
`true`, `false`, and `null` for never-scanned. Collapsing `null` into `false` would
overstate what is known, and collapsing it into `true` would overstate what was found.
`hexkernels.library.find(elf_confirmed="unscanned")` selects the third state
explicitly rather than letting it hide.

## Using it

```python
from hexkernels.anticheat import anticheat
from hexkernels.library import find

k = find(origin="model", name="i32___and___Tensor")[0]
k.elf_confirmed          # True  -- the detector saw it
k.spec["verified"]["mechanisms_fired"]
```

Scanning a kernel you have built yourself needs the Hexagon SDK, since it disassembles
a real ELF. See `hexkernels/core/toolchain.py` for the paths.
