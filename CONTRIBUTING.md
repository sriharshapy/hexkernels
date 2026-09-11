# Contributing

Contributions are welcome — kernels especially, but also detectors, docs, and
anything that makes a claim in this repo more falsifiable.

## The one rule

**A library entry must actually reach the accelerator.** If `kernel.c` contains no
HVX (`HVX_Vector`, `Q6_V*`, `Q6_W*`) or HMX intrinsic, it is not an entry. This is
enforced by a test, not by review taste:

```bash
python -m pytest tests/test_library.py::test_every_entry_actually_reaches_the_accelerator
```

Scalar implementations are still wanted — as `reference.c`, which is a different and
equally necessary job. A kernel with no reference cannot be checked, and a corpus of
unverifiable kernels is a corpus of assertions.

## Getting set up

```bash
git clone https://github.com/sriharshapy/hexkernels
cd hexkernels
pip install -e ".[dev]"
python -m pytest tests/ -q          # 25 tests, no SDK needed
```

Browsing, querying and testing the corpus needs **nothing but Python**. You only need
the Hexagon SDK to build, simulate, or disassemble — so most contributions, including
docs and loader work, need no hardware at all.

| you want to… | you need |
|---|---|
| query the corpus, add metadata, fix docs | Python 3.10+ |
| build or run a kernel | + Hexagon SDK |
| produce an ELF verdict | + Hexagon SDK (`objdump`) |
| measure on real hardware | + Qualcomm Device Cloud access |

## Adding a kernel

Every entry has the same shape, whatever its origin:

```
kernels/<origin>/<name>/
    kernel.c       the accelerated kernel, entry point `candidate_kernel`
    reference.c    scalar ground truth
    harness.c      checks kernel against reference
    kernel_api.h   entry-point declaration            (optional)
    nearmiss_*.c   plausible WRONG kernels the harness must reject (encouraged)
    PROMPT.md      the ask that produced it           (optional)
    spec.json      normalised metadata
```

Pick `origin` honestly:

- **`expert`** — you wrote it by hand.
- **`mined`** — it targets an operator from an automated walk.
- **`model`** — a language model wrote it. Say which model, in `provenance`.

### Check your kernel is entitled to what it uses

The mechanisms a kernel may reach for are **derived from its working-set size**, not
chosen. A kernel that issues `l2fetch` in a T0 task is rejected by the static
entitlement gate before it ever runs:

```
T0 does not grant l2fetch (granted: ['hvx']) [Q6_l2fetch_AP]
```

The tier table is in [`docs/TIERS.md`](docs/TIERS.md). Getting this wrong is the single
most common reason a contributed kernel does not land.

### Near-miss kernels earn their keep

If you can, include `nearmiss_*.c` — plausible *wrong* implementations your harness
rejects. A harness that no wrong answer can fail is not testing anything. All 350
expert kernels ship theirs; that is why their passing verdicts mean something.

## Reporting a verdict

If you claim a kernel does or does not reach a mechanism, **say which detector said
so.** There are two, and they disagree:

- **static** — reads the disassembled ELF, needs no timing;
- **runtime** — a PMU counter, which for `used_vtcm` **silently reads 0** when timing
  is disabled, exactly like a kernel that never used it.

A number without its detector named is not a result. Same for cycles: quote
`kernel_cycles`, never whole-program `cycles`, and attach the toolchain version.

And keep `elf_confirmed` tri-state. `null` means *never scanned* — it is not a quiet
way of saying `false`. Collapsing the two overstates what is known, which is the exact
failure mode this project exists to catch.

## Things that will get a PR sent back

- A scalar kernel submitted as a library entry.
- A mechanism claim with source intrinsics as its only evidence. Three of the twelve
  model kernels we scanned had intrinsics in source and nothing in the binary.
- A speedup quoted from whole-program cycles or wall clock.
- A hand-edited file under `kernels/` — that tree is generated. Change
  `tools/assemble_library.py` and rebuild, so the next rebuild does not silently undo you.
- Credentials, absolute paths, or personal identifiers anywhere in the diff.

## Style

Match the surrounding code. This codebase comments **why**, not what — especially where
a choice looks wrong until you know what went wrong before it. If you are fixing
something subtle, leaving a note about the failure it prevents is worth more than the
fix alone.

## Regenerating the corpus

`kernels/` is assembled, not authored:

```bash
python tools/assemble_library.py --hexbench ../hexbench --v6 ../HVX-clean/data/v6
```

This needs the upstream source repos. If you do not have them, add your kernel
directory by hand *and* the assembler change that would produce it, and say so in the
PR — a reviewer will rebuild.

## Reporting security issues

Please do not open a public issue for anything that looks like a vulnerability. Use
GitHub's private [security advisory](https://github.com/sriharshapy/hexkernels/security/advisories/new)
flow instead.

## License

By contributing, you agree your contributions are licensed under
[Apache-2.0](LICENSE), the same as the rest of the project.
