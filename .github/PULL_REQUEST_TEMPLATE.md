<!--
Thanks for contributing. Delete the sections that don't apply — a one-line
docs fix does not need a measurement table.
-->

## What this changes

<!-- One or two sentences. What is different after this merges? -->

## Why

<!-- The problem it solves. Link an issue if there is one. -->

---

## If this adds or changes a kernel

- [ ] `kernel.c` uses a real HVX or HMX intrinsic — **a scalar kernel is never a library entry**
- [ ] `reference.c` is scalar, and is genuinely the ground truth (not a copy of the kernel)
- [ ] `harness.c` checks the kernel against the reference, including the tail and edge cases
- [ ] `spec.json` validates — run `python -m pytest tests/test_library.py`
- [ ] The mechanisms claimed are ones the task's **tier** entitles it to (see [`docs/TIERS.md`](../blob/main/docs/TIERS.md))

**Evidence it reaches the accelerator** — source intrinsics are not evidence:

```
<!-- anti-cheat scan output, or the relevant disassembly lines -->
```

**Measurement**, if you have one. `kernel_cycles` against the reference, never
whole-program cycles, with the toolchain version attached:

| | cycles |
|---|---:|
| reference | |
| this kernel | |
| toolchain | |

## If this changes a detector or a verdict

- [ ] I have said which detector produced each number (static disassembly vs. runtime PMU)
- [ ] Any newly `confirmed`/`refuted` verdict has disassembly behind it
- [ ] `elf_confirmed` stays tri-state — `null` (never scanned) is not folded into `false`

## Checks

- [ ] `python -m pytest tests/ -q` passes
- [ ] Docs updated if behaviour or counts changed
- [ ] No credentials, absolute paths, or personal identifiers in the diff

<!--
If you regenerated kernels/, please say so — reviewers should see whether
index.json changed because of your kernel or because of a rebuild.
-->
