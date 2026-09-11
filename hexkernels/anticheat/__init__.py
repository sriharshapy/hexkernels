"""Anti-cheat: decide whether a kernel GENUINELY used the accelerator by
reading the disassembled ELF, not the source text.

Four of the five detectors are static (architecture-independent);
`used_vtcm` is a RUNTIME PMU counter and reads 0 without timing enabled.
"""
