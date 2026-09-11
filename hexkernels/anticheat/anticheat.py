"""ELF-level used_hvx/used_hmx anti-cheat -- STATIC detection.

Ported byte-identical (logic-frozen) from the old m1_driver/evaluate.py
(lines ~218-370): the disassembly-based HVX/HMX detection that keys on the
V-register / vmem *operand*, not the mnemonic, so scalar GPR-pair ops that
disassemble with HVX-looking mnemonics (vaddw/vaddub on rN registers) are
correctly rejected. This is the credibility core of the eval loop -- it must
never grant HVX/HMX credit on uncertainty (fail-closed).

WHAT THESE DETECTORS CAN AND CANNOT ANSWER
==========================================
Every function here answers exactly one question: **is the instruction present
in the disassembly?** None of them can answer "did it execute?" or "did it move
any bytes?", because a static disassembly contains no such information.

That gap is real and measured, not theoretical. Demonstrated 2026-08-02: a
kernel whose HVX compute and l2fetch sit in a never-taken branch, plus a live
``Q6_l2fetch_AR(buf, 0)`` with Height=0, produced::

    used_hvx = used_hvx_compute = used_l2fetch = True
    ... while 0 of 2 HVX packets executed, and the one l2fetch that did execute
        registered L2FETCH_COMMAND_KILLED=1 with L2FETCH_ACCESS=0.

Use ``hexkernels.anticheat.anticheat_runtime`` to add execution and work evidence. It
filters a disassembly down to packets that actually committed and then runs
these same frozen detectors against the result, so the detection logic below is
reused verbatim rather than reimplemented::

    from hexkernels.anticheat import anticheat_runtime as rt
    prof = rt.parse_packet_profile(open(packet_analyze_json).read())
    rt.genuine_mechanism(disasm_of_linked_elf, prof, "l2fetch",
                         _disasm_has_l2fetch)

Neither layer detects code that executes and whose result is then discarded
into an unused buffer; that needs differential ablation. Conformance figures
remain upper bounds.

ENVIRONMENT THESE CITATIONS DESCRIBE
------------------------------------
* Hexagon SDK **6.4.0.2**, tools **19.0.04**
* Target ``-mv68`` -> rev_id ``0x00008d68`` (``v68n_1024``); 6 HW threads,
  L1-I 32 KB, L1-D 16 KB, L2 1024 KB, VTCM 4096 KB, HVX 128B
* Instruction and PMU semantics: **Hexagon V68 Programmer's Reference Manual,
  80-N2040-46 Rev. B** (memory ch 5 p79-102, PMU events ch 9 p140-152)
* VTCM: **Hexagon V68 HVX Programmer's Reference Manual, 80-N2040-47 Rev. E,
  section 3.2 p15**

Full citation chain: ``docs/hexagon/SDK_DOC_INDEX.md``.
"""
import re

from hexkernels.core import target as _target
from hexkernels.core.toolchain import run, SIM_TIMEOUT_S

# Strip C comments before testing for HVX usage so names in comments don't count.
_HVX_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
# Match real HVX vector types or HVX vector/vector-pair intrinsics in non-comment code.
# Q6_[VW] captures HVX vector (V) and vector-pair (W) intrinsics; scalar Q6_R_* excluded.
_HVX_USE = re.compile(r"\bHVX_Vector(?:Pair)?\b|\bQ6_[VW]\w+")


# --- ELF-level HVX detection (anti-cheat) --------------------------------
# A real HVX vector instruction references an HVX vector register (v0..v31,
# incl. pairs v1:0) or an HVX vector memory op (vmem/vmemu/vgather/vscatter).
# Scalar GPR-pair ops disassemble with HVX-looking MNEMONICS (vaddw/vaddub)
# but operate on rN registers -- the V-register / vmem operand is the
# discriminator, so we never key on the mnemonic.
_HVX_INSN = re.compile(r"\bv\d+(?::\d+)?\b|\bvmemu?\b|\bvgather\b|\bvscatter\b")


def _disasm_has_hvx(disasm_text: str) -> bool:
    """True iff any instruction line in objdump -d output references an HVX
    vector register or vector memory op. Pure (no I/O) -> fast-testable."""
    for line in disasm_text.splitlines():
        if "\t" not in line:
            continue  # symbol headers / blank lines have no instruction column
        asm = line.rsplit("\t", 1)[-1]  # assembly text is the last tab field
        if _HVX_INSN.search(asm):
            return True
    return False


# HVX *memory* ops (vector load / store / gather / scatter). A kernel that only
# moves data through the vector unit but does its arithmetic in scalar registers
# ("load-only HVX") is NOT genuinely accelerated -- these are the ops to exclude
# when asking "did the vector unit do real compute?".
_HVX_MEM = re.compile(r"\bvmemu?\b|\bvgather\b|\bvscatter\b")


def _disasm_has_hvx_compute(disasm_text: str) -> bool:
    """True iff the disasm contains a genuine HVX vector *compute* op -- a
    V-register instruction that is not a pure vector memory op.

    This is the anti-cheat discriminator the REWARD gate uses in place of a
    vec_frac density threshold: density wrongly rejects fused/bandwidth kernels
    whose defining vector work (e.g. one vrmpy or vmin) is a small fraction of
    instructions, yet those are exactly the accelerable tasks. Presence of a
    vector-arithmetic/permute op (combined with the reward's continuous >=1.2x
    speedup magnitude) is the right, over-rejection-free bar, and it matches the
    achievability ceiling's genuineness definition. Pure (no I/O)."""
    for line in disasm_text.splitlines():
        if "\t" not in line:
            continue
        asm = line.rsplit("\t", 1)[-1]
        if _HVX_INSN.search(asm) and not _HVX_MEM.search(asm):
            return True
    return False


# A genuine HMX matmul loads BOTH matrix operands into the matrix array: an
# `activation.` operand AND a `weight.` operand (the defining act of a
# multiply). Real hexagon-llvm-objdump forms, verified by compiling the
# hmx_tile_matmul helpers (-mv68 -mhvx -mhmx -O2): int8 `activation.ub =
# mxmem(r4,r7)` / `weight.b = mxmem(r3,r8)`, fp16 `activation.hf = mxmem(...)`
# / `weight.hf = mxmem(...)`. Requiring the operand PAIR closes the old hole
# where a single stray `mxmem` (matrix-memory move) or a bare accumulator store
# flipped used_hmx without any multiply happening. Keyed on the assembly column
# (last tab field), like _disasm_has_hvx, so a token inside a symbol name never
# grants credit. Fail-closed.
_HMX_ACT = re.compile(r"\bactivation\.")
_HMX_WGT = re.compile(r"\bweight\.")


def _disasm_has_hmx(disasm_text: str) -> bool:
    """True iff the disasm shows a genuine HMX matmul -- BOTH an `activation.`
    matrix operand AND a `weight.` matrix operand loaded into the HMX array.
    Matrix-memory movement alone (a lone `mxmem` / accumulator store) is not a
    multiply and does NOT count. Column-scoped + fail-closed. Pure (no I/O).

    KNOWN LIMITATION: the operand pair is matched over the WHOLE text, not per
    function or per region, so an `activation.` in one function and a `weight.`
    in an unrelated (even dead) one would satisfy it. Filtering through
    ``anticheat_runtime.executed_disasm`` removes the dead-code half of that;
    region scoping remains unaddressed.

    HMX PMU counters are not mapped yet -- the HVX PRM's own PMU chapter
    (80-N2040-47 Rev. E ch 4 "HVX PMU events", p21-22) has not been read.
    ``COPROC_BUSY_PVIEW_CYCLES`` (0xed) exists in the V68 PRM and is a stall
    counter, not proof of a multiply.
    """
    has_act = has_wgt = False
    for line in disasm_text.splitlines():
        if "\t" not in line:
            continue  # symbol headers / blank lines have no instruction column
        asm = line.rsplit("\t", 1)[-1]  # assembly text is the last tab field
        if _HMX_ACT.search(asm):
            has_act = True
        if _HMX_WGT.search(asm):
            has_wgt = True
        if has_act and has_wgt:
            return True
    return False


def _disasm_static_counts(disasm_text: str) -> tuple[int, int, int]:
    """Count (instructions, packets, hvx_vector_instructions) in objdump -d output.

    Hexagon VLIW packets are brace-delimited: a packet opens with '{' and closes
    with '}' (a 1-instruction packet carries both on its line), so the count of
    closing braces is the packet count. Each disassembled instruction is one line
    with a tab-separated assembly field (same convention as _disasm_has_hvx). HVX
    vector instrs are those whose asm references a V-register / vector-memory op
    (_HVX_INSN). Candidate-only (the object is compiled with -c) -> excludes
    harness/CRT/libc, so this is a clean static measure. Pure (no I/O)."""
    insns = packets = hvx = 0
    for line in disasm_text.splitlines():
        if "\t" not in line:
            continue  # symbol headers / blank lines have no instruction column
        asm = line.rsplit("\t", 1)[-1]  # assembly text is the last tab field
        insns += 1
        if "}" in asm:
            packets += 1
        if _HVX_INSN.search(asm):
            hvx += 1
    return insns, packets, hvx


def detect_used_hvx_elf(obj_path: str, objdump: str, env: dict) -> bool:
    """True iff the disassembled object contains a real HVX vector instruction.

    Fail-closed: a missing objdump, nonzero exit, empty output, a hang, or any
    error returns False -- the anti-cheat must never GRANT HVX credit on
    uncertainty, nor block the eval loop (hence the timeout, like the sim).
    """
    try:
        rc, out, _, timed_out = run([objdump, "-d", obj_path], env, timeout=SIM_TIMEOUT_S)
        if timed_out:
            return False
    except Exception:
        return False
    if rc != 0 or not out:
        return False
    return _disasm_has_hvx(out)


def detect_used_hvx(src_path: str) -> bool:
    """Return True if the source actually uses HVX types/intrinsics (not just headers).

    Requires an HVX_Vector, HVX_VectorPair, or Q6_V*/Q6_W* intrinsic in non-comment
    code. A bare header include (hexagon_protos.h) no longer counts — that was
    gameable: a scalar loop could include the header and claim the HVX reward bonus.
    """
    try:
        with open(src_path, "r", errors="replace") as f:
            src = f.read()
    except OSError:
        return False
    code = _HVX_COMMENT.sub(" ", src)  # remove comments; don't count names inside them
    return bool(_HVX_USE.search(code))


# --- Additional mechanism detectors (ADDITIVE; HVX/HMX above are frozen) -----
# Hexagon user-DMA descriptor-engine ops. Unlike HVX, these mnemonics are
# unambiguous (no scalar op shares them), so a mnemonic match on the assembly
# column is sound. Fail-closed, pure (no I/O), same tab-field convention as
# _disasm_has_hvx.
_DMA_INSN = re.compile(
    r"\bdm(?:start|link|wait|pause|resume|poll|cfgrd|cfgwr|syncht|tlbsynch)\b")
_L2FETCH_INSN = re.compile(r"\bl2fetch\b")


def _disasm_has_dma(disasm_text: str) -> bool:
    """True iff any instruction line references a user-DMA op. Pure.

    Shares the structural limitation of every detector here: a ``dmstart`` in a
    never-executed branch passes, and a descriptor with zero length executes
    while moving nothing. No dedicated user-DMA PMU counter is confirmed yet --
    the ``AXI_*`` family (V68 PRM ch 9) measures system-wide off-chip traffic and
    cannot attribute it to the DMA engine -- so ``anticheat_runtime`` supplies
    execution evidence for DMA but not work evidence, and reports work as
    ``None`` (unknown) rather than guessing.
    """
    for line in disasm_text.splitlines():
        if "\t" not in line:
            continue
        asm = line.rsplit("\t", 1)[-1]
        if _DMA_INSN.search(asm):
            return True
    return False


def _disasm_has_l2fetch(disasm_text: str) -> bool:
    """True iff any instruction line issues an l2fetch prefetch. Pure.

    WEAKEST DETECTOR IN THIS MODULE -- do not use it alone as evidence of
    genuine prefetching.

    ``l2fetch(Rs,Rt)`` is a 2-D background prefetch (V68 PRM 80-N2040-46 Rev. B
    section 5.10.6 p97-99, instruction reference p382-384): Rs is the virtual
    start address; ``Rt[15:8]`` Width in bytes, ``Rt[7:0]`` Height in blocks,
    ``Rt[31:16]`` Stride. It is non-blocking, slot 0 only, and groupable only
    with ALU32 or non-floating-point XTYPE instructions (p52).

    The PRM documents **seven** ways an l2fetch can be present -- even execute --
    and still move nothing. Five have dedicated PMU counters (ch 9 p140-152):

        L2FETCH_COMMAND_KILLED            0x92  killed by a Stop command
        L2FETCH_COMMAND_OVERWRITE         0x93  superseded by a later command
        L2FETCH_ACCESS_CREDIT_FAIL        0x94  blocked, no L2FETCH/L2evict credit
        L2FETCH_COMMAND_PAGE_TERMINATION  0xd3  no VA->PA translation, or permission error
        L2FETCH_DROP                      0xe0  dropped, prior eviction incomplete

    The remaining two need no counter and no adversarial intent:

      * "If the lines of interest are already in the L2, no action is
        performed." (section 5.10.6 p98)
      * **Height = 0 or Width = 0** encodes a legal instruction that fetches
        nothing. ``Q6_l2fetch_AR(p, 0)`` compiles, executes, and earns full
        credit here for zero work -- verified 2026-08-02.

    ``L2FETCH_ACCESS`` (0x7e) -- "any access to the L2 cache from the L2 prefetch
    engine that was initiated by programing the L2FETCH engine" -- is the counter
    that proves real work. Note that the generic L2 counters do NOT substitute
    for it: ``L2_ACCESS`` (0x81) "does not include internally generated accesses
    like L2FETCH", and ``L2_IU_PREFETCH_ACCESS`` (0x78) "does not include
    L2FETCH-generated accesses". ``L2_DU_PREFETCH_ACCESS`` is the dcfetch /
    hardware-prefetcher path, not this one.

    See ``anticheat_runtime.mechanism_did_work(profile, "l2fetch")``.
    """
    for line in disasm_text.splitlines():
        if "\t" not in line:
            continue
        asm = line.rsplit("\t", 1)[-1]
        if _L2FETCH_INSN.search(asm):
            return True
    return False


# VTCM addressing. VTCM is a fixed physical aperture, not an instruction set: code
# "uses VTCM" by naming an address inside it, so unlike DMA/HMX there is no mnemonic
# to match -- we match the ADDRESS. Hexagon materialises such a 32-bit constant
# through the immediate extender, e.g. `immext(#0xd8402000)`.
#
# THE APERTURE IS EXACT, NOT "GENEROUS" (corrected 2026-08-02). Both bounds are now
# read from the vendor's own configuration table rather than assumed:
#
#   HVX PRM 80-N2040-47 Rev. E section 3.2 p15: "The size of the memory is
#   implementation-defined. The size is discoverable from the configuration table
#   defined in the V68 system architecture specification."
#
#   Config-table offsets come from the SDK's hexagon_standalone.h
#   (__vtcm_base=0x038, __tcm_size=0x03c), read via __rdcfg() on -mv68:
#       __rdcfg(__vtcm_base) = 0xd840  -> base 0xd8400000
#       __rdcfg(__tcm_size)  = 4096    -> 4096 KB = 4 MB
#   so the aperture is 0xd8400000-0xd87fffff, which the pattern below spans
#   EXACTLY. Units are KB, cross-validated: __rdcfg(__l2_array_size) = 1024 agrees
#   with the simulator's independently printed "L2$ 1024" column.
#
# NOTE hexbench/forge/oracle/sizing.py sets VTCM_BYTES = 256 KB. That is the minimum
# VTCM *allocation and page size*, not the capacity -- the SDK system-integration
# guide states "256KB is the minimum VTCM allocation size, 256K, 1M, 4M are
# supported page sizes" and partitions a 4 MB VTCM. It is wrong by 16x.
#
# Column-scoping is load-bearing here, not cosmetic. objdump prints the raw encoding
# bytes before the assembly, and those bytes routinely contain "d840" by coincidence
# -- an observed example is `8040d840 { r1:0 = vaslw(r1:0,#0x18) }`, a shift with no
# VTCM involvement at all. Matching the whole line would score it as VTCM use, so we
# match only the assembly field, exactly as _disasm_has_hvx/_hmx/_dma do.
# BUILT FROM THE TARGET, never written by hand (fixed 2026-08-03). A hardcoded
# aperture is exactly what breaks on retarget, and it breaks SILENTLY: the v68
# literal `#0x[dD]8[4-7]......` matches nothing on v73/v75/v79, where VTCM sits
# at 0xd9000000 instead of 0xd8400000. Every kernel would have reported
# used_vtcm=False -- fail-closed, so no false credit, but total blindness with
# nothing to flag it.
_VTCM_ADDR = re.compile(_target.vtcm_immediate_pattern(_target.current()))


def _disasm_has_vtcm(disasm_text: str) -> bool:
    """True iff any instruction line names an address inside the VTCM aperture.

    STATIC counterpart to the runtime PMU counter. The counter alone (UDMA_VTCM_RD/WR)
    only sees VTCM traffic that a *DMA* engine moved, so it misses two real, measured
    cases: VTCM used as direct HVX scratch (no DMA at all -- how the HMX helper works),
    and a single large DMA transfer, which registers 0 while the same bytes split into
    4 KB chunks register 480. Both under-count genuine use.

    Column-scoped + fail-closed + pure (no I/O), same convention as the others."""
    for line in disasm_text.splitlines():
        if "\t" not in line:
            continue  # symbol headers / blank lines have no instruction column
        asm = line.rsplit("\t", 1)[-1]  # assembly text is the last tab field
        if _VTCM_ADDR.search(asm):
            return True
    return False
