"""Runtime mechanism evidence -- did the instruction RUN, and did it DO WORK?

WHY THIS MODULE EXISTS
======================
``anticheat.py`` answers exactly one question: *is the instruction present in the
disassembly?* That is necessary but not sufficient, and the gap is measurable.

Demonstrated 2026-08-02 on this toolchain. A kernel whose HVX compute and
``l2fetch`` sit inside a never-taken branch (``volatile int gate = 0``), plus one
live ``Q6_l2fetch_AR(buf, 0)`` with Height=0::

    STATIC (anticheat.py)        RUNTIME (--packet_analyze)
    used_hvx         = True      HVX compute: 2 static hits, 0 executed
    used_hvx_compute = True      l2fetch:     3 static hits, 1 executed
    used_l2fetch     = True      L2FETCH_COMMAND_KILLED=1, L2FETCH_ACCESS=0

Every static detector granted credit for work that never happened. This module
supplies the two missing questions, both answerable from documented simulator
output:

1. **Did it execute?** ``--packet_analyze`` reports a per-packet ``commits``
   count. A packet with no ``commits`` is present in the ELF and never ran.
2. **Did it do work?** PMU event counters distinguish a mechanism that moved
   bytes from one that was issued and then killed, dropped, or starved.

SCOPE -- what this does NOT fix
-------------------------------
Code that executes and whose *result is discarded* into an unused buffer still
passes. Detecting that requires differential ablation (strip the mechanism,
re-run, compare output), which is out of scope here. Conformance numbers remain
upper bounds, just tighter ones.

ADDRESS SPACES -- read before wiring this in
--------------------------------------------
``evaluate.py`` currently disassembles ``candidate.o``, a **relocatable object**
whose addresses are unrelocated and start at zero. ``--packet_analyze`` reports
**linked-ELF runtime virtual addresses**. The two cannot be correlated directly.
Callers must disassemble the *linked* ELF and scope to the candidate's symbols
(see ``executed_disasm``), or the filter silently matches nothing and every
mechanism reads as unexecuted -- which fails closed, but for the wrong reason.

ENVIRONMENT THESE CITATIONS DESCRIBE
------------------------------------
* Hexagon SDK **6.4.0.2**, tools **19.0.04**
* Target ``-mv68`` -> rev_id ``0x00008d68`` (``v68n_1024``), Hexagon 780 /
  Snapdragon 888-class; 6 HW threads, L1-I 32 KB, L1-D 16 KB, L2 1024 KB,
  HVX 128B
* Counter semantics: **Hexagon V68 Programmer's Reference Manual,
  80-N2040-46 Rev. B, ch 9 "PMU Events", p140-152**
* Packet-statistics format: **Hexagon Simulator User Guide, 80-N2040-1786
  Rev. AF, section 3.9, p45**

The simulator must run with ``--timing`` for stall and event data, and with
``--packet_analyze <file>`` to emit the JSON. See
``docs/hexagon/SDK_DOC_INDEX.md`` for the full citation chain.
"""
import collections
import json
import re

from hexkernels.anticheat import anticheat
from hexkernels.core import target as _target

# --- Environment this module's citations describe -------------------------
SDK_VERSION = "6.4.0.2"
TOOLS_VERSION = "19.0.04"
TARGET_ARCH = "v68"
TARGET_REVID = "0x00008d68"  # v68n_1024

# --- Packet-statistics parsing -------------------------------------------
# Simulator UG 80-N2040-1786 Rev. AF section 3.9 p45 documents the schema and
# states: "The JSON format used in a packet statistics file will not change in
# the future." Observed schema (verified 2026-08-02) is richer than the guide's
# example -- the guide shows `bus`, the simulator emits `events`:
#
#   { "version": "2.5",
#     "siminfo": { "revid": "8d68", "core": "V68N_1024",
#                  "cache_config": "L1-I$ = 32 Kb, L1-D$ = 16 Kb, L2-$ = 1024 Kb" },
#     "core_packet_profile": {
#       "0x000069bc": { "commits": 12,
#                       "stalls": { "TOTAL_STALLS": 195, ... },
#                       "events": { "AXI_READ_REQUEST": 2, ... } } } }
_PROFILE_KEY = "core_packet_profile"


def parse_packet_profile(json_text: str) -> dict:
    """Parse ``--packet_analyze`` JSON into ``{packet_start_address: entry}``.

    Addresses are the hex string keys of ``core_packet_profile`` converted to
    int. Entries are passed through unchanged (``commits`` / ``stalls`` /
    ``events``); note that a packet may legitimately carry none of those keys,
    which means it was never committed.

    Fail-closed: malformed JSON, a missing profile section, or an unparseable
    address yields an empty mapping rather than raising, so a profiling failure
    can never *grant* mechanism credit.
    """
    try:
        doc = json.loads(json_text)
    except (ValueError, TypeError):
        return {}
    if not isinstance(doc, dict):
        return {}
    profile = doc.get(_PROFILE_KEY)
    if not isinstance(profile, dict):
        return {}
    out = {}
    for key, entry in profile.items():
        try:
            out[int(key, 16)] = entry if isinstance(entry, dict) else {}
        except (ValueError, TypeError):
            continue
    return out


def siminfo(json_text: str) -> dict:
    """Return the ``siminfo`` block: revid, core, q6version, cache_config, and
    the full simulator command line.

    Every profiling run self-documents the configuration it was produced under,
    which is what makes a measurement citable rather than merely asserted. The
    observed ``cache_config`` on our target reads
    ``"L1-I$ = 32 Kb, L1-D$ = 16 Kb, L2-$ = 1024 Kb"``, independently
    reconfirming the L1-D and L2 figures used by the tier rules.
    """
    try:
        doc = json.loads(json_text)
    except (ValueError, TypeError):
        return {}
    info = doc.get("siminfo") if isinstance(doc, dict) else None
    return info if isinstance(info, dict) else {}


def executed_packet_addresses(profile: dict) -> frozenset:
    """Packet start addresses that actually committed at least once.

    ``commits`` is the packet execution count (Simulator UG section 3.9). A
    packet present in the profile with no ``commits`` key, or a count of zero,
    did not execute.
    """
    return frozenset(
        addr for addr, entry in profile.items()
        if isinstance(entry, dict) and (entry.get("commits") or 0) > 0
    )


# --- Disassembly packet segmentation -------------------------------------
# Hexagon is VLIW: instructions are grouped into brace-delimited packets, and
# `--packet_analyze` keys on the packet's START address. objdump prints one line
# per instruction, so an instruction address must be mapped to its containing
# packet before it can be correlated.
#
# Verified format (hexagon-llvm-objdump 19.0.04). Tab-separated fields are
# address, encoding bytes, word + brace column, assembly:
#
#   '      44:\t00 40 00 7f\t7f004000 { \tnop'          <- packet opens
#   '      48:\t00 40 00 7f\t7f004000   \tnop'          <- continuation
#   '      4c:\t00 c0 00 7f\t7f00c000   \tnop } '       <- packet closes
#   '    4e28:\t07 d3 17 d3\td317d307 { \t... } '       <- single-instruction packet
#
# The opening '{' lives in the brace column (second-to-last field); the closing
# '}' lives in the assembly column (last field). Lines with fewer than two tabs
# (symbol headers, '<unknown>' encodings, blanks) carry no packet structure.
_ADDR = re.compile(r"^\s*([0-9a-fA-F]+):")


def iter_packets(disasm_text: str):
    """Yield ``(packet_start_address, [line, ...])`` for each packet in order.

    Lines outside any packet (symbol headers, blanks, ``<unknown>`` encodings)
    are yielded with a start address of ``None`` so callers can preserve or drop
    them deliberately rather than by accident.
    """
    start = None
    buf = []
    for line in disasm_text.splitlines():
        fields = line.split("\t")
        if len(fields) < 3:
            if start is None:
                yield None, [line]
            else:
                buf.append(line)  # malformed line inside an open packet
            continue
        m = _ADDR.match(line)
        opens = "{" in fields[-2]
        closes = "}" in fields[-1]
        if opens and start is None:
            start = int(m.group(1), 16) if m else None
            buf = [line]
        else:
            buf.append(line)
        if closes and buf:
            yield start, buf
            start, buf = None, []
    if buf:
        yield start, buf


def executed_disasm(disasm_text: str, executed: frozenset) -> str:
    """Return only the disassembly of packets that actually executed.

    This is deliberately a *filter*, not a new detector: the output is ordinary
    objdump text, so every existing detector in ``anticheat.py`` can be run
    against it unchanged. The frozen HVX/HMX logic keeps its exact semantics and
    simply stops seeing code that never ran::

        executed = executed_packet_addresses(parse_packet_profile(js))
        live = executed_disasm(disasm_of_linked_elf, executed)
        genuine_hvx = anticheat._disasm_has_hvx_compute(live)

    Fail-closed: an empty ``executed`` set yields empty text, so a missing or
    failed profiling run withholds all mechanism credit rather than granting it.

    Note the address-space requirement in this module's docstring -- ``disasm_text``
    must come from the **linked ELF**, not from ``candidate.o``.
    """
    out = []
    for start, lines in iter_packets(disasm_text):
        if start is not None and start in executed:
            out.extend(lines)
    return "\n".join(out)


# --- Symbol scoping -------------------------------------------------------
# The execution filter requires linked-ELF addresses, but the linked ELF also
# contains the harness, CRT and libc. evaluate.py's existing design deliberately
# disassembles `candidate.o` alone so that "harness/CRT/libc are a different
# translation unit and cannot leak HVX credit" -- that property must survive.
#
# So: take the symbol names DEFINED by candidate.o, and keep only those regions
# of the linked-ELF disassembly. objdump emits a symbol header line per region:
#
#     00000000 <candidate_kernel>:
#
# `static inline` helpers are inlined into the candidate's own symbols and are
# therefore inside the kept region, which is the intended behaviour: calling a
# helper that emits real intrinsics counts as genuine use.
_SYM_HEADER = re.compile(r"^[0-9a-fA-F]+\s+<([^>]+)>:\s*$")


def defined_symbols(obj_disasm: str) -> frozenset:
    """Symbol names defined in an objdump -d of the candidate-only object."""
    return frozenset(
        m.group(1) for m in (_SYM_HEADER.match(l) for l in obj_disasm.splitlines()) if m
    )


def iter_symbol_regions(disasm_text: str):
    """Yield ``(symbol_name, region_text)`` for each symbol in a disassembly."""
    name, buf = None, []
    for line in disasm_text.splitlines():
        m = _SYM_HEADER.match(line)
        if m:
            if name is not None:
                yield name, "\n".join(buf)
            name, buf = m.group(1), []
            continue
        if name is not None:
            buf.append(line)
    if name is not None:
        yield name, "\n".join(buf)


def has_hmx_matmul_in_one_region(disasm_text: str) -> bool:
    """True iff a SINGLE symbol contains both HMX matrix operands.

    ``anticheat._disasm_has_hmx`` matches ``activation.`` and ``weight.``
    anywhere in the text, so the pair can be satisfied by two unrelated
    functions -- an `activation.` load in one and a `weight.` load in another
    never compose into a multiply. This tightens the same evidence to per-symbol
    scope, which is the right granularity: a matmul loop lives in one function,
    and `static inline` helpers are inlined into their caller's symbol (so
    calling the HMX helper still counts, as intended).

    Still not a proof of a multiply -- two operands in one large function need
    not feed the same instruction. Basic-block or dataflow scoping would be
    tighter and needs a CFG.
    """
    for _name, region in iter_symbol_regions(disasm_text):
        if anticheat._disasm_has_hmx(region):
            return True
    return False


def scope_to_symbols(elf_disasm: str, names) -> str:
    """Keep only the regions of a linked-ELF disassembly belonging to ``names``.

    Preserves the candidate-only judging property of the existing static path
    while yielding linked-ELF addresses that correlate with ``--packet_analyze``.

    Fail-closed: an empty ``names`` yields empty text.
    """
    if not names:
        return ""
    out, keep = [], False
    for line in elf_disasm.splitlines():
        m = _SYM_HEADER.match(line)
        if m:
            keep = m.group(1) in names
            if keep:
                out.append(line)
            continue
        if keep:
            out.append(line)
    return "\n".join(out)


# --- PMU work evidence ----------------------------------------------------
def event_totals(profile: dict) -> collections.Counter:
    """Sum every PMU event across all packets in the profile."""
    total = collections.Counter()
    for entry in profile.values():
        if isinstance(entry, dict):
            events = entry.get("events")
            if isinstance(events, dict):
                total.update({k: v for k, v in events.items()
                              if isinstance(v, (int, float))})
    return total


# Counters proving a mechanism did real work, and counters proving an issued
# command achieved nothing. All symbols and codes from V68 PRM 80-N2040-46
# Rev. B ch 9, p140-152.
#
# l2fetch is the only mechanism fully mapped here, and it is the one that most
# needs it -- the PRM documents seven ways an l2fetch can be present, even
# execute, and still move nothing (five with counters, plus "if the lines of
# interest are already in the L2, no action is performed" (section 5.10.6 p98),
# plus a legal Height=0 or Width=0 encoding that fetches nothing).
WORK_COUNTERS = {
    # 0x7e "L2FETCH access from the data unit. Any access to the L2 cache from
    # the L2 prefetch engine that was initiated by programing the L2FETCH
    # engine." CONFIRMED: fires on i8_satadd_xl / i8_relu_l2fetch.
    "l2fetch": ("L2FETCH_ACCESS",),
    # HVX PRM 80-N2040-47 Rev. E ch 4 p21-22, events 296-299: "Executed simple
    # ALU instruction" / "Executed multiply or abs-diff" / "Executed bit shift
    # or bit count" / "Executed permute or cross-lane data movement". These
    # count EXECUTED vector compute, excluding pure loads and stores -- exactly
    # the load-only-HVX discriminator, at runtime.
    # CONFIRMED 2026-08-02: hvx_vnot_i8 -> ALU 21; dequant_i8_i16_dma -> ALU
    # 70655, MPY 30726, SHIFT 18432, PERM 6144.
    "hvx_compute": ("HVXPIPE_ALU", "HVXPIPE_MPY", "HVXPIPE_SHIFT", "HVXPIPE_PERM"),
}

# DO NOT add HVX_PKT_THREAD (event 274). Despite the name it is "committed
# packets on a thread with the XE bit set, whether executed in Q6 or
# coprocessor" -- i.e. every packet on an HVX-enabled thread, HVX or not.
# Measured: 15,958 on hvx_vnot_i8, a kernel with 14 HVX packets; and non-zero on
# a probe containing no HVX at all. It is not evidence of HVX use.
#
# VTCM has NO usable work counter. Measured negative result 2026-08-02: on
# dequant_i8_i16_dma -- a real expert declaring ['dma','hvx','vtcm'] -- both
# HVXST_VTCM (event 294, "Vector store to VTCM") and TCM_DU_ACCESS (0x87) are
# ABSENT from the profile entirely. Plausibly because its VTCM writes come from
# the DMA engine rather than HVX vector stores, but unexplained. An earlier
# draft of this file mapped TCM_DU_ACCESS for VTCM; that mapping never fired and
# was removed rather than left in place looking like coverage. Use the address
# region via `region_dead_stores` instead.
#
# HMX has no documented counter. The HVX PRM says coprocessor events live at
# >=384 in the V68 PRM; the V68 PRM's ch 9 does not list them. The only
# candidate, COPROC_BUSY_PVIEW_CYCLES (0xed), also fires for pure HVX (14 on
# hvx_vnot_i8), so it cannot attribute work to HMX.

# Counters proving an l2fetch command was issued and then achieved nothing.
# Presence of these ALONE (with no L2FETCH_ACCESS) is affirmative evidence of a
# futile prefetch -- which is exactly what the Height=0 demonstration produced.
FUTILE_COUNTERS = {
    "l2fetch": (
        "L2FETCH_COMMAND_KILLED",            # 0x92 killed by a Stop command
        "L2FETCH_COMMAND_OVERWRITE",         # 0x93 superseded by a later command
        "L2FETCH_ACCESS_CREDIT_FAIL",        # 0x94 blocked, no L2FETCH/L2evict credit
        "L2FETCH_COMMAND_PAGE_TERMINATION",  # 0xd3 no VA->PA translation, or permission error
        "L2FETCH_DROP",                      # 0xe0 dropped, prior eviction incomplete
    ),
}

# Mechanisms with no usable counter, for the reasons documented above. Work
# evidence is UNKNOWN for these and the gate falls back to execution evidence
# alone. DMA is here because the AXI_* family measures system-wide off-chip
# traffic and cannot attribute it to the user-DMA engine.
UNMAPPED = ("hvx", "hmx", "dma", "vtcm")


def mechanism_did_work(profile: dict, mechanism: str):
    """Did this mechanism do real work? ``True`` / ``False`` / ``None``.

    * ``True``  -- a work counter is non-zero: the mechanism demonstrably ran.
    * ``False`` -- work counters are zero AND a futility counter fired: the
      mechanism was issued and provably achieved nothing.
    * ``None``  -- no counter is mapped for this mechanism, or the profile
      carries no relevant evidence either way. **Unknown, not negative.**

    The tri-state matters. Collapsing ``None`` into ``False`` would reject every
    genuine HVX and DMA kernel purely because those counters are not mapped yet,
    turning an incomplete mapping into a false accusation.
    """
    totals = event_totals(profile)
    work = sum(totals.get(c, 0) for c in WORK_COUNTERS.get(mechanism, ()))
    if work > 0:
        return True
    futile = sum(totals.get(c, 0) for c in FUTILE_COUNTERS.get(mechanism, ()))
    if futile > 0:
        return False
    return None


# --- Dead-store detection (the discarded-result hole) ---------------------
# Execution evidence closes "present but never ran". It does NOT close the case
# that motivated all of this: second light found generators writing HMX assembly
# into a VTCM scratch buffer that was then DISCARDED -- the ELF shows a genuine
# matmul, the code executes, and the real output came from HVX. `commits` is
# non-zero, so the execution filter passes it straight through.
#
# The compiler is no help. Measured 2026-08-02, -O2:
#   plain local scratch, result never read   -> ELIMINATED (dead-store elimination)
#   scratch at a VTCM absolute address       -> SURVIVED
#   size-guarded fast path, never taken      -> SURVIVED
# Every real mechanism buffer is an integer cast to a fixed hardware address
# (see hmx_helpers.h), which the compiler cannot prove dead. So dead-store
# elimination protects only against the naive case nobody was worried about.
#
# This detector asks a narrower question that needs no taint analysis and no
# register tracking: **were the bytes written into the mechanism's region ever
# read back?** Staged data that is never re-read went nowhere.
#
# Memory-trace format (Simulator UG 80-N2040-1786 Rev. AF section 3.7.2 p36,
# verified against a real --memtrace run 2026-08-02):
#
#   TNUM=0:TYPE=DR:PCYC=3454:PC=1b0:VA=30:PA=30:WIDTH=4:DATA=0x00000001:1:
#
# Cost: 4.9 MB of trace for a program that prints six values. Strictly opt-in;
# never enable this on a full sweep without filtering (section 3.3.16).
_MT = re.compile(r"TYPE=(D[RW]):.*?:VA=([0-9a-fA-F]+):.*?:WIDTH=(\d+)")

# VTCM aperture for the ACTIVE target, read from its configuration table.
# Never hardcode: v68 is 4 MB @ 0xd8400000, v75 is 8 MB @ 0xd9000000, and a
# literal from one silently matches nothing on the other.
VTCM_LO = _target.current().vtcm_base
VTCM_HI = _target.current().vtcm_end


def region_dead_stores(memtrace_lines, lo: int = VTCM_LO, hi: int = VTCM_HI) -> dict:
    """Bytes written into ``[lo, hi)`` that are never subsequently read back.

    ``memtrace_lines`` is any iterable of trace lines (stream a file handle --
    these traces are large).

    Returns ``{stores, loads, written_bytes, dead_bytes, dead_fraction}``.

    ONE-SIDED, and this is the important caveat. Bytes written and never read
    are *proof* the staged data was discarded. The converse does not hold: data
    can be read back and still not reach the output, so ``dead_bytes == 0`` is
    NOT proof of genuine use. Treat this as a detector of decorative mechanism
    use, never as a certificate of honest use -- which is the correct shape for
    an anti-cheat.
    """
    pending = set()   # byte addresses written into the region, not yet read
    dead = set()      # written, and still unread when overwritten or at exit
    stores = loads = written = 0
    for line in memtrace_lines:
        m = _MT.search(line)
        if not m:
            continue
        kind, va_s, w_s = m.group(1), m.group(2), m.group(3)
        try:
            va, width = int(va_s, 16), int(w_s)
        except ValueError:
            continue
        if va + width <= lo or va >= hi:
            continue
        addrs = range(max(va, lo), min(va + width, hi))
        if kind == "DW":
            stores += 1
            written += len(addrs)
            pending.update(addrs)
        else:
            loads += 1
            pending.difference_update(addrs)
    return {
        "stores": stores,
        "loads": loads,
        "written_bytes": written,
        "dead_bytes": len(pending),
        "dead_fraction": (len(pending) / written) if written else 0.0,
    }


def genuine_mechanism(disasm_text: str, profile: dict, mechanism: str,
                      static_detector) -> bool:
    """Combined verdict: present **and** executed **and** not provably futile.

    ``static_detector`` is any ``anticheat._disasm_has_*`` function; it is run
    against the execution-filtered disassembly, so the frozen detection logic is
    reused verbatim rather than reimplemented.

    Fail-closed at every step -- an empty profile, an empty filtered disassembly,
    or a proven-futile counter all withhold credit.
    """
    executed = executed_packet_addresses(profile)
    if not executed:
        return False
    if not static_detector(executed_disasm(disasm_text, executed)):
        return False
    return mechanism_did_work(profile, mechanism) is not False
