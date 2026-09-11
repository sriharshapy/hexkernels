"""Static gate on a candidate, BEFORE it is compiled or simulated.

WHY THIS EXISTS
---------------
Between "a model emits a translation unit" and "the simulator says INCORRECT
twenty minutes later" there was nothing. Every rule below encodes a mistake that
actually happened in batches 1-25 and cost at least one round:

| rule                    | what it cost                                          |
|-------------------------|-------------------------------------------------------|
| `dma-descriptor-storage`| a function-local descriptor failed 2,979 of 3,211,264 |
|                         | elements; a second kernel crashed 0x28 the same way   |
| `dma-descriptor-dstate` | a reused descriptor inherited COMPLETE and its        |
|                         | transfer was SILENTLY SKIPPED -- stale data, no fault |
| `dma-priming`           | two `dmstart`s as the program's first DMA act, 0x28   |
| `vtcm-aperture`         | a wrong base reads uninitialised scratchpad AS A      |
|                         | RESULT; one probe round was retracted over this       |
| `fabricated-intrinsic`  | an intrinsic that does not exist, caught only at the  |
|                         | compile the linter runs before                        |
| `tier-entitlement`      | a T0 kernel prefetching -- measured at ~9% cost for   |
|                         | nothing, and it exceeds what the size grants          |
| `vector-lane-width`     | `N = 64` floats where an HVX vector is 32 lanes: half |
|                         | the array uncomputed, every function over budget      |
| `entry-point-signature` | a mangled or renamed symbol fails to link             |
| `rd-leak`               | seven candidates included `harness_common.h`          |

SEVERITY IS NOT DECORATION. `error` means the finding is provable from the source
text alone; `warn` means the pattern is suspicious and a human should look. Only
`error` blocks, because a linter that blocks on heuristics gets switched off, and
a switched-off linter catches nothing. Where a rule could only be a heuristic it
is a `warn` and says so in its own message.

WHAT THIS IS NOT. It does not check that the kernel computes the right thing --
that is the harness's job and cannot be done statically. It checks the classes of
mistake that are invisible in the output, or that produce a wall of wrong values
whose cause is unrelated to the arithmetic.
"""
import os
import re
from dataclasses import dataclass

from hexkernels.core import target as _target
from hexkernels.core.toolchain import DEFAULT_SDK_ROOT, find_toolchain_bin
from hexkernels.forge.provenance import rd_leaks

#: Vendor headers a candidate may include. The identifier check scans all four,
#: not just `hvx_hexagon_protos.h` -- `Q6_dmstart_A`, `Q6_R_dmwait` and
#: `Q6_l2fetch_AP` live in `hexagon_protos.h`, so scanning one header would report
#: every DMA kernel's own intrinsics as fabricated.
VENDOR_HEADERS = ("hexagon_types.h", "hexagon_protos.h", "hvx_hexagon_protos.h",
                  "hmx_hexagon_protos.h", "hexagon_circ_brev_intrinsics.h")

#: Bytes per element, by the dtype name the pipeline uses. An HVX vector is 128
#: bytes, so lanes = 128 / this.
_WIDTH = {"float32": 4, "int32": 4, "uint32": 4, "float16": 2, "int16": 2,
          "uint16": 2, "int8": 1, "uint8": 1, "bool": 1, "int64": 8}


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str          # "error" (provable) | "warn" (heuristic)
    message: str
    evidence: str = ""

    def __str__(self):
        tail = f"  [{self.evidence}]" if self.evidence else ""
        return f"{self.severity.upper():5s} {self.rule}: {self.message}{tail}"


def _strip_comments(src: str) -> str:
    """Source with comments blanked but LINE COUNT and offsets preserved.

    Rules about code must not fire on prose. Every one of these headers carries a
    long comment naming the intrinsics and the failures it is about, so a rule
    matching `Q6_dmstart_A` or `harness_common.h` in a comment would flag the
    documentation the corpus is FOR. Newlines are kept so an evidence line number
    still means something.

    `rd-leak` is the deliberate exception and runs on the RAW source: a comment
    saying "I did not use the helper because ..." still evidences having read R&D
    material, and that rule's docstring says so.
    """
    out = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)),
                 src, flags=re.S)
    return re.sub(r"//[^\n]*", "", out)


#: Statements whose `keyword (...) { ... }` shape is indistinguishable from a
#: function definition by regex.
_KEYWORDS = frozenset(("if", "for", "while", "switch", "do", "else", "catch",
                       "return", "sizeof", "alignof", "static_assert"))


def _bodies(code):
    """`(name, start, end)` for every function definition, by brace matching.

    Needed because PRESENCE OF AN IDENTIFIER IS NOT USE OF A MECHANISM, and the
    first version of this module got that wrong in both directions:

    * The shared math core defines `l2fetch_block` as a `static inline` wrapper.
      Every kernel that includes the core carries the text `Q6_l2fetch_AP`, so the
      entitlement rule reported ten T0 kernels as prefetching when the wrapper is
      never called and the compiler drops it. That is exactly why the anti-cheat
      reads the linked object instead of the source.
    * `fp16_atan2` primes the DMA engine through a `dma_wait()` wrapper, so the
      priming rule saw two `dmstart`s with no wait between them and warned about a
      kernel that is correct.

    So a call THROUGH a one-level wrapper counts as a call, and a mention inside
    the wrapper's own definition does not.
    """
    out = []
    for m in re.finditer(r"\b(\w+)\s*\([^;{)]*\)\s*(?:const\s*)?\{", code):
        # `if (cond) {` matches this shape too, and without this the rule treated
        # `if` as a function named `if` whose body contained `Q6_dmstart_A` -- so
        # every `if (` in the kernel was counted as a dmstart call site and the
        # priming rule warned about four of its own correct templates.
        if m.group(1) in _KEYWORDS:
            continue
        i, depth = m.end() - 1, 0
        while i < len(code):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out.append((m.group(1), m.start(), i + 1))
    return out


#: Function names that are entry points -- reached by definition, not by a call
#: this translation unit contains.
_ENTRY = ("candidate_kernel", "main")


def _calls(code, intrinsic):
    """Offsets in the ENTRY POINT at which `intrinsic` is effectively reached.

    Resolves one level of wrapper. For each function whose body mentions the
    intrinsic:

    * if it is the entry point, its direct mentions are the use points;
    * otherwise it is a wrapper, and its use points are the places it is CALLED
      from outside its own body. A wrapper nobody calls contributes nothing --
      which is the whole point, because the shared math core defines
      `l2fetch_block` in every kernel that includes it and the compiler drops it
      when unused.

    The first version of this got it backwards: it treated every function
    mentioning the intrinsic as a wrapper and excluded that function's own body,
    which classified `candidate_kernel` as a wrapper for its own DMA and made 29
    T2/T3 kernels read as not using DMA at all. A rule can be wrong in the
    permissive direction and look like it works.

    One level, not a call graph. A wrapper reached only through a second wrapper
    under-reports, which is the safe direction for a gate that blocks.
    """
    pts = []
    rx_i = re.compile(rf"\b{re.escape(intrinsic)}\s*\(")
    for name, s, e in _bodies(code):
        if not rx_i.search(code[s:e]):
            continue
        if name in _ENTRY:
            pts += [s + m.start() for m in rx_i.finditer(code[s:e])]
            continue
        rx_n = re.compile(rf"\b{re.escape(name)}\s*\(")
        pts += [m.start() for m in rx_n.finditer(code)
                if not (s <= m.start() < e)]
    return sorted(pts)


def vendor_identifiers(sdk_root=DEFAULT_SDK_ROOT) -> frozenset:
    """Every `Q6_*` name the vendor headers declare."""
    inc = os.path.normpath(os.path.join(find_toolchain_bin(sdk_root), "..",
                                        "target", "hexagon", "include"))
    names = set()
    for h in VENDOR_HEADERS:
        p = os.path.join(inc, h)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            text = f.read()
        # both the function-like declarations and the `#define Q6_x(...)` forms
        names |= set(re.findall(r"\b(Q6_[A-Za-z0-9_]+)\s*[(\s]", text))
    return frozenset(names)


_VENDOR_CACHE = {}


def _vendor(sdk_root):
    if sdk_root not in _VENDOR_CACHE:
        try:
            _VENDOR_CACHE[sdk_root] = vendor_identifiers(sdk_root)
        except Exception:
            # No toolchain on this machine: the identifier rule cannot run, and
            # SKIPPING it is right. Returning an empty set would report every
            # intrinsic in every kernel as fabricated -- a check that fires when
            # it cannot see anything is worse than one that stands down.
            _VENDOR_CACHE[sdk_root] = None
    return _VENDOR_CACHE[sdk_root]


# --------------------------------------------------------------------- rules
def _canon_sig(text):
    """Whitespace-insensitive form of a declaration, for comparison only.

    Collapsing runs of whitespace is NOT enough, and the gap cost real data.
    `candidate_kernel( const _Float16 *x)` survives `\\s+ -> " "` as a MISMATCH
    against `candidate_kernel(const _Float16 *x)`, because the difference is a space
    added where there was none. That declaration links perfectly -- the compiler does
    not see formatting -- so the rule was failing kernels for pretty-printing. It
    rejected 19 of 342 graded rung-0 attempts (5.6%), each recorded as "did not
    compile", which understates correctness and blames the model for a whitespace
    preference.

    Spaces are therefore dropped next to `( ) , *` as well. Everything the rule
    exists to catch survives that: a renamed parameter, a changed type, a reordered
    parameter list and a missing `extern "C"` all still differ after
    canonicalisation, because names, types and order are all still compared.
    """
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s*([(),*])\s*", r"\1", text)


def _r_signature(code, signature):
    if not signature:
        return []
    want = _canon_sig(signature).strip()
    got = _canon_sig(code)
    if want in got:
        return []
    m = re.search(r'extern\s+"C"\s+void\s+candidate_kernel\s*\([^)]*\)',
                  re.sub(r"\s+", " ", code))
    return [Finding("entry-point-signature", "error",
                    "the entry point must match the prompt EXACTLY -- a renamed "
                    "parameter or a changed type does not link, and a missing "
                    'extern "C" mangles the symbol',
                    f"want {want!r}; found {m.group(0)!r}" if m
                    else "no candidate_kernel declaration found")]


def _r_fabricated(code, sdk_root):
    known = _vendor(sdk_root)
    if known is None:
        return []
    used = set(re.findall(r"\b(Q6_[A-Za-z0-9_]+)", code))
    bad = sorted(used - known)
    return [Finding("fabricated-intrinsic", "error",
                    "no vendor header declares this name, so it cannot compile. "
                    "Checked against " + ", ".join(VENDOR_HEADERS[:4]),
                    n) for n in bad]


_DESC_DECL = re.compile(
    r"^(?P<line>[^\n;]*\bhexagon_udma_descriptor_type\d_t\s+"
    r"(?P<name>\w+)\s*(\[[^\]]*\])?\s*(?P<attrs>[^;\n]*));", re.M)


def _r_descriptor_storage(code):
    """Descriptor STORAGE must be static and aligned.

    ALIGNMENT SEVERITY IS SET BY THE EVIDENCE, not by the prompt. The prompt asks
    for `aligned(64)`; 25 candidates across batches 2-15 declare `aligned(32)` and
    all 25 are correct on the simulator, and a type0 descriptor is 32 bytes. So
    there is no measurement showing 32 fails, and calling it an error would have
    this module reject a quarter of a corpus it was written after. Below 32, or
    absent, is a different matter: absent means whatever the linker felt like.

    The `static` half IS an error and does have a measurement behind it: the same
    kernel with function-local descriptors failed 2,979 of 3,211,264 elements and
    with static ones failed 0, pipeline untouched.
    """
    out = []
    for m in _DESC_DECL.finditer(code):
        line, name, attrs = m.group("line"), m.group("name"), m.group("attrs")
        if re.search(r"\btypedef\b", line):
            continue                                  # a type alias, not storage
        if re.search(r"\bhexagon_udma_descriptor_type\d_t\s*\*", line):
            continue                                  # a pointer, not storage
        if not re.search(r"\bstatic\b", line):
            out.append(Finding(
                "dma-descriptor-storage", "error",
                f"descriptor `{name}` is not `static`. It is read by an EXTERNAL "
                "engine, which does not tolerate the alignment a stack frame "
                "happens to give a 32-byte struct: measured, the same kernel with "
                "function-local descriptors failed 2,979 of 3,211,264 elements "
                "and with static ones failed 0, pipeline untouched",
                line.strip()))
        al = re.search(r"aligned\s*\(\s*(\d+)\s*\)", attrs or "")
        if not al:
            out.append(Finding(
                "dma-descriptor-storage", "error",
                f"descriptor `{name}` has no alignment attribute, so its address "
                "is whatever the linker chose; declare it "
                "`__attribute__((aligned(64)))`", line.strip()))
        elif int(al.group(1)) < 32:
            out.append(Finding(
                "dma-descriptor-storage", "error",
                f"descriptor `{name}` is aligned to {al.group(1)} bytes, less than "
                "the 32-byte descriptor itself", line.strip()))
        elif int(al.group(1)) < 64:
            out.append(Finding(
                "dma-descriptor-storage", "warn",
                f"descriptor `{name}` is aligned to {al.group(1)}; the prompt asks "
                "for 64. 32 is the descriptor's own size and 25 candidates in this "
                "corpus use it and are correct, so this is a consistency note, not "
                "a defect", line.strip()))
    return out


def _r_descriptor_dstate(code):
    """A descriptor written field by field must set `dstate`.

    The engine sets `dstate` to COMPLETE when a transfer retires. A descriptor
    reused without clearing it is a transfer that is SILENTLY SKIPPED -- the
    destination keeps the previous tile's data, there is no fault, and the kernel
    reports a plausible wrong answer. Detected by the presence of `length` writes
    without any `dstate` write, which is the shape every hand-rolled fill has.
    """
    if not re.search(r"\bhexagon_udma_descriptor_type\d_t\b", code):
        return []
    writes_len = re.search(r"(?:->|\.)\s*length\s*=", code)
    if not writes_len:
        return []
    if re.search(r"(?:->|\.)\s*dstate\s*=", code):
        return []
    return [Finding("dma-descriptor-dstate", "error",
                    "this kernel writes descriptor fields but never assigns "
                    "`dstate`. The engine leaves it COMPLETE on retirement, and a "
                    "descriptor inherited in that state has its transfer SKIPPED "
                    "with no error -- set it to "
                    "HEXAGON_UDMA_DESC_DSTATE_INCOMPLETE on every use",
                    writes_len.group(0))]


def _r_priming(code):
    # Through `_calls`, so a wait reached through a wrapper counts: `fp16_atan2`
    # and `i8_depthwise_conv2d_stream` both prime through `dma_wait()` /
    # `dma_wait_all()`, and matching the bare intrinsic warned about two kernels
    # that are correct.
    starts = _calls(code, "Q6_dmstart_A")
    if len(starts) < 2:
        return []
    waits = _calls(code, "Q6_R_dmwait")
    if any(starts[0] < w < starts[1] for w in waits):
        return []
    return [Finding("dma-priming", "warn",
                    "no `Q6_R_dmwait` appears between the first two "
                    "`Q6_dmstart_A` calls. A second dmstart may only be "
                    "outstanding after the program's FIRST transfer has "
                    "completed -- two as the engine's first act fault 0x28. "
                    "Prime with one start/wait pair, or chain descriptors "
                    "through `next` and issue ONE start per tile. Textual order "
                    "is not execution order, so this is a warning: check it",
                    f"dmstart at offsets {starts[0]} and {starts[1]}")]


def _r_vtcm_aperture(code, tgt):
    """Every literal that looks like a VTCM base must BE this target's base.

    A wrong base does not fault -- it reads a different part of the scratchpad,
    which in this corpus once meant reading uninitialised VTCM AS A RESULT and
    publishing a conclusion that had to be retracted. `0xd8...`/`0xd9...` is the
    aperture range across the cores the pipeline targets, so a literal in that
    range that is not the base is worth naming.
    """
    out = []
    # EVERY SUPPORTED TARGET'S BASE IS LEGAL, not just the current one. Each DMA
    # kernel in this corpus opens with
    #     #if __HEXAGON_ARCH__ >= 73 ... 0xd9000000u #else 0xd8400000u #endif
    # so a rule comparing against `current()` alone condemned the v68 arm of a
    # correct conditional in 48 of 125 kernels -- flagging portability as a defect.
    legal = {t.vtcm_base for t in _target.TARGETS.values()}
    # `(?![0-9a-fA-F])` and not `\b`: every VTCM base here is written
    # `0xd9000000u`, and a trailing `\b` cannot match before the `u` suffix -- so
    # the first version of this rule matched NOTHING on any kernel, and its
    # silence looked like a pass. Caught only by planting a wrong base and
    # noticing a DIFFERENT rule was what fired.
    for m in re.finditer(r"\b0[xX](d[0-9a-fA-F]{7})(?![0-9a-fA-F])", code):
        val = int(m.group(1), 16)
        if val in legal:
            continue
        if tgt.vtcm_base < val < tgt.vtcm_end:
            out.append(Finding(
                "vtcm-aperture", "warn",
                f"{m.group(0)} is inside {tgt.core}'s VTCM aperture but is not "
                f"its base (0x{tgt.vtcm_base:08x}). Legal, but write it as an "
                "offset from a named base so it moves with the target",
                m.group(0)))
        else:
            out.append(Finding(
                "vtcm-aperture", "error",
                f"{m.group(0)} looks like a VTCM address but is outside "
                f"{tgt.core}'s aperture "
                f"[0x{tgt.vtcm_base:08x}, 0x{tgt.vtcm_end:08x}). A wrong base "
                "does not fault -- it reads a different part of the scratchpad "
                "and returns whatever is there",
                m.group(0)))
    return out


def _r_entitlement(code, tier, mechanisms):
    """Using MORE than the size grants is an error; using less is a choice.

    The entitlement is a ceiling. `fp32_flipud` is a pure row permutation and
    genuinely uses no compute mechanism at all, so a rule requiring every grant to
    be exercised would flag the one kernel that is being honest.
    """
    if tier is None:
        return []
    out = []
    granted = set(mechanisms or ())
    # CALLS, not mentions. The shared math core DEFINES `l2fetch_block`, so every
    # kernel including it carries the text `Q6_l2fetch_AP` whether or not it
    # prefetches; matching the identifier reported ten correct T0 kernels as
    # exceeding their entitlement. This is the same reason the anti-cheat reads
    # the linked object rather than the source.
    uses_l2 = bool(_calls(code, "Q6_l2fetch_AP"))
    uses_dma = bool(_calls(code, "Q6_dmstart_A"))
    if uses_l2 and "l2fetch" not in granted:
        out.append(Finding(
            "tier-entitlement", "error",
            f"{tier} does not grant `l2fetch` (granted: "
            f"{sorted(granted) or 'none'}), because the working set fits the "
            "cache this prefetch would fill. Measured elsewhere in this corpus "
            "at ~9% cost for no gain -- the hardware prefetcher already covers a "
            "short sequential stream", "Q6_l2fetch_AP"))
    if uses_dma and "dma" not in granted:
        out.append(Finding(
            "tier-entitlement", "error",
            f"{tier} does not grant `dma` (granted: "
            f"{sorted(granted) or 'none'})", "Q6_dmstart_A"))
    if "dma" in granted and not uses_dma:
        out.append(Finding(
            "tier-entitlement", "warn",
            f"{tier} grants `dma` and this kernel does not use it. Not an error "
            "-- the grant is a ceiling -- but at this size the input cannot be "
            "resident, so check that the omission is deliberate", ""))
    return out


def _r_lane_width(code, dtype, nelem):
    """A vector-count divisor that disagrees with the element width.

    An HVX vector is 128 bytes: 32 fp32 lanes, 64 fp16, 128 int8. Dividing the
    element count by the wrong one is the bug that made every function in the
    math harness report over budget with its first error at index 32 -- `N = 64`
    floats, one vector computed, the other half of the array untouched. It looked
    like sixteen broken functions.
    """
    out = []
    # TWO INDEPENDENT CHECKS, and gating both on `dtype` was a bug: the element
    # total is provable from `nelem` alone, and a caller that passed `nelem`
    # without `dtype` got no check at all -- which is how a planted wrong NELEM
    # went unreported while the other nine negative cases fired.
    if dtype in _WIDTH:
        want = 128 // _WIDTH[dtype]
        for m in re.finditer(r"/\s*(\d+)\s*\)?\s*$|/\s*(32|64|128)\b", code, re.M):
            div = int(m.group(1) or m.group(2))
            if div in (32, 64, 128) and div != want:
                out.append(Finding(
                    "vector-lane-width", "warn",
                    f"dividing by {div} to count vectors, but a 128-byte vector "
                    f"holds {want} {dtype} lanes. If this is a vector count it is "
                    f"wrong by {want / div:g}x and the tail of the buffer is never "
                    "touched; if it is a tile or byte count, ignore this",
                    m.group(0).strip()))
    # The provable half: an explicit total that is not the task's total.
    if nelem:
        for m in re.finditer(r"#define\s+\w*(?:NELEM|N_ELEM|TOTAL)\w*\s+(\d+)",
                             code):
            if int(m.group(1)) != nelem:
                out.append(Finding(
                    "vector-lane-width", "error",
                    f"declares {m.group(1)} elements; the task has {nelem}",
                    m.group(0)))
    return out


def _r_rd_leak(raw_source):
    hits = rd_leaks(raw_source)
    if not hits:
        return []
    return [Finding("rd-leak", "error",
                    "references R&D-only material. Nothing under "
                    "hexbench/env/harness/ may appear in a forge v2 kernel, and "
                    "a MENTION counts -- carrying the R&D spelling of a hardware "
                    "fact is evidence of having read the R&D file instead of the "
                    "documentation", t) for t in hits]


# ----------------------------------------------------------------- interface
def lint(source: str, *, signature=None, tier=None, mechanisms=(), nelem=None,
         dtype=None, sdk_root=DEFAULT_SDK_ROOT, tgt=None) -> list:
    """Every finding for one candidate translation unit, worst first."""
    tgt = tgt or _target.current()
    code = _strip_comments(source)
    findings = (
        _r_signature(code, signature)
        + _r_rd_leak(source)                       # raw: mentions count
        + _r_fabricated(code, sdk_root)
        + _r_descriptor_storage(code)
        + _r_descriptor_dstate(code)
        + _r_priming(code)
        + _r_vtcm_aperture(code, tgt)
        + _r_entitlement(code, tier, mechanisms)
        + _r_lane_width(code, dtype, nelem)
    )
    order = {"error": 0, "warn": 1}
    return sorted(findings, key=lambda f: (order[f.severity], f.rule, f.evidence))


def errors(findings) -> list:
    return [f for f in findings if f.severity == "error"]


def format_findings(findings) -> str:
    return "\n".join(str(f) for f in findings)
