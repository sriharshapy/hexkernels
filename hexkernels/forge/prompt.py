"""Build the acceleration prompt: scalar reference + schedule + mechanism budget.

WHAT THIS STAGE ASKS FOR
------------------------
Everything before this point produces the *question*: a portable scalar C++
kernel that computes the right answer slowly. This module turns that question
into the text a model is asked to answer, and the whole design question is what
to put in it.

Three things go in, and the middle one is the reason this module exists:

1. **The scalar reference** -- the semantics. Measured earlier on this
   benchmark: handing a model the reference took compute-only tasks from 0 to
   6/12, beating hand-written prompts. It answers "what should this compute".

2. **The schedule annotation** -- `iterator_types`, indexing maps, and which
   loop is vectorisable. The same measurement found the reference did *nothing*
   for memory-schedule tasks (0/7), because a scalar loop nest says nothing
   about how to walk memory. This is the part that addresses that, and it is
   the Linalg information (`hexkernels.forge.frontend.schedule`).

3. **The mechanism budget** -- which mechanisms the size entitles this kernel
   to, each with the arithmetic that justifies it. Stating the budget rather
   than naming a mechanism keeps the request honest: below L2 the data is
   already resident, so asking for DMA there would be asking for something that
   cannot help.

WHAT IS DELIBERATELY WITHHELD
-----------------------------
No intrinsic names, no worked example, no target code. Supplying those would
make the benchmark measure retrieval rather than the thing it is built to
measure. The prompt states the shape of the problem and the hardware's
parameters; choosing `Q6_Vw_vadd_VwVw` over a scalar loop is the task.

Prompt construction is pure and deterministic -- same spec in, byte-identical
text out -- so a prompt can be diffed across runs and a result attributed to a
prompt change rather than to sampling.
"""
from hexkernels.core import target as _target
from hexkernels.forge.frontend import schedule as _schedule


def hardware_facts(tgt=None) -> str:
    """The target's parameters, probed rather than recited.

    Values come from `core.target`, which reads them from the part's own
    configuration table. A prompt that hardcoded them would go quietly stale on
    retarget -- and would then be telling the model a v68 VTCM size while the
    kernel is compiled and judged on v75.
    """
    t = tgt or _target.current()
    return "\n".join([
        f"Target: Hexagon {t.arch} ({t.core}), compiled with hexagon-clang++ -std=c++17.",
        f"  HVX vector width : {t.hvx_bytes} bytes",
        f"  L1 data cache    : {t.l1d_bytes} bytes",
        f"  L2 cache         : {t.l2_bytes} bytes",
        f"  VTCM             : {t.vtcm_bytes} bytes at {t.vtcm_base:#010x} "
        "(software-managed scratchpad, not part of the automatic hierarchy)",
    ])


def _schedule_section(graph) -> str:
    """The Linalg-equivalent annotation, per primitive."""
    return _schedule.graph_text(graph)


def _budget_section(plan) -> str:
    lines = [f"Working set: {plan.working_set_bytes} bytes -> tier {plan.tier}.",
             "Mechanisms this size justifies:"]
    if not plan.mechanisms:
        lines.append("  (none -- this size does not justify any special mechanism)")
    seen = set()
    for mech, why in plan.reasons:
        if mech in plan.mechanisms:
            lines.append(f"  {mech}: {why}" if mech not in seen
                         else f"        {why}")
            seen.add(mech)
    return "\n".join(lines)


# Carried IN the prompt rather than only in whatever brief accompanies it.
#
# This exists because the first batch was generated with the rule stated only in
# the dispatch brief, and it was omitted from two of them: one kernel was written
# from private reference solutions, and a second copied the pattern from that one
# without touching the private data itself. Contamination travelled a hop further
# than the original mistake, and both had to be discarded despite being correct.
#
# A constraint that lives beside the artifact travels with it; a constraint that
# lives in a covering note does not.
PROVENANCE_RULE = """\
Provenance (this governs whether the result is usable at all):
  - Derive this kernel from the reference, the schedule, and the vendor's
    intrinsic headers ONLY.
  - NOTHING from this repository may appear in the translation unit -- no
    `harness_common.h`, no `hmx_helpers.h`, nothing else under
    hexbench/env/harness/. Those are R&D material. The directory is not on the
    compile's include path, so such an include is a build failure, and a COPIED
    body (the VTCM base, the alignment attribute, a crouton offset, a tolerance
    compare) is rejected before compiling. Hardware facts are yours to derive
    under your own names, with a comment saying where you got them.
  - Do not copy from, or pattern it on, any existing solution to this or a
    similar kernel -- including any private reference corpus and including other
    kernels generated alongside this one.
  - This pipeline's claim is that its corpus comes from public sources: an op
    list from the operator registry, sizes from the memory hierarchy, mechanisms
    computed from those sizes. A kernel copied from an existing solution breaks
    that claim no matter how well it performs, and cannot be used."""

#: The layout and aliasing licence. RUNG0_CONTRACT states it; `CONTRACT` (rung 2's)
#: does not, so rung 2 has to carry it separately or it would WITHHOLD a fact rung 1
#: gives -- and the rung-1-to-rung-2 delta is only interpretable if rung 2 gives
#: strictly more. Measured: it was missing from the first 37 rung-2 prompts generated,
#: which were discarded.
BUFFER_GUARANTEE = ("Every buffer is dense and row-major in the shapes given "
                    "above, and no two of them overlap.")

CONTRACT = """\
Rules:
  - Entry point must be exactly:  extern "C" {signature}
    The "extern \\"C\\"" is required; without it the symbol mangles and the
    harness fails to link.
  - It is compiled as C++17 and run on the simulator against golden vectors.
    Producing the wrong numbers fails, however fast it is.
  - Do not change the signature, the parameter order, or the memory layout.
    Parameter order follows the traced graph's placeholders.
  - Include what you use. The vendor headers are available and are the only ones
    that resolve: <hexagon_types.h> (which also carries the user-DMA descriptor
    types), <hexagon_protos.h>, <hvx_hexagon_protos.h>, <hmx_hexagon_protos.h>,
    plus the C++ standard library. No repo-local header is on the include path.
  - The harness enables the HMX context before calling you, so HMX code needs no
    SSR setup (doing it anyway is harmless -- it is idempotent).
  - If you use the DMA engine, declare every descriptor
    `static hexagon_udma_descriptor_typeN_t d[...] __attribute__((aligned(64)));`.
    A descriptor is read by an EXTERNAL engine and does not tolerate the alignment
    a stack frame happens to give a 16- or 32-byte struct. Measured: the same
    kernel with function-local descriptors failed 2,979 of 3,211,264 elements and
    with static aligned ones passed 0 of 3,211,264, pipeline untouched. A second
    kernel crashed 0x28 the same way.
  - Also: a second `dmstart` may only be outstanding after the program's FIRST
    transfer has completed. Prime the engine with one `dmstart`/`dmwait` in a
    prologue and then overlap freely, or chain descriptors through `next` and issue
    ONE `dmstart` per tile. Two `dmstart`s as the engine's first act fault 0x28.
  - Return only the C++ translation unit, no prose and no markdown fence."""


#: MEASURED behaviour of the instructions this corpus keeps getting wrong.
#:
#: NOT PART OF THE FIRST PROMPT -- see `retry_facts` below for where it goes and
#: why. WHY IT EXISTS AT ALL. Five of the numeric failures in batches 1-25
#: were a wrong belief about a named intrinsic, not a wrong algorithm -- three lane
#: orders, one rounding path, and one claim that a selectable instruction was
#: unselectable. Each author re-derived these from scratch and three got them
#: wrong, while the answers were already measured and sitting in this repository.
#: A pipeline that withholds a measured hardware fact is not testing the author's
#: reasoning, it is testing whether they happen to guess the same way.
#:
#: THE LINE THIS DOES NOT CROSS. Everything here is a property of the HARDWARE:
#: which instruction the compiler can select, which order a permute leaves lanes
#: in, what the rounding path through qf32 does. Nothing here is about the task --
#: no schedule, no tiling, no algorithm, no hint about which mechanism to use. Any
#: of that would be handing over the answer, and the prompt already carries the
#: reference and the schedule for exactly the part the author IS meant to solve.
ISA_FACTS = """\
Measured behaviour of this core, so you do not have to re-derive it. All of the
below was established by disassembly or by running on the simulator; the general
documentation does not state most of it.

FLOAT ARITHMETIC GOES THROUGH qf32/qf16.
  - `V6_vadd_sf`, `V6_vsub_sf` and `V6_vmpy_sf_sf` are DECLARED in the header and
    CANNOT BE SELECTED on v75 -- eight such float intrinsics exist, and
    `-fsyntax-only` accepts every one of them. Compile with -O2 before believing a
    float intrinsic works.
  - So an IEEE-out add is "do it in qf32, then convert back":
        Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
  - MIN, MAX and COMPARE are the exception and take IEEE operands directly.
    `Q6_Vsf_vmax_VsfVsf`, `Q6_Vsf_vmin_VsfVsf` and `Q6_Vhf_vmax_VhfVhf` ARE
    selectable -- verified in the disassembly as `v2.sf = vmax(v0.sf,v1.sf)`.
  - MAGIC-CONSTANT ROUNDING DOES NOT WORK through qf32: adding 2^23 to force a
    rounding does not round, because qf32 lacks fp32's ULP at that magnitude
    (probed: 0.4 -> 1, 1.5 -> 1, 3.5 -> 3, 99.5 -> 99). Use the truncating convert
    `Q6_Vw_equals_Vsf` and repair ties explicitly.
  - `Q6_Vw_equals_Vsf` TRUNCATES toward zero (12.8 -> 12, -29.7 -> -29). It is the
    only rounding mode the hardware gives directly.

LANE ORDER. Three of this corpus's wrong kernels were a wrong assumption here, one
of them 97% wrong, so these are stated rather than left to inference:
  - Every WIDENING (`Q6_Ww_vsxt_Vh`, `Q6_Wh_vsxt_Vb`, `Q6_Wqf32_vmpy_VhfVhf`)
    delivers its pair in DEAL order -- even-indexed elements in the low vector.
  - Every NARROWING PACK (`Q6_Vh_vpack_VwVw_sat`, `Q6_Vub_vpack_VhVh_sat`)
    CONCATENATES, with the SECOND operand in the low half, and saturates
    (400 -> 255, -50 -> 0).
  - THE INVERSE OF A DEAL IS A SHUFFLE, NOT A CONCATENATION. A pack does not undo
    a widening deal. `Q6_Vh_vshuffe_VhVh` INTERLEAVES the even halfwords; the
    stride-2 gather is `Q6_V_lo_W(Q6_W_vdeal_VVR(hi, lo, -2))`.
  - `Q6_V_vlalign_VVR(v, fill, 4*n)` is a PREFIX shift: lane i receives v[i-n].
    `valign` shifts the other way.

REQUANT. `Q6_Vw_vmpyo_VwVh_s1_rnd_sat` IS the multiply-shift-round: it equals
`((int64)acc * mult + (1 << 14)) >> 15`, saturated (verified on 16 cases). The
multiplier goes in the ODD halfword.

WHAT PAYS. DMA and l2fetch begin to pay above roughly THREE vector operations per
element, not above a particular size -- below that the staging costs more than the
latency it hides. Two DMA transfers cannot overlap until one has completed, so the
engine must be primed. `dmwait` is engine-wide: one wait retires every outstanding
transfer, so one wait can cover a whole chained descriptor list."""


LINALG_INTRO = """\
The same computation as MLIR Linalg, emitted by torch-mlir from this graph. This \
is the compiler's own structured form: `iterator_types` marks each loop parallel \
or reduction, and the `#map` aliases are the affine indexing maps -- one result \
per operand axis, so an axis pinned to a constant (`(d0, 0)`) is a broadcast that \
does not advance. Named ops such as `linalg.matmul` carry their iterator types in \
the op definition rather than spelling them out."""


def build(name, signature, scalar_source, graph, plan, tgt=None,
          linalg_ir=None) -> str:
    """The full acceleration prompt for one kernel.

    Pure: no I/O, no clock, no RNG. Same inputs give byte-identical output --
    `linalg_ir` is passed IN rather than produced here so that stays true and
    the module keeps no dependency on torch-mlir.

    `linalg_ir` is the compiler's own Linalg (see `forge2.linalg`). It is
    optional: the derived schedule below carries the same iterator types and
    maps and additionally answers `vectorizable_loop`, which Linalg does not, so
    a prompt is complete without it. When both are present they are shown
    together -- the derived form is the Hexagon-facing summary, the Linalg is
    ground truth, and they have been diffed (7 exact / 3 named-op / 0
    disagreements across both batches).
    """
    parts = [
        f"Accelerate the kernel `{name}` for the Hexagon NSP.",
        hardware_facts(tgt),
        "The reference below is correct and deliberately slow. It is the "
        "specification: your kernel must compute exactly the same values.",
        f"```cpp\n{scalar_source.rstrip()}\n```",
    ]
    if linalg_ir:
        parts += [LINALG_INTRO, f"```mlir\n{linalg_ir.rstrip()}\n```"]
    parts += [
        "Loop schedule of each primitive. `parallel` iterations are independent; "
        "`reduction` iterations accumulate and must not be reordered across the "
        "accumulator. The indexing maps give each operand's access pattern -- an "
        "operand whose map pins an axis to a constant does not advance along it "
        "(a broadcast, stride 0). `vectorizable_loop` is the axis along which "
        "every operand is unit- or zero-stride, so a contiguous vector load and "
        "store are legal:",
        _schedule_section(graph),
        _budget_section(plan),
        PROVENANCE_RULE,
        CONTRACT.format(signature=signature),
    ]
    return "\n\n".join(parts) + "\n"


#: RUNG 0 of PLAN.md section 2: "bare prompt, 1 shot -- where is the frontier,
#: unaided?"
#:
#: WHAT "BARE" HAS TO MEAN, and the line this draws. Rung 0 is the CONTROL. It is
#: what defeats the objection "you showed it scalar code, so it wrote scalar code",
#: so anything that nudges the model toward an accelerator has to be absent -- and
#: `build()` above is a catalogue of exactly those nudges: the scalar reference, the
#: Linalg IR, the loop schedule with its `vectorizable_loop`, the mechanism budget,
#: the cache and VTCM sizes, and (on a retry) named intrinsics.
#:
#: But bare is not the same as UNANSWERABLE, and three things had to stay:
#:
#:   * THE SIGNATURE. `candidate_kernel`'s declaration carries no sizes, so without
#:     it a kernel cannot link and without shapes it cannot know a single loop
#:     bound. Withholding either would make every attempt fail for a reason that has
#:     nothing to do with whether the model reaches for the hardware, which is the
#:     one thing rung 0 exists to measure.
#:   * WHAT TO COMPUTE. The operator and its schema, plus the values this pipeline
#:     fixed for the non-tensor arguments -- the golden was computed with those, so
#:     a prompt that omits them is asking for a different function than the one it
#:     grades.
#:   * THE TARGET'S NAME. "Hexagon NSP" is the question; the cache sizes are help.
#:
#: THE VENDOR HEADERS ARE DELIBERATELY NOT ENUMERATED, unlike in `CONTRACT`. Listing
#: `<hvx_hexagon_protos.h>` and `<hmx_hexagon_protos.h>` would name the two
#: mechanisms the benchmark is about and hand over the answer in the shape of a
#: build instruction. So rung 0 says the SDK's headers are on the path and leaves
#: the model to know which one it wants. A wrong include is then a real rung-0
#: failure and is recorded as one -- rungs 1-3 are where that stops being the
#: measurement.
RUNG0_CONTRACT = """\
Rules:
  - Entry point must be exactly:  extern "C" {signature}
    The "extern \\"C\\"" is required; without it the symbol mangles and the
    harness fails to link.
  - Do not change the signature, the parameter order, or the memory layout.
  - Every buffer is dense and row-major in the shapes given above, and no two
    of them overlap.
  - It is compiled as C++17 and run on a simulator against golden vectors taken
    from PyTorch. Producing the wrong numbers fails, however fast it is.
  - The Hexagon SDK's own headers are on the include path, as is the C++
    standard library. No header from the benchmark repository resolves; include
    what you use.
  - Return only the C++ translation unit, no prose and no markdown fence."""


#: The integrity rule, restated without the leaks `PROVENANCE_RULE` can afford.
#:
#: That one names `hmx_helpers.h`, the VTCM base and a crouton offset as examples of
#: what not to copy, and tells the author to derive from "the reference, the
#: schedule" -- three mechanism names and two artifacts that do not exist at this
#: rung. The CONSTRAINT is identical; only the examples are dropped.
RUNG0_PROVENANCE = """\
Provenance (this governs whether the result is usable at all):
  - Write this kernel yourself. Do not copy it from, or pattern it on, any
    existing solution to this or a similar kernel, including any private
    reference corpus.
  - Nothing from the benchmark repository may appear in the translation unit.
    Its directories are not on the compile's include path, and a copied body is
    rejected before compiling."""


def _buffer_table(graph) -> str:
    """Each parameter's direction, element type and shape, in signature order.

    The signature gives pointer types and names and NOTHING about extent, so this
    is what makes the task answerable at all (see `RUNG0_CONTRACT`). Rendered from
    the traced graph rather than from the selection's `shape` field because a
    per-position shape can differ from the task's nominal one -- a convolution
    weight is (Cout, Cin, K, K) and is not the input's shape.
    """
    from hexkernels.forge.frontend.emit import cname, output_shape_dtype

    rows = []
    for node in graph.inputs:
        rows.append((cname(node.name), "in", node.dtype, tuple(node.shape)))
    for k, out_name in enumerate(graph.outputs):
        shape, dtype = output_shape_dtype(graph, out_name)
        rows.append((f"out{k}", "out", dtype, tuple(shape)))
    width = max(len(r[0]) for r in rows)
    lines = ["Buffers, in the order the signature takes them:"]
    for pname, direction, dtype, shape in rows:
        n = 1
        for d in shape:
            n *= d
        lines.append(f"  {pname:<{width}}  {direction:<3}  {dtype:<8} "
                     f"shape {shape}, {n} elements")
    return "\n".join(lines)


def task_statement(entry) -> str:
    """WHAT to compute, from the selection entry's own operator identity.

    Built from `op`/`overload`/`schema`/`stages` rather than from the traced
    graph's primitives, and the distinction is the whole point: the primitive
    decomposition is a piece of the ANSWER (it is what `build()`'s schedule section
    hands over), while the operator's identity is the QUESTION.

    A synth row's fixed non-tensor arguments are rendered explicitly. `MinedCall`
    baked them in before tracing, so the golden depends on them; a prompt that
    named the op without them would grade against a function it never specified.
    """
    stages = entry.get("stages")
    if stages:
        lines = ["Compute this composition of PyTorch operators, in order:"]
        prev = "the input buffer(s)"
        for i, (op, ov) in enumerate(stages):
            call = f"aten::{op}" + (f".{ov}" if ov else "")
            lines.append(f"  {i + 1}. t{i} = {call}({prev})")
            prev = f"t{i}"
        lines.append(f"  and store {prev} to the output buffer.")
        lines.append("")
        lines.append("This composition is not itself a PyTorch operator; each "
                     "step is.")
        if entry.get("schema"):
            lines += ["", f"Schema of step 1: {entry['schema']}"]
        return "\n".join(lines)

    call = f"aten::{entry['op']}" + (f".{entry['overload']}"
                                     if entry.get("overload") else "")
    lines = [f"Compute the PyTorch operator `{call}`, whose schema is:",
             f"  {entry['schema']}"]
    plan = entry.get("synth_plan")
    if plan:
        args, t = [], 0
        for kind, value in plan:
            if kind == "t":
                args.append(f"<input {t}>")
                t += 1
            else:
                args.append(repr(value))
        lines += ["",
                  "It is called with these arguments, and the expected output "
                  "depends on them:",
                  f"  {call}({', '.join(args)})",
                  "where <input k> is the k-th input buffer below."]
    else:
        lines += ["",
                  "The input buffers below fill its tensor arguments in order; "
                  "every other argument keeps its schema default."]
    return "\n".join(lines)


def build_rung0(name, signature, graph, entry, tgt=None) -> str:
    """The bare, one-shot ask for one task. Rung 0.

    Pure, like `build()`: same inputs give byte-identical output. `entry` is the
    task's `benchmark/selection.json` row, which is where the operator identity
    lives; `graph` supplies only parameter shapes and dtypes, never its primitive
    decomposition.

    What this deliberately does NOT contain is asserted by
    `hexkernels/forge/tests/test_rung0.py`, which is the guard that keeps rung 0 a control
    rather than a second measurement of rung 2.
    """
    t = tgt or _target.current()
    parts = [
        f"Write a kernel named `{name}` for the Qualcomm Hexagon NSP "
        f"(architecture {t.arch}), compiled with hexagon-clang++ -std=c++17.",
        task_statement(entry),
        _buffer_table(graph),
        RUNG0_PROVENANCE,
        RUNG0_CONTRACT.format(signature=signature),
    ]
    return "\n\n".join(parts) + "\n"


#: Rung 1's integrity rule. Rung 0's cannot be reused, and the reason is not cosmetic.
#:
#: `RUNG0_PROVENANCE` says "Write this kernel yourself. Do not copy it from, or
#: pattern it on, any existing solution" -- correct at rung 0, where no reference is
#: given. Carried into rung 1 verbatim it would forbid using the very reference this
#: rung exists to hand over, and the model would have to guess which instruction wins.
#:
#: So the permission is explicit, exactly as `PROVENANCE_RULE` grants it at rung 2 --
#: minus that rule's examples, which name `hmx_helpers.h`, the VTCM base, a crouton
#: offset and the schedule. The CONSTRAINT is identical; only the leaks are dropped.
RUNG1_PROVENANCE = """\
Provenance (this governs whether the result is usable at all):
  - Derive this kernel from the reference above. Any header the rules below
    permit is fair game; nothing else is.
  - Nothing from the benchmark repository may appear in the translation unit.
    Its directories are not on the compile's include path, and a copied body is
    rejected before compiling.
  - Do not copy from, or pattern it on, any existing solution to this or a
    similar kernel, including any private reference corpus. The reference above
    is not such a solution -- it is the specification, and it is deliberately
    slow."""


def build_rung1(name, signature, graph, entry, scalar_source, tgt=None) -> str:
    """Rung 0's ask plus the correct-but-slow reference. Rung 1.

    Pure, like `build_rung0`: same inputs give byte-identical output.

    WHAT MAKES THIS RUNG 1 AND NOT RUNG 2. PLAN.md section 2 gives rung 1 "PyTorch
    reference + scalar code" and asks whether help with CORRECTNESS produces
    MECHANISM. So the reference body is added and nothing else is: no hardware facts,
    no working-set or tier, no mechanism budget, no schedule, no Linalg, no primitive
    decomposition, no header that names an accelerator. `build()` supplies all of
    those and is rung 2; the gap between the two is what PLAN.md section 6 calls the
    spine of the paper, and `hexkernels/forge/tests/test_rung1.py` is the guard on it.

    The operator identity and schema are already rung 0's (`task_statement`), which is
    the "PyTorch reference" half: the aten call, its schema, and its fixed non-tensor
    arguments. No synthesized `torch` snippet is emitted -- one would have to be
    written by mapping aten names to Python by hand, and an unverified snippet that
    misstated the semantics would corrupt the measurement in the direction of looking
    like a model failure. The emitted C++ below IS a verified statement of those
    semantics: it reproduces PyTorch's own golden vectors bit-for-bit, which is what
    stage (g) proves.
    """
    t = tgt or _target.current()
    parts = [
        f"Write a kernel named `{name}` for the Qualcomm Hexagon NSP "
        f"(architecture {t.arch}), compiled with hexagon-clang++ -std=c++17.",
        task_statement(entry),
        _buffer_table(graph),
        # Same sentence `build()` uses, because it is the rung-appropriate framing at
        # both rungs: the reference is the SPEC, and its slowness is not the point.
        "The reference below is correct and deliberately slow. It is the "
        "specification: your kernel must compute exactly the same values.",
        f"```cpp\n{scalar_source.rstrip()}\n```",
        RUNG1_PROVENANCE,
        RUNG0_CONTRACT.format(signature=signature),
    ]
    return "\n\n".join(parts) + "\n"


def build_rung2(name, signature, graph, entry, scalar_source, plan, tgt=None,
                linalg_ir=None) -> str:
    """The full Forge ask for one task. Rung 2.

    Pure, like every other builder here: same inputs give byte-identical output.

    WHY THIS IS NOT `build()` ITSELF. `build()` is the BATCH Forge prompt and it
    opens with the reference and the schedule, because a corpus author already knows
    which operator they were handed. Used verbatim as rung 2, it would withhold the
    operator identity, its schema, its fixed non-tensor arguments and the buffer
    table -- all of which rungs 0 AND 1 hand over -- and the ladder would stop being
    monotone at the one rung PLAN.md section 6 calls the spine. A reviewer would be
    right to say rung 2 had been given LESS than rung 1 in one respect and more in
    another, which makes the delta uninterpretable.

    So this is rung 1's opening plus everything `build()` adds. The consequence to
    disclose rather than hide: rung 2's prompt is a strict SUPERSET of the batch
    Forge prompt. The added text is redundant given the reference (the reference
    computes the operator; the buffer table restates extents the reference's own loop
    bounds imply), so it cannot manufacture a mechanism delta -- it removes a
    confound instead of creating one.

    WHAT IS STILL WITHHELD, and it is the last thing: intrinsic names. `retry_facts`
    supplies those on a RETRY, and `test_rung2.py::test_the_first_prompt_names_no_
    intrinsic` is the guard -- written for rung 2 because the guard `retry_facts`
    cites was never ported into this repository.
    """
    t = tgt or _target.current()
    parts = [
        f"Write a kernel named `{name}` for the Qualcomm Hexagon NSP "
        f"(architecture {t.arch}), compiled with hexagon-clang++ -std=c++17.",
        task_statement(entry),
        _buffer_table(graph),
        BUFFER_GUARANTEE,
        hardware_facts(t),
        "The reference below is correct and deliberately slow. It is the "
        "specification: your kernel must compute exactly the same values.",
        f"```cpp\n{scalar_source.rstrip()}\n```",
    ]
    if linalg_ir:
        parts += [LINALG_INTRO, f"```mlir\n{linalg_ir.rstrip()}\n```"]
    parts += [
        "Loop schedule of each primitive. `parallel` iterations are independent; "
        "`reduction` iterations accumulate and must not be reordered across the "
        "accumulator. The indexing maps give each operand's access pattern -- an "
        "operand whose map pins an axis to a constant does not advance along it "
        "(a broadcast, stride 0). `vectorizable_loop` is the axis along which "
        "every operand is unit- or zero-stride, so a contiguous vector load and "
        "store are legal:",
        _schedule_section(graph),
        _budget_section(plan),
        PROVENANCE_RULE,
        CONTRACT.format(signature=signature),
    ]
    return "\n\n".join(parts) + "\n"


def retry_facts(verdict=None) -> str:
    """The measured-ISA note, for a RETRY after a failure -- never the first ask.

    THE CONFLICT THIS RESOLVES, stated plainly because it is a real one. Five of
    the numeric failures in batches 1-25 were a wrong belief about a named
    intrinsic rather than a wrong algorithm, and the answers were already measured
    and sitting in this repository -- so withholding them costs rounds for no
    measurement. But `ISA_FACTS` names roughly fifteen intrinsics, and this
    module's own policy is that naming one "would make this measure retrieval
    rather than the thing the benchmark is built to measure", guarded by
    `test_prompt_does_not_hand_over_the_answer`. Adding it to `build()` fails that
    test, correctly.

    Both are right, and they are about DIFFERENT MOMENTS:

      * The FIRST attempt is the measurement. Whether the author reaches for a
        vector intrinsic at all, and which one, is the thing being measured, so it
        gets the reference, the schedule and the budget -- and no intrinsic names.
      * A RETRY is no longer a measurement of that. The first answer already
        recorded what the author knew unaided; the remaining question is whether a
        specific undocumented hardware behaviour can be communicated well enough to
        fix the kernel. `Q6_Vsf_vmax_VsfVsf` being selectable while `V6_vadd_sf` is
        not cannot be derived from anything -- not from the reference, not from the
        header, not from `-fsyntax-only`, which accepts both.

    So pass@1 stays clean and pass@k with feedback gets the facts. A run that
    reports the two together without saying which is which would be the actual
    problem, and `model_client` labels the rounds.
    """
    return ISA_FACTS
