Accelerate the kernel `fp16_diagflat` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static _Float16 v_view[512];
static _Float16 v_unsqueeze[512];
static _Float16 v_permute[512];
static int64_t v_arange[512];
static int64_t v_arange_1[512];
static int64_t v_unsqueeze_1[512];
static unsigned char v_eq[262144];
static unsigned char v_view_1[262144];
static _Float16 v_scalar_tensor[1];
static _Float16 v_where[262144];

extern "C" void candidate_kernel(const _Float16 *v_args_0, _Float16 *out0) {
  for (int i = 0; i < 512; i++) v_view[i] = v_args_0[i];
  for (int i = 0; i < 512; i++) v_unsqueeze[i] = v_view[i];
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_permute[i1*1] = v_unsqueeze[i0*512 + i1];
    }
  }
  for (int i0 = 0; i0 < 512; i0++) {
    v_arange[i0*1] = (int64_t)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 512; i0++) {
    v_arange_1[i0*1] = (int64_t)(0 + i0 * 1);
  }
  for (int i = 0; i < 512; i++) v_unsqueeze_1[i] = v_arange_1[i];
  for (int i0 = 0; i0 < 512; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_eq[i0*512 + i1*1] = (v_arange[i1*1] == v_unsqueeze_1[i0*1]);
    }
  }
  for (int i = 0; i < 262144; i++) v_view_1[i] = v_eq[i];
  v_scalar_tensor[0] = (_Float16)(0);
  for (int i0 = 0; i0 < 512; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_where[i0*512 + i1*1] = (v_view_1[i0*512 + i1*1] ? v_permute[i1*1] : v_scalar_tensor[0]);
    }
  }
  for (int i = 0; i < 262144; i++) { out0[i] = v_where[i]; }
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten.view.default
  iterator_types = ['parallel']
  loops:
    d0: 512 (parallel)
  indexing_maps:
    args_0 (8, 64) operand: affine_map<(d0) -> (0, d0)>
    view (512,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.unsqueeze.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 512 (parallel)
  indexing_maps:
    view (512,) operand: affine_map<(d0, d1) -> (d1)>
    unsqueeze (1, 512) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.permute.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 512 (parallel)
  indexing_maps:
    unsqueeze (1, 512) operand: affine_map<(d0, d1) -> (d0, d1)>
    permute (1, 512) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.arange.start_step
  iterator_types = ['parallel']
  loops:
    d0: 512 (parallel)
  indexing_maps:
    arange (512,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.arange.start_step
  iterator_types = ['parallel']
  loops:
    d0: 512 (parallel)
  indexing_maps:
    arange_1 (512,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.unsqueeze.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 512 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    arange_1 (512,) operand: affine_map<(d0, d1) -> (d1)>
    unsqueeze_1 (512, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.eq.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 512 (parallel)
    d1: 512 (parallel)
  indexing_maps:
    arange (512,) operand: affine_map<(d0, d1) -> (d1)>
    unsqueeze_1 (512, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    eq (512, 512) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.view.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 512 (parallel)
    d1: 512 (parallel)
  indexing_maps:
    eq (512, 512) operand: affine_map<(d0, d1) -> (d0, d1)>
    view_1 (512, 512) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.scalar_tensor.default
  iterator_types = []
  loops:
  indexing_maps:
    scalar_tensor () result: affine_map<() -> ()>
  vectorizable_loop: none
aten.where.self
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 512 (parallel)
    d1: 512 (parallel)
  indexing_maps:
    view_1 (512, 512) operand: affine_map<(d0, d1) -> (d0, d1)>
    permute (1, 512) operand: affine_map<(d0, d1) -> (0, d1)>
    scalar_tensor () operand: affine_map<(d0, d1) -> ()>
    where (512, 512) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1

Working set: 525312 bytes -> tier T1.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop
  l2fetch: working set 525312 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide

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
    that claim no matter how well it performs, and cannot be used.

Rules:
  - Entry point must be exactly:  extern "C" void candidate_kernel(const _Float16 *v_args_0, _Float16 *out0)
    The "extern \"C\"" is required; without it the symbol mangles and the
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
  - Return only the C++ translation unit, no prose and no markdown fence.
