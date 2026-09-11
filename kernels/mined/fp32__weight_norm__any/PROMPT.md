Accelerate the kernel `fp32__weight_norm__any` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v_linalg_vector_norm[32];
static float v_div[2048];
static float v_mul[2048];
static unsigned char v_any_1[1];

extern "C" void candidate_kernel(const float *v_xs_0, const float *v_xs_1, unsigned char *out0) {
  for (int i0 = 0; i0 < 32; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc + v_xs_0[i0*64 + r1*1]*v_xs_0[i0*64 + r1*1]);
      }
      v_linalg_vector_norm[i0*1] = sqrtf(acc);
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_div[i0*64 + i1*1] = (v_xs_1[i0*64 + i1*1] / v_linalg_vector_norm[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul[i0*64 + i1*1] = (v_xs_0[i0*64 + i1*1] * v_div[i0*64 + i1*1]);
    }
  }
  {
    int32_t acc = 0;
    for (int r0 = 0; r0 < 32; r0++) {
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc || (v_mul[r0*64 + r1*1] != 0));
      }
    }
    v_any_1[0] = (unsigned char)(acc);
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_any_1[i]; }
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten.linalg_vector_norm.default
  iterator_types = ['parallel', 'reduction']
  loops:
    d0: 32 (parallel)
    d1: 64 (reduction)
  indexing_maps:
    xs_0 (32, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    linalg_vector_norm (32, 1) result: affine_map<(d0, d1) -> (d0, 0)>
  vectorizable_loop: none
aten.div.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 32 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    xs_1 (32, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    linalg_vector_norm (32, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    div (32, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 32 (parallel)
    d1: 64 (parallel)
  indexing_maps:
    xs_0 (32, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    div (32, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul (32, 64) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.any.default
  iterator_types = ['reduction', 'reduction']
  loops:
    d0: 32 (reduction)
    d1: 64 (reduction)
  indexing_maps:
    mul (32, 64) operand: affine_map<(d0, d1) -> (d0, d1)>
    any_1 () result: affine_map<(d0, d1) -> ()>
  vectorizable_loop: none

Working set: 16388 bytes -> tier T1.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop
  l2fetch: working set 16388 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide

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
  - Entry point must be exactly:  extern "C" void candidate_kernel(const float *v_xs_0, const float *v_xs_1, unsigned char *out0)
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
