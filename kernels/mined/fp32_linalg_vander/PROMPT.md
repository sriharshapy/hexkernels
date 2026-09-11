Accelerate the kernel `fp32_linalg_vander` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v_unsqueeze[6144];
static float v_expand[780288];
static float v_cumprod[780288];
static float v_full[6144];
static float v_cat[786432];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i = 0; i < 6144; i++) v_unsqueeze[i] = v_args_0[i];
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 127; i2++) {
        v_expand[i0*16256 + i1*127 + i2*1] = v_unsqueeze[i0*128 + i1];
      }
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      float acc = 1;
      for (int s2 = 0; s2 < 127; s2++) {
        acc = (acc * v_expand[i0*16256 + i1*127 + s2*1]);
        v_cumprod[i0*16256 + i1*127 + s2*1] = (float)acc;
      }
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_full[i0*128 + i1*1] = (float)(1);
      }
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_cat[(i0)*16384 + (i1)*128 + (i2)] = v_full[i0*128 + i1*1];
      }
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 127; i2++) {
        v_cat[(i0)*16384 + (i1)*128 + (i2 + 1)] = v_cumprod[i0*16256 + i1*127 + i2*1];
      }
    }
  }
  for (int i = 0; i < 786432; i++) { out0[i] = v_cat[i]; }
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten.unsqueeze.default
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 48 (parallel)
    d1: 128 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    args_0 (48, 128) operand: affine_map<(d0, d1, d2) -> (d1, d2)>
    unsqueeze (48, 128, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.expand.default
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 48 (parallel)
    d1: 128 (parallel)
    d2: 127 (parallel)
  indexing_maps:
    unsqueeze (48, 128, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, 0)>
    expand (48, 128, 127) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.cumprod.default
  iterator_types = ['parallel', 'parallel', 'scan']
  loops:
    d0: 48 (parallel)
    d1: 128 (parallel)
    d2: 127 (scan)
  indexing_maps:
    expand (48, 128, 127) operand: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    cumprod (48, 128, 127) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: none
aten.full.default
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 48 (parallel)
    d1: 128 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    full (48, 128, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.cat.default
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 48 (parallel)
    d1: 128 (parallel)
    d2: 128 (parallel)
  indexing_maps:
    full (48, 128, 1) operand: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    cumprod (48, 128, 127) operand: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    cat (48, 128, 128) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2

Working set: 3170304 bytes -> tier T2.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop
  l2fetch: working set 3170304 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide
  dma: working set 3170304 B exceeds L2 1048576 B, so the data does not stay resident and staging is what buys bandwidth
  vtcm: the staged tiles need a software-managed scratchpad

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
  - Entry point must be exactly:  extern "C" void candidate_kernel(const float *v_args_0, float *out0)
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
