Accelerate the kernel `fp32_adaptive_avg_pool1d` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v_unsqueeze[32];
static float v__adaptive_avg_pool2d[16];
static float v_squeeze[16];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i = 0; i < 32; i++) v_unsqueeze[i] = v_args_0[i];
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        for (int i3 = 0; i3 < 4; i3++) {
          const int s0 = (i2 * 1) / 1;
          const int e0 = ((i2 + 1) * 1 + 1 - 1) / 1;
          const int s1 = (i3 * 8) / 4;
          const int e1 = ((i3 + 1) * 8 + 4 - 1) / 4;
          float acc = 0;
          int cnt = 0;
          for (int k0 = s0; k0 < e0; k0++) {
            for (int k1 = s1; k1 < e1; k1++) {
              acc += v_unsqueeze[(i0)*32 + (i1)*8 + (k0)*8 + (k1)];
              cnt++;
            }
          }
          v__adaptive_avg_pool2d[i1*4 + i3*1] = acc / (float)cnt;
        }
      }
    }
  }
  for (int i = 0; i < 16; i++) v_squeeze[i] = v__adaptive_avg_pool2d[i];
  for (int i = 0; i < 16; i++) { out0[i] = v_squeeze[i]; }
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten.unsqueeze.default
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 4 (parallel)
    d2: 1 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    args_0 (1, 4, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d2, d3)>
    unsqueeze (1, 4, 1, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten._adaptive_avg_pool2d.default
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel', 'reduction', 'reduction']
  loops:
    d0: 1 (parallel)
    d1: 4 (parallel)
    d2: 1 (parallel)
    d3: 4 (parallel)
    d4: 2 (reduction)
    d5: 3 (reduction)
  indexing_maps:
    unsqueeze (1, 4, 1, 8) operand: affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>
    _adaptive_avg_pool2d (1, 4, 1, 4) result: affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.squeeze.dims
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 4 (parallel)
    d2: 4 (parallel)
  indexing_maps:
    _adaptive_avg_pool2d (1, 4, 1, 4) operand: affine_map<(d0, d1, d2) -> (0, d0, 0, d2)>
    squeeze (1, 4, 4) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2

Working set: 192 bytes -> tier T0.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop

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
