Accelerate the kernel `fp32_avg_pool3d` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v_avg_pool3d[65536];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 16; i2++) {
        for (int i3 = 0; i3 < 16; i3++) {
          for (int i4 = 0; i4 < 16; i4++) {
            float acc = 0;
            for (int k0 = 0; k0 < 2; k0++) {
              for (int k1 = 0; k1 < 2; k1++) {
                for (int k2 = 0; k2 < 2; k2++) {
                  acc += v_args_0[(i0)*524288 + (i1)*32768 + (i2*2 + k0 - 0)*1024 + (i3*2 + k1 - 0)*32 + (i4*2 + k2 - 0)];
                }
              }
            }
            v_avg_pool3d[i1*4096 + i2*256 + i3*16 + i4*1] = acc / (float)8;
          }
        }
      }
    }
  }
  for (int i = 0; i < 65536; i++) { out0[i] = v_avg_pool3d[i]; }
}
```

The same computation as MLIR Linalg, emitted by torch-mlir from this graph. This is the compiler's own structured form: `iterator_types` marks each loop parallel or reduction, and the `#map` aliases are the affine indexing maps -- one result per operand axis, so an axis pinned to a constant (`(d0, 0)`) is a broadcast that does not advance. Named ops such as `linalg.matmul` carry their iterator types in the op definition rather than spelling them out.

```mlir
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
func.func @main(%arg0: tensor<1x16x32x32x32xf32>) -> tensor<1x16x16x16x16xf32> {
  %transposed = linalg.transpose ins(%arg0 : tensor<1x16x32x32x32xf32>) outs(%1 : tensor<1x32x32x32x16xf32>) permutation = [0, 2, 3, 4, 1]
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x16x16x16x16xf32>) -> tensor<1x16x16x16x16xf32>
  %4 = linalg.pooling_ndhwc_sum {dilations = dense<1> : vector<3xi64>, strides = dense<2> : vector<3xi64>} ins(%transposed, %0 : tensor<1x32x32x32x16xf32>, tensor<2x2x2xf32>) outs(%3 : tensor<1x16x16x16x16xf32>) -> tensor<1x16x16x16x16xf32>
  %transposed_1 = linalg.transpose ins(%4 : tensor<1x16x16x16x16xf32>) outs(%2 : tensor<1x16x16x16x16xf32>) permutation = [0, 4, 1, 2, 3]
  %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed_1 : tensor<1x16x16x16x16xf32>) outs(%2 : tensor<1x16x16x16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
    %6 = arith.divf %in, %cst_0 : f32
    linalg.yield %6 : f32
    } -> tensor<1x16x16x16x16xf32>
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten.avg_pool3d.default
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel', 'parallel', 'reduction', 'reduction', 'reduction']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 16 (parallel)
    d3: 16 (parallel)
    d4: 16 (parallel)
    d5: 2 (reduction)
    d6: 2 (reduction)
    d7: 2 (reduction)
  indexing_maps:
    args_0 (1, 16, 32, 32, 32) operand: affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2 * 2 + d5, d3 * 2 + d6, d4 * 2 + d7)>
    avg_pool3d (1, 16, 16, 16, 16) result: affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
  vectorizable_loop: none

Working set: 2359296 bytes -> tier T2.
Mechanisms this size justifies:
  hvx: reduction loops vectorise through an accumulator plus a horizontal reduce
  l2fetch: working set 2359296 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide
  dma: working set 2359296 B exceeds L2 1048576 B, so the data does not stay resident and staging is what buys bandwidth
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
