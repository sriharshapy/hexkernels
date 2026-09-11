Accelerate the kernel `fp32_binary_cross_entropy` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v_sub[1048576];
static float v_neg[1048576];
static float v_log1p[1048576];
static float v_full[1];
static float v_maximum[1048576];
static float v_mul[1048576];
static float v_log[1048576];
static float v_full_1[1];
static float v_maximum_1[1048576];
static float v_mul_1[1048576];
static float v_sub_1[1048576];
static float v_mean[1];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_sub[i0*1024 + i1*1] = (v_args_1[i0*1024 + i1*1] - 1);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_neg[i0*1024 + i1*1] = (-v_args_0[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_log1p[i0*1024 + i1*1] = log1pf(v_neg[i0*1024 + i1*1]);
    }
  }
  v_full[0] = (float)(-100);
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_maximum[i0*1024 + i1*1] = (v_log1p[i0*1024 + i1*1] > v_full[0] ? v_log1p[i0*1024 + i1*1] : v_full[0]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_mul[i0*1024 + i1*1] = (v_sub[i0*1024 + i1*1] * v_maximum[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_log[i0*1024 + i1*1] = logf(v_args_0[i0*1024 + i1*1]);
    }
  }
  v_full_1[0] = (float)(-100);
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_maximum_1[i0*1024 + i1*1] = (v_log[i0*1024 + i1*1] > v_full_1[0] ? v_log[i0*1024 + i1*1] : v_full_1[0]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_mul_1[i0*1024 + i1*1] = (v_args_1[i0*1024 + i1*1] * v_maximum_1[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_sub_1[i0*1024 + i1*1] = (v_mul[i0*1024 + i1*1] - v_mul_1[i0*1024 + i1*1]);
    }
  }
  {
    float acc = 0;
    for (int r0 = 0; r0 < 1024; r0++) {
      for (int r1 = 0; r1 < 1024; r1++) {
        acc = (acc + v_sub_1[r0*1024 + r1*1]);
      }
    }
    v_mean[0] = (acc / (float)1048576);
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_mean[i]; }
}
```

The same computation as MLIR Linalg, emitted by torch-mlir from this graph. This is the compiler's own structured form: `iterator_types` marks each loop parallel or reduction, and the `#map` aliases are the affine indexing maps -- one result per operand axis, so an axis pinned to a constant (`(d0, 0)`) is a broadcast that does not advance. Named ops such as `linalg.matmul` carry their iterator types in the op definition rather than spelling them out.

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<() -> ()>
func.func @main(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>) -> tensor<f32> {
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
    %14 = arith.subf %in, %cst_1 : f32
    linalg.yield %14 : f32
    } -> tensor<1024x1024xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
    %14 = arith.negf %in : f32
    linalg.yield %14 : f32
    } -> tensor<1024x1024xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
    %14 = math.log1p %in : f32
    linalg.yield %14 : f32
    } -> tensor<1024x1024xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %cst_0 : tensor<1024x1024xf32>, tensor<f32>) outs(%0 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
    %14 = arith.cmpf ogt, %in, %in_3 : f32
    %15 = arith.select %14, %in, %in_3 : f32
    linalg.yield %15 : f32
    } -> tensor<1024x1024xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %4 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
    %14 = arith.mulf %in, %in_3 : f32
    linalg.yield %14 : f32
    } -> tensor<1024x1024xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
    %14 = math.log %in : f32
    linalg.yield %14 : f32
    } -> tensor<1024x1024xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %cst_0 : tensor<1024x1024xf32>, tensor<f32>) outs(%0 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
    %14 = arith.cmpf ogt, %in, %in_3 : f32
    %15 = arith.select %14, %in, %in_3 : f32
    linalg.yield %15 : f32
    } -> tensor<1024x1024xf32>
  %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %7 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
    %14 = arith.mulf %in, %in_3 : f32
    linalg.yield %14 : f32
    } -> tensor<1024x1024xf32>
  %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %8 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%0 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
    %14 = arith.subf %in, %in_3 : f32
    linalg.yield %14 : f32
    } -> tensor<1024x1024xf32>
  %11 = linalg.fill ins(%cst : f32) outs(%10 : tensor<f32>) -> tensor<f32>
  %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction"]} ins(%9 : tensor<1024x1024xf32>) outs(%11 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
    %14 = arith.addf %in, %out : f32
    linalg.yield %14 : f32
    } -> tensor<f32>
  %13 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%12 : tensor<f32>) outs(%10 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
    %14 = arith.divf %in, %cst_2 : f32
    linalg.yield %14 : f32
    } -> tensor<f32>
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1024 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    args_1 (1024, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    sub (1024, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.neg.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1024 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    args_0 (1024, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    neg (1024, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.log1p.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1024 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    neg (1024, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    log1p (1024, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.full.default
  iterator_types = []
  loops:
  indexing_maps:
    full () result: affine_map<() -> ()>
  vectorizable_loop: none
aten.maximum.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1024 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    log1p (1024, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    full () operand: affine_map<(d0, d1) -> ()>
    maximum (1024, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1024 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    sub (1024, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    maximum (1024, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul (1024, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.log.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1024 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    args_0 (1024, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    log (1024, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.full.default
  iterator_types = []
  loops:
  indexing_maps:
    full_1 () result: affine_map<() -> ()>
  vectorizable_loop: none
aten.maximum.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1024 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    log (1024, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    full_1 () operand: affine_map<(d0, d1) -> ()>
    maximum_1 (1024, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1024 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    args_1 (1024, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    maximum_1 (1024, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul_1 (1024, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1024 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    mul (1024, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul_1 (1024, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    sub_1 (1024, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mean.default
  iterator_types = ['reduction', 'reduction']
  loops:
    d0: 1024 (reduction)
    d1: 1024 (reduction)
  indexing_maps:
    sub_1 (1024, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    mean () result: affine_map<(d0, d1) -> ()>
  vectorizable_loop: none

Working set: 8388612 bytes -> tier T3.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop
  l2fetch: working set 8388612 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide
  dma: working set 8388612 B exceeds L2 1048576 B, so the data does not stay resident and staging is what buys bandwidth
  vtcm: the staged tiles need a software-managed scratchpad
        working set 8388612 B also exceeds VTCM 8388608 B, so it must be streamed in tiles -- double-buffered

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
  - Entry point must be exactly:  extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0)
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
