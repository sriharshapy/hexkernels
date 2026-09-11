Accelerate the kernel `fp32__safe_softmax` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v_amax[1024];
static float v_sub[1310720];
static float v_exp[1310720];
static float v_sum_1[1024];
static float v_div[1310720];
static unsigned char v_eq[1310720];
static unsigned char v_logical_not[1310720];
static unsigned char v_any_1[1024];
static unsigned char v_logical_not_1[1024];
static float v_full_like[1310720];
static float v_where[1310720];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i1 = 0; i1 < 1024; i1++) {
    {
      float acc = -INFINITY;
      for (int r0 = 0; r0 < 1280; r0++) {
        acc = (v_args_0[r0*1024 + i1*1] > acc ? v_args_0[r0*1024 + i1*1] : acc);
      }
      v_amax[i1*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_sub[i0*1024 + i1*1] = (v_args_0[i0*1024 + i1*1] - v_amax[i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_exp[i0*1024 + i1*1] = expf(v_sub[i0*1024 + i1*1]);
    }
  }
  for (int i1 = 0; i1 < 1024; i1++) {
    {
      float acc = 0;
      for (int r0 = 0; r0 < 1280; r0++) {
        acc = (acc + v_exp[r0*1024 + i1*1]);
      }
      v_sum_1[i1*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_div[i0*1024 + i1*1] = (v_exp[i0*1024 + i1*1] / v_sum_1[i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_eq[i0*1024 + i1*1] = (v_args_0[i0*1024 + i1*1] == (-INFINITY));
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_logical_not[i0*1024 + i1*1] = (!v_eq[i0*1024 + i1*1]);
    }
  }
  for (int i1 = 0; i1 < 1024; i1++) {
    {
      int32_t acc = 0;
      for (int r0 = 0; r0 < 1280; r0++) {
        acc = (acc || (v_logical_not[r0*1024 + i1*1] != 0));
      }
      v_any_1[i1*1] = (unsigned char)(acc);
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_logical_not_1[i1*1] = (!v_any_1[i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_full_like[i0*1024 + i1*1] = (float)(0);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_where[i0*1024 + i1*1] = (v_logical_not_1[i1*1] ? v_full_like[i0*1024 + i1*1] : v_div[i0*1024 + i1*1]);
    }
  }
  for (int i = 0; i < 1310720; i++) { out0[i] = v_where[i]; }
}
```

The same computation as MLIR Linalg, emitted by torch-mlir from this graph. This is the compiler's own structured form: `iterator_types` marks each loop parallel or reduction, and the `#map` aliases are the affine indexing maps -- one result per operand axis, so an axis pinned to a constant (`(d0, 0)`) is a broadcast that does not advance. Named ops such as `linalg.matmul` carry their iterator types in the op definition rather than spelling them out.

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (0, d1)>
#map3 = affine_map<(d0, d1) -> ()>
func.func @main(%arg0: tensor<1280x1024xf32>) -> tensor<1280x1024xf32> {
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1024xi64>) -> tensor<1024xi64>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1024xf32>) -> tensor<1024xf32>
  %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<1280x1024xf32>) outs(%3, %1 : tensor<1024xf32>, tensor<1024xi64>) {
    ^bb0(%in: f32, %out: f32, %out_3: i64):
    %18 = linalg.index 0 : index
    %19 = arith.index_cast %18 : index to i64
    %20 = arith.maximumf %in, %out : f32
    %21 = arith.cmpf ogt, %in, %out : f32
    %22 = arith.select %21, %19, %out_3 : i64
    linalg.yield %20, %22 : f32, i64
    } -> (tensor<1024xf32>, tensor<1024xi64>)
  %6 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %expanded : tensor<1280x1024xf32>, tensor<1x1024xf32>) outs(%5 : tensor<1280x1024xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
    %18 = arith.subf %in, %in_3 : f32
    linalg.yield %18 : f32
    } -> tensor<1280x1024xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<1280x1024xf32>) outs(%5 : tensor<1280x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
    %18 = math.exp %in : f32
    linalg.yield %18 : f32
    } -> tensor<1280x1024xf32>
  %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
  %10 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["reduction", "parallel"]} ins(%7 : tensor<1280x1024xf32>) outs(%9 : tensor<1x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
    %18 = arith.addf %in, %out : f32
    linalg.yield %18 : f32
    } -> tensor<1x1024xf32>
  %11 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %10 : tensor<1280x1024xf32>, tensor<1x1024xf32>) outs(%5 : tensor<1280x1024xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
    %18 = arith.divf %in, %in_3 : f32
    linalg.yield %18 : f32
    } -> tensor<1280x1024xf32>
  %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1280x1024xf32>) outs(%12 : tensor<1280x1024xi1>) {
    ^bb0(%in: f32, %out: i1):
    %18 = arith.extf %in : f32 to f64
    %19 = arith.cmpf oeq, %18, %cst_2 : f64
    linalg.yield %19 : i1
    } -> tensor<1280x1024xi1>
  %15 = linalg.fill ins(%true : i1) outs(%14 : tensor<1x1024xi1>) -> tensor<1x1024xi1>
  %16 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["reduction", "parallel"]} ins(%13 : tensor<1280x1024xi1>) outs(%15 : tensor<1x1024xi1>) {
    ^bb0(%in: i1, %out: i1):
    %18 = arith.andi %in, %out : i1
    linalg.yield %18 : i1
    } -> tensor<1x1024xi1>
  %17 = linalg.generic {indexing_maps = [#map2, #map3, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%16, %cst_1, %11 : tensor<1x1024xi1>, tensor<f32>, tensor<1280x1024xf32>) outs(%5 : tensor<1280x1024xf32>) {
    ^bb0(%in: i1, %in_3: f32, %in_4: f32, %out: f32):
    %18 = arith.select %in, %in_3, %in_4 : f32
    linalg.yield %18 : f32
    } -> tensor<1280x1024xf32>
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten.amax.default
  iterator_types = ['reduction', 'parallel']
  loops:
    d0: 1280 (reduction)
    d1: 1024 (parallel)
  indexing_maps:
    args_0 (1280, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    amax (1, 1024) result: affine_map<(d0, d1) -> (0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1280 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    args_0 (1280, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    amax (1, 1024) operand: affine_map<(d0, d1) -> (0, d1)>
    sub (1280, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.exp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1280 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    sub (1280, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    exp (1280, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sum.dim_IntList
  iterator_types = ['reduction', 'parallel']
  loops:
    d0: 1280 (reduction)
    d1: 1024 (parallel)
  indexing_maps:
    exp (1280, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    sum_1 (1, 1024) result: affine_map<(d0, d1) -> (0, d1)>
  vectorizable_loop: d1
aten.div.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1280 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    exp (1280, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    sum_1 (1, 1024) operand: affine_map<(d0, d1) -> (0, d1)>
    div (1280, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.eq.Scalar
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1280 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    args_0 (1280, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    eq (1280, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.logical_not.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1280 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    eq (1280, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    logical_not (1280, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.any.dim
  iterator_types = ['reduction', 'parallel']
  loops:
    d0: 1280 (reduction)
    d1: 1024 (parallel)
  indexing_maps:
    logical_not (1280, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    any_1 (1, 1024) result: affine_map<(d0, d1) -> (0, d1)>
  vectorizable_loop: d1
aten.logical_not.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    any_1 (1, 1024) operand: affine_map<(d0, d1) -> (0, d1)>
    logical_not_1 (1, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.full_like.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1280 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    full_like (1280, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.where.self
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1280 (parallel)
    d1: 1024 (parallel)
  indexing_maps:
    logical_not_1 (1, 1024) operand: affine_map<(d0, d1) -> (0, d1)>
    full_like (1280, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    div (1280, 1024) operand: affine_map<(d0, d1) -> (d0, d1)>
    where (1280, 1024) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1

Working set: 10485760 bytes -> tier T3.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop
  l2fetch: working set 10485760 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide
  dma: working set 10485760 B exceeds L2 1048576 B, so the data does not stay resident and staging is what buys bandwidth
  vtcm: the staged tiles need a software-managed scratchpad
        working set 10485760 B also exceeds VTCM 8388608 B, so it must be streamed in tiles -- double-buffered

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
