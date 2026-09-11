Accelerate the kernel `fp32_fmax__mish` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v_fmax[98304];
static float v_exp[98304];
static float v_log1p[98304];
static unsigned char v_gt[98304];
static float v_where[98304];
static float v_tanh[98304];
static float v_mul[98304];

extern "C" void candidate_kernel(const float *v_xs_0, const float *v_xs_1, float *out0) {
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_fmax[i0*384 + i1*1] = fmaxf(v_xs_0[i0*384 + i1*1], v_xs_1[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_exp[i0*384 + i1*1] = expf(v_fmax[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_log1p[i0*384 + i1*1] = log1pf(v_exp[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_gt[i0*384 + i1*1] = (v_fmax[i0*384 + i1*1] > 20);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_where[i0*384 + i1*1] = (v_gt[i0*384 + i1*1] ? v_fmax[i0*384 + i1*1] : v_log1p[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_tanh[i0*384 + i1*1] = tanhf(v_where[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_mul[i0*384 + i1*1] = (v_fmax[i0*384 + i1*1] * v_tanh[i0*384 + i1*1]);
    }
  }
  for (int i = 0; i < 98304; i++) { out0[i] = v_mul[i]; }
}
```

The same computation as MLIR Linalg, emitted by torch-mlir from this graph. This is the compiler's own structured form: `iterator_types` marks each loop parallel or reduction, and the `#map` aliases are the affine indexing maps -- one result per operand axis, so an axis pinned to a constant (`(d0, 0)`) is a broadcast that does not advance. Named ops such as `linalg.matmul` carry their iterator types in the op definition rather than spelling them out.

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%arg0: tensor<256x384xf32>, %arg1: tensor<256x384xf32>) -> tensor<256x384xf32> {
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<256x384xf32>, tensor<256x384xf32>) outs(%0 : tensor<256x384xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
    %13 = arith.cmpf ogt, %in, %in_0 : f32
    %14 = arith.select %13, %in, %in_0 : f32
    linalg.yield %14 : f32
    } -> tensor<256x384xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg0 : tensor<256x384xf32>, tensor<256x384xf32>) outs(%2 : tensor<256x384xi1>) {
    ^bb0(%in: f32, %in_0: f32, %out: i1):
    %13 = arith.cmpf une, %in, %in_0 : f32
    linalg.yield %13 : i1
    } -> tensor<256x384xi1>
  %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg1 : tensor<256x384xf32>, tensor<256x384xf32>) outs(%2 : tensor<256x384xi1>) {
    ^bb0(%in: f32, %in_0: f32, %out: i1):
    %13 = arith.cmpf une, %in, %in_0 : f32
    linalg.yield %13 : i1
    } -> tensor<256x384xi1>
  %5 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %arg0, %1 : tensor<256x384xi1>, tensor<256x384xf32>, tensor<256x384xf32>) outs(%0 : tensor<256x384xf32>) {
    ^bb0(%in: i1, %in_0: f32, %in_1: f32, %out: f32):
    %13 = arith.select %in, %in_0, %in_1 : f32
    linalg.yield %13 : f32
    } -> tensor<256x384xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %arg1, %5 : tensor<256x384xi1>, tensor<256x384xf32>, tensor<256x384xf32>) outs(%0 : tensor<256x384xf32>) {
    ^bb0(%in: i1, %in_0: f32, %in_1: f32, %out: f32):
    %13 = arith.select %in, %in_0, %in_1 : f32
    linalg.yield %13 : f32
    } -> tensor<256x384xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<256x384xf32>) outs(%0 : tensor<256x384xf32>) {
    ^bb0(%in: f32, %out: f32):
    %13 = math.exp %in : f32
    linalg.yield %13 : f32
    } -> tensor<256x384xf32>
  %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<256x384xf32>) outs(%0 : tensor<256x384xf32>) {
    ^bb0(%in: f32, %out: f32):
    %13 = math.log1p %in : f32
    linalg.yield %13 : f32
    } -> tensor<256x384xf32>
  %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<256x384xf32>) outs(%2 : tensor<256x384xi1>) {
    ^bb0(%in: f32, %out: i1):
    %13 = arith.extf %in : f32 to f64
    %14 = arith.cmpf ogt, %13, %cst : f64
    linalg.yield %14 : i1
    } -> tensor<256x384xi1>
  %10 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%9, %6, %8 : tensor<256x384xi1>, tensor<256x384xf32>, tensor<256x384xf32>) outs(%0 : tensor<256x384xf32>) {
    ^bb0(%in: i1, %in_0: f32, %in_1: f32, %out: f32):
    %13 = arith.select %in, %in_0, %in_1 : f32
    linalg.yield %13 : f32
    } -> tensor<256x384xf32>
  %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<256x384xf32>) outs(%0 : tensor<256x384xf32>) {
    ^bb0(%in: f32, %out: f32):
    %13 = math.tanh %in : f32
    linalg.yield %13 : f32
    } -> tensor<256x384xf32>
  %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %11 : tensor<256x384xf32>, tensor<256x384xf32>) outs(%0 : tensor<256x384xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
    %13 = arith.mulf %in, %in_0 : f32
    linalg.yield %13 : f32
    } -> tensor<256x384xf32>
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten.fmax.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    xs_0 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    xs_1 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    fmax (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.exp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    fmax (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    exp (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.log1p.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    exp (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    log1p (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.gt.Scalar
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    fmax (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    gt (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.where.self
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    gt (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    fmax (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    log1p (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    where (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.tanh.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    where (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    tanh (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    fmax (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    tanh (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1

Working set: 1179648 bytes -> tier T2.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop
  l2fetch: working set 1179648 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide
  dma: working set 1179648 B exceeds L2 1048576 B, so the data does not stay resident and staging is what buys bandwidth
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
  - Entry point must be exactly:  extern "C" void candidate_kernel(const float *v_xs_0, const float *v_xs_1, float *out0)
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
