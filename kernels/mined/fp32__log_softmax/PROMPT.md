Accelerate the kernel `fp32__log_softmax` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v_amax[512];
static float v_sub[196608];
static float v_exp[196608];
static float v_sum_1[512];
static float v_log[512];
static float v_sub_1[196608];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i1 = 0; i1 < 512; i1++) {
    {
      float acc = -INFINITY;
      for (int r0 = 0; r0 < 384; r0++) {
        acc = (v_args_0[r0*512 + i1*1] > acc ? v_args_0[r0*512 + i1*1] : acc);
      }
      v_amax[i1*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_sub[i0*512 + i1*1] = (v_args_0[i0*512 + i1*1] - v_amax[i1*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_exp[i0*512 + i1*1] = expf(v_sub[i0*512 + i1*1]);
    }
  }
  for (int i1 = 0; i1 < 512; i1++) {
    {
      float acc = 0;
      for (int r0 = 0; r0 < 384; r0++) {
        acc = (acc + v_exp[r0*512 + i1*1]);
      }
      v_sum_1[i1*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_log[i1*1] = logf(v_sum_1[i1*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_sub_1[i0*512 + i1*1] = (v_sub[i0*512 + i1*1] - v_log[i1*1]);
    }
  }
  for (int i = 0; i < 196608; i++) { out0[i] = v_sub_1[i]; }
}
```

The same computation as MLIR Linalg, emitted by torch-mlir from this graph. This is the compiler's own structured form: `iterator_types` marks each loop parallel or reduction, and the `#map` aliases are the affine indexing maps -- one result per operand axis, so an axis pinned to a constant (`(d0, 0)`) is a broadcast that does not advance. Named ops such as `linalg.matmul` carry their iterator types in the op definition rather than spelling them out.

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (0, d1)>
func.func @main(%arg0: tensor<384x512xf32>) -> tensor<384x512xf32> {
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<512xi64>) -> tensor<512xi64>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<512xf32>) -> tensor<512xf32>
  %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<384x512xf32>) outs(%3, %1 : tensor<512xf32>, tensor<512xi64>) {
    ^bb0(%in: f32, %out: f32, %out_1: i64):
    %13 = linalg.index 0 : index
    %14 = arith.index_cast %13 : index to i64
    %15 = arith.maximumf %in, %out : f32
    %16 = arith.cmpf ogt, %in, %out : f32
    %17 = arith.select %16, %14, %out_1 : i64
    linalg.yield %15, %17 : f32, i64
    } -> (tensor<512xf32>, tensor<512xi64>)
  %6 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %expanded : tensor<384x512xf32>, tensor<1x512xf32>) outs(%5 : tensor<384x512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
    %13 = arith.subf %in, %in_1 : f32
    linalg.yield %13 : f32
    } -> tensor<384x512xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<384x512xf32>) outs(%5 : tensor<384x512xf32>) {
    ^bb0(%in: f32, %out: f32):
    %13 = math.exp %in : f32
    linalg.yield %13 : f32
    } -> tensor<384x512xf32>
  %9 = linalg.fill ins(%cst_0 : f32) outs(%8 : tensor<1x512xf32>) -> tensor<1x512xf32>
  %10 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["reduction", "parallel"]} ins(%7 : tensor<384x512xf32>) outs(%9 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
    %13 = arith.addf %in, %out : f32
    linalg.yield %13 : f32
    } -> tensor<1x512xf32>
  %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<1x512xf32>) outs(%8 : tensor<1x512xf32>) {
    ^bb0(%in: f32, %out: f32):
    %13 = math.log %in : f32
    linalg.yield %13 : f32
    } -> tensor<1x512xf32>
  %12 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %11 : tensor<384x512xf32>, tensor<1x512xf32>) outs(%5 : tensor<384x512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
    %13 = arith.subf %in, %in_1 : f32
    linalg.yield %13 : f32
    } -> tensor<384x512xf32>
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten.amax.default
  iterator_types = ['reduction', 'parallel']
  loops:
    d0: 384 (reduction)
    d1: 512 (parallel)
  indexing_maps:
    args_0 (384, 512) operand: affine_map<(d0, d1) -> (d0, d1)>
    amax (1, 512) result: affine_map<(d0, d1) -> (0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 384 (parallel)
    d1: 512 (parallel)
  indexing_maps:
    args_0 (384, 512) operand: affine_map<(d0, d1) -> (d0, d1)>
    amax (1, 512) operand: affine_map<(d0, d1) -> (0, d1)>
    sub (384, 512) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.exp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 384 (parallel)
    d1: 512 (parallel)
  indexing_maps:
    sub (384, 512) operand: affine_map<(d0, d1) -> (d0, d1)>
    exp (384, 512) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sum.dim_IntList
  iterator_types = ['reduction', 'parallel']
  loops:
    d0: 384 (reduction)
    d1: 512 (parallel)
  indexing_maps:
    exp (384, 512) operand: affine_map<(d0, d1) -> (d0, d1)>
    sum_1 (1, 512) result: affine_map<(d0, d1) -> (0, d1)>
  vectorizable_loop: d1
aten.log.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 512 (parallel)
  indexing_maps:
    sum_1 (1, 512) operand: affine_map<(d0, d1) -> (0, d1)>
    log (1, 512) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 384 (parallel)
    d1: 512 (parallel)
  indexing_maps:
    sub (384, 512) operand: affine_map<(d0, d1) -> (d0, d1)>
    log (1, 512) operand: affine_map<(d0, d1) -> (0, d1)>
    sub_1 (384, 512) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1

Working set: 1572864 bytes -> tier T2.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop
  l2fetch: working set 1572864 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide
  dma: working set 1572864 B exceeds L2 1048576 B, so the data does not stay resident and staging is what buys bandwidth
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
