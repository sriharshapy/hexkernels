Accelerate the kernel `fp32__adaptive_avg_pool3d` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v__adaptive_avg_pool3d[4096];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 8; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          for (int i4 = 0; i4 < 8; i4++) {
            const int s0 = (i2 * 16) / 8;
            const int e0 = ((i2 + 1) * 16 + 8 - 1) / 8;
            const int s1 = (i3 * 16) / 8;
            const int e1 = ((i3 + 1) * 16 + 8 - 1) / 8;
            const int s2 = (i4 * 16) / 8;
            const int e2 = ((i4 + 1) * 16 + 8 - 1) / 8;
            float acc = 0;
            int cnt = 0;
            for (int k0 = s0; k0 < e0; k0++) {
              for (int k1 = s1; k1 < e1; k1++) {
                for (int k2 = s2; k2 < e2; k2++) {
                  acc += v_args_0[(i0)*32768 + (i1)*4096 + (k0)*256 + (k1)*16 + (k2)];
                  cnt++;
                }
              }
            }
            v__adaptive_avg_pool3d[i1*512 + i2*64 + i3*8 + i4*1] = acc / (float)cnt;
          }
        }
      }
    }
  }
  for (int i = 0; i < 4096; i++) { out0[i] = v__adaptive_avg_pool3d[i]; }
}
```

The same computation as MLIR Linalg, emitted by torch-mlir from this graph. This is the compiler's own structured form: `iterator_types` marks each loop parallel or reduction, and the `#map` aliases are the affine indexing maps -- one result per operand axis, so an axis pinned to a constant (`(d0, 0)`) is a broadcast that does not advance. Named ops such as `linalg.matmul` carry their iterator types in the op definition rather than spelling them out.

```mlir
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
func.func @main(%arg0: tensor<1x8x16x16x16xf32>) -> tensor<1x8x8x8x8xf32> {
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x8x8x8x8xf32>) -> tensor<1x8x8x8x8xf32>
  %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0 : tensor<3x3x3xi1>) outs(%3, %1 : tensor<1x8x8x8x8xf32>, tensor<8x8x8xf32>) {
    ^bb0(%in: i1, %out: f32, %out_0: f32):
    %6 = linalg.index 1 : index
    %7 = linalg.index 2 : index
    %8 = linalg.index 3 : index
    %9 = linalg.index 4 : index
    %10 = linalg.index 5 : index
    %11 = linalg.index 6 : index
    %12 = linalg.index 7 : index
    %13 = arith.muli %7, %c16 : index
    %14 = arith.floordivsi %13, %c8 : index
    %15 = arith.addi %7, %c1 : index
    %16 = arith.muli %15, %c16 : index
    %17 = arith.subi %16, %c1 : index
    %18 = arith.floordivsi %17, %c8 : index
    %19 = arith.addi %18, %c1 : index
    %20 = arith.muli %8, %c16 : index
    %21 = arith.floordivsi %20, %c8 : index
    %22 = arith.addi %8, %c1 : index
    %23 = arith.muli %22, %c16 : index
    %24 = arith.subi %23, %c1 : index
    %25 = arith.floordivsi %24, %c8 : index
    %26 = arith.addi %25, %c1 : index
    %27 = arith.muli %9, %c16 : index
    %28 = arith.floordivsi %27, %c8 : index
    %29 = arith.addi %9, %c1 : index
    %30 = arith.muli %29, %c16 : index
    %31 = arith.subi %30, %c1 : index
    %32 = arith.floordivsi %31, %c8 : index
    %33 = arith.addi %32, %c1 : index
    %34 = arith.addi %14, %10 : index
    %35 = arith.addi %21, %11 : index
    %36 = arith.addi %28, %12 : index
    %extracted = tensor.extract %padded[%c0, %6, %34, %35, %36] : tensor<1x8x17x17x17xf32>
    %37 = arith.cmpi ult, %34, %19 : index
    %38 = arith.select %37, %extracted, %cst : f32
    %39 = arith.cmpi ult, %35, %26 : index
    %40 = arith.select %39, %38, %cst : f32
    %41 = arith.cmpi ult, %36, %33 : index
    %42 = arith.select %41, %40, %cst : f32
    %43 = arith.addf %42, %out : f32
    %44 = arith.subi %19, %14 : index
    %45 = arith.subi %26, %21 : index
    %46 = arith.muli %44, %45 : index
    %47 = arith.subi %33, %28 : index
    %48 = arith.muli %46, %47 : index
    %49 = arith.index_cast %48 : index to i64
    %50 = arith.sitofp %49 : i64 to f32
    linalg.yield %43, %50 : f32, f32
    } -> (tensor<1x8x8x8x8xf32>, tensor<8x8x8xf32>)
  %5 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%4#1 : tensor<8x8x8xf32>) outs(%4#0 : tensor<1x8x8x8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
    %6 = arith.divf %out, %in : f32
    linalg.yield %6 : f32
    } -> tensor<1x8x8x8x8xf32>
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten._adaptive_avg_pool3d.default
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel', 'parallel', 'reduction', 'reduction', 'reduction']
  loops:
    d0: 1 (parallel)
    d1: 8 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
    d4: 8 (parallel)
    d5: 3 (reduction)
    d6: 3 (reduction)
    d7: 3 (reduction)
  indexing_maps:
    args_0 (1, 8, 16, 16, 16) operand: affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d6, d7)>
    _adaptive_avg_pool3d (1, 8, 8, 8, 8) result: affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
  vectorizable_loop: d4

Working set: 147456 bytes -> tier T1.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop
  l2fetch: working set 147456 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide

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
