Accelerate the kernel `fp32_upsample_bilinear2d` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v__to_copy[4096];
static int64_t v_arange[8];
static float v__to_copy_1[8];
static float v_add[8];
static float v_mul[8];
static float v_sub[8];
static float v_clamp[8];
static float v_view[8];
static int64_t v__to_copy_2[8];
static int64_t v_add_1[8];
static int64_t v_clamp_1[8];
static int64_t v_arange_1[8];
static float v__to_copy_3[8];
static float v_add_2[8];
static float v_mul_1[8];
static float v_sub_1[8];
static float v_clamp_2[8];
static float v_view_1[8];
static int64_t v__to_copy_4[8];
static int64_t v_add_3[8];
static int64_t v_clamp_3[8];
static float v_index[1024];
static float v_index_1[1024];
static float v_index_2[1024];
static float v_index_3[1024];
static float v_sub_2[8];
static float v_clamp_4[8];
static float v__to_copy_5[8];
static float v_sub_3[1024];
static float v_mul_2[1024];
static float v_add_4[1024];
static float v_sub_4[1024];
static float v_mul_3[1024];
static float v_add_5[1024];
static float v_sub_5[8];
static float v_clamp_5[8];
static float v__to_copy_6[8];
static float v_sub_6[1024];
static float v_mul_4[1024];
static float v_add_6[1024];
static float v__to_copy_7[1024];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 16; i2++) {
        for (int i3 = 0; i3 < 16; i3++) {
          v__to_copy[i1*256 + i2*16 + i3*1] = v_args_0[i1*256 + i2*16 + i3*1];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_arange[i0*1] = (int64_t)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v__to_copy_1[i0*1] = v_arange[i0*1];
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_add[i0*1] = (v__to_copy_1[i0*1] + 0.5f);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_mul[i0*1] = (v_add[i0*1] * 2.0f);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_sub[i0*1] = (v_mul[i0*1] - 0.5f);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_clamp[i0*1] = (v_sub[i0*1] < 0.0f ? 0.0f : v_sub[i0*1]);
  }
  for (int i = 0; i < 8; i++) v_view[i] = v_clamp[i];
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v__to_copy_2[i0*1] = v_view[i0*1];
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_add_1[i0*1] = (v__to_copy_2[i0*1] + 1);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_1[i0*1] = (v_add_1[i0*1] > 15 ? 15 : v_add_1[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_arange_1[i0*1] = (int64_t)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v__to_copy_3[i0*1] = v_arange_1[i0*1];
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_add_2[i0*1] = (v__to_copy_3[i0*1] + 0.5f);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_mul_1[i0*1] = (v_add_2[i0*1] * 2.0f);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_sub_1[i0*1] = (v_mul_1[i0*1] - 0.5f);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_clamp_2[i0*1] = (v_sub_1[i0*1] < 0.0f ? 0.0f : v_sub_1[i0*1]);
  }
  for (int i = 0; i < 8; i++) v_view_1[i] = v_clamp_2[i];
  for (int i0 = 0; i0 < 8; i0++) {
    v__to_copy_4[i0*1] = v_view_1[i0*1];
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_add_3[i0*1] = (v__to_copy_4[i0*1] + 1);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_clamp_3[i0*1] = (v_add_3[i0*1] > 15 ? 15 : v_add_3[i0*1]);
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v_index[i1*64 + i2*8 + i3*1] = v__to_copy[(i0)*4096 + (i1)*256 + (v__to_copy_2[(i2)])*16 + (v__to_copy_4[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v_index_1[i1*64 + i2*8 + i3*1] = v__to_copy[(i0)*4096 + (i1)*256 + (v__to_copy_2[(i2)])*16 + (v_clamp_3[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v_index_2[i1*64 + i2*8 + i3*1] = v__to_copy[(i0)*4096 + (i1)*256 + (v_clamp_1[(i2)])*16 + (v__to_copy_4[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v_index_3[i1*64 + i2*8 + i3*1] = v__to_copy[(i0)*4096 + (i1)*256 + (v_clamp_1[(i2)])*16 + (v_clamp_3[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_sub_2[i0*1] = (v_view_1[i0*1] - v__to_copy_4[i0*1]);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_clamp_4[i0*1] = ((v_sub_2[i0*1] < 0.0f ? 0.0f : v_sub_2[i0*1]) > 1.0f ? 1.0f : (v_sub_2[i0*1] < 0.0f ? 0.0f : v_sub_2[i0*1]));
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v__to_copy_5[i0*1] = v_clamp_4[i0*1];
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v_sub_3[i1*64 + i2*8 + i3*1] = (v_index_1[i1*64 + i2*8 + i3*1] - v_index[i1*64 + i2*8 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v_mul_2[i1*64 + i2*8 + i3*1] = (v_sub_3[i1*64 + i2*8 + i3*1] * v__to_copy_5[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v_add_4[i1*64 + i2*8 + i3*1] = (v_index[i1*64 + i2*8 + i3*1] + v_mul_2[i1*64 + i2*8 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v_sub_4[i1*64 + i2*8 + i3*1] = (v_index_3[i1*64 + i2*8 + i3*1] - v_index_2[i1*64 + i2*8 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v_mul_3[i1*64 + i2*8 + i3*1] = (v_sub_4[i1*64 + i2*8 + i3*1] * v__to_copy_5[i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v_add_5[i1*64 + i2*8 + i3*1] = (v_index_2[i1*64 + i2*8 + i3*1] + v_mul_3[i1*64 + i2*8 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_sub_5[i0*1] = (v_view[i0*1] - v__to_copy_2[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_5[i0*1] = ((v_sub_5[i0*1] < 0.0f ? 0.0f : v_sub_5[i0*1]) > 1.0f ? 1.0f : (v_sub_5[i0*1] < 0.0f ? 0.0f : v_sub_5[i0*1]));
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v__to_copy_6[i0*1] = v_clamp_5[i0*1];
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v_sub_6[i1*64 + i2*8 + i3*1] = (v_add_5[i1*64 + i2*8 + i3*1] - v_add_4[i1*64 + i2*8 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v_mul_4[i1*64 + i2*8 + i3*1] = (v_sub_6[i1*64 + i2*8 + i3*1] * v__to_copy_6[i2*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v_add_6[i1*64 + i2*8 + i3*1] = (v_add_4[i1*64 + i2*8 + i3*1] + v_mul_4[i1*64 + i2*8 + i3*1]);
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v__to_copy_7[i1*64 + i2*8 + i3*1] = v_add_6[i1*64 + i2*8 + i3*1];
        }
      }
    }
  }
  for (int i = 0; i < 1024; i++) { out0[i] = v__to_copy_7[i]; }
}
```

The same computation as MLIR Linalg, emitted by torch-mlir from this graph. This is the compiler's own structured form: `iterator_types` marks each loop parallel or reduction, and the `#map` aliases are the affine indexing maps -- one result per operand axis, so an axis pinned to a constant (`(d0, 0)`) is a broadcast that does not advance. Named ops such as `linalg.matmul` carry their iterator types in the op definition rather than spelling them out.

```mlir
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, 0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d2, 0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @main(%arg0: tensor<1x16x16x16xf32>) -> tensor<1x16x8x8xf32> {
  %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<8xi64>) {
    ^bb0(%out: i64):
    %38 = linalg.index 0 : index
    %39 = arith.index_cast %38 : index to i64
    linalg.yield %39 : i64
    } -> tensor<8xi64>
  %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<8xi64>) outs(%2 : tensor<8xf32>) {
    ^bb0(%in: i64, %out: f32):
    %38 = arith.sitofp %in : i64 to f32
    linalg.yield %38 : f32
    } -> tensor<8xf32>
  %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%3 : tensor<8xf32>) outs(%2 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
    %38 = arith.addf %in, %cst : f32
    linalg.yield %38 : f32
    } -> tensor<8xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%4 : tensor<8xf32>) outs(%2 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
    %38 = arith.mulf %in, %cst_0 : f32
    linalg.yield %38 : f32
    } -> tensor<8xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%5 : tensor<8xf32>) outs(%2 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
    %38 = arith.subf %in, %cst : f32
    linalg.yield %38 : f32
    } -> tensor<8xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%6 : tensor<8xf32>) outs(%2 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
    %38 = arith.cmpf ult, %in, %cst_1 : f32
    %39 = arith.select %38, %cst_1, %in : f32
    linalg.yield %39 : f32
    } -> tensor<8xf32>
  %9 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<8x1xf32>) outs(%8 : tensor<8x1xi64>) {
    ^bb0(%in: f32, %out: i64):
    %38 = arith.fptosi %in : f32 to i64
    linalg.yield %38 : i64
    } -> tensor<8x1xi64>
  %10 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<8x1xi64>) outs(%8 : tensor<8x1xi64>) {
    ^bb0(%in: i64, %out: i64):
    %38 = arith.addi %in, %c1_i64 : i64
    linalg.yield %38 : i64
    } -> tensor<8x1xi64>
  %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<8x1xi64>) outs(%8 : tensor<8x1xi64>) {
    ^bb0(%in: i64, %out: i64):
    %38 = arith.cmpi sge, %in, %c15_i64 : i64
    %39 = arith.select %38, %c15_i64, %in : i64
    linalg.yield %39 : i64
    } -> tensor<8x1xi64>
  %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%7 : tensor<8xf32>) outs(%0 : tensor<8xi64>) {
    ^bb0(%in: f32, %out: i64):
    %38 = arith.fptosi %in : f32 to i64
    linalg.yield %38 : i64
    } -> tensor<8xi64>
  %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%12 : tensor<8xi64>) outs(%0 : tensor<8xi64>) {
    ^bb0(%in: i64, %out: i64):
    %38 = arith.addi %in, %c1_i64 : i64
    linalg.yield %38 : i64
    } -> tensor<8xi64>
  %14 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%13 : tensor<8xi64>) outs(%0 : tensor<8xi64>) {
    ^bb0(%in: i64, %out: i64):
    %38 = arith.cmpi sge, %in, %c15_i64 : i64
    %39 = arith.select %38, %c15_i64, %in : i64
    linalg.yield %39 : i64
    } -> tensor<8xi64>
  %16 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%15 : tensor<16xi64>) {
    ^bb0(%out: i64):
    %38 = linalg.index 0 : index
    %39 = arith.index_cast %38 : index to i64
    linalg.yield %39 : i64
    } -> tensor<16xi64>
  %18 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%17 : tensor<1xi64>) {
    ^bb0(%out: i64):
    linalg.yield %c0_i64 : i64
    } -> tensor<1xi64>
  %20 = linalg.generic {indexing_maps = [#map2, #map3, #map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_4, %expanded_3, %9, %12 : tensor<1x1x1x1xi64>, tensor<16x1x1xi64>, tensor<8x1xi64>, tensor<8xi64>) outs(%19 : tensor<1x16x8x8xf32>) {
    ^bb0(%in: i64, %in_5: i64, %in_6: i64, %in_7: i64, %out: f32):
    %38 = arith.cmpi slt, %in, %c0_i64 : i64
    %39 = arith.addi %in, %c1_i64 : i64
    %40 = arith.select %38, %39, %in : i64
    %41 = arith.index_cast %40 : i64 to index
    %42 = arith.cmpi slt, %in_5, %c0_i64 : i64
    %43 = arith.addi %in_5, %c16_i64 : i64
    %44 = arith.select %42, %43, %in_5 : i64
    %45 = arith.index_cast %44 : i64 to index
    %46 = arith.cmpi slt, %in_6, %c0_i64 : i64
    %47 = arith.addi %in_6, %c16_i64 : i64
    %48 = arith.select %46, %47, %in_6 : i64
    %49 = arith.index_cast %48 : i64 to index
    %50 = arith.cmpi slt, %in_7, %c0_i64 : i64
    %51 = arith.addi %in_7, %c16_i64 : i64
    %52 = arith.select %50, %51, %in_7 : i64
    %53 = arith.index_cast %52 : i64 to index
    %extracted = tensor.extract %arg0[%41, %45, %49, %53] : tensor<1x16x16x16xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x16x8x8xf32>
  %21 = linalg.generic {indexing_maps = [#map2, #map3, #map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_4, %expanded_3, %9, %14 : tensor<1x1x1x1xi64>, tensor<16x1x1xi64>, tensor<8x1xi64>, tensor<8xi64>) outs(%19 : tensor<1x16x8x8xf32>) {
    ^bb0(%in: i64, %in_5: i64, %in_6: i64, %in_7: i64, %out: f32):
    %38 = arith.cmpi slt, %in, %c0_i64 : i64
    %39 = arith.addi %in, %c1_i64 : i64
    %40 = arith.select %38, %39, %in : i64
    %41 = arith.index_cast %40 : i64 to index
    %42 = arith.cmpi slt, %in_5, %c0_i64 : i64
    %43 = arith.addi %in_5, %c16_i64 : i64
    %44 = arith.select %42, %43, %in_5 : i64
    %45 = arith.index_cast %44 : i64 to index
    %46 = arith.cmpi slt, %in_6, %c0_i64 : i64
    %47 = arith.addi %in_6, %c16_i64 : i64
    %48 = arith.select %46, %47, %in_6 : i64
    %49 = arith.index_cast %48 : i64 to index
    %50 = arith.cmpi slt, %in_7, %c0_i64 : i64
    %51 = arith.addi %in_7, %c16_i64 : i64
    %52 = arith.select %50, %51, %in_7 : i64
    %53 = arith.index_cast %52 : i64 to index
    %extracted = tensor.extract %arg0[%41, %45, %49, %53] : tensor<1x16x16x16xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x16x8x8xf32>
  %22 = linalg.generic {indexing_maps = [#map2, #map3, #map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_4, %expanded_3, %11, %12 : tensor<1x1x1x1xi64>, tensor<16x1x1xi64>, tensor<8x1xi64>, tensor<8xi64>) outs(%19 : tensor<1x16x8x8xf32>) {
    ^bb0(%in: i64, %in_5: i64, %in_6: i64, %in_7: i64, %out: f32):
    %38 = arith.cmpi slt, %in, %c0_i64 : i64
    %39 = arith.addi %in, %c1_i64 : i64
    %40 = arith.select %38, %39, %in : i64
    %41 = arith.index_cast %40 : i64 to index
    %42 = arith.cmpi slt, %in_5, %c0_i64 : i64
    %43 = arith.addi %in_5, %c16_i64 : i64
    %44 = arith.select %42, %43, %in_5 : i64
    %45 = arith.index_cast %44 : i64 to index
    %46 = arith.cmpi slt, %in_6, %c0_i64 : i64
    %47 = arith.addi %in_6, %c16_i64 : i64
    %48 = arith.select %46, %47, %in_6 : i64
    %49 = arith.index_cast %48 : i64 to index
    %50 = arith.cmpi slt, %in_7, %c0_i64 : i64
    %51 = arith.addi %in_7, %c16_i64 : i64
    %52 = arith.select %50, %51, %in_7 : i64
    %53 = arith.index_cast %52 : i64 to index
    %extracted = tensor.extract %arg0[%41, %45, %49, %53] : tensor<1x16x16x16xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x16x8x8xf32>
  %23 = linalg.generic {indexing_maps = [#map2, #map3, #map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_4, %expanded_3, %11, %14 : tensor<1x1x1x1xi64>, tensor<16x1x1xi64>, tensor<8x1xi64>, tensor<8xi64>) outs(%19 : tensor<1x16x8x8xf32>) {
    ^bb0(%in: i64, %in_5: i64, %in_6: i64, %in_7: i64, %out: f32):
    %38 = arith.cmpi slt, %in, %c0_i64 : i64
    %39 = arith.addi %in, %c1_i64 : i64
    %40 = arith.select %38, %39, %in : i64
    %41 = arith.index_cast %40 : i64 to index
    %42 = arith.cmpi slt, %in_5, %c0_i64 : i64
    %43 = arith.addi %in_5, %c16_i64 : i64
    %44 = arith.select %42, %43, %in_5 : i64
    %45 = arith.index_cast %44 : i64 to index
    %46 = arith.cmpi slt, %in_6, %c0_i64 : i64
    %47 = arith.addi %in_6, %c16_i64 : i64
    %48 = arith.select %46, %47, %in_6 : i64
    %49 = arith.index_cast %48 : i64 to index
    %50 = arith.cmpi slt, %in_7, %c0_i64 : i64
    %51 = arith.addi %in_7, %c16_i64 : i64
    %52 = arith.select %50, %51, %in_7 : i64
    %53 = arith.index_cast %52 : i64 to index
    %extracted = tensor.extract %arg0[%41, %45, %49, %53] : tensor<1x16x16x16xf32>
    linalg.yield %extracted : f32
    } -> tensor<1x16x8x8xf32>
  %24 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%7, %12 : tensor<8xf32>, tensor<8xi64>) outs(%2 : tensor<8xf32>) {
    ^bb0(%in: f32, %in_5: i64, %out: f32):
    %38 = arith.sitofp %in_5 : i64 to f32
    %39 = arith.subf %in, %38 : f32
    linalg.yield %39 : f32
    } -> tensor<8xf32>
  %25 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%24 : tensor<8xf32>) outs(%2 : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
    %38 = arith.cmpf ult, %in, %cst_1 : f32
    %39 = arith.select %38, %cst_1, %in : f32
    %40 = arith.cmpf ugt, %39, %cst_2 : f32
    %41 = arith.select %40, %cst_2, %39 : f32
    linalg.yield %41 : f32
    } -> tensor<8xf32>
  %26 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21, %20 : tensor<1x16x8x8xf32>, tensor<1x16x8x8xf32>) outs(%19 : tensor<1x16x8x8xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
    %38 = arith.subf %in, %in_5 : f32
    linalg.yield %38 : f32
    } -> tensor<1x16x8x8xf32>
  %27 = linalg.generic {indexing_maps = [#map6, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %25 : tensor<1x16x8x8xf32>, tensor<8xf32>) outs(%19 : tensor<1x16x8x8xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
    %38 = arith.mulf %in, %in_5 : f32
    linalg.yield %38 : f32
    } -> tensor<1x16x8x8xf32>
  %28 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %27 : tensor<1x16x8x8xf32>, tensor<1x16x8x8xf32>) outs(%19 : tensor<1x16x8x8xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
    %38 = arith.addf %in, %in_5 : f32
    linalg.yield %38 : f32
    } -> tensor<1x16x8x8xf32>
  %29 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23, %22 : tensor<1x16x8x8xf32>, tensor<1x16x8x8xf32>) outs(%19 : tensor<1x16x8x8xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
    %38 = arith.subf %in, %in_5 : f32
    linalg.yield %38 : f32
    } -> tensor<1x16x8x8xf32>
  %30 = linalg.generic {indexing_maps = [#map6, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%29, %25 : tensor<1x16x8x8xf32>, tensor<8xf32>) outs(%19 : tensor<1x16x8x8xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
    %38 = arith.mulf %in, %in_5 : f32
    linalg.yield %38 : f32
    } -> tensor<1x16x8x8xf32>
  %31 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %30 : tensor<1x16x8x8xf32>, tensor<1x16x8x8xf32>) outs(%19 : tensor<1x16x8x8xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
    %38 = arith.addf %in, %in_5 : f32
    linalg.yield %38 : f32
    } -> tensor<1x16x8x8xf32>
  %33 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded, %9 : tensor<8x1xf32>, tensor<8x1xi64>) outs(%32 : tensor<8x1xf32>) {
    ^bb0(%in: f32, %in_5: i64, %out: f32):
    %38 = arith.sitofp %in_5 : i64 to f32
    %39 = arith.subf %in, %38 : f32
    linalg.yield %39 : f32
    } -> tensor<8x1xf32>
  %34 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%33 : tensor<8x1xf32>) outs(%32 : tensor<8x1xf32>) {
    ^bb0(%in: f32, %out: f32):
    %38 = arith.cmpf ult, %in, %cst_1 : f32
    %39 = arith.select %38, %cst_1, %in : f32
    %40 = arith.cmpf ugt, %39, %cst_2 : f32
    %41 = arith.select %40, %cst_2, %39 : f32
    linalg.yield %41 : f32
    } -> tensor<8x1xf32>
  %35 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %28 : tensor<1x16x8x8xf32>, tensor<1x16x8x8xf32>) outs(%19 : tensor<1x16x8x8xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
    %38 = arith.subf %in, %in_5 : f32
    linalg.yield %38 : f32
    } -> tensor<1x16x8x8xf32>
  %36 = linalg.generic {indexing_maps = [#map6, #map4, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35, %34 : tensor<1x16x8x8xf32>, tensor<8x1xf32>) outs(%19 : tensor<1x16x8x8xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
    %38 = arith.mulf %in, %in_5 : f32
    linalg.yield %38 : f32
    } -> tensor<1x16x8x8xf32>
  %37 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %36 : tensor<1x16x8x8xf32>, tensor<1x16x8x8xf32>) outs(%19 : tensor<1x16x8x8xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
    %38 = arith.addf %in, %in_5 : f32
    linalg.yield %38 : f32
    } -> tensor<1x16x8x8xf32>
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten._to_copy.default
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 16 (parallel)
    d3: 16 (parallel)
  indexing_maps:
    args_0 (1, 16, 16, 16) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    _to_copy (1, 16, 16, 16) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.arange.start_step
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    arange (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten._to_copy.default
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    arange (8,) operand: affine_map<(d0) -> (d0)>
    _to_copy_1 (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.add.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    _to_copy_1 (8,) operand: affine_map<(d0) -> (d0)>
    add (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.mul.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    add (8,) operand: affine_map<(d0) -> (d0)>
    mul (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.sub.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    mul (8,) operand: affine_map<(d0) -> (d0)>
    sub (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    sub (8,) operand: affine_map<(d0) -> (d0)>
    clamp (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.view.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 8 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    clamp (8,) operand: affine_map<(d0, d1) -> (d1)>
    view (8, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten._to_copy.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 8 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    view (8, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    _to_copy_2 (8, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.add.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 8 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    _to_copy_2 (8, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    add_1 (8, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 8 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    add_1 (8, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_1 (8, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.arange.start_step
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    arange_1 (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten._to_copy.default
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    arange_1 (8,) operand: affine_map<(d0) -> (d0)>
    _to_copy_3 (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.add.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    _to_copy_3 (8,) operand: affine_map<(d0) -> (d0)>
    add_2 (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.mul.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    add_2 (8,) operand: affine_map<(d0) -> (d0)>
    mul_1 (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.sub.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    mul_1 (8,) operand: affine_map<(d0) -> (d0)>
    sub_1 (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    sub_1 (8,) operand: affine_map<(d0) -> (d0)>
    clamp_2 (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.view.default
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    clamp_2 (8,) operand: affine_map<(d0) -> (d0)>
    view_1 (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten._to_copy.default
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    view_1 (8,) operand: affine_map<(d0) -> (d0)>
    _to_copy_4 (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.add.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    _to_copy_4 (8,) operand: affine_map<(d0) -> (d0)>
    add_3 (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    add_3 (8,) operand: affine_map<(d0) -> (d0)>
    clamp_3 (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    _to_copy (1, 16, 16, 16) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    _to_copy_2 (8, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    _to_copy_4 (8,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index (1, 16, 8, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    _to_copy (1, 16, 16, 16) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    _to_copy_2 (8, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_3 (8,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_1 (1, 16, 8, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    _to_copy (1, 16, 16, 16) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_1 (8, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    _to_copy_4 (8,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_2 (1, 16, 8, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    _to_copy (1, 16, 16, 16) operand: affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
    clamp_1 (8, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    clamp_3 (8,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    index_3 (1, 16, 8, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.sub.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    view_1 (8,) operand: affine_map<(d0) -> (d0)>
    _to_copy_4 (8,) operand: affine_map<(d0) -> (d0)>
    sub_2 (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.clamp.default
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    sub_2 (8,) operand: affine_map<(d0) -> (d0)>
    clamp_4 (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten._to_copy.default
  iterator_types = ['parallel']
  loops:
    d0: 8 (parallel)
  indexing_maps:
    clamp_4 (8,) operand: affine_map<(d0) -> (d0)>
    _to_copy_5 (8,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    index_1 (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    index (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    sub_3 (1, 16, 8, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    sub_3 (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    _to_copy_5 (8,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_2 (1, 16, 8, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    index (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_2 (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_4 (1, 16, 8, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    index_3 (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    index_2 (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    sub_4 (1, 16, 8, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    sub_4 (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    _to_copy_5 (8,) operand: affine_map<(d0, d1, d2, d3) -> (d3)>
    mul_3 (1, 16, 8, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    index_2 (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_3 (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_5 (1, 16, 8, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 8 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    view (8, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    _to_copy_2 (8, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    sub_5 (8, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.clamp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 8 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    sub_5 (8, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    clamp_5 (8, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten._to_copy.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 8 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    clamp_5 (8, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    _to_copy_6 (8, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    add_5 (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_4 (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    sub_6 (1, 16, 8, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    sub_6 (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    _to_copy_6 (8, 1) operand: affine_map<(d0, d1, d2, d3) -> (d2, 0)>
    mul_4 (1, 16, 8, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten.add.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    add_4 (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    mul_4 (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    add_6 (1, 16, 8, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3
aten._to_copy.default
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
  indexing_maps:
    add_6 (1, 16, 8, 8) operand: affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
    _to_copy_7 (1, 16, 8, 8) result: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  vectorizable_loop: d3

Working set: 20480 bytes -> tier T1.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop
  l2fetch: working set 20480 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide

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
