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
