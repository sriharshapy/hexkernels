#include <stdint.h>
#include <math.h>

static int64_t v_searchsorted[786432];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, int64_t *out0) {
  for (int i0 = 0; i0 < 768; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      int lo = 0, hi = 1024;
      while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (v_args_0[i0*1024 + mid] < v_args_1[i0*1024 + i1*1]) lo = mid + 1;
        else hi = mid;
      }
      v_searchsorted[i0*1024 + i1*1] = (int64_t)lo;
    }
  }
  for (int i = 0; i < 786432; i++) { out0[i] = v_searchsorted[i]; }
}
