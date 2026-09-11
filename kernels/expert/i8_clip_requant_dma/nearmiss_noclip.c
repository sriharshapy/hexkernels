/* Near-miss: applies int8 saturation but SKIPS the application-level clip to
 * [lo,hi]. Correct wherever the requantized value already lies in [lo,hi] but
 * differs whenever it falls outside (|value| > hi/|lo|). */
#include <stdint.h>
void candidate_kernel(const int32_t *a, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp, int8_t lo, int8_t hi){
  (void)lo; (void)hi;
  for (int i=0;i<n;i++){
    int64_t v=(int64_t)a[i]*(int64_t)mult;
    int64_t half=(shift>0)?((int64_t)1<<(shift-1)):0;
    int64_t r=(v>=0)?((v+half)>>shift):-(((-v)+half)>>shift);
    r+=zp; if(r>127)r=127; if(r<-128)r=-128;   /* int8 sat only, no [lo,hi] clip */
    out[i]=(int8_t)r;
  }
}
