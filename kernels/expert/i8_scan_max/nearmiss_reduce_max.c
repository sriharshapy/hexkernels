/* Near-miss: computes the global maximum and broadcasts it to every output element.
   i.e. out[i] = global_max(in) for all i.
   This is correct for out[n-1] (the last element) but wrong for all earlier positions
   where the running max is still increasing. Fails immediately at index 0 which
   should be -128 (the first element) but gets 127 (the global max). */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int n){
    /* BUG: global max instead of prefix max */
    int8_t gmax = -128;
    for (int i=0; i<n; i++) if(in[i]>gmax) gmax=in[i];
    for (int i=0; i<n; i++) out[i]=gmax;
}
