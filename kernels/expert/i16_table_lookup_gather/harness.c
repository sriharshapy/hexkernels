/* i16_table_lookup_gather harness (v5, L2: gather + vtcm). int16 table lookup
 * out[i]=table[idx[i]] over a large DDR-resident table, random indices. Harness
 * owns main(): fills table + indices, builds the scalar reference, poisons out,
 * times the candidate (kernel-only pcycles), bit-exact compares. Maps VTCM
 * identity so a gather candidate can stage the table on-chip. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef TW
#define TW 16384             /* table halfwords = 32KB */
#endif
#ifndef NI
#define NI 32768             /* lookups; out = 64KB. Multiple of 64. */
#endif

static int16_t table[TW] HVX_ALIGN;
static int16_t idx[NI]   HVX_ALIGN;
static int16_t out[NI]   HVX_ALIGN;
static int16_t ref[NI]   HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0x51550713u;
    for (int i = 0; i < TW; i++) table[i] = (int16_t)(hvx_lcg(&s) >> 16);
    for (int i = 0; i < NI; i++) idx[i] = (int16_t)(hvx_lcg(&s) % TW);
    idx[0] = 0; idx[1] = TW - 1; idx[2] = 0; idx[NI-1] = TW/2;

    for (int i = 0; i < NI; i++) ref[i] = table[idx[i]];

    { HVX_Vector vp = Q6_V_vsplat_R(0xA5A5A5A5u);
      int i=0; for (; i+64<=NI; i+=64) *(HVX_Vector*)(out+i)=vp;
      for (; i<NI; i++) out[i]=(int16_t)0xA5A5; }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(table, idx, out, TW, NI); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors=0, fb=-1; long gv=0, ev=0;
    for (int i=0;i<NI;i++) if (out[i]!=ref[i]) { errors++; if(fb<0){fb=i;gv=out[i];ev=ref[i];} }
    hvx_report(errors, NI, fb, gv, ev);
    return errors?1:0;
}
