/* i8_table_lookup_gather harness (v5, L2: gather + vtcm). Word table lookup
 * out[i]=table[idx[i]] over a large DDR-resident table with random indices.
 * Harness owns main(): fills table + indices (deterministic LCG), builds the
 * scalar reference gather, poisons out, times the candidate (kernel-only
 * pcycles), bit-exact compares. Maps VTCM identity so a gather candidate can
 * stage the table on-chip (historical vgather fault 0x26 = VTCM not mapped). */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef TW
#define TW 16384             /* table words = 64KB (VTCM-resident, exceeds L1) */
#endif
#ifndef NI
#define NI 32768             /* lookups; out = 128KB. Multiple of 32. */
#endif

static int32_t table[TW] HVX_ALIGN;
static int32_t idx[NI]   HVX_ALIGN;
static int32_t out[NI]   HVX_ALIGN;
static int32_t ref[NI]   HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0xABCD1234u;
    for (int i = 0; i < TW; i++) table[i] = (int32_t)hvx_lcg(&s);
    for (int i = 0; i < NI; i++) idx[i] = (int32_t)(hvx_lcg(&s) % TW);
    /* edge cases: first/last table entry, repeated index, boundary positions */
    idx[0] = 0; idx[1] = TW - 1; idx[2] = 0; idx[NI-1] = TW/2;

    for (int i = 0; i < NI; i++) ref[i] = table[idx[i]];

    { HVX_Vector vp = Q6_V_vsplat_R(0xA5A5A5A5u);
      int i=0; for (; i+32<=NI; i+=32) *(HVX_Vector*)(out+i)=vp;
      for (; i<NI; i++) out[i]=(int32_t)0xA5A5A5A5; }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(table, idx, out, TW, NI); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors=0, fb=-1; long gv=0, ev=0;
    for (int i=0;i<NI;i++) if (out[i]!=ref[i]) { errors++; if(fb<0){fb=i;gv=out[i];ev=ref[i];} }
    hvx_report(errors, NI, fb, gv, ev);
    return errors?1:0;
}
