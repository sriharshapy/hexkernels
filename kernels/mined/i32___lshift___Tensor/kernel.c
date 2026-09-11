/* i32___lshift___Tensor -- `aten::__lshift__.Tensor` at (8, 64)
 *
 * MINED, NOT CHOSEN BY HAND. The op came from `coverage`'s measured `covered` list
 * in harvest order, the size was solved for tier T0, and the mechanisms follow
 * from that size -- see `forge2/mine.py` and the note in the prompt. What is
 * hand-written is the arithmetic below and the memory plan.
 *
 * Derived from the reference in the prompt, the loop schedule shipped with it, and
 * the vendor headers under
 *   <SDK>/tools/HEXAGON_Tools/19.0.04/Tools/target/hexagon/include/
 *   (hexagon_types.h, hexagon_protos.h, hvx_hexagon_protos.h),
 * plus the V75 PRM (l2fetch descriptor, 80-N2040-57_AB sec 5.10.6.3 p87 and the
 * instruction page p317).
 *
 * THE ARITHMETIC: a per-lane variable left shift, one instruction and exact. The shift COUNT is a vector here, not a scalar, which is why this is `vasl_VwVw` and not `vasl_VwR` -- the reference shifts each element by its own operand
 *
 * THE SHARED CORE below is one implementation reused across the mined batches
 * rather than fifty separate ones. Every function in it was checked against libm
 * ON THE SIMULATOR before any kernel used it -- 21 functions, worst case 1.3e-5
 * absolute and 1.5e-4 relative against the harness's 1e-4 + 1e-3 budget. Doing it
 * the other way round is where three of this session's bugs came from: a bad
 * `log2` would otherwise have looked like eight unrelated failures. Unused
 * functions are `static inline` and are dropped by the compiler.
 *
 * MEMORY PLAN: 12 KB or less fits inside the 16 KB L1D, so `plan_for` grants NO memory mechanism and this kernel issues no prefetch. Issuing one anyway would exceed the entitlement, and was measured elsewhere in this corpus at ~9% cost for nothing -- the hardware prefetcher already covers a sequential stream this short.
 */
#include <stdint.h>
#include <stddef.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static int32_t g_pad[32] __attribute__((aligned(128)));

/* l2fetch(Rs,Rtt): Rtt[31:16] Width, Rtt[15:0] Height, Rtt[47:32] Stride
 * (V75 PRM p317). A zero subfield CANCELS the prefetch rather than issuing one. */
static inline void l2fetch_block(const void *p, unsigned width, unsigned rows) {
    uint64_t rtt = ((uint64_t)width << 32) | ((uint64_t)width << 16)
                 | (uint64_t)rows;
    Q6_l2fetch_AP((void *)p, rtt);
}

static inline int f32_bits(float f) { int w; __builtin_memcpy(&w, &f, 4); return w; }
static inline float bits_f32(int w) { float f; __builtin_memcpy(&f, &w, 4); return f; }
static inline HVX_Vector vsplatf(float f) { return Q6_V_vsplat_R(f32_bits(f)); }

/* IEEE-out float arithmetic on v75 goes through qf32 -- V6_vadd_sf / V6_vsub_sf /
 * V6_vmpy_sf_sf are declared and unselectable, so each operation is "do it in
 * qf32, then Vd.sf=Vu.qf32". MIN/MAX/COMPARE are the exception and take IEEE
 * operands directly (verified in the disassembly: `v2.sf = vmax(v0.sf,v1.sf)`). */
static inline HVX_Vector sf_add(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
static inline HVX_Vector sf_sub(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
}
static inline HVX_Vector sf_mul(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}
static inline HVX_Vector to_qf32(HVX_Vector a) {
    return Q6_Vqf32_vadd_VsfVsf(a, Q6_V_vzero());
}

#define ABS_MASK 0x7FFFFFFF
#define SGN_MASK 0x80000000
static inline HVX_Vector sf_abs(HVX_Vector x) {
    return Q6_V_vand_VV(x, Q6_V_vsplat_R(ABS_MASK));
}

/* ---- reciprocal: Newton from the classic bit-trick seed --------------------
 * seed = 0x7EF311C3 - bits(x)  (~5e-2 relative); each step squares the error, so
 * two steps reach ~1e-7 -- four orders inside the harness's 1e-3 relative. */
static inline HVX_Vector vrecip(HVX_Vector x) {
    const HVX_Vector two = vsplatf(2.0f);
    HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311C3), x);
    y = sf_mul(y, sf_sub(two, sf_mul(x, y)));
    y = sf_mul(y, sf_sub(two, sf_mul(x, y)));
    return y;
}

/* ---- reciprocal square root, then sqrt -------------------------------------
 * seed = 0x5F375A86 - (bits(x) >> 1); Newton step y *= 1.5 - 0.5*x*y*y. The
 * arithmetic shift is safe because every argument here is non-negative, so the
 * sign bit is clear and an arithmetic shift equals a logical one. */
static inline HVX_Vector vrsqrt(HVX_Vector x) {
    const HVX_Vector half = vsplatf(0.5f), op5 = vsplatf(1.5f);
    const HVX_Vector hx = sf_mul(x, half);
    HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x5F375A86),
                                   Q6_Vw_vasr_VwR(x, 1));
    y = sf_mul(y, sf_sub(op5, sf_mul(hx, sf_mul(y, y))));
    y = sf_mul(y, sf_sub(op5, sf_mul(hx, sf_mul(y, y))));
    return y;
}
static inline HVX_Vector vsqrt(HVX_Vector x) {
    /* x * rsqrt(x), with the zero case forced: rsqrt(0) is +inf and 0*inf is NaN,
     * while sqrt(0) is 0. One compare is cheaper than a special-cased Newton. */
    const HVX_Vector r = sf_mul(x, vrsqrt(x));
    const HVX_VectorPred z = Q6_Q_vcmp_eq_VwVw(sf_abs(x), Q6_V_vzero());
    return Q6_V_vmux_QVV(z, Q6_V_vzero(), r);
}

/* ---- exp2, and exp/expm1 from it ------------------------------------------
 * Range reduction in BASE TWO, which avoids the ln2 cancellation a base-e
 * reduction needs:  2^x = 2^n * 2^f,  n = round(x), f = x - n in [-0.5, 0.5].
 *
 * `Q6_Vw_equals_Vsf` TRUNCATES toward zero (measured: 12.8 -> 12, -29.7 -> -29),
 * so n is built as trunc(x + copysign(0.5, x)) -- round-half-away, which is what
 * keeps f inside the interval the polynomial is fitted for on BOTH signs. An
 * earlier version used trunc(x - 0.5), correct only for x <= 0.
 *
 * 2^f is a degree-6 Horner in ln(2)^k/k!; the first dropped term is bounded by
 * (ln2^7/5040)*0.5^7 ~ 1.2e-7 relative, three orders inside the budget. 2^n is
 * assembled as an IEEE exponent field and clamped at both ends so an extreme
 * argument flushes to 0 or saturates rather than aliasing into a bogus exponent. */
#define E2C1 0.69314718055994531f
#define E2C2 0.24022650695910071f
#define E2C3 0.055504108664821579f
#define E2C4 0.0096181291076284772f
#define E2C5 0.0013333558146428443f
#define E2C6 0.00015403530393381610f

static inline HVX_Vector vexp2(HVX_Vector x) {
    const HVX_Vector half = vsplatf(0.5f);
    const HVX_Vector sgn = Q6_V_vand_VV(x, Q6_V_vsplat_R(SGN_MASK));
    const HVX_Vector bias = Q6_V_vor_VV(half, sgn);        /* copysign(0.5, x) */
    const HVX_Vector n = Q6_Vw_equals_Vsf(sf_add(x, bias));
    const HVX_Vector f = to_qf32(sf_sub(x, Q6_Vsf_equals_Vw(n)));

    HVX_Vector e = Q6_Vw_vadd_VwVw(n, Q6_V_vsplat_R(127));
    e = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(e, Q6_V_vzero()), Q6_V_vsplat_R(254));
    const HVX_Vector p2 = Q6_Vw_vasl_VwR(e, 23);

    HVX_Vector p = to_qf32(vsplatf(E2C6));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, f), vsplatf(E2C5));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, f), vsplatf(E2C4));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, f), vsplatf(E2C3));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, f), vsplatf(E2C2));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, f), vsplatf(E2C1));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, f), vsplatf(1.0f));
    return sf_mul(p2, Q6_Vsf_equals_Vqf32(p));
}

/* the three log/exp constants, defined before their first use */
#define LOG2_E   1.44269504088896341f
#define LN2      0.69314718055994531f
#define SQRT2    1.41421356237309505f

static inline HVX_Vector vexp(HVX_Vector x) {
    return vexp2(sf_mul(x, vsplatf(LOG2_E)));
}

/* ---- log2, and log from it -------------------------------------------------
 * Split the IEEE fields: x = m * 2^e with m in [1, 2), so log2(x) = e + log2(m).
 *
 * A TAYLOR SERIES IN (m - 1) IS NOT GOOD ENOUGH, measured: u = m-1 reaches 1 at
 * the top of the mantissa range, where a degree-6 truncation is off by 8.9e-2
 * absolute (max_rel 3.8 at x = 0.9839 -- worst exactly where e flips and m is
 * near 2). Every function built on log inherited it: asinh, acosh, atanh all
 * failed with it and pass without.
 *
 * The fix is the standard one -- rewrite as an atanh, whose argument is bounded
 * far more tightly:
 *
 *     log(m) = 2 * atanh(s),   s = (m - 1) / (m + 1),   |s| <= 1/3 for m in [1,2)
 *
 * and in fact |s| <= 0.1716 once the mantissa is centred on sqrt(2) below. The
 * odd series s + s^3/3 + s^5/5 + s^7/7 then has a next term of s^9/9 <= 1.4e-8,
 * four orders inside the harness's budget, with four multiplies. */
static inline HVX_Vector vlog2(HVX_Vector x) {
    /* centre the mantissa on sqrt(2): m in [sqrt(2)/2, sqrt(2)) so |s| <= 0.1716 */
    HVX_Vector e = Q6_Vw_vsub_VwVw(
        Q6_Vuw_vlsr_VuwR(Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7F800000)), 23),
        Q6_V_vsplat_R(127));
    HVX_Vector m = Q6_V_vor_VV(Q6_V_vand_VV(x, Q6_V_vsplat_R(0x007FFFFF)),
                               Q6_V_vsplat_R(0x3F800000));
    const HVX_VectorPred hi = Q6_Q_vcmp_gt_VsfVsf(m, vsplatf(SQRT2));
    m = Q6_V_vmux_QVV(hi, sf_mul(m, vsplatf(0.5f)), m);
    e = Q6_Vw_vadd_VwVw(e, Q6_V_vmux_QVV(hi, Q6_V_vsplat_R(1), Q6_V_vzero()));

    const HVX_Vector one = vsplatf(1.0f);
    const HVX_Vector sv = sf_mul(sf_sub(m, one), vrecip(sf_add(m, one)));
    const HVX_Vector s2 = to_qf32(sf_mul(sv, sv));

    /* s + s^3/3 + s^5/5 + s^7/7, Horner in s^2 */
    HVX_Vector p = to_qf32(vsplatf(1.0f / 7.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, s2), vsplatf(1.0f / 5.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, s2), vsplatf(1.0f / 3.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, s2), one);
    /* log(m) = 2*s*p  ->  log2(m) = 2*s*p * log2(e) */
    const HVX_Vector logm = sf_mul(sf_mul(sv, Q6_Vsf_equals_Vqf32(p)),
                                   vsplatf(2.0f * LOG2_E));
    return sf_add(Q6_Vsf_equals_Vw(e), logm);
}
static inline HVX_Vector vlog(HVX_Vector x) {
    return sf_mul(vlog2(x), vsplatf(LN2));
}

/* ---- atan, and the inverse circular family from it ------------------------
 * TWO reductions, because one is not enough. Measured with only the |t| <= 1
 * reduction and a degree-9 fit: max_rel 5.0e-3 at t = 0.9, five times the budget,
 * and asin/acos inherited it at 8.6e-3.
 *
 *     |x| > 1            ->  atan(x) = pi/2 - atan(1/x)        gives |t| <= 1
 *     t > tan(pi/8)      ->  atan(t) = pi/4 + atan((t-1)/(t+1)) gives |t| <= 0.4143
 *
 * On that interval the plain odd Taylor series through t^11 has a next term of
 * t^13/13 <= 5.6e-7 -- no minimax fit needed, and the coefficients are exact
 * reciprocals rather than transcribed constants, which is one less thing to get
 * wrong. */
#define PI_2      1.57079632679489662f
#define PI_4      0.78539816339744831f
#define TAN_PI_8  0.41421356237309505f

static inline HVX_Vector vatan(HVX_Vector x) {
    const HVX_Vector one = vsplatf(1.0f);
    const HVX_Vector ax = sf_abs(x);
    /* first reduction: fold |x| > 1 through the reciprocal */
    const HVX_VectorPred big = Q6_Q_vcmp_gt_VsfVsf(ax, one);
    HVX_Vector t = Q6_V_vmux_QVV(big, vrecip(ax), ax);
    /* second reduction: fold t > tan(pi/8) through the tangent-difference form */
    const HVX_VectorPred oct = Q6_Q_vcmp_gt_VsfVsf(t, vsplatf(TAN_PI_8));
    t = Q6_V_vmux_QVV(oct, sf_mul(sf_sub(t, one), vrecip(sf_add(t, one))), t);

    const HVX_Vector t2 = to_qf32(sf_mul(t, t));
    HVX_Vector p = to_qf32(vsplatf(-1.0f / 11.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, t2), vsplatf(1.0f / 9.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, t2), vsplatf(-1.0f / 7.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, t2), vsplatf(1.0f / 5.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, t2), vsplatf(-1.0f / 3.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, t2), one);
    HVX_Vector r = sf_mul(t, Q6_Vsf_equals_Vqf32(p));

    /* undo the reductions, innermost first */
    r = Q6_V_vmux_QVV(oct, sf_add(r, vsplatf(PI_4)), r);
    r = Q6_V_vmux_QVV(big, sf_sub(vsplatf(PI_2), r), r);
    /* atan is odd */
    return Q6_V_vor_VV(r, Q6_V_vand_VV(x, Q6_V_vsplat_R(SGN_MASK)));
}

/* asin(x) = atan(x / sqrt(1 - x^2)), acos(x) = pi/2 - asin(x).
 * Both are used only on |x| <= 0.9 here (the task's stated range), so the
 * sqrt argument stays comfortably away from zero and no endpoint special case is
 * needed -- which IS a range assumption and is stated rather than hidden. */
static inline HVX_Vector vasin(HVX_Vector x) {
    const HVX_Vector one = vsplatf(1.0f);
    return vatan(sf_mul(x, vrsqrt(sf_sub(one, sf_mul(x, x)))));
}
static inline HVX_Vector vacos(HVX_Vector x) {
    return sf_sub(vsplatf(PI_2), vasin(x));
}
/* asinh(x) = log(x + sqrt(x^2+1)) -- stable for the signed range used here.
 * acosh(x) = log(x + sqrt(x^2-1)), valid for x >= 1 (the task's range).
 * atanh(x) = 0.5*log((1+x)/(1-x)), valid for |x| < 1. */
static inline HVX_Vector vasinh(HVX_Vector x) {
    const HVX_Vector one = vsplatf(1.0f);
    return vlog(sf_add(x, vsqrt(sf_add(sf_mul(x, x), one))));
}
static inline HVX_Vector vacosh(HVX_Vector x) {
    const HVX_Vector one = vsplatf(1.0f);
    return vlog(sf_add(x, vsqrt(sf_sub(sf_mul(x, x), one))));
}
static inline HVX_Vector vatanh(HVX_Vector x) {
    const HVX_Vector one = vsplatf(1.0f), half = vsplatf(0.5f);
    return sf_mul(half, vlog(sf_mul(sf_add(one, x), vrecip(sf_sub(one, x)))));
}

/* cos/sin: reduce by 2/pi to a quadrant index and a fraction, then one of two
 * degree-6/7 polynomials. Inputs here are bounded by ~2.5, so the reduction is a
 * single subtraction and no Payne-Hanek machinery is warranted. */
#define TWO_OVER_PI 0.63661977236758134f
#define S1 -0.16666667f
#define S2  0.00833333f
#define S3 -0.00019841f
#define C1 -0.5f
#define C2  0.04166666f
#define C3 -0.00138889f
#define C4  0.00002480f

static inline HVX_Vector vcos_poly(HVX_Vector z2) {
    HVX_Vector p = to_qf32(vsplatf(C4));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, z2), vsplatf(C3));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, z2), vsplatf(C2));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, z2), vsplatf(C1));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, z2), vsplatf(1.0f));
    return Q6_Vsf_equals_Vqf32(p);
}
static inline HVX_Vector vsin_poly(HVX_Vector z, HVX_Vector z2) {
    HVX_Vector p = to_qf32(vsplatf(S3));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, z2), vsplatf(S2));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, z2), vsplatf(S1));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, z2), vsplatf(1.0f));
    return sf_mul(z, Q6_Vsf_equals_Vqf32(p));
}
/* cos(x) with a quadrant reduction: q = round(x * 2/pi), z = x - q*pi/2. */
#define PI_2_HI 1.57079632679489662f
static inline HVX_Vector vcos(HVX_Vector x) {
    const HVX_Vector half = vsplatf(0.5f);
    const HVX_Vector ax = sf_abs(x);          /* cos is even */
    const HVX_Vector qf = sf_add(sf_mul(ax, vsplatf(TWO_OVER_PI)), half);
    const HVX_Vector q = Q6_Vw_equals_Vsf(qf);
    const HVX_Vector z = sf_sub(ax, sf_mul(Q6_Vsf_equals_Vw(q),
                                           vsplatf(PI_2_HI)));
    const HVX_Vector z2 = to_qf32(sf_mul(z, z));
    const HVX_Vector cc = vcos_poly(z2);
    const HVX_Vector ss = vsin_poly(z, z2);
    /* quadrant: 0 -> cos, 1 -> -sin, 2 -> -cos, 3 -> sin */
    const HVX_Vector q3 = Q6_V_vand_VV(q, Q6_V_vsplat_R(3));
    const HVX_VectorPred odd = Q6_Q_vcmp_eq_VwVw(
        Q6_V_vand_VV(q3, Q6_V_vsplat_R(1)), Q6_V_vsplat_R(1));
    HVX_Vector r = Q6_V_vmux_QVV(odd, ss, cc);
    const HVX_VectorPred neg = Q6_Q_vcmp_eq_VwVw(
        Q6_V_vand_VV(Q6_Vw_vadd_VwVw(q3, Q6_V_vsplat_R(1)),
                     Q6_V_vsplat_R(2)), Q6_V_vsplat_R(2));
    return Q6_V_vmux_QVV(neg, Q6_V_vxor_VV(r, Q6_V_vsplat_R(SGN_MASK)), r);
}


/* ---- erf, and erfc/gelu from it -------------------------------------------
 * A plain ODD TAYLOR series, which is the right choice here and would not be for a
 * general-purpose erf:
 *
 *     erf(x) = 2/sqrt(pi) * sum_k (-1)^k x^(2k+1) / (k! (2k+1))
 *
 * Twelve terms give 2.0e-6 absolute on |x| <= 1.5 (checked numerically before
 * being written), and every argument in this corpus is inside that: `erfc` sees
 * the raw draw [0.5, 1.5) and `gelu` sees x/sqrt(2) <= 1.06. The usual
 * Abramowitz-Stegun 7.1.26 rational is slightly more accurate (1.4e-7) and needs an
 * `exp` -- more work for accuracy nothing here consumes. The series diverges in
 * practice above |x| ~ 3, so this is a RANGE-SPECIFIC implementation and says so.
 *
 * Coefficients are 2/sqrt(pi) * (-1)^k / (k! (2k+1)), exact rationals times one
 * constant rather than transcribed decimals. */
#define ERF_C 1.12837916709551257f      /* 2/sqrt(pi) */

static inline HVX_Vector verf(HVX_Vector x) {
    const HVX_Vector x2 = to_qf32(sf_mul(x, x));
    /* Horner from the highest term down, k = 11 .. 0 */
    static const float c[12] = {
        1.0f,            -1.0f/3.0f,       1.0f/10.0f,      -1.0f/42.0f,
        1.0f/216.0f,     -1.0f/1320.0f,    1.0f/9360.0f,    -1.0f/75600.0f,
        1.0f/685440.0f,  -1.0f/6894720.0f, 1.0f/76204800.0f,-1.0f/918086400.0f,
    };
    HVX_Vector p = to_qf32(vsplatf(c[11]));
    for (int k = 10; k >= 0; k--)
        p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, x2), vsplatf(c[k]));
    return sf_mul(sf_mul(x, vsplatf(ERF_C)), Q6_Vsf_equals_Vqf32(p));
}
static inline HVX_Vector verfc(HVX_Vector x) {
    return sf_sub(vsplatf(1.0f), verf(x));
}
/* gelu, exact form (not the tanh approximation): 0.5*x*(1 + erf(x/sqrt2)). */
#define INV_SQRT2 0.70710678118654752f
static inline HVX_Vector vgelu(HVX_Vector x) {
    const HVX_Vector e = verf(sf_mul(x, vsplatf(INV_SQRT2)));
    return sf_mul(sf_mul(x, vsplatf(0.5f)), sf_add(vsplatf(1.0f), e));
}

/* ---- lgamma, on [0.5, 1.5) only -------------------------------------------
 * The Taylor series of log(Gamma(1+u)) about u = 0:
 *
 *     -gamma*u + sum_{k>=2} (-1)^k zeta(k) u^k / k
 *
 * Sixteen terms give 1.5e-6 absolute for |u| <= 0.5 (checked numerically: 8 terms
 * is 4.0e-4 and would NOT clear the harness's 1e-4 absolute term). That covers the
 * task's whole input range and nothing else -- a general lgamma needs Stirling
 * above and the reflection formula below, and neither is written here because
 * neither is reachable by this task. The range is part of the task.
 *
 * lgamma(1) is exactly 0 and the series gives exactly 0 at u = 0, so the point
 * where a relative tolerance would be meaningless is also the point where the
 * approximation is exact. */
static inline HVX_Vector vlgamma_near1(HVX_Vector x) {
    static const float z[16] = {
        -0.5772156649f,  0.8224645334f, -0.4006856240f,  0.2705808084f,
        -0.2073855510f,  0.1695571770f, -0.1440498968f,  0.1255096695f,
        -0.1113342659f,  0.1000994575f, -0.0909540171f,  0.0833538405f,
        -0.0769325164f,  0.0714329463f, -0.0666687059f,  0.0625009551f,
    };
    const HVX_Vector u = to_qf32(sf_sub(x, vsplatf(1.0f)));
    HVX_Vector p = to_qf32(vsplatf(z[15]));
    for (int k = 14; k >= 0; k--)
        p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, u), vsplatf(z[k]));
    /* the series is u * (z1 + z2 u + ...) */
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(p, u));
}

/* trunc / floor / ceil via the truncating convert. Exact for |x| < 2^31, which
 * every input here satisfies; outside that the convert saturates and the value is
 * already an integer, so passing it through unchanged is correct. */
static inline HVX_Vector vtrunc(HVX_Vector x) {
    const HVX_Vector t = Q6_Vsf_equals_Vw(Q6_Vw_equals_Vsf(x));
    const HVX_VectorPred big = Q6_Q_vcmp_gt_VsfVsf(sf_abs(x), vsplatf(2147483520.0f));
    return Q6_V_vmux_QVV(big, x, t);
}
static inline HVX_Vector vfloor(HVX_Vector x) {
    const HVX_Vector t = vtrunc(x);
    const HVX_VectorPred lt = Q6_Q_vcmp_gt_VsfVsf(t, x);   /* trunc went up: x < 0 */
    return Q6_V_vmux_QVV(lt, sf_sub(t, vsplatf(1.0f)), t);
}
static inline HVX_Vector vceil(HVX_Vector x) {
    const HVX_Vector t = vtrunc(x);
    const HVX_VectorPred gt = Q6_Q_vcmp_gt_VsfVsf(x, t);   /* trunc went down */
    return Q6_V_vmux_QVV(gt, sf_add(t, vsplatf(1.0f)), t);
}

/* ---- sin, and tan from it --------------------------------------------------
 * `vcos` already reduces by 2/pi to a quadrant index and picks one of two short
 * polynomials. `sin` is the same machinery with the quadrant table rotated by one,
 * so it reuses `vcos_poly`/`vsin_poly` and fits nothing new.
 *
 * sin is ODD, so the argument's sign is carried out and reapplied; cos is even and
 * needs no such step. With q = round(|x| * 2/pi):
 *   q mod 4 = 0 -> +sin(z)   1 -> +cos(z)   2 -> -sin(z)   3 -> -cos(z)
 */
static inline HVX_Vector vsin(HVX_Vector x) {
    const HVX_Vector half = vsplatf(0.5f);
    const HVX_Vector sgn = Q6_V_vand_VV(x, Q6_V_vsplat_R(SGN_MASK));
    const HVX_Vector ax = sf_abs(x);
    const HVX_Vector qf = sf_add(sf_mul(ax, vsplatf(TWO_OVER_PI)), half);
    const HVX_Vector q = Q6_Vw_equals_Vsf(qf);
    const HVX_Vector z = sf_sub(ax, sf_mul(Q6_Vsf_equals_Vw(q), vsplatf(PI_2_HI)));
    const HVX_Vector z2 = to_qf32(sf_mul(z, z));
    const HVX_Vector cc = vcos_poly(z2);
    const HVX_Vector ss = vsin_poly(z, z2);

    const HVX_Vector q3 = Q6_V_vand_VV(q, Q6_V_vsplat_R(3));
    /* odd quadrant -> cos, even -> sin: the opposite of vcos's table */
    const HVX_VectorPred odd = Q6_Q_vcmp_eq_VwVw(
        Q6_V_vand_VV(q3, Q6_V_vsplat_R(1)), Q6_V_vsplat_R(1));
    HVX_Vector r = Q6_V_vmux_QVV(odd, cc, ss);
    /* negate for q mod 4 in {2, 3} */
    const HVX_VectorPred neg = Q6_Q_vcmp_gt_VwVw(q3, Q6_V_vsplat_R(1));
    r = Q6_V_vmux_QVV(neg, Q6_V_vxor_VV(r, Q6_V_vsplat_R(SGN_MASK)), r);
    /* sin(-x) = -sin(x) */
    return Q6_V_vxor_VV(r, sgn);
}

/* tan = sin/cos, one reciprocal. This is the honest form for THIS task and not a
 * general tan: a dedicated polynomial would need its own reduction against the pole
 * at pi/2, and near that pole the reciprocal amplifies cos's error without bound.
 * The draw is [0.5, 1.5) and tan(1.5) = 14.1, comfortably finite. */
static inline HVX_Vector vtan(HVX_Vector x) {
    return sf_mul(vsin(x), vrecip(vcos(x)));
}

/* ---- log1p -----------------------------------------------------------------
 * NOT `vlog(1 + x)`. For small x the addition discards x's significant bits before
 * the log ever sees them, which is the entire reason log1p exists separately.
 *
 * `vlog2`'s core is already an atanh series in s = (m-1)/(m+1), and the same
 * substitution applies to 1+x WITHOUT forming it:
 *     log(1+x) = 2 * atanh(s),   s = x / (2 + x)
 * so the cancellation never happens. |s| <= 1/3 for x in [0, 1], and the odd series
 * through s^7 has a next term below 1.4e-8.
 *
 * THE SWITCH IS AT |x| = 0.5, NOT 1, AND THE DIFFERENCE IS 3.1e-1. With the
 * threshold at 1 this took the series path for x down to -0.95, where
 * s = -0.95/1.05 = -0.905 and a 4-term odd series is nowhere near converged: the next
 * term alone is s^9/9 = 0.048, and measured error was 3.08e-1 ABSOLUTE -- three
 * thousand times the budget. It was caught by `binary_cross_entropy`, whose
 * log(1-p) = log1p(-p) reaches -0.95 exactly, and the earlier accuracy check had only
 * covered [-0.5, 2.5] where the series is fine.
 *
 * At |x| <= 0.5 we have |s| <= 1/3 and the first dropped term is s^9/9 < 2e-6.
 * Outside it, `vlog(1+x)` is correct and does not cancel: for x in [-1, -0.5] the sum
 * 1+x is EXACT by Sterbenz (both operands within a factor of two), and for x above 0.5
 * it is large enough that the rounding is irrelevant. The one range where forming 1+x
 * genuinely loses information is x NEAR ZERO -- which is precisely where the series is
 * best -- so the two arms cover each other's weakness rather than overlapping.
 *
 * The lesson is the one this corpus keeps relearning: a primitive is only verified over
 * the range it was MEASURED on, and extending a caller's domain re-opens the question.
 */
static inline HVX_Vector vlog1p(HVX_Vector x) {
    const HVX_Vector one = vsplatf(1.0f), two = vsplatf(2.0f);
    const HVX_Vector sv = sf_mul(x, vrecip(sf_add(two, x)));
    const HVX_Vector s2 = to_qf32(sf_mul(sv, sv));
    HVX_Vector p = to_qf32(vsplatf(1.0f / 7.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, s2), vsplatf(1.0f / 5.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, s2), vsplatf(1.0f / 3.0f));
    p = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(p, s2), one);
    const HVX_Vector small = sf_mul(sf_mul(sv, Q6_Vsf_equals_Vqf32(p)), two);
    const HVX_Vector big = vlog(sf_add(one, x));
    const HVX_VectorPred large = Q6_Q_vcmp_gt_VsfVsf(sf_abs(x), vsplatf(0.5f));
    return Q6_V_vmux_QVV(large, big, small);
}

/* ---- tanh, and sigmoid from the same exponential ---------------------------
 * tanh(x) = 1 - 2/(e^{2x} + 1): ONE exponential instead of the two in
 * (e^x - e^-x)/(e^x + e^-x), and no subtraction of nearly equal terms.
 *
 * Evaluated on |x| and re-signed because tanh is odd. That also bounds the
 * exponential: `vexp2` clamps its assembled exponent field at both ends, so for
 * large |x| the result saturates to exactly +-1, which is the correct fp32 answer
 * rather than an overflow. */
static inline HVX_Vector vtanh(HVX_Vector x) {
    const HVX_Vector one = vsplatf(1.0f), two = vsplatf(2.0f);
    const HVX_Vector sgn = Q6_V_vand_VV(x, Q6_V_vsplat_R(SGN_MASK));
    const HVX_Vector ax = sf_abs(x);
    const HVX_Vector e = vexp(sf_mul(two, ax));
    const HVX_Vector r = sf_sub(one, sf_mul(two, vrecip(sf_add(e, one))));
    return Q6_V_vxor_VV(r, sgn);
}

/* sigmoid / expit: 1/(1 + e^-x). Same single-exponential shape as tanh, and
 * `vexp2`'s exponent clamp bounds both tails the same way. */
static inline HVX_Vector vsigmoid(HVX_Vector x) {
    const HVX_Vector one = vsplatf(1.0f);
    const HVX_Vector e = vexp(Q6_V_vxor_VV(x, Q6_V_vsplat_R(SGN_MASK)));
    return vrecip(sf_add(one, e));
}

/* sinh: (e^x - e^-x)/2, counterpart to the cosh already in the core; the negation
 * is a sign-bit xor, exact and one instruction. This DOES cancel as x approaches 0
 * -- both terms approach 1 -- so it is right for this task's [0.5, 1.5) draw and a
 * general sinh would need an expm1-based form near zero. Stated, not hidden. */
static inline HVX_Vector vsinh(HVX_Vector x) {
    return sf_mul(vsplatf(0.5f),
                  sf_sub(vexp(x), vexp(Q6_V_vxor_VV(x, Q6_V_vsplat_R(SGN_MASK)))));
}

/* softplus: log1p(e^x), through log1p rather than log(1 + e^x) so the small-x end
 * keeps its precision. torch's softplus switches to the linear regime above
 * threshold 20; the draw here reaches 1.5, so it is never taken and the reference
 * does not take it either. */
static inline HVX_Vector vsoftplus(HVX_Vector x) {
    return vlog1p(vexp(x));
}

/* sign / sgn: -1, 0, +1 from two compares and a select. No float arithmetic, so
 * exact. Both zeros map to zero, because the test is on the stripped magnitude. */
static inline HVX_Vector vsign(HVX_Vector x) {
    const HVX_Vector one = vsplatf(1.0f);
    const HVX_VectorPred zero = Q6_Q_vcmp_eq_VwVw(sf_abs(x), Q6_V_vzero());
    const HVX_Vector signed_one =
        Q6_V_vor_VV(one, Q6_V_vand_VV(x, Q6_V_vsplat_R(SGN_MASK)));
    return Q6_V_vmux_QVV(zero, Q6_V_vzero(), signed_one);
}

/* nextafter(x, y): the adjacent representable float, computed ENTIRELY ON THE
 * INTEGER BIT PATTERN -- no float arithmetic anywhere, so it is exact by
 * construction rather than to a tolerance.
 *
 * IEEE-754 orders same-sign floats exactly as their bit patterns order as integers,
 * so a step is +-1 on those bits. The cases:
 *   x == y             -> y (no step at all)
 *   x == 0             -> the smallest subnormal, signed like y
 *   magnitude grows    -> bits + 1
 *   magnitude shrinks  -> bits - 1
 * Whether the magnitude grows is NOT just "is y above x": for negative x, moving up
 * the real line makes the magnitude smaller. So it is (y > x) XOR (x is negative),
 * which is the one part of this worth writing down. */
static inline HVX_Vector vnextafter(HVX_Vector x, HVX_Vector y) {
    const HVX_Vector smask = Q6_V_vsplat_R(SGN_MASK);
    const HVX_Vector one_i = Q6_V_vsplat_R(1);
    const HVX_VectorPred eq = Q6_Q_vcmp_eq_VwVw(x, y);
    const HVX_VectorPred xzero = Q6_Q_vcmp_eq_VwVw(sf_abs(x), Q6_V_vzero());

    const HVX_VectorPred up = Q6_Q_vcmp_gt_VsfVsf(y, x);
    const HVX_VectorPred xneg = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(x, smask), smask);
    const HVX_Vector up_v = Q6_V_vmux_QVV(up, one_i, Q6_V_vzero());
    const HVX_Vector neg_v = Q6_V_vmux_QVV(xneg, one_i, Q6_V_vzero());
    const HVX_VectorPred grow = Q6_Q_vcmp_eq_VwVw(
        Q6_V_vxor_VV(up_v, neg_v), one_i);

    const HVX_Vector stepped = Q6_V_vmux_QVV(
        grow, Q6_Vw_vadd_VwVw(x, one_i), Q6_Vw_vsub_VwVw(x, one_i));
    const HVX_Vector from_zero = Q6_V_vor_VV(one_i, Q6_V_vand_VV(y, smask));
    return Q6_V_vmux_QVV(eq, y, Q6_V_vmux_QVV(xzero, from_zero, stepped));
}

#define NELEM 512
#define NV    (NELEM / 32)

extern "C" void candidate_kernel(const int32_t *v_args_0, const int32_t *v_args_1, int32_t *out0) {

    const HVX_Vector *xv = (const HVX_Vector *)v_args_0;
    const HVX_Vector *yv = (const HVX_Vector *)v_args_1;
    HVX_Vector *ov = (HVX_Vector *)out0;

    for (int k = 0; k < NV; k++) {
        const HVX_Vector x = xv[k];
        const HVX_Vector y = yv[k];
        ov[k] = Q6_Vw_vasl_VwVw(x, y);
    }
}
