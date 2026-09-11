#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>

/* ---- everything this harness needs, emitted rather than included ----
 * Self-contained on purpose: no header from hexbench/env/harness/ is on
 * this compile's include path, so the R&D material cannot reach a forge v2
 * translation unit even by accident. See oracle.harness_c's docstring. */

/* Enable the HMX extension context: SSR.XE (bit 29) + SSR.XA=2 (bits
 * 27:25). The standalone runtime enables HVX only, so without this an
 * `mxmem` access raises exception 0x18. The program runs privileged. */
static inline void forge2_hmx_enable(void) {
    unsigned m = (1u << 29) | (2u << 25);  /* 0x24000000 */
    unsigned t;
    __asm__ volatile("%0=ssr\n\t %0=or(%0,%1)\n\t ssr=%0\n\t isync\n\t"
                     : "=&r"(t) : "r"(m));
}

/* Tolerance compare for FLOATING-POINT results. HVX float arithmetic
 * goes through the non-IEEE qfloat path and reductions reorder, so a
 * correct vectorised kernel is not bit-identical to the IEEE scalar
 * golden. Each dtype is judged at ~1-2 ULP OF ITSELF; integer results
 * stay exact and get no tolerance at all. */
static inline int forge2_close_f32(float g, float e) {
    float d = g - e; if (d < 0) d = -d;
    float ae = e < 0 ? -e : e;
    return d <= 1e-4f + 1e-3f * ae;
}

extern "C" void candidate_kernel(const float* in0, float* out0);

/* base64 -> bytes, for the golden buffers emitted as a byte image.
 * See oracle.B64_THRESHOLD for why they are not element literals. */
static int forge2_b64_val(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;                 /* '=' padding, or anything unexpected */
}
static void forge2_b64(const char *src, void *dst, unsigned long nbytes) {
    unsigned char *out = (unsigned char *)dst;
    unsigned long w = 0;
    unsigned acc = 0;
    int have = 0;
    for (const char *p = src; *p; p++) {
        int v = forge2_b64_val(*p);
        if (v < 0) continue;
        acc = (acc << 6) | (unsigned)v;
        have += 6;
        if (have >= 8) {
            have -= 8;
            if (w < nbytes) out[w++] = (unsigned char)((acc >> have) & 0xFF);
        }
    }
    /* Fail loudly rather than run against a half-filled buffer: a short
     * decode would look exactly like a wrong kernel. */
    if (w != nbytes) {
        printf("HVXENV_INCORRECT errors=-1 n=%lu first_bad=-1 (b64 decoded %lu of %lu bytes)\n", nbytes, w, nbytes);
        exit(1);
    }
}

static float golden_in_0[6144] __attribute__((aligned(128)));
static const char b64_in_0[] =
    "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"
    "UT/+ik0/"
    ;

static const float golden_out_0[1128] __attribute__((aligned(128))) = { 0x1.240cea0000000p+2f, 0x1.221d2e0000000p+2f, 0x1.3688fc0000000p+2f, 0x1.195d2a0000000p+2f, 0x1.1c11600000000p+2f, 0x1.1792ca0000000p+2f, 0x1.13a3c20000000p+2f, 0x1.2d4bc40000000p+2f, 0x1.257e7a0000000p+2f, 0x1.1246c00000000p+2f, 0x1.3567e40000000p+2f, 0x1.1ece260000000p+2f, 0x1.11d25e0000000p+2f, 0x1.12b5f60000000p+2f, 0x1.3a604a0000000p+2f, 0x1.3a6ea40000000p+2f, 0x1.252b240000000p+2f, 0x1.3078cc0000000p+2f, 0x1.25dba40000000p+2f, 0x1.088eea0000000p+2f, 0x1.2ea7d60000000p+2f, 0x1.fa30420000000p+1f, 0x1.1cad020000000p+2f, 0x1.f144300000000p+1f, 0x1.3142600000000p+2f, 0x1.2c44a60000000p+2f, 0x1.2018ee0000000p+2f, 0x1.12ecc00000000p+2f, 0x1.0541220000000p+2f, 0x1.2a9d400000000p+2f, 0x1.113efa0000000p+2f, 0x1.3d11a00000000p+2f, 0x1.3196ba0000000p+2f, 0x1.1a37340000000p+2f, 0x1.1a50220000000p+2f, 0x1.1a46640000000p+2f, 0x1.f64ce60000000p+1f, 0x1.2c9e360000000p+2f, 0x1.282a960000000p+2f, 0x1.2307a20000000p+2f, 0x1.33359e0000000p+2f, 0x1.286f640000000p+2f, 0x1.296f740000000p+2f, 0x1.12e3040000000p+2f, 0x1.16f9ea0000000p+2f, 0x1.1341060000000p+2f, 0x1.020aa80000000p+2f, 0x1.45d0600000000p+2f, 0x1.2d6c7e0000000p+2f, 0x1.357afe0000000p+2f, 0x1.21e4b00000000p+2f, 0x1.3ce7e20000000p+2f, 0x1.213d640000000p+2f, 0x1.221aa20000000p+2f, 0x1.2b92be0000000p+2f, 0x1.187d920000000p+2f, 0x1.2639600000000p+2f, 0x1.16c7760000000p+2f, 0x1.2b05160000000p+2f, 0x1.2501720000000p+2f, 0x1.2b37280000000p+2f, 0x1.2dac500000000p+2f, 0x1.1714300000000p+2f, 0x1.3ad2180000000p+2f, 0x1.1182780000000p+2f, 0x1.2256ca0000000p+2f, 0x1.20cd1a0000000p+2f, 0x1.24b7000000000p+2f, 0x1.4164dc0000000p+2f, 0x1.47ab800000000p+2f, 0x1.31de9a0000000p+2f, 0x1.1c9a020000000p+2f, 0x1.19b6420000000p+2f, 0x1.43d3c80000000p+2f, 0x1.37a9f60000000p+2f, 0x1.d8d5b60000000p+1f, 0x1.2847f80000000p+2f, 0x1.3f1b300000000p+2f, 0x1.1cf44e0000000p+2f, 0x1.1e80700000000p+2f, 0x1.0da6860000000p+2f, 0x1.2240a40000000p+2f, 0x1.2532700000000p+2f, 0x1.3465160000000p+2f, 0x1.3885fe0000000p+2f, 0x1.2d0a4e0000000p+2f, 0x1.25433c0000000p+2f, 0x1.3e1bf20000000p+2f, 0x1.3263200000000p+2f, 0x1.3729980000000p+2f, 0x1.16be040000000p+2f, 0x1.33f5b20000000p+2f, 0x1.0ec7f80000000p+2f, 0x1.3838860000000p+2f, 0x1.2eb31e0000000p+2f, 0x1.3e93860000000p+2f, 0x1.2e1cf80000000p+2f, 0x1.285a6e0000000p+2f, 0x1.402eb40000000p+2f, 0x1.20df3e0000000p+2f, 0x1.1eb6de0000000p+2f, 0x1.3bda9a0000000p+2f, 0x1.1ab4d40000000p+2f, 0x1.2f5a4c0000000p+2f, 0x1.1606a20000000p+2f, 0x1.3d00300000000p+2f, 0x1.3c6c260000000p+2f, 0x1.37b4ee0000000p+2f, 0x1.3644620000000p+2f, 0x1.3d06f80000000p+2f, 0x1.341e000000000p+2f, 0x1.30f5c20000000p+2f, 0x1.2b224c0000000p+2f, 0x1.3690700000000p+2f, 0x1.31f7ba0000000p+2f, 0x1.1d54b20000000p+2f, 0x1.39043e0000000p+2f, 0x1.1024b20000000p+2f, 0x1.20af8c0000000p+2f, 0x1.1af5f80000000p+2f, 0x1.33f7e40000000p+2f, 0x1.07fe360000000p+2f, 0x1.3a9c860000000p+2f, 0x1.37786a0000000p+2f, 0x1.175e480000000p+2f, 0x1.3b837a0000000p+2f, 0x1.443ffc0000000p+2f, 0x1.3c6a820000000p+2f, 0x1.25ab080000000p+2f, 0x1.1a5de00000000p+2f, 0x1.23eee40000000p+2f, 0x1.1b11d60000000p+2f, 0x1.0c5e160000000p+2f, 0x1.1f86060000000p+2f, 0x1.409b6e0000000p+2f, 0x1.1ac9ba0000000p+2f, 0x1.1a6e120000000p+2f, 0x1.28e9a20000000p+2f, 0x1.2c2b120000000p+2f, 0x1.2ead2e0000000p+2f, 0x1.381bc40000000p+2f, 0x1.20e43a0000000p+2f, 0x1.3d72540000000p+2f, 0x1.2563200000000p+2f, 0x1.35eac20000000p+2f, 0x1.2fe9f60000000p+2f, 0x1.2864f20000000p+2f, 0x1.2daa7c0000000p+2f, 0x1.422b860000000p+2f, 0x1.212f720000000p+2f, 0x1.30f0c40000000p+2f, 0x1.20f4d40000000p+2f, 0x1.3a95ac0000000p+2f, 0x1.2b71b60000000p+2f, 0x1.32ecb20000000p+2f, 0x1.3b175a0000000p+2f, 0x1.11fb960000000p+2f, 0x1.33c8c80000000p+2f, 0x1.269d820000000p+2f, 0x1.1896340000000p+2f, 0x1.3f60aa0000000p+2f, 0x1.0c99e60000000p+2f, 0x1.236b880000000p+2f, 0x1.281de00000000p+2f, 0x1.26c6f00000000p+2f, 0x1.3a7b9c0000000p+2f, 0x1.31cee20000000p+2f, 0x1.2881760000000p+2f, 0x1.17d4d00000000p+2f, 0x1.389bbe0000000p+2f, 0x1.3e66380000000p+2f, 0x1.21a9960000000p+2f, 0x1.3b80fc0000000p+2f, 0x1.2e347e0000000p+2f, 0x1.1415480000000p+2f, 0x1.246d080000000p+2f, 0x1.35a4560000000p+2f, 0x1.2c17dc0000000p+2f, 0x1.368db80000000p+2f, 0x1.28cccc0000000p+2f, 0x1.40c62a0000000p+2f, 0x1.20bbaa0000000p+2f, 0x1.23c14e0000000p+2f, 0x1.2ed0620000000p+2f, 0x1.1682d80000000p+2f, 0x1.2bc1400000000p+2f, 0x1.2cf0240000000p+2f, 0x1.144b040000000p+2f, 0x1.1888d60000000p+2f, 0x1.2c54820000000p+2f, 0x1.12d8a20000000p+2f, 0x1.1cd6800000000p+2f, 0x1.1699f60000000p+2f, 0x1.25e94a0000000p+2f, 0x1.2552820000000p+2f, 0x1.30a05e0000000p+2f, 0x1.30c2060000000p+2f, 0x1.197c160000000p+2f, 0x1.23f60e0000000p+2f, 0x1.15e0060000000p+2f, 0x1.22c5d40000000p+2f, 0x1.2dffda0000000p+2f, 0x1.25c2480000000p+2f, 0x1.0bc5440000000p+2f, 0x1.1a648c0000000p+2f, 0x1.1009fe0000000p+2f, 0x1.304be60000000p+2f, 0x1.3353c40000000p+2f, 0x1.314f5a0000000p+2f, 0x1.2bf8900000000p+2f, 0x1.3e8c340000000p+2f, 0x1.311d160000000p+2f, 0x1.252b5a0000000p+2f, 0x1.48a2b60000000p+2f, 0x1.1f9d1a0000000p+2f, 0x1.3d386a0000000p+2f, 0x1.2236de0000000p+2f, 0x1.30e90c0000000p+2f, 0x1.31e8700000000p+2f, 0x1.2e66ac0000000p+2f, 0x1.15e9860000000p+2f, 0x1.2818920000000p+2f, 0x1.23a4be0000000p+2f, 0x1.315f520000000p+2f, 0x1.25dd880000000p+2f, 0x1.3cda880000000p+2f, 0x1.1fe64e0000000p+2f, 0x1.446dca0000000p+2f, 0x1.260c340000000p+2f, 0x1.3a08220000000p+2f, 0x1.3247fa0000000p+2f, 0x1.15d8d60000000p+2f, 0x1.1b23260000000p+2f, 0x1.246fb40000000p+2f, 0x1.32d1720000000p+2f, 0x1.39b85e0000000p+2f, 0x1.3092a80000000p+2f, 0x1.30cab00000000p+2f, 0x1.269be40000000p+2f, 0x1.23c6a40000000p+2f, 0x1.2745340000000p+2f, 0x1.1ed6be0000000p+2f, 0x1.3858680000000p+2f, 0x1.3f5a700000000p+2f, 0x1.2ea6160000000p+2f, 0x1.1688640000000p+2f, 0x1.1e94800000000p+2f, 0x1.1f11b80000000p+2f, 0x1.1de1120000000p+2f, 0x1.1e81a40000000p+2f, 0x1.31731a0000000p+2f, 0x1.4aa7b00000000p+2f, 0x1.1fadc60000000p+2f, 0x1.0bf1580000000p+2f, 0x1.3e91440000000p+2f, 0x1.4e93ac0000000p+2f, 0x1.18ad580000000p+2f, 0x1.448d9e0000000p+2f, 0x1.27d1420000000p+2f, 0x1.2441980000000p+2f, 0x1.2630ae0000000p+2f, 0x1.2d34620000000p+2f, 0x1.3531580000000p+2f, 0x1.1d9f140000000p+2f, 0x1.10e12c0000000p+2f, 0x1.3995520000000p+2f, 0x1.2c7ef80000000p+2f, 0x1.204b1a0000000p+2f, 0x1.3a1f1c0000000p+2f, 0x1.306ef20000000p+2f, 0x1.1ef5520000000p+2f, 0x1.3cf1d80000000p+2f, 0x1.1d31980000000p+2f, 0x1.3eeed00000000p+2f, 0x1.191b600000000p+2f, 0x1.2711be0000000p+2f, 0x1.3ff20e0000000p+2f, 0x1.17ddc60000000p+2f, 0x1.34c6960000000p+2f, 0x1.24c1d60000000p+2f, 0x1.32756e0000000p+2f, 0x1.3ef7600000000p+2f, 0x1.2c73e80000000p+2f, 0x1.244c280000000p+2f, 0x1.2b88000000000p+2f, 0x1.3491160000000p+2f, 0x1.246b280000000p+2f, 0x1.26d0320000000p+2f, 0x1.16beb00000000p+2f, 0x1.30d2760000000p+2f, 0x1.290d420000000p+2f, 0x1.38573a0000000p+2f, 0x1.0b248c0000000p+2f, 0x1.34ed6a0000000p+2f, 0x1.21a70c0000000p+2f, 0x1.1cdaf40000000p+2f, 0x1.2d0ce80000000p+2f, 0x1.1fdab60000000p+2f, 0x1.2933ae0000000p+2f, 0x1.347b600000000p+2f, 0x1.28fd040000000p+2f, 0x1.25900a0000000p+2f, 0x1.3364820000000p+2f, 0x1.193ac40000000p+2f, 0x1.5472200000000p+2f, 0x1.08bc000000000p+2f, 0x1.19f9360000000p+2f, 0x1.21899e0000000p+2f, 0x1.1b36ec0000000p+2f, 0x1.189e0c0000000p+2f, 0x1.1b2aa60000000p+2f, 0x1.00d30a0000000p+2f, 0x1.2221560000000p+2f, 0x1.2017b00000000p+2f, 0x1.204c140000000p+2f, 0x1.1eab2e0000000p+2f, 0x1.27d8b60000000p+2f, 0x1.1d82a80000000p+2f, 0x1.3849ac0000000p+2f, 0x1.2330200000000p+2f, 0x1.1902b00000000p+2f, 0x1.1dd54e0000000p+2f, 0x1.189df80000000p+2f, 0x1.1d29600000000p+2f, 0x1.17df340000000p+2f, 0x1.22294c0000000p+2f, 0x1.126e2c0000000p+2f, 0x1.158d600000000p+2f, 0x1.1e7fb20000000p+2f, 0x1.2126ac0000000p+2f, 0x1.185c7e0000000p+2f, 0x1.049b1c0000000p+2f, 0x1.3453120000000p+2f, 0x1.2c0a380000000p+2f, 0x1.1738b20000000p+2f, 0x1.29dd7e0000000p+2f, 0x1.09a66e0000000p+2f, 0x1.0a65ba0000000p+2f, 0x1.2d1f6a0000000p+2f, 0x1.3201a80000000p+2f, 0x1.24a21a0000000p+2f, 0x1.0cc7260000000p+2f, 0x1.27753a0000000p+2f, 0x1.12ad400000000p+2f, 0x1.2f65fe0000000p+2f, 0x1.204ebc0000000p+2f, 0x1.201c3a0000000p+2f, 0x1.1504aa0000000p+2f, 0x1.309bba0000000p+2f, 0x1.1059020000000p+2f, 0x1.344a3e0000000p+2f, 0x1.2024a80000000p+2f, 0x1.172d620000000p+2f, 0x1.27db6c0000000p+2f, 0x1.2194f80000000p+2f, 0x1.2e14be0000000p+2f, 0x1.187bea0000000p+2f, 0x1.1e41ce0000000p+2f, 0x1.3771520000000p+2f, 0x1.3372e20000000p+2f, 0x1.3a5d1c0000000p+2f, 0x1.15a9d40000000p+2f, 0x1.2eef560000000p+2f, 0x1.220f4a0000000p+2f, 0x1.37dd3e0000000p+2f, 0x1.1fb6c20000000p+2f, 0x1.27dd1c0000000p+2f, 0x1.48ab900000000p+2f, 0x1.1936740000000p+2f, 0x1.0536400000000p+2f, 0x1.0a7de20000000p+2f, 0x1.21e2680000000p+2f, 0x1.3b57ec0000000p+2f, 0x1.1141080000000p+2f, 0x1.1be12c0000000p+2f, 0x1.3c08b20000000p+2f, 0x1.29fb860000000p+2f, 0x1.36568c0000000p+2f, 0x1.1caea00000000p+2f, 0x1.204c2c0000000p+2f, 0x1.41360e0000000p+2f, 0x1.2666480000000p+2f, 0x1.29d5ba0000000p+2f, 0x1.1f00800000000p+2f, 0x1.2a4ef00000000p+2f, 0x1.33b0cc0000000p+2f, 0x1.204c1e0000000p+2f, 0x1.2dba2c0000000p+2f, 0x1.37f3040000000p+2f, 0x1.29a4c60000000p+2f, 0x1.2cb3080000000p+2f, 0x1.1e26be0000000p+2f, 0x1.27bffa0000000p+2f, 0x1.2ea7b40000000p+2f, 0x1.0fdece0000000p+2f, 0x1.2974160000000p+2f, 0x1.2932780000000p+2f, 0x1.2ee24e0000000p+2f, 0x1.29e8b60000000p+2f, 0x1.0d2dde0000000p+2f, 0x1.3a22dc0000000p+2f, 0x1.2302740000000p+2f, 0x1.1d346a0000000p+2f, 0x1.256ac20000000p+2f, 0x1.1fa9840000000p+2f, 0x1.1b7a020000000p+2f, 0x1.0f43f00000000p+2f, 0x1.2092960000000p+2f, 0x1.1741100000000p+2f, 0x1.3a65b20000000p+2f, 0x1.07d26a0000000p+2f, 0x1.14916e0000000p+2f, 0x1.18f9ac0000000p+2f, 0x1.2711fa0000000p+2f, 0x1.26d2f00000000p+2f, 0x1.2ef83a0000000p+2f, 0x1.1e64560000000p+2f, 0x1.16a6d40000000p+2f, 0x1.2308a40000000p+2f, 0x1.29b0160000000p+2f, 0x1.37b1100000000p+2f, 0x1.1e7e520000000p+2f, 0x1.2e3e4e0000000p+2f, 0x1.2579020000000p+2f, 0x1.20d6340000000p+2f, 0x1.2488ca0000000p+2f, 0x1.2f5bf00000000p+2f, 0x1.29da220000000p+2f, 0x1.1e09c60000000p+2f, 0x1.24ddb00000000p+2f, 0x1.195bac0000000p+2f, 0x1.1a42180000000p+2f, 0x1.28698e0000000p+2f, 0x1.2c0ff20000000p+2f, 0x1.0c81260000000p+2f, 0x1.3dffc00000000p+2f, 0x1.0c46740000000p+2f, 0x1.fea8e20000000p+1f, 0x1.22b10e0000000p+2f, 0x1.1bfca60000000p+2f, 0x1.0ae7760000000p+2f, 0x1.1d91920000000p+2f, 0x1.2d04420000000p+2f, 0x1.33b73a0000000p+2f, 0x1.0356780000000p+2f, 0x1.0ac0880000000p+2f, 0x1.258b2a0000000p+2f, 0x1.2d108c0000000p+2f, 0x1.1a513a0000000p+2f, 0x1.12ff080000000p+2f, 0x1.1f7a140000000p+2f, 0x1.14d3e20000000p+2f, 0x1.15b56a0000000p+2f, 0x1.1f2a6e0000000p+2f, 0x1.19b84c0000000p+2f, 0x1.2502e60000000p+2f, 0x1.22f9c20000000p+2f, 0x1.0fcc7e0000000p+2f, 0x1.2571980000000p+2f, 0x1.22d0d80000000p+2f, 0x1.1b8a3a0000000p+2f, 0x1.1975c00000000p+2f, 0x1.1b20340000000p+2f, 0x1.0741940000000p+2f, 0x1.2caed80000000p+2f, 0x1.2b32700000000p+2f, 0x1.2c543c0000000p+2f, 0x1.2692d00000000p+2f, 0x1.2d3f780000000p+2f, 0x1.2632360000000p+2f, 0x1.297b100000000p+2f, 0x1.18d49c0000000p+2f, 0x1.35bc660000000p+2f, 0x1.3233440000000p+2f, 0x1.27c7d80000000p+2f, 0x1.23d9be0000000p+2f, 0x1.0fe7f60000000p+2f, 0x1.2c45060000000p+2f, 0x1.1c4e7c0000000p+2f, 0x1.3841660000000p+2f, 0x1.26946a0000000p+2f, 0x1.2eb8da0000000p+2f, 0x1.396d940000000p+2f, 0x1.3117020000000p+2f, 0x1.2e0a760000000p+2f, 0x1.2528560000000p+2f, 0x1.2aa0300000000p+2f, 0x1.232a460000000p+2f, 0x1.41f8280000000p+2f, 0x1.2e57780000000p+2f, 0x1.2acd480000000p+2f, 0x1.2185500000000p+2f, 0x1.29ebd60000000p+2f, 0x1.428b760000000p+2f, 0x1.392bfa0000000p+2f, 0x1.299b6c0000000p+2f, 0x1.253d8e0000000p+2f, 0x1.3074100000000p+2f, 0x1.152c9c0000000p+2f, 0x1.1da07e0000000p+2f, 0x1.24324e0000000p+2f, 0x1.369e780000000p+2f, 0x1.1894ea0000000p+2f, 0x1.306c7a0000000p+2f, 0x1.214f740000000p+2f, 0x1.2aa6300000000p+2f, 0x1.13f4080000000p+2f, 0x1.201d9e0000000p+2f, 0x1.1c1aac0000000p+2f, 0x1.1a3e540000000p+2f, 0x1.2b3d940000000p+2f, 0x1.24a48e0000000p+2f, 0x1.1a979a0000000p+2f, 0x1.27c7e80000000p+2f, 0x1.0f038a0000000p+2f, 0x1.1db36c0000000p+2f, 0x1.222edc0000000p+2f, 0x1.194c460000000p+2f, 0x1.2805d40000000p+2f, 0x1.0a64620000000p+2f, 0x1.1f48b00000000p+2f, 0x1.2f56820000000p+2f, 0x1.2a15560000000p+2f, 0x1.2ab7ac0000000p+2f, 0x1.2e47860000000p+2f, 0x1.13fdcc0000000p+2f, 0x1.1a5e500000000p+2f, 0x1.1719f60000000p+2f, 0x1.27ce000000000p+2f, 0x1.24eaa80000000p+2f, 0x1.16d2ee0000000p+2f, 0x1.30d8560000000p+2f, 0x1.083ac40000000p+2f, 0x1.0a89380000000p+2f, 0x1.2ac5080000000p+2f, 0x1.1d15a60000000p+2f, 0x1.2b79d80000000p+2f, 0x1.165b440000000p+2f, 0x1.2925de0000000p+2f, 0x1.15eeca0000000p+2f, 0x1.262d740000000p+2f, 0x1.15fece0000000p+2f, 0x1.2e1eba0000000p+2f, 0x1.1db5180000000p+2f, 0x1.2f41420000000p+2f, 0x1.0f3d720000000p+2f, 0x1.1e0a540000000p+2f, 0x1.1cb62c0000000p+2f, 0x1.26d7ae0000000p+2f, 0x1.2a86000000000p+2f, 0x1.1c73ac0000000p+2f, 0x1.386f5a0000000p+2f, 0x1.38516e0000000p+2f, 0x1.1435ce0000000p+2f, 0x1.2952e00000000p+2f, 0x1.33c7880000000p+2f, 0x1.22809c0000000p+2f, 0x1.2f31160000000p+2f, 0x1.2abb560000000p+2f, 0x1.2609fc0000000p+2f, 0x1.23361e0000000p+2f, 0x1.1308c40000000p+2f, 0x1.3b76960000000p+2f, 0x1.360d6e0000000p+2f, 0x1.33fd620000000p+2f, 0x1.1df9d20000000p+2f, 0x1.2682860000000p+2f, 0x1.3488160000000p+2f, 0x1.3bbad80000000p+2f, 0x1.18e70e0000000p+2f, 0x1.27ba3c0000000p+2f, 0x1.02d7380000000p+2f, 0x1.23b9e60000000p+2f, 0x1.1b7e800000000p+2f, 0x1.25748a0000000p+2f, 0x1.286f680000000p+2f, 0x1.3795440000000p+2f, 0x1.1eecf60000000p+2f, 0x1.16516a0000000p+2f, 0x1.1602aa0000000p+2f, 0x1.2ecefc0000000p+2f, 0x1.220a980000000p+2f, 0x1.26508c0000000p+2f, 0x1.150c560000000p+2f, 0x1.3e75fe0000000p+2f, 0x1.260a440000000p+2f, 0x1.208c060000000p+2f, 0x1.1289e20000000p+2f, 0x1.4961020000000p+2f, 0x1.2393ea0000000p+2f, 0x1.0e20ec0000000p+2f, 0x1.2bf43c0000000p+2f, 0x1.2351a00000000p+2f, 0x1.1dc5400000000p+2f, 0x1.1151fe0000000p+2f, 0x1.1bef980000000p+2f, 0x1.3796220000000p+2f, 0x1.22cd180000000p+2f, 0x1.241a9e0000000p+2f, 0x1.19254c0000000p+2f, 0x1.1d45be0000000p+2f, 0x1.192ada0000000p+2f, 0x1.23b9680000000p+2f, 0x1.41eade0000000p+2f, 0x1.2f44c20000000p+2f, 0x1.2ba56c0000000p+2f, 0x1.2c14580000000p+2f, 0x1.3f5dce0000000p+2f, 0x1.31cd520000000p+2f, 0x1.123b860000000p+2f, 0x1.fe697a0000000p+1f, 0x1.130fe00000000p+2f, 0x1.1d8b980000000p+2f, 0x1.1df90a0000000p+2f, 0x1.2132560000000p+2f, 0x1.37d24c0000000p+2f, 0x1.2220440000000p+2f, 0x1.2e65360000000p+2f, 0x1.0c24f80000000p+2f, 0x1.2ed5240000000p+2f, 0x1.317a840000000p+2f, 0x1.1c5bfa0000000p+2f, 0x1.24b4c40000000p+2f, 0x1.28ab900000000p+2f, 0x1.2124ae0000000p+2f, 0x1.22401e0000000p+2f, 0x1.1774720000000p+2f, 0x1.2010280000000p+2f, 0x1.2b924c0000000p+2f, 0x1.21e9200000000p+2f, 0x1.1c54900000000p+2f, 0x1.0897d00000000p+2f, 0x1.2960760000000p+2f, 0x1.39653e0000000p+2f, 0x1.3ee20c0000000p+2f, 0x1.29d04e0000000p+2f, 0x1.34271c0000000p+2f, 0x1.267c8e0000000p+2f, 0x1.3d9f600000000p+2f, 0x1.28da340000000p+2f, 0x1.3d2ff80000000p+2f, 0x1.1b82500000000p+2f, 0x1.0ad7120000000p+2f, 0x1.299e400000000p+2f, 0x1.379c640000000p+2f, 0x1.2a56520000000p+2f, 0x1.2e91ec0000000p+2f, 0x1.36b6fe0000000p+2f, 0x1.4429aa0000000p+2f, 0x1.2551ac0000000p+2f, 0x1.1e799a0000000p+2f, 0x1.2177500000000p+2f, 0x1.21a57e0000000p+2f, 0x1.37803a0000000p+2f, 0x1.3364fa0000000p+2f, 0x1.3255880000000p+2f, 0x1.214c240000000p+2f, 0x1.39ffa40000000p+2f, 0x1.2ad29c0000000p+2f, 0x1.3b30660000000p+2f, 0x1.3ffed80000000p+2f, 0x1.2e32ac0000000p+2f, 0x1.3879f60000000p+2f, 0x1.19d98e0000000p+2f, 0x1.0e0ba60000000p+2f, 0x1.2318600000000p+2f, 0x1.1d47a60000000p+2f, 0x1.1fb0360000000p+2f, 0x1.17a86e0000000p+2f, 0x1.1ce5b40000000p+2f, 0x1.26dd080000000p+2f, 0x1.1f28ca0000000p+2f, 0x1.0b767a0000000p+2f, 0x1.26bf480000000p+2f, 0x1.1a03920000000p+2f, 0x1.f7bc120000000p+1f, 0x1.131c1a0000000p+2f, 0x1.2ec75e0000000p+2f, 0x1.1f098c0000000p+2f, 0x1.11a7ca0000000p+2f, 0x1.1706920000000p+2f, 0x1.1fe4be0000000p+2f, 0x1.14128e0000000p+2f, 0x1.2334160000000p+2f, 0x1.34a1600000000p+2f, 0x1.1cf4100000000p+2f, 0x1.3a53ee0000000p+2f, 0x1.2c085a0000000p+2f, 0x1.208e920000000p+2f, 0x1.1a2a3e0000000p+2f, 0x1.2191a00000000p+2f, 0x1.2c739c0000000p+2f, 0x1.0b28220000000p+2f, 0x1.36cf6a0000000p+2f, 0x1.28b8e20000000p+2f, 0x1.19fb780000000p+2f, 0x1.3338400000000p+2f, 0x1.3688020000000p+2f, 0x1.1a59180000000p+2f, 0x1.30d5ee0000000p+2f, 0x1.1a02ce0000000p+2f, 0x1.22e37e0000000p+2f, 0x1.32ae180000000p+2f, 0x1.427fb40000000p+2f, 0x1.29e7a40000000p+2f, 0x1.130d7e0000000p+2f, 0x1.30a7960000000p+2f, 0x1.2b83b20000000p+2f, 0x1.3353560000000p+2f, 0x1.2082f60000000p+2f, 0x1.42557a0000000p+2f, 0x1.22d6280000000p+2f, 0x1.264f1c0000000p+2f, 0x1.36544e0000000p+2f, 0x1.23a1e20000000p+2f, 0x1.2eccb40000000p+2f, 0x1.24a1560000000p+2f, 0x1.33fd160000000p+2f, 0x1.4970ec0000000p+2f, 0x1.2045540000000p+2f, 0x1.30d40c0000000p+2f, 0x1.2836cc0000000p+2f, 0x1.2d52fa0000000p+2f, 0x1.153b600000000p+2f, 0x1.2956e60000000p+2f, 0x1.13bdcc0000000p+2f, 0x1.44a3be0000000p+2f, 0x1.444ec60000000p+2f, 0x1.23f5ce0000000p+2f, 0x1.1053720000000p+2f, 0x1.474f780000000p+2f, 0x1.2742200000000p+2f, 0x1.1b39720000000p+2f, 0x1.33217c0000000p+2f, 0x1.420f6e0000000p+2f, 0x1.26bdd40000000p+2f, 0x1.22b2580000000p+2f, 0x1.1aae540000000p+2f, 0x1.31d8260000000p+2f, 0x1.30bd780000000p+2f, 0x1.32a3000000000p+2f, 0x1.2fb1880000000p+2f, 0x1.21480e0000000p+2f, 0x1.1cf3300000000p+2f, 0x1.2faa940000000p+2f, 0x1.2e3adc0000000p+2f, 0x1.3459280000000p+2f, 0x1.225ad20000000p+2f, 0x1.3bea500000000p+2f, 0x1.2b98d80000000p+2f, 0x1.0d6a460000000p+2f, 0x1.1885860000000p+2f, 0x1.3067500000000p+2f, 0x1.1a27dc0000000p+2f, 0x1.2e07520000000p+2f, 0x1.0fdb040000000p+2f, 0x1.1c13920000000p+2f, 0x1.1e173c0000000p+2f, 0x1.31e24e0000000p+2f, 0x1.20f9be0000000p+2f, 0x1.1733380000000p+2f, 0x1.1efc7a0000000p+2f, 0x1.25a2320000000p+2f, 0x1.37c69c0000000p+2f, 0x1.19cd160000000p+2f, 0x1.28b9500000000p+2f, 0x1.24f3580000000p+2f, 0x1.28be400000000p+2f, 0x1.30899c0000000p+2f, 0x1.41038e0000000p+2f, 0x1.35b74c0000000p+2f, 0x1.35ea660000000p+2f, 0x1.33f6ce0000000p+2f, 0x1.45f9aa0000000p+2f, 0x1.2185c60000000p+2f, 0x1.1d49c60000000p+2f, 0x1.2217f00000000p+2f, 0x1.42ef0e0000000p+2f, 0x1.2807420000000p+2f, 0x1.14c1420000000p+2f, 0x1.3a1f760000000p+2f, 0x1.111c000000000p+2f, 0x1.1c9d2c0000000p+2f, 0x1.277afc0000000p+2f, 0x1.27bcb00000000p+2f, 0x1.237dd40000000p+2f, 0x1.16f3600000000p+2f, 0x1.2dd9b00000000p+2f, 0x1.2aec180000000p+2f, 0x1.3076c80000000p+2f, 0x1.18d8360000000p+2f, 0x1.3077a60000000p+2f, 0x1.2e21260000000p+2f, 0x1.2eab4c0000000p+2f, 0x1.334faa0000000p+2f, 0x1.252ac00000000p+2f, 0x1.17d9d00000000p+2f, 0x1.26d7d60000000p+2f, 0x1.325d080000000p+2f, 0x1.2c51e40000000p+2f, 0x1.2f164c0000000p+2f, 0x1.39d8640000000p+2f, 0x1.12d6f40000000p+2f, 0x1.228ee80000000p+2f, 0x1.07678c0000000p+2f, 0x1.254c020000000p+2f, 0x1.24ed140000000p+2f, 0x1.1f20ae0000000p+2f, 0x1.2d39b00000000p+2f, 0x1.19d2760000000p+2f, 0x1.1760dc0000000p+2f, 0x1.019d460000000p+2f, 0x1.1ad0cc0000000p+2f, 0x1.2883740000000p+2f, 0x1.10d0fa0000000p+2f, 0x1.26b2ee0000000p+2f, 0x1.3a9dbc0000000p+2f, 0x1.15578e0000000p+2f, 0x1.3ad9be0000000p+2f, 0x1.2155c60000000p+2f, 0x1.218c180000000p+2f, 0x1.366e9c0000000p+2f, 0x1.2ab9a20000000p+2f, 0x1.32f29a0000000p+2f, 0x1.251e080000000p+2f, 0x1.2a59380000000p+2f, 0x1.1e001a0000000p+2f, 0x1.297ed60000000p+2f, 0x1.2d42740000000p+2f, 0x1.2d23d00000000p+2f, 0x1.1ed2e20000000p+2f, 0x1.25aaae0000000p+2f, 0x1.14f0d00000000p+2f, 0x1.2dbbca0000000p+2f, 0x1.2207dc0000000p+2f, 0x1.22f7680000000p+2f, 0x1.3e2d9c0000000p+2f, 0x1.4ff4b60000000p+2f, 0x1.1b39360000000p+2f, 0x1.210d080000000p+2f, 0x1.26b0900000000p+2f, 0x1.2c27160000000p+2f, 0x1.40b9d60000000p+2f, 0x1.2428680000000p+2f, 0x1.1908e40000000p+2f, 0x1.2087ac0000000p+2f, 0x1.2b901e0000000p+2f, 0x1.1774be0000000p+2f, 0x1.2451440000000p+2f, 0x1.3ac2540000000p+2f, 0x1.3cc4060000000p+2f, 0x1.299b1a0000000p+2f, 0x1.2c2d380000000p+2f, 0x1.2d0f980000000p+2f, 0x1.2353b60000000p+2f, 0x1.2d59540000000p+2f, 0x1.12a98e0000000p+2f, 0x1.22020a0000000p+2f, 0x1.f9f3ae0000000p+1f, 0x1.118a040000000p+2f, 0x1.2a47a20000000p+2f, 0x1.26a9a00000000p+2f, 0x1.086ae40000000p+2f, 0x1.174f6c0000000p+2f, 0x1.efd19c0000000p+1f, 0x1.291c840000000p+2f, 0x1.4194b00000000p+2f, 0x1.20b5440000000p+2f, 0x1.2952e20000000p+2f, 0x1.2bd40c0000000p+2f, 0x1.1bf0fa0000000p+2f, 0x1.2317980000000p+2f, 0x1.2f1ccc0000000p+2f, 0x1.131bf80000000p+2f, 0x1.1b3ec20000000p+2f, 0x1.29d8de0000000p+2f, 0x1.0fafc80000000p+2f, 0x1.1b1d720000000p+2f, 0x1.2887860000000p+2f, 0x1.1e57b20000000p+2f, 0x1.2423f40000000p+2f, 0x1.4622020000000p+2f, 0x1.339f8a0000000p+2f, 0x1.0e070c0000000p+2f, 0x1.2082360000000p+2f, 0x1.3edab00000000p+2f, 0x1.22cb360000000p+2f, 0x1.22aeb40000000p+2f, 0x1.2a74580000000p+2f, 0x1.03e87a0000000p+2f, 0x1.0ec4f00000000p+2f, 0x1.1869d60000000p+2f, 0x1.2aab6c0000000p+2f, 0x1.13b6460000000p+2f, 0x1.24bada0000000p+2f, 0x1.0d5f6c0000000p+2f, 0x1.1f45100000000p+2f, 0x1.1bca500000000p+2f, 0x1.19540c0000000p+2f, 0x1.1abd480000000p+2f, 0x1.2a0a400000000p+2f, 0x1.0e3c660000000p+2f, 0x1.1269b00000000p+2f, 0x1.3566bc0000000p+2f, 0x1.1aac0a0000000p+2f, 0x1.18d2fe0000000p+2f, 0x1.2936d40000000p+2f, 0x1.223bec0000000p+2f, 0x1.2e17080000000p+2f, 0x1.2b5ce40000000p+2f, 0x1.1b9c280000000p+2f, 0x1.224d1c0000000p+2f, 0x1.13ae620000000p+2f, 0x1.0f1c580000000p+2f, 0x1.1c8eb20000000p+2f, 0x1.2827f20000000p+2f, 0x1.19f1ca0000000p+2f, 0x1.11ff360000000p+2f, 0x1.1ceb620000000p+2f, 0x1.16b78a0000000p+2f, 0x1.09ee0e0000000p+2f, 0x1.169ca40000000p+2f, 0x1.2723a60000000p+2f, 0x1.20916a0000000p+2f, 0x1.f382d40000000p+1f, 0x1.2365fe0000000p+2f, 0x1.2641920000000p+2f, 0x1.22c4f20000000p+2f, 0x1.2973d40000000p+2f, 0x1.23cc2a0000000p+2f, 0x1.0d01f00000000p+2f, 0x1.0e68340000000p+2f, 0x1.0843a00000000p+2f, 0x1.13a07e0000000p+2f, 0x1.2b820e0000000p+2f, 0x1.2149de0000000p+2f, 0x1.2674740000000p+2f, 0x1.2241e20000000p+2f, 0x1.2aae540000000p+2f, 0x1.307f0e0000000p+2f, 0x1.3bc02e0000000p+2f, 0x1.461dee0000000p+2f, 0x1.2a013a0000000p+2f, 0x1.1464d20000000p+2f, 0x1.2cc8960000000p+2f, 0x1.47488e0000000p+2f, 0x1.34c06a0000000p+2f, 0x1.4fd9700000000p+2f, 0x1.0776aa0000000p+2f, 0x1.1c39f60000000p+2f, 0x1.365d760000000p+2f, 0x1.1d28d80000000p+2f, 0x1.1687f60000000p+2f, 0x1.22eec60000000p+2f, 0x1.2339ec0000000p+2f, 0x1.20400a0000000p+2f, 0x1.346f880000000p+2f, 0x1.2d005a0000000p+2f, 0x1.0aa4ac0000000p+2f, 0x1.3265500000000p+2f, 0x1.2456da0000000p+2f, 0x1.fc61560000000p+1f, 0x1.1580f40000000p+2f, 0x1.3006b40000000p+2f, 0x1.0e85d60000000p+2f, 0x1.220f100000000p+2f, 0x1.0a61860000000p+2f, 0x1.2e4c840000000p+2f, 0x1.1353080000000p+2f, 0x1.0f947a0000000p+2f, 0x1.21e2580000000p+2f, 0x1.0f22ae0000000p+2f, 0x1.44b5c40000000p+2f, 0x1.05afb40000000p+2f, 0x1.07d1a80000000p+2f, 0x1.23cd700000000p+2f, 0x1.28abe40000000p+2f, 0x1.21ab460000000p+2f, 0x1.0f7f120000000p+2f, 0x1.11441e0000000p+2f, 0x1.1b18940000000p+2f, 0x1.15ebb40000000p+2f, 0x1.3b01c80000000p+2f, 0x1.2a3f180000000p+2f, 0x1.18a4d40000000p+2f, 0x1.2469820000000p+2f, 0x1.28a3a80000000p+2f, 0x1.26f2000000000p+2f, 0x1.16ff5c0000000p+2f, 0x1.1c80220000000p+2f, 0x1.28c7540000000p+2f, 0x1.1600600000000p+2f, 0x1.22e5de0000000p+2f, 0x1.2cd3200000000p+2f, 0x1.181c880000000p+2f, 0x1.1dfd6a0000000p+2f, 0x1.2c203c0000000p+2f, 0x1.0fe7840000000p+2f, 0x1.2a791c0000000p+2f, 0x1.2ab0560000000p+2f, 0x1.2055420000000p+2f, 0x1.1f7dc60000000p+2f, 0x1.1cec0a0000000p+2f, 0x1.1dbd3a0000000p+2f, 0x1.165a460000000p+2f, 0x1.2d2ff80000000p+2f, 0x1.1ee4400000000p+2f, 0x1.1f7c9c0000000p+2f, 0x1.515a980000000p+2f, 0x1.1a051a0000000p+2f, 0x1.20fb200000000p+2f, 0x1.3b79c60000000p+2f, 0x1.380b160000000p+2f, 0x1.2e196e0000000p+2f, 0x1.3253aa0000000p+2f, 0x1.350fb20000000p+2f, 0x1.40f8ac0000000p+2f, 0x1.24c04c0000000p+2f, 0x1.328e820000000p+2f, 0x1.2e60d60000000p+2f, 0x1.1f3bc20000000p+2f, 0x1.2681d20000000p+2f, 0x1.2a099e0000000p+2f, 0x1.1bfc480000000p+2f, 0x1.28e14a0000000p+2f, 0x1.31186e0000000p+2f, 0x1.1637d00000000p+2f, 0x1.440f760000000p+2f, 0x1.3168320000000p+2f, 0x1.1c1a8c0000000p+2f, 0x1.1701f80000000p+2f, 0x1.2b5cd40000000p+2f, 0x1.3bf1d40000000p+2f, 0x1.2ed2440000000p+2f, 0x1.2eaabe0000000p+2f, 0x1.3979be0000000p+2f, 0x1.31e6aa0000000p+2f, 0x1.1dfdb00000000p+2f, 0x1.20b3ec0000000p+2f, 0x1.0a12000000000p+2f, 0x1.3187200000000p+2f, 0x1.fd5d9e0000000p+1f, 0x1.0b93900000000p+2f, 0x1.0eb2980000000p+2f, 0x1.015bce0000000p+2f, 0x1.1b08240000000p+2f, 0x1.161ff40000000p+2f, 0x1.27c86e0000000p+2f, 0x1.1e5e5c0000000p+2f, 0x1.13c61c0000000p+2f, 0x1.3300da0000000p+2f, 0x1.2551380000000p+2f, 0x1.2037780000000p+2f, 0x1.2c1eca0000000p+2f, 0x1.2db22e0000000p+2f, 0x1.144bc60000000p+2f, 0x1.222f820000000p+2f, 0x1.0e47da0000000p+2f, 0x1.2028060000000p+2f, 0x1.28c2620000000p+2f, 0x1.1ee6b40000000p+2f, 0x1.1854280000000p+2f, 0x1.127a800000000p+2f, 0x1.3ade460000000p+2f, 0x1.41d8740000000p+2f, 0x1.171b760000000p+2f, 0x1.15046e0000000p+2f, 0x1.37287e0000000p+2f, 0x1.26be5e0000000p+2f, 0x1.3c19760000000p+2f, 0x1.3b6df00000000p+2f, 0x1.3d96ae0000000p+2f, 0x1.237f260000000p+2f, 0x1.1143360000000p+2f, 0x1.26fea40000000p+2f, 0x1.10ed880000000p+2f, 0x1.268a1e0000000p+2f, 0x1.1cf6520000000p+2f, 0x1.2a41e20000000p+2f, 0x1.1a9ac00000000p+2f, 0x1.26f2d60000000p+2f, 0x1.1891aa0000000p+2f, 0x1.06e5980000000p+2f, 0x1.4534980000000p+2f, 0x1.40a0640000000p+2f, 0x1.273b820000000p+2f, 0x1.16ff420000000p+2f, 0x1.2c75620000000p+2f, 0x1.3b67c20000000p+2f, 0x1.2d7c8a0000000p+2f, 0x1.3587400000000p+2f, 0x1.14abec0000000p+2f, 0x1.2275da0000000p+2f, 0x1.33b7f60000000p+2f, 0x1.2e6ff40000000p+2f, 0x1.337e6a0000000p+2f, 0x1.1a50d40000000p+2f, 0x1.175a800000000p+2f, 0x1.18007e0000000p+2f, 0x1.2f6e5e0000000p+2f, 0x1.0849160000000p+2f, 0x1.1fbbaa0000000p+2f, 0x1.1e69360000000p+2f, 0x1.0345360000000p+2f, 0x1.1948520000000p+2f, 0x1.239ce80000000p+2f, 0x1.2ebcac0000000p+2f, 0x1.1fb68e0000000p+2f, 0x1.15f5700000000p+2f, 0x1.3e3bfe0000000p+2f, 0x1.2c0c860000000p+2f, 0x1.2e4b000000000p+2f, 0x1.15ea000000000p+2f, 0x1.1f31c40000000p+2f, 0x1.3264ae0000000p+2f, 0x1.2e6df20000000p+2f, 0x1.19fca40000000p+2f, 0x1.0fc9360000000p+2f, 0x1.2598020000000p+2f, 0x1.2620980000000p+2f, 0x1.2f9df20000000p+2f, 0x1.1ec4500000000p+2f, 0x1.3de4860000000p+2f, 0x1.2d29800000000p+2f, 0x1.15da8c0000000p+2f, 0x1.2c1e0c0000000p+2f, 0x1.1730a00000000p+2f, 0x1.22439a0000000p+2f };
static float out_0[1128] __attribute__((aligned(128)));

int main() {
    forge2_hmx_enable();
    forge2_b64(b64_in_0, golden_in_0, sizeof(golden_in_0));
    int total_errors = 0;
    int total_n = 0;
    int first_bad = -1;
    int last_bad = -1, bad_stride = -1, run_len = 0;
    int r32 = -1, r64 = -1, r128 = -1;
    int same32 = 1, same64 = 1, same128 = 1, uniform = 1;
    candidate_kernel(golden_in_0, out_0);

    for (int i = 0; i < 1128; i++) {
        if (!forge2_close_f32((float)out_0[i], (float)golden_out_0[i])) {
            int bi = total_n + i;
            if (first_bad < 0) {
                first_bad = bi;
                r32 = bi & 31; r64 = bi & 63; r128 = bi & 127;
                run_len = 1;
            } else {
                int gap = bi - last_bad;
                if (bad_stride < 0) bad_stride = gap;
                else if (gap != bad_stride) uniform = 0;
                if (gap == 1 && run_len == bi - first_bad) run_len++;
                if ((bi & 31)  != r32)  same32 = 0;
                if ((bi & 63)  != r64)  same64 = 0;
                if ((bi & 127) != r128) same128 = 0;
            }
            last_bad = bi;
            total_errors++;
        }
    }
    total_n += 1128;

    if (total_errors == 0) {
        printf("HVXENV_CORRECT errors=0 n=%d\n", total_n);
        return 0;
    } else {
        /* the SHAPE of the bad set: what names the bug.
         * mod32/mod64/mod128 are the shared residue when EVERY
         * bad index has one, and -1 when they do not. */
        printf("HVXENV_INCORRECT errors=%d n=%d first_bad=%d last_bad=%d bad_stride=%d uniform_stride=%d run_len=%d mod32=%d mod64=%d mod128=%d\n",
               total_errors, total_n, first_bad, last_bad,
               bad_stride, (bad_stride > 0 ? uniform : 0),
               run_len,
               (same32 ? r32 : -1), (same64 ? r64 : -1),
               (same128 ? r128 : -1));
        return 1;
    }
}
