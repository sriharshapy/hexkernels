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
    "/lekP27vgT8bCis/MnmRP8l0fj8eGpA/AMaHP20osD+68y4/9uFXP36DGz9QGQQ/mLJAP/KksD97lCU/gjeKP15Lej8Vwyw/So0iPz0IBD9MRpU/HRtuP+jj"
    "CT9+kKo/+AC+Px2rWz8yDCw/TAaVP34cMz/t0CI/yaerPyxvkT8dLGc/2kEdP/pfrT9tbAM/Y6qRP8kzKD9TRx4/EOi7PwHQtT/nN2U/1vZAP6EvRz9kfY4/"
    "lIxrPwSEtz/ibp8/A2OVPwgVaz/4dok/ipC+P/oQrj98ApE/QGUSP9ZaLz8Ns0A/ajWoP8k7MD90kqs/nSF/P/SVJD9uh6w/aISOP03bDz9QnY8/KF6dP1Av"
    "qD9o7h0/Iv+yPzwDbD/nbAo/Rw6bPwZkhT8v4ao/FnupPy5Avj8pTwc/sddMP8YIkj+2d5I/V/M5P9e5rT/41Lk/kW8sP2TkBz9cnrs/RqcmPxrWqD8mjL8/"
    "6xZtPy4VoT+oPIM/F0Y1P5qLrj8tyEA/PDuLP/j0NT9wWog/MlglP//VcT+gPqA/1AkGP4gXDj/oZTc/VlCEP/hkuT+m+o8/vac6P7A8qj/js4Q/n5sBPx0G"
    "oT/PYJ8/8MtuP87rbz9cLB4/QK2KPxCREj/W3SU/ATVAP86Kgz9ZGTg/urMYP+PaMj9GcL0/kq2GP+40rj9fYgc/nMYJP0zrZD9m4yM/5iqkP04nQz/GP78/"
    "5QxHP7wEoD86rq8/F8m1P56odT/3v0s/WDZiP9TAvT8oMwI/w4RaP7SzAT+qapo/zMakP6oMWT+4k5Y/4M8AP9caRT93BEQ/Ktx7PwpRgD+CEzo/5c2qP/ui"
    "ST8Q240/2fCEP6YjkT87ODg/QImVP6BONz/+NQ4/XwcCP0sDtD8M63k/b146P+rsYj/NKW0/FcCBP66nfj9QWYs/2+ZlP9kiOj/onyc/BiyeP13KWD+ayrk/"
    "ZfYDPxZ7tD+UHJA/2itoP5/tNT/mrlI/BneTP7GcYz/FAiY/mE0+P08ZcT/qQVw/FlwGPzcXkz+yW2k/puePP+5Apz8OVVg/tNesP0XavT9Gj4Q/1ZScP7HA"
    "tj9zAGE/K52yP+kBPD+ti6w/QHeHP3GeVD+UOY8/rHKAP4c+ez9KmzU/g7etP/Rgiz8OP5k/1fx4PwUgRD9KJKU/+go0P4HkYz86F7Q/9ipdPyvNNz+PZRE/"
    "tm0zP2Qzsz+In58/h5dAPxHEOT/NH0I/TRKkP5BciD9QWWQ//WcEP3oCIz9oorQ/CDipP2x6HT/Bk2A/V7olP16qjD/FWo4/ze1PP8z7Mz+YJIs/8fF3P4Hu"
    "MT9IyYQ//UGtPzvZbD+Hc1w/R/qzP4YutT8lI5Q/kH04P8x0jT9GEHI/ntIeP8e9Hj/MEHE/xbtlP09vQT+WZiI/XoazP7bIsT+9zkI/YamGPzv0uD8kZDM/"
    "1bOKPyzALD/3JBM//z+hP2xypz/Ttog/YlOoPxJpLj8HC3Y/+/EoP434Iz/ETUo//m5KPwYfKz+MF4g/g465P/82Ez+UBRg/dCSPP9zRRz+gqLM/fhyIP+rc"
    "DT8fUKo/HAGUP45RRj+Z+LA/SPqKPxz/Mz/dt4A/8Cw1P/lgWz96tEo/JHq0P87Yvj9GHrg//8BsP09JDT95vXk/sHKuP4eVqz+EZxk/bGkFP7a5sD/tKCE/"
    "fPi0PxSloz9oLKU/HYCuP3rYQz/4za8/eFENP5MhMT/gCZM/PFicP5CWhD8A4Vw/c28BP0UjUz/aXK0/ZuSpPwMtEz8QaqQ/9omJP51zID9yPa0/ABiWP3Qj"
    "mj+INqg/22JbP6gQij+BUbQ/ebyfP33Rvj+XUQQ/orsIP6pspD+2SBQ/d2pFPxRCrT8N840/4gFAP+Eqsz9037I/EBF4P4ksPD/JfGI/8c+RP8vDrD+XcAI/"
    "DhSNPxjsgz8xLEQ/DtagPxtHlD9c06g/BdaDP5ABsT9BWwI/JCyTPw9AQT+EJjI/DLm1P5DOhD84Umk/3fB9P5MwrT8M2LY/HTJCP0C7Yz96eQY/+EeEP0n4"
    "pT+Ckks/VLenP116JT/70l8/b9oEP5CSKD+2EAI/tuGYP3YTuz88cJQ/nry2P4vXYD+ykrg/jBibP5ZOpj/RZDE/YJSYPzUUqT9wQGk/m3tsP7KstD+1i0Y/"
    "Ro1rP4MHlj+mYFE/0kalPx8xGD9Jp2Q/YFwwP+lMrD9RrpA/yfmrP0BIpT/oKQg/LhZ4P2Imjj+ZIwY/8iqcP2oerz++Pow/oC69P+V2uT/VKko/KGtEP53d"
    "aT/muQg/TGAVP39QbD/JMwc/kaIpP4sPcj9pnb4/HJOLP1IAqD8mZqY/6dV6P1brYz+lM7Q/UblyP9IWDj96HIo/QKZbP+4iSD9qUKc/7zcRPw5nKD+4olg/"
    "6riBPwrZlz/m03Y/BCsGP9yMRz+R/rs/C6G8Px1eLj9WFpg/92+yP/uQND/etAw/PoKuP+ybrT+sY0E/1jebP/yKrD+DjjU/wDSfP98jTD/AKAU/sAm5P49s"
    "fT+fRY4/37hYP+RdGT++hb0/8I6DPz7YvT+w86Q/jEJ6P/nzqz/mrak/blZdP+THPD/WSZ8/CRF9P8IOhT/yYqE/XgWoP/1YqT9FMkk/nEKGPxTvqT/kMFg/"
    "Dyy8P4J+gz/cAp0//GWiP/ARrj/bkz0/1nOrP5+Sfz+4T48/Yee+P4mqTD9rows/xHOGP4LJqT9pXVA/lV8uPxlcPz9N+yQ/z9uEPyQpDT/+2BU/uui/P3Y8"
    "Kz/V1Rg/kgkOPywauD8Wgl4/7Xx2P+Djij9OUJE/WlAuPxKKjD/wzbc/tPF7P+hRiz/gJbE/moKCP4jJVz9waZc/Fa+jPz34vT/LBUU/DOqGP1r+nz9Bcn0/"
    "dvCxPzGmAD/AyqI/naIlP7Ivrj/cA3o/xvyjPwIhFz/HIVE/1AY3P6xgtT/k6os/kJAuP0cgEz98haA/WBW7Px2oCT9EIK0/7oFqP+YGRz/CW4I/PP5TP9Np"
    "mz9VMDo/25MBP5a4Oz95GaY/Ha4cP+FIpT+a/lM/CsmzP2Xnoz8SB7k/+E8jP3tXrT+sILA/yuO8P6wosT9EXJw/sl1IP9l2mz9oUY8/HpY6P9wOuj+P+nk/"
    "7yFJP012OT9hUBc/Sru8P5+QET+t1Is/iAuEP9O1jj8MAp0/1g2GPzodhD/8N6k/8Hc4P4ZYmT/QjTM/8dwMPzZ/hT/cMKk/e2ZZP6FUWD+JboY/Bwi9PwNu"
    "ND/uuVE/nua2P4ZGqj9kqFA/pYN/P66HnT95Lgg//oWSP+p5oT+CJJU/gD+LP82oFz9ypoE/2F5aP4rqvT+yHAM/lel+P1cNLz/J1VY/lYcnPzYmmj+DImw/"
    "+W2rP+N8WD/y4IM/z+qfP5RCtT9nsao/xS6oPxYUiT/F678/TM67PzUdQj9teQk/gHVqP51DOT9eUTY/YoC6P1Jmvj9IOxc/1tmYP/AGjD81RrY/E3SAP4ls"
    "cD9RjW4/DE2PPyQdnD+MOAk/aGA4P5/SBT8yfIM/GeobP5w7bz84CKo/tjeSP4v6Fj/crCE/0g+nP/Qvjz/i7oE/0nZRP7CmiT+t2jc/140JPzONhz9GdEs/"
    "SmK1P2uYvD/ql4k/3nmIP7bMsT9UL7o/EEpZP4x/gz80eoI/E5acP1Blmz+1uUA/KPSvP5gqnD8eDoA/9EqqP4DBLD++q5M/05sjP35+Lj/SsRA/JwNwPzfR"
    "Dz8cbIE/ZhUSP8elhz9X/j4/KllqP+yCtT+A1TU/tuiXP4mUaz8fYwI/g0W1P2aqaj8TgRI/NF9+P3IQuj++0aY/5AGtP1bMgz93T5Y/XDGvP/bZWT/Cjq4/"
    "MuVzPzTKNj9cppg/kA6rP6tenj+siaY/cKwuP2GfOD8LNHQ/iGGxP4mnDj/vlSw/9pQZP8cGqz/7mVw/SwRfP4F7Ez93/VY/vg6cP308tD8//bM/B0QpP+yh"
    "Rz8yyEM/BE2SP26HWz+VUpk/ziicPxYkEz/kMac/ji6KP/LCrT9yGKU/EuqJP68/Dz8P4Wg/RFl4P+pKuz+r5RM/vtKKP2wkGz9UphU/hEiWP54duz+VzAQ/"
    "giGmP9fVUz/nkns/MZ2SP27enD9MNLw/MdB3P5TRkj/cA2I/3EemP1EFhT8AZig/AuJmP2i1Cz9Kih4/zAqNP0zzpj/Dp5Q/41CQP26GsT9yoGo/OhxKPzmZ"
    "gD9IWRg/+NqkP3a7rz8AlSw/smNpP4WnPT+ISI0/Wm1uPyCNtD90lZc/39w/PyD1rT9cJ7Y/t7WFPwMaBj9imrQ/5ASAP3cKkz8A7qw/0AWsP5b5Yz8cQbU/"
    "khGpP0LbqD+w2j4/3uWuPz7DkD+QtoM/XwmfP+aEgT/MU7s/qkmWPz7fkz+oTgA/bkitP5L2vz8luWU/3pttPx/vfD9ASRU/VjG9PzQcID+JCFQ/P5RMPz5f"
    "lD++E1U/tM2KP/XBFz/uDos/A5BvP+TXgz9+G5s/numYP3bImD/TAWg/CXelP4KDCD/ncIk/xsZSP4oKtz+6vCU/Td5lPyVEBD/2PoY/SLBuP6JDrD+t4aQ/"
    "uomjPzQOqz/k870/NviCPzxmgz+4qG0/jyYmP1KpqD+d5pc/kDyiP3ibiz9xSIA/TjiiP/h9uD8Ld7U/uUkXP3S1lT/YGAE/HCllP8BrFT/sopc/u/c9PzKZ"
    "mj9c16A/xlKeP8PdET8PO4g/xoyWPzLDhj+jGQo/BfF1P0pTvD/Cbq8/5M2VP7eGST+Dplo/TOA7P2Z+Gj/guEc/Sf1MP2Iqsz9+dqY/CbdaP0VnhD89864/"
    "LJsxPxibIz+wS6w/SiyvPwWVrT9nhSY/7ttQPzY9aj8Hk7k/vZFEP4Brvz8dvXM/DjWePwpFiD/l80g/wnyTP9GjBz+BJXE/NLRCPz7Plj/GISM/X9kLPxR7"
    "Pj/6KbA/mpKsP6gMlT+Wc7I/Ogo8P0X5Gz93onI/1mxpP/ATvD+O+ww/VbtJP8nnXj8ZiLE/ZhmyP+MtGT8tSaE/xme9P1tOCj+p9Is/4xFuP8Pqkz/b5SQ/"
    "NCaPP5DgnT+Tulg/zl+lP8PFVT9pJzc/CKqAP1PoRT/sVK8/+MJ3P+AAvj/uJ6U/aHUEP9Hpnz9nJoQ/YtykPzPPNT/I6Uk/0QS6P9M2ET9BTE8/x6sVP1d7"
    "hT+ef5E/6Gm8P3rwnj8J/Ug/ay1WP9XBtj8/MlY/qNk5P5pVBz8pwX4/u5c2P2d4sT+tKiI/ltWgPxdMAT+7CXE/WQm7P0DYgD9+BUQ/PkZqPwAVrz95vJc/"
    "VMi4PzockD8lxjI/jRwEP3asvz+cixg/eIsIP9DZHj9twCY/3c1WPywaQz8zMWE/Fs6OP3Z7pT/ZWjY/AXFuPy4NLz+TFyQ/UshXP7llKD/BK0Y/gg4pPxr/"
    "Jj9AZZE/eiYPPydVAj+rd1Y/LlWPPyRPAT9ccYU/8pGQP+5+lT9yW08/ElywP/I9qj/s8pA/zji1P8tfbD9lB4I/BCVDPyENOD8Aoy0/JBk8Pyndpj9Nl7U/"
    "ea4WP8jFvz+wPHc/iO2CP61goD+lzV0/LXEYPy2UJz/a8pM/kSguP7yPVD804LE/aBaeP/wYpz/An0Y/qMOCP/BuXT98jrI/6LJrP7OrnD+6tBA/5R+7P1PU"
    "mT/KJyI/0ly6P6q8hj+iJ6g/JmekP138Uz/WJbk/pHyPP6TRTT/nSnM/G9ESP4ICsj8zJhg/6OSbPzU4vT84xZ0/47OwP6mDqj8U5L4/NkmSPxi+gz9iDIQ/"
    "OEeQP9yHqz/+07s/rc8OP4aFqz9w8E8/FLVUP8TYuj9MQrE/vzWBP2ELXD9UIrw/ILUVP2ilDT8iCo0/PgKxP2RwkD/Lppw/EjSxP1qODD9oMBM/gGeIP7ju"
    "ij9jcrY/YxeIP5yZAz/+PIs/Ze5fP4/Vez+btKQ/8xQ8Px0boz98HYU/WKaqP1qsMj9sfL0/pKQYP8emlj9lUHo/IQ2gPzY5sT9Rchs/ivdIPxE2TT9+9jU/"
    "bXgVPzzEoj9CMLU/KV12Pzh+bz+ssE8/MlmOP575OT9zOj0/A8wSP8jdRT98nIs/bH+GPxvuOD9K0oU/AR6ZP7qfhj8w54Y/eFGnP+o+XD80/7c/fPmlP1ob"
    "hj9wepQ/8AwPP1zcrD+Aj4E/qolxP0Yllj/nr2E/ivQ6P7jBvj+y1k4/mlOfP1wksD/fTgg/cxwOP0e/Cj/G8KE/+rGTP79sSz/ALn8/WRSEP8z9aD9Wo4c/"
    "3pWiPxAVtT9phyk/CLKNPzv7BT+l1xk/6jOHP6ZKIT8Wq7U/JyuGPyasqD9KGXA/yPYwP9JlOz9s2Ys/RF2eP8ipoT/6s58/XzA2Pz75gj+Z2ws/iiG0PzCw"
    "FT9Qt6s/WJeCP6cwWj9HSkc/5YN6P9jXuD/ICyQ/sm2KP07Hlj882J4/DEKKPxAfUz8dgg8/Eqe+P0aMkz++31c/jQlKP6OpWT9Aek0/EoGiPyLJSj90Kqg/"
    "SmiJP2WcOD8Cnaw/HUuKP9MqVT+gVk0/YgSDP7HIMT/wUrk/b19WPxMOlD+uUkE/g403P0pEkz/oYrc/+4wJP/upSj8+XTc/Ak6rP0TLhj9V6hs/Dl82PwbJ"
    "CT9W45s/anCqP8qyhD+wcRA/xWk8PzHQND+qBH0/HIeAP512fT//1rw/zjyFP1WGTj+S/Lw/XsCFP90wnz9zrX0/tVl/P+p9uz8inzo/hHoLPxqPhT86sYE/"
    "0XKPP3SzOT/TaHQ/jURgP/yOlj9n924/KZkPP86Fkz9o24s/Tqu9P7ZKNz90OBA/3IpZP3NuND/CFQU/WLegPzkBnj8v4wc/YwmkP/ZJlD8ke5M/tvRGP5Zt"
    "oT+RagE/OVCkP2ILND8+HJw/rVp3P8EJSj+mq6M/5EYRP7nNqD9rebg/HMIAP7igpj8QUoc/a3icP3WJKj8RSzY/zZk2Px2iOD+bYic/x5YrP5wsJT/7eGk/"
    "ISKhP9odrT9sH0Y/QgW8P3Zisz+CSLw/dP2hPxhlhT9oPJI/WA2TPwMohj+T7qs/XjKiP35ogD/GUV4//KoQP85JuT/OfII/AMG8P17Zgz9TEkE/B7aNP+z4"
    "GD9KIzY/FsC9P5xOlT/AqFc/yqgjP8tFYD9QpZ4/lmqnP35UUT87Cn8/jrUMP0Chrj/9g1w/oV8eP4cboD8CCq0/uDeRPznghj9f3jk/mP6SPxjWmz/SHp4/"
    "U6k/P9JIuT8DXH4/CcoqP+xuuT9Z8VY/iDpYP1ePeD+nSzQ//jgzP7sKtj879Vk/MBSfPx/hnz88Qpo/pgmOPxwLsT/mskc/02CaPwDhBz8CSUE/mSqfP3bD"
    "Xj/y1BU/1WOuPwtAkT/MiX4/DhplPwAnlD/9fDw/p41oP1Rekj8Xkls/+y4CP3zlsz8WD1Q/LIgWPzi9GD86LLs/2kWyPx4neD+zeAg/FRWuP53uvz+I3To/"
    "yB6NP46IKD/AkR0/abqTPz/ZZT8TG7s/29N1PxfQlj/2HIc/qLy0PySkgD9b1CA/48yYP3pOvT93jR8/zAqIP0Khrz/R8ng/uFx1P+kMWz/Mxio/DaBuP/Lp"
    "iD+2vUI/QLaTP7nkgD9+e4o/ieOPPyKYqD+a+Ds/ozQdP+4jlT9E9UY/kPaWPyDPbD+cLJE/1OuVP5pVlT+PI3s/3/8AP/D4lz+iSLI/YclQP9AVKz8qKZ8/"
    "Ual2P6HTHT8KBy4/w5kgP98Wez+8RaA/dpR0P5slfz+KGBQ//psPP/MAST9VZqA/nYgaP0TfvD/agoo/Vnc4P+qZCT9wrwQ/Ig1PPxPLVz8jYn0/BBquP5eZ"
    "Pz9QZ6k/UoF+PyNEPD8A3Zw/XwclP8Jfmj91x34/2tdJPxXrjD94jJU/5epzP+9LCD/kp24/Zom1PxVJmT8O4ps/jdl9PxfROT8NpAQ/JM6EP1cEpz+aiaw/"
    "qoKAPyxPZz/qKJ0//ywAPwrRpD/Cq4w/ISuuPyIfBz/+zQw/eo2SPz6GiT/8hb8/dkwZP22auj8BC6Q/yESqP7x3SD+I/5U/x5VlP1tDBj8e6mU/g6mHP/Hk"
    "Az++MaE/nDJzP1ietT9QzIA/FTsGPxrvhz954CA//VyhP5+Rpj//8G4/hChbP+WmLD92T7E/eGI5P6RDez+GSDo/nGYtP9KiMj8ZDwQ/rNW6P7zwQD/oMa0/"
    "6zshP6MJSz/WeEs/M5SLP6BlUj8vyxw/RLCWP1bymD/sTqk/YhkfP2gHJj9Q2IU/8vN+P/lZNT/OhqA/YO6sPyxTjT9o4IE/+msbP65VRj8sXLg/Rs9PP0wA"
    "nT+vlgY/jbAjP4UAEz8TJhQ/oKeqP7QDsj+WcaY/CTS1P1byET+vZws/MFFaP2udUz/8EhI/nNivP02rOj8Gyoc/TGNxP0hrfT/yzYU/UfeiP4lUKz+kFWs/"
    "GElvPxwjmD80V4E/eB6/P/fxVT+Rbz8/4e6QP7S3OT/c/qQ/UygXP96WpD8bBJc/RpwBP8Djmj/IFUE/Kx5WP6kUfz+UmgY/rnutPxt/ID/MT0U/lEx3P8P+"
    "NT+rLqM/BYGNP3c5SD/sFQ0/7HIDP2+QrT+gvp8/K6VVP7rCbz/Sva8/c0SDP06ghz+w7IE/rGB6P5YvlT9n1Yc/pXhwP8MCfz8SaUA/AJsqP6RSYD8yxJQ/"
    "VNGPP3x7BT9eTTE/92I0P1ZtrT+DjXo/++e/P7AjoT8ATwk/3kWHP7NPWD8S8hE/JI6wP70jkT82Y5g/mDmJP7yGtT+sQhQ/i68IP91UET/E9qw/f6dKP2ne"
    "CT/wS4w/7uhYP9aotj+0yW0/50VSPygEoT9KGRY/UrG4P1LEiD9HwYA/VTRXP53Rhj8ZsWE/B1cyP1oUpD/wwTc/ZfURPxARoj8+4qU/DFUSP+ETgz9Gm4U/"
    "TBg2P0SRqz/CBJg/qjOQP2A+HT9KBWM/zDK7P9rXqz/x7S8/X3MFP1wBaz9fy0k/Qs6CP4BBMD/N1zQ/gstdPxYRuz+SLRc/vfcdP/VaVT+ZZ1w/bj2yP6vg"
    "VD9HHCM/iBGnP11nLz/EXD8/kaEnP9hAiD/TtJA/N25xP6F1gj+CyHM/ev+GPzaWGD/TCK4/8biDPy/upT/MXIg/knaaP6q2ST+W5pw/QnSZP23RZD/4DYc/"
    "aACDP1Jkvz9PtYo/ONaQP2fxZT8j7Tk/XJUaPzzcmT8PBJM/ZLl6P0nftT8A+oI/kHgRP1Dxvz8ayEc/5KebP6oBEz9SXkE/e2NLP4Jipz8GWQY/2UQKP8Nd"
    "Gj/p44A/24iWP5I9iD+Mojg/C+8KP2kAMj+9vkU/DtejP6oOoj9Thrg/Ww2lP0LShT8AO3s/gB2gP++iMT9VHoI/B6N6P2ANkT/P4y4/aEOsP55tQz8OQgc/"
    "HOJHP9sqVD+whGQ/QJ93P5+XtT+57zg/CNt8P7R/vT9ORY0/9sKeP/4Irz8g/Jo/9sgAP1OwXj/9j6Q/y/iBP5o1hD8v8yc/MNqtP1hJHT8USWs/Y0kyP5yy"
    "mj9j7lE/GEMAP9znCT9zimM/USNDP3NMLT/5onE/VBK0P4wLKD/mdJs/FouePzrCYT+1MIk/sHM3PzRPCz9nNQQ/aIwbP2BBoz8931s/Jd6BPwOHcD9uOLU/"
    "oUGsP+0LRj+R/Tc/7HC+P4rsvj8+R7w/Nw5vP9KISz+lG0o/zgOsPzZ5eD9IPqA/afCiP/AMsT+8ERM/Nc0vP6D2mj/qsJY/URJ8PwArej8U5hM/VhyuP6F1"
    "rD9sfhc/0k8cP6gvqT9U7rk/MSB/P/wgvT8mMoY/GAAuP7Y6aT9agy0/tQ2iP9vXBz9IiT8/sCmOP5esZj+uCao/tvGsP8Qivz9HkGI/LL6cP9qsgj/MklI/"
    "V1BwP1wuNT/iiL8/aNW7P3xvYT/Cg3w/yturP+c6Jz/qtFg/sD9XP+zrXD/tyLg/RURxP5batz/wSgI/fNKPP3vIVD8W1r8/Q5uzP25Ejj/Aeqc/5pYjP9ct"
    "cj+iFA4/qlWhP6LXqT8zA3U/XaAGP3N9Rz9O54k/Q4gHP+wWkj/GRo4/dnS8P9lUID+IO4s/P6gQP0K1nj9elnk/NSEfP4J4sj8sxB4/23iiP5DQoT+gynU/"
    "P9eIP1MFuD9wugM/cHOBP7oehz+n32Y/AARgP9EvLj8BmxY/tKqcP22krj++KLM/PjayP8hqmD/kJq0/l7hzP9XMlz9wMoc/pJwqPxQLqj/i1iU/YlwsP/Q5"
    "NT+AjrQ/hHdlP1iKlD/ojLk/31QGP2Z0oD+ZO1I/U0KkPws1Lz8Xy1c//Sx7P23LiT/8mpE/8GyZPxv/aD9UoR8/8vR+PzSfmT/capc/1k+7Pxo8LD/stbY/"
    "OJKWP5AvPz+qYr8/NP6dP4yIhz/bH00/djCvP4FWNT/GtIg/iKMBP4iJsT+sebc/cIyPP3FkqT8gTr8/C4ADP+TJgz/+4aY/jzEiP9oUpT/TQp0/UVcdPw7+"
    "Oz++IpA/xKyjP+1/NT8zJiI/evudPyrMdD+vlVQ/H/8yPxXiDz+J9yI/igc5P4HpoT8xv1s/hteLP9jsgT8zhkg/7UuqPyY2kj93nz8/ugigP4/yNj9C6Zk/"
    "bniwP6uHYj/+4YM/hVOXP0WKTj/2rIE/M5uHP+ZgSD9Kvb4/nCCOP7fMYz/EMYw/eiisP8qloT8uLX8/kDa7P7hoCz+0rLM/I6ETP6yKvD86MJ4/hdKtPz9/"
    "IT8RtyY/zb96P9X8qD+eUaE/opBVP8c6mz+JoVU/GLxZPyeNGD88LCM/0OGcPxPuND/XoFM/b9AYP5Jxfj+p1Bg/PjifP7kQVj9/MJo/ouAgPzqXtD8u6bY/"
    "9I0AP8ADFz8ib4c/7045P2Q1tz/o2qg/bG28PxS7qT+6OyI/7TJSP86elj+gwLE/9a0IP04qjz9GNpk/qF2QP1F0az/QKas/2kyOP8G8bj9TXjs/bSGtP9ip"
    "Lj/637Q/6mK4P+V7GT88EWo/tKcVP9YHoT+embc/kgNQPyXSFz9BS64/toMDP4wZJD/dLXM/ooZxP1yyiT+3L58/zKSIP79JWT+dmAk/2CSjP/xcmT9fr1Q/"
    "XrU5P1HQOT9yrrM/MuWEP8iEsD9+2AQ//sYbP7+jDz8dGHg/ltoxP9GyVj9lf68/F1M+P2EgtT+6+bY/YZdJP6vtHj8+4pU/xhy6P73BOj+9UZw/F6dmP3Xx"
    "MT9WW5A/wMU1Py3lBT8KY1c/8/5UP/mThj/KIjQ/3CV3P1CGez/qur4/Ga0WP4P/Pz8TRZA/aQpaP388vz8Lkq4/WOGBP4ZRrD+x41U/WHCiP87inz/dF50/"
    "Uu2aP/h8ST86lBI/bCGQPyTFtT/f254/BfGSP4cxIz9Yybg/vHG7P3qDmz+emV4/d+2PP+MKkD8YvZ4/d2qFPySBXD8V710/avsfP+MkJD/oU5Y/rX5JP9XH"
    "Gz+0vYo/T2JlP0+9rz95vDQ/WnxePwwgpj8d/b4/Fp+VP29WvD/sbJ8/G1JTPzQxTD+7GQ0/xHqAP5DyVz9coF8/YUZjP1t0KT95H1U/iLafP/hMkT90J1A/"
    "oBVjP9ZlDz/FJRY/IoARP6xdtz+UGgo/wwAvP+B5Uj+cznc/zaKTP2dksj9yiDY/bEiRPxkxOj/22I4//qmyP3x8CT9NkDQ/4W4mP2Kpkj8omVc/st6IP1h7"
    "qD+o7Lc/HYunP1SnhD+DZos/NPmwP5eAkz/U36g/2dyJP3BMjj8ehjM/od+tP6fkMz+OiA0/ey6KP+mWlD9MzL4/11sIP3DkSD+YsKM/BKq0PxNhsD+oVUc/"
    "4Qm6P7rprD9PRxw/hbBiP9EjsD9Arro/NlSIP6cKOD/nk6g/rQotP+pDtz98Nmo/8vuHPy2bNz88M78/9mSQP5uZQD9GVIA/BDeCP+N2gD/Tyiw/264kPyL8"
    "oj/GNRY/FYiIPz4LPz93kK8/3HK/PwzpJj8vPSM/sAiAP8I3IT9Jj20/hvuaP8+5TD9KVL8/4sCJP+Q7Dz9W0ZU/AB63PzjWvT/bJko/VlhfP9whBj//eUU/"
    "RuUjPwMcCz/zeAU/uagdP4+ZpD/uf3o/TSFjP85wfz8yxns/TB6VP24mIj/IxXE/1BMEP6iGBD/kixQ/lRBXP0LMrj8FQCc/XpmrP9G5Mz885KU/dgCcP0IL"
    "Yz9Uil0/1fegP6dTej9Mv7E/2T18P/y2aj8JnHQ/aUFfP0BaRj+Vcrk/eP4+Py7FVz+2UjU/jJCaP9BGMT/XYqE/FqZtP6JVXD9jcQ8/l5M2P6Qpvz/piTM/"
    "aLepP5g5Pz8UFa4/UGOpPyI/RT/D5GU/tvGnP9a0Jj88GZU/JklvPw9jOT8sqKE/YXpUP6jClT9uZR4/qKsYP4/+Oj/d0X4/kMh7P+AyTD9gPqs/Vk0nP/0Q"
    "OT9OBI8/wtUrP3zGuT8pZzw/k2s4P1wgLT94sa4/PedpPxxDCj9OHLc/bsFWP0RknD9Hhpg/HWmZP5FYQT+IyKI/VWNlP2ZjTD/4+w8/5HqEP8bAdz/IdTA/"
    "bReIP9HliD9iuVw/t0YDP9xtgT+alj0/JhFRPxTGqD8itqo/ENSxP62JJD9UrrA/Zco6P+CJhT9ukaA/ltadPyVcLj98X3k/fU+SP2bekD+CaJg/IBWOP8yw"
    "iz8BySE/talSP8lKBD/0aAM/cuW9P+SyWz/R8Ug/0mE7PxtRSj9mEIw/6rSsP1DVED8WI64/8pqVP46tRT/ufgI/gqmRP3IvpD+X3bI/xFN4P171AT829aE/"
    "l3ZUP733mz8QD6A/1maiP9DMrj+IyVY/HrCmP8cBcT9VRjw/mqI/PzkcLz+bqA0/k5IKP04eWD9YvEI//LaJP6gDsT99uqU/Jla4PyrNWj9y9pY/fueXP1vz"
    "oT9NHTY/qNY/P+UAUD8hqyo/6MxLPwRLmj+y82g/40NYPyRxNT8emVg/lM2sPzgZfj/a7bc/XFO7P2EvVz+cT1c/pKSCPw94vj+hEx8/Km+RP8YanD+Z5bM/"
    "CZF6Pwadnz+oFLQ/NOq/P/mpHj9Y+3U/7lGIP9Cvrz8qiVU/h1o5P2yctz/qJqg/JpCEP4z4sj9zkW4/ITxLP8ANqD9D/yk/zmiJPzCUjj8yabg//WtaPzyJ"
    "QD8GILw/v+6qP8Z1ET+qwgA/9kWgP0wqlz9VgHs/cJeoP4stYz9ndw8/+y8pP+zHlD98RwY/OMKnP0vPoT9/qms/j6kDP58BiT/d0Hs/6c13P6xHrj908bY/"
    "qG23P6/RWz/0KSM/APVUP8S/Fj8aRyE/qhO7P390XT9zRx0/rRwuP9E1UD9Bs4w/mU4IPzdVMD/6jLA/usOyP/9jHD/gNKM/cvuDPwCRbD9elrg/iA24P89N"
    "bT8b16k/wbAoP+g4XD9co4I/5LFxPwiDbT/cxos/4jpEP3YNsT8CtRQ/qSG8P5QFhT8e/54/7BKIP2aqgj8a4pU/PIkSP+Casz9lYI4/tlgtP2OCVj8dkIo/"
    "XnicP2bVlT8jrmw/3GesP1LuGz8q1ok/ozY6Pxa1TT9Z4I0/4GyvPz4foT9Glo0/8NhbP3CzCT9pCKQ/Ae46PyqpkT+ygIo/hCoiP3OfOT+Gnx8/af40P5Y7"
    "sz+uu0s/Vp4UP8IKrT9CALo/EpOyP0sboz/2ao0/+4EWP+SYgT+s7nc/xIJgP924qz/uxZE/LXUXPx7unj914h0/7rEBP2BYqT8gQwc/cOOuP2qclT+D1oE/"
    "ThFOP5H6CT/qbpw/oM1tP7R2lT989ps/fV6TPzJ9rT9uaJU/KREAP+SNSD8+JZ4/EJhRP17XgD/NqLQ/RJKEP+TQjD8GBqo/2aEeP5BjPj++ooY/VLGsP4ZQ"
    "vj8l9nY/5j24P2CLgz8ej74/+fAhP4jhrz9KtLE/O5YhP4JGrj+Dkwc/04AGP5XXET+C65o/J8ZJP9YRmT98XpM/aNM6PxUdcz9TKro//XEAPzxrNT9f7h4/"
    "BL+KPzgxRz/n9BM/HG29PzJYVT88VTQ/klKdP9q4Wj9KYzM/YZaaP0ftuD/7bHc/ieciPzZjuj8u97c/x9aaP3vAfT9sSFo/SflHPzBtjj9NSh0/s2lePwyK"
    "Dz92PrQ/RDCwP8iAhj+3Pp0/erubPxjNbj85KXg/7oowP1/+gz8Xygc/DaKKP9BkYz8T6pg/2MRhP76oPj/mbEY/ZjKpP6KfvT9PMIM/mJAcP3UoHj/WDVk/"
    "WHdzP8h3hD+WznU/wOi2P1wejD9HeoY/8XVjP151CT9ulqw/i5+bP3FESz/KFyk/zi8+P4CXsj8ko6I/BPkMP7/7ZD8pBq0/KI2FP2GmbT80zYA/eHGjP1bs"
    "sT9I/CQ/kG2qPwwmET/OiJg/avoNP4yIPD8jT5g/EFMOP6Z7jj8+gIE/1MhPPxRRWz9ZDx0/zmmaP/bZtD8WQ6w/K1AKP/WQgz99pqs/Aj8qPxtqOT/8E6M/"
    "aOKsPx0xHD9EPJw/o5lhP72ssD+uhV8/sS6tP/1qnz94aY4/qq92P7nLmj93KKA/m6+VP/hcWj8mDWo/okOHP0thqz+dUFw/Ms6XPwyunj+wm7w/XIAbP0Yb"
    "Cj/anFs/E6wBPyCwrz8qnzw/tLuuP4ruiT+W4Zk/zkSQPx28sD9f7RE/VLqEP02tnz/V7Z4/jtWdP2tPcD9F6kg/665JP1w+sD+oABg/faYVP1hGdD8U2Zc/"
    "ax2OP8WBtD/ghkI/ryiqP5crtz+jf5Q/AREtP95JGD8K5K0/ezE4PyYkTz+QPUA/vtmbP6FVZD/cCJ0/Eh2xP84LBj8fs0o/VkVZP1w/GD8HAzY//3eiPz+L"
    "VT+KnqA/YB2FPwo/uj+k9rQ/VFGcPxbdTD/yj5c/S7S1P25atT9TlYg/xRoMP3zcSz+q378/+oa+P0YKjj+8GAc/TW6UP664gT8C76M/PymkP7vNYD8pObw/"
    "DGYnP5c2kT+85WE/QXkDP7BYcz+NIa8/qV20PwRNWD/OMX0/asokP7PjED9TwR4/QJ2BPxWrnz8lVY8/gqunPxS1qD9z0nY/12+YP7qHuj/epZg/cLGVP+DW"
    "kT9H8HU/p+xBP2BBjz/K8AA/pv1OP8y3tz9qw5Y/9CQ9Pw/ERD9qTJU/gkMlP54lsj/OraI/WoCFP5DFhz+/ZnM/50o5PzlmFz9J1QQ/mHO5Pw2SXD+CP7k/"
    "qyw7P6lmMD8gU18/OzYaPxbMtj/Z+Ws/wXiqP4OKfj9OjQg/tI+SP5CpmT9CXGc/33pdPz5qvT9dMnM/hJGJPxSxjD+kPpo/T0xOP55vBj9Zcpw/4GOLPxDZ"
    "sj8DQx4/vQBfP/46JT9yTV4/chA+P142Lz/ssbw/aDaAP0ClpT/9Ki4/LXiWP1zsOj9f3wA/Vx5FP0Eccj/yb5c/TJqQP0qAED9C8Qs/K1Q7P0BkHz93gS0/"
    "kBqYP3DGrz+i3aI/9wl/P5PTbz+wSKE/WDCJP6iGiT9+m7Y/FIBIPz3ErT/gpn4/rCu4PwKcHj+xBIU/DR89PzGCIT+BHpA/A9hnP9REtT9/FUQ/5jGXP2S7"
    "hD8oyYw/V0SmPw5jNz9n1CQ/1362P30lvD/pO2g/xniSP1ZQkz9+Po4/fC8QP0t5Oz/Tbpg/ZeNUP4GbNz9CgpE/zoaNP2Ikvz9cenw/kf2hPwjEKT86LbM/"
    "Vny9P/RAVz+6iag//aySP7AXNz+W4lw/q89QP1tHfT+OthA/lZ4BP0BGfj9woiU/ORVaP8nGEz+mJ6w/PXoNP3HdZj9mtJE/JTeWP3O2Dj/DKCk/2H+nP6U3"
    "Qz/rAr4/P2BSPwhVej+4sa0/yzshP/40kj/pjWw//laKP7CRrT8MT5c/RFJwP9vpsj8tkoo/TOBSP4aoBj/vzgw/4g2zP7ziqT/NSz0/yhUnP+UIjz/za0k/"
    "0AOHP8p0ez+ho3E/2hNgP5GLAz9YdBk/ageyP06/dj8UZ1Y/TPa8P3wUPT+Xby4/keRXP/vslz8Y/6c/H957P9COlz/1j1w/QfmAP0J1Vj/MSqk/NQJ8P6rC"
    "rj8J5bU/PgFlP07bgT98/jw/q323P5eCED9ZsL8/DGEbPyRrsT8FSVc/UGVMP3Bpkj98W1w/EP+RP0CZiT/mZJE/Kx0PP0GndD+4lYk/8nItP4Y+VD/ij4k/"
    "sMqMP8bjij9xypo/w+5xP1bcXj9rEAo/utVMP3z1cj9UG1U/vfu0PyxWmD8Bl2g/krGLPy4yeD/IWrA/+D2BP+xqIj8kA4o/flahP53GGT+rMVE/F9wTPzSo"
    "mz+PgyU/2LCZP469QT/gTWI/l5FwP5/gYj/SmhY/qlFWPxDblD9CVpc/RH6NPxPGQz9GpbA/R/t3P8+UvD8shWQ/tjOMPzJRtT9AXyw//hx1PwhqmD9P93c/"
    "USumP69/Aj8ohiU/BrusPyb9Ij8wH4U/UMp5PwiPfj+rUhc/SIk5PyUdNj9OJUo/s+EOP0X2rj/66I8/x2ZsP45UWT8254g/OBquP45efj/+dzM/72olP2ND"
    "rT96LYo/+MiTP/yejj+JrG8/LGmjP1AEtD/MHFk/I/IXP0XVvj/Gs6o/Lhm6P2nTez9wzoc/VPabPzwenT+mJr0/ce5BPz8OnT8IPJc/hdGJP/tprT9vfYs/"
    "Ot2MP+mNGT/s/qA/gH6qPxbahj8ie5A/dRskP49jaj+wyqU/5iE7Pxa9Hz9hJI0/VNSMP2ldVz8YVZY/rAoNP8/cOz+xDls/iH6tP96FpT9Neqs/UvwHP/Mv"
    "dj8oxb8/khdNP7aaoz9A5ZU/RtcnP85VrT8c1kA/mPwrPxuhEz+05jM/bb0LP9xttj9EZS4/xr+mP2dahT/q9E4/jJ2RP+Bvjj+hsUo/9rqpP+ZJXz+7IFg/"
    "oDuCP87GJD/eC38/olCUP+Rjqj/SD24/PztKP19hHT9Tvy4/7vmMP5gQtD9c6Yg/ePhnP3rnhj/QXKs/RoKMPxlUrT9JeXk/CoBtP+agpj/EyiI/kaWQP5BB"
    "fT+Jfw8/rwqiPwQmvz+0tWI/4om7Pw05aT/cVIA/i7+EP6LUAz8adKk/BVedP0RChD9mnJw/speqP8IKjT+OkWA/VR9sPzoNCz/oXJ4/rra8P16uWz+lJoE/"
    "3hRkP4qZZj9zfhQ/8CicP+Q2Zj/Up1E/1ti0P6jjvz9Al2A/3q5OP2okkD8m6Yk/yRJXPz7fNT9NRwU/GAVZP5mrtj957Jc/H41sP7MzQD+uZJE/cvZYP27F"
    "nz/VGD0/qF1oP+V+Mj/Bz4s/2hUlPzhmfj/CS1U/hCa6P8PEPT8zWKw/lbS8P8SitD/vPEs/EZYlP5zySD9oe1g/za2HPwSrpD82u08/5KAKP0JMjT+Qfa8/"
    "wT5SP4zPtD+UYBA/d8o8PwXCQD+Ana8/2PO0Pz5ugz/tRVY/r9FQP1rTgD9/Kk4/KDmdP6j1jD/0pD4/Wok2PzW/WT9fuxg/c/U6P8uFtT/YEqY/xICpP0/y"
    "KT+95LE/B8eJPzyqqT+afZ8/7I1/P62BQj8cKLk/LatLP/i+qz8o/rM/jGoSP9CJgj+cAqI/9DcdPySCRz8IoK4/yR9BPyAcnz84hiI/hzuvPw0IQT9Fn6g/"
    "JFOaP+p8uz+erkk/ZCK9PwFzAz+GKa4/8meDP86VRD8kaQE/ZWgiP4CFVj9HO10/4j97P/z8ED+Y0r8//rSZP2jeij8SQhE/8x+mPyBDsT8CuUI/9J01P4Vo"
    "Cz/sOq8/1jlrP4O3oj8gHGk/5FtNP7QvJD+7Tw8/eS4PP80QnT+8loY/ZdJxP2RUUD/4uak/FB2pPxCHoD/TZqw/NCS4P/mzQj9xVDc/zFCMPyQajT+S5Vs/"
    "YIgqP9AvjD+Q6q8/6ASeP5Uaaz9/CxQ/VdKkPxZsNT+E1II/P520P6Lemz8igDc/MvCNP1begD9I7Ks/BHO+P5eNDT86kXM/khwjP6YUTD/oPBg/ITZwP1fs"
    "Sz/OoRM/2n9zP/dpKz9v4rU/EBucPw8Yrz93dwQ/GNqOP6mxOz/gMDs/z0yMP/6ARD+Wdhc/9BZwP+qepj8jFBU/lfGvPx7xuT/0lYQ/hlyzP+bdkz8PAJU/"
    "zk1PP61oYD/seYk/1vwLP31ApD9Ik68/P5yyP+wxJT9abrk/RrKmPxJXFT8f0jI/KDiQPwbhQj9drWw/NiZNP2kbPT/ME5c/OM65P63/Ej93Vyw/mNy2PwR1"
    "PD8D4n0/cCWBP4SHPT+iRDk/y+USPwSqUD8opY8/ucdIP1b6mT+aRls/nsIjP3zWlD/+0b8/YE93PyX4oT+C8aI/oPWeP9lRvz9aNaQ/RNEpP+Gmuj9qe5k/"
    "diiHPyj1KD9o/Z0/w/qdP7UOJj/WI7Y/v2MnP5GlHz9siLU/xyCdP0mJGj91YUE/Aq2mPy6Vuz9nVCI/kDohP9FnZj/Jdo4/BBKaP+UFQj8r+1Q/WaucP1wK"
    "rD+qG5U/Fre/P9jnvD81tAo/tjyJP2rmST8beyo/6GCTPwaPhT+qlaY/27WIP7Gbvz+0iTw/uleaPy6JGT8sVhk/ZhdOP4RBhD+nfoA/twgMP6Hgnj/wiDc/"
    "lHeFP073ez95/zI/6EJMP5M2pD+444o/QZpVP0vcdj8MtIk/HrSNP+/zMj+jjFo/5sCiP6JQnT/uD60/t0u4P5HpID/2F1M/n1QcP+NDmz8NRB0/bsu1P42c"
    "cT9TjJo/KLqmP/rdmD9FNaI/Y+WXP/Y2oj/a7Jg/VUezP5uOez+kYow/Q0EEP7j1oz/Kt4w/WDy3PycSYj9Ri5A/o2B1P0RJPz9PSKc/GNMzP/DTsj/LxXo/"
    "ZH4qP2w9nD+M6L0/imUKP5HnCz9IToY/IDkNP37TNj+Q4aY/3NG7Pzt4Kz8aELk/BypKP6oeND9yGZ4/5qILP80goj+wK7A/hOCoP7D0rj96uJY/PPyGP4aP"
    "nz+ytWQ/imi/P+43Pz8UmLM/GBenPwP1VD+XSmM/vHU5P2rcuz/GhAY/C2AJP8S/pD8nw3s/GX+HP4b+ND+GCUs/kr2mP0k7aT9nL3E/BgRGPzIVOT9by1U/"
    "WNmGP+LghD+GRYQ/pi21P2Y9ST+m+qo/LNi9P7H9mT+2NpQ/4CmeP2kkiz/8U4k/Oq6ZP6ncLD+Tnnk/KYSHP184Hj9kAZk/ZqNnPy59oT9ALqc//4YkP8tE"
    "cj9koqs/8OlcP8JRfD+xH28/dMk4P3MIuz8yuo4/uBSKPwaymT96pEk/vxh7P1DJfT9tpHY/ukqVP3L3gD/6pWo/2jagPxxINz/cJpg/9IBRPwh1qD/2wp8/"
    "vLq6P5igFD+uZaQ/hUGNP3fzKj/Q4zw/swmBP5L9mj97CVc/7jJiPzy/kT+q9Lo/llmzP86Ioj/JEW4/eFB8P01Ogj/fdAE/lds4P8jjuD+X9ZY/1WNxP2Ic"
    "Rz8TOAo/L91UP2gfcj/nMGQ/0FhYP9dGvD9aor0/SNENP3iFUD+g0Vo/6V07P2M6MD9kUTs/AWhKPy7RgT8mWqQ/LL8sP8DRcT84YmA/sdxxP4bfcD+YA4Q/"
    "crlRPzQRMj88hrM/VB+tPyxfXD+8RFI/YRwUPw+UET/sQoE/jaklPz11Gj8SXlA/ST6hP8yVlD9TMg0/xcqTP7SQYz/6X5k/IuOJP3Yrlz9uCbo/Uh4SP980"
    "Dz/adWU/Wz4cPzhpVT+6aLI/wFc4P09etT+ysV8/lKSCP+LMAj9iMac/F1OPP0rqBD9f2XM/fVJFPwLFZD/FvKU/CYhbPzqZAz8Kmqk/Y526P8rTFz8BzEs/"
    "tdpNP8i6tj/MhL8/jrOjP3/8ED84RLU/qKcLPzOOsz/lrHw/JZlZP2a7eT/ziBE/tBOXP8XGAj8KrSk/MjyyP8LnkT/b/TI/VsEkP5SMiz86QqE/zYcsP0wu"
    "rD/RvzY/6+MnP5gEFT9+OzU/9CNbPyx8fj98na8/KEJmP3BwgD/2wHM/FoqfP7jghD/eRJI/SmgNP2/SIT8TvaQ/QaAnP6LgtD8B0J4/ToEKP1w/Oj/BHgs/"
    "R9tVP4MDij+MDIg/85QzP2/Alz8cm60/7L0iPyQ1MT8symI/HeEcPzEBOT/g4rA/U5MXP0ENuD+q+5M/bVh2P9D0qz9yoYM/qzYdP5XzZT9gYpo/PpMBP7oF"
    "mz+mpHE/s7R7P9Z2qj937BU/5E8TPzlMOz9e678/t4GPP/Topz8iYhg/L0cQPyQzoT9YpnY/+6q7P+mCqT/6W0k/9Y13P+FgcT+gyq8/HIe6PyP7oT/6NaQ/"
    "68GMP5XuXD98rhI/qwgYP4Iyij9L4KE/6cCdP7KYGT88xJw/is2+P2vgtT+zmJE/dMaeP3e+Mj+wlzM/M2xbPzInlz+yBYc/uQJ3P7dlKD8E9po/I542P8aU"
    "pT/uUbA/xjYkP478jD+evIM/IJ+cPylplD9p/Uk/480PP2rWqz/bTVs/cqY6P47fhj+yEyE/+dNvPx7jsz/F3KA/Vad8P1iFDz/zr3g/9EOfPypjgD9XkVc/"
    "NOWkP+w3ND+b6UQ/GvqhP95inz/GKJ8/jtuhP5LjCD/82rI/z062PyacgD8iVLE/lpZ8PyRuoT91UGw/04C1Pxlxmj81dyw/I/CSP85aLD+DnmE/Pt4HP5Jf"
    "cD+DIF0/0pm+Pwu/bz+amIY/AxiLP3xksz/T3mY/stIVPx3Pjj8mHkQ/2pYWPznIUT+P7ko/UcOtP8lwIz8yzKg/o7kAPxavpD+qRp4/x+pGP+lybz/3dX8/"
    "lQmpP7efgz9uNZ0/rYl/P+Rksj++DqQ/Ttm7PxnVFD8xNng/56xQP91mnD9zWQI/GLBWP20Juj9GYFA/+IkkP+9hnj+eL5o/qmEDP6N2dz+kwE0/7jGEP21M"
    "rD/Xc18/B6tlP9b9BD+b8yo/SG5PPyiEkT8Il6M/4tSpP96NOj8qGB4/6dSxP9y/bj+GzU8/Mm+SP0cibj8/iSM/uRBxP2MLUT/PwrU/GueOP5J2oj/Yegc/"
    "XlBdP64wtD+Wz7Y//hR7P1rLqz9n6bo/zIosP2DrLj+EtjI/R5YIP5nQBj/9fzc/4JyZP1SyoT9+nJY/TOusP0vjgz9m1Wg/nceIP8oKvT/VxmA/NjUoP5Bh"
    "oj+RaKA/WMMVP7ZSHD+cMQk/b1orP9fuvj82T50/ahywP0qBED9GV2g/qRifP7SMuj+Qvrs/tt6fP2oShD+BL7w/VsWsP6dnOz+VxKU/dHw3P+DnQT9vH70/"
    "Jj5BP5qKKD/2dBg/UAy9P1g/Vz+s+YA/+7QIPxm8Kz8g34E/RbKDP9bxiz/GZKo/QsqSP6v1Fj8Fd4Q/ISNTPwZ5fD9shoQ/l/dQPwR3bj9F9R0/MOEgP9P2"
    "az/O0YI/fq+/P0VLnD819RM/kgGJPzD6pj9QFRM/D8GyP07MiT/c0m8/zjWPP4bFHD84dpU/GLeGPxZHAT/w1FU/ZCK3P3Merj+dq1o/dLIQP8U7ez8wB0M/"
    "2K1XP1h/hj/DRjI/4Y8EP+R3cD/+mCQ/exNCP9Ftjz+AKJ8/+q9TP/kApj9e+K8/NECgP5QVEj+aKq8/Fn6qP3xjOj8oc7k/v55KPxIyLD9EcoQ/MjYcPx5i"
    "iT+piUg/kQsFPz0Tsj8qobM/TleOP1pNkD/wIyA/c1KVP5y3tz/ZbDo/YOsZP84Lpj9uz3A/SZAdP9Wuaz9KIhc/GFsWP7WkZj98qYw/b7srP88ooj9Q8as/"
    "SLJ/P7xZij+KrAQ/NlNCP4PmSj97OgE/KkC9P0xUZD92qXg/2B2aPzSFPj9JwIs/EHwHP4bYnz/ioYc/Rum7Pxyelz+WkAM/PlkbP3LceT+4Smk/OlubP08m"
    "VT8YbKQ/pqW5P8qVuz/4y7Q/iLSfP/RZZT+d4G8/iM1XP10kCD/EDQA/m0d3P5MQTz9A1hU/kE4sP3amcz/nX74/4wgPP/AmuT9CZrU/PC6ZP9KuqD9sBkM/"
    "ep2VP5k8mz9aiZs/j6NhP/ELlT+TQHM/6sO8P4z6kD+RFEY/DVGVP/aHIT8thko/yTYEPy3Ibj/AYno/sRJfP/KBuz8OVjg/0iuwP4GMDz8BZFA/AWdLP6pF"
    "jj9qUZQ/eSNcP4Pndz9y1jc/B6uzP7funj/7hY4/JT51P4ztXz+2ypU/p5a3P+uDZz8/Zn0/KVkIP+gMeD9QWrQ/dmlHP0S/hj+qJKs/eYqKP+pggj9wakk/"
    "5geSPxujUj++QqI/Z9R7P1nXNj/4lmw/+oYkPwqQnz+sM5k/EKORP5xppz+AN20/q96nPzhlOz+s8TM//SlJP/k2bz9SJAg/ZjmYP9h9mT/iU2w/aN8AP6wp"
    "GD+Ab7Q/UpdzP2BYnD+dPBM/B0cJPwokBT/ON6E/d80NP8lzGD8YL78/TOqyP6etjD8qXHI/H3RxP/wZOz/r1bs/r7agP3TIjT/QGGY/G9UiP3/3oT9xi3I/"
    "1o60P7YbVz/UOFQ/Ll4xP2Vobj+16h8/sIAWP5IbDD+3LVk/rkiJPwQkHT+kGGY/KdxFP7C0Xz9GmKc/EXN3P9wLNT8S7Zk/E4ajPxJzUz8ulZY/Z4m5P1CZ"
    "hT86qII/HieIP2ZZOT8kS6k/6nm9P4QPEz9e3mw/pv5yP4l9Nj+s9L8/8jwOP+NQHz/ySZM/48yqP8fcvT/xc1k/VzE8P0LAQj8oLAo/PSSqP+AJmz/usYI/"
    "7PCMP32MKz8S4xM//FZhP0wUkz/cqLY/CD0+P7LLtz/VIDU/r4WfP6DRvD+nnxM/et81PxQujD/ihl8/XP1XPw6Loj8tujo/y8ErP9L4nj9P3Qc/oK9jPwF4"
    "iT/3gq8/heCIPxWuhz+aM74/Kr+OP25NiD9gOrQ/53YfP08YIz+w174/K220P4F3GD+cxhU/oAqFPzJlsT+qK14/LMN+Pylluj9MiEA/kAAjPwEIkT+AsYM/"
    "JywRP8Sikz/046g/StCrP9wTrD9r4VY/WBRAP670nD+GYr4/uUOiP88bCj/Cu4U/xB22P5wASD9EYnk/tkeQP57Iuj8QUb4/Z/sMP17bnD9ou3k/e1YyP8Do"
    "oj9k27g/5jSPP3NHvD+cF5o/zMw4P+32jz8ipIk/rHczPyjNhD/9LT0/uuivP6QHqz/mD78/sD+dP3wJtz/6Y2w/tK6vP4WSJj/QAbw/3BWOP3MdqT9q9qs/"
    "4bp/P/Zkhz/f9r0/CtQkP9KDQD8mUbs/MjudP07Lpz8Y864/m9VUP8cSMD8U6a8/HNyfP46fnj8cjaQ/ObJjP4NRTD8Lr5A/5a2mP+TXdD/hlDU/aqlGP8fJ"
    "VD88Cqc/BwZxPwittT+3jpY/bEhvP4FIdT/mHaI/j0Z0P9F8sz8okbQ/nflBPxURuj/opLg/Fj4RPwSDgj+5MEk/3aBPP553vz8QKgY/xMmyP1TfkT9Orgo/"
    "ZiuXP4lLRz/fJl4/a5eMPzvTnz9ahI4/aS4uP8A7hT8ELZc/tuC1P/Cwvj8G4B4/GoWjPwItQz+IEZU/pz92P5WPrz+TGqw/PTlAP6q3qD+zAzQ/0mNSP1QE"
    "sz/7Xl0/JdYJP8RjaD+5CSw/QVUJP9F+Kz+ify8/3jiZP5Raoz8WEZc/XPaSPyc0FD84cqQ/nwaJP0oBiz/qBII/ZEhdPwHgWj/oKK8/oDmzP4dFUz+3uU8/"
    "go9/P7aFZD9ulzw/cdMRP+xuvz+s6KM/b2YzP0hDET8kD50/QoyhP7kNbT/0OYg/nh4OP/e/fj/xIog/fIg7P7A8mj/kIj4/CfwXPyqpqz9+qKo/WpImPwoL"
    "oT+zTGM/4dpKP9oLBj92AZQ/MeGtP0I/mj+dWJE/sO1JP2crrD+arhQ/DOmjP4Jobz9/Jpg/oSC4P97fcj/EU70/rw+uP4F9kz943wc/Z7JzP3qfIj/clJE/"
    "cs+QP2XzfT/MQKs/EVySP/1gcD/5KGc/7pM3PzzFij98T6I/KYlgP1z2rD+HciE/6l6vP3tTVz9vsaw/W9JuP8CyWT/ADY4/m2pBPy7nsj88qjI/8L60Py1Y"
    "oz9sMTw/n3C/PzUAkD+4C0I/PtkGP+4bvj+JUXg/K4CcP9xxlz/kbVc/s5CPP3llSD/ExA4/QIt5P6/lLT+0HpY/8Tq9P3NNXz8PYxU/tuW4PycIuT86R7k/"
    "YacyP6AUrD/GqKI/rDynP9Zykj+7oHc/AgVBP9OpEj8UGK8/1TNcP7YItz/qoYo/J7pUP1aPsz/8NEc/znKUP/SntT+cibo/YkCRP+syUj8wWWc/MYeEP1hD"
    "oD9yYZM/xYB0Pyo0vD/Q5II/eqgEPyhanz+uL6M/K9iKP4cXUj/bTYk/eUhGP6zUsT/UyVg/vkKGP7Amrj/BqS4/+uQ9P4RyEz98ZbA/O1tdP/Nncz8Arrg/"
    "t5yTP321Zj/Xu7c/IveBPwn5YT+gmXM/7/68P6ZcsT8+7YI/QUK/P42hVj89sVY/ineJP3O1rT8mlpk/ygdbP1m8Sj8ccE4/mt2oP4RFsD/fkVM/dkIcPz3t"
    "qT/C1JM/7ReNP94KbT/HNgs/45WcP/lrbj9oEZs/vzKzPy+NCD9Eg2w/kUR5P55UiD/IAGY/rEmsP3b0jT9bu6w/wKGDP0cBUT843aM/uyl+PyAbhz/sMLg/"
    "6zG9PzAchz+WEDI/LJlPP/Znqz/oUaw/jtWmPzJmUz/8AjU/HgqYPxhROz+xxhk/iKeDP1SPdT9KmXY/DZhsPw7Bvz+02KQ/1gGIP8rruz+is50/mX1mP9hC"
    "Fz9g+qg/6wY+P8Avjz+S7yA/lti1Pxq1lj/+Z0s/VrSoP2ljkj8s0b4/cCkhP5ceST9F5TM/ecW6PySifj/Jio4/5Ol0P26NSz+umr8/7bqwP/iYmT9yXoE/"
    "OZWGP0+Lcz9iup4/ZzAHP+4btD/OoF8/fL1JP9Vobz+i6IA/Ie13P55dmT8KQrw/t2llPz8bHj8hjIs/SIAxP+T6tz93OzU/hHkkP4JjWT/0FDI/IimYP6ey"
    "bj/vYE8/201AP478Hj9mvJs/CDgRPxTQvT9cKjw/THChP6khcj+oxrs/UPNdP31KNz+t/Hw/wMOxP2hrrz/ZqTI/oac1P+lxgT9YDpc/BFU6P0iuqD/1lGw/"
    "36MdP5RwlD88UKs/oGqOP8H+uz/CtI0/zYZ1P1LZrD/LOEA/7CaaP9PEmj8dUEM/IcJDP5I/qT+SBoo/NiqOP6iVsz+bbI4/THKjP5ZSSD/qu3k/gf5xP1fS"
    "qD9vFrk/SBqEPzZAiT9EOpY/wt6CP35Vdj+Eibg/XCeAP4OhFT8kX58/afstP358uT9Xv1Q/IracP5AkjT9f54o/5xUWP3wboz9b/II/sDqBP+0EvD96klo/"
    "S4kEP34vuj8SPrI/UN+SPwZogD9IzKw/qUd5P8InCD8EA4E/UXMmP7UMtD9kqwY/UOmdP7yvJT/WPDU/XSxFP5bbrD912k0/bNWXP5zxhD/zFxQ/XJidP3cK"
    "jD+dOiA/Bo6BP9LzXD9U8Jg/m5MlP5Rxnj9yrXA/Op9/P5MlTj/IoaI/rn0YP3nsoT9VnVs/7ItVPwrPnT/zClg/MY0jPxr2sz9A24E/a/IgP/U5CT+ZFjE/"
    "bMmQP4eekz/8SiQ/VaWKP97/hj9t1Eg/0KmUP0sstD/4OT4/Fk6BP/gkED/sPLw/+VmJP+8BSz/FDz8/YlQfPz0gsz+M75k/5HKJP9dsdj+Y5ks/zCCYP9FA"
    "FD/MiR0/LFAzP0ckmz9yPoM/EyosP7Ssrz/VpHM/eu0FP8T3gz+nBno/cgx/P2YJUj+ldA4/nOstPwkbqj9aVXk/pgOtP3a3fj+sOlI/MHKLP/zykD+OPEI/"
    "LsMxPwbtdj8lALs//WqwP5ISjj/ZoU4/o6odP0nZZT/fgng/V/BlP9nMPT/qd54/qaRzP6qhPz8Gpjg/nv1/P9q1kD9PaYg/kaekP9UMez+f1zA/zN6sPyXp"
    "mD+uY54/G7aTP0BNoD+7ah0/+ouTP5fqIT8OsKc/NmdjPzRiCT/kqbg/oHluP99QFz9Y8o8/wFaKPwpKnD8iDas/NhOXPyTkaT9w6Uo/R4UOP1yYez+UtJI/"
    "dcuMP/jOnT86d7s/0msFP1VYoz8Djno/ffFSP7hDtT8tz7I/xut1P8EXcz/CqF0/Brh6P8EQMD+xp7M/ZxCKP7JVQz8KCoA/MlkQP9msPj+KMlA/CekgP3La"
    "kj+Y0R0/vRWLP9a3rD9+OA8/PB9+PyQbBj/QRDQ/oIakP0L8Gz82rBk/CrStP5x9Xz9X9FU/DhAuP+VuYj/NTos/xkOUP0LrQj+iAL4/4Q1rP/kKcj+GowU/"
    "RRREP0BJeT88x4M/CH8rP0sxaj9ZxB4/PF43P04Ctj/bCgM/QMlnP8C1qD+mZwM/ZQwfP/DMGD+AKYQ/UHKePxE2rz9cgow/5XyNP0UgXj8v6hg//peeP6PY"
    "lz/eEqE/VyhRPwqQgz+0NZk/VmK2P+65jj82No4/K3w4P/2ooz/OuIY/xWgyP+ILMD/EpJk/tdNvP2BEiD/N4GE/RLKrP4gpqT9L9bk/q60rPxTAiD+FfI0/"
    "RAV7PxydaD/7BaQ/AjCCP7KkvT96PB8/wEFbP1pJvD/udLg/JReFP0zmnj+ulKw//85rP6iChz88u7g/BvWhP+gunD/lDjQ/TACyP2SKJz/SIlU/sAosP0Uj"
    "OD8jdqE/6JYSP9N6VD+Eapg/nq9BP7CCmz8vmnQ/sAKmP3oWnz9O7xQ/H+FePxkPtT82xgU/xQKQP03lvz9SPEY/fjiaP9KfaD/133Y/RvmJPwSagD9NAVg/"
    "Se1kP2l8Nj+85z4/yUUhP8yOhD8mfoc/SSBiPwv9qD/gq3s/MnuNP+ZcZj8a150/nJW7P5E+Jz82dbA/KQmPP1ILHz8ITF8/wQs3P30Voz/bsEE/FhaFP0We"
    "Sz+HtUg/g3mvP1tXZT93+H0/47NhP4U/GD99YiE//dREP7jXvj/bZGc/v791P2Jbkj9O3Yc//RaRP01YnD+yZS8/lUkXP+JsvT8ccVI/JrpyP/XveD8H/2o/"
    "eZ4fPx4YTD/puWI/88eLP/6Hmj8uSaE/MiwJP3DYZT9Gjks//cEuP9oNkD9R8EI/Gu24P0WOSz972B4/1e8GPwhrtz8XrT4/YPWwP55kOD+oIIc/4NBXP2wz"
    "lT/dPjU/1OKlP+l5bj8yG0M/7pS9P3+UDD+26I4/ZVmoP8RoqT+InJc/I4IlP7mvpD9MRAw/O4ZFPxb5tj9FO2U/HPRXPyT6jD+qTzw/WjamPxecuT/BnK4/"
    "+DSHPzSiYj9TXjo/FiyWPwJOhT/KHiA/li+CP0SnET8cHHA/SW8hP7GNtj83/gA/3gFuPzxFtD9lUz0/yY6VPwGnkT8MV7Y/3E+nP6Jimz9c17A/vMe4Pz5M"
    "gz+yO7w/YGqKP4Q/gD/h9nk/gI6eP2RTvz8yCoE/ON2NP82jfj8MpJU/3wknP5SMpT+hry8/Vf2jP4nbPD+pAII/qe+FP6bBNz95WhI/eDOTP2OAgT9oKZU/"
    "wPSMP/1eqz+K8J4/mA06PzXugj9in0A/LbGRP8FgJD94JkU/e1tyP5KmMz/WPpU/wBAtPxCeCj+YjWk/YtFSP19eaj80gIc/GtS7P8d6Oz/FpqQ/NotLP1ZQ"
    "pD9oJC4/XBpHP/H+cj+4/aM/KnSuP7rivD8TLa0//jCbP/8eWT8+EJc/D1KrPxzoMz+yFBI/Ze5yP3SatD86el0/VlKEPzq8lD/VH5k/hka+P20vZT8TfLw/"
    "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"
    "Az9jErA/"
    ;

static const float golden_out_0[1] __attribute__((aligned(128))) = { 0x1.0088520000000p+0f };
static float out_0[1] __attribute__((aligned(128)));

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

    for (int i = 0; i < 1; i++) {
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
    total_n += 1;

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
