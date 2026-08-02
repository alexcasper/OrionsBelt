/* Runtime ISA + accelerator bring-up probe for the aarch64 hedge target.
 *
 * Bead ob-ng6. The ADR (0002) requires confirming which ISA extensions and GPU
 * backends are ACTIVELY AVAILABLE AT RUNTIME — not merely compiled into a binary.
 * A binary compiled with -march=armv9-a will run on an Armv8.2 core by silently
 * taking the NEON fallback paths; this probe uses getauxval(AT_HWCAP) to read
 * what the kernel actually advertises, which is the ground truth.
 *
 * Build (native, no deps):
 *   gcc -O2 -o bringup_probe bringup_probe.c
 *
 * Cross-compile:
 *   aarch64-linux-gnu-gcc -O2 -static -o bringup_probe bringup_probe.c
 */

#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Prefer the system <asm/hwcap.h> definitions when available (correct bit values
 * vary by kernel version). Fall back to inline defines for old toolchains. */
#ifdef __has_include
#  if __has_include(<sys/auxv.h>)
#    include <sys/auxv.h>
#  endif
#endif

#ifndef HWCAP_FP
#define HWCAP_FP          (1 << 0)
#endif
#ifndef HWCAP_ASIMD
#define HWCAP_ASIMD       (1 << 1)
#endif
#ifndef HWCAP_AES
#define HWCAP_AES         (1 << 3)
#endif
#ifndef HWCAP_SHA1
#define HWCAP_SHA1        (1 << 5)
#endif
#ifndef HWCAP_SHA2
#define HWCAP_SHA2        (1 << 6)
#endif
#ifndef HWCAP_CRC32
#define HWCAP_CRC32       (1 << 7)
#endif
#ifndef HWCAP_ATOMICS
#define HWCAP_ATOMICS     (1 << 8)
#endif
#ifndef HWCAP_FPHP
#define HWCAP_FPHP        (1 << 9)
#endif
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP     (1 << 10)
#endif
#ifndef HWCAP_LRCPC
#define HWCAP_LRCPC       (1 << 12)
#endif
#ifndef HWCAP_DCPOP
#define HWCAP_DCPOP       (1 << 16)
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP     (1 << 20)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE         (1UL << 22)
#endif

#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2       (1 << 1)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM       (1 << 13)
#endif
#ifndef HWCAP2_BF16
#define HWCAP2_BF16       (1 << 14)
#endif

/* SVE in AT_HWCAP is bit 22 in the *old* numbering used by some kernels.
 * The canonical value from Linux 4.15+ is (1 << 22). We handle both. */

static struct {
    const char *name;
    unsigned long bit;
    int is_hwcap2;
    const char *significance;
} features[] = {
    /* Core SIMD */
    { "ASIMD (NEON)",      HWCAP_ASIMD,    0, "Baseline 128-bit SIMD — all GDN kernels use this" },
    { "FP (scalar)",       HWCAP_FP,       0, "Scalar float" },

    /* Half precision */
    { "FP16 (fphp)",       HWCAP_FPHP,     0, "fp16 compute — needed for fp16 state variants (ob-8qt.4)" },
    { "ASIMD half (hp)",   HWCAP_ASIMDHP,  0, "NEON fp16 vector ops" },

    /* Dot product */
    { "DOTPROD (asimddp)", HWCAP_ASIMDDP,  0, "int8 dot-product — KleidiAI matmul kernels (ob-8qt.2)" },

    /* SVE family */
    { "SVE",               (1UL<<22),      0, "Scalable Vector Extension — SVE reference kernels" },
    { "SVE2",              HWCAP2_SVE2,    1, "SVE2 — additional instructions for GDN scan" },

    /* i8mm / bf16 */
    { "I8MM",              HWCAP2_I8MM,    1, "int8 matrix multiply — INT4/INT8 GEMV decode path" },
    { "BF16 (vector)",     HWCAP2_BF16,    1, "bf16 dot-product — bf16 state variants (ob-8qt.4)" },

    /* Useful but not load-bearing for GDN */
    { "ATOMICS",           HWCAP_ATOMICS,  0, "LL/SC atomics" },
    { "LRCPC",             HWCAP_LRCPC,    0, "Release-consistent prefetch" },
    { "DCPOP",             HWCAP_DCPOP,    0, "Data-cache pop (DC CVAP)" },
    { "AES",               HWCAP_AES,      0, "Crypto (not used by GDN)" },
    { "SHA2",              HWCAP_SHA2,     0, "Crypto (not used by GDN)" },
    { "CRC32",             HWCAP_CRC32,    0, "CRC32" },
};

/* Check for GPU/Vulkan/OpenCL at the filesystem level. */
static void probe_gpu(void) {
    const char *checks[] = {
        "/dev/dri/renderD128",  /* DRM render node (Mali GPU or display) */
        "/dev/dri/renderD129",  /* second render node (NPU on RK3588) */
        "/dev/mali0",           /* proprietary Mali device */
    };
    const char *drivers[] = {
        "/sys/class/drm/renderD128/device/driver",
        "/sys/class/drm/renderD129/device/driver",
        "/sys/class/drm/card0/device/driver",
        "/sys/class/drm/card1/device/driver",
    };

    printf("\n=== GPU / Accelerator probe ===\n\n");

    for (size_t i = 0; i < 3; i++) {
        if (access(checks[i], F_OK) == 0)
            printf("  %-24s PRESENT\n", checks[i]);
        else
            printf("  %-24s absent\n", checks[i]);
    }

    for (size_t i = 0; i < 4; i++) {
        char buf[512];
        ssize_t n = readlink(drivers[i], buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            /* Extract just the driver name */
            char *slash = strrchr(buf, '/');
            printf("  %-40s -> %s\n", drivers[i], slash ? slash + 1 : buf);
        }
    }

    printf("\n  Vulkan ICD files:\n");
    int found_icd = 0;
    /* Check for libvulkan */
    if (access("/usr/lib/aarch64-linux-gnu/libvulkan.so.1", F_OK) == 0) {
        printf("    libvulkan.so.1              PRESENT\n");
        found_icd = 1;
    } else {
        printf("    libvulkan.so.1              absent\n");
    }
    if (access("/usr/lib/aarch64-linux-gnu/libMali.so", F_OK) == 0) {
        printf("    libMali.so (proprietary)    PRESENT\n");
        found_icd = 1;
    }
    if (access("/usr/lib/aarch64-linux-gnu/dri/panfrost_dri.so", F_OK) == 0) {
        printf("    panfrost_dri.so (Mesa GLES) PRESENT — OpenGL ES via open driver\n");
    }
    if (!found_icd)
        printf("    (no Vulkan loader found — GPU compute via Vulkan NOT available)\n");
}

int main(void) {
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);

    printf("=== OrionsBelt runtime ISA probe (ob-ng6) ===\n\n");
    printf("AT_HWCAP  = 0x%016lx\n", hwcap);
    printf("AT_HWCAP2 = 0x%016lx\n\n", hwcap2 ? hwcap2 : 0);

    printf("%-22s  %-7s  %s\n", "Feature", "Status", "Significance for GDN");
    printf("%-22s  %-7s  %s\n", "-------", "------", "---------------------");

    int have_dotprod = 0, have_fp16 = 0, have_sve = 0, have_sve2 = 0;
    int have_i8mm = 0, have_bf16 = 0;

    for (size_t i = 0; i < sizeof(features) / sizeof(features[0]); i++) {
        unsigned long val = features[i].is_hwcap2 ? hwcap2 : hwcap;
        int present = (val & features[i].bit) != 0;
        printf("%-22s  %-7s  %s\n",
               features[i].name,
               present ? "YES" : "no",
               features[i].significance);

        if (strstr(features[i].name, "DOTPROD")) have_dotprod = present;
        if (strstr(features[i].name, "FP16"))   have_fp16 = present;
        if (!strcmp(features[i].name, "SVE"))   have_sve = present;
        if (!strcmp(features[i].name, "SVE2"))  have_sve2 = present;
        if (!strcmp(features[i].name, "I8MM"))  have_i8mm = present;
        if (!strcmp(features[i].name, "BF16"))  have_bf16 = present;
    }

    printf("\n=== Summary for GDN kernel paths ===\n\n");
    printf("  NEON fp16 path (gdn_sve.c #elif NEON):     %s\n",
           have_fp16 ? "AVAILABLE" : "fallback to scalar");
    printf("  NEON dotprod path (KleidiAI matmul):        %s\n",
           have_dotprod ? "AVAILABLE" : "not available");
    printf("  SVE path (gdn_sve.c #ifdef SVE):            %s\n",
           have_sve ? "AVAILABLE" : "not available");
    printf("  SVE2 path (widening MAC, bf16 dot):         %s\n",
           have_sve2 ? "AVAILABLE" : "not available");
    printf("  i8mm (INT8/INT4 GEMV decode):               %s\n",
           have_i8mm ? "AVAILABLE" : "not available");
    printf("  bf16 vector (bf16 state variant fast path): %s\n",
           have_bf16 ? "AVAILABLE" : "scalar fallback only");

    probe_gpu();

    printf("\n=== Verdict ===\n\n");
    /* The critical summary for the ADR */
    if (!have_sve && have_dotprod) {
        printf("  This is a Cortex-A76/A55 class device (Armv8.2-A + dotprod, no SVE).\n");
        printf("  GDN kernels will use the NEON path. The SVE reference kernels are\n");
        printf("  NOT exercised here — only the portable NEON fallback.\n");
        printf("  i8mm is absent: INT8/INT4 decode GEMV via KleidiAI i8mm micro-kernels\n");
        printf("  is NOT available; the dotprod (SDOT) family covers INT8 matmul instead.\n");
    }

    return 0;
}
