#include "instrset.h"

#ifdef VCL_NAMESPACE
namespace VCL_NAMESPACE {
#endif

#ifdef DFTTEST_X86
// Define interface to xgetbv instruction
static inline uint64_t xgetbv(int ctr) {
#if (defined(_MSC_FULL_VER) && _MSC_FULL_VER >= 160040000) || (defined(__INTEL_COMPILER) && __INTEL_COMPILER >= 1200)
    return uint64_t(_xgetbv(ctr));
#elif defined(__GNUC__) || defined(__clang__)
    uint32_t a, d;
    __asm("xgetbv" : "=a"(a), "=d"(d) : "c"(ctr) : );
    return a | (uint64_t(d) << 32);
#else
    uint32_t a, d;
    __asm {
        mov ecx, ctr
        _emit 0x0f
        _emit 0x01
        _emit 0xd0
        mov a, eax
        mov d, edx
    }
    return a | (uint64_t(d) << 32);
#endif
}
#endif

int instrset_detect(void) {
    static int iset = -1;
    if (iset >= 0) {
        return iset;
    }
    iset = 0;

#if defined(__aarch64__) || defined(__arm__)
    // Assume NEON support on ARM
    iset = 6;  // Simulate support for SSE4.2 using NEON
    return iset;
#else
    int abcd[4] = {0, 0, 0, 0};
    cpuid(abcd, 0);
    if (abcd[0] == 0) return iset;
    cpuid(abcd, 1);
    if ((abcd[3] & (1 << 0)) == 0) return iset;
    if ((abcd[3] & (1 << 23)) == 0) return iset;
    if ((abcd[3] & (1 << 15)) == 0) return iset;
    if ((abcd[3] & (1 << 24)) == 0) return iset;
    if ((abcd[3] & (1 << 25)) == 0) return iset;
    iset = 1;
    if ((abcd[3] & (1 << 26)) == 0) return iset;
    iset = 2;
    if ((abcd[2] & (1 << 0)) == 0) return iset;
    iset = 3;
    if ((abcd[2] & (1 << 9)) == 0) return iset;
    iset = 4;
    if ((abcd[2] & (1 << 19)) == 0) return iset;
    iset = 5;
    if ((abcd[2] & (1 << 23)) == 0) return iset;
    if ((abcd[2] & (1 << 20)) == 0) return iset;
    iset = 6;
    if ((abcd[2] & (1 << 27)) == 0) return iset;
    if ((xgetbv(0) & 6) != 6) return iset;
    if ((abcd[2] & (1 << 28)) == 0) return iset;
    iset = 7;
    cpuid(abcd, 7);
    if ((abcd[1] & (1 << 5)) == 0) return iset;
    iset = 8;
    if ((abcd[1] & (1 << 16)) == 0) return iset;
    cpuid(abcd, 0xD);
    if ((abcd[0] & 0x60) != 0x60) return iset;
    iset = 9;
    cpuid(abcd, 7);
    if ((abcd[1] & (1 << 31)) == 0) return iset;
    if ((abcd[1] & 0x40020000) != 0x40020000) return iset;
    iset = 10;
#endif
    return iset;
}

bool hasFMA3(void) {
    if (instrset_detect() < 7) return false;
    int abcd[4];
    cpuid(abcd, 1);
    return ((abcd[2] & (1 << 12)) != 0);
}

bool hasFMA4(void) {
    if (instrset_detect() < 7) return false;
    int abcd[4];
    cpuid(abcd, 0x80000001);
    return ((abcd[2] & (1 << 16)) != 0);
}

bool hasXOP(void) {
    if (instrset_detect() < 7) return false;
    int abcd[4];
    cpuid(abcd, 0x80000001);
    return ((abcd[2] & (1 << 11)) != 0);
}

bool hasF16C(void) {
    if (instrset_detect() < 7) return false;
    int abcd[4];
    cpuid(abcd, 1);
    return ((abcd[2] & (1 << 29)) != 0);
}

bool hasAVX512ER(void) {
    if (instrset_detect() < 9) return false;
    int abcd[4];
    cpuid(abcd, 7);
    return ((abcd[1] & (1 << 27)) != 0);
}

bool hasAVX512VBMI(void) {
    if (instrset_detect() < 10) return false;
    int abcd[4];
    cpuid(abcd, 7);
    return ((abcd[2] & (1 << 1)) != 0);
}

bool hasAVX512VBMI2(void) {
    if (instrset_detect() < 10) return false;
    int abcd[4];
    cpuid(abcd, 7);
    return ((abcd[2] & (1 << 6)) != 0);
}

#ifdef VCL_NAMESPACE
}
#endif