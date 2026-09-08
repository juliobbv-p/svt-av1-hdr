/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "common_dsp_rtcd.h"
#include "hadamard_path_avx2.h"

// Exact floor((a +/- b)/4) without overflowing signed words. The shared XOR
// isolates differing bits, leaving independent add/sub branches to schedule.
static INLINE __m256i hadamard_add_shift2_avx2(__m256i a, __m256i b) {
    return _mm256_srai_epi16(_mm256_add_epi16(_mm256_and_si256(a, b), _mm256_srai_epi16(_mm256_xor_si256(a, b), 1)), 1);
}

static INLINE __m256i hadamard_sub_shift2_avx2(__m256i a, __m256i b) {
    return _mm256_srai_epi16(_mm256_sub_epi16(_mm256_srai_epi16(_mm256_xor_si256(a, b), 1), _mm256_andnot_si256(a, b)),
                             1);
}

// Stream eight-row bands in source order. The horizontal 16x16 merge is
// already in q; finish its vertical merge immediately before the 32x32 merge.
// This avoids materializing four complete 16x16 transforms. Scratch storage
// is 2 KiB for 8-bit and 4 KiB for 10-bit, allocated by the public wrappers.
SIMD_INLINE int hadamard_32x32_avx2(const uint8_t* src, ptrdiff_t ss, const uint8_t* pred, ptrdiff_t ps, __m256i* coeff,
                                    int hbd) {
    const int count = hbd ? 16 : 8;
    __m256i*  q[4][2];
    for (int band = 0; band < 4; ++band) {
        for (int col = 0; col < 2; ++col) {
            q[band][col] = coeff + (2 * band + col) * count;
            hadamard_band_avx2(src + 8 * band * ss + (16 * col << hbd),
                               ss,
                               pred + 8 * band * ps + (16 * col << hbd),
                               ps,
                               q[band][col],
                               hbd);
        }
    }
    __m256i sum0 = _mm256_setzero_si256(), sum1 = _mm256_setzero_si256();
    for (int i = 0; i < (hbd ? 16 : 8); ++i) {
        for (int j = 0; j < 2; ++j) {
            if (hbd) {
                const __m256i a0 = j ? _mm256_sub_epi32(q[0][0][i], q[1][0][i])
                                     : _mm256_add_epi32(q[0][0][i], q[1][0][i]);
                const __m256i a1 = j ? _mm256_sub_epi32(q[0][1][i], q[1][1][i])
                                     : _mm256_add_epi32(q[0][1][i], q[1][1][i]);
                const __m256i b0 = j ? _mm256_sub_epi32(q[2][0][i], q[3][0][i])
                                     : _mm256_add_epi32(q[2][0][i], q[3][0][i]);
                const __m256i b1 = j ? _mm256_sub_epi32(q[2][1][i], q[3][1][i])
                                     : _mm256_add_epi32(q[2][1][i], q[3][1][i]);
                const __m256i x0 = _mm256_srai_epi32(_mm256_add_epi32(a0, a1), 2);
                const __m256i x1 = _mm256_srai_epi32(_mm256_sub_epi32(a0, a1), 2);
                const __m256i y0 = _mm256_srai_epi32(_mm256_add_epi32(b0, b1), 2);
                const __m256i y1 = _mm256_srai_epi32(_mm256_sub_epi32(b0, b1), 2);
                sum0             = _mm256_add_epi32(sum0, _mm256_max_epi32(_mm256_abs_epi32(x0), _mm256_abs_epi32(y0)));
                sum1             = _mm256_add_epi32(sum1, _mm256_max_epi32(_mm256_abs_epi32(x1), _mm256_abs_epi32(y1)));
            } else {
                const __m256i a0 = j ? _mm256_sub_epi16(q[0][0][i], q[1][0][i])
                                     : _mm256_add_epi16(q[0][0][i], q[1][0][i]);
                const __m256i a1 = j ? _mm256_sub_epi16(q[0][1][i], q[1][1][i])
                                     : _mm256_add_epi16(q[0][1][i], q[1][1][i]);
                const __m256i b0 = j ? _mm256_sub_epi16(q[2][0][i], q[3][0][i])
                                     : _mm256_add_epi16(q[2][0][i], q[3][0][i]);
                const __m256i b1 = j ? _mm256_sub_epi16(q[2][1][i], q[3][1][i])
                                     : _mm256_add_epi16(q[2][1][i], q[3][1][i]);
                const __m256i x0 = hadamard_add_shift2_avx2(a0, a1);
                const __m256i x1 = hadamard_sub_shift2_avx2(a0, a1);
                const __m256i y0 = hadamard_add_shift2_avx2(b0, b1);
                const __m256i y1 = hadamard_sub_shift2_avx2(b0, b1);
                const __m256i m0 = _mm256_max_epi16(_mm256_abs_epi16(x0), _mm256_abs_epi16(y0));
                const __m256i m1 = _mm256_max_epi16(_mm256_abs_epi16(x1), _mm256_abs_epi16(y1));
                sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(_mm256_add_epi16(m0, m1), _mm256_set1_epi16(1)));
            }
        }
    }
    return hadamard_reduce_avx2(_mm256_add_epi32(sum0, sum1)) * 2;
}

int svt_av1_hadamard_satd_4x4_avx2(const uint8_t* src, ptrdiff_t ss, const uint8_t* pred, ptrdiff_t ps) {
    return hadamard_small_satd_avx2(src, ss, pred, ps, 4, 0);
}

int svt_av1_hadamard_satd_8x8_avx2(const uint8_t* src, ptrdiff_t ss, const uint8_t* pred, ptrdiff_t ps) {
    return hadamard_small_satd_avx2(src, ss, pred, ps, 8, 0);
}

int svt_av1_hadamard_satd_16x16_avx2(const uint8_t* src, ptrdiff_t ss, const uint8_t* pred, ptrdiff_t ps) {
    __m256i coeff[2][8];
    return hadamard_16x16_avx2(src, ss, pred, ps, coeff[0], coeff[1], 0);
}

int svt_av1_hadamard_satd_32x32_avx2(const uint8_t* src, ptrdiff_t ss, const uint8_t* pred, ptrdiff_t ps) {
    __m256i coeff[64];
    return hadamard_32x32_avx2(src, ss, pred, ps, coeff, 0);
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
int svt_av1_highbd_hadamard_satd_4x4_avx2(const uint16_t* src, ptrdiff_t ss, const uint16_t* pred, ptrdiff_t ps) {
    return hadamard_small_satd_avx2((const uint8_t*)src, 2 * ss, (const uint8_t*)pred, 2 * ps, 4, 1);
}

int svt_av1_highbd_hadamard_satd_8x8_avx2(const uint16_t* src, ptrdiff_t ss, const uint16_t* pred, ptrdiff_t ps) {
    return hadamard_small_satd_avx2((const uint8_t*)src, 2 * ss, (const uint8_t*)pred, 2 * ps, 8, 1);
}

int svt_av1_highbd_hadamard_satd_16x16_avx2(const uint16_t* src, ptrdiff_t ss, const uint16_t* pred, ptrdiff_t ps) {
    __m256i coeff[2][16];
    return hadamard_16x16_avx2((const uint8_t*)src, 2 * ss, (const uint8_t*)pred, 2 * ps, coeff[0], coeff[1], 1);
}

int svt_av1_highbd_hadamard_satd_32x32_avx2(const uint16_t* src, ptrdiff_t ss, const uint16_t* pred, ptrdiff_t ps) {
    __m256i coeff[128];
    return hadamard_32x32_avx2((const uint8_t*)src, 2 * ss, (const uint8_t*)pred, 2 * ps, coeff, 1);
}
#endif
