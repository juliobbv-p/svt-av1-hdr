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
#include "hadamard_path_avx512.h"

// Four horizontal 8x8 tiles share contiguous row loads. Merge the 32x32
// halves with YMM operations so each lane computes a distinct coefficient.
// Compute (a +/- b) >> 2 without overflowing words: (a&b) + ((a^b)>>1)
// is floor((a+b)/2), and ((a^b)>>1) - (~a&b) is floor((a-b)/2).
static INLINE __m256i hadamard_add_shift2_256_avx512(__m256i a, __m256i b) {
    return _mm256_srai_epi16(_mm256_add_epi16(_mm256_and_si256(a, b), _mm256_srai_epi16(_mm256_xor_si256(a, b), 1)), 1);
}

static INLINE __m256i hadamard_sub_shift2_256_avx512(__m256i a, __m256i b) {
    return _mm256_srai_epi16(_mm256_sub_epi16(_mm256_srai_epi16(_mm256_xor_si256(a, b), 1), _mm256_andnot_si256(a, b)),
                             1);
}

SIMD_INLINE void hadamard_lowbd_band_avx512(const uint8_t* src, ptrdiff_t ss, const uint8_t* pred, ptrdiff_t ps,
                                            __m512i q[8]) {
    __m512i a[8];
    for (int i = 0; i < 8; ++i) {
        a[i] = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(src + i * ss))),
                                _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(pred + i * ps))));
    }
    hadamard_v8_avx512(a);
    hadamard_transpose8_avx512(a);
    hadamard_v8_avx512(a);
    for (int i = 0; i < 8; ++i) {
        const __m512i sw = _mm512_shuffle_i32x4(a[i], a[i], 0xb1);
        q[i] = _mm512_srai_epi16(_mm512_mask_sub_epi16(_mm512_add_epi16(a[i], sw), 0xff00ff00, sw, a[i]), 1);
    }
}

SIMD_INLINE int hadamard_lowbd_32x32_avx512(const uint8_t* src, ptrdiff_t ss, const uint8_t* pred, ptrdiff_t ps) {
    __m512i q[4][8];
    for (int band = 0; band < 4; ++band) {
        hadamard_lowbd_band_avx512(src + 8 * band * ss, ss, pred + 8 * band * ps, ps, q[band]);
    }
    __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < 8; ++i) {
        const __m512i a[2] = {_mm512_add_epi16(q[0][i], q[1][i]), _mm512_sub_epi16(q[0][i], q[1][i])};
        const __m512i b[2] = {_mm512_add_epi16(q[2][i], q[3][i]), _mm512_sub_epi16(q[2][i], q[3][i])};
        for (int j = 0; j < 2; ++j) {
            const __m256i al = _mm512_castsi512_si256(a[j]), ar = _mm512_extracti64x4_epi64(a[j], 1);
            const __m256i bl = _mm512_castsi512_si256(b[j]), br = _mm512_extracti64x4_epi64(b[j], 1);
            const __m256i x0 = hadamard_add_shift2_256_avx512(al, ar), x1 = hadamard_sub_shift2_256_avx512(al, ar);
            const __m256i y0 = hadamard_add_shift2_256_avx512(bl, br), y1 = hadamard_sub_shift2_256_avx512(bl, br);
            const __m256i m0 = _mm256_max_epi16(_mm256_abs_epi16(x0), _mm256_abs_epi16(y0));
            const __m256i m1 = _mm256_max_epi16(_mm256_abs_epi16(x1), _mm256_abs_epi16(y1));
            // Each maximum is <= 16320; their sum still fits signed words.
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_add_epi16(m0, m1), _mm256_set1_epi16(1)));
        }
    }
    __m128i total = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    total         = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0x4e));
    total         = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0xb1));
    return _mm_cvtsi128_si32(total) * 2;
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
// Keep four quadrants in ZMM lanes. The first five stages fit signed
// words at 10-bit; PMADDWD produces the final sums/differences as dwords.
static INLINE int hadamard_highbd_16x16_avx512(const uint8_t* src, ptrdiff_t src_stride, const uint8_t* pred,
                                               ptrdiff_t pred_stride) {
    __m512i a[8];
    for (int i = 0; i < 8; ++i) {
        const __m256i s0 = _mm256_loadu_si256((const __m256i*)(src + i * src_stride));
        const __m256i s1 = _mm256_loadu_si256((const __m256i*)(src + (i + 8) * src_stride));
        const __m256i p0 = _mm256_loadu_si256((const __m256i*)(pred + i * pred_stride));
        const __m256i p1 = _mm256_loadu_si256((const __m256i*)(pred + (i + 8) * pred_stride));
        a[i]             = _mm512_sub_epi16(_mm512_inserti64x4(_mm512_castsi256_si512(s0), s1, 1),
                                _mm512_inserti64x4(_mm512_castsi256_si512(p0), p1, 1));
    }
    hadamard_v8_avx512(a);
    hadamard_transpose8_avx512(a);
    hadamard_v4_avx512(a);
    const __m512i plus  = _mm512_set1_epi16(1);
    const __m512i minus = _mm512_set1_epi32((int)0xffff0001);
    __m256i       sum   = _mm256_setzero_si256();
    for (int i = 0; i < 4; ++i) {
        const __m512i lo   = _mm512_unpacklo_epi16(a[i], a[i + 4]);
        const __m512i hi   = _mm512_unpackhi_epi16(a[i], a[i + 4]);
        __m512i       b[4] = {_mm512_madd_epi16(lo, plus),
                              _mm512_madd_epi16(hi, plus),
                              _mm512_madd_epi16(lo, minus),
                              _mm512_madd_epi16(hi, minus)};
        for (int j = 0; j < 4; ++j) {
            const __m512i swap = _mm512_shuffle_i32x4(b[j], b[j], 0xb1);
            const __m512i x = _mm512_srai_epi32(_mm512_mask_sub_epi32(_mm512_add_epi32(b[j], swap), 0xf0f0, swap, b[j]),
                                                1);
            // Reduce only the distinct half of the final butterfly;
            // apply the factor of two after the horizontal reduction.
            sum = _mm256_add_epi32(sum,
                                   _mm256_max_epi32(_mm256_abs_epi32(_mm512_castsi512_si256(x)),
                                                    _mm256_abs_epi32(_mm512_extracti64x4_epi64(x, 1))));
        }
    }
    __m128i total = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    total         = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0x4e));
    total         = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0xb1));
    return _mm_cvtsi128_si32(total) * 2;
}

SIMD_INLINE void hadamard_highbd_8x32_avx512(const uint8_t* src, ptrdiff_t src_stride, const uint8_t* pred,
                                             ptrdiff_t pred_stride, __m512i* coeff) {
    __m512i a[8];
    for (int i = 0; i < 8; ++i) {
        a[i] = _mm512_sub_epi16(_mm512_loadu_si512(src + i * src_stride), _mm512_loadu_si512(pred + i * pred_stride));
    }
    hadamard_v8_avx512(a);
    hadamard_transpose8_avx512(a);
    hadamard_v4_avx512(a);
    const __m512i plus  = _mm512_set1_epi16(1);
    const __m512i minus = _mm512_set1_epi32((int)0xffff0001);
    for (int i = 0; i < 4; ++i) {
        const __m512i lo   = _mm512_unpacklo_epi16(a[i], a[i + 4]);
        const __m512i hi   = _mm512_unpackhi_epi16(a[i], a[i + 4]);
        const __m512i b[4] = {_mm512_madd_epi16(lo, plus),
                              _mm512_madd_epi16(hi, plus),
                              _mm512_madd_epi16(lo, minus),
                              _mm512_madd_epi16(hi, minus)};
        for (int j = 0; j < 4; ++j) {
            const __m512i swap = _mm512_shuffle_i32x4(b[j], b[j], 0xb1);
            coeff[4 * i + j]   = _mm512_srai_epi32(
                _mm512_mask_sub_epi32(_mm512_add_epi32(b[j], swap), 0xf0f0, swap, b[j]), 1);
        }
    }
}

static INLINE int hadamard_highbd_32x32_avx512(const uint8_t* src, ptrdiff_t src_stride, const uint8_t* pred,
                                               ptrdiff_t pred_stride) {
    __m512i q[4][16];
    for (int band = 0; band < 4; ++band) {
        hadamard_highbd_8x32_avx512(
            src + 8 * band * src_stride, src_stride, pred + 8 * band * pred_stride, pred_stride, q[band]);
    }
    __m512i sum0 = _mm512_setzero_si512(), sum1 = _mm512_setzero_si512();
    // The horizontal 16x16 merge and its >> 1 are already in q. Complete
    // the vertical merge, then combine the left/right 16x16 columns (the
    // two 256-bit halves) with >> 2. Fuse the final top/bottom butterfly.
    for (int i = 0; i < 16; ++i) {
        const __m512i a[2] = {_mm512_add_epi32(q[0][i], q[1][i]), _mm512_sub_epi32(q[0][i], q[1][i])};
        const __m512i b[2] = {_mm512_add_epi32(q[2][i], q[3][i]), _mm512_sub_epi32(q[2][i], q[3][i])};
        for (int j = 0; j < 2; ++j) {
            const __m512i as = _mm512_shuffle_i32x4(a[j], a[j], 0x4e);
            const __m512i bs = _mm512_shuffle_i32x4(b[j], b[j], 0x4e);
            const __m512i x = _mm512_srai_epi32(_mm512_mask_sub_epi32(_mm512_add_epi32(a[j], as), 0xff00, as, a[j]), 2);
            const __m512i y = _mm512_srai_epi32(_mm512_mask_sub_epi32(_mm512_add_epi32(b[j], bs), 0xff00, bs, b[j]), 2);
            if (j == 0) {
                sum0 = _mm512_add_epi32(sum0, hadamard_max_abs_avx512(x, y));
            } else {
                sum1 = _mm512_add_epi32(sum1, hadamard_max_abs_avx512(x, y));
            }
        }
    }
    return _mm512_reduce_add_epi32(_mm512_add_epi32(sum0, sum1)) * 2;
}

#endif

// Reuse AVX2 where its small transforms are competitive. Keep the larger
// 10-bit and 32x32 transforms below in AVX-512.
int svt_av1_hadamard_satd_4x4_avx512(const uint8_t* src, ptrdiff_t ss, const uint8_t* pred, ptrdiff_t ps) {
    return svt_av1_hadamard_satd_4x4_avx2(src, ss, pred, ps);
}

int svt_av1_hadamard_satd_8x8_avx512(const uint8_t* src, ptrdiff_t ss, const uint8_t* pred, ptrdiff_t ps) {
    return svt_av1_hadamard_satd_8x8_avx2(src, ss, pred, ps);
}

int svt_av1_hadamard_satd_16x16_avx512(const uint8_t* src, ptrdiff_t ss, const uint8_t* pred, ptrdiff_t ps) {
    return svt_av1_hadamard_satd_16x16_avx2(src, ss, pred, ps);
}

int svt_av1_hadamard_satd_32x32_avx512(const uint8_t* src, ptrdiff_t src_stride, const uint8_t* pred,
                                       ptrdiff_t pred_stride) {
    return hadamard_lowbd_32x32_avx512(src, src_stride, pred, pred_stride);
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
int svt_av1_highbd_hadamard_satd_4x4_avx512(const uint16_t* src, ptrdiff_t ss, const uint16_t* pred, ptrdiff_t ps) {
    return svt_av1_highbd_hadamard_satd_4x4_avx2(src, ss, pred, ps);
}

int svt_av1_highbd_hadamard_satd_8x8_avx512(const uint16_t* src, ptrdiff_t ss, const uint16_t* pred, ptrdiff_t ps) {
    const __m512i top    = _mm512_sub_epi16(hadamard_load8x4_avx512((const uint8_t*)src, 2 * ss, 1),
                                         hadamard_load8x4_avx512((const uint8_t*)pred, 2 * ps, 1));
    const __m512i bottom = _mm512_sub_epi16(hadamard_load8x4_avx512((const uint8_t*)(src + 4 * ss), 2 * ss, 1),
                                            hadamard_load8x4_avx512((const uint8_t*)(pred + 4 * ps), 2 * ps, 1));
    return hadamard_satd8_packed_avx512(top, bottom, NULL, 1);
}

int svt_av1_highbd_hadamard_satd_16x16_avx512(const uint16_t* src, ptrdiff_t ss, const uint16_t* pred, ptrdiff_t ps) {
    return hadamard_highbd_16x16_avx512((const uint8_t*)src, 2 * ss, (const uint8_t*)pred, 2 * ps);
}

int svt_av1_highbd_hadamard_satd_32x32_avx512(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                              ptrdiff_t pred_stride) {
    return hadamard_highbd_32x32_avx512((const uint8_t*)src, src_stride * 2, (const uint8_t*)pred, pred_stride * 2);
}

#endif
