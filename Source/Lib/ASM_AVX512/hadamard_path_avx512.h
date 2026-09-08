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

#ifndef SVT_AV1_HADAMARD_PATH_AVX512_H_
#define SVT_AV1_HADAMARD_PATH_AVX512_H_

#include <immintrin.h>
#include "definitions.h"

// Strides and pointers in these helpers are in bytes.
SIMD_INLINE void hadamard_v4_avx512(__m512i a[8]) {
    const __m512i b0 = _mm512_add_epi16(a[0], a[1]);
    const __m512i b1 = _mm512_sub_epi16(a[0], a[1]);
    const __m512i b2 = _mm512_add_epi16(a[2], a[3]);
    const __m512i b3 = _mm512_sub_epi16(a[2], a[3]);
    const __m512i b4 = _mm512_add_epi16(a[4], a[5]);
    const __m512i b5 = _mm512_sub_epi16(a[4], a[5]);
    const __m512i b6 = _mm512_add_epi16(a[6], a[7]);
    const __m512i b7 = _mm512_sub_epi16(a[6], a[7]);
    const __m512i c0 = _mm512_add_epi16(b0, b2);
    const __m512i c1 = _mm512_add_epi16(b1, b3);
    const __m512i c2 = _mm512_sub_epi16(b0, b2);
    const __m512i c3 = _mm512_sub_epi16(b1, b3);
    const __m512i c4 = _mm512_add_epi16(b4, b6);
    const __m512i c5 = _mm512_add_epi16(b5, b7);
    const __m512i c6 = _mm512_sub_epi16(b4, b6);
    const __m512i c7 = _mm512_sub_epi16(b5, b7);
    a[0]             = c0;
    a[1]             = c1;
    a[2]             = c2;
    a[3]             = c3;
    a[4]             = c4;
    a[5]             = c5;
    a[6]             = c6;
    a[7]             = c7;
}

SIMD_INLINE void hadamard_v8_avx512(__m512i a[8]) {
    hadamard_v4_avx512(a);
    for (int i = 0; i < 4; ++i) {
        const __m512i x = a[i];
        const __m512i y = a[i + 4];
        a[i]            = _mm512_add_epi16(x, y);
        a[i + 4]        = _mm512_sub_epi16(x, y);
    }
}

// Transpose four independent 8x8 word matrices without crossing 128-bit lanes.
SIMD_INLINE void hadamard_transpose8_avx512(__m512i a[8]) {
    const __m512i b0 = _mm512_unpacklo_epi16(a[0], a[1]);
    const __m512i b1 = _mm512_unpackhi_epi16(a[0], a[1]);
    const __m512i b2 = _mm512_unpacklo_epi16(a[2], a[3]);
    const __m512i b3 = _mm512_unpackhi_epi16(a[2], a[3]);
    const __m512i b4 = _mm512_unpacklo_epi16(a[4], a[5]);
    const __m512i b5 = _mm512_unpackhi_epi16(a[4], a[5]);
    const __m512i b6 = _mm512_unpacklo_epi16(a[6], a[7]);
    const __m512i b7 = _mm512_unpackhi_epi16(a[6], a[7]);
    const __m512i c0 = _mm512_unpacklo_epi32(b0, b2);
    const __m512i c1 = _mm512_unpackhi_epi32(b0, b2);
    const __m512i c2 = _mm512_unpacklo_epi32(b1, b3);
    const __m512i c3 = _mm512_unpackhi_epi32(b1, b3);
    const __m512i c4 = _mm512_unpacklo_epi32(b4, b6);
    const __m512i c5 = _mm512_unpackhi_epi32(b4, b6);
    const __m512i c6 = _mm512_unpacklo_epi32(b5, b7);
    const __m512i c7 = _mm512_unpackhi_epi32(b5, b7);
    a[0]             = _mm512_unpacklo_epi64(c0, c4);
    a[1]             = _mm512_unpackhi_epi64(c0, c4);
    a[2]             = _mm512_unpacklo_epi64(c1, c5);
    a[3]             = _mm512_unpackhi_epi64(c1, c5);
    a[4]             = _mm512_unpacklo_epi64(c2, c6);
    a[5]             = _mm512_unpackhi_epi64(c2, c6);
    a[6]             = _mm512_unpacklo_epi64(c3, c7);
    a[7]             = _mm512_unpackhi_epi64(c3, c7);
}

// First horizontal butterfly, independently on each adjacent pair of words.
static INLINE __m512i hadamard_h1_avx512(__m512i a) {
    const __m512i b = _mm512_shuffle_epi8(a, _mm512_set4_epi32(0x0d0c0f0e, 0x09080b0a, 0x05040706, 0x01000302));
    return _mm512_mask_sub_epi16(_mm512_add_epi16(a, b), 0xaaaaaaaa, b, a);
}

static INLINE __m512i hadamard_h2_avx512(__m512i a) {
    const __m512i b = _mm512_shuffle_epi32(a, 0xb1);
    return _mm512_mask_sub_epi16(_mm512_add_epi16(a, b), 0xcccccccc, b, a);
}

// Pack four rows of a single 8x8 tile into one register for the small path.
static INLINE __m512i hadamard_load8x4_avx512(const uint8_t* src, ptrdiff_t stride, int hbd) {
    if (hbd) {
        __m512i a = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i*)src));
        a         = _mm512_inserti32x4(a, _mm_loadu_si128((const __m128i*)(src + stride)), 1);
        a         = _mm512_inserti32x4(a, _mm_loadu_si128((const __m128i*)(src + 2 * stride)), 2);
        return _mm512_inserti32x4(a, _mm_loadu_si128((const __m128i*)(src + 3 * stride)), 3);
    }
    const __m128i a = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)src),
                                         _mm_loadl_epi64((const __m128i*)(src + stride)));
    const __m128i b = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(src + 2 * stride)),
                                         _mm_loadl_epi64((const __m128i*)(src + 3 * stride)));
    return _mm512_cvtepu8_epi16(_mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1));
}

static INLINE __m512i hadamard_v4_packed_avx512(__m512i a) {
    __m512i b = _mm512_shuffle_i32x4(a, a, 0xb1);
    a         = _mm512_mask_sub_epi16(_mm512_add_epi16(a, b), 0xff00ff00, b, a);
    b         = _mm512_shuffle_i32x4(a, a, 0x4e);
    return _mm512_mask_sub_epi16(_mm512_add_epi16(a, b), 0xffff0000, b, a);
}

static INLINE int hadamard_satd8_packed_avx512(__m512i top, __m512i bottom, int* dc, int hbd) {
    const __m512i ones = _mm512_set1_epi16(1);
    if (dc != NULL) {
        *dc = _mm512_reduce_add_epi32(_mm512_madd_epi16(_mm512_add_epi16(top, bottom), ones));
    }
    top              = hadamard_v4_packed_avx512(top);
    bottom           = hadamard_v4_packed_avx512(bottom);
    const __m512i a  = _mm512_add_epi16(top, bottom);
    const __m512i b  = _mm512_sub_epi16(top, bottom);
    const __m512i x  = _mm512_abs_epi16(hadamard_h2_avx512(hadamard_h1_avx512(a)));
    const __m512i y  = _mm512_abs_epi16(hadamard_h2_avx512(hadamard_h1_avx512(b)));
    const __m512i mx = _mm512_max_epi16(x, _mm512_shuffle_epi32(x, 0x4e));
    const __m512i my = _mm512_max_epi16(y, _mm512_shuffle_epi32(y, 0x4e));
    // Each maximum occurs twice, supplying the factor of two for the fused
    // butterfly without masking out duplicates and rescaling the reduction.
    // Residual inputs can have either sign. The two maxima sum to at most
    // 64*M: 16320 for 8-bit, but 65472 for 10-bit. Only the former can feed
    // signed PMADDWD directly after a word addition.
    const __m512i sum = hbd ? _mm512_add_epi32(_mm512_madd_epi16(mx, ones), _mm512_madd_epi16(my, ones))
                            : _mm512_madd_epi16(_mm512_add_epi16(mx, my), ones);
    return _mm512_reduce_add_epi32(sum);
}

// Sum four dwords independently in each 128-bit lane, replicating each sum.
static INLINE __m512i hadamard_sum_lanes_avx512(__m512i a) {
    a = _mm512_add_epi32(a, _mm512_shuffle_epi32(a, 0xb1));
    return _mm512_add_epi32(a, _mm512_shuffle_epi32(a, 0x4e));
}

// Psy-only reduction: the original samples are unsigned in [0,M]. Every
// non-DC coefficient has balanced signs and magnitude <= 16*M after five
// butterflies. In each four-column half, any two of rows 1..3 have a joint
// absolute bound of 16*M, and all three have a bound of 24*M. Selecting the
// maxima from two halves therefore gives m[1]+m[2]+m[3] <= 32*M: 32736 at
// 10-bit. Together with row 0, the sum fits unsigned words (64*M <= 65472).
// Biased reductions XOR with 0x8000 before signed PMADDWD; the caller adds
// 8*32768 per tile before rounding. All psy shapes share this reduction;
// signed residuals in SATD use a separate, wider reduction.
SIMD_INLINE __m512i hadamard_psy8_sum_avx512(const __m512i a[8], int hbd) {
    __m512i m[4];
    for (int i = 0; i < 4; ++i) {
        m[i] = _mm512_max_epi16(_mm512_abs_epi16(a[i]), _mm512_abs_epi16(a[i + 4]));
    }
    const __m512i ones = _mm512_set1_epi16(1);
    const __m512i pair = _mm512_add_epi16(m[2], m[3]);
    if (hbd) {
        return _mm512_madd_epi16(
            _mm512_xor_si512(_mm512_add_epi16(_mm512_add_epi16(m[0], m[1]), pair), _mm512_set1_epi16(-32768)), ones);
    }
    // All four 8-bit maxima fit together: 4*32*255 = 32640.
    return _mm512_madd_epi16(_mm512_add_epi16(_mm512_add_epi16(m[0], m[1]), pair), ones);
}

static INLINE __m512i hadamard_max_abs_avx512(__m512i a, __m512i b) {
    return _mm512_max_epi32(_mm512_abs_epi32(a), _mm512_abs_epi32(b));
}

#endif // SVT_AV1_HADAMARD_PATH_AVX512_H_
