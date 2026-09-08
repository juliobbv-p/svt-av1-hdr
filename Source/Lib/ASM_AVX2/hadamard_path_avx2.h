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

#ifndef SVT_AV1_HADAMARD_PATH_AVX2_H_
#define SVT_AV1_HADAMARD_PATH_AVX2_H_

#include <immintrin.h>
#include "definitions.h"

// Strides and pointers in these helpers are in bytes.
// Saturating subtraction is exact in these stages: the largest 10-bit
// intermediate is 32*1023 = 32736. It prevents compiler reassociation of
// paired differences into longer dependency chains. The short psy path
// selects ordinary subtraction, which schedules better for that layout.
SIMD_INLINE void hadamard_v4_avx2(__m256i a[8], int preserve_pairs) {
    const __m256i b0 = _mm256_add_epi16(a[0], a[1]);
    const __m256i b1 = preserve_pairs ? _mm256_subs_epi16(a[0], a[1]) : _mm256_sub_epi16(a[0], a[1]);
    const __m256i b2 = _mm256_add_epi16(a[2], a[3]);
    const __m256i b3 = preserve_pairs ? _mm256_subs_epi16(a[2], a[3]) : _mm256_sub_epi16(a[2], a[3]);
    const __m256i b4 = _mm256_add_epi16(a[4], a[5]);
    const __m256i b5 = preserve_pairs ? _mm256_subs_epi16(a[4], a[5]) : _mm256_sub_epi16(a[4], a[5]);
    const __m256i b6 = _mm256_add_epi16(a[6], a[7]);
    const __m256i b7 = preserve_pairs ? _mm256_subs_epi16(a[6], a[7]) : _mm256_sub_epi16(a[6], a[7]);
    const __m256i c0 = _mm256_add_epi16(b0, b2);
    const __m256i c1 = _mm256_add_epi16(b1, b3);
    const __m256i c2 = (preserve_pairs ? _mm256_subs_epi16(b0, b2) : _mm256_sub_epi16(b0, b2));
    const __m256i c3 = (preserve_pairs ? _mm256_subs_epi16(b1, b3) : _mm256_sub_epi16(b1, b3));
    const __m256i c4 = _mm256_add_epi16(b4, b6);
    const __m256i c5 = _mm256_add_epi16(b5, b7);
    const __m256i c6 = (preserve_pairs ? _mm256_subs_epi16(b4, b6) : _mm256_sub_epi16(b4, b6));
    const __m256i c7 = (preserve_pairs ? _mm256_subs_epi16(b5, b7) : _mm256_sub_epi16(b5, b7));
    a[0]             = c0;
    a[1]             = c1;
    a[2]             = c2;
    a[3]             = c3;
    a[4]             = c4;
    a[5]             = c5;
    a[6]             = c6;
    a[7]             = c7;
}

SIMD_INLINE void hadamard_v8_avx2(__m256i a[8], int preserve_pairs) {
    hadamard_v4_avx2(a, preserve_pairs);
    for (int i = 0; i < 4; ++i) {
        const __m256i x = a[i];
        const __m256i y = a[i + 4];
        a[i]            = _mm256_add_epi16(x, y);
        a[i + 4]        = (preserve_pairs ? _mm256_subs_epi16(x, y) : _mm256_sub_epi16(x, y));
    }
}

// Transpose two independent 8x8 word matrices without crossing 128-bit lanes.
SIMD_INLINE void hadamard_transpose8_avx2(__m256i a[8]) {
    const __m256i b0 = _mm256_unpacklo_epi16(a[0], a[1]);
    const __m256i b1 = _mm256_unpackhi_epi16(a[0], a[1]);
    const __m256i b2 = _mm256_unpacklo_epi16(a[2], a[3]);
    const __m256i b3 = _mm256_unpackhi_epi16(a[2], a[3]);
    const __m256i b4 = _mm256_unpacklo_epi16(a[4], a[5]);
    const __m256i b5 = _mm256_unpackhi_epi16(a[4], a[5]);
    const __m256i b6 = _mm256_unpacklo_epi16(a[6], a[7]);
    const __m256i b7 = _mm256_unpackhi_epi16(a[6], a[7]);
    const __m256i c0 = _mm256_unpacklo_epi32(b0, b2);
    const __m256i c1 = _mm256_unpackhi_epi32(b0, b2);
    const __m256i c2 = _mm256_unpacklo_epi32(b1, b3);
    const __m256i c3 = _mm256_unpackhi_epi32(b1, b3);
    const __m256i c4 = _mm256_unpacklo_epi32(b4, b6);
    const __m256i c5 = _mm256_unpackhi_epi32(b4, b6);
    const __m256i c6 = _mm256_unpacklo_epi32(b5, b7);
    const __m256i c7 = _mm256_unpackhi_epi32(b5, b7);
    a[0]             = _mm256_unpacklo_epi64(c0, c4);
    a[1]             = _mm256_unpackhi_epi64(c0, c4);
    a[2]             = _mm256_unpacklo_epi64(c1, c5);
    a[3]             = _mm256_unpackhi_epi64(c1, c5);
    a[4]             = _mm256_unpacklo_epi64(c2, c6);
    a[5]             = _mm256_unpackhi_epi64(c2, c6);
    a[6]             = _mm256_unpacklo_epi64(c3, c7);
    a[7]             = _mm256_unpackhi_epi64(c3, c7);
}

static INLINE __m256i hadamard_load4x4_avx2(const uint8_t* src, ptrdiff_t stride, int hbd) {
    if (hbd) {
        const __m128i top    = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)src),
                                               _mm_loadl_epi64((const __m128i*)(src + stride)));
        const __m128i bottom = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(src + 2 * stride)),
                                                  _mm_loadl_epi64((const __m128i*)(src + 3 * stride)));
        return _mm256_inserti128_si256(_mm256_castsi128_si256(top), bottom, 1);
    }
    uint32_t row[4];
    for (int i = 0; i < 4; ++i) {
        memcpy(&row[i], src + i * stride, sizeof(row[i]));
    }
    return _mm256_cvtepu8_epi16(_mm_setr_epi32((int)row[0], (int)row[1], (int)row[2], (int)row[3]));
}

static INLINE int hadamard_satd4_packed_avx2(__m256i a, int* dc) {
    // Preserve the arithmetic shift at the first butterfly of each pass.
    __m256i b = _mm256_shuffle_epi32(a, 0x4e);
    a         = _mm256_srai_epi16(_mm256_blend_epi16(_mm256_add_epi16(a, b), _mm256_sub_epi16(b, a), 0xf0), 1);
    b         = _mm256_permute2x128_si256(a, a, 1);
    a         = _mm256_blend_epi32(_mm256_add_epi16(a, b), _mm256_sub_epi16(b, a), 0xf0);
    b         = _mm256_shuffle_epi8(
        a,
        _mm256_setr_epi32(
            0x01000302, 0x05040706, 0x09080b0a, 0x0d0c0f0e, 0x01000302, 0x05040706, 0x09080b0a, 0x0d0c0f0e));
    a = _mm256_srai_epi16(_mm256_add_epi16(_mm256_sign_epi16(a, _mm256_set1_epi32((int)0xffff0001)), b), 1);
    if (dc != NULL) {
        const __m128i d = _mm_madd_epi16(_mm256_castsi256_si128(a), _mm_set1_epi32(1));
        *dc             = _mm_cvtsi128_si32(d) + _mm_extract_epi32(d, 1);
    }
    a = _mm256_abs_epi16(a);
    a = _mm256_max_epi16(a, _mm256_shuffle_epi32(a, 0xb1));
    // Penultimate magnitudes are <= 2*M. Reduce four rows in words:
    // 4*2*1023 = 8184, before the final adjacent-word sum widens to dwords.
    __m128i sum = _mm_add_epi16(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    sum         = _mm_add_epi16(sum, _mm_shuffle_epi32(sum, 0x4e));
    sum         = _mm_madd_epi16(sum, _mm_set1_epi16(1));
    return _mm_cvtsi128_si32(sum) * 2;
}

// Sum four dwords independently in each 128-bit lane, replicating each sum.
static INLINE __m256i hadamard_sum_lanes_avx2(__m256i a) {
    a = _mm256_add_epi32(a, _mm256_shuffle_epi32(a, 0xb1));
    return _mm256_add_epi32(a, _mm256_shuffle_epi32(a, 0x4e));
}

// Psy-only reduction: the original samples are unsigned in [0,M]. Every
// non-DC coefficient has balanced signs and magnitude <= 16*M after five
// butterflies. In each four-column half, any two of rows 1..3 have a joint
// absolute bound of 16*M, and all three have a bound of 24*M. Selecting the
// maxima from two halves therefore gives m[1]+m[2]+m[3] <= 32*M: 32736 at
// 10-bit. Row 0 can contain 32*M by itself. All four maxima still fit an
// unsigned word; only their pairwise reduction needs dwords.
SIMD_INLINE __m256i hadamard_psy8_sum_avx2(const __m256i a[8], int hbd) {
    __m256i m[4];
    for (int i = 0; i < 4; ++i) {
        m[i] = _mm256_max_epi16(_mm256_abs_epi16(a[i]), _mm256_abs_epi16(a[i + 4]));
    }
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i pair = _mm256_add_epi16(m[2], m[3]);
    if (hbd) {
        // m0 <= 32*M and m1+m2+m3 <= 32*M: at most 65472.
        // Bias each unsigned word into the signed PMADDWD range. The
        // eight-word reduction is corrected by 8*32768 at final rounding.
        const __m256i sum = _mm256_add_epi16(_mm256_add_epi16(m[0], m[1]), pair);
        return _mm256_madd_epi16(_mm256_xor_si256(sum, _mm256_set1_epi16(-32768)), ones);
    }
    // All four 8-bit maxima fit together: 4*32*255 = 32640.
    return _mm256_madd_epi16(_mm256_add_epi16(_mm256_add_epi16(m[0], m[1]), pair), ones);
}

// Shared with AVX-512VL callers: narrow blocks keep 128/256-bit data paths,
// while the target compiler can allocate the additional vector registers.
static INLINE int hadamard_small_satd_avx2(const uint8_t* src, ptrdiff_t ss, const uint8_t* pred, ptrdiff_t ps,
                                           int size, int hbd) {
    if (size == 4) {
        return hadamard_satd4_packed_avx2(
            _mm256_sub_epi16(hadamard_load4x4_avx2(src, ss, hbd), hadamard_load4x4_avx2(pred, ps, hbd)), NULL);
    }
    __m256i a[8];
    for (int i = 0; i < 8; ++i) {
        const __m128i s = hbd ? _mm_loadu_si128((const __m128i*)(src + i * ss))
                              : _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src + i * ss)));
        const __m128i p = hbd ? _mm_loadu_si128((const __m128i*)(pred + i * ps))
                              : _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(pred + i * ps)));
        a[i]            = _mm256_castsi128_si256(_mm_sub_epi16(s, p));
    }
    hadamard_v8_avx2(a, 1);
    hadamard_transpose8_avx2(a);
    hadamard_v4_avx2(a, 1);
    if (!hbd) {
        // Four penultimate magnitudes total at most 4*32*255 = 32640.
        // Reduce in words before widening once for the horizontal sum.
        __m128i sum16 = _mm_setzero_si128();
        for (int i = 0; i < 4; ++i) {
            sum16 = _mm_add_epi16(sum16,
                                  _mm_max_epi16(_mm_abs_epi16(_mm256_castsi256_si128(a[i])),
                                                _mm_abs_epi16(_mm256_castsi256_si128(a[i + 4]))));
        }
        __m128i sum32 = _mm_madd_epi16(sum16, _mm_set1_epi16(1));
        sum32         = _mm_add_epi32(sum32, _mm_shuffle_epi32(sum32, 0x4e));
        sum32         = _mm_add_epi32(sum32, _mm_shuffle_epi32(sum32, 0xb1));
        return _mm_cvtsi128_si32(sum32) * 2;
    }
    __m128i sum = _mm_setzero_si128();
    for (int i = 0; i < 4; ++i) {
        const __m128i m = _mm_max_epi16(_mm_abs_epi16(_mm256_castsi256_si128(a[i])),
                                        _mm_abs_epi16(_mm256_castsi256_si128(a[i + 4])));
        sum             = _mm_add_epi32(sum, _mm_madd_epi16(m, _mm_set1_epi16(1)));
    }
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4e));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xb1));
    return _mm_cvtsi128_si32(sum) * 2;
}

// Two horizontally adjacent 8x8 transforms. Keep five stages in signed words
// even for 10-bit residuals: 32*1023 = 32736. Only the last stage needs dwords.
// Merge left/right quadrants here, preserving the 16x16 arithmetic >> 1.
SIMD_INLINE void hadamard_band_avx2(const uint8_t* src, ptrdiff_t ss, const uint8_t* pred, ptrdiff_t ps, __m256i* coeff,
                                    int hbd) {
    __m256i a[8];
    for (int i = 0; i < 8; ++i) {
        const __m256i s = hbd ? _mm256_loadu_si256((const __m256i*)(src + i * ss))
                              : _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(src + i * ss)));
        const __m256i p = hbd ? _mm256_loadu_si256((const __m256i*)(pred + i * ps))
                              : _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(pred + i * ps)));
        a[i]            = _mm256_sub_epi16(s, p);
    }
    hadamard_v8_avx2(a, 1);
    hadamard_transpose8_avx2(a);
    if (!hbd) {
        hadamard_v8_avx2(a, 1);
        for (int i = 0; i < 8; ++i) {
            const __m256i swap = _mm256_permute2x128_si256(a[i], a[i], 1);
            coeff[i]           = _mm256_srai_epi16(
                _mm256_blend_epi32(_mm256_add_epi16(a[i], swap), _mm256_sub_epi16(swap, a[i]), 0xf0), 1);
        }
    } else {
        hadamard_v4_avx2(a, 1);
        const __m256i plus  = _mm256_set1_epi16(1);
        const __m256i minus = _mm256_set1_epi32((int)0xffff0001);
        for (int i = 0; i < 4; ++i) {
            const __m256i lo   = _mm256_unpacklo_epi16(a[i], a[i + 4]);
            const __m256i hi   = _mm256_unpackhi_epi16(a[i], a[i + 4]);
            const __m256i b[4] = {_mm256_madd_epi16(lo, plus),
                                  _mm256_madd_epi16(hi, plus),
                                  _mm256_madd_epi16(lo, minus),
                                  _mm256_madd_epi16(hi, minus)};
            for (int j = 0; j < 4; ++j) {
                const __m256i swap = _mm256_permute2x128_si256(b[j], b[j], 1);
                coeff[4 * i + j]   = _mm256_srai_epi32(
                    _mm256_blend_epi32(_mm256_add_epi32(b[j], swap), _mm256_sub_epi32(swap, b[j]), 0xf0), 1);
            }
        }
    }
}

static INLINE int hadamard_reduce_avx2(__m256i sum) {
    __m128i total = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    total         = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0x4e));
    total         = _mm_add_epi32(total, _mm_shuffle_epi32(total, 0xb1));
    return _mm_cvtsi128_si32(total);
}

SIMD_INLINE int hadamard_16x16_avx2(const uint8_t* src, ptrdiff_t ss, const uint8_t* pred, ptrdiff_t ps, __m256i* top,
                                    __m256i* bottom, int hbd) {
    hadamard_band_avx2(src, ss, pred, ps, top, hbd);
    hadamard_band_avx2(src + 8 * ss, ss, pred + 8 * ps, ps, bottom, hbd);
    const int count = hbd ? 16 : 8;
    __m256i   sum   = _mm256_setzero_si256();
    for (int i = 0; i < count; ++i) {
        if (hbd) {
            sum = _mm256_add_epi32(sum, _mm256_max_epi32(_mm256_abs_epi32(top[i]), _mm256_abs_epi32(bottom[i])));
        } else {
            const __m256i m = _mm256_max_epi16(_mm256_abs_epi16(top[i]), _mm256_abs_epi16(bottom[i]));
            sum             = _mm256_add_epi32(sum, _mm256_madd_epi16(m, _mm256_set1_epi16(1)));
        }
    }
    // |a+b| + |a-b| = 2*max(|a|,|b|): omit the last butterfly and coefficient stores.
    return hadamard_reduce_avx2(sum) * 2;
}

SIMD_INLINE __m128i psy_energy4_rows_avx2(const __m128i row[4]) {
    __m256i a    = _mm256_inserti128_si256(_mm256_castsi128_si256(row[0]), row[1], 1);
    __m256i b    = _mm256_inserti128_si256(_mm256_castsi128_si256(row[2]), row[3], 1);
    __m256i swap = _mm256_permute2x128_si256(a, a, 1);
    a            = _mm256_srai_epi16(_mm256_blend_epi32(_mm256_add_epi16(a, swap), _mm256_sub_epi16(swap, a), 0xf0), 1);
    swap         = _mm256_permute2x128_si256(b, b, 1);
    b            = _mm256_srai_epi16(_mm256_blend_epi32(_mm256_add_epi16(b, swap), _mm256_sub_epi16(swap, b), 0xf0), 1);
    const __m256i x       = a;
    a                     = _mm256_add_epi16(x, b);
    b                     = _mm256_sub_epi16(x, b);
    const __m256i shuffle = _mm256_setr_epi32(
        0x01000302, 0x05040706, 0x09080b0a, 0x0d0c0f0e, 0x01000302, 0x05040706, 0x09080b0a, 0x0d0c0f0e);
    const __m256i signs = _mm256_set1_epi32((int)0xffff0001);
    a          = _mm256_srai_epi16(_mm256_add_epi16(_mm256_sign_epi16(a, signs), _mm256_shuffle_epi8(a, shuffle)), 1);
    b          = _mm256_srai_epi16(_mm256_add_epi16(_mm256_sign_epi16(b, signs), _mm256_shuffle_epi8(b, shuffle)), 1);
    __m128i dc = _mm_madd_epi16(_mm256_castsi256_si128(a), _mm_set1_epi32(1));
    dc         = _mm_add_epi32(dc, _mm_shuffle_epi32(dc, 0xb1));
    a          = _mm256_abs_epi16(a);
    b          = _mm256_abs_epi16(b);
    a          = _mm256_max_epi16(a, _mm256_shuffle_epi32(a, 0xb1));
    b          = _mm256_max_epi16(b, _mm256_shuffle_epi32(b, 0xb1));
    a          = _mm256_add_epi16(a, b);
    // Four penultimate magnitudes total at most 8*1023 = 8184.
    __m128i sum = _mm_add_epi16(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    sum         = _mm_madd_epi16(sum, _mm_set1_epi16(1));
    return _mm_sub_epi32(_mm_slli_epi32(sum, 2), dc);
}

SIMD_INLINE uint32_t psy_gap4_avx2(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, int hbd) {
    __m128i row[4];
    for (int i = 0; i < 4; ++i) {
        if (hbd) {
            row[i] = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(s + i * ss)),
                                        _mm_loadl_epi64((const __m128i*)(r + i * rs)));
        } else {
            uint32_t x, y;
            memcpy(&x, s + i * ss, 4);
            memcpy(&y, r + i * rs, 4);
            row[i] = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(_mm_cvtsi32_si128((int)x), _mm_cvtsi32_si128((int)y)));
        }
    }
    const __m128i energy = psy_energy4_rows_avx2(row);
    return (uint32_t)_mm_cvtsi128_si32(_mm_abs_epi32(_mm_sub_epi32(energy, _mm_shuffle_epi32(energy, 0x4e))));
}

SIMD_INLINE __m256i psy_energy8_avx2(__m256i a[8], int hbd) {
    hadamard_v8_avx2(a, 1);
    hadamard_transpose8_avx2(a);
    hadamard_v4_avx2(a, 1);
    // Only the first word in each lane is the DC coefficient. Its full
    // unsigned range is 64*1023 = 65472, so no widening is needed yet.
    __m256i dc        = _mm256_add_epi16(a[0], a[4]);
    dc                = _mm256_srli_epi16(_mm256_add_epi16(dc, _mm256_set1_epi16(2)), 2);
    dc                = _mm256_shuffle_epi32(_mm256_and_si256(dc, _mm256_set1_epi32(65535)), 0);
    const __m256i sum = hadamard_sum_lanes_avx2(hadamard_psy8_sum_avx2(a, hbd));
    return _mm256_sub_epi32(_mm256_srli_epi32(_mm256_add_epi32(sum, _mm256_set1_epi32(hbd ? (8 * 32768 + 1) : 1)), 1),
                            dc);
}

// Pairing source and reconstruction in lanes favors the earlier DC sum and
// separate signed-word reductions. Keep this short path independent of the
// unsigned reduction used when batching neighboring tiles.
SIMD_INLINE __m256i psy_energy8_narrow_avx2(__m256i a[8]) {
    hadamard_v8_avx2(a, 0);
    const __m256i dc = hadamard_sum_lanes_avx2(_mm256_madd_epi16(a[0], _mm256_set1_epi16(1)));
    hadamard_transpose8_avx2(a);
    hadamard_v4_avx2(a, 0);
    __m256i m[4];
    for (int i = 0; i < 4; ++i) {
        m[i] = _mm256_max_epi16(_mm256_abs_epi16(a[i]), _mm256_abs_epi16(a[i + 4]));
    }
    __m256i sum = _mm256_add_epi32(
        _mm256_madd_epi16(m[0], _mm256_set1_epi16(1)),
        _mm256_madd_epi16(_mm256_add_epi16(m[1], _mm256_add_epi16(m[2], m[3])), _mm256_set1_epi16(1)));
    sum = hadamard_sum_lanes_avx2(sum);
    return _mm256_sub_epi32(_mm256_srli_epi32(_mm256_add_epi32(sum, _mm256_set1_epi32(1)), 1),
                            _mm256_srli_epi32(_mm256_add_epi32(dc, _mm256_set1_epi32(2)), 2));
}

// For a single tile, source and reconstruction occupy the two 128-bit lanes.
// This also handles the final padded edge tile without a wider row load.
SIMD_INLINE uint32_t psy_gap8_avx2(const uint8_t* src, ptrdiff_t ss, const uint8_t* rec, ptrdiff_t rs, int hbd) {
    __m256i a[8];
    for (int i = 0; i < 8; ++i) {
        if (hbd) {
            a[i] = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(src + i * ss))),
                                           _mm_loadu_si128((const __m128i*)(rec + i * rs)),
                                           1);
        } else {
            a[i] = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(src + i * ss)),
                                                           _mm_loadl_epi64((const __m128i*)(rec + i * rs))));
        }
    }
    const __m256i e = hbd ? psy_energy8_narrow_avx2(a) : psy_energy8_avx2(a, hbd);
    return (uint32_t)_mm_cvtsi128_si32(
        _mm_abs_epi32(_mm_sub_epi32(_mm256_castsi256_si128(e), _mm256_extracti128_si256(e, 1))));
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
// This 8x32 cache path takes strides in 16-bit samples, unlike the byte
// pointer helpers above. Complete each source tile before loading recon.
SIMD_INLINE uint64_t psy_gap8x32_strided_avx2(const uint16_t* s, uint32_t ss, const uint16_t* r, uint32_t rs) {
    uint64_t gap = 0;
    for (int y = 0; y < 32; y += 8) {
        __m256i a[8];
        for (int i = 0; i < 8; ++i) {
            a[i] = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(s + (y + i) * ss)));
        }
        const int source = _mm_cvtsi128_si32(_mm256_castsi256_si128(psy_energy8_narrow_avx2(a)));
        for (int i = 0; i < 8; ++i) {
            a[i] = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(r + (y + i) * rs)));
        }
        const int recon = _mm_cvtsi128_si32(_mm256_castsi256_si128(psy_energy8_narrow_avx2(a)));
        gap += (uint32_t)abs(source - recon);
    }
    return gap << 2;
}
#endif

// Share cropped-region traversal while retaining each ISA's coded-block
// workers. Batch only complete padded tiles: for example, width 12 at tile
// size 8 contains two tiles and can use the 16x8 worker. Each callback gets
// a coded shape, so it cannot recurse into this fallback.
SIMD_INLINE uint64_t psy_distortion_tiles(
    const uint8_t* input, ptrdiff_t ss, const uint8_t* recon, ptrdiff_t rs, uint32_t width, uint32_t height, int hbd,
    uint64_t (*lbd)(const uint8_t*, uint32_t, const uint8_t*, uint32_t, uint32_t, uint32_t),
    uint64_t (*highbd)(const uint16_t*, uint32_t, const uint16_t*, uint32_t, uint32_t, uint32_t)) {
    (void)hbd;
    (void)highbd;
    const uint32_t tile = width >= 8 && height >= 8 ? 8 : 4;
    uint64_t       sum  = 0;
    for (uint32_t y = 0; y < height; y += tile) {
        for (uint32_t x = 0; x < width;) {
            const uint32_t remaining = width - x;
            const uint32_t batch     = tile == 8 ? (remaining > 24      ? 32
                                                        : remaining > 8 ? 16
                                                                        : 8)
                                                 : (remaining > 12      ? 16
                                                        : remaining > 4 ? 8
                                                                        : 4);
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
            if (hbd) {
                sum += highbd((const uint16_t*)(input + y * ss) + x,
                              (uint32_t)(ss / 2),
                              (const uint16_t*)(recon + y * rs) + x,
                              (uint32_t)(rs / 2),
                              batch,
                              tile);
            } else
#endif
            {
                sum += lbd(input + y * ss + x, (uint32_t)ss, recon + y * rs + x, (uint32_t)rs, batch, tile);
            }
            x += batch;
        }
    }
    return sum;
}

#endif // SVT_AV1_HADAMARD_PATH_AVX2_H_
