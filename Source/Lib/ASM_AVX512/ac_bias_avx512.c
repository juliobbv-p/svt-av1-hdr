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

#include <stdlib.h>

#include "common_dsp_rtcd.h"
#include "hadamard_path_avx512.h"
#include "hadamard_path_avx2.h"

// Biased wide-tile sums are corrected before rounding; dc is already rounded.
SIMD_INLINE __m512i psy_energy8_from_sum_avx512(__m512i sum, __m512i dc, int hbd) {
    return _mm512_sub_epi32(_mm512_srli_epi32(_mm512_add_epi32(sum, _mm512_set1_epi32(hbd ? (8 * 32768 + 1) : 1)), 1),
                            dc);
}

// Four 4x4 tiles retain their columns throughout both passes.
SIMD_INLINE __m256i psy_energy4_rows_avx512(const __m256i rows[4]) {
    __m512i a, b;
    a               = _mm512_inserti64x4(_mm512_castsi256_si512(rows[0]), rows[1], 1);
    b               = _mm512_inserti64x4(_mm512_castsi256_si512(rows[2]), rows[3], 1);
    __m512i swap    = _mm512_shuffle_i32x4(a, a, 0x4e);
    a               = _mm512_srai_epi16(_mm512_mask_sub_epi16(_mm512_add_epi16(a, swap), 0xffff0000, swap, a), 1);
    swap            = _mm512_shuffle_i32x4(b, b, 0x4e);
    b               = _mm512_srai_epi16(_mm512_mask_sub_epi16(_mm512_add_epi16(b, swap), 0xffff0000, swap, b), 1);
    const __m512i x = a;
    a               = _mm512_srai_epi16(hadamard_h1_avx512(_mm512_add_epi16(x, b)), 1);
    b               = _mm512_srai_epi16(hadamard_h1_avx512(_mm512_sub_epi16(x, b)), 1);
    __m256i dc      = _mm256_madd_epi16(_mm512_castsi512_si256(a), _mm256_set1_epi32(1));
    dc              = _mm256_add_epi32(dc, _mm256_shuffle_epi32(dc, 0xb1));
    a               = _mm512_abs_epi16(a);
    b               = _mm512_abs_epi16(b);
    a               = _mm512_max_epi16(a, _mm512_shuffle_epi32(a, 0xb1));
    b               = _mm512_max_epi16(b, _mm512_shuffle_epi32(b, 0xb1));
    a               = _mm512_add_epi16(a, b);
    __m256i sum     = _mm256_add_epi16(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1));
    sum             = _mm256_madd_epi16(sum, _mm256_set1_epi16(1));
    return _mm256_sub_epi32(_mm256_slli_epi32(sum, 2), dc);
}

SIMD_INLINE __m256i psy_energy4_strip_avx512(const uint8_t* src, ptrdiff_t stride, int hbd) {
    __m256i rows[4];
    for (int i = 0; i < 4; ++i) {
        rows[i] = hbd ? _mm256_loadu_si256((const __m256i*)(src + i * stride))
                      : _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(src + i * stride)));
    }
    return psy_energy4_rows_avx512(rows);
}

SIMD_INLINE uint32_t psy_gap4_two_avx512(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, int vertical,
                                         int hbd) {
    __m256i rows[4];
    for (int i = 0; i < 4; ++i) {
        __m128i x, y;
        if (!vertical) {
            if (hbd) {
                x = _mm_loadu_si128((const __m128i*)(s + i * ss));
                y = _mm_loadu_si128((const __m128i*)(r + i * rs));
            } else {
                x = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(s + i * ss)));
                y = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(r + i * rs)));
            }
        } else if (hbd) {
            x = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(s + i * ss)),
                                   _mm_loadl_epi64((const __m128i*)(s + (i + 4) * ss)));
            y = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(r + i * rs)),
                                   _mm_loadl_epi64((const __m128i*)(r + (i + 4) * rs)));
        } else {
            uint32_t a, b, c, d;
            memcpy(&a, s + i * ss, 4);
            memcpy(&b, s + (i + 4) * ss, 4);
            memcpy(&c, r + i * rs, 4);
            memcpy(&d, r + (i + 4) * rs, 4);
            x = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(_mm_cvtsi32_si128((int)a), _mm_cvtsi32_si128((int)b)));
            y = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(_mm_cvtsi32_si128((int)c), _mm_cvtsi32_si128((int)d)));
        }
        rows[i] = _mm256_inserti128_si256(_mm256_castsi128_si256(x), y, 1);
    }
    const __m256i e   = psy_energy4_rows_avx512(rows);
    const __m128i gap = _mm_abs_epi32(_mm_sub_epi32(_mm256_castsi256_si128(e), _mm256_extracti128_si256(e, 1)));
    return (uint32_t)_mm_cvtsi128_si32(_mm_add_epi32(gap, _mm_shuffle_epi32(gap, 0x4e)));
}

// Two source tiles and their reconstructions fill all four 128-bit lanes.
// Narrow rectangles then need only one transform pass per pair of tiles.
SIMD_INLINE __m512i psy_gap8_pair_avx512(const uint8_t* src, ptrdiff_t ss, const uint8_t* rec, ptrdiff_t rs,
                                         int horizontal, int hbd) {
    __m512i a[8], dc;
    for (int i = 0; i < 8; ++i) {
        const uint8_t* s = src + i * ss;
        const uint8_t* r = rec + i * rs;
        if (horizontal) {
            const __m256i x = hbd ? _mm256_loadu_si256((const __m256i*)s)
                                  : _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)s));
            const __m256i y = hbd ? _mm256_loadu_si256((const __m256i*)r)
                                  : _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)r));
            a[i]            = _mm512_inserti64x4(_mm512_castsi256_si512(x), y, 1);
        } else if (hbd) {
            a[i] = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i*)s));
            a[i] = _mm512_inserti32x4(a[i], _mm_loadu_si128((const __m128i*)(s + 8 * ss)), 1);
            a[i] = _mm512_inserti32x4(a[i], _mm_loadu_si128((const __m128i*)r), 2);
            a[i] = _mm512_inserti32x4(a[i], _mm_loadu_si128((const __m128i*)(r + 8 * rs)), 3);
        } else {
            const __m128i x = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)s),
                                                 _mm_loadl_epi64((const __m128i*)(s + 8 * ss)));
            const __m128i y = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)r),
                                                 _mm_loadl_epi64((const __m128i*)(r + 8 * rs)));
            a[i]            = _mm512_cvtepu8_epi16(_mm256_inserti128_si256(_mm256_castsi128_si256(x), y, 1));
        }
    }
    hadamard_v8_avx512(a);
    hadamard_transpose8_avx512(a);
    hadamard_v4_avx512(a);
    dc                = _mm512_add_epi16(a[0], a[4]);
    dc                = _mm512_srli_epi16(_mm512_add_epi16(dc, _mm512_set1_epi16(2)), 2);
    dc                = _mm512_shuffle_epi32(_mm512_and_si512(dc, _mm512_set1_epi32(65535)), 0);
    const __m512i sum = hadamard_sum_lanes_avx512(hadamard_psy8_sum_avx512(a, hbd));
    const __m512i e   = psy_energy8_from_sum_avx512(sum, dc, hbd);
    return _mm512_maskz_mov_epi32(0x0011, _mm512_abs_epi32(_mm512_sub_epi32(e, _mm512_shuffle_i32x4(e, e, 0x4e))));
}

// Keep source and reconstruction butterflies independent until each tile
// has been normalized. Consume a 32x8, 16x16 or 8x32 group without compacting
// its energies across lanes; four tile gaps can accumulate directly.
SIMD_INLINE __m512i psy_gap8_four_avx512(const uint8_t* src, ptrdiff_t ss, const uint8_t* rec, ptrdiff_t rs, int width,
                                         int hbd) {
    __m512i a[8], b[8];
    for (int i = 0; i < 8; ++i) {
        if (width == 32) {
            a[i] = hbd ? _mm512_loadu_si512(src + i * ss)
                       : _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(src + i * ss)));
            b[i] = hbd ? _mm512_loadu_si512(rec + i * rs)
                       : _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(rec + i * rs)));
        } else if (width == 16) {
            if (hbd) {
                a[i] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(src + i * ss))),
                                          _mm256_loadu_si256((const __m256i*)(src + (i + 8) * ss)),
                                          1);
                b[i] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(rec + i * rs))),
                                          _mm256_loadu_si256((const __m256i*)(rec + (i + 8) * rs)),
                                          1);
            } else {
                a[i] = _mm512_cvtepu8_epi16(
                    _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(src + i * ss))),
                                            _mm_loadu_si128((const __m128i*)(src + (i + 8) * ss)),
                                            1));
                b[i] = _mm512_cvtepu8_epi16(
                    _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(rec + i * rs))),
                                            _mm_loadu_si128((const __m128i*)(rec + (i + 8) * rs)),
                                            1));
            }
        } else {
            a[i] = hadamard_load8x4_avx512(src + i * ss, 8 * ss, hbd);
            b[i] = hadamard_load8x4_avx512(rec + i * rs, 8 * rs, hbd);
        }
    }
    hadamard_v8_avx512(a);
    hadamard_v8_avx512(b);
    hadamard_transpose8_avx512(a);
    hadamard_transpose8_avx512(b);
    hadamard_v4_avx512(a);
    hadamard_v4_avx512(b);
    // Within each 128-bit lane, reduce source and reconstruction together:
    // [s0+s2, s1+s3, r0+r2, r1+r3] becomes [S,S,R,R]. Each tile
    // keeps its own rounded DC and absolute energy gap.
    const __m512i dc0    = _mm512_add_epi16(a[0], a[4]);
    const __m512i dc1    = _mm512_add_epi16(b[0], b[4]);
    const __m512i sum0   = hadamard_psy8_sum_avx512(a, hbd);
    const __m512i sum1   = hadamard_psy8_sum_avx512(b, hbd);
    __m512i       sum    = _mm512_add_epi32(_mm512_unpacklo_epi64(sum0, sum1), _mm512_unpackhi_epi64(sum0, sum1));
    sum                  = _mm512_add_epi32(sum, _mm512_shuffle_epi32(sum, 0xb1));
    __m512i dc           = _mm512_unpacklo_epi64(dc0, dc1);
    dc                   = _mm512_shuffle_epi32(_mm512_and_si512(dc, _mm512_set1_epi32(65535)), 0xa0);
    dc                   = _mm512_srli_epi16(_mm512_add_epi16(dc, _mm512_set1_epi16(2)), 2);
    const __m512i energy = psy_energy8_from_sum_avx512(sum, dc, hbd);
    return _mm512_abs_epi32(_mm512_sub_epi32(energy, _mm512_shuffle_epi32(energy, 0x4e)));
}

// Only the first dword of each 128-bit lane contributes. Fold the four
// lanes without reducing their unused or duplicated dwords as well.
SIMD_INLINE uint32_t psy_sum_gaps_avx512(__m512i gap) {
    const __m256i half = _mm256_add_epi32(_mm512_castsi512_si256(gap), _mm512_extracti64x4_epi64(gap, 1));
    return (uint32_t)_mm_cvtsi128_si32(_mm_add_epi32(_mm256_castsi256_si128(half), _mm256_extracti128_si256(half, 1)));
}

// Coded block dimensions. Cropped frame-edge regions are handled separately.
static INLINE int psy_block_shape_avx512(uint32_t width, uint32_t height) {
    return width >= 4 && height >= 4 && width <= 128 && height <= 128 && !(width & (width - 1)) &&
        !(height & (height - 1)) && width <= 4 * height && height <= 4 * width;
}

// Coded widths of 32..128. The scalar dispatcher handles narrower shapes.
// Consume four contiguous 8x8 tiles per band.
// Even the loose bound SATD8 <= 4096*M gives a total gap <= 256*1024*1023,
// so dword accumulators suffice. Widen only for the public return/scaling.
SIMD_INLINE uint64_t psy_distortion_bands_avx512(const uint8_t* input, ptrdiff_t input_stride, const uint8_t* recon,
                                                 ptrdiff_t recon_stride, uint32_t width, uint32_t height, int hbd) {
    __m512i gap = _mm512_setzero_si512();
    for (uint32_t y = 0; y < height; y += 8) {
        for (uint32_t x = 0; x < width; x += 32) {
            gap = _mm512_add_epi32(gap,
                                   psy_gap8_four_avx512(input + y * input_stride + (x << hbd),
                                                        input_stride,
                                                        recon + y * recon_stride + (x << hbd),
                                                        recon_stride,
                                                        32,
                                                        hbd));
        }
    }
    return (uint64_t)psy_sum_gaps_avx512(gap) << (2 * hbd);
}

// Coded widths of 8 or 16 use vertical pairs or four-tile groups.
SIMD_INLINE uint64_t psy_distortion_narrow_avx512(const uint8_t* input, ptrdiff_t input_stride, const uint8_t* recon,
                                                  ptrdiff_t recon_stride, uint32_t width, uint32_t height, int hbd) {
    // A 4-KiB byte pitch maps rows onto the same L1 sets. Use two
    // eight-row passes for 16x16 instead of keeping sixteen rows live.
    if (hbd && width == 16 && height == 16 && (input_stride & 4095) == 0) {
        __m512i gap = _mm512_setzero_si512();
        for (uint32_t y = 0; y < height; y += 8) {
            gap = _mm512_add_epi32(
                gap,
                psy_gap8_pair_avx512(
                    input + y * input_stride, input_stride, recon + y * recon_stride, recon_stride, 1, hbd));
        }
        return (uint64_t)psy_sum_gaps_avx512(gap) << (2 * hbd);
    }
    if (width == 16) {
        __m512i gap = _mm512_setzero_si512();
        if (height == 8) {
            gap = psy_gap8_pair_avx512(input, input_stride, recon, recon_stride, 1, hbd);
        } else {
            for (uint32_t y = 0; y < height; y += 16) {
                gap = _mm512_add_epi32(
                    gap,
                    psy_gap8_four_avx512(
                        input + y * input_stride, input_stride, recon + y * recon_stride, recon_stride, 16, hbd));
            }
        }
        return (uint64_t)psy_sum_gaps_avx512(gap) << (2 * hbd);
    }
    if (height == 32) {
        return (uint64_t)psy_sum_gaps_avx512(psy_gap8_four_avx512(input, input_stride, recon, recon_stride, 8, hbd))
            << (2 * hbd);
    }
    // The dispatcher admits only 8x16 here; 8x8 has its own narrow worker.
    return (uint64_t)psy_sum_gaps_avx512(psy_gap8_pair_avx512(input, input_stride, recon, recon_stride, 0, hbd))
        << (2 * hbd);
}

SIMD_INLINE uint64_t psy_strip4_avx512(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, uint32_t width,
                                       uint32_t height, int hbd) {
    uint32_t sum = 0;
    if (width == 4) {
        for (uint32_t y = 0; y < height; y += 8) {
            sum += psy_gap4_two_avx512(s + y * ss, ss, r + y * rs, rs, 1, hbd);
        }
    } else if (width == 8) {
        sum = psy_gap4_two_avx512(s, ss, r, rs, 0, hbd);
    } else {
        const __m256i a = psy_energy4_strip_avx512(s, ss, hbd), b = psy_energy4_strip_avx512(r, rs, hbd);
        const __m256i gap = _mm256_abs_epi32(_mm256_sub_epi32(a, b));
        __m128i       t   = _mm_add_epi32(_mm256_castsi256_si128(gap), _mm256_extracti128_si256(gap, 1));
        sum               = (uint32_t)_mm_cvtsi128_si32(_mm_add_epi32(t, _mm_shuffle_epi32(t, 0x4e)));
    }
    return (uint64_t)sum << (2 * hbd);
}

static NOINLINE uint64_t psy_distortion_wide_avx512(const uint8_t* input, uint32_t input_stride, const uint8_t* recon,
                                                    uint32_t recon_stride, uint32_t width, uint32_t height) {
    // Share the edge traversal and reuse the existing AVX-512 tile workers.
    if (!psy_block_shape_avx512(width, height)) {
        return psy_distortion_tiles(
            input, input_stride, recon, recon_stride, width, height, 0, svt_psy_distortion_avx512, NULL);
    }
    return psy_distortion_bands_avx512(input, input_stride, recon, recon_stride, width, height, 0);
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
static NOINLINE uint64_t psy_distortion_hbd_wide_avx512(const uint16_t* input, uint32_t input_stride,
                                                        const uint16_t* recon, uint32_t recon_stride, uint32_t width,
                                                        uint32_t height) {
    const uint8_t*  s  = (const uint8_t*)input;
    const uint8_t*  r  = (const uint8_t*)recon;
    const ptrdiff_t ss = (ptrdiff_t)input_stride * sizeof(*input);
    const ptrdiff_t rs = (ptrdiff_t)recon_stride * sizeof(*recon);
    // Share the edge traversal and reuse the existing AVX-512 tile workers.
    if (!psy_block_shape_avx512(width, height)) {
        return psy_distortion_tiles(s, ss, r, rs, width, height, 1, NULL, svt_psy_distortion_hbd_avx512);
    }
    return psy_distortion_bands_avx512(s, ss, r, rs, width, height, 1);
}

static NOINLINE uint64_t psy_small4_hbd_avx512(const uint16_t* s, uint32_t ss, const uint16_t* r, uint32_t rs) {
    return (uint64_t)psy_gap4_avx2((const uint8_t*)s, (ptrdiff_t)ss << 1, (const uint8_t*)r, (ptrdiff_t)rs << 1, 1)
        << 2;
}

static NOINLINE uint64_t psy_small8_hbd_avx512(const uint16_t* s, uint32_t ss, const uint16_t* r, uint32_t rs) {
    return (uint64_t)psy_gap8_avx2((const uint8_t*)s, (ptrdiff_t)ss << 1, (const uint8_t*)r, (ptrdiff_t)rs << 1, 1)
        << 2;
}

static NOINLINE uint64_t psy_strip4_hbd_avx512(const uint16_t* input, uint32_t input_stride, const uint16_t* recon,
                                               uint32_t recon_stride, uint32_t width, uint32_t height) {
    return psy_strip4_avx512((const uint8_t*)input,
                             (ptrdiff_t)input_stride << 1,
                             (const uint8_t*)recon,
                             (ptrdiff_t)recon_stride << 1,
                             width,
                             height,
                             1);
}

static NOINLINE uint64_t psy_8x16_hbd_avx512(const uint16_t* s, uint32_t ss, const uint16_t* r, uint32_t rs) {
    return psy_distortion_narrow_avx512(
        (const uint8_t*)s, (ptrdiff_t)ss << 1, (const uint8_t*)r, (ptrdiff_t)rs << 1, 8, 16, 1);
}

static NOINLINE uint64_t psy_8x32_hbd_avx512(const uint16_t* s, uint32_t ss, const uint16_t* r, uint32_t rs) {
    return psy_distortion_narrow_avx512(
        (const uint8_t*)s, (ptrdiff_t)ss << 1, (const uint8_t*)r, (ptrdiff_t)rs << 1, 8, 32, 1);
}

// Keep scalar dispatch separate from the vector workers, so small blocks
// do not inherit the wide worker's register saves and address setup.
static NOINLINE uint64_t psy_width16_hbd_avx512(const uint16_t* s, uint32_t ss, const uint16_t* r, uint32_t rs,
                                                uint32_t width, uint32_t height) {
    (void)width;
    return psy_distortion_narrow_avx512(
        (const uint8_t*)s, (ptrdiff_t)ss << 1, (const uint8_t*)r, (ptrdiff_t)rs << 1, 16, height, 1);
}

static NOINLINE uint64_t psy_32x8_hbd_avx512(const uint16_t* s, uint32_t ss, const uint16_t* r, uint32_t rs) {
    return (uint64_t)psy_sum_gaps_avx512(psy_gap8_four_avx512(
               (const uint8_t*)s, (ptrdiff_t)ss << 1, (const uint8_t*)r, (ptrdiff_t)rs << 1, 32, 1))
        << 2;
}

uint64_t svt_psy_distortion_hbd_avx512(const uint16_t* input, uint32_t input_stride, const uint16_t* recon,
                                       uint32_t recon_stride, uint32_t width, uint32_t height) {
    if (width == 4 && height == 4) {
        return psy_small4_hbd_avx512(input, input_stride, recon, recon_stride);
    }
    if (width == 8 && height == 8) {
        return psy_small8_hbd_avx512(input, input_stride, recon, recon_stride);
    }
    if ((width == 4 && (height == 8 || height == 16)) || (height == 4 && (width == 8 || width == 16))) {
        return psy_strip4_hbd_avx512(input, input_stride, recon, recon_stride, width, height);
    }
    // Reuse AVX2's eight-row schedule for the 4-KiB cache-conflict case.
    if (width == 8 && height == 32 && input_stride >= 2048 && !(input_stride & 2047) && ((uintptr_t)input & 63) <= 48) {
        return svt_psy_distortion_hbd_avx2(input, input_stride, recon, recon_stride, width, height);
    }
    if (width == 8 && height == 16) {
        return psy_8x16_hbd_avx512(input, input_stride, recon, recon_stride);
    }
    if (width == 8 && height == 32) {
        return psy_8x32_hbd_avx512(input, input_stride, recon, recon_stride);
    }
    if (width == 16 && height >= 8 && height <= 64 && !(height & (height - 1))) {
        return psy_width16_hbd_avx512(input, input_stride, recon, recon_stride, width, height);
    }
    if (width == 32 && height == 8) {
        return psy_32x8_hbd_avx512(input, input_stride, recon, recon_stride);
    }
    return psy_distortion_hbd_wide_avx512(input, input_stride, recon, recon_stride, width, height);
}

#endif

static NOINLINE uint64_t psy_small4_avx512(const uint8_t* s, uint32_t ss, const uint8_t* r, uint32_t rs) {
    return (uint64_t)psy_gap4_avx2(s, ss, r, rs, 0);
}

static NOINLINE uint64_t psy_small8_avx512(const uint8_t* s, uint32_t ss, const uint8_t* r, uint32_t rs) {
    return (uint64_t)psy_gap8_avx2(s, ss, r, rs, 0);
}

static NOINLINE uint64_t psy_strip4_lbd_avx512(const uint8_t* input, uint32_t input_stride, const uint8_t* recon,
                                               uint32_t recon_stride, uint32_t width, uint32_t height) {
    return psy_strip4_avx512((const uint8_t*)input,
                             (ptrdiff_t)input_stride,
                             (const uint8_t*)recon,
                             (ptrdiff_t)recon_stride,
                             width,
                             height,
                             0);
}

static NOINLINE uint64_t psy_8x16_avx512(const uint8_t* s, uint32_t ss, const uint8_t* r, uint32_t rs) {
    return psy_distortion_narrow_avx512(s, ss, r, rs, 8, 16, 0);
}

static NOINLINE uint64_t psy_8x32_avx512(const uint8_t* s, uint32_t ss, const uint8_t* r, uint32_t rs) {
    return psy_distortion_narrow_avx512(s, ss, r, rs, 8, 32, 0);
}

static NOINLINE uint64_t psy_width16_avx512(const uint8_t* s, uint32_t ss, const uint8_t* r, uint32_t rs,
                                            uint32_t width, uint32_t height) {
    (void)width;
    return psy_distortion_narrow_avx512(
        (const uint8_t*)s, (ptrdiff_t)ss, (const uint8_t*)r, (ptrdiff_t)rs, 16, height, 0);
}

static NOINLINE uint64_t psy_32x8_avx512(const uint8_t* s, uint32_t ss, const uint8_t* r, uint32_t rs) {
    return (uint64_t)psy_sum_gaps_avx512(
        psy_gap8_four_avx512((const uint8_t*)s, (ptrdiff_t)ss, (const uint8_t*)r, (ptrdiff_t)rs, 32, 0));
}

uint64_t svt_psy_distortion_avx512(const uint8_t* input, uint32_t input_stride, const uint8_t* recon,
                                   uint32_t recon_stride, uint32_t width, uint32_t height) {
    if (width == 4 && height == 4) {
        return psy_small4_avx512(input, input_stride, recon, recon_stride);
    }
    if (width == 8 && height == 8) {
        return psy_small8_avx512(input, input_stride, recon, recon_stride);
    }
    if ((width == 4 && (height == 8 || height == 16)) || (height == 4 && (width == 8 || width == 16))) {
        return psy_strip4_lbd_avx512(input, input_stride, recon, recon_stride, width, height);
    }
    if (width == 8 && height == 16) {
        return psy_8x16_avx512(input, input_stride, recon, recon_stride);
    }
    if (width == 8 && height == 32) {
        return psy_8x32_avx512(input, input_stride, recon, recon_stride);
    }
    if (width == 16 && height >= 8 && height <= 64 && !(height & (height - 1))) {
        return psy_width16_avx512(input, input_stride, recon, recon_stride, width, height);
    }
    if (width == 32 && height == 8) {
        return psy_32x8_avx512(input, input_stride, recon, recon_stride);
    }
    return psy_distortion_wide_avx512(input, input_stride, recon, recon_stride, width, height);
}
