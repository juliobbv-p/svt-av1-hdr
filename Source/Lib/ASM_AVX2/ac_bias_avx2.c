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

// Two 4x4 tiles retain their columns throughout both passes. With vertical
// tiles, pack four samples from each tile into a virtual eight-sample row.
SIMD_INLINE __m128i psy_energy4_pair_avx2(const uint8_t* src, ptrdiff_t stride, int vertical, int hbd) {
    __m128i row[4];
    for (int i = 0; i < 4; ++i) {
        const uint8_t* p = src + i * stride;
        if (!vertical) {
            row[i] = hbd ? _mm_loadu_si128((const __m128i*)p) : _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)p));
        } else if (hbd) {
            row[i] = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)p),
                                        _mm_loadl_epi64((const __m128i*)(p + 4 * stride)));
        } else {
            uint32_t top, bottom;
            memcpy(&top, p, sizeof(top));
            memcpy(&bottom, p + 4 * stride, sizeof(bottom));
            row[i] = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(_mm_cvtsi32_si128((int)top), _mm_cvtsi32_si128((int)bottom)));
        }
    }
    return psy_energy4_rows_avx2(row);
}

SIMD_INLINE uint32_t psy_gap4_pair_avx2(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, int vertical,
                                        int hbd) {
    const __m128i a   = psy_energy4_pair_avx2(s, ss, vertical, hbd);
    const __m128i b   = psy_energy4_pair_avx2(r, rs, vertical, hbd);
    const __m128i gap = _mm_abs_epi32(_mm_sub_epi32(a, b));
    return (uint32_t)_mm_cvtsi128_si32(_mm_add_epi32(gap, _mm_shuffle_epi32(gap, 0x4e)));
}

// Complete each two-tile transform before loading the next. Retain only
// its partial sum and DC, keeping eight transform registers live.
SIMD_INLINE __m256i psy_partial8_avx2(const uint8_t* src, ptrdiff_t stride, int hbd, __m256i* dc) {
    __m256i a[8];
    for (int i = 0; i < 8; ++i) {
        a[i] = hbd ? _mm256_loadu_si256((const __m256i*)(src + i * stride))
                   : _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(src + i * stride)));
    }
    hadamard_v8_avx2(a, 1);
    hadamard_transpose8_avx2(a);
    hadamard_v4_avx2(a, 1);
    *dc = _mm256_add_epi16(a[0], a[4]);
    return hadamard_psy8_sum_avx2(a, hbd);
}

// Share the reduction of four neighboring tiles or two source/recon pairs.
// Unpack/add leaves [E0,E0,E2,E2] and [E1,E1,E3,E3] in the two lanes;
// each caller consumes each tile energy once.
SIMD_INLINE __m256i psy_reduce8_pair_avx2(__m256i s0, __m256i s1, __m256i dc0, __m256i dc1, int hbd) {
    __m256i sum = _mm256_add_epi32(_mm256_unpacklo_epi64(s0, s1), _mm256_unpackhi_epi64(s0, s1));
    sum         = _mm256_add_epi32(sum, _mm256_shuffle_epi32(sum, 0xb1));
    __m256i dc  = _mm256_unpacklo_epi64(dc0, dc1);
    dc          = _mm256_srli_epi16(_mm256_add_epi16(dc, _mm256_set1_epi16(2)), 2);
    dc          = _mm256_shuffle_epi32(_mm256_and_si256(dc, _mm256_set1_epi32(65535)), 0xa0);
    return _mm256_sub_epi32(_mm256_srli_epi32(_mm256_add_epi32(sum, _mm256_set1_epi32(hbd ? (8 * 32768 + 1) : 1)), 1),
                            dc);
}

SIMD_INLINE __m256i psy_energy8_four_avx2(const uint8_t* src, ptrdiff_t stride, int hbd) {
    __m256i       dc0, dc1;
    const __m256i s0 = psy_partial8_avx2(src, stride, hbd, &dc0);
    const __m256i s1 = psy_partial8_avx2(src + (16 << hbd), stride, hbd, &dc1);
    return psy_reduce8_pair_avx2(s0, s1, dc0, dc1, hbd);
}

// Cropped regions reuse the same vector workers as coded blocks.
static NOINLINE uint64_t psy_distortion_cropped_avx2(const uint8_t* input, ptrdiff_t ss, const uint8_t* recon,
                                                     ptrdiff_t rs, uint32_t width, uint32_t height, int hbd) {
    return psy_distortion_tiles(input,
                                ss,
                                recon,
                                rs,
                                width,
                                height,
                                hbd,
                                svt_psy_distortion_avx2,
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
                                svt_psy_distortion_hbd_avx2
#else
                                NULL
#endif
    );
}

// Four adjacent tiles per iteration. For coded blocks through 128x128,
// even the loose total bound 256*1024*1023 fits signed dwords.
SIMD_INLINE uint64_t psy_bands_avx2(const uint8_t* input, ptrdiff_t ss, const uint8_t* recon, ptrdiff_t rs,
                                    uint32_t width, uint32_t height, int hbd) {
    __m256i gaps = _mm256_setzero_si256();
    for (uint32_t y = 0; y < height; y += 8) {
        for (uint32_t x = 0; x < width; x += 32) {
            const __m256i a = psy_energy8_four_avx2(input + y * ss + (x << hbd), ss, hbd);
            const __m256i b = psy_energy8_four_avx2(recon + y * rs + (x << hbd), rs, hbd);
            gaps            = _mm256_add_epi32(gaps, _mm256_abs_epi32(_mm256_sub_epi32(a, b)));
        }
    }
    gaps = _mm256_add_epi32(gaps, _mm256_shuffle_epi32(gaps, 0x4e));
    return (uint64_t)(uint32_t)_mm_cvtsi128_si32(
               _mm_add_epi32(_mm256_castsi256_si128(gaps), _mm256_extracti128_si256(gaps, 1)))
        << (2 * hbd);
}

// Coded four-pixel strips have length 8 or 16. Cropped tails go through the
// common fallback, so no single-tile transform is duplicated in this loop.
SIMD_INLINE uint64_t psy_strip_avx2(const uint8_t* input, ptrdiff_t ss, const uint8_t* recon, ptrdiff_t rs,
                                    uint32_t width, uint32_t height, int hbd) {
    const int       vertical = width == 4;
    const uint32_t  length   = vertical ? height : width;
    const ptrdiff_t step_s   = vertical ? ss : (1 << hbd);
    const ptrdiff_t step_r   = vertical ? rs : (1 << hbd);
    uint64_t        gap      = 0;
    uint32_t        n        = 0;
    for (; length - n >= 8; n += 8) {
        gap += psy_gap4_pair_avx2(input + n * step_s, ss, recon + n * step_r, rs, vertical, hbd);
    }
    return gap << (2 * hbd);
}

SIMD_INLINE uint64_t psy_width16_avx2(const uint8_t* input, ptrdiff_t ss, const uint8_t* recon, ptrdiff_t rs,
                                      uint32_t height, int hbd) {
    __m256i gaps = _mm256_setzero_si256();
    for (uint32_t y = 0; y < height; y += 8) {
        __m256i       dc0, dc1;
        const __m256i s0 = psy_partial8_avx2(input + y * ss, ss, hbd, &dc0);
        const __m256i s1 = psy_partial8_avx2(recon + y * rs, rs, hbd, &dc1);
        const __m256i e  = psy_reduce8_pair_avx2(s0, s1, dc0, dc1, hbd);
        gaps             = _mm256_add_epi32(gaps, _mm256_abs_epi32(_mm256_sub_epi32(e, _mm256_shuffle_epi32(e, 0x4e))));
    }
    return (uint64_t)(uint32_t)_mm_cvtsi128_si32(
               _mm_add_epi32(_mm256_castsi256_si128(gaps), _mm256_extracti128_si256(gaps, 1)))
        << (2 * hbd);
}

SIMD_INLINE uint64_t psy_width8_avx2(const uint8_t* input, ptrdiff_t ss, const uint8_t* recon, ptrdiff_t rs,
                                     uint32_t height, int hbd) {
    uint64_t gap = 0;
    for (uint32_t y = 0; y < height; y += 8) {
        gap += psy_gap8_avx2(input + y * ss, ss, recon + y * rs, rs, hbd);
    }
    return gap << (2 * hbd);
}

SIMD_INLINE uint64_t psy_wide_avx2(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, uint32_t width,
                                   uint32_t height, int hbd) {
    if (width == 16 && height >= 8) {
        return psy_width16_avx2(s, ss, r, rs, height, hbd);
    }
    if ((width == 4 && (height == 8 || height == 16)) || (height == 4 && (width == 8 || width == 16))) {
        return psy_strip_avx2(s, ss, r, rs, width, height, hbd);
    }
    if (width >= 32 && width <= 128 && height >= 8 && height <= 128 && !(width & (width - 1)) &&
        !(height & (height - 1)) && width <= 4 * height && height <= 4 * width) {
        return psy_bands_avx2(s, ss, r, rs, width, height, hbd);
    }
    return psy_distortion_cropped_avx2(s, ss, r, rs, width, height, hbd);
}

// Isolate the small and narrow transforms from the wide driver's register
// saves. Typed workers make bit depth constant inside each vector loop.
static NOINLINE uint64_t psy_small4_worker_avx2(const uint8_t* input, uint32_t input_stride, const uint8_t* recon,
                                                uint32_t recon_stride, uint32_t width, uint32_t height) {
    const uint8_t*  s  = (const uint8_t*)input;
    const uint8_t*  r  = (const uint8_t*)recon;
    const ptrdiff_t ss = (ptrdiff_t)input_stride, rs = (ptrdiff_t)recon_stride;
    (void)width;
    (void)height;
    return (uint64_t)psy_gap4_avx2(s, ss, r, rs, 0);
}

static NOINLINE uint64_t psy_width8_worker_avx2(const uint8_t* input, uint32_t input_stride, const uint8_t* recon,
                                                uint32_t recon_stride, uint32_t width, uint32_t height) {
    const uint8_t*  s  = (const uint8_t*)input;
    const uint8_t*  r  = (const uint8_t*)recon;
    const ptrdiff_t ss = (ptrdiff_t)input_stride, rs = (ptrdiff_t)recon_stride;
    (void)width;
    return psy_width8_avx2(s, ss, r, rs, height, 0);
}

static NOINLINE uint64_t psy_wide_worker_avx2(const uint8_t* input, uint32_t input_stride, const uint8_t* recon,
                                              uint32_t recon_stride, uint32_t width, uint32_t height) {
    const uint8_t*  s  = (const uint8_t*)input;
    const uint8_t*  r  = (const uint8_t*)recon;
    const ptrdiff_t ss = (ptrdiff_t)input_stride, rs = (ptrdiff_t)recon_stride;
    return psy_wide_avx2(s, ss, r, rs, width, height, 0);
}

uint64_t svt_psy_distortion_avx2(const uint8_t* input, uint32_t input_stride, const uint8_t* recon,
                                 uint32_t recon_stride, uint32_t width, uint32_t height) {
    if (width == 4 && height == 4) {
        return psy_small4_worker_avx2(input, input_stride, recon, recon_stride, width, height);
    }
    if (width == 8 && height >= 8) {
        return psy_width8_worker_avx2(input, input_stride, recon, recon_stride, width, height);
    }
    return psy_wide_worker_avx2(input, input_stride, recon, recon_stride, width, height);
}
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
static NOINLINE uint64_t psy_width8_strided_avx2(const uint16_t* s, uint32_t ss, const uint16_t* r, uint32_t rs) {
    return psy_gap8x32_strided_avx2(s, ss, r, rs);
}

static NOINLINE uint64_t psy_small4_hbd_worker_avx2(const uint16_t* input, uint32_t input_stride, const uint16_t* recon,
                                                    uint32_t recon_stride, uint32_t width, uint32_t height) {
    const uint8_t*  s  = (const uint8_t*)input;
    const uint8_t*  r  = (const uint8_t*)recon;
    const ptrdiff_t ss = (ptrdiff_t)input_stride << 1, rs = (ptrdiff_t)recon_stride << 1;
    (void)width;
    (void)height;
    return (uint64_t)psy_gap4_avx2(s, ss, r, rs, 1) << 2;
}

static NOINLINE uint64_t psy_width8_hbd_worker_avx2(const uint16_t* input, uint32_t input_stride, const uint16_t* recon,
                                                    uint32_t recon_stride, uint32_t width, uint32_t height) {
    const uint8_t*  s  = (const uint8_t*)input;
    const uint8_t*  r  = (const uint8_t*)recon;
    const ptrdiff_t ss = (ptrdiff_t)input_stride << 1, rs = (ptrdiff_t)recon_stride << 1;
    (void)width;
    return psy_width8_avx2(s, ss, r, rs, height, 1);
}

static NOINLINE uint64_t psy_wide_hbd_worker_avx2(const uint16_t* input, uint32_t input_stride, const uint16_t* recon,
                                                  uint32_t recon_stride, uint32_t width, uint32_t height) {
    const uint8_t*  s  = (const uint8_t*)input;
    const uint8_t*  r  = (const uint8_t*)recon;
    const ptrdiff_t ss = (ptrdiff_t)input_stride << 1, rs = (ptrdiff_t)recon_stride << 1;
    return psy_wide_avx2(s, ss, r, rs, width, height, 1);
}

uint64_t svt_psy_distortion_hbd_avx2(const uint16_t* input, uint32_t input_stride, const uint16_t* recon,
                                     uint32_t recon_stride, uint32_t width, uint32_t height) {
    if (width == 4 && height == 4) {
        return psy_small4_hbd_worker_avx2(input, input_stride, recon, recon_stride, width, height);
    }
    if (width == 8 && height >= 8) {
        if (height == 32 && input_stride >= 2048 && (input_stride & 2047) == 0 && ((uintptr_t)input & 63) <= 48) {
            return psy_width8_strided_avx2(input, input_stride, recon, recon_stride);
        }
        return psy_width8_hbd_worker_avx2(input, input_stride, recon, recon_stride, width, height);
    }
    return psy_wide_hbd_worker_avx2(input, input_stride, recon, recon_stride, width, height);
}
#endif
