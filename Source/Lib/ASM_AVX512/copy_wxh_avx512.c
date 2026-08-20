/*
* Copyright(c) 2024 Meta Platforms, Inc. and affiliates.
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include "definitions.h"

#if EN_AVX512_SUPPORT

#include <immintrin.h>
#include <string.h>
#include "common_dsp_rtcd.h"

// Copy `nbytes` per row for `height` rows, unaligned. Strides are in bytes.
// Every case is a compile-time-constant size -> straight-line, fully-unrolled
// stores (no runtime-size memcpy, no runtime-bounded loop).
static AOM_FORCE_INLINE void svt_copy_rows_avx512(const uint8_t* src, uint32_t src_stride_b, uint8_t* dst,
                                                  uint32_t dst_stride_b, uint32_t height, uint32_t nbytes) {
    switch (nbytes) {
    case 4: // 32-bit scalar move in a GPR (no vector register); constant memcpy -> mov r32
        for (uint32_t j = 0; j < height; ++j) {
            memcpy(dst + (size_t)j * dst_stride_b, src + (size_t)j * src_stride_b, 4);
        }
        break;
    case 8: // 64-bit scalar move in a GPR (no vector register); constant memcpy -> mov r64
        for (uint32_t j = 0; j < height; ++j) {
            memcpy(dst + (size_t)j * dst_stride_b, src + (size_t)j * src_stride_b, 8);
        }
        break;
    case 16:
        for (uint32_t j = 0; j < height; ++j) {
            const uint8_t* s = src + (size_t)j * src_stride_b;
            uint8_t*       d = dst + (size_t)j * dst_stride_b;
            _mm_storeu_si128((__m128i*)d, _mm_loadu_si128((const __m128i*)s));
        }
        break;
    case 32:
        for (uint32_t j = 0; j < height; ++j) {
            const uint8_t* s = src + (size_t)j * src_stride_b;
            uint8_t*       d = dst + (size_t)j * dst_stride_b;
            _mm256_storeu_si256((__m256i*)d, _mm256_loadu_si256((const __m256i*)s));
        }
        break;
    case 64:
        for (uint32_t j = 0; j < height; ++j) {
            const uint8_t* s = src + (size_t)j * src_stride_b;
            uint8_t*       d = dst + (size_t)j * dst_stride_b;
            _mm512_storeu_si512((void*)d, _mm512_loadu_si512((const void*)s));
        }
        break;
    case 128:
        for (uint32_t j = 0; j < height; ++j) {
            const uint8_t* s = src + (size_t)j * src_stride_b;
            uint8_t*       d = dst + (size_t)j * dst_stride_b;
            _mm512_storeu_si512((void*)(d + 0), _mm512_loadu_si512((const void*)(s + 0)));
            _mm512_storeu_si512((void*)(d + 64), _mm512_loadu_si512((const void*)(s + 64)));
        }
        break;
    case 256:
        for (uint32_t j = 0; j < height; ++j) {
            const uint8_t* s = src + (size_t)j * src_stride_b;
            uint8_t*       d = dst + (size_t)j * dst_stride_b;
            _mm512_storeu_si512((void*)(d + 0), _mm512_loadu_si512((const void*)(s + 0)));
            _mm512_storeu_si512((void*)(d + 64), _mm512_loadu_si512((const void*)(s + 64)));
            _mm512_storeu_si512((void*)(d + 128), _mm512_loadu_si512((const void*)(s + 128)));
            _mm512_storeu_si512((void*)(d + 192), _mm512_loadu_si512((const void*)(s + 192)));
        }
        break;
    default: // non-power-of-two widths only (rare)
        for (uint32_t j = 0; j < height; ++j) {
            memcpy(dst + (size_t)j * dst_stride_b, src + (size_t)j * src_stride_b, nbytes);
        }
        break;
    }
}

void svt_av1_copy_wxh_8bit_avx512(uint8_t* src, uint32_t src_stride, uint8_t* dst, uint32_t dst_stride, uint32_t height,
                                  uint32_t width) {
    svt_copy_rows_avx512(src, src_stride, dst, dst_stride, height, width);
}

void svt_av1_copy_wxh_16bit_avx512(uint16_t* src, uint32_t src_stride, uint16_t* dst, uint32_t dst_stride,
                                   uint32_t height, uint32_t width) {
    svt_copy_rows_avx512((const uint8_t*)src, src_stride << 1, (uint8_t*)dst, dst_stride << 1, height, width << 1);
}

#endif // EN_AVX512_SUPPORT
