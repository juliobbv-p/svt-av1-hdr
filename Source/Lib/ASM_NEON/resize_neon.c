/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>

#include "definitions.h"
#include "resize.h"

void svt_av1_down2_symeven_neon(const uint8_t* const input, int length, uint8_t* output) {
    const int16_t* filter          = svt_aom_av1_down2_symeven_half_filter;
    const int      filter_len_half = sizeof(svt_aom_av1_down2_symeven_half_filter) / 2;
    int            i, j;
    uint8_t*       optr = output;
    int            l1   = filter_len_half;
    int            l2   = (length - filter_len_half);
    l1 += (l1 & 1);
    l2 += (l2 & 1);

    if (l1 > l2) {
        // Short input length: no room for an unclamped middle part.
        for (i = 0; i < length; i += 2) {
            int sum = (1 << (FILTER_BITS - 1));
            for (j = 0; j < filter_len_half; ++j) {
                sum += (input[AOMMAX(i - j, 0)] + input[AOMMIN(i + 1 + j, length - 1)]) * filter[j];
            }
            sum >>= FILTER_BITS;
            *optr++ = clip_pixel(sum);
        }
        return;
    }

    // Initial part: left edge clamped, right side always in-bounds here.
    for (i = 0; i < l1; i += 2) {
        int sum = (1 << (FILTER_BITS - 1));
        for (j = 0; j < filter_len_half; ++j) {
            sum += (input[AOMMAX(i - j, 0)] + input[i + 1 + j]) * filter[j];
        }
        sum >>= FILTER_BITS;
        *optr++ = clip_pixel(sum);
    }

    // Middle part: unclamped, 8-byte window per output sample.
    // w = input[i-3 .. i+4]; vrev64_u8(w)[k] = w[7-k], so
    // vaddl_u8(w, vrev64_u8(w))[0..3] = [w0+w7, w1+w6, w2+w5, w3+w4]
    //                                 = [pair(j=3), pair(j=2), pair(j=1), pair(j=0)].
    // Dotting with the reversed filter {filter[3],filter[2],filter[1],filter[0]}
    // reproduces sum(filter[j] * pair(j)) for j=0..3, order-independent since it's
    // plain integer addition. Derive the reversed constant from `filter` itself
    // (vrev64_s16 reverses all 4 lanes, same trick as vrev64_u8 above) so it can't
    // drift out of sync if the canonical filter is ever changed.
    const int16x4_t filter_rev_vec = vrev64_s16(vld1_s16(filter));
    for (; i < l2; i += 2) {
        const uint8x8_t  w         = vld1_u8(input + i - 3);
        const uint8x8_t  w_rev     = vrev64_u8(w);
        const uint16x8_t pairsum16 = vaddl_u8(w, w_rev);
        const int16x4_t  pairsum   = vreinterpret_s16_u16(vget_low_u16(pairsum16));
        const int32x4_t  prod      = vmull_s16(pairsum, filter_rev_vec);
        int32_t          sum       = (1 << (FILTER_BITS - 1)) + vaddvq_s32(prod);
        sum >>= FILTER_BITS;
        *optr++ = clip_pixel(sum);
    }

    // End part: right edge clamped, left side always in-bounds here.
    for (; i < length; i += 2) {
        int sum = (1 << (FILTER_BITS - 1));
        for (j = 0; j < filter_len_half; ++j) {
            sum += (input[i - j] + input[AOMMIN(i + 1 + j, length - 1)]) * filter[j];
        }
        sum >>= FILTER_BITS;
        *optr++ = clip_pixel(sum);
    }
}
