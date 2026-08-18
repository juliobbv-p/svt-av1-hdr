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
    const int      filter_len_half = sizeof(svt_aom_av1_down2_symeven_half_filter) /
        sizeof(svt_aom_av1_down2_symeven_half_filter[0]);
    // The vectorized paths below hardcode these taps as compile-time
    // constants instead of reading filter[j] at runtime - this assert
    // keeps them in sync with the canonical array, which this filter's
    // fixed 2:1 downsampling role means won't change.
    assert(filter_len_half == 4 && filter[0] == 56 && filter[1] == 12 && filter[2] == -3 && filter[3] == -1);
    int      i, j;
    uint8_t* optr = output;
    int      l1   = filter_len_half;
    int      l2   = (length - filter_len_half);
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

    // Middle part: unclamped region, split into an 8-wide batch (this section)
    // and a 1-wide tail (below) for whatever doesn't fill a full 8-batch.
    //
    // For tap j and output center c = i+2k (k = 0..7), lane k of the
    // stride-2 deinterleaved loads laJ/raJ holds input[c-j] and
    // input[c+1+j]; pJ = laJ+raJ is that tap's pair sum, in [0,510]. The
    // per-output result is sum(filter[j] * pJ) for j=0..3 - see the tap
    // grouping and int16-safety comment below for how p0..p3 combine.
    //
    // Guarded by two conditions: i+16<=l2 (equivalent to "last center
    // i+14 < l2", since i and l2 are both even) keeps all 8 output centers
    // inside the already-safe unclamped region; i+19<=length keeps the
    // furthest-reaching physical vld2_u8 read (v_d below, touching
    // input[i+3..i+18]) from reading past the end of input, even though
    // only its even-offset lanes are actually used.
    while (i + 16 <= l2 && i + 19 <= length) {
        // vld2_u8(ptr).val[1] is always the same 8 lanes as
        // vld2_u8(ptr+1).val[0] (both are the odd-offset-from-ptr bytes), so
        // 4 loads 2 bytes apart cover the 8 streams that 8 individually
        // offset loads would - la3/la2 from v_a, la1/la0 from v_b, ra0/ra1
        // from v_c, ra2/ra3 from v_d.
        const uint8x8x2_t v_a = vld2_u8(input + i - 3);
        const uint8x8x2_t v_b = vld2_u8(input + i - 1);
        const uint8x8x2_t v_c = vld2_u8(input + i + 1);
        const uint8x8x2_t v_d = vld2_u8(input + i + 3);

        const int16x8_t p0 = vreinterpretq_s16_u16(vaddl_u8(v_b.val[1], v_c.val[0])); // la0+ra0
        const int16x8_t p1 = vreinterpretq_s16_u16(vaddl_u8(v_b.val[0], v_c.val[1])); // la1+ra1
        const int16x8_t p2 = vreinterpretq_s16_u16(vaddl_u8(v_a.val[1], v_d.val[0])); // la2+ra2
        const int16x8_t p3 = vreinterpretq_s16_u16(vaddl_u8(v_a.val[0], v_d.val[1])); // la3+ra3

        // Each pair p0..p3 is in [0,510]. Pairing taps (0,2) and (1,3) - one
        // positive and one negative coefficient each, from filter
        // {56,12,-3,-1} - keeps both partial sums within int16 range even
        // at the extremes
        // (56*510-3*0=28560 down to 56*0-3*510=-1530 for the first group;
        // 12*510-1*0=6120 down to 12*0-1*510=-510 for the second), so they
        // can accumulate natively in 16 bits instead of widening every tap
        // to 32 bits. Grouping taps (0,1) or (0,3) instead would let the
        // partial sum reach 56*510+12*510=34680, which overflows int16.
        int16x8_t a = vmulq_n_s16(p0, 56);
        a           = vmlaq_n_s16(a, p2, -3);
        // p3's tap is -1; use an explicit subtract rather than relying on
        // the compiler to fold a *(-1) multiply into one.
        int16x8_t b = vmulq_n_s16(p1, 12);
        b           = vsubq_s16(b, p3);

        // a+b (the full sum) can itself exceed int16 range (up to 28560+6120
        // = 34680), so combine with a saturating add rather than a plain
        // one. That's still safe: 34680 saturates to INT16_MAX (32767), and
        // clip_pixel((32767+64)>>7) = clip_pixel(256) = 255, the same
        // output every true sum >= 32576 already clips to (32576 being the
        // smallest sum for which (sum+64)>>7 >= 255) - so the saturated
        // approximation and the exact sum always agree after the shift and
        // clip. The negative side never approaches int16's range (minimum
        // is -1530-510 = -2040), so it never saturates.
        const int16x8_t acc = vqaddq_s16(a, b);
        vst1_u8(optr, vqrshrun_n_s16(acc, FILTER_BITS));
        optr += 8;
        i += 16;
    }

    // 1-wide tail: covers what's left between the 8-wide batch above and
    // l2 (at most 7 outputs, or the whole unclamped region if the batch
    // above never ran). w = input[i-3 .. i+4]; vrev64_u8(w)[k] = w[7-k], so
    // vaddl_u8(w, vrev64_u8(w))[0..3] = [w0+w7, w1+w6, w2+w5, w3+w4]
    //                                 = [pair(j=3), pair(j=2), pair(j=1), pair(j=0)].
    // Dotting with the reversed filter {filter[3],filter[2],filter[1],filter[0]}
    // reproduces sum(filter[j] * pair(j)) for j=0..3, order-independent since it's
    // plain integer addition. Hardcoded like the taps above (guarded by the
    // same assert), rather than derived from `filter` at runtime.
    static const int16_t filter_rev[4]  = {-1, -3, 12, 56};
    const int16x4_t      filter_rev_vec = vld1_s16(filter_rev);
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
