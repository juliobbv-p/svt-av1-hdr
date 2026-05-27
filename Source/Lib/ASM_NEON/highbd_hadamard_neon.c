/*
 * Copyright (c) 2023 The WebM project authors. All rights reserved.
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved.
 *
 *  This source code is subject to the terms of the BSD 2 Clause License and
 *  the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 *  was not distributed with this source code in the LICENSE file, you can
 *  obtain it at www.aomedia.org/license/software. If the Alliance for Open
 *  Media Patent License 1.0 was not distributed with this source code in the
 *  PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */
#include <arm_neon.h>

#include "aom_dsp_rtcd.h"
#include "coding_loop.h"
#include "definitions.h"
#include "hadamard_path_neon.h"
#include "mem_neon.h"
#include "sum_neon.h"
#include "transpose_neon.h"

static inline void hadamard_highbd_col4_second_pass(int16x4_t a0, int16x4_t a1, int16x4_t a2, int16x4_t a3,
                                                    int16x4_t a4, int16x4_t a5, int16x4_t a6, int16x4_t a7,
                                                    int32_t* coeff) {
    int32x4_t b0 = vaddl_s16(a0, a1);
    int32x4_t b1 = vsubl_s16(a0, a1);
    int32x4_t b2 = vaddl_s16(a2, a3);
    int32x4_t b3 = vsubl_s16(a2, a3);
    int32x4_t b4 = vaddl_s16(a4, a5);
    int32x4_t b5 = vsubl_s16(a4, a5);
    int32x4_t b6 = vaddl_s16(a6, a7);
    int32x4_t b7 = vsubl_s16(a6, a7);

    int32x4_t c0 = vaddq_s32(b0, b2);
    int32x4_t c2 = vsubq_s32(b0, b2);
    int32x4_t c1 = vaddq_s32(b1, b3);
    int32x4_t c3 = vsubq_s32(b1, b3);
    int32x4_t c4 = vaddq_s32(b4, b6);
    int32x4_t c6 = vsubq_s32(b4, b6);
    int32x4_t c5 = vaddq_s32(b5, b7);
    int32x4_t c7 = vsubq_s32(b5, b7);

    int32x4_t d0 = vaddq_s32(c0, c4);
    int32x4_t d2 = vsubq_s32(c0, c4);
    int32x4_t d7 = vaddq_s32(c1, c5);
    int32x4_t d6 = vsubq_s32(c1, c5);
    int32x4_t d3 = vaddq_s32(c2, c6);
    int32x4_t d1 = vsubq_s32(c2, c6);
    int32x4_t d4 = vaddq_s32(c3, c7);
    int32x4_t d5 = vsubq_s32(c3, c7);

    vst1q_s32(coeff + 0, d0);
    vst1q_s32(coeff + 4, d1);
    vst1q_s32(coeff + 8, d2);
    vst1q_s32(coeff + 12, d3);
    vst1q_s32(coeff + 16, d4);
    vst1q_s32(coeff + 20, d5);
    vst1q_s32(coeff + 24, d6);
    vst1q_s32(coeff + 28, d7);
}

void svt_aom_highbd_hadamard_8x8_neon(const int16_t* src_diff, ptrdiff_t src_stride, int32_t* coeff) {
    int16x4_t b0, b1, b2, b3, b4, b5, b6, b7;

    int16x8_t s[8] = {
        vld1q_s16(src_diff + 0 * src_stride),
        vld1q_s16(src_diff + 1 * src_stride),
        vld1q_s16(src_diff + 2 * src_stride),
        vld1q_s16(src_diff + 3 * src_stride),
        vld1q_s16(src_diff + 4 * src_stride),
        vld1q_s16(src_diff + 5 * src_stride),
        vld1q_s16(src_diff + 6 * src_stride),
        vld1q_s16(src_diff + 7 * src_stride),
    };

    // For the first pass we can stay in 16-bit elements (4095*8 = 32760).
    hadamard_8x8_v_pass_neon(s);

    transpose_elems_inplace_s16_8x8(&s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7]);

    // For the second pass we need to widen to 32-bit elements, so we're
    // processing 4 columns at a time.
    // Skip the second transpose because it is not required.

    b0 = vget_low_s16(s[0]);
    b1 = vget_low_s16(s[1]);
    b2 = vget_low_s16(s[2]);
    b3 = vget_low_s16(s[3]);
    b4 = vget_low_s16(s[4]);
    b5 = vget_low_s16(s[5]);
    b6 = vget_low_s16(s[6]);
    b7 = vget_low_s16(s[7]);

    hadamard_highbd_col4_second_pass(b0, b1, b2, b3, b4, b5, b6, b7, coeff);

    b0 = vget_high_s16(s[0]);
    b1 = vget_high_s16(s[1]);
    b2 = vget_high_s16(s[2]);
    b3 = vget_high_s16(s[3]);
    b4 = vget_high_s16(s[4]);
    b5 = vget_high_s16(s[5]);
    b6 = vget_high_s16(s[6]);
    b7 = vget_high_s16(s[7]);

    hadamard_highbd_col4_second_pass(b0, b1, b2, b3, b4, b5, b6, b7, coeff + 32);
}

int svt_av1_highbd_hadamard_satd_4x4_neon(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                          ptrdiff_t pred_stride) {
    const uint16x8_t s02 = load_u16_4x2(src + 0 * src_stride, 2 * src_stride);
    const uint16x8_t s13 = load_u16_4x2(src + 1 * src_stride, 2 * src_stride);
    const uint16x8_t p02 = load_u16_4x2(pred + 0 * pred_stride, 2 * pred_stride);
    const uint16x8_t p13 = load_u16_4x2(pred + 1 * pred_stride, 2 * pred_stride);

    const int16x8_t d02 = vreinterpretq_s16_u16(vsubq_u16(s02, p02));
    const int16x8_t d13 = vreinterpretq_s16_u16(vsubq_u16(s13, p13));

    int16x8_t a0 = vhaddq_s16(d02, d13);
    int16x8_t a1 = vhsubq_s16(d02, d13);

    int16x8_t b0 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a0), vreinterpretq_s64_s16(a1)));
    int16x8_t b1 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a0), vreinterpretq_s64_s16(a1)));

    a0 = vaddq_s16(b0, b1);
    a1 = vsubq_s16(b0, b1);

    b0 = vtrn1q_s16(a0, a1);
    b1 = vtrn2q_s16(a0, a1);

    a0 = vabsq_s16(vhaddq_s16(b0, b1));
    a1 = vabsq_s16(vhsubq_s16(b0, b1));

    b0 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a1)));
    b1 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a1)));

    return vaddlvq_s16(vmaxq_s16(b0, b1)) << 1;
}

static void highbd_hadamard_8x8_coeff_neon(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                           ptrdiff_t pred_stride, int32x4_t coeff[16]) {
    EB_ASSUME(pred != NULL);
    highbd_hadamard_8x8_neon(src, src_stride, pred, pred_stride, coeff, NULL, NULL);
}

int svt_av1_highbd_hadamard_satd_8x8_neon(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                          ptrdiff_t pred_stride) {
    EB_ASSUME(pred != NULL);
    uint32x4_t satd;
    highbd_hadamard_8x8_neon(src, src_stride, pred, pred_stride, NULL, &satd, NULL);
    return vaddvq_u32(satd) << 1;
}

int svt_av1_highbd_hadamard_satd_16x16_neon(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                            ptrdiff_t pred_stride) {
    int32x4_t q0[16], q1[16], q2[16], q3[16];
    highbd_hadamard_8x8_coeff_neon(src + 0 + 0 * src_stride, src_stride, pred + 0 + 0 * pred_stride, pred_stride, q0);
    highbd_hadamard_8x8_coeff_neon(src + 8 + 0 * src_stride, src_stride, pred + 8 + 0 * pred_stride, pred_stride, q1);
    highbd_hadamard_8x8_coeff_neon(src + 0 + 8 * src_stride, src_stride, pred + 0 + 8 * pred_stride, pred_stride, q2);
    highbd_hadamard_8x8_coeff_neon(src + 8 + 8 * src_stride, src_stride, pred + 8 + 8 * pred_stride, pred_stride, q3);

    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    for (int i = 0; i < 16; ++i) {
        const int32x4_t a0 = vabsq_s32(vhaddq_s32(q0[i], q1[i]));
        const int32x4_t a1 = vabsq_s32(vhsubq_s32(q0[i], q1[i]));
        const int32x4_t a2 = vabsq_s32(vhaddq_s32(q2[i], q3[i]));
        const int32x4_t a3 = vabsq_s32(vhsubq_s32(q2[i], q3[i]));

        acc0 = vaddq_s32(acc0, vmaxq_s32(a0, a2));
        acc1 = vaddq_s32(acc1, vmaxq_s32(a1, a3));
    }
    return vaddvq_s32(vaddq_s32(acc0, acc1)) << 1;
}

static void hadamard_highbd_16x16_neon(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                       ptrdiff_t pred_stride, int32x4_t coeff[64]) {
    int32x4_t q0[16], q1[16], q2[16], q3[16];
    highbd_hadamard_8x8_coeff_neon(src + 0 + 0 * src_stride, src_stride, pred + 0 + 0 * pred_stride, pred_stride, q0);
    highbd_hadamard_8x8_coeff_neon(src + 8 + 0 * src_stride, src_stride, pred + 8 + 0 * pred_stride, pred_stride, q1);
    highbd_hadamard_8x8_coeff_neon(src + 0 + 8 * src_stride, src_stride, pred + 0 + 8 * pred_stride, pred_stride, q2);
    highbd_hadamard_8x8_coeff_neon(src + 8 + 8 * src_stride, src_stride, pred + 8 + 8 * pred_stride, pred_stride, q3);

    for (int i = 0; i < 16; ++i) {
        const int32x4_t b0 = vhaddq_s32(q0[i], q1[i]);
        const int32x4_t b1 = vhsubq_s32(q0[i], q1[i]);
        const int32x4_t b2 = vhaddq_s32(q2[i], q3[i]);
        const int32x4_t b3 = vhsubq_s32(q2[i], q3[i]);

        coeff[4 * i + 0] = vaddq_s32(b0, b2);
        coeff[4 * i + 1] = vaddq_s32(b1, b3);
        coeff[4 * i + 2] = vsubq_s32(b0, b2);
        coeff[4 * i + 3] = vsubq_s32(b1, b3);
    }
}

int svt_av1_highbd_hadamard_satd_32x32_neon(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                            ptrdiff_t pred_stride) {
    int32x4_t q0[64], q1[64], q2[64], q3[64];
    hadamard_highbd_16x16_neon(src + 0 + 0 * src_stride, src_stride, pred + 0 + 0 * pred_stride, pred_stride, q0);
    hadamard_highbd_16x16_neon(src + 16 + 0 * src_stride, src_stride, pred + 16 + 0 * pred_stride, pred_stride, q1);
    hadamard_highbd_16x16_neon(src + 0 + 16 * src_stride, src_stride, pred + 0 + 16 * pred_stride, pred_stride, q2);
    hadamard_highbd_16x16_neon(src + 16 + 16 * src_stride, src_stride, pred + 16 + 16 * pred_stride, pred_stride, q3);

    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    for (int i = 0; i < 64; ++i) {
        const int32x4_t a0 = vabsq_s32(vshrq_n_s32(vaddq_s32(q0[i], q1[i]), 2));
        const int32x4_t a1 = vabsq_s32(vshrq_n_s32(vsubq_s32(q0[i], q1[i]), 2));
        const int32x4_t a2 = vabsq_s32(vshrq_n_s32(vaddq_s32(q2[i], q3[i]), 2));
        const int32x4_t a3 = vabsq_s32(vshrq_n_s32(vsubq_s32(q2[i], q3[i]), 2));

        acc0 = vaddq_s32(acc0, vmaxq_s32(a0, a2));
        acc1 = vaddq_s32(acc1, vmaxq_s32(a1, a3));
    }
    return vaddvq_s32(vaddq_s32(acc0, acc1)) << 1;
}
