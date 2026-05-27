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

#ifndef SVT_AV1_HADAMARD_PATH_NEON_H_
#define SVT_AV1_HADAMARD_PATH_NEON_H_

#include <arm_neon.h>

#include "mem_neon.h"
#include "sum_neon.h"

static inline int psy_energy_4x4_neon(const uint8_t* src, ptrdiff_t src_stride) {
    const uint8x8_t s02_u8 = load_u8x4_strided_x2((uint8_t*)src + 0 * src_stride, 2 * src_stride);
    const uint8x8_t s13_u8 = load_u8x4_strided_x2((uint8_t*)src + 1 * src_stride, 2 * src_stride);

    const int16x8_t s02 = vreinterpretq_s16_u16(vmovl_u8(s02_u8));
    const int16x8_t s13 = vreinterpretq_s16_u16(vmovl_u8(s13_u8));

    int16x8_t a0 = vhaddq_s16(s02, s13);
    int16x8_t a1 = vhsubq_s16(s02, s13);

    int16x8_t b0 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a0), vreinterpretq_s64_s16(a1)));
    int16x8_t b1 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a0), vreinterpretq_s64_s16(a1)));

    a0 = vaddq_s16(b0, b1);
    a1 = vsubq_s16(b0, b1);

    b0 = vtrn1q_s16(a0, a1);
    b1 = vtrn2q_s16(a0, a1);

    a0 = vhaddq_s16(b0, b1);
    a1 = vhsubq_s16(b0, b1);

    b0 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a1)));
    b1 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a1)));

    const int dc = vgetq_lane_s16(vaddq_s16(b0, b1), 0);

    a0 = vabsq_s16(b0);
    a1 = vabsq_s16(b1);

    const int satd = vaddlvq_s16(vmaxq_s16(a0, a1)) << 1;
    return (satd << 1) - dc;
}

static inline int highbd_psy_energy_4x4_neon(const uint16_t* src, ptrdiff_t src_stride) {
    int16x4_t s0, s1, s2, s3;
    load_s16_4x4((int16_t*)src, src_stride, &s0, &s1, &s2, &s3);

    int16x8_t a0 = vcombine_s16(vhadd_s16(s0, s1), vhsub_s16(s0, s1));
    int16x8_t a1 = vcombine_s16(vhadd_s16(s2, s3), vhsub_s16(s2, s3));

    int16x8_t b0 = vaddq_s16(a0, a1);
    int16x8_t b1 = vsubq_s16(a0, a1);

    a0 = vtrn1q_s16(b0, b1);
    a1 = vtrn2q_s16(b0, b1);

    b0 = vhaddq_s16(a0, a1);
    b1 = vhsubq_s16(a0, a1);

    a0 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b0), vreinterpretq_s32_s16(b1)));
    a1 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b0), vreinterpretq_s32_s16(b1)));

    const int dc = vgetq_lane_s16(vaddq_s16(a0, a1), 0);

    const int16x8_t max = vmaxq_s16(vabsq_s16(a0), vabsq_s16(a1));
    const int       sum = vaddlvq_s16(max);

    return (sum << 2) - dc;
}

static inline void hadamard_8x8_v_pass_neon(int16x8_t* a) {
    const int16x8_t b0 = vaddq_s16(a[0], a[1]);
    const int16x8_t b1 = vsubq_s16(a[0], a[1]);
    const int16x8_t b2 = vaddq_s16(a[2], a[3]);
    const int16x8_t b3 = vsubq_s16(a[2], a[3]);
    const int16x8_t b4 = vaddq_s16(a[4], a[5]);
    const int16x8_t b5 = vsubq_s16(a[4], a[5]);
    const int16x8_t b6 = vaddq_s16(a[6], a[7]);
    const int16x8_t b7 = vsubq_s16(a[6], a[7]);

    const int16x8_t c0 = vaddq_s16(b0, b2);
    const int16x8_t c1 = vaddq_s16(b1, b3);
    const int16x8_t c2 = vsubq_s16(b0, b2);
    const int16x8_t c3 = vsubq_s16(b1, b3);
    const int16x8_t c4 = vaddq_s16(b4, b6);
    const int16x8_t c5 = vaddq_s16(b5, b7);
    const int16x8_t c6 = vsubq_s16(b4, b6);
    const int16x8_t c7 = vsubq_s16(b5, b7);

    a[0] = vaddq_s16(c0, c4);
    a[1] = vsubq_s16(c2, c6);
    a[2] = vsubq_s16(c0, c4);
    a[3] = vaddq_s16(c2, c6);
    a[4] = vaddq_s16(c3, c7);
    a[5] = vsubq_s16(c3, c7);
    a[6] = vsubq_s16(c1, c5);
    a[7] = vaddq_s16(c1, c5);
}

static inline void hadamard_8x8_neon(const uint8_t* src, ptrdiff_t src_stride, const uint8_t* pred,
                                     ptrdiff_t pred_stride, int16x8_t* coeff, int32x4_t* satd, int16x8_t* dc) {
    uint8x8_t s[8];
    load_u8_8x8(src, src_stride, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7]);

    int16x8_t a[8];
    if (pred != NULL) {
        uint8x8_t p[8];
        load_u8_8x8(pred, pred_stride, &p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &p[6], &p[7]);
        for (int i = 0; i < 8; ++i) {
            a[i] = vreinterpretq_s16_u16(vsubl_u8(s[i], p[i]));
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            a[i] = vreinterpretq_s16_u16(vmovl_u8(s[i]));
        }
    }

    hadamard_8x8_v_pass_neon(a);

    if (dc != NULL) {
        *dc = a[0];
    }

    int16x8_t b0 = vtrn1q_s16(a[0], a[1]);
    int16x8_t b1 = vtrn2q_s16(a[0], a[1]);
    int16x8_t b2 = vtrn1q_s16(a[2], a[3]);
    int16x8_t b3 = vtrn2q_s16(a[2], a[3]);
    int16x8_t b4 = vtrn1q_s16(a[4], a[5]);
    int16x8_t b5 = vtrn2q_s16(a[4], a[5]);
    int16x8_t b6 = vtrn1q_s16(a[6], a[7]);
    int16x8_t b7 = vtrn2q_s16(a[6], a[7]);

    a[0] = vaddq_s16(b0, b1);
    a[1] = vsubq_s16(b0, b1);
    a[2] = vaddq_s16(b2, b3);
    a[3] = vsubq_s16(b2, b3);
    a[4] = vaddq_s16(b4, b5);
    a[5] = vsubq_s16(b4, b5);
    a[6] = vaddq_s16(b6, b7);
    a[7] = vsubq_s16(b6, b7);

    b0 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a[0]), vreinterpretq_s32_s16(a[1])));
    b1 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a[0]), vreinterpretq_s32_s16(a[1])));
    b2 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a[2]), vreinterpretq_s32_s16(a[3])));
    b3 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a[2]), vreinterpretq_s32_s16(a[3])));
    b4 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a[4]), vreinterpretq_s32_s16(a[5])));
    b5 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a[4]), vreinterpretq_s32_s16(a[5])));
    b6 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a[6]), vreinterpretq_s32_s16(a[7])));
    b7 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a[6]), vreinterpretq_s32_s16(a[7])));

    if (coeff != NULL) {
        a[0] = vaddq_s16(b0, b1);
        a[1] = vsubq_s16(b0, b1);
        a[2] = vaddq_s16(b2, b3);
        a[3] = vsubq_s16(b2, b3);
        a[4] = vaddq_s16(b4, b5);
        a[5] = vsubq_s16(b4, b5);
        a[6] = vaddq_s16(b6, b7);
        a[7] = vsubq_s16(b6, b7);

        b0 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a[0]), vreinterpretq_s64_s16(a[1])));
        b1 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a[0]), vreinterpretq_s64_s16(a[1])));
        b2 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a[2]), vreinterpretq_s64_s16(a[3])));
        b3 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a[2]), vreinterpretq_s64_s16(a[3])));
        b4 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a[4]), vreinterpretq_s64_s16(a[5])));
        b5 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a[4]), vreinterpretq_s64_s16(a[5])));
        b6 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a[6]), vreinterpretq_s64_s16(a[7])));
        b7 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a[6]), vreinterpretq_s64_s16(a[7])));

        coeff[0] = vaddq_s16(b0, b1);
        coeff[1] = vsubq_s16(b0, b1);
        coeff[2] = vaddq_s16(b2, b3);
        coeff[3] = vsubq_s16(b2, b3);
        coeff[4] = vaddq_s16(b4, b5);
        coeff[5] = vsubq_s16(b4, b5);
        coeff[6] = vaddq_s16(b6, b7);
        coeff[7] = vsubq_s16(b6, b7);
        return;
    }

    a[0] = vabsq_s16(vaddq_s16(b0, b1));
    a[1] = vabdq_s16(b0, b1);
    a[2] = vabsq_s16(vaddq_s16(b2, b3));
    a[3] = vabdq_s16(b2, b3);
    a[4] = vabsq_s16(vaddq_s16(b4, b5));
    a[5] = vabdq_s16(b4, b5);
    a[6] = vabsq_s16(vaddq_s16(b6, b7));
    a[7] = vabdq_s16(b6, b7);

    b0 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a[0]), vreinterpretq_s64_s16(a[1])));
    b1 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a[0]), vreinterpretq_s64_s16(a[1])));
    b2 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a[2]), vreinterpretq_s64_s16(a[3])));
    b3 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a[2]), vreinterpretq_s64_s16(a[3])));
    b4 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a[4]), vreinterpretq_s64_s16(a[5])));
    b5 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a[4]), vreinterpretq_s64_s16(a[5])));
    b6 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a[6]), vreinterpretq_s64_s16(a[7])));
    b7 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a[6]), vreinterpretq_s64_s16(a[7])));

    int16x8_t max[4];
    max[0] = vmaxq_s16(b0, b1);
    max[1] = vmaxq_s16(b2, b3);
    max[2] = vmaxq_s16(b4, b5);
    max[3] = vmaxq_s16(b6, b7);

    *satd = horizontal_add_4d_s16x8(max);
}

static inline void highbd_hadamard_8x8_neon(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                            ptrdiff_t pred_stride, int32x4_t* coeff, uint32x4_t* satd, int32x4_t* dc) {
    uint16x8_t s[8];
    load_u16_8x8(src, src_stride, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7]);

    int16x8_t a[8];
    if (pred != NULL) {
        uint16x8_t p[8];
        load_u16_8x8(pred, pred_stride, &p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &p[6], &p[7]);
        for (int i = 0; i < 8; ++i) {
            a[i] = vreinterpretq_s16_u16(vsubq_u16(s[i], p[i]));
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            a[i] = vreinterpretq_s16_u16(s[i]);
        }
    }

    hadamard_8x8_v_pass_neon(a);

    if (dc != NULL) {
        *dc = vpaddlq_s16(a[0]);
    }

    int16x8_t a0 = vtrn1q_s16(a[0], a[1]);
    int16x8_t a1 = vtrn2q_s16(a[0], a[1]);
    int16x8_t a2 = vtrn1q_s16(a[2], a[3]);
    int16x8_t a3 = vtrn2q_s16(a[2], a[3]);
    int16x8_t a4 = vtrn1q_s16(a[4], a[5]);
    int16x8_t a5 = vtrn2q_s16(a[4], a[5]);
    int16x8_t a6 = vtrn1q_s16(a[6], a[7]);
    int16x8_t a7 = vtrn2q_s16(a[6], a[7]);

    if (coeff != NULL) {
        int16x8_t b0 = vaddq_s16(a0, a1);
        int16x8_t b1 = vsubq_s16(a0, a1);
        int16x8_t b2 = vaddq_s16(a2, a3);
        int16x8_t b3 = vsubq_s16(a2, a3);
        int16x8_t b4 = vaddq_s16(a4, a5);
        int16x8_t b5 = vsubq_s16(a4, a5);
        int16x8_t b6 = vaddq_s16(a6, a7);
        int16x8_t b7 = vsubq_s16(a6, a7);

        a0 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b0), vreinterpretq_s32_s16(b1)));
        a1 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b0), vreinterpretq_s32_s16(b1)));
        a2 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b2), vreinterpretq_s32_s16(b3)));
        a3 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b2), vreinterpretq_s32_s16(b3)));
        a4 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b4), vreinterpretq_s32_s16(b5)));
        a5 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b4), vreinterpretq_s32_s16(b5)));
        a6 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b6), vreinterpretq_s32_s16(b7)));
        a7 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b6), vreinterpretq_s32_s16(b7)));

        b0 = vaddq_s16(a0, a1);
        b1 = vsubq_s16(a0, a1);
        b2 = vaddq_s16(a2, a3);
        b3 = vsubq_s16(a2, a3);
        b4 = vaddq_s16(a4, a5);
        b5 = vsubq_s16(a4, a5);
        b6 = vaddq_s16(a6, a7);
        b7 = vsubq_s16(a6, a7);

        a0 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(b0), vreinterpretq_s64_s16(b1)));
        a1 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(b0), vreinterpretq_s64_s16(b1)));
        a2 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(b2), vreinterpretq_s64_s16(b3)));
        a3 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(b2), vreinterpretq_s64_s16(b3)));
        a4 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(b4), vreinterpretq_s64_s16(b5)));
        a5 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(b4), vreinterpretq_s64_s16(b5)));
        a6 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(b6), vreinterpretq_s64_s16(b7)));
        a7 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(b6), vreinterpretq_s64_s16(b7)));

        coeff[0]  = vaddl_s16(vget_low_s16(a0), vget_low_s16(a1));
        coeff[1]  = vsubl_s16(vget_low_s16(a0), vget_low_s16(a1));
        coeff[2]  = vaddl_s16(vget_high_s16(a0), vget_high_s16(a1));
        coeff[3]  = vsubl_s16(vget_high_s16(a0), vget_high_s16(a1));
        coeff[4]  = vaddl_s16(vget_low_s16(a2), vget_low_s16(a3));
        coeff[5]  = vsubl_s16(vget_low_s16(a2), vget_low_s16(a3));
        coeff[6]  = vaddl_s16(vget_high_s16(a2), vget_high_s16(a3));
        coeff[7]  = vsubl_s16(vget_high_s16(a2), vget_high_s16(a3));
        coeff[8]  = vaddl_s16(vget_low_s16(a4), vget_low_s16(a5));
        coeff[9]  = vsubl_s16(vget_low_s16(a4), vget_low_s16(a5));
        coeff[10] = vaddl_s16(vget_high_s16(a4), vget_high_s16(a5));
        coeff[11] = vsubl_s16(vget_high_s16(a4), vget_high_s16(a5));
        coeff[12] = vaddl_s16(vget_low_s16(a6), vget_low_s16(a7));
        coeff[13] = vsubl_s16(vget_low_s16(a6), vget_low_s16(a7));
        coeff[14] = vaddl_s16(vget_high_s16(a6), vget_high_s16(a7));
        coeff[15] = vsubl_s16(vget_high_s16(a6), vget_high_s16(a7));
        return;
    }

    int16x8_t b0 = vaddq_s16(a0, a1);
    int16x8_t b1 = vsubq_s16(a0, a1);
    int16x8_t b2 = vaddq_s16(a2, a3);
    int16x8_t b3 = vsubq_s16(a2, a3);
    int16x8_t b4 = vaddq_s16(a4, a5);
    int16x8_t b5 = vsubq_s16(a4, a5);
    int16x8_t b6 = vaddq_s16(a6, a7);
    int16x8_t b7 = vsubq_s16(a6, a7);

    a0 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b0), vreinterpretq_s32_s16(b1)));
    a1 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b0), vreinterpretq_s32_s16(b1)));
    a2 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b2), vreinterpretq_s32_s16(b3)));
    a3 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b2), vreinterpretq_s32_s16(b3)));
    a4 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b4), vreinterpretq_s32_s16(b5)));
    a5 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b4), vreinterpretq_s32_s16(b5)));
    a6 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b6), vreinterpretq_s32_s16(b7)));
    a7 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b6), vreinterpretq_s32_s16(b7)));

    b0 = vabsq_s16(vaddq_s16(a0, a1));
    b1 = vabdq_s16(a0, a1);
    b2 = vabsq_s16(vaddq_s16(a2, a3));
    b3 = vabdq_s16(a2, a3);
    b4 = vabsq_s16(vaddq_s16(a4, a5));
    b5 = vabdq_s16(a4, a5);
    b6 = vabsq_s16(vaddq_s16(a6, a7));
    b7 = vabdq_s16(a6, a7);

    a0 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(b0), vreinterpretq_s64_s16(b1)));
    a1 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(b0), vreinterpretq_s64_s16(b1)));
    a2 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(b2), vreinterpretq_s64_s16(b3)));
    a3 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(b2), vreinterpretq_s64_s16(b3)));
    a4 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(b4), vreinterpretq_s64_s16(b5)));
    a5 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(b4), vreinterpretq_s64_s16(b5)));
    a6 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(b6), vreinterpretq_s64_s16(b7)));
    a7 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(b6), vreinterpretq_s64_s16(b7)));

    const uint16x8_t max[4] = {
        vmaxq_u16(vreinterpretq_u16_s16(a0), vreinterpretq_u16_s16(a1)),
        vmaxq_u16(vreinterpretq_u16_s16(a2), vreinterpretq_u16_s16(a3)),
        vmaxq_u16(vreinterpretq_u16_s16(a4), vreinterpretq_u16_s16(a5)),
        vmaxq_u16(vreinterpretq_u16_s16(a6), vreinterpretq_u16_s16(a7)),
    };

    *satd = horizontal_add_4d_u16x8(max);
}

static inline uint32x4_t highbd_hadamard_8x8_h_pass_satd_neon(int16x8_t s[8]) {
    int16x8_t a0 = vtrn1q_s16(s[0], s[1]);
    int16x8_t a1 = vtrn2q_s16(s[0], s[1]);
    int16x8_t a2 = vtrn1q_s16(s[2], s[3]);
    int16x8_t a3 = vtrn2q_s16(s[2], s[3]);
    int16x8_t a4 = vtrn1q_s16(s[4], s[5]);
    int16x8_t a5 = vtrn2q_s16(s[4], s[5]);
    int16x8_t a6 = vtrn1q_s16(s[6], s[7]);
    int16x8_t a7 = vtrn2q_s16(s[6], s[7]);

    int16x8_t b0 = vaddq_s16(a0, a1);
    int16x8_t b1 = vsubq_s16(a0, a1);
    int16x8_t b2 = vaddq_s16(a2, a3);
    int16x8_t b3 = vsubq_s16(a2, a3);
    int16x8_t b4 = vaddq_s16(a4, a5);
    int16x8_t b5 = vsubq_s16(a4, a5);
    int16x8_t b6 = vaddq_s16(a6, a7);
    int16x8_t b7 = vsubq_s16(a6, a7);

    a0 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b0), vreinterpretq_s32_s16(b1)));
    a1 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b0), vreinterpretq_s32_s16(b1)));
    a2 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b2), vreinterpretq_s32_s16(b3)));
    a3 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b2), vreinterpretq_s32_s16(b3)));
    a4 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b4), vreinterpretq_s32_s16(b5)));
    a5 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b4), vreinterpretq_s32_s16(b5)));
    a6 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b6), vreinterpretq_s32_s16(b7)));
    a7 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b6), vreinterpretq_s32_s16(b7)));

    b0 = vabsq_s16(vaddq_s16(a0, a1));
    b1 = vabdq_s16(a0, a1);
    b2 = vabsq_s16(vaddq_s16(a2, a3));
    b3 = vabdq_s16(a2, a3);
    b4 = vabsq_s16(vaddq_s16(a4, a5));
    b5 = vabdq_s16(a4, a5);
    b6 = vabsq_s16(vaddq_s16(a6, a7));
    b7 = vabdq_s16(a6, a7);

    a0 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(b0), vreinterpretq_s64_s16(b1)));
    a1 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(b0), vreinterpretq_s64_s16(b1)));
    a2 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(b2), vreinterpretq_s64_s16(b3)));
    a3 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(b2), vreinterpretq_s64_s16(b3)));
    a4 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(b4), vreinterpretq_s64_s16(b5)));
    a5 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(b4), vreinterpretq_s64_s16(b5)));
    a6 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(b6), vreinterpretq_s64_s16(b7)));
    a7 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(b6), vreinterpretq_s64_s16(b7)));

    const uint16x8_t max[4] = {
        vmaxq_u16(vreinterpretq_u16_s16(a0), vreinterpretq_u16_s16(a1)),
        vmaxq_u16(vreinterpretq_u16_s16(a2), vreinterpretq_u16_s16(a3)),
        vmaxq_u16(vreinterpretq_u16_s16(a4), vreinterpretq_u16_s16(a5)),
        vmaxq_u16(vreinterpretq_u16_s16(a6), vreinterpretq_u16_s16(a7)),
    };

    return horizontal_add_4d_u16x8(max);
}

#endif // SVT_AV1_HADAMARD_PATH_NEON_H_
