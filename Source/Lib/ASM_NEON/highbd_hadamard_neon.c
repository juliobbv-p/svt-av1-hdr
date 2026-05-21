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
#include "mem_neon.h"
#include "sum_neon.h"
#include "transpose_neon.h"

static inline void hadamard_highbd_col8_first_pass(int16x8_t* a0, int16x8_t* a1, int16x8_t* a2, int16x8_t* a3,
                                                   int16x8_t* a4, int16x8_t* a5, int16x8_t* a6, int16x8_t* a7) {
    int16x8_t b0 = vaddq_s16(*a0, *a1);
    int16x8_t b1 = vsubq_s16(*a0, *a1);
    int16x8_t b2 = vaddq_s16(*a2, *a3);
    int16x8_t b3 = vsubq_s16(*a2, *a3);
    int16x8_t b4 = vaddq_s16(*a4, *a5);
    int16x8_t b5 = vsubq_s16(*a4, *a5);
    int16x8_t b6 = vaddq_s16(*a6, *a7);
    int16x8_t b7 = vsubq_s16(*a6, *a7);

    int16x8_t c0 = vaddq_s16(b0, b2);
    int16x8_t c2 = vsubq_s16(b0, b2);
    int16x8_t c1 = vaddq_s16(b1, b3);
    int16x8_t c3 = vsubq_s16(b1, b3);
    int16x8_t c4 = vaddq_s16(b4, b6);
    int16x8_t c6 = vsubq_s16(b4, b6);
    int16x8_t c5 = vaddq_s16(b5, b7);
    int16x8_t c7 = vsubq_s16(b5, b7);

    *a0 = vaddq_s16(c0, c4);
    *a2 = vsubq_s16(c0, c4);
    *a7 = vaddq_s16(c1, c5);
    *a6 = vsubq_s16(c1, c5);
    *a3 = vaddq_s16(c2, c6);
    *a1 = vsubq_s16(c2, c6);
    *a4 = vaddq_s16(c3, c7);
    *a5 = vsubq_s16(c3, c7);
}

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

    int16x8_t s0 = vld1q_s16(src_diff + 0 * src_stride);
    int16x8_t s1 = vld1q_s16(src_diff + 1 * src_stride);
    int16x8_t s2 = vld1q_s16(src_diff + 2 * src_stride);
    int16x8_t s3 = vld1q_s16(src_diff + 3 * src_stride);
    int16x8_t s4 = vld1q_s16(src_diff + 4 * src_stride);
    int16x8_t s5 = vld1q_s16(src_diff + 5 * src_stride);
    int16x8_t s6 = vld1q_s16(src_diff + 6 * src_stride);
    int16x8_t s7 = vld1q_s16(src_diff + 7 * src_stride);

    // For the first pass we can stay in 16-bit elements (4095*8 = 32760).
    hadamard_highbd_col8_first_pass(&s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);

    transpose_elems_inplace_s16_8x8(&s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);

    // For the second pass we need to widen to 32-bit elements, so we're
    // processing 4 columns at a time.
    // Skip the second transpose because it is not required.

    b0 = vget_low_s16(s0);
    b1 = vget_low_s16(s1);
    b2 = vget_low_s16(s2);
    b3 = vget_low_s16(s3);
    b4 = vget_low_s16(s4);
    b5 = vget_low_s16(s5);
    b6 = vget_low_s16(s6);
    b7 = vget_low_s16(s7);

    hadamard_highbd_col4_second_pass(b0, b1, b2, b3, b4, b5, b6, b7, coeff);

    b0 = vget_high_s16(s0);
    b1 = vget_high_s16(s1);
    b2 = vget_high_s16(s2);
    b3 = vget_high_s16(s3);
    b4 = vget_high_s16(s4);
    b5 = vget_high_s16(s5);
    b6 = vget_high_s16(s6);
    b7 = vget_high_s16(s7);

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

static inline void hadamard_highbd_8x8(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                       ptrdiff_t pred_stride, int32x4_t coeff[16]) {
    uint16x8_t s0, s1, s2, s3, s4, s5, s6, s7;
    uint16x8_t p0, p1, p2, p3, p4, p5, p6, p7;
    load_u16_8x8(src, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);
    load_u16_8x8(pred, pred_stride, &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7);

    int16x8_t d0 = vreinterpretq_s16_u16(vsubq_u16(s0, p0));
    int16x8_t d1 = vreinterpretq_s16_u16(vsubq_u16(s1, p1));
    int16x8_t d2 = vreinterpretq_s16_u16(vsubq_u16(s2, p2));
    int16x8_t d3 = vreinterpretq_s16_u16(vsubq_u16(s3, p3));
    int16x8_t d4 = vreinterpretq_s16_u16(vsubq_u16(s4, p4));
    int16x8_t d5 = vreinterpretq_s16_u16(vsubq_u16(s5, p5));
    int16x8_t d6 = vreinterpretq_s16_u16(vsubq_u16(s6, p6));
    int16x8_t d7 = vreinterpretq_s16_u16(vsubq_u16(s7, p7));
    hadamard_highbd_col8_first_pass(&d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7);

    const int16x8_t a0 = vtrn1q_s16(d0, d1);
    const int16x8_t a1 = vtrn2q_s16(d0, d1);
    const int16x8_t a2 = vtrn1q_s16(d2, d3);
    const int16x8_t a3 = vtrn2q_s16(d2, d3);
    const int16x8_t a4 = vtrn1q_s16(d4, d5);
    const int16x8_t a5 = vtrn2q_s16(d4, d5);
    const int16x8_t a6 = vtrn1q_s16(d6, d7);
    const int16x8_t a7 = vtrn2q_s16(d6, d7);

    int32x4_t b0  = vaddl_s16(vget_low_s16(a0), vget_low_s16(a1));
    int32x4_t b1  = vsubl_s16(vget_low_s16(a0), vget_low_s16(a1));
    int32x4_t b2  = vaddl_s16(vget_high_s16(a0), vget_high_s16(a1));
    int32x4_t b3  = vsubl_s16(vget_high_s16(a0), vget_high_s16(a1));
    int32x4_t b4  = vaddl_s16(vget_low_s16(a2), vget_low_s16(a3));
    int32x4_t b5  = vsubl_s16(vget_low_s16(a2), vget_low_s16(a3));
    int32x4_t b6  = vaddl_s16(vget_high_s16(a2), vget_high_s16(a3));
    int32x4_t b7  = vsubl_s16(vget_high_s16(a2), vget_high_s16(a3));
    int32x4_t b8  = vaddl_s16(vget_low_s16(a4), vget_low_s16(a5));
    int32x4_t b9  = vsubl_s16(vget_low_s16(a4), vget_low_s16(a5));
    int32x4_t b10 = vaddl_s16(vget_high_s16(a4), vget_high_s16(a5));
    int32x4_t b11 = vsubl_s16(vget_high_s16(a4), vget_high_s16(a5));
    int32x4_t b12 = vaddl_s16(vget_low_s16(a6), vget_low_s16(a7));
    int32x4_t b13 = vsubl_s16(vget_low_s16(a6), vget_low_s16(a7));
    int32x4_t b14 = vaddl_s16(vget_high_s16(a6), vget_high_s16(a7));
    int32x4_t b15 = vsubl_s16(vget_high_s16(a6), vget_high_s16(a7));

    const int32x4_t c0  = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(b0), vreinterpretq_s64_s32(b1)));
    const int32x4_t c1  = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(b0), vreinterpretq_s64_s32(b1)));
    const int32x4_t c2  = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(b2), vreinterpretq_s64_s32(b3)));
    const int32x4_t c3  = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(b2), vreinterpretq_s64_s32(b3)));
    const int32x4_t c4  = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(b4), vreinterpretq_s64_s32(b5)));
    const int32x4_t c5  = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(b4), vreinterpretq_s64_s32(b5)));
    const int32x4_t c6  = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(b6), vreinterpretq_s64_s32(b7)));
    const int32x4_t c7  = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(b6), vreinterpretq_s64_s32(b7)));
    const int32x4_t c8  = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(b8), vreinterpretq_s64_s32(b9)));
    const int32x4_t c9  = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(b8), vreinterpretq_s64_s32(b9)));
    const int32x4_t c10 = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(b10), vreinterpretq_s64_s32(b11)));
    const int32x4_t c11 = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(b10), vreinterpretq_s64_s32(b11)));
    const int32x4_t c12 = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(b12), vreinterpretq_s64_s32(b13)));
    const int32x4_t c13 = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(b12), vreinterpretq_s64_s32(b13)));
    const int32x4_t c14 = vreinterpretq_s32_s64(vtrn1q_s64(vreinterpretq_s64_s32(b14), vreinterpretq_s64_s32(b15)));
    const int32x4_t c15 = vreinterpretq_s32_s64(vtrn2q_s64(vreinterpretq_s64_s32(b14), vreinterpretq_s64_s32(b15)));

    b0  = vaddq_s32(c0, c1);
    b1  = vsubq_s32(c0, c1);
    b2  = vaddq_s32(c2, c3);
    b3  = vsubq_s32(c2, c3);
    b4  = vaddq_s32(c4, c5);
    b5  = vsubq_s32(c4, c5);
    b6  = vaddq_s32(c6, c7);
    b7  = vsubq_s32(c6, c7);
    b8  = vaddq_s32(c8, c9);
    b9  = vsubq_s32(c8, c9);
    b10 = vaddq_s32(c10, c11);
    b11 = vsubq_s32(c10, c11);
    b12 = vaddq_s32(c12, c13);
    b13 = vsubq_s32(c12, c13);
    b14 = vaddq_s32(c14, c15);
    b15 = vsubq_s32(c14, c15);

    coeff[0]  = vaddq_s32(b0, b2);
    coeff[1]  = vsubq_s32(b0, b2);
    coeff[2]  = vaddq_s32(b1, b3);
    coeff[3]  = vsubq_s32(b1, b3);
    coeff[4]  = vaddq_s32(b4, b6);
    coeff[5]  = vsubq_s32(b4, b6);
    coeff[6]  = vaddq_s32(b5, b7);
    coeff[7]  = vsubq_s32(b5, b7);
    coeff[8]  = vaddq_s32(b8, b10);
    coeff[9]  = vsubq_s32(b8, b10);
    coeff[10] = vaddq_s32(b9, b11);
    coeff[11] = vsubq_s32(b9, b11);
    coeff[12] = vaddq_s32(b12, b14);
    coeff[13] = vsubq_s32(b12, b14);
    coeff[14] = vaddq_s32(b13, b15);
    coeff[15] = vsubq_s32(b13, b15);
}

int svt_av1_highbd_hadamard_satd_8x8_neon(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                          ptrdiff_t pred_stride) {
    uint16x8_t s0, s1, s2, s3, s4, s5, s6, s7;
    uint16x8_t p0, p1, p2, p3, p4, p5, p6, p7;
    load_u16_8x8(src, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);
    load_u16_8x8(pred, pred_stride, &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7);

    int16x8_t a0 = vreinterpretq_s16_u16(vsubq_u16(s0, p0));
    int16x8_t a1 = vreinterpretq_s16_u16(vsubq_u16(s1, p1));
    int16x8_t a2 = vreinterpretq_s16_u16(vsubq_u16(s2, p2));
    int16x8_t a3 = vreinterpretq_s16_u16(vsubq_u16(s3, p3));
    int16x8_t a4 = vreinterpretq_s16_u16(vsubq_u16(s4, p4));
    int16x8_t a5 = vreinterpretq_s16_u16(vsubq_u16(s5, p5));
    int16x8_t a6 = vreinterpretq_s16_u16(vsubq_u16(s6, p6));
    int16x8_t a7 = vreinterpretq_s16_u16(vsubq_u16(s7, p7));

    hadamard_highbd_col8_first_pass(&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);

    int16x8_t b0 = vtrn1q_s16(a0, a1);
    int16x8_t b1 = vtrn2q_s16(a0, a1);
    int16x8_t b2 = vtrn1q_s16(a2, a3);
    int16x8_t b3 = vtrn2q_s16(a2, a3);
    int16x8_t b4 = vtrn1q_s16(a4, a5);
    int16x8_t b5 = vtrn2q_s16(a4, a5);
    int16x8_t b6 = vtrn1q_s16(a6, a7);
    int16x8_t b7 = vtrn2q_s16(a6, a7);

    a0 = vaddq_s16(b0, b1);
    a1 = vsubq_s16(b0, b1);
    a2 = vaddq_s16(b2, b3);
    a3 = vsubq_s16(b2, b3);
    a4 = vaddq_s16(b4, b5);
    a5 = vsubq_s16(b4, b5);
    a6 = vaddq_s16(b6, b7);
    a7 = vsubq_s16(b6, b7);

    b0 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a1)));
    b1 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a1)));
    b2 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a2), vreinterpretq_s32_s16(a3)));
    b3 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a2), vreinterpretq_s32_s16(a3)));
    b4 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a4), vreinterpretq_s32_s16(a5)));
    b5 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a4), vreinterpretq_s32_s16(a5)));
    b6 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a6), vreinterpretq_s32_s16(a7)));
    b7 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a6), vreinterpretq_s32_s16(a7)));

    a0 = vabsq_s16(vaddq_s16(b0, b1));
    a1 = vabdq_s16(b0, b1);
    a2 = vabsq_s16(vaddq_s16(b2, b3));
    a3 = vabdq_s16(b2, b3);
    a4 = vabsq_s16(vaddq_s16(b4, b5));
    a5 = vabdq_s16(b4, b5);
    a6 = vabsq_s16(vaddq_s16(b6, b7));
    a7 = vabdq_s16(b6, b7);

    b0 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a0), vreinterpretq_s64_s16(a1)));
    b1 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a0), vreinterpretq_s64_s16(a1)));
    b2 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a2), vreinterpretq_s64_s16(a3)));
    b3 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a2), vreinterpretq_s64_s16(a3)));
    b4 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a4), vreinterpretq_s64_s16(a5)));
    b5 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a4), vreinterpretq_s64_s16(a5)));
    b6 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a6), vreinterpretq_s64_s16(a7)));
    b7 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a6), vreinterpretq_s64_s16(a7)));

    int16x8_t max[4];
    max[0] = vmaxq_s16(b0, b1);
    max[1] = vmaxq_s16(b2, b3);
    max[2] = vmaxq_s16(b4, b5);
    max[3] = vmaxq_s16(b6, b7);
    return vaddvq_s32(horizontal_add_4d_s16x8(max)) << 1;
}

int svt_av1_highbd_hadamard_satd_16x16_neon(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                            ptrdiff_t pred_stride) {
    int32x4_t q0[16], q1[16], q2[16], q3[16];
    hadamard_highbd_8x8(src + 0 + 0 * src_stride, src_stride, pred + 0 + 0 * pred_stride, pred_stride, q0);
    hadamard_highbd_8x8(src + 8 + 0 * src_stride, src_stride, pred + 8 + 0 * pred_stride, pred_stride, q1);
    hadamard_highbd_8x8(src + 0 + 8 * src_stride, src_stride, pred + 0 + 8 * pred_stride, pred_stride, q2);
    hadamard_highbd_8x8(src + 8 + 8 * src_stride, src_stride, pred + 8 + 8 * pred_stride, pred_stride, q3);

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

static inline void hadamard_highbd_16x16_neon(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                              ptrdiff_t pred_stride, int32x4_t coeff[64]) {
    int32x4_t q0[16], q1[16], q2[16], q3[16];
    hadamard_highbd_8x8(src + 0 + 0 * src_stride, src_stride, pred + 0 + 0 * pred_stride, pred_stride, q0);
    hadamard_highbd_8x8(src + 8 + 0 * src_stride, src_stride, pred + 8 + 0 * pred_stride, pred_stride, q1);
    hadamard_highbd_8x8(src + 0 + 8 * src_stride, src_stride, pred + 0 + 8 * pred_stride, pred_stride, q2);
    hadamard_highbd_8x8(src + 8 + 8 * src_stride, src_stride, pred + 8 + 8 * pred_stride, pred_stride, q3);

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
