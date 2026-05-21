/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>

#include "aom_dsp_rtcd.h"
#include "coding_loop.h"
#include "definitions.h"
#include "mem_neon.h"
#include "sum_neon.h"
#include "transpose_neon.h"

static inline void hadamard_4x4_one_pass(int16x4_t* a0, int16x4_t* a1, int16x4_t* a2, int16x4_t* a3) {
    const int16x4_t b0 = vhadd_s16(*a0, *a1);
    const int16x4_t b1 = vhsub_s16(*a0, *a1);
    const int16x4_t b2 = vhadd_s16(*a2, *a3);
    const int16x4_t b3 = vhsub_s16(*a2, *a3);

    *a0 = vadd_s16(b0, b2);
    *a1 = vadd_s16(b1, b3);
    *a2 = vsub_s16(b0, b2);
    *a3 = vsub_s16(b1, b3);
}

void svt_aom_hadamard_4x4_neon(const int16_t* src_diff, ptrdiff_t src_stride, tran_low_t* coeff) {
    int16x4_t a0 = vld1_s16(src_diff);
    int16x4_t a1 = vld1_s16(src_diff + src_stride);
    int16x4_t a2 = vld1_s16(src_diff + 2 * src_stride);
    int16x4_t a3 = vld1_s16(src_diff + 3 * src_stride);

    hadamard_4x4_one_pass(&a0, &a1, &a2, &a3);

    transpose_elems_inplace_s16_4x4(&a0, &a1, &a2, &a3);

    hadamard_4x4_one_pass(&a0, &a1, &a2, &a3);

    store_s16_to_tran_low(coeff, a0);
    store_s16_to_tran_low(coeff + 4, a1);
    store_s16_to_tran_low(coeff + 8, a2);
    store_s16_to_tran_low(coeff + 12, a3);
}

static inline void hadamard_8x8_one_pass(int16x8_t* a) {
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

void svt_aom_hadamard_8x8_neon(const int16_t* src_diff, ptrdiff_t src_stride, int32_t* coeff) {
    int16x8_t a[8];

    a[0] = vld1q_s16(src_diff);
    a[1] = vld1q_s16(src_diff + src_stride);
    a[2] = vld1q_s16(src_diff + 2 * src_stride);
    a[3] = vld1q_s16(src_diff + 3 * src_stride);
    a[4] = vld1q_s16(src_diff + 4 * src_stride);
    a[5] = vld1q_s16(src_diff + 5 * src_stride);
    a[6] = vld1q_s16(src_diff + 6 * src_stride);
    a[7] = vld1q_s16(src_diff + 7 * src_stride);

    hadamard_8x8_one_pass(a);
    transpose_elems_inplace_s16_8x8(a + 0, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6, a + 7);
    hadamard_8x8_one_pass(a);

    store_s16q_to_tran_low(coeff + 0, a[0]);
    store_s16q_to_tran_low(coeff + 8, a[1]);
    store_s16q_to_tran_low(coeff + 16, a[2]);
    store_s16q_to_tran_low(coeff + 24, a[3]);
    store_s16q_to_tran_low(coeff + 32, a[4]);
    store_s16q_to_tran_low(coeff + 40, a[5]);
    store_s16q_to_tran_low(coeff + 48, a[6]);
    store_s16q_to_tran_low(coeff + 56, a[7]);
}

void svt_aom_hadamard_16x16_neon(const int16_t* src_diff, ptrdiff_t src_stride, tran_low_t* coeff) {
    /* Rearrange 16x16 to 8x32 and remove stride.
     * Top left first. */
    svt_aom_hadamard_8x8_neon(src_diff + 0 + 0 * src_stride, src_stride, coeff + 0);
    /* Top right. */
    svt_aom_hadamard_8x8_neon(src_diff + 8 + 0 * src_stride, src_stride, coeff + 64);
    /* Bottom left. */
    svt_aom_hadamard_8x8_neon(src_diff + 0 + 8 * src_stride, src_stride, coeff + 128);
    /* Bottom right. */
    svt_aom_hadamard_8x8_neon(src_diff + 8 + 8 * src_stride, src_stride, coeff + 192);

    // Each iteration of the loop operates on entire rows (16 samples each)
    // because we need to swap the second and third quarters of every row in the
    // output to match AVX2 output (i.e., aom_hadamard_16x16_avx2). See the for
    // loop at the end of aom_hadamard_16x16_c.
    for (int i = 0; i < 64; i += 16) {
        const int32x4_t a00 = vld1q_s32(coeff + 0);
        const int32x4_t a01 = vld1q_s32(coeff + 64);
        const int32x4_t a02 = vld1q_s32(coeff + 128);
        const int32x4_t a03 = vld1q_s32(coeff + 192);

        const int32x4_t b00 = vhaddq_s32(a00, a01);
        const int32x4_t b01 = vhsubq_s32(a00, a01);
        const int32x4_t b02 = vhaddq_s32(a02, a03);
        const int32x4_t b03 = vhsubq_s32(a02, a03);

        const int32x4_t c00 = vaddq_s32(b00, b02);
        const int32x4_t c01 = vaddq_s32(b01, b03);
        const int32x4_t c02 = vsubq_s32(b00, b02);
        const int32x4_t c03 = vsubq_s32(b01, b03);

        const int32x4_t a10 = vld1q_s32(coeff + 4 + 0);
        const int32x4_t a11 = vld1q_s32(coeff + 4 + 64);
        const int32x4_t a12 = vld1q_s32(coeff + 4 + 128);
        const int32x4_t a13 = vld1q_s32(coeff + 4 + 192);

        const int32x4_t b10 = vhaddq_s32(a10, a11);
        const int32x4_t b11 = vhsubq_s32(a10, a11);
        const int32x4_t b12 = vhaddq_s32(a12, a13);
        const int32x4_t b13 = vhsubq_s32(a12, a13);

        const int32x4_t c10 = vaddq_s32(b10, b12);
        const int32x4_t c11 = vaddq_s32(b11, b13);
        const int32x4_t c12 = vsubq_s32(b10, b12);
        const int32x4_t c13 = vsubq_s32(b11, b13);

        const int32x4_t a20 = vld1q_s32(coeff + 8 + 0);
        const int32x4_t a21 = vld1q_s32(coeff + 8 + 64);
        const int32x4_t a22 = vld1q_s32(coeff + 8 + 128);
        const int32x4_t a23 = vld1q_s32(coeff + 8 + 192);

        const int32x4_t b20 = vhaddq_s32(a20, a21);
        const int32x4_t b21 = vhsubq_s32(a20, a21);
        const int32x4_t b22 = vhaddq_s32(a22, a23);
        const int32x4_t b23 = vhsubq_s32(a22, a23);

        const int32x4_t c20 = vaddq_s32(b20, b22);
        const int32x4_t c21 = vaddq_s32(b21, b23);
        const int32x4_t c22 = vsubq_s32(b20, b22);
        const int32x4_t c23 = vsubq_s32(b21, b23);

        const int32x4_t a30 = vld1q_s32(coeff + 12 + 0);
        const int32x4_t a31 = vld1q_s32(coeff + 12 + 64);
        const int32x4_t a32 = vld1q_s32(coeff + 12 + 128);
        const int32x4_t a33 = vld1q_s32(coeff + 12 + 192);

        const int32x4_t b30 = vhaddq_s32(a30, a31);
        const int32x4_t b31 = vhsubq_s32(a30, a31);
        const int32x4_t b32 = vhaddq_s32(a32, a33);
        const int32x4_t b33 = vhsubq_s32(a32, a33);

        const int32x4_t c30 = vaddq_s32(b30, b32);
        const int32x4_t c31 = vaddq_s32(b31, b33);
        const int32x4_t c32 = vsubq_s32(b30, b32);
        const int32x4_t c33 = vsubq_s32(b31, b33);

        vst1q_s32(coeff + 0 + 0, c00);
        vst1q_s32(coeff + 0 + 4, c20);
        vst1q_s32(coeff + 0 + 8, c10);
        vst1q_s32(coeff + 0 + 12, c30);

        vst1q_s32(coeff + 64 + 0, c01);
        vst1q_s32(coeff + 64 + 4, c21);
        vst1q_s32(coeff + 64 + 8, c11);
        vst1q_s32(coeff + 64 + 12, c31);

        vst1q_s32(coeff + 128 + 0, c02);
        vst1q_s32(coeff + 128 + 4, c22);
        vst1q_s32(coeff + 128 + 8, c12);
        vst1q_s32(coeff + 128 + 12, c32);

        vst1q_s32(coeff + 192 + 0, c03);
        vst1q_s32(coeff + 192 + 4, c23);
        vst1q_s32(coeff + 192 + 8, c13);
        vst1q_s32(coeff + 192 + 12, c33);

        coeff += 16;
    }
}

void svt_aom_hadamard_32x32_neon(const int16_t* src_diff, ptrdiff_t src_stride, tran_low_t* coeff) {
    /* Top left first. */
    svt_aom_hadamard_16x16_neon(src_diff + 0 + 0 * src_stride, src_stride, coeff + 0);
    /* Top right. */
    svt_aom_hadamard_16x16_neon(src_diff + 16 + 0 * src_stride, src_stride, coeff + 256);
    /* Bottom left. */
    svt_aom_hadamard_16x16_neon(src_diff + 0 + 16 * src_stride, src_stride, coeff + 512);
    /* Bottom right. */
    svt_aom_hadamard_16x16_neon(src_diff + 16 + 16 * src_stride, src_stride, coeff + 768);

    for (int i = 0; i < 256; i += 4) {
        const int32x4_t a0 = vld1q_s32(coeff);
        const int32x4_t a1 = vld1q_s32(coeff + 256);
        const int32x4_t a2 = vld1q_s32(coeff + 512);
        const int32x4_t a3 = vld1q_s32(coeff + 768);

        const int32x4_t b0 = vshrq_n_s32(vaddq_s32(a0, a1), 2);
        const int32x4_t b1 = vshrq_n_s32(vsubq_s32(a0, a1), 2);
        const int32x4_t b2 = vshrq_n_s32(vaddq_s32(a2, a3), 2);
        const int32x4_t b3 = vshrq_n_s32(vsubq_s32(a2, a3), 2);

        const int32x4_t c0 = vaddq_s32(b0, b2);
        const int32x4_t c1 = vaddq_s32(b1, b3);
        const int32x4_t c2 = vsubq_s32(b0, b2);
        const int32x4_t c3 = vsubq_s32(b1, b3);

        vst1q_s32(coeff + 0, c0);
        vst1q_s32(coeff + 256, c1);
        vst1q_s32(coeff + 512, c2);
        vst1q_s32(coeff + 768, c3);

        coeff += 4;
    }
}

int svt_av1_hadamard_satd_4x4_neon(const uint8_t* src, ptrdiff_t src_stride, const uint8_t* pred,
                                   ptrdiff_t pred_stride) {
    // Calculate residuals.
    uint8x8_t s02 = load_u8x4_strided_x2(src + 0 * src_stride, 2 * src_stride);
    uint8x8_t s13 = load_u8x4_strided_x2(src + 1 * src_stride, 2 * src_stride);

    uint8x8_t p02 = load_u8x4_strided_x2(pred + 0 * pred_stride, 2 * pred_stride);
    uint8x8_t p13 = load_u8x4_strided_x2((uint8_t*)pred + 1 * pred_stride, 2 * pred_stride);

    int16x8_t d02 = vreinterpretq_s16_u16(vsubl_u8(s02, p02));
    int16x8_t d13 = vreinterpretq_s16_u16(vsubl_u8(s13, p13));

    // Hadamard transform.
    int16x8_t a0 = vhaddq_s16(d02, d13);
    int16x8_t a1 = vhsubq_s16(d02, d13);

    int16x8_t b0 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(a0), vreinterpretq_s64_s16(a1)));
    int16x8_t b1 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(a0), vreinterpretq_s64_s16(a1)));

    a0 = vaddq_s16(b0, b1);
    a1 = vsubq_s16(b0, b1);

    b0 = vtrn1q_s16(a0, a1);
    b1 = vtrn2q_s16(a0, a1);

    // Fused last Hadamard step and SATD accumulation.
    a0 = vabsq_s16(vhaddq_s16(b0, b1));
    a1 = vabsq_s16(vhsubq_s16(b0, b1));

    b0 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a1)));
    b1 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(a0), vreinterpretq_s32_s16(a1)));

    return vaddlvq_s16(vmaxq_s16(b0, b1)) << 1;
}

int svt_av1_hadamard_satd_8x8_neon(const uint8_t* src, ptrdiff_t src_stride, const uint8_t* pred,
                                   ptrdiff_t pred_stride) {
    uint8x8_t s0, s1, s2, s3, s4, s5, s6, s7;
    uint8x8_t p0, p1, p2, p3, p4, p5, p6, p7;

    load_u8_8x8(src, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);
    load_u8_8x8(pred, pred_stride, &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7);

    int16x8_t a[8];
    a[0] = vreinterpretq_s16_u16(vsubl_u8(s0, p0));
    a[1] = vreinterpretq_s16_u16(vsubl_u8(s1, p1));
    a[2] = vreinterpretq_s16_u16(vsubl_u8(s2, p2));
    a[3] = vreinterpretq_s16_u16(vsubl_u8(s3, p3));
    a[4] = vreinterpretq_s16_u16(vsubl_u8(s4, p4));
    a[5] = vreinterpretq_s16_u16(vsubl_u8(s5, p5));
    a[6] = vreinterpretq_s16_u16(vsubl_u8(s6, p6));
    a[7] = vreinterpretq_s16_u16(vsubl_u8(s7, p7));

    hadamard_8x8_one_pass(a);

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

    int32x4_t sum = horizontal_add_4d_s16x8(max);
    return vaddvq_s32(sum) << 1;
}

static inline void hadamard_8x8_neon(const uint8_t* src, ptrdiff_t src_stride, const uint8_t* pred,
                                     ptrdiff_t pred_stride, int16x8_t coeff[8]) {
    uint8x8_t s0, s1, s2, s3, s4, s5, s6, s7;
    uint8x8_t p0, p1, p2, p3, p4, p5, p6, p7;

    load_u8_8x8(src, src_stride, &s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);
    load_u8_8x8(pred, pred_stride, &p0, &p1, &p2, &p3, &p4, &p5, &p6, &p7);

    int16x8_t a[8];
    a[0] = vreinterpretq_s16_u16(vsubl_u8(s0, p0));
    a[1] = vreinterpretq_s16_u16(vsubl_u8(s1, p1));
    a[2] = vreinterpretq_s16_u16(vsubl_u8(s2, p2));
    a[3] = vreinterpretq_s16_u16(vsubl_u8(s3, p3));
    a[4] = vreinterpretq_s16_u16(vsubl_u8(s4, p4));
    a[5] = vreinterpretq_s16_u16(vsubl_u8(s5, p5));
    a[6] = vreinterpretq_s16_u16(vsubl_u8(s6, p6));
    a[7] = vreinterpretq_s16_u16(vsubl_u8(s7, p7));

    hadamard_8x8_one_pass(a);

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
}

int svt_av1_hadamard_satd_16x16_neon(const uint8_t* src, ptrdiff_t src_stride, const uint8_t* pred,
                                     ptrdiff_t pred_stride) {
    // Divide 16x16 block into 8x8 quadrants.
    int16x8_t q0[8], q1[8], q2[8], q3[8];

    hadamard_8x8_neon(src + 0 + 0 * src_stride, src_stride, pred + 0 + 0 * pred_stride, pred_stride, q0);
    hadamard_8x8_neon(src + 8 + 0 * src_stride, src_stride, pred + 8 + 0 * pred_stride, pred_stride, q1);
    hadamard_8x8_neon(src + 0 + 8 * src_stride, src_stride, pred + 0 + 8 * pred_stride, pred_stride, q2);
    hadamard_8x8_neon(src + 8 + 8 * src_stride, src_stride, pred + 8 + 8 * pred_stride, pred_stride, q3);

    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    for (int i = 0; i < 8; ++i) {
        const int16x8_t a0 = vabsq_s16(vhaddq_s16(q0[i], q1[i]));
        const int16x8_t a1 = vabsq_s16(vhsubq_s16(q0[i], q1[i]));
        const int16x8_t a2 = vabsq_s16(vhaddq_s16(q2[i], q3[i]));
        const int16x8_t a3 = vabsq_s16(vhsubq_s16(q2[i], q3[i]));

        acc0 = vpadalq_s16(acc0, vmaxq_s16(a0, a2));
        acc1 = vpadalq_s16(acc1, vmaxq_s16(a1, a3));
    }

    return vaddvq_s32(vaddq_s32(acc0, acc1)) << 1;
}

static inline void hadamard_16x16_neon(const uint8_t* src, ptrdiff_t src_stride, const uint8_t* pred,
                                       ptrdiff_t pred_stride, int16x8_t coeff[32]) {
    // Divide 16x16 block into 8x8 quadrants.
    int16x8_t q0[8], q1[8], q2[8], q3[8];

    hadamard_8x8_neon(src + 0 + 0 * src_stride, src_stride, pred + 0 + 0 * pred_stride, pred_stride, q0);
    hadamard_8x8_neon(src + 8 + 0 * src_stride, src_stride, pred + 8 + 0 * pred_stride, pred_stride, q1);
    hadamard_8x8_neon(src + 0 + 8 * src_stride, src_stride, pred + 0 + 8 * pred_stride, pred_stride, q2);
    hadamard_8x8_neon(src + 8 + 8 * src_stride, src_stride, pred + 8 + 8 * pred_stride, pred_stride, q3);

    for (int i = 0; i < 8; ++i) {
        const int16x8_t b0 = vhaddq_s16(q0[i], q1[i]);
        const int16x8_t b1 = vhsubq_s16(q0[i], q1[i]);
        const int16x8_t b2 = vhaddq_s16(q2[i], q3[i]);
        const int16x8_t b3 = vhsubq_s16(q2[i], q3[i]);

        coeff[4 * i + 0] = vaddq_s16(b0, b2);
        coeff[4 * i + 1] = vaddq_s16(b1, b3);
        coeff[4 * i + 2] = vsubq_s16(b0, b2);
        coeff[4 * i + 3] = vsubq_s16(b1, b3);
    }
}

int svt_av1_hadamard_satd_32x32_neon(const uint8_t* src, ptrdiff_t src_stride, const uint8_t* pred,
                                     ptrdiff_t pred_stride) {
    // Divide 32x32 block into 16x16 quadrants.
    int16x8_t q0[32], q1[32], q2[32], q3[32];

    hadamard_16x16_neon(src + 0 + 0 * src_stride, src_stride, pred + 0 + 0 * pred_stride, pred_stride, q0);
    hadamard_16x16_neon(src + 16 + 0 * src_stride, src_stride, pred + 16 + 0 * pred_stride, pred_stride, q1);
    hadamard_16x16_neon(src + 0 + 16 * src_stride, src_stride, pred + 0 + 16 * pred_stride, pred_stride, q2);
    hadamard_16x16_neon(src + 16 + 16 * src_stride, src_stride, pred + 16 + 16 * pred_stride, pred_stride, q3);

    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    for (int i = 0; i < 32; ++i) {
        const int16x8_t a0 = vhaddq_s16(q0[i], q1[i]);
        const int16x8_t a1 = vhsubq_s16(q0[i], q1[i]);
        const int16x8_t a2 = vhaddq_s16(q2[i], q3[i]);
        const int16x8_t a3 = vhsubq_s16(q2[i], q3[i]);

        const int16x8_t b0 = vshrq_n_s16(a0, 1);
        const int16x8_t b1 = vshrq_n_s16(a1, 1);
        const int16x8_t b2 = vshrq_n_s16(a2, 1);
        const int16x8_t b3 = vshrq_n_s16(a3, 1);

        acc0 = vpadalq_s16(acc0, vmaxq_s16(vabsq_s16(b0), vabsq_s16(b2)));
        acc1 = vpadalq_s16(acc1, vmaxq_s16(vabsq_s16(b1), vabsq_s16(b3)));
    }

    return vaddvq_s32(vaddq_s32(acc0, acc1)) << 1;
}
