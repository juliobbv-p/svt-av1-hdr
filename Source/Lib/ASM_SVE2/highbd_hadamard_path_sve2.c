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

#include "common_dsp_rtcd.h"
#include "hadamard_path_sve2.h"

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
static void highbd_hadamard_8x8_coeff_sve2(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                           ptrdiff_t pred_stride, int32x4_t coeff[16]) {
    EB_ASSUME(pred != NULL);
    highbd_hadamard_8x8_sve2(src, src_stride, pred, pred_stride, coeff, NULL, NULL);
}

int svt_av1_highbd_hadamard_satd_8x8_sve2(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                          ptrdiff_t pred_stride) {
    EB_ASSUME(pred != NULL);
    int32x4_t satd;
    highbd_hadamard_8x8_sve2(src, src_stride, pred, pred_stride, NULL, &satd, NULL);
    return vaddvq_s32(satd) << 1;
}

int svt_av1_highbd_hadamard_satd_16x16_sve2(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                            ptrdiff_t pred_stride) {
    int32x4_t q0[16], q1[16], q2[16], q3[16];

    highbd_hadamard_8x8_coeff_sve2(src + 0 + 0 * src_stride, src_stride, pred + 0 + 0 * pred_stride, pred_stride, q0);
    highbd_hadamard_8x8_coeff_sve2(src + 8 + 0 * src_stride, src_stride, pred + 8 + 0 * pred_stride, pred_stride, q1);
    highbd_hadamard_8x8_coeff_sve2(src + 0 + 8 * src_stride, src_stride, pred + 0 + 8 * pred_stride, pred_stride, q2);
    highbd_hadamard_8x8_coeff_sve2(src + 8 + 8 * src_stride, src_stride, pred + 8 + 8 * pred_stride, pred_stride, q3);

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

static void highbd_hadamard_16x16_sve2(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                       ptrdiff_t pred_stride, int32x4_t coeff[64]) {
    int32x4_t q0[16], q1[16], q2[16], q3[16];

    highbd_hadamard_8x8_coeff_sve2(src + 0 + 0 * src_stride, src_stride, pred + 0 + 0 * pred_stride, pred_stride, q0);
    highbd_hadamard_8x8_coeff_sve2(src + 8 + 0 * src_stride, src_stride, pred + 8 + 0 * pred_stride, pred_stride, q1);
    highbd_hadamard_8x8_coeff_sve2(src + 0 + 8 * src_stride, src_stride, pred + 0 + 8 * pred_stride, pred_stride, q2);
    highbd_hadamard_8x8_coeff_sve2(src + 8 + 8 * src_stride, src_stride, pred + 8 + 8 * pred_stride, pred_stride, q3);

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

int svt_av1_highbd_hadamard_satd_32x32_sve2(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                            ptrdiff_t pred_stride) {
    int32x4_t q0[64], q1[64], q2[64], q3[64];

    highbd_hadamard_16x16_sve2(src + 0 + 0 * src_stride, src_stride, pred + 0 + 0 * pred_stride, pred_stride, q0);
    highbd_hadamard_16x16_sve2(src + 16 + 0 * src_stride, src_stride, pred + 16 + 0 * pred_stride, pred_stride, q1);
    highbd_hadamard_16x16_sve2(src + 0 + 16 * src_stride, src_stride, pred + 0 + 16 * pred_stride, pred_stride, q2);
    highbd_hadamard_16x16_sve2(src + 16 + 16 * src_stride, src_stride, pred + 16 + 16 * pred_stride, pred_stride, q3);

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
#endif
