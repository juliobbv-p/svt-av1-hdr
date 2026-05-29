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

#ifndef SVT_AV1_HADAMARD_PATH_SVE2_H_
#define SVT_AV1_HADAMARD_PATH_SVE2_H_

#include <arm_neon.h>

#include "hadamard_path_neon.h"
#include "mem_neon.h"
#include "neon_sve_bridge.h"
#include "neon_sve2_bridge.h"
#include "sum_neon.h"

static const uint16_t kHadamard8x8CADDTbl[8] = {0, 2, 4, 6, 1, 3, 5, 7};

static inline void hadamard_8x8_cadd_pass_sve2(int16x8_t a[8]) {
    const uint16x8_t idx = vld1q_u16(kHadamard8x8CADDTbl);

    a[0] = svt_tbl_s16(svt_caddq_s16_90(a[0], a[0]), idx);
    a[1] = svt_tbl_s16(svt_caddq_s16_90(a[1], a[1]), idx);
    a[2] = svt_tbl_s16(svt_caddq_s16_90(a[2], a[2]), idx);
    a[3] = svt_tbl_s16(svt_caddq_s16_90(a[3], a[3]), idx);
    a[4] = svt_tbl_s16(svt_caddq_s16_90(a[4], a[4]), idx);
    a[5] = svt_tbl_s16(svt_caddq_s16_90(a[5], a[5]), idx);
    a[6] = svt_tbl_s16(svt_caddq_s16_90(a[6], a[6]), idx);
    a[7] = svt_tbl_s16(svt_caddq_s16_90(a[7], a[7]), idx);

    a[0] = svt_tbl_s16(svt_caddq_s16_90(a[0], a[0]), idx);
    a[1] = svt_tbl_s16(svt_caddq_s16_90(a[1], a[1]), idx);
    a[2] = svt_tbl_s16(svt_caddq_s16_90(a[2], a[2]), idx);
    a[3] = svt_tbl_s16(svt_caddq_s16_90(a[3], a[3]), idx);
    a[4] = svt_tbl_s16(svt_caddq_s16_90(a[4], a[4]), idx);
    a[5] = svt_tbl_s16(svt_caddq_s16_90(a[5], a[5]), idx);
    a[6] = svt_tbl_s16(svt_caddq_s16_90(a[6], a[6]), idx);
    a[7] = svt_tbl_s16(svt_caddq_s16_90(a[7], a[7]), idx);

    a[0] = svt_caddq_s16_90(a[0], a[0]);
    a[1] = svt_caddq_s16_90(a[1], a[1]);
    a[2] = svt_caddq_s16_90(a[2], a[2]);
    a[3] = svt_caddq_s16_90(a[3], a[3]);
    a[4] = svt_caddq_s16_90(a[4], a[4]);
    a[5] = svt_caddq_s16_90(a[5], a[5]);
    a[6] = svt_caddq_s16_90(a[6], a[6]);
    a[7] = svt_caddq_s16_90(a[7], a[7]);
}

static inline void hadamard_8x8_sve2(const uint8_t* src, ptrdiff_t src_stride, const uint8_t* pred,
                                     ptrdiff_t pred_stride, int16x8_t* coeff, int32x4_t* satd, int32x4_t* dc) {
    uint8x8_t s[8];
    load_u8_8x8(src, src_stride, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7]);

    int16x8_t a[8];
    int16x8_t c0, c1, c2, c3, c4, c5, c6, c7;

    if (pred == NULL && coeff == NULL) {
        a[0] = vreinterpretq_s16_u16(vaddl_u8(s[0], s[1]));
        a[1] = vreinterpretq_s16_u16(vsubl_u8(s[0], s[1]));
        a[2] = vreinterpretq_s16_u16(vaddl_u8(s[2], s[3]));
        a[3] = vreinterpretq_s16_u16(vsubl_u8(s[2], s[3]));
        a[4] = vreinterpretq_s16_u16(vaddl_u8(s[4], s[5]));
        a[5] = vreinterpretq_s16_u16(vsubl_u8(s[4], s[5]));
        a[6] = vreinterpretq_s16_u16(vaddl_u8(s[6], s[7]));
        a[7] = vreinterpretq_s16_u16(vsubl_u8(s[6], s[7]));

        hadamard_8x8_cadd_pass_sve2(a);

        c0 = vabsq_s16(vaddq_s16(a[0], a[2]));
        c1 = vabdq_s16(a[0], a[2]);
        c2 = vabsq_s16(vaddq_s16(a[1], a[3]));
        c3 = vabdq_s16(a[1], a[3]);
        c4 = vabsq_s16(vaddq_s16(a[4], a[6]));
        c5 = vabdq_s16(a[4], a[6]);
        c6 = vabsq_s16(vaddq_s16(a[5], a[7]));
        c7 = vabdq_s16(a[5], a[7]);
    } else {
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

        hadamard_8x8_cadd_pass_sve2(a);

        if (coeff != NULL) {
            for (int i = 0; i < 8; ++i) {
                coeff[i] = a[i];
            }
            hadamard_8x8_v_pass_neon(coeff);
            return;
        }

        const int16x8_t b0 = vaddq_s16(a[0], a[1]);
        const int16x8_t b1 = vsubq_s16(a[0], a[1]);
        const int16x8_t b2 = vaddq_s16(a[2], a[3]);
        const int16x8_t b3 = vsubq_s16(a[2], a[3]);
        const int16x8_t b4 = vaddq_s16(a[4], a[5]);
        const int16x8_t b5 = vsubq_s16(a[4], a[5]);
        const int16x8_t b6 = vaddq_s16(a[6], a[7]);
        const int16x8_t b7 = vsubq_s16(a[6], a[7]);

        c0 = vabsq_s16(vaddq_s16(b0, b2));
        c1 = vabdq_s16(b0, b2);
        c2 = vabsq_s16(vaddq_s16(b1, b3));
        c3 = vabdq_s16(b1, b3);
        c4 = vabsq_s16(vaddq_s16(b4, b6));
        c5 = vabdq_s16(b4, b6);
        c6 = vabsq_s16(vaddq_s16(b5, b7));
        c7 = vabdq_s16(b5, b7);
    }

    if (dc != NULL) {
        *dc = vaddl_high_s16(c0, c4);
    }

    const int16x8_t max[4] = {
        vmaxq_s16(c0, c4),
        vmaxq_s16(c1, c5),
        vmaxq_s16(c2, c6),
        vmaxq_s16(c3, c7),
    };
    *satd = horizontal_add_4d_s16x8(max);
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
static inline void highbd_hadamard_8x8_sve2(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                            ptrdiff_t pred_stride, int32x4_t* coeff, int32x4_t* satd, int32x4_t* dc) {
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

    hadamard_8x8_cadd_pass_sve2(a);

    const int16x8_t b0 = vaddq_s16(a[0], a[1]);
    const int16x8_t b1 = vsubq_s16(a[0], a[1]);
    const int16x8_t b2 = vaddq_s16(a[2], a[3]);
    const int16x8_t b3 = vsubq_s16(a[2], a[3]);
    const int16x8_t b4 = vaddq_s16(a[4], a[5]);
    const int16x8_t b5 = vsubq_s16(a[4], a[5]);
    const int16x8_t b6 = vaddq_s16(a[6], a[7]);
    const int16x8_t b7 = vsubq_s16(a[6], a[7]);

    if (coeff != NULL) {
        const int16x8_t c0 = vaddq_s16(b0, b2);
        const int16x8_t c1 = vaddq_s16(b1, b3);
        const int16x8_t c2 = vsubq_s16(b0, b2);
        const int16x8_t c3 = vsubq_s16(b1, b3);
        const int16x8_t c4 = vaddq_s16(b4, b6);
        const int16x8_t c5 = vaddq_s16(b5, b7);
        const int16x8_t c6 = vsubq_s16(b4, b6);
        const int16x8_t c7 = vsubq_s16(b5, b7);

        coeff[0] = vaddl_s16(vget_low_s16(c0), vget_low_s16(c4));
        coeff[1] = vsubl_s16(vget_low_s16(c2), vget_low_s16(c6));
        coeff[2] = vsubl_s16(vget_low_s16(c0), vget_low_s16(c4));
        coeff[3] = vaddl_s16(vget_low_s16(c2), vget_low_s16(c6));
        coeff[4] = vaddl_s16(vget_low_s16(c3), vget_low_s16(c7));
        coeff[5] = vsubl_s16(vget_low_s16(c3), vget_low_s16(c7));
        coeff[6] = vsubl_s16(vget_low_s16(c1), vget_low_s16(c5));
        coeff[7] = vaddl_s16(vget_low_s16(c1), vget_low_s16(c5));

        coeff[8]  = vaddl_s16(vget_high_s16(c0), vget_high_s16(c4));
        coeff[9]  = vsubl_s16(vget_high_s16(c2), vget_high_s16(c6));
        coeff[10] = vsubl_s16(vget_high_s16(c0), vget_high_s16(c4));
        coeff[11] = vaddl_s16(vget_high_s16(c2), vget_high_s16(c6));
        coeff[12] = vaddl_s16(vget_high_s16(c3), vget_high_s16(c7));
        coeff[13] = vsubl_s16(vget_high_s16(c3), vget_high_s16(c7));
        coeff[14] = vsubl_s16(vget_high_s16(c1), vget_high_s16(c5));
        coeff[15] = vaddl_s16(vget_high_s16(c1), vget_high_s16(c5));
        return;
    }

    const int16x8_t c0 = vabsq_s16(vaddq_s16(b0, b2));
    const int16x8_t c1 = vabdq_s16(b0, b2);
    const int16x8_t c2 = vabsq_s16(vaddq_s16(b1, b3));
    const int16x8_t c3 = vabdq_s16(b1, b3);
    const int16x8_t c4 = vabsq_s16(vaddq_s16(b4, b6));
    const int16x8_t c5 = vabdq_s16(b4, b6);
    const int16x8_t c6 = vabsq_s16(vaddq_s16(b5, b7));
    const int16x8_t c7 = vabdq_s16(b5, b7);

    if (dc != NULL) {
        *dc = vaddl_high_s16(c0, c4);
    }

    const int16x8_t max[4] = {
        vmaxq_s16(c0, c4),
        vmaxq_s16(c1, c5),
        vmaxq_s16(c2, c6),
        vmaxq_s16(c3, c7),
    };
    *satd = horizontal_add_4d_s16x8(max);
}
#endif

#endif // SVT_AV1_HADAMARD_PATH_SVE2_H_
