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
#include "enc_dec_process.h"

static AOM_FORCE_INLINE void ssim_hbd_row(uint16x8_t vs, uint16x8_t vr, uint32x4_t* acc_s, uint32x4_t* acc_r,
                                          uint32x4_t* acc_sq_s, uint32x4_t* acc_sq_r, uint32x4_t* acc_sxr) {
    *acc_s    = vpadalq_u16(*acc_s, vs);
    *acc_r    = vpadalq_u16(*acc_r, vr);
    *acc_sq_s = vmlal_u16(vmlal_high_u16(*acc_sq_s, vs, vs), vget_low_u16(vs), vget_low_u16(vs));
    *acc_sq_r = vmlal_u16(vmlal_high_u16(*acc_sq_r, vr, vr), vget_low_u16(vr), vget_low_u16(vr));
    *acc_sxr  = vmlal_u16(vmlal_high_u16(*acc_sxr, vs, vr), vget_low_u16(vs), vget_low_u16(vr));
}

double svt_ssim_8x8_hbd_neon(const uint16_t* s, uint32_t sp, const uint16_t* r, uint32_t rp) {
    uint32x4_t acc_s = vdupq_n_u32(0), acc_r = vdupq_n_u32(0);
    uint32x4_t acc_sq_s = vdupq_n_u32(0), acc_sq_r = vdupq_n_u32(0), acc_sxr = vdupq_n_u32(0);
    for (int i = 0; i < 8; ++i) {
        ssim_hbd_row(vld1q_u16(s), vld1q_u16(r), &acc_s, &acc_r, &acc_sq_s, &acc_sq_r, &acc_sxr);
        s += sp;
        r += rp;
    }
    return svt_aom_similarity(
        vaddvq_u32(acc_s), vaddvq_u32(acc_r), vaddvq_u32(acc_sq_s), vaddvq_u32(acc_sq_r), vaddvq_u32(acc_sxr), 64, 10);
}

double svt_ssim_4x4_hbd_neon(const uint16_t* s, uint32_t sp, const uint16_t* r, uint32_t rp) {
    uint32x4_t acc_s = vdupq_n_u32(0), acc_r = vdupq_n_u32(0);
    uint32x4_t acc_sq_s = vdupq_n_u32(0), acc_sq_r = vdupq_n_u32(0), acc_sxr = vdupq_n_u32(0);
    for (int i = 0; i < 4; ++i) {
        const uint16x4_t vs = vld1_u16(s);
        const uint16x4_t vr = vld1_u16(r);
        acc_s               = vaddw_u16(acc_s, vs);
        acc_r               = vaddw_u16(acc_r, vr);
        acc_sq_s            = vmlal_u16(acc_sq_s, vs, vs);
        acc_sq_r            = vmlal_u16(acc_sq_r, vr, vr);
        acc_sxr             = vmlal_u16(acc_sxr, vs, vr);
        s += sp;
        r += rp;
    }
    return svt_aom_similarity(
        vaddvq_u32(acc_s), vaddvq_u32(acc_r), vaddvq_u32(acc_sq_s), vaddvq_u32(acc_sq_r), vaddvq_u32(acc_sxr), 16, 10);
}
