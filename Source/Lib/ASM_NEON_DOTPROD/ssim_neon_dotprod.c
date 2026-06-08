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
#include <string.h>

#include "aom_dsp_rtcd.h"
#include "enc_dec_process.h"

double svt_ssim_8x8_neon_dotprod(const uint8_t* s, uint32_t sp, const uint8_t* r, uint32_t rp) {
    const uint8x16_t ones  = vdupq_n_u8(1);
    uint32x4_t       acc_s = vdupq_n_u32(0), acc_r = vdupq_n_u32(0);
    uint32x4_t       acc_sq_s = vdupq_n_u32(0), acc_sq_r = vdupq_n_u32(0), acc_sxr = vdupq_n_u32(0);
    for (int i = 0; i < 8; i += 2) {
        // Two 8-wide rows packed into one 16-byte vector.
        const uint8x16_t vs = vcombine_u8(vld1_u8(s), vld1_u8(s + sp));
        const uint8x16_t vr = vcombine_u8(vld1_u8(r), vld1_u8(r + rp));
        acc_s               = vdotq_u32(acc_s, vs, ones);
        acc_r               = vdotq_u32(acc_r, vr, ones);
        acc_sq_s            = vdotq_u32(acc_sq_s, vs, vs);
        acc_sq_r            = vdotq_u32(acc_sq_r, vr, vr);
        acc_sxr             = vdotq_u32(acc_sxr, vs, vr);
        s += 2 * sp;
        r += 2 * rp;
    }
    return svt_aom_similarity(
        vaddvq_u32(acc_s), vaddvq_u32(acc_r), vaddvq_u32(acc_sq_s), vaddvq_u32(acc_sq_r), vaddvq_u32(acc_sxr), 64, 8);
}

double svt_ssim_4x4_neon_dotprod(const uint8_t* s, uint32_t sp, const uint8_t* r, uint32_t rp) {
    // Pack the four 4-byte rows into one 16-byte vector.
    uint32_t s0, s1, s2, s3, r0, r1, r2, r3;
    memcpy(&s0, s, 4);
    memcpy(&s1, s + sp, 4);
    memcpy(&s2, s + 2 * sp, 4);
    memcpy(&s3, s + 3 * sp, 4);
    memcpy(&r0, r, 4);
    memcpy(&r1, r + rp, 4);
    memcpy(&r2, r + 2 * rp, 4);
    memcpy(&r3, r + 3 * rp, 4);
    const uint8x16_t vs   = vreinterpretq_u8_u32((uint32x4_t){s0, s1, s2, s3});
    const uint8x16_t vr   = vreinterpretq_u8_u32((uint32x4_t){r0, r1, r2, r3});
    const uint8x16_t ones = vdupq_n_u8(1);
    return svt_aom_similarity(vaddvq_u32(vdotq_u32(vdupq_n_u32(0), vs, ones)),
                              vaddvq_u32(vdotq_u32(vdupq_n_u32(0), vr, ones)),
                              vaddvq_u32(vdotq_u32(vdupq_n_u32(0), vs, vs)),
                              vaddvq_u32(vdotq_u32(vdupq_n_u32(0), vr, vr)),
                              vaddvq_u32(vdotq_u32(vdupq_n_u32(0), vs, vr)),
                              16,
                              8);
}
