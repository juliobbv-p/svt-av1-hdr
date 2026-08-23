/*
 * Copyright (c) 2025, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>
#include <arm_neon_sve_bridge.h>
#include <assert.h>
#include <math.h>

#include "aom_dsp_rtcd.h"
#include "utility.h"

int32_t svt_av1_compute_cul_level_sve(const int16_t* const scan, const int32_t* const quant_coeff, int32_t eob,
                                      int32_t n_coeffs) {
    int32x4_t sum_s32[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};

    if (eob * 12 < n_coeffs) {
        // Sparse: scan-based gather, bounded by eob.
        int32x4_t      zeros    = vdupq_n_s32(0);
        int32_t        sz       = eob;
        const svbool_t p4       = svptrue_pat_b32(SV_VL4);
        const int16_t* scan_ptr = scan;
        while (sz >= 8) {
            svint32_t scan_lo = svld1sh_s32(p4, scan_ptr + 0);
            svint32_t scan_hi = svld1sh_s32(p4, scan_ptr + 4);

            svint32_t       quant_sve_lo     = svld1_gather_s32index_s32(p4, quant_coeff, scan_lo);
            svint32_t       quant_sve_hi     = svld1_gather_s32index_s32(p4, quant_coeff, scan_hi);
            const int32x4_t quant_coeff_0123 = svget_neonq_s32(quant_sve_lo);
            const int32x4_t quant_coeff_4567 = svget_neonq_s32(quant_sve_hi);

            sum_s32[0] = vabaq_s32(sum_s32[0], quant_coeff_0123, zeros);
            sum_s32[1] = vabaq_s32(sum_s32[1], quant_coeff_4567, zeros);

            scan_ptr += 8;
            sz -= 8;
        }

        if (sz != 0) {
            const svbool_t p0 = svwhilelt_b32_s32(0, sz);
            const svbool_t p1 = svwhilelt_b32_s32(4, sz);

            svint32_t scan_lo = svld1sh_s32(p0, scan_ptr + 0);
            svint32_t scan_hi = svld1sh_s32(p1, scan_ptr + 4);

            svint32_t       quant_sve_lo     = svld1_gather_s32index_s32(p0, quant_coeff, scan_lo);
            svint32_t       quant_sve_hi     = svld1_gather_s32index_s32(p1, quant_coeff, scan_hi);
            const int32x4_t quant_coeff_0123 = svget_neonq_s32(quant_sve_lo);
            const int32x4_t quant_coeff_4567 = svget_neonq_s32(quant_sve_hi);

            sum_s32[0] = vabaq_s32(sum_s32[0], quant_coeff_0123, zeros);
            sum_s32[1] = vabaq_s32(sum_s32[1], quant_coeff_4567, zeros);
        }
        return vaddvq_s32(vaddq_s32(sum_s32[0], sum_s32[1]));
    }

    // Dense: gather-free linear over the whole block. n_coeffs is always a multiple of 16.
    assert(n_coeffs % 8 == 0);
    for (int32_t i = 0; i < n_coeffs; i += 8) {
        sum_s32[0] = vaddq_s32(sum_s32[0], vabsq_s32(vld1q_s32(quant_coeff + i)));
        sum_s32[1] = vaddq_s32(sum_s32[1], vabsq_s32(vld1q_s32(quant_coeff + i + 4)));
    }
    return vaddvq_s32(vaddq_s32(sum_s32[0], sum_s32[1]));
}
