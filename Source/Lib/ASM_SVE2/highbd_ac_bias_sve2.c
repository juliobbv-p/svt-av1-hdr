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
#include <stdlib.h>

#include "common_dsp_rtcd.h"
#include "hadamard_path_neon.h"
#include "hadamard_path_sve2.h"

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
static int highbd_psy_energy_8x8_sve2(const uint16_t* src, ptrdiff_t src_stride) {
    int32x4_t dc;
    int32x4_t sum;
    highbd_hadamard_8x8_sve2(src, src_stride, NULL, 0, NULL, &sum, &dc);

    const int satd   = vaddvq_s32(sum) << 1;
    const int dc_sum = vgetq_lane_s32(dc, 3);
    return ((satd + 2) >> 2) - ((dc_sum + 2) >> 2);
}

static int32x4_t highbd_psy_energy_16x16_sve2(const uint16_t* src, ptrdiff_t src_stride) {
    int32x4_t dc_coeff[4];
    int32x4_t sum[4];

    highbd_hadamard_8x8_sve2(src + 0 * src_stride + 0, src_stride, NULL, 0, NULL, &sum[0], &dc_coeff[0]);
    highbd_hadamard_8x8_sve2(src + 0 * src_stride + 8, src_stride, NULL, 0, NULL, &sum[1], &dc_coeff[1]);
    highbd_hadamard_8x8_sve2(src + 8 * src_stride + 0, src_stride, NULL, 0, NULL, &sum[2], &dc_coeff[2]);
    highbd_hadamard_8x8_sve2(src + 8 * src_stride + 8, src_stride, NULL, 0, NULL, &sum[3], &dc_coeff[3]);

    dc_coeff[0] = vzip2q_s32(dc_coeff[0], dc_coeff[1]);
    dc_coeff[2] = vzip2q_s32(dc_coeff[2], dc_coeff[3]);
    dc_coeff[0] = vcombine_s32(vget_high_s32(dc_coeff[0]), vget_high_s32(dc_coeff[2]));

    sum[0] = vpaddq_s32(sum[0], sum[1]);
    sum[2] = vpaddq_s32(sum[2], sum[3]);
    sum[0] = vpaddq_s32(sum[0], sum[2]);

    const int32x4_t dc_energy   = vrshrq_n_s32(vabsq_s32(dc_coeff[0]), 2);
    const int32x4_t sa8d_energy = vrshrq_n_s32(sum[0], 1);

    return vsubq_s32(sa8d_energy, dc_energy);
}

uint64_t svt_psy_distortion_hbd_sve2(const uint16_t* input, const uint32_t input_stride, const uint16_t* recon,
                                     const uint32_t recon_stride, const uint32_t width, const uint32_t height) {
    uint64_t energy_gap = 0;

    if (width % 16 == 0 && height % 16 == 0) {
        int32x4_t energy_gap_16x16 = vdupq_n_s32(0);

        for (uint32_t j = 0; j < height; j += 16) {
            for (uint32_t i = 0; i < width; i += 16) {
                const int32x4_t input_energy = highbd_psy_energy_16x16_sve2(input + j * input_stride + i, input_stride);
                const int32x4_t recon_energy = highbd_psy_energy_16x16_sve2(recon + j * recon_stride + i, recon_stride);

                energy_gap_16x16 = vabaq_s32(energy_gap_16x16, input_energy, recon_energy);
            }
        }

        energy_gap += (uint64_t)vaddlvq_s32(energy_gap_16x16);
    } else if (width >= 8 && height >= 8) {
        for (uint32_t j = 0; j < height; j += 8) {
            for (uint32_t i = 0; i < width; i += 8) {
                const int input_energy = highbd_psy_energy_8x8_sve2(input + j * input_stride + i, input_stride);
                const int recon_energy = highbd_psy_energy_8x8_sve2(recon + j * recon_stride + i, recon_stride);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    } else {
        for (uint32_t j = 0; j < height; j += 4) {
            for (uint32_t i = 0; i < width; i += 4) {
                const int input_energy = highbd_psy_energy_4x4_neon(input + j * input_stride + i, input_stride);
                const int recon_energy = highbd_psy_energy_4x4_neon(recon + j * recon_stride + i, recon_stride);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    }

    return energy_gap << 2;
}
#endif
