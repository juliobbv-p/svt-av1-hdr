/*
* Copyright(c) 2024-2025 Psychovisual Experts Group
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include <math.h>
#include <stdbool.h>
#include "ac_bias.h"
#include "aom_dsp_rtcd.h"
#include "common_dsp_rtcd.h"

static int psy_energy_4x4_c(const uint8_t* src, ptrdiff_t src_stride) {
    int16_t src16[16];
    int32_t coeff[16];

    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            src16[row * 4 + col] = src[row * src_stride + col];
        }
    }

    svt_aom_hadamard_4x4(src16, 4, coeff);

    return (svt_aom_satd(coeff, 16) << 1) - coeff[0];
}

static int psy_energy_8x8_c(const uint8_t* src, ptrdiff_t src_stride) {
    int16_t src16[64];
    int32_t coeff[64];

    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            src16[row * 8 + col] = src[row * src_stride + col];
        }
    }

    svt_aom_hadamard_8x8(src16, 8, coeff);

    return ((svt_aom_satd(coeff, 64) + 2) >> 2) - ((coeff[0] + 2) >> 2);
}

/* Regular version of "AC Bias"
 *
 * Based on adding an "energy gap" term to each candidate block's distortion, which is the difference
 * of the "energy" (SATD - SAD) of the source and recon blocks
 */
uint64_t svt_psy_distortion_c(const uint8_t* input, const uint32_t input_stride, const uint8_t* recon,
                              const uint32_t recon_stride, const uint32_t width, const uint32_t height) {
    uint64_t energy_gap = 0;

    if (width >= 8 && height >= 8) { /* >8x8 */
        for (uint32_t j = 0; j < height; j += 8) {
            for (uint32_t i = 0; i < width; i += 8) {
                const int input_energy = psy_energy_8x8_c(input + j * input_stride + i, input_stride);
                const int recon_energy = psy_energy_8x8_c(recon + j * recon_stride + i, recon_stride);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    } else {
        for (uint32_t j = 0; j < height; j += 4) { /* 4x4, 4x8, 4x16, 8x4, and 16x4 */
            for (uint32_t i = 0; i < width; i += 4) {
                const int input_energy = psy_energy_4x4_c(input + j * input_stride + i, input_stride);
                const int recon_energy = psy_energy_4x4_c(recon + j * recon_stride + i, recon_stride);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    }

    return energy_gap;
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
static int highbd_psy_energy_4x4_c(const uint16_t* src, ptrdiff_t src_stride) {
    int32_t coeff[16];

    svt_aom_hadamard_4x4((const int16_t*)src, src_stride, coeff);

    return (svt_aom_satd(coeff, 16) << 1) - coeff[0];
}

static int highbd_psy_energy_8x8_c(const uint16_t* src, ptrdiff_t src_stride) {
    int32_t coeff[64];

    svt_aom_highbd_hadamard_8x8((const int16_t*)src, src_stride, coeff);

    return ((svt_aom_satd(coeff, 64) + 2) >> 2) - ((coeff[0] + 2) >> 2);
}

/* High bit-depth version of "AC Bias" */
uint64_t svt_psy_distortion_hbd_c(const uint16_t* input, const uint32_t input_stride, const uint16_t* recon,
                                  const uint32_t recon_stride, const uint32_t width, const uint32_t height) {
    uint64_t energy_gap = 0;

    if (width >= 8 && height >= 8) { /* >8x8 */
        for (uint32_t j = 0; j < height; j += 8) {
            for (uint32_t i = 0; i < width; i += 8) {
                const int input_energy = highbd_psy_energy_8x8_c(input + j * input_stride + i, input_stride);
                const int recon_energy = highbd_psy_energy_8x8_c(recon + j * recon_stride + i, recon_stride);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    } else {
        for (uint64_t j = 0; j < height; j += 4) { /* 4x4, 4x8, 4x16, 8x4, and 16x4 */
            for (uint64_t i = 0; i < width; i += 4) {
                const int input_energy = highbd_psy_energy_4x4_c(input + j * input_stride + i, input_stride);
                const int recon_energy = highbd_psy_energy_4x4_c(recon + j * recon_stride + i, recon_stride);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    }

    // Energy is scaled to approximately match equivalent 8-bit strengths
    return energy_gap << 2;
}
#endif

/*
 * Public function that mirrors the arguments of `spatial_full_dist_type_fun()`
 */
uint64_t get_svt_psy_full_dist(const void* s, const uint32_t so, const uint32_t sp, const void* r, const uint32_t ro,
                               const uint32_t rp, const uint32_t w, const uint32_t h, const uint8_t is_hbd,
                               const double ac_bias) {
    if (is_hbd)
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
        return llrint(svt_psy_distortion_hbd((const uint16_t*)s + so, sp, (uint16_t*)r + ro, rp, w, h) * ac_bias);
#else
        return 0;
#endif
    else {
        return llrint(svt_psy_distortion((const uint8_t*)s + so, sp, (const uint8_t*)r + ro, rp, w, h) * ac_bias);
    }
}

/*
 * Light version of "AC Bias", called by the Light-PD code paths
 *
 * Based on adjusting each block's rate so blocks with more energy (sum of AC coeffs) appear "cheaper" to the encoder,
 * thus making them more favorable to be picked by the RDO process. This tends to increase the image's total "energy"
 * (in contrast to `get_svt_psy_full_dist()` which tries to reduce the "energy gap" between source and recon)
 *
 * Much faster than `get_svt_psy_full_dist()` as it can re-use existing block coefficients instead of computing new
 * ones, but subjective visual quality benefits are significantly more modest
 */
uint64_t svt_psy_adjust_rate_light(const int32_t* coeff, uint64_t coeff_bits, const uint32_t width,
                                   const uint32_t height, const double ac_bias) {
    uint64_t       energy = 0;
    const int32_t* buf    = coeff;

    for (uint32_t j = 0; j < height; j++) {
        // Skip the DC coefficient from the calculation
        for (uint32_t i = j ? 0 : 1; i < width; i++) {
            energy += (uint64_t)llabs((int64_t)buf[i]);
        }
        buf += width;
    }

    if (energy > 0) {
        uint64_t coeff_bits_adj = (int)(energy * ac_bias * 100);

        // When the adjustment rate is greater than the rate, keep rate (coeff_bits) positive
        coeff_bits = (coeff_bits > coeff_bits_adj) ? (coeff_bits - coeff_bits_adj) : 1;
    }

    return coeff_bits;
}

double get_effective_ac_bias(const double ac_bias, const bool is_islice, const uint8_t temporal_layer_index) {
    if (is_islice) {
        return ac_bias * 0.3;
    }
    switch (temporal_layer_index) {
    case 0:
        return ac_bias * 0.6;
    case 1:
        return ac_bias * 0.8;
    case 2:
        return ac_bias * 0.9;
    default:
        return ac_bias;
    }
}
