/*
* Copyright(c) 2024 Gianni Rosato
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include "ac_bias.h"
#include "aom_dsp_rtcd.h"

#define BITS_PER_SUM 16
#define BITS_PER_SUM_HBD 32

// in: a pseudo-simd number of the form x+(y<<EW)
// return: abs(x)+(abs(y)<<16)
static inline uint16x2_t abs2(const uint16x2_t a) {
    const uint16x2_t mask = (a >> (BITS_PER_SUM - 1)) & (((uint16x2_t)1 << BITS_PER_SUM) + 1);
    const uint16x2_t s = (mask << BITS_PER_SUM) - mask;
    return (a + s) ^ s;
}

static inline uint32x2_t abs2_hbd(const uint32x2_t a) {
    const uint32x2_t mask = (a >> (BITS_PER_SUM_HBD - 1)) & (((uint32x2_t)1 << BITS_PER_SUM_HBD) + 1);
    const uint32x2_t s = (mask << BITS_PER_SUM_HBD) - mask;
    return (a + s) ^ s;
}

static void svt_psy_hadamard_4x4_diff_c(uint32_t* a, uint32_t* b, uint32_t* c, uint32_t* d, uint32_t w, uint32_t x, uint32_t y, uint32_t z) {
    const uint16x2_t s0 = w, s1 = x, s2 = y, s3 = z;
    const uint16x2_t t0 = s0 + s1, t1 = s0 - s1, t2 = s2 + s3, t3 = s2 - s3;
    *a = t0 + t2;
    *b = t0 - t2;
    *c = t1 + t3;
    *d = t1 - t3;
}

static void svt_psy_hadamard_4x4_hbd_diff_c(uint64_t* a, uint64_t* b, uint64_t* c, uint64_t* d, uint64_t w, uint64_t x, uint64_t y, uint64_t z) {
    const uint64_t s0 = w, s1 = x, s2 = y, s3 = z;
    const uint64_t t0 = s0 + s1, t1 = s0 - s1, t2 = s2 + s3, t3 = s2 - s3;
    *a = t0 + t2;
    *b = t0 - t2;
    *c = t1 + t3;
    *d = t1 - t3;
}

/*
 * 8-bit functions
 */
static uint64_t svt_psy_sa8d_8x8(const uint8_t* s, const uint32_t sp, const uint8_t* r, const uint32_t rp) {
    uint16x2_t tmp[8][4];
    uint16x2_t a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3;
    uint16x2_t sum = 0;

    for (int i = 0; i < 8; i++, s += sp, r += rp) {
        a0 = s[0] - r[0];
        a1 = s[1] - r[1];
        b0 = (a0 + a1) + ((a0 - a1) << BITS_PER_SUM);
        a2 = s[2] - r[2];
        a3 = s[3] - r[3];
        b1 = (a2 + a3) + ((a2 - a3) << BITS_PER_SUM);
        a4 = s[4] - r[4];
        a5 = s[5] - r[5];
        b2 = (a4 + a5) + ((a4 - a5) << BITS_PER_SUM);
        a6 = s[6] - r[6];
        a7 = s[7] - r[7];
        b3 = (a6 + a7) + ((a6 - a7) << BITS_PER_SUM);
        svt_psy_hadamard_4x4_diff_c(&tmp[i][0], &tmp[i][1], &tmp[i][2], &tmp[i][3], b0, b1, b2, b3);
    }
    for (int i = 0; i < 4; i++) {
        svt_psy_hadamard_4x4_diff_c(&a0, &a1, &a2, &a3, tmp[0][i], tmp[1][i], tmp[2][i], tmp[3][i]);
        svt_psy_hadamard_4x4_diff_c(&a4, &a5, &a6, &a7, tmp[4][i], tmp[5][i], tmp[6][i], tmp[7][i]);
        b0  = abs2(a0 + a4) + abs2(a0 - a4);
        b0 += abs2(a1 + a5) + abs2(a1 - a5);
        b0 += abs2(a2 + a6) + abs2(a2 - a6);
        b0 += abs2(a3 + a7) + abs2(a3 - a7);
        sum += (uint16_t)b0 + (b0 >> BITS_PER_SUM);
    }

    return (uint64_t)((sum + 2) >> 2);
}

static uint64_t svt_psy_satd_4x4(const uint8_t* s, const uint32_t sp, const uint8_t* r, const uint32_t rp) {
    uint16x2_t tmp[4][2];
    uint16x2_t a0, a1, a2, a3, b0, b1;
    uint16x2_t sum = 0;

    for (int i = 0; i < 4; i++, s += sp, r += rp) {
        a0 = s[0] - r[0];
        a1 = s[1] - r[1];
        b0 = (a0 + a1) + ((a0 - a1) << BITS_PER_SUM);
        a2 = s[2] - r[2];
        a3 = s[3] - r[3];
        b1 = (a2 + a3) + ((a2 - a3) << BITS_PER_SUM);
        tmp[i][0] = b0 + b1;
        tmp[i][1] = b0 - b1;
    }
    for (int i = 0; i < 2; i++) {
        svt_psy_hadamard_4x4_diff_c(&a0, &a1, &a2, &a3, tmp[0][i], tmp[1][i], tmp[2][i], tmp[3][i]);
        a0 = abs2(a0) + abs2(a1) + abs2(a2) + abs2(a3);
        sum += ((uint16_t)a0) + (a0 >> BITS_PER_SUM);
    }

    return (uint64_t)(sum >> 1);
}

uint64_t svt_psy_distortion(const uint8_t* input, const uint32_t input_stride,
                            const uint8_t* recon, const uint32_t recon_stride,
                            const uint32_t width, const uint32_t height) {

    static uint8_t zero_buffer[8] = { 0 };
    uint64_t total_nrg = 0;

    if (width >= 8 && height >= 8) { /* >8x8 */
        for (uint64_t i = 0; i < height; i += 8)
            for (uint64_t j = 0; j < width; j += 8) {
                const int32_t input_nrg = svt_psy_sa8d_8x8(input + i * input_stride + j, input_stride, zero_buffer, 0) -
                    (svt_aom_sad8x8(input + i * input_stride + j, input_stride, zero_buffer, 0) >> 2);
                const int32_t recon_nrg = svt_psy_sa8d_8x8(recon + i * recon_stride + j, recon_stride, zero_buffer, 0) -
                    (svt_aom_sad8x8(recon + i * recon_stride + j, recon_stride, zero_buffer, 0) >> 2);
                total_nrg += abs(input_nrg - recon_nrg);
            }
        return (total_nrg >> 1);
    }
    for (uint64_t i = 0; i < height; i += 4)
        for (uint64_t j = 0; j < width; j += 4) {
            const int32_t input_nrg = svt_psy_satd_4x4(input + i * input_stride + j, input_stride, zero_buffer, 0) -
                (svt_aom_sad4x4(input + i * input_stride + j, input_stride, zero_buffer, 0) >> 2);
            const int32_t recon_nrg = svt_psy_satd_4x4(recon + i * recon_stride + j, recon_stride, zero_buffer, 0) -
                (svt_aom_sad4x4(recon + i * recon_stride + j, recon_stride, zero_buffer, 0) >> 2);
            total_nrg += abs(input_nrg - recon_nrg);
        }
    // Energy is scaled to match equivalent HBD strengths
    return (total_nrg >> 1);
}

/*
 * 10-bit functions
 */
static uint64_t svt_psy_sa8d_8x8_hbd(const uint16_t* s, const uint32_t sp, const uint16_t* r, const uint32_t rp) {
    uint32x2_t tmp[8][4];
    uint32x2_t a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3;
    uint32x2_t sum = 0;

    for (int i = 0; i < 8; i++, s += sp, r += rp) {
        a0 = s[0] - r[0];
        a1 = s[1] - r[1];
        b0 = (a0 + a1) + ((a0 - a1) << BITS_PER_SUM_HBD);
        a2 = s[2] - r[2];
        a3 = s[3] - r[3];
        b1 = (a2 + a3) + ((a2 - a3) << BITS_PER_SUM_HBD);
        a4 = s[4] - r[4];
        a5 = s[5] - r[5];
        b2 = (a4 + a5) + ((a4 - a5) << BITS_PER_SUM_HBD);
        a6 = s[6] - r[6];
        a7 = s[7] - r[7];
        b3 = (a6 + a7) + ((a6 - a7) << BITS_PER_SUM_HBD);
        svt_psy_hadamard_4x4_hbd_diff_c(&tmp[i][0], &tmp[i][1], &tmp[i][2], &tmp[i][3], b0, b1, b2, b3);
    }
    for (int i = 0; i < 4; i++) {
        svt_psy_hadamard_4x4_hbd_diff_c(&a0, &a1, &a2, &a3, tmp[0][i], tmp[1][i], tmp[2][i], tmp[3][i]);
        svt_psy_hadamard_4x4_hbd_diff_c(&a4, &a5, &a6, &a7, tmp[4][i], tmp[5][i], tmp[6][i], tmp[7][i]);
        b0  = abs2_hbd(a0 + a4) + abs2_hbd(a0 - a4);
        b0 += abs2_hbd(a1 + a5) + abs2_hbd(a1 - a5);
        b0 += abs2_hbd(a2 + a6) + abs2_hbd(a2 - a6);
        b0 += abs2_hbd(a3 + a7) + abs2_hbd(a3 - a7);
        sum += (uint32_t)b0 + (b0 >> BITS_PER_SUM_HBD);
    }

    return (sum + 2) >> 2;
}

static uint64_t svt_psy_satd_4x4_hbd(const uint16_t* s, const uint32_t sp, const uint16_t* r, const uint32_t rp) {
    uint32x2_t tmp[4][2];
    uint32x2_t a0, a1, a2, a3, b0, b1;
    uint32x2_t sum = 0;

    for (int i = 0; i < 4; i++, s += sp, r += rp) {
        a0 = s[0] - r[0];
        a1 = s[1] - r[1];
        b0 = (a0 + a1) + ((a0 - a1) << BITS_PER_SUM_HBD);
        a2 = s[2] - r[2];
        a3 = s[3] - r[3];
        b1 = (a2 + a3) + ((a2 - a3) << BITS_PER_SUM_HBD);
        tmp[i][0] = b0 + b1;
        tmp[i][1] = b0 - b1;
    }
    for (int i = 0; i < 2; i++) {
        svt_psy_hadamard_4x4_hbd_diff_c(&a0, &a1, &a2, &a3, tmp[0][i], tmp[1][i], tmp[2][i], tmp[3][i]);
        a0 = abs2_hbd(a0) + abs2_hbd(a1) + abs2_hbd(a2) + abs2_hbd(a3);
        sum += ((uint32_t)a0) + (a0 >> BITS_PER_SUM_HBD);
    }

    return sum >> 1;
}

uint64_t svt_psy_distortion_hbd(uint16_t* input, const uint32_t input_stride,
                                uint16_t* recon, const uint32_t recon_stride,
                                const uint32_t width, const uint32_t height) {

    static uint16_t zero_buffer[8] = { 0 };
    uint64_t total_nrg = 0;

    if (width >= 8 && height >= 8) { /* >8x8 */
        for (uint64_t i = 0; i < height; i += 8)
            for (uint64_t j = 0; j < width; j += 8) {
                const int32_t input_nrg = (svt_psy_sa8d_8x8_hbd(input + i * input_stride + j, input_stride, zero_buffer, 0)) -
                    (sad_16b_kernel(input + i * input_stride + j, input_stride, zero_buffer, 0, 8, 8) >> 2);
                const int32_t recon_nrg = (svt_psy_sa8d_8x8_hbd(recon + i * recon_stride + j, recon_stride, zero_buffer, 0)) -
                    (sad_16b_kernel(recon + i * recon_stride + j, recon_stride, zero_buffer, 0, 8, 8) >> 2);
                total_nrg += abs(input_nrg - recon_nrg);
            }
        return (total_nrg << 2);
    }
    for (uint64_t i = 0; i < height; i += 4) /* 4x4, 4x8, 4x16, 8x4, and 16x4 */
        for (uint64_t j = 0; j < width; j += 4) {
            const int32_t input_nrg = svt_psy_satd_4x4_hbd(input + i * input_stride + j, input_stride, zero_buffer, 0) -
                (sad_16b_kernel(input + i * input_stride + j, input_stride, zero_buffer, 0, 4, 4) >> 2);
            const int32_t recon_nrg = svt_psy_satd_4x4_hbd(recon + i * recon_stride + j, recon_stride, zero_buffer, 0) -
                (sad_16b_kernel(recon + i * recon_stride + j, recon_stride, zero_buffer, 0, 4, 4) >> 2);
            total_nrg += abs(input_nrg - recon_nrg);
        }
    return (total_nrg << 2);
}

/*
 * Public function that mirrors the arguments of `spatial_full_dist_type_fun()`
 */
uint64_t get_svt_psy_full_dist(const void* s, const uint32_t so, const uint32_t sp,
                               const void* r, const uint32_t ro, const uint32_t rp,
                               const uint32_t w, const uint32_t h, const uint8_t is_hbd,
                               const double ac_bias) {
    if (is_hbd)
        return llrint(svt_psy_distortion_hbd((uint16_t*)s + so, sp, (uint16_t*)r + ro, rp, w, h) * ac_bias);
    else
        return llrint(svt_psy_distortion((const uint8_t*)s + so, sp, (const uint8_t*)r + ro, rp, w, h) * ac_bias);
}

double get_effective_ac_bias(const double ac_bias, const bool is_islice, const uint8_t temporal_layer_index) {
    if (is_islice) return ac_bias * 0.4;
    switch (temporal_layer_index) {
    case 0: return ac_bias * 0.75;
    case 1: return ac_bias * 0.9;
    case 2: return ac_bias * 0.95;
    default: return ac_bias;
    }
}
