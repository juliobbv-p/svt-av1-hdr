/*
* Copyright(c) 2019 Intel Corporation
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include <stddef.h>
#include <stdint.h>

#include "aom_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include "utility.h"

static void hadamard_highbd_col8_first_pass(const int16_t* src_diff, ptrdiff_t src_stride, int16_t* coeff) {
    int16_t b0 = src_diff[0 * src_stride] + src_diff[1 * src_stride];
    int16_t b1 = src_diff[0 * src_stride] - src_diff[1 * src_stride];
    int16_t b2 = src_diff[2 * src_stride] + src_diff[3 * src_stride];
    int16_t b3 = src_diff[2 * src_stride] - src_diff[3 * src_stride];
    int16_t b4 = src_diff[4 * src_stride] + src_diff[5 * src_stride];
    int16_t b5 = src_diff[4 * src_stride] - src_diff[5 * src_stride];
    int16_t b6 = src_diff[6 * src_stride] + src_diff[7 * src_stride];
    int16_t b7 = src_diff[6 * src_stride] - src_diff[7 * src_stride];

    int16_t c0 = b0 + b2;
    int16_t c1 = b1 + b3;
    int16_t c2 = b0 - b2;
    int16_t c3 = b1 - b3;
    int16_t c4 = b4 + b6;
    int16_t c5 = b5 + b7;
    int16_t c6 = b4 - b6;
    int16_t c7 = b5 - b7;

    coeff[0] = c0 + c4;
    coeff[7] = c1 + c5;
    coeff[3] = c2 + c6;
    coeff[4] = c3 + c7;
    coeff[2] = c0 - c4;
    coeff[6] = c1 - c5;
    coeff[1] = c2 - c6;
    coeff[5] = c3 - c7;
}

// src_diff: 16 bit, dynamic range [-32760, 32760]
// coeff: 19 bit
static void hadamard_highbd_col8_second_pass(const int16_t* src_diff, ptrdiff_t src_stride, int32_t* coeff) {
    int32_t b0 = src_diff[0 * src_stride] + src_diff[1 * src_stride];
    int32_t b1 = src_diff[0 * src_stride] - src_diff[1 * src_stride];
    int32_t b2 = src_diff[2 * src_stride] + src_diff[3 * src_stride];
    int32_t b3 = src_diff[2 * src_stride] - src_diff[3 * src_stride];
    int32_t b4 = src_diff[4 * src_stride] + src_diff[5 * src_stride];
    int32_t b5 = src_diff[4 * src_stride] - src_diff[5 * src_stride];
    int32_t b6 = src_diff[6 * src_stride] + src_diff[7 * src_stride];
    int32_t b7 = src_diff[6 * src_stride] - src_diff[7 * src_stride];

    int32_t c0 = b0 + b2;
    int32_t c1 = b1 + b3;
    int32_t c2 = b0 - b2;
    int32_t c3 = b1 - b3;
    int32_t c4 = b4 + b6;
    int32_t c5 = b5 + b7;
    int32_t c6 = b4 - b6;
    int32_t c7 = b5 - b7;

    coeff[0] = c0 + c4;
    coeff[7] = c1 + c5;
    coeff[3] = c2 + c6;
    coeff[4] = c3 + c7;
    coeff[2] = c0 - c4;
    coeff[6] = c1 - c5;
    coeff[1] = c2 - c6;
    coeff[5] = c3 - c7;
}

// The order of the output coeff of the hadamard is not important. For
// optimization purposes the final transpose may be skipped.
void svt_aom_highbd_hadamard_8x8_c(const int16_t* src_diff, ptrdiff_t src_stride, int32_t* coeff) {
    int      idx;
    int16_t  buffer[64];
    int32_t  buffer2[64];
    int16_t* tmp_buf = &buffer[0];
    for (idx = 0; idx < 8; ++idx) {
        // src_diff: 13 bit
        // buffer: 16 bit, dynamic range [-32760, 32760]
        hadamard_highbd_col8_first_pass(src_diff, src_stride, tmp_buf);
        tmp_buf += 8;
        ++src_diff;
    }

    tmp_buf = &buffer[0];
    for (idx = 0; idx < 8; ++idx) {
        // buffer: 16 bit
        // buffer2: 19 bit, dynamic range [-262080, 262080]
        hadamard_highbd_col8_second_pass(tmp_buf, 8, buffer2 + 8 * idx);
        ++tmp_buf;
    }

    for (idx = 0; idx < 64; ++idx) {
        coeff[idx] = (int32_t)buffer2[idx];
    }
}

void svt_aom_highbd_hadamard_16x16_c(const int16_t* src_diff, ptrdiff_t src_stride, int32_t* coeff) {
    int idx;
    for (idx = 0; idx < 4; ++idx) {
        // src_diff: 11 bit, dynamic range [-1023, 1023]
        const int16_t* src_ptr = src_diff + (idx >> 1) * 8 * src_stride + (idx & 0x01) * 8;
        svt_aom_highbd_hadamard_8x8_c(src_ptr, src_stride, coeff + idx * 64);
    }

    // coeff: 17 bit, dynamic range [-65472, 65472]
    for (idx = 0; idx < 64; ++idx) {
        const int32_t a0 = coeff[0];
        const int32_t a1 = coeff[64];
        const int32_t a2 = coeff[128];
        const int32_t a3 = coeff[192];

        const int32_t b0 = (a0 + a1) >> 1; // (a0 + a1): 18 bit, [-130944, 130944]
        const int32_t b1 = (a0 - a1) >> 1; // b0-b3: 17 bit, dynamic range
        const int32_t b2 = (a2 + a3) >> 1; // [-65472, 65472]
        const int32_t b3 = (a2 - a3) >> 1;

        coeff[0]   = b0 + b2; // 18 bit, [-130944, 130944]
        coeff[64]  = b1 + b3;
        coeff[128] = b0 - b2;
        coeff[192] = b1 - b3;

        ++coeff;
    }
}

void svt_aom_highbd_hadamard_32x32_c(const int16_t* src_diff, ptrdiff_t src_stride, int32_t* coeff) {
    int idx;
    for (idx = 0; idx < 4; ++idx) {
        // src_diff: 11 bit, dynamic range [-1023, 1023]
        const int16_t* src_ptr = src_diff + (idx >> 1) * 16 * src_stride + (idx & 0x01) * 16;
        svt_aom_highbd_hadamard_16x16_c(src_ptr, src_stride, coeff + idx * 256);
    }

    // coeff: 18 bit, dynamic range [-130944, 130944]
    for (idx = 0; idx < 256; ++idx) {
        const int32_t a0 = coeff[0];
        const int32_t a1 = coeff[256];
        const int32_t a2 = coeff[512];
        const int32_t a3 = coeff[768];

        const int32_t b0 = (a0 + a1) >> 2; // (a0 + a1): 19 bit, [-261888, 261888]
        const int32_t b1 = (a0 - a1) >> 2; // b0-b3: 17 bit, dynamic range
        const int32_t b2 = (a2 + a3) >> 2; // [-65472, 65472]
        const int32_t b3 = (a2 - a3) >> 2;

        coeff[0]   = b0 + b2; // 18 bit, [-130944, 130944]
        coeff[256] = b1 + b3;
        coeff[512] = b0 - b2;
        coeff[768] = b1 - b3;

        ++coeff;
    }
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
int svt_av1_highbd_hadamard_satd_4x4_c(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                       ptrdiff_t pred_stride) {
    DECLARE_ALIGNED(16, int16_t, diff[4 * 4]);
    DECLARE_ALIGNED(16, int32_t, coeff[4 * 4]);
    svt_aom_highbd_subtract_block(
        4, 4, diff, 4, (const uint8_t*)src, src_stride, (const uint8_t*)pred, pred_stride, 10);
    svt_aom_hadamard_4x4(diff, 4, coeff);
    return svt_aom_satd(coeff, 4 * 4);
}

int svt_av1_highbd_hadamard_satd_8x8_c(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                       ptrdiff_t pred_stride) {
    DECLARE_ALIGNED(16, int16_t, diff[8 * 8]);
    DECLARE_ALIGNED(16, int32_t, coeff[8 * 8]);
    svt_aom_highbd_subtract_block(
        8, 8, diff, 8, (const uint8_t*)src, src_stride, (const uint8_t*)pred, pred_stride, 10);
    svt_aom_highbd_hadamard_8x8(diff, 8, coeff);
    return svt_aom_satd(coeff, 8 * 8);
}

int svt_av1_highbd_hadamard_satd_16x16_c(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                         ptrdiff_t pred_stride) {
    DECLARE_ALIGNED(16, int16_t, diff[16 * 16]);
    DECLARE_ALIGNED(16, int32_t, coeff[16 * 16]);
    svt_aom_highbd_subtract_block(
        16, 16, diff, 16, (const uint8_t*)src, src_stride, (const uint8_t*)pred, pred_stride, 10);
    svt_aom_highbd_hadamard_16x16(diff, 16, coeff);
    return svt_aom_satd(coeff, 16 * 16);
}

int svt_av1_highbd_hadamard_satd_32x32_c(const uint16_t* src, ptrdiff_t src_stride, const uint16_t* pred,
                                         ptrdiff_t pred_stride) {
    DECLARE_ALIGNED(16, int16_t, diff[32 * 32]);
    DECLARE_ALIGNED(16, int32_t, coeff[32 * 32]);
    svt_aom_highbd_subtract_block(
        32, 32, diff, 32, (const uint8_t*)src, src_stride, (const uint8_t*)pred, pred_stride, 10);
    svt_aom_highbd_hadamard_32x32(diff, 32, coeff);
    return svt_aom_satd(coeff, 32 * 32);
}
#endif
