/*
* Copyright(c) 2026 Meta Platforms, Inc. and affiliates.
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

// Low-bit-depth (8-bit) forward transforms with int16 intermediates, native AVX512BW.
// A full 32-wide row/column is ONE zmm (32 int16), so fdct32 runs once per pass
// (vs the AVX2 int16 kernel's two ymm groups) and the whole 32x32 block stays
// register-resident across the transform + transpose. Butterflies use 512-bit
// madd_epi16 / mulhrs_epi16; the transpose uses the lane-local unpack cascade
// (epi16/32/64) followed by a 4x4 shuffle_i64x2 lane transpose.
//
// Bit-exact with the AVX2 int16 lowbd kernel (and thus the int32 transform for
// 8-bit): same Q13 cospi weights, same rounding (madd+rs13, mulhrs==vqrdmulh).

#include <immintrin.h>

#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "transforms.h"

#define TXFM_COS_BIT_MIN_512 10

// cospi constants in Q2.13, indexed [cos_bit - 10]; identical table to the AVX2 port.
static const int16_t fwd_cospi_arr_q13_avx512[4][128] = {
    {
        5792, 5792, -5792, -5792, 7568, 3136, -7568, -3136, 8032, 1600, -8032, -1600, 6808, 4552, -6808, -4552,
        8152, 800,  -8152, -800,  7840, 2376, -7840, -2376, 7224, 3864, -7224, -3864, 6336, 5200, -6336, -5200,
        8184, 400,  -8184, -400,  8104, 1200, -8104, -1200, 7944, 1992, -7944, -1992, 7712, 2760, -7712, -2760,
        7408, 3504, -7408, -3504, 7024, 4208, -7024, -4208, 6576, 4880, -6576, -4880, 6072, 5504, -6072, -5504,
        8192, 200,  -8192, -200,  8168, 600,  -8168, -600,  8128, 1000, -8128, -1000, 8072, 1400, -8072, -1400,
        7992, 1792, -7992, -1792, 7896, 2184, -7896, -2184, 7776, 2568, -7776, -2568, 7640, 2952, -7640, -2952,
        7488, 3320, -7488, -3320, 7320, 3680, -7320, -3680, 7128, 4040, -7128, -4040, 6920, 4384, -6920, -4384,
        6696, 4720, -6696, -4720, 6456, 5040, -6456, -5040, 6200, 5352, -6200, -5352, 5936, 5648, -5936, -5648,
    },
    {
        5792, 5792, -5792, -5792, 7568, 3136, -7568, -3136, 8036, 1600, -8036, -1600, 6812, 4552, -6812, -4552,
        8152, 804,  -8152, -804,  7840, 2380, -7840, -2380, 7224, 3860, -7224, -3860, 6332, 5196, -6332, -5196,
        8184, 400,  -8184, -400,  8104, 1204, -8104, -1204, 7948, 1992, -7948, -1992, 7712, 2760, -7712, -2760,
        7404, 3504, -7404, -3504, 7028, 4212, -7028, -4212, 6580, 4880, -6580, -4880, 6068, 5500, -6068, -5500,
        8188, 200,  -8188, -200,  8168, 604,  -8168, -604,  8132, 1004, -8132, -1004, 8072, 1400, -8072, -1400,
        7992, 1796, -7992, -1796, 7896, 2184, -7896, -2184, 7780, 2568, -7780, -2568, 7644, 2948, -7644, -2948,
        7488, 3320, -7488, -3320, 7316, 3684, -7316, -3684, 7128, 4036, -7128, -4036, 6920, 4384, -6920, -4384,
        6696, 4716, -6696, -4716, 6460, 5040, -6460, -5040, 6204, 5352, -6204, -5352, 5932, 5648, -5932, -5648,
    },
    {
        5792, 5792, -5792, -5792, 7568, 3134, -7568, -3134, 8034, 1598, -8034, -1598, 6812, 4552, -6812, -4552,
        8152, 802,  -8152, -802,  7840, 2378, -7840, -2378, 7224, 3862, -7224, -3862, 6332, 5196, -6332, -5196,
        8182, 402,  -8182, -402,  8104, 1202, -8104, -1202, 7946, 1990, -7946, -1990, 7714, 2760, -7714, -2760,
        7406, 3502, -7406, -3502, 7026, 4212, -7026, -4212, 6580, 4880, -6580, -4880, 6070, 5502, -6070, -5502,
        8190, 202,  -8190, -202,  8170, 602,  -8170, -602,  8130, 1002, -8130, -1002, 8072, 1400, -8072, -1400,
        7992, 1794, -7992, -1794, 7896, 2184, -7896, -2184, 7778, 2570, -7778, -2570, 7644, 2948, -7644, -2948,
        7490, 3320, -7490, -3320, 7318, 3684, -7318, -3684, 7128, 4038, -7128, -4038, 6922, 4382, -6922, -4382,
        6698, 4718, -6698, -4718, 6458, 5040, -6458, -5040, 6204, 5350, -6204, -5350, 5934, 5648, -5934, -5648,
    },
    {
        5793, 5793, -5793, -5793, 7568, 3135, -7568, -3135, 8035, 1598, -8035, -1598, 6811, 4551, -6811, -4551,
        8153, 803,  -8153, -803,  7839, 2378, -7839, -2378, 7225, 3862, -7225, -3862, 6333, 5197, -6333, -5197,
        8182, 402,  -8182, -402,  8103, 1202, -8103, -1202, 7946, 1990, -7946, -1990, 7713, 2760, -7713, -2760,
        7405, 3503, -7405, -3503, 7027, 4212, -7027, -4212, 6580, 4880, -6580, -4880, 6070, 5501, -6070, -5501,
        8190, 201,  -8190, -201,  8170, 603,  -8170, -603,  8130, 1003, -8130, -1003, 8071, 1401, -8071, -1401,
        7993, 1795, -7993, -1795, 7895, 2185, -7895, -2185, 7779, 2570, -7779, -2570, 7643, 2948, -7643, -2948,
        7489, 3320, -7489, -3320, 7317, 3683, -7317, -3683, 7128, 4038, -7128, -4038, 6921, 4383, -6921, -4383,
        6698, 4717, -6698, -4717, 6458, 5040, -6458, -5040, 6203, 5351, -6203, -5351, 5933, 5649, -5933, -5649,
    }};

static INLINE const int16_t* fwd_cospi_q13_avx512(int cos_bit) {
    return fwd_cospi_arr_q13_avx512[cos_bit - TXFM_COS_BIT_MIN_512];
}

static INLINE __m512i wpair_512(const int16_t* w, int a, int b) {
    return _mm512_set1_epi32((int32_t)(((uint32_t)(uint16_t)w[b] << 16) | (uint16_t)w[a]));
}

// out0 = rs13(in0*w[l0] + in1*w[l1]); out1 = rs13(in0*w[l2] + in1*w[l3]).
// Element-wise across all 32 columns; unpack/madd/packs stay lane-local so each
// column keeps its position.
static INLINE void butterfly_x32(const __m512i in0, const __m512i in1, const __m512i w01, const __m512i w23,
                                 __m512i* out0, __m512i* out1) {
    const __m512i rnd = _mm512_set1_epi32(1 << 12);
    const __m512i lo  = _mm512_unpacklo_epi16(in0, in1);
    const __m512i hi  = _mm512_unpackhi_epi16(in0, in1);
    const __m512i u0  = _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(lo, w01), rnd), 13);
    const __m512i u1  = _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(hi, w01), rnd), 13);
    const __m512i v0  = _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(lo, w23), rnd), 13);
    const __m512i v1  = _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(hi, w23), rnd), 13);
    *out0             = _mm512_packs_epi32(u0, u1);
    *out1             = _mm512_packs_epi32(v0, v1);
}

// out0 = rdmulh(in0+in1, w32), out1 = rdmulh(in0-in1, w32). mulhrs == vqrdmulh.
static INLINE void btf_cospi32_0112_x32(const __m512i w32, const __m512i in0, const __m512i in1, __m512i* out0,
                                        __m512i* out1) {
    *out0 = _mm512_mulhrs_epi16(_mm512_adds_epi16(in0, in1), w32);
    *out1 = _mm512_mulhrs_epi16(_mm512_subs_epi16(in0, in1), w32);
}

static INLINE void butterfly_dct_pre_x32(const __m512i* in, __m512i* out, int n) {
    for (int i = 0; i < n / 2; ++i) {
        out[i] = _mm512_adds_epi16(in[i], in[n - i - 1]);
    }
    for (int i = 0; i < n / 2; ++i) {
        out[n / 2 + i] = _mm512_subs_epi16(in[n / 2 - i - 1], in[n / 2 + i]);
    }
}

static INLINE void butterfly_dct_post_x32(const __m512i* in0, const __m512i* in1, __m512i* out, int n) {
    for (int i = 0; i < n / 4; ++i) {
        out[i] = _mm512_adds_epi16(in0[i], in1[n / 2 - i - 1]);
    }
    for (int i = 0; i < n / 4; ++i) {
        out[n / 4 + i] = _mm512_subs_epi16(in0[n / 4 - i - 1], in1[n / 4 + i]);
    }
    for (int i = 0; i < n / 4; ++i) {
        out[n / 2 + i] = _mm512_subs_epi16(in0[n - i - 1], in1[n / 2 + i]);
    }
    for (int i = 0; i < n / 4; ++i) {
        out[(3 * n) / 4 + i] = _mm512_adds_epi16(in0[(3 * n) / 4 + i], in1[(3 * n) / 4 - i - 1]);
    }
}

// 32-point forward DCT, int16, 32 columns per zmm. Mirrors fdct32_x16_avx2.
static INLINE void fdct32_x32_avx512(const __m512i* input, __m512i* output, int cos_bit) {
    const int16_t* cospi = fwd_cospi_q13_avx512(cos_bit);
    const __m512i  w32   = _mm512_set1_epi16((short)(cospi[0] * 4));
    const int16_t *c32 = cospi + 4 * 0, *c16 = cospi + 4 * 1;
    const int16_t *c8 = cospi + 4 * 2, *c24 = cospi + 4 * 3;
    const int16_t *c4 = cospi + 4 * 4, *c12 = cospi + 4 * 5;
    const int16_t *c20 = cospi + 4 * 6, *c28 = cospi + 4 * 7;
    const int16_t *c2 = cospi + 4 * 8, *c6 = cospi + 4 * 9;
    const int16_t *c10 = cospi + 4 * 10, *c14 = cospi + 4 * 11;
    const int16_t *c18 = cospi + 4 * 12, *c22 = cospi + 4 * 13;
    const int16_t *c26 = cospi + 4 * 14, *c30 = cospi + 4 * 15;

    __m512i x1[32];
    butterfly_dct_pre_x32(input, x1, 32);
    __m512i x2[32];
    butterfly_dct_pre_x32(x1, x2, 16);
    btf_cospi32_0112_x32(w32, x1[27], x1[20], &x2[27], &x2[20]);
    btf_cospi32_0112_x32(w32, x1[26], x1[21], &x2[26], &x2[21]);
    btf_cospi32_0112_x32(w32, x1[25], x1[22], &x2[25], &x2[22]);
    btf_cospi32_0112_x32(w32, x1[24], x1[23], &x2[24], &x2[23]);

    __m512i x3[32];
    butterfly_dct_pre_x32(x2, x3, 8);
    btf_cospi32_0112_x32(w32, x2[13], x2[10], &x3[13], &x3[10]);
    btf_cospi32_0112_x32(w32, x2[12], x2[11], &x3[12], &x3[11]);
    butterfly_dct_post_x32(x1 + 16, x2 + 16, x3 + 16, 16);

    __m512i x4[32];
    butterfly_dct_pre_x32(x3, x4, 4);
    btf_cospi32_0112_x32(w32, x3[6], x3[5], &x4[6], &x4[5]);
    butterfly_dct_post_x32(x2 + 8, x3 + 8, x4 + 8, 8);
    butterfly_x32(x3[29], x3[18], wpair_512(c16, 0, 1), wpair_512(c16, 1, 2), &x4[29], &x4[18]);
    butterfly_x32(x3[28], x3[19], wpair_512(c16, 0, 1), wpair_512(c16, 1, 2), &x4[28], &x4[19]);
    butterfly_x32(x3[27], x3[20], wpair_512(c16, 1, 2), wpair_512(c16, 2, 3), &x4[27], &x4[20]);
    butterfly_x32(x3[26], x3[21], wpair_512(c16, 1, 2), wpair_512(c16, 2, 3), &x4[26], &x4[21]);

    __m512i x5[32];
    butterfly_x32(x4[0], x4[1], wpair_512(c32, 0, 1), wpair_512(c32, 1, 2), &output[0], &output[16]);
    butterfly_x32(x4[3], x4[2], wpair_512(c16, 0, 1), wpair_512(c16, 1, 2), &output[8], &output[24]);
    butterfly_dct_post_x32(x3 + 4, x4 + 4, x5 + 4, 4);
    butterfly_x32(x4[14], x4[9], wpair_512(c16, 0, 1), wpair_512(c16, 1, 2), &x5[14], &x5[9]);
    butterfly_x32(x4[13], x4[10], wpair_512(c16, 1, 2), wpair_512(c16, 2, 3), &x5[13], &x5[10]);
    butterfly_dct_post_x32(x3 + 16, x4 + 16, x5 + 16, 8);
    butterfly_dct_post_x32(x3 + 24, x4 + 24, x5 + 24, 8);

    __m512i x6[32];
    butterfly_x32(x5[7], x5[4], wpair_512(c8, 0, 1), wpair_512(c8, 1, 2), &output[4], &output[28]);
    butterfly_x32(x5[6], x5[5], wpair_512(c24, 1, 0), wpair_512(c24, 0, 3), &output[20], &output[12]);
    butterfly_dct_post_x32(x4 + 8, x5 + 8, x6 + 8, 4);
    butterfly_dct_post_x32(x4 + 12, x5 + 12, x6 + 12, 4);
    butterfly_x32(x5[30], x5[17], wpair_512(c8, 0, 1), wpair_512(c8, 1, 2), &x6[30], &x6[17]);
    butterfly_x32(x5[29], x5[18], wpair_512(c8, 1, 2), wpair_512(c8, 2, 3), &x6[29], &x6[18]);
    butterfly_x32(x5[26], x5[21], wpair_512(c24, 1, 0), wpair_512(c24, 0, 3), &x6[26], &x6[21]);
    butterfly_x32(x5[25], x5[22], wpair_512(c24, 0, 3), wpair_512(c24, 3, 2), &x6[25], &x6[22]);

    __m512i x7[32];
    butterfly_x32(x6[15], x6[8], wpair_512(c4, 0, 1), wpair_512(c4, 1, 2), &output[2], &output[30]);
    butterfly_x32(x6[14], x6[9], wpair_512(c28, 1, 0), wpair_512(c28, 0, 3), &output[18], &output[14]);
    butterfly_x32(x6[13], x6[10], wpair_512(c20, 0, 1), wpair_512(c20, 1, 2), &output[10], &output[22]);
    butterfly_x32(x6[12], x6[11], wpair_512(c12, 1, 0), wpair_512(c12, 0, 3), &output[26], &output[6]);
    butterfly_dct_post_x32(x5 + 16, x6 + 16, x7 + 16, 4);
    butterfly_dct_post_x32(x5 + 20, x6 + 20, x7 + 20, 4);
    butterfly_dct_post_x32(x5 + 24, x6 + 24, x7 + 24, 4);
    butterfly_dct_post_x32(x5 + 28, x6 + 28, x7 + 28, 4);

    butterfly_x32(x7[31], x7[16], wpair_512(c2, 0, 1), wpair_512(c2, 1, 2), &output[1], &output[31]);
    butterfly_x32(x7[30], x7[17], wpair_512(c30, 1, 0), wpair_512(c30, 0, 3), &output[17], &output[15]);
    butterfly_x32(x7[29], x7[18], wpair_512(c18, 0, 1), wpair_512(c18, 1, 2), &output[9], &output[23]);
    butterfly_x32(x7[28], x7[19], wpair_512(c14, 1, 0), wpair_512(c14, 0, 3), &output[25], &output[7]);
    butterfly_x32(x7[27], x7[20], wpair_512(c10, 0, 1), wpair_512(c10, 1, 2), &output[5], &output[27]);
    butterfly_x32(x7[26], x7[21], wpair_512(c22, 1, 0), wpair_512(c22, 0, 3), &output[21], &output[11]);
    butterfly_x32(x7[25], x7[22], wpair_512(c26, 0, 1), wpair_512(c26, 1, 2), &output[13], &output[19]);
    butterfly_x32(x7[24], x7[23], wpair_512(c6, 1, 0), wpair_512(c6, 0, 3), &output[29], &output[3]);
}

// 32x32 int16 transpose, register-resident. Stages 1-3 (unpack epi16/32/64)
// transpose the 8x8 block inside each 128-bit lane; stage 4 is a 4x4 transpose
// of the 128-bit lanes (shuffle_i64x2) that swaps block-row/block-col.
static INLINE void transpose_32x32_epi16_avx512(const __m512i* in, __m512i* out) {
    __m512i s1[32], s2[32], s3[32];
    for (int i = 0; i < 16; i++) {
        s1[2 * i + 0] = _mm512_unpacklo_epi16(in[2 * i], in[2 * i + 1]);
        s1[2 * i + 1] = _mm512_unpackhi_epi16(in[2 * i], in[2 * i + 1]);
    }
    for (int j = 0; j < 8; j++) {
        s2[4 * j + 0] = _mm512_unpacklo_epi32(s1[4 * j + 0], s1[4 * j + 2]);
        s2[4 * j + 1] = _mm512_unpackhi_epi32(s1[4 * j + 0], s1[4 * j + 2]);
        s2[4 * j + 2] = _mm512_unpacklo_epi32(s1[4 * j + 1], s1[4 * j + 3]);
        s2[4 * j + 3] = _mm512_unpackhi_epi32(s1[4 * j + 1], s1[4 * j + 3]);
    }
    for (int g = 0; g < 4; g++) {
        s3[8 * g + 0] = _mm512_unpacklo_epi64(s2[8 * g + 0], s2[8 * g + 4]);
        s3[8 * g + 1] = _mm512_unpackhi_epi64(s2[8 * g + 0], s2[8 * g + 4]);
        s3[8 * g + 2] = _mm512_unpacklo_epi64(s2[8 * g + 1], s2[8 * g + 5]);
        s3[8 * g + 3] = _mm512_unpackhi_epi64(s2[8 * g + 1], s2[8 * g + 5]);
        s3[8 * g + 4] = _mm512_unpacklo_epi64(s2[8 * g + 2], s2[8 * g + 6]);
        s3[8 * g + 5] = _mm512_unpackhi_epi64(s2[8 * g + 2], s2[8 * g + 6]);
        s3[8 * g + 6] = _mm512_unpacklo_epi64(s2[8 * g + 3], s2[8 * g + 7]);
        s3[8 * g + 7] = _mm512_unpackhi_epi64(s2[8 * g + 3], s2[8 * g + 7]);
    }
    // 4x4 128-bit lane transpose across the four block-rows (regs k, 8+k, 16+k, 24+k).
    for (int k = 0; k < 8; k++) {
        const __m512i a = s3[k], b = s3[8 + k], c = s3[16 + k], d = s3[24 + k];
        const __m512i t0 = _mm512_shuffle_i64x2(a, b, 0x88);
        const __m512i t1 = _mm512_shuffle_i64x2(a, b, 0xdd);
        const __m512i t2 = _mm512_shuffle_i64x2(c, d, 0x88);
        const __m512i t3 = _mm512_shuffle_i64x2(c, d, 0xdd);
        out[k + 0]       = _mm512_shuffle_i64x2(t0, t2, 0x88);
        out[k + 16]      = _mm512_shuffle_i64x2(t0, t2, 0xdd);
        out[k + 8]       = _mm512_shuffle_i64x2(t1, t3, 0x88);
        out[k + 24]      = _mm512_shuffle_i64x2(t1, t3, 0xdd);
    }
}

// 32x32 DCT_DCT, 8-bit, int16. Native AVX512: one zmm per 32-wide row.
void svt_lbd_fwd_txfm2d_32x32_dct_avx512(int16_t* input, int32_t* output, uint32_t stride) {
    __m512i buf[32], cb[32];
    for (int r = 0; r < 32; r++) {
        buf[r] = _mm512_slli_epi16(_mm512_loadu_si512((const void*)(input + r * stride)), 2);
    }
    fdct32_x32_avx512(buf, cb, 12);
    const __m512i rnd = _mm512_set1_epi16(8);
    for (int k = 0; k < 32; k++) {
        cb[k] = _mm512_srai_epi16(_mm512_add_epi16(cb[k], rnd), 4);
    }

    transpose_32x32_epi16_avx512(cb, buf);
    fdct32_x32_avx512(buf, cb, 12);
    transpose_32x32_epi16_avx512(cb, buf);

    for (int r = 0; r < 32; r++) {
        const __m256i lo = _mm512_castsi512_si256(buf[r]);
        const __m256i hi = _mm512_extracti64x4_epi64(buf[r], 1);
        _mm512_storeu_si512((void*)(output + r * 32 + 0), _mm512_cvtepi16_epi32(lo));
        _mm512_storeu_si512((void*)(output + r * 32 + 16), _mm512_cvtepi16_epi32(hi));
    }
}

// ---------------------------------------------------------------------------
// 64x64 int16 forward DCT (native AVX512BW), port of av1_fdct64_new_avx512.
//
// Bit-exact by construction: every btf_32_type0/1 maps to madd_epi16 with the
// full cospi weights (<=8192, fit int16) and >>cos_bit -- identical int32
// arithmetic. The cospi32 butterflies instead use mulhrs_epi16 (doubling
// Q15 multiply): mulhrs(x, cospi32<<(15-cos_bit)) == (x*cospi32 + rnd)>>cos_bit
// exactly, and stays int16 (no unpack/widen/pack). Pre-stages use saturating
// adds_epi16/subs_epi16. fwd_shift_64x64={0,-2,-2} keeps intermediates in int16.

// out0 = rs(in0*w0 + in1*w1, bit); out1 = rs(in0*w1 - in1*w0, bit)   (btf type0)
static INLINE void btf16_0_x32(int16_t w0, int16_t w1, __m512i in0, __m512i in1, __m512i* o0, __m512i* o1,
                               const __m512i rnd, int bit) {
    const __m512i wp0 = _mm512_set1_epi32((int32_t)(((uint32_t)(uint16_t)w1 << 16) | (uint16_t)w0)); // (w0, w1)
    const __m512i wp1 = _mm512_set1_epi32((int32_t)(((uint32_t)(uint16_t)(-w0) << 16) | (uint16_t)w1)); // (w1, -w0)
    const __m512i lo  = _mm512_unpacklo_epi16(in0, in1);
    const __m512i hi  = _mm512_unpackhi_epi16(in0, in1);
    *o0               = _mm512_packs_epi32(_mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(lo, wp0), rnd), bit),
                             _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(hi, wp0), rnd), bit));
    *o1               = _mm512_packs_epi32(_mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(lo, wp1), rnd), bit),
                             _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(hi, wp1), rnd), bit));
}

// btf_32_type1(w0,w1,in0,in1) == btf_32_type0(w1,w0,in1,in0)
static INLINE void btf16_1_x32(int16_t w0, int16_t w1, __m512i in0, __m512i in1, __m512i* o0, __m512i* o1,
                               const __m512i rnd, int bit) {
    btf16_0_x32(w1, w0, in1, in0, o0, o1, rnd, bit);
}

// btf_32_type0(cospi_m32, cospi_p32, in0, in1): out0=(in1-in0)*c32>>bit, out1=(in0+in1)*c32>>bit
static INLINE void btf_c32_x32(__m512i w32, __m512i in0, __m512i in1, __m512i* o0, __m512i* o1) {
    *o0 = _mm512_mulhrs_epi16(_mm512_subs_epi16(in1, in0), w32);
    *o1 = _mm512_mulhrs_epi16(_mm512_adds_epi16(in0, in1), w32);
}

#define A16(a, b) _mm512_adds_epi16((a), (b))
#define S16(a, b) _mm512_subs_epi16((a), (b))

// 64-point forward DCT, int16, 32 columns per zmm. Faithful port of the int32
// av1_fdct64_new_avx512 stage network. `output` holds all 64 coeff rows.
static void fdct64_x32_avx512(const __m512i* in, __m512i* output, int cos_bit) {
    const int32_t* cospi = cospi_arr(cos_bit);
    const __m512i  rnd   = _mm512_set1_epi32(1 << (cos_bit - 1));
    const __m512i  w32   = _mm512_set1_epi16((int16_t)(cospi[32] << (15 - cos_bit)));
    const int16_t  c32 = (int16_t)cospi[32], c16 = (int16_t)cospi[16], c48 = (int16_t)cospi[48];
    const int16_t  c8 = (int16_t)cospi[8], c56 = (int16_t)cospi[56], c40 = (int16_t)cospi[40], c24 = (int16_t)cospi[24];
    const int16_t  c4 = (int16_t)cospi[4], c60 = (int16_t)cospi[60], c36 = (int16_t)cospi[36], c28 = (int16_t)cospi[28];
    const int16_t  c20 = (int16_t)cospi[20], c44 = (int16_t)cospi[44], c52 = (int16_t)cospi[52],
                  c12 = (int16_t)cospi[12];
    const int16_t c2 = (int16_t)cospi[2], c62 = (int16_t)cospi[62], c34 = (int16_t)cospi[34], c30 = (int16_t)cospi[30];
    const int16_t c46 = (int16_t)cospi[46], c18 = (int16_t)cospi[18], c50 = (int16_t)cospi[50],
                  c14 = (int16_t)cospi[14];
    const int16_t c54 = (int16_t)cospi[54], c10 = (int16_t)cospi[10], c42 = (int16_t)cospi[42],
                  c22 = (int16_t)cospi[22];
    const int16_t c38 = (int16_t)cospi[38], c26 = (int16_t)cospi[26], c58 = (int16_t)cospi[58], c6 = (int16_t)cospi[6];
    const int16_t c63 = (int16_t)cospi[63], c1 = (int16_t)cospi[1], c31 = (int16_t)cospi[31], c33 = (int16_t)cospi[33];
    const int16_t c47 = (int16_t)cospi[47], c17 = (int16_t)cospi[17], c49 = (int16_t)cospi[49],
                  c15 = (int16_t)cospi[15];
    const int16_t c55 = (int16_t)cospi[55], c9 = (int16_t)cospi[9], c41 = (int16_t)cospi[41], c23 = (int16_t)cospi[23];
    const int16_t c39 = (int16_t)cospi[39], c25 = (int16_t)cospi[25], c57 = (int16_t)cospi[57], c7 = (int16_t)cospi[7];
    const int16_t c59 = (int16_t)cospi[59], c5 = (int16_t)cospi[5], c37 = (int16_t)cospi[37], c27 = (int16_t)cospi[27];
    const int16_t c43 = (int16_t)cospi[43], c21 = (int16_t)cospi[21], c53 = (int16_t)cospi[53],
                  c11 = (int16_t)cospi[11];
    const int16_t c51 = (int16_t)cospi[51], c13 = (int16_t)cospi[13], c19 = (int16_t)cospi[19],
                  c45 = (int16_t)cospi[45];
    const int16_t c35 = (int16_t)cospi[35], c29 = (int16_t)cospi[29], c3 = (int16_t)cospi[3], c61 = (int16_t)cospi[61];

    __m512i x1[64];
    for (int i = 0; i < 32; i++) {
        x1[i]      = A16(in[i], in[63 - i]);
        x1[63 - i] = S16(in[i], in[63 - i]);
    }
    // stage 2
    __m512i x2[54];
    for (int i = 0; i < 16; i++) {
        x2[i]      = A16(x1[i], x1[31 - i]);
        x2[31 - i] = S16(x1[i], x1[31 - i]);
    }
    for (int i = 0; i < 8; i++) {
        btf_c32_x32(w32, x1[40 + i], x1[55 - i], &x2[32 + i], &x2[47 - i]);
    }
    // stage 3
    __m512i x3[56];
    for (int i = 0; i < 8; i++) {
        x3[i]      = A16(x2[i], x2[15 - i]);
        x3[15 - i] = S16(x2[i], x2[15 - i]);
    }
    for (int i = 0; i < 4; i++) {
        btf_c32_x32(w32, x2[20 + i], x2[27 - i], &x3[16 + i], &x3[23 - i]);
    }
    for (int i = 0; i < 8; i++) {
        x3[32 + i] = A16(x1[32 + i], x2[39 - i]);
        x3[47 - i] = S16(x1[32 + i], x2[39 - i]);
    }
    for (int i = 0; i < 8; i++) {
        x3[24 + i] = A16(x1[63 - i], x2[40 + i]);
        x3[48 + i] = S16(x1[63 - i], x2[40 + i]);
    }
    // stage 4
    for (int i = 0; i < 4; i++) {
        x1[i]     = A16(x3[i], x3[7 - i]);
        x1[7 - i] = S16(x3[i], x3[7 - i]);
    }
    btf_c32_x32(w32, x3[10], x3[13], &x1[8], &x1[11]);
    btf_c32_x32(w32, x3[11], x3[12], &x1[9], &x1[10]);
    for (int i = 0; i < 4; i++) {
        x1[12 + i] = A16(x2[16 + i], x3[19 - i]);
        x1[19 - i] = S16(x2[16 + i], x3[19 - i]);
    }
    for (int i = 0; i < 4; i++) {
        x1[20 + i] = S16(x2[31 - i], x3[20 + i]);
        x1[27 - i] = A16(x2[31 - i], x3[20 + i]);
    }
    for (int i = 0; i < 4; i++) {
        btf16_0_x32(-c16, c48, x3[36 + i], x3[28 + i], &x1[28 + i], &x1[43 - i], rnd, cos_bit);
    }
    for (int i = 0; i < 4; i++) {
        btf16_0_x32(-c48, -c16, x3[40 + i], x3[55 - i], &x1[32 + i], &x1[39 - i], rnd, cos_bit);
    }
    // stage 5
    for (int i = 0; i < 2; i++) {
        x2[i]     = A16(x1[i], x1[3 - i]);
        x2[3 - i] = S16(x1[i], x1[3 - i]);
    }
    btf_c32_x32(w32, x1[5], x1[6], &x2[4], &x2[5]);
    for (int i = 0; i < 2; i++) {
        x2[6 + i] = A16(x3[8 + i], x1[9 - i]);
        x2[9 - i] = S16(x3[8 + i], x1[9 - i]);
    }
    for (int i = 0; i < 2; i++) {
        x2[10 + i] = S16(x3[15 - i], x1[10 + i]);
        x2[13 - i] = A16(x3[15 - i], x1[10 + i]);
    }
    btf16_0_x32(-c16, c48, x1[14], x1[25], &x2[14], &x2[21], rnd, cos_bit);
    btf16_0_x32(-c16, c48, x1[15], x1[24], &x2[15], &x2[20], rnd, cos_bit);
    btf16_0_x32(-c48, -c16, x1[16], x1[23], &x2[16], &x2[19], rnd, cos_bit);
    btf16_0_x32(-c48, -c16, x1[17], x1[22], &x2[17], &x2[18], rnd, cos_bit);
    for (int i = 0; i < 4; i++) {
        x2[22 + i] = A16(x3[32 + i], x1[31 - i]);
        x2[29 - i] = S16(x3[32 + i], x1[31 - i]);
    }
    for (int i = 0; i < 4; i++) {
        x2[30 + i] = S16(x3[47 - i], x1[32 + i]);
        x2[37 - i] = A16(x3[47 - i], x1[32 + i]);
    }
    for (int i = 0; i < 4; i++) {
        x2[38 + i] = A16(x3[48 + i], x1[39 - i]);
        x2[45 - i] = S16(x3[48 + i], x1[39 - i]);
    }
    for (int i = 0; i < 4; i++) {
        x2[46 + i] = S16(x3[24 + i], x1[40 + i]);
        x2[53 - i] = A16(x3[24 + i], x1[40 + i]);
    }
    // stage 6
    btf16_0_x32(c32, c32, x2[0], x2[1], &output[0], &output[32], rnd, cos_bit);
    btf16_1_x32(c48, c16, x2[2], x2[3], &output[16], &output[48], rnd, cos_bit);
    x3[0] = A16(x1[4], x2[4]);
    x3[1] = S16(x1[4], x2[4]);
    x3[2] = S16(x1[7], x2[5]);
    x3[3] = A16(x1[7], x2[5]);
    btf16_0_x32(-c16, c48, x2[7], x2[12], &x3[4], &x3[7], rnd, cos_bit);
    btf16_0_x32(-c48, -c16, x2[8], x2[11], &x3[5], &x3[6], rnd, cos_bit);
    x3[8]  = A16(x1[12], x2[15]);
    x3[11] = S16(x1[12], x2[15]);
    x3[9]  = A16(x1[13], x2[14]);
    x3[10] = S16(x1[13], x2[14]);
    x3[12] = S16(x1[19], x2[16]);
    x3[15] = A16(x1[19], x2[16]);
    x3[13] = S16(x1[18], x2[17]);
    x3[14] = A16(x1[18], x2[17]);
    x3[16] = A16(x1[20], x2[19]);
    x3[19] = S16(x1[20], x2[19]);
    x3[17] = A16(x1[21], x2[18]);
    x3[18] = S16(x1[21], x2[18]);
    x3[20] = S16(x1[27], x2[20]);
    x3[23] = A16(x1[27], x2[20]);
    x3[21] = S16(x1[26], x2[21]);
    x3[22] = A16(x1[26], x2[21]);
    btf16_0_x32(-c8, c56, x2[24], x2[51], &x3[24], &x3[39], rnd, cos_bit);
    btf16_0_x32(-c8, c56, x2[25], x2[50], &x3[25], &x3[38], rnd, cos_bit);
    btf16_0_x32(-c56, -c8, x2[26], x2[49], &x3[26], &x3[37], rnd, cos_bit);
    btf16_0_x32(-c56, -c8, x2[27], x2[48], &x3[27], &x3[36], rnd, cos_bit);
    btf16_0_x32(-c40, c24, x2[32], x2[43], &x3[28], &x3[35], rnd, cos_bit);
    btf16_0_x32(-c40, c24, x2[33], x2[42], &x3[29], &x3[34], rnd, cos_bit);
    btf16_0_x32(-c24, -c40, x2[34], x2[41], &x3[30], &x3[33], rnd, cos_bit);
    btf16_0_x32(-c24, -c40, x2[35], x2[40], &x3[31], &x3[32], rnd, cos_bit);
    // stage 7
    btf16_1_x32(c56, c8, x3[0], x3[3], &output[8], &output[56], rnd, cos_bit);
    btf16_1_x32(c24, c40, x3[1], x3[2], &output[40], &output[24], rnd, cos_bit);
    x1[0] = A16(x2[6], x3[4]);
    x1[1] = S16(x2[6], x3[4]);
    x1[2] = S16(x2[9], x3[5]);
    x1[3] = A16(x2[9], x3[5]);
    x1[4] = A16(x2[10], x3[6]);
    x1[5] = S16(x2[10], x3[6]);
    x1[6] = S16(x2[13], x3[7]);
    x1[7] = A16(x2[13], x3[7]);
    btf16_0_x32(-c8, c56, x3[9], x3[22], &x1[8], &x1[15], rnd, cos_bit);
    btf16_0_x32(-c56, -c8, x3[10], x3[21], &x1[9], &x1[14], rnd, cos_bit);
    btf16_0_x32(-c40, c24, x3[13], x3[18], &x1[10], &x1[13], rnd, cos_bit);
    btf16_0_x32(-c24, -c40, x3[14], x3[17], &x1[11], &x1[12], rnd, cos_bit);
    x1[16] = A16(x2[22], x3[25]);
    x1[17] = S16(x2[22], x3[25]);
    x1[19] = A16(x2[23], x3[24]);
    x1[20] = S16(x2[23], x3[24]);
    x1[18] = S16(x2[29], x3[26]);
    x1[21] = A16(x2[29], x3[26]);
    x1[22] = S16(x2[28], x3[27]);
    x1[23] = A16(x2[28], x3[27]);
    x1[24] = A16(x2[30], x3[29]);
    x1[25] = S16(x2[30], x3[29]);
    x1[26] = A16(x2[31], x3[28]);
    x1[27] = S16(x2[31], x3[28]);
    x1[28] = S16(x2[37], x3[30]);
    x1[29] = A16(x2[37], x3[30]);
    x1[30] = S16(x2[36], x3[31]);
    x1[31] = A16(x2[36], x3[31]);
    x1[32] = A16(x2[38], x3[33]);
    x1[33] = S16(x2[38], x3[33]);
    x1[34] = A16(x2[39], x3[32]);
    x1[35] = S16(x2[39], x3[32]);
    x1[36] = S16(x2[45], x3[34]);
    x1[37] = A16(x2[45], x3[34]);
    x1[38] = S16(x2[44], x3[35]);
    x1[39] = A16(x2[44], x3[35]);
    x1[40] = A16(x2[46], x3[37]);
    x1[41] = S16(x2[46], x3[37]);
    x1[42] = A16(x2[47], x3[36]);
    x1[43] = S16(x2[47], x3[36]);
    x1[44] = S16(x2[53], x3[38]);
    x1[45] = A16(x2[53], x3[38]);
    x1[46] = S16(x2[52], x3[39]);
    x1[47] = A16(x2[52], x3[39]);
    // stage 8
    btf16_1_x32(c60, c4, x1[0], x1[7], &output[4], &output[60], rnd, cos_bit);
    btf16_1_x32(c28, c36, x1[1], x1[6], &output[36], &output[28], rnd, cos_bit);
    btf16_1_x32(c44, c20, x1[2], x1[5], &output[20], &output[44], rnd, cos_bit);
    btf16_1_x32(c12, c52, x1[3], x1[4], &output[52], &output[12], rnd, cos_bit);
    x2[0]  = A16(x3[8], x1[8]);
    x2[1]  = S16(x3[8], x1[8]);
    x2[2]  = S16(x3[11], x1[9]);
    x2[3]  = A16(x3[11], x1[9]);
    x2[4]  = A16(x3[12], x1[10]);
    x2[5]  = S16(x3[12], x1[10]);
    x2[6]  = S16(x3[15], x1[11]);
    x2[7]  = A16(x3[15], x1[11]);
    x2[8]  = A16(x3[16], x1[12]);
    x2[9]  = S16(x3[16], x1[12]);
    x2[10] = S16(x3[19], x1[13]);
    x2[11] = A16(x3[19], x1[13]);
    x2[12] = A16(x3[20], x1[14]);
    x2[13] = S16(x3[20], x1[14]);
    x2[14] = S16(x3[23], x1[15]);
    x2[15] = A16(x3[23], x1[15]);
    btf16_0_x32(-c4, c60, x1[19], x1[47], &x2[16], &x2[31], rnd, cos_bit);
    btf16_0_x32(-c60, -c4, x1[20], x1[46], &x2[17], &x2[30], rnd, cos_bit);
    btf16_0_x32(-c36, c28, x1[22], x1[43], &x2[18], &x2[29], rnd, cos_bit);
    btf16_0_x32(-c28, -c36, x1[23], x1[42], &x2[19], &x2[28], rnd, cos_bit);
    btf16_0_x32(-c20, c44, x1[26], x1[39], &x2[20], &x2[27], rnd, cos_bit);
    btf16_0_x32(-c44, -c20, x1[27], x1[38], &x2[21], &x2[26], rnd, cos_bit);
    btf16_0_x32(-c52, c12, x1[30], x1[35], &x2[22], &x2[25], rnd, cos_bit);
    btf16_0_x32(-c12, -c52, x1[31], x1[34], &x2[23], &x2[24], rnd, cos_bit);
    // stage 9
    btf16_1_x32(c62, c2, x2[0], x2[15], &output[2], &output[62], rnd, cos_bit);
    btf16_1_x32(c30, c34, x2[1], x2[14], &output[34], &output[30], rnd, cos_bit);
    btf16_1_x32(c46, c18, x2[2], x2[13], &output[18], &output[46], rnd, cos_bit);
    btf16_1_x32(c14, c50, x2[3], x2[12], &output[50], &output[14], rnd, cos_bit);
    btf16_1_x32(c54, c10, x2[4], x2[11], &output[10], &output[54], rnd, cos_bit);
    btf16_1_x32(c22, c42, x2[5], x2[10], &output[42], &output[22], rnd, cos_bit);
    btf16_1_x32(c38, c26, x2[6], x2[9], &output[26], &output[38], rnd, cos_bit);
    btf16_1_x32(c6, c58, x2[7], x2[8], &output[58], &output[6], rnd, cos_bit);
    x3[0]  = A16(x1[16], x2[16]);
    x3[1]  = S16(x1[16], x2[16]);
    x3[2]  = S16(x1[17], x2[17]);
    x3[3]  = A16(x1[17], x2[17]);
    x3[4]  = A16(x1[18], x2[18]);
    x3[5]  = S16(x1[18], x2[18]);
    x3[6]  = S16(x1[21], x2[19]);
    x3[7]  = A16(x1[21], x2[19]);
    x3[8]  = A16(x1[24], x2[20]);
    x3[9]  = S16(x1[24], x2[20]);
    x3[10] = S16(x1[25], x2[21]);
    x3[11] = A16(x1[25], x2[21]);
    x3[12] = A16(x1[28], x2[22]);
    x3[13] = S16(x1[28], x2[22]);
    x3[14] = S16(x1[29], x2[23]);
    x3[15] = A16(x1[29], x2[23]);
    x3[16] = A16(x1[32], x2[24]);
    x3[17] = S16(x1[32], x2[24]);
    x3[18] = S16(x1[33], x2[25]);
    x3[19] = A16(x1[33], x2[25]);
    x3[20] = A16(x1[36], x2[26]);
    x3[21] = S16(x1[36], x2[26]);
    x3[22] = S16(x1[37], x2[27]);
    x3[23] = A16(x1[37], x2[27]);
    x3[24] = A16(x1[40], x2[28]);
    x3[25] = S16(x1[40], x2[28]);
    x3[26] = S16(x1[41], x2[29]);
    x3[27] = A16(x1[41], x2[29]);
    x3[28] = A16(x1[44], x2[30]);
    x3[29] = S16(x1[44], x2[30]);
    x3[30] = S16(x1[45], x2[31]);
    x3[31] = A16(x1[45], x2[31]);
    // stage 10
    btf16_1_x32(c63, c1, x3[0], x3[31], &output[1], &output[63], rnd, cos_bit);
    btf16_1_x32(c31, c33, x3[1], x3[30], &output[33], &output[31], rnd, cos_bit);
    btf16_1_x32(c47, c17, x3[2], x3[29], &output[17], &output[47], rnd, cos_bit);
    btf16_1_x32(c15, c49, x3[3], x3[28], &output[49], &output[15], rnd, cos_bit);
    btf16_1_x32(c55, c9, x3[4], x3[27], &output[9], &output[55], rnd, cos_bit);
    btf16_1_x32(c23, c41, x3[5], x3[26], &output[41], &output[23], rnd, cos_bit);
    btf16_1_x32(c39, c25, x3[6], x3[25], &output[25], &output[39], rnd, cos_bit);
    btf16_1_x32(c7, c57, x3[7], x3[24], &output[57], &output[7], rnd, cos_bit);
    btf16_1_x32(c59, c5, x3[8], x3[23], &output[5], &output[59], rnd, cos_bit);
    btf16_1_x32(c27, c37, x3[9], x3[22], &output[37], &output[27], rnd, cos_bit);
    btf16_1_x32(c43, c21, x3[10], x3[21], &output[21], &output[43], rnd, cos_bit);
    btf16_1_x32(c11, c53, x3[11], x3[20], &output[53], &output[11], rnd, cos_bit);
    btf16_1_x32(c51, c13, x3[12], x3[19], &output[13], &output[51], rnd, cos_bit);
    btf16_1_x32(c19, c45, x3[13], x3[18], &output[45], &output[19], rnd, cos_bit);
    btf16_1_x32(c35, c29, x3[14], x3[17], &output[29], &output[35], rnd, cos_bit);
    btf16_1_x32(c3, c61, x3[15], x3[16], &output[61], &output[3], rnd, cos_bit);
}

// 64x64 column pass in int16, emitting the post-round result as int32 in the
// layout the int32 row pipeline expects: out[p*4 + g] = coeff row p, cols
// 16g..16g+15. Column values fit int16 (peak ~8160), so this is bit-exact with
// the int32 column transform; the row pass stays int32 (values reach ~2^17).
void svt_lbd_fwd_col64_avx512(const int16_t* input, uint32_t stride, __m512i* out) {
    const __m512i r2 = _mm512_set1_epi16(2); // round for >>2 (shift[1] = -2)
    for (int g2 = 0; g2 < 2; g2++) { // two 32-column groups
        __m512i cin[64], cout[64];
        for (int p = 0; p < 64; p++) {
            cin[p] = _mm512_loadu_si512((const void*)(input + p * stride + 32 * g2));
        }
        fdct64_x32_avx512(cin, cout, 13); // fwd_cos_bit_col[TX_64X64] = 13
        for (int p = 0; p < 64; p++) {
            const __m512i v         = _mm512_srai_epi16(_mm512_adds_epi16(cout[p], r2), 2);
            const __m256i lo        = _mm512_castsi512_si256(v); // cols 32*g2 .. +15
            const __m256i hi        = _mm512_extracti64x4_epi64(v, 1); // cols 32*g2+16 .. +31
            out[p * 4 + 2 * g2 + 0] = _mm512_cvtepi16_epi32(lo);
            out[p * 4 + 2 * g2 + 1] = _mm512_cvtepi16_epi32(hi);
        }
    }
}

// 32x64 column pass: fdct64 on 32 columns (1 group), shift[0]=0, >>2 (shift[1]).
// Emits int32 layout out[p*2 + g] = row p, cols 16g..16g+15. Column fits int16.
void svt_lbd_fwd_col64_32w_avx512(const int16_t* input, uint32_t stride, __m512i* out) {
    const __m512i r2 = _mm512_set1_epi16(2);
    __m512i       cin[64], cout[64];
    for (int p = 0; p < 64; p++) {
        cin[p] = _mm512_loadu_si512((const void*)(input + p * stride));
    }
    fdct64_x32_avx512(cin, cout, 13);
    for (int p = 0; p < 64; p++) {
        const __m512i v = _mm512_srai_epi16(_mm512_adds_epi16(cout[p], r2), 2);
        out[p * 2 + 0]  = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v));
        out[p * 2 + 1]  = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1));
    }
}

// 64x32 column pass: fdct32 on 64 columns (two 32-col groups), shift[0]=2, >>4
// (shift[1]). Identical to the 32x32 column transform, so bit-exact + fits int16.
// Emits int32 layout out[p*4 + g] = row p (0..31), cols 16g..16g+15.
void svt_lbd_fwd_col32_64w_avx512(const int16_t* input, uint32_t stride, __m512i* out) {
    const __m512i r8 = _mm512_set1_epi16(8); // round for >>4
    for (int g2 = 0; g2 < 2; g2++) {
        __m512i cin[32], cout[32];
        for (int p = 0; p < 32; p++) {
            cin[p] = _mm512_slli_epi16(_mm512_loadu_si512((const void*)(input + p * stride + 32 * g2)), 2);
        }
        fdct32_x32_avx512(cin, cout, 12);
        for (int p = 0; p < 32; p++) {
            const __m512i v         = _mm512_srai_epi16(_mm512_adds_epi16(cout[p], r8), 4);
            out[p * 4 + 2 * g2 + 0] = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v));
            out[p * 4 + 2 * g2 + 1] = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1));
        }
    }
}
