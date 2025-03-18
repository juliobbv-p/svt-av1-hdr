/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>
#include <arm_sve.h>

#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "mem_neon.h"
#include "neon_sve_bridge.h"
#include "pickrst_neon.h"
#include "pickrst_sve.h"
#include "restoration.h"
#include "restoration_pick.h"
#include "sum_neon.h"
#include "transpose_neon.h"
#include "utility.h"

static inline uint8_t find_average_sve(const uint8_t *src, int src_stride, int width, int height) {
    uint32x4_t avg_u32 = vdupq_n_u32(0);
    uint8x16_t ones    = vdupq_n_u8(1);

    // Use a predicate to compute the last columns.
    svbool_t pattern = svwhilelt_b8_u32(0, width % 16);

    int h = height;
    do {
        int            j       = width;
        const uint8_t *src_ptr = src;
        while (j >= 16) {
            uint8x16_t s = vld1q_u8(src_ptr);
            avg_u32      = vdotq_u32(avg_u32, s, ones);

            j -= 16;
            src_ptr += 16;
        }
        uint8x16_t s_end = svget_neonq_u8(svld1_u8(pattern, src_ptr));
        avg_u32          = vdotq_u32(avg_u32, s_end, ones);

        src += src_stride;
    } while (--h != 0);
    return (uint8_t)(vaddlvq_u32(avg_u32) / (width * height));
}

static inline void compute_sub_avg(const uint8_t *buf, int buf_stride, int avg, int16_t *buf_avg, int buf_avg_stride,
                                   int width, int height) {
    uint8x8_t avg_u8 = vdup_n_u8(avg);

    // Use a predicate to compute the last columns.
    svbool_t pattern = svwhilelt_b8_u32(0, width % 8);

    uint8x8_t avg_end = vget_low_u8(svget_neonq_u8(svdup_n_u8_z(pattern, avg)));

    do {
        int            j           = width;
        const uint8_t *buf_ptr     = buf;
        int16_t       *buf_avg_ptr = buf_avg;
        while (j >= 8) {
            uint8x8_t d = vld1_u8(buf_ptr);
            vst1q_s16(buf_avg_ptr, vreinterpretq_s16_u16(vsubl_u8(d, avg_u8)));

            j -= 8;
            buf_ptr += 8;
            buf_avg_ptr += 8;
        }
        uint8x8_t d_end = vget_low_u8(svget_neonq_u8(svld1_u8(pattern, buf_ptr)));
        vst1q_s16(buf_avg_ptr, vreinterpretq_s16_u16(vsubl_u8(d_end, avg_end)));

        buf += buf_stride;
        buf_avg += buf_avg_stride;
    } while (--height > 0);
}

void svt_av1_compute_stats_sve(int32_t wiener_win, const uint8_t *dgd, const uint8_t *src, int32_t h_start,
                               int32_t h_end, int32_t v_start, int32_t v_end, int32_t dgd_stride, int32_t src_stride,
                               int64_t *M, int64_t *H) {
    const int32_t wiener_win2    = wiener_win * wiener_win;
    const int32_t wiener_halfwin = (wiener_win >> 1);
    const int32_t width          = h_end - h_start;
    const int32_t height         = v_end - v_start;
    const int32_t d_stride       = (width + 2 * wiener_halfwin + 15) & ~15;
    const int32_t s_stride       = (width + 15) & ~15;
    int16_t      *d, *s;

    const uint8_t *dgd_start = dgd + h_start + v_start * dgd_stride;
    const uint8_t *src_start = src + h_start + v_start * src_stride;
    const uint16_t avg       = find_average_sve(dgd_start, dgd_stride, width, height);

    // The maximum input size is width * height, which is
    // (9 / 4) * RESTORATION_UNITSIZE_MAX * RESTORATION_UNITSIZE_MAX. Enlarge to
    // 3 * RESTORATION_UNITSIZE_MAX * RESTORATION_UNITSIZE_MAX considering
    // paddings.
    d = svt_aom_memalign(32, sizeof(*d) * 6 * RESTORATION_UNITSIZE_MAX * RESTORATION_UNITSIZE_MAX);
    s = d + 3 * RESTORATION_UNITSIZE_MAX * RESTORATION_UNITSIZE_MAX;

    compute_sub_avg(src_start, src_stride, avg, s, s_stride, width, height);
    compute_sub_avg(dgd + (v_start - wiener_halfwin) * dgd_stride + h_start - wiener_halfwin,
                    dgd_stride,
                    avg,
                    d,
                    d_stride,
                    width + 2 * wiener_halfwin,
                    height + 2 * wiener_halfwin);

    if (wiener_win == WIENER_WIN) {
        compute_stats_win7_sve(d, d_stride, s, s_stride, width, height, M, H);
    } else if (wiener_win == WIENER_WIN_CHROMA) {
        compute_stats_win5_sve(d, d_stride, s, s_stride, width, height, M, H);
    } else {
        assert(wiener_win == WIENER_WIN_3TAP);
        compute_stats_win3_sve(d, d_stride, s, s_stride, width, height, M, H);
    }

    // H is a symmetric matrix, so we only need to fill out the upper triangle.
    // We can copy it down to the lower triangle outside the (i, j) loops.
    diagonal_copy_stats_neon(wiener_win2, H);

    svt_aom_free(d);
}

int64_t svt_av1_lowbd_pixel_proj_error_sve(const uint8_t *src8, int32_t width, int32_t height, int32_t src_stride,
                                           const uint8_t *dat8, int32_t dat_stride, int32_t *flt0, int32_t flt0_stride,
                                           int32_t *flt1, int32_t flt1_stride, int32_t xq[2],
                                           const SgrParamsType *params) {
    if (width % 16 != 0) {
        return svt_av1_lowbd_pixel_proj_error_c(
            src8, width, height, src_stride, dat8, dat_stride, flt0, flt0_stride, flt1, flt1_stride, xq, params);
    }

    int64x2_t sse_s64 = vdupq_n_s64(0);

    if (params->r[0] > 0 && params->r[1] > 0) {
        int32x2_t xq_v     = vld1_s32(xq);
        int16x4_t xq_sum_v = vreinterpret_s16_s32(vshl_n_s32(vpadd_s32(xq_v, xq_v), SGRPROJ_RST_BITS));

        do {
            int j = 0;

            do {
                const uint8x8_t d      = vld1_u8(&dat8[j]);
                const uint8x8_t s      = vld1_u8(&src8[j]);
                int32x4_t       flt0_0 = vld1q_s32(&flt0[j]);
                int32x4_t       flt0_1 = vld1q_s32(&flt0[j + 4]);
                int32x4_t       flt1_0 = vld1q_s32(&flt1[j]);
                int32x4_t       flt1_1 = vld1q_s32(&flt1[j + 4]);

                int32x4_t offset = vdupq_n_s32(1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1));
                int32x4_t v0     = vmlaq_lane_s32(offset, flt0_0, xq_v, 0);
                int32x4_t v1     = vmlaq_lane_s32(offset, flt0_1, xq_v, 0);

                v0 = vmlaq_lane_s32(v0, flt1_0, xq_v, 1);
                v1 = vmlaq_lane_s32(v1, flt1_1, xq_v, 1);

                int16x8_t d_s16 = vreinterpretq_s16_u16(vmovl_u8(d));
                v0              = vmlsl_lane_s16(v0, vget_low_s16(d_s16), xq_sum_v, 0);
                v1              = vmlsl_lane_s16(v1, vget_high_s16(d_s16), xq_sum_v, 0);

                int16x4_t vr0 = vshrn_n_s32(v0, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);
                int16x4_t vr1 = vshrn_n_s32(v1, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);

                int16x8_t diff = vreinterpretq_s16_u16(vsubl_u8(d, s));
                int16x8_t e    = vaddq_s16(vcombine_s16(vr0, vr1), diff);
                sse_s64        = svt_sdotq_s16(sse_s64, e, e);

                j += 8;
            } while (j != width);

            dat8 += dat_stride;
            src8 += src_stride;
            flt0 += flt0_stride;
            flt1 += flt1_stride;
        } while (--height != 0);
    } else if (params->r[0] > 0 || params->r[1] > 0) {
        const int32_t  xq_active  = (params->r[0] > 0) ? xq[0] : xq[1];
        const int32_t *flt        = (params->r[0] > 0) ? flt0 : flt1;
        const int32_t  flt_stride = (params->r[0] > 0) ? flt0_stride : flt1_stride;
        int32x2_t      xq_v       = vdup_n_s32(xq_active);

        do {
            int j = 0;

            do {
                const uint8x8_t d     = vld1_u8(&dat8[j]);
                const uint8x8_t s     = vld1_u8(&src8[j]);
                int32x4_t       flt_0 = vld1q_s32(&flt[j]);
                int32x4_t       flt_1 = vld1q_s32(&flt[j + 4]);
                int16x8_t       d_s16 = vreinterpretq_s16_u16(vshll_n_u8(d, SGRPROJ_RST_BITS));

                int32x4_t sub_0 = vsubw_s16(flt_0, vget_low_s16(d_s16));
                int32x4_t sub_1 = vsubw_s16(flt_1, vget_high_s16(d_s16));

                int32x4_t offset = vdupq_n_s32(1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1));
                int32x4_t v0     = vmlaq_lane_s32(offset, sub_0, xq_v, 0);
                int32x4_t v1     = vmlaq_lane_s32(offset, sub_1, xq_v, 0);

                int16x4_t vr0 = vshrn_n_s32(v0, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);
                int16x4_t vr1 = vshrn_n_s32(v1, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);

                int16x8_t diff = vreinterpretq_s16_u16(vsubl_u8(d, s));
                int16x8_t e    = vaddq_s16(vcombine_s16(vr0, vr1), diff);
                sse_s64        = svt_sdotq_s16(sse_s64, e, e);

                j += 8;
            } while (j != width);

            dat8 += dat_stride;
            src8 += src_stride;
            flt += flt_stride;
        } while (--height != 0);
    } else {
        uint32x4_t sse_u32 = vdupq_n_u32(0);

        do {
            int j = 0;
            do {
                const uint8x16_t d    = vld1q_u8(&dat8[j]);
                const uint8x16_t s    = vld1q_u8(&src8[j]);
                uint8x16_t       diff = vabdq_u8(d, s);

                sse_u32 = vdotq_u32(sse_u32, diff, diff);

                j += 16;
            } while (j != width);

            dat8 += dat_stride;
            src8 += src_stride;
        } while (--height != 0);

        return (int64_t)vaddlvq_u32(sse_u32);
    }

    return vaddvq_s64(sse_s64);
}

int64_t svt_av1_highbd_pixel_proj_error_sve(const uint8_t *src8, int32_t width, int32_t height, int32_t src_stride,
                                            const uint8_t *dat8, int32_t dat_stride, int32_t *flt0, int32_t flt0_stride,
                                            int32_t *flt1, int32_t flt1_stride, int32_t xq[2],
                                            const SgrParamsType *params) {
    if (width % 8 != 0) {
        return svt_av1_highbd_pixel_proj_error_c(
            src8, width, height, src_stride, dat8, dat_stride, flt0, flt0_stride, flt1, flt1_stride, xq, params);
    }
    const uint16_t *src     = CONVERT_TO_SHORTPTR(src8);
    const uint16_t *dat     = CONVERT_TO_SHORTPTR(dat8);
    int64x2_t       sse_s64 = vdupq_n_s64(0);

    if (params->r[0] > 0 && params->r[1] > 0) {
        int32x2_t  xq_v     = vld1_s32(xq);
        uint16x4_t xq_sum_v = vreinterpret_u16_s32(vshl_n_s32(vpadd_s32(xq_v, xq_v), SGRPROJ_RST_BITS));

        do {
            int j = 0;

            do {
                const uint16x8_t d      = vld1q_u16(&dat[j]);
                const uint16x8_t s      = vld1q_u16(&src[j]);
                int32x4_t        flt0_0 = vld1q_s32(&flt0[j]);
                int32x4_t        flt0_1 = vld1q_s32(&flt0[j + 4]);
                int32x4_t        flt1_0 = vld1q_s32(&flt1[j]);
                int32x4_t        flt1_1 = vld1q_s32(&flt1[j + 4]);

                int32x4_t d_s32_lo = vreinterpretq_s32_u32(vmull_lane_u16(vget_low_u16(d), xq_sum_v, 0));
                int32x4_t d_s32_hi = vreinterpretq_s32_u32(vmull_lane_u16(vget_high_u16(d), xq_sum_v, 0));

                int32x4_t v0 = vsubq_s32(vdupq_n_s32(1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1)), d_s32_lo);
                int32x4_t v1 = vsubq_s32(vdupq_n_s32(1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1)), d_s32_hi);

                v0 = vmlaq_lane_s32(v0, flt0_0, xq_v, 0);
                v1 = vmlaq_lane_s32(v1, flt0_1, xq_v, 0);
                v0 = vmlaq_lane_s32(v0, flt1_0, xq_v, 1);
                v1 = vmlaq_lane_s32(v1, flt1_1, xq_v, 1);

                int16x4_t vr0 = vshrn_n_s32(v0, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);
                int16x4_t vr1 = vshrn_n_s32(v1, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);

                int16x8_t e = vaddq_s16(vcombine_s16(vr0, vr1), vreinterpretq_s16_u16(vsubq_u16(d, s)));
                sse_s64     = svt_sdotq_s16(sse_s64, e, e);

                j += 8;
            } while (j != width);

            dat += dat_stride;
            src += src_stride;
            flt0 += flt0_stride;
            flt1 += flt1_stride;
        } while (--height != 0);
    } else if (params->r[0] > 0 || params->r[1] > 0) {
        int       xq_active  = (params->r[0] > 0) ? xq[0] : xq[1];
        int32_t  *flt        = (params->r[0] > 0) ? flt0 : flt1;
        int       flt_stride = (params->r[0] > 0) ? flt0_stride : flt1_stride;
        int32x4_t xq_v       = vdupq_n_s32(xq_active);

        do {
            int j = 0;

            do {
                const uint16x8_t d0     = vld1q_u16(&dat[j]);
                const uint16x8_t s0     = vld1q_u16(&src[j]);
                int32x4_t        flt0_0 = vld1q_s32(&flt[j]);
                int32x4_t        flt0_1 = vld1q_s32(&flt[j + 4]);

                uint16x8_t d_u16 = vshlq_n_u16(d0, SGRPROJ_RST_BITS);
                int32x4_t  sub0  = vreinterpretq_s32_u32(vsubw_u16(vreinterpretq_u32_s32(flt0_0), vget_low_u16(d_u16)));
                int32x4_t  sub1 = vreinterpretq_s32_u32(vsubw_u16(vreinterpretq_u32_s32(flt0_1), vget_high_u16(d_u16)));

                int32x4_t v0 = vmlaq_s32(vdupq_n_s32(1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1)), sub0, xq_v);
                int32x4_t v1 = vmlaq_s32(vdupq_n_s32(1 << (SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS - 1)), sub1, xq_v);

                int16x4_t vr0 = vshrn_n_s32(v0, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);
                int16x4_t vr1 = vshrn_n_s32(v1, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);

                int16x8_t e = vaddq_s16(vcombine_s16(vr0, vr1), vreinterpretq_s16_u16(vsubq_u16(d0, s0)));
                sse_s64     = svt_sdotq_s16(sse_s64, e, e);

                j += 8;
            } while (j != width);

            dat += dat_stride;
            flt += flt_stride;
            src += src_stride;
        } while (--height != 0);
    } else {
        do {
            int j = 0;

            do {
                const uint16x8_t d = vld1q_u16(&dat[j]);
                const uint16x8_t s = vld1q_u16(&src[j]);

                int16x8_t diff = vreinterpretq_s16_u16(vabdq_u16(d, s));
                sse_s64        = svt_sdotq_s16(sse_s64, diff, diff);

                j += 8;
            } while (j != width);

            dat += dat_stride;
            src += src_stride;
        } while (--height != 0);
    }

    return vaddvq_s64(sse_s64);
}
