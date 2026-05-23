/*
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>
#include <assert.h>

#include "aom_dsp_rtcd.h"
#include "mem_neon.h"
#include "var_filter_neon.h"

#ifdef __clang__
#define DISABLE_LOOP_UNROLL 1
#else
#define DISABLE_LOOP_UNROLL 0
#endif

static inline void variance_4xh_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride,
                                             int h, uint32_t* sse, int* sum) {
    uint32x4_t src_sum = vdupq_n_u32(0);
    uint32x4_t ref_sum = vdupq_n_u32(0);
    uint32x4_t sse_u32 = vdupq_n_u32(0);

    do {
        uint8x16_t s = load_u8_4x4(src, src_stride);
        uint8x16_t r = load_u8_4x4(ref, ref_stride);

        src_sum = vdotq_u32(src_sum, s, vdupq_n_u8(1));
        ref_sum = vdotq_u32(ref_sum, r, vdupq_n_u8(1));

        uint8x16_t abs_diff = vabdq_u8(s, r);
        sse_u32             = vdotq_u32(sse_u32, abs_diff, abs_diff);

        src += 4 * src_stride;
        ref += 4 * ref_stride;
        h -= 4;
    } while (h != 0);

    int32x4_t sum_diff = vsubq_s32(vreinterpretq_s32_u32(src_sum), vreinterpretq_s32_u32(ref_sum));
    *sum               = vaddvq_s32(sum_diff);
    *sse               = vaddvq_u32(sse_u32);
}

static inline void variance_8xh_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride,
                                             int h, uint32_t* sse, int* sum) {
    uint32x4_t src_sum = vdupq_n_u32(0);
    uint32x4_t ref_sum = vdupq_n_u32(0);
    uint32x4_t sse_u32 = vdupq_n_u32(0);

#if DISABLE_LOOP_UNROLL
#pragma clang loop unroll(disable)
#endif
    do {
        uint8x16_t s = load_u8_8x2(src, src_stride);
        uint8x16_t r = load_u8_8x2(ref, ref_stride);

        src_sum = vdotq_u32(src_sum, s, vdupq_n_u8(1));
        ref_sum = vdotq_u32(ref_sum, r, vdupq_n_u8(1));

        uint8x16_t abs_diff = vabdq_u8(s, r);
        sse_u32             = vdotq_u32(sse_u32, abs_diff, abs_diff);

        src += 2 * src_stride;
        ref += 2 * ref_stride;
        h -= 2;
    } while (h != 0);

    int32x4_t sum_diff = vsubq_s32(vreinterpretq_s32_u32(src_sum), vreinterpretq_s32_u32(ref_sum));
    *sum               = vaddvq_s32(sum_diff);
    *sse               = vaddvq_u32(sse_u32);
}

static inline void variance_16xh_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride,
                                              int h, uint32_t* sse, int* sum) {
    uint32x4_t src_sum[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};
    uint32x4_t ref_sum[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};
    uint32x4_t sse_u32[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};

#if DISABLE_LOOP_UNROLL
#pragma clang loop unroll(disable)
#endif
    do {
        uint8x16_t s0 = vld1q_u8(src);
        uint8x16_t r0 = vld1q_u8(ref);
        uint8x16_t s1 = vld1q_u8(src + src_stride);
        uint8x16_t r1 = vld1q_u8(ref + ref_stride);

        src_sum[0] = vdotq_u32(src_sum[0], s0, vdupq_n_u8(1));
        ref_sum[0] = vdotq_u32(ref_sum[0], r0, vdupq_n_u8(1));
        src_sum[1] = vdotq_u32(src_sum[1], s1, vdupq_n_u8(1));
        ref_sum[1] = vdotq_u32(ref_sum[1], r1, vdupq_n_u8(1));

        uint8x16_t d0 = vabdq_u8(s0, r0);
        sse_u32[0]    = vdotq_u32(sse_u32[0], d0, d0);
        uint8x16_t d1 = vabdq_u8(s1, r1);
        sse_u32[1]    = vdotq_u32(sse_u32[1], d1, d1);

        src += 2 * src_stride;
        ref += 2 * ref_stride;
        h -= 2;
    } while (h != 0);

    src_sum[0]         = vaddq_u32(src_sum[0], src_sum[1]);
    ref_sum[0]         = vaddq_u32(ref_sum[0], ref_sum[1]);
    int32x4_t sum_diff = vsubq_s32(vreinterpretq_s32_u32(src_sum[0]), vreinterpretq_s32_u32(ref_sum[0]));
    *sum               = vaddvq_s32(sum_diff);
    *sse               = vaddvq_u32(vaddq_u32(sse_u32[0], sse_u32[1]));
}

static inline void variance_large_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride,
                                               int w, int h, uint32_t* sse, int* sum) {
    uint32x4_t src_sum[4] = {vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)};
    uint32x4_t ref_sum[4] = {vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)};
    uint32x4_t sse_u32[4] = {vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)};

#if DISABLE_LOOP_UNROLL
#pragma clang loop unroll(disable)
#endif
    do {
        int i = 0;
        int t = 0;
        do {
            uint8x16_t s = vld1q_u8(src + i);
            uint8x16_t r = vld1q_u8(ref + i);

            src_sum[t] = vdotq_u32(src_sum[t], s, vdupq_n_u8(1));
            ref_sum[t] = vdotq_u32(ref_sum[t], r, vdupq_n_u8(1));

            uint8x16_t abs_diff = vabdq_u8(s, r);
            sse_u32[t]          = vdotq_u32(sse_u32[t], abs_diff, abs_diff);

            i += 16;
            t = (t + 1) & 3;
        } while (i < w);

        src += src_stride;
        ref += ref_stride;
    } while (--h != 0);

    uint32x4_t src_sum_v = vaddq_u32(vaddq_u32(src_sum[0], src_sum[1]), vaddq_u32(src_sum[2], src_sum[3]));
    uint32x4_t ref_sum_v = vaddq_u32(vaddq_u32(ref_sum[0], ref_sum[1]), vaddq_u32(ref_sum[2], ref_sum[3]));
    int32x4_t  sum_diff  = vsubq_s32(vreinterpretq_s32_u32(src_sum_v), vreinterpretq_s32_u32(ref_sum_v));
    *sum                 = vaddvq_s32(sum_diff);
    uint32x4_t sse_sum   = vaddq_u32(vaddq_u32(sse_u32[0], sse_u32[1]), vaddq_u32(sse_u32[2], sse_u32[3]));
    *sse                 = vaddvq_u32(sse_sum);
}

static inline void variance_32xh_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride,
                                              int h, uint32_t* sse, int* sum) {
    variance_large_neon_dotprod(src, src_stride, ref, ref_stride, 32, h, sse, sum);
}

static inline void variance_64xh_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride,
                                              int h, uint32_t* sse, int* sum) {
    variance_large_neon_dotprod(src, src_stride, ref, ref_stride, 64, h, sse, sum);
}

static inline void variance_128xh_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride,
                                               int h, uint32_t* sse, int* sum) {
    variance_large_neon_dotprod(src, src_stride, ref, ref_stride, 128, h, sse, sum);
}

#define VARIANCE_WXH_NEON_DOTPROD(w, h, shift)                                                       \
    unsigned int svt_aom_variance##w##x##h##_neon_dotprod(                                           \
        const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, unsigned int* sse) { \
        int sum;                                                                                     \
        variance_##w##xh_neon_dotprod(src, src_stride, ref, ref_stride, h, sse, &sum);               \
        return *sse - (uint32_t)(((int64_t)sum * sum) >> shift);                                     \
    }

// The Armv8.0 Neon implementation is faster than Neon DotProd for 4x4.
VARIANCE_WXH_NEON_DOTPROD(4, 8, 5)
VARIANCE_WXH_NEON_DOTPROD(4, 16, 6)

VARIANCE_WXH_NEON_DOTPROD(8, 4, 5)
VARIANCE_WXH_NEON_DOTPROD(8, 8, 6)
VARIANCE_WXH_NEON_DOTPROD(8, 16, 7)
VARIANCE_WXH_NEON_DOTPROD(8, 32, 8)

VARIANCE_WXH_NEON_DOTPROD(16, 8, 7)
VARIANCE_WXH_NEON_DOTPROD(16, 16, 8)
VARIANCE_WXH_NEON_DOTPROD(16, 32, 9)
VARIANCE_WXH_NEON_DOTPROD(16, 4, 6)
VARIANCE_WXH_NEON_DOTPROD(16, 64, 10)

VARIANCE_WXH_NEON_DOTPROD(32, 16, 9)
VARIANCE_WXH_NEON_DOTPROD(32, 32, 10)
VARIANCE_WXH_NEON_DOTPROD(32, 64, 11)
VARIANCE_WXH_NEON_DOTPROD(32, 8, 8)

VARIANCE_WXH_NEON_DOTPROD(64, 32, 11)
VARIANCE_WXH_NEON_DOTPROD(64, 64, 12)
VARIANCE_WXH_NEON_DOTPROD(64, 128, 13)
VARIANCE_WXH_NEON_DOTPROD(64, 16, 10)

VARIANCE_WXH_NEON_DOTPROD(128, 64, 13)
VARIANCE_WXH_NEON_DOTPROD(128, 128, 14)

#undef VARIANCE_WXH_NEON_DOTPROD

// =============================================================================
// Small-block sub_pixel_variance helpers (4xH) dotprod
// =============================================================================
#define SUBPEL_VARIANCE_4XH_NEON_DOTPROD(h, padding)                                       \
    unsigned int svt_aom_sub_pixel_variance4x##h##_neon_dotprod(const uint8_t* src,        \
                                                                int            src_stride, \
                                                                int            xoffset,    \
                                                                int            yoffset,    \
                                                                const uint8_t* ref,        \
                                                                int            ref_stride, \
                                                                uint32_t*      sse) {           \
        uint8_t tmp0[4 * (h + padding)];                                                   \
        uint8_t tmp1[4 * h];                                                               \
        var_filter_block2d_bil_w4(src, tmp0, src_stride, 1, (h + padding), xoffset);       \
        var_filter_block2d_bil_w4(tmp0, tmp1, 4, 4, h, yoffset);                           \
        return svt_aom_variance4x##h##_neon_dotprod(tmp1, 4, ref, ref_stride, sse);        \
    }

SUBPEL_VARIANCE_4XH_NEON_DOTPROD(8, 2)
SUBPEL_VARIANCE_4XH_NEON_DOTPROD(16, 2)

#undef SUBPEL_VARIANCE_4XH_NEON_DOTPROD

static inline uint8x8_t spv_dp_w8_h_bil(const uint8_t* p, int off, uint8x8_t f0, uint8x8_t f1) {
    uint8x8_t s0 = vld1_u8(p);
    uint8x8_t s1 = vld1_u8(p + 1);
    if (off == 2) {
        return vrhadd_u8(s0, vhadd_u8(s0, s1));
    }
    if (off == 6) {
        return vrhadd_u8(s1, vhadd_u8(s0, s1));
    }
    uint16x8_t b = vmull_u8(s0, f0);
    b            = vmlal_u8(b, s1, f1);
    return vrshrn_n_u16(b, 3);
}

static inline uint8x8_t spv_dp_w8_v_bil(uint8x8_t a, uint8x8_t b, int off, uint8x8_t f0, uint8x8_t f1) {
    if (off == 2) {
        return vrhadd_u8(a, vhadd_u8(a, b));
    }
    if (off == 6) {
        return vrhadd_u8(b, vhadd_u8(a, b));
    }
    uint16x8_t v = vmull_u8(a, f0);
    v            = vmlal_u8(v, b, f1);
    return vrshrn_n_u16(v, 3);
}

static inline uint8x8_t spv_dp_w8_h_0(const uint8_t* p) {
    return vld1_u8(p);
}

static inline uint8x8_t spv_dp_w8_h_2(const uint8_t* p) {
    uint8x8_t s0 = vld1_u8(p);
    uint8x8_t s1 = vld1_u8(p + 1);
    return vrhadd_u8(s0, vhadd_u8(s0, s1));
}

static inline uint8x8_t spv_dp_w8_h_4(const uint8_t* p) {
    return vrhadd_u8(vld1_u8(p), vld1_u8(p + 1));
}

static inline uint8x8_t spv_dp_w8_h_6(const uint8_t* p) {
    uint8x8_t s0 = vld1_u8(p);
    uint8x8_t s1 = vld1_u8(p + 1);
    return vrhadd_u8(s1, vhadd_u8(s0, s1));
}

static inline uint8x8_t spv_dp_w8_v_2(uint8x8_t a, uint8x8_t b) {
    return vrhadd_u8(a, vhadd_u8(a, b));
}

static inline uint8x8_t spv_dp_w8_v_4(uint8x8_t a, uint8x8_t b) {
    return vrhadd_u8(a, b);
}

static inline uint8x8_t spv_dp_w8_v_6(uint8x8_t a, uint8x8_t b) {
    return vrhadd_u8(b, vhadd_u8(a, b));
}

#define SPV_KEY(xoff, yoff) (((xoff) << 3) | (yoff))

#define SPV_DP_W8_ACCUM(pred, refptr)                    \
    do {                                                 \
        uint8x8_t _r  = vld1_u8(refptr);                 \
        src_sum       = vdot_u32(src_sum, (pred), ones); \
        ref_sum       = vdot_u32(ref_sum, _r, ones);     \
        uint8x8_t _ad = vabd_u8((pred), _r);             \
        sse_u32       = vdot_u32(sse_u32, _ad, _ad);     \
    } while (0)

#define SPV_DP_W8_RUN_H(H, HF)                            \
    do {                                                  \
        for (int r = 0; r < (H); ++r) {                   \
            const uint8_t* sp_row = src + r * src_stride; \
            const uint8_t* rp_row = ref + r * ref_stride; \
            uint8x8_t      pred   = HF(sp_row);           \
            SPV_DP_W8_ACCUM(pred, rp_row);                \
        }                                                 \
    } while (0)

#define SPV_DP_W8_RUN_SW(H, HF, VF)                             \
    do {                                                        \
        uint8x8_t Hprev = HF(src);                              \
        for (int r = 0; r < (H); ++r) {                         \
            const uint8_t* sp_row = src + (r + 1) * src_stride; \
            const uint8_t* rp_row = ref + r * ref_stride;       \
            uint8x8_t      Hcur   = HF(sp_row);                 \
            uint8x8_t      pred   = VF(Hprev, Hcur);            \
            SPV_DP_W8_ACCUM(pred, rp_row);                      \
            Hprev = Hcur;                                       \
        }                                                       \
    } while (0)

#define SPV_DP_W8_HGEN(p) spv_dp_w8_h_bil((p), xoffset, f0_x, f1_x)
#define SPV_DP_W8_VGEN(a, b) spv_dp_w8_v_bil((a), (b), yoffset, f0_y, f1_y)

static NOINLINE unsigned int sub_pixel_variance_w8_neon_dotprod(const uint8_t* src, int src_stride, int xoffset,
                                                                int yoffset, const uint8_t* ref, int ref_stride, int h,
                                                                unsigned int* sse) {
    int sum;
    if (xoffset == 0 && yoffset == 0) {
        variance_8xh_neon_dotprod(src, src_stride, ref, ref_stride, h, sse, &sum);
        return *sse - (uint32_t)(((int64_t)sum * sum) >> svt_ctz(8 * h));
    }
    const uint8x8_t ones    = vdup_n_u8(1);
    uint32x2_t      src_sum = vdup_n_u32(0);
    uint32x2_t      ref_sum = vdup_n_u32(0);
    uint32x2_t      sse_u32 = vdup_n_u32(0);

    if (yoffset == 0) {
        switch (xoffset) {
        case 2:
            SPV_DP_W8_RUN_H(h, spv_dp_w8_h_2);
            break;
        case 4:
            SPV_DP_W8_RUN_H(h, spv_dp_w8_h_4);
            break;
        case 6:
            SPV_DP_W8_RUN_H(h, spv_dp_w8_h_6);
            break;
        default: {
            const uint8x8_t f0_x = vdup_n_u8((uint8_t)(8 - xoffset));
            const uint8x8_t f1_x = vdup_n_u8((uint8_t)xoffset);
            SPV_DP_W8_RUN_H(h, SPV_DP_W8_HGEN);
            break;
        }
        }
    } else {
        const uint8x8_t f0_x = vdup_n_u8((uint8_t)(8 - xoffset));
        const uint8x8_t f1_x = vdup_n_u8((uint8_t)xoffset);
        const uint8x8_t f0_y = vdup_n_u8((uint8_t)(8 - yoffset));
        const uint8x8_t f1_y = vdup_n_u8((uint8_t)yoffset);
        switch (SPV_KEY(xoffset, yoffset)) {
        case SPV_KEY(0, 2):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_0, spv_dp_w8_v_2);
            break;
        case SPV_KEY(0, 4):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_0, spv_dp_w8_v_4);
            break;
        case SPV_KEY(0, 6):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_0, spv_dp_w8_v_6);
            break;
        case SPV_KEY(0, 1):
        case SPV_KEY(0, 3):
        case SPV_KEY(0, 5):
        case SPV_KEY(0, 7):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_0, SPV_DP_W8_VGEN);
            break;
        case SPV_KEY(2, 2):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_2, spv_dp_w8_v_2);
            break;
        case SPV_KEY(2, 4):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_2, spv_dp_w8_v_4);
            break;
        case SPV_KEY(2, 6):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_2, spv_dp_w8_v_6);
            break;
        case SPV_KEY(2, 1):
        case SPV_KEY(2, 3):
        case SPV_KEY(2, 5):
        case SPV_KEY(2, 7):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_2, SPV_DP_W8_VGEN);
            break;
        case SPV_KEY(4, 2):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_4, spv_dp_w8_v_2);
            break;
        case SPV_KEY(4, 4):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_4, spv_dp_w8_v_4);
            break;
        case SPV_KEY(4, 6):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_4, spv_dp_w8_v_6);
            break;
        case SPV_KEY(4, 1):
        case SPV_KEY(4, 3):
        case SPV_KEY(4, 5):
        case SPV_KEY(4, 7):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_4, SPV_DP_W8_VGEN);
            break;
        case SPV_KEY(6, 2):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_6, spv_dp_w8_v_2);
            break;
        case SPV_KEY(6, 4):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_6, spv_dp_w8_v_4);
            break;
        case SPV_KEY(6, 6):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_6, spv_dp_w8_v_6);
            break;
        case SPV_KEY(6, 1):
        case SPV_KEY(6, 3):
        case SPV_KEY(6, 5):
        case SPV_KEY(6, 7):
            SPV_DP_W8_RUN_SW(h, spv_dp_w8_h_6, SPV_DP_W8_VGEN);
            break;
        case SPV_KEY(1, 2):
        case SPV_KEY(3, 2):
        case SPV_KEY(5, 2):
        case SPV_KEY(7, 2):
            SPV_DP_W8_RUN_SW(h, SPV_DP_W8_HGEN, spv_dp_w8_v_2);
            break;
        case SPV_KEY(1, 4):
        case SPV_KEY(3, 4):
        case SPV_KEY(5, 4):
        case SPV_KEY(7, 4):
            SPV_DP_W8_RUN_SW(h, SPV_DP_W8_HGEN, spv_dp_w8_v_4);
            break;
        case SPV_KEY(1, 6):
        case SPV_KEY(3, 6):
        case SPV_KEY(5, 6):
        case SPV_KEY(7, 6):
            SPV_DP_W8_RUN_SW(h, SPV_DP_W8_HGEN, spv_dp_w8_v_6);
            break;
        default:
            SPV_DP_W8_RUN_SW(h, SPV_DP_W8_HGEN, SPV_DP_W8_VGEN);
            break;
        }
    }

    int32x2_t sum_diff = vsub_s32(vreinterpret_s32_u32(src_sum), vreinterpret_s32_u32(ref_sum));
    sum                = vaddv_s32(sum_diff);
    *sse               = vaddv_u32(sse_u32);
    return *sse - (uint32_t)(((int64_t)sum * sum) >> svt_ctz(8 * h));
}

#define FUSED_SUBPEL_VARIANCE_W8H_NEON_DOTPROD(H)                                                                \
    unsigned int svt_aom_sub_pixel_variance8x##H##_neon_dotprod(const uint8_t* src,                              \
                                                                int            src_stride,                       \
                                                                int            xoffset,                          \
                                                                int            yoffset,                          \
                                                                const uint8_t* ref,                              \
                                                                int            ref_stride,                       \
                                                                unsigned int*  sse) {                             \
        return sub_pixel_variance_w8_neon_dotprod(src, src_stride, xoffset, yoffset, ref, ref_stride, (H), sse); \
    }

FUSED_SUBPEL_VARIANCE_W8H_NEON_DOTPROD(4)
FUSED_SUBPEL_VARIANCE_W8H_NEON_DOTPROD(8)
FUSED_SUBPEL_VARIANCE_W8H_NEON_DOTPROD(16)
FUSED_SUBPEL_VARIANCE_W8H_NEON_DOTPROD(32)

#undef FUSED_SUBPEL_VARIANCE_W8H_NEON_DOTPROD
#undef SPV_DP_W8_VGEN
#undef SPV_DP_W8_HGEN
#undef SPV_DP_W8_RUN_SW
#undef SPV_DP_W8_RUN_H
#undef SPV_DP_W8_ACCUM

// =============================================================================
// Fused-loop sub_pixel_variance for {16,32,64,128} x H (dotprod)
// =============================================================================
//
// Mirror of FUSED_SUBPEL_VARIANCE_WXH_NEON for the dotprod path. The udot
// accumulator (`src_sum`, `ref_sum`, `sse_u32` all uint32x4_t) absorbs
// per-byte contributions without int16 overflow risk, so no periodic
// absorb step is needed for any (W, H) up to 128x128.
//
// Bit-exact with the upstream bil_variance_neon_dotprod / avg_variance_
// neon_dotprod kernels: the per-pixel pred computation uses identical
// vrhaddq_u8 / vhaddq_u8 / vmull_u8+vmlal_u8+vrshrn_n_u16 chains, and the
// sum / sse accumulation uses the same vdotq_u32(.., ones) / vdotq_u32(..,
// abs_diff, abs_diff) reduction. Integer addition is associative so the
// per-tile reordering vs the upstream 2-pass version is bit-equivalent.

// spv_dp_h_bil / spv_dp_v_bil: generic bilinear for any offset in {1..7}.
// Includes urhadd-chain fast paths for off ∈ {2, 6}; see the NEON-side
// spv_h_bil doc-comment for the identity proof.
static inline uint8x16_t spv_dp_h_bil(const uint8_t* p, int off, uint8x8_t f0, uint8x8_t f1) {
    uint8x16_t s0 = vld1q_u8(p);
    uint8x16_t s1 = vld1q_u8(p + 1);
    if (off == 2) {
        return vrhaddq_u8(s0, vhaddq_u8(s0, s1));
    }
    if (off == 6) {
        return vrhaddq_u8(s1, vhaddq_u8(s0, s1));
    }
    uint16x8_t bl = vmull_u8(vget_low_u8(s0), f0);
    bl            = vmlal_u8(bl, vget_low_u8(s1), f1);
    uint16x8_t bh = vmull_u8(vget_high_u8(s0), f0);
    bh            = vmlal_u8(bh, vget_high_u8(s1), f1);
    return vcombine_u8(vrshrn_n_u16(bl, 3), vrshrn_n_u16(bh, 3));
}

static inline uint8x16_t spv_dp_v_bil(uint8x16_t a, uint8x16_t b, int off, uint8x8_t f0, uint8x8_t f1) {
    if (off == 2) {
        return vrhaddq_u8(a, vhaddq_u8(a, b));
    }
    if (off == 6) {
        return vrhaddq_u8(b, vhaddq_u8(a, b));
    }
    uint16x8_t vl = vmull_u8(vget_low_u8(a), f0);
    vl            = vmlal_u8(vl, vget_low_u8(b), f1);
    uint16x8_t vh = vmull_u8(vget_high_u8(a), f0);
    vh            = vmlal_u8(vh, vget_high_u8(b), f1);
    return vcombine_u8(vrshrn_n_u16(vl, 3), vrshrn_n_u16(vh, 3));
}

static inline uint8x16_t spv_dp_h_zero(const uint8_t* p) {
    return vld1q_u8(p);
}

static inline uint8x16_t spv_dp_h_two(const uint8_t* p) {
    uint8x16_t s0 = vld1q_u8(p);
    uint8x16_t s1 = vld1q_u8(p + 1);
    return vrhaddq_u8(s0, vhaddq_u8(s0, s1));
}

static inline uint8x16_t spv_dp_h_four(const uint8_t* p) {
    return vrhaddq_u8(vld1q_u8(p), vld1q_u8(p + 1));
}

static inline uint8x16_t spv_dp_h_six(const uint8_t* p) {
    uint8x16_t s0 = vld1q_u8(p);
    uint8x16_t s1 = vld1q_u8(p + 1);
    return vrhaddq_u8(s1, vhaddq_u8(s0, s1));
}

static inline uint8x16_t spv_dp_v_two(uint8x16_t a, uint8x16_t b) {
    return vrhaddq_u8(a, vhaddq_u8(a, b));
}

static inline uint8x16_t spv_dp_v_four(uint8x16_t a, uint8x16_t b) {
    return vrhaddq_u8(a, b);
}

static inline uint8x16_t spv_dp_v_six(uint8x16_t a, uint8x16_t b) {
    return vrhaddq_u8(b, vhaddq_u8(a, b));
}

// SPV_DP_TILES(W): number of 16-byte tile columns per row (1/2/4/8).
#define SPV_DP_TILES(W) ((W) / 16)
#define SPV_DP_NACCUM(W) ((SPV_DP_TILES(W) >= 4) ? 4 : SPV_DP_TILES(W))

// Per-tile pred → src_sum/ref_sum/sse_u32 accumulator. Uses udot for both
// the sum-of-bytes (against `ones`) and the SSE (against abs_diff). The
// uint32 accumulator has no overflow risk for any (W, H) up to 128x128.
#define SPV_DP_ACCUM_TILE(W, pred, refptr)                     \
    do {                                                       \
        uint8x16_t _r  = vld1q_u8(refptr);                     \
        const int  _i  = t & (SPV_DP_NACCUM(W) - 1);           \
        src_sum[_i]    = vdotq_u32(src_sum[_i], (pred), ones); \
        ref_sum[_i]    = vdotq_u32(ref_sum[_i], _r, ones);     \
        uint8x16_t _ad = vabdq_u8((pred), _r);                 \
        sse_u32[_i]    = vdotq_u32(sse_u32[_i], _ad, _ad);     \
    } while (0)

// H-only row body (yoffset == 0). One pass through src per output row;
// dotprod accumulator absorbs per-tile contributions with no overflow risk.
#define SPV_DP_RUN_H(W, H, HF)                               \
    do {                                                     \
        for (int r = 0; r < (H); ++r) {                      \
            const uint8_t* sp_row = src + r * src_stride;    \
            const uint8_t* rp_row = ref + r * ref_stride;    \
            for (int t = 0; t < SPV_DP_TILES(W); ++t) {      \
                uint8x16_t pred = HF(sp_row + t * 16);       \
                SPV_DP_ACCUM_TILE(W, pred, rp_row + t * 16); \
            }                                                \
        }                                                    \
    } while (0)

// Sliding-window row body (yoffset != 0). Hprev[t] holds H-filter(prev row,
// tile t); each iteration computes Hcur, applies V-filter, accumulates, and
// rotates. Mirrors SPV_RUN_SW from the NEON file, with udot accumulation.
#define SPV_DP_RUN_SW(W, H, HF, VF)                             \
    do {                                                        \
        uint8x16_t Hprev[SPV_DP_TILES(W)];                      \
        for (int t = 0; t < SPV_DP_TILES(W); ++t) {             \
            Hprev[t] = HF(src + t * 16);                        \
        }                                                       \
        for (int r = 0; r < (H); ++r) {                         \
            const uint8_t* sp_row = src + (r + 1) * src_stride; \
            const uint8_t* rp_row = ref + r * ref_stride;       \
            for (int t = 0; t < SPV_DP_TILES(W); ++t) {         \
                uint8x16_t Hcur = HF(sp_row + t * 16);          \
                uint8x16_t pred = VF(Hprev[t], Hcur);           \
                SPV_DP_ACCUM_TILE(W, pred, rp_row + t * 16);    \
                Hprev[t] = Hcur;                                \
            }                                                   \
        }                                                       \
    } while (0)

#define SPV_DP_HGEN_TILE(p) spv_dp_h_bil((p), xoffset, f0_x, f1_x)
#define SPV_DP_VGEN_TILE(a, b) spv_dp_v_bil((a), (b), yoffset, f0_y, f1_y)

// FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(W, H)
//
// Emits svt_aom_sub_pixel_variance{W}x{H}_neon_dotprod for the same 13
// (W, H) pairs as the NEON-side FUSED macro. Same dispatch structure --
// (0,0) short-circuit, yoffset==0 fast path, otherwise sliding-window
// outer dispatch on (xc, yc). The udot accumulator chain has uniform
// width across every (W, H) and needs no periodic-absorb.
//
// Shift is computed at compile time via svt_ctz((W) * (H)).
#define FUSED_SUBPEL_VARIANCE_WX_NEON_DOTPROD(W)                                                            \
    static NOINLINE unsigned int sub_pixel_variance_w##W##_neon_dotprod(const uint8_t* src,                 \
                                                                        int            src_stride,          \
                                                                        int            xoffset,             \
                                                                        int            yoffset,             \
                                                                        const uint8_t* ref,                 \
                                                                        int            ref_stride,          \
                                                                        int            h,                   \
                                                                        unsigned int*  sse) {                \
        int sum;                                                                                            \
        if (xoffset == 0 && yoffset == 0) {                                                                 \
            variance_##W##xh_neon_dotprod(src, src_stride, ref, ref_stride, h, sse, &sum);                  \
            return *sse - (uint32_t)(((int64_t)sum * sum) >> svt_ctz((W) * h));                             \
        }                                                                                                   \
        const uint8x16_t ones = vdupq_n_u8(1);                                                              \
        uint32x4_t       src_sum[SPV_DP_NACCUM(W)];                                                         \
        uint32x4_t       ref_sum[SPV_DP_NACCUM(W)];                                                         \
        uint32x4_t       sse_u32[SPV_DP_NACCUM(W)];                                                         \
        for (int i = 0; i < SPV_DP_NACCUM(W); ++i) {                                                        \
            src_sum[i] = vdupq_n_u32(0);                                                                    \
            ref_sum[i] = vdupq_n_u32(0);                                                                    \
            sse_u32[i] = vdupq_n_u32(0);                                                                    \
        }                                                                                                   \
                                                                                                            \
        if (yoffset == 0) {                                                                                 \
            switch (xoffset) {                                                                              \
            case 2:                                                                                         \
                SPV_DP_RUN_H(W, h, spv_dp_h_two);                                                           \
                break;                                                                                      \
            case 4:                                                                                         \
                SPV_DP_RUN_H(W, h, spv_dp_h_four);                                                          \
                break;                                                                                      \
            case 6:                                                                                         \
                SPV_DP_RUN_H(W, h, spv_dp_h_six);                                                           \
                break;                                                                                      \
            default: {                                                                                      \
                const uint8x8_t f0_x = vdup_n_u8((uint8_t)(8 - xoffset));                                   \
                const uint8x8_t f1_x = vdup_n_u8((uint8_t)xoffset);                                         \
                for (int r = 0; r < (h); ++r) {                                                             \
                    const uint8_t* sp_row = src + r * src_stride;                                           \
                    const uint8_t* rp_row = ref + r * ref_stride;                                           \
                    for (int t = 0; t < SPV_DP_TILES(W); ++t) {                                             \
                        uint8x16_t pred = spv_dp_h_bil(sp_row + t * 16, xoffset, f0_x, f1_x);               \
                        SPV_DP_ACCUM_TILE(W, pred, rp_row + t * 16);                                        \
                    }                                                                                       \
                }                                                                                           \
                break;                                                                                      \
            }                                                                                               \
            }                                                                                               \
        } else {                                                                                            \
            const uint8x8_t f0_x = vdup_n_u8((uint8_t)(8 - xoffset));                                       \
            const uint8x8_t f1_x = vdup_n_u8((uint8_t)xoffset);                                             \
            const uint8x8_t f0_y = vdup_n_u8((uint8_t)(8 - yoffset));                                       \
            const uint8x8_t f1_y = vdup_n_u8((uint8_t)yoffset);                                             \
            switch (SPV_KEY(xoffset, yoffset)) {                                                            \
            case SPV_KEY(0, 2):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_zero, spv_dp_v_two);                                           \
                break;                                                                                      \
            case SPV_KEY(0, 4):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_zero, spv_dp_v_four);                                          \
                break;                                                                                      \
            case SPV_KEY(0, 6):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_zero, spv_dp_v_six);                                           \
                break;                                                                                      \
            case SPV_KEY(0, 1):                                                                             \
            case SPV_KEY(0, 3):                                                                             \
            case SPV_KEY(0, 5):                                                                             \
            case SPV_KEY(0, 7):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_zero, SPV_DP_VGEN_TILE);                                       \
                break;                                                                                      \
            case SPV_KEY(2, 2):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_two, spv_dp_v_two);                                            \
                break;                                                                                      \
            case SPV_KEY(2, 4):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_two, spv_dp_v_four);                                           \
                break;                                                                                      \
            case SPV_KEY(2, 6):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_two, spv_dp_v_six);                                            \
                break;                                                                                      \
            case SPV_KEY(2, 1):                                                                             \
            case SPV_KEY(2, 3):                                                                             \
            case SPV_KEY(2, 5):                                                                             \
            case SPV_KEY(2, 7):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_two, SPV_DP_VGEN_TILE);                                        \
                break;                                                                                      \
            case SPV_KEY(4, 2):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_four, spv_dp_v_two);                                           \
                break;                                                                                      \
            case SPV_KEY(4, 4):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_four, spv_dp_v_four);                                          \
                break;                                                                                      \
            case SPV_KEY(4, 6):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_four, spv_dp_v_six);                                           \
                break;                                                                                      \
            case SPV_KEY(4, 1):                                                                             \
            case SPV_KEY(4, 3):                                                                             \
            case SPV_KEY(4, 5):                                                                             \
            case SPV_KEY(4, 7):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_four, SPV_DP_VGEN_TILE);                                       \
                break;                                                                                      \
            case SPV_KEY(6, 2):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_six, spv_dp_v_two);                                            \
                break;                                                                                      \
            case SPV_KEY(6, 4):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_six, spv_dp_v_four);                                           \
                break;                                                                                      \
            case SPV_KEY(6, 6):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_six, spv_dp_v_six);                                            \
                break;                                                                                      \
            case SPV_KEY(6, 1):                                                                             \
            case SPV_KEY(6, 3):                                                                             \
            case SPV_KEY(6, 5):                                                                             \
            case SPV_KEY(6, 7):                                                                             \
                SPV_DP_RUN_SW(W, h, spv_dp_h_six, SPV_DP_VGEN_TILE);                                        \
                break;                                                                                      \
            case SPV_KEY(1, 2):                                                                             \
            case SPV_KEY(3, 2):                                                                             \
            case SPV_KEY(5, 2):                                                                             \
            case SPV_KEY(7, 2):                                                                             \
                SPV_DP_RUN_SW(W, h, SPV_DP_HGEN_TILE, spv_dp_v_two);                                        \
                break;                                                                                      \
            case SPV_KEY(1, 4):                                                                             \
            case SPV_KEY(3, 4):                                                                             \
            case SPV_KEY(5, 4):                                                                             \
            case SPV_KEY(7, 4):                                                                             \
                SPV_DP_RUN_SW(W, h, SPV_DP_HGEN_TILE, spv_dp_v_four);                                       \
                break;                                                                                      \
            case SPV_KEY(1, 6):                                                                             \
            case SPV_KEY(3, 6):                                                                             \
            case SPV_KEY(5, 6):                                                                             \
            case SPV_KEY(7, 6):                                                                             \
                SPV_DP_RUN_SW(W, h, SPV_DP_HGEN_TILE, spv_dp_v_six);                                        \
                break;                                                                                      \
            default:                                                                                        \
                SPV_DP_RUN_SW(W, h, SPV_DP_HGEN_TILE, SPV_DP_VGEN_TILE);                                    \
                break;                                                                                      \
            }                                                                                               \
        }                                                                                                   \
                                                                                                            \
        uint32x4_t src_sum_v = src_sum[0];                                                                  \
        uint32x4_t ref_sum_v = ref_sum[0];                                                                  \
        uint32x4_t sse_sum   = sse_u32[0];                                                                  \
        for (int i = 1; i < SPV_DP_NACCUM(W); ++i) {                                                        \
            src_sum_v = vaddq_u32(src_sum_v, src_sum[i]);                                                   \
            ref_sum_v = vaddq_u32(ref_sum_v, ref_sum[i]);                                                   \
            sse_sum   = vaddq_u32(sse_sum, sse_u32[i]);                                                     \
        }                                                                                                   \
        *sse               = vaddvq_u32(sse_sum);                                                           \
        int32x4_t sum_diff = vsubq_s32(vreinterpretq_s32_u32(src_sum_v), vreinterpretq_s32_u32(ref_sum_v)); \
        sum                = vaddvq_s32(sum_diff);                                                          \
        return *sse - (uint32_t)(((int64_t)sum * sum) >> svt_ctz((W) * h));                                 \
    }

FUSED_SUBPEL_VARIANCE_WX_NEON_DOTPROD(16)
FUSED_SUBPEL_VARIANCE_WX_NEON_DOTPROD(32)
FUSED_SUBPEL_VARIANCE_WX_NEON_DOTPROD(64)
FUSED_SUBPEL_VARIANCE_WX_NEON_DOTPROD(128)

#define FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(W, H)                                                                 \
    unsigned int svt_aom_sub_pixel_variance##W##x##H##_neon_dotprod(const uint8_t* src,                              \
                                                                    int            src_stride,                       \
                                                                    int            xoffset,                          \
                                                                    int            yoffset,                          \
                                                                    const uint8_t* ref,                              \
                                                                    int            ref_stride,                       \
                                                                    unsigned int*  sse) {                             \
        return sub_pixel_variance_w##W##_neon_dotprod(src, src_stride, xoffset, yoffset, ref, ref_stride, (H), sse); \
    }

FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 4)
FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 8)
FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 16)
FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 32)
FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 64)
FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 8)
FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 16)
FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 32)
FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 64)
FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(64, 16)
FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(64, 32)
FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(64, 64)
FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(64, 128)
FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(128, 64)
FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD(128, 128)

#undef FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD
#undef FUSED_SUBPEL_VARIANCE_WX_NEON_DOTPROD
#undef SPV_DP_VGEN_TILE
#undef SPV_DP_HGEN_TILE
#undef SPV_DP_RUN_SW
#undef SPV_DP_RUN_H
#undef SPV_DP_ACCUM_TILE
#undef SPV_DP_NACCUM
#undef SPV_DP_TILES
#undef SPV_KEY

unsigned int svt_aom_mse16x16_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride) {
    uint32x4_t sse_u32[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};

    int h = 16;
    do {
        uint8x16_t s0 = vld1q_u8(src);
        uint8x16_t s1 = vld1q_u8(src + src_stride);
        uint8x16_t r0 = vld1q_u8(ref);
        uint8x16_t r1 = vld1q_u8(ref + ref_stride);

        uint8x16_t abs_diff0 = vabdq_u8(s0, r0);
        uint8x16_t abs_diff1 = vabdq_u8(s1, r1);

        sse_u32[0] = vdotq_u32(sse_u32[0], abs_diff0, abs_diff0);
        sse_u32[1] = vdotq_u32(sse_u32[1], abs_diff1, abs_diff1);

        src += 2 * src_stride;
        ref += 2 * ref_stride;
        h -= 2;
    } while (h != 0);

    unsigned int sse = vaddvq_u32(vaddq_u32(sse_u32[0], sse_u32[1]));
    return sse;
}
