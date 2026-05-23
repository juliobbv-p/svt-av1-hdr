/*
* Copyright (c) 2023, Alliance for Open Media. All rights reserved
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
#include "mem_neon.h"
#include "sum_neon.h"
#include "var_filter_neon.h"

#ifdef __clang__
#define DISABLE_LOOP_UNROLL 1
#else
#define DISABLE_LOOP_UNROLL 0
#endif

static inline void variance_4xh_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int h,
                                     uint32_t* sse, int* sum) {
    int16x8_t  sum_s16 = vdupq_n_s16(0);
    uint32x4_t sse_u32 = vdupq_n_u32(0);

    // Number of rows we can process before 'sum_s16' overflows:
    // 32767 / 255 ~= 128, but we use an 8-wide accumulator; so 256 4-wide rows.
    assert(h <= 256);

    int i = h;
    do {
        uint8x8_t s    = load_u8_4x2(src, src_stride);
        uint8x8_t r    = load_u8_4x2(ref, ref_stride);
        int16x8_t diff = vreinterpretq_s16_u16(vsubl_u8(s, r));

        sum_s16 = vaddq_s16(sum_s16, diff);

        uint16x8_t sq = vreinterpretq_u16_s16(vmulq_s16(diff, diff));
        sse_u32       = vpadalq_u16(sse_u32, sq);

        src += 2 * src_stride;
        ref += 2 * ref_stride;
        i -= 2;
    } while (i != 0);

    *sum = vaddlvq_s16(sum_s16);
    *sse = vaddvq_u32(sse_u32);
}

static inline void variance_8xh_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int h,
                                     uint32_t* sse, int* sum) {
    int16x8_t  sum_s16[2] = {vdupq_n_s16(0), vdupq_n_s16(0)};
    uint32x4_t sse_u32[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};

    assert(h <= 128);
    assert((h & 1) == 0);

    int i = h;
#if DISABLE_LOOP_UNROLL
#pragma clang loop unroll(disable)
#endif
    do {
        uint8x8_t s0 = vld1_u8(src);
        uint8x8_t r0 = vld1_u8(ref);
        uint8x8_t s1 = vld1_u8(src + src_stride);
        uint8x8_t r1 = vld1_u8(ref + ref_stride);

        int16x8_t diff0 = vreinterpretq_s16_u16(vsubl_u8(s0, r0));
        int16x8_t diff1 = vreinterpretq_s16_u16(vsubl_u8(s1, r1));

        sum_s16[0] = vaddq_s16(sum_s16[0], diff0);
        sum_s16[1] = vaddq_s16(sum_s16[1], diff1);

        uint16x8_t sq0 = vreinterpretq_u16_s16(vmulq_s16(diff0, diff0));
        uint16x8_t sq1 = vreinterpretq_u16_s16(vmulq_s16(diff1, diff1));

        sse_u32[0] = vpadalq_u16(sse_u32[0], sq0);
        sse_u32[1] = vpadalq_u16(sse_u32[1], sq1);

        src += 2 * src_stride;
        ref += 2 * ref_stride;
        i -= 2;
    } while (i != 0);

    *sum = vaddlvq_s16(vaddq_s16(sum_s16[0], sum_s16[1]));
    *sse = vaddvq_u32(vaddq_u32(sse_u32[0], sse_u32[1]));
}

static inline void variance_16xh_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int h,
                                      uint32_t* sse, int* sum) {
    int16x8_t  sum_s16[2] = {vdupq_n_s16(0), vdupq_n_s16(0)};
    uint32x4_t sse_u32[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};

    // Number of rows we can process before 'sum_s16' accumulators overflow:
    // 32767 / 255 ~= 128, so 128 16-wide rows.
    assert(h <= 128);
    assert((h & 1) == 0);

    int i = h;
#if DISABLE_LOOP_UNROLL
#pragma clang loop unroll(disable)
#endif
    do {
        uint8x16_t s0 = vld1q_u8(src);
        uint8x16_t r0 = vld1q_u8(ref);
        uint8x16_t s1 = vld1q_u8(src + src_stride);
        uint8x16_t r1 = vld1q_u8(ref + ref_stride);

        int16x8_t diff0_l = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s0), vget_low_u8(r0)));
        int16x8_t diff0_h = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s0), vget_high_u8(r0)));
        int16x8_t diff1_l = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s1), vget_low_u8(r1)));
        int16x8_t diff1_h = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s1), vget_high_u8(r1)));

        sum_s16[0] = vaddq_s16(sum_s16[0], diff0_l);
        sum_s16[1] = vaddq_s16(sum_s16[1], diff0_h);
        sum_s16[0] = vaddq_s16(sum_s16[0], diff1_l);
        sum_s16[1] = vaddq_s16(sum_s16[1], diff1_h);

        uint16x8_t sq0_l = vreinterpretq_u16_s16(vmulq_s16(diff0_l, diff0_l));
        uint16x8_t sq0_h = vreinterpretq_u16_s16(vmulq_s16(diff0_h, diff0_h));
        uint16x8_t sq1_l = vreinterpretq_u16_s16(vmulq_s16(diff1_l, diff1_l));
        uint16x8_t sq1_h = vreinterpretq_u16_s16(vmulq_s16(diff1_h, diff1_h));

        sse_u32[0] = vpadalq_u16(sse_u32[0], sq0_l);
        sse_u32[1] = vpadalq_u16(sse_u32[1], sq0_h);
        sse_u32[0] = vpadalq_u16(sse_u32[0], sq1_l);
        sse_u32[1] = vpadalq_u16(sse_u32[1], sq1_h);

        src += 2 * src_stride;
        ref += 2 * ref_stride;
        i -= 2;
    } while (i != 0);

    *sum = vaddlvq_s16(vaddq_s16(sum_s16[0], sum_s16[1]));
    *sse = vaddvq_u32(vaddq_u32(sse_u32[0], sse_u32[1]));
}

static inline void variance_large_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int w,
                                       int h, int h_limit, uint32_t* sse, int* sum) {
    int32x4_t sum_s32 = vdupq_n_s32(0);

    uint32x4_t sse_u32[4] = {
        vdupq_n_u32(0),
        vdupq_n_u32(0),
        vdupq_n_u32(0),
        vdupq_n_u32(0),
    };

    // 'h_limit' is the number of 'w'-width rows we can process before our 16-bit
    // accumulator overflows. After hitting this limit we accumulate into 32-bit
    // elements.
    int h_tmp = h > h_limit ? h_limit : h;

    int i = 0;
#if DISABLE_LOOP_UNROLL
#pragma clang loop unroll(disable)
#endif
    do {
        int16x8_t sum_s16_0 = vdupq_n_s16(0);
        int16x8_t sum_s16_1 = vdupq_n_s16(0);

        do {
            int j = 0;
            int t = 0;
            do {
                uint8x16_t s = vld1q_u8(src + j);
                uint8x16_t r = vld1q_u8(ref + j);

                const int16x8_t diff_l = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s), vget_low_u8(r)));
                const int16x8_t diff_h = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s), vget_high_u8(r)));

                sum_s16_0 = vaddq_s16(sum_s16_0, diff_l);
                sum_s16_1 = vaddq_s16(sum_s16_1, diff_h);

                uint16x8_t sq_l = vreinterpretq_u16_s16(vmulq_s16(diff_l, diff_l));
                uint16x8_t sq_h = vreinterpretq_u16_s16(vmulq_s16(diff_h, diff_h));
                sse_u32[t]      = vpadalq_u16(sse_u32[t], sq_l);
                sse_u32[t]      = vpadalq_u16(sse_u32[t], sq_h);

                j += 16;
                t = (t + 1) & 3;
            } while (j < w);

            src += src_stride;
            ref += ref_stride;
            i++;
        } while (i < h_tmp);

        sum_s32 = vpadalq_s16(sum_s32, sum_s16_0);
        sum_s32 = vpadalq_s16(sum_s32, sum_s16_1);

        h_tmp += h_limit;
    } while (i < h);

    *sum = vaddvq_s32(sum_s32);

    uint32x4_t sse_sum = vaddq_u32(vaddq_u32(sse_u32[0], sse_u32[1]), vaddq_u32(sse_u32[2], sse_u32[3]));
    *sse               = vaddvq_u32(sse_sum);
}

static inline void variance_32xh_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int h,
                                      uint32_t* sse, int* sum) {
    variance_large_neon(src, src_stride, ref, ref_stride, 32, h, 64, sse, sum);
}

static inline void variance_64xh_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int h,
                                      uint32_t* sse, int* sum) {
    variance_large_neon(src, src_stride, ref, ref_stride, 64, h, 32, sse, sum);
}

static inline void variance_128xh_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int h,
                                       uint32_t* sse, int* sum) {
    variance_large_neon(src, src_stride, ref, ref_stride, 128, h, 16, sse, sum);
}

#define VARIANCE_WXH_NEON(w, h, shift)                                                               \
    unsigned int svt_aom_variance##w##x##h##_neon(                                                   \
        const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, unsigned int* sse) { \
        int sum;                                                                                     \
        variance_##w##xh_neon(src, src_stride, ref, ref_stride, h, sse, &sum);                       \
        return *sse - (uint32_t)(((int64_t)sum * sum) >> shift);                                     \
    }

VARIANCE_WXH_NEON(4, 4, 4)
VARIANCE_WXH_NEON(4, 8, 5)
VARIANCE_WXH_NEON(4, 16, 6)

VARIANCE_WXH_NEON(8, 4, 5)
VARIANCE_WXH_NEON(8, 8, 6)
VARIANCE_WXH_NEON(8, 16, 7)
VARIANCE_WXH_NEON(8, 32, 8)

VARIANCE_WXH_NEON(16, 4, 6)
VARIANCE_WXH_NEON(16, 8, 7)
VARIANCE_WXH_NEON(16, 16, 8)
VARIANCE_WXH_NEON(16, 32, 9)
VARIANCE_WXH_NEON(16, 64, 10)

VARIANCE_WXH_NEON(32, 8, 8)
VARIANCE_WXH_NEON(32, 16, 9)
VARIANCE_WXH_NEON(32, 32, 10)
VARIANCE_WXH_NEON(32, 64, 11)

VARIANCE_WXH_NEON(64, 16, 10)
VARIANCE_WXH_NEON(64, 32, 11)
VARIANCE_WXH_NEON(64, 64, 12)
VARIANCE_WXH_NEON(64, 128, 13)

VARIANCE_WXH_NEON(128, 64, 13)
VARIANCE_WXH_NEON(128, 128, 14)

#undef VARIANCE_WXH_NEON

// =============================================================================
// Small-block sub_pixel_variance helpers (4xH)
// =============================================================================
#define SUBPEL_VARIANCE_4XH_NEON(h, padding)                                         \
    unsigned int svt_aom_sub_pixel_variance4##x##h##_neon(const uint8_t* src,        \
                                                          int            src_stride, \
                                                          int            xoffset,    \
                                                          int            yoffset,    \
                                                          const uint8_t* ref,        \
                                                          int            ref_stride, \
                                                          uint32_t*      sse) {           \
        uint8_t tmp0[4 * (h + padding)];                                             \
        uint8_t tmp1[4 * h];                                                         \
        var_filter_block2d_bil_w4(src, tmp0, src_stride, 1, (h + padding), xoffset); \
        var_filter_block2d_bil_w4(tmp0, tmp1, 4, 4, h, yoffset);                     \
        return svt_aom_variance4##x##h##_neon(tmp1, 4, ref, ref_stride, sse);        \
    }

SUBPEL_VARIANCE_4XH_NEON(4, 2)
SUBPEL_VARIANCE_4XH_NEON(8, 2)
SUBPEL_VARIANCE_4XH_NEON(16, 2)

#undef SUBPEL_VARIANCE_4XH_NEON

static inline uint8x8_t spv_w8_h_bil(const uint8_t* p, int off, uint8x8_t f0, uint8x8_t f1) {
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

static inline uint8x8_t spv_w8_v_bil(uint8x8_t a, uint8x8_t b, int off, uint8x8_t f0, uint8x8_t f1) {
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

static inline uint8x8_t spv_w8_h_0(const uint8_t* p) {
    return vld1_u8(p);
}

static inline uint8x8_t spv_w8_h_2(const uint8_t* p) {
    uint8x8_t s0 = vld1_u8(p);
    uint8x8_t s1 = vld1_u8(p + 1);
    return vrhadd_u8(s0, vhadd_u8(s0, s1));
}

static inline uint8x8_t spv_w8_h_4(const uint8_t* p) {
    return vrhadd_u8(vld1_u8(p), vld1_u8(p + 1));
}

static inline uint8x8_t spv_w8_h_6(const uint8_t* p) {
    uint8x8_t s0 = vld1_u8(p);
    uint8x8_t s1 = vld1_u8(p + 1);
    return vrhadd_u8(s1, vhadd_u8(s0, s1));
}

static inline uint8x8_t spv_w8_v_2(uint8x8_t a, uint8x8_t b) {
    return vrhadd_u8(a, vhadd_u8(a, b));
}

static inline uint8x8_t spv_w8_v_4(uint8x8_t a, uint8x8_t b) {
    return vrhadd_u8(a, b);
}

static inline uint8x8_t spv_w8_v_6(uint8x8_t a, uint8x8_t b) {
    return vrhadd_u8(b, vhadd_u8(a, b));
}

#define SPV_KEY(xoff, yoff) (((xoff) << 3) | (yoff))

#define SPV_W8_ACCUM(pred, refptr)                       \
    do {                                                 \
        uint8x8_t  _r  = vld1_u8(refptr);                \
        uint8x8_t  _ad = vabd_u8((pred), _r);            \
        uint16x8_t _sq = vmull_u8(_ad, _ad);             \
        sse_u32        = vpadalq_u16(sse_u32, _sq);      \
        sum_src_u16    = vpadal_u8(sum_src_u16, (pred)); \
        sum_ref_u16    = vpadal_u8(sum_ref_u16, _r);     \
    } while (0)

#define SPV_W8_RUN_H(H, HF)                               \
    do {                                                  \
        for (int r = 0; r < (H); ++r) {                   \
            const uint8_t* sp_row = src + r * src_stride; \
            const uint8_t* rp_row = ref + r * ref_stride; \
            uint8x8_t      pred   = HF(sp_row);           \
            SPV_W8_ACCUM(pred, rp_row);                   \
        }                                                 \
    } while (0)

#define SPV_W8_RUN_SW(H, HF, VF)                                \
    do {                                                        \
        uint8x8_t Hprev = HF(src);                              \
        for (int r = 0; r < (H); ++r) {                         \
            const uint8_t* sp_row = src + (r + 1) * src_stride; \
            const uint8_t* rp_row = ref + r * ref_stride;       \
            uint8x8_t      Hcur   = HF(sp_row);                 \
            uint8x8_t      pred   = VF(Hprev, Hcur);            \
            SPV_W8_ACCUM(pred, rp_row);                         \
            Hprev = Hcur;                                       \
        }                                                       \
    } while (0)

#define SPV_W8_HGEN(p) spv_w8_h_bil((p), xoffset, f0_x, f1_x)
#define SPV_W8_VGEN(a, b) spv_w8_v_bil((a), (b), yoffset, f0_y, f1_y)

static NOINLINE unsigned int sub_pixel_variance_w8_neon(const uint8_t* src, int src_stride, int xoffset, int yoffset,
                                                        const uint8_t* ref, int ref_stride, int h, unsigned int* sse) {
    int sum;
    if (xoffset == 0 && yoffset == 0) {
        variance_8xh_neon(src, src_stride, ref, ref_stride, h, sse, &sum);
        return *sse - (uint32_t)(((int64_t)sum * sum) >> svt_ctz(8 * h));
    }
    uint16x4_t sum_src_u16 = vdup_n_u16(0);
    uint16x4_t sum_ref_u16 = vdup_n_u16(0);
    uint32x4_t sse_u32     = vdupq_n_u32(0);

    if (yoffset == 0) {
        switch (xoffset) {
        case 2:
            SPV_W8_RUN_H(h, spv_w8_h_2);
            break;
        case 4:
            SPV_W8_RUN_H(h, spv_w8_h_4);
            break;
        case 6:
            SPV_W8_RUN_H(h, spv_w8_h_6);
            break;
        default: {
            const uint8x8_t f0_x = vdup_n_u8((uint8_t)(8 - xoffset));
            const uint8x8_t f1_x = vdup_n_u8((uint8_t)xoffset);
            SPV_W8_RUN_H(h, SPV_W8_HGEN);
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
            SPV_W8_RUN_SW(h, spv_w8_h_0, spv_w8_v_2);
            break;
        case SPV_KEY(0, 4):
            SPV_W8_RUN_SW(h, spv_w8_h_0, spv_w8_v_4);
            break;
        case SPV_KEY(0, 6):
            SPV_W8_RUN_SW(h, spv_w8_h_0, spv_w8_v_6);
            break;
        case SPV_KEY(0, 1):
        case SPV_KEY(0, 3):
        case SPV_KEY(0, 5):
        case SPV_KEY(0, 7):
            SPV_W8_RUN_SW(h, spv_w8_h_0, SPV_W8_VGEN);
            break;
        case SPV_KEY(2, 2):
            SPV_W8_RUN_SW(h, spv_w8_h_2, spv_w8_v_2);
            break;
        case SPV_KEY(2, 4):
            SPV_W8_RUN_SW(h, spv_w8_h_2, spv_w8_v_4);
            break;
        case SPV_KEY(2, 6):
            SPV_W8_RUN_SW(h, spv_w8_h_2, spv_w8_v_6);
            break;
        case SPV_KEY(2, 1):
        case SPV_KEY(2, 3):
        case SPV_KEY(2, 5):
        case SPV_KEY(2, 7):
            SPV_W8_RUN_SW(h, spv_w8_h_2, SPV_W8_VGEN);
            break;
        case SPV_KEY(4, 2):
            SPV_W8_RUN_SW(h, spv_w8_h_4, spv_w8_v_2);
            break;
        case SPV_KEY(4, 4):
            SPV_W8_RUN_SW(h, spv_w8_h_4, spv_w8_v_4);
            break;
        case SPV_KEY(4, 6):
            SPV_W8_RUN_SW(h, spv_w8_h_4, spv_w8_v_6);
            break;
        case SPV_KEY(4, 1):
        case SPV_KEY(4, 3):
        case SPV_KEY(4, 5):
        case SPV_KEY(4, 7):
            SPV_W8_RUN_SW(h, spv_w8_h_4, SPV_W8_VGEN);
            break;
        case SPV_KEY(6, 2):
            SPV_W8_RUN_SW(h, spv_w8_h_6, spv_w8_v_2);
            break;
        case SPV_KEY(6, 4):
            SPV_W8_RUN_SW(h, spv_w8_h_6, spv_w8_v_4);
            break;
        case SPV_KEY(6, 6):
            SPV_W8_RUN_SW(h, spv_w8_h_6, spv_w8_v_6);
            break;
        case SPV_KEY(6, 1):
        case SPV_KEY(6, 3):
        case SPV_KEY(6, 5):
        case SPV_KEY(6, 7):
            SPV_W8_RUN_SW(h, spv_w8_h_6, SPV_W8_VGEN);
            break;
        case SPV_KEY(1, 2):
        case SPV_KEY(3, 2):
        case SPV_KEY(5, 2):
        case SPV_KEY(7, 2):
            SPV_W8_RUN_SW(h, SPV_W8_HGEN, spv_w8_v_2);
            break;
        case SPV_KEY(1, 4):
        case SPV_KEY(3, 4):
        case SPV_KEY(5, 4):
        case SPV_KEY(7, 4):
            SPV_W8_RUN_SW(h, SPV_W8_HGEN, spv_w8_v_4);
            break;
        case SPV_KEY(1, 6):
        case SPV_KEY(3, 6):
        case SPV_KEY(5, 6):
        case SPV_KEY(7, 6):
            SPV_W8_RUN_SW(h, SPV_W8_HGEN, spv_w8_v_6);
            break;
        default:
            SPV_W8_RUN_SW(h, SPV_W8_HGEN, SPV_W8_VGEN);
            break;
        }
    }

    sum  = (int)(uint32_t)vaddv_u16(sum_src_u16) - (int)(uint32_t)vaddv_u16(sum_ref_u16);
    *sse = vaddvq_u32(sse_u32);
    return *sse - (uint32_t)(((int64_t)sum * sum) >> svt_ctz(8 * h));
}

#define FUSED_SUBPEL_VARIANCE_W8H_NEON(H)                                                                \
    unsigned int svt_aom_sub_pixel_variance8x##H##_neon(const uint8_t* src,                              \
                                                        int            src_stride,                       \
                                                        int            xoffset,                          \
                                                        int            yoffset,                          \
                                                        const uint8_t* ref,                              \
                                                        int            ref_stride,                       \
                                                        unsigned int*  sse) {                             \
        return sub_pixel_variance_w8_neon(src, src_stride, xoffset, yoffset, ref, ref_stride, (H), sse); \
    }

FUSED_SUBPEL_VARIANCE_W8H_NEON(4)
FUSED_SUBPEL_VARIANCE_W8H_NEON(8)
FUSED_SUBPEL_VARIANCE_W8H_NEON(16)
FUSED_SUBPEL_VARIANCE_W8H_NEON(32)

#undef FUSED_SUBPEL_VARIANCE_W8H_NEON
#undef SPV_W8_VGEN
#undef SPV_W8_HGEN
#undef SPV_W8_RUN_SW
#undef SPV_W8_RUN_H
#undef SPV_W8_ACCUM

// =============================================================================
// Fused-loop sub_pixel_variance for {16,32,64,128} x H
// =============================================================================
//
// One macro -- FUSED_SUBPEL_VARIANCE_WXH_NEON(W, H) -- emits
// svt_aom_sub_pixel_variance{W}x{H}_neon for every (W, H) the codebase
// currently uses with W ∈ {16, 32, 64, 128}. The emitted kernel folds the
// upstream 3-pass (H-filter → V-filter → variance) chain into a single row
// loop that holds the H-filter output in NEON registers via a sliding-
// window state (`Hprev[t]`) and consumes it immediately by the V-filter +
// SSE/sum accumulation. This eliminates the upstream tmp[] stack frames
// (up to W·(H+1) bytes for W=128) and the load-after-store dependency
// chain on Apple Silicon's store-to-load-forwarding path.
//
// PER-PIXEL HELPER FAMILY (spv_h_* / spv_v_*)
//
// `spv_h_*` and `spv_v_*` are 16-byte tile primitives shared by every
// (W, H) instantiation. They return a uint8x16_t of filtered pixels for
// one tile column of the output and chain together horizontally + vertically
// to produce the final pred byte.
//
// Bit-exact with the upstream 2-pass var_filter_block2d_* + variance path:
//
//   xoff/yoff == 4  -> (a + b + 1) >> 1                  ≡ vrhaddq_u8(a, b)
//   xoff/yoff == 2  -> (3a + b + 2) >> 2                 ≡ vrhaddq_u8(a, vhaddq_u8(a, b))
//   xoff/yoff == 6  -> (a + 3b + 2) >> 2                 ≡ vrhaddq_u8(b, vhaddq_u8(a, b))
//   other (qpel)    -> (f0*a + f1*b + 4) >> 3            ≡ vrshrn_n_u16(vmlal_u8(vmull_u8(a,f0),b,f1), 3)
//
// The urhadd-chain identity uses the inner non-rounding vhaddq_u8 (not
// vrhaddq_u8). vrhaddq(a, vrhaddq(a,b)) biases high by 1 when (a+b) is odd
// and 3a+b ≡ 1 mod 4 -- not bit-exact.

// spv_h_bil / spv_v_bil: generic bilinear for any (sub)pixel offset in {1..7}.
// Used by the OTHER offset class (xoff/yoff ∈ {1, 3, 5, 7}). Includes the
// urhadd-chain fast paths for off ∈ {2, 6} so the same primitive is safe to
// call from the hpel / qpel hot paths in dispatch order.
static inline uint8x16_t spv_h_bil(const uint8_t* p, int off, uint8x8_t f0, uint8x8_t f1) {
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

static inline uint8x16_t spv_v_bil(uint8x16_t a, uint8x16_t b, int off, uint8x8_t f0, uint8x8_t f1) {
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

// Per-offset-class specialised hpel / qpel helpers. The per-class branch
// (off == 0/2/4/6) is lifted out of the row loop into the outer dispatch
// switch, so each specialised inner loop has zero per-iteration offset
// branches.
static inline uint8x16_t spv_h_0(const uint8_t* p) {
    return vld1q_u8(p);
}

static inline uint8x16_t spv_h_2(const uint8_t* p) {
    uint8x16_t s0 = vld1q_u8(p);
    uint8x16_t s1 = vld1q_u8(p + 1);
    return vrhaddq_u8(s0, vhaddq_u8(s0, s1));
}

static inline uint8x16_t spv_h_4(const uint8_t* p) {
    return vrhaddq_u8(vld1q_u8(p), vld1q_u8(p + 1));
}

static inline uint8x16_t spv_h_6(const uint8_t* p) {
    uint8x16_t s0 = vld1q_u8(p);
    uint8x16_t s1 = vld1q_u8(p + 1);
    return vrhaddq_u8(s1, vhaddq_u8(s0, s1));
}

static inline uint8x16_t spv_v_2(uint8x16_t a, uint8x16_t b) {
    return vrhaddq_u8(a, vhaddq_u8(a, b));
}

static inline uint8x16_t spv_v_4(uint8x16_t a, uint8x16_t b) {
    return vrhaddq_u8(a, b);
}

static inline uint8x16_t spv_v_6(uint8x16_t a, uint8x16_t b) {
    return vrhaddq_u8(b, vhaddq_u8(a, b));
}

// SPV_TILES(W): number of 16-byte tile columns per row. 1 / 2 / 4 / 8.
//
// SPV_ABSORB_INTERVAL(W): how many rows of int16 sum we can accumulate
// before risking int16 overflow per lane. Each row contributes up to
// TILES_PER_ROW × 255 per lane (16-byte tile splits diff into 8-lane
// halves; lane gets exactly one diff per tile per row). Budget per lane:
// 32767 / 255 ≈ 128 row-tile-pairs. Power-of-two for compile-time div.
#define SPV_TILES(W) ((W) / 16)
#define SPV_NACCUM(W) ((SPV_TILES(W) >= 4) ? 4 : SPV_TILES(W))
#define SPV_ABSORB_INTERVAL(W) (128 / SPV_TILES(W))

// Per-tile diff → sum/sse accumulator. Reads `pred` (uint8x16_t tile) and
// 16 bytes of ref at `refptr`. Updates surrounding-scope sum_lo / sum_hi
// (int16x8_t) and sse_s32_{0..3} (int32x4_t partial accumulators) using the
// upstream subl+mlal accumulator pattern that the bit-exact match requires.
#define SPV_ACCUM_TILE(W, pred, refptr)                                   \
    do {                                                                  \
        uint8x16_t _r   = vld1q_u8(refptr);                               \
        uint8x16_t _ad  = vabdq_u8((pred), _r);                           \
        uint16x8_t _sl  = vmull_u8(vget_low_u8(_ad), vget_low_u8(_ad));   \
        uint16x8_t _sh  = vmull_u8(vget_high_u8(_ad), vget_high_u8(_ad)); \
        const int  _i   = t & (SPV_NACCUM(W) - 1);                        \
        sse_u32_lo[_i]  = vpadalq_u16(sse_u32_lo[_i], _sl);               \
        sse_u32_hi[_i]  = vpadalq_u16(sse_u32_hi[_i], _sh);               \
        sum_src_u16[_i] = vpadalq_u8(sum_src_u16[_i], (pred));            \
        sum_ref_u16[_i] = vpadalq_u8(sum_ref_u16[_i], _r);                \
    } while (0)

// Periodic absorb of the int16 sum accumulators into the int32 sum
// accumulators. The condition `SPV_ABSORB_INTERVAL(W) < (H)` is compile-
// time, so the entire branch evaporates for any (W, H) with H below the
// budget (e.g. every W=16 case, and every W=32 case with H ≤ 64).
#define SPV_MAYBE_ABSORB(W, H, R)                                                        \
    do {                                                                                 \
        if (SPV_ABSORB_INTERVAL(W) < (H) && (((R) + 1) % SPV_ABSORB_INTERVAL(W)) == 0) { \
            for (int _i = 0; _i < SPV_NACCUM(W); ++_i) {                                 \
                sum_src_u32[_i] = vpadalq_u16(sum_src_u32[_i], sum_src_u16[_i]);         \
                sum_ref_u32[_i] = vpadalq_u16(sum_ref_u32[_i], sum_ref_u16[_i]);         \
                sum_src_u16[_i] = vdupq_n_u16(0);                                        \
                sum_ref_u16[_i] = vdupq_n_u16(0);                                        \
            }                                                                            \
        }                                                                                \
    } while (0)

// H-only row body (yoffset == 0 path). One pass through src for every output
// row; no Hprev sliding-window state required.
#define SPV_RUN_H(W, H, HF)                               \
    do {                                                  \
        for (int r = 0; r < (H); ++r) {                   \
            const uint8_t* sp_row = src + r * src_stride; \
            const uint8_t* rp_row = ref + r * ref_stride; \
            for (int t = 0; t < SPV_TILES(W); ++t) {      \
                uint8x16_t pred = HF(sp_row + t * 16);    \
                SPV_ACCUM_TILE(W, pred, rp_row + t * 16); \
            }                                             \
            SPV_MAYBE_ABSORB(W, H, r);                    \
        }                                                 \
    } while (0)

// Sliding-window row body (yoffset != 0 path). Hprev[t] is one q-reg per
// tile column; on entry it holds H-filter(src row 0, tile t). Each output
// row computes Hcur, blends, accumulates, and rotates Hprev[t] <- Hcur.
#define SPV_RUN_SW(W, H, HF, VF)                                \
    do {                                                        \
        uint8x16_t Hprev[SPV_TILES(W)];                         \
        for (int t = 0; t < SPV_TILES(W); ++t) {                \
            Hprev[t] = HF(src + t * 16);                        \
        }                                                       \
        for (int r = 0; r < (H); ++r) {                         \
            const uint8_t* sp_row = src + (r + 1) * src_stride; \
            const uint8_t* rp_row = ref + r * ref_stride;       \
            for (int t = 0; t < SPV_TILES(W); ++t) {            \
                uint8x16_t Hcur = HF(sp_row + t * 16);          \
                uint8x16_t pred = VF(Hprev[t], Hcur);           \
                SPV_ACCUM_TILE(W, pred, rp_row + t * 16);       \
                Hprev[t] = Hcur;                                \
            }                                                   \
            SPV_MAYBE_ABSORB(W, H, r);                          \
        }                                                       \
    } while (0)

// Generic eighth-pel helpers that close over xoffset / yoffset / f0_x / f1_x
// / f0_y / f1_y in the surrounding scope. The compiler inlines these and
// folds the off==2/6 fast-path branches against the closed-over constant.
#define SPV_HGEN_TILE(p) spv_h_bil((p), xoffset, f0_x, f1_x)
#define SPV_VGEN_TILE(a, b) spv_v_bil((a), (b), yoffset, f0_y, f1_y)

// FUSED_SUBPEL_VARIANCE_WXH_NEON(W, H)
//
// Emits svt_aom_sub_pixel_variance{W}x{H}_neon for the given (W, H) ∈
// {(16,16), (16,32), (16,64), (32,8), (32,16), (32,32), (32,64),
//  (64,16), (64,32), (64,64), (64,128), (128,64), (128,128)}.
//
// The macro body is structured as:
//   1. (0, 0) short-circuit to the integer-variance kernel.
//   2. yoffset == 0 fast path -- pure H-filter loop, no sliding window.
//   3. yoffset != 0 path -- sliding-window H+V loop, outer dispatch on
//      (xoffset class, yoffset class) via xc*5 + yc.
// Each branch uses the spv_h_* / spv_v_* tile helpers above with per-tile
// SPV_ACCUM_TILE accumulation into int16 sum + 4-way int32 sse partials.
// The int16 sum is periodically absorbed into int32 sum partials (no-op
// for narrow / shallow cases by constant folding).
#define FUSED_SUBPEL_VARIANCE_WX_NEON(W)                                                    \
    static NOINLINE unsigned int sub_pixel_variance_w##W##_neon(const uint8_t* src,         \
                                                                int            src_stride,  \
                                                                int            xoffset,     \
                                                                int            yoffset,     \
                                                                const uint8_t* ref,         \
                                                                int            ref_stride,  \
                                                                int            h,           \
                                                                unsigned int*  sse) {        \
        int sum;                                                                            \
        if (xoffset == 0 && yoffset == 0) {                                                 \
            variance_##W##xh_neon(src, src_stride, ref, ref_stride, h, sse, &sum);          \
            return *sse - (uint32_t)(((int64_t)sum * sum) >> svt_ctz((W) * h));             \
        }                                                                                   \
        uint16x8_t sum_src_u16[SPV_NACCUM(W)];                                              \
        uint16x8_t sum_ref_u16[SPV_NACCUM(W)];                                              \
        uint32x4_t sum_src_u32[SPV_NACCUM(W)];                                              \
        uint32x4_t sum_ref_u32[SPV_NACCUM(W)];                                              \
        uint32x4_t sse_u32_lo[SPV_NACCUM(W)];                                               \
        uint32x4_t sse_u32_hi[SPV_NACCUM(W)];                                               \
        for (int i = 0; i < SPV_NACCUM(W); ++i) {                                           \
            sum_src_u16[i] = vdupq_n_u16(0);                                                \
            sum_ref_u16[i] = vdupq_n_u16(0);                                                \
            sum_src_u32[i] = vdupq_n_u32(0);                                                \
            sum_ref_u32[i] = vdupq_n_u32(0);                                                \
            sse_u32_lo[i]  = vdupq_n_u32(0);                                                \
            sse_u32_hi[i]  = vdupq_n_u32(0);                                                \
        }                                                                                   \
                                                                                            \
        if (yoffset == 0) {                                                                 \
            switch (xoffset) {                                                              \
            case 2:                                                                         \
                SPV_RUN_H(W, h, spv_h_2);                                                   \
                break;                                                                      \
            case 4:                                                                         \
                SPV_RUN_H(W, h, spv_h_4);                                                   \
                break;                                                                      \
            case 6:                                                                         \
                SPV_RUN_H(W, h, spv_h_6);                                                   \
                break;                                                                      \
            default: {                                                                      \
                const uint8x8_t f0_x = vdup_n_u8((uint8_t)(8 - xoffset));                   \
                const uint8x8_t f1_x = vdup_n_u8((uint8_t)xoffset);                         \
                SPV_RUN_H(W, h, SPV_HGEN_TILE);                                             \
                break;                                                                      \
            }                                                                               \
            }                                                                               \
        } else {                                                                            \
            const uint8x8_t f0_x = vdup_n_u8((uint8_t)(8 - xoffset));                       \
            const uint8x8_t f1_x = vdup_n_u8((uint8_t)xoffset);                             \
            const uint8x8_t f0_y = vdup_n_u8((uint8_t)(8 - yoffset));                       \
            const uint8x8_t f1_y = vdup_n_u8((uint8_t)yoffset);                             \
            switch (SPV_KEY(xoffset, yoffset)) {                                            \
            case SPV_KEY(0, 2):                                                             \
                SPV_RUN_SW(W, h, spv_h_0, spv_v_2);                                         \
                break;                                                                      \
            case SPV_KEY(0, 4):                                                             \
                SPV_RUN_SW(W, h, spv_h_0, spv_v_4);                                         \
                break;                                                                      \
            case SPV_KEY(0, 6):                                                             \
                SPV_RUN_SW(W, h, spv_h_0, spv_v_6);                                         \
                break;                                                                      \
            case SPV_KEY(0, 1):                                                             \
            case SPV_KEY(0, 3):                                                             \
            case SPV_KEY(0, 5):                                                             \
            case SPV_KEY(0, 7):                                                             \
                SPV_RUN_SW(W, h, spv_h_0, SPV_VGEN_TILE);                                   \
                break;                                                                      \
            case SPV_KEY(2, 2):                                                             \
                SPV_RUN_SW(W, h, spv_h_2, spv_v_2);                                         \
                break;                                                                      \
            case SPV_KEY(2, 4):                                                             \
                SPV_RUN_SW(W, h, spv_h_2, spv_v_4);                                         \
                break;                                                                      \
            case SPV_KEY(2, 6):                                                             \
                SPV_RUN_SW(W, h, spv_h_2, spv_v_6);                                         \
                break;                                                                      \
            case SPV_KEY(2, 1):                                                             \
            case SPV_KEY(2, 3):                                                             \
            case SPV_KEY(2, 5):                                                             \
            case SPV_KEY(2, 7):                                                             \
                SPV_RUN_SW(W, h, spv_h_2, SPV_VGEN_TILE);                                   \
                break;                                                                      \
            case SPV_KEY(4, 2):                                                             \
                SPV_RUN_SW(W, h, spv_h_4, spv_v_2);                                         \
                break;                                                                      \
            case SPV_KEY(4, 4):                                                             \
                SPV_RUN_SW(W, h, spv_h_4, spv_v_4);                                         \
                break;                                                                      \
            case SPV_KEY(4, 6):                                                             \
                SPV_RUN_SW(W, h, spv_h_4, spv_v_6);                                         \
                break;                                                                      \
            case SPV_KEY(4, 1):                                                             \
            case SPV_KEY(4, 3):                                                             \
            case SPV_KEY(4, 5):                                                             \
            case SPV_KEY(4, 7):                                                             \
                SPV_RUN_SW(W, h, spv_h_4, SPV_VGEN_TILE);                                   \
                break;                                                                      \
            case SPV_KEY(6, 2):                                                             \
                SPV_RUN_SW(W, h, spv_h_6, spv_v_2);                                         \
                break;                                                                      \
            case SPV_KEY(6, 4):                                                             \
                SPV_RUN_SW(W, h, spv_h_6, spv_v_4);                                         \
                break;                                                                      \
            case SPV_KEY(6, 6):                                                             \
                SPV_RUN_SW(W, h, spv_h_6, spv_v_6);                                         \
                break;                                                                      \
            case SPV_KEY(6, 1):                                                             \
            case SPV_KEY(6, 3):                                                             \
            case SPV_KEY(6, 5):                                                             \
            case SPV_KEY(6, 7):                                                             \
                SPV_RUN_SW(W, h, spv_h_6, SPV_VGEN_TILE);                                   \
                break;                                                                      \
            case SPV_KEY(1, 2):                                                             \
            case SPV_KEY(3, 2):                                                             \
            case SPV_KEY(5, 2):                                                             \
            case SPV_KEY(7, 2):                                                             \
                SPV_RUN_SW(W, h, SPV_HGEN_TILE, spv_v_2);                                   \
                break;                                                                      \
            case SPV_KEY(1, 4):                                                             \
            case SPV_KEY(3, 4):                                                             \
            case SPV_KEY(5, 4):                                                             \
            case SPV_KEY(7, 4):                                                             \
                SPV_RUN_SW(W, h, SPV_HGEN_TILE, spv_v_4);                                   \
                break;                                                                      \
            case SPV_KEY(1, 6):                                                             \
            case SPV_KEY(3, 6):                                                             \
            case SPV_KEY(5, 6):                                                             \
            case SPV_KEY(7, 6):                                                             \
                SPV_RUN_SW(W, h, SPV_HGEN_TILE, spv_v_6);                                   \
                break;                                                                      \
            default:                                                                        \
                SPV_RUN_SW(W, h, SPV_HGEN_TILE, SPV_VGEN_TILE);                             \
                break;                                                                      \
            }                                                                               \
        }                                                                                   \
                                                                                            \
        for (int i = 1; i < SPV_NACCUM(W); ++i) {                                           \
            sse_u32_lo[0]  = vaddq_u32(sse_u32_lo[0], sse_u32_lo[i]);                       \
            sse_u32_hi[0]  = vaddq_u32(sse_u32_hi[0], sse_u32_hi[i]);                       \
            sum_src_u16[0] = vaddq_u16(sum_src_u16[0], sum_src_u16[i]);                     \
            sum_ref_u16[0] = vaddq_u16(sum_ref_u16[0], sum_ref_u16[i]);                     \
            sum_src_u32[0] = vaddq_u32(sum_src_u32[0], sum_src_u32[i]);                     \
            sum_ref_u32[0] = vaddq_u32(sum_ref_u32[0], sum_ref_u32[i]);                     \
        }                                                                                   \
        sum_src_u32[0] = vpadalq_u16(sum_src_u32[0], sum_src_u16[0]);                       \
        sum_ref_u32[0] = vpadalq_u16(sum_ref_u32[0], sum_ref_u16[0]);                       \
        sum            = (int)vaddvq_u32(sum_src_u32[0]) - (int)vaddvq_u32(sum_ref_u32[0]); \
        *sse           = vaddvq_u32(vaddq_u32(sse_u32_lo[0], sse_u32_hi[0]));               \
        return *sse - (uint32_t)(((int64_t)sum * sum) >> svt_ctz((W) * h));                 \
    }

FUSED_SUBPEL_VARIANCE_WX_NEON(16)
FUSED_SUBPEL_VARIANCE_WX_NEON(32)
FUSED_SUBPEL_VARIANCE_WX_NEON(64)
FUSED_SUBPEL_VARIANCE_WX_NEON(128)

#define FUSED_SUBPEL_VARIANCE_WXH_NEON(W, H)                                                                 \
    unsigned int svt_aom_sub_pixel_variance##W##x##H##_neon(const uint8_t* src,                              \
                                                            int            src_stride,                       \
                                                            int            xoffset,                          \
                                                            int            yoffset,                          \
                                                            const uint8_t* ref,                              \
                                                            int            ref_stride,                       \
                                                            unsigned int*  sse) {                             \
        return sub_pixel_variance_w##W##_neon(src, src_stride, xoffset, yoffset, ref, ref_stride, (H), sse); \
    }

FUSED_SUBPEL_VARIANCE_WXH_NEON(16, 4)
FUSED_SUBPEL_VARIANCE_WXH_NEON(16, 8)
FUSED_SUBPEL_VARIANCE_WXH_NEON(16, 16)
FUSED_SUBPEL_VARIANCE_WXH_NEON(16, 32)
FUSED_SUBPEL_VARIANCE_WXH_NEON(16, 64)
FUSED_SUBPEL_VARIANCE_WXH_NEON(32, 8)
FUSED_SUBPEL_VARIANCE_WXH_NEON(32, 16)
FUSED_SUBPEL_VARIANCE_WXH_NEON(32, 32)
FUSED_SUBPEL_VARIANCE_WXH_NEON(32, 64)
FUSED_SUBPEL_VARIANCE_WXH_NEON(64, 16)
FUSED_SUBPEL_VARIANCE_WXH_NEON(64, 32)
FUSED_SUBPEL_VARIANCE_WXH_NEON(64, 64)
FUSED_SUBPEL_VARIANCE_WXH_NEON(64, 128)
FUSED_SUBPEL_VARIANCE_WXH_NEON(128, 64)
FUSED_SUBPEL_VARIANCE_WXH_NEON(128, 128)

#undef FUSED_SUBPEL_VARIANCE_WXH_NEON
#undef FUSED_SUBPEL_VARIANCE_WX_NEON
#undef SPV_VGEN_TILE
#undef SPV_HGEN_TILE
#undef SPV_RUN_SW
#undef SPV_RUN_H
#undef SPV_MAYBE_ABSORB
#undef SPV_ACCUM_TILE
#undef SPV_ABSORB_INTERVAL
#undef SPV_NACCUM
#undef SPV_TILES
#undef SPV_KEY

unsigned int svt_aom_mse16x16_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride) {
    uint32x4_t sse_u32[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};

    int i = 16;
    do {
        uint8x16_t s0 = vld1q_u8(src);
        uint8x16_t r0 = vld1q_u8(ref);

        int16x8_t diff0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s0), vget_low_u8(r0)));
        int16x8_t diff1 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s0), vget_high_u8(r0)));

        uint16x8_t sq0 = vreinterpretq_u16_s16(vmulq_s16(diff0, diff0));
        uint16x8_t sq1 = vreinterpretq_u16_s16(vmulq_s16(diff1, diff1));
        sse_u32[0]     = vpadalq_u16(sse_u32[0], sq0);
        sse_u32[1]     = vpadalq_u16(sse_u32[1], sq1);

        src += src_stride;
        ref += ref_stride;
    } while (--i != 0);

    return vaddvq_u32(vaddq_u32(sse_u32[0], sse_u32[1]));
}
