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

#include <stdint.h>

#include <arm_neon.h>

#include "aom_dsp_rtcd.h"
#include "mem_neon.h"

// svt_aom_sad<W>x<H>x4d() computes the SAD of one source block against four
// arbitrary reference blocks (these references are not adjacent, unlike the
// internal sadWxHx4d helpers used by svt_sad_loop_kernel). Each source row is
// loaded once and differenced against all four references, so the source load
// is reused 4x -- a saving a per-reference wrapper around the single-ref SAD
// kernel cannot capture. SAD is an exact integer sum, so the result is
// bit-identical to svt_aom_sad<W>x<H>x4d_c.
static inline void sadwxhx4d_neon(const uint8_t* src, int src_stride, const uint8_t* const ref_array[], int ref_stride,
                                  int w, int h, uint32_t* sad_array) {
    const uint8_t* r0 = ref_array[0];
    const uint8_t* r1 = ref_array[1];
    const uint8_t* r2 = ref_array[2];
    const uint8_t* r3 = ref_array[3];
    if (w >= 16) {
        // Accumulate abs-diff straight into u16 lanes with vpadalq_u8 (one reduction
        // op/chunk, matching the house single-ref idiom), and fold the u16 partials
        // into u32 only as often as the dynamic range demands. Each chunk adds <= 510
        // to a u16 lane (pairwise sum of two |s-r| <= 255), so a lane is safe for up
        // to 128 chunks; with w/16 chunks per row that is 2048/w rows between folds.
        // Only 64x{64,128} and 128x{64,128} ever fold mid-block; the other w>=16
        // sizes accumulate the whole block in u16 and fold once at the end.
        const int  fold_rows = 2048 / w;
        uint32x4_t a0 = vdupq_n_u32(0), a1 = a0, a2 = a0, a3 = a0;
        uint16x8_t b0 = vdupq_n_u16(0), b1 = b0, b2 = b0, b3 = b0;
        int        since = 0;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; x += 16) {
                uint8x16_t s = vld1q_u8(src + x);
                b0           = vpadalq_u8(b0, vabdq_u8(s, vld1q_u8(r0 + x)));
                b1           = vpadalq_u8(b1, vabdq_u8(s, vld1q_u8(r1 + x)));
                b2           = vpadalq_u8(b2, vabdq_u8(s, vld1q_u8(r2 + x)));
                b3           = vpadalq_u8(b3, vabdq_u8(s, vld1q_u8(r3 + x)));
            }
            src += src_stride;
            r0 += ref_stride;
            r1 += ref_stride;
            r2 += ref_stride;
            r3 += ref_stride;
            if (++since == fold_rows) {
                a0 = vpadalq_u16(a0, b0);
                a1 = vpadalq_u16(a1, b1);
                a2 = vpadalq_u16(a2, b2);
                a3 = vpadalq_u16(a3, b3);
                b0 = b1 = b2 = b3 = vdupq_n_u16(0);
                since             = 0;
            }
        }
        sad_array[0] = vaddvq_u32(vpadalq_u16(a0, b0));
        sad_array[1] = vaddvq_u32(vpadalq_u16(a1, b1));
        sad_array[2] = vaddvq_u32(vpadalq_u16(a2, b2));
        sad_array[3] = vaddvq_u32(vpadalq_u16(a3, b3));
    } else {
        // w == 4 or w == 8: the tallest such blocks are 4x16 and 8x32, so the
        // per-lane sum (max 255*32 = 8160) stays well inside 16 bits.
        uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
        for (int y = 0; y < h; ++y) {
            uint8x8_t s = (w == 8) ? vld1_u8(src) : load_u8_4x1(src);
            a0          = vaddw_u8(a0, vabd_u8(s, (w == 8) ? vld1_u8(r0) : load_u8_4x1(r0)));
            a1          = vaddw_u8(a1, vabd_u8(s, (w == 8) ? vld1_u8(r1) : load_u8_4x1(r1)));
            a2          = vaddw_u8(a2, vabd_u8(s, (w == 8) ? vld1_u8(r2) : load_u8_4x1(r2)));
            a3          = vaddw_u8(a3, vabd_u8(s, (w == 8) ? vld1_u8(r3) : load_u8_4x1(r3)));
            src += src_stride;
            r0 += ref_stride;
            r1 += ref_stride;
            r2 += ref_stride;
            r3 += ref_stride;
        }
        sad_array[0] = vaddlvq_u16(a0);
        sad_array[1] = vaddlvq_u16(a1);
        sad_array[2] = vaddlvq_u16(a2);
        sad_array[3] = vaddlvq_u16(a3);
    }
}

// One out-of-line core per width. Keeping w a compile-time constant lets the
// w>=16 branch and the inner column loop fold away (the width specialization
// that earns its keep), while the height stays a runtime argument so the 22
// public sizes share six bodies instead of emitting a fully unrolled body each.
// noinline pins the fold: without it each core would be re-inlined -- and
// re-specialized on h -- into its wrappers, undoing the saving. The per-call
// cost is a single tail-branch, which microbenchmarks at parity with the fully
// specialized form.
#define SAD_WXHX4D_NEON(w)                                                                     \
    static void __attribute__((noinline)) sad##w##xhx4d_neon(const uint8_t*       src,         \
                                                             int                  src_stride,  \
                                                             const uint8_t* const ref_array[], \
                                                             int                  ref_stride,  \
                                                             int                  h,           \
                                                             uint32_t*            sad_array) {            \
        sadwxhx4d_neon(src, src_stride, ref_array, ref_stride, (w), h, sad_array);             \
    }

SAD_WXHX4D_NEON(4)
SAD_WXHX4D_NEON(8)
SAD_WXHX4D_NEON(16)
SAD_WXHX4D_NEON(32)
SAD_WXHX4D_NEON(64)
SAD_WXHX4D_NEON(128)

#define SAD_NXMX4D_NEON(w, h)                                                                                        \
    void svt_aom_sad##w##x##h##x4d_neon(                                                                             \
        const uint8_t* src, int src_stride, const uint8_t* const ref_array[], int ref_stride, uint32_t* sad_array) { \
        sad##w##xhx4d_neon(src, src_stride, ref_array, ref_stride, (h), sad_array);                                  \
    }

SAD_NXMX4D_NEON(128, 128)
SAD_NXMX4D_NEON(128, 64)
SAD_NXMX4D_NEON(64, 128)
SAD_NXMX4D_NEON(64, 64)
SAD_NXMX4D_NEON(64, 32)
SAD_NXMX4D_NEON(64, 16)
SAD_NXMX4D_NEON(32, 64)
SAD_NXMX4D_NEON(32, 32)
SAD_NXMX4D_NEON(32, 16)
SAD_NXMX4D_NEON(32, 8)
SAD_NXMX4D_NEON(16, 64)
SAD_NXMX4D_NEON(16, 32)
SAD_NXMX4D_NEON(16, 16)
SAD_NXMX4D_NEON(16, 8)
SAD_NXMX4D_NEON(16, 4)
SAD_NXMX4D_NEON(8, 32)
SAD_NXMX4D_NEON(8, 16)
SAD_NXMX4D_NEON(8, 8)
SAD_NXMX4D_NEON(8, 4)
SAD_NXMX4D_NEON(4, 16)
SAD_NXMX4D_NEON(4, 8)
SAD_NXMX4D_NEON(4, 4)

// svt_aom_sad<W>x<H>() computes the SAD of one source block against one reference
// block. Wraps the per-width NEON SAD helper; bit-identical to svt_aom_sad<W>x<H>_c.
#define SAD_NXM_NEON(w, h)                                                                                         \
    uint32_t svt_aom_sad##w##x##h##_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride) { \
        return svt_nxm_sad_kernel_helper_neon(                                                                     \
            src, (uint32_t)src_stride, ref, (uint32_t)ref_stride, (uint32_t)(h), (uint32_t)(w));                   \
    }

SAD_NXM_NEON(128, 128)
SAD_NXM_NEON(128, 64)
SAD_NXM_NEON(64, 128)
SAD_NXM_NEON(64, 64)
SAD_NXM_NEON(64, 32)
SAD_NXM_NEON(64, 16)
SAD_NXM_NEON(32, 64)
SAD_NXM_NEON(32, 32)
SAD_NXM_NEON(32, 16)
SAD_NXM_NEON(32, 8)
SAD_NXM_NEON(16, 64)
SAD_NXM_NEON(16, 32)
SAD_NXM_NEON(16, 16)
SAD_NXM_NEON(16, 8)
SAD_NXM_NEON(16, 4)
SAD_NXM_NEON(8, 32)
SAD_NXM_NEON(8, 16)
SAD_NXM_NEON(8, 8)
SAD_NXM_NEON(8, 4)
SAD_NXM_NEON(4, 16)
SAD_NXM_NEON(4, 8)
SAD_NXM_NEON(4, 4)
