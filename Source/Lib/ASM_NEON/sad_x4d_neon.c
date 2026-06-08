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

#include "aom_dsp_rtcd.h"

// svt_aom_sad<W>x<H>x4d() computes the SAD of one source block against four
// arbitrary reference blocks. The four references are not adjacent (unlike the
// internal sadWxHx4d helpers used by svt_sad_loop_kernel), so each is evaluated
// with the per-width NEON SAD helper. SAD is an exact integer sum, so the result
// is bit-identical to svt_aom_sad<W>x<H>x4d_c.
#define SAD_NXMX4D_NEON(w, h)                                                                                        \
    void svt_aom_sad##w##x##h##x4d_neon(                                                                             \
        const uint8_t* src, int src_stride, const uint8_t* const ref_array[], int ref_stride, uint32_t* sad_array) { \
        for (int i = 0; i < 4; ++i) {                                                                                \
            sad_array[i] = svt_nxm_sad_kernel_helper_neon(                                                           \
                src, (uint32_t)src_stride, ref_array[i], (uint32_t)ref_stride, (uint32_t)(h), (uint32_t)(w));        \
        }                                                                                                            \
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
