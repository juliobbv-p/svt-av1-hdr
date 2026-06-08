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

// Verifies that the NEON single-reference (svt_aom_sad<W>x<H>) and 4-reference
// (svt_aom_sad<W>x<H>x4d) SAD kernels are bit-identical to their C references.

#include <cstdint>

#include "gtest/gtest.h"
#include "aom_dsp_rtcd.h"
#include "random.h"

namespace {
using svt_av1_test_tool::SVTRandom;

typedef uint32_t (*SadFn)(const uint8_t *, int, const uint8_t *, int);
typedef void (*SadX4Fn)(const uint8_t *, int, const uint8_t *const[], int,
                        uint32_t *);

struct SadEntry {
    int w, h;
    SadFn c, neon;
    SadX4Fn c_x4, neon_x4;
};

#define SAD_ENTRY(w, h)            \
    {w,                            \
     h,                            \
     &svt_aom_sad##w##x##h##_c,    \
     &svt_aom_sad##w##x##h##_neon, \
     &svt_aom_sad##w##x##h##x4d_c, \
     &svt_aom_sad##w##x##h##x4d_neon}

const SadEntry kSadSizes[] = {
    SAD_ENTRY(128, 128), SAD_ENTRY(128, 64), SAD_ENTRY(64, 128),
    SAD_ENTRY(64, 64),   SAD_ENTRY(64, 32),  SAD_ENTRY(64, 16),
    SAD_ENTRY(32, 64),   SAD_ENTRY(32, 32),  SAD_ENTRY(32, 16),
    SAD_ENTRY(32, 8),    SAD_ENTRY(16, 64),  SAD_ENTRY(16, 32),
    SAD_ENTRY(16, 16),   SAD_ENTRY(16, 8),   SAD_ENTRY(16, 4),
    SAD_ENTRY(8, 32),    SAD_ENTRY(8, 16),   SAD_ENTRY(8, 8),
    SAD_ENTRY(8, 4),     SAD_ENTRY(4, 16),   SAD_ENTRY(4, 8),
    SAD_ENTRY(4, 4)};

class SadNeonTest : public ::testing::TestWithParam<int> {};

TEST_P(SadNeonTest, MatchesC) {
    const SadEntry &e = kSadSizes[GetParam()];
    const int stride = 128;
    SVTRandom rnd(8, false);

    static uint8_t src[128 * 128];
    static uint8_t ref[(128 + 8) * 128];

    for (int trial = 0; trial < 32; ++trial) {
        for (int i = 0; i < (int)sizeof(src); ++i)
            src[i] = (uint8_t)rnd.random();
        for (int i = 0; i < (int)sizeof(ref); ++i)
            ref[i] = (uint8_t)rnd.random();

        // Single reference.
        const uint32_t sad_c = e.c(src, stride, ref, stride);
        const uint32_t sad_neon = e.neon(src, stride, ref, stride);
        ASSERT_EQ(sad_c, sad_neon)
            << e.w << "x" << e.h << " single, trial " << trial;

        // Four arbitrary references.
        const uint8_t *refs[4] = {
            ref, ref + 7, ref + stride + 3, ref + 2 * stride + 11};
        uint32_t res_c[4], res_neon[4];
        e.c_x4(src, stride, refs, stride, res_c);
        e.neon_x4(src, stride, refs, stride, res_neon);
        for (int i = 0; i < 4; ++i) {
            ASSERT_EQ(res_c[i], res_neon[i])
                << e.w << "x" << e.h << " x4d[" << i << "], trial " << trial;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(NEON, SadNeonTest,
                         ::testing::Range(0, (int)(sizeof(kSadSizes) /
                                                   sizeof(kSadSizes[0]))));
}  // namespace
