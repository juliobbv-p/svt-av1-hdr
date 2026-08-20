/*
 * Copyright(c) 2019 Netflix, Inc.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

/******************************************************************************
 * @file intrapred_edge_filter_test.cc
 *
 * @brief Unit test for chroma from luma prediction:
 * - svt_cfl_predict_hbd_avx2
 * - svt_cfl_predict_lbd_avx2
 * - svt_cfl_luma_subsampling_420_lbd_avx2
 * - svt_cfl_luma_subsampling_420_hbd_avx2
 *
 * @author Cidana-Wenyao
 *
 ******************************************************************************/

#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include "aom_dsp_rtcd.h"
#include "common_utils.h"
#include "random.h"
#include "util.h"
namespace {
using svt_av1_test_tool::SVTRandom;

using CFL_PRED_HBD = void (*)(const int16_t *pred_buf_q3, uint16_t *pred,
                              int32_t pred_stride, uint16_t *dst,
                              int32_t dst_stride, int32_t alpha_q3,
                              int32_t bit_depth, int32_t width, int32_t height);
using CFL_PRED_LBD = void (*)(const int16_t *pred_buf_q3, uint8_t *pred,
                              int32_t pred_stride, uint8_t *dst,
                              int32_t dst_stride, int32_t alpha_q3,
                              int32_t bit_depth, int32_t width, int32_t height);
/**
 * @brief Unit test for chroma from luma prediction:
 * - svt_cfl_predict_hbd_avx2
 * - svt_cfl_predict_lbd_avx2
 *
 * Test strategy:
 * Verify this assembly code by comparing with reference c implementation.
 * Feed the same data and check test output and reference output.
 * Define a templete class to handle the common process, and
 * declare sub class to handle different bitdepth and function types.
 *
 * Expect result:
 * Output from assemble functions should be the same with output from c.
 *
 * Test coverage:
 * Test cases:
 * pred buffer and dst buffer: Fill with random values
 * TxSize: all the TxSize.
 * alpha_q3: [-16, 16]
 * BitDepth: 8bit and 10bit
 */
template <typename Sample, typename FuncType, int bd, FuncType ref_func>
class CflPredTest : public ::testing::TestWithParam<FuncType> {
  public:
    void RunAllTest() {
        // for pred_buf, after sampling and subtracted from average
        SVTRandom pred_rnd{bd + 3 + 1, true};
        SVTRandom dst_rnd{8, false};
        constexpr int c_stride = CFL_BUF_LINE;
        for (int tx = TX_4X4; tx < TX_SIZES_ALL; ++tx) {
            const int c_w = tx_size_wide[tx];
            const int c_h = tx_size_high[tx];
            if (c_w > 32 || c_h > 32) {
                continue;
            }
            pred_buf_q3.fill(0);
            dst_buf_ref_data_.fill(0);
            dst_buf_tst_data_.fill(0);

            for (int alpha_q3 = -16; alpha_q3 <= 16; ++alpha_q3) {
                // prepare data
                // Implicit assumption: The dst_buf is supposed to be populated
                // by dc prediction before cfl prediction.
                const Sample rnd_dc = dst_rnd.random();
                for (int y = 0; y < c_h; ++y) {
                    for (int x = 0; x < c_w; ++x) {
                        pred_buf_q3[y * c_stride + x] =
                            (Sample)pred_rnd.random();
                        dst_buf_ref_[y * c_stride + x] =
                            dst_buf_tst_[y * c_stride + x] = rnd_dc;
                    }
                }

                ref_func(pred_buf_q3.data(),
                         dst_buf_ref_,
                         CFL_BUF_LINE,
                         dst_buf_ref_,
                         CFL_BUF_LINE,
                         alpha_q3,
                         bd,
                         c_w,
                         c_h);
                tst_func_(pred_buf_q3.data(),
                          dst_buf_tst_,
                          c_stride,
                          dst_buf_tst_,
                          c_stride,
                          alpha_q3,
                          bd,
                          c_w,
                          c_h);

                for (int y = 0; y < c_h; ++y) {
                    for (int x = 0; x < c_w; ++x) {
                        ASSERT_EQ(dst_buf_ref_[y * c_stride + x],
                                  dst_buf_tst_[y * c_stride + x])
                            << "tx_size: " << tx << " alpha_q3 " << alpha_q3
                            << " expect " << dst_buf_ref_[y * c_stride + x]
                            << " got " << dst_buf_tst_[y * c_stride + x]
                            << " at [ " << x << " x " << y << " ]";
                    }
                }
            }
        }
    }

    static constexpr int alignment = 32;
    std::array<int16_t, CFL_BUF_SQUARE> pred_buf_q3{};
    std::array<Sample, CFL_BUF_SQUARE + alignment - 1> dst_buf_ref_data_{};
    std::array<Sample, CFL_BUF_SQUARE + alignment - 1> dst_buf_tst_data_{};
    Sample *const dst_buf_ref_{reinterpret_cast<Sample *>(
        (reinterpret_cast<intptr_t>(dst_buf_ref_data_.data()) + alignment - 1) &
        ~(alignment - 1))};
    Sample *const dst_buf_tst_{reinterpret_cast<Sample *>(
        (reinterpret_cast<intptr_t>(dst_buf_tst_data_.data()) + alignment - 1) &
        ~(alignment - 1))};
    const FuncType tst_func_{this->GetParam()};
};

using LbdCflPredTest =
    CflPredTest<uint8_t, CFL_PRED_LBD, 8, svt_cfl_predict_lbd_c>;
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LbdCflPredTest);

TEST_P(LbdCflPredTest, MatchTest) {
    RunAllTest();
}
#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(AVX2, LbdCflPredTest,
                         ::testing::Values(svt_cfl_predict_lbd_avx2));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(NEON, LbdCflPredTest,
                         ::testing::Values(svt_aom_cfl_predict_lbd_neon));
#endif  // ARCH_AARCH64

using HbdCflPredTest =
    CflPredTest<uint16_t, CFL_PRED_HBD, 10, svt_cfl_predict_hbd_c>;

TEST_P(HbdCflPredTest, MatchTest) {
    RunAllTest();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(AVX2, HbdCflPredTest,
                         ::testing::Values(svt_cfl_predict_hbd_avx2));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(NEON, HbdCflPredTest,
                         ::testing::Values(svt_cfl_predict_hbd_neon));
#endif  // ARCH_AARCH64

using AomUpsampledPredFunc = decltype(&svt_aom_upsampled_pred_c);

using AomUpsampledPredParam =
    ::testing::tuple<BlockSize, AomUpsampledPredFunc, int, int, int, uint64_t>;

class AomUpsampledPredTest
    : public ::testing::TestWithParam<AomUpsampledPredParam> {
  public:
    void run_test() {
        const int block_size = TEST_GET_PARAM(0);
        const AomUpsampledPredFunc test_impl = TEST_GET_PARAM(1);
        const int subpel_search = TEST_GET_PARAM(2);
        const int subpel_x_q3 = TEST_GET_PARAM(3);
        const int subpel_y_q3 = TEST_GET_PARAM(4);
        const int width = block_size_wide[block_size];
        const int height = block_size_high[block_size];
        alignas(16) std::array<uint8_t, MAX_SB_SQUARE * 2> ref_;
        alignas(16) std::array<uint8_t, MAX_SB_SQUARE * 2> comp_pred_ref_;
        alignas(16) std::array<uint8_t, MAX_SB_SQUARE * 2> comp_pred_tst_;

        comp_pred_ref_.fill(1);
        comp_pred_tst_.fill(1);

        // Function svt_aom_upsampled_pred call inside function pointer
        // which have to be set properly by
        // svt_aom_setup_common_rtcd_internal(), we want to test intrinsic
        // version of it, so feature flag is necessary
        svt_aom_setup_common_rtcd_internal(TEST_GET_PARAM(5));

        constexpr int run_times = 100;
        for (int i = 0; i < run_times; ++i) {
            ref_.fill(1);
            std::generate_n(ref_.begin(), width * height + 3 * width, [&]() {
                return rnd_.random();
            });

            svt_aom_upsampled_pred_c(NULL,
                                     NULL,
                                     0,
                                     0,
                                     comp_pred_ref_.data(),
                                     width,
                                     height,
                                     subpel_x_q3,
                                     subpel_y_q3,
                                     ref_.data() + 3 * width,
                                     width,
                                     subpel_search);
            test_impl(NULL,
                      NULL,
                      0,
                      0,
                      comp_pred_tst_.data(),
                      width,
                      height,
                      subpel_x_q3,
                      subpel_y_q3,
                      ref_.data() + 3 * width,
                      width,
                      subpel_search);

            ASSERT_EQ(comp_pred_ref_, comp_pred_tst_);
        }
    }

  private:
    SVTRandom rnd_{0, 255};
};

TEST_P(AomUpsampledPredTest, MatchTest) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE2, AomUpsampledPredTest,
    ::testing::Combine(
        ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
        ::testing::Values(svt_aom_upsampled_pred_sse2),
        ::testing::Values((int)USE_2_TAPS, (int)USE_4_TAPS, (int)USE_8_TAPS),
        ::testing::Values(0, 1, 2), ::testing::Values(0, 1, 2),
        ::testing::Values(EB_CPU_FLAGS_SSSE3, EB_CPU_FLAGS_AVX2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, AomUpsampledPredTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_upsampled_pred_neon),
                       ::testing::Values((int)USE_2_TAPS, (int)USE_4_TAPS,
                                         (int)USE_8_TAPS),
                       ::testing::Values(0, 1, 2), ::testing::Values(0, 1, 2),
                       ::testing::Values(EB_CPU_FLAGS_NEON)));

#if HAVE_NEON_DOTPROD
INSTANTIATE_TEST_SUITE_P(
    NEON_DOTPROD, AomUpsampledPredTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_upsampled_pred_neon),
                       ::testing::Values((int)USE_2_TAPS, (int)USE_4_TAPS,
                                         (int)USE_8_TAPS),
                       ::testing::Values(0, 1, 2), ::testing::Values(0, 1, 2),
                       ::testing::Values(EB_CPU_FLAGS_NEON_DOTPROD)));
#endif  // HAVE_NEON_DOTPROD

#if HAVE_NEON_I8MM
INSTANTIATE_TEST_SUITE_P(
    NEON_I8MM, AomUpsampledPredTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_upsampled_pred_neon),
                       ::testing::Values((int)USE_2_TAPS, (int)USE_4_TAPS,
                                         (int)USE_8_TAPS),
                       ::testing::Values(0, 1, 2), ::testing::Values(0, 1, 2),
                       ::testing::Values(EB_CPU_FLAGS_NEON_I8MM)));
#endif  // HAVE_NEON_I8MM
#endif  // ARCH_AARCH64

using CflLumaSubsamplingLbdFunc = decltype(&svt_cfl_luma_subsampling_420_lbd_c);

using CflLumaSubsamplingLbdParam =
    ::testing::tuple<BlockSize, CflLumaSubsamplingLbdFunc>;

class CflLumaSubsamplingLbdTest
    : public ::testing::TestWithParam<CflLumaSubsamplingLbdParam> {
  public:
    void run_test() {
        const int block_size = TEST_GET_PARAM(0);
        const CflLumaSubsamplingLbdFunc test_impl = TEST_GET_PARAM(1);
        const int width = block_size_wide[block_size];
        const int height = block_size_high[block_size];
        // CFL prediction only operates on blocks where
        // max(width, height) <= 32.
        if (width > 32 || height > 32)
            return;
        alignas(16) std::array<uint8_t, MAX_SB_SQUARE> input;
        alignas(16) std::array<int16_t, MAX_SB_SQUARE> output_q3_ref_;
        alignas(16) std::array<int16_t, MAX_SB_SQUARE> output_q3_tst_;

        output_q3_ref_.fill(1);
        output_q3_tst_.fill(1);

        const int run_times = 100;
        for (int i = 0; i < run_times; ++i) {
            std::generate(
                input.begin(), input.end(), [&]() { return rnd_.random(); });

            svt_cfl_luma_subsampling_420_lbd_c(
                input.data(), width, output_q3_ref_.data(), width, height);

            test_impl(
                input.data(), width, output_q3_tst_.data(), width, height);

            ASSERT_EQ(output_q3_ref_, output_q3_tst_);
        }
    }

  private:
    SVTRandom rnd_{0, 255};
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(CflLumaSubsamplingLbdTest);

TEST_P(CflLumaSubsamplingLbdTest, MatchTest) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, CflLumaSubsamplingLbdTest,
    ::testing::Combine(
        ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
        ::testing::Values(svt_cfl_luma_subsampling_420_lbd_avx2)));
#endif

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, CflLumaSubsamplingLbdTest,
    ::testing::Combine(
        ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
        ::testing::Values(svt_cfl_luma_subsampling_420_lbd_neon)));
#endif

using CflLumaSubsamplingHbdFunc = decltype(&svt_cfl_luma_subsampling_420_hbd_c);
using CflLumaSubsamplingHbdParam =
    ::testing::tuple<BlockSize, CflLumaSubsamplingHbdFunc>;

class CflLumaSubsamplingHbdTest
    : public ::testing::TestWithParam<CflLumaSubsamplingHbdParam> {
  public:
    void run_test() {
        const int block_size = TEST_GET_PARAM(0);
        const CflLumaSubsamplingHbdFunc test_impl = TEST_GET_PARAM(1);
        const int width = block_size_wide[block_size];
        const int height = block_size_high[block_size];
        // CFL prediction only operates on blocks where
        // max(width, height) <= 32.
        if (width > 32 || height > 32)
            return;
        alignas(16) std::array<uint16_t, MAX_SB_SQUARE> input;
        alignas(16) std::array<int16_t, MAX_SB_SQUARE> output_q3_ref_;
        alignas(16) std::array<int16_t, MAX_SB_SQUARE> output_q3_tst_;

        output_q3_ref_.fill(1);
        output_q3_tst_.fill(1);

        const int run_times = 100;
        for (int i = 0; i < run_times; ++i) {
            std::generate(
                input.begin(), input.end(), [&]() { return rnd_.random(); });

            svt_cfl_luma_subsampling_420_hbd_c(
                input.data(), width, output_q3_ref_.data(), width, height);

            test_impl(
                input.data(), width, output_q3_tst_.data(), width, height);

            ASSERT_EQ(output_q3_ref_, output_q3_tst_);
        }
    }

  private:
    SVTRandom rnd_{0, 1023};
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(CflLumaSubsamplingHbdTest);

TEST_P(CflLumaSubsamplingHbdTest, MatchTest) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, CflLumaSubsamplingHbdTest,
    ::testing::Combine(
        ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
        ::testing::Values(svt_cfl_luma_subsampling_420_hbd_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, CflLumaSubsamplingHbdTest,
    ::testing::Combine(
        ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
        ::testing::Values(svt_cfl_luma_subsampling_420_hbd_neon)));
#endif  // ARCH_AARCH64

}  // namespace
