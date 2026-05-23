/*
 * Copyright (c) 2026, Meta Platforms, Inc. and affiliates.
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
 * SubpelVariancePerfTest.cc
 *
 * Per-(function, xoffset, yoffset) microbenchmark for
 *   - svt_aom_sub_pixel_varianceWxH_neon
 *   - svt_aom_sub_pixel_varianceWxH_neon_dotprod
 *
 * Reports a single average nanoseconds-per-call value per cell.
 ******************************************************************************/

#include <stdint.h>
#include <stdio.h>

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "gtest/gtest.h"
#include "utility.h"

#if ARCH_AARCH64

namespace {

using SubpixVarFn = unsigned int (*)(const uint8_t *src, int src_stride,
                                     int xoffset, int yoffset,
                                     const uint8_t *ref, int ref_stride,
                                     unsigned int *sse);

// Number of warm-up runs per cell before timed measurement begins.
constexpr int kWarmupRuns = 3;

// Reproducible seed for src/ref buffers.
constexpr uint64_t kSeed = 0xC0FFEE12345ULL;

struct OffsetPair {
    int xoff;
    int yoff;
};
constexpr std::array<OffsetPair, 15> kOffsets = {{
    // hpel
    {4, 0},
    {0, 4},
    {4, 4},
    // qpel single-axis cheap 1D bilinear
    {2, 0},
    {6, 0},
    {0, 2},
    {0, 6},
    // hpel/qpel 2-pass
    {4, 2},
    {4, 6},
    {2, 4},
    {6, 4},
    // qpel/qpel 2-pass
    {2, 2},
    {6, 2},
    {2, 6},
    {6, 6},
}};

struct KernelEntry {
    const char *name;
    SubpixVarFn fn;
    int w;
    int h;
};

// Volatile sink to prevent dead-code elimination of the kernel calls.
static volatile uint64_t g_sink = 0;

static double run_cell(const KernelEntry &k, int xoff, int yoff,
                       const uint8_t *src, int src_stride, const uint8_t *ref,
                       int ref_stride) {
    using clock = std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::nanoseconds;

    const int iters = 100000000 / (k.w * k.h);

    uint64_t local_sink = 0;

    // Warm-up: kWarmupRuns untimed batches.
    for (int b = 0; b < kWarmupRuns; ++b) {
        uint64_t acc = 0;
        for (int i = 0; i < iters; ++i) {
            unsigned int sse = 0;
            unsigned int v =
                k.fn(src, src_stride, xoff, yoff, ref, ref_stride, &sse);
            acc += sse + v;
        }
        local_sink += acc;
    }

    // Single timed batch.
    uint64_t acc = 0;
    auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        unsigned int sse = 0;
        unsigned int v =
            k.fn(src, src_stride, xoff, yoff, ref, ref_stride, &sse);
        acc += sse + v;
    }
    auto t1 = clock::now();
    local_sink += acc;
    g_sink += local_sink;

    const double total_ns =
        static_cast<double>(duration_cast<nanoseconds>(t1 - t0).count());
    const double avg_ns = total_ns / static_cast<double>(iters);

    std::cout << "  " << std::left << std::setw(48) << k.name
              << " W=" << std::right << std::setw(3) << k.w
              << " H=" << std::setw(3) << k.h << "  (xoff=" << std::setw(1)
              << xoff << ", yoff=" << std::setw(1) << yoff << ")"
              << "  iters=" << std::setw(7) << iters << "  avg=" << std::fixed
              << std::setprecision(2) << std::setw(8) << avg_ns << " ns"
              << std::endl;
    return avg_ns;
}

class SubpelVariancePerf : public ::testing::Test {
  protected:
    void SetUp() override {
        constexpr int max_w = 128;
        constexpr int max_h = 128;
        src_stride_ = max_w;
        ref_stride_ = max_w + 1;
        const size_t src_sz = static_cast<size_t>(src_stride_) * max_h;
        const size_t ref_sz = static_cast<size_t>(ref_stride_) * (max_h + 1);
        src_ = static_cast<uint8_t *>(svt_aom_memalign(64, src_sz));
        ref_ = static_cast<uint8_t *>(svt_aom_memalign(64, ref_sz));
        ASSERT_NE(src_, nullptr);
        ASSERT_NE(ref_, nullptr);

        std::mt19937 rng(static_cast<uint32_t>(kSeed));
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        for (size_t i = 0; i < src_sz; ++i) {
            src_[i] = dist(rng);
        }
        for (size_t i = 0; i < ref_sz; ++i) {
            ref_[i] = dist(rng);
        }

        // NEON dotprod kernels call out to variance helpers through the rtcd
        // function-pointer table; ensure it is populated for this thread.
        svt_aom_setup_rtcd_internal(EB_CPU_FLAGS_ALL);
    }

    void TearDown() override {
        if (src_) {
            svt_aom_free(src_);
            src_ = nullptr;
        }
        if (ref_) {
            svt_aom_free(ref_);
            ref_ = nullptr;
        }
    }

    uint8_t *src_ = nullptr;
    uint8_t *ref_ = nullptr;
    int src_stride_ = 0;
    int ref_stride_ = 0;
};

// Token-pasting macros to keep the kernel table dense. PIX_VAR / PIX_VAR_DP
// expand to one KernelEntry row each, deriving the symbol name + function
// pointer from (W, H).
#define PIX_VAR(W, H)                                \
    {"svt_aom_sub_pixel_variance" #W "x" #H "_neon", \
     &svt_aom_sub_pixel_variance##W##x##H##_neon,    \
     W,                                              \
     H}
#define PIX_VAR_DP(W, H)                                     \
    {"svt_aom_sub_pixel_variance" #W "x" #H "_neon_dotprod", \
     &svt_aom_sub_pixel_variance##W##x##H##_neon_dotprod,    \
     W,                                                      \
     H}

TEST_F(SubpelVariancePerf, DISABLED_Offsets) {
    std::vector<KernelEntry> kernels = {
        // NEON path.
        PIX_VAR(4, 4),
        PIX_VAR(4, 8),
        PIX_VAR(4, 16),
        PIX_VAR(8, 4),
        PIX_VAR(8, 8),
        PIX_VAR(8, 16),
        PIX_VAR(8, 32),
        PIX_VAR(16, 4),
        PIX_VAR(16, 8),
        PIX_VAR(16, 16),
        PIX_VAR(16, 32),
        PIX_VAR(16, 64),
        PIX_VAR(32, 8),
        PIX_VAR(32, 16),
        PIX_VAR(32, 32),
        PIX_VAR(32, 64),
        PIX_VAR(64, 16),
        PIX_VAR(64, 32),
        PIX_VAR(64, 64),
        PIX_VAR(64, 128),
        PIX_VAR(128, 64),
        PIX_VAR(128, 128),
        // NEON_DOTPROD path. No 4x4 kernel exists; 4x4 SUBPEL falls back to
        // the plain NEON path.
        PIX_VAR_DP(4, 8),
        PIX_VAR_DP(4, 16),
        PIX_VAR_DP(8, 4),
        PIX_VAR_DP(8, 8),
        PIX_VAR_DP(8, 16),
        PIX_VAR_DP(8, 32),
        PIX_VAR_DP(16, 4),
        PIX_VAR_DP(16, 8),
        PIX_VAR_DP(16, 16),
        PIX_VAR_DP(16, 32),
        PIX_VAR_DP(16, 64),
        PIX_VAR_DP(32, 8),
        PIX_VAR_DP(32, 16),
        PIX_VAR_DP(32, 32),
        PIX_VAR_DP(32, 64),
        PIX_VAR_DP(64, 16),
        PIX_VAR_DP(64, 32),
        PIX_VAR_DP(64, 64),
        PIX_VAR_DP(64, 128),
        PIX_VAR_DP(128, 64),
        PIX_VAR_DP(128, 128),
    };

    // Aggregation accumulators: sum of avg_ns and sample count, split by arch.
    std::map<int, std::pair<double, uint64_t>> by_wh_neon;
    std::map<int, std::pair<double, uint64_t>> by_wh_dp;
    std::map<std::pair<int, int>, std::pair<double, uint64_t>> by_xy_neon;
    std::map<std::pair<int, int>, std::pair<double, uint64_t>> by_xy_dp;

    for (const auto &k : kernels) {
        const bool is_dp =
            std::string(k.name).find("_dotprod") != std::string::npos;
        for (const auto &off : kOffsets) {
            const double avg_ns = run_cell(
                k, off.xoff, off.yoff, src_, src_stride_, ref_, ref_stride_);
            auto &wh = is_dp ? by_wh_dp[k.w] : by_wh_neon[k.w];
            wh.first += avg_ns;
            wh.second += 1;
            auto &xy = is_dp ? by_xy_dp[{off.xoff, off.yoff}]
                             : by_xy_neon[{off.xoff, off.yoff}];
            xy.first += avg_ns;
            xy.second += 1;
        }
    }

    auto print_wh = [](const char *label,
                       const std::map<int, std::pair<double, uint64_t>> &m) {
        for (const auto &kv : m) {
            const int w = kv.first;
            const double sum = kv.second.first;
            const uint64_t n = kv.second.second;
            const double mean = (n > 0) ? (sum / static_cast<double>(n)) : 0.0;
            std::cout << "  " << std::left << std::setw(48) << label
                      << " W=" << std::right << std::setw(3) << w
                      << " H=" << std::setw(3) << '*' << "  (xoff=*, yoff=*)"
                      << "  iters=" << std::setw(7) << n
                      << "  avg=" << std::fixed << std::setprecision(2)
                      << std::setw(8) << mean << " ns" << std::endl;
        }
    };

    auto print_xy = [](const char *label,
                       const std::map<std::pair<int, int>,
                                      std::pair<double, uint64_t>> &m) {
        for (const auto &kv : m) {
            const int xoff = kv.first.first;
            const int yoff = kv.first.second;
            const double sum = kv.second.first;
            const uint64_t n = kv.second.second;
            const double mean = (n > 0) ? (sum / static_cast<double>(n)) : 0.0;
            std::cout << "  " << std::left << std::setw(48) << label
                      << " W=" << std::right << std::setw(3) << '*'
                      << " H=" << std::setw(3) << '*'
                      << "  (xoff=" << std::setw(1) << xoff
                      << ", yoff=" << std::setw(1) << yoff << ")"
                      << "  iters=" << std::setw(7) << n
                      << "  avg=" << std::fixed << std::setprecision(2)
                      << std::setw(8) << mean << " ns" << std::endl;
        }
    };

    // Aggregate table 1: average ns by W, across all (xoff, yoff) and H.
    std::cout << std::endl;
    std::cout << "  --- Average ns by W [across all (xoff, yoff) and H] ---"
              << std::endl;
    std::cout << "  -- NEON --" << std::endl;
    print_wh("(W) aggregate [neon]", by_wh_neon);
    std::cout << "  -- NEON_DOTPROD --" << std::endl;
    print_wh("(W) aggregate [neon_dotprod]", by_wh_dp);

    // Aggregate table 2: average ns by (xoff, yoff), across all shapes.
    std::cout << std::endl;
    std::cout << "  --- Average ns by (xoff, yoff) [across all shapes] ---"
              << std::endl;
    std::cout << "  -- NEON --" << std::endl;
    print_xy("(xoff, yoff) aggregate [neon]", by_xy_neon);
    std::cout << "  -- NEON_DOTPROD --" << std::endl;
    print_xy("(xoff, yoff) aggregate [neon_dotprod]", by_xy_dp);

    // Print sink to ensure no dead-code elimination across the whole run.
    std::cout << "  sink=0x" << std::hex << std::setw(16) << std::setfill('0')
              << static_cast<unsigned long long>(g_sink) << std::dec
              << std::setfill(' ') << std::endl;
}

#undef PIX_VAR
#undef PIX_VAR_DP

}  // namespace

#endif  // ARCH_AARCH64
