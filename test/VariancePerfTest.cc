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
 * VariancePerfTest.cc
 *
 * Per-(function, W, H) microbenchmark for
 *   - svt_aom_varianceWxH_neon
 *   - svt_aom_varianceWxH_neon_dotprod
 *
 * Reports a single average nanoseconds-per-call value per cell.
 ******************************************************************************/

#include <stdint.h>
#include <stdio.h>

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

using VarFn = unsigned int (*)(const uint8_t *src, int src_stride,
                               const uint8_t *ref, int ref_stride,
                               unsigned int *sse);

// Number of warm-up runs per cell before timed measurement begins.
constexpr int kWarmupRuns = 3;

// Reproducible seed for src/ref buffers.
constexpr uint64_t kSeed = 0xC0FFEE12345ULL;

struct KernelEntry {
    const char *name;
    VarFn fn;
    int w;
    int h;
};

// Volatile sink to prevent dead-code elimination of the kernel calls.
static volatile uint64_t g_sink = 0;

static double run_cell(const KernelEntry &k, const uint8_t *src, int src_stride,
                       const uint8_t *ref, int ref_stride) {
    using clock = std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::nanoseconds;

    const int iters = 1000000000 / (k.w * k.h);

    uint64_t local_sink = 0;

    // Warm-up: kWarmupRuns untimed batches.
    for (int b = 0; b < kWarmupRuns; ++b) {
        uint64_t acc = 0;
        for (int i = 0; i < iters; ++i) {
            unsigned int sse = 0;
            unsigned int v = k.fn(src, src_stride, ref, ref_stride, &sse);
            acc += sse + v;
        }
        local_sink += acc;
    }

    // Single timed batch.
    uint64_t acc = 0;
    auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        unsigned int sse = 0;
        unsigned int v = k.fn(src, src_stride, ref, ref_stride, &sse);
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
              << " H=" << std::setw(3) << k.h << "  iters=" << std::setw(8)
              << iters << "  avg=" << std::fixed << std::setprecision(2)
              << std::setw(8) << avg_ns << " ns" << std::endl;
    return avg_ns;
}

class VariancePerf : public ::testing::Test {
  protected:
    void SetUp() override {
        constexpr int max_w = 128;
        constexpr int max_h = 128;
        src_stride_ = max_w;
        ref_stride_ = max_w + 1;
        const size_t src_sz = static_cast<size_t>(src_stride_) * max_h;
        const size_t ref_sz = static_cast<size_t>(ref_stride_) * max_h;
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

#define INT_VAR(W, H)                      \
    {"svt_aom_variance" #W "x" #H "_neon", \
     &svt_aom_variance##W##x##H##_neon,    \
     W,                                    \
     H}
#define INT_VAR_DP(W, H)                           \
    {"svt_aom_variance" #W "x" #H "_neon_dotprod", \
     &svt_aom_variance##W##x##H##_neon_dotprod,    \
     W,                                            \
     H}

TEST_F(VariancePerf, DISABLED_Shapes) {
    std::vector<KernelEntry> kernels = {
        // NEON path.
        INT_VAR(4, 4),
        INT_VAR(4, 8),
        INT_VAR(4, 16),
        INT_VAR(8, 4),
        INT_VAR(8, 8),
        INT_VAR(8, 16),
        INT_VAR(8, 32),
        INT_VAR(16, 4),
        INT_VAR(16, 8),
        INT_VAR(16, 16),
        INT_VAR(16, 32),
        INT_VAR(16, 64),
        INT_VAR(32, 8),
        INT_VAR(32, 16),
        INT_VAR(32, 32),
        INT_VAR(32, 64),
        INT_VAR(64, 16),
        INT_VAR(64, 32),
        INT_VAR(64, 64),
        INT_VAR(64, 128),
        INT_VAR(128, 64),
        INT_VAR(128, 128),
        // NEON_DOTPROD path. No 4x4 kernel exists in the dotprod variant.
        INT_VAR_DP(4, 8),
        INT_VAR_DP(4, 16),
        INT_VAR_DP(8, 4),
        INT_VAR_DP(8, 8),
        INT_VAR_DP(8, 16),
        INT_VAR_DP(8, 32),
        INT_VAR_DP(16, 4),
        INT_VAR_DP(16, 8),
        INT_VAR_DP(16, 16),
        INT_VAR_DP(16, 32),
        INT_VAR_DP(16, 64),
        INT_VAR_DP(32, 8),
        INT_VAR_DP(32, 16),
        INT_VAR_DP(32, 32),
        INT_VAR_DP(32, 64),
        INT_VAR_DP(64, 16),
        INT_VAR_DP(64, 32),
        INT_VAR_DP(64, 64),
        INT_VAR_DP(64, 128),
        INT_VAR_DP(128, 64),
        INT_VAR_DP(128, 128),
    };

    std::map<int, std::pair<double, uint64_t>> by_w_neon;
    std::map<int, std::pair<double, uint64_t>> by_w_dp;

    for (const auto &k : kernels) {
        const bool is_dp =
            std::string(k.name).find("_dotprod") != std::string::npos;
        const double avg_ns = run_cell(k, src_, src_stride_, ref_, ref_stride_);
        auto &w = is_dp ? by_w_dp[k.w] : by_w_neon[k.w];
        w.first += avg_ns;
        w.second += 1;
    }

    auto print_w = [](const char *label,
                      const std::map<int, std::pair<double, uint64_t>> &m) {
        for (const auto &kv : m) {
            const int w = kv.first;
            const double sum = kv.second.first;
            const uint64_t n = kv.second.second;
            const double mean = (n > 0) ? (sum / static_cast<double>(n)) : 0.0;
            std::cout << "  " << std::left << std::setw(48) << label
                      << " W=" << std::right << std::setw(3) << w
                      << "  iters=" << std::setw(7) << n
                      << "  avg=" << std::fixed << std::setprecision(2)
                      << std::setw(8) << mean << " ns" << std::endl;
        }
    };

    std::cout << std::endl;
    std::cout << "  --- Average ns by W ---" << std::endl;
    std::cout << "  -- NEON --" << std::endl;
    print_w("W aggregate [neon]", by_w_neon);
    std::cout << "  -- NEON_DOTPROD --" << std::endl;
    print_w("W aggregate [neon_dotprod]", by_w_dp);

    std::cout << "  sink=0x" << std::hex << std::setw(16) << std::setfill('0')
              << static_cast<unsigned long long>(g_sink) << std::dec
              << std::setfill(' ') << std::endl;
}

#undef INT_VAR
#undef INT_VAR_DP

}  // namespace

#endif  // ARCH_AARCH64
