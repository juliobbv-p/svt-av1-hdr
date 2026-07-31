/*
 *  Copyright (c) 2019, Alliance for Open Media. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <algorithm>
#include <ostream>

#include "aom_dsp_rtcd.h"
#include "test/acm_random.h"
#include "util.h"

namespace {

using libaom_test::ACMRandom;

using HadamardFunc = void (*)(const int16_t *a, ptrdiff_t a_stride, int32_t *b);

template <typename HadamardFuncType>
struct FuncWithSize {
    explicit FuncWithSize(HadamardFuncType f, int s) : func(f), block_size(s) {
    }
    HadamardFuncType func;
    int block_size;
};

using HadamardFuncWithSize = FuncWithSize<HadamardFunc>;

template <typename HadamardFuncType>
std::ostream &operator<<(std::ostream &os,
                         const FuncWithSize<HadamardFuncType> &hfs) {
    return os << "block size: " << hfs.block_size;
}

template <typename OutputType, typename HadamardFuncType>
class HadamardTestBase
    : public ::testing::TestWithParam<FuncWithSize<HadamardFuncType>> {
  public:
    explicit HadamardTestBase(const FuncWithSize<HadamardFuncType> &func_param,
                              HadamardFuncType ref_func)
        : bwh_(func_param.block_size),
          block_size_(bwh_ * bwh_),
          h_func_(func_param.func),
          h_ref_func_(ref_func) {
    }

    virtual void SetUp() {
        rnd_.Reset(ACMRandom::DeterministicSeed());
    }

    virtual int16_t Rand() = 0;

    void CompareReferenceRandom() {
        const int kMaxBlockSize = 32 * 32;
        DECLARE_ALIGNED(16, int16_t, a[kMaxBlockSize]);
        DECLARE_ALIGNED(16, OutputType, b[kMaxBlockSize]);
        memset(a, 0, sizeof(a));
        memset(b, 0, sizeof(b));

        OutputType b_ref[kMaxBlockSize];
        memset(b_ref, 0, sizeof(b_ref));

        for (int i = 0; i < block_size_; ++i)
            a[i] = Rand();

        h_ref_func_(a, bwh_, b_ref);
        h_func_(a, bwh_, b);

        // The order of the output is not important. Sort before checking.
        std::sort(b, b + block_size_);
        std::sort(b_ref, b_ref + block_size_);
        EXPECT_EQ(0, memcmp(b, b_ref, sizeof(b)));
    }

    void VaryStride() {
        const int kMaxBlockSize = 32 * 32;
        DECLARE_ALIGNED(16, int16_t, a[kMaxBlockSize * 8]);
        DECLARE_ALIGNED(16, OutputType, b[kMaxBlockSize]);
        memset(a, 0, sizeof(a));
        for (int i = 0; i < kMaxBlockSize * 8; ++i)
            a[i] = Rand();

        OutputType b_ref[kMaxBlockSize];
        for (int i = bwh_; i < 64; i += 4) {
            memset(b, 0, sizeof(b));
            memset(b_ref, 0, sizeof(b_ref));

            h_ref_func_(a, i, b_ref);
            h_func_(a, i, b);

            // The order of the output is not important. Sort before checking.
            std::sort(b, b + block_size_);
            std::sort(b_ref, b_ref + block_size_);
            EXPECT_EQ(0, memcmp(b, b_ref, sizeof(b)));
        }
    }

    ACMRandom rnd_;

  private:
    int bwh_;
    int block_size_;
    HadamardFuncType h_func_;
    HadamardFuncType h_ref_func_;
};

class HadamardLowbdTest : public HadamardTestBase<int32_t, HadamardFunc> {
  public:
    HadamardLowbdTest()
        : HadamardTestBase(GetParam(),
                           GetReferenceFunc(GetParam().block_size)) {
    }
    virtual int16_t Rand() {
        return rnd_.Rand9Signed();
    }

  private:
    static HadamardFunc GetReferenceFunc(int block_size) {
        switch (block_size) {
        case 4: return svt_aom_hadamard_4x4_c;
        case 8: return svt_aom_hadamard_8x8_c;
        case 16: return svt_aom_hadamard_16x16_c;
        case 32: return svt_aom_hadamard_32x32_c;
        default: return nullptr;
        }
    }
};

TEST_P(HadamardLowbdTest, CompareReferenceRandom) {
    CompareReferenceRandom();
}

TEST_P(HadamardLowbdTest, VaryStride) {
    VaryStride();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, HadamardLowbdTest,
    ::testing::Values(HadamardFuncWithSize(&svt_aom_hadamard_4x4_sse2, 4),
                      HadamardFuncWithSize(&svt_aom_hadamard_8x8_sse2, 8),
                      HadamardFuncWithSize(&svt_aom_hadamard_16x16_avx2, 16),
                      HadamardFuncWithSize(&svt_aom_hadamard_32x32_avx2, 32)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, HadamardLowbdTest,
    ::testing::Values(HadamardFuncWithSize(&svt_aom_hadamard_4x4_neon, 4),
                      HadamardFuncWithSize(&svt_aom_hadamard_8x8_neon, 8),
                      HadamardFuncWithSize(&svt_aom_hadamard_16x16_neon, 16),
                      HadamardFuncWithSize(&svt_aom_hadamard_32x32_neon, 32)));
#endif  // ARCH_AARCH64

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
class HadamardHighbdTest : public HadamardTestBase<int32_t, HadamardFunc> {
  protected:
    HadamardHighbdTest()
        : HadamardTestBase(GetParam(), svt_aom_highbd_hadamard_8x8_c) {
    }
    // Use values between -4095 (0xF001) and 4095 (0x0FFF)
    int16_t Rand() override {
        int16_t src = rnd_.Rand12();
        int16_t pred = rnd_.Rand12();
        return src - pred;
    }
};

TEST_P(HadamardHighbdTest, CompareReferenceRandom) {
    CompareReferenceRandom();
}

TEST_P(HadamardHighbdTest, VaryStride) {
    VaryStride();
}

#if ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(AVX2, HadamardHighbdTest,
                         ::testing::Values(HadamardFuncWithSize(
                             &svt_aom_highbd_hadamard_8x8_avx2, 8)));
#endif  // ARCH_X86_64

#if ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(NEON, HadamardHighbdTest,
                         ::testing::Values(HadamardFuncWithSize(
                             &svt_aom_highbd_hadamard_8x8_neon, 8)));
#endif  // ARCH_AARCH64

#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH

}  // namespace
