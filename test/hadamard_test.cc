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
#include <type_traits>

#include "aom_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include "test/acm_random.h"
#include "util.h"

namespace {

using libaom_test::ACMRandom;

using HadamardFunc = void (*)(const int16_t *a, ptrdiff_t a_stride, int32_t *b);
using HadamardSatdFunc = int (*)(const uint8_t *src, ptrdiff_t src_stride,
                                 const uint8_t *pred, ptrdiff_t pred_stride);
using HighbdHadamardSatdFunc = int (*)(const uint16_t *src,
                                       ptrdiff_t src_stride,
                                       const uint16_t *pred,
                                       ptrdiff_t pred_stride);

template <typename HadamardFuncType>
struct FuncWithSize {
    explicit FuncWithSize(HadamardFuncType f, int s) : func(f), block_size(s) {
    }
    HadamardFuncType func;
    int block_size;
};

using HadamardFuncWithSize = FuncWithSize<HadamardFunc>;
using HadamardSatdFuncWithSize = FuncWithSize<HadamardSatdFunc>;
using HighbdHadamardSatdFuncWithSize = FuncWithSize<HighbdHadamardSatdFunc>;

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

    void CompareReferenceVaryStride() {
        const int kMaxBlockSize = 32 * 32;
        DECLARE_ALIGNED(16, int16_t, a[kMaxBlockSize * 8]);
        DECLARE_ALIGNED(16, OutputType, b[kMaxBlockSize]);
        memset(a, 0, sizeof(a));
        for (int i = 0; i < kMaxBlockSize * 8; ++i)
            a[i] = Rand();

        OutputType b_ref[kMaxBlockSize];
        const int kStrides[] = {4, 8, 16, 24, 32, 40, 48, 56};
        for (const int stride : kStrides) {
            if (stride == 4 && bwh_ != 4)
                continue;

            memset(b, 0, sizeof(b));
            memset(b_ref, 0, sizeof(b_ref));

            h_ref_func_(a, stride, b_ref);
            h_func_(a, stride, b);

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

TEST_P(HadamardLowbdTest, CompareReferenceVaryStride) {
    CompareReferenceVaryStride();
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

#ifdef ARCH_AARCH64
template <typename PixelType, typename HadamardSatdFuncType>
class HadamardSatdTestBase
    : public ::testing::TestWithParam<FuncWithSize<HadamardSatdFuncType>> {
  public:
    HadamardSatdTestBase(const FuncWithSize<HadamardSatdFuncType> &func_param,
                         HadamardSatdFuncType ref_func)
        : bwh_(func_param.block_size),
          block_size_(bwh_ * bwh_),
          max_value_(std::is_same<PixelType, uint8_t>::value ? 255 : 1023),
          satd_func_(func_param.func),
          satd_ref_func_(ref_func) {
    }

    void SetUp() override {
        rnd_.Reset(ACMRandom::DeterministicSeed());
    }

    void CompareReferenceVaryStride() {
        const int kMaxBlockSize = 32 * 32;
        DECLARE_ALIGNED(16, PixelType, src[kMaxBlockSize * 8]);
        DECLARE_ALIGNED(16, PixelType, pred[kMaxBlockSize * 8]);
        FillRandom(src, pred, kMaxBlockSize * 8);

        // Force RTCD dispatch to C so the C SATD reference does not call
        // optimized Hadamard function pointers while comparing against the SIMD
        // implementation.
        svt_aom_setup_common_rtcd_internal(0);
        svt_aom_setup_rtcd_internal(0);
        for (int stride = 8; stride < 64; stride += 8) {
            if (stride < bwh_)
                continue;

            const int satd_ref = satd_ref_func_(src, stride, pred, stride);
            const int satd = satd_func_(src, stride, pred, stride);
            EXPECT_EQ(satd_ref, satd);
        }
    }

    void ExtremeValues() {
        const int kMaxBlockSize = 32 * 32;
        DECLARE_ALIGNED(16, PixelType, src[kMaxBlockSize]);
        DECLARE_ALIGNED(16, PixelType, pred[kMaxBlockSize]);

        // Force RTCD dispatch to C so the C SATD reference does not call
        // optimized Hadamard function pointers while comparing against the SIMD
        // implementation.
        svt_aom_setup_common_rtcd_internal(0);
        svt_aom_setup_rtcd_internal(0);
        for (int pattern = 0; pattern < 4; ++pattern) {
            FillExtreme(src, pred, block_size_, pattern);

            const int satd_ref = satd_ref_func_(src, bwh_, pred, bwh_);
            const int satd = satd_func_(src, bwh_, pred, bwh_);
            EXPECT_EQ(satd_ref, satd);
        }
    }

  private:
    void FillRandom(PixelType *src, PixelType *pred, int count) {
        for (int i = 0; i < count; ++i) {
            src[i] = static_cast<PixelType>(rnd_.Rand16() & max_value_);
            pred[i] = static_cast<PixelType>(rnd_.Rand16() & max_value_);
        }
    }

    void FillExtreme(PixelType *src, PixelType *pred, int count, int pattern) {
        for (int i = 0; i < count; ++i) {
            const bool high = pattern < 2 ? pattern == 0 : ((i + pattern) & 1);
            src[i] = static_cast<PixelType>(high ? max_value_ : 0);
            pred[i] = static_cast<PixelType>(high ? 0 : max_value_);
        }
    }

    ACMRandom rnd_;
    int bwh_;
    int block_size_;
    int max_value_;
    HadamardSatdFuncType satd_func_;
    HadamardSatdFuncType satd_ref_func_;
};

class HadamardSatdTest
    : public HadamardSatdTestBase<uint8_t, HadamardSatdFunc> {
  public:
    HadamardSatdTest()
        : HadamardSatdTestBase(GetParam(),
                               GetReferenceFunc(GetParam().block_size)) {
    }

  private:
    static HadamardSatdFunc GetReferenceFunc(int block_size) {
        switch (block_size) {
        case 4: return svt_av1_hadamard_satd_4x4_c;
        case 8: return svt_av1_hadamard_satd_8x8_c;
        case 16: return svt_av1_hadamard_satd_16x16_c;
        case 32: return svt_av1_hadamard_satd_32x32_c;
        default: return nullptr;
        }
    }
};

TEST_P(HadamardSatdTest, CompareReferenceVaryStride) {
    CompareReferenceVaryStride();
}

TEST_P(HadamardSatdTest, ExtremeValues) {
    ExtremeValues();
}

INSTANTIATE_TEST_SUITE_P(
    NEON, HadamardSatdTest,
    ::testing::Values(
        HadamardSatdFuncWithSize(&svt_av1_hadamard_satd_4x4_neon, 4),
        HadamardSatdFuncWithSize(&svt_av1_hadamard_satd_8x8_neon, 8),
        HadamardSatdFuncWithSize(&svt_av1_hadamard_satd_16x16_neon, 16),
        HadamardSatdFuncWithSize(&svt_av1_hadamard_satd_32x32_neon, 32)));
#endif  // ARCH_AARCH64

#if CONFIG_ENABLE_HIGH_BIT_DEPTH

#if ARCH_AARCH64
class HighbdHadamardSatdTest
    : public HadamardSatdTestBase<uint16_t, HighbdHadamardSatdFunc> {
  public:
    HighbdHadamardSatdTest()
        : HadamardSatdTestBase(GetParam(),
                               GetReferenceFunc(GetParam().block_size)) {
    }

  private:
    static HighbdHadamardSatdFunc GetReferenceFunc(int block_size) {
        switch (block_size) {
        case 4: return svt_av1_highbd_hadamard_satd_4x4_c;
        case 8: return svt_av1_highbd_hadamard_satd_8x8_c;
        case 16: return svt_av1_highbd_hadamard_satd_16x16_c;
        case 32: return svt_av1_highbd_hadamard_satd_32x32_c;
        default: return nullptr;
        }
    }
};

TEST_P(HighbdHadamardSatdTest, CompareReferenceVaryStride) {
    CompareReferenceVaryStride();
}

TEST_P(HighbdHadamardSatdTest, ExtremeValues) {
    ExtremeValues();
}

INSTANTIATE_TEST_SUITE_P(
    NEON, HighbdHadamardSatdTest,
    ::testing::Values(
        HighbdHadamardSatdFuncWithSize(&svt_av1_highbd_hadamard_satd_4x4_neon,
                                       4),
        HighbdHadamardSatdFuncWithSize(&svt_av1_highbd_hadamard_satd_8x8_neon,
                                       8),
        HighbdHadamardSatdFuncWithSize(&svt_av1_highbd_hadamard_satd_16x16_neon,
                                       16),
        HighbdHadamardSatdFuncWithSize(&svt_av1_highbd_hadamard_satd_32x32_neon,
                                       32)));
#endif  // ARCH_AARCH64

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

TEST_P(HadamardHighbdTest, CompareReferenceVaryStride) {
    CompareReferenceVaryStride();
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
