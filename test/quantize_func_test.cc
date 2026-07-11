/*
 * Copyright(c) 2019 Netflix, Inc.
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>
#include "aom_dsp_rtcd.h"
#include "md_config_process.h"
#include "random.h"
#include "util.h"
#include "definitions.h"
#include "pcs.h"
#include "q_matrices.h"

namespace {
using std::make_tuple;
using svt_av1_test_tool::SVTRandom;

template <typename T>
class aligned_allocator {
  public:
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    aligned_allocator() noexcept = default;

    template <typename U>
    explicit aligned_allocator(const aligned_allocator<U> &) noexcept {
    }

    template <typename U>
    struct rebind {
        using other = aligned_allocator<U>;
    };

    static pointer allocate(size_type n, const void * = nullptr) {
        if (T *ptr = reinterpret_cast<T *>(svt_aom_memalign(32, n * sizeof(T))))
            return ptr;
        throw std::bad_alloc();
    }

    static void deallocate(pointer ptr, size_type) noexcept {
        svt_aom_free(ptr);
    }
};

template <typename T, typename U>
bool operator==(const aligned_allocator<T> &, const aligned_allocator<U> &) {
    return true;
}

template <typename T, typename U>
bool operator!=(const aligned_allocator<T> &, const aligned_allocator<U> &) {
    return false;
}

#define QUAN_PARAM_LIST                                                      \
    const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,    \
        const int16_t *round_ptr, const int16_t *quant_ptr,                  \
        const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr,                 \
        TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, \
        const int16_t *scan, const int16_t *iscan
#define QUAN_HBD_PARAM int16_t log_scale
#define QUAN_QM_PARAM_LIST                                                   \
    const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr,    \
        const int16_t *round_ptr, const int16_t *quant_ptr,                  \
        const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr,                 \
        TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, \
        const int16_t *scan, const int16_t *iscan, const QmVal *qm_ptr,      \
        const QmVal *iqm_ptr

using QuantizeFunc = void (*)(QUAN_PARAM_LIST);
using QuantizeHbdFunc = void (*)(QUAN_PARAM_LIST, QUAN_HBD_PARAM);
using QuantizeQmFunc = void (*)(QUAN_QM_PARAM_LIST, QUAN_HBD_PARAM);

enum QuantType { TYPE_B, TYPE_DC, TYPE_FP };

using QuantizeParam =
    std::tuple<QuantizeFunc, QuantizeFunc, TxSize, QuantType, EbBitDepth>;
using QuantizeHbdParam =
    std::tuple<QuantizeHbdFunc, QuantizeHbdFunc, TxSize, QuantType, EbBitDepth>;
using QuantizeQmParam =
    std::tuple<QuantizeQmFunc, QuantizeQmFunc, TxSize, QuantType, EbBitDepth>;

struct alignas(32) QuanTable {
    Quants quant;
    Dequants dequant;
};

constexpr int kTestNum = 1000;

template <typename ParamType, typename FuncType>
class QuantizeTest : public ::testing::TestWithParam<ParamType> {
  public:
    void *operator new(size_t size) {
        if (void *ptr = svt_aom_memalign(alignof(QuantizeTest), size))
            return ptr;
        throw std::bad_alloc();
    }

    void operator delete(void *ptr) {
        svt_aom_free(ptr);
    }

    QuantizeTest() {
        PictureParentControlSet pcs;
        pcs.frm_hdr.quantization_params.base_q_idx = 0;
        SequenceControlSet scs{};
        pcs.scs = &scs;
        scs.static_config.sharpness = 0;
        svt_av1_build_quantizer(
            &pcs, bd_, 0, 0, 0, 0, 0, &qtab_.quant, &qtab_.dequant);
        const int n_coeffs = coeff_num();
        coeff_.resize(6 * n_coeffs);
    }

  protected:
    int coeff_num() const {
        return av1_get_max_eob(tx_size_);
    }

    void FillCoeff(TranLow c) {
        const int n_coeffs = coeff_num();
        std::fill_n(coeff_.begin(), n_coeffs, c);
    }

    void FillCoeffRandom() {
        const int n_coeffs = coeff_num();
        const int num = rnd_.random() % n_coeffs;
        FillCoeffRandomRows(num);
    }

    void FillCoeffRandomRows(int num) {
        FillCoeffZero();
        std::generate_n(
            coeff_.begin(), num, [this]() { return GetRandomCoeff(); });
    }

    void FillCoeffZero() {
        FillCoeff(0);
    }

    void FillDcOnly() {
        FillCoeffZero();
        coeff_[0] = GetRandomCoeff();
    }

    void FillDcLargeNegative() {
        FillCoeffZero();
        // Generate a qcoeff which contains 512/-512 (0x0100/0xFE00) to catch
        // issues like BUG=883 where the constant being compared was incorrectly
        // initialized.
        coeff_[0] = -8191;
    }

    TranLow GetRandomCoeff() {
        if (bd_ == EB_EIGHT_BIT) {
            return clamp(
                static_cast<int16_t>(rnd_.random()), INT16_MIN + 1, INT16_MAX);
        }
        const TranLow min = -(1 << (7 + bd_));
        const TranLow max = -min - 1;
        return clamp(static_cast<TranLow>(rnd_.random()), min, max);
    }

    SVTRandom rnd_{32, true};
    QuanTable qtab_{};
    std::vector<TranLow, aligned_allocator<TranLow>> coeff_{};
    TxSize tx_size_{TEST_GET_PARAM(2)};
    EbBitDepth bd_{TEST_GET_PARAM(4)};
};

class QuantizeLbdTest : public QuantizeTest<QuantizeParam, QuantizeFunc> {
  protected:
    template <bool is_loop>
    void QuantizeRun(int q = 0, int test_num = 1) {
        const auto quant_ref = TEST_GET_PARAM(0);
        const auto quant_tst = TEST_GET_PARAM(1);
        const auto type = TEST_GET_PARAM(3);

        TranLow *coeff_ptr = coeff_.data();
        const intptr_t n_coeffs = coeff_num();

        TranLow *qcoeff_ref = coeff_ptr + n_coeffs;
        TranLow *dqcoeff_ref = qcoeff_ref + n_coeffs;

        TranLow *qcoeff = dqcoeff_ref + n_coeffs;
        TranLow *dqcoeff = qcoeff + n_coeffs;
        uint16_t *eob = reinterpret_cast<uint16_t *>(dqcoeff + n_coeffs);

        // Testing uses 2-D DCT scan order table
        const ScanOrder *const sc = get_scan_order(tx_size_, DCT_DCT);

        // Testing uses luminance quantization table
        const int16_t *zbin = qtab_.quant.y_zbin[q];

        const int16_t *round = 0;
        const int16_t *quant = 0;
        if (type == TYPE_B) {
            round = qtab_.quant.y_round[q];
            quant = qtab_.quant.y_quant[q];
        } else if (type == TYPE_FP) {
            round = qtab_.quant.y_round_fp[q];
            quant = qtab_.quant.y_quant_fp[q];
        }

        const int16_t *quant_shift = qtab_.quant.y_quant_shift[q];
        const int16_t *dequant = qtab_.dequant.y_dequant_qtx[q];

        for (int i = 0; i < test_num; ++i) {
            if (is_loop)
                FillCoeffRandom();

            std::fill_n(qcoeff_ref, 5 * n_coeffs, 0);

            quant_ref(coeff_ptr,
                      n_coeffs,
                      zbin,
                      round,
                      quant,
                      quant_shift,
                      qcoeff_ref,
                      dqcoeff_ref,
                      dequant,
                      &eob[0],
                      sc->scan,
                      sc->iscan);

            quant_tst(coeff_ptr,
                      n_coeffs,
                      zbin,
                      round,
                      quant,
                      quant_shift,
                      qcoeff,
                      dqcoeff,
                      dequant,
                      &eob[1],
                      sc->scan,
                      sc->iscan);

            for (int j = 0; j < n_coeffs; ++j) {
                ASSERT_EQ(qcoeff_ref[j], qcoeff[j])
                    << "Q mismatch on test: " << i << " at position: " << j
                    << " Q: " << q << " coeff: " << coeff_ptr[j];
            }

            for (int j = 0; j < n_coeffs; ++j) {
                ASSERT_EQ(dqcoeff_ref[j], dqcoeff[j])
                    << "Dq mismatch on test: " << i << " at position: " << j
                    << " Q: " << q << " coeff: " << coeff_ptr[j];
            }

            ASSERT_EQ(eob[0], eob[1])
                << "eobs mismatch on test: " << i << " Q: " << q;
        }
    }

    // Microbenchmark for the flat (no quantization matrix) quantizer: C
    // reference vs SIMD on a dense coefficient block. Provides context for the
    // qm-kernel speedups. Disabled by default.
    void RunSpeedTest() {
        const auto quant_ref = TEST_GET_PARAM(0);
        const auto quant_tst = TEST_GET_PARAM(1);
        const auto type = TEST_GET_PARAM(3);

        TranLow *coeff_ptr = coeff_.data();
        const intptr_t n_coeffs = coeff_num();
        TranLow *qcoeff_ref = coeff_ptr + n_coeffs;
        TranLow *dqcoeff_ref = qcoeff_ref + n_coeffs;
        TranLow *qcoeff = dqcoeff_ref + n_coeffs;
        TranLow *dqcoeff = qcoeff + n_coeffs;
        uint16_t *eob = reinterpret_cast<uint16_t *>(dqcoeff + n_coeffs);

        const ScanOrder *const sc = get_scan_order(tx_size_, DCT_DCT);
        const int q = 0;
        const int16_t *zbin = qtab_.quant.y_zbin[q];
        const int16_t *round = (type == TYPE_B) ? qtab_.quant.y_round[q]
                                                : qtab_.quant.y_round_fp[q];
        const int16_t *quant = (type == TYPE_B) ? qtab_.quant.y_quant[q]
                                                : qtab_.quant.y_quant_fp[q];
        const int16_t *quant_shift = qtab_.quant.y_quant_shift[q];
        const int16_t *dequant = qtab_.dequant.y_dequant_qtx[q];

        FillCoeffRandomRows(static_cast<int>(n_coeffs));  // dense block
        std::fill_n(qcoeff_ref, 5 * n_coeffs, 0);

        auto run =
            [&](QuantizeFunc fn, TranLow *qc, TranLow *dqc, uint16_t *e) {
                fn(coeff_ptr,
                   n_coeffs,
                   zbin,
                   round,
                   quant,
                   quant_shift,
                   qc,
                   dqc,
                   dequant,
                   e,
                   sc->scan,
                   sc->iscan);
            };
        const uint64_t num_loop = 300000;
        for (int i = 0; i < 4000; ++i) {
            run(quant_ref, qcoeff_ref, dqcoeff_ref, &eob[0]);
            run(quant_tst, qcoeff, dqcoeff, &eob[1]);
        }
        auto t0 = std::chrono::steady_clock::now();
        for (uint64_t i = 0; i < num_loop; ++i)
            run(quant_ref, qcoeff_ref, dqcoeff_ref, &eob[0]);
        auto t1 = std::chrono::steady_clock::now();
        for (uint64_t i = 0; i < num_loop; ++i)
            run(quant_tst, qcoeff, dqcoeff, &eob[1]);
        auto t2 = std::chrono::steady_clock::now();
        const double c_ns =
            std::chrono::duration<double, std::nano>(t1 - t0).count() /
            num_loop;
        const double n_ns =
            std::chrono::duration<double, std::nano>(t2 - t1).count() /
            num_loop;
        printf(
            "[ SPEED    ] tx=%2d bd=%2d n=%4d : C %8.1f ns  SIMD %8.1f ns  "
            "speedup %.2fx (flat)\n",
            static_cast<int>(tx_size_),
            static_cast<int>(bd_),
            static_cast<int>(n_coeffs),
            c_ns,
            n_ns,
            c_ns / n_ns);
    }
};

TEST_P(QuantizeLbdTest, DISABLED_Speed) {
    RunSpeedTest();
}

TEST_P(QuantizeLbdTest, ZeroInput) {
    FillCoeffZero();
    QuantizeRun<false>();
}

TEST_P(QuantizeLbdTest, LargeNegativeInput) {
    FillDcLargeNegative();
    QuantizeRun<false>(0, 1);
}

TEST_P(QuantizeLbdTest, DcOnlyInput) {
    FillDcOnly();
    QuantizeRun<false>(0, 1);
}

TEST_P(QuantizeLbdTest, RandomInput) {
    QuantizeRun<true>(0, kTestNum);
}

TEST_P(QuantizeLbdTest, MultipleQ) {
    for (int q = 0; q < QINDEX_RANGE; ++q) {
        QuantizeRun<true>(q, kTestNum);
    }
}

// Force the coeff to be half the value of the dequant.  This exposes a
// mismatch found in av1_quantize_fp_sse2().
TEST_P(QuantizeLbdTest, CoeffHalfDequant) {
    FillCoeff(16);
    QuantizeRun<false>(25, 1);
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
class QuantizeHbdTest : public QuantizeTest<QuantizeHbdParam, QuantizeHbdFunc> {
  protected:
    template <bool is_loop>
    void QuantizeRun(int q = 0, int test_num = 1) {
        const auto quant_ref = TEST_GET_PARAM(0);
        const auto quant_tst = TEST_GET_PARAM(1);
        const auto type = TEST_GET_PARAM(3);
        TranLow *coeff_ptr = coeff_.data();
        const intptr_t n_coeffs = coeff_num();

        TranLow *qcoeff_ref = coeff_ptr + n_coeffs;
        TranLow *dqcoeff_ref = qcoeff_ref + n_coeffs;

        TranLow *qcoeff = dqcoeff_ref + n_coeffs;
        TranLow *dqcoeff = qcoeff + n_coeffs;
        uint16_t *eob = reinterpret_cast<uint16_t *>(dqcoeff + n_coeffs);

        // Testing uses 2-D DCT scan order table
        const ScanOrder *const sc = get_scan_order(tx_size_, DCT_DCT);

        // Testing uses luminance quantization table
        const int16_t *zbin = qtab_.quant.y_zbin[q];

        const int16_t *round = 0;
        const int16_t *quant = 0;
        if (type == TYPE_B) {
            round = qtab_.quant.y_round[q];
            quant = qtab_.quant.y_quant[q];
        } else if (type == TYPE_FP) {
            round = qtab_.quant.y_round_fp[q];
            quant = qtab_.quant.y_quant_fp[q];
        }

        const int16_t *quant_shift = qtab_.quant.y_quant_shift[q];
        const int16_t *dequant = qtab_.dequant.y_dequant_qtx[q];

        for (int i = 0; i < test_num; ++i) {
            if (is_loop)
                FillCoeffRandom();

            std::fill_n(qcoeff_ref, 5 * n_coeffs, 0);

            quant_ref(coeff_ptr,
                      n_coeffs,
                      zbin,
                      round,
                      quant,
                      quant_shift,
                      qcoeff_ref,
                      dqcoeff_ref,
                      dequant,
                      &eob[0],
                      sc->scan,
                      sc->iscan,
                      av1_get_tx_scale(tx_size_));

            quant_tst(coeff_ptr,
                      n_coeffs,
                      zbin,
                      round,
                      quant,
                      quant_shift,
                      qcoeff,
                      dqcoeff,
                      dequant,
                      &eob[1],
                      sc->scan,
                      sc->iscan,
                      av1_get_tx_scale(tx_size_));

            for (int j = 0; j < n_coeffs; ++j) {
                ASSERT_EQ(qcoeff_ref[j], qcoeff[j])
                    << "Q mismatch on test: " << i << " at position: " << j
                    << " Q: " << q << " coeff: " << coeff_ptr[j];
            }

            for (int j = 0; j < n_coeffs; ++j) {
                ASSERT_EQ(dqcoeff_ref[j], dqcoeff[j])
                    << "Dq mismatch on test: " << i << " at position: " << j
                    << " Q: " << q << " coeff: " << coeff_ptr[j];
            }

            ASSERT_EQ(eob[0], eob[1])
                << "eobs mismatch on test: " << i << " Q: " << q;
        }
    }
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantizeHbdTest);

TEST_P(QuantizeHbdTest, ZeroInput) {
    FillCoeffZero();
    QuantizeRun<false>();
}

TEST_P(QuantizeHbdTest, LargeNegativeInput) {
    FillDcLargeNegative();
    QuantizeRun<false>(0, 1);
}

TEST_P(QuantizeHbdTest, DcOnlyInput) {
    FillDcOnly();
    QuantizeRun<false>(0, 1);
}

TEST_P(QuantizeHbdTest, RandomInput) {
    QuantizeRun<true>(0, kTestNum);
}

TEST_P(QuantizeHbdTest, MultipleQ) {
    for (int q = 0; q < QINDEX_RANGE; ++q) {
        QuantizeRun<true>(q, kTestNum);
    }
}

// Force the coeff to be half the value of the dequant.  This exposes a
// mismatch found in av1_quantize_fp_sse2().
TEST_P(QuantizeHbdTest, CoeffHalfDequant) {
    FillCoeff(16);
    QuantizeRun<false>(25, 1);
}

#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH

class QuantizeQmTest : public QuantizeTest<QuantizeQmParam, QuantizeQmFunc> {
  protected:
    QuantizeQmTest() {
#if CONFIG_ENABLE_QUANT_MATRIX
        constexpr uint8_t num_planes = 1;
        for (uint8_t q = 0; q < NUM_QM_LEVELS; ++q) {
            for (uint8_t c = 0; c < num_planes; ++c) {
                int32_t current = 0;
                auto &qmatrix_qc = qmatrix_[q][c];
                auto &iqmatrix_qc = iqmatrix_[q][c];
                for (uint8_t t = 0; t < TX_SIZES_ALL; ++t) {
                    const int32_t size = tx_size_2d[t];
                    const TxSize qm_tx_size =
                        av1_get_adjusted_tx_size(static_cast<TxSize>(t));
                    if (q == NUM_QM_LEVELS - 1) {
                        qmatrix_qc[t] = NULL;
                        iqmatrix_qc[t] = NULL;
                    } else if (t != qm_tx_size) {
                        // Reuse matrices for 'qm_tx_size'
                        qmatrix_qc[t] = qmatrix_qc[qm_tx_size];
                        iqmatrix_qc[t] = iqmatrix_qc[qm_tx_size];
                    } else {
                        assert(current + size <= QM_TOTAL_SIZE);
                        // The following lines are suppressed since it could
                        // break if we set num_planes to 3
                        // cppcheck-suppress knownConditionTrueFalse
                        qmatrix_qc[t] = &wt_matrix_ref[q][c >= 1][current];
                        // cppcheck-suppress knownConditionTrueFalse
                        iqmatrix_qc[t] = &iwt_matrix_ref[q][c >= 1][current];

                        current += size;
                    }
                }
            }
        }
#endif  // CONFIG_ENABLE_QUANT_MATRIX
    }

    template <bool is_loop>
    void QuantizeRun(int q = 0, int test_num = 1) {
        const auto quant_ref = TEST_GET_PARAM(0);
        const auto quant_tst = TEST_GET_PARAM(1);

        TranLow *coeff_ptr = coeff_.data();
        const intptr_t n_coeffs = coeff_num();

        TranLow *qcoeff_ref = coeff_ptr + n_coeffs;
        TranLow *dqcoeff_ref = qcoeff_ref + n_coeffs;

        TranLow *qcoeff = dqcoeff_ref + n_coeffs;
        TranLow *dqcoeff = qcoeff + n_coeffs;
        uint16_t *eob = reinterpret_cast<uint16_t *>(dqcoeff + n_coeffs);

        // Testing uses 2-D DCT scan order table
        const ScanOrder *const sc = get_scan_order(tx_size_, DCT_DCT);

        // Testing uses luminance quantization table
        const int16_t *zbin = qtab_.quant.y_zbin[q];

        // ASSERT_EQ(type_ == TYPE_FP);
        const int16_t *round = qtab_.quant.y_round_fp[q];
        const int16_t *quant = qtab_.quant.y_quant_fp[q];

        const int16_t *quant_shift = qtab_.quant.y_quant_shift[q];
        const int16_t *dequant = qtab_.dequant.y_dequant_qtx[q];

        const TxSize qm_tx_size = av1_get_adjusted_tx_size(tx_size_);
        const QmVal *qm_ptr = qmatrix_[qm_level_][0][qm_tx_size];
        const QmVal *iqm_ptr = iqmatrix_[qm_level_][0][qm_tx_size];

        for (int i = 0; i < test_num; ++i) {
            if (is_loop)
                FillCoeffRandom();

            std::fill_n(qcoeff_ref, 5 * n_coeffs, 0);
            const int log_scale = av1_get_tx_scale(tx_size_);

            quant_ref(coeff_ptr,
                      n_coeffs,
                      zbin,
                      round,
                      quant,
                      quant_shift,
                      qcoeff_ref,
                      dqcoeff_ref,
                      dequant,
                      &eob[0],
                      sc->scan,
                      sc->iscan,
                      qm_ptr,
                      iqm_ptr,
                      log_scale);

            quant_tst(coeff_ptr,
                      n_coeffs,
                      zbin,
                      round,
                      quant,
                      quant_shift,
                      qcoeff,
                      dqcoeff,
                      dequant,
                      &eob[1],
                      sc->scan,
                      sc->iscan,
                      qm_ptr,
                      iqm_ptr,
                      log_scale);

            for (int j = 0; j < n_coeffs; ++j) {
                ASSERT_EQ(qcoeff_ref[j], qcoeff[j])
                    << "Q mismatch on test: " << i << " at position: " << j
                    << " Q: " << q << " coeff: " << coeff_ptr[j];
            }

            for (int j = 0; j < n_coeffs; ++j) {
                ASSERT_EQ(dqcoeff_ref[j], dqcoeff[j])
                    << "Dq mismatch on test: " << i << " at position: " << j
                    << " Q: " << q << " coeff: " << coeff_ptr[j];
            }

            ASSERT_EQ(eob[0], eob[1])
                << "eobs mismatch on test: " << i << " Q: " << q;
        }
    }

    // Microbenchmark: per-call latency of the C reference vs the SIMD kernel on
    // a dense (fully populated) coefficient block, qm_level_ = 8 (a non-flat
    // matrix). Disabled by default; run with --gtest_also_run_disabled_tests.
    void RunSpeedTest() {
        const auto quant_ref = TEST_GET_PARAM(0);
        const auto quant_tst = TEST_GET_PARAM(1);

        TranLow *coeff_ptr = coeff_.data();
        const intptr_t n_coeffs = coeff_num();
        TranLow *qcoeff_ref = coeff_ptr + n_coeffs;
        TranLow *dqcoeff_ref = qcoeff_ref + n_coeffs;
        TranLow *qcoeff = dqcoeff_ref + n_coeffs;
        TranLow *dqcoeff = qcoeff + n_coeffs;
        uint16_t *eob = reinterpret_cast<uint16_t *>(dqcoeff + n_coeffs);

        const ScanOrder *const sc = get_scan_order(tx_size_, DCT_DCT);
        const int q = 0;
        const int16_t *zbin = qtab_.quant.y_zbin[q];
        const int16_t *round = qtab_.quant.y_round_fp[q];
        const int16_t *quant = qtab_.quant.y_quant_fp[q];
        const int16_t *quant_shift = qtab_.quant.y_quant_shift[q];
        const int16_t *dequant = qtab_.dequant.y_dequant_qtx[q];
        const TxSize qm_tx_size = av1_get_adjusted_tx_size(tx_size_);
        const QmVal *qm_ptr = qmatrix_[qm_level_][0][qm_tx_size];
        const QmVal *iqm_ptr = iqmatrix_[qm_level_][0][qm_tx_size];
        const int log_scale = av1_get_tx_scale(tx_size_);

        // quant_ref / quant_tst are opaque function pointers, so the indirect
        // calls (and their stores) cannot be hoisted or eliminated.
        auto run =
            [&](QuantizeQmFunc fn, TranLow *qc, TranLow *dqc, uint16_t *e) {
                fn(coeff_ptr,
                   n_coeffs,
                   zbin,
                   round,
                   quant,
                   quant_shift,
                   qc,
                   dqc,
                   dequant,
                   e,
                   sc->scan,
                   sc->iscan,
                   qm_ptr,
                   iqm_ptr,
                   log_scale);
            };
        const uint64_t num_loop = 300000;
        // Measure across coefficient densities (full / quarter / eighth), since
        // sparser blocks shorten the eob-bounded loop and shrink the SIMD
        // margin.
        const int densities[] = {static_cast<int>(n_coeffs),
                                 static_cast<int>(n_coeffs) / 4,
                                 static_cast<int>(n_coeffs) / 8};
        for (int nz : densities) {
            if (nz < 1)
                nz = 1;
            FillCoeffRandomRows(nz);
            std::fill_n(qcoeff_ref, 5 * n_coeffs, 0);
            for (int i = 0; i < 4000; ++i) {  // warm
                run(quant_ref, qcoeff_ref, dqcoeff_ref, &eob[0]);
                run(quant_tst, qcoeff, dqcoeff, &eob[1]);
            }
            auto t0 = std::chrono::steady_clock::now();
            for (uint64_t i = 0; i < num_loop; ++i)
                run(quant_ref, qcoeff_ref, dqcoeff_ref, &eob[0]);
            auto t1 = std::chrono::steady_clock::now();
            for (uint64_t i = 0; i < num_loop; ++i)
                run(quant_tst, qcoeff, dqcoeff, &eob[1]);
            auto t2 = std::chrono::steady_clock::now();
            const double c_ns =
                std::chrono::duration<double, std::nano>(t1 - t0).count() /
                num_loop;
            const double n_ns =
                std::chrono::duration<double, std::nano>(t2 - t1).count() /
                num_loop;
            printf(
                "[ SPEED    ] tx=%2d bd=%2d n=%4d nz=%4d : C %8.1f ns  "
                "SIMD %8.1f ns  speedup %.2fx\n",
                static_cast<int>(tx_size_),
                static_cast<int>(bd_),
                static_cast<int>(n_coeffs),
                nz,
                c_ns,
                n_ns,
                c_ns / n_ns);
        }
    }

  private:
    static TxSize av1_get_adjusted_tx_size(TxSize tx_size) {
        switch (tx_size) {
        case TX_64X64:
        case TX_64X32:
        case TX_32X64: return TX_32X32;
        case TX_64X16: return TX_32X16;
        case TX_16X64: return TX_16X32;
        default: return tx_size;
        }
    }

    const QmVal *iqmatrix_[NUM_QM_LEVELS][3][TX_SIZES_ALL]{};
    const QmVal *qmatrix_[NUM_QM_LEVELS][3][TX_SIZES_ALL]{};
    const int qm_level_{8};
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantizeQmTest);

TEST_P(QuantizeQmTest, ZeroInput) {
    FillCoeffZero();
    QuantizeRun<false>();
}

TEST_P(QuantizeQmTest, LargeNegativeInput) {
    FillDcLargeNegative();
    QuantizeRun<false>(0, 1);
}

TEST_P(QuantizeQmTest, DcOnlyInput) {
    FillDcOnly();
    QuantizeRun<false>(0, 1);
}

TEST_P(QuantizeQmTest, RandomInput) {
    QuantizeRun<true>(0, kTestNum);
}

TEST_P(QuantizeQmTest, MultipleQ) {
    for (int q = 0; q < QINDEX_RANGE; ++q) {
        QuantizeRun<true>(q, kTestNum);
    }
}

// Force the coeff to be half the value of the dequant.  This exposes a
// mismatch found in av1_quantize_fp_sse2().
TEST_P(QuantizeQmTest, CoeffHalfDequant) {
    FillCoeff(16);
    QuantizeRun<false>(25, 1);
}

TEST_P(QuantizeQmTest, DISABLED_Speed) {
    RunSpeedTest();
}

using QuantizeQmHbdTest = QuantizeQmTest;
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantizeQmHbdTest);

TEST_P(QuantizeQmHbdTest, ZeroInput) {
    FillCoeffZero();
    QuantizeRun<false>();
}

TEST_P(QuantizeQmHbdTest, LargeNegativeInput) {
    FillDcLargeNegative();
    QuantizeRun<false>(0, 1);
}

TEST_P(QuantizeQmHbdTest, DcOnlyInput) {
    FillDcOnly();
    QuantizeRun<false>(0, 1);
}

TEST_P(QuantizeQmHbdTest, RandomInput) {
    QuantizeRun<true>(0, kTestNum);
}

TEST_P(QuantizeQmHbdTest, MultipleQ) {
    for (int q = 0; q < QINDEX_RANGE; ++q) {
        QuantizeRun<true>(q, kTestNum);
    }
}

// Force the coeff to be half the value of the dequant.  This exposes a
// mismatch found in av1_quantize_fp_sse2().
TEST_P(QuantizeQmHbdTest, CoeffHalfDequant) {
    FillCoeff(16);
    QuantizeRun<false>(25, 1);
}

TEST_P(QuantizeQmHbdTest, DISABLED_Speed) {
    RunSpeedTest();
}

#ifdef ARCH_X86_64
const QuantizeParam kQParamArrayAvx2[] = {
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_sse4_1, TX_16X16,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_sse4_1, TX_4X16,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_sse4_1, TX_16X4,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_sse4_1, TX_32X8,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_sse4_1, TX_8X32,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_avx2, TX_16X16,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_avx2, TX_4X16,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_avx2, TX_16X4,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_avx2, TX_32X8,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_avx2, TX_8X32,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_32x32_c, &svt_av1_quantize_fp_32x32_avx2,
               TX_32X32, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_32x32_c, &svt_av1_quantize_fp_32x32_avx2,
               TX_16X64, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_32x32_c, &svt_av1_quantize_fp_32x32_avx2,
               TX_64X16, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_32x32_c, &svt_av1_quantize_fp_32x32_sse4_1,
               TX_32X32, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_32x32_c, &svt_av1_quantize_fp_32x32_sse4_1,
               TX_16X64, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_32x32_c, &svt_av1_quantize_fp_32x32_sse4_1,
               TX_64X16, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_64x64_c, &svt_av1_quantize_fp_64x64_avx2,
               TX_64X64, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_64x64_c, &svt_av1_quantize_fp_64x64_sse4_1,
               TX_64X64, TYPE_FP, EB_EIGHT_BIT)};

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
const QuantizeHbdParam kQHbdParamArraySse41[] = {
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_16X16, TYPE_FP,
               EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_4X16, TYPE_FP,
               EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_16X4, TYPE_FP,
               EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_32X8, TYPE_FP,
               EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_8X32, TYPE_FP,
               EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_64X64, TYPE_FP,
               EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_16X16, TYPE_FP,
               EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_4X16, TYPE_FP,
               EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_16X4, TYPE_FP,
               EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_32X8, TYPE_FP,
               EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_8X32, TYPE_FP,
               EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_64X64, TYPE_FP,
               EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_16X16, TYPE_FP,
               EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_4X16, TYPE_FP,
               EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_16X4, TYPE_FP,
               EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_32X8, TYPE_FP,
               EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_8X32, TYPE_FP,
               EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c,
               &svt_av1_highbd_quantize_fp_sse4_1, TX_64X64, TYPE_FP,
               EB_TWELVE_BIT)};

const QuantizeHbdParam kQHbdParamArrayAvx2[] = {
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_16X16, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_4X16, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_16X4, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_32X8, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_8X32, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_64X64, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_16X16, TYPE_FP, EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_4X16, TYPE_FP, EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_16X4, TYPE_FP, EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_32X8, TYPE_FP, EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_8X32, TYPE_FP, EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_64X64, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_16X16, TYPE_FP, EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_4X16, TYPE_FP, EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_16X4, TYPE_FP, EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_32X8, TYPE_FP, EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_8X32, TYPE_FP, EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_avx2,
               TX_64X64, TYPE_FP, EB_TWELVE_BIT)};
#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH

const QuantizeQmParam kQmParamArrayAvx2[] = {
    make_tuple(&svt_av1_quantize_fp_qm_c, &svt_av1_quantize_fp_qm_avx2,
               TX_16X16, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_qm_c, &svt_av1_quantize_fp_qm_avx2, TX_4X16,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_qm_c, &svt_av1_quantize_fp_qm_avx2, TX_16X4,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_qm_c, &svt_av1_quantize_fp_qm_avx2, TX_32X8,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_qm_c, &svt_av1_quantize_fp_qm_avx2, TX_8X32,
               TYPE_FP, EB_EIGHT_BIT)};

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
const QuantizeQmParam kQmParamHbdArrayAvx2[] = {
    make_tuple(&svt_av1_highbd_quantize_fp_qm_c,
               &svt_av1_highbd_quantize_fp_qm_avx2, TX_16X16, TYPE_FP,
               EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_qm_c,
               &svt_av1_highbd_quantize_fp_qm_avx2, TX_4X16, TYPE_FP,
               EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_qm_c,
               &svt_av1_highbd_quantize_fp_qm_avx2, TX_16X4, TYPE_FP,
               EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_qm_c,
               &svt_av1_highbd_quantize_fp_qm_avx2, TX_32X8, TYPE_FP,
               EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_qm_c,
               &svt_av1_highbd_quantize_fp_qm_avx2, TX_8X32, TYPE_FP,
               EB_TEN_BIT)};
#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH

INSTANTIATE_TEST_SUITE_P(AVX2, QuantizeLbdTest,
                         ::testing::ValuesIn(kQParamArrayAvx2));
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
INSTANTIATE_TEST_SUITE_P(SSE4_1, QuantizeHbdTest,
                         ::testing::ValuesIn(kQHbdParamArraySse41));
INSTANTIATE_TEST_SUITE_P(AVX2, QuantizeHbdTest,
                         ::testing::ValuesIn(kQHbdParamArrayAvx2));
#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH
INSTANTIATE_TEST_SUITE_P(AVX2, QuantizeQmTest,
                         ::testing::ValuesIn(kQmParamArrayAvx2));
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
INSTANTIATE_TEST_SUITE_P(AVX2, QuantizeQmHbdTest,
                         ::testing::ValuesIn(kQmParamHbdArrayAvx2));
#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
const QuantizeParam kQParamArrayNeon[] = {
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_neon, TX_16X16,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_neon, TX_4X16,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_neon, TX_16X4,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_neon, TX_32X8,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_c, &svt_av1_quantize_fp_neon, TX_8X32,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_32x32_c, &svt_av1_quantize_fp_32x32_neon,
               TX_32X32, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_32x32_c, &svt_av1_quantize_fp_32x32_neon,
               TX_16X64, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_32x32_c, &svt_av1_quantize_fp_32x32_neon,
               TX_64X16, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_64x64_c, &svt_av1_quantize_fp_64x64_neon,
               TX_64X64, TYPE_FP, EB_EIGHT_BIT)};

INSTANTIATE_TEST_SUITE_P(NEON, QuantizeLbdTest,
                         ::testing::ValuesIn(kQParamArrayNeon));

const QuantizeQmParam kQmParamArrayNeon[] = {
    make_tuple(&svt_av1_quantize_fp_qm_c, &svt_av1_quantize_fp_qm_neon,
               TX_16X16, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_qm_c, &svt_av1_quantize_fp_qm_neon, TX_4X16,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_qm_c, &svt_av1_quantize_fp_qm_neon, TX_16X4,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_qm_c, &svt_av1_quantize_fp_qm_neon, TX_32X8,
               TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_quantize_fp_qm_c, &svt_av1_quantize_fp_qm_neon, TX_8X32,
               TYPE_FP, EB_EIGHT_BIT)};

INSTANTIATE_TEST_SUITE_P(NEON, QuantizeQmTest,
                         ::testing::ValuesIn(kQmParamArrayNeon));

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
const QuantizeQmParam kQmParamHbdArrayNeon[] = {
    make_tuple(&svt_av1_highbd_quantize_fp_qm_c,
               &svt_av1_highbd_quantize_fp_qm_neon, TX_16X16, TYPE_FP,
               EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_qm_c,
               &svt_av1_highbd_quantize_fp_qm_neon, TX_32X8, TYPE_FP,
               EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_qm_c,
               &svt_av1_highbd_quantize_fp_qm_neon, TX_8X32, TYPE_FP,
               EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_qm_c,
               &svt_av1_highbd_quantize_fp_qm_neon, TX_16X16, TYPE_FP,
               EB_TWELVE_BIT)};

INSTANTIATE_TEST_SUITE_P(NEON, QuantizeQmHbdTest,
                         ::testing::ValuesIn(kQmParamHbdArrayNeon));

const QuantizeHbdParam kQHbdParamArrayNeon[] = {
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_16X16, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_4X16, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_16X4, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_32X8, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_8X32, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_64X64, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_16X16, TYPE_FP, EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_4X16, TYPE_FP, EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_16X4, TYPE_FP, EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_32X8, TYPE_FP, EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_8X32, TYPE_FP, EB_TEN_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_64X64, TYPE_FP, EB_EIGHT_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_16X16, TYPE_FP, EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_4X16, TYPE_FP, EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_16X4, TYPE_FP, EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_32X8, TYPE_FP, EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_8X32, TYPE_FP, EB_TWELVE_BIT),
    make_tuple(&svt_av1_highbd_quantize_fp_c, &svt_av1_highbd_quantize_fp_neon,
               TX_64X64, TYPE_FP, EB_TWELVE_BIT)};

INSTANTIATE_TEST_SUITE_P(NEON, QuantizeHbdTest,
                         ::testing::ValuesIn(kQHbdParamArrayNeon));
#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH
#endif  // ARCH_AARCH64

using ComputeCulLevelTest =
    ::testing::TestWithParam<decltype(&svt_av1_compute_cul_level_c)>;

TEST_P(ComputeCulLevelTest, test_match) {
    const auto test_func_{GetParam()};
    SVTRandom rnd{0, (1 << 10) - 1};
    SVTRandom quant_rnd{-10, 10};
    constexpr int max_size = 50;
    // scan[] is a set of indexes for quant_coeff[]
    std::array<int16_t, max_size> scan{};
    std::array<int32_t, max_size> quant_coeff{};

    for (int test = 0; test < 1000; test++) {
        uint16_t eob_ref = rnd.random() % max_size, eob_test = eob_ref;

        if (eob_ref == 0) {
            quant_coeff.fill(0);
        } else {
            std::generate(quant_coeff.begin(), quant_coeff.end(), [&]() {
                return quant_rnd.random();
            });
        }

        std::generate(scan.begin() + 1, scan.end(), [&]() {
            return rnd.random() % max_size;
        });

        int32_t ref_res = svt_av1_compute_cul_level_c(
            scan.data(), quant_coeff.data(), &eob_ref);

        int32_t test_res =
            test_func_(scan.data(), quant_coeff.data(), &eob_test);

        EXPECT_EQ(ref_res, test_res);
        EXPECT_EQ(eob_ref, eob_test);
    }
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(AVX2, ComputeCulLevelTest,
                         ::testing::Values(svt_av1_compute_cul_level_avx2));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(NEON, ComputeCulLevelTest,
                         ::testing::Values(svt_av1_compute_cul_level_neon));

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(SVE, ComputeCulLevelTest,
                         ::testing::Values(svt_av1_compute_cul_level_sve));
#endif  // HAVE_SVE
#endif  // ARCH_AARCH64
}  // namespace
