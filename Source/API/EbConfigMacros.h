/*
* Copyright(c) 2025 Meta Platforms, Inc. and affiliates.
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

/*
* This file contains configuration macros that control which parts of code are used
* Macros could be fed via command line, so all macros here must check if they are
* already defined!
* All macros must have the following format:
* - all macros must be prefixed with CONFIG_
*/

#ifndef EbConfigMacros_h
#define EbConfigMacros_h

// clang-format off

#ifndef RTC_BUILD
#define RTC_BUILD 0
#endif

#ifndef MINIMAL_BUILD
#define MINIMAL_BUILD 0
#endif

#if RTC_BUILD

#if MINIMAL_BUILD
#define MIN_ENC_PRESET                      ENC_M9

#ifndef CONFIG_LOG_QUIET
#define CONFIG_LOG_QUIET                    1
#endif

// Below features are disabled based on 2 assumptions:
// 1. 8-bit input/processing only
// 2. MIN_ENC_PRESET=ENC_M9

#ifndef CONFIG_ENABLE_QUANT_MATRIX
#define CONFIG_ENABLE_QUANT_MATRIX          0
#endif
#ifndef CONFIG_ENABLE_OBMC
#define CONFIG_ENABLE_OBMC                  0
#endif
#ifndef CONFIG_ENABLE_FILM_GRAIN
#define CONFIG_ENABLE_FILM_GRAIN            0
#endif
#ifndef CONFIG_ENABLE_HIGH_BIT_DEPTH
#define CONFIG_ENABLE_HIGH_BIT_DEPTH        0
#endif
#ifndef CONFIG_ENABLE_RESTORATION
#define CONFIG_ENABLE_RESTORATION           0
#endif
#ifndef CONFIG_ENABLE_GLOBAL_MOTION
#define CONFIG_ENABLE_GLOBAL_MOTION         0
#endif
#ifndef CONFIG_ENABLE_INTER_COMPOUND
#define CONFIG_ENABLE_INTER_COMPOUND        0
#endif
#ifndef CONFIG_ENABLE_INTER_INTRA
#define CONFIG_ENABLE_INTER_INTRA           0
#endif
#ifndef CONFIG_ENABLE_RANDOM_ACCESS
#define CONFIG_ENABLE_RANDOM_ACCESS         0
#endif
#ifndef CONFIG_ENABLE_FULL_SUBPEL
#define CONFIG_ENABLE_FULL_SUBPEL           0
#endif
#ifndef CONFIG_ENABLE_FILTER_INTRA
#define CONFIG_ENABLE_FILTER_INTRA          0
#endif
#ifndef CONFIG_ENABLE_SUPERRES
#define CONFIG_ENABLE_SUPERRES              0
#endif
#ifndef CONFIG_ENABLE_PALETTE
#define CONFIG_ENABLE_PALETTE               0
#endif
#ifndef CONFIG_ENABLE_INTRA_BC
#define CONFIG_ENABLE_INTRA_BC              0
#endif
#ifndef CONFIG_ENABLE_LOSSLESS
#define CONFIG_ENABLE_LOSSLESS              0
#endif
#ifndef CONFIG_ENABLE_TEMPORAL_FILTERING
#define CONFIG_ENABLE_TEMPORAL_FILTERING    0
#endif
#ifndef CONFIG_ENABLE_WARP
#define CONFIG_ENABLE_WARP                  0
#endif
#ifndef CONFIG_ENABLE_TPL
#define CONFIG_ENABLE_TPL                   0
#endif
#ifndef CONFIG_ENABLE_VMAF
#define CONFIG_ENABLE_VMAF                  0
#endif
#ifndef CONFIG_ENABLE_RESIZE
#define CONFIG_ENABLE_RESIZE                0
#endif
#ifndef CONFIG_ENABLE_TX_PF_N2
#define CONFIG_ENABLE_TX_PF_N2              0
#endif
#ifndef CONFIG_ENABLE_NON_DCT_LARGE_TX
#define CONFIG_ENABLE_NON_DCT_LARGE_TX      0
#endif
#ifndef CONFIG_ENABLE_MD_CDF_UPDATE
#define CONFIG_ENABLE_MD_CDF_UPDATE         0
#endif
#ifndef CONFIG_ENABLE_DLF_SEARCH
#define CONFIG_ENABLE_DLF_SEARCH            0
#endif
#ifndef CONFIG_RTC_SB64
#define CONFIG_RTC_SB64                     1
#endif
#endif // MINIMAL_BUILD

#endif

#ifndef MIN_ENC_PRESET
#define MIN_ENC_PRESET                      ENC_MRS
#endif

// When set to 1, EB_CPU_FLAGS_NEON is unconditionally set for all ARCH_AARCH64
// builds, i.e. requiring Neon for library to work. This also allows linker to
// strip code for all C functions which are optimized with Neon SIMD and thus
// reduce final binary size.
// Neon is mandatory in Armv8.0-A (AArch64), which is our minimum Arm target,
// so it is guaranteed for deployment builds, however tests use C functions,
// and hence for unit-test builds (SVT_AV1_UNIT_TEST_BUILD, set by CMake when
// BUILD_TESTING is ON) this must stay at 0.
#if defined(ARCH_AARCH64) && !defined(SVT_AV1_UNIT_TEST_BUILD)
#define CONFIG_ARM_NEON_IS_GUARANTEED       1
#endif

#ifndef CONFIG_ARM_NEON_IS_GUARANTEED
#define CONFIG_ARM_NEON_IS_GUARANTEED       0
#endif

// Same for x86 builds and AVX2 as minimum required SIMD level.
// AVX2 was first released in 2013 on Haswell microarchitecture, all x86
// processors since support it.
// You can set it to 1 to reduce binary size if deployment platforms are
// guaranteed to be not older than Haswell.
#if 0
#define CONFIG_X86_AVX2_IS_GUARANTEED       0
#endif

#ifndef CONFIG_X86_AVX2_IS_GUARANTEED
#define CONFIG_X86_AVX2_IS_GUARANTEED       0
#endif

#ifndef CONFIG_LOG_QUIET
#define CONFIG_LOG_QUIET                    0
#endif

#ifndef CONFIG_ENABLE_QUANT_MATRIX
#define CONFIG_ENABLE_QUANT_MATRIX          1
#endif

#ifndef CONFIG_ENABLE_OBMC
#define CONFIG_ENABLE_OBMC                  1
#endif

#ifndef CONFIG_ENABLE_FILM_GRAIN
#define CONFIG_ENABLE_FILM_GRAIN            1
#endif

#ifndef CONFIG_ENABLE_HIGH_BIT_DEPTH
#define CONFIG_ENABLE_HIGH_BIT_DEPTH        1
#endif

#ifndef CONFIG_ENABLE_RESTORATION
#define CONFIG_ENABLE_RESTORATION           1
#endif

#ifndef CONFIG_ENABLE_GLOBAL_MOTION
#define CONFIG_ENABLE_GLOBAL_MOTION         1
#endif

#ifndef CONFIG_ENABLE_INTER_COMPOUND
#define CONFIG_ENABLE_INTER_COMPOUND        1
#endif

#ifndef CONFIG_ENABLE_INTER_INTRA
#define CONFIG_ENABLE_INTER_INTRA           1
#endif

#ifndef CONFIG_ENABLE_RANDOM_ACCESS
#define CONFIG_ENABLE_RANDOM_ACCESS         1
#endif

#ifndef CONFIG_ENABLE_FULL_SUBPEL
#define CONFIG_ENABLE_FULL_SUBPEL           1
#endif

#ifndef CONFIG_ENABLE_FILTER_INTRA
#define CONFIG_ENABLE_FILTER_INTRA          1
#endif

#ifndef CONFIG_ENABLE_SUPERRES
#define CONFIG_ENABLE_SUPERRES              1
#endif

#ifndef CONFIG_ENABLE_PALETTE
#define CONFIG_ENABLE_PALETTE               1
#endif
#ifndef CONFIG_ENABLE_INTRA_BC
#define CONFIG_ENABLE_INTRA_BC              1
#endif

#ifndef CONFIG_ENABLE_LOSSLESS
#define CONFIG_ENABLE_LOSSLESS             1
#endif

#ifndef CONFIG_ENABLE_TEMPORAL_FILTERING
#define CONFIG_ENABLE_TEMPORAL_FILTERING   1
#endif

#ifndef CONFIG_ENABLE_WARP
#define CONFIG_ENABLE_WARP                 1
#endif

#ifndef CONFIG_ENABLE_TPL
#define CONFIG_ENABLE_TPL                  1
#endif

#ifndef CONFIG_ENABLE_VMAF
#define CONFIG_ENABLE_VMAF                 1
#endif

#ifndef CONFIG_ENABLE_RESIZE
#define CONFIG_ENABLE_RESIZE               1
#endif

#ifndef CONFIG_ENABLE_TX_PF_N2
#define CONFIG_ENABLE_TX_PF_N2             1
#endif

#ifndef CONFIG_ENABLE_NON_DCT_LARGE_TX
#define CONFIG_ENABLE_NON_DCT_LARGE_TX     1
#endif

#ifndef CONFIG_ENABLE_MD_CDF_UPDATE
#define CONFIG_ENABLE_MD_CDF_UPDATE        1
#endif

#ifndef CONFIG_ENABLE_DLF_SEARCH
#define CONFIG_ENABLE_DLF_SEARCH          1
#endif

// RTC forces super_block_size = 64 (enc_handle.c), so 128-wide blocks never occur and the 128-wide
// variance / sub-pel variance / SAD kernels are never dispatched. When set, their RTCD registration
// is dropped and LTO strips them. Bit-exact at SB=64.
#ifndef CONFIG_RTC_SB64
#define CONFIG_RTC_SB64                   0
#endif

// Fast (non-bit-exact) all-int16 forward transforms for the LBD path: every
// two-product cospi butterfly uses per-product vqrdmulhq_s16 instead of the
// widening multiply. ~3x fewer instructions; tiny rounding error vs the
// bit-exact int16 path. Guarded so a build picks exactly one path.
#ifndef CONFIG_ENABLE_FAST_LBD_TXFM
#define CONFIG_ENABLE_FAST_LBD_TXFM         0
#endif

// Single-thread kernel dispatch: at lp=1, bypass thread creation and run all
// pipeline kernels cooperatively on one thread. Eliminates 15 context switches
// per frame and all inter-stage semaphore/mutex overhead.
#ifndef CONFIG_SINGLE_THREAD_KERNEL
#define CONFIG_SINGLE_THREAD_KERNEL         1
#endif

// Native 8-bit CDEF path (interior blocks in uint8 lanes). NEON, and x86 from
// SSE4.1 up.
#if defined(ARCH_AARCH64) || defined(ARCH_X86_64)
#define CDEF_8BITS_PATH 1
#else
#define CDEF_8BITS_PATH 0
#endif

// When high-bit-depth (10/12-bit) support is compiled out, fold the effective encoder bit depth to
// the compile-time constant EB_EIGHT_BIT so that `bit_depth > EB_EIGHT_BIT` (and derived 16-bit)
// branches become dead and are eliminated by the optimizer -- no per-site #if guards, no empty ifs.
// SVT_EFFECTIVE_* read the effective value in an expression; SVT_FOLD_* pin a local to it in place
// (a no-op when HBD is enabled, so it never becomes a self-assignment).
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
#define SVT_EFFECTIVE_BIT_DEPTH(bit_depth) (bit_depth)
#define SVT_EFFECTIVE_IS_16BIT_PIPELINE(v) (v)
#define SVT_EFFECTIVE_HBD_MD(v)            (v)
#define SVT_FOLD_HBD_MD(hbd_md)            ((void)0)
#define SVT_FOLD_BIT_DEPTH(bit_depth)      ((void)0)
#else
// The dead ternary branch keeps the argument "used" (avoids -Wunused on locals that only feed this
// macro) while still folding to a compile-time constant so branches are eliminated by the optimizer.
#define SVT_EFFECTIVE_BIT_DEPTH(bit_depth) (0 ? (bit_depth) : EB_EIGHT_BIT)
#define SVT_EFFECTIVE_IS_16BIT_PIPELINE(v) (0 ? (v) : 0)
#define SVT_EFFECTIVE_HBD_MD(v)            (0 ? (v) : 0)
#define SVT_FOLD_HBD_MD(hbd_md)            ((hbd_md) = 0)
#define SVT_FOLD_BIT_DEPTH(bit_depth)      ((bit_depth) = EB_EIGHT_BIT)
#endif

// clang-format on

// RTC-minimal is always rtc-tuned and never all-intra; fold these to compile-time constants so the
// _default/_allintra signal-derivation variants dead-code-eliminate.
#define SVT_RTC_TUNE(scs) (RTC_BUILD ? true : (scs)->static_config.rtc)
#define SVT_ALLINTRA(scs) (RTC_BUILD ? false : (scs)->allintra)

#endif // EbConfigMacros_h
