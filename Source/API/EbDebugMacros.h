/*
* Copyright(c) 2020 Intel Corporation
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

/*
* This file contains only debug macros that are used during the development
* and are supposed to be cleaned up every tag cycle
* all macros must have the following format:
* - adding a new feature should be prefixed by FTR_
* - tuning a feature should be prefixed by TUNE_
* - enabling a feature should be prefixed by EN_
* - disabling a feature should be prefixed by DIS_
* - bug fixes should be prefixed by FIX_
* - code refactors should be prefixed by RFCTR_
* - code cleanups should be prefixed by CLN_
* - optimizations should be prefixed by OPT_
* - all macros must have a coherent comment explaining what the MACRO is doing
* - #if 0 / #if 1 are not to be used
*/

#ifndef EbDebugMacros_h
#define EbDebugMacros_h

// clang-format off

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define OPT_NSC_STILL_IMAGE         1 // optimize the NSC path for still-image
#define OPT_SC_STILL_IMAGE          1 // optimize the SC path (scm1/scm2) for still-image

#define FIX_CDEF                    1
#if FIX_CDEF
#define FIX_Q_STRENGTH              1 // Fix initialization issue for use_qp_strength
#define OPT_CDEF_SKIP_CHROMA_BORDER 1 // Skip all chroma border copies when all UV strengths are 0 for the entire frame
#define CLN_FINISH_CDEF             1 // Remove dead allocation/write/free for use_qp_strength and use_reference_cdef_fs paths, unify CDEF strength-per-mi propagation through propagate_cdef_strength()
#define OPT_SC_CDEF_QP              1 // SC model for CDEF from QP for sc-class1
#endif

#define FTR_COUPLE_VLPD0_TXS        1 // Use fastest TXS when VERY_LIGHT_PD0 is active and default TXS is off
#if FTR_COUPLE_VLPD0_TXS
#define FTR_COUPLE_VLPD0_TXS_PER_SB 1
#endif
#define OPT_SC_RA                   1 // optimize the SC path (scm1) for RA
#define OPT_SC_RTC                  1 // optimize the SC path (scm1) for RTC

#define OPT_VLPD0_COST             1 // Optimize VLPD0 inter-depth partitioning
#define OPT_VLPD0_COST_BIS         1 // Faster VLPD0 inter-depth partitioning

#define TUNE_CDEF_LEVEL            1 // Simplify RTC CDEF level derivation (remove sc_class1 branch, split flat/3L)
#define OPT_CDEF_PRI_ONLY          1 // Only test pri-only strengths {0, 15*4}. Halves search evaluations
                                     // and ensures the NEON kernel always takes its sec==0 fast path.
#define OPT_CDEF_SKIP_TH           1 // Replace use_skip_detector bool with configurable skip_th percentage

#define OPT_COEFF_SHAVING          1 // Post-quantization coefficient shaving: retract EOB and optionally zero block
#define OPT_COEFF_LEVEL            1 // Remove noise-level gating from derive_inter_coeff_level
#define OPT_PERIODIC_CDF_UPDATE    1 // Selective CDF disable for M12+ RTC
#define OPT_EC_INTERP              1 // Fixed EIGHTTAP_REGULAR when IFS is off
#define OPT_EC_DC_ONLY             1 // Fast entropy-coding path for eob==1 (DC-only blocks)
#define OPT_EC_MERGE_COEFF_LOOPS   1 // Merge backward/forward coefficient coding loops
#define OPT_GATE_SB_LAMBDA_MOD     1 // Gate stats_based_sb_lambda_modulation behind preset check

#define OPT_LPD1                   1 // Optimize LPD1: fixed-stage subpel, bias_fp, unify VLPD1, remove skip_zz_mv, fix rate-est
#define TUNE_LPD1_LEVEL            1 // Unified pic_lpd1_lvl derivation for RTC (remove sc_class1 special case)
#define OPT_STATS_MUTEX            1 // Use local accumulators for qindex stats instead of per-block mutex
#define OPT_SKIP_INTRA             1 // Skip INTRA using me-distortion


//FOR DEBUGGING - Do not remove
#define LOG_ENC_DONE            0 // log encoder job one
#define DEBUG_TPL               0 // Prints to debug TPL
#define DETAILED_FRAME_OUTPUT   0 // Prints detailed frame output from the library for debugging
#define DEBUG_BUFFERS           0 // Print process count and segments info
#define TUNE_CHROMA_SSIM        0 // Allows for Chroma and SSIM BDR-based Tuning
#define TUNE_CQP_CHROMA_SSIM    0 // Tune CQP qp scaling towards improved chroma and SSIM BDR

#define MIN_PIC_PARALLELIZATION 0 // Use the minimum amount of picture parallelization
#define SRM_REPORT              0 // Report SRM status
#define LAD_MG_PRINT            0 // Report LAD
#define RC_NO_R2R               0 // This is a debugging flag for RC and makes encoder to run with no R2R in RC mode
                                  // Note that the speed might impacted significantly
#if !RC_NO_R2R
#define FTR_KF_ON_FLY_SAMPLE         0 // Sample code to signal KF
#define FTR_RES_ON_FLY_SAMPLE        0 // Sample functions to change the resolution on the fly
#define FTR_RATE_ON_FLY_SAMPLE       0 // Sample functions to change bit rate
#define FTR_FRAME_RATE_ON_FLY_SAMPLE 0 // Sample functions to change frame rate
#define FTR_PER_FRAME_QUALITY_SAMPLE 0 // Sample functions to compute PSNR per frame
#endif
// Super-resolution debugging code
#define DEBUG_SCALING           0
#define DEBUG_TF                0
#define DEBUG_SUPERRES_RECODE   0
#define DEBUG_SUPERRES_ENERGY   0
#define DEBUG_RC_CAP_LOG        0 // Prints for RC cap

// Switch frame debugging code
#define DEBUG_SFRAME            0

// Variance Boost debugging code
#define DEBUG_VAR_BOOST         0
#define DEBUG_VAR_BOOST_QP      0
#define DEBUG_VAR_BOOST_STATS   0

// Anti-alias aware screen content mode debugging code
#define DEBUG_AA_SCM            0

// QP scaling debugging code
#define DEBUG_QP_SCALING        0

// Quantization matrices
#define DEBUG_QM_LEVEL          0
#define DEBUG_STARTUP_MG_SIZE   0
#define DEBUG_SEGMENT_QP        0
#define DEBUG_ROI               0
#ifdef __cplusplus
}
#endif // __cplusplus

// clang-format on

#endif // EbDebugMacros_h
