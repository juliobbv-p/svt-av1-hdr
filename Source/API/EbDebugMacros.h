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

#define FTR_FRAME_RATE_ON_THE_FLY   1 // Add ability to change frame rate on the fly (without inserting keyframe)
#define OPT_RATE_ON_THE_FLY_NO_KF   1 // Add ability to change bitrate on the fly without inserting keyframe
#define FTR_PER_FRAME_QUALITY       1 // Add ability to compute quality for specific frame
#define FTR_SFRAME_FLEX             1 // Add S-Frame Flexible ARF Mode
#if FTR_SFRAME_FLEX
#define FTR_SFRAME_POSI             1 // Add parameter to allow user insert S-Frames by picture number
#if FTR_SFRAME_POSI
#define FTR_SFRAME_QP               1 // Add parameter to allow user set QP of S-Frame
#define FTR_SFRAME_DEC_POSI         1 // New S-Frame mode to insert S-Frames at specific position in decode order
#endif // FTR_SFRAME_POSI
#endif // FTR_SFRAME_FLEX
#define FIX_TUNE_SSIM               1 // Fix SSIM mode
#define RFCTR_PARSE_LIST            1 // Refactor parameter parsing list and support the range of int8 and uint8
#define FIX_QUEUE_DEADLOCK          1 // Use min-heap instead of queue to manage out-of-order decode orders
#define FIX_INTRA_BLUR_QP62         1 // Intra lambda-weight tuning for INTRA frames at high QPs (>=62) to reduce blurriness
#define FIX_FPS_CALC                1 // Fix frame-rate derivation to handle < 1 fps

#define CLN_MDC_FUNCS               1 // Clean-up unused MDC functions and variables
#define CLN_RECON_FUNC              1 // Clean-up MD recon function
#define CLN_REMOVE_OIS_FLAG         1 // Clean-up OIS functions and variables
#define FIX_EOB_COEF_CTX            1 // Fix the number of contexts used for signaling the EOB
#define FIX_IND_UV_SEARCH_TX        1 // Set MD stage to 0 for independent chroma search to prevent using uninit'd data to skip TX
#define FIX_CHROMA_SKIP             1 // Properly update chroma settings when chroma is skipped
#define CLN_MOVE_CHROMA_CHECK       1 // Move chroma complexity check inside MD stage check for PD1
#define CLN_MDS0_DIST_PD1           1 // Fix fast-cost derivation
#define FTR_STILL_IMAGE_UP_TO_M12   1 // Still-image up to M12
#define FTR_DEPTH_REMOVAL_INTRA     1 // Add support for variance-based depth removal for INTRA frames to speed up still-image processing
#define TUNE_STILL_IMAGE_0          1 // Perform a first round of tuning for still-image
#define OPT_MD_SIGNALS              1 // Clean up INTRA-level derivation, fix PD0/PD1 interactions, and adaptively set the intra-edge-filter flag
#define FIX_TUNE_SSIM_LAMBDA        1 // Fix SSIM mode lambda: fix per-SB lambda for balanced inter-depth decisions, and use default factors when deriving the lambda scaling factors
#define TUNE_STILL_IMAGE_1          1 // Perform a second round of tuning for still-image
#define OPT_DEFAULT_LAMBDA_MULT     1 // Use default lambda at RDOQ
#define FTR_USE_HADAMARD_MDS0       1 // Use Hadamard at mds0
#define TUNE_STILL_IMAGE_2          1 // Perform a third round of tuning for still-image
#define OPT_SSIM_METRIC             1 // Remove SSIM distortion calculations for INTRA frames
#define OPT_RECON_OPERATIONS        1 // Remove unnecessary reconstruction operations
#define OPT_FD2_FD1_STILL_IMAGE     1 // Add fast-decode for still-image

//FOR DEBUGGING - Do not remove
#define LOG_ENC_DONE            0 // log encoder job one
#define DEBUG_TPL               0 // Prints to debug TPL
#define DETAILED_FRAME_OUTPUT   0 // Prints detailed frame output from the library for debugging
#define DEBUG_BUFFERS           0 // Print process count and segments info
#define TUNE_CHROMA_SSIM        1 // Allows for Chroma and SSIM BDR-based Tuning
#define TUNE_CQP_CHROMA_SSIM    1 // Tune CQP qp scaling towards improved chroma and SSIM BDR

#define MIN_PIC_PARALLELIZATION 0 // Use the minimum amount of picture parallelization
#define SRM_REPORT              0 // Report SRM status
#define LAD_MG_PRINT            0 // Report LAD
#define RC_NO_R2R               0 // This is a debugging flag for RC and makes encoder to run with no R2R in RC mode
                                  // Note that the speed might impacted significantly
#if !RC_NO_R2R
#define FTR_KF_ON_FLY_SAMPLE         0 // Sample code to signal KF
#define FTR_RES_ON_FLY_SAMPLE        0 // Sample functions to change the resolution on the fly
#define FTR_RATE_ON_FLY_SAMPLE       0 // Sample functions to change bit rate
#if FTR_FRAME_RATE_ON_THE_FLY
#define FTR_FRAME_RATE_ON_FLY_SAMPLE 0 // Sample functions to change frame rate
#endif
#if FTR_PER_FRAME_QUALITY
#define FTR_PER_FRAME_QUALITY_SAMPLE 0 // Sample functions to compute PSNR per frame
#endif
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
