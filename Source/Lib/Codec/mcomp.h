/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef AOM_AV1_ENCODER_MCOMP_H_
#define AOM_AV1_ENCODER_MCOMP_H_

#include "mv.h"
#include "coding_unit.h"
#include "block_structures.h"
#include "av1_common.h"
#include "av1me.h"
#include "rd_cost.h"

#ifdef __cplusplus
extern "C" {
#endif
// =============================================================================
//  Cost functions
// =============================================================================

enum {
    MV_COST_ENTROPY, // Use the entropy rate of the mv as the cost
    MV_COST_OPT,
    MV_COST_NONE // Use 0 as as cost irrespective of the current mv
} UENUM1BYTE(MV_COST_TYPE);

typedef struct svt_mv_cost_param {
    // The reference mv used to compute the mv cost
    Mv           ref_mv;
    Mv           full_ref_mv;
    MV_COST_TYPE mv_cost_type;
    const int*   mvjcost;
    const int*   mvcost[2];
    int          error_per_bit;
    int          early_exit_th;
    // A multiplier used to convert rate to sad cost
    int sad_per_bit;
} svt_mv_cost_param;

// =============================================================================
//  Motion Search
// =============================================================================
typedef struct svt_buf_2d {
    uint8_t* buf;
    int      width;
    int      height;
    int      stride;
} svt_buf_2d;

typedef struct {
    // The reference buffer
    svt_buf_2d* ref;

    // The source and predictors/mask used by translational search
    svt_buf_2d* src;
} MSBuffers;

// =============================================================================
//  Subpixel Motion Search
// =============================================================================
typedef struct {
    const AomVarianceFnPtr* vfp;

    SUBPEL_SEARCH_TYPE subpel_search_type;

    // Source and reference buffers
    MSBuffers ms_buffers;

    int w, h;
    int32_t
        bias_fp; // Bias towards fpel at the MD subpel-search: apply a penalty to the cost of fractional positions during the subpel-search each time we check against a full-pel MV
} SUBPEL_SEARCH_VAR_PARAMS;

// This struct holds subpixel motion search parameters that should be constant
// during the search
typedef struct {
    // High level motion search settings
    int               allow_hp;
    SUBPEL_FORCE_STOP forced_stop;
    int               iters_per_step;
    int               pred_variance_th;
    uint8_t           abs_th_mult;
    int               round_dev_th;
    uint8_t           skip_diag_refinement;
    SUBPEL_STAGE      search_stage; //0: ME  1: PME
    uint8_t           list_idx;
    uint8_t           ref_idx;
    SubpelMvLimits    mv_limits;
    // For calculating mv cost
    svt_mv_cost_param mv_cost_params;

    // Distortion calculation params
    SUBPEL_SEARCH_VAR_PARAMS var_params;
} SUBPEL_MOTION_SEARCH_PARAMS;

typedef int(fractional_mv_step_fp)(void* ictx, MacroBlockD* xd, const struct AV1Common* const cm,
                                   SUBPEL_MOTION_SEARCH_PARAMS* ms_params, Mv start_mv, Mv* bestmv, BlockSize bsize);
extern fractional_mv_step_fp svt_av1_find_best_sub_pixel_tree;
extern fractional_mv_step_fp svt_av1_find_best_sub_pixel_tree_pruned;

static INLINE void svt_av1_set_subpel_mv_search_range(SubpelMvLimits* subpel_limits, const FullMvLimits* mv_limits,
                                                      const Mv ref_mv) {
    const int max_mv = GET_MV_SUBPEL(MAX_FULL_PEL_VAL);
    const int minc   = AOMMAX(GET_MV_SUBPEL(mv_limits->col_min), ref_mv.x - max_mv);
    const int maxc   = AOMMIN(GET_MV_SUBPEL(mv_limits->col_max), ref_mv.x + max_mv);
    const int minr   = AOMMAX(GET_MV_SUBPEL(mv_limits->row_min), ref_mv.y - max_mv);
    const int maxr   = AOMMIN(GET_MV_SUBPEL(mv_limits->row_max), ref_mv.y + max_mv);

    subpel_limits->col_min = AOMMAX(MV_LOW + 1, minc);
    subpel_limits->col_max = AOMMIN(MV_UPP - 1, maxc);
    subpel_limits->row_min = AOMMAX(MV_LOW + 1, minr);
    subpel_limits->row_max = AOMMIN(MV_UPP - 1, maxr);
}

static INLINE int svt_av1_is_subpelmv_in_range(const SubpelMvLimits* mv_limits, Mv mv) {
    return (mv.x >= mv_limits->col_min) && (mv.x <= mv_limits->col_max) && (mv.y >= mv_limits->row_min) &&
        (mv.y <= mv_limits->row_max);
}

// Returns the rate of encoding the current motion vector based on the
// joint_cost and comp_cost. joint_costs covers the cost of transmitting
// JOINT_MV, and comp_cost covers the cost of transmitting the actual motion
// vector.
static INLINE int svt_mv_cost(const Mv mv, const int* joint_cost, const int* const comp_cost[2]) {
    // The y-component (row component) of the MV is coded first, so the cost is in the 0th idx
    return joint_cost[svt_av1_get_mv_joint(mv)] + comp_cost[0][CLIP3(MV_LOW, MV_UPP, mv.y)] +
        comp_cost[1][CLIP3(MV_LOW, MV_UPP, mv.x)];
}

#define PIXEL_TRANSFORM_ERROR_SCALE 4

// Returns the cost of using the current mv during the motion search. This is
// used when var is used as the error metric.
static INLINE int svt_mv_err_cost(Mv mv, Mv ref_mv, const int* mvjcost, const int* const mvcost[2], int error_per_bit,
                                  MV_COST_TYPE mv_cost_type) {
    const Mv diff     = {{(int16_t)(mv.x - ref_mv.x), (int16_t)(mv.y - ref_mv.y)}};
    const Mv abs_diff = {{(int16_t)abs(diff.x), (int16_t)abs(diff.y)}};

    switch (mv_cost_type) {
    case MV_COST_ENTROPY:
        assert(mvcost);
        return (int)ROUND_POWER_OF_TWO_64(
            (int64_t)svt_mv_cost(diff, mvjcost, mvcost) * error_per_bit,
            RDDIV_BITS + AV1_PROB_COST_SHIFT - RD_EPB_SHIFT + PIXEL_TRANSFORM_ERROR_SCALE);
    case MV_COST_OPT: {
        return (int)ROUND_POWER_OF_TWO_64(
            (int64_t)((abs_diff.y + abs_diff.x) << 8) * error_per_bit,
            RDDIV_BITS + AV1_PROB_COST_SHIFT - RD_EPB_SHIFT + PIXEL_TRANSFORM_ERROR_SCALE);
    }
    case MV_COST_NONE:
        return 0;
    default:
        assert(0 && "Invalid rd_cost_type");
        return 0;
    }
}

static INLINE int svt_aom_fp_mv_err_cost(Mv mv, const svt_mv_cost_param* mv_cost_params) {
    return svt_mv_err_cost(mv,
                           mv_cost_params->ref_mv,
                           mv_cost_params->mvjcost,
                           mv_cost_params->mvcost,
                           mv_cost_params->error_per_bit,
                           mv_cost_params->mv_cost_type);
}

#ifdef __cplusplus
} // extern "C"
#endif
#endif // AOM_AV1_ENCODER_MCOMP_H_
