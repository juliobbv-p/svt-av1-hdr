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

#include "pcs.h"
#include "sequence_control_set.h"
#include "entropy_coding.h"

#include "rc_process.h"

#define USE_SCENE_CUT_WORKAROUND 0

// Binary search evaluation function type
typedef double (*arg_eval_fn)(void* ctx, int arg);

static uint8_t NOINLINE clamp_qindex(SequenceControlSet* scs, int qindex) {
    int qmin = quantizer_to_qindex[scs->static_config.min_qp_allowed];
    int qmax = quantizer_to_qindex[scs->static_config.max_qp_allowed];
    return (uint8_t)CLIP3(qmin, qmax, qindex);
}

// These are unscaled bits. To map into frame size see av1_estimate_frame_size()
static double av1_estimate_bits_at_qindex(PictureControlSet* pcs, int qindex, double rcf, uint64_t me_dist) {
    EbBitDepth bit_depth = pcs->scs->encoder_bit_depth;

    // , ppcs->sc_class1
    double  quantizer  = svt_av1_convert_qindex_to_q(qindex, bit_depth);
    double  ref_scale  = 1.0;
    int64_t complexity = 5600;

    if (pcs->ppcs->frm_hdr.frame_type != KEY_FRAME) {
        double ref_q = svt_av1_convert_qindex_to_q(pcs->ref_base_q_idx[REF_LIST_0][0], bit_depth);
        ref_scale    = sqrt(ref_q) / quantizer;
        complexity   = AOMMAX(me_dist, 64 * 64 / 4);
    }

    return rcf * complexity * ref_scale / quantizer;
}

typedef struct {
    PictureControlSet* pcs;
    double             rcf;
    uint64_t           me_dist;
} EvalBitsCtx;

static double eval_block_bits(void* ctx, int qindex) {
    EvalBitsCtx* c = (EvalBitsCtx*)ctx;
    return av1_estimate_bits_at_qindex(c->pcs, qindex, c->rcf, c->me_dist);
}

// Generic binary search: finds arg whose eval(ctx, arg) is closest to target.
// eval() must be monotonically decreasing with arg.
static int find_closest_arg(double target, int min_arg, int max_arg, arg_eval_fn eval, void* ctx) {
    int lo_arg = min_arg;
    int hi_arg = max_arg;
    while (lo_arg < hi_arg) {
        int    mid_arg = (lo_arg + hi_arg) >> 1;
        double mid_val = eval(ctx, mid_arg);
        if (mid_val > target) {
            lo_arg = mid_arg + 1;
        } else {
            hi_arg = mid_arg;
        }
    }
    assert(lo_arg == hi_arg);

    int curr_arg = lo_arg;
    if (curr_arg > min_arg) {
        double prev_val = eval(ctx, lo_arg - 1);
        double curr_val = eval(ctx, lo_arg);
        if (fabs(prev_val - target) < fabs(curr_val - target)) {
            curr_arg = lo_arg - 1;
        }
    }

    return curr_arg;
}

/******************************************************************************
* compute_cr_deltaq
* Find deltaq which changes the size according to rate_ratio
*******************************************************************************/
static int compute_cr_deltaq(PictureControlSet* pcs, int qindex, double rcf, uint64_t me_dist, double rate_ratio) {
    CyclicRefresh* cr = &pcs->ppcs->cyclic_refresh;
    RATE_CONTROL*  rc = &pcs->scs->enc_ctx->rc;

    // Get current estimated bits for given qindex, then target
    double base_bits   = av1_estimate_bits_at_qindex(pcs, qindex, rcf, me_dist);
    double target_bits = rate_ratio * base_bits;

    int min_qindex = (rate_ratio < 1.0) ? qindex : rc->best_quality;
    int max_qindex = (rate_ratio < 1.0) ? rc->worst_quality : qindex;

    EvalBitsCtx ctx = {pcs, rcf, me_dist};

    // Find closest qindex to generate this size
    int target_index = find_closest_arg(target_bits, min_qindex, max_qindex, eval_block_bits, &ctx);

    target_index = clamp_qindex(pcs->scs, target_index);
    return AOMMAX(target_index - qindex, -cr->max_qdelta_perc * qindex / 100);
}

static void cyclic_refresh_compute_cr_qdeltas(PictureControlSet* pcs, int qindex, double rcf) {
    CyclicRefresh* cr   = &pcs->ppcs->cyclic_refresh;
    cr->qindex_delta[0] = 0;
    cr->qindex_delta[1] = 0;
    cr->qindex_delta[2] = 0;
    if (cr->actual_num_seg1_sbs) {
        double qdelta       = AOMMIN(4.0, cr->rate_ratio_qdelta);
        cr->qindex_delta[1] = compute_cr_deltaq(pcs, qindex, rcf, cr->me_distortion[1], qdelta);
    }
    if (cr->actual_num_seg2_sbs) {
        double qdelta       = AOMMIN(8.0, cr->rate_ratio_qdelta_seg2);
        cr->qindex_delta[2] = compute_cr_deltaq(pcs, qindex, rcf, cr->me_distortion[2], qdelta);
    }
}

static int av1_estimate_frame_size(PictureControlSet* pcs, int qindex, double rcf, bool calc_sb_qindex) {
    CyclicRefresh* cr = &pcs->ppcs->cyclic_refresh;
    RATE_CONTROL*  rc = &pcs->scs->enc_ctx->rc;

    double estimated_size;
    if (cr->apply_cyclic_refresh) {
        if (calc_sb_qindex) {
            cyclic_refresh_compute_cr_qdeltas(pcs, qindex, rcf);
        }

        // Weight for non-base segments
        double w1 = (double)cr->actual_num_seg1_sbs / pcs->ppcs->b64_total_count;
        double w2 = (double)cr->actual_num_seg2_sbs / pcs->ppcs->b64_total_count;

        // Take segment weighted average for estimated bits.
        estimated_size = (1.0 - w1 - w2) * av1_estimate_bits_at_qindex(pcs, qindex, rcf, cr->me_distortion[0]) +
            w1 * av1_estimate_bits_at_qindex(pcs, qindex + cr->qindex_delta[1], rcf, cr->me_distortion[1]) +
            w2 * av1_estimate_bits_at_qindex(pcs, qindex + cr->qindex_delta[2], rcf, cr->me_distortion[2]);
    } else {
        estimated_size = av1_estimate_bits_at_qindex(pcs, qindex, rcf, rc->cur_avg_base_me_dist);
    }

    // scale to resolution
    FrameSize* frm_size = &pcs->ppcs->av1_cm->frm_size;
    return estimated_size * frm_size->frame_width * frm_size->frame_height / 512;
}

typedef struct {
    PictureControlSet* pcs;
    double             rcf;
} EvalFrameSizeCtx;

static double eval_frame_size(void* ctx, int qindex) {
    EvalFrameSizeCtx* c = (EvalFrameSizeCtx*)ctx;
    return av1_estimate_frame_size(c->pcs, qindex, c->rcf, true);
}

static int calc_pframe_target_size(PictureParentControlSet* ppcs) {
    SequenceControlSet* scs          = ppcs->scs;
    RATE_CONTROL*       rc           = &scs->enc_ctx->rc;
    RateControlCfg*     rc_cfg       = &scs->enc_ctx->rc_cfg;
    int                 diff         = rc->buffer_level - rc->optimal_buffer_level;
    double              one_pct_bits = 1.0 + rc->optimal_buffer_level / 100.0;
    int                 target       = rc->avg_frame_bandwidth;

    if (rc->ema_me_dist <= 0) {
        rc->ema_me_dist = rc->cur_avg_base_me_dist; // initialize on first frame
    }
    rc->ema_me_dist = AOMMAX(rc->ema_me_dist, 64 * 64 / 16);

    // complexity modulation: scale target based on deviation of
    // cur_avg_base_me_dist from its exponentially smoothed average
    // double complexity_weight = 0.0;
    // const char* env = getenv("SVT_COMPLEXITY_WEIGHT");
    // if (env) {
    //     complexity_weight = atof(env);
    // }
    // if (rc->cur_avg_base_me_dist > 0 && complexity_weight != 0) {
    //     double deviation = (rc->cur_avg_base_me_dist - rc->ema_me_dist) / rc->ema_me_dist;
    //     target = (int)(target * (1.0 + tanh(complexity_weight * deviation)));
    // }

    // employ RCFs weighting to adapt to different encoding complexities automatically
    // TODO: may tune these allocation with pow(rcf, parameter)
    int max_layers = ppcs->scs->static_config.hierarchical_levels + 1;
    if (ppcs->temporal_layer_index == 0 && max_layers > 1) {
        int    n = 0, m = 1; // to produce 1, 1, 2, 4, 8, ...
        double sum_factors = 0.0;
        for (int k = 0; k < max_layers; k++) {
            sum_factors += rc->rate_correction_factors[k + 1] * m;
            m += n;
            n = m;
        }
        double avg_factor = sum_factors / m;
        for (int k = 0; k < max_layers; k++) {
            rc->target_size_factors[k] = rc->rate_correction_factors[k + 1] / avg_factor;
        }
    }

    // temporal dependency and mode decision modulation
    target = (int)(target * rc->target_size_factors[ppcs->temporal_layer_index]);

#if USE_SCENE_CUT_WORKAROUND
    if (rc->cur_avg_base_me_dist > rc->ema_me_dist * 1.5) {
        target = (int)(target * rc->cur_avg_base_me_dist / rc->ema_me_dist);
    }
#endif

    // buffer adjustment
    if (diff > 0) {
        // Lower the target for this frame.
        double pct = AOMMIN(diff / one_pct_bits, rc_cfg->over_shoot_pct);
        target -= (int)(target * pct / 200);
    } else if (rc->buffer_level < rc->avg_frame_bandwidth) {
        // Increase the target for this frame, less aggresively
        double pct = AOMMIN(-diff / one_pct_bits, rc_cfg->under_shoot_pct);
        target += (int)(target * pct / 400);
    }

    int min_frame_target = AOMMAX(rc->avg_frame_bandwidth >> 4, FRAME_OVERHEAD_BITS);
    return AOMMAX(min_frame_target, target);
}

static void cyclic_refresh_init(PictureParentControlSet* ppcs) {
    SequenceControlSet* scs = ppcs->scs;
    RATE_CONTROL*       rc  = &scs->enc_ctx->rc;
    CyclicRefresh*      cr  = &ppcs->cyclic_refresh;

    cr->apply_cyclic_refresh = ppcs->slice_type != I_SLICE && (scs->use_flat_ipp || ppcs->temporal_layer_index == 0);

    if (scs->super_block_size != 64) {
        cr->apply_cyclic_refresh = 0;
    }

    // TODO: this must be adaptive!
    int cr_num_layers = 2;
    if (ppcs->temporal_layer_index >= cr_num_layers) {
        cr->apply_cyclic_refresh = 0;
    }

    int qp_min_thresh = AOMMAX(16, rc->best_quality + 4);
    int qp_max_thresh = 118 * MAXQ >> 7;

    if (rc->avg_frame_qindex[INTER_FRAME] > qp_max_thresh) {
        cr->apply_cyclic_refresh = 0;
    }

    if (rc->avg_frame_qindex[INTER_FRAME] < qp_min_thresh) {
        cr->apply_cyclic_refresh = 0;
    }

    if (rc->avg_frame_low_motion < 50) {
        cr->apply_cyclic_refresh = 0;
    }

    cr->percent_refresh = 20;

    if (cr->percent_refresh <= 0) {
        cr->apply_cyclic_refresh = 0;
    }

    if (!cr->apply_cyclic_refresh) {
        return;
    }

    uint16_t sb_cnt         = scs->sb_total_count;
    cr->sb_start            = scs->enc_ctx->cr_sb_end;
    cr->sb_end              = AOMMIN(cr->sb_start + sb_cnt * cr->percent_refresh / 100, sb_cnt);
    scs->enc_ctx->cr_sb_end = cr->sb_end >= sb_cnt ? 0 : cr->sb_end;

    // Quantizer-based multiplicative adjustment
    double avg_q = svt_av1_convert_qindex_to_q(rc->avg_frame_qindex[INTER_FRAME], scs->encoder_bit_depth);

    // these values depend in R-Q model in av1_estimate_bits_at_qindex
    // RTC RC uses quadratic model and hence values are not comparable to
    // regular CBR which uses linear model
    double rate_ratio_qdelta_base  = 2.7;
    double rate_ratio_qdelta_scale = 170.0;
    int    rate_boost_fac          = 18;

    cr->max_qdelta_perc   = 60;
    cr->rate_ratio_qdelta = rate_ratio_qdelta_base + avg_q / rate_ratio_qdelta_scale;
    cr->rate_boost_fac    = rate_boost_fac;
}

static double get_rate_correction_factor(PictureParentControlSet* ppcs, int width, int height) {
    RATE_CONTROL* rc       = &ppcs->scs->enc_ctx->rc;
    FrameSize*    frm_size = &ppcs->av1_cm->frm_size;

    svt_block_on_mutex(rc->rc_mutex);
    int    k   = ppcs->frm_hdr.frame_type == KEY_FRAME ? 0 : ppcs->temporal_layer_index + 1;
    double rcf = rc->rate_correction_factors[k];
    svt_release_mutex(rc->rc_mutex);

    rcf *= (double)(frm_size->frame_width * frm_size->frame_height) / (width * height);
    return rcf;
}

static void set_rate_correction_factor(PictureParentControlSet* ppcs, double rcf, int width, int height) {
    RATE_CONTROL* rc       = &ppcs->scs->enc_ctx->rc;
    FrameSize*    frm_size = &ppcs->av1_cm->frm_size;

    rcf = fclamp(rcf, MIN_BPB_FACTOR, MAX_BPB_FACTOR);

    // Normalize RCF to account for the size-dependent scaling factor.
    rcf /= (double)(frm_size->frame_width * frm_size->frame_height) / (width * height);

    svt_block_on_mutex(rc->rc_mutex);
    if (ppcs->frm_hdr.frame_type == KEY_FRAME) {
        rc->rate_correction_factors[0] = rcf;
    } else {
        if (rc->frames_since_key < 4) {
            // Reset all factors at start as initial values could be off
            int max_layers = ppcs->scs->static_config.hierarchical_levels + 1;
            for (int i = 1; i < max_layers + 1; ++i) {
                rc->rate_correction_factors[i] = AOMMAX(rcf, rc->rate_correction_factors[i]);
            }
        } else {
            rc->rate_correction_factors[ppcs->temporal_layer_index + 1] = rcf;
        }
    }
    svt_release_mutex(rc->rc_mutex);
}

#define DEFAULT_KF_BOOST_RT 2300
#define DEFAULT_GF_BOOST_RT 2000

static double calculate_qindex(PictureControlSet* pcs, SequenceControlSet* scs) {
    PictureParentControlSet* ppcs   = pcs->ppcs;
    RATE_CONTROL*            rc     = &scs->enc_ctx->rc;
    RateControlCfg*          rc_cfg = &scs->enc_ctx->rc_cfg;

    int min_qindex = rc->best_quality;
    int max_qindex = rc->worst_quality;
    int max_size   = rc->max_frame_bandwidth;

    if (frame_is_intra_only(ppcs)) {
        rc->kf_boost      = DEFAULT_KF_BOOST_RT;
        rc->frames_to_key = scs->static_config.intra_period_length + 1;

        // just to match existent CBR
        ppcs->base_frame_target = rc->starting_buffer_level / 2;

        if (rc_cfg->max_intra_bitrate_pct) {
            int maxi = rc->avg_frame_bandwidth * rc_cfg->max_intra_bitrate_pct / 100;
            max_size = AOMMIN(max_size, maxi);
        }
    } else {
        if (pcs->ref_slice_type[REF_LIST_0][0] == I_SLICE) {
            min_qindex = MAX(min_qindex, pcs->ref_base_q_idx[REF_LIST_0][0] - 1 * 4);
        } else {
            min_qindex = MAX(min_qindex, pcs->ref_base_q_idx[REF_LIST_0][0] - 4 * 4);
            max_qindex = MIN(max_qindex, pcs->ref_base_q_idx[REF_LIST_0][0] + 16 * 4);
        }
        ppcs->base_frame_target = calc_pframe_target_size(ppcs);

        if (rc_cfg->max_inter_bitrate_pct) {
            int maxp = rc->avg_frame_bandwidth * rc_cfg->max_inter_bitrate_pct / 100;
            max_size = AOMMIN(max_size, maxp);
        }
    }
    ppcs->base_frame_target = AOMMIN(ppcs->base_frame_target, max_size);
    ppcs->this_frame_target = ppcs->base_frame_target;

    cyclic_refresh_init(ppcs);
    svt_aom_cyclic_refresh_setup(ppcs);

    int    width  = ppcs->av1_cm->frm_size.frame_width;
    int    height = ppcs->av1_cm->frm_size.frame_height;
    double rcf    = get_rate_correction_factor(ppcs, width, height);

    EvalFrameSizeCtx ctx = {pcs, rcf};

    int qindex = find_closest_arg(ppcs->this_frame_target, min_qindex, max_qindex, eval_frame_size, &ctx);
    return clamp_qindex(scs, qindex);
}

void svt_av1_rc_calc_qindex_rtc_cbr(PictureControlSet* pcs) {
    PictureParentControlSet* ppcs = pcs->ppcs;
    SequenceControlSet*      scs  = ppcs->scs;

    if (pcs->picture_number == 0 || ppcs->seq_param_changed) {
        RATE_CONTROL*   rc     = &scs->enc_ctx->rc;
        RateControlCfg* rc_cfg = &scs->enc_ctx->rc_cfg;

        int32_t bandwidth = scs->static_config.target_bit_rate;
        int64_t starting  = rc_cfg->starting_buffer_level_ms;
        int64_t optimal   = rc_cfg->optimal_buffer_level_ms;
        int64_t maximum   = rc_cfg->maximum_buffer_size_ms;

        // API uses inverse leaky bucket definition with maximum size and
        // other RC modes model it such that buffer level could go negative,
        // which is rather confusing.
        // Convert parameters to classic leaky bucket where input rate is
        // rate of data generated by encoder and output rate is network rate.
        // a) keep starting same as original CBR for easier comparison
        rc->starting_buffer_level = starting * bandwidth / 1000;
        // b) invert optimal buffer level
        rc->optimal_buffer_level = (maximum - optimal) * bandwidth / 1000;
        // c) maximum is irrelevant now, as buffer is clipped as max(0, buf)
        rc->maximum_buffer_size = maximum * bandwidth / 1000;

        svt_av1_rc_init(scs);

        // d) invert current buffer level
        rc->buffer_level = (maximum - starting) * bandwidth / 1000;

        int max_layers = scs->static_config.hierarchical_levels + 1;
        for (int k = 0; k < max_layers; k++) {
            rc->target_size_factors[k] = 1;
        }
    }

    int qindex = calculate_qindex(pcs, scs);

    if (ppcs->cyclic_refresh.apply_cyclic_refresh) {
        // compute CR qdeltas with final qindex
        int width  = ppcs->av1_cm->frm_size.frame_width;
        int height = ppcs->av1_cm->frm_size.frame_height;

        double rcf = get_rate_correction_factor(ppcs, width, height);
        cyclic_refresh_compute_cr_qdeltas(pcs, qindex, rcf);
    }

    ppcs->frm_hdr.quantization_params.base_q_idx = qindex;
}

static void av1_rc_update_rate_correction_factors(PictureParentControlSet* ppcs) {
    int    width      = ppcs->av1_cm->frm_size.frame_width;
    int    height     = ppcs->av1_cm->frm_size.frame_height;
    int    base_q_idx = ppcs->frm_hdr.quantization_params.base_q_idx;
    double rcf        = get_rate_correction_factor(ppcs, width, height);

    // Do not update the rate factors for arf overlay frames.
    if (ppcs->is_overlay) {
        return;
    }

    RATE_CONTROL* rc = &ppcs->scs->enc_ctx->rc;
    if (ppcs->frm_hdr.frame_type != KEY_FRAME) {
        // Do not update the rate factors for repeated pictures
        if (rc->cur_avg_base_me_dist == 0) {
            return;
        }
    }

    // Work out how big we would have expected the frame to be at this Q given
    // the current correction factor.
    int estimated_size = av1_estimate_frame_size(ppcs->child_pcs, base_q_idx, rcf, false);

    // Work out a size correction factor.
    double correction_factor = 1.0 * ppcs->projected_frame_size / AOMMAX(estimated_size, 1);

    // Clamp correction factor to prevent anything too extreme
    correction_factor = AOMMAX(correction_factor, 0.25);
    correction_factor = AOMMIN(correction_factor, 4.0);

    // Decide how heavily to dampen the adjustment
    // Compute adjustment_limit from correction_factor error and content volatility.
    double rcf_base    = 0.2;
    double rcf_damping = 1.0;

    double log_error        = fabs(log10(correction_factor));
    double base_limit       = rcf_base * AOMMIN(1.0, log_error + 0.5);
    double me_deviation     = fabs(rc->cur_avg_base_me_dist - rc->ema_me_dist) / AOMMAX(rc->ema_me_dist, 1);
    double adjustment_limit = base_limit / (1.0 + rcf_damping * me_deviation);

    if (correction_factor > 1) {
        correction_factor = 1.0 + (correction_factor - 1.0) * adjustment_limit;
        rcf *= correction_factor;
    } else {
        correction_factor = 1.0 + (1.0 / correction_factor - 1.0) * adjustment_limit / 2;
        rcf /= correction_factor;
    }

    set_rate_correction_factor(ppcs, rcf, width, height);
}

// Update the buffer level: leaky bucket model.
// In contrast to other RC modes bucket here is infinite in size - no data is dropped at
// encoder itself. Higher level simply means delayed transmission over constant rate network.
// However bucket cannot go below empty (negative) - unused bits are simply lost.
static void update_buffer_level(PictureParentControlSet* ppcs, int encoded_frame_size) {
    RATE_CONTROL* rc = &ppcs->scs->enc_ctx->rc;

    // Non-viewable frames are a special case and are treated as pure overhead.
    rc->buffer_level += encoded_frame_size;
    if (ppcs->frm_hdr.showable_frame) {
        rc->buffer_level -= rc->avg_frame_bandwidth;
    }

    // Clip the buffer level.
    rc->buffer_level = AOMMAX(0, rc->buffer_level);
}

void svt_av1_rc_postencode_update_rtc_cbr(PictureParentControlSet* ppcs) {
    RATE_CONTROL* rc      = &ppcs->scs->enc_ctx->rc;
    FrameHeader*  frm_hdr = &ppcs->frm_hdr;

    // Update rate control heuristics
    ppcs->projected_frame_size = (int)ppcs->total_num_bits;

    // Post encode loop adjustment of Q prediction.
    av1_rc_update_rate_correction_factors(ppcs);

    update_buffer_level(ppcs, ppcs->projected_frame_size);

    double alpha = 0.5; // EMA smoothing factor
    rc->ema_me_dist += alpha * (rc->cur_avg_base_me_dist - rc->ema_me_dist);

    int qindex = frm_hdr->quantization_params.base_q_idx;

    if (frm_hdr->frame_type == KEY_FRAME) {
        rc->avg_frame_qindex[KEY_FRAME] = ROUND_POWER_OF_TWO(3 * rc->avg_frame_qindex[KEY_FRAME] + qindex, 2);

        rc->frames_since_key = 0;
    } else if (!ppcs->is_overlay) {
        rc->avg_frame_qindex[INTER_FRAME] = ROUND_POWER_OF_TWO(3 * rc->avg_frame_qindex[INTER_FRAME] + qindex, 2);

        int avg_cnt_zeromv = (int)ppcs->child_pcs->avg_cnt_zeromv;
        if (rc->avg_frame_low_motion == 0) {
            rc->avg_frame_low_motion = avg_cnt_zeromv;
        } else {
            rc->avg_frame_low_motion = ROUND_POWER_OF_TWO(3 * rc->avg_frame_low_motion + avg_cnt_zeromv, 2);
        }
    }
}
