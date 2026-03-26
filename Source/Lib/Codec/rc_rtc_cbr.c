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

// If test videos are concatenated then at concatenation point scene change is
// abrupt and using low qindex for first frame gives huge quality boost for next frames.
// Regular CBR mistakenly allows such low qindex due to inability to properly
// predict frame size and thus benefits from quality propagation.
// This flag is to make comparison for fair, only for testing purposes.
#define USE_SCENE_CUT_WORKAROUND 0

// Binary search evaluation function type
typedef double (*arg_eval_fn)(void* ctx, int arg);

static uint8_t NOINLINE clamp_qindex(SequenceControlSet* scs, int qindex) {
    int qmin = quantizer_to_qindex[scs->static_config.min_qp_allowed];
    int qmax = quantizer_to_qindex[scs->static_config.max_qp_allowed];
    return (uint8_t)CLIP3(qmin, qmax, qindex);
}

static EbReferenceObject* get_ref_obj(PictureControlSet* pcs, RefList ref_list, int idx) {
    return pcs->ref_pic_ptr_array[ref_list][idx]->object_ptr;
}

static int NOINLINE get_min_ref_base_q_idx(PictureControlSet* pcs) {
    assert(pcs->ppcs->ref_list0_count_try > 0);
    int q_idx = pcs->ref_base_q_idx[REF_LIST_0][0]; // LAST
    if (pcs->ppcs->ref_list1_count_try > 0) {
        q_idx = AOMMIN(q_idx, pcs->ref_base_q_idx[REF_LIST_1][0]); // BWD
    }
    return q_idx;
}

// These are unscaled bits. To map into frame size see av1_estimate_frame_size()
static double av1_estimate_bits_at_qindex(PictureControlSet* pcs, int qindex, double rcf, uint64_t me_dist) {
    EbBitDepth bit_depth = pcs->scs->encoder_bit_depth;

    double  quantizer  = svt_av1_convert_qindex_to_q(qindex, bit_depth);
    double  ref_scale  = 1.0;
    int64_t complexity = 5600;

    if (pcs->ppcs->frm_hdr.frame_type != KEY_FRAME) {
        RATE_CONTROL* rc = &pcs->scs->enc_ctx->rc;

        double ref_q = svt_av1_convert_qindex_to_q(rc->min_ref_base_q_idx, bit_depth);
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

static void normalize_factors(double* dst, double* src, int i_start, int i_end) {
    double sum = 0.0;
    for (int k = i_start; k < i_end; k++) {
        sum += src[k] * (1 << AOMMAX(k - i_start - 1, 0));
    }
    double avg_factor = sum / (1 << AOMMAX(i_end - i_start - 1, 0));
    for (int k = i_start; k < i_end; k++) {
        dst[k] = src[k] / avg_factor;
    }
}

static int index2tl(int index, int levels) {
    return index ? levels - get_msb(index ^ (index - 1)) : 0;
}

static int calc_pframe_target_size(PictureParentControlSet* ppcs) {
    SequenceControlSet* scs    = ppcs->scs;
    RATE_CONTROL*       rc     = &scs->enc_ctx->rc;
    RateControlCfg*     rc_cfg = &scs->enc_ctx->rc_cfg;

#if USE_SCENE_CUT_WORKAROUND
    double me_dist = rc->cur_avg_base_me_dist;
    if (rc->ema_me_dist <= 0) {
        rc->ema_me_dist = me_dist;
    }
#endif

    if (ppcs->temporal_layer_index == 0 && rc->mini_qop_size > 1) {
        double weights[1 + MAX_TEMPORAL_LAYERS] = {0};
        double rcf_tlx[1 + MAX_TEMPORAL_LAYERS] = {0};

        // prepare weighted RCFs - core components of layer weights
        int num_layers = scs->static_config.hierarchical_levels + 1;
        svt_block_on_mutex(rc->rc_mutex);
        for (int k = 1; k < rc->mini_qop_size + 1; k++) {
            int k_tl = index2tl(k - 1, num_layers - 1);
            rcf_tlx[k_tl + 1] += rc->rcf_values[k] / (1 << AOMMAX(k_tl - 1, 0));
        }
        svt_release_mutex(rc->rc_mutex);

        if (scs->use_flat_ipp) {
            for (int k = 1; k < num_layers + 1; k++) {
                weights[k] = rcf_tlx[k];
            }
        } else {
            normalize_factors(rcf_tlx, rcf_tlx, 1, num_layers + 1);

            // Adaptive rcf_last + avg_frame_qindex + avg_zeromv model:
            double rcf_last   = rcf_tlx[1 + num_layers - 1];
            double avg_qindex = rc->avg_frame_qindex[INTER_FRAME];
            double avg_zeromv = rc->avg_frame_low_motion;

            // Base layer weight is derived from encoding properties:
            double w0 = -9.32 * rcf_last + 0.023 * avg_qindex + 0.034 * avg_zeromv + 8.14;

            // Saturating intermediate weights:
            // W_k = 1 + amplitude * (1 - exp(-rate * (W0 - 1)))
            // Generalizes to N layers: k=1 is base, k=num_layers is top (weight=1.0)
            double amplitude = 0.81;
            double rate      = 0.83;

            // adjustment for bidiriectional prediction efficiency
            double scale = (scs->static_config.pred_structure == RANDOM_ACCESS) ? 0.5 : 1.0;

            weights[1] = w0;
            for (int k = 2; k < num_layers + 1; k++) {
                double amp = amplitude * (num_layers - k) / (num_layers - 1);
                double w_k = 1.0 + amp * (1.0 - exp(-rate * (AOMMAX(w0, 1.0) - 1.0)));
                weights[k] = w_k * scale;
            }
        }

        // Enforce w0 >= w1 >= ... >= wN (monotonically non-increasing)
        for (int k = num_layers; k >= 2; k--) {
            weights[k - 1] = AOMMAX(weights[k - 1], weights[k]);
        }

        normalize_factors(rc->target_size_factors, weights, 1, num_layers + 1);
    }

    double frame_target = rc->avg_frame_bandwidth;
    double buffer_diff  = rc->buffer_level - rc->optimal_buffer_level;
    double one_pct_bits = 1.0 + rc->optimal_buffer_level / 100.0;

    // temporal dependency and mode decision modulation
    frame_target *= rc->target_size_factors[ppcs->temporal_layer_index + 1];

#if USE_SCENE_CUT_WORKAROUND
    if (me_dist > rc->ema_me_dist * 1.5 && scs->use_flat_ipp) {
        frame_target *= me_dist / rc->ema_me_dist;
    }
#endif

    // buffer adjustment, estimate buffer level after this frame
    buffer_diff += frame_target - rc->avg_frame_bandwidth;
    if (buffer_diff > 0) {
        // Lower the target for this frame.
        double pct = AOMMIN(buffer_diff / one_pct_bits, rc_cfg->over_shoot_pct);
        frame_target *= 1.0 - pct / 200;
    } else if (rc->buffer_level < rc->avg_frame_bandwidth / 4) {
        // Increase the target for this frame, less aggresively
        double pct = AOMMIN(-buffer_diff / one_pct_bits, rc_cfg->under_shoot_pct);
        frame_target *= 1.0 + pct / 400;
    }

    double min_frame_target = AOMMAX(rc->avg_frame_bandwidth >> 4, FRAME_OVERHEAD_BITS);
    return AOMMAX(min_frame_target, frame_target);
}

static void cyclic_refresh_init(PictureParentControlSet* ppcs) {
    SequenceControlSet* scs = ppcs->scs;
    RATE_CONTROL*       rc  = &scs->enc_ctx->rc;
    CyclicRefresh*      cr  = &ppcs->cyclic_refresh;

    bool is_inter_base_layer = ppcs->slice_type != I_SLICE && (scs->use_flat_ipp || ppcs->temporal_layer_index == 0);
    // Technically it could be used in VBR too, but difference in goals for between CBR and VBR is unclear.
    // Right now VBR forces enormous buffer, which essentially makes it unbounded to set bitrate,
    // while CBR follows set buffer limitation and follows bitrate closely.
    cr->apply_cyclic_refresh = scs->enc_ctx->rc_cfg.mode == AOM_CBR && is_inter_base_layer;

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

static int get_rcf_index(PictureParentControlSet* ppcs) {
    return ppcs->frm_hdr.frame_type == KEY_FRAME ? 0 : ppcs->pred_struct_index + 1;
}

static double get_rate_correction_factor(PictureParentControlSet* ppcs, int width, int height) {
    RATE_CONTROL* rc = &ppcs->scs->enc_ctx->rc;
    svt_block_on_mutex(rc->rc_mutex);
    double rcf = rc->rcf_values[get_rcf_index(ppcs)];
    svt_release_mutex(rc->rc_mutex);

    // Normalize RCF to account for the size-dependent scaling factor.
    FrameSize* frm_size  = &ppcs->av1_cm->frm_size;
    double     res_scale = (double)(frm_size->frame_width * frm_size->frame_height) / (width * height);

    return rcf * res_scale;
}

static void set_rate_correction_factor(PictureParentControlSet* ppcs, double rcf, int width, int height) {
    // Normalize RCF to account for the size-dependent scaling factor.
    FrameSize* frm_size  = &ppcs->av1_cm->frm_size;
    double     res_scale = (double)(frm_size->frame_width * frm_size->frame_height) / (width * height);

    rcf = fclamp(rcf, MIN_BPB_FACTOR, MAX_BPB_FACTOR) / res_scale;

    RATE_CONTROL* rc = &ppcs->scs->enc_ctx->rc;
    svt_block_on_mutex(rc->rc_mutex);
    rc->rcf_values[get_rcf_index(ppcs)] = rcf;
    if (rc->frames_since_key < rc->mini_qop_size) {
        // Reset all factors pessimistically at start as initial values could be off
        for (int i = 1; i < rc->mini_qop_size + 1; ++i) {
            rc->rcf_values[i] = AOMMAX(rcf, rc->rcf_values[i]);
        }
    }
    svt_release_mutex(rc->rc_mutex);
}

static double calculate_qindex(PictureControlSet* pcs, SequenceControlSet* scs) {
    PictureParentControlSet* ppcs   = pcs->ppcs;
    RATE_CONTROL*            rc     = &scs->enc_ctx->rc;
    RateControlCfg*          rc_cfg = &scs->enc_ctx->rc_cfg;

    int min_qindex = rc->best_quality;
    int max_qindex = rc->worst_quality;
    int max_size   = rc->max_frame_bandwidth;

    if (frame_is_intra_only(ppcs)) {
        rc->frames_to_key = scs->static_config.intra_period_length + 1;

        ppcs->base_frame_target = (int)(rc->avg_frame_bandwidth * rc->target_size_factors[0]);

        if (rc_cfg->max_intra_bitrate_pct && scs->enc_ctx->rc_cfg.mode == AOM_CBR) {
            int maxi = rc->avg_frame_bandwidth * rc_cfg->max_intra_bitrate_pct / 100;
            max_size = AOMMIN(max_size, maxi);
        }
    } else {
        rc->min_ref_base_q_idx = get_min_ref_base_q_idx(pcs);

        if (pcs->ref_slice_type[REF_LIST_0][0] == I_SLICE) {
            min_qindex = MAX(min_qindex, rc->min_ref_base_q_idx - 1 * 4);
        } else {
            EbReferenceObject* ref_obj = get_ref_obj(pcs, REF_LIST_0, 0);
            bool is_higher_layer       = ref_obj->tmp_layer_idx < pcs->temporal_layer_index && !scs->use_flat_ipp;
            int  min_limit             = is_higher_layer ? 0 : 4;
            int  max_limit             = is_higher_layer ? 32 : 16;
            min_qindex                 = MAX(min_qindex, rc->min_ref_base_q_idx - min_limit * 4);
            max_qindex                 = MIN(max_qindex, rc->min_ref_base_q_idx + max_limit * 4);
        }
        ppcs->base_frame_target = calc_pframe_target_size(ppcs);

        if (rc_cfg->max_inter_bitrate_pct && scs->enc_ctx->rc_cfg.mode == AOM_CBR) {
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

        int num_layers = scs->static_config.hierarchical_levels + 1;
        for (int k = 0; k < num_layers + 1; k++) {
            rc->target_size_factors[k] = 1.0; // flat at start
        }
        rc->mini_qop_size = 1 << (num_layers - 1);
        for (int k = 0; k < rc->mini_qop_size + 1; k++) {
            rc->rcf_values[k]   = 0.7;
            rc->rcf_kalman_P[k] = 1.0; // high initial uncertainty for fast convergence
            rc->rcf_kalman_R[k] = 0.08;
        }
        if (scs->enc_ctx->rc_cfg.mode == AOM_CBR) {
            // just to match existent CBR
            rc->target_size_factors[0] = rc->starting_buffer_level / (2.0 * rc->avg_frame_bandwidth);
        } else {
            // TODO: VBR
            rc->target_size_factors[0] = 10;
        }

#if USE_SCENE_CUT_WORKAROUND
        rc->ema_me_dist = 0;
#endif
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

    // Kalman filter update for RCF.
    // State: log(rcf). Observation: log(rcf * correction_factor) = log(rcf) + log(cf).
    // Innovation is log(correction_factor) — how much the model was off.
    //
    // Process noise Q: how much log(rcf) can change per frame (content variation).
    // Measurement noise R: adaptive, estimated from observed innovation variance.
    static const double Q     = 0.04;
    static const double R_MIN = 0.04; // floor to prevent over-reaction

    int k = get_rcf_index(ppcs);

    // Innovation in log domain
    double innovation = log(correction_factor);

    // Adaptive R: track EMA of squared innovations as measurement noise estimate
    rc->rcf_kalman_R[k] += 0.05 * (innovation * innovation - rc->rcf_kalman_R[k]);
    double R = AOMMAX(R_MIN, rc->rcf_kalman_R[k]);

    // Predict step: P increases by process noise
    double P = rc->rcf_kalman_P[k] + Q;
    // Update step: Kalman gain
    double K = P / (P + R);
    // Update state: log(rcf) += K * innovation  →  rcf *= exp(K * innovation)
    rcf *= exp(K * innovation);
    // Update estimation variance
    rc->rcf_kalman_P[k] = (1.0 - K) * P;

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

#if USE_SCENE_CUT_WORKAROUND
    rc->ema_me_dist += 0.5 * (rc->cur_avg_base_me_dist - rc->ema_me_dist);
#endif

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
