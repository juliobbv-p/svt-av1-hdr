#include <stdio.h>
#include <stdlib.h>

#include "pcs.h"
#include "resize.h"
#include "enc_dec_process.h"
#include "pd_process.h"
#include "pic_buffer_desc.h"

uint16_t svt_aom_get_max_can_count(EncMode enc_mode);
void     svt_aom_md_pme_search_controls(ModeDecisionContext* ctx, uint8_t md_pme_level);

void    svt_aom_set_txt_controls(ModeDecisionContext* ctx, uint8_t txt_level);
void    svt_aom_set_wm_controls(ModeDecisionContext* ctx, uint8_t wm_level);
uint8_t svt_aom_set_nic_controls(ModeDecisionContext* ctx, uint8_t nic_level);
uint8_t svt_aom_set_chroma_controls(ModeDecisionContext* ctx, uint8_t uv_level);
uint8_t svt_aom_get_update_cdf_level_default(EncMode enc_mode, SliceType is_islice, uint8_t is_base, uint8_t sc_class1);
uint8_t svt_aom_get_update_cdf_level_rtc(EncMode enc_mode, SliceType is_islice, uint8_t is_base, uint8_t sc_class1);
uint8_t svt_aom_get_update_cdf_level_allintra(EncMode enc_mode);
uint8_t svt_aom_get_chroma_level_default(EncMode enc_mode, const uint8_t is_islice);
uint8_t svt_aom_get_chroma_level_rtc(EncMode enc_mode, const uint8_t is_islice);
uint8_t svt_aom_get_chroma_level_allintra(EncMode enc_mode);
uint8_t svt_aom_get_bypass_encdec_default(EncMode enc_mode, uint8_t encoder_bit_depth);
uint8_t svt_aom_get_bypass_encdec_rtc(EncMode enc_mode, uint8_t encoder_bit_depth);
uint8_t svt_aom_get_bypass_encdec_allintra(EncMode enc_mode);
uint8_t svt_aom_get_nic_level_default(EncMode enc_mode, uint8_t is_base, uint8_t sc_class1);
uint8_t svt_aom_get_nic_level_rtc(EncMode enc_mode, bool use_flat_ipp);
uint8_t svt_aom_get_nic_level_allintra(EncMode enc_mode);
uint8_t svt_aom_get_enable_me_16x16(EncMode enc_mode);
bool    svt_aom_is_ref_same_size(PictureControlSet* pcs, uint8_t list_idx, uint8_t ref_idx);
uint8_t svt_aom_get_enable_me_8x8(EncMode enc_mode, ResolutionRange input_resolution, const bool rtc_tune,
                                  const bool flat_rtc_tune);
void    svt_aom_sig_deriv_mode_decision_config_default(SequenceControlSet* scs, PictureControlSet* pcs);
void    svt_aom_sig_deriv_mode_decision_config_rtc(SequenceControlSet* scs, PictureControlSet* pcs);
void    svt_aom_sig_deriv_mode_decision_config_allintra(SequenceControlSet* scs, PictureControlSet* pcs);
void    svt_aom_sig_deriv_block(PictureControlSet* pcs, ModeDecisionContext* ctx);
void    svt_aom_sig_deriv_pre_analysis_pcs(PictureParentControlSet* pcs);
void    svt_aom_sig_deriv_pre_analysis_scs(SequenceControlSet* scs);
void    svt_aom_sig_deriv_multi_processes_default(SequenceControlSet* scs, PictureParentControlSet* pcs);
void    svt_aom_sig_deriv_multi_processes_rtc(SequenceControlSet* scs, PictureParentControlSet* pcs);
void    svt_aom_sig_deriv_multi_processes_allintra(SequenceControlSet* scs, PictureParentControlSet* pcs);
void    svt_aom_sig_deriv_me_tf(PictureParentControlSet* pcs, MeContext* me_ctx);

void svt_aom_sig_deriv_enc_dec_light_pd1(PictureControlSet* pcs, ModeDecisionContext* ctx);
void svt_aom_sig_deriv_enc_dec_light_pd0(SequenceControlSet* scs, PictureControlSet* pcs, ModeDecisionContext* ctx);
void svt_aom_sig_deriv_enc_dec_common(SequenceControlSet* scs, PictureControlSet* pcs, ModeDecisionContext* ctx);

void svt_aom_sig_deriv_me(SequenceControlSet* scs, PictureParentControlSet* pcs, MeContext* me_ctx);
void svt_aom_sig_deriv_enc_dec_default(PictureControlSet* pcs, ModeDecisionContext* ctx);
void svt_aom_sig_deriv_enc_dec_rtc(PictureControlSet* pcs, ModeDecisionContext* ctx);
void svt_aom_sig_deriv_enc_dec_allintra(PictureControlSet* pcs, ModeDecisionContext* ctx);

void    svt_aom_set_gm_controls(PictureParentControlSet* pcs, uint8_t gm_level);
uint8_t svt_aom_derive_gm_level(PictureParentControlSet* pcs, bool super_res_off);

uint8_t svt_aom_get_enable_sg_default(EncMode enc_mode, uint8_t input_resolution, uint8_t fast_decode);
uint8_t svt_aom_get_enable_sg_rtc(EncMode enc_mode, uint8_t input_resolution, uint8_t fast_decode);
uint8_t svt_aom_get_enable_sg_allintra();
uint8_t svt_aom_get_enable_restoration_default(EncMode enc_mode, int8_t config_enable_restoration,
                                               uint8_t input_resolution, uint8_t fast_decode);
uint8_t svt_aom_get_enable_restoration_rtc(EncMode enc_mode, int8_t config_enable_restoration, uint8_t input_resolution,
                                           uint8_t fast_decode);
uint8_t svt_aom_get_enable_restoration_allintra(EncMode enc_mode, int8_t config_enable_restoration);
void    svt_aom_set_dist_based_ref_pruning_controls(ModeDecisionContext* ctx, uint8_t dist_based_ref_pruning_level);
bool    svt_aom_get_disallow_4x4_default(EncMode enc_mode);
bool    svt_aom_get_disallow_4x4_rtc(EncMode enc_mode);
bool    svt_aom_get_disallow_4x4_allintra(EncMode enc_mode);

bool    svt_aom_get_disallow_8x8_default();
bool    svt_aom_get_disallow_8x8_rtc(EncMode enc_mode, const uint16_t aligned_width, const uint16_t aligned_height);
bool    svt_aom_get_disallow_8x8_allintra();
uint8_t svt_aom_get_nsq_geom_level_default(EncMode enc_mode, InputCoeffLvl coeff_lvl);
uint8_t svt_aom_get_nsq_geom_level_rtc(EncMode enc_mode);
uint8_t svt_aom_get_nsq_geom_level_allintra(EncMode enc_mode);

uint8_t svt_aom_get_nsq_search_level_default(PictureControlSet* pcs, EncMode enc_mode, InputCoeffLvl coeff_lvl,
                                             uint32_t qp);
uint8_t svt_aom_get_nsq_search_level_rtc(PictureControlSet* pcs, EncMode enc_mode, InputCoeffLvl coeff_lvl,
                                         uint32_t qp);
uint8_t svt_aom_get_nsq_search_level_allintra(PictureControlSet* pcs, EncMode enc_mode, uint32_t qp);
uint8_t get_inter_compound_level(EncMode enc_mode);
uint8_t get_filter_intra_level_default(EncMode enc_mode);
uint8_t get_filter_intra_level_rtc(EncMode enc_mode);
uint8_t get_filter_intra_level_allintra(EncMode enc_mode);
uint8_t svt_aom_get_inter_intra_level(EncMode enc_mode, uint8_t transition_present);
uint8_t svt_aom_get_obmc_level(EncMode enc_mode, uint32_t qp, uint8_t seq_qp_mod);
void    svt_aom_set_nsq_geom_ctrls(ModeDecisionContext* ctx, uint8_t nsq_geom_level, uint8_t* allow_HVA_HVB,
                                   uint8_t* allow_HV4, uint8_t* min_nsq_bsize);
void    svt_aom_get_intra_mode_levels_default(EncMode enc_mode, bool is_islice, bool is_base, int transition_present,
                                              uint32_t* intra_level_ptr, uint32_t* dist_based_ang_intra_level_ptr);
void    svt_aom_get_intra_mode_levels_rtc(EncMode enc_mode, bool is_islice, bool sc_class1, int transition_present,
                                          bool flat_rtc_tune, uint32_t* intra_level_ptr,
                                          uint32_t* dist_based_ang_intra_level_ptr);
void    svt_aom_get_intra_mode_levels_allintra(EncMode enc_mode, uint32_t* intra_level_ptr,
                                               uint32_t* dist_based_ang_intra_level_ptr);
uint8_t svt_aom_get_tpl_synthesizer_block_size(int8_t tpl_level, uint32_t picture_width, uint32_t picture_height);

void svt_aom_set_mfmv_config(SequenceControlSet* scs);
void svt_aom_get_qp_based_th_scaling_factors(bool enable_qp_based_th_scaling, uint32_t* ret_q_weight,
                                             uint32_t* ret_q_weight_denom, uint32_t qp);
