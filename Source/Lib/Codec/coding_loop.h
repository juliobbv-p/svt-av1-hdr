/*
* Copyright(c) 2019 Intel Corporation
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#ifndef EbCodingLoop_h
#define EbCodingLoop_h

#include "coding_unit.h"
#include "sequence_control_set.h"
#include "md_process.h"
#include "enc_dec_process.h"

#ifdef __cplusplus
extern "C" {
#endif
/*******************************************
     * ModeDecisionSb
     *   performs CL (SB)
     *******************************************/
bool svt_aom_pick_partition_pd0(SequenceControlSet* scs, PictureControlSet* pcs, ModeDecisionContext* ctx, MdScan* mds,
                                PC_TREE* pc_tree, int mi_row, int mi_col);
void svt_aom_init_sb_data(SequenceControlSet* scs, PictureControlSet* pcs, ModeDecisionContext* ctx);
bool svt_aom_pick_partition(SequenceControlSet* scs, PictureControlSet* pcs, ModeDecisionContext* ctx, MdScan* mds,
                            PC_TREE* pc_tree, int mi_row, int mi_col);
void svt_aom_pick_partition_lpd1(SequenceControlSet* scs, PictureControlSet* pcs, ModeDecisionContext* ctx, MdScan* mds,
                                 PC_TREE* pc_tree, int mi_row, int mi_col);
void svt_aom_encode_sb(SequenceControlSet* scs, PictureControlSet* pcs, EncDecContext* ctx, SuperBlock* sb_ptr,
                       PC_TREE* pc_tree, PARTITION_TREE* ptree, int mi_row, int mi_col);

void svt_aom_store16bit_input_src(EbPictureBufferDesc* input_sample16bit_buffer, PictureControlSet* pcs, uint32_t sb_x,
                                  uint32_t sb_y, uint32_t sb_w, uint32_t sb_h);

void svt_aom_residual_kernel(uint8_t* input, uint32_t input_offset, uint32_t input_stride, uint8_t* pred,
                             uint32_t pred_offset, uint32_t pred_stride, int16_t* residual, uint32_t residual_offset,
                             uint32_t residual_stride, bool hbd, uint32_t area_width, uint32_t area_height);

// Compute quadrant SSE from the selected reconstruction for partition pruning.
void svt_aom_calc_src_to_recon_dist_per_quadrant(PictureControlSet* pcs, ModeDecisionContext* ctx,
                                                 EbPictureBufferDesc* input_pic, uint32_t input_origin_index,
                                                 uint32_t                     input_cb_origin_in_index,
                                                 ModeDecisionCandidateBuffer* cand_bf, uint32_t blk_origin_index,
                                                 uint32_t blk_chroma_origin_index);

// Probe horizontal/vertical transforms for partition pruning; invalidate skipped probes.
void svt_aom_non_normative_txs(PictureControlSet* pcs, ModeDecisionContext* ctx, BlkStruct* blk_ptr,
                               ModeDecisionCandidateBuffer* cand_bf);

void svt_aom_move_blk_data(PictureControlSet* pcs, EncDecContext* ed_ctx, BlkStruct* src_cu, EcBlkStruct* dst_cu);
#ifdef __cplusplus
}
#endif
#endif // EbCodingLoop_h
