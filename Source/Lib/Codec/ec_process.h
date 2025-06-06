/*
* Copyright(c) 2019 Intel Corporation
* Copyright (c) 2019, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#ifndef EbEntropyCodingProcess_h
#define EbEntropyCodingProcess_h

#include "definitions.h"

#include "sys_resource_manager.h"
#include "pic_buffer_desc.h"
#include "enc_inter_prediction.h"
#include "entropy_coding.h"
#include "coding_unit.h"
#include "object.h"

/**************************************
 * Enc Dec Context
 **************************************/
typedef struct EntropyCodingContext {
    EbDctor  dctor;
    EbFifo  *enc_dec_input_fifo_ptr;
    EbFifo  *entropy_coding_output_fifo_ptr; // to packetization
    EbFifo  *rate_control_output_fifo_ptr; // feedback to rate control
    uint32_t sb_total_count;
    // Coding Unit Workspace---------------------------
    EbPictureBufferDesc *coeff_buffer_sb; //Used to hold quantized coeff for one TB in EncPass.

    //  Context Variables---------------------------------
    BlkStruct *blk_ptr;
    //const CodedBlockStats           *cu_stats;
    uint32_t blk_index;
    uint8_t  cu_depth;
    uint32_t cu_size;
    uint32_t cu_size_log2;
    uint32_t blk_org_x;
    uint32_t blk_org_y;
    uint32_t sb_origin_x;
    uint32_t sb_origin_y;
    uint32_t pu_itr;
    uint32_t pu_origin_x;
    uint32_t pu_origin_y;
    uint32_t pu_width;
    uint32_t pu_height;
    MvUnit   mv_unit;

    uint32_t txb_itr;
    uint32_t txb_origin_x;
    uint32_t txb_origin_y;
    uint32_t txb_size;

    // MCP Context
    bool        is_16bit; //enable 10 bit encode in CL
    int32_t     coded_area_sb;
    int32_t     coded_area_sb_uv;
    TOKENEXTRA *tok;
    MbModeInfo *mbmi;
} EntropyCodingContext;

/**************************************
 * Extern Function Declarations
 **************************************/
extern EbErrorType svt_aom_entropy_coding_context_ctor(EbThreadContext *thread_ctx, const EbEncHandle *enc_handle_ptr,
                                                       int index, int rate_control_index);

extern void *svt_aom_entropy_coding_kernel(void *input_ptr);

#endif // EbEntropyCodingProcess_h
