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

#include <stdlib.h>
#include <string.h>

#include "neighbor_arrays.h"
#include "utility.h"

static void neighbor_array_unit_dctor(EbPtr p) {
    NeighborArrayUnit* obj = (NeighborArrayUnit*)p;
    EB_FREE(obj->left_array);
    EB_FREE(obj->top_array);
    EB_FREE(obj->top_left_array);
}

EbErrorType svt_aom_neighbor_array_unit_ctor(NeighborArrayUnit* na_unit_ptr, uint32_t max_picture_width,
                                             uint32_t max_picture_height, uint32_t unit_size,
                                             uint8_t granularity_normal, uint8_t type_mask) {
    na_unit_ptr->dctor            = neighbor_array_unit_dctor;
    na_unit_ptr->unit_size        = (uint8_t)(unit_size);
    na_unit_ptr->granularity_log2 = (uint8_t)(svt_log2f(granularity_normal));

    na_unit_ptr->left_array_size     = (type_mask & NEIGHBOR_ARRAY_UNIT_LEFT_MASK)
            ? max_picture_height >> na_unit_ptr->granularity_log2
            : 0;
    na_unit_ptr->top_array_size      = (type_mask & NEIGHBOR_ARRAY_UNIT_TOP_MASK)
             ? max_picture_width >> na_unit_ptr->granularity_log2
             : 0;
    na_unit_ptr->top_left_array_size = (type_mask & NEIGHBOR_ARRAY_UNIT_TOPLEFT_MASK)
        ? (max_picture_width + max_picture_height) >> na_unit_ptr->granularity_log2
        : 0;

    if (na_unit_ptr->left_array_size) {
        EB_MALLOC(na_unit_ptr->left_array, na_unit_ptr->unit_size * na_unit_ptr->left_array_size);
    }
    if (na_unit_ptr->top_array_size) {
        EB_MALLOC(na_unit_ptr->top_array, na_unit_ptr->unit_size * na_unit_ptr->top_array_size);
    }
    if (na_unit_ptr->top_left_array_size) {
        EB_MALLOC(na_unit_ptr->top_left_array, na_unit_ptr->unit_size * na_unit_ptr->top_left_array_size);
    }
    return EB_ErrorNone;
}

void svt_aom_neighbor_array_unit_reset(NeighborArrayUnit* na_unit_ptr) {
    if (na_unit_ptr->left_array) {
        svt_memset(
            na_unit_ptr->left_array, NEIGHBOR_ARRAY_INVALID, na_unit_ptr->unit_size * na_unit_ptr->left_array_size);
    }
    if (na_unit_ptr->top_array) {
        svt_memset(
            na_unit_ptr->top_array, NEIGHBOR_ARRAY_INVALID, na_unit_ptr->unit_size * na_unit_ptr->top_array_size);
    }
    if (na_unit_ptr->top_left_array) {
        svt_memset(na_unit_ptr->top_left_array,
                   NEIGHBOR_ARRAY_INVALID,
                   na_unit_ptr->unit_size * na_unit_ptr->top_left_array_size);
    }
}
