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

#ifndef EbNeighborArrays_h
#define EbNeighborArrays_h

#include "definitions.h"
#include "object.h"
#include <string.h> /* memcpy / memset (compiler builtins) */

#ifdef __cplusplus
extern "C" {
#endif
// Neighbor Array Granulairity
#define PU_NEIGHBOR_ARRAY_GRANULARITY 4
#define SAMPLE_NEIGHBOR_ARRAY_GRANULARITY 1

typedef enum NeighborArrayType {
    NEIGHBOR_ARRAY_LEFT    = 0,
    NEIGHBOR_ARRAY_TOP     = 1,
    NEIGHBOR_ARRAY_TOPLEFT = 2,
    NEIGHBOR_ARRAY_INVALID = ~0
} NeighborArrayType;

#define NEIGHBOR_ARRAY_UNIT_LEFT_MASK (1 << NEIGHBOR_ARRAY_LEFT)
#define NEIGHBOR_ARRAY_UNIT_TOP_MASK (1 << NEIGHBOR_ARRAY_TOP)
#define NEIGHBOR_ARRAY_UNIT_TOPLEFT_MASK (1 << NEIGHBOR_ARRAY_TOPLEFT)

#define NEIGHBOR_ARRAY_UNIT_FULL_MASK \
    (NEIGHBOR_ARRAY_UNIT_LEFT_MASK | NEIGHBOR_ARRAY_UNIT_TOP_MASK | NEIGHBOR_ARRAY_UNIT_TOPLEFT_MASK)
#define NEIGHBOR_ARRAY_UNIT_TOP_AND_LEFT_ONLY_MASK (NEIGHBOR_ARRAY_UNIT_LEFT_MASK | NEIGHBOR_ARRAY_UNIT_TOP_MASK)

typedef struct NeighborArrayUnit {
    EbDctor  dctor;
    uint8_t* left_array;
    uint8_t* top_array;
    uint8_t* top_left_array;
    uint32_t left_array_size;
    uint32_t top_array_size;
    uint32_t top_left_array_size;
    uint32_t unit_size;
    uint8_t  granularity_log2;
} NeighborArrayUnit;

EbErrorType svt_aom_neighbor_array_unit_ctor(NeighborArrayUnit* na_unit_ptr, uint32_t max_picture_width,
                                             uint32_t max_picture_height, uint32_t unit_size,
                                             uint8_t granularity_normal, uint8_t type_mask);

void svt_aom_neighbor_array_unit_reset(NeighborArrayUnit* na_unit_ptr);

static INLINE uint32_t get_neighbor_array_unit_left_index(NeighborArrayUnit* na_unit_ptr, uint32_t loc_y) {
    return loc_y >> na_unit_ptr->granularity_log2;
}

static INLINE uint32_t get_neighbor_array_unit_top_index(NeighborArrayUnit* na_unit_ptr, uint32_t loc_x) {
    return loc_x >> na_unit_ptr->granularity_log2;
}

static INLINE uint32_t svt_aom_get_neighbor_array_unit_top_left_index(NeighborArrayUnit* na_unit_ptr, int32_t loc_x,
                                                                      int32_t loc_y) {
    return na_unit_ptr->left_array_size + (loc_x >> na_unit_ptr->granularity_log2) -
        (loc_y >> na_unit_ptr->granularity_log2);
}

/* Sample-grained top-left ring offset (skips no-op shift). */
static INLINE uint32_t svt_aom_na_topleft_offset(const NeighborArrayUnit* na, uint32_t x, uint32_t y) {
    return na->left_array_size + x - y;
}

/* Top-left ring pointer helpers — HBD-aware (unit_size baked when HBD=0). */
static INLINE uint8_t* svt_aom_na_topleft_ptr(NeighborArrayUnit* na, int32_t x, int32_t y) {
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
    return na->top_left_array + svt_aom_get_neighbor_array_unit_top_left_index(na, x, y) * na->unit_size;
#else
    return na->top_left_array + svt_aom_get_neighbor_array_unit_top_left_index(na, x, y);
#endif
}

/* Bottom-row corner of the block (x, y+h-1): used as the dst entry that the
 * block's bottom row is stored into for the NEXT block's top context. */
static INLINE uint8_t* svt_aom_na_botleft_ptr(NeighborArrayUnit* na, int32_t x, int32_t y, int32_t h) {
    return svt_aom_na_topleft_ptr(na, x, y + (h - 1));
}

/* Right-column corner of the block (x+w-1, y): used as the dst entry that the
 * block's right column is stored into for the NEXT block's left context. */
static INLINE uint8_t* svt_aom_na_topright_ptr(NeighborArrayUnit* na, int32_t x, int32_t y, int32_t w) {
    return svt_aom_na_topleft_ptr(na, x + (w - 1), y);
}

/* HBD-aware pointer helpers: returns na->{top,left}_array + idx*unit_size. */
static INLINE uint8_t* svt_aom_na_top_ptr(const NeighborArrayUnit* na, uint32_t loc_x) {
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
    return na->top_array + (loc_x >> na->granularity_log2) * na->unit_size;
#else
    return na->top_array + (loc_x >> na->granularity_log2);
#endif
}

static INLINE uint8_t* svt_aom_na_left_ptr(const NeighborArrayUnit* na, uint32_t loc_y) {
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
    return na->left_array + (loc_y >> na->granularity_log2) * na->unit_size;
#else
    return na->left_array + (loc_y >> na->granularity_log2);
#endif
}

/* Shape-B PU-granularity index helpers (log2=2 baked). */
static INLINE uint32_t svt_aom_na_top_index_pu(uint32_t loc_x) {
    return loc_x >> 2;
}

static INLINE uint32_t svt_aom_na_left_index_pu(uint32_t loc_y) {
    return loc_y >> 2;
}

/*************************************************
 * Neighbor Array Sample Update (8-bit)
 *
 * When CONFIG_ENABLE_HIGH_BIT_DEPTH==0, all NAs that flow here have
 * unit_size=1; bake as const so the compiler folds every `* na_unit_size`
 * at every inlined call site. HBD=1 path unchanged.
 *************************************************/
static INLINE void svt_aom_neighbor_array_unit_sample_write(NeighborArrayUnit* na_unit_ptr, uint8_t* src_ptr,
                                                            uint32_t stride, uint32_t src_origin_x,
                                                            uint32_t src_origin_y, uint32_t pic_origin_x,
                                                            uint32_t pic_origin_y, uint32_t bw, uint32_t bh,
                                                            uint8_t mask) {
    uint32_t idx;
    uint8_t* dst_ptr;
    uint8_t* read_ptr;

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
    const uint32_t na_unit_size = na_unit_ptr->unit_size;
#else
    const uint32_t na_unit_size = 1;
#endif

    src_ptr += ((src_origin_y * stride) + src_origin_x) * na_unit_size;

    if (mask & NEIGHBOR_ARRAY_UNIT_TOP_MASK) {
        read_ptr = src_ptr + ((bh - 1) * stride);

        dst_ptr = svt_aom_na_top_ptr(na_unit_ptr, pic_origin_x);

        /* Top: bottom row of source — unit-stride forward, fits memcpy. */
        for (idx = 0; idx < bw; ++idx) {
            *dst_ptr = *read_ptr;
            dst_ptr += na_unit_size;
            read_ptr += na_unit_size;
        }
    }

    if (mask & NEIGHBOR_ARRAY_UNIT_LEFT_MASK) {
        read_ptr = src_ptr + (bw - 1);

        dst_ptr = svt_aom_na_left_ptr(na_unit_ptr, pic_origin_y);

        /* Left: column gather — dst step=1, read step=stride. */
        for (idx = 0; idx < bh; ++idx) {
            *dst_ptr++ = *read_ptr;
            read_ptr += stride;
        }
    }

    if (mask & NEIGHBOR_ARRAY_UNIT_TOPLEFT_MASK) {
        read_ptr = src_ptr + ((bh - 1) * stride);

        dst_ptr = svt_aom_na_botleft_ptr(na_unit_ptr, pic_origin_x, pic_origin_y, bh);

        memcpy(dst_ptr, read_ptr, bw);

        read_ptr = src_ptr + (bw - 1);

        dst_ptr = svt_aom_na_topright_ptr(na_unit_ptr, pic_origin_x, pic_origin_y, bw);

        /* Top-left right column: dst step=-1, read step=stride. */
        for (idx = 0; idx < bh; ++idx) {
            *dst_ptr-- = *read_ptr;
            read_ptr += stride;
        }
    }
}

/* Shape-B PU-granularity pointer helpers (unit_size=1, log2=2). */
static INLINE uint8_t* svt_aom_na_top_ptr_pu(const NeighborArrayUnit* na, uint32_t loc_x) {
    return na->top_array + (loc_x >> 2);
}

static INLINE uint8_t* svt_aom_na_left_ptr_pu(const NeighborArrayUnit* na, uint32_t loc_y) {
    return na->left_array + (loc_y >> 2);
}

/*************************************************
 * Update Recon Neighbor Array (8-bit)
 *
 * When CONFIG_ENABLE_HIGH_BIT_DEPTH==0, all NAs that flow here have
 * unit_size=1 (the 16-bit twins don't exist), so we bake it as const and
 * the compiler folds every `* na_unit_size` to identity. HBD=1 path stays
 * runtime-read for safety.
 *************************************************/
static INLINE void svt_aom_update_recon_neighbor_array(NeighborArrayUnit* na_unit_ptr, uint8_t* src_ptr_top,
                                                       uint8_t* src_ptr_left, uint32_t pic_origin_x,
                                                       uint32_t pic_origin_y, uint32_t bw, uint32_t bh) {
    uint8_t* dst_ptr;

    dst_ptr = svt_aom_na_top_ptr(na_unit_ptr, pic_origin_x);
    memcpy(dst_ptr, src_ptr_top, bw);

    dst_ptr = svt_aom_na_left_ptr(na_unit_ptr, pic_origin_y);
    memcpy(dst_ptr, src_ptr_left, bh);

    /* Top-left bottom row: forward unit-stride memcpy. */
    dst_ptr = svt_aom_na_botleft_ptr(na_unit_ptr, pic_origin_x, pic_origin_y, bh);
    memcpy(dst_ptr, src_ptr_top, bw);

    /* Top-left right column: dst step=-1, read step=1 (unit-stride consume of src_ptr_left). */
    uint8_t* read_ptr = src_ptr_left;
    dst_ptr           = svt_aom_na_topright_ptr(na_unit_ptr, pic_origin_x, pic_origin_y, bw);

    for (uint32_t idx = 0; idx < bh; ++idx) {
        *dst_ptr-- = *read_ptr++;
    }
}

/*************************************************
 * Update Recon Neighbor Array (16-bit)
 * Moved to header as static inline (Step C). Body verbatim from OOL.
 *************************************************/
static INLINE void svt_aom_update_recon_neighbor_array16bit(NeighborArrayUnit* na_unit_ptr, uint16_t* src_ptr_top,
                                                            uint16_t* src_ptr_left, uint32_t pic_origin_x,
                                                            uint32_t pic_origin_y, uint32_t bw, uint32_t bh) {
    uint16_t* dst_ptr;
    dst_ptr = (uint16_t*)svt_aom_na_top_ptr(na_unit_ptr, pic_origin_x);
    memcpy(dst_ptr, src_ptr_top, bw * sizeof(uint16_t));

    dst_ptr = (uint16_t*)svt_aom_na_left_ptr(na_unit_ptr, pic_origin_y);
    memcpy(dst_ptr, src_ptr_left, bh * sizeof(uint16_t));

    /* Top-left bottom row: forward unit-stride memcpy. */
    dst_ptr = (uint16_t*)svt_aom_na_botleft_ptr(na_unit_ptr, pic_origin_x, pic_origin_y, bh);
    memcpy(dst_ptr, src_ptr_top, bw * sizeof(uint16_t));

    /* Top-left right column: dst step=-1, read step=1 (unit-stride consume). */
    uint16_t* read_ptr = src_ptr_left;
    dst_ptr            = (uint16_t*)svt_aom_na_topright_ptr(na_unit_ptr, pic_origin_x, pic_origin_y, bw);
    for (uint32_t idx = 0; idx < bh; ++idx) {
        *dst_ptr-- = *read_ptr++;
    }
}

/*************************************************
 * Copy Neighbor Array — generic version (reads shape from NA fields).
 * Use this only when shape isn't known at the call site. Otherwise prefer
 * svt_aom_copy_neigh_arr_pu() which bakes Shape-B constants.
 *
 * When CONFIG_ENABLE_HIGH_BIT_DEPTH==0 (RTC build), no 16-bit NA ever
 * exists so every NA passed here has unit_size==1. Baking that as a const
 * lets the compiler fold every `na_unit_size *` multiply at every inlined
 * call site, including the 29 Shape-A recon call sites that we cannot
 * route through `_pu` (different granularity_log2).
 *************************************************/
static INLINE void svt_aom_copy_neigh_arr(NeighborArrayUnit* na_src, NeighborArrayUnit* na_dst, uint32_t org_x,
                                          uint32_t org_y, uint32_t bw, uint32_t bh, uint8_t mask) {
    uint8_t *dst_ptr, *src_ptr;
    uint32_t count;
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
    uint32_t na_unit_size = na_src->unit_size;
#else
    const uint32_t na_unit_size = 1; /* HBD off: every NA has unit_size=1 */
#endif

    if (mask & NEIGHBOR_ARRAY_UNIT_TOP_MASK) {
        src_ptr = svt_aom_na_top_ptr(na_src, org_x);
        dst_ptr = svt_aom_na_top_ptr(na_dst, org_x);
        count   = bw >> na_src->granularity_log2;
        memcpy(dst_ptr, src_ptr, na_unit_size * count);
    }
    if (mask & NEIGHBOR_ARRAY_UNIT_LEFT_MASK) {
        src_ptr = svt_aom_na_left_ptr(na_src, org_y);
        dst_ptr = svt_aom_na_left_ptr(na_dst, org_y);
        count   = bh >> na_src->granularity_log2;
        memcpy(dst_ptr, src_ptr, na_unit_size * count);
    }
    if (mask & NEIGHBOR_ARRAY_UNIT_TOPLEFT_MASK) {
        src_ptr = svt_aom_na_topleft_ptr(na_src, org_x, org_y + (bh - 1));
        dst_ptr = svt_aom_na_topleft_ptr(na_dst, org_x, org_y + (bh - 1));
        count   = ((bw + bh) >> na_src->granularity_log2) - 1;
        memcpy(dst_ptr, src_ptr, na_unit_size * count);
    }
}

/*************************************************
 * Shape-B specialized copy: caller asserts unit_size=1, granularity=PU=4
 * (log2=2). Multiplies fold to identity, shifts fold to constants.
 * memcpy still used for the bulk byte copy.
 *************************************************/
static INLINE void svt_aom_copy_neigh_arr_pu(NeighborArrayUnit* na_src, NeighborArrayUnit* na_dst, uint32_t org_x,
                                             uint32_t org_y, uint32_t bw, uint32_t bh, uint8_t mask) {
    if (mask & NEIGHBOR_ARRAY_UNIT_TOP_MASK) {
        uint32_t off   = org_x >> 2;
        uint32_t bytes = bw >> 2;
        memcpy(na_dst->top_array + off, na_src->top_array + off, bytes);
    }
    if (mask & NEIGHBOR_ARRAY_UNIT_LEFT_MASK) {
        uint32_t off   = org_y >> 2;
        uint32_t bytes = bh >> 2;
        memcpy(na_dst->left_array + off, na_src->left_array + off, bytes);
    }
    if (mask & NEIGHBOR_ARRAY_UNIT_TOPLEFT_MASK) {
        uint32_t off   = (uint32_t)na_src->left_array_size + (org_x >> 2) - ((org_y + (bh - 1)) >> 2);
        uint32_t bytes = ((bw + bh) >> 2) - 1;
        memcpy(na_dst->top_left_array + off, na_src->top_left_array + off, bytes);
    }
}

/*************************************************
 * Neighbor Array Sample Update (16-bit)
 * Moved to header as static inline (Step E). Body verbatim from OOL.
 *************************************************/
static INLINE void svt_aom_neighbor_array_unit16bit_sample_write(NeighborArrayUnit* na_unit_ptr, uint16_t* src_ptr,
                                                                 uint32_t stride, uint32_t src_origin_x,
                                                                 uint32_t src_origin_y, uint32_t pic_origin_x,
                                                                 uint32_t pic_origin_y, uint32_t bw, uint32_t bh,
                                                                 uint8_t mask) {
    uint32_t  idx;
    uint16_t* dst_ptr;
    uint16_t* read_ptr;

    src_ptr += ((src_origin_y * stride) + src_origin_x);

    if (mask & NEIGHBOR_ARRAY_UNIT_TOP_MASK) {
        read_ptr = src_ptr + ((bh - 1) * stride);
        dst_ptr  = (uint16_t*)(na_unit_ptr->top_array) + get_neighbor_array_unit_top_index(na_unit_ptr, pic_origin_x);
        /* Top: bottom row — unit-stride forward, fits memcpy. */
        memcpy(dst_ptr, read_ptr, bw * sizeof(uint16_t));
    }

    if (mask & NEIGHBOR_ARRAY_UNIT_LEFT_MASK) {
        read_ptr = src_ptr + (bw - 1);
        dst_ptr  = (uint16_t*)(na_unit_ptr->left_array) + get_neighbor_array_unit_left_index(na_unit_ptr, pic_origin_y);
        /* Left: column gather — dst step=1, read step=stride. */
        for (idx = 0; idx < bh; ++idx) {
            *dst_ptr++ = *read_ptr;
            read_ptr += stride;
        }
    }

    if (mask & NEIGHBOR_ARRAY_UNIT_TOPLEFT_MASK) {
        read_ptr = src_ptr + ((bh - 1) * stride);
        dst_ptr  = (uint16_t*)(na_unit_ptr->top_left_array) +
            svt_aom_get_neighbor_array_unit_top_left_index(na_unit_ptr, pic_origin_x, pic_origin_y + (bh - 1));
        /* Top-left bottom row: forward unit-stride memcpy. */
        memcpy(dst_ptr, read_ptr, bw * sizeof(uint16_t));

        read_ptr = src_ptr + (bw - 1);
        dst_ptr  = (uint16_t*)(na_unit_ptr->top_left_array) +
            svt_aom_get_neighbor_array_unit_top_left_index(na_unit_ptr, pic_origin_x + (bw - 1), pic_origin_y);
        /* Top-left right column: dst step=-1, read step=stride. */
        for (idx = 0; idx < bh; ++idx) {
            *dst_ptr-- = *read_ptr;
            read_ptr += stride;
        }
    }
}

/*************************************************
 * Neighbor Array Unit Mode Write — generic version (reads shape from NA
 * fields). Use this when the NA shape isn't known at the call site. If you
 * know the shape, prefer svt_aom_neighbor_array_unit_mode_write_pu() — it
 * bakes Shape-B constants (unit_size=1, granularity_log2=2) into the body
 * and the compiler folds away the multiplies, shifts, and inner per-byte
 * loop entirely.
 *************************************************/
static INLINE void svt_aom_neighbor_array_unit_mode_write(NeighborArrayUnit* na_unit_ptr, uint8_t* value,
                                                          uint32_t org_x, uint32_t org_y, uint32_t bw, uint32_t bh,
                                                          uint8_t mask) {
    uint32_t idx;
    uint8_t* dst_ptr;

    uint32_t count;
    uint32_t na_offset;
    uint32_t na_unit_size;

    na_unit_size = na_unit_ptr->unit_size;

    if (mask & NEIGHBOR_ARRAY_UNIT_TOP_MASK) {
        na_offset = get_neighbor_array_unit_top_index(na_unit_ptr, org_x);
        dst_ptr   = na_unit_ptr->top_array + na_offset * na_unit_size;
        count     = bw >> na_unit_ptr->granularity_log2;
        for (idx = 0; idx < count; ++idx) {
            memcpy(dst_ptr, value, na_unit_size);
            dst_ptr += na_unit_size;
        }
    }
    if (mask & NEIGHBOR_ARRAY_UNIT_LEFT_MASK) {
        na_offset = get_neighbor_array_unit_left_index(na_unit_ptr, org_y);
        dst_ptr   = na_unit_ptr->left_array + na_offset * na_unit_size;
        count     = bh >> na_unit_ptr->granularity_log2;
        for (idx = 0; idx < count; ++idx) {
            memcpy(dst_ptr, value, na_unit_size);
            dst_ptr += na_unit_size;
        }
    }
    if (mask & NEIGHBOR_ARRAY_UNIT_TOPLEFT_MASK) {
        na_offset = svt_aom_get_neighbor_array_unit_top_left_index(na_unit_ptr, org_x, org_y + (bh - 1));
        dst_ptr   = na_unit_ptr->top_left_array + na_offset * na_unit_size;
        count     = ((bw + bh) >> na_unit_ptr->granularity_log2) - 1;
        for (idx = 0; idx < count; ++idx) {
            memcpy(dst_ptr, value, na_unit_size);
            dst_ptr += na_unit_size;
        }
    }
}

/*************************************************
 * Shape-B specialized mode-write: caller asserts unit_size=1,
 * granularity=PU=4 (log2=2). Constants are literal; compiler folds the
 * multiplies/shifts and the inner per-byte fill becomes a single byte store.
 * API matches svt_aom_neighbor_array_unit_mode_write — drop-in replacement
 * at every Shape-B call site.
 *************************************************/
static INLINE void svt_aom_neighbor_array_unit_mode_write_pu(NeighborArrayUnit* na_unit_ptr, uint8_t* value,
                                                             uint32_t org_x, uint32_t org_y, uint32_t bw, uint32_t bh,
                                                             uint8_t mask) {
    const uint8_t v = *value; /* unit_size=1: read once */

    if (mask & NEIGHBOR_ARRAY_UNIT_TOP_MASK) {
        uint8_t* dst   = na_unit_ptr->top_array + (org_x >> 2);
        uint32_t count = bw >> 2;
        memset(dst, v, count);
    }
    if (mask & NEIGHBOR_ARRAY_UNIT_LEFT_MASK) {
        uint8_t* dst   = na_unit_ptr->left_array + (org_y >> 2);
        uint32_t count = bh >> 2;
        memset(dst, v, count);
    }
    if (mask & NEIGHBOR_ARRAY_UNIT_TOPLEFT_MASK) {
        uint32_t off   = (uint32_t)na_unit_ptr->left_array_size + (org_x >> 2) - ((org_y + (bh - 1)) >> 2);
        uint8_t* dst   = na_unit_ptr->top_left_array + off;
        uint32_t count = ((bw + bh) >> 2) - 1;
        memset(dst, v, count);
    }
}
#ifdef __cplusplus
}
#endif
#endif //EbNeighborArrays_h
