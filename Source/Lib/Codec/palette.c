/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#include <math.h>
#include <stdlib.h>
#include "definitions.h"
#include "md_process.h"
#include "aom_dsp_rtcd.h"

#define DIVIDE_AND_ROUND(x, y) (((x) + ((y) >> 1)) / (y))

#define AV1_K_MEANS_RENAME(func, dim) func##_dim##dim##_c

void AV1_K_MEANS_RENAME(svt_av1_calc_indices, 1)(const int* data, const int* centroids, uint8_t* indices, int n, int k);
void AV1_K_MEANS_RENAME(svt_av1_calc_indices, 2)(const int* data, const int* centroids, uint8_t* indices, int n, int k);
void AV1_K_MEANS_RENAME(svt_av1_k_means, 1)(const int* data, int* centroids, uint8_t* indices, int n, int k,
                                            int max_itr);
void AV1_K_MEANS_RENAME(svt_av1_k_means, 2)(const int* data, int* centroids, uint8_t* indices, int n, int k,
                                            int max_itr);

// Given 'n' 'data' points and 'k' 'centroids' each of dimension 'dim',
// calculate the centroid 'indices' for the data points.
static inline void av1_calc_indices(const int* data, const int* centroids, uint8_t* indices, int n, int k, int dim) {
    if (dim == 1) {
        svt_av1_calc_indices_dim1(data, centroids, indices, n, k);
    } else if (dim == 2) {
        svt_av1_calc_indices_dim2(data, centroids, indices, n, k);
    } else {
        assert(0 && "Untemplated k means dimension");
    }
}

// Given 'n' 'data' points and an initial guess of 'k' 'centroids' each of
// dimension 'dim', runs up to 'max_itr' iterations of k-means algorithm to get
// updated 'centroids' and the centroid 'indices' for elements in 'data'.
// Note: the output centroids are rounded off to nearest integers.
static inline void av1_k_means(const int* data, int* centroids, uint8_t* indices, int n, int k, int dim, int max_itr) {
    if (dim == 1) {
        svt_av1_k_means_dim1(data, centroids, indices, n, k, max_itr);
    } else if (dim == 2) {
        svt_av1_k_means_dim2(data, centroids, indices, n, k, max_itr);
    } else {
        assert(0 && "Untemplated k means dimension");
    }
}

#define AV1_K_MEANS_DIM 1
#include "k_means_template.h"
#undef AV1_K_MEANS_DIM
#define AV1_K_MEANS_DIM 2
#include "k_means_template.h"
#undef AV1_K_MEANS_DIM

static int int_comparer(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

static int av1_remove_duplicates(int* centroids, int num_centroids) {
    int num_unique; // number of unique centroids
    int i;
    qsort(centroids, num_centroids, sizeof(*centroids), int_comparer);
    // Remove duplicates.
    num_unique = 1;
    for (i = 1; i < num_centroids; ++i) {
        if (centroids[i] != centroids[i - 1]) { // found a new unique centroid
            centroids[num_unique++] = centroids[i];
        }
    }
    return num_unique;
}

static int delta_encode_cost(const int* colors, int num, int bit_depth, int min_val) {
    if (num <= 0) {
        return 0;
    }
    int bits_cost = bit_depth;
    if (num == 1) {
        return bits_cost;
    }
    bits_cost += 2;
    int       max_delta = 0;
    int       deltas[PALETTE_MAX_SIZE];
    const int min_bits = bit_depth - 3;
    for (int i = 1; i < num; ++i) {
        const int delta = colors[i] - colors[i - 1];
        deltas[i - 1]   = delta;
        assert(delta >= min_val);
        if (delta > max_delta) {
            max_delta = delta;
        }
    }
    int bits_per_delta = AOMMAX(av1_ceil_log2(max_delta + 1 - min_val), min_bits);
    assert(bits_per_delta <= bit_depth);
    int range = (1 << bit_depth) - colors[0] - min_val;
    for (int i = 0; i < num - 1; ++i) {
        bits_cost += bits_per_delta;
        range -= deltas[i];
        bits_per_delta = AOMMIN(bits_per_delta, av1_ceil_log2(range));
    }
    return bits_cost;
}

int svt_av1_index_color_cache(const uint16_t* color_cache, int n_cache, const uint16_t* colors, int n_colors,
                              uint8_t* cache_color_found, int* out_cache_colors) {
    if (n_cache <= 0) {
        for (int i = 0; i < n_colors; ++i) {
            out_cache_colors[i] = colors[i];
        }
        return n_colors;
    }
    memset(cache_color_found, 0, n_cache * sizeof(*cache_color_found));
    int n_in_cache = 0;
    int in_cache_flags[PALETTE_MAX_SIZE];
    memset(in_cache_flags, 0, sizeof(in_cache_flags));
    for (int i = 0; i < n_cache && n_in_cache < n_colors; ++i) {
        for (int j = 0; j < n_colors; ++j) {
            if (colors[j] == color_cache[i]) {
                in_cache_flags[j]    = 1;
                cache_color_found[i] = 1;
                ++n_in_cache;
                break;
            }
        }
    }
    int j = 0;
    for (int i = 0; i < n_colors; ++i) {
        if (!in_cache_flags[i]) {
            out_cache_colors[j++] = colors[i];
        }
    }
    assert(j == n_colors - n_in_cache);
    return j;
}

int svt_av1_palette_color_cost_y(const PaletteModeInfo* const pmi, uint16_t* color_cache, const int palette_size,
                                 int n_cache, int bit_depth) {
    const int n = palette_size;
    int       out_cache_colors[PALETTE_MAX_SIZE];
    uint8_t   cache_color_found[2 * PALETTE_MAX_SIZE];
    const int n_out_cache = svt_av1_index_color_cache(
        color_cache, n_cache, pmi->palette_colors, n, cache_color_found, out_cache_colors);
    const int total_bits = n_cache + delta_encode_cost(out_cache_colors, n_out_cache, bit_depth, 1);
    return av1_cost_literal(total_bits);
}

static void palette_add_to_cache(uint16_t* cache, int* n, uint16_t val) {
    // Do not add an already existing value
    if (*n > 0 && val == cache[*n - 1]) {
        return;
    }

    cache[(*n)++] = val;
}

// Get palette cache for luma only
int svt_get_palette_cache_y(const MacroBlockD* const xd, uint16_t* cache) {
    const int row = -xd->mb_to_top_edge >> 3;
    // Do not refer to above SB row when on SB boundary.
    const MbModeInfo* const above_mi = (row % (1 << MIN_SB_SIZE_LOG2)) ? xd->above_mbmi : NULL;
    const MbModeInfo* const left_mi  = xd->left_mbmi;
    int                     above_n = 0, left_n = 0;
    if (above_mi) {
        above_n = above_mi->palette_mode_info.palette_size;
    }
    if (left_mi) {
        left_n = left_mi->palette_mode_info.palette_size;
    }
    if (above_n == 0 && left_n == 0) {
        return 0;
    }
    int             above_idx    = 0;
    int             left_idx     = 0;
    int             n            = 0;
    const uint16_t* above_colors = above_mi ? above_mi->palette_mode_info.palette_colors : NULL;
    const uint16_t* left_colors  = left_mi ? left_mi->palette_mode_info.palette_colors : NULL;
    // Merge the sorted lists of base colors from above and left to get
    // combined sorted color cache.
    while (above_n > 0 && left_n > 0) {
        uint16_t v_above = above_colors[above_idx];
        uint16_t v_left  = left_colors[left_idx];
        if (v_left < v_above) {
            palette_add_to_cache(cache, &n, v_left);
            ++left_idx, --left_n;
        } else {
            palette_add_to_cache(cache, &n, v_above);
            ++above_idx, --above_n;
            if (v_left == v_above) {
                ++left_idx, --left_n;
            }
        }
    }
    while (above_n-- > 0) {
        uint16_t val = above_colors[above_idx++];
        palette_add_to_cache(cache, &n, val);
    }
    while (left_n-- > 0) {
        uint16_t val = left_colors[left_idx++];
        palette_add_to_cache(cache, &n, val);
    }
    assert(n <= 2 * PALETTE_MAX_SIZE);
    return n;
}

// Returns sub-sampled dimensions of the given block.
// The output values for 'rows_within_bounds' and 'cols_within_bounds' will
// differ from 'height' and 'width' when part of the block is outside the
// right
// and/or bottom image boundary.
void svt_aom_get_block_dimensions(BlockSize bsize, int plane, const MacroBlockD* xd, int* width, int* height,
                                  int* rows_within_bounds, int* cols_within_bounds) {
    const int block_height = block_size_high[bsize];
    const int block_width  = block_size_wide[bsize];
    const int block_rows   = (xd->mb_to_bottom_edge >= 0) ? block_height : (xd->mb_to_bottom_edge >> 3) + block_height;
    const int block_cols   = (xd->mb_to_right_edge >= 0) ? block_width : (xd->mb_to_right_edge >> 3) + block_width;

    uint8_t subsampling_x = plane == 0 ? 0 : 1;
    uint8_t subsampling_y = plane == 0 ? 0 : 1;

    assert(block_width >= block_cols);
    assert(block_height >= block_rows);
    const int plane_block_width  = block_width >> subsampling_x;
    const int plane_block_height = block_height >> subsampling_y;
    // Special handling for chroma sub8x8.
    const int is_chroma_sub8_x = plane > 0 && plane_block_width < 4;
    const int is_chroma_sub8_y = plane > 0 && plane_block_height < 4;
    if (width) {
        *width = plane_block_width + 2 * is_chroma_sub8_x;
    }
    if (height) {
        *height = plane_block_height + 2 * is_chroma_sub8_y;
    }
    if (rows_within_bounds) {
        *rows_within_bounds = (block_rows >> subsampling_y) + 2 * is_chroma_sub8_y;
    }
    if (cols_within_bounds) {
        *cols_within_bounds = (block_cols >> subsampling_x) + 2 * is_chroma_sub8_x;
    }
}

// Bias toward using colors in the cache.
// TODO: Try other schemes to improve compression.
static AOM_INLINE void optimize_palette_colors(uint16_t* color_cache, int n_cache, int n_colors, int stride,
                                               int* centroids, int bit_depth, uint8_t qp_index) {
    if (n_cache <= 0) {
        return;
    }
    for (int i = 0; i < n_colors * stride; i += stride) {
        int min_diff = abs((int)centroids[i] - (int)color_cache[0]);
        int idx      = 0;
        for (int j = 1; j < n_cache; ++j) {
            const int this_diff = abs((int)centroids[i] - (int)color_cache[j]);
            if (this_diff < min_diff) {
                min_diff = this_diff;
                idx      = j;
            }
        }
        const int min_threshold = (6 + (qp_index >> 6)) << (bit_depth - 8);
        if (min_diff <= min_threshold) {
            centroids[i] = color_cache[idx];
        }
    }
}

// Extends 'color_map' array from 'orig_width x orig_height' to 'new_width x
// new_height'. Extra rows and columns are filled in by copying last valid
// row/column.
static AOM_INLINE void extend_palette_color_map(uint8_t* const color_map, int orig_width, int orig_height,
                                                int new_width, int new_height) {
    int j;
    assert(new_width >= orig_width);
    assert(new_height >= orig_height);
    if (new_width == orig_width && new_height == orig_height) {
        return;
    }

    for (j = orig_height - 1; j >= 0; --j) {
        memmove(color_map + j * new_width, color_map + j * orig_width, orig_width);
        // Copy last column to extra columns.
        memset(
            color_map + j * new_width + orig_width, color_map[j * new_width + orig_width - 1], new_width - orig_width);
    }
    // Copy last row to extra rows.
    for (j = orig_height; j < new_height; ++j) {
        svt_memcpy(color_map + j * new_width, color_map + (orig_height - 1) * new_width, new_width);
    }
}

static void palette_rd_y(PaletteInfo* palette_info, uint8_t* palette_size_array, ModeDecisionContext* ctx,
                         bool opt_colors, BlockSize bsize, const int* data, int* centroids, int n,
                         uint16_t* color_cache, int n_cache, int bit_depth) {
    if (opt_colors) {
        optimize_palette_colors(color_cache, n_cache, n, 1, centroids, bit_depth, ctx->qp_index);
    }
    int k = av1_remove_duplicates(centroids, n);
    if (k < PALETTE_MIN_SIZE) {
        // Too few unique colors to create a palette. And DC_PRED will work
        // well for that case anyway. So skip.
        palette_size_array[0] = 0;
        return;
    }

    if (bit_depth > EB_EIGHT_BIT) {
        for (int i = 0; i < k; ++i) {
            palette_info->pmi.palette_colors[i] = clip_pixel_highbd((int)centroids[i], bit_depth);
        }
    } else {
        for (int i = 0; i < k; ++i) {
            palette_info->pmi.palette_colors[i] = clip_pixel(centroids[i]);
        }
    }
    palette_size_array[0]    = k;
    uint8_t* const color_map = palette_info->color_idx_map;
    int            block_width, block_height, rows, cols;
    svt_aom_get_block_dimensions(bsize, 0, ctx->blk_ptr->av1xd, &block_width, &block_height, &rows, &cols);
    av1_calc_indices(data, centroids, color_map, rows * cols, k, 1);
    extend_palette_color_map(color_map, cols, rows, block_width, block_height);
}

int svt_av1_count_colors(const uint8_t* src, int stride, int rows, int cols, int* val_count);
int svt_av1_count_colors_highbd(uint16_t* src, int stride, int rows, int cols, int bit_depth, int* val_count);

static void cache_based_centroid_refinement(int* data, int rows, int cols, int n, int* centroids,
                                            uint8_t* color_idx_map, uint16_t* color_cache, int n_cache) {
    const int total = rows * cols;
    uint8_t   temp_map[MAX_SB_SQUARE];

    // Compute baseline SSE
    uint64_t baseline_sse = 0;
    for (int i = 0; i < total; i++) {
        int diff = data[i] - centroids[color_idx_map[i]];
        baseline_sse += (uint64_t)diff * diff;
    }

    for (int c = 0; c < n; c++) {
        int      original = centroids[c];
        int      best_val = original;
        uint64_t best_sse = baseline_sse;

        for (int k = 0; k < n_cache; k++) {
            int candidate = color_cache[k];
            if (candidate == original) {
                continue;
            }

            centroids[c] = candidate;

            // Reassign pixels
            for (int i = 0; i < total; i++) {
                int best_idx  = 0;
                int best_dist = abs(data[i] - centroids[0]);

                for (int j = 1; j < n; j++) {
                    int dist = abs(data[i] - centroids[j]);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_idx  = j;
                    }
                }
                temp_map[i] = best_idx;
            }

            // Compute SSE
            uint64_t sse = 0;
            for (int i = 0; i < total; i++) {
                int diff = data[i] - centroids[temp_map[i]];
                sse += (uint64_t)diff * diff;
            }

            if (sse < best_sse) {
                best_sse = sse;
                best_val = candidate;
                memcpy(color_idx_map, temp_map, total);
            }
        }

        centroids[c] = best_val;
    }
}

void search_palette_luma(PictureControlSet* pcs, ModeDecisionContext* ctx, PaletteInfo* palette_cand,
                         uint8_t* palette_size_array, uint32_t* tot_palette_cands) {
    int  colors;
    bool is16bit = ctx->hbd_md > 0;

    EbPictureBufferDesc* src_pic = is16bit ? pcs->input_frame16bit : pcs->ppcs->enhanced_pic;

    const int src_stride        = src_pic->y_stride;
    unsigned  palette_bit_depth = is16bit ? EB_TEN_BIT : EB_EIGHT_BIT;

    const uint8_t* const src = src_pic->y_buffer +
        (((ctx->blk_org_x) + (ctx->blk_org_y) * src_pic->y_stride) << is16bit);

    int block_width, block_height, rows, cols;
    svt_aom_get_block_dimensions(
        ctx->blk_geom->bsize, 0, ctx->blk_ptr->av1xd, &block_width, &block_height, &rows, &cols);

    int      count_buf[1 << 12];
    unsigned bit_depth = pcs->ppcs->scs->encoder_bit_depth;

    if (is16bit) {
        colors = svt_av1_count_colors_highbd((uint16_t*)src, src_stride, rows, cols, bit_depth, count_buf);
    } else {
        colors = svt_av1_count_colors(src, src_stride, rows, cols, count_buf);
    }

    if (colors <= 1 || colors > 64) {
        return;
    }

    const int max_n = AOMMIN(colors, PALETTE_MAX_SIZE);
    const int min_n = PALETTE_MIN_SIZE;

    int* data = ctx->palette_buffer->kmeans_data_buf;
    int  centroids[PALETTE_MAX_SIZE];
    int  lb, ub;

    lb = ub = is16bit ? ((uint16_t*)src)[0] : ((uint8_t*)src)[0];

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int val = is16bit ? ((uint16_t*)src)[r * src_stride + c] : ((uint8_t*)src)[r * src_stride + c];

            data[r * cols + c] = val;
            if (val < lb) {
                lb = val;
            }
            if (val > ub) {
                ub = val;
            }
        }
    }
#if !OPT_SC_STILL_IMAGE
    uint16_t  color_cache[2 * PALETTE_MAX_SIZE];
    const int n_cache = svt_get_palette_cache_y(ctx->blk_ptr->av1xd, color_cache);
#endif
#if !OPT_SC_STILL_IMAGE
    //  Extract dominant colors
    int top_colors[PALETTE_MAX_SIZE] = {0};
    for (int i = 0; i < max_n; ++i) {
        int max_count = 0;
        for (int j = 0; j < (1 << palette_bit_depth); ++j) {
            if (count_buf[j] > max_count) {
                max_count     = count_buf[j];
                top_colors[i] = j;
            }
        }
        count_buf[top_colors[i]] = 0;
    }
#endif
    //  Dominant-color search
    uint8_t dominant_color_step = pcs->ppcs->palette_ctrls.dominant_color_step;
    if (dominant_color_step != (uint8_t)~0) {
#if OPT_SC_STILL_IMAGE
        //  Extract dominant colors
        int top_colors[PALETTE_MAX_SIZE] = {0};
        for (int i = 0; i < max_n; ++i) {
            int max_count = 0;
            for (int j = 0; j < (1 << palette_bit_depth); ++j) {
                if (count_buf[j] > max_count) {
                    max_count     = count_buf[j];
                    top_colors[i] = j;
                }
            }
            count_buf[top_colors[i]] = 0;
        }
#endif
        for (int n = max_n; n >= min_n; n -= dominant_color_step) {
            for (int i = 0; i < n; ++i) {
                centroids[i] = top_colors[i];
            }

            uint32_t cand_index = *tot_palette_cands;

            palette_rd_y(&palette_cand[cand_index],
                         &palette_size_array[cand_index],
                         ctx,
                         false,
                         ctx->blk_geom->bsize,
                         data,
                         centroids,
                         n,
#if OPT_SC_STILL_IMAGE
                         NULL,
                         0,
#else
                         color_cache,
                         n_cache,
#endif
                         palette_bit_depth);

            if (palette_size_array[cand_index] >= PALETTE_MIN_SIZE) {
                (*tot_palette_cands)++;
            }
        }
    }

    // K-means search
    uint8_t kmean_color_step = pcs->ppcs->palette_ctrls.kmean_color_step;
    if (kmean_color_step != (uint8_t)~0) {
#if OPT_SC_STILL_IMAGE
        uint16_t  color_cache[2 * PALETTE_MAX_SIZE];
        const int n_cache = svt_get_palette_cache_y(ctx->blk_ptr->av1xd, color_cache);
#endif
        for (int n = max_n; n >= min_n; n -= kmean_color_step) {
            if (colors == PALETTE_MIN_SIZE) {
                centroids[0] = lb;
                centroids[1] = ub;
            } else {
                for (int i = 0; i < n; ++i) {
                    centroids[i] = lb + (2 * i + 1) * (ub - lb) / n / 2;
                }
#if OPT_SC_STILL_IMAGE
                av1_k_means(data,
                            centroids,
                            palette_cand[*tot_palette_cands].color_idx_map,
                            rows * cols,
                            n,
                            1,
                            pcs->ppcs->palette_ctrls.k_means_max_itr);
#else
                av1_k_means(data, centroids, palette_cand[*tot_palette_cands].color_idx_map, rows * cols, n, 1, 50);
#endif
            }
            if (pcs->ppcs->palette_ctrls.centroid_refinement) {
                cache_based_centroid_refinement(data,
                                                rows,
                                                cols,
                                                n,
                                                centroids,
                                                palette_cand[*tot_palette_cands].color_idx_map,
                                                color_cache,
                                                n_cache);
            }
            uint32_t cand_index = *tot_palette_cands;

            palette_rd_y(&palette_cand[cand_index],
                         &palette_size_array[cand_index],
                         ctx,
                         true,
                         ctx->blk_geom->bsize,
                         data,
                         centroids,
                         n,
                         color_cache,
                         n_cache,
                         palette_bit_depth);

            if (palette_size_array[cand_index] >= PALETTE_MIN_SIZE) {
                (*tot_palette_cands)++;
            }
        }
    }
}

typedef AomCdfProb (*MapCdf)[PALETTE_COLOR_INDEX_CONTEXTS][CDF_SIZE(PALETTE_COLORS)];
typedef const int (*ColorCost)[PALETTE_SIZES][PALETTE_COLOR_INDEX_CONTEXTS][PALETTE_COLORS];

typedef struct {
    int       rows;
    int       cols;
    int       n_colors;
    int       plane_width;
    uint8_t*  color_map;
    MapCdf    map_cdf;
    ColorCost color_cost;
} Av1ColorMapParam;

static void get_palette_params(FRAME_CONTEXT* frame_context, EcBlkStruct* blk_ptr, int plane, BlockSize bsize,
                               Av1ColorMapParam* params) {
    const MacroBlockD* const xd = blk_ptr->av1xd;
    params->color_map           = blk_ptr->palette_info->color_idx_map;
    params->map_cdf    = plane ? frame_context->palette_uv_color_index_cdf : frame_context->palette_y_color_index_cdf;
    params->color_cost = NULL;
    params->n_colors   = blk_ptr->palette_size[plane];
    svt_aom_get_block_dimensions(bsize, plane, xd, &params->plane_width, NULL, &params->rows, &params->cols);
}

static void get_color_map_params(FRAME_CONTEXT* frame_context, EcBlkStruct* blk_ptr, int plane, BlockSize bsize,
                                 TxSize tx_size, COLOR_MAP_TYPE type, Av1ColorMapParam* params) {
    (void)tx_size;
    memset(params, 0, sizeof(*params));
    switch (type) {
    case PALETTE_MAP:
        get_palette_params(frame_context, blk_ptr, plane, bsize, params);
        break;
    default:
        assert(0 && "Invalid color map type");
        return;
    }
}

static void get_palette_params_rate(ModeDecisionCandidate* cand, MdRateEstimationContext* rate_table,
                                    BlkStruct* blk_ptr, int plane, BlockSize bsize, Av1ColorMapParam* params) {
    PaletteInfo* palette_info = cand->palette_info;

    const MacroBlockD* const xd = blk_ptr->av1xd;
    params->color_map           = palette_info->color_idx_map;
    params->map_cdf             = NULL;
    params->color_cost          = plane ? NULL : (ColorCost)&rate_table->palette_ycolor_fac_bitss;
    params->n_colors            = cand->palette_size[plane];

    svt_aom_get_block_dimensions(bsize, plane, xd, &params->plane_width, NULL, &params->rows, &params->cols);
}

static void get_color_map_params_rate(ModeDecisionCandidate* cand, MdRateEstimationContext* rate_table,
                                      /*const MACROBLOCK *const x*/ BlkStruct* blk_ptr, int plane, BlockSize bsize,
                                      COLOR_MAP_TYPE type, Av1ColorMapParam* params) {
    memset(params, 0, sizeof(*params));
    switch (type) {
    case PALETTE_MAP:
        get_palette_params_rate(cand, rate_table, blk_ptr, plane, bsize, params);
        break;
    default:
        assert(0 && "Invalid color map type");
        return;
    }
}

#define SWAP(i, j)                               \
    do {                                         \
        const uint8_t tmp_score = score_rank[i]; \
        const uint8_t tmp_color = color_rank[i]; \
        score_rank[i]           = score_rank[j]; \
        color_rank[i]           = color_rank[j]; \
        score_rank[j]           = tmp_score;     \
        color_rank[j]           = tmp_color;     \
    } while (0)

#define MAX_COLOR_CONTEXT_HASH 8
// Negative values are invalid
int svt_aom_palette_color_index_context_lookup[MAX_COLOR_CONTEXT_HASH + 1] = {-1, -1, 0, -1, -1, 4, 3, 2, 1};
#define NUM_PALETTE_NEIGHBORS 3 // left, top-left and top.
#define INVALID_COLOR_IDX (UINT8_MAX)

static inline int av1_fast_palette_color_index_context_on_edge(const uint8_t* color_map, int stride, int r, int c,
                                                               int* color_idx) {
    const bool has_left  = (c - 1 >= 0);
    const bool has_above = (r - 1 >= 0);
    assert(r > 0 || c > 0);
    assert(has_above ^ has_left);
    assert(color_idx);
    (void)has_left;

    const uint8_t color_neighbor = has_above ? color_map[(r - 1) * stride + (c - 0)]
                                             : color_map[(r - 0) * stride + (c - 1)];
    // If the neighbor color has higher index than current color index, then we
    // move up by 1.
    const uint8_t current_color = *color_idx = color_map[r * stride + c];
    if (color_neighbor > current_color) {
        (*color_idx)++;
    } else if (color_neighbor == current_color) {
        *color_idx = 0;
    }

    // Get hash value of context.
    // The non-diagonal neighbors get a weight of 2.
    const uint8_t color_score          = 2;
    const uint8_t hash_multiplier      = 1;
    const uint8_t color_index_ctx_hash = color_score * hash_multiplier;

    // Lookup context from hash.
    const int color_index_ctx = svt_aom_palette_color_index_context_lookup[color_index_ctx_hash];
    assert(color_index_ctx == 0);
    (void)color_index_ctx;
    return 0;
}

// A faster version of av1_get_palette_color_index_context used by the encoder
// exploiting the fact that the encoder does not need to maintain a color order.
static inline int av1_fast_palette_color_index_context(const uint8_t* color_map, int stride, int r, int c,
                                                       int* color_idx) {
    assert(r > 0 || c > 0);

    const bool has_above = (r - 1 >= 0);
    const bool has_left  = (c - 1 >= 0);
    assert(has_above || has_left);
    if (has_above ^ has_left) {
        return av1_fast_palette_color_index_context_on_edge(color_map, stride, r, c, color_idx);
    }

    // This goes in the order of left, top, and top-left. This has the advantage
    // that unless anything here are not distinct or invalid, this will already
    // be in sorted order. Furthermore, if either of the first two is
    // invalid, we know the last one is also invalid.
    uint8_t color_neighbors[NUM_PALETTE_NEIGHBORS];
    color_neighbors[0] = color_map[(r - 0) * stride + (c - 1)];
    color_neighbors[1] = color_map[(r - 1) * stride + (c - 0)];
    color_neighbors[2] = color_map[(r - 1) * stride + (c - 1)];

    // Aggregate duplicated values.
    // Since our array is so small, using a couple if statements is faster
    uint8_t scores[NUM_PALETTE_NEIGHBORS] = {2, 2, 1};
    uint8_t num_invalid_colors            = 0;
    if (color_neighbors[0] == color_neighbors[1]) {
        scores[0] += scores[1];
        color_neighbors[1] = INVALID_COLOR_IDX;
        num_invalid_colors += 1;

        if (color_neighbors[0] == color_neighbors[2]) {
            scores[0] += scores[2];
            num_invalid_colors += 1;
        }
    } else if (color_neighbors[0] == color_neighbors[2]) {
        scores[0] += scores[2];
        num_invalid_colors += 1;
    } else if (color_neighbors[1] == color_neighbors[2]) {
        scores[1] += scores[2];
        num_invalid_colors += 1;
    }

    const uint8_t num_valid_colors = NUM_PALETTE_NEIGHBORS - num_invalid_colors;

    uint8_t* color_rank = color_neighbors;
    uint8_t* score_rank = scores;

    // Sort everything
    if (num_valid_colors > 1) {
        if (color_neighbors[1] == INVALID_COLOR_IDX) {
            scores[1]          = scores[2];
            color_neighbors[1] = color_neighbors[2];
        }

        // We need to swap the first two elements if they have the same score but
        // the color indices are not in the right order
        if (score_rank[0] < score_rank[1] || (score_rank[0] == score_rank[1] && color_rank[0] > color_rank[1])) {
            SWAP(0, 1);
        }
        if (num_valid_colors > 2) {
            if (score_rank[0] < score_rank[2]) {
                SWAP(0, 2);
            }
            if (score_rank[1] < score_rank[2]) {
                SWAP(1, 2);
            }
        }
    }

    // If any of the neighbor colors has higher index than current color index,
    // then we move up by 1 unless the current color is the same as one of the
    // neighbors.
    const uint8_t current_color = *color_idx = color_map[r * stride + c];
    for (int idx = 0; idx < num_valid_colors; idx++) {
        if (color_rank[idx] > current_color) {
            (*color_idx)++;
        } else if (color_rank[idx] == current_color) {
            *color_idx = idx;
            break;
        }
    }

    // Get hash value of context.
    uint8_t              color_index_ctx_hash                    = 0;
    static const uint8_t hash_multipliers[NUM_PALETTE_NEIGHBORS] = {1, 2, 2};
    for (int idx = 0; idx < num_valid_colors; ++idx) {
        color_index_ctx_hash += score_rank[idx] * hash_multipliers[idx];
    }
    assert(color_index_ctx_hash > 0);
    assert(color_index_ctx_hash <= MAX_COLOR_CONTEXT_HASH);

    // Lookup context from hash.
    const int color_index_ctx = 9 - color_index_ctx_hash;
    assert(color_index_ctx == svt_aom_palette_color_index_context_lookup[color_index_ctx_hash]);
    assert(color_index_ctx >= 0);
    assert(color_index_ctx < PALETTE_COLOR_INDEX_CONTEXTS);
    return color_index_ctx;
}

#undef INVALID_COLOR_IDX
#undef SWAP

static int cost_and_tokenize_map(Av1ColorMapParam* param, TOKENEXTRA** t, int plane, int calc_rate,
                                 int allow_update_cdf, MapCdf map_pb_cdf) {
    const uint8_t* const color_map         = param->color_map;
    MapCdf               map_cdf           = param->map_cdf;
    ColorCost            color_cost        = param->color_cost;
    const int            plane_block_width = param->plane_width;
    const int            rows              = param->rows;
    const int            cols              = param->cols;
    const int            n                 = param->n_colors;
    const int            palette_size_idx  = n - PALETTE_MIN_SIZE;
    int                  this_rate         = 0;

    (void)plane;

    for (int k = 1; k < rows + cols - 1; ++k) {
        for (int j = AOMMIN(k, cols - 1); j >= AOMMAX(0, k - rows + 1); --j) {
            int       i = k - j;
            int       color_new_idx;
            const int color_ctx = av1_fast_palette_color_index_context(
                color_map, plane_block_width, i, j, &color_new_idx);
            assert(color_new_idx >= 0 && color_new_idx < n);
            if (calc_rate) {
                this_rate += (*color_cost)[palette_size_idx][color_ctx][color_new_idx];
            } else {
                (*t)->token         = color_new_idx;
                (*t)->color_map_cdf = map_pb_cdf[palette_size_idx][color_ctx];
                ++(*t);
                if (allow_update_cdf) {
                    update_cdf(map_cdf[palette_size_idx][color_ctx], color_new_idx, n);
                }
            }
        }
    }
    return this_rate;
}

void svt_av1_tokenize_color_map(FRAME_CONTEXT* frame_context, EcBlkStruct* blk_ptr, int plane, TOKENEXTRA** t,
                                BlockSize bsize, TxSize tx_size, COLOR_MAP_TYPE type, int allow_update_cdf) {
    assert(plane == 0 || plane == 1);
    Av1ColorMapParam color_map_params;
    get_color_map_params(frame_context, blk_ptr, plane, bsize, tx_size, type, &color_map_params);
    // The first color index does not use context or entropy.
    (*t)->token         = color_map_params.color_map[0];
    (*t)->color_map_cdf = NULL;
    ++(*t);
    MapCdf map_pb_cdf = plane ? frame_context->palette_uv_color_index_cdf : frame_context->palette_y_color_index_cdf;
    cost_and_tokenize_map(&color_map_params, t, plane, 0, allow_update_cdf, map_pb_cdf);
}

int svt_av1_cost_color_map(ModeDecisionCandidate* cand, MdRateEstimationContext* rate_table, BlkStruct* blk_ptr,
                           int plane, BlockSize bsize, COLOR_MAP_TYPE type) {
    assert(plane == 0 || plane == 1);
    Av1ColorMapParam color_map_params;
    get_color_map_params_rate(cand, rate_table, blk_ptr, plane, bsize, type, &color_map_params);
    MapCdf map_pb_cdf = NULL;
    return cost_and_tokenize_map(&color_map_params, NULL, plane, 1, 0, map_pb_cdf);
}
