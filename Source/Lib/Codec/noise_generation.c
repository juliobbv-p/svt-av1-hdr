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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "definitions.h"
#include "sequence_control_set.h"
#include "svt_log.h"

typedef struct {
    int32_t lag;
    int32_t shift;

    int32_t cY[24];
    int32_t cCb[25];
    int32_t cCr[25];
} NoiseCoeffTable;

/* AR coefficients extracted from a grain‑table file generated from a sample noise
 * clip encoded with SVT‑AV1 and film‑grain enabled. Each table corresponds to a
 * grain‑size value; higher indices produce larger‑looking grain.
 */
static const NoiseCoeffTable coeffs[] = {
    {0, 6, {0}, {0}, {0}},
    {3,
     8,
     {7, -2, -15, -19, -9, -2, 1, -5, -15, -15, -30, -10, -16, -4, -14, -11, -16, 5, -14, -10, -18, -16, -24, 10},
     {-10, -24, -35, -15,  -28, -9,  -6, -13, -42, -51, -95, -23, -18,
      -7,  -48, -45, -117, -53, -95, -9, 16,  1,   -76, -40, 21},
     {-8, -23, -28, -24,  -17, -22, 2,   -5,  -32, -58, -80, -45, -13,
      -1, -35, -45, -104, -50, -77, -12, -14, 6,   -73, -46, 14}},
    {3,
     8,
     {5, -3, -15, -17, -12, -4, 2, -4, -17, -11, -32, -11, -14, -5, -15, -7, -18, 8, -17, -9, -13, -13, -24, 9},
     {-23, -31, -44, -29,  -32, 1,   -9,  -1,  -43, -41, -96, -32, -16,
      1,   -30, -45, -112, -57, -98, -20, -11, -7,  -72, -41, 38},
     {-19, -22, -19, -23, -42, -14, -8, -22, -55, -38, -72, -27, -5,
      -17, -10, -41, -98, -46, -64, -9, -7,  2,   -63, -39, -24}},
    {3,
     8,
     {0, -13, -20, -11, -14, -6, -2, -15, -5, -6, -26, -5, -6, -6, -17, -1, -18, 25, -14, -4, -17, -10, -21, 24},
     {-23, -18, -37, -19,  -9,  -16, -5,  -16, -52, -34, -110, -26, -7,
      -16, -16, -24, -111, -10, -76, -28, 22,  15,  -80, -19,  -47},
     {-9, -12, -27, -3, -17, -25, 7, -6, -32, -19, -102, -23, -10, -4, -13, -17, -88, 5, -77, -19, 3, 22, -70, 2, 14}},
    {3,
     8,
     {-8, -8, -7, 2, -4, -10, -1, -13, 8, -4, -22, -11, 11, -7, -3, -3, -12, 42, -5, -7, -8, -1, -26, 43},
     {-20, -4, -27, -17, -17, -15, -5, -7, -50, -23, -87, -38, -8, -19, -21, -9, -113, 23, -58, -39, -6, 4, -80, 34, 4},
     {-15, 8,   -20, -9,   -15, -26, 3,   -9, -34, -28, -79, -54, -1,
      -27, -28, -2,  -100, 29,  -59, -45, -7, 17,  -85, 27,  -19}},
    {3,
     8,
     {-8, -3, 1, 8, 0, -2, -8, -2, 6, -3, -30, -6, 7, -11, 3, 1, -31, 73, -5, -10, 7, 7, -35, 74},
     {-2, -5, -4, 15, -11, -6, -19, -11, -25, -34, -76, -38, 13, -11, -10, -24, -78, 46, -32, -57, -6, 15, -79, 38, -9},
     {-34, -13, -33, -6,   -17, -11, -2,  -13, -33, -27,  -94, -15, -23,
      11,  -23, -12, -103, 58,  -38, -24, -11, 9,   -100, 59,  13}},
    {3,
     8,
     {-4, 4, 2, 3, 4, 7, -1, 5, 6, -7, -23, -16, 2, -6, 1, -6, -12, 65, 12, -12, 6, 0, -28, 75},
     {-8, -3, 0, 5, -29, -2, 6, 3, -41, 2, -79, -27, -3, -3, -15, 5, -77, 74, -50, -28, -19, 20, -92, 85, -26},
     {-6, -12, -22, -18, -4, -10, -26, 5,  -41, -17, -80, -42, -18,
      18, -19, 6,   -67, 69, -17, -48, -6, 18,  -75, 66,  -20}},
    {3,
     8,
     {0, 7, 0, 0, 3, 3, 0, 4, -3, -3, -27, -12, 3, 3, -1, 3, -18, 80, 11, -10, -5, -1, -27, 81},
     {12, 2, -32, 5, 0, -17, 14, -14, -23, -19, -87, -31, 2, -23, 2, -13, -56, 87, -30, -27, -10, -2, -63, 69, -28},
     {-9, 15, -28, 9, -31, -13, -13, 31, -53, 4, -86, 2, -15, 7, -41, 17, -76, 89, -53, -19, -25, 35, -70, 79, -6}},
    {3,
     8,
     {3, 4, -4, 4, -1, 0, 2, 2, -5, -1, -30, -13, 3, 2, -2, 4, -14, 79, 22, -8, -10, 0, -31, 92},
     {-15, 3, -33, 13, -13, -19, -5, 7, -34, 16, -93, -5, -13, -5, -23, -3, -66, 97, -32, -13, -31, 6, -56, 80, 2},
     {-13, -11, -26, -14, -14, -4, -6, 20, -36, 2, -88, -25, -10, -9, -35, -2, -58, 84, -25, -32, -16, 15, -76, 74, 8}},
    {3,
     8,
     {2, 0, -4, 4, 2, -4, 5, 0, -3, -2, -31, -16, 5, -3, -4, 1, -13, 88, 23, -7, -8, 2, -29, 95},
     {-6, -9, -43, 15, -30, -3, 5, -4, -27, 25, -78, -1, -30, -10, -24, 11, -80, 99, -37, 5, -23, 17, -83, 93, -12},
     {-8, 1, -27, 5, -11, -25, -8, -10, -22, 13, -75, -10, -2, -8, -8, 14, -66, 99, -24, -14, -36, 16, -84, 114, 6}},
    {3,
     8,
     {5, -6, -2, 7, 4, -7, 4, -8, 6, -6, -28, -24, 12, -5, -1, 0, -16, 96, 29, -7, -7, 3, -31, 104},
     {8,   -10, -33, 11,   -4,  -7,  -2,  -11, -14, 10,  -72, -22, 11,
      -15, -24, 43,  -104, 123, -23, -22, -18, 12,  -87, 117, -4},
     {6, -15, -12, 21, -11, -20, 2, 10, -35, 26, -88, 0, 2, 3, -30, 42, -96, 125, -17, -17, -14, 37, -88, 110, 5}},
    {3,
     8,
     {3, -3, -2, 8, 4, -2, -1, -9, 7, -4, -31, -21, 7, -2, 0, 0, -19, 100, 30, -10, -2, 9, -37, 112},
     {3, -18, -28, 31, -12, 5, -10, -16, 7, 3, -70, -27, 5, 1, -31, 10, -83, 118, -13, -21, -12, 18, -85, 118, -13},
     {4, -14, -11, 5, -4, -22, 5, 10, -31, 29, -71, -20, 15, -4, -24, 19, -86, 113, -8, -37, 13, 23, -78, 107, -6}},
    {3,
     7,
     {0, 0, -1, 6, 4, -3, 0, -4, 4, -2, -17, -12, 5, -2, 3, -2, -10, 51, 20, -11, 4, 4, -21, 62},
     {-4, -10, -9, 12, 3, -11, -9, -6, 3, 5, -37, -7, 9, -9, -15, 13, -54, 72, -17, -5, -7, 7, -33, 61, 7},
     {-9, -4, -22, 14, -5, -2, -13, 4, -5, 20, -38, -4, 10, -3, -15, 15, -50, 62, -8, -14, -3, 13, -37, 63, 1}},
    {3,
     7,
     {-1, -2, 2, 7, 3, -1, -3, -3, 6, 0, -21, -9, 4, -1, 1, 1, -21, 60, 15, -5, -1, 9, -26, 69},
     {-8, 6, -11, 10, -2, -2, -5, 4, -8, 16, -31, -8, 3, -6, -10, 18, -64, 71, -8, -5, 3, 17, -38, 70, 3},
     {-4, -3, -16, 7, 6, -5, -3, 8, -14, 28, -41, -7, -1, 4, -16, 21, -66, 65, -10, -4, -12, 18, -46, 63, -4}},
};

typedef struct {
    uint32_t     width;
    uint32_t     height;
    uint32_t     str_luma;
    int32_t      str_chroma;
    int8_t       chroma_from_luma;
    int8_t       grain_size;
    EbColorRange color_range;
} NoiseArgs;

static EbColorRange find_color_range(EbSvtAv1EncConfiguration *config) {
    // If the color range is explicitly provided, use it.
    if (config->color_range_provided) {
        return config->color_range;
    }
    return config->avif ? EB_CR_FULL_RANGE : EB_CR_STUDIO_RANGE;
}

static int32_t get_output_noise(int32_t setting_noise, int32_t target_noise, int32_t cutoff) {
    // using cutoff to decide at which setting the noise is allowed to drop to 0
    return (target_noise > 0) ? target_noise : (setting_noise > cutoff) ? 1 : 0;
}

static void set_scaling_points_y(AomFilmGrain *film_grain, const NoiseArgs *noise_args, const uint8_t grain_size) {
    const int32_t num_points    = 6;
    const int32_t range_min     = (noise_args->color_range == EB_CR_STUDIO_RANGE) ? 16 : 0;
    const int32_t range_max     = (noise_args->color_range == EB_CR_STUDIO_RANGE) ? 235 : 255;
    const int32_t range         = range_max - range_min;
    const int32_t noise_setting = noise_args->str_luma;
    // at larger grain size AR coeffs amplify strength look, so with size increase up to 13 we scale noise strength down to ~43%
    const double noise       = (23 - grain_size) * noise_setting / 50.0;
    const double range_ratio = range / 255.0;
    const double ramp_size   = 100 * range_ratio;

    film_grain->num_y_points = num_points;

    // min range point
    film_grain->scaling_points_y[0][0] = range_min;
    film_grain->scaling_points_y[0][1] = 0;

    // lower mid ramp point
    film_grain->scaling_points_y[1][0] = range_min + 6;
    film_grain->scaling_points_y[1][1] = get_output_noise(noise_setting, noise / 4, 1);

    // noise scaling points
    film_grain->scaling_points_y[2][0] = range_min + ramp_size * range_ratio;
    film_grain->scaling_points_y[2][1] = get_output_noise(noise_setting, noise, 0);
    film_grain->scaling_points_y[3][0] = range_max - ramp_size * range_ratio;
    film_grain->scaling_points_y[3][1] = get_output_noise(noise_setting, noise, 0);

    // upper mid ramp point
    film_grain->scaling_points_y[4][0] = range_max - 6;
    film_grain->scaling_points_y[4][1] = get_output_noise(noise_setting, noise / 4, 1);

    // max range point
    film_grain->scaling_points_y[5][0] = range_max;
    film_grain->scaling_points_y[5][1] = 0;
}

static void set_scaling_points_uv(AomFilmGrain *film_grain, const NoiseArgs *noise_args, const uint8_t grain_size) {
    const int32_t noise_setting = (noise_args->str_chroma == -1) ? noise_args->str_luma * 0.6 : noise_args->str_chroma;
    const double  noise         = (23 - grain_size) * noise_setting / 50.0;
    const int32_t range_min     = (noise_args->color_range == EB_CR_STUDIO_RANGE) ? 16 : 0;

    if (noise_args->chroma_from_luma == 0) {
        const int32_t num_points = 4;
        const int32_t midpoint_l = 127;
        const int32_t midpoint_u = 129;
        const int32_t ramp_size  = 4;

        // set multipliers to enable scaling from chroma only
        film_grain->cr_mult = film_grain->cb_mult = 192;
        film_grain->cr_luma_mult = film_grain->cb_luma_mult = 128;
        film_grain->cr_offset = film_grain->cb_offset = 256;

        film_grain->num_cr_points = film_grain->num_cb_points = num_points;

        // lower noise scaling point
        film_grain->scaling_points_cr[0][0] = film_grain->scaling_points_cb[0][0] = midpoint_l - ramp_size;
        film_grain->scaling_points_cr[0][1] = film_grain->scaling_points_cb[0][1] = get_output_noise(
            noise_setting, noise, 0);
        // lower midpoint
        film_grain->scaling_points_cr[1][0] = film_grain->scaling_points_cb[1][0] = midpoint_l;
        film_grain->scaling_points_cr[1][1] = film_grain->scaling_points_cb[1][1] = 0;
        // upper midpoint
        film_grain->scaling_points_cr[2][0] = film_grain->scaling_points_cb[2][0] = midpoint_u;
        film_grain->scaling_points_cr[2][1] = film_grain->scaling_points_cb[2][1] = 0;
        // upper noise scaling point
        film_grain->scaling_points_cr[3][0] = film_grain->scaling_points_cb[3][0] = midpoint_u + ramp_size;
        film_grain->scaling_points_cr[3][1] = film_grain->scaling_points_cb[3][1] = get_output_noise(
            noise_setting, noise, 0);
    } else {
        const int32_t num_points  = 6;
        const int32_t range_max   = (noise_args->color_range == EB_CR_STUDIO_RANGE) ? 235 : 255;
        const int32_t range       = range_max - range_min;
        const double  range_ratio = range / 255.0;
        const double  ramp_size   = 100 * range_ratio;

        // set multipliers to enable scaling from luma only
        film_grain->cr_mult = film_grain->cb_mult = 128;
        film_grain->cr_luma_mult = film_grain->cb_luma_mult = 192;
        film_grain->cr_offset = film_grain->cb_offset = 256;

        film_grain->num_cr_points = film_grain->num_cb_points = num_points;

        // min range point
        film_grain->scaling_points_cr[0][0] = film_grain->scaling_points_cb[0][0] = range_min;
        film_grain->scaling_points_cr[0][1] = film_grain->scaling_points_cb[0][1] = 0;
        // lower mid ramp point
        film_grain->scaling_points_cr[1][0] = film_grain->scaling_points_cb[1][0] = range_min + 6;
        film_grain->scaling_points_cr[1][1] = film_grain->scaling_points_cr[1][1] = get_output_noise(
            noise_setting, noise / 4, 1);
        // noise scaling points
        film_grain->scaling_points_cr[2][0] = film_grain->scaling_points_cb[2][0] = range_min + ramp_size;
        film_grain->scaling_points_cr[2][1] = film_grain->scaling_points_cb[2][1] = get_output_noise(
            noise_setting, noise, 0);
        film_grain->scaling_points_cr[3][0] = film_grain->scaling_points_cb[3][0] = range_max - ramp_size;
        film_grain->scaling_points_cr[3][1] = film_grain->scaling_points_cb[3][1] = get_output_noise(
            noise_setting, noise, 0);
        // upper mid ramp point
        film_grain->scaling_points_cr[4][0] = film_grain->scaling_points_cb[4][0] = range_max - 6;
        film_grain->scaling_points_cr[4][1] = film_grain->scaling_points_cb[4][1] = get_output_noise(
            noise_setting, noise / 4, 1);
        // max range point
        film_grain->scaling_points_cr[5][0] = film_grain->scaling_points_cb[5][0] = range_max;
        film_grain->scaling_points_cr[5][1] = film_grain->scaling_points_cb[5][1] = 0;
    }
}

static uint8_t get_grain_size(const NoiseArgs *noise_args) {
    if (noise_args->grain_size != -1) {
        return noise_args->grain_size;
    }
    const int32_t width      = noise_args->width;
    const int32_t height     = noise_args->height;
    const int32_t large_side = (width > height) ? width : height;

    if (large_side <= 1280)
        return 0;
    if (large_side >= 3840)
        return 13;

    // arbitrary function that calculates the grain size based on input's largest side,
    // where ~720p and lower gives 0, ~1080p gives 2, and ~2160p and above gives 13
    const double p          = log(13.0 / 2.0) / log(4.0);
    double       normalized = (large_side - 1280.0) / 640.0;
    return (int32_t)(2.0 * pow(normalized, p));
}

static void svt_av1_generate_noise(const NoiseArgs *noise_args, EbSvtAv1EncConfiguration *cfg) {
    AomFilmGrain *film_grain;
    film_grain                 = (AomFilmGrain *)calloc(1, sizeof(AomFilmGrain));
    const int32_t noise_chroma = noise_args->str_chroma;
    const int32_t grain_size   = get_grain_size(noise_args);
    set_scaling_points_y(film_grain, noise_args, grain_size);
    if (noise_chroma == 0) {
        film_grain->num_cr_points = film_grain->num_cb_points = 0;
        film_grain->cr_mult = film_grain->cb_mult = 0;
        film_grain->cr_luma_mult = film_grain->cb_luma_mult = 0;
    } else {
        set_scaling_points_uv(film_grain, noise_args, grain_size);
    }
    film_grain->apply_grain       = 1;
    film_grain->ignore_ref        = 0;
    film_grain->update_parameters = 1;
    film_grain->scaling_shift     = 8;
    film_grain->ar_coeff_lag      = coeffs[grain_size].lag;
    memcpy(film_grain->ar_coeffs_y, coeffs[grain_size].cY, sizeof(film_grain->ar_coeffs_y));
    memcpy(film_grain->ar_coeffs_cb, coeffs[grain_size].cCb, sizeof(film_grain->ar_coeffs_cb));
    memcpy(film_grain->ar_coeffs_cr, coeffs[grain_size].cCr, sizeof(film_grain->ar_coeffs_cr));
    film_grain->ar_coeff_shift           = coeffs[grain_size].shift;
    film_grain->overlap_flag             = 1;
    film_grain->grain_scale_shift        = 0;
    film_grain->chroma_scaling_from_luma = 0;
    film_grain->clip_to_restricted_range = (noise_args->color_range == EB_CR_STUDIO_RANGE) ? 1 : 0;
    cfg->fgs_table                       = film_grain;
}

EbErrorType svt_av1_generate_noise_table(EbSvtAv1EncConfiguration *config) {
    const NoiseArgs args = {.width            = config->source_width,
                            .height           = config->source_height,
                            .str_luma         = config->noise_strength,
                            .str_chroma       = config->noise_strength_chroma,
                            .chroma_from_luma = config->noise_chroma_from_luma,
                            .grain_size       = config->noise_size,
                            .color_range      = find_color_range(config)};

    svt_av1_generate_noise(&args, config);

    return EB_ErrorNone;
}
