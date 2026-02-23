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
    double (*to_linear)(double);
    double (*from_linear)(double);
    double mid_tone;
} TransferFunction;

typedef struct {
    uint32_t         width;
    uint32_t         height;
    uint32_t         iso_setting;
    uint8_t          chroma_setting;
    TransferFunction transfer_function;
} PhotonNoiseArgs;

// Transfer function implementations
static double maxf(double a, double b) { return a > b ? a : b; }
static double minf(double a, double b) { return a < b ? a : b; }

static double gamma22_to_linear(double g) { return pow(g, 2.2); }
static double gamma22_from_linear(double l) { return pow(l, 1.0 / 2.2); }

static double gamma28_to_linear(double g) { return pow(g, 2.8); }
static double gamma28_from_linear(double l) { return pow(l, 1.0 / 2.8); }

static double srgb_to_linear(double srgb) { return srgb <= 0.04045 ? srgb / 12.92 : pow((srgb + 0.055) / 1.055, 2.4); }
static double srgb_from_linear(double linear) {
    return linear <= 0.0031308 ? 12.92 * linear : 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
}

static const double kPqM1 = 2610.0 / 16384;
static const double kPqM2 = 128 * 2523.0 / 4096;
static const double kPqC1 = 3424.0 / 4096;
static const double kPqC2 = 32 * 2413.0 / 4096;
static const double kPqC3 = 32 * 2392.0 / 4096;

static double pq_to_linear(double pq) {
    double pq_pow_inv_m2 = pow(pq, 1.0 / kPqM2);
    return pow(maxf(0.0, pq_pow_inv_m2 - kPqC1) / (kPqC2 - kPqC3 * pq_pow_inv_m2), 1.0 / kPqM1);
}
static double pq_from_linear(double linear) {
    double linear_pow_m1 = pow(linear, kPqM1);
    return pow((kPqC1 + kPqC2 * linear_pow_m1) / (1.0 + kPqC3 * linear_pow_m1), kPqM2);
}

static const double kHlgA = 0.17883277;
static const double kHlgB = 0.28466892;
static const double kHlgC = 0.55991073;

static double hlg_to_linear(double hlg) {
    double linear = hlg <= 0.5 ? hlg * hlg / 3.0 : (exp((hlg - kHlgC) / kHlgA) + kHlgB) / 12.0;
    return pow(linear, 1.2);
}
static double hlg_from_linear(double linear) {
    linear = pow(linear, 1.0 / 1.2);
    return linear <= 1.0 / 12.0 ? sqrt(3.0 * linear) : kHlgA * log(12.0 * linear - kHlgB) + kHlgC;
}

static double bt601_from_linear(double L) {
    if (L < 0.018)
        return 4.5 * L;
    else
        return 1.099 * pow(L, 0.45) - 0.099;
}

static double bt601_to_linear(double E) {
    if (E < 0.08145)
        return E / 4.5;
    else
        return pow((E + 0.099) / 1.099, 1.0 / 0.45);
}

static double smpte240m_to_linear(double x) { return pow(x, 2.222); }
static double smpte240m_from_linear(double x) { return pow(x, 1.0 / 2.222); }

static double log100_to_linear(double x) {
    double L = x * (log10(101.0) - 0.0) + 0.0;
    return pow(10.0, L) - 1.0;
}
static double log100_from_linear(double y) { return (log10(y + 1.0) - 0.0) / (log10(101.0) - 0.0); }

static double log100_sqrt10_to_linear(double x) {
    double Lmax = log10(100.0 * 3.1622776601683795 + 1.0);
    double L    = x * (Lmax - 0.0);
    return pow(10.0, L) - 1.0;
}
static double log100_sqrt10_from_linear(double y) {
    double Lmax = log10(100.0 * 3.1622776601683795 + 1.0);
    return (log10(y + 1.0) - 0.0) / (Lmax - 0.0);
}

static double iec61966_to_linear(double x) {
    const double a = 0.1555, b = 0.2847;
    return pow((x + a) / (1.0 + b), 2.6);
}
static double iec61966_from_linear(double y) {
    const double a = 0.1555, b = 0.2847;
    return (pow(y, 1.0 / 2.6) * (1.0 + b)) - a;
}

static double bt1361_to_linear(double x) {
    if (x <= 0.08)
        return x / 4.5;
    return pow((x + 0.099) / 1.099, 2.2);
}
static double bt1361_from_linear(double y) {
    if (y <= (0.08 / 4.5))
        return y * 4.5;
    return (1.099 * pow(y, 1.0 / 2.2) - 0.099);
}

static double smpte428_to_linear(double x) {
    const double a = 0.002, b = 0.56, R = 95.0 / 1023.0;
    return pow(10.0, (x - b) * R) - a;
}
static double smpte428_from_linear(double y) {
    const double a = 0.002, b = 0.56, R = 95.0 / 1023.0;
    return (log10(y + a) / R) + b;
}

static double identity_to_linear(double x) { return x; }
static double identity_from_linear(double x) { return x; }

static EbColorRange find_color_range(EbSvtAv1EncConfiguration *config) {
    // If the color range is explicitly provided, use it.
    if (config->color_range_provided) {
        return config->color_range;
    }
    return config->avif ? EB_CR_FULL_RANGE : EB_CR_STUDIO_RANGE;
}

static TransferFunction find_transfer_function(EbTransferCharacteristics tc) {
    TransferFunction tf = {NULL, NULL, 0.18};

    switch (tc) {
    // Rec.709 and Rec.2020 define the same nonlinear transfer function for gamma correction as Rec.601,
    // with expection of 12 bit BT.2020 parameters being more precise.
    // Thus, Rec.601 transfer function can be reused here.
    case EB_CICP_TC_BT_709:
    case EB_CICP_TC_BT_601:
    case EB_CICP_TC_BT_2020_10_BIT:
    case EB_CICP_TC_BT_2020_12_BIT:
        tf.to_linear   = bt601_to_linear;
        tf.from_linear = bt601_from_linear;
        break;

    case EB_CICP_TC_UNSPECIFIED:
        tf.to_linear   = srgb_to_linear;
        tf.from_linear = srgb_from_linear;
        break;

    case EB_CICP_TC_BT_470_M:
        tf.to_linear   = gamma22_to_linear;
        tf.from_linear = gamma22_from_linear;
        break;

    case EB_CICP_TC_BT_470_B_G:
        tf.to_linear   = gamma28_to_linear;
        tf.from_linear = gamma28_from_linear;
        break;

    case EB_CICP_TC_SMPTE_240:
        tf.to_linear   = smpte240m_to_linear;
        tf.from_linear = smpte240m_from_linear;
        break;

    case EB_CICP_TC_LINEAR:
        tf.to_linear   = identity_to_linear;
        tf.from_linear = identity_from_linear;
        tf.mid_tone    = 0.50;
        break;

    case EB_CICP_TC_LOG_100:
        tf.to_linear   = log100_to_linear;
        tf.from_linear = log100_from_linear;
        break;

    case EB_CICP_TC_LOG_100_SQRT10:
        tf.to_linear   = log100_sqrt10_to_linear;
        tf.from_linear = log100_sqrt10_from_linear;
        break;

    case EB_CICP_TC_IEC_61966:
        tf.to_linear   = iec61966_to_linear;
        tf.from_linear = iec61966_from_linear;
        break;

    case EB_CICP_TC_BT_1361:
        tf.to_linear   = bt1361_to_linear;
        tf.from_linear = bt1361_from_linear;
        break;

    case EB_CICP_TC_SRGB:
        tf.to_linear   = srgb_to_linear;
        tf.from_linear = srgb_from_linear;
        break;

    case EB_CICP_TC_SMPTE_2084:
        tf.to_linear   = pq_to_linear;
        tf.from_linear = pq_from_linear;
        tf.mid_tone    = 26.0 / 10000.0;
        break;

    case EB_CICP_TC_SMPTE_428:
        tf.to_linear   = smpte428_to_linear;
        tf.from_linear = smpte428_from_linear;
        break;

    case EB_CICP_TC_HLG:
        tf.to_linear   = hlg_to_linear;
        tf.from_linear = hlg_from_linear;
        tf.mid_tone    = 26.0 / 1000.0;
        break;

    default:
        // RESERVED or unknown: fallback to sRGB
        SVT_WARN("Warning: unimplemented transfer function %d, defaulting to sRGB\n", tc);
        tf.to_linear   = srgb_to_linear;
        tf.from_linear = srgb_from_linear;
        break;
    }

    return tf;
}

static void svt_av1_generate_photon_noise(const PhotonNoiseArgs *photon_noise_args, EbSvtAv1EncConfiguration *cfg,
                                          const EbColorRange color_range) {
    AomFilmGrain *film_grain;
    film_grain = (AomFilmGrain *)calloc(1, sizeof(AomFilmGrain));
    // Assumes a daylight-like spectrum.
    // https://www.strollswithmydog.com/effective-quantum-efficiency-of-sensor/#:~:text=11%2C260%20photons/um%5E2/lx-s
    static const double kPhotonsPerLxSPerUm2 = 11260.0;

    // Order of magnitude for cameras in the 2010-2020 decade, taking the CFA into
    // account.
    static const double kEffectiveQuantumEfficiency = 0.20;

    // Also reasonable values for current cameras. The read noise is typically
    // higher than this at low ISO settings but it matters less there.
    static const double kPhotoResponseNonUniformity = 0.005;
    static const double kInputReferredReadNoise     = 1.5;
    /* OK */

    // Focal plane exposure for a mid-tone (typically a 18% reflectance card), in
    // lx·s.
    const double mid_tone_exposure = 10.0 / photon_noise_args->iso_setting;

    // In microns. Assumes a 35mm sensor (36mm × 24mm).
    const double pixel_area_um2 = (36000.0 * 24000.0) / (photon_noise_args->width * photon_noise_args->height);

    const double mid_tone_electrons_per_pixel = kEffectiveQuantumEfficiency * kPhotonsPerLxSPerUm2 * mid_tone_exposure *
        pixel_area_um2;
    const double max_electrons_per_pixel = mid_tone_electrons_per_pixel / photon_noise_args->transfer_function.mid_tone;

    const uint32_t max_value   = (color_range == EB_CR_STUDIO_RANGE) ? 235 : 255;
    const uint32_t min_value   = (color_range == EB_CR_STUDIO_RANGE) ? 16 : 0;
    const uint32_t range       = max_value - min_value;
    const uint32_t ramp_offset = 3;

    film_grain->num_y_points = 14;

    const int32_t min_edge = 0;
    const int32_t max_edge = film_grain->num_y_points - 1;

    for (int32_t i = 0; i < film_grain->num_y_points; ++i) {
        double x = (ramp_offset + (range - 2 * ramp_offset) * ((i - 1) / (film_grain->num_y_points - 3.0))) / range;

        // Applying photon noise "as is" results in unwanted brightening of darkest and darkening of brightest luma values;
        // clamping scaling points to a maximum of 1 at those min and max values prevents that.
        if (i == min_edge)
            x = 0;
        else if (i == max_edge)
            x = 1;

        const double linear              = photon_noise_args->transfer_function.to_linear(x);
        const double electrons_per_pixel = max_electrons_per_pixel * linear;
        // Quadrature sum of the relevant sources of noise, in electrons rms. Photon
        // shot noise is sqrt(electrons) so we can skip the square root and the
        // squaring.
        // https://en.wikipedia.org/wiki/Addition_in_quadrature
        // https://doi.org/10.1117/3.725073
        const double noise_in_electrons = sqrt(
            kInputReferredReadNoise * kInputReferredReadNoise + electrons_per_pixel +
            (kPhotoResponseNonUniformity * kPhotoResponseNonUniformity * electrons_per_pixel * electrons_per_pixel));
        const double linear_noise       = noise_in_electrons / max_electrons_per_pixel;
        const double linear_range_start = maxf(0.0, linear - 2.0 * linear_noise);
        const double linear_range_end   = minf(1.0, linear + 2.0 * linear_noise);
        const double tf_slope           = (photon_noise_args->transfer_function.from_linear(linear_range_end) -
                                 photon_noise_args->transfer_function.from_linear(linear_range_start)) /
            (linear_range_end - linear_range_start);
        double encoded_noise = linear_noise * tf_slope;

        x             = round(min_value + (range * x));
        encoded_noise = minf((double)range, round(range * 7.88 * encoded_noise));

        film_grain->scaling_points_y[i][0] = (int32_t)x;

        if (i == min_edge || i == max_edge) {
            film_grain->scaling_points_y[i][1] = (encoded_noise >= 1) ? 1 : 0;
            continue;
        }

        film_grain->scaling_points_y[i][1] = (int32_t)encoded_noise;
    }

    film_grain->apply_grain       = 1;
    film_grain->ignore_ref        = 1;
    film_grain->update_parameters = 1;
    film_grain->num_cb_points     = 0;
    film_grain->num_cr_points     = 0;
    film_grain->scaling_shift     = 8;
    film_grain->ar_coeff_lag      = 0;
    memset(film_grain->ar_coeffs_y, 0, sizeof(film_grain->ar_coeffs_y));
    memset(film_grain->ar_coeffs_cb, 0, sizeof(film_grain->ar_coeffs_cb));
    memset(film_grain->ar_coeffs_cr, 0, sizeof(film_grain->ar_coeffs_cr));
    film_grain->ar_coeff_shift           = 6;
    film_grain->cb_mult                  = 0;
    film_grain->cb_luma_mult             = 0;
    film_grain->cb_offset                = 0;
    film_grain->cr_mult                  = 0;
    film_grain->cr_luma_mult             = 0;
    film_grain->cr_offset                = 0;
    film_grain->overlap_flag             = 1;
    film_grain->grain_scale_shift        = 0;
    film_grain->chroma_scaling_from_luma = photon_noise_args->chroma_setting;
    film_grain->clip_to_restricted_range = (color_range == EB_CR_STUDIO_RANGE) ? 1 : 0;
    cfg->fgs_table                       = film_grain;
}

EbErrorType svt_av1_generate_photon_noise_table(EbSvtAv1EncConfiguration *config) {
    EbColorRange    color_range = find_color_range(config);
    PhotonNoiseArgs args        = {.width             = config->source_width,
                                   .height            = config->source_height,
                                   .iso_setting       = config->photon_noise_iso,
                                   .chroma_setting    = config->enable_photon_noise_chroma,
                                   .transfer_function = find_transfer_function(config->transfer_characteristics)};

    svt_av1_generate_photon_noise(&args, config, color_range);

    return EB_ErrorNone;
}
