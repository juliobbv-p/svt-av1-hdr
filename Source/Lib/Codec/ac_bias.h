/*
* Copyright(c) 2024-2025 Psychovisual Experts Group
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include <stdbool.h>
#include <stdint.h>

#include "common_dsp_rtcd.h"

#ifdef __cplusplus
extern "C" {
#endif

uint64_t get_svt_psy_full_dist(const void* s, uint32_t so, uint32_t sp, const void* r, uint32_t ro, uint32_t rp,
                               const uint32_t w, const uint32_t h, const uint8_t is_hbd, const double ac_bias);
uint64_t svt_psy_adjust_rate_light(const int32_t* coeff, uint64_t coeff_bits, const uint32_t bwidth,
                                   const uint32_t bheight, const double ac_bias);
double   get_effective_ac_bias(const double ac_bias, const bool is_islice, const uint8_t temporal_layer_index);

#ifdef __cplusplus
}
#endif
