/*
* Copyright(c) 2024 Gianni Rosato
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t uint16x2_t;
typedef uint64_t uint32x2_t;

uint64_t svt_psy_distortion(const uint8_t* input, const uint32_t input_stride,
                            const uint8_t* recon, const uint32_t recon_stride,
                            const uint32_t width, const uint32_t height);
uint64_t svt_psy_distortion_hbd(uint16_t* input, const uint32_t input_stride,
                                uint16_t* recon, const uint32_t recon_stride,
                                const uint32_t width, const uint32_t height);
uint64_t get_svt_psy_full_dist(const void* s, uint32_t so, uint32_t sp,
                               const void* r, uint32_t ro, uint32_t rp,
                               const uint32_t w, const uint32_t h, const uint8_t is_hbd,
                               const double ac_bias);
double get_effective_ac_bias(const double ac_bias, const bool is_islice, const uint8_t temporal_layer_index);

#ifdef __cplusplus
}
#endif
