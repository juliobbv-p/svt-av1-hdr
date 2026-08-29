/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <nmmintrin.h>
#include <string.h>

#include "aom_dsp_rtcd.h"

/* CRC-32C (Castagnoli) using the SSE4.2 crc32 instruction. Bit-exact with
 * svt_av1_get_crc32c_value_c. */
uint32_t svt_av1_get_crc32c_value_sse4_2(const uint8_t* buf, size_t len) {
    const uint8_t* next = buf;
    uint32_t       crc  = 0xFFFFFFFF;

    while (len >= 8) {
        uint64_t v;
        memcpy(&v, next, sizeof(v));
        crc = (uint32_t)_mm_crc32_u64(crc, v);
        next += 8;
        len -= 8;
    }
    if (len >= 4) {
        uint32_t v;
        memcpy(&v, next, sizeof(v));
        crc = _mm_crc32_u32(crc, v);
        next += 4;
        len -= 4;
    }
    if (len >= 2) {
        uint16_t v;
        memcpy(&v, next, sizeof(v));
        crc = _mm_crc32_u16(crc, v);
        next += 2;
        len -= 2;
    }
    if (len) {
        crc = _mm_crc32_u8(crc, *next);
    }
    return crc ^ 0xFFFFFFFF;
}
