/*
* Copyright (c) 2021, Alliance for Open Media. All rights reserved.
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#ifndef SVT_QM_PSNR_H
#define SVT_QM_PSNR_H

#include "definitions.h"
#include "inv_transforms.h"

// QM-PSNR metric, round each weighted square modulated by the quantizer matrix
static INLINE int64_t svt_qm_coeff_dist(int32_t coeff, int32_t recon, int shift, const QmVal* qm, int index) {
    int64_t diff = ((int64_t)coeff - recon) * ((int64_t)1 << shift);
    if (!qm) {
        return diff * diff;
    }
    diff *= qm[index];
    return (diff * diff + ((int64_t)1 << (2 * AOM_QM_BITS - 1))) >> (2 * AOM_QM_BITS);
}

#endif // SVT_QM_PSNR_H
