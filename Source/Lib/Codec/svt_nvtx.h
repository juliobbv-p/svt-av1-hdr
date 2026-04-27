/*
* Copyright(c) 2026 Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 3-Clause Clear License and
* the Alliance for Open Media Patent License 1.0. If the BSD 3-Clause Clear License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

/*
 * NVTX (NVIDIA Tools Extension) integration for Nsight Systems.
 *
 * All macros expand to no-ops unless SVT_AV1_NVTX is defined at build time
 * (controlled by the SVT_AV1_NVTX CMake option, default OFF).
 *
 * When enabled, the encoder emits per-thread "wait" ranges around
 * svt_get_full_object so each pipeline stage's idle time is visible on the
 * Nsight Systems timeline. Stage identity is conveyed by per-thread names set
 * via pthread_setname_np in svt_create_thread; nsys reads those automatically.
 */

#ifndef Svt_Nvtx_h
#define Svt_Nvtx_h

#if defined(SVT_AV1_NVTX) && SVT_AV1_NVTX
#include <nvtx3/nvToolsExt.h>
#define SVT_NVTX_RANGE_PUSH(name) (void)nvtxRangePushA(name)
#define SVT_NVTX_RANGE_POP() (void)nvtxRangePop()
#else
#define SVT_NVTX_RANGE_PUSH(name) ((void)0)
#define SVT_NVTX_RANGE_POP() ((void)0)
#endif

#endif // Svt_Nvtx_h
