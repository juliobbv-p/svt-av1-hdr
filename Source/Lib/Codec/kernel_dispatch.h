/*
* Copyright(c) 2025 Meta Platforms, Inc. and affiliates.
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#ifndef EbKernelDispatch_h
#define EbKernelDispatch_h

#include "EbConfigMacros.h"

#if CONFIG_SINGLE_THREAD_KERNEL
#include "definitions.h"
#include "sys_resource_manager.h"
#include "svt_threads.h"

#ifdef __cplusplus
extern "C" {
#endif

/**************************************
 * Kernel iteration function type.
 * Executes exactly one iteration of a kernel's main loop.
 *
 * Returns:
 *   EB_ErrorNone             - processed one item successfully
 *   EB_NoErrorFifoShutdown   - shutdown signal received, kernel should exit
 **************************************/
typedef EbErrorType (*SvtKernelIterFn)(void* context);

/**************************************
 * Kernel descriptor — one per pipeline stage.
 **************************************/
#define SVT_MAX_KERNELS 32

typedef struct SvtKernelDesc {
    SvtKernelIterFn iter_fn; // single-iteration function
    void*           context; // kernel-specific context (EbThreadContext->priv)
    EbFifo*         input_fifo; // primary input FIFO — dispatcher checks this for pending items
    const char*     name; // debug name
} SvtKernelDesc;

/**************************************
 * Kernel dispatcher — cooperative scheduler for single-thread mode.
 * Replaces 16 OS threads with one cooperative loop.
 **************************************/
typedef struct SvtKernelDispatcher {
    SvtKernelDesc kernels[SVT_MAX_KERNELS];
    uint32_t      num_kernels;
    bool          active; // true when single-thread dispatch mode is enabled
} SvtKernelDispatcher;

// Initialize the dispatcher (zeroes everything, sets active=false)
void svt_kernel_dispatcher_init(SvtKernelDispatcher* dispatcher);

// Register a kernel for single-thread dispatch.
// Call in pipeline order (stage 0 first, stage 15 last).
void svt_kernel_dispatcher_register(SvtKernelDispatcher* dispatcher, SvtKernelIterFn iter_fn, void* context,
                                    EbFifo* input_fifo, const char* name);

// Run cooperative dispatch loop — calls _iter() for each kernel that has
// pending input, repeats until no kernel has work.  Returns when pipeline
// is idle (all FIFOs drained) or shutdown is signaled.
void svt_kernel_dispatcher_run(SvtKernelDispatcher* dispatcher);

/**************************************
 * Wrapper: create kernel thread OR register for dispatch.
 *
 * When dispatcher->active && iter_fn != NULL:
 *   - Registers the kernel with the dispatcher (no thread created)
 *   - Sets *thread_handle = NULL
 *
 * Otherwise:
 *   - Creates an OS thread running kernel_fn (existing behavior)
 *
 * Parameters:
 *   thread_handle  - output thread handle
 *   kernel_fn      - original kernel entry point (void*(*)(void*))
 *   iter_fn        - single-iteration function (NULL if not yet extracted)
 *   thread_ctx     - EbThreadContext* (thread_ctx->priv = kernel context)
 *   input_fifo     - primary input FIFO for dispatch checking
 *   name           - thread/kernel name
 *   dispatcher     - dispatcher (NULL = always create thread)
 **************************************/
EbErrorType svt_create_kernel_or_thread(EbHandle* thread_handle, void* (*kernel_fn)(void*), SvtKernelIterFn iter_fn,
                                        EbThreadContext* thread_ctx, EbFifo* input_fifo, const char* name,
                                        SvtKernelDispatcher* dispatcher);

/**************************************
 * All 16 pipeline kernel _iter() functions.
 * Each executes exactly one iteration of the kernel's main loop.
 **************************************/
EbErrorType svt_aom_resource_coordination_kernel_iter(void* context);
EbErrorType svt_aom_picture_analysis_kernel_iter(void* context);
EbErrorType svt_aom_picture_decision_kernel_iter(void* context);
EbErrorType svt_aom_motion_estimation_kernel_iter(void* context);
EbErrorType svt_aom_initial_rate_control_kernel_iter(void* context);
EbErrorType svt_aom_source_based_operations_kernel_iter(void* context);
EbErrorType svt_aom_tpl_disp_kernel_iter(void* context);
EbErrorType svt_aom_picture_manager_kernel_iter(void* context);
EbErrorType svt_aom_rate_control_kernel_iter(void* context);
EbErrorType svt_aom_mode_decision_configuration_kernel_iter(void* context);
EbErrorType svt_aom_mode_decision_kernel_iter(void* context);
EbErrorType svt_aom_dlf_kernel_iter(void* context);
EbErrorType svt_aom_cdef_kernel_iter(void* context);
EbErrorType svt_aom_rest_kernel_iter(void* context);
EbErrorType svt_aom_entropy_coding_kernel_iter(void* context);
EbErrorType svt_aom_packetization_kernel_iter(void* context);

#ifdef __cplusplus
}
#endif

#endif // CONFIG_SINGLE_THREAD_KERNEL
#endif // EbKernelDispatch_h
