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

#include "kernel_dispatch.h"

#if CONFIG_SINGLE_THREAD_KERNEL
#include <string.h>
#include <assert.h>
#include "enc_handle.h" // for EbThreadContext (struct _EbThreadContext)

void svt_kernel_dispatcher_init(SvtKernelDispatcher* dispatcher) {
    memset(dispatcher, 0, sizeof(*dispatcher));
}

void svt_kernel_dispatcher_register(SvtKernelDispatcher* dispatcher, SvtKernelIterFn iter_fn, void* context,
                                    EbFifo* input_fifo, const char* name) {
    assert(dispatcher->num_kernels < SVT_MAX_KERNELS);
    SvtKernelDesc* desc = &dispatcher->kernels[dispatcher->num_kernels++];
    desc->iter_fn       = iter_fn;
    desc->context       = context;
    desc->input_fifo    = input_fifo;
    desc->name          = name;
}

void svt_kernel_dispatcher_run(SvtKernelDispatcher* dispatcher) {
    bool progress;
    do {
        progress = false;
        for (uint32_t i = 0; i < dispatcher->num_kernels; i++) {
            SvtKernelDesc* desc = &dispatcher->kernels[i];
            // Process all pending items for this kernel before moving to next
            while (svt_fifo_has_items_st(desc->input_fifo)) {
                EbErrorType err = desc->iter_fn(desc->context);
                if (err == EB_NoErrorFifoShutdown) {
                    return;
                }
                progress = true;
            }
        }
    } while (progress);
}

EbErrorType svt_create_kernel_or_thread(EbHandle* thread_handle, void* (*kernel_fn)(void*), SvtKernelIterFn iter_fn,
                                        EbThreadContext* thread_ctx, EbFifo* input_fifo, const char* name,
                                        SvtKernelDispatcher* dispatcher) {
    if (dispatcher && dispatcher->active && iter_fn) {
        // Single-thread mode: register kernel for cooperative dispatch
        svt_kernel_dispatcher_register(dispatcher, iter_fn, thread_ctx->priv, input_fifo, name);
        *thread_handle = NULL;
    } else {
        // Multi-thread mode: create OS thread with original kernel function
        *thread_handle = svt_create_thread(kernel_fn, thread_ctx, name);
        if (!*thread_handle) {
            return EB_ErrorInsufficientResources;
        }
        EB_ADD_MEM(*thread_handle, 1, EB_THREAD);
    }
    return EB_ErrorNone;
}

#endif // CONFIG_SINGLE_THREAD_KERNEL
