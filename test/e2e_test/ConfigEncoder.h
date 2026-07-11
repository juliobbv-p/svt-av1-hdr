/*
 * Copyright(c) 2019 Intel Corporation
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

#ifndef _SVT_AV1_CONFIG_ENCODER_H_
#define _SVT_AV1_CONFIG_ENCODER_H_
#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"
#include "app_config.h"
#include <string>
#include <memory>

using EbConfigWrapper = std::unique_ptr<EbConfig, void (*)(EbConfig *)>;

EbConfigWrapper create_enc_config(void);
void release_enc_config(EbConfig *config_ptr);
EbErrorType set_enc_config(EbConfig *config, const char *name,
                           const char *value);
void copy_enc_param(EbSvtAv1EncConfiguration *dst_enc_config,
                    EbConfig *config_ptr);
std::string get_enc_token(const char *name);

#endif
