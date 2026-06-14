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

#include "ConfigEncoder.h"
#include "app_config.h"
#include <memory>
#include <string>

extern "C" {
#include "app_config.c"
}

EbErrorType set_enc_config(EbConfig *config_ptr, const char *name,
                           const char *value) {
    return set_config_value(config_ptr, name, value);
}

bool set_default_config(EbSvtAv1EncConfiguration *config) {
    EbComponentType *handle;
    if (svt_av1_enc_init_handle(&handle, config) != EB_ErrorNone) {
        return false;
    }
    svt_av1_enc_deinit_handle(handle);
    return true;
}

void release_enc_config(EbConfig *config_ptr) {
    svt_config_dtor(config_ptr);
}

EbConfigWrapper create_enc_config() {
    EbConfigWrapper app_cfg(svt_config_ctor(false), &svt_config_dtor);
    assert(app_cfg);
    if (!set_default_config(&app_cfg->config)) {
        return EbConfigWrapper(nullptr, &svt_config_dtor);
    }
    return app_cfg;
}

std::string get_enc_token(const char *name) {
    ConfigEntry *ent = find_entry(name);
    return ent ? ent->token : std::string{};
}
