/*
 * Copyright(c) 2026 Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

/******************************************************************************
 * @file SvtAv1EncMgSizeApiTest.cc
 *
 * @brief Regression tests for runtime MG-size (temporal-layer) switching and
 *        the max_hierarchical_levels buffer-pool ceiling.
 *
 * Adaptive temporal layering starts a session flat (L1T1, hierarchical_levels
 * == 0) and raises the mini-GOP size later with MG_SIZE_CHANGE_EVENT. Buffer
 * pools are sized once at init, so a session that intends to switch must
 * declare the ceiling via max_hierarchical_levels; otherwise the flat-IPP
 * sizing shortcut leaves the input-picture pool too small and the L1T2 frames
 * exhaust it.
 *
 * Two groups of tests:
 *   1. Validation of the new field (svt_av1_verify_settings) — every rejection
 *      is paired with a max_hierarchical_levels == 0 control that must be
 *      ACCEPTED, so a pass can never come from an unrelated validation rule.
 *   2. The load-bearing switch test: init flat, run frames, switch to L1T2,
 *      keep running. With the ceiling declared the pools cover it and the whole
 *      run completes. (Reverting the sizing gate makes the flag a no-op, the
 *      pool exhausts, and this build blocks inside svt_av1_enc_send_picture —
 *      caught by the watchdog as a test failure rather than a hang of the
 *      whole suite.)
 *
 * Note on the error-return path: in single-thread RTC builds
 * (CONFIG_SINGLE_THREAD_KERNEL) an exhausted pool makes svt_get_empty_object
 * return EB_ErrorInsufficientResources instead of blocking, and the fix checks
 * that return. This suite is a normal multithreaded build where the acquire
 * blocks, so the undeclared-ceiling case is a deadlock (guarded here by the
 * watchdog), not an error return.
 ******************************************************************************/

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>

#include "DummyVideoSource.h"
#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"
#include "gtest/gtest.h"

namespace {

constexpr uint32_t kWidth = 320;
constexpr uint32_t kHeight = 240;

// Configure the encoder the way the RTC wrapper does: RTC low-delay CBR, flat
// IPP. Mirrors SvtAv1EncOnTheFlyApiTest so the pool sizing under test is the
// one production uses.
void fill_rtc_low_delay_cbr(EbSvtAv1EncConfiguration& cfg) {
    cfg.source_width = kWidth;
    cfg.source_height = kHeight;
    cfg.frame_rate_numerator = 30000;
    cfg.frame_rate_denominator = 1000;
    cfg.encoder_bit_depth = 8;
    cfg.encoder_color_format = EB_YUV420;
    cfg.enc_mode = 9;
    cfg.tune = 1;
    cfg.rtc = true;
    cfg.intra_period_length = 100;
    cfg.hierarchical_levels = 0;
    cfg.pred_structure = LOW_DELAY;
    cfg.rate_control_mode = SVT_AV1_RC_MODE_CBR;
    cfg.target_bit_rate = 500000;
    cfg.max_qp_allowed = 63;
    cfg.min_qp_allowed = 4;
    cfg.look_ahead_distance = 0;
    cfg.recode_loop = 0;
}

// mutate(cfg, default_tbr) breaks exactly one thing about an otherwise valid
// config; default_tbr is the library default target bitrate captured right
// after init_handle (needed to reach the CQP/CRF path, which rejects any other
// target bitrate first).
using Mutation = std::function<void(EbSvtAv1EncConfiguration&, uint32_t)>;

// A rejection only proves the new rule if the SAME configuration is accepted
// with max_hierarchical_levels == 0 — otherwise an unrelated validation rule is
// doing the work and the test would pass whether or not the new rule exists.
void expect_rejected_only_by_max_levels(const Mutation& mutate,
                                        uint8_t levels) {
    // Control: identical mutation, ceiling not declared → must be accepted.
    {
        EbComponentType* enc = nullptr;
        EbSvtAv1EncConfiguration cfg{};
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc, &cfg));
        const uint32_t default_tbr = cfg.target_bit_rate;
        fill_rtc_low_delay_cbr(cfg);
        mutate(cfg, default_tbr);
        cfg.max_hierarchical_levels = 0;
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc, &cfg))
            << "control config is invalid for an unrelated reason";
        svt_av1_enc_deinit_handle(enc);
    }
    // Candidate: same mutation, ceiling declared → must be rejected.
    {
        EbComponentType* enc = nullptr;
        EbSvtAv1EncConfiguration cfg{};
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc, &cfg));
        const uint32_t default_tbr = cfg.target_bit_rate;
        fill_rtc_low_delay_cbr(cfg);
        mutate(cfg, default_tbr);
        cfg.max_hierarchical_levels = levels;
        EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_set_parameter(enc, &cfg))
            << "levels=" << +levels;
        svt_av1_enc_deinit_handle(enc);
    }
}

TEST(SvtAv1MgSizeApiTest, DefaultIsZero) {
    // Poisoned before svt_av1_set_default_params runs, so the default has to be
    // written rather than inherited from the caller's struct.
    EbComponentType* enc = nullptr;
    EbSvtAv1EncConfiguration cfg{};
    cfg.max_hierarchical_levels = 0xFF;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc, &cfg));
    EXPECT_EQ(0, cfg.max_hierarchical_levels);
    svt_av1_enc_deinit_handle(enc);
}

TEST(SvtAv1MgSizeApiTest, AcceptedOnRtcLowDelayCbr) {
    for (uint8_t levels = 1; levels <= 2; ++levels) {
        EbComponentType* enc = nullptr;
        EbSvtAv1EncConfiguration cfg{};
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc, &cfg));
        fill_rtc_low_delay_cbr(cfg);
        cfg.max_hierarchical_levels = levels;
        EXPECT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc, &cfg))
            << "levels=" << +levels;
        svt_av1_enc_deinit_handle(enc);
    }
}

TEST(SvtAv1MgSizeApiTest, RejectedAboveTwo) {
    for (uint8_t levels = 3; levels <= 5; ++levels) {
        expect_rejected_only_by_max_levels(
            [](EbSvtAv1EncConfiguration&, uint32_t) {}, levels);
    }
}

TEST(SvtAv1MgSizeApiTest, RejectedWhenNotRtc) {
    expect_rejected_only_by_max_levels(
        [](EbSvtAv1EncConfiguration& c, uint32_t) { c.rtc = false; }, 1);
}

TEST(SvtAv1MgSizeApiTest, RejectedOutsideCbr) {
    expect_rejected_only_by_max_levels(
        [](EbSvtAv1EncConfiguration& c, uint32_t default_tbr) {
            c.rate_control_mode = SVT_AV1_RC_MODE_CQP_OR_CRF;
            c.target_bit_rate = default_tbr;
        },
        1);
}

TEST(SvtAv1MgSizeApiTest, RejectedBelowConfiguredHierarchicalLevels) {
    expect_rejected_only_by_max_levels(
        [](EbSvtAv1EncConfiguration& c, uint32_t) {
            c.hierarchical_levels = 2;
        },
        1);
}

// -------- switch test (buffer-pool sizing) --------

constexpr uint32_t kFramesBeforeSwitch = 10;
constexpr uint32_t kFramesAfterSwitch = 90;
// The fixed library encodes these tiny frames in well under a second; only a
// genuine deadlock (undeclared ceiling → exhausted pool → blocking acquire)
// reaches this bound.
constexpr int kWatchdogSec = 30;

struct SwitchResult {
    bool all_sends_ok = true;
    size_t frames_sent = 0;
    size_t packets = 0;
    bool completed = false;
};

// Init flat, send kFramesBeforeSwitch frames, attach MG_SIZE_CHANGE_EVENT(1) to
// switch to L1T2, send kFramesAfterSwitch more, draining one packet per
// accepted frame (low-delay get_packet blocks, so get/produce must stay
// balanced), then flush. Runs on a worker thread.
void run_switch_loop(uint8_t max_hierarchical_levels, SwitchResult* out) {
    EbComponentType* enc = nullptr;
    EbSvtAv1EncConfiguration cfg{};
    if (svt_av1_enc_init_handle(&enc, &cfg) != EB_ErrorNone) {
        return;
    }
    fill_rtc_low_delay_cbr(cfg);
    cfg.max_hierarchical_levels = max_hierarchical_levels;
    if (svt_av1_enc_set_parameter(enc, &cfg) != EB_ErrorNone ||
        svt_av1_enc_init(enc) != EB_ErrorNone) {
        svt_av1_enc_deinit_handle(enc);
        return;
    }

    svt_av1_video_source::DummyVideoSource src(IMG_FMT_420, kWidth, kHeight, 8);
    if (src.open_source(0, 0) != 0) {
        svt_av1_enc_deinit(enc);
        svt_av1_enc_deinit_handle(enc);
        return;
    }

    EbBufferHeaderType in_hdr{};
    in_hdr.size = sizeof(EbBufferHeaderType);

    const uint32_t total = kFramesBeforeSwitch + kFramesAfterSwitch;
    for (uint32_t fi = 0; fi < total; ++fi) {
        EbSvtIOFormat* frame = src.get_next_frame();
        if (!frame) {
            break;
        }

        // send_picture copies the private-data list, so these only need to
        // outlive the call below.
        SvtAv1MgSizeInfo mg{};
        EbPrivDataNode node{};

        in_hdr.p_buffer = reinterpret_cast<uint8_t*>(frame);
        in_hdr.n_filled_len = src.get_frame_size();
        in_hdr.pts = fi;
        in_hdr.flags = 0;
        in_hdr.pic_type =
            (fi == 0) ? EB_AV1_KEY_PICTURE : EB_AV1_INVALID_PICTURE;
        in_hdr.p_app_private = nullptr;

        if (fi == kFramesBeforeSwitch) {
            mg.hierarchical_levels = 1;  // switch to L1T2
            node.node_type = MG_SIZE_CHANGE_EVENT;
            node.data = &mg;
            node.size = sizeof(SvtAv1MgSizeInfo);
            node.next = nullptr;
            in_hdr.p_app_private = &node;
        }

        const EbErrorType rc = svt_av1_enc_send_picture(enc, &in_hdr);
        if (rc != EB_ErrorNone) {
            out->all_sends_ok = false;
            break;
        }
        ++out->frames_sent;

        EbBufferHeaderType* o = nullptr;
        if (svt_av1_enc_get_packet(enc, &o, 0) == EB_ErrorNone && o) {
            if (!(o->flags & EB_BUFFERFLAG_EOS)) {
                ++out->packets;
            }
            svt_av1_enc_release_out_buffer(&o);
        }
    }

    EbBufferHeaderType eos{};
    eos.size = sizeof(EbBufferHeaderType);
    eos.flags = EB_BUFFERFLAG_EOS;
    eos.pic_type = EB_AV1_INVALID_PICTURE;
    svt_av1_enc_send_picture(enc, &eos);
    for (;;) {
        EbBufferHeaderType* o = nullptr;
        if (svt_av1_enc_get_packet(enc, &o, 1) != EB_ErrorNone || !o) {
            break;
        }
        const bool is_eos = (o->flags & EB_BUFFERFLAG_EOS) != 0;
        if (!is_eos) {
            ++out->packets;
        }
        svt_av1_enc_release_out_buffer(&o);
        if (is_eos) {
            break;
        }
    }

    svt_av1_enc_deinit(enc);
    svt_av1_enc_deinit_handle(enc);
    out->completed = true;
}

// Returns true if the loop finished within the watchdog window; false if it
// looks deadlocked (the worker is left detached — it is stuck inside a blocking
// encoder call and cannot be joined).
bool run_switch_with_watchdog(uint8_t max_hierarchical_levels,
                              SwitchResult* out) {
    std::atomic<bool> done{false};
    std::thread worker([&] {
        run_switch_loop(max_hierarchical_levels, out);
        done.store(true, std::memory_order_release);
    });

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(kWatchdogSec);
    while (!done.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    if (done.load(std::memory_order_acquire)) {
        worker.join();
        return true;
    }
    worker.detach();
    return false;
}

// With the ceiling declared the pools cover L1T2, so the full flat→L1T2 run
// completes: every send is accepted and the stream produces packets.
TEST(SvtAv1MgSizeApiTest, SwitchToL1T2WithMaxLevelsSucceeds) {
    SwitchResult r;
    ASSERT_TRUE(run_switch_with_watchdog(/*max_hierarchical_levels=*/1, &r))
        << "flat->L1T2 switch deadlocked: input pool was not sized for the "
           "declared ceiling (sizing gate regressed)";
    EXPECT_TRUE(r.completed);
    EXPECT_TRUE(r.all_sends_ok);
    EXPECT_EQ(kFramesBeforeSwitch + kFramesAfterSwitch, r.frames_sent);
    EXPECT_GT(r.packets, 0u) << "encoder produced no packets";
}

}  // namespace
