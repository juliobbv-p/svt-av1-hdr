/*
 * Copyright(c) 2019 Netflix, Inc.
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
 * @file SvtAv1EncApiTest.cc
 *
 * @brief SVT-AV1 encoder api test, check invalid input
 *
 * @author Cidana-Edmond, Cidana-Ryan, Cidana-Wenyao
 *
 ******************************************************************************/
#include "EbSvtAv1Enc.h"
#include "gtest/gtest.h"
#include "SvtAv1EncApiTest.h"

using namespace svt_av1_test;

namespace {

/** @brief set_parameter_null_pointer is a death test case
 * EncApiDeathTest.set_parameter_null_pointer is a test case for reporting a
 * death condition lead to ececptions or signals
 *
 * Test strategy: <br>
 * Report a death caused by set a null pointer to function
 * svt_av1_enc_set_parameter, which should not happen <br>
 *
 * Expected result: <br>
 * Capture a signal of death in function svt_av1_enc_set_parameter <br>
 *
 * Test coverage:
 * svt_av1_enc_set_parameter.
 */
TEST(EncApiDeathTest, set_parameter_null_pointer) {
    // death tests: TODO: alert, fix me! fix me!! fix me!!!
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    SvtAv1Context context{};

    // initialize encoder and get handle
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_init_handle(&context.enc_handle, nullptr));
    // watch out, function down
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_set_parameter(context.enc_handle, nullptr));
    // destroy encoder handle
    EXPECT_EQ(EB_ErrorInvalidComponent, svt_av1_enc_deinit_handle(nullptr));
    SUCCEED();
}

/** @brief check_null_pointer is a api test case
 * EncApiTest.check_null_pointer is a api test case for checking null pointer
 * parameters setting into api functions and expect report for a
 * EB_ErrorBadParameter return
 *
 * Test strategy: <br>
 * Input nullptr to the encoder API and check the return value.
 *
 * Expected result: <br>
 * Encoder API should not crash and report EB_ErrorBadParameter.
 *
 * Test coverage:
 * All the encoder parameters.
 */
TEST(EncApiTest, check_null_pointer) {
    SvtAv1Context context{};

    // initialize encoder and with all null pointer
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_init_handle(nullptr, nullptr));
    // initialize encoder and with all null pointer and get handle
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_init_handle(&context.enc_handle, nullptr));
    // setup encoder parameters with null pointer
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_set_parameter(nullptr, nullptr));
    // TODO: Some function will crash with nullptr input,
    // and it will block test on linux platform. please refer to
    // EncApiDeathTest-->check_null_pointer
    // EXPECT_EQ(EB_ErrorBadParameter,
    //          svt_av1_enc_set_parameter(context.enc_handle,
    //          nullptr));
    // open encoder with null pointer
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_init(nullptr));
    // get end of sequence NAL with null pointer
    // EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_send_picture(nullptr,
    // nullptr)); EXPECT_EQ(EB_ErrorBadParameter,
    // svt_av1_enc_get_packet(nullptr, nullptr, 0));
    // EXPECT_EQ(EB_ErrorBadParameter, svt_av1_get_recon(nullptr, nullptr)); No
    // return value, just feed nullptr as parameter. release output buffer with
    // null pointer
    svt_av1_enc_release_out_buffer(nullptr);
    // close encoder with null pointer
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_deinit(nullptr));
    // destroy encoder handle with null pointer
    EXPECT_EQ(EB_ErrorInvalidComponent, svt_av1_enc_deinit_handle(nullptr));
    SUCCEED();
}

/** @brief check_normal_setup is a api test case
 * EncApiTest.check_normal_setup is a api test case with a normal setup
 * parameters into api functions and expect report for return EB_ErrorNone
 *
 * Test strategy: <br>
 * Input normal parameters to the encoder API and check the return value.
 *
 * Expected result: <br>
 * Encoder API should not crash and report EB_ErrorNone.
 *
 * Test coverage:
 * All the encoder parameters.
 *
 * Comments:
 * Disabled for it will hang after test report, might be thread can not exit
 * only happens without in IDE debugging mode.
 */
TEST(EncApiTest, DISABLED_check_normal_setup) {
    SvtAv1Context context{};

    const int width = 1280;
    const int height = 720;

    // initialize encoder and get handle
    EXPECT_EQ(EB_ErrorNone,
              svt_av1_enc_init_handle(&context.enc_handle, &context.enc_params))
        << "svt_av1_enc_init_handle failed";
    // setup source width/height with default value
    context.enc_params.source_width = width;
    context.enc_params.source_height = height;
    // setup encoder parameters with all default
    EXPECT_EQ(
        EB_ErrorNone,
        svt_av1_enc_set_parameter(context.enc_handle, &context.enc_params))
        << "svt_av1_enc_set_parameter failed";
    // open encoder
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_init(context.enc_handle))
        << "svt_av1_enc_init failed";
    // close encoder
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit(context.enc_handle))
        << "svt_av1_enc_deinit failed";
    // destroy encoder
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(context.enc_handle))
        << "svt_av1_enc_deinit_handle failed";
}

/** @brief repeat_normal_setup is a api test case
 * EncApiTest.repeat_normal_setup is a api test case of repeating test with a
 * default normal setup to check for a resource or memory leak
 *
 * Test strategy: <br>
 * Input normal parameters to the encoder API and repeat the processing of
 * initialize and destroy encoder handle and check the return value.
 *
 * Expected result: <br>
 * Encoder can initialize normally without any error reported.
 *
 * Test coverage:
 * Initialize and destroy APIs.
 *
 * Comments:
 * Disabled for it causes memory leak, and lead to effect other tests
 */
TEST(EncApiTest, DISABLED_repeat_normal_setup) {
    SvtAv1Context context{};

    const int width = 1280;
    const int height = 720;

    for (size_t i = 0; i < 500; ++i) {
        // initialize encoder and get handle
        ASSERT_EQ(
            EB_ErrorNone,
            svt_av1_enc_init_handle(&context.enc_handle, &context.enc_params))
            << "svt_av1_enc_init_handle failed at " << i << " times";
        // setup source width/height with default value
        context.enc_params.source_width = width;
        context.enc_params.source_height = height;
        // setup encoder parameters with all default
        ASSERT_EQ(
            EB_ErrorNone,
            svt_av1_enc_set_parameter(context.enc_handle, &context.enc_params))
            << "svt_av1_enc_set_parameter failed at " << i << " times";
        // TODO:if not calls svt_av1_enc_deinit, there is huge memory leak, fix
        // me
        // ASSERT_EQ(EB_ErrorNone, svt_av1_enc_deinit(context.enc_handle))
        //    << "svt_av1_enc_deinit failed";
        // destroy encoder
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(context.enc_handle))
            << "svt_av1_enc_deinit_handle failed at " << i << " times";
    }
}

}  // namespace

/**
 * @brief Small resolution encoding tests
 *
 * Test strategy:
 * Initialize encoder with small resolutions, send a dummy frame,
 * retrieve output, and verify no crashes or hangs occur.
 *
 * Covers: 1x1, 3x3, 4x4, 64x1, 65x1, 360x1, 1x360
 * These test the minimum dimension support and the segment count
 * fix for non-square small images.
 */

struct SmallResParam {
    uint32_t width;
    uint32_t height;
};

using SmallResEncodeTest = ::testing::TestWithParam<SmallResParam>;

TEST_P(SmallResEncodeTest, encode_one_frame) {
    const SmallResParam &p = GetParam();
    svt_av1_test::SvtAv1Context ctxt{};

    // init handle
    ASSERT_EQ(EB_ErrorNone,
              svt_av1_enc_init_handle(&ctxt.enc_handle, &ctxt.enc_params));

    // configure for small resolution AVIF-style encoding
    ctxt.enc_params.source_width = p.width;
    ctxt.enc_params.source_height = p.height;
    ctxt.enc_params.encoder_bit_depth = 8;
    ctxt.enc_params.encoder_color_format = EB_YUV420;
    ctxt.enc_params.pred_structure = ALL_INTRA;

    ASSERT_EQ(EB_ErrorNone,
              svt_av1_enc_set_parameter(ctxt.enc_handle, &ctxt.enc_params));

    // open encoder
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init(ctxt.enc_handle));

    // allocate and send a dummy input frame (all zeros)
    EbBufferHeaderType input_buf{};
    input_buf.size = sizeof(EbBufferHeaderType);

    EbSvtIOFormat input_pic{};

    const uint32_t luma_size = p.width * p.height;
    const uint32_t chroma_w = (p.width + 1) / 2;
    const uint32_t chroma_h = (p.height + 1) / 2;
    const uint32_t chroma_size = chroma_w * chroma_h;

    std::vector<uint8_t> luma(luma_size, 128);
    std::vector<uint8_t> cb(chroma_size, 128);
    std::vector<uint8_t> cr(chroma_size, 128);

    input_pic.luma = luma.data();
    input_pic.cb = cb.data();
    input_pic.cr = cr.data();
    input_pic.y_stride = p.width;
    input_pic.cb_stride = chroma_w;
    input_pic.cr_stride = chroma_w;

    input_buf.p_buffer = reinterpret_cast<uint8_t *>(&input_pic);
    input_buf.n_filled_len = luma_size + 2 * chroma_size;
    input_buf.pts = 0;
    input_buf.pic_type = EB_AV1_INVALID_PICTURE;

    ASSERT_EQ(EB_ErrorNone,
              svt_av1_enc_send_picture(ctxt.enc_handle, &input_buf));

    // send EOS
    EbBufferHeaderType eos_buf{};
    eos_buf.size = sizeof(EbBufferHeaderType);
    eos_buf.flags = EB_BUFFERFLAG_EOS;
    ASSERT_EQ(EB_ErrorNone,
              svt_av1_enc_send_picture(ctxt.enc_handle, &eos_buf));

    // get output — drain until EOS
    bool got_eos = false;
    bool got_data = false;
    while (!got_eos) {
        EbBufferHeaderType *out = nullptr;
        EbErrorType ret =
            svt_av1_enc_get_packet(ctxt.enc_handle, &out, /*pic_send_done=*/1);
        if (ret == EB_ErrorNone && out != nullptr) {
            if (out->n_filled_len > 0)
                got_data = true;
            if (out->flags & EB_BUFFERFLAG_EOS)
                got_eos = true;
            svt_av1_enc_release_out_buffer(&out);
        } else {
            break;
        }
    }
    EXPECT_TRUE(got_eos) << "Did not receive EOS for " << p.width << "x"
                         << p.height;
    EXPECT_TRUE(got_data) << "No encoded data for " << p.width << "x"
                          << p.height;

    // cleanup
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_deinit(ctxt.enc_handle));
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(ctxt.enc_handle));
}

INSTANTIATE_TEST_SUITE_P(
    SmallResolutions, SmallResEncodeTest,
    ::testing::Values(SmallResParam{1, 1},    // minimum possible
                      SmallResParam{2, 2},    // sub-MI size
                      SmallResParam{3, 3},    // odd sub-MI size
                      SmallResParam{4, 4},    // 1 MI unit
                      SmallResParam{8, 8},    // 1 MIN_BLOCK_SIZE
                      SmallResParam{64, 1},   // single SB width, minimal height
                      SmallResParam{65, 1},   // multi-SB width, minimal height
                      SmallResParam{65, 4},   // multi-SB width, small height
                      SmallResParam{128, 1},  // 2 SB width, minimal height
                      SmallResParam{360, 1},  // wide strip, minimal height
                      SmallResParam{1, 64},   // minimal width, single SB height
                      SmallResParam{1, 65},   // minimal width, multi-SB height
                      SmallResParam{1, 360}   // tall strip, minimal width
                      ));
