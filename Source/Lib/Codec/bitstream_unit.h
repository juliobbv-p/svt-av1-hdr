/*
* Copyright(c) 2019 Intel Corporation
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#ifndef EbBitstreamUnit_h
#define EbBitstreamUnit_h

#include "object.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/**********************************
 * Bitstream Unit Types
 **********************************/
typedef struct OutputBitstreamUnit {
    EbDctor  dctor;
    uint32_t size; // allocated buffer size
    uint8_t* buffer_begin_av1; // the byte buffer
    uint8_t* buffer_av1; // the byte buffer
} OutputBitstreamUnit;

/**********************************
     * Extern Function Declarations
     **********************************/
EbErrorType svt_aom_output_bitstream_unit_ctor(OutputBitstreamUnit* bitstream_ptr, uint32_t buffer_size);

EbErrorType svt_aom_output_bitstream_reset(OutputBitstreamUnit* bitstream_ptr);

/********************************************************************************************************************************/
/********************************************************************************************************************************/
/********************************************************************************************************************************/
#include "cabac_context_model.h"

#define OD_DIVU_DMAX (1024)

extern uint32_t svt_aom_od_divu_small_consts[OD_DIVU_DMAX][2];

/*Translate unsigned division by small divisors into multiplications.*/
#define OD_DIVU_SMALL(_x, _d)                                                 \
    ((uint32_t)((svt_aom_od_divu_small_consts[(_d) - 1][0] * (uint64_t)(_x) + \
                 svt_aom_od_divu_small_consts[(_d) - 1][1]) >>                \
                32) >>                                                        \
     (svt_log2f(_d)))

#define OD_DIVU(_x, _d) (((_d) < OD_DIVU_DMAX) ? (OD_DIVU_SMALL((_x), (_d))) : ((_x) / (_d)))
#define OD_ILOG_NZ(_x) (svt_log2f(_x) + 1)

/*Enable special features for gcc and compatible compilers.*/
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__)
#define OD_GNUC_PREREQ(maj, min, pat) \
    ((__GNUC__ << 16) + (__GNUC_MINOR__ << 8) + __GNUC_PATCHLEVEL__ >= ((maj) << 16) + ((min) << 8) + pat) // NOLINT
#else
#define OD_GNUC_PREREQ(maj, min, pat) (0)
#endif

#if OD_GNUC_PREREQ(3, 4, 0)
#define OD_WARN_UNUSED_RESULT __attribute__((__warn_unused_result__))
#else
#define OD_WARN_UNUSED_RESULT
#endif

#if OD_GNUC_PREREQ(3, 4, 0)
#define OD_ARG_NONNULL(x) __attribute__((__nonnull__(x)))
#else
#define OD_ARG_NONNULL(x)
#endif

/** Copy n elements of memory from src to dst. The 0* term provides
compile-time type checking  */
#if !defined(OVERRIDE_OD_COPY)
#define OD_COPY(dst, src, n) (svt_memcpy((dst), (src), sizeof(*(dst)) * (n) + 0 * ((dst) - (src))))
#endif

/********************************************************************************************************************************/
//entcode.h
#define EC_PROB_SHIFT 6
#define EC_MIN_PROB 4 // must be <= (1<<EC_PROB_SHIFT)/16

/*The resolution of fractional-precision bit usage measurements, i.e.,
    3 => 1/8th bits.*/
#define OD_BITRES (3)

#define OD_ICDF AOM_ICDF

/********************************************************************************************************************************/
//entenc.h
typedef uint64_t OdEcWindow;
#define OD_EC_WINDOW_SIZE ((int32_t)sizeof(OdEcWindow) * CHAR_BIT)
#define OD_MEASURE_EC_OVERHEAD (0)

/*The entropy encoder context.*/
typedef struct OdEcEnc {
    /*The low end of the current range.*/
    OdEcWindow low;
    /*The number of values in the current range.
      Widened to uint32_t to eliminate uxth zero-extension instructions on ARM64.
      Value is always in [0x8000, 0xFFFF] post-normalize.*/
    uint32_t rng;
    /*The number of bits of data in the current value.*/
    int16_t cnt;
    /*Nonzero if an error occurred.*/
    int16_t error;
    /*Buffered output. Borrowed from OutputBitstreamUnit via aom_start_encode().*/
    unsigned char* buf;
    /*Write pointer: next byte to write. Invariant: ptr = buf + (bytes written).*/
    unsigned char* ptr;
#if OD_MEASURE_EC_OVERHEAD
    double entropy;
    int    nb_symbols;
#endif
} OdEcEnc;

/*See entenc.c for further documentation.*/
void svt_od_ec_enc_init(OdEcEnc* enc) OD_ARG_NONNULL(1);
void svt_od_ec_enc_reset(OdEcEnc* enc) OD_ARG_NONNULL(1);
void svt_od_ec_encode_bool_eq_q15(OdEcEnc* enc, int32_t val) OD_ARG_NONNULL(1);
void svt_od_ec_encode_bool_q15(OdEcEnc* enc, int32_t val, unsigned f_q15) OD_ARG_NONNULL(1);
void svt_od_ec_encode_cdf_q15(OdEcEnc* enc, int32_t s, const uint16_t* cdf, int32_t nsyms) OD_ARG_NONNULL(1)
    OD_ARG_NONNULL(3);
OD_WARN_UNUSED_RESULT uint8_t* svt_od_ec_enc_done(OdEcEnc* enc, uint32_t* nbytes) OD_ARG_NONNULL(1) OD_ARG_NONNULL(2);
OD_WARN_UNUSED_RESULT int32_t  svt_od_ec_enc_tell(const OdEcEnc* enc) OD_ARG_NONNULL(1);
OD_WARN_UNUSED_RESULT uint32_t svt_od_ec_enc_tell_frac(const OdEcEnc* enc) OD_ARG_NONNULL(1);

/************* endian_inl.h ********************************/
#if defined(__GNUC__)
#define LOCAL_GCC_VERSION ((__GNUC__ << 8) | __GNUC_MINOR__)
#define LOCAL_GCC_PREREQ(maj, min) (LOCAL_GCC_VERSION >= (((maj) << 8) | (min)))
#else
#define LOCAL_GCC_VERSION 0
#define LOCAL_GCC_PREREQ(maj, min) 0
#endif

// handle clang compatibility
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

// some endian fix (e.g.: mips-gcc doesn't define __BIG_ENDIAN__)
#if !defined(WORDS_BIGENDIAN) &&                   \
    (defined(__BIG_ENDIAN__) || defined(_M_PPC) || \
     (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)))
#define WORDS_BIGENDIAN
#endif

#if defined(WORDS_BIGENDIAN)
#define HToLE32 BSwap32
#define HToLE16 BSwap16
#define HToBE64(x) (x)
#define HToBE32(x) (x)
#else
#define HToLE32(x) (x)
#define HToLE16(x) (x)
#define HToBE64(X) BSwap64(X)
#define HToBE32(X) BSwap32(X)
#endif

#if LOCAL_GCC_PREREQ(4, 8) || __has_builtin(__builtin_bswap16)
#define HAVE_BUILTIN_BSWAP16
#endif

#if LOCAL_GCC_PREREQ(4, 3) || __has_builtin(__builtin_bswap32)
#define HAVE_BUILTIN_BSWAP32
#endif

#if LOCAL_GCC_PREREQ(4, 3) || __has_builtin(__builtin_bswap64)
#define HAVE_BUILTIN_BSWAP64
#endif

static inline uint16_t BSwap16(uint16_t x) {
#if defined(HAVE_BUILTIN_BSWAP16)
    return __builtin_bswap16(x);
#elif defined(_MSC_VER)
    return _byteswap_ushort(x);
#else
    // gcc will recognize a 'rorw $8, ...' here:
    return (x >> 8) | ((x & 0xff) << 8);
#endif // HAVE_BUILTIN_BSWAP16
}

static inline uint32_t BSwap32(uint32_t x) {
#if defined(HAVE_BUILTIN_BSWAP32)
    return __builtin_bswap32(x);
#elif defined(__i386__) || defined(__x86_64__)
    uint32_t swapped_bytes;
    __asm__ volatile("bswap %0" : "=r"(swapped_bytes) : "0"(x));
    return swapped_bytes;
#elif defined(_MSC_VER)
    return (uint32_t)_byteswap_ulong(x);
#else
    return (x >> 24) | ((x >> 8) & 0xff00) | ((x << 8) & 0xff0000) | (x << 24);
#endif // HAVE_BUILTIN_BSWAP32
}

static inline uint64_t BSwap64(uint64_t x) {
#if defined(HAVE_BUILTIN_BSWAP64)
    return __builtin_bswap64(x);
#elif defined(__x86_64__)
    uint64_t swapped_bytes;
    __asm__ volatile("bswapq %0" : "=r"(swapped_bytes) : "0"(x));
    return swapped_bytes;
#elif defined(_MSC_VER)
    return (uint64_t)_byteswap_uint64(x);
#else // generic code for swapping 64-bit values (suggested by bdb@)
    x = ((x & 0xffffffff00000000ull) >> 32) | ((x & 0x00000000ffffffffull) << 32);
    x = ((x & 0xffff0000ffff0000ull) >> 16) | ((x & 0x0000ffff0000ffffull) << 16);
    x = ((x & 0xff00ff00ff00ff00ull) >> 8) | ((x & 0x00ff00ff00ff00ffull) << 8);
    return x;
#endif // HAVE_BUILTIN_BSWAP64
}

/********************************************************************************************************************************/
//bitwriter.h
typedef struct AomWriter {
    OdEcEnc  ec;
    uint32_t allow_update_cdf;
    uint32_t pos;
    // save a pointer to the container holding the buffer, in case the buffer must be resized
    OutputBitstreamUnit* buffer_parent;
} AomWriter;

static INLINE void aom_start_encode(AomWriter* br, OutputBitstreamUnit* source) {
    br->buffer_parent = source;
    br->pos           = 0;
    // Borrow tile buffer: EC writes directly to OutputBitstreamUnit's buffer
    br->ec.buf = source->buffer_begin_av1;
    svt_od_ec_enc_reset(&br->ec);
}

EbErrorType svt_realloc_output_bitstream_unit(OutputBitstreamUnit* output_bitstream_ptr, uint32_t sz);

/*Ensures the EC buffer has at least min_free bytes of free space.
  Reallocs through the AomWriter's buffer_parent (OutputBitstreamUnit).
  Should be called before encoding each SB.*/
EbErrorType svt_aom_ec_ensure_capacity(AomWriter* w, uint32_t min_free);

static INLINE void aom_stop_encode(AomWriter* w) {
    uint32_t bytes = 0;
    uint8_t* data  = svt_od_ec_enc_done(&w->ec, &bytes);
    if (!data) {
        return;
    }
    // EC wrote directly to buffer_parent's buffer — no memcpy needed.
    w->pos = bytes;
}

static INLINE void aom_write_bit(AomWriter* w, int bit) {
    svt_od_ec_encode_bool_eq_q15(&w->ec, bit);
}

static INLINE void aom_write_literal(AomWriter* w, unsigned data, int bits) {
    for (int bit = bits - 1; bit >= 0; bit--) {
        aom_write_bit(w, 1 & (data >> bit));
    }
}

static INLINE void aom_write_symbol(AomWriter* w, int symb, AomCdfProb* cdf, int nsymbs) {
    if (nsymbs == 2) {
        // Binary CDF specialization: route directly to the optimal bool encoder.
        // For nsyms==2, the CDF encode path is provably equivalent to
        // svt_od_ec_encode_bool_q15(enc, symb, cdf[0]).
        // When nsymbs is a compile-time constant 2, this branch folds away.
        svt_od_ec_encode_bool_q15(&w->ec, symb, cdf[0]);
    } else {
        svt_od_ec_encode_cdf_q15(&w->ec, symb, cdf, nsymbs);
    }

    if (w->allow_update_cdf) {
        update_cdf(cdf, symb, nsymbs);
    }
}

/********************************************************************************************************************************/
/********************************************************************************************************************************/
#ifdef __cplusplus
}
#endif

#endif // EbBitstreamUnit_h
