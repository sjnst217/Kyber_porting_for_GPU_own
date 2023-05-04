#pragma once
#define FIPS202_H

#include <stddef.h>
#include <stdint.h>

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136
#define SHA3_256_RATE 136
#define SHA3_384_RATE 104
#define SHA3_512_RATE 72

#define NROUNDS 24
#define ROL(a, offset) (((a) << (offset)) ^ ((a) >> (64 - (offset))))

#define PQC_SHAKEINCCTX_BYTES (sizeof(uint64_t)*26)
#define PQC_SHAKECTX_BYTES (sizeof(uint64_t)*25)

// Context for incremental API
typedef struct {
    uint64_t* ctx;
} shake128incctx;

// Context for non-incremental API
typedef struct {
    uint64_t* ctx;
} shake128ctx;

// Context for incremental API
typedef struct {
    uint64_t* ctx;
} shake256incctx;

// Context for non-incremental API
typedef struct {
    uint64_t* ctx;
} shake256ctx;

// Context for incremental API
typedef struct {
    uint64_t* ctx;
} sha3_256incctx;

// Context for incremental API
typedef struct {
    uint64_t* ctx;
} sha3_384incctx;

// Context for incremental API
typedef struct {
    uint64_t* ctx;
} sha3_512incctx;