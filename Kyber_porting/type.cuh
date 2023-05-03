#pragma once

#include "params.cuh"
#include "kem.cuh"
#include "fips202.cuh"

typedef struct {
    int16_t coeffs[KYBER_N];
} poly;

typedef struct {
    poly vec[KYBER_K];
} polyvec;

typedef shake128ctx xof_state;

//#define BLOCKSIZE  20
//#define THREADSIZE  2