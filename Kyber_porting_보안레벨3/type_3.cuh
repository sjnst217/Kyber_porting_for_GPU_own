#pragma once

#include "params_3.cuh"
#include "kem_3.cuh"
#include "fips202_3.cuh"

typedef struct {
    int16_t coeffs[KYBER_N];
} poly;

typedef struct {
    poly vec[KYBER_K];
} polyvec;

typedef shake128ctx xof_state;