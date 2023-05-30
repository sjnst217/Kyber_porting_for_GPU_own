#pragma once

#include "params_1.cuh"
#include "kem_1.cuh"
#include "fips202_1.cuh"

typedef struct {
    int16_t coeffs[KYBER_N];
} poly;

typedef struct {
    poly vec[KYBER_K];
} polyvec;

typedef shake128ctx xof_state;

#define     N_TIMES 1    //동시에 동작하는 Kyber의 개수
//#define BLOCKSIZE  20
//#define THREADSIZE  2