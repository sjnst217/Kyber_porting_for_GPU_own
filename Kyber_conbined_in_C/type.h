#pragma once

#include "params.h"
#include "kem.h"
#include "fips202.h"

typedef struct {
    int16_t coeffs[KYBER_N];
} poly;

typedef struct {
    poly vec[KYBER_K];
} polyvec;

typedef shake128ctx xof_state;

#define     N_TIMES 1    //���ÿ� �����ϴ� Kyber�� ����
//#define BLOCKSIZE  20
//#define THREADSIZE  2