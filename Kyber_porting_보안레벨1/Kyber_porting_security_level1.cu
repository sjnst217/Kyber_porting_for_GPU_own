#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "kem_1.cuh"
#include "type_1.cuh"
#include "fips202_1.cuh"
#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define hash_h(OUT, IN, INBYTES) sha3_256(OUT, IN, INBYTES)
#define hash_g(OUT, IN, INBYTES) sha3_512(OUT, IN, INBYTES)
#define gen_a(A,B)  PQCLEAN_KYBER512_CLEAN_gen_matrix(A,B,0)
#define gen_at(A,B) PQCLEAN_KYBER512_CLEAN_gen_matrix(A,B,1)

#define xof_absorb(STATE, SEED, X, Y) PQCLEAN_KYBER512_CLEAN_kyber_shake128_absorb(STATE, SEED, X, Y)
#define xof_squeezeblocks(OUT, OUTBLOCKS, STATE) shake128_squeezeblocks(OUT, OUTBLOCKS, STATE)
#define xof_ctx_release(STATE) shake128_ctx_release(STATE)
#define prf(OUT, OUTBYTES, KEY, NONCE) PQCLEAN_KYBER512_CLEAN_kyber_shake256_prf(OUT, OUTBYTES, KEY, NONCE)
#define kdf(OUT, IN, INBYTES) shake256(OUT, KYBER_SSBYTES, IN, INBYTES)

#define GEN_MATRIX_NBLOCKS ((12*KYBER_N/8*(1 << 12)/KYBER_Q + XOF_BLOCKBYTES)/XOF_BLOCKBYTES)

#define XOF_BLOCKBYTES SHAKE128_RATE

__device__ static void randombytes_win32_randombytes(uint8_t* buf, const size_t n) {
    for (int i = 0; i < 32; i++) {
        *buf = i;
        buf++;
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_poly_frommsg(poly* r, const uint8_t msg[KYBER_INDCPA_MSGBYTES]) {
    size_t i, j;
    int16_t mask;

    for (i = 0; i < KYBER_N / 8; i++) {
        for (j = 0; j < 8; j++) {
            mask = -(int16_t)((msg[i] >> j) & 1);
            r->coeffs[8 * i + j] = mask & ((KYBER_Q + 1) / 2);
        }
    }
}

__device__ static const uint64_t KeccakF_RoundConstants[NROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};
__device__ const int16_t PQCLEAN_KYBER512_CLEAN_zetas[128] = {
    -1044,  -758,  -359, -1517,  1493,  1422,   287,   202,
        -171,   622,  1577,   182,   962, -1202, -1474,  1468,
        573, -1325,   264,   383,  -829,  1458, -1602,  -130,
        -681,  1017,   732,   608, -1542,   411,  -205, -1571,
        1223,   652,  -552,  1015, -1293,  1491,  -282, -1544,
        516,    -8,  -320,  -666, -1618, -1162,   126,  1469,
        -853,   -90,  -271,   830,   107, -1421,  -247,  -951,
        -398,   961, -1508,  -725,   448, -1065,   677, -1275,
        -1103,   430,   555,   843, -1251,   871,  1550,   105,
        422,   587,   177,  -235,  -291,  -460,  1574,  1653,
        -246,   778,  1159,  -147,  -777,  1483,  -602,  1119,
        -1590,   644,  -872,   349,   418,   329,  -156,   -75,
        817,  1097,   603,   610,  1322, -1285, -1465,   384,
        -1215,  -136,  1218, -1335,  -874,   220, -1187, -1659,
        -1185, -1530, -1278,   794, -1510,  -854,  -870,   478,
        -108,  -308,   996,   991,   958, -1460,  1522,  1628
};

__device__ static void store64(uint8_t* x, uint64_t u) {
    for (size_t i = 0; i < 8; ++i) {
        x[i] = (uint8_t)(u >> 8 * i);
    }
}
__device__ static uint64_t load64(const uint8_t* x) {
    uint64_t r = 0;
    for (size_t i = 0; i < 8; ++i) {
        r |= (uint64_t)x[i] << 8 * i;
    }

    return r;
}
__device__ static uint32_t load24_littleendian(const uint8_t x[3])
{
    uint32_t r;
    r = (uint32_t)x[0];
    r |= (uint32_t)x[1] << 8;
    r |= (uint32_t)x[2] << 16;
    return r;
}
__device__ static uint32_t load32_littleendian(const uint8_t x[4]) {
    uint32_t r;
    r = (uint32_t)x[0];
    r |= (uint32_t)x[1] << 8;
    r |= (uint32_t)x[2] << 16;
    r |= (uint32_t)x[3] << 24;
    return r;
}

__device__ void PQCLEAN_KYBER512_CLEAN_poly_tobytes(uint8_t r[KYBER_POLYBYTES], const poly* a)
{
    size_t i;
    uint16_t t0, t1;

    for (i = 0; i < KYBER_N / 2; i++) {
        // map to positive standard representatives
        t0 = a->coeffs[2 * i];
        t0 += ((int16_t)t0 >> 15) & KYBER_Q;
        t1 = a->coeffs[2 * i + 1];
        t1 += ((int16_t)t1 >> 15) & KYBER_Q;
        r[3 * i + 0] = (uint8_t)(t0 >> 0);
        r[3 * i + 1] = (uint8_t)((t0 >> 8) | (t1 << 4));
        r[3 * i + 2] = (uint8_t)(t1 >> 4);
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_poly_tomsg(uint8_t msg[KYBER_INDCPA_MSGBYTES], const poly* a) {
    size_t i, j;
    uint16_t t;

    for (i = 0; i < KYBER_N / 8; i++) {
        msg[i] = 0;
        for (j = 0; j < 8; j++) {
            t = a->coeffs[8 * i + j];
            t += ((int16_t)t >> 15) & KYBER_Q;
            t = (((t << 1) + KYBER_Q / 2) / KYBER_Q) & 1;
            msg[i] |= t << j;
        }
    }
}

__device__ static void KeccakF1600_StatePermute(uint64_t* state) {
    int round;

    uint64_t Aba, Abe, Abi, Abo, Abu;
    uint64_t Aga, Age, Agi, Ago, Agu;
    uint64_t Aka, Ake, Aki, Ako, Aku;
    uint64_t Ama, Ame, Ami, Amo, Amu;
    uint64_t Asa, Ase, Asi, Aso, Asu;
    uint64_t BCa, BCe, BCi, BCo, BCu;
    uint64_t Da, De, Di, Do, Du;
    uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
    uint64_t Ega, Ege, Egi, Ego, Egu;
    uint64_t Eka, Eke, Eki, Eko, Eku;
    uint64_t Ema, Eme, Emi, Emo, Emu;
    uint64_t Esa, Ese, Esi, Eso, Esu;

    // copyFromState(A, state)
    Aba = state[0];
    Abe = state[1];
    Abi = state[2];
    Abo = state[3];
    Abu = state[4];
    Aga = state[5];
    Age = state[6];
    Agi = state[7];
    Ago = state[8];
    Agu = state[9];
    Aka = state[10];
    Ake = state[11];
    Aki = state[12];
    Ako = state[13];
    Aku = state[14];
    Ama = state[15];
    Ame = state[16];
    Ami = state[17];
    Amo = state[18];
    Amu = state[19];
    Asa = state[20];
    Ase = state[21];
    Asi = state[22];
    Aso = state[23];
    Asu = state[24];

    for (round = 0; round < NROUNDS; round += 2) {
        //    prepareTheta
        BCa = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
        BCe = Abe ^ Age ^ Ake ^ Ame ^ Ase;
        BCi = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
        BCo = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
        BCu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

        // thetaRhoPiChiIotaPrepareTheta(round  , A, E)
        Da = BCu ^ ROL(BCe, 1);
        De = BCa ^ ROL(BCi, 1);
        Di = BCe ^ ROL(BCo, 1);
        Do = BCi ^ ROL(BCu, 1);
        Du = BCo ^ ROL(BCa, 1);

        Aba ^= Da;
        BCa = Aba;
        Age ^= De;
        BCe = ROL(Age, 44);
        Aki ^= Di;
        BCi = ROL(Aki, 43);
        Amo ^= Do;
        BCo = ROL(Amo, 21);
        Asu ^= Du;
        BCu = ROL(Asu, 14);
        Eba = BCa ^ ((~BCe) & BCi);
        Eba ^= KeccakF_RoundConstants[round];
        Ebe = BCe ^ ((~BCi) & BCo);
        Ebi = BCi ^ ((~BCo) & BCu);
        Ebo = BCo ^ ((~BCu) & BCa);
        Ebu = BCu ^ ((~BCa) & BCe);

        Abo ^= Do;
        BCa = ROL(Abo, 28);
        Agu ^= Du;
        BCe = ROL(Agu, 20);
        Aka ^= Da;
        BCi = ROL(Aka, 3);
        Ame ^= De;
        BCo = ROL(Ame, 45);
        Asi ^= Di;
        BCu = ROL(Asi, 61);
        Ega = BCa ^ ((~BCe) & BCi);
        Ege = BCe ^ ((~BCi) & BCo);
        Egi = BCi ^ ((~BCo) & BCu);
        Ego = BCo ^ ((~BCu) & BCa);
        Egu = BCu ^ ((~BCa) & BCe);

        Abe ^= De;
        BCa = ROL(Abe, 1);
        Agi ^= Di;
        BCe = ROL(Agi, 6);
        Ako ^= Do;
        BCi = ROL(Ako, 25);
        Amu ^= Du;
        BCo = ROL(Amu, 8);
        Asa ^= Da;
        BCu = ROL(Asa, 18);
        Eka = BCa ^ ((~BCe) & BCi);
        Eke = BCe ^ ((~BCi) & BCo);
        Eki = BCi ^ ((~BCo) & BCu);
        Eko = BCo ^ ((~BCu) & BCa);
        Eku = BCu ^ ((~BCa) & BCe);

        Abu ^= Du;
        BCa = ROL(Abu, 27);
        Aga ^= Da;
        BCe = ROL(Aga, 36);
        Ake ^= De;
        BCi = ROL(Ake, 10);
        Ami ^= Di;
        BCo = ROL(Ami, 15);
        Aso ^= Do;
        BCu = ROL(Aso, 56);
        Ema = BCa ^ ((~BCe) & BCi);
        Eme = BCe ^ ((~BCi) & BCo);
        Emi = BCi ^ ((~BCo) & BCu);
        Emo = BCo ^ ((~BCu) & BCa);
        Emu = BCu ^ ((~BCa) & BCe);

        Abi ^= Di;
        BCa = ROL(Abi, 62);
        Ago ^= Do;
        BCe = ROL(Ago, 55);
        Aku ^= Du;
        BCi = ROL(Aku, 39);
        Ama ^= Da;
        BCo = ROL(Ama, 41);
        Ase ^= De;
        BCu = ROL(Ase, 2);
        Esa = BCa ^ ((~BCe) & BCi);
        Ese = BCe ^ ((~BCi) & BCo);
        Esi = BCi ^ ((~BCo) & BCu);
        Eso = BCo ^ ((~BCu) & BCa);
        Esu = BCu ^ ((~BCa) & BCe);

        //    prepareTheta
        BCa = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
        BCe = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
        BCi = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
        BCo = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
        BCu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

        // thetaRhoPiChiIotaPrepareTheta(round+1, E, A)
        Da = BCu ^ ROL(BCe, 1);
        De = BCa ^ ROL(BCi, 1);
        Di = BCe ^ ROL(BCo, 1);
        Do = BCi ^ ROL(BCu, 1);
        Du = BCo ^ ROL(BCa, 1);

        Eba ^= Da;
        BCa = Eba;
        Ege ^= De;
        BCe = ROL(Ege, 44);
        Eki ^= Di;
        BCi = ROL(Eki, 43);
        Emo ^= Do;
        BCo = ROL(Emo, 21);
        Esu ^= Du;
        BCu = ROL(Esu, 14);
        Aba = BCa ^ ((~BCe) & BCi);
        Aba ^= KeccakF_RoundConstants[round + 1];
        Abe = BCe ^ ((~BCi) & BCo);
        Abi = BCi ^ ((~BCo) & BCu);
        Abo = BCo ^ ((~BCu) & BCa);
        Abu = BCu ^ ((~BCa) & BCe);

        Ebo ^= Do;
        BCa = ROL(Ebo, 28);
        Egu ^= Du;
        BCe = ROL(Egu, 20);
        Eka ^= Da;
        BCi = ROL(Eka, 3);
        Eme ^= De;
        BCo = ROL(Eme, 45);
        Esi ^= Di;
        BCu = ROL(Esi, 61);
        Aga = BCa ^ ((~BCe) & BCi);
        Age = BCe ^ ((~BCi) & BCo);
        Agi = BCi ^ ((~BCo) & BCu);
        Ago = BCo ^ ((~BCu) & BCa);
        Agu = BCu ^ ((~BCa) & BCe);

        Ebe ^= De;
        BCa = ROL(Ebe, 1);
        Egi ^= Di;
        BCe = ROL(Egi, 6);
        Eko ^= Do;
        BCi = ROL(Eko, 25);
        Emu ^= Du;
        BCo = ROL(Emu, 8);
        Esa ^= Da;
        BCu = ROL(Esa, 18);
        Aka = BCa ^ ((~BCe) & BCi);
        Ake = BCe ^ ((~BCi) & BCo);
        Aki = BCi ^ ((~BCo) & BCu);
        Ako = BCo ^ ((~BCu) & BCa);
        Aku = BCu ^ ((~BCa) & BCe);

        Ebu ^= Du;
        BCa = ROL(Ebu, 27);
        Ega ^= Da;
        BCe = ROL(Ega, 36);
        Eke ^= De;
        BCi = ROL(Eke, 10);
        Emi ^= Di;
        BCo = ROL(Emi, 15);
        Eso ^= Do;
        BCu = ROL(Eso, 56);
        Ama = BCa ^ ((~BCe) & BCi);
        Ame = BCe ^ ((~BCi) & BCo);
        Ami = BCi ^ ((~BCo) & BCu);
        Amo = BCo ^ ((~BCu) & BCa);
        Amu = BCu ^ ((~BCa) & BCe);

        Ebi ^= Di;
        BCa = ROL(Ebi, 62);
        Ego ^= Do;
        BCe = ROL(Ego, 55);
        Eku ^= Du;
        BCi = ROL(Eku, 39);
        Ema ^= Da;
        BCo = ROL(Ema, 41);
        Ese ^= De;
        BCu = ROL(Ese, 2);
        Asa = BCa ^ ((~BCe) & BCi);
        Ase = BCe ^ ((~BCi) & BCo);
        Asi = BCi ^ ((~BCo) & BCu);
        Aso = BCo ^ ((~BCu) & BCa);
        Asu = BCu ^ ((~BCa) & BCe);
    }

    // copyToState(state, A)
    state[0] = Aba;
    state[1] = Abe;
    state[2] = Abi;
    state[3] = Abo;
    state[4] = Abu;
    state[5] = Aga;
    state[6] = Age;
    state[7] = Agi;
    state[8] = Ago;
    state[9] = Agu;
    state[10] = Aka;
    state[11] = Ake;
    state[12] = Aki;
    state[13] = Ako;
    state[14] = Aku;
    state[15] = Ama;
    state[16] = Ame;
    state[17] = Ami;
    state[18] = Amo;
    state[19] = Amu;
    state[20] = Asa;
    state[21] = Ase;
    state[22] = Asi;
    state[23] = Aso;
    state[24] = Asu;
}
__device__ static void keccak_absorb(uint64_t* s, uint32_t r, const uint8_t* m, size_t mlen, uint8_t p) {
    size_t i;
    uint8_t t[200];

    /* Zero state */
    for (i = 0; i < 25; ++i) {
        s[i] = 0;
    }

    while (mlen >= r) {
        for (i = 0; i < r / 8; ++i) {
            s[i] ^= load64(m + 8 * i);
        }

        KeccakF1600_StatePermute(s);
        mlen -= r;
        m += r;
    }

    for (i = 0; i < r; ++i) {
        t[i] = 0;
    }
    for (i = 0; i < mlen; ++i) {
        t[i] = m[i];
    }
    t[i] = p;
    t[r - 1] |= 128;
    for (i = 0; i < r / 8; ++i) {
        s[i] ^= load64(t + 8 * i);
    }
}
__device__ void shake128_ctx_release(shake128ctx* state)
{
    free(state->ctx);
}
__device__ void shake128_absorb(shake128ctx* state, const uint8_t* input, size_t inlen)
{
    state->ctx = (uint64_t*)malloc(PQC_SHAKECTX_BYTES);

    keccak_absorb(state->ctx, SHAKE128_RATE, input, inlen, 0x1F);
}
__device__ static void keccak_squeezeblocks(uint8_t* h, size_t nblocks, uint64_t* s, uint32_t r) {
    while (nblocks > 0) {
        KeccakF1600_StatePermute(s);
        for (size_t i = 0; i < (r >> 3); i++) {
            store64(h + 8 * i, s[i]);
        }
        h += r;
        nblocks--;
    }
}
__device__ void shake128_squeezeblocks(uint8_t* output, size_t nblocks, shake128ctx* state)
{
    keccak_squeezeblocks(output, nblocks, state->ctx, SHAKE128_RATE);
}

__device__ void PQCLEAN_KYBER512_CLEAN_poly_compress(uint8_t r[KYBER_POLYCOMPRESSEDBYTES], const poly* a) {
    size_t i, j;
    int16_t u;
    uint8_t t[8];

    for (i = 0; i < KYBER_N / 8; i++) {
        for (j = 0; j < 8; j++) {
            // map to positive standard representatives
            u = a->coeffs[8 * i + j];
            u += (u >> 15) & KYBER_Q;
            t[j] = ((((uint16_t)u << 4) + KYBER_Q / 2) / KYBER_Q) & 15;
        }

        r[0] = t[0] | (t[1] << 4);
        r[1] = t[2] | (t[3] << 4);
        r[2] = t[4] | (t[5] << 4);
        r[3] = t[6] | (t[7] << 4);
        r += 4;
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_polyvec_compress(uint8_t r[KYBER_POLYVECCOMPRESSEDBYTES], const polyvec* a) {
    unsigned int i, j, k;

    uint16_t t[4];
    for (i = 0; i < KYBER_K; i++) {
        for (j = 0; j < KYBER_N / 4; j++) {
            for (k = 0; k < 4; k++) {
                t[k] = a->vec[i].coeffs[4 * j + k];
                t[k] += ((int16_t)t[k] >> 15) & KYBER_Q;
                t[k] = ((((uint32_t)t[k] << 10) + KYBER_Q / 2) / KYBER_Q) & 0x3ff;
            }

            r[0] = (uint8_t)(t[0] >> 0);
            r[1] = (uint8_t)((t[0] >> 8) | (t[1] << 2));
            r[2] = (uint8_t)((t[1] >> 6) | (t[2] << 4));
            r[3] = (uint8_t)((t[2] >> 4) | (t[3] << 6));
            r[4] = (uint8_t)(t[3] >> 2);
            r += 5;
        }
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_polyvec_tobytes(uint8_t r[KYBER_POLYVECBYTES], const polyvec* a)
{
    unsigned int i;
    for (i = 0; i < KYBER_K; i++) {
        PQCLEAN_KYBER512_CLEAN_poly_tobytes(r + i * KYBER_POLYBYTES, &a->vec[i]);
    }
}
__device__ static void pack_sk(uint8_t r[KYBER_INDCPA_SECRETKEYBYTES], polyvec* sk)
{
    PQCLEAN_KYBER512_CLEAN_polyvec_tobytes(r, sk);
}
__device__ static void pack_pk(uint8_t r[KYBER_INDCPA_PUBLICKEYBYTES], polyvec* pk, const uint8_t seed[KYBER_SYMBYTES])
{
    size_t i;
    PQCLEAN_KYBER512_CLEAN_polyvec_tobytes(r, pk);
    for (i = 0; i < KYBER_SYMBYTES; i++) {
        r[i + KYBER_POLYVECBYTES] = seed[i];
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_kyber_shake128_absorb(xof_state* state, const uint8_t seed[KYBER_SYMBYTES], uint8_t x, uint8_t y)
{
    uint8_t extseed[KYBER_SYMBYTES + 2];

    memcpy(extseed, seed, KYBER_SYMBYTES);
    extseed[KYBER_SYMBYTES + 0] = x;
    extseed[KYBER_SYMBYTES + 1] = y;

    shake128_absorb(state, extseed, sizeof(extseed));
}
__device__ static unsigned int rej_uniform(int16_t* r, unsigned int len, const uint8_t* buf, unsigned int buflen)
{
    unsigned int ctr, pos;
    uint16_t val0, val1;

    ctr = pos = 0;
    while (ctr < len && pos + 3 <= buflen) {
        val0 = ((buf[pos + 0] >> 0) | ((uint16_t)buf[pos + 1] << 8)) & 0xFFF;
        val1 = ((buf[pos + 1] >> 4) | ((uint16_t)buf[pos + 2] << 4)) & 0xFFF;
        pos += 3;

        if (val0 < KYBER_Q) {
            r[ctr++] = val0;
        }
        if (ctr < len && val1 < KYBER_Q) {
            r[ctr++] = val1;
        }
    }

    return ctr;
}
__device__ void PQCLEAN_KYBER512_CLEAN_gen_matrix(polyvec* a, const uint8_t seed[KYBER_SYMBYTES], int transposed) {
    unsigned int ctr, i, j, k;
    unsigned int buflen, off;
    uint8_t buf[GEN_MATRIX_NBLOCKS * XOF_BLOCKBYTES + 2];
    xof_state state;

    for (i = 0; i < KYBER_K; i++) {
        for (j = 0; j < KYBER_K; j++) {
            if (transposed) {
                xof_absorb(&state, seed, (uint8_t)i, (uint8_t)j);
            }
            else {
                xof_absorb(&state, seed, (uint8_t)j, (uint8_t)i);
            }

            xof_squeezeblocks(buf, GEN_MATRIX_NBLOCKS, &state);
            buflen = GEN_MATRIX_NBLOCKS * XOF_BLOCKBYTES;
            ctr = rej_uniform(a[i].vec[j].coeffs, KYBER_N, buf, buflen);

            while (ctr < KYBER_N) {
                off = buflen % 3;
                for (k = 0; k < off; k++) {
                    buf[k] = buf[buflen - off + k];
                }
                xof_squeezeblocks(buf + off, 1, &state);
                buflen = off + XOF_BLOCKBYTES;
                ctr += rej_uniform(a[i].vec[j].coeffs + ctr, KYBER_N - ctr, buf, buflen);
            }
            xof_ctx_release(&state);
        }
    }
}

__device__ void sha3_256(uint8_t* output, const uint8_t* input, size_t inlen)
{
    uint64_t s[25];
    uint8_t t[SHA3_256_RATE];

    /* Absorb input */
    keccak_absorb(s, SHA3_256_RATE, input, inlen, 0x06);

    /* Squeeze output */
    keccak_squeezeblocks(t, 1, s, SHA3_256_RATE);

    for (size_t i = 0; i < 32; i++) {
        output[i] = t[i];
    }
}
__device__ void sha3_512(uint8_t* output, const uint8_t* input, size_t inlen) {
    uint64_t s[25];
    uint8_t t[SHA3_512_RATE];

    /* Absorb input */
    keccak_absorb(s, SHA3_512_RATE, input, inlen, 0x06);

    /* Squeeze output */
    keccak_squeezeblocks(t, 1, s, SHA3_512_RATE);

    for (size_t i = 0; i < 64; i++) {
        output[i] = t[i];
    }
}
__device__ void shake256_absorb(shake256ctx* state, const uint8_t* input, size_t inlen)
{
    state->ctx = (uint64_t*)malloc(PQC_SHAKECTX_BYTES);

    keccak_absorb(state->ctx, SHAKE256_RATE, input, inlen, 0x1F);
}
__device__ void shake256_squeezeblocks(uint8_t* output, size_t nblocks, shake256ctx* state)
{
    keccak_squeezeblocks(output, nblocks, state->ctx, SHAKE256_RATE);
}
__device__ void shake256_ctx_release(shake256ctx* state)
{
    free(state->ctx);
}
__device__ void shake256(uint8_t* output, size_t outlen, const uint8_t* input, size_t inlen)
{
    size_t nblocks = outlen / SHAKE256_RATE;
    uint8_t t[SHAKE256_RATE];
    shake256ctx s;

    shake256_absorb(&s, input, inlen);
    shake256_squeezeblocks(output, nblocks, &s);

    output += nblocks * SHAKE256_RATE;
    outlen -= nblocks * SHAKE256_RATE;

    if (outlen) {
        shake256_squeezeblocks(t, 1, &s);
        for (size_t i = 0; i < outlen; ++i) {
            output[i] = t[i];
        }
    }
    shake256_ctx_release(&s);
}

__device__ static void cbd3(poly* r, const uint8_t buf[3 * KYBER_N / 4])
{
    unsigned int i, j;
    uint32_t t, d;
    int16_t a, b;

    for (i = 0; i < KYBER_N / 4; i++) {
        t = load24_littleendian(buf + 3 * i);
        d = t & 0x00249249;
        d += (t >> 1) & 0x00249249;
        d += (t >> 2) & 0x00249249;

        for (j = 0; j < 4; j++) {
            a = (d >> (6 * j + 0)) & 0x7;
            b = (d >> (6 * j + 3)) & 0x7;
            r->coeffs[4 * i + j] = a - b;
        }
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_poly_cbd_eta1(poly* r, const uint8_t buf[KYBER_ETA1 * KYBER_N / 4])
{
    cbd3(r, buf);
}
__device__ void PQCLEAN_KYBER512_CLEAN_kyber_shake256_prf(uint8_t* out, size_t outlen, const uint8_t key[KYBER_SYMBYTES], uint8_t nonce) {
    uint8_t extkey[KYBER_SYMBYTES + 1];

    memcpy(extkey, key, KYBER_SYMBYTES);
    extkey[KYBER_SYMBYTES] = nonce;

    shake256(out, outlen, extkey, sizeof(extkey));
}
__device__ void PQCLEAN_KYBER512_CLEAN_poly_getnoise_eta1(poly* r, const uint8_t seed[KYBER_SYMBYTES], uint8_t nonce) {
    uint8_t buf[KYBER_ETA1 * KYBER_N / 4];
    prf(buf, sizeof(buf), seed, nonce);
    PQCLEAN_KYBER512_CLEAN_poly_cbd_eta1(r, buf);
}
__device__ int16_t PQCLEAN_KYBER512_CLEAN_montgomery_reduce(int32_t a)
{
    int16_t t;

    t = (int16_t)a * QINV;
    t = (a - (int32_t)t * KYBER_Q) >> 16;
    return t;
}
__device__ static int16_t fqmul(int16_t a, int16_t b)
{
    return PQCLEAN_KYBER512_CLEAN_montgomery_reduce((int32_t)a * b);
}
__device__ void PQCLEAN_KYBER512_CLEAN_ntt(int16_t r[256])
{
    unsigned int len, start, j, k;
    int16_t t, zeta;

    k = 1;
    for (len = 128; len >= 2; len >>= 1) {
        for (start = 0; start < 256; start = j + len) {
            zeta = PQCLEAN_KYBER512_CLEAN_zetas[k++];
            for (j = start; j < start + len; j++) {
                t = fqmul(zeta, r[j + len]);
                r[j + len] = r[j] - t;
                r[j] = r[j] + t;
            }
        }
    }
}
__device__ int16_t PQCLEAN_KYBER512_CLEAN_barrett_reduce(int16_t a) {
    int16_t t;
    const int16_t v = ((1 << 26) + KYBER_Q / 2) / KYBER_Q;

    t = ((int32_t)v * a + (1 << 25)) >> 26;
    t *= KYBER_Q;
    return a - t;
}
__device__ void PQCLEAN_KYBER512_CLEAN_poly_sub(poly* r, const poly* a, const poly* b) {
    size_t i;
    for (i = 0; i < KYBER_N; i++) {
        r->coeffs[i] = a->coeffs[i] - b->coeffs[i];
    }
}

__device__ void PQCLEAN_KYBER512_CLEAN_poly_reduce(poly* r)
{
    size_t i;
    for (i = 0; i < KYBER_N; i++) {
        r->coeffs[i] = PQCLEAN_KYBER512_CLEAN_barrett_reduce(r->coeffs[i]);
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_poly_ntt(poly* r)
{
    PQCLEAN_KYBER512_CLEAN_ntt(r->coeffs);
    PQCLEAN_KYBER512_CLEAN_poly_reduce(r);
}
__device__ void PQCLEAN_KYBER512_CLEAN_polyvec_ntt(polyvec* r)
{
    unsigned int i;
    for (i = 0; i < KYBER_K; i++) {
        PQCLEAN_KYBER512_CLEAN_poly_ntt(&r->vec[i]);
    }
}

__device__ void PQCLEAN_KYBER512_CLEAN_basemul(int16_t r[2], const int16_t a[2], const int16_t b[2], int16_t zeta)
{
    r[0] = fqmul(a[1], b[1]);
    r[0] = fqmul(r[0], zeta);
    r[0] += fqmul(a[0], b[0]);
    r[1] = fqmul(a[0], b[1]);
    r[1] += fqmul(a[1], b[0]);
}
__device__ void PQCLEAN_KYBER512_CLEAN_poly_basemul_montgomery(poly* r, const poly* a, const poly* b)
{
    size_t i;
    for (i = 0; i < KYBER_N / 4; i++) {
        PQCLEAN_KYBER512_CLEAN_basemul(&r->coeffs[4 * i], &a->coeffs[4 * i], &b->coeffs[4 * i], PQCLEAN_KYBER512_CLEAN_zetas[64 + i]);
        PQCLEAN_KYBER512_CLEAN_basemul(&r->coeffs[4 * i + 2], &a->coeffs[4 * i + 2], &b->coeffs[4 * i + 2], -PQCLEAN_KYBER512_CLEAN_zetas[64 + i]);
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_poly_add(poly* r, const poly* a, const poly* b)
{
    size_t i;
    for (i = 0; i < KYBER_N; i++) {
        r->coeffs[i] = a->coeffs[i] + b->coeffs[i];
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_polyvec_basemul_acc_montgomery(poly* r, const polyvec* a, const polyvec* b)
{
    unsigned int i;
    poly t;

    PQCLEAN_KYBER512_CLEAN_poly_basemul_montgomery(r, &a->vec[0], &b->vec[0]);
    for (i = 1; i < KYBER_K; i++) {
        PQCLEAN_KYBER512_CLEAN_poly_basemul_montgomery(&t, &a->vec[i], &b->vec[i]);
        PQCLEAN_KYBER512_CLEAN_poly_add(r, r, &t);
    }

    PQCLEAN_KYBER512_CLEAN_poly_reduce(r);
}
__device__ void PQCLEAN_KYBER512_CLEAN_poly_tomont(poly* r)
{
    size_t i;
    const int16_t f = (1ULL << 32) % KYBER_Q;
    for (i = 0; i < KYBER_N; i++) {
        r->coeffs[i] = PQCLEAN_KYBER512_CLEAN_montgomery_reduce((int32_t)r->coeffs[i] * f);
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_polyvec_add(polyvec* r, const polyvec* a, const polyvec* b)
{
    unsigned int i;
    for (i = 0; i < KYBER_K; i++) {
        PQCLEAN_KYBER512_CLEAN_poly_add(&r->vec[i], &a->vec[i], &b->vec[i]);
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_polyvec_reduce(polyvec* r) {
    unsigned int i;
    for (i = 0; i < KYBER_K; i++) {
        PQCLEAN_KYBER512_CLEAN_poly_reduce(&r->vec[i]);
    }
}

__device__ void PQCLEAN_KYBER512_CLEAN_indcpa_keypair(uint8_t pk[KYBER_INDCPA_PUBLICKEYBYTES], uint8_t sk[KYBER_INDCPA_SECRETKEYBYTES])
{
    unsigned int i;
    uint8_t buf[2 * KYBER_SYMBYTES];                    // 해시, 시드를 저장하는 배열
    const uint8_t* publicseed = buf;                    // 공개키의 seed값의 저장을 위한 publicseed 선언
    const uint8_t* noiseseed = buf + KYBER_SYMBYTES;    // noise값의 seed값을 저장하기 위한 noiseseed 선언

    /** 과정 3 **/
    uint8_t nonce = 0;

    polyvec a[KYBER_K], e, pkpv, skpv;

    /** 과정 1 **/
    //randombytes(buf, KYBER_SYMBYTES);                 // 랜덤한 32바이트 생성
    for (int i = 0; i < 32; i++) {
        buf[i] = i;
    }

    /** 과정 2 **/
    hash_g(buf, buf, KYBER_SYMBYTES);                   // SHA3_512로 랜덤값을 Seed로 만들어줌 -> SHA3_512의 앞 32byte를 공개키의 seed값, 뒤 32byte를 noise의 seed값 으로 사용

    /** 과정 4 ~ 8 **/
    gen_a(a, publicseed);                               //공개행렬 a 생성 (2x2)

    /** 과정 9 ~ 12 **/
    for (i = 0; i < KYBER_K; i++) {
        PQCLEAN_KYBER512_CLEAN_poly_getnoise_eta1(&skpv.vec[i], noiseseed, nonce++);    //skpv.vec[0], noiseseed, 0 -> skpv.vec[1], noiseseed, 1
    }   // 비밀키 s 생성

    /** 과정 13 ~ 16 **/
    for (i = 0; i < KYBER_K; i++) {
        PQCLEAN_KYBER512_CLEAN_poly_getnoise_eta1(&e.vec[i], noiseseed, nonce++);       //e.vec[0], noiseseed, 2 -> e.vec[1], noiseseed, 3
    }   // 에러다항식 e 생성


    /** 과정 17 ~ 18 **/
    PQCLEAN_KYBER512_CLEAN_polyvec_ntt(&skpv);  // s를 NTT변환!
    PQCLEAN_KYBER512_CLEAN_polyvec_ntt(&e);     // e를 NTT 변환!


    /** 과정 19 **/
    // matrix-vector multiplication
    for (i = 0; i < KYBER_K; i++) {
        PQCLEAN_KYBER512_CLEAN_polyvec_basemul_acc_montgomery(&pkpv.vec[i], &a[i], &skpv);
        PQCLEAN_KYBER512_CLEAN_poly_tomont(&pkpv.vec[i]);
    } // X = A*s

    PQCLEAN_KYBER512_CLEAN_polyvec_add(&pkpv, &pkpv, &e);       // X + e
    PQCLEAN_KYBER512_CLEAN_polyvec_reduce(&pkpv);               // 다항식 계수들에 대해 mod q

    /** 과정 20 ~ 21 **/
    pack_sk(sk, &skpv);                                         //sk 직렬화 -> 즉 바이트 배열로 저장
    pack_pk(pk, &pkpv, publicseed);                             //pk 직렬화 -> 즉 바이트 배열로 저장
}
__device__ void PQCLEAN_KYBER512_CLEAN_poly_frombytes(poly* r, const uint8_t a[KYBER_POLYBYTES]) {
    size_t i;
    for (i = 0; i < KYBER_N / 2; i++) {
        r->coeffs[2 * i] = ((a[3 * i + 0] >> 0) | ((uint16_t)a[3 * i + 1] << 8)) & 0xFFF;
        r->coeffs[2 * i + 1] = ((a[3 * i + 1] >> 4) | ((uint16_t)a[3 * i + 2] << 4)) & 0xFFF;
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_polyvec_frombytes(polyvec* r, const uint8_t a[KYBER_POLYVECBYTES]) {
    unsigned int i;
    for (i = 0; i < KYBER_K; i++) {
        PQCLEAN_KYBER512_CLEAN_poly_frombytes(&r->vec[i], a + i * KYBER_POLYBYTES);
    }
}

__device__ static void unpack_pk(polyvec* pk, uint8_t seed[KYBER_SYMBYTES], const uint8_t packedpk[KYBER_INDCPA_PUBLICKEYBYTES])
{
    size_t i;
    /** 과정 2 **/
    PQCLEAN_KYBER512_CLEAN_polyvec_frombytes(pk, packedpk);
    for (i = 0; i < KYBER_SYMBYTES; i++) {
        /** 과정 3 **/
        seed[i] = packedpk[i + KYBER_POLYVECBYTES];
    }
}
__device__ static void cbd2(poly* r, const uint8_t buf[2 * KYBER_N / 4]) {
    unsigned int i, j;
    uint32_t t, d;
    int16_t a, b;

    for (i = 0; i < KYBER_N / 8; i++) {
        t = load32_littleendian(buf + 4 * i);
        d = t & 0x55555555;
        d += (t >> 1) & 0x55555555;

        for (j = 0; j < 8; j++) {
            a = (d >> (4 * j + 0)) & 0x3;
            b = (d >> (4 * j + 2)) & 0x3;
            r->coeffs[8 * i + j] = a - b;
        }
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_poly_cbd_eta2(poly* r, const uint8_t buf[KYBER_ETA2 * KYBER_N / 4]) {
    cbd2(r, buf);
}
__device__ void PQCLEAN_KYBER512_CLEAN_poly_getnoise_eta2(poly* r, const uint8_t seed[KYBER_SYMBYTES], uint8_t nonce) {
    uint8_t buf[KYBER_ETA2 * KYBER_N / 4];
    prf(buf, sizeof(buf), seed, nonce);
    PQCLEAN_KYBER512_CLEAN_poly_cbd_eta2(r, buf);
}

__device__ void PQCLEAN_KYBER512_CLEAN_invntt(int16_t r[256]) {
    unsigned int start, len, j, k;
    int16_t t, zeta;
    const int16_t f = 1441; // mont^2/128

    k = 127;
    for (len = 2; len <= 128; len <<= 1) {
        for (start = 0; start < 256; start = j + len) {
            zeta = PQCLEAN_KYBER512_CLEAN_zetas[k--];
            for (j = start; j < start + len; j++) {
                t = r[j];
                r[j] = PQCLEAN_KYBER512_CLEAN_barrett_reduce(t + r[j + len]);
                r[j + len] = r[j + len] - t;
                r[j + len] = fqmul(zeta, r[j + len]);
            }
        }
    }

    for (j = 0; j < 256; j++) {
        r[j] = fqmul(r[j], f);
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_poly_invntt_tomont(poly* r) {
    PQCLEAN_KYBER512_CLEAN_invntt(r->coeffs);
}
__device__ void PQCLEAN_KYBER512_CLEAN_polyvec_invntt_tomont(polyvec* r) {
    unsigned int i;
    for (i = 0; i < KYBER_K; i++) {
        PQCLEAN_KYBER512_CLEAN_poly_invntt_tomont(&r->vec[i]);
    }
}

__device__ static void pack_ciphertext(uint8_t r[KYBER_INDCPA_BYTES], polyvec* b, poly* v) {
    PQCLEAN_KYBER512_CLEAN_polyvec_compress(r, b);
    PQCLEAN_KYBER512_CLEAN_poly_compress(r + KYBER_POLYVECCOMPRESSEDBYTES, v);
}
__device__ void PQCLEAN_KYBER512_CLEAN_indcpa_enc(uint8_t* c, const uint8_t m[KYBER_INDCPA_MSGBYTES], const uint8_t pk[KYBER_INDCPA_PUBLICKEYBYTES], const uint8_t coins[KYBER_SYMBYTES])   //알고리즘 4
{
    unsigned int i;
    uint8_t seed[KYBER_SYMBYTES];
    uint8_t nonce = 0;                      // /** 과정 1 **/
    polyvec sp, pkpv, ep, at[KYBER_K], b;
    poly v, k, epp;

    /** 과정 2, 3 **/
    unpack_pk(&pkpv, seed, pk);                 //array -> module 이라고 생각하는게 편함


    /** 과정 4 ~ 8 **/
    gen_at(at, seed);

    /** 과정 9 ~ 12 **/
    for (i = 0; i < KYBER_K; i++) {
        PQCLEAN_KYBER512_CLEAN_poly_getnoise_eta1(sp.vec + i, coins, nonce++);
    }

    /** 과정 13 ~ 16 **/
    for (i = 0; i < KYBER_K; i++) {
        PQCLEAN_KYBER512_CLEAN_poly_getnoise_eta2(ep.vec + i, coins, nonce++);
    }

    /** 과정 17 **/
    PQCLEAN_KYBER512_CLEAN_poly_getnoise_eta2(&epp, coins, nonce++);

    /** 과정 18 **/
    PQCLEAN_KYBER512_CLEAN_polyvec_ntt(&sp);

    /** 과정 19 - mul(AT, r) 부분 **/
    // matrix-vector multiplication
    for (i = 0; i < KYBER_K; i++) {
        PQCLEAN_KYBER512_CLEAN_polyvec_basemul_acc_montgomery(&b.vec[i], &at[i], &sp);
    }

    /** 과정 20 - mul(tT, r) 부분 **/
    PQCLEAN_KYBER512_CLEAN_polyvec_basemul_acc_montgomery(&v, &pkpv, &sp);

    /** 과정 19 invNTT(mul(AT, r)) 부분 **/
    PQCLEAN_KYBER512_CLEAN_polyvec_invntt_tomont(&b);

    /** 과정 20 invNTT(mul(tT, r)) 부분 **/
    PQCLEAN_KYBER512_CLEAN_poly_invntt_tomont(&v);

    /** 과정 19 invNTT(mul(AT, r)) + e1 부분 **/
    PQCLEAN_KYBER512_CLEAN_polyvec_add(&b, &b, &ep);

    /** 과정 20 invNTT(mul(tT, r)) + e2  + Decompress_q부분 **/
    PQCLEAN_KYBER512_CLEAN_poly_add(&v, &v, &epp);  // + e2 과정
    PQCLEAN_KYBER512_CLEAN_poly_frommsg(&k, m);     //m의 값(message)을 k 값(polynomial)로 변경, (Decompress_q(Decode_1(m), 1) 과정)
    PQCLEAN_KYBER512_CLEAN_poly_add(&v, &v, &k);    // + k 과정

    /** 과정 21 **/
    PQCLEAN_KYBER512_CLEAN_polyvec_reduce(&b);

    /** 과정 22 **/
    PQCLEAN_KYBER512_CLEAN_poly_reduce(&v);

    /** 과정 23 **/
    pack_ciphertext(c, &b, &v);
}

__device__ void PQCLEAN_KYBER512_CLEAN_crypto_kem_keypair(uint8_t* pk, uint8_t* sk)
{
    size_t i;
    PQCLEAN_KYBER512_CLEAN_indcpa_keypair(pk, sk);                      //알고리즘 3 -> KEY 생성함수 : KYBER의 CPA(Chosen Plaintext Attack) 안전성을 만족하는 PKE 스킴의 키생성 과정
    for (i = 0; i < KYBER_INDCPA_PUBLICKEYBYTES; i++) {
        sk[i + KYBER_INDCPA_SECRETKEYBYTES] = pk[i];
    }
    hash_h(sk + KYBER_SECRETKEYBYTES - 2 * KYBER_SYMBYTES, pk, KYBER_PUBLICKEYBYTES);
    /* Value z for pseudo-random output on reject */
    randombytes_win32_randombytes(sk + KYBER_SECRETKEYBYTES - KYBER_SYMBYTES, KYBER_SYMBYTES);
}

__device__ void PQCLEAN_KYBER512_CLEAN_crypto_kem_enc(uint8_t* ct, uint8_t* ss, uint8_t* pk)  //알고리즘 7 Encapsulation
{
    uint8_t buf[2 * KYBER_SYMBYTES];
    /* Will contain key, coins */
    uint8_t kr[2 * KYBER_SYMBYTES];

    /** 과정 1 **/
    randombytes_win32_randombytes(buf, KYBER_SYMBYTES);

    /* Don't release system RNG output */
    /** 과정 2 **/
    hash_h(buf, buf, KYBER_SYMBYTES);

    /* Multitarget countermeasure for coins + contributory KEM */
    /** 과정 3 **/
    hash_h(buf + KYBER_SYMBYTES, pk, KYBER_PUBLICKEYBYTES);
    hash_g(kr, buf, 2 * KYBER_SYMBYTES);

    /* coins are in kr+KYBER_SYMBYTES */
    /** 과정 4 **/
    PQCLEAN_KYBER512_CLEAN_indcpa_enc(ct, buf, pk, kr + KYBER_SYMBYTES);

    /** 과정 5 **/
    /* overwrite coins in kr with H(c) */
    hash_h(kr + KYBER_SYMBYTES, ct, KYBER_CIPHERTEXTBYTES);
    /* hash concatenation of pre-k and H(c) to k */
    kdf(ss, kr, 2 * KYBER_SYMBYTES);
}

__device__ void PQCLEAN_KYBER512_CLEAN_polyvec_decompress(polyvec* r, const uint8_t a[KYBER_POLYVECCOMPRESSEDBYTES]) {
    unsigned int i, j, k;

    uint16_t t[4];
    for (i = 0; i < KYBER_K; i++) {
        for (j = 0; j < KYBER_N / 4; j++) {
            t[0] = (a[0] >> 0) | ((uint16_t)a[1] << 8);
            t[1] = (a[1] >> 2) | ((uint16_t)a[2] << 6);
            t[2] = (a[2] >> 4) | ((uint16_t)a[3] << 4);
            t[3] = (a[3] >> 6) | ((uint16_t)a[4] << 2);
            a += 5;

            for (k = 0; k < 4; k++) {
                r->vec[i].coeffs[4 * j + k] = ((uint32_t)(t[k] & 0x3FF) * KYBER_Q + 512) >> 10;
            }
        }
    }
}

__device__ void PQCLEAN_KYBER512_CLEAN_poly_decompress(poly* r, const uint8_t a[KYBER_POLYCOMPRESSEDBYTES]) {
    size_t i;

    for (i = 0; i < KYBER_N / 2; i++) {
        r->coeffs[2 * i + 0] = (((uint16_t)(a[0] & 15) * KYBER_Q) + 8) >> 4;
        r->coeffs[2 * i + 1] = (((uint16_t)(a[0] >> 4) * KYBER_Q) + 8) >> 4;
        a += 1;
    }
}

__device__ static void unpack_ciphertext(polyvec* b, poly* v, const uint8_t c[KYBER_INDCPA_BYTES]) {
    PQCLEAN_KYBER512_CLEAN_polyvec_decompress(b, c);
    PQCLEAN_KYBER512_CLEAN_poly_decompress(v, c + KYBER_POLYVECCOMPRESSEDBYTES);
}
__device__ static void unpack_sk(polyvec* sk, const uint8_t packedsk[KYBER_INDCPA_SECRETKEYBYTES]) {
    PQCLEAN_KYBER512_CLEAN_polyvec_frombytes(sk, packedsk);
}

__device__ int PQCLEAN_KYBER512_CLEAN_verify(const uint8_t* a, const uint8_t* b, size_t len) {
    size_t i;
    uint8_t r = 0;

    for (i = 0; i < len; i++) {
        r |= a[i] ^ b[i];
    }

    return ((~(uint64_t)r) + 1) >> 63;
}

__device__ void PQCLEAN_KYBER512_CLEAN_cmov(uint8_t* r, const uint8_t* x, size_t len, uint8_t b) {
    size_t i;

    b = -b;
    for (i = 0; i < len; i++) {
        r[i] ^= b & (r[i] ^ x[i]);
    }
}
__device__ void PQCLEAN_KYBER512_CLEAN_indcpa_dec(uint8_t m[KYBER_INDCPA_MSGBYTES], const uint8_t c[KYBER_INDCPA_BYTES], const uint8_t sk[KYBER_INDCPA_SECRETKEYBYTES])
{
    polyvec b, skpv;
    poly v, mp;

    unpack_ciphertext(&b, &v, c);
    unpack_sk(&skpv, sk);

    PQCLEAN_KYBER512_CLEAN_polyvec_ntt(&b);
    PQCLEAN_KYBER512_CLEAN_polyvec_basemul_acc_montgomery(&mp, &skpv, &b);
    PQCLEAN_KYBER512_CLEAN_poly_invntt_tomont(&mp);

    PQCLEAN_KYBER512_CLEAN_poly_sub(&mp, &v, &mp);
    PQCLEAN_KYBER512_CLEAN_poly_reduce(&mp);

    PQCLEAN_KYBER512_CLEAN_poly_tomsg(m, &mp);
}
__device__ void PQCLEAN_KYBER512_CLEAN_crypto_kem_dec(uint8_t* ss, const uint8_t* ct, const uint8_t* sk)
{
    size_t i;
    int fail;
    uint8_t buf[2 * KYBER_SYMBYTES];
    /* Will contain key, coins */
    uint8_t kr[2 * KYBER_SYMBYTES];
    uint8_t cmp[KYBER_CIPHERTEXTBYTES];
    const uint8_t* pk = sk + KYBER_INDCPA_SECRETKEYBYTES;

    PQCLEAN_KYBER512_CLEAN_indcpa_dec(buf, ct, sk);

    /* Multitarget countermeasure for coins + contributory KEM */
    for (i = 0; i < KYBER_SYMBYTES; i++) {
        buf[KYBER_SYMBYTES + i] = sk[KYBER_SECRETKEYBYTES - 2 * KYBER_SYMBYTES + i];
    }
    hash_g(kr, buf, 2 * KYBER_SYMBYTES);

    /* coins are in kr+KYBER_SYMBYTES */
    PQCLEAN_KYBER512_CLEAN_indcpa_enc(cmp, buf, pk, kr + KYBER_SYMBYTES);

    fail = PQCLEAN_KYBER512_CLEAN_verify(ct, cmp, KYBER_CIPHERTEXTBYTES);

    /* overwrite coins in kr with H(c) */
    hash_h(kr + KYBER_SYMBYTES, ct, KYBER_CIPHERTEXTBYTES);

    /* Overwrite pre-k with z on re-encryption failure */
    PQCLEAN_KYBER512_CLEAN_cmov(kr, sk + KYBER_SECRETKEYBYTES - KYBER_SYMBYTES, KYBER_SYMBYTES, (uint8_t)fail);

    /* hash concatenation of pre-k and H(c) to k */
    kdf(ss, kr, 2 * KYBER_SYMBYTES);
}

__global__ void GPU_Kyber(uint8_t* pk, uint8_t* sk, uint8_t* ct, uint8_t* ss, uint8_t* ss2)
{
    int tid;

    tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    PQCLEAN_KYBER512_CLEAN_crypto_kem_keypair(pk + PQCLEAN_KYBER512_CLEAN_CRYPTO_PUBLICKEYBYTES * tid, sk + PQCLEAN_KYBER512_CLEAN_CRYPTO_SECRETKEYBYTES * tid);			// KEY 생성 + KEY 교환
    PQCLEAN_KYBER512_CLEAN_crypto_kem_enc(ct + KYBER_CIPHERTEXTBYTES * tid, ss + 32 * tid, pk + PQCLEAN_KYBER512_CLEAN_CRYPTO_PUBLICKEYBYTES * tid);
    PQCLEAN_KYBER512_CLEAN_crypto_kem_dec(ss2 + 32 * tid, ct + KYBER_CIPHERTEXTBYTES * tid, sk + PQCLEAN_KYBER512_CLEAN_CRYPTO_SECRETKEYBYTES * tid);
}



void test_Kyber(uint64_t blocksize, uint64_t threadsize)
{
    uint8_t* pk = NULL;
    uint8_t* sk = NULL;
    uint8_t* ct = NULL;
    uint8_t* ss = NULL;
    uint8_t* ss2 = NULL;

    cudaEvent_t start, stop;
    float elapsed_time_ms = 0.0f;

    pk = (uint8_t*)malloc(PQCLEAN_KYBER512_CLEAN_CRYPTO_PUBLICKEYBYTES * blocksize * threadsize); //PQCLEAN_KYBER512_CLEAN_CRYPTO_PUBLICKEYBYTES -> (2 * 384 + 32) * blocksize * threadsize
    sk = (uint8_t*)malloc(PQCLEAN_KYBER512_CLEAN_CRYPTO_SECRETKEYBYTES * blocksize * threadsize); //PQCLEAN_KYBER512_CLEAN_CRYPTO_SECRETKEYBYTES -> ((2 * 384) + (2 * 384 + 32) + (2 * 32)) * blocksize * threadsize
    ct = (uint8_t*)malloc(KYBER_CIPHERTEXTBYTES * blocksize * threadsize);
    ss = (uint8_t*)malloc(32 * blocksize * threadsize);
    ss2 = (uint8_t*)malloc(32 * blocksize * threadsize);

    uint8_t* GPU_pk;
    uint8_t* GPU_sk;
    uint8_t* GPU_ct;
    uint8_t* GPU_ss;
    uint8_t* GPU_ss2;

    cudaMalloc((void**)&GPU_pk, PQCLEAN_KYBER512_CLEAN_CRYPTO_PUBLICKEYBYTES * blocksize * threadsize);
    cudaMalloc((void**)&GPU_sk, PQCLEAN_KYBER512_CLEAN_CRYPTO_SECRETKEYBYTES * blocksize * threadsize);
    cudaMalloc((void**)&GPU_ct, KYBER_CIPHERTEXTBYTES * blocksize * threadsize);
    cudaMalloc((void**)&GPU_ss, 32 * blocksize * threadsize);
    cudaMalloc((void**)&GPU_ss2, 32 * blocksize * threadsize);


    printf("\nStart...\n");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < N_TIMES; i++)
        GPU_Kyber << <blocksize, threadsize >> > (GPU_pk, GPU_sk, GPU_ct, GPU_ss, GPU_ss2);

    printf("%d\n", N_TIMES);

    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    elapsed_time_ms /= N_TIMES;

    printf("elapsed time: %4.2f ms\n\n", elapsed_time_ms);
    
    elapsed_time_ms = (1000 / elapsed_time_ms) * blocksize * threadsize;
    printf("Grid : %ld, Block : %ld, Performance : %4.2f Kyber/s\n", blocksize, threadsize, elapsed_time_ms);

    cudaMemcpy(ss, GPU_ss, 32 * blocksize * threadsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ss2, GPU_ss2, 32 * blocksize * threadsize, cudaMemcpyDeviceToHost);

    if (!memcmp(ss, ss2, 32 * blocksize * threadsize))
        printf("\n\nSuccess!\n\n");
    else
        printf("\n\nFail\n\n");

    getchar();
    getchar();

    printf("ss_0: = \n");
    for (int i = 0; i < 32 * blocksize * threadsize; i++) {
        printf("%02X ", ss[i]);

        if ((i + 1) % 32 == 0)
        {
            printf("\nss_%d: = \n", (i + 1)/32);
            printf("\n");
        }
    }
    printf("\n\n");

    printf("ss2_0: = \n");
    for (int i = 0; i < 32 * blocksize * threadsize; i++) {
        printf("%02X ", ss2[i]);

        if ((i + 1) % 32 == 0)
        {
            printf("\nss2_%d: = \n", (i + 1) / 32);
            printf("\n");
        }
    }

    cudaFree(GPU_pk);
    cudaFree(GPU_sk);
    cudaFree(GPU_ct);
    cudaFree(GPU_ss);
    cudaFree(GPU_ss2);
    free(pk);
    free(sk);
    free(ct);
    free(ss);
    free(ss2);
}

int main()
{
    test_Kyber(256, 256);

    return 0;
}