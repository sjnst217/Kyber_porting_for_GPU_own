#pragma once
#ifndef PQCLEAN_KYBER512_CLEAN_PARAMS_H
#define PQCLEAN_KYBER512_CLEAN_PARAMS_H

#define MONT (-1044) // 2^16 mod q
#define QINV (-3327) // q^-1 mod 2^16

#define KYBER_N 256																							// KYBER_N = 256													// -> 다항식의 차수, 즉 256차 다항식
#define KYBER_Q 3329																						// KYBER_Q = 3329													// -> 다항식의 계수, 즉 다항식의 최대 계수의 크기가 3328이라는 의미 -> 이 수는 12bit로 나타낼 수 있음

#define KYBER_SYMBYTES 32   /* size in bytes of hashes, and seeds */										// KYBER_SYMBYTES = 32
#define KYBER_SSBYTES  32   /* size in bytes of shared key */												// KYBER_SSBYTES =	32

#define KYBER_POLYBYTES     384																				// KYBER_POLYBYTES = 384											// -> 2개의 12비트 계수는 3바이트에 저장될 수 있기 때문에, 인코딩 과정을 거쳐 256 * 12/8 바이트만으로 전송할 수 있다. 즉 하나의 다항식을 저장하는 메모리의 표현
#define KYBER_POLYVECBYTES  (KYBER_K * KYBER_POLYBYTES)														// KYBER_POLYVECBYTES = 2 * 384

#define KYBER_K 2																							// KYBER_K = 2														// -> module의 개수, 즉 2 * 2 크기의 행렬곱셈임을 나타냄
#define KYBER_ETA1 3																						// KYBER_ETA1 = 3													// -> 
#define KYBER_POLYCOMPRESSEDBYTES    128																	// KYBER_POLYCOMPRESSEDBYTES = 128
#define KYBER_POLYVECCOMPRESSEDBYTES (KYBER_K * 320)														// KYBER_POLYVECCOMPRESSEDBYTES = 2 * 320

#define KYBER_ETA2 2																						// KYBER_ETA2 = 2

#define KYBER_INDCPA_MSGBYTES       (KYBER_SYMBYTES)														// KYBER_INDCPA_MSGBYTES = 32
#define KYBER_INDCPA_PUBLICKEYBYTES (KYBER_POLYVECBYTES + KYBER_SYMBYTES)									// KYBER_INDCPA_PUBLICKEYBYTES = 2 * 384 + 32
#define KYBER_INDCPA_SECRETKEYBYTES (KYBER_POLYVECBYTES)													// KYBER_INDCPA_SECRETKEYBYTES = 2 * 384
#define KYBER_INDCPA_BYTES          (KYBER_POLYVECCOMPRESSEDBYTES + KYBER_POLYCOMPRESSEDBYTES)				// KYBER_INDCPA_BYTES = (2 * 320) + 128

#define KYBER_PUBLICKEYBYTES  (KYBER_INDCPA_PUBLICKEYBYTES)													// KYBER_PUBLICKEYBYTES = 2 * 384 + 32
/* 32 bytes of additional space to save H(pk) */
#define KYBER_SECRETKEYBYTES  (KYBER_INDCPA_SECRETKEYBYTES + KYBER_INDCPA_PUBLICKEYBYTES + 2*KYBER_SYMBYTES)// KYBER_SECRETKEYBYTES = 2 * 384 + (2 * 384 + 320) + 2 * 32
#define KYBER_CIPHERTEXTBYTES (KYBER_INDCPA_BYTES)															// KYBER_CIPHERTEXTBYTES = (2 * 320) + 128

#endif
