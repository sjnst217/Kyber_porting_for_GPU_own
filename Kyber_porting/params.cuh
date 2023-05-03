#pragma once
#ifndef PQCLEAN_KYBER512_CLEAN_PARAMS_H
#define PQCLEAN_KYBER512_CLEAN_PARAMS_H

#define MONT (-1044) // 2^16 mod q
#define QINV (-3327) // q^-1 mod 2^16

#define KYBER_N 256																							// KYBER_N = 256													// -> ���׽��� ����, �� 256�� ���׽�
#define KYBER_Q 3329																						// KYBER_Q = 3329													// -> ���׽��� ���, �� ���׽��� �ִ� ����� ũ�Ⱑ 3328�̶�� �ǹ� -> �� ���� 12bit�� ��Ÿ�� �� ����

#define KYBER_SYMBYTES 32   /* size in bytes of hashes, and seeds */										// KYBER_SYMBYTES = 32
#define KYBER_SSBYTES  32   /* size in bytes of shared key */												// KYBER_SSBYTES =	32

#define KYBER_POLYBYTES     384																				// KYBER_POLYBYTES = 384											// -> 2���� 12��Ʈ ����� 3����Ʈ�� ����� �� �ֱ� ������, ���ڵ� ������ ���� 256 * 12/8 ����Ʈ������ ������ �� �ִ�. �� �ϳ��� ���׽��� �����ϴ� �޸��� ǥ��
#define KYBER_POLYVECBYTES  (KYBER_K * KYBER_POLYBYTES)														// KYBER_POLYVECBYTES = 2 * 384

#define KYBER_K 2																							// KYBER_K = 2														// -> module�� ����, �� 2 * 2 ũ���� ��İ������� ��Ÿ��
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
