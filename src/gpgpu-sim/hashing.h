// author: Mahmoud Khairy, (Purdue Univ)
// email: abdallm@purdue.edu

/*
 * [한국어 설명] 메모리 뱅크/셋 인덱싱 해시 함수 헤더 (hashing.h)
 *
 * === 파일의 역할 ===
 * GPU DRAM 뱅크 및 L2 캐시 셋에 메모리 요청을 분산시키기 위한 해시 함수 3종을
 * 선언한다. 기본 순차 인덱싱은 stride 패턴 접근 시 특정 뱅크에 요청이 몰리는
 * 뱅크 충돌(bank conflict) 문제를 일으킨다. 이 파일의 함수들은 주소 비트를
 * 비선형 방식으로 치환하여 뱅크/셋 접근을 균등하게 분산시킨다. 선택된 해시
 * 함수는 gpgpusim.config의 memory_partition_indexing 옵션으로 결정된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * addrdec.cc의 linear_to_raw_address_translation()이 물리 주소를 채널·뱅크·행·열로
 * 분해하는 과정에서 이 함수들을 호출한다. 즉, 메모리 요청(mem_fetch)이 DRAM 컨트롤러
 * (dram.cc)로 전달되기 직전, 주소 디코딩 단계에서 뱅크 인덱스를 결정하는 데 사용된다.
 * 실행 컨텍스트: 호스트 유저스페이스 — 매 메모리 접근마다(L2 미스 또는 직접 DRAM 접근)
 * 사이클 루프 내에서 호출된다.
 *
 * === 타 모듈과의 연결 ===
 * - 이 파일이 의존하는 모듈: abstract_hardware_model.h (new_addr_type 타입 정의),
 *   gpu-cache.h (캐시 관련 상수), option_parser.h (설정 파싱 지원)
 * - 이 파일에 의존하는 모듈: addrdec.cc (linear_to_raw_address_translation에서 호출)
 * - 데이터 흐름: 물리 주소의 상위 비트(higher_bits)와 현재 인덱스(index)를 받아
 *   새로운 뱅크/셋 인덱스를 반환한다. 반환값은 DRAM 뱅크 또는 L2 셋 번호로 사용된다.
 * - 설정 연동: gpgpusim.config의 memory_partition_indexing 값에 따라 addrdec.cc가
 *   ipoly(2), bitwise(1), PAE(3) 중 하나를 선택하여 호출한다.
 *
 * === 주요 함수/구조체 요약 ===
 * - ipoly_hash_function: IPOLY(37) GF(2) 다항식 기반 해시 — 2^n stride 충돌 완전 제거,
 *   16/32/64 뱅크 지원 (ISCA 1991, IEEE TPDS 2017 기반)
 * - bitwise_hash_function: XOR 기반 단순 비트와이즈 해시 — 오버헤드 최소, 불규칙 stride에 약함
 * - PAE_hash_function: 페이지 주소 엔트로피 기반 해시 — 32 뱅크 전용, 전력 효율 최적화 목적
 */

/* [한국어] assert.h, stdio.h, stdlib.h — 해시 함수 내부의 assert() 및 표준 I/O/메모리 함수 사용을 위한 포함 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
/* [한국어] option_parser.h — 설정 파라미터 파싱 지원 (memory_partition_indexing 옵션 등) */
#include "../option_parser.h"

#ifndef HASHING_H
#define HASHING_H

/* [한국어] abstract_hardware_model.h — new_addr_type (GPU 메모리 주소 타입, 보통 unsigned long long) 정의 포함 */
#include "../abstract_hardware_model.h"
/* [한국어] gpu-cache.h — 캐시 셋 수, 뱅크 수 등 캐시 파라미터 관련 상수 및 타입 포함 */
#include "gpu-cache.h"

/*
 * [한국어]
 * ipoly_hash_function - IPOLY GF(2) 다항식 기반 비선형 뱅크/셋 인덱스 해시 함수
 *
 * @higher_bits: 캐시 라인/뱅크 선택에 사용되는 물리 주소의 상위 비트들.
 *               뱅크 수와 캐시 라인 크기를 고려하여 addrdec.cc에서 추출된 값이다.
 * @index: 단순 순차 방식(modulo)으로 계산한 초기 뱅크/셋 인덱스.
 *         이 값을 해시 결과로 치환하여 뱅크 충돌을 줄인다.
 * @bank_set_num: 뱅크(또는 L2 셋) 총 개수. 16, 32, 64 중 하나여야 한다.
 * @return: 재계산된 뱅크/셋 인덱스 (0 이상 bank_set_num 미만).
 *
 * IPOLY(Irreducible Polynomial) 해시는 GF(2)(이진 유한체) 위의 기약 다항식 모듈러
 * 연산을 기반으로 한다. 2^n 스트라이드 접근 패턴에서 충돌이 수학적으로 보장되지 않는
 * 단순 XOR 해시와 달리, IPOLY는 모든 2^n 스트라이드에 대해 완전한 충돌 회피를 보장한다.
 * GPGPU 애플리케이션에서 코얼레스싱(coalescing) 후에도 stride 패턴이 매우 흔하므로
 * 이 함수는 메모리 대역폭 활용을 크게 개선한다.
 * - 16 뱅크: IPOLY(5) 방정식 사용
 * - 32 뱅크: IPOLY(37) 방정식 사용
 * - 64 뱅크: IPOLY(67) 방정식 사용
 *
 * 호출 체인: addrdec.cc::linear_to_raw_address_translation() → [ipoly_hash_function]
 */
unsigned ipoly_hash_function(new_addr_type higher_bits, unsigned index,
                             unsigned bank_set_num);

/*
 * [한국어]
 * bitwise_hash_function - XOR 기반 비트와이즈 뱅크/셋 인덱스 해시 함수
 *
 * @higher_bits: 물리 주소의 상위 비트들 (뱅크 수 비트만큼 하위 비트를 추출하여 사용).
 * @index: 초기 순차 뱅크/셋 인덱스.
 * @bank_set_num: 뱅크(또는 셋) 총 개수. 2의 거듭제곱이어야 한다.
 * @return: index XOR (higher_bits & (bank_set_num - 1)) — 비트와이즈 XOR로 치환된 인덱스.
 *
 * 가장 단순한 해시 방식으로 구현 오버헤드가 최소이다. index에 higher_bits의 하위
 * log2(bank_set_num) 비트를 XOR하여 인덱스를 분산시킨다. 2^n 스트라이드에서는 IPOLY보다
 * 충돌 회피 능력이 약하지만, 불규칙한 접근 패턴에서는 간단하면서도 효과적이다.
 *
 * 호출 체인: addrdec.cc::linear_to_raw_address_translation() → [bitwise_hash_function]
 */
unsigned bitwise_hash_function(new_addr_type higher_bits, unsigned index,
                               unsigned bank_set_num);

/*
 * [한국어]
 * PAE_hash_function - 페이지 주소 엔트로피(Page Address Entropy) 기반 해시 함수
 *
 * @higher_bits: 물리 주소의 상위 비트들 (페이지 내 및 페이지 간 비트를 혼합).
 * @index: 초기 순차 뱅크/셋 인덱스.
 * @bank_set_num: 뱅크(또는 셋) 총 개수. 현재는 32만 지원된다. 그 외 값이면 assert로 중단.
 * @return: 페이지 주소 비트와 뱅크 비트를 혼합하여 재계산된 인덱스 (32 미만).
 *
 * PAE 방식은 페이지 주소 비트와 기존 뱅크 비트를 무작위로 선택하여 XOR 조합함으로써
 * 뱅크 충돌을 줄이는 동시에 DRAM 행 버퍼 지역성(row buffer locality)을 유지하여
 * 전력 효율을 최적화한다. Liu et al. "Get Out of the Valley" 논문의 아이디어를 기반으로 한다.
 * 현재 구현은 32 뱅크만 지원하며, 다른 뱅크 수 요청 시 assert로 시뮬레이터가 종료된다.
 *
 * 호출 체인: addrdec.cc::linear_to_raw_address_translation() → [PAE_hash_function]
 */
unsigned PAE_hash_function(new_addr_type higher_bits, unsigned index,
                           unsigned bank_set_num);

#endif
