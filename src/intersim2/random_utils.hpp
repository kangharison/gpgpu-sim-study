// $Id: random_utils.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 * [한국어 설명] 무작위 수 생성 유틸리티 헤더 (random_utils.hpp)
 *
 * === 파일의 역할 ===
 * intersim2 시뮬레이터 전반에서 사용하는 결정론적(deterministic) 무작위 수
 * 생성 인터페이스를 제공한다. Knuth의 RANARRAY 알고리즘 기반 정수 RNG(ran_*)와
 * 부동소수점 RNG(ranf_*)를 공통 래퍼 함수들로 감싸, 시뮬레이션 재현성(reproducibility)을
 * 보장한다. 동일 시드(seed)를 설정하면 항상 동일한 난수 시퀀스가 생성되므로,
 * NoC 트래픽 패턴·중재(arbitration)·라우팅 결정 등의 재현 실험에 필수적이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 트래픽 생성기(TrafficManager, injection process)와 중재기(Allocator) 등
 * intersim2 컴포넌트들이 무작위 수가 필요할 때 이 헤더의 inline 함수들을
 * 직접 호출한다. 실제 RNG 상태는 rng_wrapper.cpp(정수)와
 * rng_double_wrapper.cpp(부동소수점)에 있는 전역 C 변수가 보유한다.
 *
 * 호출 체인:
 *   TrafficManager / Allocator / TrafficPattern
 *     → RandomSeed() / RandomInt() / RandomFloat()  [이 파일]
 *         → ran_start() / ran_next() / ranf_start() / ranf_next()
 *             → rng.c / rng-double.c (Knuth RANARRAY)
 *
 * 실행 컨텍스트: 호스트 CPU 단일 스레드 (GPGPU-Sim 시뮬레이션 루프).
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - rng_wrapper.cpp     : ran_start(), ran_next() 구현 (정수 RNG 상태 보유).
 *   - rng_double_wrapper.cpp : ranf_start(), ranf_next() 구현 (실수 RNG 상태 보유).
 * 이 파일을 사용하는 모듈:
 *   - trafficmanager.cc   : 패킷 주입 시 무작위 목적지/타이밍 결정에 사용.
 *   - allocator/          : iSLIP 등 무작위 초기 우선순위 설정에 사용.
 *   - traffic/            : 각종 트래픽 패턴 생성기에서 목적지 랜덤 선택에 사용.
 * 데이터 흐름:
 *   시뮬레이터 초기화 시 RandomSeed(seed) 1회 호출 → 이후 매 사이클
 *   RandomInt()/RandomFloat()으로 값을 소비.
 *
 * === 주요 함수/구조체 요약 ===
 * - RandomSeed(seed)        : 정수·실수 RNG 모두 동일 시드로 초기화 (재현성 보장).
 * - RandomIntLong()         : [0, LONG_MAX] 범위 부호 없는 정수 반환.
 * - RandomInt(max)          : [0, max] 범위 정수 균등 분포 반환.
 * - RandomFloat()           : [0.0, 1.0] 범위 부동소수점 균등 분포 반환.
 * - RandomFloat(max)        : [0.0, max] 범위 부동소수점 균등 분포 반환.
 */

#ifndef _RANDOM_UTILS_HPP_
#define _RANDOM_UTILS_HPP_

// [한국어] Knuth RANARRAY 정수 RNG 외부 함수 선언 — 구현은 rng_wrapper.cpp (rng.c 포함)
// interface to Knuth's RANARRAY RNG
void   ran_start(long seed); // [한국어] 정수 RNG 상태를 seed 값으로 초기화. 동일 seed → 동일 시퀀스 보장.
long   ran_next( );          // [한국어] 정수 RNG에서 다음 난수 반환. 내부적으로 ran_arr_next() 호출.

// [한국어] Knuth RANARRAY 부동소수점 RNG 외부 함수 선언 — 구현은 rng_double_wrapper.cpp (rng-double.c 포함)
void   ranf_start(long seed); // [한국어] 실수 RNG 상태를 seed 값으로 초기화. 정수 RNG와 독립된 상태를 유지.
double ranf_next( );          // [한국어] 실수 RNG에서 [0.0, 1.0) 범위 부동소수점 난수 반환.

/*
 * [한국어]
 * RandomSeed — 정수 및 실수 RNG를 동일 시드로 동시 초기화
 *
 * @seed  : 시뮬레이션 재현성을 위한 시드 값. gpgpusim.config의 설정에서 전달됨.
 * @return: void
 *
 * 두 개의 독립 RNG(정수용 ran_*, 실수용 ranf_*)를 동일한 seed로 초기화한다.
 * 시뮬레이터 초기화 시 단 한 번 호출되며, 이후 RandomInt()/RandomFloat()이
 * 각각 독립적인 시퀀스를 소비한다. 동일 seed에서는 항상 동일한 난수 시퀀스가
 * 생성되므로 실험 재현성이 보장된다.
 *
 * 실행 컨텍스트: 시뮬레이터 초기화 단계, 단일 스레드.
 * 호출 체인: TrafficManager::TrafficManager() 또는 gpgpusim_entrypoint
 *             → RandomSeed()
 */
inline void RandomSeed( long seed ) {
  ran_start( seed );  // [한국어] 정수 RNG(ran_arr_*)의 내부 상태 배열을 seed 기반으로 초기화
  ranf_start( seed ); // [한국어] 실수 RNG(ranf_arr_*)의 내부 상태 배열을 동일 seed로 초기화
                       //          정수와 실수 RNG가 독립 상태이므로 서로 간섭하지 않음
}

/*
 * [한국어]
 * RandomIntLong — 부호 없는 long 범위의 정수 난수 반환
 *
 * @return: [0, LONG_MAX] 범위의 unsigned long 난수.
 *           내부적으로 ran_next()의 long 반환값을 unsigned long으로 재해석.
 *
 * 넓은 범위의 정수 난수가 필요할 때 사용. 일반적으로 RandomInt(max)가 더
 * 자주 쓰이며, 이 함수는 범위 제한 없이 원시(raw) 난수 값이 필요한 경우에 사용.
 *
 * 실행 컨텍스트: 트래픽 생성기 / 중재기 내부, 매 사이클 필요 시 호출.
 * 호출 체인: TrafficPattern / Allocator → RandomIntLong() → ran_next()
 */
inline unsigned long RandomIntLong( ) {
  return ran_next( ); // [한국어] ran_next()의 long 값을 unsigned long으로 암시적 변환하여 반환
}

/*
 * [한국어]
 * RandomInt — [0, max] 범위의 균등 분포 정수 난수 반환
 *
 * @max   : 반환값의 상한 (포함). max=0이면 항상 0 반환. 음수 불가.
 * @return: [0, max] 범위의 int 난수 (균등 분포 근사).
 *
 * ran_next() % (max+1)로 구현되므로 엄밀한 균등 분포는 아니다(modulo bias).
 * 그러나 시뮬레이터 용도에서는 통계적으로 충분히 균일하다고 간주한다.
 * 주로 무작위 목적지 노드 선택, 중재 우선순위 결정 등에 사용.
 *
 * 실행 컨텍스트: TrafficPattern::dest(), Allocator::arbitrate() 등, 매 사이클 가능.
 * 호출 체인: UniformRandomTrafficPattern::dest() → RandomInt(num_nodes-1) → ran_next()
 */
// Returns a random integer in the range [0,max]
inline int RandomInt( int max ) {
  return ( ran_next( ) % (max+1) ); // [한국어] 정수 RNG에서 난수 획득 후 (max+1)로 나눈 나머지로 [0,max] 범위 투영.
                                     //          modulo bias가 있으나 시뮬레이션 용도에서는 허용 수준.
}

/*
 * [한국어]
 * RandomFloat (인자 없음) — [0.0, 1.0] 범위의 균등 부동소수점 난수 반환
 *
 * @return: [0.0, 1.0] 범위의 double 난수 (Knuth RANARRAY 기반, 고품질 균등 분포).
 *
 * ranf_next()는 Knuth의 RANARRAY 알고리즘으로 [0.0, 1.0) 구간의 고품질
 * 부동소수점 난수를 생성한다. 부하 생성(load generation) 시 패킷 주입 여부를
 * 확률적으로 결정하거나, 포아송 프로세스 모델링 등에 사용된다.
 *
 * 실행 컨텍스트: TrafficManager::Step() 내 주입 결정, 매 사이클 호출 가능.
 * 호출 체인: TrafficManager → RandomFloat() → ranf_next()
 */
// Returns a random floating-point value in the rage [0,1]
inline double RandomFloat(  ) {
  return ranf_next( ); // [한국어] 실수 RNG에서 [0.0, 1.0) 범위 부동소수점 난수 반환.
                        //          반환값이 정확히 1.0이 될 가능성은 알고리즘상 없음.
}

/*
 * [한국어]
 * RandomFloat (max 인자) — [0.0, max] 범위의 균등 부동소수점 난수 반환
 *
 * @max   : 반환값의 상한 배율 (양의 실수). 음수나 0이면 의미 없는 값 반환.
 * @return: [0.0, max] 범위의 double 난수.
 *
 * ranf_next()의 [0.0, 1.0) 값에 max를 곱하여 원하는 스케일로 조정한다.
 * 예를 들어 RandomFloat(100.0)은 [0.0, 100.0) 범위의 난수를 반환하며,
 * 확률을 백분율로 비교할 때 유용하다.
 *
 * 실행 컨텍스트: TrafficManager 또는 트래픽 패턴 내부, 매 사이클 호출 가능.
 * 호출 체인: TrafficManager → RandomFloat(load) → ranf_next()
 */
// Returns a random floating-point value in the rage [0,max]
inline double RandomFloat( double max ) {
  return ( ranf_next( ) * max ); // [한국어] [0.0, 1.0) 부동소수점 난수에 max를 곱하여 [0.0, max) 범위로 스케일 조정
}

#endif  // [한국어] _RANDOM_UTILS_HPP_ 헤더 가드 종료 — 중복 include 방지
