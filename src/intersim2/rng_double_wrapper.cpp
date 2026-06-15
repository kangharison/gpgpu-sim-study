// $Id: rng_double_wrapper.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] 부동소수점 난수 생성기 C 래퍼 (rng_double_wrapper.cpp)
 *
 * === 파일의 역할 ===
 * Knuth의 RANARRAY 알고리즘을 double(부동소수점)로 구현한 C 소스 파일
 * (rng-double.c)을 C++ 빌드 환경에 통합하기 위한 최소한의 래퍼이다.
 * rng_wrapper.cpp가 정수 RNG를 담당하는 것과 대칭적으로, 이 파일은 실수
 * 난수(ranf_*)를 담당한다. 두 파일의 RNG 상태는 완전히 독립되어 있으며,
 * 서로 다른 내부 배열을 사용한다. random_utils.hpp의 RandomFloat() 계열
 * 함수가 이 파일의 ranf_next()를 통해 난수를 소비한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * random_utils.hpp에서 선언된 ranf_start()/ranf_next() 함수를 구현하는
 * TU(Translation Unit)이다. rng_wrapper.cpp와 함께 intersim2의 전체
 * 난수 생성 인프라를 구성한다.
 *
 * 호출 체인:
 *   random_utils.hpp::RandomSeed(seed) → ranf_start(seed)  [rng-double.c 내 구현]
 *   random_utils.hpp::RandomFloat()    → ranf_next()       [이 파일]
 *                                           → ranf_arr_next() [rng-double.c 내 매크로]
 *
 * 실행 컨텍스트: 호스트 CPU 단일 스레드 (GPGPU-Sim 시뮬레이션 루프).
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - rng-double.c : Knuth RANARRAY 부동소수점 RNG C 구현. ranf_u[] 배열,
 *                    ranf_start(), ranf_arr_next() 등 포함.
 *                    이 파일에 #include로 직접 삽입됨.
 * 이 파일을 사용하는 모듈:
 *   - random_utils.hpp : ranf_next() 선언. 헤더 포함자 모두가 간접 사용.
 *   - trafficmanager.cc : RandomFloat()을 통해 패킷 주입 확률 결정에 간접 사용.
 * 데이터 흐름:
 *   시드 → ranf_start() → 전역 실수 RNG 배열(ranf_u[]) 초기화
 *   → ranf_arr_next() → [0.0, 1.0) 난수 소비 → RandomFloat()로 노출.
 *
 * === 주요 함수/구조체 요약 ===
 * - #define main rng_double_main : rng-double.c의 main()을 rng_double_main()으로
 *                                   rename하여 링커 충돌 방지.
 * - #include "rng-double.c"      : Knuth RANARRAY double RNG 전체 구현을 이 TU에 통합.
 * - ranf_next()                   : ranf_arr_next() 호출 래퍼 — C++에서 호출 가능한
 *                                   진입점. random_utils.hpp에 선언됨.
 */

#define main rng_double_main  // [한국어] rng-double.c 내의 main() 함수를 rng_double_main()으로 rename.
                               // rng_wrapper.cpp가 main을 rng_main으로 rename한 것과 동일한 패턴.
                               // 두 래퍼 파일이 각각 다른 이름으로 rename하므로 심볼 충돌 없이
                               // 하나의 바이너리에 공존할 수 있다.

#include "rng-double.c"        // [한국어] Knuth RANARRAY 부동소수점 RNG 구현을 이 TU에 직접 삽입.
                                // ranf_u[] 배열, ranf_start(), ranf_arr_next() 등 모든 상태와
                                // 함수가 이 컴파일 단위 내에 포함됨. rng.c와 완전히 독립된
                                // 별도 상태를 유지하므로 정수/실수 시퀀스가 서로 간섭하지 않음.

/*
 * [한국어]
 * ranf_next — Knuth RANARRAY 부동소수점 RNG에서 다음 난수 반환
 *
 * @return: double 타입 [0.0, 1.0) 범위 부동소수점 난수.
 *           ranf_arr_next()는 rng-double.c 내의 매크로로, RNG 배열에서
 *           다음 원소를 소비하고 필요하면 배열을 재생성(refill)한다.
 *
 * 이 함수는 C++ 코드에서 호출 가능한 진입점으로,
 * rng-double.c 내부의 ranf_arr_next()를 C++ linkage로 노출한다.
 * random_utils.hpp의 RandomFloat() 계열이 이 함수를 호출한다.
 *
 * 반환값이 정확히 1.0이 되는 경우는 알고리즘상 없다(개구간 상한).
 * 반환값을 확률 임계값과 비교할 때 이 점을 고려해야 한다.
 *
 * 실행 컨텍스트: 매 사이클 RandomFloat() 호출 시, 단일 스레드.
 * 호출 체인:
 *   RandomFloat() → ranf_next() → ranf_arr_next()  [rng-double.c 내 매크로]
 */
double ranf_next( )
{
  return ranf_arr_next( ); // [한국어] rng-double.c에서 #define된 ranf_arr_next() 매크로/함수 호출.
                            // 내부 실수 RNG 배열(ranf_u[])에서 다음 [0.0, 1.0) 값을 꺼내며,
                            // 배열이 소진되면 ranf_arr_cycle()을 호출하여 새 시퀀스를 생성한다.
}
