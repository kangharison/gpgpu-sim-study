// $Id: rng_wrapper.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] 정수 난수 생성기 C 래퍼 (rng_wrapper.cpp)
 *
 * === 파일의 역할 ===
 * Knuth의 RANARRAY 알고리즘을 구현한 C 소스 파일(rng.c)을 C++ 빌드 환경에
 * 통합하기 위한 최소한의 래퍼(wrapper)이다. rng.c는 원본 그대로 #include로
 * 삽입하되, main 함수 충돌을 방지하기 위해 #define main rng_main 트릭을
 * 사용한다. C++ 인터페이스로는 ran_next()만 노출하며, 초기화 함수
 * ran_start()는 rng.c 내부에서 직접 링크된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * random_utils.hpp에서 선언된 ran_start()/ran_next() 함수를 구현하는 TU
 * (Translation Unit)이다. 부동소수점 RNG는 rng_double_wrapper.cpp가 담당하며,
 * 두 파일은 독립적인 전역 상태를 유지하여 정수/실수 시퀀스가 서로 간섭하지
 * 않는다.
 *
 * 호출 체인:
 *   random_utils.hpp::RandomSeed(seed) → ran_start(seed)  [rng.c 내 구현]
 *   random_utils.hpp::RandomInt(max)   → ran_next()       [이 파일]
 *                                           → ran_arr_next() [rng.c 내 매크로]
 *
 * 실행 컨텍스트: 호스트 CPU 단일 스레드 (GPGPU-Sim 시뮬레이션 루프).
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - rng.c : Knuth RANARRAY 알고리즘 C 구현. ran_arr_next(), ran_start() 포함.
 *             이 파일에 #include로 직접 삽입됨(컴파일 단위 내 inlining).
 * 이 파일을 사용하는 모듈:
 *   - random_utils.hpp : ran_next() 선언. 헤더 포함자 모두가 간접 사용.
 *   - trafficmanager.cc, allocator/, traffic/ 등 : RandomInt()를 통해 간접 사용.
 * 데이터 흐름:
 *   시드 → ran_start() → 전역 RNG 배열(rng.c 내 ran_u[]) 초기화
 *   → ran_arr_next() → 난수 소비 → RandomInt()로 노출.
 *
 * === 주요 함수/구조체 요약 ===
 * - #define main rng_main : rng.c의 main() 을 rng_main()으로 이름 변경하여
 *                            링커 충돌 방지.
 * - #include "rng.c"      : Knuth RANARRAY 전체 구현을 이 TU에 통합.
 * - ran_next()            : ran_arr_next() 호출 래퍼 — C++에서 호출 가능한
 *                            진입점. random_utils.hpp에 선언됨.
 */

#define main rng_main  // [한국어] rng.c 내의 main() 함수를 rng_main()으로 rename.
                        // rng.c는 테스트용 main()을 포함하므로, 이를 그대로 include하면
                        // 링커가 main() 심볼 충돌을 일으킨다. 이 매크로 트릭으로 회피.

#include "rng.c"        // [한국어] Knuth RANARRAY 정수 RNG 구현을 이 TU에 직접 삽입.
                         // ran_u[] 배열, ran_start(), ran_arr_next() 등 모든 상태와
                         // 함수가 이 컴파일 단위 내에서 static으로 존재하게 됨.
                         // 별도 .o로 분리하지 않고 이렇게 포함하는 이유:
                         // rng.c가 C89 스타일로 작성되어 C++ 헤더 분리가 어렵고,
                         // main() 이름 충돌 문제를 컴파일 단계에서 가장 깔끔하게
                         // 처리할 수 있기 때문이다.

/*
 * [한국어]
 * ran_next — Knuth RANARRAY 정수 RNG에서 다음 난수 반환
 *
 * @return: long 타입 정수 난수 ([0, LONG_MAX] 범위).
 *           ran_arr_next()는 rng.c 내의 매크로로, RNG 배열에서 다음 원소를
 *           소비하고 필요하면 배열을 재생성(refill)한다.
 *
 * 이 함수는 C++ 코드에서 호출 가능한 진입점(entry point)으로,
 * rng.c 내부의 ran_arr_next()를 C++ linkage로 노출한다.
 * random_utils.hpp의 RandomInt()/RandomIntLong() 이 이 함수를 호출한다.
 *
 * 실행 컨텍스트: 매 사이클 RandomInt() 호출 시, 단일 스레드.
 * 호출 체인:
 *   RandomInt(max) → ran_next() → ran_arr_next()  [rng.c 내 매크로]
 */
long ran_next( )
{
  return ran_arr_next( ); // [한국어] rng.c에서 #define된 ran_arr_next() 매크로/함수 호출.
                           // 내부 RNG 배열(ran_u[])에서 다음 난수를 꺼내며, 배열이 소진되면
                           // ran_arr_cycle() 을 호출하여 새 시퀀스를 생성한다.
}
