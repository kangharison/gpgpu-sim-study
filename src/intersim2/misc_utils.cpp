// $Id: misc_utils.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] BookSim NoC 시뮬레이터 수학 유틸리티 구현 (misc_utils.cpp)
 *
 * === 파일의 역할 ===
 * misc_utils.hpp에서 선언된 정수 수학 유틸리티 함수(powi, log_two)의 구현을 담는다.
 * powi는 반복 곱셈으로 정수 거듭제곱을 계산하고, log_two는 비트 시프트를
 * 반복하여 이진 로그를 계산한다. 두 함수 모두 시뮬레이터 초기화 시 토폴로지
 * 파라미터(노드 수, 차원 수, VC 비트 폭 등)를 산출하는 1회성 계산에 사용된다.
 * 실행 중(사이클 루프) 성능에는 영향을 주지 않는다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2/ 디렉토리의 유틸리티 계층 최하단에 위치한다.
 * 실행 컨텍스트: 호스트(CPU) 유저스페이스 — GPGPU-Sim 시뮬레이션 초기화 단계.
 * 호출 체인: network.cc, routefunc.cc 초기화 코드 → [powi / log_two] → 정수 반환.
 * 이 파일은 어떤 전역 상태도 유지하지 않으며, 두 함수 모두 순수 함수(pure function)이다.
 *
 * === 타 모듈과의 연결 ===
 * 의존: booksim.hpp (BookSim 공통 타입/매크로), misc_utils.hpp (자기 선언).
 * 피의존: intersim2 내 토폴로지/라우팅 초기화 코드.
 * 데이터 흐름: 정수 입력 → 계산 → 정수 반환 (전역 상태 없음).
 * 공유 자료구조: 없음.
 *
 * === 주요 함수/구조체 요약 ===
 * powi(x, y)   - x^y 정수 거듭제곱; y번 반복 곱셈으로 구현 (O(y))
 * log_two(x)   - floor(log₂(x)); x를 오른쪽으로 1비트씩 시프트하며 0이 될 때까지 카운트
 */

#include "booksim.hpp"     // [한국어] BookSim 공통 타입, 매크로, assert 등 기반 헤더
#include "misc_utils.hpp"  // [한국어] 이 파일에서 구현하는 powi, log_two 선언 헤더

/*
 * [한국어]
 * powi - 정수 x의 y제곱(x^y)을 반복 곱셈으로 계산한다.
 *
 * @x: 밑(base) 정수. k-ary 토폴로지에서 라우터당 포트 수(k) 또는
 *     radix 값으로 사용된다.
 * @y: 지수(exponent) 정수. 네트워크 차원(n)으로 사용된다.
 * @return: x^y 정수값. y=0이면 1(항등원). y<0은 지원하지 않음.
 *
 * 표준 라이브러리의 pow()는 double을 사용하므로 대규모 정수에서
 * 반올림 오차가 발생할 수 있다. 이 함수는 순수 정수 곱셈만 사용하여
 * 정확한 결과를 보장한다. O(y) 시간 복잡도이나 초기화 시 1회만 호출된다.
 * 실행 컨텍스트: 호스트 CPU, 시뮬레이터 초기화 단계(사이클 루프 외부).
 *
 * 호출 체인:
 *   network.cc / routefunc.cc → [powi] → 정수 반환
 */
int powi( int x, int y ) // compute x to the y
{
  int r = 1; // [한국어] 누적 곱 결과값 초기화; y=0인 경우 x^0=1을 올바르게 처리

  for ( int i = 0; i < y; ++i ) { // [한국어] y번 반복하여 x를 곱함; i는 0부터 y-1까지
    r *= x; // [한국어] 매 반복마다 r에 x를 곱하여 x^(i+1)을 누적
  }

  return r; // [한국어] 최종 x^y 값 반환
}

/*
 * [한국어]
 * log_two - 정수 x에 대한 이진 로그 floor(log₂(x))를 비트 시프트로 계산한다.
 *
 * @x: 이진 로그를 구할 양의 정수. 2의 거듭제곱인 경우 정확한 정수값을 반환한다.
 *     예: x=1→0, x=2→1, x=4→2, x=8→3, x=16→4.
 *     x=0은 정의되지 않으므로 호출자가 양수를 보장해야 한다.
 * @return: floor(log₂(x)) 정수값.
 *          x가 2의 거듭제곱이 아닌 경우 내림(floor) 값을 반환한다.
 *
 * k-ary n-cube 네트워크에서 차원 수 n = log₂(총 노드 수 / k) 등을
 * 계산할 때 사용된다. 비트를 오른쪽으로 1씩 시프트하면서 0이 될 때까지
 * 카운트하는 방식으로, 최상위 비트의 위치를 세는 것과 동일하다.
 * 실행 컨텍스트: 호스트 CPU, 시뮬레이터 초기화 단계(사이클 루프 외부).
 *
 * 호출 체인:
 *   network.cc / routefunc.cc → [log_two] → 정수 반환
 */
int log_two( int x )
{
  int r = 0; // [한국어] 비트 시프트 횟수 카운터 초기화; 최종적으로 floor(log₂(x))가 된다

  x >>= 1; // [한국어] x를 1비트 오른쪽으로 시프트하여 최하위 비트를 제거 (x/2 효과)
           // [한국어] x=1인 경우 이 시프트로 0이 되어 while 루프에 진입하지 않으므로 r=0 반환

  while( x ) {         // [한국어] x가 0이 아닌 동안 반복; x의 최상위 비트 위치를 카운트
    r++; x >>= 1;      // [한국어] 카운터 증가 후 x를 다시 1비트 오른쪽 시프트
                       // [한국어] 두 문장을 한 줄에 쓴 것은 원본 스타일 유지 (코드 수정 금지)
  }

  return r; // [한국어] floor(log₂(원래 x)) 반환
}
