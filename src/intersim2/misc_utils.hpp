// $Id: misc_utils.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] BookSim NoC 시뮬레이터 수학 유틸리티 헤더 (misc_utils.hpp)
 *
 * === 파일의 역할 ===
 * BookSim NoC(Network-on-Chip) 시뮬레이터 내부에서 공통적으로 필요한
 * 정수 수학 유틸리티 함수들을 선언하는 헤더 파일이다.
 * 구체적으로 정수 거듭제곱(powi)과 이진 로그(log_two) 두 함수를 제공하며,
 * 라우터 토폴로지 계산(포트 수, 버퍼 깊이, 채널 수 등)에서 빈번하게 사용된다.
 * 별도의 클래스 없이 순수 C 스타일 전역 함수로 선언되어 있어 어디서나 포함 가능하다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2 디렉토리 내 BookSim 시뮬레이터의 기반 유틸리티 계층에 속한다.
 * 실행 컨텍스트: 호스트(CPU) 유저스페이스 — GPGPU-Sim 시뮬레이션 루프 내에서 호출된다.
 * 호출 체인: network.cc, router.cc, routefunc.cc 등 토폴로지 초기화 코드
 *   → [이 헤더를 include] → powi/log_two 호출.
 * 이 파일 자체에는 상태가 없으며 순수 함수(side-effect 없음)만 선언한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존: 없음 (표준 라이브러리만 사용).
 * 피의존: intersim2 내 토폴로지/라우팅 코드가 #include "misc_utils.hpp"로 사용.
 * 데이터 흐름: 입력 정수값 → 계산 결과 정수 반환 (단방향, 전역 상태 없음).
 * 공유 자료구조: 없음 (전역 변수, 구조체 의존 없음).
 *
 * === 주요 함수/구조체 요약 ===
 * log_two(x)  - x의 이진 로그(floor(log2(x)))를 비트 시프트로 계산; 토폴로지 차원 수 산출에 사용
 * powi(x, y)  - 정수 x의 y제곱을 반복 곱셈으로 계산; k-진 fat-tree/butterfly 라우터 포트 수 산출에 사용
 */

#ifndef _MISC_UTILS_HPP_ // [한국어] 헤더 중복 포함 방지 가드 시작
#define _MISC_UTILS_HPP_ // [한국어] 가드 매크로 정의

/*
 * [한국어]
 * log_two - 정수 x의 이진 로그(floor(log₂(x)))를 계산한다.
 *
 * @x: 로그를 구할 양의 정수. 2의 거듭제곱이어야 정확한 결과를 얻는다.
 *     예: x=8 → 3, x=16 → 4. 구현은 비트 시프트 카운팅 방식.
 * @return: floor(log₂(x))에 해당하는 정수값.
 *          x=1이면 0, x=0이면 0 (정의되지 않은 입력이므로 주의).
 *
 * 토폴로지 초기화 시 네트워크 차원 수(k-ary n-cube의 n)나
 * 라우터 내부 VC 비트 폭을 계산하기 위해 사용된다.
 * 비트를 오른쪽으로 1씩 시프트하며 0이 될 때까지의 횟수를 센다.
 * 실행 컨텍스트: 호스트 CPU, 시뮬레이터 초기화 단계(1회성 계산).
 *
 * 호출 체인:
 *   network.cc / routefunc.cc → [log_two] (직접 호출)
 */
int log_two( int x );

/*
 * [한국어]
 * powi - 정수 x의 y제곱(x^y)을 반복 곱셈으로 계산한다.
 *
 * @x: 밑(base). 네트워크 토폴로지에서 라우터 수나 포트 수로 사용된다.
 * @y: 지수(exponent). 네트워크 차원(dimension) 수로 사용된다.
 * @return: x^y. y=0이면 1, y<0은 지원하지 않는다(결과가 정수가 아님).
 *
 * k-ary n-cube 네트워크에서 노드 수(k^n) 또는 fat-tree 포트 수를
 * 계산할 때 사용된다. 표준 라이브러리의 pow()는 부동소수점을 사용하므로
 * 정수 연산에서 반올림 오차가 생길 수 있어 이 정수 전용 버전을 사용한다.
 * 실행 컨텍스트: 호스트 CPU, 시뮬레이터 초기화 단계(1회성 계산).
 *
 * 호출 체인:
 *   network.cc / routefunc.cc → [powi] (직접 호출)
 */
int powi( int x, int y );

#endif // [한국어] _MISC_UTILS_HPP_ 헤더 가드 끝
