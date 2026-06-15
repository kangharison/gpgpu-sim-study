/*****************************************************************************
 *                                McPAT
 *                      SOFTWARE LICENSE AGREEMENT
 *            Copyright 2012 Hewlett-Packard Development Company, L.P.
 *                          All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.”
 *
 ***************************************************************************/

/*
 * [한국어 설명] McPAT/AccelWattch 전역 변수 선언 헤더 (globalvar.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 McPAT 및 AccelWattch 전체에서 공유되는 전역 변수를 단 하나의
 * 번역 단위(Translation Unit)에서만 정의(define)되고 나머지 모든 TU에서는
 * extern 선언으로만 노출되도록 하는 EXTERN 매크로 패턴을 구현한다.
 * 현재 정의된 전역 변수는 `opt_for_clk` 하나뿐이며, 이 변수는 CACTI SRAM
 * 최적화가 클럭 주파수 제약을 목표로 할지 여부를 전역적으로 제어한다.
 * 이 패턴은 C++ 전역 변수의 다중 정의 링크 오류(ODR 위반)를 방지하기 위한
 * 관용적 기법이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch는 GPGPU-Sim 타이밍 시뮬레이터의 전력 추정 서브시스템으로,
 * McPAT(Multi-core Power, Area, and Timing) 라이브러리를 내장 확장한 형태다.
 * 이 파일은 McPAT 초기화 경로(array.cc가 GLOBALVAR를 정의하고 include함)에서
 * opt_for_clk를 실제로 정의하며, 이후 CACTI 배열 최적화 루프에서 참조된다.
 * 실행 컨텍스트: 호스트 유저스페이스, McPAT 초기화 단계에서 단 한 번 설정되며
 * 이후 읽기 전용으로 사용된다.
 *
 * === 타 모듈과의 연결 ===
 * - 정의 TU: array.cc — `#define GLOBALVAR` 후 이 헤더를 include하여 변수를 실체화
 * - 선언 TU: McPAT/AccelWattch 내 나머지 모든 .cc 파일 — GLOBALVAR 미정의 상태에서
 *   include하여 `extern bool opt_for_clk`만 선언함
 * - 읽는 곳: CACTI의 배열 면적/전력/타이밍 최적화 루프에서 opt_for_clk를 참조해
 *   최적화 목표(클럭 주파수 vs 에너지)를 분기
 * - 데이터 흐름: McPAT 파서 → opt_for_clk 설정 → CACTI 최적화 루프 참조 → 배열 파라미터 반환
 *
 * === 주요 함수/구조체 요약 ===
 * - EXTERN 매크로: GLOBALVAR 정의 여부에 따라 정의(공백) 또는 extern으로 확장되는 패턴
 * - opt_for_clk: CACTI 최적화 목표 선택 플래그
 *   (true=클럭 주파수 제약 최적화, false=에너지/면적 최적화)
 */

#ifndef GLOBALVAR_H_
#define GLOBALVAR_H_

/*
 * [한국어] EXTERN 매크로 — 전역 변수 단일 정의 패턴(Singleton Definition Pattern) 구현.
 *
 * C++에서 헤더 파일에 전역 변수를 직접 선언하면 이 헤더를 include하는 모든
 * .cc 파일에서 변수가 중복 정의되어 링크 오류(multiple definition)가 발생한다.
 * 이를 방지하기 위해 단 하나의 TU(여기서는 array.cc)에서만 `#define GLOBALVAR`를
 * 먼저 선언한 뒤 이 헤더를 include한다. 그 TU에서 EXTERN은 빈 문자열로 확장되어
 * `bool opt_for_clk;` 형태의 실제 정의가 생성된다. 나머지 모든 TU에서는
 * GLOBALVAR가 정의되지 않으므로 EXTERN이 `extern`으로 확장되어
 * `extern bool opt_for_clk;` 형태의 선언만 생성된다.
 */
#ifdef GLOBALVAR
#define EXTERN        // [한국어] GLOBALVAR 정의된 TU: EXTERN → 공백 → 실제 변수 정의(definition) 생성
#else
#define EXTERN extern // [한국어] 나머지 모든 TU: EXTERN → extern → 외부 선언(declaration)만 생성; 링크 오류 방지
#endif

EXTERN bool opt_for_clk;
/* [한국어] opt_for_clk — CACTI SRAM 배열 최적화 목표 선택 전역 플래그.
 * true이면 CACTI가 클럭 주파수(타이밍) 제약을 만족하는 방향으로 SRAM 배열
 * 파라미터(워드라인 드라이버 크기, 감지 증폭기 개수, 뱅킹 구조 등)를 최적화한다.
 * false이면 에너지 또는 면적 최소화를 목표로 최적화한다.
 * 설정자: McPAT 초기화 경로에서 XML 파서 또는 커맨드라인 옵션을 통해 설정됨
 *         (array.cc의 초기화 코드에서 단 한 번 쓰임).
 * 읽는 자: CACTI의 배열 최적화 루프(Ucache_org_t 탐색 과정)에서 매 최적화 반복마다
 *          참조하여 목적 함수(objective function)의 분기를 결정함.
 * 값 범위: true(클럭 제약 우선) 또는 false(에너지/면적 우선).
 * 동기화: McPAT 초기화 단계에서 단 한 번 설정된 뒤 읽기 전용으로 사용되므로
 *         별도 동기화 메커니즘 불필요; 멀티스레드 동시 쓰기 시나리오 없음. */

#endif /* GLOBALVAR_H_ */
