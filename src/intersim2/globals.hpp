// $Id: globals.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] NoC 시뮬레이터 전역 변수 및 함수 선언 (globals.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 Booksim2 기반 NoC(Network-on-Chip) 시뮬레이터 전체에서 공유되는
 * 전역 변수와 전역 함수의 extern 선언을 모아 놓은 헤더다. 시뮬레이션 상태
 * (현재 사이클, 토폴로지 파라미터), 통계 시스템 접근, 디버그/트레이스 플래그,
 * 그리고 GPGPU-Sim과의 연결점인 g_icnt_interface까지 일괄 선언한다.
 * 실제 정의(definition)는 main.cpp 또는 각 모듈의 .cpp 파일에 존재한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2/ 패키지 전반에서 include하는 공통 글로벌 헤더이다. 라우터, 네트워크,
 * 트래픽 패턴, 통계 모듈 등 대부분의 .cpp 파일이 이 헤더를 include한다.
 *
 * 실행 컨텍스트: 호스트 유저스페이스 — GPGPU-Sim 시뮬레이션 루프 단일 스레드
 *
 * 데이터 흐름:
 *   icnt_wrapper_init() → IntersimConfig 파싱 → gK, gN, gC, gNodes 설정 →
 *   Booksim2 네트워크 객체 초기화 → 매 사이클 GetSimTime()으로 현재 시각 조회 →
 *   GetStats()로 통계 데이터 수집
 *
 * === 타 모듈과의 연결 ===
 * - main.cpp / icnt_wrapper.cc: gK, gN, gC, gNodes, gTrace, gPrintActivity,
 *   gWatchOut를 실제 정의하고 초기화한다.
 * - network.cpp: gNodes, gK, gN을 사용해 토폴로지 크기와 라우터 수를 결정한다.
 * - router.cpp / IQRouter: gTrace, gWatchOut를 사용해 디버그 출력을 제어한다.
 * - stats.hpp: GetStats()를 통해 이름 기반으로 Stats 객체에 접근한다.
 * - interconnect_interface.hpp: g_icnt_interface — GPGPU-Sim과 Booksim2 NoC를
 *   연결하는 인터페이스 객체 포인터.
 *
 * === 주요 함수/구조체 요약 ===
 * - GetSimTime()         : 현재 시뮬레이션 사이클 번호(시각)를 반환
 * - GetStats(name)       : 이름으로 Stats 통계 객체 포인터 반환
 * - g_icnt_interface     : GPGPU-Sim ↔ Booksim2 인터커넥트 인터페이스
 * - gPrintActivity       : 사이클별 활동 출력 여부 제어 플래그
 * - gK, gN, gC           : NoC 토폴로지 파라미터 (k-ary n-fly, VC 수)
 * - gNodes               : NoC의 총 단말 노드(endpoint) 수
 * - gTrace               : 패킷 트레이스 출력 활성화 플래그
 * - gWatchOut            : 디버그/트레이스 출력 대상 스트림 포인터
 */

#ifndef _GLOBALS_HPP_
#define _GLOBALS_HPP_

#include <string>   // [한국어] GetStats()의 파라미터 타입 std::string에 필요
#include <vector>   // [한국어] 향후 확장 또는 하위 파일에서 사용하는 vector 타입을 위해 포함
#include <iostream> // [한국어] gWatchOut (ostream*) 타입 선언 및 디버그 출력에 필요

/*all declared in main.cpp*/

/*
 * [한국어]
 * GetSimTime - 현재 시뮬레이션 사이클 번호(시각)를 반환한다.
 *
 * @return: int — 현재 시뮬레이션 사이클(정수). 0부터 시작하며 매 Step()마다 증가.
 *
 * Booksim2 NoC 시뮬레이터의 내부 사이클 카운터를 읽는다. 라우터, 통계 모듈
 * 등이 현재 시각을 기준으로 레이턴시, 타임아웃 등을 계산할 때 호출한다.
 * 실제 정의는 main.cpp(Booksim2 standalone) 또는 icnt_wrapper.cc(GPGPU-Sim 통합)에 있다.
 *
 * 호출 체인:
 *   Router::_Step() / Stats 수집 코드 → GetSimTime()
 */
int GetSimTime();

class Stats;

/*
 * [한국어]
 * GetStats - 이름으로 등록된 Stats 통계 객체를 반환한다.
 *
 * @param name: 조회할 통계 항목의 이름 문자열 (예: "packet_latency", "flit_latency")
 * @return: Stats* — 해당 이름의 Stats 객체 포인터. 없으면 NULL 또는 오류.
 *
 * NoC 시뮬레이터 전체에서 이름 기반으로 통계 객체를 공유한다. 각 라우터와
 * 네트워크 모듈이 통계를 기록할 때 사용하며, 최종 결과 출력 시에도 활용된다.
 * 실제 정의 및 등록 맵은 main.cpp에 있다.
 *
 * 호출 체인:
 *   Network 초기화 / Router 초기화 → GetStats("latency") → Stats 객체에 기록
 */
Stats * GetStats(const std::string & name);

class InterconnectInterface;

extern InterconnectInterface *g_icnt_interface;
/* [한국어] GPGPU-Sim 상위 레이어와 Booksim2 NoC를 연결하는 인터페이스 객체 포인터.
 * 설정자: icnt_wrapper_init()에서 InterconnectInterface 인스턴스를 생성하여 할당.
 * 읽는 자: Booksim2 내부의 일부 콜백/주입 함수가 상위 인터페이스에 접근할 때 사용.
 *          shader.cc, gpu-sim.cc 등 GPGPU-Sim 타이밍 모델이 icnt_push/pop을 통해 접근.
 * 값 범위: 유효한 InterconnectInterface 포인터 (NULL이면 초기화 미완료 상태).
 * 동기화: 단일 시뮬레이션 스레드에서 접근하므로 별도 락 불필요.
 * 선언 위치: interconnect_interface.hpp에 클래스 정의, 실제 정의는 icnt_wrapper.cc. */

extern bool gPrintActivity;
/* [한국어] 매 시뮬레이션 사이클의 네트워크 활동(패킷 이동, 크레딧 전송 등)을
 * 표준 출력 또는 gWatchOut에 상세히 출력할지 결정하는 플래그.
 * 설정자: gpgpusim.config의 설정 또는 커맨드라인 옵션에 의해 초기화됨.
 * 읽는 자: Router::_Step() 등 매 사이클 동작 함수에서 if(gPrintActivity) 분기.
 * 값 범위: true(상세 출력 활성) / false(비활성, 성능 우선).
 * 동기화: 읽기 전용으로 사용되어 동기화 불필요. */

extern int gK;
/* [한국어] NoC 토폴로지의 k 파라미터 — k-ary n-fly 네트워크에서 각 차원의 기수(radix).
 * 예: gK=8이면 각 라우터가 8개의 포트를 가지는 8-ary 구조.
 * 설정자: IntersimConfig 파싱 후 네트워크 초기화 시 설정됨.
 * 읽는 자: 라우터 주소 계산, 라우팅 알고리즘(dimension-order routing 등)에서 사용.
 * 값 범위: 2 이상의 양의 정수. 실제 GPU 설정에서는 SM 수에 따라 결정됨.
 * 동기화: 초기화 후 읽기 전용 — 별도 락 불필요. */

extern int gN;
/* [한국어] NoC 토폴로지의 n 파라미터 — k-ary n-fly 네트워크의 차원(dimension) 수.
 * 예: gN=2이면 2차원(mesh/torus), gN=1이면 1차원(ring/bus) 구조.
 * 설정자: IntersimConfig 파싱 후 네트워크 초기화 시 설정됨.
 * 읽는 자: 라우팅 함수가 차원 수에 따라 루프를 돌며 주소를 분리할 때 사용.
 * 값 범위: 1 이상의 양의 정수. GPGPU-Sim에서는 주로 1~2.
 * 동기화: 초기화 후 읽기 전용 — 별도 락 불필요. */

extern int gC;
/* [한국어] NoC의 VC(Virtual Channel) 수 또는 채널 수 파라미터.
 * Booksim2에서 gC는 concentrator/channel 계수를 의미하며, 네트워크의
 * 각 라우터 포트에 연결된 단말 채널 수를 나타낸다.
 * 설정자: IntersimConfig 파싱 후 네트워크 초기화 시 설정됨.
 * 읽는 자: 네트워크 토폴로지 생성 및 단말 주소 매핑 계산에 사용.
 * 값 범위: 1 이상의 양의 정수.
 * 동기화: 초기화 후 읽기 전용 — 별도 락 불필요. */

extern int gNodes;
/* [한국어] NoC에 연결된 총 단말 노드(endpoint) 수.
 * GPGPU-Sim 맥락에서 단말 노드는 SM(Shader Core)과 메모리 파티션의 합이다.
 * 예: SM 수 + 메모리 파티션 수 = gNodes.
 * 설정자: icnt_wrapper_init()에서 SM 수와 메모리 노드 수를 합산하여 설정.
 * 읽는 자: 네트워크 생성 시 라우터/링크 수 결정, 라우팅 테이블 크기 계산에 사용.
 * 값 범위: 양의 정수. 일반적으로 SM 수(예: 28~80) + 메모리 파티션 수(예: 8~32).
 * 동기화: 초기화 후 읽기 전용 — 별도 락 불필요. */

extern bool gTrace;
/* [한국어] 패킷 트레이스 출력 활성화 플래그.
 * true이면 각 패킷(Flit)의 생성, 라우팅, 소비 이벤트를 gWatchOut에 출력한다.
 * 설정자: gpgpusim.config 또는 커맨드라인 옵션에 의해 초기화됨.
 * 읽는 자: Flit 생성/소비 지점, 라우터의 _Step() 함수에서 if(gTrace) 분기.
 * 값 범위: true(트레이스 출력 활성) / false(비활성, 성능 우선).
 * 동기화: 읽기 전용으로 사용되어 동기화 불필요. */

extern std::ostream * gWatchOut;
/* [한국어] 디버그/트레이스 출력 대상 스트림 포인터.
 * 설정자: main.cpp / icnt_wrapper_init()에서 cerr 또는 파일 스트림으로 초기화됨.
 * 읽는 자: gPrintActivity 또는 gTrace가 활성화된 모든 출력 지점에서 *gWatchOut로 출력.
 * 값 범위: 유효한 ostream 포인터 (NULL이면 출력 비활성화로 처리해야 함).
 * 동기화: 단일 시뮬레이션 스레드에서 접근하므로 별도 락 불필요. */

#endif
