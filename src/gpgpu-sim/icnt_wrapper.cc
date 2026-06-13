// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/*
 * [한국어 설명] 인터커넥트 네트워크(ICNT) 래퍼 구현 (icnt_wrapper.cc)
 *
 * === 파일의 역할 ===
 * icnt_wrapper.h에서 선언된 전역 함수 포인터 변수들을 정의하고,
 * g_network_mode(-network_mode 설정값)에 따라 두 가지 ICNT 구현체 중 하나로
 * 바인딩하는 icnt_wrapper_init()을 구현한다. 또한 각 구현체(intersim2,
 * local_interconnect)의 멤버 함수를 래핑하는 static 함수들을 정의하여
 * 함수 포인터 타입과 인터페이스를 맞춘다. icnt_reg_options()는 OptionParser에
 * ICNT 관련 설정 옵션들을 등록하여 gpgpusim.config 파일에서 설정 가능하게 한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 시뮬레이터 초기화 흐름의 일부로, 시뮬레이터 시작 시 한 번만 실행된다.
 * icnt_reg_options() → 설정 파싱 → icnt_wrapper_init() 순서로 진행된다.
 * 이후 사이클 루프에서는 바인딩된 함수 포인터(icnt_push, icnt_pop, icnt_transfer)가
 * 직접 호출되므로 이 파일의 추가 개입 없이 구현체가 실행된다.
 * 호출 체인: gpgpusim_entrypoint.cc → icnt_reg_options() → (옵션 파싱) →
 *            icnt_wrapper_init() → [icnt_push/pop/transfer 사이클 루프]
 *
 * === 타 모듈과의 연결 ===
 * - icnt_wrapper.h: 이 파일이 구현하는 인터페이스 선언 포함
 * - intersim2/globals.hpp: g_icnt_interface(InterconnectInterface*) 전역 변수 참조
 * - intersim2/interconnect_interface.hpp: InterconnectInterface::New() 팩토리 메서드
 * - local_interconnect.h: LocalInterconnect::New() 팩토리 메서드
 * - option_parser.h: option_parser_register() 함수
 * 데이터 흐름: gpgpusim.config의 -network_mode → g_network_mode →
 *              icnt_wrapper_init() → 함수 포인터 바인딩 → 사이클 루프에서 호출
 *
 * === 주요 함수/구조체 요약 ===
 * intersim2_* 함수들  - intersim2/InterconnectInterface 메서드를 래핑하는 static 함수들
 * LocalInterconnect_* - local_interconnect/LocalInterconnect 메서드를 래핑하는 static 함수들
 * icnt_reg_options()  - -network_mode, -inter_config_file, local_xbar 파라미터 등록
 * icnt_wrapper_init() - g_network_mode에 따라 구현체 생성 및 함수 포인터 바인딩
 * g_inct_config       - local_xbar용 설정 구조체 (버퍼 크기, 중재 알고리즘 등)
 */

#include "icnt_wrapper.h"
#include <assert.h>                                    /* [한국어] assert() — icnt_wrapper_init()에서 unknown 모드 검출용 */
#include "../intersim2/globals.hpp"                    /* [한국어] g_icnt_interface(InterconnectInterface*) 전역 포인터 선언 포함 */
#include "../intersim2/interconnect_interface.hpp"     /* [한국어] InterconnectInterface 클래스 및 New() 팩토리 메서드 */
#include "local_interconnect.h"                        /* [한국어] LocalInterconnect 클래스 및 inct_config 구조체 */

/* [한국어] 전역 함수 포인터 변수 정의 — icnt_wrapper.h에서 extern 선언된 변수들의 실제 정의.
 * 초기값 NULL (전역 포인터 기본값), icnt_wrapper_init()에서 구현체 함수로 바인딩됨. */
icnt_create_p icnt_create;               /* [한국어] 네트워크 생성 함수 포인터 */
icnt_init_p icnt_init;                   /* [한국어] 네트워크 초기화 함수 포인터 */
icnt_has_buffer_p icnt_has_buffer;       /* [한국어] 버퍼 여유 확인 함수 포인터 */
icnt_push_p icnt_push;                   /* [한국어] 패킷 삽입 함수 포인터 */
icnt_pop_p icnt_pop;                     /* [한국어] 패킷 수신 함수 포인터 */
icnt_transfer_p icnt_transfer;           /* [한국어] 사이클 진행 함수 포인터 */
icnt_busy_p icnt_busy;                   /* [한국어] ICNT 바쁨 여부 확인 함수 포인터 */
icnt_display_stats_p icnt_display_stats; /* [한국어] 통계 출력 함수 포인터 */
icnt_display_overall_stats_p icnt_display_overall_stats; /* [한국어] 누적 통계 출력 함수 포인터 */
icnt_display_state_p icnt_display_state; /* [한국어] 내부 상태 덤프 함수 포인터 */
icnt_get_flit_size_p icnt_get_flit_size; /* [한국어] 플릿 크기 반환 함수 포인터 */

unsigned g_network_mode;              /* [한국어] 선택된 네트워크 모드 — INTERSIM(1) 또는 LOCAL_XBAR(2), option_parser가 설정 */
char* g_network_config_filename;      /* [한국어] intersim2 설정 파일 경로 문자열 포인터 — -inter_config_file 옵션값 */

struct inct_config g_inct_config;         /* [한국어] local_xbar용 전역 설정 구조체 — 버퍼 크기, 중재 알고리즘 등 option_parser로 채워짐 */
LocalInterconnect* g_localicnt_interface; /* [한국어] LOCAL_XBAR 모드에서 사용할 LocalInterconnect 객체 포인터 */

#include "../option_parser.h"  /* [한국어] option_parser_register() 함수 사용을 위해 함수 정의 이후에 포함 (구현부에서만 필요) */

// Wrapper to intersim2 to accompany old icnt_wrapper
// TODO: use delegate/boost/c++11<funtion> instead
/* [한국어] intersim2 래퍼 함수들 — InterconnectInterface 객체 메서드를 C 함수 포인터 인터페이스로 감쌈.
 * TODO: C++11 std::function 또는 boost::function으로 교체하면 더 깔끔하게 구현 가능. */

/*
 * [한국어]
 * intersim2_create - intersim2 네트워크 토폴로지 생성 래퍼
 *
 * @n_shader: SM 클러스터(소스 노드) 수
 * @n_mem: 메모리 파티션(목적지 노드) 수
 * @return: 없음 (void)
 *
 * g_icnt_interface(InterconnectInterface*)의 CreateInterconnect()를 호출하여
 * BookSim 설정에 따른 NoC 토폴로지(메쉬, 버터플라이 등)를 생성한다.
 * icnt_create 함수 포인터에 바인딩되어 INTERSIM 모드에서 사용된다.
 *
 * 호출 체인: icnt_create(n_shader, n_mem) → [이 함수] → g_icnt_interface->CreateInterconnect()
 */
static void intersim2_create(unsigned int n_shader, unsigned int n_mem) {
  g_icnt_interface->CreateInterconnect(n_shader, n_mem);
  /* [한국어] BookSim 기반 NoC를 n_shader 소스 노드, n_mem 목적지 노드로 초기화 */
}

/*
 * [한국어]
 * intersim2_init - intersim2 네트워크 초기화 래퍼
 *
 * @return: 없음 (void)
 *
 * g_icnt_interface->Init()을 호출하여 BookSim 라우터 및 링크 초기 상태를 설정한다.
 * icnt_init 함수 포인터에 바인딩된다.
 *
 * 호출 체인: icnt_init() → [이 함수] → g_icnt_interface->Init()
 */
static void intersim2_init() { g_icnt_interface->Init(); }
/* [한국어] intersim2(BookSim) 초기화 — 라우터 버퍼, 링크, 통계 카운터 초기화 */

/*
 * [한국어]
 * intersim2_has_buffer - intersim2 입력 버퍼 여유 확인 래퍼
 *
 * @input: 확인할 소스 노드 ID
 * @size: 삽입하려는 패킷의 크기(바이트)
 * @return: 버퍼 여유가 있으면 true, 없으면 false (back-pressure 신호)
 *
 * icnt_has_buffer 함수 포인터에 바인딩된다.
 * false 반환 시 상위 레벨(shader.cc 등)에서 삽입을 미루고 대기해야 한다.
 *
 * 호출 체인: icnt_has_buffer(input, size) → [이 함수] → g_icnt_interface->HasBuffer()
 */
static bool intersim2_has_buffer(unsigned input, unsigned int size) {
  return g_icnt_interface->HasBuffer(input, size);
  /* [한국어] intersim2 입력 포트 input에 size 크기 패킷을 수용할 공간이 있는지 확인 */
}

/*
 * [한국어]
 * intersim2_push - intersim2에 패킷 삽입 래퍼
 *
 * @input: 소스 노드 ID (SM 클러스터 인덱스 또는 메모리 파티션 인덱스)
 * @output: 목적지 노드 ID
 * @data: 전달할 mem_fetch 등 데이터 포인터
 * @size: 패킷 크기(바이트) — 플릿 수 계산용
 * @return: 없음 (void)
 *
 * HasBuffer() 확인 후 호출해야 한다. intersim2 내부에서 data를 플릿으로 분해하여
 * BookSim 라우터에 주입한다.
 *
 * 호출 체인: icnt_push(input, output, data, size) → [이 함수] → g_icnt_interface->Push()
 */
static void intersim2_push(unsigned input, unsigned output, void* data,
                           unsigned int size) {
  g_icnt_interface->Push(input, output, data, size);
  /* [한국어] intersim2에 input→output 방향 패킷 삽입 — 내부적으로 플릿 단위로 변환하여 라우터 입력 버퍼에 넣음 */
}

/*
 * [한국어]
 * intersim2_pop - intersim2에서 도착 패킷 수신 래퍼
 *
 * @output: 수신 노드 ID
 * @return: 도착한 데이터 포인터 (없으면 NULL)
 *
 * intersim2 출력 버퍼에서 output 노드로 완전히 도착한 패킷을 꺼낸다.
 * 반환 포인터는 통상 mem_fetch*로 캐스팅하여 사용한다.
 *
 * 호출 체인: icnt_pop(output) → [이 함수] → g_icnt_interface->Pop()
 */
static void* intersim2_pop(unsigned output) {
  return g_icnt_interface->Pop(output);
  /* [한국어] intersim2 출력 버퍼 output에서 완전히 전달된 패킷 하나를 꺼내 반환 */
}

/*
 * [한국어]
 * intersim2_transfer - intersim2 한 사이클 진행 래퍼
 *
 * @return: 없음 (void)
 *
 * g_icnt_interface->Advance()를 호출하여 BookSim 라우터의 플릿을 한 사이클 전진시킨다.
 * 매 시뮬레이션 사이클마다 gpu-sim.cc의 사이클 루프에서 호출된다.
 *
 * 호출 체인: icnt_transfer() → [이 함수] → g_icnt_interface->Advance()
 */
static void intersim2_transfer() { g_icnt_interface->Advance(); }
/* [한국어] intersim2 내부의 플릿을 한 사이클 전진 — 라우터 파이프라인 진행 */

/*
 * [한국어]
 * intersim2_busy - intersim2 in-flight 패킷 존재 여부 확인 래퍼
 *
 * @return: 아직 전달 중인 패킷이 있으면 true, 없으면 false
 *
 * 시뮬레이션 종료 시 ICNT가 완전히 드레인되었는지 확인하는 데 사용된다.
 *
 * 호출 체인: icnt_busy() → [이 함수] → g_icnt_interface->Busy()
 */
static bool intersim2_busy() { return g_icnt_interface->Busy(); }
/* [한국어] intersim2 내에 아직 전달 중인 패킷이 존재하는지 확인 */

/*
 * [한국어]
 * intersim2_display_stats - intersim2 통계 출력 래퍼
 *
 * @return: 없음 (void)
 *
 * 시뮬레이션 완료 후 BookSim의 throughput, latency, 링크 utilization 등 NoC 통계를 출력한다.
 *
 * 호출 체인: icnt_display_stats() → [이 함수] → g_icnt_interface->DisplayStats()
 */
static void intersim2_display_stats() { g_icnt_interface->DisplayStats(); }
/* [한국어] intersim2 통계 출력 — 네트워크 처리량, 레이턴시, 링크 활용도 등 */

/*
 * [한국어]
 * intersim2_display_overall_stats - intersim2 누적 전체 통계 출력 래퍼
 *
 * @return: 없음 (void)
 *
 * 여러 커널 실행에 걸친 전체 통계를 출력한다.
 *
 * 호출 체인: icnt_display_overall_stats() → [이 함수] → g_icnt_interface->DisplayOverallStats()
 */
static void intersim2_display_overall_stats() {
  g_icnt_interface->DisplayOverallStats();
  /* [한국어] intersim2 전체 누적 통계 출력 */
}

/*
 * [한국어]
 * intersim2_display_state - intersim2 내부 상태 덤프 래퍼
 *
 * @fp: 출력 대상 파일 포인터
 * @return: 없음 (void)
 *
 * 현재 라우터 버퍼, 링크, in-flight 플릿 등의 상태를 fp에 출력한다.
 * 디버깅 목적으로 사용된다.
 *
 * 호출 체인: icnt_display_state(fp) → [이 함수] → g_icnt_interface->DisplayState(fp)
 */
static void intersim2_display_state(FILE* fp) {
  g_icnt_interface->DisplayState(fp);
  /* [한국어] intersim2 현재 내부 상태(라우터 버퍼, in-flight 패킷 등) fp에 덤프 */
}

/*
 * [한국어]
 * intersim2_get_flit_size - intersim2 플릿 크기 반환 래퍼
 *
 * @return: 플릿 크기 (바이트 단위, unsigned)
 *
 * 패킷 크기를 플릿 수로 변환할 때 분모로 사용한다.
 * BookSim 설정 파일에서 정의된 플릿 크기를 반환한다.
 *
 * 호출 체인: icnt_get_flit_size() → [이 함수] → g_icnt_interface->GetFlitSize()
 */
static unsigned intersim2_get_flit_size() {
  return g_icnt_interface->GetFlitSize();
  /* [한국어] intersim2 설정에서 정의된 플릿 크기(바이트) 반환 */
}

//////////////////////////////////////////////////////
/* [한국어] LocalInterconnect 래퍼 함수들 — local_interconnect.h의 LocalInterconnect 멤버 메서드를
 * 함수 포인터 인터페이스로 래핑한다. LOCAL_XBAR 모드에서 함수 포인터에 바인딩됨. */

/*
 * [한국어]
 * LocalInterconnect_create - local_interconnect 네트워크 토폴로지 생성 래퍼
 *
 * @n_shader: SM 클러스터(소스 노드) 수
 * @n_mem: 메모리 파티션(목적지 노드) 수
 * @return: 없음 (void)
 *
 * g_localicnt_interface->CreateInterconnect()를 호출하여 n_subnets 개의
 * xbar_router 객체를 생성한다. 기본 2개 서브넷(REQ_NET, REPLY_NET).
 *
 * 호출 체인: icnt_create(n_shader, n_mem) → [이 함수] → g_localicnt_interface->CreateInterconnect()
 */
static void LocalInterconnect_create(unsigned int n_shader,
                                     unsigned int n_mem) {
  g_localicnt_interface->CreateInterconnect(n_shader, n_mem);
  /* [한국어] local_xbar 인터커넥트를 n_shader 입력, n_mem 출력으로 설정하여 xbar_router 생성 */
}

/*
 * [한국어]
 * LocalInterconnect_init - local_interconnect 초기화 래퍼 (현재 no-op)
 *
 * @return: 없음 (void)
 *
 * LocalInterconnect::Init()은 현재 아무 동작도 하지 않는 빈 함수다.
 * 인터페이스 일관성을 위해 유지된다.
 *
 * 호출 체인: icnt_init() → [이 함수] → g_localicnt_interface->Init()
 */
static void LocalInterconnect_init() { g_localicnt_interface->Init(); }
/* [한국어] local_xbar 초기화 — 현재 Init()이 빈 함수이므로 실질적 동작 없음 */

/*
 * [한국어]
 * LocalInterconnect_has_buffer - local_interconnect 입력 버퍼 여유 확인 래퍼
 *
 * @input: 확인할 소스 노드 ID
 * @size: 삽입하려는 패킷 크기
 * @return: 버퍼 여유가 있으면 true, 없으면 false
 *
 * HasBuffer()는 내부적으로 deviceID가 메모리 노드이면 REPLY_NET,
 * SM 노드이면 REQ_NET의 Has_Buffer_In()을 호출한다.
 *
 * 호출 체인: icnt_has_buffer(input, size) → [이 함수] → g_localicnt_interface->HasBuffer()
 */
static bool LocalInterconnect_has_buffer(unsigned input, unsigned int size) {
  return g_localicnt_interface->HasBuffer(input, size);
  /* [한국어] local_xbar의 입력 노드 input에 size 크기 패킷을 수용할 버퍼 여유가 있는지 확인 */
}

/*
 * [한국어]
 * LocalInterconnect_push - local_interconnect에 패킷 삽입 래퍼
 *
 * @input: 소스 노드 ID, @output: 목적지 노드 ID
 * @data: 전달할 데이터 포인터, @size: 패킷 크기
 * @return: 없음 (void)
 *
 * 소스 노드의 위치(SM or 메모리)에 따라 REQ_NET 또는 REPLY_NET의 입력 버퍼에 삽입한다.
 *
 * 호출 체인: icnt_push(input, output, data, size) → [이 함수] → g_localicnt_interface->Push()
 */
static void LocalInterconnect_push(unsigned input, unsigned output, void* data,
                                   unsigned int size) {
  g_localicnt_interface->Push(input, output, data, size);
  /* [한국어] local_xbar의 input→output 방향 패킷 삽입 — 서브넷 선택 후 xbar_router의 입력 버퍼에 추가 */
}

/*
 * [한국어]
 * LocalInterconnect_pop - local_interconnect에서 도착 패킷 수신 래퍼
 *
 * @output: 수신 노드 ID
 * @return: 도착한 데이터 포인터 (없으면 NULL)
 *
 * 수신 노드가 SM이면 REPLY_NET, 메모리 파티션이면 REQ_NET에서 꺼낸다.
 *
 * 호출 체인: icnt_pop(output) → [이 함수] → g_localicnt_interface->Pop()
 */
static void* LocalInterconnect_pop(unsigned output) {
  return g_localicnt_interface->Pop(output);
  /* [한국어] local_xbar에서 output 노드의 출력 버퍼에 도착한 패킷 하나를 꺼내 반환 */
}

/*
 * [한국어]
 * LocalInterconnect_transfer - local_interconnect 한 사이클 진행 래퍼
 *
 * @return: 없음 (void)
 *
 * 모든 서브넷(REQ_NET, REPLY_NET)의 xbar_router->Advance()를 순서대로 호출하여
 * 중재 알고리즘(RR or iSLIP)을 실행하고 패킷을 입력 버퍼에서 출력 버퍼로 이동한다.
 *
 * 호출 체인: icnt_transfer() → [이 함수] → g_localicnt_interface->Advance()
 */
static void LocalInterconnect_transfer() { g_localicnt_interface->Advance(); }
/* [한국어] local_xbar의 모든 서브넷 xbar_router를 한 사이클 진행 */

/*
 * [한국어]
 * LocalInterconnect_busy - local_interconnect in-flight 패킷 존재 여부 확인 래퍼
 *
 * @return: 전달 중 패킷이 있으면 true, 없으면 false
 *
 * 모든 서브넷의 모든 입력/출력 버퍼가 비어 있으면 false를 반환한다.
 *
 * 호출 체인: icnt_busy() → [이 함수] → g_localicnt_interface->Busy()
 */
static bool LocalInterconnect_busy() { return g_localicnt_interface->Busy(); }
/* [한국어] local_xbar의 모든 버퍼가 비었는지 확인 — 시뮬레이션 종료 조건 판단에 사용 */

/*
 * [한국어]
 * LocalInterconnect_display_stats - local_interconnect 통계 출력 래퍼
 *
 * @return: 없음 (void)
 *
 * REQ_NET, REPLY_NET 각각의 throughput, conflict rate, buffer utilization 등을 출력한다.
 *
 * 호출 체인: icnt_display_stats() → [이 함수] → g_localicnt_interface->DisplayStats()
 */
static void LocalInterconnect_display_stats() {
  g_localicnt_interface->DisplayStats();
  /* [한국어] local_xbar 통계 출력 — 요청/응답 네트워크별 패킷 수, 충돌률, 버퍼 활용도 */
}

/*
 * [한국어]
 * LocalInterconnect_display_overall_stats - local_interconnect 전체 통계 출력 래퍼
 *
 * @return: 없음 (void)
 *
 * LocalInterconnect::DisplayOverallStats()는 현재 빈 함수다. 인터페이스 일관성용.
 *
 * 호출 체인: icnt_display_overall_stats() → [이 함수] → g_localicnt_interface->DisplayOverallStats()
 */
static void LocalInterconnect_display_overall_stats() {
  g_localicnt_interface->DisplayOverallStats();
  /* [한국어] local_xbar 전체 누적 통계 출력 — 현재 빈 함수 */
}

/*
 * [한국어]
 * LocalInterconnect_display_state - local_interconnect 내부 상태 덤프 래퍼
 *
 * @fp: 출력 대상 파일 포인터
 * @return: 없음 (void)
 *
 * 현재 "Under implementation" 메시지만 출력한다. 디버그 인터페이스 예약 슬롯.
 *
 * 호출 체인: icnt_display_state(fp) → [이 함수] → g_localicnt_interface->DisplayState(fp)
 */
static void LocalInterconnect_display_state(FILE* fp) {
  g_localicnt_interface->DisplayState(fp);
  /* [한국어] local_xbar 내부 상태 fp에 출력 — 현재 "Under implementation" 메시지만 출력 */
}

/*
 * [한국어]
 * LocalInterconnect_get_flit_size - local_interconnect 플릿 크기 반환 래퍼
 *
 * @return: 플릿 크기 (바이트 단위, unsigned) — LOCAL_INCT_FLIT_SIZE(40) 고정 반환
 *
 * local_xbar는 모든 패킷을 1 플릿으로 처리하는 단순화 모델이므로
 * 플릿 크기는 40바이트로 고정되어 있다.
 *
 * 호출 체인: icnt_get_flit_size() → [이 함수] → g_localicnt_interface->GetFlitSize()
 */
static unsigned LocalInterconnect_get_flit_size() {
  return g_localicnt_interface->GetFlitSize();
  /* [한국어] local_xbar의 고정 플릿 크기(LOCAL_INCT_FLIT_SIZE = 40바이트) 반환 */
}

///////////////////////////

/*
 * [한국어]
 * icnt_reg_options - OptionParser에 ICNT 관련 설정 옵션 등록
 *
 * @opp: gpgpusim.config 파싱을 위한 OptionParser 객체 포인터
 * @return: 없음 (void)
 *
 * gpgpusim.config 파일에서 ICNT 관련 파라미터를 읽어 전역 변수/구조체에 저장하도록
 * option_parser_register()를 통해 옵션들을 등록한다.
 * 등록 옵션:
 *   -network_mode: 1(INTERSIM) 또는 2(LOCAL_XBAR). 기본값 1.
 *   -inter_config_file: BookSim 설정 파일 경로. 기본값 "mesh".
 *   -icnt_in_buffer_limit: local_xbar 입력 버퍼 크기. 기본값 64.
 *   -icnt_out_buffer_limit: local_xbar 출력 버퍼 크기. 기본값 64.
 *   -icnt_subnets: 서브넷 수 (REQ+REPLY). 기본값 2.
 *   -icnt_arbiter_algo: 중재 알고리즘 (0=NAIVE_RR, 1=iSLIP). 기본값 1.
 *   -icnt_verbose: 디버그 출력 레벨. 기본값 0 (비활성).
 *   -icnt_grant_cycles: iSLIP grant 유지 사이클. 기본값 1.
 *
 * 호출 체인: gpgpusim_entrypoint.cc 옵션 등록 단계 → [이 함수]
 */
void icnt_reg_options(class OptionParser* opp) {
  option_parser_register(opp, "-network_mode", OPT_INT32, &g_network_mode,
                         "Interconnection network mode", "1");
  /* [한국어] -network_mode: 1=INTERSIM(BookSim 기반), 2=LOCAL_XBAR(단순 크로스바). 기본값 1 */
  option_parser_register(opp, "-inter_config_file", OPT_CSTR,
                         &g_network_config_filename,
                         "Interconnection network config file", "mesh");
  /* [한국어] -inter_config_file: intersim2에 전달할 BookSim 토폴로지/라우팅 설정 파일 경로. 기본 "mesh" */

  // parameters for local xbar
  /* [한국어] local_xbar(LOCAL_XBAR 모드) 전용 파라미터들 — g_inct_config 구조체 필드에 바인딩 */
  option_parser_register(opp, "-icnt_in_buffer_limit", OPT_UINT32,
                         &g_inct_config.in_buffer_limit, "in_buffer_limit",
                         "64");
  /* [한국어] -icnt_in_buffer_limit: xbar_router 입력 버퍼 최대 패킷 수. 기본값 64 */
  option_parser_register(opp, "-icnt_out_buffer_limit", OPT_UINT32,
                         &g_inct_config.out_buffer_limit, "out_buffer_limit",
                         "64");
  /* [한국어] -icnt_out_buffer_limit: xbar_router 출력 버퍼 최대 패킷 수. 기본값 64 */
  option_parser_register(opp, "-icnt_subnets", OPT_UINT32,
                         &g_inct_config.subnets, "subnets", "2");
  /* [한국어] -icnt_subnets: 서브네트워크 수. 기본값 2 (REQ_NET=0, REPLY_NET=1) */
  option_parser_register(opp, "-icnt_arbiter_algo", OPT_UINT32,
                         &g_inct_config.arbiter_algo, "arbiter_algo", "1");
  /* [한국어] -icnt_arbiter_algo: 중재 알고리즘 선택. 0=NAIVE_RR(단순 라운드로빈), 1=iSLIP. 기본값 1 */
  option_parser_register(opp, "-icnt_verbose", OPT_UINT32,
                         &g_inct_config.verbose, "inct_verbose", "0");
  /* [한국어] -icnt_verbose: local_xbar 디버그 출력 활성화 여부. 0=비활성(기본값) */
  option_parser_register(opp, "-icnt_grant_cycles", OPT_UINT32,
                         &g_inct_config.grant_cycles, "grant_cycles", "1");
  /* [한국어] -icnt_grant_cycles: iSLIP에서 하나의 grant를 몇 사이클 유지할지. 기본값 1 */
}

/*
 * [한국어]
 * icnt_wrapper_init - g_network_mode에 따라 ICNT 구현체를 생성하고 함수 포인터 바인딩
 *
 * @return: 없음 (void)
 *
 * option_parser가 설정을 파싱한 뒤 호출된다. g_network_mode 값에 따라
 * INTERSIM 모드이면 InterconnectInterface::New()로 BookSim 객체를 생성하고
 * intersim2_* 래퍼 함수들을 전역 함수 포인터들에 할당한다.
 * LOCAL_XBAR 모드이면 LocalInterconnect::New()로 단순 크로스바 객체를 생성하고
 * LocalInterconnect_* 래퍼 함수들을 할당한다.
 * 이후 시뮬레이터 코드는 icnt_push/pop 등을 직접 호출하여 구현체 무관하게 동작한다.
 * 알 수 없는 모드는 assert(0)으로 즉시 종료한다.
 *
 * 호출 체인: gpgpusim_entrypoint.cc 초기화 → [이 함수]
 */
void icnt_wrapper_init() {
  switch (g_network_mode) {
    case INTERSIM:
      // FIXME: delete the object: may add icnt_done wrapper
      /* [한국어] FIXME: 시뮬레이션 종료 시 InterconnectInterface 객체를 delete하는 icnt_done() 래퍼 추가 필요 */
      g_icnt_interface = InterconnectInterface::New(g_network_config_filename);
      /* [한국어] g_network_config_filename 경로의 BookSim 설정 파일로 intersim2 객체 생성 */
      icnt_create = intersim2_create;             /* [한국어] 생성 함수 포인터를 intersim2_create로 바인딩 */
      icnt_init = intersim2_init;                 /* [한국어] 초기화 함수 포인터 바인딩 */
      icnt_has_buffer = intersim2_has_buffer;     /* [한국어] 버퍼 여유 확인 함수 포인터 바인딩 */
      icnt_push = intersim2_push;                 /* [한국어] 패킷 삽입 함수 포인터 바인딩 */
      icnt_pop = intersim2_pop;                   /* [한국어] 패킷 수신 함수 포인터 바인딩 */
      icnt_transfer = intersim2_transfer;         /* [한국어] 사이클 진행 함수 포인터 바인딩 */
      icnt_busy = intersim2_busy;                 /* [한국어] 바쁨 여부 확인 함수 포인터 바인딩 */
      icnt_display_stats = intersim2_display_stats; /* [한국어] 통계 출력 함수 포인터 바인딩 */
      icnt_display_overall_stats = intersim2_display_overall_stats; /* [한국어] 누적 통계 출력 함수 포인터 바인딩 */
      icnt_display_state = intersim2_display_state; /* [한국어] 상태 덤프 함수 포인터 바인딩 */
      icnt_get_flit_size = intersim2_get_flit_size; /* [한국어] 플릿 크기 반환 함수 포인터 바인딩 */
      break;
    case LOCAL_XBAR:
      g_localicnt_interface = LocalInterconnect::New(g_inct_config);
      /* [한국어] g_inct_config 설정으로 LocalInterconnect 객체 생성 (팩토리 메서드) */
      icnt_create = LocalInterconnect_create;     /* [한국어] 생성 함수 포인터를 LocalInterconnect_create로 바인딩 */
      icnt_init = LocalInterconnect_init;         /* [한국어] 초기화 함수 포인터 바인딩 */
      icnt_has_buffer = LocalInterconnect_has_buffer; /* [한국어] 버퍼 여유 확인 함수 포인터 바인딩 */
      icnt_push = LocalInterconnect_push;         /* [한국어] 패킷 삽입 함수 포인터 바인딩 */
      icnt_pop = LocalInterconnect_pop;           /* [한국어] 패킷 수신 함수 포인터 바인딩 */
      icnt_transfer = LocalInterconnect_transfer; /* [한국어] 사이클 진행 함수 포인터 바인딩 */
      icnt_busy = LocalInterconnect_busy;         /* [한국어] 바쁨 여부 확인 함수 포인터 바인딩 */
      icnt_display_stats = LocalInterconnect_display_stats; /* [한국어] 통계 출력 함수 포인터 바인딩 */
      icnt_display_overall_stats = LocalInterconnect_display_overall_stats; /* [한국어] 누적 통계 출력 함수 포인터 바인딩 */
      icnt_display_state = LocalInterconnect_display_state; /* [한국어] 상태 덤프 함수 포인터 바인딩 */
      icnt_get_flit_size = LocalInterconnect_get_flit_size; /* [한국어] 플릿 크기 반환 함수 포인터 바인딩 */
      break;
    default:
      assert(0);
      /* [한국어] 알 수 없는 네트워크 모드 — assert로 즉시 종료. 유효한 값은 1(INTERSIM) 또는 2(LOCAL_XBAR) */
      break;
  }
}
