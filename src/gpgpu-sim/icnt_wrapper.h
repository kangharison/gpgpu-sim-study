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
 * [한국어 설명] 인터커넥트 네트워크(ICNT) 래퍼 헤더 (icnt_wrapper.h)
 *
 * === 파일의 역할 ===
 * GPGPU-Sim의 온칩 네트워크(NoC, Network-on-Chip)를 추상화하는 함수 포인터
 * 인터페이스를 정의한다. 시뮬레이터 내부의 SM 클러스터(shader core)와
 * L2 메모리 파티션 사이의 패킷 전송을 담당하는 ICNT 모듈을 두 가지 구현체
 * (BookSim 기반 intersim2와 단순 크로스바 local_interconnect) 중 하나로
 * 교체 가능하게 한다. 런타임에 gpgpusim.config의 -network_mode 설정에 따라
 * 함수 포인터를 해당 구현체의 함수로 연결(icnt_wrapper_init)하는 전략 패턴을 사용한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * SM 클러스터(shader_core_ctx) ↔ L2 메모리 파티션(memory_partition_unit) 사이의
 * 패킷 전달 계층이다. gpu-sim.cc의 사이클 루프에서 icnt_transfer()가 매 사이클
 * 호출되어 ICNT 내부 상태를 진행시키고, SM 쪽은 icnt_push()로 요청 삽입,
 * 메모리 파티션 쪽은 icnt_pop()으로 도착 패킷을 꺼낸다.
 * 호출 체인: gpgpu_sim::cycle() → icnt_transfer() 및 icnt_push()/icnt_pop()
 * 실행 컨텍스트: 시뮬레이터 메인 스레드, 단일 스레드 동기 실행.
 *
 * === 타 모듈과의 연결 ===
 * - intersim2/interconnect_interface.hpp: BookSim NoC 구현 (INTERSIM 모드)
 * - local_interconnect.h: 단순 크로스바 구현 (LOCAL_XBAR 모드)
 * - icnt_wrapper.cc: 함수 포인터 변수 정의 및 icnt_wrapper_init() 구현
 * - shader.cc: icnt_push()로 SM→메모리 방향 mem_fetch 삽입
 * - gpu-sim.cc: icnt_transfer(), icnt_busy() 등 사이클 진행 및 종료 조건 확인
 * - option_parser: icnt_reg_options()를 통해 -network_mode 등 설정 등록
 *
 * === 주요 함수/구조체 요약 ===
 * icnt_create_p  - 네트워크 생성 함수 포인터 타입 (n_shader, n_mem으로 위상 설정)
 * icnt_push_p    - SM→메모리 또는 메모리→SM 방향 패킷 삽입 함수 포인터 타입
 * icnt_pop_p     - 도착 패킷 꺼내기 함수 포인터 타입 (반환: void* 데이터 포인터)
 * icnt_transfer_p - 매 사이클 ICNT 내부 상태 진행 함수 포인터 타입
 * icnt_wrapper_init() - g_network_mode에 따라 함수 포인터를 구현체에 연결
 * icnt_reg_options()  - -network_mode, -inter_config_file 등 ICNT 관련 옵션 등록
 */

#ifndef ICNT_WRAPPER_H
#define ICNT_WRAPPER_H

#include <stdio.h>  /* [한국어] FILE* 타입 사용을 위한 C 표준 I/O 헤더 (icnt_display_state_p 파라미터용) */

// functional interface to the interconnect
/* [한국어] ICNT 기능 인터페이스 — 함수 포인터 타입 정의 블록.
 * 전략 패턴(Strategy Pattern)으로, 런타임에 intersim2 또는 local_xbar 중 하나로
 * 바인딩하여 ICNT 구현체를 동적으로 교체 가능하게 한다. */

typedef void (*icnt_create_p)(unsigned n_shader, unsigned n_mem);
/* [한국어] 네트워크 토폴로지 생성 함수 포인터 타입.
 * n_shader: SM 클러스터 수(소스 노드), n_mem: 메모리 파티션 수(목적지 노드).
 * 구현체: intersim2_create() 또는 LocalInterconnect_create(). */

typedef void (*icnt_init_p)();
/* [한국어] 네트워크 초기화 함수 포인터 타입.
 * BookSim 설정 파일 로드 등 create() 이후 추가 초기화 수행.
 * 구현체: intersim2_init() 또는 LocalInterconnect_init(). */

typedef bool (*icnt_has_buffer_p)(unsigned input, unsigned int size);
/* [한국어] 지정 입력 포트에 size 크기의 패킷을 수용할 버퍼 여유가 있는지 확인하는 함수 포인터 타입.
 * input: 삽입하려는 소스 노드 ID. size: 패킷 크기(바이트 또는 플릿 단위).
 * 반환값 false이면 상위 레벨에서 back-pressure를 가해 삽입을 미룬다.
 * 구현체: intersim2_has_buffer() 또는 LocalInterconnect_has_buffer(). */

typedef void (*icnt_push_p)(unsigned input, unsigned output, void* data,
                            unsigned int size);
/* [한국어] 패킷을 ICNT에 삽입하는 함수 포인터 타입.
 * input: 소스 노드 ID (SM 클러스터 or 메모리 파티션 인덱스).
 * output: 목적지 노드 ID.
 * data: 전달할 mem_fetch 또는 기타 데이터 포인터.
 * size: 패킷 크기 (플릿 수 계산용).
 * icnt_has_buffer()로 사전 확인 후 호출해야 한다.
 * 구현체: intersim2_push() 또는 LocalInterconnect_push(). */

typedef void* (*icnt_pop_p)(unsigned output);
/* [한국어] 지정 출력 포트에 도착한 패킷을 꺼내는 함수 포인터 타입.
 * output: 수신 노드 ID. 반환값: 도착한 데이터 포인터(없으면 NULL).
 * 반환된 포인터는 통상 mem_fetch*로 캐스팅하여 사용한다.
 * 구현체: intersim2_pop() 또는 LocalInterconnect_pop(). */

typedef void (*icnt_transfer_p)();
/* [한국어] 매 시뮬레이션 사이클마다 ICNT 내부 상태를 한 사이클 진행시키는 함수 포인터 타입.
 * intersim2의 경우 BookSim 라우터의 Advance()를 호출하여 플릿을 한 홉씩 전진시킨다.
 * local_xbar의 경우 크로스바의 중재(arbitration) 논리를 실행한다.
 * 구현체: intersim2_transfer() 또는 LocalInterconnect_transfer(). */

typedef bool (*icnt_busy_p)();
/* [한국어] ICNT 내부에 아직 전달 중인 패킷이 있는지 확인하는 함수 포인터 타입.
 * 시뮬레이션 종료 전 ICNT 드레인(drain) 완료 여부 확인에 사용.
 * 반환값 true이면 아직 전달 중인 패킷이 존재함.
 * 구현체: intersim2_busy() 또는 LocalInterconnect_busy(). */

typedef void (*icnt_drain_p)();
/* [한국어] ICNT를 드레인(모든 in-flight 패킷을 목적지까지 전달)하는 함수 포인터 타입.
 * 현재 icnt_wrapper.cc에서 이 포인터는 정의만 되고 구현체가 바인딩되지 않는다.
 * (TODO 상태) 시뮬레이션 종료 시 정상 종료 플로우에 사용할 목적으로 예약된 슬롯. */

typedef void (*icnt_display_stats_p)();
/* [한국어] ICNT 통계(throughput, latency 등)를 출력하는 함수 포인터 타입.
 * 시뮬레이션 완료 후 결과 리포트에 포함된다.
 * 구현체: intersim2_display_stats() 또는 LocalInterconnect_display_stats(). */

typedef void (*icnt_display_overall_stats_p)();
/* [한국어] 전체 시뮬레이션의 누적 ICNT 통계를 출력하는 함수 포인터 타입.
 * 여러 커널 실행에 걸친 누적 통계용 (intersim2에서 별도 집계).
 * 구현체: intersim2_display_overall_stats() 또는 LocalInterconnect_display_overall_stats(). */

typedef void (*icnt_display_state_p)(FILE* fp);
/* [한국어] ICNT의 현재 내부 상태(라우터 버퍼, 링크 상태 등)를 fp에 출력하는 함수 포인터 타입.
 * 디버깅 및 상태 스냅샷 덤프에 사용.
 * 구현체: intersim2_display_state() 또는 LocalInterconnect_display_state(). */

typedef unsigned (*icnt_get_flit_size_p)();
/* [한국어] ICNT의 플릿(flit) 크기(바이트)를 반환하는 함수 포인터 타입.
 * 플릿(flit, flow control digit)은 NoC에서 흐름 제어의 최소 단위.
 * 패킷 크기를 플릿 수로 환산할 때 사용한다.
 * local_xbar는 40바이트 고정(LOCAL_INCT_FLIT_SIZE), intersim2는 설정 파일 기반.
 * 구현체: intersim2_get_flit_size() 또는 LocalInterconnect_get_flit_size(). */

extern icnt_create_p icnt_create;               /* [한국어] 네트워크 생성 함수 포인터 — icnt_wrapper_init()에서 구현체 바인딩 */
extern icnt_init_p icnt_init;                   /* [한국어] 네트워크 초기화 함수 포인터 */
extern icnt_has_buffer_p icnt_has_buffer;       /* [한국어] 버퍼 여유 확인 함수 포인터 — back-pressure 제어에 사용 */
extern icnt_push_p icnt_push;                   /* [한국어] 패킷 삽입 함수 포인터 — SM/메모리 파티션이 호출 */
extern icnt_pop_p icnt_pop;                     /* [한국어] 패킷 수신 함수 포인터 — 목적지 노드가 호출 */
extern icnt_transfer_p icnt_transfer;           /* [한국어] 매 사이클 ICNT 진행 함수 포인터 — gpu-sim의 사이클 루프에서 호출 */
extern icnt_busy_p icnt_busy;                   /* [한국어] ICNT 바쁨 여부 확인 함수 포인터 */
extern icnt_drain_p icnt_drain;                 /* [한국어] ICNT 드레인 함수 포인터 (현재 미바인딩 상태) */
extern icnt_display_stats_p icnt_display_stats; /* [한국어] 통계 출력 함수 포인터 */
extern icnt_display_overall_stats_p icnt_display_overall_stats; /* [한국어] 누적 통계 출력 함수 포인터 */
extern icnt_display_state_p icnt_display_state; /* [한국어] 내부 상태 덤프 함수 포인터 */
extern icnt_get_flit_size_p icnt_get_flit_size; /* [한국어] 플릿 크기 반환 함수 포인터 */
extern unsigned g_network_mode;                 /* [한국어] 현재 선택된 네트워크 모드 — INTERSIM(1) 또는 LOCAL_XBAR(2) */

/* [한국어] 네트워크 모드 열거형 — gpgpusim.config의 -network_mode 값과 대응.
 * INTERSIM(1): BookSim 기반 intersim2 — 정확한 NoC 레이턴시/대역폭 시뮬레이션.
 * LOCAL_XBAR(2): 단순 크로스바 버퍼 — 빠른 시뮬레이션을 위한 단순화 모델.
 * N_NETWORK_MODE: 모드 수 상한 (유효성 검사용). */
enum network_mode { INTERSIM = 1, LOCAL_XBAR = 2, N_NETWORK_MODE };

/*
 * [한국어]
 * icnt_wrapper_init - g_network_mode에 따라 ICNT 함수 포인터를 구현체에 바인딩
 *
 * @return: 없음 (void)
 *
 * icnt_reg_options()가 -network_mode 설정을 파싱한 뒤 호출된다.
 * g_network_mode가 INTERSIM이면 intersim2_* 함수들을 각 함수 포인터에 할당하고,
 * LOCAL_XBAR이면 LocalInterconnect_* 함수들을 할당한다.
 * 이후 코드는 icnt_push/icnt_pop 등을 직접 호출하여 구현체에 무관하게 동작한다.
 *
 * 호출 체인: gpgpusim_entrypoint.cc 초기화 → [이 함수]
 */
void icnt_wrapper_init();

/*
 * [한국어]
 * icnt_reg_options - OptionParser에 ICNT 관련 커맨드라인/설정 파일 옵션 등록
 *
 * @opp: gpgpusim.config 파싱에 사용되는 OptionParser 객체 포인터
 * @return: 없음 (void)
 *
 * -network_mode: 1(INTERSIM) 또는 2(LOCAL_XBAR)를 선택하는 정수 옵션.
 * -inter_config_file: intersim2 설정 파일 경로 (BookSim 토폴로지/라우팅 정의).
 * -icnt_in_buffer_limit, -icnt_out_buffer_limit: local_xbar 버퍼 크기.
 * -icnt_subnets: 서브네트워크 수 (기본 2: 요청망 + 응답망).
 * -icnt_arbiter_algo: 중재 알고리즘 (0: NAIVE_RR, 1: iSLIP).
 * -icnt_verbose, -icnt_grant_cycles: 디버그 출력 및 그랜트 사이클 설정.
 *
 * 호출 체인: gpgpusim_entrypoint.cc 옵션 파싱 단계 → [이 함수]
 */
void icnt_reg_options(class OptionParser* opp);

#endif
