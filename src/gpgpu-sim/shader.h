/*
 * [한국어 설명] GPGPU-Sim SM(Streaming Multiprocessor) 파이프라인 헤더 (shader.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim에서 GPU의 SM(Streaming Multiprocessor)을 사이클-레벨로 모델링하는
 * 모든 핵심 클래스와 구조체를 선언한다. CUDA/GPU 실행의 타이밍 모델 전체가 이 파일에서
 * 시작된다. SM 하나의 전체 파이프라인(fetch → decode → operand collect → execute →
 * writeback), warp 스케줄러, 오퍼랜드 콜렉터, 로드/스토어 유닛, SM 설정 파라미터,
 * SM 통계 수집 구조체, SM 클러스터까지 GPU 마이크로아키텍처 시뮬레이션에 필요한
 * 거의 모든 선언이 포함되어 있다.
 * shader.cc 및 exec_shader_core_ctx.cc 등에서 이 헤더를 포함하여 구현된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 시뮬레이션의 핵심 타이밍 모델 계층에 위치한다.
 * 호출 체인:
 *   gpu-sim.cc (gpgpu_sim::cycle) → simt_core_cluster::core_cycle()
 *       → shader_core_ctx::cycle() (= exec_shader_core_ctx)
 *           → fetch() → decode() → issue() → execute() → writeback()
 * 실행 컨텍스트: 호스트 유저스페이스, 싱글 스레드 시뮬레이션 루프.
 * GPU SM에서 매 사이클마다 파이프라인 스테이지들이 역방향(writeback→fetch)으로 호출된다.
 *
 * === 타 모듈과의 연결 ===
 * - abstract_hardware_model.h: warp_inst_t, core_t, active_mask_t, register_set 등
 *   상위 추상 타입을 공급받는다.
 * - scoreboard.h: Scoreboard 클래스(RAW 해저드 감지)를 사용한다.
 * - gpu-cache.h: L1I/L1D/L1T/L1C 캐시 클래스를 사용한다.
 * - mem_fetch.h: 메모리 요청 패킷(mem_fetch)을 생성·전달한다.
 * - dram.h / intersim2: mem_fetch가 NoC를 통해 DRAM 컨트롤러로 전달된다.
 * - gpu-sim.cc: simt_core_cluster를 통해 SM을 생성하고 사이클 함수를 호출한다.
 * - accelwattch: shader_core_stats의 카운터 값들이 전력 모델 입력으로 사용된다.
 * 설정 연동: gpgpusim.config의 -gpgpu_n_clusters, -gpgpu_n_cores_per_cluster,
 *   -gpgpu_shader_core_pipeline, -gpgpu_sched 등이 shader_core_config에 매핑된다.
 *
 * === 주요 함수/구조체 요약 ===
 * - shd_warp_t: SM 내 개별 warp 상태 (PC, 활성 마스크, 미완료 스레드 수, ibuffer 등)
 * - scheduler_unit (+ lrr/gto/two_level 파생): warp 스케줄러 추상 기반 클래스
 * - opndcoll_rfu_t: 오퍼랜드 콜렉터 RFU — 레지스터 파일 다중 뱅크 충돌 중재기
 * - ldst_unit: 로드/스토어 실행 유닛 (L1D/L1T/L1C/공유 메모리/원자 연산 처리)
 * - shader_core_config: SM 파라미터 집합 (파이프라인 너비, L1 캐시 설정, 스케줄러 타입 등)
 * - shader_core_stats: SM 통계 집계 (AccelWattch 전력 모델 입력 포함)
 * - shader_core_ctx: SM 전체를 모델링하는 핵심 클래스 (파이프라인 스테이지 구현)
 * - exec_shader_core_ctx: execution-driven 모드에서 실제 사용되는 shader_core_ctx 파생
 * - simt_core_cluster: 여러 SM을 묶는 클러스터 (ICNT 포트 공유)
 */
// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Andrew Turner,
// Ali Bakhoda, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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

#ifndef SHADER_H  /* [한국어] 헤더 중복 포함 방지 가드 — shader.h가 여러 번 include되어도 한 번만 처리 */
#define SHADER_H

#include <assert.h>   /* [한국어] assert() — 런타임 불변식 검사에 사용. 시뮬레이터 내 논리 오류 조기 감지 */
#include <math.h>     /* [한국어] 수학 함수 (floor, ceil 등) — 사이클 계산 및 레이턴시 계산에 사용 */
#include <stdio.h>    /* [한국어] fprintf/printf — 파이프라인 상태 출력·디버그 덤프에 사용 */
#include <stdlib.h>   /* [한국어] malloc/free/abort — 동적 메모리 및 치명 오류 처리에 사용 */
#include <algorithm>  /* [한국어] std::sort, std::find — warp 우선순위 정렬 등에 사용 */
#include <bitset>     /* [한국어] std::bitset — 활성 마스크(active_mask), 레지스터 준비 여부 추적에 사용 */
#include <deque>      /* [한국어] std::deque — two-level 스케줄러의 pending warp 큐에 사용 */
#include <list>       /* [한국어] std::list — 응답 FIFO(m_response_fifo), 아비터 요청 큐에 사용 */
#include <map>        /* [한국어] std::map — pending writes, LDGSTS 추적 맵에 사용 */
#include <set>        /* [한국어] std::set — 배리어 warp 집합 추적에 사용 */
#include <utility>    /* [한국어] std::pair — 키-값 쌍 반환 등에 사용 */
#include <vector>     /* [한국어] std::vector — warp 배열, FU 배열, 파이프라인 레지스터 배열에 사용 */

//#include "../cuda-sim/ptx.tab.h"  /* [한국어] PTX 파서 토큰 — 현재 미사용(주석 처리) */

#include "../abstract_hardware_model.h"  /* [한국어] GPU 추상 모델 — warp_inst_t, core_t, active_mask_t, register_set, simt_stack 등 상위 타입 정의 */
#include "delayqueue.h"                  /* [한국어] 지연 큐 — 파이프라인 레이턴시 시뮬레이션용 fifo 구조 */
#include "dram.h"                        /* [한국어] DRAM 타이밍 모델 — mem_fetch가 최종적으로 도달하는 메모리 컨트롤러 */
#include "gpu-cache.h"                   /* [한국어] 캐시 계층 — L1I, L1D, L1T, L1C 캐시 클래스 선언 */
#include "mem_fetch.h"                   /* [한국어] 메모리 요청 패킷 — SM이 캐시/DRAM으로 보내는 요청 객체 */
#include "scoreboard.h"                  /* [한국어] 스코어보드 — RAW(Read-After-Write) 해저드 감지용 레지스터 의존성 추적 */
#include "stack.h"                       /* [한국어] SIMT 스택 — warp divergence/reconvergence 처리용 post-dominance 스택 */
#include "stats.h"                       /* [한국어] 통계 타입 — cache_stats 등 캐시 통계 집계 구조체 */
#include "traffic_breakdown.h"           /* [한국어] NoC 트래픽 분석 — 코어-메모리 간 데이터 전송량 추적 */

/* [한국어] NO_OP_FLAG: 파이프라인 레지스터가 no-op(유효 명령 없음) 슬롯임을 표시하는 마커.
 * 0xFF 값은 유효 명령어와 구별 가능한 sentinel 값으로 사용된다. */
#define NO_OP_FLAG 0xFF

/* READ_PACKET_SIZE:
   bytes: 6 address (flit can specify chanel so this gives up to ~2GB/channel,
   so good for now), 2 bytes   [shaderid + mshrid](14 bits) + req_size(0-2 bits
   if req_size variable) - so up to 2^14 = 16384 mshr total
 */

/* [한국어] READ_PACKET_SIZE: 읽기 요청 패킷 크기 (바이트).
 * 6바이트 주소 + 2바이트 메타데이터(shader ID + MSHR ID 14비트 + 요청 크기 2비트).
 * MSHR(Miss Status Holding Register)은 L1 캐시 미스를 추적하는 버퍼로,
 * 최대 2^14 = 16384개의 미스를 동시에 추적할 수 있다. */
#define READ_PACKET_SIZE 8

// WRITE_PACKET_SIZE: bytes: 6 address, 2 miscelaneous.
/* [한국어] WRITE_PACKET_SIZE: 쓰기 요청 패킷 크기 (바이트). 읽기와 동일하게 8바이트. */
#define WRITE_PACKET_SIZE 8

/* [한국어] WRITE_MASK_SIZE: 쓰기 바이트 마스크 크기 (바이트). 어느 바이트를 실제로 쓸지 표시. */
#define WRITE_MASK_SIZE 8

class gpgpu_context;  /* [한국어] 전방 선언 — gpgpu_context는 시뮬레이터 전역 설정·옵션을 보유하는 최상위 컨텍스트 */

/*
 * [한국어] exec_unit_type_t — SM 내 실행 유닛(Function Unit) 종류 열거형.
 * warp_inst_t가 어느 FU에서 실행될지를 나타내며, 스케줄러가 issue할 출력 포트를
 * 선택할 때 사용된다. AccelWattch 전력 모델에서도 이 타입별로 전력을 카운트한다.
 */
enum exec_unit_type_t {
  NONE = 0,        /* [한국어] 실행 유닛 없음 (초기화/빈 슬롯) */
  SP = 1,          /* [한국어] SP(Single Precision) 유닛 — FP32 산술 연산 (FADD, FMUL 등) */
  SFU = 2,         /* [한국어] SFU(Special Function Unit) — 초월 함수 (sin, cos, sqrt, log 등) */
  MEM = 3,         /* [한국어] MEM(Memory) 유닛 — 로드/스토어/텍스처/원자 연산, ldst_unit으로 전달 */
  DP = 4,          /* [한국어] DP(Double Precision) 유닛 — FP64 산술 연산 */
  INT = 5,         /* [한국어] INT(Integer) 유닛 — 정수 ALU 연산 (IADD, IMUL 등) */
  TENSOR = 6,      /* [한국어] TENSOR 코어 유닛 — 행렬 곱 가속 (Turing/Volta+ 전용) */
  SPECIALIZED = 7  /* [한국어] 사용자 정의 특수 유닛 — specialized_unit_params로 설정 가능 */
};

/*
 * [한국어] thread_ctx_t — SM 내 개별 하드웨어 스레드 컨텍스트.
 * SM은 최대 MAX_THREAD_PER_SM 개의 스레드를 동시에 보유하며,
 * 각 스레드의 소속 CTA(Cooperative Thread Array), 명령어 실행 통계, 활성 여부를
 * 이 구조체에 저장한다. shader_core_ctx::m_threadState 배열로 관리된다.
 * 설정자: issue_block2core()에서 CTA 배정 시 초기화.
 * 읽는 자: 통계 수집(print), 기능 시뮬레이션(cuda-sim) 결과 반영 시.
 */
class thread_ctx_t {
 public:
  unsigned m_cta_id;  // hardware CTA this thread belongs
  /* [한국어] 이 스레드가 속한 하드웨어 CTA(블록) ID.
   * 설정자: reinit() / issue_block2core()에서 CTA 배정 시 설정.
   * 읽는 자: barrier_set_t가 CTA 내 warp 동기화 관리 시 사용.
   * 값 범위: 0 ~ max_cta_per_core-1. 미배정 시 유효하지 않은 값. */

  // per thread stats (ac stands for accumulative).
  unsigned n_insn;
  /* [한국어] 이 사이클까지 이 스레드가 커밋한 명령어 수 (비누적, 이번 실행 세션).
   * 설정자: warp_inst_complete() 호출 시 해당 스레드 lane 마다 증가.
   * 읽는 자: 통계 출력 함수. 동기화: 단일 시뮬레이션 스레드가 접근하므로 락 불필요. */
  unsigned n_insn_ac;
  /* [한국어] 누적(accumulative) 명령어 실행 카운터 — 재초기화 없이 누적.
   * 설정자/읽는 자: n_insn과 동일 경로. "ac" 접미사가 누적 카운터를 의미. */
  unsigned n_l1_mis_ac;
  /* [한국어] L1 데이터 캐시 미스 누적 횟수.
   * 설정자: ldst_unit에서 L1D miss 발생 시 해당 스레드 lane에 대해 증가.
   * 읽는 자: 통계 집계·출력 함수. */
  unsigned n_l1_mrghit_ac;
  /* [한국어] L1 캐시 병합 히트(merged hit) 누적 횟수 — 동일 캐시 라인에 여러 스레드가 접근할 때
   * 하나의 요청으로 병합 처리된 경우. 메모리 요청 병합 효율성 측정에 사용. */
  unsigned n_l1_access_ac;
  /* [한국어] L1 데이터 캐시 총 접근 누적 횟수 (히트+미스).
   * 설정자: ldst_unit memory_cycle() 경로에서 접근마다 증가.
   * 읽는 자: 캐시 miss rate 계산용 통계 출력 함수. */

  bool m_active;
  /* [한국어] 이 하드웨어 스레드 슬롯이 현재 활성 상태인지 여부.
   * true: 유효한 스레드가 배정되어 있음. false: 빈 슬롯 또는 완료된 스레드.
   * 설정자: reinit()에서 false 초기화, init_warps()에서 활성 스레드에 true 설정.
   * 읽는 자: isactive(), 통계 함수, 기능 시뮬레이션 컨텍스트 확인 시. */
};

/*
 * [한국어] shd_warp_t — SM 내 하나의 warp(32스레드 묶음) 상태를 표현하는 클래스.
 *
 * GPGPU-Sim에서 warp는 SM 파이프라인의 기본 스케줄링 단위이다. 각 warp는 최대 32개의
 * SIMT 스레드를 포함하며, 동일한 PC에서 동일한 명령어를 (활성 마스크에 따라) 실행한다.
 * SM 하나는 최대 max_warps_per_shader 개의 warp 슬롯을 가지며(shader_core_ctx::m_warp[]),
 * 각 슬롯이 shd_warp_t 인스턴스이다. warp가 활성(CTA 배정 상태)이면 fetch, decode,
 * issue, execute, writeback 단계를 순환하고, 모든 스레드가 완료되면 CTA 해제 후 재사용된다.
 *
 * 주요 상태:
 * - m_next_pc: fetch 유닛이 다음에 가져올 명령어 주소
 * - m_active_threads: 현재 살아있는 스레드들의 비트마스크 (warp divergence와 무관)
 * - n_completed: 완료된 스레드 수 (== warp_size이면 이 warp는 종료)
 * - m_ibuffer: instruction buffer — fetch로 가져온 명령어를 decode 전까지 보관 (2슬롯)
 * - m_imiss_pending: L1I 미스로 인해 fetch가 stall 중인지 여부
 * - m_membar: 메모리 배리어(MEMBAR/FENCE) 대기 중인지 여부
 * - m_stores_outstanding: 완료 확인을 기다리는 스토어 요청 수
 * - m_inst_in_pipeline: ibuffer에 있거나 파이프라인 스테이지에 걸려있는 명령어 총 수
 *
 * 호출 체인:
 *   shader_core_ctx::fetch() → shd_warp_t::ibuffer_fill(), set_imiss_pending()
 *   scheduler_unit::cycle() → shd_warp_t::ibuffer_next_inst(), ibuffer_free()
 *   ldst_unit::writeback() → shd_warp_t::dec_store_req(), clear_membar()
 */
class shd_warp_t {
 public:
  /*
   * [한국어] shd_warp_t 생성자 — warp 슬롯 초기화.
   * @shader: 이 warp가 속한 SM(shader_core_ctx) 포인터. fetch/issue 시 SM 상태 참조에 사용.
   * @warp_size: SM 설정의 warp 크기 (일반적으로 32). n_completed 초기값으로도 사용.
   * 생성 직후 reset()을 호출해 모든 상태를 초기화한다.
   */
  shd_warp_t(class shader_core_ctx *shader, unsigned warp_size)
      : m_shader(shader), m_warp_size(warp_size) {
    m_stores_outstanding = 0;   /* [한국어] 미완료 스토어 요청 수 0으로 초기화 — 이전 CTA 잔류값 방지 */
    m_inst_in_pipeline = 0;     /* [한국어] 파이프라인 내 명령어 수 0으로 초기화 */
    reset();                    /* [한국어] 나머지 모든 warp 상태 필드를 초기값으로 재설정 */
  }
  /*
   * [한국어] reset() — warp 슬롯을 "미사용(idle)" 상태로 완전 초기화.
   * CTA가 이 warp 슬롯에서 해제된 후 새 CTA가 배정되기 전에 호출된다.
   * m_stores_outstanding == 0, m_inst_in_pipeline == 0 조건을 assert로 확인한 뒤
   * 모든 상태를 idle 초기값으로 설정한다.
   * 호출자: shd_warp_t 생성자, register_cta_thread_exit() (CTA 완료 시).
   */
  void reset() {
    assert(m_stores_outstanding == 0);  /* [한국어] reset 호출 전 모든 스토어가 완료됐음을 보장 — 미완료 스토어가 있으면 reset은 불가 */
    assert(m_inst_in_pipeline == 0);   /* [한국어] 파이프라인에 잔류 명령어가 없음을 보장 — 누수 방지 */
    m_imiss_pending = false;           /* [한국어] 명령어 캐시 미스 대기 플래그 해제 — 새 fetch 가능 */
    m_warp_id = (unsigned)-1;          /* [한국어] 하드웨어 warp ID를 무효값으로 — 미할당 상태 표시 */
    m_dynamic_warp_id = (unsigned)-1;  /* [한국어] 동적 warp ID(실행 순서 고유 번호)를 무효값으로 초기화 */
    n_completed = m_warp_size;         /* [한국어] 완료 스레드 수를 warp_size로 설정 — 모든 스레드가 완료된 것처럼 처리 (idle 상태) */
    m_n_atomic = 0;                    /* [한국어] 미완료 원자 연산 수 0 — 원자 연산 완료 대기 없음 */
    m_membar = false;                  /* [한국어] 메모리 배리어 대기 해제 — MEMBAR 완료 또는 없음 */
    m_done_exit = true;                /* [한국어] 스레드 종료 처리 완료 플래그 true — idle 상태에서 기본값 */
    m_last_fetch = 0;                  /* [한국어] 마지막 fetch 사이클 번호 초기화 — fetch 간격 측정용 */
    m_next = 0;                        /* [한국어] ibuffer 다음 소비 포인터 0으로 초기화 */
    m_streamID = (unsigned long long)-1;  /* [한국어] 소속 CUDA 스트림 ID를 무효값으로 — 미할당 표시 */

    // Jin: cdp support
    m_cdp_latency = 0;    /* [한국어] CDP(CUDA Dynamic Parallelism) 레이턴시 카운터 초기화 */
    m_cdp_dummy = false;  /* [한국어] CDP 더미 warp 여부 플래그 초기화 */

    // Ni: Initialize ldgdepbar_id
    m_ldgdepbar_id = 0;     /* [한국어] LDGDEPBAR 명령어 배리어 ID 초기화 — 다음 배리어 번호 추적 */
    m_depbar_start_id = 0;  /* [한국어] DEPBAR 시작 ID 초기화 */
    m_depbar_group = 0;     /* [한국어] DEPBAR 그룹 번호 초기화 */

    // Ni: Set waiting to false
    m_waiting_ldgsts = false;  /* [한국어] LDGSTS(Load Global to Shared) 완료 대기 여부 초기화 */

    // Ni: Clear m_ldgdepbar_buf
    for (unsigned i = 0; i < m_ldgdepbar_buf.size(); i++) {
      m_ldgdepbar_buf[i].clear();  /* [한국어] 각 LDGDEPBAR 배리어 버퍼 내용 비우기 */
    }
    m_ldgdepbar_buf.clear();  /* [한국어] LDGDEPBAR 버퍼 벡터 자체도 비우기 — 메모리 해제 */
  }

  /*
   * [한국어] init() — CTA 배정 시 warp 슬롯을 실제 실행 상태로 초기화.
   * @start_pc: 이 warp가 fetch를 시작할 첫 번째 PC 주소.
   * @cta_id: 이 warp가 속한 CTA(블록) ID.
   * @wid: SM 내 하드웨어 warp 슬롯 번호 (0 ~ max_warps_per_shader-1).
   * @active: 이 warp에서 처음 활성화된 스레드들의 비트마스크.
   * @dynamic_warp_id: 이 실행에서 이 warp에 부여된 전역 고유 ID (GTO 스케줄러의 나이 비교 기준).
   * @streamID: 이 warp를 생성한 CUDA 스트림 ID.
   * 호출자: init_warps() (shader_core_ctx) — issue_block2core() 흐름에서 CTA를 SM에 배정할 때.
   */
  void init(address_type start_pc, unsigned cta_id, unsigned wid,
            const std::bitset<MAX_WARP_SIZE> &active, unsigned dynamic_warp_id,
            unsigned long long streamID) {
    m_streamID = streamID;             /* [한국어] 소속 CUDA 스트림 ID 기록 */
    m_cta_id = cta_id;                 /* [한국어] 소속 CTA ID 설정 */
    m_warp_id = wid;                   /* [한국어] 하드웨어 warp 슬롯 번호 설정 */
    m_dynamic_warp_id = dynamic_warp_id;  /* [한국어] 전역 고유 동적 warp ID 설정 — GTO 스케줄러의 나이 비교에 사용 */
    m_next_pc = start_pc;              /* [한국어] 첫 fetch 대상 PC 주소 설정 */
    assert(n_completed >= active.count());  /* [한국어] 현재 완료 카운터가 활성 스레드 수 이상임을 보장 */
    assert(n_completed <= m_warp_size);     /* [한국어] 완료 카운터가 warp 크기를 초과하지 않음을 보장 */
    n_completed -= active.count();  // active threads are not yet completed
    /* [한국어] 활성 스레드 수만큼 완료 카운터를 감소 — 활성 스레드들은 아직 완료되지 않았으므로 */
    m_active_threads = active;         /* [한국어] 이 warp의 초기 활성 스레드 비트마스크 설정 */
    m_done_exit = false;               /* [한국어] 스레드 종료 처리 완료 플래그 false — 아직 실행 중 */

    // Jin: cdp support
    m_cdp_latency = 0;    /* [한국어] CDP 레이턴시 카운터 0으로 초기화 */
    m_cdp_dummy = false;  /* [한국어] CDP 더미 warp 아님으로 초기화 */

    // Ni: Initialize ldgdepbar_id
    m_ldgdepbar_id = 0;     /* [한국어] LDGDEPBAR 배리어 ID를 0으로 리셋 */
    m_depbar_start_id = 0;  /* [한국어] DEPBAR 시작 ID 0으로 리셋 */
    m_depbar_group = 0;     /* [한국어] DEPBAR 그룹 번호 0으로 리셋 */

    // Ni: Set waiting to false
    m_waiting_ldgsts = false;  /* [한국어] LDGSTS 완료 대기 없음으로 초기화 */

    // Ni: Clear m_ldgdepbar_buf
    for (unsigned i = 0; i < m_ldgdepbar_buf.size(); i++) {
      m_ldgdepbar_buf[i].clear();  /* [한국어] 기존 LDGDEPBAR 버퍼 항목 비우기 */
    }
    m_ldgdepbar_buf.clear();  /* [한국어] LDGDEPBAR 버퍼 벡터 비우기 */
  }

  /*
   * [한국어] functional_done() — 기능 시뮬레이션(PTX 실행) 관점에서 warp 완료 여부 확인.
   * n_completed == m_warp_size이면 모든 스레드가 exit 명령을 실행했음을 의미한다.
   * 호출자: hardware_done()과 연계되어 CTA 해제 조건 판단에 사용.
   */
  bool functional_done() const;
  /*
   * [한국어] waiting() — warp가 현재 issue될 수 없는 대기 상태인지 확인.
   * not const: membar 상태 확인 중 내부 상태를 변경할 수 있기 때문.
   * 반환값 true: imiss_pending, membar 대기, 원자 연산 대기, cdp 레이턴시 등
   * 호출자: scheduler_unit::cycle()에서 warp가 issue 가능한지 판단.
   */
  bool waiting();  // not const due to membar
  /*
   * [한국어] hardware_done() — 하드웨어 타이밍 모델 관점에서 warp 완료 여부 확인.
   * functional_done()이면서 ibuffer가 비어있고 파이프라인에 명령어가 없어야 true.
   * 호출자: register_cta_thread_exit() — 모든 warp가 hardware_done이면 CTA를 해제.
   */
  bool hardware_done() const;

  /* [한국어] done_exit() — 이 warp의 모든 스레드가 exit 처리를 완료했는지 반환. */
  bool done_exit() const { return m_done_exit; }
  /* [한국어] set_done_exit() — 스레드 exit 처리 완료 마킹. register_cta_thread_exit()에서 호출. */
  void set_done_exit() { m_done_exit = true; }

  /* [한국어] print() — 이 warp 상태(PC, 완료 수, 활성 마스크 등)를 파일에 출력. 디버그용. */
  void print(FILE *fout) const;
  /* [한국어] print_ibuffer() — ibuffer 내 명령어들을 파일에 출력. 파이프라인 덤프용. */
  void print_ibuffer(FILE *fout) const;

  /* [한국어] get_n_completed() — 완료된 스레드 수 반환. == warp_size이면 전체 완료 상태. */
  unsigned get_n_completed() const { return n_completed; }
  /*
   * [한국어] set_completed() — 특정 lane(스레드)이 exit 명령 실행 완료 시 호출.
   * @lane: 완료된 스레드의 warp 내 인덱스 (0 ~ warp_size-1).
   * 활성 마스크에서 해당 비트를 제거하고 n_completed를 증가시킨다.
   * 호출자: func_exec_inst() 또는 checkExecutionStatusAndUpdate() 경로에서 exit 확인 시.
   */
  void set_completed(unsigned lane) {
    assert(m_active_threads.test(lane));  /* [한국어] lane이 실제로 활성 상태임을 검증 */
    m_active_threads.reset(lane);         /* [한국어] 해당 스레드를 활성 마스크에서 제거 */
    n_completed++;                        /* [한국어] 완료 카운터 증가 — warp_size가 되면 functional_done() */
  }

  /*
   * [한국어] set_last_fetch() — 마지막 fetch 시각 기록.
   * @sim_cycle: 현재 시뮬레이션 사이클 번호.
   * 호출자: shader_core_ctx::fetch()에서 이 warp의 명령어를 L1I로부터 가져왔을 때.
   * fetch 간격 통계 및 stall 감지에 사용.
   */
  void set_last_fetch(unsigned long long sim_cycle) {
    m_last_fetch = sim_cycle;  /* [한국어] 마지막 fetch 사이클 타임스탬프 기록 */
  }

  /* [한국어] get_n_atomic() — 현재 미완료 원자 연산(atomic) 수 반환. 0이 될 때까지 issue 불가. */
  unsigned get_n_atomic() const { return m_n_atomic; }
  /* [한국어] inc_n_atomic() — 원자 연산 요청 발행 시 카운터 증가. ldst_unit::issue()에서 호출. */
  void inc_n_atomic() { m_n_atomic++; }
  /* [한국어] dec_n_atomic() — 원자 연산 완료 응답 수신 시 카운터 감소. ldst_unit::fill()에서 호출. */
  void dec_n_atomic(unsigned n) { m_n_atomic -= n; }

  /* [한국어] set_membar() — MEMBAR(메모리 배리어) 명령 실행 시 대기 상태 진입. */
  void set_membar() { m_membar = true; }
  /* [한국어] clear_membar() — 배리어 조건 충족(미완료 메모리 연산 모두 완료) 시 해제. */
  void clear_membar() { m_membar = false; }
  /* [한국어] get_membar() — 현재 메모리 배리어 대기 중인지 반환. waiting()에서 확인. */
  bool get_membar() const { return m_membar; }
  /* [한국어] get_pc() — 이 warp의 다음 fetch 대상 PC 반환. scheduler_unit이 fetch 주소 결정에 사용. */
  virtual address_type get_pc() const { return m_next_pc; }
  /* [한국어] get_kernel_info() — 이 warp가 실행 중인 커널 정보 반환. cuda-sim 컨텍스트 확인용. */
  virtual kernel_info_t *get_kernel_info() const;
  /* [한국어] set_next_pc() — 분기/점프 실행 후 새로운 다음 PC 설정. SIMT 스택이 reconvergence 처리 시 호출. */
  void set_next_pc(address_type pc) { m_next_pc = pc; }

  /*
   * [한국어] store_info_of_last_inst_at_barrier() — barrier 도달 시 마지막 명령어 정보 저장.
   * @pI: barrier에서 멈춘 명령어 포인터.
   * barrier 해제 후 해당 명령어부터 재개하기 위해 복사본을 보관한다.
   */
  void store_info_of_last_inst_at_barrier(const warp_inst_t *pI) {
    m_inst_at_barrier = *pI;  /* [한국어] 명령어 내용을 복사 저장 — 포인터가 아닌 값 복사 */
  }
  /*
   * [한국어] restore_info_of_last_inst_at_barrier() — barrier 해제 후 저장된 명령어 복원.
   * 반환값: 저장된 m_inst_at_barrier의 포인터 — issue 재개 시 사용.
   */
  warp_inst_t *restore_info_of_last_inst_at_barrier() {
    return &m_inst_at_barrier;  /* [한국어] 저장된 barrier 명령어의 주소 반환 */
  }

  /*
   * [한국어] ibuffer_fill() — fetch된 명령어를 instruction buffer(ibuffer)에 채움.
   * @slot: 채울 ibuffer 슬롯 번호 (0 ~ IBUFFER_SIZE-1).
   * @pI: fetch된 warp_inst_t 포인터 (L1I 캐시에서 디코딩된 명령어).
   * 호출자: shader_core_ctx::decode()에서 L1I 응답을 ibuffer에 적재할 때.
   */
  void ibuffer_fill(unsigned slot, const warp_inst_t *pI) {
    assert(slot < IBUFFER_SIZE);     /* [한국어] 슬롯 번호가 유효 범위(0~1) 내임을 검증 */
    m_ibuffer[slot].m_inst = pI;    /* [한국어] 명령어 포인터 저장 */
    m_ibuffer[slot].m_valid = true; /* [한국어] 슬롯 유효 플래그 설정 */
    m_next = 0;                     /* [한국어] 소비 포인터를 슬롯 0부터 시작하도록 리셋 */
  }
  /*
   * [한국어] ibuffer_empty() — ibuffer가 비어있는지 확인.
   * 모든 슬롯이 invalid이면 true 반환. hardware_done() 조건 확인에 사용.
   */
  bool ibuffer_empty() const {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
      if (m_ibuffer[i].m_valid) return false;  /* [한국어] 유효 슬롯이 하나라도 있으면 false */
    return true;  /* [한국어] 모든 슬롯이 비어있음 */
  }
  /*
   * [한국어] ibuffer_flush() — ibuffer의 모든 슬롯을 강제로 비움.
   * 각 유효 슬롯에 대해 m_inst_in_pipeline을 감소시켜 누수 방지.
   * 호출자: 브랜치 잘못 예측 후 flush, 또는 warp kill 시.
   */
  void ibuffer_flush() {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid) dec_inst_in_pipeline();  /* [한국어] 유효 명령어가 파이프라인에서 제거됨을 카운터에 반영 */
      m_ibuffer[i].m_inst = NULL;   /* [한국어] 명령어 포인터 무효화 */
      m_ibuffer[i].m_valid = false; /* [한국어] 슬롯 유효 플래그 초기화 */
    }
  }
  /* [한국어] ibuffer_next_inst() — 현재 소비 포인터(m_next) 위치의 명령어 포인터 반환. 스케줄러가 다음 issue할 명령어 확인 시 사용. */
  const warp_inst_t *ibuffer_next_inst() { return m_ibuffer[m_next].m_inst; }
  /* [한국어] ibuffer_next_valid() — 현재 소비 포인터 위치 슬롯이 유효한지 반환. 스케줄러가 issue 가능 여부 판단 시 사용. */
  bool ibuffer_next_valid() { return m_ibuffer[m_next].m_valid; }
  /*
   * [한국어] ibuffer_free() — 현재 소비 포인터 위치 슬롯의 내용을 비움.
   * 명령어가 issue되어 파이프라인 레지스터로 이동한 후 ibuffer 슬롯을 해제.
   */
  void ibuffer_free() {
    m_ibuffer[m_next].m_inst = NULL;   /* [한국어] 명령어 포인터 제거 */
    m_ibuffer[m_next].m_valid = false; /* [한국어] 슬롯 무효화 */
  }
  /*
   * [한국어] ibuffer_step() — 소비 포인터를 다음 슬롯으로 이동 (환형 버퍼).
   * 명령어를 issue한 다음 사이클에 호출하여 다음 슬롯의 명령어를 준비.
   */
  void ibuffer_step() { m_next = (m_next + 1) % IBUFFER_SIZE; }

  /* [한국어] imiss_pending() — L1I 캐시 미스로 인한 fetch stall 중인지 반환. waiting()에서 확인. */
  bool imiss_pending() const { return m_imiss_pending; }
  /* [한국어] set_imiss_pending() — L1I 미스 발생 시 fetch stall 표시. shader_core_ctx::fetch()에서 호출. */
  void set_imiss_pending() { m_imiss_pending = true; }
  /* [한국어] clear_imiss_pending() — L1I 미스 응답 도착 시 stall 해제. accept_fetch_response()에서 호출. */
  void clear_imiss_pending() { m_imiss_pending = false; }

  /* [한국어] stores_done() — 미완료 스토어 요청이 없는지 확인. hardware_done() 조건 중 하나. */
  bool stores_done() const { return m_stores_outstanding == 0; }
  /* [한국어] inc_store_req() — 스토어 명령 issue 시 미완료 스토어 카운터 증가. ldst_unit::issue()에서 호출. */
  void inc_store_req() { m_stores_outstanding++; }
  /*
   * [한국어] dec_store_req() — 스토어 완료 응답(store_ack) 수신 시 카운터 감소.
   * 카운터가 0 아래로 내려가지 않도록 assert로 보호.
   * 호출자: shader_core_ctx::store_ack() (mem_fetch 응답 처리 시).
   */
  void dec_store_req() {
    assert(m_stores_outstanding > 0);  /* [한국어] 미완료 스토어가 최소 1개 있어야 감소 가능 */
    m_stores_outstanding--;            /* [한국어] 완료된 스토어 반영 */
  }

  /* [한국어] num_inst_in_buffer() — ibuffer에 있는 유효 명령어 수 반환. 통계 출력용. */
  unsigned num_inst_in_buffer() const {
    unsigned count = 0;
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid) count++;  /* [한국어] 유효 슬롯 카운트 */
    }
    return count;
  }
  /* [한국어] num_inst_in_pipeline() — ibuffer + 파이프라인 스테이지에 걸린 전체 명령어 수 반환. */
  unsigned num_inst_in_pipeline() const { return m_inst_in_pipeline; }
  /*
   * [한국어] num_issued_inst_in_pipeline() — ibuffer를 제외하고 실제 파이프라인 스테이지에
   * 진입한 명령어 수 반환 (= 전체 - ibuffer 내).
   */
  unsigned num_issued_inst_in_pipeline() const {
    return (num_inst_in_pipeline() - num_inst_in_buffer());
  }
  /* [한국어] inst_in_pipeline() — 파이프라인(ibuffer 포함)에 명령어가 하나라도 있는지 확인. */
  bool inst_in_pipeline() const { return m_inst_in_pipeline > 0; }
  /* [한국어] inc_inst_in_pipeline() — ibuffer_fill 또는 issue 시 파이프라인 내 명령어 수 증가. */
  void inc_inst_in_pipeline() { m_inst_in_pipeline++; }
  /*
   * [한국어] dec_inst_in_pipeline() — writeback 또는 ibuffer_flush 시 파이프라인 내 명령어 수 감소.
   * assert로 0 아래로 내려가지 않도록 보호.
   */
  void dec_inst_in_pipeline() {
    assert(m_inst_in_pipeline > 0);  /* [한국어] 파이프라인에 명령어가 있음을 검증 */
    m_inst_in_pipeline--;            /* [한국어] 명령어 완료 반영 */
  }

  /* [한국어] get_streamID() — 소속 CUDA 스트림 ID 반환. 스트림 통계 및 CDP 지원에 사용. */
  unsigned long long get_streamID() const { return m_streamID; }
  /* [한국어] get_cta_id() — 소속 CTA ID 반환. barrier_set_t가 CTA 내 warp 관리 시 사용. */
  unsigned get_cta_id() const { return m_cta_id; }

  /* [한국어] get_dynamic_warp_id() — 전역 고유 동적 warp ID 반환. GTO/oldest 스케줄러의 나이 비교 기준. */
  unsigned get_dynamic_warp_id() const { return m_dynamic_warp_id; }
  /* [한국어] get_warp_id() — 하드웨어 warp 슬롯 번호 반환. 스코어보드 인덱스, 레지스터 뱅크 계산에 사용. */
  unsigned get_warp_id() const { return m_warp_id; }

  /*
   * [한국어] get_shader() — 이 warp가 속한 SM(shader_core_ctx) 포인터 반환.
   * warp 내부에서 SM 상태를 참조해야 할 때 사용. 주로 cuda-sim 기능 시뮬레이션에서 호출.
   */
  class shader_core_ctx *get_shader() {
    return m_shader;  /* [한국어] 소속 SM 포인터 반환 */
  }

 private:
  static const unsigned IBUFFER_SIZE = 2;
  /* [한국어] IBUFFER_SIZE: instruction buffer 슬롯 수. 2슬롯 — fetch된 명령어를 decode까지 보관.
   * 2슬롯으로 제한함으로써 fetch 대역폭과 issue 대역폭 간의 간단한 분리를 모델링한다. */

  class shader_core_ctx *m_shader;
  /* [한국어] 이 warp가 속한 SM(shader_core_ctx) 포인터.
   * 설정자: shd_warp_t 생성자에서 초기화.
   * 읽는 자: get_shader(), get_kernel_info() 등에서 SM 상태 참조 시.
   * 동기화: 단일 시뮬레이션 스레드가 접근, 락 불필요. */

  unsigned long long m_streamID;
  /* [한국어] 이 warp를 발행한 CUDA 스트림 ID.
   * 설정자: init()에서 CTA 배정 시.
   * 읽는 자: get_streamID(), CDP 지원 로직, 스트림 통계.
   * 값 범위: 유효 스트림 ID 또는 (unsigned long long)-1 (idle). */

  unsigned m_cta_id;
  /* [한국어] 이 warp가 속한 하드웨어 CTA(블록) ID.
   * 설정자: init()에서 설정.
   * 읽는 자: get_cta_id(), barrier_set_t에서 CTA 단위 배리어 관리 시.
   * 값 범위: 0 ~ max_cta_per_core-1. */

  unsigned m_warp_id;
  /* [한국어] SM 내 하드웨어 warp 슬롯 번호 (정적 ID).
   * 설정자: init()에서 CTA 배정 시.
   * 읽는 자: 스코어보드 인덱스, register_bank() 계산, warp 스케줄러 발행 결정 시.
   * 값 범위: 0 ~ max_warps_per_shader-1. */

  unsigned m_warp_size;
  /* [한국어] 이 warp의 스레드 수 (일반적으로 32).
   * 설정자: 생성자에서 shader_core_config::warp_size로부터 전달.
   * 읽는 자: n_completed 비교, hw_tid_from_wid() 등 스레드 ID 계산 시.
   * 값 범위: 반드시 MAX_WARP_SIZE 이하. */

  unsigned m_dynamic_warp_id;
  /* [한국어] 실행 순서 고유 동적 warp ID — SM 전체 수명 동안 단조 증가하는 번호.
   * 설정자: init()에서 shader_core_ctx::m_dynamic_warp_id++로 할당.
   * 읽는 자: GTO/oldest 스케줄러가 warp 나이 비교(오래된 warp 우선) 시 사용.
   * warp_id(슬롯)와 달리 재사용되지 않아 실행 순서를 정확히 추적 가능. */

  address_type m_next_pc;
  /* [한국어] 다음 사이클에 fetch할 PC(프로그램 카운터) 주소.
   * 설정자: init()에서 초기 PC 설정, set_next_pc()에서 분기 후 갱신.
   * 읽는 자: shader_core_ctx::fetch()가 L1I 요청 주소로 사용, get_pc() 반환.
   * 값 범위: 유효한 PTX/SASS 명령어 주소. */

  unsigned n_completed;  // number of threads in warp completed
  /* [한국어] 이 warp에서 exit 명령을 실행하고 종료된 스레드 수.
   * 설정자: set_completed()에서 스레드 완료 시 증가, init()에서 비활성 스레드 수로 초기화.
   * 읽는 자: functional_done() (== warp_size이면 warp 전체 완료), get_n_completed().
   * 값 범위: 0 ~ m_warp_size. m_warp_size이면 CTA 해제 조건 판단 대상. */

  std::bitset<MAX_WARP_SIZE> m_active_threads;
  /* [한국어] 이 warp에서 현재 살아있는(exit 미실행) 스레드들의 비트마스크.
   * SIMT divergence와 관계없이 아직 종료하지 않은 스레드를 표시한다.
   * 설정자: init()에서 초기 active 마스크 설정, set_completed()에서 종료 스레드 제거.
   * 읽는 자: 스케줄러, functional_done(), 기능 시뮬레이션.
   * 동기화: 단일 시뮬레이션 스레드 접근이므로 락 불필요. */

  bool m_imiss_pending;
  /* [한국어] L1I(instruction cache) 미스가 발생하여 응답 대기 중인지 여부.
   * true이면 fetch()에서 이 warp에 새로운 fetch 요청을 보내지 않는다.
   * 설정자: shader_core_ctx::fetch()에서 L1I miss 시 set_imiss_pending() 호출.
   * 읽는 자: imiss_pending(), waiting()에서 issue 불가 상태 판단.
   * 해제자: accept_fetch_response()에서 L1I 응답 도착 시 clear_imiss_pending() 호출. */

  /*
   * [한국어] ibuffer_entry — instruction buffer의 단일 슬롯 구조체.
   * fetch()에서 가져온 명령어를 decode()가 처리하기 전까지 임시 보관한다.
   * IBUFFER_SIZE(=2) 개의 슬롯이 있어 최대 2개의 명령어를 버퍼링할 수 있다.
   */
  struct ibuffer_entry {
    ibuffer_entry() {
      m_valid = false;  /* [한국어] 생성 시 슬롯 무효 상태로 초기화 */
      m_inst = NULL;    /* [한국어] 명령어 포인터 NULL로 초기화 */
    }
    const warp_inst_t *m_inst;
    /* [한국어] fetch된 명령어 포인터 (L1I 또는 기능 시뮬레이션에서 제공).
     * 설정자: ibuffer_fill()에서 설정. 읽는 자: ibuffer_next_inst(), 스케줄러.
     * 포인터가 가리키는 명령어 객체의 소유권은 이 슬롯에 없음 (단순 참조). */
    bool m_valid;
    /* [한국어] 이 슬롯에 유효한 명령어가 있는지 여부.
     * 설정자: ibuffer_fill()에서 true, ibuffer_free()/ibuffer_flush()에서 false.
     * 읽는 자: ibuffer_empty(), ibuffer_next_valid(), num_inst_in_buffer(). */
  };

  warp_inst_t m_inst_at_barrier;
  /* [한국어] barrier 도달 시점의 마지막 명령어를 저장하는 임시 버퍼.
   * barrier가 해제되면 이 명령어부터 실행을 재개할 수 있도록 보관.
   * 설정자: store_info_of_last_inst_at_barrier(). 읽는 자: restore_info_of_last_inst_at_barrier(). */

  ibuffer_entry m_ibuffer[IBUFFER_SIZE];
  /* [한국어] 2슬롯 instruction buffer 배열.
   * m_ibuffer[0], m_ibuffer[1] 두 슬롯에 각각 fetch된 명령어를 저장한다.
   * 설정자: ibuffer_fill()에서 슬롯 번호 지정 후 채움.
   * 읽는 자: ibuffer_next_inst(), 스케줄러의 issue 결정 시. */

  unsigned m_next;
  /* [한국어] ibuffer 환형 버퍼의 현재 소비 포인터 — 다음에 issue할 슬롯 인덱스.
   * 설정자: ibuffer_fill()에서 0으로 리셋, ibuffer_step()에서 증가.
   * 읽는 자: ibuffer_next_inst(), ibuffer_next_valid(), ibuffer_free(). */

  unsigned m_n_atomic;  // number of outstanding atomic operations
  /* [한국어] 발행되었지만 아직 완료 응답이 없는 원자 연산(atomic) 수.
   * 원자 연산은 완료 순서 보장이 필요하여, 미완료 원자가 있으면 새 issue 불가.
   * 설정자: inc_n_atomic()(ldst_unit::issue 시), dec_n_atomic()(응답 수신 시).
   * 읽는 자: waiting()에서 m_n_atomic > 0이면 issue 블로킹. */

  bool m_membar;        // if true, warp is waiting at memory barrier
  /* [한국어] MEMBAR/FENCE 명령 실행으로 인한 메모리 배리어 대기 상태.
   * true이면 모든 이전 메모리 연산이 완료될 때까지 이 warp를 issue하지 않는다.
   * 설정자: set_membar()(MEMBAR 명령 실행 시), 해제자: clear_membar()(조건 충족 시).
   * 읽는 자: waiting() — get_membar()로 확인. */

  bool m_done_exit;  // true once thread exit has been registered for threads in
                     // this warp
  /* [한국어] 이 warp의 모든 스레드 종료 처리가 완료되었음을 표시하는 플래그.
   * true: exit 처리 완료 (warp 슬롯 재사용 가능), false: 실행 중.
   * 설정자: set_done_exit()(warp_exit 처리 완료 시), 초기값: reset()에서 true.
   * 읽는 자: done_exit(), hardware_done() 조건 판단. */

  unsigned long long m_last_fetch;
  /* [한국어] 이 warp에 마지막으로 fetch를 수행한 시뮬레이션 사이클 번호.
   * 설정자: set_last_fetch()에서 fetch() 성공 시 기록.
   * 읽는 자: fetch 통계 출력, fetch stall 간격 분석. */

  unsigned m_stores_outstanding;  // number of store requests sent but not yet
                                  // acknowledged
  /* [한국어] 발행되었지만 완료 확인(store_ack)이 아직 도착하지 않은 스토어 요청 수.
   * 설정자: inc_store_req()(스토어 issue 시), dec_store_req()(store_ack 수신 시).
   * 읽는 자: stores_done() — 0이 아니면 CTA 해제 불가, waiting()에서 확인. */

  unsigned m_inst_in_pipeline;
  /* [한국어] ibuffer 포함 파이프라인 전 스테이지에 걸쳐 있는 이 warp의 명령어 총 수.
   * fetch → ibuffer → operand collect → execute → writeback 전 과정의 명령어 추적.
   * 설정자: inc_inst_in_pipeline()(fetch/fill 시 증가), dec_inst_in_pipeline()(writeback/flush 시 감소).
   * 읽는 자: hardware_done()(== 0이어야 완료), inst_in_pipeline(), num_inst_in_pipeline(). */

  // Jin: cdp support
 public:
  unsigned int m_cdp_latency;
  /* [한국어] CDP(CUDA Dynamic Parallelism) 관련 레이턴시 카운터.
   * GPU 커널이 새 커널을 동적으로 launch할 때 발생하는 추가 레이턴시를 모델링.
   * 설정자/읽는 자: CDP 관련 cuda-sim 경로. 0이 아닌 동안 warp issue 지연. */
  bool m_cdp_dummy;
  /* [한국어] CDP 더미 warp 여부 — CDP에 의해 생성된 빈 warp 마킹에 사용.
   * 설정자: CDP 커널 launch 처리 시. 읽는 자: waiting() 또는 스케줄러에서 필터링. */

  // Ni: LDGDEPBAR barrier support
 public:
  unsigned int m_ldgdepbar_id;  // LDGDEPBAR barrier ID
  /* [한국어] LDGDEPBAR(Load Global Dependency Barrier) 명령의 현재 배리어 ID.
   * LDGDEPBAR는 Ampere GPU에서 비동기 전역 메모리 로드와 공유 메모리 쓰기 간 종속성 관리 명령.
   * 설정자: LDGDEPBAR 명령 실행 시 증가. 읽는 자: 배리어 완료 조건 판단 시. */

  std::vector<std::vector<warp_inst_t>>
      m_ldgdepbar_buf;  // LDGDEPBAR barrier buffer
  /* [한국어] LDGDEPBAR 배리어에 대기 중인 명령어들의 버퍼.
   * 외부 인덱스: 배리어 그룹, 내부 벡터: 해당 그룹에 대기 중인 warp_inst_t 목록.
   * 설정자: LDGDEPBAR 명령 처리 시 대기 명령어 추가. 읽는 자: 배리어 해제 시 명령어 재발행. */

  unsigned int m_depbar_start_id;
  /* [한국어] DEPBAR 시작 그룹 ID — 현재 처리 중인 배리어 그룹의 시작 번호 추적. */
  unsigned int m_depbar_group;
  /* [한국어] DEPBAR 그룹 번호 — 동시에 여러 개의 비동기 배리어를 그룹으로 관리. */
  bool m_waiting_ldgsts;  // Ni: Whether the warp is waiting for the LDGSTS
                          // instrs to finish
  /* [한국어] LDGSTS(Load Global to Shared, 비동기 전역→공유메모리 로드) 완료 대기 여부.
   * true이면 LDGSTS 명령의 비동기 완료를 기다리는 중 — issue 블로킹.
   * 설정자: LDGSTS 명령 발행 시 true. 해제자: 비동기 로드 완료 확인 시 false. */
};

/*
 * [한국어] hw_tid_from_wid() — warp ID와 warp 내 레인 인덱스로 전역 하드웨어 스레드 ID 계산.
 * @wid: SM 내 warp 슬롯 번호.
 * @warp_size: warp당 스레드 수 (보통 32).
 * @i: warp 내 레인 인덱스 (0 ~ warp_size-1).
 * @return: SM 내 전역 하드웨어 스레드 ID (0 ~ max_threads_per_SM-1).
 * 호출 체인: 기능 시뮬레이션에서 스레드별 메모리 주소 계산 시 사용.
 */
inline unsigned hw_tid_from_wid(unsigned wid, unsigned warp_size, unsigned i) {
  return wid * warp_size + i;  /* [한국어] 하드웨어 스레드 ID = warp 번호 × warp 크기 + 레인 인덱스 */
};
/*
 * [한국어] wid_from_hw_tid() — 전역 하드웨어 스레드 ID로부터 warp 슬롯 번호 역산.
 * @tid: SM 내 전역 하드웨어 스레드 ID.
 * @warp_size: warp당 스레드 수.
 * @return: 이 스레드가 속한 warp 슬롯 번호.
 */
inline unsigned wid_from_hw_tid(unsigned tid, unsigned warp_size) {
  return tid / warp_size;  /* [한국어] warp ID = 전역 스레드 ID / warp 크기 (정수 나눗셈) */
};

/* [한국어] WARP_PER_CTA_MAX: CTA(블록) 하나에 포함될 수 있는 최대 warp 수 (= 64).
 * 32스레드/warp × 64warp = 2048 스레드/CTA — CUDA 최대 스레드/블록 제한과 일치. */
const unsigned WARP_PER_CTA_MAX = 64;
/* [한국어] warp_set_t: CTA 내 warp 비트마스크 타입. barrier_set_t에서 CTA 내 warp 상태 추적에 사용. */
typedef std::bitset<WARP_PER_CTA_MAX> warp_set_t;

/*
 * [한국어] register_bank() — 레지스터 번호와 warp ID로 레지스터 파일 뱅크 번호 계산.
 * @regnum: 레지스터 번호.
 * @wid: warp ID.
 * @num_banks: 총 레지스터 파일 뱅크 수 (-gpgpu_num_reg_banks 설정).
 * @sub_core_model: sub-core 모델 사용 여부 (-sub_core_model 설정).
 * @banks_per_sched: 스케줄러당 뱅크 수 (sub_core_model 활성 시 유효).
 * @sched_id: 이 명령을 issue한 스케줄러 ID.
 * @return: 접근해야 할 레지스터 파일 뱅크 번호.
 * 오퍼랜드 콜렉터(opndcoll_rfu_t)가 뱅크 충돌 감지 시 이 함수를 사용한다.
 * 구현: scoreboard.cc 또는 shader.cc에 위치.
 */
unsigned register_bank(int regnum, int wid, unsigned num_banks,
                       bool sub_core_model, unsigned banks_per_sched,
                       unsigned sched_id);

class shader_core_ctx;    /* [한국어] 전방 선언 — SM 전체를 모델링하는 핵심 클래스 */
class shader_core_config; /* [한국어] 전방 선언 — SM 설정 파라미터 클래스 */
class shader_core_stats;  /* [한국어] 전방 선언 — SM 통계 집계 클래스 */

/*
 * [한국어] scheduler_prioritization_type — warp 우선순위 결정 알고리즘 열거형.
 * scheduler_unit::order_by_priority()와 two_level_active_scheduler의 내부/외부 레벨
 * 우선순위 설정에 사용된다. gpgpusim.config의 -gpgpu_scheduler 옵션으로 선택.
 */
enum scheduler_prioritization_type {
  SCHEDULER_PRIORITIZATION_LRR = 0,   // Loose Round Robin
  /* [한국어] LRR(Loose Round Robin): 마지막으로 issue된 warp 다음부터 순환 — 느슨한 라운드로빈 */
  SCHEDULER_PRIORITIZATION_SRR,       // Strict Round Robin
  /* [한국어] SRR(Strict Round Robin): 항상 warp 번호 0부터 순서대로 — 엄격한 라운드로빈 */
  SCHEDULER_PRIORITIZATION_GTO,       // Greedy Then Oldest
  /* [한국어] GTO(Greedy Then Oldest): 직전에 issue된 warp를 먼저, 그 다음 가장 오래된 warp 순 */
  SCHEDULER_PRIORITIZATION_GTLRR,     // Greedy Then Loose Round Robin
  /* [한국어] GTLRR: 직전 warp 우선 후 LRR 순서로 — GTO와 LRR의 조합 */
  SCHEDULER_PRIORITIZATION_GTY,       // Greedy Then Youngest
  /* [한국어] GTY: 직전 warp 우선 후 가장 젊은(최근 생성) warp 순 */
  SCHEDULER_PRIORITIZATION_OLDEST,    // Oldest First
  /* [한국어] OLDEST: 동적 warp ID가 가장 작은(가장 오래된) warp를 항상 우선 */
  SCHEDULER_PRIORITIZATION_YOUNGEST,  // Youngest First
  /* [한국어] YOUNGEST: 동적 warp ID가 가장 큰(가장 최근) warp를 항상 우선 */
};

// Each of these corresponds to a string value in the gpgpsim.config file
// For example - to specify the LRR scheudler the config must contain lrr
/*
 * [한국어] concrete_scheduler — gpgpusim.config에서 선택 가능한 실제 스케줄러 종류.
 * 문자열 "lrr", "gto", "two_level_active:N:X:Y", "rrr", "warp_limiting:N:X", "oldest_first"로
 * 설정 파일에서 지정한다. 각 값은 대응하는 scheduler_unit 파생 클래스에 매핑된다.
 */
enum concrete_scheduler {
  CONCRETE_SCHEDULER_LRR = 0,          /* [한국어] Loose Round Robin 스케줄러 (lrr_scheduler) */
  CONCRETE_SCHEDULER_GTO,              /* [한국어] Greedy Then Oldest 스케줄러 (gto_scheduler) */
  CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE, /* [한국어] 두 레벨 활성 warp 스케줄러 (two_level_active_scheduler) */
  CONCRETE_SCHEDULER_RRR,              /* [한국어] Rigid Round Robin 스케줄러 (rrr_scheduler) */
  CONCRETE_SCHEDULER_WARP_LIMITING,    /* [한국어] 정적 warp 수 제한 스케줄러 (swl_scheduler) */
  CONCRETE_SCHEDULER_OLDEST_FIRST,     /* [한국어] 가장 오래된 warp 우선 스케줄러 (oldest_scheduler) */
  NUM_CONCRETE_SCHEDULERS              /* [한국어] 스케줄러 종류 총 수 — 배열 크기 계산용 */
};

/*
 * [한국어] scheduler_unit — warp 스케줄러 추상 기반 클래스.
 *
 * SM 내 하나의 warp 스케줄러를 모델링한다. SM은 gpgpu_num_sched_per_core 개의
 * scheduler_unit 인스턴스를 가지며, 각 스케줄러는 자신이 담당하는 warp 집합
 * (m_supervised_warps)에서 매 사이클마다 issue 가능한 warp를 선택한다.
 *
 * cycle() 메서드가 매 사이클 호출되어 m_next_cycle_prioritized_warps 목록에서
 * 스코어보드 충돌, 기능 유닛 준비 상태, warp 대기 여부를 체크하고 최대
 * gpgpu_max_insn_issue_per_warp 개의 명령어를 issue한다.
 *
 * 파생 클래스는 order_warps()를 구현하여 m_next_cycle_prioritized_warps 순서를 결정한다:
 * - lrr_scheduler: Loose Round Robin
 * - gto_scheduler: Greedy Then Oldest
 * - two_level_active_scheduler: 활성 pool + 대기 pool의 두 레벨
 * - swl_scheduler: Static Warp Limiting
 * - oldest_scheduler: 동적 warp ID 기준 최고령 우선
 *
 * 실행 컨텍스트: shader_core_ctx::issue()에서 호출. 단일 시뮬레이션 스레드.
 * 호출 체인: shader_core_ctx::cycle() → issue() → scheduler_unit::cycle() → issue_warp()
 */
class scheduler_unit {  // this can be copied freely, so can be used in std
                        // containers.
 public:
  /*
   * [한국어] scheduler_unit 생성자 — 스케줄러에 필요한 SM 자원 포인터들을 주입받아 초기화.
   * @stats: SM 통계 객체 포인터.
   * @shader: 소속 SM 포인터.
   * @scoreboard: 레지스터 의존성 추적 스코어보드.
   * @simt: SIMT 스택 배열 (warp divergence 처리용).
   * @warp: SM의 전체 warp 배열 포인터.
   * @sp_out ~ @mem_out: 각 FU로 향하는 출력 파이프라인 레지스터 집합 포인터.
   * @spec_cores_out: 특수 유닛 출력 레지스터 벡터.
   * @id: 이 스케줄러의 SM 내 ID (sub-core 모델에서 뱅크 파티셔닝에 사용).
   */
  scheduler_unit(shader_core_stats *stats, shader_core_ctx *shader,
                 Scoreboard *scoreboard, simt_stack **simt,
                 std::vector<shd_warp_t *> *warp, register_set *sp_out,
                 register_set *dp_out, register_set *sfu_out,
                 register_set *int_out, register_set *tensor_core_out,
                 std::vector<register_set *> &spec_cores_out,
                 register_set *mem_out, int id)
      : m_supervised_warps(),
        m_stats(stats),
        m_shader(shader),
        m_scoreboard(scoreboard),
        m_simt_stack(simt),
        /*m_pipeline_reg(pipe_regs),*/ m_warp(warp),
        m_sp_out(sp_out),
        m_dp_out(dp_out),
        m_sfu_out(sfu_out),
        m_int_out(int_out),
        m_tensor_core_out(tensor_core_out),
        m_mem_out(mem_out),
        m_spec_cores_out(spec_cores_out),
        m_id(id) {}
  virtual ~scheduler_unit() {}
  /*
   * [한국어] add_supervised_warp_id() — 이 스케줄러가 담당할 warp 슬롯을 등록.
   * @i: 등록할 warp 슬롯 번호.
   * SM 초기화 시 scheduler_unit별로 warp 슬롯을 분배할 때 호출된다.
   * 여러 스케줄러가 있을 때 각 스케줄러는 자신의 warp 부분집합만 관리한다.
   */
  virtual void add_supervised_warp_id(int i) {
    m_supervised_warps.push_back(&warp(i));  /* [한국어] 담당 warp 포인터를 supervised 목록에 추가 */
  }
  /*
   * [한국어] done_adding_supervised_warps() — 담당 warp 등록 완료 후 last_supervised_issued 초기화.
   * 기본 구현은 end()로 설정 (LRR: 마지막 다음부터 시작). GTO는 begin()으로 오버라이드.
   */
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.end();  /* [한국어] LRR 기본값: 목록 끝 다음부터 순환 시작 */
  }

  // The core scheduler cycle method is meant to be common between
  // all the derived schedulers.  The scheduler's behaviour can be
  // modified by changing the contents of the m_next_cycle_prioritized_warps
  // list.
  /*
   * [한국어] cycle() — 매 SM 사이클마다 호출되는 스케줄러 핵심 메서드.
   * order_warps()로 m_next_cycle_prioritized_warps를 갱신한 뒤,
   * 우선순위 순으로 warp를 순회하며 스코어보드 충돌/waiting 여부/FU 가용성을 확인하여
   * 최대 gpgpu_max_insn_issue_per_warp 개의 명령어를 issue_warp()로 발행한다.
   * 실행 컨텍스트: shader_core_ctx::issue() 내에서 호출 (단일 시뮬레이션 스레드).
   * 호출 체인: shader_core_ctx::cycle() → issue() → scheduler_unit::cycle() → shader_core_ctx::issue_warp()
   */
  void cycle();

  // These are some common ordering fucntions that the
  // higher order schedulers can take advantage of
  /*
   * [한국어] order_lrr() — Loose Round Robin 방식으로 warp를 result_list에 정렬.
   * last_issued_from_input 다음 위치부터 순환하며 num_warps_to_add 개만큼 추가.
   * 호출자: lrr_scheduler::order_warps(), two_level_active_scheduler::order_warps().
   */
  template <typename T>
  void order_lrr(
      typename std::vector<T> &result_list,
      const typename std::vector<T> &input_list,
      const typename std::vector<T>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add);
  /*
   * [한국어] order_rrr() — Rigid Round Robin 방식으로 warp를 result_list에 정렬.
   * LRR과 달리 항상 첫 번째 warp부터 순환 — 더 엄격한 라운드로빈.
   */
  template <typename T>
  void order_rrr(
      typename std::vector<T> &result_list,
      const typename std::vector<T> &input_list,
      const typename std::vector<T>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add);

  /*
   * [한국어] OrderingType — order_by_priority()에서 탐욕적 스케줄링 여부를 제어하는 열거형.
   */
  enum OrderingType {
    // The item that issued last is prioritized first then the sorted result
    // of the priority_function
    ORDERING_GREEDY_THEN_PRIORITY_FUNC = 0,
    /* [한국어] 직전 사이클에 issue된 warp를 먼저 배치한 후 priority_func 결과로 정렬.
     * GTO 스케줄러의 "Greedy" 부분: 한 번 선택된 warp를 계속 issue하려는 경향. */
    // No greedy scheduling based on last to issue. Only the priority function
    // determines priority
    ORDERED_PRIORITY_FUNC_ONLY,
    /* [한국어] 탐욕적 부분 없이 priority_func만으로 순서 결정.
     * OLDEST/YOUNGEST 스케줄러 등에서 사용. */
    NUM_ORDERING,  /* [한국어] OrderingType 총 수 */
  };
  /*
   * [한국어] order_by_priority() — 우선순위 함수와 OrderingType으로 warp 목록 정렬.
   * @result_list: 정렬된 warp 포인터를 담을 출력 벡터.
   * @input_list: 정렬 대상 warp 포인터 목록.
   * @last_issued_from_input: 직전 issue된 warp의 iterator (greedy 기준).
   * @num_warps_to_add: 결과에 포함할 warp 수.
   * @age_ordering: 탐욕 + 우선순위 또는 우선순위만.
   * @priority_func: 두 warp 포인터 비교 함수 (true이면 lhs가 더 높은 우선순위).
   */
  template <typename U>
  void order_by_priority(
      std::vector<U> &result_list, const typename std::vector<U> &input_list,
      const typename std::vector<U>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add, OrderingType age_ordering,
      bool (*priority_func)(U lhs, U rhs));
  /*
   * [한국어] sort_warps_by_oldest_dynamic_id() — 동적 warp ID 기준 오래된 warp 우선 비교 함수.
   * priority_func 인자로 order_by_priority()에 전달된다.
   * @lhs, @rhs: 비교할 warp 포인터.
   * @return: lhs의 dynamic_warp_id < rhs이면 true (lhs가 오래됨 = 더 높은 우선순위).
   */
  static bool sort_warps_by_oldest_dynamic_id(shd_warp_t *lhs, shd_warp_t *rhs);

  // Derived classes can override this function to populate
  // m_supervised_warps with their scheduling policies
  /*
   * [한국어] order_warps() — 순수 가상 함수. 파생 스케줄러가 구현하여 m_next_cycle_prioritized_warps 갱신.
   * 각 사이클 cycle() 시작 시 호출되며, 파생 클래스별 정렬 알고리즘이 적용된다.
   */
  virtual void order_warps() = 0;

  /* [한국어] get_schd_id() — 이 스케줄러의 SM 내 ID 반환. sub-core 모델에서 레지스터 뱅크 분리에 사용. */
  int get_schd_id() const { return m_id; }

 protected:
  /*
   * [한국어] do_on_warp_issued() — warp issue 후 스케줄러 상태 업데이트 훅.
   * @warp_id: 방금 issue된 warp ID.
   * @num_issued: 이번에 issue된 명령어 수.
   * @prioritized_iter: 발행된 warp의 우선순위 목록 iterator.
   * 기본 구현: m_last_supervised_issued 갱신. two_level_active_scheduler는 오버라이드.
   */
  virtual void do_on_warp_issued(
      unsigned warp_id, unsigned num_issued,
      const std::vector<shd_warp_t *>::const_iterator &prioritized_iter);
  /*
   * [한국어] get_sid() — 소속 SM의 shader ID 반환. 통계 인덱싱에 사용.
   * 인라인 정의는 파일 끝에 위치 (scheduler_unit::get_sid()).
   */
  inline int get_sid() const;

 protected:
  /*
   * [한국어] warp() — warp 슬롯 번호로 shd_warp_t 참조 반환.
   * 내부적으로 m_warp 벡터를 역참조한다. 스케줄러가 warp 상태에 접근할 때 사용.
   */
  shd_warp_t &warp(int i);

  // This is the prioritized warp list that is looped over each cycle to
  // determine which warp gets to issue.
  std::vector<shd_warp_t *> m_next_cycle_prioritized_warps;
  /* [한국어] 이번 사이클에 issue 검토 순서로 정렬된 warp 포인터 목록.
   * order_warps()가 이 목록을 채우고, cycle()이 순서대로 순회하며 issue 결정.
   * 설정자: order_warps() (파생 클래스 구현). 읽는 자: cycle(). */

  // The m_supervised_warps list is all the warps this scheduler is supposed to
  // arbitrate between.  This is useful in systems where there is more than
  // one warp scheduler. In a single scheduler system, this is simply all
  // the warps assigned to this core.
  std::vector<shd_warp_t *> m_supervised_warps;
  /* [한국어] 이 스케줄러가 중재를 담당하는 전체 warp 포인터 목록.
   * 여러 스케줄러가 있을 때 SM의 warp 집합이 균등 분배된다.
   * 설정자: add_supervised_warp_id()에서 SM 초기화 시 구성.
   * 읽는 자: order_warps(), order_lrr(), order_rrr(), order_by_priority(). */

  // This is the iterator pointer to the last supervised warp you issued
  std::vector<shd_warp_t *>::const_iterator m_last_supervised_issued;
  /* [한국어] 직전 사이클에 마지막으로 issue한 supervised warp의 iterator.
   * LRR에서 다음 순환 시작 위치를, GTO에서 탐욕적 우선 대상을 결정하는 데 사용.
   * 설정자: do_on_warp_issued(), done_adding_supervised_warps().
   * 읽는 자: order_lrr(), order_by_priority(). */

  shader_core_stats *m_stats;
  /* [한국어] SM 통계 집계 객체 포인터 — issue 통계(event_warp_issued 등) 업데이트에 사용. */
  shader_core_ctx *m_shader;
  /* [한국어] 소속 SM 포인터 — issue_warp() 호출, get_sid() 등 SM 상태 접근에 사용. */
  // these things should become accessors: but would need a bigger rearchitect
  // of how shader_core_ctx interacts with its parts.
  Scoreboard *m_scoreboard;
  /* [한국어] 레지스터 의존성 추적 스코어보드 포인터 — issue 가능 여부 판단 시 check()로 충돌 검사. */
  simt_stack **m_simt_stack;
  /* [한국어] SIMT 스택 배열 포인터 — warp별 divergence/reconvergence 상태 확인 시 참조. */
  // warp_inst_t** m_pipeline_reg;
  std::vector<shd_warp_t *> *m_warp;
  /* [한국어] SM 전체 warp 배열 포인터 — warp() 헬퍼 함수로 슬롯 접근 시 사용. */
  register_set *m_sp_out;          /* [한국어] SP(FP32) 유닛 입력 파이프라인 레지스터 집합 포인터 */
  register_set *m_dp_out;          /* [한국어] DP(FP64) 유닛 입력 파이프라인 레지스터 집합 포인터 */
  register_set *m_sfu_out;         /* [한국어] SFU 유닛 입력 파이프라인 레지스터 집합 포인터 */
  register_set *m_int_out;         /* [한국어] INT 유닛 입력 파이프라인 레지스터 집합 포인터 */
  register_set *m_tensor_core_out; /* [한국어] 텐서 코어 입력 파이프라인 레지스터 집합 포인터 */
  register_set *m_mem_out;         /* [한국어] 메모리(ldst) 유닛 입력 파이프라인 레지스터 집합 포인터 */
  std::vector<register_set *> &m_spec_cores_out;
  /* [한국어] 특수 유닛들의 입력 파이프라인 레지스터 집합 벡터 참조 — specialized_unit 지원 */
  unsigned m_num_issued_last_cycle;
  /* [한국어] 직전 사이클에 이 스케줄러가 issue한 명령어 수 — 이중 발행(dual issue) 통계 집계용. */
  unsigned m_current_turn_warp;
  /* [한국어] 현재 순환 중인 warp 인덱스 (일부 스케줄러 구현에서 상태 유지용). */

  int m_id;
  /* [한국어] 이 스케줄러의 SM 내 번호 (0 ~ gpgpu_num_sched_per_core-1).
   * sub-core 모델에서 레지스터 뱅크를 스케줄러별로 분리할 때 banks_per_sched 계산에 사용. */
};

/*
 * [한국어] lrr_scheduler — Loose Round Robin (느슨한 라운드 로빈) 워프 스케줄러
 *
 * === 클래스 역할 ===
 * scheduler_unit의 구체적 구현으로, 감독 워프 목록을 느슨한 라운드 로빈 순서로
 * 정렬한다. "느슨한(Loose)"이란 모든 워프를 매 사이클 엄격하게 순환하는 대신,
 * 마지막으로 이슈된 워프 다음 위치부터 이슈 가능한 워프를 탐색하는 방식을 뜻한다.
 * 이로써 특정 워프가 스톨되어도 다른 워프가 진행될 수 있도록 허용하면서,
 * 장기 기아(starvation)를 방지하는 공정성을 제공한다.
 *
 * === 아키텍처에서의 위치 ===
 * 타이밍 모델(gpgpu-sim) 계층에서 SM의 워프 스케줄러 서브시스템을 구성한다.
 * shader_core_ctx가 create_schedulers()에서 config 문자열("lrr")을 파싱해 생성한다.
 * order_warps()가 매 사이클 scheduler_unit::cycle()에서 호출되어 워프 우선순위를 갱신한다.
 *
 * === 타 모듈과의 연결 ===
 * - 부모: scheduler_unit (워프 이슈 로직 전체를 제공)
 * - 사용: m_supervised_warps, m_last_supervised_issued (부모 protected 멤버)
 * - 호출 체인: shader_core_ctx::issue() → lrr_scheduler::order_warps() → scheduler_unit::cycle()
 */
class lrr_scheduler : public scheduler_unit {
 public:
  /*
   * [한국어] lrr_scheduler 생성자 — 부모 scheduler_unit 생성자를 위임 호출한다.
   *
   * @param stats: SM 통계 객체 (AccelWattch 전력 모델 카운터 포함)
   * @param shader: 이 스케줄러가 소속된 SM (shader_core_ctx)
   * @param scoreboard: RAW 해저드 감지 스코어보드
   * @param simt: SIMT 스택 배열 (warp 수만큼)
   * @param warp: 전체 warp 상태 포인터 벡터
   * @param sp_out, dp_out, sfu_out, int_out, tensor_core_out: 각 실행 유닛 입력 파이프라인 레지스터 집합
   * @param spec_cores_out: specialized unit 파이프라인 레지스터 목록
   * @param mem_out: 로드/스토어 유닛 입력 파이프라인 레지스터 집합
   * @param id: 이 스케줄러의 인스턴스 ID (SM 내 여러 스케줄러 구분)
   *
   * 부모 scheduler_unit 생성자에 모든 인자를 그대로 전달하며, 추가 초기화는 없다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_schedulers() → [lrr_scheduler 생성자] → scheduler_unit(...)
   */
  lrr_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id) {}
  virtual ~lrr_scheduler() {}
  /*
   * [한국어] order_warps — 느슨한 라운드 로빈 순서로 감독 워프를 정렬한다.
   *
   * m_last_supervised_issued가 가리키는 워프 다음 위치부터 시작하여,
   * m_next_cycle_prioritized_warps 리스트를 재구성한다.
   * 이전 사이클에 이슈된 워프의 바로 다음 워프가 우선권을 얻어,
   * 이슈된 워프가 계속 선점하는 것을 방지한다.
   *
   * 호출 체인:
   *   scheduler_unit::cycle() → [order_warps()] → m_next_cycle_prioritized_warps 갱신
   */
  virtual void order_warps();
  /*
   * [한국어] done_adding_supervised_warps — 감독 워프 추가 완료 후 마지막 이슈 포인터를 초기화한다.
   *
   * lrr 정책에서는 마지막 이슈 위치를 end()로 설정해, 다음 사이클에 처음부터 탐색을 시작한다.
   * (gto/oldest와 달리 begin()이 아닌 end()를 사용하는 것이 LRR의 특징이다.)
   *
   * 호출 체인:
   *   shader_core_ctx::create_schedulers() 이후 초기화 단계에서 호출
   */
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.end(); // [한국어] 마지막 이슈 포인터를 끝으로 설정 — 다음 탐색은 전체 목록 처음부터 시작
  }
};

/*
 * [한국어] rrr_scheduler — Rigid Round Robin (엄격한 라운드 로빈) 워프 스케줄러
 *
 * === 클래스 역할 ===
 * scheduler_unit의 구체적 구현으로, 매 사이클 감독 워프 목록을 엄격한 라운드 로빈 순서로
 * 정렬한다. LRR과 달리 "엄격한(Rigid)" 방식은 전체 워프를 고정된 순서로 순환하며,
 * 워프의 이슈 가능 여부와 무관하게 순서를 유지한다.
 * 공정한 자원 배분이 필요하지만, 스톨된 워프로 인한 지연이 발생할 수 있다.
 *
 * === 아키텍처에서의 위치 ===
 * shader_core_ctx::create_schedulers()에서 config 문자열("rrr")로 생성된다.
 * order_warps()가 매 사이클 호출되며, 워프 우선순위 순서를 갱신한다.
 *
 * === 타 모듈과의 연결 ===
 * - 부모: scheduler_unit
 * - 호출 체인: shader_core_ctx::issue() → rrr_scheduler::order_warps() → scheduler_unit::cycle()
 */
class rrr_scheduler : public scheduler_unit {
 public:
  /*
   * [한국어] rrr_scheduler 생성자 — 부모 scheduler_unit 생성자를 위임 호출한다.
   * 모든 인자는 lrr_scheduler와 동일하며, 추가 초기화가 없다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_schedulers() → [rrr_scheduler 생성자] → scheduler_unit(...)
   */
  rrr_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id) {}
  virtual ~rrr_scheduler() {}
  /*
   * [한국어] order_warps — 엄격한 라운드 로빈 순서로 감독 워프를 정렬한다.
   *
   * 전체 감독 워프 목록을 고정된 순서(warp ID 기준)로 순환하여
   * m_next_cycle_prioritized_warps를 재구성한다.
   *
   * 호출 체인:
   *   scheduler_unit::cycle() → [order_warps()] → m_next_cycle_prioritized_warps 갱신
   */
  virtual void order_warps();
  /*
   * [한국어] done_adding_supervised_warps — 감독 워프 추가 완료 후 마지막 이슈 포인터를 끝으로 초기화한다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_schedulers() 초기화 단계에서 호출
   */
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.end(); // [한국어] end()로 설정 — LRR/RRR에서는 전체 목록 처음부터 탐색 시작
  }
};

/*
 * [한국어] gto_scheduler — Greedy Then Oldest (탐욕적 이후 최고령) 워프 스케줄러
 *
 * === 클래스 역할 ===
 * 가장 최근에 이슈된 워프를 우선 실행하고, 해당 워프가 스톨되면 가장 오래된 워프로
 * 전환하는 2단계 정책을 구현한다. 캐시 지역성을 극대화하기 위해 하나의 워프를
 * 가능한 한 계속 실행하다가(Greedy), 스톨 발생 시 가장 오래된 워프(Oldest)로
 * 전환해 TLP(Thread-Level Parallelism)를 유지한다.
 * NVIDIA Fermi 아키텍처에서 사용된 스케줄링 정책과 유사하다.
 *
 * === 아키텍처에서의 위치 ===
 * shader_core_ctx::create_schedulers()에서 config 문자열("gto")으로 생성된다.
 * done_adding_supervised_warps()에서 begin()을 사용해 가장 오래된 워프부터 시작하도록 초기화한다.
 *
 * === 타 모듈과의 연결 ===
 * - 부모: scheduler_unit
 * - 호출 체인: shader_core_ctx::issue() → gto_scheduler::order_warps() → scheduler_unit::cycle()
 */
class gto_scheduler : public scheduler_unit {
 public:
  /*
   * [한국어] gto_scheduler 생성자 — 부모 scheduler_unit 생성자를 위임 호출한다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_schedulers() → [gto_scheduler 생성자] → scheduler_unit(...)
   */
  gto_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id) {}
  virtual ~gto_scheduler() {}
  /*
   * [한국어] order_warps — GTO 정책으로 감독 워프를 정렬한다.
   *
   * 현재 이슈 중인 워프(greedy)를 목록 앞에 배치하고,
   * 나머지 워프들은 동적 워프 ID(가장 오래된 순)로 정렬한다.
   * 현재 워프가 스톨되면 자동으로 다음으로 오래된 워프를 선택한다.
   *
   * 호출 체인:
   *   scheduler_unit::cycle() → [order_warps()] → m_next_cycle_prioritized_warps 갱신
   */
  virtual void order_warps();
  /*
   * [한국어] done_adding_supervised_warps — 마지막 이슈 포인터를 목록 시작으로 초기화한다.
   *
   * begin()으로 설정해 가장 오래된 워프(목록 앞쪽)부터 탐색을 시작하도록 준비한다.
   * LRR/RRR의 end()와 달리 GTO는 begin()을 사용한다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_schedulers() 초기화 단계에서 호출
   */
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin(); // [한국어] begin()으로 설정 — 가장 오래된 워프부터 탐색 시작 (GTO/oldest 정책)
  }
};

/*
 * [한국어] oldest_scheduler — Oldest-First (최고령 우선) 워프 스케줄러
 *
 * === 클래스 역할 ===
 * 동적 워프 ID가 가장 작은(가장 오래 전에 생성된) 워프를 항상 최우선으로 이슈한다.
 * 이 정책은 메모리 지연 숨기기(latency hiding)를 극대화하는 대신,
 * 최근에 생성된 워프는 오랫동안 이슈되지 않을 수 있다(잠재적 기아 문제).
 * GTO와 비슷하지만, GTO처럼 현재 이슈 워프를 탐욕적으로 유지하지 않고
 * 매 사이클 전체 목록을 오래된 순으로 재정렬한다.
 *
 * === 아키텍처에서의 위치 ===
 * shader_core_ctx::create_schedulers()에서 config 문자열("oldest")으로 생성된다.
 * order_warps()가 매 사이클 동적 워프 ID 기준 오름차순 정렬을 수행한다.
 *
 * === 타 모듈과의 연결 ===
 * - 부모: scheduler_unit
 * - 호출 체인: shader_core_ctx::issue() → oldest_scheduler::order_warps() → scheduler_unit::cycle()
 */
class oldest_scheduler : public scheduler_unit {
 public:
  /*
   * [한국어] oldest_scheduler 생성자 — 부모 scheduler_unit 생성자를 위임 호출한다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_schedulers() → [oldest_scheduler 생성자] → scheduler_unit(...)
   */
  oldest_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                   Scoreboard *scoreboard, simt_stack **simt,
                   std::vector<shd_warp_t *> *warp, register_set *sp_out,
                   register_set *dp_out, register_set *sfu_out,
                   register_set *int_out, register_set *tensor_core_out,
                   std::vector<register_set *> &spec_cores_out,
                   register_set *mem_out, int id)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id) {}
  virtual ~oldest_scheduler() {}
  /*
   * [한국어] order_warps — 동적 워프 ID 기준 오름차순으로 감독 워프를 정렬한다.
   *
   * 매 사이클 전체 감독 워프를 동적 warp ID 기준 정렬해 가장 오래된 워프가 항상 앞에 오도록 한다.
   * 스코어보드 해저드나 메모리 스톨로 인해 이슈 불가한 경우에는 다음으로 오래된 워프를 시도한다.
   *
   * 호출 체인:
   *   scheduler_unit::cycle() → [order_warps()] → m_next_cycle_prioritized_warps 갱신
   */
  virtual void order_warps();
  /*
   * [한국어] done_adding_supervised_warps — 마지막 이슈 포인터를 목록 시작으로 초기화한다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_schedulers() 초기화 단계에서 호출
   */
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin(); // [한국어] begin()으로 설정 — 가장 오래된 워프부터 탐색 시작
  }
};

/*
 * [한국어] two_level_active_scheduler — 2단계 활성 풀 워프 스케줄러
 *
 * === 클래스 역할 ===
 * 워프를 '활성 풀(active pool)'과 '대기 풀(pending pool)'의 두 단계로 나누어 관리한다.
 * 활성 풀은 최대 m_max_active_warps개의 워프를 보유하며, 이 풀에서 우선적으로 이슈한다.
 * 활성 풀 워프가 이슈되면 대기 풀에서 새 워프를 활성 풀로 승격시킨다.
 * 내부 우선순위(inner_level_prioritization)는 활성 풀 내에서,
 * 외부 우선순위(outer_level_prioritization)는 대기 풀 내에서 정렬 방식을 결정한다.
 * gpgpusim.config의 형식: "two_level_active:<max_active>:<inner_prio>:<outer_prio>"
 *
 * === 아키텍처에서의 위치 ===
 * shader_core_ctx::create_schedulers()에서 config 문자열 파싱으로 생성된다.
 * Intel/AMD GPU의 '웨이브프론트 그룹' 기반 스케줄링 개념과 유사하다.
 *
 * === 타 모듈과의 연결 ===
 * - 부모: scheduler_unit
 * - 데이터: m_pending_warps(대기 풀), m_next_cycle_prioritized_warps(활성 풀 — 부모 멤버)
 * - 호출 체인: shader_core_ctx::issue() → two_level_active_scheduler::order_warps()
 *             → do_on_warp_issued() (워프 이슈 시 풀 갱신)
 */
class two_level_active_scheduler : public scheduler_unit {
 public:
  /*
   * [한국어] two_level_active_scheduler 생성자 — 설정 문자열을 파싱해 2단계 정책 파라미터를 초기화한다.
   *
   * @param config_str: "two_level_active:<max_active>:<inner_prio>:<outer_prio>" 형식의 문자열
   *        (gpgpusim.config의 -gpgpu_scheduler 옵션 값)
   *
   * sscanf로 최대 활성 워프 수와 내/외부 우선순위 타입을 파싱하여 멤버에 저장한다.
   * 파싱 실패 시 assert(3 == ret)으로 즉시 abort한다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_schedulers() → [two_level_active_scheduler 생성자]
   */
  two_level_active_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                             Scoreboard *scoreboard, simt_stack **simt,
                             std::vector<shd_warp_t *> *warp,
                             register_set *sp_out, register_set *dp_out,
                             register_set *sfu_out, register_set *int_out,
                             register_set *tensor_core_out,
                             std::vector<register_set *> &spec_cores_out,
                             register_set *mem_out, int id, char *config_str)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id),
        m_pending_warps() { // [한국어] 대기 풀(pending pool) 초기화 — 처음에는 비어 있음
    unsigned inner_level_readin; // [한국어] 내부 우선순위 타입을 정수로 읽어들이는 임시 변수
    unsigned outer_level_readin; // [한국어] 외부 우선순위 타입을 정수로 읽어들이는 임시 변수
    int ret =
        sscanf(config_str, "two_level_active:%d:%d:%d", &m_max_active_warps,
               &inner_level_readin, &outer_level_readin); // [한국어] 설정 문자열에서 3개 파라미터 파싱
    assert(3 == ret); // [한국어] 파싱 실패 시 abort — 설정 문자열 형식 오류 방지
    m_inner_level_prioritization =
        (scheduler_prioritization_type)inner_level_readin; // [한국어] 정수 → 열거형 변환 (활성 풀 내 우선순위)
    m_outer_level_prioritization =
        (scheduler_prioritization_type)outer_level_readin; // [한국어] 정수 → 열거형 변환 (대기 풀 내 우선순위)
  }
  virtual ~two_level_active_scheduler() {}
  /*
   * [한국어] order_warps — 2단계 정책으로 활성 풀과 대기 풀을 정렬한다.
   *
   * 1단계: 활성 풀(m_next_cycle_prioritized_warps)을 inner_level_prioritization으로 정렬
   * 2단계: 대기 풀(m_pending_warps)을 outer_level_prioritization으로 정렬
   * 이슈 후 do_on_warp_issued()에서 대기 풀 → 활성 풀 승격이 이루어진다.
   *
   * 호출 체인:
   *   scheduler_unit::cycle() → [order_warps()] → m_next_cycle_prioritized_warps 갱신
   */
  virtual void order_warps();
  /*
   * [한국어] add_supervised_warp_id — 활성 풀 또는 대기 풀에 워프를 추가한다.
   *
   * @param i: 추가할 워프의 하드웨어 슬롯 ID
   *
   * 활성 풀이 m_max_active_warps 미만이면 활성 풀에 바로 추가하고,
   * 가득 찬 경우 대기 풀에 추가한다. 부모의 add_supervised_warp_id()를 오버라이드한다.
   *
   * 호출 체인:
   *   scheduler_unit::add_supervised_warp_id() 오버라이드 → 풀 선택 후 push_back
   */
  void add_supervised_warp_id(int i) {
    if (m_next_cycle_prioritized_warps.size() < m_max_active_warps) { // [한국어] 활성 풀에 공간이 있으면 활성 풀에 추가
      m_next_cycle_prioritized_warps.push_back(&warp(i)); // [한국어] 워프 참조를 활성 풀(prioritized 목록)에 삽입
    } else {
      m_pending_warps.push_back(&warp(i)); // [한국어] 활성 풀이 가득 차면 대기 풀에 추가
    }
  }
  /*
   * [한국어] done_adding_supervised_warps — 마지막 이슈 포인터를 목록 시작으로 초기화한다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_schedulers() 초기화 단계에서 호출
   */
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin(); // [한국어] begin()으로 초기화 — 가장 오래된 워프부터 탐색 시작
  }

 protected:
  /*
   * [한국어] do_on_warp_issued — 워프 이슈 완료 후 풀 갱신을 수행한다.
   *
   * @param warp_id: 방금 이슈된 워프의 하드웨어 슬롯 ID
   * @param num_issued: 이 사이클에 이슈된 명령어 수
   * @param prioritized_iter: 이슈된 워프를 가리키는 prioritized 목록 이터레이터
   *
   * 이슈된 워프가 활성 풀에서 나간 후, 대기 풀에서 워프를 활성 풀로 승격시킨다.
   * 부모 scheduler_unit의 do_on_warp_issued()를 오버라이드한다.
   *
   * 호출 체인:
   *   scheduler_unit::cycle() → do_on_warp_issued() → m_pending_warps에서 m_next_cycle_prioritized_warps로 이동
   */
  virtual void do_on_warp_issued(
      unsigned warp_id, unsigned num_issued,
      const std::vector<shd_warp_t *>::const_iterator &prioritized_iter);

 private:
  std::deque<shd_warp_t *> m_pending_warps;
  /* [한국어] 활성 풀이 가득 찰 때 대기하는 워프들의 덱(deque).
   * 설정자: add_supervised_warp_id()에서 활성 풀 초과 시 push_back.
   * 읽는 자: do_on_warp_issued()에서 활성 풀에 빈 슬롯 생기면 pop_front하여 승격.
   * 값 범위: 0 ~ (전체 워프 수 - m_max_active_warps) 개 워프 포인터.
   * 동기화: 단일 스케줄러 스레드에서만 접근하므로 별도 락 불필요. */

  scheduler_prioritization_type m_inner_level_prioritization;
  /* [한국어] 활성 풀(active pool) 내 워프들의 정렬 기준.
   * 설정자: 생성자에서 config_str 파싱으로 결정.
   * 읽는 자: order_warps()에서 활성 풀 정렬 시 사용.
   * 값 범위: SCHEDULER_PRIORITIZATION_* 열거형 값 (LRR, GTO, OLDEST 등).
   * 동기화: 읽기 전용 (생성 후 변경 없음). */

  scheduler_prioritization_type m_outer_level_prioritization;
  /* [한국어] 대기 풀(pending pool) 내 워프들의 정렬 기준.
   * 설정자: 생성자에서 config_str 파싱으로 결정.
   * 읽는 자: order_warps()에서 대기 풀 정렬 시 사용.
   * 값 범위: SCHEDULER_PRIORITIZATION_* 열거형 값.
   * 동기화: 읽기 전용 (생성 후 변경 없음). */

  unsigned m_max_active_warps;
  /* [한국어] 활성 풀이 보유할 수 있는 최대 워프 수.
   * 설정자: 생성자에서 config_str의 첫 번째 숫자로 결정 ("two_level_active:N:...").
   * 읽는 자: add_supervised_warp_id()에서 풀 선택 판단에 사용.
   * 값 범위: 1 ~ 전체 워프 수. 일반적으로 4~8로 설정.
   * 동기화: 읽기 전용 (생성 후 변경 없음). */
};

/*
 * [한국어] swl_scheduler — Static Warp Limiting (정적 워프 제한) 스케줄러
 *
 * === 클래스 역할 ===
 * SM에서 동시에 이슈할 수 있는 활성 워프의 수를 정적으로 제한하는 스케줄러이다.
 * m_num_warps_to_limit 값으로 동시 활성 워프 수를 제한해 레지스터 파일 충돌,
 * 공유 메모리 뱅크 충돌 등의 자원 경쟁을 줄이는 대신 TLP를 제한한다.
 * gpgpusim.config 형식: "swl:<prioritization>:<num_warps_to_limit>"
 *
 * === 아키텍처에서의 위치 ===
 * shader_core_ctx::create_schedulers()에서 config 문자열 파싱으로 생성된다.
 * two_level_active_scheduler의 단순화 변형으로 볼 수 있다.
 *
 * === 타 모듈과의 연결 ===
 * - 부모: scheduler_unit
 * - 설정: gpgpusim.config의 -gpgpu_scheduler 옵션
 * - 호출 체인: shader_core_ctx::issue() → swl_scheduler::order_warps() → scheduler_unit::cycle()
 */
// Static Warp Limiting Scheduler
class swl_scheduler : public scheduler_unit {
 public:
  /*
   * [한국어] swl_scheduler 생성자 — 설정 문자열을 파싱해 제한 파라미터를 초기화한다.
   *
   * @param config_string: "swl:<prioritization>:<num_warps>" 형식의 문자열
   *
   * sscanf로 우선순위 타입과 제한 워프 수를 파싱하여 멤버에 저장한다.
   * 구현은 shader.cc에 있으며, 파싱 실패 시 abort한다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_schedulers() → [swl_scheduler 생성자]
   */
  swl_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id, char *config_string);
  virtual ~swl_scheduler() {}
  /*
   * [한국어] order_warps — m_num_warps_to_limit 개까지만 워프를 활성 목록에 포함시킨다.
   *
   * 전체 감독 워프 중 m_prioritization 기준으로 정렬한 뒤,
   * 앞쪽 m_num_warps_to_limit개만 m_next_cycle_prioritized_warps에 유지한다.
   *
   * 호출 체인:
   *   scheduler_unit::cycle() → [order_warps()] → m_next_cycle_prioritized_warps 갱신
   */
  virtual void order_warps();
  /*
   * [한국어] done_adding_supervised_warps — 마지막 이슈 포인터를 목록 시작으로 초기화한다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_schedulers() 초기화 단계에서 호출
   */
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin(); // [한국어] begin()으로 초기화 — 가장 오래된 워프부터 탐색
  }

 protected:
  scheduler_prioritization_type m_prioritization;
  /* [한국어] 제한된 워프 집합 내에서의 우선순위 정렬 방식.
   * 설정자: 생성자에서 config_string 파싱으로 결정.
   * 읽는 자: order_warps()에서 정렬 기준으로 사용.
   * 값 범위: SCHEDULER_PRIORITIZATION_* 열거형.
   * 동기화: 읽기 전용 (생성 후 변경 없음). */

  unsigned m_num_warps_to_limit;
  /* [한국어] 동시에 이슈 가능한 최대 워프 수 (정적 제한값).
   * 설정자: 생성자에서 config_string의 두 번째 숫자로 결정.
   * 읽는 자: order_warps()에서 활성 워프 수 제한에 사용.
   * 값 범위: 1 ~ 전체 워프 수. 낮을수록 자원 경쟁 감소, TLP 감소.
   * 동기화: 읽기 전용 (생성 후 변경 없음). */
};

/*
 * [한국어] opndcoll_rfu_t — 오퍼랜드 컬렉터 기반 레지스터 파일 유닛 (Operand Collector Register File Unit)
 *
 * === 클래스 역할 ===
 * GPU SM의 레지스터 파일 뱅크 충돌(bank conflict)을 처리하는 핵심 구조체이다.
 * NVIDIA GPU의 레지스터 파일은 여러 뱅크로 나뉘어져 있고, 동시에 같은 뱅크에 접근하는
 * 오퍼랜드들은 직렬화(stall)되어야 한다. opndcoll_rfu_t는 collector unit(CU)이라는
 * 버퍼에 명령어를 보관하면서, 각 소스 오퍼랜드가 레지스터 파일 뱅크에서 성공적으로 읽힐
 * 때까지 기다린 후 실행 유닛으로 디스패치한다.
 * 이 구조는 Fung et al. (MICRO 2007) "Dynamic Warp Formation and Scheduling"에서 제안된
 * 오퍼랜드 컬렉터 모델을 기반으로 한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 파이프라인 단계: ID(명령어 디코드) → [OC: 오퍼랜드 컬렉션] → EX(실행)
 * 명령어가 ID 단계를 통과한 후 파이프라인 레지스터(ID_OC_*)에 놓이면,
 * opndcoll_rfu_t::step()이 매 사이클 호출되어:
 *   1. allocate_cu(): ID_OC 레지스터에서 빈 CU로 명령어를 할당
 *   2. allocate_reads(): arbiter를 통해 레지스터 뱅크 읽기 요청 중재
 *   3. dispatch_ready_cu(): 모든 오퍼랜드 수집 완료된 CU를 OC_EX로 디스패치
 *   4. process_banks(): 뱅크 할당 상태 초기화
 *
 * === 타 모듈과의 연결 ===
 * - 사용자: shader_core_ctx (step() 매 사이클 호출, writeback() 라이트백 시 호출)
 * - 사용: ldst_unit (오퍼랜드 컬렉터를 통해 레지스터 파일 접근)
 * - 공유: register_set (ID_OC_*, OC_EX_* 파이프라인 레지스터)
 * - 공유: Scoreboard (오퍼랜드 준비 여부 확인)
 *
 * === 주요 내부 클래스 요약 ===
 * - op_t: 단일 레지스터 오퍼랜드 읽기/쓰기 요청 (레지스터 번호, 뱅크 번호 포함)
 * - allocation_t: 레지스터 뱅크의 현재 할당 상태 (FREE / READ / WRITE)
 * - arbiter_t: 라운드 로빈 방식으로 뱅크 접근 요청을 중재하는 아비터
 * - collector_unit_t: 명령어를 보관하며 모든 오퍼랜드 수집을 기다리는 버퍼 유닛
 * - dispatch_unit_t: 준비된 CU를 실행 유닛으로 디스패치하는 유닛
 * - input_port_t: 입력 포트와 출력 포트, CU 집합의 연결을 묶은 구조체
 */
class opndcoll_rfu_t {  // operand collector based register file unit
 public:
  /*
   * [한국어] opndcoll_rfu_t 기본 생성자 — 멤버를 기본값(0/NULL/false)으로 초기화한다.
   *
   * init()이 호출되기 전까지는 m_initialized = false로 미초기화 상태이며,
   * step() 등을 호출해서는 안 된다.
   *
   * 호출 체인:
   *   shader_core_ctx 생성자에서 m_operand_collector 멤버 기본 생성 → [opndcoll_rfu_t()]
   */
  // constructors
  opndcoll_rfu_t() {
    m_num_banks = 0;   // [한국어] 레지스터 파일 뱅크 수 — init() 호출 전까지 0
    m_shader = NULL;   // [한국어] 소속 SM 포인터 — init() 호출 전까지 NULL
    m_initialized = false; // [한국어] 초기화 완료 플래그 — init() 호출 후 true로 변경
  }
  /*
   * [한국어] add_cu_set — 특정 CU 집합(set)을 추가하고 콜렉터 유닛과 디스패치 유닛을 생성한다.
   *
   * @param cu_set: CU 집합의 ID (실행 유닛 종류에 대응, e.g., SP=0, DP=1, SFU=2 ...)
   * @param num_cu: 이 집합에 속하는 콜렉터 유닛의 수
   * @param num_dispatch: 이 집합을 위한 디스패치 유닛의 수
   *
   * m_cus 맵에 collector_unit_t 벡터를 추가하고, 각 dispatch_unit_t도 생성한다.
   * 호출 순서는 add_cu_set() → add_port() → init() 이어야 한다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_exec_pipeline() → [add_cu_set()] → m_cus, m_dispatch_units 갱신
   */
  void add_cu_set(unsigned cu_set, unsigned num_cu, unsigned num_dispatch);
  typedef std::vector<register_set *> port_vector_t; // [한국어] 파이프라인 레지스터 집합의 포인터 벡터 타입 별칭
  typedef std::vector<unsigned int> uint_vector_t;   // [한국어] CU 집합 ID의 unsigned int 벡터 타입 별칭
  /*
   * [한국어] add_port — 입력/출력 포트와 연결된 CU 집합을 등록한다.
   *
   * @param input: 이 포트로 들어오는 ID_OC_* 파이프라인 레지스터 집합들
   * @param output: 이 포트에서 나가는 OC_EX_* 파이프라인 레지스터 집합들
   * @param cu_sets: 이 포트가 사용하는 CU 집합 ID들
   *
   * input_port_t를 생성해 m_in_ports에 추가한다.
   * 각 실행 유닛 유형(SP, DP, SFU 등)마다 별도 포트가 추가된다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_exec_pipeline() → [add_port()] → m_in_ports 갱신
   */
  void add_port(port_vector_t &input, port_vector_t &ouput,
                uint_vector_t cu_sets);
  /*
   * [한국어] init — 레지스터 파일 뱅크 수와 SM 포인터로 전체 오퍼랜드 컬렉터를 초기화한다.
   *
   * @param num_banks: 레지스터 파일 뱅크 수 (gpgpusim.config의 -gpgpu_num_reg_banks)
   * @param shader: 소속 SM (shader_core_ctx*)
   *
   * arbiter_t를 초기화하고, 모든 collector_unit_t를 초기화한다.
   * add_cu_set()과 add_port() 호출 이후에 반드시 호출해야 한다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_exec_pipeline() → add_cu_set() → add_port() → [init()]
   */
  void init(unsigned num_banks, shader_core_ctx *shader);

  /*
   * [한국어] writeback — 실행 완료된 명령어의 목적 레지스터에 쓰기 할당(write-allocate)을 수행한다.
   *
   * @param warp: 라이트백 단계(EX_WB)를 통과한 warp_inst_t
   * @return: 쓰기 뱅크 충돌이 없으면 true, 충돌로 스톨되면 false
   *
   * 목적 레지스터의 뱅크를 WRITE_ALLOC 상태로 잠그고, arbiter에 쓰기 요청을 등록한다.
   * 충돌 발생 시 해당 사이클에 라이트백을 수행하지 않고 스톨한다.
   *
   * 호출 체인:
   *   shader_core_ctx::writeback() → [writeback(warp)] → arbiter_t::allocate_bank_for_write()
   */
  // modifiers
  bool writeback(warp_inst_t &warp);

  /*
   * [한국어] step — 오퍼랜드 컬렉터의 1 사이클 동작을 수행한다.
   *
   * 매 사이클 shader_core_ctx::read_operands()에서 호출된다.
   * 실행 순서:
   *   1. dispatch_ready_cu(): 모든 오퍼랜드가 준비된 CU를 OC_EX 레지스터로 디스패치
   *   2. allocate_reads(): 아직 수집되지 않은 오퍼랜드들의 뱅크 읽기 요청을 중재 및 실행
   *   3. allocate_cu(p): 각 입력 포트의 ID_OC 레지스터에서 대기 중인 명령어를 빈 CU에 할당
   *   4. process_banks(): 아비터의 뱅크 할당 상태를 다음 사이클을 위해 초기화
   *
   * 호출 체인:
   *   shader_core_ctx::read_operands() → [step()] → dispatch_ready_cu() / allocate_reads() / allocate_cu() / process_banks()
   */
  void step() {
    dispatch_ready_cu();                                            // [한국어] 1단계: 준비된 CU를 실행 유닛 입력 파이프라인으로 디스패치
    allocate_reads();                                              // [한국어] 2단계: 아비터를 통해 레지스터 뱅크 읽기 요청 처리
    for (unsigned p = 0; p < m_in_ports.size(); p++) allocate_cu(p); // [한국어] 3단계: 각 입력 포트에서 새 명령어를 CU에 할당
    process_banks();                                               // [한국어] 4단계: 아비터 뱅크 할당 상태를 다음 사이클을 위해 리셋
  }

  /*
   * [한국어] dump — 현재 오퍼랜드 컬렉터 상태를 fp에 출력한다 (디버깅용).
   *
   * @param fp: 출력 파일 포인터
   *
   * 모든 CU의 상태와 arbiter 상태를 텍스트로 출력한다.
   *
   * 호출 체인:
   *   shader_core_ctx::display_pipeline() → [dump(fp)]
   */
  void dump(FILE *fp) const {
    fprintf(fp, "\n");
    fprintf(fp, "Operand Collector State:\n"); // [한국어] 섹션 헤더 출력
    for (unsigned n = 0; n < m_cu.size(); n++) { // [한국어] 모든 CU 순회
      fprintf(fp, "   CU-%2u: ", n);             // [한국어] CU 인덱스 출력
      m_cu[n]->dump(fp, m_shader);               // [한국어] 개별 CU 상태 출력
    }
    m_arbiter.dump(fp); // [한국어] 아비터 상태 출력 (뱅크별 요청 큐 및 할당 상태)
  }

  /*
   * [한국어] shader_core — 소속 SM 포인터를 반환한다.
   *
   * @return: 이 오퍼랜드 컬렉터가 속한 shader_core_ctx 포인터
   *
   * 호출 체인:
   *   collector_unit_t::dump() 내부에서 SM 정보 접근 시 사용
   */
  shader_core_ctx *shader_core() { return m_shader; }

 private:
  /*
   * [한국어] process_banks — 아비터의 뱅크 할당 상태를 리셋하여 다음 사이클을 준비한다.
   *
   * step()의 마지막 단계로, 이번 사이클에 설정된 READ/WRITE 할당을 NO_ALLOC으로 초기화한다.
   * 뱅크 할당은 단일 사이클 내에서만 유효하다.
   *
   * 호출 체인:
   *   step() → [process_banks()] → m_arbiter.reset_alloction()
   */
  void process_banks() { m_arbiter.reset_alloction(); } // [한국어] 아비터 뱅크 할당 전체 리셋 — 매 사이클 마지막에 호출

  /*
   * [한국어] dispatch_ready_cu — 모든 소스 오퍼랜드 수집이 완료된 CU를 실행 유닛으로 디스패치한다.
   *
   * 각 dispatch_unit_t를 순회하며 ready() 상태의 CU를 찾아 해당 OC_EX 파이프라인 레지스터에
   * 명령어를 이동시키고 CU를 해제(free)한다.
   *
   * 호출 체인:
   *   step() → [dispatch_ready_cu()] → collector_unit_t::dispatch() → OC_EX_* 레지스터에 명령어 삽입
   */
  void dispatch_ready_cu();
  /*
   * [한국어] allocate_cu — 특정 입력 포트의 ID_OC 레지스터에서 명령어를 빈 CU에 할당한다.
   *
   * @param port: 처리할 입력 포트 인덱스 (m_in_ports의 인덱스)
   *
   * ID_OC_* 파이프라인 레지스터에 대기 중인 명령어를 해당 CU 집합에서 빈 CU를 찾아 할당한다.
   * CU 할당 성공 시 ID_OC 레지스터에서 명령어를 꺼내 CU에 복사하고 소스 오퍼랜드 수집을 시작한다.
   *
   * 호출 체인:
   *   step() → [allocate_cu(port)] → collector_unit_t::allocate() → arbiter_t::add_read_requests()
   */
  void allocate_cu(unsigned port);
  /*
   * [한국어] allocate_reads — arbiter를 통해 레지스터 뱅크 읽기 요청을 중재하고 오퍼랜드를 수집한다.
   *
   * 각 CU에 등록된 미수집 오퍼랜드의 뱅크 읽기 요청을 arbiter에 전달하고,
   * arbiter::allocate_reads()를 호출해 충돌 없는 오퍼랜드들을 이번 사이클에 수집한다.
   * 수집 완료된 오퍼랜드는 m_not_ready 비트맵에서 클리어된다.
   *
   * 호출 체인:
   *   step() → [allocate_reads()] → arbiter_t::add_read_requests() → arbiter_t::allocate_reads()
   *            → collector_unit_t::collect_operand()
   */
  void allocate_reads();

  // types

  class collector_unit_t; // [한국어] 전방 선언 — op_t가 collector_unit_t 포인터를 참조하기 위해 필요

  /*
   * [한국어] op_t — 단일 레지스터 오퍼랜드 요청 (Operand Request)
   *
   * === 클래스 역할 ===
   * 레지스터 파일 뱅크에 대한 읽기(소스 오퍼랜드) 또는 쓰기(목적 오퍼랜드) 요청을
   * 나타내는 경량 데이터 구조이다. arbiter_t의 요청 큐와 allocation_t에서 사용된다.
   * 두 가지 생성자가 있다:
   *   1. collector_unit_t 기반: 소스 오퍼랜드 수집을 위한 CU → 뱅크 읽기 요청
   *   2. warp_inst_t 기반: 라이트백(writeback) 시 목적 레지스터 쓰기 요청
   *
   * === 아키텍처에서의 위치 ===
   * opndcoll_rfu_t 내부에서만 사용된다.
   * arbiter_t::m_queue[bank]에 op_t 요청들이 적재되어 뱅크 충돌을 중재받는다.
   *
   * === 타 모듈과의 연결 ===
   * - 생산자: allocate_cu() (CU 기반 op_t 생성), writeback() (warp 기반 op_t 생성)
   * - 소비자: arbiter_t (allocate_reads()에서 충돌 없는 요청을 선택하여 처리)
   */
  class op_t {
   public:
    /*
     * [한국어] op_t 기본 생성자 — 유효하지 않은 오퍼랜드 요청을 생성한다.
     * m_valid = false로 초기화되어, valid() 검사로 빈 슬롯임을 알 수 있다.
     */
    op_t() { m_valid = false; }
    /*
     * [한국어] op_t(collector_unit_t 기반) — CU의 소스 오퍼랜드 읽기 요청을 생성한다.
     *
     * @param cu: 이 오퍼랜드를 수집하려는 콜렉터 유닛
     * @param op: 명령어 내 오퍼랜드 위치 인덱스 (0 = 첫 번째 소스 레지스터)
     * @param reg: 소스 레지스터 번호 (예: R3)
     * @param num_banks: 레지스터 파일의 총 뱅크 수
     * @param sub_core_model: 서브코어 모델 활성화 여부 (스케줄러별 전용 뱅크 분리)
     * @param banks_per_sched: 서브코어 모델에서 스케줄러당 뱅크 수
     * @param sched_id: 이 명령어를 이슈한 스케줄러 ID
     *
     * register_bank()를 호출해 레지스터 번호와 warp ID로 뱅크 번호를 결정한다.
     *
     * 호출 체인:
     *   allocate_cu() → collector_unit_t::allocate() → [op_t(cu, op, reg, ...)] → arbiter에 등록
     */
    op_t(collector_unit_t *cu, unsigned op, unsigned reg, unsigned num_banks,
         bool sub_core_model, unsigned banks_per_sched, unsigned sched_id) {
      m_valid = true;      // [한국어] 유효한 요청으로 표시
      m_warp = NULL;       // [한국어] CU 기반 요청이므로 warp 포인터는 NULL
      m_cu = cu;           // [한국어] 이 오퍼랜드를 수집할 CU
      m_operand = op;      // [한국어] 명령어 내 오퍼랜드 인덱스
      m_register = reg;    // [한국어] 접근할 레지스터 번호
      m_shced_id = sched_id; // [한국어] 이슈한 스케줄러 ID (서브코어 모델에서 뱅크 분리에 사용)
      m_bank = register_bank(reg, cu->get_warp_id(), num_banks, sub_core_model,
                             banks_per_sched, sched_id); // [한국어] 레지스터 번호와 warp ID로 해당 뱅크 번호 계산
    }
    /*
     * [한국어] op_t(warp_inst_t 기반) — 라이트백 단계의 목적 레지스터 쓰기 요청을 생성한다.
     *
     * @param warp: 라이트백할 명령어 (EX_WB 파이프라인 레지스터)
     * @param reg: 목적 레지스터 번호
     * @param num_banks, sub_core_model, banks_per_sched, sched_id: 뱅크 계산 파라미터
     *
     * CU 포인터 없이 warp_inst_t 포인터만으로 생성되며, writeback() 경로에서 사용된다.
     *
     * 호출 체인:
     *   writeback() → [op_t(warp, reg, ...)] → arbiter_t::allocate_bank_for_write()
     */
    op_t(const warp_inst_t *warp, unsigned reg, unsigned num_banks,
         bool sub_core_model, unsigned banks_per_sched, unsigned sched_id) {
      m_valid = true;       // [한국어] 유효한 요청으로 표시
      m_warp = warp;        // [한국어] 라이트백할 warp 명령어 포인터
      m_register = reg;     // [한국어] 쓸 목적 레지스터 번호
      m_cu = NULL;          // [한국어] warp 기반 요청이므로 CU 포인터는 NULL
      m_operand = -1;       // [한국어] 라이트백 요청은 특정 오퍼랜드 인덱스가 없으므로 -1
      m_shced_id = sched_id; // [한국어] 이슈한 스케줄러 ID
      m_bank = register_bank(reg, warp->warp_id(), num_banks, sub_core_model,
                             banks_per_sched, sched_id); // [한국어] warp ID로 뱅크 번호 계산
    }

    /*
     * [한국어] valid — 이 오퍼랜드 요청이 유효한지 확인한다.
     * @return: m_valid가 true이면 유효한 요청, false이면 빈 슬롯
     */
    // accessors
    bool valid() const { return m_valid; }
    /*
     * [한국어] get_reg — 이 요청의 레지스터 번호를 반환한다.
     * @return: 레지스터 번호 (m_valid가 true인 경우만 호출 가능)
     */
    unsigned get_reg() const {
      assert(m_valid); // [한국어] 유효하지 않은 요청에 대한 접근 방지
      return m_register;
    }
    /*
     * [한국어] get_wid — 이 요청이 속한 warp ID를 반환한다.
     * CU 기반이면 CU에서, warp 기반이면 warp에서 warp ID를 가져온다.
     * 둘 다 NULL이면 abort() (논리적 불가 상태).
     */
    unsigned get_wid() const {
      if (m_warp)          // [한국어] warp 기반 요청 (라이트백)
        return m_warp->warp_id();
      else if (m_cu)       // [한국어] CU 기반 요청 (읽기)
        return m_cu->get_warp_id();
      else
        abort(); // [한국어] 둘 다 NULL이면 비정상 상태 — abort
    }
    /*
     * [한국어] get_sid — 이 요청을 이슈한 스케줄러 ID를 반환한다.
     * 서브코어 모델에서 스케줄러별 뱅크 파티셔닝에 사용된다.
     */
    unsigned get_sid() const { return m_shced_id; }
    /*
     * [한국어] get_active_count — 이 warp의 활성 스레드 수를 반환한다.
     * CU 또는 warp에서 active mask의 popcount를 가져온다.
     */
    unsigned get_active_count() const {
      if (m_warp)
        return m_warp->active_count();
      else if (m_cu)
        return m_cu->get_active_count();
      else
        abort();
    }
    /*
     * [한국어] get_active_mask — 이 warp의 활성 스레드 마스크를 반환한다.
     * 32비트 비트맵으로, 각 비트가 스레드의 활성 여부를 나타낸다.
     */
    const active_mask_t &get_active_mask() {
      if (m_warp)
        return m_warp->get_active_mask();
      else if (m_cu)
        return m_cu->get_active_mask();
      else
        abort();
    }
    /*
     * [한국어] get_sp_op — 이 명령어의 단정밀도 연산 타입을 반환한다.
     * AccelWattch 전력 모델에서 연산 유형별 전력 카운팅에 사용된다.
     */
    unsigned get_sp_op() const {
      if (m_warp)
        return m_warp->sp_op;
      else if (m_cu)
        return m_cu->get_sp_op();
      else
        abort();
    }
    /*
     * [한국어] get_oc_id — 이 요청이 속한 CU의 하드웨어 ID를 반환한다.
     * CU 기반 요청에서만 유효 (라이트백 경로에서는 호출 불가).
     */
    unsigned get_oc_id() const { return m_cu->get_id(); }
    /*
     * [한국어] get_bank — 이 레지스터가 속한 뱅크 번호를 반환한다.
     * 생성자에서 register_bank()로 미리 계산된 값을 반환한다.
     */
    unsigned get_bank() const { return m_bank; }
    /*
     * [한국어] get_operand — 명령어 내 오퍼랜드 위치 인덱스를 반환한다.
     * 라이트백 요청의 경우 -1 (unsigned 최대값)을 반환한다.
     */
    unsigned get_operand() const { return m_operand; }
    /*
     * [한국어] dump — 이 오퍼랜드 요청의 상태를 fp에 출력한다 (디버깅용).
     */
    void dump(FILE *fp) const {
      if (m_cu)                      // [한국어] CU 기반 읽기 요청 출력
        fprintf(fp, " <R%u, CU:%u, w:%02u> ", m_register, m_cu->get_id(),
                m_cu->get_warp_id());
      else if (!m_warp->empty())    // [한국어] warp 기반 쓰기 요청 출력 (warp가 비어있지 않은 경우)
        fprintf(fp, " <R%u, wid:%02u> ", m_register, m_warp->warp_id());
    }
    /*
     * [한국어] get_reg_string — 레지스터 번호를 "R<N>" 형식의 문자열로 반환한다.
     * 디버깅 및 로그 출력에 사용된다.
     */
    std::string get_reg_string() const {
      char buffer[64];              // [한국어] 로컬 버퍼 (최대 64바이트면 충분)
      snprintf(buffer, 64, "R%u", m_register); // [한국어] "R<번호>" 형식으로 포맷
      return std::string(buffer);  // [한국어] std::string으로 반환
    }

    /*
     * [한국어] reset — 이 오퍼랜드 요청을 무효화한다.
     * 수집 완료 또는 CU 해제 시 호출된다.
     */
    // modifiers
    void reset() { m_valid = false; } // [한국어] 유효 플래그를 false로 설정 — 빈 슬롯으로 표시

   private:
    bool m_valid;
    /* [한국어] 이 오퍼랜드 요청의 유효 여부.
     * 설정자: 생성자(true), reset()(false).
     * 읽는 자: valid()로 arbiter가 유효 요청 여부를 판단.
     * 값 범위: true(유효) / false(빈 슬롯).
     * 동기화: 단일 스레드(step()) 내에서만 접근하므로 락 불필요. */

    collector_unit_t *m_cu;
    /* [한국어] 이 오퍼랜드를 보유한 콜렉터 유닛 포인터.
     * 설정자: CU 기반 생성자에서 설정, warp 기반 생성자에서는 NULL.
     * 읽는 자: get_wid(), get_oc_id(), dump() 등에서 사용.
     * 값 범위: 유효한 CU 포인터 또는 NULL (warp 기반 경우).
     * 동기화: 단일 스레드 접근, 별도 락 불필요. */

    const warp_inst_t *m_warp;
    /* [한국어] warp 기반 쓰기 요청의 경우 라이트백 명령어 포인터.
     * 설정자: warp 기반 생성자에서 설정, CU 기반 생성자에서는 NULL.
     * 읽는 자: get_wid(), get_active_count(), dump() 등에서 사용.
     * 값 범위: 유효한 warp_inst_t 포인터 또는 NULL.
     * 동기화: 읽기 전용 (포인터 자체는 변경되지 않음). */

    unsigned m_operand;  // operand offset in instruction. e.g., add r1,r2,r3;
                         // r2 is oprd 0, r3 is 1 (r1 is dst)
    /* [한국어] 명령어 내 오퍼랜드 위치 인덱스.
     * 예: "add r1, r2, r3"에서 r2는 인덱스 0, r3는 인덱스 1 (r1은 목적지).
     * 설정자: CU 기반 생성자에서 op 파라미터로, warp 기반에서는 -1(unsigned 최대값).
     * 읽는 자: get_operand(), collector_unit_t::collect_operand()에서 m_not_ready 비트 클리어 시 사용.
     * 값 범위: 0 ~ MAX_REG_OPERANDS*2-1 또는 (unsigned)-1 (warp 기반).
     * 동기화: 읽기 전용. */

    unsigned m_register;
    /* [한국어] 접근할 레지스터 번호 (가상 레지스터 인덱스).
     * 설정자: 두 생성자 모두 reg 파라미터로 설정.
     * 읽는 자: get_reg(), register_bank() (뱅크 계산), dump().
     * 값 범위: 0 ~ (레지스터 파일 크기-1).
     * 동기화: 읽기 전용. */

    unsigned m_bank;
    /* [한국어] 이 레지스터가 속한 뱅크 번호 (미리 계산된 값).
     * 설정자: 생성자에서 register_bank(reg, warp_id, num_banks, ...) 결과.
     * 읽는 자: arbiter_t에서 m_queue[m_bank]에 요청을 추가할 때 사용.
     * 값 범위: 0 ~ (num_banks-1).
     * 동기화: 읽기 전용. */

    unsigned m_shced_id;  // scheduler id that has issued this inst
    /* [한국어] 이 명령어를 이슈한 스케줄러의 ID.
     * 서브코어 모델(sub_core_model)에서는 스케줄러마다 전용 뱅크 파티션이 존재하며,
     * 이 ID로 접근 가능한 뱅크 범위를 제한한다.
     * 설정자: 두 생성자 모두 sched_id 파라미터로 설정.
     * 읽는 자: get_sid(), register_bank() 계산 시 참조.
     * 값 범위: 0 ~ (gpgpu_num_sched_per_core-1).
     * 동기화: 읽기 전용. */
  };

  /*
   * [한국어] alloc_t — 레지스터 파일 뱅크의 할당 상태 열거형
   *
   * 각 뱅크가 현재 사이클에 어떤 용도로 예약되었는지를 나타낸다.
   * 뱅크는 한 사이클에 읽기 또는 쓰기 중 하나만 허용된다.
   */
  enum alloc_t {
    NO_ALLOC,    // [한국어] 미할당 상태 — 이번 사이클에 이 뱅크로의 요청 없음
    READ_ALLOC,  // [한국어] 읽기 할당 — 소스 오퍼랜드 읽기를 위해 예약됨
    WRITE_ALLOC, // [한국어] 쓰기 할당 — 목적 레지스터 라이트백을 위해 예약됨
  };

  /*
   * [한국어] allocation_t — 레지스터 파일 뱅크의 현재 사이클 할당 상태 관리 클래스
   *
   * === 클래스 역할 ===
   * arbiter_t의 m_allocated_bank 배열에서 각 뱅크(bank)의 할당 상태를 추적한다.
   * 매 사이클 process_banks() → reset_alloction()에서 NO_ALLOC으로 초기화된다.
   * alloc_read() / alloc_write()를 통해 이번 사이클의 뱅크 사용 목적을 기록하며,
   * 동일 사이클에 같은 뱅크에 대해 두 번 이상 할당 시도를 차단한다 (assert).
   *
   * === 타 모듈과의 연결 ===
   * - 소유자: arbiter_t::m_allocated_bank 배열
   * - 갱신자: arbiter_t::allocate_bank_for_write(), allocate_for_read()
   * - 리셋: arbiter_t::reset_alloction() (매 사이클)
   */
  class allocation_t {
   public:
    /*
     * [한국어] allocation_t 생성자 — 기본 상태(미할당)로 초기화한다.
     */
    allocation_t() { m_allocation = NO_ALLOC; } // [한국어] 기본값: 미할당 상태
    /*
     * [한국어] is_read — 이 뱅크가 현재 읽기 할당 중인지 확인한다.
     * @return: READ_ALLOC 상태이면 true
     */
    bool is_read() const { return m_allocation == READ_ALLOC; }
    /*
     * [한국어] is_write — 이 뱅크가 현재 쓰기 할당 중인지 확인한다.
     * @return: WRITE_ALLOC 상태이면 true
     */
    bool is_write() const { return m_allocation == WRITE_ALLOC; }
    /*
     * [한국어] is_free — 이 뱅크가 현재 미할당 상태인지 확인한다.
     * @return: NO_ALLOC 상태이면 true — arbiter가 새 요청 배정 가능 여부 판단에 사용
     */
    bool is_free() const { return m_allocation == NO_ALLOC; }
    /*
     * [한국어] dump — 이 뱅크의 할당 상태를 fp에 출력한다 (디버깅용).
     */
    void dump(FILE *fp) const {
      if (m_allocation == NO_ALLOC) {       // [한국어] 미할당 상태
        fprintf(fp, "<free>");
      } else if (m_allocation == READ_ALLOC) { // [한국어] 읽기 할당 상태 — 어떤 오퍼랜드 요청인지 출력
        fprintf(fp, "rd: ");
        m_op.dump(fp);
      } else if (m_allocation == WRITE_ALLOC) { // [한국어] 쓰기 할당 상태 — 어떤 라이트백인지 출력
        fprintf(fp, "wr: ");
        m_op.dump(fp);
      }
      fprintf(fp, "\n");
    }
    /*
     * [한국어] alloc_read — 이 뱅크를 읽기 목적으로 할당한다.
     *
     * @param op: 읽기 요청 오퍼랜드 정보
     *
     * 이미 할당된 뱅크에 다시 할당 시도 시 assert로 즉시 abort한다.
     * 같은 사이클에 한 뱅크는 하나의 읽기 요청만 처리 가능하다.
     *
     * 호출 체인:
     *   arbiter_t::allocate_reads() → allocate_for_read() → [alloc_read()]
     */
    void alloc_read(const op_t &op) {
      assert(is_free()); // [한국어] 이미 할당된 뱅크에 중복 할당 방지 — 뱅크 충돌 오류 감지
      m_allocation = READ_ALLOC; // [한국어] 읽기 할당 상태로 변경
      m_op = op;                 // [한국어] 이 뱅크를 사용하는 오퍼랜드 요청 저장
    }
    /*
     * [한국어] alloc_write — 이 뱅크를 쓰기 목적으로 할당한다.
     *
     * @param op: 쓰기(라이트백) 요청 오퍼랜드 정보
     *
     * 읽기와 동일하게 중복 할당을 assert로 차단한다.
     *
     * 호출 체인:
     *   arbiter_t::allocate_bank_for_write() → [alloc_write()]
     */
    void alloc_write(const op_t &op) {
      assert(is_free()); // [한국어] 이미 할당된 뱅크에 쓰기 시도 방지
      m_allocation = WRITE_ALLOC; // [한국어] 쓰기 할당 상태로 변경
      m_op = op;                  // [한국어] 라이트백 오퍼랜드 요청 저장
    }
    /*
     * [한국어] reset — 이 뱅크의 할당 상태를 NO_ALLOC으로 초기화한다.
     * 매 사이클 process_banks() → reset_alloction()에서 호출된다.
     */
    void reset() { m_allocation = NO_ALLOC; } // [한국어] 다음 사이클을 위해 뱅크 할당 상태 초기화

   private:
    enum alloc_t m_allocation;
    /* [한국어] 현재 사이클에서 이 뱅크의 할당 상태.
     * 설정자: alloc_read()(READ_ALLOC), alloc_write()(WRITE_ALLOC), reset()(NO_ALLOC).
     * 읽는 자: is_read(), is_write(), is_free(), dump().
     * 값 범위: NO_ALLOC / READ_ALLOC / WRITE_ALLOC.
     * 동기화: 단일 사이클 단일 스레드 접근, 락 불필요. */

    op_t m_op;
    /* [한국어] 이 뱅크를 사용하고 있는 오퍼랜드 요청의 사본.
     * 설정자: alloc_read() / alloc_write()에서 op를 복사 저장.
     * 읽는 자: dump()에서 어떤 레지스터가 이 뱅크를 사용 중인지 출력.
     * 값 범위: 유효한 op_t (NO_ALLOC 상태에서는 이전 값이 남아있을 수 있으나 무시됨).
     * 동기화: alloc_* 후 reset() 전까지 읽기 전용. */
  };

  /*
   * [한국어] arbiter_t — 레지스터 파일 뱅크 접근을 중재하는 라운드 로빈 아비터
   *
   * === 클래스 역할 ===
   * 여러 CU에서 동일 사이클에 동일 뱅크로 오는 읽기/쓰기 요청들을 중재한다.
   * 각 뱅크에는 요청 큐(m_queue[bank])가 있고, 매 사이클 allocate_reads()를 호출하면
   * iSLIP 또는 라운드 로빈 방식으로 각 뱅크에 하나의 CU만 서비스를 허용한다.
   * 충돌하는 나머지 CU는 다음 사이클에 재시도한다.
   *
   * 내부적으로 _inmatch, _outmatch, _request 행렬을 사용하는
   * 이분 그래프 매칭(bipartite matching) 방식으로 CU-뱅크 충돌을 해소한다.
   *
   * === 아키텍처에서의 위치 ===
   * opndcoll_rfu_t 내부에 단 하나의 인스턴스로 존재한다.
   * 모든 CU가 공유하며, 매 사이클 step()의 allocate_reads() 단계에서 동작한다.
   *
   * === 타 모듈과의 연결 ===
   * - 소유자: opndcoll_rfu_t::m_arbiter
   * - 호출자: opndcoll_rfu_t::allocate_reads(), writeback()
   * - 데이터: collector_unit_t에서 읽기 요청, writeback()에서 쓰기 요청
   */
  class arbiter_t {
   public:
    /*
     * [한국어] arbiter_t 기본 생성자 — 모든 포인터를 NULL로, m_last_cu를 0으로 초기화한다.
     * init() 호출 전에는 사용 불가하다.
     */
    // constructors
    arbiter_t() {
      m_queue = NULL;            // [한국어] 뱅크별 요청 큐 — init() 후 new로 할당
      m_allocated_bank = NULL;   // [한국어] 뱅크별 할당 상태 배열 — init() 후 할당
      m_allocator_rr_head = NULL; // [한국어] CU별 라운드 로빈 헤드 — init() 후 할당
      _inmatch = NULL;           // [한국어] 이분 매칭용 입력 매칭 배열 — init() 후 할당
      _outmatch = NULL;          // [한국어] 이분 매칭용 출력 매칭 배열 — init() 후 할당
      _request = NULL;           // [한국어] 이분 매칭용 요청 행렬 — init() 후 할당
      m_last_cu = 0;             // [한국어] 라운드 로빈에서 마지막으로 서비스된 CU 인덱스
    }
    /*
     * [한국어] init — 뱅크 수와 CU 수로 아비터를 초기화한다.
     *
     * @param num_cu: 전체 collector unit 수 (이분 그래프의 한 쪽)
     * @param num_banks: 레지스터 파일 뱅크 수 (이분 그래프의 다른 쪽)
     *
     * 이분 매칭에 필요한 배열들을 동적 할당하고,
     * 각 CU의 라운드 로빈 헤드를 n % num_banks로 초기화(분산 배치)한다.
     *
     * 호출 체인:
     *   opndcoll_rfu_t::init() → [arbiter_t::init()]
     */
    void init(unsigned num_cu, unsigned num_banks) {
      assert(num_cu > 0);    // [한국어] CU 수는 반드시 양수
      assert(num_banks > 0); // [한국어] 뱅크 수는 반드시 양수
      m_num_collectors = num_cu;   // [한국어] CU 총 개수 저장
      m_num_banks = num_banks;     // [한국어] 뱅크 총 개수 저장
      _inmatch = new int[m_num_banks];      // [한국어] 뱅크당 1개의 매칭 결과 저장 배열
      _outmatch = new int[m_num_collectors]; // [한국어] CU당 1개의 매칭 결과 저장 배열
      _request = new int *[m_num_banks];    // [한국어] 행렬[뱅크][CU]: 요청 여부 (0/1)
      for (unsigned i = 0; i < m_num_banks; i++)
        _request[i] = new int[m_num_collectors]; // [한국어] 각 행(뱅크)에 CU 수만큼 열 할당
      m_queue = new std::list<op_t>[num_banks]; // [한국어] 뱅크별 요청 큐 배열 동적 할당
      m_allocated_bank = new allocation_t[num_banks]; // [한국어] 뱅크별 할당 상태 배열 할당
      m_allocator_rr_head = new unsigned[num_cu]; // [한국어] CU별 라운드 로빈 시작 뱅크 인덱스
      for (unsigned n = 0; n < num_cu; n++)
        m_allocator_rr_head[n] = n % num_banks; // [한국어] CU마다 시작 뱅크를 다르게 설정 — 초기 분산
      reset_alloction(); // [한국어] 초기화 완료 후 모든 뱅크 할당 상태를 NO_ALLOC으로 리셋
    }

    /*
     * [한국어] dump — 아비터의 현재 상태(요청 큐 및 뱅크 할당)를 fp에 출력한다 (디버깅용).
     */
    // accessors
    void dump(FILE *fp) const {
      fprintf(fp, "\n");
      fprintf(fp, "  Arbiter State:\n");
      fprintf(fp, "  requests:\n"); // [한국어] 뱅크별 대기 요청 큐 출력
      for (unsigned b = 0; b < m_num_banks; b++) {
        fprintf(fp, "    bank %u : ", b); // [한국어] 뱅크 번호 출력
        std::list<op_t>::const_iterator o = m_queue[b].begin();
        for (; o != m_queue[b].end(); o++) { // [한국어] 이 뱅크의 모든 대기 요청 출력
          o->dump(fp);
        }
        fprintf(fp, "\n");
      }
      fprintf(fp, "  grants:\n"); // [한국어] 이번 사이클 뱅크 할당 결과 출력
      for (unsigned b = 0; b < m_num_banks; b++) {
        fprintf(fp, "    bank %u : ", b);
        m_allocated_bank[b].dump(fp); // [한국어] 이 뱅크의 할당 상태 출력
      }
      fprintf(fp, "\n");
    }

    /*
     * [한국어] allocate_reads — 이번 사이클의 뱅크 읽기 요청들을 중재하여 서비스 가능한 것들을 반환한다.
     *
     * @return: 이번 사이클에 서비스될 op_t 목록 (충돌 없이 뱅크에 접근 가능한 요청들)
     *
     * 각 뱅크의 요청 큐(m_queue[bank])에서 라운드 로빈으로 하나씩 선택하며,
     * 선택된 요청은 m_allocated_bank[bank]에 READ_ALLOC으로 기록한다.
     * 서비스되지 못한 요청은 다음 사이클에 재시도된다.
     *
     * 호출 체인:
     *   opndcoll_rfu_t::allocate_reads() → [arbiter_t::allocate_reads()] → op_t 목록 반환
     *                                      → collector_unit_t::collect_operand()
     */
    // modifiers
    std::list<op_t> allocate_reads();

    /*
     * [한국어] add_read_requests — CU의 미수집 오퍼랜드들을 뱅크 요청 큐에 추가한다.
     *
     * @param cu: 오퍼랜드를 수집 중인 콜렉터 유닛
     *
     * CU의 m_src_op 배열에서 유효한(valid()) 오퍼랜드를 순회하여
     * 각 오퍼랜드의 뱅크에 해당하는 m_queue[bank]에 op_t를 push_back한다.
     * 이 큐들은 다음 allocate_reads() 호출에서 중재된다.
     *
     * 호출 체인:
     *   opndcoll_rfu_t::allocate_reads() → [add_read_requests(cu)] → m_queue[bank].push_back()
     */
    void add_read_requests(collector_unit_t *cu) {
      const op_t *src = cu->get_operands(); // [한국어] CU의 소스 오퍼랜드 배열 포인터 획득
      for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) { // [한국어] 모든 오퍼랜드 슬롯 순회 (소스+목적 각 MAX_REG_OPERANDS개)
        const op_t &op = src[i]; // [한국어] i번째 오퍼랜드 참조
        if (op.valid()) {         // [한국어] 유효한 오퍼랜드만 처리 (미할당 슬롯 스킵)
          unsigned bank = op.get_bank(); // [한국어] 이 오퍼랜드의 레지스터가 속한 뱅크 번호
          m_queue[bank].push_back(op);   // [한국어] 해당 뱅크의 요청 큐에 추가 (FIFO)
        }
      }
    }
    /*
     * [한국어] bank_idle — 특정 뱅크가 이번 사이클에 미할당 상태인지 확인한다.
     *
     * @param bank: 확인할 뱅크 번호
     * @return: 미할당(NO_ALLOC) 상태이면 true
     *
     * writeback() 등에서 쓰기 할당 전 뱅크 가용성 확인에 사용된다.
     */
    bool bank_idle(unsigned bank) const {
      return m_allocated_bank[bank].is_free(); // [한국어] 이 뱅크의 할당 상태가 NO_ALLOC인지 반환
    }
    /*
     * [한국어] allocate_bank_for_write — 특정 뱅크를 쓰기(라이트백) 목적으로 예약한다.
     *
     * @param bank: 쓰기 할당할 뱅크 번호
     * @param op: 라이트백 오퍼랜드 요청 정보
     *
     * 유효한 뱅크 번호인지 assert 검사 후 allocation_t::alloc_write()를 호출한다.
     *
     * 호출 체인:
     *   opndcoll_rfu_t::writeback() → [allocate_bank_for_write()]
     */
    void allocate_bank_for_write(unsigned bank, const op_t &op) {
      assert(bank < m_num_banks); // [한국어] 유효 범위 검사 — 잘못된 뱅크 번호 방지
      m_allocated_bank[bank].alloc_write(op); // [한국어] 이 뱅크를 WRITE_ALLOC으로 설정
    }
    /*
     * [한국어] allocate_for_read — 특정 뱅크를 읽기 목적으로 예약한다.
     *
     * @param bank: 읽기 할당할 뱅크 번호
     * @param op: 읽기 오퍼랜드 요청 정보
     *
     * allocate_reads()에서 중재 결과로 선택된 요청을 뱅크에 공식 등록할 때 호출된다.
     *
     * 호출 체인:
     *   opndcoll_rfu_t::allocate_reads() → arbiter_t::allocate_reads() → [allocate_for_read()]
     */
    void allocate_for_read(unsigned bank, const op_t &op) {
      assert(bank < m_num_banks); // [한국어] 유효 범위 검사
      m_allocated_bank[bank].alloc_read(op); // [한국어] 이 뱅크를 READ_ALLOC으로 설정
    }
    /*
     * [한국어] reset_alloction — 모든 뱅크의 할당 상태를 NO_ALLOC으로 리셋한다.
     *
     * 매 사이클 process_banks()에서 호출되어 다음 사이클을 위해 뱅크를 해제한다.
     * 참고: 함수명의 "alloction"은 "allocation"의 오타이나 원본 코드 유지를 위해 그대로 사용.
     *
     * 호출 체인:
     *   opndcoll_rfu_t::process_banks() → [reset_alloction()]
     */
    void reset_alloction() {
      for (unsigned b = 0; b < m_num_banks; b++) m_allocated_bank[b].reset(); // [한국어] 모든 뱅크를 NO_ALLOC으로 리셋
    }

   private:
    unsigned m_num_banks;
    /* [한국어] 레지스터 파일의 총 뱅크 수.
     * 설정자: init()에서 num_banks 파라미터로 설정.
     * 읽는 자: dump(), reset_alloction(), allocate_bank_for_write(), allocate_for_read().
     * 값 범위: gpgpusim.config의 -gpgpu_num_reg_banks 값 (보통 8~32).
     * 동기화: 읽기 전용 (init() 이후 변경 없음). */

    unsigned m_num_collectors;
    /* [한국어] 전체 콜렉터 유닛의 수.
     * 설정자: init()에서 num_cu 파라미터로 설정.
     * 읽는 자: _outmatch 배열 크기, m_allocator_rr_head 초기화에 사용.
     * 값 범위: 실행 유닛별 CU 수의 합계.
     * 동기화: 읽기 전용. */

    allocation_t *m_allocated_bank;  // bank # -> register that wins
    /* [한국어] 각 뱅크의 현재 사이클 할당 상태 배열 (크기: m_num_banks).
     * 설정자: alloc_read/write → allocation_t::alloc_*().
     * 읽는 자: bank_idle(), dump(), 라이트백 충돌 검사.
     * 값 범위: allocation_t 배열, 각 원소는 NO_ALLOC/READ_ALLOC/WRITE_ALLOC.
     * 동기화: 매 사이클 reset_alloction()으로 클리어, 단일 스레드 접근. */

    std::list<op_t> *m_queue;
    /* [한국어] 뱅크별 읽기 요청 대기 큐 (크기: m_num_banks, 각 원소는 list<op_t>).
     * 설정자: add_read_requests()에서 유효한 오퍼랜드를 push_back.
     * 읽는 자: allocate_reads()에서 중재 시 순회하여 pop_front.
     * 값 범위: 빈 리스트 ~ 최대 m_num_collectors개 요청.
     * 동기화: 단일 step() 흐름 내에서 접근, 별도 락 불필요. */

    unsigned *
        m_allocator_rr_head;  // cu # -> next bank to check for request (rr-arb)
    /* [한국어] CU별 라운드 로빈 아비트레이션 시작 뱅크 인덱스 배열 (크기: m_num_collectors).
     * 각 CU마다 독립적인 RR 상태를 유지하여, 동일 CU가 항상 같은 뱅크를 우선하는 것을 방지.
     * 설정자: init()에서 n % num_banks로 초기화, allocate_reads()에서 서비스 후 갱신.
     * 읽는 자: allocate_reads()에서 어떤 뱅크부터 중재를 시작할지 결정.
     * 값 범위: 0 ~ m_num_banks-1.
     * 동기화: 단일 스레드 접근. */

    unsigned m_last_cu;       // first cu to check while arb-ing banks (rr)
    /* [한국어] 마지막으로 서비스된 CU 인덱스 (뱅크 단위 라운드 로빈의 시작점).
     * 설정자: allocate_reads()에서 CU를 서비스할 때마다 갱신.
     * 읽는 자: allocate_reads()에서 다음 사이클의 시작 CU 결정.
     * 값 범위: 0 ~ m_num_collectors-1.
     * 동기화: 단일 스레드 접근. */

    int *_inmatch;
    /* [한국어] 이분 매칭 알고리즘용 — 각 뱅크(입력 포트)에 매칭된 CU 인덱스 배열 (크기: m_num_banks).
     * -1이면 미매칭, 양수이면 해당 CU와 매칭됨.
     * 설정자: allocate_reads() 내 매칭 알고리즘에서 설정.
     * 읽는 자: 매칭 결과를 뱅크 할당에 반영할 때 사용.
     * 동기화: allocate_reads() 호출 내에서만 사용. */

    int *_outmatch;
    /* [한국어] 이분 매칭 알고리즘용 — 각 CU(출력 포트)에 매칭된 뱅크 인덱스 배열 (크기: m_num_collectors).
     * -1이면 미매칭, 양수이면 해당 뱅크와 매칭됨.
     * 설정자/읽는 자: allocate_reads() 내 매칭 알고리즘.
     * 동기화: allocate_reads() 호출 내에서만 사용. */

    int **_request;
    /* [한국어] 이분 매칭용 요청 행렬 — _request[bank][cu]: CU가 이 뱅크를 요청 중인지(1/0).
     * 크기: m_num_banks × m_num_collectors.
     * 설정자: allocate_reads() 시작 시 m_queue를 스캔하여 구성.
     * 읽는 자: 매칭 알고리즘에서 가능한 CU-뱅크 연결 탐색.
     * 동기화: allocate_reads() 내에서만 사용. */
  };

  /*
   * [한국어] input_port_t — 오퍼랜드 컬렉터의 입/출력 포트와 연결된 CU 집합 묶음
   *
   * === 클래스 역할 ===
   * 하나의 실행 유닛 경로(예: SP 유닛 경로)에 대응하는 입력 파이프라인 레지스터 집합(ID_OC_SP 등),
   * 출력 파이프라인 레지스터 집합(OC_EX_SP 등), 그리고 이 포트를 담당하는 CU 집합 ID들을
   * 하나로 묶는 구조체이다.
   *
   * === 아키텍처에서의 위치 ===
   * opndcoll_rfu_t::m_in_ports 벡터에 저장된다. add_port()에서 추가된다.
   * allocate_cu(port)에서 m_in_ports[port]를 통해 어떤 파이프라인 레지스터를 서비스할지 결정한다.
   *
   * === 타 모듈과의 연결 ===
   * - 생성자: opndcoll_rfu_t::add_port() (shader_core_ctx::create_exec_pipeline()에서 호출)
   * - 사용자: allocate_cu()에서 m_in, m_out, m_cu_sets 접근
   */
  class input_port_t {
   public:
    /*
     * [한국어] input_port_t 생성자 — 입출력 포트와 CU 집합을 초기화한다.
     *
     * @param input: ID_OC_* 파이프라인 레지스터들의 포인터 벡터 (명령어가 들어오는 쪽)
     * @param output: OC_EX_* 파이프라인 레지스터들의 포인터 벡터 (명령어가 나가는 쪽)
     * @param cu_sets: 이 포트를 처리하는 CU 집합 ID 목록
     *
     * input과 output의 크기가 같아야 하고 (1:1 대응), cu_sets은 비어있으면 안 된다.
     */
    input_port_t(port_vector_t &input, port_vector_t &output,
                 uint_vector_t cu_sets)
        : m_in(input), m_out(output), m_cu_sets(cu_sets) { // [한국어] 레퍼런스로 받아 복사 저장
      assert(input.size() == output.size()); // [한국어] 입출력 포트 수가 같아야 함 — 1:1 대응 보장
      assert(not m_cu_sets.empty());         // [한국어] 최소 1개의 CU 집합은 있어야 함
    }
    // private:
    port_vector_t m_in, m_out;
    /* [한국어] m_in: 이 포트로 명령어가 들어오는 ID_OC_* 파이프라인 레지스터 집합 포인터 벡터.
     *          m_out: 이 포트에서 명령어가 나가는 OC_EX_* 파이프라인 레지스터 집합 포인터 벡터.
     * 설정자: 생성자에서 초기화 후 변경 없음.
     * 읽는 자: allocate_cu()에서 대기 중인 명령어 확인 및 디스패치 대상 레지스터 결정.
     * 값 범위: 유효한 register_set* 포인터들. 크기는 실행 유닛 종류 수에 따라 결정.
     * 동기화: 읽기 전용. */

    uint_vector_t m_cu_sets;
    /* [한국어] 이 포트가 사용할 CU 집합(set) ID들의 목록.
     * 예: SP 포트는 SP 전용 CU 집합과 범용 CU 집합을 모두 포함할 수 있다.
     * 설정자: 생성자에서 초기화 후 변경 없음.
     * 읽는 자: allocate_cu()에서 어떤 CU 집합에서 빈 CU를 찾을지 결정.
     * 값 범위: m_cus 맵의 유효한 키 값들.
     * 동기화: 읽기 전용. */
  };

  /*
   * [한국어] collector_unit_t — 오퍼랜드 수집 버퍼 유닛 (Collector Unit)
   *
   * === 클래스 역할 ===
   * 파이프라인 ID_OC_* 레지스터에서 할당받은 하나의 명령어를 보관하며,
   * 해당 명령어의 모든 소스 레지스터 오퍼랜드가 레지스터 파일에서 읽힐 때까지 대기한다.
   * m_not_ready 비트셋으로 아직 수집되지 않은 오퍼랜드를 추적하며,
   * 모든 비트가 클리어되면 ready() = true가 되고 dispatch_unit_t에 의해 OC_EX_*로 디스패치된다.
   *
   * 일반적으로 SM당 SP/DP/SFU/MEM 유닛별로 여러 CU가 존재해,
   * 하나의 CU가 대기하는 동안 다른 CU의 명령어가 진행될 수 있다.
   *
   * === 아키텍처에서의 위치 ===
   * opndcoll_rfu_t::m_cus 맵에서 CU 집합별로 vector<collector_unit_t>로 관리된다.
   * m_cu 포인터 벡터가 전체 CU를 참조한다.
   *
   * === 타 모듈과의 연결 ===
   * - 할당자: opndcoll_rfu_t::allocate_cu()
   * - 수집자: opndcoll_rfu_t::allocate_reads() → collect_operand()
   * - 디스패처: dispatch_unit_t::find_ready() → dispatch()
   */
  class collector_unit_t {
   public:
    /*
     * [한국어] collector_unit_t 기본 생성자 — 빈 CU를 초기화한다.
     *
     * m_free = true(사용 가능), m_src_op 배열을 MAX_REG_OPERANDS*2개로 할당,
     * m_not_ready를 0으로 리셋, m_warp_id를 -1(미할당)로 초기화한다.
     *
     * 호출 체인:
     *   opndcoll_rfu_t::add_cu_set() → vector<collector_unit_t> 생성 → [collector_unit_t()]
     */
    // constructors
    collector_unit_t() {
      m_free = true;              // [한국어] 초기에는 미사용 상태(빈 CU)
      m_warp = NULL;              // [한국어] 보유 중인 명령어 없음
      m_output_register = NULL;   // [한국어] 디스패치 대상 레지스터 미지정
      m_src_op = new op_t[MAX_REG_OPERANDS * 2]; // [한국어] 소스 오퍼랜드 배열 동적 할당 (읽기+쓰기 각 MAX_REG_OPERANDS개)
      m_not_ready.reset();        // [한국어] 미수집 오퍼랜드 비트셋 초기화 (모두 0 = ready)
      m_warp_id = -1;             // [한국어] warp ID 미할당 상태 (-1)
      m_num_banks = 0;            // [한국어] 뱅크 수 — init() 호출 전 0
    }
    /*
     * [한국어] ready — 이 CU가 모든 소스 오퍼랜드를 수집 완료했는지 확인한다.
     *
     * @return: m_not_ready가 모두 0이고 m_free가 false이면 true (디스패치 가능)
     *
     * m_not_ready.none()이면 모든 오퍼랜드가 수집된 상태이다.
     * 구현은 shader.cc에 있다.
     *
     * 호출 체인:
     *   dispatch_unit_t::find_ready() → [ready()]
     */
    // accessors
    bool ready() const;
    /*
     * [한국어] get_operands — 소스 오퍼랜드 요청 배열의 포인터를 반환한다.
     *
     * @return: m_src_op 배열의 시작 포인터 (크기: MAX_REG_OPERANDS * 2)
     *
     * arbiter_t::add_read_requests()에서 이 배열을 순회하며 뱅크 요청을 추가한다.
     */
    const op_t *get_operands() const { return m_src_op; }
    /*
     * [한국어] dump — 이 CU의 현재 상태를 fp에 출력한다 (디버깅용).
     *
     * @param shader: SM 포인터 (레지스터 이름 등 추가 정보 출력에 사용)
     */
    void dump(FILE *fp, const shader_core_ctx *shader) const;

    /*
     * [한국어] get_warp_id — 이 CU가 보유한 명령어의 warp ID를 반환한다.
     * @return: m_warp_id (allocate() 시 설정된 값, 미할당 시 -1)
     */
    unsigned get_warp_id() const { return m_warp_id; }
    /*
     * [한국어] get_active_count — 보유 중인 warp의 활성 스레드 수를 반환한다.
     * AccelWattch 전력 카운터 계산에 사용된다.
     */
    unsigned get_active_count() const { return m_warp->active_count(); }
    /*
     * [한국어] get_active_mask — 보유 중인 warp의 활성 스레드 마스크를 반환한다.
     */
    const active_mask_t &get_active_mask() const {
      return m_warp->get_active_mask();
    }
    /*
     * [한국어] get_sp_op — 보유 중인 명령어의 단정밀도 연산 타입을 반환한다.
     */
    unsigned get_sp_op() const { return m_warp->sp_op; }
    /*
     * [한국어] get_id — 이 CU의 하드웨어 고유 ID를 반환한다.
     * @return: m_cuid (전체 CU 배열에서의 고유 인덱스)
     */
    unsigned get_id() const { return m_cuid; }  // returns CU hw id
    /*
     * [한국어] get_reg_id — 서브코어 모델에서 이 CU가 접근 가능한 레지스터 범위 ID를 반환한다.
     */
    unsigned get_reg_id() const { return m_reg_id; }

    /*
     * [한국어] init — 이 CU를 특정 ID와 뱅크 수로 초기화한다.
     *
     * @param n: 이 CU의 하드웨어 ID
     * @param num_banks: 레지스터 파일 총 뱅크 수
     * @param config: SM 설정 (warp 크기 등)
     * @param rfu: 소속 opndcoll_rfu_t (디스패치 시 참조)
     * @param m_sub_core_model: 서브코어 모델 활성화 여부
     * @param reg_id: 서브코어 모델에서 이 CU가 담당하는 레지스터 범위 ID
     * @param num_banks_per_sched: 서브코어 모델에서 스케줄러당 뱅크 수
     *
     * 호출 체인:
     *   opndcoll_rfu_t::init() → [collector_unit_t::init()]
     */
    // modifiers
    void init(unsigned n, unsigned num_banks, const core_config *config,
              opndcoll_rfu_t *rfu, bool m_sub_core_model, unsigned reg_id,
              unsigned num_banks_per_sched);
    /*
     * [한국어] allocate — ID_OC 파이프라인 레지스터에서 명령어를 꺼내 이 CU에 할당한다.
     *
     * @param pipeline_reg: 명령어가 대기 중인 ID_OC_* 파이프라인 레지스터 집합
     * @param output_reg: 준비 완료 시 명령어를 내보낼 OC_EX_* 파이프라인 레지스터 집합
     * @return: 할당 성공 시 true, 실패(CU가 이미 사용 중 등) 시 false
     *
     * 성공 시 m_free = false로 설정, m_warp에 명령어 포인터 저장, m_src_op에 오퍼랜드 정보 설정,
     * m_not_ready 비트를 소스 레지스터 수만큼 세트한다.
     *
     * 호출 체인:
     *   opndcoll_rfu_t::allocate_cu() → [allocate(pipeline_reg, output_reg)]
     */
    bool allocate(register_set *pipeline_reg, register_set *output_reg);

    /*
     * [한국어] collect_operand — op번째 소스 오퍼랜드 수집 완료를 기록한다.
     *
     * @param op: 수집 완료된 오퍼랜드의 인덱스 (m_src_op 배열 인덱스)
     *
     * m_not_ready 비트셋의 op 번째 비트를 클리어한다.
     * 모든 비트가 클리어되면 ready() = true가 된다.
     *
     * 호출 체인:
     *   opndcoll_rfu_t::allocate_reads() → arbiter_t::allocate_reads() 결과 처리 → [collect_operand()]
     */
    void collect_operand(unsigned op) { m_not_ready.reset(op); } // [한국어] op번째 비트 클리어 — 해당 오퍼랜드 수집 완료 표시
    /*
     * [한국어] get_num_operands — 보유 명령어의 총 소스 오퍼랜드 수를 반환한다.
     */
    unsigned get_num_operands() const { return m_warp->get_num_operands(); }
    /*
     * [한국어] get_num_regs — 보유 명령어가 사용하는 레지스터 수를 반환한다.
     */
    unsigned get_num_regs() const { return m_warp->get_num_regs(); }
    /*
     * [한국어] dispatch — 모든 오퍼랜드 수집 완료 후 명령어를 OC_EX 파이프라인 레지스터로 이동한다.
     *
     * m_output_register에 명령어를 이동시키고 CU를 해제(m_free = true)한다.
     * 구현은 shader.cc에 있다.
     *
     * 호출 체인:
     *   opndcoll_rfu_t::dispatch_ready_cu() → [dispatch()] → m_output_register에 명령어 삽입
     */
    void dispatch();
    /*
     * [한국어] is_free — 이 CU가 현재 미사용 상태인지 확인한다.
     * @return: m_free가 true이면 이 CU에 새 명령어를 할당할 수 있음
     */
    bool is_free() { return m_free; }

   private:
    bool m_free;
    /* [한국어] 이 CU의 사용 가능 여부.
     * 설정자: allocate()에서 false로, dispatch()에서 true로 변경.
     * 읽는 자: is_free()로 allocate_cu()가 빈 CU 탐색, find_ready()도 참조.
     * 값 범위: true(빈 슬롯) / false(명령어 보유 중).
     * 동기화: 단일 스레드(step() 흐름) 내 접근. */

    unsigned m_cuid;  // collector unit hw id
    /* [한국어] 이 CU의 하드웨어 고유 ID.
     * 전체 CU 배열에서 이 CU를 식별하는 인덱스.
     * 설정자: init()에서 n 파라미터로 설정.
     * 읽는 자: get_id(), dump().
     * 값 범위: 0 ~ (총 CU 수 - 1).
     * 동기화: 읽기 전용. */

    unsigned m_warp_id;
    /* [한국어] 이 CU가 보유한 명령어의 warp 슬롯 ID.
     * 설정자: allocate()에서 m_warp->warp_id()로 설정, 해제 시 -1.
     * 읽는 자: get_warp_id()로 arbiter가 뱅크 번호 계산 시 사용.
     * 값 범위: 0 ~ (max_warps_per_shader-1) 또는 -1(미할당).
     * 동기화: 단일 스레드 접근. */

    warp_inst_t *m_warp;
    /* [한국어] 이 CU가 보유한 명령어 포인터 (ID_OC 레지스터에서 이동된 warp_inst_t).
     * 설정자: allocate()에서 pipeline_reg에서 꺼낸 warp_inst_t로 설정.
     * 읽는 자: get_active_count(), get_active_mask(), get_sp_op(), get_num_operands() 등.
     * 값 범위: 유효한 warp_inst_t 포인터(보유 중) 또는 NULL(미할당).
     * 동기화: 단일 스레드 접근. */

    register_set
        *m_output_register;  // pipeline register to issue to when ready
    /* [한국어] 오퍼랜드 수집 완료 후 명령어를 내보낼 OC_EX_* 파이프라인 레지스터 집합.
     * 설정자: allocate()에서 output_reg 파라미터로 설정.
     * 읽는 자: dispatch()에서 명령어를 이 레지스터로 이동.
     * 값 범위: 유효한 register_set 포인터 (실행 유닛별 OC_EX 레지스터).
     * 동기화: dispatch() 전까지 읽기 전용. */

    op_t *m_src_op;
    /* [한국어] 소스 오퍼랜드 요청 배열 (크기: MAX_REG_OPERANDS * 2, 소스+목적 레지스터 포함).
     * 각 원소가 하나의 레지스터 접근 요청(op_t)이며, arbiter에 등록된다.
     * 설정자: allocate()에서 명령어의 소스 레지스터마다 op_t를 생성해 채움.
     * 읽는 자: arbiter_t::add_read_requests()에서 순회하며 뱅크 요청 추가.
     *          collect_operand()에서 수집 완료 표시 (op_t::reset() 등).
     * 값 범위: 유효한 op_t들 (valid=true) + 빈 슬롯 (valid=false).
     * 동기화: 단일 스레드 접근. */

    std::bitset<MAX_REG_OPERANDS * 2> m_not_ready;
    /* [한국어] 아직 수집되지 않은 소스 오퍼랜드의 비트맵.
     * 비트 i가 1이면 m_src_op[i]가 아직 레지스터 파일에서 읽히지 않았음.
     * 설정자: allocate()에서 소스 레지스터마다 해당 비트를 set().
     *         collect_operand(op)에서 해당 비트를 reset().
     * 읽는 자: ready()에서 none()으로 모든 오퍼랜드 수집 완료 여부 판단.
     * 값 범위: 0 (모두 수집 완료, ready) ~ 111...1 (모두 미수집).
     * 동기화: 단일 스레드 접근. */

    unsigned m_num_banks;
    /* [한국어] 레지스터 파일의 총 뱅크 수 (init()에서 전달받음).
     * 설정자: init()에서 설정.
     * 읽는 자: allocate()에서 register_bank() 계산 시 사용.
     * 동기화: 읽기 전용. */

    opndcoll_rfu_t *m_rfu;
    /* [한국어] 이 CU가 속한 opndcoll_rfu_t 역참조 포인터.
     * 서브코어 모델 관련 정보 접근에 사용된다.
     * 설정자: init()에서 설정.
     * 읽는 자: allocate(), dispatch() 등에서 상위 RFU 정보 참조.
     * 동기화: 읽기 전용. */

    unsigned m_num_banks_per_sched;
    /* [한국어] 서브코어 모델에서 스케줄러당 전용 뱅크 수.
     * 설정자: init()에서 num_banks_per_sched 파라미터로 설정.
     * 읽는 자: allocate()에서 register_bank() 계산 시 사용.
     * 동기화: 읽기 전용. */

    bool m_sub_core_model;
    /* [한국어] 서브코어 모델 활성화 여부.
     * true이면 각 스케줄러(서브코어)가 전용 뱅크 파티션을 사용.
     * 설정자: init()에서 설정 (config에서 가져옴).
     * 읽는 자: allocate()에서 register_bank() 호출 시 분기.
     * 동기화: 읽기 전용. */

    unsigned m_reg_id;  // if sub_core_model enabled, limit regs this cu can r/w
    /* [한국어] 서브코어 모델에서 이 CU가 접근 가능한 레지스터 그룹 ID.
     * 서브코어 모델 활성화 시 각 CU는 특정 레지스터 범위만 읽고 쓸 수 있다.
     * 이 ID를 통해 어떤 뱅크 파티션을 사용할지 결정한다.
     * 설정자: init()에서 reg_id 파라미터로 설정.
     * 읽는 자: get_reg_id(), allocate()에서 뱅크 제한 적용.
     * 동기화: 읽기 전용. */
  };

  /*
   * [한국어] dispatch_unit_t — 준비된 콜렉터 유닛을 실행 유닛으로 디스패치하는 유닛
   *
   * === 클래스 역할 ===
   * 담당하는 CU 집합에서 모든 오퍼랜드 수집이 완료된(ready()) CU를 찾아
   * 해당 CU의 명령어를 OC_EX_* 파이프라인 레지스터로 내보낸다.
   * 라운드 로빈 방식으로 CU를 순환하여, 매번 같은 CU만 서비스되는 것을 방지한다.
   * 서브코어 모델이 활성화된 경우, 마지막으로 디스패치한 CU와 다른 서브코어에 속한
   * CU를 우선 탐색하여 스케줄러 간 공정성을 보장한다.
   *
   * === 아키텍처에서의 위치 ===
   * opndcoll_rfu_t::m_dispatch_units 벡터에 실행 유닛별로 존재한다.
   * dispatch_ready_cu()에서 매 사이클 모든 dispatch_unit_t를 순회한다.
   *
   * === 타 모듈과의 연결 ===
   * - 데이터: m_collector_units 벡터 (공유, 소유권 없음)
   * - 호출자: opndcoll_rfu_t::dispatch_ready_cu()
   * - 결과: collector_unit_t::dispatch() → OC_EX_* 레지스터
   */
  class dispatch_unit_t {
   public:
    /*
     * [한국어] dispatch_unit_t 생성자 — 담당할 CU 집합 포인터로 초기화한다.
     *
     * @param cus: 이 디스패치 유닛이 담당하는 collector_unit_t 벡터의 포인터
     *
     * m_last_cu = 0, m_next_cu = 0으로 라운드 로빈 상태를 초기화한다.
     *
     * 호출 체인:
     *   opndcoll_rfu_t::add_cu_set() → [dispatch_unit_t(cus)]
     */
    dispatch_unit_t(std::vector<collector_unit_t> *cus) {
      m_last_cu = 0;                     // [한국어] 라운드 로빈 시작점 — 첫 사이클에는 0번 CU부터 탐색
      m_collector_units = cus;           // [한국어] 담당 CU 집합 포인터 저장
      m_num_collectors = (*cus).size();  // [한국어] CU 총 개수 캐시
      m_next_cu = 0;                     // [한국어] 초기화용 변수 (현재는 m_last_cu로 대체됨)
    }
    /*
     * [한국어] init — 서브코어 모델 여부와 스케줄러 수로 디스패치 유닛을 초기화한다.
     *
     * @param sub_core_model: 서브코어 모델 활성화 여부
     * @param num_warp_scheds: SM 내 warp 스케줄러 수 (서브코어 수)
     *
     * 호출 체인:
     *   opndcoll_rfu_t::init() → [dispatch_unit_t::init()]
     */
    void init(bool sub_core_model, unsigned num_warp_scheds) {
      m_sub_core_model = sub_core_model;   // [한국어] 서브코어 모델 활성화 여부 저장
      m_num_warp_scheds = num_warp_scheds; // [한국어] 스케줄러 수 저장 (CU를 서브코어로 분할하는 단위)
    }

    /*
     * [한국어] find_ready — 라운드 로빈으로 준비된 CU를 탐색하여 반환한다.
     *
     * @return: 준비된 CU 포인터, 없으면 NULL
     *
     * 서브코어 모델에서는 마지막 디스패치 CU와 다른 서브코어의 CU부터 탐색을 시작한다.
     * (같은 서브코어 CU를 연속으로 서비스하면 특정 스케줄러에 치우칠 수 있으므로)
     * 준비된 CU를 찾으면 m_last_cu를 갱신하고 해당 CU 포인터를 반환한다.
     *
     * 호출 체인:
     *   opndcoll_rfu_t::dispatch_ready_cu() → [find_ready()] → collector_unit_t::ready() 확인
     */
    collector_unit_t *find_ready() {
      // With sub-core enabled round robin starts with the next cu assigned to a
      // different sub-core than the one that dispatched last
      unsigned cusPerSched = m_num_collectors / m_num_warp_scheds; // [한국어] 서브코어당 CU 수 계산
      unsigned rr_increment =
          m_sub_core_model ? cusPerSched - (m_last_cu % cusPerSched) : 1; // [한국어] 서브코어 모델: 다음 서브코어 첫 CU로 건너뜀, 일반: 1씩 증가
      for (unsigned n = 0; n < m_num_collectors; n++) { // [한국어] 전체 CU 순환 탐색
        unsigned c = (m_last_cu + n + rr_increment) % m_num_collectors; // [한국어] RR 탐색 인덱스 계산
        if ((*m_collector_units)[c].ready()) { // [한국어] 이 CU가 모든 오퍼랜드 수집 완료했는지 확인
          m_last_cu = c;             // [한국어] 마지막 서비스 CU 갱신 (다음 사이클 RR 시작점)
          return &((*m_collector_units)[c]); // [한국어] 준비된 CU 포인터 반환
        }
      }
      return NULL; // [한국어] 이번 사이클에 준비된 CU 없음
    }

   private:
    unsigned m_num_collectors;
    /* [한국어] 이 디스패치 유닛이 담당하는 CU의 총 수.
     * 설정자: 생성자에서 cus->size()로 초기화.
     * 읽는 자: find_ready()에서 RR 순환 범위 결정.
     * 동기화: 읽기 전용. */

    std::vector<collector_unit_t> *m_collector_units;
    /* [한국어] 담당하는 CU 집합 벡터의 포인터.
     * 소유권 없음 — m_cus 맵의 벡터를 가리킨다.
     * 설정자: 생성자에서 설정.
     * 읽는 자: find_ready()에서 각 CU의 ready() 상태 확인.
     * 동기화: opndcoll_rfu_t::step() 내 단일 흐름 접근. */

    unsigned m_last_cu;  // dispatch ready cu's rr
    /* [한국어] 마지막으로 디스패치한 CU의 인덱스 (라운드 로빈 상태).
     * 설정자: find_ready()에서 ready CU 발견 시 갱신.
     * 읽는 자: find_ready()에서 다음 탐색 시작점 결정.
     * 값 범위: 0 ~ m_num_collectors-1.
     * 동기화: 단일 스레드 접근. */

    unsigned m_next_cu;  // for initialization
    /* [한국어] 초기화 단계에서 사용하는 다음 CU 인덱스.
     * 현재 구현에서는 m_last_cu로 대부분 대체되었으나 남아있는 필드.
     * 동기화: 읽기 전용 (초기화 후 변경 없음). */

    bool m_sub_core_model;
    /* [한국어] 서브코어 모델 활성화 여부.
     * true이면 find_ready()에서 서브코어 경계를 고려한 RR 시작점 계산.
     * 설정자: init()에서 설정.
     * 동기화: 읽기 전용. */

    unsigned m_num_warp_scheds;
    /* [한국어] SM 내 워프 스케줄러(서브코어) 수.
     * cusPerSched = m_num_collectors / m_num_warp_scheds 계산에 사용.
     * 설정자: init()에서 설정.
     * 동기화: 읽기 전용. */
  };

  // opndcoll_rfu_t data members
  bool m_initialized;
  /* [한국어] opndcoll_rfu_t 초기화 완료 여부.
   * 설정자: init()에서 true로 변경.
   * 읽는 자: (현재 명시적 사용 없으나 디버그 목적으로 유지).
   * 동기화: 읽기 전용 (init() 이후). */

  unsigned m_num_collector_sets;
  /* [한국어] 등록된 CU 집합(collector set)의 수.
   * 각 실행 유닛 종류(SP/DP/SFU/MEM 등)에 대응하는 CU 집합의 수.
   * 설정자: add_cu_set() 호출마다 증가.
   * 읽는 자: init() 등에서 전체 집합 수 확인.
   * 동기화: add_cu_set() / init() 이후 읽기 전용. */

  // unsigned m_num_collectors; // [한국어] 주석 처리됨 — 현재는 m_cu.size()로 대체
  unsigned m_num_banks;
  /* [한국어] 레지스터 파일의 총 뱅크 수.
   * 설정자: init()에서 num_banks 파라미터로 설정.
   * 읽는 자: writeback(), allocate_reads(), arbiter_t::init() 등에서 사용.
   * 값 범위: gpgpusim.config의 -gpgpu_num_reg_banks 값.
   * 동기화: 읽기 전용. */

  unsigned m_warp_size;
  /* [한국어] 워프의 크기 (스레드 수, 보통 32).
   * 설정자: init()에서 shader 설정으로부터 가져옴.
   * 읽는 자: 오퍼랜드 수집 관련 계산에 사용.
   * 동기화: 읽기 전용. */

  std::vector<collector_unit_t *> m_cu;
  /* [한국어] 전체 CU 포인터 목록 (모든 CU 집합의 CU를 평탄화한 벡터).
   * m_cus 맵에 저장된 vector<collector_unit_t>의 원소들을 가리키는 포인터들.
   * 설정자: add_cu_set()에서 각 집합의 CU들을 push_back으로 추가.
   * 읽는 자: dump()에서 전체 CU 순회.
   * 값 범위: 전체 CU 수만큼의 유효한 포인터들.
   * 동기화: init() 이후 읽기 전용. */

  arbiter_t m_arbiter;
  /* [한국어] 레지스터 파일 뱅크 접근 중재 아비터.
   * 모든 CU의 읽기 요청과 writeback의 쓰기 요청을 중재한다.
   * 설정자: init()에서 m_arbiter.init() 호출.
   * 읽는 자: allocate_reads(), writeback(), dump().
   * 동기화: step() 내 단일 스레드 접근. */

  unsigned m_num_banks_per_sched;
  /* [한국어] 서브코어 모델에서 스케줄러당 전용 뱅크 수.
   * m_num_banks / m_num_warp_scheds 로 계산된다.
   * 설정자: init()에서 설정.
   * 읽는 자: CU 초기화 시 num_banks_per_sched 파라미터로 전달.
   * 동기화: 읽기 전용. */

  unsigned m_num_warp_scheds;
  /* [한국어] SM 내 워프 스케줄러(서브코어) 수.
   * 설정자: init()에서 shader 설정으로부터 가져옴.
   * 읽는 자: dispatch_unit_t::init(), arbiter_t 초기화에 사용.
   * 동기화: 읽기 전용. */

  bool sub_core_model;
  /* [한국어] 서브코어 모델 활성화 여부.
   * true이면 각 스케줄러가 전용 CU 집합과 뱅크 파티션을 갖는다.
   * 설정자: init()에서 shader 설정으로부터 가져옴.
   * 읽는 자: allocate_cu(), dispatch_ready_cu(), CU/dispatch_unit 초기화.
   * 동기화: 읽기 전용. */

  // unsigned m_num_ports;
  // std::vector<warp_inst_t**> m_input;
  // std::vector<warp_inst_t**> m_output;
  // std::vector<unsigned> m_num_collector_units;
  // warp_inst_t **m_alu_port;
  // [한국어] 위 필드들은 구 설계의 잔재 — 현재 add_port() / input_port_t 구조로 대체됨

  std::vector<input_port_t> m_in_ports;
  /* [한국어] 모든 입력 포트의 목록 (실행 유닛 경로별 ID_OC/OC_EX 레지스터 연결).
   * 설정자: add_port()에서 input_port_t를 push_back.
   * 읽는 자: step() → allocate_cu()에서 각 포트 순회.
   * 값 범위: 실행 유닛 종류 수만큼의 input_port_t.
   * 동기화: init() 이후 읽기 전용. */

  typedef std::map<unsigned /* collector set */,
                   std::vector<collector_unit_t> /*collector sets*/>
      cu_sets_t; // [한국어] CU 집합 ID → collector_unit_t 벡터 맵 타입 별칭
  cu_sets_t m_cus;
  /* [한국어] CU 집합 ID를 키로 하는 collector_unit_t 벡터 맵.
   * 각 실행 유닛 종류별 CU들을 집합으로 관리한다.
   * 설정자: add_cu_set()에서 새 집합을 m_cus에 삽입.
   * 읽는 자: add_port()에서 cu_sets 연결, m_cu 포인터 수집에 사용.
   * 동기화: init() 이후 vector 원소는 읽기 전용 (포인터만 m_cu에 복사). */

  std::vector<dispatch_unit_t> m_dispatch_units;
  /* [한국어] 모든 디스패치 유닛의 목록 (CU 집합별로 대응).
   * 설정자: add_cu_set()에서 각 CU 집합에 대응하는 dispatch_unit_t를 push_back.
   * 읽는 자: dispatch_ready_cu()에서 매 사이클 순회.
   * 동기화: init() 이후 읽기 전용. */

  // typedef std::map<warp_inst_t**/*port*/,dispatch_unit_t> port_to_du_t;
  // port_to_du_t                     m_dispatch_units;
  // std::map<warp_inst_t**,std::list<collector_unit_t*> > m_free_cu;
  // [한국어] 위 타입들은 구 설계 잔재 — 현재 구조로 리팩터링됨

  shader_core_ctx *m_shader;
  /* [한국어] 이 오퍼랜드 컬렉터가 속한 SM (shader_core_ctx) 포인터.
   * 설정자: init()에서 설정.
   * 읽는 자: shader_core() 반환, dump()에서 SM 정보 접근.
   * 동기화: 읽기 전용. */
};

/*
 * [한국어] barrier_set_t — CUDA __syncthreads() 배리어 추적 구조체
 *
 * === 클래스 역할 ===
 * SM 내에서 실행 중인 CTA(Cooperative Thread Array, 스레드 블록)들의
 * __syncthreads() 배리어 도달 상태를 추적한다.
 * 각 CTA는 여러 개의 배리어 ID(named barrier)를 사용할 수 있으며,
 * CTA에 속한 모든 워프가 배리어에 도달해야만 해제(release)된다.
 * 배리어에 도달하지 못한 워프들은 m_warp_at_barrier에 등록되어
 * warp_waiting_at_barrier()가 true를 반환, 스케줄러가 해당 워프를 이슈하지 않는다.
 *
 * === 아키텍처에서의 위치 ===
 * shader_core_ctx::m_barriers 멤버로 존재하며, SM당 하나이다.
 * 기능 시뮬레이션(cuda-sim)과 타이밍 시뮬레이션(gpgpu-sim)이 모두 접근한다.
 *
 * === 타 모듈과의 연결 ===
 * - 갱신자: warp_reaches_barrier() — ldst_unit에서 배리어 명령어 실행 시 호출
 * - 해제자: broadcast_barrier_reduction() — 마지막 워프 도달 시 전체 워프 해제
 * - 조회자: scheduler_unit에서 warp_waiting_at_barrier()로 이슈 차단 여부 확인
 *
 * === 주요 함수 요약 ===
 * - allocate_barrier(): CTA 할당 시 해당 CTA의 워프 집합 등록
 * - warp_reaches_barrier(): 개별 워프의 배리어 도달 처리
 * - warp_exit(): CTA 종료 워프 제거
 * - warp_waiting_at_barrier(): 스케줄러의 워프 이슈 가능 여부 확인
 */
class barrier_set_t {
 public:
  /*
   * [한국어] barrier_set_t 생성자 — 최대 CTA/워프/배리어 수 설정으로 초기화한다.
   *
   * @param shader: 소속 SM (shader_core_ctx*)
   * @param max_warps_per_core: SM당 최대 warp 슬롯 수
   * @param max_cta_per_core: SM당 최대 동시 CTA 수
   * @param max_barriers_per_cta: CTA당 최대 named barrier 수
   * @param warp_size: 워프 크기 (보통 32)
   *
   * 호출 체인:
   *   shader_core_ctx 생성자 → [barrier_set_t(...)]
   */
  barrier_set_t(shader_core_ctx *shader, unsigned max_warps_per_core,
                unsigned max_cta_per_core, unsigned max_barriers_per_cta,
                unsigned warp_size);

  /*
   * [한국어] allocate_barrier — CTA 할당 시 해당 CTA의 배리어 구조를 초기화한다.
   *
   * @param cta_id: 새로 할당된 CTA의 ID
   * @param warps: 이 CTA에 속하는 warp ID들의 집합 (warp_set_t)
   *
   * m_cta_to_warps에 CTA ID → warps 매핑을 추가하고,
   * m_warp_active에 이 워프들을 활성 상태로 등록한다.
   *
   * 호출 체인:
   *   shader_core_ctx::issue_block2core() → [allocate_barrier()]
   */
  // during cta allocation
  void allocate_barrier(unsigned cta_id, warp_set_t warps);

  /*
   * [한국어] deallocate_barrier — CTA 해제 시 배리어 상태를 정리한다.
   *
   * @param cta_id: 해제할 CTA의 ID
   *
   * m_cta_to_warps에서 해당 CTA를 제거하고, m_warp_active에서 관련 워프를 제거한다.
   *
   * 호출 체인:
   *   shader_core_ctx::register_cta_thread_exit() → [deallocate_barrier()]
   */
  // during cta deallocation
  void deallocate_barrier(unsigned cta_id);

  typedef std::map<unsigned, warp_set_t> cta_to_warp_t; // [한국어] CTA ID → 소속 warp 집합 맵 타입
  typedef std::map<unsigned, warp_set_t>
      bar_id_to_warp_t; /*set of warps reached a specific barrier id*/
  // [한국어] bar_id_to_warp_t: 배리어 ID → 해당 배리어에 도달한 warp 집합 맵 타입

  /*
   * [한국어] warp_reaches_barrier — 특정 워프가 배리어에 도달했음을 기록한다.
   *
   * @param cta_id: 이 워프가 속한 CTA ID
   * @param warp_id: 배리어에 도달한 워프의 슬롯 ID
   * @param inst: 배리어 명령어 (bar.sync 등)
   *
   * m_warp_at_barrier에 이 워프를 추가하고, m_bar_id_to_warps[bar_id]에 등록한다.
   * CTA의 모든 활성 워프가 이 배리어에 도달하면 broadcast_barrier_reduction()을 호출해 해제한다.
   *
   * 호출 체인:
   *   ldst_unit::memory_cycle() (배리어 명령어 처리) → [warp_reaches_barrier()]
   */
  // individual warp hits barrier
  void warp_reaches_barrier(unsigned cta_id, unsigned warp_id,
                            warp_inst_t *inst);

  /*
   * [한국어] warp_exit — 실행이 완료된 워프를 활성 집합에서 제거한다.
   *
   * @param warp_id: 종료된 워프의 슬롯 ID
   *
   * m_warp_active에서 해당 워프를 제거한다.
   * CTA 내 마지막 워프가 종료되면 해당 CTA의 배리어가 자동으로 해제될 수 있다.
   *
   * 호출 체인:
   *   shader_core_ctx::warp_exit() → [warp_exit()]
   */
  // warp reaches exit
  void warp_exit(unsigned warp_id);

  /*
   * [한국어] warp_waiting_at_barrier — 특정 워프가 현재 배리어에서 대기 중인지 확인한다.
   *
   * @param warp_id: 확인할 워프의 슬롯 ID
   * @return: m_warp_at_barrier에 이 워프가 있으면 true (이슈 차단)
   *
   * 스케줄러가 매 사이클 이슈 전 이 함수를 호출하여 배리어 대기 워프를 건너뛴다.
   *
   * 호출 체인:
   *   scheduler_unit::cycle() → shd_warp_t::waiting() → [warp_waiting_at_barrier()]
   */
  // assertions
  bool warp_waiting_at_barrier(unsigned warp_id) const;

  /*
   * [한국어] dump — 배리어 상태를 stdout에 출력한다 (디버깅용).
   */
  // debug
  void dump();

 private:
  unsigned m_max_cta_per_core;
  /* [한국어] SM에서 동시 실행 가능한 최대 CTA 수.
   * 설정자: 생성자에서 파라미터로 설정.
   * 읽는 자: allocate_barrier()에서 경계 검사.
   * 동기화: 읽기 전용. */

  unsigned m_max_warps_per_core;
  /* [한국어] SM의 최대 warp 슬롯 수.
   * 설정자: 생성자에서 파라미터로 설정.
   * 읽는 자: warp_set_t 크기 결정에 사용.
   * 동기화: 읽기 전용. */

  unsigned m_max_barriers_per_cta;
  /* [한국어] CTA당 최대 named barrier 수.
   * CUDA에서 bar.sync 명령어로 특정 배리어 ID를 지정할 수 있다.
   * 설정자: 생성자에서 파라미터로 설정.
   * 동기화: 읽기 전용. */

  unsigned m_warp_size;
  /* [한국어] 워프 크기 (스레드 수, 보통 32).
   * 설정자: 생성자에서 파라미터로 설정.
   * 동기화: 읽기 전용. */

  cta_to_warp_t m_cta_to_warps;
  /* [한국어] CTA ID → 이 CTA에 속하는 활성 warp ID 집합 매핑.
   * 설정자: allocate_barrier()에서 CTA 할당 시 추가, deallocate_barrier()에서 제거.
   * 읽는 자: warp_reaches_barrier()에서 해당 CTA의 warp 집합 조회.
   * 동기화: 단일 스레드(cycle()) 내 접근. */

  bar_id_to_warp_t m_bar_id_to_warps;
  /* [한국어] 배리어 ID → 해당 배리어에 도달한 warp ID 집합 매핑.
   * 설정자: warp_reaches_barrier()에서 워프가 배리어에 도달할 때 추가.
   * 읽는 자: 모든 워프 도달 여부 확인 시 사용.
   * 동기화: 단일 스레드 접근. */

  warp_set_t m_warp_active;
  /* [한국어] 현재 SM에서 활성 상태인 warp ID들의 비트셋.
   * 설정자: allocate_barrier()에서 set, warp_exit()에서 clear.
   * 읽는 자: warp_reaches_barrier()에서 CTA 내 활성 warp 수 계산.
   * 동기화: 단일 스레드 접근. */

  warp_set_t m_warp_at_barrier;
  /* [한국어] 현재 배리어에서 대기 중인 warp ID들의 비트셋.
   * 설정자: warp_reaches_barrier()에서 set, 배리어 해제 시 clear.
   * 읽는 자: warp_waiting_at_barrier()에서 스케줄러의 이슈 차단 여부 확인.
   * 동기화: 단일 스레드 접근. */

  shader_core_ctx *m_shader;
  /* [한국어] 소속 SM 역참조 포인터.
   * 설정자: 생성자에서 설정.
   * 읽는 자: broadcast_barrier_reduction() 호출 시 사용.
   * 동기화: 읽기 전용. */
};

/*
 * [한국어] insn_latency_info — 명령어 PC와 파이프라인 레이턴시 기록 구조체
 *
 * === 구조체 역할 ===
 * 특정 PC 위치의 명령어가 파이프라인에서 얼마나 오래 걸리는지를 캐싱하여,
 * 동일 PC의 명령어를 재실행할 때 레이턴시 계산을 빠르게 수행한다.
 * ldst_unit이나 타이밍 모델에서 스톨 계산에 활용된다.
 *
 * === 타 모듈과의 연결 ===
 * - 생성자: 파이프라인 레이턴시 계산 로직에서 캐시 항목으로 생성
 * - 사용자: 레이턴시 조회 시 PC로 캐시 히트 확인
 */
struct insn_latency_info {
  unsigned pc;
  /* [한국어] 이 레이턴시 정보가 대응하는 명령어의 프로그램 카운터.
   * 설정자: 레이턴시 정보 생성 시.
   * 읽는 자: 레이턴시 조회 시 캐시 키로 사용.
   * 값 범위: 유효한 PTX/SASS 명령어 주소. */

  unsigned long latency;
  /* [한국어] 이 명령어의 파이프라인 레이턴시 (사이클 단위).
   * 설정자: 명령어 레이턴시 계산 시.
   * 읽는 자: 파이프라인 스톨 계산 시.
   * 값 범위: 1 이상의 양수. */
};

/*
 * [한국어] ifetch_buffer_t — 명령어 패치 요청 버퍼 (Instruction Fetch Buffer)
 *
 * === 구조체 역할 ===
 * fetch() 단계에서 L1 명령어 캐시(m_L1I)에 발행한 명령어 패치 요청의 정보를 보관한다.
 * L1I가 응답하기 전까지 이 버퍼에 펜딩 요청이 남아있으며, m_valid = true이면
 * 이미 패치 요청이 발행된 상태라 새 요청을 발행하지 않는다.
 * 응답이 오면(accept_fetch_response()) m_valid = false로 리셋하여 다음 패치를 허용한다.
 *
 * === 아키텍처에서의 위치 ===
 * shader_core_ctx::m_inst_fetch_buffer 멤버로 SM당 하나 존재한다.
 * 파이프라인 IF(Instruction Fetch) 단계에 해당한다.
 * IF → ID → OC → EX → WB 파이프라인의 첫 단계이다.
 *
 * === 타 모듈과의 연결 ===
 * - 설정자: shader_core_ctx::fetch()에서 L1I에 요청 발행 시 채움
 * - 클리어: shader_core_ctx::accept_fetch_response()에서 응답 수신 시 m_valid = false
 * - 읽는 자: fetch()에서 이미 펜딩 요청이 있는지 m_valid로 확인
 */
struct ifetch_buffer_t {
  /*
   * [한국어] ifetch_buffer_t 기본 생성자 — 유효하지 않은(빈) 상태로 초기화한다.
   * m_valid = false이면 현재 펜딩 패치 요청 없음.
   */
  ifetch_buffer_t() { m_valid = false; } // [한국어] 초기화: 펜딩 요청 없음

  /*
   * [한국어] ifetch_buffer_t 생성자 — 새 패치 요청 정보로 초기화한다.
   *
   * @param pc: 패치할 명령어의 프로그램 카운터
   * @param nbytes: 패치할 바이트 수 (캐시 라인 크기에 정렬)
   * @param warp_id: 이 패치 요청을 발행한 워프의 슬롯 ID
   *
   * 호출 체인:
   *   shader_core_ctx::fetch() → m_L1I->access() 후 → [ifetch_buffer_t(pc, nbytes, warp_id)]
   */
  ifetch_buffer_t(address_type pc, unsigned nbytes, unsigned warp_id) {
    m_valid = true;      // [한국어] 펜딩 요청 있음 표시
    m_pc = pc;           // [한국어] 패치할 PC 저장
    m_nbytes = nbytes;   // [한국어] 패치 크기 저장
    m_warp_id = warp_id; // [한국어] 요청 워프 ID 저장
  }

  bool m_valid;
  /* [한국어] 현재 유효한 펜딩 패치 요청이 있는지 여부.
   * 설정자: 생성자(true), accept_fetch_response()에서 false로 클리어.
   * 읽는 자: fetch()에서 새 요청 발행 전 체크 (true이면 이미 요청 중이므로 대기).
   * 값 범위: true(요청 중) / false(빈 상태).
   * 동기화: 단일 스레드(cycle() 흐름) 내 접근. */

  address_type m_pc;
  /* [한국어] 패치 중인 명령어의 프로그램 카운터.
   * 설정자: 패치 요청 발행 시.
   * 읽는 자: accept_fetch_response()에서 어떤 PC의 응답인지 확인.
   * 동기화: m_valid = true인 동안 읽기 전용. */

  unsigned m_nbytes;
  /* [한국어] 패치 요청한 바이트 수 (캐시 라인 크기 이하).
   * 설정자: 패치 요청 시 명령어 크기에 따라 설정.
   * 읽는 자: 디버그 및 통계 목적.
   * 동기화: m_valid = true인 동안 읽기 전용. */

  unsigned m_warp_id;
  /* [한국어] 이 패치 요청을 발행한 워프의 하드웨어 슬롯 ID.
   * 설정자: 패치 요청 시.
   * 읽는 자: accept_fetch_response()에서 어떤 워프의 응답인지 식별.
   * 값 범위: 0 ~ (max_warps_per_shader-1).
   * 동기화: m_valid = true인 동안 읽기 전용. */
};

class shader_core_config; // [한국어] 전방 선언 — simd_function_unit이 shader_core_config 포인터를 멤버로 가짐

/*
 * [한국어] simd_function_unit — SM 내 모든 SIMD 실행 유닛의 추상 기반 클래스
 *
 * === 클래스 역할 ===
 * SP(Single Precision), DP(Double Precision), SFU(Special Function Unit),
 * INT(Integer ALU), Tensor Core, ldst_unit 등 모든 실행 유닛의 공통 인터페이스를 정의한다.
 * 핵심 가상 함수: cycle() (매 사이클 파이프라인 진행), issue() (명령어 이슈),
 * active_lanes_in_pipeline() (AccelWattch 전력 카운터 갱신)
 *
 * m_dispatch_reg: OC_EX_* 레지스터에서 이 유닛으로 발행된 명령어를 보관하는 단일 슬롯
 * occupied: 비트셋으로 레이턴시 사이클을 추적 — 0번 비트가 지금 사이클, 1번이 다음 사이클 등
 *           (inst.latency 비트가 세트되어 있으면 해당 레이턴시 후 결과 버스 충돌 방지)
 *
 * === 아키텍처에서의 위치 ===
 * shader_core_ctx::m_fu 벡터에 등록된다. execute() 단계에서 모든 유닛의 cycle()이 호출된다.
 * 파이프라인 단계: OC_EX_* → [실행 유닛] → EX_WB
 *
 * === 타 모듈과의 연결 ===
 * - 소유자: shader_core_ctx::m_fu
 * - 발행자: opndcoll_rfu_t → OC_EX 레지스터 → issue()
 * - 결과: EX_WB 파이프라인 레지스터 → writeback()
 */
class simd_function_unit {
 public:
  /*
   * [한국어] simd_function_unit 생성자 — m_dispatch_reg를 할당하고 occupied를 초기화한다.
   *
   * @param config: SM 설정 (파이프라인 깊이 등 유닛 파라미터 결정에 사용)
   *
   * 호출 체인:
   *   shader_core_ctx::create_exec_pipeline() → 각 유닛 생성자 → [simd_function_unit()]
   */
  simd_function_unit(const shader_core_config *config);
  /*
   * [한국어] simd_function_unit 소멸자 — m_dispatch_reg를 해제한다.
   */
  ~simd_function_unit() { delete m_dispatch_reg; } // [한국어] dispatch 레지스터 동적 메모리 해제

  /*
   * [한국어] issue — OC_EX 파이프라인 레지스터에서 명령어를 꺼내 m_dispatch_reg에 이동한다.
   *
   * @param source_reg: 명령어가 대기 중인 OC_EX_* 파이프라인 레지스터 집합
   *
   * source_reg에서 warp_inst_t를 m_dispatch_reg로 이동시킨다.
   * 서브클래스에서 오버라이드하여 occupied 비트 세트 등 추가 동작을 수행한다.
   *
   * 호출 체인:
   *   shader_core_ctx::execute() → [issue(source_reg)] → move_warp(m_dispatch_reg, source_reg)
   */
  // modifiers
  virtual void issue(register_set &source_reg);
  /*
   * [한국어] cycle — 이 유닛의 1 사이클 파이프라인 동작을 수행한다. (순수 가상 함수)
   *
   * 서브클래스(pipelined_simd_unit 등)에서 구현한다.
   * m_dispatch_reg에서 m_pipeline_reg로 명령어를 이동시키고
   * 파이프라인 끝에 도달한 명령어를 m_result_port(EX_WB)로 내보낸다.
   *
   * 호출 체인:
   *   shader_core_ctx::execute() → [cycle()]
   */
  virtual void cycle() = 0;
  /*
   * [한국어] active_lanes_in_pipeline — 이 유닛의 활성 레인 수를 AccelWattch 카운터에 반영한다. (순수 가상 함수)
   *
   * 파이프라인 내 각 단계에서 활성 스레드(레인) 수를 집계해 전력 모델에 전달한다.
   *
   * 호출 체인:
   *   shader_core_ctx::execute() → [active_lanes_in_pipeline()]
   */
  virtual void active_lanes_in_pipeline() = 0;

  /*
   * [한국어] clock_multiplier — 이 유닛의 클럭 배율을 반환한다.
   *
   * @return: 기본값 1 (코어 클럭과 같은 속도). ldst_unit은 더 느린 배율 가능.
   */
  // accessors
  virtual unsigned clock_multiplier() const { return 1; }
  /*
   * [한국어] can_issue — 이 사이클에 이 유닛이 주어진 명령어를 받을 수 있는지 확인한다.
   *
   * @param inst: 이슈하려는 warp_inst_t
   * @return: m_dispatch_reg가 비어있고, 해당 레이턴시 사이클의 결과 버스가 비어있으면 true
   *
   * occupied.test(inst.latency)로 레이턴시 후 결과 버스 충돌 여부를 확인한다.
   * 충돌이 있으면 스코어보드 해저드와 별개로 이슈를 차단한다.
   *
   * 호출 체인:
   *   scheduler_unit::cycle() → [can_issue(inst)] → 이슈 가능 여부 판단
   */
  virtual bool can_issue(const warp_inst_t &inst) const {
    return m_dispatch_reg->empty() && !occupied.test(inst.latency); // [한국어] dispatch 슬롯이 비어있고 해당 레이턴시 결과 버스도 여유있어야 이슈 가능
  }
  /*
   * [한국어] is_issue_partitioned — 이 유닛이 서브코어 파티셔닝을 지원하는지 반환한다. (순수 가상 함수)
   * SP/DP/SFU 등 pipelined 유닛은 true, ldst_unit은 false를 반환한다.
   */
  virtual bool is_issue_partitioned() = 0;
  /*
   * [한국어] get_issue_reg_id — 서브코어 모델에서 이 유닛이 허용하는 발행 레지스터 집합 ID를 반환한다. (순수 가상 함수)
   */
  virtual unsigned get_issue_reg_id() = 0;
  /*
   * [한국어] stallable — 이 유닛이 스톨 가능한지 반환한다. (순수 가상 함수)
   * pipelined 유닛은 false (비스톨), ldst_unit은 true (스톨 가능).
   * 스톨 가능 유닛은 m_fu 벡터의 마지막에 배치된다.
   */
  virtual bool stallable() const = 0;
  /*
   * [한국어] print — 이 유닛의 이름과 dispatch 레지스터 상태를 fp에 출력한다 (디버깅용).
   */
  virtual void print(FILE *fp) const {
    fprintf(fp, "%s dispatch= ", m_name.c_str()); // [한국어] 유닛 이름 출력
    m_dispatch_reg->print(fp);                    // [한국어] dispatch 레지스터 내용 출력
  }
  /*
   * [한국어] get_name — 이 유닛의 이름 문자열을 반환한다.
   * @return: m_name.c_str() (예: "SP", "DP", "SFU", "INT", "TC", "MEM")
   */
  const char *get_name() { return m_name.c_str(); }

 protected:
  std::string m_name;
  /* [한국어] 이 실행 유닛의 이름 (예: "SP", "DP", "SFU", "INT", "TC", "MEM").
   * 설정자: 각 서브클래스 생성자에서 설정.
   * 읽는 자: get_name(), print().
   * 동기화: 읽기 전용. */

  const shader_core_config *m_config;
  /* [한국어] SM 설정 포인터 (파이프라인 깊이, 레이턴시 등 파라미터 접근).
   * 설정자: 생성자에서 설정.
   * 동기화: 읽기 전용. */

  warp_inst_t *m_dispatch_reg;
  /* [한국어] OC_EX_* 레지스터에서 이 유닛으로 발행된 명령어를 보관하는 단일 슬롯.
   * 설정자: issue()에서 source_reg로부터 이동.
   * 읽는 자: can_issue() (비어있는지 확인), cycle() (파이프라인으로 이동).
   * 값 범위: 유효한 warp_inst_t 또는 빈 상태.
   * 동기화: 단일 스레드 접근. */

  static const unsigned MAX_ALU_LATENCY = 512; // [한국어] 결과 버스 추적 최대 레이턴시 (512 사이클) — occupied 비트셋 크기
  std::bitset<MAX_ALU_LATENCY> occupied;
  /* [한국어] 결과 버스 충돌 방지를 위한 레이턴시 점유 비트셋.
   * i번 비트가 세트되면 현재+i 사이클에 이 유닛의 결과가 결과 버스를 사용 중임을 의미.
   * 설정자: issue()에서 occupied.set(inst.latency) 호출.
   * 읽는 자: can_issue()에서 occupied.test(inst.latency)로 충돌 확인.
   * 갱신: cycle()에서 매 사이클 occupied >>= 1로 오른쪽 시프트 (시간 경과 반영).
   * 동기화: 단일 스레드 접근. */
};

/*
 * [한국어] pipelined_simd_unit — 다단계 파이프라인 SIMD 실행 유닛 (추상 클래스)
 *
 * === 클래스 역할 ===
 * simd_function_unit을 상속하여 다단계 파이프라인(m_pipeline_reg 배열)을 구현한다.
 * 매 사이클 cycle()이 호출되면 파이프라인 레지스터를 한 단계씩 시프트한다:
 *   dispatch → pipeline[0] → pipeline[1] → ... → pipeline[depth-1] → result_port(EX_WB)
 * 파이프라인이 채워진 상태에서는 매 사이클 결과를 출력할 수 있다 (full throughput).
 *
 * SP, DP, SFU, INT, Tensor Core, specialized_unit이 이 클래스를 상속한다.
 * ldst_unit은 별도로 stallable() = true이므로 이 클래스와 다른 처리가 필요하다.
 *
 * === 아키텍처에서의 위치 ===
 * shader_core_ctx::create_exec_pipeline()에서 생성되어 m_fu 벡터에 등록된다.
 * OC_EX_* → [pipelined_simd_unit 파이프라인] → EX_WB 파이프라인 레지스터
 *
 * === 타 모듈과의 연결 ===
 * - 발행자: opndcoll_rfu_t::dispatch_ready_cu() → OC_EX_* → issue()
 * - 결과: m_result_port(EX_WB) → shader_core_ctx::writeback()
 */
class pipelined_simd_unit : public simd_function_unit {
 public:
  /*
   * [한국어] pipelined_simd_unit 생성자 — 파이프라인 레지스터 배열을 할당하고 초기화한다.
   *
   * @param result_port: 파이프라인 끝에서 결과를 내보낼 EX_WB 파이프라인 레지스터 집합
   * @param config: SM 설정
   * @param max_latency: 파이프라인 깊이 (m_pipeline_depth 결정)
   * @param core: 소속 SM (shader_core_ctx*)
   * @param issue_reg_id: 서브코어 모델에서 이 유닛이 허용하는 발행 레지스터 집합 ID
   *
   * m_pipeline_reg[0..max_latency-1]를 동적 할당하고 각 슬롯을 초기화한다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_exec_pipeline() → [pipelined_simd_unit 생성자]
   */
  pipelined_simd_unit(register_set *result_port,
                      const shader_core_config *config, unsigned max_latency,
                      shader_core_ctx *core, unsigned issue_reg_id);

  /*
   * [한국어] cycle — 파이프라인 레지스터를 한 단계씩 전진시킨다.
   *
   * 파이프라인의 마지막 단계(m_pipeline_reg[m_pipeline_depth-1])에 있는 명령어를
   * m_result_port(EX_WB)로 이동시키고, 나머지 단계를 하나씩 앞으로 시프트한다.
   * occupied 비트셋도 오른쪽으로 1비트 시프트한다.
   *
   * 호출 체인:
   *   shader_core_ctx::execute() → [cycle()]
   */
  // modifiers
  virtual void cycle();
  /*
   * [한국어] issue — 명령어를 m_dispatch_reg로 이동하고 occupied 비트를 세트한다.
   *
   * @param source_reg: OC_EX_* 파이프라인 레지스터 집합
   *
   * 부모의 simd_function_unit::issue()를 호출 후 occupied.set(inst.latency)으로
   * 해당 레이턴시 사이클의 결과 버스를 예약한다.
   *
   * 호출 체인:
   *   shader_core_ctx::execute() → [issue(source_reg)] → simd_function_unit::issue() → occupied.set()
   */
  virtual void issue(register_set &source_reg);
  /*
   * [한국어] get_active_lanes_in_pipeline — 현재 파이프라인에서 진행 중인 활성 레인 수를 반환한다.
   *
   * @return: 파이프라인의 모든 단계에서 활성 스레드 수의 합계
   *
   * AccelWattch 전력 모델에서 유닛별 전력 소비 계산에 사용된다.
   */
  virtual unsigned get_active_lanes_in_pipeline();

  /*
   * [한국어] active_lanes_in_pipeline — AccelWattch 카운터 갱신 (순수 가상 함수).
   * 서브클래스에서 유닛 종류에 맞는 카운터를 증가시킨다.
   */
  virtual void active_lanes_in_pipeline() = 0;

  /*
   * [한국어] stallable — 이 유닛이 스톨 불가능함을 반환한다.
   * @return: 항상 false — pipelined 유닛은 비스톨 (ldst_unit만 stallable)
   */
  // accessors
  virtual bool stallable() const { return false; }
  /*
   * [한국어] can_issue — 이 유닛이 주어진 명령어를 받을 수 있는지 확인한다.
   * 부모 simd_function_unit::can_issue()에 위임한다.
   */
  virtual bool can_issue(const warp_inst_t &inst) const {
    return simd_function_unit::can_issue(inst); // [한국어] dispatch 슬롯 및 occupied 비트 검사
  }
  /*
   * [한국어] is_issue_partitioned — 서브코어 파티셔닝 지원 여부 (순수 가상 함수).
   * SP/DP 등은 true, ldst_unit은 false를 반환한다.
   */
  virtual bool is_issue_partitioned() = 0;
  /*
   * [한국어] get_issue_reg_id — 서브코어 모델에서 이 유닛의 발행 레지스터 집합 ID를 반환한다.
   * @return: m_issue_reg_id
   */
  unsigned get_issue_reg_id() { return m_issue_reg_id; }
  /*
   * [한국어] print — 이 유닛의 dispatch 레지스터와 파이프라인 내 모든 명령어를 fp에 출력한다 (디버깅용).
   */
  virtual void print(FILE *fp) const {
    simd_function_unit::print(fp); // [한국어] 부모의 print (유닛 이름 + dispatch 레지스터)
    for (int s = m_pipeline_depth - 1; s >= 0; s--) { // [한국어] 파이프라인 끝 → 시작 순으로 출력
      if (!m_pipeline_reg[s]->empty()) { // [한국어] 비어있지 않은 단계만 출력
        fprintf(fp, "      %s[%2d] ", m_name.c_str(), s);
        m_pipeline_reg[s]->print(fp); // [한국어] s번째 파이프라인 단계의 명령어 출력
      }
    }
  }

 protected:
  unsigned m_pipeline_depth;
  /* [한국어] 파이프라인 깊이 (단계 수) — 명령어가 issue에서 result까지 걸리는 사이클 수.
   * 설정자: 생성자에서 max_latency 파라미터로 결정.
   * 읽는 자: cycle(), get_active_lanes_in_pipeline(), print().
   * 값 범위: SP 유닛은 보통 1~4, SFU는 더 길 수 있음.
   * 동기화: 읽기 전용. */

  warp_inst_t **m_pipeline_reg;
  /* [한국어] 파이프라인 레지스터 배열 (크기: m_pipeline_depth).
   * m_pipeline_reg[0]이 입력(dispatch에서 이동), m_pipeline_reg[depth-1]이 출력(result_port로).
   * 설정자: 생성자에서 동적 할당, issue()에서 m_dispatch_reg → [0]으로 이동, cycle()에서 시프트.
   * 읽는 자: cycle()에서 시프트, get_active_lanes_in_pipeline()에서 카운팅.
   * 동기화: 단일 스레드 접근. */

  register_set *m_result_port;
  /* [한국어] 파이프라인 완료 결과를 내보낼 EX_WB 파이프라인 레지스터 집합 포인터.
   * 설정자: 생성자에서 result_port 파라미터로 설정.
   * 읽는 자: cycle()에서 파이프라인 끝 명령어를 여기로 이동.
   * 동기화: 읽기 전용 (포인터 자체). */

  class shader_core_ctx *m_core;
  /* [한국어] 소속 SM (shader_core_ctx) 포인터.
   * AccelWattch 카운터 증가 함수 (incsp_stat 등) 호출에 사용.
   * 설정자: 생성자에서 설정.
   * 동기화: 읽기 전용. */

  unsigned m_issue_reg_id;  // if sub_core_model is enabled we can only issue
                            // from a subset of operand collectors
  /* [한국어] 서브코어 모델에서 이 유닛이 허용하는 OC_EX 레지스터 집합 ID.
   * 스케줄러 ID와 연관되어, 특정 스케줄러 담당 CU에서만 명령어를 받을 수 있다.
   * 설정자: 생성자에서 issue_reg_id 파라미터로 설정.
   * 읽는 자: get_issue_reg_id(), 스케줄러의 이슈 가능 여부 판단.
   * 동기화: 읽기 전용. */

  unsigned active_insts_in_pipeline;
  /* [한국어] 현재 파이프라인 내에서 진행 중인 명령어 수.
   * 설정자: issue() 시 증가, cycle()에서 출력 시 감소.
   * 읽는 자: get_active_lanes_in_pipeline() 등에서 활성도 판단.
   * 동기화: 단일 스레드 접근. */
};

/*
 * [한국어] sfu — Special Function Unit (특수 함수 유닛)
 *
 * === 클래스 역할 ===
 * sin, cos, sqrt, rcp(역수), rsqrt(역제곱근) 등 초월 함수를 처리하는 실행 유닛이다.
 * SFU_OP, ALU_SFU_OP, DP_OP (Fermi/GT200의 경우) 명령어를 처리한다.
 * 일반 SP 유닛보다 레이턴시가 길며, SM당 SP 유닛보다 적은 수로 존재한다.
 * (예: Volta에서 SM당 SP 64개, SFU 16개)
 *
 * === 아키텍처에서의 위치 ===
 * shader_core_ctx::create_exec_pipeline()에서 m_fu에 등록된다.
 * OC_EX_SFU → [sfu 파이프라인] → EX_WB
 *
 * === 타 모듈과의 연결 ===
 * - AccelWattch: active_lanes_in_pipeline()에서 m_core->incsfuactivelanes_stat() 호출
 * - 설정: gpgpusim.config의 -gpgpu_num_sfu_units, -gpgpu_sfu_latency
 */
class sfu : public pipelined_simd_unit {
 public:
  /*
   * [한국어] sfu 생성자 — SFU 파이프라인을 초기화한다.
   *
   * gpgpusim.config의 max_sfu_latency 설정으로 파이프라인 깊이가 결정된다.
   *
   * 호출 체인:
   *   shader_core_ctx::create_exec_pipeline() → [sfu(...)]
   */
  sfu(register_set *result_port, const shader_core_config *config,
      shader_core_ctx *core, unsigned issue_reg_id);
  /*
   * [한국어] can_issue — SFU가 처리 가능한 명령어 유형인지 확인한다.
   *
   * @param inst: 이슈하려는 명령어
   * @return: SFU_OP, ALU_SFU_OP, DP_OP(Fermi/GT200)이고 dispatch 슬롯이 비어있으면 true
   *
   * DP_OP는 compute capability <= 2.x (Fermi/GT200)에서만 SFU로 라우팅된다.
   */
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case SFU_OP:      break; // [한국어] SFU 전용 연산 (sin, cos, sqrt, rcp, rsqrt 등)
      case ALU_SFU_OP:  break; // [한국어] SP ALU도 SFU도 처리 가능한 혼합 명령어
      case DP_OP:       break;  // for compute <= 29 (i..e Fermi and GT200)
      default:          return false; // [한국어] 그 외 명령어는 SFU가 처리 불가
    }
    return pipelined_simd_unit::can_issue(inst); // [한국어] dispatch 슬롯 및 occupied 비트 추가 검사
  }
  /*
   * [한국어] active_lanes_in_pipeline — SFU 파이프라인 내 활성 레인 수를 AccelWattch에 보고한다.
   * m_core->incsfuactivelanes_stat() 및 incsfu_stat()을 호출한다.
   */
  virtual void active_lanes_in_pipeline();
  /*
   * [한국어] issue — 명령어를 SFU 파이프라인에 발행하고 전력 카운터를 갱신한다.
   * 부모 issue() 후 AccelWattch SFU 접근 카운터를 증가시킨다.
   */
  virtual void issue(register_set &source_reg);
  /*
   * [한국어] is_issue_partitioned — 서브코어 파티셔닝 지원: SFU는 true를 반환한다.
   */
  bool is_issue_partitioned() { return true; }
};

/*
 * [한국어] dp_unit — Double Precision 부동소수점 유닛 (배정밀도 FP)
 *
 * === 클래스 역할 ===
 * DP_OP 명령어(배정밀도 FP 연산)를 처리하는 실행 유닛이다.
 * SP 유닛보다 레이턴시가 길고 SM당 더 적은 수로 존재한다.
 * (예: Volta에서 SM당 DP 유닛 32개, SP 유닛 64개)
 *
 * === 설정 ===
 * gpgpusim.config: -gpgpu_num_dp_units, -gpgpu_dp_latency (max_dp_latency)
 */
class dp_unit : public pipelined_simd_unit {
 public:
  /*
   * [한국어] dp_unit 생성자 — DP 파이프라인을 초기화한다.
   * max_dp_latency 설정으로 파이프라인 깊이가 결정된다.
   */
  dp_unit(register_set *result_port, const shader_core_config *config,
          shader_core_ctx *core, unsigned issue_reg_id);
  /*
   * [한국어] can_issue — DP_OP 명령어만 처리 가능함을 확인한다.
   * DP_OP 이외의 명령어는 false를 반환한다.
   */
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case DP_OP: break; // [한국어] 배정밀도 FP 연산 (fadd.f64, fmul.f64 등)
      default:    return false; // [한국어] 그 외 명령어는 DP 유닛이 처리 불가
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  /*
   * [한국어] active_lanes_in_pipeline — DP 파이프라인의 활성 레인 수를 AccelWattch에 보고한다.
   */
  virtual void active_lanes_in_pipeline();
  /*
   * [한국어] issue — 명령어를 DP 파이프라인에 발행하고 전력 카운터를 갱신한다.
   */
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

/*
 * [한국어] tensor_core — Tensor Core 유닛 (행렬 곱셈/누산 가속기)
 *
 * === 클래스 역할 ===
 * TENSOR_CORE_OP 명령어(wmma.mma 등, 행렬-행렬 곱 누산)를 처리한다.
 * Volta(V100) 이상의 GPU에서 혼합 정밀도(FP16/BF16/INT8) 행렬 연산을 가속한다.
 * gpgpusim.config의 -gpgpu_tensor_core_avail 옵션으로 활성화/비활성화된다.
 *
 * === 설정 ===
 * gpgpusim.config: -gpgpu_num_tensor_core_units, -gpgpu_tensor_core_latency (max_tensor_core_latency)
 */
class tensor_core : public pipelined_simd_unit {
 public:
  /*
   * [한국어] tensor_core 생성자 — Tensor Core 파이프라인을 초기화한다.
   * max_tensor_core_latency 설정으로 파이프라인 깊이가 결정된다.
   */
  tensor_core(register_set *result_port, const shader_core_config *config,
              shader_core_ctx *core, unsigned issue_reg_id);
  /*
   * [한국어] can_issue — TENSOR_CORE_OP 명령어만 처리 가능함을 확인한다.
   */
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case TENSOR_CORE_OP: break; // [한국어] wmma.mma 등 행렬 곱 누산 명령어
      default:             return false; // [한국어] 그 외 명령어는 Tensor Core 처리 불가
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  /*
   * [한국어] active_lanes_in_pipeline — Tensor Core 파이프라인의 활성 레인 수를 AccelWattch에 보고한다.
   */
  virtual void active_lanes_in_pipeline();
  /*
   * [한국어] issue — 명령어를 Tensor Core 파이프라인에 발행하고 전력 카운터를 갱신한다.
   */
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

/*
 * [한국어] int_unit — 정수 ALU 유닛 (Integer Arithmetic Logic Unit)
 *
 * === 클래스 역할 ===
 * 정수 덧셈, 논리 연산, 비교, 시프트 등 정수 연산 명령어를 처리한다.
 * SFU, LOAD/STORE, 메모리 배리어, SP(FP32), DP(FP64) 명령어를 제외한
 * 나머지 명령어(INT 계열)를 처리한다.
 * Turing 이후 GPU에서 SP와 분리된 전용 INT 유닛이 존재한다.
 *
 * === 설정 ===
 * gpgpusim.config: -gpgpu_num_int_units, -gpgpu_int_latency (max_int_latency)
 */
class int_unit : public pipelined_simd_unit {
 public:
  /*
   * [한국어] int_unit 생성자 — 정수 ALU 파이프라인을 초기화한다.
   */
  int_unit(register_set *result_port, const shader_core_config *config,
           shader_core_ctx *core, unsigned issue_reg_id);
  /*
   * [한국어] can_issue — 정수 ALU가 처리할 수 있는 명령어인지 확인한다.
   *
   * 처리 불가 명령어를 명시적으로 제외하고, 나머지는 pipelined_simd_unit::can_issue()로 검사한다.
   */
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case SFU_OP:              return false; // [한국어] 특수 함수 — SFU 담당
      case LOAD_OP:             return false; // [한국어] 메모리 로드 — ldst_unit 담당
      case TENSOR_CORE_LOAD_OP: return false; // [한국어] Tensor Core 로드 — ldst_unit 담당
      case STORE_OP:            return false; // [한국어] 메모리 스토어 — ldst_unit 담당
      case TENSOR_CORE_STORE_OP:return false; // [한국어] Tensor Core 스토어 — ldst_unit 담당
      case MEMORY_BARRIER_OP:   return false; // [한국어] 메모리 배리어 — ldst_unit 담당
      case SP_OP:               return false; // [한국어] FP32 연산 — SP 유닛 담당
      case DP_OP:               return false; // [한국어] FP64 연산 — DP 유닛 담당
      default:                  break;        // [한국어] 정수 ALU 명령어 — 처리 가능
    }
    return pipelined_simd_unit::can_issue(inst); // [한국어] dispatch 슬롯 및 occupied 비트 검사
  }
  /*
   * [한국어] active_lanes_in_pipeline — 정수 ALU 파이프라인의 활성 레인 수를 AccelWattch에 보고한다.
   */
  virtual void active_lanes_in_pipeline();
  /*
   * [한국어] issue — 명령어를 정수 ALU 파이프라인에 발행하고 전력 카운터를 갱신한다.
   */
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

/*
 * [한국어] sp_unit — Single Precision 부동소수점 유닛 (FP32 ALU)
 *
 * === 클래스 역할 ===
 * SP_OP 및 정수(INT) 명령어를 포함한 단정밀도 FP32 연산을 처리한다.
 * GPU의 CUDA 코어는 주로 SP 유닛으로 모델링되며, SM에서 가장 많이 존재한다.
 * int_unit과 달리 SP_OP도 처리 가능 (SP와 INT가 공유 파이프라인인 구식 아키텍처 지원).
 *
 * === 설정 ===
 * gpgpusim.config: -gpgpu_num_sp_units, -gpgpu_sp_latency (max_sp_latency)
 */
class sp_unit : public pipelined_simd_unit {
 public:
  /*
   * [한국어] sp_unit 생성자 — SP(FP32) 파이프라인을 초기화한다.
   */
  sp_unit(register_set *result_port, const shader_core_config *config,
          shader_core_ctx *core, unsigned issue_reg_id);
  /*
   * [한국어] can_issue — SP 유닛이 처리할 수 있는 명령어인지 확인한다.
   *
   * SFU, 메모리, DP(FP64) 명령어를 제외한 모든 명령어를 처리 가능으로 본다.
   * INT 명령어는 SP 파이프라인에서도 처리 가능 (아키텍처에 따라 공유).
   */
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case SFU_OP:              return false; // [한국어] 특수 함수 — SFU 담당
      case LOAD_OP:             return false; // [한국어] 메모리 로드 — ldst_unit 담당
      case TENSOR_CORE_LOAD_OP: return false; // [한국어] Tensor Core 로드 — ldst_unit 담당
      case STORE_OP:            return false; // [한국어] 메모리 스토어 — ldst_unit 담당
      case TENSOR_CORE_STORE_OP:return false; // [한국어] Tensor Core 스토어 — ldst_unit 담당
      case MEMORY_BARRIER_OP:   return false; // [한국어] 메모리 배리어 — ldst_unit 담당
      case DP_OP:               return false; // [한국어] FP64 연산 — DP 유닛 담당
      default:                  break;        // [한국어] FP32/INT 명령어 — SP 유닛 처리 가능
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  /*
   * [한국어] active_lanes_in_pipeline — SP 파이프라인의 활성 레인 수를 AccelWattch에 보고한다.
   * m_core->incspactivelanes_stat() 및 incsp_stat() 호출.
   */
  virtual void active_lanes_in_pipeline();
  /*
   * [한국어] issue — 명령어를 SP 파이프라인에 발행하고 전력 카운터를 갱신한다.
   */
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

/*
 * [한국어] specialized_unit — 사용자 설정 가능한 특수 목적 실행 유닛
 *
 * === 클래스 역할 ===
 * gpgpusim.config의 -specialized_unit 옵션으로 정의되는 커스텀 실행 유닛이다.
 * 특정 명령어 유형(m_supported_op)만 처리하며, 사용자가 레이턴시와 유닛 수를 설정한다.
 * SPEC_OP1, SPEC_OP2 등 사용자 정의 opcode에 대응하는 유닛을 추가할 때 사용한다.
 *
 * === 설정 ===
 * gpgpusim.config: -specialized_unit "<enabled>,<num_units>,<latency>,<id_oc_width>,<oc_ex_width>,<name>"
 */
class specialized_unit : public pipelined_simd_unit {
 public:
  /*
   * [한국어] specialized_unit 생성자 — 지정된 명령어 유형과 레이턴시로 유닛을 초기화한다.
   *
   * @param supported_op: 이 유닛이 처리하는 명령어 유형 ID
   * @param unit_name: 유닛 이름 문자열 (설정 파일에서 지정)
   * @param latency: 파이프라인 레이턴시 (사이클)
   *
   * 호출 체인:
   *   shader_core_ctx::create_exec_pipeline() → [specialized_unit(...)]
   */
  specialized_unit(register_set *result_port, const shader_core_config *config,
                   shader_core_ctx *core, int supported_op, char *unit_name,
                   unsigned latency, unsigned issue_reg_id);
  /*
   * [한국어] can_issue — 이 유닛이 지원하는 명령어 유형인지 확인한다.
   *
   * m_supported_op와 inst.op가 일치하는 명령어만 처리 가능하다.
   */
  virtual bool can_issue(const warp_inst_t &inst) const {
    if (inst.op != m_supported_op) { // [한국어] 지원하는 명령어 유형이 아니면 거부
      return false;
    }
    return pipelined_simd_unit::can_issue(inst); // [한국어] dispatch 슬롯 및 occupied 비트 추가 검사
  }
  /*
   * [한국어] active_lanes_in_pipeline — 이 유닛의 활성 레인 수를 AccelWattch에 보고한다.
   */
  virtual void active_lanes_in_pipeline();
  /*
   * [한국어] issue — 명령어를 이 유닛의 파이프라인에 발행한다.
   */
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }

 private:
  int m_supported_op;
  /* [한국어] 이 유닛이 처리하는 명령어 유형 ID.
   * 설정자: 생성자에서 supported_op 파라미터로 설정.
   * 읽는 자: can_issue()에서 inst.op와 비교.
   * 값 범위: SPEC_OP1, SPEC_OP2 등 GPGPU-Sim 정의 열거형 값.
   * 동기화: 읽기 전용. */
};

class simt_core_cluster;           // [한국어] 전방 선언 — ldst_unit이 simt_core_cluster 접근
class shader_memory_interface;     // [한국어] 전방 선언
class shader_core_mem_fetch_allocator; // [한국어] 전방 선언 — ldst_unit이 mem_fetch 생성에 사용
class cache_t;                     // [한국어] 전방 선언 — ldst_unit이 캐시 접근에 사용

/*
 * [한국어] ldst_unit — 로드/스토어 유닛 (Load/Store Unit)
 *
 * === 클래스 역할 ===
 * SM 내에서 모든 메모리 명령어를 처리하는 핵심 유닛이다.
 * 처리 대상: LOAD_OP, STORE_OP, TENSOR_CORE_LOAD/STORE_OP, MEMORY_BARRIER_OP
 *
 * 메모리 접근 경로:
 *   - 공유 메모리(shared): shared_cycle() → L1D 없이 직접 처리
 *   - 상수 캐시(L1C): constant_cycle() → m_L1C (read_only_cache)
 *   - 텍스처 캐시(L1T): texture_cycle() → m_L1T (tex_cache)
 *   - 전역 메모리(L1D/L2/DRAM): memory_cycle() → m_L1D → ICNT → L2 → DRAM
 *   - 원자적 연산: memory_cycle() 내 ATOM 처리 경로
 *
 * pipelined_simd_unit을 상속하지만 stallable() = true이므로
 * 파이프라인이 채워지지 않아도 사이클을 소비할 수 있다.
 * 스톨 발생 시 cycle()이 m_mem_rc를 통해 stall 이유를 기록한다.
 *
 * === 아키텍처에서의 위치 ===
 * shader_core_ctx::m_ldst_unit에 단 하나의 인스턴스로 존재한다.
 * 파이프라인: OC_EX_MEM → [ldst_unit] → (ICNT or 직접 완료) → EX_WB
 *
 * === 타 모듈과의 연결 ===
 * - 상위: shader_core_ctx (issue(), fill(), writeback() 호출)
 * - 캐시: m_L1D/L1C/L1T (gpu-cache.h)
 * - ICNT: m_icnt (mem_fetch_interface → icnt_wrapper)
 * - 스코어보드: m_scoreboard (로드 응답 후 레지스터 해제)
 * - 오퍼랜드 컬렉터: m_operand_collector (라이트백 시 writeback() 호출)
 * - DRAM: mem_fetch → ICNT → memory_partition_unit → dram_t
 *
 * === 주요 함수 요약 ===
 * - issue(): OC_EX_MEM에서 명령어를 받아 메모리 접근 경로로 분기
 * - cycle(): 매 사이클 파이프라인 진행 (메모리 스톨 처리)
 * - fill(): ICNT에서 메모리 응답이 도착하면 m_response_fifo에 추가
 * - writeback(): m_response_fifo에서 응답을 꺼내 오퍼랜드 컬렉터로 라이트백
 */
class ldst_unit : public pipelined_simd_unit {
 public:
  /*
   * [한국어] ldst_unit 주 생성자 — L1D 캐시를 내부에서 생성하는 일반 생성자.
   *
   * @param icnt: ICNT 인터페이스 (메모리 계층으로 mem_fetch를 전달)
   * @param mf_allocator: mem_fetch 패킷 생성 팩토리
   * @param core: 소속 SM (shader_core_ctx*)
   * @param operand_collector: 오퍼랜드 컬렉터 (라이트백 시 사용)
   * @param scoreboard: RAW 해저드 스코어보드 (로드 완료 후 레지스터 해제)
   * @param config, mem_config: SM 및 메모리 설정
   * @param stats: 통계 수집 객체
   * @param sid: SM ID
   * @param tpc: TPC(Texture Processing Cluster) ID
   * @param gpu: 최상위 시뮬레이터 포인터
   *
   * 호출 체인:
   *   exec_shader_core_ctx::create_exec_pipeline() → [ldst_unit(...)]
   */
  ldst_unit(mem_fetch_interface *icnt,
            shader_core_mem_fetch_allocator *mf_allocator,
            shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard, const shader_core_config *config,
            const memory_config *mem_config, class shader_core_stats *stats,
            unsigned sid, unsigned tpc, gpgpu_sim *gpu);

  // Add a structure to record the LDGSTS instructions,
  // similar to m_pending_writes, but since LDGSTS does not have a output
  // register to write to, so a new structure needs to be added
  /* A multi-level map: unsigned (warp_id) -> unsigned (pc) -> unsigned (addr)
   * -> unsigned (count)
   */
  std::map<unsigned /*warp_id*/,
           std::map<unsigned /*pc*/,
                    std::map<unsigned /*addr*/, unsigned /*count*/>>>
      m_pending_ldgsts;
  /* [한국어] LDGSTS (ld.global.sync with shared memory) 명령어의 미완료 요청 추적 맵.
   * LDGSTS는 목적 레지스터 없이 전역→공유 메모리 직접 복사이므로, 완료 감지에 별도 맵 필요.
   * 구조: [warp_id][pc][주소] = 남은 요청 수
   * 설정자: issue() 시 요청 추가.
   * 읽는 자: fill() 응답 수신 시 카운트 감소, 0이 되면 완료로 처리.
   * 동기화: 단일 스레드(cycle()) 내 접근. */

  /*
   * [한국어] issue — OC_EX_MEM 레지스터에서 명령어를 받아 메모리 파이프라인에 발행한다.
   *
   * @param inst: OC_EX_MEM 파이프라인 레지스터 집합 (발행할 명령어 포함)
   *
   * m_dispatch_reg에 명령어를 이동시키고, m_pending_writes에 스토어 요청을 등록한다.
   * Scoreboard에 레지스터를 예약(reserve)하는 것은 이슈 시점이 아닌 execute() 전에 처리된다.
   *
   * 호출 체인:
   *   shader_core_ctx::execute() → [issue(inst)] → m_dispatch_reg에 명령어 저장
   */
  // modifiers
  virtual void issue(register_set &inst);
  /*
   * [한국어] is_issue_partitioned — ldst_unit은 서브코어 파티셔닝 미지원 (false 반환).
   * 메모리 유닛은 모든 스케줄러가 공유하므로 파티셔닝되지 않는다.
   */
  bool is_issue_partitioned() { return false; }
  /*
   * [한국어] cycle — ldst_unit의 1 사이클 메모리 파이프라인 동작을 수행한다.
   *
   * 실행 순서:
   *   1. writeback(): 완료된 메모리 응답을 오퍼랜드 컬렉터로 전달
   *   2. L1_latency_queue_cycle(): L1 캐시 레이턴시 큐 진행
   *   3. shared_cycle/constant_cycle/texture_cycle/memory_cycle(): 명령어 유형별 접근 처리
   *
   * 스톨 발생 시 m_mem_rc에 stall 이유를 기록하고 다음 사이클에 재시도한다.
   *
   * 호출 체인:
   *   shader_core_ctx::execute() → [cycle()]
   */
  virtual void cycle();

  /*
   * [한국어] fill — ICNT에서 도착한 메모리 응답 패킷을 m_response_fifo에 추가한다.
   *
   * @param mf: 완료된 메모리 요청 패킷 (mem_fetch)
   *
   * accept_ldst_unit_response()에서 호출되며, 다음 cycle()의 writeback()에서 처리된다.
   *
   * 호출 체인:
   *   shader_core_ctx::accept_ldst_unit_response() → [fill(mf)] → m_response_fifo.push_back()
   */
  void fill(mem_fetch *mf);
  /*
   * [한국어] flush — L1D 캐시를 플러시(dirty 라인을 L2로 write-back)한다.
   * CUDA의 cuCtxSynchronize() 또는 커널 종료 시 호출될 수 있다.
   */
  void flush();
  /*
   * [한국어] invalidate — L1D 캐시의 모든 라인을 무효화(invalidate)한다.
   * cache coherence 이벤트 또는 캐시 설정 변경 시 호출된다.
   */
  void invalidate();
  /*
   * [한국어] writeback — m_response_fifo에서 완료된 메모리 응답을 꺼내 레지스터 파일에 라이트백한다.
   *
   * m_response_fifo의 앞쪽 mem_fetch를 처리하며, m_pending_writes 카운트를 감소시킨다.
   * 해당 레지스터의 모든 요청이 완료되면 scoreboard->releaseRegister()를 호출해 RAW 해저드 해제.
   * m_next_wb에 완료 명령어를 저장하고 opndcoll_rfu_t::writeback()으로 레지스터 파일에 기록.
   *
   * 호출 체인:
   *   cycle() → [writeback()] → scoreboard->releaseRegister() → m_operand_collector->writeback()
   */
  void writeback();

  /*
   * [한국어] clock_multiplier — ldst_unit의 클럭 배율을 반환한다.
   *
   * @return: 메모리 설정에 따른 클럭 배율 (예: L1D가 코어보다 느린 경우 > 1)
   *
   * 호출 체인:
   *   shader_core_ctx::execute() → [clock_multiplier()] → cycle() 호출 빈도 결정
   */
  // accessors
  virtual unsigned clock_multiplier() const;

  /*
   * [한국어] can_issue — ldst_unit이 주어진 명령어를 받을 수 있는지 확인한다.
   *
   * @param inst: 이슈하려는 명령어
   * @return: LOAD/STORE/BARRIER 계열이고 m_dispatch_reg가 비어있으면 true
   *
   * pipelined_simd_unit과 달리 occupied 비트 검사 없이 dispatch 슬롯만 확인한다.
   * (메모리 유닛은 결과 버스 대신 응답 FIFO를 사용)
   */
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case LOAD_OP:              break; // [한국어] 일반 메모리 로드 (ld.global, ld.local 등)
      case TENSOR_CORE_LOAD_OP:  break; // [한국어] Tensor Core 입력 행렬 로드
      case STORE_OP:             break; // [한국어] 일반 메모리 스토어 (st.global, st.local 등)
      case TENSOR_CORE_STORE_OP: break; // [한국어] Tensor Core 출력 행렬 스토어
      case MEMORY_BARRIER_OP:    break; // [한국어] bar.sync, membar.gl 등 메모리 배리어
      default:                   return false; // [한국어] 그 외 명령어는 메모리 유닛 처리 불가
    }
    return m_dispatch_reg->empty(); // [한국어] dispatch 슬롯이 비어있어야 이슈 가능 (occupied 비트 검사 없음)
  }

  /*
   * [한국어] active_lanes_in_pipeline — 메모리 파이프라인의 활성 레인 수를 AccelWattch에 보고한다.
   */
  virtual void active_lanes_in_pipeline();
  /*
   * [한국어] stallable — ldst_unit이 스톨 가능함을 반환한다.
   * @return: 항상 true — 메모리 스톨 발생 시 파이프라인이 차단됨
   */
  virtual bool stallable() const { return true; }
  /*
   * [한국어] response_buffer_full — m_response_fifo가 가득 찼는지 확인한다.
   * @return: m_response_fifo.size() >= ldst_unit_response_queue_size이면 true
   *
   * 가득 찬 경우 ICNT는 새 응답을 전달하지 않는다.
   */
  bool response_buffer_full() const;
  /*
   * [한국어] print — 이 유닛의 현재 상태를 fout에 출력한다 (디버깅용).
   */
  void print(FILE *fout) const;
  /*
   * [한국어] print_cache_stats — L1D 캐시 통계를 출력한다.
   */
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses);
  /*
   * [한국어] get_cache_stats — 캐시 통계를 특정 유형(L1D/L1T/L1C)으로 조회한다.
   */
  void get_cache_stats(unsigned &read_accesses, unsigned &write_accesses,
                       unsigned &read_misses, unsigned &write_misses,
                       unsigned cache_type);
  void get_cache_stats(cache_stats &cs);

  /*
   * [한국어] get_L1D/C/T_sub_stats — L1D/상수/텍스처 캐시의 상세 통계를 수집한다.
   * simt_core_cluster::get_cache_stats() 집계 시 호출된다.
   */
  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;

 protected:
  /*
   * [한국어] ldst_unit 보호 생성자 — 외부에서 L1D 캐시 포인터를 주입하는 변형 생성자.
   * 서브클래스(SST 변형 등)에서 커스텀 L1D 캐시를 주입할 때 사용한다.
   *
   * 호출 체인:
   *   sst 관련 서브클래스 생성자 → [ldst_unit(..., new_l1d_cache)]
   */
  ldst_unit(mem_fetch_interface *icnt,
            shader_core_mem_fetch_allocator *mf_allocator,
            shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard, const shader_core_config *config,
            const memory_config *mem_config, shader_core_stats *stats,
            unsigned sid, unsigned tpc, l1_cache *new_l1d_cache);
  /*
   * [한국어] init — 두 생성자가 공통으로 호출하는 초기화 함수.
   *
   * m_icnt, m_mf_allocator, m_core, m_operand_collector, m_scoreboard 등을 설정하고,
   * m_response_fifo를 초기화한다.
   *
   * 호출 체인:
   *   ldst_unit 두 생성자 → [init(...)]
   */
  void init(mem_fetch_interface *icnt,
            shader_core_mem_fetch_allocator *mf_allocator,
            shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard, const shader_core_config *config,
            const memory_config *mem_config, shader_core_stats *stats,
            unsigned sid, unsigned tpc);

 protected:
  /*
   * [한국어] shared_cycle — 공유 메모리 접근을 처리한다.
   *
   * @param inst: 처리할 메모리 명령어
   * @param rc_fail: 스톨 발생 시 실패 이유를 기록
   * @param fail_type: 메모리 접근 유형 (공유 메모리)
   * @return: 성공(진행) 가능하면 true, 스톨이면 false
   *
   * 공유 메모리 뱅크 충돌 감지, 동기화 처리를 담당한다.
   *
   * 호출 체인:
   *   cycle() → [shared_cycle()]
   */
  bool shared_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                    mem_stage_access_type &fail_type);
  /*
   * [한국어] constant_cycle — 상수 캐시(L1C) 접근을 처리한다.
   *
   * m_L1C에 read_only 요청을 발행하고, 히트/미스/스톨 여부를 반환한다.
   *
   * 호출 체인:
   *   cycle() → [constant_cycle()]
   */
  bool constant_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                      mem_stage_access_type &fail_type);
  /*
   * [한국어] texture_cycle — 텍스처 캐시(L1T) 접근을 처리한다.
   *
   * m_L1T에 텍스처 요청을 발행하고 히트/미스를 처리한다.
   *
   * 호출 체인:
   *   cycle() → [texture_cycle()]
   */
  bool texture_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                     mem_stage_access_type &fail_type);
  /*
   * [한국어] memory_cycle — 전역 메모리(L1D → L2 → DRAM) 접근을 처리한다.
   *
   * L1D 캐시에 요청을 발행하고, 미스 시 ICNT를 통해 L2/DRAM으로 mem_fetch를 전달한다.
   * 원자적 연산(ATOM)도 이 경로를 통해 처리된다.
   *
   * 호출 체인:
   *   cycle() → [memory_cycle()] → m_L1D->access() → process_cache_access()
   *             → m_icnt->push(mf) (미스 시)
   */
  bool memory_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                    mem_stage_access_type &fail_type);

  /*
   * [한국어] process_cache_access — 캐시 접근 결과(히트/미스/스톨)를 처리한다.
   *
   * @param cache: 접근한 캐시 인스턴스
   * @param address: 접근 주소
   * @param inst: 처리 중인 명령어
   * @param events: 캐시 이벤트 목록 (fill, eviction 등)
   * @param mf: 관련 mem_fetch 패킷
   * @param status: 캐시 접근 결과 (HIT/MISS/RESERVATION_FAIL 등)
   * @return: 메모리 스톨 유형 (NO_RC_FAIL이면 진행 가능)
   *
   * 히트이면 바로 결과 사용, 미스이면 m_icnt로 요청 전달, 스톨이면 m_mem_rc 설정.
   *
   * 호출 체인:
   *   memory_cycle() → cache->access() → [process_cache_access()]
   */
  virtual mem_stage_stall_type process_cache_access(
      cache_t *cache, new_addr_type address, warp_inst_t &inst,
      std::list<cache_event> &events, mem_fetch *mf,
      enum cache_request_status status);
  /*
   * [한국어] process_memory_access_queue — 명령어의 모든 메모리 접근 요청을 캐시에 제출한다.
   *
   * warp_inst_t가 생성한 mem_access_t 목록을 순회하며 각 접근을 cache->access()로 처리한다.
   * 스톨 발생 시 남은 접근은 다음 사이클에 재시도된다.
   *
   * 호출 체인:
   *   memory_cycle() → [process_memory_access_queue(cache, inst)]
   */
  mem_stage_stall_type process_memory_access_queue(cache_t *cache,
                                                   warp_inst_t &inst);
  /*
   * [한국어] process_memory_access_queue_l1cache — L1D 전용 접근 큐 처리 함수.
   * l1_cache 타입으로 특화되어 추가 기능(LDGSTS 등)을 지원한다.
   */
  mem_stage_stall_type process_memory_access_queue_l1cache(l1_cache *cache,
                                                           warp_inst_t &inst);
  gpgpu_sim *m_gpu;
  /* [한국어] 최상위 시뮬레이터 포인터.
   * 설정자: 생성자에서 설정.
   * 읽는 자: LDGSTS 처리, 시뮬레이션 전역 정보 접근.
   * 동기화: 읽기 전용. */

  const memory_config *m_memory_config;
  /* [한국어] 메모리 계층 설정 포인터 (L2/DRAM 파라미터 포함).
   * 설정자: init()에서 설정.
   * 동기화: 읽기 전용. */

  class mem_fetch_interface *m_icnt;
  /* [한국어] ICNT(on-chip network) 인터페이스 포인터.
   * L1D 미스 시 mem_fetch를 ICNT를 통해 메모리 파티션으로 전달한다.
   * 설정자: init()에서 설정.
   * 읽는 자: memory_cycle()에서 L1 미스 처리 시 push().
   * 동기화: 단일 스레드 접근. */

  shader_core_mem_fetch_allocator *m_mf_allocator;
  /* [한국어] mem_fetch 패킷 생성 팩토리.
   * memory_cycle()에서 각 메모리 접근마다 mem_fetch 객체를 생성할 때 사용.
   * 설정자: init()에서 설정.
   * 동기화: 읽기 전용. */

  class shader_core_ctx *m_core;
  /* [한국어] 소속 SM (shader_core_ctx) 포인터.
   * AccelWattch 카운터 증가, store_ack 호출 등에 사용.
   * 설정자: init()에서 설정.
   * 동기화: 읽기 전용. */

  unsigned m_sid;
  /* [한국어] 소속 SM의 ID (shader ID).
   * mem_fetch 생성 시 소스 SM 식별에 사용.
   * 설정자: init()에서 설정.
   * 동기화: 읽기 전용. */

  unsigned m_tpc;
  /* [한국어] TPC (Texture Processing Cluster) ID.
   * mem_fetch 생성 시 클러스터 식별에 사용.
   * 설정자: init()에서 설정.
   * 동기화: 읽기 전용. */

  tex_cache *m_L1T;        // texture cache
  /* [한국어] L1 텍스처 캐시 포인터.
   * texture_cycle()에서 TEX_MEM_SPACE 접근 요청 처리에 사용.
   * 설정자: 생성자에서 new tex_cache()로 생성.
   * 동기화: 단일 스레드 접근. */

  read_only_cache *m_L1C;  // constant cache
  /* [한국어] L1 상수 캐시 포인터 (읽기 전용).
   * constant_cycle()에서 CONST_MEM_SPACE 접근 요청 처리에 사용.
   * 설정자: 생성자에서 new read_only_cache()로 생성.
   * 동기화: 단일 스레드 접근. */

  l1_cache *m_L1D;         // data cache
  /* [한국어] L1 데이터 캐시 포인터 (전역 메모리 및 로컬 메모리 접근).
   * memory_cycle()에서 GLOBAL_MEM_SPACE, LOCAL_MEM_SPACE 접근에 사용.
   * 설정자: 주 생성자에서 내부 생성, 보호 생성자에서 new_l1d_cache를 주입.
   * 동기화: 단일 스레드 접근. */

  std::map<unsigned /*warp_id*/,
           std::map<unsigned /*regnum*/, unsigned /*count*/>>
      m_pending_writes;
  /* [한국어] 미완료 메모리 로드 요청 추적 맵 (warp_id → 레지스터 번호 → 남은 요청 수).
   * 하나의 로드 명령어가 여러 캐시 라인(sector)에 걸쳐 접근할 수 있으므로 카운팅이 필요하다.
   * 설정자: issue() 시 로드 명령어의 목적 레지스터마다 카운트 설정.
   * 읽는 자: writeback()에서 응답 수신 시 카운트 감소, 0이 되면 scoreboard 해제.
   * 동기화: 단일 스레드(cycle()) 내 접근. */

  std::list<mem_fetch *> m_response_fifo;
  /* [한국어] ICNT에서 도착한 메모리 응답 패킷의 FIFO 큐.
   * 설정자: fill()에서 ICNT 응답 수신 시 push_back.
   * 읽는 자: writeback()에서 front를 꺼내 레지스터 파일에 결과 저장.
   * 크기 제한: ldst_unit_response_queue_size (response_buffer_full() 확인).
   * 동기화: fill()은 ICNT 응답 경로, writeback()은 cycle()에서 호출 — 단일 스레드 보장. */

  opndcoll_rfu_t *m_operand_collector;
  /* [한국어] 오퍼랜드 컬렉터 역참조 포인터.
   * writeback() 완료 시 opndcoll_rfu_t::writeback()을 호출해 레지스터 파일에 기록.
   * 설정자: init()에서 설정.
   * 동기화: 읽기 전용. */

  Scoreboard *m_scoreboard;
  /* [한국어] RAW 해저드 스코어보드 포인터.
   * writeback() 시 완료된 로드의 목적 레지스터를 releaseRegister()로 해제.
   * 설정자: init()에서 설정.
   * 동기화: 읽기 전용. */

  mem_fetch *m_next_global;
  /* [한국어] 다음 전역 메모리 응답 처리를 위한 임시 포인터.
   * 처리 중인 mem_fetch를 임시로 보관하는 데 사용된다.
   * 설정자: writeback() / memory_cycle()에서 임시 사용.
   * 동기화: 단일 스레드 접근. */

  warp_inst_t m_next_wb;
  /* [한국어] 다음으로 라이트백될 명령어의 사본.
   * 오퍼랜드 컬렉터의 writeback()에 전달하기 위해 임시로 보관.
   * 설정자: writeback() 내에서 응답 mem_fetch로부터 복원.
   * 동기화: 단일 스레드 접근. */

  unsigned m_writeback_arb;  // round-robin arbiter for writeback contention
                             // between L1T, L1C, shared
  /* [한국어] L1T, L1C, 공유 메모리 간 라이트백 경쟁을 중재하는 라운드 로빈 아비터 상태.
   * 매 사이클 중 어떤 캐시(L1T, L1C, 공유)에서 라이트백을 먼저 처리할지 결정한다.
   * 설정자: init()에서 0으로 초기화, writeback()에서 순환 갱신.
   * 동기화: 단일 스레드 접근. */

  unsigned m_num_writeback_clients;
  /* [한국어] 라이트백 경쟁에 참여하는 클라이언트 수 (L1T, L1C, 공유 = 3).
   * 설정자: init()에서 설정.
   * 동기화: 읽기 전용. */

  enum mem_stage_stall_type m_mem_rc;
  /* [한국어] 현재 메모리 파이프라인의 스톨 상태 코드.
   * NO_RC_FAIL이면 진행 가능, 그 외이면 스톨 중.
   * 설정자: shared/constant/texture/memory_cycle()에서 스톨 발생 시 설정.
   * 읽는 자: cycle()에서 진행 여부 판단, 통계 집계.
   * 동기화: 단일 스레드 접근. */

  shader_core_stats *m_stats;
  /* [한국어] SM 통계 수집 객체.
   * 메모리 접근 카운터 업데이트에 사용.
   * 설정자: init()에서 설정.
   * 동기화: 읽기 전용. */

  // for debugging
  unsigned long long m_last_inst_gpu_sim_cycle;
  /* [한국어] 마지막 명령어 이슈 시점의 시뮬레이션 사이클 번호 (디버깅용).
   * 설정자: issue() 시 현재 사이클로 업데이트.
   * 동기화: 단일 스레드 접근. */

  unsigned long long m_last_inst_gpu_tot_sim_cycle;
  /* [한국어] 마지막 명령어 이슈 시점의 전체 시뮬레이션 사이클 번호 (멀티코어 총합, 디버깅용).
   * 설정자: issue() 시 업데이트.
   * 동기화: 단일 스레드 접근. */

  std::vector<std::deque<mem_fetch *>> l1_latency_queue;
  /* [한국어] L1 캐시 히트 레이턴시를 모델링하기 위한 지연 큐 벡터.
   * L1D 히트이더라도 즉시 결과를 사용할 수 없고 몇 사이클 지연이 있다.
   * l1_latency_queue[사이클오프셋]에 mem_fetch를 삽입하고, L1_latency_queue_cycle()이 매 사이클 전진.
   * 설정자: memory_cycle()에서 L1 히트 시 해당 오프셋에 push_back.
   * 읽는 자: L1_latency_queue_cycle()에서 front 원소를 m_response_fifo로 이동.
   * 동기화: 단일 스레드 접근. */

  /*
   * [한국어] L1_latency_queue_cycle — L1 캐시 레이턴시 큐를 한 사이클 전진시킨다.
   *
   * l1_latency_queue의 맨 앞 원소들(만료된 레이턴시)을 m_response_fifo로 이동한다.
   * 매 사이클 cycle()에서 가장 먼저 호출된다.
   *
   * 호출 체인:
   *   cycle() → [L1_latency_queue_cycle()] → m_response_fifo.push_back()
   */
  void L1_latency_queue_cycle();
};

/*
 * [한국어] pipeline_stage_name_t — SM 파이프라인 단계 이름 열거형
 *
 * === 열거형 역할 ===
 * SM 파이프라인의 각 단계(Instruction Decode → Operand Collect → Execute → Write-Back)를
 * 열거형 ID로 식별한다.
 * ID_OC_*: ID → OC 단계 파이프라인 레지스터 (명령어 디코드 → 오퍼랜드 컬렉터 입력)
 * OC_EX_*: OC → EX 단계 파이프라인 레지스터 (오퍼랜드 컬렉터 출력 → 실행 유닛 입력)
 * EX_WB: EX → WB 단계 파이프라인 레지스터 (실행 완료 → 라이트백)
 *
 * === 사용처 ===
 * - shader_core_ctx::m_pipeline_reg 배열의 인덱스로 사용
 * - shader_core_config::pipe_widths[] 배열의 인덱스로 사용 (gpgpusim.config 파싱)
 * - pipelined_simd_unit의 결과 포트 연결에 사용
 */
enum pipeline_stage_name_t {
  ID_OC_SP = 0,      // [한국어] ID → OC 레지스터: SP(FP32) ALU 경로 (인덱스 0)
  ID_OC_DP,          // [한국어] ID → OC 레지스터: DP(FP64) ALU 경로
  ID_OC_INT,         // [한국어] ID → OC 레지스터: INT ALU 경로
  ID_OC_SFU,         // [한국어] ID → OC 레지스터: SFU(특수 함수) 경로
  ID_OC_MEM,         // [한국어] ID → OC 레지스터: MEM(로드/스토어) 경로
  OC_EX_SP,          // [한국어] OC → EX 레지스터: SP(FP32) ALU 경로
  OC_EX_DP,          // [한국어] OC → EX 레지스터: DP(FP64) ALU 경로
  OC_EX_INT,         // [한국어] OC → EX 레지스터: INT ALU 경로
  OC_EX_SFU,         // [한국어] OC → EX 레지스터: SFU 경로
  OC_EX_MEM,         // [한국어] OC → EX 레지스터: MEM 경로
  EX_WB,             // [한국어] EX → WB 레지스터: 모든 실행 유닛의 결과가 합류하는 라이트백 경로
  ID_OC_TENSOR_CORE, // [한국어] ID → OC 레지스터: Tensor Core 경로 (Volta 이후)
  OC_EX_TENSOR_CORE, // [한국어] OC → EX 레지스터: Tensor Core 경로
  N_PIPELINE_STAGES  // [한국어] 파이프라인 단계 총 수 (배열 크기 결정에 사용, =13)
};

const char *const pipeline_stage_name_decode[] = {
    // [한국어] pipeline_stage_name_t 열거형 값 → 문자열 변환 배열 (디버그 출력용)
    "ID_OC_SP",          "ID_OC_DP",         "ID_OC_INT", "ID_OC_SFU",
    "ID_OC_MEM",         "OC_EX_SP",         "OC_EX_DP",  "OC_EX_INT",
    "OC_EX_SFU",         "OC_EX_MEM",        "EX_WB",     "ID_OC_TENSOR_CORE",
    "OC_EX_TENSOR_CORE", "N_PIPELINE_STAGES"};

/*
 * [한국어] specialized_unit_params — 사용자 설정 특수 목적 실행 유닛의 구성 파라미터
 *
 * === 구조체 역할 ===
 * gpgpusim.config의 -specialized_unit 옵션에서 파싱된 특수 유닛 설정을 보관한다.
 * shader_core_config::m_specialized_unit 벡터에 저장되며,
 * create_exec_pipeline()에서 specialized_unit 인스턴스 생성에 사용된다.
 *
 * === 타 모듈과의 연결 ===
 * - 파싱: shader_core_config::init()에서 specialized_unit_string 파싱
 * - 사용: shader_core_ctx::create_exec_pipeline()에서 specialized_unit 생성
 */
struct specialized_unit_params {
  unsigned latency;
  /* [한국어] 이 특수 유닛의 파이프라인 레이턴시 (사이클 수).
   * 설정자: shader_core_config::init()에서 config 문자열 파싱으로 설정.
   * 읽는 자: create_exec_pipeline()에서 specialized_unit 생성자에 전달.
   * 값 범위: 1 이상. */

  unsigned num_units;
  /* [한국어] 이 유형의 특수 유닛 수 (SM당).
   * 설정자: config 문자열 파싱.
   * 읽는 자: create_exec_pipeline()에서 유닛 생성 반복 횟수 결정.
   * 값 범위: 1 이상. */

  unsigned id_oc_spec_reg_width;
  /* [한국어] ID_OC_SPEC 파이프라인 레지스터 너비 (동시 이슈 가능한 명령어 수).
   * 설정자: config 문자열 파싱.
   * 읽는 자: create_exec_pipeline()에서 파이프라인 레지스터 크기 설정.
   * 동기화: 읽기 전용. */

  unsigned oc_ex_spec_reg_width;
  /* [한국어] OC_EX_SPEC 파이프라인 레지스터 너비.
   * 설정자: config 문자열 파싱.
   * 읽는 자: create_exec_pipeline()에서 파이프라인 레지스터 크기 설정.
   * 동기화: 읽기 전용. */

  char name[20];
  /* [한국어] 이 특수 유닛의 이름 문자열 (최대 19자 + null terminator).
   * 설정자: shader_core_config::init()에서 strncpy()로 복사.
   * 읽는 자: specialized_unit 생성자에서 m_name 설정, 디버그 출력.
   * 값 예: "SPEC_OP1", "SPEC_OP2" 등 사용자 정의. */

  unsigned ID_OC_SPEC_ID;
  /* [한국어] 이 특수 유닛의 ID_OC 파이프라인 레지스터 인덱스.
   * N_PIPELINE_STAGES + offset 형태로 계산된다.
   * 설정자: shader_core_config::init()에서 설정.
   * 읽는 자: create_exec_pipeline()에서 파이프라인 레지스터 접근에 사용. */

  unsigned OC_EX_SPEC_ID;
  /* [한국어] 이 특수 유닛의 OC_EX 파이프라인 레지스터 인덱스.
   * 설정자: shader_core_config::init()에서 설정.
   * 읽는 자: create_exec_pipeline()에서 사용. */
};

/*
 * [한국어] shader_core_config — SM(Shader Core) 설정 클래스
 *
 * === 클래스 역할 ===
 * gpgpusim.config 파일에서 파싱된 SM(Streaming Multiprocessor) 관련 모든 설정값을 보관한다.
 * 파이프라인 구조(너비, 레이턴시), 캐시 설정(L1I/D/C/T), 스케줄러 설정, 레지스터 파일 설정,
 * 실행 유닛 수, 전력 모델 설정 등 SM의 마이크로아키텍처를 완전히 정의한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 시뮬레이션 시작 시 gpgpu_sim_config에서 reg_options()로 옵션을 등록하고,
 * init()에서 파싱된 문자열을 검증·구조화한다.
 * shader_core_ctx, ldst_unit, simd_function_unit 등 모든 SM 구성요소가 이 설정을 참조한다.
 *
 * === 타 모듈과의 연결 ===
 * - 소유자: gpgpu_sim (gpgpu_sim_config 통해 접근)
 * - 사용자: shader_core_ctx, simt_core_cluster, ldst_unit, opndcoll_rfu_t 등 SM 관련 모든 클래스
 * - 설정 파일: gpgpusim.config (-gpgpu_shader_core_pipeline, -gpgpu_num_sp_units 등)
 *
 * === 주요 함수 요약 ===
 * - init(): 파싱된 문자열을 검증하고 파생 값(max_warps_per_shader 등) 계산
 * - reg_options(): OptionParser에 모든 설정 옵션 등록
 * - max_cta(): 주어진 커널 파라미터로 SM에서 동시 실행 가능한 최대 CTA 수 계산
 * - sid_to_cluster()/sid_to_cid(): SM ID ↔ 클러스터 ID 변환
 */
class shader_core_config : public core_config {
 public:
  /*
   * [한국어] shader_core_config 생성자 — 기본값으로 초기화한다.
   *
   * @param ctx: GPGPU-Sim 실행 컨텍스트 (전역 설정 포함)
   *
   * core_config(ctx)를 위임 호출하고, pipeline_widths_string을 NULL로 초기화한다.
   * reg_options() 호출 전에는 모든 설정이 기본값(0/NULL/false)이다.
   *
   * 호출 체인:
   *   gpgpu_sim_config 생성 → [shader_core_config(ctx)]
   */
  shader_core_config(gpgpu_context *ctx) : core_config(ctx) {
    pipeline_widths_string = NULL; // [한국어] 파이프라인 너비 문자열 — reg_options()/init() 전까지 NULL
    gpgpu_ctx = ctx;               // [한국어] 전역 실행 컨텍스트 역참조 저장
  }

  /*
   * [한국어] init — 파싱된 설정 문자열을 검증하고 파생 값을 계산한다.
   *
   * reg_options() 이후 OptionParser::parse() 완료 후 호출된다.
   * 주요 동작:
   *   1. gpgpu_shader_core_pipeline_opt 파싱 → n_thread_per_shader, warp_size 추출
   *   2. pipeline_widths_string 파싱 → pipe_widths[] 배열 채움
   *   3. max_warps_per_shader = n_thread_per_shader / warp_size 계산
   *   4. L1I/T/C/D 캐시 설정 초기화
   *   5. specialized_unit_string[] 파싱 → m_specialized_unit 벡터 채움
   *   6. adaptive cache config 파싱 (shmem_opt_list)
   *
   * 호출 체인:
   *   gpgpu_sim_config::init() → [shader_core_config::init()]
   */
  void init() {
    int ntok = sscanf(gpgpu_shader_core_pipeline_opt, "%d:%d",
                      &n_thread_per_shader, &warp_size); // [한국어] "N_threads:warp_size" 형식 파싱
    if (ntok != 2) { // [한국어] 파싱 실패 시 오류 메시지 출력 후 abort
      printf(
          "GPGPU-Sim uArch: error while parsing configuration string "
          "gpgpu_shader_core_pipeline_opt\n");
      abort(); // [한국어] 설정 문자열 오류로 시뮬레이터 중단
    }

    char *toks = new char[100]; // [한국어] 문자열 토크나이저용 임시 버퍼
    char *tokd = toks;          // [한국어] 나중에 delete[]하기 위한 원본 포인터 보관
    strcpy(toks, pipeline_widths_string); // [한국어] pipeline_widths_string 복사 (strtok가 원본을 수정하므로)

    toks = strtok(toks, ","); // [한국어] 첫 번째 쉼표 구분 토큰 추출

    /*	Removing the tensorcore pipeline while reading the config files if the
       tensor core is not available. If we won't remove it, old regression will
       be broken. So to support the legacy config files it's best to handle in
       this way.
     */
    // [한국어] Tensor Core 미사용 시 ID_OC_TENSOR_CORE, OC_EX_TENSOR_CORE 2개 단계를 건너뜀
    // 구형 설정 파일(Tensor Core 항목 없음) 호환성 유지를 위해
    int num_config_to_read = N_PIPELINE_STAGES - 2 * (!gpgpu_tensor_core_avail); // [한국어] 읽을 파이프라인 단계 수: TC 없으면 N-2개

    for (int i = 0; i < num_config_to_read; i++) { // [한국어] 각 파이프라인 단계의 너비를 순서대로 파싱
      assert(toks); // [한국어] 아직 읽을 토큰이 있어야 함 — 부족하면 설정 오류
      ntok = sscanf(toks, "%d", &pipe_widths[i]); // [한국어] 이 파이프라인 단계의 너비(동시 이슈 가능 명령어 수) 파싱
      assert(ntok == 1); // [한국어] 파싱 성공 보장
      toks = strtok(NULL, ","); // [한국어] 다음 토큰으로 이동
    }

    delete[] tokd; // [한국어] 복사 버퍼 해제

    if (n_thread_per_shader > MAX_THREAD_PER_SM) { // [한국어] 설정된 스레드 수가 컴파일 타임 상수보다 크면 오류
      printf(
          "GPGPU-Sim uArch: Error ** increase MAX_THREAD_PER_SM in "
          "abstract_hardware_model.h from %u to %u\n",
          MAX_THREAD_PER_SM, n_thread_per_shader);
      abort(); // [한국어] abstract_hardware_model.h의 MAX_THREAD_PER_SM을 늘려야 함
    }
    max_warps_per_shader = n_thread_per_shader / warp_size;
    assert(!(n_thread_per_shader % warp_size));

    set_pipeline_latency();

    m_L1I_config.init(m_L1I_config.m_config_string, FuncCachePreferNone);
    m_L1T_config.init(m_L1T_config.m_config_string, FuncCachePreferNone);
    m_L1C_config.init(m_L1C_config.m_config_string, FuncCachePreferNone);
    m_L1D_config.init(m_L1D_config.m_config_string, FuncCachePreferNone);
    gpgpu_cache_texl1_linesize = m_L1T_config.get_line_sz();
    gpgpu_cache_constl1_linesize = m_L1C_config.get_line_sz();
    m_valid = true;

    m_specialized_unit_num = 0;
    // parse the specialized units
    for (unsigned i = 0; i < SPECIALIZED_UNIT_NUM; ++i) {
      unsigned enabled;
      specialized_unit_params sparam; // [한국어] 이 특수 유닛의 파라미터 임시 변수
      sscanf(specialized_unit_string[i], "%u,%u,%u,%u,%u,%s", &enabled,
             &sparam.num_units, &sparam.latency, &sparam.id_oc_spec_reg_width,
             &sparam.oc_ex_spec_reg_width, sparam.name); // [한국어] "enabled,num,latency,id_oc,oc_ex,name" 형식 파싱

      if (enabled) { // [한국어] 활성화된 특수 유닛만 목록에 추가
        m_specialized_unit.push_back(sparam); // [한국어] 파라미터 벡터에 추가
        strncpy(m_specialized_unit.back().name, sparam.name,
                sizeof(m_specialized_unit.back().name)); // [한국어] 이름 문자열 별도 복사 (push_back 후 포인터 무효화 방지)
        m_specialized_unit_num += sparam.num_units; // [한국어] 전체 특수 유닛 수 누적
      } else
        break;  // we only accept continuous specialized_units, i.e., 1,2,3,4
                // [한국어] 비활성화 유닛 이후는 더 이상 파싱하지 않음 (연속 활성화만 허용)
    }

    // parse gpgpu_shmem_option for adpative cache config
    // [한국어] 적응형 캐시 설정(adaptive_cache_config)이 활성화된 경우 공유 메모리 옵션 파싱
    if (adaptive_cache_config) {
      std::stringstream ss(gpgpu_shmem_option); // [한국어] 공유 메모리 크기 옵션 문자열을 스트림으로 변환
      while (ss.good()) {
        std::string option;
        std::getline(ss, option, ','); // [한국어] 쉼표로 구분된 각 옵션 값을 읽음
        shmem_opt_list.push_back((unsigned)std::stoi(option) * 1024); // [한국어] KB → 바이트 변환 후 목록에 추가
      }
      std::sort(shmem_opt_list.begin(), shmem_opt_list.end()); // [한국어] 오름차순 정렬 (최적 설정 탐색 시 사용)
    }
  }
  /*
   * [한국어] reg_options — OptionParser에 모든 SM 설정 옵션을 등록한다.
   * gpgpusim.config의 -gpgpu_shader_core_pipeline 등 모든 옵션을 opp에 등록한다.
   * 구현은 shader.cc에 있다.
   *
   * 호출 체인:
   *   gpgpu_sim_config::reg_options() → [shader_core_config::reg_options()]
   */
  void reg_options(class OptionParser *opp);
  /*
   * [한국어] max_cta — 주어진 커널의 파라미터로 이 SM에서 동시 실행 가능한 최대 CTA 수를 계산한다.
   *
   * @param k: 커널 정보 (스레드 수, 공유 메모리 크기, 레지스터 수 포함)
   * @return: SM이 수용 가능한 최대 CTA 수 (하드웨어 자원 제한: 스레드/레지스터/공유메모리 중 최솟값)
   *
   * 호출 체인:
   *   simt_core_cluster::max_cta() → [max_cta(kernel)]
   */
  unsigned max_cta(const kernel_info_t &k) const;
  /*
   * [한국어] num_shader — 전체 SM 수를 반환한다.
   * @return: n_simt_clusters * n_simt_cores_per_cluster (클러스터 수 × 클러스터당 SM 수)
   */
  unsigned num_shader() const {
    return n_simt_clusters * n_simt_cores_per_cluster; // [한국어] 총 SM 수 = 클러스터 수 × 클러스터당 SM 수
  }
  /*
   * [한국어] sid_to_cluster — SM ID를 클러스터 ID로 변환한다.
   * @param sid: SM의 전역 ID
   * @return: 해당 SM이 속한 클러스터 ID (sid / n_simt_cores_per_cluster)
   */
  unsigned sid_to_cluster(unsigned sid) const {
    return sid / n_simt_cores_per_cluster; // [한국어] 클러스터 ID = 전역 SM ID / 클러스터당 SM 수
  }
  /*
   * [한국어] sid_to_cid — SM ID를 클러스터 내 로컬 SM ID로 변환한다.
   * @param sid: SM의 전역 ID
   * @return: 클러스터 내에서의 SM 인덱스 (sid % n_simt_cores_per_cluster)
   */
  unsigned sid_to_cid(unsigned sid) const {
    return sid % n_simt_cores_per_cluster; // [한국어] 클러스터 내 SM 인덱스 = 전역 SM ID % 클러스터당 SM 수
  }
  /*
   * [한국어] cid_to_sid — 클러스터 내 SM 인덱스와 클러스터 ID로 전역 SM ID를 계산한다.
   * @param cid: 클러스터 내 SM 인덱스
   * @param cluster_id: 클러스터 ID
   * @return: 전역 SM ID (cluster_id * n_simt_cores_per_cluster + cid)
   */
  unsigned cid_to_sid(unsigned cid, unsigned cluster_id) const {
    return cluster_id * n_simt_cores_per_cluster + cid; // [한국어] 전역 SM ID = 클러스터 ID × 클러스터당 SM 수 + 클러스터 내 SM 인덱스
  }
  /*
   * [한국어] set_pipeline_latency — 파이프라인 레이턴시 설정을 계산하고 저장한다.
   * init() 내부에서 호출되며, max_sp_latency, max_sfu_latency 등 실행 유닛별 최대 레이턴시를 결정한다.
   */
  void set_pipeline_latency();

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  /* [한국어] GPGPU-Sim 전역 실행 컨텍스트 역참조.
   * PTX 시뮬레이션 상태, 전역 함수 테이블 등 전역 정보 접근에 사용.
   * 동기화: 읽기 전용. */

  // data
  char *gpgpu_shader_core_pipeline_opt;
  /* [한국어] "n_thread:warp_size" 형식의 파이프라인 설정 문자열.
   * gpgpusim.config: -gpgpu_shader_core_pipeline "<n_thread>:<warp_size>"
   * 설정자: reg_options()에서 등록, OptionParser 파싱 후 채워짐.
   * 읽는 자: init()에서 sscanf로 파싱해 n_thread_per_shader, warp_size 설정. */

  bool gpgpu_perfect_mem;
  /* [한국어] 완전 메모리 모드 활성화 여부 (레이턴시 없는 완벽한 메모리 시스템).
   * true이면 캐시/ICNT를 건너뛰고 바로 메모리 응답을 반환한다.
   * gpgpusim.config: -gpgpu_perfect_mem
   * 동기화: 읽기 전용. */

  bool gpgpu_clock_gated_reg_file;
  /* [한국어] 레지스터 파일 클럭 게이팅 활성화 여부.
   * true이면 비활성 레인의 레지스터 파일 접근 에너지를 AccelWattch에서 제외.
   * gpgpusim.config: -gpgpu_clock_gated_reg_file
   * 동기화: 읽기 전용. */

  bool gpgpu_clock_gated_lanes;
  /* [한국어] SIMD 레인 클럭 게이팅 활성화 여부.
   * true이면 비활성 스레드의 ALU 레인 에너지를 AccelWattch에서 제외.
   * gpgpusim.config: -gpgpu_clock_gated_lanes
   * 동기화: 읽기 전용. */

  enum divergence_support_t model;
  /* [한국어] SIMT 분기 지원 모델 타입.
   * POST_DOMINATOR 등 SIMT 스택 기반 분기 처리 방식을 결정.
   * gpgpusim.config: -gpgpu_simd_model
   * 동기화: 읽기 전용. */

  unsigned n_thread_per_shader;
  /* [한국어] SM당 최대 동시 실행 스레드 수.
   * gpgpu_shader_core_pipeline_opt에서 파싱된 첫 번째 값.
   * 값 범위: MAX_THREAD_PER_SM 이하. (예: 2048 for Volta)
   * 동기화: 읽기 전용 (init() 이후). */

  unsigned n_regfile_gating_group;
  /* [한국어] 레지스터 파일 게이팅 그룹 수.
   * 레지스터 파일을 몇 개의 그룹으로 나누어 게이팅할지 결정.
   * gpgpusim.config: -gpgpu_reg_file_port_throughput (일부)
   * 동기화: 읽기 전용. */

  unsigned max_warps_per_shader;
  /* [한국어] SM당 최대 동시 워프 수 = n_thread_per_shader / warp_size.
   * init()에서 계산된 파생 값.
   * 예: 2048 / 32 = 64 warps (Volta)
   * 동기화: 읽기 전용 (init() 이후). */

  unsigned
      max_cta_per_core;  // Limit on number of concurrent CTAs in shader core
  /* [한국어] SM당 최대 동시 CTA 수 (하드웨어 제한).
   * gpgpusim.config: -gpgpu_max_cta_per_core 또는 아키텍처별 기본값.
   * max_cta()에서도 자원(레지스터/공유메모리) 제한에 의한 값과 비교해 최솟값을 취함.
   * 동기화: 읽기 전용. */

  unsigned max_barriers_per_cta;
  /* [한국어] CTA당 최대 named barrier 수.
   * Kepler 이후 GPU에서 bar.sync/arrive 명령어로 특정 배리어 ID를 지정 가능.
   * gpgpusim.config: -gpgpu_max_barriers_per_cta
   * 동기화: 읽기 전용. */

  char *gpgpu_scheduler_string;
  /* [한국어] 워프 스케줄러 유형 설정 문자열.
   * "lrr", "gto", "two_level_active:N:P1:P2", "swl:P:N" 등.
   * gpgpusim.config: -gpgpu_scheduler
   * 읽는 자: create_schedulers()에서 스케줄러 인스턴스 생성 시 파싱. */

  unsigned gpgpu_shmem_per_block;
  /* [한국어] CTA당 최대 공유 메모리 크기 (바이트).
   * gpgpusim.config: -gpgpu_shmem_per_block
   * max_cta()에서 자원 제한 계산에 사용. */

  unsigned gpgpu_registers_per_block;
  /* [한국어] CTA당 최대 레지스터 수.
   * gpgpusim.config: -gpgpu_registers_per_block
   * max_cta()에서 레지스터 제한 계산에 사용. */

  char *pipeline_widths_string;
  /* [한국어] 파이프라인 단계별 너비 설정 문자열 (쉼표 구분).
   * gpgpusim.config: -gpgpu_pipeline_widths "N,N,N,...,N" (N_PIPELINE_STAGES개)
   * init()에서 파싱해 pipe_widths[] 배열에 저장. */

  int pipe_widths[N_PIPELINE_STAGES];
  /* [한국어] 파이프라인 각 단계의 너비 배열 (동시 처리 가능한 명령어 수).
   * pipeline_widths_string에서 파싱된 값.
   * 인덱스는 pipeline_stage_name_t 열거형 값에 대응.
   * 예: pipe_widths[ID_OC_SP] = 4 → ID_OC_SP 레지스터에 4개 슬롯.
   * 동기화: 읽기 전용 (init() 이후). */

  mutable cache_config m_L1I_config;
  /* [한국어] L1 명령어 캐시 설정 (크기, 연관도, 라인 크기 등).
   * gpgpusim.config: -gpgpu_cache:il1
   * mutable: const 함수에서도 캐시 설정을 조회할 수 있도록. */

  mutable cache_config m_L1T_config;
  /* [한국어] L1 텍스처 캐시 설정.
   * gpgpusim.config: -gpgpu_tex_cache:l1 */

  mutable cache_config m_L1C_config;
  /* [한국어] L1 상수 캐시 설정.
   * gpgpusim.config: -gpgpu_const_cache:l1 */

  mutable l1d_cache_config m_L1D_config;
  /* [한국어] L1 데이터 캐시 설정 (l1d_cache_config는 set_index_function 등 추가 기능 포함).
   * gpgpusim.config: -gpgpu_cache:dl1 */

  bool gpgpu_dwf_reg_bankconflict;
  /* [한국어] Dynamic Warp Formation 레지스터 뱅크 충돌 시뮬레이션 활성화 여부 (레거시).
   * 동기화: 읽기 전용. */

  unsigned gpgpu_num_sched_per_core;
  /* [한국어] SM당 워프 스케줄러 수 (서브코어 수와 같음).
   * gpgpusim.config: -gpgpu_num_sched_per_core
   * 예: 4 (Volta는 SM당 4개 스케줄러) */

  int gpgpu_max_insn_issue_per_warp;
  /* [한국어] 사이클당 워프당 최대 이슈 명령어 수.
   * gpgpusim.config: -gpgpu_max_insn_issue_per_warp
   * 1이면 단일 이슈, 2이면 듀얼 이슈 가능. */

  bool gpgpu_dual_issue_diff_exec_units;
  /* [한국어] 듀얼 이슈 시 서로 다른 실행 유닛으로만 이슈 허용 여부.
   * true이면 두 명령어가 같은 유닛(SP-SP)으로 동시 이슈 불가.
   * gpgpusim.config: -gpgpu_dual_issue_diff_exec_units */

  // op collector
  // [한국어] 오퍼랜드 컬렉터 설정 옵션들 — gpgpusim.config: -gpgpu_operand_collector_* 시리즈
  bool enable_specialized_operand_collector;
  /* [한국어] 실행 유닛별 특화 오퍼랜드 컬렉터 집합 활성화 여부.
   * false이면 단일 범용 컬렉터 집합 사용. */

  int gpgpu_operand_collector_num_units_sp;
  /* [한국어] SP 유닛 전용 오퍼랜드 컬렉터(CU) 수. */
  int gpgpu_operand_collector_num_units_dp;
  /* [한국어] DP 유닛 전용 오퍼랜드 컬렉터 수. */
  int gpgpu_operand_collector_num_units_sfu;
  /* [한국어] SFU 전용 오퍼랜드 컬렉터 수. */
  int gpgpu_operand_collector_num_units_tensor_core;
  /* [한국어] Tensor Core 전용 오퍼랜드 컬렉터 수. */
  int gpgpu_operand_collector_num_units_mem;
  /* [한국어] 메모리(ldst_unit) 전용 오퍼랜드 컬렉터 수. */
  int gpgpu_operand_collector_num_units_gen;
  /* [한국어] 범용(general) 오퍼랜드 컬렉터 수. */
  int gpgpu_operand_collector_num_units_int;
  /* [한국어] INT 유닛 전용 오퍼랜드 컬렉터 수. */

  unsigned int gpgpu_operand_collector_num_in_ports_sp;
  /* [한국어] SP 오퍼랜드 컬렉터 입력 포트 수 (ID_OC_SP 레지스터 너비). */
  unsigned int gpgpu_operand_collector_num_in_ports_dp;
  unsigned int gpgpu_operand_collector_num_in_ports_sfu;
  unsigned int gpgpu_operand_collector_num_in_ports_tensor_core;
  unsigned int gpgpu_operand_collector_num_in_ports_mem;
  unsigned int gpgpu_operand_collector_num_in_ports_gen;
  unsigned int gpgpu_operand_collector_num_in_ports_int;

  unsigned int gpgpu_operand_collector_num_out_ports_sp;
  /* [한국어] SP 오퍼랜드 컬렉터 출력 포트 수 (OC_EX_SP 레지스터 너비). */
  unsigned int gpgpu_operand_collector_num_out_ports_dp;
  unsigned int gpgpu_operand_collector_num_out_ports_sfu;
  unsigned int gpgpu_operand_collector_num_out_ports_tensor_core;
  unsigned int gpgpu_operand_collector_num_out_ports_mem;
  unsigned int gpgpu_operand_collector_num_out_ports_gen;
  unsigned int gpgpu_operand_collector_num_out_ports_int;

  unsigned int gpgpu_num_sp_units;
  /* [한국어] SM당 SP(FP32) 실행 유닛 수.
   * gpgpusim.config: -gpgpu_num_sp_units
   * 예: Volta SM당 32개 (워프 크기의 반). */
  unsigned int gpgpu_tensor_core_avail;
  /* [한국어] Tensor Core 유닛 활성화 여부 (0: 비활성, 1: 활성).
   * gpgpusim.config: -gpgpu_tensor_core_avail
   * 비활성화 시 파이프라인 레지스터 2개(ID_OC_TC, OC_EX_TC) 건너뜀. */
  unsigned int gpgpu_num_dp_units;
  /* [한국어] SM당 DP(FP64) 실행 유닛 수. */
  unsigned int gpgpu_num_sfu_units;
  /* [한국어] SM당 SFU 실행 유닛 수. */
  unsigned int gpgpu_num_tensor_core_units;
  /* [한국어] SM당 Tensor Core 유닛 수. */
  unsigned int gpgpu_num_mem_units;
  /* [한국어] SM당 메모리(ldst) 유닛 수 (현재는 항상 1). */
  unsigned int gpgpu_num_int_units;
  /* [한국어] SM당 INT ALU 유닛 수. */

  // Shader core resources
  // [한국어] SM 하드웨어 자원 설정 옵션들
  unsigned gpgpu_shader_registers;
  /* [한국어] SM당 총 레지스터 파일 크기 (레지스터 수).
   * gpgpusim.config: -gpgpu_shader_registers
   * max_cta()에서 레지스터 제한 계산에 사용. */
  int gpgpu_warpdistro_shader;
  /* [한국어] 워프 분포 통계를 출력할 SM ID (-1이면 모든 SM).
   * 디버그 목적. */
  int gpgpu_warp_issue_shader;
  /* [한국어] 워프 이슈 통계를 출력할 SM ID (-1이면 모든 SM). */
  unsigned gpgpu_num_reg_banks;
  /* [한국어] 레지스터 파일 뱅크 수.
   * gpgpusim.config: -gpgpu_num_reg_banks
   * opndcoll_rfu_t::init()에 전달됨. */
  bool gpgpu_reg_bank_use_warp_id;
  /* [한국어] 레지스터 뱅크 계산 시 warp ID 사용 여부 (true: warp ID 포함, false: 레지스터 번호만).
   * register_bank() 함수의 동작을 제어. */
  bool gpgpu_local_mem_map;
  /* [한국어] 로컬 메모리 주소 매핑 방식 설정.
   * gpgpusim.config: -gpgpu_local_mem_map */
  bool gpgpu_ignore_resources_limitation;
  /* [한국어] 자원 제한(레지스터/공유메모리) 무시 여부.
   * true이면 max_cta() 계산 시 자원 제한을 건너뜀 (이상적 시나리오 분석용). */
  bool sub_core_model;
  /* [한국어] 서브코어 모델 활성화 여부.
   * true이면 각 스케줄러(서브코어)가 전용 CU 집합과 레지스터 뱅크 파티션 사용.
   * gpgpusim.config: -gpgpu_sub_core_model */

  unsigned max_sp_latency;
  /* [한국어] SP(FP32) 유닛의 최대 파이프라인 레이턴시 (사이클).
   * set_pipeline_latency()에서 계산.
   * 읽는 자: pipelined_simd_unit 생성 시 max_latency 파라미터로 전달.
   * 동기화: 읽기 전용 (init() 이후). */
  unsigned max_int_latency;
  /* [한국어] INT ALU 유닛의 최대 파이프라인 레이턴시. */
  unsigned max_sfu_latency;
  /* [한국어] SFU(특수 함수 유닛)의 최대 파이프라인 레이턴시. */
  unsigned max_dp_latency;
  /* [한국어] DP(FP64) 유닛의 최대 파이프라인 레이턴시. */
  unsigned max_tensor_core_latency;
  /* [한국어] Tensor Core 유닛의 최대 파이프라인 레이턴시. */

  unsigned n_simt_cores_per_cluster;
  /* [한국어] 클러스터(TPC)당 SM 수.
   * gpgpusim.config: -gpgpu_n_cores_per_cluster
   * 읽는 자: sid_to_cluster(), sid_to_cid(), cid_to_sid() 등 ID 변환 함수.
   * 동기화: 읽기 전용. */
  unsigned n_simt_clusters;
  /* [한국어] 전체 클러스터(TPC) 수.
   * gpgpusim.config: -gpgpu_n_clusters
   * 읽는 자: num_shader() (= n_simt_clusters × n_simt_cores_per_cluster).
   * 동기화: 읽기 전용. */
  unsigned n_simt_ejection_buffer_size;
  /* [한국어] SM → 메모리 파티션 방향의 배출 버퍼(ejection buffer) 크기.
   * ICNT에서 SM으로 반환되는 메모리 응답 버퍼.
   * gpgpusim.config: -gpgpu_simt_core_sim_order (일부 설정).
   * 동기화: 읽기 전용. */
  unsigned ldst_unit_response_queue_size;
  /* [한국어] ldst_unit의 응답 FIFO 최대 크기 (메모리 응답 큐 한도).
   * response_buffer_full() 판단 기준으로 사용.
   * gpgpusim.config: -gpgpu_ldst_unit_response_queue
   * 동기화: 읽기 전용. */

  int simt_core_sim_order;
  /* [한국어] SM 시뮬레이션 순서 결정 방식 (0: 고정 순서, 1: 동적 순서).
   * gpgpusim.config: -gpgpu_simt_core_sim_order
   * 읽는 자: simt_core_cluster::core_cycle()에서 SM 순서 결정.
   * 동기화: 읽기 전용. */

  unsigned smem_latency;
  /* [한국어] 공유 메모리 접근 레이턴시 (사이클).
   * gpgpusim.config: -gpgpu_smem_latency
   * 읽는 자: ldst_unit::shared_cycle()에서 스톨 사이클 계산.
   * 동기화: 읽기 전용. */

  /*
   * [한국어] mem2device — 메모리 파티션 ID를 디바이스 ID로 변환한다.
   *
   * @param memid: 메모리 파티션의 로컬 인덱스 (0 ~ n_mem_partitions-1)
   * @return: ICNT 디바이스 ID (n_simt_clusters + memid)
   *
   * GPGPU-Sim에서 ICNT 디바이스 ID는 0 ~ n_simt_clusters-1이 SM 클러스터,
   * n_simt_clusters ~ 이후가 메모리 파티션이다.
   */
  unsigned mem2device(unsigned memid) const { return memid + n_simt_clusters; } // [한국어] 메모리 파티션 로컬 ID → ICNT 디바이스 ID 변환

  // Jin: concurrent kernel on sm
  bool gpgpu_concurrent_kernel_sm;
  /* [한국어] 하나의 SM에서 여러 커널을 동시에 실행하는 기능 활성화 여부.
   * true이면 SM이 비어있는 슬롯이 있을 때 다른 커널의 CTA를 받아들인다.
   * gpgpusim.config: -gpgpu_concurrent_kernel_sm
   * 동기화: 읽기 전용. */

  bool perfect_inst_const_cache;
  /* [한국어] 완전한 명령어/상수 캐시 시뮬레이션 여부.
   * true이면 L1I/L1C 캐시를 완벽한(레이턴시 0) 캐시로 모델링.
   * gpgpusim.config: -perfect_inst_const_cache
   * 동기화: 읽기 전용. */
  unsigned inst_fetch_throughput;
  /* [한국어] 사이클당 명령어 패치 처리량 (패치 가능한 명령어 수).
   * gpgpusim.config: -inst_fetch_throughput
   * 읽는 자: shader_core_ctx::fetch()에서 사이클당 패치 횟수 결정.
   * 동기화: 읽기 전용. */
  unsigned reg_file_port_throughput;
  /* [한국어] 레지스터 파일 포트 처리량 (사이클당 읽기/쓰기 가능한 오퍼랜드 수).
   * gpgpusim.config: -gpgpu_reg_file_port_throughput
   * 읽는 자: 오퍼랜드 컬렉터 뱅크 그룹 수 결정.
   * 동기화: 읽기 전용. */

  // specialized unit config strings
  char *specialized_unit_string[SPECIALIZED_UNIT_NUM];
  /* [한국어] 특수 목적 실행 유닛의 설정 문자열 배열 (최대 SPECIALIZED_UNIT_NUM개).
   * 각 원소는 "enabled,num_units,latency,id_oc_width,oc_ex_width,name" 형식.
   * gpgpusim.config: -specialized_unit "<설정>"
   * 읽는 자: init()에서 sscanf로 파싱해 m_specialized_unit 벡터에 저장.
   * 동기화: 읽기 전용 (init() 이후). */
  mutable std::vector<specialized_unit_params> m_specialized_unit;
  /* [한국어] 활성화된 특수 유닛의 파라미터 벡터.
   * 설정자: init()에서 enabled=1인 유닛을 push_back.
   * 읽는 자: create_exec_pipeline()에서 specialized_unit 인스턴스 생성.
   * mutable: const 함수에서도 참조 가능하도록.
   * 동기화: init() 이후 읽기 전용. */
  unsigned m_specialized_unit_num;
  /* [한국어] 전체 활성화된 특수 유닛 수 (m_specialized_unit의 num_units 합계).
   * 설정자: init()에서 각 유닛의 num_units를 누적 합산.
   * 읽는 자: create_exec_pipeline()에서 유닛 생성 시 사용.
   * 동기화: 읽기 전용 (init() 이후). */
};

/*
 * [한국어] shader_core_stats_pod — SM 통계 카운터의 POD(Plain Old Data) 집합
 *
 * === 구조체 역할 ===
 * SM별 및 전체 GPU 통계 카운터를 단순 포인터/정수 필드로 모아 놓은 POD 구조체이다.
 * shader_core_stats가 이를 상속하여 memset으로 일괄 0 초기화할 수 있게 한다.
 * 모든 포인터 필드는 calloc으로 num_shader() 크기의 배열을 가리키며,
 * SM ID를 인덱스로 사용한다 (예: m_num_sim_insn[sid]).
 *
 * === 전체 아키텍처에서의 위치 ===
 * shader_core_stats::shader_core_stats_pod_start를 통해 구조체 시작을 직접 접근한다.
 * AccelWattch 전력 카운터, ICNT 트래픽 통계, 캐시/메모리 접근 카운터를 포함한다.
 *
 * === 타 모듈과의 연결 ===
 * - 갱신자: shader_core_ctx의 incsp_stat() 등 인라인 stat 증가 함수들
 *           ldst_unit의 메모리 접근 카운터, simd_function_unit의 active_lanes_in_pipeline()
 * - 읽는 자: AccelWattch 전력 모델, gpgpu_sim의 최종 통계 출력
 */
struct shader_core_stats_pod {
  void *
      shader_core_stats_pod_start[0];  // DO NOT MOVE FROM THE TOP - spaceless
                                       // pointer to the start of this structure
  // [한국어] memset 초기화 시작점 마커 — shader_core_stats 생성자에서 이 포인터부터 sizeof(*pod)만큼 0으로 초기화

  unsigned long long *shader_cycles;
  /* [한국어] SM별 총 시뮬레이션 사이클 수 배열 (num_shader() 크기).
   * 설정자: shader_core_ctx::cycle()에서 매 사이클 shader_cycles[m_sid]++ 호출.
   * 읽는 자: 최종 통계 출력. */

  unsigned *m_num_sim_insn;   // number of scalar thread instructions committed
                              // by this shader core
  /* [한국어] SM별 커밋된 스칼라 스레드 명령어 수 (워프 내 모든 활성 스레드의 합).
   * 스레드 수준 IPC 계산에 사용.
   * 설정자: shader_core_ctx::warp_inst_complete() 시 active thread 수만큼 증가. */

  unsigned *m_num_sim_winsn;  // number of warp instructions committed by this
                              // shader core
  /* [한국어] SM별 커밋된 워프 명령어 수 (warp-level IPC 계산에 사용). */
  unsigned *m_last_num_sim_insn;
  /* [한국어] 이전 사이클의 m_num_sim_insn 값 (IPC 추세 계산용). */
  unsigned *m_last_num_sim_winsn;
  /* [한국어] 이전 사이클의 m_num_sim_winsn 값. */

  unsigned *
      m_num_decoded_insn;  // number of instructions decoded by this shader core
  /* [한국어] SM별 디코딩된 명령어 수. */
  float *m_pipeline_duty_cycle;
  /* [한국어] SM별 파이프라인 사용률 (0.0~1.0). AccelWattch 전력 계산에 사용. */
  unsigned *m_num_FPdecoded_insn;
  /* [한국어] SM별 디코딩된 FP 명령어 수. */
  unsigned *m_num_INTdecoded_insn;
  /* [한국어] SM별 디코딩된 INT 명령어 수. */
  unsigned *m_num_storequeued_insn;
  /* [한국어] SM별 스토어 큐에 들어간 명령어 수. */
  unsigned *m_num_loadqueued_insn;
  /* [한국어] SM별 로드 큐에 들어간 명령어 수. */
  unsigned *m_num_tex_inst;
  /* [한국어] SM별 텍스처 접근 명령어 수. */

  double *m_num_ialu_acesses;
  /* [한국어] SM별 INT ALU 접근 횟수 (AccelWattch에서 INT 동적 전력 계산). */
  double *m_num_fp_acesses;
  /* [한국어] SM별 FP 연산 접근 횟수. */
  double *m_num_imul_acesses;
  /* [한국어] SM별 INT 곱셈 접근 횟수. */
  double *m_num_fpmul_acesses;
  /* [한국어] SM별 FP 곱셈 접근 횟수. */
  double *m_num_idiv_acesses;
  /* [한국어] SM별 INT 나눗셈 접근 횟수. */
  double *m_num_fpdiv_acesses;
  /* [한국어] SM별 FP 나눗셈 접근 횟수. */
  double *m_num_sp_acesses;
  /* [한국어] SM별 SP(FP32) 유닛 접근 횟수. */
  double *m_num_sfu_acesses;
  /* [한국어] SM별 SFU 접근 횟수. */
  double *m_num_tensor_core_acesses;
  /* [한국어] SM별 Tensor Core 접근 횟수. */
  double *m_num_tex_acesses;
  /* [한국어] SM별 텍스처 캐시 접근 횟수. */
  double *m_num_const_acesses;
  /* [한국어] SM별 상수 캐시 접근 횟수. */
  double *m_num_dp_acesses;
  /* [한국어] SM별 DP(FP64) 접근 횟수. */
  double *m_num_dpmul_acesses;
  /* [한국어] SM별 DP 곱셈 접근 횟수. */
  double *m_num_dpdiv_acesses;
  /* [한국어] SM별 DP 나눗셈 접근 횟수. */
  double *m_num_sqrt_acesses;
  /* [한국어] SM별 제곱근(sqrt) 접근 횟수. SFU 담당. */
  double *m_num_log_acesses;
  /* [한국어] SM별 log 접근 횟수. SFU 담당. */
  double *m_num_sin_acesses;
  /* [한국어] SM별 sin 접근 횟수. SFU 담당. */
  double *m_num_exp_acesses;
  /* [한국어] SM별 exp 접근 횟수. SFU 담당. */
  double *m_num_mem_acesses;
  /* [한국어] SM별 메모리(ldst_unit) 접근 횟수. */

  unsigned *m_num_sp_committed;
  /* [한국어] SM별 SP 유닛에서 커밋된 명령어 수. */
  unsigned *m_num_tlb_hits;
  /* [한국어] SM별 TLB 히트 수. */
  unsigned *m_num_tlb_accesses;
  /* [한국어] SM별 TLB 접근 수. */
  unsigned *m_num_sfu_committed;
  /* [한국어] SM별 SFU에서 커밋된 명령어 수. */
  unsigned *m_num_tensor_core_committed;
  /* [한국어] SM별 Tensor Core에서 커밋된 명령어 수. */
  unsigned *m_num_mem_committed;
  /* [한국어] SM별 메모리 유닛에서 커밋된 명령어 수. */
  unsigned *m_read_regfile_acesses;
  /* [한국어] SM별 레지스터 파일 읽기 접근 수 (AccelWattch 전력 계산). */
  unsigned *m_write_regfile_acesses;
  /* [한국어] SM별 레지스터 파일 쓰기 접근 수. */
  unsigned *m_non_rf_operands;
  /* [한국어] SM별 레지스터 파일이 아닌 오퍼랜드 수 (즉치값, 상수 등). */
  double *m_num_imul24_acesses;
  /* [한국어] SM별 24비트 INT 곱셈 접근 횟수. */
  double *m_num_imul32_acesses;
  /* [한국어] SM별 32비트 INT 곱셈 접근 횟수. */

  unsigned *m_active_sp_lanes;
  /* [한국어] SM별 SP 유닛에서 활성 레인 수 누적 (AccelWattch 전력 모델). */
  unsigned *m_active_sfu_lanes;
  /* [한국어] SM별 SFU에서 활성 레인 수 누적. */
  unsigned *m_active_tensor_core_lanes;
  /* [한국어] SM별 Tensor Core에서 활성 레인 수 누적. */
  unsigned *m_active_fu_lanes;
  /* [한국어] SM별 모든 실행 유닛의 총 활성 레인 수 누적. */
  unsigned *m_active_fu_mem_lanes;
  /* [한국어] SM별 메모리 유닛의 활성 레인 수 누적. */
  double *m_active_exu_threads;  // For power model
  /* [한국어] SM별 실행 유닛에서 활성 스레드 수 누적 (AccelWattch 전력 모델). */
  double *m_active_exu_warps;    // For power model
  /* [한국어] SM별 실행 유닛에서 활성 워프 수 누적 (AccelWattch 전력 모델). */

  unsigned *m_n_diverge;  // number of divergence occurring in this shader
  /* [한국어] SM별 워프 분기(divergence) 발생 횟수.
   * SIMT 스택에서 분기가 일어날 때마다 증가. */

  unsigned gpgpu_n_load_insn;
  /* [한국어] GPU 전체 로드 명령어 수 (모든 SM 합산). */
  unsigned gpgpu_n_store_insn;
  /* [한국어] GPU 전체 스토어 명령어 수. */
  unsigned gpgpu_n_shmem_insn;
  /* [한국어] GPU 전체 공유 메모리 명령어 수. */
  unsigned gpgpu_n_sstarr_insn;
  /* [한국어] GPU 전체 구조화 공유 메모리 어레이 접근 명령어 수. */
  unsigned gpgpu_n_tex_insn;
  /* [한국어] GPU 전체 텍스처 접근 명령어 수. */
  unsigned gpgpu_n_const_insn;
  /* [한국어] GPU 전체 상수 메모리 접근 명령어 수. */
  unsigned gpgpu_n_param_insn;
  /* [한국어] GPU 전체 파라미터 메모리 접근 명령어 수. */
  unsigned gpgpu_n_shmem_bkconflict;
  /* [한국어] GPU 전체 공유 메모리 뱅크 충돌 횟수. */
  unsigned gpgpu_n_l1cache_bkconflict;
  /* [한국어] GPU 전체 L1 캐시 뱅크 충돌 횟수. */
  int gpgpu_n_intrawarp_mshr_merge;
  /* [한국어] GPU 전체 동일 워프 내 MSHR 병합 횟수. */
  unsigned gpgpu_n_cmem_portconflict;
  /* [한국어] GPU 전체 상수 메모리 포트 충돌 횟수. */

  unsigned gpu_stall_shd_mem_breakdown[N_MEM_STAGE_ACCESS_TYPE]
                                      [N_MEM_STAGE_STALL_TYPE];
  /* [한국어] 메모리 접근 유형 × 스톨 유형별 스톨 사이클 수 2D 배열.
   * 공유메모리/전역메모리/텍스처 각각에서 BANK_CONFLICT, WB_UNAVAIL 등 스톨 이유 분류.
   * 분석용: "왜 메모리 스톨이 발생했는가"를 정밀하게 보여준다. */

  unsigned gpu_reg_bank_conflict_stalls;
  /* [한국어] 레지스터 파일 뱅크 충돌로 인한 스톨 사이클 수 합계. */

  unsigned *shader_cycle_distro;
  /* [한국어] warp_size+3 크기의 사이클당 이슈 워프 수 분포 히스토그램.
   * shader_cycle_distro[n] = n개의 워프가 동시에 이슈된 사이클 수.
   * 설정자: shader_core_ctx::issue()에서 이슈된 워프 수마다 ++.
   * 읽는 자: 통계 출력 시. */
  unsigned *last_shader_cycle_distro;
  /* [한국어] 이전 출력 시점의 shader_cycle_distro 값 (incremental 출력용). */
  unsigned *num_warps_issuable;
  /* [한국어] 이슈 가능한 워프 수 분포 히스토그램 (num_sched_per_core 크기). */
  unsigned gpgpu_n_stall_shd_mem;
  /* [한국어] 메모리 스톨로 인한 총 사이클 수. */
  unsigned *single_issue_nums;
  /* [한국어] 스케줄러별 단일 이슈 횟수 배열 (gpgpu_num_sched_per_core 크기). */
  unsigned *dual_issue_nums;
  /* [한국어] 스케줄러별 듀얼 이슈 횟수 배열. */

  unsigned ctas_completed;
  /* [한국어] 이번 커널 실행에서 완료된 CTA 수 (커널 종료 감지에 사용). */

  // memory access classification
  // [한국어] 메모리 접근 유형별 카운터 — 디버그 및 성능 분석용
  int gpgpu_n_mem_read_local;
  /* [한국어] 로컬(스택) 메모리 읽기 접근 수. */
  int gpgpu_n_mem_write_local;
  /* [한국어] 로컬 메모리 쓰기 접근 수. */
  int gpgpu_n_mem_texture;
  /* [한국어] 텍스처 메모리 접근 수. */
  int gpgpu_n_mem_const;
  /* [한국어] 상수 메모리 접근 수. */
  int gpgpu_n_mem_read_global;
  /* [한국어] 전역 메모리 읽기 접근 수. */
  int gpgpu_n_mem_write_global;
  /* [한국어] 전역 메모리 쓰기 접근 수. */
  int gpgpu_n_mem_read_inst;
  /* [한국어] 명령어 패치(instruction fetch)를 위한 메모리 읽기 수. */

  int gpgpu_n_mem_l2_writeback;
  /* [한국어] L1D → L2 write-back 발생 횟수. */
  int gpgpu_n_mem_l1_write_allocate;
  /* [한국어] L1D write-allocate 정책에 의한 할당 횟수. */
  int gpgpu_n_mem_l2_write_allocate;
  /* [한국어] L2 write-allocate 발생 횟수. */

  unsigned made_write_mfs;
  /* [한국어] 생성된 쓰기 mem_fetch 패킷 수. */
  unsigned made_read_mfs;
  /* [한국어] 생성된 읽기 mem_fetch 패킷 수. */

  unsigned *gpgpu_n_shmem_bank_access;
  /* [한국어] SM별 공유 메모리 뱅크 접근 수 배열 (뱅크 충돌 분석용). */
  long *n_simt_to_mem;  // Interconnect power stats
  /* [한국어] SM → 메모리 파티션 방향 ICNT 트래픽 바이트 수 (전력 모델용). */
  long *n_mem_to_simt;
  /* [한국어] 메모리 파티션 → SM 방향 ICNT 트래픽 바이트 수 (전력 모델용). */
};

/*
 * [한국어] shader_core_stats — SM 통계 수집 클래스
 *
 * === 클래스 역할 ===
 * shader_core_stats_pod를 상속하여 모든 통계 카운터 배열을 동적 할당(calloc)하고 관리한다.
 * 생성자에서 num_shader() 크기의 배열을 SM당 하나씩 할당하고, 소멸자에서 free한다.
 * AccelWattch 전력 모델, 메모리 접근 통계, 실행 유닛 활성도 등 시뮬레이션 전체의 통계를 집계한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * gpgpu_sim 인스턴스가 소유한다. simt_core_cluster를 통해 각 SM이 참조한다.
 * shader_core_ctx, ldst_unit, simd_function_unit이 각 통계 포인터에 직접 접근해 카운터를 증가.
 *
 * === 타 모듈과의 연결 ===
 * - 갱신자: shader_core_ctx (inc*_stat 함수들), ldst_unit, sfu, sp_unit 등 각 실행 유닛
 * - 읽는 자: gpgpu_sim::print_stats(), AccelWattch 전력 모델
 *
 * === 주요 함수 요약 ===
 * - 생성자: num_shader()×각 카운터 배열을 calloc으로 할당
 * - ~shader_core_stats(): 모든 동적 배열 free
 * - event_warp_issued(): 워프 이슈 이벤트 기록 (warp_slot_issue_distro 갱신)
 * - visualizer_print(): 시각화 도구(AerialVision)용 통계 출력
 */
class shader_core_stats : public shader_core_stats_pod {
 public:
  /*
   * [한국어] shader_core_stats 생성자 — 모든 통계 배열을 calloc으로 동적 할당한다.
   *
   * @param config: SM 설정 (num_shader(), warp_size 등 크기 계산에 사용)
   *
   * shader_core_stats_pod 구조체 전체를 memset으로 0 초기화한 후,
   * 각 필드에 num_shader() 크기의 배열을 calloc으로 할당한다.
   *
   * 호출 체인:
   *   gpgpu_sim 생성자 → [shader_core_stats(config)]
   */
  shader_core_stats(const shader_core_config *config) {
    m_config = config; // [한국어] 설정 포인터 저장 (warp_size 등 참조에 사용)
    shader_core_stats_pod *pod = reinterpret_cast<shader_core_stats_pod *>(
        this->shader_core_stats_pod_start); // [한국어] POD 시작 포인터 획득 (spaceless zero-length array 트릭)
    memset(pod, 0, sizeof(shader_core_stats_pod)); // [한국어] POD 영역 전체를 0으로 초기화 (모든 포인터/카운터 리셋)
    // [한국어] 이하 각 필드마다 num_shader() 원소의 0-초기화 배열을 calloc으로 할당한다.
    // SM ID를 인덱스로 사용: 예) shader_cycles[m_sid]++
    shader_cycles = (unsigned long long *)calloc(config->num_shader(),
                                                 sizeof(unsigned long long)); // [한국어] SM별 총 사이클 카운터 배열
    m_num_sim_insn = (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 커밋 스칼라 명령어 수
    m_num_sim_winsn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 커밋 워프 명령어 수
    m_last_num_sim_winsn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] 이전 사이클 워프 명령어 수
    m_last_num_sim_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] 이전 사이클 스칼라 명령어 수
    m_pipeline_duty_cycle =
        (float *)calloc(config->num_shader(), sizeof(float)); // [한국어] SM별 파이프라인 사용률
    m_num_decoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 디코딩 명령어 수
    m_num_FPdecoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 FP 디코딩 명령어 수
    m_num_storequeued_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 스토어 큐 명령어 수
    m_num_loadqueued_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 로드 큐 명령어 수
    m_num_tex_inst = (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 텍스처 명령어 수
    m_num_INTdecoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 INT 디코딩 명령어 수
    m_num_ialu_acesses = (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 INT ALU 접근 횟수
    m_num_fp_acesses = (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 FP 접근 횟수
    m_num_imul_acesses = (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 INT 곱셈 접근
    m_num_imul24_acesses =
        (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 INT24 곱셈 접근
    m_num_imul32_acesses =
        (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 INT32 곱셈 접근
    m_num_fpmul_acesses =
        (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 FP 곱셈 접근
    m_num_idiv_acesses = (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 INT 나눗셈 접근
    m_num_fpdiv_acesses =
        (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 FP 나눗셈 접근
    m_num_dp_acesses = (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 DP(FP64) 접근
    m_num_dpmul_acesses =
        (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 DP 곱셈 접근
    m_num_dpdiv_acesses =
        (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 DP 나눗셈 접근
    m_num_sp_acesses = (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 SP(FP32) 접근
    m_num_sfu_acesses = (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 SFU 접근
    m_num_tensor_core_acesses =
        (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 Tensor Core 접근
    m_num_const_acesses =
        (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 상수 캐시 접근
    m_num_tex_acesses = (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 텍스처 접근
    m_num_sqrt_acesses = (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 sqrt 접근
    m_num_log_acesses = (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 log 접근
    m_num_sin_acesses = (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 sin 접근
    m_num_exp_acesses = (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 exp 접근
    m_num_mem_acesses = (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 메모리 유닛 접근
    m_num_sp_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 SP 커밋 명령어 수
    m_num_tlb_hits = (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 TLB 히트
    m_num_tlb_accesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 TLB 접근
    m_active_sp_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 SP 활성 레인 누적
    m_active_sfu_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 SFU 활성 레인 누적
    m_active_tensor_core_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 TC 활성 레인 누적
    m_active_fu_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 모든 FU 활성 레인 누적
    m_active_exu_threads =
        (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 실행 유닛 활성 스레드 수 (AccelWattch)
    m_active_exu_warps = (double *)calloc(config->num_shader(), sizeof(double)); // [한국어] SM별 실행 유닛 활성 워프 수
    m_active_fu_mem_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 메모리 FU 활성 레인 누적
    m_num_sfu_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 SFU 커밋 명령어 수
    m_num_tensor_core_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 TC 커밋 명령어 수
    m_num_mem_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 MEM 커밋 명령어 수
    m_read_regfile_acesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 레지스터 읽기 접근
    m_write_regfile_acesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 레지스터 쓰기 접근
    m_non_rf_operands =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 비 레지스터 오퍼랜드 수
    m_n_diverge = (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 분기 발생 횟수
    shader_cycle_distro =
        (unsigned *)calloc(config->warp_size + 3, sizeof(unsigned)); // [한국어] 이슈 워프 수 분포 히스토그램 (warp_size+3 크기)
    last_shader_cycle_distro =
        (unsigned *)calloc(m_config->warp_size + 3, sizeof(unsigned)); // [한국어] 이전 출력 시점의 분포 (incremental 계산용)
    single_issue_nums =
        (unsigned *)calloc(config->gpgpu_num_sched_per_core, sizeof(unsigned)); // [한국어] 스케줄러별 단일 이슈 횟수
    dual_issue_nums =
        (unsigned *)calloc(config->gpgpu_num_sched_per_core, sizeof(unsigned)); // [한국어] 스케줄러별 듀얼 이슈 횟수

    ctas_completed = 0; // [한국어] 커널 시작 시 완료 CTA 수 0으로 초기화
    n_simt_to_mem = (long *)calloc(config->num_shader(), sizeof(long)); // [한국어] SM→메모리 ICNT 트래픽 바이트
    n_mem_to_simt = (long *)calloc(config->num_shader(), sizeof(long)); // [한국어] 메모리→SM ICNT 트래픽 바이트

    m_outgoing_traffic_stats = new traffic_breakdown("coretomem"); // [한국어] SM → 메모리 방향 트래픽 통계 객체 생성
    m_incoming_traffic_stats = new traffic_breakdown("memtocore"); // [한국어] 메모리 → SM 방향 트래픽 통계 객체 생성

    gpgpu_n_shmem_bank_access =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned)); // [한국어] SM별 공유 메모리 뱅크 접근 수

    m_shader_dynamic_warp_issue_distro.resize(config->num_shader()); // [한국어] SM별 동적 워프 이슈 분포 벡터 크기 설정
    m_shader_warp_slot_issue_distro.resize(config->num_shader()); // [한국어] SM별 워프 슬롯 이슈 분포 벡터 크기 설정
  }

  /*
   * [한국어] ~shader_core_stats 소멸자 — 모든 동적 할당된 통계 배열을 해제한다.
   *
   * 생성자에서 calloc/new로 할당한 모든 포인터를 free/delete한다.
   * traffic_breakdown 객체는 delete, 나머지는 free로 해제한다.
   *
   * 호출 체인:
   *   gpgpu_sim 소멸자 → [~shader_core_stats()]
   */
  ~shader_core_stats() {
    delete m_outgoing_traffic_stats; // [한국어] core→mem 트래픽 통계 객체 해제
    delete m_incoming_traffic_stats; // [한국어] mem→core 트래픽 통계 객체 해제
    free(m_num_sim_insn); // [한국어] 이하 모든 calloc 배열 해제
    free(m_num_sim_winsn);
    free(m_num_FPdecoded_insn);
    free(m_num_INTdecoded_insn);
    free(m_num_storequeued_insn);
    free(m_num_loadqueued_insn);
    free(m_num_ialu_acesses);
    free(m_num_fp_acesses);
    free(m_num_imul_acesses);
    free(m_num_tex_inst);
    free(m_num_fpmul_acesses);
    free(m_num_idiv_acesses);
    free(m_num_fpdiv_acesses);
    free(m_num_sp_acesses);
    free(m_num_sfu_acesses);
    free(m_num_tensor_core_acesses);
    free(m_num_tex_acesses);
    free(m_num_const_acesses);
    free(m_num_dp_acesses);
    free(m_num_dpmul_acesses);
    free(m_num_dpdiv_acesses);
    free(m_num_sqrt_acesses);
    free(m_num_log_acesses);
    free(m_num_sin_acesses);
    free(m_num_exp_acesses);
    free(m_num_mem_acesses);
    free(m_num_sp_committed);
    free(m_num_tlb_hits);
    free(m_num_tlb_accesses);
    free(m_num_sfu_committed);
    free(m_num_tensor_core_committed);
    free(m_num_mem_committed);
    free(m_read_regfile_acesses);
    free(m_write_regfile_acesses);
    free(m_non_rf_operands);
    free(m_num_imul24_acesses);
    free(m_num_imul32_acesses);
    free(m_active_sp_lanes);
    free(m_active_sfu_lanes);
    free(m_active_tensor_core_lanes);
    free(m_active_fu_lanes);
    free(m_active_exu_threads);
    free(m_active_exu_warps);
    free(m_active_fu_mem_lanes);
    free(m_n_diverge);
    free(shader_cycle_distro);
    free(last_shader_cycle_distro);
  }

  /*
   * [한국어] new_grid — 새 커널 실행 시작 시 통계를 초기화한다 (현재 비어있음).
   * 향후 커널 경계 통계 리셋 로직이 들어올 수 있도록 인터페이스를 유지.
   */
  void new_grid() {}

  /*
   * [한국어] event_warp_issued — 워프 이슈 이벤트를 통계에 기록한다.
   *
   * @param s_id: SM ID
   * @param warp_id: 이슈된 워프의 슬롯 ID
   * @param num_issued: 이번 사이클에 이슈된 명령어 수 (1 또는 2)
   * @param dynamic_warp_id: 동적 워프 ID (통계 분포 기록에 사용)
   *
   * m_shader_dynamic_warp_issue_distro와 m_shader_warp_slot_issue_distro를 갱신한다.
   *
   * 호출 체인:
   *   scheduler_unit::cycle() → [event_warp_issued()]
   */
  void event_warp_issued(unsigned s_id, unsigned warp_id, unsigned num_issued,
                         unsigned dynamic_warp_id);

  /*
   * [한국어] visualizer_print — AerialVision 시각화 도구용 통계 데이터를 gzip 파일에 출력한다.
   * @param visualizer_file: gzip 압축 출력 파일 핸들
   */
  void visualizer_print(gzFile visualizer_file);

  /*
   * [한국어] print — 모든 SM 통계를 텍스트 형식으로 fout에 출력한다.
   * @param fout: 출력 파일 포인터 (stdout 또는 결과 파일)
   */
  void print(FILE *fout) const;

  /*
   * [한국어] get_dynamic_warp_issue — SM별 동적 워프 이슈 분포 벡터를 반환한다.
   * AccelWattch 또는 성능 분석 도구에서 사용.
   */
  const std::vector<std::vector<unsigned>> &get_dynamic_warp_issue() const {
    return m_shader_dynamic_warp_issue_distro; // [한국어] SM별 동적 워프 이슈 횟수 분포
  }

  /*
   * [한국어] get_warp_slot_issue — SM별 워프 슬롯 이슈 분포 벡터를 반환한다.
   */
  const std::vector<std::vector<unsigned>> &get_warp_slot_issue() const {
    return m_shader_warp_slot_issue_distro; // [한국어] SM별 워프 슬롯 이슈 횟수 분포
  }

 private:
  const shader_core_config *m_config;
  /* [한국어] SM 설정 포인터 (warp_size, num_shader() 등 크기 계산에 사용).
   * 설정자: 생성자에서 설정.
   * 동기화: 읽기 전용. */

  traffic_breakdown *m_outgoing_traffic_stats;  // core to memory partitions
  /* [한국어] SM → 메모리 파티션 방향 ICNT 트래픽 통계 객체.
   * 설정자: 생성자에서 new traffic_breakdown("coretomem")으로 생성.
   * 읽는 자: 최종 통계 출력 시. */
  traffic_breakdown *m_incoming_traffic_stats;  // memory partition to core
  /* [한국어] 메모리 파티션 → SM 방향 ICNT 트래픽 통계 객체. */

  // Counts the instructions issued for each dynamic warp.
  std::vector<std::vector<unsigned>> m_shader_dynamic_warp_issue_distro;
  /* [한국어] SM별 동적 워프 이슈 분포 (num_shader() × max_dynamic_warp_id 2D 벡터).
   * event_warp_issued()에서 [s_id][dynamic_warp_id]를 증가.
   * 동기화: 단일 스레드 접근. */
  std::vector<unsigned> m_last_shader_dynamic_warp_issue_distro;
  /* [한국어] 이전 출력 시점의 동적 워프 이슈 분포 (incremental 계산용). */
  std::vector<std::vector<unsigned>> m_shader_warp_slot_issue_distro;
  /* [한국어] SM별 워프 슬롯 이슈 분포 (이슈된 워프 슬롯 ID × 횟수). */
  std::vector<unsigned> m_last_shader_warp_slot_issue_distro;
  /* [한국어] 이전 출력 시점의 슬롯 이슈 분포. */

  // [한국어] friend 선언: 아래 클래스들이 private 멤버에 직접 접근 가능
  friend class power_stat_t;        // [한국어] AccelWattch 전력 통계 클래스
  friend class shader_core_ctx;     // [한국어] SM 구현 클래스
  friend class ldst_unit;           // [한국어] 로드/스토어 유닛
  friend class simt_core_cluster;   // [한국어] SM 클러스터 관리자
  friend class sst_simt_core_cluster; // [한국어] SST 연동 클러스터
  friend class scheduler_unit;      // [한국어] 워프 스케줄러
  friend class TwoLevelScheduler;   // [한국어] 두 단계 활성 워프 스케줄러
  friend class LooseRoundRobbinScheduler; // [한국어] 느슨한 라운드 로빈 스케줄러
};

class memory_config; // [한국어] 전방 선언 — shader_core_mem_fetch_allocator가 memory_config를 사용

/*
 * [한국어] shader_core_mem_fetch_allocator — SM용 mem_fetch 패킷 생성 팩토리
 *
 * === 클래스 역할 ===
 * ldst_unit이 메모리 접근 시 mem_fetch 패킷을 생성할 때 사용하는 팩토리 클래스이다.
 * mem_fetch_allocator 인터페이스를 구현하며, SM ID와 클러스터 ID를 포함한 패킷을 생성한다.
 * 생성된 mem_fetch는 ICNT를 통해 메모리 파티션으로 전달된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * ldst_unit 생성 시 주입되어, memory_cycle()에서 각 메모리 접근마다 alloc()을 호출한다.
 * 세 가지 alloc() 오버로드: 간단한 주소+타입, 상세 마스크 포함, warp_inst_t/mem_access_t 쌍.
 *
 * === 타 모듈과의 연결 ===
 * - 생성자: exec_shader_core_ctx::create_exec_pipeline()에서 new로 생성
 * - 사용자: ldst_unit::memory_cycle(), process_memory_access_queue_l1cache()
 * - 결과: mem_fetch 패킷 → m_icnt::push() → ICNT → 메모리 파티션
 */
class shader_core_mem_fetch_allocator : public mem_fetch_allocator {
 public:
  /*
   * [한국어] shader_core_mem_fetch_allocator 생성자 — SM ID, 클러스터 ID, 메모리 설정으로 초기화한다.
   *
   * @param core_id: 이 팩토리가 속한 SM의 로컬 ID (클러스터 내 인덱스)
   * @param cluster_id: 클러스터(TPC) ID
   * @param config: 메모리 계층 설정 (mem_fetch 생성에 필요)
   *
   * 호출 체인:
   *   exec_shader_core_ctx::create_exec_pipeline() → [shader_core_mem_fetch_allocator(...)]
   */
  shader_core_mem_fetch_allocator(unsigned core_id, unsigned cluster_id,
                                  const memory_config *config) {
    m_core_id = core_id;         // [한국어] 클러스터 내 SM 로컬 ID 저장
    m_cluster_id = cluster_id;   // [한국어] 클러스터(TPC) ID 저장
    m_memory_config = config;    // [한국어] 메모리 설정 포인터 저장
  }
  /*
   * [한국어] alloc (간단 버전) — 주소, 타입, 크기로 mem_fetch를 생성한다.
   *
   * @param addr: 접근 주소
   * @param type: 메모리 접근 유형 (GLOBAL_ACC_R, GLOBAL_ACC_W 등)
   * @param size: 패킷 크기 (바이트)
   * @param wr: 쓰기 여부
   * @param cycle: 현재 시뮬레이션 사이클
   * @param streamID: CUDA 스트림 ID
   * @return: 새로 생성된 mem_fetch 포인터
   *
   * 구현은 shader.cc에 있다.
   *
   * 호출 체인:
   *   ldst_unit::memory_cycle() → [alloc(addr, type, size, wr, cycle, streamID)]
   */
  mem_fetch *alloc(new_addr_type addr, mem_access_type type, unsigned size,
                   bool wr, unsigned long long cycle,
                   unsigned long long streamID) const;
  /*
   * [한국어] alloc (상세 버전) — 액티브 마스크/바이트 마스크/섹터 마스크를 포함한 mem_fetch를 생성한다.
   *
   * @param active_mask: 이 접근에 참여하는 스레드 마스크 (warp 내 활성 스레드 비트맵)
   * @param byte_mask: 접근되는 바이트 위치 마스크
   * @param sector_mask: 캐시 섹터 마스크 (sectored cache 지원)
   * @param wid: 워프 ID
   * @param sid: SM ID
   * @param tpc: TPC ID
   * @param original_mf: 이 접근을 유발한 원본 mem_fetch (응답 시 역추적에 사용)
   *
   * 구현은 shader.cc에 있다.
   */
  mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                   const active_mask_t &active_mask,
                   const mem_access_byte_mask_t &byte_mask,
                   const mem_access_sector_mask_t &sector_mask, unsigned size,
                   bool wr, unsigned long long cycle, unsigned wid,
                   unsigned sid, unsigned tpc, mem_fetch *original_mf,
                   unsigned long long streamID) const;
  /*
   * [한국어] alloc (warp_inst_t 버전) — 명령어와 메모리 접근 정보로 mem_fetch를 생성한다.
   *
   * @param inst: mem_fetch를 유발한 warp_inst_t (PC, 워프 ID, 레지스터 정보 포함)
   * @param access: mem_access_t (주소, 마스크, 크기 등 포함)
   * @param cycle: 현재 사이클
   * @return: 새로 생성된 mem_fetch 포인터
   *
   * 이 버전은 인라인으로 구현되어 있으며 mem_fetch 생성자를 직접 호출한다.
   * WRITE_PACKET_SIZE / READ_PACKET_SIZE로 쓰기/읽기 패킷 크기를 결정한다.
   *
   * 호출 체인:
   *   ldst_unit::process_memory_access_queue_l1cache() → [alloc(inst, access, cycle)]
   */
  mem_fetch *alloc(const warp_inst_t &inst, const mem_access_t &access,
                   unsigned long long cycle) const {
    warp_inst_t inst_copy = inst; // [한국어] inst의 사본 생성 (mem_fetch가 소유하므로 복사 필요)
    mem_fetch *mf = new mem_fetch(
        access, &inst_copy, inst.get_streamID(),
        access.is_write() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, // [한국어] 쓰기: 데이터 포함 패킷, 읽기: 헤더만
        inst.warp_id(), m_core_id, m_cluster_id, m_memory_config, cycle); // [한국어] 워프/SM/클러스터/메모리설정/사이클 기록
    return mf; // [한국어] 소유권을 호출자(ldst_unit)에게 넘김 — 호출자가 delete 책임
  }

 private:
  unsigned m_core_id;
  /* [한국어] 클러스터 내 이 SM의 로컬 인덱스.
   * mem_fetch 생성 시 소스 SM 식별에 사용.
   * 동기화: 읽기 전용. */
  unsigned m_cluster_id;
  /* [한국어] 이 SM이 속한 클러스터(TPC) ID.
   * mem_fetch 생성 시 소스 클러스터 식별에 사용.
   * 동기화: 읽기 전용. */
  const memory_config *m_memory_config;
  /* [한국어] 메모리 계층 설정 포인터 (L2/DRAM 파라미터 포함).
   * mem_fetch 생성자에 전달되어 라우팅 결정에 사용.
   * 동기화: 읽기 전용. */
};

/*
 * [한국어] shader_core_ctx — SM(Streaming Multiprocessor) 타이밍 시뮬레이션 구현 클래스
 *
 * === 클래스 역할 ===
 * GPGPU-Sim의 SM을 사이클-정확도(cycle-accurate)로 시뮬레이션하는 핵심 클래스이다.
 * 모든 SM 파이프라인 단계(Fetch → Decode → Issue/OC → Execute → Writeback)를 구현하며,
 * 워프 스케줄러, 오퍼랜드 컬렉터, 실행 유닛, 스코어보드, 배리어, L1 캐시를 통합 관리한다.
 * 사이클마다 cycle() 함수가 호출되어 각 파이프라인 단계를 순서대로 실행한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * simt_core_cluster가 소유하며, gpgpu_sim::cycle()에서 클러스터를 통해 이 클래스의 cycle()을 호출.
 * 파이프라인: fetch() → decode() → issue() → execute() → writeback()
 * 메모리 요청은 ldst_unit → ICNT → memory_partition_unit → DRAM → ICNT → accept_ldst_unit_response()로 순환.
 *
 * === 타 모듈과의 연결 ===
 * - 상위: simt_core_cluster (m_core[] 배열로 소유)
 * - 스케줄러: m_schedulers (scheduler_unit 하위 클래스들)
 * - 오퍼랜드 컬렉터: m_operand_collector (opndcoll_rfu_t)
 * - 실행 유닛: m_fu 벡터 (sp_unit, sfu, dp_unit, tensor_core, int_unit, ldst_unit)
 * - 스코어보드: m_scoreboard (Scoreboard)
 * - 배리어: m_barriers (barrier_set_t)
 * - 파이프라인 레지스터: m_pipeline_reg[] (register_set, pipeline_stage_name_t 인덱스)
 *
 * === 주요 함수 요약 ===
 * - cycle(): 매 사이클 파이프라인 진행 (writeback→execute→issue→decode→fetch 역순 실행)
 * - issue_block2core(): 클러스터에서 CTA를 이 SM에 할당
 * - fetch(): 워프의 다음 명령어를 L1I 캐시에서 패치
 * - decode(): 패치된 명령어를 워프별 명령어 버퍼에 디코딩
 * - issue(): 워프 스케줄러가 ID_OC 레지스터로 명령어를 이슈
 * - execute(): 실행 유닛들의 cycle()을 호출하고 OC_EX → 실행
 * - writeback(): EX_WB 레지스터의 완료 명령어를 스코어보드에서 해제
 */
class shader_core_ctx : public core_t {
 public:
  /*
   * [한국어] shader_core_ctx 생성자 — SM을 초기화하고 파이프라인 자원을 생성한다.
   *
   * @param gpu: 최상위 시뮬레이터 포인터
   * @param cluster: 이 SM이 속한 simt_core_cluster
   * @param shader_id: 전역 SM ID
   * @param tpc_id: TPC(Texture Processing Cluster) ID
   * @param config: SM 설정
   * @param mem_config: 메모리 계층 설정
   * @param stats: 통계 수집 객체
   *
   * 내부에서 create_front_pipeline(), create_schedulers(), create_exec_pipeline()을 호출한다.
   *
   * 호출 체인:
   *   simt_core_cluster 생성자 → [shader_core_ctx(...)]
   */
  // creator:
  shader_core_ctx(class gpgpu_sim *gpu, class simt_core_cluster *cluster,
                  unsigned shader_id, unsigned tpc_id,
                  const shader_core_config *config,
                  const memory_config *mem_config, shader_core_stats *stats);

  // used by simt_core_cluster:
  // modifiers
  /*
   * [한국어] cycle — SM 파이프라인의 1 사이클을 진행한다.
   *
   * 역순으로(writeback → execute → issue → decode → fetch) 각 단계를 실행한다.
   * 이 순서는 새 명령어가 이전 단계의 결과를 바로 사용하는 것을 방지한다 (스냅샷 시뮬레이션).
   * simt_core_cluster::core_cycle()에서 매 사이클 호출된다.
   *
   * 호출 체인:
   *   simt_core_cluster::core_cycle() → [cycle()] → writeback()→execute()→issue()→decode()→fetch()
   */
  void cycle();
  /*
   * [한국어] reinit — SM을 새 CTA 실행을 위해 초기화한다.
   *
   * @param start_thread: 초기화할 스레드 슬롯 시작 인덱스
   * @param end_thread: 초기화할 스레드 슬롯 끝 인덱스
   * @param reset_not_completed: m_not_completed 카운터를 리셋할지 여부
   *
   * 호출 체인:
   *   issue_block2core() → [reinit(start, end, false)]
   */
  void reinit(unsigned start_thread, unsigned end_thread,
              bool reset_not_completed);
  /*
   * [한국어] issue_block2core — 커널의 다음 CTA를 이 SM에 할당한다.
   *
   * @param kernel: CTA를 제공하는 커널 정보
   *
   * 빈 워프 슬롯을 찾아 CTA의 스레드를 할당하고, 배리어와 스코어보드를 초기화한다.
   * 기능 시뮬레이션(cuda-sim)과 타이밍 시뮬레이션 모두에서 필요한 초기화를 수행한다.
   *
   * 호출 체인:
   *   simt_core_cluster::issue_block2core() → [issue_block2core(kernel)]
   */
  void issue_block2core(class kernel_info_t &kernel);

  /*
   * [한국어] cache_flush — 이 SM의 모든 L1 캐시(L1D/L1T/L1C)를 플러시한다.
   * 커널 종료 또는 cuCtxSynchronize() 시 호출된다.
   */
  void cache_flush();
  /*
   * [한국어] cache_invalidate — 이 SM의 모든 L1 캐시를 무효화한다.
   * 캐시 일관성 이벤트 또는 explicit 무효화 명령 시 호출된다.
   */
  void cache_invalidate();
  /*
   * [한국어] accept_fetch_response — L1I 캐시 패치 응답을 수신한다.
   *
   * @param mf: 완료된 명령어 패치 요청 패킷
   *
   * m_inst_fetch_buffer를 클리어하여 다음 사이클에 새 패치 요청을 허용한다.
   *
   * 호출 체인:
   *   simt_core_cluster::icnt_pop_llc_ejection_buffer() → [accept_fetch_response(mf)]
   */
  void accept_fetch_response(mem_fetch *mf);
  /*
   * [한국어] accept_ldst_unit_response — 메모리 계층에서 로드/스토어 응답을 수신한다.
   *
   * @param mf: 완료된 메모리 요청 패킷
   *
   * ldst_unit::fill()을 호출하여 m_response_fifo에 추가한다.
   *
   * 호출 체인:
   *   simt_core_cluster::icnt_pop_llc_ejection_buffer() → [accept_ldst_unit_response(mf)]
   */
  void accept_ldst_unit_response(class mem_fetch *mf);
  /*
   * [한국어] broadcast_barrier_reduction — 배리어에 모든 워프가 도달했을 때 해당 워프들을 깨운다.
   *
   * @param cta_id: 배리어가 속한 CTA ID
   * @param bar_id: 해제된 배리어 ID
   * @param warps: 깨울 워프들의 비트셋
   *
   * m_barriers에서 waiting 비트를 클리어하여 스케줄러가 다시 이 워프들을 이슈할 수 있게 한다.
   *
   * 호출 체인:
   *   barrier_set_t::warp_reaches_barrier() → [broadcast_barrier_reduction()]
   */
  void broadcast_barrier_reduction(unsigned cta_id, unsigned bar_id,
                                   warp_set_t warps);
  /*
   * [한국어] set_kernel — 이 SM이 실행할 커널을 설정한다.
   *
   * @param k: 실행할 커널 정보 포인터 (assert로 null 체크)
   *
   * m_kernel 포인터를 설정하고 디버그 메시지를 출력한다.
   *
   * 호출 체인:
   *   simt_core_cluster::issue_block2core() → [set_kernel(k)]
   */
  void set_kernel(kernel_info_t *k) {
    assert(k); // [한국어] null 커널 포인터 방어
    m_kernel = k; // [한국어] 이 SM에 할당된 커널 포인터 저장
    //        k->inc_running();
    printf("GPGPU-Sim uArch: Shader %d bind to kernel %u \'%s\'\n", m_sid,
           m_kernel->get_uid(), m_kernel->name().c_str()); // [한국어] SM-커널 바인딩 디버그 출력
  }
  PowerscalingCoefficients *scaling_coeffs;
  /* [한국어] AccelWattch 전력 스케일링 계수 포인터.
   * 각 실행 유닛의 전력 소비 계산에 사용되는 하드웨어별 계수.
   * 설정자: accelwattch 초기화 시. */

  // accessors
  /*
   * [한국어] fetch_unit_response_buffer_full — L1I 패치 응답 버퍼가 가득 찼는지 확인한다.
   * @return: m_inst_fetch_buffer.m_valid가 true이면 true (이미 응답 대기 중)
   */
  bool fetch_unit_response_buffer_full() const;
  /*
   * [한국어] ldst_unit_response_buffer_full — ldst_unit 응답 버퍼가 가득 찼는지 확인한다.
   * @return: m_ldst_unit->response_buffer_full()의 결과
   */
  bool ldst_unit_response_buffer_full() const;
  /*
   * [한국어] get_not_completed — 완료되지 않은 스레드 수를 반환한다.
   * @return: m_not_completed (0이면 이 SM의 모든 스레드 완료)
   */
  unsigned get_not_completed() const { return m_not_completed; }
  /*
   * [한국어] get_n_active_cta — 현재 이 SM에서 실행 중인 CTA 수를 반환한다.
   * @return: m_n_active_cta
   */
  unsigned get_n_active_cta() const { return m_n_active_cta; }
  /*
   * [한국어] isactive — 이 SM에 활성 CTA가 있는지 확인한다.
   * @return: m_n_active_cta > 0이면 1, 아니면 0
   */
  unsigned isactive() const {
    if (m_n_active_cta > 0) // [한국어] 활성 CTA가 있으면
      return 1; // [한국어] 활성 상태
    else
      return 0; // [한국어] 유휴 상태
  }
  /*
   * [한국어] get_kernel — 이 SM에 현재 할당된 커널 포인터를 반환한다.
   * @return: m_kernel (NULL이면 커널 없음)
   */
  kernel_info_t *get_kernel() { return m_kernel; }
  /*
   * [한국어] get_sid — 이 SM의 전역 ID를 반환한다.
   * @return: m_sid
   */
  unsigned get_sid() const { return m_sid; }

  // used by functional simulation:
  // modifiers
  /*
   * [한국어] warp_exit — 워프가 실행을 완료했음을 기록한다.
   *
   * @param warp_id: 완료된 워프의 슬롯 ID
   *
   * m_not_completed를 감소시키고 m_barriers.warp_exit()로 배리어 상태를 갱신한다.
   * 기능 시뮬레이션(cuda-sim)에서도 호출된다.
   *
   * 호출 체인:
   *   PTX 기능 시뮬레이션 완료 → [warp_exit(warp_id)] / writeback() 완료 시
   */
  virtual void warp_exit(unsigned warp_id);

  // Ni: Unset ldgdepbar
  /*
   * [한국어] unset_depbar — LDGDEPBAR 의존성 배리어를 해제한다.
   *
   * @param inst: LDGDEPBAR 명령어 (의존성 배리어 해제 대상)
   *
   * Volta 이후 GPU의 소프트웨어 의존성 배리어(LDGDEPBAR/DEPBAR) 지원.
   * 비동기 메모리 복사 완료 감지에 사용된다.
   */
  void unset_depbar(const warp_inst_t &inst);

  // accessors
  /*
   * [한국어] warp_waiting_at_barrier — 워프가 __syncthreads() 배리어에서 대기 중인지 확인한다.
   * @param warp_id: 확인할 워프 슬롯 ID
   * @return: m_barriers.warp_waiting_at_barrier(warp_id)의 결과
   */
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const;
  /*
   * [한국어] get_pdom_stack_top_info — 특정 스레드의 PDOM 스택 최상위(PC, RPC)를 반환한다.
   * @param tid: 스레드 ID
   * @param pc: 현재 실행 PC 출력
   * @param rpc: 재합류(reconvergence) PC 출력
   */
  void get_pdom_stack_top_info(unsigned tid, unsigned *pc, unsigned *rpc) const;
  /*
   * [한국어] get_current_occupancy — 이 SM의 현재 점유율을 반환한다.
   * @param active: 활성 슬롯 수 출력
   * @param total: 전체 슬롯 수 출력
   * @return: 현재 SM 점유율 (0.0 ~ 1.0)
   */
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const;

  // used by pipeline timing model components:
  // modifiers
  /*
   * [한국어] mem_instruction_stats — 메모리 명령어 유형별 전역 통계 카운터를 갱신한다.
   * @param inst: 처리 중인 메모리 명령어
   * gpgpu_n_load_insn, gpgpu_n_store_insn 등을 증가시킨다.
   */
  void mem_instruction_stats(const warp_inst_t &inst);
  /*
   * [한국어] decrement_atomic_count — 특정 워프의 원자적 연산 카운터를 감소시킨다.
   * @param wid: 워프 ID
   * @param n: 감소 값
   * m_warp[wid]->dec_atomic_count(n) 호출.
   */
  void decrement_atomic_count(unsigned wid, unsigned n);
  /*
   * [한국어] inc_store_req — 특정 워프의 미완료 스토어 요청 수를 증가시킨다.
   * @param warp_id: 워프 ID
   * 메모리 유닛에서 스토어를 발행할 때 호출.
   */
  void inc_store_req(unsigned warp_id) { m_warp[warp_id]->inc_store_req(); } // [한국어] 해당 워프의 미완료 스토어 카운터 증가
  /*
   * [한국어] dec_inst_in_pipeline — 파이프라인 내 명령어 수를 감소시킨다.
   * @param warp_id: 워프 ID
   * writeback() 완료 시 또는 warp_inst_complete()에서 호출.
   */
  void dec_inst_in_pipeline(unsigned warp_id) {
    m_warp[warp_id]->dec_inst_in_pipeline(); // [한국어] 해당 워프의 파이프라인 내 명령어 수 감소
  }  // also used in writeback()
  /*
   * [한국어] store_ack — 스토어 명령어의 메모리 확인(acknowledgment)을 처리한다.
   * @param mf: 완료된 스토어 mem_fetch 패킷
   * m_warp[wid]->dec_store_req()로 미완료 스토어 카운터를 감소시킨다.
   * 호출 체인: accept_ldst_unit_response() → ldst_unit::fill() → [store_ack(mf)]
   */
  void store_ack(class mem_fetch *mf);
  /*
   * [한국어] warp_waiting_at_mem_barrier — 워프가 메모리 배리어(membar)에서 대기 중인지 확인한다.
   * @param warp_id: 확인할 워프 슬롯 ID
   * @return: 해당 워프에 완료되지 않은 스토어가 있으면 true (메모리 순서 보장을 위해 대기)
   */
  bool warp_waiting_at_mem_barrier(unsigned warp_id);
  /*
   * [한국어] set_max_cta — 커널의 자원 요구사항으로 이 SM의 최대 CTA 수를 계산하고 설정한다.
   * @param kernel: 실행할 커널 정보
   */
  void set_max_cta(const kernel_info_t &kernel);
  /*
   * [한국어] warp_inst_complete — 워프의 명령어 실행이 완료됨을 기록한다.
   * @param inst: 완료된 명령어
   * m_num_sim_insn, m_num_sim_winsn 등 통계를 갱신한다.
   */
  void warp_inst_complete(const warp_inst_t &inst);

  // accessors
  /*
   * [한국어] get_regs_written — 주어진 명령어가 쓰는 레지스터 목록을 반환한다.
   * @param fvt: 검사할 명령어 (inst_t)
   * @return: 이 명령어가 기록하는 레지스터 번호 목록
   */
  std::list<unsigned> get_regs_written(const inst_t &fvt) const;
  /*
   * [한국어] get_config — 이 SM의 설정 포인터를 반환한다.
   * @return: m_config (shader_core_config*)
   */
  const shader_core_config *get_config() const { return m_config; }
  /*
   * [한국어] print_cache_stats — L1D 캐시 통계를 출력한다.
   * @param fp: 출력 파일
   * @param dl1_accesses: 접근 수 출력
   * @param dl1_misses: 미스 수 출력
   */
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses);

  /*
   * [한국어] get_cache_stats — 이 SM의 모든 캐시 통계를 수집한다.
   */
  void get_cache_stats(cache_stats &cs);
  /*
   * [한국어] get_L1I/D/C/T_sub_stats — L1 캐시 종류별 상세 통계를 수집한다.
   * simt_core_cluster::get_cache_stats() 집계 시 호출된다.
   */
  void get_L1I_sub_stats(struct cache_sub_stats &css) const;
  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;

  /*
   * [한국어] get_icnt_power_stats — ICNT 전력 통계를 반환한다.
   * @param n_simt_to_mem: SM→메모리 트래픽 출력
   * @param n_mem_to_simt: 메모리→SM 트래픽 출력
   */
  void get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;

  // debug:
  /*
   * [한국어] display_simt_state — SIMT 스택 상태를 fout에 출력한다 (디버깅용).
   */
  void display_simt_state(FILE *fout, int mask) const;
  /*
   * [한국어] display_pipeline — SM 파이프라인 레지스터 상태를 fout에 출력한다 (디버깅용).
   */
  void display_pipeline(FILE *fout, int print_mem, int mask3bit) const;

  // [한국어] inc*_stat 함수 시리즈: AccelWattch 전력 모델을 위한 통계 카운터 증가 함수들.
  // 각 함수는 해당 실행 유닛의 접근 횟수와 활성 스레드/워프 수를 갱신한다.
  // clock_gated_lanes가 false이면 비활성 레인의 에너지도 추가로 계산한다.
  // 패턴: m_stats->m_num_*[m_sid] += active_count * latency (+ 비활성 레인 기여분)
  //        m_stats->m_active_exu_threads[m_sid] += active_count
  //        m_stats->m_active_exu_warps[m_sid]++

  void incload_stat() { m_stats->m_num_loadqueued_insn[m_sid]++; } // [한국어] 로드 큐 명령어 수 증가
  void incstore_stat() { m_stats->m_num_storequeued_insn[m_sid]++; } // [한국어] 스토어 큐 명령어 수 증가
  // [한국어] 이하 각 inc*_stat 함수는 AccelWattch 전력 카운터를 갱신한다.
  // active_count: 이 사이클에 활성화된 스레드(레인) 수 (0~32)
  // latency: 해당 유닛의 파이프라인 레이턴시 (사이클)
  // clock_gated_lanes=false이면 비활성 레인의 에너지도 inactive_lanes_accesses_*()로 추가 계산

  void incialu_stat(unsigned active_count, double latency) { // [한국어] INT ALU 전력 카운터 갱신
    if (m_config->gpgpu_clock_gated_lanes == false) { // [한국어] 클럭 게이팅 비활성: 비활성 레인 에너지도 포함
      m_stats->m_num_ialu_acesses[m_sid] =
          m_stats->m_num_ialu_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency); // [한국어] non-SFU 비활성 레인 기여 추가
    } else { // [한국어] 클럭 게이팅 활성: 활성 레인 에너지만
      m_stats->m_num_ialu_acesses[m_sid] =
          m_stats->m_num_ialu_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count; // [한국어] 활성 스레드 수 누적
    m_stats->m_active_exu_warps[m_sid]++; // [한국어] 활성 워프 수 누적
  }
  void incimul_stat(unsigned active_count, double latency) { // [한국어] INT 곱셈 전력 카운터 갱신
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_imul_acesses[m_sid] =
          m_stats->m_num_imul_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_imul_acesses[m_sid] =
          m_stats->m_num_imul_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incimul24_stat(unsigned active_count, double latency) { // [한국어] INT24 곱셈 전력 카운터 갱신
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_imul24_acesses[m_sid] =
          m_stats->m_num_imul24_acesses[m_sid] +
          (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_imul24_acesses[m_sid] =
          m_stats->m_num_imul24_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incimul32_stat(unsigned active_count, double latency) { // [한국어] INT32 곱셈 전력 카운터 갱신 (SFU 형 비활성 레인 계산)
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_imul32_acesses[m_sid] =
          m_stats->m_num_imul32_acesses[m_sid] +
          (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency); // [한국어] SFU 형 비활성 레인 기여
    } else {
      m_stats->m_num_imul32_acesses[m_sid] =
          m_stats->m_num_imul32_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incidiv_stat(unsigned active_count, double latency) { // [한국어] INT 나눗셈 전력 카운터 갱신
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_idiv_acesses[m_sid] =
          m_stats->m_num_idiv_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_idiv_acesses[m_sid] =
          m_stats->m_num_idiv_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incfpalu_stat(unsigned active_count, double latency) { // [한국어] FP ALU 전력 카운터 갱신
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_fp_acesses[m_sid] =
          m_stats->m_num_fp_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_fp_acesses[m_sid] =
          m_stats->m_num_fp_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incfpmul_stat(unsigned active_count, double latency) { // [한국어] FP 곱셈 전력 카운터 갱신
    // printf("FP MUL stat increament\n");
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_fpmul_acesses[m_sid] =
          m_stats->m_num_fpmul_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_fpmul_acesses[m_sid] =
          m_stats->m_num_fpmul_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incfpdiv_stat(unsigned active_count, double latency) { // [한국어] FP 나눗셈 전력 카운터 갱신
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_fpdiv_acesses[m_sid] =
          m_stats->m_num_fpdiv_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_fpdiv_acesses[m_sid] =
          m_stats->m_num_fpdiv_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incdpalu_stat(unsigned active_count, double latency) { // [한국어] DP(FP64) ALU 전력 카운터 갱신
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_dp_acesses[m_sid] =
          m_stats->m_num_dp_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_dp_acesses[m_sid] =
          m_stats->m_num_dp_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incdpmul_stat(unsigned active_count, double latency) { // [한국어] DP 곱셈 전력 카운터 갱신
    // printf("FP MUL stat increament\n");
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_dpmul_acesses[m_sid] =
          m_stats->m_num_dpmul_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_dpmul_acesses[m_sid] =
          m_stats->m_num_dpmul_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incdpdiv_stat(unsigned active_count, double latency) { // [한국어] DP 나눗셈 전력 카운터 갱신
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_dpdiv_acesses[m_sid] =
          m_stats->m_num_dpdiv_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_dpdiv_acesses[m_sid] =
          m_stats->m_num_dpdiv_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void incsqrt_stat(unsigned active_count, double latency) { // [한국어] sqrt 전력 카운터 갱신 (SFU)
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_sqrt_acesses[m_sid] =
          m_stats->m_num_sqrt_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_sqrt_acesses[m_sid] =
          m_stats->m_num_sqrt_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void inclog_stat(unsigned active_count, double latency) { // [한국어] log 전력 카운터 갱신 (SFU)
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_log_acesses[m_sid] =
          m_stats->m_num_log_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_log_acesses[m_sid] =
          m_stats->m_num_log_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void incexp_stat(unsigned active_count, double latency) { // [한국어] exp 전력 카운터 갱신 (SFU)
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_exp_acesses[m_sid] =
          m_stats->m_num_exp_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_exp_acesses[m_sid] =
          m_stats->m_num_exp_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void incsin_stat(unsigned active_count, double latency) { // [한국어] sin 전력 카운터 갱신 (SFU)
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_sin_acesses[m_sid] =
          m_stats->m_num_sin_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_sin_acesses[m_sid] =
          m_stats->m_num_sin_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void inctensor_stat(unsigned active_count, double latency) { // [한국어] Tensor Core 전력 카운터 갱신
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_tensor_core_acesses[m_sid] =
          m_stats->m_num_tensor_core_acesses[m_sid] +
          (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_tensor_core_acesses[m_sid] =
          m_stats->m_num_tensor_core_acesses[m_sid] +
          (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void inctex_stat(unsigned active_count, double latency) { // [한국어] 텍스처 캐시 접근 전력 카운터 갱신
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_tex_acesses[m_sid] =
          m_stats->m_num_tex_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_tex_acesses[m_sid] =
          m_stats->m_num_tex_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void inc_const_accesses(unsigned active_count) { // [한국어] 상수 캐시 접근 수 증가
    m_stats->m_num_const_acesses[m_sid] =
        m_stats->m_num_const_acesses[m_sid] + active_count;
  }

  void incsfu_stat(unsigned active_count, double latency) { // [한국어] SFU 접근 에너지 카운터 갱신
    m_stats->m_num_sfu_acesses[m_sid] =
        m_stats->m_num_sfu_acesses[m_sid] + (double)active_count * latency;
  }
  void incsp_stat(unsigned active_count, double latency) { // [한국어] SP(FP32) 접근 에너지 카운터 갱신
    m_stats->m_num_sp_acesses[m_sid] =
        m_stats->m_num_sp_acesses[m_sid] + (double)active_count * latency;
  }
  void incmem_stat(unsigned active_count, double latency) { // [한국어] 메모리 유닛 접근 에너지 카운터 갱신
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_mem_acesses[m_sid] =
          m_stats->m_num_mem_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_mem_acesses[m_sid] =
          m_stats->m_num_mem_acesses[m_sid] + (double)active_count * latency;
    }
  }
  /*
   * [한국어] incexecstat — 실행 명령어 유형별 통계 카운터를 갱신한다.
   * @param inst: 완료된 명령어
   * m_num_sp_committed, m_num_sfu_committed 등을 명령어 유형에 따라 증가.
   * 구현은 shader.cc에 있다.
   */
  void incexecstat(warp_inst_t *&inst);

  void incregfile_reads(unsigned active_count) { // [한국어] 레지스터 파일 읽기 접근 수 증가
    m_stats->m_read_regfile_acesses[m_sid] =
        m_stats->m_read_regfile_acesses[m_sid] + active_count;
  }
  void incregfile_writes(unsigned active_count) { // [한국어] 레지스터 파일 쓰기 접근 수 증가
    m_stats->m_write_regfile_acesses[m_sid] =
        m_stats->m_write_regfile_acesses[m_sid] + active_count;
  }
  void incnon_rf_operands(unsigned active_count) { // [한국어] 비 레지스터 오퍼랜드 수 증가
    m_stats->m_non_rf_operands[m_sid] =
        m_stats->m_non_rf_operands[m_sid] + active_count;
  }

  void incspactivelanes_stat(unsigned active_count) { // [한국어] SP 활성 레인 수 누적 (AccelWattch)
    m_stats->m_active_sp_lanes[m_sid] =
        m_stats->m_active_sp_lanes[m_sid] + active_count;
  }
  void incsfuactivelanes_stat(unsigned active_count) { // [한국어] SFU 활성 레인 수 누적
    m_stats->m_active_sfu_lanes[m_sid] =
        m_stats->m_active_sfu_lanes[m_sid] + active_count;
  }
  void incfuactivelanes_stat(unsigned active_count) { // [한국어] 모든 FU 활성 레인 수 누적
    m_stats->m_active_fu_lanes[m_sid] =
        m_stats->m_active_fu_lanes[m_sid] + active_count;
  }
  void incfumemactivelanes_stat(unsigned active_count) { // [한국어] 메모리 FU 활성 레인 수 누적
    m_stats->m_active_fu_mem_lanes[m_sid] =
        m_stats->m_active_fu_mem_lanes[m_sid] + active_count;
  }

  void inc_simt_to_mem(unsigned n_flits) { // [한국어] SM→메모리 방향 ICNT 트래픽 플릿 수 누적
    m_stats->n_simt_to_mem[m_sid] += n_flits;
  }
  /*
   * [한국어] check_if_non_released_reduction_barrier — reduction 배리어가 아직 해제되지 않았는지 확인한다.
   * @param inst: 검사할 배리어 명령어
   * @return: 미해제 reduction 배리어가 있으면 true (스케줄러가 이 워프를 이슈하지 않도록)
   */
  bool check_if_non_released_reduction_barrier(warp_inst_t &inst);

 protected:
  /*
   * [한국어] inactive_lanes_accesses_sfu — SFU 형 비활성 레인 에너지 기여를 계산한다.
   * @param active_count: 활성 스레드 수 (0~32)
   * @param latency: 유닛 레이턴시 (사이클)
   * @return: 비활성 레인의 에너지 기여 (32-active 기준, 3-term 합산)
   *
   * SFU(Special Function Unit)는 warp 내 비활성 레인도 일부 에너지를 소모한다.
   * AccelWattch 모델에서 비활성 레인의 스위칭 활동을 추정하는 휴리스틱.
   * (32-n)>>1, (32-n)>>3, (32-n)>>3 세 항의 합으로 계산.
   *
   * 호출 체인: incimul32_stat/incidiv_stat/incfpdiv_stat/incsqrt_stat 등 → [이 함수]
   */
  unsigned inactive_lanes_accesses_sfu(unsigned active_count, double latency) {
    return (((32 - active_count) >> 1) * latency) + // [한국어] 비활성 레인의 1/2 기여 (주 항)
           (((32 - active_count) >> 3) * latency) + // [한국어] 비활성 레인의 1/8 기여 (보조 항)
           (((32 - active_count) >> 3) * latency);  // [한국어] 비활성 레인의 1/8 기여 (보조 항 반복)
  }
  /*
   * [한국어] inactive_lanes_accesses_nonsfu — non-SFU 형 비활성 레인 에너지 기여를 계산한다.
   * @param active_count: 활성 스레드 수
   * @param latency: 유닛 레이턴시
   * @return: 비활성 레인의 에너지 기여 (1/2 단항만)
   *
   * INT ALU, FP ALU, FPMUL 등 non-SFU 유닛은 비활성 레인 에너지가 SFU보다 작다.
   * (32-n)>>1 하나만 사용하는 간소화 모델.
   *
   * 호출 체인: incialu_stat/incfpalu_stat/incfpmul_stat 등 → [이 함수]
   */
  unsigned inactive_lanes_accesses_nonsfu(unsigned active_count,
                                          double latency) {
    return (((32 - active_count) >> 1) * latency); // [한국어] 비활성 레인 1/2 에너지 기여
  }

  /*
   * [한국어] test_res_bus — 지정된 레이턴시 슬롯의 결과 버스(result bus)가 비어 있는지 검사한다.
   * @param latency: 검사할 파이프라인 레이턴시 슬롯 인덱스
   * @return: 해당 슬롯이 비어 있으면 1, 이미 점유됐으면 0
   *
   * result bus는 FU 실행 결과가 WB 단계로 전달되는 파이프라인 레지스터.
   * m_result_bus 비트셋에서 해당 사이클 슬롯의 점유 여부를 검사한다.
   * 구현: shader.cc
   *
   * 호출 체인: issue_warp() → [이 함수]
   */
  int test_res_bus(int latency);
  /*
   * [한국어] next_pc — 워프의 특정 스레드 다음 실행 PC를 반환한다.
   * @param tid: 하드웨어 스레드 id (전역 번호)
   * @return: 해당 스레드의 다음 PC (PTX 시뮬레이션에서 ptx_thread_info를 통해 조회)
   *
   * fetch() 단계에서 워프의 다음 명령어 주소를 결정할 때 사용.
   * 구현: shader.cc
   */
  address_type next_pc(int tid) const;
  /*
   * [한국어] fetch — L1 명령어 캐시(m_L1I)에서 워프 명령어를 인출한다.
   * 워프 스케줄러가 선택한 워프의 PC로 m_L1I에 접근, 캐시 히트 시 즉시 디코딩 단계로,
   * 미스 시 m_inst_fetch_buffer에 펜딩하여 ICNT를 통해 메모리에서 명령어를 가져온다.
   * GPU 코어 클럭 1 사이클에 한 번 호출됨(cycle() 내부).
   *
   * 호출 체인: cycle() → [이 함수] → m_L1I->access(), ICNT
   */
  void fetch();
  /*
   * [한국어] register_cta_thread_exit — CTA(블록) 내 스레드 하나가 종료될 때 호출되어,
   * 해당 CTA의 완료 여부를 추적하고 모든 스레드가 끝나면 CTA 자원을 해제한다.
   * @param cta_num: 종료되는 CTA의 하드웨어 슬롯 번호 (0~MAX_CTA_PER_SHADER-1)
   * @param kernel: 해당 CTA가 속한 커널 정보
   *
   * m_cta_status와 m_n_active_cta를 갱신. CTA가 완료되면 자원(shmem, regs)을 릴리즈.
   *
   * 호출 체인: writeback()/checkExecutionStatusAndUpdate() → [이 함수]
   */
  void register_cta_thread_exit(unsigned cta_num, kernel_info_t *kernel);

  /*
   * [한국어] decode — IF 단계에서 인출된 명령어를 디코딩하여 스코어보드 검사 후
   * ID_OC 파이프라인 레지스터로 발행한다.
   * fetch()가 m_inst_fetch_buffer에 넣은 명령어 집합(워프 단위)을 m_pipeline_reg[ID_OC_*]로 전달.
   * 구현: shader.cc
   *
   * 호출 체인: cycle() → [이 함수] → scoreboard, m_pipeline_reg
   */
  void decode();

  /*
   * [한국어] issue — 디코딩된 명령어를 실행 유닛으로 이슈한다.
   * 각 워프 스케줄러가 우선순위 큐를 순회하며 실행 준비된 워프를 선택하고
   * 해당 FU의 파이프라인 입력 레지스터로 명령어를 밀어 넣는다.
   * 구현: shader.cc
   *
   * 호출 체인: cycle() → [이 함수] → issue_warp()
   */
  void issue();
  friend class scheduler_unit;  // [한국어] scheduler_unit이 issue_warp() private 메서드를 호출해야 하므로 friend 선언
  friend class TwoLevelScheduler;  // [한국어] 2-level 스케줄러도 issue_warp() 접근 필요
  friend class LooseRoundRobbinScheduler; // [한국어] LooseRoundRobbin 스케줄러도 동일
  /*
   * [한국어] issue_warp — 선택된 워프의 명령어를 파이프라인 레지스터에 발행한다.
   * @param warp: 발행 대상 파이프라인 레지스터 집합 (OC_EX_SP 등)
   * @param pI: 발행할 명령어 (warp_inst_t)
   * @param active_mask: 이 발행에서 활성화된 스레드 마스크
   * @param warp_id: 발행하는 워프의 슬롯 id
   * @param sch_id: 발행하는 워프 스케줄러 id (sub-core model에서 사용)
   *
   * 스코어보드에 레지스터를 예약하고, 파이프라인 레지스터에 명령어를 기록.
   * virtual: exec/trace 드라이버가 서로 다른 initializtion 로직을 포함할 수 있음.
   *
   * 호출 체인: issue() → scheduler_unit::cycle() → [이 함수] → scoreboard->reserveRegisters()
   */
  virtual void issue_warp(register_set &warp, const warp_inst_t *pI,
                          const active_mask_t &active_mask, unsigned warp_id,
                          unsigned sch_id);

  /*
   * [한국어] create_front_pipeline — IF/ID 파이프라인 레지스터를 초기화한다.
   * m_pipeline_reg 벡터를 구성하고, L1I 캐시를 할당.
   * 서브클래스 생성자에서 호출된다.
   */
  void create_front_pipeline();
  /*
   * [한국어] create_schedulers — gpgpusim.config의 스케줄러 설정에 따라
   * lrr/rrr/gto/oldest/two_level_active/swl 중 하나를 선택하여 m_schedulers 벡터를 채운다.
   * sub-core model 시 스케줄러 수 = warp 수 / warp-per-scheduler.
   */
  void create_schedulers();
  /*
   * [한국어] create_exec_pipeline — 실행 유닛(SP/SFU/DP/INT/MEM 등)과
   * result bus를 생성하여 m_fu, m_ldst_unit, m_result_bus를 초기화한다.
   */
  void create_exec_pipeline();

  // pure virtual methods implemented based on the current execution mode
  // (execution-driven vs trace-driven)
  /*
   * [한국어] init_warps — CTA가 코어에 할당될 때 워프 슬롯을 초기화한다.
   * @param cta_id: 하드웨어 CTA 슬롯
   * @param start_thread: 이 CTA의 첫 스레드 hw id
   * @param end_thread: 이 CTA의 마지막 스레드 hw id + 1
   * @param ctaid: 소프트웨어 CTA id
   * @param cta_size: 이 CTA의 스레드 수
   * @param kernel: 커널 정보
   *
   * 실행 드라이버(exec/trace)에 따라 오버라이드:
   * - exec: ptx 스레드 초기화 (sim_init_thread)
   * - trace: 트레이스 파일에서 워프 PC 로드
   */
  virtual void init_warps(unsigned cta_id, unsigned start_thread,
                          unsigned end_thread, unsigned ctaid, int cta_size,
                          kernel_info_t &kernel);
  /*
   * [한국어] checkExecutionStatusAndUpdate — 명령어 실행 완료 후 스레드 완료 여부를 확인한다.
   * @param inst: 방금 실행된 명령어
   * @param t: 워프 내 스레드 위치 (lane index)
   * @param tid: 하드웨어 스레드 id
   *
   * = 0: 순수 가상 — exec/trace 모드에서 각각 다르게 구현.
   * exec: ptx_thread_info의 상태를 확인해 EXIT 여부 판단
   */
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) = 0;
  /*
   * [한국어] func_exec_inst — 명령어의 기능 시뮬레이션을 실행한다.
   * @param inst: 실행할 명령어
   *
   * = 0: 순수 가상.
   * exec 모드: ptx_thread_info::ptx_exec_inst() 호출 → PTX 시맨틱 실행
   * trace 모드: 트레이스 정보 기반으로 레지스터 기록 (실제 연산 없음)
   */
  virtual void func_exec_inst(warp_inst_t &inst) = 0;

  /*
   * [한국어] sim_init_thread — 워프 내 단일 스레드의 기능 시뮬레이션 컨텍스트를 초기화한다.
   * @param kernel: 초기화할 커널
   * @param thread_info: 생성된 ptx_thread_info 포인터를 반환받을 이중 포인터
   * @param sid: 셰이더(SM) id
   * @param tid: 하드웨어 스레드 id
   * @param threads_left, num_threads: CTA 내 남은/전체 스레드 수
   * @param core: 코어 참조 (core_t*)
   * @param hw_cta_id, hw_warp_id: 하드웨어 CTA/warp 슬롯 id
   * @param gpu: 최상위 GPU 시뮬레이터 객체
   * @return: 초기화된 스레드 수 (0 이면 실패 또는 범위 초과)
   *
   * = 0: 순수 가상. exec 모드에서만 PTX 스레드 컨텍스트를 실제로 생성한다.
   */
  virtual unsigned sim_init_thread(kernel_info_t &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   gpgpu_t *gpu) = 0;

  /*
   * [한국어] create_shd_warp — 실행 모드에 맞는 shd_warp_t 파생 객체를 생성하여 m_warp를 채운다.
   * = 0: 순수 가상. exec 모드: exec_warp_inst_t 래퍼, trace 모드: trace_warp_inst_t 래퍼.
   */
  virtual void create_shd_warp() = 0;

  /*
   * [한국어] get_next_inst — 지정된 워프의 PC에 해당하는 다음 명령어를 가져온다.
   * @param warp_id: 대상 워프 슬롯 id
   * @param pc: 현재 PC
   * @return: 해당 PC의 warp_inst_t 포인터 (null 이면 페치 대기)
   *
   * exec 모드: PTX 명령어 객체를 PTX IR에서 조회
   * trace 모드: 미리 로드된 트레이스 버퍼에서 명령어 반환
   */
  virtual const warp_inst_t *get_next_inst(unsigned warp_id,
                                           address_type pc) = 0;
  /*
   * [한국어] get_pdom_stack_top_info — SIMT 스택 최상단의 PC/RPC를 조회한다.
   * @param warp_id: 대상 워프 슬롯
   * @param pI: 현재 명령어 (분기 정보 포함)
   * @param pc: [out] 현재 실행 PC
   * @param rpc: [out] reconvergence PC (합류 지점)
   *
   * warp divergence 분석 및 fetch 제어에 사용.
   */
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc) = 0;
  /*
   * [한국어] get_active_mask — 주어진 워프와 명령어에 대한 활성 스레드 마스크를 반환한다.
   * @param warp_id: 대상 워프
   * @param pI: 현재 명령어
   * @return: 32비트 active_mask_t — 각 비트가 스레드 레인의 활성 여부를 나타냄
   *
   * decode/issue 단계에서 어떤 레인이 실제로 연산해야 하는지를 결정한다.
   */
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI) = 0;

  /*
   * [한국어] translate_local_memaddr — 로컬(private) 메모리 주소를 전역 메모리 공간의 주소로 변환한다.
   * @param localaddr: PTX local 주소 공간 내 주소
   * @param tid: 하드웨어 스레드 id
   * @param num_shader: 총 SM 수
   * @param datasize: 접근 크기 (바이트)
   * @param translated_addrs: [out] 변환된 전역 주소 배열 (coalescing 전 각 스레드 주소)
   * @return: 변환된 주소 개수
   *
   * GPU 로컬 메모리는 실제로 전역 DRAM에 스레드별 분리 영역으로 배치된다.
   * 각 스레드의 로컬 주소를 (tid, SM id)를 인덱스로 하는 전역 주소로 매핑.
   *
   * 호출 체인: ldst_unit::memory_cycle() → [이 함수]
   */
  // Returns numbers of addresses in translated_addrs
  unsigned translate_local_memaddr(address_type localaddr, unsigned tid,
                                   unsigned num_shader, unsigned datasize,
                                   new_addr_type *translated_addrs);

  /*
   * [한국어] read_operands — Operand Collector(opndcoll_rfu_t)를 통해 레지스터 오퍼랜드를 읽는다.
   * OC_EX 파이프라인 레지스터로 명령어가 전달되기 전에, 레지스터 파일 뱅크 충돌을 처리하고
   * 필요한 모든 소스 레지스터를 수집한다. 수집이 완료된 Collector Unit(CU)을
   * arbiter가 실행 유닛으로 이슈한다.
   *
   * 호출 체인: cycle() → [이 함수] → m_operand_collector.step()
   */
  void read_operands();

  /*
   * [한국어] execute — 각 실행 유닛(m_fu)의 파이프라인을 1 사이클 진행한다.
   * m_fu 벡터의 각 simd_function_unit::cycle()을 순서대로 호출하고,
   * 유닛 완료 결과를 result bus를 통해 WB 단계로 전달한다.
   * ldst_unit의 경우 메모리 요청 생성 및 반환 처리도 수행한다.
   *
   * 호출 체인: cycle() → [이 함수] → simd_function_unit::cycle(), ldst_unit::cycle()
   */
  void execute();

  /*
   * [한국어] writeback — 실행 완료된 명령어를 레지스터 파일에 기록(writeback)한다.
   * m_result_bus 비트셋에서 이번 사이클에 완료된 명령어를 찾아 스코어보드의
   * 예약을 해제하고, 레지스터에 결과를 반영한다.
   * 기능 시뮬레이션(func_exec_inst)은 이미 execute() 시점에 완료된 상태이므로,
   * writeback은 타이밍 모델 상의 WB 단계를 처리한다.
   *
   * 호출 체인: cycle() → [이 함수] → scoreboard->releaseRegisters()
   */
  void writeback();

  // used in display_pipeline():
  /*
   * [한국어] dump_warp_state — 디버그용: 현재 모든 워프의 상태(PC, 활성 마스크 등)를 출력한다.
   */
  void dump_warp_state(FILE *fout) const;
  /*
   * [한국어] print_stage — 디버그용: 지정된 파이프라인 단계의 파이프라인 레지스터 내용을 출력한다.
   */
  void print_stage(unsigned int stage, FILE *fout) const;

  unsigned long long m_last_inst_gpu_sim_cycle;
  /* [한국어] 마지막으로 명령어가 발행된 GPU 시뮬레이션 사이클 번호.
   * 설정자: issue_warp()에서 워프가 이슈될 때마다 갱신.
   * 읽는 자: 통계 출력, 파이프라인 디버그.
   * 값 범위: 0 이상 (단조 증가, 커널 단위로 리셋되지 않음).
   * 동기화: 단일 코어 사이클 루프에서만 접근 — 별도 락 불필요. */
  unsigned long long m_last_inst_gpu_tot_sim_cycle;
  /* [한국어] 마지막으로 명령어가 발행된 GPU 총 시뮬레이션 사이클 번호 (모든 커널 누적).
   * 설정자: issue_warp().
   * 읽는 자: 통계/디버그.
   * 값 범위: 0 이상 (누적 단조 증가).
   * 동기화: 단일 코어 사이클 루프. */

  // general information
  unsigned m_sid;  // shader id
  /* [한국어] 이 SM의 전역 셰이더 id (0 ~ num_shader-1).
   * 설정자: 생성자(shader_core_ctx constructor)에서 shader_id 파라미터로 초기화.
   * 읽는 자: 통계 배열 인덱스(m_stats->m_num_sp_acesses[m_sid]), mem_fetch 생성 시 출처 표시.
   * 값 범위: 0 이상 (config에 의해 결정됨).
   * 동기화: 초기화 후 읽기 전용 — 락 불필요. */
  unsigned m_tpc;  // texture processor cluster id (aka, node id when using
                   // interconnect concentration)
  /* [한국어] 이 SM이 속한 TPC(Texture Processor Cluster) id.
   * 클러스터(simt_core_cluster) 내 위치를 나타내며, ICNT 라우팅 시 소스 노드 id로 사용된다.
   * 설정자: 생성자에서 tpc_id 파라미터로 초기화.
   * 읽는 자: mem_fetch 생성 시 m_mem_fetch_allocator, ICNT 패킷 전송 시 소스 id.
   * 값 범위: 0 ~ (num_cluster - 1).
   * 동기화: 초기화 후 읽기 전용. */
  const shader_core_config *m_config;
  /* [한국어] 이 SM의 모든 설정 파라미터를 담고 있는 불변 설정 객체 포인터.
   * gpgpusim.config에서 파싱된 값들(warp 수, L1I 크기, 스케줄러 타입 등)이 들어있다.
   * 설정자: 생성자 파라미터로 전달, 이후 변경 없음.
   * 읽는 자: 파이프라인 각 단계, 통계 함수, 스케줄러 생성 등 거의 모든 멤버 함수.
   * 값 범위: 유효한 포인터 (NULL 불가).
   * 동기화: 읽기 전용. */
  const memory_config *m_memory_config;
  /* [한국어] 메모리 시스템 설정(L1D/L2/DRAM/NoC 파라미터)을 담은 불변 설정 포인터.
   * ldst_unit, 캐시 생성 시 참조된다.
   * 설정자: 생성자 파라미터로 전달.
   * 읽는 자: ldst_unit 생성자, cache 초기화.
   * 값 범위: 유효한 포인터 (NULL 불가).
   * 동기화: 읽기 전용. */
  class simt_core_cluster *m_cluster;
  /* [한국어] 이 SM이 속한 시뮬레이터 클러스터 객체.
   * SM이 메모리 응답을 수신하거나 캐시 무효화가 필요할 때 클러스터와 상호작용.
   * 설정자: 생성자 파라미터로 전달.
   * 읽는 자: accept_ldst_unit_response(), cache_flush() 등에서 클러스터 API 호출 시.
   * 동기화: 읽기 전용 포인터, 접근 시 클러스터 내부 락 적용. */

  // statistics
  shader_core_stats *m_stats;
  /* [한국어] 이 SM의 모든 성능 통계 카운터를 담고 있는 동적 할당 객체.
   * inc*_stat 함수들이 직접 이 객체의 배열(인덱스 m_sid)을 갱신한다.
   * 설정자: 생성자 파라미터로 전달(시뮬레이터 전체에서 단일 객체).
   * 읽는 자: visualizer_print(), print(), AccelWattch 전력 계산.
   * 동기화: 단일 코어 사이클 루프에서만 접근(m_sid 슬롯은 이 SM 전용). */

  // CTA scheduling / hardware thread allocation
  unsigned m_n_active_cta;  // number of Cooperative Thread Arrays (blocks)
                            // currently running on this shader.
  /* [한국어] 현재 이 SM에서 실행 중인 CTA(스레드 블록) 수.
   * 설정자: issue_block2core()에서 블록 할당 시 증가, register_cta_thread_exit()에서 완료 시 감소.
   * 읽는 자: can_issue_1block()으로 새 블록 수용 가능 여부 판단 시, done_cycle() 조건 판단.
   * 값 범위: 0 ~ m_config->max_cta().
   * 동기화: 단일 코어 사이클 루프. */
  unsigned m_cta_status[MAX_CTA_PER_SHADER];  // CTAs status
  /* [한국어] 각 CTA 슬롯의 활성 스레드 수를 추적하는 배열.
   * index: 하드웨어 CTA 슬롯 번호 (0 ~ MAX_CTA_PER_SHADER-1).
   * 값: 해당 슬롯에 남아 있는 미완료 스레드 수. 0이 되면 CTA 완료.
   * 설정자: issue_block2core()에서 초기값(스레드 수)로 설정.
   *         register_cta_thread_exit()에서 스레드 완료 시 감소.
   * 읽는 자: register_cta_thread_exit().
   * 동기화: 단일 코어 사이클 루프. */
  unsigned m_not_completed;  // number of threads to be completed (==0 when all
                             // thread on this core completed)
  /* [한국어] 이 SM에서 아직 완료되지 않은 스레드의 총 수.
   * 설정자: issue_block2core()에서 CTA 할당 시 증가, writeback/checkExecution에서 스레드 종료 시 감소.
   * 읽는 자: done_cycle() — 0이 되면 이 SM의 모든 실행이 완료됨을 의미.
   * 값 범위: 0 ~ m_config->n_thread_per_shader.
   * 동기화: 단일 코어 사이클 루프. */
  std::bitset<MAX_THREAD_PER_SM> m_active_threads;
  /* [한국어] 현재 이 SM에서 활성화된 스레드의 비트마스크.
   * 비트 i: 하드웨어 스레드 i가 현재 실행 중(1) / 완료(0).
   * 설정자: issue_block2core()에서 CTA 할당 시 관련 비트 set, 스레드 종료 시 reset.
   * 읽는 자: 스케줄러, is_thread_done() 류 보조 함수.
   * 동기화: 단일 코어 사이클 루프. */

  // thread contexts
  thread_ctx_t *m_threadState;
  /* [한국어] 이 SM의 모든 하드웨어 스레드 컨텍스트 배열 (크기: n_thread_per_shader).
   * 각 엔트리: thread_ctx_t — 스레드의 완료 여부, 커널 포인터 등 경량 상태.
   * 상세 기능 시뮬레이션 상태는 exec 모드에서 ptx_thread_info가 별도 관리.
   * 설정자: 생성자에서 calloc, issue_block2core()에서 CTA 할당 시 각 스레드 초기화.
   * 읽는 자: checkExecutionStatusAndUpdate(), register_cta_thread_exit().
   * 동기화: 단일 코어 사이클 루프. */

  // interconnect interface
  mem_fetch_interface *m_icnt;
  /* [한국어] SM에서 메모리 시스템(L2 캐시/DRAM)으로 메모리 요청을 전송하는 ICNT 인터페이스.
   * mem_fetch_interface의 push()/has_buffer_space() 메서드로 ICNT 패킷을 전송.
   * 실제 구현: simt_core_cluster::create_l1_cache_interface()에서 설정.
   * 설정자: 생성자 또는 create_exec_pipeline()에서 주입.
   * 읽는 자: ldst_unit::memory_cycle() — L1 미스 시 mem_fetch를 ICNT로 전송.
   * 동기화: ldst_unit의 단일 사이클 루프 내에서 접근. */
  shader_core_mem_fetch_allocator *m_mem_fetch_allocator;
  /* [한국어] mem_fetch 패킷 생성 팩토리 객체.
   * ldst_unit이 L1D 미스 시 새 mem_fetch를 만들기 위해 이 팩토리를 사용.
   * m_sid, m_tpc 정보가 자동으로 패킷에 삽입된다.
   * 설정자: 생성자에서 new shader_core_mem_fetch_allocator(m_sid, m_tpc, m_memory_config).
   * 읽는 자: ldst_unit의 메모리 접근 처리 함수들.
   * 동기화: 단일 코어 사이클 루프. */

  // fetch
  read_only_cache *m_L1I;  // instruction cache
  /* [한국어] L1 명령어 캐시 (read-only_cache). SM당 하나 존재.
   * fetch() 단계에서 워프의 PC를 이용하여 명령어 캐시 접근을 시뮬레이션한다.
   * 캐시 미스 시 ICNT를 통해 L2에서 명령어 블록을 가져온다(accept_fetch_response 경로).
   * 설정자: create_front_pipeline()에서 m_config 설정으로 할당.
   * 읽는 자: fetch() 매 사이클 호출.
   * 동기화: 단일 코어 사이클 루프(fetch는 한 번에 하나의 워프 처리). */
  int m_last_warp_fetched;
  /* [한국어] 마지막으로 명령어 인출을 시도한 워프 슬롯 번호.
   * fetch()에서 라운드 로빈 순서로 다음 워프를 선택할 때 시작점으로 사용.
   * 설정자: fetch()가 인출 시도 시 갱신.
   * 읽는 자: fetch() — 다음 사이클 인출 워프 탐색 시작점.
   * 값 범위: -1(초기) ~ (m_config->n_warps_per_shader - 1).
   * 동기화: 단일 코어 사이클 루프. */

  // decode/dispatch
  std::vector<shd_warp_t *> m_warp;  // per warp information array
  /* [한국어] 이 SM의 모든 워프 슬롯 배열 (크기: n_warps_per_shader).
   * 각 shd_warp_t: 해당 워프의 PC, 활성 마스크, 스코어보드 예약 정보, 스케줄 상태 등.
   * 설정자: create_shd_warp()에서 exec/trace 드라이버별로 파생 객체 생성 후 채움.
   *         issue_block2core()에서 CTA 할당 시 워프 초기화.
   * 읽는 자: 모든 파이프라인 단계 함수, 스케줄러 unit.
   * 동기화: 단일 코어 사이클 루프 내에서만 접근. */
  barrier_set_t m_barriers;
  /* [한국어] __syncthreads() 배리어 동기화를 추적하는 배리어 집합 객체.
   * 각 CTA의 배리어 도착 여부를 추적하고, 모든 워프가 도달하면 해제.
   * 설정자: issue_block2core()에서 CTA 할당 시 배리어 슬롯 초기화.
   *         writeback()에서 배리어 명령어 완료 시 도착 기록.
   * 읽는 자: 스케줄러 — 배리어 대기 중인 워프를 이슈하지 않도록 차단.
   * 동기화: 단일 코어 사이클 루프. */
  ifetch_buffer_t m_inst_fetch_buffer;
  /* [한국어] L1I 캐시 미스 후 메모리에서 명령어 블록이 돌아오기를 기다리는 펜딩 버퍼.
   * fetch()가 L1I 미스 시 여기에 대기 요청을 기록하고, accept_fetch_response()가 완료를 처리.
   * 설정자: fetch()에서 미스 발생 시 설정.
   * 읽는 자: fetch() — 이미 펜딩 중인 페치가 있으면 중복 발행 방지.
   *          accept_fetch_response() — 응답 수신 시 valid 플래그 클리어.
   * 동기화: 단일 코어 사이클 루프. */
  std::vector<register_set> m_pipeline_reg;
  /* [한국어] 파이프라인 단계 간 명령어를 전달하는 파이프라인 레지스터 배열.
   * 인덱스: pipeline_stage_name_t enum 값 (ID_OC_SP, OC_EX_SP, EX_WB 등 13개 스테이지).
   * 각 register_set은 n_warps_per_scheduler 슬롯을 가지며, 해당 단계에 있는 명령어들을 보관.
   * 설정자: create_front_pipeline()에서 크기 결정 및 할당.
   *         decode/issue/execute/writeback에서 각 단계마다 명령어를 push/pop.
   * 읽는 자: 모든 파이프라인 단계 함수.
   * 동기화: 단일 코어 사이클 루프. */
  Scoreboard *m_scoreboard;
  /* [한국어] RAW(Read-After-Write) 해저드를 방지하는 스코어보드 객체.
   * 레지스터가 실행 유닛에 의해 사용 중임을 기록하고, 완료 후 예약을 해제한다.
   * 설정자: create_front_pipeline()에서 new Scoreboard(m_sid, m_config->n_warps_per_shader).
   *         issue_warp()에서 reserveRegisters() 호출.
   *         writeback()에서 releaseRegisters() 호출.
   * 읽는 자: decode() 단계에서 소스 레지스터 충돌 여부 검사.
   * 동기화: 단일 코어 사이클 루프. */
  opndcoll_rfu_t m_operand_collector;
  /* [한국어] 레지스터 파일 뱅크 충돌을 처리하는 Operand Collector(오퍼랜드 수집기) 유닛.
   * GPU 레지스터 파일은 다수의 뱅크로 구성되며, 같은 뱅크에 동시 접근 시 충돌 발생.
   * m_operand_collector는 CU(Collector Unit)와 arbiter를 통해 충돌 없이 오퍼랜드를 수집.
   * 설정자: create_front_pipeline()에서 init() 호출로 구성.
   * 읽는 자: read_operands() — 매 사이클 step() 호출.
   * 동기화: 단일 코어 사이클 루프. */
  int m_active_warps;
  /* [한국어] 현재 이 SM에서 실행 가능 상태인 워프의 수.
   * 설정자: issue_block2core()에서 CTA 할당 시 증가, 워프 완료 시 감소.
   * 읽는 자: 통계 출력, 워프 스케줄러의 비어있는 슬롯 판단.
   * 값 범위: 0 ~ m_config->n_warps_per_shader.
   * 동기화: 단일 코어 사이클 루프. */
  std::vector<register_set *> m_specilized_dispatch_reg;
  /* [한국어] 특수 실행 유닛(specialized unit, 사용자 정의)을 위한 별도 파이프라인 레지스터 포인터 벡터.
   * m_config->m_specialized_unit에 설정된 수만큼 추가 dispatch 레지스터 슬롯을 제공.
   * 설정자: create_exec_pipeline()에서 specialized unit 생성 시 m_pipeline_reg 슬롯 포인터 추가.
   * 읽는 자: issue() — specialized unit으로의 명령어 발행 경로.
   * 동기화: 단일 코어 사이클 루프. */

  // schedule
  std::vector<scheduler_unit *> schedulers;
  /* [한국어] 이 SM에 설치된 워프 스케줄러 객체 벡터.
   * 크기: gpgpusim.config의 gpgpu_num_sched_per_core (sub-core model 시 여러 개).
   * 각 scheduler_unit은 담당 워프 집합을 가지고 매 사이클 cycle()을 호출받아 이슈 대상 선택.
   * 설정자: create_schedulers()에서 설정 타입에 따라 lrr/gto/two_level 등 파생 객체 생성.
   * 읽는 자: issue() — 스케줄러 순서대로 cycle() 호출.
   * 동기화: 단일 코어 사이클 루프. */

  // issue
  unsigned int Issue_Prio;
  /* [한국어] 다음 이슈 사이클에서 먼저 호출될 스케줄러 인덱스 (라운드 로빈 시작점).
   * issue()에서 Issue_Prio부터 시작하여 schedulers 벡터를 순환한다.
   * 설정자: issue()가 한 번 돌 때마다 (Issue_Prio + 1) % num_schedulers 로 갱신.
   * 읽는 자: issue().
   * 값 범위: 0 ~ (schedulers.size() - 1).
   * 동기화: 단일 코어 사이클 루프. */

  // execute
  unsigned m_num_function_units;
  /* [한국어] 이 SM에 설치된 실행 유닛(FU)의 총 수 (m_fu.size()).
   * SP, SFU, DP, INT, Tensor Core, ldst, 특수 유닛 등의 합계.
   * 설정자: create_exec_pipeline() 완료 후 m_fu.size()로 설정.
   * 읽는 자: execute()에서 m_fu 순회 범위 결정.
   * 동기화: 읽기 전용 (파이프라인 생성 후 변경 없음). */
  std::vector<unsigned> m_dispatch_port;
  /* [한국어] 각 실행 유닛의 파이프라인 레지스터 dispatch 포트 인덱스 벡터.
   * m_dispatch_port[i]: i번째 FU가 명령어를 받는 m_pipeline_reg 슬롯의 인덱스
   *                     (예: ID_OC_SP, ID_OC_SFU 등).
   * 설정자: create_exec_pipeline()에서 각 FU 생성 시 포트 인덱스 추가.
   * 읽는 자: issue()에서 FU별 dispatch 레지스터 선택.
   * 동기화: 읽기 전용. */
  std::vector<unsigned> m_issue_port;
  /* [한국어] 각 실행 유닛의 issue 포트 인덱스 벡터 (OC_EX_* 단계에 해당).
   * m_issue_port[i]: i번째 FU의 OC→EX 단계 레지스터 슬롯 인덱스.
   * 설정자: create_exec_pipeline()에서 각 FU 생성 시 추가.
   * 읽는 자: read_operands(), execute().
   * 동기화: 읽기 전용. */
  std::vector<simd_function_unit *>
      m_fu;  // stallable pipelines should be last in this array
  /* [한국어] 이 SM의 모든 실행 유닛(SIMD FU) 벡터.
   * 순서: non-stallable FU(SP, SFU, DP, INT 등) 먼저, stallable FU(ldst) 마지막.
   * stallable FU는 뒤쪽에 배치해야 issue() 로직에서 블로킹 우선 처리가 올바르게 동작함.
   * 설정자: create_exec_pipeline()에서 구성.
   * 읽는 자: execute() — 매 사이클 각 FU의 cycle() 호출.
   *          issue() — dispatch port 매핑으로 명령어 라우팅.
   * 동기화: 단일 코어 사이클 루프. */
  ldst_unit *m_ldst_unit;
  /* [한국어] 이 SM의 로드/스토어 유닛 (Load-Store Unit) 포인터.
   * L1D 캐시, 텍스처 캐시, 공유 메모리, 원자 연산을 처리하는 핵심 메모리 인터페이스.
   * m_fu 벡터에도 포함되어 있으며, 직접 접근이 필요한 곳에서 이 포인터를 사용.
   * 설정자: create_exec_pipeline()에서 new ldst_unit() 생성 후 저장.
   * 읽는 자: accept_ldst_unit_response(), cache_flush(), execute() 등.
   * 동기화: 단일 코어 사이클 루프. */
  static const unsigned MAX_ALU_LATENCY = 512;
  /* [한국어] result bus 비트셋의 최대 크기 — 허용 가능한 최대 ALU 레이턴시(사이클 수).
   * result bus는 m_result_bus[i]가 MAX_ALU_LATENCY 비트의 bitset으로 구현된다.
   * 이 값을 초과하는 레이턴시는 지원되지 않는다. */
  unsigned num_result_bus;
  /* [한국어] 이 SM의 result bus 수 (= 실행 유닛 포트 수와 일치).
   * 설정자: create_exec_pipeline()에서 FU 생성 완료 후 m_result_bus.size()로 설정.
   * 읽는 자: test_res_bus(), writeback().
   * 동기화: 읽기 전용. */
  std::vector<std::bitset<MAX_ALU_LATENCY> *> m_result_bus;
  /* [한국어] 각 result bus의 사이클-레벨 점유 비트셋 벡터.
   * m_result_bus[i]는 비트셋으로, 비트 j가 1이면 j 사이클 후에 결과가 도착함을 예약.
   * 설정자: create_exec_pipeline()에서 각 FU에 대해 new bitset 생성.
   *         issue_warp()에서 결과 도착 사이클에 비트를 set.
   *         writeback()에서 완료 후 해당 비트 clear.
   * 읽는 자: test_res_bus() — 해당 레이턴시 슬롯이 비었는지 확인.
   * 동기화: 단일 코어 사이클 루프. */

  // used for local address mapping with single kernel launch
  unsigned kernel_max_cta_per_shader;
  /* [한국어] 현재 실행 중인 커널이 이 SM에 배치할 수 있는 최대 CTA 수.
   * 로컬 메모리 주소 번역(translate_local_memaddr)에서 CTA당 영역 크기 계산에 사용.
   * 설정자: issue_block2core()에서 커널 파라미터 기반으로 설정.
   * 읽는 자: translate_local_memaddr().
   * 동기화: 단일 코어 사이클 루프. */
  unsigned kernel_padded_threads_per_cta;
  /* [한국어] 패딩이 적용된 CTA당 스레드 수 (warp 경계 정렬).
   * 로컬 메모리 주소 계산 시 CTA 간 영역 간격으로 사용.
   * 설정자: issue_block2core()에서 커널의 pad_threads_per_cta로 설정.
   * 읽는 자: translate_local_memaddr().
   * 동기화: 단일 코어 사이클 루프. */
  // Used for handing out dynamic warp_ids to new warps.
  // the differnece between a warp_id and a dynamic_warp_id
  // is that the dynamic_warp_id is a running number unique to every warp
  // run on this shader, where the warp_id is the static warp slot.
  unsigned m_dynamic_warp_id;
  /* [한국어] 새 워프에 부여되는 동적(고유) 워프 id 카운터.
   * warp_id는 정적 슬롯 번호(재사용됨)이지만, dynamic_warp_id는 이 SM에서
   * 실행된 모든 워프를 통틀어 고유한 단조 증가 번호이다. 통계/디버그에 사용.
   * 설정자: issue_block2core()에서 새 CTA의 각 워프에 부여 후 증가.
   * 읽는 자: 통계 출력, 디버그 로그.
   * 값 범위: 0 이상 (단조 증가).
   * 동기화: 단일 코어 사이클 루프. */

  // Jin: concurrent kernels on a sm
 public:
  /*
   * [한국어] can_issue_1block — 지정된 커널의 CTA 하나를 이 SM에 수용 가능한지 검사한다.
   * @param kernel: 발행하려는 커널 (스레드 수, 공유 메모리, 레지스터 요구량 정보 포함)
   * @return: 이 SM에 자원(스레드 슬롯, shmem, regs, CTA 슬롯)이 충분하면 true
   *
   * m_occupied_* 필드와 비교하여 실제 점유 자원 대비 가용성을 확인한다.
   * 동시 커널(concurrent kernel on SM) 기능 구현을 위해 "Jin"이 추가.
   *
   * 호출 체인: simt_core_cluster::issue_block2core() → [이 함수]
   */
  bool can_issue_1block(kernel_info_t &kernel);
  /*
   * [한국어] occupy_shader_resource_1block — CTA 하나의 자원을 예약하거나 해제한다.
   * @param kernel: 자원 점유/해제 대상 커널
   * @param occupy: true이면 자원 예약, false이면 해제
   * @return: 예약 성공(또는 해제 완료)이면 true, 실패 시 false
   *
   * m_occupied_n_threads, m_occupied_shmem, m_occupied_regs, m_occupied_ctas를 갱신.
   * can_issue_1block()으로 확인 후 실제 점유에 사용된다.
   *
   * 호출 체인: simt_core_cluster::issue_block2core() → [이 함수]
   */
  bool occupy_shader_resource_1block(kernel_info_t &kernel, bool occupy);
  /*
   * [한국어] release_shader_resource_1block — CTA 완료 시 해당 CTA의 자원을 반환한다.
   * @param hw_ctaid: 완료된 CTA의 하드웨어 슬롯 id
   * @param kernel: 완료된 CTA가 속한 커널
   *
   * register_cta_thread_exit()에서 CTA가 완전 종료될 때 호출되어
   * m_occupied_* 필드를 감소시키고 hwtid 비트마스크를 클리어한다.
   *
   * 호출 체인: register_cta_thread_exit() → [이 함수]
   */
  void release_shader_resource_1block(unsigned hw_ctaid, kernel_info_t &kernel);
  /*
   * [한국어] find_available_hwtid — cta_size개의 연속된 빈 하드웨어 스레드 슬롯을 탐색한다.
   * @param cta_size: 필요한 연속 스레드 슬롯 수
   * @param occupy: true이면 찾은 슬롯을 즉시 예약
   * @return: 첫 번째 빈 슬롯 인덱스, 없으면 -1
   *
   * m_occupied_hwtid 비트셋을 순회하여 연속된 빈 구간을 찾는다.
   * SM의 하드웨어 스레드 슬롯 할당의 핵심 로직.
   *
   * 호출 체인: occupy_shader_resource_1block() → [이 함수]
   */
  int find_available_hwtid(unsigned int cta_size, bool occupy);

 private:
  unsigned int m_occupied_n_threads;
  /* [한국어] 현재 이 SM에서 점유 중인 하드웨어 스레드 슬롯의 총 수.
   * 설정자: occupy_shader_resource_1block()에서 증가, release_shader_resource_1block()에서 감소.
   * 읽는 자: can_issue_1block() — 새 CTA 수용 가능 여부 판단.
   * 값 범위: 0 ~ m_config->n_thread_per_shader.
   * 동기화: 단일 코어 사이클 루프 (GPU 메인 스레드에서만 접근). */
  unsigned int m_occupied_shmem;
  /* [한국어] 현재 이 SM에서 점유 중인 공유 메모리(shared memory) 바이트 수.
   * 설정자: occupy_shader_resource_1block()에서 CTA의 shmem 요구량만큼 증가.
   *         release_shader_resource_1block()에서 감소.
   * 읽는 자: can_issue_1block().
   * 값 범위: 0 ~ m_config->gpgpu_shmem_size.
   * 동기화: 단일 코어 사이클 루프. */
  unsigned int m_occupied_regs;
  /* [한국어] 현재 이 SM에서 점유 중인 레지스터 수 (32비트 레지스터 단위).
   * 설정자: occupy_shader_resource_1block()에서 CTA의 regs 요구량 × 스레드 수만큼 증가.
   *         release_shader_resource_1block()에서 감소.
   * 읽는 자: can_issue_1block().
   * 값 범위: 0 ~ m_config->gpgpu_shader_registers.
   * 동기화: 단일 코어 사이클 루프. */
  unsigned int m_occupied_ctas;
  /* [한국어] 현재 이 SM에서 점유 중인 CTA(스레드 블록) 슬롯 수.
   * m_n_active_cta와 유사하지만, 동시 커널 관리를 위해 별도로 추적한다.
   * 설정자: occupy_shader_resource_1block()에서 증가, release_shader_resource_1block()에서 감소.
   * 읽는 자: can_issue_1block() — max_cta() 제한 비교.
   * 값 범위: 0 ~ m_config->max_cta_per_shader().
   * 동기화: 단일 코어 사이클 루프. */
  std::bitset<MAX_THREAD_PER_SM> m_occupied_hwtid;
  /* [한국어] 각 하드웨어 스레드 슬롯의 점유 여부를 나타내는 비트마스크.
   * 비트 i=1: 스레드 슬롯 i가 현재 CTA에 의해 점유됨.
   * 설정자: find_available_hwtid()에서 슬롯 예약 시 set.
   *         release_shader_resource_1block()에서 CTA 완료 시 reset.
   * 읽는 자: find_available_hwtid() — 연속된 빈 슬롯 탐색.
   * 동기화: 단일 코어 사이클 루프. */
  std::map<unsigned int, unsigned int> m_occupied_cta_to_hwtid;
  /* [한국어] CTA 슬롯 번호 → 해당 CTA의 첫 번째 하드웨어 스레드 id 매핑.
   * release 시 어느 hwtid 구간을 해제해야 하는지 역참조하기 위해 필요.
   * 설정자: occupy_shader_resource_1block()에서 CTA 점유 시 삽입.
   *         release_shader_resource_1block()에서 CTA 완료 시 erase.
   * 읽는 자: release_shader_resource_1block().
   * 동기화: 단일 코어 사이클 루프. */
};

/*
 * [한국어 설명] 실행 드라이브(execution-driven) SM 구현 클래스 (exec_shader_core_ctx)
 *
 * === 파일의 역할 ===
 * exec_shader_core_ctx는 shader_core_ctx의 순수 가상 메서드를 execution-driven 모드로
 * 구현한다. CUDA 바이너리를 런타임에 인터셉트하여 PTX/SASS 명령어를 ptx_thread_info를
 * 통해 실제로 기능 시뮬레이션하는 방식이다. libcuda에서 호출한 cuLaunchKernel이 이 경로를
 * 거쳐 시뮬레이터로 전달된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPU 시뮬레이터 계층: cuda-sim(PTX 기능 실행) + gpgpu-sim(타이밍 모델)의 교차점.
 * exec_simt_core_cluster가 create_shader_core_ctx()에서 exec_shader_core_ctx를 생성.
 * 호출 체인: gpgpusim_entrypoint → cuda-sim::ptx_sim_init → exec_simt_core_cluster::create → exec_shader_core_ctx
 *
 * === 타 모듈과의 연결 ===
 * - 상속: shader_core_ctx (타이밍 모델 파이프라인)
 * - cuda-sim/ptx_sim_main.cc: ptx_thread_info 관리, sim_init_thread 구현
 * - cuda-sim/instructions.cc: func_exec_inst()가 각 PTX 명령어 시맨틱을 호출
 * - libcuda/: 런타임 인터셉트 → gpgpusim_entrypoint → 이 클래스로 연결
 *
 * === 주요 함수/구조체 요약 ===
 * - 생성자: create_front_pipeline, create_shd_warp, create_schedulers, create_exec_pipeline 순서 초기화
 * - checkExecutionStatusAndUpdate(): 명령어 실행 완료 후 스레드 종료 여부 확인
 * - func_exec_inst(): PTX 명령어의 기능 시뮬레이션 수행 (cuda-sim 호출)
 * - sim_init_thread(): CTA 할당 시 각 스레드의 PTX 컨텍스트를 초기화
 * - create_shd_warp(): exec_warp_inst_t 래퍼로 m_warp 벡터 구성
 * - get_next_inst(): PTX IR에서 해당 PC의 명령어 객체를 반환
 */
class exec_shader_core_ctx : public shader_core_ctx {
 public:
  /*
   * [한국어] exec_shader_core_ctx 생성자 — 실행 드라이브 SM을 초기화한다.
   * @param gpu: 최상위 gpgpu_sim 객체
   * @param cluster: 이 SM이 속한 simt_core_cluster
   * @param shader_id, tpc_id: SM의 전역 id와 TPC id
   * @param config, mem_config: SM 및 메모리 설정
   * @param stats: 공유 통계 객체
   *
   * 부모 클래스 생성자 호출 후 파이프라인 4단계를 순서대로 구성:
   * 1) create_front_pipeline(): IF/ID 레지스터, L1I 캐시
   * 2) create_shd_warp(): exec_warp_inst_t 기반 워프 슬롯
   * 3) create_schedulers(): warp 스케줄러 생성
   * 4) create_exec_pipeline(): FU/result bus 생성
   *
   * 호출 체인: exec_simt_core_cluster::create_shader_core_ctx() → [이 생성자]
   */
  exec_shader_core_ctx(class gpgpu_sim *gpu, class simt_core_cluster *cluster,
                       unsigned shader_id, unsigned tpc_id,
                       const shader_core_config *config,
                       const memory_config *mem_config,
                       shader_core_stats *stats)
      : shader_core_ctx(gpu, cluster, shader_id, tpc_id, config, mem_config,
                        stats) {
    create_front_pipeline();  // [한국어] IF/ID 파이프라인 레지스터 및 L1I 캐시 생성
    create_shd_warp();        // [한국어] exec_warp_inst_t 기반 워프 슬롯 배열 생성
    create_schedulers();      // [한국어] 설정에 따른 warp 스케줄러 생성
    create_exec_pipeline();   // [한국어] SP/SFU/DP/MEM FU 및 result bus 생성
  }

  /*
   * [한국어] checkExecutionStatusAndUpdate — 실행 완료 후 스레드 종료 여부를 확인한다.
   * @param inst: 방금 실행 완료된 명령어
   * @param t: warp 내 레인 인덱스 (0~31)
   * @param tid: 하드웨어 스레드 id
   *
   * ptx_thread_info의 is_done() 상태를 조회하여 스레드가 EXIT 명령어까지
   * 실행을 완료했는지 확인한다. 종료된 스레드는 register_cta_thread_exit()로 처리.
   *
   * 호출 체인: writeback() → [이 함수] → register_cta_thread_exit()
   */
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid);
  /*
   * [한국어] func_exec_inst — PTX 명령어를 기능 시뮬레이션으로 실행한다.
   * @param inst: 실행할 warp_inst_t (모든 활성 레인의 명령어)
   *
   * 활성 마스크의 각 레인에 대해 ptx_thread_info::ptx_exec_inst()를 호출하여
   * PTX 시맨틱(레지스터 읽기/쓰기, 메모리 접근, 분기 등)을 수행.
   * 실행 결과는 타이밍 모델과 독립적으로 즉시 적용된다(기능 시뮬은 타이밍보다 먼저 완료).
   *
   * 호출 체인: execute() → [이 함수] → ptx_thread_info::ptx_exec_inst()
   */
  virtual void func_exec_inst(warp_inst_t &inst);
  /*
   * [한국어] sim_init_thread — CTA 할당 시 각 스레드의 PTX 기능 시뮬레이션 컨텍스트를 초기화한다.
   * @param kernel: 초기화할 커널
   * @param thread_info: [out] 생성된 ptx_thread_info 반환
   * @param sid: SM id
   * @param tid: 하드웨어 스레드 id
   * @param threads_left, num_threads: CTA 내 위치 정보
   * @param core, hw_cta_id, hw_warp_id: 코어 참조 및 하드웨어 슬롯 정보
   * @param gpu: gpgpu_t 객체
   * @return: 초기화 성공 시 1, 실패 시 0
   *
   * cuda-sim의 ptx_sim_init_thread()를 래핑하여 PTX 스레드 초기화 수행.
   *
   * 호출 체인: init_warps() → [이 함수] → cuda-sim::ptx_sim_init_thread()
   */
  virtual unsigned sim_init_thread(kernel_info_t &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   gpgpu_t *gpu);
  /*
   * [한국어] create_shd_warp — exec_warp_inst_t 기반 워프 슬롯 배열을 생성한다.
   * m_warp 벡터에 n_warps_per_shader 개의 exec_warp_inst_t 객체를 할당.
   * exec_warp_inst_t는 shd_warp_t를 상속하며 PTX 실행 컨텍스트를 포함한다.
   */
  virtual void create_shd_warp();
  /*
   * [한국어] get_next_inst — 해당 워프의 PC에 있는 PTX 명령어 객체를 반환한다.
   * @param warp_id: 대상 워프
   * @param pc: 조회할 PC
   * @return: PTX IR에서 해당 PC의 warp_inst_t 포인터 (캐시됨)
   *
   * cuda-sim의 명령어 맵에서 PC로 warp_inst_t를 조회하여 반환.
   */
  virtual const warp_inst_t *get_next_inst(unsigned warp_id, address_type pc);
  /*
   * [한국어] get_pdom_stack_top_info — SIMT(Post-Dominator) 스택 최상단의 PC/RPC를 반환한다.
   * @param warp_id: 대상 워프
   * @param pI: 현재 명령어
   * @param pc: [out] 현재 실행 PC
   * @param rpc: [out] reconvergence PC
   *
   * exec 모드: simt_stack(pdom 스택)에서 top().pc, top().rpc를 조회.
   */
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc);
  /*
   * [한국어] get_active_mask — 주어진 워프/명령어의 현재 활성 스레드 마스크를 반환한다.
   * exec 모드: simt_stack의 활성 마스크를 사용하여 warp divergence 상태를 반영.
   */
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI);
};

/*
 * [한국어 설명] SM 클러스터 추상 기반 클래스 (simt_core_cluster)
 *
 * === 파일의 역할 ===
 * simt_core_cluster는 여러 SM(shader_core_ctx)을 하나의 클러스터로 묶고,
 * 클러스터 단위의 사이클 진행(core_cycle, icnt_cycle), CTA 발행(issue_block2core),
 * 캐시 관리(flush/invalidate) 및 ICNT 패킷 송수신을 담당한다.
 * GPGPU-Sim에서 GPU TPC(Texture Processor Cluster)에 해당하는 시뮬레이터 추상이다.
 * create_shader_core_ctx()는 순수 가상으로, 파생 클래스가 실행 모드에 맞는 SM을 생성한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 계층: gpgpu_sim → simt_core_cluster → shader_core_ctx → 파이프라인 단계
 * gpu-sim.cc의 cycle()에서 모든 클러스터의 core_cycle()/icnt_cycle()을 호출한다.
 * issue_block2core()는 GPU 스케줄러가 CTA를 SM에 배정하는 핵심 경로다.
 * 호출 체인: gpgpu_sim::cycle() → [이 클래스]::core_cycle()/icnt_cycle()
 *
 * === 타 모듈과의 연결 ===
 * - gpgpu_sim (gpu-sim.cc): m_cluster 벡터를 순회하며 사이클 진행
 * - intersim2 (ICNT): icnt_inject_request_packet()으로 NoC에 요청 삽입
 * - shader_core_ctx: m_core 배열의 각 SM에 cycle() 위임
 * - mem_fetch: m_response_fifo로 메모리 응답 패킷 전달
 *
 * === 주요 함수/구조체 요약 ===
 * - core_cycle(): 이 클러스터 내 모든 SM의 cycle()을 순서대로 호출
 * - icnt_cycle(): ICNT에서 수신된 메모리 응답 패킷을 해당 SM으로 전달
 * - issue_block2core(): 대기 중인 CTA를 자원이 충분한 SM에 배정
 * - icnt_inject_request_packet(): SM의 메모리 요청을 ICNT(NoC)에 주입
 * - m_core[]: 클러스터에 속한 SM 배열 (크기: n_simt_cores_per_cluster)
 */
class simt_core_cluster {
 public:
  /*
   * [한국어] simt_core_cluster 생성자 — 클러스터를 초기화한다.
   * @param gpu: 최상위 gpgpu_sim 객체
   * @param cluster_id: 이 클러스터의 전역 id (0 ~ num_cluster-1)
   * @param config, mem_config: SM 및 메모리 설정
   * @param stats: 공유 통계 객체
   * @param mstats: 메모리 통계 객체
   *
   * m_core 배열 할당 및 m_cta_issue_next_core 초기화 수행.
   * 구체적 SM 객체 생성은 파생 클래스의 create_shader_core_ctx()에서 수행.
   */
  simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                    const shader_core_config *config,
                    const memory_config *mem_config, shader_core_stats *stats,
                    memory_stats_t *mstats);

  /*
   * [한국어] core_cycle — 이 클러스터 내 모든 SM의 1 사이클을 진행한다.
   * m_core_sim_order 리스트 순서대로 각 SM의 cycle()을 호출.
   * 호출 체인: gpgpu_sim::cycle() → [이 함수] → shader_core_ctx::cycle()
   */
  void core_cycle();
  /*
   * [한국어] icnt_cycle — ICNT(NoC)에서 수신된 메모리 응답 패킷을 처리한다.
   * intersim2의 출력 큐를 폴링하여 mem_fetch를 꺼내고,
   * 해당 SM의 accept_ldst_unit_response() 또는 accept_fetch_response()로 전달.
   * 호출 체인: gpgpu_sim::cycle() → [이 함수] → shader_core_ctx::accept_*_response()
   */
  void icnt_cycle();

  /*
   * [한국어] reinit — 새 커널 시작 시 클러스터 상태를 초기화한다.
   * 각 SM의 reinit()을 호출하고 m_response_fifo를 비운다.
   */
  void reinit();
  /*
   * [한국어] issue_block2core — 대기 중인 CTA를 자원이 충분한 SM에 배정한다.
   * @return: 이번 호출에서 발행한 CTA 수
   *
   * 클러스터 내 SM을 라운드 로빈으로 순회하며 can_issue_1block()으로
   * 자원 가용성을 확인하고 shader_core_ctx::issue_block2core()로 배정.
   * 호출 체인: gpgpu_sim::issue_block2core() → [이 함수] → shader_core_ctx::issue_block2core()
   */
  unsigned issue_block2core();
  void cache_flush();       // [한국어] 클러스터 내 모든 SM의 L1 캐시를 플러시한다
  void cache_invalidate();  // [한국어] 클러스터 내 모든 SM의 L1 캐시를 무효화한다
  /*
   * [한국어] icnt_injection_buffer_full — ICNT 주입 버퍼가 가득 찼는지 확인한다.
   * @param size: 전송 패킷 크기 (바이트)
   * @param write: 쓰기 요청 여부
   * @return: 버퍼가 가득 차 있으면 true (ldst_unit이 이 값으로 대기 여부 결정)
   */
  bool icnt_injection_buffer_full(unsigned size, bool write);
  /*
   * [한국어] icnt_inject_request_packet — SM의 메모리 요청 mem_fetch를 ICNT에 주입한다.
   * @param mf: 주입할 메모리 요청 패킷
   *
   * intersim2의 icnt_push()를 호출하여 NoC 네트워크에 패킷을 삽입.
   * 호출 체인: shader_memory_interface::push() → [이 함수] → icnt_push()
   */
  void icnt_inject_request_packet(class mem_fetch *mf);
  void update_icnt_stats(class mem_fetch *mf); // [한국어] ICNT 트래픽 통계 갱신

  // for perfect memory interface
  /*
   * [한국어] response_queue_full — perfect memory 모드의 응답 큐 포화 여부를 반환한다.
   * perfect_memory_interface::full()에서 호출. ejection 버퍼 크기와 비교.
   */
  bool response_queue_full() {
    return (m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size); // [한국어] ejection 버퍼 크기 초과 여부 확인
  }
  /*
   * [한국어] push_response_fifo — perfect memory 모드에서 응답을 직접 FIFO 큐에 삽입한다.
   * @param mf: 삽입할 응답 패킷
   * ICNT를 우회하여 직접 SM에 메모리 응답을 전달하는 경로(perfect mem interface 전용).
   */
  void push_response_fifo(class mem_fetch *mf) {
    m_response_fifo.push_back(mf); // [한국어] 응답 큐에 패킷 추가 (icnt_cycle에서 SM으로 전달됨)
  }

  /*
   * [한국어] get_pdom_stack_top_info — 특정 스레드의 SIMT 스택 상단 PC/RPC를 반환한다.
   * @param sid: SM id, @param tid: 스레드 id
   * @param pc, rpc: [out] 현재 PC / reconvergence PC
   */
  void get_pdom_stack_top_info(unsigned sid, unsigned tid, unsigned *pc,
                               unsigned *rpc) const;
  unsigned max_cta(const kernel_info_t &kernel);    // [한국어] 이 클러스터에 배정 가능한 최대 CTA 수
  unsigned get_not_completed() const;               // [한국어] 아직 완료되지 않은 스레드 수
  void print_not_completed(FILE *fp) const;         // [한국어] 미완료 스레드 현황 출력
  unsigned get_n_active_cta() const;                // [한국어] 실행 중인 총 CTA 수
  unsigned get_n_active_sms() const;                // [한국어] 활성화된 SM 수
  gpgpu_sim *get_gpu() { return m_gpu; }            // [한국어] 최상위 gpgpu_sim 포인터 반환

  void display_pipeline(unsigned sid, FILE *fout, int print_mem, int mask); // [한국어] 디버그용 파이프라인 상태 출력
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses) const; // [한국어] L1 캐시 접근/미스 통계 출력

  void get_cache_stats(cache_stats &cs) const;                      // [한국어] L1 캐시 통계 집계
  void get_L1I_sub_stats(struct cache_sub_stats &css) const;        // [한국어] L1I 캐시 세부 통계
  void get_L1D_sub_stats(struct cache_sub_stats &css) const;        // [한국어] L1D 캐시 세부 통계
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;        // [한국어] L1C(상수) 캐시 세부 통계
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;        // [한국어] L1T(텍스처) 캐시 세부 통계

  /*
   * [한국어] get_icnt_stats — ICNT 트래픽 통계(SM→메모리, 메모리→SM 플릿 수)를 반환한다.
   */
  void get_icnt_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;
  /*
   * [한국어] get_current_occupancy — 현재 클러스터의 워프 점유율을 반환한다.
   * @param active: [out] 현재 활성 워프 수
   * @param total: [out] 최대 가능 워프 수
   * @return: 점유율 (active/total, 0.0~1.0)
   */
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const;
  /*
   * [한국어] create_shader_core_ctx — 실행 모드에 맞는 SM 객체를 생성하는 순수 가상 함수.
   * exec 모드: exec_shader_core_ctx 생성 (exec_simt_core_cluster에서 구현).
   * SST 모드: exec_shader_core_ctx 그대로 사용 (sst_simt_core_cluster).
   */
  virtual void create_shader_core_ctx() = 0;

 protected:
  unsigned m_cluster_id;
  /* [한국어] 이 클러스터의 전역 id (0 ~ num_simt_clusters-1).
   * ICNT에서 소스/목적지 노드 id로 사용되어 패킷 라우팅에 활용.
   * 설정자: 생성자 파라미터. 읽기 전용. */
  gpgpu_sim *m_gpu;
  /* [한국어] 최상위 GPU 시뮬레이터 포인터.
   * 커널 큐 접근, 전역 사이클 번호 조회 등 전역 상태 접근에 사용.
   * 설정자: 생성자 파라미터. 읽기 전용. */
  const shader_core_config *m_config;
  /* [한국어] SM 설정 파라미터 (클러스터 내 모든 SM이 공유하는 불변 포인터).
   * 설정자: 생성자 파라미터. 읽기 전용. */
  shader_core_stats *m_stats;
  /* [한국어] 클러스터 내 SM들이 공유하는 성능 통계 객체.
   * 각 SM은 m_sid 슬롯만 접근하므로 충돌 없음.
   * 설정자: 생성자 파라미터. */
  memory_stats_t *m_memory_stats;
  /* [한국어] 메모리 시스템 통계 객체 (캐시 히트/미스, DRAM 접근 수 등).
   * 설정자: 생성자 파라미터. */
  shader_core_ctx **m_core;
  /* [한국어] 이 클러스터에 속한 SM 배열 포인터 (크기: n_simt_cores_per_cluster).
   * 설정자: 생성자에서 calloc, create_shader_core_ctx()에서 각 SM 생성 후 채움.
   * 읽는 자: core_cycle(), icnt_cycle(), issue_block2core() 등.
   * 동기화: 단일 GPU 메인 사이클 루프. */
  const memory_config *m_mem_config;
  /* [한국어] 메모리 시스템 설정 파라미터 포인터. SM 생성 시 각 shader_core_ctx에 전달. */

  unsigned m_cta_issue_next_core;
  /* [한국어] 다음 CTA 발행 대상 SM의 슬롯 인덱스 (라운드 로빈 시작점).
   * 설정자: issue_block2core()에서 발행 성공 후 (idx+1) % n_core로 갱신.
   * 값 범위: 0 ~ (config->n_simt_cores_per_cluster - 1).
   * 동기화: 단일 GPU 메인 사이클 루프. */
  std::list<unsigned> m_core_sim_order;
  /* [한국어] 클러스터 내 SM의 사이클 진행 순서 리스트.
   * core_cycle()이 이 순서대로 각 SM의 cycle()을 호출한다.
   * 설정자: 생성자에서 0, 1, ... n-1 순서로 초기화. */
  std::list<mem_fetch *> m_response_fifo;
  /* [한국어] ICNT 또는 perfect memory에서 수신한 메모리 응답 패킷 FIFO 큐.
   * icnt_cycle()이 intersim2에서 꺼내 여기에 저장하거나,
   * perfect_memory_interface가 직접 push_response_fifo()로 삽입한다.
   * 설정자: icnt_cycle() 또는 push_response_fifo()에서 삽입.
   * 읽는 자: icnt_cycle() — SM에 응답 전달 후 pop.
   * 동기화: 단일 GPU 메인 사이클 루프. */
};

/*
 * [한국어 설명] 실행 드라이브 SM 클러스터 구현 클래스 (exec_simt_core_cluster)
 *
 * === 파일의 역할 ===
 * exec_simt_core_cluster는 simt_core_cluster의 create_shader_core_ctx()를 구현하여
 * execution-driven 모드(CUDA 바이너리 직접 인터셉트)에서 동작하는
 * exec_shader_core_ctx SM 객체들을 생성한다.
 * 표준 GPGPU-Sim 실행 경로에서 사용되는 기본 클러스터 클래스다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * gpgpu_sim → exec_simt_core_cluster → exec_shader_core_ctx → 파이프라인
 * gpgpusim_entrypoint.cc에서 gpgpu_sim 생성 시 이 클래스의 클러스터 배열이 구성된다.
 *
 * === 타 모듈과의 연결 ===
 * - 상속: simt_core_cluster (클러스터 공통 로직)
 * - exec_shader_core_ctx: create_shader_core_ctx()에서 생성되는 SM 타입
 * - gpgpusim_entrypoint: 클러스터 벡터 초기화
 *
 * === 주요 함수/구조체 요약 ===
 * - 생성자: 부모 생성자 호출 후 create_shader_core_ctx()로 SM 배열 구성
 * - create_shader_core_ctx(): exec_shader_core_ctx 객체를 m_core[]에 할당
 */
class exec_simt_core_cluster : public simt_core_cluster {
 public:
  /*
   * [한국어] exec_simt_core_cluster 생성자 — execution-driven 클러스터를 초기화한다.
   * 부모 생성자에서 m_core 배열을 할당한 뒤 create_shader_core_ctx()로 각 SM을 생성.
   *
   * 호출 체인: gpgpu_sim 생성자 → [이 생성자] → create_shader_core_ctx() → exec_shader_core_ctx 생성
   */
  exec_simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                         const shader_core_config *config,
                         const memory_config *mem_config,
                         class shader_core_stats *stats,
                         class memory_stats_t *mstats)
      : simt_core_cluster(gpu, cluster_id, config, mem_config, stats, mstats) {
    create_shader_core_ctx(); // [한국어] exec_shader_core_ctx SM 객체 배열 생성
  }

  /*
   * [한국어] create_shader_core_ctx — exec_shader_core_ctx SM 배열을 생성한다.
   * m_core[i] = new exec_shader_core_ctx(...) 형태로 n_simt_cores_per_cluster 개의 SM을 할당.
   * 호출 체인: [생성자] → [이 함수]
   */
  virtual void create_shader_core_ctx();
};

/*
 * [한국어 설명] SST 연동 SM 클러스터 클래스 (sst_simt_core_cluster)
 *
 * === 파일의 역할 ===
 * sst_simt_core_cluster는 exec_simt_core_cluster를 상속하여 SST(Structural Simulation Toolkit)
 * 메모리 시스템과 연동하기 위한 추가 메서드를 제공한다. SST는 Balar 등
 * 외부 컴포넌트와의 공동 시뮬레이션에 사용되며, ICNT 대신 SST 메모리 버스로 요청을 전달한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * gpgpu_sim → sst_simt_core_cluster → exec_shader_core_ctx → 파이프라인
 * SST 모드에서는 ICNT 대신 SST가 메모리 트랜잭션을 중개한다.
 *
 * === 타 모듈과의 연결 ===
 * - 상속: exec_simt_core_cluster → simt_core_cluster
 * - Balar (SST 컴포넌트): is_SST_buffer_full(), SST 메모리 요청 전송
 * - sst_memory_interface: push()에서 icnt_inject_request_packet_to_SST() 호출
 *
 * === 주요 함수/구조체 요약 ===
 * - SST_injection_buffer_full(): SST 쪽 버퍼가 가득 찼는지 확인
 * - icnt_inject_request_packet_to_SST(): SST 메모리 시스템으로 요청 전달
 * - icnt_cycle_SST(): SST-코어 간 ICNT 사이클 진행
 */
/**
 * @brief SST cluster class
 *
 */
class sst_simt_core_cluster : public exec_simt_core_cluster {
 public:
  /*
   * [한국어] sst_simt_core_cluster 생성자 — SST 연동 클러스터를 초기화한다.
   * exec_simt_core_cluster 생성자에 위임하여 SM 배열 구성을 동일하게 진행.
   * SST 특화 초기화는 Balar 컴포넌트에서 별도로 수행.
   */
  sst_simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                        const shader_core_config *config,
                        const memory_config *mem_config,
                        class shader_core_stats *stats,
                        class memory_stats_t *mstats)
      : exec_simt_core_cluster(gpu, cluster_id, config, mem_config, stats,
                               mstats) {} // [한국어] 부모 생성자에 위임

  /**
   * @brief Check if SST memory request injection
   *        buffer is full by using extern
   *        function is_SST_buffer_full()
   *        defined in Balar
   *
   * @param size
   * @param write
   * @param type
   * @return true
   * @return false
   */
  /*
   * [한국어] SST_injection_buffer_full — SST 쪽 메모리 요청 주입 버퍼 포화 여부를 확인한다.
   * @param size: 패킷 크기, @param write: 쓰기 여부, @param type: 메모리 접근 타입
   * @return: SST 버퍼가 가득 차 있으면 true
   *
   * Balar(SST 컴포넌트)에 정의된 extern 함수 is_SST_buffer_full()을 호출.
   * sst_memory_interface::full()에서 호출된다.
   */
  bool SST_injection_buffer_full(unsigned size, bool write,
                                 mem_access_type type);

  /**
   * @brief Send memory request packets to SST
   *        memory
   *
   * @param mf
   */
  /*
   * [한국어] icnt_inject_request_packet_to_SST — 메모리 요청 패킷을 SST 메모리 시스템으로 전달한다.
   * @param mf: 전달할 mem_fetch 패킷
   *
   * ICNT(intersim2) 대신 SST 버스를 통해 메모리 요청을 전송.
   * 호출 체인: sst_memory_interface::push() → [이 함수] → SST 메모리
   */
  void icnt_inject_request_packet_to_SST(class mem_fetch *mf);

  /**
   * @brief Advance ICNT between core and SST
   *
   */
  /*
   * [한국어] icnt_cycle_SST — SST와 코어 간의 ICNT 사이클을 진행한다.
   * SST에서 도착한 메모리 응답을 m_response_fifo에 넣어 SM으로 전달.
   * gpgpu_sim::cycle()에서 SST 모드일 때 icnt_cycle() 대신 또는 함께 호출.
   */
  void icnt_cycle_SST();
};

/*
 * [한국어 설명] 표준 ICNT 메모리 인터페이스 클래스 (shader_memory_interface)
 *
 * === 파일의 역할 ===
 * shader_memory_interface는 ldst_unit이 메모리 시스템과 통신할 때 사용하는
 * mem_fetch_interface의 구체 구현이다. full()로 ICNT 버퍼 포화를 확인하고,
 * push()로 mem_fetch를 ICNT(NoC)에 주입한다. 표준 GPGPU-Sim 실행 모드에서 사용.
 *
 * === 전체 아키텍처에서의 위치 ===
 * ldst_unit → shader_memory_interface → simt_core_cluster → ICNT(intersim2)
 *
 * === 타 모듈과의 연결 ===
 * - mem_fetch_interface: 추상 기반 클래스 (full/push 인터페이스 정의)
 * - simt_core_cluster: 실제 ICNT 주입 로직 위임
 * - shader_core_ctx: ICNT 트래픽 통계(inc_simt_to_mem) 갱신
 *
 * === 주요 함수/구조체 요약 ===
 * - full(): ICNT 주입 버퍼 포화 여부 — ldst_unit이 대기 여부 결정
 * - push(): 통계 갱신 후 ICNT에 mem_fetch 삽입
 */
class shader_memory_interface : public mem_fetch_interface {
 public:
  /*
   * [한국어] shader_memory_interface 생성자 — SM과 클러스터 포인터를 저장한다.
   * @param core: 이 인터페이스를 소유하는 SM (통계 갱신용)
   * @param cluster: ICNT 주입 대상 클러스터
   */
  shader_memory_interface(shader_core_ctx *core, simt_core_cluster *cluster) {
    m_core = core;       // [한국어] 소속 SM 포인터 저장 (통계 갱신용)
    m_cluster = cluster; // [한국어] ICNT 주입 대상 클러스터 포인터 저장
  }
  /*
   * [한국어] full — ICNT 주입 버퍼가 가득 찼는지 확인한다.
   * @param size: 전송 패킷 크기, @param write: 쓰기 여부
   * @return: 가득 차 있으면 true (ldst_unit이 이 결과로 stall 여부 결정)
   */
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->icnt_injection_buffer_full(size, write); // [한국어] 클러스터의 ICNT 버퍼 상태 조회
  }
  /*
   * [한국어] push — mem_fetch를 ICNT에 주입하고 통계를 갱신한다.
   * @param mf: 전송할 메모리 요청 패킷
   *
   * 먼저 inc_simt_to_mem()으로 SM→메모리 트래픽 플릿 수를 누적하고,
   * 이후 icnt_inject_request_packet()으로 NoC에 패킷을 삽입한다.
   */
  virtual void push(mem_fetch *mf) {
    m_core->inc_simt_to_mem(mf->get_num_flits(true)); // [한국어] SM→메모리 방향 플릿 수 통계 갱신
    m_cluster->icnt_inject_request_packet(mf);        // [한국어] ICNT(intersim2)에 요청 패킷 주입
  }

 private:
  shader_core_ctx *m_core;
  /* [한국어] 이 인터페이스를 소유하는 SM 포인터 (통계 갱신 전용).
   * 설정자: 생성자. 읽기 전용. */
  simt_core_cluster *m_cluster;
  /* [한국어] ICNT 주입 대상 클러스터 포인터.
   * 설정자: 생성자. 읽기 전용. */
};

/*
 * [한국어 설명] 완벽 메모리 인터페이스 클래스 (perfect_memory_interface)
 *
 * === 파일의 역할 ===
 * perfect_memory_interface는 ICNT 레이턴시 없이 메모리 요청을 즉시 응답하는
 * "perfect memory" 시뮬레이션 모드를 구현한다. 연구 목적으로 메모리 레이턴시를
 * 제거하여 SM 파이프라인만 독립적으로 연구할 때 사용된다.
 * 원자적 연산은 "메모리 서브시스템 내부"에서 즉시 실행된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * ldst_unit → perfect_memory_interface → simt_core_cluster::push_response_fifo()
 * (ICNT를 우회하여 직접 응답 큐에 삽입)
 *
 * === 타 모듈과의 연결 ===
 * - mem_fetch_interface: 추상 기반 클래스
 * - simt_core_cluster: response_queue_full()/push_response_fifo() 호출
 *
 * === 주요 함수/구조체 요약 ===
 * - full(): 응답 큐 포화 여부 확인 (ejection buffer 크기 비교)
 * - push(): 원자 연산 즉시 실행 후 응답 큐에 직접 삽입
 */
class perfect_memory_interface : public mem_fetch_interface {
 public:
  /*
   * [한국어] perfect_memory_interface 생성자 — SM과 클러스터 포인터를 저장한다.
   */
  perfect_memory_interface(shader_core_ctx *core, simt_core_cluster *cluster) {
    m_core = core;       // [한국어] 소속 SM 포인터 저장
    m_cluster = cluster; // [한국어] 클러스터 포인터 저장 (응답 큐 접근용)
  }
  /*
   * [한국어] full — 클러스터의 응답 큐(response_fifo)가 가득 찼는지 확인한다.
   * ICNT 버퍼 대신 ejection buffer 크기를 기준으로 포화 여부를 결정.
   */
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->response_queue_full(); // [한국어] 클러스터 응답 큐 포화 여부 확인
  }
  /*
   * [한국어] push — 원자 연산을 즉시 실행하고 응답을 직접 응답 큐에 삽입한다.
   * @param mf: 전송할 메모리 요청 패킷
   *
   * 원자 연산(isatomic())이면 do_atomic()으로 즉시 처리(ICNT 통과 없이).
   * 그 후 inc_simt_to_mem()으로 통계를 갱신하고 push_response_fifo()에 삽입.
   * 메모리 레이턴시 없이 SM이 즉시 응답을 받을 수 있다.
   */
  virtual void push(mem_fetch *mf) {
    if (mf && mf->isatomic())
      mf->do_atomic();  // execute atomic inside the "memory subsystem"
      // [한국어] 원자 연산을 메모리 서브시스템 내에서 즉시 실행 (실제 GPU도 L2에서 처리)
    m_core->inc_simt_to_mem(mf->get_num_flits(true)); // [한국어] SM→메모리 트래픽 통계 갱신
    m_cluster->push_response_fifo(mf); // [한국어] ICNT를 우회해 응답 큐에 직접 삽입 (즉시 응답)
  }

 private:
  shader_core_ctx *m_core;
  /* [한국어] 소속 SM 포인터 (통계 갱신용). */
  simt_core_cluster *m_cluster;
  /* [한국어] 클러스터 포인터 (응답 큐 접근용). */
};

/*
 * [한국어 설명] SST 연동 메모리 인터페이스 클래스 (sst_memory_interface)
 *
 * === 파일의 역할 ===
 * sst_memory_interface는 SST(Structural Simulation Toolkit) 메모리 시스템과
 * 연동하는 mem_fetch_interface 구현이다. 상수/텍스처/명령어 캐시 접근은 표준 ICNT를
 * 통하지만, 일반 데이터 접근은 SST 메모리 시스템으로 직접 전달된다.
 * full()은 mem_access_type 정보를 필요로 하여 2-인자 버전을 사용해야 한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * ldst_unit → sst_memory_interface → sst_simt_core_cluster → SST 메모리
 *
 * === 타 모듈과의 연결 ===
 * - mem_fetch_interface: 추상 기반 클래스
 * - sst_simt_core_cluster: SST 버퍼 포화 확인 및 패킷 전달
 * - shader_core_ctx: 트래픽 통계 갱신
 *
 * === 주요 함수/구조체 요약 ===
 * - full(size, write, type): SST 버퍼 포화 여부 확인 (타입 정보 필요)
 * - push(): 통계 갱신 후 SST 메모리로 패킷 전달
 */
/**
 * @brief SST memory interface
 *
 */
class sst_memory_interface : public mem_fetch_interface {
 public:
  /*
   * [한국어] sst_memory_interface 생성자 — SM과 SST 클러스터 포인터를 저장한다.
   */
  sst_memory_interface(shader_core_ctx *core, sst_simt_core_cluster *cluster) {
    m_core = core;       // [한국어] 소속 SM 포인터 저장 (통계 갱신용)
    m_cluster = cluster; // [한국어] SST 클러스터 포인터 저장 (버퍼 확인 및 패킷 전달용)
  }
  /**
   * @brief For constant, inst, tex cache access
   *
   * @param size
   * @param write
   * @return true
   * @return false
   */
  /*
   * [한국어] full(size, write) — 2-인자 버전은 SST 모드에서 사용 금지.
   * 상수/명령어/텍스처 캐시 접근에 대한 기본 인터페이스이나,
   * SST 모드에서는 mem_access_type 없이 버퍼 포화를 판단할 수 없으므로 assert로 차단.
   */
  virtual bool full(unsigned size, bool write) const {
    assert(false && "Use the full() method with access type instead!"); // [한국어] SST 모드에서는 타입 포함 버전 사용 필수
    return true;
  }

  /**
   * @brief With SST, the core will direct all mem access except for
   *        constant, tex, and inst reads to SST mem system
   *        (i.e. not modeling constant mem right now), thus
   *        requiring the mem_access_type information to be passed in
   *
   * @param size
   * @param write
   * @param type
   * @return true
   * @return false
   */
  /*
   * [한국어] full(size, write, type) — SST 쪽 메모리 요청 버퍼 포화 여부를 확인한다.
   * @param type: 메모리 접근 타입 (GLOBAL_ACC_R, LOCAL_ACC_R 등) — SST 경로 결정에 사용
   * @return: SST 버퍼가 가득 차 있으면 true (ldst_unit이 stall 결정)
   */
  bool full(unsigned size, bool write, mem_access_type type) const {
    return m_cluster->SST_injection_buffer_full(size, write, type); // [한국어] SST 측 버퍼 포화 여부를 클러스터에 위임
  }

  /**
   * @brief Push memory request to SST memory system and
   *        update stats
   *
   * @param mf
   */
  /*
   * [한국어] push — 통계를 갱신하고 mem_fetch를 SST 메모리 시스템으로 전달한다.
   * @param mf: 전송할 메모리 요청 패킷
   *
   * inc_simt_to_mem()으로 SM→메모리 트래픽 통계를 갱신한 후
   * icnt_inject_request_packet_to_SST()로 SST 메모리에 패킷 전달.
   */
  virtual void push(mem_fetch *mf) {
    m_core->inc_simt_to_mem(mf->get_num_flits(true));    // [한국어] SM→메모리 방향 플릿 수 통계 갱신
    m_cluster->icnt_inject_request_packet_to_SST(mf);   // [한국어] SST 메모리 시스템으로 패킷 전달
  }

 private:
  shader_core_ctx *m_core;
  /* [한국어] 소속 SM 포인터 (통계 갱신 전용).
   * 설정자: 생성자. 읽기 전용. */
  sst_simt_core_cluster *m_cluster;
  /* [한국어] SST 연동 클러스터 포인터 (버퍼 확인 및 패킷 전달).
   * 설정자: 생성자. 읽기 전용. */
};

/*
 * [한국어] scheduler_unit::get_sid — 이 스케줄러가 속한 SM의 전역 id를 반환한다.
 * @return: m_shader->get_sid() — 소속 shader_core_ctx의 m_sid 값
 *
 * scheduler_unit이 shader_core_ctx의 m_sid를 직접 접근할 수 없으므로
 * 이 인라인 헬퍼를 통해 SM id를 조회한다.
 * shader.h 말미에 정의된 이유: scheduler_unit 선언이 shader_core_ctx 선언보다 앞에 있기 때문.
 */
inline int scheduler_unit::get_sid() const { return m_shader->get_sid(); }

#endif /* SHADER_H */
