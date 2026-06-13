// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// Ivan Sham, George L. Yuan, Vijay Kandiah, Nikos Hardavellas,
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

/*
 * [한국어 설명] DRAM 타이밍 시뮬레이터 구현 (dram.cc)
 *
 * === 파일의 역할 ===
 * dram.h에 선언된 dram_t(DRAM 채널 타이밍 모델), dram_req_t(DRAM 요청 패킷) 클래스의
 * 전체 구현을 담고 있다. GPU의 각 메모리 파티션마다 하나의 dram_t 인스턴스가 존재하며,
 * 이 파일의 dram_t::cycle()이 매 DRAM 클럭 사이클마다 호출되어 뱅크별 타이밍 카운터를
 * 감소시키고 ACT/PRE/RD/WR 커맨드를 발행하며 통계를 수집한다. DDR SDRAM의
 * row-buffer 모델(open-page 정책)과 FIFO/FR-FCFS 두 가지 스케줄러를 지원한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 메모리 계층 최하단 타이밍 모델이다. SM에서 발생한 캐시 미스가
 * L1D cache miss → L2 cache miss → NoC(ICNT) → memory_partition_unit::push()
 * → dram_t::push() 경로로 이 파일에 도달한다. 처리 완료 후 READ 응답은
 * returnq → memory_partition_unit → ICNT → L2 → SM 방향으로 역류한다.
 * 호출 체인: gpgpu_sim::cycle() → memory_partition_unit::dram_cycle()
 *           → dram_t::cycle() → scheduler_fifo()/scheduler_frfcfs()
 *           → issue_col_command(j) / issue_row_command(j)
 * 실행 컨텍스트: 호스트 CPU 단일 스레드 시뮬레이션 루프 (사이클-레벨 타이밍 모델)
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - dram_sched.h/cc: FR-FCFS 스케줄러 구현 (frfcfs_scheduler 클래스)
 *   - gpu-sim.h: gpgpu_sim 전역 상태 (사이클 카운터, memory_config 접근)
 *   - hashing.h: bitwise_hash_function, ipoly_hash_function — 뱅크 인덱싱에 사용
 *   - l2cache.h: memory_partition_unit — set_done(), returnq 소비자
 *   - mem_fetch.h: 메모리 요청 패킷 — push()의 입력, returnq의 출력
 *   - mem_latency_stat.h: memlatstat_dram_access() — 레이턴시 통계 기록
 *   - gpu-misc.h: StatCreate() — mrqq_Dist 히스토그램 생성
 * 역방향 의존:
 *   - l2cache.cc(memory_partition_unit): push(), cycle(), return_queue_pop() 호출
 *   - accelwattch/power_interface.cc: set_dram_power_stats() 호출
 *   - gpu-sim.cc: print(), print_stat(), visualizer_print() 호출
 * 데이터 흐름:
 *   mem_fetch → dram_req_t (생성자에서 row/col/bk 추출)
 *   → mrqq (FIFO 대기 큐) / frfcfs_scheduler (FR-FCFS 대기 큐)
 *   → bk[n]->mrq (뱅크 배정)
 *   → rwq (ACT 후 RD/WR 파이프라인, CL/WL 깊이)
 *   → returnq (CL 통과 완료 후 READ 응답 보관)
 *   → memory_partition_unit::return_queue_pop() → ICNT → SM
 *
 * === 주요 함수/구조체 요약 ===
 * dram_t::dram_t()         : 뱅크 배열/큐/스케줄러/카운터 전체 초기화
 * dram_t::push()           : 새 메모리 요청을 dram_req_t로 변환하여 mrqq에 삽입
 * dram_t::cycle()          : 매 DRAM 사이클 핵심 루프 — 커맨드 발행·카운터 감소·통계
 * dram_t::issue_col_command(): Row Buffer Hit 조건에서 RD/WR 커맨드 발행
 * dram_t::issue_row_command(): BANK_IDLE/ACTIVE 상태에 따라 ACT/PRE 커맨드 발행
 * dram_t::scheduler_fifo()  : FIFO 정책으로 mrqq 맨 앞 요청을 뱅크에 배정
 * dram_t::scheduler_frfcfs(): FR-FCFS 정책으로 Row Hit 우선 뱅크 배정 위임
 * dram_req_t::dram_req_t()  : mem_fetch에서 DRAM 레벨 요청(row/col/bk/rw) 추출
 */

#include "dram.h"          /* [한국어] 이 파일의 클래스 선언 (dram_t, dram_req_t, bankgrp_t, bank_t) */
#include "dram_sched.h"    /* [한국어] FR-FCFS 스케줄러 구현 (frfcfs_scheduler 클래스) */
#include "gpu-misc.h"      /* [한국어] StatCreate() 등 GPU 시뮬레이션 공통 유틸리티 */
#include "gpu-sim.h"       /* [한국어] gpgpu_sim 전역 상태, memory_config, DRAM_FRFCFS/DRAM_FIFO 상수 */
#include "hashing.h"       /* [한국어] bitwise_hash_function(), ipoly_hash_function() — 뱅크 인덱싱 해시 함수 */
#include "l2cache.h"       /* [한국어] memory_partition_unit 클래스 선언 — set_done(), get_mgpu() 접근 */
#include "mem_fetch.h"     /* [한국어] 메모리 요청 패킷 클래스 — get_tlx_addr(), get_is_write(), set_status() */
#include "mem_latency_stat.h"  /* [한국어] 메모리 레이턴시 통계 — memlatstat_dram_access() 선언 */

/* [한국어] DRAM_VERIFY 디버그 모드: 커맨드 발행 시 상세 로그 출력 활성화.
 * PRINT_CYCLE은 특정 사이클에 대해 ACT/PRE/RD/WR 커맨드 로그를 출력할지 제어하는 플래그이다. */
#ifdef DRAM_VERIFY
int PRINT_CYCLE = 0;  /* [한국어] 현재 사이클에 커맨드가 발행되었으면 1로 설정 → 로그 출력 트리거 */
#endif

/* [한국어] fifo_pipeline 템플릿의 명시적 인스턴스화.
 * 헤더에 선언된 fifo_pipeline<T> 템플릿을 mem_fetch와 dram_req_t 타입으로 각각 인스턴스화하여
 * rwq, mrqq, returnq 파이프라인 큐의 링크 타임 오류를 방지한다. */
template class fifo_pipeline<mem_fetch>;    /* [한국어] returnq (완료 응답 큐) 타입 인스턴스화 */
template class fifo_pipeline<dram_req_t>;   /* [한국어] rwq (RD/WR 파이프라인), mrqq (요청 대기 큐) 타입 인스턴스화 */

/*
 * [한국어]
 * dram_t 생성자 - DRAM 채널 타이밍 모델 전체 초기화
 *
 * @partition_id: 이 DRAM 채널의 고유 번호 (GPU의 메모리 파티션 인덱스, 0부터 시작)
 * @config: DRAM 타이밍 파라미터 집합 포인터 (gpgpusim.config에서 파싱된 memory_config)
 * @stats: 메모리 레이턴시/접근 통계 수집 객체 (memlatstat_dram_access 호출에 사용)
 * @mp: 이 DRAM 채널을 소유하는 memory_partition_unit (set_done(), get_mgpu() 접근)
 * @gpu: gpgpu_sim 전역 상태 (gpu_sim_cycle, gpu_tot_sim_cycle 타임스탬프 계산용)
 *
 * 초기화 순서:
 *   1. 멤버 포인터 및 설정 저장
 *   2. Row BLP 통계 카운터 전부 0으로 초기화
 *   3. 채널 전체 타이밍 카운터(CCDc, RRDc, RTWc, WTRc) 0으로 초기화
 *   4. BW 낭비 분석 카운터 0으로 초기화
 *   5. Bank Group 배열(bkgrp) calloc 할당 및 포인터 배열 구성
 *   6. Bank 배열(bk) calloc 할당, 초기 상태(BANK_IDLE), bkgrpindex 계산
 *   7. rwq/mrqq/returnq 파이프라인 큐 생성
 *   8. FR-FCFS 스케줄러 조건부 생성 (scheduler_type == DRAM_FRFCFS)
 *   9. 커맨드 카운터 전부 0 초기화
 *   10. mrqq_Dist 통계 히스토그램 생성
 *
 * 실행 컨텍스트: 시뮬레이션 초기화 단계 (한 번만 호출, 단일 스레드)
 * 호출 체인: memory_partition_unit 생성자 → [dram_t 생성자]
 */
dram_t::dram_t(unsigned int partition_id, const memory_config *config,
               memory_stats_t *stats, memory_partition_unit *mp,
               gpgpu_sim *gpu) {
  id = partition_id;              /* [한국어] 이 DRAM 채널의 파티션 번호 저장 (print()에서 "DRAM[%d]"에 사용) */
  m_memory_partition_unit = mp;  /* [한국어] 상위 memory_partition_unit 포인터 저장 (set_done() 호출 경로) */
  m_stats = stats;               /* [한국어] 통계 수집 객체 포인터 저장 (push()에서 memlatstat_dram_access 호출) */
  m_config = config;             /* [한국어] DRAM 타이밍 파라미터 포인터 저장 (tRCD, tRAS, CL, nbk 등 전반 사용) */
  m_gpu = gpu;                   /* [한국어] 시뮬레이터 전역 상태 포인터 저장 (사이클 번호 타임스탬프 계산용) */

  // rowblp
  /* [한국어] Row Buffer Locality(RBL) 및 BLP(Bank Level Parallelism) 통계 카운터 초기화.
   * 이 카운터들은 cycle() 내부에서 매 사이클 갱신되며 시뮬레이션 종료 시 print()에서 출력된다. */
  access_num = 0;                /* [한국어] 총 DRAM 접근 수 초기화 */
  hits_num = 0;                  /* [한국어] Row Buffer Hit 총 횟수 초기화 */
  read_num = 0;                  /* [한국어] 총 READ 접근 수 초기화 */
  write_num = 0;                 /* [한국어] 총 WRITE 접근 수 초기화 */
  hits_read_num = 0;             /* [한국어] READ 중 Row Buffer Hit 횟수 초기화 */
  hits_write_num = 0;            /* [한국어] WRITE 중 Row Buffer Hit 횟수 초기화 */
  banks_1time = 0;               /* [한국어] 사이클별 mrq 보유 뱅크 수 총합 (BLP 분자) 초기화 */
  banks_acess_total = 0;         /* [한국어] mrq 보유 뱅크가 하나 이상인 사이클 수 (BLP 분모) 초기화 */
  banks_acess_total_after = 0;   /* [한국어] 커맨드 발행 후 mrq 보유 뱅크 있는 사이클 수 초기화 */
  banks_time_ready = 0;          /* [한국어] Column 커맨드 즉시 발행 가능 뱅크 수 총합 초기화 */
  banks_access_ready_total = 0;  /* [한국어] Column 커맨드 발행 가능 뱅크 있는 사이클 수 초기화 */
  issued_two = 0;                /* [한국어] dual_bus: Row+Col 동시 발행 횟수 초기화 */
  issued_total = 0;              /* [한국어] Row 또는 Col 커맨드 발행된 총 사이클 수 초기화 */
  issued_total_row = 0;          /* [한국어] Row 커맨드(ACT/PRE) 발행 총 횟수 초기화 */
  issued_total_col = 0;          /* [한국어] Column 커맨드(RD/WR) 발행 총 횟수 초기화 */

  /* [한국어] 채널 전체 타이밍 카운터 초기화.
   * 이 카운터들은 커맨드 발행 후 특정 값으로 설정되어 매 사이클 DEC2ZERO로 감소한다. */
  CCDc = 0;   /* [한국어] CCD(Column-to-Column Delay) 카운터 0으로 초기화 */
  RRDc = 0;   /* [한국어] RRD(Rank-to-Rank Delay) 카운터 0으로 초기화 */
  RTWc = 0;   /* [한국어] RTW(Read-to-Write 패널티) 카운터 0으로 초기화 */
  WTRc = 0;   /* [한국어] WTR(Write-to-Read 패널티) 카운터 0으로 초기화 */

  /* [한국어] BW 낭비 원인 분석 카운터 및 병목 카운터 전부 0으로 초기화. */
  wasted_bw_row = 0;                      /* [한국어] Row 커맨드 진행 중 BW 낭비 사이클 수 */
  wasted_bw_col = 0;                      /* [한국어] 타이밍 제약으로 Col 커맨드 차단된 사이클 수 */
  util_bw = 0;                            /* [한국어] 유효 데이터 전송 사이클 수 */
  idle_bw = 0;                            /* [한국어] 요청 없는 완전 유휴 사이클 수 */
  RCDc_limit = 0;                         /* [한국어] tRCD 병목 카운터 */
  CCDLc_limit = 0;                        /* [한국어] Bank Group CCD_L 병목 카운터 */
  CCDLc_limit_alone = 0;                  /* [한국어] CCDLc 단독 병목 카운터 */
  CCDc_limit = 0;                         /* [한국어] 채널 CCD 병목 카운터 */
  WTRc_limit = 0;                         /* [한국어] WTR 병목 카운터 */
  WTRc_limit_alone = 0;                   /* [한국어] WTR 단독 병목 카운터 */
  RCDWRc_limit = 0;                       /* [한국어] tRCDWR 병목 카운터 */
  RTWc_limit = 0;                         /* [한국어] RTW 병목 카운터 */
  RTWc_limit_alone = 0;                   /* [한국어] RTW 단독 병목 카운터 */
  rwq_limit = 0;                          /* [한국어] rwq 포화로 인한 Col 커맨드 차단 카운터 */
  write_to_read_ratio_blp_rw_average = 0; /* [한국어] Row Hit 사이클에서의 WRITE/총 비율 누적합 */
  bkgrp_parallsim_rw = 0;                 /* [한국어] Row Hit 사이클별 활성 Bank Group 수 누적합 */

  rw = READ;  // read mode is default
  /* [한국어] 채널의 초기 방향을 READ로 설정.
   * issue_col_command()에서 방향 전환(READ→WRITE 또는 WRITE→READ) 시 rwq의 min_length를
   * CL(READ latency) 또는 WL(WRITE latency)로 조정하는 기준으로 사용된다. */

  /* [한국어] Bank Group 배열 동적 할당 (calloc → 0 초기화 보장).
   * bkgrp는 포인터 배열이고 실제 bankgrp_t 원소는 연속 배열로 하나만 할당한다.
   * 주의: calloc의 size 인자가 sizeof(bankgrp_t *)이 아닌 sizeof(bank_t)인 것은 원본 코드의 특성이다.
   *       bank_t >= bankgrp_t이므로 실제 bankgrp_t 원소들을 수용하기에 충분하다. */
  bkgrp = (bankgrp_t **)calloc(sizeof(bankgrp_t *), m_config->nbkgrp);  /* [한국어] nbkgrp개의 bankgrp_t 포인터 배열 할당 */
  bkgrp[0] = (bankgrp_t *)calloc(sizeof(bank_t), m_config->nbkgrp);     /* [한국어] nbkgrp개의 bankgrp_t 실제 원소 연속 배열 할당 */
  for (unsigned i = 1; i < m_config->nbkgrp; i++) {
    bkgrp[i] = bkgrp[0] + i;  /* [한국어] bkgrp[0] 연속 배열의 i번째 원소 주소를 포인터 배열에 설정 */
  }
  for (unsigned i = 0; i < m_config->nbkgrp; i++) {
    bkgrp[i]->CCDLc = 0;  /* [한국어] 그룹 i의 CCD_L 카운터 0 초기화 (calloc이 이미 0이지만 명시적으로 재확인) */
    bkgrp[i]->RTPLc = 0;  /* [한국어] 그룹 i의 RTP_L 카운터 0 초기화 */
  }

  /* [한국어] Bank 배열 동적 할당 (bkgrp와 동일한 포인터 배열 + 연속 배열 패턴 사용). */
  bk = (bank_t **)calloc(sizeof(bank_t *), m_config->nbk);  /* [한국어] nbk개의 bank_t 포인터 배열 할당 */
  bk[0] = (bank_t *)calloc(sizeof(bank_t), m_config->nbk);  /* [한국어] nbk개의 bank_t 원소 연속 배열 할당 */
  for (unsigned i = 1; i < m_config->nbk; i++) bk[i] = bk[0] + i;  /* [한국어] 포인터 배열에 각 원소 주소 설정 */
  for (unsigned i = 0; i < m_config->nbk; i++) {
    bk[i]->state = BANK_IDLE;  /* [한국어] 모든 뱅크를 IDLE(아무 Row도 열리지 않은 초기 상태)로 설정 */
    bk[i]->bkgrpindex = i / (m_config->nbk / m_config->nbkgrp);
    /* [한국어] 뱅크 i가 속하는 Bank Group 번호 계산.
     * nbk/nbkgrp는 그룹당 뱅크 수. 예: nbk=16, nbkgrp=4 → 그룹당 4뱅크 → bk[5]는 그룹 1. */
  }
  prio = 0;  /* [한국어] 라운드로빈 우선순위를 뱅크 0부터 시작하도록 초기화 */

  /* [한국어] RD/WR 파이프라인 큐(rwq) 생성.
   * 최소 길이(min_length)를 CL로, 최대 길이를 CL+1로 설정하여 CL 사이클 지연을 모델링.
   * READ 커맨드가 rwq에 push되면 CL 사이클 후 pop할 수 있어 DRAM CAS Latency를 시뮬레이션한다. */
  rwq = new fifo_pipeline<dram_req_t>("rwq", m_config->CL, m_config->CL + 1);

  /* [한국어] DRAM 입력 요청 큐(mrqq) 생성.
   * FIFO 스케줄러 사용 시 push()로 들어온 dram_req_t가 scheduler_fifo()에서 뱅크 배정 전까지 대기.
   * 최소 길이 0, 최대 길이 2 (즉시 처리 가능하도록 작은 버퍼). */
  mrqq = new fifo_pipeline<dram_req_t>("mrqq", 0, 2);

  /* [한국어] 완료 응답 큐(returnq) 생성.
   * DRAM CL 파이프라인을 모두 통과한 READ 응답을 memory_partition_unit이 꺼낼 때까지 보관.
   * gpgpu_dram_return_queue_size가 0이면 기본값 1024를 사용한다. */
  returnq = new fifo_pipeline<mem_fetch>(
      "dramreturnq", 0,
      m_config->gpgpu_dram_return_queue_size == 0  /* [한국어] 설정값이 0이면 무제한 대신 1024로 처리 */
          ? 1024
          : m_config->gpgpu_dram_return_queue_size);

  /* [한국어] FR-FCFS 스케줄러 조건부 생성.
   * DRAM_FRFCFS 모드가 아니면 NULL로 두어 scheduler_fifo()를 사용한다. */
  m_frfcfs_scheduler = NULL;  /* [한국어] 기본값 NULL (FIFO 모드 또는 초기화 전 상태) */
  if (m_config->scheduler_type == DRAM_FRFCFS)   /* [한국어] gpgpusim.config에서 FR-FCFS 스케줄러 선택된 경우 */
    m_frfcfs_scheduler = new frfcfs_scheduler(m_config, this, stats);  /* [한국어] FR-FCFS 스케줄러 생성 (dram_t 자신을 전달하여 뱅크 상태 접근) */

  /* [한국어] 커맨드 및 요청 수 통계 카운터 초기화. */
  n_cmd = 0;         /* [한국어] 총 커맨드 사이클 수 */
  n_activity = 0;    /* [한국어] 활성 사이클 수 */
  n_nop = 0;         /* [한국어] NOP 사이클 수 */
  n_act = 0;         /* [한국어] ACT 커맨드 횟수 */
  n_pre = 0;         /* [한국어] PRE 커맨드 횟수 */
  n_rd = 0;          /* [한국어] 일반 READ 커맨드 횟수 */
  n_wr = 0;          /* [한국어] 일반 WRITE 커맨드 횟수 */
  n_wr_WB = 0;       /* [한국어] L2 writeback WRITE 커맨드 횟수 */
  n_rd_L2_A = 0;     /* [한국어] L2 Write-Allocate READ 커맨드 횟수 */
  n_req = 0;         /* [한국어] 총 요청 수 */
  max_mrqs_temp = 0; /* [한국어] 임시 큐 최대 길이 추적 */
  bwutil = 0;        /* [한국어] BW utilization 누적 카운터 */
  max_mrqs = 0;      /* [한국어] 전체 시뮬레이션 큐 최대 길이 */
  ave_mrqs = 0;      /* [한국어] 큐 길이 누적 합 (평균 계산용) */

  /* [한국어] BW utilization/efficiency 히스토그램 배열 초기화 (10구간 각각 0으로). */
  for (unsigned i = 0; i < 10; i++) {
    dram_util_bins[i] = 0;  /* [한국어] i번째 utilization 구간 (0~9 → 0~9%, 10~19%, ..., 90~100%) */
    dram_eff_bins[i] = 0;   /* [한국어] i번째 efficiency 구간 */
  }
  last_n_cmd = last_n_activity = last_bwutil = 0;  /* [한국어] 이전 샘플링 인터벌 스냅샷 값들 초기화 */

  /* [한국어] AerialVision 시각화용 인터벌 카운터(partial) 전부 초기화.
   * visualizer_print()가 출력 후 이 값들을 0으로 리셋하여 다음 인터벌을 측정한다. */
  n_cmd_partial = 0;       /* [한국어] 인터벌 커맨드 사이클 수 */
  n_activity_partial = 0;  /* [한국어] 인터벌 활성 사이클 수 */
  n_nop_partial = 0;       /* [한국어] 인터벌 NOP 사이클 수 */
  n_act_partial = 0;       /* [한국어] 인터벌 ACT 횟수 */
  n_pre_partial = 0;       /* [한국어] 인터벌 PRE 횟수 */
  n_req_partial = 0;       /* [한국어] 인터벌 요청 수 */
  ave_mrqs_partial = 0;    /* [한국어] 인터벌 큐 길이 누적 합 */
  bwutil_partial = 0;      /* [한국어] 인터벌 BW utilization 누적 */

  /* [한국어] mrqq 큐 길이 분포 히스토그램 통계 객체 생성.
   * queue_limit()이 0(무제한)이면 최대 64 구간으로 추적한다. */
  if (queue_limit())                                            /* [한국어] 유한한 큐 크기 설정인 경우 */
    mrqq_Dist = StatCreate("mrqq_length", 1, queue_limit());   /* [한국어] 1~queue_limit() 구간 히스토그램 생성 */
  else                                             // queue length is unlimited;
    mrqq_Dist = StatCreate("mrqq_length", 1, 64);  // track up to 64 entries
    /* [한국어] 무제한 큐의 경우 최대 64 항목까지만 추적 (메모리 절약) */
}

/*
 * [한국어]
 * dram_t::full - DRAM 요청 큐 포화 여부 확인
 *
 * @is_write: true이면 쓰기 전용 큐 포화 여부, false이면 읽기(또는 통합) 큐 포화 여부
 * @return: 큐가 가득 차 있어 새 요청을 받을 수 없으면 true, 여유가 있으면 false
 *
 * 스케줄러 타입에 따라 다른 방식으로 포화를 판단한다:
 *   - FR-FCFS + 큐 크기 무제한(0): 항상 false (무제한 수용)
 *   - FR-FCFS + 읽기/쓰기 분리 큐(seperate_write_queue_enabled): 방향별 큐 크기와 비교
 *   - FR-FCFS + 통합 큐: 통합 큐 대기 수와 gpgpu_frfcfs_dram_sched_queue_size 비교
 *   - FIFO: mrqq->full() 반환
 * memory_partition_unit이 push() 전에 이 함수로 수락 가능 여부를 확인한다.
 *
 * 호출 체인: memory_partition_unit::full() → [dram_t::full()]
 */
bool dram_t::full(bool is_write) const {
  if (m_config->scheduler_type == DRAM_FRFCFS) {  /* [한국어] FR-FCFS 스케줄러 사용 중인 경우 */
    if (m_config->gpgpu_frfcfs_dram_sched_queue_size == 0) return false;  /* [한국어] 큐 크기 0 = 무제한 → 항상 수용 가능 */
    if (m_config->seperate_write_queue_enabled) {  /* [한국어] 읽기/쓰기 전용 큐를 분리 운영하는 설정인 경우 */
      if (is_write)
        return m_frfcfs_scheduler->num_write_pending() >=
               m_config->gpgpu_frfcfs_dram_write_queue_size;
        /* [한국어] WRITE 요청: 쓰기 전용 큐의 대기 수가 gpgpu_frfcfs_dram_write_queue_size 이상이면 포화 */
      else
        return m_frfcfs_scheduler->num_pending() >=
               m_config->gpgpu_frfcfs_dram_sched_queue_size;
        /* [한국어] READ 요청: 통합(읽기) 큐의 대기 수가 gpgpu_frfcfs_dram_sched_queue_size 이상이면 포화 */
    } else
      return m_frfcfs_scheduler->num_pending() >=
             m_config->gpgpu_frfcfs_dram_sched_queue_size;
      /* [한국어] 큐 분리 없음: 통합 큐의 대기 수를 단일 임계값과 비교 */
  } else
    return mrqq->full();  /* [한국어] FIFO 스케줄러: mrqq의 물리적 포화 여부 반환 (용량 2) */
}

/*
 * [한국어]
 * dram_t::que_length - 현재 DRAM 스케줄러 큐에 대기 중인 요청 수 반환
 *
 * @return: FR-FCFS 모드이면 frfcfs_scheduler::num_pending(), FIFO 모드이면 mrqq->get_length()
 *
 * memory_partition_unit이나 통계 수집 코드가 DRAM 채널의 현재 부하 수준을 파악할 때 호출한다.
 * max_mrqs_temp 추적 및 ave_mrqs 계산에도 활용된다.
 *
 * 호출 체인: memory_partition_unit / dram_t::cycle() → [dram_t::que_length()]
 */
unsigned dram_t::que_length() const {
  unsigned nreqs = 0;  /* [한국어] 대기 요청 수 결과 변수 초기화 */
  if (m_config->scheduler_type == DRAM_FRFCFS) {  /* [한국어] FR-FCFS 스케줄러 사용 시 내부 대기 큐에서 집계 */
    nreqs = m_frfcfs_scheduler->num_pending();     /* [한국어] frfcfs_scheduler가 관리하는 전체 대기 요청 수 반환 */
  } else {
    nreqs = mrqq->get_length();  /* [한국어] FIFO 모드: mrqq 파이프라인 큐의 현재 길이 반환 */
  }
  return nreqs;  /* [한국어] 집계된 대기 요청 수 반환 */
}

/*
 * [한국어]
 * dram_t::returnq_full - 완료 응답 큐(returnq) 포화 여부 확인
 *
 * @return: returnq가 가득 찼으면 true
 *
 * dram_t::cycle()에서 rwq로부터 완료된 READ 요청을 returnq에 push하기 전에 이 함수로
 * 공간 여부를 확인한다. returnq가 포화되면 rwq->pop()을 건너뛰어 역압력(backpressure)이 발생한다.
 *
 * 호출 체인: dram_t::cycle() → [dram_t::returnq_full()]
 */
bool dram_t::returnq_full() const { return returnq->full(); }

/*
 * [한국어]
 * dram_t::queue_limit - DRAM 스케줄러 큐의 설정된 최대 크기 반환
 *
 * @return: gpgpu_frfcfs_dram_sched_queue_size (0이면 무제한 의미)
 *
 * 생성자에서 mrqq_Dist 통계 히스토그램 범위 설정에 사용된다.
 * 0 반환 시 호출자는 무제한으로 처리해야 한다.
 *
 * 호출 체인: dram_t 생성자 → [dram_t::queue_limit()]
 */
unsigned int dram_t::queue_limit() const {
  return m_config->gpgpu_frfcfs_dram_sched_queue_size;  /* [한국어] gpgpusim.config의 DRAM 스케줄러 큐 크기 설정값 반환 */
}

/*
 * [한국어]
 * dram_req_t 생성자 - mem_fetch를 DRAM 레벨 요청으로 변환
 *
 * @mf: L2 캐시 미스로 인해 DRAM까지 내려온 메모리 요청 패킷
 * @banks: 이 DRAM 채널의 총 뱅크 수 (memory_config::nbk) — 뱅크 인덱스 유효성 검증용
 * @dram_bnk_indexing_policy: 뱅크 인덱싱 정책 열거값 (LINEAR/BITWISE_XOR/IPOLY/CUSTOM)
 * @gpu: 시뮬레이터 전역 상태 (gpu_tot_sim_cycle + gpu_sim_cycle으로 타임스탬프 계산)
 *
 * mem_fetch의 tlx_addr(물리 주소 디코딩 결과 addrdec_t)에서 row/col/bk를 추출한다.
 * 뱅크 인덱싱 정책에 따라 bk를 원래 값(LINEAR), XOR 해시(BITWISE_XORING), 또는
 * IPOLY 해시로 재매핑하여 뱅크 충돌 분산 효과를 제공할 수 있다.
 * txbytes/dqbytes는 0으로 초기화하여 전송 시작 전 상태를 나타낸다.
 *
 * 실행 컨텍스트: dram_t::push() 내부, 단일 시뮬레이션 스레드
 * 호출 체인: dram_t::push() → [dram_req_t 생성자] → mrqq->push(mrq)
 */
dram_req_t::dram_req_t(class mem_fetch *mf, unsigned banks,
                       unsigned dram_bnk_indexing_policy,
                       class gpgpu_sim *gpu) {
  txbytes = 0;  /* [한국어] 전송된 바이트 수 0으로 초기화 (RD/WR 커맨드 발행마다 dram_atom_size씩 증가) */
  dqbytes = 0;  /* [한국어] DQ 출력 바이트 수 0으로 초기화 (rwq->pop() 시마다 dram_atom_size씩 증가) */
  data = mf;    /* [한국어] 원본 mem_fetch 포인터 저장 (완료 시 returnq에 push하거나 delete에 사용) */
  m_gpu = gpu;  /* [한국어] 시뮬레이터 전역 상태 포인터 저장 (타임스탬프 계산에 사용) */

  const addrdec_t &tlx = mf->get_tlx_addr();
  /* [한국어] mem_fetch의 물리 주소 디코딩 결과(addrdec_t) 참조를 얻는다.
   * addrdec_t는 GPU 물리 주소를 chip/bank/row/col 필드로 분해한 구조체이다 (addrdec.h 참조).
   * 이 참조를 통해 DRAM 레벨 좌표(bk, row, col)를 추출한다. */

  /* [한국어] 뱅크 인덱싱 정책에 따라 최종 뱅크 번호(bk)를 결정한다.
   * 정책별로 뱅크 분산 방식이 다르며 설정 파일의 gpgpu_dram_bnk_indexing_policy 옵션으로 선택된다. */
  switch (dram_bnk_indexing_policy) {
    case LINEAR_BK_INDEX: {
      bk = tlx.bk;  /* [한국어] 선형 정책: addrdec_t가 디코딩한 뱅크 번호를 변환 없이 그대로 사용 */
      break;
    }
    case BITWISE_XORING_BK_INDEX: {
      // xoring bank bits with lower bits of the page
      bk = bitwise_hash_function(tlx.row, tlx.bk, banks);
      /* [한국어] XOR 해싱: Row의 하위 비트와 기본 뱅크 번호를 XOR하여 뱅크 번호 재계산.
       * 같은 열의 연속 주소가 다른 뱅크에 분산되어 뱅크 충돌(bank conflict) 감소. */
      assert(bk < banks);  /* [한국어] 해시 결과가 유효 뱅크 범위 내인지 검증 */
      break;
    }
    case IPOLY_BK_INDEX: {
      /*IPOLY for bank indexing function from "Pseudo-randomly interleaved
       * memory." Rau, B. R et al. ISCA 1991
       * http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=348DEA37A3E440473B3C075EAABC63B6?doi=10.1.1.12.7149&rep=rep1&type=pdf
       */
      // xoring bank bits with lower bits of the page
      bk = ipoly_hash_function(tlx.row, tlx.bk, banks);
      /* [한국어] IPOLY 해싱: 다항식 기반 해시로 뱅크 번호 결정. XOR보다 더 균등한 분산 제공.
       * Rau et al. ISCA 1991 "Pseudo-randomly interleaved memory" 논문 알고리즘 구현. */
      assert(bk < banks);  /* [한국어] 해시 결과가 유효 뱅크 범위 내인지 검증 */
      break;
    }
    case CUSTOM_BK_INDEX:
      /* No custom set function implemented */
      // Do you custom index here
      /* [한국어] 사용자 정의 인덱싱: 현재 구현 없음 (bk를 설정하지 않으므로 tlx.bk가 그대로 유지됨).
       * 새로운 인덱싱 정책 실험 시 이 분기에 코드를 추가한다. */
      break;
    default:
      assert("\nUndefined bank index function.\n" && 0);
      /* [한국어] 정의되지 않은 뱅크 인덱싱 정책 → 시뮬레이션 중단 (설정 파일 오류) */
      break;
  }

  row = tlx.row;  /* [한국어] 물리 주소 디코딩 결과에서 DRAM Row 번호 추출 */
  col = tlx.col;  /* [한국어] 물리 주소 디코딩 결과에서 DRAM Column 번호 추출 */
  nbytes = mf->get_data_size();  /* [한국어] 이 요청의 총 전송 바이트 수 (보통 캐시 라인 크기, 예: 128 bytes) */

  timestamp = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
  /* [한국어] 이 요청이 생성된 절대 시뮬레이션 사이클 번호 기록.
   * gpu_tot_sim_cycle은 이전 커널들의 누적 사이클, gpu_sim_cycle은 현재 커널 사이클. */

  addr = mf->get_addr();  /* [한국어] 원본 물리 주소 저장 (디버그 출력 및 로깅용) */
  insertion_time = (unsigned)m_gpu->gpu_sim_cycle;
  /* [한국어] 현재 커널 기준 삽입 시점 사이클 번호 저장 (레이턴시 통계 계산 시 기준값). */

  rw = data->get_is_write() ? WRITE : READ;
  /* [한국어] mem_fetch의 쓰기 여부로 이 요청의 방향(READ 또는 WRITE)을 결정.
   * issue_col_command()에서 RD/WR 커맨드 분기 및 BLP 통계 분류에 사용된다. */
}

/*
 * [한국어]
 * dram_t::push - 새 메모리 요청을 DRAM 채널에 투입
 *
 * @data: L2 캐시 미스(또는 writeback)로 DRAM까지 내려온 메모리 요청 패킷
 *
 * mem_fetch를 dram_req_t로 변환하여 DRAM 스케줄러 큐에 삽입한다. 구체적으로:
 *   1. chip 번호 검증 (이 파티션으로 올바르게 라우팅되었는지 확인)
 *   2. dram_req_t 생성 (row/col/bk/rw 추출, 타임스탬프 기록)
 *   3. mem_fetch 상태를 IN_PARTITION_MC_INTERFACE_QUEUE로 갱신
 *   4. mrqq(FIFO) 또는 frfcfs_scheduler(FR-FCFS)에 삽입
 *   5. n_req 카운터 증가, max_mrqs_temp 갱신
 *   6. 레이턴시 통계 기록
 * 주의: 이 함수를 호출하기 전에 full()로 큐 포화 여부를 반드시 확인해야 한다.
 *
 * 실행 컨텍스트: 단일 시뮬레이션 스레드 (memory_partition_unit::push() 내부에서 호출)
 * 호출 체인: memory_partition_unit::push() → [dram_t::push()] → new dram_req_t()
 *           → mrqq->push(mrq) → m_stats->memlatstat_dram_access()
 */
void dram_t::push(class mem_fetch *data) {
  assert(id == data->get_tlx_addr()
                   .chip);  // Ensure request is in correct memory partition
  /* [한국어] chip 번호가 이 파티션 id와 일치하는지 검증.
   * addrdec_t::chip이 이 파티션의 번호와 다르면 잘못된 파티션으로 라우팅된 것이므로 중단. */

  dram_req_t *mrq =
      new dram_req_t(data, m_config->nbk, m_config->dram_bnk_indexing_policy,
                     m_memory_partition_unit->get_mgpu());
  /* [한국어] mem_fetch를 DRAM 레벨 요청(dram_req_t)으로 변환하여 힙에 할당.
   * get_mgpu()는 memory_partition_unit이 보유한 gpgpu_sim 포인터를 반환하여
   * dram_req_t 생성자에서 타임스탬프 계산에 사용된다. */

  data->set_status(IN_PARTITION_MC_INTERFACE_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  /* [한국어] mem_fetch의 상태를 IN_PARTITION_MC_INTERFACE_QUEUE로 갱신하여
   * 현재 절대 사이클 번호와 함께 기록한다. 이 상태는 레이턴시 추적에 사용된다. */

  mrqq->push(mrq);
  /* [한국어] 생성된 dram_req_t를 mrqq(입력 대기 큐)에 삽입.
   * FR-FCFS 모드에서는 scheduler_frfcfs()가 mrqq가 아닌 내부 큐에서 요청을 가져간다.
   * 그러나 dram_req_t 자체는 항상 일단 mrqq에 들어간 뒤 scheduler가 이동시킨다. */

  // stats...
  n_req += 1;          /* [한국어] 이 DRAM 채널의 총 요청 수 누적 (시뮬레이션 전체 기간) */
  n_req_partial += 1;  /* [한국어] 인터벌 요청 수 누적 (visualizer_print()에서 리셋) */
  if (m_config->scheduler_type == DRAM_FRFCFS) {
    unsigned nreqs = m_frfcfs_scheduler->num_pending();  /* [한국어] FR-FCFS 큐의 현재 대기 요청 수 조회 */
    if (nreqs > max_mrqs_temp) max_mrqs_temp = nreqs;    /* [한국어] 현재 시뮬레이션 구간 최대 큐 길이 갱신 */
  } else {
    max_mrqs_temp = (max_mrqs_temp > mrqq->get_length()) ? max_mrqs_temp
                                                         : mrqq->get_length();
    /* [한국어] FIFO 모드: mrqq의 현재 길이와 비교하여 max_mrqs_temp 갱신. 3항 연산자로 max() 구현. */
  }
  m_stats->memlatstat_dram_access(data);
  /* [한국어] 메모리 레이턴시 통계에 이 DRAM 접근 사실을 기록.
   * data(mem_fetch)의 타임스탬프와 접근 유형을 기반으로 통계를 수집한다. */
}

/*
 * [한국어]
 * dram_t::scheduler_fifo - FIFO 스케줄러: mrqq 맨 앞 요청을 해당 뱅크에 배정
 *
 * mrqq(입력 큐)의 맨 앞 요청을 꺼내 해당 뱅크(bk[bkn]->mrq)에 배정한다.
 * 배정 조건: bk[bkn]->mrq가 NULL이어야 한다 (이미 다른 요청을 서비스 중이면 대기).
 * 단순 FCFS(First Come First Served) 정책이라 Row Buffer Hit/Miss를 고려하지 않는다.
 * Row Miss 요청도 도착 순서대로 처리되므로 불필요한 ACT/PRE가 많이 발생할 수 있다.
 *
 * 실행 컨텍스트: dram_t::cycle() 내부, 매 DRAM 사이클 (scheduler_type == DRAM_FIFO일 때)
 * 호출 체인: dram_t::cycle() → [scheduler_fifo()] → mrqq->pop() → bk[bkn]->mrq 배정
 */
void dram_t::scheduler_fifo() {
  if (!mrqq->empty()) {  /* [한국어] 대기 중인 요청이 있는 경우에만 진행 */
    unsigned int bkn;  /* [한국어] 요청의 대상 뱅크 번호 */
    dram_req_t *head_mrqq = mrqq->top();  /* [한국어] mrqq의 맨 앞 요청을 peek (아직 pop하지 않음) */
    head_mrqq->data->set_status(
        IN_PARTITION_MC_BANK_ARB_QUEUE,
        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    /* [한국어] mem_fetch의 상태를 IN_PARTITION_MC_BANK_ARB_QUEUE(뱅크 중재 큐)로 갱신.
     * 현재 절대 사이클 번호와 함께 기록하여 레이턴시 추적을 지원한다. */
    bkn = head_mrqq->bk;  /* [한국어] 이 요청이 배정되어야 하는 뱅크 번호 추출 */
    if (!bk[bkn]->mrq) bk[bkn]->mrq = mrqq->pop();
    /* [한국어] 해당 뱅크에 현재 서비스 중인 요청이 없으면(mrq == NULL) mrqq에서 pop하여 뱅크에 배정.
     * 이미 다른 요청이 배정된 뱅크라면 mrqq에 그대로 두고 다음 사이클에 다시 시도한다.
     * 이 단순 정책은 같은 뱅크의 요청이 연속으로 쌓일 때 다른 뱅크 요청의 발행을 지연시킬 수 있다. */
  }
}

/* [한국어] DEC2ZERO(x): 카운터 x를 0이 될 때까지 1씩 감소시키는 매크로.
 * DRAM 타이밍 카운터는 커맨드 발행 시 특정 값으로 설정되고 매 사이클 이 매크로로 감소한다.
 * 0 이하로 내려가지 않도록 3항 연산자로 보호한다. cycle() 끝에서 모든 카운터에 적용된다. */
#define DEC2ZERO(x) x = (x) ? (x - 1) : 0;

/* [한국어] SWAP(a, b): XOR 기반 임시 변수 없는 정수 스왑 매크로.
 * 포인터나 복잡한 타입에는 사용하지 않으며, unsigned 정수 타입에만 안전하다.
 * 현재 코드에서는 직접 사용되지 않고 보조 유틸리티로 선언되어 있다. */
#define SWAP(a, b) \
  a ^= b;          \
  b ^= a;          \
  a ^= b;

/*
 * [한국어]
 * dram_t::cycle - DRAM 채널 사이클 단위 타이밍 진행 함수 (핵심)
 *
 * 매 DRAM 클럭 사이클마다 memory_partition_unit::dram_cycle()에 의해 호출된다.
 * 실제 DDR SDRAM 컨트롤러가 클럭마다 수행하는 동작을 소프트웨어로 모델링한다.
 *
 * 수행 단계:
 *   [단계 1] rwq(RD/WR 파이프라인 큐) pop → CL/WL 파이프라인 완료된 요청 처리
 *            - READ 요청: returnq에 push (SM으로 응답 전송 대기)
 *            - WRITE/Writeback 요청: set_done() 후 delete (응답 불필요)
 *   [단계 2] 스케줄러 실행 → 다음 서비스할 요청을 뱅크에 배정
 *            - DRAM_FIFO: scheduler_fifo() (FCFS 정책)
 *            - DRAM_FRFCFS: scheduler_frfcfs() (Row Hit 우선 정책)
 *   [단계 3] 통계 수집 — Row Buffer Locality, BLP, BW 병목 분석 카운터 갱신
 *   [단계 4] 커맨드 발행 — 각 뱅크를 라운드로빈으로 순회하며:
 *            - dual_bus: Row 커맨드 1개 + Column 커맨드 1개 동시 발행 가능
 *            - single_bus: Row 또는 Column 중 하나만 발행
 *            - issue_col_command(j): Row Buffer Hit 조건에서 RD/WR 발행
 *            - issue_row_command(j): IDLE 뱅크에 ACT 또는 Row Miss 시 PRE 발행
 *   [단계 5] BW 낭비 원인 분석 카운터 갱신 (타이밍 제약별 병목 추적)
 *   [단계 6] 모든 타이밍 카운터 DEC2ZERO 감소 (RRDc, CCDc, RTWc, WTRc, 각 뱅크 카운터들)
 *   [단계 7] DRAM_VISUALIZE 활성화 시 visualize() 호출
 *
 * 실행 컨텍스트: 호스트 CPU 시뮬레이션 루프 단일 스레드
 * 호출 체인: gpgpu_sim::cycle() → memory_partition_unit::dram_cycle()
 *           → [dram_t::cycle()] → issue_col_command() / issue_row_command()
 */
void dram_t::cycle() {
  /* [단계 1] rwq 파이프라인 큐에서 CL/WL 지연이 만료된 요청을 꺼내 완료 처리한다.
   * returnq가 포화된 경우 pop을 건너뛰어 역압력(backpressure)이 rwq로 전파된다. */
  if (!returnq->full()) {  /* [한국어] returnq에 공간이 있을 때만 rwq에서 pop 시도 */
    dram_req_t *cmd = rwq->pop();  /* [한국어] CL(또는 WL) 사이클 경과 후 rwq 파이프라인에서 꺼낸 요청 */
    if (cmd) {  /* [한국어] pop에서 실제 요청이 있는 경우 (NULL이면 파이프라인이 비어있음) */
#ifdef DRAM_VIEWCMD
      printf("\tDQ: BK%d Row:%03x Col:%03x", cmd->bk, cmd->row,
             cmd->col + cmd->dqbytes);
      /* [한국어] DRAM_VIEWCMD 디버그 모드: 현재 처리 중인 요청의 뱅크/로우/컬럼 출력 */
#endif
      cmd->dqbytes += m_config->dram_atom_size;
      /* [한국어] DQ(Data Queue) 버스를 통해 dram_atom_size 바이트를 전송 완료로 기록.
       * dram_atom_size는 한 사이클에 전송하는 최소 단위 (버스 폭 × BL 비율) */

      if (cmd->dqbytes >= cmd->nbytes) {
        /* [한국어] 이 요청의 모든 바이트가 DQ를 통해 출력 완료 → 최종 처리 단계 */
        mem_fetch *data = cmd->data;  /* [한국어] 원본 mem_fetch 포인터 추출 */
        data->set_status(IN_PARTITION_MC_RETURNQ,
                         m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        /* [한국어] mem_fetch 상태를 IN_PARTITION_MC_RETURNQ로 갱신하여 완료 단계 기록 */
        if (data->get_access_type() != L1_WRBK_ACC &&
            data->get_access_type() != L2_WRBK_ACC) {
          /* [한국어] L1/L2 Writeback 요청이 아닌 경우(일반 READ 또는 L2 Write-Allocate):
           * SM에 응답을 돌려보내야 하므로 returnq에 push한다. */
          data->set_reply();    /* [한국어] mem_fetch를 응답 패킷으로 전환 (요청 → 응답 방향) */
          returnq->push(data);  /* [한국어] returnq에 삽입 → memory_partition_unit이 꺼내 ICNT 경유 SM으로 전송 */
        } else {
          /* [한국어] L1/L2 Writeback 요청: 데이터를 DRAM에 썼으므로 응답이 필요 없음.
           * set_done()으로 완료를 상위에 알리고 mem_fetch 객체를 delete한다. */
          m_memory_partition_unit->set_done(data);  /* [한국어] memory_partition_unit에 writeback 완료 통보 */
          delete data;  /* [한국어] writeback mem_fetch 객체 메모리 해제 */
        }
        delete cmd;  /* [한국어] 처리 완료된 dram_req_t 객체 메모리 해제 */
      }
#ifdef DRAM_VIEWCMD
      printf("\n");  /* [한국어] DRAM_VIEWCMD 디버그 모드: 출력 줄 완료 */
#endif
    }
  }

  /* check if the upcoming request is on an idle bank */
  /* Should we modify this so that multiple requests are checked? */

  /* [단계 2] 스케줄러 실행 — 다음 처리할 요청을 대기 큐에서 뱅크에 배정한다.
   * 스케줄러는 mrq가 NULL인 뱅크에 요청을 배정하며, 이 배정 후 cycle() 내에서 커맨드 발행을 시도한다. */
  switch (m_config->scheduler_type) {
    case DRAM_FIFO:
      scheduler_fifo();  /* [한국어] FIFO(FCFS) 정책: mrqq 맨 앞 요청을 해당 뱅크에 배정 */
      break;
    case DRAM_FRFCFS:
      scheduler_frfcfs();  /* [한국어] FR-FCFS 정책: Row Hit 요청 우선 배정 (frfcfs_scheduler 위임) */
      break;
    default:
      printf("Error: Unknown DRAM scheduler type\n");  /* [한국어] 알 수 없는 스케줄러 타입 오류 메시지 */
      assert(0);  /* [한국어] 설정 파일 오류 → 시뮬레이션 중단 */
  }

  /* [한국어] 큐 길이 통계 갱신: 스케줄러 실행 후 현재 대기 큐 길이를 max_mrqs 및 ave_mrqs에 반영 */
  if (m_config->scheduler_type == DRAM_FRFCFS) {
    unsigned nreqs = m_frfcfs_scheduler->num_pending();  /* [한국어] FR-FCFS 스케줄러 내부 큐의 현재 대기 수 */
    if (nreqs > max_mrqs) {
      max_mrqs = nreqs;  /* [한국어] 전체 시뮬레이션 구간 최대 큐 길이 갱신 */
    }
    ave_mrqs += nreqs;          /* [한국어] 평균 큐 길이 계산을 위한 누적 합 (나중에 n_cmd로 나눔) */
    ave_mrqs_partial += nreqs;  /* [한국어] 인터벌 평균 큐 길이 누적 합 */
  } else {
    if (mrqq->get_length() > max_mrqs) {
      max_mrqs = mrqq->get_length();  /* [한국어] FIFO 모드: mrqq 현재 길이로 최대값 갱신 */
    }
    ave_mrqs += mrqq->get_length();          /* [한국어] FIFO 모드: mrqq 길이 누적 */
    ave_mrqs_partial += mrqq->get_length();  /* [한국어] FIFO 모드: 인터벌 mrqq 길이 누적 */
  }

  unsigned k = m_config->nbk;  /* [한국어] 비유휴 뱅크 수 카운터 (k > 0이면 n_activity 증가) */
  bool issued = false;  /* [한국어] 이 사이클에 Row 또는 Column 커맨드가 발행되었는지 여부 */

  // collect row buffer locality, BLP and other statistics
  /////////////////////////////////////////////////////////////////////////
  /* [단계 3] Row Buffer Locality, Bank Level Parallelism(BLP), BW 효율 통계 수집 */

  /* [한국어] 현재 뱅크별 mrq 보유 여부로 총 활성 뱅크 수(memory_pending) 집계 → BLP 분자 */
  unsigned int memory_pending = 0;  /* [한국어] mrq가 배정된 뱅크 수 (서비스 대기 중) */
  for (unsigned i = 0; i < m_config->nbk; i++) {
    if (bk[i]->mrq) memory_pending++;  /* [한국어] 뱅크 i에 서비스 대기 중인 요청이 있으면 카운트 */
  }
  banks_1time += memory_pending;           /* [한국어] BLP 분자: 활성 뱅크 수를 매 사이클 누적 */
  if (memory_pending > 0) banks_acess_total++;  /* [한국어] BLP 분모: 하나 이상의 뱅크가 활성인 사이클 수 */

  /* [한국어] Row Buffer Hit 상태 뱅크를 방향별(READ/WRITE)로 집계 → BLP_Col 통계 */
  unsigned int memory_pending_rw = 0;  /* [한국어] Row Buffer Hit 상태인 뱅크 수 (RD+WR 합산) */
  unsigned read_blp_rw = 0;            /* [한국어] Row Buffer Hit 상태에서 READ를 대기 중인 뱅크 수 */
  unsigned write_blp_rw = 0;           /* [한국어] Row Buffer Hit 상태에서 WRITE를 대기 중인 뱅크 수 */
  std::bitset<8> bnkgrp_rw_found;  // assume max we have 8 bank groups
  /* [한국어] Bank Group Level Parallelism 추적용 비트셋.
   * Row Buffer Hit 요청을 보유한 Bank Group의 비트를 1로 설정하여 활성 그룹 수를 count()로 집계. */

  for (unsigned j = 0; j < m_config->nbk; j++) {
    unsigned grp = get_bankgrp_number(j);  /* [한국어] 뱅크 j가 속하는 Bank Group 번호 계산 */
    if (bk[j]->mrq &&
        (((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
          (bk[j]->state == BANK_ACTIVE)))) {
      /* [한국어] 뱅크 j가 ACTIVE 상태이고, 현재 열린 Row와 요청의 Row가 일치(Row Buffer Hit)하며 READ인 경우 */
      memory_pending_rw++;     /* [한국어] Row Hit 뱅크 수 증가 */
      read_blp_rw++;           /* [한국어] READ Row Hit 뱅크 수 증가 */
      bnkgrp_rw_found.set(grp);  /* [한국어] 이 Bank Group이 활성임을 비트셋에 마킹 */
    } else if (bk[j]->mrq &&
               (((bk[j]->curr_row == bk[j]->mrq->row) &&
                 (bk[j]->mrq->rw == WRITE) && (bk[j]->state == BANK_ACTIVE)))) {
      /* [한국어] 뱅크 j가 ACTIVE 상태이고, Row Buffer Hit이며 WRITE인 경우 */
      memory_pending_rw++;     /* [한국어] Row Hit 뱅크 수 증가 */
      write_blp_rw++;          /* [한국어] WRITE Row Hit 뱅크 수 증가 */
      bnkgrp_rw_found.set(grp);  /* [한국어] 이 Bank Group이 활성임을 비트셋에 마킹 */
    }
  }
  banks_time_rw += memory_pending_rw;  /* [한국어] BLP_Col 분자: Row Hit 뱅크 수 누적 */
  bkgrp_parallsim_rw += bnkgrp_rw_found.count();
  /* [한국어] Bank Group Level Parallelism 분자: 활성 Bank Group 수 누적 */
  if (memory_pending_rw > 0) {
    write_to_read_ratio_blp_rw_average +=
        (double)write_blp_rw / (write_blp_rw + read_blp_rw);
    /* [한국어] Row Hit 사이클에서 WRITE/(WRITE+READ) 비율 누적 (평균 계산용 분자) */
    banks_access_rw_total++;  /* [한국어] Row Hit 뱅크가 하나 이상인 사이클 수 (BLP_Col 분모) */
  }

  /* [한국어] 즉시 Column 커맨드 발행 가능 뱅크 수 집계 (모든 타이밍 제약이 해제된 뱅크) */
  unsigned int memory_Pending_ready = 0;  /* [한국어] 이번 사이클에 RD/WR 커맨드를 즉시 발행 가능한 뱅크 수 */
  for (unsigned j = 0; j < m_config->nbk; j++) {
    unsigned grp = get_bankgrp_number(j);  /* [한국어] 뱅크 j의 Bank Group 번호 */
    if (bk[j]->mrq &&
        ((!CCDc && !bk[j]->RCDc && !(bkgrp[grp]->CCDLc) &&
          (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
          (WTRc == 0) && (bk[j]->state == BANK_ACTIVE) && !rwq->full()) ||
         /* [한국어] READ 즉시 발행 가능 조건:
          *   CCDc==0 (채널 CCD 해제), RCDc==0 (ACT→RD 딜레이 해제),
          *   CCDLc==0 (Bank Group CCD_L 해제), Row Hit, READ 방향,
          *   WTRc==0 (WTR 패널티 없음), BANK_ACTIVE, rwq 여유 있음 */
         (!CCDc && !bk[j]->RCDWRc && !(bkgrp[grp]->CCDLc) &&
          (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == WRITE) &&
          (RTWc == 0) && (bk[j]->state == BANK_ACTIVE) && !rwq->full()))) {
         /* [한국어] WRITE 즉시 발행 가능 조건:
          *   CCDc==0, RCDWRc==0 (ACT→WR 딜레이 해제), CCDLc==0, Row Hit, WRITE 방향,
          *   RTWc==0 (RTW 패널티 없음), BANK_ACTIVE, rwq 여유 있음 */
      memory_Pending_ready++;  /* [한국어] 즉시 발행 가능 뱅크 수 증가 */
    }
  }
  banks_time_ready += memory_Pending_ready;  /* [한국어] BLP_Ready 분자: 즉시 발행 가능 뱅크 수 누적 */
  if (memory_Pending_ready > 0) banks_access_ready_total++;
  /* [한국어] BLP_Ready 분모: 즉시 발행 가능 뱅크가 하나 이상인 사이클 수 */
  ///////////////////////////////////////////////////////////////////////////////////

  bool issued_col_cmd = false;  /* [한국어] 이 사이클에 Column 커맨드(RD/WR)가 발행되었는지 여부 */
  bool issued_row_cmd = false;  /* [한국어] 이 사이클에 Row 커맨드(ACT/PRE)가 발행되었는지 여부 */

  /* [단계 4] 커맨드 발행: 뱅크를 prio 기준 라운드로빈으로 순회하며 커맨드를 발행한다.
   * dual_bus_interface 옵션에 따라 Row+Col 동시 발행(dual) 또는 하나만 발행(single)한다. */
  if (m_config->dual_bus_interface) {
    // dual bus interface
    // issue one row command and one column command
    /* [한국어] Dual Bus Interface: 같은 사이클에 Row 커맨드(ACT/PRE) 1개와 Column 커맨드(RD/WR) 1개를
     * 독립적으로 발행할 수 있다. Row 버스와 Column 버스가 물리적으로 분리되어 있을 때 사용한다. */
    for (unsigned i = 0; i < m_config->nbk; i++) {
      unsigned j = (i + prio) % m_config->nbk;  /* [한국어] prio 기준 라운드로빈 뱅크 인덱스 계산 */
      issued_col_cmd = issue_col_command(j);     /* [한국어] 뱅크 j에 Column 커맨드(RD/WR) 발행 시도 */
      if (issued_col_cmd) break;  /* [한국어] 발행 성공 시 더 이상 다른 뱅크를 순회하지 않음 */
    }
    for (unsigned i = 0; i < m_config->nbk; i++) {
      unsigned j = (i + prio) % m_config->nbk;  /* [한국어] Column 커맨드와 독립적으로 Row 커맨드 발행 시도 */
      issued_row_cmd = issue_row_command(j);     /* [한국어] 뱅크 j에 Row 커맨드(ACT/PRE) 발행 시도 */
      if (issued_row_cmd) break;  /* [한국어] 발행 성공 시 순회 종료 */
    }
    for (unsigned i = 0; i < m_config->nbk; i++) {
      unsigned j = (i + prio) % m_config->nbk;
      if (!bk[j]->mrq) {  /* [한국어] 이 뱅크에 배정된 요청이 없는 경우 유휴(idle) 처리 */
        if (!CCDc && !RRDc && !RTWc && !WTRc && !bk[j]->RCDc && !bk[j]->RASc &&
            !bk[j]->RCc && !bk[j]->RPc && !bk[j]->RCDWRc)
          k--;
          /* [한국어] 요청도 없고 모든 타이밍 카운터도 0인 완전 유휴 뱅크는 k에서 제외.
           * k == 0이면 모든 뱅크가 완전 유휴 → n_activity를 증가하지 않음. */
        bk[j]->n_idle++;  /* [한국어] 이 뱅크의 유휴 사이클 카운터 증가 */
      }
    }
  } else {
    // single bus interface
    // issue only one row/column command
    /* [한국어] Single Bus Interface: 한 사이클에 Row 또는 Column 커맨드 중 하나만 발행 가능.
     * Column 커맨드(RD/WR)를 Row 커맨드(ACT/PRE)보다 우선시하여 처리량 극대화. */
    for (unsigned i = 0; i < m_config->nbk; i++) {
      unsigned j = (i + prio) % m_config->nbk;  /* [한국어] prio 기준 라운드로빈 뱅크 인덱스 */
      if (!issued_col_cmd) issued_col_cmd = issue_col_command(j);
      /* [한국어] 아직 Column 커맨드를 발행하지 않은 경우에만 시도
       * (이미 발행했으면 이 사이클에 다른 뱅크의 Column 커맨드도 발행 가능 — single bus이므로 한 번만) */

      if (!issued_col_cmd && !issued_row_cmd)
        issued_row_cmd = issue_row_command(j);
      /* [한국어] Column 커맨드와 Row 커맨드 모두 아직 발행 안 한 경우에만 Row 커맨드 시도.
       * Column을 우선하는 이유: Row Hit 요청의 즉각적인 처리가 처리량에 더 유리하기 때문. */

      if (!bk[j]->mrq) {  /* [한국어] 이 뱅크에 배정된 요청이 없는 경우 유휴 처리 */
        if (!CCDc && !RRDc && !RTWc && !WTRc && !bk[j]->RCDc && !bk[j]->RASc &&
            !bk[j]->RCc && !bk[j]->RPc && !bk[j]->RCDWRc)
          k--;  /* [한국어] 완전 유휴 뱅크: k에서 제외하여 n_activity 집계에 반영 */
        bk[j]->n_idle++;  /* [한국어] 유휴 사이클 카운터 증가 */
      }
    }
  }

  issued = issued_row_cmd || issued_col_cmd;  /* [한국어] 이 사이클에 어떤 커맨드라도 발행되었으면 true */
  if (!issued) {
    /* [한국어] 이 사이클에 아무 커맨드도 발행되지 않음 → NOP (No Operation) 사이클 */
    n_nop++;          /* [한국어] 전체 NOP 사이클 수 누적 */
    n_nop_partial++;  /* [한국어] 인터벌 NOP 사이클 수 누적 */
#ifdef DRAM_VIEWCMD
    printf("\tNOP                        ");  /* [한국어] DRAM_VIEWCMD 모드에서 NOP 표시 */
#endif
  }
  if (k) {
    /* [한국어] 완전 유휴가 아닌 뱅크가 하나 이상 있는 경우 (타이밍 카운터가 남아있거나 요청이 있음)
     * → "활성" 사이클로 간주하여 n_activity 증가. n_activity는 효율 계산 분모가 된다. */
    n_activity++;          /* [한국어] 전체 활성 사이클 수 누적 */
    n_activity_partial++;  /* [한국어] 인터벌 활성 사이클 수 누적 */
  }
  n_cmd++;          /* [한국어] 전체 커맨드 사이클 수 누적 (cycle()이 호출될 때마다 1 증가) */
  n_cmd_partial++;  /* [한국어] 인터벌 커맨드 사이클 수 누적 */
  if (issued) {
    issued_total++;  /* [한국어] Row 또는 Col 커맨드가 발행된 사이클 총 수 */
    if (issued_col_cmd && issued_row_cmd) issued_two++;
    /* [한국어] dual_bus_interface에서 Row+Col이 동시에 발행된 사이클 수 (효율 지표) */
  }
  if (issued_col_cmd) issued_total_col++;  /* [한국어] Col 커맨드(RD/WR) 발행 총 사이클 수 */
  if (issued_row_cmd) issued_total_row++;  /* [한국어] Row 커맨드(ACT/PRE) 발행 총 사이클 수 */

  // Collect some statistics
  // check the limitation, see where BW is wasted?
  /////////////////////////////////////////////////////////
  /* [단계 5] BW 낭비 원인 분석: 커맨드 발행 결과에 따라 util_bw, wasted_bw_col, wasted_bw_row, idle_bw 분류 */

  /* [한국어] 커맨드 발행 후 시점에서 mrq 보유 뱅크 수 재집계 (단계 3의 사전 집계와 구별) */
  unsigned int memory_pending_found = 0;  /* [한국어] 커맨드 발행 후 여전히 mrq가 있는 뱅크 수 */
  for (unsigned i = 0; i < m_config->nbk; i++) {
    if (bk[i]->mrq) memory_pending_found++;  /* [한국어] mrq가 NULL이 아닌 뱅크 수 집계 */
  }
  if (memory_pending_found > 0) banks_acess_total_after++;
  /* [한국어] 커맨드 발행 후에도 mrq 보유 뱅크가 있는 사이클 수 누적 (after 접미사) */

  /* [한국어] 커맨드 발행 후 Row Buffer Hit 상태인 뱅크가 있는지 확인 */
  bool memory_pending_rw_found = false;  /* [한국어] Row Hit 상태인 뱅크가 하나라도 있으면 true */
  for (unsigned j = 0; j < m_config->nbk; j++) {
    if (bk[j]->mrq &&
        (((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
          (bk[j]->state == BANK_ACTIVE)) ||
         ((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == WRITE) &&
          (bk[j]->state == BANK_ACTIVE))))
      memory_pending_rw_found = true;  /* [한국어] Row Hit READ 또는 WRITE 뱅크 발견 */
  }

  /* [한국어] BW 분류:
   *   util_bw: Column 커맨드 발행 또는 CCDc > 0 (버스가 데이터 전송 중)
   *   wasted_bw_col: Row Hit 요청이 있음에도 타이밍 제약으로 Col 커맨드 차단
   *   wasted_bw_row: 요청은 있지만 Row Buffer가 준비되지 않은 상태 (ACT/PRE 진행 중)
   *   idle_bw: 서비스할 요청이 전혀 없는 완전 유휴 */
  if (issued_col_cmd || CCDc)
    util_bw++;  /* [한국어] Col 커맨드가 발행되었거나 CCD 파이프라인이 진행 중이면 유효 BW */
  else if (memory_pending_rw_found) {
    /* [한국어] Col 커맨드가 발행되지 않았지만 Row Hit 요청이 있음 → 타이밍 제약이 원인인 낭비 */
    wasted_bw_col++;  /* [한국어] Column 커맨드 차단으로 인한 BW 낭비 사이클 증가 */
    for (unsigned j = 0; j < m_config->nbk; j++) {
      unsigned grp = get_bankgrp_number(j);  /* [한국어] 뱅크 j의 Bank Group 번호 */
      // read
      if (bk[j]->mrq &&
          (((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
            (bk[j]->state == BANK_ACTIVE)))) {
        /* [한국어] READ Row Hit 뱅크: 어떤 타이밍 제약이 RD 커맨드를 차단하고 있는지 집계 */
        if (bk[j]->RCDc) RCDc_limit++;             /* [한국어] tRCD 카운터가 남아있어 차단 */
        if (bkgrp[grp]->CCDLc) CCDLc_limit++;      /* [한국어] Bank Group CCD_L 카운터가 남아있어 차단 */
        if (WTRc) WTRc_limit++;                     /* [한국어] WTR(Write-to-Read 패널티) 카운터가 남아있어 차단 */
        if (CCDc) CCDc_limit++;                     /* [한국어] 채널 CCD 카운터가 남아있어 차단 */
        if (rwq->full()) rwq_limit++;               /* [한국어] rwq 파이프라인 큐가 포화되어 차단 */
        if (bkgrp[grp]->CCDLc && !WTRc) CCDLc_limit_alone++;   /* [한국어] CCDLc만 단독 병목 */
        if (!bkgrp[grp]->CCDLc && WTRc) WTRc_limit_alone++;    /* [한국어] WTRc만 단독 병목 */
      }
      // write
      else if (bk[j]->mrq &&
               ((bk[j]->curr_row == bk[j]->mrq->row) &&
                (bk[j]->mrq->rw == WRITE) && (bk[j]->state == BANK_ACTIVE))) {
        /* [한국어] WRITE Row Hit 뱅크: 어떤 타이밍 제약이 WR 커맨드를 차단하고 있는지 집계 */
        if (bk[j]->RCDWRc) RCDWRc_limit++;         /* [한국어] tRCDWR 카운터가 남아있어 차단 */
        if (bkgrp[grp]->CCDLc) CCDLc_limit++;      /* [한국어] Bank Group CCD_L 카운터가 남아있어 차단 */
        if (RTWc) RTWc_limit++;                     /* [한국어] RTW(Read-to-Write 패널티) 카운터가 남아있어 차단 */
        if (CCDc) CCDc_limit++;                     /* [한국어] 채널 CCD 카운터가 남아있어 차단 */
        if (rwq->full()) rwq_limit++;               /* [한국어] rwq 파이프라인 큐가 포화되어 차단 */
        if (bkgrp[grp]->CCDLc && !RTWc) CCDLc_limit_alone++;   /* [한국어] CCDLc만 단독 병목 */
        if (!bkgrp[grp]->CCDLc && RTWc) RTWc_limit_alone++;    /* [한국어] RTWc만 단독 병목 */
      }
    }
  } else if (memory_pending_found)
    wasted_bw_row++;  /* [한국어] mrq는 있지만 Row Buffer가 아직 준비되지 않은 경우(ACT/PRE 진행 중) */
  else if (!memory_pending_found)
    idle_bw++;  /* [한국어] 서비스할 요청이 전혀 없는 완전 유휴 사이클 */
  else
    assert(1);  /* [한국어] 도달 불가 경로 — 논리 오류 검증용 */

  /////////////////////////////////////////////////////////

  // decrements counters once for each time dram_issueCMD is called
  /* [단계 6] 모든 타이밍 카운터를 1 감소 (0 이하로 내려가지 않도록 DEC2ZERO 매크로 사용).
   * 이 단계는 사이클의 마지막에 실행되어 다음 사이클의 커맨드 발행 가능 여부를 결정한다. */
  DEC2ZERO(RRDc);   /* [한국어] Rank-to-Rank Delay 카운터 감소 */
  DEC2ZERO(CCDc);   /* [한국어] Column-to-Column Delay 카운터 감소 */
  DEC2ZERO(RTWc);   /* [한국어] Read-to-Write 패널티 카운터 감소 */
  DEC2ZERO(WTRc);   /* [한국어] Write-to-Read 패널티 카운터 감소 */
  for (unsigned j = 0; j < m_config->nbk; j++) {
    /* [한국어] 각 뱅크별 타이밍 카운터를 하나씩 감소 */
    DEC2ZERO(bk[j]->RCDc);    /* [한국어] tRCD(ACT→RD) 카운터 감소 */
    DEC2ZERO(bk[j]->RASc);    /* [한국어] tRAS(ACT→PRE) 최소 대기 카운터 감소 */
    DEC2ZERO(bk[j]->RCc);     /* [한국어] tRC(ACT→ACT) 사이클 카운터 감소 */
    DEC2ZERO(bk[j]->RPc);     /* [한국어] tRP(PRE→ACT) 대기 카운터 감소 */
    DEC2ZERO(bk[j]->RCDWRc);  /* [한국어] tRCDWR(ACT→WR) 카운터 감소 */
    DEC2ZERO(bk[j]->WTPc);    /* [한국어] tWTP(WR→PRE) 카운터 감소 */
    DEC2ZERO(bk[j]->RTPc);    /* [한국어] tRTP(RD→PRE) 카운터 감소 */
  }
  for (unsigned j = 0; j < m_config->nbkgrp; j++) {
    /* [한국어] 각 Bank Group의 타이밍 카운터를 하나씩 감소 */
    DEC2ZERO(bkgrp[j]->CCDLc);  /* [한국어] Bank Group CCD_L(동일 그룹 내 Col→Col) 카운터 감소 */
    DEC2ZERO(bkgrp[j]->RTPLc);  /* [한국어] Bank Group RTP_L(동일 그룹 내 RD→PRE) 카운터 감소 */
  }

/* [단계 7] 디버그 시각화 (DRAM_VISUALIZE 컴파일 옵션 활성화 시만 실행) */
#ifdef DRAM_VISUALIZE
  visualize();  /* [한국어] 매 사이클 현재 DRAM 상태(뱅크별 타이밍 카운터, mrq 상태) 출력 */
#endif
}

/*
 * [한국어]
 * dram_t::issue_col_command - 뱅크 j에 Column 커맨드(RD 또는 WR) 발행 시도
 *
 * @j: Column 커맨드 발행을 시도할 뱅크 번호 (0 ~ nbk-1)
 * @return: 이 사이클에 RD 또는 WR 커맨드를 발행했으면 true, 타이밍 제약으로 발행 못하면 false
 *
 * Column 커맨드(CAS: Column Address Strobe)를 발행하기 위한 조건:
 *   READ 발행 조건 (모두 충족해야 함):
 *     - bk[j]->mrq 존재 (서비스할 요청이 있음)
 *     - CCDc == 0 (채널 전체 CCD 타이밍 해제)
 *     - bk[j]->RCDc == 0 (ACT→RD 딜레이 완료)
 *     - bkgrp[grp]->CCDLc == 0 (Bank Group CCD_L 타이밍 해제)
 *     - curr_row == mrq->row (Row Buffer Hit — 요청하는 Row가 이미 열려있음)
 *     - mrq->rw == READ
 *     - WTRc == 0 (이전 WRITE 후 WTR 패널티 완료)
 *     - state == BANK_ACTIVE
 *     - !rwq->full() (RD/WR 파이프라인 큐에 여유 있음)
 *   WRITE 발행 조건: 위와 유사하되 RCDWRc, RTWc 사용
 *
 * RD 발행 후 설정되는 카운터: CCDc=tCCD, CCDLc=tCCDL, RTWc=tRTW, RTPc, RTPLc
 * WR 발행 후 설정되는 카운터: CCDc=tCCD, CCDLc=tCCDL, WTRc=tWTR, WTPc=tWTP
 * txbytes가 nbytes에 도달하면 전송 완료 → bk[j]->mrq = NULL (다음 요청 수용 가능)
 *
 * 방향 전환(READ↔WRITE) 시 rwq->set_min_length()로 파이프라인 깊이를 CL 또는 WL로 조정.
 *
 * 실행 컨텍스트: dram_t::cycle() 내부, 매 DRAM 사이클
 * 호출 체인: dram_t::cycle() → [issue_col_command(j)] → rwq->push(mrq)
 */
bool dram_t::issue_col_command(int j) {
  bool issued = false;  /* [한국어] 이 호출에서 커맨드 발행 여부 초기값 false */
  unsigned grp = get_bankgrp_number(j);  /* [한국어] 뱅크 j가 속하는 Bank Group 번호 계산 */
  if (bk[j]->mrq) {  // if currently servicing a memory request
    /* [한국어] 이 뱅크에 서비스 대기 중인 요청이 있는 경우에만 커맨드 발행을 시도한다 */
    bk[j]->mrq->data->set_status(
        IN_PARTITION_DRAM, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    /* [한국어] mem_fetch 상태를 IN_PARTITION_DRAM으로 갱신하여 현재 사이클 번호와 함께 기록.
     * 이 상태 갱신은 메모리 레이턴시 통계 추적에 사용된다. */

    // correct row activated for a READ
    /* [한국어] READ 커맨드 발행 시도: 모든 타이밍 제약이 해제되고 Row Buffer Hit이어야 한다 */
    if (!issued && !CCDc && !bk[j]->RCDc && !(bkgrp[grp]->CCDLc) &&
        (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
        (WTRc == 0) && (bk[j]->state == BANK_ACTIVE) && !rwq->full()) {
      /* [한국어] 조건 설명:
       *   !issued: 이 사이클에 아직 다른 커맨드가 발행되지 않음
       *   !CCDc: 채널 전체 CCD(Column-to-Column Delay) 타이밍 해제
       *   !bk[j]->RCDc: 이 뱅크의 ACT→RD 딜레이(tRCD) 완료
       *   !bkgrp[grp]->CCDLc: 동일 Bank Group 내 CCD_L 타이밍 해제
       *   curr_row == mrq->row: Row Buffer Hit (현재 열린 Row와 요청 Row 일치)
       *   mrq->rw == READ: 읽기 요청
       *   WTRc == 0: 이전 WRITE 후 WTR(Write-to-Read) 패널티 타이밍 완료
       *   BANK_ACTIVE: 뱅크가 활성(Row가 열려있는) 상태
       *   !rwq->full(): CAS 파이프라인 큐에 여유 공간 있음 */
      if (rw == WRITE) {
        /* [한국어] 이전 커맨드가 WRITE였고 이제 READ로 방향이 전환되는 경우.
         * rwq 파이프라인의 최소 길이를 WL(Write Latency)에서 CL(Read CAS Latency)로 재조정한다. */
        rw = READ;  /* [한국어] 채널 방향을 READ로 변경 */
        rwq->set_min_length(m_config->CL);  /* [한국어] rwq 파이프라인 깊이를 CL 사이클로 설정 */
      }
      rwq->push(bk[j]->mrq);  /* [한국어] 이 요청을 RD/WR 파이프라인 큐(rwq)에 삽입. CL 사이클 후 pop됨. */
      bk[j]->mrq->txbytes += m_config->dram_atom_size;
      /* [한국어] 이 RD 커맨드로 dram_atom_size 바이트를 전송 시작으로 기록 */

      /* [한국어] RD 커맨드 발행 후 관련 타이밍 카운터 재설정 */
      CCDc = m_config->tCCD;                    /* [한국어] 다음 Col 커맨드까지 CCD 사이클 대기 */
      bkgrp[grp]->CCDLc = m_config->tCCDL;     /* [한국어] 동일 Bank Group 내 다음 Col까지 CCD_L 사이클 대기 */
      RTWc = m_config->tRTW;                    /* [한국어] RD→WR 방향 전환 패널티 타이머 설정 */
      bk[j]->RTPc = m_config->BL / m_config->data_command_freq_ratio;
      /* [한국어] tRTP(RD→PRE): BL/data_command_freq_ratio 사이클 동안 PRE 금지 */
      bkgrp[grp]->RTPLc = m_config->tRTPL;     /* [한국어] Bank Group tRTP_L: 동일 그룹 내 RD→PRE 최소 간격 */
      issued = true;  /* [한국어] 이 사이클에 RD 커맨드 발행 완료 표시 */

      /* [한국어] READ 커맨드 카운터 증가: 접근 유형에 따라 n_rd_L2_A 또는 n_rd */
      if (bk[j]->mrq->data->get_access_type() == L2_WR_ALLOC_R)
        n_rd_L2_A++;  /* [한국어] L2 Write-Allocate를 위한 READ (쓰기 할당 시 먼저 읽어오는 요청) */
      else
        n_rd++;  /* [한국어] 일반 READ 커맨드 횟수 증가 */

      bwutil += m_config->BL / m_config->data_command_freq_ratio;
      /* [한국어] BW utilization: 이 RD 커맨드가 BL/data_command_freq_ratio 사이클 동안 버스를 사용 */
      bwutil_partial += m_config->BL / m_config->data_command_freq_ratio;
      /* [한국어] 인터벌 BW utilization 카운터도 동일하게 증가 */
      bk[j]->n_access++;  /* [한국어] 이 뱅크의 총 컬럼 커맨드 횟수 증가 */

#ifdef DRAM_VERIFY
      PRINT_CYCLE = 1;  /* [한국어] 이 사이클 로그 출력 활성화 */
      printf("\tRD  Bk:%d Row:%03x Col:%03x \n", j, bk[j]->curr_row,
             bk[j]->mrq->col + bk[j]->mrq->txbytes - m_config->dram_atom_size);
      /* [한국어] 발행된 RD 커맨드의 뱅크/로우/컬럼 좌표 출력 (txbytes 기반 컬럼 오프셋 계산) */
#endif
      // transfer done
      if (!(bk[j]->mrq->txbytes < bk[j]->mrq->nbytes)) {
        /* [한국어] txbytes가 nbytes 이상: 이 요청의 모든 컬럼 커맨드 발행 완료.
         * 뱅크를 NULL로 설정하여 다음 요청을 받을 수 있도록 해제한다.
         * 실제 데이터는 rwq에 남아 CL 사이클 후 dqbytes가 nbytes에 도달하면 returnq로 이동한다. */
        bk[j]->mrq = NULL;  /* [한국어] 뱅크 배정 해제 → 다음 요청 수용 가능 */
      }
    } else
      // correct row activated for a WRITE
      /* [한국어] WRITE 커맨드 발행 시도: READ와 유사하지만 RCDWRc, RTWc(대신 WTRc) 조건 사용 */
      if (!issued && !CCDc && !bk[j]->RCDWRc && !(bkgrp[grp]->CCDLc) &&
          (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == WRITE) &&
          (RTWc == 0) && (bk[j]->state == BANK_ACTIVE) && !rwq->full()) {
        /* [한국어] WRITE 발행 조건:
         *   !bk[j]->RCDWRc: ACT→WR 딜레이(tRCDWR) 완료 (READ의 tRCD와 별도 값)
         *   RTWc == 0: 이전 READ 후 RTW(Read-to-Write) 패널티 타이밍 완료 */
        if (rw == READ) {
          /* [한국어] 이전 커맨드가 READ였고 WRITE로 방향 전환:
           * rwq 파이프라인 깊이를 CL에서 WL(Write Latency)로 재조정한다. */
          rw = WRITE;  /* [한국어] 채널 방향을 WRITE로 변경 */
          rwq->set_min_length(m_config->WL);  /* [한국어] rwq 파이프라인 깊이를 WL 사이클로 설정 */
        }
        rwq->push(bk[j]->mrq);  /* [한국어] 이 WRITE 요청을 rwq 파이프라인 큐에 삽입. WL 사이클 후 pop됨. */

        bk[j]->mrq->txbytes += m_config->dram_atom_size;
        /* [한국어] 이 WR 커맨드로 dram_atom_size 바이트 전송 기록 */

        /* [한국어] WR 커맨드 발행 후 관련 타이밍 카운터 재설정 */
        CCDc = m_config->tCCD;                 /* [한국어] 다음 Col 커맨드까지 CCD 사이클 대기 */
        bkgrp[grp]->CCDLc = m_config->tCCDL;  /* [한국어] 동일 Bank Group 내 CCD_L 대기 */
        WTRc = m_config->tWTR;                 /* [한국어] WR→RD 방향 전환 패널티 타이머 설정 */
        bk[j]->WTPc = m_config->tWTP;          /* [한국어] tWTP(WR→PRE): WR 완료 후 PRE 금지 기간 */
        issued = true;  /* [한국어] 이 사이클에 WR 커맨드 발행 완료 표시 */

        /* [한국어] WRITE 커맨드 카운터 증가: 접근 유형에 따라 n_wr_WB 또는 n_wr */
        if (bk[j]->mrq->data->get_access_type() == L2_WRBK_ACC)
          n_wr_WB++;  /* [한국어] L2 캐시 Dirty Eviction Writeback으로 인한 WRITE */
        else
          n_wr++;  /* [한국어] 일반 WRITE 커맨드 횟수 증가 */

        bwutil += m_config->BL / m_config->data_command_freq_ratio;
        /* [한국어] BW utilization: WR 커맨드가 사용하는 버스 사이클 수 증가 */
        bwutil_partial += m_config->BL / m_config->data_command_freq_ratio;
        /* [한국어] 인터벌 BW utilization 카운터도 동일하게 증가 */
#ifdef DRAM_VERIFY
        PRINT_CYCLE = 1;  /* [한국어] 이 사이클 로그 출력 활성화 */
        printf(
            "\tWR  Bk:%d Row:%03x Col:%03x \n", j, bk[j]->curr_row,
            bk[j]->mrq->col + bk[j]->mrq->txbytes - m_config->dram_atom_size);
        /* [한국어] 발행된 WR 커맨드의 뱅크/로우/컬럼 좌표 출력 */
#endif
        // transfer done
        if (!(bk[j]->mrq->txbytes < bk[j]->mrq->nbytes)) {
          /* [한국어] txbytes >= nbytes: 이 WRITE 요청의 모든 컬럼 커맨드 발행 완료 → 뱅크 해제 */
          bk[j]->mrq = NULL;  /* [한국어] 뱅크 배정 해제 */
        }
      }
  }

  return issued;  /* [한국어] RD 또는 WR 커맨드가 발행되었으면 true, 아니면 false 반환 */
}

/*
 * [한국어]
 * dram_t::issue_row_command - 뱅크 j에 Row 커맨드(ACT 또는 PRE) 발행 시도
 *
 * @j: Row 커맨드 발행을 시도할 뱅크 번호 (0 ~ nbk-1)
 * @return: 이 사이클에 ACT 또는 PRE 커맨드를 발행했으면 true, 발행 못하면 false
 *
 * 뱅크 상태에 따라 두 가지 커맨드 중 하나를 선택하여 발행한다:
 *
 * ACT(Activate) 발행 조건 (BANK_IDLE 상태에서 새 Row를 열어야 할 때):
 *   - bk[j]->mrq 존재 (서비스할 요청이 있음)
 *   - !RRDc (채널 전체 RRD: Rank-to-Rank Delay 해제)
 *   - state == BANK_IDLE (현재 열린 Row 없음)
 *   - !RPc (이전 PRE 후 tRP 대기 완료)
 *   - !RCc (이전 ACT 후 tRC 대기 완료)
 *   ACT 발행 효과: curr_row = mrq->row, state = BANK_ACTIVE
 *   ACT 발행 후 설정: RRDc=tRRD, RCDc=tRCD, RCDWRc=tRCDWR, RASc=tRAS, RCc=tRC
 *
 * PRE(Precharge) 발행 조건 (BANK_ACTIVE 상태에서 다른 Row로 전환해야 할 때):
 *   - bk[j]->mrq 존재
 *   - curr_row != mrq->row (Row Miss: 다른 Row가 열려있어 먼저 닫아야 함)
 *   - state == BANK_ACTIVE
 *   - !RASc (ACT 이후 최소 유지 기간 tRAS 완료)
 *   - !WTPc (이전 WR 후 tWTP 완료)
 *   - !RTPc (이전 RD 후 tRTP 완료)
 *   - !bkgrp[grp]->RTPLc (Bank Group RTP_L 완료)
 *   PRE 발행 효과: state = BANK_IDLE, RPc = tRP 설정
 *
 * ACT/PRE 발행 후 공통: prio = (j+1)%nbk (다음 사이클 라운드로빈 우선순위 전진)
 *
 * 실행 컨텍스트: dram_t::cycle() 내부, 매 DRAM 사이클
 * 호출 체인: dram_t::cycle() → [issue_row_command(j)] → 뱅크 state 전환
 */
bool dram_t::issue_row_command(int j) {
  bool issued = false;  /* [한국어] 이 호출에서 커맨드 발행 여부 초기값 false */
  unsigned grp = get_bankgrp_number(j);  /* [한국어] 뱅크 j의 Bank Group 번호 계산 (PRE 조건의 RTPLc 확인에 사용) */
  if (bk[j]->mrq) {  // if currently servicing a memory request
    /* [한국어] 이 뱅크에 서비스할 요청이 있는 경우에만 Row 커맨드를 고려한다 */
    bk[j]->mrq->data->set_status(
        IN_PARTITION_DRAM, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    /* [한국어] mem_fetch 상태를 IN_PARTITION_DRAM으로 갱신 (레이턴시 추적용 타임스탬프 기록) */

    //     bank is idle
    // else
    /* [한국어] ACT(Activate) 발행 시도: 뱅크가 IDLE 상태이고 타이밍 제약이 없을 때 */
    if (!issued && !RRDc && (bk[j]->state == BANK_IDLE) && !bk[j]->RPc &&
        !bk[j]->RCc) {  //
      /* [한국어] ACT 발행 조건:
       *   !RRDc: Rank-to-Rank Delay 완료 (이전 다른 뱅크 ACT 이후 최소 간격)
       *   BANK_IDLE: 현재 이 뱅크에는 열려있는 Row가 없음
       *   !RPc: 이전 PRE 후 tRP(Row Precharge time) 완료
       *   !RCc: 이전 ACT 후 tRC(Row Cycle time) 완료 */
#ifdef DRAM_VERIFY
      PRINT_CYCLE = 1;  /* [한국어] 이 사이클의 로그 출력 활성화 */
      printf("\tACT BK:%d NewRow:%03x From:%03x \n", j, bk[j]->mrq->row,
             bk[j]->curr_row);
      /* [한국어] ACT 커맨드 발행 로그: 뱅크 번호, 새로 열 Row, 기존 Row(의미없는 값일 수 있음) */
#endif
      // activate the row with current memory request
      /* [한국어] ACT 커맨드 발행 효과 적용 */
      bk[j]->curr_row = bk[j]->mrq->row;  /* [한국어] 이 요청의 Row를 Row Buffer에 열림 상태로 기록 */
      bk[j]->state = BANK_ACTIVE;          /* [한국어] 뱅크 상태를 ACTIVE로 전환 */

      /* [한국어] ACT 발행 후 타이밍 카운터 설정 */
      RRDc = m_config->tRRD;            /* [한국어] 다른 뱅크의 다음 ACT까지 RRD 사이클 대기 */
      bk[j]->RCDc = m_config->tRCD;    /* [한국어] ACT→RD 딜레이: tRCD 사이클 후 RD 커맨드 가능 */
      bk[j]->RCDWRc = m_config->tRCDWR;/* [한국어] ACT→WR 딜레이: tRCDWR 사이클 후 WR 커맨드 가능 */
      bk[j]->RASc = m_config->tRAS;    /* [한국어] 이 Row를 최소 tRAS 사이클 동안 ACTIVE 유지 필요 */
      bk[j]->RCc = m_config->tRC;      /* [한국어] 다음 ACT까지 tRC 사이클 대기 (tRC = tRAS + tRP) */
      prio = (j + 1) % m_config->nbk;  /* [한국어] 라운드로빈 우선순위 전진: 다음 사이클에서 j+1번 뱅크부터 시작 */
      issued = true;  /* [한국어] ACT 커맨드 발행 완료 표시 */
      n_act_partial++; /* [한국어] 인터벌 ACT 카운터 증가 */
      n_act++;         /* [한국어] 전체 ACT 카운터 증가 */
    }

    else
      // different row activated
      /* [한국어] PRE(Precharge) 발행 시도: 다른 Row가 열려있어 Close가 필요한 경우 */
      if ((!issued) && (bk[j]->curr_row != bk[j]->mrq->row) &&
          (bk[j]->state == BANK_ACTIVE) &&
          (!bk[j]->RASc && !bk[j]->WTPc && !bk[j]->RTPc &&
           !bkgrp[grp]->RTPLc)) {
        /* [한국어] PRE 발행 조건:
         *   curr_row != mrq->row: Row Miss — 현재 열린 Row와 요청 Row가 다름
         *   BANK_ACTIVE: 현재 Row가 열려있는 상태 (닫아야 함)
         *   !RASc: ACT 이후 최소 유지 기간(tRAS) 완료 — 너무 빨리 닫으면 안 됨
         *   !WTPc: 이전 WR 커맨드 이후 tWTP 완료 — WRITE 데이터가 셀에 기록 완료
         *   !RTPc: 이전 RD 커맨드 이후 tRTP 완료 — READ 버스트 완료
         *   !RTPLc: Bank Group RTP_L 완료 */
        // make the bank idle again
        /* [한국어] PRE 커맨드 발행 효과 적용 */
        bk[j]->state = BANK_IDLE;     /* [한국어] 뱅크를 IDLE 상태로 전환 (Row Buffer 닫힘) */
        bk[j]->RPc = m_config->tRP;   /* [한국어] PRE→ACT 대기: tRP 사이클 후 다음 ACT 가능 */
        prio = (j + 1) % m_config->nbk;  /* [한국어] 라운드로빈 우선순위 전진 */
        issued = true;  /* [한국어] PRE 커맨드 발행 완료 표시 */
        n_pre++;         /* [한국어] 전체 PRE 카운터 증가 */
        n_pre_partial++; /* [한국어] 인터벌 PRE 카운터 증가 */
#ifdef DRAM_VERIFY
        PRINT_CYCLE = 1;  /* [한국어] 이 사이클 로그 출력 활성화 */
        printf("\tPRE BK:%d Row:%03x \n", j, bk[j]->curr_row);
        /* [한국어] PRE 커맨드 발행 로그: 뱅크 번호와 닫히는 Row 번호 출력 */
#endif
      }
  }
  return issued;  /* [한국어] ACT 또는 PRE 커맨드가 발행되었으면 true, 아니면 false 반환 */
}

/*
 * [한국어]
 * dram_t::return_queue_pop - returnq에서 완료된 READ 응답 패킷을 꺼내 반환
 *
 * @return: DRAM CL 파이프라인 완료 후 returnq에 대기 중이던 mem_fetch 포인터.
 *          returnq가 비어있으면 NULL 반환.
 *
 * dram_t::cycle()에서 rwq 파이프라인을 완전히 통과한 READ 요청이 returnq에 투입된다.
 * memory_partition_unit::dram_cycle()이 ICNT 클럭 기준으로 이 함수를 호출하여
 * 완료된 응답을 꺼내 ICNT를 통해 SM으로 회신한다.
 * WRITE/Writeback 요청은 returnq가 아닌 set_done()으로 처리되므로 이 함수를 통하지 않는다.
 *
 * // if mrq is being serviced by dram, gets popped after CL latency fulfilled
 * 호출 체인: memory_partition_unit::dram_cycle() → [dram_t::return_queue_pop()]
 *           → mem_fetch가 ICNT를 통해 SM으로 반환됨
 */
class mem_fetch *dram_t::return_queue_pop() {
  return returnq->pop();  /* [한국어] returnq의 맨 앞 항목을 제거하며 반환 (비어있으면 NULL) */
}

/*
 * [한국어]
 * dram_t::return_queue_top - returnq의 맨 앞 항목 조회 (pop하지 않음)
 *
 * @return: returnq 맨 앞의 mem_fetch 포인터. 비어있으면 NULL.
 *
 * memory_partition_unit이 returnq에 항목이 있는지 확인하거나
 * pop하기 전에 peek하는 용도로 호출한다. 큐 상태를 변경하지 않는다.
 *
 * 호출 체인: memory_partition_unit → [dram_t::return_queue_top()]
 */
class mem_fetch *dram_t::return_queue_top() {
  return returnq->top();  /* [한국어] returnq의 맨 앞 항목을 제거하지 않고 반환 (비어있으면 NULL) */
}

/*
 * [한국어]
 * dram_t::print - DRAM 채널 전체 통계 상세 출력 (시뮬레이션 종료 시)
 *
 * @simFile: 출력 대상 FILE 포인터 (보통 시뮬레이터 통계 출력 파일)
 *
 * 시뮬레이션 종료 후 이 DRAM 채널의 전체 통계를 상세하게 출력한다:
 *   1. 타이밍 파라미터 요약 (nbk, busW, BL, CL, tRRD, tCCD 등)
 *   2. 커맨드 카운터 (n_cmd, n_nop, n_act, n_pre, n_ref, n_req, n_rd, n_wr)
 *   3. BW utilization 분석 (util_bw, wasted_bw_col, wasted_bw_row, idle_bw)
 *   4. 병목 카운터 (RCDc_limit, WTRc_limit, RTWc_limit, CCDLc_limit 등)
 *   5. Row Buffer Locality 통계
 *   6. Bank Level Parallelism 통계
 *   7. Bank Group Level Parallelism
 *   8. Dual Bus Interface 효율
 *   9. dram_util_bins/dram_eff_bins 히스토그램
 *
 * simFile과 stdout 두 곳에 각각 출력하는 부분이 섞여있다 (fprintf/printf 혼용).
 *
 * 실행 컨텍스트: 시뮬레이션 종료 후 통계 출력 단계 (단일 스레드)
 * 호출 체인: gpgpu_sim::print_stats() → memory_partition_unit::print_stat()
 *           → [dram_t::print()]
 */
void dram_t::print(FILE *simFile) const {
  unsigned i;  /* [한국어] 루프 인덱스 변수 */
  fprintf(simFile, "DRAM[%d]: %d bks, busW=%d BL=%d CL=%d, ", id, m_config->nbk,
          m_config->busW, m_config->BL, m_config->CL);
  /* [한국어] DRAM 채널 ID와 핵심 타이밍 파라미터 출력: 뱅크 수, 버스 폭, Burst Length, CAS Latency */
  fprintf(simFile, "tRRD=%d tCCD=%d, tRCD=%d tRAS=%d tRP=%d tRC=%d\n",
          m_config->tRRD, m_config->tCCD, m_config->tRCD, m_config->tRAS,
          m_config->tRP, m_config->tRC);
  /* [한국어] DRAM 타이밍 파라미터 출력: tRRD(Rank-to-Rank), tCCD(Column-to-Column),
   *         tRCD(RAS-to-CAS), tRAS(Row Active), tRP(Row Precharge), tRC(Row Cycle) */

  fprintf(
      simFile,
      "n_cmd=%llu n_nop=%llu n_act=%llu n_pre=%llu n_ref_event=%llu n_req=%llu "
      "n_rd=%llu n_rd_L2_A=%llu n_write=%llu n_wr_bk=%llu bw_util=%.4g\n",
      n_cmd, n_nop, n_act, n_pre, n_ref, n_req, n_rd, n_rd_L2_A, n_wr, n_wr_WB,
      (float)bwutil / n_cmd);
  /* [한국어] 커맨드 카운터 및 BW utilization(bwutil/n_cmd) 출력.
   *         bw_util은 전체 사이클 중 실제 데이터 전송에 사용된 비율을 나타냄. */

  fprintf(simFile, "n_activity=%llu dram_eff=%.4g\n", n_activity,
          (float)bwutil / n_activity);
  /* [한국어] 활성 사이클 수와 DRAM 효율(bwutil/n_activity) 출력.
   *         dram_eff는 DRAM이 활성 상태일 때 실제로 데이터를 전송한 비율. */

  for (i = 0; i < m_config->nbk; i++) {
    fprintf(simFile, "bk%d: %da %di ", i, bk[i]->n_access, bk[i]->n_idle);
    /* [한국어] 각 뱅크별 컬럼 접근 횟수(n_access)와 유휴 사이클 수(n_idle) 출력 */
  }
  fprintf(simFile, "\n");
  fprintf(simFile,
          "\n------------------------------------------------------------------"
          "------\n");
  /* [한국어] 구분선 출력 */

  /* [한국어] Row Buffer Locality 통계 출력 (stdout으로) */
  printf("\nRow_Buffer_Locality = %.6f", (float)hits_num / access_num);
  /* [한국어] 전체 요청 중 Row Buffer Hit 비율 (높을수록 메모리 효율 우수) */
  printf("\nRow_Buffer_Locality_read = %.6f", (float)hits_read_num / read_num);
  /* [한국어] READ 요청 중 Row Buffer Hit 비율 */
  printf("\nRow_Buffer_Locality_write = %.6f",
         (float)hits_write_num / write_num);
  /* [한국어] WRITE 요청 중 Row Buffer Hit 비율 */

  /* [한국어] Bank Level Parallelism(BLP) 통계 출력 */
  printf("\nBank_Level_Parallism = %.6f",
         (float)banks_1time / banks_acess_total);
  /* [한국어] 요청 있는 사이클에서 평균 활성 뱅크 수 (높을수록 병렬성 우수) */
  printf("\nBank_Level_Parallism_Col = %.6f",
         (float)banks_time_rw / banks_access_rw_total);
  /* [한국어] Row Hit 상태 사이클에서 평균 활성 뱅크 수 */
  printf("\nBank_Level_Parallism_Ready = %.6f",
         (float)banks_time_ready / banks_access_ready_total);
  /* [한국어] 즉시 발행 가능 사이클에서 평균 준비된 뱅크 수 */
  printf("\nwrite_to_read_ratio_blp_rw_average = %.6f",
         write_to_read_ratio_blp_rw_average / banks_access_rw_total);
  /* [한국어] Row Hit 사이클에서의 평균 WRITE/(WRITE+READ) 비율 */
  printf("\nGrpLevelPara = %.6f \n",
         (float)bkgrp_parallsim_rw / banks_access_rw_total);
  /* [한국어] Bank Group Level Parallelism: Row Hit 사이클에서 활성 Bank Group 수 */

  /* [한국어] BW Utilization 상세 분석 출력 */
  printf("\nBW Util details:\n");
  printf("bwutil = %.6f \n", (float)bwutil / n_cmd);     /* [한국어] 전체 BW 사용률 */
  printf("total_CMD = %llu \n", n_cmd);                   /* [한국어] 총 사이클 수 */
  printf("util_bw = %llu \n", util_bw);                   /* [한국어] 유효 BW 사이클 수 */
  printf("Wasted_Col = %llu \n", wasted_bw_col);          /* [한국어] 타이밍 제약으로 Col 차단 사이클 수 */
  printf("Wasted_Row = %llu \n", wasted_bw_row);          /* [한국어] Row 준비 대기 낭비 사이클 수 */
  printf("Idle = %llu \n", idle_bw);                      /* [한국어] 완전 유휴 사이클 수 */

  /* [한국어] BW 병목 원인별 카운터 출력 */
  printf("\nBW Util Bottlenecks: \n");
  printf("RCDc_limit = %llu \n", RCDc_limit);            /* [한국어] tRCD 병목 사이클 수 */
  printf("RCDWRc_limit = %llu \n", RCDWRc_limit);        /* [한국어] tRCDWR 병목 사이클 수 */
  printf("WTRc_limit = %llu \n", WTRc_limit);            /* [한국어] WTR 패널티 병목 사이클 수 */
  printf("RTWc_limit = %llu \n", RTWc_limit);            /* [한국어] RTW 패널티 병목 사이클 수 */
  printf("CCDLc_limit = %llu \n", CCDLc_limit);          /* [한국어] Bank Group CCD_L 병목 사이클 수 */
  printf("rwq = %llu \n", rwq_limit);                    /* [한국어] rwq 포화 병목 사이클 수 */
  printf("CCDLc_limit_alone = %llu \n", CCDLc_limit_alone);  /* [한국어] CCDLc 단독 병목 사이클 수 */
  printf("WTRc_limit_alone = %llu \n", WTRc_limit_alone);    /* [한국어] WTRc 단독 병목 사이클 수 */
  printf("RTWc_limit_alone = %llu \n", RTWc_limit_alone);    /* [한국어] RTWc 단독 병목 사이클 수 */

  /* [한국어] 커맨드 유형별 세부 카운터 출력 */
  printf("\nCommands details: \n");
  printf("total_CMD = %llu \n", n_cmd);                          /* [한국어] 총 사이클 수 */
  printf("n_nop = %llu \n", n_nop);                              /* [한국어] NOP 사이클 수 */
  printf("Read = %llu \n", n_rd);                                /* [한국어] 일반 READ 커맨드 횟수 */
  printf("Write = %llu \n", n_wr);                               /* [한국어] 일반 WRITE 커맨드 횟수 */
  printf("L2_Alloc = %llu \n", n_rd_L2_A);                      /* [한국어] L2 Write-Allocate READ 횟수 */
  printf("L2_WB = %llu \n", n_wr_WB);                           /* [한국어] L2 Writeback WRITE 횟수 */
  printf("n_act = %llu \n", n_act);                              /* [한국어] ACT 커맨드 횟수 */
  printf("n_pre = %llu \n", n_pre);                              /* [한국어] PRE 커맨드 횟수 */
  printf("n_ref = %llu \n", n_ref);                              /* [한국어] Refresh 커맨드 횟수 (미구현) */
  printf("n_req = %llu \n", n_req);                              /* [한국어] 총 요청 수 */
  printf("total_req = %llu \n", n_rd + n_wr + n_rd_L2_A + n_wr_WB);  /* [한국어] 모든 유형 요청 합계 */

  /* [한국어] Dual Bus Interface 효율 통계 출력 */
  printf("\nDual Bus Interface Util: \n");
  printf("issued_total_row = %llu \n", issued_total_row);         /* [한국어] Row 커맨드 발행 총 횟수 */
  printf("issued_total_col = %llu \n", issued_total_col);         /* [한국어] Col 커맨드 발행 총 횟수 */
  printf("Row_Bus_Util =  %.6f \n", (float)issued_total_row / n_cmd);   /* [한국어] Row 버스 사용률 */
  printf("CoL_Bus_Util = %.6f \n", (float)issued_total_col / n_cmd);    /* [한국어] Col 버스 사용률 */
  printf("Either_Row_CoL_Bus_Util = %.6f \n", (float)issued_total / n_cmd);  /* [한국어] 어느 버스든 발행된 사이클 비율 */
  printf("Issued_on_Two_Bus_Simul_Util = %.6f \n", (float)issued_two / n_cmd);  /* [한국어] 두 버스 동시 발행 비율 */
  printf("issued_two_Eff = %.6f \n", (float)issued_two / issued_total);          /* [한국어] 발행 사이클 중 동시 발행 비율 */
  printf("queue_avg = %.6f \n\n", (float)ave_mrqs / n_cmd);       /* [한국어] 평균 큐 길이 */

  /* [한국어] BW utilization 및 efficiency 히스토그램 출력 */
  fprintf(simFile, "\n");
  fprintf(simFile, "dram_util_bins:");
  for (i = 0; i < 10; i++) fprintf(simFile, " %d", dram_util_bins[i]);  /* [한국어] utilization 구간별 카운터 */
  fprintf(simFile, "\ndram_eff_bins:");
  for (i = 0; i < 10; i++) fprintf(simFile, " %d", dram_eff_bins[i]);   /* [한국어] efficiency 구간별 카운터 */
  fprintf(simFile, "\n");
  if (m_config->scheduler_type == DRAM_FRFCFS)
    fprintf(simFile, "mrqq: max=%d avg=%g\n", max_mrqs,
            (float)ave_mrqs / n_cmd);
    /* [한국어] FR-FCFS 모드: 큐 최대 길이와 평균 길이 출력 */
}

/*
 * [한국어]
 * dram_t::visualize - 현재 DRAM 채널 상태를 콘솔에 출력 (DRAM_VISUALIZE 디버그 모드)
 *
 * DRAM_VISUALIZE 컴파일 매크로가 활성화된 경우 dram_t::cycle() 끝에서 호출된다.
 * 매 사이클 호출되어 현재 채널의 타이밍 카운터, 큐 길이, 각 뱅크의 상세 상태를 출력한다.
 * 출력 내용:
 *   - 채널 전체: RRDc, CCDc, mrqq 길이, rwq 길이
 *   - 각 뱅크: state(I/A), curr_row, RCDc, RASc, RPc, RCc, mrq 포인터
 *   - mrq가 있는 뱅크: nbytes(목표 전송량), txbytes(전송 진행량)
 *   - FR-FCFS 스케줄러 내부 상태 (m_frfcfs_scheduler->print())
 *
 * 실행 컨텍스트: dram_t::cycle() 끝 (DRAM_VISUALIZE ifdef 내부), 매 DRAM 사이클
 * 호출 체인: dram_t::cycle() (DRAM_VISUALIZE) → [dram_t::visualize()]
 */
void dram_t::visualize() const {
  printf("RRDc=%d CCDc=%d mrqq.Length=%d rwq.Length=%d\n", RRDc, CCDc,
         mrqq->get_length(), rwq->get_length());
  /* [한국어] 채널 전체 타이밍 카운터(RRDc, CCDc)와 큐 길이 출력 */
  for (unsigned i = 0; i < m_config->nbk; i++) {
    printf("BK%d: state=%c curr_row=%03x, %2d %2d %2d %2d %p ", i, bk[i]->state,
           bk[i]->curr_row, bk[i]->RCDc, bk[i]->RASc, bk[i]->RPc, bk[i]->RCc,
           bk[i]->mrq);
    /* [한국어] 뱅크 i의 상태(I=Idle/A=Active), curr_row, RCDc, RASc, RPc, RCc, mrq 포인터 출력 */
    if (bk[i]->mrq)
      printf("txf: %d %d", bk[i]->mrq->nbytes, bk[i]->mrq->txbytes);
      /* [한국어] 서비스 중인 요청이 있으면 목표 전송량(nbytes)과 현재 전송량(txbytes) 출력 */
    printf("\n");
  }
  if (m_frfcfs_scheduler) m_frfcfs_scheduler->print(stdout);
  /* [한국어] FR-FCFS 스케줄러가 활성화된 경우 스케줄러 내부 대기 큐 상태도 출력 */
}

/*
 * [한국어]
 * dram_t::print_stat - DRAM 채널 요약 통계 출력 (주기적 통계 덤프용)
 *
 * @simFile: 출력 대상 FILE 포인터
 *
 * print()의 간략 버전으로 핵심 카운터만 한 줄에 요약 출력한다.
 * 출력 항목: n_cmd, n_nop, n_act, n_pre, n_ref, n_req, n_rd, n_wr, bw_util,
 *            mrqq max/avg/max_mrqs_temp, dram_util_bins, dram_eff_bins
 * 출력 후 max_mrqs_temp를 0으로 리셋하여 다음 인터벌을 위해 초기화한다.
 *
 * 실행 컨텍스트: 시뮬레이션 중 주기적 통계 수집 또는 종료 시 (단일 스레드)
 * 호출 체인: gpu-sim.cc의 통계 덤프 루프 → [dram_t::print_stat()]
 */
void dram_t::print_stat(FILE *simFile) {
  fprintf(simFile,
          "DRAM (%u): n_cmd=%llu n_nop=%llu n_act=%llu n_pre=%llu n_ref=%llu "
          "n_req=%llu n_rd=%llu n_write=%llu bw_util=%.4g ",
          id, n_cmd, n_nop, n_act, n_pre, n_ref, n_req, n_rd, n_wr,
          (float)bwutil / n_cmd);
  /* [한국어] 이 DRAM 채널의 주요 카운터와 BW utilization을 한 줄에 출력 */
  fprintf(simFile, "mrqq: %d %.4g mrqsmax=%llu ", max_mrqs,
          (float)ave_mrqs / n_cmd, max_mrqs_temp);
  /* [한국어] 큐 최대 길이(max_mrqs), 평균 큐 길이(ave_mrqs/n_cmd), 임시 최대값(max_mrqs_temp) 출력 */
  fprintf(simFile, "\n");
  fprintf(simFile, "dram_util_bins:");
  for (unsigned i = 0; i < 10; i++) fprintf(simFile, " %d", dram_util_bins[i]);
  /* [한국어] BW utilization 구간별 히스토그램 출력 */
  fprintf(simFile, "\ndram_eff_bins:");
  for (unsigned i = 0; i < 10; i++) fprintf(simFile, " %d", dram_eff_bins[i]);
  /* [한국어] BW efficiency 구간별 히스토그램 출력 */
  fprintf(simFile, "\n");
  max_mrqs_temp = 0;  /* [한국어] 임시 최대 큐 길이 추적 변수를 0으로 리셋 (다음 인터벌 측정 준비) */
}

/*
 * [한국어]
 * dram_t::visualizer_print - AerialVision 시각화 도구용 gzip 통계 출력
 *
 * @visualizer_file: gzip 압축 출력 스트림 (AerialVision .gz 로그 파일)
 *
 * 매 통계 수집 인터벌마다 이 DRAM 채널의 인터벌 통계를 gzip 압축 파일에 기록한다.
 * AerialVision GPU 시뮬레이션 가시화 도구가 이 파일을 파싱하여 시간 경과에 따른
 * DRAM 성능 변화를 시각화한다.
 * 출력 항목:
 *   - 인터벌 커맨드 수, NOP, ACT, PRE, 요청 수, 평균 큐 길이
 *   - BW utilization % (100 * bwutil_partial / n_cmd_partial)
 *   - BW efficiency % (100 * bwutil_partial / n_activity_partial)
 *   - 뱅크별 접근 유형 분류 (GLOBAL/LOCAL/CONST/TEXTURE × R/W)
 * 출력 후 모든 partial 카운터를 0으로 리셋한다 (다음 인터벌 측정 시작).
 *
 * 실행 컨텍스트: gpgpu_sim::cycle() 내 시각화 인터벌 도달 시 (단일 스레드)
 * 호출 체인: gpu-sim.cc 시각화 루프 → memory_partition_unit → [dram_t::visualizer_print()]
 */
void dram_t::visualizer_print(gzFile visualizer_file) {
  // dram specific statistics
  /* [한국어] 인터벌 커맨드 및 요청 통계를 gzip 파일에 기록 */
  gzprintf(visualizer_file, "dramncmd: %u %u\n", id, n_cmd_partial);
  /* [한국어] "dramncmd: <채널id> <인터벌 커맨드 사이클 수>" */
  gzprintf(visualizer_file, "dramnop: %u %u\n", id, n_nop_partial);
  /* [한국어] "dramnop: <채널id> <인터벌 NOP 사이클 수>" */
  gzprintf(visualizer_file, "dramnact: %u %u\n", id, n_act_partial);
  /* [한국어] "dramnact: <채널id> <인터벌 ACT 커맨드 횟수>" */
  gzprintf(visualizer_file, "dramnpre: %u %u\n", id, n_pre_partial);
  /* [한국어] "dramnpre: <채널id> <인터벌 PRE 커맨드 횟수>" */
  gzprintf(visualizer_file, "dramnreq: %u %u\n", id, n_req_partial);
  /* [한국어] "dramnreq: <채널id> <인터벌 요청 수>" */
  gzprintf(visualizer_file, "dramavemrqs: %u %u\n", id,
           n_cmd_partial ? (ave_mrqs_partial / n_cmd_partial) : 0);
  /* [한국어] "dramavemrqs: <채널id> <인터벌 평균 큐 길이>"
   *         n_cmd_partial이 0이면 0 출력 (0-나누기 방지) */

  // utilization and efficiency
  /* [한국어] BW utilization(%) 및 efficiency(%) 출력 */
  gzprintf(visualizer_file, "dramutil: %u %u\n", id,
           n_cmd_partial ? 100 * bwutil_partial / n_cmd_partial : 0);
  /* [한국어] "dramutil: <채널id> <BW utilization %>: 전체 사이클 중 데이터 전송 비율" */
  gzprintf(visualizer_file, "drameff: %u %u\n", id,
           n_activity_partial ? 100 * bwutil_partial / n_activity_partial : 0);
  /* [한국어] "drameff: <채널id> <BW efficiency %>: 활성 사이클 중 데이터 전송 비율" */

  // reset for next interval
  /* [한국어] 다음 인터벌을 위해 모든 partial 카운터를 0으로 리셋 */
  bwutil_partial = 0;      /* [한국어] BW utilization 누적 카운터 리셋 */
  n_activity_partial = 0;  /* [한국어] 활성 사이클 카운터 리셋 */
  ave_mrqs_partial = 0;    /* [한국어] 큐 길이 누적 합 리셋 */
  n_cmd_partial = 0;       /* [한국어] 커맨드 사이클 카운터 리셋 */
  n_nop_partial = 0;       /* [한국어] NOP 사이클 카운터 리셋 */
  n_act_partial = 0;       /* [한국어] ACT 횟수 카운터 리셋 */
  n_pre_partial = 0;       /* [한국어] PRE 횟수 카운터 리셋 */
  n_req_partial = 0;       /* [한국어] 요청 수 카운터 리셋 */

  // dram access type classification
  /* [한국어] 뱅크별 메모리 접근 유형 분류 통계 출력.
   * GLOBAL/LOCAL/CONST/TEXTURE 메모리 접근을 읽기/쓰기로 구분하여 각 뱅크별로 기록한다.
   * m_stats->mem_access_type_stats[type][id][bk]는 3차원 배열: [접근유형][채널][뱅크] */
  for (unsigned j = 0; j < m_config->nbk; j++) {
    gzprintf(visualizer_file, "dramglobal_acc_r: %u %u %u\n", id, j,
             m_stats->mem_access_type_stats[GLOBAL_ACC_R][id][j]);
    /* [한국어] Global 메모리 READ 접근 수: 채널id, 뱅크j, 횟수 */
    gzprintf(visualizer_file, "dramglobal_acc_w: %u %u %u\n", id, j,
             m_stats->mem_access_type_stats[GLOBAL_ACC_W][id][j]);
    /* [한국어] Global 메모리 WRITE 접근 수 */
    gzprintf(visualizer_file, "dramlocal_acc_r: %u %u %u\n", id, j,
             m_stats->mem_access_type_stats[LOCAL_ACC_R][id][j]);
    /* [한국어] Local 메모리(per-thread 스택) READ 접근 수 */
    gzprintf(visualizer_file, "dramlocal_acc_w: %u %u %u\n", id, j,
             m_stats->mem_access_type_stats[LOCAL_ACC_W][id][j]);
    /* [한국어] Local 메모리 WRITE 접근 수 */
    gzprintf(visualizer_file, "dramconst_acc_r: %u %u %u\n", id, j,
             m_stats->mem_access_type_stats[CONST_ACC_R][id][j]);
    /* [한국어] Constant 메모리 READ 접근 수 */
    gzprintf(visualizer_file, "dramtexture_acc_r: %u %u %u\n", id, j,
             m_stats->mem_access_type_stats[TEXTURE_ACC_R][id][j]);
    /* [한국어] Texture 메모리 READ 접근 수 */
  }
}

/*
 * [한국어]
 * dram_t::set_dram_power_stats - AccelWattch 전력 모델에 DRAM 성능 카운터 노출
 *
 * @cmd: 출력 참조 — 총 커맨드 사이클 수 (n_cmd)
 * @activity: 출력 참조 — 활성 사이클 수 (n_activity)
 * @nop: 출력 참조 — NOP 사이클 수 (n_nop)
 * @act: 출력 참조 — ACT 커맨드 횟수 (n_act)
 * @pre: 출력 참조 — PRE 커맨드 횟수 (n_pre)
 * @rd: 출력 참조 — READ 커맨드 횟수 (n_rd)
 * @wr: 출력 참조 — WRITE 커맨드 횟수 (n_wr)
 * @wr_WB: 출력 참조 — Writeback WRITE 횟수 (n_wr_WB)
 * @req: 출력 참조 — 총 요청 수 (n_req)
 *
 * AccelWattch(전력 모델)가 매 전력 샘플링 인터벌마다 호출하여 DRAM 동작 카운터를 수집한다.
 * 이 카운터들은 McPAT 기반 동적/정적 전력 계산의 입력 데이터로 사용된다.
 * DRAM ACT/PRE는 Row 활성화 전력, RD/WR은 데이터 전송 전력에 대응한다.
 *
 * 실행 컨텍스트: AccelWattch 전력 샘플링 인터벌 (단일 스레드)
 * 호출 체인: accelwattch/power_interface.cc → [dram_t::set_dram_power_stats()]
 */
void dram_t::set_dram_power_stats(unsigned &cmd, unsigned &activity,
                                  unsigned &nop, unsigned &act, unsigned &pre,
                                  unsigned &rd, unsigned &wr, unsigned &wr_WB,
                                  unsigned &req) const {
  // Point power performance counters to low-level DRAM counters
  /* [한국어] AccelWattch 전력 카운터 포인터를 DRAM 내부 카운터 값에 복사한다.
   * 참조 파라미터를 통해 호출자의 변수를 직접 설정하므로 반환값이 없다. */
  cmd = n_cmd;           /* [한국어] 총 커맨드 사이클 수 → McPAT 배경 전력 계산에 사용 */
  activity = n_activity; /* [한국어] 활성 사이클 수 → DRAM 활성화 전력 추정에 사용 */
  nop = n_nop;           /* [한국어] NOP 사이클 수 → 유휴 전력 계산에 사용 */
  act = n_act;           /* [한국어] ACT 커맨드 횟수 → Row 활성화 에너지 계산에 사용 */
  pre = n_pre;           /* [한국어] PRE 커맨드 횟수 → Precharge 에너지 계산에 사용 */
  rd = n_rd;             /* [한국어] READ 커맨드 횟수 → 데이터 읽기 에너지 계산에 사용 */
  wr = n_wr;             /* [한국어] WRITE 커맨드 횟수 → 데이터 쓰기 에너지 계산에 사용 */
  wr_WB = n_wr_WB;       /* [한국어] Writeback WRITE 횟수 → 캐시 eviction 에너지 계산에 사용 */
  req = n_req;           /* [한국어] 총 요청 수 → 전력 모델의 활동 지표로 사용 */
}

/*
 * [한국어]
 * dram_t::get_bankgrp_number - 뱅크 번호에서 Bank Group 번호 계산
 *
 * @i: 뱅크 번호 (0 ~ nbk-1)
 * @return: 이 뱅크가 속하는 Bank Group 번호 (0 ~ nbkgrp-1)
 *
 * dram_bnkgrp_indexing_policy에 따라 두 가지 방식으로 그룹 번호를 추출한다:
 *   HIGHER_BITS: i >> bk_tag_length (뱅크 번호 상위 비트 사용)
 *     예) nbk=16, nbkgrp=4, bk_tag_length=2 → bk[13](0b1101) → 0b1101>>2=3
 *   LOWER_BITS: i & (nbkgrp-1) (뱅크 번호 하위 비트 AND 마스크)
 *     예) nbkgrp=4, bk[13](0b1101) → 0b1101 & 0b0011 = 0b0001 = 1
 * 이 함수는 issue_col_command(), issue_row_command(), cycle() 통계 루프에서
 * bkgrp[] 배열 인덱싱에 사용된다.
 *
 * 실행 컨텍스트: dram_t::cycle() 및 커맨드 발행 함수 내부 (매 DRAM 사이클, 단일 스레드)
 * 호출 체인: cycle()/issue_col_command()/issue_row_command() → [get_bankgrp_number()]
 */
unsigned dram_t::get_bankgrp_number(unsigned i) {
  if (m_config->dram_bnkgrp_indexing_policy == HIGHER_BITS) {  // higher bits
    return i >> m_config->bk_tag_length;
    /* [한국어] 뱅크 번호를 bk_tag_length만큼 오른쪽 시프트하여 상위 비트로 그룹 번호 추출.
     * bk_tag_length = log2(nbk/nbkgrp) = 그룹당 뱅크 수의 비트 폭 */
  } else if (m_config->dram_bnkgrp_indexing_policy ==
             LOWER_BITS) {  // lower bits
    return i & ((m_config->nbkgrp - 1));
    /* [한국어] 뱅크 번호에 (nbkgrp-1) AND 마스크를 적용하여 하위 비트로 그룹 번호 추출.
     * nbkgrp가 2의 거듭제곱이므로 (nbkgrp-1)은 하위 비트 마스크가 된다. */
  } else {
    assert(1);  /* [한국어] 정의되지 않은 Bank Group 인덱싱 정책 → 시뮬레이션 중단 (설정 오류) */
  }
  return 0;  // we should never get here
  /* [한국어] assert(1) 이후 도달 불가 경로 — 컴파일러 경고 억제를 위한 반환문 */
}
