// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, George L. Yuan,
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
 * [한국어 설명] DRAM FR-FCFS 스케줄러 구현 (dram_sched.cc)
 *
 * === 파일의 역할 ===
 * dram_sched.h에 선언된 frfcfs_scheduler 클래스의 모든 메서드를 구현한다.
 * FR-FCFS(First-Ready First-Come-First-Served) 스케줄링 알고리즘은 DRAM의
 * row-buffer locality를 최대한 활용한다. 현재 열려 있는 row로의 접근 요청이
 * 있으면 도착 순서에 관계없이 먼저 발행하고, 없을 때만 FCFS 순서(대기 가장 긴
 * 요청의 row)로 row를 전환한다. 또한 dram_t::scheduler_frfcfs()를 구현하여
 * DRAM 컨트롤러의 mrqq 입력 큐를 비워 스케줄러로 넘기고, 각 bank에 요청을
 * 배정하는 최상위 DRAM 스케줄링 흐름을 담당한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * gpgpu-sim 타이밍 모델의 메모리 서브시스템에서 DRAM 컨트롤러(dram.cc) 내부에
 * 포함되는 스케줄링 정책 구현체이다. gpu-sim.cc의 사이클 루프가 dram_t::cycle()을
 * 호출하면, 그 안에서 scheduler_frfcfs()가 호출되어 이 파일의 로직이 실행된다.
 * 호출 체인: gpgpu_sim::cycle() → dram_t::cycle() → dram_t::scheduler_frfcfs()
 *            → frfcfs_scheduler::schedule() → dram bank 상태 갱신
 * 실행 컨텍스트: 시뮬레이터 메인 스레드, 매 시뮬레이션 사이클마다 동기 실행.
 *
 * === 타 모듈과의 연결 ===
 * - dram_sched.h: 이 파일이 구현하는 클래스 선언 포함
 * - abstract_hardware_model.h: mem_fetch 타입 (dram_req_t::data 필드 접근용)
 * - gpu-misc.h: LOGB2 매크로 (레이턴시 통계 버킷 계산용)
 * - gpu-sim.h: memory_config, memory_stats_t, gpgpu_sim 타입
 * - mem_latency_stat.h: mrq_lat_table, tot_mrq_latency 등 지연시간 통계 구조체
 * 데이터 흐름: dram_t::mrqq(입력 큐) → add_req() → m_queue/m_write_queue →
 *              schedule() → dram_t::bk[b]->mrq(bank 서비스 큐) → DRAM 타이밍 모델
 *
 * === 주요 함수/구조체 요약 ===
 * frfcfs_scheduler::frfcfs_scheduler() - bank별 큐/bin/타임스탬프 배열 초기화
 * frfcfs_scheduler::add_req()          - 새 요청을 큐와 row-bin에 동시 등록
 * frfcfs_scheduler::data_collection()  - row 전환 시 서비스 시간/activate 통계 갱신
 * frfcfs_scheduler::schedule()         - FR-FCFS 기준 발행 요청 선택 및 큐에서 제거
 * dram_t::scheduler_frfcfs()           - mrqq를 비우고 bank에 요청 배정하는 최상위 루틴
 */

#include "dram_sched.h"
#include "../abstract_hardware_model.h"  /* [한국어] mem_fetch 타입 (dram_req_t::data→is_write() 호출용) */
#include "gpu-misc.h"                    /* [한국어] LOGB2 매크로 — 레이턴시를 log2 버킷으로 변환하여 히스토그램 통계 기록 */
#include "gpu-sim.h"                     /* [한국어] memory_config, memory_stats_t, gpgpu_sim::gpu_sim_cycle 접근 */
#include "mem_latency_stat.h"            /* [한국어] tot_mrq_latency, mrq_lat_table 등 메모리 요청 레이턴시 통계 구조체 */

/*
 * [한국어]
 * frfcfs_scheduler::frfcfs_scheduler - FR-FCFS 스케줄러 생성자
 *
 * @config: DRAM 설정 파라미터 포인터 (nbk, 큐 크기, watermark 등)
 * @dm:     이 스케줄러가 소속된 dram_t 컨트롤러 포인터
 * @stats:  전역 메모리 통계 구조체 포인터
 * @return: 없음 (생성자)
 *
 * bank 수(config->nbk)에 맞게 m_queue, m_bins, m_last_row 배열을 동적 할당하고,
 * 모든 bank의 초기 상태(빈 큐, NULL last_row, 타임스탬프 0)를 설정한다.
 * seperate_write_queue_enabled가 true이면 쓰기 전용 자료구조도 추가 할당한다.
 * 초기 동작 모드는 READ_MODE(읽기 우선)이다.
 *
 * 호출 체인: dram_t 생성자 → [이 함수]
 */
frfcfs_scheduler::frfcfs_scheduler(const memory_config *config, dram_t *dm,
                                   memory_stats_t *stats) {
  m_config = config;    /* [한국어] DRAM 설정 포인터 저장 — nbk, 큐 크기 등 이후 모든 메서드에서 참조 */
  m_stats = stats;      /* [한국어] 통계 구조체 포인터 저장 — data_collection(), schedule()에서 갱신 */
  m_num_pending = 0;    /* [한국어] 읽기 대기 요청 수 초기화 — add_req()에서 증가, schedule()에서 감소 */
  m_num_write_pending = 0; /* [한국어] 쓰기 대기 요청 수 초기화 — 분리 쓰기 큐 미사용 시 항상 0 */
  m_dram = dm;          /* [한국어] 부모 DRAM 컨트롤러 포인터 저장 — data_collection()에서 id, gpu_sim_cycle 접근 */
  m_queue = new std::list<dram_req_t *>[m_config->nbk];  /* [한국어] bank별 읽기 요청 리스트 배열 동적 할당 (nbk 크기) */
  m_bins = new std::map<
      unsigned, std::list<std::list<dram_req_t *>::iterator> >[m_config->nbk]; /* [한국어] bank별 row→이터레이터 맵 배열 동적 할당 — row-hit 탐색 가속 보조 인덱스 */
  m_last_row =
      new std::list<std::list<dram_req_t *>::iterator> *[m_config->nbk]; /* [한국어] bank별 현재 서비스 row bin 포인터 배열 동적 할당 */
  curr_row_service_time = new unsigned[m_config->nbk]; /* [한국어] bank별 현재 row 서비스 누적 시간 배열 동적 할당 */
  row_service_timestamp = new unsigned[m_config->nbk]; /* [한국어] bank별 현재 row 서비스 시작 시점(사이클) 배열 동적 할당 */
  for (unsigned i = 0; i < m_config->nbk; i++) {   /* [한국어] 모든 bank에 대해 초기 상태 설정 루프 */
    m_queue[i].clear();                             /* [한국어] 읽기 요청 리스트 초기화 — 빈 리스트 상태로 시작 */
    m_bins[i].clear();                              /* [한국어] row→이터레이터 맵 초기화 — 빈 맵 상태로 시작 */
    m_last_row[i] = NULL;                           /* [한국어] 현재 서비스 row 없음 표시 — 첫 schedule() 시 row 선택 필요 */
    curr_row_service_time[i] = 0;                   /* [한국어] 현재 row 서비스 시간 0으로 초기화 */
    row_service_timestamp[i] = 0;                   /* [한국어] row 서비스 시작 시점 0으로 초기화 — 아직 서비스 시작 전 */
  }
  if (m_config->seperate_write_queue_enabled) {   /* [한국어] 분리 쓰기 큐 옵션이 활성화된 경우 쓰기 전용 자료구조 추가 할당 */
    m_write_queue = new std::list<dram_req_t *>[m_config->nbk]; /* [한국어] bank별 쓰기 요청 전용 리스트 배열 할당 */
    m_write_bins = new std::map<
        unsigned, std::list<std::list<dram_req_t *>::iterator> >[m_config->nbk]; /* [한국어] bank별 쓰기 요청용 row→이터레이터 맵 배열 할당 */
    m_last_write_row =
        new std::list<std::list<dram_req_t *>::iterator> *[m_config->nbk]; /* [한국어] bank별 쓰기 서비스 중 row bin 포인터 배열 할당 */

    for (unsigned i = 0; i < m_config->nbk; i++) { /* [한국어] 모든 bank의 쓰기 전용 자료구조 초기화 루프 */
      m_write_queue[i].clear();                     /* [한국어] 쓰기 요청 리스트 초기화 */
      m_write_bins[i].clear();                      /* [한국어] 쓰기 row 맵 초기화 */
      m_last_write_row[i] = NULL;                   /* [한국어] 쓰기 서비스 중 row 없음 표시 */
    }
  }
  m_mode = READ_MODE; /* [한국어] 스케줄러 초기 모드는 READ_MODE — 기본적으로 읽기 요청을 우선 처리 */
}

/*
 * [한국어]
 * frfcfs_scheduler::add_req - 새 DRAM 요청을 스케줄러 큐에 등록
 *
 * @req: 등록할 dram_req_t 포인터.
 *       req->bk(bank 번호), req->row(목표 DRAM row), req->data(mem_fetch*)를 사용.
 * @return: 없음 (void)
 *
 * 분리 쓰기 큐 활성화 시 쓰기 요청이면 m_write_queue/m_write_bins에,
 * 그 외 모든 요청은 m_queue/m_bins에 등록한다. 두 경우 모두 동일한 이중 등록
 * 패턴을 사용: 1) 해당 bank의 FIFO 리스트 앞에 삽입(push_front),
 * 2) 리스트 내 이터레이터를 row 번호 기준 bin에도 등록.
 * push_front로 최신 요청이 리스트 앞에 위치하지만, schedule()은 FR-FCFS를 위해
 * .back()(가장 오래 대기한 요청)을 기준으로 FCFS를 처리한다.
 * assert로 큐 용량 초과를 검사하여 백프레셔(back-pressure)가 동작했는지 확인한다.
 *
 * 호출 체인: dram_t::scheduler_frfcfs() → [이 함수]
 */
void frfcfs_scheduler::add_req(dram_req_t *req) {
  if (m_config->seperate_write_queue_enabled && req->data->is_write()) {
    /* [한국어] 분리 쓰기 큐 활성화 상태에서 쓰기 요청인 경우 쓰기 전용 큐에 등록 */
    assert(m_num_write_pending < m_config->gpgpu_frfcfs_dram_write_queue_size);
    /* [한국어] 쓰기 큐 용량 초과 방지 — 이 assert가 터지면 상위 레벨의 back-pressure가 실패한 것 */
    m_num_write_pending++;  /* [한국어] 쓰기 대기 카운터 증가 — schedule()의 watermark 비교에 사용 */
    m_write_queue[req->bk].push_front(req); /* [한국어] 해당 bank의 쓰기 큐 앞에 삽입 (최신 요청이 앞) */
    std::list<dram_req_t *>::iterator ptr = m_write_queue[req->bk].begin();
    /* [한국어] 방금 삽입한 요소의 이터레이터 획득 — begin()이 push_front된 첫 요소를 가리킴 */
    m_write_bins[req->bk][req->row].push_front(ptr);  // newest reqs to the
                                                      // front
    /* [한국어] 쓰기 bin에도 이터레이터 등록 — m_write_queue와 m_write_bins는 항상 동기화 상태 유지 */
  } else {
    /* [한국어] 읽기 요청이거나 분리 쓰기 큐 미사용인 경우 일반 읽기/공용 큐에 등록 */
    assert(m_num_pending < m_config->gpgpu_frfcfs_dram_sched_queue_size);
    /* [한국어] 읽기 큐 용량 초과 방지 — 이 assert가 터지면 L2 캐시에서 back-pressure를 놓친 것 */
    m_num_pending++;  /* [한국어] 읽기 대기 카운터 증가 */
    m_queue[req->bk].push_front(req);  /* [한국어] 해당 bank의 읽기 큐 앞에 삽입 */
    std::list<dram_req_t *>::iterator ptr = m_queue[req->bk].begin();
    /* [한국어] 방금 삽입한 요소의 이터레이터 획득 */
    m_bins[req->bk][req->row].push_front(ptr);  // newest reqs to the front
    /* [한국어] 읽기 bin에 이터레이터 등록 — schedule()이 row-hit 탐색 시 이 bin을 조회 */
  }
}

/*
 * [한국어]
 * frfcfs_scheduler::data_collection - row 전환 시 서비스 시간 및 activate 통계 갱신
 *
 * @bank: 통계를 갱신할 bank 인덱스 (0 ~ m_config->nbk-1)
 * @return: 없음 (void)
 *
 * schedule()에서 row miss가 발생하여 새로운 row를 선택해야 할 때 호출된다.
 * 직전 row가 서비스받은 시간(현재 사이클 - row_service_timestamp)을 계산하고,
 * 최댓값 통계(max_servicetime2samerow)를 갱신한다.
 * 그 후 직전 row에 동시에 접근한 최대 요청 수(max_conc_access2samerow)를
 * concurrent_row_access와 비교하여 최댓값을 기록하고 concurrent_row_access를 0으로 초기화.
 * 마지막으로 num_activates를 증가시켜 이 bank에서의 DRAM row activate 횟수를 누적한다.
 * DRAM activate는 closed row를 열기 위한 비용이 큰 연산이므로 이 통계는 성능 분석에 중요하다.
 *
 * 호출 체인: frfcfs_scheduler::schedule() → [이 함수]
 */
void frfcfs_scheduler::data_collection(unsigned int bank) {
  if (m_dram->m_gpu->gpu_sim_cycle > row_service_timestamp[bank]) {
    /* [한국어] 시뮬레이션이 진행된 경우(현재 사이클 > 서비스 시작 시점)에만 시간 계산 */
    curr_row_service_time[bank] =
        m_dram->m_gpu->gpu_sim_cycle - row_service_timestamp[bank];
    /* [한국어] 직전 row가 서비스된 사이클 수 = 현재 사이클 - row 서비스 시작 사이클 */
    if (curr_row_service_time[bank] >
        m_stats->max_servicetime2samerow[m_dram->id][bank])
      m_stats->max_servicetime2samerow[m_dram->id][bank] =
          curr_row_service_time[bank];
    /* [한국어] 동일 row 연속 서비스 최대 시간 갱신 — 행 블록 레이턴시(row-blocking) 분석용 */
  }
  curr_row_service_time[bank] = 0;  /* [한국어] 새 row 서비스 시작을 위해 서비스 시간 초기화 */
  row_service_timestamp[bank] = m_dram->m_gpu->gpu_sim_cycle;
  /* [한국어] 새 row의 서비스 시작 시점을 현재 사이클로 갱신 */
  if (m_stats->concurrent_row_access[m_dram->id][bank] >
      m_stats->max_conc_access2samerow[m_dram->id][bank]) {
    m_stats->max_conc_access2samerow[m_dram->id][bank] =
        m_stats->concurrent_row_access[m_dram->id][bank];
    /* [한국어] 동일 row에 동시에 서비스된 최대 요청 수 갱신 — row-buffer hit 효율 지표 */
  }
  m_stats->concurrent_row_access[m_dram->id][bank] = 0;
  /* [한국어] 새 row 서비스 시작을 위해 동시 접근 카운터 초기화 */
  m_stats->num_activates[m_dram->id][bank]++;
  /* [한국어] DRAM row activate 횟수 증가 — activate는 closed row를 열기 위한 비용이 큰 연산 */
}

/*
 * [한국어]
 * frfcfs_scheduler::schedule - FR-FCFS 기준으로 지정 bank의 다음 발행 요청 선택
 *
 * @bank:     요청을 선택할 bank 인덱스 (0 ~ m_config->nbk-1)
 * @curr_row: 현재 해당 bank에 열려 있는 DRAM row (row-buffer 상태)
 * @return:   선택된 dram_req_t 포인터 (큐에서 제거된 상태로 반환).
 *            발행할 요청이 없으면 NULL 반환.
 *
 * FR-FCFS 정책의 핵심 구현:
 *   1) 현재 m_last_row[bank] 캐시가 있으면(이전 사이클에 선택된 row가 있으면)
 *      그 row의 bin에서 가장 오래 대기한 요청(.back())을 선택한다.
 *   2) 캐시가 NULL이면 m_current_bins[bank].find(curr_row)로 row-hit를 탐색:
 *      - 현재 열린 curr_row에 요청이 있으면 그 bin을 캐시(rowhit=true)
 *      - 없으면 FCFS 기준(.back()의 row)으로 row를 선택하고 data_collection()을
 *        호출해 row 전환 통계를 기록(rowhit=false)
 * 분리 쓰기 큐 모드: m_num_write_pending과 watermark를 비교하여 m_mode를 전환하고,
 * WRITE_MODE이면 m_write_queue/m_write_bins를 m_current_queue/m_current_bins로 사용.
 * 발행 후 큐와 bin에서 요청을 제거하고, bin이 비면 m_last_row를 NULL로 초기화.
 * 발행된 요청 정보로 rowblp(row-level bank-level parallelism) 통계를 갱신한다.
 *
 * 호출 체인: dram_t::scheduler_frfcfs() → [이 함수] → data_collection()
 */
dram_req_t *frfcfs_scheduler::schedule(unsigned bank, unsigned curr_row) {
  // row
  bool rowhit = true;  /* [한국어] 현재 열린 row에 대한 요청인지 여부 — rowblp 통계 구분용 */
  std::list<dram_req_t *> *m_current_queue = m_queue;
  /* [한국어] 기본적으로 읽기 큐를 현재 큐로 사용 — WRITE_MODE에서 쓰기 큐로 교체됨 */
  std::map<unsigned, std::list<std::list<dram_req_t *>::iterator> >
      *m_current_bins = m_bins;
  /* [한국어] 기본적으로 읽기 bin을 현재 bin으로 사용 */
  std::list<std::list<dram_req_t *>::iterator> **m_current_last_row =
      m_last_row;
  /* [한국어] 기본적으로 읽기 last_row 캐시를 사용 */

  if (m_config->seperate_write_queue_enabled) {
    /* [한국어] 분리 쓰기 큐 옵션이 활성화된 경우 watermark 기반으로 모드 전환 */
    if (m_mode == READ_MODE &&
        ((m_num_write_pending >= m_config->write_high_watermark)
         // || (m_queue[bank].empty() && !m_write_queue[bank].empty())
         )) {
      m_mode = WRITE_MODE;
      /* [한국어] 쓰기 대기 수가 상위 watermark에 도달하면 WRITE_MODE로 전환
       * — 쓰기 요청이 너무 쌓이면 읽기를 멈추고 쓰기를 소진하여 DRAM 효율 유지 */
    } else if (m_mode == WRITE_MODE &&
               ((m_num_write_pending < m_config->write_low_watermark)
                //  || (!m_queue[bank].empty() && m_write_queue[bank].empty())
                )) {
      m_mode = READ_MODE;
      /* [한국어] 쓰기 대기 수가 하위 watermark 미만으로 내려가면 READ_MODE로 복귀
       * — 이력 현상(hysteresis)으로 모드 전환을 너무 빈번히 하지 않도록 상하 watermark 분리 */
    }
  }

  if (m_mode == WRITE_MODE) {
    /* [한국어] WRITE_MODE에서는 쓰기 전용 큐/bin/last_row를 현재 처리 대상으로 교체 */
    m_current_queue = m_write_queue;       /* [한국어] 쓰기 큐를 현재 큐로 전환 */
    m_current_bins = m_write_bins;         /* [한국어] 쓰기 bin을 현재 bin으로 전환 */
    m_current_last_row = m_last_write_row; /* [한국어] 쓰기 last_row 캐시를 현재 캐시로 전환 */
  }

  if (m_current_last_row[bank] == NULL) {
    /* [한국어] 현재 서비스 중인 row가 없으면(캐시가 NULL이면) 새 row 선택 */
    if (m_current_queue[bank].empty()) return NULL;
    /* [한국어] 해당 bank에 대기 요청이 없으면 발행할 요청 없음 → NULL 반환 */

    std::map<unsigned, std::list<std::list<dram_req_t *>::iterator> >::iterator
        bin_ptr = m_current_bins[bank].find(curr_row);
    /* [한국어] FR-FCFS의 "FR(First-Ready)" 부분: 현재 열린 row에 대한 요청이 있는지 탐색 */
    if (bin_ptr == m_current_bins[bank].end()) {
      /* [한국어] curr_row에 대한 요청이 없음 → row miss, FCFS 기준으로 가장 오래 대기한 요청의 row 선택 */
      dram_req_t *req = m_current_queue[bank].back();
      /* [한국어] .back()은 가장 일찍 삽입된(가장 오래 대기한) 요청 — push_front로 삽입했으므로 tail이 가장 오래된 것 */
      bin_ptr = m_current_bins[bank].find(req->row);
      /* [한국어] 가장 오래된 요청의 row에 해당하는 bin 탐색 */
      assert(bin_ptr !=
             m_current_bins[bank].end());  // where did the request go???
      /* [한국어] bin은 항상 m_queue와 동기화되어야 하므로 반드시 찾아야 함 */
      m_current_last_row[bank] = &(bin_ptr->second);
      /* [한국어] 새로 선택한 row의 bin 포인터를 캐시에 저장 */
      data_collection(bank);  /* [한국어] row 전환 시 직전 row 서비스 시간 및 activate 통계 갱신 */
      rowhit = false;          /* [한국어] row miss 발생 표시 — rowblp 통계에서 miss로 카운트 */
    } else {
      /* [한국어] curr_row에 요청이 있음 → row hit, 현재 row 계속 서비스 */
      m_current_last_row[bank] = &(bin_ptr->second);
      /* [한국어] 현재 열린 row의 bin 포인터를 캐시에 저장 */
      rowhit = true;  /* [한국어] row hit 표시 — rowblp 통계에서 hit으로 카운트 */
    }
  }
  std::list<dram_req_t *>::iterator next = m_current_last_row[bank]->back();
  /* [한국어] 선택된 row의 bin에서 가장 오래 대기한 요청의 이터레이터 획득
   * (.back()은 bin에 push_front로 삽입했으므로 가장 먼저 들어온 요청을 가리킴) */
  dram_req_t *req = (*next);
  /* [한국어] 이터레이터를 역참조하여 실제 dram_req_t 포인터 획득 */

  // rowblp stats
  m_dram->access_num++;  /* [한국어] DRAM 전체 접근 횟수 증가 (row-level bank-level parallelism 분석용) */
  bool is_write = req->data->is_write();  /* [한국어] 이 요청이 쓰기인지 확인 — 읽기/쓰기별 통계 분리 */
  if (is_write)
    m_dram->write_num++;   /* [한국어] 쓰기 접근 횟수 증가 */
  else
    m_dram->read_num++;    /* [한국어] 읽기 접근 횟수 증가 */

  if (rowhit) {
    /* [한국어] row-buffer hit인 경우 hit 통계 증가 */
    m_dram->hits_num++;    /* [한국어] 전체 row-hit 횟수 증가 */
    if (is_write)
      m_dram->hits_write_num++;  /* [한국어] 쓰기 row-hit 횟수 증가 */
    else
      m_dram->hits_read_num++;   /* [한국어] 읽기 row-hit 횟수 증가 */
  }

  m_stats->concurrent_row_access[m_dram->id][bank]++;
  /* [한국어] 현재 row에 연속으로 서비스된 요청 수 증가 — data_collection()에서 최댓값 갱신에 사용 */
  m_stats->row_access[m_dram->id][bank]++;
  /* [한국어] 이 bank의 전체 row 접근 횟수 증가 */
  m_current_last_row[bank]->pop_back();
  /* [한국어] bin에서 발행할 요청의 이터레이터 제거 (.back()을 pop_back으로 삭제) */

  m_current_queue[bank].erase(next);
  /* [한국어] 메인 FIFO 큐에서 실제 요청 제거 — iterator next로 O(1) 삭제 가능 */
  if (m_current_last_row[bank]->empty()) {
    /* [한국어] 현재 서비스 중인 row의 bin이 비었으면 bin 엔트리와 캐시 포인터 모두 초기화 */
    m_current_bins[bank].erase(req->row);
    /* [한국어] 빈 bin 엔트리를 맵에서 삭제하여 메모리 누수 방지 */
    m_current_last_row[bank] = NULL;
    /* [한국어] 캐시 포인터 NULL로 초기화 — 다음 schedule() 호출 시 새 row 선택 유도 */
  }
#ifdef DEBUG_FAST_IDEAL_SCHED
  if (req)
    printf("%08u : DRAM(%u) scheduling memory request to bank=%u, row=%u\n",
           (unsigned)gpu_sim_cycle, m_dram->id, req->bk, req->row);
  /* [한국어] DEBUG_FAST_IDEAL_SCHED 컴파일 플래그 활성화 시 스케줄링 결정 디버그 출력 */
#endif

  if (m_config->seperate_write_queue_enabled && req->data->is_write()) {
    /* [한국어] 분리 쓰기 큐 모드에서 쓰기 요청이 발행된 경우 쓰기 대기 카운터 감소 */
    assert(req != NULL && m_num_write_pending != 0);
    /* [한국어] req가 NULL이면 로직 오류, m_num_write_pending이 0이면 카운터 언더플로 오류 */
    m_num_write_pending--;  /* [한국어] 쓰기 대기 수 감소 — watermark 비교에 반영 */
  } else {
    /* [한국어] 읽기 요청이 발행된 경우 읽기 대기 카운터 감소 */
    assert(req != NULL && m_num_pending != 0);
    /* [한국어] req가 NULL이면 로직 오류, m_num_pending이 0이면 카운터 언더플로 오류 */
    m_num_pending--;  /* [한국어] 읽기 대기 수 감소 */
  }

  return req;  /* [한국어] 큐에서 제거된 dram_req_t 반환 — dram_t::scheduler_frfcfs()가 bank->mrq에 배정 */
}

/*
 * [한국어]
 * frfcfs_scheduler::print - 각 bank의 대기 요청 수 디버그 출력
 *
 * @fp: 출력 대상 파일 포인터 (현재 구현에서는 무시하고 printf를 직접 사용)
 * @return: 없음 (void)
 *
 * m_config->nbk 개 bank 각각의 읽기 큐(m_queue) 길이를 순서대로 출력한다.
 * 시뮬레이션 상태 진단이나 디버깅 시 스케줄러의 큐 포화 여부를 빠르게 파악하는 용도.
 * 쓰기 큐(m_write_queue)는 출력하지 않는 한계가 있다.
 *
 * 호출 체인: 외부 디버그 경로(예: dram_t::print()) → [이 함수]
 */
void frfcfs_scheduler::print(FILE *fp) {
  for (unsigned b = 0; b < m_config->nbk; b++) {  /* [한국어] 모든 bank에 대해 순서대로 출력 */
    printf(" %u: queue length = %u\n", b, (unsigned)m_queue[b].size());
    /* [한국어] bank 인덱스와 해당 bank의 읽기 대기 큐 길이 출력 */
  }
}

/*
 * [한국어]
 * dram_t::scheduler_frfcfs - DRAM FR-FCFS 스케줄러 최상위 진입 루틴
 *
 * @return: 없음 (void), dram_t의 멤버 함수
 *
 * 이 함수는 두 단계로 동작한다:
 * 1) 입력 큐(mrqq) 비우기: mrqq에 쌓인 모든 요청을 pop하여 frfcfs_scheduler에
 *    add_req()로 등록한다. 이때 전력/접근 통계(total_n_access 등)를 갱신하고
 *    mem_fetch의 상태를 IN_PARTITION_MC_INPUT_QUEUE로 업데이트한다.
 * 2) Bank 배정 (Round-Robin + FR-FCFS): prio 변수로 라운드로빈 시작 bank를
 *    결정하고, 각 bank에 이미 처리 중인 요청(mrq)이 없으면 schedule()을 호출하여
 *    다음 발행 요청을 선택한다. 요청을 찾으면 bank의 mrq에 배정하고
 *    레이턴시 통계(mrq_latency, mrq_lat_table)를 기록한 뒤 루프를 종료(break).
 *    즉 한 사이클에 최대 하나의 bank에만 새 요청이 배정된다.
 * 실행 컨텍스트: dram_t::cycle()에서 매 DRAM 사이클마다 동기 호출된다.
 *
 * 호출 체인: dram_t::cycle() → [이 함수] → frfcfs_scheduler::add_req(),
 *                                           frfcfs_scheduler::schedule()
 */
void dram_t::scheduler_frfcfs() {
  unsigned mrq_latency;  /* [한국어] 메모리 요청 큐잉 레이턴시 임시 변수 — 통계 기록용 */
  frfcfs_scheduler *sched = m_frfcfs_scheduler;  /* [한국어] 이 DRAM 컨트롤러에 속한 FR-FCFS 스케줄러 포인터 */
  while (!mrqq->empty()) {
    /* [한국어] L2 캐시 등 상위 레벨에서 도착한 미처리 요청을 모두 스케줄러로 이관 */
    dram_req_t *req = mrqq->pop();  /* [한국어] 입력 큐(mrqq)에서 요청 꺼내기 */

    // Power stats
    // if(req->data->get_type() != READ_REPLY && req->data->get_type() !=
    // WRITE_ACK)
    m_stats->total_n_access++;  /* [한국어] 전력 모델(AccelWattch)용 총 DRAM 접근 횟수 증가 */

    if (req->data->get_type() == WRITE_REQUEST) {
      m_stats->total_n_writes++;  /* [한국어] 쓰기 요청 총 횟수 증가 — 전력/통계 분석용 */
    } else if (req->data->get_type() == READ_REQUEST) {
      m_stats->total_n_reads++;   /* [한국어] 읽기 요청 총 횟수 증가 — 전력/통계 분석용 */
    }

    req->data->set_status(IN_PARTITION_MC_INPUT_QUEUE,
                          m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    /* [한국어] mem_fetch의 상태를 "메모리 컨트롤러 입력 큐"로 업데이트하여 레이턴시 추적 활성화
     * — gpu_sim_cycle + gpu_tot_sim_cycle은 절대 시뮬레이션 시간 */
    sched->add_req(req);  /* [한국어] FR-FCFS 스케줄러에 요청 등록 (bank별 큐 및 row-bin에 삽입) */
  }

  dram_req_t *req;   /* [한국어] schedule()이 반환한 발행 대상 요청 포인터 */
  unsigned i;        /* [한국어] bank 순회 루프 변수 */
  for (i = 0; i < m_config->nbk; i++) {
    /* [한국어] 라운드로빈 방식으로 bank 순회 — prio에서 시작하여 공평하게 bank 간 우선순위 부여 */
    unsigned b = (i + prio) % m_config->nbk;
    /* [한국어] prio를 기준으로 라운드로빈 순서의 bank 인덱스 계산 */
    if (!bk[b]->mrq) {
      /* [한국어] 해당 bank에 현재 처리 중인 요청이 없을 때만 새 요청 배정 */
      req = sched->schedule(b, bk[b]->curr_row);
      /* [한국어] FR-FCFS 스케줄러에 bank b와 현재 열린 row를 전달하여 발행할 요청 선택 */

      if (req) {
        /* [한국어] 발행 가능한 요청이 선택된 경우 */
        req->data->set_status(IN_PARTITION_MC_BANK_ARB_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        /* [한국어] mem_fetch 상태를 "메모리 컨트롤러 bank 중재 큐"로 업데이트 */
        prio = (prio + 1) % m_config->nbk;
        /* [한국어] 다음 사이클의 라운드로빈 시작 bank를 한 칸 전진 — bank 간 공평성 보장 */
        bk[b]->mrq = req;
        /* [한국어] 선택된 요청을 bank의 처리 중 요청 슬롯에 배정 — DRAM 타이밍 모델이 이후 처리 */
        if (m_config->gpgpu_memlatency_stat) {
          /* [한국어] 메모리 레이턴시 통계 수집이 활성화된 경우 지연시간 측정 */
          mrq_latency = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle -
                        bk[b]->mrq->timestamp;
          /* [한국어] 큐잉 레이턴시 = 현재 사이클 - 요청 도착 타임스탬프 */
          m_stats->tot_mrq_latency += mrq_latency;  /* [한국어] 총 레이턴시 누적 */
          m_stats->tot_mrq_num++;                   /* [한국어] 레이턴시 측정 요청 수 증가 */
          bk[b]->mrq->timestamp =
              m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
          /* [한국어] 타임스탬프 갱신 — 이후 단계에서의 레이턴시 측정을 위한 기준점 재설정 */
          m_stats->mrq_lat_table[LOGB2(mrq_latency)]++;
          /* [한국어] LOGB2로 레이턴시를 log2 버킷으로 변환하여 히스토그램 테이블에 기록 */
          if (mrq_latency > m_stats->max_mrq_latency) {
            m_stats->max_mrq_latency = mrq_latency;
            /* [한국어] 최대 메모리 요청 큐잉 레이턴시 갱신 */
          }
        }

        break;
        /* [한국어] 한 사이클에 최대 하나의 bank에만 요청을 배정 — 다음 사이클에 나머지 bank 순서 처리 */
      }
    }
  }
}
