// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, George L. Yuan
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
 * [한국어 설명] DRAM FR-FCFS 스케줄러 헤더 (dram_sched.h)
 *
 * === 파일의 역할 ===
 * DRAM 메모리 컨트롤러의 요청 스케줄링 정책을 정의하는 헤더 파일이다.
 * FR-FCFS(First-Ready First-Come-First-Served) 알고리즘을 구현하는
 * frfcfs_scheduler 클래스를 선언한다. FR-FCFS는 현재 열린 row에 접근하는
 * 요청(row-buffer hit)을 FCFS 순서보다 먼저 발행함으로써 DRAM 대역폭을
 * 극대화하는 스케줄링 전략이다. 선택적으로 읽기/쓰기 요청 큐를 분리하여
 * 쓰기 요청이 읽기 레이턴시를 방해하지 않도록 제어할 수 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * gpgpu-sim 타이밍 모델의 메모리 서브시스템 계층에 속한다.
 * dram.cc의 dram_t::scheduler_frfcfs()가 매 DRAM 사이클마다 이 스케줄러를
 * 호출하여 각 bank에서 다음에 발행할 메모리 요청을 결정한다.
 * 호출 체인: gpu-sim.cc cycle() → dram.cc dram_t::cycle() →
 *            dram_t::scheduler_frfcfs() → frfcfs_scheduler::schedule()
 * 실행 컨텍스트: 시뮬레이터 메인 스레드의 사이클 루프 내 동기 실행이며
 * 병렬 접근은 없다.
 *
 * === 타 모듈과의 연결 ===
 * - dram.h: dram_t(DRAM 컨트롤러), dram_req_t(메모리 요청 패킷) 의존
 * - gpu-sim.h: memory_config(DRAM 설정 파라미터), memory_stats_t(통계) 의존
 * - shader.h: mem_fetch 타입 간접 의존 (dram_req_t::data 필드를 통해)
 * - dram_sched.cc: 이 헤더를 포함하여 모든 메서드를 구현
 * 데이터 흐름: mrqq(메모리 요청 입력 큐) → frfcfs_scheduler::add_req() →
 *              m_queue/m_write_queue → schedule() → dram bank의 mrq 필드
 *
 * === 주요 함수/구조체 요약 ===
 * frfcfs_scheduler::add_req()  - 새 메모리 요청을 bank별 큐와 row별 bin에 등록
 * frfcfs_scheduler::schedule() - 지정 bank에서 FR-FCFS 기준 최적 요청 선택
 * frfcfs_scheduler::data_collection() - row 전환 시 서비스 시간 통계 갱신
 * frfcfs_scheduler::print()    - 각 bank의 대기 큐 길이 디버그 출력
 * memory_mode enum             - READ_MODE/WRITE_MODE 스케줄러 동작 모드
 */

#ifndef dram_sched_h_INCLUDED
#define dram_sched_h_INCLUDED

#include <list>   /* [한국어] std::list - bank별 요청 큐 및 row별 bin의 이터레이터 리스트에 사용 */
#include <map>    /* [한국어] std::map  - row 번호를 키로 요청 이터레이터를 그룹화하는 bin 자료구조에 사용 */
#include "dram.h"       /* [한국어] dram_t(DRAM 컨트롤러), dram_req_t(요청 패킷) 타입 제공 */
#include "gpu-misc.h"   /* [한국어] GPU 시뮬레이터 공통 유틸리티 및 매크로 */
#include "gpu-sim.h"    /* [한국어] memory_config(DRAM 설정), memory_stats_t(통계 구조체) 제공 */
#include "shader.h"     /* [한국어] shader_core_ctx 및 mem_fetch 타입 간접 의존 */

/* [한국어] DRAM 스케줄러의 동작 모드를 나타내는 열거형.
 * FR-FCFS 스케줄러가 분리 쓰기 큐(seperate_write_queue) 옵션 활성화 시
 * 현재 읽기 요청을 처리 중인지 쓰기 요청을 처리 중인지 추적한다.
 * 쓰기 대기 수가 write_high_watermark를 초과하면 WRITE_MODE로 전환되어
 * 쓰기 요청을 우선 소진하고, write_low_watermark 미만이 되면 다시 READ_MODE로 복귀한다. */
enum memory_mode { READ_MODE = 0, WRITE_MODE };

/*
 * [한국어]
 * frfcfs_scheduler - FR-FCFS(First-Ready First-Come-First-Served) DRAM 스케줄러
 *
 * DRAM bank별로 메모리 요청 큐를 관리하고, row-buffer hit 요청을 우선 발행하여
 * DRAM 대역폭 효율을 극대화하는 스케줄러 클래스이다.
 * 핵심 자료구조는 두 레벨의 인덱싱을 사용한다:
 *   1) m_queue[bank]: 해당 bank의 모든 대기 요청 (FCFS 순으로 연결 리스트)
 *   2) m_bins[bank][row]: 동일 row를 향하는 요청들의 이터레이터 집합
 *      → row-buffer hit 요청을 O(1)에 찾을 수 있도록 row 단위로 그룹화
 * m_last_row[bank]는 현재 서비스 중인 row의 bin 포인터를 캐시하여
 * 연속적인 row-hit 발행을 효율적으로 처리한다.
 * 분리 쓰기 큐 옵션(seperate_write_queue_enabled) 활성화 시 m_write_queue/
 * m_write_bins/m_last_write_row를 추가로 관리하며 m_mode로 읽기/쓰기 전환한다.
 */
class frfcfs_scheduler {
 public:
  /*
   * [한국어]
   * frfcfs_scheduler - 생성자: bank별 큐/bin/통계 자료구조 초기화
   *
   * @config: DRAM 설정 (nbk, gpgpu_frfcfs_dram_sched_queue_size 등)
   *          dram_t로부터 전달된 memory_config 포인터
   * @dm:     이 스케줄러가 속한 DRAM 컨트롤러 객체 (통계 접근용)
   * @stats:  글로벌 메모리 통계 구조체 (row 서비스 시간, activate 횟수 등 기록)
   *
   * bank 수(nbk)만큼 배열을 동적 할당하고 초기화한다.
   * seperate_write_queue_enabled 설정 시 쓰기 전용 큐/bin도 추가 할당한다.
   * 모드는 항상 READ_MODE로 시작한다.
   *
   * 호출 체인: dram_t 생성자 → [이 함수]
   */
  frfcfs_scheduler(const memory_config *config, dram_t *dm,
                   memory_stats_t *stats);

  /*
   * [한국어]
   * add_req - 새 DRAM 요청을 bank별 큐 및 row별 bin에 등록
   *
   * @req: dram_t의 mrqq에서 꺼낸 dram_req_t 포인터.
   *       req->bk(bank 번호), req->row(목표 row), req->data(mem_fetch*)를 사용한다.
   * @return: 없음 (void)
   *
   * 요청을 m_queue[req->bk]의 앞쪽(front)에 삽입하고(최신 요청이 앞),
   * 동시에 m_bins[req->bk][req->row]에 그 이터레이터를 등록한다.
   * seperate_write_queue_enabled이고 요청이 쓰기이면 m_write_queue/m_write_bins에
   * 삽입하고 m_num_write_pending을 증가시키고, 그렇지 않으면 m_queue/m_bins에
   * 삽입하고 m_num_pending을 증가시킨다.
   * assert로 큐 크기 한도 초과를 검사한다.
   *
   * 호출 체인: dram_t::scheduler_frfcfs() → [이 함수]
   */
  void add_req(dram_req_t *req);

  /*
   * [한국어]
   * data_collection - bank의 row 전환 시점에 row 서비스 시간 통계를 갱신
   *
   * @bank: 통계를 갱신할 bank 인덱스 (0 ~ nbk-1)
   * @return: 없음 (void)
   *
   * 직전 row 서비스 시간(curr_row_service_time)을 계산하여 최댓값 통계
   * (max_servicetime2samerow)를 갱신한다. 또한 동일 row에 동시 접근한
   * 최대 요청 수(max_conc_access2samerow)를 갱신하고 카운터를 초기화한다.
   * 마지막으로 num_activates를 증가시켜 row activate 횟수를 누적한다.
   * schedule()에서 row miss가 발생할 때(새로운 row를 활성화해야 할 때) 호출된다.
   *
   * 호출 체인: frfcfs_scheduler::schedule() → [이 함수]
   */
  void data_collection(unsigned bank);

  /*
   * [한국어]
   * schedule - 지정 bank에서 FR-FCFS 기준으로 다음에 발행할 요청을 선택
   *
   * @bank:     발행 대상 bank 인덱스
   * @curr_row: 현재 해당 bank에 열려 있는 row (DRAM row-buffer 상태)
   * @return:   선택된 dram_req_t 포인터. 발행할 요청이 없으면 NULL.
   *            반환된 요청은 큐에서 제거된다.
   *
   * FR-FCFS 정책: 현재 열린 curr_row에 대한 요청(row-buffer hit)이 있으면
   * 그것을 우선 반환한다. hit 요청이 없으면 가장 오래 대기한 요청(FCFS)의
   * row로 전환하고 data_collection()을 호출해 row 전환 통계를 기록한다.
   * seperate_write_queue_enabled 모드에서는 m_mode(READ/WRITE)에 따라
   * 읽기 또는 쓰기 큐 중 하나를 선택하여 동일 로직을 적용한다.
   * m_last_row 캐시 포인터를 활용해 동일 row의 연속 요청을 효율적으로 처리한다.
   * rowblp(row-level bank-level parallelism) 통계도 갱신한다.
   *
   * 호출 체인: dram_t::scheduler_frfcfs() → [이 함수] → data_collection()
   */
  dram_req_t *schedule(unsigned bank, unsigned curr_row);

  /*
   * [한국어]
   * print - 각 bank의 대기 요청 수를 출력 (디버그용)
   *
   * @fp: 출력 대상 파일 포인터 (실제로는 내부에서 printf를 사용)
   * @return: 없음 (void)
   *
   * 각 bank의 m_queue 크기를 순서대로 출력한다. 시뮬레이션 상태 진단 시 사용.
   *
   * 호출 체인: 외부 디버그 경로 → [이 함수]
   */
  void print(FILE *fp);

  /* [한국어] 현재 읽기 큐에 대기 중인 요청 수를 반환하는 인라인 접근자.
   * dram_t가 버퍼 포화 여부를 빠르게 확인할 때 사용한다.
   * 값 범위: 0 ~ gpgpu_frfcfs_dram_sched_queue_size. */
  unsigned num_pending() const { return m_num_pending; }

  /* [한국어] 현재 쓰기 전용 큐에 대기 중인 요청 수를 반환하는 인라인 접근자.
   * seperate_write_queue_enabled 시에만 의미 있는 값을 반환한다.
   * 값 범위: 0 ~ gpgpu_frfcfs_dram_write_queue_size. */
  unsigned num_write_pending() const { return m_num_write_pending; }

 private:
  const memory_config *m_config;
  /* [한국어] DRAM 설정 파라미터 포인터 (bank 수, 큐 크기, watermark 등).
   * 설정자: 생성자에서 주입. 읽는 자: add_req(), schedule() 전반에서 참조.
   * 값 범위: 시뮬레이터 초기화 시 gpgpusim.config에서 로드된 불변 객체.
   * 동기화: 단일 스레드 시뮬레이터에서 읽기 전용으로 사용. */

  dram_t *m_dram;
  /* [한국어] 이 스케줄러가 속한 DRAM 컨트롤러 객체 포인터.
   * 설정자: 생성자에서 주입. 읽는 자: data_collection()에서 m_dram->id,
   *         m_dram->m_gpu->gpu_sim_cycle, m_dram->access_num 등 접근.
   * 값 범위: 유효한 dram_t 포인터 (NULL 불가).
   * 동기화: 단일 스레드 접근이므로 별도 락 불필요. */

  unsigned m_num_pending;
  /* [한국어] 읽기 전용 큐(m_queue)에 현재 대기 중인 요청의 총 수.
   * 설정자: add_req()에서 증가, schedule()에서 감소.
   * 읽는 자: num_pending() 접근자, add_req()에서 상한 assert 검사.
   * 값 범위: 0 ~ gpgpu_frfcfs_dram_sched_queue_size.
   * 동기화: 단일 스레드 시뮬레이터에서 원자성 불필요. */

  unsigned m_num_write_pending;
  /* [한국어] 쓰기 전용 큐(m_write_queue)에 현재 대기 중인 요청의 총 수.
   * seperate_write_queue_enabled가 false이면 항상 0.
   * 설정자: add_req()에서 증가, schedule()에서 감소.
   * 읽는 자: num_write_pending() 접근자, schedule()의 watermark 비교.
   * 값 범위: 0 ~ gpgpu_frfcfs_dram_write_queue_size.
   * 동기화: 단일 스레드. */

  std::list<dram_req_t *> *m_queue;
  /* [한국어] bank별 읽기 요청 대기 리스트 배열 (크기: nbk).
   * 각 원소는 해당 bank를 향한 미발행 읽기(또는 비분리 모드의 쓰기) 요청 리스트.
   * push_front로 삽입하므로 최신 요청이 앞에 위치하지만, schedule()은
   * m_bins를 통해 row-hit 우선, 그 다음 .back()(가장 오래된 요청) 기준으로 선택.
   * 설정자: 생성자에서 new[], add_req()에서 push_front.
   * 읽는 자: schedule()에서 erase, print()에서 size().
   * 동기화: 단일 스레드. */

  std::map<unsigned, std::list<std::list<dram_req_t *>::iterator> > *m_bins;
  /* [한국어] bank별 row 인덱스(key) → 해당 row를 향한 요청 이터레이터 리스트(value) 맵 배열.
   * m_bins[bank][row] 에 저장된 이터레이터들은 m_queue[bank]의 요소들을 가리킨다.
   * row-buffer hit 요청을 O(log n)으로 찾을 수 있게 하는 보조 인덱스 역할.
   * 설정자: add_req()에서 push_front, schedule()에서 pop_back 및 erase.
   * 읽는 자: schedule()에서 find()로 현재 row의 요청 집합 탐색.
   * 동기화: 단일 스레드. */

  std::list<std::list<dram_req_t *>::iterator> **m_last_row;
  /* [한국어] bank별로 현재 서비스 중인 row의 bin 포인터 캐시 배열 (크기: nbk).
   * NULL이면 다음 schedule() 호출 시 row를 새로 선택해야 함을 의미.
   * 동일 row에 대한 연속 요청을 처리하는 동안 bin 탐색을 건너뛰어 성능 향상.
   * 설정자: schedule()에서 row 선택 후 설정, 해당 row의 요청이 소진되면 NULL로 초기화.
   * 읽는 자: schedule()에서 다음 발행 요청 결정.
   * 동기화: 단일 스레드. */

  unsigned *curr_row_service_time;  // one set of variables for each bank.
  /* [한국어] bank별 현재 row 서비스 시간 (사이클 단위) 배열 (크기: nbk).
   * 하나의 row가 얼마나 오래 독점적으로 서비스를 받았는지 측정하는 누적값.
   * 설정자: data_collection()에서 경과 사이클로 계산 후 갱신.
   * 읽는 자: data_collection()에서 max_servicetime2samerow 통계 갱신 시.
   * 동기화: 단일 스레드. */

  unsigned *row_service_timestamp;  // tracks when scheduler began servicing
                                    // current row
  /* [한국어] bank별 현재 row 서비스 시작 시점(gpu_sim_cycle) 배열 (크기: nbk).
   * data_collection() 호출 시 이 값을 기준으로 현재 row의 서비스 시간을 계산.
   * 설정자: 생성자에서 0으로 초기화, data_collection()에서 현재 사이클로 갱신.
   * 읽는 자: data_collection()에서 현재 사이클과의 차이를 계산.
   * 동기화: 단일 스레드. */

  std::list<dram_req_t *> *m_write_queue;
  /* [한국어] bank별 쓰기 요청 전용 대기 리스트 배열 (크기: nbk).
   * seperate_write_queue_enabled가 true일 때만 생성되고 사용됨.
   * 쓰기 요청을 읽기 큐와 분리하여 쓰기 폭풍(write storm)이 읽기 레이턴시를
   * 방해하지 않도록 한다. watermark 기반으로 m_mode를 전환해 소진한다.
   * 설정자: 생성자에서 new[], add_req()에서 push_front.
   * 읽는 자: schedule()에서 m_mode가 WRITE_MODE일 때 m_current_queue로 사용.
   * 동기화: 단일 스레드. */

  std::map<unsigned, std::list<std::list<dram_req_t *>::iterator> >
      *m_write_bins;
  /* [한국어] 쓰기 전용 큐의 row별 이터레이터 그룹 맵 배열 (크기: nbk).
   * m_bins의 쓰기 요청 버전. WRITE_MODE에서 schedule()이 row-hit를 찾을 때 사용.
   * 설정자: 생성자에서 new[], add_req()에서 push_front.
   * 읽는 자: schedule()에서 WRITE_MODE일 때 m_current_bins로 사용.
   * 동기화: 단일 스레드. */

  std::list<std::list<dram_req_t *>::iterator> **m_last_write_row;
  /* [한국어] 쓰기 요청의 현재 서비스 row bin 포인터 캐시 배열 (크기: nbk).
   * m_last_row의 쓰기 큐 버전. WRITE_MODE에서 m_current_last_row로 사용.
   * 설정자: schedule()에서 WRITE_MODE 진입 시 설정, 소진 시 NULL로 초기화.
   * 읽는 자: schedule()에서 WRITE_MODE일 때 참조.
   * 동기화: 단일 스레드. */

  enum memory_mode m_mode;
  /* [한국어] 스케줄러의 현재 동작 모드 (READ_MODE 또는 WRITE_MODE).
   * seperate_write_queue_enabled가 false이면 항상 READ_MODE.
   * 설정자: 생성자에서 READ_MODE로 초기화, schedule()에서 watermark 조건에 따라 전환.
   * 읽는 자: schedule()에서 어느 큐를 사용할지 결정.
   * 전환 조건: m_num_write_pending >= write_high_watermark → WRITE_MODE,
   *            m_num_write_pending < write_low_watermark → READ_MODE.
   * 동기화: 단일 스레드. */

  memory_stats_t *m_stats;
  /* [한국어] 전역 메모리 통계 구조체 포인터.
   * data_collection()에서 max_servicetime2samerow, max_conc_access2samerow,
   * concurrent_row_access, num_activates 등을 갱신한다.
   * schedule()에서 concurrent_row_access, row_access를 증가시킨다.
   * 설정자: 생성자에서 주입. 읽는 자: data_collection(), schedule().
   * 동기화: 단일 스레드 시뮬레이터에서 별도 락 불필요. */
};

#endif
