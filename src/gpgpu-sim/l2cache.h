// Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas,
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
 * [한국어 설명] L2 캐시 및 메모리 파티션 헤더 (l2cache.h)
 *
 * === 파일의 역할 ===
 * GPGPU-Sim의 L2 캐시 슬라이스와 메모리 파티션(DRAM 채널)을 모델링하는
 * 세 클래스를 선언한다:
 *   - partition_mf_allocator: L2 캐시 내부에서 새 mem_fetch 패킷을 동적 생성하는 팩토리
 *   - memory_partition_unit: DRAM 채널 1개에 대응하는 최상위 파티션 유닛
 *     (여러 sub_partition을 중재하며 DRAM 타이밍 모델을 소유)
 *   - memory_sub_partition: DRAM 채널 내 L2 캐시 슬라이스 1개에 대응
 *     (ICNT↔L2↔DRAM 사이의 FIFO 큐와 ROP 지연 큐를 관리)
 * 이 계층은 SM에서 출발한 mem_fetch가 NoC를 통해 도달한 후 L2 캐시 룩업 →
 * DRAM 접근(미스 시) → 응답 반환의 전체 흐름을 구현한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPU 메모리 계층:
 *   SM(L1 미스) → ICNT(intersim2) → memory_sub_partition::push()
 *               → m_icnt_L2_queue → L2 캐시 룩업
 *               → 미스 시 m_L2_dram_queue → memory_partition_unit::dram_cycle()
 *               → dram_t(FRFCFS 스케줄러) → DRAM 접근
 *               → m_dram_L2_queue → L2 채우기 → m_L2_icnt_queue
 *               → ICNT → SM(L1 채우기)
 * 실행 컨텍스트: 호스트 유저스페이스, gpgpu_sim::cycle() 내 cache_cycle()/dram_cycle() 호출
 * 사이클 단위 모델: 매 사이클마다 cache_cycle()/dram_cycle()이 호출됨
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - abstract_hardware_model.h: mem_fetch, mem_access_t, warp_inst_t, fifo_pipeline 등
 *   - dram.h: dram_t — DRAM 타이밍 시뮬레이터 (FRFCFS 스케줄러 포함)
 *   - gpu-cache.h: l2_cache — 캐시 룩업/채우기/flush/invalidate 구현
 *   - gpu-sim.h: gpgpu_sim — 전역 시뮬레이터 상태
 *   - mem_latency_stat.h: memory_stats_t — 레이턴시 통계 수집
 * 이 파일에 의존하는 모듈:
 *   - gpu-sim.cc: memory_partition_unit 배열 생성·관리
 *   - shader.cc: SM이 push()로 mem_fetch를 서브 파티션에 주입
 *   - dram.cc: dram_t가 L2_dram_queue 인터페이스를 통해 요청 수신
 * 데이터 흐름:
 *   SM → [ICNT] → m_icnt_L2_queue → L2cache → m_L2_dram_queue →
 *   [dram_t] → m_dram_L2_queue → L2cache 채우기 → m_L2_icnt_queue → [ICNT] → SM
 *
 * === 주요 함수/구조체 요약 ===
 * partition_mf_allocator::alloc()    - L2 내부 라이트백용 mem_fetch 동적 생성
 * memory_partition_unit::cache_cycle()- ICNT 팝, 아비트레이션, DRAM 지연 큐 처리
 * memory_partition_unit::dram_cycle() - DRAM 타이밍 모델 1 사이클 진행
 * arbitration_metadata               - 서브 파티션 간 DRAM 크레딧 기반 중재 구조체
 * memory_sub_partition::cache_cycle()- L2 캐시 접근(룩업/채우기/ROP 지연) 처리
 * memory_sub_partition::push/pop/top()- ICNT←→서브 파티션 인터페이스
 * L2interface                        - l2_cache가 L2→DRAM 큐에 push하는 인터페이스
 */

#ifndef MC_PARTITION_INCLUDED
#define MC_PARTITION_INCLUDED

#include "../abstract_hardware_model.h"  /* [한국어] mem_fetch, fifo_pipeline, warp_inst_t 등 하드웨어 추상 모델 */
#include "dram.h"                        /* [한국어] dram_t — DRAM 타이밍 시뮬레이터 */

#include <list>   /* [한국어] std::list — dram_delay_t 지연 큐에 사용 */
#include <queue>  /* [한국어] std::queue — ROP 지연 큐(rop_delay_t)에 사용 */

class mem_fetch;  /* [한국어] 전방 선언: 메모리 요청 패킷 */

/*
 * [한국어]
 * partition_mf_allocator - L2 캐시 내부에서 새 mem_fetch 패킷을 생성하는 팩토리 클래스
 *
 * mem_fetch_allocator 인터페이스를 구현하며, L2 캐시가 라이트백(write-back)이나
 * 채우기(fill) 과정에서 새 mem_fetch 객체를 동적으로 생성할 때 사용한다.
 * warp_inst_t 기반 alloc()은 abort() — L2 파티션 컨텍스트에서는 사용하지 않음.
 * 실제로 사용되는 두 alloc() 오버로드는 주소/타입/크기를 받아 새 mem_fetch를 생성한다.
 */
class partition_mf_allocator : public mem_fetch_allocator {
 public:
  /*
   * [한국어]
   * partition_mf_allocator 생성자
   *
   * @config: memory_config — mem_fetch 생성에 필요한 메모리 구성 파라미터
   * m_memory_config에 저장하여 alloc() 시 참조한다.
   */
  partition_mf_allocator(const memory_config *config) {
    m_memory_config = config;  /* [한국어] 메모리 구성 참조 저장 — alloc() 시 mem_fetch 생성에 사용 */
  }

  /*
   * [한국어]
   * alloc (warp_inst_t 기반) - 지원하지 않는 오버로드 (abort)
   *
   * @inst: warp 명령어 (L2 파티션 컨텍스트에서는 사용 안 함)
   * @access: 메모리 접근 타입
   * @cycle: 현재 사이클
   * @return: NULL (실제로는 abort()로 종료)
   *
   * L2 파티션에서는 warp 명령어 기반 alloc을 사용하지 않는다.
   * 호출 시 프로그램이 abort() 처리된다.
   */
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const {
    abort();   /* [한국어] 이 오버로드는 L2 파티션에서 사용하지 않음 — 호출 시 프로그램 종료 */
    return NULL;
  }

  /*
   * [한국어]
   * alloc (주소/타입/크기 기반, 단순) - L2 라이트백용 mem_fetch 생성
   *
   * @addr: 대상 물리 주소
   * @type: 메모리 접근 타입 (L2_WR_ALLOC_R, L1_WRBK_ACC 등)
   * @size: 요청 크기 (바이트)
   * @wr: true=쓰기, false=읽기
   * @cycle: 현재 사이클
   * @streamID: CUDA 스트림 ID
   * @return: 새로 할당된 mem_fetch 포인터
   *
   * l2cache.cc에 구현. L2 미스 시 DRAM 읽기 요청 생성에 사용.
   */
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned long long streamID) const;

  /*
   * [한국어]
   * alloc (전체 정보 기반) - 섹터 마스크를 포함한 완전한 mem_fetch 생성
   *
   * @addr: 대상 물리 주소
   * @type: 메모리 접근 타입
   * @active_mask: 활성 스레드 마스크 (warp 내 32 스레드)
   * @byte_mask: 바이트 단위 접근 마스크
   * @sector_mask: 섹터 단위 접근 마스크 (캐시 섹터 선택)
   * @size: 요청 크기 (바이트)
   * @wr: true=쓰기
   * @cycle: 현재 사이클
   * @wid: warp ID
   * @sid: SM ID
   * @tpc: TPC(텍스처 처리 클러스터) ID
   * @original_mf: 이 요청을 유발한 원본 mem_fetch (연결 추적용)
   * @streamID: CUDA 스트림 ID
   * @return: 새로 할당된 mem_fetch 포인터
   *
   * 섹터 기반 캐시(sector cache) 환경에서 채우기/라이트백에 사용.
   */
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned wid, unsigned sid, unsigned tpc,
                           mem_fetch *original_mf,
                           unsigned long long streamID) const;

 private:
  const memory_config *m_memory_config;
  /* [한국어] 메모리 구성 참조 — alloc() 시 mem_fetch 생성에 필요한 파라미터 제공.
   * 설정자: 생성자에서 1회 초기화, 이후 읽기 전용.
   * 읽는 자: alloc() 오버로드들.
   * 동기화: 읽기 전용, 단일 스레드. */
};

// Memory partition unit contains all the units assolcated with a single DRAM
// channel.
// - It arbitrates the DRAM channel among multiple sub partitions.
// - It does not connect directly with the interconnection network.
/*
 * [한국어]
 * memory_partition_unit - DRAM 채널 1개에 대응하는 최상위 메모리 파티션 유닛
 *
 * 하나의 DRAM 채널(메모리 파티션)에 속한 모든 유닛을 포함한다:
 *   - m_sub_partition[]: 이 채널을 나누는 서브 파티션(L2 캐시 슬라이스) 배열
 *   - m_dram: DRAM 타이밍 모델 (FRFCFS 스케줄러 포함)
 *   - arbitration_metadata: 서브 파티션 간 DRAM 접근 크레딧 기반 중재
 *   - m_dram_latency_queue: L2→DRAM 사이의 고정 레이턴시 모델 큐
 * ICNT와는 직접 연결되지 않으며, 각 서브 파티션이 ICNT 인터페이스를 담당한다.
 */
class memory_partition_unit {
 public:
  /*
   * [한국어]
   * memory_partition_unit 생성자 - 파티션 유닛 초기화
   *
   * @partition_id: 이 파티션의 전역 ID (0..m_n_mem-1)
   * @config: 메모리 구성 (m_n_sub_partition_per_channel, DRAM 파라미터 등)
   * @stats: 메모리 레이턴시 통계 수집 객체
   * @gpu: 시뮬레이터 전역 상태
   *
   * m_sub_partition 배열을 생성하고 dram_t를 초기화한다.
   */
  memory_partition_unit(unsigned partition_id, const memory_config *config,
                        class memory_stats_t *stats, class gpgpu_sim *gpu);
  ~memory_partition_unit();

  /*
   * [한국어]
   * busy - 이 파티션이 아직 처리 중인 요청이 있는지 확인
   *
   * @return: 서브 파티션 중 하나라도 busy()이면 true
   *
   * gpgpu_sim이 시뮬레이션을 종료할지 결정할 때 사용한다.
   * 모든 파티션이 not-busy여야 커널 완료로 판정한다.
   */
  bool busy() const;

  /*
   * [한국어]
   * cache_cycle - L2 캐시 및 DRAM 큐 아비트레이션 1 사이클 진행
   *
   * @cycle: 현재 시뮬레이션 사이클 번호
   *
   * 매 사이클마다 호출되어:
   *   1. ICNT 팝: m_icnt→m_sub_partition[i]에 도달한 패킷 처리
   *   2. 서브 파티션 간 DRAM 크레딧 아비트레이션
   *   3. m_dram_latency_queue: ready_cycle 도달 시 DRAM에 요청 전달
   *   4. 각 서브 파티션의 cache_cycle() 호출
   *   5. DRAM→L2 응답 처리 (dram_L2_queue)
   */
  void cache_cycle(unsigned cycle);

  /*
   * [한국어]
   * dram_cycle - DRAM 타이밍 모델 1 사이클 진행
   *
   * m_dram->cycle()을 호출하여 FRFCFS 스케줄러가 다음 DRAM 명령을
   * 선택하고 타이밍을 처리하게 한다.
   * gpgpu_sim::cycle()에서 cache_cycle()과 별도로 호출된다.
   */
  void dram_cycle();

  /*
   * [한국어]
   * simple_dram_model_cycle - 단순화된 DRAM 모델(고정 레이턴시) 1 사이클 진행
   *
   * -gpgpu_simple_dram_model 옵션 활성화 시 dram_cycle() 대신 호출.
   * FRFCFS 스케줄러 대신 고정 레이턴시(dram_latency 파라미터)로 모델링.
   */
  void simple_dram_model_cycle();

  /*
   * [한국어]
   * set_done - mem_fetch의 완료 상태를 설정하고 통계 기록
   *
   * @mf: 완료된 메모리 요청 패킷
   *
   * DRAM 응답이 L2로 돌아올 때 호출되어 mem_fetch의 상태를 완료로 변경하고
   * memlatstat_done()을 통해 레이턴시를 기록한다.
   */
  void set_done(mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file) const;
  /* [한국어] AerialVision 가시화 파일에 파티션 통계 출력 */

  void print_stat(FILE *fp) { m_dram->print_stat(fp); }
  /* [한국어] DRAM 통계 출력 — m_dram에 위임 */

  void visualize() const { m_dram->visualize(); }
  /* [한국어] DRAM 시각화 — m_dram에 위임 */

  void print(FILE *fp) const;
  /* [한국어] 파티션 전체 상태 출력 (서브 파티션, DRAM 포함) */

  void handle_memcpy_to_gpu(size_t dst_start_addr, unsigned subpart_id,
                            mem_access_sector_mask_t mask);
  /* [한국어] cudaMemcpy → GPU 메모리 복사 처리: L2 태그 강제 업데이트 */

  class memory_sub_partition *get_sub_partition(int sub_partition_id) {
    return m_sub_partition[sub_partition_id];
    /* [한국어] 서브 파티션 ID로 해당 memory_sub_partition 포인터 반환
     * gpu-sim.cc가 ICNT 라우팅 시 사용 */
  }

  // Power model
  /*
   * [한국어]
   * set_dram_power_stats - 전력 모델용 DRAM 통계 카운터를 out-param으로 제공
   *
   * @n_cmd, @n_activity, @n_nop, @n_act, @n_pre, @n_rd, @n_wr, @n_wr_WB, @n_req:
   *   [출력] DRAM 명령 수 카운터들 — m_dram의 내부 카운터를 복사
   *
   * AccelWattch가 mcpat_cycle() 호출 시 DRAM 전력 계산에 사용한다.
   */
  void set_dram_power_stats(unsigned &n_cmd, unsigned &n_activity,
                            unsigned &n_nop, unsigned &n_act, unsigned &n_pre,
                            unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
                            unsigned &n_req) const;

  /*
   * [한국어]
   * global_sub_partition_id_to_local_id - 전역 서브 파티션 ID를 파티션 내 로컬 ID로 변환
   *
   * @global_sub_partition_id: 전역 서브 파티션 ID (0..총 서브 파티션 수-1)
   * @return: 이 파티션 내 로컬 서브 파티션 ID (0..m_n_sub_partition-1)
   *
   * gpu-sim.cc가 ICNT 패킷을 올바른 서브 파티션으로 라우팅할 때 사용.
   */
  int global_sub_partition_id_to_local_id(int global_sub_partition_id) const;

  unsigned get_mpid() const { return m_id; }
  /* [한국어] 이 메모리 파티션의 전역 ID 반환 */

  class gpgpu_sim *get_mgpu() const {
    return m_gpu;
    /* [한국어] 시뮬레이터 전역 상태 포인터 반환 */
  }

 private:
  unsigned m_id;
  /* [한국어] 이 메모리 파티션의 전역 ID (0..m_n_mem-1).
   * 설정자: 생성자에서 partition_id 인자로 초기화.
   * 읽는 자: global_sub_partition_id_to_local_id(), get_mpid().
   * 동기화: 읽기 전용. */

  const memory_config *m_config;
  /* [한국어] 메모리 구성 참조 — DRAM 파라미터, 서브 파티션 수 등.
   * 설정자: 생성자에서 초기화. 읽기 전용. */

  class memory_stats_t *m_stats;
  /* [한국어] 레이턴시 통계 수집 객체 참조.
   * 설정자: 생성자에서 stats 인자로 초기화.
   * 읽는 자: memlatstat_icnt2mem_pop() 등 통계 함수 호출 시 전달. */

  class memory_sub_partition **m_sub_partition;
  /* [한국어] 이 파티션에 속한 서브 파티션 포인터 배열 (m_n_sub_partition_per_channel 크기).
   * 설정자: 생성자에서 new memory_sub_partition()으로 동적 생성.
   * 읽는 자: cache_cycle(), dram_cycle(), set_done(), busy() 등.
   * 동기화: 단일 시뮬레이션 루프, 읽기 전용 (포인터 자체는 불변). */

  class dram_t *m_dram;
  /* [한국어] 이 파티션의 DRAM 타이밍 시뮬레이터 (FRFCFS 스케줄러 포함).
   * 설정자: 생성자에서 new dram_t()으로 생성.
   * 읽는 자: dram_cycle(), simple_dram_model_cycle(), set_dram_power_stats().
   * 동기화: 단일 시뮬레이션 루프. */

  /*
   * [한국어]
   * arbitration_metadata - 서브 파티션 간 DRAM 채널 접근 크레딧 기반 중재 메타데이터
   *
   * 여러 서브 파티션이 동일 DRAM 채널을 공유하므로, 특정 서브 파티션이
   * 채널을 독점하는 것을 방지하기 위해 크레딧(credit) 기반 중재를 수행한다.
   * m_private_credit: 각 서브 파티션의 현재 미결 요청 크레딧
   * m_shared_credit: 모든 서브 파티션이 공유하는 추가 크레딧
   * can_issue_to_dram()이 이 구조체를 사용하여 서브 파티션의 DRAM 접근 허용 여부를 결정.
   */
  class arbitration_metadata {
   public:
    /*
     * [한국어]
     * arbitration_metadata 생성자
     *
     * @config: 메모리 구성 — m_private_credit_limit, m_shared_credit_limit 파라미터 포함
     * m_last_borrower, m_shared_credit 초기화 및 m_private_credit 벡터 크기 설정.
     */
    arbitration_metadata(const memory_config *config);

    // check if a subpartition still has credit
    /*
     * [한국어]
     * has_credits - 서브 파티션이 DRAM 접근 크레딧이 남아있는지 확인
     *
     * @inner_sub_partition_id: 파티션 내 서브 파티션 로컬 ID
     * @return: private 크레딧 또는 shared 크레딧이 있으면 true
     */
    bool has_credits(int inner_sub_partition_id) const;

    // borrow a credit for a subpartition
    /*
     * [한국어]
     * borrow_credit - 서브 파티션이 DRAM 요청 발행 시 크레딧 1개 차감
     *
     * @inner_sub_partition_id: 크레딧을 빌리는 서브 파티션 ID
     *
     * m_last_borrower를 갱신하고 private 크레딧이 있으면 private에서,
     * 없으면 shared에서 1 차감한다.
     */
    void borrow_credit(int inner_sub_partition_id);

    // return a credit from a subpartition
    /*
     * [한국어]
     * return_credit - DRAM 응답 완료 시 크레딧 1개 반환
     *
     * @inner_sub_partition_id: 크레딧을 반환하는 서브 파티션 ID
     *
     * private 크레딧 한도 미만이면 private에, 그 외에는 shared에 반환.
     */
    void return_credit(int inner_sub_partition_id);

    // return the last subpartition that borrowed credit
    int last_borrower() const { return m_last_borrower; }
    /* [한국어] 마지막으로 크레딧을 빌린 서브 파티션 ID 반환
     * Round-Robin 중재에서 다음 순서 결정에 사용 */

    /*
     * [한국어]
     * print - 현재 크레딧 상태 출력 (디버그용)
     */
    void print(FILE *fp) const;

   private:
    // id of the last subpartition that borrowed credit
    int m_last_borrower;
    /* [한국어] 마지막으로 크레딧을 빌린 서브 파티션의 로컬 ID.
     * 설정자: borrow_credit()에서 갱신.
     * 읽는 자: last_borrower(), can_issue_to_dram()의 Round-Robin 중재 로직.
     * 값 범위: 0..m_n_sub_partition_per_channel-1.
     * 동기화: 단일 스레드. */

    int m_shared_credit_limit;
    /* [한국어] 공유 크레딧 최대값 (gpgpusim.config 파라미터).
     * 설정자: 생성자에서 config로부터 초기화, 이후 읽기 전용.
     * 읽는 자: borrow_credit() — shared 크레딧 사용 가능 여부 판단. */

    int m_private_credit_limit;
    /* [한국어] 서브 파티션 개별 크레딧 최대값.
     * 설정자: 생성자에서 config로부터 초기화, 이후 읽기 전용.
     * 읽는 자: borrow_credit(), return_credit(). */

    // credits borrowed by the subpartitions
    std::vector<int> m_private_credit;
    /* [한국어] 각 서브 파티션이 현재 사용 중인 private 크레딧 수 벡터.
     * 설정자: 생성자에서 0으로 초기화; borrow_credit()에서 증가, return_credit()에서 감소.
     * 읽는 자: has_credits() — private 크레딧 여유 판단.
     * 값 범위: 0..m_private_credit_limit.
     * 동기화: 단일 스레드 순차 실행. */

    int m_shared_credit;
    /* [한국어] 현재 사용 중인 공유 크레딧 수.
     * 설정자: borrow_credit()에서 증가, return_credit()에서 감소.
     * 읽는 자: has_credits() — shared 크레딧 여유 판단.
     * 값 범위: 0..m_shared_credit_limit.
     * 동기화: 단일 스레드 순차 실행. */
  };
  arbitration_metadata m_arbitration_metadata;
  /* [한국어] 이 파티션의 서브 파티션 간 DRAM 중재 상태.
   * 설정자: 생성자에서 초기화; cache_cycle()에서 borrow/return_credit() 호출로 갱신.
   * 읽는 자: can_issue_to_dram()에서 중재 판단. */

  // determine wheither a given subpartition can issue to DRAM
  /*
   * [한국어]
   * can_issue_to_dram - 특정 서브 파티션이 DRAM에 요청을 발행할 수 있는지 판단
   *
   * @inner_sub_partition_id: 파티션 내 서브 파티션 로컬 ID
   * @return: 크레딧이 있으면 true, 없으면 false
   *
   * m_arbitration_metadata.has_credits()를 확인하고,
   * L2_dram_queue가 비어있지 않은지도 함께 검사한다.
   */
  bool can_issue_to_dram(int inner_sub_partition_id);

  // model DRAM access scheduler latency (fixed latency between L2 and DRAM)
  struct dram_delay_t {
    unsigned long long ready_cycle;
    /* [한국어] 이 요청이 DRAM에 전달될 준비가 되는 사이클 번호.
     * 설정자: cache_cycle()에서 현재 사이클 + m_config->dram_latency로 계산.
     * 읽는 자: cache_cycle()에서 ready_cycle <= 현재 사이클인 요청을 DRAM에 전달. */
    class mem_fetch *req;
    /* [한국어] 이 지연 항목에 대응하는 메모리 요청 패킷.
     * 설정자: L2→DRAM 경로에서 cache_cycle()이 m_dram_latency_queue에 추가할 때 설정.
     * 읽는 자: 지연 만료 시 dram_t에 전달. */
  };
  std::list<dram_delay_t> m_dram_latency_queue;
  /* [한국어] L2→DRAM 고정 레이턴시 모델링 큐.
   * L2 캐시 미스 후 DRAM에 도달하기까지 m_config->dram_latency 사이클의 파이프라인 지연을 모델링.
   * 설정자: cache_cycle()에서 새 항목을 push_back.
   * 읽는 자: cache_cycle()에서 ready_cycle이 지난 항목을 pop하여 m_dram에 전달.
   * 동기화: 단일 시뮬레이션 루프 스레드. */

  class gpgpu_sim *m_gpu;
  /* [한국어] 시뮬레이터 전역 상태 포인터.
   * 설정자: 생성자에서 초기화, 읽기 전용.
   * 읽는 자: 현재 사이클 번호, 통계 컨텍스트 등 조회. */
};

/*
 * [한국어]
 * memory_sub_partition - DRAM 채널 내 L2 캐시 슬라이스 1개에 대응하는 서브 파티션
 *
 * memory_partition_unit(DRAM 채널)의 하위 유닛으로, 하나의 L2 캐시와
 * ICNT ↔ L2 ↔ DRAM 사이의 FIFO 큐 집합을 관리한다.
 * ROP(Raster Operations Pipeline) 지연 큐: SM 쓰기 요청이 L2에 도달하기 전
 *   고정 레이턴시를 모델링하는 단순 FIFO (rop_delay_t).
 * 주요 FIFO 큐 4개:
 *   m_icnt_L2_queue: ICNT→L2 방향 (SM에서 온 요청)
 *   m_L2_dram_queue: L2→DRAM 방향 (L2 미스 요청)
 *   m_dram_L2_queue: DRAM→L2 방향 (DRAM 응답, L2 채우기 입력)
 *   m_L2_icnt_queue: L2→ICNT 방향 (L2 히트 응답 또는 채우기 후 SM으로 반환)
 */
class memory_sub_partition {
 public:
  /*
   * [한국어]
   * memory_sub_partition 생성자
   *
   * @sub_partition_id: 전역 서브 파티션 ID
   * @config: 메모리 구성
   * @stats: 레이턴시 통계 객체
   * @gpu: 시뮬레이터 전역 상태
   *
   * m_L2cache(l2_cache 객체), 4개의 FIFO 큐, m_mf_allocator를 초기화한다.
   */
  memory_sub_partition(unsigned sub_partition_id, const memory_config *config,
                       class memory_stats_t *stats, class gpgpu_sim *gpu);
  ~memory_sub_partition();

  unsigned get_id() const { return m_id; }
  /* [한국어] 전역 서브 파티션 ID 반환 */

  /*
   * [한국어]
   * busy - 이 서브 파티션에 미완료 요청이 있는지 확인
   *
   * @return: m_request_tracker가 비어있지 않으면 true
   *
   * 추적 중인 미완료 mem_fetch가 있으면 파티션이 아직 처리 중임을 의미.
   */
  bool busy() const;

  /*
   * [한국어]
   * cache_cycle - L2 캐시 접근(룩업/채우기/ROP 지연) 1 사이클 진행
   *
   * @cycle: 현재 시뮬레이션 사이클
   *
   * 매 사이클:
   *   1. m_rop에서 ready_cycle이 된 항목을 m_icnt_L2_queue에 이동
   *   2. m_icnt_L2_queue에서 요청 꺼내어 L2 캐시 룩업
   *      - 히트: m_L2_icnt_queue에 응답 추가
   *      - 미스: m_L2_dram_queue에 DRAM 요청 추가
   *   3. m_dram_L2_queue에서 DRAM 응답 받아 L2 채우기
   *   4. 섹터 캐시의 경우 breakdown_request_to_sector_requests()로 분해
   */
  void cache_cycle(unsigned cycle);

  bool full() const;
  /* [한국어] m_icnt_L2_queue가 가득 찼는지 확인 — ICNT가 push 전에 확인 */
  bool full(unsigned size) const;
  /* [한국어] 주어진 크기의 요청이 들어갈 공간이 있는지 확인 */

  /*
   * [한국어]
   * push - ICNT에서 도착한 mem_fetch를 ROP 큐(쓰기) 또는 icnt_L2_queue(읽기)에 추가
   *
   * @mf: ICNT에서 팝된 메모리 요청 패킷
   * @clock_cycle: 현재 사이클 (ROP ready_cycle 계산에 사용)
   *
   * 쓰기 요청: ROP 지연(m_config->rop_latency) 후 처리되도록 m_rop에 추가.
   * 읽기 요청: 바로 m_icnt_L2_queue에 추가.
   * m_request_tracker에 등록하여 미완료 요청 추적 시작.
   */
  void push(class mem_fetch *mf, unsigned long long clock_cycle);

  /*
   * [한국어]
   * pop - m_L2_icnt_queue에서 완료된 응답 패킷 꺼내기
   *
   * @return: m_L2_icnt_queue의 front 패킷 (없으면 NULL)
   *
   * gpu-sim.cc가 ICNT로 보낼 응답을 꺼낼 때 호출.
   */
  class mem_fetch *pop();

  /*
   * [한국어]
   * top - m_L2_icnt_queue의 front 패킷 확인 (꺼내지 않음)
   *
   * @return: m_L2_icnt_queue의 front 패킷 포인터 (없으면 NULL)
   */
  class mem_fetch *top();

  /*
   * [한국어]
   * set_done - mem_fetch 완료 처리 (request_tracker 제거 + 통계 기록)
   *
   * @mf: 완료된 메모리 요청 패킷
   *
   * m_request_tracker에서 제거하고 memlatstat_read_done() 호출.
   */
  void set_done(mem_fetch *mf);

  /*
   * [한국어]
   * flushL2 - L2 캐시의 더티 라인을 DRAM에 모두 라이트백하고 플러시
   *
   * @return: 플러시된 캐시 라인 수
   *
   * cudaDeviceSynchronize() 또는 커널 종료 시 L2를 클린 상태로 만들기 위해 호출.
   */
  unsigned flushL2();

  /*
   * [한국어]
   * invalidateL2 - L2 캐시의 모든 라인을 무효화 (라이트백 없이)
   *
   * @return: 무효화된 캐시 라인 수
   */
  unsigned invalidateL2();

  // interface to L2_dram_queue
  bool L2_dram_queue_empty() const;
  /* [한국어] L2→DRAM 큐가 비어있는지 확인 — dram_t가 요청 가져갈 때 확인 */
  class mem_fetch *L2_dram_queue_top() const;
  /* [한국어] L2→DRAM 큐의 front 패킷 (꺼내지 않음) — dram_t가 다음 요청 확인 */
  void L2_dram_queue_pop();
  /* [한국어] L2→DRAM 큐에서 front 패킷 꺼내기 — dram_t가 요청 수신 확인 후 호출 */

  // interface to dram_L2_queue
  bool dram_L2_queue_full() const;
  /* [한국어] DRAM→L2 응답 큐가 가득 찼는지 확인 — dram_t가 응답 push 전 확인 */
  void dram_L2_queue_push(class mem_fetch *mf);
  /* [한국어] DRAM 응답 패킷을 dram_L2_queue에 추가 — dram_t가 응답 완료 시 호출 */

  void visualizer_print(gzFile visualizer_file);
  /* [한국어] AerialVision 가시화 파일에 서브 파티션 통계 출력 */

  void print_cache_stat(unsigned &accesses, unsigned &misses) const;
  /* [한국어] L2 캐시 접근/미스 수 출력 */

  void print(FILE *fp) const;
  /* [한국어] 서브 파티션 전체 상태 출력 */

  void accumulate_L2cache_stats(class cache_stats &l2_stats) const;
  /* [한국어] L2 캐시 통계를 l2_stats에 누적 — 전체 파티션 합산에 사용 */

  void get_L2cache_sub_stats(struct cache_sub_stats &css) const;
  /* [한국어] L2 캐시 서브 통계 조회 (접근/히트/미스/라이트백 수) */

  // Support for getting per-window L2 stats for AerialVision
  void get_L2cache_sub_stats_pw(struct cache_sub_stats_pw &css) const;
  /* [한국어] AerialVision 샘플링 창별 L2 통계 조회 */

  void clear_L2cache_stats_pw();
  /* [한국어] 샘플링 창 L2 통계 초기화 — clear_L2_stats_pw()에서 호출 */

  void force_l2_tag_update(new_addr_type addr, unsigned time,
                           mem_access_sector_mask_t mask) {
    m_L2cache->force_tag_access(addr, m_memcpy_cycle_offset + time, mask);
    /* [한국어] cudaMemcpy 접근을 L2 태그에 강제 기록
     * m_memcpy_cycle_offset: cudaMemcpy 사이클을 시뮬레이션 사이클과 분리하기 위한 오프셋 */
    m_memcpy_cycle_offset += 1;
    /* [한국어] cudaMemcpy 접근마다 오프셋 1 증가
     * 커널 실행 사이클에 영향을 주지 않도록 분리 */
  }

 private:
  // data
  unsigned m_id;  //< the global sub partition ID
  /* [한국어] 전역 서브 파티션 ID (0..총 서브 파티션 수-1).
   * 설정자: 생성자에서 sub_partition_id 인자로 초기화.
   * 읽는 자: get_id(), ICNT 라우팅, 통계 출력.
   * 동기화: 읽기 전용. */

  const memory_config *m_config;
  /* [한국어] 메모리 구성 참조 — FIFO 큐 크기, ROP 레이턴시, L2 파라미터 등.
   * 설정자: 생성자에서 초기화. 읽기 전용. */

  class l2_cache *m_L2cache;
  /* [한국어] L2 캐시 구현 객체 (gpu-cache.h의 l2_cache).
   * 설정자: 생성자에서 new l2_cache()로 생성.
   * 읽는 자: cache_cycle()의 캐시 룩업/채우기, flushL2(), invalidateL2().
   * 동기화: 단일 스레드. */

  class L2interface *m_L2interface;
  /* [한국어] l2_cache → m_L2_dram_queue 연결 인터페이스 (mem_fetch_interface 구현).
   * l2_cache가 미스 시 DRAM 요청을 발행할 때 이 인터페이스를 통해 m_L2_dram_queue에 push. */

  class gpgpu_sim *m_gpu;
  /* [한국어] 시뮬레이터 전역 상태 참조. */

  partition_mf_allocator *m_mf_allocator;
  /* [한국어] L2 캐시 내부에서 새 mem_fetch 생성 팩토리.
   * L2 라이트백 또는 채우기 시 새 mem_fetch 객체가 필요할 때 사용. */

  // model delay of ROP units with a fixed latency
  struct rop_delay_t {
    unsigned long long ready_cycle;
    /* [한국어] 이 쓰기 요청이 ROP 지연을 마치고 L2에 전달될 준비가 되는 사이클.
     * 설정자: push()에서 현재 사이클 + m_config->rop_latency로 계산.
     * 읽는 자: cache_cycle()에서 ready_cycle <= 현재 사이클인 항목을 L2에 전달. */
    class mem_fetch *req;
    /* [한국어] 이 ROP 지연 항목에 대응하는 쓰기 요청 mem_fetch.
     * 설정자: push()에서 mf로 설정.
     * 읽는 자: cache_cycle()에서 ROP 지연 만료 시 m_icnt_L2_queue에 이동. */
  };
  std::queue<rop_delay_t> m_rop;
  /* [한국어] ROP(Raster Operations Pipeline) 고정 레이턴시 지연 큐.
   * 쓰기 요청이 L2에 실제로 도달하기 전 m_config->rop_latency 사이클을 대기하도록 모델링.
   * 설정자: push()에서 쓰기 요청을 추가.
   * 읽는 자: cache_cycle()에서 ready_cycle 만료 항목을 m_icnt_L2_queue로 이동.
   * 동기화: 단일 스레드. */

  // these are various FIFOs between units within a memory partition
  fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
  /* [한국어] ICNT→L2 방향 FIFO 큐 — SM에서 ICNT를 통해 도착한 메모리 요청.
   * 설정자: push()에서 읽기 요청 또는 ROP 만료 쓰기 요청이 추가됨.
   * 읽는 자: cache_cycle()에서 꺼내 L2 룩업 수행.
   * 크기: m_config->m_L2_icnt_config의 큐 깊이.
   * 동기화: 단일 스레드. */

  fifo_pipeline<mem_fetch> *m_L2_dram_queue;
  /* [한국어] L2→DRAM 방향 FIFO 큐 — L2 미스 시 DRAM에 보낼 요청.
   * 설정자: L2interface::push()에서 추가 (l2_cache 미스 시 자동 호출).
   * 읽는 자: memory_partition_unit::cache_cycle()에서 DRAM 지연 큐로 이동.
   * 크기: m_config->m_L2_dram_config의 큐 깊이.
   * 동기화: 단일 스레드. */

  fifo_pipeline<mem_fetch> *m_dram_L2_queue;
  /* [한국어] DRAM→L2 방향 FIFO 큐 — DRAM 응답 패킷이 L2 채우기를 기다리는 큐.
   * 설정자: dram_L2_queue_push()에서 dram_t가 응답 완료 시 추가.
   * 읽는 자: cache_cycle()에서 꺼내 L2 채우기(fill) 수행.
   * 크기: m_config->m_dram_L2_config의 큐 깊이.
   * 동기화: 단일 스레드. */

  fifo_pipeline<mem_fetch> *m_L2_icnt_queue;  // L2 cache hit response queue
  /* [한국어] L2→ICNT 방향 FIFO 큐 — L2 히트 응답 또는 채우기 완료 후 SM으로 반환할 패킷.
   * 설정자: cache_cycle()에서 L2 히트 또는 채우기 완료 시 추가.
   * 읽는 자: pop()/top()을 통해 gpu-sim.cc가 ICNT로 전달.
   * 크기: m_config->m_L2_icnt_config의 큐 깊이.
   * 동기화: 단일 스레드. */

  class mem_fetch *L2dramout;
  /* [한국어] 현재 L2→DRAM 큐에서 진행 중인 mem_fetch (DRAM 전송 중인 요청).
   * 설정자: cache_cycle()에서 L2_dram_queue에서 꺼낼 때 설정.
   * 읽는 자: DRAM 전송 완료 확인 시 참조.
   * 동기화: 단일 스레드. */

  unsigned long long int wb_addr;
  /* [한국어] 현재 진행 중인 라이트백의 대상 DRAM 주소.
   * 설정자: 라이트백 요청 발행 시 설정.
   * 읽는 자: 라이트백 완료 확인 시 참조.
   * 동기화: 단일 스레드. */

  class memory_stats_t *m_stats;
  /* [한국어] 레이턴시 통계 수집 객체 참조.
   * 설정자: 생성자에서 초기화. 읽기 전용.
   * 읽는 자: cache_cycle()에서 memlatstat_* 함수 호출 시 전달. */

  std::set<mem_fetch *> m_request_tracker;
  /* [한국어] 이 서브 파티션이 현재 처리 중인 모든 미완료 mem_fetch 집합.
   * 설정자: push()에서 insert(); set_done()에서 erase().
   * 읽는 자: busy() — set이 비어있지 않으면 처리 중.
   * 동기화: 단일 스레드. */

  friend class L2interface;  /* [한국어] L2interface가 m_L2_dram_queue에 직접 접근하도록 허용 */

  /*
   * [한국어]
   * breakdown_request_to_sector_requests - mem_fetch를 섹터 단위 요청으로 분해
   *
   * @mf: 분해할 원본 메모리 요청
   * @return: 섹터별 mem_fetch 포인터 벡터
   *
   * 섹터 기반 캐시(sector cache)에서 하나의 캐시 라인 요청을
   * 활성 섹터 마스크에 따라 여러 개의 섹터 요청으로 분해한다.
   */
  std::vector<mem_fetch *> breakdown_request_to_sector_requests(mem_fetch *mf);

  // This is a cycle offset that has to be applied to the l2 accesses to account
  // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
  // kernel execution but we want cudamemcpy to go through the L2. Everytime an
  // access is made from cudamemcpy this counter is incremented, and when the l2
  // is accessed (in both cudamemcpyies and otherwise) this value is added to
  // the gpgpu-sim cycle counters.
  unsigned m_memcpy_cycle_offset;
  /* [한국어] cudaMemcpy 접근을 시뮬레이션 사이클과 분리하기 위한 사이클 오프셋.
   * cudaMemcpy가 L2를 통과할 때 이 값이 L2 태그 접근 사이클에 더해진다.
   * 이렇게 하면 cudaMemcpy 오버헤드가 커널 실행 사이클 통계에 포함되지 않는다.
   * 설정자: force_l2_tag_update()에서 cudaMemcpy 접근마다 1 증가.
   * 읽는 자: force_l2_tag_update() — m_memcpy_cycle_offset + time을 L2 태그 접근 사이클로 사용.
   * 값 범위: 0 이상의 unsigned; cudaMemcpy 횟수에 비례.
   * 동기화: 단일 스레드. */
};

/*
 * [한국어]
 * L2interface - l2_cache가 L2→DRAM 큐에 요청을 push하는 인터페이스 어댑터
 *
 * mem_fetch_interface(gpu-cache.h 정의)를 구현하여 l2_cache 내부에서 미스가 발생했을 때
 * 이 인터페이스의 push()를 호출하면 memory_sub_partition의 m_L2_dram_queue에 추가된다.
 * 이를 통해 l2_cache는 memory_sub_partition의 내부 구조를 직접 알지 않아도
 * 표준 인터페이스로 DRAM 요청을 발행할 수 있다 (인터페이스 추상화).
 */
class L2interface : public mem_fetch_interface {
 public:
  /*
   * [한국어]
   * L2interface 생성자
   *
   * @unit: 이 인터페이스가 연결된 memory_sub_partition
   */
  L2interface(memory_sub_partition *unit) { m_unit = unit; }
  /* [한국어] m_unit에 서브 파티션 포인터 저장 — push() 시 m_unit->m_L2_dram_queue에 접근 */

  virtual ~L2interface() {}

  /*
   * [한국어]
   * full - L2→DRAM 큐가 가득 찼는지 확인
   *
   * @size: 요청 크기 (바이트, 현재 read/write 동일 크기로 가정)
   * @write: true=쓰기 요청 (현재 read/write 구분 없이 큐 full 여부만 확인)
   * @return: m_L2_dram_queue가 가득 찼으면 true
   *
   * l2_cache가 DRAM 요청 발행 전에 공간 확인 용도로 호출.
   */
  virtual bool full(unsigned size, bool write) const {
    // assume read and write packets all same size
    return m_unit->m_L2_dram_queue->full();
    /* [한국어] L2→DRAM 큐의 full 여부 반환 — 크기/쓰기 여부 구분 없이 큐 상태만 확인 */
  }

  /*
   * [한국어]
   * push - 미스된 mem_fetch를 L2→DRAM 큐에 추가
   *
   * @mf: L2 미스가 발생한 메모리 요청 패킷
   *
   * mf의 상태를 IN_PARTITION_L2_TO_DRAM_QUEUE로 변경하고
   * m_unit->m_L2_dram_queue에 push한다.
   */
  virtual void push(mem_fetch *mf) {
    mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE, 0 /*FIXME*/);
    /* [한국어] mem_fetch 상태를 "L2→DRAM 큐 대기 중"으로 설정
     * 두 번째 인자(0)는 FIXME — 원래 사이클 번호를 전달해야 하나 현재 미구현 */
    m_unit->m_L2_dram_queue->push(mf);
    /* [한국어] friend class 접근으로 서브 파티션의 m_L2_dram_queue에 직접 push */
  }

 private:
  memory_sub_partition *m_unit;
  /* [한국어] 이 인터페이스가 연결된 memory_sub_partition 포인터.
   * 설정자: 생성자에서 1회 초기화, 이후 읽기 전용.
   * 읽는 자: full() — m_L2_dram_queue full 확인; push() — m_L2_dram_queue에 push.
   * 동기화: 읽기 전용. */
};

#endif
