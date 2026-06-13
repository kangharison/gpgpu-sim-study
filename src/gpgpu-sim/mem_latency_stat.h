// Copyright (c) 2009-2011, Tor M. Aamodt
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
 * [한국어 설명] 메모리 레이턴시 통계 헤더 (mem_latency_stat.h)
 *
 * === 파일의 역할 ===
 * GPU 시뮬레이션 중 메모리 서브시스템의 레이턴시와 접근 패턴을 수집하는
 * memory_stats_t 클래스를 선언하는 헤더 파일이다.
 * mem_fetch(메모리 요청 패킷)가 SM 발행 → ICNT → L2 캐시 → DRAM 경로를 따라 이동할 때,
 * 각 경계에서 타임스탬프를 기록하고 레이턴시 히스토그램·최댓값·평균값을 계산한다.
 * 또한 DRAM 뱅크별 읽기/쓰기 접근 수, 행(row) 지역성 통계,
 * AerialVision 시각화용 L2 캐시 히트/미스 통계도 포함한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 메모리 계층 통계 수집 경로:
 *   SM(shader core) → ICNT(intersim2) → L2/메모리 파티션 → DRAM
 *   각 단계에서 memory_stats_t의 memlatstat_* 메서드가 호출됨
 * 호출 관계:
 *   mem_fetch 완료 시 → memlatstat_done()
 *   L2 캐시 읽기 완료 시 → memlatstat_read_done()
 *   DRAM 접근 시 → memlatstat_dram_access()
 *   ICNT→MEM 팝 시 → memlatstat_icnt2mem_pop()
 *   매 샘플링 창 → memlatstat_lat_pw()
 * 실행 컨텍스트: 호스트 유저스페이스, gpgpu_sim::cycle() 내부에서 호출
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - mem_fetch.h: mem_fetch — 레이턴시 측정 대상 메모리 요청 객체
 *   - shader.h: shader_core_config — SM 개수 등 코어 구성
 *   - gpu-sim.h: gpgpu_sim — 전역 시뮬레이터 상태 (현재 사이클 조회 등)
 *   - gpu-cache.h: 캐시 접근 관련 호출
 * 이 파일에 의존하는 모듈:
 *   - l2cache.cc: memory_partition_unit/memory_sub_partition이 memlatstat_* 호출
 *   - dram.cc: dram_t가 memlatstat_dram_access() 호출
 *   - mem_fetch.cc: mem_fetch 완료 처리 시 memlatstat_done() 호출
 *   - power_stat.h: power_mem_stat_t가 memory_stats_t를 참조
 * 데이터 흐름:
 *   mem_fetch 생성 시 issue_time 기록 → 각 단계에서 time delta 계산 → 히스토그램 누적
 *
 * === 주요 함수/구조체 요약 ===
 * memory_stats_t 생성자 - n_shader * m_n_mem * gpu_mem_n_bk 크기의 3D 통계 배열 할당
 * memlatstat_done()     - mem_fetch가 최종 완료될 때 전체 레이턴시(mf_latency) 기록
 * memlatstat_read_done()- L2→SM 읽기 완료 시 mf_latency를 히스토그램에 기록
 * memlatstat_dram_access() - DRAM 접근 시 mrq 레이턴시 및 뱅크 접근 통계 갱신
 * memlatstat_icnt2mem_pop()- ICNT→MEM 팝 시 NoC 레이턴시 기록
 * memlatstat_lat_pw()   - 샘플링 창(Per Window) 레이턴시 통계 갱신
 * memlatstat_print()    - 전체 레이턴시 히스토그램 및 뱅크 통계 출력
 */

#ifndef MEM_LATENCY_STAT_H
#define MEM_LATENCY_STAT_H

#include <stdio.h>  /* [한국어] fprintf, FILE — 통계 출력용 표준 I/O */
#include <zlib.h>   /* [한국어] gzFile — AerialVision 압축 출력용 */
#include <map>      /* [한국어] std::map — 타입별 접근 통계에 사용 */

class memory_config;  /* [한국어] 전방 선언: 메모리 서브시스템 구성 (m_n_mem, m_n_bk 등) */

/*
 * [한국어]
 * memory_stats_t - GPU 메모리 서브시스템의 레이턴시 및 접근 통계 수집 클래스
 *
 * SM에서 발행된 mem_fetch가 ICNT, L2 캐시, DRAM을 통과하는 동안의 레이턴시를
 * 히스토그램 형태로 수집한다. 아울러 DRAM 뱅크별 읽기/쓰기 횟수, 행(row) 지역성,
 * L2 히트/미스 통계를 기록하며, AerialVision 시각화 및 결과 보고서 출력에 사용된다.
 */
class memory_stats_t {
 public:
  /*
   * [한국어]
   * memory_stats_t 생성자 - 레이턴시 통계에 필요한 모든 동적 배열 할당 및 초기화
   *
   * @n_shader: SM(Streaming Multiprocessor) 개수 — bankwrites/bankreads 1차원
   * @shader_config: shader_core_config — SM 구성 (num_shader 등)
   * @mem_config: memory_config — DRAM 채널 수(m_n_mem), 뱅크 수(m_n_bk) 등
   * @gpu: gpgpu_sim — 시뮬레이터 전역 상태 (현재 사이클 번호 조회 등)
   * @return: 없음 (생성자)
   *
   * 다음 다차원 배열들을 calloc으로 할당 (0 초기화):
   *   - bankreads[n_shader][m_n_mem][m_n_bk]: SM별 DRAM 뱅크 읽기 횟수
   *   - bankwrites[n_shader][m_n_mem][m_n_bk]: SM별 DRAM 뱅크 쓰기 횟수
   *   - totalbankreads/writes/accesses[m_n_mem][m_n_bk]: 전체 누적 뱅크 접근
   *   - mf_total_lat_table[m_n_mem][m_n_bk]: 뱅크별 총 mem_fetch 레이턴시
   *   - mf_max_lat_table[m_n_mem][m_n_bk]: 뱅크별 최대 mem_fetch 레이턴시
   *   - concurrent_row_access/num_activates/row_access 등 행 지역성 배열
   *   - num_MCBs_accessed[n_shader]: warp 캐시미스 시 접근한 메모리 컨트롤러 수
   *   - position_of_mrq_chosen[m_n_mem * m_n_bk]: MRQ 선택 위치 히스토그램
   *   - mem_access_type_stats[n_shader][m_n_mem][m_n_bk]: 접근 타입별 분류
   *
   * 호출 체인:
   *   gpgpu_sim 생성자 → new memory_stats_t()
   */
  memory_stats_t(unsigned n_shader,
                 const class shader_core_config *shader_config,
                 const memory_config *mem_config, const class gpgpu_sim *gpu);

  /*
   * [한국어]
   * memlatstat_done - mem_fetch 최종 완료 시 전체 레이턴시 통계 갱신
   *
   * @mf: 완료된 메모리 요청 패킷 (issue_time 포함)
   * @return: 해당 mem_fetch의 레이턴시 사이클 수 (현재 사이클 - mf->get_issue_cycle())
   *
   * L2 히트 또는 DRAM 응답이 SM에 도달하면 호출된다.
   * 계산된 레이턴시를 mf_latency에 저장하고:
   *   - mf_lat_table[32]: 로그₂ 히스토그램 버킷에 분류 (bucket=floor(log2(latency)))
   *   - max_mf_latency: 최댓값 갱신
   *   - num_mfs: 완료된 요청 수 증가
   *   - mf_total_lat: 누적 레이턴시 합산
   *
   * 호출 체인:
   *   memory_sub_partition::cache_cycle() (L2 히트 응답) → [memlatstat_done()]
   */
  unsigned memlatstat_done(class mem_fetch *mf);

  /*
   * [한국어]
   * memlatstat_read_done - L2→SM 읽기 완료 시 mf_latency를 per-window 히스토그램에 기록
   *
   * @mf: 완료된 읽기 요청 패킷
   * @return: 없음 (void)
   *
   * memlatstat_done() 호출로 mf->get_mf_type()이 READ인 경우 추가로 호출된다.
   * mf_lat_pw_table[32] (샘플링 창별 히스토그램)과 mf_num_lat_pw(창별 요청 수),
   * mf_tot_lat_pw(창별 레이턴시 합)를 업데이트한다.
   * AerialVision 시각화 데이터 생성에 사용된다.
   *
   * 호출 체인:
   *   memory_sub_partition::cache_cycle() → memlatstat_done() → [memlatstat_read_done()]
   */
  void memlatstat_read_done(class mem_fetch *mf);

  /*
   * [한국어]
   * memlatstat_dram_access - DRAM 접근 시 mrq 레이턴시 및 뱅크 접근 통계 갱신
   *
   * @mf: DRAM에 접근하는 메모리 요청 패킷
   * @return: 없음 (void)
   *
   * mem_fetch가 L2 미스 후 DRAM 접근이 발생할 때 호출된다.
   * 다음 통계를 갱신한다:
   *   - mrq_lat_table[32]: MRQ(Memory Request Queue) 레이턴시 히스토그램
   *   - max_mrq_latency, tot_mrq_latency, tot_mrq_num
   *   - bankreads/bankwrites[sm_id][chip_id][bank_id]: SM별 DRAM 뱅크 접근 카운터
   *   - totalbankreads/bankwrites/bankaccesses[chip_id][bank_id]: 전체 누적
   *   - concurrent_row_access, row_access 등 행 지역성 통계
   *
   * 호출 체인:
   *   dram_t::issue_col() → [memlatstat_dram_access()]
   */
  void memlatstat_dram_access(class mem_fetch *mf);

  /*
   * [한국어]
   * memlatstat_icnt2mem_pop - ICNT→MEM 팝 시 NoC 레이턴시 기록
   *
   * @mf: ICNT 출구에서 꺼내진 메모리 요청 패킷
   * @return: 없음 (void)
   *
   * intersim2(NoC)에서 메모리 파티션 측으로 패킷이 도달할 때 호출된다.
   * ICNT를 통과하는 데 걸린 레이턴시(현재 사이클 - mf 발행 사이클)를
   * icnt2mem_lat_table[24]에 기록하고 max_icnt2mem_latency, tot_icnt2mem_latency를 갱신한다.
   *
   * 호출 체인:
   *   memory_partition_unit::cache_cycle() (ICNT pop) → [memlatstat_icnt2mem_pop()]
   */
  void memlatstat_icnt2mem_pop(class mem_fetch *mf);

  /*
   * [한국어]
   * memlatstat_lat_pw - 샘플링 창(Per Window) 레이턴시 통계 갱신 및 초기화
   *
   * @return: 없음 (void)
   *
   * 매 stat_sample_freq 사이클마다 호출되어 이번 창의 평균 레이턴시를 계산하고
   * mf_lat_pw_table, mf_num_lat_pw, mf_tot_lat_pw를 다음 창을 위해 0으로 리셋한다.
   * AerialVision 시각화에서 시계열 레이턴시 그래프 데이터를 제공한다.
   *
   * 호출 체인:
   *   gpgpu_sim::cycle() (stat 샘플링 지점) → [memlatstat_lat_pw()]
   */
  void memlatstat_lat_pw();

  /*
   * [한국어]
   * memlatstat_print - 전체 메모리 레이턴시 통계 및 뱅크 접근 통계 출력
   *
   * @n_mem: DRAM 채널(메모리 파티션) 수
   * @gpu_mem_n_bk: DRAM 채널당 뱅크 수
   * @return: 없음 (void)
   *
   * 시뮬레이션 종료 후 호출되어 다음 항목을 stdout에 출력한다:
   *   - mf_lat_table, mrq_lat_table, dq_lat_table: 레이턴시 히스토그램
   *   - icnt2mem_lat_table, icnt2sh_lat_table: NoC 레이턴시 히스토그램
   *   - mf_total_lat_table, mf_max_lat_table: 뱅크별 총/최대 레이턴시
   *   - bankreads/bankwrites: SM별 뱅크 접근 행렬
   *   - totalbankaccesses: 뱅크별 총 접근 수
   * -DSSST 컴파일 옵션 시 SST 포맷으로 출력 분기가 있다.
   *
   * 호출 체인:
   *   gpgpu_sim::print_stats() → [memlatstat_print()]
   */
  void memlatstat_print(unsigned n_mem, unsigned gpu_mem_n_bk);

  /*
   * [한국어]
   * visualizer_print - AerialVision 가시화 파일에 L2 및 레이턴시 통계 출력
   *
   * @visualizer_file: gzFile — AerialVision 압축 출력 파일
   * @return: 없음 (void)
   *
   * 현재 샘플링 창의 L2 읽기/쓰기 히트·미스 수와 mf 레이턴시 per-window 값을
   * gzprintf()로 기록한다. AerialVision이 이 파일을 읽어 시계열 그래프를 생성한다.
   *
   * 호출 체인:
   *   gpgpu_sim::visualizer_printout() → [memory_stats_t::visualizer_print()]
   */
  void visualizer_print(gzFile visualizer_file);

  // Reset local L2 stats that are aggregated each sampling window
  /*
   * [한국어]
   * clear_L2_stats_pw - 샘플링 창별 L2 통계 카운터 초기화
   *
   * @return: 없음 (void)
   *
   * 매 샘플링 창 시작 시 호출되어 L2_read_miss, L2_write_miss,
   * L2_read_hit, L2_write_hit를 0으로 리셋한다.
   * AerialVision 시각화 데이터가 창별 증분이 되도록 한다.
   *
   * 호출 체인:
   *   l2cache.cc::memory_stats_t::clear_L2_stats_pw() 구현 → 샘플링 루프에서 호출
   */
  void clear_L2_stats_pw();

  unsigned m_n_shader;
  /* [한국어] SM(Streaming Multiprocessor) 개수.
   * 설정자: 생성자에서 n_shader 인자로 초기화.
   * 읽는 자: bankreads/bankwrites 배열 1차원 크기, memlatstat_print() 출력 루프.
   * 값 범위: gpgpusim.config의 -gpgpu_n_shader 설정값 (일반적으로 2~80).
   * 동기화: 읽기 전용 (생성 후 변경 없음). */

  const shader_core_config *m_shader_config;
  /* [한국어] SM 구성 참조 포인터.
   * 설정자: 생성자 인자로 전달, 이후 변경 없음.
   * 읽는 자: 통계 출력 시 SM 수(num_shader()) 조회.
   * 값 범위: 유효한 shader_core_config 포인터 (NULL 불가).
   * 동기화: 읽기 전용. */

  const memory_config *m_memory_config;
  /* [한국어] 메모리 서브시스템 구성 참조 포인터.
   * 설정자: 생성자 인자로 전달.
   * 읽는 자: m_n_mem(DRAM 채널 수), m_n_bk(뱅크 수) 조회.
   * 값 범위: 유효한 memory_config 포인터 (NULL 불가).
   * 동기화: 읽기 전용. */

  const class gpgpu_sim *m_gpu;
  /* [한국어] 시뮬레이터 전역 상태 참조.
   * 설정자: 생성자 인자로 전달.
   * 읽는 자: 현재 사이클 번호(gpu_sim_cycle) 조회 — 레이턴시 계산에 사용.
   * 값 범위: 유효한 gpgpu_sim 포인터 (NULL 불가).
   * 동기화: 읽기 전용. */

  unsigned max_mrq_latency;
  /* [한국어] 지금까지 관측된 최대 MRQ(Memory Request Queue) 레이턴시 (사이클 수).
   * 설정자: memlatstat_dram_access()에서 현재값이 이 값을 넘으면 갱신.
   * 읽는 자: memlatstat_print()에서 출력.
   * 값 범위: 0 이상의 사이클 수 (일반적으로 수백~수천 사이클).
   * 동기화: 단일 호스트 스레드에서 시뮬레이션 순차 실행. */

  unsigned max_dq_latency;
  /* [한국어] 지금까지 관측된 최대 dq(DRAM 큐) 레이턴시 (사이클 수).
   * 설정자: memlatstat_dram_access()에서 갱신.
   * 읽는 자: memlatstat_print().
   * 값 범위: 0 이상의 사이클 수.
   * 동기화: 단일 스레드 순차 실행. */

  unsigned max_mf_latency;
  /* [한국어] 지금까지 관측된 최대 mem_fetch 전체 레이턴시 (사이클 수).
   * 설정자: memlatstat_done()에서 갱신.
   * 읽는 자: memlatstat_print().
   * 값 범위: 0 이상의 사이클 수.
   * 동기화: 단일 스레드 순차 실행. */

  unsigned max_icnt2mem_latency;
  /* [한국어] ICNT→MEM 방향의 최대 NoC 레이턴시 (사이클 수).
   * 설정자: memlatstat_icnt2mem_pop()에서 갱신.
   * 읽는 자: memlatstat_print().
   * 값 범위: 0 이상의 사이클 수.
   * 동기화: 단일 스레드 순차 실행. */

  unsigned long long int tot_icnt2mem_latency;
  /* [한국어] ICNT→MEM 레이턴시의 전체 누적 합 (사이클 수).
   * 설정자: memlatstat_icnt2mem_pop()에서 각 mem_fetch의 레이턴시를 더함.
   * 읽는 자: memlatstat_print()에서 평균 계산 (tot/num_mfs).
   * 값 범위: unsigned long long — 긴 시뮬레이션에서도 오버플로 방지.
   * 동기화: 단일 스레드 순차 실행. */

  unsigned long long int tot_icnt2sh_latency;
  /* [한국어] MEM→SIMT(SM) 방향 NoC 레이턴시 누적 합.
   * 설정자: 메모리→SM 응답 경로에서 갱신.
   * 읽는 자: memlatstat_print().
   * 값 범위: unsigned long long.
   * 동기화: 단일 스레드. */

  unsigned long long int tot_mrq_latency;
  /* [한국어] MRQ(Memory Request Queue) 대기 레이턴시 누적 합.
   * 설정자: memlatstat_dram_access()에서 갱신.
   * 읽는 자: memlatstat_print() — 평균 MRQ 레이턴시 계산.
   * 값 범위: unsigned long long.
   * 동기화: 단일 스레드. */

  unsigned long long int tot_mrq_num;
  /* [한국어] 지금까지 DRAM에 접근한 MRQ(메모리 요청) 총 수.
   * 설정자: memlatstat_dram_access()에서 매 접근마다 1 증가.
   * 읽는 자: memlatstat_print() — 평균 레이턴시 계산 시 분모.
   * 값 범위: unsigned long long.
   * 동기화: 단일 스레드. */

  unsigned max_icnt2sh_latency;
  /* [한국어] MEM→SM 방향 최대 NoC 레이턴시 (사이클 수).
   * 설정자: 응답 패킷 SM 도착 시 갱신.
   * 읽는 자: memlatstat_print().
   * 값 범위: 0 이상의 사이클 수.
   * 동기화: 단일 스레드. */

  unsigned mrq_lat_table[32];
  /* [한국어] MRQ 레이턴시 로그₂ 히스토그램 (32개 버킷).
   * 설정자: memlatstat_dram_access()에서 bucket=floor(log2(latency))로 인덱싱하여 증가.
   * 읽는 자: memlatstat_print()에서 전체 히스토그램 출력.
   * 값 범위: 각 버킷은 unsigned 카운터; mrq_lat_table[k]는 2^k~2^(k+1)-1 사이클 범위의 요청 수.
   * 동기화: 단일 스레드. */

  unsigned dq_lat_table[32];
  /* [한국어] DRAM 큐(dq) 레이턴시 로그₂ 히스토그램 (32개 버킷).
   * 설정자: DRAM 접근 완료 시 갱신.
   * 읽는 자: memlatstat_print().
   * 값 범위: 버킷 인덱스 = floor(log2(레이턴시)).
   * 동기화: 단일 스레드. */

  unsigned mf_lat_table[32];
  /* [한국어] mem_fetch 전체 레이턴시 로그₂ 히스토그램 (32개 버킷).
   * 설정자: memlatstat_done()에서 완료된 mem_fetch마다 갱신.
   * 읽는 자: memlatstat_print().
   * 값 범위: 버킷 k = floor(log2(전체 레이턴시)).
   * 동기화: 단일 스레드. */

  unsigned icnt2mem_lat_table[24];
  /* [한국어] ICNT→MEM 방향 NoC 레이턴시 히스토그램 (24개 버킷).
   * 버킷 수가 32가 아닌 24인 이유: NoC 레이턴시는 일반적으로 더 작은 범위에 집중.
   * 설정자: memlatstat_icnt2mem_pop()에서 갱신.
   * 읽는 자: memlatstat_print().
   * 동기화: 단일 스레드. */

  unsigned icnt2sh_lat_table[24];
  /* [한국어] MEM→SM(icnt2sh) 방향 NoC 레이턴시 히스토그램 (24개 버킷).
   * 설정자: 응답 패킷 SM 도착 시 갱신.
   * 읽는 자: memlatstat_print().
   * 동기화: 단일 스레드. */

  unsigned mf_lat_pw_table[32];  // table storing values of mf latency Per
                                 // Window
  /* [한국어] 샘플링 창별(Per Window) mem_fetch 레이턴시 히스토그램.
   * 설정자: memlatstat_read_done()에서 각 읽기 완료 시 현재 창의 버킷 증가.
   * 읽는 자: visualizer_print()에서 AerialVision 출력; memlatstat_lat_pw()에서 리셋.
   * 값 범위: 창 단위 카운터 — memlatstat_lat_pw()가 호출될 때마다 0으로 초기화됨.
   * 동기화: 단일 스레드. */

  unsigned mf_num_lat_pw;
  /* [한국어] 현재 샘플링 창에서 완료된 읽기 mem_fetch의 수.
   * 설정자: memlatstat_read_done()에서 1 증가; memlatstat_lat_pw()에서 0 리셋.
   * 읽는 자: memlatstat_lat_pw()에서 평균 레이턴시 계산 시 분모.
   * 동기화: 단일 스레드. */

  unsigned max_warps;
  /* [한국어] 동시에 접근한 최대 warp 수 통계용 필드.
   * 설정자: 뱅크 접근 통계 수집 시 갱신될 수 있음.
   * 읽는 자: 통계 출력.
   * 동기화: 단일 스레드. */

  unsigned mf_tot_lat_pw;  // total latency summed up per window. divide by
                           // mf_num_lat_pw to obtain average latency Per Window
  /* [한국어] 현재 샘플링 창의 총 mem_fetch 레이턴시 합 (사이클).
   * 설정자: memlatstat_read_done()에서 각 읽기 완료 시 레이턴시를 더함; memlatstat_lat_pw()에서 리셋.
   * 읽는 자: visualizer_print(), memlatstat_lat_pw() — mf_tot_lat_pw / mf_num_lat_pw = 창 평균.
   * 동기화: 단일 스레드. */

  unsigned long long int mf_total_lat;
  /* [한국어] 시뮬레이션 전체 누적 mem_fetch 레이턴시 합 (사이클).
   * 설정자: memlatstat_done()에서 각 완료 mem_fetch의 레이턴시를 더함.
   * 읽는 자: memlatstat_print()에서 전체 평균 레이턴시 = mf_total_lat / num_mfs.
   * 동기화: 단일 스레드. */

  unsigned long long int *
      *mf_total_lat_table;      // mf latency sums[dram chip id][bank id]
  /* [한국어] DRAM 뱅크별 총 mem_fetch 레이턴시 합 2D 배열 [chip_id][bank_id].
   * 설정자: memlatstat_dram_access()에서 해당 뱅크에 레이턴시 누적.
   * 읽는 자: memlatstat_print()에서 뱅크별 평균 레이턴시 계산.
   * 값 범위: [m_n_mem][m_n_bk] 크기; calloc으로 0 초기화.
   * 동기화: 단일 스레드. */

  unsigned **mf_max_lat_table;  // mf latency sums[dram chip id][bank id]
  /* [한국어] DRAM 뱅크별 최대 mem_fetch 레이턴시 [chip_id][bank_id].
   * 설정자: memlatstat_dram_access()에서 현재 레이턴시가 이 값을 초과하면 갱신.
   * 읽는 자: memlatstat_print().
   * 값 범위: [m_n_mem][m_n_bk].
   * 동기화: 단일 스레드. */

  unsigned num_mfs;
  /* [한국어] 시뮬레이션 전체에서 완료된 mem_fetch 수.
   * 설정자: memlatstat_done()에서 매 완료마다 1 증가.
   * 읽는 자: memlatstat_print() — 평균 레이턴시 계산 분모.
   * 동기화: 단일 스레드. */

  unsigned int ***bankwrites;  // bankwrites[shader id][dram chip id][bank id]
  /* [한국어] SM별 DRAM 뱅크 쓰기 횟수 3D 배열 [sm_id][chip_id][bank_id].
   * 설정자: memlatstat_dram_access()에서 쓰기 요청 시 ++bankwrites[sm_id][chip_id][bank_id].
   * 읽는 자: memlatstat_print()에서 SM별 쓰기 패턴 출력.
   * 값 범위: [n_shader][m_n_mem][m_n_bk]; calloc으로 0 초기화.
   * 동기화: 단일 스레드. */

  unsigned int ***bankreads;   // bankreads[shader id][dram chip id][bank id]
  /* [한국어] SM별 DRAM 뱅크 읽기 횟수 3D 배열 [sm_id][chip_id][bank_id].
   * 설정자: memlatstat_dram_access()에서 읽기 요청 시 ++bankreads[sm_id][chip_id][bank_id].
   * 읽는 자: memlatstat_print()에서 SM별 읽기 패턴 출력.
   * 값 범위: [n_shader][m_n_mem][m_n_bk].
   * 동기화: 단일 스레드. */

  unsigned int **totalbankwrites;    // bankwrites[dram chip id][bank id]
  /* [한국어] 전체 SM 합산 DRAM 뱅크 쓰기 횟수 [chip_id][bank_id].
   * 설정자: memlatstat_dram_access()에서 bankwrites와 동시에 갱신.
   * 읽는 자: memlatstat_print()에서 뱅크별 총 쓰기 출력.
   * 값 범위: [m_n_mem][m_n_bk].
   * 동기화: 단일 스레드. */

  unsigned int **totalbankreads;     // bankreads[dram chip id][bank id]
  /* [한국어] 전체 SM 합산 DRAM 뱅크 읽기 횟수 [chip_id][bank_id].
   * 설정자: memlatstat_dram_access()에서 갱신.
   * 읽는 자: memlatstat_print().
   * 동기화: 단일 스레드. */

  unsigned int **totalbankaccesses;  // bankaccesses[dram chip id][bank id]
  /* [한국어] 전체 DRAM 뱅크 접근 횟수 (읽기+쓰기) [chip_id][bank_id].
   * 설정자: memlatstat_dram_access()에서 매 접근마다 갱신.
   * 읽는 자: memlatstat_print() — 뱅크 불균형(bank imbalance) 분석에 활용.
   * 값 범위: [m_n_mem][m_n_bk].
   * 동기화: 단일 스레드. */

  unsigned int
      *num_MCBs_accessed;  // tracks how many memory controllers are accessed
                           // whenever any thread in a warp misses in cache
  /* [한국어] warp가 캐시 미스 시 접근한 메모리 컨트롤러(MCB) 수 히스토그램 [n_shader].
   * 설정자: 캐시 미스 시 warp가 몇 개의 메모리 파티션에 요청을 보내는지 기록.
   * 읽는 자: memlatstat_print() — 메모리 컨트롤러 분산 패턴 분석.
   * 값 범위: [n_shader] 크기; num_MCBs_accessed[i]는 동시에 i개의 MCB를 접근한 warp 수.
   * 동기화: 단일 스레드. */

  unsigned int *position_of_mrq_chosen;  // position of mrq in m_queue chosen
  /* [한국어] FRFCFS 스케줄러가 선택한 MRQ의 큐 내 위치 히스토그램.
   * FRFCFS(First-Ready First-Come-First-Served): 행 히트 우선 + FCFS 조합 DRAM 스케줄러.
   * 설정자: dram 스케줄러가 MRQ를 선택할 때 큐에서의 위치를 기록.
   * 읽는 자: memlatstat_print() — 스케줄러 효율성(큐 앞/뒤 선택 비율) 분석.
   * 값 범위: [m_n_mem * m_n_bk] 크기 히스토그램.
   * 동기화: 단일 스레드. */

  unsigned ***mem_access_type_stats;  // dram access type classification
  /* [한국어] DRAM 접근 타입별 분류 3D 배열 [sm_id][chip_id][bank_id].
   * 설정자: memlatstat_dram_access()에서 mem_fetch 타입(읽기/쓰기/텍스처/상수 등)에 따라 갱신.
   * 읽는 자: memlatstat_print() — 접근 타입 분포 출력.
   * 값 범위: [n_shader][m_n_mem][m_n_bk].
   * 동기화: 단일 스레드. */

  // AerialVision L2 stats
  unsigned L2_read_miss;
  /* [한국어] 현재 샘플링 창에서 L2 캐시 읽기 미스 횟수.
   * 설정자: L2 미스 발생 시 memory_sub_partition::cache_cycle()에서 증가;
   *         clear_L2_stats_pw()에서 0으로 리셋.
   * 읽는 자: visualizer_print()에서 AerialVision 출력.
   * 값 범위: 0 이상의 unsigned; 샘플링 창마다 리셋됨.
   * 동기화: 단일 스레드. */

  unsigned L2_write_miss;
  /* [한국어] 현재 샘플링 창에서 L2 캐시 쓰기 미스 횟수.
   * 설정자: L2 쓰기 미스 시 증가; clear_L2_stats_pw()에서 리셋.
   * 읽는 자: visualizer_print().
   * 동기화: 단일 스레드. */

  unsigned L2_read_hit;
  /* [한국어] 현재 샘플링 창에서 L2 캐시 읽기 히트 횟수.
   * 설정자: L2 읽기 히트 시 증가; clear_L2_stats_pw()에서 리셋.
   * 읽는 자: visualizer_print().
   * 동기화: 단일 스레드. */

  unsigned L2_write_hit;
  /* [한국어] 현재 샘플링 창에서 L2 캐시 쓰기 히트 횟수.
   * 설정자: L2 쓰기 히트 시 증가; clear_L2_stats_pw()에서 리셋.
   * 읽는 자: visualizer_print().
   * 동기화: 단일 스레드. */

  // L2 cache stats
  unsigned int *L2_cbtoL2length;
  /* [한국어] CB(Crossbar)→L2 방향 데이터 길이(바이트) 히스토그램 배열.
   * 설정자: SM→L2 요청이 ICNT를 통해 도착할 때 요청 크기 기록.
   * 읽는 자: 통계 출력.
   * 값 범위: 가변 크기 배열; 크기는 생성자에서 결정.
   * 동기화: 단일 스레드. */

  unsigned int *L2_cbtoL2writelength;
  /* [한국어] CB→L2 쓰기 방향 데이터 길이 히스토그램.
   * 설정자: L2 쓰기 요청 도착 시 크기 기록.
   * 읽는 자: 통계 출력.
   * 동기화: 단일 스레드. */

  unsigned int *L2_L2tocblength;
  /* [한국어] L2→CB(Crossbar, 즉 ICNT) 방향 응답 데이터 길이 히스토그램.
   * 설정자: L2→SM 응답 패킷 크기 기록.
   * 읽는 자: 통계 출력.
   * 동기화: 단일 스레드. */

  unsigned int *L2_dramtoL2length;
  /* [한국어] DRAM→L2 방향 응답 데이터 길이 히스토그램.
   * 설정자: DRAM이 L2 미스 요청에 응답할 때 데이터 크기 기록.
   * 읽는 자: 통계 출력.
   * 동기화: 단일 스레드. */

  unsigned int *L2_dramtoL2writelength;
  /* [한국어] DRAM→L2 쓰기(Write-back) 데이터 길이 히스토그램.
   * 설정자: L2 라이트백 시 기록.
   * 읽는 자: 통계 출력.
   * 동기화: 단일 스레드. */

  unsigned int *L2_L2todramlength;
  /* [한국어] L2→DRAM 방향 라이트백 데이터 길이 히스토그램.
   * 설정자: L2가 다티 라인을 DRAM으로 내보낼 때 크기 기록.
   * 읽는 자: 통계 출력.
   * 동기화: 단일 스레드. */

  // DRAM access row locality stats
  unsigned int *
      *concurrent_row_access;    // concurrent_row_access[dram chip id][bank id]
  /* [한국어] 동일 DRAM 행(row)에 대한 동시 접근 수 통계 [chip_id][bank_id].
   * 설정자: memlatstat_dram_access()에서 현재 열린 행과 동일한 행에 접근할 때 증가.
   * 읽는 자: memlatstat_print() — 행 지역성(row locality) 분석.
   * 값 범위: [m_n_mem][m_n_bk].
   * 동기화: 단일 스레드. */

  unsigned int **num_activates;  // num_activates[dram chip id][bank id]
  /* [한국어] DRAM 뱅크별 row activate 명령 수 [chip_id][bank_id].
   * 설정자: DRAM 뱅크가 새 행을 열 때(activate) 증가.
   * 읽는 자: memlatstat_print() — activate 빈도로 행 교체율 분석.
   * 동기화: 단일 스레드. */

  unsigned int **row_access;     // row_access[dram chip id][bank id]
  /* [한국어] 현재 열린 행에 대한 누적 접근 수 [chip_id][bank_id].
   * 설정자: memlatstat_dram_access()에서 행 히트 시 증가.
   * 읽는 자: memlatstat_print().
   * 동기화: 단일 스레드. */

  unsigned int **max_conc_access2samerow;  // max_conc_access2samerow[dram chip
                                           // id][bank id]
  /* [한국어] 단일 행에 대한 최대 연속 접근 수 [chip_id][bank_id].
   * 설정자: 연속 행 접근이 이전 최댓값을 넘으면 갱신.
   * 읽는 자: memlatstat_print() — FRFCFS 스케줄러의 행 히트 연속성 분석.
   * 동기화: 단일 스레드. */

  unsigned int **max_servicetime2samerow;  // max_servicetime2samerow[dram chip
                                           // id][bank id]
  /* [한국어] 단일 행에 대한 최대 서비스 시간 (사이클) [chip_id][bank_id].
   * 설정자: 행이 닫힐 때(precharge) 해당 행의 총 서비스 시간이 최댓값을 넘으면 갱신.
   * 읽는 자: memlatstat_print().
   * 동기화: 단일 스레드. */

  // Power stats
  unsigned total_n_access;
  /* [한국어] 시뮬레이션 전체 메모리 접근 수 (읽기+쓰기 합산).
   * 설정자: memlatstat_dram_access()에서 매 접근마다 증가.
   * 읽는 자: 전력 통계 계산 — AccelWattch가 DRAM 전력 추정에 사용.
   * 동기화: 단일 스레드. */

  unsigned total_n_reads;
  /* [한국어] 시뮬레이션 전체 DRAM 읽기 접근 수.
   * 설정자: memlatstat_dram_access()에서 읽기 요청마다 증가.
   * 읽는 자: 전력/성능 통계 출력.
   * 동기화: 단일 스레드. */

  unsigned total_n_writes;
  /* [한국어] 시뮬레이션 전체 DRAM 쓰기 접근 수.
   * 설정자: memlatstat_dram_access()에서 쓰기 요청마다 증가.
   * 읽는 자: 전력/성능 통계 출력.
   * 동기화: 단일 스레드. */
};

#endif /*MEM_LATENCY_STAT_H*/
