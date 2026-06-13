// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung,
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
 * [한국어 설명] AerialVision 시각화 지원 및 메모리 레이턴시 추적 구현 (visualizer.cc)
 *
 * === 파일의 역할 ===
 * 두 가지 주요 기능을 구현한다:
 * (1) gpgpu_sim::visualizer_printstat(): 매 시뮬레이션 인터벌마다 AerialVision용
 *     성능 통계를 gzip 압축 파일에 기록한다. CF 지역성, CTA 분포, 메모리 파티션 통계,
 *     SM 통계, 메모리 통계, 전력 통계, 전역 사이클/명령어 카운터를 포함한다.
 * (2) my_time_vector 클래스와 time_vector_* 함수: 각 메모리 요청이 ICNT→DRAM→캐시→SM
 *     파이프라인을 통과하는 타임스탬프를 추적하여 레이턴시 분포를 계산하고 출력한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: gpu-sim.cc::cycle() → visualizer_printstat() → gzip 파일 기록
 *   mem_fetch 파이프라인 통과 시 → time_vector_update() → 타임스탬프 누적
 *   → visualizer_printstat()에서 time_vector_print_interval2gzfile() 호출
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 루프 내.
 *
 * === 타 모듈과의 연결 ===
 * 의존: visualizer.h, gpu-sim.h (gpgpu_sim), shader.h (m_shader_stats),
 *       mem_latency_stat.h (NUM_MEM_REQ_STAT, IN_ICNT_TO_MEM, IN_SHADER_FETCHED),
 *       mem_fetch.h (READ_REQUEST, WRITE_REQUEST 등), stat-tool.h, power_stat.h,
 *       l2cache.h, gpu-cache.h, option_parser.h, zlib.h
 * 데이터 흐름: mem_fetch→uid/type → time_vector_update() → ld/st_time_map 누적
 *   → calculate_ld/st_dist() → ld/st_time_dist → print_to_gzfile() 출력
 *
 * === 주요 함수/구조체 요약 ===
 * gpgpu_sim::visualizer_printstat() - AerialVision gzip 파일에 인터벌 통계 기록
 * my_time_vector::update_ld/st()    - UID별 LD/ST 타임스탬프 기록
 * my_time_vector::calculate_dist()  - 완료된 요청의 레이턴시 분포 계산
 * my_time_vector::print_to_gzfile() - 레이턴시 분포를 AerialVision 파일에 출력
 * time_vector_update()              - 전역 인터페이스 함수 (mem_fetch에서 호출)
 */

#include "visualizer.h"

#include "../option_parser.h" /* [한국어] gpgpusim.config 옵션 (g_visualizer_enabled 등) */
#include "gpu-sim.h"           /* [한국어] gpgpu_sim 클래스 (m_config, m_memory_partition_unit 등) */
#include "l2cache.h"           /* [한국어] L2 캐시 통계 (visualizer_print) */
#include "mem_latency_stat.h"  /* [한국어] NUM_MEM_REQ_STAT, IN_ICNT_TO_MEM, IN_SHADER_FETCHED 등 */
#include "power_stat.h"        /* [한국어] 전력 통계 (visualizer_print) */
#include "shader.h"            /* [한국어] SM 통계 (m_shader_stats->visualizer_print) */
//#include "../../../mcpat/processor.h"
#include "gpu-cache.h"  /* [한국어] 캐시 통계 */
#include "stat-tool.h"  /* [한국어] cflog_visualizer_gzprint, shader_CTA_count_visualizer_gzprint */

#include <string.h> /* [한국어] 문자열 처리 */
#include <time.h>   /* [한국어] 시간 함수 */
#include <zlib.h>   /* [한국어] gzFile, gzopen, gzprintf, gzclose — 압축 파일 I/O */

/* [한국어] time_vector_print_interval2gzfile()는 my_time_vector 클래스 정의 이후에
 * static 함수로 구현됨 (파일 하단 참조). visualizer_printstat()보다 먼저 호출되므로
 * 순방향 선언이 필요하나, 실제 정의는 클래스 아래에 위치. */
static void time_vector_print_interval2gzfile(gzFile outfile);

/*
 * [한국어]
 * gpgpu_sim::visualizer_printstat() - AerialVision 성능 시각화 파일에 인터벌 통계 기록
 *
 * g_visualizer_enabled 설정이 false이면 즉시 반환.
 * 첫 호출 시 "w" 모드로 파일 생성, 이후 "a" 모드로 추가.
 * gzip 압축 레벨은 g_visualizer_zlevel로 설정.
 * 다음 통계를 순서대로 기록:
 *   1. cflog_visualizer_gzprint(): 스레드 CF 지역성 희소 히스토그램
 *   2. shader_CTA_count_visualizer_gzprint(): SM별 CTA 수 분포
 *   3. 각 메모리 파티션의 visualizer_print(): L2/DRAM 통계
 *   4. m_shader_stats/m_memory_stats/m_power_stats의 visualizer_print()
 *   5. globalcyclecount/globalinsncount/globaltotinsncount
 *   6. time_vector_print_interval2gzfile(): LD/ST 레이턴시 분포
 * 실행 컨텍스트: 시뮬레이션 인터벌마다 gpu-sim.cc::cycle()에서 호출.
 *
 * 호출 체인:
 *   gpu-sim.cc::gpgpu_sim::cycle() → [visualizer_printstat()] → gzFile 출력
 */
void gpgpu_sim::visualizer_printstat() {
  gzFile visualizer_file = NULL;  // gzFile is basically a pointer to a struct,
                                  // so it is fine to initialize it as NULL
  /* [한국어] gzFile은 포인터 타입이므로 NULL로 초기화 가능 */
  if (!m_config.g_visualizer_enabled) return;
  /* [한국어] gpgpusim.config의 시각화 활성화 옵션 확인 — 비활성이면 즉시 반환 */

  // clean the content of the visualizer log if it is the first time, otherwise
  // attach at the end
  static bool visualizer_first_printstat = true;
  /* [한국어] 첫 호출 여부 추적 정적 변수. 첫 호출=파일 생성("w"), 이후=추가("a"). */

  visualizer_file = gzopen(m_config.g_visualizer_filename,
                           (visualizer_first_printstat) ? "w" : "a");
  /* [한국어] gzip 압축 파일 열기. 첫 호출이면 "w"(덮어쓰기), 이후 "a"(추가). */
  if (visualizer_file == NULL) {
    printf("error - could not open visualizer trace file.\n");
    exit(1); /* [한국어] 파일 열기 실패 — 시뮬레이터 종료 */
  }
  gzsetparams(visualizer_file, m_config.g_visualizer_zlevel,
              Z_DEFAULT_STRATEGY);
  /* [한국어] gzip 압축 레벨 설정. g_visualizer_zlevel: 0(비압축)~9(최대압축). */
  visualizer_first_printstat = false; /* [한국어] 이후 호출은 "a" 모드 사용 */

  cflog_visualizer_gzprint(visualizer_file);
  /* [한국어] 스레드 제어 흐름 지역성 희소 히스토그램 출력 */
  shader_CTA_count_visualizer_gzprint(visualizer_file);
  /* [한국어] SM별 현재 활성 CTA 수 분포 출력 */

  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    m_memory_partition_unit[i]->visualizer_print(visualizer_file);
  /* [한국어] 각 메모리 파티션(L2/DRAM)의 시각화 통계 출력 */
  m_shader_stats->visualizer_print(visualizer_file);
  /* [한국어] SM(shader core) 성능 통계 출력 */
  m_memory_stats->visualizer_print(visualizer_file);
  /* [한국어] 메모리 계층 통계 출력 */
  m_power_stats->visualizer_print(visualizer_file);
  /* [한국어] 전력 모델 통계 출력 (AccelWattch) */
  // proc->visualizer_print(visualizer_file);
  /* [한국어] McPAT 프로세서 모델 시각화 — 현재 주석 처리 */
  // other parameters for graphing
  gzprintf(visualizer_file, "globalcyclecount: %lld\n", gpu_sim_cycle);
  /* [한국어] 현재 커널의 시뮬레이션 사이클 수 */
  gzprintf(visualizer_file, "globalinsncount: %lld\n", gpu_sim_insn);
  /* [한국어] 현재 커널의 명령어 실행 수 */
  gzprintf(visualizer_file, "globaltotinsncount: %lld\n", gpu_tot_sim_insn);
  /* [한국어] 전체 시뮬레이션 명령어 실행 수 (커널 간 누적) */

  time_vector_print_interval2gzfile(visualizer_file);
  /* [한국어] LD/ST 메모리 레이턴시 분포를 gzip 파일에 출력 */

  gzclose(visualizer_file); /* [한국어] gzip 파일 닫기 (버퍼 플러시 및 파일 완성) */
  /*
     gzprintf(visualizer_file, "CacheMissRate_GlobalLocalL1_All: ");
     for (unsigned i=0;i<m_n_shader;i++)
        gzprintf(visualizer_file, "%0.4f ",
     m_sc[i]->L1_windowed_cache_miss_rate(0)); gzprintf(visualizer_file, "\n");
     gzprintf(visualizer_file, "CacheMissRate_TextureL1_All: ");
     for (unsigned i=0;i<m_n_shader;i++)
        gzprintf(visualizer_file, "%0.4f ",
     m_sc[i]->L1tex_windowed_cache_miss_rate(0)); gzprintf(visualizer_file,
     "\n"); gzprintf(visualizer_file, "CacheMissRate_ConstL1_All: "); for
     (unsigned i=0;i<m_n_shader;i++) gzprintf(visualizer_file, "%0.4f ",
     m_sc[i]->L1const_windowed_cache_miss_rate(0)); gzprintf(visualizer_file,
     "\n"); gzprintf(visualizer_file, "CacheMissRate_GlobalLocalL1_noMgHt: ");
     for (unsigned i=0;i<m_n_shader;i++)
        gzprintf(visualizer_file, "%0.4f ",
     m_sc[i]->L1_windowed_cache_miss_rate(1)); gzprintf(visualizer_file, "\n");
     gzprintf(visualizer_file, "CacheMissRate_TextureL1_noMgHt: ");
     for (unsigned i=0;i<m_n_shader;i++)
        gzprintf(visualizer_file, "%0.4f ",
     m_sc[i]->L1tex_windowed_cache_miss_rate(1)); gzprintf(visualizer_file,
     "\n"); gzprintf(visualizer_file, "CacheMissRate_ConstL1_noMgHt: "); for
     (unsigned i=0;i<m_n_shader;i++) gzprintf(visualizer_file, "%0.4f ",
     m_sc[i]->L1const_windowed_cache_miss_rate(1)); gzprintf(visualizer_file,
     "\n");
     // reset for next interval
     for (unsigned i=0;i<m_n_shader;i++)
        m_sc[i]->new_cache_window();
     */
  /* [한국어] 캐시 미스율 출력 코드 — 현재 주석 처리 (L1 캐시 인터페이스 변경으로 미사용) */
}

#include <iostream> /* [한국어] std::cout — 디버그 출력용 */
#include <list>     /* [한국어] std::list (직접 사용 없음, 미래 확장용) */
#include <map>      /* [한국어] std::map — uid → 타임스탬프 벡터 매핑 */
#include <vector>   /* [한국어] std::vector — 단계별 타임스탬프 저장 */
#include "../gpgpu-sim/shader.h"
/* [한국어] shader.h: IN_ICNT_TO_MEM, IN_SHADER_FETCHED 등 메모리 요청 단계 열거형 상수 */

/*
 * [한국어]
 * class my_time_vector — 메모리 요청(mem_fetch) 단계별 타임스탬프 추적기
 *
 * 각 mem_fetch 요청이 ICNT 투입 → DRAM 처리 → L2 캐시 → SM 반환까지
 * 통과하는 파이프라인 각 단계의 사이클 타임스탬프를 uid 기반으로 기록한다.
 * 완료된 요청들의 레이턴시 분포를 계산(calculate_dist)하여 AerialVision 파일에 출력한다.
 *
 * LD(읽기)와 ST(쓰기) 요청을 별도 map으로 관리한다.
 * 각 요청은 uid를 키로 하는 long int 벡터(크기: NUM_MEM_REQ_STAT 슬롯)로 저장된다.
 *
 * 전역 인스턴스: g_my_time_vector (time_vector_create()로 생성)
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 루프 내. 단일 스레드.
 */
class my_time_vector {
 private:
  std::map<unsigned int, std::vector<long int> > ld_time_map;
  /* [한국어] LD 요청 타임스탬프 맵: uid → 각 파이프라인 단계별 사이클 타임스탬프 벡터.
   * 설정자: update_ld()가 슬롯(단계 인덱스)에 사이클 값 기록.
   * 읽는 자: calculate_ld_dist()가 완료 요청 처리 후 맵에서 삭제(erase).
   * 값 범위: key=uid(32비트 고유 ID), value=크기 ld_vector_size의 long int 벡터(초기값 0).
   * 동기화: 단일 시뮬레이션 스레드에서만 접근 — 별도 락 불필요. */

  std::map<unsigned int, std::vector<long int> > st_time_map;
  /* [한국어] ST 요청 타임스탬프 맵: uid → 각 파이프라인 단계별 사이클 타임스탬프 벡터.
   * 설정자: update_st()가 슬롯에 사이클 값 기록.
   * 읽는 자: calculate_st_dist()가 완료 요청 처리 후 맵에서 삭제.
   * 값 범위: key=uid, value=크기 st_vector_size의 long int 벡터. LD와 동일 구조.
   * 동기화: 단일 시뮬레이션 스레드 전용. */

  unsigned ld_vector_size;
  /* [한국어] LD 요청의 단계 수 (= NUM_MEM_REQ_STAT).
   * 설정자: 생성자에서 ld_size 인자로 초기화.
   * 읽는 자: update_ld(), calculate_ld_dist(), print 계열 함수.
   * 값 범위: 양의 정수, mem_latency_stat.h의 NUM_MEM_REQ_STAT와 동일.
   * 동기화: 생성 후 변경 없음. */

  unsigned st_vector_size;
  /* [한국어] ST 요청의 단계 수 (= NUM_MEM_REQ_STAT).
   * 설정자: 생성자에서 st_size 인자로 초기화.
   * 읽는 자: update_st(), calculate_st_dist(), print 계열 함수.
   * 값 범위: ld_vector_size와 동일 값으로 초기화됨(time_vector_create에서 size,size 전달).
   * 동기화: 생성 후 변경 없음. */

  std::vector<double> ld_time_dist;
  /* [한국어] 현재 인터벌 LD 요청의 단계별 평균 레이턴시 분포 (단위: 사이클).
   * 설정자: calculate_ld_dist()에서 완료 요청들의 단계 간 시간 차(diff) 누적 후 평균.
   * 읽는 자: print_to_file(), print_to_gzfile()에서 AerialVision 파일 출력.
   * 값 범위: [0, ∞) double. 각 슬롯 i의 값은 해당 단계에 소요된 평균 사이클 수.
   * 동기화: calculate_ld_dist() 진입 시 clear() 후 resize() 재초기화. */

  std::vector<double> st_time_dist;
  /* [한국어] 현재 인터벌 ST 요청의 단계별 평균 레이턴시 분포 (단위: 사이클).
   * 설정자: calculate_st_dist()에서 완료 요청들의 단계 간 시간 차 누적 후 평균.
   * 읽는 자: print_to_file(), print_to_gzfile()에서 "STmemlatdist" 항목 출력.
   * 값 범위: [0, ∞) double. ld_time_dist와 동일 구조.
   * 동기화: calculate_st_dist() 진입 시 재초기화. */

  std::vector<double> overal_ld_time_dist;
  /* [한국어] 전체 시뮬레이션 LD 요청의 누적 가중 평균 레이턴시 분포.
   * 설정자: calculate_ld_dist()에서 인터벌 완료 시 이동 평균 업데이트
   *   (new_avg = (old_avg * old_cnt + interval_sum) / (old_cnt + interval_cnt)).
   * 읽는 자: print_dist()에서 "LD_mem_lat_dist" 출력.
   * 값 범위: [0, ∞) double. 전체 커널 실행에 걸친 레이턴시 평균.
   * 동기화: 단일 스레드 전용. */

  std::vector<double> overal_st_time_dist;
  /* [한국어] 전체 시뮬레이션 ST 요청의 누적 가중 평균 레이턴시 분포.
   * 설정자: calculate_st_dist()에서 이동 평균 업데이트.
   * 읽는 자: print_dist()에서 "ST_mem_lat_dist" 출력.
   * 값 범위: overal_ld_time_dist와 동일 구조.
   * 동기화: 단일 스레드 전용. */

  int overal_ld_count;
  /* [한국어] 전체 시뮬레이션에서 완료 처리된 LD 요청 수 (이동 평균 분모).
   * 설정자: calculate_ld_dist()에서 += finished_count로 누적.
   * 읽는 자: calculate_ld_dist()에서 이동 평균 계산 시 사용.
   * 값 범위: [0, ∞) int.
   * 동기화: 단일 스레드 전용. */

  int overal_st_count;
  /* [한국어] 전체 시뮬레이션에서 완료 처리된 ST 요청 수 (이동 평균 분모).
   * 설정자: calculate_st_dist()에서 += finished_count로 누적.
   * 읽는 자: calculate_st_dist()에서 이동 평균 계산 시 사용.
   * 값 범위: [0, ∞) int.
   * 동기화: 단일 스레드 전용. */

 public:
  /*
   * [한국어]
   * my_time_vector() - 타임스탬프 추적기 생성자
   *
   * @ld_size: LD 요청 단계 수 (NUM_MEM_REQ_STAT)
   * @st_size: ST 요청 단계 수 (NUM_MEM_REQ_STAT, ld_size와 동일)
   *
   * ld/st 벡터 크기 및 전체 누적 카운터를 초기화.
   * time_vector_create()에서 new my_time_vector(size, size)로 1회 호출.
   *
   * 호출 체인:
   *   time_vector_create(size) → [my_time_vector(size, size)]
   */
  my_time_vector(int ld_size, int st_size) {
    ld_vector_size = ld_size; /* [한국어] LD 단계 수 저장 */
    st_vector_size = st_size; /* [한국어] ST 단계 수 저장 */
    ld_time_dist.resize(ld_size);           /* [한국어] 인터벌 LD 분포 벡터 초기화 */
    st_time_dist.resize(st_size);           /* [한국어] 인터벌 ST 분포 벡터 초기화 */
    overal_ld_time_dist.resize(ld_size);    /* [한국어] 전체 누적 LD 분포 벡터 초기화 */
    overal_st_time_dist.resize(st_size);    /* [한국어] 전체 누적 ST 분포 벡터 초기화 */
    overal_ld_count = 0; /* [한국어] 완료된 LD 요청 수 0으로 초기화 */
    overal_st_count = 0; /* [한국어] 완료된 ST 요청 수 0으로 초기화 */
  }

  /*
   * [한국어]
   * update_ld() - LD 요청 특정 단계의 타임스탬프 기록
   *
   * @uid:  메모리 요청 UID
   * @slot: 파이프라인 단계 인덱스 (IN_ICNT_TO_MEM, IN_PARTITION_RDY_ICNT_POP 등)
   * @time: 현재 시뮬레이션 사이클 (타임스탬프)
   *
   * uid가 이미 맵에 있으면 해당 슬롯 타임스탬프만 갱신.
   * 없고 slot < NUM_MEM_REQ_STAT이면 새 벡터를 생성하여 맵에 삽입.
   * slot >= NUM_MEM_REQ_STAT이면 MSHR 병합 요청 — 무시 (merged mshr는 추적하지 않음).
   *
   * 호출 체인:
   *   time_vector_update() → [update_ld()] (type가 READ_REQUEST/READ_REPLY일 때)
   */
  void update_ld(unsigned int uid, unsigned int slot, long int time) {
    if (ld_time_map.find(uid) != ld_time_map.end()) {
      /* [한국어] 이미 추적 중인 UID: 해당 슬롯 타임스탬프 업데이트 */
      ld_time_map[uid][slot] = time;
    } else if (slot < NUM_MEM_REQ_STAT) {
      /* [한국어] 새 UID이며 유효한 슬롯: 새 타임스탬프 벡터 생성 및 삽입 */
      std::vector<long int> time_vec;        /* [한국어] 단계별 타임스탬프 벡터 생성 */
      time_vec.resize(ld_vector_size);       /* [한국어] NUM_MEM_REQ_STAT 슬롯으로 초기화 (0) */
      time_vec[slot] = time;                 /* [한국어] 현재 단계 타임스탬프 기록 */
      ld_time_map[uid] = time_vec;           /* [한국어] 맵에 삽입 */
    } else {
      // It's a merged mshr! forget it
      /* [한국어] MSHR 병합 요청: 최초 요청과 중복 추적하지 않음 — 무시 */
    }
  }

  /*
   * [한국어]
   * update_st() - ST 요청 특정 단계의 타임스탬프 기록
   *
   * @uid:  메모리 요청 UID
   * @slot: 파이프라인 단계 인덱스
   * @time: 현재 시뮬레이션 사이클
   *
   * update_ld()와 동일 로직. ST는 MSHR 병합이 없으므로 slot 범위 검사 없이
   * uid가 없으면 무조건 새 벡터 생성.
   *
   * 호출 체인:
   *   time_vector_update() → [update_st()] (type가 WRITE_REQUEST/WRITE_ACK일 때)
   */
  void update_st(unsigned int uid, unsigned int slot, long int time) {
    if (st_time_map.find(uid) != st_time_map.end()) {
      /* [한국어] 이미 추적 중인 UID: 해당 슬롯 타임스탬프 업데이트 */
      st_time_map[uid][slot] = time;
    } else {
      /* [한국어] 새 UID: 타임스탬프 벡터 생성 및 삽입 */
      std::vector<long int> time_vec;
      time_vec.resize(st_vector_size);
      time_vec[slot] = time;
      st_time_map[uid] = time_vec;
    }
  }

  /*
   * [한국어]
   * check_ld_update() - LD 레이턴시 계산 일관성 검증
   *
   * @uid:     메모리 요청 UID
   * @slot:    검증할 단계 인덱스
   * @latency: 외부에서 계산된 레이턴시 (slot_time - IN_ICNT_TO_MEM_time)
   *
   * ld_time_map[uid][slot] - ld_time_map[uid][IN_ICNT_TO_MEM]이 latency와
   * 일치하는지 assert로 검증. uid가 없고 slot이 유효하면 abort() (버그).
   * 디버그/진단 목적으로만 사용.
   *
   * 호출 체인:
   *   check_time_vector_update() → [check_ld_update()] (READ 타입)
   */
  void check_ld_update(unsigned int uid, unsigned int slot, long int latency) {
    if (ld_time_map.find(uid) != ld_time_map.end()) {
      /* [한국어] uid 존재: 내부 계산 레이턴시와 외부 레이턴시가 일치하는지 검증 */
      int our_latency =
          ld_time_map[uid][slot] - ld_time_map[uid][IN_ICNT_TO_MEM];
      /* [한국어] IN_ICNT_TO_MEM은 ICNT 투입 시점 — 이를 기준 시각으로 사용 */
      assert(our_latency == latency); /* [한국어] 불일치 시 assert 실패 → 디버그 종료 */
    } else if (slot < NUM_MEM_REQ_STAT) {
      abort(); /* [한국어] uid가 없는데 유효 슬롯이면 추적 누락 버그 → 강제 종료 */
    }
  }

  /*
   * [한국어]
   * check_st_update() - ST 레이턴시 계산 일관성 검증
   *
   * @uid:     메모리 요청 UID
   * @slot:    검증할 단계 인덱스
   * @latency: 외부에서 계산된 레이턴시
   *
   * check_ld_update()와 동일 로직. ST는 uid 없으면 무조건 abort().
   *
   * 호출 체인:
   *   check_time_vector_update() → [check_st_update()] (WRITE 타입)
   */
  void check_st_update(unsigned int uid, unsigned int slot, long int latency) {
    if (st_time_map.find(uid) != st_time_map.end()) {
      /* [한국어] uid 존재: 내부 계산 레이턴시와 일치 검증 */
      int our_latency =
          st_time_map[uid][slot] - st_time_map[uid][IN_ICNT_TO_MEM];
      assert(our_latency == latency);
    } else {
      abort(); /* [한국어] ST에서 uid 없는 경우는 예외 없는 버그 → 강제 종료 */
    }
  }

 private:
  /*
   * [한국어]
   * calculate_ld_dist() - 완료된 LD 요청들의 레이턴시 분포 계산
   *
   * ld_time_map 순회 → IN_SHADER_FETCHED 슬롯이 0이면 미완료 요청 skip.
   * 완료 요청: 첫 번째 비-0 슬롯(first)부터 순차적으로 인접 슬롯 간 시간 차(diff)를
   *   ld_time_dist[i]에 누적. 완료 요청 맵에서 삭제.
   * 인터벌 평균: finished_count로 나눠 각 슬롯의 평균 레이턴시 계산.
   * 전체 이동 평균 업데이트: (old_avg * old_cnt + interval_sum) / (old_cnt + interval_cnt).
   *
   * 호출 체인:
   *   calculate_dist() → [calculate_ld_dist()]
   */
  void calculate_ld_dist(void) {
    unsigned i, first;
    long int last_update, diff;
    int finished_count = 0;   /* [한국어] 이 인터벌에 완료된 LD 요청 수 */
    ld_time_dist.clear();
    ld_time_dist.resize(ld_vector_size); /* [한국어] 인터벌 분포 초기화 */
    std::map<unsigned int, std::vector<long int> >::iterator iter, iter_temp;
    iter = ld_time_map.begin(); /* [한국어] LD 맵 순회 시작 */
    while (iter != ld_time_map.end()) {
      last_update = 0; /* [한국어] 이전 단계 타임스탬프 (0 = 초기화) */
      first = -1;      /* [한국어] unsigned 음수: 루프에서 ++first 후 0 → 첫 유효 슬롯 탐색용 */
      if (!iter->second[IN_SHADER_FETCHED]) {
        // this request is not done yet skip it!
        /* [한국어] IN_SHADER_FETCHED 슬롯이 0 = 아직 SM으로 반환되지 않은 미완료 요청 */
        ++iter;
        continue;
      }
      while (!last_update) {
        /* [한국어] 첫 번째 비-0 타임스탬프(최초 단계)를 탐색 — 0인 슬롯은 기록 안 된 단계 */
        first++;
        assert(first < iter->second.size()); /* [한국어] 벡터 범위 초과 방어 */
        last_update = iter->second[first];   /* [한국어] 첫 유효 타임스탬프 설정 */
      }

      for (i = first; i < ld_vector_size; i++) {
        /* [한국어] 첫 유효 슬롯부터 마지막 슬롯까지 단계 간 시간 차 누적 */
        diff = iter->second[i] - last_update; /* [한국어] 현재 단계 - 이전 단계 = 소요 사이클 */
        if (diff > 0) {
          /* [한국어] 양수 diff만 누적 (0 = 기록 없는 슬롯, 음수 = 오류 방어) */
          ld_time_dist[i] += diff;           /* [한국어] 누적 합산 */
          last_update = iter->second[i];     /* [한국어] 다음 단계 기준 타임스탬프 갱신 */
        }
      }
      iter_temp = iter;
      iter++;
      ld_time_map.erase(iter_temp); /* [한국어] 완료 요청 맵에서 제거 (메모리 해제) */
      finished_count++;             /* [한국어] 완료 카운트 증가 */
    }
    if (finished_count) {
      /* [한국어] 이 인터벌에 완료된 요청이 있을 때만 평균 업데이트 */
      for (i = 0; i < ld_vector_size; i++) {
        /* [한국어] 전체 누적 이동 평균 업데이트: 기존 평균 * 기존 수 + 이번 합산 / 새 총계 */
        overal_ld_time_dist[i] =
            (overal_ld_time_dist[i] * overal_ld_count + ld_time_dist[i]) /
            (overal_ld_count + finished_count);
      }
      overal_ld_count += finished_count; /* [한국어] 전체 완료 수 업데이트 */
      for (i = 0; i < ld_vector_size; i++) {
        ld_time_dist[i] /= finished_count; /* [한국어] 인터벌 평균 계산 */
      }
    }
  }

  /*
   * [한국어]
   * calculate_st_dist() - 완료된 ST 요청들의 레이턴시 분포 계산
   *
   * calculate_ld_dist()와 동일한 알고리즘을 st_time_map에 적용.
   * ST 요청은 WRITE_ACK을 IN_SHADER_FETCHED와 동등한 완료 신호로 사용.
   *
   * 호출 체인:
   *   calculate_dist() → [calculate_st_dist()]
   */
  void calculate_st_dist(void) {
    unsigned i, first;
    long int last_update, diff;
    int finished_count = 0;   /* [한국어] 이 인터벌에 완료된 ST 요청 수 */
    st_time_dist.clear();
    st_time_dist.resize(st_vector_size); /* [한국어] 인터벌 ST 분포 초기화 */
    std::map<unsigned int, std::vector<long int> >::iterator iter, iter_temp;
    iter = st_time_map.begin(); /* [한국어] ST 맵 순회 시작 */
    while (iter != st_time_map.end()) {
      last_update = 0;
      first = -1;
      if (!iter->second[IN_SHADER_FETCHED]) {
        // this request is not done yet skip it!
        /* [한국어] 미완료 ST 요청 — skip */
        ++iter;
        continue;
      }
      while (!last_update) {
        /* [한국어] 첫 유효 타임스탬프 슬롯 탐색 */
        first++;
        assert(first < iter->second.size());
        last_update = iter->second[first];
      }

      for (i = first; i < st_vector_size; i++) {
        /* [한국어] 단계 간 시간 차 누적 */
        diff = iter->second[i] - last_update;
        if (diff > 0) {
          st_time_dist[i] += diff;
          last_update = iter->second[i];
        }
      }
      iter_temp = iter;
      iter++;
      st_time_map.erase(iter_temp); /* [한국어] 완료 ST 요청 맵에서 제거 */
      finished_count++;
    }
    if (finished_count) {
      /* [한국어] ST 전체 누적 이동 평균 업데이트 */
      for (i = 0; i < st_vector_size; i++) {
        overal_st_time_dist[i] =
            (overal_st_time_dist[i] * overal_st_count + st_time_dist[i]) /
            (overal_st_count + finished_count);
      }
      overal_st_count += finished_count; /* [한국어] 전체 완료 ST 수 업데이트 */
      for (i = 0; i < st_vector_size; i++) {
        st_time_dist[i] /= finished_count; /* [한국어] 인터벌 ST 평균 계산 */
      }
    }
  }

 public:
  /*
   * [한국어]
   * clear_time_map_vectors() - LD/ST 타임스탬프 맵 초기화
   *
   * 양쪽 맵의 모든 요청 타임스탬프를 삭제.
   * 현재 코드에서는 직접 호출되지 않음 — 추후 인터벌 리셋 용도로 예약.
   *
   * 호출 체인: (예약 — 현재 미사용)
   */
  void clear_time_map_vectors(void) {
    ld_time_map.clear(); /* [한국어] LD 맵 전체 삭제 */
    st_time_map.clear(); /* [한국어] ST 맵 전체 삭제 */
  }

  /*
   * [한국어]
   * print_all_ld() - 현재 추적 중인 모든 LD 요청 타임스탬프를 stdout에 출력 (디버그용)
   *
   * "ld_uid<uid> t0 t1 t2 ..." 형식으로 각 LD 요청의 모든 슬롯 타임스탬프 출력.
   *
   * 호출 체인: 직접 호출 (디버그 목적)
   */
  void print_all_ld(void) {
    unsigned i;
    std::map<unsigned int, std::vector<long int> >::iterator iter;
    for (iter = ld_time_map.begin(); iter != ld_time_map.end(); ++iter) {
      std::cout << "ld_uid" << iter->first; /* [한국어] LD 요청 UID 출력 */
      for (i = 0; i < ld_vector_size; i++) {
        std::cout << " " << iter->second[i]; /* [한국어] 각 단계 타임스탬프 출력 */
      }
      std::cout << std::endl;
    }
  }

  /*
   * [한국어]
   * print_all_st() - 현재 추적 중인 모든 ST 요청 타임스탬프를 stdout에 출력 (디버그용)
   *
   * "st_uid<uid> t0 t1 t2 ..." 형식으로 각 ST 요청의 모든 슬롯 타임스탬프 출력.
   *
   * 호출 체인: 직접 호출 (디버그 목적)
   */
  void print_all_st(void) {
    unsigned i;
    std::map<unsigned int, std::vector<long int> >::iterator iter;

    for (iter = st_time_map.begin(); iter != st_time_map.end(); ++iter) {
      std::cout << "st_uid" << iter->first; /* [한국어] ST 요청 UID 출력 */
      for (i = 0; i < st_vector_size; i++) {
        std::cout << " " << iter->second[i]; /* [한국어] 각 단계 타임스탬프 출력 */
      }
      std::cout << std::endl;
    }
  }

  /*
   * [한국어]
   * calculate_dist() - LD/ST 레이턴시 분포 계산 (공개 진입점)
   *
   * calculate_ld_dist()와 calculate_st_dist()를 순서대로 호출.
   * print_dist(), print_to_file(), print_to_gzfile()에서 호출되어
   * 완료 요청들의 레이턴시를 계산하고 맵에서 제거.
   *
   * 호출 체인:
   *   print_dist()/print_to_file()/print_to_gzfile() → [calculate_dist()]
   *     → calculate_ld_dist() + calculate_st_dist()
   */
  void calculate_dist() {
    calculate_ld_dist(); /* [한국어] 완료 LD 요청 레이턴시 분포 계산 및 맵 정리 */
    calculate_st_dist(); /* [한국어] 완료 ST 요청 레이턴시 분포 계산 및 맵 정리 */
  }

  /*
   * [한국어]
   * print_dist() - 전체 누적 LD/ST 레이턴시 분포를 stdout에 출력
   *
   * calculate_dist() 후 overal_ld/st_time_dist를 "LD_mem_lat_dist ..."
   * / "ST_mem_lat_dist ..." 형식으로 출력. 시뮬레이션 종료 시 요약에 사용.
   *
   * 호출 체인:
   *   time_vector_print() → [print_dist()]
   */
  void print_dist(void) {
    unsigned i;
    calculate_dist(); /* [한국어] 미처리된 완료 요청 레이턴시 계산 */
    std::cout << "LD_mem_lat_dist ";
    for (i = 0; i < ld_vector_size; i++) {
      std::cout << " " << (int)overal_ld_time_dist[i]; /* [한국어] 전체 평균 LD 분포 출력 */
    }
    std::cout << std::endl;
    std::cout << "ST_mem_lat_dist ";
    for (i = 0; i < st_vector_size; i++) {
      std::cout << " " << (int)overal_st_time_dist[i]; /* [한국어] 전체 평균 ST 분포 출력 */
    }
    std::cout << std::endl;
  }

  /*
   * [한국어]
   * print_to_file() - 현재 인터벌 LD/ST 레이턴시 분포를 FILE*에 출력
   *
   * @outfile: 출력 대상 FILE 포인터
   *
   * "LDmemlatdist: v0 v1 ..." / "STmemlatdist: v0 v1 ..." 형식으로 기록.
   * AerialVision 파일이 아닌 일반 텍스트 파일 출력 시 사용.
   *
   * 호출 체인: 직접 호출 (현재 visualizer.cc 내에서 직접 사용 없음,
   *   외부 모듈에서 참조 가능)
   */
  void print_to_file(FILE* outfile) {
    unsigned i;
    calculate_dist(); /* [한국어] 완료 요청 레이턴시 계산 */
    fprintf(outfile, "LDmemlatdist:"); /* [한국어] AerialVision 파서가 인식하는 키 */
    for (i = 0; i < ld_vector_size; i++) {
      fprintf(outfile, " %d", (int)ld_time_dist[i]); /* [한국어] 인터벌 LD 슬롯별 평균 출력 */
    }
    fprintf(outfile, "\n");
    fprintf(outfile, "STmemlatdist:");
    for (i = 0; i < st_vector_size; i++) {
      fprintf(outfile, " %d", (int)st_time_dist[i]); /* [한국어] 인터벌 ST 슬롯별 평균 출력 */
    }
    fprintf(outfile, "\n");
  }

  /*
   * [한국어]
   * print_to_gzfile() - 현재 인터벌 LD/ST 레이턴시 분포를 gzFile에 출력
   *
   * @outfile: AerialVision gzip 출력 파일 스트림
   *
   * print_to_file()과 동일하지만 gzprintf를 사용하여 압축 파일에 직접 기록.
   * visualizer_printstat()에서 time_vector_print_interval2gzfile()를 통해 호출.
   *
   * 호출 체인:
   *   visualizer_printstat() → time_vector_print_interval2gzfile()
   *     → [print_to_gzfile()]
   */
  void print_to_gzfile(gzFile outfile) {
    unsigned i;
    calculate_dist(); /* [한국어] 완료 요청 레이턴시 계산 */
    gzprintf(outfile, "LDmemlatdist:"); /* [한국어] AerialVision 키 문자열 */
    for (i = 0; i < ld_vector_size; i++) {
      gzprintf(outfile, " %d", (int)ld_time_dist[i]); /* [한국어] 인터벌 LD 슬롯 평균 */
    }
    gzprintf(outfile, "\n");
    gzprintf(outfile, "STmemlatdist:");
    for (i = 0; i < st_vector_size; i++) {
      gzprintf(outfile, " %d", (int)st_time_dist[i]); /* [한국어] 인터벌 ST 슬롯 평균 */
    }
    gzprintf(outfile, "\n");
  }
};

my_time_vector* g_my_time_vector;
/* [한국어] my_time_vector 전역 인스턴스 포인터.
 * 설정자: time_vector_create()에서 new로 생성.
 * 읽는 자: time_vector_print/update/check_time_vector_update 등 전역 함수.
 * 값 범위: 유효한 my_time_vector 포인터 (time_vector_create 호출 후).
 * 동기화: 단일 시뮬레이션 스레드 전용. */

/*
 * [한국어]
 * time_vector_create() - my_time_vector 전역 객체 생성
 *
 * @size: 추적할 메모리 요청 단계 수 (NUM_MEM_REQ_STAT)
 *
 * g_my_time_vector = new my_time_vector(size, size)로 LD/ST 동일 크기 초기화.
 * 시뮬레이터 초기화 시(gpgpusim_entrypoint.cc 등) 1회 호출.
 *
 * 호출 체인:
 *   시뮬레이터 초기화 → [time_vector_create(NUM_MEM_REQ_STAT)]
 */
void time_vector_create(int size) {
  g_my_time_vector = new my_time_vector(size, size);
  /* [한국어] LD/ST 모두 동일 슬롯 수(NUM_MEM_REQ_STAT)로 초기화 */
}

/*
 * [한국어]
 * time_vector_print() - 전체 LD/ST 레이턴시 분포를 stdout에 출력
 *
 * g_my_time_vector->print_dist()로 위임.
 * 시뮬레이션 종료 후 최종 통계 출력 시 호출.
 *
 * 호출 체인:
 *   gpgpu_sim::print_stats() 또는 최종 정리 코드 → [time_vector_print()]
 *     → my_time_vector::print_dist()
 */
void time_vector_print(void) { g_my_time_vector->print_dist(); }

/*
 * [한국어]
 * time_vector_print_interval2gzfile() - 인터벌 LD/ST 레이턴시 분포를 gzFile에 출력
 *
 * @outfile: AerialVision gzip 출력 파일 스트림
 *
 * 파일 범위 내부 정적 함수. visualizer_printstat()만이 호출.
 * g_my_time_vector->print_to_gzfile()으로 위임.
 *
 * 호출 체인:
 *   visualizer_printstat() → [time_vector_print_interval2gzfile()]
 *     → my_time_vector::print_to_gzfile()
 */
static void time_vector_print_interval2gzfile(gzFile outfile) {
  g_my_time_vector->print_to_gzfile(outfile);
  /* [한국어] 현재 인터벌의 LD/ST 분포를 gzip 파일에 기록 */
}

#include "../gpgpu-sim/mem_fetch.h"
/* [한국어] READ_REQUEST, READ_REPLY, WRITE_REQUEST, WRITE_ACK 열거형 상수 포함 */

/*
 * [한국어]
 * time_vector_update() - mem_fetch 단계 진입 시 타임스탬프 기록
 *
 * @uid:   mem_fetch의 고유 UID
 * @slot:  진입한 파이프라인 단계 인덱스 (IN_ICNT_TO_MEM, IN_PARTITION_RDY_ICNT_POP 등)
 * @cycle: 현재 시뮬레이션 사이클 (타임스탬프로 기록)
 * @type:  메모리 요청 타입 (READ_REQUEST/READ_REPLY → LD, WRITE_REQUEST/WRITE_ACK → ST)
 * @return: void
 *
 * type 기반으로 LD/ST 분기 후 g_my_time_vector의 update_ld/update_st 호출.
 * 알 수 없는 type이면 abort() (프로그래밍 오류).
 * mem_fetch.cc에서 메모리 파이프라인 각 단계 진입 시 호출.
 *
 * 호출 체인:
 *   mem_fetch.cc 파이프라인 단계 코드 → [time_vector_update()]
 *     → my_time_vector::update_ld() 또는 update_st()
 */
void time_vector_update(unsigned int uid, int slot, long int cycle, int type) {
  if ((type == READ_REQUEST) || (type == READ_REPLY)) {
    /* [한국어] 읽기 요청 (캐시 미스 페치/응답) → LD 맵에 타임스탬프 기록 */
    g_my_time_vector->update_ld(uid, slot, cycle);
  } else if ((type == WRITE_REQUEST) || (type == WRITE_ACK)) {
    /* [한국어] 쓰기 요청/확인 → ST 맵에 타임스탬프 기록 */
    g_my_time_vector->update_st(uid, slot, cycle);
  } else {
    abort(); /* [한국어] 알 수 없는 type → 프로그래밍 오류, 강제 종료 */
  }
}

/*
 * [한국어]
 * check_time_vector_update() - 레이턴시 계산 일관성 검증
 *
 * @uid:     mem_fetch의 고유 UID
 * @slot:    검증할 파이프라인 단계 인덱스
 * @latency: 외부에서 계산된 레이턴시 값 (검증 대상)
 * @type:    메모리 요청 타입 (READ/WRITE)
 * @return:  void (실패 시 assert/abort)
 *
 * time_vector_update()와 동일한 type 분기로 check_ld_update/check_st_update 호출.
 * 타임스탬프 기반 레이턴시와 외부 계산 레이턴시가 일치하는지 assert 검증.
 * 디버그 빌드에서 타임스탬프 추적 정확도 보장 목적.
 *
 * 호출 체인:
 *   mem_fetch.cc 진단 코드 → [check_time_vector_update()]
 *     → my_time_vector::check_ld_update() 또는 check_st_update()
 */
void check_time_vector_update(unsigned int uid, int slot, long int latency,
                              int type) {
  if ((type == READ_REQUEST) || (type == READ_REPLY)) {
    /* [한국어] 읽기 요청 → LD 레이턴시 일관성 검증 */
    g_my_time_vector->check_ld_update(uid, slot, latency);
  } else if ((type == WRITE_REQUEST) || (type == WRITE_ACK)) {
    /* [한국어] 쓰기 요청 → ST 레이턴시 일관성 검증 */
    g_my_time_vector->check_st_update(uid, slot, latency);
  } else {
    abort(); /* [한국어] 알 수 없는 type → 프로그래밍 오류, 강제 종료 */
  }
}
