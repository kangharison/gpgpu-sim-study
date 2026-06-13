// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// George L. Yuan
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
 * [한국어 설명] 메모리 레이턴시 통계 구현 (mem_latency_stat.cc)
 *
 * === 파일의 역할 ===
 * memory_stats_t 클래스의 모든 멤버 함수를 구현한다. 이 클래스는 GPU 시뮬레이션 중
 * mem_fetch(메모리 요청 패킷)가 SM → ICNT → L2 → DRAM 경로를 따라 이동할 때 각 경계에서
 * 레이턴시를 측정하고 히스토그램 형태로 수집한다. 또한 DRAM 뱅크별 접근 통계, 행(row)
 * 지역성 통계, AerialVision 시각화용 per-window L2 통계를 관리한다.
 * 시뮬레이션 종료 시 memlatstat_print()가 전체 히스토그램과 뱅크 불균형 분석 결과를 출력한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 흐름:
 *   gpgpu_sim 생성자 → memory_stats_t 생성자 (대규모 동적 배열 할당)
 *   gpgpu_sim::cycle() 내부 →
 *     memory_sub_partition::cache_cycle() → memlatstat_read_done()/memlatstat_done()
 *     memory_partition_unit::cache_cycle() → memlatstat_icnt2mem_pop()
 *     dram_t::issue_col() → memlatstat_dram_access()
 *     샘플링 창 경계 → memlatstat_lat_pw()
 *   gpgpu_sim::print_stats() → memlatstat_print()
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 시뮬레이션 루프 스레드
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - mem_fetch.h: mem_fetch — get_timestamp(), get_sid(), get_tlx_addr() 등
 *   - gpu-sim.h: gpgpu_sim — gpu_sim_cycle, gpu_tot_sim_cycle 현재 사이클 정보
 *   - dram.h: dram_t — DRAM 접근 시 memlatstat_dram_access() 호출
 *   - shader.h: shader_core_config — SM 수, warp_size 등
 *   - stat-tool.h: LOGB2() 매크로 — 로그₂ 히스토그램 버킷 계산
 *   - ptx-stats.h: ptx_file_line_stats_add_dram_traffic() — per-PC DRAM 트래픽 추적
 * 이 파일에 의존하는 모듈:
 *   - l2cache.cc: L2 미스/히트 시 memory_stats_t 필드 직접 접근
 *   - gpu-sim.cc: memlatstat_print(), visualizer_print(), clear_L2_stats_pw() 호출
 *   - power_stat.cc: power_mem_stat_t 생성자에서 memory_stats_t 참조
 *
 * === 주요 함수/구조체 요약 ===
 * memory_stats_t 생성자 - 3D 뱅크 접근 배열, 레이턴시 테이블 등 대규모 동적 배열 calloc 할당
 * memlatstat_done()     - mem_fetch 완료 시 전체 레이턴시 기록 (mf_lat_table, max_mf_latency)
 * memlatstat_read_done()- L2→SM 완료 시 SST/일반 모드 분기 후 mf_lat, icnt2sh_lat 기록
 * memlatstat_dram_access() - DRAM 접근 시 뱅크별 읽기/쓰기 카운터 및 행 지역성 통계 갱신
 * memlatstat_icnt2mem_pop() - ICNT→MEM 팝 시 NoC 레이턴시 기록
 * memlatstat_lat_pw()   - 샘플링 창 경계에서 창별 평균 레이턴시 계산 후 카운터 리셋
 * memlatstat_print()    - 전체 히스토그램 및 뱅크 불균형(skew) 분석 결과 출력
 */

#include "mem_latency_stat.h"                  /* [한국어] 자기 자신의 헤더 — memory_stats_t 선언 */
#include "../abstract_hardware_model.h"        /* [한국어] warp, thread block 등 추상 하드웨어 모델 */
#include "../cuda-sim/ptx-stats.h"             /* [한국어] ptx_file_line_stats — per-PC DRAM 트래픽 추적 */
#include "dram.h"                              /* [한국어] dram_t — DRAM 타이밍 시뮬레이터 */
#include "gpu-cache.h"                         /* [한국어] cache_t, cache_stats — 캐시 계층 */
#include "gpu-misc.h"                          /* [한국어] 공통 유틸리티 (LOGB2 등) */
#include "gpu-sim.h"                           /* [한국어] gpgpu_sim — 시뮬레이터 전역 상태, 현재 사이클 */
#include "mem_fetch.h"                         /* [한국어] mem_fetch — 메모리 요청 패킷 (타임스탬프, sid 등) */
#include "shader.h"                            /* [한국어] shader_core_config — SM 구성 파라미터 */
#include "stat-tool.h"                         /* [한국어] LOGB2() 매크로 — 로그₂ 히스토그램 버킷 인덱스 계산 */
#include "visualizer.h"                        /* [한국어] shader_mem_lat_log, shader_mem_acc_log — 시각화 로그 */

#include <math.h>    /* [한국어] ceil() — DRAM atom size로 나눌 때 올림 연산 */
#include <stdio.h>   /* [한국어] printf() — 통계 출력 */
#include <stdlib.h>  /* [한국어] calloc(), malloc(), free() — 동적 배열 할당 */
#include <string.h>  /* [한국어] memset() — 히스토그램 배열 0 초기화 */

#include "../../libcuda/gpgpu_context.h"       /* [한국어] gpgpu_context — PTX 통계 접근 경로 */

/*
 * [한국어]
 * memory_stats_t 생성자 - 레이턴시 통계에 필요한 모든 동적 배열 할당 및 초기화
 *
 * @n_shader: SM(Streaming Multiprocessor) 개수
 * @shader_config: SM 구성 (n_thread_per_shader, warp_size 등)
 * @mem_config: 메모리 구성 (m_n_mem=DRAM 채널 수, nbk=뱅크 수, dram_atom_size 등)
 * @gpu: 시뮬레이터 전역 상태 (현재 사이클 번호 조회용)
 * @return: 없음 (생성자)
 *
 * 1. 설정 유효성 assert() 검증.
 * 2. DRAM 행(row) 지역성 2D 배열 [m_n_mem][nbk] calloc:
 *    concurrent_row_access, num_activates, row_access,
 *    max_conc_access2samerow, max_servicetime2samerow
 * 3. 스칼라 카운터/최댓값 모두 0 초기화.
 * 4. 히스토그램 배열 memset 0: mrq_lat_table[32], dq_lat_table[32],
 *    mf_lat_table[32], icnt2mem/icnt2sh_lat_table[24], mf_lat_pw_table[32]
 * 5. max_warps 계산: n_shader * (n_thread_per_shader/warp_size + 1)
 * 6. 2D/3D 뱅크 접근 배열 [m_n_mem][nbk] 및 [n_shader][m_n_mem][nbk] calloc:
 *    totalbankreads, totalbankwrites, totalbankaccesses,
 *    mf_total_lat_table, mf_max_lat_table,
 *    bankreads, bankwrites (3D: SM별 × 채널별 × 뱅크별)
 * 7. num_MCBs_accessed [m_n_mem * nbk] calloc.
 * 8. position_of_mrq_chosen: FRFCFS 큐 크기 또는 기본 1024로 calloc.
 * 9. mem_access_type_stats [NUM_MEM_ACCESS_TYPE][m_n_mem][nbk+1] malloc+calloc.
 * 10. AerialVision L2 통계 L2_read_miss/hit 등 0 초기화.
 * 11. L2 데이터 길이 배열 [m_n_mem] calloc.
 *
 * 호출 체인:
 *   gpgpu_sim 생성자 → new memory_stats_t()
 */
memory_stats_t::memory_stats_t(unsigned n_shader,
                               const shader_core_config *shader_config,
                               const memory_config *mem_config,
                               const class gpgpu_sim *gpu) {
  assert(mem_config->m_valid);        /* [한국어] 메모리 설정이 유효하게 초기화되었는지 검증 */
  assert(shader_config->m_valid);     /* [한국어] SM 설정 유효성 검증 */

  unsigned i, j;  /* [한국어] 루프 인덱스 (DRAM 채널, 뱅크 순회에 사용) */

  /* [한국어] DRAM 행(row) 지역성 통계 배열 1차원 포인터 배열 할당 (m_n_mem 개)
   * 각 원소는 뒤에서 [nbk] 크기의 1D 배열로 초기화된다 */
  concurrent_row_access =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  /* [한국어] 행 동시 접근 수 배열 포인터 테이블 할당 */
  num_activates =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  /* [한국어] 뱅크별 ACT(Activate) 명령 수 배열 포인터 테이블 할당 */
  row_access =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  /* [한국어] 현재 열린 행에 대한 누적 접근 수 배열 포인터 테이블 할당 */
  max_conc_access2samerow =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  /* [한국어] 단일 행 최대 연속 접근 수 배열 포인터 테이블 할당 */
  max_servicetime2samerow =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  /* [한국어] 단일 행 최대 서비스 시간 배열 포인터 테이블 할당 */

  for (unsigned i = 0; i < mem_config->m_n_mem; i++) {  /* [한국어] 각 DRAM 채널(chip)별 2D 배열 구성 */
    concurrent_row_access[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    /* [한국어] 채널 i의 뱅크별 동시 행 접근 수 배열 할당 */
    row_access[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    /* [한국어] 채널 i의 뱅크별 행 접근 수 배열 할당 */
    num_activates[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    /* [한국어] 채널 i의 뱅크별 activate 명령 수 배열 할당 */
    max_conc_access2samerow[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    /* [한국어] 채널 i의 뱅크별 최대 연속 접근 수 배열 할당 */
    max_servicetime2samerow[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    /* [한국어] 채널 i의 뱅크별 최대 서비스 시간 배열 할당 */
  }

  m_n_shader = n_shader;       /* [한국어] SM 개수 저장 */
  m_memory_config = mem_config; /* [한국어] 메모리 구성 참조 저장 */
  m_gpu = gpu;                  /* [한국어] 시뮬레이터 전역 상태 참조 저장 (사이클 번호 조회용) */
  total_n_access = 0;           /* [한국어] 전체 메모리 접근 수 초기화 */
  total_n_reads = 0;            /* [한국어] 전체 읽기 접근 수 초기화 */
  total_n_writes = 0;           /* [한국어] 전체 쓰기 접근 수 초기화 */
  max_mrq_latency = 0;          /* [한국어] 최대 MRQ 레이턴시 초기화 */
  max_dq_latency = 0;           /* [한국어] 최대 DQ 레이턴시 초기화 */
  max_mf_latency = 0;           /* [한국어] 최대 mem_fetch 레이턴시 초기화 */
  max_icnt2mem_latency = 0;     /* [한국어] 최대 ICNT→MEM 레이턴시 초기화 */
  max_icnt2sh_latency = 0;      /* [한국어] 최대 MEM→SM 레이턴시 초기화 */
  tot_icnt2mem_latency = 0;     /* [한국어] ICNT→MEM 레이턴시 누적 합 초기화 */
  tot_icnt2sh_latency = 0;      /* [한국어] MEM→SM 레이턴시 누적 합 초기화 */
  tot_mrq_num = 0;              /* [한국어] MRQ 총 수 초기화 */
  tot_mrq_latency = 0;          /* [한국어] MRQ 레이턴시 누적 합 초기화 */
  memset(mrq_lat_table, 0, sizeof(unsigned) * 32);    /* [한국어] MRQ 레이턴시 히스토그램 32버킷 초기화 */
  memset(dq_lat_table, 0, sizeof(unsigned) * 32);     /* [한국어] DQ 레이턴시 히스토그램 32버킷 초기화 */
  memset(mf_lat_table, 0, sizeof(unsigned) * 32);     /* [한국어] mem_fetch 레이턴시 히스토그램 32버킷 초기화 */
  memset(icnt2mem_lat_table, 0, sizeof(unsigned) * 24); /* [한국어] ICNT→MEM 레이턴시 히스토그램 24버킷 초기화 */
  memset(icnt2sh_lat_table, 0, sizeof(unsigned) * 24);  /* [한국어] MEM→SM 레이턴시 히스토그램 24버킷 초기화 */
  memset(mf_lat_pw_table, 0, sizeof(unsigned) * 32);    /* [한국어] per-window 레이턴시 히스토그램 32버킷 초기화 */
  mf_num_lat_pw = 0;    /* [한국어] 창별 완료된 mem_fetch 수 초기화 */
  max_warps =
      n_shader *
      (shader_config->n_thread_per_shader / shader_config->warp_size + 1);
  /* [한국어] 전체 SM의 최대 warp 수 계산:
   * 각 SM이 가질 수 있는 최대 warp = ceil(n_thread_per_shader / warp_size)
   * +1은 나머지 스레드를 포함한 warp 하나를 위한 여유 */
  mf_tot_lat_pw = 0;  // total latency summed up per window. divide by
                      // mf_num_lat_pw to obtain average latency Per Window
  /* [한국어] 현재 샘플링 창의 mem_fetch 레이턴시 합 초기화
   * mf_tot_lat_pw / mf_num_lat_pw = 현재 창의 평균 레이턴시 */
  mf_total_lat = 0;   /* [한국어] 전체 누적 레이턴시 합 초기화 */
  num_mfs = 0;        /* [한국어] 완료된 mem_fetch 총 수 초기화 */
  printf("*** Initializing Memory Statistics ***\n");  /* [한국어] 메모리 통계 초기화 시작 로그 */

  /* [한국어] 뱅크별 접근 통계 2D 배열 포인터 테이블 할당 (m_n_mem 크기)
   * 실제 뱅크별 배열은 두 번째 루프에서 [nbk] 크기로 할당된다 */
  totalbankreads =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  /* [한국어] 채널별 총 읽기 접근 수 포인터 배열 할당 */
  totalbankwrites =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  /* [한국어] 채널별 총 쓰기 접근 수 포인터 배열 할당 */
  totalbankaccesses =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  /* [한국어] 채널별 총 접근 수 (읽기+쓰기) 포인터 배열 할당 */
  mf_total_lat_table = (unsigned long long int **)calloc(
      mem_config->m_n_mem, sizeof(unsigned long long *));
  /* [한국어] 채널별 총 mem_fetch 레이턴시 합 포인터 배열 할당 (long long: 오버플로 방지) */
  mf_max_lat_table =
      (unsigned **)calloc(mem_config->m_n_mem, sizeof(unsigned *));
  /* [한국어] 채널별 최대 mem_fetch 레이턴시 포인터 배열 할당 */
  bankreads = (unsigned int ***)calloc(n_shader, sizeof(unsigned int **));
  /* [한국어] SM별 DRAM 뱅크 읽기 횟수 3D 배열 1차원 (n_shader) 할당 */
  bankwrites = (unsigned int ***)calloc(n_shader, sizeof(unsigned int **));
  /* [한국어] SM별 DRAM 뱅크 쓰기 횟수 3D 배열 1차원 (n_shader) 할당 */
  num_MCBs_accessed = (unsigned int *)calloc(
      mem_config->m_n_mem * mem_config->nbk, sizeof(unsigned int));
  /* [한국어] MCB(메모리 컨트롤러 뱅크) 접근 히스토그램: m_n_mem * nbk 크기
   * num_MCBs_accessed[i]는 한 warp가 동시에 i개의 MCB를 접근한 횟수 */
  if (mem_config->gpgpu_frfcfs_dram_sched_queue_size) {  /* [한국어] FRFCFS 큐 크기가 설정되어 있으면 */
    position_of_mrq_chosen = (unsigned int *)calloc(
        mem_config->gpgpu_frfcfs_dram_sched_queue_size, sizeof(unsigned int));
    /* [한국어] 설정된 FRFCFS 큐 크기만큼 MRQ 위치 히스토그램 배열 할당 */
  } else
    position_of_mrq_chosen = (unsigned int *)calloc(1024, sizeof(unsigned int));
    /* [한국어] FRFCFS 큐 크기 미설정 시 기본 1024 크기로 할당 */

  for (i = 0; i < n_shader; i++) {  /* [한국어] SM별 뱅크 접근 3D 배열 2~3차원 구성 */
    bankreads[i] =
        (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
    /* [한국어] SM i의 채널별 읽기 포인터 배열 할당 */
    bankwrites[i] =
        (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
    /* [한국어] SM i의 채널별 쓰기 포인터 배열 할당 */
    for (j = 0; j < mem_config->m_n_mem; j++) {  /* [한국어] SM i의 채널 j별 뱅크 배열 할당 */
      bankreads[i][j] =
          (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
      /* [한국어] SM i, 채널 j의 뱅크별 읽기 카운터 배열 (nbk 크기, 0 초기화) */
      bankwrites[i][j] =
          (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
      /* [한국어] SM i, 채널 j의 뱅크별 쓰기 카운터 배열 (nbk 크기, 0 초기화) */
    }
  }

  for (i = 0; i < mem_config->m_n_mem; i++) {  /* [한국어] 채널별 totalbankread/write 배열 실제 할당 */
    totalbankreads[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    /* [한국어] 채널 i의 뱅크별 총 읽기 카운터 배열 할당 */
    totalbankwrites[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    /* [한국어] 채널 i의 뱅크별 총 쓰기 카운터 배열 할당 */
    totalbankaccesses[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    /* [한국어] 채널 i의 뱅크별 총 접근 (읽기+쓰기) 카운터 배열 할당 */
    mf_total_lat_table[i] = (unsigned long long int *)calloc(
        mem_config->nbk, sizeof(unsigned long long int));
    /* [한국어] 채널 i의 뱅크별 누적 레이턴시 합 배열 (long long 사용) */
    mf_max_lat_table[i] = (unsigned *)calloc(mem_config->nbk, sizeof(unsigned));
    /* [한국어] 채널 i의 뱅크별 최대 레이턴시 배열 */
  }

  mem_access_type_stats =
      (unsigned ***)malloc(NUM_MEM_ACCESS_TYPE * sizeof(unsigned **));
  /* [한국어] DRAM 접근 타입별 통계 배열 1차원 (NUM_MEM_ACCESS_TYPE) 할당
   * NUM_MEM_ACCESS_TYPE: 읽기/쓰기/텍스처/상수 등 메모리 접근 타입 수 */
  for (i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {  /* [한국어] 각 접근 타입별 채널×뱅크 배열 구성 */
    int j;
    mem_access_type_stats[i] =
        (unsigned **)calloc(mem_config->m_n_mem, sizeof(unsigned *));
    /* [한국어] 접근 타입 i의 채널별 포인터 배열 할당 */
    for (j = 0; (unsigned)j < mem_config->m_n_mem; j++) {
      mem_access_type_stats[i][j] =
          (unsigned *)calloc((mem_config->nbk + 1), sizeof(unsigned *));
      /* [한국어] 접근 타입 i, 채널 j의 뱅크별 카운터 배열 (nbk+1 크기 — 1개 여유 포함) */
    }
  }

  // AerialVision L2 stats
  L2_read_miss = 0;   /* [한국어] 현재 샘플링 창 L2 읽기 미스 수 초기화 */
  L2_write_miss = 0;  /* [한국어] 현재 샘플링 창 L2 쓰기 미스 수 초기화 */
  L2_read_hit = 0;    /* [한국어] 현재 샘플링 창 L2 읽기 히트 수 초기화 */
  L2_write_hit = 0;   /* [한국어] 현재 샘플링 창 L2 쓰기 히트 수 초기화 */

  /* [한국어] L2 데이터 흐름 길이(바이트) 히스토그램 배열: m_n_mem(채널 수) 크기로 calloc */
  L2_cbtoL2length =
      (unsigned int *)calloc(mem_config->m_n_mem, sizeof(unsigned int));
  /* [한국어] CB(ICNT)→L2 요청 데이터 길이 히스토그램 배열 */
  L2_cbtoL2writelength =
      (unsigned int *)calloc(mem_config->m_n_mem, sizeof(unsigned int));
  /* [한국어] CB→L2 쓰기 요청 데이터 길이 히스토그램 배열 */
  L2_L2tocblength =
      (unsigned int *)calloc(mem_config->m_n_mem, sizeof(unsigned int));
  /* [한국어] L2→CB(ICNT) 응답 데이터 길이 히스토그램 배열 */
  L2_dramtoL2length =
      (unsigned int *)calloc(mem_config->m_n_mem, sizeof(unsigned int));
  /* [한국어] DRAM→L2 응답 데이터 길이 히스토그램 배열 */
  L2_dramtoL2writelength =
      (unsigned int *)calloc(mem_config->m_n_mem, sizeof(unsigned int));
  /* [한국어] DRAM→L2 라이트백 데이터 길이 히스토그램 배열 */
  L2_L2todramlength =
      (unsigned int *)calloc(mem_config->m_n_mem, sizeof(unsigned int));
  /* [한국어] L2→DRAM 라이트백 데이터 길이 히스토그램 배열 */
}

/*
 * [한국어]
 * memory_stats_t::memlatstat_done - mem_fetch 완료 시 전체 레이턴시 기록
 *
 * @mf: 완료된 메모리 요청 패킷
 * @return: 계산된 레이턴시 (현재 사이클 - mf 발행 사이클, 단위: 사이클)
 *
 * mem_fetch가 L2 히트 또는 DRAM 응답으로 SM에 최종 도달할 때 호출된다.
 * mf->get_timestamp()는 mem_fetch가 SM에서 발행된 사이클을 저장하고 있다.
 * 현재 사이클(gpu_sim_cycle + gpu_tot_sim_cycle)에서 이 값을 빼면
 * SM 발행부터 최종 완료까지의 전체 레이턴시가 된다.
 * 계산된 레이턴시로 per-window 카운터와 히스토그램, 뱅크별 누적값을 업데이트한다.
 *
 * 호출 체인:
 *   memory_sub_partition::cache_cycle() → [memlatstat_done()] ← memlatstat_read_done()
 */
// record the total latency
unsigned memory_stats_t::memlatstat_done(mem_fetch *mf) {
  unsigned mf_latency;  /* [한국어] 계산된 mem_fetch 전체 레이턴시 (사이클) */
  mf_latency =
      (m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) - mf->get_timestamp();
  /* [한국어] 현재 총 사이클(시뮬레이션 사이클 + 이전 커널 누적) - 발행 사이클
   * = SM 발행 시점부터 응답 도달까지 경과 사이클 */
  mf_num_lat_pw++;              /* [한국어] 현재 샘플링 창에서 완료된 mem_fetch 수 1 증가 */
  mf_tot_lat_pw += mf_latency;  /* [한국어] 현재 샘플링 창의 레이턴시 합에 추가 */
  unsigned idx = LOGB2(mf_latency);  /* [한국어] 로그₂ 히스토그램 버킷 인덱스 계산 (stat-tool.h LOGB2 매크로) */
  assert(idx < 32);             /* [한국어] 버킷 인덱스가 배열 범위(32) 내에 있는지 검증 */
  mf_lat_table[idx]++;          /* [한국어] 해당 레이턴시 범위의 버킷 카운터 1 증가 */
  shader_mem_lat_log(mf->get_sid(), mf_latency);
  /* [한국어] visualizer.h의 SM별 레이턴시 로그 함수 호출 — AerialVision 시각화 데이터 기록 */
  mf_total_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk] +=
      mf_latency;
  /* [한국어] 해당 DRAM 채널(chip)과 뱅크(bk)의 누적 레이턴시에 추가
   * get_tlx_addr().chip: 메모리 파티션(DRAM 채널) ID
   * get_tlx_addr().bk: DRAM 뱅크 ID */
  if (mf_latency > max_mf_latency) max_mf_latency = mf_latency;
  /* [한국어] 지금까지의 최대 레이턴시 갱신 */
  return mf_latency;  /* [한국어] 계산된 레이턴시 반환 — 호출자(memlatstat_read_done)가 icnt2sh 계산에 사용 */
}

/*
 * [한국어]
 * memory_stats_t::memlatstat_read_done - L2→SM 읽기 완료 시 레이턴시 통계 기록
 *
 * @mf: 완료된 읽기 mem_fetch 패킷
 * @return: 없음 (void)
 *
 * SST_mode(SystemC-SystemC-Time 모드)와 일반 모드로 동작을 분기한다:
 *   - SST_mode: 단순히 레이턴시만 계산하여 num_mfs, mf_total_lat, max_mf_latency 갱신.
 *               SST는 외부 SystemC 시뮬레이터와 연동 모드이므로 히스토그램 불필요.
 *   - 일반 모드(gpgpu_memlatency_stat 활성화): memlatstat_done()으로 전체 레이턴시 기록 후
 *     추가로 mf_max_lat_table(뱅크별 최대값)과 icnt2sh 레이턴시(DRAM 응답 → SM 도달 시간)도 기록.
 *     icnt2sh_latency = 현재 사이클 - mf->get_return_timestamp()
 *     (return_timestamp: DRAM 응답이 ICNT 주입된 사이클)
 *
 * 호출 체인:
 *   memory_sub_partition::cache_cycle() (L2→SM 응답) → [memlatstat_read_done()]
 */
void memory_stats_t::memlatstat_read_done(mem_fetch *mf) {
  if (m_memory_config->SST_mode) {  /* [한국어] SST(SystemC) 연동 모드: 단순 레이턴시만 기록 */
    // in SST mode, we just calculate mem latency
    unsigned mf_latency;  /* [한국어] SST 모드 레이턴시 계산 변수 */
    mf_latency =
        (m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) - mf->get_timestamp();
    /* [한국어] 현재 사이클 - 발행 사이클 = 전체 레이턴시 (SST에서는 히스토그램 없이 평균만 계산) */
    num_mfs++;                    /* [한국어] 완료된 mem_fetch 총 수 1 증가 */
    mf_total_lat += mf_latency;   /* [한국어] 전체 누적 레이턴시에 추가 (평균 계산용) */
    if (mf_latency > max_mf_latency) max_mf_latency = mf_latency;
    /* [한국어] SST 모드에서도 최대 레이턴시 갱신 */
  } else if (m_memory_config->gpgpu_memlatency_stat) {  /* [한국어] 일반 레이턴시 통계 활성화 모드 */
    unsigned mf_latency = memlatstat_done(mf);
    /* [한국어] 전체 레이턴시 기록 (mf_lat_table 히스토그램, max_mf_latency 갱신) */
    if (mf_latency >
        mf_max_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk])
      mf_max_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk] =
          mf_latency;
    /* [한국어] 해당 DRAM 채널·뱅크의 최대 레이턴시 갱신 (뱅크별 최악 케이스 분석) */
    unsigned icnt2sh_latency;  /* [한국어] DRAM 응답 → SM ICNT 출구 레이턴시 (사이클) */
    icnt2sh_latency = (m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle) -
                      mf->get_return_timestamp();
    /* [한국어] get_return_timestamp(): DRAM이 응답을 ICNT에 주입한 사이클
     * 현재 사이클 - return_timestamp = ICNT 역방향(MEM→SM) 레이턴시 */
    tot_icnt2sh_latency += icnt2sh_latency;  /* [한국어] MEM→SM NoC 레이턴시 누적 합 */
    icnt2sh_lat_table[LOGB2(icnt2sh_latency)]++;
    /* [한국어] MEM→SM 레이턴시 로그₂ 히스토그램 버킷 증가 */
    if (icnt2sh_latency > max_icnt2sh_latency)
      max_icnt2sh_latency = icnt2sh_latency;
    /* [한국어] MEM→SM 방향 최대 레이턴시 갱신 */
  }
}

/*
 * [한국어]
 * memory_stats_t::memlatstat_dram_access - DRAM 접근 시 뱅크별 읽기/쓰기 통계 갱신
 *
 * @mf: DRAM에 접근하는 메모리 요청 패킷
 * @return: 없음 (void)
 *
 * L2 캐시 미스 후 실제 DRAM 접근이 발생할 때 호출된다.
 * gpgpu_memlatency_stat이 활성화된 경우:
 *   - 쓰기 요청:
 *       L2 라이트백(get_sid() >= m_n_shader)은 SM별 bankwrites에서 제외
 *       (L2 → DRAM 라이트백은 L2 소유 요청이므로 SM ID가 가상 ID)
 *       totalbankwrites는 데이터 크기를 dram_atom_size로 나눈 원자 수 단위로 카운트
 *   - 읽기 요청:
 *       bankreads[sm_id][chip][bank]++: SM별 뱅크 읽기 기록
 *       totalbankreads: 원자 수 단위로 카운트
 *   - mem_access_type_stats: 접근 타입별 원자 수 기록
 * 아울러 get_pc()가 유효하면 PTX 파일라인 통계에 DRAM 트래픽 기록 (per-instruction 분석용).
 *
 * 호출 체인:
 *   dram_t::issue_col() → [memlatstat_dram_access()]
 */
void memory_stats_t::memlatstat_dram_access(mem_fetch *mf) {
  unsigned dram_id = mf->get_tlx_addr().chip;  /* [한국어] DRAM 채널(파티션) ID — tlx_addr에서 추출 */
  unsigned bank = mf->get_tlx_addr().bk;       /* [한국어] DRAM 뱅크 ID — tlx_addr에서 추출 */
  if (m_memory_config->gpgpu_memlatency_stat) {  /* [한국어] 레이턴시 통계 수집이 활성화된 경우에만 처리 */
    if (mf->get_is_write()) {  /* [한국어] 쓰기 요청인 경우 */
      if (mf->get_sid() < m_n_shader) {  // do not count L2_writebacks here
        /* [한국어] SM ID가 유효한 경우에만 SM별 bankwrites 기록
         * L2 라이트백의 경우 sid >= m_n_shader (가상 ID)이므로 SM별 통계에서 제외 */
        bankwrites[mf->get_sid()][dram_id][bank]++;  /* [한국어] 해당 SM, 채널, 뱅크의 쓰기 카운터 증가 */
        shader_mem_acc_log(mf->get_sid(), dram_id, bank, 'w');
        /* [한국어] visualizer.h: SM별 메모리 접근 로그 기록 ('w'=쓰기) */
      }
      totalbankwrites[dram_id][bank] +=
          ceil(mf->get_data_size() / m_memory_config->dram_atom_size);
      /* [한국어] 전체 DRAM 채널·뱅크 쓰기 카운터: 요청 크기를 DRAM 원자 크기로 나눈 원자 수 단위
       * ceil(): 부분 원자도 1로 카운트 (올림 연산) */
    } else {  /* [한국어] 읽기 요청인 경우 */
      bankreads[mf->get_sid()][dram_id][bank]++;  /* [한국어] 해당 SM, 채널, 뱅크의 읽기 카운터 증가 */
      shader_mem_acc_log(mf->get_sid(), dram_id, bank, 'r');
      /* [한국어] visualizer.h: SM별 메모리 접근 로그 기록 ('r'=읽기) */
      totalbankreads[dram_id][bank] +=
          ceil(mf->get_data_size() / m_memory_config->dram_atom_size);
      /* [한국어] 전체 DRAM 채널·뱅크 읽기 카운터 원자 수 단위로 갱신 */
    }
    mem_access_type_stats[mf->get_access_type()][dram_id][bank] +=
        ceil(mf->get_data_size() / m_memory_config->dram_atom_size);
    /* [한국어] 메모리 접근 타입(읽기/쓰기/텍스처/상수 등)별 원자 수 통계 갱신
     * get_access_type()는 mem_fetch_type enum 값 반환 */
  }

  if (mf->get_pc() != (unsigned)-1)  /* [한국어] PC가 유효한 경우 (PC=-1은 하드웨어 발생 요청) */
    m_gpu->gpgpu_ctx->stats->ptx_file_line_stats_add_dram_traffic(
        mf->get_pc(), mf->get_data_size());
  /* [한국어] PTX 소스 파일라인별 DRAM 트래픽 통계 기록
   * 특정 PTX 명령어가 얼마나 많은 DRAM 트래픽을 발생시키는지 분석 가능 */
}

/*
 * [한국어]
 * memory_stats_t::memlatstat_icnt2mem_pop - ICNT→MEM 팝 시 NoC 레이턴시 기록
 *
 * @mf: ICNT 출구에서 꺼내진 메모리 요청 패킷
 * @return: 없음 (void)
 *
 * intersim2(NoC)에서 메모리 파티션 측으로 패킷이 도달(팝)할 때 호출된다.
 * mf->get_timestamp()는 SM에서 ICNT에 주입된 사이클이므로,
 * 현재 사이클 - timestamp = SM→MEM 방향 NoC 통과 시간이 된다.
 * 이 값을 icnt2mem_lat_table[24] 히스토그램에 기록하고 최댓값/누적값을 갱신한다.
 *
 * 호출 체인:
 *   memory_partition_unit::cache_cycle() (ICNT 팝 지점) → [memlatstat_icnt2mem_pop()]
 */
void memory_stats_t::memlatstat_icnt2mem_pop(mem_fetch *mf) {
  if (m_memory_config->gpgpu_memlatency_stat) {  /* [한국어] 레이턴시 통계 활성화 시에만 처리 */
    unsigned icnt2mem_latency;  /* [한국어] ICNT→MEM 방향 NoC 레이턴시 (사이클) */
    icnt2mem_latency =
        (m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle) - mf->get_timestamp();
    /* [한국어] SM 발행 사이클부터 MEM 도달까지의 NoC 레이턴시 계산
     * gpu_tot_sim_cycle: 이전 커널 누적, gpu_sim_cycle: 현재 커널 사이클 */
    tot_icnt2mem_latency += icnt2mem_latency;  /* [한국어] ICNT→MEM 레이턴시 누적 합 갱신 */
    icnt2mem_lat_table[LOGB2(icnt2mem_latency)]++;
    /* [한국어] 로그₂ 히스토그램 버킷 증가 (24개 버킷, NoC는 짧은 레이턴시 범위) */
    if (icnt2mem_latency > max_icnt2mem_latency)
      max_icnt2mem_latency = icnt2mem_latency;
    /* [한국어] ICNT→MEM 방향 최대 레이턴시 갱신 */
  }
}

/*
 * [한국어]
 * memory_stats_t::memlatstat_lat_pw - 샘플링 창(Per Window) 레이턴시 통계 갱신 및 리셋
 *
 * @return: 없음 (void)
 *
 * 매 stat_sample_freq 사이클마다(또는 AerialVision 가시화 포인트에서) 호출된다.
 * mf_num_lat_pw(이번 창 완료 수)가 0이 아니고 통계가 활성화된 경우:
 *   1. mf_total_lat에 이번 창 레이턴시 합(mf_tot_lat_pw) 추가
 *   2. num_mfs에 이번 창 완료 수(mf_num_lat_pw) 추가
 *   3. 이번 창 평균 레이턴시(mf_tot_lat_pw/mf_num_lat_pw)를 로그₂ 히스토그램에 기록
 *   4. 창 카운터 mf_tot_lat_pw, mf_num_lat_pw를 0으로 리셋
 *
 * 호출 체인:
 *   gpgpu_sim::cycle() (샘플링 창 경계) → [memlatstat_lat_pw()]
 */
void memory_stats_t::memlatstat_lat_pw() {
  if (mf_num_lat_pw && m_memory_config->gpgpu_memlatency_stat) {
    /* [한국어] 이번 창에 완료된 mem_fetch가 있고 통계 활성화 시 처리 */
    assert(mf_tot_lat_pw);  /* [한국어] 완료 요청이 있으면 레이턴시 합도 0이 아니어야 함 */
    mf_total_lat += mf_tot_lat_pw;    /* [한국어] 전체 누적 레이턴시에 이번 창 합 추가 */
    num_mfs += mf_num_lat_pw;         /* [한국어] 전체 누적 완료 수에 이번 창 수 추가 */
    mf_lat_pw_table[LOGB2(mf_tot_lat_pw / mf_num_lat_pw)]++;
    /* [한국어] 이번 창의 평균 레이턴시(합/수)를 로그₂ 히스토그램에 기록
     * 이 히스토그램은 AerialVision 시각화용 창별 레이턴시 분포를 나타냄 */
    mf_tot_lat_pw = 0;   /* [한국어] 다음 창을 위해 레이턴시 합 리셋 */
    mf_num_lat_pw = 0;   /* [한국어] 다음 창을 위해 완료 수 리셋 */
  }
}

/*
 * [한국어]
 * memory_stats_t::memlatstat_print - 전체 메모리 레이턴시 통계 및 뱅크 접근 통계 출력
 *
 * @n_mem: DRAM 채널(파티션) 수
 * @gpu_mem_n_bk: DRAM 채널당 뱅크 수
 * @return: 없음 (void)
 *
 * 시뮬레이션 종료 후 gpgpu_sim::print_stats()에서 호출된다.
 * SST 모드와 일반 모드로 분기:
 *   SST 모드: 최대/평균 레이턴시만 출력.
 *   일반 모드: 다음 순서로 출력
 *     1. 최대/평균 레이턴시 스칼라 값
 *     2. 히스토그램: mrq/dq/mf/icnt2mem/icnt2sh/mf_lat_pw
 *     3. 행 지역성: max_conc_access2samerow, max_servicetime2samerow, row_access/activate
 *     4. 뱅크 접근 매트릭스: totalbankaccesses/reads/writes + bank skew/chip skew
 *     5. 뱅크별 평균/최대 레이턴시: mf_total_lat_table/mf_max_lat_table
 *   GPU_MEMLATSTAT_MC 플래그가 설정된 경우 추가로:
 *     - MCB(메모리 컨트롤러 뱅크) 접근 히스토그램 및 평균
 *     - position_of_mrq_chosen 히스토그램 및 평균 위치
 *
 * 호출 체인:
 *   gpgpu_sim::print_stats() → [memlatstat_print()]
 */
void memory_stats_t::memlatstat_print(unsigned n_mem, unsigned gpu_mem_n_bk) {
  unsigned i, j, k, l, m;  /* [한국어] 루프 인덱스 및 임시 카운터 변수 */
  unsigned max_bank_accesses, min_bank_accesses, max_chip_accesses,
      min_chip_accesses;
  /* [한국어] 뱅크/채널별 최대·최소 접근 수 (bank skew / chip skew 계산용) */

  if (m_memory_config->SST_mode) {  /* [한국어] SST(SystemC) 연동 모드: 간략 출력 */
    // in SST mode, we just calculate mem latency
    printf("max_mem_SST_latency = %d \n", max_mf_latency);  /* [한국어] SST 모드 최대 레이턴시 */
    if (num_mfs)
      printf("average_mf_SST_latency = %lld \n", mf_total_lat / num_mfs);
    /* [한국어] SST 모드 평균 레이턴시 = 누적합 / 완료 수 */
  } else if (m_memory_config->gpgpu_memlatency_stat) {  /* [한국어] 일반 레이턴시 통계 모드 */
    printf("maxmflatency = %d \n", max_mf_latency);             /* [한국어] 최대 mem_fetch 레이턴시 */
    printf("max_icnt2mem_latency = %d \n", max_icnt2mem_latency); /* [한국어] 최대 ICNT→MEM 레이턴시 */
    printf("maxmrqlatency = %d \n", max_mrq_latency);           /* [한국어] 최대 MRQ 레이턴시 */
    // printf("maxdqlatency = %d \n", max_dq_latency);
    /* [한국어] DQ 레이턴시 출력 (현재 주석 처리) */
    printf("max_icnt2sh_latency = %d \n", max_icnt2sh_latency); /* [한국어] 최대 MEM→SM 레이턴시 */
    if (num_mfs) {  /* [한국어] 완료된 mem_fetch가 있을 때만 평균 출력 (0 나눗셈 방지) */
      printf("averagemflatency = %lld \n", mf_total_lat / num_mfs);             /* [한국어] 평균 mem_fetch 레이턴시 */
      printf("avg_icnt2mem_latency = %lld \n", tot_icnt2mem_latency / num_mfs); /* [한국어] 평균 ICNT→MEM 레이턴시 */
      if (tot_mrq_num)
        printf("avg_mrq_latency = %lld \n", tot_mrq_latency / tot_mrq_num);     /* [한국어] 평균 MRQ 레이턴시 */

      printf("avg_icnt2sh_latency = %lld \n", tot_icnt2sh_latency / num_mfs);   /* [한국어] 평균 MEM→SM 레이턴시 */
    }

    /* [한국어] 히스토그램 출력 섹션: 각 테이블을 탭 구분으로 출력 */
    printf("mrq_lat_table:");
    for (i = 0; i < 32; i++) {
      printf("%d \t", mrq_lat_table[i]);  /* [한국어] i번 버킷: 2^i~2^(i+1) 사이클 범위의 MRQ 수 */
    }
    printf("\n");
    printf("dq_lat_table:");
    for (i = 0; i < 32; i++) {
      printf("%d \t", dq_lat_table[i]);   /* [한국어] DRAM 큐 레이턴시 히스토그램 */
    }
    printf("\n");
    printf("mf_lat_table:");
    for (i = 0; i < 32; i++) {
      printf("%d \t", mf_lat_table[i]);   /* [한국어] mem_fetch 전체 레이턴시 히스토그램 */
    }
    printf("\n");
    printf("icnt2mem_lat_table:");
    for (i = 0; i < 24; i++) {
      printf("%d \t", icnt2mem_lat_table[i]);  /* [한국어] ICNT→MEM NoC 레이턴시 히스토그램 (24버킷) */
    }
    printf("\n");
    printf("icnt2sh_lat_table:");
    for (i = 0; i < 24; i++) {
      printf("%d \t", icnt2sh_lat_table[i]);   /* [한국어] MEM→SM NoC 레이턴시 히스토그램 (24버킷) */
    }
    printf("\n");
    printf("mf_lat_pw_table:");
    for (i = 0; i < 32; i++) {
      printf("%d \t", mf_lat_pw_table[i]);  /* [한국어] 샘플링 창별 평균 레이턴시 히스토그램 */
    }
    printf("\n");

    /*MAXIMUM CONCURRENT ACCESSES TO SAME ROW*/
    printf("maximum concurrent accesses to same row:\n");
    /* [한국어] FRFCFS 스케줄러의 행(row) 히트 연속성 분석: 단일 행에 최대 몇 번 연속 접근했는지 */
    for (i = 0; i < n_mem; i++) {       /* [한국어] 각 DRAM 채널별 출력 */
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {   /* [한국어] 각 뱅크별 출력 */
        printf("%9d ", max_conc_access2samerow[i][j]);  /* [한국어] 채널 i, 뱅크 j의 최대 연속 행 접근 수 */
      }
      printf("\n");
    }

    /*MAXIMUM SERVICE TIME TO SAME ROW*/
    printf("maximum service time to same row:\n");
    /* [한국어] 단일 행에 대한 최대 서비스 시간 (사이클) */
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        printf("%9d ", max_servicetime2samerow[i][j]);  /* [한국어] 채널 i, 뱅크 j의 최대 서비스 시간 */
      }
      printf("\n");
    }

    /*AVERAGE ROW ACCESSES PER ACTIVATE*/
    int total_row_accesses = 0;    /* [한국어] 전체 채널·뱅크에 걸친 행 접근 수 합산 */
    int total_num_activates = 0;   /* [한국어] 전체 채널·뱅크에 걸친 activate 명령 수 합산 */
    printf("average row accesses per activate:\n");
    /* [한국어] 행 지역성(row locality) 분석: activate 1회당 평균 몇 번 행 히트했는지 */
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        total_row_accesses += row_access[i][j];        /* [한국어] 채널 i, 뱅크 j의 행 접근 수 누적 */
        total_num_activates += num_activates[i][j];    /* [한국어] 채널 i, 뱅크 j의 activate 수 누적 */
        printf("%9f ", (float)row_access[i][j] / num_activates[i][j]);
        /* [한국어] 뱅크별 activate당 평균 행 접근 수 = 행 지역성 (높을수록 좋음) */
      }
      printf("\n");
    }
    printf("average row locality = %d/%d = %f\n", total_row_accesses,
           total_num_activates,
           (float)total_row_accesses / total_num_activates);
    /* [한국어] 전체 평균 행 지역성 출력 */

    /*MEMORY ACCESSES*/
    /* [한국어] 뱅크별 전체 접근 수 출력 및 불균형(skew) 분석 */
    k = 0;             /* [한국어] 전체 접근 수 합산 변수 */
    l = 0;             /* [한국어] 현재 뱅크 접근 수 임시 변수 */
    m = 0;             /* [한국어] 현재 채널 접근 수 합산 변수 */
    max_bank_accesses = 0;
    max_chip_accesses = 0;
    min_bank_accesses = 0xFFFFFFFF;  /* [한국어] 초기값 최대 unsigned — 실제 최솟값으로 교체됨 */
    min_chip_accesses = 0xFFFFFFFF;
    printf("number of total memory accesses made:\n");
    for (i = 0; i < n_mem; i++) {   /* [한국어] 각 DRAM 채널 */
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {  /* [한국어] 각 뱅크 */
        l = totalbankaccesses[i][j];         /* [한국어] 채널 i, 뱅크 j의 총 접근 수 */
        if (l < min_bank_accesses) min_bank_accesses = l;  /* [한국어] 뱅크 최솟값 갱신 */
        if (l > max_bank_accesses) max_bank_accesses = l;  /* [한국어] 뱅크 최댓값 갱신 */
        k += l;  /* [한국어] 전체 합산 */
        m += l;  /* [한국어] 현재 채널 합산 */
        printf("%9d ", l);
      }
      if (m < min_chip_accesses) min_chip_accesses = m;  /* [한국어] 채널 최솟값 갱신 */
      if (m > max_chip_accesses) max_chip_accesses = m;  /* [한국어] 채널 최댓값 갱신 */
      m = 0;   /* [한국어] 다음 채널을 위해 채널 합산 변수 리셋 */
      printf("\n");
    }
    printf("total accesses: %d\n", k);  /* [한국어] 전체 접근 수 */
    if (min_bank_accesses)
      printf("bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses,
             (float)max_bank_accesses / min_bank_accesses);
      /* [한국어] bank skew: 가장 많이 접근된 뱅크 / 가장 적게 접근된 뱅크 비율
       * 1.0에 가까울수록 뱅크 부하 균형이 좋음 */
    else
      printf("min_bank_accesses = 0!\n");  /* [한국어] 접근 없는 뱅크가 있음 — 주의 필요 */
    if (min_chip_accesses)
      printf("chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses,
             (float)max_chip_accesses / min_chip_accesses);
      /* [한국어] chip skew: 채널 간 부하 불균형 비율 */
    else
      printf("min_chip_accesses = 0!\n");

    /*READ ACCESSES*/
    /* [한국어] 읽기 접근 수 출력 및 불균형 분석 */
    k = 0;
    l = 0;
    m = 0;
    max_bank_accesses = 0;
    max_chip_accesses = 0;
    min_bank_accesses = 0xFFFFFFFF;
    min_chip_accesses = 0xFFFFFFFF;
    printf("number of total read accesses:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        l = totalbankreads[i][j];  /* [한국어] 채널 i, 뱅크 j의 읽기 접근 수 */
        if (l < min_bank_accesses) min_bank_accesses = l;
        if (l > max_bank_accesses) max_bank_accesses = l;
        k += l;
        m += l;
        printf("%9d ", l);
      }
      if (m < min_chip_accesses) min_chip_accesses = m;
      if (m > max_chip_accesses) max_chip_accesses = m;
      m = 0;
      printf("\n");
    }
    printf("total dram reads = %d\n", k);  /* [한국어] 전체 DRAM 읽기 수 */
    if (min_bank_accesses)
      printf("bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses,
             (float)max_bank_accesses / min_bank_accesses);
    else
      printf("min_bank_accesses = 0!\n");
    if (min_chip_accesses)
      printf("chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses,
             (float)max_chip_accesses / min_chip_accesses);
    else
      printf("min_chip_accesses = 0!\n");

    /*WRITE ACCESSES*/
    /* [한국어] 쓰기 접근 수 출력 및 불균형 분석 (구조는 READ와 동일) */
    k = 0;
    l = 0;
    m = 0;
    max_bank_accesses = 0;
    max_chip_accesses = 0;
    min_bank_accesses = 0xFFFFFFFF;
    min_chip_accesses = 0xFFFFFFFF;
    printf("number of total write accesses:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        l = totalbankwrites[i][j];  /* [한국어] 채널 i, 뱅크 j의 쓰기 접근 수 */
        if (l < min_bank_accesses) min_bank_accesses = l;
        if (l > max_bank_accesses) max_bank_accesses = l;
        k += l;
        m += l;
        printf("%9d ", l);
      }
      if (m < min_chip_accesses) min_chip_accesses = m;
      if (m > max_chip_accesses) max_chip_accesses = m;
      m = 0;
      printf("\n");
    }
    printf("total dram writes = %d\n", k);  /* [한국어] 전체 DRAM 쓰기 수 */
    if (min_bank_accesses)
      printf("bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses,
             (float)max_bank_accesses / min_bank_accesses);
    else
      printf("min_bank_accesses = 0!\n");
    if (min_chip_accesses)
      printf("chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses,
             (float)max_chip_accesses / min_chip_accesses);
    else
      printf("min_chip_accesses = 0!\n");

    /*AVERAGE MF LATENCY PER BANK*/
    printf("average mf latency per bank:\n");
    /* [한국어] 뱅크별 평균 mem_fetch 레이턴시 = mf_total_lat_table / (reads + writes) */
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        k = totalbankwrites[i][j] + totalbankreads[i][j];  /* [한국어] 해당 뱅크의 총 접근 수 */
        if (k)
          printf("%10lld", mf_total_lat_table[i][j] / k);
          /* [한국어] 뱅크 접근이 있는 경우 평균 레이턴시 출력 */
        else
          printf("    none  ");  /* [한국어] 접근이 없는 뱅크는 "none" 출력 */
      }
      printf("\n");
    }

    /*MAXIMUM MF LATENCY PER BANK*/
    printf("maximum mf latency per bank:\n");
    /* [한국어] 뱅크별 최대 mem_fetch 레이턴시 (최악 케이스 분석) */
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        printf("%10d", mf_max_lat_table[i][j]);  /* [한국어] 채널 i, 뱅크 j의 최대 레이턴시 */
      }
      printf("\n");
    }
  }

  if (m_memory_config->gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC) {
    /* [한국어] GPU_MEMLATSTAT_MC 플래그: 메모리 컨트롤러(MCB) 접근 분포 통계 활성화 */
    printf(
        "\nNumber of Memory Banks Accessed per Memory Operation per Warp (from "
        "0):\n");
    /* [한국어] warp 캐시 미스 1회당 몇 개의 메모리 뱅크에 동시 접근했는지 분포 */
    unsigned long long accum_MCBs_accessed = 0;    /* [한국어] 가중 합산 (i * num_MCBs_accessed[i]) */
    unsigned long long tot_mem_ops_per_warp = 0;   /* [한국어] 전체 캐시 미스 warp 수 */
    for (i = 0; i < n_mem * gpu_mem_n_bk; i++) {  /* [한국어] 0~(m_n_mem*nbk-1) 개수 범위 */
      accum_MCBs_accessed += i * num_MCBs_accessed[i];  /* [한국어] i개 MCB 접근 × 발생 횟수 */
      tot_mem_ops_per_warp += num_MCBs_accessed[i];      /* [한국어] 발생 횟수 합산 */
      printf("%d\t", num_MCBs_accessed[i]);              /* [한국어] i개 MCB를 동시 접근한 warp 수 */
    }

    printf(
        "\nAverage # of Memory Banks Accessed per Memory Operation per "
        "Warp=%f\n",
        (float)accum_MCBs_accessed / tot_mem_ops_per_warp);
    /* [한국어] 평균 동시 MCB 접근 수 = 가중합 / 전체 수 */

    // printf("\nAverage Difference Between First and Last Response from Memory
    // System per warp = ");

    printf("\nposition of mrq chosen\n");
    /* [한국어] FRFCFS 스케줄러가 선택한 MRQ의 큐 내 위치 히스토그램 출력 */

    if (!m_memory_config->gpgpu_frfcfs_dram_sched_queue_size)
      j = 1024;   /* [한국어] FRFCFS 큐 크기 미설정 시 기본 1024 사용 */
    else
      j = m_memory_config->gpgpu_frfcfs_dram_sched_queue_size;
      /* [한국어] 설정된 FRFCFS 큐 크기 사용 */
    k = 0;  /* [한국어] 선택된 MRQ 총 수 */
    l = 0;  /* [한국어] 위치 가중 합산 (i * position_of_mrq_chosen[i]) */
    for (i = 0; i < j; i++) {
      printf("%d\t", position_of_mrq_chosen[i]);    /* [한국어] 위치 i에서 선택된 MRQ 수 */
      k += position_of_mrq_chosen[i];               /* [한국어] 전체 선택 수 합산 */
      l += i * position_of_mrq_chosen[i];           /* [한국어] 위치 가중 합산 */
    }
    printf("\n");
    printf("\naverage position of mrq chosen = %f\n", (float)l / k);
    /* [한국어] 평균 선택 위치 = 가중합 / 전체 수
     * 낮을수록 큐 앞쪽(FCFS 의미)이 많이 선택됨, 높을수록 FRFCFS가 뒤쪽 행 히트를 우선 선택 */
  }
}
