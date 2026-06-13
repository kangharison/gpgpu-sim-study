// Copyright (c) 2009-2021,  Tor M. Aamodt, Ahmed El-Shafiey, Tayler
// Hetherington, Vijay Kandiah, Nikos Hardavellas, Mahmoud Khairy, Junrui Pan,
// Timothy G. Rogers The University of British Columbia, Northwestern
// University, Purdue University All rights reserved.
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
 * [한국어 설명] AccelWattch 전력 통계 카운터 구현 (power_stat.cc)
 *
 * === 파일의 역할 ===
 * power_stat.h에 선언된 power_core_stat_t, power_mem_stat_t, power_stat_t 클래스의
 * 메서드를 구현한다. 주요 작업은:
 *   1. 초기화(init): CURRENT 슬롯을 시뮬레이터 원본 통계에 포인터로 연결,
 *                    PREV 슬롯은 calloc으로 독립 배열 할당
 *   2. 저장(save_stats): CURRENT 값을 PREV에 복사하여 다음 샘플링 창의 델타 기준점 갱신
 *   3. 출력(print): 현재 카운터 값을 파일에 덤프
 *   4. 초기화(clear): HW/하이브리드 모드 커널 경계에서 모든 카운터를 0으로 리셋
 *
 * === 전체 아키텍처에서의 위치 ===
 * 데이터 흐름:
 *   shader_core_stats (shader.h) ← SM 파이프라인이 매 사이클 업데이트
 *   cache_stats (gpu-cache.h)    ← 캐시 접근마다 업데이트
 *   dram_t (dram.h)              ← DRAM 명령 스케줄마다 업데이트
 *   [power_stat.cc의 init()]     → 위 카운터들에 CURRENT 포인터 연결
 *   [power_stat.cc의 save_stats()]→ CURRENT를 PREV에 복사
 *   power_interface.cc           ← get_*() 메서드로 델타 쿼리
 * 실행 컨텍스트: 호스트 유저스페이스, 사이클 루프와 동기적으로 동작.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - power_stat.h: 이 파일에서 구현하는 클래스 선언
 *   - shader.h: shader_core_stats, shader_core_config
 *   - dram.h: DRAM 통계 (현재 직접 참조하지 않고 포인터 배열로 관리)
 *   - gpu-sim.h: gpgpu_sim, memory_config
 *   - gpu-cache.h(간접): cache_stats
 *
 * === 주요 함수/구조체 요약 ===
 * power_mem_stat_t::init()       - DRAM/NoC/공유메모리 카운터 배열 calloc 할당 및 연결
 * power_mem_stat_t::save_stats() - 메모리 카운터 CURRENT→PREV 복사
 * power_core_stat_t::init()      - SM 카운터 CURRENT 포인터 연결, PREV 배열 calloc
 * power_core_stat_t::save_stats()- SM 카운터 CURRENT→PREV 복사
 * power_stat_t 생성자            - 두 stat 객체 생성 및 *_kernel, *_execution 필드 초기화
 * power_stat_t::clear()          - HW 모드 커널 경계에서 모든 카운터 0 리셋
 */

#include "power_stat.h"                        /* [한국어] 이 파일에서 구현하는 클래스 선언 */
#include "../abstract_hardware_model.h"        /* [한국어] warp_inst_t, mem_fetch 등 추상 HW 모델 */
#include "../cuda-sim/ptx-stats.h"             /* [한국어] PTX 레벨 통계 (현재 직접 사용 안 함) */
#include "dram.h"                              /* [한국어] dram_t — DRAM 컨트롤러 통계 참조 */
#include "gpu-misc.h"                          /* [한국어] GPU 유틸리티 매크로/함수 */
#include "gpu-sim.h"                           /* [한국어] gpgpu_sim, memory_config, memory_stats_t */
#include "mem_fetch.h"                         /* [한국어] mem_fetch — 메모리 요청 패킷 */
#include "shader.h"                            /* [한국어] shader_core_stats, shader_core_config */
#include "stat-tool.h"                         /* [한국어] LOGB2 등 통계 유틸리티 */
#include "visualizer.h"                        /* [한국어] gzFile 기반 시각화 파일 출력 유틸 */

#include <stdio.h>    /* [한국어] fprintf, printf — 통계 출력 */
#include <stdlib.h>   /* [한국어] calloc — PREV 슬롯 배열 동적 할당 */
#include <string.h>   /* [한국어] memset — 구조체 초기화 */

/*
 * [한국어]
 * power_mem_stat_t::power_mem_stat_t - 메모리 전력 통계 객체 생성자
 *
 * @mem_config: 메모리 구성 파라미터 (DRAM 채널 수 m_n_mem 등)
 * @shdr_config: SM 구성 파라미터 (SM 개수, SIMT 클러스터 수 등)
 * @mem_stats: memory_stats_t — 현재 직접 사용 안 하지만 향후 확장용으로 보관
 * @shdr_stats: shader_core_stats — shmem_access CURRENT 슬롯 연결에 사용
 *
 * 유효성 검증 후 포인터를 저장하고 init()을 호출하여 배열을 할당한다.
 * m_config->m_valid를 assert로 검증 — 설정 미파싱 상태에서의 생성을 방지.
 *
 * 호출 체인:
 *   power_stat_t 생성자 → new power_mem_stat_t() → [이 함수] → init()
 */
power_mem_stat_t::power_mem_stat_t(const memory_config *mem_config,
                                   const shader_core_config *shdr_config,
                                   memory_stats_t *mem_stats,
                                   shader_core_stats *shdr_stats) {
  assert(mem_config->m_valid);    /* [한국어] 메모리 설정이 파싱 완료된 상태인지 검증 */
  m_mem_stats = mem_stats;        /* [한국어] memory_stats_t 포인터 보관 (향후 확장용) */
  m_config = mem_config;          /* [한국어] DRAM 채널 수 등 메모리 구성 파라미터 보관 */
  m_core_stats = shdr_stats;      /* [한국어] SM 카운터 원본 — shmem_access 연결에 사용 */
  m_core_config = shdr_config;    /* [한국어] SM 개수 등 코어 구성 파라미터 보관 */

  init();  /* [한국어] CURRENT/PREV 배열 calloc 할당 및 포인터 연결 */
}

/*
 * [한국어]
 * power_stat_t::clear - HW/하이브리드 모드 커널 경계에서 모든 카운터를 0으로 초기화
 *
 * @return: 없음 (void)
 *
 * calculate_hw_mcpat()의 마지막에 호출되어 다음 커널의 카운터 측정을 위한 깨끗한 상태 보장.
 * CURRENT와 PREV 두 슬롯(NUM_STAT_IDX=2 반복) 모두 초기화하여
 * 이후 save_stats()나 get_*()에서 이전 커널의 값이 영향을 미치지 않게 한다.
 * cache_stats는 clear() 메서드로, SM/DRAM 카운터는 원소별로 0 대입한다.
 *
 * 주의: CURRENT 슬롯은 shader_core_stats의 실제 배열을 가리키므로
 * 이 함수에서 CURRENT[i][j]=0으로 쓰면 시뮬레이터의 실제 카운터도 리셋된다.
 * HW 모드 전용 — Accel-Sim 모드(mcpat_cycle)에서는 save_stats()만 사용한다.
 *
 * 호출 체인:
 *   calculate_hw_mcpat() 말미 → [power_stat_t::clear()]
 */
void power_stat_t::clear() {
  for (unsigned i = 0; i < NUM_STAT_IDX; ++i) {  /* [한국어] CURRENT와 PREV 두 슬롯 모두 초기화 */
    pwr_mem_stat->core_cache_stats[i].clear();    /* [한국어] L1 캐시 통계 객체 내부 카운터 0으로 리셋 */
    pwr_mem_stat->l2_cache_stats[i].clear();      /* [한국어] L2 캐시 통계 객체 내부 카운터 0으로 리셋 */
    for (unsigned j = 0; j < m_config->num_shader(); ++j) {  /* [한국어] 각 SM별 카운터 초기화 */
      pwr_core_stat->m_pipeline_duty_cycle[i][j] = 0;      /* [한국어] SM j의 파이프라인 듀티 사이클 리셋 */
      pwr_core_stat->m_num_decoded_insn[i][j] = 0;         /* [한국어] SM j의 디코드 명령어 수 리셋 */
      pwr_core_stat->m_num_FPdecoded_insn[i][j] = 0;       /* [한국어] FP 명령어 수 리셋 */
      pwr_core_stat->m_num_INTdecoded_insn[i][j] = 0;      /* [한국어] INT 명령어 수 리셋 */
      pwr_core_stat->m_num_storequeued_insn[i][j] = 0;     /* [한국어] 스토어 큐 명령어 수 리셋 */
      pwr_core_stat->m_num_loadqueued_insn[i][j] = 0;      /* [한국어] 로드 큐 명령어 수 리셋 */
      pwr_core_stat->m_num_tex_inst[i][j] = 0;             /* [한국어] 텍스처 명령어 수 리셋 */
      pwr_core_stat->m_num_ialu_acesses[i][j] = 0;         /* [한국어] IALU 접근 수 리셋 */
      pwr_core_stat->m_num_fp_acesses[i][j] = 0;           /* [한국어] SP FPU 접근 수 리셋 */
      pwr_core_stat->m_num_imul_acesses[i][j] = 0;         /* [한국어] IMUL 접근 수 리셋 */
      pwr_core_stat->m_num_imul24_acesses[i][j] = 0;       /* [한국어] IMUL24 접근 수 리셋 */
      pwr_core_stat->m_num_imul32_acesses[i][j] = 0;       /* [한국어] IMUL32 접근 수 리셋 */
      pwr_core_stat->m_num_fpmul_acesses[i][j] = 0;        /* [한국어] FPMUL 접근 수 리셋 */
      pwr_core_stat->m_num_idiv_acesses[i][j] = 0;         /* [한국어] IDIV 접근 수 리셋 */
      pwr_core_stat->m_num_fpdiv_acesses[i][j] = 0;        /* [한국어] FPDIV 접근 수 리셋 */
      pwr_core_stat->m_num_dp_acesses[i][j] = 0;           /* [한국어] DP FPU 접근 수 리셋 */
      pwr_core_stat->m_num_dpmul_acesses[i][j] = 0;        /* [한국어] DPMUL 접근 수 리셋 */
      pwr_core_stat->m_num_dpdiv_acesses[i][j] = 0;        /* [한국어] DPDIV 접근 수 리셋 */
      pwr_core_stat->m_num_tensor_core_acesses[i][j] = 0;  /* [한국어] 텐서 코어 접근 수 리셋 */
      pwr_core_stat->m_num_const_acesses[i][j] = 0;        /* [한국어] 상수 캐시 접근 수 리셋 */
      pwr_core_stat->m_num_tex_acesses[i][j] = 0;          /* [한국어] 텍스처 캐시 접근 수 리셋 */
      pwr_core_stat->m_num_sp_acesses[i][j] = 0;           /* [한국어] SP 파이프라인 접근 수 리셋 */
      pwr_core_stat->m_num_sfu_acesses[i][j] = 0;          /* [한국어] SFU 파이프라인 접근 수 리셋 */
      pwr_core_stat->m_num_sqrt_acesses[i][j] = 0;         /* [한국어] sqrt 접근 수 리셋 */
      pwr_core_stat->m_num_log_acesses[i][j] = 0;          /* [한국어] log 접근 수 리셋 */
      pwr_core_stat->m_num_sin_acesses[i][j] = 0;          /* [한국어] sin 접근 수 리셋 */
      pwr_core_stat->m_num_exp_acesses[i][j] = 0;          /* [한국어] exp 접근 수 리셋 */
      pwr_core_stat->m_num_mem_acesses[i][j] = 0;          /* [한국어] 메모리 파이프라인 접근 수 리셋 */
      pwr_core_stat->m_num_sp_committed[i][j] = 0;         /* [한국어] SP 커밋 명령어 수 리셋 */
      pwr_core_stat->m_num_sfu_committed[i][j] = 0;        /* [한국어] SFU 커밋 명령어 수 리셋 */
      pwr_core_stat->m_num_mem_committed[i][j] = 0;        /* [한국어] MEM 커밋 명령어 수 리셋 */
      pwr_core_stat->m_read_regfile_acesses[i][j] = 0;     /* [한국어] RF 읽기 접근 수 리셋 */
      pwr_core_stat->m_write_regfile_acesses[i][j] = 0;    /* [한국어] RF 쓰기 접근 수 리셋 */
      pwr_core_stat->m_non_rf_operands[i][j] = 0;          /* [한국어] 비RF 오퍼랜드 수 리셋 */
      pwr_core_stat->m_active_sp_lanes[i][j] = 0;          /* [한국어] SP 활성 레인 수 리셋 */
      pwr_core_stat->m_active_sfu_lanes[i][j] = 0;         /* [한국어] SFU 활성 레인 수 리셋 */
      pwr_core_stat->m_active_exu_threads[i][j] = 0;       /* [한국어] 활성 스레드 수 리셋 */
      pwr_core_stat->m_active_exu_warps[i][j] = 0;         /* [한국어] 활성 warp 수 리셋 */
    }
    for (unsigned j = 0; j < m_mem_config->m_n_mem; ++j) { /* [한국어] 각 DRAM 채널 카운터 초기화 */
      pwr_mem_stat->n_rd[i][j] = 0;   /* [한국어] DRAM 읽기 명령 수 리셋 */
      pwr_mem_stat->n_wr[i][j] = 0;   /* [한국어] DRAM 쓰기 명령 수 리셋 */
      pwr_mem_stat->n_pre[i][j] = 0;  /* [한국어] DRAM 프리차지 명령 수 리셋 */
    }
  }
}

/*
 * [한국어]
 * power_mem_stat_t::init - CURRENT/PREV 슬롯 배열 초기화
 *
 * @return: 없음 (void)
 *
 * 1. shmem_access[CURRENT]: shader_core_stats의 실제 배열 포인터에 직접 연결
 *    (이후 SM이 공유메모리를 접근할 때마다 이 배열이 자동으로 업데이트됨)
 *    shmem_access[PREV]: [num_shader] 크기의 새 배열 calloc 할당 (스냅샷용)
 * 2. cache_stats: clear()로 내부 카운터 초기화
 * 3. DRAM 카운터 배열(n_cmd, n_activity, n_nop, n_act, n_pre, n_rd, n_wr, n_wr_WB, n_req):
 *    CURRENT/PREV 두 슬롯 모두 [m_n_mem] 크기로 calloc 할당 (0으로 초기화됨)
 *    주의: 이 DRAM 카운터의 CURRENT는 dram_t가 주기적으로 직접 set_dram_power_stats()로 채움
 * 4. NoC 카운터 배열(n_mem_to_simt, n_simt_to_mem):
 *    [n_simt_clusters] 크기로 calloc 할당
 *
 * 호출 체인:
 *   power_mem_stat_t 생성자 → [init()]
 */
void power_mem_stat_t::init() {
  shmem_access[CURRENT_STAT_IDX] =
      m_core_stats->gpgpu_n_shmem_bank_access;  // Shared memory access
  /* [한국어] CURRENT 슬롯을 shader_core_stats의 공유 메모리 뱅크 접근 카운터 배열에 직접 연결
   * 이 포인터를 통해 SM이 공유 메모리 접근 시 자동으로 이 카운터가 업데이트됨 */
  shmem_access[PREV_STAT_IDX] =
      (unsigned *)calloc(m_core_config->num_shader(), sizeof(unsigned));
  /* [한국어] PREV 슬롯: SM 개수 크기의 스냅샷 배열 calloc 할당 (0으로 초기화됨) */

  for (unsigned i = 0; i < NUM_STAT_IDX; ++i) {  /* [한국어] CURRENT와 PREV 두 슬롯 모두 처리 */
    core_cache_stats[i].clear();  /* [한국어] L1 캐시 통계 cache_stats 객체 내부 카운터 초기화 */
    l2_cache_stats[i].clear();    /* [한국어] L2 캐시 통계 cache_stats 객체 내부 카운터 초기화 */

    /* [한국어] DRAM 컨트롤러 카운터 배열: m_n_mem(DRAM 채널 수) 크기로 calloc 할당
     * calloc은 0으로 초기화되므로 별도 memset 불필요 */
    n_cmd[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));       /* [한국어] 전체 명령 수 배열 */
    n_activity[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));  /* [한국어] 활성 사이클 배열 */
    n_nop[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));       /* [한국어] NOP 명령 수 배열 */
    n_act[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));       /* [한국어] ACT(Activate) 명령 수 배열 */
    n_pre[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));       /* [한국어] PRE(Precharge) 명령 수 배열 */
    n_rd[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));        /* [한국어] READ 명령 수 배열 */
    n_wr[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));        /* [한국어] WRITE 명령 수 배열 */
    n_wr_WB[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));     /* [한국어] Write-Back WRITE 명령 수 배열 */
    n_req[i] = (unsigned *)calloc(m_config->m_n_mem, sizeof(unsigned));       /* [한국어] 전체 요청 수 배열 */

    // Interconnect stats
    /* [한국어] NoC 트래픽 카운터 배열: n_simt_clusters(SIMT 클러스터 수) 크기로 calloc 할당 */
    n_mem_to_simt[i] = (long *)calloc(m_core_config->n_simt_clusters,
                                      sizeof(long));  // Counted at SM
    /* [한국어] 메모리→SIMT 방향 플릿 수 배열 (SM 측에서 집계) */
    n_simt_to_mem[i] = (long *)calloc(m_core_config->n_simt_clusters,
                                      sizeof(long));  // Counted at SM
    /* [한국어] SIMT→메모리 방향 플릿 수 배열 (SM 측에서 집계) */
  }
}

/*
 * [한국어]
 * power_mem_stat_t::save_stats - 현재 카운터를 PREV에 복사하여 다음 샘플링 창 기준점 갱신
 *
 * @return: 없음 (void)
 *
 * CURRENT의 모든 메모리 카운터를 PREV에 복사한다:
 *   - cache_stats: 대입 연산자(=)로 CURRENT→PREV 전체 복사
 *   - shmem_access: SM별 원소 복사
 *   - DRAM 카운터(n_cmd, n_activity, ...): DRAM 채널별 원소 복사
 *   - NoC 카운터(n_simt_to_mem, n_mem_to_simt): SIMT 클러스터별 원소 복사
 * 복사 후 CURRENT는 계속 시뮬레이터가 증분하고, 다음 전력 계산 시
 * CURRENT - PREV로 이번 샘플링 창의 증분을 얻는다.
 *
 * 호출 체인:
 *   power_stat_t::save_stats() → [power_mem_stat_t::save_stats()]
 */
void power_mem_stat_t::save_stats() {
  core_cache_stats[PREV_STAT_IDX] = core_cache_stats[CURRENT_STAT_IDX]; /* [한국어] L1 캐시 통계 스냅샷 저장 */
  l2_cache_stats[PREV_STAT_IDX] = l2_cache_stats[CURRENT_STAT_IDX];     /* [한국어] L2 캐시 통계 스냅샷 저장 */

  for (unsigned i = 0; i < m_core_config->num_shader(); ++i) {  /* [한국어] 각 SM별 공유 메모리 접근 스냅샷 */
    shmem_access[PREV_STAT_IDX][i] =
        shmem_access[CURRENT_STAT_IDX][i];  // Shared memory access
    /* [한국어] SM i의 공유 메모리 접근 수 현재값을 PREV에 저장 */
  }

  for (unsigned i = 0; i < m_config->m_n_mem; ++i) {  /* [한국어] 각 DRAM 채널별 카운터 스냅샷 */
    n_cmd[PREV_STAT_IDX][i] = n_cmd[CURRENT_STAT_IDX][i];             /* [한국어] 전체 명령 수 스냅샷 */
    n_activity[PREV_STAT_IDX][i] = n_activity[CURRENT_STAT_IDX][i];   /* [한국어] 활성 사이클 스냅샷 */
    n_nop[PREV_STAT_IDX][i] = n_nop[CURRENT_STAT_IDX][i];             /* [한국어] NOP 명령 수 스냅샷 */
    n_act[PREV_STAT_IDX][i] = n_act[CURRENT_STAT_IDX][i];             /* [한국어] ACT 명령 수 스냅샷 */
    n_pre[PREV_STAT_IDX][i] = n_pre[CURRENT_STAT_IDX][i];             /* [한국어] PRE 명령 수 스냅샷 */
    n_rd[PREV_STAT_IDX][i] = n_rd[CURRENT_STAT_IDX][i];               /* [한국어] READ 명령 수 스냅샷 */
    n_wr[PREV_STAT_IDX][i] = n_wr[CURRENT_STAT_IDX][i];               /* [한국어] WRITE 명령 수 스냅샷 */
    n_wr_WB[PREV_STAT_IDX][i] = n_wr_WB[CURRENT_STAT_IDX][i];         /* [한국어] WB WRITE 명령 수 스냅샷 */
    n_req[PREV_STAT_IDX][i] = n_req[CURRENT_STAT_IDX][i];             /* [한국어] 전체 요청 수 스냅샷 */
  }

  for (unsigned i = 0; i < m_core_config->n_simt_clusters; i++) {  /* [한국어] 각 SIMT 클러스터별 NoC 스냅샷 */
    n_simt_to_mem[PREV_STAT_IDX][i] =
        n_simt_to_mem[CURRENT_STAT_IDX][i];  // Interconnect
    /* [한국어] SIMT→MEM 방향 플릿 수 스냅샷 */
    n_mem_to_simt[PREV_STAT_IDX][i] =
        n_mem_to_simt[CURRENT_STAT_IDX][i];  // Interconnect
    /* [한국어] MEM→SIMT 방향 플릿 수 스냅샷 */
  }
}

/*
 * [한국어]
 * power_mem_stat_t::visualizer_print - AerialVision 가시화 파일에 메모리 전력 통계 출력 (미구현 스텁)
 *
 * @power_visualizer_file: gzFile — AerialVision이 읽는 압축 출력 파일
 * @return: 없음 (void)
 *
 * 현재는 빈 스텁(stub)으로 아무 동작도 하지 않는다.
 * AccelWattch 전력 시각화가 필요할 경우 이 함수에 per-채널 DRAM 전력 통계 등을
 * gzprintf()로 기록하는 로직을 추가해야 한다.
 *
 * 호출 체인:
 *   power_stat_t::visualizer_print() → [power_mem_stat_t::visualizer_print()]
 */
void power_mem_stat_t::visualizer_print(gzFile power_visualizer_file) {}

/*
 * [한국어]
 * power_mem_stat_t::print - 누적 메모리 전력 통계를 파일에 출력
 *
 * @fout: 출력 대상 파일 포인터 (주로 stdout 또는 결과 로그 파일)
 * @return: 없음 (void)
 *
 * 다음 항목을 출력한다:
 *   - 전체 DRAM 채널에 걸친 총 읽기/쓰기(+WB) 접근 수 합계
 *   - core_cache_stats (L1 캐시 통계) 상세
 *   - l2_cache_stats (L2 캐시 통계) 상세
 * 단일 슬롯 CURRENT를 사용하여 시뮬레이션 전체 누적값을 보여준다.
 * TODO 주석: stream ID 입력 없이는 print_stats()의 일부 기능이 제한됨.
 *
 * 호출 체인:
 *   power_stat_t::print() → [power_mem_stat_t::print()]
 */
void power_mem_stat_t::print(FILE *fout) const {
  fprintf(fout, "\n\n==========Power Metrics -- Memory==========\n"); /* [한국어] 구분선 출력 */
  unsigned total_mem_reads = 0;   /* [한국어] 전체 DRAM 채널의 READ 접근 누적 카운터 */
  unsigned total_mem_writes = 0;  /* [한국어] 전체 DRAM 채널의 WRITE + WB WRITE 접근 누적 카운터 */
  for (unsigned i = 0; i < m_config->m_n_mem; ++i) {  /* [한국어] 각 DRAM 채널(메모리 파티션)별 합산 */
    total_mem_reads += n_rd[CURRENT_STAT_IDX][i];  /* [한국어] i번 채널의 누적 READ 명령 수 더함 */
    total_mem_writes +=
        n_wr[CURRENT_STAT_IDX][i] + n_wr_WB[CURRENT_STAT_IDX][i];
    /* [한국어] i번 채널의 일반 WRITE + 라이트백(WB) WRITE 명령 수 합산 */
  }
  fprintf(fout, "Total memory controller accesses: %u\n",
          total_mem_reads + total_mem_writes);  /* [한국어] 읽기+쓰기 전체 접근 수 출력 */
  fprintf(fout, "Total memory controller reads: %u\n", total_mem_reads);   /* [한국어] 읽기 총계 출력 */
  fprintf(fout, "Total memory controller writes: %u\n", total_mem_writes); /* [한국어] 쓰기 총계 출력 */
  // TODO: print_stats(require stream ID input)
  /* [한국어] stream ID 없이는 print_stats()의 per-stream 세분화가 불가 — 향후 개선 필요 */
  fprintf(fout, "Core cache stats:\n");           /* [한국어] L1 캐시 통계 헤더 */
  core_cache_stats->print_stats(fout, -1);        /* [한국어] -1은 stream ID 미지정(모든 스트림 합산) */
  fprintf(fout, "L2 cache stats:\n");             /* [한국어] L2 캐시 통계 헤더 */
  l2_cache_stats->print_stats(fout, -1);          /* [한국어] L2 캐시 통계 출력 */
}

/*
 * [한국어]
 * power_core_stat_t 생성자 - 코어 전력 통계 객체 초기화
 *
 * @shader_config: shader_core_config — SM 개수·실행 유닛 구성 등 코어 구성 파라미터
 * @core_stats: shader_core_stats — SM이 실제로 업데이트하는 카운터 배열들을 보유
 * @return: 없음 (생성자)
 *
 * 1. shader_config->m_valid를 assert()로 검증하여 초기화 전 설정 유효성 확인.
 * 2. this 포인터를 shader_core_power_stats_pod*로 캐스팅하여 memset()으로 POD 부분 전체를 0으로 초기화.
 *    이로써 CURRENT/PREV 포인터 배열이 모두 nullptr로 시작.
 * 3. m_config, m_core_stats 저장 후 init()을 호출하여 포인터 배열을 실제 카운터에 연결.
 *
 * 호출 체인:
 *   power_stat_t 생성자 → new power_core_stat_t() → [power_core_stat_t 생성자] → init()
 */
power_core_stat_t::power_core_stat_t(const shader_core_config *shader_config,
                                     shader_core_stats *core_stats) {
  assert(shader_config->m_valid);  /* [한국어] 설정 객체가 유효하게 초기화되었는지 검증 */
  m_config = shader_config;        /* [한국어] SM 구성 참조 저장 — num_shader(), 실행 유닛 수 등에 사용 */
  shader_core_power_stats_pod *pod = this;  /* [한국어] 상속된 POD 구조체 부분에 접근하기 위한 포인터 캐스트 */
  memset(pod, 0, sizeof(shader_core_power_stats_pod));
  /* [한국어] CURRENT/PREV 포인터 배열 및 모든 POD 필드를 0(nullptr)으로 초기화
   * init()에서 실제 주소를 할당하기 전 안전한 초기 상태 설정 */
  m_core_stats = core_stats;  /* [한국어] SM이 직접 업데이트하는 카운터 배열 참조 저장 */

  init();  /* [한국어] CURRENT 슬롯을 core_stats의 실제 배열에 연결, PREV 슬롯을 calloc 할당 */
}

/*
 * [한국어]
 * power_core_stat_t::visualizer_print - AerialVision 가시화 파일에 코어 전력 통계 출력 (미구현 스텁)
 *
 * @visualizer_file: gzFile — AerialVision이 읽는 압축 출력 파일
 * @return: 없음 (void)
 *
 * 현재는 빈 스텁(stub)으로 아무 동작도 하지 않는다.
 * 향후 SM별 파이프라인 duty cycle, ALU/FP/SFU 접근 카운터 등을 gzprintf()로 기록하는
 * 로직 추가가 필요하다.
 *
 * 호출 체인:
 *   power_stat_t::visualizer_print() → [power_core_stat_t::visualizer_print()]
 */
void power_core_stat_t::visualizer_print(gzFile visualizer_file) {}

/*
 * [한국어]
 * power_core_stat_t::print - 누적 코어별 전력 통계를 파일에 출력
 *
 * @fout: 출력 대상 파일 포인터 (주로 stdout 또는 결과 로그 파일)
 * @return: 없음 (void)
 *
 * 모든 SM(코어)에 대해 반복하며 다음 항목을 출력한다:
 *   - 파이프라인 duty cycle (0.0~1.0 실수)
 *   - 디코딩된 총 명령어 수, FP/INT 분류
 *   - 로드/스토어 큐에 올라간 명령어 수
 *   - 각 실행 유닛(IALU, FP, DP, IMUL, IDIV, SFU 계열, Tensor Core, TEX, MEM) 접근 수
 *   - SP/SFU/MEM 커밋 명령어 수
 *   - 레지스터 파일 읽기/쓰기, 비RF 오퍼랜드 수
 * CURRENT 슬롯의 누적값을 사용하므로 시뮬레이션 전체 통계를 나타낸다.
 *
 * 호출 체인:
 *   power_stat_t::print() → [power_core_stat_t::print()]
 */
void power_core_stat_t::print(FILE *fout) {
  // per core statistics
  fprintf(fout, "Power Metrics: \n");  /* [한국어] 섹션 헤더 출력 */
  for (unsigned i = 0; i < m_config->num_shader(); i++) {  /* [한국어] SM(shader core)별 반복 */
    fprintf(fout, "core %u:\n", i);  /* [한국어] SM 인덱스 출력 */
    fprintf(fout, "\tpipeline duty cycle =%f\n",
            m_pipeline_duty_cycle[CURRENT_STAT_IDX][i]);  /* [한국어] 파이프라인 평균 활성 비율 */
    fprintf(fout, "\tTotal Deocded Instructions=%u\n",
            m_num_decoded_insn[CURRENT_STAT_IDX][i]);     /* [한국어] 총 디코딩 명령어 수 */
    fprintf(fout, "\tTotal FP Deocded Instructions=%u\n",
            m_num_FPdecoded_insn[CURRENT_STAT_IDX][i]);   /* [한국어] FP 명령어 디코딩 수 */
    fprintf(fout, "\tTotal INT Deocded Instructions=%u\n",
            m_num_INTdecoded_insn[CURRENT_STAT_IDX][i]);  /* [한국어] 정수 명령어 디코딩 수 */
    fprintf(fout, "\tTotal LOAD Queued Instructions=%u\n",
            m_num_loadqueued_insn[CURRENT_STAT_IDX][i]);  /* [한국어] 로드 큐 진입 명령어 수 */
    fprintf(fout, "\tTotal STORE Queued Instructions=%u\n",
            m_num_storequeued_insn[CURRENT_STAT_IDX][i]); /* [한국어] 스토어 큐 진입 명령어 수 */
    fprintf(fout, "\tTotal IALU Acesses=%f\n",
            m_num_ialu_acesses[CURRENT_STAT_IDX][i]);     /* [한국어] 정수 ALU 접근 횟수 */
    fprintf(fout, "\tTotal FP Acesses=%f\n",
            m_num_fp_acesses[CURRENT_STAT_IDX][i]);       /* [한국어] FP 연산 접근 횟수 */
    fprintf(fout, "\tTotal DP Acesses=%f\n",
            m_num_dp_acesses[CURRENT_STAT_IDX][i]);       /* [한국어] DP(배정밀도) 연산 접근 횟수 */
    fprintf(fout, "\tTotal IMUL Acesses=%f\n",
            m_num_imul_acesses[CURRENT_STAT_IDX][i]);     /* [한국어] 정수 곱셈(IMUL) 접근 횟수 */
    fprintf(fout, "\tTotal IMUL24 Acesses=%f\n",
            m_num_imul24_acesses[CURRENT_STAT_IDX][i]);   /* [한국어] 24비트 정수 곱셈 접근 횟수 */
    fprintf(fout, "\tTotal IMUL32 Acesses=%f\n",
            m_num_imul32_acesses[CURRENT_STAT_IDX][i]);   /* [한국어] 32비트 정수 곱셈 접근 횟수 */
    fprintf(fout, "\tTotal IDIV Acesses=%f\n",
            m_num_idiv_acesses[CURRENT_STAT_IDX][i]);     /* [한국어] 정수 나눗셈 접근 횟수 */
    fprintf(fout, "\tTotal FPMUL Acesses=%f\n",
            m_num_fpmul_acesses[CURRENT_STAT_IDX][i]);    /* [한국어] FP 곱셈 접근 횟수 */
    fprintf(fout, "\tTotal DPMUL Acesses=%f\n",
            m_num_dpmul_acesses[CURRENT_STAT_IDX][i]);    /* [한국어] DP 곱셈 접근 횟수 */
    fprintf(fout, "\tTotal SQRT Acesses=%f\n",
            m_num_sqrt_acesses[CURRENT_STAT_IDX][i]);     /* [한국어] 제곱근(sqrt) SFU 접근 횟수 */
    fprintf(fout, "\tTotal LOG Acesses=%f\n",
            m_num_log_acesses[CURRENT_STAT_IDX][i]);      /* [한국어] 로그(log) SFU 접근 횟수 */
    fprintf(fout, "\tTotal SIN Acesses=%f\n",
            m_num_sin_acesses[CURRENT_STAT_IDX][i]);      /* [한국어] 사인(sin) SFU 접근 횟수 */
    fprintf(fout, "\tTotal EXP Acesses=%f\n",
            m_num_exp_acesses[CURRENT_STAT_IDX][i]);      /* [한국어] 지수(exp) SFU 접근 횟수 */
    fprintf(fout, "\tTotal FPDIV Acesses=%f\n",
            m_num_fpdiv_acesses[CURRENT_STAT_IDX][i]);    /* [한국어] FP 나눗셈 접근 횟수 */
    fprintf(fout, "\tTotal DPDIV Acesses=%f\n",
            m_num_dpdiv_acesses[CURRENT_STAT_IDX][i]);    /* [한국어] DP 나눗셈 접근 횟수 */
    fprintf(fout, "\tTotal TENSOR Acesses=%f\n",
            m_num_tensor_core_acesses[CURRENT_STAT_IDX][i]); /* [한국어] 텐서 코어 접근 횟수 */
    fprintf(fout, "\tTotal CONST Acesses=%f\n",
            m_num_const_acesses[CURRENT_STAT_IDX][i]);    /* [한국어] 상수 캐시 접근 횟수 */
    fprintf(fout, "\tTotal TEX Acesses=%f\n",
            m_num_tex_acesses[CURRENT_STAT_IDX][i]);      /* [한국어] 텍스처 캐시 접근 횟수 */
    fprintf(fout, "\tTotal SFU Acesses=%f\n",
            m_num_sfu_acesses[CURRENT_STAT_IDX][i]);      /* [한국어] SFU 전체 접근 횟수 */
    fprintf(fout, "\tTotal SP Acesses=%f\n",
            m_num_sp_acesses[CURRENT_STAT_IDX][i]);       /* [한국어] SP(단정밀도) 접근 횟수 */
    fprintf(fout, "\tTotal MEM Acesses=%f\n",
            m_num_mem_acesses[CURRENT_STAT_IDX][i]);      /* [한국어] 메모리 파이프라인 접근 횟수 */
    fprintf(fout, "\tTotal SFU Commissions=%u\n",
            m_num_sfu_committed[CURRENT_STAT_IDX][i]);    /* [한국어] SFU 커밋(완료) 명령어 수 */
    fprintf(fout, "\tTotal SP Commissions=%u\n",
            m_num_sp_committed[CURRENT_STAT_IDX][i]);     /* [한국어] SP 커밋 명령어 수 */
    fprintf(fout, "\tTotal MEM Commissions=%u\n",
            m_num_mem_committed[CURRENT_STAT_IDX][i]);    /* [한국어] MEM 커밋 명령어 수 */
    fprintf(fout, "\tTotal REG Reads=%u\n",
            m_read_regfile_acesses[CURRENT_STAT_IDX][i]); /* [한국어] 레지스터 파일 읽기 접근 수 */
    fprintf(fout, "\tTotal REG Writes=%u\n",
            m_write_regfile_acesses[CURRENT_STAT_IDX][i]); /* [한국어] 레지스터 파일 쓰기 접근 수 */
    fprintf(fout, "\tTotal NON REG=%u\n",
            m_non_rf_operands[CURRENT_STAT_IDX][i]);      /* [한국어] 레지스터 파일을 거치지 않은 오퍼랜드 수 */
  }
}
/*
 * [한국어]
 * power_core_stat_t::init - CURRENT 슬롯을 shader_core_stats에 연결하고 PREV 슬롯을 calloc 할당
 *
 * @return: 없음 (void)
 *
 * 이 함수는 두 단계로 나뉜다:
 *
 * 1단계 (CURRENT 연결): shader_core_power_stats_pod의 각 CURRENT_STAT_IDX 슬롯을
 *   shader_core_stats (m_core_stats)가 소유한 실제 배열 포인터에 직접 연결.
 *   이 연결 후 SM(shader core)이 명령을 실행하며 m_core_stats의 배열을 업데이트하면
 *   CURRENT 슬롯도 자동으로 최신 값을 반영하게 된다 (포인터 공유이므로).
 *
 * 2단계 (PREV 할당): PREV_STAT_IDX 슬롯에는 SM 개수(num_shader()) 크기의 새 배열을
 *   calloc으로 할당하여 스냅샷 저장 공간을 확보.
 *   이 PREV 배열은 save_stats()가 주기적으로 CURRENT를 복사해 채우고,
 *   mcpat_cycle()이 CURRENT - PREV로 증분을 계산할 때 기준점으로 사용된다.
 *
 * 호출 체인:
 *   power_core_stat_t 생성자 → [init()]
 */
void power_core_stat_t::init() {
  /* [한국어] 1단계: 각 CURRENT 슬롯을 m_core_stats의 실제 배열에 직접 연결 (포인터 대입)
   * 이후 SM 파이프라인이 이 배열을 직접 업데이트하면 CURRENT 슬롯도 자동 반영 */
  m_pipeline_duty_cycle[CURRENT_STAT_IDX] = m_core_stats->m_pipeline_duty_cycle; /* [한국어] 파이프라인 duty cycle 배열 연결 */
  m_num_decoded_insn[CURRENT_STAT_IDX] = m_core_stats->m_num_decoded_insn;       /* [한국어] 디코딩 명령어 수 배열 연결 */
  m_num_FPdecoded_insn[CURRENT_STAT_IDX] = m_core_stats->m_num_FPdecoded_insn;   /* [한국어] FP 디코딩 수 배열 연결 */
  m_num_INTdecoded_insn[CURRENT_STAT_IDX] = m_core_stats->m_num_INTdecoded_insn; /* [한국어] INT 디코딩 수 배열 연결 */
  m_num_storequeued_insn[CURRENT_STAT_IDX] =
      m_core_stats->m_num_storequeued_insn;  /* [한국어] 스토어 큐 명령어 수 배열 연결 */
  m_num_loadqueued_insn[CURRENT_STAT_IDX] = m_core_stats->m_num_loadqueued_insn; /* [한국어] 로드 큐 명령어 수 배열 연결 */
  m_num_ialu_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_ialu_acesses;       /* [한국어] IALU 접근 배열 연결 */
  m_num_fp_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_fp_acesses;           /* [한국어] FP 접근 배열 연결 */
  m_num_imul_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_imul_acesses;       /* [한국어] IMUL 접근 배열 연결 */
  m_num_imul24_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_imul24_acesses;   /* [한국어] IMUL24 접근 배열 연결 */
  m_num_imul32_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_imul32_acesses;   /* [한국어] IMUL32 접근 배열 연결 */
  m_num_fpmul_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_fpmul_acesses;     /* [한국어] FP 곱셈 접근 배열 연결 */
  m_num_idiv_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_idiv_acesses;       /* [한국어] 정수 나눗셈 접근 배열 연결 */
  m_num_fpdiv_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_fpdiv_acesses;     /* [한국어] FP 나눗셈 접근 배열 연결 */
  m_num_dp_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_dp_acesses;           /* [한국어] DP 접근 배열 연결 */
  m_num_dpmul_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_dpmul_acesses;     /* [한국어] DP 곱셈 접근 배열 연결 */
  m_num_dpdiv_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_dpdiv_acesses;     /* [한국어] DP 나눗셈 접근 배열 연결 */
  m_num_sp_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_sp_acesses;           /* [한국어] SP 파이프라인 접근 배열 연결 */
  m_num_sfu_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_sfu_acesses;         /* [한국어] SFU 접근 배열 연결 */
  m_num_sqrt_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_sqrt_acesses;       /* [한국어] sqrt SFU 접근 배열 연결 */
  m_num_log_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_log_acesses;         /* [한국어] log SFU 접근 배열 연결 */
  m_num_sin_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_sin_acesses;         /* [한국어] sin SFU 접근 배열 연결 */
  m_num_exp_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_exp_acesses;         /* [한국어] exp SFU 접근 배열 연결 */
  m_num_tensor_core_acesses[CURRENT_STAT_IDX] =
      m_core_stats->m_num_tensor_core_acesses;  /* [한국어] 텐서 코어 접근 배열 연결 */
  m_num_const_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_const_acesses;     /* [한국어] 상수 캐시 접근 배열 연결 */
  m_num_tex_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_tex_acesses;         /* [한국어] 텍스처 캐시 접근 배열 연결 */
  m_num_mem_acesses[CURRENT_STAT_IDX] = m_core_stats->m_num_mem_acesses;         /* [한국어] 메모리 파이프라인 접근 배열 연결 */
  m_num_sp_committed[CURRENT_STAT_IDX] = m_core_stats->m_num_sp_committed;       /* [한국어] SP 커밋 배열 연결 */
  m_num_sfu_committed[CURRENT_STAT_IDX] = m_core_stats->m_num_sfu_committed;     /* [한국어] SFU 커밋 배열 연결 */
  m_num_mem_committed[CURRENT_STAT_IDX] = m_core_stats->m_num_mem_committed;     /* [한국어] MEM 커밋 배열 연결 */
  m_read_regfile_acesses[CURRENT_STAT_IDX] =
      m_core_stats->m_read_regfile_acesses;    /* [한국어] 레지스터 파일 읽기 접근 배열 연결 */
  m_write_regfile_acesses[CURRENT_STAT_IDX] =
      m_core_stats->m_write_regfile_acesses;   /* [한국어] 레지스터 파일 쓰기 접근 배열 연결 */
  m_non_rf_operands[CURRENT_STAT_IDX] = m_core_stats->m_non_rf_operands;         /* [한국어] 비RF 오퍼랜드 배열 연결 */
  m_active_sp_lanes[CURRENT_STAT_IDX] = m_core_stats->m_active_sp_lanes;         /* [한국어] SP 활성 레인 배열 연결 */
  m_active_sfu_lanes[CURRENT_STAT_IDX] = m_core_stats->m_active_sfu_lanes;       /* [한국어] SFU 활성 레인 배열 연결 */
  m_active_exu_threads[CURRENT_STAT_IDX] = m_core_stats->m_active_exu_threads;   /* [한국어] 활성 스레드 수 배열 연결 */
  m_active_exu_warps[CURRENT_STAT_IDX] = m_core_stats->m_active_exu_warps;       /* [한국어] 활성 warp 수 배열 연결 */
  m_num_tex_inst[CURRENT_STAT_IDX] = m_core_stats->m_num_tex_inst;               /* [한국어] 텍스처 명령어 수 배열 연결 */

  /* [한국어] 2단계: 각 PREV 슬롯을 SM 개수만큼 calloc으로 새 배열 할당
   * calloc은 0으로 초기화하므로 처음 save_stats() 전에는 "기준점=0"이 된다 */
  m_pipeline_duty_cycle[PREV_STAT_IDX] =
      (float *)calloc(m_config->num_shader(), sizeof(float));  /* [한국어] float 배열: duty cycle 스냅샷용 */
  m_num_decoded_insn[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));         /* [한국어] 디코딩 수 스냅샷 배열 */
  m_num_FPdecoded_insn[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));         /* [한국어] FP 디코딩 수 스냅샷 */
  m_num_INTdecoded_insn[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));         /* [한국어] INT 디코딩 수 스냅샷 */
  m_num_storequeued_insn[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));         /* [한국어] 스토어 큐 수 스냅샷 */
  m_num_loadqueued_insn[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));         /* [한국어] 로드 큐 수 스냅샷 */
  m_num_tex_inst[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));         /* [한국어] 텍스처 명령어 수 스냅샷 */

  m_num_ialu_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] IALU 접근 스냅샷(double) */
  m_num_fp_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] FP 접근 스냅샷 */
  m_num_imul_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] IMUL 접근 스냅샷 */
  m_num_imul24_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] IMUL24 접근 스냅샷 */
  m_num_imul32_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] IMUL32 접근 스냅샷 */
  m_num_fpmul_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] FP 곱셈 접근 스냅샷 */
  m_num_idiv_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] 정수 나눗셈 접근 스냅샷 */
  m_num_fpdiv_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] FP 나눗셈 접근 스냅샷 */
  m_num_dp_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] DP 접근 스냅샷 */
  m_num_dpmul_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] DP 곱셈 접근 스냅샷 */
  m_num_dpdiv_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] DP 나눗셈 접근 스냅샷 */
  m_num_tensor_core_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] 텐서 코어 접근 스냅샷 */
  m_num_const_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] 상수 캐시 접근 스냅샷 */
  m_num_tex_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] 텍스처 캐시 접근 스냅샷 */
  m_num_sp_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] SP 접근 스냅샷 */
  m_num_sfu_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] SFU 접근 스냅샷 */
  m_num_sqrt_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] sqrt SFU 접근 스냅샷 */
  m_num_log_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] log SFU 접근 스냅샷 */
  m_num_sin_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] sin SFU 접근 스냅샷 */
  m_num_exp_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] exp SFU 접근 스냅샷 */
  m_num_mem_acesses[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] MEM 파이프라인 접근 스냅샷 */
  m_num_sp_committed[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));         /* [한국어] SP 커밋 수 스냅샷 */
  m_num_sfu_committed[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));         /* [한국어] SFU 커밋 수 스냅샷 */
  m_num_mem_committed[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));         /* [한국어] MEM 커밋 수 스냅샷 */
  m_read_regfile_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));         /* [한국어] RF 읽기 접근 스냅샷 */
  m_write_regfile_acesses[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));         /* [한국어] RF 쓰기 접근 스냅샷 */
  m_non_rf_operands[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));         /* [한국어] 비RF 오퍼랜드 스냅샷 */
  m_active_sp_lanes[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));         /* [한국어] SP 활성 레인 스냅샷 */
  m_active_sfu_lanes[PREV_STAT_IDX] =
      (unsigned *)calloc(m_config->num_shader(), sizeof(unsigned));         /* [한국어] SFU 활성 레인 스냅샷 */
  m_active_exu_threads[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] 활성 스레드 수 스냅샷(double) */
  m_active_exu_warps[PREV_STAT_IDX] =
      (double *)calloc(m_config->num_shader(), sizeof(double));             /* [한국어] 활성 warp 수 스냅샷(double) */
}

/*
 * [한국어]
 * power_core_stat_t::save_stats - SM별 CURRENT 카운터를 PREV에 복사하여 다음 샘플링 창 기준점 갱신
 *
 * @return: 없음 (void)
 *
 * 모든 SM(인덱스 0..num_shader()-1)에 대해 CURRENT 슬롯의 값을 PREV 슬롯에 복사한다.
 * 복사 대상: m_pipeline_duty_cycle, m_num_decoded_insn, m_num_FPdecoded_insn,
 *            m_num_INTdecoded_insn, m_num_storequeued/loadqueued_insn,
 *            모든 ALU/SFU/DP/텐서/메모리 접근 카운터, m_num_*_committed,
 *            m_read/write_regfile_acesses, m_non_rf_operands,
 *            m_active_sp/sfu_lanes, m_active_exu_threads/warps
 * 이 스냅샷을 기준으로 다음 mcpat_cycle() 호출 시 CURRENT - PREV = 이번 창 증분.
 *
 * 호출 체인:
 *   power_stat_t::save_stats() → [power_core_stat_t::save_stats()]
 */
void power_core_stat_t::save_stats() {
  for (unsigned i = 0; i < m_config->num_shader(); ++i) {  /* [한국어] 모든 SM별 반복 */
    m_pipeline_duty_cycle[PREV_STAT_IDX][i] =
        m_pipeline_duty_cycle[CURRENT_STAT_IDX][i];         /* [한국어] duty cycle 스냅샷 */
    m_num_decoded_insn[PREV_STAT_IDX][i] =
        m_num_decoded_insn[CURRENT_STAT_IDX][i];             /* [한국어] 디코딩 수 스냅샷 */
    m_num_FPdecoded_insn[PREV_STAT_IDX][i] =
        m_num_FPdecoded_insn[CURRENT_STAT_IDX][i];           /* [한국어] FP 디코딩 수 스냅샷 */
    m_num_INTdecoded_insn[PREV_STAT_IDX][i] =
        m_num_INTdecoded_insn[CURRENT_STAT_IDX][i];          /* [한국어] INT 디코딩 수 스냅샷 */
    m_num_storequeued_insn[PREV_STAT_IDX][i] =
        m_num_storequeued_insn[CURRENT_STAT_IDX][i];         /* [한국어] 스토어 큐 수 스냅샷 */
    m_num_loadqueued_insn[PREV_STAT_IDX][i] =
        m_num_loadqueued_insn[CURRENT_STAT_IDX][i];          /* [한국어] 로드 큐 수 스냅샷 */
    m_num_ialu_acesses[PREV_STAT_IDX][i] =
        m_num_ialu_acesses[CURRENT_STAT_IDX][i];             /* [한국어] IALU 접근 스냅샷 */
    m_num_fp_acesses[PREV_STAT_IDX][i] = m_num_fp_acesses[CURRENT_STAT_IDX][i]; /* [한국어] FP 접근 스냅샷 */
    m_num_tex_inst[PREV_STAT_IDX][i] = m_num_tex_inst[CURRENT_STAT_IDX][i];     /* [한국어] 텍스처 명령어 수 스냅샷 */
    m_num_imul_acesses[PREV_STAT_IDX][i] =
        m_num_imul_acesses[CURRENT_STAT_IDX][i];             /* [한국어] IMUL 접근 스냅샷 */
    m_num_imul24_acesses[PREV_STAT_IDX][i] =
        m_num_imul24_acesses[CURRENT_STAT_IDX][i];           /* [한국어] IMUL24 접근 스냅샷 */
    m_num_imul32_acesses[PREV_STAT_IDX][i] =
        m_num_imul32_acesses[CURRENT_STAT_IDX][i];           /* [한국어] IMUL32 접근 스냅샷 */
    m_num_fpmul_acesses[PREV_STAT_IDX][i] =
        m_num_fpmul_acesses[CURRENT_STAT_IDX][i];            /* [한국어] FP 곱셈 접근 스냅샷 */
    m_num_idiv_acesses[PREV_STAT_IDX][i] =
        m_num_idiv_acesses[CURRENT_STAT_IDX][i];             /* [한국어] 정수 나눗셈 스냅샷 */
    m_num_fpdiv_acesses[PREV_STAT_IDX][i] =
        m_num_fpdiv_acesses[CURRENT_STAT_IDX][i];            /* [한국어] FP 나눗셈 스냅샷 */
    m_num_sp_acesses[PREV_STAT_IDX][i] = m_num_sp_acesses[CURRENT_STAT_IDX][i]; /* [한국어] SP 접근 스냅샷 */
    m_num_sfu_acesses[PREV_STAT_IDX][i] =
        m_num_sfu_acesses[CURRENT_STAT_IDX][i];              /* [한국어] SFU 접근 스냅샷 */
    m_num_sqrt_acesses[PREV_STAT_IDX][i] =
        m_num_sqrt_acesses[CURRENT_STAT_IDX][i];             /* [한국어] sqrt SFU 접근 스냅샷 */
    m_num_log_acesses[PREV_STAT_IDX][i] =
        m_num_log_acesses[CURRENT_STAT_IDX][i];              /* [한국어] log SFU 접근 스냅샷 */
    m_num_sin_acesses[PREV_STAT_IDX][i] =
        m_num_sin_acesses[CURRENT_STAT_IDX][i];              /* [한국어] sin SFU 접근 스냅샷 */
    m_num_exp_acesses[PREV_STAT_IDX][i] =
        m_num_exp_acesses[CURRENT_STAT_IDX][i];              /* [한국어] exp SFU 접근 스냅샷 */
    m_num_dp_acesses[PREV_STAT_IDX][i] = m_num_dp_acesses[CURRENT_STAT_IDX][i]; /* [한국어] DP 접근 스냅샷 */
    m_num_dpmul_acesses[PREV_STAT_IDX][i] =
        m_num_dpmul_acesses[CURRENT_STAT_IDX][i];            /* [한국어] DP 곱셈 접근 스냅샷 */
    m_num_dpdiv_acesses[PREV_STAT_IDX][i] =
        m_num_dpdiv_acesses[CURRENT_STAT_IDX][i];            /* [한국어] DP 나눗셈 스냅샷 */
    m_num_tensor_core_acesses[PREV_STAT_IDX][i] =
        m_num_tensor_core_acesses[CURRENT_STAT_IDX][i];      /* [한국어] 텐서 코어 접근 스냅샷 */
    m_num_const_acesses[PREV_STAT_IDX][i] =
        m_num_const_acesses[CURRENT_STAT_IDX][i];            /* [한국어] 상수 캐시 접근 스냅샷 */
    m_num_tex_acesses[PREV_STAT_IDX][i] =
        m_num_tex_acesses[CURRENT_STAT_IDX][i];              /* [한국어] 텍스처 캐시 접근 스냅샷 */
    m_num_mem_acesses[PREV_STAT_IDX][i] =
        m_num_mem_acesses[CURRENT_STAT_IDX][i];              /* [한국어] MEM 파이프라인 접근 스냅샷 */
    m_num_sp_committed[PREV_STAT_IDX][i] =
        m_num_sp_committed[CURRENT_STAT_IDX][i];             /* [한국어] SP 커밋 수 스냅샷 */
    m_num_sfu_committed[PREV_STAT_IDX][i] =
        m_num_sfu_committed[CURRENT_STAT_IDX][i];            /* [한국어] SFU 커밋 수 스냅샷 */
    m_num_mem_committed[PREV_STAT_IDX][i] =
        m_num_mem_committed[CURRENT_STAT_IDX][i];            /* [한국어] MEM 커밋 수 스냅샷 */
    m_read_regfile_acesses[PREV_STAT_IDX][i] =
        m_read_regfile_acesses[CURRENT_STAT_IDX][i];         /* [한국어] RF 읽기 접근 스냅샷 */
    m_write_regfile_acesses[PREV_STAT_IDX][i] =
        m_write_regfile_acesses[CURRENT_STAT_IDX][i];        /* [한국어] RF 쓰기 접근 스냅샷 */
    m_non_rf_operands[PREV_STAT_IDX][i] =
        m_non_rf_operands[CURRENT_STAT_IDX][i];              /* [한국어] 비RF 오퍼랜드 스냅샷 */
    m_active_sp_lanes[PREV_STAT_IDX][i] =
        m_active_sp_lanes[CURRENT_STAT_IDX][i];              /* [한국어] SP 활성 레인 스냅샷 */
    m_active_sfu_lanes[PREV_STAT_IDX][i] =
        m_active_sfu_lanes[CURRENT_STAT_IDX][i];             /* [한국어] SFU 활성 레인 스냅샷 */
    m_active_exu_threads[PREV_STAT_IDX][i] =
        m_active_exu_threads[CURRENT_STAT_IDX][i];           /* [한국어] 활성 스레드 수 스냅샷 */
    m_active_exu_warps[PREV_STAT_IDX][i] =
        m_active_exu_warps[CURRENT_STAT_IDX][i];             /* [한국어] 활성 warp 수 스냅샷 */
  }
}

/*
 * [한국어]
 * power_stat_t 생성자 - 최상위 전력 통계 객체 초기화
 *
 * @shader_config: shader_core_config — SM 구성 정보 (num_shader 등)
 * @average_pipeline_duty_cycle: float* — 시뮬레이터 전역 평균 파이프라인 활성 비율 포인터
 * @active_sms: float* — 현재 실행 중인 활성 SM 수 포인터
 * @shader_stats: shader_core_stats — SM이 직접 업데이트하는 카운터 배열 소유자
 * @mem_config: memory_config — 메모리 서브시스템 구성 (m_n_mem, DRAM 채널 수 등)
 * @memory_stats: memory_stats_t — 메모리 레이턴시 통계 객체
 * @return: 없음 (생성자)
 *
 * 1. shader_config->m_valid, mem_config->m_valid를 assert()로 검증.
 * 2. pwr_core_stat: power_core_stat_t 동적 생성 (SM별 실행 유닛 카운터 관리)
 * 3. pwr_mem_stat: power_mem_stat_t 동적 생성 (캐시/DRAM/NoC 카운터 관리)
 * 4. m_average_pipeline_duty_cycle, m_active_sms 포인터 저장
 *    (gpu_sim이 관리하는 전역 float 변수를 직접 가리킨다)
 * 5. *_kernel 필드(커널 기준점) 전체 0 초기화
 *    calculate_hw_mcpat() 첫 호출 전 기준점이 없으므로 0에서 시작
 * 6. *_execution 필드(멀티커널 누적) 전체 0 초기화
 *    aggregate_power_stats=true 모드에서 커널마다 증분이 더해진다
 *
 * 호출 체인:
 *   gpgpu_sim 생성자 → new power_stat_t() → [power_stat_t 생성자]
 *                                          → new power_core_stat_t()
 *                                          → new power_mem_stat_t()
 */
power_stat_t::power_stat_t(const shader_core_config *shader_config,
                           float *average_pipeline_duty_cycle,
                           float *active_sms, shader_core_stats *shader_stats,
                           const memory_config *mem_config,
                           memory_stats_t *memory_stats) {
  assert(shader_config->m_valid);   /* [한국어] SM 설정 유효성 검증 */
  assert(mem_config->m_valid);      /* [한국어] 메모리 설정 유효성 검증 */
  pwr_core_stat = new power_core_stat_t(shader_config, shader_stats);
  /* [한국어] SM별 실행 유닛(ALU/SFU/DP/텐서 코어 등) 카운터 관리 객체 생성 */
  pwr_mem_stat = new power_mem_stat_t(mem_config, shader_config, memory_stats,
                                      shader_stats);
  /* [한국어] 캐시(L1/L2)/DRAM/NoC 전력 카운터 관리 객체 생성 */
  m_average_pipeline_duty_cycle = average_pipeline_duty_cycle;
  /* [한국어] 전체 SM에 걸친 평균 파이프라인 활성 비율 포인터 저장
   * gpu_sim이 매 사이클 업데이트하는 전역 float 변수를 가리킨다 */
  m_active_sms = active_sms;   /* [한국어] 현재 실행 중인 활성 SM 수 포인터 저장 */
  m_config = shader_config;    /* [한국어] SM 구성 참조 저장 */
  m_mem_config = mem_config;   /* [한국어] 메모리 구성 참조 저장 */

  /* [한국어] *_kernel 필드: HW/하이브리드 모드에서 커널 시작 시점의 스냅샷 기준점
   * 처음엔 모두 0 — 첫 커널 완료 시 calculate_hw_mcpat()이 현재 값으로 업데이트함 */
  l1r_hits_kernel = 0;          /* [한국어] L1 읽기 히트 커널 기준점 초기화 */
  l1r_misses_kernel = 0;        /* [한국어] L1 읽기 미스 커널 기준점 초기화 */
  l1w_hits_kernel = 0;          /* [한국어] L1 쓰기 히트 커널 기준점 초기화 */
  l1w_misses_kernel = 0;        /* [한국어] L1 쓰기 미스 커널 기준점 초기화 */
  shared_accesses_kernel = 0;   /* [한국어] 공유 메모리 접근 커널 기준점 초기화 */
  cc_accesses_kernel = 0;       /* [한국어] 상수/텍스처 캐시 접근 커널 기준점 초기화 */
  dram_rd_kernel = 0;           /* [한국어] DRAM 읽기 커널 기준점 초기화 */
  dram_wr_kernel = 0;           /* [한국어] DRAM 쓰기 커널 기준점 초기화 */
  dram_pre_kernel = 0;          /* [한국어] DRAM 프리차지 커널 기준점 초기화 */
  l1i_hits_kernel = 0;          /* [한국어] L1 명령어 캐시 히트 커널 기준점 초기화 */
  l1i_misses_kernel = 0;        /* [한국어] L1 명령어 캐시 미스 커널 기준점 초기화 */
  l2r_hits_kernel = 0;          /* [한국어] L2 읽기 히트 커널 기준점 초기화 */
  l2r_misses_kernel = 0;        /* [한국어] L2 읽기 미스 커널 기준점 초기화 */
  l2w_hits_kernel = 0;          /* [한국어] L2 쓰기 히트 커널 기준점 초기화 */
  l2w_misses_kernel = 0;        /* [한국어] L2 쓰기 미스 커널 기준점 초기화 */
  noc_tr_kernel = 0;            /* [한국어] NoC 전송 플릿 커널 기준점 초기화 */
  noc_rc_kernel = 0;            /* [한국어] NoC 수신 플릿 커널 기준점 초기화 */

  /* [한국어] *_execution 필드: aggregate_power_stats=true 모드에서 다중 커널 누적값
   * 시뮬레이션 시작 시점에 모두 0으로 초기화, 커널 완료마다 증분이 더해진다 */
  tot_inst_execution = 0;             /* [한국어] 전체 명령어 수 누적 초기화 */
  tot_int_inst_execution = 0;         /* [한국어] 정수 명령어 수 누적 초기화 */
  tot_fp_inst_execution = 0;          /* [한국어] FP 명령어 수 누적 초기화 */
  commited_inst_execution = 0;        /* [한국어] 커밋 명령어 수 누적 초기화 */
  ialu_acc_execution = 0;             /* [한국어] IALU 접근 누적 초기화 */
  imul24_acc_execution = 0;           /* [한국어] IMUL24 접근 누적 초기화 */
  imul32_acc_execution = 0;           /* [한국어] IMUL32 접근 누적 초기화 */
  imul_acc_execution = 0;             /* [한국어] IMUL 접근 누적 초기화 */
  idiv_acc_execution = 0;             /* [한국어] IDIV 접근 누적 초기화 */
  dp_acc_execution = 0;               /* [한국어] DP 접근 누적 초기화 */
  dpmul_acc_execution = 0;            /* [한국어] DP 곱셈 접근 누적 초기화 */
  dpdiv_acc_execution = 0;            /* [한국어] DP 나눗셈 접근 누적 초기화 */
  fp_acc_execution = 0;               /* [한국어] FP 접근 누적 초기화 */
  fpmul_acc_execution = 0;            /* [한국어] FP 곱셈 접근 누적 초기화 */
  fpdiv_acc_execution = 0;            /* [한국어] FP 나눗셈 접근 누적 초기화 */
  sqrt_acc_execution = 0;             /* [한국어] sqrt SFU 접근 누적 초기화 */
  log_acc_execution = 0;              /* [한국어] log SFU 접근 누적 초기화 */
  sin_acc_execution = 0;              /* [한국어] sin SFU 접근 누적 초기화 */
  exp_acc_execution = 0;              /* [한국어] exp SFU 접근 누적 초기화 */
  tensor_acc_execution = 0;           /* [한국어] 텐서 코어 접근 누적 초기화 */
  tex_acc_execution = 0;              /* [한국어] 텍스처 캐시 접근 누적 초기화 */
  tot_fpu_acc_execution = 0;          /* [한국어] FPU 전체 접근 누적 초기화 */
  tot_sfu_acc_execution = 0;          /* [한국어] SFU 전체 접근 누적 초기화 */
  tot_threads_acc_execution = 0;      /* [한국어] 활성 스레드 수 누적 초기화 */
  tot_warps_acc_execution = 0;        /* [한국어] 활성 warp 수 누적 초기화 */
  sp_active_lanes_execution = 0;      /* [한국어] SP 활성 레인 누적 초기화 */
  sfu_active_lanes_execution = 0;     /* [한국어] SFU 활성 레인 누적 초기화 */
}

/*
 * [한국어]
 * power_stat_t::visualizer_print - AerialVision 가시화 파일에 전체 전력 통계 출력 (위임)
 *
 * @visualizer_file: gzFile — AerialVision 압축 출력 파일
 * @return: 없음 (void)
 *
 * pwr_core_stat와 pwr_mem_stat 각각의 visualizer_print()로 위임한다.
 * 현재 두 함수 모두 빈 스텁이므로 실제 출력은 없다.
 *
 * 호출 체인:
 *   gpgpu_sim::visualizer_printout() → [power_stat_t::visualizer_print()]
 *     → power_core_stat_t::visualizer_print()
 *     → power_mem_stat_t::visualizer_print()
 */
void power_stat_t::visualizer_print(gzFile visualizer_file) {
  pwr_core_stat->visualizer_print(visualizer_file);  /* [한국어] SM 코어 전력 통계 가시화 출력 (현재 스텁) */
  pwr_mem_stat->visualizer_print(visualizer_file);   /* [한국어] 메모리 전력 통계 가시화 출력 (현재 스텁) */
}

/*
 * [한국어]
 * power_stat_t::print - 전체 전력 통계를 파일에 출력 (위임)
 *
 * @fout: 출력 대상 파일 포인터
 * @return: 없음 (void)
 *
 * 다음 순서로 출력한다:
 *   1. 전역 평균 파이프라인 duty cycle (*m_average_pipeline_duty_cycle)
 *   2. pwr_core_stat->print(): SM별 실행 유닛 카운터
 *   3. pwr_mem_stat->print(): 메모리/캐시/DRAM 카운터
 *
 * 호출 체인:
 *   gpu_sim::print_stats() → [power_stat_t::print()]
 *     → power_core_stat_t::print()
 *     → power_mem_stat_t::print()
 */
void power_stat_t::print(FILE *fout) const {
  fprintf(fout, "average_pipeline_duty_cycle=%f\n",
          *m_average_pipeline_duty_cycle);  /* [한국어] 전역 평균 파이프라인 활성 비율 출력 */
  pwr_core_stat->print(fout);               /* [한국어] SM별 실행 유닛 카운터 상세 출력 */
  pwr_mem_stat->print(fout);                /* [한국어] 메모리/캐시/DRAM 카운터 상세 출력 */
}
