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
 * [한국어 설명] AccelWattch 전력 통계 카운터 헤더 (power_stat.h)
 *
 * === 파일의 역할 ===
 * GPGPU-Sim 타이밍 시뮬레이터가 매 사이클마다 업데이트하는 마이크로아키텍처 카운터들을
 * AccelWattch 전력 계산 목적으로 집계하는 자료구조와 클래스를 정의한다.
 * 핵심 아이디어는 "현재값(CURRENT)"과 "이전 샘플값(PREV)"을 쌍으로 유지하여,
 * 두 값의 차이(델타)로 샘플링 창(stat_sample_freq 사이클) 동안의 카운터 증분을 얻는 것이다.
 * 이 델타들이 power_interface.cc를 통해 McPAT으로 전달된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 데이터 흐름:
 *   shader.cc (SM 파이프라인 실행) → shader_core_stats 카운터 직접 업데이트
 *   gpu-cache.cc / l2cache.cc     → cache_stats 업데이트
 *   dram.cc                       → DRAM 명령 카운터 업데이트
 *   [power_stat.h/cc]              → 위 카운터들에 대한 포인터 유지 (CURRENT/PREV 쌍)
 *   power_interface.cc             → 델타값 쿼리 → McPAT 전력 계산
 * 실행 컨텍스트: 호스트 유저스페이스. 카운터는 사이클 루프에서 갱신, 읽기는 샘플링 주기에서.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - gpu-sim.h: gpgpu_sim_config, memory_config, memory_stats_t
 *   - mem_latency_stat.h: memory_stats_t (L2/DRAM 통계의 원본)
 *   - shader.h(간접): shader_core_stats, shader_core_config (SM 카운터 원본)
 *   - gpu-cache.h(간접): cache_stats (L1/L2 캐시 히트/미스 통계)
 * 이 파일에 의존하는 모듈:
 *   - power_interface.cc: power_stat_t의 get_*() 메서드로 카운터 쿼리
 *   - gpu-sim.cc: power_stat_t 객체 생성 및 소유
 *
 * === 주요 함수/구조체 요약 ===
 * stat_idx (enum)          - CURRENT/PREV 슬롯 인덱스 (배열 [2]의 인덱스로 사용)
 * shader_core_power_stats_pod - SM별 실행 카운터 포인터 배열 (두 슬롯 모두)
 * power_core_stat_t        - shader_core_stats 포인터를 보관하는 코어 전력 통계 클래스
 * mem_power_stats_pod      - 캐시/DRAM/NoC 통계 포인터 배열
 * power_mem_stat_t         - 메모리 관련 전력 통계 클래스
 * power_stat_t             - 코어+메모리 통계를 통합하고 get_*() 쿼리 메서드를 제공하는 최상위 클래스
 */

#ifndef POWER_STAT_H
#define POWER_STAT_H

#include <stdio.h>  /* [한국어] fprintf, FILE* — 통계 출력용 */
#include <zlib.h>   /* [한국어] gzFile — 압축된 시각화 파일 출력용 */
#include "gpu-sim.h"           /* [한국어] gpgpu_sim_config, memory_config — 설정 파라미터 */
#include "mem_latency_stat.h"  /* [한국어] memory_stats_t — 메모리 레이턴시/접근 통계 원본 */

/*
 * [한국어]
 * stat_idx - 전력 카운터 배열의 슬롯 인덱스
 *
 * 각 카운터는 [CURRENT_STAT_IDX]와 [PREV_STAT_IDX] 두 슬롯으로 관리된다.
 * - CURRENT_STAT_IDX: 시뮬레이터 원본 통계와 포인터가 연결된 "현재 누적값" 슬롯
 * - PREV_STAT_IDX: save_stats() 호출 시 현재값을 복사해 저장하는 "이전 샘플 기준점" 슬롯
 * 델타 = CURRENT - PREV로 이번 샘플링 창의 증분 카운터를 계산한다.
 */
typedef enum _stat_idx {
  CURRENT_STAT_IDX = 0,  // Current activity count
  /* [한국어] 현재 누적 카운터 — 시뮬레이터가 매 사이클 갱신하는 원본 배열에 대한 포인터 */
  PREV_STAT_IDX,         // Previous sample activity count
  /* [한국어] 이전 샘플 기준점 — save_stats() 시 CURRENT를 복사하여 저장 */
  NUM_STAT_IDX           // Total number of samples
  /* [한국어] 슬롯 총 개수 = 2 — 배열 크기 선언에 사용 */
} stat_idx;

/*
 * [한국어]
 * shader_core_power_stats_pod - SM별 마이크로아키텍처 전력 카운터 포인터 집합 (POD 구조체)
 *
 * 각 필드는 NUM_STAT_IDX(=2)개의 포인터 배열: [CURRENT_STAT_IDX]와 [PREV_STAT_IDX].
 * CURRENT 슬롯 포인터는 shader_core_stats의 실제 배열을 직접 가리킨다 (복사 없음).
 * PREV 슬롯 포인터는 calloc()으로 별도 할당된 스냅샷 배열을 가리킨다.
 * 배열의 인덱스는 SM(shader core) ID — 각 SM의 카운터가 독립적으로 저장된다.
 * 전력 계산 시 모든 SM의 합산이 필요하며, power_stat_t의 get_*() 메서드가 이를 수행한다.
 *
 * 동기화: 이 구조체의 CURRENT 포인터들은 시뮬레이터 사이클 루프에서 직접 업데이트되고
 * 전력 계산 시(stat_sample_freq 사이클마다) 단일 스레드에서 읽히므로 별도 락 불필요.
 */
struct shader_core_power_stats_pod {
  // [CURRENT_STAT_IDX] = CURRENT_STAT_IDX stat, [PREV_STAT_IDX] = last reading
  float *m_pipeline_duty_cycle[NUM_STAT_IDX];
  /* [한국어] 파이프라인 듀티 사이클: SM 파이프라인이 활성(비유휴)이었던 비율
   * 설정자: shader_core_ctx::cycle()에서 매 사이클 증분 (0~1 범위 float)
   * 읽는 자: power_stat_t::save_stats() 후 mcpat_cycle()에서 샘플 평균 계산
   * 값 범위: 0.0~1.0 (0=완전 유휴, 1=완전 활성)
   * 동기화: 단일 SM 스레드에서만 쓰기, 전력 계산은 사이클 루프 외부에서 읽기 */

  unsigned *m_num_decoded_insn[NUM_STAT_IDX];  // number of instructions
                                               // committed by this shader core
  /* [한국어] 이 SM이 디코드/커밋한 전체 명령어 수 (INT+FP+MEM 모두 포함)
   * 설정자: shader_core_ctx::execute()에서 명령어 발행 시마다 증분
   * 읽는 자: power_stat_t::get_total_inst() — 전체 SM 합산
   * 값 범위: 0~(MAX_UINT32) 단조 증가 카운터 (리셋 없음)
   * 동기화: SM당 독립 배열 원소 — SM 간 공유 없음 */

  unsigned
      *m_num_FPdecoded_insn[NUM_STAT_IDX];  // number of instructions committed
                                            // by this shader core
  /* [한국어] 이 SM이 커밋한 부동소수점(FP, SP+DP) 명령어 수
   * 설정자: shader_core_ctx::execute()에서 FP 명령어 발행 시 증분
   * 읽는 자: power_stat_t::get_total_fp_inst() — SP/DP 유닛 전력 스케일링에 사용
   * 값 범위: 0~m_num_decoded_insn 이하 (FP 비율에 따라 변동)
   * 동기화: SM당 독립, 별도 락 불필요 */

  unsigned
      *m_num_INTdecoded_insn[NUM_STAT_IDX];  // number of instructions committed
                                             // by this shader core
  /* [한국어] 이 SM이 커밋한 정수(INT) 명령어 수
   * 설정자: shader_core_ctx::execute()에서 INT 명령어 발행 시 증분
   * 읽는 자: power_stat_t::get_total_int_inst() — 정수 ALU 전력 스케일링에 사용
   * 값 범위: 0~m_num_decoded_insn 이하
   * 동기화: SM당 독립 */

  unsigned *m_num_storequeued_insn[NUM_STAT_IDX];
  /* [한국어] 스토어(store) 큐에 삽입된 명령어 수 (메모리 쓰기 요청)
   * 설정자: 스토어 명령어가 LSQ(Load-Store Queue)에 enqueue될 때 증분
   * 읽는 자: power_stat_t::get_total_store_inst() — 메모리 인터페이스 전력 모델
   * 값 범위: 0~전체 스토어 명령어 수
   * 동기화: SM당 독립 */

  unsigned *m_num_loadqueued_insn[NUM_STAT_IDX];
  /* [한국어] 로드(load) 큐에 삽입된 명령어 수 (메모리 읽기 요청)
   * 설정자: 로드 명령어가 LSQ에 enqueue될 때 증분
   * 읽는 자: power_stat_t::get_total_load_inst() — 메모리 인터페이스 전력 모델
   * 값 범위: 0~전체 로드 명령어 수
   * 동기화: SM당 독립 */

  unsigned *m_num_tex_inst[NUM_STAT_IDX];
  /* [한국어] 텍스처 명령어 수 (텍스처 페치 전용 명령어 카운트)
   * 설정자: 텍스처 명령어 발행 시 증분
   * 읽는 자: power_stat_t::get_tex_inst() — 텍스처 유닛 전력 모델
   * 값 범위: 0~전체 텍스처 명령어 수
   * 동기화: SM당 독립 */

  double *m_num_ialu_acesses[NUM_STAT_IDX];
  /* [한국어] 정수 ALU(Integer ALU) 접근 횟수 (IALU 파이프라인 실행 카운터)
   * 설정자: IALU 파이프라인에서 명령어 완료 시 증분 (double로 저장)
   * 읽는 자: power_stat_t::get_ialu_accessess() — McPAT INT ALU 전력 입력
   * 값 범위: 0~전체 INT 명령어 수 (FP보다 매우 크거나 작을 수 있음)
   * 동기화: SM당 독립 */

  double *m_num_fp_acesses[NUM_STAT_IDX];
  /* [한국어] 단정밀도 FP(SP) 유닛 접근 횟수
   * 설정자: SP(Single-Precision) 파이프라인 명령어 완료 시 증분
   * 읽는 자: power_stat_t::get_fp_accessess() — McPAT FP ALU 전력 입력
   * 값 범위: 0~전체 SP 명령어 수
   * 동기화: SM당 독립 */

  double *m_num_imul_acesses[NUM_STAT_IDX];
  /* [한국어] 범용 정수 곱셈(Integer Multiply) 접근 횟수
   * 설정자: IMUL 명령어 완료 시 증분 (SFU 또는 별도 IMUL 유닛)
   * 읽는 자: power_stat_t::get_intmul_accessess() — McPAT INT 곱셈 전력
   * 값 범위: 0~전체 IMUL 명령어 수
   * 동기화: SM당 독립 */

  double *m_num_imul32_acesses[NUM_STAT_IDX];
  /* [한국어] 32비트 정수 곱셈(IMUL32) 접근 횟수
   * 설정자: IMUL32 명령어 완료 시 증분
   * 읽는 자: power_stat_t::get_intmul32_accessess()
   * 값 범위: 0~전체 IMUL32 명령어 수
   * 동기화: SM당 독립 */

  double *m_num_imul24_acesses[NUM_STAT_IDX];
  /* [한국어] 24비트 정수 곱셈(IMUL24) 접근 횟수 — Tesla 아키텍처 전용 24비트 곱셈
   * 설정자: IMUL24 명령어 완료 시 증분
   * 읽는 자: power_stat_t::get_intmul24_accessess()
   * 값 범위: 0~전체 IMUL24 명령어 수 (구형 PTX에서 주로 발생)
   * 동기화: SM당 독립 */

  double *m_num_fpmul_acesses[NUM_STAT_IDX];
  /* [한국어] 단정밀도 FP 곱셈(FP MUL) 접근 횟수
   * 설정자: FPMUL 명령어 완료 시 증분
   * 읽는 자: power_stat_t::get_fpmul_accessess()
   * 값 범위: 0~전체 FPMUL 명령어 수
   * 동기화: SM당 독립 */

  double *m_num_idiv_acesses[NUM_STAT_IDX];
  /* [한국어] 정수 나눗셈(Integer Divide) 접근 횟수 — SFU에서 처리되는 고비용 연산
   * 설정자: IDIV 명령어 완료 시 증분
   * 읽는 자: power_stat_t::get_intdiv_accessess() — McPAT SFU 전력의 일부
   * 값 범위: 0~전체 IDIV 명령어 수 (일반적으로 매우 적음)
   * 동기화: SM당 독립 */

  double *m_num_fpdiv_acesses[NUM_STAT_IDX];
  /* [한국어] 단정밀도 FP 나눗셈(FP DIV) 접근 횟수
   * 설정자: FPDIV 명령어 완료 시 증분 (SFU에서 처리)
   * 읽는 자: power_stat_t::get_fpdiv_accessess()
   * 값 범위: 0~전체 FPDIV 명령어 수
   * 동기화: SM당 독립 */

  double *m_num_dp_acesses[NUM_STAT_IDX];
  /* [한국어] 배정밀도 FP(DP) 유닛 접근 횟수
   * 설정자: DP(Double-Precision) 명령어 완료 시 증분
   * 읽는 자: power_stat_t::get_dp_accessess() — McPAT DP FPU 전력 입력
   * 값 범위: 0~전체 DP 명령어 수
   * 동기화: SM당 독립 */

  double *m_num_dpmul_acesses[NUM_STAT_IDX];
  /* [한국어] 배정밀도 FP 곱셈(DP MUL) 접근 횟수
   * 설정자: DPMUL 명령어 완료 시 증분
   * 읽는 자: power_stat_t::get_dpmul_accessess()
   * 동기화: SM당 독립 */

  double *m_num_dpdiv_acesses[NUM_STAT_IDX];
  /* [한국어] 배정밀도 FP 나눗셈(DP DIV) 접근 횟수
   * 설정자: DPDIV 명령어 완료 시 증분
   * 읽는 자: power_stat_t::get_dpdiv_accessess()
   * 동기화: SM당 독립 */

  double *m_num_sp_acesses[NUM_STAT_IDX];
  /* [한국어] SP(단정밀도) 파이프라인 전체 접근 횟수
   * 설정자: SP 파이프라인 명령어(FP add/mul/fma 등) 완료 시 증분
   * 읽는 자: power_stat_t::get_sp_accessess() — McPAT SP FPU 전력
   * 동기화: SM당 독립 */

  double *m_num_sfu_acesses[NUM_STAT_IDX];
  /* [한국어] SFU(Special Function Unit: 초월함수, 특수 연산) 전체 접근 횟수
   * 설정자: SFU 파이프라인 명령어(sqrt/sin/log/exp 등) 완료 시 증분
   * 읽는 자: power_stat_t::get_sfu_accessess() — McPAT SFU 전력
   * 동기화: SM당 독립 */

  double *m_num_sqrt_acesses[NUM_STAT_IDX];
  /* [한국어] sqrt(제곱근) SFU 접근 횟수
   * 설정자: SQRT 명령어 완료 시 증분 (SFU에서 처리)
   * 읽는 자: power_stat_t::get_sqrt_accessess() — AccelWattch 초월함수 전력
   * 동기화: SM당 독립 */

  double *m_num_log_acesses[NUM_STAT_IDX];
  /* [한국어] log(로그) SFU 접근 횟수
   * 읽는 자: power_stat_t::get_log_accessess()
   * 동기화: SM당 독립 */

  double *m_num_sin_acesses[NUM_STAT_IDX];
  /* [한국어] sin(사인) SFU 접근 횟수
   * 읽는 자: power_stat_t::get_sin_accessess()
   * 동기화: SM당 독립 */

  double *m_num_exp_acesses[NUM_STAT_IDX];
  /* [한국어] exp(지수) SFU 접근 횟수
   * 읽는 자: power_stat_t::get_exp_accessess()
   * 동기화: SM당 독립 */

  double *m_num_tensor_core_acesses[NUM_STAT_IDX];
  /* [한국어] Tensor Core 접근 횟수 (Volta/Turing 이상 행렬 곱셈 전용 유닛)
   * 설정자: HMMA/IMMA 명령어 완료 시 증분
   * 읽는 자: power_stat_t::get_tensor_accessess() — McPAT 텐서 코어 전력
   * 값 범위: 0~전체 HMMA/IMMA 명령어 수 (비Tensor 워크로드에서는 0)
   * 동기화: SM당 독립 */

  double *m_num_const_acesses[NUM_STAT_IDX];
  /* [한국어] 상수 캐시(Constant Cache) 접근 횟수
   * 설정자: 상수 메모리 접근(CONST_ACC_R) 시 증분
   * 읽는 자: power_stat_t::get_const_accessess() — McPAT 상수 캐시 전력
   * 동기화: SM당 독립 */

  double *m_num_tex_acesses[NUM_STAT_IDX];
  /* [한국어] 텍스처 캐시 접근 횟수 (텍스처 페치 연산)
   * 설정자: 텍스처 명령어 실행 시 증분
   * 읽는 자: power_stat_t::get_tex_accessess() — McPAT 텍스처 캐시 전력
   * 동기화: SM당 독립 */

  double *m_num_mem_acesses[NUM_STAT_IDX];
  /* [한국어] 메모리 파이프라인(LD/ST unit) 접근 횟수
   * 설정자: 메모리 명령어(LD/ST)가 실행 파이프라인에서 처리될 때 증분
   * 읽는 자: power_stat_t::get_mem_accessess() — 메모리 인터페이스 전력
   * 동기화: SM당 독립 */

  unsigned *m_num_sp_committed[NUM_STAT_IDX];
  /* [한국어] SP 파이프라인에서 커밋(완료)된 명령어 수
   * 설정자: SP 파이프라인 명령어가 write-back 단계 완료 시 증분
   * 읽는 자: power_stat_t::get_sp_committed_inst(), get_committed_inst()
   * 동기화: SM당 독립 */

  unsigned *m_num_sfu_committed[NUM_STAT_IDX];
  /* [한국어] SFU 파이프라인에서 커밋된 명령어 수
   * 설정자: SFU 명령어 완료 시 증분
   * 읽는 자: power_stat_t::get_sfu_committed_inst(), get_committed_inst()
   * 동기화: SM당 독립 */

  unsigned *m_num_mem_committed[NUM_STAT_IDX];
  /* [한국어] 메모리 파이프라인에서 커밋된 명령어 수
   * 설정자: 메모리 명령어 완료 시 증분
   * 읽는 자: power_stat_t::get_mem_committed_inst(), get_committed_inst()
   * 동기화: SM당 독립 */

  unsigned *m_active_sp_lanes[NUM_STAT_IDX];
  /* [한국어] SP 파이프라인의 활성 레인 수 합산 (warp 실행 시 활성 스레드 수)
   * 설정자: SP 파이프라인 명령어 실행 시 active mask의 popcount 누적
   * 읽는 자: power_stat_t::get_sp_active_lanes() — 평균 활성 레인율 계산
   * 값 범위: 0~32*실행된_SP_명령어_수 (레인당 카운트)
   * 동기화: SM당 독립 */

  unsigned *m_active_sfu_lanes[NUM_STAT_IDX];
  /* [한국어] SFU 파이프라인의 활성 레인 수 합산
   * 읽는 자: power_stat_t::get_sfu_active_lanes() — SFU 레인 활성화율
   * 동기화: SM당 독립 */

  double *m_active_exu_threads[NUM_STAT_IDX];
  /* [한국어] 실행 유닛(EXU)에서 처리된 활성 스레드 수 합산
   * 설정자: 명령어 실행 시 해당 명령의 active thread 수 누적
   * 읽는 자: power_stat_t::get_active_threads() — warp당 평균 활성 스레드 수
   * 동기화: SM당 독립 */

  double *m_active_exu_warps[NUM_STAT_IDX];
  /* [한국어] 실행 유닛에서 처리된 warp 수 합산
   * 설정자: 명령어가 실행될 때마다 1 증분 (warp 단위)
   * 읽는 자: power_stat_t::get_active_threads() — warp당 평균 스레드 분모로 사용
   * 동기화: SM당 독립 */

  unsigned *m_read_regfile_acesses[NUM_STAT_IDX];
  /* [한국어] 레지스터 파일(RF) 읽기 접근 횟수
   * 설정자: operand collector가 RF에서 오퍼랜드를 읽을 때마다 증분
   * 읽는 자: power_stat_t::get_regfile_reads() — McPAT RF 읽기 전력
   * 동기화: SM당 독립 */

  unsigned *m_write_regfile_acesses[NUM_STAT_IDX];
  /* [한국어] 레지스터 파일(RF) 쓰기 접근 횟수
   * 설정자: 실행 결과가 RF에 write-back될 때마다 증분
   * 읽는 자: power_stat_t::get_regfile_writes() — McPAT RF 쓰기 전력
   * 동기화: SM당 독립 */

  unsigned *m_non_rf_operands[NUM_STAT_IDX];
  /* [한국어] 레지스터 파일 외 오퍼랜드(즉시값, 상수 등) 수
   * 설정자: 명령어 디코드 시 RF 비경유 오퍼랜드(즉시 상수 등)마다 증분
   * 읽는 자: power_stat_t::get_non_regfile_operands() — RF 활성화 인수 모델링
   * 동기화: SM당 독립 */
};

/*
 * [한국어]
 * power_core_stat_t - SM 코어 전력 카운터 관리 클래스
 *
 * shader_core_power_stats_pod를 상속하여 SM 카운터 포인터 배열을 소유하며,
 * init()에서 CURRENT 슬롯을 shader_core_stats의 실제 배열에 연결하고
 * PREV 슬롯은 calloc()으로 독립 할당한다.
 * save_stats()는 CURRENT 배열 전체를 PREV에 복사하여 다음 샘플링 창의 델타 계산 기준을 갱신.
 * 이 클래스의 메서드는 power_stat_t가 호출하며, 외부에서 직접 사용하지 않는다.
 *
 * 호출 체인:
 *   power_stat_t 생성자 → power_core_stat_t() → init()
 *   power_stat_t::save_stats() → power_core_stat_t::save_stats()
 */
class power_core_stat_t : public shader_core_power_stats_pod {
 public:
  /*
   * [한국어] 생성자 - shader_core_stats에 대한 포인터를 보관하고 init() 호출
   * @shader_config: SM 개수, 유닛 수 등 코어 구성 파라미터
   * @core_stats: shader_core_stats — SM 카운터들의 실제 저장소
   */
  power_core_stat_t(const shader_core_config *shader_config,
                    shader_core_stats *core_stats);
  /*
   * [한국어] visualizer_print - AerialVision 시각화 파일에 통계 출력 (현재 미구현)
   * @visualizer_file: gzip 압축 시각화 출력 파일 핸들
   */
  void visualizer_print(gzFile visualizer_file);
  /*
   * [한국어] print - SM별 전력 카운터를 표준 출력 파일에 덤프
   * @fout: 출력 파일 포인터 (gpgpusim 전체 통계 출력 시 사용)
   */
  void print(FILE *fout);
  /*
   * [한국어] init - CURRENT 슬롯을 shader_core_stats 배열에 연결, PREV 슬롯 calloc 초기화
   * 이 함수 이후로 CURRENT 포인터가 유효하며 시뮬레이터가 직접 카운터를 갱신한다.
   */
  void init();
  /*
   * [한국어] save_stats - 현재 CURRENT 값을 PREV에 복사하여 샘플링 기준점 갱신
   * power_stat_t::save_stats()에서 호출됨 — mcpat_cycle()의 stat_sample_freq 주기에 맞춰 실행
   */
  void save_stats();

 private:
  shader_core_stats *m_core_stats;
  /* [한국어] SM 카운터 배열의 실제 소유자 — CURRENT 슬롯 포인터들이 이 객체의 배열을 가리킴
   * 설정자: 생성자에서 초기화, 이후 변경 없음
   * 읽는 자: init()에서 CURRENT 슬롯 연결에 사용
   * 동기화: 생성 후 읽기 전용 */

  const shader_core_config *m_config;
  /* [한국어] SM 개수(num_shader()) 등 코어 구성 정보 — init()에서 배열 크기 결정에 사용
   * 설정자: 생성자에서 초기화
   * 동기화: const 포인터, 읽기 전용 */

  float average_duty_cycle;
  /* [한국어] 평균 파이프라인 듀티 사이클 (현재 미사용 — 향후 확장 목적)
   * 설정자: 미사용
   * 읽는 자: 미사용 */
};

/*
 * [한국어]
 * mem_power_stats_pod - 메모리 계층(캐시/DRAM/NoC) 전력 카운터 포인터 집합 (POD 구조체)
 *
 * shader_core_power_stats_pod와 마찬가지로 CURRENT/PREV 쌍으로 관리된다.
 * core_cache_stats, l2_cache_stats는 포인터가 아닌 객체 직접 저장 (cache_stats는 값 타입).
 * DRAM 및 NoC 카운터는 포인터 배열로, CURRENT는 dram_t 등의 실제 카운터를 가리키고
 * PREV는 calloc으로 별도 할당된 스냅샷 배열이다.
 * 배열 인덱스는 DRAM 채널 ID 또는 SIMT 클러스터 ID이다.
 */
struct mem_power_stats_pod {
  // [CURRENT_STAT_IDX] = CURRENT_STAT_IDX stat, [PREV_STAT_IDX] = last reading
  class cache_stats core_cache_stats[NUM_STAT_IDX];  // Total core stats
  /* [한국어] L1 데이터/명령어/상수/텍스처 캐시 통합 통계 (모든 SM의 L1 캐시 합산)
   * 설정자: gpu-cache.cc에서 캐시 접근마다 update() 호출로 갱신
   * 읽는 자: power_stat_t::get_l1d_read_hits() 등 L1 캐시 쿼리 메서드
   * 값 범위: cache_stats 내부에 [access_type][request_status] 2D 카운터 배열
   * 동기화: CURRENT는 캐시 접근마다 갱신, PREV는 save_stats()에서 복사 */

  class cache_stats l2_cache_stats[NUM_STAT_IDX];    // Total L2 partition stats
  /* [한국어] L2 캐시 전체 파티션 통합 통계
   * 설정자: l2cache.cc에서 L2 캐시 접근마다 갱신
   * 읽는 자: power_stat_t::get_l2_read_hits() 등 L2 캐시 쿼리 메서드
   * 동기화: CURRENT는 L2 접근마다 갱신, PREV는 save_stats()에서 복사 */

  unsigned *shmem_access[NUM_STAT_IDX];  // Shared memory access
  /* [한국어] SM당 공유 메모리(Shared Memory, SRAM) 접근 횟수 배열 [SM 개수]
   * 설정자: shader_core_ctx::cycle()에서 SM의 shared memory bank 접근마다 증분
   * 읽는 자: power_stat_t::get_shmem_access() — McPAT 공유 메모리 전력 입력
   * 값 범위: 0~ (단조 증가)
   * 동기화: SM당 독립 배열 원소 */

  // Low level DRAM stats
  unsigned *n_cmd[NUM_STAT_IDX];
  /* [한국어] DRAM 채널별 전체 명령 수 (n_cmd = n_nop + n_act + n_pre + n_rd + n_wr)
   * 설정자: dram_t::cycle()에서 매 사이클 DRAM 명령 발행 시 증분
   * 읽는 자: power_stat_t::get_dram_cmd() — DRAM 컨트롤러 활성도 측정
   * 배열 크기: [m_n_mem] (DRAM 채널 수)
   * 동기화: DRAM 채널당 독립 */

  unsigned *n_activity[NUM_STAT_IDX];
  /* [한국어] DRAM 활성(비유휴) 사이클 수 — 어떤 명령이든 처리 중인 사이클 카운트
   * 설정자: dram_t::cycle()에서 활성 DRAM 사이클마다 증분
   * 읽는 자: power_stat_t::get_dram_activity()
   * 배열 크기: [m_n_mem]
   * 동기화: DRAM 채널당 독립 */

  unsigned *n_nop[NUM_STAT_IDX];
  /* [한국어] DRAM NOP(No-Operation) 명령 수 — DRAM 컨트롤러가 아무 요청도 처리 못할 때 발행
   * 설정자: dram_t::cycle()에서 NOP 명령 발행 시 증분
   * 읽는 자: power_stat_t::get_dram_nop()
   * 배열 크기: [m_n_mem]
   * 동기화: DRAM 채널당 독립 */

  unsigned *n_act[NUM_STAT_IDX];
  /* [한국어] DRAM ACT(Activate) 명령 수 — 뱅크의 Row를 열어 접근 가능하게 하는 명령
   * 설정자: dram_t::cycle()에서 ACT 명령 발행 시 증분
   * 읽는 자: power_stat_t::get_dram_act() — McPAT DRAM 활성화 전력
   * 배열 크기: [m_n_mem]
   * 동기화: DRAM 채널당 독립 */

  unsigned *n_pre[NUM_STAT_IDX];
  /* [한국어] DRAM PRE(Precharge) 명령 수 — 열린 Row를 닫아 다음 ACT를 위한 준비 명령
   * 설정자: dram_t::cycle()에서 PRE 명령 발행 시 증분
   * 읽는 자: power_stat_t::get_dram_pre() — McPAT DRAM 프리차지 전력
   * 배열 크기: [m_n_mem]
   * 동기화: DRAM 채널당 독립 */

  unsigned *n_rd[NUM_STAT_IDX];
  /* [한국어] DRAM READ 명령 수 — DRAM에서 데이터를 읽는 명령
   * 설정자: dram_t::cycle()에서 READ 명령 스케줄링 시 증분
   * 읽는 자: power_stat_t::get_dram_rd() — McPAT DRAM 읽기 전력 및 HW 모드 비교
   * 배열 크기: [m_n_mem]
   * 동기화: DRAM 채널당 독립 */

  unsigned *n_wr[NUM_STAT_IDX];
  /* [한국어] DRAM WRITE 명령 수 (워크로드에서 직접 발생한 쓰기)
   * 설정자: dram_t::cycle()에서 WRITE 명령 스케줄링 시 증분
   * 읽는 자: power_stat_t::get_dram_wr() — n_wr + n_wr_WB 합산으로 사용
   * 배열 크기: [m_n_mem]
   * 동기화: DRAM 채널당 독립 */

  unsigned *n_wr_WB[NUM_STAT_IDX];
  /* [한국어] 캐시 Write-Back으로 발생한 DRAM WRITE 명령 수
   * 설정자: L2 캐시에서 더티 라인 축출(eviction) 시 DRAM 쓰기 명령 발행마다 증분
   * 읽는 자: power_stat_t::get_dram_wr() — n_wr와 합산 (WB 포함 전체 쓰기)
   * 배열 크기: [m_n_mem]
   * 동기화: DRAM 채널당 독립 */

  unsigned *n_req[NUM_STAT_IDX];
  /* [한국어] DRAM 컨트롤러에 제출된 전체 메모리 요청 수 (READ+WRITE 합산)
   * 설정자: dram_t에 mem_fetch가 push될 때마다 증분
   * 읽는 자: power_stat_t::get_dram_req()
   * 배열 크기: [m_n_mem]
   * 동기화: DRAM 채널당 독립 */

  // Interconnect stats
  long *n_simt_to_mem[NUM_STAT_IDX];
  /* [한국어] SIMT 클러스터 → 메모리 파티션 방향 NoC 플릿(flit) 수 배열 [클러스터 수]
   * 설정자: shader_core_ctx 또는 ICNT 레이어에서 패킷 전송 시마다 증분
   * 읽는 자: power_stat_t::get_icnt_simt_to_mem() — McPAT NoC 전력 입력
   * 값 범위: 0~ (long — 매우 많은 플릿이 전송될 수 있음)
   * 동기화: 클러스터당 독립 배열 원소 */

  long *n_mem_to_simt[NUM_STAT_IDX];
  /* [한국어] 메모리 파티션 → SIMT 클러스터 방향 NoC 플릿 수 배열 [클러스터 수]
   * 설정자: 메모리 파티션 응답이 ICNT를 통해 SM으로 전달될 때마다 증분
   * 읽는 자: power_stat_t::get_icnt_mem_to_simt() — McPAT NoC 전력 입력
   * 동기화: 클러스터당 독립 */
};

/*
 * [한국어]
 * power_mem_stat_t - 메모리 전력 카운터 관리 클래스
 *
 * mem_power_stats_pod를 상속하여 캐시/DRAM/NoC 카운터 포인터 배열을 초기화하고 관리한다.
 * init()에서 DRAM/NoC 카운터 PREV 슬롯을 calloc으로 할당하고,
 * cache_stats 객체를 clear()로 초기화한다.
 * save_stats()는 DRAM/NoC/공유메모리 카운터의 CURRENT를 PREV에 복사하고
 * cache_stats PREV를 CURRENT로 대입한다.
 */
class power_mem_stat_t : public mem_power_stats_pod {
 public:
  /*
   * [한국어] 생성자 - 설정 포인터 저장 후 init() 호출
   * @mem_config: DRAM 채널 수, 뱅크 수 등 메모리 구성
   * @shdr_config: SM 개수, SIMT 클러스터 수 — NoC 배열 크기 결정
   * @mem_stats: memory_stats_t — DRAM/NoC 통계 원본 (현재 미사용, 향후 확장용)
   * @shdr_stats: shader_core_stats — 공유 메모리 카운터 원본
   */
  power_mem_stat_t(const memory_config *mem_config,
                   const shader_core_config *shdr_config,
                   memory_stats_t *mem_stats, shader_core_stats *shdr_stats);
  /*
   * [한국어] visualizer_print - AerialVision 시각화 출력 (현재 미구현)
   */
  void visualizer_print(gzFile visualizer_file);
  /*
   * [한국어] print - 메모리 전력 통계 (DRAM 읽기/쓰기 총수, 캐시 통계) 출력
   * @fout: 출력 파일 포인터
   */
  void print(FILE *fout) const;
  /*
   * [한국어] init - CURRENT/PREV 슬롯 초기화
   * PREV 슬롯: calloc으로 [m_n_mem] 또는 [n_simt_clusters] 크기 배열 할당 후 0으로 초기화
   * CURRENT 슬롯: shader_core_stats의 공유메모리 카운터에 연결, DRAM/NoC는 별도 할당
   * 주의: DRAM CURRENT 카운터는 dram_t에서 직접 업데이트하므로 여기서는 연결만 수행
   */
  void init();
  /*
   * [한국어] save_stats - 현재 카운터를 PREV에 복사하여 다음 샘플링 창 기준점 갱신
   * cache_stats PREV = CURRENT (대입 연산자)
   * DRAM/NoC/공유메모리 PREV[i] = CURRENT[i] (원소별 복사)
   * power_stat_t::save_stats()에서 호출
   */
  void save_stats();

 private:
  memory_stats_t *m_mem_stats;
  /* [한국어] 메모리 레이턴시/접근 통계 원본 — 현재 이 클래스에서 직접 미사용 (향후 확장 목적)
   * 설정자: 생성자에서 초기화
   * 동기화: 생성 후 읽기 전용 포인터 */

  shader_core_stats *m_core_stats;
  /* [한국어] SM 카운터 원본 — init()에서 shmem_access CURRENT 슬롯 연결에 사용
   * 설정자: 생성자에서 초기화
   * 동기화: 생성 후 읽기 전용 포인터 */

  const memory_config *m_config;
  /* [한국어] DRAM 채널 수(m_n_mem), 뱅크 수 등 메모리 구성 — 배열 크기 결정에 사용
   * 설정자: 생성자에서 초기화
   * 동기화: const 포인터, 읽기 전용 */

  const shader_core_config *m_core_config;
  /* [한국어] SM 개수, SIMT 클러스터 수 — shmem_access, NoC 배열 크기 결정에 사용
   * 설정자: 생성자에서 초기화
   * 동기화: const 포인터, 읽기 전용 */
};

/*
 * [한국어]
 * power_stat_t - AccelWattch 전력 계산을 위한 최상위 통계 집계 클래스
 *
 * power_core_stat_t(SM 코어 카운터)와 power_mem_stat_t(메모리 카운터)를 소유하며,
 * power_interface.cc가 호출하는 모든 get_*() 쿼리 메서드를 제공한다.
 * 두 가지 계산 모드를 지원한다:
 *   - 델타 모드(aggregate_stat=false): CURRENT-PREV 차이 — 이번 샘플링 창의 증분
 *   - 누적 모드(aggregate_stat=true): CURRENT 절대값 — 시뮬레이터 시작 이후 합산
 *
 * 또한 HW/하이브리드 모드에서 커널 경계를 추적하기 위한 *_kernel 필드와
 * 여러 커널 누적용 *_execution 필드를 public으로 직접 노출한다.
 *
 * 이 클래스는 gpu-sim.cc에서 생성되어 시뮬레이터 수명 내내 유지된다.
 */
class power_stat_t {
 public:
  /*
   * [한국어] 생성자 - power_core_stat_t와 power_mem_stat_t를 동적 할당하고 모든 필드를 0으로 초기화
   * @shader_config: SM 구성 — 코어 카운터 배열 크기 결정
   * @average_pipeline_duty_cycle: 외부(gpu-sim)에서 매 사이클 업데이트하는 듀티 사이클 포인터
   * @active_sms: 외부에서 매 사이클 업데이트하는 활성 SM 수 포인터
   * @shader_stats: shader_core_stats — SM 카운터 원본
   * @mem_config: 메모리 구성
   * @memory_stats: 메모리 레이턴시 통계 원본
   */
  power_stat_t(const shader_core_config *shader_config,
               float *average_pipeline_duty_cycle, float *active_sms,
               shader_core_stats *shader_stats, const memory_config *mem_config,
               memory_stats_t *memory_stats);
  /*
   * [한국어] visualizer_print - 시각화 파일에 코어+메모리 통계 출력
   */
  void visualizer_print(gzFile visualizer_file);
  /*
   * [한국어] print - stdout에 평균 듀티 사이클과 모든 카운터 출력
   */
  void print(FILE *fout) const;
  /*
   * [한국어] save_stats - 샘플링 주기 완료 시 호출되어 현재 카운터를 PREV에 복사
   * pwr_core_stat->save_stats() 및 pwr_mem_stat->save_stats() 위임 후
   * m_average_pipeline_duty_cycle과 m_active_sms를 0으로 리셋하여
   * 다음 샘플링 창의 누산을 시작한다.
   *
   * 호출 체인: mcpat_cycle() 내부에서 wrapper->compute() 완료 후 호출
   */
  void save_stats() {
    pwr_core_stat->save_stats();              /* [한국어] SM 코어 카운터 CURRENT→PREV 복사 */
    pwr_mem_stat->save_stats();              /* [한국어] 메모리 카운터 CURRENT→PREV 복사 */
    *m_average_pipeline_duty_cycle = 0;     /* [한국어] 듀티 사이클 누산기를 0으로 리셋 (새 창 시작) */
    *m_active_sms = 0;                      /* [한국어] 활성 SM 누산기를 0으로 리셋 (새 창 시작) */
  }
  /*
   * [한국어] clear - HW/하이브리드 모드 커널 완료 후 모든 CURRENT/PREV 카운터를 0으로 초기화
   * calculate_hw_mcpat()에서 커널 경계에서 호출되어 다음 커널의 카운터가 이전 커널과 섞이지 않게 함.
   * cache_stats, 코어 카운터, DRAM 카운터 모두 0으로 초기화.
   */
  void clear();

  /* [한국어] === HW/하이브리드 모드 커널 기준점 필드 ===
   * 이 필드들은 커널 시작 시점의 누적 카운터 스냅샷이다.
   * 다음 커널 완료 시 현재값 - 기준점 = 이 커널의 카운터 증분
   * 설정자: calculate_hw_mcpat() 말미에서 현재 누적값으로 갱신
   * 읽는 자: calculate_hw_mcpat() 에서 델타 계산 시 차감
   * 동기화: 단일 스레드, 커널 완료 시에만 접근 */
  unsigned l1i_misses_kernel;           /* [한국어] 이전 커널 완료 시점의 L1I 미스 누적값 */
  unsigned l1i_hits_kernel;             /* [한국어] 이전 커널 완료 시점의 L1I 히트 누적값 */
  unsigned long long l1r_hits_kernel;   /* [한국어] 이전 커널 완료 시점의 L1D 읽기 히트 누적값 */
  unsigned long long l1r_misses_kernel; /* [한국어] 이전 커널 완료 시점의 L1D 읽기 미스 누적값 */
  unsigned long long l1w_hits_kernel;   /* [한국어] 이전 커널 완료 시점의 L1D 쓰기 히트 누적값 */
  unsigned long long l1w_misses_kernel; /* [한국어] 이전 커널 완료 시점의 L1D 쓰기 미스 누적값 */
  unsigned long long shared_accesses_kernel; /* [한국어] 이전 커널의 공유 메모리 접근 기준점 */
  unsigned long long cc_accesses_kernel;     /* [한국어] 이전 커널의 상수 캐시 접근 기준점 */
  unsigned long long dram_rd_kernel;    /* [한국어] 이전 커널 완료 시점의 DRAM 읽기 명령 누적값 */
  unsigned long long dram_wr_kernel;    /* [한국어] 이전 커널 완료 시점의 DRAM 쓰기 명령 누적값 */
  unsigned long long dram_pre_kernel;   /* [한국어] 이전 커널 완료 시점의 DRAM 프리차지 명령 누적값 */
  unsigned long long l2r_hits_kernel;   /* [한국어] 이전 커널의 L2 읽기 히트 기준점 */
  unsigned long long l2r_misses_kernel; /* [한국어] 이전 커널의 L2 읽기 미스 기준점 */
  unsigned long long l2w_hits_kernel;   /* [한국어] 이전 커널의 L2 쓰기 히트 기준점 */
  unsigned long long l2w_misses_kernel; /* [한국어] 이전 커널의 L2 쓰기 미스 기준점 */
  unsigned long long noc_tr_kernel;     /* [한국어] 이전 커널의 NoC SIMT→MEM 플릿 기준점 */
  unsigned long long noc_rc_kernel;     /* [한국어] 이전 커널의 NoC MEM→SIMT 플릿 기준점 */

  /* [한국어] === aggregate_power_stats 모드에서 여러 커널에 걸쳐 누적하는 필드 ===
   * calculate_hw_mcpat()의 aggregate_power_stats=true 경우에만 사용.
   * 각 커널의 카운터를 더해 전체 워크로드의 합산 전력 계산에 사용.
   * 설정자: calculate_hw_mcpat()에서 커널 완료 시 현재값을 더함
   * 읽는 자: calculate_hw_mcpat()에서 wrapper->set_*_power()에 전달
   * 동기화: 단일 스레드 */
  unsigned long long tot_inst_execution;      /* [한국어] 누적 전체 명령어 수 */
  unsigned long long tot_int_inst_execution;  /* [한국어] 누적 정수 명령어 수 */
  unsigned long long tot_fp_inst_execution;   /* [한국어] 누적 FP 명령어 수 */
  unsigned long long commited_inst_execution; /* [한국어] 누적 커밋된 명령어 수 */
  unsigned long long ialu_acc_execution;      /* [한국어] 누적 정수 ALU 접근 수 */
  unsigned long long imul24_acc_execution;    /* [한국어] 누적 24비트 IMUL 접근 수 */
  unsigned long long imul32_acc_execution;    /* [한국어] 누적 32비트 IMUL 접근 수 */
  unsigned long long imul_acc_execution;      /* [한국어] 누적 IMUL 접근 수 */
  unsigned long long idiv_acc_execution;      /* [한국어] 누적 IDIV 접근 수 */
  unsigned long long dp_acc_execution;        /* [한국어] 누적 DP FPU 접근 수 */
  unsigned long long dpmul_acc_execution;     /* [한국어] 누적 DP 곱셈 접근 수 */
  unsigned long long dpdiv_acc_execution;     /* [한국어] 누적 DP 나눗셈 접근 수 */
  unsigned long long fp_acc_execution;        /* [한국어] 누적 SP FPU 접근 수 */
  unsigned long long fpmul_acc_execution;     /* [한국어] 누적 SP 곱셈 접근 수 */
  unsigned long long fpdiv_acc_execution;     /* [한국어] 누적 SP 나눗셈 접근 수 */
  unsigned long long sqrt_acc_execution;      /* [한국어] 누적 sqrt 접근 수 */
  unsigned long long log_acc_execution;       /* [한국어] 누적 log 접근 수 */
  unsigned long long sin_acc_execution;       /* [한국어] 누적 sin 접근 수 */
  unsigned long long exp_acc_execution;       /* [한국어] 누적 exp 접근 수 */
  unsigned long long tensor_acc_execution;    /* [한국어] 누적 텐서 코어 접근 수 */
  unsigned long long tex_acc_execution;       /* [한국어] 누적 텍스처 접근 수 */
  unsigned long long tot_fpu_acc_execution;   /* [한국어] 누적 전체 FPU(SP+DP) 접근 수 */
  unsigned long long tot_sfu_acc_execution;   /* [한국어] 누적 전체 SFU 접근 수 */
  unsigned long long tot_threads_acc_execution; /* [한국어] 누적 활성 스레드 수 */
  unsigned long long tot_warps_acc_execution;   /* [한국어] 누적 활성 warp 수 */
  unsigned long long sp_active_lanes_execution; /* [한국어] 누적 SP 파이프라인 활성 레인 수 */
  unsigned long long sfu_active_lanes_execution;/* [한국어] 누적 SFU 파이프라인 활성 레인 수 */
  /*
   * [한국어]
   * get_total_inst - 전체 디코드된 명령어 수 반환
   *
   * @aggregate_stat: false=델타(이번 창 증분), true=누적값(CURRENT 절대값)
   * @return: 모든 SM의 합산 명령어 수 (double 반환으로 대형 GPU에서 오버플로 방지)
   *
   * aggregate_stat=false: CURRENT[i] - PREV[i]의 전체 SM 합산 — mcpat_cycle() 사용
   * aggregate_stat=true: CURRENT[i]의 전체 SM 합산 — calculate_hw_mcpat() 사용
   *
   * 호출 체인: mcpat_cycle() 또는 calculate_hw_mcpat() → wrapper->set_inst_power()
   */
  double get_total_inst(bool aggregate_stat) {
    double total_inst = 0;  /* [한국어] 전체 SM 합산 카운터 초기화 */
    for (unsigned i = 0; i < m_config->num_shader(); i++) {  /* [한국어] 모든 SM에 대해 순회 */
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_decoded_insn[CURRENT_STAT_IDX][i]); /* [한국어] 누적 모드: 현재값 합산 */
      else
        total_inst += (pwr_core_stat->m_num_decoded_insn[CURRENT_STAT_IDX][i]) -
                      (pwr_core_stat->m_num_decoded_insn[PREV_STAT_IDX][i]);    /* [한국어] 델타 모드: 이번 창 증분 합산 */
    }
    return total_inst;
  }
  double get_total_int_inst(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst +=
            (pwr_core_stat->m_num_INTdecoded_insn[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
            (pwr_core_stat->m_num_INTdecoded_insn[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_INTdecoded_insn[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_total_fp_inst(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst +=
            (pwr_core_stat->m_num_FPdecoded_insn[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
            (pwr_core_stat->m_num_FPdecoded_insn[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_FPdecoded_insn[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_total_load_inst() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst +=
          (pwr_core_stat->m_num_loadqueued_insn[CURRENT_STAT_IDX][i]) -
          (pwr_core_stat->m_num_loadqueued_insn[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_total_store_inst() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst +=
          (pwr_core_stat->m_num_storequeued_insn[CURRENT_STAT_IDX][i]) -
          (pwr_core_stat->m_num_storequeued_insn[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_sp_committed_inst() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_num_sp_committed[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_sp_committed[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_sfu_committed_inst() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_num_sfu_committed[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_sfu_committed[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_mem_committed_inst() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_num_mem_committed[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_mem_committed[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_committed_inst(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst +=
            (pwr_core_stat->m_num_mem_committed[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_sfu_committed[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_sp_committed[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
            (pwr_core_stat->m_num_mem_committed[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_mem_committed[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_sfu_committed[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_sfu_committed[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_sp_committed[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_sp_committed[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_regfile_reads(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst +=
            (pwr_core_stat->m_read_regfile_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
            (pwr_core_stat->m_read_regfile_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_read_regfile_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_regfile_writes(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst +=
            (pwr_core_stat->m_write_regfile_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
            (pwr_core_stat->m_write_regfile_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_write_regfile_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  float get_pipeline_duty() {
    float total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst +=
          (pwr_core_stat->m_pipeline_duty_cycle[CURRENT_STAT_IDX][i]) -
          (pwr_core_stat->m_pipeline_duty_cycle[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_non_regfile_operands(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_non_rf_operands[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_non_rf_operands[CURRENT_STAT_IDX][i]) -
                      (pwr_core_stat->m_non_rf_operands[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_sp_accessess() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_num_sp_acesses[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_sp_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_sfu_accessess() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_num_sfu_acesses[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_sfu_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_sqrt_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_sqrt_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_sqrt_acesses[CURRENT_STAT_IDX][i]) -
                      (pwr_core_stat->m_num_sqrt_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_log_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_log_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_log_acesses[CURRENT_STAT_IDX][i]) -
                      (pwr_core_stat->m_num_log_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_sin_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_sin_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_sin_acesses[CURRENT_STAT_IDX][i]) -
                      (pwr_core_stat->m_num_sin_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }
  double get_exp_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_exp_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_exp_acesses[CURRENT_STAT_IDX][i]) -
                      (pwr_core_stat->m_num_exp_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_mem_accessess() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_num_mem_acesses[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_mem_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_intdiv_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_idiv_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_idiv_acesses[CURRENT_STAT_IDX][i]) -
                      (pwr_core_stat->m_num_idiv_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_fpdiv_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_fpdiv_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
            (pwr_core_stat->m_num_fpdiv_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_fpdiv_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_intmul32_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst +=
            (pwr_core_stat->m_num_imul32_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
            (pwr_core_stat->m_num_imul32_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_imul32_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_intmul24_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst +=
            (pwr_core_stat->m_num_imul24_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
            (pwr_core_stat->m_num_imul24_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_imul24_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_intmul_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_imul_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_imul_acesses[CURRENT_STAT_IDX][i]) -
                      (pwr_core_stat->m_num_imul_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_fpmul_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_fpmul_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
            (pwr_core_stat->m_num_fpmul_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_fpmul_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_fp_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_fp_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_fp_acesses[CURRENT_STAT_IDX][i]) -
                      (pwr_core_stat->m_num_fp_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_dp_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_dp_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_dp_acesses[CURRENT_STAT_IDX][i]) -
                      (pwr_core_stat->m_num_dp_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_dpmul_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_dpmul_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
            (pwr_core_stat->m_num_dpmul_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_dpmul_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_dpdiv_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_dpdiv_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
            (pwr_core_stat->m_num_dpdiv_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_dpdiv_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_tensor_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst +=
            (pwr_core_stat->m_num_tensor_core_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
            (pwr_core_stat->m_num_tensor_core_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_tensor_core_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_const_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += pwr_core_stat->m_num_const_acesses[CURRENT_STAT_IDX][i];
      else
        total_inst +=
            (pwr_core_stat->m_num_const_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_const_acesses[PREV_STAT_IDX][i]);
    }
    return (total_inst);
  }

  double get_tex_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_tex_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_tex_acesses[CURRENT_STAT_IDX][i]) -
                      (pwr_core_stat->m_num_tex_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_sp_active_lanes() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_active_sp_lanes[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_active_sp_lanes[PREV_STAT_IDX][i]);
    }
    return (total_inst / m_config->num_shader()) / m_config->gpgpu_num_sp_units;
  }

  float get_sfu_active_lanes() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_active_sfu_lanes[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_active_sfu_lanes[PREV_STAT_IDX][i]);
    }

    return (total_inst / m_config->num_shader()) /
           m_config->gpgpu_num_sfu_units;
  }

  float get_active_threads(bool aggregate_stat) {
    unsigned total_threads = 0;
    unsigned total_warps = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat) {
        total_threads +=
            (pwr_core_stat->m_active_exu_threads[CURRENT_STAT_IDX][i]);
        total_warps += (pwr_core_stat->m_active_exu_warps[CURRENT_STAT_IDX][i]);
      } else {
        total_threads +=
            (pwr_core_stat->m_active_exu_threads[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_active_exu_threads[PREV_STAT_IDX][i]);
        total_warps +=
            (pwr_core_stat->m_active_exu_warps[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_active_exu_warps[PREV_STAT_IDX][i]);
      }
    }
    if (total_warps != 0)
      return (float)((float)total_threads / (float)total_warps);
    else
      return 0;
  }

  unsigned long long get_tot_threads_kernel(bool aggregate_stat) {
    unsigned total_threads = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat) {
        total_threads +=
            (pwr_core_stat->m_active_exu_threads[CURRENT_STAT_IDX][i]);
      } else {
        total_threads +=
            (pwr_core_stat->m_active_exu_threads[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_active_exu_threads[PREV_STAT_IDX][i]);
      }
    }

    return total_threads;
  }
  unsigned long long get_tot_warps_kernel(bool aggregate_stat) {
    unsigned long long total_warps = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat) {
        total_warps += (pwr_core_stat->m_active_exu_warps[CURRENT_STAT_IDX][i]);
      } else {
        total_warps +=
            (pwr_core_stat->m_active_exu_warps[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_active_exu_warps[PREV_STAT_IDX][i]);
      }
    }
    return total_warps;
  }

  double get_tot_fpu_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_fp_acesses[CURRENT_STAT_IDX][i]) +
                      (pwr_core_stat->m_num_dp_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_fp_acesses[CURRENT_STAT_IDX][i]) -
                      (pwr_core_stat->m_num_fp_acesses[PREV_STAT_IDX][i]) +
                      (pwr_core_stat->m_num_dp_acesses[CURRENT_STAT_IDX][i]) -
                      (pwr_core_stat->m_num_dp_acesses[PREV_STAT_IDX][i]);
    }
    // total_inst +=
    // get_total_load_inst()+get_total_store_inst()+get_tex_inst();
    return total_inst;
  }

  double get_tot_sfu_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst +=
            (pwr_core_stat->m_num_idiv_acesses[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_imul32_acesses[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_sqrt_acesses[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_log_acesses[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_sin_acesses[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_exp_acesses[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_fpdiv_acesses[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_fpmul_acesses[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_dpmul_acesses[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_dpdiv_acesses[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_imul24_acesses[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_imul_acesses[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_tensor_core_acesses[CURRENT_STAT_IDX][i]) +
            (pwr_core_stat->m_num_tex_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst +=
            (pwr_core_stat->m_num_idiv_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_idiv_acesses[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_imul32_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_imul32_acesses[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_sqrt_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_sqrt_acesses[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_log_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_log_acesses[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_sin_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_sin_acesses[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_exp_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_exp_acesses[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_fpdiv_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_fpdiv_acesses[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_fpmul_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_fpmul_acesses[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_dpmul_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_dpmul_acesses[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_dpdiv_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_dpdiv_acesses[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_imul24_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_imul24_acesses[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_imul_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_imul_acesses[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_tensor_core_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_tensor_core_acesses[PREV_STAT_IDX][i]) +
            (pwr_core_stat->m_num_tex_acesses[CURRENT_STAT_IDX][i]) -
            (pwr_core_stat->m_num_tex_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_ialu_accessess(bool aggregate_stat) {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_core_stat->m_num_ialu_acesses[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_core_stat->m_num_ialu_acesses[CURRENT_STAT_IDX][i]) -
                      (pwr_core_stat->m_num_ialu_acesses[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_tex_inst() {
    double total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      total_inst += (pwr_core_stat->m_num_tex_inst[CURRENT_STAT_IDX][i]) -
                    (pwr_core_stat->m_num_tex_inst[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  double get_constant_c_accesses() {
    enum mem_access_type access_type[] = {CONST_ACC_R};
    enum cache_request_status request_status[] = {HIT, MISS, HIT_RESERVED};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
  }
  double get_constant_c_misses() {
    enum mem_access_type access_type[] = {CONST_ACC_R};
    enum cache_request_status request_status[] = {MISS};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
  }
  double get_constant_c_hits() {
    return (get_constant_c_accesses() - get_constant_c_misses());
  }
  double get_texture_c_accesses() {
    enum mem_access_type access_type[] = {TEXTURE_ACC_R};
    enum cache_request_status request_status[] = {HIT, MISS, HIT_RESERVED};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
  }
  double get_texture_c_misses() {
    enum mem_access_type access_type[] = {TEXTURE_ACC_R};
    enum cache_request_status request_status[] = {MISS};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status)) -
           (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
               access_type, num_access_type, request_status,
               num_request_status));
  }
  double get_texture_c_hits() {
    return (get_texture_c_accesses() - get_texture_c_misses());
  }
  double get_inst_c_accesses(bool aggregate_stat) {
    enum mem_access_type access_type[] = {INST_ACC_R};
    enum cache_request_status request_status[] = {HIT, MISS, HIT_RESERVED};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);
    if (aggregate_stat)
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
          access_type, num_access_type, request_status, num_request_status));
    else
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status)) -
             (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status));
  }
  double get_inst_c_misses(bool aggregate_stat) {
    enum mem_access_type access_type[] = {INST_ACC_R};
    enum cache_request_status request_status[] = {MISS};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);
    if (aggregate_stat)
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
          access_type, num_access_type, request_status, num_request_status));
    else
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status)) -
             (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status));
  }
  double get_inst_c_hits(bool aggregate_stat) {
    return (get_inst_c_accesses(aggregate_stat) -
            get_inst_c_misses(aggregate_stat));
  }

  double get_l1d_read_accesses(bool aggregate_stat) {
    enum mem_access_type access_type[] = {GLOBAL_ACC_R, LOCAL_ACC_R};
    enum cache_request_status request_status[] = {HIT, MISS, SECTOR_MISS};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    if (aggregate_stat) {
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
          access_type, num_access_type, request_status, num_request_status));
    } else {
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status)) -
             (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status));
    }
  }
  double get_l1d_read_misses(bool aggregate_stat) {
    return (get_l1d_read_accesses(aggregate_stat) -
            get_l1d_read_hits(aggregate_stat));
  }
  double get_l1d_read_hits(bool aggregate_stat) {
    enum mem_access_type access_type[] = {GLOBAL_ACC_R, LOCAL_ACC_R};
    enum cache_request_status request_status[] = {HIT, MSHR_HIT};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    if (aggregate_stat) {
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
          access_type, num_access_type, request_status, num_request_status));
    } else {
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status)) -
             (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status));
    }
  }
  double get_l1d_write_accesses(bool aggregate_stat) {
    enum mem_access_type access_type[] = {GLOBAL_ACC_W, LOCAL_ACC_W};
    enum cache_request_status request_status[] = {HIT, MISS, SECTOR_MISS};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    if (aggregate_stat) {
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
          access_type, num_access_type, request_status, num_request_status));
    } else {
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status)) -
             (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status));
    }
  }
  double get_l1d_write_misses(bool aggregate_stat) {
    return (get_l1d_write_accesses(aggregate_stat) -
            get_l1d_write_hits(aggregate_stat));
  }
  double get_l1d_write_hits(bool aggregate_stat) {
    enum mem_access_type access_type[] = {GLOBAL_ACC_W, LOCAL_ACC_W};
    enum cache_request_status request_status[] = {HIT, MSHR_HIT};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);

    if (aggregate_stat) {
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
          access_type, num_access_type, request_status, num_request_status));
    } else {
      return (pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status)) -
             (pwr_mem_stat->core_cache_stats[PREV_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status));
    }
  }
  double get_cache_misses() {
    return get_l1d_read_misses(0) + get_constant_c_misses() +
           get_l1d_write_misses(0) + get_texture_c_misses();
  }

  double get_cache_read_misses() {
    return get_l1d_read_misses(0) + get_constant_c_misses() +
           get_texture_c_misses();
  }

  double get_cache_write_misses() { return get_l1d_write_misses(0); }

  double get_shmem_access(bool aggregate_stat) {
    unsigned total_inst = 0;
    for (unsigned i = 0; i < m_config->num_shader(); i++) {
      if (aggregate_stat)
        total_inst += (pwr_mem_stat->shmem_access[CURRENT_STAT_IDX][i]);
      else
        total_inst += (pwr_mem_stat->shmem_access[CURRENT_STAT_IDX][i]) -
                      (pwr_mem_stat->shmem_access[PREV_STAT_IDX][i]);
    }
    return total_inst;
  }

  unsigned long long get_l2_read_accesses(bool aggregate_stat) {
    enum mem_access_type access_type[] = {
        GLOBAL_ACC_R, LOCAL_ACC_R, CONST_ACC_R, TEXTURE_ACC_R, INST_ACC_R};
    enum cache_request_status request_status[] = {HIT, HIT_RESERVED, MISS,
                                                  SECTOR_MISS};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);
    if (aggregate_stat) {
      return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
          access_type, num_access_type, request_status, num_request_status));
    } else {
      return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status)) -
             (pwr_mem_stat->l2_cache_stats[PREV_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status));
    }
  }

  unsigned long long get_l2_read_misses(bool aggregate_stat) {
    return (get_l2_read_accesses(aggregate_stat) -
            get_l2_read_hits(aggregate_stat));
  }

  unsigned long long get_l2_read_hits(bool aggregate_stat) {
    enum mem_access_type access_type[] = {
        GLOBAL_ACC_R, LOCAL_ACC_R, CONST_ACC_R, TEXTURE_ACC_R, INST_ACC_R};
    enum cache_request_status request_status[] = {HIT, HIT_RESERVED};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);
    if (aggregate_stat) {
      return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
          access_type, num_access_type, request_status, num_request_status));
    } else {
      return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status)) -
             (pwr_mem_stat->l2_cache_stats[PREV_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status));
    }
  }

  unsigned long long get_l2_write_accesses(bool aggregate_stat) {
    enum mem_access_type access_type[] = {GLOBAL_ACC_W, LOCAL_ACC_W,
                                          L1_WRBK_ACC};
    enum cache_request_status request_status[] = {HIT, HIT_RESERVED, MISS,
                                                  SECTOR_MISS};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);
    if (aggregate_stat) {
      return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
          access_type, num_access_type, request_status, num_request_status));
    } else {
      return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status)) -
             (pwr_mem_stat->l2_cache_stats[PREV_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status));
    }
  }

  unsigned long long get_l2_write_misses(bool aggregate_stat) {
    return (get_l2_write_accesses(aggregate_stat) -
            get_l2_write_hits(aggregate_stat));
  }
  unsigned long long get_l2_write_hits(bool aggregate_stat) {
    enum mem_access_type access_type[] = {GLOBAL_ACC_W, LOCAL_ACC_W,
                                          L1_WRBK_ACC};
    enum cache_request_status request_status[] = {HIT, HIT_RESERVED};
    unsigned num_access_type =
        sizeof(access_type) / sizeof(enum mem_access_type);
    unsigned num_request_status =
        sizeof(request_status) / sizeof(enum cache_request_status);
    if (aggregate_stat) {
      return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
          access_type, num_access_type, request_status, num_request_status));
    } else {
      return (pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status)) -
             (pwr_mem_stat->l2_cache_stats[PREV_STAT_IDX].get_stats(
                 access_type, num_access_type, request_status,
                 num_request_status));
    }
  }
  double get_dram_cmd() {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      total += (pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i] -
                pwr_mem_stat->n_cmd[PREV_STAT_IDX][i]);
    }
    return total;
  }
  double get_dram_activity() {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      total += (pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i] -
                pwr_mem_stat->n_activity[PREV_STAT_IDX][i]);
    }
    return total;
  }
  double get_dram_nop() {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      total += (pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i] -
                pwr_mem_stat->n_nop[PREV_STAT_IDX][i]);
    }
    return total;
  }
  double get_dram_act() {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      total += (pwr_mem_stat->n_act[CURRENT_STAT_IDX][i] -
                pwr_mem_stat->n_act[PREV_STAT_IDX][i]);
    }
    return total;
  }
  double get_dram_pre(bool aggregate_stat) {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      if (aggregate_stat) {
        total += pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i];
      } else {
        total += (pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i] -
                  pwr_mem_stat->n_pre[PREV_STAT_IDX][i]);
      }
    }
    return total;
  }
  double get_dram_rd(bool aggregate_stat) {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      if (aggregate_stat) {
        total += pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i];
      } else {
        total += (pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i] -
                  pwr_mem_stat->n_rd[PREV_STAT_IDX][i]);
      }
    }
    return total;
  }
  double get_dram_wr(bool aggregate_stat) {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      if (aggregate_stat) {
        total += pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i] +
                 pwr_mem_stat->n_wr_WB[CURRENT_STAT_IDX][i];
      } else {
        total += (pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i] -
                  pwr_mem_stat->n_wr[PREV_STAT_IDX][i]) +
                 (pwr_mem_stat->n_wr_WB[CURRENT_STAT_IDX][i] -
                  pwr_mem_stat->n_wr_WB[PREV_STAT_IDX][i]);
      }
    }
    return total;
  }
  double get_dram_req() {
    unsigned total = 0;
    for (unsigned i = 0; i < m_mem_config->m_n_mem; ++i) {
      total += (pwr_mem_stat->n_req[CURRENT_STAT_IDX][i] -
                pwr_mem_stat->n_req[PREV_STAT_IDX][i]);
    }
    return total;
  }

  unsigned long long get_icnt_simt_to_mem(bool aggregate_stat) {
    long total = 0;
    for (unsigned i = 0; i < m_config->n_simt_clusters; ++i) {
      if (aggregate_stat) {
        total += pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i];
      } else {
        total += (pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i] -
                  pwr_mem_stat->n_simt_to_mem[PREV_STAT_IDX][i]);
      }
    }
    return total;
  }

  unsigned long long get_icnt_mem_to_simt(bool aggregate_stat) {
    long total = 0;
    for (unsigned i = 0; i < m_config->n_simt_clusters; ++i) {
      if (aggregate_stat) {
        total += pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i];
      }

      else {
        total += (pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i] -
                  pwr_mem_stat->n_mem_to_simt[PREV_STAT_IDX][i]);
      }
    }
    return total;
  }

  power_core_stat_t *pwr_core_stat;
  /* [한국어] SM 코어 전력 카운터 관리 객체 — 생성자에서 new로 할당
   * 설정자: power_stat_t 생성자에서 초기화
   * 읽는 자: get_*() 메서드들, save_stats(), clear()
   * 동기화: 단일 스레드 접근, 별도 락 불필요 */

  power_mem_stat_t *pwr_mem_stat;
  /* [한국어] 메모리(캐시/DRAM/NoC) 전력 카운터 관리 객체 — 생성자에서 new로 할당
   * 설정자: power_stat_t 생성자에서 초기화
   * 읽는 자: get_l1d_*(), get_l2_*(), get_dram_*(), get_icnt_*() 메서드들
   * 동기화: 단일 스레드 접근 */

  float *m_average_pipeline_duty_cycle;
  /* [한국어] 샘플링 창 동안의 파이프라인 듀티 사이클 누산기 포인터
   * 설정자: gpu-sim.cc에서 매 사이클 SM 활성도에 따라 증분 (외부에서 직접 업데이트)
   * 읽는 자: mcpat_cycle()에서 stat_sample_freq로 나눠 평균 계산
   * save_stats()에서 0으로 리셋 — 다음 창 누산 시작
   * 동기화: 사이클 루프에서 쓰기, 샘플링 주기에서 읽기 (단일 스레드) */

  float *m_active_sms;
  /* [한국어] 샘플링 창 동안의 활성 SM 수 누산기 포인터
   * 설정자: gpu-sim.cc에서 매 사이클 활성 SM 수를 더함 (외부에서 직접 업데이트)
   * 읽는 자: mcpat_cycle()에서 stat_sample_freq로 나눠 평균 활성 SM 수 계산
   * save_stats()에서 0으로 리셋
   * 동기화: 사이클 루프에서 쓰기, 샘플링 주기에서 읽기 */

  const shader_core_config *m_config;
  /* [한국어] SM 개수(num_shader()), gpgpu_num_sp_units 등 코어 구성 정보
   * 설정자: 생성자에서 초기화
   * 읽는 자: get_*() 메서드들에서 SM 순회 루프 범위 결정에 사용
   * 동기화: const 포인터, 읽기 전용 */

  const memory_config *m_mem_config;
  /* [한국어] DRAM 채널 수(m_n_mem) 등 메모리 구성 정보
   * 설정자: 생성자에서 초기화
   * 읽는 자: get_dram_*() 메서드들에서 DRAM 채널 순회 범위 결정에 사용
   * 동기화: const 포인터, 읽기 전용 */
};

#endif /*POWER_LATENCY_STAT_H*/
