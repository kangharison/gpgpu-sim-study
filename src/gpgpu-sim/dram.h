// Copyright (c) 2009-2021, Tor M. Aamodt, Ivan Sham, Ali Bakhoda,
// George L. Yuan, Wilson W.L. Fung, Vijay Kandiah, Nikos Hardavellas,
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
 * [한국어 설명] DRAM 타이밍 시뮬레이터 헤더 (dram.h)
 *
 * === 파일의 역할 ===
 * GPGPU-Sim의 DRAM 채널(메모리 파티션) 타이밍 모델을 정의하는 헤더 파일이다.
 * 실제 DDR SDRAM의 Bank/Row/Column 구조와 RCD/CL/RP/RC/RRD/CCD 등의
 * 타이밍 파라미터를 사이클-레벨로 시뮬레이션하기 위한 모든 자료구조를 선언한다.
 * GPU 메모리 요청(mem_fetch)이 DRAM 레벨 요청(dram_req_t)으로 변환되어
 * 이 헤더의 dram_t 클래스가 FIFO 또는 FR-FCFS 스케줄러를 통해 처리한다.
 * 각 사이클마다 bank별 타이밍 카운터를 감소시키고 ACT/PRE/RD/WR 커맨드를 발행한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 메모리 계층 최하단에 위치한다. GPU SM에서 발생한 캐시 미스가
 * L1→L2→NoC(ICNT)→memory_partition_unit 경로를 거쳐 최종적으로 dram_t::push()에
 * 도달한다. dram_t::cycle()이 매 DRAM 클럭마다 호출되어 타이밍 상태를 전진시키고,
 * 처리 완료된 READ 요청은 returnq를 통해 다시 ICNT→L2→SM 경로로 응답이 반환된다.
 * 호출 체인: gpu-sim.cc(gpu_sim_cycle 루프) → memory_partition_unit::dram_cycle()
 *           → dram_t::cycle() → scheduler_fifo()/scheduler_frfcfs()
 *           → issue_col_command() / issue_row_command()
 * 실행 컨텍스트: 호스트 CPU 단일 스레드 시뮬레이션 루프 (사이클-레벨 타이밍 모델)
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - mem_fetch.h: DRAM으로 전달되는 메모리 요청 패킷 (dram_req_t의 data 필드)
 *   - dram_sched.h: FR-FCFS 스케줄러 클래스 (frfcfs_scheduler) 선언
 *   - delayqueue.h: 파이프라인 큐(fifo_pipeline) 구현 — rwq, mrqq, returnq에 사용
 *   - memory_config (gpu-sim.h): DRAM 타이밍 파라미터 집합 (tRCD, tCL, tRP 등)
 *   - l2cache.h (memory_partition_unit): dram_t를 소유하는 상위 파티션 단위
 * 역방향 의존 (이 파일에 의존):
 *   - dram.cc: 실제 구현체
 *   - l2cache.cc (memory_partition_unit): dram_t::push(), dram_t::cycle() 호출
 *   - accelwattch 전력 모델: set_dram_power_stats()로 카운터 수집
 * 데이터 흐름:
 *   mem_fetch(캐시 미스 패킷) → dram_req_t(DRAM 요청) → mrqq(입력 큐)
 *   → bk[n]->mrq(뱅크 배정) → rwq(RD/WR 파이프라인) → returnq(완료 패킷)
 *   → memory_partition_unit이 returnq에서 pop하여 ICNT로 전송
 *
 * === 주요 함수/구조체 요약 ===
 * dram_req_t      : mem_fetch를 DRAM 레벨 요청으로 변환한 구조체 (bank/row/col 포함)
 * bankgrp_t       : DDR4 Bank Group 단위 타이밍 카운터 (CCDLc, RTPLc)
 * bank_t          : 개별 DRAM 뱅크의 타이밍 카운터 및 상태 (RCDc, RASc, RPc 등)
 * bank_index_function : 뱅크 인덱싱 정책 열거형 (LINEAR, XOR, IPOLY, CUSTOM)
 * dram_t          : DRAM 채널 타이밍 모델의 핵심 클래스 — 스케줄러, 큐, 통계 보유
 *   - push()      : 상위(memory_partition_unit)에서 새 메모리 요청을 mrqq에 삽입
 *   - cycle()     : 매 DRAM 사이클마다 호출 — 커맨드 발행, 카운터 감소, 통계 수집
 *   - issue_col_command() : RD/WR 컬럼 커맨드 발행 (row-buffer hit 조건 확인)
 *   - issue_row_command() : ACT/PRE 로우 커맨드 발행 (bank idle/active 전환)
 *   - return_queue_pop()  : 처리 완료된 mem_fetch를 returnq에서 꺼내 상위로 반환
 */

#ifndef DRAM_H
#define DRAM_H

#include <stdio.h>   /* [한국어] 표준 입출력 (fprintf/printf — print()/print_stat() 통계 출력에 사용) */
#include <stdlib.h>  /* [한국어] 동적 메모리 (calloc — bk[], bkgrp[] 배열 초기화에 사용) */
#include <zlib.h>    /* [한국어] gzip 압축 스트림 (gzFile/gzprintf — visualizer_print()에서 .gz 로그 출력) */
#include <bitset>    /* [한국어] C++ bitset (bnkgrp_rw_found: Bank Group 레벨 병렬성 추적용 비트셋) */
#include <fstream>   /* [한국어] 파일 스트림 (dram_log()에서 로그 파일 기록용) */
#include <iomanip>   /* [한국어] 출력 형식 지정 (setw/setfill — 디버그 출력 정렬) */
#include <set>       /* [한국어] 순서 있는 집합 (dram_sched.h에서 사용하는 자료구조 공유) */
#include <sstream>   /* [한국어] 문자열 스트림 (mrqq_Dist 통계 이름 생성용) */
#include <string>    /* [한국어] C++ 문자열 (파이프라인 큐 이름 "rwq"/"mrqq" 등 관리) */
#include <vector>    /* [한국어] 동적 배열 (dram_sched.h의 요청 리스트 관리에 사용) */
#include "delayqueue.h"  /* [한국어] fifo_pipeline<T> 템플릿 — rwq(읽기/쓰기 파이프라인),
                          * mrqq(메모리 요청 큐), returnq(완료 응답 큐) 구현체 */

/* [한국어] READ/WRITE: DRAM 요청 및 뱅크 동작 방향 식별자.
 * dram_req_t::rw와 bank_t::rw 필드에 저장되어 현재 요청이 읽기인지 쓰기인지 구별한다.
 * RTW(Read-To-Write penalty), WTR(Write-To-Read penalty) 계산 시 방향 전환 감지에 사용한다. */
#define READ 'R'  // define read and write states
#define WRITE 'W'
/* [한국어] BANK_IDLE/BANK_ACTIVE: DRAM 뱅크 상태 식별자.
 * BANK_IDLE은 Precharge 완료 후 아무 Row도 열려있지 않은 상태이고,
 * BANK_ACTIVE는 ACT(Activate) 커맨드로 특정 Row가 Row Buffer에 열린(open) 상태이다.
 * issue_row_command()가 이 상태를 보고 ACT 또는 PRE 커맨드 발행 여부를 결정한다. */
#define BANK_IDLE 'I'
#define BANK_ACTIVE 'A'

/*
 * [한국어]
 * dram_req_t - DRAM 레벨 메모리 요청 패킷
 *
 * mem_fetch(상위 캐시 레벨의 메모리 요청)를 DRAM 타이밍 모델이 직접 처리할 수 있는
 * 형태로 변환한 구조체이다. mem_fetch에는 가상 주소와 캐시 관련 메타데이터가 담겨 있지만,
 * dram_req_t는 DRAM 물리 주소 디코딩 결과(bank/row/col)와 전송 진행 상태(txbytes)를
 * 추가로 보유한다. dram_t의 mrqq(입력 큐)와 bk[n]->mrq(뱅크 배정 슬롯)에서 사용된다.
 *
 * 생성: dram_t::push() → new dram_req_t(mem_fetch, ...) — mem_fetch가 DRAM 채널에 도달할 때
 * 소멸: dram_t::cycle()에서 txbytes >= nbytes(전송 완료) 조건 충족 시 delete
 */
class dram_req_t {
 public:
  /*
   * [한국어]
   * dram_req_t 생성자 - mem_fetch를 DRAM 요청으로 변환
   *
   * @mf: 상위 캐시(L2) 미스로 인해 DRAM까지 내려온 메모리 요청 패킷
   * @banks: 이 DRAM 채널의 총 뱅크 수 (memory_config::nbk) — 뱅크 인덱스 범위 검증용
   * @dram_bnk_indexing_policy: 뱅크 인덱싱 정책 (LINEAR/XOR/IPOLY/CUSTOM)
   * @gpu: 시뮬레이터 전역 상태 접근용 (gpu_sim_cycle, gpu_tot_sim_cycle 읽기)
   *
   * mem_fetch의 tlx_addr(물리 주소 디코딩 결과)에서 row/col/bk를 추출하고,
   * 뱅크 인덱싱 정책에 따라 bk를 XOR 해싱 또는 IPOLY 해싱으로 재매핑할 수 있다.
   * 뱅크 충돌(bank conflict) 분산을 위해 LINEAR 외의 정책이 사용될 수 있다.
   *
   * 호출 체인: dram_t::push() → [dram_req_t 생성자]
   */
  dram_req_t(class mem_fetch *data, unsigned banks,
             unsigned dram_bnk_indexing_policy, class gpgpu_sim *gpu);

  unsigned int row;
  /* [한국어] 이 요청이 접근하는 DRAM Row 번호 (물리 주소 디코딩 결과의 addrdec_t::row).
   * 설정자: 생성자에서 tlx.row로 초기화.
   * 읽는 자: issue_col_command()가 bk[j]->curr_row와 비교하여 row-buffer hit 판정;
   *          issue_row_command()가 ACT 커맨드 발행 후 bk[j]->curr_row에 복사.
   * 값 범위: 0 ~ (2^row_bits - 1), memory_config에서 정해진 주소 비트 수에 의존.
   * 동기화: 단일 시뮬레이션 스레드에서만 접근되므로 별도 락 불필요. */

  unsigned int col;
  /* [한국어] 이 요청이 접근하는 DRAM Column 번호 (addrdec_t::col).
   * 설정자: 생성자에서 tlx.col로 초기화.
   * 읽는 자: dram_t::visualize(), dram_t::print() — 디버그 출력에서 col 위치 표시.
   *          실제 타이밍 제어에서는 col 값 자체보다 txbytes/dqbytes 카운터가 사용됨.
   * 값 범위: 0 ~ (2^col_bits - 1).
   * 동기화: 단일 시뮬레이션 스레드 접근. */

  unsigned int bk;
  /* [한국어] 이 요청이 배정된 DRAM Bank 번호 (뱅크 인덱싱 정책 적용 후 값).
   * 설정자: 생성자에서 dram_bnk_indexing_policy에 따라 tlx.bk, 또는 XOR/IPOLY 해시 결과로 설정.
   * 읽는 자: dram_t::scheduler_fifo()가 bk[bkn]->mrq에 이 요청을 배정할 때;
   *          frfcfs_scheduler가 뱅크별 요청 리스트 관리 시 사용.
   * 값 범위: 0 ~ (memory_config::nbk - 1).
   * 동기화: 단일 시뮬레이션 스레드 접근. */

  unsigned int nbytes;
  /* [한국어] 이 요청이 전송해야 하는 총 바이트 수 (mem_fetch::get_data_size() 반환값).
   * 설정자: 생성자에서 mf->get_data_size()로 초기화 (보통 캐시 라인 크기, 예: 128 bytes).
   * 읽는 자: issue_col_command()에서 txbytes >= nbytes 조건으로 전송 완료 판정.
   * 값 범위: 캐시 라인 크기 (gpgpusim.config의 gpgpu_cache_dl1_size 등에서 결정).
   * 동기화: 단일 시뮬레이션 스레드 접근. */

  unsigned int txbytes;
  /* [한국어] 지금까지 데이터 버스(DQ)를 통해 전송된 바이트 수 (누적).
   * 설정자: 생성자에서 0으로 초기화; issue_col_command()에서 RD/WR 커맨드 발행마다
   *         dram_atom_size씩 증가 (bk[j]->mrq->txbytes += m_config->dram_atom_size).
   * 읽는 자: issue_col_command()에서 txbytes < nbytes 조건으로 아직 전송 미완료 판정;
   *          visualize()/print()에서 col 오프셋 표시에 사용.
   * 값 범위: 0 ~ nbytes (nbytes 도달 시 전송 완료 → bk[j]->mrq = NULL).
   * 동기화: 단일 시뮬레이션 스레드 접근. */

  unsigned int dqbytes;
  /* [한국어] 데이터 큐(Data Queue, DQ)에서 실제 출력된 바이트 수 (rwq 파이프라인 통과 후).
   * 설정자: 생성자에서 0으로 초기화; dram_t::cycle()의 rwq->pop() 단계에서
   *         dram_atom_size씩 증가 (cmd->dqbytes += m_config->dram_atom_size).
   * 읽는 자: cycle()에서 dqbytes >= nbytes 조건으로 최종 전송 완료 판정 → returnq 투입.
   * 값 범위: 0 ~ nbytes; txbytes보다 tCL(CL) 사이클 지연 후 도달.
   * 동기화: 단일 시뮬레이션 스레드 접근. */

  unsigned int age;
  /* [한국어] 이 요청의 시뮬레이션 사이클 기준 나이 (aging 기반 스케줄링 지원용).
   * 설정자: 명시적 초기화 없음 (현재 미사용 필드; 향후 aging 정책 확장 예비).
   * 읽는 자: frfcfs_scheduler의 우선순위 비교 로직에서 참조 가능.
   * 값 범위: 0 이상의 사이클 수.
   * 동기화: 단일 시뮬레이션 스레드 접근. */

  unsigned int timestamp;
  /* [한국어] 이 요청이 생성된 시점의 절대 시뮬레이션 사이클 번호.
   * 설정자: 생성자에서 gpu_tot_sim_cycle + gpu_sim_cycle로 초기화.
   * 읽는 자: frfcfs_scheduler가 FCFS(First Come First Served) 우선순위 결정 시 비교;
   *          메모리 레이턴시 통계 계산 시 현재 사이클과의 차이로 대기 시간 측정.
   * 값 범위: 0 이상의 절대 사이클 번호 (시뮬레이션 시작 이후 누적).
   * 동기화: 단일 시뮬레이션 스레드 접근. */

  unsigned char rw;  // is the request a read or a write?
  /* [한국어] 이 요청의 방향: READ('R') 또는 WRITE('W').
   * 설정자: 생성자에서 mem_fetch::get_is_write() ? WRITE : READ로 초기화.
   * 읽는 자: issue_col_command()에서 RD/WR 커맨드 분기 결정;
   *          dram_t::cycle()의 BLP 통계 수집 루프에서 read_blp_rw/write_blp_rw 분류.
   * 값 범위: 'R' 또는 'W' (dram.h 상단 #define).
   * 동기화: 단일 시뮬레이션 스레드 접근. */

  unsigned long long int addr;
  /* [한국어] 이 요청의 원본 메모리 물리 주소 (byte 단위).
   * 설정자: 생성자에서 mf->get_addr()로 초기화.
   * 읽는 자: 디버그 출력 및 로깅 목적; frfcfs_scheduler가 같은 row 집합 비교 시 사용 가능.
   * 값 범위: GPU 물리 주소 공간 전체 (64비트).
   * 동기화: 단일 시뮬레이션 스레드 접근. */

  unsigned int insertion_time;
  /* [한국어] 이 요청이 mrqq(메모리 요청 큐)에 삽입된 시점의 gpu_sim_cycle 값 (부분 사이클 기준).
   * 설정자: 생성자에서 (unsigned)gpu->gpu_sim_cycle로 초기화 (커널당 사이클 카운터).
   * 읽는 자: 메모리 레이턴시 통계 (mem_latency_stat)에서 삽입→완료 구간 측정;
   *          frfcfs_scheduler의 FCFS 순서 결정 보조.
   * 값 범위: 0 ~ 현재 커널의 실행 사이클 수 (gpu_sim_cycle은 커널마다 리셋 가능).
   * 동기화: 단일 시뮬레이션 스레드 접근. */

  class mem_fetch *data;
  /* [한국어] 이 DRAM 요청에 대응하는 상위 캐시 레벨의 메모리 요청 패킷 포인터.
   * 설정자: 생성자에서 mf 인자를 그대로 저장.
   * 읽는 자: dram_t::cycle()에서 dqbytes 완료 시 returnq에 data를 push하거나
   *          set_done(data) 후 delete(L1/L2 writeback 요청인 경우);
   *          issue_col_command()에서 data->set_status()로 IN_PARTITION_DRAM 상태 갱신.
   * 값 범위: 유효한 mem_fetch 포인터 (NULL 불가; 완료 후 dram_t에서 delete 됨).
   * 동기화: 단일 시뮬레이션 스레드 접근. */

  class gpgpu_sim *m_gpu;
  /* [한국어] 시뮬레이터 전역 상태 객체 포인터.
   * 설정자: 생성자에서 gpu 인자로 초기화.
   * 읽는 자: 생성자 내 timestamp = gpu->gpu_tot_sim_cycle + gpu->gpu_sim_cycle 계산;
   *          insertion_time = (unsigned)gpu->gpu_sim_cycle 계산.
   * 값 범위: 시뮬레이션 전체에서 유효한 단일 gpgpu_sim 인스턴스 포인터.
   * 동기화: 시뮬레이션 루프 단일 스레드 환경에서만 접근. */
};

/*
 * [한국어]
 * bankgrp_t - DDR4 Bank Group 단위 타이밍 카운터 구조체
 *
 * DDR4부터 도입된 Bank Group 개념을 모델링한다. 동일 Bank Group 내의 연속 RD/WR 커맨드 간에는
 * 더 긴 CCD_L(Column-to-Column Delay, Long) 제약이 적용되어 같은 그룹 내 빠른 연속 접근이 제한된다.
 * 다른 Bank Group 간에는 CCD_S(Short)가 적용되어 더 빠른 연속 접근이 가능하므로, 스케줄러는
 * 가능하면 다른 Bank Group 요청을 교차 발행하여 처리량을 극대화한다.
 * dram_t::bkgrp[] 배열의 각 원소가 이 구조체이며, 매 사이클 DEC2ZERO 매크로로 카운터가 감소한다.
 */
struct bankgrp_t {
  unsigned int CCDLc;
  /* [한국어] CCD_L(Column-to-Column Delay Long) 카운터 — 동일 Bank Group 내 연속 RD 또는 WR 간격 제약.
   * 설정자: issue_col_command()에서 RD 또는 WR 커맨드 발행 시 bkgrp[grp]->CCDLc = m_config->tCCDL로 재설정.
   * 읽는 자: issue_col_command()에서 !bkgrp[grp]->CCDLc 조건으로 다음 커맨드 발행 가능 여부 확인;
   *          dram_t::cycle()의 BW wasted 통계 수집 루프에서 병목 카운터로 집계.
   * 값 범위: 0 ~ m_config->tCCDL; 매 사이클 DEC2ZERO(bkgrp[j]->CCDLc)로 감소; 0이면 제약 해제.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int RTPLc;
  /* [한국어] RTP_L(Read-to-Precharge Long) 카운터 — 동일 Bank Group 내 RD 후 PRE 커맨드 최소 간격.
   * 설정자: issue_col_command()에서 READ 커맨드 발행 시 bkgrp[grp]->RTPLc = m_config->tRTPL로 재설정.
   * 읽는 자: issue_row_command()에서 PRE(Precharge) 발행 조건 (!bkgrp[grp]->RTPLc) 확인.
   * 값 범위: 0 ~ m_config->tRTPL; 매 사이클 DEC2ZERO(bkgrp[j]->RTPLc)로 감소.
   * 동기화: 단일 시뮬레이션 스레드. */
};

/*
 * [한국어]
 * bank_t - DRAM 개별 뱅크의 타이밍 카운터 및 상태 구조체
 *
 * 실제 DDR SDRAM의 단일 Bank가 가지는 타이밍 제약과 상태를 사이클 단위로 모델링한다.
 * DRAM 커맨드(ACT/PRE/RD/WR) 발행 후에는 해당 타이밍 파라미터 값이 카운터에 로드되고,
 * 매 사이클 DEC2ZERO 매크로로 카운터가 0을 향해 감소한다. 0이 되면 해당 제약이 해제된다.
 * dram_t::bk[] 배열의 각 원소가 이 구조체이고, 매 사이클 cycle() 함수가 모든 뱅크를 순회한다.
 */
struct bank_t {
  unsigned int RCDc;
  /* [한국어] tRCD(Row-to-Column Delay) 카운터 — ACT 커맨드 후 RD 커맨드를 발행하기까지 기다려야 하는 사이클.
   * 설정자: issue_row_command()에서 ACT 발행 시 bk[j]->RCDc = m_config->tRCD으로 재설정.
   * 읽는 자: issue_col_command()에서 READ 발행 조건 (!bk[j]->RCDc) 확인;
   *          BW wasted 통계에서 읽기 병목 원인 집계 (RCDc_limit++).
   * 값 범위: 0 ~ m_config->tRCD; 0이면 RD 커맨드 발행 가능.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int RCDWRc;
  /* [한국어] tRCDWR(Row-to-Column Delay for Write) 카운터 — ACT 커맨드 후 WR 커맨드 발행까지 대기 사이클.
   * 설정자: issue_row_command()에서 ACT 발행 시 bk[j]->RCDWRc = m_config->tRCDWR로 재설정.
   * 읽는 자: issue_col_command()에서 WRITE 발행 조건 (!bk[j]->RCDWRc) 확인;
   *          BW wasted 통계에서 쓰기 병목 원인 집계 (RCDWRc_limit++).
   * 값 범위: 0 ~ m_config->tRCDWR; 보통 tRCD보다 짧거나 같음.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int RASc;
  /* [한국어] tRAS(Row Active to Precharge Delay) 카운터 — ACT 이후 PRE 발행까지 최소 대기 사이클.
   * 설정자: issue_row_command()에서 ACT 발행 시 bk[j]->RASc = m_config->tRAS로 재설정.
   * 읽는 자: issue_row_command()에서 PRE(Precharge) 발행 조건 (!bk[j]->RASc) 확인.
   * 값 범위: 0 ~ m_config->tRAS; tRAS 동안 PRE 불가 → row-buffer open 유지 보장.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int RPc;
  /* [한국어] tRP(Row Precharge time) 카운터 — PRE 커맨드 후 다음 ACT 발행까지 대기 사이클.
   * 설정자: issue_row_command()에서 PRE 발행 시 bk[j]->RPc = m_config->tRP로 재설정.
   * 읽는 자: issue_row_command()에서 다음 ACT 발행 조건 (!bk[j]->RPc) 확인.
   * 값 범위: 0 ~ m_config->tRP; 이 카운터가 0이어야만 뱅크를 다시 Activate 할 수 있음.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int RCc;
  /* [한국어] tRC(Row Cycle time) 카운터 — ACT→ACT 간 최소 사이클 (같은 뱅크에서 연속 Activate 간격).
   * 설정자: issue_row_command()에서 ACT 발행 시 bk[j]->RCc = m_config->tRC로 재설정.
   * 읽는 자: issue_row_command()에서 새 ACT 발행 조건 (!bk[j]->RCc) 확인.
   * 값 범위: 0 ~ m_config->tRC; tRC = tRAS + tRP로 계산되는 것이 DRAM 스펙의 통상값.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int WTPc;  // write to precharge
  /* [한국어] tWTP(Write-to-Precharge) 카운터 — WR 커맨드 발행 후 PRE 발행까지 최소 대기 사이클.
   * 설정자: issue_col_command()에서 WRITE 발행 시 bk[j]->WTPc = m_config->tWTP으로 재설정.
   * 읽는 자: issue_row_command()에서 PRE 발행 조건 (!bk[j]->WTPc) 확인.
   * 값 범위: 0 ~ m_config->tWTP; 쓰기 데이터가 DRAM 셀에 안정적으로 기록될 때까지 PRE 금지.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int RTPc;  // read to precharge
  /* [한국어] tRTP(Read-to-Precharge) 카운터 — RD 커맨드 발행 후 PRE 발행까지 최소 대기 사이클.
   * 설정자: issue_col_command()에서 READ 발행 시 bk[j]->RTPc = m_config->BL/data_command_freq_ratio로 재설정.
   * 읽는 자: issue_row_command()에서 PRE 발행 조건 (!bk[j]->RTPc) 확인.
   * 값 범위: 0 ~ (BL / data_command_freq_ratio); burst 전송이 끝날 때까지 PRE 금지.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned char rw;     // is the bank reading or writing?
  /* [한국어] 현재 이 뱅크에서 처리 중인 요청의 방향: READ('R') 또는 WRITE('W').
   * 설정자: scheduler_fifo()/scheduler_frfcfs()가 bk[n]->mrq에 요청을 배정할 때 mrq->rw로 동기화.
   * 읽는 자: BLP(Bank Level Parallelism) 통계 수집 루프에서 read_blp_rw/write_blp_rw 집계.
   * 값 범위: 'R' 또는 'W'.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned char state;  // is the bank active or idle?
  /* [한국어] 뱅크의 현재 Row Buffer 상태: BANK_IDLE('I') 또는 BANK_ACTIVE('A').
   * 설정자: dram_t 생성자에서 BANK_IDLE로 초기화;
   *         issue_row_command()에서 ACT 발행 시 BANK_ACTIVE로, PRE 발행 시 BANK_IDLE로 전환.
   * 읽는 자: issue_col_command()에서 RD/WR 발행 조건 (state == BANK_ACTIVE) 확인;
   *          issue_row_command()에서 ACT/PRE 분기 결정.
   * 값 범위: BANK_IDLE 또는 BANK_ACTIVE (dram.h 상단 #define).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int curr_row;
  /* [한국어] 현재 Row Buffer에 열려(open) 있는 Row 번호 (마지막 ACT 커맨드의 대상 row).
   * 설정자: issue_row_command()에서 ACT 발행 시 bk[j]->curr_row = bk[j]->mrq->row로 갱신.
   * 읽는 자: issue_col_command()에서 curr_row == mrq->row 비교로 row-buffer hit 판정;
   *          issue_row_command()에서 curr_row != mrq->row 비교로 row-buffer miss(PRE 필요) 판정.
   * 값 범위: 0 ~ (2^row_bits - 1); BANK_IDLE 상태에서는 의미 없는 마지막 값이 남아있을 수 있음.
   * 동기화: 단일 시뮬레이션 스레드. */

  dram_req_t *mrq;
  /* [한국어] 현재 이 뱅크에 배정되어 서비스 중인 DRAM 요청 포인터.
   * 설정자: scheduler_fifo()가 mrqq에서 pop하여 bk[bkn]->mrq에 배정;
   *         issue_col_command()에서 txbytes >= nbytes(전송 완료) 시 bk[j]->mrq = NULL로 해제.
   * 읽는 자: issue_col_command()/issue_row_command()에서 현재 요청의 row/rw 정보 접근;
   *          dram_t::cycle()의 BLP 통계 루프에서 mrq 존재 여부 확인.
   * 값 범위: 유효한 dram_req_t 포인터 또는 NULL (NULL이면 이 뱅크에 배정된 요청 없음).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int n_access;
  /* [한국어] 이 뱅크에서 발행된 총 컬럼 커맨드(RD+WR) 횟수 누적.
   * 설정자: issue_col_command()에서 RD 커맨드 발행마다 bk[j]->n_access++로 증가.
   * 읽는 자: dram_t::print()에서 "bk%d: %da %di" 형식으로 출력.
   * 값 범위: 0 이상의 누적 정수; 시뮬레이션 전체 기간의 합계.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int n_writes;
  /* [한국어] 이 뱅크에서 발행된 총 WR 커맨드 횟수 누적 (현재 코드에서 미사용 필드).
   * 설정자: 명시적으로 증가하는 코드 없음 (예비 필드로 보임).
   * 읽는 자: 미사용.
   * 값 범위: 0 (기본값; calloc으로 0 초기화).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int n_idle;
  /* [한국어] 이 뱅크에 배정된 요청(mrq)이 없어서 유휴 상태였던 사이클 수 누적.
   * 설정자: dram_t::cycle()에서 bk[j]->mrq가 NULL이고 모든 타이밍 카운터가 0일 때 bk[j]->n_idle++.
   * 읽는 자: dram_t::print()에서 "bk%d: %da %di" 형식으로 출력.
   * 값 범위: 0 이상의 누적 정수.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int bkgrpindex;
  /* [한국어] 이 뱅크가 속하는 Bank Group 번호.
   * 설정자: dram_t 생성자에서 bk[i]->bkgrpindex = i / (nbk / nbkgrp)로 초기화
   *         (뱅크 번호를 균등 분할하여 그룹 인덱스 부여).
   * 읽는 자: dram_t::get_bankgrp_number()가 이 값을 반환하여 bkgrp[] 인덱싱에 사용;
   *          issue_col_command()/issue_row_command()에서 bankgrp_t 카운터 접근 시 참조.
   * 값 범위: 0 ~ (memory_config::nbkgrp - 1).
   * 동기화: 단일 시뮬레이션 스레드. */
};

/*
 * [한국어]
 * bank_index_function - DRAM 뱅크 인덱싱 정책 열거형
 *
 * 물리 주소에서 뱅크 번호를 결정하는 매핑 함수의 종류를 정의한다.
 * 뱅크 인덱싱 정책은 gpgpusim.config의 gpgpu_dram_bnk_indexing_policy 옵션으로 선택된다.
 * LINEAR는 주소 비트를 직접 사용하므로 인접 주소가 같은 뱅크에 몰려 충돌이 발생할 수 있다.
 * BITWISE_XORING과 IPOLY는 해싱을 통해 주소를 분산시켜 뱅크 충돌 빈도를 줄이고 병렬성을 높인다.
 * 이 값은 memory_config::dram_bnk_indexing_policy에 저장되어 dram_req_t 생성자에서 사용된다.
 */
enum bank_index_function {
  LINEAR_BK_INDEX = 0,
  /* [한국어] 선형 인덱싱: 주소 비트를 변환 없이 직접 뱅크 번호로 사용.
   * 구현이 단순하지만 스트라이드 접근 패턴에서 특정 뱅크에 충돌이 집중될 수 있다. */

  BITWISE_XORING_BK_INDEX,
  /* [한국어] XOR 해싱: tlx.row의 하위 비트와 tlx.bk를 XOR하여 뱅크 번호 결정.
   * bitwise_hash_function()을 사용; 행 비트가 분산 효과를 제공하여 뱅크 충돌을 줄인다. */

  IPOLY_BK_INDEX,
  /* [한국어] IPOLY(Interleaved Polynomial) 해싱: 다항식 기반 해시로 뱅크 번호 결정.
   * ipoly_hash_function()을 사용; Rau et al. ISCA 1991 "Pseudo-randomly interleaved memory" 기법.
   * XOR보다 더 균등한 뱅크 분산을 제공하여 Row Buffer Locality를 유지하면서 충돌을 감소. */

  CUSTOM_BK_INDEX
  /* [한국어] 사용자 정의 인덱싱: dram_req_t 생성자의 CUSTOM_BK_INDEX 분기에서 직접 구현 가능.
   * 현재는 구현되지 않은 예비 옵션 (생성자에서 bk를 설정하지 않으므로 tlx.bk가 그대로 유지됨). */
};

/*
 * [한국어]
 * bank_grp_bits_position - Bank Group 번호 추출 시 비트 위치 정책 열거형
 *
 * 뱅크 번호에서 Bank Group 인덱스를 추출할 때 상위 비트를 사용할지 하위 비트를 사용할지 결정한다.
 * gpgpusim.config의 gpgpu_dram_bnkgrp_indexing_policy 옵션으로 선택되며,
 * memory_config::dram_bnkgrp_indexing_policy에 저장된다.
 * dram_t::get_bankgrp_number()에서 이 값에 따라 비트 시프트 또는 AND 마스크 연산으로
 * bank group 번호를 계산하여 bkgrp[] 배열 인덱스로 사용한다.
 */
enum bank_grp_bits_position {
  HIGHER_BITS = 0,
  /* [한국어] 상위 비트 사용: 뱅크 번호를 bk_tag_length만큼 오른쪽 시프트하여 그룹 번호 추출.
   * (예: bk=0b1010, bk_tag_length=1 → grp=0b101=5) */

  LOWER_BITS
  /* [한국어] 하위 비트 사용: 뱅크 번호에 (nbkgrp-1) AND 마스크를 적용하여 그룹 번호 추출.
   * (예: bk=0b1010, nbkgrp=4 → grp = 0b1010 & 0b0011 = 0b0010 = 2) */
};

class mem_fetch;      /* [한국어] 상위 캐시 레벨의 메모리 요청 패킷 전방 선언 — push()/returnq에서 사용 */
class memory_config;  /* [한국어] DRAM 타이밍 파라미터 집합 전방 선언 — m_config 포인터 타입 */

/*
 * [한국어]
 * dram_t - DRAM 채널(메모리 파티션) 타이밍 모델 클래스
 *
 * GPU의 하나의 DRAM 채널(memory partition)에 대한 사이클-레벨 타이밍 모델이다.
 * 실제 DDR SDRAM 컨트롤러가 수행하는 ACT(Activate), PRE(Precharge), RD(Read), WR(Write)
 * 커맨드 스케줄링과 타이밍 제약(tRCD, tRAS, tRP, tRC, tCCD, tRRD, tRTW, tWTR 등)을
 * 사이클 단위 카운터로 에뮬레이션한다. 매 DRAM 사이클마다 cycle()이 호출되어 커맨드를 발행하고
 * 타이밍 카운터를 감소시키며 처리 완료된 응답을 returnq에 투입한다.
 * FIFO와 FR-FCFS(First Ready, First Come First Served) 두 가지 스케줄러를 지원한다.
 *
 * 생성: memory_partition_unit 생성자에서 new dram_t(partition_id, config, stats, mp, gpu)
 * 호출: memory_partition_unit::dram_cycle() → dram_t::cycle() (매 DRAM 클럭)
 *       memory_partition_unit::full() → dram_t::full()
 *       memory_partition_unit::push() → dram_t::push()
 */
class dram_t {
 public:
  /*
   * [한국어]
   * dram_t 생성자 - DRAM 채널 타이밍 모델 초기화
   *
   * @partition_id: 이 DRAM 채널의 고유 번호 (GPU의 메모리 파티션 인덱스)
   * @config: DRAM 타이밍 파라미터 집합 (tRCD, tRAS, tRP, CL, BL, nbk, nbkgrp 등)
   * @stats: 메모리 접근 통계 수집 객체 (memlatstat_dram_access 호출용)
   * @mp: 이 DRAM을 소유하는 상위 memory_partition_unit (set_done(), get_mgpu() 호출용)
   * @gpu: 시뮬레이터 전역 상태 (gpu_sim_cycle, gpu_tot_sim_cycle 접근용)
   *
   * bk[] 배열(nbk개 뱅크), bkgrp[] 배열(nbkgrp개 뱅크 그룹)을 calloc으로 초기화하고
   * rwq/mrqq/returnq 파이프라인 큐를 생성한다. FR-FCFS 스케줄러 선택 시 frfcfs_scheduler를 생성한다.
   * 모든 통계 카운터를 0으로 초기화하며 mrqq_Dist 통계 히스토그램 객체를 생성한다.
   *
   * 호출 체인: memory_partition_unit 생성자 → [dram_t 생성자]
   */
  dram_t(unsigned int parition_id, const memory_config *config,
         class memory_stats_t *stats, class memory_partition_unit *mp,
         class gpgpu_sim *gpu);

  /*
   * [한국어]
   * full - DRAM 요청 큐 포화 여부 확인
   *
   * @is_write: true이면 쓰기 요청 큐, false이면 읽기 요청 큐 포화 확인
   * @return: 큐가 가득 찼으면 true (새 요청 수락 불가), 여유 있으면 false
   *
   * FR-FCFS 스케줄러 사용 시 읽기/쓰기 분리 큐 크기와 비교하고,
   * FIFO 스케줄러 사용 시 mrqq->full()을 반환한다.
   * memory_partition_unit이 새 요청을 push()하기 전에 이 함수로 수락 가능 여부를 확인한다.
   *
   * 호출 체인: memory_partition_unit::full() → [dram_t::full()]
   */
  bool full(bool is_write) const;

  /*
   * [한국어]
   * print - DRAM 채널 상세 통계 출력 (시뮬레이션 종료 시)
   *
   * @simFile: 출력 대상 FILE 포인터 (보통 stdout 또는 통계 파일)
   *
   * DRAM 타이밍 파라미터, 커맨드 수(n_cmd, n_act, n_pre, n_rd, n_wr 등),
   * BW utilization, Row Buffer Locality, Bank Level Parallelism, 병목 분석 결과를
   * 상세하게 출력한다. 시뮬레이션 완료 후 gpgpu_sim::print_stats()에서 호출된다.
   *
   * 호출 체인: gpgpu_sim::print_stats() → memory_partition_unit::print_stat()
   *           → [dram_t::print()]
   */
  void print(FILE *simFile) const;

  /*
   * [한국어]
   * visualize - 현재 DRAM 상태 콘솔 출력 (DRAM_VISUALIZE 매크로 활성화 시)
   *
   * 각 뱅크의 state, curr_row, 타이밍 카운터(RCDc, RASc, RPc, RCc), mrq 포인터,
   * 전송 진행 상태(nbytes, txbytes)를 한 줄씩 출력한다.
   * DRAM_VISUALIZE 컴파일 옵션이 활성화된 디버그 빌드에서만 호출된다.
   *
   * 호출 체인: dram_t::cycle() (DRAM_VISUALIZE ifdef) → [dram_t::visualize()]
   */
  void visualize() const;

  /*
   * [한국어]
   * print_stat - DRAM 채널 요약 통계를 파일에 출력 (주기적 통계 덤프용)
   *
   * @simFile: 출력 대상 FILE 포인터
   *
   * print()보다 간략한 형태로 핵심 카운터(n_cmd, n_act, n_pre, n_rd, n_wr, bwutil 등)만
   * 한 줄에 요약 출력한다. 주기적 통계 수집 시 호출된다.
   *
   * 호출 체인: gpu-sim.cc의 통계 덤프 루프 → [dram_t::print_stat()]
   */
  void print_stat(FILE *simFile);

  /*
   * [한국어]
   * que_length - 현재 DRAM 스케줄러 큐에 대기 중인 요청 수 반환
   *
   * @return: FR-FCFS 사용 시 frfcfs_scheduler::num_pending(), FIFO 사용 시 mrqq->get_length()
   *
   * memory_partition_unit이 큐 길이 통계 수집이나 부하 확인 용도로 호출한다.
   *
   * 호출 체인: memory_partition_unit → [dram_t::que_length()]
   */
  unsigned que_length() const;

  /*
   * [한국어]
   * returnq_full - 완료 응답 큐(returnq) 포화 여부 확인
   *
   * @return: returnq가 가득 찼으면 true
   *
   * dram_t::cycle()에서 rwq로부터 완료된 요청을 returnq에 push하기 전에 확인한다.
   * returnq가 가득 차면 rwq pop 자체를 보류하여 역압력(backpressure)이 발생한다.
   *
   * 호출 체인: dram_t::cycle() → [dram_t::returnq_full()]
   */
  bool returnq_full() const;

  /*
   * [한국어]
   * queue_limit - DRAM 스케줄러 큐 최대 크기 반환
   *
   * @return: m_config->gpgpu_frfcfs_dram_sched_queue_size (0이면 무제한)
   *
   * 생성자에서 mrqq_Dist 통계 객체 생성 시 최대 추적 범위 결정에 사용된다.
   *
   * 호출 체인: dram_t 생성자 → [dram_t::queue_limit()]
   */
  unsigned int queue_limit() const;

  /*
   * [한국어]
   * visualizer_print - AerialVision 시각화 도구용 gzip 압축 통계 출력
   *
   * @visualizer_file: gzip 압축 출력 스트림 (AerialVision .gz 로그 파일)
   *
   * 매 통계 수집 인터벌마다 DRAM 커맨드 수, NOP, ACT, PRE, 요청 수, 평균 큐 길이,
   * BW utilization, efficiency 등을 gzprintf로 출력한 후 partial 카운터를 리셋한다.
   * 또한 뱅크별 접근 유형(GLOBAL_ACC_R/W, LOCAL_ACC_R/W, CONST_ACC_R, TEXTURE_ACC_R) 통계도 출력한다.
   *
   * 호출 체인: gpu-sim.cc visualizer 루프 → memory_partition_unit → [dram_t::visualizer_print()]
   */
  void visualizer_print(gzFile visualizer_file);

  /*
   * [한국어]
   * return_queue_pop - 처리 완료된 mem_fetch를 returnq에서 꺼내 반환
   *
   * @return: 처리 완료된 mem_fetch 포인터 (returnq가 비어있으면 NULL)
   *
   * DRAM 처리 및 CL(Column Latency) 파이프라인을 모두 통과한 READ 응답 패킷을
   * memory_partition_unit이 주기적으로 pop하여 ICNT를 통해 SM으로 회신한다.
   * WRITE 요청(writeback)은 returnq가 아닌 set_done()으로 완료 처리된다.
   *
   * 호출 체인: memory_partition_unit::dram_cycle() → [dram_t::return_queue_pop()]
   *           → mem_fetch가 ICNT를 통해 SM으로 반환
   */
  class mem_fetch *return_queue_pop();

  /*
   * [한국어]
   * return_queue_top - returnq의 맨 앞 항목 반환 (pop하지 않음)
   *
   * @return: returnq 맨 앞의 mem_fetch 포인터 (비어있으면 NULL)
   *
   * memory_partition_unit이 returnq에 데이터가 있는지 확인하거나
   * pop 전에 peek하는 용도로 사용한다.
   *
   * 호출 체인: memory_partition_unit → [dram_t::return_queue_top()]
   */
  class mem_fetch *return_queue_top();

  /*
   * [한국어]
   * push - 새 메모리 요청을 DRAM 채널에 투입
   *
   * @data: DRAM까지 전달된 메모리 요청 패킷 (L2 캐시 미스 등으로 발생)
   *
   * mem_fetch로부터 dram_req_t를 생성하여 mrqq(또는 FR-FCFS 스케줄러)에 삽입한다.
   * mem_fetch의 상태를 IN_PARTITION_MC_INTERFACE_QUEUE로 갱신하고 통계를 기록한다.
   * 호출 전 full()로 큐 포화 여부를 반드시 확인해야 한다.
   *
   * 호출 체인: memory_partition_unit::push() → [dram_t::push()]
   *           → new dram_req_t() → mrqq->push()
   */
  void push(class mem_fetch *data);

  /*
   * [한국어]
   * cycle - DRAM 사이클 단위 타이밍 진행 함수 (핵심 함수)
   *
   * 매 DRAM 클럭 사이클마다 memory_partition_unit::dram_cycle()에 의해 호출된다.
   * 수행 순서:
   *   1. rwq(RD/WR 파이프라인 큐)에서 완료된 요청 pop → returnq에 투입 (READ) 또는 set_done() (WRITE)
   *   2. 스케줄러(FIFO 또는 FR-FCFS) 실행 → 다음 서비스할 요청을 뱅크에 배정
   *   3. 각 뱅크에 대해 issue_col_command() (RD/WR) 또는 issue_row_command() (ACT/PRE) 발행
   *   4. BW utilization, Row Buffer Locality, BLP 통계 수집
   *   5. 모든 타이밍 카운터(RRDc, CCDc, RTWc, WTRc, 각 뱅크의 RCDc 등) DEC2ZERO 감소
   *
   * 실행 컨텍스트: 호스트 CPU 시뮬레이션 루프 단일 스레드
   * 호출 체인: gpu-sim.cc::gpgpu_sim::cycle() → memory_partition_unit::dram_cycle()
   *           → [dram_t::cycle()] → issue_col_command() / issue_row_command()
   */
  void cycle();

  /*
   * [한국어]
   * dram_log - DRAM 로그 파일 기록 (로깅 기능)
   *
   * @task: 로그 작업 유형 (0: 초기화/열기, 1: 현재 상태 기록, 2: 닫기 등)
   *
   * DRAM_LOG 컴파일 옵션이 활성화된 경우 DRAM 커맨드 이력을 파일에 기록한다.
   *
   * 호출 체인: dram_t::cycle() → [dram_t::dram_log()]
   */
  void dram_log(int task);

  class memory_partition_unit *m_memory_partition_unit;
  /* [한국어] 이 dram_t를 소유하는 상위 memory_partition_unit 포인터.
   * 설정자: 생성자에서 mp 인자로 초기화.
   * 읽는 자: dram_t::cycle()에서 READ 외의 요청(writeback) 완료 시 set_done(data) 호출;
   *          dram_req_t 생성자에서 get_mgpu()를 통해 gpgpu_sim 포인터 획득.
   * 값 범위: 유효한 memory_partition_unit 포인터 (NULL 불가).
   * 동기화: 단일 시뮬레이션 스레드. */

  class gpgpu_sim *m_gpu;
  /* [한국어] 시뮬레이터 전역 상태 객체 포인터.
   * 설정자: 생성자에서 gpu 인자로 초기화.
   * 읽는 자: push(), cycle() 등에서 gpu_sim_cycle + gpu_tot_sim_cycle로 타임스탬프 계산;
   *          mem_fetch::set_status() 호출 시 현재 사이클 번호 전달.
   * 값 범위: 시뮬레이션 전체에서 유효한 단일 gpgpu_sim 인스턴스.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int id;
  /* [한국어] 이 DRAM 채널의 고유 식별자 (메모리 파티션 인덱스).
   * 설정자: 생성자에서 partition_id 인자로 초기화.
   * 읽는 자: print()/print_stat()에서 "DRAM[%d]" 형식으로 채널 번호 출력;
   *          visualizer_print()에서 출력 태그에 포함.
   * 값 범위: 0 ~ (num_mem_controllers - 1).
   * 동기화: 단일 시뮬레이션 스레드. */

  // Power Model
  /*
   * [한국어]
   * set_dram_power_stats - AccelWattch 전력 모델에 DRAM 성능 카운터 노출
   *
   * @cmd, @activity, @nop, @act, @pre, @rd, @wr, @wr_WB, @req: 출력 참조 변수들
   *   (각각 n_cmd, n_activity, n_nop, n_act, n_pre, n_rd, n_wr, n_wr_WB, n_req에 대응)
   *
   * AccelWattch(전력 모델)가 매 전력 샘플링 인터벌마다 이 함수를 호출하여
   * DRAM 동작 카운터를 수집한다. McPAT 기반 동적/정적 전력 계산의 입력 데이터가 된다.
   *
   * 호출 체인: accelwattch/power_interface.cc → [dram_t::set_dram_power_stats()]
   */
  void set_dram_power_stats(unsigned &cmd, unsigned &activity, unsigned &nop,
                            unsigned &act, unsigned &pre, unsigned &rd,
                            unsigned &wr, unsigned &wr_WB, unsigned &req) const;

  const memory_config *m_config;
  /* [한국어] DRAM 타이밍 파라미터 집합 포인터 (읽기 전용).
   * 설정자: 생성자에서 config 인자로 초기화.
   * 읽는 자: cycle(), issue_col_command(), issue_row_command() 전반에서
   *          tRCD, tRAS, tRP, tRC, tCCD, tRRD, tRTW, tWTR, BL, CL, WL, nbk, nbkgrp 등 참조.
   * 값 범위: gpgpusim.config에서 파싱된 유효한 memory_config 인스턴스 포인터.
   * 동기화: 읽기 전용 (수정 없음). */

 private:
  bankgrp_t **bkgrp;
  /* [한국어] Bank Group별 타이밍 카운터 배열 (크기: nbkgrp).
   * 설정자: 생성자에서 calloc으로 nbkgrp개 bankgrp_t 배열 할당 후 포인터 배열로 초기화.
   * 읽는 자: issue_col_command()/issue_row_command()에서 get_bankgrp_number(j)로 그룹 번호 얻은 후
   *          bkgrp[grp]->CCDLc, bkgrp[grp]->RTPLc 접근.
   * 값 범위: nbkgrp개의 bankgrp_t 포인터 (NULL 불가; calloc으로 0 초기화됨).
   * 동기화: 단일 시뮬레이션 스레드. */

  bank_t **bk;
  /* [한국어] 뱅크별 타이밍 카운터 및 상태 배열 (크기: nbk).
   * 설정자: 생성자에서 calloc으로 nbk개 bank_t 배열 할당 후 포인터 배열로 초기화;
   *         각 bk[i]->state = BANK_IDLE, bkgrpindex = i/(nbk/nbkgrp) 설정.
   * 읽는 자: cycle(), issue_col_command(), issue_row_command() 전반에서 뱅크별 상태 접근.
   * 값 범위: nbk개의 bank_t 포인터 (NULL 불가).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int prio;
  /* [한국어] 다음 커맨드 발행 시 시작 뱅크 인덱스 (라운드로빈 우선순위).
   * 설정자: 생성자에서 0으로 초기화; issue_row_command()에서 ACT 또는 PRE 발행 시
   *         prio = (j + 1) % nbk 로 다음 뱅크로 이동.
   * 읽는 자: cycle() 내 뱅크 순회 루프에서 j = (i + prio) % nbk 로 시작 뱅크 결정.
   * 값 범위: 0 ~ (nbk - 1).
   * 동기화: 단일 시뮬레이션 스레드. */

  /*
   * [한국어]
   * get_bankgrp_number - 뱅크 번호에서 Bank Group 번호 계산
   *
   * @i: 뱅크 번호 (0 ~ nbk-1)
   * @return: 이 뱅크가 속하는 Bank Group 번호 (0 ~ nbkgrp-1)
   *
   * dram_bnkgrp_indexing_policy에 따라 HIGHER_BITS(오른쪽 시프트) 또는
   * LOWER_BITS(AND 마스크)로 그룹 번호를 추출한다.
   * issue_col_command()/issue_row_command()/cycle() 통계 루프에서 bkgrp[] 배열 인덱싱에 사용.
   *
   * 호출 체인: cycle()/issue_col_command()/issue_row_command() → [get_bankgrp_number()]
   */
  unsigned get_bankgrp_number(unsigned i);

  /*
   * [한국어]
   * scheduler_fifo - FIFO 스케줄러: mrqq 맨 앞 요청을 해당 뱅크에 배정
   *
   * mrqq가 비어있지 않으면 top() 요청의 bk 필드를 읽어 bk[bkn]->mrq가 NULL인 경우에만
   * mrqq->pop()으로 꺼내 배정한다. 해당 뱅크가 이미 다른 요청을 서비스 중이면 대기한다.
   * 단순한 FCFS 정책으로 Row Buffer Hit/Miss를 고려하지 않아 효율이 낮을 수 있다.
   *
   * 실행 컨텍스트: dram_t::cycle() 내부 (매 DRAM 사이클)
   * 호출 체인: dram_t::cycle() → [scheduler_fifo()] → mrqq->pop() → bk[bkn]->mrq 배정
   */
  void scheduler_fifo();

  /*
   * [한국어]
   * scheduler_frfcfs - FR-FCFS 스케줄러: Row Buffer Hit 요청 우선 배정
   *
   * frfcfs_scheduler 객체에 위임하여 Row Buffer Hit(현재 열린 Row에 접근하는 요청)을
   * 우선 선택하고, 같은 우선순위에서는 도착 순서(FCFS)로 배정한다.
   * FR-FCFS는 FIFO 대비 Row Buffer Locality를 극대화하여 처리량을 높이지만
   * Row-miss 요청의 기아(starvation) 가능성이 있다.
   *
   * 실행 컨텍스트: dram_t::cycle() 내부 (매 DRAM 사이클)
   * 호출 체인: dram_t::cycle() → [scheduler_frfcfs()] → frfcfs_scheduler::schedule()
   */
  void scheduler_frfcfs();

  /*
   * [한국어]
   * issue_col_command - 컬럼 커맨드(RD 또는 WR) 발행 시도
   *
   * @j: 커맨드 발행을 시도할 뱅크 번호
   * @return: 커맨드를 발행했으면 true, 타이밍 제약으로 발행 못하면 false
   *
   * bk[j]->mrq가 존재하고, 다음 조건이 모두 충족될 때 RD 또는 WR 커맨드를 발행한다:
   *   - CCDc == 0 (채널 전체 CCD 제약 해제)
   *   - bkgrp[grp]->CCDLc == 0 (Bank Group 내 CCD_L 제약 해제)
   *   - RCDc == 0 (이 뱅크의 ACT→RD 딜레이 해제)
   *   - curr_row == mrq->row (Row Buffer Hit — 맞는 Row가 열려있음)
   *   - state == BANK_ACTIVE
   *   - WTRc == 0 (READ의 경우 이전 WRITE 후 WTR 패널티 해제)
   *   - RTWc == 0 (WRITE의 경우 이전 READ 후 RTW 패널티 해제)
   *   - !rwq->full() (RD/WR 파이프라인 큐 여유 있음)
   * 발행 시 rwq에 요청을 push하고 관련 타이밍 카운터(CCDc, CCDLc, RTWc/WTRc, RTPc/WTPc)를 설정한다.
   * txbytes가 nbytes에 도달하면 전송 완료로 bk[j]->mrq = NULL 처리한다.
   *
   * 실행 컨텍스트: dram_t::cycle() 내부 (매 DRAM 사이클)
   * 호출 체인: dram_t::cycle() → [issue_col_command(j)] → rwq->push()
   */
  bool issue_col_command(int j);

  /*
   * [한국어]
   * issue_row_command - 로우 커맨드(ACT 또는 PRE) 발행 시도
   *
   * @j: 커맨드 발행을 시도할 뱅크 번호
   * @return: 커맨드를 발행했으면 true, 타이밍 제약으로 발행 못하면 false
   *
   * bk[j]->mrq가 존재하는 경우에 한해 두 가지 케이스를 처리한다:
   *   ACT(Activate): state == BANK_IDLE && !RRDc && !RPc && !RCc
   *     → bk[j]->curr_row = mrq->row, state = BANK_ACTIVE
   *     → RRDc, RCDc, RCDWRc, RASc, RCc 카운터 설정
   *   PRE(Precharge): curr_row != mrq->row && state == BANK_ACTIVE
   *                   && !RASc && !WTPc && !RTPc && !bkgrp[grp]->RTPLc
   *     → state = BANK_IDLE, bk[j]->RPc 설정
   * ACT/PRE 발행 후 prio = (j+1)%nbk로 라운드로빈 우선순위 전진.
   *
   * 실행 컨텍스트: dram_t::cycle() 내부 (매 DRAM 사이클)
   * 호출 체인: dram_t::cycle() → [issue_row_command(j)] → 뱅크 상태 전환
   */
  bool issue_row_command(int j);

  unsigned int RRDc;
  /* [한국어] tRRD(Rank-to-Rank Delay) 카운터 — 서로 다른 뱅크 간 ACT 커맨드 최소 간격.
   * 설정자: issue_row_command()에서 ACT 발행 시 RRDc = m_config->tRRD로 재설정.
   * 읽는 자: issue_row_command()에서 새 ACT 발행 조건 (!RRDc) 확인.
   * 값 범위: 0 ~ m_config->tRRD; 전체 DRAM 채널에 공유 적용.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int CCDc;
  /* [한국어] tCCD(Column-to-Column Delay) 카운터 — 다른 Bank Group 간 연속 RD/WR 커맨드 최소 간격.
   * 설정자: issue_col_command()에서 RD/WR 발행 시 CCDc = m_config->tCCD로 재설정.
   * 읽는 자: issue_col_command()에서 새 RD/WR 발행 조건 (!CCDc) 확인;
   *          BW wasted 통계에서 병목 카운터로 집계 (CCDc_limit++).
   * 값 범위: 0 ~ m_config->tCCD; 전체 DRAM 채널에 공유 적용.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int RTWc;  // read to write penalty applies across banks
  /* [한국어] tRTW(Read-to-Write Turn-around) 카운터 — 채널 전체에 RD 후 WR 방향 전환 패널티.
   * 설정자: issue_col_command()에서 RD 커맨드 발행 시 RTWc = m_config->tRTW로 재설정.
   * 읽는 자: issue_col_command()에서 WR 발행 조건 (RTWc == 0) 확인;
   *          BW wasted 통계에서 RTWc_limit++ 집계.
   * 값 범위: 0 ~ m_config->tRTW; 전체 채널에 공유 (모든 뱅크의 WR을 일시 차단).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int WTRc;  // write to read penalty applies across banks
  /* [한국어] tWTR(Write-to-Read Turn-around) 카운터 — 채널 전체에 WR 후 RD 방향 전환 패널티.
   * 설정자: issue_col_command()에서 WR 커맨드 발행 시 WTRc = m_config->tWTR로 재설정.
   * 읽는 자: issue_col_command()에서 RD 발행 조건 (WTRc == 0) 확인;
   *          BW wasted 통계에서 WTRc_limit++ 집계.
   * 값 범위: 0 ~ m_config->tWTR; 전체 채널에 공유 (모든 뱅크의 RD를 일시 차단).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned char
      rw;  // was last request a read or write? (important for RTW, WTR)
  /* [한국어] 채널 전체에서 마지막으로 발행된 커맨드의 방향 (READ 또는 WRITE).
   * 설정자: 생성자에서 READ로 초기화; issue_col_command()에서 RD/WR 방향이 전환될 때 갱신.
   * 읽는 자: issue_col_command()에서 rw == WRITE 상태에서 RD 발행 시 rwq->set_min_length(CL)로
   *          파이프라인 길이 재조정; 반대 방향 전환 시 WL로 조정.
   * 값 범위: 'R' 또는 'W'.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int pending_writes;
  /* [한국어] 현재 미완료 WRITE 요청 수 (쓰기-읽기 전환 정책 보조용).
   * 설정자/읽는 자: 현재 코드에서 명시적으로 사용되지 않는 예비 필드.
   * 값 범위: 0 이상의 정수.
   * 동기화: 단일 시뮬레이션 스레드. */

  fifo_pipeline<dram_req_t> *rwq;
  /* [한국어] RD/WR 데이터 전송 파이프라인 큐 (CL 또는 WL 깊이의 지연 큐).
   * 목적: RD 커맨드 발행 후 CL(Column Latency) 사이클이 지나야 데이터가 출력되는
   *        물리적 파이프라인을 모델링. WR는 WL(Write Latency) 적용.
   * 설정자: 생성자에서 new fifo_pipeline("rwq", CL, CL+1)로 생성.
   *         issue_col_command()에서 push; cycle()에서 pop.
   * 읽는 자: cycle()이 매 사이클 pop()하여 dqbytes를 증가시키고 완료 여부 확인.
   * 값 범위: 0 ~ CL+1 개 dram_req_t 항목; full()이면 새 커맨드 발행 차단.
   * 동기화: 단일 시뮬레이션 스레드. */

  fifo_pipeline<dram_req_t> *mrqq;
  /* [한국어] DRAM 입력 메모리 요청 큐 (FIFO 스케줄러 사용 시 대기 큐).
   * 목적: push()로 들어온 dram_req_t가 스케줄러에 의해 뱅크에 배정되기 전까지 대기.
   * 설정자: 생성자에서 new fifo_pipeline("mrqq", 0, 2)로 생성; push()에서 mrqq->push(mrq).
   * 읽는 자: scheduler_fifo()가 top()/pop()으로 맨 앞 요청 꺼내 뱅크 배정.
   * 값 범위: FIFO 사용 시 최대 2개 항목 (용량 2); FR-FCFS는 frfcfs_scheduler 내부 큐 사용.
   * 동기화: 단일 시뮬레이션 스레드. */

  // buffer to hold packets when DRAM processing is over
  // should be filled with dram clock and popped with l2or icnt clock
  fifo_pipeline<mem_fetch> *returnq;
  /* [한국어] DRAM 처리 완료된 READ 응답 패킷 대기 큐.
   * 목적: rwq 파이프라인을 모두 통과한 READ mem_fetch를 보관하다가
   *        memory_partition_unit이 ICNT 클럭에 맞춰 pop하여 SM으로 회신.
   *        DRAM 클럭(빠름)과 ICNT 클럭(느림) 간의 속도 차이를 완충하는 버퍼 역할.
   * 설정자: 생성자에서 gpgpu_dram_return_queue_size (0이면 1024) 용량으로 생성;
   *         cycle()에서 dqbytes >= nbytes 조건 충족 시 READ 요청의 data를 push.
   * 읽는 자: memory_partition_unit::dram_cycle()이 return_queue_pop()으로 pop.
   * 값 범위: 0 ~ gpgpu_dram_return_queue_size 개 mem_fetch.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int dram_util_bins[10];
  /* [한국어] DRAM BW 사용률(utilization) 구간별 사이클 수 히스토그램 (0~10%, 10~20%, ..., 90~100%).
   * 설정자: visualizer_print() 또는 주기적 통계 수집 시 갱신 (현재 코드에서는 print()에서 출력만).
   * 읽는 자: print()/print_stat()에서 "dram_util_bins:" 줄 출력.
   * 값 범위: 10개 원소, 각각 0 이상의 누적 카운터.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int dram_eff_bins[10];
  /* [한국어] DRAM BW 효율(efficiency) 구간별 사이클 수 히스토그램 (util_bw / n_activity 기준).
   * 설정자: 주기적 통계 수집 시 갱신.
   * 읽는 자: print()/print_stat()에서 "dram_eff_bins:" 줄 출력.
   * 값 범위: 10개 원소.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int last_n_cmd, last_n_activity, last_bwutil;
  /* [한국어] 이전 샘플링 인터벌의 n_cmd, n_activity, bwutil 스냅샷.
   * 설정자: 주기적 통계 샘플링 시 현재 값을 저장.
   * 읽는 자: 현재 인터벌의 변화량(delta) 계산에 사용.
   * 값 범위: 0 이상의 누적 정수.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long n_cmd;
  /* [한국어] 이 DRAM 채널에서 처리된 총 커맨드 사이클 수 (매 cycle() 호출마다 1 증가).
   * 설정자: cycle()에서 n_cmd++.
   * 읽는 자: print()/print_stat()에서 BW util 분모 (bwutil/n_cmd); set_dram_power_stats()에서 노출.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long n_activity;
  /* [한국어] DRAM이 유휴하지 않은(활성) 사이클 수 — 적어도 하나의 뱅크가 타이밍 제약을 갖고 있을 때.
   * 설정자: cycle()에서 k > 0 (비유휴 뱅크가 있음)일 때 n_activity++.
   * 읽는 자: print()에서 dram_eff 분모 (bwutil/n_activity); set_dram_power_stats()에서 노출.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long n_nop;
  /* [한국어] 아무 커맨드도 발행되지 않은 NOP 사이클 수 누적.
   * 설정자: cycle()에서 issued == false일 때 n_nop++.
   * 읽는 자: print()에서 출력; set_dram_power_stats()에서 전력 모델에 노출.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long n_act;
  /* [한국어] 발행된 ACT(Activate) 커맨드 총 횟수.
   * 설정자: issue_row_command()에서 ACT 발행 시 n_act++.
   * 읽는 자: print()/print_stat()에서 출력; set_dram_power_stats()에서 전력 모델에 노출.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long n_pre;
  /* [한국어] 발행된 PRE(Precharge) 커맨드 총 횟수.
   * 설정자: issue_row_command()에서 PRE 발행 시 n_pre++.
   * 읽는 자: print()/print_stat()에서 출력; set_dram_power_stats()에서 전력 모델에 노출.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long n_ref;
  /* [한국어] DRAM Refresh 커맨드 횟수 (현재 코드에서 Refresh 시뮬레이션은 미구현, 통계만 존재).
   * 설정자: 명시적으로 증가하는 코드 없음 (0으로 초기화된 채 유지).
   * 읽는 자: print()/print_stat()에서 "n_ref_event=%llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long n_rd;
  /* [한국어] 일반 READ 커맨드 발행 횟수 (L2_WR_ALLOC_R 제외한 순수 읽기).
   * 설정자: issue_col_command()에서 READ 발행 시, 접근 유형이 L2_WR_ALLOC_R가 아니면 n_rd++.
   * 읽는 자: print()에서 출력; set_dram_power_stats()에서 전력 모델에 노출.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long n_rd_L2_A;
  /* [한국어] L2 Write-Allocate를 위한 READ 커맨드 발행 횟수 (L2_WR_ALLOC_R 유형 요청).
   * 설정자: issue_col_command()에서 READ 발행 시, 접근 유형이 L2_WR_ALLOC_R이면 n_rd_L2_A++.
   * 읽는 자: print()에서 "L2_Alloc = %llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long n_wr;
  /* [한국어] 일반 WRITE 커맨드 발행 횟수 (L2_WRBK_ACC 제외한 순수 쓰기).
   * 설정자: issue_col_command()에서 WRITE 발행 시, 접근 유형이 L2_WRBK_ACC가 아니면 n_wr++.
   * 읽는 자: print()/print_stat()에서 출력; set_dram_power_stats()에서 전력 모델에 노출.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long n_wr_WB;
  /* [한국어] L2 Writeback(Dirty Eviction)에 의한 WRITE 커맨드 발행 횟수 (L2_WRBK_ACC 유형).
   * 설정자: issue_col_command()에서 WRITE 발행 시, 접근 유형이 L2_WRBK_ACC이면 n_wr_WB++.
   * 읽는 자: print()에서 "L2_WB = %llu" 출력; set_dram_power_stats()에서 wr_WB로 노출.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long n_req;
  /* [한국어] 이 DRAM 채널로 도착한 총 메모리 요청 수 (push() 호출 횟수).
   * 설정자: push()에서 n_req++, n_req_partial++.
   * 읽는 자: print()/print_stat()에서 출력; set_dram_power_stats()에서 전력 모델에 노출.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long max_mrqs_temp;
  /* [한국어] push() 이후 현재까지 관측된 큐 최대 길이 (임시 추적 — print_stat() 호출 시 리셋).
   * 설정자: push()에서 현재 큐 길이가 max_mrqs_temp를 초과하면 갱신; print_stat()에서 0으로 리셋.
   * 읽는 자: print_stat()에서 "mrqsmax=%llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  // some statistics to see where BW is wasted?
  unsigned long long wasted_bw_row;
  /* [한국어] Row 커맨드(ACT/PRE) 때문에 Column 커맨드를 발행하지 못한 낭비 사이클 수.
   * 설정자: cycle()에서 memory_pending_found > 0 && !memory_pending_rw_found 조건에서 wasted_bw_row++.
   * 의미: 요청은 있지만 Row Buffer가 아직 준비되지 않아(ACT/PRE 진행 중) BW가 낭비됨.
   * 읽는 자: print()에서 "Wasted_Row = %llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long wasted_bw_col;
  /* [한국어] Row Buffer Hit 요청이 있음에도 타이밍 제약(CCDc, WTRc, RTWc, RCDc 등)으로
   *         Column 커맨드를 발행하지 못한 낭비 사이클 수.
   * 설정자: cycle()에서 memory_pending_rw_found == true && !issued_col_cmd && !CCDc 조건에서 wasted_bw_col++.
   * 읽는 자: print()에서 "Wasted_Col = %llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long util_bw;
  /* [한국어] 실제 데이터 전송이 이루어진 유효 BW 사이클 수 (issued_col_cmd 또는 CCDc > 0인 사이클).
   * 설정자: cycle()에서 issued_col_cmd || CCDc 조건일 때 util_bw++.
   * 읽는 자: print()에서 "util_bw = %llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long idle_bw;
  /* [한국어] 어떤 요청도 없어서 DRAM이 완전히 유휴한 사이클 수.
   * 설정자: cycle()에서 !memory_pending_found 조건일 때 idle_bw++.
   * 읽는 자: print()에서 "Idle = %llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long RCDc_limit;
  /* [한국어] READ 요청이 있음에도 tRCD 카운터 때문에 RD 커맨드를 발행 못한 사이클 수.
   * 설정자: cycle()의 BW 병목 분석 루프에서 bk[j]->RCDc != 0일 때 RCDc_limit++.
   * 읽는 자: print()에서 "RCDc_limit = %llu" 출력 (병목 원인 분석).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long CCDLc_limit;
  /* [한국어] Bank Group 내 CCD_L 카운터 때문에 RD/WR 커맨드를 발행 못한 사이클 수.
   * 설정자: cycle()의 BW 병목 분석 루프에서 bkgrp[grp]->CCDLc != 0일 때 CCDLc_limit++.
   * 읽는 자: print()에서 "CCDLc_limit = %llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long CCDLc_limit_alone;
  /* [한국어] WTR/RTW 등 다른 제약 없이 오직 CCDLc만이 단독 병목인 사이클 수.
   * 설정자: cycle()에서 CCDLc != 0 && WTRc == 0 (READ의 경우) 조건일 때 CCDLc_limit_alone++.
   * 읽는 자: print()에서 "CCDLc_limit_alone = %llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long CCDc_limit;
  /* [한국어] 채널 전체 CCD 카운터 때문에 RD/WR 커맨드를 발행 못한 사이클 수.
   * 설정자: cycle()의 BW 병목 분석 루프에서 CCDc != 0일 때 CCDc_limit++.
   * 읽는 자: print()에서 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long WTRc_limit;
  /* [한국어] WTR(Write-to-Read) 카운터 때문에 READ 커맨드를 발행 못한 사이클 수.
   * 설정자: cycle()에서 WTRc != 0 && READ 요청 대기 중일 때 WTRc_limit++.
   * 읽는 자: print()에서 "WTRc_limit = %llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long WTRc_limit_alone;
  /* [한국어] CCDLc 등 다른 제약 없이 오직 WTRc만이 단독 병목인 사이클 수.
   * 설정자: cycle()에서 WTRc != 0 && CCDLc == 0 조건일 때 WTRc_limit_alone++.
   * 읽는 자: print()에서 "WTRc_limit_alone = %llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long RCDWRc_limit;
  /* [한국어] WRITE 요청이 있음에도 tRCDWR 카운터 때문에 WR 커맨드를 발행 못한 사이클 수.
   * 설정자: cycle()의 BW 병목 분석 루프에서 bk[j]->RCDWRc != 0일 때 RCDWRc_limit++.
   * 읽는 자: print()에서 "RCDWRc_limit = %llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long RTWc_limit;
  /* [한국어] RTW(Read-to-Write) 카운터 때문에 WRITE 커맨드를 발행 못한 사이클 수.
   * 설정자: cycle()에서 RTWc != 0 && WRITE 요청 대기 중일 때 RTWc_limit++.
   * 읽는 자: print()에서 "RTWc_limit = %llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long RTWc_limit_alone;
  /* [한국어] CCDLc 등 다른 제약 없이 오직 RTWc만이 단독 병목인 사이클 수.
   * 설정자: cycle()에서 RTWc != 0 && CCDLc == 0 조건일 때 RTWc_limit_alone++.
   * 읽는 자: print()에서 "RTWc_limit_alone = %llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long rwq_limit;
  /* [한국어] rwq(RD/WR 파이프라인 큐) 포화로 인해 새 Column 커맨드를 발행 못한 사이클 수.
   * 설정자: cycle()의 BW 병목 분석 루프에서 rwq->full()일 때 rwq_limit++.
   * 읽는 자: print()에서 "rwq = %llu" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  // row locality, BLP and other statistics
  unsigned long long access_num;
  /* [한국어] 총 DRAM 접근 수 (n_rd + n_rd_L2_A + n_wr + n_wr_WB 합계).
   * 설정자: issue_col_command()에서 RD/WR 커맨드 발행 시마다 증가 (현재 코드에서는 dram_req_t 생성자나 push 시 증가).
   * 읽는 자: print()에서 Row Buffer Locality 분모 (hits_num / access_num).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long read_num;
  /* [한국어] 총 READ 접근 수 누적.
   * 읽는 자: print()에서 Row Buffer Locality_read 분모 (hits_read_num / read_num).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long write_num;
  /* [한국어] 총 WRITE 접근 수 누적.
   * 읽는 자: print()에서 Row Buffer Locality_write 분모 (hits_write_num / write_num).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long hits_num;
  /* [한국어] Row Buffer Hit(현재 열린 Row에 접근한) 요청 수 누적 (READ + WRITE 합산).
   * 읽는 자: print()에서 Row Buffer Locality = hits_num / access_num 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long hits_read_num;
  /* [한국어] READ 요청 중 Row Buffer Hit 수 누적.
   * 읽는 자: print()에서 Row Buffer Locality_read 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long hits_write_num;
  /* [한국어] WRITE 요청 중 Row Buffer Hit 수 누적.
   * 읽는 자: print()에서 Row Buffer Locality_write 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long banks_1time;
  /* [한국어] 사이클별 mrq를 보유한 뱅크 수의 총합 (BLP 분자).
   * 설정자: cycle()의 BLP 통계 루프에서 banks_1time += memory_pending (memory_pending: mrq 있는 뱅크 수).
   * 읽는 자: print()에서 BLP = banks_1time / banks_acess_total 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long banks_acess_total;
  /* [한국어] 적어도 하나의 뱅크에 mrq가 있었던 사이클 수 (BLP 분모).
   * 설정자: cycle()에서 memory_pending > 0일 때 banks_acess_total++.
   * 읽는 자: print()에서 Bank Level Parallelism 분모로 사용.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long banks_acess_total_after;
  /* [한국어] 커맨드 발행 처리 이후(after) 적어도 하나의 뱅크에 mrq가 있었던 사이클 수.
   * 설정자: cycle()의 후반부 통계 루프에서 memory_pending_found > 0이면 banks_acess_total_after++.
   * 읽는 자: 현재 print()에서 직접 출력되지 않음 (내부 분석용 예비 통계).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long banks_time_rw;
  /* [한국어] Row Buffer Hit 상태로 RD/WR 커맨드를 발행 가능한 뱅크 수의 사이클별 총합.
   * 설정자: cycle()에서 memory_pending_rw 변수로 집계 후 banks_time_rw += memory_pending_rw.
   * 읽는 자: print()에서 BLP_Col = banks_time_rw / banks_access_rw_total 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long banks_access_rw_total;
  /* [한국어] Row Buffer Hit 상태인 뱅크가 하나라도 있었던 사이클 수 (banks_time_rw의 분모).
   * 설정자: cycle()에서 memory_pending_rw > 0이면 banks_access_rw_total++.
   * 읽는 자: print()에서 BLP_Col 및 write_to_read_ratio_blp_rw_average 분모로 사용.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long banks_time_ready;
  /* [한국어] 모든 타이밍 제약이 해제되어 즉시 Column 커맨드 발행이 가능한 뱅크 수의 총합.
   * 설정자: cycle()에서 memory_Pending_ready 변수로 집계 후 banks_time_ready += memory_Pending_ready.
   * 읽는 자: print()에서 BLP_Ready = banks_time_ready / banks_access_ready_total 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long banks_access_ready_total;
  /* [한국어] 즉시 Column 커맨드 발행 가능한 뱅크가 하나라도 있었던 사이클 수 (banks_time_ready 분모).
   * 설정자: cycle()에서 memory_Pending_ready > 0이면 banks_access_ready_total++.
   * 읽는 자: print()에서 BLP_Ready 분모로 사용.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long issued_two;
  /* [한국어] dual_bus_interface 모드에서 같은 사이클에 Row 커맨드와 Column 커맨드를 동시에 발행한 횟수.
   * 설정자: cycle()에서 issued_col_cmd && issued_row_cmd 조건일 때 issued_two++.
   * 읽는 자: print()에서 "Issued_on_Two_Bus_Simul_Util" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long issued_total;
  /* [한국어] Row 또는 Column 커맨드 중 하나라도 발행된 사이클 총 횟수.
   * 설정자: cycle()에서 issued_row_cmd || issued_col_cmd이면 issued_total++.
   * 읽는 자: print()에서 "Either_Row_CoL_Bus_Util" 및 "issued_two_Eff" 분모로 사용.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long issued_total_row;
  /* [한국어] Row 커맨드(ACT 또는 PRE)가 발행된 사이클 총 횟수.
   * 설정자: cycle()에서 issued_row_cmd이면 issued_total_row++.
   * 읽는 자: print()에서 Row Bus utilization 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long issued_total_col;
  /* [한국어] Column 커맨드(RD 또는 WR)가 발행된 사이클 총 횟수.
   * 설정자: cycle()에서 issued_col_cmd이면 issued_total_col++.
   * 읽는 자: print()에서 Column Bus utilization 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  double write_to_read_ratio_blp_rw_average;
  /* [한국어] Row Buffer Hit 상태인 사이클에서 WRITE/(WRITE+READ) 비율의 누적 평균.
   * 설정자: cycle()에서 memory_pending_rw > 0일 때 write_to_read_ratio_blp_rw_average +=
   *         write_blp_rw / (write_blp_rw + read_blp_rw).
   * 읽는 자: print()에서 banks_access_rw_total으로 나눠 평균 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned long long bkgrp_parallsim_rw;
  /* [한국어] Row Buffer Hit 상태인 사이클에서 Hit 요청을 보유한 서로 다른 Bank Group 수의 누적합.
   * 설정자: cycle()에서 bnkgrp_rw_found.count()를 bkgrp_parallsim_rw에 누적.
   * 읽는 자: print()에서 GrpLevelPara = bkgrp_parallsim_rw / banks_access_rw_total 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int bwutil;
  /* [한국어] BW(Bandwidth) utilization 누적 카운터 — 매 RD/WR 커맨드 발행마다 BL/data_command_freq_ratio 증가.
   * 설정자: issue_col_command()에서 RD/WR 발행 시 bwutil += BL/data_command_freq_ratio.
   * 읽는 자: print()/print_stat()에서 bwutil/n_cmd로 BW utilization % 계산; visualizer_print()에서 활용.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int max_mrqs;
  /* [한국어] 시뮬레이션 전체에서 관측된 DRAM 큐 최대 길이 (스케줄러 큐 peak load).
   * 설정자: cycle()에서 현재 큐 길이가 max_mrqs보다 크면 갱신.
   * 읽는 자: print()/print_stat()에서 "mrqq: max=%d" 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int ave_mrqs;
  /* [한국어] 사이클별 DRAM 큐 길이의 누적 합계 (평균 계산용 분자).
   * 설정자: cycle()에서 매 사이클 현재 큐 길이를 ave_mrqs에 누적.
   * 읽는 자: print()/print_stat()에서 ave_mrqs/n_cmd로 평균 큐 길이 계산.
   * 동기화: 단일 시뮬레이션 스레드. */

  class frfcfs_scheduler *m_frfcfs_scheduler;
  /* [한국어] FR-FCFS(First Ready, First Come First Served) DRAM 스케줄러 포인터.
   * 설정자: 생성자에서 scheduler_type == DRAM_FRFCFS이면 new frfcfs_scheduler(...), 아니면 NULL.
   * 읽는 자: full()에서 대기 요청 수 확인; scheduler_frfcfs()에서 스케줄링 위임;
   *          set_dram_power_stats() 등 여러 곳에서 NULL 체크 후 접근.
   * 값 범위: 유효한 frfcfs_scheduler 포인터 또는 NULL (FIFO 모드).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int n_cmd_partial;
  /* [한국어] 마지막 visualizer_print() 이후 경과한 커맨드 사이클 수 (인터벌 카운터).
   * 설정자: cycle()에서 n_cmd_partial++; visualizer_print()에서 출력 후 0으로 리셋.
   * 읽는 자: visualizer_print()에서 gzprintf("dramncmd: %u %u\n", id, n_cmd_partial).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int n_activity_partial;
  /* [한국어] 마지막 visualizer_print() 이후 활성 사이클 수 (인터벌 카운터).
   * 설정자: cycle()에서 k > 0이면 n_activity_partial++; visualizer_print()에서 리셋.
   * 읽는 자: visualizer_print()에서 DRAM efficiency 계산 분모.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int n_nop_partial;
  /* [한국어] 마지막 visualizer_print() 이후 NOP 사이클 수 (인터벌 카운터).
   * 설정자: cycle()에서 !issued이면 n_nop_partial++; visualizer_print()에서 리셋.
   * 읽는 자: visualizer_print()에서 gzprintf("dramnop: %u %u\n", id, n_nop_partial).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int n_act_partial;
  /* [한국어] 마지막 visualizer_print() 이후 ACT 커맨드 발행 횟수 (인터벌 카운터).
   * 설정자: issue_row_command()에서 ACT 발행 시 n_act_partial++; visualizer_print()에서 리셋.
   * 읽는 자: visualizer_print()에서 gzprintf("dramnact: %u %u\n", id, n_act_partial).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int n_pre_partial;
  /* [한국어] 마지막 visualizer_print() 이후 PRE 커맨드 발행 횟수 (인터벌 카운터).
   * 설정자: issue_row_command()에서 PRE 발행 시 n_pre_partial++; visualizer_print()에서 리셋.
   * 읽는 자: visualizer_print()에서 gzprintf("dramnpre: %u %u\n", id, n_pre_partial).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int n_req_partial;
  /* [한국어] 마지막 visualizer_print() 이후 도착한 요청 수 (인터벌 카운터).
   * 설정자: push()에서 n_req_partial++; visualizer_print()에서 리셋.
   * 읽는 자: visualizer_print()에서 gzprintf("dramnreq: %u %u\n", id, n_req_partial).
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int ave_mrqs_partial;
  /* [한국어] 마지막 visualizer_print() 이후 사이클별 큐 길이의 누적 합 (평균 계산용 인터벌 카운터).
   * 설정자: cycle()에서 매 사이클 현재 큐 길이를 누적; visualizer_print()에서 리셋.
   * 읽는 자: visualizer_print()에서 ave_mrqs_partial/n_cmd_partial으로 평균 큐 길이 출력.
   * 동기화: 단일 시뮬레이션 스레드. */

  unsigned int bwutil_partial;
  /* [한국어] 마지막 visualizer_print() 이후 BW utilization 누적 카운터 (인터벌 카운터).
   * 설정자: issue_col_command()에서 RD/WR 발행 시 bwutil_partial += BL/data_command_freq_ratio;
   *         visualizer_print()에서 리셋.
   * 읽는 자: visualizer_print()에서 bwutil_partial/n_cmd_partial(utilization) 및
   *          bwutil_partial/n_activity_partial(efficiency) 계산.
   * 동기화: 단일 시뮬레이션 스레드. */

  class memory_stats_t *m_stats;
  /* [한국어] 메모리 레이턴시/접근 통계 수집 객체 포인터.
   * 설정자: 생성자에서 stats 인자로 초기화.
   * 읽는 자: push()에서 m_stats->memlatstat_dram_access(data) 호출로 레이턴시 통계 기록;
   *          visualizer_print()에서 mem_access_type_stats 배열 접근.
   * 값 범위: 유효한 memory_stats_t 포인터 (NULL 불가).
   * 동기화: 단일 시뮬레이션 스레드. */

  class Stats *mrqq_Dist;  // memory request queue inside DRAM
  /* [한국어] DRAM 입력 큐(mrqq) 길이 분포 히스토그램 통계 객체.
   * 설정자: 생성자에서 StatCreate("mrqq_length", 1, queue_limit() or 64)로 생성.
   * 읽는 자: cycle() 내에서 mrqq 길이 변화 시 통계 업데이트 (현재는 주로 외부 통계 인프라에서 활용).
   * 값 범위: queue_limit()==0이면 최대 64 구간, 아니면 queue_limit() 구간의 히스토그램.
   * 동기화: 단일 시뮬레이션 스레드. */

  friend class frfcfs_scheduler;
  /* [한국어] frfcfs_scheduler가 dram_t의 private 멤버(bk[], bkgrp[], prio 등)에 직접 접근할 수 있도록 허용.
   * FR-FCFS 스케줄러는 뱅크 상태(curr_row, state, mrq)를 직접 읽어 Row Buffer Hit 판단 및 요청 배정을 수행한다. */
};

#endif /*DRAM_H*/
