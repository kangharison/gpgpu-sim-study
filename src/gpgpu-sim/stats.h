/*
 * [한국어 설명] 메모리 파이프라인 통계 관련 열거형 정의 (stats.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 메모리 파이프라인에서 접근 유형, TLB 요청 상태, 스톨 유형을
 * 분류하기 위한 열거형(enum)을 정의한다. 시뮬레이터는 GPU의 메모리 접근을 사이클마다
 * 추적하고 통계를 수집하는데, 이 파일의 enum 값들이 해당 통계 배열의 인덱스로 사용된다.
 * 예를 들어, 어느 메모리 접근 유형(상수/텍스처/공유/전역/지역)이 얼마나 발생했는지,
 * 어떤 원인으로 메모리 파이프라인이 스톨되었는지를 정밀하게 분류·집계하는 데 쓰인다.
 * 이를 통해 GPU 아키텍처의 메모리 병목을 분석하고, 캐시/MSHR/인터커넥트 설계를 평가할 수 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 이 파일은 타이밍 시뮬레이션(gpgpu-sim/) 계층의 메모리 파이프라인 통계 수집 기반을 형성한다.
 * 위치:
 *   SM(shader core) 메모리 파이프라인 (shader.cc)
 *     → 메모리 요청 발행 시 접근 유형(mem_stage_access_type) 및
 *       스톨 원인(mem_stage_stall_type) 분류
 *         → 통계 배열 인덱싱에 사용
 *             → 시뮬레이션 종료 시 리포트 출력 (gpu-sim.cc)
 * 실행 컨텍스트: 호스트 유저스페이스의 사이클-레벨 시뮬레이션 루프에서 참조된다.
 * 이 파일 자체는 타입 정의만 포함하며, 실제 통계 카운터는 shader.cc/gpu-sim.cc에 존재한다.
 *
 * === 타 모듈과의 연결 ===
 * 사용처 (이 파일에 의존하는 모듈):
 *   - shader.cc / shader.h : SM 메모리 파이프라인(mem_stage)에서 mem_stage_access_type,
 *     mem_stage_stall_type을 배열 인덱스로 사용해 통계를 누적한다.
 *   - gpu-sim.cc : 시뮬레이션 종료 시 N_MEM_STAGE_ACCESS_TYPE, N_MEM_STAGE_STALL_TYPE
 *     값을 이용해 통계 배열 전체를 순회하며 리포트를 출력한다.
 *   - gpu-cache.cc : TLB(Translation Lookaside Buffer) 관련 처리에서
 *     tlb_request_status를 사용한다.
 * 데이터 흐름:
 *   메모리 명령 실행 → access_type/stall_type 분류 → 통계 배열[type]++ 누적
 *   → 시뮬레이션 종료 후 분석 보고서 출력
 *
 * === 주요 함수/구조체 요약 ===
 * mem_stage_access_type  : 메모리 접근 유형 7종 + 카운터용 N값 (C_MEM~L_MEM_ST).
 *                          캐시 계층(상수/텍스처/공유/전역/지역)별 분류 인덱스.
 * tlb_request_status     : TLB(주소 변환 캐시) 요청의 3가지 처리 상태 (HIT/READY/PENDING).
 * mem_stage_stall_type   : 메모리 파이프라인 스톨 원인 9종 + 카운터용 N값.
 *                          스톨 없음부터 뱅크충돌/MSHR/ICNT/합병/TLB/포트/라이트백 등 분류.
 */

// Copyright (c) 2009-2011, Tor M. Aamodt,
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

/* [한국어] 헤더 가드 시작 — STATS_INCLUDED가 이미 정의되어 있으면 이 파일 전체를 건너뜀.
 * 여러 번역 단위에서 stats.h를 포함해도 enum이 중복 정의되지 않도록 보호한다. */
#ifndef STATS_INCLUDED
#define STATS_INCLUDED

/*
 * [한국어] mem_stage_access_type — 메모리 파이프라인 단계별 접근 유형 분류 열거형.
 *
 * SM(Streaming Multiprocessor) 내의 메모리 파이프라인은 명령어의 메모리 접근이
 * 어느 캐시/메모리 계층을 대상으로 하는지에 따라 요청을 분류한다.
 * 이 enum 값들은 통계 배열의 인덱스로 사용되어, 각 유형의 접근 횟수를 별도로 집계한다.
 * N_MEM_STAGE_ACCESS_TYPE은 배열 크기 결정과 루프 종료 조건에 사용되는 센티넬 값이다.
 *
 * 설정자: shader.cc의 메모리 파이프라인 단계에서 명령어 분류 시 설정.
 * 읽는 자: gpu-sim.cc의 통계 출력 루프 및 shader.cc의 stall 판정 로직.
 * 값 범위: 0(C_MEM)부터 N_MEM_STAGE_ACCESS_TYPE-1(L_MEM_ST)까지.
 * 동기화: 단일 스레드 시뮬레이션이므로 별도 동기화 불필요.
 */
enum mem_stage_access_type {
  C_MEM,
  /* [한국어] C_MEM (Constant Memory): 상수 메모리 접근.
   * CUDA의 __constant__ 키워드로 선언된 읽기 전용 메모리 영역.
   * SM마다 전용 상수 캐시(constant cache)가 있으며, 모든 스레드가 동일한 주소를
   * 읽는 경우(uniform access) 단일 사이클에 처리된다. 브로드캐스트 최적화 적용.
   * PTX 명령어: ld.const 계열. 접근 실패(miss) 시 L2 캐시로 이어진다. */

  T_MEM,
  /* [한국어] T_MEM (Texture Memory): 텍스처 메모리 접근.
   * CUDA의 텍스처 유닛(texture unit)을 통한 접근. 2D 공간 지역성을 활용하는
   * 전용 텍스처 캐시(L1 texture cache)가 SM에 있다. 보간(interpolation) 하드웨어와
   * 연계되며, 컴퓨팅 워크로드보다 그래픽/이미지 처리에 더 자주 등장한다.
   * PTX 명령어: tex.* 계열. 접근 실패 시 L2 캐시로 이어진다. */

  S_MEM,
  /* [한국어] S_MEM (Shared Memory): 공유 메모리 접근.
   * SM 내 스레드 블록 전체가 공유하는 고속 온칩 메모리 (CUDA의 __shared__ 변수).
   * L1 캐시와 물리적으로 동일한 SRAM을 나누어 쓰는 경우도 있다(설정에 따라 다름).
   * 뱅크 충돌(bank conflict) 발생 시 스톨이 생기며, 이 경우 BK_CONF 스톨로 집계된다.
   * 오프칩 메모리(전역/지역)보다 훨씬 낮은 지연시간(수 사이클)을 가진다.
   * PTX 명령어: ld.shared, st.shared 계열. */

  G_MEM_LD,
  /* [한국어] G_MEM_LD (Global Memory Load): 전역 메모리 로드 접근.
   * CUDA의 일반 전역 메모리(__device__ 변수, malloc 결과)에 대한 읽기 요청.
   * L1 데이터 캐시 → L2 캐시 → DRAM 순서로 계층적으로 탐색된다.
   * 워프 내 스레드들의 접근 주소가 연속적이면 합병(coalescing)되어 단일 트랜잭션으로
   * 처리되고, 산발적이면 여러 트랜잭션으로 분리된다.
   * PTX 명령어: ld.global 계열. 가장 빈번하게 발생하는 접근 유형 중 하나. */

  L_MEM_LD,
  /* [한국어] L_MEM_LD (Local Memory Load): 지역 메모리 로드 접근.
   * CUDA의 지역 메모리(local memory)에 대한 읽기 요청. 지역 메모리는 각 스레드
   * 전용 공간이지만 물리적으로는 전역 메모리(DRAM)에 위치한다 — 레지스터 파일이
   * 가득 찼을 때(레지스터 스필, register spill) 자동으로 사용된다.
   * 스택 프레임, 대형 배열 등 컴파일러가 레지스터에 담기 어려운 데이터를 저장.
   * 성능에 매우 불리하므로, PTX 디스어셈블리에서 ld.local이 보이면 최적화 필요. */

  G_MEM_ST,
  /* [한국어] G_MEM_ST (Global Memory Store): 전역 메모리 스토어 접근.
   * CUDA 전역 메모리에 대한 쓰기 요청. 로드(G_MEM_LD)와 동일한 캐시 경로를 거치지만
   * 쓰기 정책(write policy: write-back 또는 write-through)에 따라 동작이 다를 수 있다.
   * 읽기보다 쓰기는 캐시 활용도가 낮은 경향이 있어, 스토어 집약적 워크로드에서
   * L2/DRAM 대역폭 병목이 두드러진다.
   * PTX 명령어: st.global 계열. */

  L_MEM_ST,
  /* [한국어] L_MEM_ST (Local Memory Store): 지역 메모리 스토어 접근.
   * CUDA 지역 메모리(레지스터 스필 공간)에 대한 쓰기 요청. L_MEM_LD와 마찬가지로
   * 물리적으로는 전역 DRAM에 쓰므로 성능 비용이 크다.
   * 컴파일러 최적화(레지스터 할당 개선, 루프 언롤링 등)로 L_MEM 접근을 줄이는 것이
   * GPU 커널 튜닝의 중요한 포인트 중 하나이다. */

  N_MEM_STAGE_ACCESS_TYPE
  /* [한국어] N_MEM_STAGE_ACCESS_TYPE: 접근 유형의 총 개수를 나타내는 센티넬 값 (= 7).
   * 설정자: enum 자동 할당 (마지막 유효 값 L_MEM_ST 다음 정수).
   * 읽는 자: shader.cc에서 통계 배열 크기 선언 시 (예: int stats[N_MEM_STAGE_ACCESS_TYPE]),
   *          gpu-sim.cc에서 통계 출력 루프의 상한값으로 사용.
   * 새로운 접근 유형을 추가할 때는 이 값 앞에 삽입해야 배열 크기가 자동으로 증가한다. */
};

/*
 * [한국어] tlb_request_status — TLB(Translation Lookaside Buffer) 요청 처리 상태 열거형.
 *
 * TLB(주소 변환 색인 버퍼)는 가상 주소를 물리 주소로 변환한 결과를 캐싱하는 구조이다.
 * GPU에서도 통합 가상 주소 지원(CUDA Unified Virtual Addressing) 시 TLB가 필요하며,
 * GPGPU-Sim은 TLB 요청의 처리 상태를 이 enum으로 추적한다.
 *
 * 설정자: gpu-cache.cc의 TLB 조회 로직에서 요청 처리 결과로 설정.
 * 읽는 자: shader.cc의 메모리 파이프라인에서 TLB 결과에 따라 스톨 여부 결정.
 * 값 범위: TLB_HIT(0), TLB_READY(1), TLB_PENDING(2).
 * 동기화: 단일 스레드 시뮬레이션, 별도 동기화 불필요.
 */
enum tlb_request_status {
  TLB_HIT = 0,
  /* [한국어] TLB_HIT (TLB 히트): 가상→물리 주소 변환 정보가 TLB에 이미 존재하는 상태.
   * 값 = 0 (명시적 초기화 — 숫자 비교 코드와의 호환성 유지를 위해 명시).
   * 이 상태에서는 주소 변환이 즉시(추가 지연 없이) 완료되며 TLB_STALL이 발생하지 않는다.
   * 대부분의 정상적인 메모리 접근에서 이 상태가 반환되어야 성능이 좋다. */

  TLB_READY,
  /* [한국어] TLB_READY (변환 완료, 사용 가능): TLB 미스 후 변환이 완료되어
   * 결과를 사용할 수 있는 상태. TLB_PENDING 상태에서 페이지 워크(page walk) 또는
   * 소프트웨어 TLB 처리가 완료되면 READY로 전이된다.
   * 이 상태에서는 대기 중이던 메모리 접근 요청이 재개될 수 있다. */

  TLB_PENDING
  /* [한국어] TLB_PENDING (변환 처리 중, 대기 상태): TLB 미스가 발생하여 가상→물리
   * 주소 변환을 위한 페이지 테이블 탐색(page walk)이 진행 중인 상태.
   * 이 상태인 동안 해당 메모리 요청은 진행할 수 없으며, 메모리 파이프라인은
   * TLB_STALL(stats.h의 mem_stage_stall_type) 상태로 표시되어 스톨된다.
   * 변환 완료 후 TLB_READY로 전이하면 파이프라인이 재개된다. */
};

/*
 * [한국어] mem_stage_stall_type — 메모리 파이프라인 스톨 원인 분류 열거형.
 *
 * SM의 메모리 파이프라인은 자원 부족이나 충돌로 인해 다양한 원인으로 스톨될 수 있다.
 * 이 enum은 각 사이클에서 발생한 스톨의 원인을 분류하여 통계 배열에 누적하는 데 쓰인다.
 * 어느 원인으로 얼마나 스톨이 발생했는지 분석함으로써 메모리 서브시스템 설계에서
 * 병목 지점을 식별하고 마이크로아키텍처 파라미터(MSHR 크기, 뱅크 수, 버퍼 크기 등)를
 * 튜닝하는 데 핵심적인 정보를 제공한다.
 *
 * 설정자: shader.cc의 mem_stage() 함수에서 자원 가용성 확인 후 스톨 원인 설정.
 * 읽는 자: gpu-sim.cc의 통계 출력 루프, 성능 분석 도구.
 * 값 범위: NO_RC_FAIL(0)부터 N_MEM_STAGE_STALL_TYPE-1(WB_CACHE_RSRV_FAIL)까지.
 * 동기화: 단일 스레드 시뮬레이션, 별도 동기화 불필요.
 */
enum mem_stage_stall_type {
  NO_RC_FAIL = 0,
  /* [한국어] NO_RC_FAIL (No Resource Conflict Failure, 스톨 없음): 정상 진행 상태.
   * 값 = 0 (명시적 초기화). 메모리 파이프라인이 스톨 없이 정상적으로 요청을 처리.
   * 통계 배열에서 이 값에 해당하는 카운터가 가장 높아야 메모리 파이프라인 효율이 좋다. */

  BK_CONF,
  /* [한국어] BK_CONF (Bank Conflict, 뱅크 충돌): 공유 메모리(S_MEM) 또는 캐시 뱅크 충돌.
   * 한 사이클에 여러 스레드가 동일한 메모리 뱅크에 접근하면 직렬화가 필요하다.
   * 공유 메모리는 여러 뱅크(보통 32개)로 구성되어 있으며, 같은 뱅크에 동시 접근하면
   * 충돌이 발생한다. 2-way 충돌은 2 사이클, N-way 충돌은 N 사이클이 걸린다.
   * 해결책: 공유 메모리 배열에 패딩을 추가하거나 접근 패턴을 재구성한다. */

  MSHR_RC_FAIL,
  /* [한국어] MSHR_RC_FAIL (MSHR Resource Conflict Failure): MSHR 자원 부족으로 인한 스톨.
   * MSHR(Miss Status Holding Register, 미스 상태 보유 레지스터)은 캐시 미스 발생 시
   * 해당 요청의 상태를 추적하는 구조이다. MSHR의 엔트리 수가 제한되어 있어, 동시에
   * 처리 가능한 미스의 수에 상한이 있다. MSHR이 가득 차면 새 메모리 요청을 받을 수 없어
   * 파이프라인이 스톨된다. MSHR 크기는 gpgpusim.config에서 설정 가능하다. */

  ICNT_RC_FAIL,
  /* [한국어] ICNT_RC_FAIL (Interconnect Resource Conflict Failure): 인터커넥트 자원 부족.
   * ICNT(Interconnect Network, NoC — intersim2 시뮬레이터)의 입력 버퍼(injection queue)가
   * 가득 찬 경우 SM에서 메모리 파티션으로 새로운 mem_fetch를 전송할 수 없어 스톨된다.
   * NoC 설정(버퍼 크기, 라우팅 알고리즘)이 이 스톨 빈도에 영향을 준다.
   * gpgpusim.config의 -gpgpu_n_mem_per_subpartition 등 파라미터와 연관된다. */

  COAL_STALL,
  /* [한국어] COAL_STALL (Coalescing Stall): 메모리 접근 합병(coalescing) 처리 중 스톨.
   * GPGPU-Sim의 메모리 합병기(coalescing unit)는 한 warp의 32개 스레드 접근 주소를
   * 분석하여 연속적인 주소는 단일 트랜잭션으로 묶는다. 이 합병 처리에 여러 사이클이
   * 걸릴 수 있으며, 그 동안 파이프라인이 스톨된다.
   * 산발적인 접근 패턴(non-coalesced access)일수록 이 스톨이 많이 발생한다. */

  TLB_STALL,
  /* [한국어] TLB_STALL (TLB Stall): TLB 미스로 인한 주소 변환 대기 스톨.
   * 메모리 접근 주소의 가상→물리 변환 정보가 TLB에 없어(tlb_request_status::TLB_PENDING)
   * 페이지 테이블 탐색이 완료될 때까지 파이프라인이 스톨되는 상태.
   * CUDA 통합 가상 주소(UVA) 또는 GPGPU-Sim의 가상 메모리 시뮬레이션이 활성화된
   * 경우에만 발생한다. TLB 크기나 적중률 설정이 이 스톨 빈도를 결정한다. */

  DATA_PORT_STALL,
  /* [한국어] DATA_PORT_STALL (Data Port Stall): 데이터 포트 충돌로 인한 스톨.
   * SM의 L1 데이터 캐시나 공유 메모리의 데이터 반환 포트(data return port)가 이미
   * 다른 요청을 처리 중이어서 새 요청의 데이터를 전달할 수 없는 상태.
   * 포트 수 제한이 있는 구조에서 여러 warp가 동시에 데이터 반환을 요구할 때 발생한다.
   * 캐시 포트 수는 gpgpusim.config에서 설정 가능하다. */

  WB_ICNT_RC_FAIL,
  /* [한국어] WB_ICNT_RC_FAIL (Write-Back Interconnect Resource Conflict Failure):
   * 라이트백(write-back) 처리 중 인터커넥트 자원 부족으로 인한 스톨.
   * 캐시 교체(eviction) 시 더티 라인(dirty line, 수정된 캐시 라인)을 하위 메모리로
   * 라이트백해야 하는데, 이때도 ICNT를 통해 메모리 파티션으로 전송해야 한다.
   * ICNT 버퍼가 가득 차 라이트백 전송이 불가능하면 이 스톨이 발생한다.
   * ICNT_RC_FAIL(새 요청 전송 실패)과 구분하여 라이트백 전용 스톨을 별도 집계한다. */

  WB_CACHE_RSRV_FAIL,
  /* [한국어] WB_CACHE_RSRV_FAIL (Write-Back Cache Reservation Failure):
   * 라이트백을 위한 캐시 예약(reservation) 실패로 인한 스톨.
   * 더티 캐시 라인을 교체하기 위해 캐시 내에 임시 예약 공간이 필요한데, 이 예약이
   * 불가능한 경우 발생한다. 주로 많은 수의 캐시 미스와 동시 라이트백이 겹칠 때
   * 캐시 내 MSHRs 및 WRITE BUFFER가 모두 점유된 상태에서 나타난다.
   * WB_ICNT_RC_FAIL과 함께 라이트백 경로의 성능 분석에 활용된다. */

  N_MEM_STAGE_STALL_TYPE
  /* [한국어] N_MEM_STAGE_STALL_TYPE: 스톨 유형의 총 개수를 나타내는 센티넬 값 (= 9).
   * 설정자: enum 자동 할당.
   * 읽는 자: shader.cc에서 통계 배열 크기 선언 (예: int stall_stats[N_MEM_STAGE_STALL_TYPE]),
   *          gpu-sim.cc의 통계 출력 루프 상한값.
   * 새로운 스톨 유형 추가 시 이 값 앞에 삽입해야 자동으로 배열 크기가 증가한다. */
};

/* [한국어] 헤더 가드 종료 */
#endif
