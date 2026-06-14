/*****************************************************************************
 *                                McPAT/CACTI
 *                      SOFTWARE LICENSE AGREEMENT
 *            Copyright 2012 Hewlett-Packard Development Company, L.P.
 *                          All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.”
 *
 ***************************************************************************/



/*
 * [한국어 설명] NUCA(Non-Uniform Cache Access) 캐시 시뮬레이션 구현 (nuca.cc)
 *
 * === 파일의 역할 ===
 * NUCA 캐시의 전체 설계 공간 탐색을 구현한다. 여러 뱅크 수(2~64)에 대해 최적 단일 뱅크
 * 조직(UCA solve())을 먼저 구하고, 이를 다양한 NoC 라우터(64/128/256 flit)와
 * 와이어 타입(Global ~ Low_swing)으로 mesh 형태 NUCA를 구성해 평균 접근 지연·전력·면적을 계산한다.
 * 마지막으로 contention.dat에서 로드한 경쟁 지연을 더해 최적 NUCA 구성을 선택한다.
 * AccelWattch의 GPU 캐시 에너지 모델에서 캐시가 NUCA로 설정된 경우 이 코드가 실행된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * cacti_interface() → Nuca::sim_nuca() → [이 파일]
 * sim_nuca() 내부에서 solve()를 반복 호출해 각 뱅크 크기의 UCA를 먼저 최적화한 뒤,
 * 그 위에 NoC 네트워크(라우터+배선)를 얹어 NUCA 전체 조직을 평가한다.
 * 호스트 유저스페이스, 단일 스레드에서 실행된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존: Ucache.h(solve, min_values_t), Wire(hop당 배선 지연/전력),
 *       MCPAT_Router(라우터 전력/지연), cacti_interface(g_ip 전역 입력)
 * 소비: cacti_interface()가 sim_nuca() 결과로 최종 NUCA 구성 g_ip->nuca_cache_sz를 업데이트
 * 공유: cont_stats(경쟁 통계 전역 배열), g_ip(InputParameter)
 *
 * === 주요 함수/구조체 요약 ===
 * Nuca::init_cont()         : contention.dat 파싱 → cont_stats 초기화
 * Nuca::sim_nuca()          : 전체 NUCA 탐색 최상위 루프 (뱅크 수 × 라우터 × 와이어)
 * Nuca::calc_cycles()       : 지연[s] → 사이클 수 변환 (클럭 스큐·래치 지연 보정)
 * Nuca::find_optimal_nuca() : 가중치 비용으로 최적 NUCA 선택
 * Nuca::calculate_nuca_area(): 행×열 그리드 총 면적 계산
 */

#include "nuca.h"
#include "Ucache.h"
#include <assert.h>

unsigned int MIN_BANKSIZE=65536; // [한국어] 최소 뱅크 크기 [바이트] — 연관도에 따라 sim_nuca()에서 동적으로 조정됨

#define FIXED_OVERHEAD 55e-12 /* clock skew and jitter in s. Ref: Hrishikesh et al ISCA 01 */
/* [한국어] 고정 오버헤드 55ps — 클럭 스큐와 지터 합산값 (ISCA 2001 Hrishikesh et al. 기준) */

#define LATCH_DELAY 28e-12 /* latch delay in s (later should use FO4 TODO) */
/* [한국어] 래치 지연 28ps — 파이프라인 레지스터 삽입 비용. 향후 FO4 기반으로 교체 예정(TODO) */

#define CONTR_2_BANK_LAT 0
/* [한국어] 컨트롤러→첫 번째 뱅크 추가 지연 (현재 0으로 설정 — 무시) */

int cont_stats[2 /*l2 or l3*/][5/* cores */][ROUTER_TYPES][7 /*banks*/][8 /* cycle time */];
/* [한국어] NUCA 네트워크 경쟁 지연 룩업 테이블 (5차원 배열).
 * 차원[0]: L2 캐시(1) 또는 L3 캐시(0)
 * 차원[1]: 코어 수 인덱스 (2=4코어, 3=8코어, 4=16코어)
 * 차원[2]: 라우터 타입 인덱스 (0=64비트, 1=128비트, 2=256비트 flit)
 * 차원[3]: 뱅크 수 인덱스 (0~6, 즉 2^1 ~ 2^7개 뱅크)
 * 차원[4]: 뱅크 사이클 타임 인덱스 (num_cyc/2-1, 최대 8가지)
 * 설정자: init_cont()가 contention.dat 파일에서 파싱해 채움
 * 읽는 자: sim_nuca()에서 최적 NUCA 접근 지연에 경쟁 지연 추가 시 참조
 * 동기화: 초기화 후 읽기 전용 */

/*
 * [한국어]
 * Nuca::Nuca - 기본 생성자: 주변 회로 글로벌 디바이스 타입으로 초기화
 *
 * @return: Nuca 객체 (deviceType = &g_tp.peri_global)
 *
 * 경쟁 통계 테이블(cont_stats)을 파일에서 로드하는 init_cont()를 호출한다.
 * 호출 체인: cacti_interface() → Nuca() → init_cont()
 */
Nuca::Nuca():deviceType(&(g_tp.peri_global))
{
  init_cont(); // [한국어] contention.dat 파싱하여 5차원 경쟁 지연 테이블 초기화
}

/*
 * [한국어]
 * Nuca::Nuca - 커스텀 디바이스 타입 지정 생성자
 *
 * @dt: 사용할 CMOS 디바이스 파라미터 포인터 (Vdd, 이동도 등)
 *
 * 지정된 dt로 deviceType을 초기화하고 경쟁 통계를 로드한다.
 * 호출 체인: cacti_interface() → Nuca(dt) → init_cont()
 */
Nuca::Nuca(
      TechnologyParameter::DeviceType *dt
      ):deviceType(dt)
{
  init_cont(); // [한국어] 커스텀 디바이스 타입으로 초기화한 뒤 경쟁 통계 로드
}

/*
 * [한국어]
 * Nuca::init_cont - contention.dat 파일을 파싱해 cont_stats 5차원 배열 초기화
 *
 * 경쟁 통계는 사이클 정확 시뮬레이션으로 미리 계산된 룩업 테이블이다
 * (CACTI 6.0 Tech Report 참조). 파일이 없으면 즉시 종료한다.
 *
 * 호출 체인: Nuca() → [이 함수]
 */
void
Nuca::init_cont()
{
  FILE *cont;           // [한국어] contention.dat 파일 핸들
  char line[5000];      // [한국어] 파일에서 읽은 한 줄 버퍼 (최대 5000자)
  char jk[5000];        // [한국어] 줄 앞부분 레이블 문자열 (파싱 후 버림)

  cont = fopen("contention.dat", "r"); // [한국어] 현재 디렉토리에서 경쟁 통계 파일 열기
  if (!cont) {
    cout << "contention.dat file is missing!\n"; // [한국어] 파일 없으면 오류 출력
    exit(0); // [한국어] 파일이 없으면 시뮬레이션 진행 불가 — 즉시 종료
  }

  for(int i=0; i<2; i++) {         // [한국어] i: 캐시 레벨 (0=L3, 1=L2)
    for(int j=2; j<5; j++) {       // [한국어] j: 코어 수 인덱스 (2=4코어, 3=8코어, 4=16코어)
      for(int k=0; k<ROUTER_TYPES; k++) { // [한국어] k: 라우터 타입 (0=64비트, 1=128비트, 2=256비트)
        for(int l=0;l<7; l++) {    // [한국어] l: 뱅크 수 인덱스 (2^1 ~ 2^7)
          int *temp = cont_stats[i/*l2 or l3*/][j/*core*/][k/*64 or 128 or 256 link bw*/][l /* no banks*/];
          // [한국어] 5차원 배열의 [i][j][k][l] 슬라이스(8개 사이클 값)를 temp 포인터로 접근
          assert(fscanf(cont, "%[^\n]\n", line) != EOF); // [한국어] 한 줄 읽기 (EOF면 파일 불완전 — 단언 실패)
          sscanf(line, "%[^:]: %d %d %d %d %d %d %d %d",jk, &temp[0], &temp[1], &temp[2], &temp[3],
              &temp[4], &temp[5], &temp[6], &temp[7]);
          // [한국어] "레이블: val0 val1 ... val7" 형식 파싱 — 8개 사이클 타임별 경쟁 지연 저장
        }
      }
    }
  }
  fclose(cont); // [한국어] 파일 닫기 — 이후 cont_stats는 읽기 전용으로 사용
}

/*
 * [한국어]
 * Nuca::print_cont_stats - cont_stats 5차원 배열 전체를 콘솔에 출력 (디버그용)
 *
 * 주의: 내부 루프에 버그 있음 — `for(int m=0;l<7; l++)` 에서 m이 아닌 l을 증가시킴.
 * 이 함수는 실제로 호출되지 않으므로 버그가 실행에 영향을 주지 않음.
 * 호출 체인: (미호출)
 */
  void
Nuca::print_cont_stats()
{
  for(int i=0; i<2; i++) {          // [한국어] i: 캐시 레벨
    for(int j=2; j<5; j++) {        // [한국어] j: 코어 수
      for(int k=0; k<ROUTER_TYPES; k++) { // [한국어] k: 라우터 타입
        for(int l=0;l<7; l++) {     // [한국어] l: 뱅크 수 인덱스
          for(int m=0;l<7; l++) {   // [한국어] 버그: m이 증가해야 하는데 l이 증가됨 (실질적으로 무한 루프 위험)
            cout << cont_stats[i][j][k][l][m] << " ";
          }
          cout << endl;
        }
      }
    }
  }
  cout << endl;
}

/*
 * [한국어]
 * Nuca::~Nuca - 소멸자: wire_vertical/wire_horizontal 배열 해제
 *
 * wt_min..wt_max 범위에서 생성된 Wire 인스턴스들을 delete한다.
 * 범위 밖 인덱스는 초기화되지 않았으므로 접근하지 않는다.
 * 호출 체인: sim_nuca() 완료 후 Nuca 객체 스택 해제 시 자동 호출
 */
Nuca::~Nuca(){
  for (int i = wt_min; i <= wt_max; i++) { // [한국어] 탐색했던 와이어 타입 범위만 해제
    delete wire_vertical[i];   // [한국어] 수직 와이어 해제
    delete wire_horizontal[i]; // [한국어] 수평 와이어 해제
  }
}

/*
 * [한국어]
 * Nuca::calc_cycles - 지연 시간을 클럭 사이클 수로 변환
 *
 * @lat      : 변환할 지연 시간 [s]
 * @oper_freq: 동작 주파수 [GHz] (1/(cycle_time_ps × 0.001)로 전달됨)
 * @return   : 지연을 커버하는 최소 사이클 수 (ceil 적용)
 *
 * 단순 lat/T 계산이 아니라, 클럭 스큐(55ps)와 래치 지연(28ps)을
 * 유효 사이클 타임에서 차감한 뒤 나눈다. 즉 실제 사이클당 전달 가능한
 * 유효 시간이 줄어들어 사이클 수가 보수적으로 증가한다.
 * (TODO: 래치 지연을 FO4 기반으로 교체 필요)
 *
 * 호출 체인: sim_nuca() → [이 함수] (hor_hop_lat, ver_hop_lat, num_cyc 계산 시)
 */
/* converts latency (in s) to cycles depending upon the FREQUENCY (in GHz) */
  int
Nuca::calc_cycles(double lat, double oper_freq)
{
  //TODO: convert latch delay to FO4 */
  double cycle_time = (1.0/(oper_freq*1e9)); // [한국어] 주파수 역수로 이상적 사이클 타임 계산 [s]
  cycle_time -= LATCH_DELAY;     // [한국어] 래치 지연 28ps 차감 — 파이프라인 레지스터 삽입 비용
  cycle_time -= FIXED_OVERHEAD;  // [한국어] 클럭 스큐+지터 55ps 차감 — 실제 유효 전달 시간 단축

  return (int)ceil(lat/cycle_time); // [한국어] 지연/유효_사이클 올림 → 최소 필요 사이클 수
}


nuca_org_t::~nuca_org_t() {
  // if(h_wire) delete h_wire;
  // if(v_wire) delete v_wire;
  // if(router) delete router;
}

/*
 * [한국어]
 * Nuca::sim_nuca - NUCA 설계 공간 전수 탐색 및 최적 구성 선택
 *
 * @return: (없음) — 전역 g_ip->nuca_cache_sz 업데이트, print_nuca()로 결과 출력
 *
 * Version 6.0 전수 탐색 알고리즘:
 * 1) 뱅크 수 반복(it 루프): 전체 캐시 크기 / MIN_BANKSIZE ~ 전체 캐시 크기까지
 *    각 bank_count에 대해 solve()로 단일 뱅크 최적 UCA 조직을 먼저 구함
 * 2) 배선 타입 반복(wr 루프): Global~Low_swing까지 각 타입의 수직/수평 와이어 모델 생성
 * 3) 라우터 반복(ro 루프): 64/128/256비트 flit 라우터 3종
 * 4) 그리드 탐색(r×c 루프): 행 수×열 수 조합별 평균 홉 수·지연·전력 계산
 *    curr_acclat = 2×평균지연 + 2×라우터지연×평균홉 + 뱅크접근지연 (왕복 모델)
 * 5) contention 보정: cont_stats에서 경쟁 지연 추가
 * 6) nuca_list에 결과 push → 전체 탐색 완료 후 find_optimal_nuca()로 최적 선택
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 스레드 (무거운 연산).
 * 호출 체인: cacti_interface() → [이 함수] → solve() / calc_cycles() / find_optimal_nuca()
 *
 * Version - 6.0
 *
 * Perform exhaustive search across different bank organizatons,
 * router configurations, grid organizations, and wire models and
 * find an optimal NUCA organization
 * For different bank count values
 * 1. Optimal bank organization is calculated
 * 2. For each bank organization, find different NUCA organizations
 *    using various router configurations, grid organizations,
 *    and wire models.
 * 3. NUCA model with the least cost is picked for
 *    this particular bank count
 * Finally include contention statistics and find the optimal
 *    NUCA configuration
 */
  void
Nuca::sim_nuca()
{
  /* temp variables */
  int it, ro, wr; // [한국어] it: 반복 인덱스(뱅크 크기), ro: 라우터 인덱스, wr: 배선 타입 인덱스
  int num_cyc;    // [한국어] 뱅크 접근 지연을 사이클로 변환한 값
  unsigned int i, j; // [한국어] 그리드 행/열 반복 인덱스
  unsigned int r, c; // [한국어] 현재 탐색 중인 행 수(r)와 열 수(c)
  int l2_c;      // [한국어] L2 캐시 여부 (1=L2, 0=L3) — cont_stats 인덱스
  int bank_count = 0; // [한국어] 현재 탐색 중인 뱅크 수 = 전체캐시크기/현재뱅크크기
  uca_org_t ures;    // [한국어] 단일 뱅크 최적 UCA 결과 — solve()가 채움
  nuca_org_t *opt_n; // [한국어] 최적 NUCA 구성 포인터 — find_optimal_nuca()가 반환
  mem_array tag, data; // [한국어] ures 태그/데이터 배열 임시 저장용
  list<nuca_org_t *> nuca_list; // [한국어] 탐색된 모든 NUCA 구성을 저장하는 리스트
  MCPAT_Router *router_s[ROUTER_TYPES]; // [한국어] 3종 라우터(64/128/256비트 flit) 배열
  router_s[0] = new MCPAT_Router(64.0, 8, 4, &(g_tp.peri_global));
  // [한국어] 64비트 flit, 버퍼 8 flit, VC 4개 라우터
  router_s[0]->print_router(); // [한국어] 64비트 라우터 파라미터 출력
  router_s[1] = new MCPAT_Router(128.0, 8, 4, &(g_tp.peri_global));
  // [한국어] 128비트 flit 라우터
  router_s[1]->print_router(); // [한국어] 128비트 라우터 파라미터 출력
  router_s[2] = new MCPAT_Router(256.0, 8, 4, &(g_tp.peri_global));
  // [한국어] 256비트 flit 라우터
  router_s[2]->print_router(); // [한국어] 256비트 라우터 파라미터 출력

  int core_in; // to store no. of cores
  // [한국어] core_in: 코어 수 범주 인덱스 — cont_stats 3번째 차원 인덱스

  /* to search diff grid organizations */
  double curr_hop, totno_hops, totno_hhops, totno_vhops, tot_lat,
         curr_acclat;
  // [한국어] 그리드 탐색 임시 변수: 홉 수 합계, 총 지연, 현재 접근 지연
  double avg_lat, avg_hop, avg_hhop, avg_vhop, avg_dyn_power,
         avg_leakage_power;
  // [한국어] 뱅크당 평균: 지연, 총 홉, 수평/수직 홉, 동적/누설 전력

  double opt_acclat = INF; // [한국어] 현재 최적 접근 지연 초기값 = INF
  int opt_rows = 0;        // [한국어] 최적 그리드 행 수
  int opt_columns = 0;     // [한국어] 최적 그리드 열 수
  double opt_avg_hop = 0;  // [한국어] 최적 구성의 평균 홉 수
  double opt_dyn_power = 0, opt_leakage_power = 0; // [한국어] 최적 구성 전력
  min_values_t minval; // [한국어] 전체 탐색 최솟값 추적기 (비용 정규화 분모)

  int bank_start = 0; // [한국어] 뱅크 수 반복 시작 인덱스 (0 또는 특정 뱅크 수 강제 시 변경)

  int flit_width = 0; // [한국어] 현재 라우터의 flit 크기 [비트]

  /* vertical and horizontal hop latency values */
  int ver_hop_lat, hor_hop_lat; /* in cycles */
  // [한국어] 수직/수평 홉 1개당 사이클 수 (calc_cycles로 계산)


  /* no. of different bank sizes to consider */
  int iterations; // [한국어] 탐색할 뱅크 크기 반복 횟수 = log2(전체캐시크기/MIN_BANKSIZE)


  g_ip->nuca_cache_sz = g_ip->cache_sz; // [한국어] NUCA 전체 캐시 크기 저장 (탐색 중 cache_sz 변경 전)
  nuca_list.push_back(new nuca_org_t()); // [한국어] 첫 번째 빈 nuca_org_t를 리스트에 추가

  if (g_ip->cache_level == 0) l2_c = 1; // [한국어] L2 캐시인 경우 l2_c=1
  else l2_c = 0;                         // [한국어] L3 이상인 경우 l2_c=0

  if (g_ip->cores <= 4) core_in = 2;       // [한국어] 코어 4개 이하: cont_stats 인덱스 2
  else if (g_ip->cores <= 8) core_in = 3;  // [한국어] 코어 8개 이하: 인덱스 3
  else if (g_ip->cores <= 16) core_in = 4; // [한국어] 코어 16개 이하: 인덱스 4
  else {cout << "Number of cores should be <= 16!\n"; exit(0);} // [한국어] 16코어 초과 미지원


  // set the lower bound to an appropriate value. this depends on cache associativity
  if (g_ip->assoc > 2) {
    // [한국어] 연관도(assoc)가 2보다 크면 MIN_BANKSIZE를 연관도에 맞게 조정
    i = 2;
    while (i != g_ip->assoc) {
      MIN_BANKSIZE *= 2; // [한국어] 연관도만큼 MIN_BANKSIZE를 2배씩 확장
      i *= 2;
    }
  }

  iterations = (int)logtwo((int)g_ip->cache_sz/MIN_BANKSIZE);
  // [한국어] 탐색 반복 횟수 = log2(캐시크기/최소뱅크크기) — 뱅크 수를 1에서 최대로 증가시키는 단계 수

  if (g_ip->force_wiretype)
  // [한국어] 배선 타입 강제 지정 모드
  {
    if (g_ip->wt == Low_swing) {
      wt_min = Low_swing; // [한국어] Low_swing 강제: 최솟값·최댓값 모두 Low_swing
      wt_max = Low_swing;
    }
    else {
      wt_min = Global;        // [한국어] 기타 타입 강제: Global~Low_swing 이전 탐색
      wt_max = Low_swing-1;
    }
  }
  else {
    wt_min = Global;     // [한국어] 모든 배선 타입 탐색: Global~Low_swing
    wt_max = Low_swing;
  }
  if (g_ip->nuca_bank_count != 0) { // simulate just one bank
    // [한국어] 특정 뱅크 수가 지정된 경우 해당 뱅크 수만 탐색
    if (g_ip->nuca_bank_count != 2 && g_ip->nuca_bank_count != 4 &&
        g_ip->nuca_bank_count != 8 && g_ip->nuca_bank_count != 16 &&
        g_ip->nuca_bank_count != 32 && g_ip->nuca_bank_count != 64) {
      fprintf(stderr,"Incorrect bank count value! Please fix the value in cache.cfg\n");
      // [한국어] 지원되지 않는 뱅크 수 오류 출력
    }
    bank_start = (int)logtwo((double)g_ip->nuca_bank_count);
    // [한국어] 시작 인덱스 = log2(지정 뱅크 수)
    iterations = bank_start+1; // [한국어] 한 번만 반복
    g_ip->cache_sz = g_ip->cache_sz/g_ip->nuca_bank_count;
    // [한국어] 단일 뱅크 크기 = 전체 캐시 크기 / 뱅크 수
  }
  cout << "Simulating various NUCA configurations\n";
  for (it=bank_start; it<iterations; it++) { /* different bank count values */
    // [한국어] 뱅크 크기 반복: it 증가 → 뱅크 크기 절반, 뱅크 수 두 배
    ures.tag_array2 = &tag;   // [한국어] ures에 임시 태그/데이터 배열 포인터 연결
    ures.data_array2 = &data;
    /*
     * find the optimal bank organization
     */
    solve(&ures); // [한국어] 현재 cache_sz에 대한 최적 UCA(단일 뱅크) 탐색
//    output_UCA(&ures);
    bank_count = g_ip->nuca_cache_sz/g_ip->cache_sz;
    // [한국어] 현재 반복의 뱅크 수 = 전체 NUCA 크기 / 현재 뱅크 크기
    cout << "====" <<  g_ip->cache_sz << "\n";

    for (wr=wt_min; wr<=wt_max; wr++) {
      // [한국어] 배선 타입 반복: Global / Global_5% / ... / Low_swing

      for (ro=0; ro<ROUTER_TYPES; ro++)
      // [한국어] 라우터 반복: 64비트(ro=0), 128비트(ro=1), 256비트(ro=2) flit
      {
        flit_width = (int) router_s[ro]->flit_size; //initialize router
        // [한국어] 현재 라우터의 flit 폭 [비트] — 배선 전력 계산 시 곱셈에 사용
        nuca_list.back()->nuca_pda.cycle_time = router_s[ro]->cycle_time;
        // [한국어] NUCA 사이클 타임 = 라우터 사이클 타임 (라우터가 병목)

        /* calculate router and wire parameters */

        double vlength = ures.cache_ht; /* length of the wire (u)*/
        // [한국어] 수직 와이어 길이 = 단일 뱅크 높이 [um]
        double hlength = ures.cache_len; // u
        // [한국어] 수평 와이어 길이 = 단일 뱅크 너비 [um]

        /* find delay, area, and power for wires */
        wire_vertical[wr] = new Wire((enum Wire_type) wr, vlength);
        // [한국어] 수직 방향 와이어 모델 생성 — 배선 타입 wr, 길이 vlength
        wire_horizontal[wr] = new Wire((enum Wire_type) wr, hlength);
        // [한국어] 수평 방향 와이어 모델 생성 — 배선 타입 wr, 길이 hlength


        hor_hop_lat = calc_cycles(wire_horizontal[wr]->delay,
            1/(nuca_list.back()->nuca_pda.cycle_time*.001));
        // [한국어] 수평 1홉 지연 [사이클] = 수평 와이어 지연 / NUCA 사이클 타임
        ver_hop_lat = calc_cycles(wire_vertical[wr]->delay,
            1/(nuca_list.back()->nuca_pda.cycle_time*.001));
        // [한국어] 수직 1홉 지연 [사이클] = 수직 와이어 지연 / NUCA 사이클 타임

        /*
         * assume a grid like topology and explore for optimal network
         * configuration using different row and column count values.
         */
        for (c=1; c<=(unsigned int)bank_count; c++) {
          while (bank_count%c != 0) c++; // [한국어] 열 수(c)가 뱅크 수의 약수인 경우만 고려
          r = bank_count/c; // [한국어] 행 수 = 전체 뱅크 수 / 열 수

          /*
           * to find the avg access latency of a NUCA cache, uncontended
           * access time to each bank from the
           * cache controller is calculated.
           * avg latency =
           * sum of the access latencies to individual banks)/bank
           * count value.
           */
          totno_hops = totno_hhops = totno_vhops = tot_lat = 0;
          // [한국어] 누산 변수 초기화 — 이 그리드 구성의 전체 합계를 구하기 위해

          for (i=0; i<r; i++) {
            // [한국어] 행 i 반복 (0 = 컨트롤러 위치 행에서 i행 떨어진 뱅크)
            for (j=0; j<c; j++) {
              // [한국어] 열 j 반복 (j=0에서 c-1까지 수평 홉 수)
              /*
               * vertical hops including the
               * first hop from the cache controller
               */
              curr_hop = i + 1; // [한국어] 수직 홉 수 = 행 거리 + 컨트롤러→첫뱅크 1홉
              curr_hop += j; /* horizontal hops */ // [한국어] 수평 홉 수 추가
              totno_hhops += j;       // [한국어] 수평 홉 수 누산
              totno_vhops += (i+1);   // [한국어] 수직 홉 수 누산
              curr_acclat = (i * ver_hop_lat + CONTR_2_BANK_LAT +
                  j * hor_hop_lat);
              // [한국어] 이 뱅크 접근 지연 = 수직홉×수직홉지연 + 컨트롤러→뱅크기본지연 + 수평홉×수평홉지연

              tot_lat += curr_acclat;  // [한국어] 전체 지연 누산
              totno_hops += curr_hop;  // [한국어] 전체 홉 수 누산
            }
          }
          avg_lat = tot_lat/bank_count; // [한국어] 뱅크당 평균 접근 지연 [사이클]
          avg_hop = totno_hops/bank_count;  // [한국어] 뱅크당 평균 총 홉 수
          avg_hhop = totno_hhops/bank_count; // [한국어] 뱅크당 평균 수평 홉 수
          avg_vhop = totno_vhops/bank_count; // [한국어] 뱅크당 평균 수직 홉 수

          /* net access latency */
          curr_acclat = 2*avg_lat + 2*(router_s[ro]->delay*avg_hop) +
            calc_cycles(ures.access_time,
                1/(nuca_list.back()->nuca_pda.cycle_time*.001));
          // [한국어] 총 접근 지연 = 2×평균지연(요청+응답) + 2×라우터지연×평균홉 + 뱅크접근지연[사이클]
          // 2× 곱: 요청(캐시컨트롤러→뱅크)와 응답(뱅크→캐시컨트롤러) 왕복 모델

          /* avg access lat of nuca */
          avg_dyn_power =
            avg_hop *
            (router_s[ro]->power.readOp.dynamic) + avg_hhop *
            (wire_horizontal[wr]->power.readOp.dynamic) *
            (g_ip->block_sz*8 + 64) + avg_vhop *
            (wire_vertical[wr]->power.readOp.dynamic) *
            (g_ip->block_sz*8 + 64) + ures.power.readOp.dynamic;
          // [한국어] 평균 동적 전력 = 라우터전력×평균홉 + 수평와이어전력×블록크기×평균수평홉
          //   + 수직와이어전력×블록크기×평균수직홉 + 뱅크전력
          // 블록크기×8+64: 캐시 블록 비트 수 + 64비트 태그/제어 신호

          avg_leakage_power =
            bank_count * router_s[ro]->power.readOp.leakage +
            avg_hhop * (wire_horizontal[wr]->power.readOp.leakage*
                wire_horizontal[wr]->delay) * flit_width +
            avg_vhop * (wire_vertical[wr]->power.readOp.leakage *
                wire_horizontal[wr]->delay);
          // [한국어] 평균 누설 전력 = 라우터누설×뱅크수 + 배선누설×배선지연×flit폭×홉수

          if (curr_acclat < opt_acclat) {
            // [한국어] 현재 구성이 지금까지 최적보다 빠르면 최적값 갱신
            opt_acclat = curr_acclat;      // [한국어] 최적 접근 지연 갱신
            opt_avg_hop = avg_hop;         // [한국어] 최적 평균 홉 수 갱신
            opt_rows = r;                  // [한국어] 최적 행 수 갱신
            opt_columns = c;               // [한국어] 최적 열 수 갱신
            opt_dyn_power = avg_dyn_power; // [한국어] 최적 동적 전력 갱신
            opt_leakage_power = avg_leakage_power; // [한국어] 최적 누설 전력 갱신
          }
          totno_hops = 0;   // [한국어] 다음 그리드 구성을 위해 누산 변수 초기화
          tot_lat = 0;
          totno_hhops = 0;
          totno_vhops = 0;
        }
        nuca_list.back()->wire_pda.power.readOp.dynamic =
          opt_avg_hop * flit_width *
          (wire_horizontal[wr]->power.readOp.dynamic +
           wire_vertical[wr]->power.readOp.dynamic);
        // [한국어] 최적 그리드의 와이어 동적 전력 = 평균홉×flit폭×(수평+수직 와이어 전력)
        nuca_list.back()->avg_hops = opt_avg_hop; // [한국어] 최적 평균 홉 수 저장
        /* network delay/power */
        nuca_list.back()->h_wire = wire_horizontal[wr]; // [한국어] 수평 와이어 모델 포인터 저장
        nuca_list.back()->v_wire = wire_vertical[wr];   // [한국어] 수직 와이어 모델 포인터 저장
        nuca_list.back()->router = router_s[ro];         // [한국어] 라우터 모델 포인터 저장
        /* bank delay/power */

        nuca_list.back()->bank_pda.delay = ures.access_time; // [한국어] 뱅크 접근 지연 [s]
        nuca_list.back()->bank_pda.power = ures.power;        // [한국어] 뱅크 전력 복사
        nuca_list.back()->bank_pda.area.h = ures.cache_ht;   // [한국어] 뱅크 높이 [um]
        nuca_list.back()->bank_pda.area.w = ures.cache_len;  // [한국어] 뱅크 너비 [um]
        nuca_list.back()->bank_pda.cycle_time = ures.cycle_time; // [한국어] 뱅크 사이클 타임

        num_cyc = calc_cycles(nuca_list.back()->bank_pda.delay /*s*/,
            1/(nuca_list.back()->nuca_pda.cycle_time*.001/*GHz*/));
        // [한국어] 뱅크 접근 지연을 NUCA 사이클로 변환 (경쟁 통계 인덱싱에 사용)
        if(num_cyc%2 != 0) num_cyc++; // [한국어] 홀수면 올림 — cont_stats가 짝수 사이클 기준
        if (num_cyc > 16) num_cyc = 16; // we have data only up to 16 cycles
        // [한국어] 경쟁 데이터가 최대 16사이클까지만 있으므로 16으로 클리핑

        if (it < 7) {
          // [한국어] 뱅크 수 반복 인덱스가 7 미만: 해당 인덱스의 cont_stats 사용
          nuca_list.back()->nuca_pda.delay = opt_acclat +
            cont_stats[l2_c][core_in][ro][it][num_cyc/2-1];
          // [한국어] NUCA 접근 지연 = 비경쟁 지연 + 경쟁 지연 보정값
          nuca_list.back()->contention =
            cont_stats[l2_c][core_in][ro][it][num_cyc/2-1];
          // [한국어] 경쟁 지연값 별도 저장
        }
        else {
          // [한국어] 7 이상: 데이터가 최대 7까지만 있으므로 인덱스 7로 클리핑
          nuca_list.back()->nuca_pda.delay = opt_acclat +
            cont_stats[l2_c][core_in][ro][7][num_cyc/2-1];
          nuca_list.back()->contention =
            cont_stats[l2_c][core_in][ro][7][num_cyc/2-1];
        }
        nuca_list.back()->nuca_pda.power.readOp.dynamic = opt_dyn_power;
        // [한국어] 최적 그리드 동적 전력 저장
        nuca_list.back()->nuca_pda.power.readOp.leakage = opt_leakage_power;
        // [한국어] 최적 그리드 누설 전력 저장

        /* array organization */
        nuca_list.back()->bank_count = bank_count;   // [한국어] 이 구성의 뱅크 수 저장
        nuca_list.back()->rows = opt_rows;           // [한국어] 최적 행 수 저장
        nuca_list.back()->columns = opt_columns;     // [한국어] 최적 열 수 저장
        calculate_nuca_area (nuca_list.back());       // [한국어] 그리드 총 면적 계산

        minval.update_min_values(nuca_list.back()); // [한국어] 전체 최솟값 갱신
        nuca_list.push_back(new nuca_org_t());      // [한국어] 다음 구성을 위해 새 항목 추가
        opt_acclat = BIGNUM; // [한국어] 다음 라우터/와이어 조합을 위해 최적값 초기화

      }
    }
    g_ip->cache_sz /= 2; // [한국어] 다음 반복: 뱅크 크기 절반 (뱅크 수는 두 배)
  }

  delete(nuca_list.back());
  nuca_list.pop_back();
  opt_n = find_optimal_nuca(&nuca_list, &minval);
  print_nuca(opt_n);
  g_ip->cache_sz = g_ip->nuca_cache_sz/opt_n->bank_count;

  list<nuca_org_t *>::iterator niter;
  for (niter = nuca_list.begin(); niter != nuca_list.end(); ++niter)
  {
    delete *niter;
  }
  nuca_list.clear();

  for(int i=0; i < ROUTER_TYPES; i++)
  {
    delete router_s[i];
  }
  g_ip->display_ip();
  //  g_ip->force_cache_config = true;
  //  g_ip->ndwl = 8;
  //  g_ip->ndbl = 16;
  //  g_ip->nspd = 4;
  //  g_ip->ndcm = 1;
  //  g_ip->ndsam1 = 8;
  //  g_ip->ndsam2 = 32;

}


/*
 * [한국어]
 * Nuca::print_nuca - 최적 NUCA 구성의 상세 통계를 콘솔에 출력
 *
 * @fr: 출력할 NUCA 구성 포인터 (find_optimal_nuca()가 반환한 최적 구성)
 *
 * 뱅크 수, 그리드 조직, 네트워크 주파수, 칩 치수, 라우터 통계,
 * 와이어 타입·지연·에너지를 printf/print_router/print_wire로 출력한다.
 * 호출 체인: sim_nuca() → [이 함수]
 */
  void
Nuca::print_nuca (nuca_org_t *fr)
{
  printf("\n---------- CACTI version 6.5, Non-uniform Cache Access "
      "----------\n\n");
  printf("Optimal number of banks - %d\n", fr->bank_count);
  printf("Grid organization rows x columns - %d x %d\n",
      fr->rows, fr->columns);
  printf("Network frequency - %g GHz\n",
      (1/fr->nuca_pda.cycle_time)*1e3);
  printf("Cache dimension (mm x mm) - %g x %g\n",
      fr->nuca_pda.area.h,
      fr->nuca_pda.area.w);

  fr->router->print_router();

  printf("\n\nWire stats:\n");
  if (fr->h_wire->wt == Global) {
    printf("\tWire type - Full swing global wires with least "
        "possible delay\n");
  }
  else if (fr->h_wire->wt == Global_5) {
    printf("\tWire type - Full swing global wires with "
        "5%% delay penalty\n");
  }
  else if (fr->h_wire->wt == Global_10) {
    printf("\tWire type - Full swing global wires with "
        "10%% delay penalty\n");
  }
  else if (fr->h_wire->wt == Global_20) {
    printf("\tWire type - Full swing global wires with "
        "20%% delay penalty\n");
  }
  else if (fr->h_wire->wt == Global_30) {
    printf("\tWire type - Full swing global wires with "
        "30%% delay penalty\n");
  }
  else if(fr->h_wire->wt == Low_swing) {
    printf("\tWire type - Low swing wires\n");
  }

  printf("\tHorizontal link delay - %g (ns)\n",
      fr->h_wire->delay*1e9);
  printf("\tVertical link delay - %g (ns)\n",
      fr->v_wire->delay*1e9);
  printf("\tDelay/length - %g (ns/mm)\n",
      fr->h_wire->delay*1e9/fr->bank_pda.area.w);
  printf("\tHorizontal link energy -dynamic/access %g (nJ)\n"
      "\t                       -leakage %g (nW)\n\n",
      fr->h_wire->power.readOp.dynamic*1e9,
      fr->h_wire->power.readOp.leakage*1e9);
  printf("\tVertical link energy -dynamic/access %g (nJ)\n"
      "\t                     -leakage %g (nW)\n\n",
      fr->v_wire->power.readOp.dynamic*1e9,
      fr->v_wire->power.readOp.leakage*1e9);
  printf("\n\n");
  fr->v_wire->print_wire();
  printf("\n\nBank stats:\n");
}


/*
 * [한국어]
 * Nuca::find_optimal_nuca - 가중치 비용 함수로 최적 NUCA 구성 선택
 *
 * @n     : 탐색된 모든 nuca_org_t 포인터 리스트
 * @minval: 지연·전력·면적·사이클 최솟값 추적기 (비용 정규화에 사용)
 * @return: 최적 nuca_org_t 포인터 (리스트에서 삭제되지 않으므로 공유 소유권)
 *
 * 세 가지 최적화 모드:
 * g_ip->ed==1: ED(에너지×지연) 최소화
 * g_ip->ed==2: ED²(에너지×지연²) 최소화
 * 그 외:       지연/사이클/동적전력/누설전력/면적 가중합 최소화 + 편차 제약 검사
 *
 * 호출 체인: sim_nuca() → [이 함수]
 */
  nuca_org_t *
Nuca::find_optimal_nuca (list<nuca_org_t *> *n, min_values_t *minval)
{
  double cost = 0;        // [한국어] 현재 구성의 가중합 비용
  double min_cost = BIGNUM; // [한국어] 최소 비용 (초기값: 무한대)
  nuca_org_t *res = NULL;
  float d, a, dp, lp, c;
  int v;
  dp = g_ip->dynamic_power_wt_nuca;
  lp = g_ip->leakage_power_wt_nuca;
  a = g_ip->area_wt_nuca;
  d = g_ip->delay_wt_nuca;
  c = g_ip->cycle_time_wt_nuca;

  list<nuca_org_t *>::iterator niter;


  for (niter = n->begin(); niter != n->end(); niter++) {
    fprintf(stderr, "\n-----------------------------"
        "---------------\n");


    printf("NUCA___stats %d \tbankcount: lat = %g \tdynP = %g \twt = %d\t "
        "bank_dpower = %g \tleak = %g \tcycle = %g\n",
        (*niter)->bank_count,
        (*niter)->nuca_pda.delay,
        (*niter)->nuca_pda.power.readOp.dynamic,
        (*niter)->h_wire->wt,
        (*niter)->bank_pda.power.readOp.dynamic,
        (*niter)->nuca_pda.power.readOp.leakage,
        (*niter)->nuca_pda.cycle_time);


    if (g_ip->ed == 1) {
      cost = ((*niter)->nuca_pda.delay/minval->min_delay)*
        ((*niter)->nuca_pda.power.readOp.dynamic/minval->min_dyn);
      if (min_cost > cost) {
        min_cost = cost;
        res = ((*niter));
      }
    }
    else if (g_ip->ed == 2) {
      cost = ((*niter)->nuca_pda.delay/minval->min_delay)*
        ((*niter)->nuca_pda.delay/minval->min_delay)*
        ((*niter)->nuca_pda.power.readOp.dynamic/minval->min_dyn);
      if (min_cost > cost) {
        min_cost = cost;
        res = ((*niter));
      }
    }
    else {
      /*
       * check whether the current organization
       * meets the input deviation constraints
       */
      v = check_nuca_org((*niter), minval);
      if (minval->min_leakage == 0) minval->min_leakage = 0.1; //FIXME remove this after leakage modeling

      if (v) {
        cost = (d  * ((*niter)->nuca_pda.delay/minval->min_delay) +
            c  * ((*niter)->nuca_pda.cycle_time/minval->min_cyc) +
            dp * ((*niter)->nuca_pda.power.readOp.dynamic/minval->min_dyn) +
            lp * ((*niter)->nuca_pda.power.readOp.leakage/minval->min_leakage) +
            a  * ((*niter)->nuca_pda.area.get_area()/minval->min_area));
        fprintf(stderr, "cost = %g\n", cost);

        if (min_cost > cost) {
          min_cost = cost;
          res = ((*niter));
        }
      }
      else {
        niter = n->erase(niter);
        if (niter !=n->begin())
        	niter --;
      }
    }
  }
  return res;
}

/*
 * [한국어]
 * Nuca::check_nuca_org - 입력 편차 제약을 만족하는지 검사
 *
 * @n     : 검사할 NUCA 구성
 * @minval: 최솟값 추적기 (각 메트릭의 기준값)
 * @return: 1=모든 제약 만족, 0=하나라도 위반
 *
 * 지연/동적전력/누설전력/사이클/면적 각각에 대해
 * (현재값 - 최솟값) / 최솟값 × 100 < 허용 편차(%) 조건을 검사한다.
 * g_ip->delay_dev_nuca 등의 허용 편차는 .cfg 파일에서 설정된다.
 * 호출 체인: find_optimal_nuca() → [이 함수]
 */
  int
Nuca::check_nuca_org (nuca_org_t *n, min_values_t *minval)
{
  if (((n->nuca_pda.delay - minval->min_delay)*100/minval->min_delay) > g_ip->delay_dev_nuca) {
    return 0; // [한국어] 지연이 최솟값 대비 허용 편차 초과 — 탈락
  }
  if (((n->nuca_pda.power.readOp.dynamic - minval->min_dyn)/minval->min_dyn)*100 >
      g_ip->dynamic_power_dev_nuca) {
    return 0; // [한국어] 동적 전력이 허용 편차 초과 — 탈락
  }
  if (((n->nuca_pda.power.readOp.leakage - minval->min_leakage)/minval->min_leakage)*100 >
      g_ip->leakage_power_dev_nuca) {
    return 0; // [한국어] 누설 전력이 허용 편차 초과 — 탈락
  }
  if (((n->nuca_pda.cycle_time - minval->min_cyc)/minval->min_cyc)*100 >
      g_ip->cycle_time_dev_nuca) {
    return 0; // [한국어] 사이클 타임이 허용 편차 초과 — 탈락
  }
  if (((n->nuca_pda.area.get_area() - minval->min_area)/minval->min_area)*100 >
      g_ip->area_dev_nuca) {
    return 0; // [한국어] 면적이 허용 편차 초과 — 탈락
  }
  return 1; // [한국어] 모든 제약 통과 — 이 구성은 비용 함수 평가 대상
}

/*
 * [한국어]
 * Nuca::calculate_nuca_area - 행×열 그리드 NUCA의 총 칩 면적 계산
 *
 * @nuca: 면적을 계산할 NUCA 구성 (rows, columns, h_wire, v_wire, router, bank_pda 사용)
 *
 * 각 행의 높이 = (수평 와이어 폭+간격) × flit_size + 뱅크 높이
 * 각 열의 너비 = (수직 와이어 폭+간격) × flit_size + 뱅크 너비
 * 총 높이 = rows × 행 높이, 총 너비 = columns × 열 너비
 *
 * 수평 와이어가 수직 방향 배선(배선 폭 × flit_size = 데이터 버스 폭)을 사용하는
 * 이유는 데이터가 수직 방향으로 흐를 때 수평 배선 치수가 수직 간격을 결정하기 때문이다.
 * 호출 체인: sim_nuca() → [이 함수]
 */
  void
Nuca::calculate_nuca_area (nuca_org_t *nuca)
{
  nuca->nuca_pda.area.h=
    nuca->rows * ((nuca->h_wire->wire_width +
          nuca->h_wire->wire_spacing)
        * nuca->router->flit_size +
        nuca->bank_pda.area.h);
  // [한국어] 총 높이 = 행 수 × (수평배선 피치 × flit폭비트 + 뱅크 높이)
  // 수평배선 폭+간격이 행 방향 버스 면적을 결정

  nuca->nuca_pda.area.w =
    nuca->columns * ((nuca->v_wire->wire_width +
          nuca->v_wire->wire_spacing)
        * nuca->router->flit_size +
        nuca->bank_pda.area.w);
  // [한국어] 총 너비 = 열 수 × (수직배선 피치 × flit폭비트 + 뱅크 너비)
}

