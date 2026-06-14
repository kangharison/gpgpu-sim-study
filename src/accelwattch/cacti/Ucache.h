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
 * [한국어 설명] UCA 설계 공간 탐색 조율 헤더 (Ucache.h)
 *
 * === 파일의 역할 ===
 * UCA 캐시의 전체 설계 공간 탐색(design space exploration)을 지원하는 자료구조와 함수를 선언한다.
 * min_values_t는 탐색 중 발견한 최소 지연·전력·면적·사이클 값을 추적하며 비용 정규화에 사용된다.
 * solve()는 Ndwl/Ndbl/Nspd/Ndsam 조합을 pthread 멀티스레딩으로 병렬 탐색해 최적 UCA를 선택한다.
 * init_tech_params()는 공정 파라미터를 태그/데이터 배열에 맞게 초기화한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * cacti_interface() → solve(fin_res) → [이 모듈의 solve/calculate_time]
 * solve()가 calc_time_mt_wrapper를 nthreads개 스레드로 실행하고,
 * 각 스레드가 calculate_time()을 반복 호출해 UCA 인스턴스를 평가한다.
 * Nuca::sim_nuca()도 solve()를 호출해 단일 뱅크의 최적 조직을 먼저 구한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존: UCA(uca.h), nuca(nuca.h), mem_array(cacti_interface.h), DynamicParameter,
 *       pthread(병렬 실행), area.h(Area), router.h(MCPAT_Router)
 * 소비: cacti_interface()가 solve()를 호출해 fin_res에 최적 조직 저장
 * 공유: g_ip(전역 입력), g_tp(전역 기술 파라미터), NTHREADS(스레드 수 상수)
 *
 * === 주요 함수/구조체 요약 ===
 * min_values_t          : 탐색 중 최솟값 추적기 — 4개 오버로드 update_min_values() 제공
 * solution              : 태그+데이터 배열 조합 하나의 캐시 솔루션 기술
 * calculate_time()      : Ndwl/Ndbl/... 하나의 파라미터 조합에 대해 UCA 생성 및 결과 추출
 * solve()               : 멀티스레드 설계 공간 탐색 최상위 함수
 * calc_time_mt_wrapper(): 각 pthread가 실행하는 래퍼 — 스레드별 파라미터 부분집합 탐색
 * update()              : 최적해 확정 후 태그/데이터 배열의 누설 전력 재계산
 */

#ifndef __UCACHE_H__
#define __UCACHE_H__

#include <list>
#include "area.h"
#include "router.h"
#include "nuca.h"


class min_values_t
{
  public:
    double min_delay;
    /* [한국어] 탐색 중 발견한 최소 접근 지연 [s].
     * 설정자: update_min_values()의 각 오버로드에서 현재 값보다 작으면 갱신.
     * 읽는 자: find_optimal_uca()/find_optimal_nuca()에서 비용 정규화 분모로 사용.
     * 초기값: BIGNUM (매우 큰 수) — 첫 번째 유효 결과가 즉시 최솟값으로 설정됨.
     * 동기화: 멀티스레드 환경에서 각 스레드가 독립 min_values_t를 가지며, pthread_join 후 병합. */

    double min_dyn;
    /* [한국어] 최소 동적(읽기) 전력 [J/access].
     * 설정자: update_min_values()에서 res.power.readOp.dynamic와 비교 후 갱신.
     * 읽는 자: 비용 함수 dp × (dyn/min_dyn) 항의 분모.
     * 초기값: BIGNUM. */

    double min_leakage;
    /* [한국어] 최소 누설 전력 [W].
     * 설정자: update_min_values()에서 갱신.
     * 읽는 자: 비용 함수 lp × (leak/min_leakage) 항의 분모.
     * 초기값: BIGNUM. 주의: NUCA에서 0이 되면 0.1로 보정하는 FIXME 코드 있음. */

    double min_area;
    /* [한국어] 최소 면적 [um²].
     * 설정자: update_min_values()에서 갱신.
     * 읽는 자: 비용 함수 a × (area/min_area) 항의 분모.
     * 초기값: BIGNUM. */

    double min_cyc;
    /* [한국어] 최소 사이클 타임 [s].
     * 설정자: update_min_values()에서 갱신.
     * 읽는 자: 비용 함수 c × (cyc/min_cyc) 항의 분모.
     * 초기값: BIGNUM. */

    min_values_t() : min_delay(BIGNUM), min_dyn(BIGNUM), min_leakage(BIGNUM), min_area(BIGNUM), min_cyc(BIGNUM) { }
    /* [한국어] 기본 생성자 — 모든 멤버를 BIGNUM으로 초기화해 최솟값 탐색 시작 준비 */

    void update_min_values(const min_values_t * val);
    /* [한국어] 다른 min_values_t의 최솟값으로 이 객체를 갱신 — 스레드 결과 병합 시 사용 */
    void update_min_values(const uca_org_t & res);
    /* [한국어] uca_org_t 결과로 최솟값 갱신 — calculate_time() 결과를 추적할 때 */
    void update_min_values(const nuca_org_t * res);
    /* [한국어] nuca_org_t 결과로 최솟값 갱신 — sim_nuca() 탐색 루프에서 */
    void update_min_values(const mem_array * res);
    /* [한국어] mem_array 결과로 최솟값 갱신 — calc_time_mt_wrapper의 data_res/tag_res 갱신 시 */
};



struct solution
{
  /* [한국어] 태그+데이터 배열 조합 하나의 완성된 캐시 솔루션 기술 구조체.
   * solve()에서 tag_arr × data_arr 카르테시안 곱을 구성할 때 각 조합을 이 구조체로 표현할 수 있다.
   * 현재 코드에서는 uca_org_t에 직접 결과를 채우므로 이 구조체는 참조용. */
  int    tag_array_index;
  /* [한국어] 태그 배열의 인덱스 — tag_arr 리스트에서의 위치 식별자. */
  int    data_array_index;
  /* [한국어] 데이터 배열의 인덱스 — data_arr 리스트에서의 위치 식별자. */
  list<mem_array *>::iterator tag_array_iter;
  /* [한국어] 선택된 태그 배열을 가리키는 리스트 이터레이터. */
  list<mem_array *>::iterator data_array_iter;
  /* [한국어] 선택된 데이터 배열을 가리키는 리스트 이터레이터. */
  double access_time;
  /* [한국어] 이 솔루션의 캐시 접근 지연 [s]. */
  double cycle_time;
  /* [한국어] 이 솔루션의 사이클 타임 [s]. */
  double area;
  /* [한국어] 이 솔루션의 총 면적 [um²]. */
  double efficiency;
  /* [한국어] 면적 효율 [%] = 셀 면적 / 총 면적 × 100. */
  powerDef total_power;
  /* [한국어] 이 솔루션의 총 전력 (readOp/writeOp/searchOp 각각의 동적·누설 전력). */
};



bool calculate_time(
    bool is_tag,
    int pure_ram,
    bool pure_cam,
    double Nspd,
    unsigned int Ndwl,
    unsigned int Ndbl,
    unsigned int Ndcm,
    unsigned int Ndsam_lev_1,
    unsigned int Ndsam_lev_2,
    mem_array *ptr_array,
    int flag_results_populate,
    results_mem_array *ptr_results,
    uca_org_t *ptr_fin_res,
    bool is_main_mem);
void update(uca_org_t *fin_res);

void solve(uca_org_t *fin_res);
void init_tech_params(double tech, bool is_tag);


struct calc_time_mt_wrapper_struct
{
  /* [한국어] pthread 스레드 하나에 전달되는 인수 구조체.
   * solve()가 nthreads개를 배열로 할당해 각 스레드에 부분집합(tid 기반 iter 분배) 탐색 위임. */
  uint32_t tid;
  /* [한국어] 스레드 ID (0 ~ nthreads-1) — iter % nthreads == tid인 파라미터 조합을 처리.
   * 설정자: solve()가 각 스레드에 순번 할당. 읽는 자: calc_time_mt_wrapper()에서 iter 분배에 사용. */
  bool     is_tag;
  /* [한국어] 태그 배열 탐색 여부 — true이면 캐시 태그 배열, false이면 데이터 배열. */
  bool     pure_ram;
  /* [한국어] 순수 RAM 모드 (캐시 태그 없이 데이터만 있는 구조). */
  bool     pure_cam;
  /* [한국어] 순수 CAM 모드 (태그만 있는 연관 검색 구조). */
  bool     is_main_mem;
  /* [한국어] 메인 메모리(DRAM) 모드 여부 — access_time 계산 방식이 달라짐. */
  double   Nspd_min;
  /* [한국어] 탐색 시작 Nspd(서브어레이 당 병렬 비트 수) 최솟값.
   * 데이터 배열: out_w / (block_sz×8), 태그/CAM/FA: 0.125(또는 1). */

  min_values_t * data_res;
  /* [한국어] 이 스레드가 발견한 데이터 배열 최솟값 추적기.
   * 설정자: calc_time_mt_wrapper()가 is_valid_partition이면 data_res->update_min_values() 호출.
   * 읽는 자: solve()가 pthread_join 후 d_min에 병합. */
  min_values_t * tag_res;
  /* [한국어] 이 스레드가 발견한 태그 배열 최솟값 추적기. */

  list<mem_array *> data_arr;
  /* [한국어] 이 스레드가 발견한 유효 데이터 배열 결과 리스트.
   * 설정자: is_valid_partition이면 data_arr.push_back(new mem_array) 후 채움.
   * 읽는 자: solve()가 pthread_join 후 모든 스레드 결과를 병합(merge). */
  list<mem_array *> tag_arr;
  /* [한국어] 이 스레드가 발견한 유효 태그 배열 결과 리스트. */
};

void *calc_time_mt_wrapper(void * void_obj);

#endif
