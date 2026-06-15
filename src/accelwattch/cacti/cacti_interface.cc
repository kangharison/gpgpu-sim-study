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
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
 *
 ***************************************************************************/

/*
 * [한국어 설명] CACTI 인터페이스 핵심 연산 구현 (cacti_interface.cc)
 *
 * === 파일의 역할 ===
 * CACTI(Cache Access and Cycle Time Information, HP Labs)의 최상위 결과 클래스인
 * uca_org_t와 설계 후보 비교자 mem_array::lt()의 핵심 멤버 함수들을 구현한다.
 * 이 파일은 CACTI 설계 공간 탐색이 완료된 후 선택된 최적 데이터/태그 어레이 쌍으로부터
 * 접근 시간(access_time), 사이클 시간(cycle_time), 전력(power), 물리 면적(area)을
 * 집계하는 후처리 연산을 담당한다.
 * AccelWattch는 이 파일의 함수들이 채운 uca_org_t 결과를 읽어 GPU 캐시 전력/지연을 추정한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름:
 *   AccelWattch(sharedcache.cc) → cacti_interface(InputParameter*)
 *     → Ucache::solve() [설계 공간 탐색: 모든 Ndwl/Ndbl/Nspd 조합 평가]
 *       → uca_org_t에 최적 tag_array2/data_array2 설정
 *         → [이 파일] uca_org_t::find_delay() / find_energy() / find_area() / find_cyc()
 *           → uca_org_t 결과를 AccelWattch로 반환
 * 실행 컨텍스트: 호스트 유저스페이스 — GPU 시뮬레이션 시작 전 초기화 단계에서 단 한 번 실행.
 * 사이클-레벨 타이밍 모델이 아니라, 캐시 설계점의 정적 특성(물리 파라미터 → 전기적 특성)을 계산.
 *
 * === 타 모듈과의 연결 ===
 * 이 파일이 의존하는 모듈:
 *   - cacti_interface.h: uca_org_t, mem_array, InputParameter, powerDef 선언
 *   - Ucache.h: Ucache::solve()가 채운 tag_array2/data_array2 포인터를 소비
 *   - parameter.h: 전역 입력 파라미터 포인터 g_ip (InputParameter*) 참조
 *   - const.h: MAX() 매크로 등 공통 상수
 *   - area.h, basic_circuit.h, component.h: CACTI 내부 면적/타이밍 계산 인프라
 * 이 파일에 의존하는 모듈:
 *   - AccelWattch sharedcache.cc: uca_org_t.power, access_time을 읽어 GPU 전력 추정
 *   - Ucache.cc: Ucache::solve()가 uca_org_t의 find_* 함수들을 호출
 * 핵심 공유 자료구조:
 *   - g_ip (InputParameter*): 전역 입력 파라미터 — pure_ram/fast_access/is_seq_acc 등 접근 모드 플래그
 *   - uca_org_t: tag_array2/data_array2 → find_*() 계산 후 access_time/power/area 등 채움
 *
 * === 주요 함수/구조체 요약 ===
 * mem_array::lt()     : mem_array 포인터 두 개를 6-key 사전순으로 비교 — STL sort/set 비교자로 사용
 * uca_org_t::find_delay() : 4가지 접근 모드(순수RAM/CAM/fast/sequential/normal)에 따라 access_time 계산
 * uca_org_t::find_energy(): 태그+데이터 어레이 전력을 합산하여 power 필드 설정
 * uca_org_t::find_area()  : 태그+데이터 어레이 높이/폭으로부터 cache_ht/cache_len/area 계산
 * uca_org_t::adjust_area(): 면적 효율이 20% 미만일 때 라우팅 오버헤드를 반영하여 면적 보정
 * uca_org_t::find_cyc()   : 태그/데이터 사이클 시간 중 최댓값으로 cycle_time 설정
 * uca_org_t::uca_org_t()  : 생성자 — tag_array2/data_array2를 NULL로 초기화
 * uca_org_t::cleanup()    : 동적 할당된 tag_array2/data_array2 해제
 */

#include <time.h>    // [한국어] time_t, clock() 등 시간 유틸 — CACTI 벤치마크 시 경과 시간 측정용
#include <math.h>    // [한국어] sqrt(), MAX() 내부 연산 등 수학 함수

#include "area.h"           // [한국어] 서브어레이/MAT 면적 계산 함수 선언
#include "basic_circuit.h"  // [한국어] 기본 회로 RC 파라미터, 게이트 지연 계산 함수
#include "component.h"      // [한국어] Component 기반 클래스 — 면적/전력/지연 계산 공통 인터페이스
#include "const.h"          // [한국어] CACTI 공통 상수 및 MAX() 매크로 정의
#include "parameter.h"      // [한국어] g_ip(InputParameter*), g_tp(TechParameter*) 전역 포인터 선언
#include "cacti_interface.h"// [한국어] uca_org_t, mem_array, InputParameter 클래스/구조체 선언
#include "Ucache.h"         // [한국어] Ucache::solve() — 설계 공간 탐색 엔진 선언

#include <pthread.h>  // [한국어] POSIX 스레드 헤더 — CACTI 멀티스레드 탐색 지원 (이 파일에서 직접 사용하지 않으나 헤더 연쇄)
#include <iostream>   // [한국어] cout/cerr — 경고/디버그 출력
#include <algorithm>  // [한국어] std::sort, std::min/max 등 — mem_array 후보 정렬 시 사용

using namespace std; // [한국어] STL 네임스페이스 전체 사용 — std:: 접두어 생략


/*
 * [한국어]
 * mem_array::lt - 두 mem_array 설계 후보를 파라미터 사전순으로 비교하는 정적 비교자
 *
 * @m1: 비교 대상 첫 번째 mem_array 포인터 — Ucache 탐색에서 생성된 후보
 * @m2: 비교 대상 두 번째 mem_array 포인터 — Ucache 탐색에서 생성된 후보
 * @return: m1이 m2보다 "작으면"(사전순으로 앞서면) true, 그렇지 않으면 false
 *
 * 이 함수는 STL sort() 또는 set<mem_array*, mem_array::lt> 같은 자료구조에서
 * 비교 기준으로 사용하기 위한 strict weak ordering 구현이다.
 * 비교 키(우선순위 순):
 *   1. Nspd (비트라인당 셀 비율)
 *   2. Ndwl (워드라인 방향 분할 수)
 *   3. Ndbl (비트라인 방향 분할 수)
 *   4. deg_bl_muxing (비트라인 먹스 차수)
 *   5. Ndsam_lev_1 (센스앰프 먹스 1차 분할)
 *   6. Ndsam_lev_2 (센스앰프 먹스 2차 분할)
 * 이 순서는 CACTI 설계 공간에서 탐색 결과를 재현 가능하게 정렬하기 위한 규약이다.
 * 실행 컨텍스트: 단일 스레드(또는 멀티스레드 탐색 완료 후 집계 단계).
 *
 * 호출 체인:
 *   Ucache::solve() → STL 정렬/비교 콜백 → [mem_array::lt] → 비교 결과 반환
 */
bool mem_array::lt(const mem_array * m1, const mem_array * m2)
{
  // [한국어] 1순위 키: Nspd(비트라인당 셀 비율) — 작은 쪽이 먼저
  if (m1->Nspd < m2->Nspd) return true;
  // [한국어] m1.Nspd가 더 크면 m1이 뒤 — 나머지 키는 무의미
  else if (m1->Nspd > m2->Nspd) return false;
  // [한국어] 2순위 키: Ndwl(워드라인 방향 분할 수) — Nspd가 같은 경우에만 진입
  else if (m1->Ndwl < m2->Ndwl) return true;
  // [한국어] m1.Ndwl이 더 크면 m1이 뒤
  else if (m1->Ndwl > m2->Ndwl) return false;
  // [한국어] 3순위 키: Ndbl(비트라인 방향 분할 수) — Nspd/Ndwl이 동일한 경우
  else if (m1->Ndbl < m2->Ndbl) return true;
  // [한국어] m1.Ndbl이 더 크면 m1이 뒤
  else if (m1->Ndbl > m2->Ndbl) return false;
  // [한국어] 4순위 키: deg_bl_muxing(비트라인 먹스 차수, 센스앰프 하나당 비트라인 수)
  else if (m1->deg_bl_muxing < m2->deg_bl_muxing) return true;
  // [한국어] m1.deg_bl_muxing이 더 크면 m1이 뒤
  else if (m1->deg_bl_muxing > m2->deg_bl_muxing) return false;
  // [한국어] 5순위 키: Ndsam_lev_1(센스앰프 먹스 1차 분할 인수)
  else if (m1->Ndsam_lev_1 < m2->Ndsam_lev_1) return true;
  // [한국어] m1.Ndsam_lev_1이 더 크면 m1이 뒤
  else if (m1->Ndsam_lev_1 > m2->Ndsam_lev_1) return false;
  // [한국어] 6순위(최종) 키: Ndsam_lev_2(센스앰프 먹스 2차 분할 인수)
  else if (m1->Ndsam_lev_2 < m2->Ndsam_lev_2) return true;
  // [한국어] 모든 키가 같거나 m1.Ndsam_lev_2 >= m2.Ndsam_lev_2 — m1이 앞서지 않음
  else return false;
}



/*
 * [한국어]
 * uca_org_t::find_delay - 캐시 접근 시간(access_time)을 4가지 접근 모드에 따라 계산
 *
 * @return: void (결과는 멤버 변수 access_time에 저장)
 *
 * 이 함수는 CACTI 설계 공간 탐색 완료 후, 선택된 tag_array2와 data_array2의 타이밍
 * 파라미터로부터 실제 캐시 접근 시간을 집계한다.
 * g_ip->pure_ram / pure_cam / fully_assoc / fast_access / is_seq_acc 플래그에 따라
 * 4가지 접근 모드 중 하나를 선택하여 access_time을 설정한다.
 *
 * [접근 모드별 동작]
 *   1. pure_ram / pure_cam / fully_assoc:
 *      태그 어레이가 없거나 태그+데이터가 일체형 — 데이터 어레이 접근 시간만 사용.
 *   2. fast_access:
 *      태그와 데이터 어레이를 동시에(병렬로) 접근하고, 전체 집합(set)을 H-tree로 전송.
 *      웨이-셀렉트(way-select) 신호를 기다리지 않으므로 더 빠르지만 전력 오버헤드가 있음.
 *      access_time = MAX(tag 접근 시간, data 접근 시간).
 *   3. is_seq_acc(sequential access):
 *      태그를 먼저 접근하여 히트를 확인한 후, 웨이-셀렉트 신호와 함께 데이터 어레이를 접근.
 *      직렬 경로이므로 느리지만 전력 절약.
 *      access_time = tag 접근 시간 + data 접근 시간.
 *   4. normal access(기본):
 *      태그와 데이터를 병렬로 접근하나, 데이터 어레이는 웨이-셀렉트를 기다린 후
 *      적절한 블록만 H-tree로 전송하는 절충형.
 *      access_time = MAX(tag 접근 시간 + data의 senseamp_mux_decoder 지연,
 *                        data의 subarray 출력 드라이버 이전까지 지연)
 *                  + data의 서브어레이 출력 드라이버 이후 출력까지 지연.
 *
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 스레드.
 * Caller: Ucache::solve() 또는 reconfigure()에서 탐색 완료 후 호출.
 *
 * 호출 체인:
 *   Ucache::solve() → [uca_org_t::find_delay] → tag_array2->access_time / data_array2->access_time 참조
 */
void uca_org_t::find_delay()
{
  // [한국어] 로컬 포인터로 접근 편의성 확보 — 멤버 포인터를 반복 역참조하는 오버헤드 제거
  mem_array * data_arr = data_array2; // [한국어] 최적 데이터 어레이 설계 후보 포인터 (Ucache가 선택)
  mem_array * tag_arr  = tag_array2;  // [한국어] 최적 태그 어레이 설계 후보 포인터 (순수 RAM이면 NULL)

  // check whether it is a regular cache or scratch ram
  // [한국어] 접근 모드 판별: 순수 RAM / CAM / 완전 결합 여부 확인
  // pure_ram: 스크래치패드 — 태그가 없어 데이터만 접근
  // pure_cam: CAM — 데이터와 태그가 통합된 매치라인 구조
  // fully_assoc: 완전 결합 캐시 — 모든 웨이를 동시에 비교하므로 데이터 접근 시간만 적용
  if (g_ip->pure_ram|| g_ip->pure_cam || g_ip->fully_assoc)
  {
    // [한국어] 태그 어레이 없이 데이터 어레이 접근 시간만 접근 시간으로 설정
    access_time = data_arr->access_time;
  }
  // Both tag and data lookup happen in parallel
  // and the entire set is sent over the data array h-tree without
  // waiting for the way-select signal --TODO add the corresponding
  // power overhead Nav
  // [한국어] fast_access 모드: 태그+데이터를 동시에 읽고 전체 집합을 H-tree로 전송.
  // 웨이-셀렉트 신호를 기다리지 않으므로 지연은 두 어레이 접근 시간의 최댓값.
  // 단, 불필요한 웨이 데이터도 전송되므로 동적 에너지 오버헤드가 있음 (TODO 주석).
  else if (g_ip->fast_access == true)
  {
    // [한국어] MAX() 매크로: 태그 접근과 데이터 접근 중 더 긴 경로가 전체 접근 시간 결정
    access_time = MAX(tag_arr->access_time, data_arr->access_time);
  }
  // Tag is accessed first. On a hit, way-select signal along with the
  // address is sent to read/write the appropriate block in the data
  // array
  // [한국어] sequential access 모드: 태그를 먼저 읽고, 히트 확인 후 데이터 어레이에 접근.
  // 직렬 경로이므로 접근 시간은 두 어레이 지연의 합.
  // 미스 시 데이터 어레이 접근이 발생하지 않아 에너지를 절약할 수 있음.
  else if (g_ip->is_seq_acc == true)
  {
    // [한국어] 직렬 경로: 태그 접근 완료 후 데이터 접근 시작 → 합산
    access_time = tag_arr->access_time + data_arr->access_time;
  }
  // Normal access: tag array access and data array access happen in parallel.
  // But, the data array will wait for the way-select and transfer only the
  // appropriate block over the h-tree.
  // [한국어] normal access 모드(기본): 태그+데이터를 병렬 접근하되,
  // 데이터 어레이는 웨이-셀렉트 신호를 받은 후 해당 블록만 H-tree로 전송하는 절충형.
  // 접근 시간 계산: 두 경로 중 늦게 완료되는 쪽이 서브어레이 출력 드라이버를 구동한 후,
  // H-tree를 통해 최종 출력에 도달하는 시간을 합산.
  else
  {
    // [한국어] 두 경로의 최댓값:
    //   경로A: 태그 어레이 접근 완료 + 데이터의 senseamp/mux/decoder 지연
    //          (웨이-셀렉트 신호가 데이터 어레이에 도달하는 시점)
    //   경로B: 데이터 어레이 서브어레이 출력 드라이버 직전까지의 지연
    //          (데이터 어레이가 내부적으로 준비 완료하는 시점)
    // 두 경로 중 더 늦은 쪽이 서브어레이 출력 드라이버를 구동하기 시작함.
    // 이후 서브어레이 출력 드라이버에서 칩 출력 핀까지의 지연을 더함.
    access_time = MAX(tag_arr->access_time + data_arr->delay_senseamp_mux_decoder,
                      data_arr->delay_before_subarray_output_driver) +
                  data_arr->delay_from_subarray_output_driver_to_output;
    // [한국어] delay_senseamp_mux_decoder: 웨이-셀렉트 신호가 데이터 서브어레이의
    //   감지증폭기 먹스 및 디코더를 통과하는 데 걸리는 추가 지연.
    // [한국어] delay_before_subarray_output_driver: 데이터 어레이가 내부적으로
    //   서브어레이 출력 드라이버를 활성화하기 직전까지 걸리는 시간 (비트라인+SA 등).
    // [한국어] delay_from_subarray_output_driver_to_output: 서브어레이 출력 드라이버에서
    //   H-tree를 거쳐 칩 출력까지의 지연 — 이 구간은 두 경로 모두 동일하게 추가됨.
  }
}



/*
 * [한국어]
 * uca_org_t::find_energy - 캐시 총 전력(power)을 데이터+태그 어레이 합산으로 계산
 *
 * @return: void (결과는 멤버 변수 power에 저장)
 *
 * 이 함수는 CACTI가 선택한 최적 데이터 어레이(data_array2)와 태그 어레이(tag_array2)의
 * 전력(powerDef: readOp/writeOp/searchOp 각각의 동적/누설/게이트누설 성분 합)을
 * 합산하여 uca_org_t.power를 설정한다.
 * 순수 RAM/CAM/완전 결합 캐시는 태그 어레이가 없으므로 데이터 어레이 전력만 사용.
 * AccelWattch(sharedcache.cc)는 이 함수 호출 후 power.readOp.dynamic 등을 읽어
 * GPU 캐시 접근당 에너지(J/access)와 누설 전력(W)을 추정한다.
 *
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 스레드.
 * Caller: Ucache::solve() 탐색 완료 후 호출.
 *
 * 호출 체인:
 *   Ucache::solve() → [uca_org_t::find_energy] → data_array2->power / tag_array2->power 참조
 */
void uca_org_t::find_energy()
{
  // [한국어] 일반 캐시(is_cache)인 경우: 태그 어레이도 존재하므로 두 어레이 전력을 합산.
  // g_ip->pure_ram / pure_cam / fully_assoc가 모두 false일 때 = 일반 집합 연관 캐시.
  // powerDef operator+는 readOp/writeOp/searchOp의 각 성분을 개별적으로 합산함.
  if (!(g_ip->pure_ram|| g_ip->pure_cam || g_ip->fully_assoc))//(g_ip->is_cache)
    // [한국어] 태그+데이터 어레이 전력 합산 (powerDef::operator+ 사용)
    power = data_array2->power + tag_array2->power;
  else
    // [한국어] 순수 RAM/CAM/완전 결합: 태그 어레이 없음 → 데이터 어레이 전력만 사용
    power = data_array2->power;
}



/*
 * [한국어]
 * uca_org_t::find_area - 캐시 물리 면적(cache_ht, cache_len, area)을 계산
 *
 * @return: void (결과는 cache_ht, cache_len, area 멤버에 저장)
 *
 * 이 함수는 최적 데이터/태그 어레이의 물리 치수(height, width)로부터
 * 캐시 전체의 바운딩 박스(bounding box)와 총 면적을 계산한다.
 * 순수 RAM/CAM/완전 결합: 데이터 어레이만 존재하므로 height/width를 그대로 사용.
 * 일반 캐시: 태그와 데이터 어레이가 나란히 배치됨 —
 *   높이 = MAX(tag.height, data.height): 더 높은 쪽에 맞춤
 *   폭   = tag.width + data.width:       좌우로 배치하여 합산
 * area = cache_ht × cache_len (직사각형 바운딩 박스 면적).
 *
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 스레드.
 * Caller: Ucache::solve() 탐색 완료 후 호출.
 *
 * 호출 체인:
 *   Ucache::solve() → [uca_org_t::find_area] → tag_array2->height/width, data_array2->height/width 참조
 */
void uca_org_t::find_area()
{
  // [한국어] 순수 RAM / CAM / 완전 결합: 태그 어레이가 없으므로 데이터 어레이 치수 그대로 사용
  if (g_ip->pure_ram|| g_ip->pure_cam || g_ip->fully_assoc)//(g_ip->is_cache == false)
  {
    // [한국어] 데이터 어레이의 높이를 캐시 높이로 설정 (단위: m)
    cache_ht  = data_array2->height;
    // [한국어] 데이터 어레이의 폭을 캐시 폭으로 설정 (단위: m)
    cache_len = data_array2->width;
  }
  else
  {
    // [한국어] 일반 집합 연관 캐시: 태그+데이터 어레이가 좌우로 나란히 배치됨.
    // 캐시 높이는 두 어레이 중 더 높은 쪽의 높이 (수직 기준 맞춤)
    cache_ht  = MAX(tag_array2->height, data_array2->height);
    // [한국어] 캐시 폭은 태그 어레이 폭 + 데이터 어레이 폭 (수평 방향 누적)
    cache_len = tag_array2->width + data_array2->width;
  }
  // [한국어] 총 면적 = 높이 × 폭 (직사각형 바운딩 박스)
  area = cache_ht * cache_len;
}

/*
 * [한국어]
 * uca_org_t::adjust_area - 면적 효율이 낮을 때 라우팅 오버헤드를 보정하여 면적 재계산
 *
 * @return: void (결과는 cache_ht, cache_len, area 멤버에 갱신)
 *
 * 이 함수는 McPAT 전용 보정 함수로, CACTI가 계산한 area_efficiency가 20% 미만인 경우
 * (즉, 실제 SRAM 셀 면적이 총 레이아웃의 20%에도 미치지 못하는 경우)
 * 라우팅/오버헤드가 과도하게 반영된 것으로 판단하여 면적을 축소 보정한다.
 * 보정 인수(area_adjust): sqrt(0.2 / (area_efficiency/100)) — 이상적 20% 기준 대비 비율의 제곱근.
 * 보정 후 cache_ht와 cache_len을 각각 area_adjust로 나누어 총 면적을 줄인다.
 * 결론적으로 area = cache_ht × cache_len이 실제 셀 기준 면적에 더 가까워진다.
 *
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 스레드.
 * Caller: McPAT가 find_area() 이후 호출.
 *
 * 호출 체인:
 *   McPAT → uca_org_t::find_area() → [uca_org_t::adjust_area] → area 갱신
 */
void uca_org_t::adjust_area()
{
  double area_adjust; // [한국어] 면적 보정 스케일 인수 — 1.0보다 크면 면적을 줄임
  // [한국어] 순수 RAM / CAM / 완전 결합 캐시에 대해서만 면적 효율 보정 적용
  if (g_ip->pure_ram|| g_ip->pure_cam || g_ip->fully_assoc)
  {
    // [한국어] 데이터 어레이의 면적 효율이 20% 미만인지 확인
    // area_efficiency는 0~100% 범위이므로 /100.0 하여 0~1.0 범위로 변환
    if (data_array2->area_efficiency/100.0<0.2)
    {
    	//area_adjust = sqrt(area/(area*(data_array2->area_efficiency/100.0)/0.2));
    	// [한국어] 보정 인수 계산: 이상적 효율 20%와 실제 효율의 비율의 제곱근.
    	// 원래 면적 = cache_ht × cache_len이 이미 설정되어 있고,
    	// 실제로는 그 중 area_efficiency/100 비율만 유효 셀이므로
    	// sqrt(0.2 / (efficiency))를 나누면 이상적 20% 기준의 면적으로 수렴.
    	area_adjust = sqrt(0.2/(data_array2->area_efficiency/100.0));
    	// [한국어] 높이를 보정 인수로 나누어 축소 (sqrt 비례 → 면적은 선형 보정)
    	cache_ht  = cache_ht/area_adjust;
    	// [한국어] 폭을 보정 인수로 나누어 축소
    	cache_len = cache_len/area_adjust;
    }
  }
  // [한국어] 최종 면적 재계산 (보정된 치수 또는 보정 불필요 시 기존 값)
  area = cache_ht * cache_len;
}

/*
 * [한국어]
 * uca_org_t::find_cyc - 사이클 시간(cycle_time)을 태그/데이터 어레이 중 최대값으로 설정
 *
 * @return: void (결과는 멤버 변수 cycle_time에 저장)
 *
 * 캐시 사이클 시간(cycle_time)은 연속 캐시 접근이 가능한 최소 시간 간격이다.
 * 이 값은 캐시가 지원하는 최대 주파수(= 1/cycle_time)를 결정한다.
 * 순수 RAM/CAM/완전 결합 캐시: 태그 어레이 없이 데이터 어레이의 사이클 시간만 사용.
 * 일반 캐시: 태그와 데이터 어레이 각각의 사이클 시간 중 더 큰 값이 전체를 제한.
 * (두 어레이가 하나의 클럭 도메인을 공유하므로 느린 쪽에 맞춰야 함.)
 *
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 스레드.
 * Caller: Ucache::solve() 탐색 완료 후 호출.
 *
 * 호출 체인:
 *   Ucache::solve() → [uca_org_t::find_cyc] → tag_array2->cycle_time / data_array2->cycle_time 참조
 */
void uca_org_t::find_cyc()
{
  // [한국어] 순수 RAM / CAM / 완전 결합: 태그 어레이 없음 → 데이터 어레이 사이클 시간만 적용
  if ((g_ip->pure_ram|| g_ip->pure_cam || g_ip->fully_assoc))//(g_ip->is_cache == false)
  {
    // [한국어] 데이터 어레이의 사이클 시간을 전체 캐시 사이클 시간으로 설정
    cycle_time = data_array2->cycle_time;
  }
  else
  {
    // [한국어] 일반 캐시: 태그/데이터 어레이 사이클 시간 중 더 긴 쪽이 전체 사이클 시간 결정.
    // 두 어레이는 동일 클럭 도메인에서 동작하므로 느린 쪽에 맞춰야 올바른 타이밍 보장.
    cycle_time = MAX(tag_array2->cycle_time,
                    data_array2->cycle_time);
  }
}

/*
 * [한국어]
 * uca_org_t::uca_org_t - 기본 생성자: 포인터 멤버를 NULL(0)로 초기화
 *
 * @return: (생성자, 반환값 없음)
 *
 * tag_array2와 data_array2는 동적으로 할당된 mem_array 객체를 가리키는 포인터이다.
 * 생성 직후에는 Ucache::solve()가 아직 이 포인터를 설정하지 않은 상태이므로
 * 안전을 위해 NULL(0)로 초기화한다.
 * cleanup() 함수는 NULL 여부를 검사하여 이중 해제(double-free)를 방지한다.
 * 숫자형 멤버(access_time, cycle_time, power, area 등)는 기본 초기화를 수행하지 않으므로
 * find_delay/find_energy/find_area/find_cyc 호출 전까지 값이 정의되지 않음에 유의.
 *
 * 실행 컨텍스트: 호스트 유저스페이스.
 * Caller: cacti_interface() 또는 reconfigure() 내부에서 uca_org_t 객체 생성 시.
 *
 * 호출 체인:
 *   cacti_interface() → [uca_org_t::uca_org_t()] → 멤버 초기화 완료
 */
uca_org_t :: uca_org_t()
:tag_array2(0),   // [한국어] 태그 어레이 포인터 NULL로 초기화 — Ucache::solve() 전까지 유효하지 않음
 data_array2(0)   // [한국어] 데이터 어레이 포인터 NULL로 초기화 — Ucache::solve() 전까지 유효하지 않음
{
  // [한국어] 생성자 본문 비어 있음 — 초기화는 멤버 초기화 리스트에서 완료됨
}

/*
 * [한국어]
 * uca_org_t::cleanup - 동적 할당된 mem_array 객체를 해제하는 소멸 전처리 함수
 *
 * @return: void
 *
 * tag_array2와 data_array2는 Ucache::solve()가 new로 동적 할당한 mem_array 객체를 가리킨다.
 * C++ 소멸자(~uca_org_t)는 비어 있으므로, 사용자가 결과를 다 쓴 후 명시적으로
 * 이 함수를 호출해야 메모리 누수가 발생하지 않는다.
 * NULL 검사(!=0)로 이중 해제를 방지한다.
 * 주의: 이 함수를 호출한 후에는 tag_array2/data_array2 포인터가 허상 포인터(dangling pointer)가 되므로
 * 이후 find_delay/find_energy 등을 호출하면 정의되지 않은 동작이 발생한다.
 *
 * 실행 컨텍스트: 호스트 유저스페이스.
 * Caller: AccelWattch 또는 Ucache 탐색 루프에서 결과 소비 완료 후 호출.
 *
 * 호출 체인:
 *   AccelWattch → uca_org_t 사용 완료 → [uca_org_t::cleanup] → delete data_array2 / delete tag_array2
 */
void uca_org_t :: cleanup()
{
  // [한국어] data_array2가 NULL이 아닌 경우에만 해제 — 이중 해제 방지
	  if (data_array2!=0)
		  delete data_array2; // [한국어] Ucache::solve()가 new로 할당한 데이터 어레이 결과 객체 해제
  // [한국어] tag_array2가 NULL이 아닌 경우에만 해제 (순수 RAM이면 NULL일 수 있음)
	  if (tag_array2!=0)
		  delete tag_array2; // [한국어] Ucache::solve()가 new로 할당한 태그 어레이 결과 객체 해제
}
