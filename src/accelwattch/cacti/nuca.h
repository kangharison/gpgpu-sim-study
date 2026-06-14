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
 * [한국어 설명] NUCA(Non-Uniform Cache Access) 캐시 모델 헤더 (nuca.h)
 *
 * === 파일의 역할 ===
 * NUCA는 뱅크 위치에 따라 접근 지연이 달라지는 멀티뱅크 캐시 구조를 모델링한다.
 * 이 파일은 NUCA 탐색 결과를 담는 nuca_org_t 클래스와, 다양한 뱅크 수/라우터 구성/
 * 와이어 모델에 대해 최적 NUCA 조직을 탐색하는 Nuca 클래스를 선언한다.
 * AccelWattch가 GPU L2 캐시의 에너지를 추정할 때 NUCA 모드가 선택되면 이 모듈이 호출된다.
 * CACTI 6.5 Tech Report에서 정의된 방법론(논문: CACTI 6.0)을 따른다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch → cacti_interface() → Nuca::sim_nuca() → [이 모듈]
 * Nuca::sim_nuca()는 UCA(solve())로 단일 뱅크 최적화를 수행한 뒤, 여러 뱅크를 NoC로
 * 연결한 NUCA 전체 조직의 지연/전력/면적을 계산하고 최적 구성을 선택한다.
 * 호스트 유저스페이스에서 실행되며, GPU 시뮬레이션 루프와 별개로 초기화 시 한 번 수행된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존: Wire(와이어 지연/전력), MCPAT_Router(NoC 라우터), Component(PDA 컨테이너),
 *       cacti_interface(g_ip 전역 입력 파라미터), Ucache(solve() - 단일 뱅크 탐색)
 * 소비: nuca_org_t는 find_optimal_nuca()가 최적 NUCA를 선택한 뒤 cacti_interface에 반환됨
 * 공유 자료구조: g_ip(InputParameter 전역), cont_stats(5차원 경쟁 통계 테이블)
 *
 * === 주요 함수/구조체 요약 ===
 * nuca_org_t      : 하나의 NUCA 구성 탐색 결과 (지연/전력/면적/라우터/와이어 포인터 보유)
 * Nuca::sim_nuca(): 모든 뱅크 수 × 라우터 × 와이어 타입 조합을 탐색하는 최상위 루프
 * Nuca::find_optimal_nuca(): 가중치 비용 함수로 최적 NUCA 구성 선택
 * Nuca::calc_cycles(): 지연 시간(초)을 사이클 수로 변환 (클럭 스큐 보정 포함)
 * Nuca::calculate_nuca_area(): 행×열 그리드 배치 시 총 칩 면적 계산
 */

#ifndef __NUCA_H__
#define __NUCA_H__

#include "basic_circuit.h"
#include "component.h"
#include "parameter.h"
#include "assert.h"
#include "cacti_interface.h"
#include "wire.h"
#include "mat.h"
#include "io.h"
#include "router.h"
#include <iostream>



class nuca_org_t {
  public:
  ~nuca_org_t();
//    int size;
    /* area, power, access time, and cycle time stats */
    Component nuca_pda;
    /* [한국어] NUCA 전체 시스템(네트워크+뱅크 포함)의 지연·전력·면적 통합 결과.
     * 설정자: sim_nuca()가 opt_acclat + 경쟁 지연을 더해 nuca_pda.delay를 설정.
     * 읽는 자: find_optimal_nuca()가 비용 함수 계산 시, print_nuca()가 출력 시 사용.
     * 값 범위: delay는 사이클 단위, power는 [W], area는 [um^2].
     * 동기화: sim_nuca()는 단일 스레드에서 호출되므로 별도 락 불필요. */

    Component bank_pda;
    /* [한국어] 단일 뱅크(UCA 최적 조직)의 지연·전력·면적 결과.
     * 설정자: sim_nuca()가 solve()로 얻은 ures.access_time/power/area를 복사.
     * 읽는 자: calculate_nuca_area()에서 뱅크 치수(area.h, area.w)로 그리드 면적 계산.
     * 값 범위: delay [s], power [W], area.h/w [um] — UCA 탐색 결과 그대로.
     * 동기화: 위와 동일. */

    Component wire_pda;
    /* [한국어] 네트워크 배선(수평+수직 와이어) 전력 결과 누산 컴포넌트.
     * 설정자: sim_nuca()가 wire_pda.power.readOp.dynamic을 평균 홉 × flit × (h+v 와이어 전력)으로 설정.
     * 읽는 자: 현재 직접 출력에는 사용되지 않으나 누적 전력 추적 목적.
     * 값 범위: dynamic [J/access] 단위.
     * 동기화: 단일 스레드. */

    Wire *h_wire;
    /* [한국어] 수평 방향 뱅크 간 배선 모델 포인터.
     * 설정자: sim_nuca()가 new Wire(wr, hlength)로 생성 후 할당.
     * 읽는 자: print_nuca()에서 와이어 타입·지연·전력 출력; calculate_nuca_area()에서 flit 폭 * wire_width 사용.
     * 값 범위: 유효한 Wire 포인터. 소유권은 Nuca 객체가 아닌 nuca_org_t에 있으나 ~nuca_org_t에서 free 하지 않음(주석 처리).
     * 동기화: 단일 스레드. */

    Wire *v_wire;
    /* [한국어] 수직 방향 뱅크 간 배선 모델 포인터.
     * 설정자: sim_nuca()가 new Wire(wr, vlength)로 생성 후 할당.
     * 읽는 자: calculate_nuca_area()에서 수직 배선 폭/간격을 행 방향 총 높이 계산에 사용.
     * 값 범위: 유효한 Wire 포인터.
     * 동기화: 단일 스레드. */

    MCPAT_Router *router;
    /* [한국어] 이 NUCA 구성에서 사용하는 NoC 라우터 모델 포인터.
     * 설정자: sim_nuca()가 router_s[ro]를 직접 대입 (소유권 공유).
     * 읽는 자: calc_cycles()에서 라우터 cycle_time으로 주파수 계산; print_nuca()에서 출력.
     * 값 범위: MCPAT_Router(64/128/256비트 flit 중 하나).
     * 동기화: 단일 스레드; 라우터 인스턴스는 여러 nuca_org_t가 공유할 수 있음. */

    /* for particular network configuration
     * calculated based on a cycle accurate
     * simulation Ref: CACTI 6 - Tech report
     */
    double contention;
    /* [한국어] 이 NUCA 구성에서 발생하는 네트워크 경쟁(contention) 추가 지연 [사이클].
     * 설정자: sim_nuca()가 cont_stats[l2_c][core_in][ro][it][num_cyc/2-1]에서 읽어 설정.
     * 읽는 자: print_nuca()에서 보고, find_optimal_nuca()는 nuca_pda.delay에 이미 포함된 값을 사용.
     * 값 범위: 0 이상의 사이클 수. contention.dat 파일에서 로드한 사이클 정확 시뮬레이션 결과.
     * 동기화: 단일 스레드. */

    /* grid network stats */
    double avg_hops;
    /* [한국어] 평균 홉 수 — 모든 뱅크까지의 평균 라우터 횡단 횟수.
     * 설정자: sim_nuca()에서 opt_avg_hop을 기록.
     * 읽는 자: wire_pda.power 계산 및 print 출력에 사용.
     * 값 범위: 1.0 이상 (최소 1홉 — 컨트롤러에서 첫 번째 라우터까지).
     * 동기화: 단일 스레드. */

    int rows;
    /* [한국어] 최적 그리드 배치의 행 수 (뱅크를 rows×columns 격자로 배치).
     * 설정자: sim_nuca()가 opt_rows 선택 후 저장.
     * 읽는 자: calculate_nuca_area()에서 총 높이 계산, print_nuca()에서 출력.
     * 값 범위: 1 이상, rows × columns == bank_count.
     * 동기화: 단일 스레드. */

    int columns;
    /* [한국어] 최적 그리드 배치의 열 수.
     * 설정자: sim_nuca()가 opt_columns 선택 후 저장.
     * 읽는 자: calculate_nuca_area()에서 총 너비 계산.
     * 값 범위: 1 이상.
     * 동기화: 단일 스레드. */

    int bank_count;
    /* [한국어] 이 NUCA 구성의 총 뱅크 수.
     * 설정자: sim_nuca()가 g_ip->nuca_cache_sz / g_ip->cache_sz로 계산해 저장.
     * 읽는 자: print_nuca(), sim_nuca()의 최적 뱅크 수 복원 시 사용.
     * 값 범위: 2, 4, 8, 16, 32, 64 중 하나 (2의 거듭제곱만 허용).
     * 동기화: 단일 스레드. */
};



class Nuca : public Component
{
  public:
    Nuca();
    /* [한국어] 기본 생성자 — deviceType을 g_tp.peri_global로 초기화하고 init_cont() 호출 */
    Nuca(
        TechnologyParameter::DeviceType *dt);
    /* [한국어] 디바이스 타입 지정 생성자 — 커스텀 CMOS 파라미터로 초기화 */
    void print_router();
    /* [한국어] (미사용) 라우터 통계 출력 래퍼 */
    ~Nuca();
    /* [한국어] 소멸자 — wire_vertical/wire_horizontal 배열의 Wire 객체들 delete */
    void sim_nuca();
    /* [한국어] NUCA 전체 탐색 최상위 함수 — 뱅크 수 × 라우터 타입 × 와이어 타입 조합 탐색 */
    void init_cont();
    /* [한국어] contention.dat 파일을 파싱해 cont_stats 5차원 배열 초기화 */
    int calc_cycles(double lat, double oper_freq);
    /* [한국어] 지연 [s]를 사이클 수로 변환 — 클럭 스큐·래치 지연 보정 포함 */
    void calculate_nuca_area (nuca_org_t *nuca);
    /* [한국어] 행×열 그리드 배치 시 NUCA 칩 총 면적(h × w) 계산 */
    int check_nuca_org (nuca_org_t *n, min_values_t *minval);
    /* [한국어] 입력 허용 편차 제약(delay_dev_nuca 등)을 만족하는지 검사 — 1이면 통과 */
    nuca_org_t * find_optimal_nuca (list<nuca_org_t *> *n, min_values_t *minval);
    /* [한국어] 가중치 비용 함수(ED/ED²/선형 가중합)로 최적 nuca_org_t 선택 */
    void print_nuca(nuca_org_t *n);
    /* [한국어] 최적 NUCA 구성의 상세 통계(뱅크 수, 그리드, 라우터, 와이어) 출력 */
    void print_cont_stats();
    /* [한국어] cont_stats 전체 배열을 콘솔에 출력하는 디버그 함수 */

  private:

    TechnologyParameter::DeviceType *deviceType;
    /* [한국어] 사용할 CMOS 공정 디바이스 파라미터 (Vdd, Vth, 이동도 등).
     * 설정자: 생성자에서 인자로 받거나 g_tp.peri_global 기본값 사용.
     * 읽는 자: Wire/MCPAT_Router 생성 시 전달됨.
     * 값 범위: TechnologyParameter::DeviceType 포인터 (null 불가).
     * 동기화: 불변 — 생성 후 변경 없음. */

    int wt_min, wt_max;
    /* [한국어] 탐색할 Wire_type 범위 (Global ~ Low_swing).
     * 설정자: sim_nuca()에서 g_ip->force_wiretype 여부에 따라 결정.
     * 읽는 자: sim_nuca()의 wr 루프에서 wt_min..wt_max를 순회.
     * 값 범위: Wire_type enum 값 (Global=0, ..., Low_swing).
     * 동기화: 단일 스레드. */

    Wire *wire_vertical[WIRE_TYPES],
         *wire_horizontal[WIRE_TYPES];
    /* [한국어] 수직/수평 와이어 인스턴스 배열 (와이어 타입별로 하나씩).
     * 설정자: sim_nuca()에서 new Wire(wr, vlength/hlength)로 생성.
     * 읽는 자: sim_nuca()에서 hop당 지연·전력 계산에 사용.
     * 값 범위: WIRE_TYPES 크기의 포인터 배열; 미사용 인덱스는 초기화되지 않을 수 있음.
     * 동기화: ~Nuca()에서 wt_min..wt_max 범위만 delete. */

};


#endif
