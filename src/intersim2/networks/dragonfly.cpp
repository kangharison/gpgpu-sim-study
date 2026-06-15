/*
  Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
  All rights reserved.

  Redistribution and use in source and binary forms, with or without modification,
  are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this list
  of conditions and the following disclaimer.
  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
  Neither the name of the Stanford University nor the names of its contributors
  may be used to endorse or promote products derived from this software without
  specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 * [한국어 설명] DragonFly 네트워크 토폴로지 구현 (dragonfly.cpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 Dragonfly 고-기수(high-radix)/저-직경(low-diameter) 네트워크 토폴로지의
 * 전체 구현을 담당한다. 구체적으로:
 *   1) dragonflynew_hopcnt(): src→dest 최소 홉 수 계산 (현재 직접 호출은 주석처리됨)
 *   2) dragonfly_port(): 현재 라우터 위치에서 최소 경로 출력 포트 번호 결정
 *   3) DragonFlyNew 클래스 멤버: 네트워크 크기 계산, 라우터/채널 생성 및 연결
 *   4) min_dragonflynew(): 최소 경로 라우팅 (2 VC, dateline 데드락 방지)
 *   5) ugal_dragonflynew(): UGAL 적응형 라우팅 (3 VC, 큐 길이 기반 경로 선택)
 * GPGPU-Sim의 intersim2(Booksim 기반) 서브시스템에서 GPU 내부 NoC 모델링에 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 실행 흐름:
 *   gpgpu_sim::cycle()
 *     → icnt_wrapper::advance_time()        (intersim2 사이클 진행)
 *         → IcntWrap::Advance()
 *             → BookSimNetwork::ReadInputs() / WriteOutputs()
 *                 → Router::AddFlit()
 *                     → min_dragonflynew() 또는 ugal_dragonflynew()  [이 파일]
 *                         → dragonfly_port()  [이 파일]
 * 실행 컨텍스트: 호스트 유저스페이스 단일 스레드. GPU 디바이스 코드가 아니며,
 * 사이클마다 시뮬레이터 메인 루프가 호출한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - booksim.hpp: Booksim 공통 타입 및 전역 변수 (gNumVCs, gWatchOut, GetSimTime 등)
 *   - dragonfly.hpp: 이 파일에서 구현하는 클래스/함수의 선언
 *   - random_utils.hpp: RandomInt() — UGAL에서 중간 노드 랜덤 선택에 사용
 *   - misc_utils.hpp: powi() — _p^n 계산에 사용
 *   - globals.hpp: gP, gA, gG, gK, gN 등 전역 파라미터; gRoutingFunctionMap
 *   - router.hpp (간접): Router 클래스 — GetID(), GetUsedCredit() 사용
 *   - flit.hpp (간접): Flit 클래스 — ph, intm, src, dest, watch 필드 사용
 * 의존받는 모듈:
 *   - icnt_wrapper.cc: icnt_wrapper_init()에서 DragonFlyNew 인스턴스 생성
 *   - routefunc.cpp: gRoutingFunctionMap을 통해 라우팅 함수 호출
 * 핵심 전역 변수:
 *   - gP(=_p): 라우터당 터미널 포트 수 — 라우팅 함수 전체에서 직접 참조
 *   - gA(=_a): 그룹당 라우터 수 — 그룹 ID/포트 계산에 사용
 *   - gG(=_g): 총 그룹 수 — UGAL의 네트워크 크기 계산에 사용
 *
 * === 주요 함수/구조체 요약 ===
 * dragonflynew_hopcnt(src, dest):
 *   - src~dest 간 최소 홉 수 계산 (0~3 홉)
 *   - 같은 그룹: 0(같은 라우터) 또는 1(다른 라우터)
 *   - 다른 그룹: src_hopcnt + 1(inter-group) + dest_hopcnt
 * dragonfly_port(rID, source, dest):
 *   - 3가지 경우 처리: (1)최종 홉 → 터미널 포트, (2)inter-group 링크 → 광학 포트,
 *     (3)intra-group 라우팅 → 내부 포트
 * DragonFlyNew::_ComputeSize():
 *   - config에서 k, n 읽어 전체 파라미터 (_p, _a, _g, _nodes, _k, _channels 등) 결정
 * DragonFlyNew::_BuildNet():
 *   - 각 라우터에 inject/eject + intra/inter-group 채널 연결 (latency 차등 적용)
 * min_dragonflynew(): dateline VC 기반 최소 라우팅
 * ugal_dragonflynew(): 큐 길이 비교 기반 UGAL 적응형 라우팅
 */

#include "booksim.hpp"   // [한국어] Booksim 공통 헤더 — gNumVCs, gWatchOut, GetSimTime 등 전역 선언
#include <vector>        // [한국어] std::vector — 채널/라우터 배열 관리용 (Network 기본 클래스에서 사용)
#include <sstream>       // [한국어] std::ostringstream — 라우터 이름 문자열 생성에 사용

#include "dragonfly.hpp"    // [한국어] DragonFlyNew 클래스 및 관련 함수 선언
#include "random_utils.hpp" // [한국어] RandomInt() — UGAL 라우팅에서 중간 노드 랜덤 선택
#include "misc_utils.hpp"   // [한국어] powi() — p의 n제곱 계산 (n>1 차원 지원 시 사용)
#include "globals.hpp"      // [한국어] gP, gA, gG, gK, gN, gRoutingFunctionMap 등 전역 변수

#define DRAGON_LATENCY
/* [한국어] DRAGON_LATENCY 매크로 — 채널별 차등 레이턴시 설정 활성화.
 * 이 매크로가 정의되면 _BuildNet()에서:
 *   - intra-group 채널: SetLatency(10) — 그룹 내 짧은 구리 케이블 모델링
 *   - inter-group 채널: SetLatency(100) — 그룹 간 광학 링크의 높은 레이턴시 모델링
 * 주석 처리 시 모든 채널이 기본 레이턴시(1 사이클)를 사용한다.
 * 실제 HPC 시스템에서 광학 링크는 구리 케이블 대비 약 10배 높은 레이턴시를 가지므로
 * 이 10:100 비율이 현실적인 Dragonfly 특성을 반영한다. */

int gP, gA, gG;
/* [한국어] Dragonfly 전역 파라미터 — 라우팅 함수에서 직접 참조하는 핵심 변수.
 * gP: 라우터당 프로세서(터미널) 포트 수 (=_p). 터미널 노드 → 라우터 연결 수.
 * gA: 그룹당 라우터 수 (=_a). n=1일 때 2*gP.
 * gG: 총 그룹 수 (=_g). gA*gP + 1.
 * 설정자: DragonFlyNew::_ComputeSize()에서 단 한 번 설정됨.
 * 읽는 자: dragonflynew_hopcnt(), dragonfly_port(), min_dragonflynew(),
 *           ugal_dragonflynew() — 이 파일의 모든 전역 함수에서 읽음.
 * 값 범위: 시뮬레이션 시작 시 설정된 후 변경되지 않음.
 * 동기화: 단일 스레드 환경이므로 별도 동기화 불필요.
 *         단, 시뮬레이터 시작 전(DragonFlyNew 생성 전)에 라우팅 함수를 호출하면 미초기화 상태 주의. */


/*
 * [한국어]
 * dragonflynew_hopcnt - src 터미널에서 dest 터미널까지의 최소 홉 수 계산
 *
 * @src:  출발 터미널 노드 ID (0 ~ gA*gP*gG - 1).
 * @dest: 도착 터미널 노드 ID (0 ~ gA*gP*gG - 1).
 * @return: 최소 홉 수 (0 ~ 3).
 *          0: src와 dest가 같은 라우터에 연결된 경우.
 *          1: 같은 그룹, 다른 라우터 (intra-group 1홉).
 *          2: 다른 그룹, src 또는 dest가 inter-group 라우터와 직결 (intra + inter 또는 inter + intra).
 *          3: 다른 그룹, src와 dest 모두 intra-group 이동 필요 (1+1+1).
 *
 * Dragonfly에서 최소 홉 수는 소스 그룹 내 이동(0~1홉) + 그룹 간 이동(1홉) +
 * 목적지 그룹 내 이동(0~1홉)으로 구성된다. 최대 3홉으로 직경이 낮다.
 * 이 함수는 UGAL 라우팅의 비최소/최소 경로 비용 비교에 사용될 수 있으나,
 * 현재 ugal_dragonflynew() 구현에서는 주석 처리되어 있고 큐 길이만 사용된다.
 *
 * 호출 체인:
 *   (참고용) ugal_dragonflynew() → [이 함수] (현재 주석처리로 미사용)
 */
//calculate the hop count between src and estination
int dragonflynew_hopcnt(int src, int dest)
{
  int hopcnt;                // [한국어] 최종 반환할 최소 홉 수
  int dest_grp_ID, src_grp_ID;   // [한국어] 목적지/출발 터미널이 속한 그룹 ID
  int src_hopcnt, dest_hopcnt;   // [한국어] 소스 그룹 내 홉 수 / 목적지 그룹 내 홉 수
  int src_intm, dest_intm;       // [한국어] 소스/목적지 그룹에서 inter-group 연결 라우터의 첫 번째 터미널 ID
  int grp_output, dest_grp_output; // [한국어] inter-group 채널 인덱스 — 목적지 그룹 방향의 광학 포트 번호
  int grp_output_RID;            // [한국어] inter-group 광학 링크를 보유한 라우터의 전역 ID

  int _grp_num_routers = gA;      // [한국어] 그룹당 라우터 수 — 전역 변수 gA를 지역 변수로 캐시 (성능/가독성)
  int _grp_num_nodes = _grp_num_routers * gP; // [한국어] 그룹당 터미널 노드 수 = 라우터 수 * 라우터당 터미널 수

  dest_grp_ID = int(dest / _grp_num_nodes); // [한국어] dest 노드가 속한 그룹 ID: 노드ID / 그룹당노드수
  src_grp_ID  = int(src  / _grp_num_nodes); // [한국어] src 노드가 속한 그룹 ID: 노드ID / 그룹당노드수

  //source and dest are in the same group, either 0-1 hop
  if (dest_grp_ID == src_grp_ID) { // [한국어] 같은 그룹 내: inter-group 링크 불필요, 최대 1홉
    if ((int)(dest / gP) == (int)(src / gP)) // [한국어] 같은 라우터에 연결된 터미널이면 0홉
      hopcnt = 0; // [한국어] 같은 라우터 — eject 직접 전달, 0홉
    else
      hopcnt = 1; // [한국어] 다른 라우터 — intra-group 링크 1홉

  } else {
    //source and dest are in the same group
    //find the number of hops in the source group
    //find the number of hops in the dest group

    // [한국어] 다른 그룹의 경우: inter-group 링크를 사용하는 라우터 찾기.
    // Dragonfly에서 inter-group 광학 링크는 그룹 내 특정 라우터에 배정된다.
    // grp_output: 소스 그룹에서 목적지 그룹 방향의 inter-group 채널 인덱스.
    // dest_grp_output: 목적지 그룹에서 소스 그룹 방향의 inter-group 채널 인덱스.
    // 채널 인덱스는 그룹 ID 비교로 결정된다 (자기 자신 그룹은 채널 목록에서 제외되므로 -1 보정).
    if (src_grp_ID > dest_grp_ID) {  // [한국어] 소스 그룹 ID가 더 크면: 목적지 그룹 채널 인덱스는 그대로
      grp_output      = dest_grp_ID;      // [한국어] 소스 그룹 기준 채널 인덱스 = dest_grp_ID
      dest_grp_output = src_grp_ID - 1;   // [한국어] 목적지 그룹 기준 채널 인덱스 = src_grp_ID-1 (자기 제외 보정)
    } else {
      grp_output      = dest_grp_ID - 1;  // [한국어] 소스 그룹 기준 채널 인덱스 = dest_grp_ID-1 (자기 제외 보정)
      dest_grp_output = src_grp_ID;       // [한국어] 목적지 그룹 기준 채널 인덱스 = src_grp_ID
    }

    // [한국어] 소스 그룹에서 inter-group 광학 링크를 가진 라우터 ID 계산.
    // grp_output / gP: 소스 그룹 내 해당 광학 링크를 보유한 라우터의 그룹 내 인덱스.
    // + src_grp_ID * _grp_num_routers: 그룹 오프셋을 더해 전역 라우터 ID로 변환.
    grp_output_RID = ((int)(grp_output / (gP))) + src_grp_ID * _grp_num_routers;
    src_intm = grp_output_RID * gP; // [한국어] 해당 라우터의 첫 번째 터미널 노드 ID

    // [한국어] 목적지 그룹에서 inter-group 광학 링크를 가진 라우터 ID 계산.
    grp_output_RID = ((int)(dest_grp_output / (gP))) + dest_grp_ID * _grp_num_routers;
    dest_intm = grp_output_RID * gP; // [한국어] 목적지 그룹의 inter-group 라우터 첫 터미널 ID

    //hop count in source group
    // [한국어] 소스 터미널이 이미 inter-group 라우터에 있으면 0홉, 아니면 1홉(intra 이동 필요).
    if ((int)(src_intm / gP) == (int)(src / gP))
      src_hopcnt = 0; // [한국어] src가 이미 inter-group 라우터에 연결 — intra 이동 불필요
    else
      src_hopcnt = 1; // [한국어] src가 다른 라우터에 있어 intra-group 1홉 필요

    //hop count in destination group
    // [한국어] 목적지 터미널이 이미 inter-group 라우터에 있으면 0홉, 아니면 1홉.
    if ((int)(dest_intm / gP) == (int)(dest / gP)) {
      dest_hopcnt = 0; // [한국어] dest가 inter-group 라우터에 직결 — 목적지 그룹 내 intra 불필요
    } else {
      dest_hopcnt = 1; // [한국어] dest가 다른 라우터에 있어 intra-group 1홉 필요
    }

    //tally
    hopcnt = src_hopcnt + 1 + dest_hopcnt; // [한국어] 총 홉 = 소스측 + inter-group 광학 링크(1홉) + 목적지측
  }

  return hopcnt; // [한국어] 최소 홉 수 반환 (0~3)
}


/*
 * [한국어]
 * dragonfly_port - 현재 라우터(rID)에서 패킷(source→dest)의 최소 경로 출력 포트 계산
 *
 * @rID:    현재 패킷이 위치한 라우터의 전역 ID (0 ~ _num_of_switch-1).
 * @source: 패킷 출발 터미널 노드 ID (실질적으로 이 함수에서는 직접 사용 안 되나 시그니처에 포함).
 * @dest:   패킷 목적지 터미널 노드 ID.
 * @return: 출력 포트 번호 (0 ~ _k-1).
 *          포트 구조:
 *            [0 ~ gP-1]              : 터미널(eject) 포트 — 최종 홉에서만 선택
 *            [gP ~ gP+(gA-2)]        : intra-group 포트 (2*gP-1개 중 앞 gA-1개)
 *            [gP+(gA-1) ~ gP+(gA-1)+(gP-1)]: inter-group 광학 포트 (gP개)
 *
 * 3가지 경우 중 하나를 선택:
 *   1) 최종 홉: dest 터미널이 현재 라우터에 연결 → 터미널 포트(dest%gP)
 *   2) inter-group 광학 홉: 목적지 그룹이 다르고 현재 라우터가 해당 광학 링크 보유
 *      → gP + (gA-1) + grp_output%(gP) 번 포트
 *   3) intra-group 이동: 현재 그룹 내에서 inter-group 라우터 또는 목적지 라우터로 이동
 *      → (grp_RID % _grp_num_routers) ± 1 + gP 번 포트
 *
 * 주의: rID 기준 좌측/우측 라우터 방향에 따라 포트 번호가 달라진다.
 *       rID < grp_RID: 상대 라우터가 오른쪽 → (grp_RID % _grp_num_routers) - 1 + gP
 *       rID > grp_RID: 상대 라우터가 왼쪽  → (grp_RID % _grp_num_routers) + gP
 *
 * 호출 체인:
 *   min_dragonflynew()      → [이 함수]
 *   ugal_dragonflynew()     → [이 함수]
 *   dragonflynew_hopcnt()는 이 함수를 호출하지 않음
 */
//packet output port based on the source, destination and current location
int dragonfly_port(int rID, int source, int dest) {
  int _grp_num_routers = gA;              // [한국어] 그룹당 라우터 수 — 전역 gA를 지역 변수로 캐시
  int _grp_num_nodes = _grp_num_routers * gP; // [한국어] 그룹당 터미널 노드 수

  int out_port  = -1;                      // [한국어] 출력 포트 번호 초기값 -1 (미결정 상태 표시)
  int grp_ID    = int(rID / _grp_num_routers);  // [한국어] 현재 라우터가 속한 그룹 ID
  int dest_grp_ID = int(dest / _grp_num_nodes); // [한국어] 목적지 터미널이 속한 그룹 ID
  int grp_output = -1; // [한국어] 소스 그룹에서 목적지 방향의 inter-group 채널 인덱스 (미결정 초기값)
  int grp_RID    = -1; // [한국어] 다음으로 이동해야 할 목표 라우터의 전역 ID (미결정 초기값)
  // int group_dest=-1;   // [한국어] (주석처리된 변수) 목적지 그룹의 첫 터미널 ID

  //which router within this group the packet needs to go to
  // [한국어] 패킷이 이동해야 할 목표 라우터(grp_RID)를 결정한다.
  // 두 경우: (A) 목적지가 같은 그룹이면 → 목적지 라우터로 직접 이동.
  //          (B) 다른 그룹이면 → inter-group 광학 링크를 보유한 중간 라우터로 이동.
  if (dest_grp_ID == grp_ID) { // [한국어] (A) 같은 그룹: 목적지 라우터가 바로 목표
    grp_RID = int(dest / gP); // [한국어] 목적지 터미널이 연결된 라우터 ID (전역)
  } else {
    // [한국어] (B) 다른 그룹: inter-group 링크 인덱스(grp_output) 결정 후 해당 라우터 ID 계산.
    // 현재 그룹(grp_ID)에서 목적지 그룹(dest_grp_ID)으로 향하는 광학 채널은
    // 그룹 ID 대소 관계로 인덱스 보정이 필요하다 (자기 그룹 ID는 채널 목록에서 제외).
    if (grp_ID > dest_grp_ID) {  // [한국어] 현재 그룹 ID가 더 크면: 자기 자신 제외 없이 그대로
      grp_output = dest_grp_ID;  // [한국어] inter-group 채널 인덱스 = 목적지 그룹 ID
    } else {
      grp_output = dest_grp_ID - 1; // [한국어] 목적지 그룹 ID가 더 크면: 자기 그룹(grp_ID) 건너뜀 → -1 보정
    }
    // [한국어] grp_output / gP: 해당 광학 채널을 담당하는 그룹 내 라우터 인덱스.
    // + grp_ID * _grp_num_routers: 현재 그룹의 시작 라우터 ID를 더해 전역 라우터 ID로 변환.
    grp_RID = int(grp_output / gP) + grp_ID * _grp_num_routers;
    // group_dest = grp_RID * gP; // [한국어] (주석처리) 목적지 그룹의 첫 터미널 — 현재 미사용
  }

  //At the last hop
  // [한국어] 경우 1: 최종 홉 — dest 터미널이 현재 라우터(rID)에 직접 연결됨.
  // rID*gP ~ (rID+1)*gP-1 범위에 dest가 있으면 현재 라우터가 목적지 라우터.
  if (dest >= rID * gP && dest < (rID + 1) * gP) {
    out_port = dest % gP; // [한국어] dest의 라우터 내 로컬 터미널 포트 번호 (0 ~ gP-1)
  } else if (grp_RID == rID) {
    // [한국어] 경우 2: inter-group 광학 홉 — 현재 라우터가 목적지 방향 광학 링크 보유 라우터.
    // grp_RID == rID이고 최종 홉이 아니면 inter-group 광학 링크를 통해 다른 그룹으로 이동.
    // 포트 번호: gP(터미널 오프셋) + (gA-1)(intra 포트 수) + grp_output%(gP)(광학 채널 내 인덱스).
    out_port = gP + (gA - 1) + grp_output % (gP);
  } else {
    // [한국어] 경우 3: intra-group 이동 — grp_RID 방향으로 그룹 내 링크를 통해 이동.
    assert(grp_RID != -1); // [한국어] grp_RID가 여전히 -1이면 로직 오류 — 방어적 검사

    // [한국어] intra-group 포트 번호 계산: 각 라우터는 자기 자신을 제외한 gA-1개 라우터와 연결.
    // 포트 번호는 [gP, gP+(gA-2)] 범위이며, 방향(왼쪽/오른쪽)에 따라 다르게 계산된다.
    if (rID < grp_RID) {
      // [한국어] 목표 라우터가 자신보다 ID가 크면(오른쪽): 포트 = (목표의 그룹 내 위치) - 1 + gP.
      // -1 보정: 자기 자신(rID)보다 오른쪽 라우터는 연결 목록에서 인덱스가 1 낮아짐.
      out_port = (grp_RID % _grp_num_routers) - 1 + gP;
    } else {
      // [한국어] 목표 라우터가 자신보다 ID가 작으면(왼쪽): 포트 = (목표의 그룹 내 위치) + gP.
      out_port = (grp_RID % _grp_num_routers) + gP;
    }
  }

  assert(out_port != -1); // [한국어] 포트가 여전히 -1이면 위 3가지 경우 모두 해당 없는 오류 — 방어적 검사
  return out_port; // [한국어] 계산된 출력 포트 번호 반환
}


/*
 * [한국어]
 * DragonFlyNew::DragonFlyNew - Dragonfly 네트워크 생성자
 *
 * @config: 시뮬레이터 설정 객체 (k, n 파라미터 포함).
 * @name:   이 네트워크 인스턴스의 식별 이름 (디버그/로그 출력용).
 * @return: 없음 (생성자).
 *
 * 초기화 순서:
 *   1) Network(config, name): 부모 클래스 생성자 — 기본 자료구조 초기화.
 *   2) _ComputeSize(config): _p, _a, _g, _k, _nodes, _channels, gP, gA, gG 결정.
 *   3) _Alloc(): Network 부모 클래스의 메모리 할당 — _routers[], _chan[] 등 배열 할당.
 *   4) _BuildNet(config): 라우터 객체 생성 및 채널 연결.
 * 완료 후 네트워크가 완전히 구성되어 사이클 시뮬레이션 가능 상태가 된다.
 * 실행 컨텍스트: 시뮬레이터 초기화 단계 (사이클 루프 시작 전, 단일 스레드).
 *
 * 호출 체인:
 *   NetworkFactory::NewNetwork() → [이 생성자] → _ComputeSize() / _Alloc() / _BuildNet()
 */
DragonFlyNew::DragonFlyNew( const Configuration &config, const string & name ) :
  Network( config, name ) // [한국어] 부모 클래스 Network 생성자 호출 — 기본 자료구조(라우터/채널 배열 등) 초기화
{
  _ComputeSize( config ); // [한국어] 설정 파일에서 k, n을 읽어 전체 네트워크 크기 파라미터 계산
  _Alloc( );              // [한국어] 계산된 크기(_nodes, _num_of_switch, _channels)로 배열 메모리 동적 할당
  _BuildNet( config );    // [한국어] 라우터 객체 생성 및 채널 연결로 실제 네트워크 위상 구성
}

/*
 * [한국어]
 * DragonFlyNew::_ComputeSize - 설정 파일 파라미터로 네트워크 크기 변수 계산
 *
 * @config: 시뮬레이터 설정 객체 — "k"(라우터당 터미널 수), "n"(차원 수) 읽기.
 * @return: 없음 (클래스 멤버 필드 및 전역 변수 설정).
 *
 * 계산 흐름:
 *   _p = config["k"]          → 라우터당 터미널 포트 수
 *   _n = config["n"]          → 차원 수 (반드시 1이어야 함)
 *   _k = _p + _p + (2*_p-1)  → 전체 radix (n=1 경우)
 *   _a = 2*_p                 → 그룹당 라우터 수 (n=1 경우)
 *   _g = _a*_p + 1            → 총 그룹 수
 *   _nodes = _a*_p*_g         → 총 터미널 노드 수
 *   _num_of_switch = _nodes/_p → 총 라우터(스위치) 수
 *   _channels = _num_of_switch * (_k-_p) → 총 내부 채널 수 (inject/eject 제외)
 *   _size = _num_of_switch    → Network 부모 클래스 필드 (라우터 수)
 *   gG, gP, gA 전역 변수 업데이트 → 라우팅 함수에서 참조
 *
 * 실행 컨텍스트: 생성자에서 단 한 번 호출 (초기화 단계).
 *
 * 호출 체인:
 *   DragonFlyNew() → [이 함수]
 */
void DragonFlyNew::_ComputeSize( const Configuration &config )
{

  // LIMITATION
  //  -- only one dimension between the group
  // _n == # of dimensions within a group
  // _p == # of processors within a router
  // inter-group ports : _p
  // terminal ports : _p
  // intra-group ports : 2*_p - 1
  _p = config.GetInt( "k" ); // [한국어] 라우터당 터미널(프로세서) 포트 수 = 설정 파일의 "k" 값
  _n = config.GetInt( "n" ); // [한국어] 그룹 내 차원 수 = 설정 파일의 "n" 값 (반드시 1이어야 함)


  assert(_n == 1); // [한국어] n>1 차원은 미구현 — 1이 아닐 경우 즉시 프로그램 종료
  // dimension

  // [한국어] 전체 라우터 radix(_k) 계산:
  //   n=1: _k = _p(터미널) + _p(inter-group) + (2*_p-1)(intra-group) = 4*_p - 1
  //   n>1: _k = _p + _p + 2*_p = 4*_p (이 분기는 assert에 의해 도달 불가)
  if (_n == 1)
    _k = _p + _p + 2 * _p - 1; // [한국어] n=1: intra-group 포트가 2*_p-1개 (자기 자신 제외)
  else
    _k = _p + _p + 2 * _p;     // [한국어] n>1: intra-group 포트가 2*_p개 (미구현 분기)


  // FIX...
  // [한국어] Booksim 전역 파라미터 gK, gN 설정 — 다른 모듈에서 k-ary n-fly 네트워크 파라미터로 참조
  gK = _p; gN = _n;

  // with 1 dimension, total of 2p routers per group
  // N = 2p * p * (2p^2 + 1)
  // a = # of routers per group
  //   = 2p (if n = 1)
  //   = p^(n) (if n > 2)
  //  g = # of groups
  //    = a * p + 1
  // N = a * p * g;

  // [한국어] 그룹당 라우터 수(_a) 계산:
  //   n=1: _a = 2*_p (Dragonfly 표준 구성)
  //   n>1: _a = _p^n (일반화된 Dragonfly, 현재 미지원)
  if (_n == 1)
    _a = 2 * _p;          // [한국어] n=1: 그룹당 라우터 수 = 2 * 터미널 포트 수
  else
    _a = powi(_p, _n);    // [한국어] n>1: 그룹당 라우터 수 = _p의 _n제곱 (미구현 분기)

  _g = _a * _p + 1;       // [한국어] 총 그룹 수 = 그룹당 라우터 수 * 터미널 수 + 1
  _nodes = _a * _p * _g;  // [한국어] 총 터미널 노드 수 = 그룹당 라우터 수 * 터미널/라우터 * 그룹 수

  _num_of_switch = _nodes / _p;              // [한국어] 총 라우터(스위치) 수 = 총 노드 수 / 라우터당 터미널 수 = _a * _g
  _channels = _num_of_switch * (_k - _p);   // [한국어] 총 내부 채널 수 = 라우터 수 * (전체radix - 터미널 포트 수)
                                              //   = _num_of_switch * (intra + inter 포트 수)
  _size = _num_of_switch;                    // [한국어] Network 기본 클래스 _size = 라우터(스위치) 수



  // [한국어] 전역 변수 업데이트 — 이후 라우팅 함수(dragonfly_port, min_dragonflynew 등)에서 직접 참조
  gG = _g;   // [한국어] 전역 총 그룹 수 = _g
  gP = _p;   // [한국어] 전역 라우터당 터미널 수 = _p
  gA = _a;   // [한국어] 전역 그룹당 라우터 수 = _a
  _grp_num_routers = gA;               // [한국어] 멤버 변수에도 동일값 저장 (코드 가독성)
  _grp_num_nodes = _grp_num_routers * gP; // [한국어] 그룹당 터미널 노드 수 = 라우터 수 * 터미널/라우터

}

/*
 * [한국어]
 * DragonFlyNew::_BuildNet - 라우터 생성 및 채널 연결로 실제 Dragonfly 네트워크 구조 구축
 *
 * @config: Router::NewRouter()에 전달하는 설정 객체 (라우터 타입, 버퍼 크기 등).
 * @return: 없음 (_routers[], _chan[], _inject[], _eject[] 배열이 완전히 채워짐).
 *
 * 처리 순서 (라우터별):
 *   1) 라우터 이름 생성 ("router_<node_id>")
 *   2) Router::NewRouter()로 라우터 객체 생성 (input/output 포트 수: _k)
 *   3) inject 채널 연결 (_p개): 터미널 → 라우터 입력 (주입 포트)
 *   4) eject 채널 연결 (_p개): 라우터 출력 → 터미널 (추출 포트)
 *   5) intra-group 출력 채널 연결 (2*_p-1개, latency=10): 같은 그룹 내 라우터로 출력
 *   6) inter-group 출력 채널 연결 (_p개, latency=100): 다른 그룹으로 나가는 광학 링크 출력
 *   7) intra-group 입력 채널 연결 (2*_p-1개): 같은 그룹 내 다른 라우터에서 입력
 *   8) inter-group 입력 채널 연결 (_p개): 다른 그룹에서 들어오는 광학 링크 입력
 *
 * 채널 인덱스 계산이 복잡하므로 주의:
 *   - 출력 채널: (2*_p-1+_p)*node + dim*(2*_p-1) + cnt (intra)
 *                (2*_p-1+_p)*node + (2*_p-1) + cnt (inter)
 *   - 입력 채널: 연결 대칭성을 위해 상대 라우터의 출력 채널을 역방향으로 참조
 *
 * n>1이면 에러 메시지 출력 후 exit(-1)로 즉시 종료 (단일 차원만 지원).
 *
 * 호출 체인:
 *   DragonFlyNew() → [이 함수] → Router::NewRouter() / Router::AddInputChannel() / AddOutputChannel()
 */
void DragonFlyNew::_BuildNet( const Configuration &config )
{

  int _output = -1;           // [한국어] 출력 채널 인덱스 (미결정 초기값 -1)
  int _input  = -1;           // [한국어] 입력 채널 인덱스 (미결정 초기값 -1)
  int _dim_ID = -1;           // [한국어] 그룹 내 라우터의 로컬 인덱스 (0 ~ _a-1)
  int _num_ports_per_switch = -1; // [한국어] 라우터당 내부 채널 수 = _k - _p (터미널 제외)
  // int _dim_size=-1;         // [한국어] (주석처리) 차원 크기 — 다차원 지원 시 사용 예정
  int c;                      // [한국어] inject/eject 채널 인덱스 임시 변수

  ostringstream router_name; // [한국어] 라우터 이름 문자열 생성용 스트림 ("router_<id>" 형식)



  // [한국어] 네트워크 파라미터 디버그 출력 — 시뮬레이터 시작 시 콘솔에 표시됨
  cout << " Dragonfly " << endl;                                   // [한국어] 토폴로지 타입 출력
  cout << " p = " << _p << " n = " << _n << endl;                  // [한국어] 터미널 포트 수 / 차원 수
  cout << " each switch - total radix =  " << _k << endl;          // [한국어] 라우터 전체 radix
  cout << " # of switches = " << _num_of_switch << endl;           // [한국어] 총 스위치(라우터) 수
  cout << " # of channels = " << _channels << endl;                // [한국어] 총 내부 채널 수
  cout << " # of nodes ( size of network ) = " << _nodes << endl;  // [한국어] 총 터미널 노드 수
  cout << " # of groups (_g) = " << _g << endl;                    // [한국어] 총 그룹 수
  cout << " # of routers per group (_a) = " << _a << endl;         // [한국어] 그룹당 라우터 수

  // [한국어] 모든 라우터 노드에 대해 순서대로 생성 및 채널 연결
  for ( int node = 0; node < _num_of_switch; ++node ) {
    // ID of the group
    int grp_ID;
    grp_ID = (int)(node / _a); // [한국어] 현재 라우터가 속한 그룹 ID = 라우터 ID / 그룹당 라우터 수
    router_name << "router";   // [한국어] 라우터 이름 시작 "router"

    router_name << "_" << node; // [한국어] 라우터 이름에 ID 추가: "router_<node>"

    // [한국어] 라우터 객체 생성: 이름, ID, 입력 포트 수(_k), 출력 포트 수(_k) 지정
    _routers[node] = Router::NewRouter( config, this, router_name.str( ),
                                        node, _k, _k );
    _timed_modules.push_back(_routers[node]); // [한국어] 타이밍 모듈 목록에 추가 — 사이클마다 Update() 호출됨

    router_name.str(""); // [한국어] 다음 라우터 이름 생성을 위해 스트림 초기화

    // [한국어] inject 채널 연결: 터미널 노드 → 라우터 입력 포트 (_p개)
    // 각 라우터는 _p개의 터미널과 연결되며, 채널 인덱스 c = node*_p + cnt
    for ( int cnt = 0; cnt < _p; ++cnt ) {
      c = _p * node + cnt; // [한국어] inject 채널 인덱스: 라우터 오프셋(_p*node) + 로컬 터미널 번호(cnt)
      _routers[node]->AddInputChannel( _inject[c], _inject_cred[c] ); // [한국어] 주입 채널(터미널→라우터) 연결

    }

    // [한국어] eject 채널 연결: 라우터 출력 포트 → 터미널 노드 (_p개)
    for ( int cnt = 0; cnt < _p; ++cnt ) {
      c = _p * node + cnt; // [한국어] eject 채널 인덱스 (inject와 동일 인덱스 — 단방향 채널 각각 존재)
      _routers[node]->AddOutputChannel( _eject[c], _eject_cred[c] ); // [한국어] 추출 채널(라우터→터미널) 연결

    }

    // add OUPUT channels
    // _k == # of processor per router
    //  need 2*_k routers  --thus,
    //  2_k-1 outputs channels within group
    //  _k-1 outputs for intra-group

    //

    // [한국어] n>1 차원은 미구현 — 해당 경우 에러 출력 후 즉시 종료
    if (_n > 1) { cout << " ERROR: n>1 dimension NOT supported yet... " << endl; exit(-1); }

    //********************************************
    //   connect OUTPUT channels
    //********************************************
    // add intra-group output channel
    // [한국어] intra-group 출력 채널 연결 (같은 그룹 내 다른 라우터들로 출력, 2*_p-1개).
    // 채널 인덱스 공식: (2*_p-1 + _p) * _n * node + (2*_p-1) * dim + cnt
    //   - (2*_p-1 + _p): 라우터당 내부 채널 수 (intra + inter 합계)
    //   - dim: 차원 인덱스 (n=1이므로 항상 0)
    //   - cnt: 그룹 내 채널 번호 (0 ~ 2*_p-2)
    for ( int dim = 0; dim < _n; ++dim ) {         // [한국어] 차원 루프 (n=1이므로 1회만 실행)
      for ( int cnt = 0; cnt < (2 * _p - 1); ++cnt ) { // [한국어] intra-group 채널 개수(2*_p-1)만큼 반복
        _output = (2 * _p - 1 + _p) * _n * node + (2 * _p - 1) * dim + cnt;
        // [한국어] 위 인덱스 계산: 이 라우터(node)의 intra-group 출력 채널 번호

        _routers[node]->AddOutputChannel( _chan[_output], _chan_cred[_output] );
        // [한국어] intra-group 출력 채널과 크레딧 채널을 라우터의 출력 포트에 등록

#ifdef DRAGON_LATENCY
        _chan[_output]->SetLatency(10);       // [한국어] intra-group 채널 레이턴시 = 10 사이클 (구리 케이블 모델링)
        _chan_cred[_output]->SetLatency(10);  // [한국어] 크레딧 채널도 동일 레이턴시 설정 (역방향 흐름 제어)
#endif
      }
    }

    // add inter-group output channel
    // [한국어] inter-group 출력 채널 연결 (다른 그룹으로 나가는 광학 링크, _p개).
    // 채널 인덱스 공식: (2*_p-1+_p)*node + (2*_p-1) + cnt
    //   intra-group 채널 블록(2*_p-1) 바로 다음에 inter-group 채널(_p개) 배치.
    for ( int cnt = 0; cnt < _p; ++cnt ) { // [한국어] inter-group 채널 개수(_p)만큼 반복
      _output = (2 * _p - 1 + _p) * node + (2 * _p - 1) + cnt;
      // [한국어] 위 인덱스 계산: intra 블록 끝 + cnt = 이 라우터의 cnt번째 inter-group 출력 채널

      //      _chan[_output].global = true;  // [한국어] (주석처리) 광학 링크 플래그 — 미사용
      _routers[node]->AddOutputChannel( _chan[_output], _chan_cred[_output] );
      // [한국어] inter-group 출력 채널과 크레딧 채널을 라우터의 출력 포트에 등록
#ifdef DRAGON_LATENCY
      _chan[_output]->SetLatency(100);      // [한국어] inter-group 광학 링크 레이턴시 = 100 사이클 (intra 대비 10배)
      _chan_cred[_output]->SetLatency(100); // [한국어] 크레딧 채널도 동일 레이턴시 (흐름 제어 지연 포함)
#endif
    }


    //********************************************
    //   connect INPUT channels
    //********************************************
    // # of non-local nodes
    // [한국어] 라우터당 내부 채널 수(intra + inter, 터미널 제외) 계산
    _num_ports_per_switch = (_k - _p); // [한국어] = (2*_p-1) + _p = 3*_p-1 (intra + inter)


    // intra-group GROUP channels
    // [한국어] intra-group 입력 채널 연결: 같은 그룹 내 다른 라우터의 출력을 입력으로 연결.
    // 각 라우터는 자기 자신을 제외한 그룹 내 나머지 라우터들(2*_p-1개)로부터 수신한다.
    // _dim_ID: 그룹 내 현재 라우터의 로컬 인덱스 (0 ~ _a-1)
    for ( int dim = 0; dim < _n; ++dim ) { // [한국어] 차원 루프 (n=1이므로 1회)

      // _dim_size = powi(_k,dim);  // [한국어] (주석처리) 차원 크기 — 다차원 지원용

      _dim_ID = ((int)(node / (powi(_p, dim)))); // [한국어] 다차원 계산용 중간값 (n=1에서 실질적으로 아래 줄에서 덮어씀)



      // NODE ID withing group
      // [한국어] n=1 기준 그룹 내 라우터 인덱스: node % _a
      _dim_ID = node % _a; // [한국어] 현재 라우터의 그룹 내 로컬 인덱스 (0 ~ 2*_p-1)




      // [한국어] 같은 그룹 내 2*_p-1개 라우터로부터 수신하는 intra-group 입력 채널 연결.
      // cnt: 입력 채널 번호 (0 ~ 2*_p-2), 각 cnt가 특정 상대 라우터를 가리킴.
      for ( int cnt = 0; cnt < (2 * _p - 1); ++cnt ) {

        // [한국어] 입력 채널 인덱스 계산 — 상대 라우터의 출력 채널을 역방향으로 참조.
        // cnt < _dim_ID: 자신보다 작은 ID의 라우터에서 오는 채널 (자신이 오른쪽 이웃)
        if ( cnt < _dim_ID) {
          // [한국어] 상대 라우터(index: _dim_ID - (cnt 역방향))에서 현재 라우터 방향 출력 채널.
          // 공식: grp_ID * _num_ports_per_switch * _a  → 그룹 시작 채널 오프셋
          //        - (_dim_ID - cnt) * _num_ports_per_switch  → cnt번째 상대 라우터 오프셋 보정
          //        + _dim_ID * _num_ports_per_switch           → 현재 라우터 출력 블록
          //        + (_dim_ID - 1)                             → intra 채널 내 위치
          _input =    grp_ID  * _num_ports_per_switch * _a -
            (_dim_ID - cnt) *  _num_ports_per_switch +
            _dim_ID * _num_ports_per_switch +
            (_dim_ID - 1);
        }
        else {
          // [한국어] cnt >= _dim_ID: 자신보다 크거나 같은 ID의 라우터에서 오는 채널.
          // 공식: grp_ID * _num_ports_per_switch * _a  → 그룹 시작 오프셋
          //        + _dim_ID * _num_ports_per_switch    → 현재 라우터 출력 블록 시작
          //        + (cnt - _dim_ID + 1) * _num_ports_per_switch → cnt번째 상대 블록
          //        + _dim_ID                             → intra 채널 내 위치
          _input =  grp_ID * _num_ports_per_switch * _a +
            _dim_ID * _num_ports_per_switch +
            (cnt - _dim_ID + 1) * _num_ports_per_switch +
            _dim_ID;

        }

        // [한국어] 인덱스 범위 검사 — 음수이면 로직 오류
        if (_input < 0) {
          cout << " ERROR: _input less than zero " << endl; // [한국어] 인덱스 음수 오류 출력
          exit(-1); // [한국어] 즉시 종료
        }


        _routers[node]->AddInputChannel( _chan[_input], _chan_cred[_input] );
        // [한국어] 계산된 intra-group 입력 채널과 크레딧 채널을 이 라우터의 입력 포트에 연결
      }
    }


    // add INPUT channels -- "optical" channels connecting the groups
    // int _grp_num_routers;  // [한국어] (주석처리) 이미 멤버 변수로 선언됨
    int grp_output; // [한국어] inter-group 채널 인덱스 임시 변수
    // int grp_ID2;   // [한국어] (주석처리) 미사용

    // [한국어] inter-group 입력 채널 연결: 다른 그룹에서 들어오는 광학 링크 입력 (_p개).
    // 현재 라우터(_dim_ID)는 그룹 외부 _p개 그룹과 각각 광학 링크로 연결된다.
    for ( int cnt = 0; cnt < _p; ++cnt ) {
      //	   _dim_ID
      // [한국어] 이 라우터가 담당하는 inter-group 광학 채널의 전역 채널 인덱스.
      // _dim_ID * _p + cnt: 그룹 내 라우터 위치와 cnt에 따라 연결되는 외부 그룹 결정.
      grp_output = _dim_ID * _p + cnt;

      // _grp_num_routers = powi(_k, _n-1); // [한국어] (주석처리) n>1 지원용 계산
      // grp_ID2 = (int) ((grp_ID - 1) / (_k - 1)); // [한국어] (주석처리) 미사용

      // [한국어] 연결되는 외부 그룹(grp_output)과 현재 그룹(grp_ID)의 대소 관계로
      // 입력 채널 인덱스를 결정한다. grp_output이 가리키는 그룹에서 현재 그룹 방향으로
      // 나가는 출력 채널을 역방향으로 참조한다.
      if ( grp_ID > grp_output) {
        // [한국어] 현재 그룹 ID가 더 큰 경우: 외부 그룹(grp_output)의 inter-group 출력 채널 중
        // 현재 그룹(grp_ID) 방향의 채널을 참조.
        // 공식 분해:
        //   (grp_output) * _num_ports_per_switch * _a: 외부 그룹 첫 번째 라우터의 채널 시작
        //   (_num_ports_per_switch - _p) * (int)((grp_ID-1)/_p): inter-group 담당 라우터 선택
        //   (_num_ports_per_switch - _p): intra 블록 건너뛰고 inter 블록 시작
        //   grp_ID - 1: inter-group 채널 내 위치 (자기 그룹 ID < grp_ID이므로 -1 보정 불필요 → grp_ID-1)
        _input = (grp_output) * _num_ports_per_switch * _a    +        // 외부 그룹 채널 시작
          (_num_ports_per_switch - _p) * (int)((grp_ID - 1) / _p) +    // 담당 라우터 오프셋
          (_num_ports_per_switch - _p) +                                // inter 블록 시작 오프셋
          grp_ID - 1;   // [한국어] 현재 그룹 ID에 해당하는 inter-group 채널 내 위치 (0-indexed, -1 보정)
      } else {
        // [한국어] 외부 그룹 ID(grp_output)가 더 큰 경우: 외부 그룹(grp_output+1)에서
        // 현재 그룹(grp_ID) 방향의 채널을 참조.
        // grp_output+1: 자기 그룹 ID(grp_ID)를 건너뛰어야 하므로 +1 보정.
        _input = (grp_output + 1) * _num_ports_per_switch * _a    +
          (_num_ports_per_switch - _p) * (int)((grp_ID) / _p) +  // 담당 라우터 오프셋
          (_num_ports_per_switch - _p) +                          // inter 블록 시작 오프셋
          grp_ID;   // [한국어] 현재 그룹 ID에 해당하는 inter-group 채널 내 위치 (자기 건너뜀 없으므로 그대로)
      }

      _routers[node]->AddInputChannel( _chan[_input], _chan_cred[_input] );
      // [한국어] 계산된 inter-group(광학 링크) 입력 채널을 이 라우터의 입력 포트에 연결
    }

  }

  cout << "Done links" << endl; // [한국어] 모든 라우터/채널 연결 완료 알림 출력
}


/*
 * [한국어]
 * DragonFlyNew::GetN - 그룹 내 차원 수 반환
 *
 * @return: _n 값 (현재 항상 1).
 *
 * 외부 모듈에서 네트워크 파라미터를 조회할 때 사용된다.
 * Booksim 통계/로그 출력에서 호출될 수 있다.
 *
 * 호출 체인:
 *   (외부 조회 코드) → [이 함수]
 */
int DragonFlyNew::GetN( ) const
{
  return _n; // [한국어] 그룹 내 차원 수 반환 (현재 항상 1)
}

/*
 * [한국어]
 * DragonFlyNew::GetK - 라우터의 전체 radix(총 포트 수) 반환
 *
 * @return: _k 값 (n=1일 때 4*_p-1).
 *
 * 외부 모듈에서 네트워크 파라미터를 조회할 때 사용된다.
 *
 * 호출 체인:
 *   (외부 조회 코드) → [이 함수]
 */
int DragonFlyNew::GetK( ) const
{
  return _k; // [한국어] 라우터 전체 radix 반환 (n=1 기준: 4*_p-1)
}

/*
 * [한국어]
 * DragonFlyNew::InsertRandomFaults - 랜덤 링크 결함 주입 (현재 미구현)
 *
 * @config: 결함 주입 관련 설정 (현재 미사용).
 * @return: 없음.
 *
 * 함수 본체가 비어 있어 실제로 아무 동작도 하지 않는다.
 * 결함 허용(fault-tolerant) 라우팅 연구를 위한 확장 포인트이나,
 * 현재 Dragonfly 구현에서는 지원하지 않는다.
 *
 * 호출 체인:
 *   (외부 결함 주입 요청) → [이 함수] (아무 동작 없음)
 */
void DragonFlyNew::InsertRandomFaults( const Configuration &config )
{
  // [한국어] 미구현 — 함수 본체가 의도적으로 비어 있음 (결함 주입 기능 미지원)
}

/*
 * [한국어]
 * DragonFlyNew::Capacity - 네트워크 용량 지표 반환
 *
 * @return: (double)_k / 8.0 — radix를 8로 나눈 정규화 용량 값.
 *
 * Booksim 내부 통계 계산 및 네트워크 비교 분석에서 사용된다.
 * 물리적 의미: 라우터당 포트 수를 8 단위로 정규화한 상대적 용량 지표.
 *
 * 호출 체인:
 *   (Booksim 통계 모듈) → [이 함수]
 */
double DragonFlyNew::Capacity( ) const
{
  return (double)_k / 8.0; // [한국어] 라우터 전체 radix를 8로 나눈 정규화 용량값 반환
}

/*
 * [한국어]
 * DragonFlyNew::RegisterRoutingFunctions - Dragonfly 라우팅 함수를 전역 맵에 등록
 *
 * @return: 없음 (전역 gRoutingFunctionMap에 함수 포인터 2개 추가).
 *
 * 정적(static) 멤버 함수로, 인스턴스 없이 초기화 단계에서 호출된다.
 * 등록되는 라우팅 함수:
 *   - "min_dragonflynew": 최소 경로 라우팅 함수 포인터
 *   - "ugal_dragonflynew": UGAL 적응형 라우팅 함수 포인터
 * 이후 Booksim이 설정 파일의 "routing_function" 값으로 실제 함수를 선택한다.
 *
 * 호출 체인:
 *   InitializeRoutingMap() / NetworkFactory → [이 함수] → gRoutingFunctionMap 갱신
 */
void DragonFlyNew::RegisterRoutingFunctions() {

  gRoutingFunctionMap["min_dragonflynew"] = &min_dragonflynew;
  // [한국어] "min_dragonflynew" 키로 최소 경로 라우팅 함수 포인터를 전역 맵에 등록
  gRoutingFunctionMap["ugal_dragonflynew"] = &ugal_dragonflynew;
  // [한국어] "ugal_dragonflynew" 키로 UGAL 적응형 라우팅 함수 포인터를 전역 맵에 등록
}


/*
 * [한국어]
 * min_dragonflynew - Dragonfly 최소 경로 라우팅 함수 (Minimal Routing)
 *
 * @r:          현재 패킷이 위치한 라우터 객체 (GetID(), FullName() 사용).
 * @f:          현재 처리 중인 Flit 객체 (ph, src, dest, watch 필드 읽기/쓰기).
 * @in_channel: 패킷이 도착한 입력 채널 번호.
 *              < gP이면 터미널 주입 포트 (소스 라우터에서의 첫 홉).
 * @outputs:    선택한 출력 포트와 VC 범위를 저장하는 결과 집합.
 * @inject:     true이면 주입 단계 — 랜덤 VC 선택 후 즉시 반환.
 * @return:     없음 (outputs 객체에 결과 저장).
 *
 * 동작 원리:
 *   항상 최단 경로(dragonfly_port()가 계산한 포트)를 사용한다.
 *   데드락 방지를 위해 dateline 기법으로 VC를 관리:
 *     f->ph = 0: inter-group 광학 링크 통과 전 (intra-group 단계)
 *     f->ph = 1: inter-group 광학 링크 통과 후 (목적지 그룹 intra-group 단계)
 *   광학 링크 포트(out_port >= gP + (gA-1))를 선택할 때 ph를 0→1로 업그레이드.
 *   out_vc = f->ph: VC 번호 = 페이즈 번호 (0 또는 1).
 *   데드락 자유 보장: 높은 VC(1)에서 낮은 VC(0)로의 역방향 전환이 없으므로
 *   사이클 없는 의존성 그래프(DAG)가 보장된다.
 *
 * 실행 컨텍스트: Booksim 사이클 루프 내 Router::Route() 호출 체인에서 실행.
 *               단일 스레드, 사이클마다 패킷당 한 번 호출.
 *
 * 호출 체인:
 *   Router::Route() → gRoutingFunctionMap["min_dragonflynew"] → [이 함수]
 *                         → dragonfly_port()
 */
void min_dragonflynew( const Router *r, const Flit *f, int in_channel,
                       OutputSet *outputs, bool inject )
{
  outputs->Clear( ); // [한국어] 이전 라우팅 결정 초기화 — 새로운 포트/VC 선택 준비

  // [한국어] 주입(inject) 단계: 네트워크에 처음 진입하는 경우
  if(inject) {
    int inject_vc = RandomInt(gNumVCs - 1); // [한국어] 랜덤으로 VC 선택 (0 ~ gNumVCs-1 중 하나)
    outputs->AddRange(-1, inject_vc, inject_vc); // [한국어] 포트=-1(inject 포트), VC=[inject_vc, inject_vc] 범위 등록
    return; // [한국어] 주입 단계는 여기서 완료 — 이후 라우팅 로직 건너뜀
  }

  int _grp_num_routers = gA; // [한국어] 그룹당 라우터 수 — 전역 gA를 지역 변수로 캐시

  int dest  = f->dest;                        // [한국어] 패킷 최종 목적지 터미널 노드 ID
  int rID   = r->GetID();                     // [한국어] 현재 라우터의 전역 ID
  int grp_ID = int(rID / _grp_num_routers);   // [한국어] 현재 라우터가 속한 그룹 ID
  int debug  = f->watch;                      // [한국어] 디버그 모드 플래그 (1이면 이 패킷을 추적 중)
  int out_port   = -1;                        // [한국어] 선택된 출력 포트 (미결정 초기값)
  int out_vc     = 0;                         // [한국어] 선택된 가상 채널 번호 (초기값 0)
  int dest_grp_ID = -1;                       // [한국어] 목적지 그룹 ID (초기값 -1, 주입 후 아래에서 계산됨)

  // [한국어] 소스 라우터에서 첫 번째 홉 진입 시 (in_channel < gP: 터미널에서 주입된 경우)
  if ( in_channel < gP ) {
    out_vc = 0;  // [한국어] 소스에서 시작: 초기 VC = 0
    f->ph  = 0;  // [한국어] 페이즈 0: inter-group 링크 아직 미통과
    // [한국어] 이미 목적지 그룹에 있으면 (dest_grp_ID == grp_ID가 참이면) 바로 ph=1로 설정.
    // 단, dest_grp_ID는 -1로 초기화되어 있어 이 조건은 항상 false — 사실상 이 분기는 실행 안 됨.
    // (버그 가능성: dest_grp_ID = int(dest/_grp_num_nodes)로 계산해야 올바름)
    if (dest_grp_ID == grp_ID) {
      f->ph = 1; // [한국어] 이미 목적지 그룹 — 광학 링크 불필요, ph를 1로 설정 (실질적으로 미실행)
    }
  }


  out_port = dragonfly_port(rID, f->src, dest); // [한국어] 최소 경로 출력 포트 결정 (핵심 라우팅 계산)

  //optical dateline
  // [한국어] inter-group 광학 링크 포트를 선택한 경우: dateline에 의해 VC를 0→1로 업그레이드.
  // gP + (gA-1): inter-group 포트의 시작 인덱스 (터미널_p개 + intra_gA-1개 이후부터).
  if (out_port >= gP + (gA - 1)) {
    f->ph = 1; // [한국어] 광학 링크 통과 표시 — 이후 목적지 그룹 intra-group은 VC 1 사용
  }

  out_vc = f->ph; // [한국어] 최종 VC 번호 = 현재 페이즈 값 (0: 소스측, 1: 목적지측)
  // [한국어] 디버그 모드에서 라우팅 결정 정보 출력
  if (debug)
    *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
               << "	through output port : " << out_port
               << " out vc: " << out_vc << endl;
  outputs->AddRange( out_port, out_vc, out_vc ); // [한국어] 선택된 포트와 VC를 결과 집합에 등록
}


/*
 * [한국어]
 * ugal_dragonflynew - Dragonfly UGAL 적응형 라우팅 함수
 *                     (Universal Globally-Adaptive Load-balanced)
 *
 * @r:          현재 패킷이 위치한 라우터 객체 (GetID(), GetUsedCredit() 사용).
 * @f:          현재 처리 중인 Flit 객체 (ph, intm, src, dest, watch 필드 읽기/쓰기).
 * @in_channel: 패킷이 도착한 입력 채널 번호 (< gP이면 소스 주입 포트).
 * @outputs:    선택한 출력 포트와 VC를 저장하는 결과 집합.
 * @inject:     true이면 주입 단계 — 랜덤 VC 선택 후 즉시 반환.
 * @return:     없음 (outputs에 결과 저장).
 *
 * UGAL 알고리즘 개요:
 *   소스 라우터에서 최소/비최소 경로 중 하나를 혼잡도 기반으로 선택한다.
 *   비최소 경로는 중간 그룹(intm)을 경유하여 부하를 분산한다 (밸런싱).
 *   결정 기준: min_queue_size * 1 ≤ nonmin_queue_size * 2 + adaptive_threshold(30)
 *             → 조건 충족: 최소 경로 (f->ph=1)
 *             → 조건 불충족: 비최소 경로 (f->ph=0)
 *   adaptive_threshold=30: 양수이므로 최소 경로에 유리하게 편향됨.
 *
 * 3 VC(가상 채널) 구조 (gNumVCs==3 필수):
 *   VC 0 (f->ph=0): 비최소 경로 단계 — 중간 그룹으로 향하는 inter-group 링크
 *   VC 1 (f->ph=1): 최소 경로 단계 — 목적지 그룹으로 향하는 inter-group 링크
 *   VC 2 (f->ph=2): 동일 그룹 내 최소 라우팅 (intra-group 링크)
 *
 * 페이즈 FSM (Finite State Machine):
 *   소스 주입 시 (in_channel < gP):
 *     - 목적지 같은 그룹 → ph=2 (intra only)
 *     - 중간 그룹=현재 그룹 → ph=1 (바로 최소 경로)
 *     - 그 외 → 혼잡도 비교로 ph=0 또는 ph=1 선택
 *   중간 홉 (이미 진행 중):
 *     - ph==0이고 중간 라우터 도달 → ph=1로 전환
 *     - ph==1이고 inter-group 광학 링크 선택 → ph=2로 전환 (dateline)
 *
 * 데드락 자유: VC 번호가 단조증가 (0→1→2) — 역방향 VC 전환 없음.
 *
 * 실행 컨텍스트: Booksim 사이클 루프 내 Router::Route() 호출 체인, 단일 스레드.
 *
 * 호출 체인:
 *   Router::Route() → gRoutingFunctionMap["ugal_dragonflynew"] → [이 함수]
 *                         → dragonfly_port() / r->GetUsedCredit()
 */
//Basic adaptive routign algorithm for the dragonfly
void ugal_dragonflynew( const Router *r, const Flit *f, int in_channel,
                        OutputSet *outputs, bool inject )
{
  //need 3 VCs for deadlock freedom
  // [한국어] UGAL은 3개의 VC가 필요함 — 2개(min_routing)로는 비최소 경로 데드락 방지 불가
  assert(gNumVCs == 3); // [한국어] VC 수 검증 — 3이 아니면 즉시 종료 (설정 오류 방어)
  outputs->Clear( ); // [한국어] 이전 라우팅 결정 초기화
  // [한국어] 주입 단계: 랜덤 VC 선택 후 즉시 반환
  if(inject) {
    int inject_vc = RandomInt(gNumVCs - 1); // [한국어] 0 ~ 2 중 랜덤 VC 선택
    outputs->AddRange(-1, inject_vc, inject_vc); // [한국어] inject 포트(-1)와 선택된 VC 등록
    return; // [한국어] 주입 단계 완료
  }

  //this constant biases the adaptive decision toward minimum routing
  //negative value woudl biases it towards nonminimum routing
  // [한국어] adaptive_threshold: 최소 경로 편향 상수.
  // 양수(30)이면 최소 경로에 유리 (min_queue ≤ 2*nonmin_queue + 30이면 최소 선택).
  // 음수이면 비최소 경로 선호 (트래픽 분산 강화).
  int adaptive_threshold = 30;

  int _grp_num_routers = gA;              // [한국어] 그룹당 라우터 수 캐시
  int _grp_num_nodes = _grp_num_routers * gP; // [한국어] 그룹당 터미널 노드 수 캐시
  int _network_size  = gA * gP * gG;     // [한국어] 전체 네트워크 터미널 노드 수 (중간 노드 랜덤 선택 범위)


  int dest    = f->dest;                       // [한국어] 최종 목적지 터미널 노드 ID
  int rID     = r->GetID();                    // [한국어] 현재 라우터 전역 ID
  int grp_ID  = (int)(rID / _grp_num_routers); // [한국어] 현재 라우터 그룹 ID
  int dest_grp_ID = int(dest / _grp_num_nodes); // [한국어] 목적지 터미널의 그룹 ID

  int debug = f->watch; // [한국어] 디버그 추적 플래그 (1이면 이 패킷 로그 출력)
  int out_port  = -1;   // [한국어] 선택된 출력 포트 (미결정 초기값)
  int out_vc    = 0;    // [한국어] 선택된 VC (초기값 0)
  int min_queue_size;    //, min_hopcnt;   // [한국어] 최소 경로 포트의 사용 크레딧(큐 길이)
  int nonmin_queue_size; //, nonmin_hopcnt; // [한국어] 비최소 경로 포트의 사용 크레딧(큐 길이)
  int intm_grp_ID; // [한국어] 랜덤으로 선택된 중간 그룹 ID
  int intm_rID;    // [한국어] 중간 노드가 위치한 라우터 ID

  if(debug){
    cout << "At router " << rID << endl; // [한국어] 디버그: 현재 라우터 ID 출력
  }
  int min_router_output, nonmin_router_output; // [한국어] 최소/비최소 경로의 출력 포트 번호

  //at the source router, make the adaptive routing decision
  // [한국어] 소스 라우터에서 첫 번째 홉: 최소/비최소 경로 결정 단계.
  // in_channel < gP: 터미널 주입 포트에서 도착 = 이 라우터가 소스 라우터임을 의미.
  if ( in_channel < gP ) {
    //dest are in the same group, only use minimum routing
    // [한국어] 목적지가 같은 그룹이면 비최소 경로 의미 없음 → intra-group 최소 라우팅(ph=2)
    if (dest_grp_ID == grp_ID) {
      f->ph = 2; // [한국어] ph=2: 동일 그룹 내 최소 라우팅 VC 사용
    } else {
      //select a random node
      // [한국어] 목적지가 다른 그룹이면 UGAL 적응형 결정 수행.
      // 1단계: 전체 네트워크에서 랜덤으로 중간 노드(intm) 선택
      f->intm = RandomInt(_network_size - 1); // [한국어] 0 ~ _network_size-1 범위에서 중간 노드 ID 랜덤 선택
      intm_grp_ID = (int)(f->intm / _grp_num_nodes); // [한국어] 중간 노드가 속한 그룹 ID 계산
      if (debug){
        cout << "Intermediate node " << f->intm << " grp id " << intm_grp_ID << endl;
        // [한국어] 디버그: 중간 노드 ID와 그룹 ID 출력
      }

      //random intermediate are in the same group, use minimum routing
      // [한국어] 중간 그룹이 현재 그룹과 같으면 비최소 우회의 의미가 없음 → 최소 경로(ph=1)
      if(grp_ID == intm_grp_ID){
        f->ph = 1; // [한국어] 중간 그룹=현재 그룹 → 비최소 우회 불필요 → 최소 경로 사용
      } else {
        //congestion metrics using queue length, obtained by GetUsedCredit()
        // min_hopcnt = dragonflynew_hopcnt(f->src, f->dest);  // [한국어] (주석처리) 홉 수 기반 비용 — 현재 미사용
        // [한국어] 최소 경로 포트의 혼잡도: 해당 포트에서 사용 중인 크레딧(큐에 쌓인 패킷 수)
        min_router_output = dragonfly_port(rID, f->src, f->dest); // [한국어] 최소 경로 출력 포트 계산
        min_queue_size = max(r->GetUsedCredit(min_router_output), 0); // [한국어] 최소 경로 포트 큐 길이 (음수 방지)


        // nonmin_hopcnt = dragonflynew_hopcnt(f->src, f->intm) +   // [한국어] (주석처리) 홉 수 비용
        //   dragonflynew_hopcnt(f->intm,f->dest);
        // [한국어] 비최소 경로 포트의 혼잡도: 중간 노드(intm) 방향 포트의 큐 길이
        nonmin_router_output = dragonfly_port(rID, f->src, f->intm); // [한국어] 비최소 경로(intm 방향) 출력 포트 계산
        nonmin_queue_size = max(r->GetUsedCredit(nonmin_router_output), 0); // [한국어] 비최소 포트 큐 길이

        //congestion comparison, could use hopcnt instead of 1 and 2
        // [한국어] UGAL 혼잡도 비교:
        //   min_queue_size * 1 ≤ nonmin_queue_size * 2 + adaptive_threshold
        //   → 최소 경로가 충분히 덜 혼잡하거나 비슷하면 최소 경로 선택 (f->ph=1)
        //   → 비최소 경로가 훨씬 덜 혼잡하면 비최소 경로 선택 (f->ph=0)
        // 계수 1과 2: 최소 경로는 1홉 이득, 비최소 경로는 2배 홉 비용으로 정규화.
        if ((1 * min_queue_size) <= (2 * nonmin_queue_size) + adaptive_threshold) {
          if (debug) cout << " MINIMAL routing " << endl; // [한국어] 디버그: 최소 경로 선택 출력
          f->ph = 1; // [한국어] 최소 경로 선택 — 직접 목적지 그룹으로 이동 (VC 1 사용)
        } else {
          f->ph = 0; // [한국어] 비최소 경로 선택 — 중간 그룹(intm) 경유하여 부하 분산 (VC 0 사용)
        }
      }
    }
  }

  //transition from nonminimal phase to minimal
  // [한국어] 비최소 경로(ph=0) 진행 중 중간 라우터(intm_rID)에 도달하면 ph=1로 전환.
  // 이후 홉부터는 목적지 방향 최소 경로를 따른다.
  if(f->ph == 0){
    intm_rID = (int)(f->intm / gP); // [한국어] 중간 노드가 위치한 라우터 ID 계산 (노드ID / 라우터당 터미널수)
    if( rID == intm_rID){           // [한국어] 현재 라우터가 중간 라우터이면
      f->ph = 1; // [한국어] 비최소 단계 완료 → 최소 단계로 전환 (VC 0→1 업그레이드)
    }
  }

  //port assignement based on the phase
  // [한국어] 페이즈에 따른 출력 포트 결정:
  //   ph=0: 중간 그룹(intm) 방향으로 최소 경로 포트
  //   ph=1: 목적지(dest) 방향으로 최소 경로 포트
  //   ph=2: 목적지(dest) 방향으로 최소 경로 포트 (동일 그룹 내)
  if(f->ph == 0){
    out_port = dragonfly_port(rID, f->src, f->intm); // [한국어] ph=0: 비최소 경로 — 중간 노드 방향 포트
  } else if(f->ph == 1){
    out_port = dragonfly_port(rID, f->src, f->dest); // [한국어] ph=1: 최소 경로 — 목적지 방향 포트
  } else if(f->ph == 2){
    out_port = dragonfly_port(rID, f->src, f->dest); // [한국어] ph=2: 동일 그룹 최소 경로 — 목적지 방향 포트
  } else {
    assert(false); // [한국어] 정의되지 않은 ph 값 — 로직 오류, 즉시 종료
  }

  //optical dateline
  // [한국어] ph=1(최소 경로)이고 inter-group 광학 링크 포트를 선택한 경우:
  // dateline 기법으로 ph를 1→2로 업그레이드 (VC 1→2).
  // 목적지 그룹에 도착한 후 intra-group 이동 시 VC 2를 사용하여 데드락 방지.
  if (f->ph == 1 && out_port >= gP + (gA - 1)) {
    f->ph = 2; // [한국어] inter-group 광학 링크 통과 → 목적지 그룹 내 라우팅은 VC 2 사용
  }

  //vc assignemnt based on phase
  // [한국어] 최종 VC 번호 = 현재 페이즈 (0, 1, 또는 2)
  out_vc = f->ph; // [한국어] ph와 VC 번호 일치: ph=0→VC0, ph=1→VC1, ph=2→VC2

  outputs->AddRange( out_port, out_vc, out_vc ); // [한국어] 선택된 포트와 VC를 결과 집합에 등록
}
