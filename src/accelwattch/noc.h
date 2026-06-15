/*****************************************************************************
 *                                McPAT
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
/********************************************************************
 *      Modified by:
 ** Jingwen Leng, Univeristy of Texas, Austin                   * Syed Gilani,
 *University of Wisconsin–Madison                * Tayler Hetherington,
 *University of British Columbia         * Ahmed ElTantawy, University of
 *British Columbia             *
 ********************************************************************/

/*
 * [한국어 설명] AccelWattch NoC(Network-on-Chip) 전력 모델 헤더 (noc.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPU 내부의 네트워크온칩(NoC) 전력 소비를 모델링하는 `NoC` 클래스를 선언한다.
 * GPGPU-Sim에서 NoC는 SM(Streaming Multiprocessor)과 L2 캐시 사이의 인터커넥트 패브릭을
 * 나타내며, AccelWattch(MICRO 2021)의 전력 모델에서 이 구조물의 동적/누설 전력과 면적을 추정한다.
 * NoC는 라우터 기반(type=1)과 버스 기반(type=0) 두 가지 토폴로지를 지원하며,
 * 각각 MCPAT_Router와 interconnect 객체를 통해 구성 요소별 전력을 분리 추적한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델 계층에서 interconnect(배선 전력)와 MCPAT_Router(라우터 전력)의
 * 상위 조합 객체 역할을 한다. GPGPU-Sim의 gpu-sim.cc가 AccelWattch를 호출할 때 NoC 전력이
 * 전체 칩 전력 합산에 포함된다.
 * 호출 체인: gpu-sim.cc → accelwattch (전력 모델 진입점) → NoC::computeEnergy() → MCPAT_Router / interconnect
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, GPGPU-Sim의 전력 계산 단계.
 *
 * === 타 모듈과의 연결 ===
 * 의존: XML_Parse.h (ParseXML — GPU 설정 파라미터 소스), array.h (MCPAT_Router 내부),
 *       basic_components.h (Component 기반 클래스, powerDef, statsDef),
 *       cacti/router.h (MCPAT_Router — 크로스바/중재기/버퍼 전력),
 *       interconnect.h (배선 링크 전력), logic.h (NoCParam 구조체).
 * 의존받음: noc.cc (구현), AccelWattch 전력 집계 모듈 (NoC::computeEnergy 결과를 참조).
 * 데이터 흐름: XML->sys.NoC[ithNoC] → set_noc_param() → nocdynp →
 *              init_router()/init_link_bus() → computeEnergy() → power/rt_power 필드.
 *
 * === 주요 함수/구조체 요약 ===
 * NoC()             : 생성자 — XML에서 파라미터를 읽어 라우터 또는 링크/버스를 초기화
 * set_noc_param()   : XML 필드를 nocdynp 구조체로 파싱 (clockRate MHz→Hz 변환 포함)
 * init_router()     : MCPAT_Router 객체 생성, longer_channel 누설 보정 적용
 * init_link_bus()   : 링크 길이를 계산하고 interconnect 객체를 생성하여 링크 전력 모델 구성
 * computeEnergy()   : TDP/Runtime 모드로 전력 계산 — GPGPU-Sim 카운터를 기반으로 실시간 전력 산출
 * displayEnergy()   : 계산된 전력/면적 결과를 표준 출력으로 출력 (라우터/링크 서브컴포넌트 포함)
 */

#ifndef NOC_H_
#define NOC_H_
#include "XML_Parse.h"         // [한국어] ParseXML — GPU/시스템 설정 XML 파싱 결과 접근 (sys.NoC[] 배열)
#include "array.h"             // [한국어] 배열/SRAM 모델 — MCPAT_Router 내부 버퍼 모델에 사용
#include "basic_components.h"  // [한국어] Component 기반 클래스 (power/area/rt_power), powerDef, statsDef
#include "cacti/parameter.h"   // [한국어] InputParameter, TechnologyParameter (g_tp) — 기술 파라미터
#include "cacti/router.h"      // [한국어] MCPAT_Router — 크로스바/중재기/가상 채널 버퍼 전력 모델
#include "interconnect.h"      // [한국어] interconnect — 온칩 배선 링크 전력/지연 모델
#include "logic.h"             // [한국어] NoCParam 구조체 선언 (flit_size, port 수, duty_cycle 등)

class NoC : public Component {
 public:
  ParseXML *XML;
  /* [한국어] 전체 시스템/GPU 설정 XML 파싱 객체 포인터.
   * 설정자: NoC 생성자 인수로 전달받아 저장.
   * 읽는 자: set_noc_param()이 XML->sys.NoC[ithNoC] 필드를 파싱; computeEnergy()가
   *          XML->sys.NoC[ithNoC].total_accesses를 읽어 런타임 전력을 계산.
   * 값 범위: 유효한 ParseXML 포인터 (NULL 불가). gpgpu-sim-study 전역 XML 구조체.
   * 동기화: AccelWattch 호출은 GPGPU-Sim 시뮬레이션 루프의 단일 스레드 컨텍스트에서만 발생. */

  int ithNoC;
  /* [한국어] 이 NoC 객체가 참조하는 XML sys.NoC[] 배열의 인덱스.
   * 설정자: NoC 생성자 인수 ithNoC_로 초기화.
   * 읽는 자: set_noc_param()이 XML->sys.NoC[ithNoC]를 인덱싱할 때, computeEnergy()가
   *          XML->sys.NoC[ithNoC].total_accesses를 읽을 때 사용.
   * 값 범위: 0 이상 정수, XML이 정의하는 NoC 수보다 작아야 함.
   * 동기화: 생성 후 변경되지 않음 — 읽기 전용 사용. */

  InputParameter interface_ip;
  /* [한국어] CACTI 기술 파라미터 인터페이스 복사본.
   * 설정자: NoC 생성자에서 *interface_ip_를 복사; init_link_bus()가 throughput/latency를 추가 설정.
   *         embedded 시스템 여부에 따라 wt, wire_is_mat_type, wire_os_mat_type이 조정됨.
   * 읽는 자: init_interface(&interface_ip)로 전역 g_tp를 초기화하는 데 사용.
   * 값 범위: InputParameter 구조체 — 기술 노드, 온도, 배선 유형, 타이밍 제약 포함.
   * 동기화: 단일 스레드 초기화 단계에서만 수정됨. */

  double link_len;
  /* [한국어] 노드 간 링크 길이 [m].
   * 설정자: init_link_bus()에서 외부 link_len_ 인수를 노드 수로 나누어 계산.
   *         link_len /= (horizontal_nodes + vertical_nodes) / 2 → 평균 노드 간 거리.
   *         total_nodes > 1이면 추가로 /2 (이웃 노드와 링크를 공유하기 때문).
   * 읽는 자: init_link_bus()가 interconnect 생성자에 전달.
   * 값 범위: 양수 [m]. total_nodes=1이면 외부에서 전달된 link_len_ 그대로 사용.
   * 동기화: 초기화 후 변경 없음. */

  double executionTime;
  /* [한국어] 전체 커널 실행 시간 [s] — 런타임 전력을 시간으로 나누어 평균 전력[W]을 산출할 때 사용.
   * 설정자: set_noc_param()에서 XML->sys.total_cycles / (target_core_clockrate * 1e6)으로 계산.
   * 읽는 자: displayEnergy()가 rt_power.readOp.dynamic / executionTime으로 평균 전력 출력.
   * 값 범위: 양수 [s]. total_cycles=0이면 0이 될 수 있어 나누기 오류 위험.
   * 동기화: 단일 스레드 초기화에서 설정, 이후 읽기 전용. */

  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  /* [한국어] 소켓 계수 및 배치 오버헤드 계수.
   * scktRatio: g_tp.sckt_co_eff — 소켓/패키징 전력 오버헤드 (통상 1.0 이상).
   *            설정자: 생성자에서 g_tp.sckt_co_eff로 초기화.
   *            읽는 자: 현재 NoC 클래스 내에서 직접 참조되지 않지만, 하위 모듈(interconnect 등)이 내부 적용.
   * chip_PR_overhead: 칩 배치 라우팅 오버헤드 계수 (현재 NoC에서 직접 사용 안 함).
   * macro_PR_overhead: 매크로 블록 배치 라우팅 오버헤드 (현재 NoC에서 직접 사용 안 함).
   * 값 범위: 1.0 이상 실수.
   * 동기화: 초기화 후 읽기 전용. */

  MCPAT_Router *router;
  /* [한국어] NoC 라우터 전력 모델 객체 포인터 (크로스바 + 중재기 + 가상 채널 버퍼 포함).
   * 설정자: init_router()에서 new MCPAT_Router(...)로 생성; 생성자에서 0으로 초기화.
   * 읽는 자: computeEnergy()가 router->power, router->rt_power 등을 누적;
   *          displayEnergy()가 라우터 서브컴포넌트 전력을 출력.
   * 값 범위: 유효한 MCPAT_Router 포인터 또는 0 (router_exist=false일 때).
   * 동기화: 단일 스레드 초기화에서 생성, 소멸자에서 해제. */

  interconnect *link_bus;
  /* [한국어] NoC 링크/버스 배선 전력 모델 객체 포인터.
   * 설정자: init_link_bus()에서 new interconnect(...)로 생성; 생성자에서 0으로 초기화.
   * 읽는 자: computeEnergy()가 link_bus->power를 pppm_t로 스케일하여 링크 전력 누적;
   *          computeEnergy() 런타임 모드에서 link_bus->rt_power를 total_accesses로 스케일.
   * 값 범위: 유효한 interconnect 포인터 또는 0 (link_bus_exist=false일 때).
   * 동기화: 단일 스레드 초기화에서 생성, 소멸자에서 해제. */

  NoCParam nocdynp;
  /* [한국어] NoC 동적 파라미터 구조체 — XML에서 파싱된 설정값의 집합.
   * 설정자: set_noc_param()이 XML->sys.NoC[ithNoC] 필드를 읽어 채움.
   * 읽는 자: init_router(), init_link_bus(), computeEnergy(), displayEnergy() 등
   *          모든 주요 함수가 이 구조체의 필드를 참조.
   * 주요 필드:
   *   type: 0=버스, 1=라우터 기반 NoC
   *   clockRate: NoC 클럭 [Hz] (set_noc_param에서 MHz→Hz 변환)
   *   flit_size: 플릿(flit) 비트 폭
   *   input_ports/output_ports: 라우터 입출력 포트 수
   *   virtual_channel_per_port: 포트당 가상 채널 수
   *   horizontal_nodes/vertical_nodes: NoC 메시 토폴로지 차원
   *   duty_cycle: 활성 사이클 비율 (0.0~1.0)
   *   total_nodes: horizontal_nodes × vertical_nodes
   *   global_linked_ports: 로컬 포트 제외한 링크 필요 포트 수
   * 동기화: 초기화 후 읽기 전용. */

  uca_org_t local_result;
  /* [한국어] init_interface() 반환값 저장용 더미 구조체.
   * 설정자: 생성자에서 init_interface(&interface_ip)의 반환값 저장.
   * 읽는 자: 사용되지 않음 (더미 — init_interface의 부수 효과만이 목적).
   * 동기화: 초기화 후 불변. */

  statsDef tdp_stats;
  /* [한국어] TDP(Thermal Design Power) 통계 구조체 — TDP 전력 계산 시 활성 접근 수 등 저장.
   * 설정자: computeEnergy(is_tdp=true) 내에서 stats_t를 복사하여 초기화.
   * 읽는 자: TDP 전력 누적 계산에서 참조.
   * 동기화: computeEnergy() 호출 시 단일 스레드에서 갱신. */

  statsDef rtp_stats;
  /* [한국어] RTP(Runtime Power) 통계 구조체 — 런타임 전력 계산 시 actual 접근 수 저장.
   * 설정자: computeEnergy(is_tdp=false) 내에서 total_accesses를 읽어 초기화.
   * 읽는 자: rt_power 계산에서 rtp_stats.readAc.access를 곱셈 인수로 사용.
   * 동기화: computeEnergy() 호출 시 단일 스레드에서 갱신. */

  statsDef stats_t;
  /* [한국어] 현재 계산 사이클의 통계 임시 버퍼.
   * 설정자: computeEnergy() 내에서 readAc.access = M (TDP) 또는 total_accesses (RTP)로 설정.
   * 읽는 자: tdp_stats 또는 rtp_stats로 복사된 후 참조.
   * 동기화: computeEnergy() 단일 스레드 내에서만 수정. */

  powerDef power_t;
  /* [한국어] 전력 계산 임시 버퍼 (현재 NoC에서 직접 사용 빈도 낮음).
   * 설정자: computeEnergy() 내 임시 계산에서 사용 가능.
   * 읽는 자: 집계된 전력 결과를 power 필드에 옮기기 전 임시 보관.
   * 동기화: computeEnergy() 단일 스레드 내부에서만 수정. */

  Component link_bus_tot_per_Router;
  /* [한국어] 라우터 1개당 총 링크/버스 전력·면적 집계 Component.
   * 설정자: init_link_bus()에서 link_bus->area * global_linked_ports로 면적 설정;
   *         computeEnergy()에서 link_bus->power * pppm_t로 전력 설정.
   * 읽는 자: computeEnergy()가 이 값에 total_nodes를 곱해 전체 NoC 링크 전력을 산출;
   *          displayEnergy()가 "Per Router Links" 항목으로 출력.
   * 동기화: 초기화 후 computeEnergy()에서 갱신. */

  bool link_bus_exist;
  /* [한국어] 링크/버스 모델이 초기화되었는지 여부.
   * 설정자: init_link_bus() 완료 시 true로 설정; 생성자에서 false로 초기화.
   * 읽는 자: computeEnergy(), displayEnergy()에서 link_bus 관련 코드 진입 조건.
   * 값 범위: true 또는 false.
   * 동기화: 초기화 후 변경 없음. */

  bool router_exist;
  /* [한국어] 라우터 모델이 초기화되었는지 여부.
   * 설정자: init_router() 완료 시 true로 설정; 생성자에서 false로 초기화.
   * 읽는 자: computeEnergy(), displayEnergy()에서 router 관련 코드 진입 조건.
   * 값 범위: true 또는 false.
   * 동기화: 초기화 후 변경 없음. */

  string name, link_name;
  /* [한국어] NoC 및 링크의 표시 이름 문자열.
   * name: "NOC" (type=1) 또는 "BUSES" (type=0) — set_noc_param()에서 설정.
   *        displayEnergy()와 interconnect 생성자에 전달되어 경고 메시지에 출력.
   * link_name: "Links" (type=1) 또는 "Bus" (type=0) — init_link_bus()에서 설정.
   *             displayEnergy()에서 서브컴포넌트 이름으로 출력.
   * 동기화: 초기화 후 변경 없음. */

  double M_traffic_pattern;
  /* [한국어] NoC 트래픽 패턴 계수 — 각 포트의 평균 사용률을 나타냄 (기본값 0.6).
   * 설정자: NoC 생성자 인수 M_traffic_pattern_ (기본값 0.6)으로 초기화.
   * 읽는 자: init_router()가 MCPAT_Router 생성자에 전달;
   *          computeEnergy() TDP 모드에서 링크 동적 전력 스케일링에 사용
   *          (pppm_t[0] = M_traffic_pattern * duty_cycle * min_ports).
   *          displayEnergy()에서 M = M_traffic_pattern * duty_cycle로 계산하여 서브컴포넌트 전력 출력.
   * 값 범위: 0.0~1.0. 0.6 = 60% 트래픽 활용률.
   * 동기화: 초기화 후 변경 없음 (단일 스레드 읽기). */

  NoC(ParseXML *XML_interface, int ithNoC_, InputParameter *interface_ip_,
      double M_traffic_pattern_ = 0.6, double link_len_ = 0);
  /* [한국어] NoC 생성자 선언.
   * XML_interface: 시스템 설정 XML 파싱 결과.
   * ithNoC_: 이 NoC가 참조할 sys.NoC[] 배열 인덱스.
   * interface_ip_: CACTI 기술 파라미터 입력.
   * M_traffic_pattern_: 트래픽 활용률 (기본 0.6 = 60%).
   * link_len_: 외부에서 전달되는 총 링크 길이 [m] (type=0 버스 모드에서 필요). */

  void set_noc_param();
  /* [한국어] XML->sys.NoC[ithNoC] 필드를 nocdynp 구조체로 파싱.
   * clockRate를 MHz에서 Hz로 변환, executionTime 계산, port 수/node 수 설정 포함. */

  void computeEnergy(bool is_tdp = true);
  /* [한국어] NoC 전력 계산 함수.
   * is_tdp=true: 최대 설계 전력(TDP) 계산 — duty_cycle과 M_traffic_pattern 스케일 적용.
   * is_tdp=false: 런타임 전력 계산 — GPGPU-Sim의 total_accesses 카운터를 사용. */

  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] 계산된 NoC 전력·면적 결과를 표준 출력으로 출력.
   * indent: 들여쓰기 공백 수, plevel: 출력 상세 수준 (>2이면 서브컴포넌트 출력), is_tdp: 모드 선택. */

  void init_link_bus(double link_len_);
  /* [한국어] 버스/링크 모델 초기화 — 링크 길이를 노드 수로 나누고 interconnect 객체 생성. */

  void init_router();
  /* [한국어] 라우터 모델 초기화 — MCPAT_Router 생성 및 longer_channel 누설 보정 적용. */

  void computeEnergy_link_bus(bool is_tdp = true);
  /* [한국어] 링크/버스 전력만 별도로 계산하는 함수 (외부 호출용, noc.cc에 구현 없음 — 미사용). */

  void displayEnergy_link_bus(uint32_t indent = 0, int plevel = 100,
                              bool is_tdp = true);
  /* [한국어] 링크/버스 전력·면적만 별도로 출력하는 함수 (미사용). */

  ~NoC();
  /* [한국어] 소멸자 — router와 link_bus 동적 객체를 안전하게 해제 (포인터를 0으로 초기화). */
};

#endif /* NOC_H_ */
