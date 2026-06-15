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
 * [한국어 설명] AccelWattch NoC(Network-on-Chip) 전력 모델 구현 (noc.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPU 내부 네트워크온칩(NoC)의 전력과 면적을 계산하는 `NoC` 클래스를 구현한다.
 * GPGPU-Sim에서 NoC는 SM(Streaming Multiprocessor)과 L2 캐시 사이의 인터커넥트 패브릭을
 * 나타내며, AccelWattch(MICRO 2021)가 이 구조물의 TDP(최대 설계 전력) 및 런타임 전력을
 * 추정하는 데 사용한다. 라우터(crossbar + arbiter + 가상채널 버퍼)와 링크(배선) 전력을
 * 분리 추적하고, GPGPU-Sim의 시뮬레이션 카운터(total_accesses)를 입력으로 실시간 전력을 산출한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델의 NoC 서브시스템에 해당한다. GPGPU-Sim에서
 * intersim2(NoC 트래픽 시뮬레이터)가 생성한 통계를 XML을 통해 읽어들여 전력을 추정한다.
 * 호출 체인: gpgpu-sim/gpu-sim.cc → accelwattch 진입점 → NoC::computeEnergy() → [이 파일]
 *             → MCPAT_Router (cacti/router.h) / interconnect (interconnect.cc)
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, GPGPU-Sim 전력 계산 단계 (단일 스레드).
 *
 * === 타 모듈과의 연결 ===
 * 의존: noc.h (클래스 선언), XML_Parse.h (ParseXML), cacti/basic_circuit.h (전기 특성 상수),
 *       const.h (전역 상수), io.h (I/O 유틸), parameter.h (init_interface, longer_channel_device_reduction),
 *       MCPAT_Router (cacti/router.h), interconnect (interconnect.cc).
 * 의존받음: AccelWattch 전력 집계 모듈 (NoC::computeEnergy() 결과인 power/rt_power를 합산).
 * 데이터 흐름: XML->sys.NoC[ithNoC] → set_noc_param() → nocdynp → init_router()/init_link_bus()
 *              → computeEnergy() → power.readOp.dynamic / rt_power.readOp.dynamic.
 * 공유 자료구조: pppm_t[4] (power_point_product_masks) — [동적스케일, 누설스케일, 게이트누설스케일, 면적스케일].
 *
 * === 주요 함수/구조체 요약 ===
 * NoC()             : 생성자 — 임베디드/서버 배선 유형 선택, set_noc_param(), init_interface() 후
 *                     라우터(init_router) 또는 버스(init_link_bus) 초기화.
 * init_router()     : MCPAT_Router 생성, total_nodes 면적 합산, longer_channel 누설 보정 적용.
 * init_link_bus()   : 링크 길이 = 총 길이 / 평균 노드 수 / 2(이웃 공유). interconnect 객체 생성.
 * computeEnergy()   : TDP 모드: duty_cycle×M_traffic_pattern 스케일. Runtime 모드: total_accesses 기반.
 * set_noc_param()   : XML 필드 → nocdynp 변환. clockRate MHz→Hz. global_linked_ports 계산.
 * ~NoC()            : router/link_bus 포인터 안전 해제.
 */

#include "noc.h"                    // [한국어] NoC 클래스 선언 및 의존 헤더 포함
#include <assert.h>                 // [한국어] assert() — 입력값 유효성 런타임 검증
#include <algorithm>                // [한국어] std::min — min_ports 계산에 사용
#include <cmath>                    // [한국어] ceil() — 파이프라인 스테이지 수 계산 (interconnect에서 사용)
#include <iostream>                 // [한국어] cout — displayEnergy() 결과 출력
#include <string>                   // [한국어] std::string — name, link_name 이름 문자열
#include "XML_Parse.h"              // [한국어] ParseXML — sys.NoC[] 파라미터 접근
#include "cacti/basic_circuit.h"    // [한국어] 전기 기본 회로 상수 및 함수
#include "const.h"                  // [한국어] AccelWattch 전역 상수 (pppm_lkg 등)
#include "io.h"                     // [한국어] I/O 관련 유틸 함수
#include "parameter.h"              // [한국어] init_interface(), longer_channel_device_reduction(), set_pppm()

/*
 * [한국어]
 * NoC::NoC() - NoC 전력 모델 생성자
 *
 * @XML_interface    : 전체 시스템/GPU 설정 XML 파싱 객체 포인터
 * @ithNoC_          : 이 객체가 참조할 sys.NoC[] 배열의 인덱스
 * @interface_ip_    : CACTI 기술 파라미터 입력 (기술 노드, 온도, 배선 유형 등)
 * @M_traffic_pattern_: 트래픽 활용률 계수 (기본값 0.6 = 60%). 동적 전력 스케일에 곱해짐.
 * @link_len_        : 외부에서 전달하는 총 링크 길이 [m]. type=0(버스) 모드에서 사용.
 *                     type=1(라우터) 모드에서는 링크 길이를 외부에서 별도 호출로 전달.
 * @return           : (생성자) — power/area 필드에 결과 저장, router 또는 link_bus 객체 생성.
 *
 * 생성자는 임베디드(Embedded) 시스템 여부에 따라 배선 유형(Global_30 vs Global)을 선택하고,
 * set_noc_param()으로 XML 파라미터를 파싱한 후 init_interface()로 전역 기술 파라미터를 초기화한다.
 * nocdynp.type이 1(라우터)이면 init_router()를, 0(버스)이면 init_link_bus()를 호출한다.
 * GPU의 NoC는 통상 라우터 기반(intersim2의 Booksim 모델)이나 단순 버스 모드도 지원한다.
 *
 * 실행 컨텍스트: GPGPU-Sim 초기화 단계, 단일 스레드.
 *
 * 호출 체인:
 *   AccelWattch 진입점 → [NoC 생성자] → set_noc_param()
 *                                      → init_interface()
 *                                      → init_router() 또는 init_link_bus()
 */
NoC::NoC(ParseXML* XML_interface, int ithNoC_, InputParameter* interface_ip_,
         double M_traffic_pattern_, double link_len_)
    : XML(XML_interface),         // [한국어] XML 파싱 객체 포인터 저장
      ithNoC(ithNoC_),            // [한국어] sys.NoC[] 인덱스 저장
      interface_ip(*interface_ip_), // [한국어] InputParameter 값 복사 (포인터 아닌 복사본)
      router(0),                  // [한국어] 라우터 포인터 null 초기화 — 아직 생성 전
      link_bus(0),                // [한국어] 링크 버스 포인터 null 초기화
      link_bus_exist(false),      // [한국어] 링크 버스 초기화 완료 플래그 false로 시작
      router_exist(false),        // [한국어] 라우터 초기화 완료 플래그 false로 시작
      M_traffic_pattern(M_traffic_pattern_) { // [한국어] 트래픽 활용률 저장 (0.6 = 60%)
  /*
   * initialize, compute and optimize individual components.
   */

  if (XML->sys.Embedded) {
    // [한국어] 임베디드 시스템(저전력, 작은 면적): Global_30 배선 유형 사용.
    //          Global_30은 30% 길이 감소를 가정하는 금속 배선 레이어로,
    //          임베디드 SoC에서 흔히 사용하는 더 짧고 조밀한 배선 레이어에 해당.
    interface_ip.wt = Global_30;        // [한국어] 배선 유형: 글로벌 30% 레이어
    interface_ip.wire_is_mat_type = 0;  // [한국어] 내부 배선(is=inner) 매트 유형 0
    interface_ip.wire_os_mat_type = 1;  // [한국어] 외부 배선(os=outer) 매트 유형 1
  } else {
    // [한국어] 서버/데스크톱 시스템: 표준 Global 배선 유형 사용.
    //          GPU는 통상 이 경로를 따름 (고성능, 긴 배선 허용).
    interface_ip.wt = Global;           // [한국어] 배선 유형: 표준 글로벌 레이어
    interface_ip.wire_is_mat_type = 2;  // [한국어] 내부 배선 매트 유형 2 (서버급)
    interface_ip.wire_os_mat_type = 2;  // [한국어] 외부 배선 매트 유형 2 (서버급)
  }
  set_noc_param(); // [한국어] XML->sys.NoC[ithNoC] 필드를 nocdynp 구조체로 파싱
  local_result = init_interface(&interface_ip);
  // [한국어] 전역 기술 파라미터(g_tp)를 interface_ip 기반으로 초기화.
  //          이후 g_tp.min_w_nmos_, g_tp.sckt_co_eff 등이 설정됨.
  //          반환값 local_result는 더미로, 부수 효과(g_tp 초기화)가 목적.
  scktRatio = g_tp.sckt_co_eff;
  // [한국어] 소켓 계수 저장 — 패키징 오버헤드 계수 (통상 1.1 이상).
  //          NoC 클래스 자체에서 직접 사용되지 않으나 하위 모듈 참조를 위해 보관.

  if (nocdynp.type) { /*
                       * if NOC compute router, router links must be computed
                       * separately and called from external since total chip
                       * area must be known first
                       */
    // [한국어] type=1: 라우터 기반 NoC (Booksim/intersim2 스타일).
    //          라우터 내부 링크는 칩 전체 면적을 알아야 길이를 계산할 수 있으므로,
    //          링크 초기화는 외부에서 init_link_bus()를 별도로 호출해야 한다.
    init_router(); // [한국어] MCPAT_Router 생성 및 total_nodes 기준 면적 합산
  } else {
    init_link_bus(link_len_);  // if bus compute bus
    // [한국어] type=0: 버스 기반 NoC. link_len_을 받아 바로 interconnect 생성.
  }

  //  //clock power
  //  clockNetwork.init_wire_external(is_default, &interface_ip);
  //  clockNetwork.clk_area           =area*1.1;//10% of placement overhead.
  //  rule of thumb clockNetwork.end_wiring_level   =5;//toplevel metal
  //  clockNetwork.start_wiring_level =5;//toplevel metal
  //  clockNetwork.num_regs           = corepipe.tot_stage_vector;
  //  clockNetwork.optimize_wire();
  // [한국어] (주석 처리됨) 클럭 네트워크 전력 계산 코드 — 현재 미사용.
  //          향후 클럭 배선 전력을 포함하려면 이 블록을 활성화해야 함.
}

/*
 * [한국어]
 * NoC::init_router() - MCPAT_Router 객체 생성 및 longer_channel 누설 보정 적용
 *
 * @return: void — router 포인터, area, router_exist 플래그를 업데이트
 *
 * MCPAT_Router를 생성하여 크로스바(crossbar), 중재기(arbiter), 가상채널 버퍼(buffer)의
 * 전력과 면적을 계산한다. 총 NoC 면적은 단일 라우터 면적 × total_nodes로 합산된다.
 * longer_channel_device_reduction()을 이용해 롱채널 소자 보정 계수를 계산하고
 * 라우터 전체 및 각 서브컴포넌트의 longer_channel_leakage를 설정한다.
 *
 * 실행 컨텍스트: NoC 생성자 내에서 단 한 번 호출 (단일 스레드).
 *
 * 호출 체인:
 *   NoC() → [init_router()] → MCPAT_Router(flit_size, vc_buffer_entries, vc_per_port,
 *                                           peri_global, input_ports, output_ports, M_traffic_pattern)
 *                            → longer_channel_device_reduction(Uncore_device)
 */
void NoC::init_router() {
  router = new MCPAT_Router(
      nocdynp.flit_size,                                           // [한국어] 플릿(flit) 비트 폭 — 라우터 입출력 버퍼 크기의 기본 단위
      nocdynp.virtual_channel_per_port * nocdynp.input_buffer_entries_per_vc,
      // [한국어] 포트당 버퍼 총 엔트리 수 = 가상채널 수 × 채널당 버퍼 엔트리 수.
      //          예: 4 VC × 4 entries = 16 entries/port. 이 값이 입력 버퍼 SRAM 크기를 결정.
      nocdynp.virtual_channel_per_port, // [한국어] 포트당 가상채널(VC) 수 — 데드락 회피 및 QoS를 위해 다수 VC 사용
      &(g_tp.peri_global),              // [한국어] 주변 회로(peripheral) 전역 기술 파라미터 포인터
      nocdynp.input_ports,              // [한국어] 라우터 입력 포트 수 (로컬 + 이웃 방향)
      nocdynp.output_ports,             // [한국어] 라우터 출력 포트 수
      M_traffic_pattern);               // [한국어] 트래픽 활용률 (0.6) — MCPAT_Router 내부 전력 스케일에 적용
  // router->print_router(); // [한국어] (주석 처리됨) 라우터 파라미터 디버그 출력

  area.set_area(area.get_area() +
                router->area.get_area() * nocdynp.total_nodes);
  // [한국어] 전체 NoC 면적 = 기존 면적 + 단일 라우터 면적 × 노드(라우터) 총 수.
  //          total_nodes = horizontal_nodes × vertical_nodes (메시 토폴로지 기준).

  double long_channel_device_reduction =
      longer_channel_device_reduction(Uncore_device);
  // [한국어] Uncore(NoC는 코어 외부) 디바이스에 대한 롱채널 누설 보정 계수 계산.
  //          롱채널 소자를 사용하면 서브스레솔드 누설이 줄어드는 계수 (0.0~1.0).

  router->power.readOp.longer_channel_leakage =
      router->power.readOp.leakage * long_channel_device_reduction;
  // [한국어] 라우터 전체 longer_channel 누설 = 기본 누설 × 보정 계수.
  //          displayEnergy()에서 XML->sys.longer_channel_device=true이면 이 값 사용.

  router->buffer.power.readOp.longer_channel_leakage =
      router->buffer.power.readOp.leakage * long_channel_device_reduction;
  // [한국어] 가상채널 버퍼(SRAM) 서브컴포넌트 누설 보정 적용

  router->crossbar.power.readOp.longer_channel_leakage =
      router->crossbar.power.readOp.leakage * long_channel_device_reduction;
  // [한국어] 크로스바(crossbar) 서브컴포넌트 누설 보정 적용.
  //          크로스바: 입력 포트에서 출력 포트로 플릿을 전달하는 멀티플렉서 네트워크.

  router->arbiter.power.readOp.longer_channel_leakage =
      router->arbiter.power.readOp.leakage * long_channel_device_reduction;
  // [한국어] 중재기(arbiter) 서브컴포넌트 누설 보정 적용.
  //          중재기: 동일 출력 포트를 요청하는 복수 입력 포트 간 충돌을 해결하는 회로.

  router_exist = true; // [한국어] 라우터 초기화 완료 플래그 설정 — computeEnergy/displayEnergy가 이 플래그 확인
}

/*
 * [한국어]
 * NoC::init_link_bus() - 링크/버스 모델 초기화 및 interconnect 객체 생성
 *
 * @link_len_: 외부에서 전달하는 총 chip 링크 길이 [m] (칩 전체 치수 기반)
 * @return: void — link_bus 포인터, link_bus_tot_per_Router.area, area, link_bus_exist 업데이트
 *
 * 총 링크 길이를 평균 노드 수와 이웃 공유를 고려하여 단일 링크 길이로 변환한 후,
 * interconnect 객체를 생성하여 배선 전력과 면적을 계산한다.
 * link_bus_tot_per_Router.area는 라우터 1개당 총 링크 면적 (단일 링크 면적 × global_linked_ports)으로 설정된다.
 * 전체 NoC 링크 면적은 라우터 1개당 면적 × total_nodes로 합산된다.
 *
 * 실행 컨텍스트: NoC 생성자 또는 외부에서 단 한 번 호출 (단일 스레드).
 *
 * 호출 체인:
 *   NoC() → [init_link_bus()] → interconnect(name, Uncore_device, 1, 1, flit_size,
 *                                            link_len, interface_ip, 3, pipelinable=true,
 *                                            route_over_perc)
 */
void NoC ::init_link_bus(double link_len_) {
  //	if (nocdynp.min_ports==1 )
  if (nocdynp.type)
    link_name = "Links"; // [한국어] 라우터 기반 NoC의 링크 이름 — displayEnergy에서 "Per Router Links"로 출력
  else
    link_name = "Bus";   // [한국어] 버스 기반 NoC의 이름 — displayEnergy에서 "Bus"로 출력

  link_len = link_len_; // [한국어] 외부에서 받은 총 링크 길이 저장 (단위: m)
  assert(link_len > 0); // [한국어] 링크 길이는 반드시 양수 — 0이면 배선 모델 계산 불가

  interface_ip.throughput = nocdynp.link_throughput / nocdynp.clockRate;
  // [한국어] throughput [s/cycle] = 링크 처리율 [사이클] / NoC 클럭 [Hz].
  //          link_throughput는 사이클 단위이므로 clockRate로 나누어 시간 단위[s]로 변환.
  //          interconnect 생성자가 이 값을 배선 타이밍 최적화의 목표로 사용.
  interface_ip.latency = nocdynp.link_latency / nocdynp.clockRate;
  // [한국어] latency [s] = 링크 지연 [사이클] / NoC 클럭 [Hz].
  //          배선 전파 지연이 이 값 이하여야 타이밍 제약을 충족.

  link_len /= (nocdynp.horizontal_nodes + nocdynp.vertical_nodes) / 2;
  // [한국어] 총 링크 길이를 노드 수의 평균으로 나누어 단일 노드 간 평균 거리 산출.
  //          (horizontal + vertical) / 2: 2D 메시에서 x/y 방향 노드 수의 평균.
  //          예: 4×4 메시 → (4+4)/2 = 4. 총 길이 / 4 = 노드 간 평균 간격.

  if (nocdynp.total_nodes > 1)
    link_len /= 2;  // All links are shared by neighbors
  // [한국어] 노드가 2개 이상이면 링크를 이웃 노드와 공유하므로 추가로 2로 나눔.
  //          메시에서 각 링크는 두 인접 노드가 공유하므로 실효 링크 길이는 절반.
  //          total_nodes=1이면 공유 없음 (단일 노드 — 버스 말단).

  link_bus = new interconnect(name, Uncore_device, 1, 1, nocdynp.flit_size,
                              link_len, &interface_ip, 3, true /*pipelinable*/,
                              nocdynp.route_over_perc);
  // [한국어] 링크/버스 배선 전력 모델 생성.
  //   name         : "NOC" 또는 "BUSES" — 경고 메시지와 식별에 사용
  //   Uncore_device: NoC는 코어 외부 → Uncore 디바이스 유형
  //   base_w=1, base_h=1 : 기반 구조물 크기 (링크 단독 모델에서는 사용 안 함)
  //   nocdynp.flit_size  : 버스 데이터 폭 [비트] — 전체 버스 전력 = per-bit × flit_size
  //   link_len           : 계산된 단일 링크 길이 [m]
  //   interface_ip       : throughput/latency 제약 포함 기술 파라미터
  //   3                  : start_wiring_level=3 (글로벌 금속 레이어에서 시작)
  //   pipelinable=true   : 처리율(throughput) 기준 최적화 수행 (버스는 파이프라인 가능)
  //   route_over_perc    : 배선이 소자 위를 지나는 비율

  link_bus_tot_per_Router.area.set_area(
      link_bus_tot_per_Router.area.get_area() +
      link_bus->area.get_area() * nocdynp.global_linked_ports);
  // [한국어] 라우터 1개당 총 링크 면적 = 단일 링크 면적 × 글로벌 포트 수.
  //          global_linked_ports = (input_ports-1) + (output_ports-1): 로컬 포트를 제외한
  //          모든 포트가 링크를 필요로 하므로, 라우터당 이 수만큼의 링크 면적이 필요.

  area.set_area(area.get_area() +
                link_bus_tot_per_Router.area.get_area() * nocdynp.total_nodes);
  // [한국어] 전체 NoC 링크 면적 합산 = 라우터당 링크 면적 × 총 노드(라우터) 수.

  link_bus_exist = true; // [한국어] 링크 버스 초기화 완료 플래그 설정
}

/*
 * [한국어]
 * NoC::computeEnergy() - NoC TDP 또는 런타임 전력 계산
 *
 * @is_tdp: true=TDP(최대 설계 전력) 계산, false=런타임 실제 전력 계산
 * @return: void — power (TDP) 또는 rt_power (런타임) 필드를 업데이트
 *
 * TDP 모드에서는 duty_cycle(활성 사이클 비율)과 M_traffic_pattern(트래픽 활용률)을
 * pppm_t 스케일 인수로 적용하여 이론적 최대 전력을 계산한다.
 * 런타임 모드에서는 GPGPU-Sim이 기록한 XML->sys.NoC[ithNoC].total_accesses 카운터를
 * 이용하여 실제 동적 에너지를 계산한다.
 *
 * pppm_t[4] = [동적_스케일, 누설_스케일, 게이트누설_스케일, 면적_스케일] 구조.
 * set_pppm(pppm_t, a, b, c, d)는 각 원소를 a, b, c, d로 설정하는 헬퍼 함수.
 *
 * 실행 컨텍스트: AccelWattch 전력 수집 단계 (단일 스레드).
 *
 * 호출 체인:
 *   AccelWattch 진입점 → [computeEnergy(is_tdp)] → router->power / link_bus->power 참조
 *                                                 → XML->sys.NoC[ithNoC].total_accesses 읽기
 */
void NoC::computeEnergy(bool is_tdp) {
  // power_point_product_masks
  double pppm_t[4] = {1, 1, 1, 1};
  // [한국어] pppm_t: power_point_product_masks — 4원소 배열.
  //          pppm_t[0]=동적전력 스케일, pppm_t[1]=누설 스케일,
  //          pppm_t[2]=게이트누설 스케일, pppm_t[3]=면적 스케일.
  //          초기값 {1,1,1,1} = 스케일 없음. set_pppm()으로 필요한 값으로 설정 후
  //          power * pppm_t 연산으로 각 항목을 독립적으로 스케일할 수 있음.

  double M = nocdynp.duty_cycle;
  // [한국어] M = duty_cycle: 전체 실행 중 NoC가 활성 상태인 사이클 비율 (0.0~1.0).
  //          XML->sys.NoC[ithNoC].duty_cycle에서 읽음. 예: 0.5 = 절반 시간 활성.

  // nocdynp.executionTime=XML->sys.total_cycles/(XML->sys.target_core_clockrate*1e6);//Syed
  // cout<<"NOC Total Cycles: "<<XML->sys.total_cycles<<endl;
  // cout<<"NOC Clock Rate: "<<XML->sys.target_core_clockrate<<endl;
  // [한국어] (주석 처리됨) executionTime 재계산 및 디버그 출력 코드 — 현재 set_noc_param()에서 처리됨

  if (is_tdp) {
    // [한국어] TDP(최대 설계 전력) 계산 경로.
    //          duty_cycle과 M_traffic_pattern을 반영하여 최악의 경우 전력을 산출.
    //          라우터 전력은 MCPAT_Router 내부에서 이미 M_traffic_pattern이 적용됐으므로
    //          여기서는 duty_cycle(M)만 추가 스케일한다.

    // init stats for TDP
    stats_t.readAc.access = M; // [한국어] 통계 버퍼에 duty_cycle을 활성 접근 수로 설정
    tdp_stats = stats_t;       // [한국어] TDP 통계 버퍼에 복사 저장

    if (router_exist) {
      // [한국어] 라우터가 초기화된 경우 라우터 전력을 duty_cycle로 스케일하여 누적
      set_pppm(pppm_t, 1 * M, 1, 1, 1);  // reset traffic pattern
      // [한국어] pppm_t = {M, 1, 1, 1}: 동적 전력만 M(duty_cycle)으로 스케일.
      //          누설/게이트누설/면적은 duty_cycle과 무관하므로 스케일 1 유지.
      router->power = router->power * pppm_t;
      // [한국어] 라우터 전력에 duty_cycle 스케일 적용.
      //          MCPAT_Router는 이미 M_traffic_pattern이 내부 적용됐으므로 M만 곱함.

      set_pppm(pppm_t, nocdynp.total_nodes, nocdynp.total_nodes,
               nocdynp.total_nodes, nocdynp.total_nodes);
      // [한국어] pppm_t = {total_nodes, total_nodes, total_nodes, total_nodes}:
      //          단일 라우터 전력을 전체 NoC 라우터 수(total_nodes)로 스케일하여 전체 NoC 전력 산출.
      power = power + router->power * pppm_t;
      // [한국어] 전체 NoC 전력에 (라우터 1개 × duty_cycle) × total_nodes를 누적
    }

    if (link_bus_exist) {
      // [한국어] 링크/버스가 초기화된 경우 링크 전력을 트래픽 패턴 및 duty_cycle로 스케일하여 누적
      if (nocdynp.type)
        set_pppm(pppm_t, 1 * M_traffic_pattern * M * (nocdynp.min_ports - 1),
                 nocdynp.global_linked_ports, nocdynp.global_linked_ports,
                 nocdynp.global_linked_ports);
      // reset traffic pattern; local port do not have router links
      // [한국어] 라우터 기반 NoC 링크 전력 스케일:
      //          동적 스케일 = M_traffic_pattern × M × (min_ports - 1).
      //          (min_ports - 1): 로컬 포트(SM 연결 포트)는 라우터 간 링크가 없으므로 제외.
      //          min_ports = min(input_ports, output_ports) — 병목(bottleneck) 포트 수.
      //          누설/게이트누설/면적 스케일 = global_linked_ports (링크 수만큼 복제).
      else
        set_pppm(pppm_t, 1 * M_traffic_pattern * M * (nocdynp.min_ports),
                 nocdynp.global_linked_ports, nocdynp.global_linked_ports,
                 nocdynp.global_linked_ports);  // reset traffic pattern
      // [한국어] 버스 기반: min_ports에서 -1 없이 그대로 사용 (로컬 포트 구분 없음)

      link_bus_tot_per_Router.power = link_bus->power * pppm_t;
      // [한국어] 라우터 1개당 링크 전력 = 단일 링크 전력 × pppm_t 스케일.
      //          이 값은 뒤에서 total_nodes를 곱해 전체 NoC 링크 전력이 됨.

      set_pppm(pppm_t, nocdynp.total_nodes, nocdynp.total_nodes,
               nocdynp.total_nodes, nocdynp.total_nodes);
      // [한국어] 전체 노드 수로 스케일 — 라우터 1개당 링크 전력을 전체 NoC로 확장
      power = power + link_bus_tot_per_Router.power * pppm_t;
      // [한국어] 전체 NoC 전력에 링크 전력 × total_nodes 누적
    }
  } else {
    // [한국어] 런타임 전력(RTP, Runtime Power) 계산 경로.
    //          GPGPU-Sim 시뮬레이션 중 실제 NoC 접근 횟수(total_accesses)를 기반으로
    //          실제 소비된 동적 에너지를 계산한다.

    rt_power.reset();                      // [한국어] 런타임 전력 누적값 초기화
    router->buffer.rt_power.reset();       // [한국어] 버퍼 런타임 전력 초기화
    router->crossbar.rt_power.reset();     // [한국어] 크로스바 런타임 전력 초기화
    router->arbiter.rt_power.reset();      // [한국어] 중재기 런타임 전력 초기화
    router->rt_power.reset();              // [한국어] 라우터 전체 런타임 전력 초기화
    // link_bus->rt_power.reset();         // [한국어] (주석 처리됨) 링크 런타임 전력 초기화 — 아래에서 처리

    // init stats for runtime power (RTP)
    stats_t.readAc.access = XML->sys.NoC[ithNoC].total_accesses;
    // [한국어] GPGPU-Sim이 intersim2 시뮬레이션 중 집계한 NoC 총 접근 횟수.
    //          이 값이 실제 플릿 전송 횟수에 해당하며, 동적 에너지 = 단위 전력 × total_accesses.
    // cout<<"NOC(computeEnergy) read accesses: "<< stats_t.readAc.access<<endl;
    // [한국어] (주석 처리됨) 디버그 출력
    rtp_stats = stats_t; // [한국어] 런타임 통계 버퍼에 복사

    set_pppm(pppm_t, 1, 0, 0, 0);
    // [한국어] pppm_t = {1, 0, 0, 0}: 동적 전력만 스케일, 누설/게이트누설/면적은 0으로 마스킹.
    //          런타임 전력은 동적 에너지만 집계하므로 누설 항목을 제외.

    if (router_exist) {
      // [한국어] 라우터 서브컴포넌트별 런타임 동적 에너지 계산

      router->buffer.rt_power.readOp.dynamic =
          (router->buffer.power.readOp.dynamic +
           router->buffer.power.writeOp.dynamic) *
          rtp_stats.readAc.access;
      // [한국어] 버퍼 런타임 에너지 = (읽기 동적 전력 + 쓰기 동적 전력) × 접근 횟수.
      //          가상채널 버퍼는 플릿이 들어올 때(write) + 나갈 때(read) 에너지가 소비됨.
      //          power.readOp.dynamic: TDP 계산 시 1회 접근 기준 단위 에너지 [J].

      router->crossbar.rt_power.readOp.dynamic =
          router->crossbar.power.readOp.dynamic * rtp_stats.readAc.access;
      // [한국어] 크로스바 런타임 에너지 = 단위 전력 × 접근 횟수.
      //          크로스바는 플릿이 통과할 때마다 입력 포트→출력 포트 경로가 스위칭되며 에너지 소비.

      router->arbiter.rt_power.readOp.dynamic =
          router->arbiter.power.readOp.dynamic * rtp_stats.readAc.access;
      // [한국어] 중재기 런타임 에너지 = 단위 전력 × 접근 횟수.
      //          매 사이클 출력 포트 경합을 해결하는 중재기가 동작할 때 에너지 소비.

      router->rt_power =
          router->rt_power +
          (router->buffer.rt_power + router->crossbar.rt_power +
           router->arbiter.rt_power) *
              pppm_t +
          router->power * pppm_lkg;  // TDP power must be calculated first!
      // [한국어] 라우터 총 런타임 전력 = 서브컴포넌트 동적 에너지 합 + 누설 에너지.
      //          (buffer + crossbar + arbiter) * pppm_t: 동적 에너지 합 (pppm_t[0]=1, 나머지 0).
      //          router->power * pppm_lkg: 누설 전력 항목.
      //          pppm_lkg: 전역 누설 스케일 마스크 (const.h에 정의됨).
      //          주석 "TDP power must be calculated first": router->power에 TDP 전력이
      //          설정되어 있어야 pppm_lkg 곱셈이 올바르게 동작함을 명시.

      rt_power = rt_power + router->rt_power;
      // [한국어] 전체 NoC 런타임 전력에 라우터 런타임 전력 누적
    }

    if (link_bus_exist) {
      // [한국어] 링크/버스 런타임 에너지 계산
      link_bus->rt_power.reset(); // [한국어] 링크 런타임 전력 초기화

      set_pppm(pppm_t, rtp_stats.readAc.access, 1, 1, rtp_stats.readAc.access);
      // [한국어] pppm_t = {total_accesses, 1, 1, total_accesses}:
      //          동적 전력을 접근 횟수로 스케일. 면적도 accesses로 스케일 (런타임 통계용).
      //          누설/게이트누설은 1 (별도 정규화 없이 그대로 유지).
      link_bus->rt_power = link_bus->power * pppm_t;
      // [한국어] 링크 런타임 에너지 = 단위 링크 전력 × total_accesses 스케일.
      rt_power = rt_power + link_bus->rt_power;
      // [한국어] 전체 NoC 런타임 전력에 링크 런타임 에너지 누적
    }
  }
}

/*
 * [한국어]
 * NoC::displayEnergy() - 계산된 NoC 전력·면적 결과를 표준 출력으로 출력
 *
 * @indent   : 들여쓰기 공백 수 (계층적 출력을 위한 들여쓰기)
 * @plevel   : 출력 상세 수준 (>2이면 크로스바/중재기/버퍼 서브컴포넌트도 출력)
 * @is_tdp   : true=TDP 전력 출력, false=런타임 전력 출력 (현재 else 분기 미구현)
 * @return   : void — 결과를 cout으로 표준 출력
 *
 * 면적, 피크 동적 전력 [W], 서브스레솔드 누설 [W], 게이트 누설 [W], 런타임 동적 전력 [W]을 출력.
 * longer_channel_device 플래그에 따라 longer_channel_leakage 또는 기본 leakage를 출력한다.
 * plevel > 2이면 크로스바, 중재기, 가상채널 버퍼를 개별 서브컴포넌트로 세분화하여 출력한다.
 *
 * 실행 컨텍스트: AccelWattch 출력 단계 (단일 스레드).
 *
 * 호출 체인:
 *   AccelWattch 출력 루프 → [displayEnergy()] → router/link_bus 서브컴포넌트 출력
 */
void NoC::displayEnergy(uint32_t indent, int plevel, bool is_tdp) {
  string indent_str(indent, ' ');           // [한국어] indent 크기의 공백 문자열 생성 (들여쓰기용)
  string indent_str_next(indent + 2, ' '); // [한국어] 서브컴포넌트 들여쓰기 (추가 2칸)
  bool long_channel = XML->sys.longer_channel_device;
  // [한국어] 롱채널 소자 사용 여부 플래그 — true이면 누설 출력 시 longer_channel_leakage 사용

  double M = M_traffic_pattern * nocdynp.duty_cycle;
  // [한국어] M = 트래픽 활용률 × duty_cycle:  서브컴포넌트 전력 출력 시 사용되는 복합 스케일.
  //          주석에 명시된 대로, MCPAT_Router::power에는 이미 M_traffic_pattern이 내부 적용됐으나
  //          crossbar/arbiter/buffer를 개별 출력할 때는 이 M을 명시적으로 곱해야 함.
  /*only router as a whole has been applied the M_traffic_pattern(0.6 by
   * default) factor in router.cc; When power of crossbars, arbiters, etc need
   * to be displayed, the M_traffic_pattern factor need to be applied together
   * with McPAT's extra traffic pattern.
   * */
  if (is_tdp) {
    // [한국어] TDP 모드 출력 경로
    cout << name << endl; // [한국어] NoC 이름 ("NOC" 또는 "BUSES") 출력
    cout << indent_str << "Area = " << area.get_area() * 1e-6 << " mm^2"
         << endl;
    // [한국어] 전체 NoC 면적 출력 [mm²] (내부 단위 m² → 1e-6으로 변환)
    cout << indent_str
         << "Peak Dynamic = " << power.readOp.dynamic * nocdynp.clockRate
         << " W" << endl;
    // [한국어] 피크 동적 전력 [W] = 단위 에너지 [J/cycle] × 클럭 주파수 [Hz].
    //          power.readOp.dynamic은 사이클당 에너지이므로 clockRate를 곱해 전력으로 변환.
    cout << indent_str << "Subthreshold Leakage = "
         << (long_channel ? power.readOp.longer_channel_leakage
                          : power.readOp.leakage)
         << " W" << endl;
    // [한국어] 서브스레솔드 누설 전력 [W] 출력 — longer_channel이면 보정값 사용
    cout << indent_str << "Gate Leakage = " << power.readOp.gate_leakage << " W"
         << endl;
    // [한국어] 게이트 누설 전력 [W] 출력 (온도·기술 노드 의존)
    cout << indent_str << "Runtime Dynamic = "
         << rt_power.readOp.dynamic / nocdynp.executionTime << " W" << endl;
    // [한국어] 런타임 평균 동적 전력 [W] = 런타임 동적 에너지 [J] / 실행 시간 [s].
    //          executionTime = total_cycles / (target_core_clockrate * 1e6).
    // cout << indent_str<< "Execution Time = " << nocdynp.executionTime << " s"
    // << endl;
    // [한국어] (주석 처리됨) 실행 시간 출력 — 디버깅용
    cout << endl;

    if (router_exist) {
      // [한국어] 라우터가 초기화된 경우 라우터 서브컴포넌트 전력 출력
      cout << indent_str << "Router: " << endl;
      cout << indent_str_next << "Area = " << router->area.get_area() * 1e-6
           << " mm^2" << endl;
      // [한국어] 단일 라우터 면적 [mm²] 출력 (전체 NoC 면적 = 이 값 × total_nodes)
      cout << indent_str_next << "Peak Dynamic = "
           << router->power.readOp.dynamic * nocdynp.clockRate << " W" << endl;
      // [한국어] 라우터 피크 동적 전력 [W] = 단위 에너지 × clockRate
      cout << indent_str_next << "Subthreshold Leakage = "
           << (long_channel ? router->power.readOp.longer_channel_leakage
                            : router->power.readOp.leakage)
           << " W" << endl;
      // [한국어] 라우터 서브스레솔드 누설 [W] — longer_channel 여부에 따라 보정값 선택
      cout << indent_str_next
           << "Gate Leakage = " << router->power.readOp.gate_leakage << " W"
           << endl;
      // [한국어] 라우터 게이트 누설 [W]
      cout << indent_str_next << "Runtime Dynamic = "
           << router->rt_power.readOp.dynamic / nocdynp.executionTime << " W"
           << endl;
      // [한국어] 라우터 런타임 평균 동적 전력 [W]
      cout << endl;
      if (plevel > 2) {
        // [한국어] 상세 출력 수준(plevel>2)이면 크로스바/중재기/버퍼를 개별 출력
        cout << indent_str << indent_str << "Virtual Channel Buffer:" << endl;
        // [한국어] 가상채널(VC) 버퍼 서브컴포넌트 시작 출력
        cout << indent_str << indent_str_next << "Area = "
             << router->buffer.area.get_area() * 1e-6 * nocdynp.input_ports
             << " mm^2" << endl;
        // [한국어] 버퍼 총 면적 = 단일 포트 버퍼 면적 × input_ports (모든 입력 포트 버퍼 합산)
        cout << indent_str << indent_str_next << "Peak Dynamic = "
             << (router->buffer.power.readOp.dynamic +
                 router->buffer.power.writeOp.dynamic) *
                    nocdynp.min_ports * M * nocdynp.clockRate
             << " W" << endl;
        // [한국어] 버퍼 피크 동적 전력 = (read + write 단위 에너지) × min_ports × M × clockRate.
        //          min_ports: 동시에 활성화될 수 있는 최대 포트 수 (input/output 중 작은 값).
        //          M = M_traffic_pattern × duty_cycle.
        cout << indent_str << indent_str_next << "Subthreshold Leakage = "
             << (long_channel
                     ? router->buffer.power.readOp.longer_channel_leakage *
                           nocdynp.input_ports
                     : router->buffer.power.readOp.leakage *
                           nocdynp.input_ports)
             << " W" << endl;
        // [한국어] 버퍼 서브스레솔드 누설 = 단위 누설 × input_ports 합산
        cout << indent_str << indent_str_next << "Gate Leakage = "
             << router->buffer.power.readOp.gate_leakage * nocdynp.input_ports
             << " W" << endl;
        // [한국어] 버퍼 게이트 누설 = 단위 게이트 누설 × input_ports
        cout << indent_str << indent_str_next << "Runtime Dynamic = "
             << router->buffer.rt_power.readOp.dynamic / nocdynp.executionTime
             << " W" << endl;
        // [한국어] 버퍼 런타임 평균 동적 전력 [W]
        cout << endl;

        cout << indent_str << indent_str << "Crossbar:" << endl;
        // [한국어] 크로스바 서브컴포넌트 시작 출력
        cout << indent_str << indent_str_next
             << "Area = " << router->crossbar.area.get_area() * 1e-6 << " mm^2"
             << endl;
        // [한국어] 크로스바 면적 [mm²]
        cout << indent_str << indent_str_next << "Peak Dynamic = "
             << router->crossbar.power.readOp.dynamic * nocdynp.clockRate *
                    nocdynp.min_ports * M
             << " W" << endl;
        // [한국어] 크로스바 피크 동적 전력 = 단위 에너지 × clockRate × min_ports × M.
        //          크로스바는 동시에 min_ports 개의 플릿을 스위칭할 수 있음.
        cout << indent_str << indent_str_next << "Subthreshold Leakage = "
             << (long_channel
                     ? router->crossbar.power.readOp.longer_channel_leakage
                     : router->crossbar.power.readOp.leakage)
             << " W" << endl;
        // [한국어] 크로스바 서브스레솔드 누설 [W]
        cout << indent_str << indent_str_next
             << "Gate Leakage = " << router->crossbar.power.readOp.gate_leakage
             << " W" << endl;
        // [한국어] 크로스바 게이트 누설 [W]
        cout << indent_str << indent_str_next << "Runtime Dynamic = "
             << router->crossbar.rt_power.readOp.dynamic / nocdynp.executionTime
             << " W" << endl;
        // [한국어] 크로스바 런타임 평균 동적 전력 [W]
        cout << endl;

        cout << indent_str << indent_str << "Arbiter:" << endl;
        // [한국어] 중재기(arbiter) 서브컴포넌트 시작 출력
        cout << indent_str << indent_str_next << "Peak Dynamic = "
             << router->arbiter.power.readOp.dynamic * nocdynp.clockRate *
                    nocdynp.min_ports * M
             << " W" << endl;
        // [한국어] 중재기 피크 동적 전력 = 단위 에너지 × clockRate × min_ports × M.
        //          중재기 면적은 작으므로 생략하고 전력만 출력.
        cout << indent_str << indent_str_next << "Subthreshold Leakage = "
             << (long_channel
                     ? router->arbiter.power.readOp.longer_channel_leakage
                     : router->arbiter.power.readOp.leakage)
             << " W" << endl;
        // [한국어] 중재기 서브스레솔드 누설 [W]
        cout << indent_str << indent_str_next
             << "Gate Leakage = " << router->arbiter.power.readOp.gate_leakage
             << " W" << endl;
        // [한국어] 중재기 게이트 누설 [W]
        cout << indent_str << indent_str_next << "Runtime Dynamic = "
             << router->arbiter.rt_power.readOp.dynamic / nocdynp.executionTime
             << " W" << endl;
        // [한국어] 중재기 런타임 평균 동적 전력 [W]
        cout << endl;
      }
    }
    if (link_bus_exist) {
      // [한국어] 링크/버스가 초기화된 경우 링크 전력·면적 출력
      cout << indent_str << (nocdynp.type ? "Per Router " : "") << link_name
           << ": " << endl;
      // [한국어] 라우터 기반이면 "Per Router Links:", 버스 기반이면 "Bus:" 출력
      cout << indent_str_next
           << "Area = " << link_bus_tot_per_Router.area.get_area() * 1e-6
           << " mm^2" << endl;
      // [한국어] 라우터 1개당 총 링크 면적 [mm²] (= 단일 링크 면적 × global_linked_ports)
      cout << indent_str_next << "Peak Dynamic = "
           << link_bus_tot_per_Router.power.readOp.dynamic * nocdynp.clockRate
           << " W" << endl;
      // [한국어] 라우터 1개당 링크 피크 동적 전력 [W]
      cout << indent_str_next << "Subthreshold Leakage = "
           << (long_channel
                   ? link_bus_tot_per_Router.power.readOp.longer_channel_leakage
                   : link_bus_tot_per_Router.power.readOp.leakage)
           << " W" << endl;
      // [한국어] 라우터 1개당 링크 서브스레솔드 누설 [W]
      cout << indent_str_next << "Gate Leakage = "
           << link_bus_tot_per_Router.power.readOp.gate_leakage << " W" << endl;
      // [한국어] 라우터 1개당 링크 게이트 누설 [W]
      cout << indent_str_next << "Runtime Dynamic = "
           << link_bus->rt_power.readOp.dynamic / nocdynp.executionTime << " W"
           << endl;
      // [한국어] 링크 런타임 평균 동적 전력 [W] = 런타임 에너지 / 실행 시간
      cout << endl;
    }
  } else {
    // [한국어] 런타임 전력 출력 경로 — 현재 미구현 (주석 처리된 코드만 존재).
    //          원본 McPAT 코드에서 IFU/LSU/MMU/EXU 등 CPU 서브시스템 출력 코드가
    //          주석 처리된 채로 남아 있음. GPU NoC용 런타임 출력은 별도 추가 필요.

    //		cout << indent_str_next << "Instruction Fetch Unit    Peak
    // Dynamic
    //=
    //"
    //<< ifu->rt_power.readOp.dynamic*clockRate << " W" << endl;
    // cout
    //<< indent_str_next << "Instruction Fetch Unit    Subthreshold Leakage = "
    // << ifu->rt_power.readOp.leakage <<" W" << endl; 		cout <<
    // indent_str_next << "Instruction Fetch Unit    Gate Leakage = " <<
    // ifu->rt_power.readOp.gate_leakage << " W" << endl; 		cout <<
    // indent_str_next << "Load Store Unit   Peak Dynamic = " <<
    // lsu->rt_power.readOp.dynamic*clockRate  << " W" << endl;
    // cout
    // << indent_str_next << "Load Store Unit   Subthreshold Leakage = " <<
    // lsu->rt_power.readOp.leakage  << " W" << endl; 		cout <<
    // indent_str_next
    // << "Load Store Unit   Gate Leakage = " <<
    // lsu->rt_power.readOp.gate_leakage
    //<< " W" << endl; 		cout << indent_str_next << "Memory Management
    // Unit Peak Dynamic = " << mmu->rt_power.readOp.dynamic*clockRate  << " W"
    // <<
    // endl; 		cout << indent_str_next << "Memory Management Unit
    // Subthreshold Leakage = " << mmu->rt_power.readOp.leakage  << " W" <<
    // endl; 		cout
    // << indent_str_next << "Memory Management Unit   Gate Leakage = " <<
    // mmu->rt_power.readOp.gate_leakage  << " W" << endl; 		cout <<
    // indent_str_next << "Execution Unit   Peak Dynamic = " <<
    // exu->rt_power.readOp.dynamic*clockRate  << " W" << endl;
    // cout
    // << indent_str_next << "Execution Unit   Subthreshold Leakage = " <<
    // exu->rt_power.readOp.leakage  << " W" << endl; 		cout <<
    // indent_str_next
    // << "Execution Unit   Gate Leakage = " <<
    // exu->rt_power.readOp.gate_leakage
    //<< " W" << endl;
  }
}

/*
 * [한국어]
 * NoC::set_noc_param() - XML 필드를 nocdynp 구조체로 파싱
 *
 * @return: void — nocdynp 구조체의 모든 필드와 name 문자열을 설정
 *
 * XML->sys.NoC[ithNoC]에서 읽은 설정값을 nocdynp 구조체로 변환한다.
 * clockRate는 XML의 MHz 단위를 Hz로 변환하고 (×1e6),
 * executionTime은 total_cycles / (target_core_clockrate × 1e6)으로 계산한다.
 * 라우터 기반(type=1)이면 global_linked_ports = (input-1)+(output-1) (로컬 포트 제외),
 * 버스 기반(type=0)이면 포트 수를 모두 1로 고정한다.
 * chip_coverage와 route_over_perc는 [0,1] 범위임을 assert로 검증한다.
 *
 * 실행 컨텍스트: NoC 생성자 내에서 단 한 번 호출.
 *
 * 호출 체인:
 *   NoC() → [set_noc_param()] → (nocdynp 필드 설정만, 외부 호출 없음)
 */
void NoC::set_noc_param() {
  nocdynp.type = XML->sys.NoC[ithNoC].type;
  // [한국어] NoC 유형 읽기: 0=버스, 1=라우터 기반 NoC

  nocdynp.clockRate = XML->sys.NoC[ithNoC].clockrate;
  // [한국어] NoC 클럭 주파수 읽기 [MHz 단위] — 다음 줄에서 Hz로 변환

  nocdynp.clockRate *= 1e6;
  // [한국어] MHz → Hz 변환: XML 입력은 MHz 단위이므로 1e6을 곱해 Hz로 변환.
  //          이후 power.readOp.dynamic * clockRate = 전력 [W] 계산에 사용됨.

  nocdynp.executionTime =
      XML->sys.total_cycles / (XML->sys.target_core_clockrate * 1e6);
  // [한국어] 실행 시간 [s] = 전체 시뮬레이션 사이클 수 / (코어 클럭 [MHz] × 1e6).
  //          total_cycles: GPGPU-Sim이 시뮬레이션한 총 GPU 사이클 수.
  //          target_core_clockrate: GPU 코어 클럭 [MHz].
  //          displayEnergy에서 rt_power / executionTime = 평균 런타임 전력 [W].

  nocdynp.flit_size = XML->sys.NoC[ithNoC].flit_bits;
  // [한국어] 플릿(flit) 비트 폭 — 라우터/링크의 데이터 폭. interconnect의 data_width가 됨.

  if (nocdynp.type) {
    // [한국어] 라우터 기반 NoC 파라미터 설정
    nocdynp.input_ports = XML->sys.NoC[ithNoC].input_ports;
    // [한국어] 라우터 입력 포트 수 (로컬 + 이웃 방향: 예, 5-port mesh router)
    nocdynp.output_ports = XML->sys.NoC[ithNoC].output_ports;  // later minus 1
    // [한국어] 라우터 출력 포트 수 (주석 "later minus 1": 로컬 포트 제외를 global_linked_ports에서 처리)
    nocdynp.min_ports = min(nocdynp.input_ports, nocdynp.output_ports);
    // [한국어] min_ports = min(입력, 출력) — 동시에 활성화 가능한 최대 포트 수 (병목 기준).
    //          예: input=5, output=5 → min=5. 크로스바/중재기 전력 스케일에 사용.
    nocdynp.global_linked_ports =
        (nocdynp.input_ports - 1) + (nocdynp.output_ports - 1);
    /*
     * 	Except local i/o ports, all ports needs links( global_linked_ports);
     *  However only min_ports can be fully active simultaneously
     *  since the fewer number of ports (input or output ) is the bottleneck.
     */
    // [한국어] global_linked_ports = (input_ports-1) + (output_ports-1):
    //          각 라우터에서 로컬(SM 연결) 포트를 제외한 모든 포트가 이웃 라우터와 링크를 필요로 함.
    //          예: 5-port router (N/S/E/W + local) → (5-1) + (5-1) = 8 글로벌 링크 포트.
    //          이 수만큼의 interconnect 배선이 라우터당 필요하며, init_link_bus에서 면적 계산에 사용.
  } else {
    // [한국어] 버스 기반 NoC 파라미터 설정 — 포트를 1개로 고정
    nocdynp.input_ports = 1;     // [한국어] 버스: 입력 포트 1개
    nocdynp.output_ports = 1;    // [한국어] 버스: 출력 포트 1개
    nocdynp.min_ports = min(nocdynp.input_ports, nocdynp.output_ports);
    // [한국어] min_ports = 1 (버스는 단일 공유 채널)
    nocdynp.global_linked_ports = 1; // [한국어] 버스: 글로벌 링크 포트 1개
  }

  nocdynp.virtual_channel_per_port =
      XML->sys.NoC[ithNoC].virtual_channel_per_port;
  // [한국어] 포트당 가상채널(VC) 수 — 데드락 방지 및 QoS를 위해 여러 VC 사용 (예: 4 VC)

  nocdynp.input_buffer_entries_per_vc =
      XML->sys.NoC[ithNoC].input_buffer_entries_per_vc;
  // [한국어] 가상채널당 입력 버퍼 엔트리 수 — 버퍼 SRAM 크기 결정 (예: 4 entries/VC)

  nocdynp.horizontal_nodes = XML->sys.NoC[ithNoC].horizontal_nodes;
  // [한국어] NoC 메시 토폴로지의 수평 방향 노드 수 (예: 4×4 메시에서 4)
  nocdynp.vertical_nodes = XML->sys.NoC[ithNoC].vertical_nodes;
  // [한국어] NoC 메시 토폴로지의 수직 방향 노드 수 (예: 4×4 메시에서 4)
  nocdynp.total_nodes = nocdynp.horizontal_nodes * nocdynp.vertical_nodes;
  // [한국어] 총 라우터/노드 수 = 수평 × 수직 (예: 4×4=16). 면적/전력 합산 인수.

  nocdynp.duty_cycle = XML->sys.NoC[ithNoC].duty_cycle;
  // [한국어] 활성 사이클 비율 (0.0~1.0) — 전체 실행 중 NoC가 동작하는 시간 비율

  nocdynp.has_global_link = XML->sys.NoC[ithNoC].has_global_link;
  // [한국어] 글로벌 배선 레이어 사용 여부 플래그 (현재 직접 사용 안 됨)

  nocdynp.link_throughput = XML->sys.NoC[ithNoC].link_throughput;
  // [한국어] 링크 처리율 [사이클] — interconnect 생성 시 throughput 제약으로 변환됨

  nocdynp.link_latency = XML->sys.NoC[ithNoC].link_latency;
  // [한국어] 링크 지연 [사이클] — interconnect 생성 시 latency 제약으로 변환됨

  nocdynp.chip_coverage = XML->sys.NoC[ithNoC].chip_coverage;
  // [한국어] 이 NoC가 칩 면적의 몇 분의 1을 커버하는지 비율 (0.0~1.0). 면적 계산에 사용.

  nocdynp.route_over_perc = XML->sys.NoC[ithNoC].route_over_perc;
  // [한국어] 배선이 소자 위를 지나는 비율 (0.0~1.0) — interconnect 면적 계산에 전달

  assert(nocdynp.chip_coverage <= 1);    // [한국어] chip_coverage는 1 이하여야 함 (100% 초과 불가)
  assert(nocdynp.route_over_perc <= 1);  // [한국어] route_over_perc는 1 이하여야 함

  if (nocdynp.type)
    name = "NOC";   // [한국어] 라우터 기반 NoC 이름 설정 — displayEnergy 헤더 출력
  else
    name = "BUSES"; // [한국어] 버스 기반 NoC 이름 설정
}

/*
 * [한국어]
 * NoC::~NoC() - NoC 소멸자 — 동적 할당된 router/link_bus 해제
 *
 * @return: (소멸자) — router, link_bus 포인터를 안전하게 delete 후 0으로 초기화
 *
 * 생성자에서 new로 할당한 MCPAT_Router(router)와 interconnect(link_bus) 객체를
 * 안전하게 해제한다. nullptr 체크 후 delete하고 포인터를 0으로 재설정하여
 * 이중 해제(double-free) 위험을 방지한다.
 *
 * 실행 컨텍스트: AccelWattch 종료 단계, 단일 스레드.
 *
 * 호출 체인:
 *   AccelWattch 종료 → [NoC 소멸자] → delete router, delete link_bus
 */
NoC ::~NoC() {
  if (router) {
    // [한국어] router 포인터가 null이 아닌 경우에만 해제 (init_router()가 호출된 경우)
    delete router; // [한국어] MCPAT_Router 객체 해제 (크로스바/중재기/버퍼 포함)
    router = 0;    // [한국어] 포인터를 0으로 초기화 — 이중 해제 방지 (NULL 대신 0 사용: C++03 스타일)
  }
  if (link_bus) {
    // [한국어] link_bus 포인터가 null이 아닌 경우에만 해제 (init_link_bus()가 호출된 경우)
    delete link_bus; // [한국어] interconnect 객체 해제 (CACTI Wire 계산 결과 포함)
    link_bus = 0;    // [한국어] 포인터를 0으로 초기화 — 이중 해제 방지
  }
}
