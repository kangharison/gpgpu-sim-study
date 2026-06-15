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

/*
 * [한국어 설명] AccelWattch 온칩 인터커넥트 전력/지연 모델 구현 (interconnect.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPU/CPU 칩 내부의 금속 배선(Metal Wire) 하나를 모델링하여 전파 지연(delay),
 * 동적 전력(dynamic power), 누설 전력(leakage power), 면적(area)을 계산한다.
 * AccelWattch(GPGPU-Sim의 전력 모델, MICRO 2021)에서 NoC 링크, 버스, 또는 기타 온칩
 * 인터커넥트의 전력 소비를 추정하는 데 사용된다. CACTI의 Wire 모델(wire.h)을 내부적으로
 * 활용하며, 배선 폭(width_scaling)과 간격(space_scaling)을 조정하여 타이밍 제약을 충족하도록
 * 자동 최적화(opt_for_clk)를 수행한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델의 인터커넥트 계층에 해당하며, NoC(noc.cc)가 내부적으로
 * `interconnect` 객체를 생성하여 링크 전력을 계산할 때 직접 사용한다.
 * 호출 체인: noc.cc의 init_link_bus() → [interconnect 생성자] → compute() → CACTI Wire
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, GPGPU-Sim 시뮬레이션 초기화 단계(사이클 루프 밖)에서
 * 단 한 번 실행되며, 이후 결과는 noc.cc가 참조하여 전력을 누적한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존: interconnect.h (클래스 선언), wire.h (CACTI Wire 모델), globalvar.h (g_tp 전역 기술
 * 파라미터), parameter.h (init_interface(), longer_channel_device_reduction()).
 * 의존받음: noc.cc (init_link_bus()가 interconnect 객체를 생성), NoC::computeEnergy()가
 * interconnect::power를 참조하여 링크 전력을 계산.
 * 데이터 흐름: InputParameter(기술 노드·배선 레이어 설정) → interconnect 생성자 →
 * CACTI Wire 모델(RC 지연 계산) → delay/power/area 필드 → noc.cc로 노출.
 *
 * === 주요 함수/구조체 요약 ===
 * interconnect() : 배선 기하 파라미터를 초기화하고 compute()를 호출하여 지연·전력을 계산.
 *                  opt_for_clk 활성화 시 타이밍 제약을 충족할 때까지 배선을 2배씩 확장.
 *                  pipelinable 배선은 처리율(throughput) 위반 시 파이프라인 스테이지를 추가.
 * compute()      : CACTI Wire 임시 객체를 생성하고 지연·전력·면적을 this에 복사한다.
 * leakage_feedback() : 온도 변화 시 init_interface()→compute() 순으로 누설 전력을 재계산한다.
 */

#include "interconnect.h" // [한국어] interconnect 클래스 선언 (필드, 생성자 시그니처)
#include <assert.h>       // [한국어] assert() — 전력/면적 값이 양수임을 런타임에 보장
#include <iostream>       // [한국어] cout — 타이밍 제약 위반 경고 메시지 출력
#include "globalvar.h"    // [한국어] g_tp (TechnologyParameter) 전역 기술 파라미터 접근
#include "wire.h"         // [한국어] CACTI Wire 모델 — RC 지연, 동적/누설 전력 계산 핵심

/*
 * [한국어]
 * interconnect::interconnect() - 온칩 배선 인터커넥트 초기화 및 전력/지연 최적화 생성자
 *
 * @name_           : 이 인터커넥트의 이름 문자열 (예: "NOC", "Links")
 * @device_ty_      : 디바이스 유형 (Core_device / Uncore_device) — 누설 보정 계수 결정
 * @base_w          : 기반 구조물(라우터 등)의 폭 [m] — 현재는 직접 사용되지 않음
 * @base_h          : 기반 구조물의 높이 [m] — 현재는 직접 사용되지 않음
 * @data_w          : 버스 비트 폭 (flit_size) — 단일 비트 전력을 이 값으로 곱해 전체 버스 전력 산출
 * @len             : 배선 길이 [m] — RC 지연과 전력의 핵심 입력값
 * @configure_interface : InputParameter 포인터 — 기술 노드, 배선 레이어, 타이밍 제약 포함
 * @start_wiring_level_ : 신호 배선을 시작할 금속 레이어 번호 (예: 3 = 글로벌 레이어)
 * @pipelinable_    : true이면 처리율(throughput) 기준으로 최적화; false이면 지연(latency) 기준
 * @route_over_perc_: 0.0~1.0 — 배선이 소자 위를 지나는 비율 (면적 계산에 사용)
 * @opt_local_      : true이면 local opt_for_clk 루프를 수행
 * @core_ty_        : 코어 유형 (OOO_core / Inorder_core) — longer_channel 보정에 사용
 * @wire_model      : Wire_type enum — Global/Semi-global/Global_30 등 배선 레이어 종류
 * @width_s         : 초기 배선 폭 스케일링 계수 (1.0 = 최소 폭)
 * @space_s         : 초기 배선 간격 스케일링 계수 (1.0 = 최소 간격)
 * @dt              : DeviceType 포인터 — n_to_p_eff_curr_drv_ratio로 min_w_pmos 계산
 * @return          : (생성자) — power/area/delay 필드에 결과가 저장됨
 *
 * 이 생성자는 CACTI Wire 모델을 통해 주어진 배선 파라미터의 RC 지연과 전력을 계산한다.
 * opt_for_clk이 활성화된 경우, 타이밍 제약(latency 또는 throughput)을 만족할 때까지
 * width_scaling과 space_scaling을 2배씩 늘리는 최적화 루프를 수행한다.
 * 배선을 넓히면 저항이 줄어 RC 지연이 줄어들지만 면적과 전력은 증가한다.
 * pipelinable 배선이 여전히 처리율을 충족하지 못하면 파이프라인 스테이지를 추가한다.
 * 최적화 완료 후 data_width를 곱해 단일 비트 전력을 전체 버스 전력으로 확장하고,
 * socket coefficient(sckt_co_eff)와 longer_channel_device 보정을 적용한다.
 *
 * 실행 컨텍스트: GPGPU-Sim 초기화 단계(사이클 루프 시작 전), 단일 스레드에서 호출.
 *
 * 호출 체인:
 *   noc.cc::init_link_bus() → [interconnect 생성자] → compute() → CACTI Wire()
 *                                                     → Wire winit() (스케일 업 후 재설정)
 *                                                     → Wire wreset() (전역 배선 상태 초기화)
 */
interconnect::interconnect(string name_, enum Device_ty device_ty_,
                           double base_w, double base_h, int data_w, double len,
                           const InputParameter *configure_interface,
                           int start_wiring_level_, bool pipelinable_,
                           double route_over_perc_, bool opt_local_,
                           enum Core_type core_ty_, enum Wire_type wire_model,
                           double width_s, double space_s,
                           TechnologyParameter::DeviceType *dt)
    : name(name_),             // [한국어] 인터커넥트 이름 (예: "NOC") — displayEnergy 출력에 사용
      device_ty(device_ty_),   // [한국어] 디바이스 유형 — longer_channel 보정 계수 선택에 사용
      in_rise_time(0),         // [한국어] 입력 라이즈 타임 [s] — 현재 초기값 0 (미사용)
      out_rise_time(0),        // [한국어] 출력 라이즈 타임 [s] — 현재 초기값 0 (미사용)
      base_width(base_w),      // [한국어] 기반 구조물 폭 [m] — 면적 레이아웃 참조용
      base_height(base_h),     // [한국어] 기반 구조물 높이 [m] — 면적 레이아웃 참조용
      data_width(data_w),      // [한국어] 버스 데이터 폭 [비트] — 단일 비트→전체 버스 전력 스케일 인수
      wt(wire_model),          // [한국어] 배선 유형 (Global/Semi-global 등) — CACTI Wire에 전달
      width_scaling(width_s),  // [한국어] 배선 폭 스케일 계수 — 최적화 루프에서 2배씩 증가
      space_scaling(space_s),  // [한국어] 배선 간격 스케일 계수 — width_scaling과 동기화하여 증가
      start_wiring_level(start_wiring_level_), // [한국어] 신호 시작 배선 레이어 번호 (3=글로벌)
      length(len),             // [한국어] 배선 길이 [m] — RC 지연에 제곱으로 비례
      // interconnect_latency(1e-12),  // [한국어] (주석 처리됨) 예비 필드
      // interconnect_throughput(1e-12),
      opt_local(opt_local_),   // [한국어] 로컬 타이밍 최적화 활성 여부
      core_ty(core_ty_),       // [한국어] 코어 유형 — longer_channel 누설 보정 시 사용
      pipelinable(pipelinable_), // [한국어] true=처리율 기준 최적화, false=지연 기준 최적화
      route_over_perc(route_over_perc_), // [한국어] 배선이 소자 위를 통과하는 비율 (0.0~1.0)
      deviceType(dt) {         // [한국어] 디바이스 기술 파라미터 포인터 — min_w_pmos 계산용

  wt = Global; // [한국어] 배선 유형을 글로벌 레이어(Global)로 강제 설정
               //          인수로 받은 wire_model을 무시하고 Global로 고정함.
               //          이는 GPU의 NoC 링크가 통상 상위 금속 레이어를 사용하기 때문

  l_ip = *configure_interface; // [한국어] InputParameter 복사 — 기술 노드, 타이밍 제약(latency/throughput),
                                //          배선 레이어 설정을 로컬 필드 l_ip에 저장

  local_result = init_interface(&l_ip); // [한국어] 전역 기술 파라미터(g_tp)를 l_ip 기반으로 초기화.
                                         //          g_tp.min_w_nmos_, sckt_co_eff 등이 이 호출로 설정됨.
                                         //          반환값 local_result는 uca_org_t (더미 — 이후 미사용)

  max_unpipelined_link_delay = 0;  // [한국어] 비파이프라인 링크 최대 지연 초기화 — TODO로 미구현
  min_w_nmos = g_tp.min_w_nmos_;  // [한국어] NMOS 최소 트랜지스터 폭 [m] — g_tp에서 읽음 (기술 노드 의존)
  min_w_pmos = deviceType->n_to_p_eff_curr_drv_ratio * min_w_nmos;
  // [한국어] PMOS 최소 트랜지스터 폭 = n_to_p 전류 구동비 × NMOS 최소폭.
  //          PMOS는 동일 전류를 위해 NMOS보다 넓어야 하므로 비율(통상 2.0~2.5)을 곱함.

  latency = l_ip.latency;         // [한국어] 지연 제약 [s] — Non-pipelinable 배선의 타이밍 목표
  throughput = l_ip.throughput;   // [한국어] 처리율 제약 [s/cycle] — Pipelinable 배선의 타이밍 목표
  latency_overflow = false;       // [한국어] 지연 제약 위반 플래그 초기화 (false=정상)
  throughput_overflow = false;    // [한국어] 처리율 제약 위반 플래그 초기화 (현재 미사용)

  /*
   * TODO: Add wiring option from semi-global to global automatically
   * And directly jump to global if semi-global cannot satisfy timing
   * Fat wires only available for global wires, thus
   * if signal wiring layer starts from semi-global,
   * the next layer up will be global, i.e., semi-global does
   * not have fat wires.
   */
  if (pipelinable == false)
  // Non-pipelinable wires, such as bypass logic, care latency
  {
    // [한국어] Non-pipelinable 경로 (우회 논리, bypass 등): 전파 지연(latency) 기준으로 최적화.
    //          단일 사이클 내에 신호가 도달해야 하므로 지연이 제약보다 작아야 한다.
    compute(); // [한국어] 현재 width_scaling/space_scaling으로 CACTI Wire 모델 실행 → delay/power 산출
    if (opt_for_clk && opt_local) {
      // [한국어] opt_for_clk: 전역 클럭 최적화 플래그 (globalvar.h). opt_local: 로컬 최적화 허용.
      //          두 조건이 모두 true일 때만 자동 배선 확장을 수행한다.
      while (delay > latency && width_scaling < 3.0) {
        // [한국어] 현재 지연이 목표 지연보다 크고, 배선 폭 스케일이 3.0배 미만인 동안 반복.
        //          3.0 상한은 과도한 면적 증가를 방지하기 위한 실용적 한계값이다.
        width_scaling *= 2; // [한국어] 배선 폭을 2배로 늘림 → 저항 감소 → RC 지연 감소
        space_scaling *= 2; // [한국어] 배선 간격을 2배로 늘림 → 커플링 커패시턴스 감소 → 전파 지연 개선
        Wire winit(width_scaling, space_scaling);
        // [한국어] CACTI Wire의 임시 객체를 새 스케일링 파라미터로 생성.
        //          이 생성자 호출이 전역 배선 상태(g_ip의 wire 관련 설정)를 업데이트하는
        //          부수 효과(side-effect)를 가진다. 즉시 소멸되지만 전역 상태 변경이 목적.
        compute(); // [한국어] 새 배선 폭/간격으로 지연·전력 재계산
      }
      if (delay > latency) {
        // [한국어] 최대 스케일(3.0배)까지 늘려도 여전히 지연 제약을 충족하지 못하는 경우
        latency_overflow = true; // [한국어] 지연 오버플로 플래그 설정 — 생성자 끝에서 경고 출력
      }
    }
  } else  // Pipelinable wires, such as bus, does not care latency but
          // throughput
  {
    /*
     * TODO: Add pipe regs power, area, and timing;
     * Pipelinable wires optimize latency first.
     */
    // [한국어] Pipelinable 경로 (버스, NoC 링크 등): 처리율(throughput) 기준으로 최적화.
    //          매 사이클 새 데이터가 전송되므로 단위 처리 지연이 throughput보다 작으면 된다.
    compute(); // [한국어] 초기 wire 파라미터로 지연·전력 계산
    if (opt_for_clk && opt_local) {
      // [한국어] 클럭 최적화 활성 시 처리율 제약 충족을 위해 배선 확장 루프 수행
      while (delay > throughput && width_scaling < 3.0) {
        // [한국어] 현재 지연이 목표 처리율보다 크면 배선 폭/간격을 2배씩 확장
        width_scaling *= 2; // [한국어] 배선 폭 2배 확장 → 저항 감소
        space_scaling *= 2; // [한국어] 배선 간격 2배 확장 → 커플링 커패시턴스 감소
        Wire winit(width_scaling, space_scaling);
        // [한국어] 전역 CACTI wire 파라미터를 새 스케일로 업데이트하는 부수 효과 목적의 임시 객체
        compute(); // [한국어] 새 스케일로 지연·전력 재계산
      }
      if (delay > throughput)
      // insert pipeline stages
      {
        // [한국어] 배선 확장으로도 처리율 제약을 충족하지 못할 경우, 파이프라인 스테이지를 삽입.
        //          배선을 여러 단계로 나누면 각 단계의 지연이 줄어든다 (레지스터 추가 필요).
        num_pipe_stages = (int)ceil(delay / throughput);
        // [한국어] 필요한 파이프라인 스테이지 수 = ceil(총 지연 / 목표 처리율).
        //          예: delay=2ns, throughput=0.8ns → ceil(2.5) = 3 스테이지.
        assert(num_pipe_stages > 0); // [한국어] 스테이지 수가 0이면 논리 오류 — 런타임 검증
        delay = delay / num_pipe_stages + num_pipe_stages * 0.05 * delay;
        // [한국어] 파이프라인 적용 후 유효 지연 재계산.
        //          delay/num_pipe_stages: 스테이지당 배선 지연.
        //          + num_pipe_stages * 0.05 * delay: 파이프라인 레지스터(플립플롭)의 오버헤드.
        //          0.05(5%)는 레지스터 타이밍 오버헤드를 모델링하는 경험적 상수.
      }
    }
  }

  power_bit = power; // [한국어] 단일 비트(1-bit) 배선의 전력을 power_bit에 저장.
                     //          이후 data_width를 곱하기 전의 값이므로 per-bit 기준임.
                     //          NoC에서 per-bit 전력을 참조할 때 사용 가능.

  power.readOp.dynamic *= data_width;     // [한국어] 동적 전력을 버스 폭(flit_size 비트)으로 스케일.
                                           //          1비트 배선 전력 × data_width = 전체 버스 동적 전력.
  power.readOp.leakage *= data_width;     // [한국어] 서브스레솔드 누설 전력을 버스 폭으로 스케일
  power.readOp.gate_leakage *= data_width; // [한국어] 게이트 누설 전력을 버스 폭으로 스케일
  area.set_area(area.get_area() * data_width); // [한국어] 배선 면적을 버스 폭만큼 복제 (병렬 배선 면적)
  no_device_under_wire_area.h *= data_width;
  // [한국어] 소자가 없는 배선 하부 면적의 높이를 data_width배로 스케일.
  //          no_device_under_wire_area는 배선 아래에 트랜지스터를 배치할 수 없는 영역을 나타냄.
  //          route_over_perc 계산에서 "배선이 소자 위를 지나는 면적"과 보완 관계.

  if (latency_overflow == true)
    cout << "Warning: " << name
         << " wire structure cannot satisfy latency constraint." << endl;
  // [한국어] 타이밍 제약 위반 경고 출력. 시뮬레이션 결과는 제약 위반 상태로도 계속 진행된다.

  assert(power.readOp.dynamic > 0);      // [한국어] 동적 전력이 반드시 양수여야 함 — 0이면 계산 오류
  assert(power.readOp.leakage > 0);      // [한국어] 누설 전력이 반드시 양수여야 함
  assert(power.readOp.gate_leakage > 0); // [한국어] 게이트 누설 전력이 반드시 양수여야 함

  double long_channel_device_reduction =
      longer_channel_device_reduction(device_ty, core_ty);
  // [한국어] longer_channel_device_reduction(): 롱채널 소자를 사용하는 경우 서브스레솔드 누설이
  //          줄어드는 계수를 반환한다 (0.0~1.0). device_ty와 core_ty에 따라 달라진다.
  //          XML->sys.longer_channel_device가 true일 때 누설 전력 보정에 사용.

  double sckRation = g_tp.sckt_co_eff;
  // [한국어] sckt_co_eff(소켓 계수): 칩 패키징/소켓에 의한 추가 전력 오버헤드 계수.
  //          통상 1.0보다 약간 큰 값 — 배선 끝의 드라이버/수신기 임피던스 매칭 손실을 반영.
  power.readOp.dynamic *= sckRation;  // [한국어] 동적 전력에 소켓 계수 적용
  power.writeOp.dynamic *= sckRation; // [한국어] 쓰기 동적 전력에 소켓 계수 적용 (대칭적 처리)
  power.searchOp.dynamic *= sckRation;// [한국어] 검색 동적 전력에 소켓 계수 적용 (캐시 TAG 검색용)

  power.readOp.longer_channel_leakage =
      power.readOp.leakage * long_channel_device_reduction;
  // [한국어] 롱채널 소자 사용 시 보정된 누설 전력 저장.
  //          displayEnergy()에서 long_channel 플래그가 true이면 leakage 대신 이 값을 출력.
  //          longer_channel_device_reduction이 1.0이면 보정 없음, 0.5이면 누설이 절반.

  if (pipelinable)  // Only global wires has the option to choose whether
                    // routing over or not
    area.set_area(area.get_area() * route_over_perc +
                  no_device_under_wire_area.get_area() * (1 - route_over_perc));
  // [한국어] Pipelinable(글로벌) 배선의 최종 면적 = 소자 위 배선 면적 + 소자 없는 배선 면적의 혼합.
  //          route_over_perc: 배선이 소자 위를 지나는 비율(0=전부 소자 없는 영역, 1=전부 소자 위).
  //          소자 위를 지나면 레이아웃 면적이 줄어드므로 이 가중합으로 실제 점유 면적을 추정.
  //          Non-pipelinable(로컬) 배선은 이 조정을 적용하지 않음.

  Wire wreset();
  // [한국어] Wire 기본 생성자를 호출하여 전역 배선 상태를 초기화한다.
  //          이는 원본 McPAT/CACTI 코드의 알려진 버그/관용구(quirk)로,
  //          최적화 루프에서 Wire winit()가 변경한 전역 기술 파라미터(g_ip wire 관련 설정)를
  //          기본값으로 되돌리기 위해 의도적으로 호출한다.
  //          wreset은 사실 로컬 변수처럼 보이지만 컴파일러는 이를 함수 선언으로 해석할 수 있음.
  //          (Clang/GCC에서는 "most vexing parse"로 처리될 수 있으나 원본 코드 그대로 유지)
}

/*
 * [한국어]
 * interconnect::compute() - CACTI Wire 임시 객체를 생성하여 지연·전력·면적을 this에 복사
 *
 * @return: void — 결과는 this->delay, this->power, this->area, this->no_device_under_wire_area에 저장
 *
 * CACTI의 Wire 클래스를 통해 현재 배선 파라미터(wt, length, width_scaling, space_scaling)를
 * 기반으로 RC 지연, 동적/누설 전력, 면적을 계산한다. Wire 객체는 계산 후 즉시 삭제된다.
 * no_device_under_wire_area는 배선 폭 + 배선 간격의 높이와 배선 길이의 곱으로 산출된다.
 * 이 함수는 생성자의 최적화 루프와 leakage_feedback()에서 반복 호출될 수 있다.
 *
 * 실행 컨텍스트: interconnect 생성자 및 leakage_feedback() 내에서만 호출 (단일 스레드).
 *
 * 호출 체인:
 *   interconnect() → [compute()] → CACTI Wire(wt, length, 1, width_scaling, space_scaling)
 *   interconnect::leakage_feedback() → [compute()]
 */
void interconnect::compute() {
  Wire *wtemp1 = 0; // [한국어] CACTI Wire 임시 포인터 초기화 — 이후 동적 할당
  wtemp1 = new Wire(wt, length, 1, width_scaling, space_scaling);
  // [한국어] CACTI Wire 객체 생성: 배선 유형(wt), 길이(length), 레인 수(1),
  //          폭 스케일(width_scaling), 간격 스케일(space_scaling)을 이용해
  //          RC 기반 지연(delay), 동적 전력(power.readOp.dynamic), 누설 전력,
  //          면적(area), 배선 폭(wire_width), 배선 간격(wire_spacing)을 계산.

  delay = wtemp1->delay; // [한국어] 배선 전파 지연 [s] 복사 — 타이밍 최적화 루프의 판단 기준
  power.readOp.dynamic = wtemp1->power.readOp.dynamic;
  // [한국어] 단일 비트 배선의 동적 에너지 [J/toggle] 복사 — 이후 data_width로 스케일됨
  power.readOp.leakage = wtemp1->power.readOp.leakage;
  // [한국어] 단일 비트 배선의 서브스레솔드 누설 전력 [W] 복사
  power.readOp.gate_leakage = wtemp1->power.readOp.gate_leakage;
  // [한국어] 단일 비트 배선의 게이트 누설 전력 [W] 복사

  area.set_area(wtemp1->area.get_area()); // [한국어] 단일 비트 배선 면적 [m²] 복사
  no_device_under_wire_area.h = (wtemp1->wire_width + wtemp1->wire_spacing);
  // [한국어] 소자 없는 영역의 높이 = 배선 폭 + 배선 간격 [m].
  //          인접 배선 사이의 소자 배치 불가 영역 높이를 나타냄.
  //          (wire_width + wire_spacing) = 배선 피치(pitch) 개념에 해당.
  no_device_under_wire_area.w = length;
  // [한국어] 소자 없는 영역의 너비 = 배선 길이 [m].
  //          no_device_under_wire_area = (wire_width + wire_spacing) × length 의 사각형.

  if (wtemp1) delete wtemp1; // [한국어] Wire 임시 객체 해제 — 결과는 이미 this에 복사됨
}

/*
 * [한국어]
 * interconnect::leakage_feedback() - 온도 변화 반영 후 누설 전력 재계산
 *
 * @temperature: 새로운 동작 온도 [K] — 온도에 따라 누설 전력이 지수적으로 변함
 * @return: void — 결과는 this->power 필드를 직접 업데이트
 *
 * AccelWattch는 온도-전력-온도 피드백 루프를 지원하며, 이 함수는 그 루프에서
 * 온도가 변경될 때마다 인터커넥트의 누설 전력을 재계산하는 역할을 한다.
 * 온도를 10K 단위로 반올림한 후 init_interface()로 전역 기술 파라미터를 갱신하고,
 * compute()를 재호출하여 새 온도 기반 전력을 산출한다.
 * 이후 data_width 스케일링과 socket coefficient, longer_channel 보정을 재적용한다.
 * 단, 면적과 num_pipe_stages는 재계산하지 않으며 타이밍 최적화도 수행하지 않는다.
 *
 * 실행 컨텍스트: 온도 피드백 루프에서 반복 호출 (단일 스레드).
 *
 * 호출 체인:
 *   (외부 온도 피드백 루프) → [leakage_feedback()] → init_interface() → compute()
 */
void interconnect::leakage_feedback(double temperature) {
  l_ip.temp = (unsigned int)round(temperature / 10.0) * 10;
  // [한국어] 온도를 10K 단위로 반올림하여 l_ip.temp에 설정.
  //          CACTI의 룩업 테이블은 10K 간격으로 구성되어 있으므로 반올림이 필요.
  //          예: 325K → 330K, 304K → 300K.

  uca_org_t init_result = init_interface(&l_ip);  // init_result is dummy
  // [한국어] 갱신된 온도로 전역 기술 파라미터(g_tp)를 재초기화.
  //          온도 의존적 파라미터(누설 전류, 임계 전압 등)가 업데이트됨.
  //          반환값 init_result는 사용되지 않음 (주석에 dummy라고 명시).

  compute();
  // [한국어] 갱신된 온도 파라미터로 배선 전력(delay/power/area) 재계산.
  //          Wire 객체 내부에서 온도 의존적 누설 모델을 재적용한다.

  power_bit = power; // [한국어] 재계산된 단일 비트 전력을 power_bit에 저장 (data_width 스케일 전)
  power.readOp.dynamic *= data_width;      // [한국어] 동적 전력을 버스 폭으로 재스케일
  power.readOp.leakage *= data_width;      // [한국어] 누설 전력을 버스 폭으로 재스케일
  power.readOp.gate_leakage *= data_width; // [한국어] 게이트 누설을 버스 폭으로 재스케일

  assert(power.readOp.dynamic > 0);       // [한국어] 재계산 후 동적 전력이 양수임을 검증
  assert(power.readOp.leakage > 0);       // [한국어] 재계산 후 누설 전력이 양수임을 검증
  assert(power.readOp.gate_leakage > 0);  // [한국어] 재계산 후 게이트 누설이 양수임을 검증

  double long_channel_device_reduction =
      longer_channel_device_reduction(device_ty, core_ty);
  // [한국어] 롱채널 소자 보정 계수 재계산 — 온도 변화가 longer_channel 계수에 영향을 줄 수 있음

  double sckRation = g_tp.sckt_co_eff;
  // [한국어] 소켓 계수 재로드 — 온도 재초기화 후 g_tp가 업데이트됐을 수 있으므로 재참조
  power.readOp.dynamic *= sckRation;   // [한국어] 동적 전력에 소켓 계수 재적용
  power.writeOp.dynamic *= sckRation;  // [한국어] 쓰기 동적 전력에 소켓 계수 재적용
  power.searchOp.dynamic *= sckRation; // [한국어] 검색 동적 전력에 소켓 계수 재적용

  power.readOp.longer_channel_leakage =
      power.readOp.leakage * long_channel_device_reduction;
  // [한국어] 롱채널 보정된 누설 전력 재계산 — 온도 변화 후 displayEnergy()에서 참조할 수 있음
}
