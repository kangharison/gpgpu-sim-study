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
 * [한국어 설명] 반복기 삽입 글로벌 와이어 모델 구현 (wire.cc)
 *
 * === 파일의 역할 ===
 * 칩 내 글로벌 배선(wire)에 최적 반복기(repeater/buffer)를 삽입했을 때의
 * 지연(delay), 동적/누설 전력(power), 면적(area)을 계산한다.
 * 배선 종류(Global, Global_5/10/20/30%, Low_swing)와 배치 위치(outside_mat,
 * inside_mat, local)에 따라 커패시턴스·저항을 산정하고, delay_optimal_wire()로
 * 최소 지연 조건의 반복기 크기와 간격을 구한다. init_wire()는 이 결과를 바탕으로
 * 5%~30% 지연 페널티를 허용하는 전력 최적 구성(global_5...global_30)을 도출한다.
 * H-tree 배선(htree2.cc), NUCA 뱅크 간 배선(nuca.cc), 라우터 링크(router.cc)에서
 * 광범위하게 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch → CACTI → Wire [이 파일].
 * cacti_interface() 초기화 시 Wire(초기화 생성자)가 정적 멤버(global, global_5,
 * ..., low_swing)를 미리 계산해 두고, 이후 모든 Wire 인스턴스는 길이에 따라
 * 이 기준값을 곱하여 지연/전력을 산출한다.
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, AccelWattch 초기화 단계.
 *
 * === 타 모듈과의 연결 ===
 * 의존:
 *   - wire.h: Wire 클래스 선언, Wire_type/Wire_placement enum, 정적 멤버
 *   - component.h/.cc: Component 기반 클래스 — area, power, delay 필드
 *   - basic_circuit.h: gate_C, drain_C_, tr_R_on, horowitz, R_to_w
 *   - parameter.h: g_ip(F_sz_um, wt, wire_* 파라미터), g_tp(wire_outside/inside/local)
 * 소비:
 *   - htree2.cc: H-tree 링크로 Wire 생성
 *   - nuca.cc: NUCA 뱅크 간 글로벌 배선 모델
 *   - router.cc: NoC 라우터 납선 커패시턴스/저항
 *
 * AccelWattch XML / gpgpusim.config 연동:
 *   - XML의 technology 노드(feature size, Vdd, 배선 층 두께/유전율 등)가
 *     g_tp.wire_outside_mat / wire_inside_mat / wire_local에 반영된다.
 *   - g_ip->wt(Wire_type)와 g_ip->wire_is_mat_type 등의 설정이
 *     calculate_wire_stats()의 분기(Global/Low_swing)를 결정한다.
 *   - g_ip->F_sz_um이 기술 노드이며, wire_width/wire_spacing의 절대값 계산에 사용된다.
 *
 * === 주요 함수/구조체 요약 ===
 * - Wire(wire_model, len) : 주 생성자 — 길이 기준으로 지연/전력/면적 산출
 * - Wire(w_s, s_s)        : 초기화 전용 생성자 — 정적 멤버 global/low_swing 등을 1회 계산
 * - calculate_wire_stats() : 배선 타입별로 per-meter 기준값을 길이만큼 스케일링
 * - delay_optimal_wire()   : 지연 최소 반복기 크기/간격과 그 때의 전력 계산
 * - wire_cap()             : 측벽·인접층·프린지 커패시턴스 합산 [F]
 * - wire_res()             : 도체 저항(스캐터/배리어/디싱 보정) 계산 [Ohm]
 * - low_swing_model()      : 차동 저전압 스윙 배선의 송신기+배선+감지증폭기 모델
 * - init_wire()            : 반복기 크기/간격 탐색 후 global/global_5~30/low_swing 설정
 * - update_fullswing()     : 지연 페널티 상한 내 최저전력 반복기 구성 선택
 * - wire_model()           : 주어진 반복기 간격/크기의 1m 기준 지연·전력 계산
 * - print_wire()           : 와이어 특성 출력 (디버그/리포트용)
 */

#include "wire.h"
#include "cmath"
// use this constructor to calculate wire stats

/*
 * [한국어]
 * Wire::Wire - 일반 생성자: 주어진 길이와 타입의 배선 통계 계산
 *
 * @wire_model: 배선 모델 종류 (Global/Global_5/10/20/30/Low_swing)
 * @wl        : 배선 길이 [um]
 * @n         : low-swing 배선에 연결된 감지증폭기 수
 * @w_s, @s_s : 폭/간격 스케일링 계수
 * @wp        : 배선 배치 위치 (outside_mat/inside_mat/local)
 * @_resistivity: 도체 저항률 (기본값 CU_RESISTIVITY)
 * @dt        : DeviceType 포인터 (기본 g_tp.peri_global)
 * @return    : (생성자, 반환값 없음)
 *
 * 정적 멤버가 초기화되어 있지 않으면 기본값으로 Wire 초기화 생성자를 호출한다.
 * calculate_wire_stats()를 통해 Global/Low_swing 모델별 지연/전력/면적을
 * 계산한 뒤, 최종 출력 단위(um, s, J)로 환산하여 저장한다.
 *
 * 호출 체인: Htree2/Nuca/Router → [Wire::Wire()] → calculate_wire_stats()
 */
Wire::Wire(
    enum Wire_type wire_model,
    double wl,
    int n,
    double w_s,
    double s_s,
    enum Wire_placement wp,
    double _resistivity,
    TechnologyParameter::DeviceType *dt
    ):wt(wire_model), wire_length(wl*1e-6), nsense(n), w_scale(w_s), s_scale(s_s),
    resistivity(_resistivity), deviceType(dt)
{

  wire_placement = wp;
  min_w_pmos     = deviceType->n_to_p_eff_curr_drv_ratio*g_tp.min_w_nmos_;
  in_rise_time   = 0;
  out_rise_time  = 0;
  if (initialized != 1) {
    cout << "Wire not initialized. Initializing it with default values\n";
    Wire winit;
  }
  calculate_wire_stats();
  // change everything back to seconds, microns, and Joules
  repeater_spacing *= 1e6;
  wire_length      *= 1e6;
  wire_width       *= 1e6;
  wire_spacing     *= 1e6;
  assert(wire_length > 0); // [한국어] 설계 가정/단위 변환 결과 검증
  assert(power.readOp.dynamic > 0); // [한국어] 설계 가정/단위 변환 결과 검증
  assert(power.readOp.leakage > 0); // [한국어] 설계 가정/단위 변환 결과 검증
  assert(power.readOp.gate_leakage > 0); // [한국어] 설계 가정/단위 변환 결과 검증
}

    // the following values are for peripheral global technology
    // specified in the input config file
    Component Wire::global;
    Component Wire::global_5;
    Component Wire::global_10;
    Component Wire::global_20;
    Component Wire::global_30;
    Component Wire::low_swing;

    int Wire::initialized;
    double Wire::wire_width_init;
    double Wire::wire_spacing_init;



/*
 * [한국어]
 * Wire::Wire - 일반 생성자: 주어진 길이와 타입의 배선 통계 계산
 *
 * @wire_model: 배선 모델 종류 (Global/Global_5/10/20/30/Low_swing)
 * @wl        : 배선 길이 [um]
 * @n         : low-swing 배선에 연결된 감지증폭기 수
 * @w_s, @s_s : 폭/간격 스케일링 계수
 * @wp        : 배선 배치 위치 (outside_mat/inside_mat/local)
 * @_resistivity: 도체 저항률 (기본값 CU_RESISTIVITY)
 * @dt        : DeviceType 포인터 (기본 g_tp.peri_global)
 * @return    : (생성자, 반환값 없음)
 *
 * 정적 멤버가 초기화되어 있지 않으면 기본값으로 Wire 초기화 생성자를 호출한다.
 * calculate_wire_stats()를 통해 Global/Low_swing 모델별 지연/전력/면적을
 * 계산한 뒤, 최종 출력 단위(um, s, J)로 환산하여 저장한다.
 *
 * 호출 체인: Htree2/Nuca/Router → [Wire::Wire()] → calculate_wire_stats()
 */
Wire::Wire(double w_s, double s_s, enum Wire_placement wp, double resis, TechnologyParameter::DeviceType *dt)
{
  w_scale        = w_s;
  s_scale        = s_s;
  deviceType     = dt;
  wire_placement = wp;
  resistivity    = resis;
  min_w_pmos     = deviceType->n_to_p_eff_curr_drv_ratio * g_tp.min_w_nmos_;
  in_rise_time   = 0;
  out_rise_time  = 0;

  switch (wire_placement) // [한국어] 배선 배치 위치별 피치/파라미터 선택
  {
    case outside_mat: wire_width = g_tp.wire_outside_mat.pitch; break;
    case inside_mat : wire_width = g_tp.wire_inside_mat.pitch;  break;
    default:          wire_width = g_tp.wire_local.pitch; break;
  }

  wire_spacing = wire_width;

  wire_width   *= (w_scale * 1e-6/2) /* (m) */;
  wire_spacing *= (s_scale * 1e-6/2) /* (m) */;

  initialized = 1;
  init_wire();
  wire_width_init = wire_width;
  wire_spacing_init = wire_spacing;

  assert(power.readOp.dynamic > 0); // [한국어] 설계 가정/단위 변환 결과 검증
  assert(power.readOp.leakage > 0); // [한국어] 설계 가정/단위 변환 결과 검증
  assert(power.readOp.gate_leakage > 0); // [한국어] 설계 가정/단위 변환 결과 검증
}




/*
 * [한국어]
 * Wire::~Wire - 소멸자
 *
 * 동적 할당 자원이 없으므로 빈 구현이다.
 */
Wire::~Wire()
{
}



void

/*
 * [한국어]
 * Wire::calculate_wire_stats - 배선 타입별 지연·전력·면적 산출
 *
 * @return: (void) — Component::delay, power, area 및 repeater_* 필드 갱신
 *
 * 배선 배치 위치에 따른 피치(pitch)를 g_tp에서 가져와 wire_width/ wire_spacing을
 * 설정한 뒤, Global 계열은 정적 멤버(global/global_5/.../global_30)의
 * per-meter 값을 wire_length로 곱하고, Low_swing은 low_swing_model()을 호출한다.
 * Global 계열의 면적은 (길이/반복기간격) × 인버터 게이트 면적으로 계산된다.
 *
 * 호출 체인: Wire::Wire() → [calculate_wire_stats()]
 */
Wire::calculate_wire_stats()
{

  if (wire_placement == outside_mat) {
    wire_width = g_tp.wire_outside_mat.pitch;
  }
  else if (wire_placement == inside_mat) {
    wire_width = g_tp.wire_inside_mat.pitch;
  }
  else {
    wire_width = g_tp.wire_local.pitch;
  }

  wire_spacing = wire_width;

  wire_width   *= (w_scale * 1e-6/2) /* (m) */;
  wire_spacing *= (s_scale * 1e-6/2) /* (m) */;


  if (wt != Low_swing) {

	  //    delay_optimal_wire();

	  if (wt == Global) {
		  delay = global.delay * wire_length; // [한국어] 총 지연 값 갱신 [s]
		  power.readOp.dynamic = global.power.readOp.dynamic * wire_length; // [한국어] 읽기 동작 동적 에너지 누적 [J]
		  power.readOp.leakage = global.power.readOp.leakage * wire_length; // [한국어] 읽기 동작 누설 전력 누적 [W]
		  power.readOp.gate_leakage = global.power.readOp.gate_leakage * wire_length; // [한국어] 읽기 동작 게이트 누설 전력 누적 [W]
		  repeater_spacing = global.area.w; // [한국어] 반복기 간격 설정 [m]
		  repeater_size = global.area.h; // [한국어] 반복기 크기(최소 대비 배율) 설정
		  area.set_area((wire_length/repeater_spacing) *
				  compute_gate_area(INV, 1, min_w_pmos * repeater_size, // [한국어] 반복기 인버터 면적 계산 [m^2]
						  g_tp.min_w_nmos_ * repeater_size, g_tp.cell_h_def));
	  }
	  else if (wt == Global_5) {
		  delay = global_5.delay * wire_length; // [한국어] 총 지연 값 갱신 [s]
		  power.readOp.dynamic = global_5.power.readOp.dynamic * wire_length; // [한국어] 읽기 동작 동적 에너지 누적 [J]
		  power.readOp.leakage = global_5.power.readOp.leakage * wire_length; // [한국어] 읽기 동작 누설 전력 누적 [W]
		  power.readOp.gate_leakage = global_5.power.readOp.gate_leakage * wire_length; // [한국어] 읽기 동작 게이트 누설 전력 누적 [W]
		  repeater_spacing = global_5.area.w; // [한국어] 반복기 간격 설정 [m]
		  repeater_size = global_5.area.h; // [한국어] 반복기 크기(최소 대비 배율) 설정
		  area.set_area((wire_length/repeater_spacing) *
				  compute_gate_area(INV, 1, min_w_pmos * repeater_size, // [한국어] 반복기 인버터 면적 계산 [m^2]
						  g_tp.min_w_nmos_ * repeater_size, g_tp.cell_h_def));
	  }
	  else if (wt == Global_10) {
		  delay = global_10.delay * wire_length; // [한국어] 총 지연 값 갱신 [s]
		  power.readOp.dynamic = global_10.power.readOp.dynamic * wire_length; // [한국어] 읽기 동작 동적 에너지 누적 [J]
		  power.readOp.leakage = global_10.power.readOp.leakage * wire_length; // [한국어] 읽기 동작 누설 전력 누적 [W]
		  power.readOp.gate_leakage = global_10.power.readOp.gate_leakage * wire_length; // [한국어] 읽기 동작 게이트 누설 전력 누적 [W]
		  repeater_spacing = global_10.area.w; // [한국어] 반복기 간격 설정 [m]
		  repeater_size = global_10.area.h; // [한국어] 반복기 크기(최소 대비 배율) 설정
		  area.set_area((wire_length/repeater_spacing) *
				  compute_gate_area(INV, 1, min_w_pmos * repeater_size, // [한국어] 반복기 인버터 면적 계산 [m^2]
						  g_tp.min_w_nmos_ * repeater_size, g_tp.cell_h_def));
	  }
	  else if (wt == Global_20) {
		  delay = global_20.delay * wire_length; // [한국어] 총 지연 값 갱신 [s]
		  power.readOp.dynamic = global_20.power.readOp.dynamic * wire_length; // [한국어] 읽기 동작 동적 에너지 누적 [J]
		  power.readOp.leakage = global_20.power.readOp.leakage * wire_length; // [한국어] 읽기 동작 누설 전력 누적 [W]
		  power.readOp.gate_leakage = global_20.power.readOp.gate_leakage * wire_length; // [한국어] 읽기 동작 게이트 누설 전력 누적 [W]
		  repeater_spacing = global_20.area.w; // [한국어] 반복기 간격 설정 [m]
		  repeater_size = global_20.area.h; // [한국어] 반복기 크기(최소 대비 배율) 설정
		  area.set_area((wire_length/repeater_spacing) *
				  compute_gate_area(INV, 1, min_w_pmos * repeater_size, // [한국어] 반복기 인버터 면적 계산 [m^2]
						  g_tp.min_w_nmos_ * repeater_size, g_tp.cell_h_def));
	  }
	  else if (wt == Global_30) {
		  delay = global_30.delay * wire_length; // [한국어] 총 지연 값 갱신 [s]
		  power.readOp.dynamic = global_30.power.readOp.dynamic * wire_length; // [한국어] 읽기 동작 동적 에너지 누적 [J]
		  power.readOp.leakage = global_30.power.readOp.leakage * wire_length; // [한국어] 읽기 동작 누설 전력 누적 [W]
		  power.readOp.gate_leakage = global_30.power.readOp.gate_leakage * wire_length; // [한국어] 읽기 동작 게이트 누설 전력 누적 [W]
		  repeater_spacing = global_30.area.w; // [한국어] 반복기 간격 설정 [m]
		  repeater_size = global_30.area.h; // [한국어] 반복기 크기(최소 대비 배율) 설정
		  area.set_area((wire_length/repeater_spacing) *
				  compute_gate_area(INV, 1, min_w_pmos * repeater_size, // [한국어] 반복기 인버터 면적 계산 [m^2]
						  g_tp.min_w_nmos_ * repeater_size, g_tp.cell_h_def));
	  }
    out_rise_time = delay*repeater_spacing/deviceType->Vth;
  }
  else if (wt == Low_swing) {
    low_swing_model ();
    repeater_spacing = wire_length; // [한국어] 반복기 간격 설정 [m]
    repeater_size = 1; // [한국어] 반복기 크기(최소 대비 배율) 설정
  }
  else {
    assert(0); // [한국어] 설계 가정/단위 변환 결과 검증
  }
}



/*
 * The fall time of an input signal to the first stage of a circuit is
 * assumed to be same as the fall time of the output signal of two
 * inverters connected in series (refer: CACTI 1 Technical report,
 * section 6.1.3)
 */
  double

/*
 * [한국어]
 * Wire::signal_fall_time - 첫 단 입력 신호의 fall time 계산
 *
 * @return: 2단 인버터 체인 출력의 fall time [s]
 *
 * CACTI 1 기술 보고서 6.1.3절을 참고하여, 최소 크기 인버터 2개를 직렬로
 * 연결했을 때의 출력 fall time을 Horowitz 모델로 근사한다. 입력 rise time과
 * 대칭적으로 동일한 가정이 사용된다.
 *
 * 호출 체인: low_swing_model() 등 → [signal_fall_time()]
 */
Wire::signal_fall_time ()
{

  /* rise time of inverter 1's output */
  double rt;
  /* fall time of inverter 2's output */
  double ft;
  double timeconst;

  timeconst = (drain_C_(g_tp.min_w_nmos_, NCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
      drain_C_(min_w_pmos, PCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
      gate_C(min_w_pmos + g_tp.min_w_nmos_, 0)) * // [한국어] 게이트 커패시턴스 계산 [F]
    tr_R_on(min_w_pmos, PCH, 1); // [한국어] 트랜지스터 온-저항 계산 [Ohm]
  rt = horowitz (0, timeconst, deviceType->Vth/deviceType->Vdd, deviceType->Vth/deviceType->Vdd, FALL) / (deviceType->Vdd - deviceType->Vth);
  timeconst = (drain_C_(g_tp.min_w_nmos_, NCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
      drain_C_(min_w_pmos, PCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
      gate_C(min_w_pmos + g_tp.min_w_nmos_, 0)) * // [한국어] 게이트 커패시턴스 계산 [F]
    tr_R_on(g_tp.min_w_nmos_, NCH, 1); // [한국어] 트랜지스터 온-저항 계산 [Ohm]
  ft = horowitz (rt, timeconst, deviceType->Vth/deviceType->Vdd, deviceType->Vth/deviceType->Vdd, RISE) / deviceType->Vth;
  return ft;
}




/*
 * [한국어]
 * Wire::signal_rise_time - 첫 단 입력 신호의 rise time 계산
 *
 * @return: 2단 인버터 체인 출력의 rise time [s]
 *
 * signal_fall_time()과 대칭적으로, 최소 크기 인버터 2단을 거친 후의
 * 출력 rise time을 계산한다. low_swing_model()에서 입력 rise time이 0일 때
 * 대체값으로 사용된다.
 *
 * 호출 체인: low_swing_model() → [signal_rise_time()]
 */
double Wire::signal_rise_time ()
{

  /* rise time of inverter 1's output */
  double ft;
  /* fall time of inverter 2's output */
  double rt;
  double timeconst;

  timeconst = (drain_C_(g_tp.min_w_nmos_, NCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
      drain_C_(min_w_pmos, PCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
      gate_C(min_w_pmos + g_tp.min_w_nmos_, 0)) * // [한국어] 게이트 커패시턴스 계산 [F]
    tr_R_on(g_tp.min_w_nmos_, NCH, 1); // [한국어] 트랜지스터 온-저항 계산 [Ohm]
  rt = horowitz (0, timeconst, deviceType->Vth/deviceType->Vdd, deviceType->Vth/deviceType->Vdd, RISE) / deviceType->Vth;
  timeconst = (drain_C_(g_tp.min_w_nmos_, NCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
      drain_C_(min_w_pmos, PCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
      gate_C(min_w_pmos + g_tp.min_w_nmos_, 0)) * // [한국어] 게이트 커패시턴스 계산 [F]
    tr_R_on(min_w_pmos, PCH, 1); // [한국어] 트랜지스터 온-저항 계산 [Ohm]
  ft = horowitz (rt, timeconst, deviceType->Vth/deviceType->Vdd, deviceType->Vth/deviceType->Vdd, FALL) / (deviceType->Vdd - deviceType->Vth);
  return ft; //sec
}



/* Wire resistance and capacitance calculations
 *   wire width
 *
 *    /__/
 *   |  |
 *   |  |  height = ASPECT_RATIO*wire width (ASPECT_RATIO = 2.2, ref: ITRS)
 *   |__|/
 *
 *   spacing between wires in same level = wire width
 *   spacing between wires in adjacent levels = wire width---this is incorrect,
 *   according to R.Ho's paper and thesis. ILD != wire width
 *
 */


/*
 * [한국어]
 * Wire::wire_cap - 단위 길이당 배선 총 커패시턴스 계산
 *
 * @len             : 배선 길이 [m]
 * @call_from_outside: true이면 um 단위의 wire_width/wire_spacing을 m로 변환 후 복원
 * @return           : 길이 len에 대한 총 배선 커패시턴스 [F]
 *
 * 측벽(sidewall) 커패시턴스, 인접 ILD 층(adj) 커패시턴스, 그리고 프린지 커패시턴스를
 * 합산한다. Miller 효과와 aspect ratio(ITRS 기준 2.2)를 반영한다.
 *
 * 호출 체인: delay_optimal_wire(), low_swing_model(), wire_model() → [wire_cap()]
 */
double Wire::wire_cap (double len /* in m */, bool call_from_outside)
{
	//TODO: this should be consistent with the wire_res in technology file
  double sidewall, adj, tot_cap;
  double wire_height;
  double epsilon0 = 8.8542e-12;
  double aspect_ratio, horiz_dielectric_constant, vert_dielectric_constant, miller_value,ild_thickness;

  switch (wire_placement) // [한국어] 배선 배치 위치별 피치/파라미터 선택
  {
    case outside_mat:
    	{
    		aspect_ratio = g_tp.wire_outside_mat.aspect_ratio;
    		horiz_dielectric_constant = g_tp.wire_outside_mat.horiz_dielectric_constant;
    		vert_dielectric_constant = g_tp.wire_outside_mat.vert_dielectric_constant;
    		miller_value = g_tp.wire_outside_mat.miller_value;
    		ild_thickness = g_tp.wire_outside_mat.ild_thickness;
    		break;
    	}
    case inside_mat :
    	{
    		aspect_ratio = g_tp.wire_inside_mat.aspect_ratio;
    		horiz_dielectric_constant = g_tp.wire_inside_mat.horiz_dielectric_constant;
    		vert_dielectric_constant = g_tp.wire_inside_mat.vert_dielectric_constant;
    		miller_value = g_tp.wire_inside_mat.miller_value;
    		ild_thickness = g_tp.wire_inside_mat.ild_thickness;
    		break;
    	}
    default:
    	{
    		aspect_ratio = g_tp.wire_local.aspect_ratio;
    		horiz_dielectric_constant = g_tp.wire_local.horiz_dielectric_constant;
    		vert_dielectric_constant = g_tp.wire_local.vert_dielectric_constant;
    		miller_value = g_tp.wire_local.miller_value;
    		ild_thickness = g_tp.wire_local.ild_thickness;
    		break;
    	}
  }

  if (call_from_outside)
  {
	  wire_width       *= 1e-6;
	  wire_spacing     *= 1e-6;
  }
  wire_height = wire_width/w_scale*aspect_ratio;
  /*
   * assuming height does not change. wire_width = width_original*w_scale
   * So wire_height does not change as wire width increases
   */

// capacitance between wires in the same level
//  sidewall = 2*miller_value * horiz_dielectric_constant * (wire_height/wire_spacing)
//    * epsilon0;

  sidewall = miller_value * horiz_dielectric_constant * (wire_height/wire_spacing)
    * epsilon0;


  // capacitance between wires in adjacent levels
  //adj = miller_value * vert_dielectric_constant *w_scale * epsilon0;
  //adj = 2*vert_dielectric_constant *wire_width/(ild_thickness*1e-6) * epsilon0;

  adj = miller_value *vert_dielectric_constant *wire_width/(ild_thickness*1e-6) * epsilon0;
  //Change ild_thickness from micron to M

  //tot_cap =  (sidewall + adj + (deviceType->C_fringe * 1e6)); //F/m
  tot_cap =  (sidewall + adj + (g_tp.fringe_cap * 1e6)); //F/m

  if (call_from_outside)
  {
	  wire_width       *= 1e6;
	  wire_spacing     *= 1e6;
  }
  return (tot_cap*len); // (F)
}


  double

/*
 * [한국어]
 * Wire::wire_res - 단위 길이당 배선 저항 계산
 *
 * @len: 배선 길이 [m]
 * @return: 길이 len에 대한 총 배선 저항 [Ohm]
 *
 * aspect ratio, 도체 저항률, 산란 계수(alpha_scatter), 디싱/배리어 두께를
 * 고려하여 실효 단면적으로 저항을 계산한다.
 *
 * 호출 체인: delay_optimal_wire(), low_swing_model(), wire_model() → [wire_res()]
 */
Wire::wire_res (double len /*(in m)*/)
{

	  double aspect_ratio,alpha_scatter =1.05, dishing_thickness=0, barrier_thickness=0;
	  //TODO: this should be consistent with the wire_res in technology file
	  //The whole computation should be consistent with the wire_res in technology.cc too!

	  switch (wire_placement) // [한국어] 배선 배치 위치별 피치/파라미터 선택
	  {
	  case outside_mat:
	  {
		  aspect_ratio = g_tp.wire_outside_mat.aspect_ratio;
		  break;
	  }
	  case inside_mat :
	  {
		  aspect_ratio = g_tp.wire_inside_mat.aspect_ratio;
		  break;
	  }
	  default:
	  {
		  aspect_ratio = g_tp.wire_local.aspect_ratio;
		  break;
	  }
	  }
	  return (alpha_scatter * resistivity * 1e-6 * len/((aspect_ratio*wire_width/w_scale-dishing_thickness - barrier_thickness)*
			  (wire_width-2*barrier_thickness)));
}

/*
 * Calculates the delay, power and area of the transmitter circuit.
 *
 * The transmitter delay is the sum of nand gate delay, inverter delay
 * low swing nmos delay, and the wire delay
 * (ref: Technical report 6)
 */
  void

/*
 * [한국어]
 * Wire::low_swing_model - 저전압 스윙(low-swing) 차동 배선 모델
 *
 * @return: (void) — transmitter, l_wire, sense_amp의 delay/power/area 갱신
 *
 * 동작 과정:
 *   1) 목표 지연(8FO4) 내에서 NMOS 드라이버 크기(nsize)를 결정한다.
 *   2) NAND → 인버터 → NMOS 드라이버 → 배선 → 감지증폭기 순서로
 *      각 단의 RC 지연을 Horowitz 모델로 누적한다.
 *   3) 차동 신호이므로 동적 에너지에 ×2를 적용하고, overdrive 전압(0.4V) 및
 *      VOL_SWING(0.1)을 반영한다.
 *   4) transmitter, 배선(l_wire), 감지증폭기(sense_amp)의 동적/누설 전력을
 *      분리 저장한 뒤 Component::power에 합산한다.
 *
 * 호출 체인: calculate_wire_stats() → [low_swing_model()]
 */
Wire::low_swing_model() // [한국어] 저전압 스윙 차동 배선 모델 계산
{
  double len = wire_length;
  double beta = pmos_to_nmos_sz_ratio(); // [한국어] PMOS/NMOS 폭 비율 계산


  double inputrise = (in_rise_time == 0) ? signal_rise_time() : in_rise_time; // [한국어] 입력 신호 rise time 계산

  /* Final nmos low swing driver size calculation:
   * Try to size the driver such that the delay
   * is less than 8FO4.
   * If the driver size is greater than
   * the max allowable size, assume max size for the driver.
   * In either case, recalculate the delay using
   * the final driver size assuming slow input with
   * finite rise time instead of ideal step input
   *
   * (ref: Technical report 6)
   */
  double cwire = wire_cap(len); /* load capacitance */
  double rwire = wire_res(len); // [한국어] 배선 저항 계산 [Ohm]

#define RES_ADJ (8.6) // Increase in resistance due to low driving vol.

  double driver_res = (-8*g_tp.FO4/(log(0.5) * cwire))/RES_ADJ;
  double nsize = R_to_w(driver_res, NCH); // [한국어] 저항→트랜지스터 채널 폭 변환

  nsize = MIN(nsize, g_tp.max_w_nmos_);
  nsize = MAX(nsize, g_tp.min_w_nmos_);

  if(rwire*cwire > 8*g_tp.FO4)
  {
    nsize = g_tp.max_w_nmos_;
  }

  // size the inverter appropriately to minimize the transmitter delay
  // Note - In order to minimize leakage, we are not adding a set of inverters to
  // bring down delay. Instead, we are sizing the single gate
  // based on the logical effort.
  double st_eff   = sqrt((2+beta/1+beta)*gate_C(nsize, 0)/(gate_C(2*g_tp.min_w_nmos_, 0) // [한국어] 게이트 커패시턴스 계산 [F]
        + gate_C(2*min_w_pmos, 0))); // [한국어] 게이트 커패시턴스 계산 [F]
  double req_cin  = ((2+beta/1+beta)*gate_C(nsize, 0))/st_eff; // [한국어] 게이트 커패시턴스 계산 [F]
  double inv_size = req_cin/(gate_C(min_w_pmos, 0) + gate_C(g_tp.min_w_nmos_, 0)); // [한국어] 게이트 커패시턴스 계산 [F]
  inv_size = MAX(inv_size, 1);

  /* nand gate delay */
  double res_eq = (2 * tr_R_on(g_tp.min_w_nmos_, NCH, 1)); // [한국어] 트랜지스터 온-저항 계산 [Ohm]
  double cap_eq = 2 * drain_C_(min_w_pmos, PCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
    drain_C_(2*g_tp.min_w_nmos_, NCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
    gate_C(inv_size*g_tp.min_w_nmos_, 0) + // [한국어] 게이트 커패시턴스 계산 [F]
    gate_C(inv_size*min_w_pmos, 0); // [한국어] 게이트 커패시턴스 계산 [F]

  double timeconst = res_eq * cap_eq;

  delay = horowitz(inputrise, timeconst, deviceType->Vth/deviceType->Vdd, // [한국어] Horowitz 모델로 단계 전파 지연 계산 [s]
      deviceType->Vth/deviceType->Vdd, RISE);
  double temp_power = cap_eq*deviceType->Vdd*deviceType->Vdd;

  inputrise = delay / (deviceType->Vdd - deviceType->Vth); /* for the next stage */

  /* Inverter delay:
   * The load capacitance of this inv depends on
   * the gate capacitance of the final stage nmos
   * transistor which in turn depends on nsize
   */
  res_eq = tr_R_on(inv_size*min_w_pmos, PCH, 1); // [한국어] 트랜지스터 온-저항 계산 [Ohm]
  cap_eq = drain_C_(inv_size*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
    drain_C_(inv_size*g_tp.min_w_nmos_, NCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
    gate_C(nsize, 0); // [한국어] 게이트 커패시턴스 계산 [F]
  timeconst = res_eq * cap_eq;

  delay += horowitz(inputrise, timeconst, deviceType->Vth/deviceType->Vdd, // [한국어] Horowitz RC 지연 모델 적용
      deviceType->Vth/deviceType->Vdd, FALL);
  temp_power += cap_eq*deviceType->Vdd*deviceType->Vdd;


  transmitter.delay = delay; // [한국어] 총 지연 값 갱신 [s]
  transmitter.power.readOp.dynamic = temp_power*2; /* since it is a diff. model*/
  transmitter.power.readOp.leakage = deviceType->Vdd * // [한국어] 읽기 동작 누설 전력 누적 [W]
    (4 * cmos_Isub_leakage(g_tp.min_w_nmos_, min_w_pmos, 2, nand) + // [한국어] 서브-임계 누설 전류 계산 [A]
     4 * cmos_Isub_leakage(g_tp.min_w_nmos_, min_w_pmos, 1, inv)); // [한국어] 서브-임계 누설 전류 계산 [A]

  transmitter.power.readOp.gate_leakage = deviceType->Vdd * // [한국어] 읽기 동작 게이트 누설 전력 누적 [W]
    (4 * cmos_Ig_leakage(g_tp.min_w_nmos_, min_w_pmos, 2, nand) + // [한국어] 게이트 절연막 누설 전류 계산 [A]
     4 * cmos_Ig_leakage(g_tp.min_w_nmos_, min_w_pmos, 1, inv)); // [한국어] 게이트 절연막 누설 전류 계산 [A]

  inputrise = delay / deviceType->Vth;

  /* nmos delay + wire delay */
  cap_eq = cwire + drain_C_(nsize, NCH, 1, 1, g_tp.cell_h_def)*2 + // [한국어] 드레인/확산 커패시턴스 계산 [F]
    nsense * sense_amp_input_cap(); //+receiver cap
  /*
   * NOTE: nmos is used as both pull up and pull down transistor
   * in the transmitter. This is because for low voltage swing, drive
   * resistance of nmos is less than pmos
   * (for a detailed graph ref: On-Chip Wires: Scaling and Efficiency)
   */
  timeconst = (tr_R_on(nsize, NCH, 1)*RES_ADJ) * (cwire + // [한국어] 트랜지스터 온-저항 계산 [Ohm]
      drain_C_(nsize, NCH, 1, 1, g_tp.cell_h_def)*2) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
    rwire*cwire/2 +
    (tr_R_on(nsize, NCH, 1)*RES_ADJ + rwire) * // [한국어] 트랜지스터 온-저항 계산 [Ohm]
    nsense * sense_amp_input_cap(); // [한국어] 감지증폭기 입력 커패시턴스 계산 [F]

  /*
   * since we are pre-equalizing and overdriving the low
   * swing wires, the net time constant is less
   * than the actual value
   */
  delay += horowitz(inputrise, timeconst, deviceType->Vth/deviceType->Vdd, .25, 0); // [한국어] Horowitz RC 지연 모델 적용
#define VOL_SWING .1
  temp_power += cap_eq*VOL_SWING*.400; /* .4v is the over drive voltage */
  temp_power *= 2; /* differential wire */

  l_wire.delay = delay - transmitter.delay; // [한국어] 총 지연 값 갱신 [s]
  l_wire.power.readOp.dynamic = temp_power - transmitter.power.readOp.dynamic; // [한국어] 읽기 동작 동적 에너지 누적 [J]
  l_wire.power.readOp.leakage = deviceType->Vdd* // [한국어] 읽기 동작 누설 전력 누적 [W]
    (4* cmos_Isub_leakage(nsize, 0, 1, nmos)); // [한국어] 서브-임계 누설 전류 계산 [A]

  l_wire.power.readOp.gate_leakage = deviceType->Vdd* // [한국어] 읽기 동작 게이트 누설 전력 누적 [W]
    (4* cmos_Ig_leakage(nsize, 0, 1, nmos)); // [한국어] 게이트 절연막 누설 전류 계산 [A]

  //double rt = horowitz(inputrise, timeconst, deviceType->Vth/deviceType->Vdd,
  //    deviceType->Vth/deviceType->Vdd, RISE)/deviceType->Vth;

  delay += g_tp.sense_delay;

  sense_amp.delay = g_tp.sense_delay; // [한국어] 총 지연 값 갱신 [s]
  out_rise_time = g_tp.sense_delay/(deviceType->Vth);
  sense_amp.power.readOp.dynamic = g_tp.sense_dy_power; // [한국어] 읽기 동작 동적 에너지 누적 [J]
  sense_amp.power.readOp.leakage = 0; //FIXME
  sense_amp.power.readOp.gate_leakage = 0; // [한국어] 읽기 동작 게이트 누설 전력 누적 [W]

  power.readOp.dynamic = temp_power + sense_amp.power.readOp.dynamic; // [한국어] 읽기 동작 동적 에너지 누적 [J]
  power.readOp.leakage = transmitter.power.readOp.leakage + // [한국어] 읽기 동작 누설 전력 누적 [W]
                         l_wire.power.readOp.leakage +
                         sense_amp.power.readOp.leakage;
  power.readOp.gate_leakage = transmitter.power.readOp.gate_leakage + // [한국어] 읽기 동작 게이트 누설 전력 누적 [W]
                         l_wire.power.readOp.gate_leakage +
                         sense_amp.power.readOp.gate_leakage;
}

  double

/*
 * [한국어]
 * Wire::sense_amp_input_cap - low-swing 감지증폭기 입력 커패시턴스
 *
 * @return: 감지증폭기 한 개의 입력 커패시턴스 [F]
 *
 * iso PMOS, sense enable/n/p 트랜지스터의 드레인/게이트 커패시턴스를 합산한다.
 * low_swing_model()에서 배선 부하에 추가된다.
 *
 * 호출 체인: low_swing_model() → [sense_amp_input_cap()]
 */
Wire::sense_amp_input_cap() // [한국어] 감지증폭기 입력 커패시턴스 계산 [F]
{
  return drain_C_(g_tp.w_iso, PCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
    gate_C(g_tp.w_sense_en + g_tp.w_sense_n, 0) + // [한국어] 게이트 커패시턴스 계산 [F]
    drain_C_(g_tp.w_sense_n, NCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
    drain_C_(g_tp.w_sense_p, PCH, 1, 1, g_tp.cell_h_def); // [한국어] 드레인/확산 커패시턴스 계산 [F]
}



/*
 * [한국어]
 * Wire::delay_optimal_wire - 최소 지연 반복기 배선 계산
 *
 * @return: (void) — delay, power, area, repeater_spacing, repeater_size 갱신
 *
 * Bakoglu의 반복기 삽입 공식을 사용하여:
 *   repeater_scaling = sqrt(out_res * wc / (wr * input_cap))
 *   repeater_spacing = sqrt(2*out_res*(out_cap+input_cap) / ((wr/len)*(wc/len)))
 * 최적 반복기 크기와 간격을 구한다. 스위칭 에너지와 short-circuit 에너지를
 * 합산하여 동적 전력을, 누설/게이트 누설 전류를 사용하여 누설 전력을 계산한다.
 *
 * 호출 체인: init_wire() → [delay_optimal_wire()]
 */
void Wire::delay_optimal_wire ()
{
  double len       = wire_length;
  //double min_wire_width = wire_width; //m
  double beta = pmos_to_nmos_sz_ratio(); // [한국어] PMOS/NMOS 폭 비율 계산
  double switching = 0;  // switching energy
  double short_ckt = 0;  // short-circuit energy
  double tc        = 0;  // time constant
  // input cap of min sized driver
  double input_cap = gate_C(g_tp.min_w_nmos_ + min_w_pmos, 0); // [한국어] 게이트 커패시턴스 계산 [F]

   // output parasitic capacitance of
   // the min. sized driver
  double out_cap = drain_C_(min_w_pmos, PCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
    drain_C_(g_tp.min_w_nmos_, NCH, 1, 1, g_tp.cell_h_def); // [한국어] 드레인/확산 커패시턴스 계산 [F]
  // drive resistance
  double out_res = (tr_R_on(g_tp.min_w_nmos_, NCH, 1) + // [한국어] 트랜지스터 온-저항 계산 [Ohm]
      tr_R_on(min_w_pmos, PCH, 1))/2; // [한국어] 트랜지스터 온-저항 계산 [Ohm]
  double wr = wire_res(len); //ohm

  // wire cap /m
  double wc = wire_cap(len); // [한국어] 배선 커패시턴스 계산 [F]

  // size the repeater such that the delay of the wire is minimum
  double repeater_scaling = sqrt(out_res*wc/(wr*input_cap)); // len will cancel

   // calc the optimum spacing between the repeaters (m)

  repeater_spacing = sqrt(2 * out_res * (out_cap + input_cap)/ // [한국어] 반복기 간격 설정 [m]
      ((wr/len)*(wc/len)));
  repeater_size = repeater_scaling; // [한국어] 반복기 크기(최소 대비 배율) 설정

  switching = (repeater_scaling * (input_cap + out_cap) +
      repeater_spacing * (wc/len)) * deviceType->Vdd * deviceType->Vdd;

  tc = out_res * (input_cap + out_cap) +
    out_res * wc/len * repeater_spacing/repeater_scaling +
    wr/len * repeater_spacing * input_cap * repeater_scaling +
    0.5 * (wr/len) * (wc/len)* repeater_spacing * repeater_spacing;

  delay = 0.693 * tc * len/repeater_spacing; // [한국어] 총 지연 값 갱신 [s]

#define Ishort_ckt 65e-6 /* across all tech Ref:Banerjee et al. {IEEE TED} */
  short_ckt = deviceType->Vdd * g_tp.min_w_nmos_ * Ishort_ckt * 1.0986 *
    repeater_scaling * tc;

  area.set_area((len/repeater_spacing) *
                compute_gate_area(INV, 1, min_w_pmos * repeater_scaling, // [한국어] 반복기 인버터 면적 계산 [m^2]
                                          g_tp.min_w_nmos_ * repeater_scaling, g_tp.cell_h_def));
  power.readOp.dynamic = ((len/repeater_spacing)*(switching + short_ckt)); // [한국어] 읽기 동작 동적 에너지 누적 [J]
  power.readOp.leakage = ((len/repeater_spacing)* // [한국어] 읽기 동작 누설 전력 누적 [W]
      deviceType->Vdd*
      cmos_Isub_leakage(g_tp.min_w_nmos_*repeater_scaling, beta*g_tp.min_w_nmos_*repeater_scaling, 1, inv)); // [한국어] 서브-임계 누설 전류 계산 [A]
  power.readOp.gate_leakage = ((len/repeater_spacing)* // [한국어] 읽기 동작 게이트 누설 전력 누적 [W]
      deviceType->Vdd*
      cmos_Ig_leakage(g_tp.min_w_nmos_*repeater_scaling, beta*g_tp.min_w_nmos_*repeater_scaling, 1, inv)); // [한국어] 게이트 절연막 누설 전류 계산 [A]
}



// calculate power/delay values for wires with suboptimal repeater sizing/spacing
void

/*
 * [한국어]
 * Wire::init_wire - 정적 배선 기준값 초기화
 *
 * @return: (void) — Wire::global, global_5/10/20/30, low_swing 정적 멤버 설정
 *
 * 최적 반복기 조합 주변의 크기/간격 그리드를 탐색하여 repeated_wire 리스트를
 * 채우고, update_fullswing()으로 5%~30% 지연 페널티 허용 범위에서 전력이
 * 최소인 구성을 선택한다. 마지막으로 Low_swing 1mm 모델을 계산하여
 * low_swing 정적 멤버를 완성한다. 이 함수는 Wire 초기화 생성자에서 1회만
 * 호출되어야 한다.
 *
 * 호출 체인: Wire(초기화 생성자) → [init_wire()]
 */
Wire::init_wire(){
  wire_length = 1;
  delay_optimal_wire(); // [한국어] 지연 최소 반복기 배선 모델 계산
    double sp, si;
  powerDef pow;
  si = repeater_size;
  sp = repeater_spacing;
  sp *= 1e6; // in microns

  double i, j, del;
  repeated_wire.push_back(Component()); // [한국어] 탐색 대상 반복기 구성 저장
  for (j=sp; j < 4*sp; j+=100) {
    for (i = si; i > 1; i--) {
      pow = wire_model(j*1e-6, i, &del);
      if (j == sp && i == si) {
        global.delay = del; // [한국어] 총 지연 값 갱신 [s]
        global.power = pow; // [한국어] Global 기준 per-meter 전력 설정 [W/m]
        global.area.h = si;
        global.area.w = sp*1e-6; // m
      }
//      cout << "Repeater size - "<< i <<
//        " Repeater spacing - " << j <<
//        " Delay - " << del <<
//        " PowerD - " << pow.readOp.dynamic <<
//        " PowerL - " << pow.readOp.leakage <<endl;
      repeated_wire.back().delay = del; // [한국어] 총 지연 값 갱신 [s]
      repeated_wire.back().power.readOp = pow.readOp;
      repeated_wire.back().area.w = j*1e-6; //m
      repeated_wire.back().area.h = i;
      repeated_wire.push_back(Component()); // [한국어] 탐색 대상 반복기 구성 저장

    }
  }
  repeated_wire.pop_back();
  update_fullswing(); // [한국어] 지연 페널티 내 최저전력 반복기 구성 선택
  Wire *l_wire = new Wire(Low_swing, 0.001/* 1 mm*/, 1);
  low_swing.delay = l_wire->delay; // [한국어] 총 지연 값 갱신 [s]
  low_swing.power = l_wire->power;
  delete l_wire;
}




/*
 * [한국어]
 * Wire::update_fullswing - 지연 페널티 상한 내 최저전력 반복기 구성 선택
 *
 * @return: (void) — global_5/10/20/30 정적 멤버 갱신
 *
 * repeated_wire 리스트에서 지연이 global.delay의 5%/10%/20%/30% 페널티
 * 임계값을 초과하지 않는 항목들만 남기고, 그 중 동적+누설 전력 비율(ncost)이
 * 최소인 항목을 global_5/global_10/global_20/global_30에 저장한다.
 *
 * 호출 체인: init_wire() → [update_fullswing()]
 */
void Wire::update_fullswing() // [한국어] 지연 페널티 내 최저전력 반복기 구성 선택
{

  list<Component>::iterator citer;
  double del[4];
  del[3] = this->global.delay + this->global.delay*.3;
  del[2] = global.delay + global.delay*.2;
  del[1] = global.delay + global.delay*.1;
  del[0] = global.delay + global.delay*.05;
  double threshold;
  double ncost;
  double cost;
  int i = 4;
  while (i>0) {
    threshold = del[i-1];
    cost = BIGNUM;
    for (citer = repeated_wire.begin(); citer != repeated_wire.end(); citer++)
    {
      if (citer->delay > threshold) {
        citer = repeated_wire.erase(citer); // [한국어] 지연 초과 반복기 구성 제거
        citer --;
      }
      else {
        ncost = citer->power.readOp.dynamic/global.power.readOp.dynamic +
                citer->power.readOp.leakage/global.power.readOp.leakage;
        if(ncost < cost)
        {
          cost = ncost;
          if (i == 4) {
            global_30.delay = citer->delay; // [한국어] 총 지연 값 갱신 [s]
            global_30.power = citer->power; // [한국어] Global 기준 per-meter 전력 설정 [W/m]
            global_30.area  = citer->area; // [한국어] Global 기준 per-meter 면적 설정 [m^2/m]
          }
          else if (i==3) {
            global_20.delay = citer->delay; // [한국어] 총 지연 값 갱신 [s]
            global_20.power = citer->power; // [한국어] Global 기준 per-meter 전력 설정 [W/m]
            global_20.area  = citer->area; // [한국어] Global 기준 per-meter 면적 설정 [m^2/m]
          }
          else if(i==2) {
            global_10.delay = citer->delay; // [한국어] 총 지연 값 갱신 [s]
            global_10.power = citer->power; // [한국어] Global 기준 per-meter 전력 설정 [W/m]
            global_10.area  = citer->area; // [한국어] Global 기준 per-meter 면적 설정 [m^2/m]
          }
          else if(i==1) {
            global_5.delay = citer->delay; // [한국어] 총 지연 값 갱신 [s]
            global_5.power = citer->power; // [한국어] Global 기준 per-meter 전력 설정 [W/m]
            global_5.area  = citer->area; // [한국어] Global 기준 per-meter 면적 설정 [m^2/m]
          }
        }
      }
    }
    i--;
  }
}




/*
 * [한국어]
 * Wire::wire_model - 주어진 반복기 간격/크기의 per-meter 지연·전력 계산
 *
 * @space : 반복기 간격 [m]
 * @size  : 반복기 크기 (최소 크기 대비 배율)
 * @delay : 출력 인자 — 해당 구성의 배선 지연 [s/m]
 * @return: 해당 구성의 1m 기준 동적/누설 전력 (powerDef)
 *
 * delay_optimal_wire()와 동일한 RC 모델을 사용하며, 외부에서 주어진
 * 반복기 간격과 크기에 대해 1m당 지연과 전력을 반환한다. init_wire()의
 * 그리드 탐색에서 반복 호출된다.
 *
 * 호출 체인: init_wire() → [wire_model()]
 */
powerDef Wire::wire_model (double space, double size, double *delay)
{
  powerDef ptemp;
  double len = 1;
  //double min_wire_width = wire_width; //m
  double beta = pmos_to_nmos_sz_ratio(); // [한국어] PMOS/NMOS 폭 비율 계산
  // switching energy
  double switching = 0;
  // short-circuit energy
  double short_ckt = 0;
  // time constant
  double tc = 0;
  // input cap of min sized driver
  double input_cap = gate_C (g_tp.min_w_nmos_ +
      min_w_pmos, 0);

   // output parasitic capacitance of
   // the min. sized driver
  double out_cap = drain_C_(min_w_pmos, PCH, 1, 1, g_tp.cell_h_def) + // [한국어] 드레인/확산 커패시턴스 계산 [F]
    drain_C_(g_tp.min_w_nmos_, NCH, 1, 1, g_tp.cell_h_def); // [한국어] 드레인/확산 커패시턴스 계산 [F]
  // drive resistance
  double out_res = (tr_R_on(g_tp.min_w_nmos_, NCH, 1) + // [한국어] 트랜지스터 온-저항 계산 [Ohm]
      tr_R_on(min_w_pmos, PCH, 1))/2; // [한국어] 트랜지스터 온-저항 계산 [Ohm]
  double wr = wire_res(len); //ohm

  // wire cap /m
  double wc = wire_cap(len); // [한국어] 배선 커패시턴스 계산 [F]

  repeater_spacing = space; // [한국어] 반복기 간격 설정 [m]
  repeater_size = size; // [한국어] 반복기 크기(최소 대비 배율) 설정

  switching = (repeater_size * (input_cap + out_cap) +
      repeater_spacing * (wc/len)) * deviceType->Vdd * deviceType->Vdd;

  tc = out_res * (input_cap + out_cap) +
    out_res * wc/len * repeater_spacing/repeater_size +
    wr/len * repeater_spacing * out_cap * repeater_size +
    0.5 * (wr/len) * (wc/len)* repeater_spacing * repeater_spacing;

  *delay = 0.693 * tc * len/repeater_spacing; // [한국어] 총 지연 값 갱신 [s]

#define Ishort_ckt 65e-6 /* across all tech Ref:Banerjee et al. {IEEE TED} */
  short_ckt = deviceType->Vdd * g_tp.min_w_nmos_ * Ishort_ckt * 1.0986 *
    repeater_size * tc;

  ptemp.readOp.dynamic = ((len/repeater_spacing)*(switching + short_ckt));
  ptemp.readOp.leakage = ((len/repeater_spacing)*
      deviceType->Vdd*
      cmos_Isub_leakage(g_tp.min_w_nmos_*repeater_size, beta*g_tp.min_w_nmos_*repeater_size, 1, inv)); // [한국어] 서브-임계 누설 전류 계산 [A]

  ptemp.readOp.gate_leakage = ((len/repeater_spacing)*
      deviceType->Vdd*
      cmos_Ig_leakage(g_tp.min_w_nmos_*repeater_size, beta*g_tp.min_w_nmos_*repeater_size, 1, inv)); // [한국어] 게이트 절연막 누설 전류 계산 [A]

  return ptemp;
}

void

/*
 * [한국어]
 * Wire::print_wire - 배선 특성 콘솔 출력
 *
 * @return: (void)
 *
 * Global(지연 최적), 5%/10%/20%/30% 오버헤드, Low-swing 각각의
 * 반복기 크기/간격, 지연, 동적/누설/게이트 누설 전력, 폭/간격을
 * stdout에 출력한다. CACTI 리포트/디버그용.
 *
 * 호출 체인: cacti_interface() 또는 main() → [print_wire()]
 */
Wire::print_wire()
{

  cout << "\nWire Properties:\n\n";
  cout << "  Delay Optimal\n\tRepeater size - "<< global.area.h <<
    " \n\tRepeater spacing - " << global.area.w*1e3 << " (mm)"
    " \n\tDelay - " << global.delay*1e6 <<  " (ns/mm)"
    " \n\tPowerD - " << global.power.readOp.dynamic *1e6<< " (nJ/mm)"
    " \n\tPowerL - " << global.power.readOp.leakage << " (mW/mm)"
    " \n\tPowerLgate - " << global.power.readOp.gate_leakage << " (mW/mm)\n";
  cout << "\tWire width - " <<wire_width_init*1e6 << " microns\n";
  cout << "\tWire spacing - " <<wire_spacing_init*1e6 << " microns\n";
  cout <<endl;

  cout << "  5% Overhead\n\tRepeater size - "<< global_5.area.h <<
    " \n\tRepeater spacing - " << global_5.area.w*1e3 << " (mm)"
    " \n\tDelay - " << global_5.delay *1e6<<  " (ns/mm)"
    " \n\tPowerD - " << global_5.power.readOp.dynamic *1e6<< " (nJ/mm)"
    " \n\tPowerL - " << global_5.power.readOp.leakage << " (mW/mm)"
    " \n\tPowerLgate - " << global_5.power.readOp.gate_leakage << " (mW/mm)\n";
  cout << "\tWire width - " <<wire_width_init*1e6 << " microns\n";
  cout << "\tWire spacing - " <<wire_spacing_init*1e6 << " microns\n";
  cout <<endl;
  cout << "  10% Overhead\n\tRepeater size - "<< global_10.area.h <<
    " \n\tRepeater spacing - " << global_10.area.w*1e3 << " (mm)"
    " \n\tDelay - " << global_10.delay *1e6<<  " (ns/mm)"
    " \n\tPowerD - " << global_10.power.readOp.dynamic *1e6<< " (nJ/mm)"
    " \n\tPowerL - " << global_10.power.readOp.leakage << " (mW/mm)"
    " \n\tPowerLgate - " << global_10.power.readOp.gate_leakage << " (mW/mm)\n";
  cout << "\tWire width - " <<wire_width_init*1e6 << " microns\n";
  cout << "\tWire spacing - " <<wire_spacing_init*1e6 << " microns\n";
  cout <<endl;
  cout << "  20% Overhead\n\tRepeater size - "<< global_20.area.h <<
    " \n\tRepeater spacing - " << global_20.area.w*1e3 << " (mm)"
    " \n\tDelay - " << global_20.delay *1e6<<  " (ns/mm)"
    " \n\tPowerD - " << global_20.power.readOp.dynamic *1e6<< " (nJ/mm)"
    " \n\tPowerL - " << global_20.power.readOp.leakage << " (mW/mm)"
    " \n\tPowerLgate - " << global_20.power.readOp.gate_leakage << " (mW/mm)\n";
  cout << "\tWire width - " <<wire_width_init*1e6 << " microns\n";
  cout << "\tWire spacing - " <<wire_spacing_init*1e6 << " microns\n";
  cout <<endl;
  cout << "  30% Overhead\n\tRepeater size - "<< global_30.area.h <<
    " \n\tRepeater spacing - " << global_30.area.w*1e3 << " (mm)"
    " \n\tDelay - " << global_30.delay *1e6<<  " (ns/mm)"
    " \n\tPowerD - " << global_30.power.readOp.dynamic *1e6<< " (nJ/mm)"
    " \n\tPowerL - " << global_30.power.readOp.leakage << " (mW/mm)"
    " \n\tPowerLgate - " << global_30.power.readOp.gate_leakage << " (mW/mm)\n";
  cout << "\tWire width - " <<wire_width_init*1e6 << " microns\n";
  cout << "\tWire spacing - " <<wire_spacing_init*1e6 << " microns\n";
  cout <<endl;
  cout << "  Low-swing wire (1 mm) - Note: Unlike repeated wires, \n\tdelay and power "
            "values of low-swing wires do not\n\thave a linear relationship with length." <<
      " \n\tdelay - " << low_swing.delay *1e9<<  " (ns)"
      " \n\tpowerD - " << low_swing.power.readOp.dynamic *1e9<< " (nJ)"
      " \n\tPowerL - " << low_swing.power.readOp.leakage << " (mW)"
      " \n\tPowerLgate - " << low_swing.power.readOp.gate_leakage << " (mW)\n";
  cout << "\tWire width - " <<wire_width_init * 2 /* differential */<< " microns\n";
  cout << "\tWire spacing - " <<wire_spacing_init * 2 /* differential */<< " microns\n";
  cout <<endl;
  cout <<endl;

}

