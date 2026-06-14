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
 * [한국어 설명] CACTI 디코더/프리디코더 모델 구현 (decoder.cc)
 *
 * === 파일의 역할 ===
 * decoder.h에 선언된 Decoder, PredecBlk, PredecBlkDrv, Predec, Driver 클래스의
 * 생성자와 compute_widths/compute_area/compute_delays/leakage_feedback 메서드를 구현한다.
 * 각 클래스는 캐시 주소 비트를 워드라인 선택 신호로 변환하는 회로 단계를 나타내며,
 * logical_effort() 함수로 최소 지연 게이트 크기를 결정하고,
 * cmos_Isub/Ig_leakage로 누설전류, horowitz()로 전파 지연을 계산한다.
 * 모든 값은 조합된 뒤 CACTI 캐시 전력/면적/지연 최적화에 입력된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch → CACTI → decoder.cc (디코더/프리디코더 회로 모델 구현).
 * 호출 체인: Mat(캐시 어레이 타일) 생성자 → Predec 생성자 → PredecBlkDrv + PredecBlk
 *   생성자 → Decoder 생성자 → compute_widths → compute_area.
 * compute_delays는 Mat의 타이밍 분석 단계에서 Predec::compute_delays → PredecBlkDrv/PredecBlk/Decoder::compute_delays 순으로 호출.
 * 실행 컨텍스트: 호스트 유저스페이스 — 시뮬레이션 초기화 시 1회 실행.
 *
 * === 타 모듈과의 연결 ===
 * 의존: area.h (Area, compute_gate_area), decoder.h (클래스 선언),
 *   parameter.h (g_tp — TechnologyParameter 전역), basic_circuit.h (gate_C, drain_C_,
 *   tr_R_on, horowitz, cmos_Isub_leakage, cmos_Ig_leakage, pmos_to_nmos_sz_ratio).
 * 이 파일에 의존: mat.cc (Mat 클래스가 Predec/Decoder 결과로 전체 MAT 타이밍 집계).
 * 데이터 흐름: compute_widths()가 게이트 폭 배열을 채우면, compute_area()가 그 결과로
 *   누설전류를 계산하고, compute_delays()가 horowitz()로 지연을 누적한다.
 *
 * === 주요 함수/구조체 요약 ===
 * Decoder::compute_widths()        : logical_effort()로 NAND2/3 + 인버터 체인 크기 결정
 * Decoder::compute_delays()        : 첫 NAND 게이트부터 최종 인버터까지 horowitz() 지연 누적
 * PredecBlk::compute_widths()      : 입력 비트 수(1~9)에 따른 L1/L2 게이트 경로 크기 결정
 * PredecBlkDrv::compute_delays()   : nand2/nand3 두 경로 드라이버 지연 계산
 * Predec::compute_delays()         : drv→blk 순서로 지연 계산, 최대 지연 경로 선택
 */

#include "area.h"      // [한국어] Area 클래스 및 compute_gate_area() — 게이트 면적 계산
#include "decoder.h"   // [한국어] Decoder, PredecBlk, PredecBlkDrv, Predec, Driver 선언
#include "parameter.h" // [한국어] g_tp (TechnologyParameter 전역) — 게이트 크기/전압/RC 파라미터
#include <iostream>    // [한국어] 디버깅용 cout (현재 미사용)
#include <math.h>      // [한국어] sqrt, log — 타이밍 계산에서 사용
#include <assert.h>    // [한국어] assert() — 잘못된 주소 비트 수 감지

using namespace std;


/*
 * [한국어]
 * Decoder::Decoder — 워드라인 디코더 생성자
 * @_num_dec_signals : 디코더 출력 신호 수 (워드라인 수)
 * @flag_way_select  : 웨이 선택 신호 추가 입력 여부
 * @_C_ld_dec_out    : 출력 부하 커패시턴스 (F)
 * @_R_wire_dec_out  : 출력 와이어 저항 (ohm)
 * @fully_assoc_     : 완전 결합 캐시 여부 (NAND2 강제 사용)
 * @is_dram_         : DRAM 여부
 * @is_wl_tr_        : 워드라인 트랜지스터 여부 (Vpp 사용)
 * @cell_            : 셀 높이/폭 참조 (area.h 기준)
 * @return           : (생성자이므로 없음)
 * _num_dec_signals를 _log2()로 변환하여 주소 비트 수를 구하고, 그 값에 따라
 * exist 플래그와 num_in_signals(2=NAND2, 3=NAND3)를 결정한다.
 * 디코더 셀 높이를 g_tp.h_dec * cell.h로 설정한 뒤 compute_widths/compute_area 호출.
 * 실행 컨텍스트: Mat 생성자 → Predec 생성자 내에서 단일 스레드 호출.
 * 호출 체인: Mat → Predec → PredecBlkDrv → [Decoder()] → compute_widths → compute_area
 */
Decoder::Decoder(
    int    _num_dec_signals,
    bool   flag_way_select,
    double _C_ld_dec_out,
    double _R_wire_dec_out,
    bool   fully_assoc_,
    bool   is_dram_,
    bool   is_wl_tr_,
    const  Area & cell_)
:exist(false),              // [한국어] 기본값: 디코더 불필요로 초기화
  C_ld_dec_out(_C_ld_dec_out),  // [한국어] 출력 부하 커패시턴스 저장
  R_wire_dec_out(_R_wire_dec_out), // [한국어] 출력 와이어 저항 저장
  num_gates(0), num_gates_min(2), // [한국어] 게이트 수 초기화 (최소 2: NAND + INV)
  delay(0),                       // [한국어] 지연 누적값 초기화
  //power(),                      // [한국어] (주석됨) Component::power 사용
  fully_assoc(fully_assoc_), is_dram(is_dram_), // [한국어] 완전 결합/DRAM 플래그
  is_wl_tr(is_wl_tr_), cell(cell_) // [한국어] 워드라인 트랜지스터 여부, 셀 참조
{

  for (int i = 0; i < MAX_NUMBER_GATES_STAGE; i++) // [한국어] 게이트 폭 배열 0으로 초기화
  {
    w_dec_n[i] = 0; // [한국어] NMOS 폭 초기화
    w_dec_p[i] = 0; // [한국어] PMOS 폭 초기화
  }

  /*
   * _num_dec_signals is the number of decoded signal as output
   * num_addr_bits_dec is the number of signal to be decoded
   * as the decoders input.
   */
  int num_addr_bits_dec = _log2(_num_dec_signals); // [한국어] 출력 신호 수를 로그₂로 변환 → 입력 주소 비트 수

  if (num_addr_bits_dec < 4) // [한국어] 주소 비트가 4 미만이면 프리디코더만으로 충분
  {
    if (flag_way_select) // [한국어] 웨이 선택 신호가 있으면 NAND2 디코더 필요
    {
      exist = true;       // [한국어] 디코더 존재 플래그 설정
      num_in_signals = 2; // [한국어] NAND2 게이트 사용 (입력 2개)
    }
    else
    {
      num_in_signals = 0; // [한국어] 디코더 불필요 — 프리디코더 출력이 직접 워드라인 구동
    }
  }
  else // [한국어] 주소 비트 4 이상 — 반드시 Decoder 필요
  {
    exist = true; // [한국어] 디코더 존재 플래그 설정

    if (flag_way_select) // [한국어] 웨이 선택이 추가되면 NAND3 (주소 2비트 + 웨이 1비트)
    {
      num_in_signals = 3; // [한국어] NAND3 게이트 사용 (입력 3개)
    }
    else
    {
      num_in_signals = 2; // [한국어] 기본 NAND2 게이트 사용
    }
  }

  assert(cell.h>0); // [한국어] 셀 높이 유효성 확인 (0이면 면적 계산 오류)
  assert(cell.w>0); // [한국어] 셀 폭 유효성 확인
  // the height of a row-decoder-driver cell is fixed to be 4 * cell.h;
  //area.h = 4 * cell.h;
  area.h = g_tp.h_dec * cell.h; // [한국어] 디코더 셀 높이 = h_dec(기술 파라미터) × 셀 높이

  compute_widths(); // [한국어] logical_effort()로 게이트 폭 결정
  compute_area();   // [한국어] 게이트 면적 및 누설전류 계산
}



/*
 * [한국어]
 * Decoder::compute_widths - logical_effort 기반 게이트 체인 크기 결정
 * NAND2(num_in_signals==2 또는 fully_assoc) 또는 NAND3(num_in_signals==3) 첫 단 게이트와
 * 이후 인버터 체인의 각 단 NMOS/PMOS 폭을 계산하여 w_dec_n/p 배열을 채운다.
 * gnand2/gnand3는 NAND 게이트의 논리적 노력 (logical effort per stage) 이다.
 * F는 전체 전기적 노력(electrical effort = C_ld_dec_out / C_in_first) × gnand이다.
 * logical_effort()가 num_gates와 w_dec_n/p[]를 결정하고 반환한다.
 * 실행 컨텍스트: Decoder 생성자 내에서 단일 스레드 호출.
 * 호출 체인: Decoder() → [compute_widths] → logical_effort
 */
void Decoder::compute_widths()
{
  double F;                                                  // [한국어] 총 전기적 노력 × 논리적 노력
  double p_to_n_sz_ratio = pmos_to_nmos_sz_ratio(is_dram, is_wl_tr); // [한국어] PMOS/NMOS 크기 비율 (이동도 차이 보정)
  double gnand2     = (2 + p_to_n_sz_ratio) / (1 + p_to_n_sz_ratio); // [한국어] NAND2 논리적 노력 g = (n+p_ratio)/(1+p_ratio)
  double gnand3     = (3 + p_to_n_sz_ratio) / (1 + p_to_n_sz_ratio); // [한국어] NAND3 논리적 노력

  if (exist) // [한국어] 디코더가 필요한 경우에만 계산
  {
    if (num_in_signals == 2 || fully_assoc) // [한국어] NAND2 게이트 사용 (완전 결합도 NAND2 강제)
    {
      w_dec_n[0] = 2 * g_tp.min_w_nmos_;           // [한국어] NAND2 NMOS: 최소 폭 × 2 (직렬 2개)
      w_dec_p[0] = p_to_n_sz_ratio * g_tp.min_w_nmos_; // [한국어] PMOS: p_to_n_ratio × 최소 폭
      F = gnand2;                                   // [한국어] 논리적 노력 초기화 (NAND2)
    }
    else // [한국어] NAND3 게이트 사용
    {
      w_dec_n[0] = 3 * g_tp.min_w_nmos_;           // [한국어] NAND3 NMOS: 최소 폭 × 3 (직렬 3개)
      w_dec_p[0] = p_to_n_sz_ratio * g_tp.min_w_nmos_; // [한국어] PMOS 폭
      F = gnand3;                                   // [한국어] 논리적 노력 초기화 (NAND3)
    }

    // [한국어] 전체 전기적 노력 = g_nand × (C_ld_out / C_in_first)
    // 입력 커패시턴스는 첫 NAND 게이트의 NMOS+PMOS 게이트 커패시턴스 합산
    F *= C_ld_dec_out / (gate_C(w_dec_n[0], 0, is_dram, false, is_wl_tr) +
                         gate_C(w_dec_p[0], 0, is_dram, false, is_wl_tr));
    // [한국어] logical_effort()가 F와 g_nand로 최적 게이트 단 수와 각 단 폭을 계산
    num_gates = logical_effort(
        num_gates_min,                              // [한국어] 최소 게이트 단 수 (2)
        num_in_signals == 2 ? gnand2 : gnand3,      // [한국어] 첫 단 논리적 노력 (NAND2 or NAND3)
        F,                                          // [한국어] 총 전기적×논리적 노력
        w_dec_n,                                    // [한국어] 출력: NMOS 폭 배열
        w_dec_p,                                    // [한국어] 출력: PMOS 폭 배열
        C_ld_dec_out,                               // [한국어] 출력 부하 커패시턴스
        p_to_n_sz_ratio,                            // [한국어] PMOS/NMOS 크기 비율
        is_dram,                                    // [한국어] DRAM 여부
        is_wl_tr,                                   // [한국어] 워드라인 트랜지스터 여부
        g_tp.max_w_nmos_dec);                       // [한국어] 디코더 NMOS 최대 폭 제한
  }
}



/*
 * [한국어]
 * Decoder::compute_area - 디코더 게이트 면적 및 누설전류 계산
 * 첫 단 NAND2 또는 NAND3 게이트와 이후 num_gates-1개 인버터의 면적을 누적하고,
 * 각 단의 서브스레숄드 누설전류(cmos_Isub_leakage)와 게이트 누설전류(cmos_Ig_leakage)를 합산.
 * 결과를 power.readOp.leakage/gate_leakage에 저장하고, area.w를 결정한다.
 * 호출 체인: Decoder() → [compute_area] → compute_gate_area, cmos_Isub/Ig_leakage
 */
void Decoder::compute_area()
{
  double cumulative_area = 0;       // [한국어] 누적 게이트 면적 (m²)
  double cumulative_curr = 0;       // cumulative leakage current — [한국어] 누적 서브스레숄드 누설전류 (A)
  double cumulative_curr_Ig = 0;    // cumulative leakage current — [한국어] 누적 게이트 누설전류 (A)

  if (exist) // First check if this decoder exists — [한국어] 디코더 존재 시에만 계산
  {
    if (num_in_signals == 2) // [한국어] NAND2 첫 단 면적/누설 계산
    {
      cumulative_area = compute_gate_area(NAND, 2, w_dec_p[0], w_dec_n[0], area.h); // [한국어] NAND2 게이트 면적
      cumulative_curr = cmos_Isub_leakage(w_dec_n[0], w_dec_p[0], 2, nand,is_dram); // [한국어] NAND2 서브스레숄드 누설
      cumulative_curr_Ig = cmos_Ig_leakage(w_dec_n[0], w_dec_p[0], 2, nand,is_dram); // [한국어] NAND2 게이트 누설
    }
    else if (num_in_signals == 3) // [한국어] NAND3 첫 단 면적/누설 계산
    {
      cumulative_area = compute_gate_area(NAND, 3, w_dec_p[0], w_dec_n[0], area.h); // [한국어] NAND3 게이트 면적
      cumulative_curr = cmos_Isub_leakage(w_dec_n[0], w_dec_p[0], 3, nand, is_dram);; // [한국어] NAND3 서브스레숄드 누설
      cumulative_curr_Ig = cmos_Ig_leakage(w_dec_n[0], w_dec_p[0], 3, nand, is_dram); // [한국어] NAND3 게이트 누설
    }

    // [한국어] 인버터 체인(2번째 단부터 마지막 단까지)의 면적과 누설전류 누적
    for (int i = 1; i < num_gates; i++)
    {
      cumulative_area += compute_gate_area(INV, 1, w_dec_p[i], w_dec_n[i], area.h); // [한국어] 인버터 면적 추가
      cumulative_curr += cmos_Isub_leakage(w_dec_n[i], w_dec_p[i], 1, inv, is_dram); // [한국어] 인버터 서브스레숄드 누설
      cumulative_curr_Ig = cmos_Ig_leakage(w_dec_n[i], w_dec_p[i], 1, inv, is_dram); // [한국어] 인버터 게이트 누설
    }
    power.readOp.leakage = cumulative_curr * g_tp.peri_global.Vdd;     // [한국어] 누설전력 = 누설전류 × Vdd
    power.readOp.gate_leakage = cumulative_curr_Ig * g_tp.peri_global.Vdd; // [한국어] 게이트 누설전력

    area.w = (cumulative_area / area.h); // [한국어] 면적 폭 = 총 면적 / 셀 높이
  }
}



/*
 * [한국어]
 * Decoder::compute_delays - 디코더 단별 전파 지연 계산 (horowitz 모델)
 * @inrisetime : 입력 신호 상승 시간 (s) — 첫 단 NAND 게이트 입력
 * @return     : 출력 신호 상승 시간 (s) — 최종 인버터 출력 (워드라인 드라이버 입력)
 * NAND 첫 단의 on-resistance(rd), 내부 노드 커패시턴스(c_intrinsic),
 * 다음 단 게이트 커패시턴스(c_load)를 구하여 RC 시정수 tf = rd*(c_int+c_load)를 계산.
 * horowitz()로 50%→50% 지연을 구하고 delay에 누적.
 * 마지막 인버터에는 R_wire_dec_out*c_load/2 와이어 지연을 추가.
 * Vpp: DRAM 워드라인은 vpp, SRAM 워드라인은 sram_cell.Vdd, 그 외는 peri_global.Vdd.
 * 마지막 단 동적 전력은 c_load*Vpp² (워드라인 스윙) + c_intrinsic*Vdd² (내부 노드).
 * 실행 컨텍스트: Mat → Predec → [Decoder::compute_delays] — 단일 스레드.
 * 호출 체인: Predec::compute_delays → [Decoder::compute_delays] → tr_R_on, gate_C, drain_C_, horowitz
 */
double Decoder::compute_delays(double inrisetime)
{
  if (exist) // [한국어] 디코더가 존재하는 경우에만 지연 계산
  {
    double ret_val = 0;  // outrisetime — [한국어] 최종 출력 상승 시간 (반환값)
    int    i;            // [한국어] 게이트 단 인덱스
    double rd, tf, this_delay, c_load, c_intrinsic, Vpp; // [한국어] 각 단 RC 파라미터
    double Vdd = g_tp.peri_global.Vdd; // [한국어] 주변부 공급 전압

    // [한국어] 워드라인 구동 전압(Vpp) 결정: DRAM → vpp, SRAM WL-TR → sram_cell.Vdd, 기타 → Vdd
    if ((is_wl_tr) && (is_dram))
    {
      Vpp = g_tp.vpp; // [한국어] DRAM 워드라인 부스트 전압 (vpp > Vdd)
    }
    else if (is_wl_tr)
    {
      Vpp = g_tp.sram_cell.Vdd; // [한국어] SRAM 셀 공급 전압 (패스 트랜지스터 구동)
    }
    else
    {
      Vpp = g_tp.peri_global.Vdd; // [한국어] 일반 주변부 전압
    }

    // first check whether a decoder is required at all — [한국어] 첫 단(NAND2 or NAND3) 지연
    rd = tr_R_on(w_dec_n[0], NCH, num_in_signals, is_dram, false, is_wl_tr); // [한국어] NAND 직렬 NMOS on-저항
    c_load = gate_C(w_dec_n[1] + w_dec_p[1], 0.0, is_dram, false, is_wl_tr); // [한국어] 다음 단 게이트 커패시턴스
    c_intrinsic = drain_C_(w_dec_p[0], PCH, 1, 1, area.h, is_dram, false, is_wl_tr) * num_in_signals + // [한국어] PMOS 드레인 × 입력 수
                  drain_C_(w_dec_n[0], NCH, num_in_signals, 1, area.h, is_dram, false, is_wl_tr);      // [한국어] 직렬 NMOS 스택 드레인
    tf = rd * (c_intrinsic + c_load); // [한국어] RC 시정수 tf = R_on × (C_int + C_next)
    this_delay = horowitz(inrisetime, tf, 0.5, 0.5, RISE); // [한국어] Horowitz 모델로 50%→50% 지연
    delay += this_delay;                                    // [한국어] 총 지연에 이 단 지연 누적
    inrisetime = this_delay / (1.0 - 0.5);                 // [한국어] 다음 단 입력 상승 시간 업데이트
    power.readOp.dynamic += (c_load + c_intrinsic) * Vdd * Vdd; // [한국어] 동적 에너지 = C × Vdd²

    // [한국어] 중간 인버터 단들(1 ~ num_gates-2)의 지연 계산
    for (i = 1; i < num_gates - 1; ++i)
    {
      rd = tr_R_on(w_dec_n[i], NCH, 1, is_dram, false, is_wl_tr); // [한국어] 단일 NMOS on-저항
      c_load = gate_C(w_dec_p[i+1] + w_dec_n[i+1], 0.0, is_dram, false, is_wl_tr); // [한국어] 다음 단 게이트 커패시턴스
      c_intrinsic = drain_C_(w_dec_p[i], PCH, 1, 1, area.h, is_dram, false, is_wl_tr) + // [한국어] PMOS 드레인
                    drain_C_(w_dec_n[i], NCH, 1, 1, area.h, is_dram, false, is_wl_tr);  // [한국어] NMOS 드레인
      tf = rd * (c_intrinsic + c_load); // [한국어] RC 시정수
      this_delay = horowitz(inrisetime, tf, 0.5, 0.5, RISE); // [한국어] 이 단 지연
      delay += this_delay;                                    // [한국어] 누적
      inrisetime = this_delay / (1.0 - 0.5);                 // [한국어] 다음 단 입력 상승 시간
      power.readOp.dynamic += (c_load + c_intrinsic) * Vdd * Vdd; // [한국어] 동적 에너지 누적
    }

    // add delay of final inverter that drives the wordline — [한국어] 최종 인버터 (워드라인 직접 구동)
    i = num_gates - 1;                              // [한국어] 마지막 단 인덱스
    c_load = C_ld_dec_out;                          // [한국어] 출력 부하 = 워드라인 커패시턴스
    rd = tr_R_on(w_dec_n[i], NCH, 1, is_dram, false, is_wl_tr); // [한국어] 최종 인버터 NMOS on-저항
    c_intrinsic = drain_C_(w_dec_p[i], PCH, 1, 1, area.h, is_dram, false, is_wl_tr) + // [한국어] PMOS 드레인
                  drain_C_(w_dec_n[i], NCH, 1, 1, area.h, is_dram, false, is_wl_tr);  // [한국어] NMOS 드레인
    tf = rd * (c_intrinsic + c_load) + R_wire_dec_out * c_load / 2; // [한국어] RC + 와이어 저항×부하/2 (분산 RC 모델)
    this_delay = horowitz(inrisetime, tf, 0.5, 0.5, RISE); // [한국어] 최종 인버터 지연
    delay  += this_delay;                                   // [한국어] 총 지연 누적
    ret_val = this_delay / (1.0 - 0.5);                    // [한국어] 출력 상승 시간 계산
    power.readOp.dynamic += c_load * Vpp * Vpp + c_intrinsic * Vdd * Vdd; // [한국어] WL: Vpp², 내부: Vdd²

    return ret_val; // [한국어] 최종 출력 상승 시간 반환 (다음 회로 입력으로 사용)
  }
  else
  {
    return 0.0; // [한국어] 디코더 불필요 시 지연 0 반환
  }
}

/*
 * [한국어]
 * Decoder::leakage_feedback - 온도에 따른 누설전류 재계산
 * @temperature : 동작 온도 (K)
 * compute_area()와 동일한 구조로 cmos_Isub/Ig_leakage를 재계산하여
 * power.readOp.leakage/gate_leakage를 현재 온도 기준으로 갱신한다.
 * 게이트 폭은 이미 결정된 w_dec_n/p를 재사용하므로 크기 재계산 불필요.
 * 호출 체인: 온도 피드백 루프 → [Decoder::leakage_feedback] → cmos_Isub/Ig_leakage
 */
void Decoder::leakage_feedback(double temperature)
{
  double cumulative_curr = 0;       // cumulative leakage current — [한국어] 누적 서브스레숄드 누설전류
  double cumulative_curr_Ig = 0;    // cumulative leakage current — [한국어] 누적 게이트 누설전류

  if (exist) // First check if this decoder exists — [한국어] 디코더 존재 시에만 재계산
  {
    if (num_in_signals == 2) // [한국어] NAND2 첫 단 누설전류 재계산
    {
      cumulative_curr = cmos_Isub_leakage(w_dec_n[0], w_dec_p[0], 2, nand,is_dram); // [한국어] NAND2 서브스레숄드 누설
      cumulative_curr_Ig = cmos_Ig_leakage(w_dec_n[0], w_dec_p[0], 2, nand,is_dram); // [한국어] NAND2 게이트 누설
    }
    else if (num_in_signals == 3) // [한국어] NAND3 첫 단 누설전류 재계산
    {
      cumulative_curr = cmos_Isub_leakage(w_dec_n[0], w_dec_p[0], 3, nand, is_dram);; // [한국어] NAND3 서브스레숄드 누설
      cumulative_curr_Ig = cmos_Ig_leakage(w_dec_n[0], w_dec_p[0], 3, nand, is_dram); // [한국어] NAND3 게이트 누설
    }

    for (int i = 1; i < num_gates; i++) // [한국어] 인버터 체인 각 단 누설전류 누적
    {
      cumulative_curr += cmos_Isub_leakage(w_dec_n[i], w_dec_p[i], 1, inv, is_dram);  // [한국어] 인버터 서브스레숄드
      cumulative_curr_Ig = cmos_Ig_leakage(w_dec_n[i], w_dec_p[i], 1, inv, is_dram); // [한국어] 인버터 게이트 누설
    }

    power.readOp.leakage = cumulative_curr * g_tp.peri_global.Vdd;     // [한국어] 누설전력 = I × Vdd
    power.readOp.gate_leakage = cumulative_curr_Ig * g_tp.peri_global.Vdd; // [한국어] 게이트 누설전력
  }
}

/*
 * [한국어]
 * PredecBlk::PredecBlk — 프리디코더 블록 생성자
 * @num_dec_signals          : 디코더 출력 신호 수 (주소 비트 수 계산용)
 * @dec_                     : 연결된 Decoder 포인터
 * @C_wire_predec_blk_out   : L2 출력 와이어 커패시턴스 (F)
 * @R_wire_predec_blk_out_  : L2 출력 와이어 저항 (ohm)
 * @num_dec_per_predec       : 이 블록에 연결된 디코더 수 (branch effort 계산용)
 * @is_dram                  : DRAM 여부
 * @is_blk1                 : 하위 비트 블록(true) 또는 상위 비트 블록(false)
 * num_dec_signals로부터 총 주소 비트를 계산하고, is_blk1에 따라 절반씩 분배.
 * is_blk1=true: 4비트 미만이면 L1만으로 직접 워드라인 구동, 4비트 이상이면 L1+L2 구성.
 * is_blk1=false: 4비트 이상일 때만 exist=true (상위 비트 블록).
 * branch_effort_predec_out: 한 L2 게이트가 구동하는 Decoder 수.
 * C_ld_predec_blk_out: L2 출력 부하 = branch_effort × 디코더 입력 커패시턴스 + 와이어.
 * 실행 컨텍스트: Predec 생성자 내에서 단일 스레드 호출.
 * 호출 체인: Predec → PredecBlkDrv → [PredecBlk()] → compute_widths → compute_area
 */
PredecBlk::PredecBlk(
    int    num_dec_signals,
    Decoder * dec_,
    double C_wire_predec_blk_out,
    double R_wire_predec_blk_out_,
    int    num_dec_per_predec,
    bool   is_dram,
    bool   is_blk1)
 :dec(dec_),              // [한국어] 연결된 Decoder 포인터 저장
  exist(false),           // [한국어] 기본값: 블록 불필요
  number_input_addr_bits(0), // [한국어] 처리할 주소 비트 수 (초기화)
  C_ld_predec_blk_out(0),    // [한국어] L2 출력 부하 커패시턴스 (초기화)
  R_wire_predec_blk_out(0),  // [한국어] L2 출력 와이어 저항 (초기화)
  branch_effort_nand2_gate_output(1), // [한국어] NAND2 분기 노력 초기화
  branch_effort_nand3_gate_output(1), // [한국어] NAND3 분기 노력 초기화
  flag_two_unique_paths(false),      // [한국어] nand2/nand3 두 경로 모두 사용 여부 초기화
  flag_L2_gate(0),                   // [한국어] L2 게이트 유형 (0=없음)
  number_inputs_L1_gate(0),          // [한국어] L1 게이트 입력 수 초기화
  number_gates_L1_nand2_path(0),     // [한국어] L1 NAND2 경로 게이트 수 초기화
  number_gates_L1_nand3_path(0),     // [한국어] L1 NAND3 경로 게이트 수 초기화
  number_gates_L2(0),                // [한국어] L2 게이트 수 초기화
  min_number_gates_L1(2),            // [한국어] L1 최소 게이트 단 수
  min_number_gates_L2(2),            // [한국어] L2 최소 게이트 단 수
  num_L1_active_nand2_path(0),       // [한국어] 활성 NAND2 경로 수 초기화
  num_L1_active_nand3_path(0),       // [한국어] 활성 NAND3 경로 수 초기화
  delay_nand2_path(0),               // [한국어] NAND2 경로 총 지연 초기화
  delay_nand3_path(0),               // [한국어] NAND3 경로 총 지연 초기화
  power_nand2_path(),                // [한국어] NAND2 경로 전력 초기화
  power_nand3_path(),                // [한국어] NAND3 경로 전력 초기화
  power_L2(),                        // [한국어] L2 전력 초기화
  is_dram_(is_dram)                  // [한국어] DRAM 여부 저장
{
  int    branch_effort_predec_out;   // [한국어] L2 게이트 1개가 구동하는 Decoder 입력 수
  double C_ld_dec_gate;              // [한국어] Decoder 1개 입력 게이트 커패시턴스 × num_dec_per_predec
  int    num_addr_bits_dec = _log2(num_dec_signals); // [한국어] 총 디코더 주소 비트 수
  int    blk1_num_input_addr_bits = (num_addr_bits_dec + 1) / 2; // [한국어] blk1 담당 비트(상위 절반)
  int    blk2_num_input_addr_bits = num_addr_bits_dec - blk1_num_input_addr_bits; // [한국어] blk2 담당 비트(하위 절반)

  w_L1_nand2_n[0] = 0; // [한국어] L1 NAND2 NMOS 폭 초기화
  w_L1_nand2_p[0] = 0; // [한국어] L1 NAND2 PMOS 폭 초기화
  w_L1_nand3_n[0] = 0; // [한국어] L1 NAND3 NMOS 폭 초기화
  w_L1_nand3_p[0] = 0; // [한국어] L1 NAND3 PMOS 폭 초기화

  if (is_blk1 == true) // [한국어] blk1(하위 비트 담당) 설정
  {
    if (num_addr_bits_dec <= 0) // [한국어] 주소 비트가 없으면 블록 불필요
    {
      return; // [한국어] 초기화 없이 종료 (exist=false 유지)
    }
    else if (num_addr_bits_dec < 4) // [한국어] 4비트 미만: L1만으로 워드라인 직접 구동
    {
      // Just one predecoder block is required with NAND2 gates. No decoder required.
      // The first level of predecoding directly drives the decoder output load
      exist = true;                                   // [한국어] 블록 필요 플래그
      number_input_addr_bits = num_addr_bits_dec;     // [한국어] 전체 주소 비트를 blk1이 담당
      R_wire_predec_blk_out = dec->R_wire_dec_out;    // [한국어] 디코더 출력 와이어 저항 그대로 사용
      C_ld_predec_blk_out = dec->C_ld_dec_out;        // [한국어] 디코더 출력 부하 커패시턴스 그대로 사용
    }
    else // [한국어] 4비트 이상: L1+L2 두 단계 프리디코딩
    {
      exist = true;                                           // [한국어] 블록 필요 플래그
      number_input_addr_bits   = blk1_num_input_addr_bits;   // [한국어] blk1 담당 비트 수
      branch_effort_predec_out = (1 << blk2_num_input_addr_bits); // [한국어] L2 한 신호가 구동하는 Decoder 수 = 2^blk2_bits
      C_ld_dec_gate = num_dec_per_predec * gate_C(dec->w_dec_n[0] + dec->w_dec_p[0], 0, is_dram_, false, false); // [한국어] 디코더 입력 커패시턴스
      R_wire_predec_blk_out = R_wire_predec_blk_out_;         // [한국어] L2 출력 와이어 저항
      C_ld_predec_blk_out = branch_effort_predec_out * C_ld_dec_gate + C_wire_predec_blk_out; // [한국어] L2 출력 부하 = 분기노력 × 디코더 커패시턴스 + 와이어
    }
  }
  else // [한국어] blk2(상위 비트 담당) 설정
  {
    if (num_addr_bits_dec >= 4) // [한국어] 4비트 이상일 때만 blk2 필요
    {
      exist = true;                                           // [한국어] 블록 필요 플래그
      number_input_addr_bits   = blk2_num_input_addr_bits;   // [한국어] blk2 담당 비트 수
      branch_effort_predec_out = (1 << blk1_num_input_addr_bits); // [한국어] 2^blk1_bits
      C_ld_dec_gate = num_dec_per_predec * gate_C(dec->w_dec_n[0] + dec->w_dec_p[0], 0, is_dram_, false, false); // [한국어] 디코더 입력 커패시턴스
      R_wire_predec_blk_out = R_wire_predec_blk_out_;         // [한국어] L2 출력 와이어 저항
      C_ld_predec_blk_out = branch_effort_predec_out * C_ld_dec_gate + C_wire_predec_blk_out; // [한국어] L2 출력 부하
    }
  }

  compute_widths(); // [한국어] L1/L2 게이트 체인 크기 결정
  compute_area();   // [한국어] 면적 및 누설전류 계산
}



void PredecBlk::compute_widths()
{
  double F, c_load_nand3_path, c_load_nand2_path;
  double p_to_n_sz_ratio = pmos_to_nmos_sz_ratio(is_dram_);
  double gnand2 = (2 + p_to_n_sz_ratio) / (1 + p_to_n_sz_ratio);
  double gnand3 = (3 + p_to_n_sz_ratio) / (1 + p_to_n_sz_ratio);

  if (exist == false) return;


  switch (number_input_addr_bits)
  {
    case 1:
      flag_two_unique_paths           = false;
      number_inputs_L1_gate           = 2;
      flag_L2_gate                    = 0;
      break;
    case 2:
      flag_two_unique_paths           = false;
      number_inputs_L1_gate           = 2;
      flag_L2_gate                    = 0;
      break;
    case 3:
      flag_two_unique_paths           = false;
      number_inputs_L1_gate           = 3;
      flag_L2_gate                    = 0;
      break;
    case 4:
      flag_two_unique_paths           = false;
      number_inputs_L1_gate           = 2;
      flag_L2_gate                    = 2;
      branch_effort_nand2_gate_output = 4;
      break;
    case 5:
      flag_two_unique_paths           = true;
      flag_L2_gate                    = 2;
      branch_effort_nand2_gate_output = 8;
      branch_effort_nand3_gate_output = 4;
      break;
    case 6:
      flag_two_unique_paths           = false;
      number_inputs_L1_gate           = 3;
      flag_L2_gate                    = 2;
      branch_effort_nand3_gate_output = 8;
      break;
    case 7:
      flag_two_unique_paths           = true;
      flag_L2_gate                    = 3;
      branch_effort_nand2_gate_output = 32;
      branch_effort_nand3_gate_output = 16;
      break;
    case 8:
      flag_two_unique_paths           = true;
      flag_L2_gate                    = 3;
      branch_effort_nand2_gate_output = 64;
      branch_effort_nand3_gate_output = 32;
      break;
    case 9:
      flag_two_unique_paths           = false;
      number_inputs_L1_gate           = 3;
      flag_L2_gate                    = 3;
      branch_effort_nand3_gate_output = 64;
      break;
    default:
      assert(0);
      break;
  }

  // find the number of gates and sizing in second level of predecoder (if there is a second level)
  if (flag_L2_gate)
  {
    if (flag_L2_gate == 2)
    { // 2nd level is a NAND2 gate
      w_L2_n[0] = 2 * g_tp.min_w_nmos_;
      F = gnand2;
    }
    else
    { // 2nd level is a NAND3 gate
      w_L2_n[0] = 3 * g_tp.min_w_nmos_;
      F = gnand3;
    }
    w_L2_p[0] = p_to_n_sz_ratio * g_tp.min_w_nmos_;
    F *= C_ld_predec_blk_out / (gate_C(w_L2_n[0], 0, is_dram_) + gate_C(w_L2_p[0], 0, is_dram_));
    number_gates_L2 = logical_effort(
        min_number_gates_L2,
        flag_L2_gate == 2 ? gnand2 : gnand3,
        F,
        w_L2_n,
        w_L2_p,
        C_ld_predec_blk_out,
        p_to_n_sz_ratio,
        is_dram_, false,
        g_tp.max_w_nmos_);

    // Now find the number of gates and widths in first level of predecoder
    if ((flag_two_unique_paths)||(number_inputs_L1_gate == 2))
    { // Whenever flag_two_unique_paths is true, it means first level of decoder employs
      // both NAND2 and NAND3 gates. Or when number_inputs_L1_gate is 2, it means
      // a NAND2 gate is used in the first level of the predecoder
      c_load_nand2_path = branch_effort_nand2_gate_output *
        (gate_C(w_L2_n[0], 0, is_dram_) +
         gate_C(w_L2_p[0], 0, is_dram_));
      w_L1_nand2_n[0] = 2 * g_tp.min_w_nmos_;
      w_L1_nand2_p[0] = p_to_n_sz_ratio * g_tp.min_w_nmos_;
      F = gnand2 * c_load_nand2_path /
        (gate_C(w_L1_nand2_n[0], 0, is_dram_) +
         gate_C(w_L1_nand2_p[0], 0, is_dram_));
      number_gates_L1_nand2_path = logical_effort(
          min_number_gates_L1,
          gnand2,
          F,
          w_L1_nand2_n,
          w_L1_nand2_p,
          c_load_nand2_path,
          p_to_n_sz_ratio,
          is_dram_, false,
          g_tp.max_w_nmos_);
    }

    //Now find widths of gates along path in which first gate is a NAND3
    if ((flag_two_unique_paths)||(number_inputs_L1_gate == 3))
    { // Whenever flag_two_unique_paths is TRUE, it means first level of decoder employs
      // both NAND2 and NAND3 gates. Or when number_inputs_L1_gate is 3, it means
      // a NAND3 gate is used in the first level of the predecoder
      c_load_nand3_path = branch_effort_nand3_gate_output *
        (gate_C(w_L2_n[0], 0, is_dram_) +
         gate_C(w_L2_p[0], 0, is_dram_));
      w_L1_nand3_n[0] = 3 * g_tp.min_w_nmos_;
      w_L1_nand3_p[0] = p_to_n_sz_ratio * g_tp.min_w_nmos_;
      F = gnand3 * c_load_nand3_path /
        (gate_C(w_L1_nand3_n[0], 0, is_dram_) +
         gate_C(w_L1_nand3_p[0], 0, is_dram_));
      number_gates_L1_nand3_path = logical_effort(
          min_number_gates_L1,
          gnand3,
          F,
          w_L1_nand3_n,
          w_L1_nand3_p,
          c_load_nand3_path,
          p_to_n_sz_ratio,
          is_dram_, false,
          g_tp.max_w_nmos_);
    }
  }
  else
  { // find number of gates and widths in first level of predecoder block when there is no second level
    if (number_inputs_L1_gate == 2)
    {
      w_L1_nand2_n[0] = 2 * g_tp.min_w_nmos_;
      w_L1_nand2_p[0] = p_to_n_sz_ratio * g_tp.min_w_nmos_;
      F = gnand2*C_ld_predec_blk_out /
        (gate_C(w_L1_nand2_n[0], 0, is_dram_) +
         gate_C(w_L1_nand2_p[0], 0, is_dram_));
      number_gates_L1_nand2_path = logical_effort(
          min_number_gates_L1,
          gnand2,
          F,
          w_L1_nand2_n,
          w_L1_nand2_p,
          C_ld_predec_blk_out,
          p_to_n_sz_ratio,
          is_dram_, false,
          g_tp.max_w_nmos_);
    }
    else if (number_inputs_L1_gate == 3)
    {
      w_L1_nand3_n[0] = 3 * g_tp.min_w_nmos_;
      w_L1_nand3_p[0] = p_to_n_sz_ratio * g_tp.min_w_nmos_;
      F = gnand3*C_ld_predec_blk_out /
        (gate_C(w_L1_nand3_n[0], 0, is_dram_) +
         gate_C(w_L1_nand3_p[0], 0, is_dram_));
      number_gates_L1_nand3_path = logical_effort(
          min_number_gates_L1,
          gnand3,
          F,
          w_L1_nand3_n,
          w_L1_nand3_p,
          C_ld_predec_blk_out,
          p_to_n_sz_ratio,
          is_dram_, false,
          g_tp.max_w_nmos_);
    }
  }
}



void PredecBlk::compute_area()
{
  if (exist)
  { // First check whether a predecoder block is needed
    int num_L1_nand2 = 0;
    int num_L1_nand3 = 0;
    int num_L2 = 0;
    double tot_area_L1_nand3  =0;
    double leak_L1_nand3      =0;
    double gate_leak_L1_nand3 =0;

    double tot_area_L1_nand2  = compute_gate_area(NAND, 2, w_L1_nand2_p[0], w_L1_nand2_n[0], g_tp.cell_h_def);
    double leak_L1_nand2      = cmos_Isub_leakage(w_L1_nand2_n[0], w_L1_nand2_p[0], 2, nand, is_dram_);
    double gate_leak_L1_nand2 = cmos_Ig_leakage(w_L1_nand2_n[0], w_L1_nand2_p[0], 2, nand, is_dram_);
    if (number_inputs_L1_gate != 3) {
      tot_area_L1_nand3 = 0;
      leak_L1_nand3 = 0;
      gate_leak_L1_nand3 =0;
    }
    else {
      tot_area_L1_nand3  = compute_gate_area(NAND, 3, w_L1_nand3_p[0], w_L1_nand3_n[0], g_tp.cell_h_def);
      leak_L1_nand3      = cmos_Isub_leakage(w_L1_nand3_n[0], w_L1_nand3_p[0], 3, nand);
      gate_leak_L1_nand3 = cmos_Ig_leakage(w_L1_nand3_n[0], w_L1_nand3_p[0], 3, nand);
    }

    switch (number_input_addr_bits)
    {
      case 1: //2 NAND2 gates
        num_L1_nand2 = 2;
        num_L2       = 0;
        num_L1_active_nand2_path =1;
        num_L1_active_nand3_path =0;
        break;
      case 2: //4 NAND2 gates
        num_L1_nand2 = 4;
        num_L2       = 0;
        num_L1_active_nand2_path =1;
        num_L1_active_nand3_path =0;
        break;
      case 3: //8 NAND3 gates
        num_L1_nand3 = 8;
        num_L2       = 0;
        num_L1_active_nand2_path =0;
        num_L1_active_nand3_path =1;
        break;
      case 4: //4 + 4 NAND2 gates
        num_L1_nand2 = 8;
        num_L2       = 16;
        num_L1_active_nand2_path =2;
        num_L1_active_nand3_path =0;
        break;
      case 5: //4 NAND2 gates, 8 NAND3 gates
        num_L1_nand2 = 4;
        num_L1_nand3 = 8;
        num_L2       = 32;
        num_L1_active_nand2_path =1;
        num_L1_active_nand3_path =1;
        break;
      case 6: //8 + 8 NAND3 gates
        num_L1_nand3 = 16;
        num_L2       = 64;
        num_L1_active_nand2_path =0;
        num_L1_active_nand3_path =2;
        break;
      case 7: //4 + 4 NAND2 gates, 8 NAND3 gates
        num_L1_nand2 = 8;
        num_L1_nand3 = 8;
        num_L2       = 128;
        num_L1_active_nand2_path =2;
        num_L1_active_nand3_path =1;
        break;
      case 8: //4 NAND2 gates, 8 + 8 NAND3 gates
        num_L1_nand2 = 4;
        num_L1_nand3 = 16;
        num_L2       = 256;
        num_L1_active_nand2_path =2;
        num_L1_active_nand3_path =2;
        break;
      case 9: //8 + 8 + 8 NAND3 gates
        num_L1_nand3 = 24;
        num_L2       = 512;
        num_L1_active_nand2_path =0;
        num_L1_active_nand3_path =3;
        break;
      default:
        break;
    }

    for (int i = 1; i < number_gates_L1_nand2_path; ++i)
    {
      tot_area_L1_nand2  += compute_gate_area(INV, 1, w_L1_nand2_p[i], w_L1_nand2_n[i], g_tp.cell_h_def);
      leak_L1_nand2      += cmos_Isub_leakage(w_L1_nand2_n[i], w_L1_nand2_p[i], 2, nand, is_dram_);
      gate_leak_L1_nand2 += cmos_Ig_leakage(w_L1_nand2_n[i], w_L1_nand2_p[i], 2, nand, is_dram_);
    }
    tot_area_L1_nand2  *= num_L1_nand2;
    leak_L1_nand2      *= num_L1_nand2;
    gate_leak_L1_nand2 *= num_L1_nand2;

    for (int i = 1; i < number_gates_L1_nand3_path; ++i)
    {
      tot_area_L1_nand3  += compute_gate_area(INV, 1, w_L1_nand3_p[i], w_L1_nand3_n[i], g_tp.cell_h_def);
      leak_L1_nand3      += cmos_Isub_leakage(w_L1_nand3_n[i], w_L1_nand3_p[i], 3, nand, is_dram_);
      gate_leak_L1_nand3 += cmos_Ig_leakage(w_L1_nand3_n[i], w_L1_nand3_p[i], 3, nand, is_dram_);
    }
    tot_area_L1_nand3  *= num_L1_nand3;
    leak_L1_nand3      *= num_L1_nand3;
    gate_leak_L1_nand3 *= num_L1_nand3;

    double cumulative_area_L1 = tot_area_L1_nand2 + tot_area_L1_nand3;
    double cumulative_area_L2 = 0.0;
    double leakage_L2         = 0.0;
    double gate_leakage_L2    = 0.0;

    if (flag_L2_gate == 2)
    {
      cumulative_area_L2 = compute_gate_area(NAND, 2, w_L2_p[0], w_L2_n[0], g_tp.cell_h_def);
      leakage_L2         = cmos_Isub_leakage(w_L2_n[0], w_L2_p[0], 2, nand, is_dram_);
      gate_leakage_L2    = cmos_Ig_leakage(w_L2_n[0], w_L2_p[0], 2, nand, is_dram_);
    }
    else if (flag_L2_gate == 3)
    {
      cumulative_area_L2 = compute_gate_area(NAND, 3, w_L2_p[0], w_L2_n[0], g_tp.cell_h_def);
      leakage_L2         = cmos_Isub_leakage(w_L2_n[0], w_L2_p[0], 3, nand, is_dram_);
      gate_leakage_L2    = cmos_Ig_leakage(w_L2_n[0], w_L2_p[0], 3, nand, is_dram_);
    }

    for (int i = 1; i < number_gates_L2; ++i)
    {
      cumulative_area_L2 += compute_gate_area(INV, 1, w_L2_p[i], w_L2_n[i], g_tp.cell_h_def);
      leakage_L2         += cmos_Isub_leakage(w_L2_n[i], w_L2_p[i], 2, inv, is_dram_);
      gate_leakage_L2    += cmos_Ig_leakage(w_L2_n[i], w_L2_p[i], 2, inv, is_dram_);
    }
    cumulative_area_L2 *= num_L2;
    leakage_L2         *= num_L2;
    gate_leakage_L2    *= num_L2;

    power_nand2_path.readOp.leakage = leak_L1_nand2 * g_tp.peri_global.Vdd;
    power_nand3_path.readOp.leakage = leak_L1_nand3 * g_tp.peri_global.Vdd;
    power_L2.readOp.leakage         = leakage_L2    * g_tp.peri_global.Vdd;
    area.set_area(cumulative_area_L1 + cumulative_area_L2);
    power_nand2_path.readOp.gate_leakage = gate_leak_L1_nand2 * g_tp.peri_global.Vdd;
    power_nand3_path.readOp.gate_leakage = gate_leak_L1_nand3 * g_tp.peri_global.Vdd;
    power_L2.readOp.gate_leakage         = gate_leakage_L2    * g_tp.peri_global.Vdd;
  }
}



pair<double, double> PredecBlk::compute_delays(
    pair<double, double> inrisetime)  // <nand2, nand3>
{
  pair<double, double> ret_val;
  ret_val.first  = 0;  // outrisetime_nand2_path
  ret_val.second = 0;  // outrisetime_nand3_path

  double inrisetime_nand2_path = inrisetime.first;
  double inrisetime_nand3_path = inrisetime.second;
  int    i;
  double rd, c_load, c_intrinsic, tf, this_delay;
  double Vdd = g_tp.peri_global.Vdd;

  // TODO: following delay calculation part can be greatly simplified.
  // first check whether a predecoder block is required
  if (exist)
  {
    //Find delay in first level of predecoder block
    //First find delay in path
    if ((flag_two_unique_paths) || (number_inputs_L1_gate == 2))
    {
      //First gate is a NAND2 gate
      rd = tr_R_on(w_L1_nand2_n[0], NCH, 2, is_dram_);
      c_load = gate_C(w_L1_nand2_n[1] + w_L1_nand2_p[1], 0.0, is_dram_);
      c_intrinsic = 2 * drain_C_(w_L1_nand2_p[0], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                        drain_C_(w_L1_nand2_n[0], NCH, 2, 1, g_tp.cell_h_def, is_dram_);
      tf = rd * (c_intrinsic + c_load);
      this_delay = horowitz(inrisetime_nand2_path, tf, 0.5, 0.5, RISE);
      delay_nand2_path += this_delay;
      inrisetime_nand2_path = this_delay / (1.0 - 0.5);
      power_nand2_path.readOp.dynamic += (c_load + c_intrinsic) * Vdd * Vdd;

      //Add delays of all but the last inverter in the chain
      for (i = 1; i < number_gates_L1_nand2_path - 1; ++i)
      {
        rd = tr_R_on(w_L1_nand2_n[i], NCH, 1, is_dram_);
        c_load = gate_C(w_L1_nand2_n[i+1] + w_L1_nand2_p[i+1], 0.0, is_dram_);
        c_intrinsic = drain_C_(w_L1_nand2_p[i], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                      drain_C_(w_L1_nand2_n[i], NCH, 1, 1, g_tp.cell_h_def, is_dram_);
        tf = rd * (c_intrinsic + c_load);
        this_delay = horowitz(inrisetime_nand2_path, tf, 0.5, 0.5, RISE);
        delay_nand2_path += this_delay;
        inrisetime_nand2_path = this_delay / (1.0 - 0.5);
        power_nand2_path.readOp.dynamic += (c_intrinsic + c_load) * Vdd * Vdd;
      }

      //Add delay of the last inverter
      i = number_gates_L1_nand2_path - 1;
      rd = tr_R_on(w_L1_nand2_n[i], NCH, 1, is_dram_);
      if (flag_L2_gate)
      {
        c_load = branch_effort_nand2_gate_output*(gate_C(w_L2_n[0], 0, is_dram_) + gate_C(w_L2_p[0], 0, is_dram_));
        c_intrinsic = drain_C_(w_L1_nand2_p[i], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                      drain_C_(w_L1_nand2_n[i], NCH, 1, 1, g_tp.cell_h_def, is_dram_);
        tf = rd * (c_intrinsic + c_load);
        this_delay = horowitz(inrisetime_nand2_path, tf, 0.5, 0.5, RISE);
        delay_nand2_path += this_delay;
        inrisetime_nand2_path = this_delay / (1.0 - 0.5);
        power_nand2_path.readOp.dynamic += (c_intrinsic + c_load) * Vdd * Vdd;
      }
      else
      { //First level directly drives decoder output load
        c_load = C_ld_predec_blk_out;
        c_intrinsic = drain_C_(w_L1_nand2_p[i], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                      drain_C_(w_L1_nand2_n[i], NCH, 1, 1, g_tp.cell_h_def, is_dram_);
        tf = rd * (c_intrinsic + c_load) + R_wire_predec_blk_out * c_load / 2;
        this_delay = horowitz(inrisetime_nand2_path, tf, 0.5, 0.5, RISE);
        delay_nand2_path += this_delay;
        ret_val.first = this_delay / (1.0 - 0.5);
        power_nand2_path.readOp.dynamic += (c_intrinsic + c_load) * Vdd * Vdd;
      }
    }

    if ((flag_two_unique_paths) || (number_inputs_L1_gate == 3))
    { //Check if the number of gates in the first level is more than 1.
      //First gate is a NAND3 gate
      rd = tr_R_on(w_L1_nand3_n[0], NCH, 3, is_dram_);
      c_load = gate_C(w_L1_nand3_n[1] + w_L1_nand3_p[1], 0.0, is_dram_);
      c_intrinsic = 3 * drain_C_(w_L1_nand3_p[0], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                        drain_C_(w_L1_nand3_n[0], NCH, 3, 1, g_tp.cell_h_def, is_dram_);
      tf = rd * (c_intrinsic + c_load);
      this_delay = horowitz(inrisetime_nand3_path, tf, 0.5, 0.5, RISE);
      delay_nand3_path += this_delay;
      inrisetime_nand3_path = this_delay / (1.0 - 0.5);
      power_nand3_path.readOp.dynamic += (c_intrinsic + c_load) * Vdd * Vdd;

      //Add delays of all but the last inverter in the chain
      for (i = 1; i < number_gates_L1_nand3_path - 1; ++i)
      {
        rd = tr_R_on(w_L1_nand3_n[i], NCH, 1, is_dram_);
        c_load = gate_C(w_L1_nand3_n[i+1] + w_L1_nand3_p[i+1], 0.0, is_dram_);
        c_intrinsic = drain_C_(w_L1_nand3_p[i], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                      drain_C_(w_L1_nand3_n[i], NCH, 1, 1, g_tp.cell_h_def, is_dram_);
        tf = rd * (c_intrinsic + c_load);
        this_delay = horowitz(inrisetime_nand3_path, tf, 0.5, 0.5, RISE);
        delay_nand3_path += this_delay;
        inrisetime_nand3_path = this_delay / (1.0 - 0.5);
        power_nand3_path.readOp.dynamic += (c_intrinsic + c_load) * Vdd * Vdd;
      }

      //Add delay of the last inverter
      i = number_gates_L1_nand3_path - 1;
      rd = tr_R_on(w_L1_nand3_n[i], NCH, 1, is_dram_);
      if (flag_L2_gate)
      {
        c_load = branch_effort_nand3_gate_output*(gate_C(w_L2_n[0], 0, is_dram_) + gate_C(w_L2_p[0], 0, is_dram_));
        c_intrinsic = drain_C_(w_L1_nand3_p[i], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                      drain_C_(w_L1_nand3_n[i], NCH, 1, 1, g_tp.cell_h_def, is_dram_);
        tf = rd * (c_intrinsic + c_load);
        this_delay = horowitz(inrisetime_nand3_path, tf, 0.5, 0.5, RISE);
        delay_nand3_path += this_delay;
        inrisetime_nand3_path = this_delay / (1.0 - 0.5);
        power_nand3_path.readOp.dynamic += (c_intrinsic + c_load) * Vdd * Vdd;
      }
      else
      { //First level directly drives decoder output load
        c_load = C_ld_predec_blk_out;
        c_intrinsic = drain_C_(w_L1_nand3_p[i], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                      drain_C_(w_L1_nand3_n[i], NCH, 1, 1, g_tp.cell_h_def, is_dram_);
        tf = rd * (c_intrinsic + c_load) + R_wire_predec_blk_out * c_load / 2;
        this_delay = horowitz(inrisetime_nand3_path, tf, 0.5, 0.5, RISE);
        delay_nand3_path += this_delay;
        ret_val.second = this_delay / (1.0 - 0.5);
        power_nand3_path.readOp.dynamic += (c_intrinsic + c_load) * Vdd * Vdd;
      }
    }

    // Find delay through second level
    if (flag_L2_gate)
    {
      if (flag_L2_gate == 2)
      {
        rd = tr_R_on(w_L2_n[0], NCH, 2, is_dram_);
        c_load = gate_C(w_L2_n[1] + w_L2_p[1], 0.0, is_dram_);
        c_intrinsic = 2 * drain_C_(w_L2_p[0], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                          drain_C_(w_L2_n[0], NCH, 2, 1, g_tp.cell_h_def, is_dram_);
        tf = rd * (c_intrinsic + c_load);
        this_delay = horowitz(inrisetime_nand2_path, tf, 0.5, 0.5, RISE);
        delay_nand2_path += this_delay;
        inrisetime_nand2_path = this_delay / (1.0 - 0.5);
        power_L2.readOp.dynamic += (c_intrinsic + c_load) * Vdd * Vdd;
      }
      else
      { // flag_L2_gate = 3
        rd = tr_R_on(w_L2_n[0], NCH, 3, is_dram_);
        c_load = gate_C(w_L2_n[1] + w_L2_p[1], 0.0, is_dram_);
        c_intrinsic = 3 * drain_C_(w_L2_p[0], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                          drain_C_(w_L2_n[0], NCH, 3, 1, g_tp.cell_h_def, is_dram_);
        tf = rd * (c_intrinsic + c_load);
        this_delay = horowitz(inrisetime_nand3_path, tf, 0.5, 0.5, RISE);
        delay_nand3_path += this_delay;
        inrisetime_nand3_path = this_delay / (1.0 - 0.5);
        power_L2.readOp.dynamic += (c_intrinsic + c_load) * Vdd * Vdd;
      }

      for (i = 1; i < number_gates_L2 - 1; ++i)
      {
        rd = tr_R_on(w_L2_n[i], NCH, 1, is_dram_);
        c_load = gate_C(w_L2_n[i+1] + w_L2_p[i+1], 0.0, is_dram_);
        c_intrinsic = drain_C_(w_L2_p[i], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                      drain_C_(w_L2_n[i], NCH, 1, 1, g_tp.cell_h_def, is_dram_);
        tf = rd * (c_intrinsic + c_load);
        this_delay = horowitz(inrisetime_nand2_path, tf, 0.5, 0.5, RISE);
        delay_nand2_path += this_delay;
        inrisetime_nand2_path = this_delay / (1.0 - 0.5);
        this_delay = horowitz(inrisetime_nand3_path, tf, 0.5, 0.5, RISE);
        delay_nand3_path += this_delay;
        inrisetime_nand3_path = this_delay / (1.0 - 0.5);
        power_L2.readOp.dynamic += (c_intrinsic + c_load) * Vdd * Vdd;
      }

      //Add delay of final inverter that drives the wordline decoders
      i = number_gates_L2 - 1;
      c_load = C_ld_predec_blk_out;
      rd = tr_R_on(w_L2_n[i], NCH, 1, is_dram_);
      c_intrinsic = drain_C_(w_L2_p[i], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                    drain_C_(w_L2_n[i], NCH, 1, 1, g_tp.cell_h_def, is_dram_);
      tf = rd * (c_intrinsic + c_load) + R_wire_predec_blk_out * c_load / 2;
      this_delay = horowitz(inrisetime_nand2_path, tf, 0.5, 0.5, RISE);
      delay_nand2_path += this_delay;
      ret_val.first = this_delay / (1.0 - 0.5);
      this_delay = horowitz(inrisetime_nand3_path, tf, 0.5, 0.5, RISE);
      delay_nand3_path += this_delay;
      ret_val.second = this_delay / (1.0 - 0.5);
      power_L2.readOp.dynamic += (c_intrinsic + c_load) * Vdd * Vdd;
    }
  }

  delay = (ret_val.first > ret_val.second) ? ret_val.first : ret_val.second;
  return ret_val;
}

void PredecBlk::leakage_feedback(double temperature)
{
  if (exist)
  { // First check whether a predecoder block is needed
    int num_L1_nand2 = 0;
    int num_L1_nand3 = 0;
    int num_L2 = 0;
    double leak_L1_nand3      =0;
    double gate_leak_L1_nand3 =0;

    double leak_L1_nand2      = cmos_Isub_leakage(w_L1_nand2_n[0], w_L1_nand2_p[0], 2, nand, is_dram_);
    double gate_leak_L1_nand2 = cmos_Ig_leakage(w_L1_nand2_n[0], w_L1_nand2_p[0], 2, nand, is_dram_);
    if (number_inputs_L1_gate != 3) {
      leak_L1_nand3 = 0;
      gate_leak_L1_nand3 =0;
    }
    else {
      leak_L1_nand3      = cmos_Isub_leakage(w_L1_nand3_n[0], w_L1_nand3_p[0], 3, nand);
      gate_leak_L1_nand3 = cmos_Ig_leakage(w_L1_nand3_n[0], w_L1_nand3_p[0], 3, nand);
    }

    switch (number_input_addr_bits)
    {
      case 1: //2 NAND2 gates
        num_L1_nand2 = 2;
        num_L2       = 0;
        num_L1_active_nand2_path =1;
        num_L1_active_nand3_path =0;
        break;
      case 2: //4 NAND2 gates
        num_L1_nand2 = 4;
        num_L2       = 0;
        num_L1_active_nand2_path =1;
        num_L1_active_nand3_path =0;
        break;
      case 3: //8 NAND3 gates
        num_L1_nand3 = 8;
        num_L2       = 0;
        num_L1_active_nand2_path =0;
        num_L1_active_nand3_path =1;
        break;
      case 4: //4 + 4 NAND2 gates
        num_L1_nand2 = 8;
        num_L2       = 16;
        num_L1_active_nand2_path =2;
        num_L1_active_nand3_path =0;
        break;
      case 5: //4 NAND2 gates, 8 NAND3 gates
        num_L1_nand2 = 4;
        num_L1_nand3 = 8;
        num_L2       = 32;
        num_L1_active_nand2_path =1;
        num_L1_active_nand3_path =1;
        break;
      case 6: //8 + 8 NAND3 gates
        num_L1_nand3 = 16;
        num_L2       = 64;
        num_L1_active_nand2_path =0;
        num_L1_active_nand3_path =2;
        break;
      case 7: //4 + 4 NAND2 gates, 8 NAND3 gates
        num_L1_nand2 = 8;
        num_L1_nand3 = 8;
        num_L2       = 128;
        num_L1_active_nand2_path =2;
        num_L1_active_nand3_path =1;
        break;
      case 8: //4 NAND2 gates, 8 + 8 NAND3 gates
        num_L1_nand2 = 4;
        num_L1_nand3 = 16;
        num_L2       = 256;
        num_L1_active_nand2_path =2;
        num_L1_active_nand3_path =2;
        break;
      case 9: //8 + 8 + 8 NAND3 gates
        num_L1_nand3 = 24;
        num_L2       = 512;
        num_L1_active_nand2_path =0;
        num_L1_active_nand3_path =3;
        break;
      default:
        break;
    }

    for (int i = 1; i < number_gates_L1_nand2_path; ++i)
    {
      leak_L1_nand2      += cmos_Isub_leakage(w_L1_nand2_n[i], w_L1_nand2_p[i], 2, nand, is_dram_);
      gate_leak_L1_nand2 += cmos_Ig_leakage(w_L1_nand2_n[i], w_L1_nand2_p[i], 2, nand, is_dram_);
    }
    leak_L1_nand2      *= num_L1_nand2;
    gate_leak_L1_nand2 *= num_L1_nand2;

    for (int i = 1; i < number_gates_L1_nand3_path; ++i)
    {
      leak_L1_nand3      += cmos_Isub_leakage(w_L1_nand3_n[i], w_L1_nand3_p[i], 3, nand, is_dram_);
      gate_leak_L1_nand3 += cmos_Ig_leakage(w_L1_nand3_n[i], w_L1_nand3_p[i], 3, nand, is_dram_);
    }
    leak_L1_nand3      *= num_L1_nand3;
    gate_leak_L1_nand3 *= num_L1_nand3;

    double leakage_L2         = 0.0;
    double gate_leakage_L2    = 0.0;

    if (flag_L2_gate == 2)
    {
      leakage_L2         = cmos_Isub_leakage(w_L2_n[0], w_L2_p[0], 2, nand, is_dram_);
      gate_leakage_L2    = cmos_Ig_leakage(w_L2_n[0], w_L2_p[0], 2, nand, is_dram_);
    }
    else if (flag_L2_gate == 3)
    {
      leakage_L2         = cmos_Isub_leakage(w_L2_n[0], w_L2_p[0], 3, nand, is_dram_);
      gate_leakage_L2    = cmos_Ig_leakage(w_L2_n[0], w_L2_p[0], 3, nand, is_dram_);
    }

    for (int i = 1; i < number_gates_L2; ++i)
    {
      leakage_L2         += cmos_Isub_leakage(w_L2_n[i], w_L2_p[i], 2, inv, is_dram_);
      gate_leakage_L2    += cmos_Ig_leakage(w_L2_n[i], w_L2_p[i], 2, inv, is_dram_);
    }
    leakage_L2         *= num_L2;
    gate_leakage_L2    *= num_L2;

    power_nand2_path.readOp.leakage = leak_L1_nand2 * g_tp.peri_global.Vdd;
    power_nand3_path.readOp.leakage = leak_L1_nand3 * g_tp.peri_global.Vdd;
    power_L2.readOp.leakage         = leakage_L2    * g_tp.peri_global.Vdd;

    power_nand2_path.readOp.gate_leakage = gate_leak_L1_nand2 * g_tp.peri_global.Vdd;
    power_nand3_path.readOp.gate_leakage = gate_leak_L1_nand3 * g_tp.peri_global.Vdd;
    power_L2.readOp.gate_leakage         = gate_leakage_L2    * g_tp.peri_global.Vdd;
  }
}

PredecBlkDrv::PredecBlkDrv(
    int    way_select_,
    PredecBlk * blk_,
    bool   is_dram)
 :flag_driver_exists(0),
  number_gates_nand2_path(0),
  number_gates_nand3_path(0),
  min_number_gates(2),
  num_buffers_driving_1_nand2_load(0),
  num_buffers_driving_2_nand2_load(0),
  num_buffers_driving_4_nand2_load(0),
  num_buffers_driving_2_nand3_load(0),
  num_buffers_driving_8_nand3_load(0),
  num_buffers_nand3_path(0),
  c_load_nand2_path_out(0),
  c_load_nand3_path_out(0),
  r_load_nand2_path_out(0),
  r_load_nand3_path_out(0),
  delay_nand2_path(0),
  delay_nand3_path(0),
  power_nand2_path(),
  power_nand3_path(),
  blk(blk_), dec(blk->dec),
  is_dram_(is_dram),
  way_select(way_select_)
{
  for (int i = 0; i < MAX_NUMBER_GATES_STAGE; i++)
  {
    width_nand2_path_n[i] = 0;
    width_nand2_path_p[i] = 0;
    width_nand3_path_n[i] = 0;
    width_nand3_path_p[i] = 0;
  }

  number_input_addr_bits = blk->number_input_addr_bits;

  if (way_select > 1)
  {
    flag_driver_exists     = 1;
    number_input_addr_bits = way_select;
    if (dec->num_in_signals == 2)
    {
      c_load_nand2_path_out = gate_C(dec->w_dec_n[0] + dec->w_dec_p[0], 0, is_dram_);
      num_buffers_driving_2_nand2_load = number_input_addr_bits;
    }
    else if (dec->num_in_signals == 3)
    {
      c_load_nand3_path_out = gate_C(dec->w_dec_n[0] + dec->w_dec_p[0], 0, is_dram_);
      num_buffers_driving_2_nand3_load = number_input_addr_bits;
    }
  }
  else if (way_select == 0)
  {
    if (blk->exist)
    {
      flag_driver_exists = 1;
    }
  }

  compute_widths();
  compute_area();
}



void PredecBlkDrv::compute_widths()
{
  // The predecode block driver accepts as input the address bits from the h-tree network. For
  // each addr bit it then generates addr and addrbar as outputs. For now ignore the effect of
  // inversion to generate addrbar and simply treat addrbar as addr.

  double F;
  double p_to_n_sz_ratio = pmos_to_nmos_sz_ratio(is_dram_);

  if (flag_driver_exists)
  {
    double C_nand2_gate_blk = gate_C(blk->w_L1_nand2_n[0] + blk->w_L1_nand2_p[0], 0, is_dram_);
    double C_nand3_gate_blk = gate_C(blk->w_L1_nand3_n[0] + blk->w_L1_nand3_p[0], 0, is_dram_);

    if (way_select == 0)
    {
      if (blk->number_input_addr_bits == 1)
      { //2 NAND2 gates
        num_buffers_driving_2_nand2_load = 1;
        c_load_nand2_path_out            = 2 * C_nand2_gate_blk;
      }
      else if (blk->number_input_addr_bits == 2)
      { //4 NAND2 gates  one 2-4 decoder
        num_buffers_driving_4_nand2_load = 2;
        c_load_nand2_path_out            = 4 * C_nand2_gate_blk;
      }
      else if (blk->number_input_addr_bits == 3)
      { //8 NAND3 gates  one 3-8 decoder
        num_buffers_driving_8_nand3_load = 3;
        c_load_nand3_path_out            = 8 * C_nand3_gate_blk;
      }
      else if (blk->number_input_addr_bits == 4)
      { //4 + 4 NAND2 gates two 2-4 decoder
        num_buffers_driving_4_nand2_load = 4;
        c_load_nand2_path_out            = 4 * C_nand2_gate_blk;
      }
      else if (blk->number_input_addr_bits == 5)
      { //4 NAND2 gates, 8 NAND3 gates one 2-4 decoder and one 3-8 decoder
        num_buffers_driving_4_nand2_load = 2;
        num_buffers_driving_8_nand3_load = 3;
        c_load_nand2_path_out            = 4 * C_nand2_gate_blk;
        c_load_nand3_path_out            = 8 * C_nand3_gate_blk;
      }
      else if (blk->number_input_addr_bits == 6)
      { //8 + 8 NAND3 gates two 3-8 decoder
        num_buffers_driving_8_nand3_load = 6;
        c_load_nand3_path_out            = 8 * C_nand3_gate_blk;
      }
      else if (blk->number_input_addr_bits == 7)
      { //4 + 4 NAND2 gates, 8 NAND3 gates two 2-4 decoder and one 3-8 decoder
        num_buffers_driving_4_nand2_load = 4;
        num_buffers_driving_8_nand3_load = 3;
        c_load_nand2_path_out            = 4 * C_nand2_gate_blk;
        c_load_nand3_path_out            = 8 * C_nand3_gate_blk;
      }
      else if (blk->number_input_addr_bits == 8)
      { //4 NAND2 gates, 8 + 8 NAND3 gates one 2-4 decoder and two 3-8 decoder
        num_buffers_driving_4_nand2_load = 2;
        num_buffers_driving_8_nand3_load = 6;
        c_load_nand2_path_out            = 4 * C_nand2_gate_blk;
        c_load_nand3_path_out            = 8 * C_nand3_gate_blk;
      }
      else if (blk->number_input_addr_bits == 9)
      { //8 + 8 + 8 NAND3 gates three 3-8 decoder
        num_buffers_driving_8_nand3_load = 9;
        c_load_nand3_path_out            = 8 * C_nand3_gate_blk;
      }
    }

    if ((blk->flag_two_unique_paths) ||
        (blk->number_inputs_L1_gate == 2) ||
        (number_input_addr_bits == 0) ||
        ((way_select)&&(dec->num_in_signals == 2)))
    { //this means that way_select is driving NAND2 in decoder.
      width_nand2_path_n[0] = g_tp.min_w_nmos_;
      width_nand2_path_p[0] = p_to_n_sz_ratio * width_nand2_path_n[0];
      F = c_load_nand2_path_out / gate_C(width_nand2_path_n[0] + width_nand2_path_p[0], 0, is_dram_);
      number_gates_nand2_path = logical_effort(
          min_number_gates,
          1,
          F,
          width_nand2_path_n,
          width_nand2_path_p,
          c_load_nand2_path_out,
          p_to_n_sz_ratio,
          is_dram_, false, g_tp.max_w_nmos_);
    }

    if ((blk->flag_two_unique_paths) ||
        (blk->number_inputs_L1_gate == 3) ||
        ((way_select)&&(dec->num_in_signals == 3)))
    { //this means that way_select is driving NAND3 in decoder.
      width_nand3_path_n[0] = g_tp.min_w_nmos_;
      width_nand3_path_p[0] = p_to_n_sz_ratio * width_nand3_path_n[0];
      F = c_load_nand3_path_out / gate_C(width_nand3_path_n[0] + width_nand3_path_p[0], 0, is_dram_);
      number_gates_nand3_path = logical_effort(
          min_number_gates,
          1,
          F,
          width_nand3_path_n,
          width_nand3_path_p,
          c_load_nand3_path_out,
          p_to_n_sz_ratio,
          is_dram_, false, g_tp.max_w_nmos_);
    }
  }
}



void PredecBlkDrv::compute_area()
{
  double area_nand2_path = 0;
  double area_nand3_path = 0;
  double leak_nand2_path = 0;
  double leak_nand3_path = 0;
  double gate_leak_nand2_path = 0;
  double gate_leak_nand3_path = 0;

  if (flag_driver_exists)
  { // first check whether a predecoder block driver is needed
    for (int i = 0; i < number_gates_nand2_path; ++i)
    {
      area_nand2_path += compute_gate_area(INV, 1, width_nand2_path_p[i], width_nand2_path_n[i], g_tp.cell_h_def);
      leak_nand2_path += cmos_Isub_leakage(width_nand2_path_n[i], width_nand2_path_p[i], 1, inv,is_dram_);
      gate_leak_nand2_path += cmos_Ig_leakage(width_nand2_path_n[i], width_nand2_path_p[i], 1, inv,is_dram_);
    }
    area_nand2_path *= (num_buffers_driving_1_nand2_load +
                        num_buffers_driving_2_nand2_load +
                        num_buffers_driving_4_nand2_load);
    leak_nand2_path *= (num_buffers_driving_1_nand2_load +
                        num_buffers_driving_2_nand2_load +
                        num_buffers_driving_4_nand2_load);
    gate_leak_nand2_path *= (num_buffers_driving_1_nand2_load +
                            num_buffers_driving_2_nand2_load +
                            num_buffers_driving_4_nand2_load);

    for (int i = 0; i < number_gates_nand3_path; ++i)
    {
      area_nand3_path += compute_gate_area(INV, 1, width_nand3_path_p[i], width_nand3_path_n[i], g_tp.cell_h_def);
      leak_nand3_path += cmos_Isub_leakage(width_nand3_path_n[i], width_nand3_path_p[i], 1, inv,is_dram_);
      gate_leak_nand3_path += cmos_Ig_leakage(width_nand3_path_n[i], width_nand3_path_p[i], 1, inv,is_dram_);
    }
    area_nand3_path *= (num_buffers_driving_2_nand3_load + num_buffers_driving_8_nand3_load);
    leak_nand3_path *= (num_buffers_driving_2_nand3_load + num_buffers_driving_8_nand3_load);
    gate_leak_nand3_path *= (num_buffers_driving_2_nand3_load + num_buffers_driving_8_nand3_load);

    power_nand2_path.readOp.leakage = leak_nand2_path * g_tp.peri_global.Vdd;
    power_nand3_path.readOp.leakage = leak_nand3_path * g_tp.peri_global.Vdd;
    power_nand2_path.readOp.gate_leakage = gate_leak_nand2_path * g_tp.peri_global.Vdd;
    power_nand3_path.readOp.gate_leakage = gate_leak_nand3_path * g_tp.peri_global.Vdd;
    area.set_area(area_nand2_path + area_nand3_path);
  }
}



pair<double, double> PredecBlkDrv::compute_delays(
    double inrisetime_nand2_path,
    double inrisetime_nand3_path)
{
  pair<double, double> ret_val;
  ret_val.first  = 0;  // outrisetime_nand2_path
  ret_val.second = 0;  // outrisetime_nand3_path
  int i;
  double rd, c_gate_load, c_load, c_intrinsic, tf, this_delay;
  double Vdd = g_tp.peri_global.Vdd;

  if (flag_driver_exists)
  {
    for (i = 0; i < number_gates_nand2_path - 1; ++i)
    {
      rd = tr_R_on(width_nand2_path_n[i], NCH, 1, is_dram_);
      c_gate_load = gate_C(width_nand2_path_p[i+1] + width_nand2_path_n[i+1], 0.0, is_dram_);
      c_intrinsic = drain_C_(width_nand2_path_p[i], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                    drain_C_(width_nand2_path_n[i], NCH, 1, 1, g_tp.cell_h_def, is_dram_);
      tf = rd * (c_intrinsic + c_gate_load);
      this_delay = horowitz(inrisetime_nand2_path, tf, 0.5, 0.5, RISE);
      delay_nand2_path += this_delay;
      inrisetime_nand2_path = this_delay / (1.0 - 0.5);
      power_nand2_path.readOp.dynamic += (c_gate_load + c_intrinsic) * 0.5 * Vdd * Vdd;
    }

    // Final inverter drives the predecoder block or the decoder output load
    if (number_gates_nand2_path != 0)
    {
      i = number_gates_nand2_path - 1;
      rd = tr_R_on(width_nand2_path_n[i], NCH, 1, is_dram_);
      c_intrinsic = drain_C_(width_nand2_path_p[i], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                    drain_C_(width_nand2_path_n[i], NCH, 1, 1, g_tp.cell_h_def, is_dram_);
      c_load = c_load_nand2_path_out;
      tf = rd * (c_intrinsic + c_load) + r_load_nand2_path_out*c_load/ 2;
      this_delay = horowitz(inrisetime_nand2_path, tf, 0.5, 0.5, RISE);
      delay_nand2_path += this_delay;
      ret_val.first = this_delay / (1.0 - 0.5);
      power_nand2_path.readOp.dynamic += (c_intrinsic + c_load) * 0.5 * Vdd * Vdd;
//      cout<< "c_intrinsic = " << c_intrinsic << "c_load" << c_load <<endl;
    }

    for (i = 0; i < number_gates_nand3_path - 1; ++i)
    {
      rd = tr_R_on(width_nand3_path_n[i], NCH, 1, is_dram_);
      c_gate_load = gate_C(width_nand3_path_p[i+1] + width_nand3_path_n[i+1], 0.0, is_dram_);
      c_intrinsic = drain_C_(width_nand3_path_p[i], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                    drain_C_(width_nand3_path_n[i], NCH, 1, 1, g_tp.cell_h_def, is_dram_);
      tf = rd * (c_intrinsic + c_gate_load);
      this_delay = horowitz(inrisetime_nand3_path, tf, 0.5, 0.5, RISE);
      delay_nand3_path += this_delay;
      inrisetime_nand3_path = this_delay / (1.0 - 0.5);
      power_nand3_path.readOp.dynamic += (c_gate_load + c_intrinsic) * 0.5 * Vdd * Vdd;
    }

    // Final inverter drives the predecoder block or the decoder output load
    if (number_gates_nand3_path != 0)
    {
      i = number_gates_nand3_path - 1;
      rd = tr_R_on(width_nand3_path_n[i], NCH, 1, is_dram_);
      c_intrinsic = drain_C_(width_nand3_path_p[i], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                    drain_C_(width_nand3_path_n[i], NCH, 1, 1, g_tp.cell_h_def, is_dram_);
      c_load = c_load_nand3_path_out;
      tf = rd*(c_intrinsic + c_load) + r_load_nand3_path_out*c_load / 2;
      this_delay = horowitz(inrisetime_nand3_path, tf, 0.5, 0.5, RISE);
      delay_nand3_path += this_delay;
      ret_val.second = this_delay / (1.0 - 0.5);
      power_nand3_path.readOp.dynamic += (c_intrinsic + c_load) * 0.5 * Vdd * Vdd;
    }
  }
  return ret_val;
}


double PredecBlkDrv::get_rdOp_dynamic_E(int num_act_mats_hor_dir)
{
  return (num_addr_bits_nand2_path()*power_nand2_path.readOp.dynamic +
          num_addr_bits_nand3_path()*power_nand3_path.readOp.dynamic) * num_act_mats_hor_dir;
}



Predec::Predec(
    PredecBlkDrv * drv1_,
    PredecBlkDrv * drv2_)
:blk1(drv1_->blk), blk2(drv2_->blk), drv1(drv1_), drv2(drv2_)
{
  driver_power.readOp.leakage = drv1->power_nand2_path.readOp.leakage +
                                drv1->power_nand3_path.readOp.leakage +
                                drv2->power_nand2_path.readOp.leakage +
                                drv2->power_nand3_path.readOp.leakage;
  block_power.readOp.leakage = blk1->power_nand2_path.readOp.leakage +
                               blk1->power_nand3_path.readOp.leakage +
                               blk1->power_L2.readOp.leakage +
                               blk2->power_nand2_path.readOp.leakage +
                               blk2->power_nand3_path.readOp.leakage +
                               blk2->power_L2.readOp.leakage;
  power.readOp.leakage = driver_power.readOp.leakage + block_power.readOp.leakage;

  driver_power.readOp.gate_leakage = drv1->power_nand2_path.readOp.gate_leakage +
                                  drv1->power_nand3_path.readOp.gate_leakage +
                                  drv2->power_nand2_path.readOp.gate_leakage +
                                  drv2->power_nand3_path.readOp.gate_leakage;
  block_power.readOp.gate_leakage = blk1->power_nand2_path.readOp.gate_leakage +
                                 blk1->power_nand3_path.readOp.gate_leakage +
                                 blk1->power_L2.readOp.gate_leakage +
                                 blk2->power_nand2_path.readOp.gate_leakage +
                                 blk2->power_nand3_path.readOp.gate_leakage +
                                 blk2->power_L2.readOp.gate_leakage;
  power.readOp.gate_leakage = driver_power.readOp.gate_leakage + block_power.readOp.gate_leakage;
}

void PredecBlkDrv::leakage_feedback(double temperature)
{
  double leak_nand2_path = 0;
  double leak_nand3_path = 0;
  double gate_leak_nand2_path = 0;
  double gate_leak_nand3_path = 0;

  if (flag_driver_exists)
  { // first check whether a predecoder block driver is needed
    for (int i = 0; i < number_gates_nand2_path; ++i)
    {
      leak_nand2_path += cmos_Isub_leakage(width_nand2_path_n[i], width_nand2_path_p[i], 1, inv,is_dram_);
      gate_leak_nand2_path += cmos_Ig_leakage(width_nand2_path_n[i], width_nand2_path_p[i], 1, inv,is_dram_);
    }
    leak_nand2_path *= (num_buffers_driving_1_nand2_load +
                        num_buffers_driving_2_nand2_load +
                        num_buffers_driving_4_nand2_load);
    gate_leak_nand2_path *= (num_buffers_driving_1_nand2_load +
                            num_buffers_driving_2_nand2_load +
                            num_buffers_driving_4_nand2_load);

    for (int i = 0; i < number_gates_nand3_path; ++i)
    {
      leak_nand3_path += cmos_Isub_leakage(width_nand3_path_n[i], width_nand3_path_p[i], 1, inv,is_dram_);
      gate_leak_nand3_path += cmos_Ig_leakage(width_nand3_path_n[i], width_nand3_path_p[i], 1, inv,is_dram_);
    }
    leak_nand3_path *= (num_buffers_driving_2_nand3_load + num_buffers_driving_8_nand3_load);
    gate_leak_nand3_path *= (num_buffers_driving_2_nand3_load + num_buffers_driving_8_nand3_load);

    power_nand2_path.readOp.leakage = leak_nand2_path * g_tp.peri_global.Vdd;
    power_nand3_path.readOp.leakage = leak_nand3_path * g_tp.peri_global.Vdd;
    power_nand2_path.readOp.gate_leakage = gate_leak_nand2_path * g_tp.peri_global.Vdd;
    power_nand3_path.readOp.gate_leakage = gate_leak_nand3_path * g_tp.peri_global.Vdd;
  }
}

double Predec::compute_delays(double inrisetime)
{
  // TODO: Jung Ho thinks that predecoder block driver locates between decoder and predecoder block.
  pair<double, double> tmp_pair1, tmp_pair2;
  tmp_pair1 = drv1->compute_delays(inrisetime, inrisetime);
  tmp_pair1 = blk1->compute_delays(tmp_pair1);
  tmp_pair2 = drv2->compute_delays(inrisetime, inrisetime);
  tmp_pair2 = blk2->compute_delays(tmp_pair2);
  tmp_pair1 = get_max_delay_before_decoder(tmp_pair1, tmp_pair2);

  driver_power.readOp.dynamic =
    drv1->num_addr_bits_nand2_path() * drv1->power_nand2_path.readOp.dynamic +
    drv1->num_addr_bits_nand3_path() * drv1->power_nand3_path.readOp.dynamic +
    drv2->num_addr_bits_nand2_path() * drv2->power_nand2_path.readOp.dynamic +
    drv2->num_addr_bits_nand3_path() * drv2->power_nand3_path.readOp.dynamic;

  block_power.readOp.dynamic =
    blk1->power_nand2_path.readOp.dynamic*blk1->num_L1_active_nand2_path +
    blk1->power_nand3_path.readOp.dynamic*blk1->num_L1_active_nand3_path +
    blk1->power_L2.readOp.dynamic +
    blk2->power_nand2_path.readOp.dynamic*blk1->num_L1_active_nand2_path  +
    blk2->power_nand3_path.readOp.dynamic*blk1->num_L1_active_nand3_path +
    blk2->power_L2.readOp.dynamic;

  power.readOp.dynamic = driver_power.readOp.dynamic + block_power.readOp.dynamic;

  delay = tmp_pair1.first;
  return  tmp_pair1.second;
}


void Predec::leakage_feedback(double temperature)
{
  drv1->leakage_feedback(temperature);
  drv2->leakage_feedback(temperature);
  blk1->leakage_feedback(temperature);
  blk2->leakage_feedback(temperature);

  driver_power.readOp.leakage = drv1->power_nand2_path.readOp.leakage +
                                drv1->power_nand3_path.readOp.leakage +
                                drv2->power_nand2_path.readOp.leakage +
                                drv2->power_nand3_path.readOp.leakage;
  block_power.readOp.leakage = blk1->power_nand2_path.readOp.leakage +
                               blk1->power_nand3_path.readOp.leakage +
                               blk1->power_L2.readOp.leakage +
                               blk2->power_nand2_path.readOp.leakage +
                               blk2->power_nand3_path.readOp.leakage +
                               blk2->power_L2.readOp.leakage;
  power.readOp.leakage = driver_power.readOp.leakage + block_power.readOp.leakage;

  driver_power.readOp.gate_leakage = drv1->power_nand2_path.readOp.gate_leakage +
                                  drv1->power_nand3_path.readOp.gate_leakage +
                                  drv2->power_nand2_path.readOp.gate_leakage +
                                  drv2->power_nand3_path.readOp.gate_leakage;
  block_power.readOp.gate_leakage = blk1->power_nand2_path.readOp.gate_leakage +
                                 blk1->power_nand3_path.readOp.gate_leakage +
                                 blk1->power_L2.readOp.gate_leakage +
                                 blk2->power_nand2_path.readOp.gate_leakage +
                                 blk2->power_nand3_path.readOp.gate_leakage +
                                 blk2->power_L2.readOp.gate_leakage;
  power.readOp.gate_leakage = driver_power.readOp.gate_leakage + block_power.readOp.gate_leakage;
}

// returns <delay, risetime>
pair<double, double> Predec::get_max_delay_before_decoder(
    pair<double, double> input_pair1,
    pair<double, double> input_pair2)
{
  pair<double, double> ret_val;
  double delay;

  delay = drv1->delay_nand2_path + blk1->delay_nand2_path;
  ret_val.first  = delay;
  ret_val.second = input_pair1.first;
  delay = drv1->delay_nand3_path + blk1->delay_nand3_path;
  if (ret_val.first < delay)
  {
    ret_val.first  = delay;
    ret_val.second = input_pair1.second;
  }
  delay = drv2->delay_nand2_path + blk2->delay_nand2_path;
  if (ret_val.first < delay)
  {
    ret_val.first  = delay;
    ret_val.second = input_pair2.first;
  }
  delay = drv2->delay_nand3_path + blk2->delay_nand3_path;
  if (ret_val.first < delay)
  {
    ret_val.first  = delay;
    ret_val.second = input_pair2.second;
  }

  return ret_val;
}



Driver::Driver(double c_gate_load_, double c_wire_load_, double r_wire_load_, bool is_dram)
:number_gates(0),
  min_number_gates(2),
  c_gate_load(c_gate_load_),
  c_wire_load(c_wire_load_),
  r_wire_load(r_wire_load_),
  delay(0),
  power(),
  is_dram_(is_dram)
{
  for (int i = 0; i < MAX_NUMBER_GATES_STAGE; i++)
  {
    width_n[i] = 0;
    width_p[i] = 0;
  }

  compute_widths();
}


void Driver::compute_widths()
{
  double p_to_n_sz_ratio = pmos_to_nmos_sz_ratio(is_dram_);
  double c_load = c_gate_load + c_wire_load;
  width_n[0] = g_tp.min_w_nmos_;
  width_p[0] = p_to_n_sz_ratio * g_tp.min_w_nmos_;

  double F = c_load / gate_C(width_n[0] + width_p[0], 0, is_dram_);
  number_gates = logical_effort(
      min_number_gates,
      1,
      F,
      width_n,
      width_p,
      c_load,
      p_to_n_sz_ratio,
      is_dram_, false,
      g_tp.max_w_nmos_);
}



double Driver::compute_delay(double inrisetime)
{
  int    i;
  double rd, c_load, c_intrinsic, tf;
  double this_delay = 0;

  for (i = 0; i < number_gates - 1; ++i)
  {
    rd = tr_R_on(width_n[i], NCH, 1, is_dram_);
    c_load = gate_C(width_n[i+1] + width_p[i+1], 0.0, is_dram_);
    c_intrinsic = drain_C_(width_p[i], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                  drain_C_(width_n[i], NCH, 1, 1, g_tp.cell_h_def, is_dram_);
    tf = rd * (c_intrinsic + c_load);
    this_delay = horowitz(inrisetime, tf, 0.5, 0.5, RISE);
    delay += this_delay;
    inrisetime = this_delay / (1.0 - 0.5);
    power.readOp.dynamic += (c_intrinsic + c_load) * g_tp.peri_global.Vdd * g_tp.peri_global.Vdd;
    power.readOp.leakage += cmos_Isub_leakage(width_n[i], width_p[i], 1, inv, is_dram_) *g_tp.peri_global.Vdd;
    power.readOp.gate_leakage += cmos_Ig_leakage(width_n[i], width_p[i], 1, inv, is_dram_)* g_tp.peri_global.Vdd;
  }

  i = number_gates - 1;
  c_load = c_gate_load + c_wire_load;
  rd = tr_R_on(width_n[i], NCH, 1, is_dram_);
  c_intrinsic = drain_C_(width_p[i], PCH, 1, 1, g_tp.cell_h_def, is_dram_) +
                drain_C_(width_n[i], NCH, 1, 1, g_tp.cell_h_def, is_dram_);
  tf = rd * (c_intrinsic + c_load) + r_wire_load * (c_wire_load / 2 + c_gate_load);
  this_delay = horowitz(inrisetime, tf, 0.5, 0.5, RISE);
  delay += this_delay;
  power.readOp.dynamic += (c_intrinsic + c_load) * g_tp.peri_global.Vdd * g_tp.peri_global.Vdd;
  power.readOp.leakage += cmos_Isub_leakage(width_n[i], width_p[i], 1, inv, is_dram_) * g_tp.peri_global.Vdd;
  power.readOp.gate_leakage += cmos_Ig_leakage(width_n[i], width_p[i], 1, inv, is_dram_)* g_tp.peri_global.Vdd;

  return this_delay / (1.0 - 0.5);
}

