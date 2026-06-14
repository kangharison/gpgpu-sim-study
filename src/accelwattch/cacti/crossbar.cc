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
 * [한국어 설명] CACTI 크로스바(Crossbar) 모델 구현 (crossbar.cc)
 *
 * === 파일의 역할 ===
 * crossbar.h에 선언된 Crossbar 클래스의 생성자, output_buffer(), compute_power(),
 * print_crossbar()를 구현한다. 크로스바는 n_inp × n_out 교차점에 트라이스테이트(tri-state)
 * 버퍼를 배치하여 임의의 입출력 연결을 지원하는 스위칭 패브릭이다.
 * 전력/면적/지연 분석을 단계별로 수행하며, 종횡비 조정을 위한 재귀적 compute_power()를 포함.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch → CACTI → crossbar.cc (스위칭 패브릭 전력/면적/지연 모델 구현).
 * 호출 체인: 상위 캐시 분석 모듈 → Crossbar 생성자 → [compute_power()]
 *   → [output_buffer()] → Wire, gate_C, drain_C_, horowitz
 * 실행 컨텍스트: 호스트 유저스페이스.
 *
 * === 타 모듈과의 연결 ===
 * 의존: Wire(wire.h) — 배선 RC 및 반복기 파라미터,
 *   Component::compute_gate_area() — 게이트 레이아웃 면적,
 *   basic_circuit.h 함수들 — 게이트 커패시턴스/저항/지연/누설 계산.
 *
 * === 주요 함수/구조체 요약 ===
 * Crossbar()      : 생성자 — min_w_pmos, Vdd, CB_ADJ 초기화
 * output_buffer() : 트라이스테이트 버퍼의 커패시턴스와 TriS1/TriS2 크기 계산
 * compute_power() : 전체 동적/누설 전력, 면적, 지연 계산 (종횡비 재귀 조정 포함)
 * print_crossbar(): 결과 출력
 */

#include "crossbar.h" // [한국어] Crossbar 클래스 선언

/* [한국어] ASPECT_THRESHOLD = 0.8 — 크로스바 종횡비(aspect ratio) 허용 최솟값.
 * 종횡비가 이 값 미만이면 CB_ADJ를 키워 재귀적으로 재계산하여 레이아웃을 정사각형에 가깝게 만든다. */
#define ASPECT_THRESHOLD .8

/* [한국어] ADJ = 1 — 와이어 길이 계산 시 반복기 간격 초과 여부를 결정하는 보정 인수 */
#define ADJ 1

/*
 * [한국어]
 * Crossbar::Crossbar - 크로스바 생성자
 * @n_inp_    : 입력 포트 수
 * @n_out_    : 출력 포트 수
 * @flit_size_: 플릿(flit) 크기 (비트) — 한 번에 전송되는 데이터 폭
 * @dt        : 트랜지스터 파라미터 포인터 (기본값: g_tp.peri_global)
 * 초기화 목록에서 n_inp, n_out, flit_size, deviceType을 설정하고,
 * min_w_pmos(PMOS 최소 폭), Vdd(공급 전압), CB_ADJ(초기값 1)를 멤버에 저장.
 * 호출 체인: CACTI 크로스바 분석 → [Crossbar()] → compute_power()
 */
Crossbar::Crossbar(
    double n_inp_,
    double n_out_,
    double flit_size_,
    TechnologyParameter::DeviceType *dt
    ):n_inp(n_inp_), n_out(n_out_), flit_size(flit_size_), deviceType(dt)
{
  min_w_pmos = deviceType->n_to_p_eff_curr_drv_ratio*g_tp.min_w_nmos_;
  // [한국어] PMOS 최소 폭 = P/N 전류비 × NMOS 최소 폭 — 동등 구동력 확보
  Vdd = dt->Vdd; // [한국어] 공급 전압 (V) — 동적 에너지 계산(C*Vdd²)에 사용
  CB_ADJ = 1;    // [한국어] 종횡비 조정 인수 초기값 = 1 (조정 없음)
}

Crossbar::~Crossbar(){} // [한국어] 소멸자 — 동적 할당 없으므로 빈 구현

/*
 * [한국어]
 * Crossbar::output_buffer - 트라이스테이트 버퍼의 커패시턴스 계산 및 게이트 크기 결정
 * @return : 교차점 하나의 총 커패시턴스 = input_cap + output_cap + ctr_cap (F)
 *
 * 알고리즘:
 *   1) 입력 와이어 유효 길이 l_eff = n_inp * flit_size * wire_pitch
 *   2) Wire 객체로 반복기 크기(repeater_size)를 구함
 *   3) 반복기 크기를 기반으로 트라이스테이트 제어 게이트(TriS1)와 드라이버(TriS2) 크기 결정:
 *      TriS1 = s1 * (NAND+NOR 분배 비율) — NAND/NOR 제어 게이트용
 *      TriS2 = s1 — 출력 드라이버(인버터) 크기
 *   4) 트라이스테이트 버퍼 내부 노드 커패시턴스(tri_int_cap): 모든 내부 드레인/게이트 합산
 *   5) 출력/입력/제어 커패시턴스를 tri_*_cap 멤버에 저장하고 합계 반환
 * 호출 체인: compute_power → [output_buffer] → Wire, gate_C, drain_C_
 */
double Crossbar::output_buffer()
{

  //Wire winit(4, 4);
  double l_eff = n_inp*flit_size*g_tp.wire_outside_mat.pitch;
  // [한국어] 유효 와이어 길이 = 입력 수 × 플릿 비트 수 × MAT 외부 배선 피치
  //         크로스바 입력 버스가 n_inp개 입력의 피치 합만큼 길다고 가정
  Wire w1(g_ip->wt, l_eff); // [한국어] 해당 길이의 와이어 객체 생성 — 반복기 파라미터 계산
  //double s1 = w1.repeater_size *l_eff*ADJ/w1.repeater_spacing;
  double s1 = w1.repeater_size * (l_eff <w1.repeater_spacing?  l_eff *ADJ/w1.repeater_spacing : ADJ);
  // [한국어] 효과적 반복기 크기: 와이어가 반복기 간격보다 짧으면 선형 보간, 아니면 ADJ배
  double pton_size = deviceType->n_to_p_eff_curr_drv_ratio;
  // [한국어] P/N 전류비 = PMOS 폭 / NMOS 폭 기준값
  // the model assumes input capacitance of the wire driver = input capacitance of nand + nor = input cap of the driver transistor
  // [한국어] 가정: 와이어 드라이버의 입력 커패시턴스 = NAND+NOR 게이트 입력 커패시턴스
  TriS1 = s1*(1 + pton_size)/(2 + pton_size + 1 + 2*pton_size);
  // [한국어] NAND+NOR 제어 게이트 크기: 전체 트라이스테이트에서 제어 게이트 비율만큼 할당
  //   분자 (1+pton_size): 인버터 1개 분 (NMOS 1 + PMOS pton)
  //   분모 (2+pton+1+2*pton): NAND2(NMOS*2+PMOS) + NOR2(NMOS+PMOS*2) 합계 폭
  TriS2 = s1; //driver transistor
  // [한국어] 출력 드라이버(인버터) 크기 = 반복기 전체 크기 (s1)

  if (TriS1 < 1)
    TriS1 = 1; // [한국어] TriS1이 최소 트랜지스터 크기 이하이면 1로 클램핑

  double input_cap = gate_C(TriS1*(2*min_w_pmos + g_tp.min_w_nmos_), 0) +
    gate_C(TriS1*(min_w_pmos + 2*g_tp.min_w_nmos_), 0);
  // [한국어] 트라이스테이트 제어 입력 커패시턴스:
  //   NAND2 게이트 입력 = TriS1*(2*min_w_pmos + min_w_nmos_)
  //   NOR2 게이트 입력  = TriS1*(min_w_pmos + 2*min_w_nmos_)

  tri_int_cap = drain_C_(TriS1*g_tp.min_w_nmos_, NCH, 1, 1, g_tp.cell_h_def) +
    drain_C_(TriS1*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def)*2 +
    gate_C(TriS2*g_tp.min_w_nmos_, 0)+
    drain_C_(TriS1*min_w_pmos, NCH, 1, 1, g_tp.cell_h_def)*2 +
    drain_C_(TriS1*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def) +
    gate_C(TriS2*min_w_pmos, 0);
  // [한국어] 트라이스테이트 내부 노드 커패시턴스:
  //   NAND2 출력 드레인: NCH drain + 2*PCH drain (NAND2 구조: NMOS 1개, PMOS 2개 병렬)
  //   NOR2 출력 게이트: TriS2 NMOS + PMOS 게이트
  //   NOR2 입력 드레인: 2*PCH drain + PCH drain (NOR2 구조: PMOS 2개 직렬, NMOS 2개 병렬)

  double output_cap = drain_C_(TriS2*g_tp.min_w_nmos_, NCH, 1, 1, g_tp.cell_h_def) +
    drain_C_(TriS2*min_w_pmos, PCH, 1, 1, g_tp.cell_h_def);
  // [한국어] 트라이스테이트 출력 드라이버(인버터)의 드레인 커패시턴스 합

  double ctr_cap = gate_C(TriS2 *(min_w_pmos + g_tp.min_w_nmos_), 0);
  // [한국어] 제어 신호가 구동하는 출력 드라이버 게이트 커패시턴스

  tri_inp_cap = input_cap;  // [한국어] 입력 커패시턴스 멤버 저장
  tri_out_cap = output_cap; // [한국어] 출력 커패시턴스 멤버 저장
  tri_ctr_cap = ctr_cap;    // [한국어] 제어 커패시턴스 멤버 저장
  return input_cap + output_cap + ctr_cap; // [한국어] 교차점 총 커패시턴스 반환
}

/*
 * [한국어]
 * Crossbar::compute_power - 크로스바 전체 동적/누설 전력, 면적, 지연 계산
 *
 * 알고리즘:
 *   1) output_buffer()로 트라이스테이트 버퍼 커패시턴스 및 크기(TriS1, TriS2) 계산
 *   2) 트라이스테이트 셀 게이트 면적 합산: INV(출력 드라이버 ×2) + NAND2 + NOR2
 *   3) 셀 폭 = 면적 / (CB_ADJ × 셀 높이) — CB_ADJ로 종횡비 조정
 *   4) 와이어 폭(area.w)과 높이(area.h) 계산 후 종횡비 검사:
 *      종횡비 < ASPECT_THRESHOLD(0.8)이면 CB_ADJ += 0.2 후 재귀 호출 (CB_ADJ < 4 한계)
 *   5) 동적 전력 = (w1 + w2 와이어 전력 + 커패시턴스 충전) × flit_size
 *   6) 누설 전력 = n_inp × n_out × flit_size × (INV+NAND+NOR 누설×Vdd + 와이어 누설)
 *   7) 지연 = horowitz(반복기 슬루율, res×cap, Vth/Vdd, Vth/Vdd, RISE)
 *      res = 와이어 저항 + 드라이버 온 저항, cap = 와이어 커패시턴스 + 버퍼 커패시턴스
 * 호출 체인: 크로스바 분석 코드 → [compute_power] → output_buffer, Wire, horowitz
 */
void Crossbar::compute_power()
{

  Wire winit(4, 4); // [한국어] 와이어 파라미터 초기화용 더미 Wire 객체 (실제 사용 안 됨)
  double tri_cap = output_buffer(); // [한국어] 트라이스테이트 커패시턴스 계산 및 TriS1/TriS2 설정
  assert(tri_cap > 0); // [한국어] 커패시턴스는 반드시 양수여야 함 (물리적으로 당연)

  //area of a tristate logic
  // [한국어] 트라이스테이트 버퍼 하나의 게이트 면적 계산
  double g_area = compute_gate_area(INV, 1, TriS2*g_tp.min_w_nmos_, TriS2*min_w_pmos, g_tp.cell_h_def);
  // [한국어] 출력 드라이버 인버터(INV) 면적 (NMOS + PMOS, 셀 높이 기준)
  g_area *= 2; // to model area of output transistors
  // [한국어] 출력 트랜지스터(패스 NMOS + 패스 PMOS) 면적 추가 = INV 면적 × 2
  g_area += compute_gate_area (NAND, 2, TriS1*2*g_tp.min_w_nmos_, TriS1*min_w_pmos, g_tp.cell_h_def);
  // [한국어] NAND2 제어 게이트 면적 추가
  g_area += compute_gate_area (NOR, 2, TriS1*g_tp.min_w_nmos_, TriS1*2*min_w_pmos, g_tp.cell_h_def);
  // [한국어] NOR2 제어 게이트 면적 추가
  double width /*per tristate*/ = g_area/(CB_ADJ * g_tp.cell_h_def);
  // [한국어] 트라이스테이트 셀 하나의 폭 = 면적 / (CB_ADJ × 셀 높이)
  //         CB_ADJ 증가 시 셀이 더 높고 얇아짐 → 크로스바 높이 증가

  // effective no. of tristate buffers that need to be laid side by side
  int ntri = (int)ceil(g_tp.cell_h_def/(g_tp.wire_outside_mat.pitch));
  // [한국어] 셀 높이를 와이어 피치로 나눈 수 = 셀 하나의 높이를 채우려면 몇 개의 버퍼가 나란히 있어야 하는지
  double wire_len = MAX(width*ntri*n_out, flit_size*g_tp.wire_outside_mat.pitch*n_out);
  // [한국어] 크로스바 폭 = MAX(버퍼 배치 필요 폭, 출력 배선 필요 폭)
  Wire w1(g_ip->wt, wire_len); // [한국어] 크로스바 수평 방향 와이어 (출력 버스)

  area.w = wire_len; // [한국어] 크로스바 폭 설정
  area.h = g_tp.wire_outside_mat.pitch*n_inp*flit_size * CB_ADJ;
  // [한국어] 크로스바 높이 = 와이어 피치 × 입력 수 × 플릿 크기 × CB_ADJ
  Wire w2(g_ip->wt, area.h); // [한국어] 크로스바 수직 방향 와이어 (입력 버스)

  double aspect_ratio_cb = (area.h/area.w)*(n_out/n_inp);
  // [한국어] 크로스바 종횡비: (높이/폭) × (출력/입력) — 사각형에 가까울수록 배선 효율 좋음
  if (aspect_ratio_cb > 1) aspect_ratio_cb = 1/aspect_ratio_cb;
  // [한국어] 종횡비를 항상 1 이하로 정규화 (min(h/w, w/h) 형태)

  if (aspect_ratio_cb < ASPECT_THRESHOLD) {
    // [한국어] 종횡비가 임계값 미만 — 레이아웃이 너무 길쭉함
    if (n_out > 2 && n_inp > 2) {
      // [한국어] 입출력이 각 3개 이상인 경우에만 조정 시도 (소규모 크로스바는 무조건 통과)
      CB_ADJ+=0.2; // [한국어] CB_ADJ를 0.2 증가하여 셀을 더 높게 — 크로스바 높이 증가
      //cout << "CB ADJ " << CB_ADJ << endl;
      if (CB_ADJ < 4) {
        // [한국어] CB_ADJ가 4 미만이면 재귀 호출하여 새 CB_ADJ로 재계산
        this->compute_power();
      }
    }
  }

  // [한국어] 동적 전력(에너지/접근) 계산:
  //   w1(수평 와이어) + w2(수직 와이어) 동적 전력 +
  //   교차점 커패시턴스 충전 에너지(모든 입/출력/제어/내부 커패시턴스)
  //   전체에 flit_size를 곱 (flit_size 비트 폭)
  power.readOp.dynamic = (w1.power.readOp.dynamic + w2.power.readOp.dynamic + (tri_inp_cap * n_out + tri_out_cap * n_inp + tri_ctr_cap + tri_int_cap) * Vdd*Vdd)*flit_size;

  power.readOp.leakage      =  n_inp * n_out * flit_size * (
    cmos_Isub_leakage(g_tp.min_w_nmos_*TriS2*2, min_w_pmos*TriS2*2, 1, inv) *Vdd+
    // [한국어] 출력 드라이버(INV) 서브임계 누설 × Vdd
	cmos_Isub_leakage(g_tp.min_w_nmos_*TriS1*3, min_w_pmos*TriS1*3, 2, nand)*Vdd+
	// [한국어] NAND2 제어 게이트 서브임계 누설 × Vdd (3배 폭은 NAND2 스택 크기)
	cmos_Isub_leakage(g_tp.min_w_nmos_*TriS1*3, min_w_pmos*TriS1*3, 2, nor) *Vdd+
	// [한국어] NOR2 제어 게이트 서브임계 누설 × Vdd
    w1.power.readOp.leakage + w2.power.readOp.leakage);
    // [한국어] 수평+수직 와이어 누설 (반복기 포함)

  power.readOp.gate_leakage = n_inp * n_out * flit_size * (
	  cmos_Ig_leakage(g_tp.min_w_nmos_*TriS2*2, min_w_pmos*TriS2*2, 1, inv) *Vdd+
	  // [한국어] 출력 드라이버 게이트 터널링 누설
	  cmos_Ig_leakage(g_tp.min_w_nmos_*TriS1*3, min_w_pmos*TriS1*3, 2, nand)*Vdd+
	  // [한국어] NAND2 게이트 터널링 누설
	  cmos_Ig_leakage(g_tp.min_w_nmos_*TriS1*3, min_w_pmos*TriS1*3, 2, nor) *Vdd+
	  // [한국어] NOR2 게이트 터널링 누설
	  w1.power.readOp.gate_leakage + w2.power.readOp.gate_leakage);
	  // [한국어] 와이어 게이트 터널링 누설

  // delay calculation
  // [한국어] 지연 계산: 드라이버 RC 모델 기반
  double l_eff = n_inp*flit_size*g_tp.wire_outside_mat.pitch;
  // [한국어] 입력 버스 유효 와이어 길이 (output_buffer와 동일)
  Wire wdriver(g_ip->wt, l_eff); // [한국어] 드라이버 반복기 파라미터 계산용 Wire
  double res = g_tp.wire_outside_mat.R_per_um * (area.w+area.h) + tr_R_on(g_tp.min_w_nmos_*wdriver.repeater_size, NCH, 1);
  // [한국어] 총 저항 = 와이어 저항(단위저항 × 총 길이) + 드라이버 NMOS 온 저항
  double cap = g_tp.wire_outside_mat.C_per_um * (area.w + area.h) + n_out*tri_inp_cap + n_inp*tri_out_cap;
  // [한국어] 총 커패시턴스 = 와이어 커패시턴스 + n_out개 입력 버퍼 + n_inp개 출력 버퍼 부하
  delay = horowitz(w1.signal_rise_time(), res*cap, deviceType->Vth/deviceType->Vdd, deviceType->Vth/deviceType->Vdd, RISE);
  // [한국어] Horowitz 모델: 입력 슬루율과 RC 시정수로 전파 지연 계산
  //         vs1=vs2=Vth/Vdd — 50% 임계전압 기준 지연

  Wire wreset(); // [한국어] 와이어 상태 초기화 (Wire 소멸자 호출 용도 — 실제로는 임시 객체)
}

/*
 * [한국어]
 * Crossbar::print_crossbar - 크로스바 분석 결과를 표준 출력으로 출력
 * 크로스바 크기(n_inp×n_out), 플릿 크기, 폭/높이, 동적/누설/게이트누설 전력, 지연 출력.
 * 동적 전력은 MIN(n_inp, n_out)배 하여 실제 사용 교차점 수 반영.
 * 호출 체인: 검증/디버깅 → [print_crossbar]
 */
void Crossbar::print_crossbar()
{
  cout << "\nCrossbar Stats (" << n_inp << "x" << n_out << ")\n\n";
  // [한국어] 크로스바 크기 출력
  cout << "Flit size        : " << flit_size << " bits" << endl;
  // [한국어] 플릿(데이터 폭) 크기 출력 (비트)
  cout << "Width            : " << area.w << " u" << endl;
  // [한국어] 크로스바 폭 출력 (um)
  cout << "Height           : " << area.h << " u" << endl;
  // [한국어] 크로스바 높이 출력 (um)
  cout << "Dynamic Power    : " << power.readOp.dynamic*1e9 * MIN(n_inp, n_out) << " (nJ)" << endl;
  // [한국어] 동적 전력: nJ 단위로 변환, MIN(입력,출력)개 교차점 동시 활성 가정
  cout << "Leakage Power    : " << power.readOp.leakage*1e3 << " (mW)" << endl;
  // [한국어] 서브임계 누설 전력 (mW)
  cout << "Gate Leakage Power    : " << power.readOp.gate_leakage*1e3 << " (mW)" << endl;
  // [한국어] 게이트 터널링 누설 전력 (mW)
  cout << "Crossbar Delay   : " << delay*1e12 << " ps\n";
  // [한국어] 크로스바 전파 지연 (ps)
}


