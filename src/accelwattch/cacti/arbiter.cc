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
 * [한국어 설명] CACTI 아비터 구현 (arbiter.cc)
 *
 * === 파일의 역할 ===
 * MCPAT_Arbiter 클래스의 메서드를 구현한다. R-입력 라운드-로빈 아비터를 CMOS NOR
 * 게이트 체인으로 모델링하여, 각 신호 노드(req, pri, grant, int)의 스위칭 커패시턴스를
 * 계산하고 동적 전력·서브-스레시홀드 누설·게이트 누설 전력을 산출한다. 아비터는
 * NUCA 라우터의 크로스바 앞단에서 여러 요청 포트 중 하나에게 그랜트를 발급하는 역할이다.
 * 회로 레벨 파라미터(트랜지스터 폭, 공급 전압)는 공정 피처 사이즈(F_sz_um)를 기반으로
 * 생성자에서 초기화된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch(accelwattch/) → CACTI 전력 모델 → NUCA 라우터 컴포넌트 → MCPAT_Arbiter.
 * 기능 시뮬레이션(cuda-sim)이나 타이밍 시뮬레이션(gpgpu-sim/shader.cc)과는 분리된
 * 전력 추정 경로에 속한다. compute_power() 결과는 Component::power.readOp.*에 저장되어
 * 상위 라우터/캐시 총 전력에 합산된다.
 *
 * === 타 모듈과의 연결 ===
 * - arbiter.h: 클래스 선언 및 멤버 변수 정의를 포함한다.
 * - basic_circuit.h: gate_C(), drain_C_(), cmos_Isub_leakage(), cmos_Ig_leakage()
 *   로 트랜지스터 레벨 커패시턴스·누설 계산을 위임한다.
 * - parameter.h: 전역 g_ip(F_sz_um, wt), g_tp(min_w_nmos_, cell_h_def) 참조.
 * - wire.h: Cw3() 내부에서 Wire 객체를 생성하여 배선 커패시턴스를 계산한다.
 * - component.h: Component 기반 클래스를 통해 power, area, delay 필드를 보유한다.
 *
 * === 주요 함수/구조체 요약 ===
 * - MCPAT_Arbiter(): 공정 파라미터 기반 트랜지스터 폭 초기화 (NTn1~PTtr).
 * - compute_power(): 동적·누설·게이트 누설 전력을 power.readOp.*에 저장하는 핵심 함수.
 * - arb_req(): 요청 신호가 NOR1/NOR2/NOT를 구동하는 스위칭 커패시턴스 반환.
 * - arb_grant(): 그랜트 출력 + 크로스바 제어선 커패시턴스 반환.
 * - Cw3() / crossbar_ctrline(): 3배 간격 배선 + 인버터 드레인/게이트 커패시턴스 계산.
 */

#include "arbiter.h" // [한국어] MCPAT_Arbiter 클래스 선언

/*
 * [한국어]
 * MCPAT_Arbiter - 생성자: 공정 파라미터 기반 트랜지스터 폭 초기화
 *
 * @n_req:      아비터 입력 요청 수 R (NOR 게이트 팬인).
 * @flit_size_: 플릿 크기(비트). 출력 보고용.
 * @output_len: 크로스바 출력 방향 제어선 길이(µm). crossbar_ctrline() 계산에 사용.
 * @dt:         공정별 디바이스 파라미터 포인터. 기본값 g_tp.peri_global(주변 회로 공정).
 * @return:     없음 (생성자)
 *
 * CACTI 아비터의 NOR1, NOR2, NOT(인버터), 전송 게이트 트랜지스터 폭을 공정
 * 피처 사이즈(F_sz_um)의 배수로 초기화한다. 계수(13.5, 76, 12.5, 25, 10, 20)는
 * MCPAT 원문에서 정의된 회로 설계 상수이다.
 *
 * 호출 체인: NUCA 라우터/Crossbar 모델 → [MCPAT_Arbiter()]
 */
MCPAT_Arbiter::MCPAT_Arbiter(
    double n_req,
    double flit_size_,
    double output_len,
    TechnologyParameter::DeviceType *dt
    ):R(n_req), flit_size(flit_size_),     // [한국어] R에 요청 수, flit_size에 플릿 크기 저장
    o_len (output_len), deviceType(dt)     // [한국어] o_len에 출력 배선 길이, deviceType에 공정 파라미터 포인터 저장
{
  min_w_pmos = deviceType->n_to_p_eff_curr_drv_ratio*g_tp.min_w_nmos_;
  // [한국어] 최소 PMOS 폭 계산: PMOS는 NMOS 대비 n_to_p_eff_curr_drv_ratio배 넓어야
  //          동일 구동 전류를 낼 수 있다 (전자 이동도 차이 보상).
  Vdd = dt->Vdd;
  // [한국어] 공급 전압 저장 — 동적 에너지(½CV²) 및 누설 전력(I×Vdd) 계산에 사용.
  double technology = g_ip->F_sz_um;
  // [한국어] 공정 피처 사이즈(µm) 가져오기 — 트랜지스터 폭의 기준 스케일.
  NTn1 = 13.5*technology/2;
  // [한국어] NOR1 게이트 NMOS 폭: 계수 13.5는 MCPAT 설계 상수 (요청 신호 구동 능력 기준).
  PTn1 = 76*technology/2;
  // [한국어] NOR1 게이트 PMOS 폭: 계수 76 — PMOS 스택이 직렬 연결되어 전류 능력 저하를 보상.
  NTn2 = 13.5*technology/2;
  // [한국어] NOR2 게이트 NMOS 폭: NOR1과 동일한 설계 상수 사용.
  PTn2 = 76*technology/2;
  // [한국어] NOR2 게이트 PMOS 폭: NOR1과 동일한 설계 상수 사용.
  NTi = 12.5*technology/2;
  // [한국어] 제어 인버터(NOT) NMOS 폭: 크로스바 제어선 구동용 인버터 크기.
  PTi = 25*technology/2;
  // [한국어] 제어 인버터(NOT) PMOS 폭: NMOS 대비 2배 (이동도 비율 반영 근사).
  NTtr = 10*technology/2; /*Transmission gate's nmos tr. length*/
  // [한국어] 전송 게이트 NMOS 폭: 크로스바 데이터 경로 트라이스테이트 버퍼 제어선 구동용.
  PTtr = 20*technology/2; /* pmos tr. length*/
  // [한국어] 전송 게이트 PMOS 폭: NMOS 대비 2배로 설정 (구동 대칭성 확보).
}

/*
 * [한국어]
 * ~MCPAT_Arbiter - 소멸자
 *
 * 동적 할당 자원이 없으므로 빈 본문. Component 소멸자 자동 호출.
 * 호출 체인: 객체 파괴 시 자동 호출
 */
MCPAT_Arbiter::~MCPAT_Arbiter(){}

/*
 * [한국어]
 * arb_req - 요청(request) 신호 스위칭 커패시턴스 계산
 *
 * @return: 요청 신호가 NOR1(R-1개) + NOR2 + NOT 게이트를 구동할 때의 총 커패시턴스(F).
 *
 * 라운드-로빈 아비터에서 req 신호는 (R-1)개의 NOR1 게이트에 입력된다.
 * 또한 NOR2의 NMOS·PMOS 게이트와 제어 인버터(NTi/PTi)의 게이트·드레인 커패시턴스도
 * 부하로 작용한다. compute_power()에서 이 값에 R×(Vdd²/2)를 곱해 동적 에너지를 얻는다.
 *
 * 호출 체인: compute_power() → [arb_req()]
 */
double
MCPAT_Arbiter::arb_req() {
  double temp = ((R-1)*(2*gate_C(NTn1, 0)+gate_C(PTn1, 0)) + 2*gate_C(NTn2, 0) +
      gate_C(PTn2, 0) + gate_C(NTi, 0) + gate_C(PTi, 0) +
      drain_C_(NTi, 0, 1, 1, g_tp.cell_h_def) + drain_C_(PTi, 1, 1, 1, g_tp.cell_h_def));
  // [한국어] (R-1)개 NOR1 게이트의 NMOS×2+PMOS 게이트 커패시턴스
  //         + NOR2 게이트의 NMOS×2+PMOS 커패시턴스
  //         + 제어 인버터(NTi/PTi) 게이트 + 드레인 커패시턴스 합산.
  //         drain_C_ 호출: NCH=0→NMOS, PCH=1→PMOS, is_dram=1(표준셀 높이 적용), g_tp.cell_h_def=셀 높이 기준.
  return temp; // [한국어] 총 요청 신호 스위칭 커패시턴스 반환
}

/*
 * [한국어]
 * arb_pri - 우선순위(priority) 테이블 갱신 시 스위칭 커패시턴스 계산
 *
 * @return: 우선순위 상태 갱신 시 NOR1 게이트 2개의 게이트 커패시턴스(F).
 *
 * 라운드-로빈에서 우선순위 테이블은 그랜트 발급 후 한 포트씩 로테이션된다.
 * 이 갱신 과정에서 스위칭되는 NOR1 게이트의 커패시턴스를 반환한다.
 * 플립플롭 내부의 스위칭 커패시턴스는 모델 단순화를 위해 무시한다(주석 참조).
 *
 * 호출 체인: compute_power() → [arb_pri()]
 */
double
MCPAT_Arbiter::arb_pri() {
  double temp = 2*(2*gate_C(NTn1, 0)+gate_C(PTn1, 0)); /* switching capacitance
                                                 of flip-flop is ignored */
  // [한국어] NOR1 게이트 2개(NMOS×2+PMOS 각각)의 게이트 커패시턴스 × 2 (양측 NOR1 고려).
  //         flip-flop 내부 스위칭은 근사 무시. compute_power()에서 R×(Vdd²/2) 곱해짐.
  return temp; // [한국어] 우선순위 신호 스위칭 커패시턴스 반환
}


/*
 * [한국어]
 * arb_grant - 그랜트(grant) 출력 노드 스위칭 커패시턴스 계산
 *
 * @return: 그랜트 신호 출력선 + 크로스바 제어선 전체 커패시턴스(F).
 *
 * NOR1 출력 노드(드레인 2×NMOS + 1×PMOS)에 crossbar_ctrline()의 부하가 달린다.
 * compute_power()에서 이 값에 Vdd²(계수 1, 즉 매 사이클 1회 전이)를 곱해 에너지를 구한다.
 *
 * 호출 체인: compute_power() → [arb_grant()] → crossbar_ctrline()
 */
double
MCPAT_Arbiter::arb_grant() {
  double temp = drain_C_(NTn1, 0, 1, 1, g_tp.cell_h_def)*2 + drain_C_(PTn1, 1, 1, 1, g_tp.cell_h_def) + crossbar_ctrline();
  // [한국어] NOR1 출력 드레인: NMOS 드레인×2 + PMOS 드레인×1 (직렬 스택 구조).
  //         crossbar_ctrline(): 크로스바 제어선 배선 + 인버터 드레인/게이트 커패시턴스 합산.
  return temp; // [한국어] 그랜트 출력 노드 총 커패시턴스 반환
}

/*
 * [한국어]
 * arb_int - 아비터 내부 중간 노드 스위칭 커패시턴스 계산
 *
 * @return: NOR1 출력 → NOR2 입력 사이 내부 노드의 총 커패시턴스(F).
 *
 * NOR1의 드레인 커패시턴스(2×NMOS + 1×PMOS)와 NOR2 게이트의 입력 커패시턴스를
 * 합산한다. 내부 노드는 평균 0.5 활동율을 가정하여 compute_power()에서 ×0.5 한다.
 *
 * 호출 체인: compute_power() → [arb_int()]
 */
double
MCPAT_Arbiter::arb_int() {
  double temp  =  (drain_C_(NTn1, 0, 1, 1, g_tp.cell_h_def)*2 + drain_C_(PTn1, 1, 1, 1, g_tp.cell_h_def) +
      2*gate_C(NTn2, 0) + gate_C(PTn2, 0));
  // [한국어] NOR1 드레인(NMOS×2 + PMOS×1) + NOR2 게이트 입력 커패시턴스(NMOS×2 + PMOS×1).
  //         이 내부 노드는 arb_req 경로와 arb_grant 경로 사이에 위치한다.
  return temp; // [한국어] 내부 노드 총 커패시턴스 반환
}

/*
 * [한국어]
 * compute_power - 아비터 전체 전력 계산 (동적 + 누설 + 게이트 누설)
 *
 * @return: void. power.readOp.dynamic / leakage / gate_leakage에 결과를 저장한다.
 *
 * 1단계: 각 신호 노드(req, pri, grant, int)의 스위칭 커패시턴스를 얻어 ½CV² 공식으로
 *        동적 에너지를 계산한다. req와 pri는 R개 포트 각각 발생(×R), grant는 1회,
 *        int는 0.5 활동율 가정(×0.5).
 * 2단계: cmos_Isub_leakage()로 각 게이트(nor1×R, nor2, not)의 서브-스레시홀드 누설 전류를
 *        구하고 Vdd를 곱해 전력(W)으로 변환한다.
 * 3단계: cmos_Ig_leakage()로 게이트 산화막 터널링 누설 전력을 계산한다.
 * NOTE: 우선순위 테이블(플립플롭) 누설은 현재 미포함(FIXME 주석 참조).
 *
 * 호출 체인: NUCA 라우터 전력 집계 → [compute_power()]
 */
void
MCPAT_Arbiter::compute_power() {
  power.readOp.dynamic =  (R*arb_req()*Vdd*Vdd/2 + R*arb_pri()*Vdd*Vdd/2 +
      arb_grant()*Vdd*Vdd + arb_int()*0.5*Vdd*Vdd);
  // [한국어] 동적 에너지 합산:
  //   R×arb_req()×Vdd²/2 — R개 요청 신호 각각의 스위칭 에너지 (½CV²)
  //   R×arb_pri()×Vdd²/2 — R개 우선순위 갱신 에너지 (½CV²)
  //   arb_grant()×Vdd²   — 그랜트 신호 1회 전이 에너지 (CV², 전이 1회로 가정)
  //   arb_int()×0.5×Vdd² — 내부 노드 에너지 (50% 활동율 가정)
  double nor1_leak = cmos_Isub_leakage(g_tp.min_w_nmos_*NTn1*2, min_w_pmos * PTn1*2, 2, nor);
  // [한국어] NOR1 게이트(2-입력 NOR)의 서브-스레시홀드 누설 전류: NMOS 폭=min_w_nmos_×NTn1×2 (스택 보정),
  //         PMOS 폭=min_w_pmos×PTn1×2, 스택 깊이=2, 게이트 종류=nor.
  double nor2_leak = cmos_Isub_leakage(g_tp.min_w_nmos_*NTn2*R, min_w_pmos * PTn2*R, 2, nor);
  // [한국어] NOR2 게이트의 누설: R개 입력이 병렬 연결되어 NMOS 폭에 R을 곱함.
  double not_leak = cmos_Isub_leakage(g_tp.min_w_nmos_*NTi, min_w_pmos * PTi, 1, inv);
  // [한국어] 제어 인버터(NOT)의 서브-스레시홀드 누설: 스택 깊이=1, 게이트 종류=inv.
  double nor1_leak_gate = cmos_Ig_leakage(g_tp.min_w_nmos_*NTn1*2, min_w_pmos * PTn1*2, 2, nor);
  // [한국어] NOR1의 게이트 산화막 터널링 누설 전류 (게이트 누설, 얇은 산화막에서 중요).
  double nor2_leak_gate = cmos_Ig_leakage(g_tp.min_w_nmos_*NTn2*R, min_w_pmos * PTn2*R, 2, nor);
  // [한국어] NOR2의 게이트 누설 전류.
  double not_leak_gate  = cmos_Ig_leakage(g_tp.min_w_nmos_*NTi, min_w_pmos * PTi, 1, inv);
  // [한국어] 인버터의 게이트 누설 전류.
  power.readOp.leakage = (nor1_leak + nor2_leak + not_leak)*Vdd; //FIXME include priority table leakage
  // [한국어] 총 서브-스레시홀드 누설 전력(W) = 누설 전류(A) × Vdd(V).
  //         우선순위 테이블(플립플롭) 누설은 현재 미포함(FIXME).
  power.readOp.gate_leakage = nor1_leak_gate*Vdd + nor2_leak_gate*Vdd + not_leak_gate*Vdd;
  // [한국어] 총 게이트 누설 전력(W) = 각 게이트 누설 전류(A) × Vdd(V) 합산.
}

/*
 * [한국어]
 * Cw3 - 3배 간격(triple-spaced) 배선 커패시턴스 계산
 *
 * @length: 배선 길이(µm). 함수 내부에서 m 단위로 변환하지 않고 Wire 생성자에 µm 직접 전달.
 * @return: 해당 길이의 3배 간격 배선 커패시턴스(F).
 *
 * Wire 모델(g_ip->wt 배선 타입, 폭 배수=1, 간격 배수=3, 3)을 생성한 뒤
 * wire_cap()으로 커패시턴스를 반환한다. 3배 간격은 라우터 제어선처럼 노이즈에
 * 민감한 신호에 적용하여 크로스토크를 줄인다.
 *
 * 호출 체인: crossbar_ctrline() → [Cw3()] → Wire::wire_cap()
 */
double //wire cap with triple spacing
MCPAT_Arbiter::Cw3(double length) {
  Wire wc(g_ip->wt, length, 1, 3, 3);
  // [한국어] Wire 객체 생성: 배선 타입=g_ip->wt, 길이=length(µm), 폭 배수=1,
  //         수평 간격 배수=3, 수직 간격 배수=3 (노이즈 마진 확보를 위한 3배 간격).
  double temp = (wc.wire_cap(length,true));
  // [한국어] wire_cap(): 배선의 단위 길이 커패시턴스 × length. true=전체 배선 커패시턴스 계산.
  return temp; // [한국어] 3배 간격 배선의 총 커패시턴스(F) 반환
}

/*
 * [한국어]
 * crossbar_ctrline - 크로스바 제어선 부하 커패시턴스 계산
 *
 * @return: 배선 커패시턴스(Cw3) + 인버터(NTi/PTi) 드레인/게이트 커패시턴스 합계(F).
 *
 * 아비터의 그랜트 신호가 크로스바의 입력 선택 스위치를 제어하는 선의 부하를 모델링한다.
 * o_len(µm)을 미터로 변환(×1e-6)하여 Cw3에 전달하고, 제어 인버터 드레인·게이트를 더한다.
 *
 * 호출 체인: arb_grant() → [crossbar_ctrline()] → Cw3()
 */
double
MCPAT_Arbiter::crossbar_ctrline() {
  double temp = (Cw3(o_len * 1e-6 /* m */) +
      drain_C_(NTi, 0, 1, 1, g_tp.cell_h_def) + drain_C_(PTi, 1, 1, 1, g_tp.cell_h_def) +
      gate_C(NTi, 0) + gate_C(PTi, 0));
  // [한국어] Cw3(o_len×1e-6): 크로스바 출력 방향 제어선 배선 커패시턴스 (µm→m 변환).
  //         drain_C_(NTi, NCH=0): 제어 인버터 NMOS 드레인 커패시턴스.
  //         drain_C_(PTi, PCH=1): 제어 인버터 PMOS 드레인 커패시턴스.
  //         gate_C(NTi/PTi): 제어 인버터 NMOS/PMOS 게이트 입력 커패시턴스.
  return temp; // [한국어] 크로스바 제어선 총 부하 커패시턴스 반환
}

/*
 * [한국어]
 * transmission_buf_ctrcap - 전송 게이트(전송 버퍼) 제어 입력 커패시턴스 계산
 *
 * @return: 전송 게이트 NMOS + PMOS 게이트 커패시턴스 합계(F).
 *
 * 크로스바 데이터 경로에서 트라이스테이트 버퍼(전송 게이트)를 온/오프하는
 * 제어 신호가 구동해야 하는 게이트 입력 커패시턴스를 반환한다.
 * Crossbar 모델에서 크로스바 스위치 당 부하 추정에 사용된다.
 *
 * 호출 체인: Crossbar 전력 계산 → [transmission_buf_ctrcap()]
 */
double
MCPAT_Arbiter::transmission_buf_ctrcap() {
  double temp = gate_C(NTtr, 0)+gate_C(PTtr, 0);
  // [한국어] gate_C(NTtr): 전송 게이트 NMOS 게이트 커패시턴스.
  //         gate_C(PTtr): 전송 게이트 PMOS 게이트 커패시턴스.
  //         두 값의 합이 제어 신호 1비트당 구동 부하.
  return temp; // [한국어] 전송 게이트 제어 커패시턴스 반환
}


/*
 * [한국어]
 * print_arbiter - 아비터 통계 출력
 *
 * @return: void
 *
 * 입력 요청 수(R), 플릿 크기, 동적 전력(나노줄 단위), 누설 전력(밀리와트 단위)을
 * 표준 출력에 인쇄한다. 디버깅 및 최종 결과 보고용으로 사용된다.
 *
 * 호출 체인: (CACTI 결과 출력 루틴) → [print_arbiter()]
 */
void MCPAT_Arbiter::print_arbiter()
{
  cout << "\nMCPAT_Arbiter Stats ("   << R << " input arbiter" << ")\n\n";
  // [한국어] 아비터 식별 헤더 출력: 입력 요청 수 R 포함.
  cout << "Flit size        : " << flit_size << " bits" << endl;
  // [한국어] 서비스하는 플릿 크기(비트) 출력.
  cout << "Dynamic Power    : " << power.readOp.dynamic*1e9 << " (nJ)" << endl;
  // [한국어] 동적 에너지(J)를 나노줄(nJ)로 변환하여 출력.
  cout << "Leakage Power    : " << power.readOp.leakage*1e3 << " (mW)" << endl;
  // [한국어] 누설 전력(W)을 밀리와트(mW)로 변환하여 출력.
}


