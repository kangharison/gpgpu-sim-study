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
 * [한국어 설명] CACTI 기본 회로 모델 함수 구현 (basic_circuit.cc)
 *
 * === 파일의 역할 ===
 * basic_circuit.h에 선언된 트랜지스터/회로 수준 물리 모델 함수들의 구현체이다.
 * MOSFET 게이트 커패시턴스(gate_C), 드레인 커패시턴스(drain_C_), 온 저항(tr_R_on),
 * Horowitz 타이밍 모델(horowitz), 서브임계/게이트 누설전류(cmos_Isub_leakage,
 * cmos_Ig_leakage), 단락전류(shortcircuit_simple) 등을 구현한다.
 * 이 함수들은 CACTI 내 모든 타이밍·전력·면적 계산의 물리적 기반을 제공한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch(GPU 전력 모델) → CACTI(캐시 에너지/면적) → basic_circuit.cc(물리 계층 구현).
 * 호출 체인: decoder.cc / crossbar.cc / mat.cc / wire.cc 등 상위 회로 모델
 *   → [basic_circuit 함수들] → g_tp(TechnologyParameter) 전역 객체.
 * 실행 컨텍스트: 호스트 유저스페이스 — 시뮬레이션 전 캐시 파라미터 초기화 단계.
 *
 * === 타 모듈과의 연결 ===
 * 의존: parameter.h의 g_tp (TechnologyParameter 전역 인스턴스) — 공정별 C_g_ideal,
 *   C_overlap, C_fringe, R_nch_on, I_off_n, I_g_on_n, l_phy, HPOWERRAIL 등 포함.
 *   Cpolywire 상수 (const.h).
 * 이 파일에 의존: decoder.cc, crossbar.cc, wire.cc, mat.cc, component.cc 등.
 *
 * === 주요 함수/구조체 요약 ===
 * _log2()             : 정수 이진 로그 (비트 이동 방식)
 * gate_C()            : 게이트 커패시턴스 — (C_g_ideal+C_overlap+3*C_fringe)*W + l_phy*Cpolywire
 * drain_C_()          : 드레인 커패시턴스 — 접합면적/측벽/프린지+오버랩/폴딩 금속 합산
 * tr_R_on()           : 온 저항 — stack * R_{n/p}ch_on / width
 * horowitz()          : Horowitz 1984 RC 타이밍 모델
 * cmos_Isub_leakage() : 게이트 유형별 평균 서브임계 누설전류 (이항계수 + 스택 효과 보정)
 * cmos_Ig_leakage()   : 게이트 유형별 평균 게이트 터널링 누설전류
 */


#include "basic_circuit.h" // [한국어] 이 파일이 구현하는 함수들의 선언
#include "parameter.h"     // [한국어] g_tp (TechnologyParameter) 전역 객체 — 공정별 물리 파라미터
#include <iostream>        // [한국어] std::cerr — 오류 출력용
#include <assert.h>        // [한국어] assert() — 입력 유효성 검사
#include <cmath>           // [한국어] log(), sqrt(), pow(), ceil() 등 수학 함수

/*
 * [한국어]
 * _log2 - 부호 없는 정수에 대한 이진 로그 계산 (비트 이동 방식)
 * @num    : 로그를 계산할 양의 정수 (0이면 오류 종료)
 * @return : floor(log₂(num)) — 비트 이동 횟수
 * num을 1보다 클 때까지 오른쪽으로 1비트씩 이동하며 횟수를 셈.
 * 디코더 비트 수(_log2(num_decoded_signals)) 계산에 주로 사용된다.
 * 호출 체인: Decoder 생성자, PredecBlk 생성자 등 → [_log2]
 */
uint32_t _log2(uint64_t num)
{
  uint32_t log2 = 0; // [한국어] 비트 이동 횟수 누적 (= floor(log₂(num)))

  if (num == 0)
  {
    std::cerr << "log0?" << std::endl; // [한국어] 0의 로그는 정의 불가 — 오류 메시지 출력
    exit(1);                           // [한국어] 비정상 종료 — 로그 인수가 0인 버그 방지
  }

  while (num > 1)
  {
    num = (num >> 1); // [한국어] 오른쪽으로 1비트 이동 = 2로 나눔
    log2++;           // [한국어] 이동 횟수 = floor(log₂(원래 num))
  }

  return log2; // [한국어] 최종 이진 로그값 반환
}


/*
 * [한국어]
 * is_pow2 - 정수가 2의 거듭제곱인지 판별
 * @val    : 검사할 정수 (0 이하이면 false 반환)
 * @return : val이 2^k 형태이면 true, 아니면 false
 * _log2(val) != _log2(val-1) 조건으로 판별 — val이 2의 거듭제곱이면
 * val-1은 하위 비트만 1인 수이므로 log₂가 달라진다.
 * 호출 체인: 캐시 구성 파라미터 유효성 검사 → [is_pow2]
 */
bool is_pow2(int64_t val)
{
  if (val <= 0)                           // [한국어] 음수나 0은 2의 거듭제곱이 아님
  {
    return false;
  }
  else if (val == 1)
  {
    return true;                          // [한국어] 2^0 = 1
  }
  else
  {
    return (_log2(val) != _log2(val-1)); // [한국어] 비트 분리점이 달라야 2의 거듭제곱
  }
}


/*
 * [한국어]
 * powers - 정수 거듭제곱 계산 (base^n)
 * @base   : 밑 (base)
 * @n      : 지수 (n >= 0)
 * @return : base^n (정수)
 * 반복 곱셈으로 계산 — 주로 소규모 n에서 호출되므로 성능 문제 없음.
 * 호출 체인: 프리디코더 레벨 수 계산 등 → [powers]
 */
int powers (int base, int n)
{
  int i, p;

  p = 1;                    // [한국어] 결과 초기값 1 (base^0)
  for (i = 1; i <= n; ++i)
    p *= base;              // [한국어] base를 n번 곱함
  return p;
}

/*----------------------------------------------------------------------*/

/*
 * [한국어]
 * logtwo - 부동소수점 이진 로그 계산 (log₂(x))
 * @x      : 입력값 (양수여야 함, assert로 검사)
 * @return : log₂(x) = log(x) / log(2.0) (double)
 * 표준 log()를 로그 변환 공식으로 이진 로그로 변환.
 * 호출 체인: 면적/디코딩 비트 수 계산 등 → [logtwo]
 */
double logtwo (double x)
{
  assert(x > 0);                                    // [한국어] 입력이 양수임을 보장
  return ((double) (log (x) / log (2.0)));          // [한국어] log 변환 공식: log₂(x) = ln(x)/ln(2)
}

/*----------------------------------------------------------------------*/


/*
 * [한국어]
 * gate_C - MOSFET 게이트 커패시턴스(farads) 계산
 * @width       : 게이트 폭 (um)
 * @wirelength  : 폴리 와이어 길이 (lambda 단위, 현재 커패시턴스 계산에 사용 안 됨)
 * @_is_dram    : true이고 _is_cell이면 DRAM 셀 접근 트랜지스터 (g_tp.dram_acc)
 * @_is_cell    : true이고 !_is_dram이면 SRAM 셀 트랜지스터 (g_tp.sram_cell)
 * @_is_wl_tr   : true이고 _is_dram이면 DRAM 워드라인 트랜지스터 (g_tp.dram_wl)
 * @return      : 게이트 커패시턴스 (F) = (C_g_ideal + C_overlap + 3*C_fringe)*width + l_phy*Cpolywire
 * 트랜지스터 유형에 따라 g_tp의 적절한 DeviceType을 선택한 뒤 동일 수식을 적용.
 * 게이트 산화막 커패시턴스(C_g_ideal), 오버랩 커패시턴스(C_overlap), 프린지 커패시턴스
 * (3*C_fringe)의 합에 폭을 곱하고, 폴리 배선 기여분(l_phy*Cpolywire)을 더한다.
 * 호출 체인: decoder/predecoder/crossbar compute_widths/compute_delays → [gate_C]
 */
double gate_C(
    double width,
    double wirelength,
    bool   _is_dram,
    bool   _is_cell,
    bool   _is_wl_tr)
{
  const TechnologyParameter::DeviceType * dt; // [한국어] 트랜지스터 유형별 물리 파라미터 포인터

  if (_is_dram && _is_cell)
  {
    dt = &g_tp.dram_acc;   //DRAM cell access transistor
    // [한국어] DRAM 셀 접근 트랜지스터 파라미터 선택 — DRAM 셀의 특수한 도핑/산화막 특성 반영
  }
  else if (_is_dram && _is_wl_tr)
  {
    dt = &g_tp.dram_wl;    //DRAM wordline transistor
    // [한국어] DRAM 워드라인 트랜지스터 — 높은 Vpp를 구동하기 위해 별도 파라미터 사용
  }
  else if (!_is_dram && _is_cell)
  {
    dt = &g_tp.sram_cell;  // SRAM cell access transistor
    // [한국어] SRAM 셀 접근 트랜지스터 — 6T SRAM 셀의 패스 트랜지스터 파라미터
  }
  else
  {
    dt = &g_tp.peri_global; // [한국어] 주변 회로(periphery) 트랜지스터 — 기본 공정 파라미터
  }

  // [한국어] 게이트 커패시턴스 수식:
  //   (C_g_ideal: 이상적 게이트 산화막 용량) + (C_overlap: 게이트-소스/드레인 오버랩)
  //   + 3*(C_fringe: 프린지 필드 효과, 팩터 3은 경험적 계수) — 모두 단위 폭당 값
  //   전체에 트랜지스터 폭(width)을 곱한 후, l_phy*Cpolywire(폴리 배선 직렬 기여)를 더함
  return (dt->C_g_ideal + dt->C_overlap + 3*dt->C_fringe)*width + dt->l_phy*Cpolywire;
}


/*
 * [한국어]
 * gate_C_pass - 패스 트랜지스터 게이트 커패시턴스 계산 (gate_C와 동일 구현)
 * @width / @wirelength / @_is_dram / @_is_cell / @_is_wl_tr : gate_C()와 동일
 * @return : 게이트 커패시턴스 (F) — gate_C()와 완전히 동일한 수식 사용
 * v5.0 이후 gate_C()와 동일하게 처리된다. 과거에는 패스 트랜지스터의 채널 전하
 * 공유(charge sharing) 효과를 별도로 처리하려 했으나 현재는 통합됨.
 * 호출 체인: 일부 레거시 경로 → [gate_C_pass]
 */
// returns gate capacitance in Farads
// actually this function is the same as gate_C() now
double gate_C_pass(
    double width,       // gate width in um (length is Lphy_periph_global)
    double wirelength,  // poly wire length going to gate in lambda
    bool   _is_dram,
    bool   _is_cell,
    bool   _is_wl_tr)
{
  // v5.0
  const TechnologyParameter::DeviceType * dt;

  if ((_is_dram) && (_is_cell))
  {
    dt = &g_tp.dram_acc;   //DRAM cell access transistor
  }
  else if ((_is_dram) && (_is_wl_tr))
  {
    dt = &g_tp.dram_wl;    //DRAM wordline transistor
  }
  else if ((!_is_dram) && _is_cell)
  {
    dt = &g_tp.sram_cell;  // SRAM cell access transistor
  }
  else
  {
    dt = &g_tp.peri_global;
  }

  return (dt->C_g_ideal + dt->C_overlap + 3*dt->C_fringe)*width + dt->l_phy*Cpolywire;
}



/*
 * [한국어]
 * drain_C_ - MOSFET 드레인 커패시턴스(farads) 계산 (트랜지스터 폴딩 포함)
 * @width        : 게이트 폭 (um) — 트랜지스터 총 폭
 * @nchannel     : 1=NMOS, 0=PMOS — 채널 방향 결정
 * @stack        : 직렬 스택 트랜지스터 수 (NAND2: stack=2, NAND3: stack=3)
 * @next_arg_thresh_folding_width_or_height_cell :
 *                0이면 fold_dimension을 폴딩 임계폭으로 해석,
 *                1이면 셀 높이로 해석하여 NMOS/PMOS 비율에서 폴딩폭 계산
 * @fold_dimension : 폴딩 임계폭(um) 또는 셀 높이(um)
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형 선택
 * @return       : 총 드레인 커패시턴스 (F) = C_area + C_sidewall + C_fringe_overlap + C_metal_fold
 *
 * 폴딩(folding): 트랜지스터 폭이 임계치보다 크면 여러 개로 접어서 레이아웃.
 * 폴딩 시 드레인이 공유되어 커패시턴스가 달라지므로 이를 상세히 모델링.
 * 각 기여 성분:
 *   - drain_C_area     : 접합 면적 × C_junc (면적 커패시턴스)
 *   - drain_C_sidewall : 접합 측벽 둘레 × C_junc_sidewall (측벽 커패시턴스)
 *   - drain_C_wrt_gate : (2*C_fringe + 2*C_overlap) × 게이트 대향 드레인 높이 합계
 *   - drain_C_metal    : 폴딩 연결 금속 배선의 커패시턴스 (C_per_um × 드레인 폭)
 * 호출 체인: decoder/predecoder/crossbar compute_delays → [drain_C_]
 */
double drain_C_(
    double width,
    int nchannel,
    int stack,
    int next_arg_thresh_folding_width_or_height_cell,
    double fold_dimension,
    bool _is_dram,
    bool _is_cell,
    bool _is_wl_tr)
{
  double w_folded_tr=0; // [한국어] 폴딩 후 하나의 핑거(finger) 폭 (um)
  const  TechnologyParameter::DeviceType * dt; // [한국어] 트랜지스터 유형별 파라미터

  if ((_is_dram) && (_is_cell))
  {
    dt = &g_tp.dram_acc;   // DRAM cell access transistor
    // [한국어] DRAM 셀 접근 트랜지스터 파라미터 선택
  }
  else if ((_is_dram) && (_is_wl_tr))
  {
    dt = &g_tp.dram_wl;    // DRAM wordline transistor
    // [한국어] DRAM 워드라인 트랜지스터 파라미터 선택
  }
  else if ((!_is_dram) && _is_cell)
  {
    dt = &g_tp.sram_cell;  // SRAM cell access transistor
    // [한국어] SRAM 셀 접근 트랜지스터 파라미터 선택
  }
  else
  {
    dt = &g_tp.peri_global; // [한국어] 주변 회로 트랜지스터 기본 파라미터
  }

  double c_junc_area = dt->C_junc;         // [한국어] 단위 접합 면적 커패시턴스 (F/um²)
  double c_junc_sidewall = dt->C_junc_sidewall; // [한국어] 단위 접합 측벽 커패시턴스 (F/um)
  double c_fringe    = 2*dt->C_fringe;     // [한국어] 드레인 양쪽 프린지 커패시턴스 (소스쪽 포함, ×2)
  double c_overlap   = 2*dt->C_overlap;    // [한국어] 드레인 양쪽 게이트-드레인 오버랩 커패시턴스 (×2)
  double drain_C_metal_connecting_folded_tr = 0; // [한국어] 폴딩 핑거 연결 금속 커패시턴스 (초기 0)

  // determine the width of the transistor after folding (if it is getting folded)
  if (next_arg_thresh_folding_width_or_height_cell == 0)
  { // interpret fold_dimension as the the folding width threshold
    // i.e. the value of transistor width above which the transistor gets folded
    w_folded_tr = fold_dimension; // [한국어] 폴딩 임계폭을 직접 사용 — 단일 핑거 최대 폭
  }
  else
  { // interpret fold_dimension as the height of the cell that this transistor is part of.
    double h_tr_region  = fold_dimension - 2 * g_tp.HPOWERRAIL;
    // [한국어] 셀 높이에서 전원 레일(HPOWERRAIL) 2개 영역을 빼서 트랜지스터 배치 가능 영역 계산
    // TODO : w_folded_tr must come from Component::compute_gate_area()
    double ratio_p_to_n = 2.0 / (2.0 + 1.0);
    // [한국어] PMOS:NMOS 영역 비율 = 2:1 (경험적 값) — p_to_n_eff_ratio 기반
    if (nchannel)
    {
      w_folded_tr = (1 - ratio_p_to_n) * (h_tr_region - g_tp.MIN_GAP_BET_P_AND_N_DIFFS);
      // [한국어] NMOS 핑거 폭: NMOS 비율(1-ratio_p_to_n) × (트랜지스터 영역 - P/N 간격)
    }
    else
    {
      w_folded_tr = ratio_p_to_n * (h_tr_region - g_tp.MIN_GAP_BET_P_AND_N_DIFFS);
      // [한국어] PMOS 핑거 폭: PMOS 비율(ratio_p_to_n) × (트랜지스터 영역 - P/N 간격)
    }
  }
  int num_folded_tr = (int) (ceil(width / w_folded_tr));
  // [한국어] 필요한 핑거 수 = ceil(총 폭 / 단일 핑거 최대 폭)

  if (num_folded_tr < 2)
  {
    w_folded_tr = width; // [한국어] 폴딩이 불필요하면 핑거 폭 = 전체 트랜지스터 폭
  }

  double total_drain_w = (g_tp.w_poly_contact + 2 * g_tp.spacing_poly_to_contact) +  // only for drain
                         (stack - 1) * g_tp.spacing_poly_to_poly;
  // [한국어] 드레인 총 폭 = 컨택 폭 + 2×컨택-폴리 간격 (드레인 단독 면적)
  //         + (stack-1) × 폴리-폴리 간격 (직렬 스택에서 내부 드레인 공유 영역)
  double drain_h_for_sidewall = w_folded_tr;
  // [한국어] 측벽 계산을 위한 드레인 높이 = 단일 핑거 폭 (폴딩 짝수면 0이 됨)
  double total_drain_height_for_cap_wrt_gate = w_folded_tr + 2 * w_folded_tr * (stack - 1);
  // [한국어] 게이트 대향 드레인 높이 합계: 1개 외부 + 2*(stack-1)개 내부 공유 드레인 × 핑거 폭
  if (num_folded_tr > 1)
  {
    total_drain_w += (num_folded_tr - 2) * (g_tp.w_poly_contact + 2 * g_tp.spacing_poly_to_contact) +
                     (num_folded_tr - 1) * ((stack - 1) * g_tp.spacing_poly_to_poly);
    // [한국어] 핑거가 2개 이상이면 공유 드레인 영역 추가: (num_folded-2)개 내부 드레인 컨택 + 내부 스택 간격

    if (num_folded_tr%2 == 0)
    {
      drain_h_for_sidewall = 0;
      // [한국어] 핑거가 짝수이면 양 끝이 소스/게이트로 닫혀 외부 드레인 측벽이 없음
    }
    total_drain_height_for_cap_wrt_gate *= num_folded_tr;
    // [한국어] 핑거 수만큼 게이트 대향 드레인 높이 스케일
    drain_C_metal_connecting_folded_tr   = g_tp.wire_local.C_per_um * total_drain_w;
    // [한국어] 폴딩된 핑거들을 연결하는 로컬 금속 배선의 커패시턴스 = C_per_um × 드레인 총 폭
  }

  double drain_C_area     = c_junc_area * total_drain_w * w_folded_tr;
  // [한국어] 면적 커패시턴스 = 단위 면적 용량 × 드레인 폭 × 드레인 높이(핑거 폭)
  double drain_C_sidewall = c_junc_sidewall * (drain_h_for_sidewall + 2 * total_drain_w);
  // [한국어] 측벽 커패시턴스 = 단위 측벽 용량 × 드레인 둘레 (측면 높이 + 2 × 드레인 폭)
  double drain_C_wrt_gate = (c_fringe + c_overlap) * total_drain_height_for_cap_wrt_gate;
  // [한국어] 게이트 대향 커패시턴스 = (프린지 + 오버랩) × 총 드레인-게이트 대향 높이

  return (drain_C_area + drain_C_sidewall + drain_C_wrt_gate + drain_C_metal_connecting_folded_tr);
  // [한국어] 4가지 기여 성분의 합계를 반환
}


/*
 * [한국어]
 * tr_R_on - MOSFET 트랜지스터 온(on) 상태 저항 계산
 * @width     : 게이트 폭 (um) — 폭이 클수록 저항 감소 (저항 ∝ 1/width)
 * @nchannel  : 1=NMOS(R_nch_on 사용), 0=PMOS(R_pch_on 사용)
 * @stack     : 직렬 스택 수 — 직렬 저항이 stack배로 증가
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형
 * @return    : 온 저항 = stack * R_{n/p}ch_on / width (ohm)
 * R_nch_on/R_pch_on은 단위 폭당 NMOS/PMOS 온 저항 (ohm*um).
 * Horowitz 타이밍 모델의 tf = R_on * C_load 계산에 핵심적으로 사용.
 * 호출 체인: decoder/predecoder/crossbar compute_delays → [tr_R_on]
 */
double tr_R_on(
    double width,
    int nchannel,
    int stack,
    bool _is_dram,
    bool _is_cell,
    bool _is_wl_tr)
{
  const TechnologyParameter::DeviceType * dt; // [한국어] 트랜지스터 유형별 파라미터

  if ((_is_dram) && (_is_cell))
  {
    dt = &g_tp.dram_acc;   //DRAM cell access transistor
  }
  else if ((_is_dram) && (_is_wl_tr))
  {
    dt = &g_tp.dram_wl;    //DRAM wordline transistor
  }
  else if ((!_is_dram) && _is_cell)
  {
    dt = &g_tp.sram_cell;  // SRAM cell access transistor
  }
  else
  {
    dt = &g_tp.peri_global; // [한국어] 주변 회로 기본 파라미터
  }

  double restrans = (nchannel) ? dt->R_nch_on : dt->R_pch_on;
  // [한국어] NMOS이면 R_nch_on, PMOS이면 R_pch_on 선택 (단위: ohm*um)
  return (stack * restrans / width);
  // [한국어] 직렬 스택 수 × 단위 온 저항 / 폭 = 실제 온 저항 (ohm)
}


/* This routine operates in reverse: given a resistance, it finds
 * the transistor width that would have this R.  It is used in the
 * data wordline to estimate the wordline driver size. */

/*
 * [한국어]
 * R_to_w - 목표 저항에서 필요한 트랜지스터 폭 역산 (tr_R_on의 역함수)
 * @res      : 목표 저항값 (ohm) — 달성하고 싶은 온 저항
 * @nchannel : 1=NMOS, 0=PMOS
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형
 * @return   : 필요한 트랜지스터 폭 (um) = R_nch_on / res
 * 워드라인 드라이버의 타이밍 요구사항으로부터 드라이버 트랜지스터 크기를 결정하는 데 사용.
 * stack=1 가정 — 스택 효과는 고려하지 않음.
 * 호출 체인: 워드라인 드라이버 크기 추정 → [R_to_w]
 */
// returns width in um
double R_to_w(
    double res,
    int   nchannel,
    bool _is_dram,
    bool _is_cell,
    bool _is_wl_tr)
{
  const TechnologyParameter::DeviceType * dt;

  if ((_is_dram) && (_is_cell))
  {
    dt = &g_tp.dram_acc;   //DRAM cell access transistor
  }
  else if ((_is_dram) && (_is_wl_tr))
  {
    dt = &g_tp.dram_wl;    //DRAM wordline transistor
  }
  else if ((!_is_dram) && (_is_cell))
  {
    dt = &g_tp.sram_cell;  // SRAM cell access transistor
  }
  else
  {
    dt = &g_tp.peri_global;
  }

  double restrans = (nchannel) ? dt->R_nch_on : dt->R_pch_on;
  return (restrans / res);
}


/*
 * [한국어]
 * pmos_to_nmos_sz_ratio - PMOS/NMOS 전류 구동력 대칭화를 위한 크기 비율 반환
 * @_is_dram  : DRAM 여부
 * @_is_wl_tr : DRAM 워드라인 트랜지스터 여부
 * @return    : n_to_p_eff_curr_drv_ratio — PMOS 폭 = ratio × NMOS 폭으로 설계 시 대칭 구동력
 * 공정 노드마다 PMOS의 홀 이동도가 낮아 NMOS보다 폭을 더 크게 해야 동일 전류를 구동.
 * 이 비율은 논리 노력(logical_effort) 기반 게이트 크기 결정의 기준값.
 * 호출 체인: decoder/predecoder compute_widths → [pmos_to_nmos_sz_ratio]
 */
double pmos_to_nmos_sz_ratio(
    bool _is_dram,
    bool _is_wl_tr)
{
  double p_to_n_sizing_ratio; // [한국어] 반환할 P/N 크기 비율
  if ((_is_dram) && (_is_wl_tr))
  { //DRAM wordline transistor
    p_to_n_sizing_ratio = g_tp.dram_wl.n_to_p_eff_curr_drv_ratio;
    // [한국어] DRAM 워드라인 트랜지스터는 고전압(Vpp) 동작으로 별도 비율 사용
  }
  else
  { //DRAM or SRAM all other transistors
    p_to_n_sizing_ratio = g_tp.peri_global.n_to_p_eff_curr_drv_ratio;
    // [한국어] 주변 회로 공통 P/N 비율 — 공정 노드별로 보통 2~3 수준
  }
  return p_to_n_sizing_ratio;
}


/*
 * [한국어]
 * horowitz - Horowitz(1984) RC 타이밍 모델로 게이트 전파 지연 계산
 * @inputramptime : 입력 신호 상승(또는 하강) 시간 (s) — 이전 게이트 출력 슬루율
 * @tf            : 게이트 RC 시정수 = R_on × C_load (s)
 * @vs1           : 입력 임계전압 비율 (예: Vth/Vdd = 0.5)
 * @vs2           : 출력 임계전압 비율 (보통 vs1과 동일하게 설정)
 * @rise          : RISE(1)이면 출력 상승, FALL(0)이면 출력 하강 타이밍
 * @return        : 게이트 전파 지연 td (s)
 * 입력 경사(ramp)가 있을 때와 없을 때를 구분하여 계산:
 *   - inputramptime==0이면: td = tf * ln(1/vs1) (단순 RC 지수감소)
 *   - 상승(RISE): a=inputramptime/tf, b=0.5
 *     td = tf*sqrt(ln(vs1)² + 2ab(1-vs1)) + tf*(ln(vs1) - ln(vs2))
 *   - 하강(FALL): b=0.4 (비대칭 계수)
 * 참고: "Timing Models for MOS Circuits", Mark Horowitz, 1984
 * 호출 체인: decoder/predecoder/crossbar/mat 지연 누적 → [horowitz]
 */
// "Timing Models for MOS Circuits" by Mark Horowitz, 1984
double horowitz(
    double inputramptime, // input rise time
    double tf,            // time constant of gate
    double vs1,           // threshold voltage
    double vs2,           // threshold voltage
    int    rise)          // whether input rises or fall
{
  if (inputramptime == 0 && vs1 == vs2)
  {
    // [한국어] 입력 경사가 없고 임계전압이 같을 때 단순 RC 지연 (이상적 스텝 입력 가정)
    return tf * (vs1 < 1 ? -log(vs1) : log(vs1));
    // [한국어] vs1<1이면 -ln(vs1) = ln(1/vs1) (일반적 경우), vs1>=1이면 ln(vs1)
  }
  double a, b, td; // [한국어] a=정규화된 입력 경사, b=비대칭 계수, td=전파 지연

  a = inputramptime / tf; // [한국어] 입력 경사를 시정수로 정규화
  if (rise == RISE)
  {
    b = 0.5; // [한국어] 상승 전이(RISE) 비대칭 계수 — 경험적 값
    td = tf * sqrt(log(vs1)*log(vs1) + 2*a*b*(1.0 - vs1)) + tf*(log(vs1) - log(vs2));
    // [한국어] Horowitz RISE 공식: 입력 경사와 RC 지연의 조합 (루트 내: 두 효과의 이차합)
  }
  else
  {
    b = 0.4; // [한국어] 하강 전이(FALL) 비대칭 계수 — PMOS 특성 차이를 반영
    td = tf * sqrt(log(1.0 - vs1)*log(1.0 - vs1) + 2*a*b*(vs1)) + tf*(log(1.0 - vs1) - log(1.0 - vs2));
    // [한국어] Horowitz FALL 공식: 1-vs1을 기준으로 대칭적 형태
  }
  return (td);
}

/*
 * [한국어]
 * cmos_Ileak - CMOS 게이트 총 서브임계 누설전류 단순 합산
 * @nWidth / @pWidth : NMOS/PMOS 폭 (um)
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형 선택
 * @return : nWidth*I_off_n + pWidth*I_off_p (A) — 상태 평균 없이 단순 합산
 * cmos_Isub_leakage()와 달리 입력 상태 확률 계산 없이 최대 누설을 추정한다.
 * 일부 모듈에서 빠른 상한 추정 또는 단순 검산용으로 사용.
 * 호출 체인: 일부 면적/전력 추정 코드 → [cmos_Ileak]
 */
double cmos_Ileak(
    double nWidth,
    double pWidth,
    bool _is_dram,
    bool _is_cell,
    bool _is_wl_tr)
{
  TechnologyParameter::DeviceType * dt; // [한국어] 트랜지스터 유형별 파라미터

  if ((!_is_dram)&&(_is_cell))
  { //SRAM cell access transistor
    dt = &(g_tp.sram_cell); // [한국어] SRAM 셀 접근 트랜지스터
  }
  else if ((_is_dram)&&(_is_wl_tr))
  { //DRAM wordline transistor
    dt = &(g_tp.dram_wl); // [한국어] DRAM 워드라인 트랜지스터
  }
  else
  { //DRAM or SRAM all other transistors
    dt = &(g_tp.peri_global); // [한국어] 주변 회로 기본 파라미터
  }
  return nWidth*dt->I_off_n + pWidth*dt->I_off_p;
  // [한국어] NMOS 누설(I_off_n×폭) + PMOS 누설(I_off_p×폭) — 단순 합산 반환
}


/*
 * [한국어]
 * simplified_nmos_leakage - NMOS 서브임계 누설전류 단위 기준값 계산
 * @nwidth    : NMOS 게이트 폭 (um)
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형
 * @return    : nwidth * I_off_n (A) — 단위 폭당 오프(off) 전류에 폭 곱
 * cmos_Isub_leakage() 내부에서 nmos_leak 기준값으로 사용.
 * I_off_n은 공정 파라미터에서 온도/공정에 따라 달라지는 단위폭 오프 전류 (A/um).
 * 호출 체인: cmos_Isub_leakage → [simplified_nmos_leakage]
 */
double simplified_nmos_leakage(
    double nwidth,
    bool _is_dram,
    bool _is_cell,
    bool _is_wl_tr)
{
  TechnologyParameter::DeviceType * dt; // [한국어] 트랜지스터 유형별 파라미터

  if ((!_is_dram)&&(_is_cell))
  { //SRAM cell access transistor
    dt = &(g_tp.sram_cell);
  }
  else if ((_is_dram)&&(_is_wl_tr))
  { //DRAM wordline transistor
    dt = &(g_tp.dram_wl);
  }
  else
  { //DRAM or SRAM all other transistors
    dt = &(g_tp.peri_global);
  }
  return nwidth * dt->I_off_n; // [한국어] NMOS 오프 전류 = 폭 × 단위폭당 오프 전류
}

/*
 * [한국어]
 * factorial - 부분 팩토리얼 계산: m × (m+1) × ... × n
 * @n : 상한 정수
 * @m : 하한 정수 (기본값 1 — 전체 n! 계산)
 * @return : m * (m+1) * ... * n
 * combination() 계산에서 이항계수를 효율적으로 구하기 위해 사용.
 * 호출 체인: combination → [factorial]
 */
int factorial(int n, int m)
{
	int fa = m, i;    // [한국어] fa = 현재까지의 누적 곱 (초기값 m)
	for (i=m+1; i<=n; i++)
		fa *=i;       // [한국어] m+1부터 n까지 순차적으로 곱
	return fa;
}

/*
 * [한국어]
 * combination - 이항 계수 C(n, m) = n! / (m! * (n-m)!) 계산
 * @n : 전체 원소 수
 * @m : 선택 원소 수
 * @return : C(n, m) = factorial(n, m+1) / factorial(n-m)
 * cmos_Isub_leakage()에서 각 입력 상태에서 OFF 트랜지스터 수의 조합 수를 계산.
 * 호출 체인: cmos_Isub_leakage/cmos_Ig_leakage → [combination]
 */
int combination(int n, int m)
{
  int ret;
  ret = factorial(n, m+1) / factorial(n - m);
  // [한국어] C(n,m) = n!/(m!*(n-m)!) = (m+1 * ... * n) / (n-m)!
  return ret;
}

/*
 * [한국어]
 * simplified_pmos_leakage - PMOS 서브임계 누설전류 단위 기준값 계산
 * @pwidth    : PMOS 게이트 폭 (um)
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형
 * @return    : pwidth * I_off_p (A)
 * cmos_Isub_leakage() 내부에서 pmos_leak 기준값으로 사용.
 * 호출 체인: cmos_Isub_leakage → [simplified_pmos_leakage]
 */
double simplified_pmos_leakage(
    double pwidth,
    bool _is_dram,
    bool _is_cell,
    bool _is_wl_tr)
{
  TechnologyParameter::DeviceType * dt; // [한국어] 트랜지스터 유형별 파라미터

  if ((!_is_dram)&&(_is_cell))
  { //SRAM cell access transistor
    dt = &(g_tp.sram_cell);
  }
  else if ((_is_dram)&&(_is_wl_tr))
  { //DRAM wordline transistor
    dt = &(g_tp.dram_wl);
  }
  else
  { //DRAM or SRAM all other transistors
    dt = &(g_tp.peri_global);
  }
  return pwidth * dt->I_off_p; // [한국어] PMOS 오프 전류 = 폭 × 단위폭당 오프 전류
}

/*
 * [한국어]
 * cmos_Ig_n - NMOS 게이트 터널링 누설전류 기준값 계산
 * @nWidth    : NMOS 게이트 폭 (um)
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형
 * @return    : nWidth * I_g_on_n (A)
 * I_g_on_n: ON 상태에서 게이트 산화막을 통한 터널링 전류 (A/um) — 극소 노드에서 크게 증가.
 * cmos_Ig_leakage()에서 nmos_leak 기준값으로 사용.
 * 호출 체인: cmos_Ig_leakage → [cmos_Ig_n]
 */
double cmos_Ig_n(
    double nWidth,
    bool _is_dram,
    bool _is_cell,
    bool _is_wl_tr)
{
  TechnologyParameter::DeviceType * dt; // [한국어] 트랜지스터 유형별 파라미터

  if ((!_is_dram)&&(_is_cell))
  { //SRAM cell access transistor
    dt = &(g_tp.sram_cell);
  }
  else if ((_is_dram)&&(_is_wl_tr))
  { //DRAM wordline transistor
    dt = &(g_tp.dram_wl);
  }
  else
  { //DRAM or SRAM all other transistors
    dt = &(g_tp.peri_global);
  }
  return nWidth*dt->I_g_on_n; // [한국어] NMOS 게이트 터널링 누설 = 폭 × 단위폭당 ON 터널링 전류
}

/*
 * [한국어]
 * cmos_Ig_p - PMOS 게이트 터널링 누설전류 기준값 계산
 * @pWidth    : PMOS 게이트 폭 (um)
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형
 * @return    : pWidth * I_g_on_p (A)
 * 호출 체인: cmos_Ig_leakage → [cmos_Ig_p]
 */
double cmos_Ig_p(
    double pWidth,
    bool _is_dram,
    bool _is_cell,
    bool _is_wl_tr)
{
  TechnologyParameter::DeviceType * dt; // [한국어] 트랜지스터 유형별 파라미터

  if ((!_is_dram)&&(_is_cell))
  { //SRAM cell access transistor
    dt = &(g_tp.sram_cell);
  }
  else if ((_is_dram)&&(_is_wl_tr))
  { //DRAM wordline transistor
    dt = &(g_tp.dram_wl);
  }
  else
  { //DRAM or SRAM all other transistors
    dt = &(g_tp.peri_global);
  }
  return pWidth*dt->I_g_on_p; // [한국어] PMOS 게이트 터널링 누설 = 폭 × 단위폭당 ON 터널링 전류
}

/*
 * [한국어]
 * cmos_Isub_leakage - CMOS 게이트 유형별 통계적 평균 서브임계 누설전류 계산
 * @nWidth / @pWidth : NMOS/PMOS 게이트 폭 (um)
 * @fanin     : 게이트 입력 수 (1 이상, assert로 검사)
 * @g_type    : 게이트 유형 (inv/nand/nor/tri/tg/nmos/pmos)
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형 선택
 * @topo      : 하프 네트워크 연결 방식 (series=직렬, parallel=병렬)
 * @return    : 2^fanin 가지 입력 상태에 대한 평균 서브임계 누설전류 (A)
 *
 * 알고리즘:
 *   1) 가능한 입력 상태 수: num_states = 2^fanin
 *   2) 각 게이트 유형/토폴로지에 따라 OFF 트랜지스터 수(num_off_tx)별 누설전류를 합산:
 *      Isub += nmos_leak * UNI_LEAK_STACK_FACTOR^(num_off_tx-1) * C(fanin, num_off_tx)
 *      (직렬 스택: OFF tx가 늘수록 0.43배씩 누설 감소; 이항계수로 해당 조합 수 반영)
 *   3) num_states로 나누어 상태 평균 계산
 *   inv: (nmos_leak + pmos_leak) / 2 (항상 풀업/풀다운 중 하나가 ON)
 *   nand: 풀업 PMOS 병렬(항상 부분 OFF 가능) + 풀다운 NMOS 직렬 스택 분리 계산
 *   nor: 풀업 PMOS 직렬 + 풀다운 NMOS 병렬 분리 계산
 *   tri: 활성/비활성 상태 각각 계산 후 평균
 *   tg: 전송 게이트 — inv와 동일하게 평균 처리
 * 호출 체인: decoder/predecoder/crossbar compute_area → [cmos_Isub_leakage]
 */
double cmos_Isub_leakage(
    double nWidth,
    double pWidth,
    int    fanin,
    enum Gate_type g_type,
    bool _is_dram,
    bool _is_cell,
    bool _is_wl_tr,
    enum Half_net_topology topo)
{
	assert (fanin>=1); // [한국어] fanin은 반드시 1 이상이어야 함
	double nmos_leak = simplified_nmos_leakage(nWidth, _is_dram, _is_cell, _is_wl_tr);
	// [한국어] NMOS 기준 누설전류 = nWidth * I_off_n (fanin=1 단일 트랜지스터 기준)
	double pmos_leak = simplified_pmos_leakage(pWidth, _is_dram, _is_cell, _is_wl_tr);
	// [한국어] PMOS 기준 누설전류 = pWidth * I_off_p
    double Isub=0;       // [한국어] 누적 서브임계 누설전류 합산 변수
    int    num_states;   // [한국어] 총 입력 상태 수 = 2^fanin
    int    num_off_tx;   // [한국어] OFF 트랜지스터 수 (루프 변수)

    num_states = int(pow(2.0, fanin)); // [한국어] 2^fanin: 가능한 모든 입력 조합의 수

    switch (g_type)
    {
    case nmos:
    	if (fanin==1)
    	{
    		Isub = nmos_leak/num_states;
    		// [한국어] NMOS 1개: 입력이 1(ON)이면 누설 없음, 0(OFF)이면 nmos_leak → 평균 nmos_leak/2
    	}
    	else
    	{
    		if (topo==parallel)
    		{
    			Isub=nmos_leak*fanin/num_states; //only when all tx are off, leakage power is non-zero. The possibility of this state is 1/num_states
    			// [한국어] 병렬 NMOS: 모두 OFF인 상태(1가지)에서만 fanin배 누설 → 확률 1/num_states 반영
    		}
    		else
    		{
    			for (num_off_tx=1; num_off_tx<=fanin; num_off_tx++) //when num_off_tx ==0 there is no leakage power
    			{
    				// [한국어] 직렬 스택: num_off_tx개 OFF일 때 스택 효과로 0.43^(num_off_tx-1) 감소
    				//         C(fanin, num_off_tx)가지 조합에서 해당 누설이 발생
    				//Isub += nmos_leak*pow(UNI_LEAK_STACK_FACTOR,(num_off_tx-1))*(factorial(fanin)/(factorial(fanin, num_off_tx)*factorial(num_off_tx)));
    				Isub += nmos_leak*pow(UNI_LEAK_STACK_FACTOR,(num_off_tx-1))*combination(fanin, num_off_tx);
    			}
    			Isub /=num_states; // [한국어] 총 상태 수로 나누어 통계적 평균 계산
    		}

    	}
    	break;
    case pmos:
    	if (fanin==1)
    	{
    		Isub = pmos_leak/num_states;
    		// [한국어] PMOS 1개: 입력이 0(ON)이면 누설 없음, 1(OFF)이면 pmos_leak → 평균
    	}
    	else
    	{
    		if (topo==parallel)
    		{
    			Isub=pmos_leak*fanin/num_states; //only when all tx are off, leakage power is non-zero. The possibility of this state is 1/num_states
    			// [한국어] 병렬 PMOS: 모두 OFF(입력 모두 1)인 경우에만 누설 발생
    		}
    		else
    		{
    			for (num_off_tx=1; num_off_tx<=fanin; num_off_tx++) //when num_off_tx ==0 there is no leakage power
    			{
    				//Isub += pmos_leak*pow(UNI_LEAK_STACK_FACTOR,(num_off_tx-1))*(factorial(fanin)/(factorial(fanin, num_off_tx)*factorial(num_off_tx)));
    				Isub += pmos_leak*pow(UNI_LEAK_STACK_FACTOR,(num_off_tx-1))*combination(fanin, num_off_tx);
    				// [한국어] 직렬 PMOS 스택: NMOS와 동일 방법으로 계산
    			}
    			Isub /=num_states;
    		}

    	}
    	break;
    case inv:
    	Isub = (nmos_leak + pmos_leak)/2;
    	// [한국어] 인버터: 입력 0→PMOS ON, NMOS OFF / 입력 1→NMOS ON, PMOS OFF
    	//         항상 한 쪽만 누설 → 두 경우의 평균
    	break;
    case nand:
    	Isub += fanin*pmos_leak;//the pullup network
    	// [한국어] NAND 풀업(PMOS 병렬): 입력이 0인 PMOS마다 누설 발생, 평균 fanin/2 개 → fanin × pmos_leak / 2
    	//         여기서 num_states로 나누기 전 fanin*pmos_leak은 모든 상태 누설의 합
    	for (num_off_tx=1; num_off_tx<=fanin; num_off_tx++) // the pulldown network
    	{
    		//Isub += nmos_leak*pow(UNI_LEAK_STACK_FACTOR,(num_off_tx-1))*(factorial(fanin)/(factorial(fanin, num_off_tx)*factorial(num_off_tx)));
            Isub += nmos_leak*pow(UNI_LEAK_STACK_FACTOR,(num_off_tx-1))*combination(fanin, num_off_tx);
            // [한국어] 풀다운 NMOS 직렬 스택: 스택 효과와 이항계수로 상태별 누설 합산
    	}
    	Isub /=num_states; // [한국어] 전체 상태 수로 나누어 평균 계산
    	break;
    case nor:
    	for (num_off_tx=1; num_off_tx<=fanin; num_off_tx++) // the pullup network
    	{
    		//Isub += pmos_leak*pow(UNI_LEAK_STACK_FACTOR,(num_off_tx-1))*(factorial(fanin)/(factorial(fanin, num_off_tx)*factorial(num_off_tx)));
    		Isub += pmos_leak*pow(UNI_LEAK_STACK_FACTOR,(num_off_tx-1))*combination(fanin, num_off_tx);
    		// [한국어] NOR 풀업 PMOS 직렬 스택: 스택 효과 반영
    	}
    	Isub += fanin*nmos_leak;//the pulldown network
    	// [한국어] NOR 풀다운 NMOS 병렬: 평균 fanin/2개가 ON → 나머지가 OFF일 때 누설
    	Isub /=num_states;
    	break;
    case tri:
    	Isub += (nmos_leak + pmos_leak)/2;//enabled
    	// [한국어] 트라이스테이트 활성 상태: 인버터처럼 평균 누설
    	Isub += nmos_leak*UNI_LEAK_STACK_FACTOR; //disabled upper bound of leakage power
    	// [한국어] 트라이스테이트 비활성 상태: 2개 직렬 스택에서 스택 효과 적용 (상한 추정)
    	Isub /=2; // [한국어] 활성/비활성 상태 50:50 평균
    	break;
    case tg:
    	Isub = (nmos_leak + pmos_leak)/2;
    	// [한국어] 전송 게이트: ON시 두 채널 모두 전도 — 인버터와 동일하게 평균 누설
    	break;
    default:
    	assert(0); // [한국어] 알 수 없는 게이트 유형 — 프로그래밍 오류
    	break;
	  }

    return Isub; // [한국어] 통계적 평균 서브임계 누설전류 (A) 반환
}


/*
 * [한국어]
 * cmos_Ig_leakage - CMOS 게이트 유형별 통계적 평균 게이트 터널링 누설전류 계산
 * @nWidth / @pWidth : NMOS/PMOS 게이트 폭 (um)
 * @fanin     : 게이트 입력 수 (1 이상)
 * @g_type    : 게이트 유형 (inv/nand/nor/tri/tg/nmos/pmos)
 * @_is_dram / @_is_cell / @_is_wl_tr : 트랜지스터 유형
 * @topo      : 하프 네트워크 연결 방식 (series/parallel)
 * @return    : 평균 게이트 터널링 누설전류 (A)
 * ON 상태 트랜지스터에서 발생하는 터널링을 2^fanin 상태에 대해 평균.
 * cmos_Isub_leakage()와 대칭 구조이나 ON tx 수(num_on_tx)를 기준으로 계산 —
 * 터널링은 OFF 상태가 아닌 ON 상태 트랜지스터에서 발생하기 때문.
 * 호출 체인: decoder/predecoder/crossbar compute_area → [cmos_Ig_leakage]
 */
double cmos_Ig_leakage(
    double nWidth,
    double pWidth,
    int    fanin,
    enum Gate_type g_type,
    bool _is_dram,
    bool _is_cell,
    bool _is_wl_tr,
    enum Half_net_topology topo)
{
	assert (fanin>=1); // [한국어] fanin은 1 이상이어야 함
		double nmos_leak = cmos_Ig_n(nWidth, _is_dram, _is_cell, _is_wl_tr);
		// [한국어] NMOS ON 상태 게이트 터널링 기준값 = nWidth * I_g_on_n
		double pmos_leak = cmos_Ig_p(pWidth, _is_dram, _is_cell, _is_wl_tr);
		// [한국어] PMOS ON 상태 게이트 터널링 기준값 = pWidth * I_g_on_p
	    double Ig_on=0;    // [한국어] 누적 게이트 터널링 누설전류 합산 변수
	    int    num_states; // [한국어] 총 입력 상태 수 = 2^fanin
	    int    num_on_tx;  // [한국어] ON 상태 트랜지스터 수 (루프 변수)

	    num_states = int(pow(2.0, fanin)); // [한국어] 2^fanin: 가능한 모든 입력 조합 수

	    switch (g_type)
	    {
	    case nmos:
	    	if (fanin==1)
	    	{
	    		Ig_on = nmos_leak/num_states;
	    	}
	    	else
	    	{
	    		if (topo==parallel)
	    		{
	    	    	for (num_on_tx=1; num_on_tx<=fanin; num_on_tx++)
	    	        {
	    	    		Ig_on += nmos_leak*combination(fanin, num_on_tx)*num_on_tx;
	    	    	}
	    		}
	    		else
	    		{
	    			Ig_on += nmos_leak * fanin;//pull down network when all TXs are on.
	    		    //num_on_tx is the number of on tx
	    			for (num_on_tx=1; num_on_tx<fanin; num_on_tx++)//when num_on_tx=[1,n-1]
	    			{
	    				Ig_on += nmos_leak*combination(fanin, num_on_tx)*num_on_tx/2;//TODO: this is a approximation now, a precise computation will be very complicated.
	    			}
	    			Ig_on /=num_states;
	    		}
	    	}
	    	break;
	    case pmos:
	    	if (fanin==1)
	    	{
	    		Ig_on = pmos_leak/num_states;
	    	}
	    	else
	    	{
	    		if (topo==parallel)
    		    {
    	    	  for (num_on_tx=1; num_on_tx<=fanin; num_on_tx++)
    	          {
    	    		  Ig_on += pmos_leak*combination(fanin, num_on_tx)*num_on_tx;
    	    	  }
    		    }
    		    else
    		    {
    			  Ig_on += pmos_leak * fanin;//pull down network when all TXs are on.
    		      //num_on_tx is the number of on tx
    			  for (num_on_tx=1; num_on_tx<fanin; num_on_tx++)//when num_on_tx=[1,n-1]
    			  {
    				  Ig_on += pmos_leak*combination(fanin, num_on_tx)*num_on_tx/2;//TODO: this is a approximation now, a precise computation will be very complicated.
    			  }
	    		  Ig_on /=num_states;
	    	    }
	    	}
	    	break;

	    case inv:
	    	Ig_on = (nmos_leak + pmos_leak)/2;
	    	// [한국어] 인버터: 입력 0→PMOS ON(터널링), 1→NMOS ON(터널링) → 두 경우의 평균
	    	break;
	    case nand:
	    	//pull up network
	    	for (num_on_tx=1; num_on_tx<=fanin; num_on_tx++)//when num_on_tx=[1,n]
	        {
	    		Ig_on += pmos_leak*combination(fanin, num_on_tx)*num_on_tx;
	    	}

	    	//pull down network
	    	Ig_on += nmos_leak * fanin;//pull down network when all TXs are on.
	    	//num_on_tx is the number of on tx
	    	for (num_on_tx=1; num_on_tx<fanin; num_on_tx++)//when num_on_tx=[1,n-1]
	    	{
	    		Ig_on += nmos_leak*combination(fanin, num_on_tx)*num_on_tx/2;//TODO: this is a approximation now, a precise computation will be very complicated.
	    	}
	    	Ig_on /=num_states;
	    	break;
	    case nor:
	    	// num_on_tx is the number of on tx in pull up network
	    	Ig_on += pmos_leak * fanin;//pull up network when all TXs are on.
	    	for (num_on_tx=1; num_on_tx<fanin; num_on_tx++)
	    	{
	    		Ig_on += pmos_leak*combination(fanin, num_on_tx)*num_on_tx/2;

	    	}
	    	//pull down network
	    	for (num_on_tx=1; num_on_tx<=fanin; num_on_tx++)//when num_on_tx=[1,n]
	        {
	    		Ig_on += nmos_leak*combination(fanin, num_on_tx)*num_on_tx;
	    	}
	    	Ig_on /=num_states;
	    	break;
	    case tri:
	    	Ig_on += (2*nmos_leak + 2*pmos_leak)/2;//enabled
	    	Ig_on += (nmos_leak + pmos_leak)/2; //disabled upper bound of leakage power
	    	Ig_on /=2;
	    	break;
	    case tg:
	    	Ig_on = (nmos_leak + pmos_leak)/2;
	    	// [한국어] 전송 게이트: NMOS+PMOS 모두 ON 상태에서 각각 터널링 → 평균
	    	break;
	    default:
	    	assert(0); // [한국어] 알 수 없는 게이트 유형 — 프로그래밍 오류
	    	break;
		  }

	    return Ig_on; // [한국어] 통계적 평균 게이트 터널링 누설전류 (A) 반환
}

/*
 * [한국어]
 * shortcircuit_simple - CMOS 인버터 단락전류 에너지 간소화 모델
 * @vt            : 임계전압 (V)
 * @velocity_index: 속도 포화(velocity saturation) 지수 — 드레인 전류 포화 특성
 * @c_in          : 입력 커패시턴스 (F)
 * @c_out         : 출력 부하 커패시턴스 (F)
 * @w_nmos/w_pmos : 트랜지스터 폭 (um, 현재 직접 사용 안 됨)
 * @i_on_n/p      : 출력 NMOS/PMOS 구동 전류 (A/um)
 * @i_on_n_in/p_in: 입력측 구동 전류 (A/um)
 * @vdd           : 공급 전압 (V)
 * @return        : 단락전류 에너지 (J) — 방전/충전 단락전류 에너지의 평균
 * 단락전류(short-circuit current): CMOS 전이 중 NMOS/PMOS 동시 ON 구간에서 직접 경로 형성.
 * 저속(low) 근사 모델: (Vdd-Vt-Vt/Vdd)³ 의존성의 경험적 수식 사용.
 * 호출 체인: 전력 모델 단락전류 추정 → [shortcircuit_simple]
 */
double shortcircuit_simple(
    double vt,
    double velocity_index,
    double c_in,
    double c_out,
    double w_nmos,
    double w_pmos,
    double i_on_n,
    double i_on_p,
    double i_on_n_in,
    double i_on_p_in,
    double vdd)
{

	double p_short_circuit, p_short_circuit_discharge, p_short_circuit_charge, p_short_circuit_discharge_low, p_short_circuit_charge_low;//this is actually energy
	// [한국어] 변수 명에도 'energy'라고 주석에 명시됨 — 주파수를 곱하면 전력(W)
	double fo_n, fo_p, fanout, beta_ratio, vt_to_vdd_ratio;

	fo_n	= i_on_n/i_on_n_in;   // [한국어] NMOS 유효 팬아웃 비 = 출력측/입력측 구동 전류
	fo_p	= i_on_p/i_on_p_in;   // [한국어] PMOS 유효 팬아웃 비
	fanout	= c_out/c_in;         // [한국어] 커패시턴스 기반 팬아웃
	beta_ratio = i_on_p/i_on_n;   // [한국어] PMOS/NMOS 전류 비 (대칭성 척도)
	vt_to_vdd_ratio = vt/vdd;     // [한국어] 정규화된 임계전압 비율 (Vt/Vdd)

	//p_short_circuit_discharge_low 	= 10/3*(pow(0.5-vt_to_vdd_ratio,3.0)/pow(velocity_index,2.0)/pow(2.0,3*vt_to_vdd_ratio*vt_to_vdd_ratio))*c_in*vdd*vdd*fo_p*fo_p/fanout/beta_ratio;
	p_short_circuit_discharge_low 	= 10/3*(pow(((vdd-vt)-vt_to_vdd_ratio),3.0)/pow(velocity_index,2.0)/pow(2.0,3*vt_to_vdd_ratio*vt_to_vdd_ratio))*c_in*vdd*vdd*fo_p*fo_p/fanout/beta_ratio;
	// [한국어] 방전 단락전류 에너지(저속 모델): PMOS→NMOS 전이 구간에서 발생하는 단락 에너지
	p_short_circuit_charge_low 		= 10/3*(pow(((vdd-vt)-vt_to_vdd_ratio),3.0)/pow(velocity_index,2.0)/pow(2.0,3*vt_to_vdd_ratio*vt_to_vdd_ratio))*c_in*vdd*vdd*fo_n*fo_n/fanout*beta_ratio;
	// [한국어] 충전 단락전류 에너지(저속 모델): NMOS→PMOS 전이 구간에서 발생하는 단락 에너지
//	double t1, t2, t3, t4, t5;
//	t1=pow(((vdd-vt)-vt_to_vdd_ratio),3);
//	t2=pow(velocity_index,2.0);
//	t3=pow(2.0,3*vt_to_vdd_ratio*vt_to_vdd_ratio);
//	t4=t1/t2/t3;
//	cout <<t1<<"t1\n"<<t2<<"t2\n"<<t3<<"t3\n"<<t4<<"t4\n"<<fanout<<endl;


//	t1=pow(((vdd-vt)-vt_to_vdd_ratio),1.5);
//	t2=pow(2, 3*vt_to_vdd_ratio+2*velocity_index);
//	t3=t1/t2;
//	cout <<t1<<"t1\n"<<t2<<"t2\n"<<t3<<"t3\n"<<t4<<"t4\n"<<fanout<<endl;
//	p_short_circuit_discharge = 1.0/(1.0/p_short_circuit_discharge_low + 1.0/p_short_circuit_discharge_high);
//	p_short_circuit_charge = 1/(1/p_short_circuit_charge_low + 1/p_short_circuit_charge_high); //harmmoic mean cannot be applied simple formulas.

	p_short_circuit_discharge = p_short_circuit_discharge_low; // [한국어] 저속 모델만 사용 (고속 모델 미구현)
	p_short_circuit_charge = p_short_circuit_charge_low;       // [한국어] 저속 모델만 사용
	p_short_circuit = (p_short_circuit_discharge + p_short_circuit_charge)/2;
	// [한국어] 방전/충전 단락전류 에너지 산술 평균 — 50% 듀티 사이클 가정

  return (p_short_circuit); // [한국어] 단락전류 에너지 (J) 반환
}

/*
 * [한국어]
 * shortcircuit - CMOS 단락전류 에너지 완전 모델 (현재 미완성 스텁)
 * @(동일 파라미터) : shortcircuit_simple()과 동일
 * @return : 0 (미구현) — 실제 계산에서는 shortcircuit_simple()이 사용됨
 * 원래 더 정확한 단락전류 모델을 구현하려 했으나 현재 비어 있는 스텁(stub) 함수.
 * 호출 체인: 단락전류 계산 → [shortcircuit] (현재 직접 사용 안 됨)
 */
double shortcircuit(
    double vt,
    double velocity_index,
    double c_in,
    double c_out,
    double w_nmos,
    double w_pmos,
    double i_on_n,
    double i_on_p,
    double i_on_n_in,
    double i_on_p_in,
    double vdd)
{

	double p_short_circuit=0; // [한국어] 미구현 — 항상 0 반환
  return (p_short_circuit);
}
