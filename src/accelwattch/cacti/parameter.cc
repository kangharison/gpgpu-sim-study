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
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
 *
 ***************************************************************************/

/*
 * [한국어 설명] CACTI 캐시/메모리 파라미터 구현 (parameter.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 CACTI(Cache Access and Cycle Time Information) 도구에서
 * 캐시/메모리 배열의 물리적 분할 구성(Ndwl, Ndbl, Nspd, Ndsam_lev_1,
 * Ndsam_lev_2)에 대한 파생 파라미터 전체를 계산하고 저장하는
 * DynamicParameter 클래스의 생성자와, TechnologyParameter 각 서브 구조체의
 * 디버그 출력 함수(display)를 구현한다. DynamicParameter 생성자는 주어진
 * 파티셔닝 조합이 물리적으로 유효한지 검증하고, 유효한 경우 서브어레이
 * 행/열 수, mat 구성, 비트라인 정전용량, 센스앰프 입력 전압(V_b_sense),
 * 데이터/태그/검색 입출력 폭 등 이후 타이밍·면적·전력 계산에 필요한 모든
 * 중간 파라미터를 확정한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CACTI는 AccelWattch 전력 모델(accelwattch/) 내에 임베드되어 있으며,
 * GPU 시뮬레이터(gpgpu-sim/)의 캐시/DRAM 전력 추정에 사용된다.
 * 호출 체인: AccelWattch 전력 요청 → cacti/cacti.cc (최상위 탐색 루프) →
 *   DynamicParameter 생성자 [이 파일] → 유효하면 UCA/NUCA 분석 진행 →
 *   결과를 AccelWattch에 반환.
 * DynamicParameter는 탐색 공간(Ndwl × Ndbl × Nspd × …) 내의 한 점을
 * 나타내며, 탐색 루프가 모든 점을 열거하면서 이 생성자를 반복 호출한다.
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 단일 스레드, 시뮬레이션 초기화 시.
 *
 * === 타 모듈과의 연결 ===
 * - 의존: parameter.h (DynamicParameter, TechnologyParameter 선언),
 *         area.h (drain_C_() 트랜지스터 정전용량 계산 함수),
 *         g_ip (전역 InputParameter 포인터 — 캐시 크기·연관도·포트 수 등),
 *         g_tp (전역 TechnologyParameter — 공정 파라미터, 셀 치수).
 * - 출력: 유효한 DynamicParameter 객체(is_valid=true)는 uca_org_t /
 *         nuca_org_t 계산 루틴(uca.cc, nuca.cc)에 전달된다.
 * - 데이터 흐름: g_ip → DynamicParameter 생성자 → num_r_subarray /
 *   num_c_subarray / num_mats / V_b_sense / num_do_b_mat 등 → UCA 분석.
 *
 * === 주요 함수/구조체 요약 ===
 * - TechnologyParameter::DeviceType::display()    : 트랜지스터 파라미터 출력
 * - TechnologyParameter::InterconnectType::display(): 배선 파라미터 출력
 * - TechnologyParameter::ScalingFactor::display() : 스케일링 계수 출력
 * - TechnologyParameter::MemoryType::display()    : 메모리 셀 치수 출력
 * - TechnologyParameter::display()                : TechnologyParameter 전체 출력
 * - DynamicParameter::DynamicParameter()          : 기본 생성자 (trivial)
 * - DynamicParameter::DynamicParameter(is_tag_, ..., is_main_mem_)
 *     : 핵심 생성자 — 파티셔닝 파라미터로부터 모든 파생값 계산;
 *       물리 제약 위반 시 is_valid=false로 조기 반환.
 */


#include <iostream>  // [한국어] cout/endl — 디버그 출력용
#include <string>    // [한국어] std::string — indent 문자열 생성용
#include <iomanip>   // [한국어] setw() — 출력 열 너비 정렬용

#include "parameter.h"  // [한국어] DynamicParameter, TechnologyParameter 선언 및 상수(VBITSENSEMIN 등)
#include "area.h"       // [한국어] drain_C_() — 트랜지스터 드레인 정전용량 계산 (비트라인 Cbl 산출에 사용)

using namespace std; // [한국어] std:: 접두사 생략 — CACTI 전체 관습


/* [한국어] 전역 파라미터 포인터/객체 정의.
 * g_ip: 사용자 입력 파라미터(캐시 크기, 연관도, 기술 노드 등)를 가리키는 전역 포인터.
 *   설정자: cacti_interface() / McPAT 진입점에서 new InputParameter 후 대입.
 *   읽는 자: DynamicParameter 생성자 및 모든 CACTI 분석 루틴.
 *   동기화: 단일 스레드 실행이므로 락 불필요.
 * g_tp: 공정 기술 파라미터(트랜지스터 특성, 배선 저항/정전용량, 셀 치수 등) 전역 객체.
 *   설정자: technological_param() (tech_params.cc)에서 기술 노드별로 초기화.
 *   읽는 자: DynamicParameter 생성자, 타이밍/면적/전력 계산 루틴 전반. */
InputParameter * g_ip; // [한국어] 전역 입력 파라미터 포인터 (캐시 구성 명세)
TechnologyParameter g_tp; // [한국어] 전역 기술 파라미터 객체 (공정 특성값 집합)


/*
 * [한국어]
 * TechnologyParameter::DeviceType::display - 트랜지스터 디바이스 파라미터를 stdout에 출력
 *
 * @indent: 출력 시 앞에 붙일 공백 수 (계층적 출력을 위한 들여쓰기)
 * @return: 없음 (void)
 *
 * TechnologyParameter 전체 출력(TechnologyParameter::display)의 일부로 호출되어
 * SRAM 셀·DRAM 접근 트랜지스터·주변 회로 트랜지스터 등 각 DeviceType 인스턴스의
 * 공정 파라미터를 사람이 읽기 쉬운 형식으로 출력한다.
 * 디버그·검증 목적으로만 사용되며 시뮬레이션 결과에는 영향을 주지 않는다.
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 스레드.
 *
 * 호출 체인:
 *   TechnologyParameter::display() → [DeviceType::display()]
 */
void TechnologyParameter::DeviceType::display(uint32_t indent)
{
  string indent_str(indent, ' '); // [한국어] indent 개수만큼 공백 문자열 생성 — 계층 출력 들여쓰기

  // [한국어] 각 트랜지스터 파라미터를 12열 너비로 정렬하여 출력
  cout << indent_str << "C_g_ideal = " << setw(12) << C_g_ideal << " F/um" << endl;   // [한국어] 이상적인 게이트 산화막 정전용량 [F/um]
  cout << indent_str << "C_fringe  = " << setw(12) << C_fringe  << " F/um" << endl;   // [한국어] 게이트 프린지 정전용량 [F/um]
  cout << indent_str << "C_overlap = " << setw(12) << C_overlap << " F/um" << endl;   // [한국어] 게이트-드레인/소스 오버랩 정전용량 [F/um]
  cout << indent_str << "C_junc    = " << setw(12) << C_junc    << " F/um^2" << endl; // [한국어] pn 접합 정전용량 [F/um²]
  cout << indent_str << "l_phy     = " << setw(12) << l_phy     << " um" << endl;     // [한국어] 물리적 게이트 길이 [um]
  cout << indent_str << "l_elec    = " << setw(12) << l_elec    << " um" << endl;     // [한국어] 전기적 유효 게이트 길이 [um]
  cout << indent_str << "R_nch_on  = " << setw(12) << R_nch_on  << " ohm-um" << endl; // [한국어] NMOS 온 저항 [ohm·um]
  cout << indent_str << "R_pch_on  = " << setw(12) << R_pch_on  << " ohm-um" << endl; // [한국어] PMOS 온 저항 [ohm·um]
  cout << indent_str << "Vdd       = " << setw(12) << Vdd       << " V" << endl;      // [한국어] 전원 전압 [V]
  cout << indent_str << "Vth       = " << setw(12) << Vth       << " V" << endl;      // [한국어] 문턱 전압 [V]
  cout << indent_str << "I_on_n    = " << setw(12) << I_on_n    << " A/um" << endl;   // [한국어] NMOS 온 전류 [A/um]
  cout << indent_str << "I_on_p    = " << setw(12) << I_on_p    << " A/um" << endl;   // [한국어] PMOS 온 전류 [A/um]
  cout << indent_str << "I_off_n   = " << setw(12) << I_off_n   << " A/um" << endl;   // [한국어] NMOS 오프(누설) 전류 [A/um]
  cout << indent_str << "I_off_p   = " << setw(12) << I_off_p   << " A/um" << endl;   // [한국어] PMOS 오프(누설) 전류 [A/um]
  cout << indent_str << "C_ox      = " << setw(12) << C_ox      << " F/um^2" << endl; // [한국어] 단위 면적당 산화막 정전용량 [F/um²]
  cout << indent_str << "t_ox      = " << setw(12) << t_ox      << " um" << endl;     // [한국어] 게이트 산화막 두께 [um]
  cout << indent_str << "n_to_p_eff_curr_drv_ratio = " << n_to_p_eff_curr_drv_ratio << endl; // [한국어] NMOS/PMOS 유효 전류 구동 비율 (인버터 크기 조정에 사용)
}



/*
 * [한국어]
 * TechnologyParameter::InterconnectType::display - 배선(인터커넥트) 파라미터를 stdout에 출력
 *
 * @indent: 출력 시 앞에 붙일 공백 수
 * @return: 없음 (void)
 *
 * local/inside-mat/outside-mat 세 종류의 배선 파라미터 인스턴스 각각에 대해
 * pitch, 단위 길이당 저항, 단위 길이당 정전용량을 출력한다.
 * TechnologyParameter::display()가 wire_local, wire_inside_mat, wire_outside_mat
 * 세 인스턴스에 대해 순서대로 호출한다.
 *
 * 호출 체인:
 *   TechnologyParameter::display() → [InterconnectType::display()]
 */
void TechnologyParameter::InterconnectType::display(uint32_t indent)
{
  string indent_str(indent, ' '); // [한국어] 들여쓰기 공백 문자열 생성

  cout << indent_str << "pitch    = " << setw(12) << pitch    << " um" << endl;     // [한국어] 배선 피치(인접 배선 중심 간격) [um]
  cout << indent_str << "R_per_um = " << setw(12) << R_per_um << " ohm/um" << endl; // [한국어] 단위 길이당 배선 저항 [ohm/um]
  cout << indent_str << "C_per_um = " << setw(12) << C_per_um << " F/um" << endl;   // [한국어] 단위 길이당 배선 정전용량 [F/um]
}

/*
 * [한국어]
 * TechnologyParameter::ScalingFactor::display - 기술 노드 스케일링 계수를 stdout에 출력
 *
 * @indent: 출력 시 앞에 붙일 공백 수
 * @return: 없음 (void)
 *
 * logic_scaling_co_eff(로직 게이트 스케일링 계수)와 core_tx_density(코어 트랜지스터
 * 집적 밀도 [개/um²])를 출력한다. McPAT 전력 스케일링 계산에서 참조되는 값이다.
 *
 * 호출 체인:
 *   TechnologyParameter::display() → [ScalingFactor::display()]
 */
void TechnologyParameter::ScalingFactor::display(uint32_t indent)
{
  string indent_str(indent, ' '); // [한국어] 들여쓰기 공백 문자열 생성

  cout << indent_str << "logic_scaling_co_eff    = " << setw(12) << logic_scaling_co_eff << endl;                     // [한국어] 로직 게이트 스케일링 계수 (무차원)
  cout << indent_str << "curr_core_tx_density = " << setw(12) << core_tx_density << " # of tx/um^2" << endl; // [한국어] 코어 트랜지스터 집적 밀도 [개/um²]
}

/*
 * [한국어]
 * TechnologyParameter::MemoryType::display - 메모리 셀 치수 파라미터를 stdout에 출력
 *
 * @indent: 출력 시 앞에 붙일 공백 수
 * @return: 없음 (void)
 *
 * SRAM 또는 DRAM 셀의 물리적 치수(b_w, b_h: 기본 비트셀 폭/높이)와 트랜지스터 폭,
 * 비트라인 프리차지 전압(Vbitpre)을 출력한다.
 * TechnologyParameter::display()가 sram, dram 두 인스턴스에 대해 호출한다.
 *
 * 호출 체인:
 *   TechnologyParameter::display() → [MemoryType::display()]
 */
void TechnologyParameter::MemoryType::display(uint32_t indent)
{
  string indent_str(indent, ' '); // [한국어] 들여쓰기 공백 문자열 생성

  cout << indent_str << "b_w         = " << setw(12) << b_w << " um" << endl;         // [한국어] 비트셀 폭 [um]
  cout << indent_str << "b_h         = " << setw(12) << b_h << " um" << endl;         // [한국어] 비트셀 높이 [um]
  cout << indent_str << "cell_a_w    = " << setw(12) << cell_a_w << " um" << endl;    // [한국어] 접근 트랜지스터 폭 [um] (drain_C_ 계산에 사용)
  cout << indent_str << "cell_pmos_w = " << setw(12) << cell_pmos_w << " um" << endl; // [한국어] SRAM 셀 내 PMOS 트랜지스터 폭 [um]
  cout << indent_str << "cell_nmos_w = " << setw(12) << cell_nmos_w << " um" << endl; // [한국어] SRAM 셀 내 NMOS 트랜지스터 폭 [um]
  cout << indent_str << "Vbitpre     = " << setw(12) << Vbitpre << " V" << endl;      // [한국어] 비트라인 프리차지 전압 [V]
}



/*
 * [한국어]
 * TechnologyParameter::display - TechnologyParameter 전체 구조를 재귀적으로 stdout에 출력
 *
 * @indent: 출력 시 앞에 붙일 공백 수 (최상위 호출 시 보통 0)
 * @return: 없음 (void)
 *
 * 최상위 기술 파라미터(배선 오버헤드, NMOS/PMOS 트랜지스터 최소/최대 폭, FO4,
 * 비교기/감지증폭기 트랜지스터 폭, DRAM 셀 파라미터 등)를 먼저 출력한 뒤,
 * DeviceType·InterconnectType·MemoryType 각 서브 구조체의 display()를 호출하여
 * 전체 기술 파라미터 집합을 계층적으로 출력한다.
 * 디버그·검증 목적으로만 사용되며, CACTI 탐색 결과에는 영향을 주지 않는다.
 *
 * 호출 체인:
 *   cacti_interface() 또는 McPAT 진단 경로 → [TechnologyParameter::display()]
 *     → DeviceType::display(), InterconnectType::display(), MemoryType::display()
 */
void TechnologyParameter::display(uint32_t indent)
{
  string indent_str(indent, ' '); // [한국어] 들여쓰기 공백 문자열 생성

  // [한국어] 최상위 전역 기술 파라미터 출력 (배선·트랜지스터 폭·타이밍 상수 등)
  cout << indent_str << "ram_wl_stitching_overhead_ = " << setw(12) << ram_wl_stitching_overhead_ << " um" << endl; // [한국어] 워드라인 스티칭 오버헤드 [um]
  cout << indent_str << "min_w_nmos_                = " << setw(12) << min_w_nmos_                << " um" << endl; // [한국어] NMOS 최소 트랜지스터 폭 [um]
  cout << indent_str << "max_w_nmos_                = " << setw(12) << max_w_nmos_                << " um" << endl; // [한국어] NMOS 최대 트랜지스터 폭 [um]
  cout << indent_str << "unit_len_wire_del          = " << setw(12) << unit_len_wire_del          << " s/um^2" << endl; // [한국어] 단위 길이²당 배선 지연 [s/um²]
  cout << indent_str << "FO4                        = " << setw(12) << FO4                        << " s" << endl;  // [한국어] FO4(Fan-Out-of-4) 인버터 지연 — 기준 타이밍 단위 [s]
  cout << indent_str << "kinv                       = " << setw(12) << kinv                       << " s" << endl;  // [한국어] 인버터 지연 상수 [s]
  cout << indent_str << "vpp                        = " << setw(12) << vpp                        << " V" << endl;  // [한국어] 워드라인 부스트 전압(Vpp) [V]
  cout << indent_str << "w_sense_en                 = " << setw(12) << w_sense_en                 << " um" << endl; // [한국어] 감지증폭기 인에이블 트랜지스터 폭 [um]
  cout << indent_str << "w_sense_n                  = " << setw(12) << w_sense_n                  << " um" << endl; // [한국어] 감지증폭기 NMOS 트랜지스터 폭 [um]
  cout << indent_str << "w_sense_p                  = " << setw(12) << w_sense_p                  << " um" << endl; // [한국어] 감지증폭기 PMOS 트랜지스터 폭 [um]
  cout << indent_str << "w_iso                      = " << setw(12) << w_iso                      << " um" << endl; // [한국어] 비트라인 격리 트랜지스터 폭 [um]
  cout << indent_str << "w_poly_contact             = " << setw(12) << w_poly_contact             << " um" << endl; // [한국어] 폴리실리콘 컨택 폭 [um]
  cout << indent_str << "spacing_poly_to_poly       = " << setw(12) << spacing_poly_to_poly       << " um" << endl; // [한국어] 폴리-폴리 간격 [um]
  cout << indent_str << "spacing_poly_to_contact    = " << setw(12) << spacing_poly_to_contact    << " um" << endl; // [한국어] 폴리-컨택 간격 [um]
  cout << endl;
  // [한국어] 비교기(comparator) 인버터 트랜지스터 폭 파라미터 출력
  cout << indent_str << "w_comp_inv_p1              = " << setw(12) << w_comp_inv_p1 << " um" << endl; // [한국어] 비교기 인버터 1단 PMOS 폭
  cout << indent_str << "w_comp_inv_p2              = " << setw(12) << w_comp_inv_p2 << " um" << endl; // [한국어] 비교기 인버터 2단 PMOS 폭
  cout << indent_str << "w_comp_inv_p3              = " << setw(12) << w_comp_inv_p3 << " um" << endl; // [한국어] 비교기 인버터 3단 PMOS 폭
  cout << indent_str << "w_comp_inv_n1              = " << setw(12) << w_comp_inv_n1 << " um" << endl; // [한국어] 비교기 인버터 1단 NMOS 폭
  cout << indent_str << "w_comp_inv_n2              = " << setw(12) << w_comp_inv_n2 << " um" << endl; // [한국어] 비교기 인버터 2단 NMOS 폭
  cout << indent_str << "w_comp_inv_n3              = " << setw(12) << w_comp_inv_n3 << " um" << endl; // [한국어] 비교기 인버터 3단 NMOS 폭
  cout << indent_str << "w_eval_inv_p               = " << setw(12) << w_eval_inv_p  << " um" << endl; // [한국어] 평가 인버터 PMOS 폭
  cout << indent_str << "w_eval_inv_n               = " << setw(12) << w_eval_inv_n  << " um" << endl; // [한국어] 평가 인버터 NMOS 폭
  cout << indent_str << "w_comp_n                   = " << setw(12) << w_comp_n      << " um" << endl; // [한국어] 비교기 NMOS 폭
  cout << indent_str << "w_comp_p                   = " << setw(12) << w_comp_p      << " um" << endl; // [한국어] 비교기 PMOS 폭
  cout << endl;
  // [한국어] DRAM 셀 전기적 파라미터 출력
  cout << indent_str << "dram_cell_I_on             = " << setw(12) << dram_cell_I_on << " A/um" << endl; // [한국어] DRAM 셀 접근 트랜지스터 온 전류
  cout << indent_str << "dram_cell_Vdd              = " << setw(12) << dram_cell_Vdd  << " V" << endl;    // [한국어] DRAM 셀 전원 전압
  cout << indent_str << "dram_cell_I_off_worst_case_len_temp = " << setw(12) << dram_cell_I_off_worst_case_len_temp << " A/um" << endl; // [한국어] DRAM 셀 최악 누설 전류 (리프레시 주기 계산 기준)
  cout << indent_str << "dram_cell_C                = " << setw(12) << dram_cell_C               << " F" << endl;    // [한국어] DRAM 스토리지 커패시터 정전용량 [F]
  cout << indent_str << "gm_sense_amp_latch         = " << setw(12) << gm_sense_amp_latch        << " F/s" << endl;  // [한국어] 감지증폭기 래치 상호전도 [F/s]
  cout << endl;
  // [한국어] 비트라인 관련 트랜지스터 폭 파라미터 출력
  cout << indent_str << "w_nmos_b_mux               = " << setw(12) << w_nmos_b_mux              << " um" << endl; // [한국어] 비트라인 MUX NMOS 폭
  cout << indent_str << "w_nmos_sa_mux              = " << setw(12) << w_nmos_sa_mux             << " um" << endl; // [한국어] 감지증폭기 MUX NMOS 폭
  cout << indent_str << "w_pmos_bl_precharge        = " << setw(12) << w_pmos_bl_precharge       << " um" << endl; // [한국어] 비트라인 프리차지 PMOS 폭
  cout << indent_str << "w_pmos_bl_eq               = " << setw(12) << w_pmos_bl_eq              << " um" << endl; // [한국어] 비트라인 이퀄라이즈 PMOS 폭
  cout << indent_str << "MIN_GAP_BET_P_AND_N_DIFFS  = " << setw(12) << MIN_GAP_BET_P_AND_N_DIFFS << " um" << endl; // [한국어] P형/N형 확산 영역 간 최소 간격
  cout << indent_str << "HPOWERRAIL                 = " << setw(12) << HPOWERRAIL                << " um" << endl; // [한국어] 전원 레일 높이 [um]
  cout << indent_str << "cell_h_def                 = " << setw(12) << cell_h_def                << " um" << endl; // [한국어] 기본 셀 높이 [um]

  // [한국어] 서브 구조체별 파라미터를 2칸 추가 들여쓰기하여 재귀 출력
  cout << endl;
  cout << indent_str << "SRAM cell transistor: " << endl;
  sram_cell.display(indent + 2); // [한국어] SRAM 셀 트랜지스터 파라미터 출력

  cout << endl;
  cout << indent_str << "DRAM access transistor: " << endl;
  dram_acc.display(indent + 2); // [한국어] DRAM 접근 트랜지스터 파라미터 출력

  cout << endl;
  cout << indent_str << "DRAM wordline transistor: " << endl;
  dram_wl.display(indent + 2); // [한국어] DRAM 워드라인 트랜지스터 파라미터 출력

  cout << endl;
  cout << indent_str << "peripheral global transistor: " << endl;
  peri_global.display(indent + 2); // [한국어] 주변 회로(전역) 트랜지스터 파라미터 출력

  cout << endl;
  cout << indent_str << "wire local" << endl;
  wire_local.display(indent + 2); // [한국어] 로컬(셀 내부) 배선 파라미터 출력

  cout << endl;
  cout << indent_str << "wire inside mat" << endl;
  wire_inside_mat.display(indent + 2); // [한국어] mat 내부 배선 파라미터 출력

  cout << endl;
  cout << indent_str << "wire outside mat" << endl;
  wire_outside_mat.display(indent + 2); // [한국어] mat 외부(H-tree) 배선 파라미터 출력

  cout << endl;
  cout << indent_str << "SRAM" << endl;
  sram.display(indent + 2); // [한국어] SRAM 셀 치수 파라미터 출력

  cout << endl;
  cout << indent_str << "DRAM" << endl;
  dram.display(indent + 2); // [한국어] DRAM 셀 치수 파라미터 출력
}


/*
 * [한국어]
 * DynamicParameter::DynamicParameter (기본 생성자) - 유효한 빈 DynamicParameter 객체 생성
 *
 * @return: 없음 (생성자)
 *
 * use_inp_params=0, cell 기본값, is_valid=true로 초기화하는 trivial 기본 생성자.
 * 실제 파티셔닝 파라미터 계산은 수행하지 않는다.
 * UCA/NUCA 분석 루틴에서 결과를 담을 빈 컨테이너로 사용되거나,
 * 파라미터 없이 구조체를 기본 초기화할 때 사용된다.
 *
 * 호출 체인:
 *   UCA / NUCA 분석 루틴 → [DynamicParameter()]
 */
DynamicParameter::DynamicParameter():
  use_inp_params(0), cell(), is_valid(true) // [한국어] 기본값: 입력 파라미터 미사용, 빈 cell 치수, 유효 상태
{
}



/*
 * [한국어]
 * DynamicParameter::DynamicParameter (주 생성자) - 파티셔닝 파라미터로부터 모든 파생 파라미터 계산
 *
 * @is_tag_       : 태그 배열 여부 (true=태그, false=데이터). 태그/데이터는 서브어레이 크기 계산 공식이 다름
 * @pure_ram_     : 순수 RAM 모드 플래그 (연관도 없는 직접 매핑 RAM)
 * @pure_cam_     : 순수 CAM(Content-Addressable Memory) 모드 플래그
 * @Nspd_         : Nspd — 비트라인당 데이터 비트 수, 즉 열 멀티플렉싱 인수
 * @Ndwl_         : Ndwl — 수평 방향(열 방향) 서브어레이 분할 수
 * @Ndbl_         : Ndbl — 수직 방향(행 방향) 서브어레이 분할 수
 * @Ndcm_         : Ndcm — 비트라인 MUX 차수 (몇 개의 비트라인이 하나의 감지증폭기를 공유하는가)
 * @Ndsam_lev_1_  : 1차 감지증폭기 MUX 계층 인수
 * @Ndsam_lev_2_  : 2차 감지증폭기 MUX 계층 인수
 * @is_main_mem_  : 주 메모리(DRAM) 모드 플래그 — 페이지 모드·프리패치 너비 계산에 영향
 * @return        : 없음 (생성자). is_valid 필드로 유효성 전달.
 *
 * CACTI 탐색 루프가 (Ndwl, Ndbl, Nspd, Ndcm, Ndsam_lev_1, Ndsam_lev_2)의
 * 모든 조합을 열거하면서 이 생성자를 반복 호출한다. 생성자는:
 *   1) FA/CAM/DRAM 전용 제약 조건 검사 (위반 시 is_valid=false로 조기 반환)
 *   2) 서브어레이 행(num_r_subarray)·열(num_c_subarray) 수 계산
 *   3) 물리 한계(MINSUBARRAYROWS ~ MAXSUBARRAYROWS 등) 검사
 *   4) 셀 물리 치수(cell.h, cell.w) 계산
 *   5) 비트라인 정전용량(C_bl)·감지 전압(V_b_sense) 계산
 *   6) mat 구성(num_mats_h_dir, num_mats_v_dir, num_mats) 결정
 *   7) 서브뱅크/mat/뱅크 단위 데이터 입출력 폭(num_do_b_mat 등) 계산
 *   8) 어드레스 비트 수(number_addr_bits_mat) 계산
 *   9) ECC 조정 적용
 *  10) is_valid = true 설정 (모든 검사 통과 시)
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 스레드, 시뮬레이션 초기화 시.
 *
 * 호출 체인:
 *   cacti.cc 탐색 루프 → [DynamicParameter(is_tag_, ..., is_main_mem_)]
 *     → drain_C_() (area.h)
 */
DynamicParameter::DynamicParameter(
    bool is_tag_,
    int pure_ram_,
    int pure_cam_,
    double Nspd_,
    unsigned int Ndwl_,
    unsigned int Ndbl_,
    unsigned int Ndcm_,
    unsigned int Ndsam_lev_1_,
    unsigned int Ndsam_lev_2_,
    bool is_main_mem_):
  // [한국어] 멤버 초기화 리스트: 인수를 그대로 저장하고 파생 값들은 0/false로 초기화
  is_tag(is_tag_), pure_ram(pure_ram_), pure_cam(pure_cam_), tagbits(0), Nspd(Nspd_), Ndwl(Ndwl_), Ndbl(Ndbl_),Ndcm(Ndcm_),
  Ndsam_lev_1(Ndsam_lev_1_), Ndsam_lev_2(Ndsam_lev_2_),
  number_way_select_signals_mat(0), V_b_sense(0), use_inp_params(0),
  is_main_mem(is_main_mem_), cell(), is_valid(false) // [한국어] is_valid=false: 모든 검사 통과 후 마지막에 true로 전환
{

  // [한국어] 입출력 비트 폭 카운터 초기화 — 이후 조건 분기에서 선택적으로 설정됨
  num_di_b_bank_per_port=0; // [한국어] 포트당 뱅크 데이터 입력 비트 수
  num_do_b_bank_per_port=0; // [한국어] 포트당 뱅크 데이터 출력 비트 수
  num_di_b_mat=0;           // [한국어] mat당 데이터 입력 비트 수
  num_do_b_mat=0;           // [한국어] mat당 데이터 출력 비트 수
  num_di_b_subbank=0;       // [한국어] 서브뱅크당 데이터 입력 비트 수
  num_do_b_subbank=0;       // [한국어] 서브뱅크당 데이터 출력 비트 수
  num_si_b_mat=0;           // [한국어] mat당 검색 데이터 입력 비트 수 (FA/CAM 전용)
  num_so_b_mat=0;           // [한국어] mat당 검색 데이터 출력 비트 수 (FA/CAM 전용)
  num_si_b_subbank=0;       // [한국어] 서브뱅크당 검색 입력 비트 수 (FA/CAM 전용)
  num_so_b_subbank=0;       // [한국어] 서브뱅크당 검색 출력 비트 수 (FA/CAM 전용)
  num_si_b_bank_per_port=0; // [한국어] 포트당 뱅크 검색 입력 비트 수 (FA/CAM 전용)
  num_so_b_bank_per_port=0; // [한국어] 포트당 뱅크 검색 출력 비트 수 (FA/CAM 전용)


  // [한국어] 셀 기술 타입 결정: 태그 배열이면 tag_arr_ram_cell_tech_type, 아니면 data_arr_ram_cell_tech_type
  ram_cell_tech_type = (is_tag) ? g_ip->tag_arr_ram_cell_tech_type : g_ip->data_arr_ram_cell_tech_type;
  // [한국어] DRAM 여부 판별: lp_dram(저전력 DRAM) 또는 comm_dram(상용 DRAM)이면 is_dram=true
  is_dram            = ((ram_cell_tech_type == lp_dram) || (ram_cell_tech_type == comm_dram));

  // [한국어] 스택 다이 레이어 수로 나눈 다이당 용량 [바이트] — 3D 스택 메모리 지원을 위한 분할
  unsigned int capacity_per_die = g_ip->cache_sz / NUMBER_STACKED_DIE_LAYERS;  // capacity per stacked die layer
  // [한국어] 로컬 배선 파라미터 참조 — 셀 치수 계산 시 wire_local.pitch 사용
  const TechnologyParameter::InterconnectType & wire_local = g_tp.wire_local;
  // [한국어] 완전 연관(Fully-Associative) 캐시 여부 — FA는 Ndwl/Ndcm/Nspd가 모두 1로 고정됨
  fully_assoc = (g_ip->fully_assoc) ? true : false;


  // -----------------------------------------------------------------------
  // [한국어] 검사 1: FA/CAM 배열의 파티셔닝 파라미터 제약 조건 확인
  // FA/CAM 배열은 CACTI 2.0 리포트 기준, Ndwl=1, Ndcm=1, Nspd=1,
  // Ndsam_lev_1=1, Ndsam_lev_2=1 로 고정되어야 하며, Ndbl >= 2 이어야 함.
  // -----------------------------------------------------------------------
  if (fully_assoc || pure_cam)
  { // fully-assocative cache -- ref: CACTi 2.0 report
	  if (Ndwl != 1 ||            //Ndwl is fixed to 1 for FA
			  Ndcm != 1 ||            //Ndcm is fixed to 1 for FA
			  Nspd < 1 || Nspd > 1 || //Nspd is fixed to 1 for FA
			  Ndsam_lev_1 != 1 ||     //Ndsam_lev_1 is fixed to one
			  Ndsam_lev_2 != 1 ||     //Ndsam_lev_2 is fixed to one
			  Ndbl < 2)
	  {
          return; // [한국어] FA/CAM 제약 위반 — is_valid=false 상태로 조기 반환
	  }
  }

  // [한국어] 검사 2: DRAM 데이터 배열에서 Ndcm > 1은 불가
  // DRAM은 각 비트라인마다 전용 감지증폭기가 있으므로 비트라인 MUX 차수는 반드시 1이어야 함
  if ((is_dram) && (!is_tag) && (Ndcm > 1))
  {
	  return;  // For a DRAM array, each bitline has its own sense-amp
  }

  // If it's not an FA tag/data array, Ndwl should be at least two and Ndbl should be
  // at least two because an array is assumed to have at least one mat. And a mat
  // is formed out of two horizontal subarrays and two vertical subarrays
  // [한국어] 검사 3: FA가 아닌 일반 배열은 Ndwl >= 1, Ndbl >= 1 이어야 함
  // (mat는 최소 2×2 서브어레이로 구성되므로 각 방향 최소 1 이상)
  if (fully_assoc == false && (Ndwl < 1 || Ndbl < 1))
  {
	  return; // [한국어] 최소 분할 조건 미달 — 조기 반환
  }

  // -----------------------------------------------------------------------
  // [한국어] 서브어레이 행(num_r_subarray)·열(num_c_subarray) 수 계산
  // 일반(non-FA/non-CAM) 배열과 FA/CAM 배열의 계산 공식이 다름
  // -----------------------------------------------------------------------
  //***********compute row, col of an subarray
  if (!(fully_assoc || pure_cam))//Not fully_asso nor cam
  {
	  // if data array, let tagbits = 0
	  if (is_tag) // [한국어] 태그 배열: tagbits 계산 후 태그 기반 서브어레이 크기 산출
	  {
		  if (g_ip->specific_tag) // [한국어] 사용자가 태그 비트 폭을 명시한 경우
		  {
			  tagbits = g_ip->tag_w; // [한국어] 명시된 태그 비트 폭을 그대로 사용
		  }
		  else // [한국어] 태그 비트 폭을 자동 계산: 주소 비트에서 인덱스·뱅크 비트를 제거
		  {
			  // [한국어] ADDRESS_BITS + EXTRA_TAG_BITS: 전체 주소 비트 + 여분 비트
			  // _log2(capacity_per_die): 인덱스 비트 수 (용량 기반)
			  // _log2(tag_assoc*2 - 1): 연관도에 따른 보정
			  // _log2(nbanks): 뱅크 선택 비트 수
			  tagbits = ADDRESS_BITS + EXTRA_TAG_BITS - _log2(capacity_per_die) +
			  _log2(g_ip->tag_assoc*2 - 1) - _log2(g_ip->nbanks);

		  }
		  // [한국어] tagbits를 4의 배수로 올림 정렬 — 물리 레이아웃 정렬 요구사항
		  tagbits = (((tagbits + 3) >> 2) << 2);

		  // [한국어] 태그 배열 서브어레이 행 수: 총 엔트리 수 / (뱅크 × 블록 × 연관도 × Ndbl × Nspd)
		  num_r_subarray = (int)ceil(capacity_per_die / (g_ip->nbanks *
				  g_ip->block_sz * g_ip->tag_assoc * Ndbl * Nspd));// + EPSILON);
		  // [한국어] 태그 배열 서브어레이 열 수: 태그 비트 × 연관도 × Nspd / Ndwl
		  num_c_subarray = (int)ceil((tagbits * g_ip->tag_assoc * Nspd / Ndwl));// + EPSILON);
		  //burst_length = 1;
	  }
	  else // [한국어] 데이터 배열: 블록 크기와 데이터 연관도 기반으로 서브어레이 크기 산출
	  {
		  // [한국어] 데이터 배열 서브어레이 행 수: 총 용량 / (뱅크 × 블록 × 데이터연관도 × Ndbl × Nspd)
		  num_r_subarray = (int)ceil(capacity_per_die / (g_ip->nbanks *
				  g_ip->block_sz * g_ip->data_assoc * Ndbl * Nspd));// + EPSILON);
		  // [한국어] 데이터 배열 서브어레이 열 수: 블록 크기(바이트→비트) × 데이터연관도 × Nspd / Ndwl
		  num_c_subarray = (int)ceil((8 * g_ip->block_sz * g_ip->data_assoc * Nspd / Ndwl));// + EPSILON); + EPSILON);
		  // burst_length = g_ip->block_sz * 8 / g_ip->out_w;
	  }

	  // [한국어] 서브어레이 크기 물리 한계 검사 — 너무 작거나 너무 크면 레이아웃 불가능
	  if (num_r_subarray < MINSUBARRAYROWS) return; // [한국어] 최소 행 수 미달 — 워드라인 드라이버 오버헤드 > 셀 면적이 됨
	  if (num_r_subarray == 0) return;              // [한국어] 행 수 0 — 분모가 너무 커 반올림 결과가 0인 경우
	  if (num_r_subarray > MAXSUBARRAYROWS) return; // [한국어] 최대 행 수 초과 — 워드라인 지연이 허용 범위 초과
	  if (num_c_subarray < MINSUBARRAYCOLS) return; // [한국어] 최소 열 수 미달
	  if (num_c_subarray > MAXSUBARRAYCOLS) return; // [한국어] 최대 열 수 초과 — 비트라인 지연이 허용 범위 초과

  }

  else
  {//either fully-asso or cam
	  if (pure_cam) // [한국어] 순수 CAM 배열: 태그만 있고 데이터 배열 없음
	  {
		  if (g_ip->specific_tag) // [한국어] 사용자 명시 태그 폭 — 8비트 단위로 올림
		  {
			  tagbits = int(ceil(g_ip->tag_w/8.0)*8); // [한국어] 바이트 정렬: 8비트 경계로 올림
		  }
		  else // [한국어] 자동 계산: 전체 주소 비트를 8비트 단위로 올림
		  {
			  tagbits = int(ceil((ADDRESS_BITS + EXTRA_TAG_BITS)/8.0)*8);
//			  cout<<"Pure CAM needs tag width to be specified"<<endl;
//			  exit(0);
		  }
		  //tagbits = (((tagbits + 3) >> 2) << 2);

		  // [한국어] CAM 태그 서브어레이 행 수: 총 엔트리 수 / (뱅크 × 태그바이트 × Ndbl)
		  tag_num_r_subarray = (int)ceil(capacity_per_die / (g_ip->nbanks*tagbits/8.0 * Ndbl));//TODO: error check input of tagbits and blocksize //TODO: for pure CAM, g_ip->block should be number of entries.
		  //tag_num_c_subarray = (int)(tagbits  + EPSILON);
		  tag_num_c_subarray = tagbits; // [한국어] CAM 태그 서브어레이 열 수 = tagbits (MUX 없음)
		  if (tag_num_r_subarray == 0) return;              // [한국어] 행 수 0 — 물리 불가
		  if (tag_num_r_subarray > MAXSUBARRAYROWS) return; // [한국어] 최대 행 수 초과
		  if (tag_num_c_subarray < MINSUBARRAYCOLS) return; // [한국어] 최소 열 수 미달
		  if (tag_num_c_subarray > MAXSUBARRAYCOLS) return; // [한국어] 최대 열 수 초과
		  num_r_subarray = tag_num_r_subarray; // [한국어] 공통 num_r_subarray에 CAM 태그 행 수 복사
	  }
	  else //fully associative
	  {
		  if (g_ip->specific_tag) // [한국어] FA 태그 폭 명시
		  {
			  tagbits = g_ip->tag_w; // [한국어] 사용자 지정 태그 비트 폭 사용
		  }
		  else // [한국어] FA 태그 폭 자동 계산: 전체 주소에서 페이지 오프셋 비트 제거
		  {
			  tagbits = ADDRESS_BITS + EXTRA_TAG_BITS - _log2(g_ip->block_sz);//TODO: should be the page_offset=log2(page size), but this info is not avail with CACTI, for McPAT this is no problem.
		  }
		  // [한국어] FA tagbits도 4비트 정렬
		  tagbits = (((tagbits + 3) >> 2) << 2);

		  // [한국어] FA 태그 서브어레이 행 수: 총 블록 수 / (뱅크 × Ndbl) — Nspd=1 고정이므로 제외
		  tag_num_r_subarray = (int)(capacity_per_die / (g_ip->nbanks*g_ip->block_sz * Ndbl));
		  // [한국어] FA 태그 서브어레이 열 수: tagbits × Nspd / Ndwl (Nspd=Ndwl=1 이므로 = tagbits)
		  tag_num_c_subarray = (int)ceil((tagbits * Nspd / Ndwl));// + EPSILON);
		  if (tag_num_r_subarray == 0) return;              // [한국어] 행 수 0 — 물리 불가
		  if (tag_num_r_subarray > MAXSUBARRAYROWS) return; // [한국어] 최대 행 수 초과
		  if (tag_num_c_subarray < MINSUBARRAYCOLS) return; // [한국어] 최소 열 수 미달
		  if (tag_num_c_subarray > MAXSUBARRAYCOLS) return; // [한국어] 최대 열 수 초과

		  // [한국어] FA 데이터 서브어레이: 태그와 동일한 행 수, 열 수는 블록 크기(비트)
		  data_num_r_subarray = tag_num_r_subarray; // [한국어] 데이터 서브어레이 행 수 = 태그와 동일
		  data_num_c_subarray = 8 * g_ip->block_sz; // [한국어] 데이터 서브어레이 열 수 = 블록 크기(비트)
		  if (data_num_r_subarray == 0) return;              // [한국어] 행 수 0 — 물리 불가
		  if (data_num_r_subarray > MAXSUBARRAYROWS) return; // [한국어] 최대 행 수 초과
		  if (data_num_c_subarray < MINSUBARRAYCOLS) return; // [한국어] 최소 열 수 미달
		  if (data_num_c_subarray > MAXSUBARRAYCOLS) return; // [한국어] 최대 열 수 초과
		  num_r_subarray = tag_num_r_subarray; // [한국어] 공통 num_r_subarray에 FA 태그 행 수 복사
	  }
  }

  // [한국어] 전체 서브어레이 수 = 수평 분할 × 수직 분할
  num_subarrays = Ndwl * Ndbl;
  //****************end of computation of row, col of an subarray

  // -----------------------------------------------------------------------
  // [한국어] 셀 물리 치수(cell.h, cell.w) 계산
  // 포트 수에 따라 각 포트당 추가 배선 피치가 필요하므로 기본 셀 치수에 더함
  // -----------------------------------------------------------------------
  // calculate wire parameters
  if (fully_assoc || pure_cam) // [한국어] FA/CAM: CAM 셀과 SRAM 셀(데이터) 치수를 별도 계산
  {
	  // [한국어] CAM 셀 높이: 기본 CAM 셀 높이 + RW/RD/WR 포트당 2배 피치 + 검색 포트 + 단독 RD 포트
	  cam_cell.h = g_tp.cam.b_h + 2 * wire_local.pitch * (g_ip->num_rw_ports-1 + g_ip->num_rd_ports + g_ip->num_wr_ports)
	  + 2 * wire_local.pitch*(g_ip->num_search_ports-1) + wire_local.pitch * g_ip->num_se_rd_ports;
	  // [한국어] CAM 셀 폭: 기본 CAM 셀 폭 + 포트 오버헤드 (높이와 동일한 공식)
	  cam_cell.w = g_tp.cam.b_w + 2 * wire_local.pitch * (g_ip->num_rw_ports-1 + g_ip->num_rd_ports + g_ip->num_wr_ports)
	  + 2 * wire_local.pitch*(g_ip->num_search_ports-1) + wire_local.pitch * g_ip->num_se_rd_ports;

	  // [한국어] FA 데이터부의 SRAM 셀 높이: 기본 SRAM 셀 높이 + WR/RW/RD 포트 + 검색 포트
	  cell.h = g_tp.sram.b_h + 2 * wire_local.pitch * (g_ip->num_wr_ports +g_ip->num_rw_ports-1 + g_ip->num_rd_ports)
	  + 2 * wire_local.pitch*(g_ip->num_search_ports-1);
	  // [한국어] FA 데이터부의 SRAM 셀 폭: RW/RD(비검색)/WR 포트 + 검색 전용 RD 포트 피치 추가
	  cell.w = g_tp.sram.b_w + 2 * wire_local.pitch * (g_ip->num_rw_ports -1 + (g_ip->num_rd_ports - g_ip->num_se_rd_ports)
			  + g_ip->num_wr_ports) + g_tp.wire_local.pitch * g_ip->num_se_rd_ports + 2 * wire_local.pitch*(g_ip->num_search_ports-1);
  }
  else // [한국어] 일반(non-FA/CAM) 배열: 태그/데이터/DRAM/SRAM 종류에 따라 셀 치수 분기
  {
	  if(is_tag) // [한국어] 태그 배열 SRAM 셀 치수
	  {
		  // [한국어] 태그 셀 높이: 기본 SRAM 높이 + RW/RD/WR 포트 오버헤드
		  cell.h = g_tp.sram.b_h + 2 * wire_local.pitch * (g_ip->num_rw_ports - 1 + g_ip->num_rd_ports +
				  g_ip->num_wr_ports);
		  // [한국어] 태그 셀 폭: RW/WR/(RD-검색RD) 포트 오버헤드 + 검색 전용 RD 포트 단일 피치
		  cell.w = g_tp.sram.b_w + 2 * wire_local.pitch * (g_ip->num_rw_ports - 1 + g_ip->num_wr_ports +
				  (g_ip->num_rd_ports - g_ip->num_se_rd_ports)) +
				  wire_local.pitch * g_ip->num_se_rd_ports;
	  }
	  else // [한국어] 데이터 배열: DRAM과 SRAM 치수가 다름
	  {
		  if (is_dram) // [한국어] DRAM 셀 치수: 포트 오버헤드 없음 (DRAM은 단일 포트)
		  {
			  cell.h = g_tp.dram.b_h; // [한국어] DRAM 기본 셀 높이 그대로 사용
			  cell.w = g_tp.dram.b_w; // [한국어] DRAM 기본 셀 폭 그대로 사용
		  }
		  else // [한국어] 데이터 배열 SRAM 셀 치수: WR/RW/RD 포트 오버헤드 포함
		  {
			  // [한국어] SRAM 데이터 셀 높이: 기본 높이 + WR/RW/RD 포트 피치 오버헤드
			  cell.h = g_tp.sram.b_h + 2 * wire_local.pitch * (g_ip->num_wr_ports +
					  g_ip->num_rw_ports - 1 + g_ip->num_rd_ports);
			  // [한국어] SRAM 데이터 셀 폭: RW/(RD-검색RD)/WR 포트 오버헤드 + 검색RD 단일 피치
			  cell.w = g_tp.sram.b_w + 2 * wire_local.pitch * (g_ip->num_rw_ports - 1 +
					  (g_ip->num_rd_ports - g_ip->num_se_rd_ports) +
					  g_ip->num_wr_ports) + g_tp.wire_local.pitch * g_ip->num_se_rd_ports;
		  }
	  }
  }

  // -----------------------------------------------------------------------
  // [한국어] 비트라인 정전용량(C_bl)과 감지 전압(V_b_sense) 계산
  // V_b_sense는 DRAM 스토리지 커패시터와 비트라인 커패시터의 전하 공유로 결정됨.
  // VBITSENSEMIN(= 80mV) 이하이면 감지증폭기가 동작 불가 → 조기 반환.
  // -----------------------------------------------------------------------
  double c_b_metal = cell.h * wire_local.C_per_um; // [한국어] 비트라인 단위 셀 높이당 금속 정전용량 [F/셀]
  double C_bl; // [한국어] 비트라인 전체 정전용량 [F] — num_r_subarray × 셀당 정전용량

  if (!(fully_assoc || pure_cam)) // [한국어] 일반 배열(non-FA/CAM): DRAM과 SRAM 분기
  {
	  if (is_dram) // [한국어] DRAM: 비트라인 MUX 차수=1, 전하 공유 공식으로 V_b_sense 계산
	  {
		  deg_bl_muxing = 1; // [한국어] DRAM은 비트라인 당 감지증폭기 1개 — MUX 차수 강제 1
		  if (ram_cell_tech_type == comm_dram) // [한국어] 상용 DRAM: 드레인 커패시턴스 무시 (금속선만)
		  {
			  C_bl  = num_r_subarray * c_b_metal; // [한국어] 비트라인 정전용량 = 행 수 × 셀당 금속 정전용량
			  // [한국어] 전하 공유: V_b_sense = (Vdd/2) × Ccell / (Ccell + Cbl)
			  V_b_sense = (g_tp.dram_cell_Vdd/2) * g_tp.dram_cell_C / (g_tp.dram_cell_C + C_bl);
			  if (V_b_sense < VBITSENSEMIN) // [한국어] 감지 전압이 최소 허용값(80mV) 미만이면 불가
			  {
				  return; // [한국어] 감지 전압 미달 — 이 파티셔닝은 물리적으로 불가능
			  }
			  V_b_sense = VBITSENSEMIN;  // in any case, we fix sense amp input signal to a constant value
			  // [한국어] 감지 전압을 최소값으로 고정: 이후 타이밍 계산의 보수적 기준점
			  dram_refresh_period = 64e-3; // [한국어] 상용 DRAM 리프레시 주기 고정: 64ms (JEDEC 표준)
		  }
		  else // [한국어] 저전력 DRAM(lp_dram): 드레인 커패시턴스 포함
		  {
			  // [한국어] 셀 접근 트랜지스터 드레인 정전용량 (인접 셀과 드레인 공유 → /2)
			  double Cbitrow_drain_cap = drain_C_(g_tp.dram.cell_a_w, NCH, 1, 0, cell.w, true, true) / 2.0;
			  // [한국어] 비트라인 총 정전용량 = 행 수 × (드레인 정전용량 + 금속 정전용량)
			  C_bl  = num_r_subarray * (Cbitrow_drain_cap + c_b_metal);
			  // [한국어] 전하 공유로 감지 전압 계산
			  V_b_sense = (g_tp.dram_cell_Vdd/2) * g_tp.dram_cell_C /(g_tp.dram_cell_C + C_bl);

			  if (V_b_sense < VBITSENSEMIN) // [한국어] 감지 전압 최소값 검사
			  {
				  return; //Sense amp input signal is smaller that minimum allowable sense amp input signal
			  }
			  V_b_sense = VBITSENSEMIN; // in any case, we fix sense amp input signal to a constant value
			  // [한국어] 감지 전압을 최소값으로 고정 (보수적 타이밍 기준)
			  //v_storage_worst = g_tp.dram_cell_Vdd / 2 - VBITSENSEMIN * (g_tp.dram_cell_C + C_bl) / g_tp.dram_cell_C;
			  //dram_refresh_period = 1.1 * g_tp.dram_cell_C * v_storage_worst / g_tp.dram_cell_I_off_worst_case_len_temp;
			  // [한국어] 리프레시 주기: 셀 커패시터가 VDD_STORAGE_LOSS_FRACTION_WORST 비율로 방전되는 시간
			  // = 0.9 × Ccell × 방전비율 × Vdd / 최악 누설전류
			  dram_refresh_period = 0.9 * g_tp.dram_cell_C * VDD_STORAGE_LOSS_FRACTION_WORST * g_tp.dram_cell_Vdd / g_tp.dram_cell_I_off_worst_case_len_temp;
		  }
	  }
	  else
	  { //SRAM
		  // [한국어] SRAM 감지 전압: Vdd의 5% 또는 VBITSENSEMIN 중 큰 값 (작은 스윙으로 빠른 감지)
		  V_b_sense = (0.05 * g_tp.sram_cell.Vdd > VBITSENSEMIN) ? 0.05 * g_tp.sram_cell.Vdd : VBITSENSEMIN;
		  deg_bl_muxing = Ndcm; // [한국어] SRAM 비트라인 MUX 차수 = 입력 파라미터 Ndcm 그대로
		  // "/ 2.0" below is due to the fact that two adjacent access transistors share drain
		  // contacts in a physical layout
		  // [한국어] 인접 셀의 접근 트랜지스터가 드레인 컨택을 공유하므로 정전용량 절반
		  double Cbitrow_drain_cap = drain_C_(g_tp.sram.cell_a_w, NCH, 1, 0, cell.w, false, true) / 2.0;
		  // [한국어] 비트라인 총 정전용량 = 행 수 × (드레인 + 금속)
		  C_bl = num_r_subarray * (Cbitrow_drain_cap + c_b_metal);
		  dram_refresh_period = 0; // [한국어] SRAM은 리프레시 불필요 — 0으로 설정
	  }
  }
  else // [한국어] FA/CAM 배열: CAM 셀 기준으로 비트라인 정전용량 계산
  {
	  c_b_metal = cam_cell.h * wire_local.C_per_um;//IBM and SUN design, SRAM array uses dummy cells to fill the blank space due to mismatch on CAM-RAM
	  // [한국어] CAM 셀 높이 기반 금속 정전용량으로 덮어씀 (IBM/SUN 설계: CAM-RAM 크기 불일치를 더미 셀로 채움)
	  V_b_sense = (0.05 * g_tp.sram_cell.Vdd > VBITSENSEMIN) ? 0.05 * g_tp.sram_cell.Vdd : VBITSENSEMIN;
	  // [한국어] FA/CAM 감지 전압: SRAM과 동일 공식
	  deg_bl_muxing = 1;//FA fix as 1
	  // [한국어] FA/CAM은 비트라인 MUX 차수 = 1 고정 (Ndcm=1 제약과 일치)
	  // "/ 2.0" below is due to the fact that two adjacent access transistors share drain
	  // contacts in a physical layout
	  // [한국어] CAM 셀 접근 트랜지스터 드레인 정전용량 (드레인 공유 → /2)
	  double Cbitrow_drain_cap = drain_C_(g_tp.cam.cell_a_w, NCH, 1, 0, cam_cell.w, false, true) / 2.0;//TODO: comment out these two lines
	  // [한국어] FA/CAM 비트라인 총 정전용량
	  C_bl = num_r_subarray * (Cbitrow_drain_cap + c_b_metal);
	  dram_refresh_period = 0; // [한국어] FA/CAM은 SRAM 기반이므로 리프레시 불필요
  }


  // do/di: data in/out, for fully associative they are the data width for normal read and write
  // so/si: search data in/out, for fully associative they are the data width for the search ops
  // for CAM, si=di, but so = matching address. do = data out = di (for normal read/write)
  // so/si needs broadcase while do/di do not
  // [한국어] 입출력 방향 정의:
  //   di/do: 일반 데이터 입력/출력 (FA에서는 일반 read/write 데이터 폭)
  //   si/so: 검색 데이터 입력/출력 (FA/CAM 전용; CAM은 si=di, so=매칭 주소)
  //   si/so는 브로드캐스트 배선 필요, di/do는 불필요

  // -----------------------------------------------------------------------
  // [한국어] mat 구성(num_mats_h_dir, num_mats_v_dir, num_mats) 및
  //          mat당 출력 비트 수(num_do_b_mat) 계산
  // mat = 4개의 서브어레이로 구성되는 물리 단위
  // -----------------------------------------------------------------------
  if (fully_assoc || pure_cam)
  {
	    switch (Ndbl) {
	      case (0): // [한국어] Ndbl=0은 비정상 입력 — 오류 출력 후 종료
	        cout <<  "   Invalid Ndbl \n"<<endl;
	        exit(0);
	        break;
	      case (1): // [한국어] Ndbl=1: 서브어레이 1개 → mat 1×1
	    	  num_mats_h_dir = 1;//one subarray per mat
	    	  num_mats_v_dir = 1;
	        break;
	      case (2): // [한국어] Ndbl=2: 서브어레이 2개 → mat 1×1 (FA/CAM은 수직 2개가 1 mat)
	    	  num_mats_h_dir = 1;//two subarrays per mat
	    	  num_mats_v_dir = 1;
	    	  break;
	      default: // [한국어] Ndbl>=4: sqrt(Ndbl/4)로 정사각형에 가까운 mat 배치 결정
	    	  num_mats_h_dir = int(floor(sqrt(Ndbl/4.0)));//4 subbarrys per mat
	    	  num_mats_v_dir = int(Ndbl/4.0 / num_mats_h_dir); // [한국어] 수직 방향 mat 수 = 전체/수평
	    }
	    num_mats = num_mats_h_dir * num_mats_v_dir; // [한국어] 전체 mat 수 = 수평 × 수직

	    if (fully_assoc) // [한국어] FA 배열: so=데이터 열 수, do=데이터+태그
	    {
	    	num_so_b_mat   = data_num_c_subarray; // [한국어] FA 검색 출력: 데이터 부분 열 수
	    	num_do_b_mat   = data_num_c_subarray + tagbits; // [한국어] FA 데이터 출력: 데이터+태그 합계
	    }
	    else // [한국어] CAM: so=매칭 행 주소 비트 수, do=태그 비트 수
	    {
	    	// [한국어] CAM 검색 출력: 매칭된 행 주소 비트 = log2(행 수) + log2(서브어레이 수)
	    	num_so_b_mat = int(ceil(log2(num_r_subarray)) + ceil(log2(num_subarrays)));//the address contains the matched data
	    	num_do_b_mat = tagbits; // [한국어] CAM 데이터 출력 = 태그 비트 수
	    }
  }
  else // [한국어] 일반 배열: Ndwl/Ndbl로 mat 수 결정
  {
	  num_mats_h_dir = MAX(Ndwl / 2, 1); // [한국어] 수평 mat 수 = Ndwl/2 (최소 1)
	  num_mats_v_dir = MAX(Ndbl / 2, 1); // [한국어] 수직 mat 수 = Ndbl/2 (최소 1)
	  num_mats       = num_mats_h_dir * num_mats_v_dir; // [한국어] 전체 mat 수
	  // [한국어] mat당 데이터 출력 비트: 서브어레이당 열 수 / (비트라인MUX × SA_MUX_L1 × SA_MUX_L2)
	  // mat당 서브어레이 수 = num_subarrays/num_mats
	  num_do_b_mat   = MAX((num_subarrays/num_mats) * num_c_subarray / (deg_bl_muxing * Ndsam_lev_1 * Ndsam_lev_2), 1);
  }

  // [한국어] 검사 4: 일반 배열에서 num_do_b_mat이 mat당 서브어레이 수보다 작으면 불가
  // (각 서브어레이가 최소 1비트 이상 출력해야 함)
  if (!(fully_assoc|| pure_cam) && (num_do_b_mat < (num_subarrays/num_mats)))
  {
	  return; // [한국어] mat당 출력 비트 < mat당 서브어레이 수 — 물리적으로 부적절
  }


  // -----------------------------------------------------------------------
  // [한국어] 서브뱅크 단위 입출력 비트 수(num_do_b_subbank) 계산
  // 서브뱅크 = 수평 방향으로 활성화되는 mat 그룹
  // deg_sa_mux_l1_non_assoc: 연관도를 제외한 실제 Ndsam_lev_1 차수
  // -----------------------------------------------------------------------
  int deg_sa_mux_l1_non_assoc;
  //TODO:the i/o for subbank is not necessary and should be removed.
  if (!(fully_assoc || pure_cam))
  {
	  if (!is_tag) // [한국어] 데이터 배열: 메인 메모리/캐시/패스트 액세스 모드에 따라 분기
	  {
		  if (is_main_mem == true) // [한국어] 주 메모리: 프리패치 너비 × 출력 폭
		  {
			  // [한국어] 서브뱅크 데이터 출력 = 내부 프리패치 폭 × 외부 출력 폭
			  num_do_b_subbank = g_ip->int_prefetch_w * g_ip->out_w;
			  deg_sa_mux_l1_non_assoc = Ndsam_lev_1; // [한국어] 주 메모리는 연관도 보정 없음
		  }
		  else // [한국어] 캐시: 패스트 액세스 여부에 따라 분기
		  {
			  if (g_ip->fast_access == true) // [한국어] 패스트 액세스: 연관도 × 출력 폭 (멀티웨이 동시 출력)
			  {
				  num_do_b_subbank = g_ip->out_w * g_ip->data_assoc; // [한국어] 모든 연관 웨이를 동시에 출력
				  deg_sa_mux_l1_non_assoc = Ndsam_lev_1; // [한국어] 패스트 액세스는 연관도 보정 없음
			  }
			  else // [한국어] 일반 액세스: 출력 폭만, Ndsam_lev_1을 연관도로 나눔
			  {

				  num_do_b_subbank = g_ip->out_w; // [한국어] 단일 출력 폭
				  deg_sa_mux_l1_non_assoc = Ndsam_lev_1 / g_ip->data_assoc;
				  // [한국어] 연관도만큼 SA MUX가 웨이 선택에 사용되므로 실제 열 MUX 차수 감소
				  if (deg_sa_mux_l1_non_assoc < 1) // [한국어] 연관도 보정 후 MUX 차수가 1 미만이면 불가
				  {
					  return; // [한국어] SA MUX 차수 부족 — 이 연관도/Ndsam 조합은 불가
				  }

			  }
		  }
	  }
	  else // [한국어] 태그 배열: 태그 비트 × 연관도
	  {
		  num_do_b_subbank = tagbits * g_ip->tag_assoc; // [한국어] 서브뱅크 태그 출력 = 태그폭 × 연관도
		  if (num_do_b_mat < tagbits) // [한국어] mat당 출력 비트가 태그 비트보다 작으면 불가
		  {
			  return; // [한국어] mat이 태그 1개도 출력할 수 없는 구성
		  }
		  deg_sa_mux_l1_non_assoc = Ndsam_lev_1; // [한국어] 태그 배열은 연관도 보정 없음
		  //num_do_b_mat = g_ip->tag_assoc / num_act_mats_hor_dir;
	  }
  }
  else // [한국어] FA/CAM: 검색·데이터 출력 폭 별도 계산
  {
	  if (fully_assoc) // [한국어] FA: so=블록 크기, do=블록+태그
	  {
		  num_so_b_subbank = 8 * g_ip->block_sz;//TODO:internal perfetch should be considered also for fa
		  // [한국어] FA 서브뱅크 검색 출력 = 블록 크기(비트)
		  num_do_b_subbank = num_so_b_subbank + tag_num_c_subarray;
		  // [한국어] FA 서브뱅크 데이터 출력 = 검색 출력 + 태그 열 수
	  }
	  else // [한국어] CAM: so=매칭 주소, do=태그 비트
	  {
		  // [한국어] CAM 서브뱅크 검색 출력 = 행 주소 + 서브어레이 선택 주소 (매칭 위치 식별)
		  num_so_b_subbank = int(ceil(log2(num_r_subarray)) + ceil(log2(num_subarrays)));//the address contains the matched data
		  num_do_b_subbank = tag_num_c_subarray; // [한국어] CAM 서브뱅크 데이터 출력 = 태그 열 수
	  }

	  deg_sa_mux_l1_non_assoc = 1; // [한국어] FA/CAM은 SA MUX 차수 = 1 (MUX 없음)
  }

  // [한국어] 비연관도 SA MUX L1 차수 저장 (이후 타이밍 계산에서 참조)
  deg_senseamp_muxing_non_associativity = deg_sa_mux_l1_non_assoc;

  // -----------------------------------------------------------------------
  // [한국어] 수평 방향 활성 mat 수(num_act_mats_hor_dir) 계산
  // 한 번의 접근에서 수평으로 동시에 활성화되는 mat 수
  // -----------------------------------------------------------------------
  if (fully_assoc || pure_cam) // [한국어] FA/CAM: 수평 활성 mat = 1 (직렬 방식)
  {
	  num_act_mats_hor_dir = 1; // [한국어] FA/CAM 일반 read/write는 수평 1개 mat만 활성화
	  num_act_mats_hor_dir_sl = num_mats_h_dir;//TODO: this is unnecessary, since search op, num_mats is used
	  // [한국어] FA/CAM 검색 연산은 모든 수평 mat을 동시에 활성화 (병렬 검색)
  }
  else // [한국어] 일반 배열: 서브뱅크 출력 비트 / mat당 출력 비트
  {
	  num_act_mats_hor_dir = num_do_b_subbank / num_do_b_mat;
	  // [한국어] 필요한 출력 비트를 mat당 출력으로 나눠 몇 개의 mat이 수평으로 활성화될지 결정
	  if (num_act_mats_hor_dir == 0) // [한국어] 활성 mat가 0이면 불가 (서브뱅크가 mat보다 너무 작음)
	  {
		  return; // [한국어] 수평 활성 mat 수 = 0 — 구성 불가
	  }
  }

  // -----------------------------------------------------------------------
  // [한국어] 태그 배열의 mat당 데이터 출력 비트 재계산 (연관도 기반)
  // -----------------------------------------------------------------------
  //compute num_do_mat for tag
  if (is_tag)
  {
	  if (!(fully_assoc || pure_cam))
	  {
		  // [한국어] 태그 mat당 출력 = 연관도 / 수평 활성 mat 수 (각 mat이 일부 웨이를 담당)
		  num_do_b_mat     = g_ip->tag_assoc / num_act_mats_hor_dir;
		  // [한국어] 서브뱅크 태그 출력 재계산 = 활성 mat 수 × mat당 출력
		  num_do_b_subbank = num_act_mats_hor_dir * num_do_b_mat;
	  }
  }

  // -----------------------------------------------------------------------
  // [한국어] 페이지 모드·주 메모리 페이지 크기 제약 검사
  // -----------------------------------------------------------------------
  if ((g_ip->is_cache == false && is_main_mem == true) || (PAGE_MODE == 1 && is_dram))
  {
	  // [한국어] 주 메모리 또는 페이지 모드 DRAM: 활성 mat × mat당 출력 × SA MUX = 페이지 크기 필요
	  if (num_act_mats_hor_dir * num_do_b_mat * Ndsam_lev_1 * Ndsam_lev_2 != (int)g_ip->page_sz_bits)
	  {
		  return; // [한국어] 페이지 크기 불일치 — 이 파티셔닝은 주 메모리 페이지 구성과 맞지 않음
	  }
  }

//  if (is_tag == false && g_ip->is_cache == true && !fully_assoc && !pure_cam && //TODO: TODO burst transfer should also apply to RAM arrays
  // [한국어] 주 메모리 데이터 배열: 버스트 전송 폭(출력 × 버스트 길이 × 연관도) 이상의 출력이 필요
  if (is_tag == false && g_ip->is_main_mem == true &&
		  num_act_mats_hor_dir*num_do_b_mat*Ndsam_lev_1*Ndsam_lev_2 < ((int) g_ip->out_w * (int) g_ip->burst_len * (int) g_ip->data_assoc))
  {
	  return; // [한국어] 버스트 전송에 필요한 최소 출력 비트 미달 — 조기 반환
  }

  // [한국어] 수평 활성 mat 수가 전체 수평 mat 수를 초과하면 불가 (존재하지 않는 mat에 접근 불가)
  if (num_act_mats_hor_dir > num_mats_h_dir)
  {
	  return; // [한국어] 활성 mat 수 > 총 수평 mat 수 — 물리적 불가
  }


  // -----------------------------------------------------------------------
  // [한국어] mat당 데이터 입력 비트 수(num_di_b_mat) 및 검색 입력(num_si_b_mat) 계산
  // -----------------------------------------------------------------------
  //compute di for mat subbank and bank
  if (!(fully_assoc ||pure_cam))
  {
	  if(!is_tag) // [한국어] 데이터 배열: 패스트 액세스 여부에 따라 입력 폭 결정
	  {
		  if(g_ip->fast_access == true) // [한국어] 패스트 액세스: 입력은 출력/연관도 (웨이 분리 불필요)
		  {
			  num_di_b_mat = num_do_b_mat / g_ip->data_assoc;
			  // [한국어] 패스트 액세스에서 입력은 한 웨이 분량만 (출력이 모든 웨이를 포함하므로 역산)
		  }
		  else // [한국어] 일반 액세스: 입력 = 출력 (1:1 대응)
		  {
			  num_di_b_mat = num_do_b_mat; // [한국어] 입력 비트 = 출력 비트
		  }
	  }
	  else // [한국어] 태그 배열: 입력 = tagbits (태그 하나씩 비교)
	  {
		  num_di_b_mat = tagbits; // [한국어] 태그 배열 mat 입력 = 태그 비트 폭
	  }
  }
  else // [한국어] FA/CAM: 일반 di와 검색 si 별도 계산
  {
	  if (fully_assoc) // [한국어] FA: di=do (일반 read/write), si=tagbits (검색 입력)
	  {
		  num_di_b_mat = num_do_b_mat;
		  //*num_subarrays/num_mats; bits per mat of CAM/FA is as same as cache,
		  //but inside the mat wire tracks need to be reserved for search data bus
		  num_si_b_mat = tagbits; // [한국어] FA 검색 입력 = 태그 비트 폭 (검색 키 너비)
	  }
	  else // [한국어] CAM: di=tagbits (일반 쓰기), si=tagbits (검색 입력)
	  {
		  num_di_b_mat = tagbits; // [한국어] CAM 일반 데이터 입력 = 태그 비트 폭
		  num_si_b_mat = tagbits;//*num_subarrays/num_mats;
		  // [한국어] CAM 검색 입력 = 태그 비트 폭 (si=di for CAM)
	  }

  }

  // [한국어] 서브뱅크 입력 비트 수: mat당 입력 × 수평 활성 mat 수
  num_di_b_subbank       = num_di_b_mat * num_act_mats_hor_dir;//normal cache or normal r/w for FA
  // [한국어] 검색 입력은 브로드캐스트이므로 mat 수와 무관하게 mat당 값 그대로 사용
  num_si_b_subbank       = num_si_b_mat; //* num_act_mats_hor_dir_sl; inside the data is broadcast

  // -----------------------------------------------------------------------
  // [한국어] 어드레스 비트 수 계산
  // num_addr_b_row_dec: 행 디코더에서 처리하는 주소 비트 수 = log2(서브어레이 행 수)
  // number_subbanks_decode: 서브뱅크 선택에 필요한 주소 비트 수
  // -----------------------------------------------------------------------
  int num_addr_b_row_dec     = _log2(num_r_subarray); // [한국어] 서브어레이 행 디코더 비트 수
  if  ((fully_assoc ||pure_cam))
	  num_addr_b_row_dec     +=_log2(num_subarrays/num_mats);
	  // [한국어] FA/CAM: 행 디코더에 서브어레이 선택 비트 추가 (mat 내 어느 서브어레이인지)
  int number_subbanks        = num_mats / num_act_mats_hor_dir; // [한국어] 총 서브뱅크 수 = mat 수 / 수평 활성 mat 수
  number_subbanks_decode = _log2(number_subbanks);//TODO: add log2(num_subarray_per_bank) to FA/CAM
  // [한국어] 서브뱅크 선택 디코더 비트 수 = log2(서브뱅크 수)

  // [한국어] 포트 수 복사: g_ip에서 DynamicParameter 멤버로 로컬 복사 (이후 타이밍 계산에서 참조)
  num_rw_ports = g_ip->num_rw_ports;       // [한국어] 읽기/쓰기 겸용 포트 수
  num_rd_ports = g_ip->num_rd_ports;       // [한국어] 읽기 전용 포트 수
  num_wr_ports = g_ip->num_wr_ports;       // [한국어] 쓰기 전용 포트 수
  num_se_rd_ports = g_ip->num_se_rd_ports; // [한국어] 단일 종단(single-ended) 읽기 포트 수
  num_search_ports = g_ip->num_search_ports; // [한국어] 검색(CAM/FA 전용) 포트 수

  // -----------------------------------------------------------------------
  // [한국어] mat당 어드레스 비트 수(number_addr_bits_mat) 계산
  // DRAM 주 메모리는 행/열 디코딩이 분리되어 MAX 연산, 나머지는 합산
  // -----------------------------------------------------------------------
  if (is_dram && is_main_mem) // [한국어] DRAM 주 메모리: 행 디코더와 열 디코더 중 더 큰 쪽 선택
  {
	  // [한국어] DRAM 주 메모리 어드레스 = MAX(행 디코더 비트, 열 MUX 비트 합계)
	  // 열 MUX 비트 = log2(BL_MUX) + log2(SA_MUX_L1) + log2(SA_MUX_L2)
	  number_addr_bits_mat = MAX((unsigned int) num_addr_b_row_dec,
			  _log2(deg_bl_muxing) + _log2(deg_sa_mux_l1_non_assoc) + _log2(Ndsam_lev_2));
  }
  else // [한국어] SRAM/캐시: 행 + 열 디코더 비트 합산 (RAS/CAS 분리 없음)
  {
	  number_addr_bits_mat = num_addr_b_row_dec + _log2(deg_bl_muxing) +
	  _log2(deg_sa_mux_l1_non_assoc) + _log2(Ndsam_lev_2);
	  // [한국어] 총 mat 어드레스 = 행 디코더 + BL_MUX + SA_MUX_L1(비연관) + SA_MUX_L2
  }

  // -----------------------------------------------------------------------
  // [한국어] 포트당 뱅크 레벨 입출력 비트 수 계산
  // -----------------------------------------------------------------------
  if (!(fully_assoc ||pure_cam)) // [한국어] 일반 배열: 태그/데이터 배열별 분기
  {
	  if (is_tag) // [한국어] 태그 배열: 입력=태그 비트, 출력=연관도(히트 웨이 인덱스)
	  {
		  num_di_b_bank_per_port = tagbits; // [한국어] 포트당 태그 입력 비트 = tagbits
		  num_do_b_bank_per_port = g_ip->data_assoc; // [한국어] 포트당 태그 출력 비트 = 연관도(웨이 선택 신호)
	  }
	  else // [한국어] 데이터 배열: 입력=출력폭+연관도(웨이 선택), 출력=출력폭
	  {
		  num_di_b_bank_per_port = g_ip->out_w + g_ip->data_assoc;
		  // [한국어] 데이터 입력 = 데이터 폭 + 웨이 선택 비트 (어느 웨이에 쓸지 지정)
		  num_do_b_bank_per_port = g_ip->out_w; // [한국어] 데이터 출력 = 출력 폭
	  }
  }
  else // [한국어] FA/CAM: 검색 포트와 일반 포트 입출력 모두 계산
  {
	  if (fully_assoc) // [한국어] FA: di/do=데이터+태그, si=태그, so=데이터
	  {
		  num_di_b_bank_per_port = g_ip->out_w + tagbits;//TODO: out_w or block_sz?
		  // [한국어] FA 포트당 입력 = 데이터 폭 + 태그 폭 (데이터+태그 동시 기록)
		  num_si_b_bank_per_port = tagbits; // [한국어] FA 검색 입력 = 태그 폭 (검색 키)
		  num_do_b_bank_per_port = g_ip->out_w + tagbits; // [한국어] FA 포트당 출력 = 데이터+태그
		  num_so_b_bank_per_port = g_ip->out_w; // [한국어] FA 검색 출력 = 데이터 폭 (히트된 데이터)
	  }
	  else // [한국어] CAM: di=si=태그, do=태그, so=매칭 위치 주소
	  {
		  num_di_b_bank_per_port = tagbits; // [한국어] CAM 포트당 데이터 입력 = 태그 비트
		  num_si_b_bank_per_port = tagbits; // [한국어] CAM 포트당 검색 입력 = 태그 비트 (si=di)
		  num_do_b_bank_per_port = tagbits; // [한국어] CAM 포트당 데이터 출력 = 태그 비트 (일반 읽기)
		  // [한국어] CAM 포트당 검색 출력 = 매칭 행·서브어레이 주소 비트 (so = 매칭 위치)
		  num_so_b_bank_per_port = int(ceil(log2(num_r_subarray)) + ceil(log2(num_subarrays)));
	  }
  }

  // -----------------------------------------------------------------------
  // [한국어] 웨이 선택 신호 수 계산
  // 다중 연관도(non-FA) 데이터 배열에서 일반 액세스 모드이면
  // 웨이 선택 신호가 H-tree를 통해 mat까지 내려가야 함
  // -----------------------------------------------------------------------
  if ((!is_tag) && (g_ip->data_assoc > 1) && (!g_ip->fast_access))
  {
	  // [한국어] mat당 웨이 선택 신호 수 = 데이터 연관도 (각 웨이에 대한 선택 신호 1개씩)
	  number_way_select_signals_mat = g_ip->data_assoc;
  }

  // -----------------------------------------------------------------------
  // [한국어] ECC 비트 조정: add_ecc_b_=true이면 모든 데이터 신호 폭에 ECC 오버헤드 추가
  // num_bits_per_ecc_b_ 비트마다 1비트의 ECC 패리티 비트를 추가
  // H-tree를 통과하는 모든 신호 폭(mat/subbank/bank 레벨)에 적용
  // -----------------------------------------------------------------------
  // add ECC adjustment to all data signals that traverse on H-trees.
  if (g_ip->add_ecc_b_ == true)
  {
	  // [한국어] mat 레벨 데이터 출력/입력에 ECC 비트 추가
	  num_do_b_mat += (int) (ceil(num_do_b_mat / num_bits_per_ecc_b_));
	  // [한국어] ceil(데이터 비트 / ECC 블록 크기) = 필요한 ECC 비트 수
	  num_di_b_mat += (int) (ceil(num_di_b_mat / num_bits_per_ecc_b_));
	  // [한국어] 서브뱅크 레벨 데이터 입출력에 ECC 비트 추가
	  num_di_b_subbank += (int) (ceil(num_di_b_subbank / num_bits_per_ecc_b_));
	  num_do_b_subbank += (int) (ceil(num_do_b_subbank / num_bits_per_ecc_b_));
	  // [한국어] 뱅크 포트 레벨 데이터 입출력에 ECC 비트 추가
	  num_di_b_bank_per_port += (int) (ceil(num_di_b_bank_per_port / num_bits_per_ecc_b_));
	  num_do_b_bank_per_port += (int) (ceil(num_do_b_bank_per_port / num_bits_per_ecc_b_));

	  // [한국어] FA/CAM 검색 신호(so/si)에도 ECC 비트 추가 (H-tree를 통과하는 신호이므로)
	  num_so_b_mat += (int) (ceil(num_so_b_mat / num_bits_per_ecc_b_));
	  num_si_b_mat += (int) (ceil(num_si_b_mat / num_bits_per_ecc_b_));
	  num_si_b_subbank += (int) (ceil(num_si_b_subbank / num_bits_per_ecc_b_));
	  num_so_b_subbank += (int) (ceil(num_so_b_subbank / num_bits_per_ecc_b_));
	  num_si_b_bank_per_port += (int) (ceil(num_si_b_bank_per_port / num_bits_per_ecc_b_));
	  num_so_b_bank_per_port += (int) (ceil(num_so_b_bank_per_port / num_bits_per_ecc_b_));
  }

  // [한국어] 모든 물리 제약 검사와 파생 파라미터 계산을 통과했으므로 유효한 구성으로 확정
  is_valid = true; // [한국어] is_valid=true: 이 (Ndwl,Ndbl,Nspd,Ndcm,Ndsam_lev_1,Ndsam_lev_2) 조합은 유효
}
