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
 * [한국어 설명] CACTI 기술 파라미터 및 동적 파라미터 헤더 (parameter.h)
 *
 * === 파일의 역할 ===
 * CACTI 캐시 모델링에 사용되는 두 가지 핵심 파라미터 클래스를 정의한다.
 * TechnologyParameter는 반도체 공정 노드에 종속적인 소자 특성(트랜지스터/배선/메모리 셀)을
 * 담으며, 전역 인스턴스 g_tp로 어디서나 참조된다.
 * DynamicParameter는 입력 캐시 설계 파라미터(Ndwl, Ndbl, Nspd 등)로부터 계산되는
 * 캐시 파티션 정보(서브어레이 수, mat 수 등)를 담아 Mat/Htree2 등에 전달된다.
 * AccelWattch는 이 파라미터들을 이용해 GPU 캐시의 전력·면적·지연을 공정 노드별로 추정한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CACTI 파라미터 흐름: gpgpusim.config → InputParameter(g_ip) → TechnologyParameter(g_tp) 초기화
 *                     → DynamicParameter 생성(최적 Ndwl/Ndbl 탐색 루프에서) → UCA/Mat/Htree2 에 전달
 * g_tp는 시뮬레이션 시작 시 한 번 초기화된 후 읽기 전용으로 전체 CACTI 코드에서 참조된다.
 * DynamicParameter는 각 탐색 후보마다 새로 생성되어 후보 결과 평가에 사용된다.
 * 실행 컨텍스트: 호스트 CPU 유저스페이스. 동기화 없이 단일 스레드에서만 접근.
 *
 * === 타 모듈과의 연결 ===
 * 의존: area.h (Area 클래스), const.h (기본 상수), cacti_interface.h (InputParameter), io.h (출력 함수)
 * 상위: 모든 CACTI 소스 파일이 parameter.h를 include하여 g_tp/g_ip에 접근
 * 하위: TechnologyParameter가 DeviceType/InterconnectType/MemoryType 내부 클래스를 포함
 * 데이터 흐름: g_ip(사용자 입력) → TechnologyParameter 초기화(tech_params.cc) →
 *              DynamicParameter 계산 → UCA/Mat/Htree2 파라미터로 전달
 *
 * === 주요 함수/구조체 요약 ===
 * TechnologyParameter          : 공정 파라미터 컨테이너 (g_tp 전역 인스턴스)
 * TechnologyParameter::DeviceType    : 트랜지스터 타입별 소자 파라미터 (Vdd, Vth, C_g, R_on 등)
 * TechnologyParameter::InterconnectType : 배선 층별 파라미터 (pitch, R/C per µm)
 * TechnologyParameter::MemoryType    : SRAM/DRAM/CAM 셀 기하 파라미터 (b_w, b_h, Vbitpre 등)
 * DynamicParameter             : 캐시 파티션 파라미터 (Ndwl/Ndbl/Nspd → num_mats, num_subarrays 등)
 */

#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#include "area.h"           // [한국어] Area 클래스 — 회로 면적(w, h) 표현
#include "const.h"          // [한국어] CACTI 공통 상수 (MIN_CELL_HEIGHT 등)
#include "cacti_interface.h" // [한국어] InputParameter 클래스, uca_org_t 등 공개 인터페이스
#include "io.h"             // [한국어] output_UCA(), output_data_csv() 선언

// parameters which are functions of certain device technology
/* [한국어] TechnologyParameter: 반도체 공정 노드에 종속적인 모든 소자 파라미터를 담는 컨테이너.
 * 시뮬레이션 시작 시 한 번 초기화(tech_params.cc)된 후 전역 인스턴스 g_tp로 참조된다.
 * DeviceType(트랜지스터), InterconnectType(배선), MemoryType(메모리 셀) 세 내부 클래스로 구성. */
class TechnologyParameter
{
 public:
  class DeviceType
  /* [한국어] 트랜지스터 타입별 소자 파라미터 클래스.
   * SRAM 셀, DRAM 접근 트랜지스터, DRAM 워드라인 트랜지스터, 주변부 글로벌 트랜지스터, CAM 셀 등
   * 각 용도에 맞는 Vdd/Vth/채널 저항/전류/커패시턴스 값을 보유한다.
   * TechnologyParameter의 sram_cell, dram_acc, dram_wl, peri_global, cam_cell 인스턴스로 사용. */
  {
   public:
    double C_g_ideal;
    /* [한국어] 이상적 게이트 커패시턴스 (C_oxide, F/m²) — Cox × W × L 에서 Cox에 해당.
     * 설정자: tech_params.cc에서 공정 노드별 룩업 테이블에서 초기화.
     * 읽는 자: basic_circuit.cc의 gate_C() 함수에서 게이트 입력 커패시턴스 계산 시 사용.
     * 값 범위: 수십~수백 fF (공정 노드에 따라 다름). 동기화: g_tp 초기화 후 읽기 전용. */

    double C_fringe;
    /* [한국어] 게이트 프린지(fringe) 커패시턴스 (F/m) — 게이트 측면에서 드레인/소스로의 기생 커패시턴스.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: gate_C()에서 C_g_ideal에 더해 실제 게이트 커패시턴스 산출.
     * 값 범위: C_g_ideal의 10~30% 수준. 동기화: 읽기 전용. */

    double C_overlap;
    /* [한국어] 게이트-드레인/소스 오버랩 커패시턴스 (F/m) — 게이트가 소스/드레인 영역과 겹치는 부분.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: drain_C_() 함수에서 드레인 커패시턴스 총합 계산 시 추가.
     * 동기화: 읽기 전용. */

    double C_junc;  // C_junc_area
    /* [한국어] PN 접합 커패시턴스 (면적 성분, F/m²) — 드레인/소스 PN 접합의 면적당 커패시턴스.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: drain_C_()에서 드레인 접합 커패시턴스 = C_junc × 면적으로 계산.
     * 동기화: 읽기 전용. */

    double C_junc_sidewall;
    /* [한국어] PN 접합 사이드월 커패시턴스 (F/m) — 드레인/소스 PN 접합의 둘레당 커패시턴스.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: drain_C_()에서 사이드월 성분으로 추가.
     * 동기화: 읽기 전용. */

    double l_phy;
    /* [한국어] 물리적 채널 길이 (Physical gate length, 미터) — 실제 제조된 게이트 길이.
     * 설정자: tech_params.cc 초기화 (예: 32nm 노드이면 약 32nm).
     * 읽는 자: tr_R_on() 등 온-저항 계산 시 채널 길이 보정에 사용.
     * 동기화: 읽기 전용. */

    double l_elec;
    /* [한국어] 전기적 유효 채널 길이 (Electrical gate length, 미터) — 핀치오프 후 실효 채널 길이.
     * l_phy보다 작으며 단채널 효과를 반영한다.
     * 설정자: tech_params.cc 초기화. 읽는 자: 일부 회로 계산 함수.
     * 동기화: 읽기 전용. */

    double R_nch_on;
    /* [한국어] NMOS 트랜지스터의 온-저항 (최소 폭 기준, Ω) — 도통 상태에서의 채널 저항.
     * 설정자: tech_params.cc에서 공정 파라미터로 계산하여 초기화.
     * 읽는 자: tr_R_on(w, NCH, nf) 함수에서 트랜지스터 폭/핑거 수로 스케일링하여 사용.
     * 값 범위: 수 kΩ ~ 수십 kΩ (최소 폭 기준). 동기화: 읽기 전용. */

    double R_pch_on;
    /* [한국어] PMOS 트랜지스터의 온-저항 (최소 폭 기준, Ω).
     * PMOS는 NMOS보다 이동도가 낮아 R_pch_on > R_nch_on (보통 2~3배).
     * 설정자: tech_params.cc 초기화. 읽는 자: tr_R_on(w, PCH, nf).
     * 동기화: 읽기 전용. */

    double Vdd;
    /* [한국어] 전원 전압 (V) — 이 디바이스 타입의 공칭 동작 전압.
     * 설정자: tech_params.cc에서 공정 노드별 테이블에서 초기화 (예: 45nm ≈ 1.0V).
     * 읽는 자: 동적 전력 계산(0.5*C*Vdd²), horowitz 모델의 Vth/Vdd 비율 계산에 사용.
     * 값 범위: 0.7V ~ 1.2V (최신 노드일수록 낮음). 동기화: 읽기 전용. */

    double Vth;
    /* [한국어] 문턱 전압 (Threshold Voltage, V) — 트랜지스터가 도통 상태가 되는 게이트 전압.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: horowitz() 모델에서 Vth/Vdd 비율로 논리 전환 임계점 계산.
     *          누설 전류(I_off) 계산에도 Vth가 간접적으로 영향.
     * 값 범위: 0.2V ~ 0.5V. 동기화: 읽기 전용. */

    double I_on_n;
    /* [한국어] NMOS 온-전류 (최소 폭 × Vdd 기준, A/m) — 도통 상태에서 채널을 흐르는 전류.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: 비트라인 방전 시간, SA 지연 등 전류 구동 기반 지연 계산에 사용.
     * 동기화: 읽기 전용. */

    double I_on_p;
    /* [한국어] PMOS 온-전류 (최소 폭 × Vdd 기준, A/m).
     * I_on_p < I_on_n (이동도 차이). 설정자: tech_params.cc 초기화.
     * 읽는 자: PMOS 구동 회로의 전류 기반 지연 계산.
     * 동기화: 읽기 전용. */

    double I_off_n;
    /* [한국어] NMOS 서브스레숄드 누설 전류 (A/m) — 게이트 전압 0V에서 흐르는 누설 전류.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: cmos_Isub_leakage()에서 서브스레숄드 누설 전력 계산.
     * 값 범위: 공정 노드에 따라 수 pA/m ~ 수 nA/m. 동기화: 읽기 전용. */

    double I_off_p;
    /* [한국어] PMOS 서브스레숄드 누설 전류 (A/m).
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: cmos_Isub_leakage()에서 PMOS 누설 기여분 계산.
     * 동기화: 읽기 전용. */

    double I_g_on_n;
    /* [한국어] NMOS 게이트 터널링 전류 (도통 상태 기준, A/m) — 게이트 산화막이 얇아질수록 증가.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: cmos_Ig_leakage()에서 게이트 누설 전력 계산.
     * 동기화: 읽기 전용. */

    double I_g_on_p;
    /* [한국어] PMOS 게이트 터널링 전류 (A/m).
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: cmos_Ig_leakage()에서 PMOS 게이트 누설 기여분 계산.
     * 동기화: 읽기 전용. */

    double C_ox;
    /* [한국어] 게이트 산화막 커패시턴스 (F/m²) — Cox = ε_ox / t_ox.
     * 설정자: tech_params.cc에서 t_ox로부터 계산.
     * 읽는 자: gate_C()에서 C_g_ideal = C_ox × W × L 계산.
     * 동기화: 읽기 전용. */

    double t_ox;
    /* [한국어] 게이트 산화막 두께 (m) — 공정 노드가 작아질수록 얇아짐 (EOT 기준).
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: C_ox = ε_ox / t_ox 계산에 사용.
     * 값 범위: 수 nm ~ 수십 nm. 동기화: 읽기 전용. */

    double n_to_p_eff_curr_drv_ratio;
    /* [한국어] NMOS 대비 PMOS의 유효 전류 구동 비율 (= I_on_p / I_on_n 근사).
     * PMOS의 이동도가 낮아 같은 전류를 내려면 폭이 이 비율만큼 커야 한다.
     * 설정자: tech_params.cc에서 I_on_n / I_on_p 로 계산하여 초기화 (보통 2~3).
     * 읽는 자: Htree2에서 min_w_pmos = ratio × min_w_nmos 계산,
     *          input_nand()/output_buffer()에서 PMOS 크기 결정에 사용.
     * 동기화: 읽기 전용. */

    double long_channel_leakage_reduction;
    /* [한국어] 장채널 소자 누설 전류 저감 계수 (dimensionless).
     * 단채널 효과(DIBL, Vth roll-off)를 보정하기 위해 누설 전류에 곱하는 팩터.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: cmos_Isub_leakage()에서 누설 전류 보정 시 사용.
     * 동기화: 읽기 전용. */

    DeviceType(): C_g_ideal(0), C_fringe(0), C_overlap(0), C_junc(0),
                  C_junc_sidewall(0), l_phy(0), l_elec(0), R_nch_on(0), R_pch_on(0),
                  Vdd(0), Vth(0),
                  I_on_n(0), I_on_p(0), I_off_n(0), I_off_p(0),I_g_on_n(0),I_g_on_p(0),
                  C_ox(0), t_ox(0), n_to_p_eff_curr_drv_ratio(0), long_channel_leakage_reduction(0) { };
    /* [한국어] DeviceType 기본 생성자 — 모든 소자 파라미터를 0으로 초기화.
     * tech_params.cc에서 실제 값으로 설정하기 전의 초기 상태. */

    /*
     * [한국어]
     * DeviceType::reset - 모든 소자 파라미터를 0으로 리셋
     *
     * @return: (void) — 모든 필드를 0으로 초기화
     *
     * TechnologyParameter::reset()에서 각 DeviceType 멤버를 초기화할 때 호출된다.
     * 새로운 공정 파라미터를 로드하기 전 이전 값을 제거하는 용도.
     */
    void reset()
    {
      C_g_ideal = 0; // [한국어] 이상적 게이트 커패시턴스 초기화
      C_fringe  = 0; // [한국어] 프린지 커패시턴스 초기화
      C_overlap = 0; // [한국어] 오버랩 커패시턴스 초기화
      C_junc    = 0; // [한국어] PN 접합 면적 커패시턴스 초기화
      l_phy     = 0; // [한국어] 물리적 채널 길이 초기화
      l_elec    = 0; // [한국어] 전기적 유효 채널 길이 초기화
      R_nch_on  = 0; // [한국어] NMOS 온-저항 초기화
      R_pch_on  = 0; // [한국어] PMOS 온-저항 초기화
      Vdd       = 0; // [한국어] 전원 전압 초기화
      Vth       = 0; // [한국어] 문턱 전압 초기화
      I_on_n    = 0; // [한국어] NMOS 온-전류 초기화
      I_on_p    = 0; // [한국어] PMOS 온-전류 초기화
      I_off_n   = 0; // [한국어] NMOS 누설 전류 초기화
      I_off_p   = 0; // [한국어] PMOS 누설 전류 초기화
      I_g_on_n   = 0; // [한국어] NMOS 게이트 터널링 전류 초기화
      I_g_on_p   = 0; // [한국어] PMOS 게이트 터널링 전류 초기화
      C_ox      = 0; // [한국어] 산화막 커패시턴스 초기화
      t_ox      = 0; // [한국어] 산화막 두께 초기화
      n_to_p_eff_curr_drv_ratio = 0; // [한국어] NMOS/PMOS 전류 구동비 초기화
      long_channel_leakage_reduction = 0; // [한국어] 장채널 누설 저감 계수 초기화
    }

    void display(uint32_t indent = 0); // [한국어] 디바이스 타입 파라미터 출력 (디버그용)
  };

  class InterconnectType
  /* [한국어] 배선 층(interconnect layer)별 RC 파라미터 클래스.
   * wire_local(셀 내부), wire_inside_mat(mat 내부), wire_outside_mat(mat 간 글로벌) 세 인스턴스로 사용.
   * 각 배선 층마다 다른 pitch, 저항/커패시턴스 값을 가지므로 독립적으로 모델링한다. */
  {
   public:
    double pitch;
    /* [한국어] 배선 피치 (인접 배선 중심 간 거리, 미터) — 배선 면적과 버스 총 폭 계산에 사용.
     * 설정자: tech_params.cc에서 공정 노드별 배선 층 파라미터로 초기화.
     * 읽는 자: in_htree()/out_htree()에서 버스 배선이 차지하는 면적 보정(× bit수 × pitch).
     * 값 범위: 수십 nm ~ 수 µm (배선 층에 따라 다름).
     * 동기화: g_tp 초기화 후 읽기 전용. */

    double R_per_um;
    /* [한국어] 단위 길이당 배선 저항 (Ω/µm) — 배선 저항 = R_per_um × 길이.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: Wire 클래스에서 배선 RC 지연 계산 시 R = R_per_um × length(µm).
     * 값 범위: 수십 mΩ/µm ~ 수 Ω/µm. 동기화: 읽기 전용. */

    double C_per_um;
    /* [한국어] 단위 길이당 배선 커패시턴스 (F/µm) — 배선 커패시턴스 = C_per_um × 길이.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: Wire::wire_cap()에서 배선 총 커패시턴스 산출.
     * 값 범위: 수백 aF/µm ~ 수 fF/µm. 동기화: 읽기 전용. */

    double horiz_dielectric_constant;
    /* [한국어] 수평 방향(inter-wire) 유전체 상수 (dimensionless) — 인접 배선 간 커패시턴스 계산용.
     * 설정자: tech_params.cc 초기화. 읽는 자: C_per_um 정교화 계산(Wire 내부).
     * 동기화: 읽기 전용. */

    double vert_dielectric_constant;
    /* [한국어] 수직 방향(wire-to-substrate) 유전체 상수 — 배선-기판 간 커패시턴스 계산용.
     * 설정자: tech_params.cc 초기화. 동기화: 읽기 전용. */

    double aspect_ratio;
    /* [한국어] 배선 단면 종횡비 (높이/폭) — 배선 저항과 커패시턴스 모델링에 사용.
     * 높은 aspect_ratio는 저항을 줄이고 커패시턴스를 증가시킨다.
     * 설정자: tech_params.cc 초기화. 동기화: 읽기 전용. */

    double miller_value;
    /* [한국어] 밀러(Miller) 커패시턴스 계수 — 신호 전환 시 인접 배선 커패시턴스 증폭 효과.
     * 동시에 반대 방향으로 스위칭하는 인접 배선이 있으면 실효 커패시턴스가 2배가 된다.
     * 설정자: tech_params.cc 초기화. 읽는 자: Wire에서 C_per_um 보정.
     * 동기화: 읽기 전용. */

    double ild_thickness;
    /* [한국어] 층간 유전체(Inter-Layer Dielectric) 두께 (미터) — 배선 층 간 절연체 두께.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: 배선 층 간 커패시턴스 계산 시 사용.
     * 동기화: 읽기 전용. */

    InterconnectType(): pitch(0), R_per_um(0), C_per_um(0) { };
    /* [한국어] InterconnectType 기본 생성자 — 핵심 RC 파라미터를 0으로 초기화. */

    /*
     * [한국어]
     * InterconnectType::reset - 배선 파라미터 전체 초기화
     *
     * @return: (void) — 모든 필드를 0으로 리셋
     *
     * TechnologyParameter::reset()에서 wire_local/inside/outside 각각 호출.
     */
    void reset()
    {
      pitch = 0;                      // [한국어] 배선 피치 초기화
      R_per_um = 0;                   // [한국어] 단위 길이당 저항 초기화
      C_per_um = 0;                   // [한국어] 단위 길이당 커패시턴스 초기화
      horiz_dielectric_constant = 0;  // [한국어] 수평 유전체 상수 초기화
      vert_dielectric_constant = 0;   // [한국어] 수직 유전체 상수 초기화
      aspect_ratio = 0;               // [한국어] 단면 종횡비 초기화
      miller_value = 0;               // [한국어] 밀러 계수 초기화
      ild_thickness = 0;              // [한국어] 층간 유전체 두께 초기화
    }

    void display(uint32_t indent = 0); // [한국어] 배선 파라미터 출력 (디버그용)
  };

  class MemoryType
  /* [한국어] 메모리 셀(SRAM/DRAM/CAM) 기하 파라미터 클래스.
   * 셀 1개의 폭/높이, 셀 내 트랜지스터 폭, 비트라인 프리차지 전압을 담는다.
   * TechnologyParameter의 sram, dram, cam 인스턴스로 사용. */
  {
   public:
    double b_w;
    /* [한국어] 비트셀 폭 (bit-cell width, 미터) — 1개 메모리 셀의 레이아웃 가로 크기.
     * 설정자: tech_params.cc에서 공정 노드별 셀 라이브러리 파라미터로 초기화.
     * 읽는 자: Subarray에서 비트라인 길이 = num_r_subarray × b_h 계산 등에 사용.
     * 값 범위: 수백 nm ~ 수 µm. 동기화: 읽기 전용. */

    double b_h;
    /* [한국어] 비트셀 높이 (bit-cell height, 미터) — 1개 메모리 셀의 레이아웃 세로 크기.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: Subarray에서 비트라인 총 길이 = num_r_subarray × b_h 계산.
     * 동기화: 읽기 전용. */

    double cell_a_w;
    /* [한국어] 셀 내 접근 트랜지스터(access transistor) 폭 (미터).
     * SRAM에서는 패스 게이트, DRAM에서는 접근 NMOS 트랜지스터의 물리적 폭.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: Subarray::Subarray()에서 비트라인 커패시턴스 계산 시 사용.
     * 동기화: 읽기 전용. */

    double cell_pmos_w;
    /* [한국어] SRAM 셀 내 PMOS 풀업 트랜지스터 폭 (미터).
     * 6T SRAM 셀의 2개 PMOS 풀업 트랜지스터 각각의 폭.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: SRAM 셀 전류 계산 시 사용.
     * 동기화: 읽기 전용. */

    double cell_nmos_w;
    /* [한국어] SRAM 셀 내 NMOS 풀다운 트랜지스터 폭 (미터).
     * 6T SRAM 셀의 2개 NMOS 풀다운(드라이버) 트랜지스터 각각의 폭.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: SRAM 셀 읽기 전류, 비트라인 방전 시간 계산.
     * 동기화: 읽기 전용. */

    double Vbitpre;
    /* [한국어] 비트라인 프리차지 전압 (V) — 읽기/쓰기 전 비트라인을 충전하는 전압.
     * SRAM은 보통 Vdd로, DRAM은 Vdd/2로 프리차지한다.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: compute_bitline_delay()에서 비트라인 전압 스윙 = Vbitpre - Vdd/2 계산.
     *          compute_power_energy()에서 비트라인 스위칭 에너지 = 0.5*C_bl*(Vbitpre²-Vsen²).
     * 동기화: 읽기 전용. */

    /*
     * [한국어]
     * MemoryType::reset - 메모리 셀 파라미터 전체 초기화
     *
     * @return: (void) — 모든 필드를 0으로 리셋
     */
    void reset()
    {
      b_w = 0;         // [한국어] 비트셀 폭 초기화
      b_h = 0;         // [한국어] 비트셀 높이 초기화
      cell_a_w = 0;    // [한국어] 접근 트랜지스터 폭 초기화
      cell_pmos_w = 0; // [한국어] PMOS 풀업 폭 초기화
      cell_nmos_w = 0; // [한국어] NMOS 풀다운 폭 초기화
      Vbitpre = 0;     // [한국어] 프리차지 전압 초기화
    }

    void display(uint32_t indent = 0); // [한국어] 메모리 셀 파라미터 출력 (디버그용)
  };

  class ScalingFactor
  /* [한국어] 기술 스케일링 보정 계수 클래스.
   * 공정 노드 변화에 따른 로직 스케일링, 트랜지스터 밀도, 누설 저감 계수를 담는다.
   * AccelWattch에서 다른 공정 노드로 결과를 외삽할 때 사용. */
  {
   public:
    double logic_scaling_co_eff;
    /* [한국어] 로직 스케일링 계수 (dimensionless) — 공정 노드 변화에 따른 면적/전력 스케일링 배수.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: McPAT 등 전력 스케일링 계산에 사용.
     * 동기화: 읽기 전용. */

    double core_tx_density;
    /* [한국어] 코어 트랜지스터 밀도 (개/mm²) — 단위 면적당 평균 트랜지스터 수.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: 코어 전력 밀도 추정. 동기화: 읽기 전용. */

    double long_channel_leakage_reduction;
    /* [한국어] 장채널 누설 저감 계수 — 단채널 소자의 누설 대비 장채널 소자의 누설 비율.
     * 설정자: tech_params.cc 초기화.
     * 읽는 자: 누설 전력 보정 계산. 동기화: 읽기 전용. */

    ScalingFactor(): logic_scaling_co_eff(0), core_tx_density(0),
    long_channel_leakage_reduction(0) { }; // [한국어] 기본 생성자 — 모든 계수를 0으로 초기화

    /*
     * [한국어]
     * ScalingFactor::reset - 스케일링 계수 전체 초기화
     */
    void reset()
    {
      logic_scaling_co_eff= 0;         // [한국어] 로직 스케일링 계수 초기화
      core_tx_density = 0;             // [한국어] 트랜지스터 밀도 초기화
      long_channel_leakage_reduction= 0; // [한국어] 누설 저감 계수 초기화
    }

    void display(uint32_t indent = 0); // [한국어] 스케일링 계수 출력 (디버그용)
  };

  double ram_wl_stitching_overhead_;
  /* [한국어] RAM 워드라인 스티칭 오버헤드 (dimensionless 또는 비율).
   * 긴 워드라인을 여러 세그먼트로 분할하여 연결할 때 추가되는 지연/면적 오버헤드.
   * 설정자: tech_params.cc 초기화. 읽는 자: 워드라인 드라이버 크기 계산.
   * 동기화: 읽기 전용. */

  double min_w_nmos_;
  /* [한국어] NMOS 최소 트랜지스터 폭 (미터) — 공정이 허용하는 가장 좁은 NMOS 폭.
   * 설정자: tech_params.cc에서 2 × l_phy 등으로 계산하여 초기화.
   * 읽는 자: Htree2에서 min_w_nmos = g_tp.min_w_nmos_로 복사하여 사용.
   *          모든 트랜지스터 크기 계산의 기준 단위가 된다.
   * 값 범위: 수십 nm ~ 수백 nm. 동기화: 읽기 전용. */

  double max_w_nmos_;
  /* [한국어] NMOS 최대 트랜지스터 폭 (미터) — 디코더 등 드라이버 크기 상한.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: Decoder 설계 시 드라이버 최대 폭 제한.
   * 동기화: 읽기 전용. */

  double max_w_nmos_dec;
  /* [한국어] 디코더용 NMOS 최대 폭 (미터) — 디코더 전용 최대 트랜지스터 폭 제한.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: Decoder 클래스에서 드라이버 크기 상한 설정.
   * 동기화: 읽기 전용. */

  double unit_len_wire_del;
  /* [한국어] 단위 길이 배선 지연 (초/미터) — 리피터 없는 배선의 단위 지연 참조값.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: Wire 클래스에서 배선 지연 빠른 추정에 사용.
   * 동기화: 읽기 전용. */

  double FO4;
  /* [한국어] FO4(Fanout-of-4) 인버터 지연 (초) — 공정 속도 기준이 되는 대표 게이트 지연.
   * 입력 커패시턴스의 4배 부하를 구동하는 최소 인버터의 지연으로, 공정 성능 비교에 사용.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 게이트 지연 정규화 계산. 동기화: 읽기 전용. */

  double kinv;
  /* [한국어] 인버터 지연 상수 (초) — 최소 크기 인버터의 고유 지연.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 회로 블록 지연 추정 시 기준 게이트 지연으로 사용.
   * 동기화: 읽기 전용. */

  double vpp;
  /* [한국어] 프로그램 전압 (DRAM 워드라인 부스트 전압, V).
   * DRAM 워드라인을 충분히 구동하기 위해 Vdd보다 높은 전압을 사용.
   * 설정자: tech_params.cc 초기화 (예: 2.5V ~ 3.3V).
   * 읽는 자: DRAM 워드라인 드라이버 전력 계산.
   * 동기화: 읽기 전용. */

  double w_sense_en;
  /* [한국어] 센스앰프 인에이블 트랜지스터 폭 (미터) — SA 활성화 회로의 NMOS 폭.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: compute_sa_delay()에서 SA 지연 계산.
   * 동기화: 읽기 전용. */

  double w_sense_n;
  /* [한국어] 센스앰프 NMOS 래치 트랜지스터 폭 (미터).
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: SA 지연·전력 계산. 동기화: 읽기 전용. */

  double w_sense_p;
  /* [한국어] 센스앰프 PMOS 래치 트랜지스터 폭 (미터).
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: SA 지연·전력 계산. 동기화: 읽기 전용. */

  double sense_delay;
  /* [한국어] 센스앰프 고정 지연 값 (초) — 기술 파라미터 기반으로 미리 계산된 SA 지연.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: compute_sa_delay()에서 SA 최종 지연 계산 시 참조.
   * 동기화: 읽기 전용. */

  double sense_dy_power;
  /* [한국어] 센스앰프 동적 전력 (J/access) — SA 1회 활성화당 소비 에너지.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: compute_power_energy()에서 power_sa 계산.
   * 동기화: 읽기 전용. */

  double w_iso;
  /* [한국어] 비트라인 격리 트랜지스터(isolation transistor) 폭 (미터).
   * 멀티-뱅크 구조에서 비활성 뱅크의 비트라인을 SA에서 분리하는 NMOS 스위치.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 비트라인 커패시턴스 및 격리 회로 면적 계산.
   * 동기화: 읽기 전용. */

  double w_poly_contact;
  /* [한국어] 폴리실리콘 컨택 폭 (미터) — 게이트 폴리와 메탈 컨택 크기.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 레이아웃 면적 계산. 동기화: 읽기 전용. */

  double spacing_poly_to_poly;
  /* [한국어] 인접 폴리실리콘 배선 간 최소 간격 (미터) — DRC 규칙 기반.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 셀 레이아웃 면적 산출. 동기화: 읽기 전용. */

  double spacing_poly_to_contact;
  /* [한국어] 폴리와 컨택 간 최소 간격 (미터).
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 셀 높이/폭 계산. 동기화: 읽기 전용. */

  double w_comp_inv_p1;
  /* [한국어] 비교기 인버터 PMOS 1단 폭 (미터) — FA 캐시 태그 비교기 회로용.
   * 설정자: tech_params.cc 초기화. 읽는 자: compute_comparator_delay().
   * 동기화: 읽기 전용. */

  double w_comp_inv_p2;
  /* [한국어] 비교기 인버터 PMOS 2단 폭 (미터). 설정자: tech_params.cc. 동기화: 읽기 전용. */

  double w_comp_inv_p3;
  /* [한국어] 비교기 인버터 PMOS 3단 폭 (미터). 설정자: tech_params.cc. 동기화: 읽기 전용. */

  double w_comp_inv_n1;
  /* [한국어] 비교기 인버터 NMOS 1단 폭 (미터). 설정자: tech_params.cc. 동기화: 읽기 전용. */

  double w_comp_inv_n2;
  /* [한국어] 비교기 인버터 NMOS 2단 폭 (미터). 설정자: tech_params.cc. 동기화: 읽기 전용. */

  double w_comp_inv_n3;
  /* [한국어] 비교기 인버터 NMOS 3단 폭 (미터). 설정자: tech_params.cc. 동기화: 읽기 전용. */

  double w_eval_inv_p;
  /* [한국어] 평가 인버터(evaluate inverter) PMOS 폭 (미터) — 비교기 평가 단계용.
   * 설정자: tech_params.cc. 읽는 자: compute_comparator_delay().
   * 동기화: 읽기 전용. */

  double w_eval_inv_n;
  /* [한국어] 평가 인버터 NMOS 폭 (미터). 설정자: tech_params.cc. 동기화: 읽기 전용. */

  double w_comp_n;
  /* [한국어] 비교기 NMOS 풀다운 트랜지스터 폭 (미터) — XOR/XNOR 기반 비교기의 NMOS.
   * 설정자: tech_params.cc. 읽는 자: compute_comparator_delay(). 동기화: 읽기 전용. */

  double w_comp_p;
  /* [한국어] 비교기 PMOS 풀업 트랜지스터 폭 (미터).
   * 설정자: tech_params.cc. 동기화: 읽기 전용. */

  double dram_cell_I_on;
  /* [한국어] DRAM 셀 접근 트랜지스터의 온-전류 (A) — DRAM 비트라인 충전 전류.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: DRAM 비트라인 지연 계산 (t_delay = C_bl × ΔV / I_on).
   * 동기화: 읽기 전용. */

  double dram_cell_Vdd;
  /* [한국어] DRAM 셀 전원 전압 (V) — DRAM 캐패시터 충전 전압 기준.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: DRAM 비트라인 전압 스윙, 셀 전하량 계산.
   * 동기화: 읽기 전용. */

  double dram_cell_I_off_worst_case_len_temp;
  /* [한국어] DRAM 셀 누설 전류 최악 조건 값 (A) — 최고 온도/최장 채널 길이 조건에서의 누설.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: DRAM 리프레시 간격 계산.
   * 동기화: 읽기 전용. */

  double dram_cell_C;
  /* [한국어] DRAM 저장 커패시터 용량 (F) — 1비트를 저장하는 커패시터 크기.
   * 설정자: tech_params.cc 초기화 (보통 수십 fF).
   * 읽는 자: DRAM 비트라인 전압 스윙 = dram_cell_C × Vdd / (C_bl + dram_cell_C).
   * 동기화: 읽기 전용. */

  double gm_sense_amp_latch;
  /* [한국어] 센스앰프 래치의 상호 컨덕턴스(gm, S) — SA 재생 속도 결정 파라미터.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: compute_sa_delay()에서 래치 재생 지연 = C / gm 로 계산.
   * 동기화: 읽기 전용. */

  double w_nmos_b_mux;
  /* [한국어] 비트 멀티플렉서 NMOS 트랜지스터 폭 (미터) — deg_bl_muxing 단계의 NMOS 선택 스위치.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 비트 mux 커패시턴스·지연 계산.
   * 동기화: 읽기 전용. */

  double w_nmos_sa_mux;
  /* [한국어] SA 멀티플렉서 NMOS 트랜지스터 폭 (미터).
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: SA mux 커패시턴스·지연 계산.
   * 동기화: 읽기 전용. */

  double w_pmos_bl_precharge;
  /* [한국어] 비트라인 프리차지 PMOS 트랜지스터 폭 (미터) — 비트라인을 Vdd로 충전하는 PMOS.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: bl_precharge_eq_drv 드라이버 크기 결정.
   * 동기화: 읽기 전용. */

  double w_pmos_bl_eq;
  /* [한국어] 비트라인 이퀄라이저 PMOS 트랜지스터 폭 (미터) — BL/BLB 간 전위 균등화.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 비트라인 이퀄라이즈 드라이버 설계.
   * 동기화: 읽기 전용. */

  double MIN_GAP_BET_P_AND_N_DIFFS;
  /* [한국어] P-type과 N-type 확산 영역 간 최소 간격 (미터) — CMOS 레이아웃 DRC 규칙.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 셀 행 높이(cell_h_def) 계산. 동기화: 읽기 전용. */

  double MIN_GAP_BET_SAME_TYPE_DIFFS;
  /* [한국어] 동일 타입 확산 영역 간 최소 간격 (미터) — 같은 타입 NMOS/PMOS 간 간격.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 셀 폭/높이 계산. 동기화: 읽기 전용. */

  double HPOWERRAIL;
  /* [한국어] 전원 레일(power rail) 높이 (미터) — Vdd/GND 메탈 배선의 물리적 높이.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 셀 행 높이(cell_h_def) 결정.
   * 동기화: 읽기 전용. */

  double cell_h_def;
  /* [한국어] 표준 셀 행 높이 정의값 (미터) — SRAM/로직 셀의 기본 단위 높이.
   * = MIN_GAP_BET_P_AND_N_DIFFS + HPOWERRAIL + ... 등으로 계산.
   * 설정자: tech_params.cc에서 DRC 간격 합산으로 계산.
   * 읽는 자: drain_C_()에서 드레인 커패시턴스 면적 계산 시 높이 기준으로 사용.
   *          셀 면적, 디코더 높이 계산에 광범위하게 사용.
   * 동기화: 읽기 전용. */

  double chip_layout_overhead;
  /* [한국어] 칩 레이아웃 오버헤드 계수 (dimensionless, 보통 1.0 이상) — 셀 면적 외 배선/격리 면적 보정.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 최종 캐시 면적 = 계산 면적 × chip_layout_overhead.
   * 동기화: 읽기 전용. */

  double macro_layout_overhead;
  /* [한국어] 매크로 레이아웃 오버헤드 계수 — 메모리 매크로 배치 시 추가 면적 비율.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 면적 결과 보정. 동기화: 읽기 전용. */

  double sckt_co_eff;
  /* [한국어] 소켓 계수(socket coefficient) — 패키지/소켓 연결에 따른 전력/면적 보정 계수.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 전력 결과 보정. 동기화: 읽기 전용. */

  double fringe_cap;
  /* [한국어] 배선 프린지(fringe) 커패시턴스 (F/m) — 배선 끝단에서 발생하는 기생 커패시턴스.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: 게이트 커패시턴스 보정 계산. 동기화: 읽기 전용. */

  uint64_t h_dec;
  /* [한국어] 디코더 행 수 (H-decoder row count) — 예비디코더 회로 설계 시 사용하는 행 개수.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: Predec/PredecBlk 설계 시 디코더 크기 결정.
   * 동기화: 읽기 전용. */

  DeviceType sram_cell;   // SRAM cell transistor
  /* [한국어] SRAM 셀 트랜지스터 파라미터 (6T SRAM의 접근/드라이버/풀업 트랜지스터 특성).
   * 설정자: tech_params.cc에서 SRAM 셀 공정 데이터로 초기화.
   * 읽는 자: SRAM 비트라인 전류, 지연 계산에 사용.
   * 동기화: 읽기 전용. */

  DeviceType dram_acc;    // DRAM access transistor
  /* [한국어] DRAM 접근 트랜지스터 파라미터 — 단일 NMOS 접근 트랜지스터의 특성.
   * 설정자: tech_params.cc에서 DRAM 셀 공정 데이터로 초기화.
   * 읽는 자: DRAM 비트라인 충전/방전 지연 계산. 동기화: 읽기 전용. */

  DeviceType dram_wl;     // DRAM wordline transistor
  /* [한국어] DRAM 워드라인 부스트 트랜지스터 파라미터 (vpp 전압 동작 기준).
   * 설정자: tech_params.cc 초기화. 읽는 자: DRAM 워드라인 드라이버 설계.
   * 동기화: 읽기 전용. */

  DeviceType peri_global; // peripheral global
  /* [한국어] 주변부 글로벌 로직 트랜지스터 파라미터 — 디코더, 드라이버, H-tree 노드 등에 사용.
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: Htree2(기본 디바이스), Decoder, Driver 등 대부분의 주변 회로 설계.
   * 동기화: 읽기 전용. */

  DeviceType cam_cell;   // SRAM cell transistor
  /* [한국어] CAM 셀 트랜지스터 파라미터 (CAM 비교기 회로 포함).
   * 설정자: tech_params.cc에서 CAM 셀 공정 데이터로 초기화.
   * 읽는 자: CAM 검색 경로 지연·전력 계산.
   * 동기화: 읽기 전용. */

  InterconnectType wire_local;
  /* [한국어] 로컬 배선 층 파라미터 (셀 내부 배선 — 가장 좁고 높은 저항).
   * 설정자: tech_params.cc 초기화. 읽는 자: Mat 내부 짧은 배선 RC 계산.
   * 동기화: 읽기 전용. */

  InterconnectType wire_inside_mat;
  /* [한국어] mat 내부 배선 층 파라미터 (서브어레이 간 배선 — 중간 수준 RC).
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: Subarray 출력 배선, 비트 mux 배선 RC 계산.
   * 동기화: 읽기 전용. */

  InterconnectType wire_outside_mat;
  /* [한국어] mat 외부 배선 층 파라미터 (글로벌 배선 — H-tree 링크, 가장 넓고 낮은 저항).
   * 설정자: tech_params.cc 초기화.
   * 읽는 자: Htree2에서 배선 pitch(g_tp.wire_outside_mat.pitch)로 H-tree 면적 계산.
   *          Wire(wire_outside_mat, len) 형태로 RC 지연 계산에 광범위하게 사용.
   * 동기화: 읽기 전용. */

  ScalingFactor scaling_factor;
  /* [한국어] 기술 스케일링 계수 인스턴스. 설정자: tech_params.cc 초기화.
   * 읽는 자: 전력·면적 스케일링 계산. 동기화: 읽기 전용. */

  MemoryType sram;
  /* [한국어] SRAM 셀 기하 파라미터 (b_w, b_h, Vbitpre 등).
   * 설정자: tech_params.cc 초기화. 읽는 자: Subarray 생성 시 셀 크기 계산.
   * 동기화: 읽기 전용. */

  MemoryType dram;
  /* [한국어] DRAM 셀 기하 파라미터.
   * 설정자: tech_params.cc 초기화. 읽는 자: DRAM Subarray 생성 시 셀 크기 계산.
   * 동기화: 읽기 전용. */

  MemoryType cam;
  /* [한국어] CAM 셀 기하 파라미터.
   * 설정자: tech_params.cc 초기화. 읽는 자: CAM Subarray 생성 시 셀 크기 계산.
   * 동기화: 읽기 전용. */

  void display(uint32_t indent = 0); // [한국어] 전체 기술 파라미터 출력 (디버그용)

  /*
   * [한국어]
   * TechnologyParameter::reset - 모든 기술 파라미터 초기화
   *
   * @return: (void) — 모든 필드와 내부 클래스 인스턴스를 0/초기 상태로 리셋
   *
   * 새로운 공정 노드 파라미터를 로드하기 전 기존 값을 제거하는 용도.
   * 내부 클래스(sram_cell, dram_acc 등)의 reset()을 순서대로 호출한다.
   */
  void reset()
  {
    dram_cell_Vdd  = 0; // [한국어] DRAM 셀 전원 전압 초기화
    dram_cell_I_on = 0; // [한국어] DRAM 셀 온-전류 초기화
    dram_cell_C    = 0; // [한국어] DRAM 저장 커패시터 초기화
    vpp            = 0; // [한국어] DRAM 워드라인 부스트 전압 초기화

    sense_delay               = 0; // [한국어] SA 기준 지연 초기화
    sense_dy_power            = 0; // [한국어] SA 동적 전력 초기화
    fringe_cap                = 0; // [한국어] 프린지 커패시턴스 초기화
//    horiz_dielectric_constant = 0;  // [한국어] (주석 처리: 현재 InterconnectType 내부로 이전)
//    vert_dielectric_constant  = 0;
//    aspect_ratio              = 0;
//    miller_value              = 0;
//    ild_thickness             = 0;

    dram_cell_I_off_worst_case_len_temp = 0; // [한국어] DRAM 최악 누설 전류 초기화

    sram_cell.reset();    // [한국어] SRAM 셀 트랜지스터 파라미터 초기화
    dram_acc.reset();     // [한국어] DRAM 접근 트랜지스터 파라미터 초기화
    dram_wl.reset();      // [한국어] DRAM 워드라인 트랜지스터 파라미터 초기화
    peri_global.reset();  // [한국어] 주변부 글로벌 트랜지스터 파라미터 초기화
    cam_cell.reset();     // [한국어] CAM 셀 트랜지스터 파라미터 초기화

    scaling_factor.reset(); // [한국어] 스케일링 계수 초기화

    wire_local.reset();        // [한국어] 로컬 배선 파라미터 초기화
    wire_inside_mat.reset();   // [한국어] mat 내부 배선 파라미터 초기화
    wire_outside_mat.reset();  // [한국어] mat 외부 배선 파라미터 초기화

    sram.reset(); // [한국어] SRAM 셀 기하 초기화
    dram.reset(); // [한국어] DRAM 셀 기하 초기화
    cam.reset();  // [한국어] CAM 셀 기하 초기화

    chip_layout_overhead  = 0; // [한국어] 칩 레이아웃 오버헤드 초기화
    macro_layout_overhead = 0; // [한국어] 매크로 레이아웃 오버헤드 초기화
    sckt_co_eff           = 0; // [한국어] 소켓 계수 초기화
  }
};



class DynamicParameter
/* [한국어] 캐시 파티션 파라미터 클래스.
 * InputParameter(g_ip)의 캐시 설계 요구사항(용량, 어소시에이티비티, 포트 수 등)과
 * 탐색 후보(Ndwl, Ndbl, Nspd, Ndcm, Ndsam_lev_1/2)로부터 계산되는 구조적 파라미터를 담는다.
 * UCA 최적 탐색 루프에서 후보마다 새로 생성되며, 유효한 구성인지를 is_valid로 표시한다.
 * Mat, Subarray, Htree2 등 모든 하위 모듈은 DynamicParameter const 참조를 통해 구조를 파악한다. */
{
  public:
    bool is_tag;
    /* [한국어] 태그 어레이 여부 플래그.
     * true이면 이 DynamicParameter가 태그 배열(주소 비교용)을 위한 파라미터임을 나타낸다.
     * 설정자: DynamicParameter 생성자의 is_tag_ 인수.
     * 읽는 자: Mat::compute_delays()에서 태그 비교기 경로 활성화 여부 결정.
     * 동기화: 단일 스레드. */

    bool pure_ram;
    /* [한국어] 순수 RAM(데이터 배열만) 모드 여부.
     * true이면 태그 배열 없이 데이터 저장 전용 구성.
     * 설정자: 생성자 인수. 읽는 자: 디코더/배선 설계 시 태그 경로 제외.
     * 동기화: 단일 스레드. */

    bool pure_cam;
    /* [한국어] 순수 CAM(Content Addressable Memory) 모드 여부.
     * true이면 전체 배열이 CAM 셀로만 구성 (데이터 RAM 없음).
     * 설정자: 생성자 인수. 읽는 자: compute_delays()에서 CAM 경로 전용으로 전환.
     * 동기화: 단일 스레드. */

    bool fully_assoc;
    /* [한국어] 완전 연관(Fully Associative) 캐시 여부.
     * true이면 모든 웨이를 병렬로 비교하는 FA 구조.
     * 설정자: 생성자 인수. 읽는 자: Mat에서 FA CAM 경로(태그 비교기, ml_to_ram_wl_drv) 활성화.
     * 동기화: 단일 스레드. */

    int tagbits;
    /* [한국어] 태그 비트 수 (is_tag=true일 때 의미).
     * = 전체 주소 비트 - 인덱스 비트 - 오프셋 비트.
     * 설정자: 생성자에서 g_ip->tag_w로 초기화.
     * 읽는 자: Mat에서 비교기 높이 계산 시 사용.
     * 동기화: 단일 스레드. */

    int num_subarrays;  // only for leakage computation  -- the number of subarrays per bank
    /* [한국어] 뱅크당 서브어레이 총 수 (누설 전력 계산 전용).
     * = Ndwl × Ndbl.
     * 설정자: 생성자에서 계산.
     * 읽는 자: Mat::compute_power_energy()에서 비활성 서브어레이 누설 전력 계산.
     * 동기화: 단일 스레드. */

    int num_mats;       // only for leakage computation  -- the number of mats per bank
    /* [한국어] 뱅크당 mat 총 수 (누설 전력 계산 전용).
     * = num_mats_h_dir × num_mats_v_dir.
     * 설정자: 생성자에서 계산.
     * 읽는 자: Mat에서 전체 뱅크의 누설 전력 = 1 mat 누설 × num_mats 스케일링.
     * 동기화: 단일 스레드. */

    double Nspd;
    /* [한국어] 서브어레이당 병렬 데이터 선(Number of Subarray Parallel Data lines).
     * 비트라인 멀티플렉싱과 관련된 파라미터로, 서브어레이 분할 방식을 결정한다.
     * 설정자: 생성자 인수 Nspd_.
     * 읽는 자: num_c_subarray, deg_bl_muxing 계산에 사용.
     * 값 범위: 0.5, 1, 2, 4, 8 등. 동기화: 단일 스레드. */

    int Ndwl;
    /* [한국어] 수평 방향 서브어레이 분할 수 (Number of Data Word-Line divisions).
     * H-tree 수평 레벨 수 = log2(Ndwl/2), mat 가로 크기에 영향.
     * 설정자: 생성자 인수 Ndwl_.
     * 읽는 자: Htree2에서 ndwl로 사용, num_mats_h_dir 계산.
     * 값 범위: 2 이상 짝수. 동기화: 단일 스레드. */

    int Ndbl;
    /* [한국어] 수직 방향 서브어레이 분할 수 (Number of Data Bit-Line divisions).
     * H-tree 수직 레벨 수 = log2(Ndbl/2), mat 세로 크기에 영향.
     * 설정자: 생성자 인수 Ndbl_.
     * 읽는 자: Htree2에서 ndbl로 사용, num_mats_v_dir 계산.
     * 값 범위: 2 이상 짝수. 동기화: 단일 스레드. */

    int Ndcm;
    /* [한국어] 열 멀티플렉서 수 (Number of Data Column Mux).
     * 비트라인 멀티플렉싱 단 수를 결정한다.
     * 설정자: 생성자 인수 Ndcm_.
     * 읽는 자: deg_bl_muxing = Ndcm 계산.
     * 값 범위: 1, 2, 4, 8 등. 동기화: 단일 스레드. */

    int deg_bl_muxing;
    /* [한국어] 비트라인 멀티플렉싱 차수 (= Ndcm).
     * 하나의 SA가 담당하는 비트라인 수.
     * 설정자: 생성자에서 Ndcm으로 초기화.
     * 읽는 자: Mat에서 SA 수 = num_c_subarray / deg_bl_muxing 계산.
     * 동기화: 단일 스레드. */

    int deg_senseamp_muxing_non_associativity;
    /* [한국어] 어소시에이티비티를 제외한 SA 멀티플렉싱 차수.
     * 어소시에이티비티로 인한 SA mux와 별개로 추가되는 mux 단.
     * 설정자: 생성자에서 계산.
     * 읽는 자: SA mux 디코더 설계. 동기화: 단일 스레드. */

    int Ndsam_lev_1;
    /* [한국어] SA 멀티플렉서 레벨 1 크기 (Number of Data Sense Amp Mux level 1).
     * 2단 SA mux의 첫 번째 단의 선택 수.
     * 설정자: 생성자 인수 Ndsam_lev_1_.
     * 읽는 자: Mat에서 SA mux lev 1 디코더 설계.
     * 동기화: 단일 스레드. */

    int Ndsam_lev_2;
    /* [한국어] SA 멀티플렉서 레벨 2 크기 (Number of Data Sense Amp Mux level 2).
     * 2단 SA mux의 두 번째 단의 선택 수.
     * 설정자: 생성자 인수 Ndsam_lev_2_.
     * 읽는 자: Mat에서 SA mux lev 2 디코더 설계.
     * 동기화: 단일 스레드. */

    int number_addr_bits_mat;             // per port
    /* [한국어] mat 1개가 디코딩해야 하는 어드레스 비트 수 (포트당).
     * 전체 어드레스 비트에서 뱅크/mat 선택에 사용된 비트를 뺀 값.
     * 설정자: 생성자에서 계산. 읽는 자: 행 디코더 설계.
     * 동기화: 단일 스레드. */

    int number_subbanks_decode;           // per_port
    /* [한국어] 서브뱅크 선택에 필요한 디코딩 비트 수 (포트당).
     * 설정자: 생성자에서 계산. 읽는 자: 디코더 설계.
     * 동기화: 단일 스레드. */

    int num_di_b_bank_per_port;
    /* [한국어] 포트당 뱅크 데이터 입력 비트 수.
     * 설정자: 생성자에서 g_ip 및 파티션 파라미터로 계산.
     * 읽는 자: Htree2(Data_in_htree) 생성 시 d_inbits 인수로 전달.
     * 동기화: 단일 스레드. */

    int num_do_b_bank_per_port;
    /* [한국어] 포트당 뱅크 데이터 출력 비트 수.
     * 설정자: 생성자에서 계산.
     * 읽는 자: Htree2(Data_out_htree) 생성 시 d_outbits 인수로 전달.
     * 동기화: 단일 스레드. */

    int num_di_b_mat;
    /* [한국어] mat 1개당 데이터 입력 비트 수 (= num_di_b_bank_per_port / num_mats_h_dir).
     * 설정자: 생성자에서 계산. 읽는 자: Mat에서 비트라인 구성 설계.
     * 동기화: 단일 스레드. */

    int num_do_b_mat;
    /* [한국어] mat 1개당 데이터 출력 비트 수.
     * 설정자: 생성자에서 계산. 읽는 자: Mat에서 SA 출력 드라이버 설계.
     * 동기화: 단일 스레드. */

    int num_di_b_subbank;
    /* [한국어] 서브뱅크당 데이터 입력 비트 수. 설정자: 생성자. 동기화: 단일 스레드. */

    int num_do_b_subbank;
    /* [한국어] 서브뱅크당 데이터 출력 비트 수. 설정자: 생성자. 동기화: 단일 스레드. */

    int num_si_b_mat;
    /* [한국어] mat 1개당 검색 입력 비트 수 (CAM Search Input bits per mat).
     * 설정자: 생성자에서 계산 (pure_cam/fully_assoc일 때만 유효).
     * 읽는 자: Mat에서 서치라인 구성 설계.
     * 동기화: 단일 스레드. */

    int num_so_b_mat;
    /* [한국어] mat 1개당 검색 출력 비트 수 (CAM Search Output bits per mat).
     * 설정자: 생성자에서 계산. 읽는 자: Mat에서 매치라인 출력 설계.
     * 동기화: 단일 스레드. */

    int num_si_b_subbank;
    /* [한국어] 서브뱅크당 검색 입력 비트 수. 설정자: 생성자. 동기화: 단일 스레드. */

    int num_so_b_subbank;
    /* [한국어] 서브뱅크당 검색 출력 비트 수. 설정자: 생성자. 동기화: 단일 스레드. */

    int num_si_b_bank_per_port;
    /* [한국어] 포트당 뱅크 검색 입력 비트 수. 설정자: 생성자. 동기화: 단일 스레드. */

    int num_so_b_bank_per_port;
    /* [한국어] 포트당 뱅크 검색 출력 비트 수. 설정자: 생성자. 동기화: 단일 스레드. */

    int number_way_select_signals_mat;
    /* [한국어] mat당 웨이 선택 신호 수 — 어소시에이티브 캐시에서 어떤 웨이를 쓸지 선택하는 신호.
     * 설정자: 생성자에서 어소시에이티비티 파라미터로 계산.
     * 읽는 자: Mat에서 웨이 선택 예비디코더 설계.
     * 동기화: 단일 스레드. */

    int num_act_mats_hor_dir;
    /* [한국어] 수평 방향 동시 활성 mat 수.
     * 데이터 출력 폭에 따라 수평으로 동시에 읽히는 mat 수.
     * 설정자: 생성자에서 num_do_b_mat 및 g_ip 파라미터로 계산.
     * 읽는 자: Mat에서 SA mux 레벨 결정, Htree2 생성 시 mat 수 결정.
     * 동기화: 단일 스레드. */

    int num_act_mats_hor_dir_sl;
    /* [한국어] 수평 방향 동시 활성 mat 수 (검색 라인 기준, CAM 전용).
     * 설정자: 생성자에서 계산. 읽는 자: CAM 검색 경로 설계.
     * 동기화: 단일 스레드. */

    bool is_dram;
    /* [한국어] DRAM 셀 타입 여부.
     * true이면 DRAM 셀 RC 파라미터(dram_acc, dram_cell_C 등)를 사용.
     * 설정자: 생성자에서 g_ip->ram_cell_tech_type으로 결정.
     * 읽는 자: Mat에서 DRAM/SRAM 경로 분기. 동기화: 단일 스레드. */

    double V_b_sense;
    /* [한국어] 비트라인 센싱 전압 (V) — SA가 감지할 수 있는 최소 비트라인 전압 차이.
     * 설정자: 생성자에서 g_ip 기반 계산 (보통 SRAM: Vdd * 0.08 정도).
     * 읽는 자: compute_sa_delay()에서 SA 입력 감도 기준으로 사용.
     * 동기화: 단일 스레드. */

    unsigned int num_r_subarray;
    /* [한국어] 서브어레이당 행(row) 수.
     * 전체 워드 수 / (Ndwl × Ndbl × 어소시에이티비티 보정).
     * 설정자: 생성자에서 계산.
     * 읽는 자: Subarray에서 비트라인 길이 = num_r_subarray × cell.h 계산.
     * 값 범위: 2 이상 짝수 (보통 32~512). 동기화: 단일 스레드. */

    unsigned int num_c_subarray;
    /* [한국어] 서브어레이당 열(column) 수 = 비트라인 수.
     * 설정자: 생성자에서 계산.
     * 읽는 자: Subarray에서 비트라인 총 커패시턴스, SA 수 계산.
     * 값 범위: 2 이상 짝수. 동기화: 단일 스레드. */

    int tag_num_r_subarray; //sheng: fully associative cache tag and data must be computed together, data and tag must be separate
    /* [한국어] FA 캐시 태그 서브어레이의 행 수 (is_tag=true인 FA 캐시 전용).
     * 태그와 데이터를 함께 계산해야 하므로 별도 필드로 관리.
     * 설정자: 생성자에서 FA 캐시 태그 배열 크기로 계산.
     * 읽는 자: Mat에서 태그 어레이 행 수 참조. 동기화: 단일 스레드. */

    int tag_num_c_subarray;
    /* [한국어] FA 캐시 태그 서브어레이의 열 수.
     * 설정자: 생성자에서 FA 캐시 태그 배열 크기로 계산.
     * 읽는 자: Mat에서 태그 비트라인 수 참조. 동기화: 단일 스레드. */

    int data_num_r_subarray;
    /* [한국어] FA 캐시 데이터 서브어레이의 행 수 (is_tag=false인 FA 캐시 전용).
     * 설정자: 생성자에서 계산. 읽는 자: Mat. 동기화: 단일 스레드. */

    int data_num_c_subarray;
    /* [한국어] FA 캐시 데이터 서브어레이의 열 수.
     * 설정자: 생성자에서 계산. 읽는 자: Mat. 동기화: 단일 스레드. */

    int num_mats_h_dir;
    /* [한국어] 수평 방향 mat 수 (= Ndwl / 2 또는 파티션에 따른 계산값).
     * 설정자: 생성자에서 Ndwl 기반 계산.
     * 읽는 자: UCA에서 mat 배열 구성, Htree2 생성 시 ndwl 인수로 전달.
     * 동기화: 단일 스레드. */

    int num_mats_v_dir;
    /* [한국어] 수직 방향 mat 수 (= Ndbl / 2 또는 파티션에 따른 계산값).
     * 설정자: 생성자에서 Ndbl 기반 계산.
     * 읽는 자: UCA에서 mat 배열 구성, Htree2 생성 시 ndbl 인수로 전달.
     * 동기화: 단일 스레드. */

    uint32_t ram_cell_tech_type;
    /* [한국어] 메모리 셀 기술 타입 코드 (inputParameter::ram_cell_tech_type에서 복사).
     * 예: 0=SRAM, 1=DRAM, 2=embedded_DRAM, 3=PCRAM 등.
     * 설정자: 생성자에서 g_ip->ram_cell_tech_type으로 초기화.
     * 읽는 자: Mat/Subarray에서 셀 파라미터 선택 분기.
     * 동기화: 단일 스레드. */

    double dram_refresh_period;
    /* [한국어] DRAM 리프레시 주기 (초) — 이 주기마다 모든 셀을 재충전해야 함.
     * 설정자: 생성자에서 g_ip 기반 계산 (보통 수십 ms).
     * 읽는 자: DRAM 평균 누설 전력 계산 시 리프레시 에너지 분배.
     * 동기화: 단일 스레드. */

    /*
     * [한국어]
     * DynamicParameter::DynamicParameter - 기본 생성자 (파라미터 없는 초기 상태)
     *
     * 모든 정수/실수 필드를 0, bool 필드를 false로 초기화.
     * 실제 계산에 사용하기 전 복사나 임시 객체에 쓰인다.
     */
    DynamicParameter();

    /*
     * [한국어]
     * DynamicParameter::DynamicParameter - 캐시 파티션 파라미터 계산 생성자
     *
     * @is_tag_        : 태그 배열 여부
     * @pure_ram_      : 순수 RAM 모드
     * @pure_cam_      : 순수 CAM 모드
     * @Nspd_          : 서브어레이 병렬 데이터 선
     * @Ndwl_          : 수평 서브어레이 분할 수
     * @Ndbl_          : 수직 서브어레이 분할 수
     * @Ndcm_          : 열 멀티플렉서 수
     * @Ndsam_lev_1_   : SA mux 레벨 1 크기
     * @Ndsam_lev_2_   : SA mux 레벨 2 크기
     * @is_main_mem_   : 주 메모리(DRAM) 모드 여부
     * @return         : (생성자) — is_valid로 유효성 표시
     *
     * InputParameter(g_ip)와 입력 파티션 파라미터로부터 서브어레이/mat 수, 비트 수,
     * 어드레스 비트 배분 등 모든 파생 파라미터를 계산한다.
     * 구성이 물리적으로 불가능하면 is_valid = false로 설정하여 호출자가 기각할 수 있게 한다.
     *
     * 호출 체인:
     *   UCA 최적 탐색 루프 → [DynamicParameter(...)] → UCA::UCA(dp)
     */
    DynamicParameter(
        bool         is_tag_,
        int          pure_ram_,
        int          pure_cam_,
        double       Nspd_,
        unsigned int Ndwl_,
        unsigned int Ndbl_,
        unsigned int Ndcm_,
        unsigned int Ndsam_lev_1_,
        unsigned int Ndsam_lev_2_,
        bool         is_main_mem_);

    int use_inp_params;
    /* [한국어] 입력 파라미터 사용 플래그.
     * 0이면 CACTI가 내부에서 파라미터를 최적화, 1이면 g_ip의 파라미터를 그대로 사용.
     * 설정자: 생성자에서 초기화. 읽는 자: 파라미터 계산 분기.
     * 동기화: 단일 스레드. */

    unsigned int num_rw_ports;
    /* [한국어] 읽기-쓰기 포트 수 (= g_ip->num_rw_ports).
     * 설정자: 생성자에서 g_ip 복사. 읽는 자: Mat에서 포트 수에 비례한 전력/면적 계산.
     * 동기화: 단일 스레드. */

    unsigned int num_rd_ports;
    /* [한국어] 읽기 전용 포트 수 (= g_ip->num_rd_ports).
     * 설정자: 생성자에서 g_ip 복사. 동기화: 단일 스레드. */

    unsigned int num_wr_ports;
    /* [한국어] 쓰기 전용 포트 수 (= g_ip->num_wr_ports).
     * 설정자: 생성자에서 g_ip 복사. 동기화: 단일 스레드. */

    unsigned int num_se_rd_ports;  // number of single ended read ports
    /* [한국어] 단일 끝단(Single-Ended) 읽기 포트 수 — 차동(differential) 비트라인이 아닌 포트.
     * 설정자: 생성자에서 g_ip 복사. 동기화: 단일 스레드. */

    unsigned int num_search_ports;
    /* [한국어] CAM 검색 포트 수 (= g_ip->num_search_ports).
     * 설정자: 생성자에서 g_ip 복사.
     * 읽는 자: CAM 검색 경로 설계 시 동시 검색 수.
     * 동기화: 단일 스레드. */

    unsigned int out_w; // == nr_bits_out
    /* [한국어] 캐시 출력 비트 수 (= g_ip->out_w, nr_bits_out).
     * 한 번의 접근에서 읽히는 데이터 비트 수 (캐시 라인 크기 × 8).
     * 설정자: 생성자에서 g_ip 복사. 읽는 자: 데이터 버스 폭 결정.
     * 동기화: 단일 스레드. */

    bool   is_main_mem;
    /* [한국어] 주 메모리(DRAM) 모드 여부.
     * true이면 DRAM 타이밍(RAS/CAS 지연)과 리프레시를 포함한 모델 적용.
     * 설정자: 생성자 인수 is_main_mem_. 읽는 자: 타이밍 모델 분기.
     * 동기화: 단일 스레드. */

    Area   cell, cam_cell; //cell is the sram_cell in both nomal cache/ram and FA.
    /* [한국어] cell    : SRAM 셀(또는 FA의 데이터 셀) 1개의 면적 (w × h).
     *          cam_cell: CAM 셀 1개의 면적 (태그 비교기 포함).
     * 설정자: 생성자에서 g_tp.sram / g_tp.cam 파라미터로 초기화.
     * 읽는 자: Mat에서 비트라인 길이·면적 계산 시 cell.h, cell.w 참조.
     * 동기화: 단일 스레드. */

    bool   is_valid;
    /* [한국어] 이 DynamicParameter 구성이 물리적으로 유효한지 여부.
     * false이면 파티션 파라미터 조합이 불가능(예: 서브어레이 행/열 수가 너무 작거나 큰 경우).
     * 설정자: 생성자에서 각종 유효성 검사 후 결정.
     * 읽는 자: UCA 탐색 루프에서 is_valid == false이면 이 후보를 즉시 기각.
     * 동기화: 단일 스레드. */
};



extern InputParameter * g_ip;
/* [한국어] 전역 InputParameter 포인터 — 사용자가 제공한 캐시 설계 요구사항.
 * 설정자: cacti_interface.cc의 cacti() 진입 함수에서 new InputParameter(...)로 할당.
 * 읽는 자: DynamicParameter 생성자, TechnologyParameter 초기화 함수, UCA 등 전체 CACTI에서 참조.
 * 값 범위: 유효한 InputParameter 포인터 (NULL이면 crash). 계산 중에는 불변.
 * 동기화: 단일 스레드 (CACTI는 단일 스레드 라이브러리). */

extern TechnologyParameter g_tp;
/* [한국어] 전역 TechnologyParameter 인스턴스 — 공정 노드별 소자/배선/셀 파라미터.
 * 설정자: tech_params.cc의 init_tech_params()에서 g_ip->F_sz_nm(공정 노드 nm)을 기반으로 초기화.
 * 읽는 자: 모든 CACTI 소스 파일 (Wire, Mat, Htree2, Decoder, Driver 등)에서 g_tp.xxx 형태로 참조.
 * 값 범위: 유효한 공정 파라미터 (init_tech_params() 호출 후). 계산 중에는 불변.
 * 동기화: init_tech_params() 완료 후 읽기 전용. 단일 스레드. */

#endif

