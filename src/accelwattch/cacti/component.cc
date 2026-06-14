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
 * [한국어 설명] CACTI 컴포넌트 기반 클래스 구현 (component.cc)
 *
 * === 파일의 역할 ===
 * Component 기반 클래스의 생성자, 소멸자 및 4개의 공통 유틸리티 함수를 구현한다.
 * compute_gate_area()는 INV/NOR/NAND 게이트의 레이아웃 면적을 PMOS/NMOS 폴딩 모델로 계산하고,
 * compute_tr_width_after_folding()은 폴딩 후 확산 영역 폭을 산출한다.
 * height_sense_amplifier()는 센스앰프 레이아웃 높이를 계산하며,
 * logical_effort()는 Logical Effort 방법론으로 버퍼 체인 최적 단 수와 트랜지스터 폭을 결정한다.
 * 이 함수들은 CACTI의 모든 하위 컴포넌트(Bank, Mat, Decoder, Subarray 등)에서 공통 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Component는 CACTI 컴포넌트 계층의 루트이며, 모든 하위 컴포넌트가 상속한다.
 * AccelWattch → cacti_interface → Bank → Mat → Subarray/Decoder 각 단계에서
 * compute_gate_area()와 logical_effort()가 트랜지스터 크기와 레이아웃 면적을 결정한다.
 * 이 파일의 함수들은 호출 빈도가 매우 높아 CACTI 계산 성능의 핵심 경로를 구성한다.
 *
 * === 타 모듈과의 연결 ===
 * - component.h: Component 클래스 선언 (area, power, delay, cycle_time, 함수 원형).
 * - bank.h: Bank 클래스 포함 (순환 의존 방지를 위한 구현 파일에서만 포함).
 * - decoder.h: Decoder 클래스 포함 (compute_gate_area 호출처 중 하나).
 * - parameter.h: g_tp(min_w_nmos_, cell_h_def, w_sense_p 등), g_ip(F_sz_um) 전역 파라미터.
 * - basic_circuit.h: gate_C() — logical_effort() 내부에서 트랜지스터 커패시턴스 계산.
 *
 * === 주요 함수/구조체 요약 ===
 * - compute_gate_area(): 게이트 타입별 폴딩 레이아웃 면적 계산 (INV/NOR/NAND 분기).
 * - compute_tr_width_after_folding(): 폴딩 횟수 계산 후 확산 영역 총 가로 폭 반환.
 * - height_sense_amplifier(): PMOS/NMOS 영역 높이 + P-N 갭 합산으로 SA 높이 반환.
 * - logical_effort(): fopt=4.0 기반 최적 버퍼 단 수 계산 + 각 단 트랜지스터 폭 역방향 산정.
 */


#include <assert.h>    // [한국어] assert() — 불변식 검증 (logical_effort에서 num_gates 범위 검사)
#include <iostream>    // [한국어] cout/cerr — 오류 메시지 출력 (compute_gate_area 미지원 게이트 타입 에러)
#include <math.h>      // [한국어] log(), pow(), ceil() — logical_effort 내 수치 계산

#include "bank.h"      // [한국어] Bank 클래스 (구현 파일에서만 포함; 헤더에서는 forward 선언)
#include "component.h" // [한국어] Component 클래스 선언
#include "decoder.h"   // [한국어] Decoder 클래스 (compute_gate_area 호출처)

using namespace std; // [한국어] std 네임스페이스 전역 using (CACTI 코드베이스 관례)



/*
 * [한국어]
 * Component - 기본 생성자
 *
 * @return: 없음 (생성자)
 *
 * area, power, rt_power를 기본 생성(0 초기화)하고 delay를 0으로 설정한다.
 * 하위 컴포넌트 생성자가 이를 자동으로 호출한다.
 *
 * 호출 체인: Bank()/Mat()/Decoder() 등 하위 생성자 → [Component()]
 */
Component::Component()
  :area(), power(), rt_power(),delay(0) // [한국어] area=0×0, power/rt_power=0, delay=0으로 초기화
{
}


/*
 * [한국어]
 * ~Component - 소멸자
 *
 * 동적 할당 자원 없음. 하위 클래스 소멸자가 이를 자동으로 호출한다.
 *
 * 호출 체인: 하위 소멸자 자동 호출
 */
Component::~Component()
{
}


/*
 * [한국어]
 * compute_diffusion_width - 스택/폴딩된 트랜지스터 확산 영역 가로 폭 계산 (내부 헬퍼)
 *
 * @num_stacked_in: 직렬 스택 입력 트랜지스터 수.
 *                  INV/NOR NMOS: 1, NAND NMOS: num_inputs, NOR PMOS: num_inputs.
 * @num_folded_tr:  폴딩 횟수. 트랜지스터를 셀 높이 내 여러 열로 접은 수.
 * @return:         확산 영역의 총 가로 폭 (µm). compute_gate_area()의 핵심 입력.
 *
 * 레이아웃 규칙:
 * - 확산 영역 좌우에 소스/드레인 콘택이 각 1개씩 필요 → 2×spacing_poly_to_poly 여유.
 * - poly 게이트 폭(w_poly) × num_stacked_in 및 게이트 간 간격(spacing_poly_to_poly).
 * - 폴딩(num_folded_tr > 1)이면 각 접힘마다 추가 콘택 공간과 poly가 추가됨.
 *
 * 호출 체인: compute_gate_area() → [compute_diffusion_width()]
 */
double Component::compute_diffusion_width(int num_stacked_in, int num_folded_tr)
{
  double w_poly = g_ip->F_sz_um;
  // [한국어] poly 게이트 폭 = 공정 피처 사이즈(µm). 각 스택 트랜지스터의 채널 길이 방향 폭.
  double spacing_poly_to_poly = g_tp.w_poly_contact + 2 * g_tp.spacing_poly_to_contact;
  // [한국어] poly-poly 간격 = 콘택 폭 + 양측 콘택-poly 최소 이격 × 2.
  //         소스/드레인 콘택을 배치하기 위한 최소 확산 폭.
  double total_diff_w = 2 * spacing_poly_to_poly +  // for both source and drain
                        num_stacked_in * w_poly +
                        (num_stacked_in - 1) * g_tp.spacing_poly_to_poly;
  // [한국어] 초기 확산 폭 계산:
  //   2×spacing_poly_to_poly — 좌우 소스/드레인 콘택 공간.
  //   num_stacked_in×w_poly — 직렬 스택 트랜지스터들의 총 poly 폭.
  //   (num_stacked_in-1)×spacing_poly_to_poly — 스택 트랜지스터 사이의 공유 드레인/소스 콘택 간격.

  if (num_folded_tr > 1) // [한국어] 폴딩이 있으면 (2번 이상 접힘) 추가 가로 폭 계산
  {
    total_diff_w += (num_folded_tr - 2) * 2 * spacing_poly_to_poly +
                    (num_folded_tr - 1) * num_stacked_in * w_poly +
                    (num_folded_tr - 1) * (num_stacked_in - 1) * g_tp.spacing_poly_to_poly;
    // [한국어] 폴딩 추가 폭:
    //   (num_folded_tr-2)×2×spacing: 첫/마지막 fold를 제외한 내부 fold의 공유 콘택 공간.
    //   (num_folded_tr-1)×num_stacked_in×w_poly: 추가 fold의 스택 poly 폭.
    //   (num_folded_tr-1)×(num_stacked_in-1)×spacing: 추가 fold 내 스택 간 간격.
  }

  return total_diff_w; // [한국어] 확산 영역의 총 가로 폭(µm) 반환
}



/*
 * [한국어]
 * compute_gate_area - INV/NOR/NAND 게이트의 레이아웃 면적 계산
 *
 * @gate_type:  게이트 종류 — INV(0), NOR(1), NAND(2).
 * @num_inputs: 게이트 팬인 수 (INV=1, NOR/NAND=입력 수).
 * @w_pmos:     PMOS 채널 폭 (µm). 0 이하이면 0.0 반환.
 * @w_nmos:     NMOS 채널 폭 (µm). 0 이하이면 0.0 반환.
 * @h_gate:     게이트 셀 높이 (µm). 파워 레일 포함 표준 셀 높이.
 * @return:     게이트 레이아웃 면적 (µm²). 유효하지 않은 입력이면 0.0.
 *
 * CMOS 레이아웃 규칙에 따라 트랜지스터를 셀 높이 내에 폴딩하여 배치하고
 * NMOS/PMOS 확산 영역의 가로 폭과 셀 높이로 면적을 계산한다.
 * - 트랜지스터 폴딩: 채널 폭이 폴딩 임계값(w_folded_*)을 초과하면 여러 열로 접음.
 * - INV: NMOS/PMOS 각 단일 스택. NOR: NMOS 병렬/PMOS 직렬. NAND: NMOS 직렬/PMOS 병렬.
 * - 면적 가로 = MAX(NMOS 확산 폭, PMOS 확산 폭). 높이 = 트랜지스터가 작으면 실제 높이, 아니면 h_gate.
 *
 * 호출 체인: Decoder/Subarray/Mat 등 → [compute_gate_area()] → compute_diffusion_width()
 */
double Component::compute_gate_area(
    int gate_type,
    int num_inputs,
    double w_pmos,
    double w_nmos,
    double h_gate)
{
  if (w_pmos <= 0.0 || w_nmos <= 0.0) // [한국어] 트랜지스터 폭이 0 이하이면 면적 0 반환 (유효하지 않은 입력)
  {
    return 0.0; // [한국어] 비유효 게이트 — 면적 없음
  }

  double w_folded_pmos, w_folded_nmos;   // [한국어] 폴딩 임계 폭: 이 값을 초과하면 여러 열로 접힘
  int    num_folded_pmos, num_folded_nmos; // [한국어] 폴딩 횟수: ceil(채널 폭 / 폴딩 임계 폭)
  double total_ndiff_w, total_pdiff_w;   // [한국어] NMOS/PMOS 확산 영역의 총 가로 폭(µm)
  Area gate;                             // [한국어] 계산된 게이트 면적을 담을 Area 객체

  double h_tr_region  = h_gate - 2 * g_tp.HPOWERRAIL;
  // [한국어] 실제 트랜지스터 배치 가능 높이 = 셀 높이 - 상하 파워 레일(VDD/GND) 높이 × 2.
  double ratio_p_to_n = w_pmos / (w_pmos + w_nmos);
  // [한국어] PMOS가 트랜지스터 영역 중 차지하는 비율 (0~1). NMOS+PMOS 면적 비율로 높이 분할.

  if (ratio_p_to_n >= 1 || ratio_p_to_n <= 0) // [한국어] 비율이 0 또는 1이면 둘 중 하나가 0 → 비유효
  {
    return 0.0; // [한국어] 비유효 비율 — 면적 0 반환
  }

  w_folded_pmos  = (h_tr_region - g_tp.MIN_GAP_BET_P_AND_N_DIFFS) * ratio_p_to_n;
  // [한국어] PMOS 폴딩 임계 폭 = 트랜지스터 영역 × PMOS 비율. 이 폭을 초과하면 PMOS를 접음.
  //         MIN_GAP_BET_P_AND_N_DIFFS: P형/N형 확산 사이 최소 이격 거리.
  w_folded_nmos  = (h_tr_region - g_tp.MIN_GAP_BET_P_AND_N_DIFFS) * (1 - ratio_p_to_n);
  // [한국어] NMOS 폴딩 임계 폭 = 트랜지스터 영역 × NMOS 비율.
  assert(w_folded_pmos > 0); // [한국어] PMOS 폴딩 임계 폭이 양수여야 함 (비율 검사 후 보장)

  num_folded_pmos = (int) (ceil(w_pmos / w_folded_pmos));
  // [한국어] PMOS 폴딩 횟수 = ceil(요구 채널 폭 / 폴딩 임계 폭). 트랜지스터를 몇 열로 접을지.
  num_folded_nmos = (int) (ceil(w_nmos / w_folded_nmos));
  // [한국어] NMOS 폴딩 횟수.

  switch (gate_type) // [한국어] 게이트 타입별 확산 폭 계산: 스택 구조와 폴딩 방식이 다름
  {
    case INV:
      // [한국어] 인버터: NMOS 1개, PMOS 1개 직렬 없음 (num_stacked_in=1).
      total_ndiff_w = compute_diffusion_width(1, num_folded_nmos);
      total_pdiff_w = compute_diffusion_width(1, num_folded_pmos);
      break;

    case NOR:
      // [한국어] NOR 게이트: NMOS num_inputs개 병렬 (각 1개씩, 폴딩 × num_inputs),
      //         PMOS num_inputs개 직렬 (스택 = num_inputs).
      total_ndiff_w = compute_diffusion_width(1, num_inputs * num_folded_nmos);
      // [한국어] NOR NMOS: 스택=1(병렬), 폴딩=num_inputs×num_folded_nmos.
      total_pdiff_w = compute_diffusion_width(num_inputs, num_folded_pmos);
      // [한국어] NOR PMOS: 스택=num_inputs(직렬), 폴딩=num_folded_pmos.
      break;

    case NAND:
      // [한국어] NAND 게이트: NMOS num_inputs개 직렬 (스택 = num_inputs),
      //         PMOS num_inputs개 병렬 (각 1개씩, 폴딩 × num_inputs).
      total_ndiff_w = compute_diffusion_width(num_inputs, num_folded_nmos);
      // [한국어] NAND NMOS: 스택=num_inputs(직렬), 폴딩=num_folded_nmos.
      total_pdiff_w = compute_diffusion_width(1, num_inputs * num_folded_pmos);
      // [한국어] NAND PMOS: 스택=1(병렬), 폴딩=num_inputs×num_folded_pmos.
      break;
    default:
      cout << "Unknown gate type: " << gate_type << endl; // [한국어] 지원하지 않는 게이트 타입 오류 출력
      exit(1); // [한국어] 비정상 종료 — 설계 오류이므로 회복 불가
  }

  gate.w = MAX(total_ndiff_w, total_pdiff_w);
  // [한국어] 게이트 셀 가로 폭 = NMOS 확산 폭과 PMOS 확산 폭 중 더 큰 값.
  //         두 영역이 같은 셀 가로 폭을 공유해야 하므로 더 큰 쪽으로 맞춤.

  if (w_folded_nmos > w_nmos) // [한국어] 폴딩 임계값이 실제 NMOS 폭보다 크면 → 폴딩 불필요, 셀 높이 줄임
  {
    //means that the height of the gate can
    //be made smaller than the input height specified, so calculate the height of the gate.
    gate.h = w_nmos + w_pmos + g_tp.MIN_GAP_BET_P_AND_N_DIFFS + 2 * g_tp.HPOWERRAIL;
    // [한국어] 실제 트랜지스터 높이 = NMOS 폭 + PMOS 폭 + P-N 이격 + 파워 레일 × 2.
    //         셀이 지정 높이보다 작게 만들 수 있음.
  }
  else // [한국어] 폴딩 임계값이 NMOS 폭보다 작거나 같으면 → 지정된 셀 높이 그대로 사용
  {
    gate.h = h_gate; // [한국어] 셀 높이 = 입력된 h_gate (표준 셀 높이 그대로)
  }
  return gate.get_area(); // [한국어] gate.w × gate.h 또는 직접 지정 area 반환 (µm²)
}



/*
 * [한국어]
 * compute_tr_width_after_folding - 폴딩 후 셀의 가로 폭(확산 폭) 계산
 *
 * @input_width:             트랜지스터의 요구 채널 폭 (µm).
 * @threshold_folding_width: 폴딩 임계 채널 폭 (µm). 초과 시 여러 열로 접음.
 * @return: 폴딩 후 셀의 총 가로 폭 (µm). input_width <= 0이면 0.
 *
 * NOTE: 이 함수가 반환하는 것은 "셀의 가로 폭"이지 "트랜지스터 채널 폭"이 아니다.
 * 두 개념은 직교(orthogonal): 채널 폭은 전류 용량을 결정하고, 셀 가로 폭은 레이아웃 공간을 결정한다.
 *
 * 계산 방법:
 * - 폴딩 횟수 = ceil(input_width / threshold_folding_width).
 * - 각 fold마다 poly 게이트(width_poly)와 좌우 콘택(spacing_poly_to_poly) 공간이 필요.
 * - 총 가로 폭 = 폴딩 수 × poly 폭 + (폴딩 수+1) × 콘택 간격.
 *
 * 호출 체인: height_sense_amplifier() → [compute_tr_width_after_folding()]
 */
double Component::compute_tr_width_after_folding(
    double input_width,
    double threshold_folding_width)
{//This is actually the width of the cell not the width of a device.
//The width of a cell and the width of a device is orthogonal.
  if (input_width <= 0) // [한국어] 채널 폭이 0 이하이면 가로 폭 없음
  {
    return 0; // [한국어] 유효하지 않은 입력 — 0 반환
  }

  int    num_folded_tr        = (int) (ceil(input_width / threshold_folding_width));
  // [한국어] 폴딩 횟수 = ceil(요구 채널 폭 / 폴딩 임계 폭). 채널을 몇 열로 접을지.
  double spacing_poly_to_poly = g_tp.w_poly_contact + 2 * g_tp.spacing_poly_to_contact;
  // [한국어] poly-poly 간격 = 콘택 폭 + 양측 poly-콘택 이격 × 2 (소스/드레인 콘택 공간).
  double width_poly           = g_ip->F_sz_um;
  // [한국어] poly 게이트 폭 = 공정 피처 사이즈(µm). 각 fold의 채널 방향 길이.
  double total_diff_width     = num_folded_tr * width_poly + (num_folded_tr + 1) * spacing_poly_to_poly;
  // [한국어] 총 확산 가로 폭:
  //   num_folded_tr × width_poly — 각 fold의 poly 폭 합산.
  //   (num_folded_tr+1) × spacing_poly_to_poly — poly 좌우(첫/마지막)와 사이의 콘택 공간.

  return total_diff_width; // [한국어] 폴딩 후 셀 가로 폭(µm) 반환
}



/*
 * [한국어]
 * height_sense_amplifier - 센스앰프(SA) 레이아웃 높이 계산
 *
 * @pitch_sense_amp: SA 피치(µm). 비트라인 1쌍(BL/BLB)이 차지하는 가로 간격.
 *                   폴딩 임계 폭으로 사용된다.
 * @return: SA 전체 레이아웃 높이 (µm). PMOS 영역 + NMOS 영역 + P-N 갭.
 *
 * CACTI 센스앰프는 2개의 cross-coupled PMOS(w_sense_p), 1개의 이솔레이션(isolation)
 * PMOS(w_iso), 2개의 cross-coupled NMOS(w_sense_n), 1개의 이네이블(enable) NMOS(w_sense_en)
 * 트랜지스터로 구성된다. 각 트랜지스터를 SA 피치(폴딩 임계)에 맞게 폴딩하고
 * 같은 타입(P-P, N-N) 사이의 최소 이격(MIN_GAP_BET_SAME_TYPE_DIFFS)을 더한다.
 *
 * 호출 체인: Subarray/Mat SA 레이아웃 → [height_sense_amplifier()]
 *              → compute_tr_width_after_folding()
 */
double Component::height_sense_amplifier(double pitch_sense_amp)
{
  // compute the height occupied by all PMOS transistors
  double h_pmos_tr = compute_tr_width_after_folding(g_tp.w_sense_p, pitch_sense_amp) * 2 +
                     compute_tr_width_after_folding(g_tp.w_iso, pitch_sense_amp) +
                     2 * g_tp.MIN_GAP_BET_SAME_TYPE_DIFFS;
  // [한국어] PMOS 영역 높이:
  //   w_sense_p를 pitch 기준 폴딩한 폭 × 2 — cross-coupled PMOS 2개.
  //   w_iso를 pitch 기준 폴딩한 폭 — 이솔레이션 PMOS 1개.
  //   MIN_GAP_BET_SAME_TYPE_DIFFS × 2 — P형 확산 영역 사이의 최소 이격 (이솔레이션 룰).

  // compute the height occupied by all NMOS transistors
  double h_nmos_tr = compute_tr_width_after_folding(g_tp.w_sense_n, pitch_sense_amp) * 2 +
                     compute_tr_width_after_folding(g_tp.w_sense_en, pitch_sense_amp) +
                     2 * g_tp.MIN_GAP_BET_SAME_TYPE_DIFFS;
  // [한국어] NMOS 영역 높이:
  //   w_sense_n를 pitch 기준 폴딩한 폭 × 2 — cross-coupled NMOS 2개.
  //   w_sense_en를 pitch 기준 폴딩한 폭 — 이네이블 NMOS 1개 (precharge/equalize 제어).
  //   MIN_GAP_BET_SAME_TYPE_DIFFS × 2 — N형 확산 영역 사이의 최소 이격.

  // compute total height by considering gap between the p and n diffusion areas
  return h_pmos_tr + h_nmos_tr + g_tp.MIN_GAP_BET_P_AND_N_DIFFS;
  // [한국어] 총 SA 높이 = PMOS 영역 높이 + NMOS 영역 높이 + P형/N형 확산 영역 사이 최소 이격.
}



/*
 * [한국어]
 * logical_effort - Logical Effort법 기반 버퍼 체인 최적화
 *
 * @num_gates_min:    최소 게이트 단 수 (결과를 이 값 미만으로 줄이지 않음).
 * @g:                논리 노력 계수 (게이트 타입별 고유값; 버퍼=1, NAND2≈4/3 등).
 * @F:                총 전기적 노력 = g × (최종 부하 / 입력 커패시턴스).
 * @w_n, @w_p:        출력 배열 — 각 게이트 단의 NMOS/PMOS 트랜지스터 폭(µm).
 *                    크기는 MAX_NUMBER_GATES_STAGE(=20) 이상이어야 함.
 * @C_load:           최종 출력 부하 커패시턴스(F).
 * @p_to_n_sz_ratio:  PMOS/NMOS 폭 비율 (통상 2.0 — 이동도 보상).
 * @is_dram_:         DRAM 공정 여부 (gate_C에 전달).
 * @is_wl_tr_:        워드라인 트랜지스터 여부 (gate_C에 전달).
 * @max_w_nmos:       최대 NMOS 폭 (µm). 초과 시 단 수를 늘려 분산.
 * @return:           최적 게이트 단 수 (짝수, num_gates_min 이상).
 *
 * Logical Effort(Sutherland, Sproull, Harris) 알고리즘:
 * 1. num_gates = floor(log(F) / log(fopt)), 짝수로 올림, num_gates_min 이상 보장.
 * 2. 각 단의 등가 팬아웃 f = F^(1/num_gates).
 * 3. 마지막 단(i=num_gates-1)의 NMOS 폭 = C_load/(f×gate_C(1)) 역산.
 * 4. max_w_nmos 초과 시 최대 폭으로 클램프하고 단 수 재계산.
 * 5. 역방향으로 w_n[i] = w_n[i+1]/f (i=num_gates-2 → 1).
 * fopt=4.0은 RC 지연 최소화 최적 팬아웃 상수 (const.h 정의).
 *
 * 호출 체인: Decoder/Subarray/Mat 버퍼 크기 결정 → [logical_effort()]
 */
int Component::logical_effort(
    int num_gates_min,
    double g,
    double F,
    double * w_n,
    double * w_p,
    double C_load,
    double p_to_n_sz_ratio,
    bool   is_dram_,
    bool   is_wl_tr_,
    double max_w_nmos)
{
  int num_gates = (int) (log(F) / log(fopt));
  // [한국어] 초기 게이트 단 수 = floor(log(F) / log(fopt)).
  //         fopt=4.0은 각 단의 최적 팬아웃 상수 (RC 지연 최소화 조건).

  // check if num_gates is odd. if so, add 1 to make it even
  num_gates+= (num_gates % 2) ? 1 : 0;
  // [한국어] 홀수이면 +1하여 짝수로 맞춤 — 인버터 체인은 짝수 단 수여야 비반전 출력을 얻음.
  num_gates = MAX(num_gates, num_gates_min);
  // [한국어] num_gates_min 이상 보장 — 설계자가 지정한 최소 버퍼 단 수 요구사항 충족.

  // recalculate the effective fanout of each stage
  double f = pow(F, 1.0 / num_gates);
  // [한국어] 각 단의 등가 팬아웃 = F^(1/num_gates) (균등 분산 가정).
  int    i = num_gates - 1; // [한국어] 마지막 단 인덱스 (역방향 계산 시작점)
  double C_in = C_load / f;
  // [한국어] 마지막 단의 입력 커패시턴스 = 부하 / 팬아웃. 역방향 계산의 기준.
  w_n[i]  = (1.0 / (1.0 + p_to_n_sz_ratio)) * C_in / gate_C(1, 0, is_dram_, false, is_wl_tr_);
  // [한국어] 마지막 단 NMOS 폭 역산:
  //   C_in = gate_C(w_n+w_p) = gate_C(w_n×(1+p_to_n_sz_ratio))
  //   → w_n = C_in / ((1+ratio) × gate_C(1µm)) 단위 게이트 커패시턴스로 정규화.
  w_n[i]  = MAX(w_n[i], g_tp.min_w_nmos_);
  // [한국어] 최소 NMOS 폭(min_w_nmos_) 이상 보장 — 공정 최소 규칙 준수.
  w_p[i]  = p_to_n_sz_ratio * w_n[i];
  // [한국어] PMOS 폭 = p_to_n_sz_ratio × NMOS 폭 (이동도 보상).

  if (w_n[i] > max_w_nmos) // [한국어] 마지막 단 NMOS 폭이 최대값 초과 → 단 수 추가하여 분산
  {
    double C_ld = gate_C((1 + p_to_n_sz_ratio) * max_w_nmos, 0, is_dram_, false, is_wl_tr_);
    // [한국어] 최대 폭 트랜지스터가 구동할 수 있는 부하 커패시턴스 계산.
    F = g * C_ld / gate_C(w_n[0] + w_p[0], 0, is_dram_, false, is_wl_tr_);
    // [한국어] max_w_nmos를 마지막 단으로 가정 시 나머지 단들이 분담해야 할 F 재계산.
    num_gates = (int) (log(F) / log(fopt)) + 1;
    // [한국어] 재계산된 F에 맞는 게이트 수 + 1 (마지막 단 max_w_nmos 고정이므로).
    num_gates+= (num_gates % 2) ? 1 : 0; // [한국어] 짝수 보장
    num_gates = MAX(num_gates, num_gates_min); // [한국어] 최소 단 수 보장
    f = pow(F, 1.0 / (num_gates - 1));
    // [한국어] 마지막 단을 제외한 (num_gates-1)개 단의 등가 팬아웃 재계산.
    i = num_gates - 1; // [한국어] 마지막 단 인덱스 갱신
    w_n[i]  = max_w_nmos; // [한국어] 마지막 단 NMOS 폭 = 최대 폭으로 고정
    w_p[i]  = p_to_n_sz_ratio * w_n[i]; // [한국어] 마지막 단 PMOS 폭 계산
  }

  for (i = num_gates - 2; i >= 1; i--) // [한국어] 마지막 단에서 첫 번째 단 방향으로 역방향 폭 계산
  {
    w_n[i] = MAX(w_n[i+1] / f, g_tp.min_w_nmos_);
    // [한국어] i번째 단 NMOS 폭 = 다음 단 폭 / 팬아웃 (역방향), 최소 폭 보장.
    w_p[i] = p_to_n_sz_ratio * w_n[i];
    // [한국어] i번째 단 PMOS 폭 = 비율 × NMOS 폭.
  }

  assert(num_gates <= MAX_NUMBER_GATES_STAGE);
  // [한국어] 게이트 단 수가 배열 크기(MAX_NUMBER_GATES_STAGE=20)를 초과하지 않는지 검증.
  return num_gates; // [한국어] 최적 게이트 단 수 반환
}

