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
 * [한국어 설명] CACTI 컴포넌트 기반 클래스 헤더 (component.h)
 *
 * === 파일의 역할 ===
 * CACTI 내 모든 하드웨어 컴포넌트(Bank, Mat, Subarray, Decoder, Wire, Arbiter, Crossbar
 * 등)가 상속하는 Component 기반 클래스를 선언한다. 이 클래스는 면적(area), 전력(power,
 * rt_power), 지연(delay), 사이클 타임(cycle_time)이라는 4가지 공통 속성을 제공한다.
 * 또한 레이아웃 면적(compute_gate_area), 트랜지스터 폴딩(compute_tr_width_after_folding),
 * 센스앰프 높이(height_sense_amplifier) 계산 및 논리 노력법(logical_effort) 기반의
 * 게이트 크기 산정 함수를 공통으로 구현한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * Component는 CACTI 컴포넌트 계층의 최상위 추상이다. 실행 흐름:
 * AccelWattch → cacti_interface → Bank → Mat → Subarray/Decoder 등.
 * 모든 컴포넌트가 Component를 상속하므로, 상위 컴포넌트는 하위 컴포넌트의
 * area.get_area(), power.readOp.*, delay 필드를 통일된 인터페이스로 읽는다.
 *
 * === 타 모듈과의 연결 ===
 * - parameter.h: g_ip, g_tp 전역 파라미터, powerDef 타입 포함.
 * - area.h: Area 클래스 — 폭/높이/면적 표현의 기본 타입.
 * - bank.h: Bank가 Component 상속 (forward 선언 포함).
 * - component.cc: compute_gate_area, compute_tr_width_after_folding, height_sense_amplifier,
 *   logical_effort의 실제 구현.
 * - basic_circuit.h: gate_C(), drain_C_() 등 — component.cc가 이를 사용.
 *
 * === 주요 함수/구조체 요약 ===
 * - compute_gate_area(): INV/NOR/NAND 게이트의 레이아웃 면적 계산 (트랜지스터 폴딩 포함).
 * - compute_tr_width_after_folding(): 폴딩 임계값 기반 확산 폭(diffusion width) 계산.
 * - height_sense_amplifier(): PMOS+NMOS 트랜지스터 폴딩 고려한 SA 높이 계산.
 * - logical_effort(): 논리 노력(Logical Effort)법으로 게이트 수·트랜지스터 폭 최적화.
 */

#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include "parameter.h" // [한국어] g_ip, g_tp 전역 파라미터, powerDef, DynamicParameter 등 CACTI 핵심 타입
#include "area.h"      // [한국어] Area 클래스 — 컴포넌트 레이아웃 면적(w, h, area) 표현

using namespace std; // [한국어] std 네임스페이스 전역 using (CACTI 코드베이스 관례)

class Crossbar; // [한국어] 크로스바 클래스 전방 선언 (component.h가 crossbar.h를 포함하지 않아 순환 의존 방지)
class Bank;     // [한국어] Bank 클래스 전방 선언 (component.h가 bank.h를 포함하지 않아 순환 의존 방지)

/*
 * [한국어]
 * Component - CACTI 하드웨어 컴포넌트의 기반 클래스
 *
 * Bank, Mat, Subarray, Decoder, Wire, MCPAT_Arbiter, Crossbar, Htree2 등 CACTI의
 * 모든 하드웨어 컴포넌트가 이 클래스를 상속한다. 공통 속성(area, power, delay,
 * cycle_time)과 레이아웃 계산 유틸리티 함수를 제공하여 코드 재사용성을 높인다.
 */
class Component
{
  public:
    /*
     * [한국어]
     * Component - 기본 생성자
     *
     * area, power, rt_power를 기본 초기화(0)하고 delay를 0으로 설정한다.
     * 모든 하위 컴포넌트 생성자가 암묵적으로 이를 호출한다.
     */
    Component();

    /*
     * [한국어]
     * ~Component - 소멸자
     *
     * 동적 할당 자원 없음. 하위 클래스 소멸자가 자동으로 이를 호출한다.
     */
    ~Component();

    Area area;
    /* area: 이 컴포넌트의 물리적 레이아웃 면적.
     * w, h가 설정되면 w×h를 면적으로 반환; 직접 지정 시 set_area() 사용.
     * 설정자: 각 컴포넌트 생성자/compute_area() 류 함수.
     * 읽는 자: 상위 컴포넌트가 면적 집계 시 area.get_area(), area.w, area.h 직접 접근.
     * 동기화: 단일 스레드 CACTI 계산; 별도 락 불필요. */

    powerDef power, rt_power;
    /* power: 이 컴포넌트의 정적 분석 전력 (readOp/writeOp/searchOp × dynamic/leakage/gate_leakage).
     *   설정자: compute_power_energy() 계열 함수 또는 누적 합산 코드.
     *   읽는 자: 상위 컴포넌트가 power.readOp.dynamic 등을 합산하여 총 전력 계산.
     * rt_power: 런타임 조건(실제 활동율 등)을 반영한 전력 (일부 컴포넌트에서 별도 계산).
     *   설정자: 런타임 파라미터 적용 시. 읽는 자: AccelWattch의 런타임 전력 추정.
     * 동기화: 단일 스레드 CACTI 계산. */

    double delay;
    /* delay: 이 컴포넌트의 신호 전파 지연(초).
     * 설정자: compute_delays() 계열 함수.
     * 읽는 자: 상위 컴포넌트가 지연 경로를 합산하여 총 접근 지연 계산.
     * 값 범위: 수백 ps ~ 수 ns (공정 및 캐시 구성에 따라 상이).
     * 동기화: 단일 스레드 CACTI 계산. */

    double cycle_time;
    /* cycle_time: 이 컴포넌트가 지원하는 최소 사이클 타임(초).
     * 파이프라인 단계 수(NUMBER_PIPELINE_STAGES)를 고려하여 설정된다.
     * 설정자: 상위 최적화 루프(find_optimal_conf)에서 delay를 기반으로 설정.
     * 읽는 자: CACTI 최적화 루프의 제약 조건 검사.
     * 동기화: 단일 스레드 CACTI 계산. */

    /*
     * [한국어]
     * compute_gate_area - INV/NOR/NAND 게이트의 레이아웃 면적 계산
     *
     * @gate_type:  게이트 종류 — INV(0), NOR(1), NAND(2) (const.h 정의).
     * @num_inputs: 게이트 입력 수 (NOR/NAND 팬인).
     * @w_pmos:     PMOS 트랜지스터 폭 (µm). 0 이하이면 0 반환.
     * @w_nmos:     NMOS 트랜지스터 폭 (µm). 0 이하이면 0 반환.
     * @h_gate:     게이트 셀 높이 (µm). 폴딩 임계값 결정에 사용.
     * @return:     게이트 레이아웃 면적 (µm²). 유효하지 않은 입력이면 0.0.
     *
     * PMOS/NMOS 트랜지스터를 셀 높이(h_gate) 내에 폴딩하여 배치하는 레이아웃을 가정하고,
     * 확산 영역(diffusion) 폭과 높이로부터 게이트 면적을 계산한다.
     * INV: NMOS/PMOS 각 1개. NOR: NMOS 병렬, PMOS 직렬. NAND: NMOS 직렬, PMOS 병렬.
     *
     * 호출 체인: Decoder/Subarray/Mat 등 → [compute_gate_area()] → compute_diffusion_width()
     */
    double compute_gate_area(
        int gate_type,
        int num_inputs,
        double w_pmos,
        double w_nmos,
        double h_gate);

    /*
     * [한국어]
     * compute_tr_width_after_folding - 폴딩 후 셀 가로 폭(확산 폭) 계산
     *
     * @input_width:           트랜지스터의 원하는 채널 폭 (µm).
     * @threshold_folding_width: 폴딩 임계 폭 (µm). 이보다 크면 폴딩 발생.
     * @return: 폴딩 후 확산 영역의 총 가로 폭 (µm). 셀 가로 크기에 해당.
     *
     * 큰 트랜지스터를 주어진 셀 높이 내에 여러 열로 접어 배치(폴딩)할 때
     * 필요한 확산 영역 가로 폭을 계산한다. input_width <= 0이면 0 반환.
     *
     * 호출 체인: height_sense_amplifier() → [compute_tr_width_after_folding()]
     */
    double compute_tr_width_after_folding(double input_width, double threshold_folding_width);

    /*
     * [한국어]
     * height_sense_amplifier - 센스앰프(SA) 레이아웃 높이 계산
     *
     * @pitch_sense_amp: SA 피치(µm). 한 SA가 차지하는 비트라인 간격.
     * @return: PMOS + NMOS 트랜지스터 영역 + P-N 갭을 합산한 SA 높이 (µm).
     *
     * g_tp.w_sense_p(×2), g_tp.w_iso, g_tp.w_sense_n(×2), g_tp.w_sense_en 트랜지스터를
     * 피치 기반으로 폴딩한 뒤 PMOS 영역 높이 + NMOS 영역 높이 + P-N 갭을 합산한다.
     *
     * 호출 체인: Subarray/Mat SA 레이아웃 계산 → [height_sense_amplifier()]
     */
    double height_sense_amplifier(double pitch_sense_amp);

  protected:
    /*
     * [한국어]
     * logical_effort - 논리 노력(Logical Effort)법 기반 게이트 수·트랜지스터 폭 산정
     *
     * @num_gates_min:    최소 게이트 단 수 (홀수이면 +1 하여 짝수로 맞춤).
     * @g:                논리 노력 계수 (게이트 타입별 고유값).
     * @F:                총 전기적 노력 (= 게이트 타입 노력 × 최종 부하 / 입력 커패시턴스).
     * @w_n, @w_p:        각 게이트 단의 NMOS/PMOS 폭 배열 (출력, 크기 MAX_NUMBER_GATES_STAGE).
     * @C_load:           최종 출력 부하 커패시턴스 (F).
     * @p_to_n_sz_ratio:  PMOS/NMOS 폭 비율 (통상 2.0).
     * @is_dram_:         DRAM 공정 여부 (gate_C 함수에 전달).
     * @is_wl_tr_:        워드라인 트랜지스터 여부 (gate_C 함수에 전달).
     * @max_w_nmos:       최대 NMOS 폭 (µm). 초과 시 게이트 수를 늘려 분산.
     * @return:           최적 게이트 단 수 (짝수, num_gates_min 이상).
     *
     * Logical Effort 방법론으로 버퍼/인버터 체인의 최적 단 수(num_gates)를 구하고,
     * 각 단의 NMOS/PMOS 폭(w_n[], w_p[])을 역방향으로 계산한다.
     * max_w_nmos 초과 시 단 수를 추가하고 재계산한다.
     * fopt=4.0 (const.h)이 최적 팬아웃 상수로 사용된다.
     *
     * 호출 체인: Decoder/Subarray/Mat 버퍼 크기 결정 → [logical_effort()]
     */
    int logical_effort(
        int    num_gates_min,
        double g,
        double F,
        double * w_n,
        double * w_p,
        double C_load,
        double p_to_n_sz_ratio,
        bool   is_dram_,
        bool   is_wl_tr_,
        double max_w_nmos);

  private:
    /*
     * [한국어]
     * compute_diffusion_width - 스택/폴딩된 트랜지스터의 확산 영역 가로 폭 계산
     *
     * @num_stacked_in:  직렬 스택 트랜지스터 수 (NOR: 1, NAND: num_inputs).
     * @num_folded_tr:   폴딩 횟수 (트랜지스터를 여러 열로 접은 수).
     * @return:          확산 영역의 총 가로 폭(µm). 게이트 면적 계산의 핵심 입력.
     *
     * poly 피치, 콘택 간격, poly-간격을 고려하여 레이아웃 면적을 계산하는 내부 헬퍼.
     * compute_gate_area()에서만 호출된다.
     *
     * 호출 체인: compute_gate_area() → [compute_diffusion_width()]
     */
    double compute_diffusion_width(int num_stacked_in, int num_folded_tr);
};

#endif

