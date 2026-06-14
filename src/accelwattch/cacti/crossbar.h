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
 * [한국어 설명] CACTI 크로스바(Crossbar) 모델 선언 (crossbar.h)
 *
 * === 파일의 역할 ===
 * NoC(Network-on-Chip) 또는 캐시 내부의 크로스바 스위치(crossbar switch)를 모델링하는
 * Crossbar 클래스를 선언한다. 크로스바는 n_inp개 입력과 n_out개 출력을 연결하는
 * 스위칭 패브릭으로, 각 교차점에 트라이스테이트(tri-state) 버퍼를 배치한다.
 * 전력(동적/누설/게이트누설), 면적, 지연을 compute_power() 한 번으로 계산한다.
 * CACTI에서 CAM이나 다중 포트 캐시의 스위칭 오버헤드를 추정할 때 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch → CACTI → Crossbar (캐시 내부 스위칭 패브릭 전력 모델).
 * Crossbar는 Component를 상속하여 area/power/delay 필드를 보유.
 * 호출 체인: uca_org_t 또는 CAM 전력 모델 → [Crossbar::compute_power()] → 결과 반환.
 * 실행 컨텍스트: 호스트 유저스페이스 — GPU 시뮬레이션 전 초기화.
 *
 * === 타 모듈과의 연결 ===
 * 의존: basic_circuit.h (gate_C, drain_C_, tr_R_on, horowitz, cmos_Isub_leakage),
 *   wire.h (Wire 클래스 — 배선 RC 파라미터), component.h (Component 기반 클래스),
 *   parameter.h (g_tp 전역 기술 파라미터).
 * 이 파일에 의존: CACTI 상위 분석 모듈 (uca_org_t, CAM 분석 등).
 *
 * === 주요 함수/구조체 요약 ===
 * Crossbar()      : 생성자 — n_inp, n_out, flit_size, deviceType 초기화
 * output_buffer() : 크로스바 한 교차점의 입출력 커패시턴스 계산 및 트라이스테이트 크기 결정
 * compute_power() : 동적 전력, 누설 전력, 지연을 한 번에 계산 (내부에서 종횡비 조정 반복)
 * print_crossbar(): 결과 요약 출력 (디버깅/검증용)
 */

#ifndef __CROSSBAR__
#define __CROSSBAR__

#include <assert.h>  // [한국어] assert() — 입력값 유효성 검사
#include <iostream>  // [한국어] print_crossbar()의 cout 출력
#include "basic_circuit.h" // [한국어] gate_C, drain_C_, tr_R_on, horowitz, 누설전류 함수
#include "cacti_interface.h" // [한국어] powerDef 등 공통 전력 자료구조
#include "component.h" // [한국어] Component 기반 클래스 (area, power, delay 필드 보유)
#include "parameter.h" // [한국어] g_tp (TechnologyParameter 전역 인스턴스)
#include "mat.h"       // [한국어] MAT(Memory Array Tile) — 어레이 파라미터 의존
#include "wire.h"      // [한국어] Wire 클래스 — 배선 지연/전력 계산

/*
 * [한국어] Crossbar — 크로스바 스위치 전력/면적/지연 모델
 * n_inp × n_out 크기의 크로스바를 트라이스테이트 버퍼 배열로 구현.
 * 각 입력 신호는 flit_size 비트 폭으로 n_out개 출력에 선택적으로 연결된다.
 */
class Crossbar : public Component
{
  public:
    /*
     * [한국어]
     * Crossbar 생성자 — 크로스바 파라미터 초기화
     * @in      : 입력 포트 수 (n_inp)
     * @out     : 출력 포트 수 (n_out)
     * @flit_sz : 플릿(flit) 크기 (비트) — 한 번에 스위칭하는 데이터 폭
     * @dt      : 트랜지스터 파라미터 포인터 (기본값: g_tp.peri_global)
     * 생성 시 min_w_pmos, Vdd, CB_ADJ를 초기화하고 compute_power()를 통해 탐색 준비.
     * 호출 체인: CACTI 크로스바 분석 코드 → [Crossbar()] → compute_power()
     */
    Crossbar(
      double in,
      double out,
      double flit_sz,
      TechnologyParameter::DeviceType *dt = &(g_tp.peri_global));
    ~Crossbar(); // [한국어] 소멸자 — 동적 할당 자원 없으므로 빈 구현

    /* [한국어]
     * print_crossbar - 크로스바 분석 결과를 표준 출력으로 출력
     * 면적(w×h), 동적 전력, 누설 전력, 게이트 누설, 지연을 출력.
     * 호출 체인: 결과 검증 시 → [print_crossbar]
     */
    void print_crossbar();

    /* [한국어]
     * output_buffer - 트라이스테이트 버퍼의 입출력/제어 커패시턴스 계산 및 크기 결정
     * @return : 교차점 하나의 총 커패시턴스 (F) = input_cap + output_cap + ctr_cap
     * 이 함수는 compute_power()에서 호출되어 TriS1(제어 게이트), TriS2(드라이버) 크기를 결정.
     * 와이어 RC 파라미터로부터 반복기(repeater) 크기를 구하고 트라이스테이트 구동력을 결정.
     * 호출 체인: compute_power → [output_buffer] → Wire, gate_C, drain_C_
     */
    double output_buffer();

    /* [한국어]
     * compute_power - 크로스바 전체 동적/누설 전력 및 지연 계산
     * 내부적으로 output_buffer()를 호출하여 커패시턴스를 구하고,
     * 트라이스테이트 버퍼 셀의 면적을 계산한 뒤,
     * 종횡비(aspect ratio)가 ASPECT_THRESHOLD(0.8) 미만이면 CB_ADJ를 키워 재귀 호출.
     * 동적 전력 = (와이어 전력 + 커패시턴스 충전) × flit_size
     * 누설 전력 = n_inp × n_out × flit_size × (INV+NAND+NOR 누설 + 와이어 누설)
     * 지연 = horowitz()로 드라이버 + 와이어 RC 조합 계산
     * 호출 체인: CACTI 크로스바 분석 → [compute_power] → output_buffer, Wire, horowitz
     */
    void compute_power();

    double n_inp, n_out; // [한국어] 입력/출력 포트 수
    double flit_size;    // [한국어] 플릿(flit) 크기 (비트) — 스위칭 데이터 폭
    double tri_inp_cap;  // [한국어] 트라이스테이트 버퍼 입력 커패시턴스 (F) — output_buffer()가 설정
    double tri_out_cap;  // [한국어] 트라이스테이트 버퍼 출력 커패시턴스 (F)
    double tri_ctr_cap;  // [한국어] 트라이스테이트 버퍼 제어 게이트 커패시턴스 (F)
    double tri_int_cap;  // [한국어] 트라이스테이트 버퍼 내부 노드 커패시턴스 (F)

  private:
	  double CB_ADJ;
	  /* 크로스바 교차점 셀 높이 조정 인수 (Crossbar cell height ADJust factor)
	   * 설정자: compute_power()에서 종횡비가 ASPECT_THRESHOLD 미만이면 0.2씩 증가 (최대 4).
	   * 읽는 자: 트라이스테이트 셀 높이 = CB_ADJ * g_tp.cell_h_def 계산 시.
	   * 값 범위: 1.0 초기, 0.2씩 증가하여 최대 4.0.
	   * 동기화: compute_power() 재귀 호출 내에서만 변경됨.
	   * 목적: 전체 크로스바의 종횡비(height/width)를 1에 가깝게 조정하여 배선 효율 개선.
	   * CB_ADJ 증가 시 트라이스테이트 셀이 더 높고 얇아져 크로스바 전체 높이가 증가. */

	TechnologyParameter::DeviceType *deviceType;
	/* 트라이스테이트 버퍼 트랜지스터 파라미터 포인터
	 * 설정자: 생성자에서 dt 파라미터로 초기화.
	 * 읽는 자: compute_power/output_buffer에서 Vdd, n_to_p_eff_curr_drv_ratio 등 참조.
	 * 값 범위: g_tp.peri_global 포인터 (기본값) 또는 사용자 지정.
	 * 동기화: 초기화 후 읽기 전용. */

    double TriS1, TriS2;
    /* 트라이스테이트 버퍼 게이트 크기 배율
     * TriS1: NAND+NOR 제어 게이트의 최소 폭 배율 (output_buffer()에서 계산)
     * TriS2: 출력 드라이버(인버터) 최소 폭 배율
     * 설정자: output_buffer()에서 반복기(repeater) 크기를 기반으로 계산.
     * 읽는 자: compute_power()에서 gate_C, drain_C_, cmos_Isub_leakage에 전달.
     * 값 범위: 1 이상 (TriS1 < 1이면 1로 클램핑). */

    double min_w_pmos, Vdd;
    /* min_w_pmos: 최소 PMOS 폭 = n_to_p_eff_curr_drv_ratio * g_tp.min_w_nmos_
     *   PMOS가 NMOS보다 이동도가 낮아 동등 구동력을 위해 더 넓어야 함.
     * Vdd: 공급 전압 (V) — deviceType->Vdd
     *   설정자: 생성자에서 초기화.
     *   읽는 자: compute_power()에서 동적 에너지(C*Vdd²) 계산에 사용. */

};




#endif
