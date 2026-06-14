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
 * [한국어 설명] 반복기(Repeater) 삽입 글로벌 와이어 전력·지연·면적 모델 헤더 (wire.h)
 *
 * === 파일의 역할 ===
 * 칩 내 배선(wire)에 최적 반복기(repeater/buffer)를 삽입했을 때의 지연·전력·면적을 계산한다.
 * 배선 타입(Global, Global_5/10/20/30%, Low_swing)과 배치(outside_mat, inside_mat, local)에 따라
 * 커패시턴스·저항을 계산하고, 최소 지연 또는 지연 페널티 허용 범위 내 최적 반복기 조합을 도출한다.
 * NUCA 뱅크 간 배선 전력 계산(sim_nuca)과 H-tree 링크 모델(htree2)에서 광범위하게 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * cacti_interface() → Wire::Wire(wt, len) → [이 모듈] → power/delay/area 제공
 * 정적 멤버(global, global_5, ..., low_swing)는 Wire(init) 생성자로 한 번 초기화된 뒤,
 * 이후 모든 Wire 인스턴스에서 배선 타입별 기준값으로 참조된다.
 * 호스트 유저스페이스, 단일/멀티 스레드 모두에서 사용된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존: basic_circuit(gate_C, drain_C_, tr_R_on, horowitz), parameter.h(g_tp, g_ip),
 *       component.h(Component — power/delay/area 기반), cacti_interface.h
 * 소비: Nuca::sim_nuca(), Htree2, MCPAT_Router::Cw3(), HighRadix 등이 Wire를 생성해 사용
 * 공유: g_tp.wire_outside_mat/inside_mat/wire_local (배선 파라미터 구조체)
 *
 * === 주요 함수/구조체 요약 ===
 * Wire(wt, len, ...)       : 주 생성자 — 배선 통계를 계산하고 단위 환산 수행
 * Wire(w_s, s_s, ...)      : 초기화 전용 생성자 — 정적 멤버(global, low_swing 등) 초기화
 * calculate_wire_stats()   : 배선 타입에 따라 delay/power/area 결정
 * delay_optimal_wire()     : 지연 최소 반복기 간격·크기 계산
 * wire_cap(len)            : 측벽·인접 층 커패시턴스 합산한 총 배선 커패시턴스 [F]
 * low_swing_model()        : 저전압 스윙 배선의 트랜스미터+배선+감지증폭기 합산 지연·전력
 * global, global_5, ...    : 배선 타입별 기준 지연·전력 정적 캐시 (단위: per meter)
 */

#ifndef __WIRE_H__
#define __WIRE_H__

#include "basic_circuit.h"
#include "component.h"
#include "parameter.h"
#include "assert.h"
#include "cacti_interface.h"
#include <iostream>
#include <list>

class Wire : public Component
{
  public:
    Wire(enum Wire_type wire_model, double len = 0/* in u*/,
         int nsense = 1/* no. of sense amps connected to the low-swing wire */,
         double width_scaling = 1,
         double spacing_scaling = 1,
         enum Wire_placement wire_placement = outside_mat,
         double resistivity = CU_RESISTIVITY,
         TechnologyParameter::DeviceType *dt = &(g_tp.peri_global));
    ~Wire();

    Wire( double width_scaling = 1,
         double spacing_scaling = 1,
         enum Wire_placement wire_placement = outside_mat,
         double resistivity = CU_RESISTIVITY,
         TechnologyParameter::DeviceType *dt = &(g_tp.peri_global)
    ); // should be used only once for initializing static members
    void init_wire();

    void calculate_wire_stats();
    void delay_optimal_wire();
    double wire_cap(double len, bool call_from_outside=false);
    double wire_res(double len);
    void low_swing_model();
    double signal_fall_time();
    double signal_rise_time();
    double sense_amp_input_cap();

    enum Wire_type wt;
    /* [한국어] 이 배선 인스턴스의 타입 (Global/Global_5/.../Low_swing).
     * 설정자: 주 생성자에서 wire_model 인자로 초기화.
     * 읽는 자: calculate_wire_stats()에서 분기 선택; print_nuca()에서 타입 문자열 출력.
     * 동기화: 불변. */

    double wire_spacing;
    /* [한국어] 와이어 간격 [um] — 계산 완료 후 단위 변환 (m → um).
     * 설정자: calculate_wire_stats()에서 pitch × s_scale × 1e6 / 2로 설정; 생성자 끝에서 ×1e6 변환.
     * 읽는 자: wire_cap()에서 측벽 커패시턴스(sidewall) 계산의 분모로 사용.
     * 동기화: 불변 (설정 후 wire_cap 호출 전 단위 재변환). */

    double wire_width;
    /* [한국어] 와이어 폭 [um] — 계산 완료 후 단위 변환 (m → um).
     * 설정자: calculate_wire_stats()에서 pitch × w_scale × 1e6 / 2로 설정.
     * 읽는 자: wire_cap()에서 와이어 높이(wire_height = wire_width/w_scale × aspect_ratio) 계산.
     * 동기화: 불변. */

    enum Wire_placement wire_placement;
    /* [한국어] 와이어 배치 위치 (outside_mat / inside_mat / 기타=local).
     * 설정자: 생성자 wp 인자 또는 Wire(w_s,s_s,...) 초기화 생성자에서 설정.
     * 읽는 자: calculate_wire_stats(), wire_cap(), wire_res()에서 피치/유전율 상수 선택에 사용.
     * 동기화: 불변. */

    double repeater_size;
    /* [한국어] 최적 반복기 크기 (최소 크기 대비 배율) [무차원].
     * 설정자: delay_optimal_wire()에서 repeater_scaling = sqrt(out_res×wc/(wr×input_cap))으로 계산.
     *         calculate_wire_stats()에서 정적 멤버의 area.h 값에서 복사.
     * 읽는 자: area 계산 (repeater 수 × gate_area × repeater_size), print_wire()에서 출력.
     * 동기화: 불변 (설정 후 um 단위로 변환 없이 사용됨). */

    double repeater_spacing;
    /* [한국어] 최적 반복기 간격 [um] — 생성 후 m에서 um으로 단위 변환.
     * 설정자: delay_optimal_wire()에서 반복기 간격 공식으로 계산 후 ×1e6 변환.
     *         calculate_wire_stats()에서 정적 멤버의 area.w [m]에서 복사 후 ×1e6.
     * 읽는 자: 배선 단위 길이당 반복기 수(wire_length / repeater_spacing)로 면적·전력 계산.
     * 동기화: 불변. */

    double wire_length;
    /* [한국어] 배선 길이 [um] — 생성 후 m에서 um으로 단위 변환.
     * 설정자: 주 생성자 wl 인자 × 1e-6(m 변환) 후 계산, 최종적으로 ×1e6 복원.
     * 읽는 자: delay/power 계산 시 wire_length × per-meter 값으로 실제 지연·전력 산출.
     * 동기화: 불변. */

    double in_rise_time, out_rise_time;
    /* [한국어] 입력 라이즈 타임(in)과 출력 라이즈 타임(out) [s].
     * in_rise_time : 이전 게이트의 출력 라이즈 타임 — 초기값 0 (이상적 스텝 입력).
     * out_rise_time: 배선+반복기 통과 후의 라이즈 타임 — 다음 게이트 지연 계산에 전달.
     * 설정자: out_rise_time은 delay_optimal_wire()와 low_swing_model()에서 설정.
     *         in_rise_time은 set_in_rise_time()으로 외부에서 설정 가능.
     * 동기화: 단일 스레드 내 파이프라인 계산에서 순차 사용. */

    void set_in_rise_time(double rt)
    {
      in_rise_time = rt; // [한국어] 입력 라이즈 타임 설정 — 다음 계산에서 이 값이 in_rise_time으로 사용됨
    }
    static Component global;
    /* [한국어] Global 타입 배선의 단위 길이당(per-m) 지연·전력·면적 기준값 (정적 캐시).
     * 설정자: Wire(초기화 생성자) → init_wire() → delay_optimal_wire() 결과를 저장.
     * 읽는 자: calculate_wire_stats()에서 wt==Global일 때 global.delay × wire_length로 배선 지연 계산.
     * 동기화: Wire::initialized==1 이후에만 읽을 것; 초기화는 단일 스레드에서 한 번 수행. */

    static Component global_5;
    /* [한국어] 5% 지연 페널티 허용 Global 배선의 기준값 — update_fullswing()에서 설정. */
    static Component global_10;
    /* [한국어] 10% 지연 페널티 허용 Global 배선의 기준값. */
    static Component global_20;
    /* [한국어] 20% 지연 페널티 허용 Global 배선의 기준값. */
    static Component global_30;
    /* [한국어] 30% 지연 페널티 허용 Global 배선의 기준값. */
    static Component low_swing;
    /* [한국어] Low-swing 차동 배선 1mm 기준 지연·전력 — low_swing_model()로 계산.
     * 주의: 반복 배선과 달리 길이에 선형 비례하지 않음 (트랜스미터/감지증폭기 고정 비용 포함). */

    static double wire_width_init;
    /* [한국어] 초기화 시 설정된 wire_width [m] — print_wire()에서 um 변환 후 출력용. */
    static double wire_spacing_init;
    /* [한국어] 초기화 시 설정된 wire_spacing [m] — print_wire()에서 um 변환 후 출력용. */
    void print_wire();

  private:

    int nsense; // no. of sense amps connected to a low-swing wire if it
                // is broadcasting data to multiple destinations
    // width and spacing scaling factor can be used
    // to model low level wires or special
    // fat wires
    double w_scale, s_scale;
    double resistivity;
    powerDef wire_model (double space, double size, double *delay);
    list <Component> repeated_wire;
    void update_fullswing();
    static int initialized;


    //low-swing
    Component transmitter;
    Component l_wire;
    Component sense_amp;

    double min_w_pmos;

    TechnologyParameter::DeviceType *deviceType;

};

#endif
