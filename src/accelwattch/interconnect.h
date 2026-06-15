/*****************************************************************************
 *                                McPAT
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
 * [한국어 설명] AccelWattch 인터커넥트(금속 배선) 전력 모델 헤더 (interconnect.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPU SM 내부의 금속 배선(metal wire) 전력 모델 클래스 `interconnect`를
 * 선언한다. 레지스터 파일과 기능 유닛(ALU/FPU) 사이의 bypass 버스, 또는 파이프라인
 * 스테이지 간 데이터 전달 버스 등의 전력을 McPAT Wire 모델을 사용하여 추정한다.
 * 배선 길이(length), 배선 종류(wt), 데이터 폭(data_width), 파이프라이닝 여부
 * (pipelinable) 등을 입력받아 CACTI Wire 클래스를 호출해 비트당 전력(power_bit)을
 * 계산한다. 생성자에서 지연 요건을 충족하도록 배선 폭(width_scaling)을 자동 최적화한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델 계층:
 *   gpgpu_sim_wrapper (gpgpu_sim_wrapper.cc)
 *     → Core 레벨 컴포넌트 (EXECU, RegU 등)
 *         → interconnect (이 파일) — bypass 버스·로컬 배선 전력
 *             → CACTI Wire 모델 (cacti/wire.h) — RC 배선 지연·에너지 계산
 * 실행 컨텍스트: 시뮬레이션 초기화 시 한 번 생성되고, 이후 compute()가 호출되어
 * power_bit를 data_width로 스케일링한 Component::power를 확정한다.
 * 이 값은 gpgpu_sim_wrapper에서 접근 횟수에 곱해져 총 배선 동적 전력을 산출한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈: CACTI Wire(cacti/wire.h) — RC 배선 모델;
 *   basic_components.h — Component 베이스, powerDef, Area, Device_ty;
 *   parameter.h — InputParameter, TechnologyParameter;
 *   subarray.h — Area 구조체 참조.
 * 이 파일에 의존하는 모듈: logic.cc, array.cc 등 AccelWattch 컴포넌트가 배선 전력
 * 추정을 위해 interconnect 객체를 생성한다; gpgpu_sim_wrapper가 총 배선 전력 합산.
 * 공유 자료구조: Component::power — 계산 결과 저장; g_tp(TechnologyParameter) — 공정 전역 상수.
 *
 * === 주요 함수/구조체 요약 ===
 * interconnect (생성자) : 배선 파라미터를 초기화하고 width_scaling을 자동 최적화.
 * compute()             : CACTI Wire 모델을 호출하여 power_bit와 Component::power 확정.
 * leakage_feedback()    : 온도 변화에 따른 배선 누설 전력 재계산.
 * set_in_rise_time()    : 입력 신호 상승 시간(slew)을 설정 — 타이밍 체인 연결용.
 * power_bit             : 비트당 동적+누설 전력 (data_width 스케일링 전 값).
 */
#ifndef __INTERCONNECT_H__
#define __INTERCONNECT_H__

#include "assert.h"               /* [한국어] assert() 매크로 — 배선 파라미터 유효성 검사 (길이 > 0, 데이터폭 > 0 등) */
#include "basic_components.h"     /* [한국어] Component 베이스 클래스, powerDef, Area, Device_ty, Core_type 등 공통 자료구조 */
#include "cacti/basic_circuit.h"  /* [한국어] 트랜지스터 게이트/드레인 커패시턴스 계산 함수 (gate_C, drain_C 등) */
#include "cacti/cacti_interface.h" /* [한국어] CACTI UCA 결과 구조체(uca_org_t) 및 주 인터페이스 — 누설 전력 저장소로 사용 */
#include "cacti/component.h"      /* [한국어] Component 베이스 클래스 정의 — power 필드(powerDef) 상속 */
#include "cacti/parameter.h"      /* [한국어] InputParameter(CACTI 입력), TechnologyParameter(공정 전역 파라미터 g_tp) 정의 */
#include "cacti/subarray.h"       /* [한국어] Area 구조체 정의 — no_device_under_wire_area 필드 타입으로 사용 */
#include "cacti/wire.h"           /* [한국어] Wire 클래스 — RC 배선 지연·에너지 모델의 핵심; compute()에서 직접 호출 */

// leakge power includes entire htree in a bank (when uca_tree == false)
// leakge power includes only part to one bank when uca_tree == true
/* [한국어] 위 영어 주석: uca_tree 옵션에 따라 누설 전력 범위가 다름을 명시.
 * interconnect 클래스에서는 직접 해당하지 않으나 CACTI 맥락에서 참조 정보로 유지. */

/*
 * [한국어]
 * interconnect — GPU SM 내부 금속 배선의 전력 모델
 *
 * 이 클래스는 레지스터 파일 ↔ 기능 유닛 사이의 bypass 버스 또는 SM 내부 데이터
 * 전달 배선의 전력을 McPAT Wire 모델로 추정한다. 배선은 지연(latency)과 대역폭
 * (throughput) 두 가지 제약을 가지며, pipelinable=true이면 대역폭 최적화,
 * false이면 지연 최적화 모드로 Wire 모델을 호출한다.
 * 생성자에서 width_scaling을 1.0 → 2.0 → 3.0으로 자동 증가시켜 지연/대역폭
 * 요건을 충족하는 최소 배선 폭을 탐색한다.
 * 최종적으로 compute()가 CACTI Wire::compute()를 호출하고, power_bit를 data_width로
 * 스케일링하여 Component::power를 확정한다.
 */
class interconnect : public Component {
 public:
  interconnect(string name_, enum Device_ty device_ty_, double base_w = 0,
               double base_h = 0, int data_w = 0, double len = 0,
               const InputParameter *configure_interface = NULL,
               int start_wiring_level_ = 0, bool pipelinable_ = false,
               double route_over_perc_ = 0.5, bool opt_local_ = true,
               enum Core_type core_ty_ = Inorder,
               enum Wire_type wire_model = Global, double width_s = 1.0,
               double space_s = 1.0,
               TechnologyParameter::DeviceType *dt = &(g_tp.peri_global));
  /* [한국어] 생성자 — 배선 파라미터를 초기화하고 width_scaling을 자동 최적화.
   * name_           : 배선 식별 이름 (디버깅/보고용).
   * device_ty_      : 회로 공정 유형 (Core_device 등).
   * base_w, base_h  : 배선이 놓이는 기반 블록의 폭·높이 (m) — 배선 길이 추정 기반.
   * data_w          : 버스 데이터 폭 (비트) — power_bit × data_w = 총 전력.
   * len             : 배선 총 길이 (m).
   * configure_interface: CACTI 공정 파라미터 포인터.
   * start_wiring_level_: 배선이 시작하는 금속 레이어 번호.
   * pipelinable_    : true이면 처리량(throughput) 최적화; false이면 지연(latency) 최적화.
   * route_over_perc_: 소자 위(over-device) 배선 비율 (0.0~1.0); 면적 계산에 영향.
   * opt_local_      : true이면 로컬(최소 폭) 배선 최적화 모드.
   * core_ty_        : 코어 유형 (Inorder / OOO).
   * wire_model      : 배선 종류 (Global / Semi-global 등 Wire_type enum).
   * width_s, space_s: 초기 배선 폭·간격 스케일링 인자 (1.0 = 최소 공정 폭).
   * dt              : 소자 공정 파라미터 포인터 (기본값: 주변 회로 g_tp.peri_global). */

  ~interconnect(){};
  /* [한국어] 소멸자 — 동적 할당 멤버 없으므로 기본 소멸자로 충분. */

  void compute();
  /* [한국어] CACTI Wire 모델을 호출하여 배선의 비트당 동적·누설 전력을 계산.
   * 내부적으로 Wire 객체를 생성하고 Wire::compute()를 호출한 뒤,
   * power_bit = 단일 비트 에너지로 설정하고 Component::power = power_bit × data_width로 스케일링.
   * 또한 no_device_under_wire_area에 배선 전용 면적을 저장한다. */

  string name;
  /* [한국어] 배선 식별 이름 (예: "bypass_bus", "read_bus" 등).
   * 설정자: 생성자 인자 name_으로 초기화.
   * 읽는 자: 디버깅 출력 및 에너지 보고 시 배선 구분용.
   * 값 범위: 임의 문자열.
   * 동기화: 생성 후 불변. */

  enum Device_ty device_ty;
  /* [한국어] 배선이 놓이는 회로 공정 유형 (Core_device / Uncore_device 등).
   * 설정자: 생성자 인자 device_ty_로 초기화.
   * 읽는 자: compute() 내부 Wire 생성 시 공정별 RC 파라미터 선택에 사용.
   * 값 범위: Device_ty enum.
   * 동기화: 생성 후 불변. */

  double in_rise_time, out_rise_time;
  /* [한국어] 입력/출력 신호 상승 시간(slew rate, 단위: 초) — 타이밍 체인 연결용.
   * in_rise_time : 이 배선의 입력 드라이버 출력 슬루율. 이전 단계의 out_rise_time.
   * out_rise_time: 이 배선이 다음 단계에 제공하는 출력 슬루율; compute() 후 채워짐.
   * 설정자: in_rise_time은 set_in_rise_time()으로 설정; out_rise_time은 compute() 후.
   * 읽는 자: Wire 모델이 타이밍 최적화 시 slew를 전파하는 데 참조.
   * 값 범위: 0 이상의 실수 (ps ~ ns 범위).
   * 동기화: 단일 스레드 초기화 흐름. */

  InputParameter l_ip;
  /* [한국어] CACTI 입력 파라미터 — 공정 노드, Vdd, 온도 등.
   * 설정자: 생성자에서 configure_interface 복사.
   * 읽는 자: compute() 내부 Wire 객체 생성 시 전달.
   * 값 범위: 유효 CACTI InputParameter.
   * 동기화: 생성 후 불변 (leakage_feedback만 온도 수정). */

  uca_org_t local_result;
  /* [한국어] CACTI UCA 결과 구조체 — 배선 누설 전력 및 면적 저장용.
   * 설정자: compute() 내부에서 Wire 결과로부터 채워짐.
   * 읽는 자: Component::power에 누설·게이트누설 전력을 옮길 때 참조.
   * 값 범위: CACTI uca_org_t 출력; 배선 전용 필드만 유효.
   * 동기화: compute() 후 불변. */

  Area no_device_under_wire_area;
  /* [한국어] 소자 없이 배선만 점유하는 면적 (단위: m²) — route_over_perc 적용 결과.
   * 배선이 소자 위를 지나는 비율(route_over_perc)에 따라 실제 면적이 감소한다.
   * 설정자: compute() 내부에서 Wire 면적 계산 결과를 저장.
   * 읽는 자: 상위 레벨 면적 집계(floor-planning) 시 참조.
   * 값 범위: 0 이상의 면적 구조체.
   * 동기화: compute() 후 불변. */

  void set_in_rise_time(double rt) { in_rise_time = rt; }
  /* [한국어] 입력 신호 상승 시간을 설정 — 멀티 스테이지 배선 타이밍 체인에서
   * 이전 Wire의 out_rise_time을 이 배선의 in_rise_time으로 연결할 때 호출. */

  void leakage_feedback(double temperature);
  /* [한국어] 온도 변화에 따른 배선 누설 전력 재계산.
   * temperature: 회로 온도(섭씨). 온도-누설 모델(CACTI Wire 내부)을 재호출한다. */

  double max_unpipelined_link_delay;
  /* [한국어] 파이프라이닝 없이 허용되는 최대 배선 지연 (단위: 초).
   * 설정자: 생성자에서 클럭 주기와 배선 구성으로부터 계산.
   * 읽는 자: 생성자 내 width_scaling 자동 최적화 루프에서 지연 제약 비교 기준.
   * 값 범위: 0 초과 (클럭 주기의 일부 또는 전체).
   * 동기화: 생성 후 불변. */

  powerDef power_bit;
  /* [한국어] 비트당 배선 전력 (data_width 스케일링 전 단위 값, 단위: W).
   * 설정자: compute() 내부에서 Wire 모델 결과의 단위 비트 전력을 저장.
   * 읽는 자: Component::power = power_bit × data_width 계산 시 참조;
   *          gpgpu_sim_wrapper가 접근 횟수에 곱해 동적 전력 누적.
   * 값 범위: 0 이상의 실수.
   * 동기화: compute() 후 불변. */

  double wire_bw;
  /* [한국어] 현재 계산에 사용 중인 유효 배선 대역폭 (비트/초).
   * 설정자: compute() 내부에서 throughput 계산 후 저장.
   * 읽는 자: throughput_overflow 판정 시 참조.
   * 값 범위: 0 초과.
   * 동기화: compute() 후 불변. */

  double init_wire_bw;  // bus width at root
  /* [한국어] 트리 최상위 노드(루트)에서의 버스 폭 (비트) — 계층적 배선 트리 모델용.
   * 설정자: 생성자에서 data_width로 초기화.
   * 읽는 자: 계층 배선 트리 탐색 시 초기 분기 폭 기준으로 참조.
   * 값 범위: 1 이상의 양수.
   * 동기화: 생성 후 불변. */

  double base_width;
  /* [한국어] 배선이 놓이는 기반 블록의 폭 (단위: m) — 배선 길이 추정 시 기준.
   * 설정자: 생성자 인자 base_w로 초기화.
   * 읽는 자: compute() 내부에서 배선 길이 결정 또는 면적 계산 시 참조.
   * 값 범위: 0 이상.
   * 동기화: 생성 후 불변. */

  double base_height;
  /* [한국어] 배선이 놓이는 기반 블록의 높이 (단위: m).
   * 설정자: 생성자 인자 base_h로 초기화.
   * 읽는 자: compute() 내부 면적/길이 계산 시 참조.
   * 값 범위: 0 이상.
   * 동기화: 생성 후 불변. */

  int data_width;
  /* [한국어] 버스 데이터 폭 (비트 수) — 총 전력 스케일링 인자.
   * 설정자: 생성자 인자 data_w로 초기화.
   * 읽는 자: compute() 내부에서 Component::power = power_bit × data_width 계산 시 참조.
   * 값 범위: 1 이상의 양수 (예: 32, 64, 128 비트).
   * 동기화: 생성 후 불변. */

  enum Wire_type wt;
  /* [한국어] 배선 종류 (Global / Semi-global / Local 등 — CACTI Wire_type enum).
   * Global   : 칩 전체 범위 배선 (가장 두꺼운 금속층).
   * Semi-global: 블록 간 중간 길이 배선.
   * Local    : 인접 셀 간 단거리 배선 (가장 얇은 금속층).
   * 설정자: 생성자 인자 wire_model로 초기화.
   * 읽는 자: compute() 내부 Wire 생성 시 배선 RC 특성 선택에 참조.
   * 값 범위: CACTI의 Wire_type enum 값.
   * 동기화: 생성 후 불변. */

  double width_scaling, space_scaling;
  /* [한국어] 배선 폭·간격 스케일링 인자 (1.0 = 해당 공정 최소 피치).
   * width_scaling: 최소 배선 폭 대비 실제 사용 폭 배수. 생성자에서 1.0 → 2.0 → 3.0으로
   *                자동 증가시켜 max_unpipelined_link_delay 제약을 만족하는 최솟값 탐색.
   * space_scaling: 최소 배선 간격 대비 실제 간격 배수 — 누화(crosstalk) 완화용.
   * 설정자: 생성자 인자로 초기화 후 자동 최적화 루프에서 수정 가능.
   * 읽는 자: compute() 내부 Wire 생성 시 전달.
   * 값 범위: 1.0 ~ 3.0 (McPAT 관례상 3배 이상은 잘 사용하지 않음).
   * 동기화: 생성자 완료 후 불변. */

  int start_wiring_level;
  /* [한국어] 배선을 시작하는 금속 레이어 번호 (낮을수록 소자에 가까운 하위 레이어).
   * 설정자: 생성자 인자 start_wiring_level_로 초기화.
   * 읽는 자: Wire 모델 내부에서 레이어별 RC 특성 선택 시 참조.
   * 값 범위: 0 이상의 정수 (공정별 최대 레이어 수 미만).
   * 동기화: 생성 후 불변. */

  double length;
  /* [한국어] 배선 총 물리 길이 (단위: m).
   * 설정자: 생성자 인자 len으로 초기화 (0이면 base_width/base_height로부터 추정).
   * 읽는 자: compute() 내부 Wire 생성 시 RC 지연 계산의 핵심 입력으로 사용.
   * 값 범위: 0 이상의 실수 (일반적으로 수십 μm ~ 수 mm).
   * 동기화: 생성 후 불변. */

  double min_w_nmos;
  /* [한국어] 해당 공정의 NMOS 최소 트랜지스터 폭 (단위: m).
   * 설정자: 생성자에서 g_tp(TechnologyParameter)로부터 복사.
   * 읽는 자: 드라이버 크기 결정 시 최소 폭 기준으로 참조.
   * 값 범위: 공정 노드에 따른 양수 (예: 28nm 공정 → ~28e-9 m).
   * 동기화: 생성 후 불변. */

  double min_w_pmos;
  /* [한국어] 해당 공정의 PMOS 최소 트랜지스터 폭 (단위: m).
   * 설정자: 생성자에서 g_tp로부터 복사.
   * 읽는 자: 드라이버 PMOS 크기 결정 시 최소 폭 기준.
   * 값 범위: 공정 노드에 따른 양수 (NMOS min_w보다 일반적으로 1.5~2배 큼).
   * 동기화: 생성 후 불변. */

  double latency, throughput;
  /* [한국어] 목표 배선 지연(초)과 목표 처리량(비트/초) 제약값.
   * latency   : 배선이 반드시 충족해야 하는 최대 허용 전달 지연.
   * throughput: 배선이 반드시 달성해야 하는 최소 데이터 처리량.
   * 설정자: 생성자에서 클럭 주기와 파이프라이닝 여부로부터 산출.
   * 읽는 자: 생성자 내 width_scaling 자동 최적화 루프에서 Wire 결과와 비교.
   * 값 범위: 0 초과.
   * 동기화: 생성 후 불변. */

  bool latency_overflow;
  /* [한국어] 목표 지연을 충족하지 못한 경우 true로 설정.
   * 설정자: 생성자의 width_scaling 최적화 루프 종료 후 최종 지연 초과 여부로 설정.
   * 읽는 자: 상위 코드(compute() 호출자)에서 경고 출력 또는 보정에 참조.
   * 값 범위: true(지연 초과) / false(충족).
   * 동기화: 생성 후 불변. */

  bool throughput_overflow;
  /* [한국어] 목표 처리량을 충족하지 못한 경우 true.
   * 설정자: 생성자 최적화 루프 후 throughput 초과 여부로 설정.
   * 읽는 자: 상위 코드에서 경고·보정에 참조.
   * 값 범위: true / false.
   * 동기화: 생성 후 불변. */

  double interconnect_latency;
  /* [한국어] compute() 후 실제 계산된 배선 전달 지연 (단위: 초).
   * 설정자: compute() 내부 Wire 모델 결과로부터 저장.
   * 읽는 자: 지연 보고 및 latency_overflow 판정 시 참조.
   * 값 범위: 0 이상의 실수.
   * 동기화: compute() 후 불변. */

  double interconnect_throughput;
  /* [한국어] compute() 후 실제 계산된 배선 데이터 처리량 (단위: 비트/초).
   * 설정자: compute() 내부 Wire 모델 결과로부터 저장.
   * 읽는 자: 처리량 보고 및 throughput_overflow 판정 시 참조.
   * 값 범위: 0 이상의 실수.
   * 동기화: compute() 후 불변. */

  bool opt_local;
  /* [한국어] 로컬 배선 최적화 모드 활성화 여부.
   * true이면 최소 폭 배선을 우선 시도하여 면적을 최소화한다.
   * 설정자: 생성자 인자 opt_local_로 초기화.
   * 읽는 자: compute() 내부 Wire 최적화 전략 선택 시 참조.
   * 값 범위: true / false.
   * 동기화: 생성 후 불변. */

  enum Core_type core_ty;
  /* [한국어] 코어 유형 (Inorder / OOO) — 배선 모델 파라미터 분기용.
   * 설정자: 생성자 인자 core_ty_로 초기화.
   * 읽는 자: compute() 내부에서 OOO 코어 특화 배선 구성 분기 시 참조.
   * 값 범위: Core_type enum.
   * 동기화: 생성 후 불변. */

  bool pipelinable;
  /* [한국어] 배선에 파이프라인 레지스터를 삽입하여 처리량을 높일 수 있는지 여부.
   * true  (버스): 처리량(throughput) 제약을 충족하도록 파이프라인 스테이지 삽입 허용.
   * false (bypass 논리): 지연(latency)을 우선하며 파이프라인 삽입 없이 단일 사이클 전달.
   * 설정자: 생성자 인자 pipelinable_로 초기화.
   * 읽는 자: compute() 내부 Wire::compute() 호출 시 pipelinable 전달 → 최적화 방향 결정.
   * 값 범위: true / false.
   * 동기화: 생성 후 불변. */

  double route_over_perc;
  /* [한국어] 배선이 소자(트랜지스터) 위를 지나는 비율 (0.0 ~ 1.0).
   * 0.0: 배선이 모두 소자 옆 채널을 사용 → 면적 영향 최대.
   * 1.0: 배선이 모두 소자 위를 통과  → 추가 면적 없음.
   * 설정자: 생성자 인자 route_over_perc_로 초기화 (기본값 0.5).
   * 읽는 자: compute() 내부 no_device_under_wire_area 계산 시 참조.
   * 값 범위: 0.0 ~ 1.0.
   * 동기화: 생성 후 불변. */

  int num_pipe_stages;
  /* [한국어] 배선에 삽입된 파이프라인 스테이지 수 (pipelinable=true일 때만 유효).
   * 설정자: compute() 내부에서 Wire 모델이 throughput 제약을 만족하는 스테이지 수를 결정.
   * 읽는 자: 전력 보고 및 파이프라인 레지스터 전력 합산 시 참조.
   * 값 범위: 0 이상의 정수 (0 = 파이프라인 없음).
   * 동기화: compute() 후 불변. */

 private:
  TechnologyParameter::DeviceType *deviceType;
  /* [한국어] 소자 공정 파라미터 포인터 — NMOS/PMOS 특성값(Vdd, 이동도 등) 접근용.
   * 설정자: 생성자 인자 dt로 초기화 (기본값: &g_tp.peri_global — 주변 회로 공정).
   * 읽는 자: compute() 내부 Wire 생성 시 디바이스 파라미터로 전달.
   * 값 범위: 유효한 TechnologyParameter::DeviceType 포인터.
   * 동기화: 생성 후 불변; Wire 내부에서만 읽기 접근. */
};

#endif
