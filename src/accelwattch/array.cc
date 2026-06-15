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
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
 *
 ***************************************************************************/

/*
 * [한국어 설명] SRAM 배열 전력/면적 모델 구현 (array.cc)
 *
 * === 파일의 역할 ===
 * ArrayST 클래스의 생성자, optimize_array(), compute_base_power(),
 * leakage_feedback(), 소멸자를 구현한다. 이 파일이 하는 핵심 작업은
 * CACTI(Cache And Memory Circuit/Technology Interface)를 호출하여
 * 주어진 SRAM 설정(크기, 연관도, 뱅크 수 등)의 동적 에너지, 누설 전력, 면적을
 * 계산하고, 글로벌 기술 파라미터(g_tp)와 공정 스케일링 계수를 적용해
 * 최종 전력/면적 값을 산출하는 것이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch GPU 전력 모델의 가장 하위 레벨 구현으로, GPU 서브유닛
 * (I-cache, D-cache, L2, Register File, Shared Memory, TLB 등)이 생성하는
 * 수십~수백 개의 ArrayST 인스턴스 각각에 대해 CACTI를 호출한다.
 * 실행 흐름:
 *   GPGPU-Sim 초기화
 *     → AccelWattch Processor 생성 → GPU 서브유닛 생성
 *         → ArrayST 생성자 (이 파일)
 *             → l_ip.error_checking()
 *             → optimize_array() → compute_base_power() → cacti_interface()
 *   온도 갱신 (시뮬레이션 중):
 *     → leakage_feedback(temp) → reconfigure() → 전력 스케일링 (이 파일)
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 스레드 (초기화 및 전력 집계 단계).
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - array.h            : ArrayST, InstCache, DataCache 클래스 선언
 *   - cacti/area.h       : CACTI 내부 면적 계산 구조체/함수
 *   - decoder.h          : CACTI 내부 디코더 모델 (간접 의존)
 *   - globalvar.h        : opt_for_clk (타이밍 최적화 전역 플래그),
 *                          g_tp (TechnologyParameter 전역 객체)
 *   - parameter.h        : CACTI 파라미터 확장 정의
 *   - basic_components.h : longer_channel_device_reduction() (누설 축소 계수)
 * 데이터 흐름:
 *   l_ip (InputParameter) → cacti_interface() → local_result (uca_org_t)
 *   → 스케일링 (sckRation, pppm_t, long_channel_device_reduction) → 최종 전력/면적
 * 공유 전역 상태:
 *   - g_tp.macro_layout_overhead, g_tp.chip_layout_overhead: 레이아웃 면적 오버헤드
 *   - g_tp.sckt_co_eff (sckRation): 소켓 계수, 동적 전력 스케일링에 사용
 *   - opt_for_clk: globalvar.h에서 전역 선언, CACTI 최적화 방향 제어
 *
 * === 주요 함수/구조체 요약 ===
 * ArrayST()             - 생성자: l_ip 복사 → cache_sz 하한 → error_checking()
 *                         → optimize_array()로 완전 초기화
 * compute_base_power()  - cacti_interface(&l_ip) 단순 래퍼, CACTI 1회 호출
 * optimize_array()      - 타이밍 제약 충족 + 동적 에너지 최소화를 위한 반복 탐색,
 *                         cycle_time_dev를 100→10으로 줄이며 후보 수집 후 최적 선택,
 *                         이후 면적/전력 스케일링(sckRation, pppm_t, nbanks, long channel)
 * leakage_feedback()    - 온도 갱신 후 reconfigure()로 누설 재계산,
 *                         optimize_array()와 동일한 스케일링 코드를 재실행
 * ~ArrayST()            - local_result.cleanup()으로 CACTI 동적 메모리 해제
 */

#define GLOBALVAR  // [한국어] globalvar.h의 전역 변수(g_tp, opt_for_clk 등)를 이 번역 단위에서 정의(define)하도록 하는 매크로.
                   // 다른 번역 단위에서는 extern 선언만 보이고, 이 파일에서 실제 메모리를 할당한다.
#include "array.h"        // [한국어] ArrayST, InstCache, DataCache 클래스 선언
#include <assert.h>       // [한국어] assert() 매크로 (방어적 검사, 현재 코드에서 직접 사용은 미미하나 CACTI 내부에서 사용됨)
#include <math.h>         // [한국어] round() 함수 (leakage_feedback에서 온도를 10 단위로 반올림하는 데 사용)
#include <iostream>       // [한국어] cout (경고 메시지 출력용)
#include "cacti/area.h"   // [한국어] CACTI 내부 면적 계산 구조체 및 함수 (ArrayST가 간접적으로 의존)
#include "decoder.h"      // [한국어] CACTI 내부 디코더 전력/면적 모델 (cacti_interface 내부에서 사용)
#include "globalvar.h"    // [한국어] opt_for_clk(전역 타이밍 최적화 플래그), g_tp(TechnologyParameter 전역 객체) 실제 정의
#include "parameter.h"    // [한국어] CACTI InputParameter 확장 필드 및 관련 상수

using namespace std;

/*
 * [한국어]
 * ArrayST::ArrayST - SRAM 배열 전력 모델 초기화 생성자
 *
 * @configure_interface: CACTI에 전달할 SRAM 설정 파라미터 (캐시 크기, 연관도,
 *   라인 크기, 뱅크 수, throughput/latency 제약 등). 호출자가 메모리를 소유하며
 *   이 생성자는 복사본(l_ip)을 만든다.
 * @_name: 이 배열의 디버깅용 식별 이름 (예: "L1_icache", "RF_data").
 * @device_ty_: 공정 기술 유형 (ITRS HP/LSTP/LOP 등), 누설 축소 계수에 영향.
 * @opt_local_: true면 이 ArrayST가 직접 CACTI 타이밍 최적화 루프를 수행.
 * @core_ty_: 코어 유형 (Inorder/OOO), 누설 축소 계수에 영향.
 * @_is_default: 기본 설정 여부 플래그.
 * @return: (생성자, 반환값 없음)
 *
 * 초기화 리스트에서 l_ip, name, device_ty, opt_local, core_ty, is_default를
 * 설정한 후, 본문에서 cache_sz 하한 보정 → error_checking() → optimize_array()
 * 순서로 완전한 SRAM 전력 모델을 구축한다.
 *
 * 호출 체인:
 *   GPU 서브유닛 생성자 → [ArrayST::ArrayST] → l_ip.error_checking()
 *                                             → ArrayST::optimize_array()
 *                                                 → ArrayST::compute_base_power()
 *                                                     → cacti_interface()
 */
ArrayST::ArrayST(const InputParameter *configure_interface, string _name,
                 enum Device_ty device_ty_, bool opt_local_,
                 enum Core_type core_ty_, bool _is_default)
    : l_ip(*configure_interface),   // [한국어] configure_interface의 InputParameter 구조체 전체를 l_ip에 복사.
                                    // 이후 optimize_array()가 l_ip를 수정하므로 원본 훼손 방지를 위해 복사본을 사용.
      name(_name),                  // [한국어] 배열 식별 이름 복사 (경고 출력 및 디버깅용)
      device_ty(device_ty_),        // [한국어] 공정 기술 유형 저장 (누설 계수 계산에 사용)
      opt_local(opt_local_),        // [한국어] 로컬 타이밍 최적화 수행 여부 플래그 저장
      core_ty(core_ty_),            // [한국어] 코어 유형 저장 (누설 계수 계산에 사용)
      is_default(_is_default) {     // [한국어] 기본 설정 여부 플래그 저장
  // [한국어] cache_sz 하한 보정: CACTI는 64바이트 미만 배열을 지원하지 않으므로
  // 너무 작은 값이 들어올 경우 강제로 64로 올린다.
  if (l_ip.cache_sz < 64) l_ip.cache_sz = 64;  // [한국어] 최소 64바이트 SRAM 크기 보장 (CACTI 하드 하한)

  l_ip.error_checking();  // not only do the error checking but also fill some
                          // missing parameters
  // [한국어] error_checking(): 단순 에러 검사를 넘어 l_ip의 누락된 파라미터(예: 뱅크 수,
  // 워드 너비 등)를 자동으로 채워준다. CACTI에 안전하게 전달할 수 있는 상태로 만든다.

  optimize_array();  // [한국어] CACTI를 반복 호출해 타이밍 제약을 만족하는 최소 동적 에너지
                     // 설정을 탐색하고 면적/전력 스케일링을 수행한다.
}

/*
 * [한국어]
 * ArrayST::compute_base_power - CACTI를 1회 호출하여 현재 l_ip 설정의 전력/면적 계산
 *
 * @return: void (결과는 local_result에 저장됨)
 *
 * cacti_interface(&l_ip)를 호출하여 현재 l_ip가 기술하는 SRAM 배열의
 * 동적 에너지(readOp.dynamic), 누설 전력(readOp.leakage), 면적, 사이클 타임,
 * 접근 시간을 계산한다. optimize_array()의 내부 루프에서 매 이터레이션마다
 * cycle_time_dev를 변경한 뒤 이 함수를 호출하여 새로운 CACTI 해를 구한다.
 *
 * 실행 컨텍스트: 단일 스레드, 시뮬레이션 초기화 단계.
 *
 * 호출 체인:
 *   ArrayST::optimize_array() → [ArrayST::compute_base_power] → cacti_interface()
 */
void ArrayST::compute_base_power() {
  // l_ip.out_w               =l_ip.line_sz*8;
  // [한국어] 위 주석은 CACTI 구버전에서 out_w(출력 비트 폭)를 수동 설정하던 코드의 흔적.
  // 현재는 error_checking()에서 자동으로 채워지므로 비활성화되어 있다.

  local_result = cacti_interface(&l_ip);
  // [한국어] CACTI의 메인 진입 함수 호출: l_ip에 기술된 SRAM 파라미터를 기반으로
  // 면적, 사이클 타임(cycle_time), 접근 시간(access_time),
  // 동적 에너지(power.readOp.dynamic), 누설 전력(power.readOp.leakage) 등을 계산하여
  // uca_org_t 구조체로 반환한다. 반환값 전체를 local_result에 복사 저장.
}

/*
 * [한국어]
 * ArrayST::optimize_array - CACTI 반복 호출로 타이밍 제약 충족 + 동적 에너지 최소화
 *
 * @return: void (결과는 local_result에 저장됨)
 *
 * 이 함수는 두 단계로 동작한다:
 *
 * [1단계: CACTI 탐색 루프 (opt_for_clk && opt_local 조건 시)]
 *   cycle_time_dev를 100에서 10까지 10씩 줄이며 compute_base_power()를 반복 호출한다.
 *   cycle_time_dev는 CACTI 내부에서 사이클 타임 목표 편차(deviation) 허용 폭으로,
 *   값이 작을수록 CACTI가 더 공격적으로 타이밍을 최적화한다 (면적이 증가하는 트레이드오프).
 *   각 이터레이션에서:
 *     - throughput 제약(cycle_time ≤ throughput)과 latency 제약(access_time ≤ latency)을
 *       모두 충족하거나, CAM/FA 구조에서 면적 효율이 너무 낮아지면 후보 리스트에 추가.
 *     - 두 제약이 모두 충족되면 루프를 조기 종료.
 *     - 충족하지 못하면 해당 CACTI 결과를 정리(cleanup)하고 다음 이터레이션 진행.
 *   루프 종료 후 후보 리스트에서 readOp.dynamic(동적 에너지)이 가장 작은 것을 선택.
 *
 * [2단계: 전력/면적 스케일링 (항상 수행)]
 *   - long_channel_device_reduction: 장채널 소자(longer-channel device)의 누설 전력
 *     축소 계수를 계산 (공정 기술 유형과 코어 유형에 따라 다름).
 *   - total_overhead = macro_layout_overhead × chip_PR_overhead:
 *     칩 레이아웃 및 배선 오버헤드를 면적에 적용.
 *   - pppm_t[4] = {total_overhead, 1, 1, total_overhead}: 전력×면적 스케일링 벡터.
 *     powerDef::operator*(pppm_t)는 [동적 전력 × pppm_t[0], ..., 누설 × pppm_t[3]] 형식.
 *   - sckRation(소켓 계수): 동적 전력에 곱해 실제 GPU 환경의 소켓 부하를 반영.
 *   - nbanks 스케일링: 누설은 뱅크 수에 선형 비례하므로 l_ip.nbanks를 곱함.
 *   - longer_channel_leakage: 장채널 누설 전력 = 기본 누설 × long_channel_device_reduction.
 *   스케일링은 local_result.power(전체), local_result.data_array2->power,
 *   (캐시 구조라면) local_result.tag_array2->power 세 곳에 동일하게 적용.
 *
 * 실행 컨텍스트: 단일 스레드, 시뮬레이션 초기화 단계.
 *
 * 호출 체인:
 *   ArrayST::ArrayST() → [ArrayST::optimize_array]
 *       → ArrayST::compute_base_power() → cacti_interface()
 *       → longer_channel_device_reduction()
 */
void ArrayST::optimize_array() {
  list<uca_org_t> candidate_solutions(0);
  // [한국어] 타이밍 제약을 충족하는 CACTI 해(解)를 모아두는 후보 리스트.
  // 빈 리스트(크기 0)로 초기화. 탐색 루프에서 조건을 만족하는 local_result를 push_back.

  list<uca_org_t>::iterator candidate_iter, min_dynamic_energy_iter;
  // [한국어] candidate_solutions 순회용 반복자 및 최소 동적 에너지 후보 가리키는 반복자.

  uca_org_t *temp_res = 0;
  // [한국어] 후보 리스트에 넣지 않고 버릴 CACTI 결과를 cleanup()하기 위한 임시 포인터.
  // nullptr(0)로 초기화. 실제로 이 포인터가 local_result를 가리킬 때만 cleanup() 호출.

  local_result.valid = false;
  // [한국어] 탐색 시작 전에 local_result를 무효(false)로 표시.
  // 탐색에 성공하면 최적 후보를 local_result에 복사하고 valid = true로 설정.

  double throughput = l_ip.throughput, latency = l_ip.latency;
  // [한국어] 사용자가 지정한 throughput(처리량) 제약: SRAM cycle_time이 이 값 이하여야 함.
  // latency 제약: SRAM access_time이 이 값 이하여야 함. 단위는 초(second).

  double area_efficiency_threshold = 20.0;
  // [한국어] CAM(Content Addressable Memory) / FA(Fully Associative) 구조에서
  // 면적 효율(area efficiency, %)이 이 값 미만이면 타이밍 최적화를 중단한다.
  // CAM은 파이프라이닝으로 타이밍 완화가 가능하므로, 지나친 면적 팽창을 막기 위해
  // 20% 임계값을 사용 (McPAT 설계 결정).

  bool throughput_overflow = true, latency_overflow = true;
  // [한국어] 현재 CACTI 해가 throughput/latency 제약을 위반하는지 나타내는 플래그.
  // true = 제약 위반(overflow), false = 제약 충족. 초기값은 true(미충족)로 보수적으로 설정.

  compute_base_power();
  // [한국어] 초기 l_ip 설정(탐색 가중치 수정 전)으로 CACTI를 1회 호출하여
  // 기본 전력/타이밍을 얻는다. 이후 opt_for_clk 조건에 따라 최적화를 추가 수행할지 결정.

  if ((local_result.cycle_time - throughput) <= 1e-10)
    throughput_overflow = false;
  // [한국어] 기본 계산 결과가 이미 throughput 제약을 충족하면(cycle_time ≤ throughput + ε)
  // throughput_overflow를 false로 설정. 부동소수점 오차를 감안해 1e-10 여유를 둠.

  if ((local_result.access_time - latency) <= 1e-10) latency_overflow = false;
  // [한국어] 기본 계산 결과가 이미 latency 제약을 충족하면(access_time ≤ latency + ε)
  // latency_overflow를 false로 설정.

  if (opt_for_clk && opt_local) {
    // [한국어] opt_for_clk: 전역 플래그 (globalvar.h), 타이밍 우선 최적화 모드.
    // opt_local: 이 ArrayST 인스턴스가 자체적으로 최적화를 수행할지 여부.
    // 두 조건이 모두 참일 때만 반복 탐색 루프에 진입한다.

    if (throughput_overflow || latency_overflow) {
      // [한국어] 기본 설정으로는 타이밍 제약이 충족되지 않았으므로 CACTI 가중치를
      // 타이밍 우선 모드로 재설정하고 반복 탐색을 준비한다.

      l_ip.ed = 0;
      // [한국어] ed(Energy-Delay product) 최적화 플래그를 0으로 끈다.
      // 에너지-지연 곱 최소화 대신 타이밍 만족을 우선하는 순수 타이밍 최적화 모드로 전환.

      l_ip.delay_wt = 100;  // Fixed number, make sure timing can be satisfied.
      // [한국어] 접근 지연(delay) 가중치를 100으로 설정. 높은 가중치는 CACTI가
      // 최적화 목적함수에서 지연 감소를 최우선으로 고려하게 만든다.

      l_ip.cycle_time_wt = 1000;
      // [한국어] 사이클 타임 가중치를 1000(최대)으로 설정.
      // delay_wt(100)보다 10배 높아 사이클 타임을 가장 강력히 최소화하도록 유도.

      l_ip.area_wt = 10;  // Fixed number, This is used to exhaustive search for
                          // individual components.
      // [한국어] 면적 가중치를 10으로 낮춘다. 타이밍 최적화를 위해 면적을 희생할 수 있음을 의미.
      // 개별 컴포넌트 전수 탐색(exhaustive search)에 사용한다고 McPAT 원주석에 명시.

      l_ip.dynamic_power_wt = 10;  // Fixed number, This is used to exhaustive
                                   // search for individual components.
      // [한국어] 동적 전력 가중치를 10으로 낮춘다. 타이밍 달성이 더 중요하므로
      // 전력을 다소 증가시키더라도 타이밍을 맞추는 방향으로 탐색.

      l_ip.leakage_power_wt = 10;
      // [한국어] 누설 전력 가중치도 10으로 낮춘다. 타이밍 우선 탐색에서 누설은 부차적 목표.

      l_ip.delay_dev =
          1000000;  // Fixed number, make sure timing can be satisfied.
      // [한국어] 지연 편차(deviation) 허용 범위를 매우 크게(100만) 설정.
      // CACTI가 지연 목표를 거의 무조건 달성 가능한 방향으로 해를 탐색하게 한다.

      l_ip.cycle_time_dev = 100;
      // [한국어] 사이클 타임 편차 허용 범위 초기값을 100으로 설정.
      // 이 값을 반복마다 10씩 줄여(100→90→80→...→10) 더 엄격한 타이밍 목표로 수렴시킨다.

      l_ip.area_dev = 1000000;  // Fixed number, This is used to exhaustive
                                // search for individual components.
      // [한국어] 면적 편차를 매우 크게 설정하여 면적 제약을 사실상 해제.
      // 타이밍 달성을 위한 전수 탐색에서 면적은 무제한에 가깝게 허용.

      l_ip.dynamic_power_dev =
          1000000;  // Fixed number, This is used to exhaustive search for
                    // individual components.
      // [한국어] 동적 전력 편차를 매우 크게 설정. 전력 제약도 사실상 해제.

      l_ip.leakage_power_dev = 1000000;
      // [한국어] 누설 전력 편차도 매우 크게 설정. 타이밍만 제약으로 남긴다.

      throughput_overflow =
          true;  // Reset overflow flag before start optimization iterations
      // [한국어] 위에서 임시로 false가 된 플래그를 다시 true로 리셋.
      // 이제부터 cycle_time_dev를 줄이며 실제 타이밍 충족 여부를 반복 검사한다.

      latency_overflow = true;
      // [한국어] latency overflow 플래그도 다시 true로 리셋하여 탐색 루프 조건을 활성화.

      temp_res = &local_result;  // Clean up the result for optimized for ED^2P
      // [한국어] 현재 local_result는 ED^2P(Energy-Delay^2 Product) 최적화로 얻은 결과이므로
      // 타이밍 최적화 탐색에서는 버려야 한다. cleanup()으로 내부 동적 메모리를 해제 준비.

      temp_res->cleanup();
      // [한국어] local_result 내부의 data_array2, tag_array2 등 CACTI가 동적 할당한
      // 포인터들의 메모리를 해제한다. 이후 compute_base_power()가 새 결과를 할당.
    }

    while ((throughput_overflow || latency_overflow) &&
           l_ip.cycle_time_dev > 10)  // && l_ip.delay_dev > 10
    // [한국어] 타이밍 제약이 아직 미충족(throughput_overflow || latency_overflow)이고
    // cycle_time_dev가 10보다 클 동안 반복한다.
    // cycle_time_dev: 100에서 시작해 매 반복마다 10씩 감소 → 10 이하가 되면 종료.
    // 즉 최대 (100-10)/10 = 9회 이터레이션. 조기 종료 조건: 두 제약 모두 충족.
    {
      compute_base_power();
      // [한국어] 현재 l_ip.cycle_time_dev 값으로 CACTI를 호출해 새 해를 구한다.
      // 결과는 local_result에 저장됨.

      l_ip.cycle_time_dev -=
          10;  // This is the time_dev to be used for next iteration
      // [한국어] 다음 이터레이션의 cycle_time_dev를 10 감소시킨다.
      // 현재 이터레이션의 결과를 먼저 평가한 후 다음을 위해 미리 감소시킴.
      // 이렇게 하면 루프가 현재 이터레이션 결과를 평가하면서 동시에
      // 다음 이터레이션 dev 값을 준비한다.

      //		from best area to worst area -->worst timing to best
      // timing
      // [한국어] 위 주석: cycle_time_dev가 클수록 면적은 최소이지만 타이밍은 나쁘고,
      // dev가 작아질수록 타이밍은 좋아지지만 면적이 커진다. 최선 면적→최악 면적 방향 탐색.

      if ((((local_result.cycle_time - throughput) <= 1e-10) &&
           (local_result.access_time - latency) <= 1e-10) ||
          (local_result.data_array2->area_efficiency <
               area_efficiency_threshold &&
           l_ip.assoc == 0)) {  // if no satisfiable solution is found,the most
                                // aggressive one is left
        // [한국어] 두 가지 중 하나라도 해당하면 후보 리스트에 추가:
        //   ① cycle_time ≤ throughput + ε 이고 access_time ≤ latency + ε (두 제약 모두 충족)
        //   ② data 배열의 면적 효율이 20% 미만이고 assoc==0 (CAM/FA 구조): 면적 효율이 너무
        //      낮아지면 더 이상 최적화할 이유가 없으므로 "만족 가능한 해가 없는 경우 가장
        //      공격적인 해"를 저장. 이 경우 타이밍 미충족이어도 후보로 남긴다.

        candidate_solutions.push_back(local_result);
        // [한국어] 현재 CACTI 해를 후보 리스트의 끝에 복사 추가.
        // uca_org_t의 복사 생성자가 호출되어 data_array2, tag_array2 등 포인터도 복사됨.
        // (얕은 복사이므로 나중에 cleanup() 호출 시 중복 해제 주의 필요 - CACTI 내부 관리)

        // output_data_csv(candidate_solutions.back());
        // [한국어] CSV 출력 코드 (디버깅용, 현재 비활성화).

        if (((local_result.cycle_time - throughput) <= 1e-10) &&
            ((local_result.access_time - latency) <= 1e-10))
        // ensure stop opt not because of cam
        // [한국어] 두 제약을 모두 충족한 경우에만 (CAM 면적 임계값 때문이 아닌 정상 종료)
        // 두 overflow 플래그를 false로 설정하여 반복 루프를 조기 종료시킨다.
        {
          throughput_overflow = false;  // [한국어] throughput 제약 충족 → 플래그 해제
          latency_overflow = false;     // [한국어] latency 제약 충족 → 플래그 해제
        }

      } else {
        // [한국어] 두 제약 모두 미충족이고 면적 효율 조건도 해당하지 않는 경우:
        // 이 이터레이션의 해는 후보가 되지 못하므로 cleanup()으로 정리한다.
        // 단, 개별 overflow 플래그는 부분 충족 여부에 따라 업데이트.

        // TODO: whether checking the partial satisfied results too, or just
        // change the mark???
        // [한국어] 위 TODO: 부분 충족 결과(한 제약만 만족)도 후보로 저장할지 여부.
        // 현재 구현은 두 제약 모두 충족해야만 후보로 저장한다.

        if ((local_result.cycle_time - throughput) <= 1e-10)
          throughput_overflow = false;
        // [한국어] throughput 제약은 충족했으면 해당 플래그를 false로 업데이트.
        // 다음 이터레이션에서 latency만 남은 조건으로 계속 탐색.

        if ((local_result.access_time - latency) <= 1e-10)
          latency_overflow = false;
        // [한국어] latency 제약은 충족했으면 해당 플래그를 false로 업데이트.

        if (l_ip.cycle_time_dev > 10) {  // if not >10 local_result is the last
                                         // result, it cannot be cleaned up
          // [한국어] cycle_time_dev가 아직 10보다 크면 다음 이터레이션이 있으므로
          // 현재 local_result는 불필요. cleanup()으로 내부 메모리 해제.
          // cycle_time_dev가 10이하면 이 결과가 마지막이므로 cleanup하지 않음
          // (루프 탈출 후 아래에서 사용될 수 있음).

          temp_res = &local_result;      // Only solutions not saved in the list
                                         // need to be cleaned up
          // [한국어] 후보 리스트에 저장되지 않은(버릴) 결과만 cleanup 대상으로 설정.

          temp_res->cleanup();
          // [한국어] local_result 내부의 CACTI 동적 할당 메모리 해제.
          // 다음 compute_base_power() 호출 시 새 메모리가 할당된다.
        }
      }
      //			l_ip.cycle_time_dev-=10;
      //			l_ip.delay_dev-=10;
      // [한국어] 위 두 줄은 루프 하단에 감소를 두었던 이전 버전 흔적. 현재는 루프 상단에서 감소.
    }

    if (l_ip.assoc > 0) {
      // [한국어] assoc > 0: 일반 set-associative 캐시 또는 direct-mapped 캐시 (RAM 계열).
      // CAM/FA 구조(assoc == 0)는 파이프라이닝으로 타이밍 완화가 가능하므로 경고 생략.
      // 일반 캐시에서 제약 미충족 시 사용자에게 경고 메시지를 출력한다.

      // For array structures except CAM and FA, Give warning but still provide
      // a result with best timing found
      if (throughput_overflow == true)
        cout << "Warning: " << name
             << " array structure cannot satisfy throughput constraint."
             << endl;
      // [한국어] throughput 제약을 만족하는 해를 찾지 못했을 경우 경고.
      // 현재까지 찾은 최선의 타이밍 해로 계속 진행함을 암묵적으로 의미.

      if (latency_overflow == true)
        cout << "Warning: " << name
             << " array structure cannot satisfy latency constraint." << endl;
      // [한국어] latency 제약을 만족하는 해를 찾지 못했을 경우 경고.
    }

    //	else
    //	{
    //		/*According to "Content-Addressable Memory (CAM) Circuits and
    //				Architectures": A Tutorial and Survey
    //				by Kostas Pagiamtzis et al.
    //				CAM structures can be heavily pipelined and use
    // look-ahead techniques, 				therefore timing can be
    // relaxed. But McPAT does not model the
    // advanced 				techniques. If continue
    // optimizing, the area efficiency will be too low
    //		*/
    //		//For CAM and FA, stop opt if area efficiency is too low
    //		if (throughput_overflow==true)
    //			cout<< "Warning: " <<" McPAT stopped optimization on
    // throughput for
    //"<< name
    //				<<" array structure because its area efficiency
    // is below
    //"<<area_efficiency_threshold<<"% " << endl; 		if
    //(latency_overflow==true) 			cout<< "Warning: " <<" McPAT
    // stopped optimization on latency for "<< name
    //				<<" array structure because its area efficiency
    // is below
    //"<<area_efficiency_threshold<<"% " << endl;
    //	}
    // [한국어] 위 전체 else 블록: CAM/FA 구조에 대한 경고 코드. 현재 비활성화.
    // McPAT 논문(Pagiamtzis et al.)에 따르면 CAM은 파이프라이닝/룩어헤드로 타이밍 완화
    // 가능하지만 McPAT는 이를 모델링하지 않으므로, 면적 효율이 너무 낮아지면 최적화를
    // 중단하고 경고 대신 조용히 처리하는 방향으로 결정됨.

    // double min_dynamic_energy, min_dynamic_power, min_leakage_power,
    // min_cycle_time;
    // [한국어] 아래에서 실제 사용되는 min_dynamic_energy만 남기고 나머지는 주석 처리된 흔적.

    double min_dynamic_energy = BIGNUM;
    // [한국어] 후보 리스트 중 최소 동적 에너지 값을 추적하기 위한 변수.
    // BIGNUM: cacti/const.h에서 정의된 매우 큰 수(초기 최솟값 역할).

    if (candidate_solutions.empty() == false) {
      // [한국어] 후보 리스트가 비어있지 않을 때만 최적 선택 로직 수행.
      // 비어있다면 local_result.valid는 false인 채로 남는다 (탐색 실패).

      local_result.valid = true;
      // [한국어] 유효한 후보가 존재하므로 local_result를 유효 상태로 표시.

      for (candidate_iter = candidate_solutions.begin();
           candidate_iter != candidate_solutions.end(); ++candidate_iter)
      // [한국어] 후보 리스트를 처음부터 끝까지 순회하며 최소 동적 에너지 후보를 찾는다.
      {
        if (min_dynamic_energy > (candidate_iter)->power.readOp.dynamic) {
          // [한국어] 현재 후보의 readOp.dynamic(읽기 동작 1회당 동적 에너지, 단위: J)이
          // 지금까지의 최솟값보다 작으면 이 후보를 새 최적 후보로 선택한다.

          min_dynamic_energy = (candidate_iter)->power.readOp.dynamic;
          // [한국어] 최솟값을 현재 후보의 에너지로 업데이트.

          min_dynamic_energy_iter = candidate_iter;
          // [한국어] 최소 에너지 후보를 가리키는 반복자 업데이트.

          local_result = *(min_dynamic_energy_iter);
          // [한국어] 최적 후보를 local_result에 복사 저장. 역참조 후 복사 할당.
          // TODO 주석: 결과가 재정렬되어 l_ip와 local_result가 불일치할 수 있음.
          // 최종 출력 스프레드시트에서 불일치가 나타날 수 있다는 McPAT의 알려진 이슈.

          // TODO: since results are reordered results and l_ip may miss match.
          // Therefore, the final output spread sheets may show the miss match.

        } else {
          candidate_iter->cleanup();
          // [한국어] 선택되지 않은(더 높은 에너지의) 후보는 cleanup()으로 내부 메모리 해제.
          // 최적 후보(local_result에 복사된 것)는 해제하지 않는다.
        }
      }
    }
    candidate_solutions.clear();
    // [한국어] 후보 리스트를 비운다. 각 원소의 소멸자는 호출되지만
    // uca_org_t의 소멸자가 내부 포인터를 해제하지는 않는다 (cleanup() 명시 호출 필요).
    // 위에서 선택되지 않은 후보들은 이미 cleanup()으로 해제되었고,
    // 최적 후보는 local_result에 복사되었으므로 안전하게 clear() 가능.
  }

  // [한국어] === 2단계: 전력/면적 스케일링 ===
  // opt_for_clk/opt_local 조건과 무관하게 항상 수행된다.

  double long_channel_device_reduction =
      longer_channel_device_reduction(device_ty, core_ty);
  // [한국어] 장채널 소자(longer-channel device)의 누설 전력 축소 계수를 계산한다.
  // ITRS 표준 공정보다 더 긴 채널을 사용하는 소자는 단채널 소자보다 누설이 적다.
  // basic_components.cc에 구현된 함수로, device_ty(HP/LSTP/LOP)와 core_ty(Inorder/OOO)에
  // 따라 0~1 범위의 축소 계수를 반환한다. 이를 기본 누설 전력에 곱해 실제 누설을 산출.

  double macro_layout_overhead = g_tp.macro_layout_overhead;
  // [한국어] 매크로(SRAM 배열 블록) 레이아웃 오버헤드 계수 (g_tp: 전역 기술 파라미터).
  // SRAM 셀 순수 면적 대비 실제 레이아웃에서 배선, 경계, 더미 행/열 등으로 인한 면적 증가 비율.
  // 값은 >1.0 (예: 1.15 → 15% 면적 증가).

  double chip_PR_overhead = g_tp.chip_layout_overhead;
  // [한국어] 칩 레벨 배선/플로어플랜 레이아웃 오버헤드 계수.
  // 칩 전체 수준에서 배선 채널, 전원 그리드, 테스트 구조 등으로 인한 추가 면적 비율.
  // 값은 >1.0.

  double total_overhead = macro_layout_overhead * chip_PR_overhead;
  // [한국어] 두 레이아웃 오버헤드를 곱해 최종 면적 스케일링 계수를 계산한다.
  // 예: macro=1.10, chip=1.05 → total_overhead=1.155 (순수 배열 면적의 15.5% 증가).

  local_result.area *= total_overhead;
  // [한국어] CACTI가 계산한 순수 SRAM 배열 면적에 total_overhead를 곱해
  // 실제 칩 레이아웃에서의 최종 면적을 산출한다. 단위: m².

  // maintain constant power density
  double pppm_t[4] = {total_overhead, 1, 1, total_overhead};
  // [한국어] pppm_t(Power Per Port Multiplier): 4원소 스케일링 벡터.
  // powerDef::operator*(pppm_t) 연산 시:
  //   [0] = total_overhead → readOp.dynamic 스케일링에 사용
  //   [1] = 1              → (중간값, 변경 없음)
  //   [2] = 1              → (중간값, 변경 없음)
  //   [3] = total_overhead → leakage 스케일링에 사용
  // "일정한 전력 밀도 유지": 면적이 total_overhead 배로 늘어나면 전력도 같은 비율로 스케일링.

  double sckRation = g_tp.sckt_co_eff;
  // [한국어] 소켓 계수(socket coefficient, g_tp.sckt_co_eff).
  // GPU를 탑재한 소켓 환경에서의 동적 전력 스케일링 계수.
  // 파워 공급 경로, 전압 레귤레이터 효율 등을 반영하는 경험적 값.

  local_result.power.readOp.dynamic *= sckRation;
  // [한국어] 전체 배열의 읽기 동작(readOp) 동적 에너지에 소켓 계수 적용.

  local_result.power.writeOp.dynamic *= sckRation;
  // [한국어] 전체 배열의 쓰기 동작(writeOp) 동적 에너지에 소켓 계수 적용.

  local_result.power.searchOp.dynamic *= sckRation;
  // [한국어] 전체 배열의 검색 동작(searchOp, CAM 전용) 동적 에너지에 소켓 계수 적용.

  local_result.power.readOp.leakage *= l_ip.nbanks;
  // [한국어] 누설 전력을 뱅크 수(nbanks)에 비례하여 스케일링.
  // CACTI는 단일 뱅크 기준으로 누설을 계산하므로, 실제 뱅크 수만큼 곱해 전체 배열의 누설을 얻는다.

  local_result.power.readOp.longer_channel_leakage =
      local_result.power.readOp.leakage * long_channel_device_reduction;
  // [한국어] 장채널 소자 기준 누설 전력 계산:
  //   기본 누설(nbanks 스케일링 적용 후) × long_channel_device_reduction 계수.
  // 이 값은 AccelWattch가 공정 기술 파라미터를 반영한 보정 누설 전력으로 별도 추적한다.

  local_result.power = local_result.power * pppm_t;
  // [한국어] pppm_t 스케일링 벡터를 powerDef::operator*()로 적용.
  // total_overhead 계수가 dynamic 및 leakage 항목에 곱해져 면적 증가에 비례한 전력 밀도를 유지한다.

  local_result.data_array2->power.readOp.dynamic *= sckRation;
  // [한국어] 데이터 배열(data_array2)의 읽기 동적 에너지에 소켓 계수 적용.
  // local_result.power는 전체 배열(data + tag), data_array2는 데이터 부분만의 전력.

  local_result.data_array2->power.writeOp.dynamic *= sckRation;
  // [한국어] 데이터 배열의 쓰기 동적 에너지에 소켓 계수 적용.

  local_result.data_array2->power.searchOp.dynamic *= sckRation;
  // [한국어] 데이터 배열의 검색 동적 에너지에 소켓 계수 적용 (CAM 구조의 경우).

  local_result.data_array2->power.readOp.leakage *= l_ip.nbanks;
  // [한국어] 데이터 배열의 누설 전력을 뱅크 수로 스케일링.

  local_result.data_array2->power.readOp.longer_channel_leakage =
      local_result.data_array2->power.readOp.leakage *
      long_channel_device_reduction;
  // [한국어] 데이터 배열의 장채널 소자 기준 누설 전력 계산.

  local_result.data_array2->power = local_result.data_array2->power * pppm_t;
  // [한국어] 데이터 배열 전력에 pppm_t 스케일링 적용.

  if (!(l_ip.pure_cam || l_ip.pure_ram || l_ip.fully_assoc) && l_ip.is_cache) {
    // [한국어] 태그 배열(tag_array2) 스케일링 조건:
    //   - pure_cam: 순수 CAM → 태그 배열 없음
    //   - pure_ram: 순수 RAM → 태그 배열 없음
    //   - fully_assoc: 완전 연관(FA) → 태그 배열 구조 다름
    //   - is_cache: 캐시 구조일 때만 태그 배열이 존재
    // 세 플래그 모두 false이고 is_cache=true인 set-associative 캐시만 이 블록 실행.

    local_result.tag_array2->power.readOp.dynamic *= sckRation;
    // [한국어] 태그 배열의 읽기 동적 에너지에 소켓 계수 적용.

    local_result.tag_array2->power.writeOp.dynamic *= sckRation;
    // [한국어] 태그 배열의 쓰기 동적 에너지에 소켓 계수 적용.

    local_result.tag_array2->power.searchOp.dynamic *= sckRation;
    // [한국어] 태그 배열의 검색 동적 에너지에 소켓 계수 적용.

    local_result.tag_array2->power.readOp.leakage *= l_ip.nbanks;
    // [한국어] 태그 배열의 누설 전력을 뱅크 수로 스케일링.

    local_result.tag_array2->power.readOp.longer_channel_leakage =
        local_result.tag_array2->power.readOp.leakage *
        long_channel_device_reduction;
    // [한국어] 태그 배열의 장채널 소자 기준 누설 전력 계산.

    local_result.tag_array2->power = local_result.tag_array2->power * pppm_t;
    // [한국어] 태그 배열 전력에 pppm_t 스케일링 적용.
  }
}

/*
 * [한국어]
 * ArrayST::leakage_feedback - 온도 변화에 따른 누설 전력 재계산
 *
 * @temperature: 현재 GPU 온도 (켈빈, K). 시뮬레이션 중 AccelWattch 상위 루틴이
 *   주기적으로 현재 온도를 측정하여 이 함수에 전달한다.
 * @return: void (갱신된 누설 전력은 local_result.power에 반영됨)
 *
 * 반도체 SRAM의 누설 전류(sub-threshold leakage, gate leakage)는 온도에
 * 강하게 의존한다. 온도가 높아질수록 누설 전류가 지수적으로 증가한다.
 * 이 함수는 시뮬레이션 중 GPU 온도가 변할 때 호출되어:
 *   1. l_ip.temp를 10K 단위로 반올림한 새 온도로 업데이트.
 *   2. reconfigure(&l_ip, &local_result)를 호출해 새 온도에 맞는 누설 전력 재계산.
 *      (CACTI 전체 재실행이 아닌 누설만 빠르게 갱신하는 경량 경로)
 *   3. optimize_array()의 스케일링 코드와 동일한 순서로 전력을 재스케일링.
 *
 * 주의: 동적 에너지(dynamic)는 온도와 무관하므로 재계산하지 않는다.
 *   그러나 코드에서는 readOp.dynamic *= sckRation 등을 반복하는 것처럼 보이는데,
 *   이는 reconfigure()가 local_result.power를 초기 CACTI 기준값으로 리셋하기 때문에
 *   sckRation/pppm_t 스케일링을 다시 적용해야 하는 구조다.
 *
 * 실행 컨텍스트: 단일 스레드, 시뮬레이션 전력 집계 단계 (초기화 이후).
 *
 * 호출 체인:
 *   AccelWattch 온도 피드백 루틴 → [ArrayST::leakage_feedback]
 *       → reconfigure() (CACTI 누설 재계산)
 *       → longer_channel_device_reduction()
 */
void ArrayST::leakage_feedback(double temperature) {
  // Update the temperature. l_ip is already set and error-checked in the
  // creator function.
  // [한국어] 온도 업데이트: l_ip는 이미 생성자에서 error_checking()이 완료된 상태이므로
  // 온도 필드(temp)만 변경하면 된다.

  l_ip.temp = (unsigned int)round(temperature / 10.0) * 10;
  // [한국어] 전달된 온도(K)를 10K 단위로 반올림하여 l_ip.temp에 저장.
  //   - round(temperature / 10.0) * 10: 예를 들어 358K → 36 → 360K.
  //   - CACTI는 10K 간격의 사전 계산된 누설 테이블을 사용하므로 이 단위로 반올림.
  //   - (unsigned int) 캐스팅: 음수 온도는 물리적으로 불가능하고 CACTI 내부가 unsigned를 기대.

  // This corresponds to cacti_interface() in the initialization process.
  // Leakage power is updated here.
  // [한국어] 초기화(생성자)에서는 cacti_interface()로 전체를 계산했지만,
  // 온도 변화에 따른 재계산은 reconfigure()만 호출하여 누설 전력만 빠르게 갱신한다.

  reconfigure(&l_ip, &local_result);
  // [한국어] CACTI 내부의 경량 재설정 함수: 새 온도(l_ip.temp)를 기반으로
  // local_result 내의 누설 전력(readOp.leakage 등)을 재계산한다.
  // 면적, 동적 에너지, 타이밍은 변경하지 않는다.
  // 이 호출 후 local_result.power는 CACTI 내부 기준값(스케일링 적용 전)으로 리셋된다.

  // Scale the power values. This is part of ArrayST::optimize_array().
  // [한국어] 아래는 optimize_array()의 스케일링 코드와 동일하다.
  // reconfigure()가 local_result.power를 기준값으로 리셋하므로 스케일링을 다시 적용해야 한다.

  double long_channel_device_reduction =
      longer_channel_device_reduction(device_ty, core_ty);
  // [한국어] optimize_array()와 동일하게 장채널 누설 축소 계수를 계산.
  // device_ty, core_ty는 객체 초기화 시 고정되므로 계수 자체는 동일하나,
  // 기준 누설 값이 온도에 따라 달라지므로 longer_channel_leakage 절대값도 달라진다.

  double macro_layout_overhead = g_tp.macro_layout_overhead;
  // [한국어] 매크로 레이아웃 오버헤드 계수 (optimize_array()와 동일한 전역값).

  double chip_PR_overhead = g_tp.chip_layout_overhead;
  // [한국어] 칩 배선 레이아웃 오버헤드 계수 (optimize_array()와 동일한 전역값).

  double total_overhead = macro_layout_overhead * chip_PR_overhead;
  // [한국어] 두 오버헤드의 곱으로 최종 스케일링 계수를 계산.

  double pppm_t[4] = {total_overhead, 1, 1, total_overhead};
  // [한국어] optimize_array()와 동일한 pppm_t 스케일링 벡터.

  double sckRation = g_tp.sckt_co_eff;
  // [한국어] 소켓 계수 (optimize_array()와 동일한 전역값).

  local_result.power.readOp.dynamic *= sckRation;
  // [한국어] 읽기 동적 에너지에 소켓 계수 재적용 (reconfigure() 후 리셋된 값에 적용).

  local_result.power.writeOp.dynamic *= sckRation;
  // [한국어] 쓰기 동적 에너지에 소켓 계수 재적용.

  local_result.power.searchOp.dynamic *= sckRation;
  // [한국어] 검색 동적 에너지에 소켓 계수 재적용.

  local_result.power.readOp.leakage *= l_ip.nbanks;
  // [한국어] 새로 계산된 누설 전력에 뱅크 수 스케일링 적용.
  // reconfigure()는 단일 뱅크 기준으로 누설을 재계산하므로 nbanks를 다시 곱해야 한다.

  local_result.power.readOp.longer_channel_leakage =
      local_result.power.readOp.leakage * long_channel_device_reduction;
  // [한국어] 새 온도 기반 누설 전력으로부터 장채널 소자 기준 누설 전력을 재계산.
  // 온도가 높아지면 leakage가 증가하고, 이에 비례하여 longer_channel_leakage도 증가한다.

  local_result.power = local_result.power * pppm_t;
  // [한국어] pppm_t 스케일링 적용 (면적 오버헤드 반영).

  local_result.data_array2->power.readOp.dynamic *= sckRation;
  // [한국어] 데이터 배열의 읽기 동적 에너지에 소켓 계수 재적용.

  local_result.data_array2->power.writeOp.dynamic *= sckRation;
  // [한국어] 데이터 배열의 쓰기 동적 에너지에 소켓 계수 재적용.

  local_result.data_array2->power.searchOp.dynamic *= sckRation;
  // [한국어] 데이터 배열의 검색 동적 에너지에 소켓 계수 재적용.

  local_result.data_array2->power.readOp.leakage *= l_ip.nbanks;
  // [한국어] 데이터 배열의 새 누설 전력에 뱅크 수 스케일링 적용.

  local_result.data_array2->power.readOp.longer_channel_leakage =
      local_result.data_array2->power.readOp.leakage *
      long_channel_device_reduction;
  // [한국어] 데이터 배열의 장채널 소자 기준 누설 전력 재계산.

  local_result.data_array2->power = local_result.data_array2->power * pppm_t;
  // [한국어] 데이터 배열 전력에 pppm_t 스케일링 재적용.

  if (!(l_ip.pure_cam || l_ip.pure_ram || l_ip.fully_assoc) && l_ip.is_cache) {
    // [한국어] optimize_array()와 동일한 조건: set-associative 캐시 구조에서만
    // 태그 배열(tag_array2) 전력을 업데이트한다.

    local_result.tag_array2->power.readOp.dynamic *= sckRation;
    // [한국어] 태그 배열의 읽기 동적 에너지에 소켓 계수 재적용.

    local_result.tag_array2->power.writeOp.dynamic *= sckRation;
    // [한국어] 태그 배열의 쓰기 동적 에너지에 소켓 계수 재적용.

    local_result.tag_array2->power.searchOp.dynamic *= sckRation;
    // [한국어] 태그 배열의 검색 동적 에너지에 소켓 계수 재적용.

    local_result.tag_array2->power.readOp.leakage *= l_ip.nbanks;
    // [한국어] 태그 배열의 새 누설 전력에 뱅크 수 스케일링 적용.

    local_result.tag_array2->power.readOp.longer_channel_leakage =
        local_result.tag_array2->power.readOp.leakage *
        long_channel_device_reduction;
    // [한국어] 태그 배열의 장채널 소자 기준 누설 전력 재계산.

    local_result.tag_array2->power = local_result.tag_array2->power * pppm_t;
    // [한국어] 태그 배열 전력에 pppm_t 스케일링 재적용.
  }
}

/*
 * [한국어]
 * ArrayST::~ArrayST - ArrayST 소멸자
 *
 * @return: (소멸자, 반환값 없음)
 *
 * local_result.cleanup()을 호출하여 CACTI가 내부적으로 동적 할당한
 * data_array2, tag_array2 등의 포인터가 가리키는 메모리를 해제한다.
 * uca_org_t 자체는 스택/멤버로 할당되므로 자동 소멸되지만, 내부 포인터가
 * 가리키는 힙 메모리는 cleanup()으로 명시적으로 해제해야 한다.
 *
 * 호출 체인:
 *   GPU 서브유닛 소멸자 → [ArrayST::~ArrayST] → local_result.cleanup()
 *                                              → (CACTI 내부 메모리 해제)
 */
ArrayST::~ArrayST() { local_result.cleanup(); }
// [한국어] local_result(uca_org_t)의 cleanup() 멤버 함수를 호출하여
// CACTI가 동적으로 할당한 data_array2, tag_array2 등 내부 포인터의 메모리를 해제한다.
// 이를 생략하면 ArrayST가 소멸될 때마다 CACTI 내부 구조체가 메모리 누수를 일으킨다.
