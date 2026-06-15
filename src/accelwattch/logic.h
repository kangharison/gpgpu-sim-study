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
/********************************************************************
 *      Modified by:
 ** Jingwen Leng, Univeristy of Texas, Austin                   * Syed Gilani,
 *University of Wisconsin–Madison                * Tayler Hetherington,
 *University of British Columbia         * Ahmed ElTantawy, University of
 *British Columbia             *
 ********************************************************************/
/*
 * [한국어 설명] AccelWattch 로직 전력 모델 헤더 (logic.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPU SM(Streaming Multiprocessor) 내부의 제어 로직 회로들에 대한
 * McPAT 기반 전력 모델 클래스를 선언한다. 구체적으로 명령어 선택 로직,
 * 레지스터 의존성 CAM 검사기, 명령어 디코더, D 플립플롭 셀, 파이프라인
 * 레지스터, 기능 유닛(ALU/MUL/FPU), 미분류(Undifferentiated) 코어 오버헤드
 * 등 7개 클래스를 제공한다. 각 클래스는 CACTI 라이브러리를 호출하여
 * CMOS 트랜지스터 수준의 에너지 계산을 수행하고 결과를 Component 부모 클래스의
 * power 필드에 저장한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * AccelWattch 전력 모델 계층:
 *   GPGPU-Sim gpgpu_sim_wrapper (gpgpu_sim_wrapper.cc/.h)
 *     → accelwattch/ 전력 컴포넌트들 (logic.h, interconnect.h, array.h ...)
 *         → CACTI 라이브러리 (cacti/) — 트랜지스터 수준 에너지 계산
 *             → 결과: Component::power (dynamic + leakage + gate_leakage)
 * 이 파일의 클래스들은 GPU의 각 사이클마다 활성화 카운트(access count)에
 * per_access_energy를 곱하는 방식으로 동적 전력을 추산한다.
 * 실행 컨텍스트: 시뮬레이션 초기화 시 한 번 생성되고, 매 사이클 cycle() 루프에서
 * 카운터 누적 후 computeEnergy()로 결과가 업데이트된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈: CACTI(cacti/), basic_components.h(statsDef/powerDef/CoreDynParam),
 * XML_Parse.h(ParseXML), arch_const.h(상수), parameter.h(InputParameter).
 * 이 파일에 의존하는 모듈: gpu_core.cc와 gpgpu_sim_wrapper.cc가 FunctionalUnit,
 * Pipeline, UndiffCore 등을 직접 생성하여 SM별 전력을 계산한다.
 * 공유 자료구조: Component::power(powerDef), statsDef(tdp_stats/rtp_stats),
 * CoreDynParam(coredynp) — SM 파이프라인 폭, 스레드 수, 클럭 등 구성 정보.
 *
 * === 주요 함수/구조체 요약 ===
 * selection_logic      : 워프 스케줄러 선택 로직의 CAM 구조 전력 (CACTI 활용).
 * dep_resource_conflict_check : RAW 해저드 감지 CAM 비교기 전력 모델.
 * inst_decoder         : 명령어 디코더(x86 2단계 포함) 전력 모델.
 * DFFCell              : D 플립플롭 셀 1개의 스위칭·대기 에너지 분리 모델.
 * Pipeline             : SM 전체 파이프라인 레지스터 집합의 전력 모델.
 * FunctionalUnit       : ALU/곱셈기/FPU 중 하나의 에너지 모델 (per_access_energy 핵심).
 * UndiffCore           : 명시적으로 모델링되지 않은 SM 잔여 회로의 면적·전력 추정.
 */
#ifndef LOGIC_H_
#define LOGIC_H_

#include <cassert>   /* [한국어] assert() 매크로 — CACTI/McPAT 내부 파라미터 유효성 검사용 */
#include <cmath>     /* [한국어] pow(), log2(), sqrt() 등 수학 함수 — 트랜지스터 크기 계산에 사용 */
#include <cstring>   /* [한국어] memset()/memcpy() — InputParameter 구조체 초기화 및 복사에 사용 */
#include <iostream>  /* [한국어] cout/cerr — 에너지 결과 출력 및 경고 메시지 출력 */
#include "XML_Parse.h"          /* [한국어] ParseXML 클래스 — gpgpusim.config에서 가져온 GPU 구성 XML 파싱 결과 제공 */
#include "arch_const.h"         /* [한국어] 아키텍처 상수 (FU_type enum, Device_ty, Core_type 등) 정의 */
#include "basic_components.h"   /* [한국어] statsDef, powerDef, CoreDynParam 등 AccelWattch 공통 자료구조 */
#include "cacti/basic_circuit.h" /* [한국어] CACTI 기본 회로 함수 (gate_C, drain_C, tr_R 등) — 트랜지스터 전기적 특성 계산 */
#include "cacti/cacti_interface.h" /* [한국어] CACTI 캐시 에너지/지연 계산 인터페이스 — UCA(Uniform Cache Access) 결과 구조체 포함 */
#include "cacti/component.h"    /* [한국어] Component 베이스 클래스 — power(powerDef) 필드와 기본 인터페이스 제공 */
#include "cacti/const.h"        /* [한국어] CACTI 내부 상수 (NAND2_LEAK 등) — 누설 전류 계산 기준값 */
#include "cacti/decoder.h"      /* [한국어] Decoder, Predec 클래스 — inst_decoder의 2단계 디코더 모델에 직접 사용 */
#include "cacti/parameter.h"    /* [한국어] InputParameter, TechnologyParameter 구조체 — 공정(technology) 파라미터와 캐시 구성 입력 */
#include "xmlParser.h"          /* [한국어] 저수준 XML 파싱 유틸리티 — ParseXML이 내부적으로 사용하는 파서 */

using namespace std; /* [한국어] std:: 접두사 생략 — string, cout 등을 짧게 쓰기 위한 McPAT 관행 */

/*
 * [한국어]
 * selection_logic — 워프 스케줄러 선택(Wakeup/Select) 로직의 전력 모델
 *
 * 이 클래스는 GPU의 Issue Window(명령어 발행 창)에서 실행 가능한 명령어를
 * 선택하는 회로의 전력 소비를 모델링한다. NVIDIA GPU의 warp scheduler가
 * ready warp를 선택하는 로직과 유사하며, CAM(Content-Addressable Memory) 구조로
 * 모델링된다. win_entries 크기의 issue window에서 매 사이클 issue_width 개의
 * 명령어를 선택하는 전력을 CACTI의 RAM 모델을 통해 추정한다.
 *
 * AccelWattch 맥락: gpgpu_sim_wrapper에서 SM의 warp scheduler 수와 issue width에
 * 따라 생성되며, selection_power()가 호출되어 Component::power를 채운다.
 * 이 값은 이후 EXECU(실행 유닛) 전력의 일부로 합산된다.
 */
class selection_logic : public Component {
 public:
  selection_logic(
      bool _is_default, int win_entries_, int issue_width_,
      const InputParameter *configure_interface,
      enum Device_ty device_ty_ = Core_device,
      enum Core_type core_ty_ = Inorder);  //, const ParseXML *_XML_interface);
  bool is_default;
  /* [한국어] 기본 구성 여부 플래그.
   * 설정자: 생성자의 _is_default 인자로 초기화됨.
   * 읽는 자: selection_power() 내부에서 구성 분기에 사용.
   * 값 범위: true(기본 McPAT 구성) / false(사용자 정의 구성).
   * 동기화: 생성 후 읽기 전용이므로 별도 락 불필요. */

  InputParameter l_ip;
  /* [한국어] CACTI에 전달되는 로컬 입력 파라미터 구조체.
   * 설정자: 생성자에서 configure_interface를 복사하여 win_entries/issue_width에
   *         맞게 조정(캐시 크기, 어소시어티비티 등을 CAM 구조에 매핑).
   * 읽는 자: selection_power()가 CACTI 호출 시 이 구조체를 넘김.
   * 값 범위: CACTI InputParameter 스펙 준수 (배열 크기 ≥ 1, 비트폭 ≥ 1).
   * 동기화: 단일 객체이며 생성 후 selection_power()까지 변경 없음. */

  uca_org_t local_result;
  /* [한국어] CACTI UCA(Uniform Cache Access) 모델이 반환하는 에너지/지연 결과.
   * 설정자: selection_power() 내부에서 cacti_interface()를 호출한 직후 저장.
   * 읽는 자: selection_power()가 동적 에너지·누설 전력을 Component::power에 옮길 때 참조.
   * 값 범위: 유효한 CAM 구조에 대한 CACTI 출력 (access_time, dynamic_energy 등 포함).
   * 동기화: 단일 스레드 초기화 단계에서 한 번만 설정됨. */

  const ParseXML *XML_interface;
  /* [한국어] GPU 구성 XML 데이터의 읽기 전용 포인터.
   * 설정자: 현재 생성자에서는 미사용 (파라미터 주석에 TODO 표시됨).
   * 읽는 자: 미래 확장을 위해 선언만 유지; 현재는 nullptr일 수 있음.
   * 값 범위: 유효한 ParseXML 포인터 또는 nullptr.
   * 동기화: 읽기 전용 포인터이므로 락 불필요. */

  int win_entries;
  /* [한국어] Issue Window(명령어 발행 창)의 엔트리 수 (= CAM 행 수).
   * 설정자: 생성자 인자 win_entries_로 초기화.
   * 읽는 자: selection_power()에서 CAM 크기 설정 시 참조.
   * 값 범위: 1 이상의 양수; GPU 워프 수에 비례 (일반적으로 32~64).
   * 동기화: 생성 후 불변. */

  int issue_width;
  /* [한국어] 매 사이클 선택(issue)할 수 있는 최대 명령어 수 (= CAM 비교기 출력 수).
   * 설정자: 생성자 인자 issue_width_로 초기화.
   * 읽는 자: selection_power()에서 CAM 비교 병렬성 계산 시 사용.
   * 값 범위: 1 이상; GPU SM에서는 보통 1~4 (warp scheduler 수에 대응).
   * 동기화: 생성 후 불변. */

  int num_threads;
  /* [한국어] 동시 실행되는 하드웨어 스레드(워프) 수.
   * 설정자: 생성자 내부에서 CoreDynParam 또는 기본값으로 설정.
   * 읽는 자: selection_power()에서 multi-threaded 선택 논리 전력 스케일링 시 참조.
   * 값 범위: 1 이상; GPU SM에서는 최대 동시 활성 워프 수 (일반적으로 32~64).
   * 동기화: 생성 후 불변. */

  enum Device_ty device_ty;
  /* [한국어] 회로 공정 유형 (Core_device / Uncore_device / LLC_device 등).
   * 설정자: 생성자 인자 device_ty_로 초기화 (기본값 Core_device).
   * 읽는 자: CACTI에 공정 파라미터(Vdd, 누설 전류 등)를 선택할 때 참조.
   * 값 범위: arch_const.h의 Device_ty enum 값.
   * 동기화: 생성 후 불변. */

  enum Core_type core_ty;
  /* [한국어] 코어 유형 (Inorder / OOO — 비순서 실행).
   * 설정자: 생성자 인자 core_ty_로 초기화 (기본값 Inorder).
   * 읽는 자: selection_power()에서 OOO 코어 특화 CAM 구조 분기 시 참조.
   * 값 범위: arch_const.h의 Core_type enum 값 (Inorder / OOO).
   * 동기화: 생성 후 불변. */

  void selection_power();
  /* [한국어] CAM 구조 기반 선택 로직의 전력을 계산하여 Component::power에 저장.
   * CACTI를 사용하여 win_entries × issue_width CAM 어레이의 에너지를 추정한다. */

  void leakage_feedback(double temperature);  // TODO
  /* [한국어] 온도 변화에 따른 누설 전력 재계산 (TODO: 현재 미구현).
   * temperature: 회로 온도(섭씨). 향후 온도-누설 모델 연동 시 사용 예정. */
};

/*
 * [한국어]
 * dep_resource_conflict_check — 레지스터 의존성 CAM 충돌 검사기 전력 모델
 *
 * 이 클래스는 GPU scoreboard(스코어보드)에서 RAW(Read-After-Write) 해저드를
 * 감지하기 위한 CAM(Content-Addressable Memory) 비교기 회로의 전력을 모델링한다.
 * 실제 GPGPU-Sim의 scoreboard(src/gpgpu-sim/scoreboard.cc)가 담당하는 기능의
 * 하드웨어 전력 비용을 추정한다. compare_bits 비트 폭의 태그를 NOR 기반 CMOS
 * 비교기로 비교하는 회로를 모델링하며, 각 트랜지스터 폭(WNORn 등)을 직접 설정하여
 * CACTI 없이 회로 수준 에너지 계산을 수행한다.
 *
 * AccelWattch 맥락: gpgpu_sim_wrapper에서 SM의 레지스터 파일 크기와 명령어 폭에
 * 따라 생성된다. conflict_check_power()가 동적 에너지를, leakage_feedback()이
 * 정적 전력을 업데이트한다.
 */
class dep_resource_conflict_check : public Component {
 public:
  dep_resource_conflict_check(const InputParameter *configure_interface,
                              const CoreDynParam &dyn_p_, int compare_bits_,
                              bool _is_default = true);
  InputParameter l_ip;
  /* [한국어] CACTI 입력 파라미터 — 공정·전압·온도 등 기본 설정.
   * 설정자: 생성자에서 configure_interface를 복사하여 초기화.
   * 읽는 자: compare_cap()과 conflict_check_power()에서 게이트 용량 계산 시 참조.
   * 값 범위: 유효한 CACTI InputParameter 값.
   * 동기화: 생성 후 불변 (leakage_feedback만 내부 온도 파라미터를 수정). */

  uca_org_t local_result;
  /* [한국어] CACTI 호출 결과 저장 구조체.
   * 설정자: conflict_check_power() 내부에서 cacti_interface() 결과를 저장.
   * 읽는 자: conflict_check_power()가 누설·게이트 누설 전력을 Component::power에 복사.
   * 값 범위: 유효 CACTI 출력; CAM 구조 에너지 포함.
   * 동기화: 단일 스레드 초기화 단계에서 설정. */

  double WNORn, WNORp, Wevalinvp, Wevalinvn, Wcompn, Wcompp, Wcomppreequ;
  /* [한국어] NOR 기반 CMOS CAM 비교기 셀의 트랜지스터 폭(단위: 최소 채널 폭의 배수).
   * WNORn   : NOR 게이트 NMOS 트랜지스터 폭 — pull-down 네트워크 드라이브 강도 결정.
   * WNORp   : NOR 게이트 PMOS 트랜지스터 폭 — pull-up 네트워크 드라이브 강도 결정.
   * Wevalinvp: 평가(evaluate) 인버터 PMOS 폭 — 비교 결과 레벨 복원용.
   * Wevalinvn: 평가 인버터 NMOS 폭.
   * Wcompn  : 비교기(comparator) NMOS 폭 — match line 충전/방전 담당.
   * Wcompp  : 비교기 PMOS 폭.
   * Wcomppreequ: 프리차지(pre-equalize) PMOS 폭 — match line을 평가 전 VDD로 충전.
   * 설정자: 생성자에서 공정 파라미터와 compare_bits에 기반하여 계산·초기화.
   * 읽는 자: compare_cap()에서 각 노드 커패시턴스 계산 시 참조.
   * 값 범위: 최소 트랜지스터 폭(min_w_nmos/min_w_pmos)의 1배 이상.
   * 동기화: 생성 후 불변. */

  CoreDynParam coredynp;
  /* [한국어] SM(코어) 동적 파라미터 구조체 — 파이프라인 폭, 스레드 수, 클럭 등.
   * 설정자: 생성자 인자 dyn_p_를 복사하여 초기화.
   * 읽는 자: conflict_check_power()에서 이슈 폭 등 참조하여 CAM 크기 결정.
   * 값 범위: basic_components.h의 CoreDynParam 구조체 전체.
   * 동기화: 생성 후 불변. */

  int compare_bits;
  /* [한국어] CAM 태그 비교 비트 수 (= 레지스터 태그 폭).
   * 설정자: 생성자 인자 compare_bits_로 초기화.
   * 읽는 자: compare_cap()에서 match line 커패시턴스를 compare_bits에 비례하여 계산.
   * 값 범위: 보통 레지스터 파일의 물리 레지스터 번호 비트 수 (예: 6비트 = 64개 레지스터).
   * 동기화: 생성 후 불변. */

  bool is_default;
  /* [한국어] 기본 구성 사용 여부.
   * 설정자: 생성자 인자 _is_default로 초기화 (기본값 true).
   * 읽는 자: conflict_check_power()에서 구성 분기에 사용.
   * 값 범위: true / false.
   * 동기화: 생성 후 불변. */

  statsDef tdp_stats;
  /* [한국어] TDP(Thermal Design Power) 기준 통계 — 최대 부하 시 활성화 카운트.
   * 설정자: 생성자에서 이슈 폭 × 클럭 주기로 초기화.
   * 읽는 자: conflict_check_power()에서 TDP 기준 에너지 계산 시 참조.
   * 값 범위: statsDef 구조체 (readAc, writeAc 카운터 포함).
   * 동기화: 생성 후 불변 (TDP는 고정 수치). */

  statsDef rtp_stats;
  /* [한국어] RTP(Runtime Power) 통계 — 실제 시뮬레이션에서 관측된 활성화 카운트.
   * 설정자: gpgpu_sim_wrapper가 매 사이클 끝에 실제 접근 횟수로 업데이트.
   * 읽는 자: conflict_check_power()에서 runtime 동적 에너지 계산 시 참조.
   * 값 범위: 0 이상의 정수 카운터.
   * 동기화: gpgpu_sim_wrapper의 단일 스레드 사이클 루프에서만 접근. */

  statsDef stats_t;
  /* [한국어] 현재 계산에 사용 중인 통계 — TDP/RTP 중 선택된 값을 임시 저장.
   * 설정자: conflict_check_power() 내부에서 is_tdp 플래그에 따라 tdp_stats 또는
   *         rtp_stats를 복사하여 사용.
   * 읽는 자: power_t 계산 시 이 값을 참조.
   * 값 범위: statsDef 구조체.
   * 동기화: 호출 단위로 갱신되므로 동시 접근 없음. */

  powerDef power_t;
  /* [한국어] 현재 계산 결과 전력값 임시 저장소.
   * 설정자: conflict_check_power()에서 동적·누설·게이트 누설 에너지를 계산하여 저장.
   * 읽는 자: 계산 후 Component::power로 복사.
   * 값 범위: powerDef 구조체 (readOp/writeOp 에너지).
   * 동기화: 단일 스레드 계산 흐름. */

  void conflict_check_power();
  /* [한국어] NOR 기반 CAM 비교기의 동적·누설 전력을 계산하여 Component::power에 저장.
   * compare_cap()으로 노드 커패시턴스를 구하고 α·C·V²·f 공식으로 에너지를 계산한다. */

  double compare_cap();
  /* [한국어] CAM 비교기 셀 1개의 match line 커패시턴스를 반환 (단위: F).
   * WNORn, WNORp, Wcompn 등을 이용해 각 트랜지스터의 드레인/게이트 커패시턴스를 합산. */

  ~dep_resource_conflict_check() { local_result.cleanup(); }
  /* [한국어] 소멸자 — CACTI가 내부적으로 할당한 local_result 메모리를 해제. */

  void leakage_feedback(double temperature);
  /* [한국어] 온도에 따른 누설 전력 재계산 — 시뮬레이션 중 온도가 변하면 호출됨. */
};

/*
 * [한국어]
 * inst_decoder — 명령어 디코더 전력 모델
 *
 * 이 클래스는 GPU SM의 명령어 디코더 회로 전력을 모델링한다. opcode_length 비트의
 * 오피코드를 num_decoders개의 병렬 디코더로 디코딩하는 전력을 CACTI Decoder/Predec
 * 모델을 사용하여 추정한다. x86 플래그가 설정되면 가변 길이 명령어의 pre-decode →
 * final-decode 2단계 구조를 모델링한다. GPU에서는 고정 폭(32비트 SASS 등)이므로
 * 일반적으로 x86=false로 사용된다.
 *
 * AccelWattch 맥락: EXECU 서브시스템 초기화 시 SM의 명령어 폭과 동시 발행 수에
 * 따라 생성된다. inst_decoder_delay_power()가 Decoder/Predec 모델을 호출하여
 * Component::power를 설정한다.
 */
class inst_decoder : public Component {
 public:
  inst_decoder(bool _is_default, const InputParameter *configure_interface,
               int opcode_length_, int num_decoders_, bool x86_,
               enum Device_ty device_ty_ = Core_device,
               enum Core_type core_ty_ = Inorder);
  inst_decoder();
  /* [한국어] 기본 생성자 — 미사용 상태(더미) 디코더 객체 생성 시 호출. */

  bool is_default;
  /* [한국어] 기본 구성 여부 플래그.
   * 설정자: 생성자 인자 _is_default로 초기화.
   * 읽는 자: inst_decoder_delay_power() 내부 분기에서 참조.
   * 값 범위: true / false.
   * 동기화: 생성 후 불변. */

  int opcode_length;
  /* [한국어] 명령어 오피코드 비트 폭.
   * 설정자: 생성자 인자 opcode_length_로 초기화.
   * 읽는 자: inst_decoder_delay_power()에서 Decoder 크기(2^opcode_length 출력) 설정.
   * 값 범위: 일반적으로 6~10비트 (GPU SASS 오피코드 폭에 따라 결정).
   * 동기화: 생성 후 불변. */

  int num_decoders;
  /* [한국어] 병렬로 동작하는 디코더 수 (= 동시 발행 슬롯 수).
   * 설정자: 생성자 인자 num_decoders_로 초기화.
   * 읽는 자: inst_decoder_delay_power()에서 총 에너지 = 단일 디코더 × num_decoders로 스케일링.
   * 값 범위: 1 이상; GPU SM에서는 보통 warp scheduler 수 × 발행 폭.
   * 동기화: 생성 후 불변. */

  bool x86;
  /* [한국어] x86 가변 길이 명령어 2단계 디코더 모델 사용 여부.
   * 설정자: 생성자 인자 x86_으로 초기화.
   * 읽는 자: inst_decoder_delay_power()에서 pre_dec 사용 여부 결정.
   * 값 범위: true(x86 2단계) / false(고정 폭 단일 단계, GPU 기본값).
   * 동기화: 생성 후 불변. */

  int num_decoder_segments;
  /* [한국어] x86 pre-decoder의 세그먼트 수 (x86=true 시만 유효).
   * 설정자: 생성자에서 opcode_length를 기반으로 계산.
   * 읽는 자: inst_decoder_delay_power()에서 pre_dec 단계 전력 계산 시 참조.
   * 값 범위: 0(x86=false) 또는 1 이상의 양수.
   * 동기화: 생성 후 불변. */

  int num_decoded_signals;
  /* [한국어] 디코더 출력 신호 수 (= 디코딩 가능한 명령어 종류 수 = 2^opcode_length).
   * 설정자: 생성자에서 1 << opcode_length 또는 유사 계산으로 설정.
   * 읽는 자: Decoder 객체 생성 시 출력 크기 지정에 사용.
   * 값 범위: 2^opcode_length.
   * 동기화: 생성 후 불변. */

  InputParameter l_ip;
  /* [한국어] CACTI 입력 파라미터 (공정·전압·온도 등).
   * 설정자: 생성자에서 configure_interface를 복사하여 초기화.
   * 읽는 자: Decoder/Predec 객체 생성 시 전달.
   * 값 범위: 유효한 CACTI 파라미터 범위.
   * 동기화: 생성 후 불변. */

  uca_org_t local_result;
  /* [한국어] CACTI 결과 저장용 — 현재 inst_decoder에서는 Decoder/Predec 객체로
   * 직접 계산하므로 간접적으로만 참조됨.
   * 설정자: inst_decoder_delay_power()에서 부분적으로 채워짐.
   * 읽는 자: 누설·게이트 누설 전력을 Component::power로 옮길 때 참조.
   * 값 범위: 유효 CACTI 출력 또는 부분 초기화 상태.
   * 동기화: 단일 스레드 초기화. */

  enum Device_ty device_ty;
  /* [한국어] 회로 공정 유형 (Core_device 등).
   * 설정자: 생성자 인자 device_ty_로 초기화.
   * 읽는 자: CACTI 공정 파라미터 선택 시 참조.
   * 값 범위: Device_ty enum.
   * 동기화: 생성 후 불변. */

  enum Core_type core_ty;
  /* [한국어] 코어 유형 (Inorder / OOO).
   * 설정자: 생성자 인자 core_ty_로 초기화.
   * 읽는 자: inst_decoder_delay_power()에서 OOO 특화 구성 분기 시 참조.
   * 값 범위: Core_type enum.
   * 동기화: 생성 후 불변. */

  Decoder *final_dec;
  /* [한국어] 최종 단계 디코더 객체 포인터 (CACTI Decoder 모델).
   * 설정자: inst_decoder_delay_power()에서 new Decoder(...)로 생성.
   * 읽는 자: inst_decoder_delay_power()에서 final_dec->compute_power() 호출 후
   *         에너지를 Component::power로 합산.
   * 값 범위: 유효한 Decoder 포인터; 소멸자에서 delete됨.
   * 동기화: 단일 스레드 수명주기. */

  Predec *pre_dec;
  /* [한국어] x86 2단계 구조의 pre-decoder 객체 포인터 (x86=false 시 nullptr).
   * 설정자: inst_decoder_delay_power()에서 x86=true일 때만 new Predec(...)로 생성.
   * 읽는 자: pre_dec->compute_power() 호출로 1단계 디코드 에너지 계산.
   * 값 범위: 유효한 Predec 포인터 또는 nullptr.
   * 동기화: 단일 스레드 수명주기. */

  statsDef tdp_stats;
  /* [한국어] TDP 기준 명령어 디코드 활성화 카운트.
   * 설정자: 생성자에서 최대 발행 폭 × 클럭으로 초기화.
   * 읽는 자: inst_decoder_delay_power()에서 TDP 에너지 계산 시 참조.
   * 값 범위: statsDef 구조체.
   * 동기화: 생성 후 불변. */

  statsDef rtp_stats;
  /* [한국어] 실제 시뮬레이션 디코드 횟수 카운터.
   * 설정자: gpgpu_sim_wrapper가 사이클마다 발행된 명령어 수로 업데이트.
   * 읽는 자: inst_decoder_delay_power()에서 runtime 에너지 계산 시 참조.
   * 값 범위: 0 이상 정수.
   * 동기화: 단일 스레드 사이클 루프. */

  statsDef stats_t;
  /* [한국어] TDP/RTP 중 현재 계산 모드에서 사용하는 통계 임시 저장.
   * 설정자: inst_decoder_delay_power() 내부에서 is_tdp 플래그에 따라 복사.
   * 읽는 자: 에너지 계산 시 접근 카운트로 사용.
   * 값 범위: statsDef.
   * 동기화: 호출 단위 갱신. */

  powerDef power_t;
  /* [한국어] 현재 계산된 디코더 전력값 임시 저장.
   * 설정자: inst_decoder_delay_power()에서 Decoder/Predec 에너지 합산 후 저장.
   * 읽는 자: Component::power에 복사.
   * 값 범위: powerDef.
   * 동기화: 단일 스레드. */

  void inst_decoder_delay_power();
  /* [한국어] 디코더의 지연시간과 전력을 계산하여 Component::power에 저장.
   * Decoder/Predec 객체를 생성하고 compute_power()를 호출하여 에너지를 합산한다. */

  ~inst_decoder();
  /* [한국어] 소멸자 — final_dec, pre_dec 포인터를 delete하고 CACTI 결과를 정리. */

  void leakage_feedback(double temperature);
  /* [한국어] 온도 변화에 따른 디코더 누설 전력 재계산. */
};

/*
 * [한국어]
 * DFFCell — D 플립플롭 셀 1개의 전력 모델
 *
 * 이 클래스는 파이프라인 레지스터(래치) 1개를 구성하는 D 플립플롭의 에너지를
 * 스위칭 전이 유형별로 분리하여 모델링한다. 0→1 스위칭(e_switch), 1 유지(e_keep_1),
 * 0 유지(e_keep_0), 클럭 트리(e_clock) 4가지 에너지를 별도 powerDef로 관리한다.
 * fpfp_node_cap()이 팬인/팬아웃에 따른 노드 커패시턴스를 계산하고,
 * compute_DFF_cell()이 각 에너지 성분을 계산한다.
 * Pipeline 클래스가 이 객체를 여러 개 생성하여 SM 전체 파이프라인 레지스터의
 * 총 에너지를 합산한다.
 */
class DFFCell : public Component {
 public:
  DFFCell(bool _is_dram, double _WdecNANDn, double _WdecNANDp,
          double _cell_load, const InputParameter *configure_interface);
  InputParameter l_ip;
  /* [한국어] CACTI 공정 파라미터 — 전압, 게이트 산화막 두께 등.
   * 설정자: 생성자에서 configure_interface 복사.
   * 읽는 자: fpfp_node_cap()에서 트랜지스터 커패시턴스 계산 시 참조.
   * 값 범위: 유효 CACTI InputParameter.
   * 동기화: 생성 후 불변. */

  bool is_dram;
  /* [한국어] DRAM 셀 용량 모델 사용 여부 (true이면 더 큰 셀 커패시턴스 가정).
   * 설정자: 생성자 인자 _is_dram으로 초기화.
   * 읽는 자: fpfp_node_cap()에서 셀 커패시턴스 값 분기 시 참조.
   * 값 범위: true(DRAM 셀 크기) / false(SRAM/로직 셀 크기).
   * 동기화: 생성 후 불변. */

  double cell_load;
  /* [한국어] 플립플롭 출력이 구동하는 부하 커패시턴스 (단위: F).
   * 설정자: 생성자 인자 _cell_load로 초기화.
   * 읽는 자: compute_DFF_cell()에서 스위칭 에너지 계산 시 부하 용량으로 사용.
   * 값 범위: 0 이상의 실수 (전형적으로 fF ~ pF 범위).
   * 동기화: 생성 후 불변. */

  double WdecNANDn;
  /* [한국어] DFF 내부 NAND 게이트의 NMOS 트랜지스터 폭 (단위: 최소 채널 폭 배수).
   * 설정자: 생성자 인자 _WdecNANDn으로 초기화.
   * 읽는 자: compute_DFF_cell()에서 NAND 게이트 커패시턴스/저항 계산 시 사용.
   * 값 범위: 1.0 이상 (최소 채널 폭 기준).
   * 동기화: 생성 후 불변. */

  double WdecNANDp;
  /* [한국어] DFF 내부 NAND 게이트의 PMOS 트랜지스터 폭.
   * 설정자: 생성자 인자 _WdecNANDp으로 초기화.
   * 읽는 자: compute_DFF_cell()에서 NAND 게이트 에너지 계산.
   * 값 범위: 1.0 이상.
   * 동기화: 생성 후 불변. */

  double clock_cap;
  /* [한국어] 클럭 신호 라인의 커패시턴스 (단위: F).
   * 설정자: compute_DFF_cell()에서 트랜지스터 폭과 공정 파라미터로부터 계산.
   * 읽는 자: e_clock 에너지 계산 시 ½·C·V²·α로 사용.
   * 값 범위: 0 이상.
   * 동기화: compute_DFF_cell() 호출 후 불변. */

  int model;
  /* [한국어] DFF 회로 모델 선택 인덱스 (CACTI 내부 모델 번호).
   * 설정자: 생성자에서 is_dram 등 조건에 따라 선택.
   * 읽는 자: compute_DFF_cell() 내부 모델 분기 시 참조.
   * 값 범위: CACTI 정의 모델 인덱스 (일반적으로 0~2).
   * 동기화: 생성 후 불변. */

  int n_switch;
  /* [한국어] 클럭 사이클당 스위칭(0→1 또는 1→0) 전이 횟수 (활성화 계수).
   * 설정자: compute_DFF_cell()에서 회로 구조에 따라 결정 (일반적으로 1~2).
   * 읽는 자: e_switch 에너지 = n_switch × ½·C·V² 계산에 사용.
   * 값 범위: 0 이상의 정수.
   * 동기화: compute_DFF_cell() 후 불변. */

  int n_keep_1;
  /* [한국어] 출력 '1' 유지 시 사이클당 내부 전이 횟수.
   * 설정자: compute_DFF_cell()에서 DFF 구조에 따라 결정.
   * 읽는 자: e_keep_1 에너지 계산 시 사용.
   * 값 범위: 0 이상의 정수.
   * 동기화: compute_DFF_cell() 후 불변. */

  int n_keep_0;
  /* [한국어] 출력 '0' 유지 시 사이클당 내부 전이 횟수.
   * 설정자: compute_DFF_cell()에서 DFF 구조에 따라 결정.
   * 읽는 자: e_keep_0 에너지 계산 시 사용.
   * 값 범위: 0 이상의 정수.
   * 동기화: compute_DFF_cell() 후 불변. */

  int n_clock;
  /* [한국어] 클럭 에지 당 클럭 라인 전이 횟수 (일반적으로 2 — 상승+하강).
   * 설정자: compute_DFF_cell()에서 고정값으로 설정.
   * 읽는 자: e_clock = n_clock × ½·clock_cap·V² 계산에 사용.
   * 값 범위: 보통 2.
   * 동기화: compute_DFF_cell() 후 불변. */

  powerDef e_switch;
  /* [한국어] 1비트 데이터 스위칭(0→1 또는 1→0) 에너지 (단위: J/transition).
   * 설정자: compute_DFF_cell()에서 n_switch × ½·C·V² 공식으로 계산.
   * 읽는 자: Pipeline::compute()에서 num_piperegs × 스위칭 확률 × e_switch로 동적 전력 산출.
   * 값 범위: 0 이상의 실수.
   * 동기화: compute_DFF_cell() 후 불변. */

  powerDef e_keep_1;
  /* [한국어] 데이터 '1' 유지 에너지 (단위: J/cycle) — 내부 재충전 전이로 인한 소비.
   * 설정자: compute_DFF_cell()에서 n_keep_1 × 내부 노드 에너지로 계산.
   * 읽는 자: Pipeline::compute()에서 총 파이프라인 유지 전력 계산 시 사용.
   * 값 범위: 0 이상의 실수.
   * 동기화: compute_DFF_cell() 후 불변. */

  powerDef e_keep_0;
  /* [한국어] 데이터 '0' 유지 에너지 (단위: J/cycle).
   * 설정자: compute_DFF_cell()에서 n_keep_0 × 내부 노드 에너지로 계산.
   * 읽는 자: Pipeline::compute()에서 유지 전력 계산.
   * 값 범위: 0 이상의 실수.
   * 동기화: compute_DFF_cell() 후 불변. */

  powerDef e_clock;
  /* [한국어] 클럭 트리 에너지 (단위: J/cycle) — 클럭 라인 충방전 에너지.
   * 설정자: compute_DFF_cell()에서 n_clock × ½·clock_cap·Vdd² 로 계산.
   * 읽는 자: Pipeline::compute()에서 클럭 전력을 별도 항목으로 합산.
   * 값 범위: 0 이상의 실수.
   * 동기화: compute_DFF_cell() 후 불변. */

  double fpfp_node_cap(unsigned int fan_in, unsigned int fan_out);
  /* [한국어] 팬인(fan_in)과 팬아웃(fan_out)이 주어진 노드의 총 커패시턴스 반환.
   * 게이트 커패시턴스(팬인) + 드레인 커패시턴스(팬아웃)를 합산하여 F 단위로 반환. */

  void compute_DFF_cell(void);
  /* [한국어] 플립플롭 각 에너지 성분(e_switch, e_keep_0/1, e_clock)을 계산.
   * 내부적으로 fpfp_node_cap()을 여러 번 호출하여 각 내부 노드의 커패시턴스를 구한다. */
};

/*
 * [한국어]
 * Pipeline — SM 전체 파이프라인 레지스터의 전력 모델
 *
 * 이 클래스는 GPU SM의 파이프라인 스테이지 간 데이터를 저장하는 모든 DFF 레지스터의
 * 총 에너지를 추정한다. compute_stage_vector()가 coredynp의 파이프라인 폭(fetch/decode/
 * issue/commit 폭)과 데이터 비트 수로부터 총 파이프라인 레지스터 수(num_piperegs)를
 * 계산하고, compute()가 DFFCell을 사용하여 총 전력을 구한다.
 *
 * AccelWattch GPU 맥락: gpgpu_sim_wrapper에서 SM별로 한 번 생성되며, 매 사이클
 * 발행된 명령어 수에 비례하는 스위칭 에너지를 누적한다.
 * process_ind=true이면 공정 독립(process-independent) 기술 스케일링을 적용한다.
 */
class Pipeline : public Component {
 public:
  Pipeline(const InputParameter *configure_interface,
           const CoreDynParam &dyn_p_, enum Device_ty device_ty_ = Core_device,
           bool _is_core_pipeline = true, bool _is_default = true);
  InputParameter l_ip;
  /* [한국어] CACTI 공정 파라미터.
   * 설정자: 생성자에서 configure_interface 복사.
   * 읽는 자: DFFCell 생성 및 compute() 내부에서 참조.
   * 값 범위: 유효 CACTI InputParameter.
   * 동기화: 생성 후 불변. */

  uca_org_t local_result;
  /* [한국어] CACTI 결과 저장 구조체 (누설 전력 저장용).
   * 설정자: compute()에서 DFFCell 계산 결과를 기반으로 채워짐.
   * 읽는 자: 소멸자에서 local_result.cleanup()으로 메모리 해제.
   * 값 범위: 부분 초기화 (파이프라인은 CACTI 전체 호출 없이 직접 계산).
   * 동기화: 단일 스레드. */

  CoreDynParam coredynp;
  /* [한국어] SM 코어 동적 파라미터 — 파이프라인 폭, 레지스터 비트 폭 등.
   * 설정자: 생성자 인자 dyn_p_를 복사.
   * 읽는 자: compute_stage_vector()에서 각 파이프라인 스테이지 레지스터 수 계산 시 참조.
   * 값 범위: CoreDynParam 전체 구조체.
   * 동기화: 생성 후 불변. */

  enum Device_ty device_ty;
  /* [한국어] 회로 공정 유형.
   * 설정자: 생성자 인자 device_ty_로 초기화.
   * 읽는 자: DFFCell 생성 시 공정 파라미터 선택에 참조.
   * 값 범위: Device_ty enum.
   * 동기화: 생성 후 불변. */

  bool is_core_pipeline, is_default;
  /* [한국어] 코어 파이프라인 여부 및 기본 구성 여부 플래그.
   * is_core_pipeline: true이면 SM 코어 파이프라인 레지스터 계산;
   *                   false이면 캐시/메모리 경로 파이프라인.
   * is_default: 기본 McPAT 구성 사용 여부.
   * 설정자: 생성자 인자 _is_core_pipeline, _is_default로 초기화.
   * 읽는 자: compute_stage_vector()에서 레지스터 집합 분기 결정에 참조.
   * 값 범위: true / false.
   * 동기화: 생성 후 불변. */

  double num_piperegs;
  /* [한국어] SM 파이프라인 전체의 총 레지스터 비트 수 (DFF 개수 × 비트폭의 합).
   * 설정자: compute_stage_vector()에서 각 파이프라인 스테이지의 레지스터 수를 합산하여 결정.
   * 읽는 자: compute()에서 총 DFF 에너지 = num_piperegs × 단일 DFF 에너지로 스케일링.
   * 값 범위: 0 이상의 실수 (수백 ~ 수천).
   * 동기화: compute_stage_vector() 호출 후 불변. */

  //	int pipeline_stages;
  //	int tot_stage_vector, per_stage_vector;
  /* [한국어] (주석 처리된 필드) 파이프라인 스테이지 수와 스테이지별 레지스터 수 —
   * 현재는 num_piperegs 하나로 통합되어 사용되지 않음. */

  bool process_ind;
  /* [한국어] 공정 독립(process-independent) 기술 스케일링 적용 여부.
   * 설정자: 생성자에서 공정 노드와 device_ty에 따라 결정.
   * 읽는 자: compute()에서 DFF 에너지 스케일링 팩터 선택 시 참조.
   * 값 범위: true(스케일링 적용) / false(공정별 직접 값 사용).
   * 동기화: 생성 후 불변. */

  double WNANDn;
  /* [한국어] DFF 내부 NAND 게이트 NMOS 폭 — DFFCell 생성자에 전달.
   * 설정자: 생성자에서 공정 최소 트랜지스터 폭으로부터 계산.
   * 읽는 자: compute()에서 DFFCell(WNANDn, WNANDp, ...) 생성 시 사용.
   * 값 범위: 최소 NMOS 폭 이상.
   * 동기화: 생성 후 불변. */

  double WNANDp;
  /* [한국어] DFF 내부 NAND 게이트 PMOS 폭 — DFFCell 생성자에 전달.
   * 설정자: 생성자에서 최소 PMOS 폭으로부터 계산.
   * 읽는 자: compute()에서 DFFCell 생성 시 사용.
   * 값 범위: 최소 PMOS 폭 이상.
   * 동기화: 생성 후 불변. */

  double load_per_pipeline_stage;
  /* [한국어] 파이프라인 스테이지 하나당 배선 부하 커패시턴스 (단위: F).
   * 설정자: 생성자에서 공정 파라미터 및 파이프라인 폭으로부터 추정.
   * 읽는 자: compute()에서 DFFCell의 cell_load로 전달.
   * 값 범위: 0 이상의 실수.
   * 동기화: 생성 후 불변. */

  //	int  Hthread, ... (하드웨어 스레드 수, 발행 폭 등 주석 처리 필드들)
  /* [한국어] (주석 처리된 필드들) 파이프라인 구성 상세 값 — CoreDynParam으로 통합됨. */

  void compute_stage_vector();
  /* [한국어] 각 파이프라인 스테이지(fetch/decode/issue/execute/commit)별 레지스터
   * 비트 수를 합산하여 num_piperegs를 결정한다.
   * coredynp의 파이프라인 폭, 데이터 비트 폭, 명령어 길이 등을 참조한다. */

  void compute();
  /* [한국어] DFFCell을 이용해 파이프라인 레지스터 전체의 동적·정적 전력을 계산하고
   * Component::power에 저장한다. compute_stage_vector() 이후에 호출되어야 한다. */

  ~Pipeline() { local_result.cleanup(); };
  /* [한국어] 소멸자 — CACTI가 할당한 local_result 내부 메모리를 해제. */
};

// class core_pipeline :public pipeline{
// public:
//	int  Hthread,  num_thread, fetchWidth, decodeWidth, issueWidth,
// commitWidth, instruction_length; 	int  PC_width, opcode_length,
// num_arch_reg_tag, data_width,num_phsical_reg_tag, address_width; 	bool
// thread_clock_gated; 	bool in_order, multithreaded; 	core_pipeline(bool
//_is_default, const InputParameter *configure_interface); 	virtual void
// compute_stage_vector();
//
//};
/* [한국어] 위 주석 처리된 core_pipeline 클래스는 Pipeline의 코어 특화 서브클래스로
 * 설계되었으나 현재 Pipeline 단일 클래스로 통합되어 사용하지 않음. */

/*
 * [한국어]
 * FunctionalUnit — ALU/곱셈기/FPU 기능 유닛의 전력 모델
 *
 * 이 클래스는 GPU SM의 기능 유닛(ALU, 정수 곱셈기, FPU) 중 하나를 모델링한다.
 * fu_type으로 유닛 종류를 선택하고, McPAT의 면적/전력 방정식을 적용하여
 * per_access_energy(접근당 에너지)와 base_energy(기본 정적 에너지)를 산출한다.
 * gpgpu_sim_wrapper는 시뮬레이션에서 관측된 ALU/FPU 사용 횟수에 per_access_energy를
 * 곱하여 총 동적 전력을 추산한다.
 *
 * AccelWattch 맥락: EXECU 컴포넌트가 FU_ALU, FU_MUL, FU_FPU 세 종류의
 * FunctionalUnit 객체를 생성하며, gpgpu_sim_wrapper에서 매 사이클 SM별
 * ALU/MUL/FPU 활성화 횟수를 카운트하여 에너지를 누적한다.
 */
class FunctionalUnit : public Component {
 public:
  ParseXML *XML;
  /* [한국어] GPU 구성 XML 데이터 포인터 — SM 수, 클럭, 레지스터 파일 크기 등 접근용.
   * 설정자: 생성자 인자 XML_interface로 초기화.
   * 읽는 자: computeEnergy()에서 SM 구성 파라미터를 읽어 FU 크기/에너지 계산.
   * 값 범위: 유효한 ParseXML 포인터 (nullptr 불가).
   * 동기화: 읽기 전용 접근; 단일 스레드 초기화 단계. */

  int ithCore;
  /* [한국어] 현재 이 객체가 모델링하는 SM(코어)의 인덱스 (0-기반).
   * 설정자: 생성자 인자 ithCore_로 초기화.
   * 읽는 자: computeEnergy()에서 XML에서 해당 SM의 구성 값을 참조할 때 인덱스로 사용.
   * 값 범위: 0 이상, SM 총 수 미만.
   * 동기화: 생성 후 불변. */

  InputParameter interface_ip;
  /* [한국어] CACTI 공정·전압·온도 입력 파라미터.
   * 설정자: 생성자 인자 interface_ip_를 복사하여 초기화.
   * 읽는 자: computeEnergy() 내부 McPAT 면적/전력 계산 루틴에서 참조.
   * 값 범위: 유효 CACTI InputParameter.
   * 동기화: 생성 후 불변 (leakage_feedback만 온도 파라미터 수정). */

  CoreDynParam coredynp;
  /* [한국어] SM 동적 파라미터 — 클럭 주파수, 발행 폭, 스레드 수 등.
   * 설정자: 생성자 인자 dyn_p_를 복사.
   * 읽는 자: computeEnergy()에서 FU 수(num_fu), 클럭 주기 계산 시 참조.
   * 값 범위: CoreDynParam 전체.
   * 동기화: 생성 후 불변. */

  double FU_height;
  /* [한국어] 기능 유닛 회로 레이아웃 높이 (단위: m) — 배선 밀도 계산용.
   * 설정자: computeEnergy()에서 McPAT 면적 모델로부터 산출.
   * 읽는 자: 면적 보고 및 상위 레벨 배치(floor-planning) 참조.
   * 값 범위: 0 이상의 실수.
   * 동기화: computeEnergy() 후 불변. */

  double clockRate, executionTime;
  /* [한국어] SM 클럭 주파수(Hz)와 총 실행 시간(초).
   * clockRate: 생성자 인자 exClockRate로 초기화; per_access_energy 계산의 기준 주기.
   * executionTime: XML에서 읽거나 gpgpu_sim_wrapper가 설정; runtime 전력 계산의 시간 분모.
   * 설정자: 생성자에서 초기화, computeEnergy()에서 일부 업데이트 가능.
   * 읽는 자: computeEnergy()에서 동적 전력 = 에너지 / 실행시간 변환 시 참조.
   * 값 범위: 0 초과 양수.
   * 동기화: 단일 스레드. */

  double num_fu;
  /* [한국어] 이 유닛 유형의 SM 내 인스턴스 수.
   * 설정자: computeEnergy()에서 coredynp와 fu_type에 따라 결정.
   * 읽는 자: computeEnergy()에서 총 에너지 = 단일 FU 에너지 × num_fu 스케일링.
   * 값 범위: 1 이상의 양수.
   * 동기화: computeEnergy() 후 불변. */

  double energy, base_energy, per_access_energy, leakage, gate_leakage;
  /* [한국어] 기능 유닛의 에너지 및 전력 항목들.
   * energy           : 총 동적 에너지 (J) — 단일 접근 에너지 × 접근 횟수.
   * base_energy      : 기본 사이클당 정적 에너지 (J/cycle) — 클럭 게이팅 없을 때 소비.
   * per_access_energy: 접근 1회당 에너지 (J/access) — gpgpu_sim_wrapper의 핵심 곱셈인자.
   * leakage          : 서브스레숄드 누설 전력 (W).
   * gate_leakage     : 게이트 산화막 터널링 누설 전력 (W).
   * 설정자: computeEnergy()에서 McPAT 모델로 계산 후 설정.
   * 읽는 자: gpgpu_sim_wrapper가 per_access_energy × 활성화 횟수로 동적 전력을 누적;
   *          leakage/gate_leakage로 정적 전력을 합산.
   * 값 범위: 0 이상의 실수.
   * 동기화: computeEnergy() 후 gpgpu_sim_wrapper가 읽기 전용으로 접근. */

  bool is_default;
  /* [한국어] 기본 McPAT 구성 사용 여부.
   * 설정자: 생성자 내부에서 결정 (일반적으로 true).
   * 읽는 자: computeEnergy() 내부 분기에서 참조.
   * 값 범위: true / false.
   * 동기화: 생성 후 불변. */

  enum FU_type fu_type;
  /* [한국어] 기능 유닛 종류 선택 (FU_ALU / FU_MUL / FU_FPU).
   * 설정자: 생성자 인자 fu_type으로 초기화.
   * 읽는 자: computeEnergy()에서 유닛별 면적/전력 공식 분기 결정.
   * 값 범위: arch_const.h의 FU_type enum 값.
   * 동기화: 생성 후 불변. */

  statsDef tdp_stats;
  /* [한국어] TDP 기준 FU 활성화 카운트 (최대 부하 가정).
   * 설정자: 생성자에서 발행 폭 × 클럭으로 초기화.
   * 읽는 자: computeEnergy(is_tdp=true) 시 에너지 계산 기준값으로 사용.
   * 값 범위: statsDef.
   * 동기화: 생성 후 불변. */

  statsDef rtp_stats;
  /* [한국어] 실제 시뮬레이션에서의 FU 활성화 횟수.
   * 설정자: gpgpu_sim_wrapper가 매 사이클 SM별 ALU/MUL/FPU 사용 횟수로 업데이트.
   * 읽는 자: computeEnergy(is_tdp=false) 시 runtime 전력 계산 기준.
   * 값 범위: 0 이상 정수.
   * 동기화: 단일 스레드 사이클 루프. */

  statsDef stats_t;
  /* [한국어] 현재 계산에 선택된 통계 임시 저장.
   * 설정자: computeEnergy() 내부에서 is_tdp 플래그에 따라 복사.
   * 읽는 자: 에너지 계산 공식의 활성화 횟수로 사용.
   * 값 범위: statsDef.
   * 동기화: 호출 단위. */

  powerDef power_t;
  /* [한국어] 현재 계산 결과 전력 임시 저장.
   * 설정자: computeEnergy()에서 계산 후 저장.
   * 읽는 자: Component::power로 복사.
   * 값 범위: powerDef.
   * 동기화: 단일 스레드. */

  FunctionalUnit(ParseXML *XML_interface, int ithCore_,
                 InputParameter *interface_ip_, const CoreDynParam &dyn_p_,
                 enum FU_type fu_type, double exClockRate);
  /* [한국어] 생성자 — XML 구성, 공정 파라미터, FU 종류를 받아 객체를 초기화.
   * computeEnergy()는 별도로 호출해야 실제 전력 값이 계산됨. */

  void computeEnergy(bool is_tdp = true);
  /* [한국어] McPAT 모델을 사용하여 FU의 면적/에너지/누설 전력을 계산.
   * is_tdp=true이면 TDP 기준(tdp_stats), false이면 runtime 기준(rtp_stats) 사용.
   * 결과는 per_access_energy, base_energy, leakage, gate_leakage 필드에 저장. */

  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] 기능 유닛 에너지를 사람이 읽을 수 있는 형태로 stdout에 출력.
   * indent: 들여쓰기 공백 수. plevel: 출력 상세 수준. is_tdp: TDP/RTP 선택. */

  void leakage_feedback(double temperature);
  /* [한국어] 온도 변화에 따른 FU 누설 전력 재계산. */
};

/*
 * [한국어]
 * UndiffCore — 미분류(Undifferentiated) 코어 오버헤드 전력 모델
 *
 * 이 클래스는 GPU SM에서 명시적으로 모델링되지 않은 잔여 회로(clock distribution,
 * power gating 컨트롤러, 기타 배선·셀 등)가 소비하는 전력을 추정한다.
 * McPAT의 "undifferentiated core" 개념을 GPU SM에 적용한 것으로, pipeline_stage 수와
 * issue_width로 결정되는 SM 복잡도에 비례하는 면적/전력을 합산한다.
 * exist=false이면 0 전력 모델(해당 기능 없음)로 동작한다.
 *
 * AccelWattch 맥락: Core 객체 초기화 시 SM별로 한 번 생성되며, displayEnergy()를
 * 통해 전력 보고서에 포함된다. 별도의 compute() 없이 생성자에서 계산이 완료된다.
 */
class UndiffCore : public Component {
 public:
  UndiffCore(ParseXML *XML_interface, int ithCore_,
             InputParameter *interface_ip_, const CoreDynParam &dyn_p_,
             bool exist_ = true, bool embedded_ = false);
  /* [한국어] 생성자 — XML 구성과 SM 파라미터로 미분류 코어 전력을 계산.
   * exist_=false이면 모든 전력을 0으로 설정. embedded_=true이면 임베디드 공정 적용. */

  ParseXML *XML;
  /* [한국어] GPU 구성 XML 포인터.
   * 설정자: 생성자 인자 XML_interface로 초기화.
   * 읽는 자: 생성자 내부에서 SM 구성 파라미터 참조.
   * 값 범위: 유효 ParseXML 포인터.
   * 동기화: 읽기 전용. */

  int ithCore;
  /* [한국어] 현재 모델링 대상 SM(코어) 인덱스.
   * 설정자: 생성자 인자 ithCore_으로 초기화.
   * 읽는 자: 생성자에서 XML에서 SM별 구성 접근 시 인덱스로 사용.
   * 값 범위: 0 이상, SM 총 수 미만.
   * 동기화: 생성 후 불변. */

  InputParameter interface_ip;
  /* [한국어] CACTI 공정·전압 파라미터.
   * 설정자: 생성자 인자 interface_ip_를 복사.
   * 읽는 자: 생성자 내부 전력 계산 루틴에서 참조.
   * 값 범위: 유효 CACTI InputParameter.
   * 동기화: 생성 후 불변. */

  CoreDynParam coredynp;
  /* [한국어] SM 동적 파라미터.
   * 설정자: 생성자 인자 dyn_p_를 복사.
   * 읽는 자: 생성자에서 pipeline_stage, issue_width 등을 통해 SM 복잡도 추정.
   * 값 범위: CoreDynParam 전체.
   * 동기화: 생성 후 불변. */

  double clockRate, executionTime;
  /* [한국어] SM 클럭 주파수(Hz)와 총 실행 시간(초).
   * 설정자: 생성자에서 coredynp/XML로부터 설정.
   * 읽는 자: 전력 = 에너지 / 실행시간 계산 시 사용.
   * 값 범위: 0 초과.
   * 동기화: 생성 후 불변. */

  double scktRatio, chip_PR_overhead, macro_PR_overhead;
  /* [한국어] 소켓/칩/매크로 면적 오버헤드 비율 — 미분류 회로의 면적 추정에 사용.
   * scktRatio        : 소켓 레벨 면적 대비 코어 면적 비율.
   * chip_PR_overhead : 칩 전체 기준 면적 오버헤드 비율 (power routing 포함).
   * macro_PR_overhead: 매크로(IP 블록) 단위 오버헤드 비율.
   * 설정자: 생성자에서 McPAT 경험적 계수로 초기화.
   * 읽는 자: 생성자 내부 면적·전력 계산 시 참조.
   * 값 범위: 0.0 ~ 1.0 (비율).
   * 동기화: 생성 후 불변. */

  enum Core_type core_ty;
  /* [한국어] 코어 유형 (Inorder / OOO).
   * 설정자: 생성자에서 coredynp로부터 결정.
   * 읽는 자: 생성자 내부 OOO 특화 오버헤드 분기 시 참조.
   * 값 범위: Core_type enum.
   * 동기화: 생성 후 불변. */

  bool opt_performance, embedded;
  /* [한국어] 성능 최적화 모드 및 임베디드 공정 여부.
   * opt_performance: true이면 성능 최적화 배치 모드 적용.
   * embedded       : true이면 임베디드(저전력) 공정 파라미터 사용.
   * 설정자: 생성자 인자로 초기화.
   * 읽는 자: 생성자 내부 공정·배치 분기 시 참조.
   * 값 범위: true / false.
   * 동기화: 생성 후 불변. */

  double pipeline_stage, num_hthreads, issue_width;
  /* [한국어] SM 파이프라인 스테이지 수, 하드웨어 스레드 수, 발행 폭.
   * pipeline_stage: SM 파이프라인 깊이 — 미분류 오버헤드 크기에 비례.
   * num_hthreads  : 동시 실행 하드웨어 스레드(워프) 수.
   * issue_width   : 사이클당 발행 명령어 수.
   * 설정자: 생성자에서 coredynp로부터 복사.
   * 읽는 자: 생성자 내부에서 미분류 회로 면적 추정에 사용.
   * 값 범위: 1 이상의 양수.
   * 동기화: 생성 후 불변. */

  bool is_default;
  /* [한국어] 기본 McPAT 구성 사용 여부.
   * 설정자: 생성자에서 결정.
   * 읽는 자: 생성자 내부 분기.
   * 값 범위: true / false.
   * 동기화: 생성 후 불변. */

  void displayEnergy(uint32_t indent = 0, int plevel = 100, bool is_tdp = true);
  /* [한국어] 미분류 코어 오버헤드 전력을 stdout에 출력.
   * indent: 들여쓰기 공백 수. plevel: 출력 상세 수준. is_tdp: TDP/RTP 모드. */

  ~UndiffCore(){};
  /* [한국어] 소멸자 — 동적 할당 없으므로 기본 소멸자로 충분. */

  bool exist;
  /* [한국어] 이 미분류 코어 회로의 존재 여부 플래그.
   * 설정자: 생성자 인자 exist_로 초기화 (기본값 true).
   * 읽는 자: displayEnergy()와 생성자에서 exist=false이면 전력을 0으로 처리.
   * 값 범위: true(회로 존재) / false(해당 기능 없는 단순 코어).
   * 동기화: 생성 후 불변. */
};
#endif /* LOGIC_H_ */
