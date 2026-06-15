// $Id: router.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 * [한국어 설명] NoC 라우터 추상 기반 클래스 선언 (router.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 BookSim2 NoC(Network-on-Chip) 시뮬레이터의 라우터 추상 기반 클래스인
 * `Router`를 선언한다. GPGPU-Sim은 GPU 내부의 SM(Streaming Multiprocessor)과 메모리
 * 파티션(L2 캐시/DRAM 컨트롤러) 사이의 패킷 전달을 모델링하기 위해 intersim2를 임베딩하며,
 * 이 파일이 그 중심 추상화 레이어를 제공한다. `Router`는 IQRouter, EventRouter,
 * ChaosRouter 세 가지 구체 구현의 공통 인터페이스를 정의하고, 채널 연결·속도 파라미터·
 * 통계 추적 필드를 캡슐화한다. 구체 라우터는 `NewRouter()` 팩토리를 통해 설정 파일의
 * "router" 키 값("iq"/"event"/"chaos")에 따라 동적으로 선택·생성된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 타이밍 모델에서 NoC(intersim2)는 SM과 메모리 파티션 사이의 데이터 이동
 * 지연을 사이클-레벨로 시뮬레이션하는 서브시스템이다. 전체 흐름은 다음과 같다:
 *
 *   SM(shader.cc) → mem_fetch 생성 → icnt_wrapper.cc → intersim2/network.*
 *     → Router::ReadInputs() → Router::Evaluate() → Router::WriteOutputs()
 *       → 다음 라우터 또는 메모리 파티션
 *
 * `Router`는 TimedModule을 상속하므로 gpu-sim.cc의 사이클 루프가 `Step()`을 호출할 때
 * 마다 `ReadInputs → Evaluate → WriteOutputs` 순서로 자동 실행된다. 이 파일은 헤더이므로
 * 실행 컨텍스트는 호스트 CPU 싱글스레드(GPGPU-Sim 시뮬레이션 메인 루프)이다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - timed_module.hpp : TimedModule 기반 클래스 — 사이클 스텝 프레임워크 제공
 *   - flit.hpp         : Flit 구조체 — NoC에서 실제로 전송되는 최소 전송 단위
 *   - credit.hpp       : Credit 구조체 — 역방향 흐름 제어 신호 (버퍼 가용 알림)
 *   - flitchannel.hpp  : FlitChannel — 라우터 간 flit 전송 채널 (지연 모델 포함)
 *   - channel.hpp      : Channel<T> 제네릭 채널 템플릿 — CreditChannel에 사용
 *   - config_utils.hpp : Configuration — gpgpusim.config 파싱 결과 보유 객체
 *
 * 이 클래스에 의존하는 모듈:
 *   - routers/iq_router.{hpp,cpp}     : 가장 일반적인 IQ(Input-Queued) 라우터 구현
 *   - routers/event_router.{hpp,cpp}  : 이벤트 구동 라우터 구현
 *   - routers/chaos_router.{hpp,cpp}  : 혼돈 라우팅 라우터 구현
 *   - network.*                       : 라우터를 생성하고 채널로 연결하는 네트워크 객체
 *   - icnt_wrapper.cc                 : GPGPU-Sim에서 intersim2를 호출하는 래퍼
 *
 * 데이터 흐름: mem_fetch → Flit 분해 → FlitChannel → Router 입력 버퍼 →
 *              크로스바(crossbar) → FlitChannel → 다음 라우터 또는 목적지
 * 역방향 흐름 제어: 목적지 라우터 → Credit → CreditChannel → 소스 라우터
 *
 * === 주요 함수/구조체 요약 ===
 * - Router() 생성자     : 설정 파일에서 지연·속도·클래스 파라미터를 읽고 통계 배열 초기화
 * - NewRouter()         : 팩토리 함수 — "router" 설정값으로 구체 라우터 타입 결정 후 생성
 * - AddInputChannel()   : 입력 flit 채널과 역방향 credit 채널을 라우터에 연결
 * - AddOutputChannel()  : 출력 flit 채널과 역방향 credit 채널을 라우터에 연결
 * - Evaluate()          : internal_speedup 누산 후 정수 사이클만큼 _InternalStep() 실행
 * - _InternalStep()     : 순수 가상 — 구체 라우터가 1 내부 사이클의 스위치 중재를 수행
 * - ReadInputs()        : 순수 가상 — 입력 채널에서 flit/credit을 읽어 내부 버퍼에 저장
 * - WriteOutputs()      : 순수 가상 — 내부 버퍼에서 flit/credit을 출력 채널에 기록
 * - STALL_* 상수        : 할당 큐에 저장되는 음수 센티널값으로 스톨 이유를 인코딩
 */

#ifndef _ROUTER_HPP_
#define _ROUTER_HPP_

#include <string>   // [한국어] 라우터 이름("router", "iq" 등) 문자열 처리용 STL 헤더
#include <vector>   // [한국어] 채널 포인터 및 통계 카운터를 동적 배열로 관리하기 위한 STL 헤더

#include "timed_module.hpp"  // [한국어] TimedModule: 사이클-레벨 시뮬레이션 프레임워크 기반 클래스 — Step()이 ReadInputs/Evaluate/WriteOutputs를 순서대로 호출
#include "flit.hpp"          // [한국어] Flit: 라우터 간 전송 최소 단위 — head/body/tail 타입 구분, 패킷 ID, VC(가상 채널) 번호, 목적지 등 보유
#include "credit.hpp"        // [한국어] Credit: 역방향 흐름 제어 신호 구조체 — 다운스트림 라우터가 버퍼 슬롯을 해제할 때 업스트림으로 전송
#include "flitchannel.hpp"   // [한국어] FlitChannel: 두 라우터 사이의 flit 전송 채널 — 전파 지연(latency) 모델 포함
#include "channel.hpp"       // [한국어] Channel<T>: 제네릭 채널 템플릿 — CreditChannel = Channel<Credit> 인스턴스화에 사용
#include "config_utils.hpp"  // [한국어] Configuration: gpgpusim.config 파싱 결과를 보유하는 키-값 저장소 — GetInt/GetFloat/GetStr 인터페이스 제공

// [한국어] CreditChannel 타입 별칭 정의.
// Channel<Credit>을 CreditChannel로 명명하여 FlitChannel과 대칭되는 이름으로 사용한다.
// FlitChannel이 flit을 순방향으로 전달하는 채널이라면, CreditChannel은 credit(흐름 제어 신호)을
// 역방향으로 전달하는 채널이다. 두 채널이 쌍으로 구성되어 wormhole 또는 VC 흐름 제어를 구현한다.
typedef Channel<Credit> CreditChannel;

// [한국어] Router: NoC 라우터 추상 기반 클래스.
// TimedModule을 상속하여 사이클 스텝 프레임워크에 자동으로 참여하며,
// IQRouter / EventRouter / ChaosRouter 세 가지 구체 구현의 공통 인터페이스를 정의한다.
// 생성자에서 채널 연결, 속도 파라미터, 통계 배열을 초기화하고,
// 순수 가상 함수(ReadInputs, _InternalStep, WriteOutputs)를 통해 구체 라우터가 자신의
// 스위치 중재 알고리즘을 구현하도록 강제한다.
class Router : public TimedModule {

protected:

  // ─────────────────────────────────────────────────────────────────────────
  // 스톨(stall) 이유 코드 — 음수 센티널 상수
  // ─────────────────────────────────────────────────────────────────────────
  // [한국어] STALL 상수들은 라우터 내부 할당 큐에 flit이 아닌 "스톨 이유"를
  // 기록할 때 사용하는 음수 센티널 값이다. 양수는 출력 포트 번호이고,
  // 음수는 스톨 상태임을 나타내므로 별도 타입 없이 int 하나로 구분된다.
  // 설정자: IQRouter의 스위치 중재 로직이 스톨을 검출했을 때 기록.
  // 읽는 자: TRACK_STALLS 집계 코드가 사이클 종료 시 카운터 증가에 참조.
  // 값 범위: -2 ~ -6 (각각 BUSY/CONFLICT/FULL/RESERVED/CROSSBAR_CONFLICT).
  // 동기화: 시뮬레이션은 단일 스레드이므로 별도 락 불필요.

  static int const STALL_BUFFER_BUSY;
  /* [한국어] 버퍼가 이미 다른 VC(가상 채널) 할당으로 점유된 상태에서 새 flit이
   * 도착했을 때 기록하는 스톨 코드. 값 = -2.
   * 원인: 동일 입력 포트의 여러 VC가 경합하여 한 VC의 버퍼가 처리 중인 상황.
   * 설정자: IQRouter의 VC 할당 단계에서 해당 버퍼가 점유 중임을 확인 시 기록.
   * 읽는 자: GetBufferBusyStalls() / ResetStallStats()를 통해 통계 시스템이 조회. */

  static int const STALL_BUFFER_CONFLICT;
  /* [한국어] 여러 flit이 동일 버퍼 슬롯을 동시에 요청하여 충돌이 발생한 스톨 코드. 값 = -3.
   * 원인: 입력 속도 배수(_input_speedup)가 적용된 가상 포트 간 버퍼 뱅크 충돌.
   * 설정자: IQRouter의 버퍼 뱅크 중재 단계에서 충돌 검출 시 기록.
   * 읽는 자: GetBufferConflictStalls() / ResetStallStats()로 통계 조회. */

  static int const STALL_BUFFER_FULL;
  /* [한국어] 입력 버퍼의 모든 슬롯이 소진되어 새 flit을 수용할 수 없는 스톨 코드. 값 = -4.
   * 원인: 다운스트림 라우터가 credit을 충분히 반환하지 않아 버퍼가 포화된 상태.
   * 설정자: IQRouter가 credit 수를 확인한 후 수용 불가 판단 시 기록.
   * 읽는 자: GetBufferFullStalls()로 혼잡도 분석 시 참조. */

  static int const STALL_BUFFER_RESERVED;
  /* [한국어] 버퍼 슬롯이 특정 패킷을 위해 예약(reserved)되어 다른 flit이 사용 불가한 스톨 코드. 값 = -5.
   * 원인: VC(가상 채널) 예약 메커니즘이 활성화되어 특정 VC 전용으로 슬롯이 잠긴 상태.
   * 설정자: IQRouter의 VC 예약 체크 단계에서 예약된 슬롯을 확인 시 기록.
   * 읽는 자: GetBufferReservedStalls()로 VC 예약 효율 분석 시 참조. */

  static int const STALL_CROSSBAR_CONFLICT;
  /* [한국어] 크로스바(crossbar) 스위치에서 두 flit이 동일 출력 포트를 요청하여 충돌한 스톨 코드. 값 = -6.
   * 원인: 여러 입력 포트의 flit이 동일 출력 포트로 동시 전달을 시도하는 구조적 충돌.
   * 설정자: IQRouter의 스위치 중재(switch arbitration) 단계에서 충돌 검출 시 기록.
   * 읽는 자: GetCrossbarConflictStalls()로 크로스바 포화도 분석 시 참조. */

  // ─────────────────────────────────────────────────────────────────────────
  // 라우터 식별 및 포트 구성
  // ─────────────────────────────────────────────────────────────────────────

  int _id;
  /* [한국어] 이 라우터의 네트워크 내 고유 식별자(ID).
   * 설정자: Router 생성자에서 `id` 파라미터로 전달되어 초기화됨.
   *         NewRouter()를 통해 network.*가 토폴로지 순서대로 할당.
   * 읽는 자: GetID() 인라인 함수로 외부에서 조회 가능.
   *          IQRouter 등 구체 클래스가 디버그 출력이나 assert에서 자신의 위치 확인에 사용.
   * 값 범위: 0 이상의 정수, 네트워크 내 총 라우터 수 미만.
   * 동기화: 생성 후 변경되지 않으므로(read-only) 락 불필요. */

  int _inputs;
  /* [한국어] 이 라우터의 물리 입력 포트 수.
   * 설정자: Router 생성자에서 `inputs` 파라미터로 전달되어 초기화됨.
   *         토폴로지(mesh, torus 등)와 SM 수에 따라 network.*가 결정.
   * 읽는 자: AddInputChannel()에서 채널 인덱스 범위 확인에 사용.
   *          NumInputs() 인라인으로 외부 조회.
   *          IQRouter가 입력 버퍼 배열 크기 계산에 사용.
   * 값 범위: 1 이상; 전형적인 GPU NoC mesh에서는 방향 수 + 인젝션 포트 수.
   * 동기화: 생성 후 불변(read-only). */

  int _outputs;
  /* [한국어] 이 라우터의 물리 출력 포트 수.
   * 설정자: Router 생성자에서 `outputs` 파라미터로 전달됨.
   * 읽는 자: AddOutputChannel()에서 _channel_faults 배열 크기 동기화에 사용.
   *          NumOutputs() 인라인으로 외부 조회.
   *          IQRouter가 출력 통계 배열(TRACK_FLOWS: _sent_flits 등) 크기 결정에 사용.
   * 값 범위: _inputs와 동일하거나 유사; 토폴로지에 따라 다름.
   * 동기화: 생성 후 불변(read-only). */

  int _classes;
  /* [한국어] 이 라우터가 처리하는 트래픽 클래스(QoS 클래스) 수.
   * 설정자: Router 생성자에서 config.GetInt("classes")로 읽힘.
   *         gpgpusim.config의 "classes" 옵션으로 설정.
   * 읽는 자: TRACK_FLOWS / TRACK_STALLS 배열의 첫 번째 차원 크기로 사용.
   *          GetReceivedFlits(c), GetBufferBusyStalls(c) 등에서 인덱스 범위 검증에 사용.
   * 값 범위: 1 이상; 일반적으로 GPGPU-Sim에서는 1(단일 클래스) 사용.
   * 동기화: 생성 후 불변(read-only). */

  // ─────────────────────────────────────────────────────────────────────────
  // 속도 배수(speedup) 파라미터
  // ─────────────────────────────────────────────────────────────────────────

  int _input_speedup;
  /* [한국어] 입력 포트 가상 확장 배수(input port speedup).
   * 물리 입력 포트 하나를 _input_speedup개의 가상 서브포트로 확장하여
   * 같은 사이클에 더 많은 flit을 처리할 수 있도록 대역폭을 모델링한다.
   * 설정자: Router 생성자에서 config.GetInt("input_speedup")으로 읽힘.
   *         gpgpusim.config의 "input_speedup" 옵션; 기본값은 통상 1(확장 없음).
   * 읽는 자: IQRouter가 입력 버퍼 뱅크 수 및 중재 슬롯 계산에 사용.
   * 값 범위: 1 이상의 정수; 1이면 가상 확장 없음.
   * 동기화: 생성 후 불변(read-only). */

  int _output_speedup;
  /* [한국어] 출력 포트 가상 확장 배수(output port speedup).
   * _input_speedup과 대칭적으로 출력 측 대역폭을 가상 확장한다.
   * 설정자: Router 생성자에서 config.GetInt("output_speedup")으로 읽힘.
   * 읽는 자: IQRouter가 출력 크로스바 슬롯 계산에 사용.
   * 값 범위: 1 이상의 정수; 통상 1.
   * 동기화: 생성 후 불변(read-only). */

  double _internal_speedup;
  /* [한국어] 내부 처리 속도 배수(internal speedup, 분수 사이클 지원).
   * 외부 사이클(gpu-sim 메인 루프) 대비 라우터 내부 처리 속도의 비율이다.
   * 예: 1.5이면 2 외부 사이클마다 3번의 내부 스텝(_InternalStep)을 실행한다.
   * 이를 통해 라우터가 네트워크 클럭과 코어 클럭의 비율을 유연하게 모델링할 수 있다.
   * 설정자: Router 생성자에서 config.GetFloat("internal_speedup")으로 읽힘.
   *         gpgpusim.config의 "internal_speedup" 옵션.
   * 읽는 자: Evaluate()에서 _partial_internal_cycles에 더해 정수 사이클 추출에 사용.
   * 값 범위: 0.0 초과 실수; 1.0이면 외부 사이클과 동일 속도.
   * 동기화: 생성 후 불변(read-only). */

  double _partial_internal_cycles;
  /* [한국어] internal_speedup의 분수(소수) 사이클 누산 어큐뮬레이터.
   * Evaluate()가 호출될 때마다 _internal_speedup을 더하고, 1.0 이상이 되면
   * 그만큼 _InternalStep()을 실행한 뒤 1.0을 뺀다. 이를 통해 분수 배속을 정확하게
   * 구현한다(예: 1.5 배속이면 사이클 1→1.5, 사이클 2→1.0+0.5=0.0+step, 사이클 3→0.5+1.5=1.0→step).
   * 설정자: Router 생성자에서 0.0으로 초기화; Evaluate()가 매 사이클 _internal_speedup을 더함.
   * 읽는 자: Evaluate() 내부에서만 참조.
   * 값 범위: 0.0 이상 _internal_speedup 미만 (항상 1.0 미만 잔류).
   * 동기화: 단일 스레드 시뮬레이션이므로 락 불필요. */

  // ─────────────────────────────────────────────────────────────────────────
  // 파이프라인 지연 파라미터
  // ─────────────────────────────────────────────────────────────────────────

  int _crossbar_delay;
  /* [한국어] 크로스바(crossbar) 스위치 통과 지연(사이클 단위).
   * gpgpusim.config의 "st_prepare_delay"(스위치 트래버설 준비 단계 지연)와
   * "st_final_delay"(스위치 트래버설 최종 단계 지연)의 합으로 계산된다.
   * 이 값은 flit이 입력 버퍼에서 출력 채널로 이동하는 데 걸리는 파이프라인 단계 수를 모델링한다.
   * 설정자: Router 생성자에서 st_prepare_delay + st_final_delay로 계산되어 초기화.
   * 읽는 자: IQRouter의 파이프라인 단계 스케줄링 로직에서 전달 지연 계산에 사용.
   * 값 범위: 0 이상의 정수; 값이 클수록 라우터 파이프라인이 깊음을 의미.
   * 동기화: 생성 후 불변(read-only). */

  int _credit_delay;
  /* [한국어] credit 신호의 전파 지연(사이클 단위).
   * 다운스트림 라우터가 버퍼를 해제하고 credit을 보낸 후, 업스트림 라우터가
   * 이를 수신하기까지 걸리는 지연을 모델링한다. 이 지연이 크면 파이프라인 버블이 발생할 수 있다.
   * 설정자: Router 생성자에서 config.GetInt("credit_delay")로 읽힘.
   *         gpgpusim.config의 "credit_delay" 옵션.
   * 읽는 자: IQRouter가 credit 수신 타이밍 계산 시 참조.
   * 값 범위: 0 이상의 정수.
   * 동기화: 생성 후 불변(read-only). */

  // ─────────────────────────────────────────────────────────────────────────
  // 채널 포인터 배열
  // ─────────────────────────────────────────────────────────────────────────

  vector<FlitChannel *>   _input_channels;
  /* [한국어] 이 라우터의 각 입력 포트에 연결된 FlitChannel 포인터 배열.
   * _input_channels[i]는 i번 입력 포트로 flit이 들어오는 채널을 가리킨다.
   * AddInputChannel() 호출 시 순서대로 push_back되므로 인덱스 = 포트 번호.
   * 설정자: AddInputChannel()에서 push_back으로 추가됨; 네트워크 초기화 시 설정.
   * 읽는 자: ReadInputs() 구현(IQRouter 등)이 매 사이클 이 채널에서 flit을 읽음.
   *          GetInputChannel(int)이 외부에서 특정 포트 채널 조회 시 사용.
   * 값 범위: 크기는 _inputs와 같아야 함; 각 포인터는 null이 아닌 유효한 채널.
   * 동기화: 초기화 후 포인터 자체는 변경되지 않음(read-only); 채널 내부 상태는 채널이 관리. */

  vector<CreditChannel *> _input_credits;
  /* [한국어] 이 라우터가 업스트림으로 credit을 되돌려 보내는 역방향 채널 배열.
   * _input_credits[i]는 i번 입력 포트의 업스트림 라우터에게 credit을 전송하는 채널이다.
   * flit이 이 라우터의 입력 버퍼에서 처리되어 슬롯이 해제될 때, 해당 credit 채널로 신호를 보낸다.
   * 설정자: AddInputChannel()에서 backchannel 파라미터로 전달되어 push_back됨.
   * 읽는 자: IQRouter의 WriteOutputs() 또는 credit 전송 단계에서 사용.
   * 값 범위: 크기는 _inputs와 동일; 각 포인터는 유효한 CreditChannel.
   * 동기화: 초기화 후 포인터 불변; 채널 내부 상태는 채널이 관리. */

  vector<FlitChannel *>   _output_channels;
  /* [한국어] 이 라우터의 각 출력 포트에 연결된 FlitChannel 포인터 배열.
   * _output_channels[i]는 i번 출력 포트로 flit을 내보내는 채널을 가리킨다.
   * AddOutputChannel() 호출 시 순서대로 push_back된다.
   * 설정자: AddOutputChannel()에서 push_back으로 추가됨.
   * 읽는 자: WriteOutputs() 구현(IQRouter 등)이 매 사이클 이 채널에 flit을 기록.
   *          GetOutputChannel(int)이 외부에서 특정 포트 채널 조회 시 사용.
   * 값 범위: 크기는 _outputs와 같아야 함.
   * 동기화: 초기화 후 포인터 불변. */

  vector<CreditChannel *> _output_credits;
  /* [한국어] 다운스트림 라우터가 업스트림(이 라우터)으로 credit을 보내는 역방향 채널 배열.
   * _output_credits[i]는 i번 출력 포트 너머의 라우터에서 credit이 들어오는 채널이다.
   * 이 라우터의 ReadInputs() 또는 Evaluate() 단계에서 이 채널을 폴링하여 credit을 수신하고,
   * 출력 버퍼의 사용 가능 슬롯 카운터를 업데이트한다.
   * 설정자: AddOutputChannel()에서 backchannel 파라미터로 전달되어 push_back됨.
   * 읽는 자: IQRouter의 credit 수신 단계에서 사용.
   * 값 범위: 크기는 _outputs와 동일.
   * 동기화: 초기화 후 포인터 불변. */

  vector<bool>            _channel_faults;
  /* [한국어] 각 출력 채널의 결함(fault) 상태 플래그 배열.
   * _channel_faults[i]가 true이면 i번 출력 채널은 결함 상태로 간주되어
   * flit 전송 대상에서 제외된다. 결함 허용(fault-tolerant) 라우팅 연구에 사용된다.
   * GPGPU-Sim 기본 설정에서는 모든 채널이 정상(false)이다.
   * 설정자: AddOutputChannel()에서 false로 초기화됨; OutChannelFault()로 변경 가능.
   * 읽는 자: IsFaultyOutput()이 라우팅 결정 시 채널 사용 가능 여부 확인에 사용.
   *          IQRouter 등 구체 라우터가 출력 포트 선택 시 참조.
   * 값 범위: true(결함) / false(정상).
   * 동기화: 단일 스레드 시뮬레이션이므로 락 불필요. */

  // ─────────────────────────────────────────────────────────────────────────
  // TRACK_FLOWS 조건부 컴파일 — 흐름 추적 통계 배열
  // ─────────────────────────────────────────────────────────────────────────
  // [한국어] TRACK_FLOWS 매크로가 정의된 경우에만 컴파일되는 흐름 추적 배열들이다.
  // 이 배열들은 라우터를 통과하는 flit과 크레딧의 수를 트래픽 클래스별·포트별로 카운팅한다.
  // 운영(production) 시뮬레이션에서는 오버헤드를 줄이기 위해 비활성화하는 것이 일반적이다.
  // 배열의 첫 번째 차원: 트래픽 클래스 인덱스 (0 ~ _classes-1)
  // 배열의 두 번째 차원: 포트 인덱스 (입력 배열은 0~_inputs-1, 출력 배열은 0~_outputs-1)

#ifdef TRACK_FLOWS
  vector<vector<int> > _received_flits;
  /* [한국어] 각 클래스별·입력 포트별로 수신된 flit 수를 누산하는 2D 카운터.
   * _received_flits[class][input_port]가 해당 클래스의 해당 입력 포트로 들어온 flit 총수.
   * 설정자: 생성자에서 resize(_classes, vector<int>(_inputs, 0))으로 0 초기화.
   *         IQRouter의 flit 수신 코드에서 수신 시 1씩 증가.
   *         ResetFlowStats(c)로 특정 클래스의 카운터를 0으로 리셋 가능.
   * 읽는 자: GetReceivedFlits(c)로 외부(통계 수집 코드)가 조회.
   * 값 범위: 0 이상의 누적 카운터.
   * 동기화: 단일 스레드 시뮬레이션. */

  vector<vector<int> > _stored_flits;
  /* [한국어] 각 클래스별로 현재 라우터 내부 버퍼에 저장 중인 flit 수를 추적하는 2D 배열.
   * 두 번째 차원의 크기는 구체 라우터 초기화 시 결정된다(router.cpp 생성자에서는 resize만 수행).
   * 설정자: IQRouter 생성자가 내부 버퍼 구성에 맞게 두 번째 차원을 설정.
   * 읽는 자: GetStoredFlits(c)로 버퍼 점유율 모니터링 시 사용.
   * 값 범위: 0 이상; 라우터 총 버퍼 슬롯 수 이하.
   * 동기화: 단일 스레드. */

  vector<vector<int> > _sent_flits;
  /* [한국어] 각 클래스별·출력 포트별로 전송 완료된 flit 수 누산 카운터.
   * _sent_flits[class][output_port]가 해당 클래스의 해당 출력 포트로 나간 flit 총수.
   * 설정자: 생성자에서 resize(_classes, vector<int>(_outputs, 0))으로 0 초기화.
   *         IQRouter의 flit 전송 코드에서 1씩 증가.
   *         ResetFlowStats(c)로 리셋 가능.
   * 읽는 자: GetSentFlits(c)로 처리량(throughput) 분석 시 사용.
   * 값 범위: 0 이상의 누적 카운터.
   * 동기화: 단일 스레드. */

  vector<vector<int> > _outstanding_credits;
  /* [한국어] 각 클래스별·출력 포트별로 아직 회수되지 않은 미결 credit 수 카운터.
   * flit을 출력 포트로 보낼 때 1씩 증가하고, 다운스트림으로부터 credit이 돌아오면 1씩 감소한다.
   * 이 값이 크면 다운스트림 버퍼가 포화되어 흐름 제어 역압이 발생 중임을 의미한다.
   * 설정자: 생성자에서 resize(_classes, vector<int>(_outputs, 0))으로 0 초기화.
   *         IQRouter의 flit 전송/credit 수신 코드에서 증감.
   * 읽는 자: GetOutstandingCredits(c)로 역압 분석 시 사용.
   * 값 범위: 0 이상; 다운스트림 버퍼 총 슬롯 수 이하.
   * 동기화: 단일 스레드. */

  vector<vector<int> > _active_packets;
  /* [한국어] 각 클래스별로 현재 이 라우터를 통과 중인 활성 패킷 수 카운터.
   * 두 번째 차원의 크기는 구체 라우터 초기화 시 결정된다.
   * head flit 도착 시 1 증가, tail flit 전송 완료 시 1 감소.
   * 설정자: IQRouter가 head/tail flit 검출 시 갱신.
   * 읽는 자: GetActivePackets(c)로 패킷 수준 점유율 모니터링 시 사용.
   * 값 범위: 0 이상의 정수.
   * 동기화: 단일 스레드. */
#endif

  // ─────────────────────────────────────────────────────────────────────────
  // TRACK_STALLS 조건부 컴파일 — 스톨 통계 배열
  // ─────────────────────────────────────────────────────────────────────────
  // [한국어] TRACK_STALLS 매크로가 정의된 경우에만 컴파일되는 스톨 카운터 배열들이다.
  // 각 배열의 크기는 _classes로, 트래픽 클래스별 스톨 빈도를 집계한다.
  // 이 카운터들은 NoC 병목 지점 분석 및 gpgpusim.config 파라미터 튜닝에 활용된다.

#ifdef TRACK_STALLS
  vector<int> _buffer_busy_stalls;
  /* [한국어] 클래스별 STALL_BUFFER_BUSY 발생 횟수 누산 카운터.
   * 설정자: IQRouter가 버퍼 점유 충돌을 감지할 때마다 해당 클래스 인덱스의 값을 1 증가.
   *         ResetStallStats(c)로 특정 클래스 카운터를 0으로 리셋.
   * 읽는 자: GetBufferBusyStalls(c)로 통계 집계 코드가 조회.
   * 값 범위: 0 이상의 누적 카운터.
   * 동기화: 단일 스레드. */

  vector<int> _buffer_conflict_stalls;
  /* [한국어] 클래스별 STALL_BUFFER_CONFLICT 발생 횟수 누산 카운터.
   * 설정자: IQRouter의 버퍼 뱅크 충돌 검출 코드에서 증가.
   * 읽는 자: GetBufferConflictStalls(c)로 조회.
   * 값 범위: 0 이상.
   * 동기화: 단일 스레드. */

  vector<int> _buffer_full_stalls;
  /* [한국어] 클래스별 STALL_BUFFER_FULL 발생 횟수 누산 카운터.
   * 설정자: IQRouter의 credit 부족 확인 코드에서 증가.
   * 읽는 자: GetBufferFullStalls(c)로 조회; 혼잡도 분석의 핵심 지표.
   * 값 범위: 0 이상.
   * 동기화: 단일 스레드. */

  vector<int> _buffer_reserved_stalls;
  /* [한국어] 클래스별 STALL_BUFFER_RESERVED 발생 횟수 누산 카운터.
   * 설정자: IQRouter의 VC 예약 검사 코드에서 증가.
   * 읽는 자: GetBufferReservedStalls(c)로 조회.
   * 값 범위: 0 이상.
   * 동기화: 단일 스레드. */

  vector<int> _crossbar_conflict_stalls;
  /* [한국어] 클래스별 STALL_CROSSBAR_CONFLICT 발생 횟수 누산 카운터.
   * 설정자: IQRouter의 스위치 중재 코드에서 크로스바 충돌 감지 시 증가.
   * 읽는 자: GetCrossbarConflictStalls(c)로 조회; 크로스바 포화 분석에 사용.
   * 값 범위: 0 이상.
   * 동기화: 단일 스레드. */
#endif

  /*
   * [한국어]
   * _InternalStep - 라우터 1 내부 사이클 처리 순수 가상 함수
   *
   * @return: 없음 (void)
   *
   * 이 함수는 라우터의 한 내부 사이클 동안 수행해야 할 모든 스위치 중재(arbitration),
   * VC(가상 채널) 할당, 버퍼 이동을 구현한다. Evaluate()에 의해 _internal_speedup에
   * 따라 외부 사이클당 1회 이상 호출될 수 있다. 구체 라우터(IQRouter 등)가 반드시
   * 오버라이드해야 하는 순수 가상 함수이다.
   *
   * 실행 컨텍스트: 호스트 CPU 단일 스레드, Evaluate() 내부에서 호출.
   *
   * 호출 체인:
   *   gpu-sim.cc::cycle() → TimedModule::Step() → Evaluate() → [_InternalStep()]
   */
  virtual void _InternalStep() = 0;

public:
  /*
   * [한국어]
   * Router - NoC 라우터 기반 클래스 생성자
   *
   * @config  : gpgpusim.config 파싱 결과 객체 — 라우터 파라미터 조회에 사용
   * @parent  : 이 라우터를 포함하는 상위 Module 객체 (계층적 이름 생성에 사용)
   * @name    : 이 라우터의 식별 이름 문자열 (디버그 출력, Module 트리에 등록)
   * @id      : 네트워크 내 라우터 고유 번호 (0-based)
   * @inputs  : 물리 입력 포트 수
   * @outputs : 물리 출력 포트 수
   * @return  : 없음 (생성자)
   *
   * TimedModule 기반 클래스를 초기화하고, 설정 파일에서 파이프라인 지연·속도 배수·
   * 클래스 수를 읽어 필드를 초기화한다. TRACK_FLOWS / TRACK_STALLS가 정의된 경우
   * 해당 통계 배열을 클래스 수·포트 수에 맞게 resize하고 0으로 초기화한다.
   *
   * 실행 컨텍스트: 네트워크 초기화 단계(시뮬레이션 시작 전), 호스트 CPU.
   *
   * 호출 체인:
   *   Network::Network() → NewRouter() → [Router()] (→ IQRouter() 등)
   */
  Router( const Configuration& config,
	  Module *parent, const string & name, int id,
	  int inputs, int outputs );

  /*
   * [한국어]
   * NewRouter - 라우터 타입별 인스턴스 생성 팩토리 함수 (정적)
   *
   * @config  : gpgpusim.config 파싱 객체 — "router" 키로 타입 결정, 생성자에 전달
   * @parent  : 상위 Module 객체
   * @name    : 라우터 이름 문자열
   * @id      : 라우터 고유 번호
   * @inputs  : 입력 포트 수
   * @outputs : 출력 포트 수
   * @return  : 생성된 라우터 포인터 (IQRouter*, EventRouter*, ChaosRouter* 중 하나);
   *            알 수 없는 타입이면 NULL 반환 후 cerr에 에러 메시지 출력
   *
   * gpgpusim.config의 "router" 설정값을 읽어 적합한 구체 라우터 클래스를 동적 할당한다.
   * "iq" → IQRouter (가장 일반적, GPGPU-Sim 기본),
   * "event" → EventRouter,
   * "chaos" → ChaosRouter.
   * 네트워크 객체가 라우터 배열을 초기화할 때 이 함수를 호출한다.
   *
   * 실행 컨텍스트: 시뮬레이션 초기화 단계, 호스트 CPU.
   *
   * 호출 체인:
   *   Network::Network() → [Router::NewRouter()] → IQRouter() / EventRouter() / ChaosRouter()
   */
  static Router *NewRouter( const Configuration& config,
			    Module *parent, const string & name, int id,
			    int inputs, int outputs );

  /*
   * [한국어]
   * AddInputChannel - 입력 flit 채널과 역방향 credit 채널 등록
   *
   * @channel     : 이 라우터 입력 포트로 flit이 들어오는 FlitChannel 포인터
   * @backchannel : 이 라우터가 업스트림으로 credit을 돌려보내는 CreditChannel 포인터
   * @return      : 없음 (void)
   *
   * _input_channels와 _input_credits 배열에 각각 push_back하고,
   * channel->SetSink(this, index)를 호출하여 채널이 이 라우터를 싱크(수신자)로 인식하게 한다.
   * 네트워크 초기화 시 모든 라우터의 입력 포트 수만큼 순서대로 호출된다.
   *
   * 실행 컨텍스트: 시뮬레이션 초기화 단계.
   *
   * 호출 체인:
   *   Network::Network() → [Router::AddInputChannel()]
   */
  virtual void AddInputChannel( FlitChannel *channel, CreditChannel *backchannel );

  /*
   * [한국어]
   * AddOutputChannel - 출력 flit 채널과 역방향 credit 채널 등록
   *
   * @channel     : 이 라우터 출력 포트에서 flit이 나가는 FlitChannel 포인터
   * @backchannel : 다운스트림 라우터가 credit을 보내오는 CreditChannel 포인터
   * @return      : 없음 (void)
   *
   * _output_channels와 _output_credits 배열에 push_back하고,
   * _channel_faults에 false(정상)를 추가한다.
   * channel->SetSource(this, index)를 호출하여 채널이 이 라우터를 소스(송신자)로 인식하게 한다.
   *
   * 실행 컨텍스트: 시뮬레이션 초기화 단계.
   *
   * 호출 체인:
   *   Network::Network() → [Router::AddOutputChannel()]
   */
  virtual void AddOutputChannel( FlitChannel *channel, CreditChannel *backchannel );

  /*
   * [한국어]
   * GetInputChannel - 지정 입력 포트의 FlitChannel 포인터 반환 (인라인)
   *
   * @input  : 조회할 입력 포트 인덱스 (0 ~ _inputs-1)
   * @return : 해당 입력 포트의 FlitChannel 포인터; 범위 초과 시 assert 실패
   *
   * 외부 모듈(network.* 등)이 채널 연결 상태를 조회하거나 디버그할 때 사용한다.
   * 인라인 함수이므로 호출 오버헤드 없음. assert로 범위 검증을 수행한다.
   *
   * 호출 체인:
   *   Network::* 또는 외부 조회 → [GetInputChannel()]
   */
  inline FlitChannel * GetInputChannel( int input ) const {
    assert((input >= 0) && (input < _inputs)); // [한국어] 입력 포트 인덱스가 유효 범위[0, _inputs) 내에 있는지 검증 — 범위 초과는 잘못된 토폴로지 초기화를 의미
    return _input_channels[input]; // [한국어] 해당 인덱스의 FlitChannel 포인터 반환
  }

  /*
   * [한국어]
   * GetOutputChannel - 지정 출력 포트의 FlitChannel 포인터 반환 (인라인)
   *
   * @output : 조회할 출력 포트 인덱스 (0 ~ _outputs-1)
   * @return : 해당 출력 포트의 FlitChannel 포인터; 범위 초과 시 assert 실패
   *
   * GetInputChannel의 출력 포트 대칭 함수. 네트워크 조회 및 디버그에 사용.
   *
   * 호출 체인:
   *   Network::* 또는 외부 조회 → [GetOutputChannel()]
   */
  inline FlitChannel * GetOutputChannel( int output ) const {
    assert((output >= 0) && (output < _outputs)); // [한국어] 출력 포트 인덱스가 유효 범위[0, _outputs) 내에 있는지 검증
    return _output_channels[output]; // [한국어] 해당 인덱스의 FlitChannel 포인터 반환
  }

  /*
   * [한국어]
   * ReadInputs - 입력 채널에서 flit과 credit을 읽어 내부 버퍼에 저장 (순수 가상)
   *
   * @return: 없음 (void)
   *
   * 매 사이클 TimedModule::Step()에 의해 Evaluate() 전에 호출된다.
   * 구체 라우터(IQRouter 등)가 이 사이클에 도착한 flit을 입력 채널에서 꺼내
   * 내부 VC 버퍼에 저장하고, credit 채널에서 credit을 수신하여 출력 버퍼 카운터를 갱신한다.
   *
   * 실행 컨텍스트: 매 시뮬레이션 사이클, 호스트 CPU 단일 스레드.
   *
   * 호출 체인:
   *   TimedModule::Step() → [ReadInputs()] → Evaluate() → WriteOutputs()
   */
  virtual void ReadInputs( ) = 0;

  /*
   * [한국어]
   * Evaluate - internal_speedup에 따라 내부 스텝을 실행하는 사이클 처리 함수
   *
   * @return: 없음 (void)
   *
   * 분수 사이클(fractional cycle) 어큐뮬레이터 패턴으로 _internal_speedup을
   * _partial_internal_cycles에 더하고, 1.0 이상이 되면 _InternalStep()을
   * 호출한 뒤 1.0을 차감한다. 이를 통해 1.5배 배속이면 평균 1.5회/외부사이클의
   * 내부 처리를 정확하게 구현한다. TimedModule 프레임워크에 의해 Step()이 호출될 때
   * ReadInputs() 다음, WriteOutputs() 이전에 호출된다.
   *
   * 실행 컨텍스트: 매 시뮬레이션 사이클, 호스트 CPU 단일 스레드.
   *
   * 호출 체인:
   *   TimedModule::Step() → ReadInputs() → [Evaluate()] → WriteOutputs()
   *   Evaluate() 내부 → _InternalStep() (1회 이상)
   */
  virtual void Evaluate( );

  /*
   * [한국어]
   * WriteOutputs - 내부 버퍼에서 flit과 credit을 출력 채널에 기록 (순수 가상)
   *
   * @return: 없음 (void)
   *
   * 매 사이클 Evaluate() 이후에 호출된다. 구체 라우터가 스위치 중재 결과로
   * 선택된 flit을 해당 출력 FlitChannel에 기록하고, 해제된 입력 버퍼 슬롯에
   * 대응하는 credit을 _input_credits 채널에 기록한다.
   *
   * 실행 컨텍스트: 매 시뮬레이션 사이클, 호스트 CPU 단일 스레드.
   *
   * 호출 체인:
   *   TimedModule::Step() → ReadInputs() → Evaluate() → [WriteOutputs()]
   */
  virtual void WriteOutputs( ) = 0;

  /*
   * [한국어]
   * OutChannelFault - 지정 출력 채널의 결함 상태를 설정
   *
   * @c    : 결함 상태를 변경할 출력 채널 인덱스 (0 ~ _outputs-1)
   * @fault: true이면 결함 채널로 표시, false이면 정상으로 복원 (기본값 true)
   * @return: 없음 (void)
   *
   * 결함 허용 라우팅 연구에서 특정 출력 링크를 결함 상태로 시뮬레이션할 때 사용한다.
   * _channel_faults[c]를 fault로 설정하며, IsFaultyOutput()으로 상태를 조회할 수 있다.
   * assert로 인덱스 유효성을 검증한다.
   *
   * 실행 컨텍스트: 시뮬레이션 중 임의 시점 (테스트 코드 또는 네트워크 설정 코드에서 호출).
   *
   * 호출 체인:
   *   테스트 코드 / Network::* → [OutChannelFault()]
   */
  void OutChannelFault( int c, bool fault = true );

  /*
   * [한국어]
   * IsFaultyOutput - 지정 출력 채널이 결함 상태인지 조회
   *
   * @c     : 조회할 출력 채널 인덱스 (0 ~ _outputs-1)
   * @return: true이면 결함 채널, false이면 정상 채널; 범위 초과 시 assert 실패
   *
   * IQRouter 등 구체 라우터가 라우팅 결정 시 특정 출력 포트를 사용할 수 있는지
   * 확인하기 위해 호출한다. 결함 채널은 라우팅 테이블에서 제외된다.
   *
   * 호출 체인:
   *   IQRouter::* → [IsFaultyOutput()] (라우팅 결정 단계)
   */
  bool IsFaultyOutput( int c ) const;

  /*
   * [한국어]
   * GetID - 이 라우터의 고유 ID 반환 (인라인)
   *
   * @return: _id 값 (0 이상의 정수)
   *
   * 네트워크 내 라우터 식별, 디버그 출력, 토폴로지 계산에 사용된다.
   * 인라인 함수이므로 호출 오버헤드 없음.
   */
  inline int GetID( ) const {return _id;} // [한국어] _id 필드를 직접 반환 — 생성 후 불변이므로 const 함수

  /*
   * [한국어]
   * GetUsedCredit - 지정 출력 포트에서 현재 사용 중인 credit 수 반환 (순수 가상)
   *
   * @o     : 출력 포트 인덱스
   * @return: 해당 출력 포트에서 현재 사용 중인(아직 회수되지 않은) credit 수
   *
   * 구체 라우터가 자신의 VC 상태를 기반으로 구현한다.
   * 통계 집계 및 네트워크 상태 모니터링에 사용된다.
   */
  virtual int GetUsedCredit(int o) const = 0;

  /*
   * [한국어]
   * GetBufferOccupancy - 지정 입력 포트의 버퍼 점유율 반환 (순수 가상)
   *
   * @i     : 입력 포트 인덱스
   * @return: 해당 입력 포트의 현재 버퍼 점유 슬롯 수
   *
   * 구체 라우터가 입력 버퍼 상태를 기반으로 구현한다.
   * 혼잡도 분석 및 결과 리포트에 사용된다.
   */
  virtual int GetBufferOccupancy(int i) const = 0;

#ifdef TRACK_BUFFERS
  /*
   * [한국어]
   * GetUsedCreditForClass - 클래스별 출력 포트 credit 사용량 반환 (순수 가상, TRACK_BUFFERS 시에만)
   *
   * @output: 출력 포트 인덱스
   * @cl    : 트래픽 클래스 인덱스
   * @return: 해당 출력 포트에서 해당 클래스에 할당된 사용 중 credit 수
   */
  virtual int GetUsedCreditForClass(int output, int cl) const = 0;

  /*
   * [한국어]
   * GetBufferOccupancyForClass - 클래스별 입력 포트 버퍼 점유율 반환 (순수 가상, TRACK_BUFFERS 시에만)
   *
   * @input : 입력 포트 인덱스
   * @cl    : 트래픽 클래스 인덱스
   * @return: 해당 입력 포트에서 해당 클래스가 점유 중인 버퍼 슬롯 수
   */
  virtual int GetBufferOccupancyForClass(int input, int cl) const = 0;
#endif

#ifdef TRACK_FLOWS
  /*
   * [한국어]
   * GetReceivedFlits - 지정 클래스의 입력 포트별 수신 flit 카운터 배열 반환 (인라인)
   *
   * @c     : 트래픽 클래스 인덱스 (0 ~ _classes-1); 범위 초과 시 assert 실패
   * @return: _received_flits[c] 에 대한 const 참조 (크기 _inputs의 int 배열)
   */
  inline vector<int> const & GetReceivedFlits(int c) const {
    assert((c >= 0) && (c < _classes)); // [한국어] 클래스 인덱스 유효성 검증 — 잘못된 클래스 접근은 프로그래밍 오류
    return _received_flits[c]; // [한국어] 해당 클래스의 입력 포트별 수신 flit 카운터 배열 참조 반환
  }

  /*
   * [한국어]
   * GetStoredFlits - 지정 클래스의 현재 버퍼 내 저장 flit 카운터 배열 반환 (인라인)
   *
   * @c     : 트래픽 클래스 인덱스
   * @return: _stored_flits[c]에 대한 const 참조
   */
  inline vector<int> const & GetStoredFlits(int c) const {
    assert((c >= 0) && (c < _classes)); // [한국어] 클래스 인덱스 유효성 검증
    return _stored_flits[c]; // [한국어] 해당 클래스의 저장 flit 카운터 배열 참조 반환
  }

  /*
   * [한국어]
   * GetSentFlits - 지정 클래스의 출력 포트별 전송 flit 카운터 배열 반환 (인라인)
   *
   * @c     : 트래픽 클래스 인덱스
   * @return: _sent_flits[c]에 대한 const 참조 (크기 _outputs의 int 배열)
   */
  inline vector<int> const & GetSentFlits(int c) const {
    assert((c >= 0) && (c < _classes)); // [한국어] 클래스 인덱스 유효성 검증
    return _sent_flits[c]; // [한국어] 해당 클래스의 출력 포트별 전송 flit 카운터 배열 참조 반환
  }

  /*
   * [한국어]
   * GetOutstandingCredits - 지정 클래스의 출력 포트별 미결 credit 카운터 배열 반환 (인라인)
   *
   * @c     : 트래픽 클래스 인덱스
   * @return: _outstanding_credits[c]에 대한 const 참조 (크기 _outputs의 int 배열)
   */
  inline vector<int> const & GetOutstandingCredits(int c) const {
    assert((c >= 0) && (c < _classes)); // [한국어] 클래스 인덱스 유효성 검증
    return _outstanding_credits[c]; // [한국어] 해당 클래스의 미결 credit 카운터 배열 참조 반환
  }

  /*
   * [한국어]
   * GetActivePackets - 지정 클래스의 현재 활성 패킷 카운터 배열 반환 (인라인)
   *
   * @c     : 트래픽 클래스 인덱스
   * @return: _active_packets[c]에 대한 const 참조
   */
  inline vector<int> const & GetActivePackets(int c) const {
    assert((c >= 0) && (c < _classes)); // [한국어] 클래스 인덱스 유효성 검증
    return _active_packets[c]; // [한국어] 해당 클래스의 활성 패킷 카운터 배열 참조 반환
  }

  /*
   * [한국어]
   * ResetFlowStats - 지정 클래스의 흐름 통계 카운터(수신/전송 flit)를 0으로 리셋 (인라인)
   *
   * @c     : 리셋할 트래픽 클래스 인덱스
   * @return: 없음 (void)
   *
   * 워밍업(warm-up) 기간 이후 통계를 초기화하거나, 측정 구간 경계에서 호출된다.
   * _received_flits[c]와 _sent_flits[c]만 리셋하며, _stored_flits, _active_packets,
   * _outstanding_credits는 상태 변수이므로 리셋하지 않는다.
   */
  inline void ResetFlowStats(int c) {
    assert((c >= 0) && (c < _classes)); // [한국어] 클래스 인덱스 유효성 검증
    _received_flits[c].assign(_received_flits[c].size(), 0); // [한국어] _received_flits[c] 배열의 모든 원소를 0으로 리셋 — assign(size, value)는 전체를 value로 채움
    _sent_flits[c].assign(_sent_flits[c].size(), 0); // [한국어] _sent_flits[c] 배열의 모든 원소를 0으로 리셋
  }
#endif

  /*
   * [한국어]
   * UsedCredits - 모든 출력 포트의 사용 중 credit 수 벡터 반환 (순수 가상)
   *
   * @return: 크기 _outputs인 int 벡터, [i]는 i번 출력 포트의 사용 중 credit 수
   *
   * 통계 수집 및 네트워크 상태 스냅샷 생성에 사용된다.
   */
  virtual vector<int> UsedCredits() const = 0;

  /*
   * [한국어]
   * FreeCredits - 모든 출력 포트의 여유 credit 수 벡터 반환 (순수 가상)
   *
   * @return: 크기 _outputs인 int 벡터, [i]는 i번 출력 포트의 여유 credit 수
   */
  virtual vector<int> FreeCredits() const = 0;

  /*
   * [한국어]
   * MaxCredits - 모든 출력 포트의 최대 credit 수 벡터 반환 (순수 가상)
   *
   * @return: 크기 _outputs인 int 벡터, [i]는 i번 출력 포트의 최대 credit 수 (버퍼 용량)
   *
   * UsedCredits + FreeCredits = MaxCredits 관계가 성립해야 한다.
   */
  virtual vector<int> MaxCredits() const = 0;

#ifdef TRACK_STALLS
  /*
   * [한국어]
   * GetBufferBusyStalls - 지정 클래스의 BUFFER_BUSY 스톨 횟수 반환 (인라인)
   *
   * @c     : 트래픽 클래스 인덱스
   * @return: _buffer_busy_stalls[c] 값
   */
  inline int GetBufferBusyStalls(int c) const {
    assert((c >= 0) && (c < _classes)); // [한국어] 클래스 인덱스 유효성 검증
    return _buffer_busy_stalls[c]; // [한국어] 해당 클래스의 BUFFER_BUSY 스톨 누산 카운터 반환
  }

  /*
   * [한국어]
   * GetBufferConflictStalls - 지정 클래스의 BUFFER_CONFLICT 스톨 횟수 반환 (인라인)
   *
   * @c     : 트래픽 클래스 인덱스
   * @return: _buffer_conflict_stalls[c] 값
   */
  inline int GetBufferConflictStalls(int c) const {
    assert((c >= 0) && (c < _classes)); // [한국어] 클래스 인덱스 유효성 검증
    return _buffer_conflict_stalls[c]; // [한국어] 해당 클래스의 BUFFER_CONFLICT 스톨 누산 카운터 반환
  }

  /*
   * [한국어]
   * GetBufferFullStalls - 지정 클래스의 BUFFER_FULL 스톨 횟수 반환 (인라인)
   *
   * @c     : 트래픽 클래스 인덱스
   * @return: _buffer_full_stalls[c] 값
   */
  inline int GetBufferFullStalls(int c) const {
    assert((c >= 0) && (c < _classes)); // [한국어] 클래스 인덱스 유효성 검증
    return _buffer_full_stalls[c]; // [한국어] 해당 클래스의 BUFFER_FULL 스톨 누산 카운터 반환
  }

  /*
   * [한국어]
   * GetBufferReservedStalls - 지정 클래스의 BUFFER_RESERVED 스톨 횟수 반환 (인라인)
   *
   * @c     : 트래픽 클래스 인덱스
   * @return: _buffer_reserved_stalls[c] 값
   */
  inline int GetBufferReservedStalls(int c) const {
    assert((c >= 0) && (c < _classes)); // [한국어] 클래스 인덱스 유효성 검증
    return _buffer_reserved_stalls[c]; // [한국어] 해당 클래스의 BUFFER_RESERVED 스톨 누산 카운터 반환
  }

  /*
   * [한국어]
   * GetCrossbarConflictStalls - 지정 클래스의 CROSSBAR_CONFLICT 스톨 횟수 반환 (인라인)
   *
   * @c     : 트래픽 클래스 인덱스
   * @return: _crossbar_conflict_stalls[c] 값
   */
  inline int GetCrossbarConflictStalls(int c) const {
    assert((c >= 0) && (c < _classes)); // [한국어] 클래스 인덱스 유효성 검증
    return _crossbar_conflict_stalls[c]; // [한국어] 해당 클래스의 CROSSBAR_CONFLICT 스톨 누산 카운터 반환
  }

  /*
   * [한국어]
   * ResetStallStats - 지정 클래스의 모든 스톨 카운터를 0으로 리셋 (인라인)
   *
   * @c     : 리셋할 트래픽 클래스 인덱스
   * @return: 없음 (void)
   *
   * 워밍업 기간 후 또는 측정 구간 경계에서 통계를 초기화할 때 사용한다.
   * BUFFER_BUSY, BUFFER_CONFLICT, BUFFER_FULL, BUFFER_RESERVED, CROSSBAR_CONFLICT
   * 다섯 가지 스톨 카운터를 모두 0으로 설정한다.
   */
  inline void ResetStallStats(int c) {
    assert((c >= 0) && (c < _classes)); // [한국어] 클래스 인덱스 유효성 검증
    _buffer_busy_stalls[c] = 0;      // [한국어] BUFFER_BUSY 스톨 카운터를 0으로 초기화
    _buffer_conflict_stalls[c] = 0;  // [한국어] BUFFER_CONFLICT 스톨 카운터를 0으로 초기화
    _buffer_full_stalls[c] = 0;      // [한국어] BUFFER_FULL 스톨 카운터를 0으로 초기화
    _buffer_reserved_stalls[c] = 0;  // [한국어] BUFFER_RESERVED 스톨 카운터를 0으로 초기화
    _crossbar_conflict_stalls[c] = 0; // [한국어] CROSSBAR_CONFLICT 스톨 카운터를 0으로 초기화
  }
#endif

  /*
   * [한국어]
   * NumInputs - 이 라우터의 물리 입력 포트 수 반환 (인라인)
   *
   * @return: _inputs 값
   */
  inline int NumInputs() const {return _inputs;} // [한국어] _inputs를 직접 반환 — 생성 후 불변이므로 const 함수

  /*
   * [한국어]
   * NumOutputs - 이 라우터의 물리 출력 포트 수 반환 (인라인)
   *
   * @return: _outputs 값
   */
  inline int NumOutputs() const {return _outputs;} // [한국어] _outputs를 직접 반환 — 생성 후 불변이므로 const 함수
};

#endif // [한국어] _ROUTER_HPP_ 인클루드 가드 종료 — 다중 포함(multiple inclusion) 방지
