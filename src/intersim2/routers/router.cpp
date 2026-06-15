// $Id: router.cpp 5188 2012-08-30 00:31:31Z dub $

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

/*router.cpp
 *
 *The base class of either iq router or event router
 *contains a list of channels and other router configuration variables
 *
 *The older version of the simulator uses an array of flits and credit to
 *simulate the channels. Newer version ueses flitchannel and credit channel
 *which can better model channel delay
 *
 *The older version of the simulator also uses vc_router and chaos router
 *which are replaced by iq rotuer and event router in the present form
 */

/*
 * [한국어 설명] NoC 라우터 추상 기반 클래스 구현 (router.cpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 router.hpp에 선언된 `Router` 추상 기반 클래스의 비순수 가상 멤버 함수들과
 * 정적(static) 상수·팩토리 함수를 구현한다. 구체적으로 다음 역할을 담당한다:
 * (1) 스톨 코드 상수(STALL_BUFFER_BUSY 등)의 정의 및 초기값 할당
 * (2) Router 생성자: 설정 파일 파라미터 파싱 및 통계 배열 초기화
 * (3) AddInputChannel / AddOutputChannel: 채널 포인터 등록 및 싱크/소스 바인딩
 * (4) Evaluate: 분수 사이클 어큐뮬레이터 패턴으로 _InternalStep() 호출 횟수 제어
 * (5) OutChannelFault / IsFaultyOutput: 결함 채널 표시·조회
 * (6) NewRouter: "router" 설정값에 따라 IQRouter/EventRouter/ChaosRouter 동적 생성
 * 실질적인 NoC 스위치 중재 로직은 모두 구체 서브클래스(iq_router.cpp 등)에 있으며,
 * 이 파일은 그 공통 기반 인프라만 제공한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 타이밍 모델에서 NoC(intersim2)는 SM과 메모리 파티션 간 통신 지연을
 * 사이클 단위로 시뮬레이션하는 서브시스템이다. 이 파일은 그 NoC 서브시스템의 라우터
 * 계층 구조 최상단에 위치하는 공통 기반 구현이다. 전체 호출 흐름은 다음과 같다:
 *
 *   gpu-sim.cc::cycle()
 *     → icnt_wrapper.cc::icnt_push() / icnt_pop()
 *         → Network::ReadInputs() / Evaluate() / WriteOutputs()
 *             → [Router::ReadInputs()] → [Router::Evaluate()] → [Router::WriteOutputs()]
 *                 → IQRouter::_InternalStep() (스위치 중재·VC 할당·flit 이동)
 *
 * 각 라우터 인스턴스는 시뮬레이션 초기화 시 NewRouter()로 생성되고,
 * 이후 매 사이클 TimedModule::Step()이 ReadInputs → Evaluate → WriteOutputs 순서로 호출한다.
 * 실행 컨텍스트: 호스트 CPU 단일 스레드(GPGPU-Sim 메인 시뮬레이션 루프).
 *
 * === 타 모듈과의 연결 ===
 * 이 파일이 의존하는 모듈:
 *   - booksim.hpp        : BookSim2 공통 매크로 및 타입 정의
 *   - router.hpp         : Router 클래스 선언
 *   - iq_router.hpp      : IQRouter — 가장 일반적인 IQ(Input-Queued) 라우터, GPGPU-Sim 기본
 *   - event_router.hpp   : EventRouter — 이벤트 구동 방식 라우터
 *   - chaos_router.hpp   : ChaosRouter — 혼돈 라우팅 라우터 (레거시)
 *   - config_utils.hpp   : Configuration — gpgpusim.config 파싱 결과 객체
 *   - flitchannel.hpp    : FlitChannel — flit 전송 채널 (지연 모델 포함)
 *   - channel.hpp        : Channel<Credit> = CreditChannel — 역방향 흐름 제어 채널
 *
 * 이 파일에 의존하는 모듈:
 *   - network.cpp        : Network가 NewRouter()를 호출해 라우터 배열 생성
 *   - iq_router.cpp 등   : Router 멤버 변수·함수를 상속하여 사용
 *
 * 데이터 흐름: FlitChannel → Router 입력 버퍼(IQRouter 내부) → 크로스바 → FlitChannel
 * 역방향 흐름 제어: CreditChannel(output_credits) → Router → CreditChannel(input_credits)
 *
 * === 주요 함수/구조체 요약 ===
 * - Router()           : 설정 파라미터(delay, speedup, classes) 파싱 및 통계 배열 초기화
 * - AddInputChannel()  : _input_channels / _input_credits에 채널 추가 + SetSink() 바인딩
 * - AddOutputChannel() : _output_channels / _output_credits에 채널 추가 + SetSource() 바인딩
 * - Evaluate()         : _partial_internal_cycles 어큐뮬레이터로 _InternalStep() 호출 횟수 결정
 * - OutChannelFault()  : 지정 출력 채널을 결함 상태로 표시 (결함 허용 연구용)
 * - IsFaultyOutput()   : 지정 출력 채널의 결함 상태 조회
 * - NewRouter()        : "router" 설정값으로 IQRouter/EventRouter/ChaosRouter 팩토리 생성
 * - STALL_* 상수 정의  : -2~-6 음수 센티널값으로 스톨 이유 인코딩
 */

#include "booksim.hpp"  // [한국어] BookSim2 공통 헤더 — ASSERT 매크로, 공통 타입, 디버그 출력 유틸리티 제공
#include <iostream>     // [한국어] cerr을 통한 에러 메시지 출력에 필요한 표준 입출력 헤더 (NewRouter의 알 수 없는 타입 경고용)
#include <cassert>      // [한국어] assert() 매크로 — 인덱스 범위·포인터 유효성 검증에 사용
#include "router.hpp"   // [한국어] Router 클래스 선언 — 이 파일이 구현하는 멤버 함수들의 선언을 포함

//////////////////Sub router types//////////////////////
// [한국어] 구체 라우터 서브클래스 헤더 포함 — NewRouter() 팩토리 함수에서 각 타입을 동적 생성하기 위해 필요
#include "iq_router.hpp"    // [한국어] IQRouter: 가장 일반적인 IQ(Input-Queued) 라우터 — GPGPU-Sim에서 기본으로 사용
#include "event_router.hpp" // [한국어] EventRouter: 이벤트 구동 방식 라우터 — 특정 트래픽 패턴 연구용
#include "chaos_router.hpp" // [한국어] ChaosRouter: 혼돈 라우팅 방식 라우터 — 레거시 지원용
///////////////////////////////////////////////////////

// ─────────────────────────────────────────────────────────────────────────────
// 스톨 코드 상수 정의 (static const 멤버는 .cpp에서 정의해야 링크 가능)
// ─────────────────────────────────────────────────────────────────────────────
// [한국어] static const 정수 멤버는 헤더의 선언만으로는 ODR(One Definition Rule) 상
// 링커가 심볼을 생성하지 못할 수 있으므로, 반드시 .cpp에서 값을 정의해야 한다.
// 이 상수들은 할당 큐에 저장되는 음수 센티널값으로, 양수 출력 포트 번호와 구별된다.

int const Router::STALL_BUFFER_BUSY      = -2; // [한국어] 버퍼가 이미 다른 VC 할당으로 점유 중인 스톨 — 값 -2
int const Router::STALL_BUFFER_CONFLICT  = -3; // [한국어] 여러 flit이 동일 버퍼 슬롯을 경합하는 스톨 — 값 -3
int const Router::STALL_BUFFER_FULL      = -4; // [한국어] 입력 버퍼가 완전히 포화되어 수용 불가한 스톨 — 값 -4
int const Router::STALL_BUFFER_RESERVED  = -5; // [한국어] VC 예약으로 인해 다른 flit이 슬롯을 사용할 수 없는 스톨 — 값 -5
int const Router::STALL_CROSSBAR_CONFLICT = -6; // [한국어] 크로스바에서 동일 출력 포트 요청 충돌로 인한 스톨 — 값 -6

/*
 * [한국어]
 * Router::Router - NoC 라우터 기반 클래스 생성자
 *
 * @config  : gpgpusim.config 파싱 결과를 보유하는 Configuration 객체 const 참조.
 *            "st_prepare_delay", "st_final_delay", "credit_delay",
 *            "input_speedup", "output_speedup", "internal_speedup", "classes" 키를 읽는다.
 * @parent  : 이 라우터를 포함하는 상위 Module 객체 포인터.
 *            TimedModule 기반 클래스의 계층적 이름 생성에 사용된다.
 * @name    : 이 라우터의 식별 이름 문자열 (예: "router0", "router1").
 * @id      : 네트워크 내 라우터 고유 번호 (0-based 정수).
 * @inputs  : 물리 입력 포트 수; 채널 배열 크기의 기준이 된다.
 * @outputs : 물리 출력 포트 수; 출력 채널 및 통계 배열 크기의 기준이 된다.
 * @return  : 없음 (생성자)
 *
 * 역할 및 동작:
 * 이 생성자는 Router 클래스의 모든 필드를 초기화하는 유일한 진입점이다.
 * TimedModule(parent, name)을 통해 모듈 계층 트리에 등록하고,
 * 멤버 초기화 목록에서 _id, _inputs, _outputs, _partial_internal_cycles(=0.0)을 설정한다.
 * 생성자 본문에서는:
 *   1. 크로스바 지연(_crossbar_delay) = st_prepare_delay + st_final_delay
 *   2. credit 지연(_credit_delay)
 *   3. 입출력 속도 배수(_input_speedup, _output_speedup)
 *   4. 내부 처리 속도 배수(_internal_speedup)
 *   5. 트래픽 클래스 수(_classes)
 * 를 gpgpusim.config에서 읽어 초기화한다.
 * TRACK_FLOWS가 정의되면 수신/저장/전송/활성/미결 credit 통계 배열을 0으로 초기화하고,
 * TRACK_STALLS가 정의되면 5종류의 스톨 카운터를 클래스 수만큼 0으로 초기화한다.
 *
 * 실행 컨텍스트: 시뮬레이션 초기화 단계(시뮬레이션 루프 시작 전), 호스트 CPU.
 *
 * 에러 처리: config 키가 없거나 잘못된 경우 Configuration::GetInt/GetFloat가 assert/예외 발생.
 *
 * 호출 체인:
 *   Network::Network() → Router::NewRouter() → [IQRouter()] → [Router()] (기반 클래스 생성자)
 */
Router::Router( const Configuration& config,
		Module *parent, const string & name, int id,
		int inputs, int outputs ) :
TimedModule( parent, name ),  // [한국어] TimedModule 기반 클래스 초기화 — 모듈 계층 트리에 이 라우터를 등록하고 Step() 프레임워크에 참여
_id( id ),                    // [한국어] 라우터 고유 ID 초기화 — 이후 읽기 전용
_inputs( inputs ),            // [한국어] 물리 입력 포트 수 초기화
_outputs( outputs ),          // [한국어] 물리 출력 포트 수 초기화
_partial_internal_cycles(0.0) // [한국어] 분수 사이클 누산기를 0으로 초기화 — 첫 번째 Evaluate() 호출 전까지 누산량 없음
{
  // [한국어] 크로스바(crossbar) 스위치 통과 총 지연 계산.
  // st_prepare_delay: 스위치 트래버설 준비 단계(SA-I, VA 등) 지연 사이클 수.
  // st_final_delay: 스위치 트래버설 최종 단계(SA-II, ST) 지연 사이클 수.
  // 둘을 합산하여 flit이 입력 버퍼에서 출력 채널에 도달하기까지의 파이프라인 깊이를 모델링한다.
  _crossbar_delay   = ( config.GetInt( "st_prepare_delay" ) +
			config.GetInt( "st_final_delay" ) );

  // [한국어] credit 신호 전파 지연 초기화.
  // credit이 다운스트림 라우터에서 출발하여 이 라우터에 도착하기까지 걸리는 사이클 수.
  // 이 지연이 크면 파이프라인 버블이 증가하여 유효 처리량이 감소한다.
  _credit_delay     = config.GetInt( "credit_delay" );

  // [한국어] 입력 포트 가상 확장 배수 초기화.
  // 물리 입력 포트 하나를 이 배수만큼 가상 서브포트로 확장하여 한 사이클에 더 많은 flit을 처리한다.
  _input_speedup    = config.GetInt( "input_speedup" );

  // [한국어] 출력 포트 가상 확장 배수 초기화.
  // _input_speedup과 대칭적으로 출력 측 크로스바 슬롯 수를 확장한다.
  _output_speedup   = config.GetInt( "output_speedup" );

  // [한국어] 내부 처리 속도 배수 초기화(실수값).
  // Evaluate()의 분수 사이클 어큐뮬레이터에 매 사이클 더해지는 값.
  // 1.0이면 외부 사이클과 동일 속도, 1.5이면 2 외부 사이클마다 3번 _InternalStep() 실행.
  _internal_speedup = config.GetFloat( "internal_speedup" );

  // [한국어] 트래픽 클래스(QoS 클래스) 수 초기화.
  // TRACK_FLOWS / TRACK_STALLS 배열의 첫 번째 차원 크기가 이 값으로 결정된다.
  // GPGPU-Sim 기본 설정에서는 통상 1(단일 클래스)을 사용한다.
  _classes          = config.GetInt( "classes" );

#ifdef TRACK_FLOWS
  // [한국어] TRACK_FLOWS 매크로가 정의된 경우 흐름 추적 통계 배열을 초기화한다.
  // 각 배열의 첫 번째 차원: 트래픽 클래스 수(_classes)
  // 각 배열의 두 번째 차원: 해당 차원에 맞는 포트 수 또는 구체 라우터에서 결정

  // [한국어] 수신 flit 카운터: [클래스][입력포트] — 각 입력 포트로 들어온 flit 수 누산
  _received_flits.resize(_classes, vector<int>(_inputs, 0));

  // [한국어] 저장 flit 카운터: [클래스][내부 위치] — 두 번째 차원 크기는 구체 라우터(IQRouter 등)에서 설정
  _stored_flits.resize(_classes);

  // [한국어] 전송 flit 카운터: [클래스][출력포트] — 각 출력 포트에서 나간 flit 수 누산
  _sent_flits.resize(_classes, vector<int>(_outputs, 0));

  // [한국어] 활성 패킷 카운터: [클래스][내부 위치] — 현재 통과 중인 패킷 수; 두 번째 차원은 구체 라우터에서 설정
  _active_packets.resize(_classes);

  // [한국어] 미결 credit 카운터: [클래스][출력포트] — 아직 반환받지 못한 credit 수 추적
  _outstanding_credits.resize(_classes, vector<int>(_outputs, 0));
#endif

#ifdef TRACK_STALLS
  // [한국어] TRACK_STALLS 매크로가 정의된 경우 스톨 카운터 배열을 초기화한다.
  // 각 배열의 크기는 _classes이며, 트래픽 클래스별 스톨 빈도를 집계한다.
  // 초기값은 모두 0으로 시뮬레이션 시작 시 스톨이 없는 상태를 나타낸다.

  // [한국어] BUFFER_BUSY 스톨 카운터: 버퍼 점유 충돌로 인한 스톨 발생 횟수
  _buffer_busy_stalls.resize(_classes, 0);

  // [한국어] BUFFER_CONFLICT 스톨 카운터: 버퍼 뱅크 경합으로 인한 스톨 발생 횟수
  _buffer_conflict_stalls.resize(_classes, 0);

  // [한국어] BUFFER_FULL 스톨 카운터: 입력 버퍼 포화로 인한 스톨 발생 횟수 — NoC 혼잡도의 핵심 지표
  _buffer_full_stalls.resize(_classes, 0);

  // [한국어] BUFFER_RESERVED 스톨 카운터: VC 예약으로 인해 슬롯 사용이 불가한 스톨 발생 횟수
  _buffer_reserved_stalls.resize(_classes, 0);

  // [한국어] CROSSBAR_CONFLICT 스톨 카운터: 크로스바 출력 포트 경합으로 인한 스톨 발생 횟수
  _crossbar_conflict_stalls.resize(_classes, 0);
#endif

} // [한국어] Router 생성자 종료 — 이후 구체 라우터(IQRouter 등)의 생성자가 내부 버퍼·VC 구조를 추가 초기화

/*
 * [한국어]
 * Router::AddInputChannel - 입력 flit 채널과 역방향 credit 채널을 이 라우터에 등록
 *
 * @channel     : 이 라우터의 입력 포트로 flit이 들어오는 FlitChannel 포인터.
 *                업스트림 라우터 또는 SM(인젝션 포트)에서 연결된 채널이다.
 * @backchannel : 이 라우터가 버퍼 슬롯을 해제할 때 업스트림으로 credit을 보내는 CreditChannel 포인터.
 * @return      : 없음 (void)
 *
 * 역할 및 동작:
 * _input_channels 배열에 channel을 추가하고, _input_credits 배열에 backchannel을 추가한다.
 * 추가 후 channel->SetSink(this, index)를 호출하여 FlitChannel이 이 라우터를
 * 자신의 싱크(데이터 수신자)로 인식하도록 바인딩한다. 이 바인딩을 통해 FlitChannel은
 * flit 전달 완료 시 이 라우터의 어느 입력 포트 번호(index)로 도착했는지 알 수 있다.
 * 네트워크 초기화 단계에서 토폴로지 설정 순서대로 호출되며, 호출 순서가 포트 인덱스가 된다.
 *
 * 에러 처리: FlitChannel이 null이면 SetSink()에서 assert 실패.
 *
 * 실행 컨텍스트: 시뮬레이션 초기화 단계, 호스트 CPU.
 *
 * 호출 체인:
 *   Network::Network() → [Router::AddInputChannel()]
 *   → FlitChannel::SetSink(this, _input_channels.size()-1)
 */
void Router::AddInputChannel( FlitChannel *channel, CreditChannel *backchannel )
{
  _input_channels.push_back( channel );    // [한국어] flit 수신 채널 포인터를 입력 채널 배열 끝에 추가 — 인덱스 = 현재 크기-1 = 새 포트 번호
  _input_credits.push_back( backchannel ); // [한국어] 역방향 credit 채널 포인터를 credit 배열 끝에 추가 — _input_channels와 동일 인덱스로 대응

  // [한국어] FlitChannel에게 이 라우터가 자신의 싱크(데이터 수신 끝점)임을 알리고,
  // 몇 번 입력 포트(index = 현재 총 입력 채널 수 - 1)로 연결됐는지 바인딩한다.
  // 이를 통해 FlitChannel은 flit 전달 시 포트 번호를 함께 제공할 수 있다.
  channel->SetSink( this, _input_channels.size() - 1 ) ;
}

/*
 * [한국어]
 * Router::AddOutputChannel - 출력 flit 채널과 역방향 credit 채널을 이 라우터에 등록
 *
 * @channel     : 이 라우터의 출력 포트에서 flit이 나가는 FlitChannel 포인터.
 *                다운스트림 라우터 또는 메모리 파티션(이젝션 포트)으로 연결된 채널.
 * @backchannel : 다운스트림 라우터가 버퍼를 해제할 때 credit을 보내오는 CreditChannel 포인터.
 * @return      : 없음 (void)
 *
 * 역할 및 동작:
 * _output_channels와 _output_credits 배열에 각각 push_back하고,
 * _channel_faults에 false(정상 상태)를 추가하여 새 출력 채널이 기본적으로 정상임을 표시한다.
 * 이후 channel->SetSource(this, index)를 호출하여 FlitChannel이 이 라우터를
 * 자신의 소스(데이터 송신자)로 인식하도록 바인딩한다. 이 바인딩을 통해 FlitChannel은
 * WriteOutputs 단계에서 이 라우터의 어느 출력 포트(index)에서 flit이 왔는지 알 수 있다.
 *
 * 에러 처리: FlitChannel이 null이면 SetSource()에서 assert 실패.
 *
 * 실행 컨텍스트: 시뮬레이션 초기화 단계, 호스트 CPU.
 *
 * 호출 체인:
 *   Network::Network() → [Router::AddOutputChannel()]
 *   → FlitChannel::SetSource(this, _output_channels.size()-1)
 */
void Router::AddOutputChannel( FlitChannel *channel, CreditChannel *backchannel )
{
  _output_channels.push_back( channel );    // [한국어] flit 송신 채널 포인터를 출력 채널 배열 끝에 추가 — 인덱스 = 새 출력 포트 번호
  _output_credits.push_back( backchannel ); // [한국어] 다운스트림에서 오는 credit 채널 포인터를 배열 끝에 추가 — _output_channels와 동일 인덱스로 대응

  // [한국어] 새 출력 채널의 결함 상태를 false(정상)로 초기화한다.
  // _channel_faults 배열은 _output_channels와 동일 인덱스로 대응하므로
  // 이 push_back이 반드시 _output_channels.push_back 이후에 와야 크기가 일치한다.
  _channel_faults.push_back( false );

  // [한국어] FlitChannel에게 이 라우터가 자신의 소스(데이터 송신 끝점)임을 알리고,
  // 몇 번 출력 포트(index = 현재 총 출력 채널 수 - 1)로 연결됐는지 바인딩한다.
  channel->SetSource( this, _output_channels.size() - 1 ) ;
}

/*
 * [한국어]
 * Router::Evaluate - internal_speedup 어큐뮬레이터 패턴으로 내부 스텝 실행
 *
 * @return: 없음 (void)
 *
 * 역할 및 동작:
 * 이 함수는 분수(fractional) 배속을 지원하는 내부 사이클 처리 함수이다.
 * _internal_speedup(예: 1.5)을 _partial_internal_cycles에 더하고,
 * 누산값이 1.0 이상인 동안 _InternalStep()을 호출하고 1.0을 차감한다.
 * 이 패턴(fractional cycle accumulator)의 동작 예:
 *   _internal_speedup = 1.5이면:
 *     - 외부 사이클 1: 0.0 + 1.5 = 1.5 → _InternalStep() 1회, 잔류 0.5
 *     - 외부 사이클 2: 0.5 + 1.5 = 2.0 → _InternalStep() 2회, 잔류 0.0
 *     - 외부 사이클 3: 0.0 + 1.5 = 1.5 → _InternalStep() 1회, 잔류 0.5
 *   즉, 평균 1.5회/외부 사이클의 내부 처리를 정확하게 구현한다.
 *
 * TimedModule::Step()이 ReadInputs() 다음에 이 함수를 호출하며,
 * 이 함수 완료 후 WriteOutputs()가 호출된다.
 *
 * 실행 컨텍스트: 매 시뮬레이션 사이클, 호스트 CPU 단일 스레드.
 *
 * 호출 체인:
 *   TimedModule::Step() → ReadInputs() → [Router::Evaluate()]
 *     → _InternalStep() (1회 이상, 구체 라우터가 구현)
 *   → WriteOutputs()
 */
void Router::Evaluate( )
{
  // [한국어] 분수 사이클 누산기에 이번 외부 사이클의 내부 속도 배수를 더한다.
  // _internal_speedup이 1.0이면 매 사이클 정확히 1번 _InternalStep()을 실행하게 된다.
  // 1.5이면 첫 번째 사이클에는 1번, 두 번째 사이클에는 2번 실행하는 방식으로 평균이 맞춰진다.
  _partial_internal_cycles += _internal_speedup;

  // [한국어] 누산값이 1.0 이상인 동안 내부 스텝을 반복 실행한다.
  // 루프가 끝나면 _partial_internal_cycles는 항상 [0.0, _internal_speedup) 범위 내의 잔류값이 된다.
  while( _partial_internal_cycles >= 1.0 ) {
    _InternalStep( ); // [한국어] 구체 라우터(IQRouter 등)가 구현한 1 내부 사이클 처리 실행 — VC 할당, 스위치 중재, flit 이동 수행
    _partial_internal_cycles -= 1.0; // [한국어] 처리한 1 내부 사이클만큼 누산기에서 차감 — 다음 이터레이션에서 잔류분이 그대로 유지됨
  }
}

/*
 * [한국어]
 * Router::OutChannelFault - 지정 출력 채널의 결함 상태를 설정
 *
 * @c    : 결함 상태를 변경할 출력 채널 인덱스 (0 ~ _outputs-1).
 *         _output_channels 배열의 인덱스와 동일하게 대응한다.
 * @fault: true이면 해당 출력 채널을 결함 상태로 표시(라우팅에서 제외),
 *         false이면 정상 상태로 복원. 기본값은 true.
 * @return: 없음 (void)
 *
 * 역할 및 동작:
 * 결함 허용(fault-tolerant) 라우팅 연구를 위해 특정 출력 링크를 결함 상태로
 * 시뮬레이션할 때 사용한다. _channel_faults[c]를 fault 값으로 설정한다.
 * 결함으로 표시된 채널은 IsFaultyOutput()이 true를 반환하여 IQRouter 등의
 * 라우팅 결정 시 해당 출력 포트를 제외하게 된다.
 * assert를 통해 인덱스 c가 유효 범위[0, _channel_faults.size()) 내에 있는지 검증한다.
 *
 * 에러 처리: c가 음수이거나 _channel_faults.size() 이상이면 assert 실패.
 *
 * 실행 컨텍스트: 시뮬레이션 중 임의 시점 (테스트 코드 또는 장애 주입 코드에서 호출).
 *
 * 호출 체인:
 *   테스트/장애 주입 코드 → [Router::OutChannelFault()]
 */
void Router::OutChannelFault( int c, bool fault )
{
  // [한국어] 채널 인덱스 c가 유효한 범위 내에 있는지 검증한다.
  // c < 0이거나 _channel_faults.size() 이상이면 잘못된 인덱스이므로 assert 실패.
  // size_t 캐스트는 부호 없는 비교 경고를 방지하기 위함이다.
  assert( ( c >= 0 ) && ( (size_t)c < _channel_faults.size( ) ) );

  // [한국어] 해당 출력 채널의 결함 상태를 fault 파라미터 값으로 설정한다.
  // true로 설정하면 IsFaultyOutput()이 true를 반환하여 해당 포트는 라우팅에서 제외된다.
  // false로 설정하면 채널이 복구된 것으로 간주하여 다시 라우팅 대상에 포함된다.
  _channel_faults[c] = fault;
}

/*
 * [한국어]
 * Router::IsFaultyOutput - 지정 출력 채널이 결함 상태인지 조회
 *
 * @c     : 조회할 출력 채널 인덱스 (0 ~ _outputs-1).
 * @return: true이면 결함 채널(라우팅 불가), false이면 정상 채널(라우팅 가능).
 *          인덱스가 유효 범위를 벗어나면 assert 실패.
 *
 * 역할 및 동작:
 * _channel_faults[c] 값을 그대로 반환하는 단순 조회 함수이다.
 * IQRouter 등의 구체 라우터가 라우팅 경로 결정 시 각 출력 포트의 사용 가능 여부를
 * 확인하기 위해 이 함수를 호출한다. 결함 채널이 반환하는 true값에 따라 라우터는
 * 해당 포트를 우회하는 대체 경로를 선택한다(결함 허용 라우팅 알고리즘에서 활용).
 *
 * 실행 컨텍스트: 매 시뮬레이션 사이클, IQRouter의 라우팅 결정 단계에서 호출.
 *
 * 호출 체인:
 *   IQRouter::* (라우팅 결정 단계) → [Router::IsFaultyOutput()]
 */
bool Router::IsFaultyOutput( int c ) const
{
  // [한국어] 채널 인덱스 c가 유효한 범위 내에 있는지 검증한다.
  assert( ( c >= 0 ) && ( (size_t)c < _channel_faults.size( ) ) );

  // [한국어] 해당 인덱스의 결함 상태 플래그를 반환한다.
  // true → 결함 채널(이 출력 포트로 flit 전송 불가)
  // false → 정상 채널(이 출력 포트 사용 가능)
  return _channel_faults[c];
}

/*Router constructor*/
/*
 * [한국어]
 * Router::NewRouter - 설정 파일에 따른 구체 라우터 인스턴스 생성 팩토리 함수 (정적)
 *
 * @config  : gpgpusim.config 파싱 결과 객체 — "router" 키로 라우터 타입을 결정하고,
 *            생성되는 라우터의 생성자에도 그대로 전달되어 파라미터 초기화에 사용된다.
 * @parent  : 상위 Module 객체 포인터 — 모듈 계층 트리 등록에 사용.
 * @name    : 라우터 이름 문자열 (예: "router0").
 * @id      : 네트워크 내 라우터 고유 번호 (0-based).
 * @inputs  : 물리 입력 포트 수.
 * @outputs : 물리 출력 포트 수.
 * @return  : 생성된 Router* 포인터 (IQRouter*, EventRouter*, ChaosRouter* 중 하나).
 *            알 수 없는 "router" 값이면 NULL을 반환하고 cerr에 에러 메시지를 출력한다.
 *
 * 역할 및 동작:
 * 이 팩토리 함수는 gpgpusim.config의 "router" 설정 문자열("iq", "event", "chaos")을
 * 읽어 해당 구체 라우터 클래스를 new로 동적 생성한다. 반환된 포인터는 Network가
 * Router* 배열에 저장하며, 이후 시뮬레이션 사이클마다 TimedModule::Step()으로 구동된다.
 * 생성된 객체의 소유권(메모리 해제 책임)은 Network에 있다.
 *
 * 지원 타입:
 *   "iq"    → IQRouter: 가상 채널(VC) 기반 IQ(Input-Queued) 라우터, GPGPU-Sim 기본값
 *   "event" → EventRouter: 이벤트 구동 방식 라우터, 특정 트래픽 패턴 연구용
 *   "chaos" → ChaosRouter: 혼돈 라우팅 방식, 레거시 지원
 *   기타    → NULL 반환 + cerr 에러 메시지, 시뮬레이터 비정상 종료 가능성 있음
 *
 * 에러 처리: 알 수 없는 타입은 NULL을 반환하므로 호출자(Network)에서 NULL 체크 필요.
 *            이후 NULL 포인터로 Step()을 호출하면 segfault 발생.
 *
 * 실행 컨텍스트: 시뮬레이션 초기화 단계, 호스트 CPU.
 *
 * 호출 체인:
 *   Network::Network() → [Router::NewRouter()]
 *     → new IQRouter(config, parent, name, id, inputs, outputs)
 *     또는 → new EventRouter(...) / new ChaosRouter(...)
 */
Router *Router::NewRouter( const Configuration& config,
			   Module *parent, const string & name, int id,
			   int inputs, int outputs )
{
  // [한국어] gpgpusim.config에서 "router" 키의 문자열 값을 읽는다.
  // 이 값이 생성할 라우터 타입을 결정한다 ("iq", "event", "chaos" 중 하나가 설정됨).
  const string type = config.GetStr( "router" );

  // [한국어] 결과 포인터를 NULL로 초기화한다.
  // 알 수 없는 타입이거나 분기가 하나도 맞지 않으면 NULL이 반환된다.
  Router *r = NULL;

  if ( type == "iq" ) {
    // [한국어] "iq": IQ(Input-Queued) 라우터 생성.
    // 가장 일반적인 NoC 라우터 아키텍처로 VC(가상 채널) 흐름 제어, SA(스위치 중재),
    // LRC(Look-ahead Routing Computation) 등을 지원한다.
    // GPGPU-Sim에서 기본으로 사용하는 라우터 타입이다.
    r = new IQRouter( config, parent, name, id, inputs, outputs );
  } else if ( type == "event" ) {
    // [한국어] "event": 이벤트 구동 방식 라우터 생성.
    // 전통적인 사이클-레벨 시뮬레이션 대신 이벤트 발생 시에만 처리를 수행하여
    // 특정 트래픽 패턴에서 시뮬레이션 속도를 높일 수 있다.
    r = new EventRouter( config, parent, name, id, inputs, outputs );
  } else if ( type == "chaos" ) {
    // [한국어] "chaos": 혼돈 라우팅(Chaos Routing) 방식 라우터 생성.
    // 데드락 회피를 위해 가상 채널 대신 특수 라우팅 알고리즘을 사용하는 레거시 방식.
    // 현재는 IQRouter로 대체되었으나 이전 연구 재현을 위해 유지된다.
    r = new ChaosRouter( config, parent, name, id, inputs, outputs );
  } else {
    // [한국어] 알 수 없는 라우터 타입 — 설정 파일 오류를 표준 에러로 출력한다.
    // r은 NULL로 유지되므로 호출자에서 NULL을 역참조하면 segfault가 발생할 수 있다.
    cerr << "Unknown router type: " << type << endl;
  }

  /*For additional router, add another else if statement*/
  /*Original booksim specifies the router using "flow_control"
   *we now simply call these types.
   */
  // [한국어] 추가 라우터 타입을 지원하려면 위 else if 체인에 분기를 추가하면 된다.
  // 원래 BookSim은 "flow_control" 키로 라우터를 지정했으나,
  // 현재 버전에서는 "router" 키로 타입 이름을 직접 지정하도록 단순화되었다.

  return r; // [한국어] 생성된 구체 라우터 포인터(또는 NULL)를 반환 — 호출자(Network)가 배열에 저장하여 관리
}
