// $Id: iq_router.cpp 5263 2012-09-20 23:40:33Z dub $

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
 * [한국어 설명] IQ(Input-Queued) 라우터 구현 (iq_router.cpp)
 *
 * === 파일의 역할 ===
 * BookSim2 기반 NoC(Network-on-Chip) 시뮬레이터의 핵심 라우팅 엔진인
 * IQRouter 클래스의 전체 구현을 담고 있다. 입력 포트마다 VC(Virtual Channel)
 * FIFO 버퍼를 두고, flit(플릿)을 5단계 파이프라인(RC→VA→SA→ST→LT)으로
 * 처리한다. 각 단계는 Evaluate(할당/스케줄 시도)와 Update(상태 확정)로
 * 나뉘며, 매 사이클 _InternalStep()이 이 순서를 한 바퀴 돌린다.
 * 투기적(speculative) 스위치 할당, 스위치 홀드(switch hold),
 * NOQ(Next-Output Queuing) 룩어헤드 라우팅, piggyback VC 할당 등
 * 다양한 고급 NoC 기법을 지원한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 NoC/ICNT 계층에서 실제 패킷 라우팅을 담당하는 핵심 모듈이다.
 * 호출 체인:
 *   gpgpu_sim::cycle() [gpu-sim.cc]
 *     → icnt_push() / icnt_pop() [icnt_wrapper.cc]
 *       → Interconnect::Advance()
 *         → Network::_Step()
 *           → IQRouter::ReadInputs() → IQRouter::_InternalStep() → IQRouter::WriteOutputs()
 * 실행 컨텍스트: 호스트 CPU의 단일 시뮬레이션 스레드. GPU 커널/디바이스 코드가 아니라
 * 순수 C++ 시뮬레이션 코드이며, 매 코어 클럭 사이클에 한 번씩 호출된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - iq_router.hpp: IQRouter 클래스 선언(필드, 파이프라인 deque, 정책 플래그)
 *   - buffer.hpp / buffer_state.hpp: 입력 VC FIFO, 다운스트림 버퍼 크레딧 추적
 *   - vc.hpp: VC 상태 머신 (idle / routing / vc_alloc / active)
 *   - allocator.hpp: VC/스위치 할당자 (iSLIP, round-robin 등)
 *   - routefunc.hpp: 라우팅 함수 포인터 타입(_rf) 및 gRoutingFunctionMap
 *   - outputset.hpp: RC 결과(출력 포트+VC 범위+우선순위) 집합
 *   - switch_monitor.hpp / buffer_monitor.hpp: NoC 전력 모델링용 통계 수집
 * 이 모듈에 의존하는 모듈:
 *   - intersim2/networks/*: IQRouter 인스턴스를 생성해 Network::_Step()에서 구동
 *   - icnt_wrapper.cc: GPGPU-Sim과의 ICNT 연동 인터페이스
 * 데이터 흐름:
 *   입력 flit → _in_queue_flits → _buf[input][vc]
 *     → RC(_route_vcs) → VA(_vc_alloc_vcs) → SA(_sw_alloc_vcs)
 *     → ST(_crossbar_flits) → _output_buffer[output] → _output_channels[output]
 *   크레딧: _output_credits[output] → _proc_credits → _next_buf[output].ProcessCredit()
 *          _out_queue_credits → _credit_buffer[input] → _input_credits[input]
 *
 * === 주요 함수/구조체 요약 ===
 * IQRouter()              - 생성자: VC 정책, 라우팅 함수, 버퍼/할당자, 홀드 상태,
 *                           NOQ 테이블, 모니터 객체 초기화
 * ~IQRouter()             - 소멸자: 동적 할당 객체 해제 및 모니터 출력
 * AddOutputChannel()      - 출력 채널 등록 시 MinLatency(크레딧 회전 최소 지연) 계산
 * ReadInputs()            - 입력 채널/크레딧 수신, _active 플래그 갱신
 * _InternalStep()         - 매 사이클 5단계 파이프라인 Evaluate/Update 시퀀스 구동
 * WriteOutputs()          - 출력 flit/크레딧 전송
 * _InputQueuing()         - 수신 flit을 VC 버퍼에 넣고 파이프라인 단계에 스케줄
 * _RouteEvaluate/Update   - RC: 라우팅 함수 호출, VC 상태 routing→vc_alloc 전환
 * _VCAllocEvaluate/Update - VA: 출력 VC 할당, TakeBuffer, VC 상태 active 전환
 * _SWHoldEvaluate/Update  - 스위치 홀드: 이미 예약된 크로스바 경로 재사용
 * _SWAllocAddReq()        - SA 요청 등록 (RR 우선순위 기반 교체/중복 방지)
 * _SWAllocEvaluate/Update - SA: 크로스바 연결 할당, 홀드 설정, 실패 시 재시도
 * _SwitchEvaluate/Update  - ST: 크로스바 횡단 후 출력 버퍼로 이동
 * _OutputQueuing()        - 업스트림으로 복귀할 크레딧을 _credit_buffer로 이동
 * _SendFlits/Credits()    - LT: flit/크레딧을 물리 채널로 송신
 * _UpdateNOQ()            - NOQ 룩어헤드: 다음 홉 라우팅 결과를 미리 계산해 저장
 *
 * === 관련 설정 옵션 (gpgpusim.config / intersim2 config) ===
 * - num_vcs: VC 개수 (_vcs)
 * - vc_allocator / sw_allocator / spec_sw_allocator: 할당자 종류
 * - vc_busy_when_full / vc_prioritize_empty / vc_shuffle_requests: VA 정책
 * - speculative / spec_check_elig / spec_check_cred / spec_mask_by_reqs: 투기적 SA
 * - routing_delay / vc_alloc_delay / sw_alloc_delay: 파이프라인 단계 지연
 * - hold_switch_for_packet: 패킷 단위 스위치 홀드
 * - noq: Next-Output Queuing(룩어헤드) 라우팅 활성화
 * - output_buffer_size: ST 이후 출력 버퍼 크기
 * - routing_function / topology: 라우팅 함수 선택 (_rf 결정)
 */

#include "iq_router.hpp"

#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <limits>

#include "globals.hpp"
#include "random_utils.hpp"
#include "vc.hpp"
#include "routefunc.hpp"
#include "outputset.hpp"
#include "buffer.hpp"
#include "buffer_state.hpp"
#include "roundrobin_arb.hpp"
#include "allocator.hpp"
#include "switch_monitor.hpp"
#include "buffer_monitor.hpp"

/*
 * [한국어]
 * IQRouter 생성자 — 입력 큐(Input-Queued) 라우터의 모든 낮은 수준 자료구조를 초기화한다.
 *
 * 초기화 항목:
 *   1) VC/투기적 SA 정책 파라미터를 config에서 읽음
 *   2) 라우팅 함수 포인터(_rf)를 gRoutingFunctionMap에서 검색
 *   3) 입력 포트별 Buffer(_buf)와 출력 포트별 BufferState(_next_buf) 생성
 *   4) VC 할당자(_vc_allocator), 스위치 할당자(_sw_allocator),
 *      투기적 스위치 할당자(_spec_sw_allocator) 생성
 *   5) NOQ 테이블, 출력/크레딧 버퍼, 스위치 홀드 상태 초기화
 *   6) 전력 모델링용 BufferMonitor, SwitchMonitor 생성
 *
 * 중요 조건:
 *   - vc_alloc_delay, sw_alloc_delay는 0이 될 수 없다.
 *   - piggyback VC allocator는 speculative=true일 때만 사용 가능하다.
 *   - noq=true이면 routing_delay==0이고 num_vcs >= outputs여야 한다.
 */
IQRouter::IQRouter( Configuration const & config, Module *parent, 
		    string const & name, int id, int inputs, int outputs )
: Router( config, parent, name, id, inputs, outputs ), _active(false)
{
  _vcs         = config.GetInt( "num_vcs" );

  _vc_busy_when_full = (config.GetInt("vc_busy_when_full") > 0);
  _vc_prioritize_empty = (config.GetInt("vc_prioritize_empty") > 0);
  _vc_shuffle_requests = (config.GetInt("vc_shuffle_requests") > 0);

  _speculative = (config.GetInt("speculative") > 0);
  _spec_check_elig = (config.GetInt("spec_check_elig") > 0);
  _spec_check_cred = (config.GetInt("spec_check_cred") > 0);
  _spec_mask_by_reqs = (config.GetInt("spec_mask_by_reqs") > 0);

  _routing_delay    = config.GetInt( "routing_delay" );
  _vc_alloc_delay   = config.GetInt( "vc_alloc_delay" );
  if(!_vc_alloc_delay) {
    Error("VC allocator cannot have zero delay.");
  }
  _sw_alloc_delay   = config.GetInt( "sw_alloc_delay" );
  if(!_sw_alloc_delay) {
    Error("Switch allocator cannot have zero delay.");
  }

  // Routing
  // [한국어] 라우팅 함수 이름은 "routing_function_topoloy" 형식으로 조합된다.
  // 예: "dim_order_mesh", "xy_torus" 등. gRoutingFunctionMap에서 검색.
  string const rf = config.GetStr("routing_function") + "_" + config.GetStr("topology");
  map<string, tRoutingFunction>::const_iterator rf_iter = gRoutingFunctionMap.find(rf);
  if(rf_iter == gRoutingFunctionMap.end()) {
    Error("Invalid routing function: " + rf);
  }
  _rf = rf_iter->second;

  // Alloc VC's
  // [한국어] 각 입력 포트마다 _outputs개 출력 포트를 후보로 하는 VC FIFO 버퍼 생성.
  _buf.resize(_inputs);
  for ( int i = 0; i < _inputs; ++i ) {
    ostringstream module_name;
    module_name << "buf_" << i;
    _buf[i] = new Buffer(config, _outputs, this, module_name.str( ) );
    module_name.str("");
  }

  // Alloc next VCs' buffer state
  // [한국어] 각 출력 포트마다 다운스트림 라우터의 VC 크레딧/점유 상태를 추적하는
  // BufferState 객체 생성. 흐름 제어(크레딧 기반)의 핵심 상태.
  _next_buf.resize(_outputs);
  for (int j = 0; j < _outputs; ++j) {
    ostringstream module_name;
    module_name << "next_vc_o" << j;
    _next_buf[j] = new BufferState( config, this, module_name.str( ) );
    module_name.str("");
  }

  // Alloc allocators
  // [한국어] VC 할당자 생성. "piggyback"이면 VC 할당을 SA 단계에 끼워 넣어
  // 별도 VC allocator 없이 처리하며, _vc_rr_offset 배열을 라운드로빈용으로 초기화.
  string vc_alloc_type = config.GetStr( "vc_allocator" );
  if(vc_alloc_type == "piggyback") {
    if(!_speculative) {
      Error("Piggyback VC allocation requires speculative switch allocation to be enabled.");
    }
    _vc_allocator = NULL;
    _vc_rr_offset.resize(_outputs*_classes, -1);
  } else {
    _vc_allocator = Allocator::NewAllocator( this, "vc_allocator", 
					     vc_alloc_type,
					     _vcs*_inputs, 
					     _vcs*_outputs );

    if ( !_vc_allocator ) {
      Error("Unknown vc_allocator type: " + vc_alloc_type);
    }
  }
  
  // [한국어] 스위치(크로스바) 할당자 생성. 입력/출력에 speedup을 곱한 차원.
  string sw_alloc_type = config.GetStr( "sw_allocator" );
  _sw_allocator = Allocator::NewAllocator( this, "sw_allocator",
					   sw_alloc_type,
					   _inputs*_input_speedup, 
					   _outputs*_output_speedup );

  if ( !_sw_allocator ) {
    Error("Unknown sw_allocator type: " + sw_alloc_type);
  }
  
  // [한국어] 투기적 SA 전용 할당자. speculative=true이고 "prio"가 아닐 때만 생성.
  // NULL이면 _sw_allocator에서 낮은 우선순위로 투기적 요청을 처리한다.
  string spec_sw_alloc_type = config.GetStr( "spec_sw_allocator" );
  if ( _speculative && ( spec_sw_alloc_type != "prio" ) ) {
    _spec_sw_allocator = Allocator::NewAllocator( this, "spec_sw_allocator",
						  spec_sw_alloc_type,
						  _inputs*_input_speedup, 
						  _outputs*_output_speedup );
    if ( !_spec_sw_allocator ) {
      Error("Unknown spec_sw_allocator type: " + spec_sw_alloc_type);
    }
  } else {
    _spec_sw_allocator = NULL;
  }

  // [한국어] 스위치 할당 라운드로빈 오프셋 초기화. vc % _input_speedup를 시작점으로
  // 하여 speedup이 1보다 클 때 각 expanded_input이 서로 다른 위상에서 출발.
  _sw_rr_offset.resize(_inputs*_input_speedup);
  for(int i = 0; i < _inputs*_input_speedup; ++i)
    _sw_rr_offset[i] = i % _input_speedup;
  
  // NOQ 초기화
  // [한국어] NOQ 모드는 lookahead 라우팅(routing_delay==0)을 필요로 하며,
  // 출력 포트 수 이상의 VC가 필요하다. 조건 불만족 시 Error.
  _noq = config.GetInt("noq") > 0;
  if(_noq) {
    if(_routing_delay) {
      Error("NOQ requires lookahead routing to be enabled.");
    }
    if(_vcs < _outputs) {
      Error("NOQ requires at least as many VCs as router outputs.");
    }
  }
  _noq_next_output_port.resize(_inputs, vector<int>(_vcs, -1));
  _noq_next_vc_start.resize(_inputs, vector<int>(_vcs, -1));
  _noq_next_vc_end.resize(_inputs, vector<int>(_vcs, -1));

  // Output queues
  // [한국어] ST 이후 출력 채널로 나가기 전 임시 대기 버퍼(_output_buffer)와
  // 업스트림으로 돌려볂 크레딧 대기 버퍼(_credit_buffer) 크기 설정.
  _output_buffer_size = config.GetInt("output_buffer_size");
  _output_buffer.resize(_outputs); 
  _credit_buffer.resize(_inputs); 

  // Switch configuration (when held for multiple cycles)
  // [한국어] 패킷 단위 스위치 홀드 정책. head flit이 SA를 통과하면 tail flit까지
  // 동일 크로스바 경로를 예약하여 매 사이클 SA 비용을 줄인다.
  _hold_switch_for_packet = (config.GetInt("hold_switch_for_packet") > 0);
  _switch_hold_in.resize(_inputs*_input_speedup, -1);
  _switch_hold_out.resize(_outputs*_output_speedup, -1);
  _switch_hold_vc.resize(_inputs*_input_speedup, -1);

  // [한국어] 전력 모델링용 모니터 객체 생성. 버퍼 읽기/쓰기 및 크로스바 횡단 이벤트를
  // 기록하여 NoC 전력 추정 시 사용된다.
  _bufferMonitor = new BufferMonitor(inputs, _classes);
  _switchMonitor = new SwitchMonitor(inputs, outputs, _classes);

#ifdef TRACK_FLOWS
  for(int c = 0; c < _classes; ++c) {
    _stored_flits[c].resize(_inputs, 0);
    _active_packets[c].resize(_inputs, 0);
  }
  _outstanding_classes.resize(_outputs, vector<queue<int> >(_vcs));
#endif
}

/*
 * [한국어]
 * IQRouter 소멸자 — 동적 할당된 Buffer, BufferState, Allocator, 모니터 객체를 해제한다.
 * gPrintActivity가 true이면 시뮬레이션 종료 시 버퍼/스위치 통계를 콘솔에 출력한다.
 */
IQRouter::~IQRouter( )
{

  if(gPrintActivity) {
    cout << Name() << ".bufferMonitor:" << endl ; 
    cout << *_bufferMonitor << endl ;
    
    cout << Name() << ".switchMonitor:" << endl ; 
    cout << "Inputs=" << _inputs ;
    cout << "Outputs=" << _outputs ;
    cout << *_switchMonitor << endl ;
  }

  for(int i = 0; i < _inputs; ++i)
    delete _buf[i];
  
  for(int j = 0; j < _outputs; ++j)
    delete _next_buf[j];

  delete _vc_allocator;
  delete _sw_allocator;
  if(_spec_sw_allocator)
    delete _spec_sw_allocator;

  delete _bufferMonitor;
  delete _switchMonitor;
}

/*
 * [한국어]
 * AddOutputChannel — 출력 flit 채널과 역방향 크레딧 채널을 라우터에 등록한다.
 *
 * 각 출력 포트의 _next_buf에 MinLatency를 설정하는데, 이는 다운스트림 버퍼로부터
 * 크레딧이 다시 돌아오기까지의 최소 사이클 수이다. 이 값보다 빨리 크레딧을
 * 발행하면 흐름 제어가 깨질 수 있으므로, BufferState::SetMinLatency()로 하한선을 준다.
 *
 * MinLatency = 1(파이프 오버헤드) + _crossbar_delay + channel latency +
 *              _routing_delay + alloc_delay + backchannel latency + _credit_delay.
 * speculative=true이면 alloc_delay = max(vc_alloc_delay, sw_alloc_delay),
 * 아니면 alloc_delay = vc_alloc_delay + sw_alloc_delay.
 */
void IQRouter::AddOutputChannel(FlitChannel * channel, CreditChannel * backchannel)
{
  int alloc_delay = _speculative ? max(_vc_alloc_delay, _sw_alloc_delay) : (_vc_alloc_delay + _sw_alloc_delay);
  int min_latency = 1 + _crossbar_delay + channel->GetLatency() + _routing_delay + alloc_delay + backchannel->GetLatency()  + _credit_delay;
  _next_buf[_output_channels.size()]->SetMinLatency(min_latency);
  Router::AddOutputChannel(channel, backchannel);
}

/*
 * [한국어]
 * ReadInputs — 매 사이클 라우터의 첫 번째 단계.
 *
 * 모든 입력 채널에서 flit을 수신(_ReceiveFlits)하고, 모든 출력 크레딧 채널에서
 * 크레딧을 수신(_ReceiveCredits)한다. 수신된 데이터가 있으면 _active를 true로
 * 설정하여 _InternalStep()이 실제 파이프라인을 진행하도록 한다.
 */
void IQRouter::ReadInputs( )
{
  bool have_flits = _ReceiveFlits( );
  bool have_credits = _ReceiveCredits( );
  _active = _active || have_flits || have_credits;
}

/*
 * [한국어]
 * _InternalStep — IQ 라우터의 메인 사이클 루프.
 *
 * _active가 false이면 아무 작업도 하지 않는다.
 * 그렇지 않으면 다음 Evaluate/Update 쌍을 순서대로 실행한다:
 *   1) _InputQueuing: 수신 flit/크레딧을 버퍼/상태에 반영
 *   2) _RouteEvaluate / _RouteUpdate: RC
 *   3) _VCAllocEvaluate / _VCAllocUpdate: VA (별도 allocator가 있을 때)
 *   4) _SWHoldEvaluate / _SWHoldUpdate: 스위치 홀드
 *   5) _SWAllocEvaluate / _SWAllocUpdate: SA
 *   6) _SwitchEvaluate / _SwitchUpdate: ST
 * 마지막으로 _OutputQueuing과 모니터 cycle()을 호출하고,
 * 파이프라인 deque에 남은 작업이 있으면 _active를 true로 유지한다.
 */
void IQRouter::_InternalStep( )
{
  if(!_active) {
    return;
  }

  _InputQueuing( );
  bool activity = !_proc_credits.empty();

  if(!_route_vcs.empty())
    _RouteEvaluate( );
  if(_vc_allocator) {
    _vc_allocator->Clear();
    if(!_vc_alloc_vcs.empty())
      _VCAllocEvaluate( );
  }
  if(_hold_switch_for_packet) {
    if(!_sw_hold_vcs.empty())
      _SWHoldEvaluate( );
  }
  _sw_allocator->Clear();
  if(_spec_sw_allocator)
    _spec_sw_allocator->Clear();
  if(!_sw_alloc_vcs.empty())
    _SWAllocEvaluate( );
  if(!_crossbar_flits.empty())
    _SwitchEvaluate( );

  if(!_route_vcs.empty()) {
    _RouteUpdate( );
    activity = activity || !_route_vcs.empty();
  }
  if(!_vc_alloc_vcs.empty()) {
    _VCAllocUpdate( );
    activity = activity || !_vc_alloc_vcs.empty();
  }
  if(_hold_switch_for_packet) {
    if(!_sw_hold_vcs.empty()) {
      _SWHoldUpdate( );
      activity = activity || !_sw_hold_vcs.empty();
    }
  }
  if(!_sw_alloc_vcs.empty()) {
    _SWAllocUpdate( );
    activity = activity || !_sw_alloc_vcs.empty();
  }
  if(!_crossbar_flits.empty()) {
    _SwitchUpdate( );
    activity = activity || !_crossbar_flits.empty();
  }

  _active = activity;

  _OutputQueuing( );

  _bufferMonitor->cycle( );
  _switchMonitor->cycle( );
}

/*
 * [한국어]
 * WriteOutputs — 매 사이클 라우터의 마지막 단계.
 *
 * _output_buffer에 대기 중인 flit을 출력 채널로 전송(_SendFlits)하고,
 * _credit_buffer에 대기 중인 크레딧을 입력 크레딧 채널로 전송(_SendCredits)한다.
 */
void IQRouter::WriteOutputs( )
{
  _SendFlits( );
  _SendCredits( );
}


//------------------------------------------------------------------------------
// read inputs
//------------------------------------------------------------------------------

/*
 * [한국어]
 * _ReceiveFlits — 모든 입력 채널에서 flit을 한 개씩 수신한다.
 *
 * 같은 사이클에 같은 입력 포트로는 flit이 1개만 도착한다고 가정하며,
 * 수신된 flit은 _in_queue_flits[input] = f 형태로 임시 저장한다.
 * TRACK_FLOWS 컴파일 옵션이 켜져 있으면 클래스별 수신 통계를 갱신한다.
 *
 * @return: flit을 하나라도 수신하면 true, 아니면 false.
 */
bool IQRouter::_ReceiveFlits( )
{
  bool activity = false;
  for(int input = 0; input < _inputs; ++input) { 
    Flit * const f = _input_channels[input]->Receive();
    if(f) {

#ifdef TRACK_FLOWS
      ++_received_flits[f->cl][input];
#endif

      if(f->watch) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		   << "Received flit " << (unsigned) f->id
		   << " from channel at input " << input
		   << "." << endl;
      }
      _in_queue_flits.insert(make_pair(input, f));
      activity = true;
    }
  }
  return activity;
}

/*
 * [한국어]
 * _ReceiveCredits — 모든 출력 크레딧 채널에서 크레딧을 수신한다.
 *
 * 수신된 Credit은 _credit_delay 사이클 후에 처리되어야 하므로,
 * _proc_credits deque에 (도착 시각, (Credit*, 출력 포트)) 형태로 저장한다.
 *
 * @return: 크레딧을 하나라도 수신하면 true, 아니면 false.
 */
bool IQRouter::_ReceiveCredits( )
{
  bool activity = false;
  for(int output = 0; output < _outputs; ++output) {  
    Credit * const c = _output_credits[output]->Receive();
    if(c) {
      _proc_credits.push_back(make_pair(GetSimTime() + _credit_delay, 
					make_pair(c, output)));
      activity = true;
    }
  }
  return activity;
}


//------------------------------------------------------------------------------
// input queuing
//------------------------------------------------------------------------------

/*
 * [한국어]
 * _InputQueuing — 이번 사이클에 수신된 flit과 크레딧을 납부하여 VC 버퍼 상태를 갱신한다.
 *
 * 주요 작업:
 *   1) _in_queue_flits에 있던 flit을 _buf[input][vc].AddFlit(f)로 삽입하고,
 *      VC 상태에 따라 다음 파이프라인 단계 deque에 등록한다.
 *      - idle 상태의 head flit:
 *          routing_delay > 0  → VC::routing, _route_vcs 등록
 *          routing_delay == 0 → VC::vc_alloc, lookahead route_set 적용,
 *                               speculative이면 _sw_alloc_vcs,
 *                               별도 allocator 있으면 _vc_alloc_vcs,
 *                               NOQ이면 _UpdateNOQ 호출
 *      - active 상태이고 front flit이면:
 *          스위치 홀드 중이면 _sw_hold_vcs, 아니면 _sw_alloc_vcs 등록
 *   2) _proc_credits에 도착 시각이 된 크레딧을 꺼내 _next_buf[output].ProcessCredit(c)
 *      로 다운스트림 VC 크레딧을 회복하고 Credit::Free()로 메모리 해제.
 *
 * 이 함수는 _InternalStep()의 Evaluate 직전에 호출되므로, 이 시점에서 등록된
 * deque 항목의 first는 -1(미평가) 상태가 된다.
 */
void IQRouter::_InputQueuing( )
{
  for(map<int, Flit *>::const_iterator iter = _in_queue_flits.begin();
      iter != _in_queue_flits.end();
      ++iter) {

    int const input = iter->first;
    assert((input >= 0) && (input < _inputs));

    Flit * const f = iter->second;
    assert(f);

    int const vc = f->vc;
    assert((vc >= 0) && (vc < _vcs));

    Buffer * const cur_buf = _buf[input];

    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		 << "Adding flit " << f->id
		 << " to VC " << vc
		 << " at input " << input
		 << " (state: " << VC::VCSTATE[cur_buf->GetState(vc)];
      if(cur_buf->Empty(vc)) {
	*gWatchOut << ", empty";
      } else {
	assert(cur_buf->FrontFlit(vc));
	*gWatchOut << ", front: " << cur_buf->FrontFlit(vc)->id;
      }
      *gWatchOut << ")." << endl;
    }
    cur_buf->AddFlit(vc, f);

#ifdef TRACK_FLOWS
    ++_stored_flits[f->cl][input];
    if(f->head) ++_active_packets[f->cl][input];
#endif

    _bufferMonitor->write(input, f) ;

    // [한국어] VC가 idle이면 방금 추가된 flit은 패킷의 head flit이어야 한다.
    if(cur_buf->GetState(vc) == VC::idle) {
      assert(cur_buf->FrontFlit(vc) == f);
      assert(cur_buf->GetOccupancy(vc) == 1);
      assert(f->head);
      // [한국어] 새 head가 들어온 expanded_input에 기존 홀드가 있어서는 안 된다.
      assert(_switch_hold_vc[input*_input_speedup + vc%_input_speedup] != vc);
      if(_routing_delay) {
        // [한국어] RC 지연 있음: routing 단계로 진입, _route_vcs에 스케줄.
        cur_buf->SetState(vc, VC::routing);
        _route_vcs.push_back(make_pair(-1, make_pair(input, vc)));
      } else {
        // [한국어] RC 지연 없음(lookahead): flit이 이미 la_route_set을 가지고 도착.
        if(f->watch) {
          *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		     << "Using precomputed lookahead routing information for VC " << vc
		     << " at input " << input
		     << " (front: " << f->id
		     << ")." << endl;
        }
        cur_buf->SetRouteSet(vc, &f->la_route_set);
        cur_buf->SetState(vc, VC::vc_alloc);
        // [한국어] 투기적 SA가 켜져 있으면 VC 할당 전에도 SA에 입찰 시도.
        if(_speculative) {
          _sw_alloc_vcs.push_back(make_pair(-1, make_pair(make_pair(input, vc),
							  -1)));
        }
        // [한국어] 별도 VC allocator가 있으면 VA 단계 등록.
        if(_vc_allocator) {
          _vc_alloc_vcs.push_back(make_pair(-1, make_pair(make_pair(input, vc), 
							  -1)));
        }
        // [한국어] NOQ 모드: 다음 홉 라우팅 정보를 미리 계산해 저장.
        if(_noq) {
          _UpdateNOQ(input, vc, f);
        }
      }
    } else if((cur_buf->GetState(vc) == VC::active) &&
	      (cur_buf->FrontFlit(vc) == f)) {
      // [한국어] 이미 active 상태이고 front flit이면 SA(또는 홀드 재사용)로 진행.
      if(_switch_hold_vc[input*_input_speedup + vc%_input_speedup] == vc) {
	_sw_hold_vcs.push_back(make_pair(-1, make_pair(make_pair(input, vc),
						       -1)));
      } else {
	_sw_alloc_vcs.push_back(make_pair(-1, make_pair(make_pair(input, vc), 
							-1)));
      }
    }
  }
  _in_queue_flits.clear();

  // [한국어] _proc_credits에 예약된 크레딧 중 도착 시각이 된 것을 처리.
  while(!_proc_credits.empty()) {

    pair<int, pair<Credit *, int> > const & item = _proc_credits.front();

    int const time = item.first;
    if(GetSimTime() < time) {
      break;
    }

    Credit * const c = item.second.first;
    assert(c);

    int const output = item.second.second;
    assert((output >= 0) && (output < _outputs));
    
    BufferState * const dest_buf = _next_buf[output];
    
#ifdef TRACK_FLOWS
    for(set<int>::const_iterator iter = c->vc.begin(); iter != c->vc.end(); ++iter) {
      int const vc = *iter;
      assert(!_outstanding_classes[output][vc].empty());
      int cl = _outstanding_classes[output][vc].front();
      _outstanding_classes[output][vc].pop();
      assert(_outstanding_credits[cl][output] > 0);
      --_outstanding_credits[cl][output];
    }
#endif

    dest_buf->ProcessCredit(c);
    c->Free();
    _proc_credits.pop_front();
  }
}


//------------------------------------------------------------------------------
// routing
//------------------------------------------------------------------------------

/*
 * [한국어]
 * _RouteEvaluate — RC(Route Computation) 단계의 Evaluate.
 *
 * _route_vcs에 first < 0인 항목들에 대해 완료 시각을
 * GetSimTime() + _routing_delay - 1로 설정한다.
 * 실제 라우팅 함수(_rf) 호출은 _RouteUpdate()에서 수행된다.
 * _routing_delay가 0이면 이 함수는 호출되지 않는다(lookahead).
 */
void IQRouter::_RouteEvaluate( )
{
  assert(_routing_delay);

  for(deque<pair<int, pair<int, int> > >::iterator iter = _route_vcs.begin();
      iter != _route_vcs.end();
      ++iter) {
    
    int const time = iter->first;
    if(time >= 0) {
      break;
    }
    iter->first = GetSimTime() + _routing_delay - 1;
    
    int const input = iter->second.first;
    assert((input >= 0) && (input < _inputs));
    int const vc = iter->second.second;
    assert((vc >= 0) && (vc < _vcs));

    Buffer const * const cur_buf = _buf[input];
    assert(!cur_buf->Empty(vc));
    assert(cur_buf->GetState(vc) == VC::routing);

    Flit const * const f = cur_buf->FrontFlit(vc);
    assert(f);
    assert(f->vc == vc);
    assert(f->head);

    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		 << "Beginning routing for VC " << vc
		 << " at input " << input
		 << " (front: " << f->id
		 << ")." << endl;
    }
  }    
}

/*
 * [한국어]
 * _RouteUpdate — RC 단계의 Update.
 *
 * 완료 시각이 된 _route_vcs 항목을 꺼내 cur_buf->Route(vc, _rf, ...)를 호출하여
 * 라우팅 함수를 실행하고, 결과 OutputSet을 VC에 저장한다.
 * 이후 VC 상태를 routing에서 vc_alloc으로 전환하고,
 * speculative이면 _sw_alloc_vcs에, 별도 allocator가 있으면 _vc_alloc_vcs에 등록한다.
 * NOQ는 lookahead 라우팅(routing_delay==0)을 전제로 하므로 여기서는 처리하지 않는다.
 */
void IQRouter::_RouteUpdate( )
{
  assert(_routing_delay);

  while(!_route_vcs.empty()) {

    pair<int, pair<int, int> > const & item = _route_vcs.front();

    int const time = item.first;
    if((time < 0) || (GetSimTime() < time)) {
      break;
    }
    assert(GetSimTime() == time);

    int const input = item.second.first;
    assert((input >= 0) && (input < _inputs));
    int const vc = item.second.second;
    assert((vc >= 0) && (vc < _vcs));
    
    Buffer * const cur_buf = _buf[input];
    assert(!cur_buf->Empty(vc));
    assert(cur_buf->GetState(vc) == VC::routing);

    Flit * const f = cur_buf->FrontFlit(vc);
    assert(f);
    assert(f->vc == vc);
    assert(f->head);

    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		 << "Completed routing for VC " << vc
		 << " at input " << input
		 << " (front: " << f->id
		 << ")." << endl;
    }

    cur_buf->Route(vc, _rf, this, f, input);
    cur_buf->SetState(vc, VC::vc_alloc);
    if(_speculative) {
      _sw_alloc_vcs.push_back(make_pair(-1, make_pair(item.second, -1)));
    }
    if(_vc_allocator) {
      _vc_alloc_vcs.push_back(make_pair(-1, make_pair(item.second, -1)));
    }
    // NOTE: No need to handle NOQ here, as it requires lookahead routing!
    _route_vcs.pop_front();
  }
}


//------------------------------------------------------------------------------
// VC allocation
//------------------------------------------------------------------------------

/*
 * [한국어]
 * _VCAllocEvaluate — VA(VC Allocation) 단계의 Evaluate.
 *
 * _vc_alloc_vcs에 first < 0인 항목들을 대상으로, 해당 VC의 head flit이
 * 원하는 route_set(OutputSet) 내 출력 포트/VC 범위를 순회하며 _vc_allocator에
 * 요청(AddRequest)을 등록한다. 각 요청은 in_priority(라우팅 함수 기반)와
 * out_priority(패킷 우선순위)를 가진다.
 *
 * 상태 판정:
 *   - route_set 내 모든 출력 VC가 사용 중이면 STALL_BUFFER_BUSY.
 *   - _vc_busy_when_full=true이고 크레딧이 꽉 찬 경우:
 *       버퍼 전체가 꽉 찼으면 STALL_BUFFER_FULL, 아니면 STALL_BUFFER_RESERVED.
 *   - 적어도 하나의 VC에 요청을 등록하면 정상 진행.
 *
 * 요청 등록 후 _vc_allocator->Allocate()를 호출하고, 그 결과를 다시 deque의
 * second 필드(output_and_vc 또는 STALL_*)에 기록한다.
 * _vc_alloc_delay > 1인 경우 추가적인 유효성 검사를 수행하여, 이미 발급된
 * grant가 Update 직전에 무효해진 경우(STALL_BUFFER_BUSY/FULL/RESERVED) 폐기한다.
 */
void IQRouter::_VCAllocEvaluate( )
{
  assert(_vc_allocator);

  bool watched = false;

  for(deque<pair<int, pair<pair<int, int>, int> > >::iterator iter = _vc_alloc_vcs.begin();
      iter != _vc_alloc_vcs.end();
      ++iter) {

    int const time = iter->first;
    if(time >= 0) {
      break;
    }

    int const input = iter->second.first.first;
    assert((input >= 0) && (input < _inputs));
    int const vc = iter->second.first.second;
    assert((vc >= 0) && (vc < _vcs));

    assert(iter->second.second == -1);

    Buffer const * const cur_buf = _buf[input];
    assert(!cur_buf->Empty(vc));
    assert(cur_buf->GetState(vc) == VC::vc_alloc);

    Flit const * const f = cur_buf->FrontFlit(vc);
    assert(f);
    assert(f->vc == vc);
    assert(f->head);

    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | " 
		 << "Beginning VC allocation for VC " << vc
		 << " at input " << input
		 << " (front: " << f->id
		 << ")." << endl;
    }
    
    OutputSet const * const route_set = cur_buf->GetRouteSet(vc);
    assert(route_set);

    int const out_priority = cur_buf->GetPriority(vc);
    set<OutputSet::sSetElement> const setlist = route_set->GetSet();

    bool elig = false;
    bool cred = false;
    bool reserved = false;

    // [한국어] NOQ 모드에서는 route_set이 단일 출력 포트로 축소되어야 한다.
    assert(!_noq || (setlist.size() == 1));

    for(set<OutputSet::sSetElement>::const_iterator iset = setlist.begin();
	iset != setlist.end();
	++iset) {

      int const out_port = iset->output_port;
      assert((out_port >= 0) && (out_port < _outputs));

      BufferState const * const dest_buf = _next_buf[out_port];

      int vc_start;
      int vc_end;
      
      // [한국어] NOQ 모드에서는 _UpdateNOQ()에서 미리 계산한 다음 홉 VC 범위를 사용.
      if(_noq && _noq_next_output_port[input][vc] >= 0) {
	assert(!_routing_delay);
	vc_start = _noq_next_vc_start[input][vc];
	vc_end = _noq_next_vc_end[input][vc];
      } else {
	vc_start = iset->vc_start;
	vc_end = iset->vc_end;
      }
      assert(vc_start >= 0 && vc_start < _vcs);
      assert(vc_end >= 0 && vc_end < _vcs);
      assert(vc_end >= vc_start);

      for(int out_vc = vc_start; out_vc <= vc_end; ++out_vc) {
	assert((out_vc >= 0) && (out_vc < _vcs));

	int in_priority = iset->pri;
	// [한국어] 비어 있지 않은 출력 VC의 우선순위를 최저로 낮춰 빈 VC를 선호.
	if(_vc_prioritize_empty && !dest_buf->IsEmptyFor(out_vc)) {
	  assert(in_priority >= 0);
	  in_priority += numeric_limits<int>::min();
	}

	// On the input input side, a VC might request several output VCs. 
	// These VCs can be prioritized by the routing function, and this is 
	// reflected in "in_priority". On the output side, if multiple VCs are 
	// requesting the same output VC, the priority of VCs is based on the 
	// actual packet priorities, which is reflected in "out_priority".
	
	if(!dest_buf->IsAvailableFor(out_vc)) {
	  if(f->watch) {
	    int const use_input_and_vc = dest_buf->UsedBy(out_vc);
	    int const use_input = use_input_and_vc / _vcs;
	    int const use_vc = use_input_and_vc % _vcs;
	    *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		       << "  VC " << out_vc 
		       << " at output " << out_port 
		       << " is in use by VC " << use_vc
		       << " at input " << use_input;
	    Flit * cf = _buf[use_input]->FrontFlit(use_vc);
	    if(cf) {
	      *gWatchOut << " (front flit: " << cf->id << ")";
	    } else {
	      *gWatchOut << " (empty)";
	    }
	    *gWatchOut << "." << endl;
	  }
	} else {
	  elig = true;
	  // [한국어] _vc_busy_when_full 정책: 크레딧이 꽉 찬 VC도 일단 예약(reserved) 가능.
	  if(_vc_busy_when_full && dest_buf->IsFullFor(out_vc)) {
	    if(f->watch)
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "  VC " << out_vc 
			 << " at output " << out_port 
			 << " is full." << endl;
	    reserved |= !dest_buf->IsFull();
	  } else {
	    cred = true;
	    if(f->watch){
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "  Requesting VC " << out_vc
			 << " at output " << out_port 
			 << " (in_pri: " << in_priority
			 << ", out_pri: " << out_priority
			 << ")." << endl;
	      watched = true;
	    }
	    // [한국어] vc_shuffle_requests에 따라 allocator 입력 인덱스 순서를 변경.
	    int const input_and_vc
	      = _vc_shuffle_requests ? (vc*_inputs + input) : (input*_vcs + vc);
	    _vc_allocator->AddRequest(input_and_vc, out_port*_vcs + out_vc, 
				      0, in_priority, out_priority);
	  }
	}
      }
    }
    // [한국어] 요청을 한 곳도 등록하지 못한 경우 실패 코드 기록.
    if(!elig) {
      iter->second.second = STALL_BUFFER_BUSY;
    } else if(_vc_busy_when_full && !cred) {
      iter->second.second = reserved ? STALL_BUFFER_RESERVED : STALL_BUFFER_FULL;
    }
  }

  if(watched) {
    *gWatchOut << GetSimTime() << " | " << _vc_allocator->FullName() << " | ";
    _vc_allocator->PrintRequests( gWatchOut );
  }

  _vc_allocator->Allocate();

  if(watched) {
    *gWatchOut << GetSimTime() << " | " << _vc_allocator->FullName() << " | ";
    _vc_allocator->PrintGrants( gWatchOut );
  }

  // [한국어] allocator 결과를 deque에 기록하고 완료 시각을 설정.
  for(deque<pair<int, pair<pair<int, int>, int> > >::iterator iter = _vc_alloc_vcs.begin();
      iter != _vc_alloc_vcs.end();
      ++iter) {

    int const time = iter->first;
    if(time >= 0) {
      break;
    }
    iter->first = GetSimTime() + _vc_alloc_delay - 1;

    int const input = iter->second.first.first;
    assert((input >= 0) && (input < _inputs));
    int const vc = iter->second.first.second;
    assert((vc >= 0) && (vc < _vcs));

    if(iter->second.second < -1) {
      continue;
    }

    assert(iter->second.second == -1);

    Buffer const * const cur_buf = _buf[input];
    assert(!cur_buf->Empty(vc));
    assert(cur_buf->GetState(vc) == VC::vc_alloc);

    Flit const * const f = cur_buf->FrontFlit(vc);
    assert(f);
    assert(f->vc == vc);
    assert(f->head);

    int const input_and_vc
      = _vc_shuffle_requests ? (vc*_inputs + input) : (input*_vcs + vc);
    int const output_and_vc = _vc_allocator->OutputAssigned(input_and_vc);

    if(output_and_vc >= 0) {

      int const match_output = output_and_vc / _vcs;
      assert((match_output >= 0) && (match_output < _outputs));
      int const match_vc = output_and_vc % _vcs;
      assert((match_vc >= 0) && (match_vc < _vcs));

      if(f->watch) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		   << "Assigning VC " << match_vc
		   << " at output " << match_output 
		   << " to VC " << vc
		   << " at input " << input
		   << "." << endl;
      }

      iter->second.second = output_and_vc;

    } else {

      if(f->watch) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		   << "VC allocation failed for VC " << vc
		   << " at input " << input
		   << "." << endl;
      }
      
      iter->second.second = STALL_BUFFER_CONFLICT;

    }
  }

  if(_vc_alloc_delay <= 1) {
    return;
  }

  // [한국어] _vc_alloc_delay > 1일 때, 이미 발급된 grant가 Update 직전에
  // 무효해진 경우(다른 라우터/VC에 의해 VC가 점유되거나 꽉 참) 폐기한다.
  for(deque<pair<int, pair<pair<int, int>, int> > >::iterator iter = _vc_alloc_vcs.begin();
      iter != _vc_alloc_vcs.end();
      ++iter) {
    
    int const time = iter->first;
    assert(time >= 0);
    if(GetSimTime() < time) {
      break;
    }
    
    assert(iter->second.second != -1);

    int const output_and_vc = iter->second.second;
    
    if(output_and_vc >= 0) {
      
      int const match_output = output_and_vc / _vcs;
      assert((match_output >= 0) && (match_output < _outputs));
      int const match_vc = output_and_vc % _vcs;
      assert((match_vc >= 0) && (match_vc < _vcs));
      
      BufferState const * const dest_buf = _next_buf[match_output];
      
      int const input = iter->second.first.first;
      assert((input >= 0) && (input < _inputs));
      int const vc = iter->second.first.second;
      assert((vc >= 0) && (vc < _vcs));
      
      Buffer const * const cur_buf = _buf[input];
      assert(!cur_buf->Empty(vc));
      assert(cur_buf->GetState(vc) == VC::vc_alloc);
      
      Flit const * const f = cur_buf->FrontFlit(vc);
      assert(f);
      assert(f->vc == vc);
      assert(f->head);
      
      if(!dest_buf->IsAvailableFor(match_vc)) {
	if(f->watch) {
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		     << "  Discarding previously generated grant for VC " << vc
		     << " at input " << input
		     << ": VC " << match_vc
		     << " at output " << match_output
		     << " is no longer available." << endl;
	}
	iter->second.second = STALL_BUFFER_BUSY;
      } else if(_vc_busy_when_full && dest_buf->IsFullFor(match_vc)) {
	if(f->watch) {
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		     << "  Discarding previously generated grant for VC " << vc
		     << " at input " << input
		     << ": VC " << match_vc
		     << " at output " << match_output
		     << " has become full." << endl;
	}
	iter->second.second = dest_buf->IsFull() ? STALL_BUFFER_FULL : STALL_BUFFER_RESERVED;
      }
    }
  }
}

/*
 * [한국어]
 * _VCAllocUpdate — VA 단계의 Update.
 *
 * 완료 시각이 된 _vc_alloc_vcs 항목을 꺼내 allocator 결과(output_and_vc)를 적용한다.
 *   - output_and_vc >= 0: dest_buf->TakeBuffer()로 출력 VC를 점유하고,
 *     cur_buf->SetOutput()로 출력 포트/VC를 기록한 뒤 VC 상태를 active로 전환.
 *     non-speculative 모드에서는 이 시점에 _sw_alloc_vcs에 등록하여 SA를 진행.
 *   - output_and_vc < 0 (STALL_*): 실패로 처리하고 항목을 _vc_alloc_vcs의 맨 뒤에
 *     재등록하여 다음 사이클 다시 시도.
 */
void IQRouter::_VCAllocUpdate( )
{
  assert(_vc_allocator);

  while(!_vc_alloc_vcs.empty()) {

    pair<int, pair<pair<int, int>, int> > const & item = _vc_alloc_vcs.front();

    int const time = item.first;
    if((time < 0) || (GetSimTime() < time)) {
      break;
    }
    assert(GetSimTime() == time);

    int const input = item.second.first.first;
    assert((input >= 0) && (input < _inputs));
    int const vc = item.second.first.second;
    assert((vc >= 0) && (vc < _vcs));
    
    assert(item.second.second != -1);

    Buffer * const cur_buf = _buf[input];
    assert(!cur_buf->Empty(vc));
    assert(cur_buf->GetState(vc) == VC::vc_alloc);
    
    Flit const * const f = cur_buf->FrontFlit(vc);
    assert(f);
    assert(f->vc == vc);
    assert(f->head);
    
    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		 << "Completed VC allocation for VC " << vc
		 << " at input " << input
		 << " (front: " << f->id
		 << ")." << endl;
    }
    
    int const output_and_vc = item.second.second;
    
    if(output_and_vc >= 0) {
      
      int const match_output = output_and_vc / _vcs;
      assert((match_output >= 0) && (match_output < _outputs));
      int const match_vc = output_and_vc % _vcs;
      assert((match_vc >= 0) && (match_vc < _vcs));
      
      if(f->watch) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		   << "  Acquiring assigned VC " << match_vc
		   << " at output " << match_output
		   << "." << endl;
      }
      
      BufferState * const dest_buf = _next_buf[match_output];
      assert(dest_buf->IsAvailableFor(match_vc));
      
      // [한국어] 출력 VC의 소유권을 (input*_vcs + vc)로 등록.
      dest_buf->TakeBuffer(match_vc, input*_vcs + vc);
	
      cur_buf->SetOutput(vc, match_output, match_vc);
      cur_buf->SetState(vc, VC::active);
      // [한국어] non-speculative 모드에서는 VC 할당이 끝난 뒤 SA에 등록.
      if(!_speculative) {
	_sw_alloc_vcs.push_back(make_pair(-1, make_pair(item.second.first, -1)));
      }
    } else {
      if(f->watch) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		   << "  No output VC allocated." << endl;
      }

#ifdef TRACK_STALLS
      assert((output_and_vc == STALL_BUFFER_BUSY) ||
	     (output_and_vc == STALL_BUFFER_CONFLICT));
      if(output_and_vc == STALL_BUFFER_BUSY) {
	++_buffer_busy_stalls[f->cl];
      } else if(output_and_vc == STALL_BUFFER_CONFLICT) {
	++_buffer_conflict_stalls[f->cl];
      }
#endif

      // [한국어] VA 실패: 다음 사이클에 다시 시도하기 위해 deque 뒤로 재등록.
      _vc_alloc_vcs.push_back(make_pair(-1, make_pair(item.second.first, -1)));
    }
    _vc_alloc_vcs.pop_front();
  }
}


//------------------------------------------------------------------------------
// switch holding
//------------------------------------------------------------------------------

/*
 * [한국어]
 * _SWHoldEvaluate — 스위치 홀드(Switch Hold) 단계의 Evaluate.
 *
 * _sw_hold_vcs에 first < 0인 항목들을 대상으로, 이미 예약된 크로스바 경로를
 * 재사용할 수 있는지 확인한다. 다운스트림 VC의 크레딧이 남아 있으면
 * expanded_output을 result에 기록하고, 없으면 STALL_BUFFER_FULL/RESERVED를 기록한다.
 */
void IQRouter::_SWHoldEvaluate( )
{
  assert(_hold_switch_for_packet);

  for(deque<pair<int, pair<pair<int, int>, int> > >::iterator iter = _sw_hold_vcs.begin();
      iter != _sw_hold_vcs.end();
      ++iter) {
    
    int const time = iter->first;
    if(time >= 0) {
      break;
    }
    iter->first = GetSimTime();
    
    int const input = iter->second.first.first;
    assert((input >= 0) && (input < _inputs));
    int const vc = iter->second.first.second;
    assert((vc >= 0) && (vc < _vcs));
    
    assert(iter->second.second == -1);

    Buffer const * const cur_buf = _buf[input];
    assert(!cur_buf->Empty(vc));
    assert(cur_buf->GetState(vc) == VC::active);
    
    Flit const * const f = cur_buf->FrontFlit(vc);
    assert(f);
    assert(f->vc == vc);

    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | " 
		 << "Beginning held switch allocation for VC " << vc
		 << " at input " << input
		 << " (front: " << f->id
		 << ")." << endl;
    }
    
    int const expanded_input = input * _input_speedup + vc % _input_speedup;
    assert(_switch_hold_vc[expanded_input] == vc);
    
    int const match_port = cur_buf->GetOutputPort(vc);
    assert((match_port >= 0) && (match_port < _outputs));
    int const match_vc = cur_buf->GetOutputVC(vc);
    assert((match_vc >= 0) && (match_vc < _vcs));
    
    int const expanded_output = match_port*_output_speedup + input%_output_speedup;
    assert(_switch_hold_in[expanded_input] == expanded_output);
    
    BufferState const * const dest_buf = _next_buf[match_port];
    
    // [한국어] 다운스트림 VC에 크레딧이 없으면 홀드 경로도 사용 불가.
    if(dest_buf->IsFullFor(match_vc)) {
      if(f->watch) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		   << "  Unable to reuse held connection from input " << input
		   << "." << (expanded_input % _input_speedup)
		   << " to output " << match_port
		   << "." << (expanded_output % _output_speedup)
		   << ": No credit available." << endl;
      }
      iter->second.second = dest_buf->IsFull() ? STALL_BUFFER_FULL : STALL_BUFFER_RESERVED;
    } else {
      if(f->watch) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		   << "  Reusing held connection from input " << input
		   << "." << (expanded_input % _input_speedup)
		   << " to output " << match_port
		   << "." << (expanded_output % _output_speedup)
		   << "." << endl;
      }
      iter->second.second = expanded_output;
    }
  }
}


/*
 * [한국어]
 * _SWHoldUpdate — 스위치 홀드 단계의 Update.
 *
 * _sw_hold_vcs에서 완료 시각이 된 항목을 꺼내 홀드 경로를 재사용하여 flit을 전송한다.
 *   - expanded_output >= 0이고 출력 버퍼에 여유가 있으면:
 *       cur_buf->RemoveFlit(vc), 버퍼 모니터 read, f->hops++, f->vc 갱신,
 *       lookahead 라우팅 정보 갱신(필요 시), dest_buf->SendingFlit(),
 *       _crossbar_flits 등록, 업스트림 크레딧 생성.
 *       버퍼가 비면 홀드 해제. tail flit이면 다음 head flit을 routing/vc_alloc으로 전환.
 *       tail이 아니면 홀드 유지(_sw_hold_vcs 재등록).
 *   - expanded_output < 0(스토올)이거나 출력 버퍼 꽉 참:
 *       홀드 해제하고 일반 SA 경로(_sw_alloc_vcs)로 복귀.
 */
void IQRouter::_SWHoldUpdate( )
{
  assert(_hold_switch_for_packet);

  while(!_sw_hold_vcs.empty()) {
    
    pair<int, pair<pair<int, int>, int> > const & item = _sw_hold_vcs.front();
    
    int const time = item.first;
    if(time < 0) {
      break;
    }
    assert(GetSimTime() == time);
    
    int const input = item.second.first.first;
    assert((input >= 0) && (input < _inputs));
    int const vc = item.second.first.second;
    assert((vc >= 0) && (vc < _vcs));
    
    assert(item.second.second != -1);

    Buffer * const cur_buf = _buf[input];
    assert(!cur_buf->Empty(vc));
    assert(cur_buf->GetState(vc) == VC::active);
    
    Flit * const f = cur_buf->FrontFlit(vc);
    assert(f);
    assert(f->vc == vc);

    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		 << "Completed held switch allocation for VC " << vc
		 << " at input " << input
		 << " (front: " << f->id
		 << ")." << endl;
    }
    
    int const expanded_input = input * _input_speedup + vc % _input_speedup;
    assert(_switch_hold_vc[expanded_input] == vc);
    
    int const expanded_output = item.second.second;
    
    // [한국어] 홀드 경로가 유효하고 출력 버퍼에 공간이 있을 때만 flit 전송.
    if(expanded_output >= 0 && ( _output_buffer_size==-1 || _output_buffer[expanded_output].size()<size_t(_output_buffer_size))) {
      
      assert(_switch_hold_in[expanded_input] == expanded_output);
      assert(_switch_hold_out[expanded_output] == expanded_input);
      
      int const output = expanded_output / _output_speedup;
      assert((output >= 0) && (output < _outputs));
      assert(cur_buf->GetOutputPort(vc) == output);
      
      int const match_vc = cur_buf->GetOutputVC(vc);
      assert((match_vc >= 0) && (match_vc < _vcs));
      
      BufferState * const dest_buf = _next_buf[output];
      
      if(f->watch) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		   << "  Scheduling switch connection from input " << input
		   << "." << (vc % _input_speedup)
		   << " to output " << output
		   << "." << (expanded_output % _output_speedup)
		   << "." << endl;
      }
      
      cur_buf->RemoveFlit(vc);

#ifdef TRACK_FLOWS
      --_stored_flits[f->cl][input];
      if(f->tail) --_active_packets[f->cl][input];
#endif

      _bufferMonitor->read(input, f) ;
      
      f->hops++;
      f->vc = match_vc;
      
      // [한국어] lookahead 라우팅: head flit이면 다음 홉의 la_route_set 갱신.
      if(!_routing_delay && f->head) {
	const FlitChannel * channel = _output_channels[output];
	const Router * router = channel->GetSink();
	if(router) {
	  if(_noq) {
	    if(f->watch) {
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "Updating lookahead routing information for flit " << f->id
			 << " (NOQ)." << endl;
	    }
	    int next_output_port = _noq_next_output_port[input][vc];
	    assert(next_output_port >= 0);
	    _noq_next_output_port[input][vc] = -1;
	    int next_vc_start = _noq_next_vc_start[input][vc];
	    assert(next_vc_start >= 0 && next_vc_start < _vcs);
	    _noq_next_vc_start[input][vc] = -1;
	    int next_vc_end = _noq_next_vc_end[input][vc];
	    assert(next_vc_end >= 0 && next_vc_end < _vcs);
	    _noq_next_vc_end[input][vc] = -1;
	    f->la_route_set.Clear();
	    f->la_route_set.AddRange(next_output_port, next_vc_start, next_vc_end);
	  } else {
	    if(f->watch) {
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "Updating lookahead routing information for flit " << f->id
			 << "." << endl;
	    }
	    int in_channel = channel->GetSinkPort();
	    _rf(router, f, in_channel, &f->la_route_set, false);
	  }
	} else {
	  f->la_route_set.Clear();
	}
      }

#ifdef TRACK_FLOWS
      ++_outstanding_credits[f->cl][output];
      _outstanding_classes[output][f->vc].push(f->cl);
#endif

      dest_buf->SendingFlit(f);

      _crossbar_flits.push_back(make_pair(-1, make_pair(f, make_pair(expanded_input, expanded_output))));
      
      // [한국어] 입력 포트에 대한 업스트림 크레딧 생성/갱신.
      if(_out_queue_credits.count(input) == 0) {
	_out_queue_credits.insert(make_pair(input, Credit::New()));
      }
      _out_queue_credits.find(input)->second->vc.insert(vc);
      
      // [한국어] 버퍼가 비거나 패킷 끝이면 홀드 해제.
      if(cur_buf->Empty(vc)) {
	if(f->watch) {
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		     << "  Cancelling held connection from input " << input
		     << "." << (expanded_input % _input_speedup)
		     << " to " << output
		     << "." << (expanded_output % _output_speedup)
		     << ": No more flits." << endl;
	}
	_switch_hold_vc[expanded_input] = -1;
	_switch_hold_in[expanded_input] = -1;
	_switch_hold_out[expanded_output] = -1;
	if(f->tail) {
	  cur_buf->SetState(vc, VC::idle);
	}
      } else {
	Flit * const nf = cur_buf->FrontFlit(vc);
	assert(nf);
	assert(nf->vc == vc);
	if(f->tail) {
	  assert(nf->head);
	  if(f->watch) {
	    *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		       << "  Cancelling held connection from input " << input
		       << "." << (expanded_input % _input_speedup)
		       << " to " << output
		       << "." << (expanded_output % _output_speedup)
		       << ": End of packet." << endl;
	  }
	  _switch_hold_vc[expanded_input] = -1;
	  _switch_hold_in[expanded_input] = -1;
	  _switch_hold_out[expanded_output] = -1;
	  // [한국어] 다음 패킷 head flit이므로 RC 또는 VC alloc 단계로 전환.
	  if(_routing_delay) {
	    cur_buf->SetState(vc, VC::routing);
	    _route_vcs.push_back(make_pair(-1, item.second.first));
	  } else {
	    if(nf->watch) {
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "Using precomputed lookahead routing information for VC " << vc
			 << " at input " << input
			 << " (front: " << nf->id
			 << ")." << endl;
	    }
	    cur_buf->SetRouteSet(vc, &nf->la_route_set);
	    cur_buf->SetState(vc, VC::vc_alloc);
	    if(_speculative) {
	      _sw_alloc_vcs.push_back(make_pair(-1, make_pair(item.second.first,
							      -1)));
	    }
	    if(_vc_allocator) {
	      _vc_alloc_vcs.push_back(make_pair(-1, make_pair(item.second.first,
							      -1)));
	    }
	    if(_noq) {
	      _UpdateNOQ(input, vc, nf);
	    }
	  }
	} else {
	  // [한국어] 동일 패킷의 다음 flit이 남았으므로 홀드 경로 유지.
	  _sw_hold_vcs.push_back(make_pair(-1, make_pair(item.second.first,
							 -1)));
	}
      }
    } else {
      //when internal speedup >1.0, the buffer stall stats may not be accruate
      // [한국어] 홀드 경로 사용 불가(크레딧 부족 또는 출력 버퍼 꽉 참). 홀드 해제 후
      // 일반 SA 경로로 복귀한다.
      assert((expanded_output == STALL_BUFFER_FULL) ||
	     (expanded_output == STALL_BUFFER_RESERVED) || !( _output_buffer_size==-1 || _output_buffer[expanded_output].size()<size_t(_output_buffer_size)));

      int const held_expanded_output = _switch_hold_in[expanded_input];
      assert(held_expanded_output >= 0);
      
      if(f->watch) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		   << "  Cancelling held connection from input " << input
		   << "." << (expanded_input % _input_speedup)
		   << " to " << (held_expanded_output / _output_speedup)
		   << "." << (held_expanded_output % _output_speedup)
		   << ": Flit not sent." << endl;
      }
      _switch_hold_vc[expanded_input] = -1;
      _switch_hold_in[expanded_input] = -1;
      _switch_hold_out[held_expanded_output] = -1;
      _sw_alloc_vcs.push_back(make_pair(-1, make_pair(item.second.first,
					      -1)));
    }
    _sw_hold_vcs.pop_front();
  }
}


//------------------------------------------------------------------------------
// switch allocation
//------------------------------------------------------------------------------

/*
 * [한국어]
 * _SWAllocAddReq — 스위치 할당자에 (input, vc, output) 요청을 등록한다.
 *
 * input/output speedup을 고려하여 expanded_input/output을 계산하고,
 * 해당 경로가 스위치 홀드로 점유되지 않았는지 확인한다.
 * 이미 동일 expanded_input/output에 다른 VC의 요청이 등록되어 있으면
 * RoundRobinArbiter::Supersedes()로 우선순위를 비교하여 높은 우선순위 요청으로
 * 교체(supersede)한다. 투기적 요청(_speculative && VC::vc_alloc)은 낮은 우선순위로
 * 처리되며, 별도 _spec_sw_allocator가 있으면 그쪽에 등록한다.
 *
 * @return: 새 요청을 등록(또는 교체)했으면 true, 기존 요청이 우선순위가 높아
 *          등록하지 못했으면 false.
 */
bool IQRouter::_SWAllocAddReq(int input, int vc, int output)
{
  assert(input >= 0 && input < _inputs);
  assert(vc >= 0 && vc < _vcs);
  assert(output >= 0 && output < _outputs);
  
  // When input_speedup > 1, the virtual channel buffers are interleaved to 
  // create multiple input ports to the switch. Similarily, the output ports 
  // are interleaved based on their originating input when output_speedup > 1.
  
  int const expanded_input = input * _input_speedup + vc % _input_speedup;
  int const expanded_output = output * _output_speedup + input % _output_speedup;
  
  Buffer const * const cur_buf = _buf[input];
  assert(!cur_buf->Empty(vc));
  assert((cur_buf->GetState(vc) == VC::active) || 
	 (_speculative && (cur_buf->GetState(vc) == VC::vc_alloc)));
  
  Flit const * const f = cur_buf->FrontFlit(vc);
  assert(f);
  assert(f->vc == vc);
  
  // [한국어] 홀드 중인 expanded_input/output에는 새 SA 요청 불가.
  if((_switch_hold_in[expanded_input] < 0) && 
     (_switch_hold_out[expanded_output] < 0)) {
    
    Allocator * allocator = _sw_allocator;
    int prio = cur_buf->GetPriority(vc);
    
    // [한국어] 투기적 요청: 별도 allocator가 없으면 _sw_allocator에서 최저 우선순위로.
    if(_speculative && (cur_buf->GetState(vc) == VC::vc_alloc)) {
      if(_spec_sw_allocator) {
	allocator = _spec_sw_allocator;
      } else {
	assert(prio >= 0);
	prio += numeric_limits<int>::min();
      }
    }
    
    Allocator::sRequest req;
    
    // [한국어] 동일 경로에 이미 요청이 있으면 우선순위 비교 후 교체 결정.
    if(allocator->ReadRequest(req, expanded_input, expanded_output)) {
      if(RoundRobinArbiter::Supersedes(vc, prio, req.label, req.in_pri, 
				       _sw_rr_offset[expanded_input], _vcs)) {
	if(f->watch) {
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		     << "  Replacing earlier request from VC " << req.label
		     << " for output " << output 
		     << "." << (expanded_output % _output_speedup)
		     << " with priority " << req.in_pri
		     << " (" << ((cur_buf->GetState(vc) == VC::active) ? 
				 "non-spec" : 
				 "spec")
		     << ", pri: " << prio
		     << ")." << endl;
	}
	allocator->RemoveRequest(expanded_input, expanded_output, req.label);
	allocator->AddRequest(expanded_input, expanded_output, vc, prio, prio);
	return true;
      }
      if(f->watch) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		   << "  Output " << output
		   << "." << (expanded_output % _output_speedup)
		   << " was already requested by VC " << req.label
		   << " with priority " << req.in_pri
		   << " (pri: " << prio
		   << ")." << endl;
      }
      return false;
    }
    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		 << "  Requesting output " << output
		 << "." << (expanded_output % _output_speedup)
		 << " (" << ((cur_buf->GetState(vc) == VC::active) ? 
			     "non-spec" : 
			     "spec")
		 << ", pri: " << prio
		 << ")." << endl;
    }
    allocator->AddRequest(expanded_input, expanded_output, vc, prio, prio);
    return true;
  }
  if(f->watch) {
    *gWatchOut << GetSimTime() << " | " << FullName() << " | "
	       << "  Ignoring output " << output
	       << "." << (expanded_output % _output_speedup)
	       << " due to switch hold (";
    if(_switch_hold_in[expanded_input] >= 0) {
      *gWatchOut << "input: " << input
		 << "." << (expanded_input % _input_speedup);
      if(_switch_hold_out[expanded_output] >= 0) {
	*gWatchOut << ", ";
      }
    }
    if(_switch_hold_out[expanded_output] >= 0) {
      *gWatchOut << "output: " << output
		 << "." << (expanded_output % _output_speedup);
    }
    *gWatchOut << ")." << endl;
  }
  return false;
}

/*
 * [한국어]
 * _SWAllocEvaluate — SA(Switch Allocation) 단계의 Evaluate.
 *
 * _sw_alloc_vcs에 first < 0인 항목들을 대상으로, active 상태의 VC는 이미 할당된
 * 출력 포트로 SA 요청을 본다. VC::vc_alloc 상태의 VC는 투기적(speculative) 요청을
 * 볼 수 있으며, _spec_check_elig/_spec_check_cred 옵션에 따라 출력 VC 가용성을
 * 사전 확인한다.
 *
 * 요청 등록 후 _sw_allocator->Allocate()를 호출하고, 투기적 allocator가 있으면
 * _spec_sw_allocator->Allocate()도 호출한다. 그 결과를 deque의 result 필드에
 * 기록하며, 그랜트가 잘못된 경우(_spec_mask_by_reqs 등)는 STALL_CROSSBAR_CONFLICT로
 * 기록한다. _sw_alloc_delay > 1이거나 speculative일 때 추가 유효성 검사를 수행.
 */
void IQRouter::_SWAllocEvaluate( )
{
  bool watched = false;

  for(deque<pair<int, pair<pair<int, int>, int> > >::iterator iter = _sw_alloc_vcs.begin();
      iter != _sw_alloc_vcs.end();
      ++iter) {

    int const time = iter->first;
    if(time >= 0) {
      break;
    }

    int const input = iter->second.first.first;
    assert((input >= 0) && (input < _inputs));
    int const vc = iter->second.first.second;
    assert((vc >= 0) && (vc < _vcs));
    
    assert(iter->second.second == -1);

    // [한국어] SA 요청은 홀드 중인 VC가 아니어야 한다.
    assert(_switch_hold_vc[input * _input_speedup + vc % _input_speedup] != vc);

    Buffer const * const cur_buf = _buf[input];
    assert(!cur_buf->Empty(vc));
    assert((cur_buf->GetState(vc) == VC::active) || 
	   (_speculative && (cur_buf->GetState(vc) == VC::vc_alloc)));
    
    Flit const * const f = cur_buf->FrontFlit(vc);
    assert(f);
    assert(f->vc == vc);

    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | " 
		 << "Beginning switch allocation for VC " << vc
		 << " at input " << input
		 << " (front: " << f->id
		 << ")." << endl;
    }
    
    // [한국어] active 상태: 이미 VC alloc이 완료되어 출력 포트/VC가 결정됨.
    if(cur_buf->GetState(vc) == VC::active) {
      
      int const dest_output = cur_buf->GetOutputPort(vc);
      assert((dest_output >= 0) && (dest_output < _outputs));
      int const dest_vc = cur_buf->GetOutputVC(vc);
      assert((dest_vc >= 0) && (dest_vc < _vcs));
      
      BufferState const * const dest_buf = _next_buf[dest_output];
      
      // [한국어] 다운스트림 VC 크레딧 부족 또는 출력 버퍼 꽉 참이면 스토올.
      if(dest_buf->IsFullFor(dest_vc) || ( _output_buffer_size!=-1  && _output_buffer[dest_output].size()>=(size_t)(_output_buffer_size))) {
	if(f->watch) {
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		     << "  VC " << dest_vc 
		     << " at output " << dest_output 
		     << " is full." << endl;
	}
	iter->second.second = dest_buf->IsFull() ? STALL_BUFFER_FULL : STALL_BUFFER_RESERVED;
	continue;
      }
      bool const requested = _SWAllocAddReq(input, vc, dest_output);
      watched |= requested && f->watch;
      continue;
    }
    // [한국어] VC::vc_alloc 상태: 투기적 SA 요청만 가능.
    assert(_speculative && (cur_buf->GetState(vc) == VC::vc_alloc));
    assert(f->head);
      
    // The following models the speculative VC allocation aspects of the 
    // pipeline. An input VC with a request in for an egress virtual channel 
    // will also speculatively bid for the switch regardless of whether the VC  
    // allocation succeeds.
    
    OutputSet const * const route_set = cur_buf->GetRouteSet(vc);
    assert(route_set);
    
    set<OutputSet::sSetElement> const setlist = route_set->GetSet();
    
    assert(!_noq || (setlist.size() == 1));

    for(set<OutputSet::sSetElement>::const_iterator iset = setlist.begin();
	iset != setlist.end();
	++iset) {
      
      int const dest_output = iset->output_port;
      assert((dest_output >= 0) && (dest_output < _outputs));
      
      BufferState const * const dest_buf = _next_buf[dest_output];
	
      bool elig = false;
      bool cred = false;

      // [한국어] _spec_check_elig가 켜진 경우: route_set 내 적합한 출력 VC가
      // 하나라도 있는지, 출력 버퍼에 공간이 있는지 확인.
      if(_spec_check_elig) {
	
	int vc_start;
	int vc_end;
	
	if(_noq && _noq_next_output_port[input][vc] >= 0) {
	  assert(!_routing_delay);
	  vc_start = _noq_next_vc_start[input][vc];
	  vc_end = _noq_next_vc_end[input][vc];
	} else {
	  vc_start = iset->vc_start;
	  vc_end = iset->vc_end;
	}
	assert(vc_start >= 0 && vc_start < _vcs);
	assert(vc_end >= 0 && vc_end < _vcs);
	assert(vc_end >= vc_start);
	
	for(int dest_vc = vc_start; dest_vc <= vc_end; ++dest_vc) {
	  assert((dest_vc >= 0) && (dest_vc < _vcs));
	  
	  if(dest_buf->IsAvailableFor(dest_vc) && ( _output_buffer_size==-1 || _output_buffer[dest_output].size()<(size_t)(_output_buffer_size))) {
	    elig = true;
	    // [한국어] _spec_check_cred가 켜진 경우: 크레딧까지 확인.
	    if(!_spec_check_cred || !dest_buf->IsFullFor(dest_vc)) {
	      cred = true;
	      break;
	    }
	  }
	}
      }
      
      if(_spec_check_elig && !elig) {
	if(f->watch) {
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		     << "  Output " << dest_output 
		     << " has no suitable VCs available." << endl;
	}
	iter->second.second = STALL_BUFFER_BUSY;
      } else if(_spec_check_cred && !cred) {
	if(f->watch) {
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		     << "  All suitable VCs at output " << dest_output 
		     << " are full." << endl;
	}
	iter->second.second = dest_buf->IsFull() ? STALL_BUFFER_FULL : STALL_BUFFER_RESERVED;
      } else {
	bool const requested = _SWAllocAddReq(input, vc, dest_output);
	watched |= requested && f->watch;
      }
    }
  }
  
  if(watched) {
    *gWatchOut << GetSimTime() << " | " << _sw_allocator->FullName() << " | ";
    _sw_allocator->PrintRequests(gWatchOut);
    if(_spec_sw_allocator) {
      *gWatchOut << GetSimTime() << " | " << _spec_sw_allocator->FullName() << " | ";
      _spec_sw_allocator->PrintRequests(gWatchOut);
    }
  }
  
  _sw_allocator->Allocate();
  if(_spec_sw_allocator)
    _spec_sw_allocator->Allocate();
  
  if(watched) {
    *gWatchOut << GetSimTime() << " | " << _sw_allocator->FullName() << " | ";
    _sw_allocator->PrintGrants(gWatchOut);
    if(_spec_sw_allocator) {
      *gWatchOut << GetSimTime() << " | " << _spec_sw_allocator->FullName() << " | ";
      _spec_sw_allocator->PrintGrants(gWatchOut);
    }
  }
  
  // [한국어] allocator 결과를 deque에 기록하고 완료 시각 설정.
  for(deque<pair<int, pair<pair<int, int>, int> > >::iterator iter = _sw_alloc_vcs.begin();
      iter != _sw_alloc_vcs.end();
      ++iter) {

    int const time = iter->first;
    if(time >= 0) {
      break;
    }
    iter->first = GetSimTime() + _sw_alloc_delay - 1;

    int const input = iter->second.first.first;
    assert((input >= 0) && (input < _inputs));
    int const vc = iter->second.first.second;
    assert((vc >= 0) && (vc < _vcs));

    if(iter->second.second < -1) {
      continue;
    }

    assert(iter->second.second == -1);

    Buffer const * const cur_buf = _buf[input];
    assert(!cur_buf->Empty(vc));
    assert((cur_buf->GetState(vc) == VC::active) || 
	   (_speculative && (cur_buf->GetState(vc) == VC::vc_alloc)));
    
    Flit const * const f = cur_buf->FrontFlit(vc);
    assert(f);
    assert(f->vc == vc);

    int const expanded_input = input * _input_speedup + vc % _input_speedup;

    int expanded_output = _sw_allocator->OutputAssigned(expanded_input);

    if(expanded_output >= 0) {
      assert((expanded_output % _output_speedup) == (input % _output_speedup));
      int const granted_vc = _sw_allocator->ReadRequest(expanded_input, expanded_output);
      if(granted_vc == vc) {
	if(f->watch) {
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		     << "Assigning output " << (expanded_output / _output_speedup)
		     << "." << (expanded_output % _output_speedup)
		     << " to VC " << vc
		     << " at input " << input
		     << "." << (vc % _input_speedup)
		     << "." << endl;
	}
	_sw_rr_offset[expanded_input] = (vc + _input_speedup) % _vcs;
	iter->second.second = expanded_output;
      } else {
	if(f->watch) {
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		     << "Switch allocation failed for VC " << vc
		     << " at input " << input
		     << ": Granted to VC " << granted_vc << "." << endl;
	}
	iter->second.second = STALL_CROSSBAR_CONFLICT;
      }
    } else if(_spec_sw_allocator) {
      // [한국어] 비투기적 allocator에서 그랜트를 받지 못했으면 투기적 allocator 확인.
      expanded_output = _spec_sw_allocator->OutputAssigned(expanded_input);
      if(expanded_output >= 0) {
	assert((expanded_output % _output_speedup) == (input % _output_speedup));
	// [한국어] _spec_mask_by_reqs: 비투기적 요청이 있는 출력은 투기적 그랜트 폐기.
	if(_spec_mask_by_reqs && 
	   _sw_allocator->OutputHasRequests(expanded_output)) {
	  if(f->watch) {
	    *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		       << "Discarding speculative grant for VC " << vc
		       << " at input " << input
		       << "." << (vc % _input_speedup)
		       << " because output " << (expanded_output / _output_speedup)
		       << "." << (expanded_output % _output_speedup)
		       << " has non-speculative requests." << endl;
	  }
	  iter->second.second = STALL_CROSSBAR_CONFLICT;
	} else if(!_spec_mask_by_reqs &&
		  (_sw_allocator->InputAssigned(expanded_output) >= 0)) {
	  // [한국어] 비투기적 그랜트가 이미 발행된 경우에도 투기적 그랜트 폐기.
	  if(f->watch) {
	    *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		       << "Discarding speculative grant for VC " << vc
		       << " at input " << input
		       << "." << (vc % _input_speedup)
		       << " because output " << (expanded_output / _output_speedup)
		       << "." << (expanded_output % _output_speedup)
		       << " has a non-speculative grant." << endl;
	  }
	  iter->second.second = STALL_CROSSBAR_CONFLICT;
	} else {
	  int const granted_vc = _spec_sw_allocator->ReadRequest(expanded_input, 
							   expanded_output);
	  if(granted_vc == vc) {
	    if(f->watch) {
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "Assigning output " << (expanded_output / _output_speedup)
			 << "." << (expanded_output % _output_speedup)
			 << " to VC " << vc
			 << " at input " << input
			 << "." << (vc % _input_speedup)
			 << "." << endl;
	    }
	    _sw_rr_offset[expanded_input] = (vc + _input_speedup) % _vcs;
	    iter->second.second = expanded_output;
	  } else {
	    if(f->watch) {
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "Switch allocation failed for VC " << vc
			 << " at input " << input
			 << ": Granted to VC " << granted_vc << "." << endl;
	    }
	    iter->second.second = STALL_CROSSBAR_CONFLICT;
	  }
	}
      } else {

	if(f->watch) {
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		     << "Switch allocation failed for VC " << vc
		     << " at input " << input
		     << ": No output granted." << endl;
	}
	
	iter->second.second = STALL_CROSSBAR_CONFLICT;

      }
    } else {
      
      if(f->watch) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		   << "Switch allocation failed for VC " << vc
		   << " at input " << input
		   << ": No output granted." << endl;
      }
      
      iter->second.second = STALL_CROSSBAR_CONFLICT;
      
    }
  }
  
  // [한국어] non-speculative이고 _sw_alloc_delay<=1이면 추가 검사 불필요.
  if(!_speculative && (_sw_alloc_delay <= 1)) {
    return;
  }

  // [한국어] speculative 모드이거나 SA 지연이 1보다 큰 경우, 이미 발급된 grant가
  // Update 직전에 무효해진 경우(홀드 충돌, misspeculation, 크레딧 부족) 폐기.
  for(deque<pair<int, pair<pair<int, int>, int> > >::iterator iter = _sw_alloc_vcs.begin();
      iter != _sw_alloc_vcs.end();
      ++iter) {

    int const time = iter->first;
    assert(time >= 0);
    if(GetSimTime() < time) {
      break;
    }

    assert(iter->second.second != -1);

    int const expanded_output = iter->second.second;
    
    if(expanded_output >= 0) {
      
      int const output = expanded_output / _output_speedup;
      assert((output >= 0) && (output < _outputs));
      
      BufferState const * const dest_buf = _next_buf[output];
      
      int const input = iter->second.first.first;
      assert((input >= 0) && (input < _inputs));
      assert((input % _output_speedup) == (expanded_output % _output_speedup));
      int const vc = iter->second.first.second;
      assert((vc >= 0) && (vc < _vcs));
      
      int const expanded_input = input * _input_speedup + vc % _input_speedup;
      assert(_switch_hold_vc[expanded_input] != vc);
      
      Buffer const * const cur_buf = _buf[input];
      assert(!cur_buf->Empty(vc));
      assert((cur_buf->GetState(vc) == VC::active) ||
	     (_speculative && (cur_buf->GetState(vc) == VC::vc_alloc)));
      
      Flit const * const f = cur_buf->FrontFlit(vc);
      assert(f);
      assert(f->vc == vc);

      // [한국어] 홀드 충돌: grant 직전에 홀드가 생긴 경우.
      if((_switch_hold_in[expanded_input] >= 0) ||
	 (_switch_hold_out[expanded_output] >= 0)) {
	if(f->watch) {
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		     << "Discarding grant from input " << input
		     << "." << (vc % _input_speedup)
		     << " to output " << output
		     << "." << (expanded_output % _output_speedup)
		     << " due to conflict with held connection at ";
	  if(_switch_hold_in[expanded_input] >= 0) {
	    *gWatchOut << "input";
	  }
	  if((_switch_hold_in[expanded_input] >= 0) && 
	     (_switch_hold_out[expanded_output] >= 0)) {
	    *gWatchOut << " and ";
	  }
	  if(_switch_hold_out[expanded_output] >= 0) {
	    *gWatchOut << "output";
	  }
	  *gWatchOut << "." << endl;
	}
	iter->second.second = STALL_CROSSBAR_CONFLICT;
      } else if(_speculative && (cur_buf->GetState(vc) == VC::vc_alloc)) {

	assert(f->head);

	if(_vc_allocator) { // separate VC and switch allocators

	  // [한국어] separate allocator: 투기적 SA grant가 실제 VA grant와 일치하는지 확인.
	  int const input_and_vc = 
	    _vc_shuffle_requests ? (vc*_inputs + input) : (input*_vcs + vc);
	  int const output_and_vc = _vc_allocator->OutputAssigned(input_and_vc);

	  if(output_and_vc < 0) {
	    if(f->watch) {
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "Discarding grant from input " << input
			 << "." << (vc % _input_speedup)
			 << " to output " << output
			 << "." << (expanded_output % _output_speedup)
			 << " due to misspeculation." << endl;
	    }
	    iter->second.second = -1; // stall is counted in VC allocation path!
	  } else if((output_and_vc / _vcs) != output) {
	    if(f->watch) {
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "Discarding grant from input " << input
			 << "." << (vc % _input_speedup)
			 << " to output " << output
			 << "." << (expanded_output % _output_speedup)
			 << " due to port mismatch between VC and switch allocator." << endl;
	    }
	    iter->second.second = STALL_BUFFER_CONFLICT; // count this case as if we had failed allocation
	  } else if(dest_buf->IsFullFor((output_and_vc % _vcs))) {
	    if(f->watch) {
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "Discarding grant from input " << input
			 << "." << (vc % _input_speedup)
			 << " to output " << output
			 << "." << (expanded_output % _output_speedup)
			 << " due to lack of credit." << endl;
	    }
	    iter->second.second = dest_buf->IsFull() ? STALL_BUFFER_FULL : STALL_BUFFER_RESERVED;
	  }

	} else { // VC allocation is piggybacked onto switch allocation

	  // [한국어] piggyback 모드: SA grant에 대해 실제 사용할 출력 VC가 있는지
	  // route_set 내에서 다시 확인.
	  OutputSet const * const route_set = cur_buf->GetRouteSet(vc);
	  assert(route_set);

	  set<OutputSet::sSetElement> const setlist = route_set->GetSet();

	  bool busy = true;
	  bool full = true;
	  bool reserved = false;

	  assert(!_noq || (setlist.size() == 1));

	  for(set<OutputSet::sSetElement>::const_iterator iset = setlist.begin();
	      iset != setlist.end();
	      ++iset) {
	    if(iset->output_port == output) {

	      int vc_start;
	      int vc_end;
	      
	      if(_noq && _noq_next_output_port[input][vc] >= 0) {
		assert(!_routing_delay);
		vc_start = _noq_next_vc_start[input][vc];
		vc_end = _noq_next_vc_end[input][vc];
	      } else {
		vc_start = iset->vc_start;
		vc_end = iset->vc_end;
	      }
	      assert(vc_start >= 0 && vc_start < _vcs);
	      assert(vc_end >= 0 && vc_end < _vcs);
	      assert(vc_end >= vc_start);
	      
	      for(int out_vc = vc_start; out_vc <= vc_end; ++out_vc) {
		assert((out_vc >= 0) && (out_vc < _vcs));
		if(dest_buf->IsAvailableFor(out_vc)) {
		  busy = false;
		  if(!dest_buf->IsFullFor(out_vc)) {
		    full = false;
		    break;
		  } else if(!dest_buf->IsFull()) {
		    reserved = true;
		  }
		}
	      }
	      if(!full) {
		break;
	      }
	    }
	  }

	  if(busy) {
	    if(f->watch) {
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "Discarding grant from input " << input
			 << "." << (vc % _input_speedup)
			 << " to output " << output
			 << "." << (expanded_output % _output_speedup)
			 << " because no suitable output VC for piggyback allocation is available." << endl;
	    }
	    iter->second.second = STALL_BUFFER_BUSY;
	  } else if(full) {
	    if(f->watch) {
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "Discarding grant from input " << input
			 << "." << (vc % _input_speedup)
			 << " to output " << output
			 << "." << (expanded_output % _output_speedup)
			 << " because all suitable output VCs for piggyback allocation are full." << endl;
	    }
	    iter->second.second = reserved ? STALL_BUFFER_RESERVED : STALL_BUFFER_FULL;
	  }

	}

      } else {
	// [한국어] non-speculative active 상태: 출력 포트 일치 및 크레딧 재확인.
	assert(cur_buf->GetOutputPort(vc) == output);
	
	int const match_vc = cur_buf->GetOutputVC(vc);
	assert((match_vc >= 0) && (match_vc < _vcs));

	if(dest_buf->IsFullFor(match_vc)) {
	  if(f->watch) {
	    *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		       << "  Discarding grant from input " << input
		       << "." << (vc % _input_speedup)
		       << " to output " << output
		       << "." << (expanded_output % _output_speedup)
		       << " due to lack of credit." << endl;
	  }
	  iter->second.second = dest_buf->IsFull() ? STALL_BUFFER_FULL : STALL_BUFFER_RESERVED;
	}
      }
    }
  }
}


/*
 * [한국어]
 * _SWAllocUpdate — SA 단계의 Update.
 *
 * _sw_alloc_vcs에서 완료 시각이 된 항목을 꺼내 그랜트를 확정한다.
 *   - expanded_output >= 0: flit을 전송. piggyback 모드에서는 여기서 출력 VC를
 *     직접 할당(TakeBuffer, SetOutput)한다. 이후 _crossbar_flits 등록,
 *     업스트림 크레딧 생성, 홀드 설정(비-tail flit이고 hold_switch_for_packet).
 *     tail flit 이후에는 다음 head flit을 routing/vc_alloc으로 전환.
 *   - expanded_output < 0: 실패 코드에 따라 통계 기록 후 _sw_alloc_vcs 재등록.
 */
void IQRouter::_SWAllocUpdate( )
{
  while(!_sw_alloc_vcs.empty()) {

    pair<int, pair<pair<int, int>, int> > const & item = _sw_alloc_vcs.front();

    int const time = item.first;
    if((time < 0) || (GetSimTime() < time)) {
      break;
    }
    assert(GetSimTime() == time);

    int const input = item.second.first.first;
    assert((input >= 0) && (input < _inputs));
    int const vc = item.second.first.second;
    assert((vc >= 0) && (vc < _vcs));
    
    Buffer * const cur_buf = _buf[input];
    assert(!cur_buf->Empty(vc));
    assert((cur_buf->GetState(vc) == VC::active) ||
	   (_speculative && (cur_buf->GetState(vc) == VC::vc_alloc)));
    
    Flit * const f = cur_buf->FrontFlit(vc);
    assert(f);
    assert(f->vc == vc);

    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		 << "Completed switch allocation for VC " << vc
		 << " at input " << input
		 << " (front: " << f->id
		 << ")." << endl;
    }
    
    int const expanded_output = item.second.second;
    
    if(expanded_output >= 0) {
      
      int const expanded_input = input * _input_speedup + vc % _input_speedup;
      assert(_switch_hold_vc[expanded_input] < 0);
      assert(_switch_hold_in[expanded_input] < 0);
      assert(_switch_hold_out[expanded_output] < 0);

      int const output = expanded_output / _output_speedup;
      assert((output >= 0) && (output < _outputs));

      BufferState * const dest_buf = _next_buf[output];

      int match_vc;

      // [한국어] piggyback VC allocator: SA와 동시에 출력 VC를 선택.
      if(!_vc_allocator && (cur_buf->GetState(vc) == VC::vc_alloc)) {

	assert(f->head);

	int const cl = f->cl;
	assert((cl >= 0) && (cl < _classes));

	int const vc_offset = _vc_rr_offset[output*_classes+cl];

	match_vc = -1;
	int match_prio = numeric_limits<int>::min();

	const OutputSet * route_set = cur_buf->GetRouteSet(vc);
	set<OutputSet::sSetElement> const setlist = route_set->GetSet();
	
	assert(!_noq || (setlist.size() == 1));
	
	for(set<OutputSet::sSetElement>::const_iterator iset = setlist.begin();
	    iset != setlist.end();
	    ++iset) {
	  if(iset->output_port == output) {

	    int vc_start;
	    int vc_end;
	    
	    if(_noq && _noq_next_output_port[input][vc] >= 0) {
	      assert(!_routing_delay);
	      vc_start = _noq_next_vc_start[input][vc];
	      vc_end = _noq_next_vc_end[input][vc];
	    } else {
	      vc_start = iset->vc_start;
	      vc_end = iset->vc_end;
	    }
	    assert(vc_start >= 0 && vc_start < _vcs);
	    assert(vc_end >= 0 && vc_end < _vcs);
	    assert(vc_end >= vc_start);

	    for(int out_vc = vc_start; out_vc <= vc_end; ++out_vc) {
	      assert((out_vc >= 0) && (out_vc < _vcs));
	      
	      int vc_prio = iset->pri;
	      // [한국어] 비어 있지 않은 VC의 우선순위를 최저로 낮춤.
	      if(_vc_prioritize_empty && !dest_buf->IsEmptyFor(out_vc)) {
		assert(vc_prio >= 0);
		vc_prio += numeric_limits<int>::min();
	      }

	      // FIXME: This check should probably be performed in Evaluate(), 
	      // not Update(), as the latter can cause the outcome to depend on 
	      // the order of evaluation!
	      // [한국어] 사용 가능하고 크레딧이 남은 VC 중 RR 우선순위가 가장 높은 것 선택.
	      if(dest_buf->IsAvailableFor(out_vc) && 
		 !dest_buf->IsFullFor(out_vc) &&
		 ((match_vc < 0) || 
		  RoundRobinArbiter::Supersedes(out_vc, vc_prio, 
						match_vc, match_prio, 
						vc_offset, _vcs))) {
		match_vc = out_vc;
		match_prio = vc_prio;
	      }
	    }	
	  }
	}
	assert(match_vc >= 0);

	if(f->watch) {
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		     << "  Allocating VC " << match_vc
		     << " at output " << output
		     << " via piggyback VC allocation." << endl;
	}

	cur_buf->SetState(vc, VC::active);
	cur_buf->SetOutput(vc, output, match_vc);
	dest_buf->TakeBuffer(match_vc, input*_vcs + vc);

	_vc_rr_offset[output*_classes+cl] = (match_vc + 1) % _vcs;

      } else {

	// [한국어] 별도 allocator 사용 시 이미 출력 VC가 할당되어 있어야 함.
	assert(cur_buf->GetOutputPort(vc) == output);

	match_vc = cur_buf->GetOutputVC(vc);

      }
      assert((match_vc >= 0) && (match_vc < _vcs));

      if(f->watch) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		   << "  Scheduling switch connection from input " << input
		   << "." << (vc % _input_speedup)
		   << " to output " << output
		   << "." << (expanded_output % _output_speedup)
		   << "." << endl;
      }

      cur_buf->RemoveFlit(vc);

#ifdef TRACK_FLOWS
      --_stored_flits[f->cl][input];
      if(f->tail) --_active_packets[f->cl][input];
#endif

      _bufferMonitor->read(input, f) ;

      f->hops++;
      f->vc = match_vc;

      // [한국어] lookahead 라우팅: head flit이면 다음 홉 la_route_set 갱신.
      if(!_routing_delay && f->head) {
	const FlitChannel * channel = _output_channels[output];
	const Router * router = channel->GetSink();
	if(router) {
	  if(_noq) {
	    if(f->watch) {
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "Updating lookahead routing information for flit " << f->id
			 << " (NOQ)." << endl;
	    }
	    int next_output_port = _noq_next_output_port[input][vc];
	    assert(next_output_port >= 0);
	    _noq_next_output_port[input][vc] = -1;
	    int next_vc_start = _noq_next_vc_start[input][vc];
	    assert(next_vc_start >= 0 && next_vc_start < _vcs);
	    _noq_next_vc_start[input][vc] = -1;
	    int next_vc_end = _noq_next_vc_end[input][vc];
	    assert(next_vc_end >= 0 && next_vc_end < _vcs);
	    _noq_next_vc_end[input][vc] = -1;
	    f->la_route_set.Clear();
	    f->la_route_set.AddRange(next_output_port, next_vc_start, next_vc_end);
	  } else {
	    if(f->watch) {
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "Updating lookahead routing information for flit " << f->id
			 << "." << endl;
	    }
	    int in_channel = channel->GetSinkPort();
	    _rf(router, f, in_channel, &f->la_route_set, false);
	  }
	} else {
	  f->la_route_set.Clear();
	}
      }

#ifdef TRACK_FLOWS
      ++_outstanding_credits[f->cl][output];
      _outstanding_classes[output][f->vc].push(f->cl);
#endif

      dest_buf->SendingFlit(f);

      _crossbar_flits.push_back(make_pair(-1, make_pair(f, make_pair(expanded_input, expanded_output))));

      // [한국어] 입력 포트에 대한 업스트림 크레딧 생성/갱신.
      if(_out_queue_credits.count(input) == 0) {
	_out_queue_credits.insert(make_pair(input, Credit::New()));
      }
      _out_queue_credits.find(input)->second->vc.insert(vc);

      // [한국어] 버퍼가 비었으면 tail인 경우 idle로, 아니면 tail이 아닌 경우
      // 홀드 설정 또는 SA 재등록.
      if(cur_buf->Empty(vc)) {
	if(f->tail) {
	  cur_buf->SetState(vc, VC::idle);
	}
      } else {
	Flit * const nf = cur_buf->FrontFlit(vc);
	assert(nf);
	assert(nf->vc == vc);
	if(f->tail) {
	  assert(nf->head);
	  // [한국어] 다음 패킷 head flit이 도착한 상태. RC 또는 VC alloc으로 전환.
	  if(_routing_delay) {
	    cur_buf->SetState(vc, VC::routing);
	    _route_vcs.push_back(make_pair(-1, item.second.first));
	  } else {
	    if(nf->watch) {
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "Using precomputed lookahead routing information for VC " << vc
			 << " at input " << input
			 << " (front: " << nf->id
			 << ")." << endl;
	    }
	    cur_buf->SetRouteSet(vc, &nf->la_route_set);
	    cur_buf->SetState(vc, VC::vc_alloc);
	    if(_speculative) {
	      _sw_alloc_vcs.push_back(make_pair(-1, make_pair(item.second.first,
							      -1)));
	    }
	    if(_vc_allocator) {
	      _vc_alloc_vcs.push_back(make_pair(-1, make_pair(item.second.first,
							      -1)));
	    }
	    if(_noq) {
	      _UpdateNOQ(input, vc, nf);
	    }
	  }
	} else {
	  // [한국어] 동일 패킷의 다음 flit이 남았으면 홀드 설정 또는 SA 재등록.
	  if(_hold_switch_for_packet) {
	    if(f->watch) {
	      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
			 << "Setting up switch hold for VC " << vc
			 << " at input " << input
			 << "." << (expanded_input % _input_speedup)
			 << " to output " << output
			 << "." << (expanded_output % _output_speedup)
			 << "." << endl;
	    }
	    _switch_hold_vc[expanded_input] = vc;
	    _switch_hold_in[expanded_input] = expanded_output;
	    _switch_hold_out[expanded_output] = expanded_input;
	    _sw_hold_vcs.push_back(make_pair(-1, make_pair(item.second.first,
							     -1)));
	  } else {
	    _sw_alloc_vcs.push_back(make_pair(-1, make_pair(item.second.first,
							    -1)));
	  }
	}
      }
    } else {
      if(f->watch) {
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		   << "  No output port allocated." << endl;
      }

#ifdef TRACK_STALLS
      assert((expanded_output == -1) || // for stalls that are accounted for in VC allocation path
	     (expanded_output == STALL_BUFFER_BUSY) ||
	     (expanded_output == STALL_BUFFER_CONFLICT) ||
	     (expanded_output == STALL_BUFFER_FULL) ||
	     (expanded_output == STALL_BUFFER_RESERVED) ||
	     (expanded_output == STALL_CROSSBAR_CONFLICT));
      if(expanded_output == STALL_BUFFER_BUSY) {
	++_buffer_busy_stalls[f->cl];
      } else if(expanded_output == STALL_BUFFER_CONFLICT) {
	++_buffer_conflict_stalls[f->cl];
      } else if(expanded_output == STALL_BUFFER_FULL) {
	++_buffer_full_stalls[f->cl];
      } else if(expanded_output == STALL_BUFFER_RESERVED) {
	++_buffer_reserved_stalls[f->cl];
      } else if(expanded_output == STALL_CROSSBAR_CONFLICT) {
	++_crossbar_conflict_stalls[f->cl];
      }
#endif

      // [한국어] SA 실패: 다음 사이클에 다시 시도하기 위해 deque 뒤로 재등록.
      _sw_alloc_vcs.push_back(make_pair(-1, make_pair(item.second.first, -1)));
    }
    _sw_alloc_vcs.pop_front();
  }
}


//------------------------------------------------------------------------------
// switch traversal
//------------------------------------------------------------------------------

/*
 * [한국어]
 * _SwitchEvaluate — ST(Switch Traversal) 단계의 Evaluate.
 *
 * _crossbar_flits에 first < 0인 항목들에 대해 완료 시각을
 * GetSimTime() + _crossbar_delay - 1로 설정한다.
 */
void IQRouter::_SwitchEvaluate( )
{
  for(deque<pair<int, pair<Flit *, pair<int, int> > > >::iterator iter = _crossbar_flits.begin();
      iter != _crossbar_flits.end();
      ++iter) {
    
    int const time = iter->first;
    if(time >= 0) {
      break;
    }
    iter->first = GetSimTime() + _crossbar_delay - 1;

    Flit const * const f = iter->second.first;
    assert(f);

    int const expanded_input = iter->second.second.first;
    int const expanded_output = iter->second.second.second;
      
    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		 << "Beginning crossbar traversal for flit " << f->id
		 << " from input " << (expanded_input / _input_speedup)
		 << "." << (expanded_input % _input_speedup)
		 << " to output " << (expanded_output / _output_speedup)
		 << "." << (expanded_output % _output_speedup)
		 << "." << endl;
    }
  }
}

/*
 * [한국어]
 * _SwitchUpdate — ST 단계의 Update.
 *
 * 완료 시각이 된 _crossbar_flits 항목을 꺼내 _switchMonitor->traversal()로
 * 기록하고, 해당 flit을 _output_buffer[output]로 푸시한다.
 * 출력 버퍼 크기는 output_buffer_size로 제한되지만, flight 중인 flit을
 * 고려하여 최대 크기에 _crossbar_delay*_output_speedup + (_output_speedup-1)
 * 만큼의 여유를 둔다.
 */
void IQRouter::_SwitchUpdate( )
{
  while(!_crossbar_flits.empty()) {

    pair<int, pair<Flit *, pair<int, int> > > const & item = _crossbar_flits.front();

    int const time = item.first;
    if((time < 0) || (GetSimTime() < time)) {
      break;
    }
    assert(GetSimTime() == time);

    Flit * const f = item.second.first;
    assert(f);

    int const expanded_input = item.second.second.first;
    int const input = expanded_input / _input_speedup;
    assert((input >= 0) && (input < _inputs));
    int const expanded_output = item.second.second.second;
    int const output = expanded_output / _output_speedup;
    assert((output >= 0) && (output < _outputs));

    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		 << "Completed crossbar traversal for flit " << f->id
		 << " from input " << input
		 << "." << (expanded_input % _input_speedup)
		 << " to output " << output
		 << "." << (expanded_output % _output_speedup)
		 << "." << endl;
    }
    _switchMonitor->traversal(input, output, f) ;

    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		 << "Buffering flit " << f->id
		 << " at output " << output
		 << "." << endl;
    }
    _output_buffer[output].push(f);
    //the output buffer size isn't precise due to flits in flight
    //but there is a maximum bound based on output speed up and ST traversal
    assert(_output_buffer[output].size()<=(size_t)_output_buffer_size+ _crossbar_delay* _output_speedup+( _output_speedup-1) ||_output_buffer_size==-1);
    _crossbar_flits.pop_front();
  }
}


//------------------------------------------------------------------------------
// output queuing
//------------------------------------------------------------------------------

/*
 * [한국어]
 * _OutputQueuing — 이번 사이클에 _out_queue_credits에 모인 크레딧들을
 * _credit_buffer[input] 큐로 이동시킨다. 이후 _SendCredits()에서 실제 채널로 전송된다.
 */
void IQRouter::_OutputQueuing( )
{
  for(map<int, Credit *>::const_iterator iter = _out_queue_credits.begin();
      iter != _out_queue_credits.end();
      ++iter) {

    int const input = iter->first;
    assert((input >= 0) && (input < _inputs));

    Credit * const c = iter->second;
    assert(c);
    assert(!c->vc.empty());

    _credit_buffer[input].push(c);
  }
  _out_queue_credits.clear();
}

//------------------------------------------------------------------------------
// write outputs
//------------------------------------------------------------------------------

/*
 * [한국어]
 * _SendFlits — 각 출력 포트의 _output_buffer에서 flit 하나를 꺼내
 * 해당 _output_channels[output]로 전송한다(LT: Link Traversal).
 * gTrace가 true이면 디버그용 "Stop Mark"를 출력한다.
 */
void IQRouter::_SendFlits( )
{
  for ( int output = 0; output < _outputs; ++output ) {
    if ( !_output_buffer[output].empty( ) ) {
      Flit * const f = _output_buffer[output].front( );
      assert(f);
      _output_buffer[output].pop( );

#ifdef TRACK_FLOWS
      ++_sent_flits[f->cl][output];
#endif

      if(f->watch)
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		    << "Sending flit " << f->id
		    << " to channel at output " << output
		    << "." << endl;
      if(gTrace) {
	cout << "Outport " << output << endl << "Stop Mark" << endl;
      }
      _output_channels[output]->Send( f );
    }
  }
}

/*
 * [한국어]
 * _SendCredits — 각 입력 포트의 _credit_buffer에서 크레딧 하나를 꺼내
 * 업스트림으로 연결된 _input_credits[input] 채널로 전송한다.
 */
void IQRouter::_SendCredits( )
{
  for ( int input = 0; input < _inputs; ++input ) {
    if ( !_credit_buffer[input].empty( ) ) {
      Credit * const c = _credit_buffer[input].front( );
      assert(c);
      _credit_buffer[input].pop( );
      _input_credits[input]->Send( c );
    }
  }
}


//------------------------------------------------------------------------------
// misc.
//------------------------------------------------------------------------------

/*
 * [한국어]
 * Display — 모든 입력 버퍼(_buf[input])의 상태를 주어진 스트림에 출력한다.
 * 디버깅/검사용.
 */
void IQRouter::Display( ostream & os ) const
{
  for ( int input = 0; input < _inputs; ++input ) {
    _buf[input]->Display( os );
  }
}

/*
 * [한국어]
 * GetUsedCredit — 지정한 출력 포트 o의 다운스트림 버퍼에서 사용 중인 크레딧 수를
 * 반환한다. icnt_wrapper의 GetUsedCredit()에서 호출된다.
 */
int IQRouter::GetUsedCredit(int o) const
{
  assert((o >= 0) && (o < _outputs));
  BufferState const * const dest_buf = _next_buf[o];
  return dest_buf->Occupancy();
}

/*
 * [한국어]
 * GetBufferOccupancy — 지정한 입력 포트 i의 모든 VC에 쌓인 flit 수를 반환한다.
 */
int IQRouter::GetBufferOccupancy(int i) const {
  assert(i >= 0 && i < _inputs);
  return _buf[i]->GetOccupancy();
}

#ifdef TRACK_BUFFERS
/*
 * [한국어]
 * GetUsedCreditForClass — TRACK_BUFFERS 컴파일 옵션 사용 시,
 * 출력 포트와 트래픽 클래스별 사용 크레딧 수를 반환한다.
 */
int IQRouter::GetUsedCreditForClass(int output, int cl) const
{
  assert((output >= 0) && (output < _outputs));
  BufferState const * const dest_buf = _next_buf[output];
  return dest_buf->OccupancyForClass(cl);
}

/*
 * [한국어]
 * GetBufferOccupancyForClass — TRACK_BUFFERS 컴파일 옵션 사용 시,
 * 입력 포트와 트래픽 클래스별 버퍼 점유 flit 수를 반환한다.
 */
int IQRouter::GetBufferOccupancyForClass(int input, int cl) const
{
  assert((input >= 0) && (input < _inputs));
  return _buf[input]->GetOccupancyForClass(cl);
}
#endif

/*
 * [한국어]
 * UsedCredits — 모든 (출력 포트 × VC) 조합의 사용 중인 크레딧 수를
 * _outputs*_vcs 길이의 벡터로 반환한다. 인덱스 o*_vcs+v에 포트 o, VC v의 값.
 */
vector<int> IQRouter::UsedCredits() const
{
  vector<int> result(_outputs*_vcs);
  for(int o = 0; o < _outputs; ++o) {
    for(int v = 0; v < _vcs; ++v) {
      result[o*_vcs+v] = _next_buf[o]->OccupancyFor(v);
    }
  }
  return result;
}

/*
 * [한국어]
 * FreeCredits — 모든 (출력 포트 × VC) 조합의 사용 가능한(남은) 크레딧 수를 반환.
 */
vector<int> IQRouter::FreeCredits() const
{
  vector<int> result(_outputs*_vcs);
  for(int o = 0; o < _outputs; ++o) {
    for(int v = 0; v < _vcs; ++v) {
      result[o*_vcs+v] = _next_buf[o]->AvailableFor(v);
    }
  }
  return result;
}

/*
 * [한국어]
 * MaxCredits — 모든 (출력 포트 × VC) 조합의 최대 크레딧 수(버퍼 용량)를 반환.
 */
vector<int> IQRouter::MaxCredits() const
{
  vector<int> result(_outputs*_vcs);
  for(int o = 0; o < _outputs; ++o) {
    for(int v = 0; v < _vcs; ++v) {
      result[o*_vcs+v] = _next_buf[o]->LimitFor(v);
    }
  }
  return result;
}

/*
 * [한국어]
 * _UpdateNOQ — NOQ(Next-Output Queuing) 룩어헤드 라우팅 정보를 계산한다.
 *
 * flit의 la_route_set은 이번 라우터의 출력 포트/VC 범위를 담고 있다.
 * 여기서는 그 출력 포트로 연결된 다음 라우터에서 _rf()를 다시 호출하여
 * 다음 홉의 출력 포트와 VC 범위를 미리 계산하고, _noq_next_* 배열에 저장한다.
 * 이 정보는 _SWHoldUpdate/_SWAllocUpdate에서 head flit이 크로스바를 통과할 때
 * flit->la_route_set에 기록되어 다음 라우터가 RC 지연 없이 VA를 시작할 수 있게 한다.
 *
 * 전제조건: _routing_delay == 0, f는 head flit, la_route_set 크기는 1.
 */
void IQRouter::_UpdateNOQ(int input, int vc, Flit const * f) {
  assert(!_routing_delay);
  assert(f);
  assert(f->vc == vc);
  assert(f->head);
  set<OutputSet::sSetElement> sl = f->la_route_set.GetSet();
  assert(sl.size() == 1);
  int out_port = sl.begin()->output_port;
  const FlitChannel * channel = _output_channels[out_port];
  const Router * router = channel->GetSink();
  if(router) {
    int in_channel = channel->GetSinkPort();
    OutputSet nos;
    _rf(router, f, in_channel, &nos, false);
    sl = nos.GetSet();
    assert(sl.size() == 1);
    OutputSet::sSetElement const & se = *sl.begin();
    int next_output_port = se.output_port;
    assert(next_output_port >= 0);
    assert(_noq_next_output_port[input][vc] < 0);
    _noq_next_output_port[input][vc] = next_output_port;
    // [한국어] 다음 홉의 VC 범위를 다음 라우터의 출력 포트 수로 균등 분할.
    int next_vc_count = (se.vc_end - se.vc_start + 1) / router->NumOutputs();
    int next_vc_start = se.vc_start + next_output_port * next_vc_count;
    assert(next_vc_start >= 0 && next_vc_start < _vcs);
    assert(_noq_next_vc_start[input][vc] < 0);
    _noq_next_vc_start[input][vc] = next_vc_start;
    int next_vc_end = se.vc_start + (next_output_port + 1) * next_vc_count - 1;
    assert(next_vc_end >= 0 && next_vc_end < _vcs);
    assert(_noq_next_vc_end[input][vc] < 0);
    _noq_next_vc_end[input][vc] = next_vc_end;
    assert(next_vc_start <= next_vc_end);
    if(f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		 << "Computing lookahead routing information for flit " << f->id
		 << " (NOQ)." << endl;
    }
  }
}
