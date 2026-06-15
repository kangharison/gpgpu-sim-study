// $Id: event_router.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] EventRouter 클래스 구현 (event_router.cpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 event_router.hpp에 선언된 EventRouter 및 EventNextVCState 클래스의
 * 멤버 함수를 모두 구현한다. EventRouter는 BookSim2 NoC 시뮬레이터의 이벤트 기반
 * 라우터로, IQRouter처럼 매 사이클 모든 VC를 폴링하지 않고 플릿/크레딧이 실제로
 * 이동할 때만 이벤트를 생성·처리하여 시뮬레이션 속도를 높인다.
 * 입력 플릿은 우선 VC 버퍼에 저장된 후 tArrivalEvent를 거쳐 출력 VC 소유권을
 * 획득하고, 크레딧이 확인되면 tTransportEvent를 생성해 크로스바를 통과한다.
 * GPGPU-Sim에서는 gpgpusim.config의 `router event` 옵션으로 활성화된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 타이밍 모델의 NoC(intersim2) 서브시스템 낸 라우터 계층에 위치한다.
 *
 *   gpu-sim.cc::cycle()
 *     → icnt_wrapper.cc (icnt_push / icnt_pop)
 *       → Network::ReadInputs() / Evaluate() / WriteOutputs()
 *         → [EventRouter::ReadInputs()]
 *         → [EventRouter::_InternalStep()]
 *           → _IncomingFlits() → _arrival_pipe → _ArrivalRequests() → _ArrivalArb()
 *           → _TransportRequests() → _TransportArb()
 *           → _crossbar_pipe / _credit_pipe → _OutputQueuing()
 *         → [EventRouter::WriteOutputs()]
 *
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 단일 시뮬레이션 스레드.
 * 매 사이클 ReadInputs → _InternalStep → WriteOutputs 순서로 호출된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - event_router.hpp : EventRouter / EventNextVCState 클래스 선언
 *   - stats.hpp        : Stats 통계 객체 (이 파일에서는 직접 사용하지 않으나
 *                        헤더 의존성으로 포함)
 *   - globals.hpp      : gRoutingFunctionMap, gWatchOut 등 전역 객체
 *   - module.hpp       : Module 계층 및 FullName(), Error()
 *   - buffer.hpp       : 입력 VC 버퍼 (Buffer 클래스)
 *   - prio_arb.hpp     : PriorityArbiter (arrival/transport 중재)
 *   - pipefifo.hpp     : PipelineFIFO (crossbar/credit/arrival 지연 모델)
 *   - flit.hpp / credit.hpp : Flit, Credit 객체
 *
 * 데이터 흐름:
 *   입력 채널 → _input_buffer → _buf[VC 버퍼]
 *     → tArrivalEvent → _arrival_pipe → _arrival_queue
 *     → _arrival_arbiter → _output_state[출력 VC] (소유권/크레딧)
 *     → tTransportEvent → _transport_queue → _transport_arbiter
 *     → _crossbar_pipe → _output_buffer → 출력 채널
 *   크레딧 역방향:
 *     다운스트림 → _output_credits → _out_cred_buffer
 *     → _output_state 크레딧 증가 → _credit_pipe → _in_cred_buffer
 *     → _input_credits → 업스트림
 *
 * === 주요 함수/구조체 요약 ===
 * EventRouter()       : num_vcs, vct, routing_function, topology 등 설정 읽고
 *                       버퍼/VC 상태/중재기/파이프라인/큐를 초기화
 * ~EventRouter()      : 동적 할당한 버퍼/상태/중재기/파이프라인을 해제
 * ReadInputs()        : _ReceiveFlits() + _ReceiveCredits() — 채널 데이터 수신
 * _InternalStep()     : 2단계 이벤트 파이프라인 전체 실행 (arrival + transport)
 * WriteOutputs()      : _SendFlits() + _SendCredits() — 채널 데이터 송신
 * _ReceiveFlits()     : 입력 채널에서 플릿을 읽어 _input_buffer에 저장
 * _ReceiveCredits()   : 출력 크레딧 채널에서 크레딧을 읽어 _out_cred_buffer에 저장
 * _ProcessWaiting()   : tail 크레딧 수신 후 출력 VC 대기열의 다음 연결로 소유권 이전
 * _IncomingFlits()    : _input_buffer → _buf VC 버퍼 삽입 + 라우팅 + arrival 이벤트 생성
 * _ArrivalRequests()  : _arrival_pipe에서 이벤트를 꺼내 중재기에 요청 등록
 * _SendTransport()    : 크레딧 유무에 따라 transport 이벤트 생성 또는 presence 증가
 * _ArrivalArb()       : 출력 포트별 도착 이벤트 중재 + 크레딧 처리
 * _TransportRequests(): transport_queue front를 입력 중재기에 요청 등록
 * _TransportArb()     : 입력 포트별 transport 중재 + 실제 플릿 크로스바 주입
 * _OutputQueuing()    : crossbar_pipe/credit_pipe 완료 항목을 출력/크레딧 버퍼로 이동
 * _SendFlits()        : _output_buffer의 플릿을 출력 채널로 전송
 * _SendCredits()      : _in_cred_buffer의 크레딧을 입력 크레딧 채널로 전송
 * EventNextVCState    : 출력 VC별 상태(idle/busy/tail_pending), 크레딧, 대기열 관리
 *
 * 설정 연동 (gpgpusim.config):
 *   - `router event`          : 이 라우터 선택
 *   - `num_vcs`               : VC 개수
 *   - `vct`                   : Virtual Cut-Through 모드 (1=활성, 0=store-and-forward)
 *   - `routing_function`      : 라우팅 함수 이름
 *   - `topology`              : 토폴로지 이름 (routing_function + "_" + topology로 맵 키 생성)
 *   - `vc_buf_size`           : VC 버퍼/다운스트림 버퍼 슬롯 수
 *   - `crossbar_delay`        : 크로스바 파이프라인 지연
 *   - `credit_delay`          : 크레딧 반환 파이프라인 지연
 */

#include <string>       // [한국어] std::string — 라우터/모듈 이름, 설정 키 조합에 사용
#include <sstream>      // [한국어] std::ostringstream — 동적 모듈 이름 생성에 사용
#include <iostream>     // [한국어] std::cout — 디버그 출력 (watch 플래그, Error 메시지)
#include <cstdlib>      // [한국어] abort/exit — Error() 호출 경로에서 간접적으로 필요
#include <cassert>      // [한국어] assert() — 인덱스 범위 검증

#include "event_router.hpp" // [한국어] EventRouter / EventNextVCState 클래스 선언
#include "stats.hpp"        // [한국어] Stats 통계 객체 헤더 (간접 포함)
#include "globals.hpp"      // [한국어] gRoutingFunctionMap, gWatchOut 등 전역 객체

/*
 * [한국어]
 * EventRouter::EventRouter - 이벤트 드리븐 라우터 생성자
 *
 * @param config  : BookSim2 설정 객체 — num_vcs, vct, routing_function, topology,
 *                  vc_buf_size, crossbar_delay, credit_delay 등을 포함
 * @param parent  : 부모 Module 포인터 (Network 객체)
 * @param name    : 이 라우터의 식별 이름 문자열 (예: "router_5")
 * @param id      : 네트워크 내 라우터 고유 ID (0-based)
 * @param inputs  : 입력 포트 수
 * @param outputs : 출력 포트 수
 * @return        : 없음 (생성자)
 *
 * 초기화 단계:
 *   1. Router 기반 클래스 생성자 호출 — 입출력 채널/크레딧 채널 배열, _inputs,
 *      _outputs, _crossbar_delay, _credit_delay 등 초기화
 *   2. num_vcs → _vcs (VC 총 개수)
 *   3. vct → _vct (Virtual Cut-Through 모드 플래그)
 *   4. gRoutingFunctionMap에서 "routing_function_topology" 키로 _rf 조회
 *   5. 입력 포트별 Buffer(_buf) 및 _active 배열 할당
 *   6. 출력 포트별 EventNextVCState(_output_state) 및 arrival_arbiter 할당
 *   7. 입력 포트별 transport_arbiter 할당
 *   8. crossbar_pipe, credit_pipe, arrival_pipe 생성
 *   9. 입력/출력 버퍼 큐, 크레딧 버퍼 큐, 이벤트 큐 크기 조정
 *  10. _transport_free (true), _transport_match (-1) 초기화
 *
 * 호출 체인: Network 생성자/AddRouter → Router::NewRouter("event") → [이 함수]
 */
EventRouter::EventRouter( const Configuration& config,
			  Module *parent, const string & name, int id,
			  int inputs, int outputs )
  : Router( config,	    // [한국어] Router 기반 클래스 생성자 호출 — 공통 라우터 필드 초기화
	    parent, name,   // [한국어] 모듈 계층 등록 및 이름 설정
	    id,		    // [한국어] 라우터 고유 ID
	    inputs, outputs ) // [한국어] 입력/출력 포트 수
{
  ostringstream module_name;
  // [한국어] 자식 모듈 이름을 동적으로 조합하기 위한 스트림 (예: "buf_0", "output1_vc_state")
  
  _vcs            = config.GetInt( "num_vcs" );
  // [한국어] VC(Virtual Channel) 총 개수 — gpgpusim.config의 num_vcs 옵션
  //   VC는 출력 포트별로 다중화된 버퍼/흐름제어 단위이며, deadlock-free 라우팅과
  //   throughput 향상을 위해 사용된다.

  // Cut-through mode --- packets are not broken
  // up and input buffers are assumed to be 
  // expressed in units of maximum size packets.
  // [한국어] VCT(Virtual Cut-Through) 모드 주석.
  //   _vct == 1이면 패킷이 플릿 단위로 쪼개지지 않고 한 번에 전송된다고 가정하며,
  //   입력 버퍼 용량도 최대 패킷 크기 단위로 표현된다.
  //   이 모드에서는 head 플릿만 arrival 이벤트를 생성하고, tail 처리가 단순화된다.
  
  _vct            = config.GetInt( "vct" );
  // [한국어] VCT 모드 플래그 — gpgpusim.config의 vct 옵션 (1=VCT, 0=store-and-forward)

  // Routing
  // [한국어] 라우팅 함수 조회: 설정 파일의 routing_function과 topology를 조합해
  //   전역 라우팅 함수 맵(gRoutingFunctionMap)에서 실제 함수 포인터를 검색한다.

  string rf = config.GetStr("routing_function") + "_" + config.GetStr("topology");
  // [한국어] 라우팅 함수 맵 키 조합 (예: "dim_order_mesh")
  map<string, tRoutingFunction>::iterator rf_iter = gRoutingFunctionMap.find(rf);
  // [한국어] 전역 맵에서 키로 함수 포인터 탐색
  if(rf_iter == gRoutingFunctionMap.end()) {
    // [한국어] 설정된 routing_function + topology 조합이 맵에 없으면 시뮬레이션 중단
    Error("Invalid routing function: " + rf);
  }
  _rf = rf_iter->second;
  // [한국어] 검색된 라우팅 함수 포인터 저장 — _IncomingFlits()에서 head 플릿 라우팅 시 사용

  // Alloc VC's
  // [한국어] 입력 포트별 VC 버퍼(_buf) 및 활성 플래그(_active) 할당

  _buf.resize(_inputs);
  // [한국어] 입력 포트 수만큼 Buffer 포인터 배열 크기 확장
  _active.resize(_inputs);
  // [한국어] 입력 포트 수만큼 활성 플래그 배열 크기 확장

  for ( int i = 0; i < _inputs; ++i ) {
    // [한국어] 각 입력 포트에 대해 Buffer 객체 생성
    module_name << "buf_" << i;
    // [한국어] Buffer 모듈 이름 생성 (예: "buf_0")
    _buf[i] = new Buffer( config, _outputs, this, module_name.str( ) );
    // [한국어] i번 입력 포트의 VC 버퍼 생성 — config, 출력 포트 수, 부모, 이름 전달
    module_name.seekp( 0, ios::beg );
    // [한국어] ostringstream의 쓰기 위치를 처음으로 되돌려 다음 이름 생성 준비
    _active[i].resize(_vcs, false);
    // [한국어] i번 입력 포트의 VC별 활성 플래그를 false(비활성)로 초기화
  }

  // Alloc next VCs' state
  // [한국어] 출력 포트별 VC 상태 추적기(_output_state) 할당

  _output_state.resize(_outputs);
  // [한국어] 출력 포트 수만큼 EventNextVCState 포인터 배열 크기 확장

  for ( int o = 0; o < _outputs; ++o ) {
    // [한국어] 각 출력 포트에 대해 EventNextVCState 객체 생성
    module_name << "output" << o << "_vc_state";
    // [한국어] 상태 모듈 이름 생성 (예: "output0_vc_state")
    _output_state[o] = new EventNextVCState(config, this, module_name.str());
    // [한국어] o번 출력 포트의 VC 상태 추적기 생성 — 다운스트림 버퍼 크레딧/대기열 관리
    module_name.seekp( 0, ios::beg );
    // [한국어] 이름 스트림 리셋
  }

  // Alloc arbiters
  // [한국어] 출력 포트별 arrival 중재기와 입력 포트별 transport 중재기 할당

  _arrival_arbiter.resize(_outputs);
  // [한국어] 출력 포트 수만큼 arrival 중재기 포인터 배열 확장

  for ( int o = 0; o < _outputs; ++o ) {
    // [한국어] 각 출력 포트에 대해 arrival 중재기 생성
    module_name << "arrival_arb_output" << o;
    // [한국어] 중재기 모듈 이름 생성 (예: "arrival_arb_output0")
    _arrival_arbiter[o] = 
      new PriorityArbiter( config, this, module_name.str( ), _inputs );
    // [한국어] o번 출력 포트의 arrival 중재기 생성 — _inputs개의 입력 요청을 중재
    module_name.seekp( 0, ios::beg );
    // [한국어] 이름 스트림 리셋
  }

  _transport_arbiter.resize(_inputs);
  // [한국어] 입력 포트 수만큼 transport 중재기 포인터 배열 확장

  for ( int i = 0; i < _inputs; ++i ) {
    // [한국어] 각 입력 포트에 대해 transport 중재기 생성
    module_name << "transport_arb_input" << i;
    // [한국어] 중재기 모듈 이름 생성 (예: "transport_arb_input0")
    _transport_arbiter[i] = 
      new PriorityArbiter( config, this, module_name.str( ), _outputs );
    // [한국어] i번 입력 포트의 transport 중재기 생성 — _outputs개의 출력 요청을 중재
    module_name.seekp( 0, ios::beg );
    // [한국어] 이름 스트림 리셋
  }

  // Alloc pipelines (to simulate processing/transmission delays)
  // [한국어] 크로스바/크레딧/도착 파이프라인 할당 — 각 단계의 지연을 사이클 단위로 모델링

  _crossbar_pipe = 
    new PipelineFIFO<Flit>( this, "crossbar_pipeline", _outputs, 
			    _crossbar_delay );
  // [한국어] 크로스바 파이프라인 생성: _outputs개 채널, _crossbar_delay 사이클 지연
  //   _TransportArb()에서 Write(f, output)으로 플릿을 넣으면
  //   _crossbar_delay 사이클 후 _OutputQueuing()에서 Read(output)로 꺼낼 수 있다.

  _credit_pipe =
    new PipelineFIFO<Credit>( this, "credit_pipeline", _inputs,
			      _credit_delay );
  // [한국어] 크레딧 반환 파이프라인 생성: _inputs개 채널, _credit_delay 사이클 지연
  //   _TransportArb()에서 Write(c, input)으로 크레딧을 넣으면
  //   _credit_delay 사이클 후 _OutputQueuing()에서 Read(input)로 꺼낸다.

  _arrival_pipe =
    new PipelineFIFO<tArrivalEvent>( this, "arrival_pipeline", _inputs,
				     0 /* FIX THIS EVENTUALLY */ );
  // [한국어] 도착 이벤트 파이프라인 생성: _inputs개 채널, 현재 0 사이클 지연
  //   라우팅/디코딩 지연을 모델링하기 위한 파이프라인이지만 현재는 지연 없음.
  //   "FIX THIS EVENTUALLY" 주석은 향후 지연 추가를 위한 개발 메모.

  // Queues
  // [한국어] 입력/출력/크레딧 버퍼 큐 및 이벤트 큐 크기 조정

  _input_buffer.resize(_inputs); 
  // [한국어] 입력 포트별 플릿 수신 대기 큐 (ReadInputs 단계에서 채널로부터 수신)
  _output_buffer.resize(_outputs); 
  // [한국어] 출력 포트별 플릿 전송 대기 큐 (WriteOutputs 단계에서 채널로 전송)

  _in_cred_buffer.resize(_inputs); 
  // [한국어] 입력 포트별 크레딧 전송 대기 큐 (업스트림으로 반환)
  _out_cred_buffer.resize(_outputs);
  // [한국어] 출력 포트별 다운스트림 크레딧 수신 대기 큐

  _arrival_queue.resize(_inputs);
  // [한국어] 입력 포트별 arrival 이벤트 큐 (_arrival_pipe에서 나온 이벤트 보관)
  _transport_queue.resize(_outputs);
  // [한국어] 출력 포트별 transport 이벤트 큐 (_ArrivalArb에서 생성된 이벤트 보관)

  // Misc.
  // [한국어] transport 중재 관련 보조 변수 초기화

  _transport_free.resize(_inputs, true);
  // [한국어] 각 입력 포트의 transport 중재기가 새로운 중재를 수행할 수 있는지 여부
  //   VCT 모드에서 연속된 플릿 전송 시 false로 잠금
  _transport_match.resize(_inputs, -1);
  // [한국어] _transport_free가 false일 때 재사용하는 이전 중재 결과(출력 포트 인덱스)
}

/*
 * [한국어]
 * EventRouter::~EventRouter - 이벤트 드리븐 라우터 소멸자
 *
 * @return: 없음 (소멸자)
 *
 * 생성자에서 new로 할당한 모든 동적 객체를 해제한다.
 * 플릿(Flit*)과 크레딧(Credit*) 객체는 BookSim2의 풀 할당자(Flit::Free,
 * Credit::Free)에서 관리되므로 큐 남은 원소는 여기서 삭제하지 않는다.
 *
 * 호출 체인: Network 소멸자 → [이 함수]
 */
EventRouter::~EventRouter( )
{
  for ( int i = 0; i < _inputs; ++i ) {
    // [한국어] 모든 입력 포트의 Buffer 객체 삭제
    delete _buf[i];
  }

  for ( int o = 0; o < _outputs; ++o ) {
    // [한국어] 모든 출력 포트의 EventNextVCState 객체 삭제
    delete _output_state[o];
  }

  for ( int o = 0; o < _outputs; ++o ) {
    // [한국어] 모든 출력 포트의 arrival 중재기 삭제
    delete _arrival_arbiter[o];
  }

  for ( int i = 0; i < _inputs; ++i ) {
    // [한국어] 모든 입력 포트의 transport 중재기 삭제
    delete _transport_arbiter[i];
  }

  delete _crossbar_pipe;
  // [한국어] 크로스바 파이프라인 FIFO 삭제
  delete _credit_pipe;
  // [한국어] 크레딧 반환 파이프라인 FIFO 삭제
  delete _arrival_pipe;
  // [한국어] 도착 이벤트 파이프라인 FIFO 삭제
}
	  
/*
 * [한국어]
 * EventRouter::ReadInputs - 모든 입력 채널에서 플릿과 크레딧 수신
 *
 * @return: 없음
 *
 * Router 기반 클래스 인터페이스 구현. 매 사이클 Network::ReadInputs()에서
 * 모든 라우터에 대해 먼저 호출된다.
 * 실제 처리는 _ReceiveFlits()와 _ReceiveCredits()로 위임한다.
 *
 * 호출 체인: Network::ReadInputs() → [이 함수] → _ReceiveFlits(), _ReceiveCredits()
 */
void EventRouter::ReadInputs( )
{
  _ReceiveFlits( );
  // [한국어] 입력 채널에서 플릿을 받아 _input_buffer에 저장
  _ReceiveCredits( );
  // [한국어] 다운스트림으로부터 크레딧을 받아 _out_cred_buffer에 저장
}

/*
 * [한국어]
 * EventRouter::_InternalStep - 매 사이클 호출되는 라우터 낸부 파이프라인 실행
 *
 * @return: 없음
 *
 * Router 기반 클래스의 순수 가상 함수를 오버라이드.
 * 이벤트 드리븐 라우터의 전체 2단계 파이프라인을 순서대로 실행한다.
 *
 * Phase 1 (Arrival):
 *   1. _IncomingFlits(): _input_buffer의 플릿을 VC 버퍼에 넣고 arrival 이벤트 생성
 *   2. _arrival_pipe->Advance(): 라우팅/디코딩 지연 진행
 *   3. arrival_arbiter Clear(): 매 사이클 중재기 리셋
 *   4. _ArrivalRequests(input): arrival_pipe에서 나온 이벤트를 중재 요청으로 등록
 *   5. _ArrivalArb(output): 출력 포트별 도착 이벤트 중재 + 크레딧 처리
 *
 * Phase 2 (Transport):
 *   6. transport_arbiter Clear(): 매 사이클 중재기 리셋
 *   7. crossbar_pipe/credit_pipe WriteAll(0): 이번 사이클 삽입 전 파이프라인 초기화
 *   8. _TransportRequests(output): transport_queue front를 입력 중재기에 등록
 *   9. _TransportArb(input): 입력 포트별 transport 중재 + 실제 플릿 크로스바 주입
 *  10. crossbar_pipe/credit_pipe Advance(): 파이프라인 시간 진행
 *  11. _OutputQueuing(): 파이프라인 완료 항목을 출력/크레딧 버퍼로 이동
 *
 * 호출 체인: Network::Evaluate() → [이 함수]
 */
void EventRouter::_InternalStep( )
{
  // Receive incoming flits
  // [한국어] 입력 버퍼에 도착한 플릿을 VC 버퍼로 옮기고 arrival 이벤트 생성
  _IncomingFlits( );

  // The input pipe simulates routing delay
  // [한국어] arrival_pipe를 한 사이클 진행 — 라우팅/디코딩 지연 시뮬레이션
  _arrival_pipe->Advance( );

  // Clear output requests
  // [한국어] 출력 포트별 arrival 중재기를 리셋 (이전 사이클의 요청/승자 제거)
  for ( int output = 0; output < _outputs; ++output ) {
    _arrival_arbiter[output]->Clear( );
  }

  // Check input arrival queues and generate
  // requests for the outputs
  // [한국어] arrival_pipe에서 나온 이벤트를 _arrival_queue에 넣고
  //   해당 출력 포트의 arrival_arbiter에 요청 등록
  for ( int input = 0; input < _inputs; ++input ) {
    _ArrivalRequests( input );
  }

  // Arbitrate between requests at outputs
  // [한국어] 각 출력 포트에서 arrival 이벤트 중재 및 크레딧 처리
  for ( int output = 0; output < _outputs; ++output ) {
    _ArrivalArb( output );
  }

  // [한국어] 입력 포트별 transport 중재기 리셋
  for ( int input = 0; input < _inputs; ++input ) {
    _transport_arbiter[input]->Clear( );
  }

  // [한국어] 크로스바와 크레딧 파이프라인의 모든 슬롯을 NULL(0)로 초기화.
  //   이번 사이클에 Write()가 발생하는 슬롯만 나중에 덮어쓰인다.
  _crossbar_pipe->WriteAll( 0 );
  _credit_pipe->WriteAll( 0 );

  // Generate transport events and their
  // requests for the inputs
  // [한국어] transport_queue의 front 이벤트를 해당 입력 포트의 transport_arbiter에 등록
  for ( int output = 0; output < _outputs; ++output ) {
    _TransportRequests( output );
  }

  // Arbitrate between requests at inputs
  // [한국어] 입력 포트별 transport 이벤트 중재 및 실제 플릿 전송
  for ( int input = 0; input < _inputs; ++input ) {
    _TransportArb( input );
  }

  // [한국어] 크로스바/크레딧 파이프라인을 한 사이클 진행
  _crossbar_pipe->Advance( );
  _credit_pipe->Advance( );

  // [한국어] 파이프라인을 완전히 통과한 플릿과 크레딧을 출력/크레딧 버퍼로 이동
  _OutputQueuing( );
}

/*
 * [한국어]
 * EventRouter::WriteOutputs - 출력 채널로 플릿과 크레딧 전송
 *
 * @return: 없음
 *
 * Router 기반 클래스 인터페이스 구현. 매 사이클 Network::WriteOutputs()에서
 * 모든 라우터에 대해 마지막으로 호출된다.
 * _OutputQueuing()에서 _output_buffer/_in_cred_buffer로 모은 데이터를
 * 실제 채널로 전송한다.
 *
 * 호출 체인: Network::WriteOutputs() → [이 함수] → _SendFlits(), _SendCredits()
 */
void EventRouter::WriteOutputs( )
{
  _SendFlits( );
  // [한국어] _output_buffer의 플릿을 출력 채널로 전송
  _SendCredits( );
  // [한국어] _in_cred_buffer의 크레딧을 입력 크레딧 채널로 전송
}

/*
 * [한국어]
 * EventRouter::_ReceiveFlits - 모든 입력 채널에서 플릿 수신
 *
 * @return: 없음
 *
 * ReadInputs()의 하위 단계. 각 입력 채널(_input_channels[input])에서 Receive()를
 * 호출하여 이번 사이클에 도착한 플릿을 _input_buffer[input]에 임시 저장한다.
 * 실제 VC 버퍼 삽입과 arrival 이벤트 생성은 다음 _InternalStep()의
 * _IncomingFlits()가 수행한다.
 *
 * 호출 체인: ReadInputs() → [이 함수]
 */
void EventRouter::_ReceiveFlits( )
{
  Flit *f;
  // [한국어] 수신된 플릿 포인터

  for ( int input = 0; input < _inputs; ++input ) { 
    // [한국어] 모든 입력 포트를 순회
    f = _input_channels[input]->Receive();
    // [한국어] input번 입력 채널에서 플릿 수신 시도 — 없으면 NULL

    if ( f ) {
      // [한국어] 플릿이 도착한 경우에만 입력 버퍼에 저장
      _input_buffer[input].push( f );
    }
  }
}

/*
 * [한국어]
 * EventRouter::_ReceiveCredits - 모든 출력 채널로부터 다운스트림 크레딧 수신
 *
 * @return: 없음
 *
 * ReadInputs()의 하위 단계. 각 출력 크레딧 채널(_output_credits[output])에서
 * Receive()를 호출하여 다운스트림 라우터가 반환한 크레딧을
 * _out_cred_buffer[output]에 임시 저장한다.
 * 크레딧 처리(상태 갱신, transport 이벤트 생성)는 _ArrivalArb()가 담당한다.
 *
 * 호출 체인: ReadInputs() → [이 함수]
 */
void EventRouter::_ReceiveCredits( )
{
  Credit *c;
  // [한국어] 수신된 크레딧 포인터

  for ( int output = 0; output < _outputs; ++output ) {  
    // [한국어] 모든 출력 포트를 순회
    c = _output_credits[output]->Receive();
    // [한국어] output번 출력 방향 크레딧 채널에서 크레딧 수신 — 없으면 NULL

    if ( c ) {
      // [한국어] 크레딧이 도착한 경우 출력 크레딧 버퍼에 저장
      _out_cred_buffer[output].push( c );
    }
  }
}

/*
 * [한국어]
 * EventRouter::_ProcessWaiting - tail 크레딧 수신 후 대기 중인 연결 처리
 *
 * @param output : 처리할 출력 포트 인덱스
 * @param out_vc : tail 크레딧이 도착한 출력 VC 인덱스
 * @return       : 없음
 *
 * out_vc에서 tail 플릿이 전송되어 다운스트림에서 크레딧이 반환되면,
 * 이 VC는 더 이상 현재 패킷을 점유하지 않게 된다.
 * _waiting[out_vc]에 대기 중인 연결이 있으면 FIFO 순서로 다음 소유자를 꺼내
 * SetState(busy), SetInput(), SetInputVC()로 소유권을 이전하고,
 * 크레딧이 남아 있으면 즉시 tTransportEvent를 생성한다.
 * 대기 중인 연결이 없으면 VC를 idle 상태로 전환한다.
 *
 * 호출 체인: _ArrivalArb() → [이 함수]
 */
void EventRouter::_ProcessWaiting( int output, int out_vc )
{
  // out_vc just sent the transport event for out_vc, 
  // check if any events are queued on that vc.  if so,
  // generate another transport event and set the 
  // owner of the vc, otherwise set the vc to idle.
  // [한국어] out_vc에서 transport 이벤트가 처리된 뒤, 해당 VC에 대기 중인
  //   연결이 있는지 확인. 있으면 새 소유자를 설정하고 transport 이벤트를 생성,
  //   없으면 idle로 전환.

  int credits;
  // [한국어] out_vc의 현재 남은 크레딧 수
  
  tTransportEvent *tevt;
  // [한국어] 생성할 전송 이벤트 포인터

  EventNextVCState::tWaiting *w;
  // [한국어] 대기열에서 꺼낼 연결 정보 포인터

  if ( _output_state[output]->IsWaiting( out_vc ) ) {
    // [한국어] out_vc에 대기 중인 연결이 하나 이상 존재
		    
    // State remains as busy, but the waiting VC takes over
    // [한국어] 상태는 busy로 유지되며 소유자만 대기 중인 연결로 교첻
    w = _output_state[output]->PopWaiting( out_vc );
    // [한국어] out_vc 대기열의 맨 앞 항목을 꺼냄 (FIFO)
    
    _output_state[output]->SetState( out_vc, EventNextVCState::busy );
    // [한국어] out_vc를 busy 상태로 유지 (새 소유자가 점유)
    _output_state[output]->SetInput( out_vc, w->input );
    // [한국어] 새 소유자의 입력 포트 설정
    _output_state[output]->SetInputVC( out_vc, w->vc );
    // [한국어] 새 소유자의 입력 VC(src_vc) 설정

    if ( w->watch ) {
      // [한국어] watch 플래그가 켜진 연결은 디버그 메시지 출력
      cout << "Dequeuing waiting arrival event at " << FullName() 
	   << " for flit " << w->id << endl;
    }
    
    credits = _output_state[output]->GetCredits( out_vc );
    // [한국어] out_vc의 남은 다운스트림 크레딧 수 조회

    // Try to queue a transmit event for a waiting packet
    // [한국어] 대기 중이던 패킷에 대해 transport 이벤트를 즉시 생성할 수 있는지 시도
    if ( credits > 0 ) {
      // [한국어] 크레딧이 있으면 즉시 전송 이벤트 생성
      tevt         = new tTransportEvent;
      tevt->src_vc = w->vc;
      // [한국어] 소스 VC (입력 VC)
      tevt->dst_vc = out_vc;
      // [한국어] 목적지 VC (출력 VC)
      tevt->input  = w->input;
      // [한국어] 입력 포트
      tevt->watch  = w->watch; // just to have something here
      // [한국어] watch 플래그 상속 (디버그 추적용)
      tevt->id     = w->id;
      // [한국어] 플릿 ID 상속
      
      _transport_queue[output].push( tevt );
      // [한국어] 생성된 transport 이벤트를 output번 큐에 삽입
      
      if ( tevt->watch ) {
        // [한국어] watch 모드 디버그 출력
	cout << "Injecting transport event at " << FullName() 
	     << " for flit " << tevt->id << endl;
      }
      
      credits--;
      // [한국어] 전송에 사용할 크레딧 1개 소모
      _output_state[output]->SetCredits( out_vc, credits );
      // [한국어] 감소된 크레딧 값 저장
      _output_state[output]->SetPresence( out_vc, w->pres - 1 );
      // [한국어] waiting 항목에 기록된 presence count에서 첫 플릿 1개를
      //   이벤트로 변환했으므로 pres-1을 프레즌스에 설정
      
    } else {
      // No credits available, just store presence
      // [한국어] 크레딧이 없으면 transport 이벤트를 생성하지 않고
      //   presence count만 기록해 둠 (크레딧 반환 시 _ArrivalArb가 처리)
      _output_state[output]->SetPresence( out_vc, w->pres );
    }

    delete w;
    // [한국어] 대기 항목 메모리 해제 (PopWaiting으로 소유권 이전되었으므로)

  } else {
    // Tail sent, none waiting => VC is idle
    // [한국어] tail이 전송되었고 대기 중인 연결이 없으면 VC를 idle로 전환
    _output_state[output]->SetState( out_vc, EventNextVCState::idle );
  }
}

/*
 * [한국어]
 * EventRouter::_IncomingFlits - 입력 버퍼의 플릿을 VC 버퍼에 삽입하고
 *                               도착 이벤트 생성
 *
 * @return: 없음
 *
 * _InternalStep()의 Phase 1 첫 단계. _input_buffer에 쌓인 플릿들을 꺼내
 * 해당 입력 포트의 VC 버퍼(_buf[input])에 추가한다.
 * head 플릿이 도착한 VC는 _active[input][vc] = false 상태이며, 이 경우
 * 라우팅 함수(_rf)를 호출해 출력 포트/VC를 결정하고 _active를 true로 설정.
 * VCT 모드가 아니거나, VCT 모드에서 head 플릿인 경우 tArrivalEvent를 생성해
 * _arrival_pipe에 삽입하여 라우팅/디코딩 지연을 모델링한다.
 *
 * 호출 체인: _InternalStep() → [이 함수]
 */
void EventRouter::_IncomingFlits( )
{
  Flit   *f;
  // [한국어] 현재 처리 중인 플릿 포인터
  Buffer *cur_buf;
  // [한국어] 현재 입력 포트의 Buffer 객체 포인터

  tArrivalEvent *aevt;
  // [한국어] 생성할 도착 이벤트 포인터

  _arrival_pipe->WriteAll( 0 );
  // [한국어] arrival_pipe의 모든 슬롯을 NULL로 초기화 — 이번 사이클 Write()만 유효

  for ( int input = 0; input < _inputs; ++input ) {
    // [한국어] 모든 입력 포트를 순회
    if ( !_input_buffer[input].empty( ) ) {
      // [한국어] 이 입력 포트에 수신 대기 중인 플릿이 있는 경우에만 처리
      f = _input_buffer[input].front( );
      // [한국어] 입력 버퍼의 맨 앞 플릿 참조
      _input_buffer[input].pop( );
      // [한국어] 맨 앞 플릿을 입력 버퍼에서 제거

      cur_buf = _buf[input];
      // [한국어] 현재 입력 포트의 VC 버퍼 객체
      int vc = f->vc;
      // [한국어] 이 플릿이 속한 VC 번호

      cur_buf->AddFlit( vc, f );
      // [한국어] 플릿을 cur_buf의 vc번 VC 큐에 추가

      // Head flit arriving at idle VC
      // [한국어] 비활성 VC에 head 플릿이 도착한 경우 — 새 패킷 시작
      if ( !_active[input][vc] ) {
	
	if ( !f->head ) {
	  // [한국어] 비활성 VC에 head가 아닌 플릿이 도착하면 프로토콜 위반
	  cout << "Non-head flit:" << endl;
	  cout << *f;
	  Error( "Received non-head flit at idle VC" );
	}

	const OutputSet *route_set;
	// [한국어] 라우팅 함수가 반환하는 출력 (포트, VC) 집합
	int out_vc, out_port;
	// [한국어] 라우팅 결과로 결정된 출력 VC와 출력 포트

	cur_buf->Route( vc, _rf, this, f, input );
	// [한국어] vc번 VC의 head 플릿에 대해 라우팅 함수 실행 — 결과는 cur_buf 낸부에 저장
	route_set = cur_buf->GetRouteSet( vc );
	// [한국어] 라우팅 함수가 채워준 OutputSet 가져오기

	if ( !route_set->GetPortVC( &out_port, &out_vc ) ) {
	  // [한국어] EventRouter는 단일 (port, vc) 결과를 요구함
	  Error( "The event-driven router requires routing functions with a single (port,vc) output" );
	}

	cur_buf->SetOutput( vc, out_port, out_vc );
	// [한국어] VC 버퍼에 출력 포트/VC를 기록 — 이후 transport 단계에서 사용
	_active[input][vc] = true;
	// [한국어] 이 VC를 활성화 — 이제 이 VC는 현재 패킷을 수용 중
      } else {
	// [한국어] 이미 활성화된 VC에 도착한 경우 — 반드시 body/tail 플릿이어야 함
	if ( f->head ) {
	  cout << *f;
	  Error( "Received head flit at non-idle VC." );
	}
      }
      
      if ( f->watch ) {
	// [한국어] watch 플래그가 켜진 플릿에 대한 디버그 출력
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		    << "Received flit at " << FullName() << ".  Output port = " 
		    << cur_buf->GetOutputPort( vc ) << ", output VC = " 
		    << cur_buf->GetOutputVC( vc ) << endl
		    << *f;
      }

      // In cut-through mode, only head flits generate arrivals,
      // otherwise all flits generate
      // [한국어] VCT 모드에서는 head 플릿만 arrival 이벤트를 생성하고,
      //   store-and-forward 모드에서는 모든 플릿이 arrival 이벤트를 생성

      if ( ( !_vct ) || ( _vct && f->head ) ) {
	// Add the arrival event to a delay pipeline to
	// account for routing/decoding time
	// [한국어] 라우팅/디코딩 시간을 모델링하는 지연 파이프라인에 도착 이벤트 추가

	aevt         = new tArrivalEvent;
	// [한국어] 새 도착 이벤트 동적 할당
	
	aevt->input  = input;
	// [한국어] 도착한 입력 포트
	aevt->output = cur_buf->GetOutputPort( vc );
	// [한국어] 라우팅 결과 출력 포트
	aevt->src_vc = f->vc;
	// [한국어] 소스 VC (입력 VC)
	aevt->dst_vc = cur_buf->GetOutputVC( vc );
	// [한국어] 목적지 VC (출력 VC)
	aevt->head   = f->head;
	// [한국어] head 플릿 여부
	aevt->tail   = f->tail;
	// [한국어] tail 플릿 여부
	
	//if ( f->head && f->tail ) {
	//	Error( "Head/tail packets not supported." );
	//}
	// [한국어] (주석처리됨) single-flit head+tail 패킷은 미지원이었으나
	//   현재는 _vct 모드로 처리될 수 있음
	
	aevt->watch  = f->watch;
	// [한국어] 디버그 추적 플래그
	aevt->id     = f->id;
	// [한국어] 플릿 고유 ID
	
	_arrival_pipe->Write( aevt, input );
	// [한국어] 도착 이벤트를 input번 arrival_pipe에 삽입 —
	//   Advance() 후 _ArrivalRequests()에서 꺼냄

	if ( aevt->watch ) {
	  // [한국어] watch 모드 디버그 출력
	  cout << "Injected arrival event at " << FullName() 
	       << " for flit " << aevt->id << endl;
	}
      }
    }
  }
}

/*
 * [한국어]
 * EventRouter::_ArrivalRequests - arrival_pipe에서 완료된 이벤트를 큐에 넣고
 *                                 중재 요청 등록
 *
 * @param input : 처리할 입력 포트 인덱스
 * @return      : 없음
 *
 * _arrival_pipe->Read(input)으로 지연 완료된 도착 이벤트를 꺼내
 * _arrival_queue[input]에 저장한다. 큐가 비어있지 않으면 맨 앞 이벤트의
 * 출력 포트를 확인하여 _arrival_arbiter[output]->AddRequest(input)으로
 * 출력 포트 중재에 참여시킨다.
 *
 * 호출 체인: _InternalStep() → [이 함수]
 */
void EventRouter::_ArrivalRequests( int input ) 
{
  tArrivalEvent *aevt;
  // [한국어] arrival_pipe에서 읽은 이벤트 포인터
  
  aevt = _arrival_pipe->Read( input );
  // [한국어] input번 arrival_pipe에서 이번 사이클에 완료된 이벤트 읽기
  if ( aevt ) {
    // [한국어] 이벤트가 존재하면 arrival_queue에 저장
    _arrival_queue[input].push( aevt );
  }
  
  if ( !_arrival_queue[input].empty( ) ) {
    // [한국어] arrival_queue에 처리 대기 중인 이벤트가 있으면
    aevt = _arrival_queue[input].front( );
    // [한국어] 맨 앞 이벤트 참조 (FIFO)
    _arrival_arbiter[aevt->output]->AddRequest( input );
    // [한국어] 이벤트의 목적지 출력 포트에 대해 input을 중재 요청으로 등록
  }
}

/*
 * [한국어]
 * EventRouter::_SendTransport - 크레딧 유무에 따라 transport 이벤트 생성
 *                               또는 presence 증가
 *
 * @param input  : 플릿이 위치한 입력 포트 인덱스
 * @param output : 플릿이 전송될 출력 포트 인덱스
 * @param aevt   : 처리할 도착 이벤트 (src_vc, dst_vc, id, watch 정보 포함)
 * @return       : 없음
 *
 * _output_state[output]->GetCredits(aevt->dst_vc)로 다운스트림 버퍼 공간을 확인.
 *   크레딧 있음: 크레딧을 1 감소시키고 tTransportEvent를 생성하여
 *               _transport_queue[output]에 push() — 실제 전송 권한 부여.
 *   크레딧 없음: SetPresence()로 _presence[dst_vc]를 +1 증가.
 *               이는 "플릿이 입력 버퍼에 존재하지만 크레딧이 없어 기다리는 중"
 *               상태를 기록한다. 나중에 크레딧이 반환되면 _ArrivalArb()가
 *               이 presence 값을 보고 transport 이벤트를 생성.
 *
 * 호출 체인: _ArrivalArb() → [이 함수]
 */
void EventRouter::_SendTransport( int input, int output, tArrivalEvent *aevt )
{
  // Try to send a transport event
  // [한국어] transport 이벤트 생성을 시도

  tTransportEvent *tevt;
  // [한국어] 생성할 전송 이벤트 포인터

  int credits;
  // [한국어] dst_vc의 남은 크레딧 수
  int pres;
  // [한국어] dst_vc의 현재 presence count

  credits = _output_state[output]->GetCredits( aevt->dst_vc );
  // [한국어] 목적지 출력 VC의 남은 다운스트림 크레딧 조회
  
  if ( credits > 0 ) {
    // Take a credit and queue a transport event
    // [한국어] 크레딧을 소모하고 transport 이벤트를 큐에 삽입
    credits--;
    // [한국어] 전송에 사용할 크레딧 1개 차감
    _output_state[output]->SetCredits( aevt->dst_vc, credits );
    // [한국어] 감소된 크레딧 값 저장
    
    tevt         = new tTransportEvent;
    // [한국어] 새 전송 이벤트 할당
    tevt->src_vc = aevt->src_vc;
    // [한국어] 소스 VC
    tevt->dst_vc = aevt->dst_vc;
    // [한국어] 목적지 VC
    tevt->input  = input;
    // [한국어] 입력 포트
    tevt->watch  = aevt->watch;
    // [한국어] watch 플래그
    tevt->id     = aevt->id;
    // [한국어] 플릿 ID
	
    _transport_queue[output].push( tevt );
    // [한국어] transport 이벤트를 output번 큐에 삽입 — 이후 _TransportArb에서 중재
	
    if ( tevt->watch ) {
      // [한국어] watch 모드 디버그 출력
      cout << "Injecting transport event at " << FullName() 
	   << " for flit " << tevt->id << endl;
    }
  } else {
    if ( aevt->watch ) {
      // [한국어] watch 모드: 크레딧 부족 상황 디버그 출력
      cout << "No credits available at " << FullName() 
	   << " for flit " << aevt->id << " storing presence." << endl;
    }
    
    // No credits available, just store presence
    // [한국어] 크레딧이 없으면 presence count만 증가 — 지연된 transport 이벤트를
    //   크레딧 반환 시 생성하기 위한 기록
    pres = _output_state[output]->GetPresence( aevt->dst_vc );
    // [한국어] 현재 presence count 조회
    _output_state[output]->SetPresence( aevt->dst_vc, pres + 1 );
    // [한국어] presence count 1 증가
  }
}

/*
 * [한국어]
 * EventRouter::_ArrivalArb - 출력 포트별 도착 이벤트 중재 및 크레딧 처리
 *
 * @param output : 중재할 출력 포트 인덱스
 * @return       : 없음
 *
 * _InternalStep()에서 각 출력 포트마다 호출되는 Phase 1 핵심 함수.
 * 두 가지 작업을 수행한다:
 *
 * 1) 다운스트림 크레딧 처리:
 *    _out_cred_buffer[output]에서 크레딧을 꺼내 _output_state[output]의
 *    크레딧 카운터를 +1한다. VCT 모드와 store-and-forward 모드에서
 *    각각 다른 방식으로 VC 상태 전환 및 transport 이벤트 생성을 처리한다.
 *    - VCT 모드: c->head가 true일 때만 크레딧 증가 + _ProcessWaiting()
 *    - store-and-forward: tail 크레딧 수신 시 _ProcessWaiting()으로 VC 해제,
 *      busy 상태에서 pres > 0이면 즉시 transport 이벤트 생성
 *
 * 2) 도착 이벤트 중재:
 *    _arrival_arbiter[output]->Arbitrate() 후 Match()로 승자 입력 결정.
 *    - head 플릿:
 *        idle  → VC 소유권 획득 + SetInput/SetInputVC + _SendTransport()
 *        busy  → tWaiting 생성 후 PushWaiting()
 *    - body 플릿:
 *        현재 소유자와 일치하면 _SendTransport()
 *        다륾면 IncrWaiting() (다른 패킷의 waiting 항목에 body 추가)
 *
 * 처리가 끝난 aevt는 delete로 메모리 해제.
 *
 * 호출 체인: _InternalStep() → [이 함수]
 */
void EventRouter::_ArrivalArb( int output )
{
  tArrivalEvent   *aevt;
  // [한국어] 처리 중인 도착 이벤트 포인터
  tTransportEvent *tevt;
  // [한국어] 생성할 전송 이벤트 포인터
  Credit          *c;
  // [한국어] 수신된 다운스트림 크레딧 포인터

  EventNextVCState::tWaiting *w;
  // [한국어] 대기열에 삽입할 연결 정보 포인터

  int input;
  // [한국어] arrival 중재 승자 입력 포트
  int credits;
  // [한국어] VC별 크레딧 임시 변수
  int pres;
  // [한국어] VC별 presence 임시 변수

  // Incoming credits can produce or enable
  // transport events --- process them first
  // [한국어] 다운스트림에서 들어온 크레딧을 먼저 처리 — 이로 인해 transport 이벤트가
  //   새로 생성되거나 기존 이벤트가 실행 가능해질 수 있음

  if ( !_out_cred_buffer[output].empty( ) ) {
    // [한국어] output 포트에 반환된 크레딧이 있는 경우
    c = _out_cred_buffer[output].front( );
    // [한국어] 버퍼 맨 앞 크레딧 참조
    _out_cred_buffer[output].pop( );
    // [한국어] 크레딧을 버퍼에서 제거
    
    assert( c->vc.size() == 1 );
    // [한국어] EventRouter에서는 한 번에 하나의 VC에 대한 크레딧만 처리
    int vc = *c->vc.begin();
    // [한국어] 크레딧이 반환된 VC 번호 추출

    EventNextVCState::eNextVCState state = 
      _output_state[output]->GetState( vc );
    // [한국어] 해당 출력 VC의 현재 상태 조회
    
    credits = _output_state[output]->GetCredits( vc );
    // [한국어] 현재 남은 크레딧 수
    pres    = _output_state[output]->GetPresence( vc );
    // [한국어] 현재 presence count (크레딧 없이 대기 중인 플릿 수)
      
    if ( _vct ) {
      // In cut-through mode, only head credits indicate a change in 
      // channel state.
      // [한국어] VCT 모드에서는 head 플릿에 대한 크레딧만 채널 상태 변화를 의미

      if ( c->head ) {
	// [한국어] head 플릿의 크레딧 반환 — 다운스트림이 head를 소비하여 VC 공간 확보
	credits++;
	_output_state[output]->SetCredits( vc, credits );
	// [한국어] 크레딧 카운터 1 증가
	_ProcessWaiting( output, vc );
	// [한국어] VC에 대기 중인 다음 연결 처리 (있으면 소유권 이전, 없으면 idle)
      }
    } else {
      // store-and-forward 모드
      credits++;
      _output_state[output]->SetCredits( vc, credits );
      // [한국어] 플릿 하나가 다운스트림에서 소비되었으므로 크레딧 1 증가

      if ( c->tail ) { // tail flit -- recycle VC
	// [한국어] tail 플릿의 크레딧 반환 — 이 VC를 점유하던 패킷이 완전히 빠져나감
	if ( state != EventNextVCState::busy ) {
	  Error( "Received tail credit at non-busy output VC" );
	}
	
	_ProcessWaiting( output, vc );
	// [한국어] VC 재활용: 대기 중인 다음 연결로 소유권 이전 또는 idle 전환
      } else if ( ( state == EventNextVCState::busy ) && ( pres > 0 ) ) {
	// Flit is present => generate transport event
	// [한국어] busy 상태에서 presence가 있으면 — 크레딧이 생겼으므로
	//   지연되었던 transport 이벤트를 즉시 생성
	
	tevt         = new tTransportEvent;
	tevt->input  = _output_state[output]->GetInput( vc );
	// [한국어] 현재 VC 소유자의 입력 포트
	tevt->src_vc = _output_state[output]->GetInputVC( vc );
	// [한국어] 현재 VC 소유자의 입력 VC
	tevt->dst_vc = vc;
	// [한국어] 목적지 VC
	tevt->watch  = false;
	// [한국어] 크레딧 반환에 의해 생성된 이벤트는 watch 비활성
	tevt->id     = -1;
	// [한국어] 마찬가지로 ID 없음
	
	_transport_queue[output].push( tevt );
	// [한국어] transport 이벤트를 output번 큐에 삽입
	
	pres--;
	// [한국어] presence count 1 감소 (이제 transport 이벤트로 전환됨)
	credits--;
	// [한국어] transport 이벤트에 사용할 크레딧 1 차감
	_output_state[output]->SetPresence( vc, pres );
	_output_state[output]->SetCredits( vc, credits );
      }
    }

    c->Free();
    // [한국어] 사용한 Credit 객체를 풀 할당자에 반환
  }

  // Now process arrival events
  // [한국어] 다운스트림 크레딧 처리 후 arrival 이벤트 중재 수행

  _arrival_arbiter[output]->Arbitrate( );
  // [한국어] output번 출력 포트에 대한 arrival 요청 중재
  input = _arrival_arbiter[output]->Match( );
  // [한국어] 중재 결과: 승자 입력 포트 (-1이면 승자 없음)

  if ( input != -1 ) {  
    // Winning arrival event gets access to output
    // [한국어] 승자 입력의 arrival 이벤트가 output VC에 접근할 권한 획득

    aevt = _arrival_queue[input].front( );
    // [한국어] 승자 입력의 arrival_queue 맨 앞 이벤트 참조
    _arrival_queue[input].pop( );
    // [한국어] 이벤트를 큐에서 제거

    if ( aevt->watch ) {
      // [한국어] watch 모드 디버그 출력
      cout << "Processing arrival event at " << FullName() 
	     << " for flit " << aevt->id << endl;
    }
      
    EventNextVCState::eNextVCState state = 
      _output_state[output]->GetState( aevt->dst_vc );
    // [한국어] 이벤트의 목적지 VC 상태 조회

    if ( aevt->head ) { // Head flits
      // [한국어] head 플릿 도착 이벤트 — 새 패킷 시작
      if ( state == EventNextVCState::idle ) {
	// Allocate the output VC and queue a transport event
	// [한국어] 목적지 VC가 idle이면 소유권 획득 및 transport 이벤트 생성
	_output_state[output]->SetState( aevt->dst_vc, EventNextVCState::busy );
	_output_state[output]->SetInput( aevt->dst_vc, input );
	_output_state[output]->SetInputVC( aevt->dst_vc, aevt->src_vc );

	_SendTransport( input, output, aevt );
	// [한국어] 크레딧 유무에 따라 transport 이벤트 생성 또는 presence 기록
      } else {
	// VC busy => queue a waiting event
	// [한국어] 목적지 VC가 busy이면 대기열에 연결 정보 추가

	w = new EventNextVCState::tWaiting;

	w->input = input;
	// [한국어] 대기 연결의 입력 포트
	w->vc    = aevt->src_vc;
	// [한국어] 대기 연결의 입력 VC
	w->id    = aevt->id;
	// [한국어] 플릿 ID
	w->watch = aevt->watch;
	// [한국어] watch 플래그
	w->pres  = 1;
	// [한국어] 현재 head 플릿 1개가 입력 버퍼에 존재

	_output_state[output]->PushWaiting( aevt->dst_vc, w );
	// [한국어] 대기 연결을 dst_vc의 waiting 리스트 끝에 FIFO 삽입
      }
    } else {
      // [한국어] body 플릿 도착 이벤트
      if ( _vct ) {
	// [한국어] VCT 모드에서는 body 플릿이 arrival 이벤트를 생성하지 않으므로
	//   이 경로로 오면 오류
	Error( "Received arrival event for non-head flit in cut-through mode" );
      }

      if ( state != EventNextVCState::busy ) {
	// [한국어] body 플릿은 반드시 busy 상태의 VC에 도착해야 함
	cout << "flit id = " << aevt->id << endl;
	Error( "Received a body flit at a non-busy output VC" );
      }
      
      if ( ( !_output_state[output]->IsInputWaiting( aevt->dst_vc, input, aevt->src_vc ) ) &&
	   ( input == _output_state[output]->GetInput( aevt->dst_vc ) ) &&
	   ( aevt->src_vc == _output_state[output]->GetInputVC( aevt->dst_vc ) ) ) {
	// Body flit part of the current active VC => queue transport event
	// (the weird IsInputWaiting call handles a body flit waiting in addition
	// to a head flit)
	// [한국어] body 플릿이 현재 VC 소유자의 연결에 속함 —
	//   IsInputWaiting이 false이고 입력/입력VC가 현재 소유자와 일치하면
	//   바로 transport 이벤트 생성.
	//   IsInputWaiting은 "head는 waiting 리스트에 있고 body가 추가 도착"하는
	//   특수 상황을 처리하기 위함

	_SendTransport( input, output, aevt );
      } else {

	// VC busy with a differnet transaction => update waiting event
	// [한국어] VC가 다른 패킷에 의해 busy이고, 이 body 플릿은 그 패킷의
	//   waiting 항목에 추가되어야 함 — presence count 증가
	_output_state[output]->IncrWaiting( aevt->dst_vc, input, aevt->src_vc );
      } 
    }

    delete aevt;
    // [한국어] 처리 완료된 도착 이벤트 메모리 해제
  }
}

/*
 * [한국어]
 * EventRouter::_TransportRequests - 전송 이벤트 큐의 front를 입력 중재기에 등록
 *
 * @param output : 전송 이벤트를 확인할 출력 포트 인덱스
 * @return       : 없음
 *
 * _transport_queue[output]이 비어있지 않으면 front()의 입력 포트를 확인하여
 * _transport_arbiter[tevt->input]->AddRequest(output)으로 요청을 등록한다.
 * 실제 플릿 전송은 _TransportArb()가 담당.
 *
 * 호출 체인: _InternalStep() → [이 함수]
 */
void EventRouter::_TransportRequests( int output )
{
  tTransportEvent *tevt;
  // [한국어] transport_queue의 front 이벤트 포인터
  
  if ( !_transport_queue[output].empty( ) ) {
    // [한국어] output번 transport_queue에 처리 대기 중인 이벤트가 있으면
    tevt = _transport_queue[output].front( );
    // [한국어] 맨 앞 이벤트 참조
    _transport_arbiter[tevt->input]->AddRequest( output );
    // [한국어] 이 이벤트의 입력 포트 중재기에 output을 요청으로 등록
  }
}

/*
 * [한국어]
 * EventRouter::_TransportArb - 입력 포트별 transport 이벤트 중재 및 실제 플릿 전송
 *
 * @param input : 중재할 입력 포트 인덱스
 * @return      : 없음
 *
 * _InternalStep()의 Phase 2 핵심 함수. 입력 포트 input에 대해:
 *   - _transport_free[input]이 true이면 _transport_arbiter[input]->Arbitrate()로
 *     출력 포트를 새로 중재.
 *   - false이면 이전 중재 결과(_transport_match[input])를 재사용 (VCT 모드에서
 *     동일 패킷의 연속 플릿 전송 시 중재 오버헤드 제거).
 *
 * 매칭된 output이 있으면:
 *   1. _transport_queue[output].front()에서 tTransportEvent를 꺼냄
 *   2. _buf[input]에서 tevt->src_vc VC의 플릿을 RemoveFlit()로 꺼냄
 *   3. VC 상태 및 tail 여부에 따라 _active/_transport_free/_transport_match 갱신
 *   4. Credit 객체 생성 (vc, head, tail, id 설정) 후 _credit_pipe->Write(c, input)
 *   5. f->hops++ 및 f->vc = 출력VC로 갱신 후 _crossbar_pipe->Write(f, output)
 *
 * 호출 체인: _InternalStep() → [이 함수]
 */
void EventRouter::_TransportArb( int input ) 
{
  tTransportEvent *tevt;
  // [한국어] 처리할 전송 이벤트 포인터

  int    output;
  // [한국어] 중재 결과로 선택된 출력 포트
  Buffer *cur_buf;
  // [한국어] 현재 입력 포트의 VC 버퍼
  Flit   *f;
  // [한국어] 꺼낸 플릿 포인터
  Credit *c;
  // [한국어] 생성한 크레딧 포인터

  if ( _transport_free[input] ) {
    // [한국어] 새로운 중재가 가능한 상태 — 입력 포트 input의 transport 중재기 실행
    _transport_arbiter[input]->Arbitrate( );
    output = _transport_arbiter[input]->Match( );
  } else {
    // [한국어] VCT 모드에서 연속 전송 중 — 이전 중재 결과를 재사용
    output = _transport_match[input];
  }

  if ( output != -1 ) {  
    // This completes the match from input to output =>
    // one flit can be transferred
    // [한국어] input → output 매칭 완료: 이번 사이클에 플릿 1개를 전송

    tevt = _transport_queue[output].front( );
    // [한국어] output번 transport_queue의 맨 앞 이벤트 참조
    
    if ( tevt->watch ) {
      // [한국어] watch 모드 디버그 출력
      cout << "Processing transport event at " << FullName() 
	   << " for flit " << tevt->id << endl;
    }

    cur_buf = _buf[input];
    // [한국어] 입력 포트 input의 VC 버퍼
    int vc = tevt->src_vc;
    // [한국어] 플릿을 꺼낼 입력 VC

    // Some sanity checking first
    // [한국어] 먼저 상태 일관성 검사 수행

    if ( !_active[input][vc] ) {
      // [한국어] transport grant를 받은 VC가 비활성 상태면 버그
      Error( "Non-active VC received grant." );
    }

    if ( cur_buf->Empty( vc ) ) {
      // [한국어] VC가 비어있는데 grant를 받은 경우 —
      //   arrival 이벤트와 transport 이벤트 타이밍 불일치로 발생할 수 있음
      return; //Error( "Empty VC received grant." );
    }

    if ( tevt->dst_vc != cur_buf->GetOutputVC( vc ) ) {
      // [한국어] transport 이벤트의 목적지 VC와 VC 버퍼에 기록된 출력 VC가 불일치
      Error( "Transport event's VC does not match input's destination VC." );
    }

    f = cur_buf->RemoveFlit( vc );
    // [한국어] 입력 VC 버퍼에서 플릿을 꺼냄 (버퍼에서 제거됨)

    if ( _vct ) {
      // [한국어] VCT 모드: 패킷 전체가 한 번에 전송되므로
      //   tail 플릿까지 같은 출력 포트로 연속 전송
      if ( f->tail ) {
	// [한국어] tail 플릿 전송 완료 — 입력 포트 중재 잠금 해제
	_transport_free[input]  = true;
	_transport_match[input] = -1;

	_transport_queue[output].pop( );
	// [한국어] 처리한 transport 이벤트 제거
	delete tevt;
	// [한국어] 이벤트 메모리 해제

	_active[input][vc] = false;
	// [한국어] VC를 비활성화 — 다음 패킷의 head를 기다릴 준비
      } else {
	// [한국어] tail이 아직 아니면 동일 출력 포트로 중재 결과 잠금 유지
	_transport_free[input]  = false;
	_transport_match[input] = output;
      }
    } else {
      // [한국어] store-and-forward 모드: 플릿마다 개별 transport 이벤트 처리
      _transport_free[input]  = true;
      _transport_match[input] = -1;

      _transport_queue[output].pop( );
      // [한국어] 처리한 transport 이벤트 제거
      delete tevt;

      if ( f->tail ) {
	// [한국어] tail 플릿 전송 완료 — VC 비활성화
	_active[input][vc] = false;
      }
    }

    c = Credit::New( );
    // [한국어] 업스트림으로 반환할 새 크레딧 객체 생성
    c->vc.insert(f->vc);
    // [한국어] 크레딧에 현재 플릿의 입력 VC 번호 기록
    c->head          = f->head;
    // [한국어] head 플릿 여부 기록 (VCT 모드에서 중요)
    c->tail          = f->tail;
    // [한국어] tail 플릿 여부 기록
    c->id            = f->id;
    // [한국어] 플릿 ID 기록
    _credit_pipe->Write( c, input );
    // [한국어] 크레딧을 input번 크레딧 파이프라인에 삽입 —
    //   _credit_delay 사이클 후 업스트림으로 전송
    
    if ( f->watch && c->tail ) {
      // [한국어] watch 모드에서 tail 크레딧 반환 로그 출력
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		  << FullName() << " sending tail credit back for flit " << f->id << endl;
    }

    // Update and forward the flit to the crossbar
    // [한국어] 플릿을 크로스바로 전달하기 전 상태 갱신

    f->hops++;
    // [한국어] 라우터 하나를 더 통과했으므로 홉 카운트 증가
    f->vc = cur_buf->GetOutputVC( vc );
    // [한국어] 플릿의 VC 번호를 출력 VC로 갱신 — 다음 라우터의 입력 VC로 사용됨
    _crossbar_pipe->Write( f, output );
    // [한국어] 플릿을 output번 크로스바 파이프라인에 삽입 —
    //   _crossbar_delay 사이클 후 출력 버퍼로 이동

    if ( f->watch ) {
      // [한국어] watch 모드 디버그 출력
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		  << "Forwarding flit through crossbar at " << FullName() << ":" << endl
		  << *f;
    }  
  }
}

/*
 * [한국어]
 * EventRouter::_OutputQueuing - crossbar_pipe/credit_pipe 완료 항목을 버퍼로 이동
 *
 * @return: 없음
 *
 * _InternalStep()의 마지막 단계. _crossbar_pipe->Read(output)으로 크로스바를
 * 통과한 플릿을 _output_buffer[output]에, _credit_pipe->Read(input)으로
 * 지연 완료된 크레딧을 _in_cred_buffer[input]에 저장한다.
 * 실제 채널 전송은 WriteOutputs() 단계의 _SendFlits()/_SendCredits()가 담당.
 *
 * 호출 체인: _InternalStep() → [이 함수]
 */
void EventRouter::_OutputQueuing( )
{
  Flit   *f;
  // [한국어] 크로스바 파이프라인에서 꺼낸 플릿
  Credit *c;
  // [한국어] 크레딧 파이프라인에서 꺼낸 크레딧

  for ( int output = 0; output < _outputs; ++output ) {
    // [한국어] 모든 출력 포트에 대해 크로스바 파이프라인 읽기
    f = _crossbar_pipe->Read( output );

    if ( f ) {
      // [한국어] 플릿이 도착한 경우 출력 버퍼에 저장
      _output_buffer[output].push( f );
    }
  }  

  for ( int input = 0; input < _inputs; ++input ) {
    // [한국어] 모든 입력 포트에 대해 크레딧 파이프라인 읽기
    c = _credit_pipe->Read( input );

    if ( c ) {
      // [한국어] 크레딧이 도착한 경우 입력 크레딧 버퍼에 저장
      _in_cred_buffer[input].push( c );
    }
  }
}

/*
 * [한국어]
 * EventRouter::_SendFlits - 출력 버퍼의 플릿을 출력 채널로 전송
 *
 * @return: 없음
 *
 * WriteOutputs()에서 호출. 각 출력 포트의 _output_buffer[output]이 비어있지
 * 않으면 front()의 플릿을 꺼내 _output_channels[output]->Send(f)로
 * 다운스트림에 전송한다.
 *
 * 호출 체인: WriteOutputs() → [이 함수]
 */
void EventRouter::_SendFlits( )
{
  for ( int output = 0; output < _outputs; ++output ) {
    // [한국어] 모든 출력 포트를 순회
    if ( !_output_buffer[output].empty( ) ) {
      // [한국어] 출력 버퍼에 전송 대기 중인 플릿이 있으면
      Flit *f = _output_buffer[output].front( );
      // [한국어] 맨 앞 플릿 참조
      _output_buffer[output].pop( );
      // [한국어] 출력 버퍼에서 제거
      _output_channels[output]->Send( f );
      // [한국어] output번 출력 채널로 플릿 전송 — 다음 라우터/노드로 이동
    }
  }
}

/*
 * [한국어]
 * EventRouter::_SendCredits - 크레딧 버퍼의 크레딧을 업스트림으로 전송
 *
 * @return: 없음
 *
 * WriteOutputs()에서 호출. 각 입력 포트의 _in_cred_buffer[input]이 비어있지
 * 않으면 front()의 크레딧을 꺼내 _input_credits[input]->Send(c)로
 * 업스트림 라우터에 반환한다.
 *
 * 호출 체인: WriteOutputs() → [이 함수]
 */
void EventRouter::_SendCredits( )
{
  for ( int input = 0; input < _inputs; ++input ) {
    // [한국어] 모든 입력 포트를 순회
    if ( !_in_cred_buffer[input].empty( ) ) {
      // [한국어] 반환할 크레딧이 있으면
      Credit *c = _in_cred_buffer[input].front( );
      // [한국어] 맨 앞 크레딧 참조
      _in_cred_buffer[input].pop( );
      // [한국어] 버퍼에서 제거
      _input_credits[input]->Send( c );
      // [한국어] input번 입력 크레딧 채널로 크레딧 전송 — 업스트림이 버퍼 공간 확보를 인지
    }
  }
}

/*
 * [한국어]
 * EventRouter::Display - 모든 입력 버퍼 상태를 출력 스트림에 출력
 *
 * @param os : 출력 대상 스트림 (기본값: cout)
 * @return   : 없음
 *
 * 디버그 목적으로 각 입력 포트의 _buf[input]->Display(os)를 호출하여
 * VC별 버퍼 내용을 출력한다.
 *
 * 호출 체인: 외부 디버그/통계 코드 → [이 함수]
 */
void EventRouter::Display( ostream & os ) const
{
  for ( int input = 0; input < _inputs; ++input ) {
    // [한국어] 모든 입력 포트를 순회하며 VC 버퍼 상태 출력
    _buf[input]->Display( os );
  }
}

/*
 * [한국어]
 * EventNextVCState::EventNextVCState - 출력 VC 상태 추적 객체 생성자
 *
 * @param config : BookSim2 설정 객체 — vc_buf_size, num_vcs 값을 읽음
 * @param parent : 부모 Module 포인터 (이 객체를 소유하는 EventRouter)
 * @param name   : 이 상태 모듈의 이름 문자열
 * @return       : 없음 (생성자)
 *
 * config에서 vc_buf_size(다운스트림 버퍼 슬롯 수)와 num_vcs(VC 수)를 읽어
 * per-VC 벡터들을 초기화한다.
 *   - _credits[v] = _buf_size: 초기에 다운스트림 버퍼는 비어있으므로
 *     최대 크레딧을 보유
 *   - _presence[v] = 0: 아직 플릿 없음
 *   - _state[v] = idle: 모든 VC가 유휴 상태
 *
 * 호출 체인: EventRouter 생성자 → [이 함수]
 */
EventNextVCState::EventNextVCState( const Configuration& config, 
				    Module *parent, const string& name ) :
  Module( parent, name )
  // [한국어] Module 기반 클래스 생성자 호출 — 모듈 계층 및 이름 설정
{
  _buf_size = config.GetInt( "vc_buf_size" );
  // [한국어] 다운스트림 VC 버퍼 크기 — gpgpusim.config의 vc_buf_size 옵션
  _vcs      = config.GetInt( "num_vcs" );
  // [한국어] VC 총 개수 — gpgpusim.config의 num_vcs 옵션

  _credits.resize(_vcs, _buf_size);
  // [한국어] VC별 남은 크레딧 수를 _buf_size로 초기화
  _presence.resize(_vcs, 0);
  // [한국어] VC별 presence count를 0으로 초기화
  _input.resize(_vcs);
  // [한국어] VC별 소유자 입력 포트 벡터 크기 확장 (초기값은 사용되지 않음)
  _inputVC.resize(_vcs);
  // [한국어] VC별 소유자 입력 VC 벡터 크기 확장
  _waiting.resize(_vcs);
  // [한국어] VC별 대기 연결 리스트 벡터 크기 확장
  _state.resize(_vcs, idle);
  // [한국어] VC별 상태를 idle로 초기화
}

/*
 * [한국어]
 * EventNextVCState::GetState - 지정 VC의 현재 상태 반환
 *
 * @param vc : 조회할 VC 인덱스 ([0, _vcs) 범위)
 * @return   : eNextVCState — idle, busy, tail_pending 중 하나
 *
 * VC 인덱스 범위를 assert로 검증한 후 _state[vc]를 반환한다.
 *
 * 호출 체인: EventRouter::_ArrivalArb(), _ProcessWaiting() 등
 */
EventNextVCState::eNextVCState EventNextVCState::GetState( int vc ) const
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  // [한국어] VC 인덱스 범위 검증
  return _state[vc];
  // [한국어] 해당 VC 상태 반환
}

/*
 * [한국어]
 * EventNextVCState::GetPresence - 지정 VC의 presence count 반환
 *
 * @param vc : 조회할 VC 인덱스 ([0, _vcs) 범위)
 * @return   : int — 현재 presence count
 *
 * 크레딧 없이 도착하여 대기 중인 플릿 수를 반환한다.
 *
 * 호출 체인: EventRouter::_ArrivalArb(), _SendTransport(), _ProcessWaiting()
 */
int EventNextVCState::GetPresence( int vc ) const
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  // [한국어] VC 인덱스 범위 검증
  return _presence[vc];
  // [한국어] 해당 VC의 presence count 반환
}

/*
 * [한국어]
 * EventNextVCState::GetCredits - 지정 VC의 남은 다운스트림 크레딧 수 반환
 *
 * @param vc : 조회할 VC 인덱스 ([0, _vcs) 범위)
 * @return   : int — 현재 크레딧 수
 *
 * _credits[vc]를 반환한다.
 *
 * 호출 체인: EventRouter::_SendTransport(), _ProcessWaiting(), _ArrivalArb()
 */
int EventNextVCState::GetCredits( int vc ) const
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  // [한국어] VC 인덱스 범위 검증
  return _credits[vc];
  // [한국어] 해당 VC의 남은 크레딧 수 반환
}

/*
 * [한국어]
 * EventNextVCState::GetInput - 지정 출력 VC를 현재 점유하는 입력 포트 반환
 *
 * @param vc : 조회할 출력 VC 인덱스 ([0, _vcs) 범위)
 * @return   : int — 현재 소유자의 입력 포트 인덱스
 *
 * busy 상태의 VC에서만 유효한 값을 반환한다.
 *
 * 호출 체인: EventRouter::_ArrivalArb(), _ProcessWaiting()
 */
int EventNextVCState::GetInput( int vc ) const
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  // [한국어] VC 인덱스 범위 검증
  return _input[vc];
  // [한국어] 해당 VC의 소유자 입력 포트 반환
}

/*
 * [한국어]
 * EventNextVCState::GetInputVC - 지정 출력 VC를 현재 점유하는 입력 VC 반환
 *
 * @param vc : 조회할 출력 VC 인덱스 ([0, _vcs) 범위)
 * @return   : int — 현재 소유자의 입력 VC 인덱스
 *
 * busy 상태의 VC에서만 유효한 값을 반환한다.
 *
 * 호출 체인: EventRouter::_ArrivalArb(), _ProcessWaiting()
 */
int EventNextVCState::GetInputVC( int vc ) const
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  // [한국어] VC 인덱스 범위 검증
  return _inputVC[vc];
  // [한국어] 해당 VC의 소유자 입력 VC 반환
}

/*
 * [한국어]
 * EventNextVCState::IsWaiting - 지정 VC의 대기 리스트가 비어있지 않은지 확인
 *
 * @param vc : 확인할 VC 인덱스 ([0, _vcs) 범위)
 * @return   : bool — true이면 대기 중인 연결 존재
 *
 * _waiting[vc].empty()의 부정을 반환한다.
 *
 * 호출 체인: EventRouter::_ProcessWaiting()
 */
bool EventNextVCState::IsWaiting( int vc ) const
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  // [한국어] VC 인덱스 범위 검증
  return !_waiting[vc].empty( );
  // [한국어] 대기 리스트가 비어있지 않으면 true 반환
}

/*
 * [한국어]
 * EventNextVCState::PushWaiting - 지정 VC의 대기 리스트에 새 항목 삽입
 *
 * @param vc : 대기 리스트에 삽입할 VC 인덱스
 * @param w  : 삽입할 tWaiting 포인터
 * @return   : 없음
 *
 * _waiting[vc].push_back(w)으로 FIFO 순서로 삽입한다.
 * watch 플래그가 true이면 디버그 메시지를 출력한다.
 *
 * 호출 체인: EventRouter::_ArrivalArb()
 */
void EventNextVCState::PushWaiting( int vc, tWaiting *w )
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  // [한국어] VC 인덱스 범위 검증

  if ( w->watch ) {
    // [한국어] watch 모드 디버그 출력
    cout << FullName() << " pushing flit " << w->id
	 << " onto a waiting queue of length " << _waiting[vc].size( ) << endl;
  }

  _waiting[vc].push_back( w );
  // [한국어] 대기 리스트 끝에 연결 정보 추가
}

/*
 * [한국어]
 * EventNextVCState::IncrWaiting - 지정 (input, vc) 쌍의 대기 항목 presence 증가
 *
 * @param vc      : 대기 리스트가 속한 VC 인덱스
 * @param w_input : 탐색할 입력 포트
 * @param w_vc    : 탐색할 입력 VC
 * @return        : 없음
 *
 * _waiting[vc] 리스트를 순차 탐색하여 (w_input, w_vc)와 일치하는 항목을 찾고
 * pres 필드를 1 증가시킨다. 일치 항목이 없으면 Error() 호출.
 *
 * 호출 체인: EventRouter::_ArrivalArb()
 */
void EventNextVCState::IncrWaiting( int vc, int w_input, int w_vc )
{
  list<tWaiting *>::iterator match;
  // [한국어] 대기 리스트 순회용 이터레이터

  // search for match
  // [한국어] 일치하는 대기 항목 탐색
  for ( match = _waiting[vc].begin( ); match != _waiting[vc].end( ); match++ ) {
    if ( ( (*match)->input == w_input ) &&
	 ( (*match)->vc    == w_vc ) ) break;
    // [한국어] 입력 포트와 입력 VC가 모두 일치하면 탐색 중단
  }

  if ( match != _waiting[vc].end( ) ) {
    // [한국어] 일치 항목을 찾은 경우 presence count 증가
    (*match)->pres++;
  } else {
    // [한국어] 일치 항목이 없으면 버그 — IncrWaiting은 반드시 기존 waiting 항목에 대해 호출되어야 함
    Error( "Did not find match in IncrWaiting" );
  }
}

/*
 * [한국어]
 * EventNextVCState::IsInputWaiting - 지정 (input, vc) 쌍이 대기 리스트에 있는지 확인
 *
 * @param vc      : 확인할 VC 인덱스
 * @param w_input : 탐색할 입력 포트
 * @param w_vc    : 탐색할 입력 VC
 * @return        : bool — 일치 항목 존재 여부
 *
 * _waiting[vc] 리스트를 순차 탐색하여 (w_input, w_vc) 일치 항목을 찾는다.
 *
 * 호출 체인: EventRouter::_ArrivalArb()
 */
bool EventNextVCState::IsInputWaiting( int vc, int w_input, int w_vc ) const
{
  list<tWaiting *>::const_iterator match;
  // [한국어] const 순회용 이터레이터
  bool r;
  // [한국어] 반환값

  // search for match
  // [한국어] 일치하는 대기 항목 탐색
  for ( match = _waiting[vc].begin( ); match != _waiting[vc].end( ); match++ ) {
    if ( ( (*match)->input == w_input ) &&
	 ( (*match)->vc    == w_vc ) ) break;
  }

  if ( match != _waiting[vc].end( ) ) {
    // [한국어] 일치 항목 발견
    r = true;
  } else {
    // [한국어] 일치 항목 없음
    r = false;
  }

  return r;
}

/*
 * [한국어]
 * EventNextVCState::PopWaiting - 지정 VC 대기 리스트의 맨 앞 항목 제거 및 반환
 *
 * @param vc : 대기 리스트가 속한 VC 인덱스
 * @return   : tWaiting* — 제거된 항목 포인터
 *
 * _waiting[vc].front()를 저장하고 pop_front()로 리스트에서 제거한 뒤 반환.
 * 반환된 포인터의 메모리는 호출자가 delete해야 한다.
 *
 * 호출 체인: EventRouter::_ProcessWaiting()
 */
EventNextVCState::tWaiting *EventNextVCState::PopWaiting( int vc )
{
  tWaiting *w;
  // [한국어] 제거할 대기 항목 포인터

  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  // [한국어] VC 인덱스 범위 검증
 
  w = _waiting[vc].front( );
  // [한국어] 대기 리스트 맨 앞 항목 참조
  _waiting[vc].pop_front( );
  // [한국어] 리스트에서 제거

  return w;
  // [한국어] 제거된 항목 반환
}

/*
 * [한국어]
 * EventNextVCState::SetState - 지정 VC의 상태 설정
 *
 * @param vc    : 상태를 변경할 VC 인덱스
 * @param state : 새 상태값
 * @return      : 없음
 *
 * _state[vc] = state로 직접 설정.
 *
 * 호출 체인: EventRouter::_ArrivalArb(), _ProcessWaiting()
 */
void EventNextVCState::SetState( int vc, eNextVCState state )
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  // [한국어] VC 인덱스 범위 검증
  _state[vc] = state;
  // [한국어] 상태 갱신
}

/*
 * [한국어]
 * EventNextVCState::SetCredits - 지정 VC의 크레딧 값 설정
 *
 * @param vc    : 크레딧을 변경할 VC 인덱스
 * @param value : 새 크레딧 값
 * @return      : 없음
 *
 * _credits[vc] = value로 직접 설정.
 *
 * 호출 체인: EventRouter::_SendTransport(), _ArrivalArb(), _ProcessWaiting()
 */
void EventNextVCState::SetCredits( int vc, int value )
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  // [한국어] VC 인덱스 범위 검증
  _credits[vc] = value;
  // [한국어] 크레딧 값 갱신
}

/*
 * [한국어]
 * EventNextVCState::SetPresence - 지정 VC의 presence count 설정
 *
 * @param vc    : presence를 변경할 VC 인덱스
 * @param value : 새 presence 값
 * @return      : 없음
 *
 * _presence[vc] = value로 직접 설정.
 *
 * 호출 체인: EventRouter::_SendTransport(), _ArrivalArb(), _ProcessWaiting()
 */
void EventNextVCState::SetPresence( int vc, int value )
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  // [한국어] VC 인덱스 범위 검증
  _presence[vc] = value;
  // [한국어] presence count 갱신
}

/*
 * [한국어]
 * EventNextVCState::SetInput - 지정 출력 VC의 현재 소유자 입력 포트 설정
 *
 * @param vc    : 소유자를 기록할 VC 인덱스
 * @param input : 소유자의 입력 포트 인덱스
 * @return      : 없음
 *
 * _input[vc] = input으로 직접 설정.
 *
 * 호출 체인: EventRouter::_ArrivalArb(), _ProcessWaiting()
 */
void EventNextVCState::SetInput( int vc, int input )
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  // [한국어] VC 인덱스 범위 검증
  _input[vc] = input;
  // [한국어] 소유자 입력 포트 갱신
}

/*
 * [한국어]
 * EventNextVCState::SetInputVC - 지정 출력 VC의 현재 소유자 입력 VC 설정
 *
 * @param vc     : 소유자를 기록할 VC 인덱스
 * @param in_vc  : 소유자의 입력 VC 인덱스
 * @return       : 없음
 *
 * _inputVC[vc] = in_vc로 직접 설정.
 *
 * 호출 체인: EventRouter::_ArrivalArb(), _ProcessWaiting()
 */
void EventNextVCState::SetInputVC( int vc, int in_vc )
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) );
  // [한국어] VC 인덱스 범위 검증
  _inputVC[vc] = in_vc;
  // [한국어] 소유자 입력 VC 갱신
}
