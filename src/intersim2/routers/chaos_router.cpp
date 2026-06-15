// $Id: chaos_router.cpp 5516 2013-10-06 02:14:48Z dub $

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
 * [한국어 설명] ChaosRouter 클래스 구현 (chaos_router.cpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 chaos_router.hpp에 선언된 ChaosRouter 클래스의 모든 멤버 함수를 구현한다.
 * ChaosRouter는 BookSim2 NoC 시뮬레이터의 적응형 라우터로, VC(Virtual Channel) 기반
 * 크레딧 흐름제어 대신 "멀티-큐(multi-queue)"를 이용한 패킷 우회(derouting) 방식으로
 * 혼잡을 완화한다. 입력 패킷은 크로스바를 통해 직접 출력 포트로 전달되거나 멀티-큐에
 * 임시 저장된 뒤 나중에 적합한 출력 포트가 생길 때 전송된다. 이를 통해 특정 출력 포트가
 * 혼잡해도 다른 경로로 패킷이 흐를 수 있어 네트워크 처리량을 향상시킨다.
 * GPGPU-Sim에서는 gpgpusim.config의 `router chaos` 설정으로 활성화된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 전체 계층에서 이 파일의 위치:
 *   CUDA 커널 실행
 *     → shader.cc (SM, 워프 스케줄러가 메모리 요청 생성)
 *       → mem_fetch.cc (메모리 요청 패킷 생성)
 *         → icnt_wrapper.cc (intersim2_push로 NoC에 패킷 주입)
 *           → intersim2/networks/*.cc (네트워크 토폴로지, 채널 연결)
 *             → [이 파일] ChaosRouter (패킷 스케줄링, 버퍼 관리, 크로스바 모델)
 *               → intersim2/channels.cc (출력 채널로 플릿 전달)
 *                 → intersim2_pop (메모리 파티션 도착)
 *                   → gpu-sim.cc → DRAM 스케줄러
 *
 * 사이클 단위 실행 순서 (Network::Evaluate() 기준):
 *   1. Network::ReadInputs() → ChaosRouter::ReadInputs() : 플릿/크레딧 수신
 *   2. Network::Evaluate() → ChaosRouter::_InternalStep() : 스케줄링 + 크로스바 이동
 *   3. Network::WriteOutputs() → ChaosRouter::WriteOutputs() : 플릿/크레딧 송신
 *
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 단일 시뮬레이션 스레드.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - booksim.hpp: BookSim2 전역 설정, 공통 타입 (Flit, Credit, Configuration 등)
 *   - random_utils.hpp: RandomInt() — 공평한 랜덤 라운드-로빈을 위한 난수 생성
 *   - globals.hpp: gRoutingFunctionMap (라우팅 함수 등록 전역 맵), gWatchOut (디버그 출력 스트림)
 *   - chaos_router.hpp: 이 파일이 구현하는 클래스 선언
 *
 * 데이터 흐름:
 *   업스트림(SM) → _input_channels[i] → _input_frame[i] (ReadInputs)
 *     → 경로 A: _crossbar_pipe (Write) → _output_frame[out] → _output_channels[out] → 다운스트림
 *     → 경로 B: _multi_queue[mq] → 이후 사이클에 _crossbar_pipe → _output_frame[out] → 다운스트림
 *   다운스트림(크레딧) → _output_credits[out] → _next_queue_cnt[out]-- (ReadInputs)
 *   이 라우터(크레딧) → _credit_queue[i] → _input_credits[i] (WriteOutputs)
 *
 * 공유하는 핵심 자료구조:
 *   - Flit: 플릿 단위 전송 객체 (head/tail 플래그, VC 번호, 페이로드). 포인터로 전달.
 *   - Credit: 크레딧 객체 (VC 번호 집합). 업스트림 흐름제어에 사용.
 *   - OutputSet: 라우팅 결정 결과 (선호 출력 포트 집합). _rf()가 채워준다.
 *
 * === 주요 함수/구조체 요약 ===
 * ChaosRouter() 생성자:
 *   모든 버퍼, 멀티-큐, 크로스바 파이프, 라우팅 함수를 초기화한다.
 *   inputs == outputs 강제 (Chaos 라우터는 대칭 포트 구조).
 *
 * ReadInputs():
 *   입력 채널에서 플릿을 읽어 _input_frame[]에 저장하고 _input_state[] 전이를 수행.
 *   head 플릿 도착 시 _rf()로 라우팅 결정. 크레딧 수신 시 _next_queue_cnt[] 감소.
 *
 * _InternalStep():
 *   _NextInterestingChannel()로 이번 사이클 처리 채널 결정, _OutputAdvance()로 실제 이동,
 *   _crossbar_pipe->Advance()로 파이프라인 진행.
 *
 * _NextInterestingChannel():
 *   핵심 스케줄링. 라운드-로빈으로 "interesting" 채널 선택 후 MQ→출력, 입력→출력, 입력→MQ 매칭.
 *   비인젝션 채널 스탈 시 _read_stall을 증가시키고 MQ 우회로 해소.
 *
 * _OutputAdvance():
 *   매칭 결과에 따라 플릿을 크로스바 또는 MQ로 물리적으로 이동. tail 처리 후 상태 리셋 및 크레딧 생성.
 *
 * _MultiQueueForOutput():
 *   선호 출력을 가진 가장 오래된 MQ 슬롯 반환. MQ가 가득 찬 경우 derouting(임의 슬롯 반환).
 */

#include "booksim.hpp"  // [한국어] BookSim2 핵심 헤더 — Configuration, Flit, Credit, Error() 등 기본 타입 및 유틸 포함
#include <string>       // [한국어] std::string — 설정 키 조회, 라우팅 함수 이름 조합에 사용
#include <sstream>      // [한국어] std::stringstream — 에러 메시지 조합 등에 사용 (직접적으로는 최소 사용)
#include <iostream>     // [한국어] std::cout, std::endl — 디버그 출력 및 에러 메시지 출력
#include <cstdlib>      // [한국어] abort(), exit() 등 — Error() 호출 경로에서 간접적으로 필요

#include "chaos_router.hpp"   // [한국어] ChaosRouter 클래스 선언 (이 파일이 구현하는 클래스)
#include "random_utils.hpp"   // [한국어] RandomInt(max) — [0, max] 범위의 균일 분포 난수 반환, 공평한 스케줄링에 사용
#include "globals.hpp"        // [한국어] gRoutingFunctionMap (라우팅 함수 이름→포인터 전역 맵), gWatchOut (watch 모드 디버그 스트림)

/*
 * [한국어]
 * ChaosRouter::ChaosRouter - ChaosRouter 생성자
 *
 * @param config: BookSim2 Configuration 객체 — vc_buf_size, multi_queue_size, routing_function,
 *                topology, const_flits_per_packet, crossbar_delay 등의 설정값을 포함
 * @param parent: 부모 Module 포인터 — 이 라우터가 속한 Network 객체
 * @param name:   이 라우터의 식별 이름 문자열 (예: "router_3")
 * @param id:     이 라우터의 고유 정수 ID (네트워크 내 인덱스, 0-based)
 * @param inputs: 입력 포트 수 (= 출력 포트 수여야 함, ChaosRouter 제약)
 * @param outputs:출력 포트 수
 * @return: 없음 (생성자)
 *
 * 초기화 단계:
 *   1. Router(config, parent, name, id, inputs, outputs) 기반 클래스 생성자 호출
 *      → _input_channels[], _output_channels[], _input_credits[], _output_credits[] 배열 할당
 *   2. inputs == outputs 검증 (아니면 Error()로 시뮬레이션 중단)
 *   3. vc_buf_size → _buffer_size, const_flits_per_packet과의 정합성 assert
 *   4. multi_queue_size → _multi_queue_size
 *   5. 라운드-로빈 포인터(_cur_channel=0), 스탈 카운터(_read_stall=0) 초기화
 *   6. gRoutingFunctionMap에서 "routing_function_topology" 형식의 키로 _rf 조회
 *   7. _input_route[0.._inputs-1]: 각 입력 포트용 OutputSet* 객체 생성
 *   8. _mq_route[0.._multi_queue_size-1]: 각 MQ 슬롯용 OutputSet* 객체 생성
 *   9. _crossbar_pipe: PipelineFIFO<Flit> 생성 (_outputs 채널, _crossbar_delay 지연)
 *   10. 모든 벡터 resize 및 초기값 설정 (상태: empty, 매칭: -1, matched: false)
 *
 * 호출 체인: Network 생성자(또는 AddRouter) → [이 함수]
 */
ChaosRouter::ChaosRouter( const Configuration& config,
		    Module *parent, const string & name, int id,
		    int inputs, int outputs )
  : Router( config,       // [한국어] 기반 클래스 Router 초기화 — 입출력 채널/크레딧 채널 배열, _inputs, _outputs, _crossbar_delay 설정
	    parent, name,  // [한국어] 모듈 계층 등록 — parent의 자식으로 이 라우터를 등록하고 name을 FullName()에 사용
	    id,            // [한국어] 라우터 고유 ID — 네트워크 내 0-based 인덱스
	    inputs, outputs ) // [한국어] 입출력 포트 수 전달 — Router가 _inputs, _outputs 멤버로 저장
{
  int i; // [한국어] 초기화 루프용 인덱스 변수

  if ( inputs != outputs ) { // [한국어] ChaosRouter 제약 검사: 입력 포트 수와 출력 포트 수가 반드시 같아야 함
    Error( "Chaos router must have equal number of input and output ports" ); // [한국어] 다르면 시뮬레이션 즉시 중단
  }

  _buffer_size      = config.GetInt( "vc_buf_size" ); // [한국어] 단일 입력/출력/MQ 슬롯의 최대 플릿 수 — gpgpusim.config의 vc_buf_size 항목
  assert(_buffer_size >= config.GetInt( "const_flits_per_packet" )); // [한국어] 안전 검사: 버퍼가 최소 패킷 1개를 수용할 수 있어야 함 (const_flits_per_packet은 패킷 내 고정 플릿 수)

  _multi_queue_size = config.GetInt( "multi_queue_size" ); // [한국어] 멀티-큐 슬롯 총 개수 — gpgpusim.config의 multi_queue_size 항목

  _cur_channel = 0; // [한국어] 라운드-로빈 포인터를 0번 채널(첫 입력 포트)에서 시작
  _read_stall  = 0; // [한국어] 스탈 카운터 초기화 — 처음에는 스탈 없음

  // Routing

  string rf = config.GetStr("routing_function") + "_" + config.GetStr("topology"); // [한국어] 라우팅 함수 이름 구성: "routing_function"+"_"+"topology" 형식의 키 (예: "dim_order_mesh")
  map<string, tRoutingFunction>::iterator rf_iter = gRoutingFunctionMap.find(rf); // [한국어] 전역 라우팅 함수 등록 맵에서 해당 키 탐색 — 모든 라우팅 함수는 시뮬레이터 초기화 시 이 맵에 등록됨
  if(rf_iter == gRoutingFunctionMap.end()) { // [한국어] 맵에서 찾지 못한 경우 — 잘못된 routing_function 또는 topology 설정
    Error("Invalid routing function: " + rf); // [한국어] 설정 오류 에러 메시지 출력 후 시뮬레이션 중단
  }
  _rf = rf_iter->second; // [한국어] 찾은 라우팅 함수 포인터를 _rf에 저장 — 이후 head 플릿 도착마다 호출됨

  _input_route.resize(_inputs); // [한국어] 입력 포트 수만큼 _input_route 벡터 크기 확장 (_inputs는 Router 기반 클래스 멤버)

  for ( i = 0; i < _inputs; ++i ) { // [한국어] 각 입력 포트에 대해 OutputSet 객체 생성
    _input_route[i] = new OutputSet( ); // [한국어] 빈 OutputSet 할당 — _rf()가 head 플릿 도착 시 이 객체에 결과를 채워 넣음
  }

  _mq_route.resize(_multi_queue_size); // [한국어] MQ 슬롯 수만큼 _mq_route 벡터 크기 확장

  for ( i = 0; i < _multi_queue_size; ++i ) { // [한국어] 각 MQ 슬롯에 대해 OutputSet 객체 생성
    _mq_route[i] = new OutputSet( ); // [한국어] 빈 OutputSet 할당 — MQ에 head 플릿이 들어올 때 _rf()로 결과를 채움
  }

  // Alloc pipelines (to simulate processing/transmission delays)

  _crossbar_pipe =
    new PipelineFIFO<Flit>( this, "crossbar_pipeline", _outputs,
			    _crossbar_delay ); // [한국어] 크로스바 파이프라인 생성 — _outputs개의 채널별 FIFO를 가지며 각 플릿은 _crossbar_delay 사이클 후 Read() 가능. _crossbar_delay는 Router 기반 클래스 멤버로 config의 crossbar_delay 값

  // Input and output queues

  _input_frame.resize(_inputs);     // [한국어] 입력 포트 수만큼 입력 프레임 큐 벡터 확장 — 각 입력 포트의 수신 플릿 FIFO
  _output_frame.resize(_outputs);   // [한국어] 출력 포트 수만큼 출력 프레임 큐 벡터 확장 — 각 출력 포트의 전송 대기 플릿 FIFO
  _multi_queue.resize(_multi_queue_size); // [한국어] MQ 슬롯 수만큼 멀티-큐 벡터 확장 — 각 슬롯의 임시 저장 플릿 FIFO

  _credit_queue.resize(_inputs); // [한국어] 입력 포트 수만큼 크레딧 큐 벡터 확장 — 각 입력 방향으로 반환할 크레딧 FIFO

  _input_state.resize(_inputs, empty);         // [한국어] 모든 입력 포트 상태를 empty(비어있음)로 초기화
  _input_output_match.resize(_inputs, -1);     // [한국어] 모든 입력-출력 매칭을 -1(미매칭)로 초기화
  _input_mq_match.resize(_inputs, -1);         // [한국어] 모든 입력-MQ 매칭을 -1(미매칭)로 초기화

  _output_matched.resize(_outputs, false);     // [한국어] 모든 출력 포트를 미매칭(false) 상태로 초기화
  _next_queue_cnt.resize(_outputs, 0);         // [한국어] 모든 출력 포트의 다운스트림 큐 카운트를 0으로 초기화 (초기에는 다운스트림 버퍼 비어있음)

  _multi_match.resize(_multi_queue_size, -1);  // [한국어] 모든 MQ-출력 매칭을 -1(미매칭)로 초기화
  _mq_age.resize(_multi_queue_size);           // [한국어] MQ 나이 벡터 크기 확장 (초기값은 불확정 — 슬롯이 empty일 때는 참조 안 함)
  _mq_matched.resize(_multi_queue_size, false); // [한국어] 모든 MQ 슬롯을 미매칭(false) 상태로 초기화
  _multi_state.resize(_multi_queue_size, empty); // [한국어] 모든 MQ 슬롯 상태를 empty로 초기화

  for ( i = 0; i < _multi_queue_size; ++i ) { // [한국어] MQ 슬롯별 명시적 초기화 루프 (resize의 기본값과 중복이지만 명확성 위해 재설정)
    _multi_state[i] = empty;   // [한국어] MQ 슬롯 상태 empty로 명시 설정
    _multi_match[i] = -1;      // [한국어] MQ-출력 매칭 -1(미매칭)으로 명시 설정
    _mq_matched[i] = false;    // [한국어] MQ 슬롯 예약 여부 false로 명시 설정
  }
}

/*
 * [한국어]
 * ChaosRouter::~ChaosRouter - 소멸자
 *
 * @param: 없음
 * @return: 없음
 *
 * 생성자에서 new로 할당된 객체들을 정리한다:
 *   - _crossbar_pipe: PipelineFIFO 객체 삭제
 *   - _input_route[0.._inputs-1]: 각 입력 포트의 OutputSet 객체 삭제
 *   - _mq_route[0.._multi_queue_size-1]: 각 MQ 슬롯의 OutputSet 객체 삭제
 *
 * 플릿(Flit*)과 크레딧(Credit*) 객체 자체는 BookSim2의 풀 할당 시스템(Flit::Free(),
 * Credit::Free())에서 관리되므로 큐 내부 원소는 여기서 삭제하지 않는다.
 *
 * 호출 체인: Network 소멸자 → [이 함수]
 */
ChaosRouter::~ChaosRouter( )
{
  int i; // [한국어] 삭제 루프용 인덱스 변수

  delete _crossbar_pipe; // [한국어] 크로스바 파이프라인 FIFO 객체 메모리 해제

  for ( i = 0; i < _inputs; ++i ) { // [한국어] 입력 포트 수만큼 반복하여 각 InputRoute OutputSet 삭제
    delete _input_route[i]; // [한국어] i번 입력 포트의 라우팅 결정 OutputSet 객체 해제
  }

  for ( i = 0; i < _multi_queue_size; ++i ) { // [한국어] MQ 슬롯 수만큼 반복하여 각 MQ 라우팅 결정 OutputSet 삭제
    delete _mq_route[i]; // [한국어] i번 MQ 슬롯의 라우팅 결정 OutputSet 객체 해제
  }
}

/*
 * [한국어]
 * ChaosRouter::ReadInputs - 매 사이클 첫 번째 단계: 입력 채널에서 플릿/크레딧 수신
 *
 * @param: 없음
 * @return: 없음
 *
 * 이 함수는 매 사이클 시작 시 Network::ReadInputs()에 의해 호출된다.
 *
 * 처리 내용 (두 단계):
 *
 * [단계 1] 플릿 수신 및 입력 상태 전이:
 *   각 입력 포트(_inputs개)에서 채널을 통해 도착한 플릿을 꺼내(_input_channels[input]->Receive())
 *   _input_frame[input] 큐에 push()한다. 플릿의 head/tail 플래그에 따라 _input_state[input]의
 *   상태 전이를 수행하며, head 플릿 도착 시 _rf()로 라우팅 결정(_input_route[input] 갱신).
 *
 *   상태 전이 요약:
 *     empty + head(+tail) → full
 *     empty + head(only) → filling
 *     filling + tail → full
 *     leaving + head → shared  (이전 패킷 tail이 아직 남아있을 때 새 패킷 head 도착)
 *     cut_through + tail → leaving
 *     shared: head/tail 추가 도착 금지 (Error())
 *
 * [단계 2] 크레딧 수신 및 다운스트림 카운트 갱신:
 *   각 출력 포트(_outputs개)에서 다운스트림이 보낸 크레딧을 수신하여(_output_credits[output]->Receive())
 *   _next_queue_cnt[output]를 감소시킨다. 카운트가 0 미만이면 Error().
 *   수신한 Credit 객체는 c->Free()로 풀에 반환한다.
 *
 * 실행 컨텍스트: 단일 시뮬레이션 스레드, 매 사이클 1회 호출.
 *
 * 호출 체인: Network::ReadInputs() → [이 함수]
 */
void ChaosRouter::ReadInputs( )
{
  Flit   *f; // [한국어] 수신된 플릿 포인터 — 채널에서 꺼낸 플릿 객체를 가리킴
  Credit *c; // [한국어] 수신된 크레딧 포인터 — 다운스트림에서 보낸 크레딧 객체를 가리킴

  for ( int input = 0; input < _inputs; ++input ) {  // [한국어] 모든 입력 포트를 순회하며 플릿 수신
    f = _input_channels[input]->Receive(); // [한국어] input번 입력 채널에서 플릿 수신 시도. 이번 사이클에 도착한 플릿이 없으면 NULL 반환

    if ( f ) { // [한국어] 플릿이 실제로 도착한 경우에만 처리
      _input_frame[input].push( f ); // [한국어] 수신한 플릿을 input번 입력 프레임 FIFO에 추가 — _OutputAdvance()에서 꺼낼 때까지 대기

      if ( f->watch ) { // [한국어] 디버그 watch 모드: 이 플릿에 watch 플래그가 설정된 경우 상세 로그 출력
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		    << "Flit arriving at " << FullName()
		    << " on channel " << input << endl
		    << *f; // [한국어] 시뮬레이션 시각, 라우터 이름, 도착 채널 번호, 플릿 내용 출력
      }

      switch( _input_state[input] ) { // [한국어] 현재 입력 포트 상태에 따라 분기 — 플릿 도착에 의한 상태 전이 결정
      case empty: // [한국어] 버퍼가 완전히 비어있는 상태에서 플릿 도착
	if ( f->head ) { // [한국어] 도착한 플릿이 패킷의 head(첫 번째 플릿)인지 확인 — empty 상태에서는 반드시 head여야 함
	  if ( f->tail ) { // [한국어] head이면서 동시에 tail인 경우 — 단일 플릿 패킷(single-flit packet)
	    _input_state[input] = full; // [한국어] 단일 플릿으로 패킷이 완성되므로 즉시 full 상태로 전이
	  } else {
	    _input_state[input] = filling; // [한국어] head만 도착, tail은 아직 오지 않음 → filling 상태 전이 (이후 플릿 계속 수신 예정)
	  }
	  _rf( this, f, input, _input_route[input], false ); // [한국어] head 플릿 도착 시 라우팅 함수 호출 → _input_route[input]에 선호 출력 포트 집합 저장. false = 단일 경로 라우팅(멀티패스 아님)
	} else {
	  cout << *f; // [한국어] 오류 디버그: 비어있는 버퍼에 head가 아닌 플릿이 도착 → 상태 이상
	  Error( "Empty buffer received non-head flit!" ); // [한국어] 프로토콜 위반 에러 — 시뮬레이션 중단
	}
	break;

      case filling: // [한국어] head를 받고 tail을 기다리는 상태에서 추가 플릿 도착
	if ( f->tail ) { // [한국어] tail 플릿이 도착한 경우 — 패킷의 마지막 플릿
	  _input_state[input] = full; // [한국어] 패킷 전체 수신 완료 → full 상태로 전이
	} else if ( f->head ) { // [한국어] filling 중에 다시 head가 도착하는 것은 비정상 (이전 패킷이 끝나지 않았는데 새 패킷 시작)
	  Error( "Input buffer received another head before previous tail!" ); // [한국어] 버퍼 오버플로우/프로토콜 위반 에러
	}
	// [한국어] head도 tail도 아닌 중간 플릿: 상태 변화 없이 계속 filling 유지, 플릿은 이미 _input_frame에 push됨
	break;

      case full: // [한국어] 버퍼가 이미 꽉 찬 상태에서 새 플릿 도착 — 허용되지 않음
	Error( "Received flit while full!" ); // [한국어] 흐름제어 실패 에러 — 다운스트림이 크레딧 없이 플릿을 전송한 경우
	break;

      case leaving: // [한국어] 이전 패킷의 head가 나갔고 tail이 아직 버퍼에 있는 상태에서 새 플릿 도착
	if ( f->head ) { // [한국어] 다음 패킷의 head가 도착 — leaving 중에 새 패킷 시작 가능 (슬롯 공유)
	  _input_state[input] = shared; // [한국어] 이전 패킷(leaving)과 새 패킷(head 도착)이 버퍼를 공유 → shared 상태 전이

	  if ( f->tail ) { // [한국어] leaving 중에 단일 플릿 패킷이 도착하면 처리 불가 — shared 상태에서 단일 플릿 패킷은 미지원
	    Error( "Received single-flit packet in leaving state!" ); // [한국어] 구현 제약 위반 에러
	  }
	} else {
	  cout << *f; // [한국어] 오류 디버그 정보 출력
	  Error( "Received non-head flit while packet leaving!" ); // [한국어] leaving 상태에서 head가 아닌 플릿 도착 — 프로토콜 위반
	}
	break;

      case cut_through: // [한국어] head가 이미 MQ로 cut-through 전송된 상태에서 후속 플릿 도착
	if ( f->tail ) { // [한국어] tail 플릿이 도착 → cut-through 중인 패킷의 마지막 플릿
	  _input_state[input] = leaving; // [한국어] tail도 이제 나갈 준비 완료 → leaving 상태 전이 (tail이 MQ로 이동되면 empty가 됨)
	}
	if ( f->head ) { // [한국어] cut_through 중에 head 도착 — 현재 패킷이 아직 진행 중인데 새 head가 오면 버퍼 충돌
	  cout << *f; // [한국어] 오류 플릿 정보 출력
	  Error( "Received head flit in cut through buffer!" ); // [한국어] 구현 제약 위반 에러
	}
	// [한국어] head도 tail도 아닌 중간 플릿: cut_through 상태 유지, 플릿은 이미 push됨
	break;

      case shared: // [한국어] 두 패킷이 버퍼를 공유하는 상태에서 추가 플릿 도착 — head나 tail 추가 금지
	if ( f->head ) { // [한국어] 세 번째 패킷 head 도착 — shared에서는 두 패킷만 허용
	  Error( "Shared buffer received another head!" ); // [한국어] 버퍼 용량 초과 에러
	} else if ( f->tail ) { // [한국어] shared에서 tail 도착 — 어느 패킷의 tail인지 모호하므로 오류
	  cout << "Input " << input << endl; // [한국어] 오류 발생한 입력 포트 번호 출력
	  cout << *f; // [한국어] 오류 플릿 내용 출력
	  Error( "Shared buffer received another tail!" ); // [한국어] shared에서 tail 추가 도착 금지 에러
	}
	// [한국어] 중간 플릿: 상태 변화 없이 shared 유지 (두 패킷의 중간 플릿들이 순서대로 push됨)
	break;
      }
    }
  }

  // Process incoming credits

  for ( int output = 0; output < _outputs; ++output ) { // [한국어] 모든 출력 포트를 순회하며 다운스트림 크레딧 수신
    c = _output_credits[output]->Receive(); // [한국어] output번 출력 방향 크레딧 채널에서 크레딧 수신 시도. 이번 사이클에 도착한 크레딧이 없으면 NULL 반환

    if ( c ) { // [한국어] 크레딧이 실제로 도착한 경우에만 처리
      _next_queue_cnt[output]--; // [한국어] 다운스트림이 플릿을 소비했음을 알리는 크레딧 → 카운터 감소 (다운스트림 버퍼 공간 1개 확보됨)

      if ( _next_queue_cnt[output] < 0 ) { // [한국어] 카운터가 0 미만으로 내려가면 크레딧 프로토콜 위반 (예상보다 많은 크레딧 수신)
	Error( "Next queue count fell below zero!" ); // [한국어] 흐름제어 논리 오류 에러
      }

      c->Free(); // [한국어] 크레딧 객체를 BookSim2 풀 할당자에 반환 (메모리 재사용)
    }
  }
}

/*
 * [한국어]
 * ChaosRouter::_InternalStep - 한 사이클의 내부 처리 단계
 *
 * @param: 없음
 * @return: 없음
 *
 * Router 기반 클래스의 순수 가상 함수를 구현. 매 사이클 Network::Evaluate()에서 호출.
 *
 * 실행 순서:
 *   1. _NextInterestingChannel(): 이번 사이클에 처리할 채널 선택 및 매칭 수행
 *      (_input_output_match[], _input_mq_match[], _multi_match[], _input_state[] 갱신)
 *   2. _OutputAdvance(): 매칭 결과에 따라 플릿을 실제로 크로스바 또는 MQ로 이동
 *      (_crossbar_pipe->Write(), _multi_queue[].push(), 크레딧 생성)
 *   3. _crossbar_pipe->Advance(): 크로스바 파이프라인을 한 사이클 진행
 *      (이번 사이클에 Write()된 플릿들이 파이프라인에 진입, _crossbar_delay 사이클 후 Read() 가능)
 *
 * 순서 중요성: Advance()는 반드시 Write() 이후에 호출되어야 이번 사이클에 Write()된 플릿이
 * 즉시 출력되지 않고 파이프라인 지연(_crossbar_delay 사이클)을 정확히 겪는다.
 *
 * 호출 체인: Network::Evaluate() → [이 함수] → _NextInterestingChannel(), _OutputAdvance(), _crossbar_pipe->Advance()
 */
void ChaosRouter::_InternalStep( )
{
  _NextInterestingChannel( ); // [한국어] 스케줄링: 이번 사이클 처리 채널 선택 및 입력/MQ→출력 매칭 결정
  _OutputAdvance( );          // [한국어] 이동: 매칭 결과에 따라 플릿을 크로스바 파이프 또는 멀티-큐로 물리적 이동

  _crossbar_pipe->Advance( ); // [한국어] 크로스바 파이프라인 시간 진행: 이번 사이클에 Write()된 플릿이 파이프라인에 진입하고, 이전 사이클들에 들어온 플릿들이 한 단계씩 앞으로 이동
}

/*
 * [한국어]
 * ChaosRouter::WriteOutputs - 매 사이클 마지막 단계: 출력 채널로 플릿/크레딧 전송
 *
 * @param: 없음
 * @return: 없음
 *
 * Router 기반 클래스의 순수 가상 함수를 구현. 매 사이클 Network::WriteOutputs()에서 호출.
 *
 * 실행 순서:
 *   1. _SendFlits(): 크로스바 파이프라인에서 나온 플릿을 출력 채널로 전송
 *   2. _SendCredits(): _credit_queue[]에 대기 중인 크레딧을 업스트림으로 전송
 *
 * 호출 체인: Network::WriteOutputs() → [이 함수] → _SendFlits(), _SendCredits()
 */
void ChaosRouter::WriteOutputs( )
{
  _SendFlits( );   // [한국어] 크로스바를 통과한 플릿을 출력 채널로 전달 및 다운스트림 카운트 갱신
  _SendCredits( ); // [한국어] _OutputAdvance()에서 생성된 크레딧을 업스트림 라우터/인젝션 소스로 전달
}

/*
 * [한국어]
 * ChaosRouter::_IsInjectionChan - 주어진 채널이 인젝션 채널인지 확인
 *
 * @param chan: 검사할 채널(입력 포트) 번호 [0, _inputs-1]
 * @return: true이면 인젝션 채널 (SM이 NoC로 패킷을 주입하는 채널)
 *
 * GPGPU-Sim의 BookSim2 연동에서 인젝션/이젝션 채널은 관례적으로 마지막 포트 번호를 사용한다.
 * (SM 수 = 포트 수 - 1개 일반 채널 + 1개 인젝션/이젝션 채널)
 * 인젝션 채널은 _NextInterestingChannel()의 _read_stall 카운팅에서 제외된다.
 * SM이 직접 주입하는 채널이므로 일반 라우팅 채널과 다른 우선순위가 적용된다.
 *
 * 호출 체인: _NextInterestingChannel() → [이 함수]
 */
bool ChaosRouter::_IsInjectionChan( int chan ) const
{
  return ( chan == _inputs - 1 ); // [한국어] 마지막 입력 포트 번호(_inputs-1)이면 인젝션 채널로 판단. _inputs는 Router 기반 클래스 멤버
}

/*
 * [한국어]
 * ChaosRouter::_IsEjectionChan - 주어진 채널이 이젝션 채널인지 확인
 *
 * @param chan: 검사할 채널(출력 포트) 번호 [0, _outputs-1]
 * @return: true이면 이젝션 채널 (NoC에서 메모리 파티션으로 패킷을 내보내는 채널)
 *
 * _MultiQueueForOutput()에서 MQ derouting(우회) 대상에서 이젝션 채널을 제외하는 데 사용.
 * 이젝션 채널로의 derouting은 목적지에 상관없이 무조건 내보내는 것이므로 의미가 없고,
 * 목적지 메모리 파티션과의 프로토콜 위반을 유발할 수 있다.
 *
 * 호출 체인: _MultiQueueForOutput() → [이 함수]
 */
bool ChaosRouter::_IsEjectionChan( int chan ) const
{
  return ( chan == _outputs - 1 ); // [한국어] 마지막 출력 포트 번호(_outputs-1)이면 이젝션 채널로 판단. _outputs는 Router 기반 클래스 멤버
}

/*
 * [한국어]
 * ChaosRouter::_InputReady - 입력 포트가 전송 가능한 패킷을 보유 중인지 확인
 *
 * @param input: 검사할 입력 포트 번호 [0, _inputs-1]
 * @return: true이면 이 입력 포트가 크로스바 전송 또는 MQ 이동 가능한 상태
 *
 * "준비됨(ready)"의 기준:
 *   - filling 상태: head가 도착했지만 tail은 아직 오지 않은 상태.
 *     cut-through 전송 허용 — head가 도착하면 바로 전송을 시작하고 이후 플릿은 도착하는 대로 전달.
 *   - full 상태: 패킷 전체(head~tail)가 버퍼에 저장된 상태. 즉시 전송 가능.
 *
 * leaving, empty, cut_through, shared 상태는 false — 이미 전송 중이거나, 아직 데이터 없거나,
 * 추가 매칭이 필요 없는 상태.
 *
 * 호출 체인: _InputForOutput() → [이 함수]
 */
bool ChaosRouter::_InputReady( int input ) const
{
  bool ready = false; // [한국어] 기본값 false — 준비 안 됨

  if ( ( _input_state[input] == filling ) ||  // [한국어] filling: head 도착, tail 미도착 — cut-through 전송 허용
       ( _input_state[input] == full ) ) {     // [한국어] full: 패킷 전체 버퍼에 있음 — 즉시 전송 가능
    ready = true; // [한국어] 두 조건 중 하나라도 만족하면 준비됨으로 설정
  }

  return ready; // [한국어] 준비 상태 반환 — _InputForOutput()에서 이 값으로 매칭 여부 결정
}

/*
 * [한국어]
 * ChaosRouter::_OutputFull - 출력 포트 전송 대기 큐가 가득 찼는지 확인
 *
 * @param out: 검사할 출력 포트 번호 [0, _outputs-1]
 * @return: true이면 _output_frame[out] 큐의 현재 크기 >= _buffer_size (전송 불가)
 *
 * _output_frame[out]이 가득 차면 크로스바에서 나온 플릿을 추가로 받을 수 없다.
 * 현재 _OutputAvail()에서 주석처리된 조건으로 간접 참조되며,
 * _MultiQueueForOutput()의 isfull 판단에서도 간접 사용된다.
 *
 * 호출 체인: (주석처리된 _OutputAvail() 조건), 참고용
 */
bool ChaosRouter::_OutputFull( int out ) const
{
  return ( _output_frame[out].size( ) >= (size_t)_buffer_size ); // [한국어] 출력 프레임 큐 현재 크기와 최대 버퍼 크기 비교. size()는 size_t(부호없는), _buffer_size는 int이므로 명시적 캐스트
}

/*
 * [한국어]
 * ChaosRouter::_OutputAvail - 출력 포트가 새 패킷을 받을 수 있는지 확인
 *
 * @param out: 검사할 출력 포트 번호 [0, _outputs-1]
 * @return: true이면 이 출력 포트가 이번 사이클에 매칭 가능 (미매칭 && 출력 큐 비어있음)
 *
 * 두 조건 모두 충족해야 "가용":
 *   1. !_output_matched[out]: 이번 사이클에 아직 다른 입력/MQ에 할당되지 않음
 *   2. _output_frame[out].empty(): 출력 큐가 완전히 비어있음 (현재 전송 대기 플릿 없음)
 *
 * 조건 2가 엄격한 이유: 크로스바에서 나온 플릿이 _output_frame[]에 대기 중이면,
 * 추가로 새 패킷을 받아 큐를 더 채우면 다운스트림으로의 전송 순서가 복잡해질 수 있다.
 * 주석처리된 세 번째 조건(_next_queue_cnt[out] == 0)은 보수적 전략으로 고려되었으나
 * 현재 비활성화 상태.
 *
 * 호출 체인: _NextInterestingChannel() → [이 함수]
 */
bool ChaosRouter::_OutputAvail( int out ) const
{
  return ( ( !_output_matched[out] ) &&  ( _output_frame[out].empty( ) ) ); // [한국어] 미매칭 && 출력 큐 비어있음 — 두 조건 모두 충족 시 가용
	   //&& ( _next_queue_cnt[out] == 0 ) );  // [한국어] (주석처리됨) 다운스트림 큐도 비어있을 때만 허용하는 더 보수적 조건
  //return ( ( !_output_matched[out] ) && !_OutputFull( out ) ); // [한국어] (주석처리됨) 출력 큐가 가득 차지 않은 경우 허용하는 덜 보수적 조건 (현재 미사용)
}

/*
 * [한국어]
 * ChaosRouter::_MultiQueueFull - 멀티-큐 슬롯이 가득 찼는지 확인
 *
 * @param mq: 검사할 멀티-큐 슬롯 번호 [0, _multi_queue_size-1]
 * @return: true이면 _multi_queue[mq] 큐의 현재 크기 >= _buffer_size (추가 플릿 수용 불가)
 *
 * _FindAvailMultiQueue()에서 수용 가능한 슬롯을 찾을 때 사용.
 * _OutputAdvance()에서 입력→MQ 이동 전 공간 확인에도 사용.
 *
 * 호출 체인: _FindAvailMultiQueue() → [이 함수], _OutputAdvance() → [이 함수]
 */
bool ChaosRouter::_MultiQueueFull( int mq ) const
{
  return ( _multi_queue[mq].size( ) >= (size_t)_buffer_size ); // [한국어] MQ 슬롯 현재 큐 크기와 최대 버퍼 크기 비교. size_t 형변환으로 부호 비교 경고 방지
}

/*
 * [한국어]
 * ChaosRouter::_InputForOutput - 주어진 출력 포트를 선호하는 준비된 입력 포트 탐색
 *
 * @param output: 찾고자 하는 출력 포트 번호 [0, _outputs-1]
 * @return: 해당 출력을 원하는 준비된 입력 포트 번호, 없으면 -1
 *
 * 탐색 방식:
 *   - RandomInt(_inputs-1)로 랜덤 시작 오프셋 생성 → 특정 입력에 편향되지 않도록 공평성 확보
 *   - (i + offset) % _inputs 패턴으로 모든 입력을 순환 탐색
 *   - 조건: _InputReady(input) (full 또는 filling 상태) &&
 *            !_input_route[input]->OutputEmpty(output) (이 출력을 선호하는 라우팅 결정)
 *   - 첫 번째로 발견된 매칭 입력 반환 (랜덤 시작으로 동일 확률 보장)
 *
 * OutputSet::OutputEmpty(output): 이 OutputSet에 output 포트로의 경로가 존재하면 false 반환.
 * (즉, _input_route[input]에 output이 포함되어 있으면 이 입력이 output을 선호함)
 *
 * 호출 체인: _NextInterestingChannel() → [이 함수] → _InputReady(), OutputSet::OutputEmpty()
 */
int ChaosRouter::_InputForOutput( int output ) const
{
  // return an input that prefers this output

  int  input;                               // [한국어] 탐색 중인 입력 포트 번호
  int  offset = RandomInt( _inputs - 1 );  // [한국어] 랜덤 시작 오프셋 — [0, _inputs-1] 범위의 균일 분포 난수. 매 호출마다 다른 시작점으로 탐색하여 특정 입력 포트 편향 방지
  bool match  = false;                      // [한국어] 매칭 발견 여부 플래그

  for ( int i = 0; ( i < _inputs ) && ( !match ); ++i ) { // [한국어] 모든 입력 포트를 최대 1회 순환하며 탐색. match가 true가 되면 즉시 탈출
    input = ( i + offset ) % _inputs; // [한국어] 랜덤 오프셋을 더한 후 모듈러 연산으로 순환 인덱스 계산 (예: offset=3, i=0 → input=3, i=1 → input=4, ...)

    if ( _InputReady( input ) &&                                // [한국어] 이 입력이 전송 준비됨 (filling 또는 full)
	 ( ! _input_route[input]->OutputEmpty( output ) ) ) {  // [한국어] 이 입력의 라우팅 결정에 원하는 output 포트가 포함됨 (OutputEmpty()가 false = 포함됨)
      match = true; // [한국어] 조건 충족 — 이 입력이 output을 선호하고 전송 준비도 됨
    }
  }

  return match ? input : -1; // [한국어] 매칭 발견 시 해당 입력 포트 번호 반환, 없으면 -1 반환
}

/*
 * [한국어]
 * ChaosRouter::_MultiQueueForOutput - 주어진 출력 포트에 보낼 최적 멀티-큐 슬롯 탐색
 *
 * @param output: 찾고자 하는 출력 포트 번호 [0, _outputs-1]
 * @return: 해당 출력을 선호하는 가장 오래된 MQ 슬롯 번호, 없으면 -1
 *          (단, MQ 전체 가득 차고 이젝션 채널 아니면 derouting으로 임의 슬롯 반환)
 *
 * 탐색 알고리즘 (우선순위 순):
 *
 * [단계 1] 선호 슬롯 탐색 (일반 경로):
 *   모든 MQ 슬롯을 순회하며:
 *   - 조건: _multi_match[i]==-1 (아직 매칭 안 됨) &&
 *            (_multi_state[i]==full || _multi_state[i]==filling) (전송 가능 상태)
 *   - 추가 조건: !_mq_route[i]->OutputEmpty(output) (이 슬롯이 output을 선호)
 *   - 이 조건을 만족하는 슬롯 중 _mq_age[i] 최대값(가장 오래 기다린) 슬롯 선택 (oldest-first)
 *
 * [단계 2] isfull 판단:
 *   MQ의 모든 슬롯이 filling/full/shared 상태이면 isfull=true
 *   (한 개라도 empty/leaving/cut_through이면 false → derouting 불필요)
 *
 * [단계 3] Derouting (선호 슬롯 없고 MQ 가득 차고 이젝션 채널 아닐 때):
 *   RandomInt로 시작점을 정해 filling/full 상태인 임의 슬롯을 반환.
 *   "derouting": 원래 목적지가 아닌 다른 출력으로 패킷을 보내 MQ 공간을 확보.
 *   이젝션 채널로의 derouting은 금지(_IsEjectionChan(output) 체크).
 *   모든 슬롯을 순회해도 없으면 "write stall" 출력 (드문 상황).
 *
 * 호출 체인: _NextInterestingChannel() → [이 함수] → _IsEjectionChan(), OutputSet::OutputEmpty()
 */
int ChaosRouter::_MultiQueueForOutput( int output ) const
{
  // return oldest multi queue that prefers the output,
  // or if none prefer and the multi queue is full,
  // return a random entry

  int mq_oldest = -1; // [한국어] 현재까지 발견된 가장 오래된 선호 슬롯 인덱스 (-1: 아직 없음)
  int mq_age;         // [한국어] mq_oldest에 해당하는 나이 값 (비교용, 초기값은 첫 매칭 시 설정)

  int m, r; // [한국어] m: derouting 루프의 모듈러 인덱스, r: derouting 시작 랜덤 오프셋

  bool isfull = true; // [한국어] MQ 전체가 가득 찼는지 여부. 초기값 true, 빈 슬롯 발견 시 false로 전환

  for ( int i = 0; i < _multi_queue_size; ++i ) { // [한국어] 모든 MQ 슬롯 순회 — 선호 슬롯 탐색 + isfull 판단
    if ( ( _multi_match[i] == -1 ) &&             // [한국어] 이 슬롯이 이번 사이클에 아직 다른 출력에 매칭되지 않은 경우
	 ( ( _multi_state[i] == full ) ||          // [한국어] 패킷 전체가 슬롯에 있음 (즉시 전송 가능)
	   (  _multi_state[i] == filling ) ) ) {   // [한국어] head만 있고 tail은 아직 수신 중 (cut-through 전송 허용)

      if ( ( !_mq_route[i]->OutputEmpty( output ) ) &&  // [한국어] 이 슬롯의 라우팅 결정에 원하는 output 포트가 포함됨 (선호)
	   ( ( mq_oldest == -1 ) || ( _mq_age[i] > mq_age ) ) ) { // [한국어] 처음 발견이거나 현재 기록보다 더 오래된 슬롯이면 갱신 (oldest-first 정책)
	mq_oldest = i;        // [한국어] 현재까지 가장 오래된 선호 슬롯 인덱스 갱신
	mq_age    = _mq_age[i]; // [한국어] 해당 슬롯의 나이 값 저장 (다음 비교 기준)
      }
    }

    // deroute only if all queues contain head flits ...

    if ( ( _multi_state[i] != full ) &&      // [한국어] 이 슬롯이 full이 아니거나
	 ( _multi_state[i] != filling ) &&   // [한국어] filling이 아니거나
	 ( _multi_state[i] != shared ) ) {   // [한국어] shared가 아닌 경우 (즉, empty/leaving/cut_through 등 빈 슬롯 존재)
      isfull = false; // [한국어] MQ 전체가 채워진 게 아님 → derouting 불필요
    }
  }

  // Don't deroute MQs to the ejection channel
  if ( ( mq_oldest == -1 ) && isfull &&         // [한국어] 선호 슬롯이 없고 MQ 전체가 가득 찬 경우에만 derouting 시도
       ( !_IsEjectionChan( output ) ) ) {        // [한국어] 단, 이젝션 채널로의 derouting은 금지 (무의미한 전달 방지)
    r = RandomInt( _multi_queue_size - 1 );      // [한국어] derouting 시작점 랜덤 오프셋 — 특정 슬롯 편향 방지

    // Find first routable multi-queue
    for ( int i = 0; i < _multi_queue_size; ++i ) { // [한국어] 모든 MQ 슬롯을 순환하며 전송 가능한 슬롯 탐색
      m = ( i + r ) % _multi_queue_size; // [한국어] 랜덤 오프셋 기반 순환 인덱스 계산
      if ( ( _multi_state[m] == filling ) || // [한국어] 이 슬롯이 filling 상태 (head 도착, tail 미도착) — 강제 우회 가능
	   ( _multi_state[m] == full ) ) {   // [한국어] 이 슬롯이 full 상태 (패킷 전체 있음) — 강제 우회 가능
	mq_oldest = m;     // [한국어] 임의 선택된 슬롯으로 mq_oldest 설정 (derouting 대상)
	//cout << "DEROUTING at " << FullName() << endl; // [한국어] (주석처리됨) derouting 발생 디버그 로그
	break;             // [한국어] 첫 번째 발견 즉시 탈출 (더 이상 탐색 불필요)
      }
    }

    if ( mq_oldest == -1 ) { // [한국어] derouting 후에도 슬롯을 찾지 못한 경우 (MQ 전체가 filling/full/shared가 아닌 상태)
      cout << "write stall" << endl; // [한국어] 완전한 쓰기 스탈 발생 — 이론적으로 드문 상황, 로그 출력
    }
  }

  return mq_oldest; // [한국어] 선택된 MQ 슬롯 번호 반환 (-1: 사용 가능한 슬롯 없음)
}

/*
 * [한국어]
 * ChaosRouter::_FindAvailMultiQueue - 비어있는(수용 가능한) 멀티-큐 슬롯 탐색
 *
 * @param: 없음
 * @return: 사용 가능한 MQ 슬롯 번호 (비어있고 매칭 안 됨), 없으면 -1
 *
 * 조건: !_MultiQueueFull(i) (슬롯이 _buffer_size 미만의 플릿을 보유) &&
 *        !_mq_matched[i] (이번 사이클에 아직 입력 채널에 예약되지 않음)
 *
 * 이 함수는 _NextInterestingChannel()에서 _read_stall > 0일 때 MQ 우회를 시도하기 전에
 * 실제로 플릿을 받을 수 있는 슬롯이 있는지 확인하는 데 사용된다.
 * 선형 탐색으로 첫 번째 적합한 슬롯을 반환하므로 앞번호 슬롯이 항상 먼저 채워지는 경향이 있다.
 * (라운드-로빈 방식이 아니므로 공평성은 _mq_age 정책으로 보완)
 *
 * 호출 체인: _NextInterestingChannel() → [이 함수] → _MultiQueueFull()
 */
int ChaosRouter::_FindAvailMultiQueue( ) const
{
  // return any empty multi queue slot

  int avail = -1; // [한국어] 찾은 사용 가능 슬롯 번호 (-1: 아직 없음)

  for ( int i = 0; i < _multi_queue_size; ++i ) { // [한국어] 모든 MQ 슬롯을 선형 탐색
    if ( ( !_MultiQueueFull( i ) ) &&  // [한국어] 이 슬롯의 큐가 가득 차지 않음 (추가 플릿 수용 가능)
	 ( !_mq_matched[i] ) ) {       // [한국어] 이번 사이클에 다른 입력 채널에 아직 예약되지 않음
      avail = i; // [한국어] 적합한 슬롯 발견 — 슬롯 번호 저장
      break;     // [한국어] 첫 번째 발견 즉시 반환 (추가 탐색 불필요)
    }
  }

  return avail; // [한국어] 발견된 슬롯 번호 반환 (-1: 모든 슬롯이 가득 찼거나 이미 예약됨)
}

/*
 * [한국어]
 * ChaosRouter::_NextInterestingChannel - 이번 사이클 처리 채널 선택 및 매칭 수행
 *
 * @param: 없음
 * @return: 없음
 * (내부 상태 갱신: _input_output_match[], _input_mq_match[], _multi_match[],
 *  _output_matched[], _input_state[], _cur_channel, _read_stall)
 *
 * ChaosRouter의 핵심 스케줄링 함수. 매 사이클 _InternalStep() 시작 시 호출된다.
 *
 * "interesting" 채널 판단 기준 (라운드-로빈으로 _cur_channel부터 탐색):
 *   Case 1: _OutputAvail(_cur_channel) == true 이고,
 *           _MultiQueueForOutput(_cur_channel) != -1 또는 _InputForOutput(_cur_channel) != -1
 *           → 이 출력 포트로 보낼 데이터가 있음
 *   Case 2: _input_state[_cur_channel] == full
 *           → 이 입력 포트가 꽉 찬 패킷을 보유 중 (스케줄링 필요)
 *
 * Interesting한 채널 발견 시 매칭 결정 (우선순위: MQ > 입력→출력):
 *   - mq_index != -1: _output_matched[_cur_channel]=true, _multi_match[mq_index]=_cur_channel
 *   - in_index != -1: _output_matched[_cur_channel]=true, _input_output_match[in_index]=_cur_channel,
 *                     _input_state[in_index] full→leaving, filling→cut_through
 *
 * 스탈(stall) 처리 (_read_stall 메커니즘):
 *   - interesting이지만 출력 포트가 없고 비인젝션 채널이면: _read_stall++
 *   - interesting하고 매칭 성공 또는 인젝션이면: _cur_channel 전진, _read_stall=0
 *   - _read_stall > 0이면 MQ 우회 경로:
 *     mq_avail = _FindAvailMultiQueue()
 *     성공: _input_state 전이, _input_mq_match[_cur_channel]=mq_avail, _mq_matched[mq_avail]=true,
 *            _cur_channel 전진, _read_stall=0
 *     실패: _read_stall++ (완전 스탈 — 이번 사이클 이동 없음)
 *
 * 호출 체인: _InternalStep() → [이 함수] → _OutputAvail(), _MultiQueueForOutput(),
 *            _InputForOutput(), _FindAvailMultiQueue(), _IsInjectionChan()
 */
void ChaosRouter::_NextInterestingChannel( )
{
  bool interesting; // [한국어] 현재 _cur_channel이 이번 사이클에 처리할 가치가 있는("interesting") 채널인지 여부

  int  mq_index;   // [한국어] _cur_channel 출력을 선호하는 MQ 슬롯 번호 (-1: 없음)
  int  in_index;   // [한국어] _cur_channel 출력을 선호하는 입력 포트 번호 (-1: 없음)
  int  mq_avail;   // [한국어] _read_stall 처리 시 입력 패킷을 MQ로 우회할 슬롯 번호 (-1: 없음)

  int c; // [한국어] 라운드-로빈 탐색 루프 카운터 — 최대 _inputs 번 반복하여 무한 루프 방지

  interesting = false; // [한국어] 초기화: 아직 interesting 채널 발견 안 됨
  mq_index = -1;       // [한국어] 초기화: MQ 슬롯 미발견
  in_index = -1;       // [한국어] 초기화: 입력 포트 미발견

  // A channel is interesting if
  //
  // ( output frame available and
  //   ( ( a multiqueue packet wants output channel ) or
  //     ( an input packet wants output channel ) or
  //     ( the multiqueue is full ) ) )
  // or
  // ( the packet at the input channel is stalled )

  for ( c = 0; ( c < _inputs ) && ( !interesting ); ++c ) { // [한국어] 최대 _inputs번 반복하여 interesting 채널 탐색. 발견 즉시 탈출
    if ( _OutputAvail( _cur_channel ) ) { // [한국어] _cur_channel 번 출력 포트가 이번 사이클에 새 패킷을 받을 수 있는지 확인
      mq_index = _MultiQueueForOutput( _cur_channel ); // [한국어] _cur_channel 출력을 선호하는(또는 derouting 가능한) MQ 슬롯 탐색
      in_index = _InputForOutput( _cur_channel );      // [한국어] _cur_channel 출력을 선호하는 준비된 입력 포트 탐색

      if ( ( mq_index != -1 ) || ( in_index != -1 ) ) { // [한국어] MQ 또는 입력 중 하나라도 이 출력을 원하면 interesting
	interesting = true; // [한국어] 출력 포트 가용 + 보낼 데이터 있음 → 처리 가치 있음
      }
    }

    if ( _input_state[_cur_channel] == full ) { // [한국어] 이 입력 채널에 패킷이 꽉 차 있으면 (출력 포트 가용 여부 무관하게) interesting
      interesting = true; // [한국어] 꽉 찬 패킷이 기다리고 있음 → 스케줄링 필요
    }

    if ( !interesting ) { // [한국어] 현재 채널이 interesting하지 않으면 다음 채널로 이동
      _cur_channel = ( _cur_channel + 1 ) % _inputs; // [한국어] 라운드-로빈: 다음 채널로 전진 (모듈러 연산으로 순환)
    }
  }

  if ( interesting ) { // [한국어] interesting 채널을 찾은 경우 매칭 수행
    //cout << _cur_channel << " is interesting at " << FullName() << endl; // [한국어] (주석처리됨) 디버그 로그

    if ( mq_index != -1 ) { // [한국어] MQ 슬롯이 출력을 선호하는 경우 → MQ→출력 매칭 우선 처리
      //cout << "Match for multi-queue " << mq_index << " at " << FullName()
      //    << ", output matched = " << _output_matched[_cur_channel] << endl; // [한국어] (주석처리됨) 매칭 디버그 로그

      _output_matched[_cur_channel] = true;    // [한국어] _cur_channel 출력 포트를 이번 사이클에 예약 (다른 입력이 이 출력을 가져가지 못하게)
      _multi_match[mq_index] = _cur_channel;   // [한국어] mq_index 슬롯이 _cur_channel 출력으로 이번 사이클에 전송하도록 매칭 설정
    } else if ( in_index != -1 ) { // [한국어] MQ 매칭은 없지만 입력 포트가 출력을 선호하는 경우 → 입력→출력 직접 매칭
      _output_matched[_cur_channel] = true;          // [한국어] _cur_channel 출력 포트를 이번 사이클에 예약
      _input_output_match[in_index] = _cur_channel;  // [한국어] in_index 입력이 _cur_channel 출력으로 크로스바를 통해 전송하도록 매칭

      //cout << "Match for input " << in_index << " at " << FullName() << endl; // [한국어] (주석처리됨) 입력 매칭 디버그 로그

      if ( _input_state[in_index] == full ) {         // [한국어] 입력이 full 상태 (패킷 전체 수신 완료) → head가 이제 나가므로 leaving으로 전이
	_input_state[in_index] = leaving;              // [한국어] head가 크로스바로 나가기 시작 → leaving 상태 (나머지 플릿은 아직 버퍼에 남음)
      } else if ( _input_state[in_index] == filling ) { // [한국어] 입력이 filling 상태 (tail 미도착) → head가 크로스바로 cut-through 전송
	_input_state[in_index] = cut_through;           // [한국어] head가 크로스바로 나가고 나머지 플릿은 계속 도착 중 → cut_through 상태
      } else {
	Error( "Tried to route input through crossbar that was not full or filling!" ); // [한국어] 예상치 못한 상태에서 크로스바 라우팅 시도 — 논리 오류
      }
    }

    // Any non-injection channel that is routable is
    // directed to the multi-queue
    if ( ( ( _input_state[_cur_channel] == filling ) || // [한국어] _cur_channel 입력이 filling 상태이거나
	   ( _input_state[_cur_channel] == full ) ) &&   // [한국어] full 상태이면서
	 ( !_IsInjectionChan( _cur_channel ) ) ) {        // [한국어] 인젝션 채널이 아닌 경우 → _read_stall 카운트 증가 (스탈 추적)
      ++_read_stall; // [한국어] 이 채널이 출력 포트를 얻지 못했거나 아직 비인젝션 입력에 매칭이 안 됨 → 스탈 카운트 1 증가
    } else {
      // go to next channel for the next cycle
      _cur_channel = ( _cur_channel + 1 ) % _inputs; // [한국어] 매칭 성공이거나 인젝션 채널인 경우 → 다음 사이클을 위해 다음 채널로 전진
      _read_stall = 0; // [한국어] 스탈 해소 — 이번 채널 처리 성공이므로 카운터 리셋
    }
  }

  if ( _read_stall > 0 ) { // [한국어] 스탈이 발생했으면 MQ 우회 경로 시도 (_read_stall은 위의 else 분기에서 리셋되지 않은 경우에만 양수)
    mq_avail = _FindAvailMultiQueue( ); // [한국어] 입력 패킷을 임시 저장할 수 있는 비어있는 MQ 슬롯 탐색

    if ( mq_avail != -1 ) { // [한국어] 사용 가능한 MQ 슬롯이 있는 경우 → MQ 우회 수행
	if ( _input_state[_cur_channel] == full ) {         // [한국어] 입력이 full 상태 → head가 MQ로 나가므로 leaving 상태 전이
	  _input_state[_cur_channel] = leaving;              // [한국어] 패킷 전체가 있었으나 head가 MQ로 cut-through → leaving
	} else if ( _input_state[_cur_channel] == filling ) { // [한국어] 입력이 filling 상태 → head가 MQ로 cut-through 전송
	  _input_state[_cur_channel] = cut_through;           // [한국어] head가 MQ로 나가고 나머지 플릿은 계속 도착 예정 → cut_through
	} else {
	  cout << "Input " << _cur_channel << " state = "
	       << _input_state[_cur_channel] << endl; // [한국어] 오류 발생 입력 포트 번호와 현재 상태 출력 (디버그)
	  Error( "Tried to route input throught multi-queue that was not full or filling!" ); // [한국어] 예상치 못한 상태에서 MQ 라우팅 시도 — 논리 오류
	}

	_input_mq_match[_cur_channel] = mq_avail;  // [한국어] _cur_channel 입력이 mq_avail 슬롯으로 이번 사이클에 이동하도록 매칭
	_mq_matched[mq_avail] = true;               // [한국어] mq_avail 슬롯을 이번 사이클에 예약 (_FindAvailMultiQueue()에서 중복 선택 방지)

	// go to next channel for the next cycle
	_cur_channel = ( _cur_channel + 1 ) % _inputs; // [한국어] MQ 우회 성공 → 다음 사이클을 위해 다음 채널로 전진
	_read_stall = 0; // [한국어] MQ 우회로 스탈 해소 → 카운터 리셋
    } else {
      ++_read_stall; // [한국어] MQ 슬롯도 없음 → 완전 스탈 (이번 사이클 이 채널의 플릿 이동 없음). 다음 사이클에 다시 시도
      //cout << "stalling at input " << _cur_channel << " (count = " << _read_stall << ")" << endl; // [한국어] (주석처리됨) 스탈 진행 디버그 로그
    }
  }
}

/*
 * [한국어]
 * ChaosRouter::_OutputAdvance - 매칭 결과에 따라 플릿을 실제로 이동
 *
 * @param: 없음
 * @return: 없음
 *
 * _NextInterestingChannel() 직후 _InternalStep()에서 호출된다.
 * 이번 사이클에 _NextInterestingChannel()이 결정한 매칭을 실제 버퍼 이동으로 실현한다.
 *
 * 처리 경로 (두 루프):
 *
 * [루프 1] 입력 포트 → 크로스바 또는 MQ (i = 0 ~ _inputs-1):
 *   조건: _input_output_match[i] != -1 또는 _input_mq_match[i] != -1 이고,
 *         _input_frame[i]가 비어있지 않음
 *
 *   경로 A (입력→크로스바): _input_output_match[i] != -1
 *     - f->tail이면 _output_matched[out] = false (출력 포트 예약 해제)
 *     - _crossbar_pipe->Write(f, out): 크로스바 파이프라인에 플릿 주입
 *     - advanced = true
 *
 *   경로 B (입력→MQ): _input_mq_match[i] != -1 이고 MQ 슬롯이 가득 차지 않은 경우
 *     - f->head이면: _rf()로 MQ 라우팅 결정 갱신, _mq_age[mq]=0 리셋
 *       _multi_state 전이: empty→filling, leaving→shared
 *     - f->tail이면: _mq_matched[mq]=false (예약 해제)
 *       _multi_state 전이: filling→full, cut_through→leaving
 *     - _multi_queue[mq].push(f): MQ에 플릿 저장
 *     - advanced = true
 *
 *   이동 성공(advanced=true) 후:
 *     - _input_frame[i].pop(): 프레임 큐에서 플릿 제거
 *     - f->tail이면 매칭 정보 리셋(_input_output_match[i]=-1, _input_mq_match[i]=-1),
 *       _input_state 전이: leaving→empty, shared→filling
 *       shared→filling 시 새 패킷 head(f2)에 대해 _rf() 재호출 (_input_route[i] 갱신)
 *     - Credit 생성 후 _credit_queue[i]에 push (업스트림 흐름제어)
 *
 * [루프 2] MQ → 크로스바 (m = 0 ~ _multi_queue_size-1):
 *   조건: _multi_match[m] != -1 (이 슬롯이 출력에 매칭됨)
 *   - _multi_queue[m]에서 플릿 꺼내기
 *   - _crossbar_pipe->Write(f, _multi_match[m]): 크로스바에 플릿 주입
 *   - f->head이면 _multi_state 전이: filling→cut_through, full→leaving
 *   - f->tail이면: _output_matched[_multi_match[m]]=false, _multi_match[m]=-1 리셋
 *     _multi_state 전이: shared→filling, leaving→empty
 *   - 루프 끝: _mq_age[m]++ (모든 슬롯의 나이 증가)
 *
 * 호출 체인: _InternalStep() → [이 함수] → _crossbar_pipe->Write(), _multi_queue[].push()
 */
void ChaosRouter::_OutputAdvance( )
{
  Flit *f, *f2;   // [한국어] f: 현재 처리 중인 플릿 포인터, f2: shared 상태 해소 시 새 패킷 head 포인터
  Credit *c;      // [한국어] 업스트림으로 반환할 크레딧 포인터
  bool advanced;  // [한국어] 이번 사이클에 플릿이 성공적으로 이동됐는지 여부 (크레딧 생성 조건)
  int  mq;        // [한국어] 입력→MQ 매칭 시 대상 MQ 슬롯 번호 임시 저장

  _crossbar_pipe->WriteAll( 0 ); // [한국어] 크로스바 파이프라인의 모든 채널을 NULL(0)로 초기화 — 이번 사이클에 Write()가 없는 채널에는 NULL이 파이프라인에 들어감. 실제 플릿이 있는 채널만 이후 Write()로 덮어씀

  for ( int i = 0; i < _inputs; ++i ) { // [한국어] 모든 입력 포트를 순회하여 크로스바 또는 MQ로 플릿 이동
    if ( ( ( _input_output_match[i] != -1 ) ||  // [한국어] 이 입력이 크로스바 직접 전송에 매칭됐거나
	   ( _input_mq_match[i] != -1 ) ) &&     // [한국어] MQ 우회에 매칭됐고
	 ( !_input_frame[i].empty( ) ) ) {       // [한국어] 이 입력의 프레임 큐에 실제 플릿이 있는 경우에만 처리

      advanced = false;                  // [한국어] 이번 반복에서 아직 플릿 이동 없음으로 초기화
      f = _input_frame[i].front( );      // [한국어] 프레임 큐의 맨 앞 플릿 참조 (아직 pop하지 않음)

      /*if ( ! ) { // [한국어] (주석처리됨) 입력 큐가 비었는데 매칭된 경우 에러 처리 (현재 비활성화)
      } else {
	cout << "Input = " << i
	     << ", input_output_match = " << _input_output_match[i]
	     << ", input_mq_match = " << _input_mq_match[i] << endl;
	Error( "Input queue empty, but matched!" );
	}*/

      if ( _input_output_match[i] != -1 ) { // [한국어] 경로 A: 이 입력이 크로스바를 통해 출력 포트로 직접 전송하는 경우
	if ( f->tail ) { // [한국어] 패킷의 마지막 플릿(tail)이 크로스바로 나가는 경우 → 출력 포트 예약 해제
	  _output_matched[_input_output_match[i]] = false; // [한국어] 이 출력 포트 매칭 플래그 해제 — 다음 사이클부터 다른 패킷이 이 포트를 사용 가능
	}

	_crossbar_pipe->Write( f, _input_output_match[i] ); // [한국어] 플릿을 _input_output_match[i]번 출력 채널 방향의 크로스바 파이프라인에 주입 (_crossbar_delay 사이클 후 Read() 가능)

	if ( f->watch ) { // [한국어] 디버그 watch 모드: 이 플릿에 watch 플래그가 설정된 경우
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		      << "Flit traversing crossbar from input queue "
		      << i << " at "
		      << FullName() << endl
		      << *f; // [한국어] 시뮬레이션 시각, 라우터 이름, 소스 입력 큐 번호, 플릿 내용 출력
	}

	advanced = true; // [한국어] 크로스바 전송 성공 — 크레딧 생성 조건 충족

      } else if ( !_MultiQueueFull( _input_mq_match[i] ) ) { // [한국어] 경로 B: MQ 우회 매칭이 있고, 대상 MQ 슬롯이 가득 차지 않은 경우

	mq = _input_mq_match[i]; // [한국어] 대상 MQ 슬롯 번호를 지역 변수에 저장 (이후 반복적 참조를 위해)

	if ( f->head ) { // [한국어] 패킷의 첫 번째 플릿(head)이 MQ로 들어오는 경우 — 라우팅 결정 및 상태 초기화
	  _rf( this, f, i, _mq_route[mq], false ); // [한국어] head 플릿에 대해 라우팅 함수 호출 → _mq_route[mq]에 이 MQ 슬롯에 저장된 패킷의 선호 출력 포트 집합 저장. MQ 내에서 나중에 다른 출력으로 우회될 수 있으므로 별도 라우팅 결정 필요
	  _mq_age[mq] = 0; // [한국어] 새 패킷이 MQ 슬롯에 들어오므로 나이 카운터를 0으로 리셋 (oldest-first 스케줄링을 위한 기준점)

	  if ( _multi_state[mq] == empty ) {       // [한국어] MQ 슬롯이 비어있었던 경우 → head 수신 후 filling 상태로 전이
	    _multi_state[mq] = filling;             // [한국어] head 도착, tail 미도착 → filling
	  } else if ( _multi_state[mq] == leaving ) { // [한국어] MQ 슬롯이 leaving 상태(이전 패킷 tail이 아직 남음)에서 새 head 도착
	    _multi_state[mq] = shared;              // [한국어] 이전 패킷 leaving + 새 패킷 head → shared (두 패킷이 슬롯 공유)
	  } else {
	    Error( "Multi-queue received head while not empty or leaving!" ); // [한국어] 예상치 못한 MQ 상태에서 head 수신 — 논리 오류
	  }
	}

	if ( f->tail ) { // [한국어] 패킷의 마지막 플릿(tail)이 MQ에 들어오는 경우 — MQ 슬롯 예약 해제 및 상태 전이
	  _mq_matched[mq] = false; // [한국어] tail이 MQ에 들어왔으므로 이 슬롯의 입력→MQ 예약 해제 (다음 사이클부터 다른 입력이 이 슬롯을 사용 가능)

	  if ( _multi_state[mq] == filling ) {      // [한국어] head~tail 모두 MQ에 들어온 경우 → 패킷 완성
	    _multi_state[mq] = full;                 // [한국어] 패킷 전체가 MQ에 있음 → full (이제 크로스바로 나갈 수 있음)
	  } else if ( _multi_state[mq] == cut_through ) { // [한국어] MQ에서 이미 head가 크로스바로 나간 후 tail이 도착한 경우
	    _multi_state[mq] = leaving;              // [한국어] head는 나갔고 나머지(tail 포함)가 MQ에 있음 → leaving
	  } else {
	    Error( "Multi-queue received tail while not filling or cutting-through!" ); // [한국어] 예상치 못한 상태에서 MQ에 tail 수신 — 논리 오류
	  }
	}

	_multi_queue[mq].push( f ); // [한국어] 플릿을 MQ 슬롯의 FIFO 큐에 추가 저장 (head, 중간, tail 순서대로 축적)

	if ( f->watch ) { // [한국어] 디버그 watch 모드
	  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		      << "Flit stored in multiqueue at "
		      << FullName() << endl
		      << "State = " << _multi_state[mq] << endl
		      << *f; // [한국어] 시뮬레이션 시각, 라우터 이름, MQ 저장 이벤트, 현재 MQ 상태, 플릿 내용 출력
	}

	advanced = true; // [한국어] MQ 저장 성공 — 크레딧 생성 조건 충족
      }

      if ( advanced ) { // [한국어] 플릿이 크로스바 또는 MQ로 성공적으로 이동된 경우 후처리 수행
	_input_frame[i].pop( ); // [한국어] 이동된 플릿을 입력 프레임 큐에서 제거

	if ( f->tail ) { // last in packet, update state // [한국어] tail 플릿이 이동됐으면 패킷 전송 완료 → 입력 상태 갱신 및 매칭 리셋
	  if ( _input_state[i] == leaving ) { // [한국어] leaving 상태(head가 나가고 남은 플릿들이 이동 완료)에서 tail이 이동됨
	    _input_state[i] = empty;           // [한국어] 이 패킷의 모든 플릿이 이동 완료 → 입력 슬롯 비어있음 (다음 패킷 받을 준비)
	  } else if ( _input_state[i] == shared ) { // [한국어] shared 상태(이전 tail 남음 + 새 head 도착)에서 이전 패킷의 tail이 이동됨
	    _input_state[i] = filling;         // [한국어] 이전 패킷 완전히 나감, 새 패킷 head가 이미 있음 → filling (새 패킷의 tail 기다리는 중)
	    f2 = _input_frame[i].front( );     // [한국어] 입력 프레임 큐에 남아있는 새 패킷의 head 플릿 포인터 참조
	    // update routes
	    _rf( this, f2, i, _input_route[i], false ); // [한국어] 새 패킷의 head(f2)에 대해 라우팅 함수 재호출 → _input_route[i]를 새 패킷의 라우팅 결정으로 갱신 (이전 패킷의 라우팅 결정 덮어쓰기)
	  }

	  _input_output_match[i] = -1; // [한국어] 이 입력의 크로스바 직접 매칭 리셋 (-1: 미매칭)
	  _input_mq_match[i]     = -1; // [한국어] 이 입력의 MQ 우회 매칭 리셋 (-1: 미매칭)
	}

	c = Credit::New( );    // [한국어] 새 Credit 객체를 풀 할당자에서 생성 (업스트림에 버퍼 공간 1개 확보됐음을 알리기 위해)
	c->vc.insert(0);       // [한국어] ChaosRouter는 단일 VC(VC 0)만 사용하므로 크레딧의 VC 집합에 0 삽입 (업스트림이 VC 0으로 인지)
	_credit_queue[i].push( c ); // [한국어] 생성된 크레딧을 i번 입력 방향 크레딧 큐에 추가 (_SendCredits()에서 업스트림으로 전송됨)
      }
    }
  }

  for ( int m = 0; m < _multi_queue_size; ++m ) { // [한국어] 모든 MQ 슬롯을 순회하여 크로스바로 전송
    if ( _multi_match[m] != -1 ) { // [한국어] 이번 사이클에 이 MQ 슬롯이 출력 포트에 매칭된 경우
      if ( _multi_queue[m].empty( ) ) { // [한국어] 매칭됐는데 큐가 비어있으면 논리 오류
	cout << "State = " << _multi_state[m] << endl; // [한국어] 현재 MQ 슬롯 상태 출력 (디버그)
	Error( "Multi queue empty, but matched!" ); // [한국어] MQ 큐가 비어있는데 매칭이 설정된 상태 — 스케줄링 버그
      }
      assert( !_multi_queue[m].empty( ) ); // [한국어] 방어적 assert: 위 Error()와 중복이지만 컴파일러 최적화/디버그 빌드에서 추가 검증
      f = _multi_queue[m].front( ); // [한국어] MQ 슬롯의 맨 앞 플릿 참조
      _multi_queue[m].pop( );       // [한국어] 맨 앞 플릿을 MQ에서 제거

      _crossbar_pipe->Write( f, _multi_match[m] ); // [한국어] MQ에서 꺼낸 플릿을 _multi_match[m]번 출력 채널 방향의 크로스바 파이프라인에 주입

      if ( f->watch ) { // [한국어] 디버그 watch 모드
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		    << "Flit traversing crossbar from multiqueue slot "
		    << m << " at "
		    << FullName() << endl
		    << *f; // [한국어] 시뮬레이션 시각, 라우터 이름, MQ 슬롯 번호, 플릿 내용 출력
      }

      if ( f->head ) { // [한국어] MQ에서 head 플릿이 크로스바로 나가는 경우 → MQ 상태 전이
	if ( _multi_state[m] == filling ) {       // [한국어] MQ가 filling 상태(head 도착, tail 아직 수신 중)에서 head가 크로스바로 나감
	  _multi_state[m] = cut_through;          // [한국어] head는 나갔고 tail은 아직 MQ로 들어오는 중 → cut_through
	} else if ( _multi_state[m] == full ) {   // [한국어] MQ가 full 상태(패킷 전체 있음)에서 head가 크로스바로 나감
	  _multi_state[m] = leaving;              // [한국어] head는 나갔고 나머지 플릿(tail 포함)은 MQ에 남아있음 → leaving
	} else {
	  Error( "Multi-queue sent head while not filling or full!" ); // [한국어] 예상치 못한 상태에서 MQ head 전송 — 논리 오류
	}
      }

      if ( f->tail ) { // [한국어] MQ에서 tail 플릿이 크로스바로 나가는 경우 → 패킷 MQ 전송 완료
	_output_matched[_multi_match[m]] = false; // [한국어] 이 출력 포트의 매칭 플래그 해제 (다음 사이클부터 다른 패킷 사용 가능)
	_multi_match[m] = -1;                     // [한국어] 이 MQ 슬롯의 출력 매칭 리셋 (-1: 미매칭, 다음 사이클에 새 매칭 가능)

	if ( _multi_state[m] == shared ) {       // [한국어] MQ가 shared 상태(이전 패킷 leaving 중 새 head 도착)에서 이전 패킷 tail이 나감
	  _multi_state[m] = filling;             // [한국어] 이전 패킷 완전히 나감, 새 패킷 head가 MQ에 남음 → filling
	} else if ( _multi_state[m] == leaving ) { // [한국어] MQ가 leaving 상태(head 나가고 나머지 있음)에서 tail 마지막 플릿이 나감
	  _multi_state[m] = empty;               // [한국어] 패킷 전체 전송 완료 → MQ 슬롯 비어있음 (다음 패킷 수용 준비)
	} else {
	  cout << "State = " << _multi_state[m] << endl; // [한국어] 오류 발생 시 현재 MQ 상태 출력
	  cout << *f;                                     // [한국어] 오류 플릿 내용 출력
	  Error( "Multi-queue sent tail while not leaving or shared!" ); // [한국어] 예상치 못한 상태에서 MQ tail 전송 — 논리 오류
	}
      }
    }

    _mq_age[m]++; // [한국어] 매 사이클 모든 MQ 슬롯의 나이를 1 증가 — 비어있는 슬롯도 나이가 증가하나 oldest-first 탐색 시 empty/leaving 슬롯은 제외됨
  }
}


/*
 * [한국어]
 * ChaosRouter::_SendFlits - 크로스바 파이프라인을 통과한 플릿을 출력 채널로 전송
 *
 * @param: 없음
 * @return: 없음
 *
 * WriteOutputs()에서 호출. _crossbar_pipe->Read(output)으로 이번 사이클에 파이프라인을
 * 완전히 통과한 플릿을 읽어 _output_frame[output]에 적재하고,
 * 다운스트림 버퍼 공간이 있는 경우(_next_queue_cnt < _buffer_size) 즉시 전송한다.
 *
 * 처리 순서 (각 출력 포트별):
 *   1. _crossbar_pipe->Read(output): 크로스바를 통과한 플릿 꺼냄 (없으면 NULL)
 *   2. 플릿이 있으면 _output_frame[output]에 push, f->hops++ (홉 카운트 증가)
 *   3. _next_queue_cnt[output] < _buffer_size && _output_frame[output] 비어있지 않으면:
 *      _output_channels[output]->Send(_output_frame[output].front()): 출력 채널로 전송
 *      _output_frame[output].pop(): 전송된 플릿 큐에서 제거
 *      ++_next_queue_cnt[output]: 다운스트림에 전송된 플릿 수 카운터 증가
 *
 * 흐름제어: _next_queue_cnt[output] >= _buffer_size이면 전송하지 않음 (다운스트림 버퍼 가득 참).
 * 다운스트림이 크레딧을 보내면 ReadInputs()에서 _next_queue_cnt 감소 → 다음 사이클에 전송 재개.
 *
 * 호출 체인: WriteOutputs() → [이 함수] → _crossbar_pipe->Read(), _output_channels[]->Send()
 */
void ChaosRouter::_SendFlits( )
{
  for ( int output = 0; output < _outputs; ++output ) { // [한국어] 모든 출력 포트를 순회하며 크로스바 파이프라인에서 플릿 꺼내 전송
    Flit *f = _crossbar_pipe->Read( output ); // [한국어] output번 채널의 크로스바 파이프라인에서 이번 사이클에 파이프를 완전히 통과한 플릿 꺼냄 (_crossbar_delay 사이클 경과한 플릿만 반환, 없으면 NULL)

    if ( f ) { // [한국어] 크로스바 파이프라인에서 플릿이 도착한 경우
      _output_frame[output].push( f ); // [한국어] 도착한 플릿을 출력 프레임 FIFO에 추가 (전송 대기)
      f->hops++; // [한국어] 이 플릿이 라우터 하나를 더 통과했으므로 홉 카운트 1 증가 (지연 추적, 라우팅 우회 감지에 사용)
    }

    if ( ( _next_queue_cnt[output] < _buffer_size ) &&  // [한국어] 다운스트림 버퍼에 공간이 있음 (전송된 플릿 수가 최대 버퍼 크기 미만)
	 ( !_output_frame[output].empty( ) ) ) {         // [한국어] 출력 프레임 큐에 전송 대기 중인 플릿이 있음
      _output_channels[output]->Send( _output_frame[output].front( ) ); // [한국어] 출력 프레임 큐의 맨 앞 플릿을 output번 출력 채널로 전송 → 다운스트림 라우터 또는 이젝션 포인트로 전달
      _output_frame[output].pop( );  // [한국어] 전송한 플릿을 출력 프레임 큐에서 제거
      ++_next_queue_cnt[output];     // [한국어] 다운스트림 버퍼에 하나 더 전송됐으므로 카운터 1 증가 (크레딧 수신 시 감소)
    }
  }
}

/*
 * [한국어]
 * ChaosRouter::_SendCredits - 입력 채널로 크레딧 반환 (업스트림 흐름제어)
 *
 * @param: 없음
 * @return: 없음
 *
 * WriteOutputs()에서 호출. _OutputAdvance()에서 플릿 이동 성공 시 생성되어
 * _credit_queue[input]에 저장된 크레딧을 업스트림 라우터(또는 SM의 인젝션 포인트)로 전송.
 *
 * 처리: 각 입력 포트별로 _credit_queue[input]이 비어있지 않으면
 *   front()로 크레딧 꺼내 _input_credits[input]->Send()로 전송하고 pop().
 *
 * 업스트림 라우터는 이 크레딧을 받으면 이 라우터의 버퍼에 공간이 생겼음을 인지하고
 * 다음 사이클부터 새 플릿을 전송할 수 있다.
 *
 * ChaosRouter에서 크레딧의 VC 번호는 항상 0 (단일 VC 사용).
 *
 * 호출 체인: WriteOutputs() → [이 함수] → _input_credits[]->Send()
 */
void ChaosRouter::_SendCredits( )
{
  for ( int input = 0; input < _inputs; ++input ) { // [한국어] 모든 입력 포트를 순회하며 대기 중인 크레딧 전송
    if ( !_credit_queue[input].empty( ) ) { // [한국어] 이 입력 방향으로 반환할 크레딧이 있는 경우
      Credit *c = _credit_queue[input].front( ); // [한국어] 크레딧 큐의 맨 앞 크레딧 포인터 가져옴
      _credit_queue[input].pop( );               // [한국어] 크레딧 큐에서 해당 크레딧 제거
      _input_credits[input]->Send( c );          // [한국어] input번 입력 방향의 크레딧 채널로 크레딧 전송 → 업스트림 라우터가 ReadInputs()에서 수신
    }
  }
}

/*
 * [한국어]
 * ChaosRouter::Display - 라우터 내부 상태를 출력 스트림에 덤프 (미구현)
 *
 * @param os: 출력 대상 스트림 (기본값: cout)
 * @return: 없음
 *
 * 디버깅 또는 상태 스냅샷 목적으로 호출될 수 있는 함수.
 * 현재 구현이 비어있으므로 호출해도 아무 출력이 없다.
 * IQRouter 등 다른 라우터는 이 함수에서 버퍼 상태, 매칭 정보, VC 상태 등을 출력한다.
 * ChaosRouter에서는 향후 디버그 필요 시 구현 예정이거나 미지원 상태.
 *
 * 호출 체인: 외부 디버그 코드 또는 Network::Display() → [이 함수]
 */
void ChaosRouter::Display( ostream & os ) const
{
  // [한국어] 현재 미구현 — 함수 본체가 비어있음. os 파라미터도 사용되지 않음
}
