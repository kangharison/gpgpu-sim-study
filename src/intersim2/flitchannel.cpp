// $Id: flitchannel.cpp 5516 2013-10-06 02:14:48Z dub $

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

// ----------------------------------------------------------------------
//
//  File Name: flitchannel.cpp
//  Author: James Balfour, Rebecca Schultz
//
// ----------------------------------------------------------------------

/*
 * [한국어 설명] NoC 물리 링크(Flit 채널) 구현 (flitchannel.cpp)
 *
 * === 파일의 역할 ===
 * FlitChannel 클래스의 구현 파일. 두 라우터 사이 물리 링크의 행동을 시뮬레이션한다.
 * 핵심 역할은 두 가지:
 *   1. Flit 전달 통계 수집 (활동/유휴 사이클 카운터)
 *   2. Channel<Flit> 기반 클래스에 위임하여 다중 사이클 지연 처리
 * Send() → ReadInputs() → (delay 사이클 대기) → WriteOutputs() 순서로 동작한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPUTrafficManager::_Step() 내 매 사이클 실행:
 *   _net[subnet]->ReadInputs()  → 모든 FlitChannel::ReadInputs() 호출
 *   _net[subnet]->Evaluate()    → 라우터 파이프라인 처리 + Router::WriteFlit() → FlitChannel::Send()
 *   _net[subnet]->WriteOutputs() → 모든 FlitChannel::WriteOutputs() 호출
 *
 * === 타 모듈과의 연결 ===
 * - channel.hpp의 Channel<Flit>: 지연 큐(_queue) 관리 로직 상속
 * - router.hpp의 Router: 소스/싱크 라우터 참조 (lookahead 라우팅에서 사용)
 * - globals.hpp의 gWatchOut: watch 플래그 활성화 시 추적 로그 출력 스트림
 *
 * === 주요 함수/구조체 요약 ===
 * - FlitChannel(): _active 벡터를 0으로 초기화, NULL 소스/싱크로 초기화
 * - SetSource/SetSink(): Network 구성 시 라우터-채널 연결 설정
 * - Send(): Flit 유효 여부에 따라 활동/유휴 카운터 업데이트 후 기반 클래스에 위임
 * - ReadInputs(): watch flit 추적 메시지 출력 후 기반 클래스에 위임
 * - WriteOutputs(): 기반 클래스에 위임 후 완료된 flit의 watch 메시지 출력
 */

#include "flitchannel.hpp"  // [한국어] FlitChannel 클래스 선언

#include <iostream>  // [한국어] cout 표준 출력
#include <iomanip>   // [한국어] setw 등 출력 서식 지정 (현재 직접 사용 안 함)

#include "router.hpp"   // [한국어] Router 타입 정의 (GetID, FullName 등)
#include "globals.hpp"  // [한국어] gWatchOut, GetSimTime() 등 전역 변수

// ----------------------------------------------------------------------
//  $Author: jbalfour $
//  $Date: 2007/06/27 23:10:17 $
//  $Id: flitchannel.cpp 5516 2013-10-06 02:14:48Z dub $
// ----------------------------------------------------------------------

/*
 * [한국어]
 * FlitChannel::FlitChannel() - Flit 채널 초기화
 *
 * @parent: 부모 Module 포인터 (보통 Network 객체)
 * @name: 채널 이름 (예: "net0:inject_0")
 * @classes: 트래픽 클래스 수 (_active 벡터 크기)
 *
 * 초기화 목록:
 *   - Channel<Flit>(parent, name): 기반 클래스 초기화 (지연 큐 생성)
 *   - _routerSource/Sink = NULL: 연결 전 소스/싱크 미설정
 *   - _routerSourcePort/_routerSinkPort = -1: 포트 미설정
 *   - _idle = 0: 유휴 카운터 초기화
 * _active 벡터는 클래스 수만큼 0으로 초기화.
 *
 * 호출 체인: Network 생성자 → [이 함수]
 */
FlitChannel::FlitChannel(Module * parent, string const & name, int classes)
: Channel<Flit>(parent, name), _routerSource(NULL), _routerSourcePort(-1),
  _routerSink(NULL), _routerSinkPort(-1), _idle(0) {
  _active.resize(classes, 0); // [한국어] 클래스 수만큼 활동 카운터 배열 초기화 (모두 0)
}

/*
 * [한국어]
 * FlitChannel::SetSource() - 소스 라우터와 출구 포트를 설정한다
 *
 * @router: 이 채널에 flit을 주입하는 상류 라우터
 * @port: 해당 라우터의 출구 포트 번호
 *
 * Network 생성 시 라우터 간 연결을 설정하는 단계에서 호출.
 *
 * 호출 체인: Network::_ComputeSize() 또는 생성자 → [이 함수]
 */
void FlitChannel::SetSource(Router const * const router, int port) {
  _routerSource = router;     // [한국어] 소스 라우터 포인터 저장
  _routerSourcePort = port;   // [한국어] 소스 라우터의 출구 포트 번호 저장
}

/*
 * [한국어]
 * FlitChannel::SetSink() - 싱크 라우터와 입구 포트를 설정한다
 *
 * @router: 이 채널에서 flit을 수신하는 하류 라우터
 * @port: 해당 라우터의 입구 포트 번호
 *
 * GPUTrafficManager::_Step()의 lookahead 라우팅에서 GetSink()로 다음 라우터를
 * 찾아 _rf(router, flit, in_channel, ...)를 사전 계산할 때 사용된다.
 *
 * 호출 체인: Network 생성자 → [이 함수]
 */
void FlitChannel::SetSink(Router const * const router, int port) {
  _routerSink = router;     // [한국어] 싱크 라우터 포인터 저장
  _routerSinkPort = port;   // [한국어] 싱크 라우터의 입구 포트 번호 저장
}

/*
 * [한국어]
 * FlitChannel::Send() - Flit을 채널에 주입한다 (활동 통계 업데이트 포함)
 *
 * @f: 전달할 Flit 포인터 (NULL이면 빈 슬롯/유휴 사이클)
 *
 * f가 유효하면 해당 트래픽 클래스(f->cl)의 _active[cl]을 증가시켜
 * 스위칭 활동도를 기록한다. f가 NULL이면 _idle을 증가시킨다.
 * 이후 Channel<Flit>::Send()에 위임하여 지연 큐(_queue)에 삽입한다.
 *
 * 호출 체인: Router 파이프라인 WriteOutputs → [이 함수] → Channel<Flit>::Send()
 */
void FlitChannel::Send(Flit * f) {
  if(f) {                     // [한국어] 유효한 flit이면 활동 카운터 증가
    ++_active[f->cl];         // [한국어] 해당 트래픽 클래스의 활성 전송 사이클 기록
  } else {                    // [한국어] NULL flit이면 유휴 사이클 카운터 증가
    ++_idle;
  }
  Channel<Flit>::Send(f);     // [한국어] 기반 클래스에 위임: 지연 큐에 flit 삽입
}

/*
 * [한국어]
 * FlitChannel::ReadInputs() - 이번 사이클에 채널로 들어온 Flit을 캡처한다
 *
 * watch 플래그가 설정된 flit이 채널을 막 진입하면 추적 메시지를 gWatchOut에 출력.
 * 이후 Channel<Flit>::ReadInputs()에 위임하여 지연 큐에 Flit을 등록한다.
 *
 * _input: Channel<Flit>의 현재 입력 포인터 (Send()로 설정된 값)
 * _delay: Channel<Flit>의 지연 사이클 수 (설정 파일의 wire_delay 등)
 *
 * 호출 체인: Network::ReadInputs() → [이 함수] → Channel<Flit>::ReadInputs()
 */
void FlitChannel::ReadInputs() {
  Flit const * const & f = _input; // [한국어] 현재 채널 입력 flit 참조 (Channel<Flit>의 _input)
  if(f && f->watch) { // [한국어] watch 플래그가 있는 flit이면 추적 메시지 출력
    *gWatchOut << GetSimTime() << " | " << FullName() << " | " // [한국어] 현재 시뮬레이션 시각과 채널 이름
	       << "Beginning channel traversal for flit " << f->id // [한국어] 채널 진입 flit ID
	       << " with delay " << _delay  // [한국어] 이 채널의 전파 지연 사이클 수
	       << "." << endl;
  }
  Channel<Flit>::ReadInputs(); // [한국어] 기반 클래스에 위임: 지연 큐에 flit 추가
}

/*
 * [한국어]
 * FlitChannel::WriteOutputs() - 지연이 경과된 Flit을 싱크에 배달한다
 *
 * Channel<Flit>::WriteOutputs()를 먼저 호출하여 _output을 업데이트하고,
 * watch 플래그가 있는 flit이 배달 완료되면 추적 메시지를 출력한다.
 *
 * _output: 지연 큐에서 꺼내져 싱크 라우터에 배달되는 Flit 포인터.
 * NULL이면 이번 사이클에 배달할 flit 없음.
 *
 * 호출 체인: Network::WriteOutputs() → [이 함수] → Channel<Flit>::WriteOutputs()
 */
void FlitChannel::WriteOutputs() {
  Channel<Flit>::WriteOutputs(); // [한국어] 기반 클래스: 지연 큐에서 flit을 꺼내 _output에 배달
  if(_output && _output->watch) { // [한국어] 이번 사이클에 배달된 flit이 watch 대상이면
    *gWatchOut << GetSimTime() << " | " << FullName() << " | " // [한국어] 현재 시각과 채널 이름
	       << "Completed channel traversal for flit " << _output->id // [한국어] 채널 통과 완료 flit ID
	       << "." << endl;
  }
}
