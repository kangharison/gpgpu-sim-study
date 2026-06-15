// $Id: channel.hpp 5188 2012-08-30 00:31:31Z dub $

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

//////////////////////////////////////////////////////////////////////
//
//  File Name: channel.hpp
//
//  The Channel models a generic channel with a multi-cycle
//   transmission delay. The channel latency can be specified as
//   an integer number of simulator cycles.
//
/////

/*
 * [한국어 설명] NoC 물리 링크(채널) 추상 기반 클래스 (channel.hpp)
 *
 * === 파일의 역할 ===
 * Channel<T>는 intersim2 NoC 시뮬레이터에서 라우터 사이의 물리 링크를 모델링하는
 * 템플릿 기반 클래스이다. 정수 사이클 단위의 전송 지연(_delay)을 지원하며,
 * 매 사이클마다 데이터가 입력(Send)되고 _delay 사이클 후에 출력(Receive)된다.
 * 이 클래스는 두 가지 구체 채널 타입의 기반이 된다:
 *   - FlitChannel: 플릿(Flit*)을 전달하는 데이터 채널
 *   - CreditChannel: 크레딧(Credit*)을 전달하는 흐름 제어 채널
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 계층: GPU-Sim → icnt_wrapper → intersim2 [여기]
 * intersim2 내부 위치: 라우터 간 데이터/크레딧 전송 물리 계층.
 * 시뮬레이션 사이클 순서:
 *   1. ReadInputs(): Send()로 들어온 데이터를 _wait_queue에 삽입
 *   2. Evaluate(): 아무 동작 없음 (기반 클래스에서 빈 구현)
 *   3. WriteOutputs(): 도착 시각이 현재 사이클과 일치하면 _output으로 이동
 * 업스트림 라우터가 Send()를 호출하고, 다운스트림 라우터가 Receive()로 꺼낸다.
 * 실행 컨텍스트: 호스트 CPU 싱글 스레드, TimedModule 프레임워크 내 매 사이클.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - globals.hpp:       GetSimTime() — 현재 시뮬레이션 사이클 번호
 *   - module.hpp:        Module 기반 클래스 — 계층형 이름/에러 지원
 *   - timed_module.hpp:  TimedModule 기반 클래스 — ReadInputs/Evaluate/WriteOutputs 인터페이스
 * 이 파일을 사용하는 모듈:
 *   - flitchannel.hpp/cc: FlitChannel (Channel<Flit>) — 라우터 간 플릿 전달
 *   - network.hpp/cc:     CreditChannel (Channel<Credit>) — 크레딧 역방향 전달
 *   - router.hpp:         FlitChannel*, CreditChannel* 보유
 *
 * === 주요 함수/구조체 요약 ===
 * Channel<T>(parent, name): 생성자 — _delay=1, _input=0, _output=0 초기화
 * SetLatency(cycles):       링크 레이턴시 설정 (양수가 아니면 에러)
 * Send(data):               현재 사이클의 입력 데이터를 _input에 임시 저장
 * Receive():                _output 포인터 반환 (WriteOutputs 후 유효)
 * ReadInputs():             _input → _wait_queue 삽입, 도착 시각 계산
 * WriteOutputs():           _wait_queue 최상단의 도착 시각 확인 후 _output으로 이동
 */

#ifndef _CHANNEL_HPP
#define _CHANNEL_HPP

#include <queue>   // [한국어] std::queue<pair<int, T*>> — 지연 전달 대기열(_wait_queue)에 사용
#include <cassert> // [한국어] assert() — 사이클 정확성 검사 (WriteOutputs의 시각 일치 검사)

#include "globals.hpp"       // [한국어] GetSimTime() — 현재 시뮬레이션 사이클 번호 반환
#include "module.hpp"        // [한국어] Module 기반 클래스 — Error(), FullName() 등 제공
#include "timed_module.hpp"  // [한국어] TimedModule 기반 클래스 — ReadInputs/Evaluate/WriteOutputs 인터페이스 제공

using namespace std; // [한국어] queue, pair 등 STL 타입을 namespace 없이 사용하기 위한 선언

/*
 * [한국어]
 * Channel<T> - 멀티 사이클 전송 지연을 가진 물리 링크 모델 (템플릿 클래스)
 *
 * T는 전달되는 데이터 타입 (Flit* 또는 Credit*).
 * _delay 사이클의 파이프라인 지연을 _wait_queue로 구현한다.
 * TimedModule을 상속하여 ReadInputs/Evaluate/WriteOutputs 인터페이스를 따른다.
 *
 * 상속: TimedModule → Module — 계층형 이름, ReadInputs/WriteOutputs 가상 함수 제공
 */
template<typename T>
class Channel : public TimedModule {
public:
  /*
   * [한국어]
   * Channel 생성자 — 링크 기본 상태 초기화
   *
   * @parent: 모듈 계층 부모 (라우터 또는 네트워크 객체)
   * @name:   이 채널 모듈의 이름 문자열 (예: "chan_0_1")
   * @return: 없음 (생성자)
   *
   * _delay = 1 (기본 1 사이클 지연)
   * _input = 0, _output = 0 (초기에는 데이터 없음)
   * _wait_queue는 빈 큐로 초기화됨.
   *
   * 호출 체인:
   *   FlitChannel::FlitChannel() 또는 CreditChannel::CreditChannel() → [Channel<T>::Channel()]
   */
  Channel(Module * parent, string const & name);
  virtual ~Channel() {}

  // Physical Parameters

  /*
   * [한국어]
   * SetLatency - 이 링크의 전송 지연 사이클 수 설정
   *
   * @cycles: 설정할 지연 사이클 수 (양의 정수만 허용)
   * @return: 없음
   *
   * cycles <= 0이면 "Channel must have positive delay." 에러.
   * gpgpusim.config의 "channel_latency" 또는 네트워크 토폴로지 설정으로 지정.
   *
   * 호출 체인:
   *   Network::_BuildNet() 또는 초기화 루틴 → [Channel<T>::SetLatency()]
   */
  void SetLatency(int cycles);

  /*
   * [한국어]
   * GetLatency - 현재 설정된 전송 지연 사이클 수 반환 (인라인)
   *
   * @return: _delay (1 이상의 양의 정수)
   *
   * 디버그 출력 또는 네트워크 분석 시 링크 지연 확인에 사용.
   */
  int GetLatency() const { return _delay ; } // [한국어] 현재 설정된 채널 지연 사이클 수 반환

  // Send data

  /*
   * [한국어]
   * Send - 이번 사이클에 전송할 데이터를 채널에 주입 (가상)
   *
   * @data: 전송할 데이터 포인터 (Flit* 또는 Credit*)
   * @return: 없음
   *
   * _input에 data를 임시 저장한다. 실제 _wait_queue 삽입은
   * 이후 ReadInputs()에서 수행된다.
   * 한 사이클에 Send()가 두 번 호출되면 두 번째 값이 _input을 덮어쓴다
   * (동일 사이클에 두 플릿 전송은 NoC 정책상 발생하지 않아야 함).
   *
   * 호출 체인:
   *   업스트림 라우터의 WriteOutputs() → [Channel<T>::Send(data)]
   */
  virtual void Send(T * data);

  // Receive data

  /*
   * [한국어]
   * Receive - 이번 사이클에 도착한 데이터 포인터 반환 (가상)
   *
   * @return: _output 포인터 — WriteOutputs() 이후 유효한 데이터 또는 NULL(도착 없음)
   *
   * 다운스트림 라우터가 ReadInputs()에서 이 함수를 호출해
   * 도착한 플릿/크레딧을 수신한다.
   *
   * 호출 체인:
   *   다운스트림 라우터의 ReadInputs() → [Channel<T>::Receive()] → 반환된 포인터로 처리
   */
  virtual T * Receive();

  /*
   * [한국어]
   * ReadInputs - Send()된 데이터를 _wait_queue에 삽입 (가상)
   *
   * @return: 없음 (TimedModule 인터페이스 구현)
   *
   * _input이 NULL이 아니면 도착 시각 = GetSimTime() + _delay - 1 을 계산해
   * (도착 시각, 데이터 포인터) 쌍을 _wait_queue에 push한다.
   * _input을 0으로 초기화한다.
   * 매 사이클 시작 시 TimedModule 프레임워크가 자동 호출.
   *
   * 호출 체인:
   *   TimedModule 프레임워크 → [Channel<T>::ReadInputs()] → _wait_queue.push()
   */
  virtual void ReadInputs();

  /*
   * [한국어]
   * Evaluate - 아무 동작 없음 (기반 클래스 빈 구현)
   *
   * Channel은 조합 로직 없이 순수한 지연 파이프라인이므로
   * Evaluate() 단계에서 수행할 작업이 없다.
   */
  virtual void Evaluate() {}

  /*
   * [한국어]
   * WriteOutputs - 도착 시각이 된 데이터를 _output으로 이동 (가상)
   *
   * @return: 없음 (TimedModule 인터페이스 구현)
   *
   * _output을 0으로 초기화한 뒤 _wait_queue가 비어 있으면 반환.
   * 큐 최상단 항목의 도착 시각이 현재 사이클보다 미래이면 반환.
   * 현재 사이클과 일치하면(assert로 검증) _output에 데이터 포인터를 설정하고
   * 큐에서 pop한다.
   * 매 사이클 끝에 TimedModule 프레임워크가 자동 호출.
   *
   * 호출 체인:
   *   TimedModule 프레임워크 → [Channel<T>::WriteOutputs()] → _output 설정
   */
  virtual void WriteOutputs();

protected:
  int _delay;
  /* 채널 전송 지연 사이클 수.
   * 설정자: Channel() 생성자에서 1로 초기화; SetLatency()로 변경.
   * 읽는 자: ReadInputs()에서 도착 시각 계산 (GetSimTime() + _delay - 1).
   * 값 범위: 1 이상 양의 정수 (SetLatency에서 <= 0이면 에러).
   * 동기화: 생성 후 SetLatency() 1회 호출 이후 읽기 전용. */

  T * _input;
  /* 이번 사이클에 Send()로 주입된 데이터 포인터 (임시 보관).
   * 설정자: Send()에서 data 포인터 저장.
   * 읽는 자: ReadInputs()에서 _wait_queue로 이동 후 0 초기화.
   * 값 범위: NULL(전송 없음) 또는 유효한 T 포인터.
   * 동기화: 단일 스레드 — 불필요. */

  T * _output;
  /* 이번 사이클에 Receive()로 꺼낼 수 있는 데이터 포인터.
   * 설정자: WriteOutputs()에서 _wait_queue 최상단을 꺼내 설정;
   *         WriteOutputs() 시작 시 0으로 초기화.
   * 읽는 자: Receive() — 다운스트림 라우터가 ReadInputs()에서 호출.
   * 값 범위: NULL(도착한 데이터 없음) 또는 유효한 T 포인터.
   * 동기화: 단일 스레드 — 불필요. */

  queue<pair<int, T *> > _wait_queue;
  /* 전송 중인 데이터의 대기 큐. 각 원소는 (도착 예정 사이클, 데이터 포인터) 쌍.
   * 설정자: ReadInputs()에서 (GetSimTime() + _delay - 1, _input) 쌍 push.
   * 읽는 자: WriteOutputs()에서 front()의 시각이 현재 사이클이면 pop하여 _output에 저장.
   * 값 범위: 비어 있거나 1개 이상의 (시각, 포인터) 쌍. 도착 시각은 단조 증가.
   * 동기화: 단일 스레드 — 불필요. */

};

/*
 * [한국어]
 * Channel<T>::Channel - 채널 기본 상태 초기화 생성자 구현
 *
 * @parent: 모듈 계층 부모
 * @name:   모듈 이름
 *
 * TimedModule(parent, name) 초기화 후 _delay=1, _input=0, _output=0 설정.
 * 템플릿 함수이므로 헤더에 정의됨.
 *
 * 호출 체인:
 *   FlitChannel() 또는 CreditChannel() → [Channel<T>::Channel()]
 */
template<typename T>
Channel<T>::Channel(Module * parent, string const & name)
  : TimedModule(parent, name), _delay(1), _input(0), _output(0) { // [한국어] 지연 1사이클, 입력/출력 포인터 NULL로 초기화
}

/*
 * [한국어]
 * Channel<T>::SetLatency - 채널 전송 지연 사이클 설정 구현
 *
 * @cycles: 설정할 지연 사이클 수
 *
 * 0 이하이면 에러, 양수이면 _delay에 저장.
 *
 * 호출 체인:
 *   Network 초기화 → [Channel<T>::SetLatency()]
 */
template<typename T>
void Channel<T>::SetLatency(int cycles) {
  if(cycles <= 0) { // [한국어] 0 또는 음수 지연은 물리적으로 불가능 — 에러
    Error("Channel must have positive delay."); // [한국어] 유효하지 않은 지연 설정 에러 출력 후 종료
  }
  _delay = cycles ; // [한국어] 유효한 지연 사이클 수를 _delay에 저장
}

/*
 * [한국어]
 * Channel<T>::Send - 이번 사이클의 전송 데이터를 _input에 임시 저장 구현
 *
 * @data: 전송할 데이터 포인터
 *
 * _input에 data를 저장한다. 실제 지연 처리는 ReadInputs()에서 수행.
 *
 * 호출 체인:
 *   업스트림 라우터 WriteOutputs() → [Channel<T>::Send(data)]
 */
template<typename T>
void Channel<T>::Send(T * data) {
  _input = data; // [한국어] 이번 사이클의 전송 데이터를 _input에 저장 (ReadInputs에서 큐에 삽입됨)
}

/*
 * [한국어]
 * Channel<T>::Receive - 현재 사이클에 도착한 데이터 반환 구현
 *
 * @return: _output 포인터 (WriteOutputs 이후 유효, 도착 없으면 NULL/0)
 *
 * WriteOutputs()가 _wait_queue에서 도착 시각이 된 데이터를 _output에 설정한 이후에
 * 다운스트림 라우터의 ReadInputs()에서 호출하여 데이터를 수신한다.
 *
 * 호출 체인:
 *   다운스트림 라우터 ReadInputs() → [Channel<T>::Receive()]
 */
template<typename T>
T * Channel<T>::Receive() {
  return _output; // [한국어] WriteOutputs()에서 설정된 도착 데이터 포인터 반환 (NULL이면 이번 사이클 도착 없음)
}

/*
 * [한국어]
 * Channel<T>::ReadInputs - Send()된 데이터를 _wait_queue에 삽입 구현
 *
 * _input이 비어 있지 않으면 도착 예정 시각(GetSimTime() + _delay - 1)을 계산해
 * _wait_queue에 push하고 _input을 0으로 초기화한다.
 * "-1"인 이유: ReadInputs → Evaluate → WriteOutputs 순서로 같은 사이클에 처리되므로
 *             _delay 사이클 후의 WriteOutputs()에서 꺼내려면 _delay-1을 더해야 한다.
 *
 * 호출 체인:
 *   TimedModule 프레임워크(매 사이클 시작) → [Channel<T>::ReadInputs()]
 */
template<typename T>
void Channel<T>::ReadInputs() {
  if(_input) { // [한국어] 이번 사이클에 Send()로 데이터가 주입되었는지 확인 (NULL이면 전송 없음)
    _wait_queue.push(make_pair(GetSimTime() + _delay - 1, _input));
    // [한국어] (도착 예정 사이클, 데이터 포인터) 쌍을 대기 큐에 삽입
    // GetSimTime() + _delay - 1: ReadInputs/WriteOutputs가 같은 사이클에 실행되므로 -1 보정
    _input = 0; // [한국어] _input 초기화 — 다음 사이클에 Send()가 없으면 NULL 유지
  }
}

/*
 * [한국어]
 * Channel<T>::WriteOutputs - 도착 시각이 된 데이터를 _output으로 꺼내는 구현
 *
 * 1. _output = 0으로 초기화 (이전 사이클의 출력 무효화)
 * 2. _wait_queue가 비어 있으면 반환
 * 3. 큐 최상단(front())의 도착 시각(time)이 현재 사이클보다 미래이면 반환
 * 4. 현재 사이클과 정확히 일치하면(assert) _output에 데이터 포인터 저장 후 pop
 *
 * assert(GetSimTime() == time)는 도착 시각 계산이 정확한지 검증한다.
 * 시각이 맞지 않으면 큐 삽입 로직 또는 사이클 진행에 버그가 있는 것.
 *
 * 호출 체인:
 *   TimedModule 프레임워크(매 사이클 끝) → [Channel<T>::WriteOutputs()]
 */
template<typename T>
void Channel<T>::WriteOutputs() {
  _output = 0; // [한국어] 이전 사이클의 출력 포인터 무효화 (매 사이클마다 갱신됨)
  if(_wait_queue.empty()) { // [한국어] 대기 중인 데이터가 없으면 아무것도 출력하지 않음
    return; // [한국어] 빈 큐 — 이번 사이클 출력 없음
  }
  pair<int, T *> const & item = _wait_queue.front(); // [한국어] 가장 먼저 도착 예정인 항목 참조 (FIFO)
  int const & time = item.first;                     // [한국어] 이 항목의 도착 예정 사이클 번호
  if(GetSimTime() < time) { // [한국어] 아직 도착 시각이 아님 — 더 기다려야 함
    return; // [한국어] 이번 사이클에는 출력 없음
  }
  assert(GetSimTime() == time); // [한국어] 현재 사이클이 도착 시각과 정확히 일치해야 함
                                // 미래(time < current)는 불가능 — 큐 삽입 로직 버그 검출
  _output = item.second; // [한국어] 도착한 데이터 포인터를 _output에 설정 — Receive()가 이 값을 반환
  assert(_output);       // [한국어] 도착한 포인터가 NULL이면 안 됨 — Send()에서 NULL 주입 방지
  _wait_queue.pop();     // [한국어] 처리 완료된 항목을 큐에서 제거
}

#endif
