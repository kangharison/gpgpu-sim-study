// $Id: flitchannel.hpp 5516 2013-10-06 02:14:48Z dub $

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
//  File Name: flitchannel.hpp
//
//  The FlitChannel models a flit channel with a multi-cycle
//   transmission delay. The channel latency can be specified as
//   an integer number of simulator cycles.
// ----------------------------------------------------------------------

/*
 * [한국어 설명] NoC 물리 링크(Flit 채널) 헤더 (flitchannel.hpp)
 *
 * === 파일의 역할 ===
 * FlitChannel은 두 라우터 사이(또는 노드-라우터 사이)의 물리적 단방향 링크를 모델링한다.
 * 링크는 설정된 지연(delay 사이클)만큼 Flit 전달을 지연시키며,
 * 이 지연이 NoC의 링크 전파 지연(propagation delay)을 시뮬레이션한다.
 * 또한 링크 활동 통계(active/idle 카운터)를 수집하여 전력 분석에 활용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   Network::ReadInputs() → FlitChannel::ReadInputs() (입력 캡처)
 *   Router가 WriteFlit() → FlitChannel::Send() (flit 전달 시작)
 *   Network::WriteOutputs() → FlitChannel::WriteOutputs() (지연 후 배달)
 *
 * 실행 컨텍스트: GPUTrafficManager::_Step() 내 _net[subnet]->ReadInputs()/WriteOutputs()
 * 에서 사이클마다 호출된다.
 *
 * === 타 모듈과의 연결 ===
 * - Channel<Flit> (channel.hpp): 다중 사이클 지연 로직을 구현하는 기반 클래스
 * - Router (router.hpp): 소스/싱크 라우터; SetSource/SetSink로 연결
 * - Network (network.hpp): 모든 FlitChannel을 생성하고 연결 구조를 관리
 * - Flit (flit.hpp): 채널을 통해 전달되는 데이터 단위
 * - Power_Module: _active/_idle 카운터로 스위칭 활동도 계산
 *
 * === 주요 함수/구조체 요약 ===
 * - FlitChannel(): 클래스 수/활동 통계 초기화
 * - SetSource/SetSink(): 연결된 소스/싱크 라우터 및 포트 번호 설정
 * - Send(): Flit을 채널에 주입 (활동 카운터 업데이트 포함)
 * - ReadInputs(): 이번 사이클 입력 캡처 (Channel<T>::ReadInputs 위임)
 * - WriteOutputs(): 지연 경과 후 Flit을 싱크에 배달
 */

#ifndef FLITCHANNEL_HPP
#define FLITCHANNEL_HPP

// ----------------------------------------------------------------------
//  $Author: jbalfour $
//  $Date: 2007/06/27 23:10:17 $
//  $Id: flitchannel.hpp 5516 2013-10-06 02:14:48Z dub $
// ----------------------------------------------------------------------

#include "channel.hpp"  // [한국어] Channel<Flit> 템플릿 기반 클래스 (다중 사이클 지연 구현)
#include "flit.hpp"     // [한국어] Flit 타입 정의

using namespace std;

class Router ; // [한국어] 전방 선언: 라우터 포인터를 멤버로 가지기 위해 필요

/*
 * [한국어]
 * FlitChannel - 두 라우터 사이의 단방향 물리 링크를 모델링하는 클래스
 *
 * Channel<Flit>을 상속하여 다중 사이클 지연 로직을 재사용하고,
 * GPU 시뮬레이션에 필요한 소스/싱크 라우터 정보와 활동 통계를 추가한다.
 *
 * 링크 지연(delay): Channel<Flit>의 _delay 필드로 설정된 사이클 수만큼
 * ReadInputs() → WriteOutputs() 사이에서 Flit이 대기한다.
 *
 * 활동 통계: 전력 분석을 위해 채널별로 flit 전송 사이클(_active)과
 * 유휴 사이클(_idle)을 누적하여 스위칭 활동도를 측정한다.
 */
class FlitChannel : public Channel<Flit> {
public:
  /*
   * [한국어]
   * FlitChannel() - 채널 초기화
   *
   * @parent: 부모 Module (주로 Network)
   * @name: 채널 이름 문자열 (예: "net0:chan_0_to_1")
   * @classes: 트래픽 클래스 수 (_active 벡터 크기)
   *
   * _active를 클래스 수만큼 0으로 초기화하고, _idle도 0으로 시작.
   * 소스/싱크 라우터 포인터는 NULL, 포트는 -1로 초기화됨.
   *
   * 호출 체인: Network 생성자 → [이 함수]
   */
  FlitChannel(Module * parent, string const & name, int classes);

  /*
   * [한국어]
   * SetSource() - 이 채널의 소스 라우터와 출구 포트를 설정한다
   *
   * @router: 이 채널에 flit을 주입하는 상류 라우터
   * @port: 해당 라우터의 출구 포트 번호
   *
   * 전력 분석에서 소스 라우터 크로스바의 활동도를 추적할 때 사용.
   *
   * 호출 체인: Network 생성자 내 채널 연결 코드 → [이 함수]
   */
  void SetSource(Router const * const router, int port) ;

  // [한국어] 소스 라우터 포인터 반환 (읽기 전용)
  inline Router const * const GetSource() const {
    return _routerSource;
  }

  // [한국어] 소스 라우터 포트 번호 반환 (읽기 전용)
  inline int const & GetSourcePort() const {
    return _routerSourcePort;
  }

  /*
   * [한국어]
   * SetSink() - 이 채널의 싱크 라우터와 입구 포트를 설정한다
   *
   * @router: 이 채널에서 flit을 수신하는 하류 라우터
   * @port: 해당 라우터의 입구 포트 번호
   *
   * lookahead 라우팅에서 "다음 라우터"를 찾을 때 사용 (inject channel의 GetSink()).
   *
   * 호출 체인: Network 생성자 → [이 함수]
   */
  void SetSink(Router const * const router, int port) ;

  // [한국어] 싱크 라우터 포인터 반환 (읽기 전용)
  inline Router const * const GetSink() const {
    return _routerSink;
  }

  // [한국어] 싱크 라우터 포트 번호 반환 (읽기 전용)
  inline int const & GetSinkPort() const {
    return _routerSinkPort;
  }

  // [한국어] 클래스별 활동 카운터 벡터 반환 (전력 분석용)
  inline vector<int> const & GetActivity() const {
    return _active;
  }

  /*
   * [한국어]
   * Send() - Flit을 이 채널에 주입하여 전달을 시작한다
   *
   * @flit: 전달할 Flit 포인터 (NULL이면 idle 사이클)
   *
   * f가 유효하면 해당 클래스(f->cl)의 _active 카운터 증가.
   * f가 NULL이면 _idle 카운터 증가 (빈 사이클).
   * 이후 Channel<Flit>::Send()에 위임하여 지연 큐에 삽입.
   *
   * 호출 체인: Router 파이프라인 → [이 함수] → Channel<Flit>::Send()
   */

  // Send flit
  virtual void Send(Flit * flit);

  /*
   * [한국어]
   * ReadInputs() - 이번 사이클에 채널로 들어온 Flit을 캡처한다
   *
   * 먼저 watch 플래그가 설정된 flit의 추적 메시지를 출력한 뒤,
   * Channel<Flit>::ReadInputs()에 위임하여 지연 큐에 Flit을 삽입.
   *
   * 호출 체인: Network::ReadInputs() → [이 함수] → Channel<Flit>::ReadInputs()
   */
  virtual void ReadInputs();

  /*
   * [한국어]
   * WriteOutputs() - 지연이 경과된 Flit을 싱크에 배달한다
   *
   * Channel<Flit>::WriteOutputs()로 _output 포인터를 업데이트한 뒤,
   * watch 플래그가 설정된 flit의 채널 통과 완료 메시지 출력.
   *
   * 호출 체인: Network::WriteOutputs() → [이 함수] → Channel<Flit>::WriteOutputs()
   */
  virtual void WriteOutputs();

private:

  ////////////////////////////////////////
  //
  // Power Models OBSOLETE
  //
  ////////////////////////////////////////
  // [한국어] 전력 모델용 통계: 구식(OBSOLETE)이지만 현재도 활동도 수집에 사용됨

  Router const * _routerSource;
  /* [한국어] 이 채널의 상류(소스) 라우터 포인터.
   * 설정자: SetSource()로 Network 생성 시 초기화.
   * 읽는 자: GetSource()를 통해 lookahead 라우팅, 전력 분석에서 사용.
   * 값 범위: 유효한 Router 포인터 또는 NULL(주입 채널의 경우 NULL).
   * 동기화: 초기화 후 읽기 전용. */

  int _routerSourcePort;
  /* [한국어] 소스 라우터에서 이 채널과 연결된 출구 포트 번호.
   * 설정자: SetSource()로 초기화.
   * 읽는 자: 전력 분석에서 특정 포트의 스위칭 활동도 분리 시.
   * 값 범위: 0 이상 정수, 또는 -1(미설정).
   * 동기화: 초기화 후 읽기 전용. */

  Router const * _routerSink;
  /* [한국어] 이 채널의 하류(싱크) 라우터 포인터.
   * 설정자: SetSink()로 Network 생성 시 초기화.
   * 읽는 자: GetSink()를 통해 lookahead 라우팅에서 "다음 라우터" 참조.
   * 값 범위: 유효한 Router 포인터 또는 NULL(배출 채널의 경우 NULL).
   * 동기화: 초기화 후 읽기 전용. */

  int _routerSinkPort;
  /* [한국어] 싱크 라우터에서 이 채널과 연결된 입구 포트 번호.
   * 설정자: SetSink()로 초기화.
   * 읽는 자: GetSinkPort()를 통해 in_channel 파라미터로 라우팅 함수에 전달.
   * 값 범위: 0 이상 정수, 또는 -1(미설정).
   * 동기화: 초기화 후 읽기 전용. */

  // Statistics for Activity Factors
  vector<int> _active;
  /* [한국어] 트래픽 클래스별 활성 전송 사이클 수 카운터.
   * 크기: 클래스 수 (FlitChannel 생성자에서 resize).
   * 설정자: Send(flit)에서 flit이 유효하면 _active[flit->cl]++.
   * 읽는 자: GetActivity()를 통해 Power_Module이 스위칭 활동도 계산.
   * 값 범위: 0 이상 누적 정수.
   * 동기화: 단일 스레드. */

  int _idle;
  /* [한국어] 채널이 빈 사이클(flit 없이 idle) 수 카운터.
   * 설정자: Send(NULL)에서 _idle++.
   * 읽는 자: 전력 분석에서 활동도 = _active / (_active + _idle) 계산.
   * 값 범위: 0 이상 누적 정수.
   * 동기화: 단일 스레드. */
};

#endif
