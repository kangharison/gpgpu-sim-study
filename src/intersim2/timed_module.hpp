// $Id: timed_module.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] 사이클 기반 동기 모듈 기반 클래스 (timed_module.hpp)
 *
 * === 파일의 역할 ===
 * TimedModule은 BookSim NoC 시뮬레이터에서 매 시뮬레이션 사이클 단위로
 * 동작하는 모든 하드웨어 모듈(라우터, 채널 등)의 공통 인터페이스를 정의한다.
 * Module의 계층 구조(이름/에러 출력)를 상속받고,
 * ReadInputs() → Evaluate() → WriteOutputs() 세 단계로 구성된
 * 사이클 루프 인터페이스를 추가한다. 이는 디지털 회로의
 * "입력 읽기 → 조합 논리 평가 → 출력 쓰기" 파이프라인을 모방한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2 기반 계층(module.hpp 위)에 위치하며,
 * Channel, Router, Network 등 사이클 단위로 상태가 진행하는 모듈이 상속.
 *
 * 실행 컨텍스트: 호스트 CPU 단일 스레드 — TrafficManager/_Step()이 매 사이클 호출.
 *
 * === 타 모듈과의 연결 ===
 * - module.hpp : Module 기반 클래스 — 이름, 계층, Error/Debug 제공.
 * - channel.hpp : Channel<T>가 TimedModule을 상속하여 링크 지연 파이프라인 구현.
 * - router.hpp  : Router가 TimedModule을 상속하여 라우터 사이클 로직 구현.
 *
 * === 주요 함수/구조체 요약 ===
 * - TimedModule(parent, name) : 생성자 — Module 기반 초기화.
 * - ~TimedModule()            : 가상 소멸자 — 다형 소멸 보장.
 * - ReadInputs()              : 매 사이클 시작 시 입력 포트/채널에서 데이터 읽기 (순수 가상).
 * - Evaluate()                : 조합/상태 논리 평가 (순수 가상).
 * - WriteOutputs()            : 매 사이클 종료 시 출력 포트/채널에 데이터 쓰기 (순수 가상).
 */

#ifndef _TIMED_MODULE_HPP_
#define _TIMED_MODULE_HPP_

#include "module.hpp" /* [한국어] Module 기반 클래스 포함 */

/*
 * [한국어]
 * TimedModule — 사이클 단위로 동작하는 하드웨어 모듈의 기반 클래스.
 *
 * 모든 시뮬레이션 대상 모듈은 이 클래스를 상속하여
 * ReadInputs(), Evaluate(), WriteOutputs()를 구현해야 한다.
 * TrafficManager가 _Step()을 실행할 때 이 세 메서드를
 * 일정한 순서로 호출하여 한 사이클의 동작을 완성한다.
 */
class TimedModule : public Module {

public:
  /*
   * [한국어]
   * TimedModule — 사이클 모듈 생성자.
   * @parent: 부모 Module 포인터 (계층 등록용).
   * @name:   이 모듈의 이름.
   * Module(parent, name)을 호출하여 이름과 계층 구조를 초기화한다.
   */
  TimedModule(Module * parent, string const & name) : Module(parent, name) {}

  /*
   * [한국어]
   * ~TimedModule — 가상 소멸자.
   * 하위 클래스(Router, Channel 등)의 소멸자가 올바르게 호출되도록 보장.
   */
  virtual ~TimedModule() {}
  
  /*
   * [한국어]
   * ReadInputs — 매 사이클 시작 단계: 입력 채널/포트로부터 데이터를 읽는다.
   * 예: Channel::ReadInputs()는 _input을 _wait_queue에 삽입.
   *     Router::ReadInputs()는 입력 플릿/크레딧을 수신.
   */
  virtual void ReadInputs() = 0;

  /*
   * [한국어]
   * Evaluate — 매 사이클 평가 단계: 상태 머신/조합 논리/중재 등을 수행.
   * 예: Router::Evaluate()는 라우팅, VC 할당, 스위치 할당, 크로스바 통과를 평가.
   */
  virtual void Evaluate() = 0;

  /*
   * [한국어]
   * WriteOutputs — 매 사이클 종료 단계: 출력 채널/포트에 데이터를 쓴다.
   * 예: Channel::WriteOutputs()는 _wait_queue에서 도착한 데이터를 _output에 설정.
   *     Router::WriteOutputs()는 다음 홉으로 플릿/크레딧을 전송.
   */
  virtual void WriteOutputs() = 0;
};

#endif
