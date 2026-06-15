// $Id: module.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] BookSim NoC 시뮬레이션 객체 기반 클래스 헤더 (module.hpp)
 *
 * === 파일의 역할 ===
 * BookSim NoC 시뮬레이터의 모든 하드웨어 구성 요소(라우터, 버퍼, 채널, 네트워크 등)가
 * 공통으로 상속받는 기반 클래스 Module을 선언한다. Module은 각 컴포넌트에
 * 이름(_name)과 계층 경로(_fullname), 자식 목록(_children)을 부여하여
 * 트리 형태의 하드웨어 계층 구조를 표현하고 디버그/에러 메시지 출력 시
 * 어느 컴포넌트에서 문제가 발생했는지 추적 가능하게 한다.
 * Display() 가상 함수를 통해 하위 클래스마다 자신의 상태를 출력할 수 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2/ 디렉토리의 기반 계층(base layer)에 위치하며, 거의 모든 BookSim 클래스가
 * 이 Module을 직접 또는 간접 상속한다.
 * 실행 컨텍스트: 호스트(CPU) 유저스페이스 — GPGPU-Sim 시뮬레이션 초기화 및 사이클 루프.
 * 호출 체인: Network::Network() → Router::Router() → Module::Module()
 *           (생성자 체인으로 계층 구조 자동 등록).
 * Module 자체는 사이클 루프를 직접 구동하지 않고, 하위 클래스(Router 등)가
 * 사이클마다 호출된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존: booksim.hpp (BookSim 공통 타입/매크로), <string>, <vector>, <iostream>.
 * 피의존: Network, Router, VC, Channel, OutputSet 등 intersim2 내 거의 모든 클래스.
 * 데이터 흐름: 생성 시 부모 모듈로부터 계층 경로를 물려받아 _fullname에 저장.
 *             자식 모듈은 _AddChild()를 통해 부모의 _children 벡터에 등록된다.
 * 공유 자료구조: _children 벡터 (계층 탐색 시 사용, 사이클 루프에서는 미사용).
 *
 * === 주요 함수/구조체 요약 ===
 * Module(parent, name)    - 생성자: 이름 설정 및 부모 모듈에 자신을 자식으로 등록
 * _AddChild(child)        - protected: 자식 모듈을 _children 벡터에 추가
 * Name() / FullName()     - 단순 이름 / 계층 경로 문자열 반환 (인라인)
 * DisplayHierarchy(level) - 들여쓰기를 적용하여 모듈 트리 구조를 재귀 출력
 * Error(msg)              - 에러 메시지를 _fullname과 함께 출력 후 exit(-1)
 * Debug(msg)              - 디버그 메시지를 _fullname과 함께 stdout 출력
 * Display(os)             - 하위 클래스에서 재정의하는 상태 출력 가상 함수
 */

#ifndef _MODULE_HPP_ // [한국어] 헤더 중복 포함 방지 가드 시작
#define _MODULE_HPP_ // [한국어] 가드 매크로 정의

#include "booksim.hpp" // [한국어] BookSim 공통 타입, 전역 매크로, assert 등 기반 헤더

#include <string>    // [한국어] std::string: 모듈 이름(_name, _fullname) 저장에 사용
#include <vector>    // [한국어] std::vector: 자식 모듈 목록(_children) 저장에 사용
#include <iostream>  // [한국어] std::ostream, std::cout: DisplayHierarchy/Display 출력에 사용

/*
 * [한국어]
 * Module - BookSim NoC 시뮬레이터의 모든 하드웨어 컴포넌트 기반 클래스.
 *
 * BookSim 내 라우터, 채널, 버퍼, 네트워크 등 모든 시뮬레이션 객체가
 * 이 클래스를 상속하여 계층 구조(부모-자식 트리)를 형성한다.
 * 각 모듈은 짧은 이름(_name)과 루트로부터의 전체 경로(_fullname)를 가지며,
 * 에러/디버그 출력 시 어느 컴포넌트에서 문제가 발생했는지 명확히 알 수 있다.
 * Display()는 가상 함수로, 하위 클래스에서 재정의하여 내부 상태를 출력한다.
 *
 * 동기화: Module 자체는 단일 스레드(시뮬레이션 주 스레드)에서만 접근되므로
 *         별도 락이 없다. GPGPU-Sim의 icnt_wrapper가 별도 스레드에서 호출하더라도
 *         Module 레벨의 계층 구조는 초기화 단계에서만 수정된다.
 */
class Module {
private:
  string _name;
  /* [한국어] 이 모듈의 짧은 이름 (부모 경로 제외).
   * 예: 네트워크 내 라우터의 경우 "router_0", "router_1" 등.
   * 설정자: Module 생성자에서 name 파라미터로 1회 설정.
   * 읽는 자: Name() 인라인 함수, DisplayHierarchy(), Error(), Debug().
   * 값 범위: 비어 있지 않은 임의의 문자열 (생성자 호출자가 결정).
   * 동기화: 초기화 후 변경되지 않으므로 별도 락 불필요. */

  string _fullname;
  /* [한국어] 루트 모듈부터 이 모듈까지의 전체 계층 경로 (슬래시 구분).
   * 예: 루트 네트워크 "net" 아래 라우터 "router_0" → "net/router_0".
   * 부모가 없는 루트 모듈은 _fullname == _name.
   * 설정자: Module 생성자에서 parent->_fullname + "/" + name으로 조합하여 설정.
   * 읽는 자: FullName(), Error(), Debug(), DisplayHierarchy().
   * 값 범위: "이름" 또는 "부모경로/이름" 형태의 슬래시 구분 경로 문자열.
   * 동기화: 초기화 후 변경되지 않으므로 별도 락 불필요. */

  vector<Module *> _children;
  /* [한국어] 이 모듈에 직접 등록된 자식 모듈들의 포인터 벡터.
   * 생성 시 자식 모듈의 생성자가 부모의 _AddChild()를 호출하여 자동 등록된다.
   * 설정자: _AddChild(child) — Module 생성자 내에서 parent->_AddChild(this)로 호출.
   * 읽는 자: DisplayHierarchy() — 트리를 재귀적으로 출력할 때 순회.
   * 값 범위: 유효한 Module* 포인터들의 배열 (NULL 없음).
   * 동기화: 초기화 단계에서만 삽입되고 이후 읽기만 수행되므로 락 불필요. */

protected:
  /*
   * [한국어]
   * _AddChild - 자식 모듈을 _children 벡터에 추가한다.
   *
   * @child: 등록할 자식 Module 포인터. 유효한 포인터여야 하며 NULL 전달 금지.
   * @return: 없음 (void).
   *
   * Module 생성자에서 parent가 존재하면 parent->_AddChild(this)를 호출하여
   * 자동으로 계층 구조를 형성한다. 이 함수는 protected이므로 외부에서 직접
   * 호출하는 것은 불가능하고, 생성자 체인에서만 사용된다.
   * 실행 컨텍스트: 호스트 CPU, 초기화 단계.
   *
   * 호출 체인:
   *   Module::Module(parent, ...) → parent->_AddChild(this)
   */
  void _AddChild( Module *child );

public:
  /*
   * [한국어]
   * Module - 모듈 생성자. 이름을 설정하고 부모 계층 구조에 자신을 등록한다.
   *
   * @parent: 이 모듈의 부모 Module 포인터. 루트 모듈은 nullptr(NULL) 전달.
   * @name:   이 모듈의 짧은 이름 문자열. 계층 경로 조합 및 디버그 출력에 사용.
   * @return: 없음 (생성자).
   *
   * parent가 NULL이 아니면 parent->_AddChild(this)를 호출하여 부모의
   * _children에 자신을 추가하고, _fullname을 parent->_fullname + "/" + name으로 설정한다.
   * parent가 NULL이면 루트 모듈로 간주하여 _fullname = name으로 설정한다.
   * 실행 컨텍스트: 호스트 CPU, 시뮬레이터 초기화 단계.
   *
   * 호출 체인:
   *   Network::Network() → [Module::Module(parent, name)] → _AddChild()
   */
  Module( Module *parent, const string& name );

  /*
   * [한국어]
   * ~Module - 가상 소멸자. 하위 클래스의 소멸자가 올바르게 호출되도록 보장.
   *
   * @return: 없음 (소멸자).
   *
   * 기본 구현은 비어 있다. _children 벡터는 포인터만 저장하므로
   * 자식 모듈의 메모리 해제는 각 하위 클래스 소멸자에서 담당한다.
   * virtual 선언으로 다형 소멸(polymorphic destruction)을 지원한다.
   */
  virtual ~Module( ) { }

  /*
   * [한국어]
   * Name - 이 모듈의 짧은 이름을 반환한다 (인라인).
   *
   * @return: _name 문자열 const 참조. 복사 없이 이름을 읽을 수 있다.
   *
   * 에러 출력, 통계 레이블, 설정 파일 매칭 등에서 사용된다.
   * 실행 컨텍스트: 호스트 CPU, 초기화 및 사이클 루프 모두에서 호출 가능.
   */
  inline const string & Name() const { return _name; }

  /*
   * [한국어]
   * FullName - 루트로부터의 전체 계층 경로 문자열을 반환한다 (인라인).
   *
   * @return: _fullname 문자열 const 참조.
   *          예: "net/router_0/vc_0".
   *
   * Error(), Debug() 내부에서 메시지와 함께 출력하여 어느 컴포넌트에서
   * 문제가 발생했는지를 즉시 파악할 수 있게 한다.
   * 실행 컨텍스트: 호스트 CPU, 초기화 및 사이클 루프 모두에서 호출 가능.
   */
  inline const string & FullName() const { return _fullname; }

  /*
   * [한국어]
   * DisplayHierarchy - 이 모듈과 자식들을 들여쓰기 트리 형태로 재귀 출력한다.
   *
   * @level: 현재 들여쓰기 수준 (공백 2칸 × level). 루트 호출 시 0.
   * @os:    출력 스트림 (기본값 std::cout).
   * @return: 없음 (void).
   *
   * 시뮬레이터 초기화 후 네트워크 전체 계층 구조를 확인할 때 사용한다.
   * _children 벡터를 순회하며 각 자식에 대해 level+1로 재귀 호출한다.
   * 실행 컨텍스트: 호스트 CPU, 초기화 또는 디버그 목적 출력 단계.
   *
   * 호출 체인:
   *   외부 코드 → [DisplayHierarchy(0)] → (*mod_iter)->DisplayHierarchy(level+1) (재귀)
   */
  void DisplayHierarchy( int level = 0, ostream & os = cout ) const;

  /*
   * [한국어]
   * Error - 에러 메시지를 _fullname과 함께 출력하고 시뮬레이터를 즉시 종료한다.
   *
   * @msg: 출력할 에러 메시지 문자열.
   * @return: 없음 (exit(-1)로 프로세스 종료).
   *
   * "Error in <fullname> : <msg>" 형식으로 stdout에 출력 후 exit(-1)을 호출한다.
   * 복구 불가능한 설정 오류, 파라미터 불일치, 내부 상태 오류 시 사용한다.
   * 실행 컨텍스트: 호스트 CPU, 초기화 또는 사이클 루프 어디서든 호출 가능.
   *
   * 호출 체인:
   *   하위 클래스 메서드 → [Error(msg)] → exit(-1)
   */
  void Error( const string& msg ) const;

  /*
   * [한국어]
   * Debug - 디버그 메시지를 _fullname과 함께 stdout에 출력한다.
   *
   * @msg: 출력할 디버그 메시지 문자열.
   * @return: 없음 (void).
   *
   * "Debug (<fullname>) : <msg>" 형식으로 출력한다.
   * Error()와 달리 종료하지 않으므로 진단 목적의 중간 상태 출력에 사용된다.
   * 실행 컨텍스트: 호스트 CPU, 사이클 루프 내에서도 호출 가능 (고빈도 호출 시 성능 주의).
   *
   * 호출 체인:
   *   하위 클래스 메서드 → [Debug(msg)] → stdout 출력
   */
  void Debug( const string& msg ) const;

  /*
   * [한국어]
   * Display - 모듈 내부 상태를 출력하는 가상 함수 (기본 구현 제공).
   *
   * @os: 출력 스트림 (기본값 std::cout).
   * @return: 없음 (void).
   *
   * 기본 구현은 "Display method not implemented for <fullname>"을 출력한다.
   * 하위 클래스(Router, VC 등)에서 재정의하여 자신의 상태(버퍼 내용,
   * 카운터 등)를 출력하도록 한다. 디버그 목적의 상태 덤프에 사용된다.
   * 실행 컨텍스트: 호스트 CPU, 주로 디버그/진단 출력 단계.
   *
   * 호출 체인:
   *   외부 디버그 코드 → [Display(os)] → 하위 클래스 override 실행
   */
  virtual void Display( ostream & os = cout ) const;
};

#endif // [한국어] _MODULE_HPP_ 헤더 가드 끝
