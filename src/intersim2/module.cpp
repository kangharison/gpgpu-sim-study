// $Id: module.cpp 5188 2012-08-30 00:31:31Z dub $

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

/*module.cpp
 *
 *The basic class that is extended by all other components of the network
 *Provides the basic hierarchy structure and basic fuctions
 *
 */

/*
 * [한국어 설명] BookSim NoC 시뮬레이션 객체 기반 클래스 구현 (module.cpp)
 *
 * === 파일의 역할 ===
 * module.hpp에 선언된 Module 클래스의 메서드를 구현한다.
 * Module은 BookSim NoC 시뮬레이터 내 모든 하드웨어 컴포넌트(라우터, 채널,
 * 버퍼, 네트워크 등)의 공통 기반 클래스로, 각 객체에 이름(_name)과
 * 계층 경로(_fullname)를 부여하며 부모-자식 트리 구조를 형성한다.
 * 생성자에서 부모 모듈에 자신을 자식으로 등록하고, Error()/Debug()/Display()를
 * 통해 계층 기반 진단 출력을 제공한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2/ 기반 계층 최하단에 위치하며, 거의 모든 BookSim 클래스의 부모이다.
 * 실행 컨텍스트: 호스트(CPU) 유저스페이스 — 시뮬레이터 초기화 단계(생성자 체인)
 *               및 에러/디버그 출력 시 호출된다.
 * 호출 체인: Network::Network() → Router::Router() → Module::Module()
 *            (모든 컴포넌트 생성자가 Module() 생성자를 먼저 호출).
 *
 * === 타 모듈과의 연결 ===
 * 의존: booksim.hpp (공통 타입/매크로), module.hpp (자기 선언),
 *       <iostream> (cout), <cstdlib> (exit).
 * 피의존: Network, Router, VC, Channel 등 intersim2 내 거의 모든 클래스.
 * 데이터 흐름: 생성자 → 부모 _children 벡터에 자신 추가 → 계층 경로 조합.
 * 공유 자료구조: _children 벡터 (초기화 시 삽입, 이후 DisplayHierarchy에서 순회).
 *
 * === 주요 함수/구조체 요약 ===
 * Module(parent, name)    - 생성자: 이름 설정 및 부모에 자식으로 등록
 * _AddChild(child)        - 자식 포인터를 _children 벡터에 push_back
 * DisplayHierarchy(level) - 들여쓰기 트리 재귀 출력
 * Error(msg)              - 에러 메시지 출력 후 exit(-1)
 * Debug(msg)              - 디버그 메시지 stdout 출력
 * Display(os)             - 기본 "미구현" 메시지 출력 (하위 클래스가 재정의)
 */

#include <iostream>  // [한국어] std::cout, std::endl: 에러/디버그 메시지 출력에 사용
#include <cstdlib>   // [한국어] exit(): Error() 함수에서 시뮬레이터 비정상 종료에 사용

#include "booksim.hpp" // [한국어] BookSim 공통 타입, 매크로, 전역 설정 기반 헤더
#include "module.hpp"  // [한국어] 이 파일에서 구현하는 Module 클래스 선언 헤더

/*
 * [한국어]
 * Module::Module - 모듈 생성자. 이름을 설정하고 부모 계층 구조에 자신을 등록한다.
 *
 * @parent: 이 모듈의 부모 Module 포인터. 루트 모듈이면 nullptr(NULL) 전달.
 * @name:   이 모듈의 짧은 이름 문자열. 계층 경로 조합 및 에러 출력에 사용.
 * @return: 없음 (생성자).
 *
 * parent가 NULL이 아닌 경우:
 *   1) parent->_AddChild(this)를 호출하여 부모의 _children에 자신을 등록한다.
 *   2) _fullname = parent->_fullname + "/" + name으로 계층 경로를 조합한다.
 * parent가 NULL인 경우: _fullname = name (루트 모듈).
 * 실행 컨텍스트: 호스트 CPU, 시뮬레이터 초기화 단계 (사이클 루프 시작 전).
 * 동기화: 초기화 단계에서 단일 스레드로 실행되므로 락 불필요.
 *
 * 호출 체인:
 *   Network::Network() / Router::Router() → [Module::Module(parent, name)] → parent->_AddChild()
 */
Module::Module( Module *parent, const string& name )
{
  _name = name; // [한국어] 짧은 이름(_name) 설정; 부모 경로 없이 이 모듈 자체의 식별자

  if ( parent ) { // [한국어] 부모 모듈이 존재하는 경우: 계층 구조에 등록
    parent->_AddChild( this ); // [한국어] 부모의 _children 벡터에 이 모듈 포인터를 추가
    _fullname = parent->_fullname + "/" + name; // [한국어] 전체 경로 = 부모 경로 + "/" + 내 이름
                                                // [한국어] 예: "net" + "/" + "router_0" → "net/router_0"
  } else { // [한국어] 부모가 없는 경우: 루트 모듈로 간주
    _fullname = name; // [한국어] 루트 모듈의 전체 경로는 자신의 이름과 동일
  }
}

/*
 * [한국어]
 * Module::_AddChild - 자식 모듈 포인터를 _children 벡터에 추가한다.
 *
 * @child: 등록할 자식 Module 포인터. NULL 전달 금지 (호출자 보장 필요).
 * @return: 없음 (void).
 *
 * Module 생성자에서만 호출되며, 자식 모듈이 생성될 때 부모에 자신을 자동 등록한다.
 * DisplayHierarchy()에서 _children를 순회할 때 이 벡터가 사용된다.
 * protected이므로 Module 클래스 자신과 하위 클래스에서만 접근 가능하다.
 * 실행 컨텍스트: 호스트 CPU, 초기화 단계 (단일 스레드, 락 불필요).
 *
 * 호출 체인:
 *   Module::Module(parent, name) → parent->[_AddChild(this)]
 */
void Module::_AddChild( Module *child )
{
  _children.push_back( child ); // [한국어] 자식 모듈 포인터를 _children 벡터 끝에 추가
                                 // [한국어] 이후 DisplayHierarchy()가 이 벡터를 순회하여 트리 출력
}

/*
 * [한국어]
 * Module::DisplayHierarchy - 이 모듈과 모든 자식 모듈을 들여쓰기 트리 형태로 재귀 출력한다.
 *
 * @level: 현재 들여쓰기 수준 (공백 2칸 × level). 최초 호출은 0.
 * @os:    출력 대상 스트림 (기본값 std::cout).
 * @return: 없음 (void).
 *
 * 먼저 level만큼 "  "(공백 2칸)를 출력하여 들여쓰기를 적용한 후,
 * 자신의 _name을 출력한다. 그런 다음 _children 벡터의 각 자식에 대해
 * level+1로 재귀 호출하여 트리 전체를 출력한다.
 * 실행 컨텍스트: 호스트 CPU, 초기화 후 진단 출력 단계.
 *
 * 호출 체인:
 *   외부 코드 → [DisplayHierarchy(0)] → (*mod_iter)->DisplayHierarchy(level+1) (재귀)
 */
void Module::DisplayHierarchy( int level, ostream & os ) const
{
  vector<Module *>::const_iterator mod_iter; // [한국어] _children 벡터 순회용 const 반복자 선언

  for ( int l = 0; l < level; l++ ) { // [한국어] 현재 레벨만큼 들여쓰기 공백 출력
    os << "  ";   // [한국어] 한 레벨당 공백 2칸 출력; 트리 시각화를 위한 들여쓰기
  }

  os << _name << endl; // [한국어] 이 모듈의 짧은 이름(_name)을 출력하고 줄바꿈

  for ( mod_iter = _children.begin( ); // [한국어] _children 벡터의 첫 번째 자식부터 순회
	mod_iter != _children.end( ); mod_iter++ ) { // [한국어] 마지막 자식까지 반복
    (*mod_iter)->DisplayHierarchy( level + 1 ); // [한국어] 자식 모듈을 한 레벨 증가시켜 재귀 출력
  }
}

/*
 * [한국어]
 * Module::Error - 에러 메시지를 모듈 전체 경로와 함께 출력하고 시뮬레이터를 종료한다.
 *
 * @msg: 출력할 에러 메시지 문자열.
 * @return: 없음 (exit(-1)로 프로세스 종료).
 *
 * "Error in <_fullname> : <msg>" 형식으로 stdout에 출력 후 exit(-1)을 호출한다.
 * 복구 불가능한 오류(설정 파라미터 불일치, NULL 포인터 역참조 등) 발생 시 사용한다.
 * _fullname을 포함하기 때문에 수십 개의 라우터/버퍼 중 어느 컴포넌트에서
 * 오류가 발생했는지 즉시 파악할 수 있다.
 * 실행 컨텍스트: 호스트 CPU, 초기화 또는 사이클 루프 어느 단계에서든 호출 가능.
 *
 * 호출 체인:
 *   하위 클래스 메서드 → [Error(msg)] → exit(-1) (프로세스 종료)
 */
void Module::Error( const string& msg ) const
{
  cout << "Error in " << _fullname << " : " << msg << endl; // [한국어] "Error in <경로> : <메시지>" 형식으로 stdout에 출력
  exit( -1 ); // [한국어] 복구 불가능한 오류이므로 시뮬레이터 즉시 종료; 반환값 -1은 비정상 종료를 의미
}

/*
 * [한국어]
 * Module::Debug - 디버그 메시지를 모듈 전체 경로와 함께 stdout에 출력한다.
 *
 * @msg: 출력할 디버그 메시지 문자열.
 * @return: 없음 (void).
 *
 * "Debug (<_fullname>) : <msg>" 형식으로 출력한다.
 * Error()와 달리 종료하지 않으므로 진단 목적의 중간 상태 확인에 사용된다.
 * 사이클 루프 내에서 고빈도로 호출하면 성능에 영향을 줄 수 있으므로
 * 개발/디버그 빌드에서만 사용하는 것이 권장된다.
 * 실행 컨텍스트: 호스트 CPU, 초기화 또는 사이클 루프 어느 단계에서든 호출 가능.
 *
 * 호출 체인:
 *   하위 클래스 메서드 → [Debug(msg)] → stdout 출력 후 반환
 */
void Module::Debug( const string& msg ) const
{
  cout << "Debug (" << _fullname << ") : " << msg << endl; // [한국어] "Debug (<경로>) : <메시지>" 형식으로 stdout에 출력
}

/*
 * [한국어]
 * Module::Display - 모듈 내부 상태 출력의 기본 구현 (하위 클래스에서 재정의 권장).
 *
 * @os: 출력 대상 스트림 (기본값 std::cout).
 * @return: 없음 (void).
 *
 * 기본 구현은 "Display method not implemented for <fullname>"을 출력하여
 * 하위 클래스가 이 함수를 재정의하지 않았을 때 개발자에게 알린다.
 * Router, VC 등 하위 클래스는 이 함수를 override하여 버퍼 내용,
 * 카운터 값 등 자신의 상세 상태를 출력한다.
 * 실행 컨텍스트: 호스트 CPU, 주로 디버그/진단 단계.
 *
 * 호출 체인:
 *   외부 디버그 코드 → [Display(os)] → 하위 클래스 override 또는 기본 구현 실행
 */
void Module::Display( ostream & os ) const
{
  os << "Display method not implemented for " << _fullname << endl; // [한국어] 하위 클래스가 Display()를 구현하지 않았음을 알리는 기본 메시지 출력
}
