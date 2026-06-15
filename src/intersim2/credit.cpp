// $Id: credit.cpp 5188 2012-08-30 00:31:31Z dub $

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

/*credit.cpp
 *
 *A class for credits
 */

/*
 * [한국어 설명] NoC 크레딧 패킷 구현 (credit.cpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 Booksim2 기반 NoC 시뮬레이터의 흐름 제어(flow control) 메커니즘에서
 * 핵심이 되는 Credit 클래스의 멤버 함수를 구현한다. 크레딧(Credit)이란 다운스트림
 * 라우터가 플릿(flit)을 소비하여 버퍼 공간을 확보했을 때, 업스트림 라우터에게
 * "전송 가능한 슬롯이 생겼다"고 알리는 역방향 제어 신호 패킷이다.
 * 빈번한 생성/소멸 비용을 줄이기 위해 풀(pool) 기반 재사용 패턴을 채택한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2 NoC 시뮬레이터 내 라우터 계층에서 사용된다. 매 사이클마다 라우터는
 * 플릿(Flit)을 다운스트림으로 내보내고, 동시에 업스트림으로부터 Credit을 수신하여
 * 자신의 VC(Virtual Channel) 버퍼 카운터를 갱신한다.
 *
 * 실행 흐름:
 *   GPU 시뮬레이션 사이클 → icnt_wrapper.cc → Booksim2 시뮬레이터 Step() →
 *   Router::_Step() → (플릿 소비) → Credit::New() → [역방향 credit 전달] →
 *   업스트림 Router가 credit 수신 → VC 버퍼 카운터 증가 → Credit::Free()
 *
 * 실행 컨텍스트: 호스트 유저스페이스 — GPGPU-Sim 시뮬레이션 루프 단일 스레드
 *
 * === 타 모듈과의 연결 ===
 * - booksim.hpp: Booksim2 공통 헤더 (assert 매크로, 기본 타입 등 포함)
 * - credit.hpp: Credit 클래스 선언 (vc, head, tail, id 필드 정의)
 * - router.cpp / IQRouter: Credit::New()로 생성 후 업스트림 포트로 전달,
 *   수신 측에서 vc 집합을 읽어 버퍼 카운터를 갱신하고 Credit::Free()로 반납
 * - flit.cpp: 반대 방향 데이터 패킷(Flit)과 동일한 풀 기반 재사용 패턴을 공유
 *
 * === 주요 함수/구조체 요약 ===
 * - Credit::Credit()   : private 생성자 — Reset()을 호출하여 초기화
 * - Credit::Reset()    : vc 클리어, head/tail=false, id=-1 로 초기화
 * - Credit::New()      : 풀에서 재사용 또는 신규 할당하여 반환
 * - Credit::Free()     : 사용 완료된 객체를 풀(_free)에 반납
 * - Credit::FreeAll()  : 시뮬레이션 종료 시 _all 전체를 실제 delete
 * - Credit::OutStanding(): 미반납(사용 중) 크레딧 수 반환 (누수 감지용)
 */

#include "booksim.hpp"  // [한국어] Booksim2 공통 헤더 — assert 매크로, 기본 유틸 포함
#include "credit.hpp"   // [한국어] Credit 클래스 선언 포함

// [한국어] _all: 지금까지 생성된 모든 Credit 포인터 스택 (FreeAll에서 실제 delete용)
stack<Credit *> Credit::_all;

// [한국어] _free: 반납된 Credit 포인터 스택 (재사용 대기 풀)
stack<Credit *> Credit::_free;

/*
 * [한국어]
 * Credit::Credit - private 기본 생성자.
 *
 * @return: (생성자 반환값 없음)
 *
 * 외부에서 직접 호출 불가. Credit::New()를 통해서만 객체가 생성된다.
 * Reset()을 호출하여 모든 필드를 초기 상태(vc 비움, head=false, tail=false,
 * id=-1)로 설정한다.
 *
 * 호출 체인:
 *   Credit::New() → new Credit() → Credit::Credit() → Reset()
 */
Credit::Credit()
{
  Reset(); // [한국어] 생성 직후 모든 필드를 초기화하여 미정의 값 사용을 방지한다
}

/*
 * [한국어]
 * Credit::Reset - Credit 객체의 모든 필드를 초기 상태로 재설정한다.
 *
 * @return: void
 *
 * 풀(pool)에서 꺼내 재사용하기 전에 이전 크레딧의 내용이 남지 않도록
 * 초기화하는 함수. 생성자에서도 호출된다.
 * - vc.clear(): 이전에 반납했던 VC 번호를 모두 지운다.
 * - head=false, tail=false: 이벤트 라우터용 마커를 비활성화한다.
 * - id=-1: 유효하지 않은 크레딧 id임을 표시한다.
 *
 * 호출 체인:
 *   Credit::Credit() → Reset()
 *   Credit::New() → c->Reset() (재사용 시)
 */
void Credit::Reset()
{
  vc.clear();    // [한국어] 이전에 설정된 VC 번호 집합을 모두 제거 — 재사용 오염 방지
  head = false;  // [한국어] 이벤트 라우터용 head 마커 해제 — 일반 크레딧은 false가 기본값
  tail = false;  // [한국어] 이벤트 라우터용 tail 마커 해제 — 일반 크레딧은 false가 기본값
  id   = -1;     // [한국어] 크레딧 식별자를 -1(미설정)로 초기화 — 이벤트 라우터용 필드
}

/*
 * [한국어]
 * Credit::New - 풀에서 Credit 객체를 하나 가져오거나, 풀이 비었으면 새로 할당한다.
 *
 * @return: Credit* — 초기화된 Credit 포인터. 반드시 사용 후 Free()로 반납해야 한다.
 *
 * 풀 기반(pool-based) 재사용 패턴으로 빈번한 new/delete 비용을 절감한다.
 * 1. _free 스택이 비어 있으면: new Credit()으로 새 객체를 생성하고 _all에 push.
 * 2. _free 스택에 반납된 객체가 있으면: top()에서 꺼내 Reset()하여 반환.
 * 라우터가 매 사이클 크레딧을 생성/소멸하는 빈도를 고려할 때 이 패턴이 중요하다.
 *
 * 호출 체인:
 *   Router::_Step() (플릿 소비 후) → Credit::New() → [credit을 upstream에 push]
 */
Credit * Credit::New() {
  Credit * c;                    // [한국어] 반환할 Credit 포인터 선언
  if(_free.empty()) {            // [한국어] 풀에 재사용 가능한 객체가 없는 경우 → 신규 할당
    c = new Credit();            // [한국어] 힙에 Credit 객체를 새로 생성 (내부에서 Reset() 호출됨)
    _all.push(c);                // [한국어] FreeAll()에서 delete할 수 있도록 _all에 등록
  } else {                       // [한국어] 풀에 반납된 객체가 있는 경우 → 재사용
    c = _free.top();             // [한국어] _free 스택 상단에서 반납된 Credit 포인터를 꺼냄
    c->Reset();                  // [한국어] 이전 사용 내역(vc, head, tail, id)을 초기화
    _free.pop();                 // [한국어] _free 스택에서 제거 (이제 활성 사용 중)
  }
  return c;                      // [한국어] 초기화된 Credit 포인터 반환 — 호출자가 Free() 책임
}

/*
 * [한국어]
 * Credit::Free - 사용 완료된 Credit 객체를 _free 풀에 반납한다.
 *
 * @return: void
 *
 * 실제 메모리를 해제하지 않고 _free 스택에 push하여 재사용을 위해 대기시킨다.
 * 시뮬레이션 종료 시 FreeAll()이 _all 스택의 모든 객체를 실제로 delete한다.
 * 이 함수를 호출한 이후 해당 포인터를 역참조하면 안 된다.
 *
 * 호출 체인:
 *   Router (upstream에서 credit 수신 처리 완료 후) → Credit::Free()
 */
void Credit::Free() {
  _free.push(this); // [한국어] this 포인터를 _free 스택에 push하여 재사용 풀에 반납
}

/*
 * [한국어]
 * Credit::FreeAll - _all 스택에 등록된 모든 Credit 객체를 실제로 delete한다.
 *
 * @return: void
 *
 * 시뮬레이션 종료 시 한 번 호출하여 풀에 누적된 모든 Credit 메모리를 해제한다.
 * _all이 빌 때까지 top()의 포인터를 delete하고 pop()을 반복한다.
 * 이 함수 호출 이후 어떤 Credit 포인터도 사용해서는 안 된다.
 * _free 스택은 별도로 비우지 않아도 되는데, _all의 포인터가 실제 메모리를
 * 가리키기 때문에 _all을 전부 delete하면 충분하다.
 *
 * 호출 체인:
 *   main() / 시뮬레이션 종료 루틴 → Credit::FreeAll()
 */
void Credit::FreeAll() {
  while(!_all.empty()) {   // [한국어] _all 스택이 빌 때까지 반복 — 모든 할당 Credit 순회
    delete _all.top();     // [한국어] 스택 상단의 Credit 포인터가 가리키는 객체를 실제 delete
    _all.pop();            // [한국어] delete된 포인터를 스택에서 제거
  }
}


/*
 * [한국어]
 * Credit::OutStanding - 현재 할당되어 사용 중인(미반납) Credit 수를 반환한다.
 *
 * @return: int — _all.size() - _free.size() = 현재 활성 크레딧 수.
 *                시뮬레이션 중간에 이 값이 비정상적으로 크면 Free()가 누락된 것이다.
 *
 * 디버그/통계 목적으로 사용된다. 총 생성 수(_all.size())에서 반납 대기 수
 * (_free.size())를 빼면 현재 실제로 라우터에서 사용 중인 크레딧 수가 된다.
 *
 * 호출 체인:
 *   시뮬레이션 통계 출력 / 디버그 코드 → Credit::OutStanding()
 */
int Credit::OutStanding(){
  return _all.size()-_free.size(); // [한국어] (총 생성 수) - (반납 대기 수) = 현재 사용 중인 크레딧 수
}
