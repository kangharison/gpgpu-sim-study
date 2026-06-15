// $Id: credit.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] NoC 크레딧 패킷 선언 (credit.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 Booksim2 기반 NoC(Network-on-Chip) 시뮬레이터에서 사용하는
 * 크레딧(Credit) 패킷의 클래스를 선언한다. 크레딧(Credit)이란 흐름 제어
 * (flow control) 메커니즘의 핵심 객체로, 다운스트림 라우터/버퍼가 플릿(flit)을
 * 소비했을 때 업스트림 라우터에게 "버퍼 공간이 생겼다"는 신호를 전달하는 패킷이다.
 * 즉, 크레딧 기반 흐름 제어(credit-based flow control)를 구현하는 데 핵심이 되는
 * 객체이며, 라우터들이 버퍼 오버플로우 없이 데이터를 전달하도록 보장한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 타이밍 모델에서 GPU 내부 NoC는 intersim2/ 모듈로 구현된다.
 * 패킷(Flit)이 업스트림 → 다운스트림 방향으로 전달되면, 다운스트림이 플릿을
 * 소비할 때마다 반대 방향(다운스트림 → 업스트림)으로 Credit 패킷을 보낸다.
 * 이 Credit 신호를 받은 업스트림 라우터는 해당 VC(Virtual Channel)의 가용
 * 버퍼 카운트를 증가시키고, 다음 플릿을 전송할 수 있다고 판단한다.
 *
 * 호출 체인:
 *   Router::_Step() → (downstream consumes flit) → Credit::New() →
 *   [credit 전송: 업스트림 방향] → Router가 credit 수신 → VC 버퍼 카운트 증가
 *
 * 실행 컨텍스트: 호스트 유저스페이스 — GPU 시뮬레이터 사이클 루프 내부
 *
 * === 타 모듈과의 연결 ===
 * - router.cpp / IQRouter 등: Credit::New()로 크레딧을 할당하고, 업스트림 포트에
 *   credit을 inject하여 흐름 제어를 수행한다.
 * - flit.hpp: 반대 방향 데이터 패킷(Flit)과 대칭 구조. Credit은 Flit의 역방향 신호.
 * - outputset.hpp, vc.hpp: 어떤 VC(가상 채널)가 버퍼를 반납했는지 Credit.vc에
 *   기록하여 upstream router의 VC 버퍼 카운터를 갱신한다.
 * - credit.cpp: 이 헤더에 선언된 모든 멤버를 구현한다.
 *
 * === 주요 함수/구조체 요약 ===
 * - Credit::New()     : 풀(pool)에서 Credit 객체를 재사용하거나 새로 할당
 * - Credit::Free()    : 사용 완료된 Credit을 풀에 반납 (실제 메모리 해제 아님)
 * - Credit::FreeAll() : 시뮬레이션 종료 시 _all 스택 전체를 실제 delete
 * - Credit::Reset()   : 재사용 전에 모든 필드를 초기 상태로 초기화
 * - Credit::OutStanding() : 현재 할당되어 사용 중인 크레딧 수 반환 (누수 감지용)
 * - vc (set<int>)     : 이 크레딧이 반납하는 가상 채널(VC) 번호 집합
 */

#ifndef _CREDIT_HPP_
#define _CREDIT_HPP_

#include <set>    // [한국어] set<int> vc 필드에 사용 — VC 번호 집합 저장
#include <stack>  // [한국어] _all, _free 정적 스택에 사용 — 풀 기반 객체 관리

/*
 * [한국어] Credit — NoC 흐름 제어 크레딧 패킷 클래스
 *
 * 크레딧(Credit)은 버퍼 기반 흐름 제어에서 업스트림 라우터에 "버퍼 공간이
 * 생겼다"는 사실을 알리는 역방향 제어 신호 객체이다. 실제 데이터(Flit)가
 * 업스트림→다운스트림으로 흐를 때, 다운스트림이 플릿을 소비할 때마다
 * Credit 패킷이 다운스트림→업스트림 방향으로 전송된다.
 *
 * 객체 생명 주기:
 *   Credit::New() → (라우터가 크레딧 사용) → Credit::Free() (풀 반납)
 *   시뮬레이션 종료: Credit::FreeAll() → 전체 메모리 해제
 *
 * 설계 특징: 풀(pool) 패턴을 사용하여 빈번한 new/delete 비용을 줄인다.
 * _all 스택은 생성된 모든 Credit을, _free 스택은 반납된 Credit을 추적한다.
 *
 * 생성자는 private이므로 Credit::New()를 통해서만 객체를 얻을 수 있다.
 */
class Credit {

public:

  set<int> vc;
  /* [한국어] 이 크레딧이 반납하는 가상 채널(VC: Virtual Channel) 번호 집합.
   * 설정자: 라우터가 다운스트림으로부터 플릿을 소비했을 때, 해당 VC 번호를
   *         이 집합에 insert하여 업스트림에 버퍼 반납을 통보한다.
   * 읽는 자: 업스트림 라우터가 크레딧을 수신하면 vc 집합을 순회하며
   *          각 VC의 가용 버퍼 카운트(credit count)를 1씩 증가시킨다.
   * 값 범위: 0 이상의 정수(VC 인덱스). 하나의 크레딧이 여러 VC를
   *          동시에 반납할 수 있으나 일반적으로는 하나의 VC만 포함.
   * 동기화: 단일 시뮬레이션 스레드 내에서 처리되므로 별도 락 불필요. */

  // these are only used by the event router
  bool head, tail;
  /* [한국어] 이벤트 기반 라우터(event router)에서만 사용하는 플릿 위치 마커.
   * head: 이 크레딧이 헤드 플릿(패킷의 첫 플릿)에 대응하는 크레딧임을 나타냄.
   * tail: 이 크레딧이 테일 플릿(패킷의 마지막 플릿)에 대응하는 크레딧임을 나타냄.
   * 설정자: 이벤트 라우터 내부에서 크레딧 생성 시 설정.
   * 읽는 자: 이벤트 라우터가 크레딧 처리 시 head/tail 여부에 따라 특수 동작 수행.
   * 값 범위: true/false. 일반 크레딧 흐름에서는 항상 false(Reset() 초기값).
   * 동기화: 단일 시뮬레이션 스레드에서만 접근하므로 별도 락 불필요. */

  int  id;
  /* [한국어] 이벤트 기반 라우터에서만 사용하는 크레딧 식별자.
   * 설정자: 이벤트 라우터가 Credit::New()로 할당 후 id를 설정.
   * 읽는 자: 이벤트 라우터가 크레딧을 수신하여 해당 패킷을 찾을 때 사용.
   * 값 범위: 음수(-1)는 미설정 상태(Reset() 초기값). 0 이상은 유효한 패킷 id.
   * 동기화: 단일 시뮬레이션 스레드에서만 접근하므로 별도 락 불필요. */

  /*
   * [한국어]
   * Reset - Credit 객체의 모든 필드를 초기 상태로 되돌린다.
   *
   * @return: void
   *
   * 풀에서 재사용할 때 이전 사용 내역이 남지 않도록 초기화한다.
   * vc 집합을 비우고, head/tail을 false로, id를 -1로 초기화한다.
   * 생성자 Credit()에서도 호출되어 새로 할당된 객체를 초기화한다.
   *
   * 호출 체인:
   *   Credit() → Reset()
   *   Credit::New() → c->Reset() (풀에서 재사용 시)
   */
  void Reset();

  /*
   * [한국어]
   * New - 풀에서 Credit 객체를 하나 가져온다 (없으면 새로 할당).
   *
   * @return: 초기화된 Credit 포인터. 호출자가 Free()로 반납해야 한다.
   *
   * 풀 기반(pool-based) 객체 재사용 패턴. _free 스택이 비어있으면
   * new Credit()으로 새로 할당하고 _all에 등록한다. 비어있지 않으면
   * _free 스택 상단에서 꺼내 Reset()하여 반환한다.
   * 이 방식은 매 사이클마다 크레딧 객체를 할당/해제하는 비용을 절감한다.
   *
   * 호출 체인:
   *   Router::_Step() → Credit::New() → [라우터에 전달] → Credit::Free()
   */
  static Credit * New();

  /*
   * [한국어]
   * Free - 사용 완료된 Credit 객체를 풀에 반납한다.
   *
   * @return: void
   *
   * 실제 메모리를 해제하지 않고 _free 스택에 push하여 재사용을 대기시킨다.
   * 시뮬레이션 종료 시 FreeAll()이 _all 스택의 객체를 실제로 delete한다.
   *
   * 호출 체인:
   *   Router (크레딧 소비 후) → Credit::Free()
   */
  void Free();

  /*
   * [한국어]
   * FreeAll - 시뮬레이션 종료 시 _all 스택의 모든 Credit 객체를 실제 delete한다.
   *
   * @return: void
   *
   * 시뮬레이션이 끝날 때 한 번 호출하여 풀에 등록된 모든 Credit 메모리를
   * 해제한다. _all 스택이 빌 때까지 top()을 delete하고 pop()한다.
   * 이 함수 호출 후에는 어떤 Credit 포인터도 사용해서는 안 된다.
   *
   * 호출 체인:
   *   main() / 시뮬레이션 종료 루틴 → Credit::FreeAll()
   */
  static void FreeAll();

  /*
   * [한국어]
   * OutStanding - 현재 할당되어 사용 중인(미반납) Credit 객체 수를 반환한다.
   *
   * @return: int — (_all.size() - _free.size()) = 현재 사용 중인 크레딧 수.
   *                이 값이 비정상적으로 크다면 Free()가 누락된 것이다.
   *
   * 디버그/통계 목적으로 사용. 총 생성 수에서 반납 수를 빼서 계산한다.
   *
   * 호출 체인:
   *   시뮬레이션 통계 출력 코드 → Credit::OutStanding()
   */
  static int OutStanding();

private:

  static stack<Credit *> _all;
  /* [한국어] 지금까지 new로 생성된 모든 Credit 포인터를 보관하는 스택.
   * 설정자: Credit::New()에서 new Credit()시 push됨.
   * 읽는 자: Credit::FreeAll()이 순회하며 전부 delete, Credit::OutStanding()이 size() 참조.
   * 값 범위: 누적 생성된 Credit 포인터만 존재. pop은 FreeAll()에서만 수행.
   * 동기화: 단일 시뮬레이션 스레드에서만 접근하므로 별도 락 불필요. */

  static stack<Credit *> _free;
  /* [한국어] Free()로 반납되어 재사용 대기 중인 Credit 포인터를 보관하는 스택.
   * 설정자: Credit::Free()에서 this를 push.
   * 읽는 자: Credit::New()에서 empty() 확인 후 top() 꺼내 재사용.
   *          Credit::OutStanding()이 size() 참조.
   * 값 범위: _all의 부분집합. _free.size() <= _all.size() 항상 성립.
   * 동기화: 단일 시뮬레이션 스레드에서만 접근하므로 별도 락 불필요. */

  /*
   * [한국어]
   * Credit - 기본 생성자 (private: 외부에서 직접 호출 불가).
   *
   * Credit 객체는 반드시 Credit::New()를 통해서만 얻어야 한다.
   * 생성자 내부에서 Reset()을 호출하여 필드를 초기화한다.
   *
   * 호출 체인:
   *   Credit::New() → new Credit() → Credit()
   */
  Credit();

  /*
   * [한국어]
   * ~Credit - 소멸자 (private: 외부에서 직접 delete 불가).
   *
   * FreeAll()이 delete 할 때만 호출된다. 빈 소멸자이며,
   * set<int> vc는 자동으로 소멸된다.
   *
   * 호출 체인:
   *   Credit::FreeAll() → delete credit_ptr → ~Credit()
   */
  ~Credit() {}

};

#endif
