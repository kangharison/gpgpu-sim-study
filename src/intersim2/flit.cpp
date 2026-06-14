// $Id: flit.cpp 5188 2012-08-30 00:31:31Z dub $

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

/*flit.cpp
 *
 *flit struct is a flit, carries all the control signals that a flit needs
 *Add additional signals as necessary. Flits has no concept of length
 *it is a singluar object.
 *
 *When adding objects make sure to set a default value in this constructor
 */

/*
 * [한국어 설명] NoC 전송 기본 단위 Flit 구현 (flit.cpp)
 *
 * === 파일의 역할 ===
 * Flit 클래스의 구현 파일. 주로 풀(pool) 기반 메모리 관리(New/Free/FreeAll),
 * 초기화(Reset), 출력 연산자를 구현한다. Flit은 NoC 내부의 유일한 전송 단위이므로
 * 시뮬레이션 중 매우 빈번하게 생성/소멸되며, 풀 재사용이 성능에 중요하다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 생성: _GeneratePacket() → Flit::New() (패킷당 n_flits 개 생성)
 * 소멸: _RetireFlit() → Flit::Free() (목적지 도달 후 풀에 반환)
 * 전체 해제: 시뮬레이터 종료 → Flit::FreeAll()
 *
 * === 타 모듈과의 연결 ===
 * - gputrafficmanager.cpp: _GeneratePacket에서 New(), _RetireFlit에서 Free() 호출
 * - vc.cpp: VC 버퍼에 Flit 저장 및 반환
 * - flitchannel.cpp: 라우터 간 물리 링크를 통해 Flit 전달
 *
 * === 주요 함수/구조체 요약 ===
 * - _all: 전체 Flit 생명주기 추적 스택 (FreeAll에서 delete 대상)
 * - _free: 재사용 가능한 Flit 풀 스택 (New/Free가 관리)
 * - Reset(): 모든 필드 기본값으로 초기화
 * - New(): 풀에서 꺼내거나 새로 생성
 * - Free(): 풀에 반환
 * - FreeAll(): 시뮬레이션 종료 시 실제 메모리 해제
 */

#include "booksim.hpp"  // [한국어] BookSim 공통 정의
#include "flit.hpp"     // [한국어] Flit 클래스 선언

// [한국어] 정적 멤버 변수 정의 (클래스 외부에서 초기화 필요)
stack<Flit *> Flit::_all;  // [한국어] new로 생성된 모든 Flit 추적 (FreeAll 대상)
stack<Flit *> Flit::_free; // [한국어] 재사용 가능한 Flit 풀 (New/Free 관리)

/*
 * [한국어]
 * operator<<() - Flit 정보를 스트림에 출력하는 연산자 오버로드
 *
 * @os: 출력 스트림
 * @f: 출력할 Flit 참조
 * @return: 스트림 참조 (체이닝 허용)
 *
 * 디버그 목적으로 Flit의 주요 필드(id, pid, type, head/tail, src, dest, 시간, vc)를 출력.
 * gWatchOut를 통한 추적(watch) 기능이 활성화된 Flit에 주로 사용된다.
 *
 * 호출 체인: 디버그 코드 → *gWatchOut << f
 */
ostream& operator<<( ostream& os, const Flit& f )
{
  os << "  Flit ID: " << f.id << " (" << &f << ")" // [한국어] Flit 고유 ID 및 메모리 주소
     << " Packet ID: " << f.pid  // [한국어] 패킷 ID (같은 패킷의 모든 flit이 동일)
     << " Type: " << f.type      // [한국어] FlitType 열거형 값 (0~4)
     << " Head: " << f.head      // [한국어] head flit 여부
     << " Tail: " << f.tail << endl; // [한국어] tail flit 여부
  os << "  Source: " << f.src << "  Dest: " << f.dest << " Intm: "<<f.intm<<endl; // [한국어] 출발/목적지/중간 노드 icntID
  os << "  Creation time: " << f.ctime << " Injection time: " << f.itime << " Arrival time: " << f.atime << " Phase: "<<f.ph<< endl; // [한국어] 각 시점의 NoC 사이클 및 라우팅 단계
  os << "  VC: " << f.vc << endl; // [한국어] 사용 중인 가상 채널 번호
  return os;
}

/*
 * [한국어]
 * Flit::Flit() - 기본 생성자 (private)
 *
 * 직접 생성 금지 — 반드시 Flit::New()를 사용해야 한다.
 * Reset()을 호출하여 모든 필드를 기본값으로 초기화한다.
 *
 * 호출 체인: Flit::New() → (new Flit 시) → [이 생성자] → Reset()
 */
Flit::Flit()
{
  Reset(); // [한국어] 모든 필드 기본값 초기화
}

/*
 * [한국어]
 * Flit::Reset() - Flit 필드 전체를 기본값으로 초기화한다
 *
 * New()가 풀에서 재사용 flit을 꺼낼 때, 이전 패킷의 상태가 남지 않도록
 * 반드시 이 함수를 호출하여 초기화한다.
 *
 * 주의: la_route_set은 여기서 초기화되지 않는다 (la_route_set.Clear()는 별도 호출).
 *
 * 호출 체인: Flit::Flit() → [이 함수]
 *         또는 Flit::New() → (재사용 시) → [이 함수]
 */
void Flit::Reset()
{
  type      = ANY_TYPE ; // [한국어] 타입 미지정 (패킷 생성 시 실제 타입으로 덮어씀)
  vc        = -1 ;       // [한국어] VC 미할당 상태 (-1은 아직 VC가 배정되지 않음을 의미)
  cl        = -1 ;       // [한국어] 클래스 미지정 (실제 사용 전 설정)
  head      = false ;    // [한국어] head flit 여부 초기화
  tail      = false ;    // [한국어] tail flit 여부 초기화
  ctime     = -1 ;       // [한국어] 생성 시각 미설정
  itime     = -1 ;       // [한국어] 주입 시각 미설정
  atime     = -1 ;       // [한국어] 도착 시각 미설정
  id        = -1 ;       // [한국어] Flit ID 미설정 (unsigned long long에 -1 대입 = 최대값)
  pid       = -1 ;       // [한국어] 패킷 ID 미설정
  hops      = 0 ;        // [한국어] 홉 카운터 0으로 초기화
  watch     = false ;    // [한국어] 추적 비활성화
  record    = false ;    // [한국어] 통계 기록 비활성화
  intm = 0;              // [한국어] 중간 목적지 초기화 (두 번 초기화됨 — 하단의 intm=-1이 최종값)
  src = -1;              // [한국어] 출발 노드 미설정
  dest = -1;             // [한국어] 목적지 노드 미설정
  pri = 0;               // [한국어] 우선순위 0 (최저)
  intm =-1;              // [한국어] 중간 목적지 미설정 (위의 intm=0을 덮어씀, 의도적 중복)
  ph = -1;               // [한국어] 라우팅 단계 미설정 (-1 = 단계 없음)
  data = 0;              // [한국어] 데이터 포인터 NULL로 초기화
}

/*
 * [한국어]
 * Flit::New() - 풀에서 Flit을 할당하거나 새로 생성한다
 *
 * @return: 초기화된 Flit 포인터 (절대 NULL 반환 없음)
 *
 * 풀(_free)이 비어있으면 new Flit을 생성하여 _all에 등록.
 * 풀에 재사용 가능한 Flit이 있으면 꺼내어 Reset() 후 반환.
 *
 * 이 패턴으로 시뮬레이션 중 동적 할당/해제 오버헤드를 최소화한다.
 * 재사용 시 생성자가 호출되지 않으므로 Reset()이 초기화를 담당.
 *
 * 호출 체인: _GeneratePacket() → [이 함수]
 */
Flit * Flit::New() {
  Flit * f;
  if(_free.empty()) { // [한국어] 재사용 가능한 flit이 없으면 새로 할당
    f = new Flit;     // [한국어] 새 Flit 생성 (생성자에서 Reset() 호출됨)
    _all.push(f);     // [한국어] FreeAll() 대상으로 _all에 등록
  } else {            // [한국어] 풀에 재사용 가능한 flit이 있으면
    f = _free.top();  // [한국어] 스택 top에서 flit 포인터 가져오기
    f->Reset();       // [한국어] 이전 패킷 상태 초기화 (생성자 대신 수동 호출)
    _free.pop();      // [한국어] 풀에서 제거
  }
  return f; // [한국어] 초기화된 flit 반환
}

/*
 * [한국어]
 * Flit::Free() - Flit을 _free 풀에 반환한다
 *
 * 이 함수는 메모리를 실제로 해제하지 않는다. _free 스택에 포인터를 push하여
 * 다음 New() 호출 시 재사용될 수 있게 한다.
 *
 * 호출 체인: _RetireFlit() → [이 함수]
 */
void Flit::Free() {
  _free.push(this); // [한국어] 이 flit을 재사용 풀에 반납 (실제 delete 없음)
}

/*
 * [한국어]
 * Flit::FreeAll() - _all 스택의 모든 Flit을 실제 delete한다
 *
 * 시뮬레이션 종료 시 한 번 호출하여 모든 Flit 메모리를 해제한다.
 * _all에는 New()에서 new로 생성된 모든 Flit이 등록되어 있다.
 * _free에 반환되었거나 현재 사용 중인 Flit 모두 _all에 포함된다.
 *
 * 주의: FreeAll() 이후 Flit 포인터를 사용하면 use-after-free 오류 발생.
 *
 * 호출 체인: 시뮬레이션 종료 코드 → [이 함수]
 */
void Flit::FreeAll() {
  while(!_all.empty()) { // [한국어] _all 스택이 빌 때까지 반복
    delete _all.top();   // [한국어] 스택 top의 Flit 실제 메모리 해제
    _all.pop();          // [한국어] 스택에서 제거
  }
}
