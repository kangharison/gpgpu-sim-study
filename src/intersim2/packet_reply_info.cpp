// $Id: packet_reply_info.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] BookSim NoC 요청 패킷 응답 추적 클래스 구현 (packet_reply_info.cpp)
 *
 * === 파일의 역할 ===
 * packet_reply_info.hpp에 선언된 PacketReplyInfo 클래스의 풀(pool) 기반
 * 메모리 관리(New/Free/FreeAll)를 구현한다.
 * PacketReplyInfo는 READ_REQUEST/WRITE_REQUEST 등의 요청 패킷이 NoC를 통해
 * 목적지에 도달했을 때, 그에 대응하는 READ_REPLY/WRITE_REPLY 응답 패킷을
 * 생성하기 위해 필요한 정보(출발 노드, 주입 시각, 기록 여부, 요청 타입)를
 * 저장하는 작은 데이터 객체이다.
 * 시뮬레이션 중 매우 빈번하게 생성/소멸되므로 new/delete 비용을 줄이기 위해
 * 정적 스택 기반의 객체 풀을 사용한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2/ 계층의 트래픽 매니저(trafficmanager.cc)와 연계하여 동작한다.
 * 실행 컨텍스트: 호스트(CPU) 유저스페이스 — 매 시뮬레이션 사이클의 패킷 처리 단계.
 * 호출 체인:
 *   TrafficManager::_Inject() — 요청 패킷 주입 시 PacketReplyInfo::New()로 생성
 *   → NoC 통과 → TrafficManager::_Eject() — 도착 시 PacketReplyInfo로 응답 패킷 생성
 *   → PacketReplyInfo::Free() — 사용 완료 후 풀에 반환
 *   → (시뮬레이션 종료) PacketReplyInfo::FreeAll() — 전체 메모리 해제
 *
 * === 타 모듈과의 연결 ===
 * 의존: packet_reply_info.hpp (클래스 선언), flit.hpp (Flit::FlitType 정의).
 * 피의존: trafficmanager.cc (합성 트래픽 모드), gputrafficmanager.cpp
 *         (GPU 모드에서 mem_fetch 기반 요청/응답 처리)에서 New/Free 호출.
 * 데이터 흐름: 요청 패킷 주입 시 생성 → 패킷과 함께 저장 → 도착 시 응답 생성에 사용.
 * 공유 자료구조: _all, _free (정적 멤버 스택 — 단일 전역 풀로 모든 인스턴스 공유).
 *
 * 관련 설정:
 *   - intersim2 설정의 sim_type, injection_rate, packet_size, reply_rate 등이
 *     요청/응답 패킷의 생성 빈도와 크기를 결정하여 PacketReplyInfo의 사용량에 영향.
 *   - GPGPU-Sim 모드에서는 gpgpusim.config의 메모리 요청 특성(요청 크기, 타입)이
 *     응답 생성 시점을 간접적으로 결정한다.
 *
 * === 주요 함수/구조체 요약 ===
 * New()     - 정적 풀에서 인스턴스를 꺼내거나 새로 할당하여 반환
 * Free()    - 이 인스턴스를 _free 스택에 반환 (메모리 해제 없이 재사용)
 * FreeAll() - _all 스택의 모든 인스턴스를 delete로 해제 (시뮬레이션 종료 시 호출)
 */

#include "packet_reply_info.hpp" // [한국어] PacketReplyInfo 클래스 선언 포함

// [한국어] New()로 생성된 모든 PacketReplyInfo 포인터를 추적하는 정적 스택 (FreeAll()에서 실제 delete)
stack<PacketReplyInfo*> PacketReplyInfo::_all;

// [한국어] Free()로 반납되어 재사용 대기 중인 PacketReplyInfo 포인터 스택 (New()에서 재사용)
stack<PacketReplyInfo*> PacketReplyInfo::_free;

/*
 * [한국어]
 * PacketReplyInfo::New - 풀에서 PacketReplyInfo 인스턴스를 하나 가져오거나,
 *                        풀이 비었으면 새로 할당한다.
 *
 * @return: PacketReplyInfo* — 호출자가 source/time/record/type 필드를 직접 설정해야 한다.
 *
 * _free 스택이 비어 있으면 new로 새 인스턴스를 생성하고 _all에 등록한다.
 * _free에 반납된 인스턴스가 있으면 top()에서 꺼내 재사용한다.
 * 생성자에서 별도 초기화를 수행하지 않으므로, 호출자는 반환 직후 필드를 채워야 한다.
 *
 * 호출 체인:
 *   TrafficManager::_Inject() / _GeneratePacket() → [PacketReplyInfo::New()]
 */
PacketReplyInfo * PacketReplyInfo::New()
{
  PacketReplyInfo * pr; // [한국어] 반환할 인스턴스 포인터
  if(_free.empty()) { // [한국어] 재사용 가능한 인스턴스가 없으면 새로 할당
    pr = new PacketReplyInfo(); // [한국어] 힙에 새 PacketReplyInfo 생성 (private 생성자)
    _all.push(pr);              // [한국어] FreeAll()에서 해제할 수 있도록 전체 목록에 등록
  } else {                      // [한국어] 반납된 인스턴스가 있으면 재사용
    pr = _free.top();           // [한국어] _free 스택 상단의 인스턴스 포인터를 꺼냄
    _free.pop();                // [한국어] 스택에서 제거 (이제 활성 사용 중)
  }
  return pr; // [한국어] 초기화되지 않은 인스턴스 반환 — 호출자가 필드 설정 책임
}

/*
 * [한국어]
 * PacketReplyInfo::Free - 사용 완료된 인스턴스를 _free 풀에 반납한다.
 *
 * @return: 없음 (void).
 *
 * 실제 메모리를 해제하지 않고 _free 스택에 this를 push하여
 * 이후 New() 호출 시 재사용될 수 있게 한다.
 * 시뮬레이션 종료 시에만 FreeAll()이 _all의 인스턴스를 실제로 delete한다.
 *
 * 호출 체인:
 *   TrafficManager::_Eject() / _RetirePacket() → [pr->Free()]
 */
void PacketReplyInfo::Free()
{
  _free.push(this); // [한국어] 현재 인스턴스 포인터를 재사용 풀에 반납
}

/*
 * [한국어]
 * PacketReplyInfo::FreeAll - _all 스택에 등록된 모든 인스턴스를 delete로 해제한다.
 *
 * @return: 없음 (void).
 *
 * 시뮬레이션 종료 시 한 번 호출하여 New()로 생성된 모든 PacketReplyInfo의
 * 메모리를 해제한다. _all이 빌 때까지 top()을 delete하고 pop()한다.
 * 이 함수 호출 이후에는 어떤 PacketReplyInfo 포인터도 사용해서는 안 된다.
 * _free 스택은 _all이 소유한 포인터의 부분집합이므로 _all을 전부 delete하면
 * _free에 남아 있는 포인터들도 무효화된다.
 *
 * 호출 체인:
 *   TrafficManager 소멸자 / 시뮬레이터 종료 코드 → [PacketReplyInfo::FreeAll()]
 */
void PacketReplyInfo::FreeAll()
{
  while(!_all.empty()) { // [한국어] _all 스택이 빌 때까지 반복
    delete _all.top();   // [한국어] 스택 상단의 PacketReplyInfo 인스턴스를 실제로 delete
    _all.pop();          // [한국어] delete된 포인터를 스택에서 제거
  }
}
