// $Id: packet_reply_info.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] BookSim NoC 요청 패킷 응답 추적 클래스 헤더 (packet_reply_info.hpp)
 *
 * === 파일의 역할 ===
 * NoC 시뮬레이터 내에서 요청 패킷(READ_REQUEST / WRITE_REQUEST)이 목적지 노드에
 * 도달한 시점을 추적하고, 해당 요청에 대한 응답 패킷(READ_REPLY / WRITE_REPLY)을
 * 언제, 어디로 생성해야 하는지를 기록하는 PacketReplyInfo 클래스를 선언한다.
 * 생성/해제 비용을 줄이기 위해 정적 풀(pool) 기반의 재사용 메커니즘(New/Free)을
 * 사용하며, FreeAll()로 시뮬레이션 종료 시 한 번에 모든 메모리를 해제한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2/ 계층에서 트래픽 생성기(trafficmanager.cc)와 연계하여 동작한다.
 * 실행 컨텍스트: 호스트(CPU) 유저스페이스 — 매 시뮬레이션 사이클의 패킷 처리 단계.
 * 호출 체인:
 *   TrafficManager::_Inject() — 요청 패킷 주입 시 PacketReplyInfo::New()로 생성
 *   → NoC 통과 → TrafficManager::_Eject() — 도착 시 PacketReplyInfo로 응답 패킷 생성
 *   → PacketReplyInfo::Free() — 사용 완료 후 풀에 반환
 *   → (시뮬레이션 종료) PacketReplyInfo::FreeAll() — 전체 메모리 해제
 *
 * === 타 모듈과의 연결 ===
 * 의존: <stack> (std::stack으로 _all, _free 풀 관리), flit.hpp (FlitType 열거형 참조).
 * 피의존: trafficmanager.cc (요청/응답 패킷 처리에서 New/Free 호출).
 * 데이터 흐름: 요청 패킷 주입 시 생성 → 패킷과 함께 저장 → 도착 시 응답 생성에 사용.
 * 공유 자료구조: _all, _free (정적 멤버 스택 — 단일 전역 풀로 모든 인스턴스 공유).
 *
 * === 주요 함수/구조체 요약 ===
 * source        - 요청 패킷의 출발 노드 번호 (응답을 보낼 목적지)
 * time          - 요청 패킷이 주입된 사이클 번호 (지연 측정용)
 * record        - 이 요청의 지연을 통계에 기록할지 여부
 * type          - 요청 패킷의 FlitType (READ_REQUEST 또는 WRITE_REQUEST)
 * New()         - 정적 풀에서 인스턴스를 꺼내거나 새로 할당하여 반환
 * Free()        - 이 인스턴스를 _free 스택에 반환 (메모리 해제 없이 재사용)
 * FreeAll()     - _all 스택의 모든 인스턴스를 delete로 해제 (시뮬레이션 종료 시 호출)
 *
 * 관련 설정:
 *   - intersim2 설정의 sim_type, injection_rate, packet_size, reply_rate 등이
 *     요청/응답 패킷의 생성 빈도와 크기를 결정하여 PacketReplyInfo의 사용량에 영향.
 *   - GPGPU-Sim 모드에서는 gpgpusim.config의 메모리 요청 특성(요청 크기, 타입)이
 *     응답 생성 시점을 간접적으로 결정한다.
 */

#ifndef _PACKET_REPLY_INFO_HPP_ // [한국어] 헤더 중복 포함 방지 가드 시작
#define _PACKET_REPLY_INFO_HPP_ // [한국어] 가드 매크로 정의

#include <stack> // [한국어] std::stack: _all(전체 인스턴스 추적)과 _free(재사용 풀) 관리에 사용

#include "flit.hpp" // [한국어] Flit::FlitType 열거형 (READ_REQUEST, WRITE_REQUEST 등) 참조를 위해 포함

//register the requests to a node
/*
 * [한국어]
 * PacketReplyInfo - 요청 패킷의 응답 생성에 필요한 정보를 담는 클래스.
 *
 * 요청 패킷(READ_REQUEST/WRITE_REQUEST)이 NoC를 통해 목적지 노드에 도달하면,
 * 해당 목적지는 응답 패킷(READ_REPLY/WRITE_REPLY)을 생성하여 출발지로 돌려보내야 한다.
 * PacketReplyInfo는 이 과정에서 필요한 정보(출발지 노드, 주입 시각, 패킷 타입 등)를
 * 한 군데 모아 보관한다.
 *
 * 메모리 관리: 생성자/소멸자를 private으로 선언하고 New()/Free()/FreeAll() 정적 함수를
 * 통해서만 인스턴스를 생성/반환/소멸한다. 이를 통해 풀(pool) 기반 재사용을 구현하여
 * 매 사이클마다 발생하는 new/delete 비용을 제거한다.
 *
 * 동기화: trafficmanager는 단일 스레드에서 실행되므로 _all, _free 스택에 대한
 *         별도 락이 없다. GPGPU-Sim의 icnt_wrapper는 별도 스레드에서 호출되지만
 *         PacketReplyInfo는 시뮬레이션 사이클 내에서만 사용되어 충돌하지 않는다.
 */
class PacketReplyInfo {

public:
  int source;
  /* [한국어] 요청 패킷이 출발한 노드 번호 (응답 패킷의 목적지).
   * 요청이 목적지에 도달하면, 이 source 노드로 응답 패킷을 돌려보낸다.
   * 설정자: TrafficManager::_Inject() — 요청 패킷 주입 시 발신 노드 번호로 설정.
   * 읽는 자: TrafficManager::_Eject() — 응답 패킷 생성 시 목적지로 사용.
   * 값 범위: 0 이상, 네트워크 노드 수 미만의 정수.
   * 동기화: 단일 trafficmanager 스레드에서만 접근, 락 불필요. */

  int time;
  /* [한국어] 요청 패킷이 주입된 시뮬레이션 사이클 번호.
   * 응답 패킷이 출발지에 도달한 시각과 이 값의 차이로 왕복 지연(latency)을 계산한다.
   * 설정자: TrafficManager::_Inject() — 패킷 주입 시 현재 사이클 번호로 설정.
   * 읽는 자: TrafficManager::_RetirePacket() 또는 지연 통계 집계 코드.
   * 값 범위: 0 이상의 정수 (시뮬레이션 사이클 번호).
   * 동기화: 단일 trafficmanager 스레드에서만 접근, 락 불필요. */

  bool record;
  /* [한국어] 이 요청 패킷의 지연(latency)을 통계에 기록할지 여부.
   * warm-up 기간 동안 주입된 패킷은 record=false로 설정하여 통계에서 제외한다.
   * 설정자: TrafficManager::_Inject() — 워밍업 여부에 따라 true/false로 설정.
   * 읽는 자: TrafficManager::_RetirePacket() — true인 경우에만 지연 통계 누산.
   * 값 범위: true(통계 기록) 또는 false(통계 제외).
   * 동기화: 단일 trafficmanager 스레드에서만 접근, 락 불필요. */

  Flit::FlitType type;
  /* [한국어] 요청 패킷의 FlitType (READ_REQUEST 또는 WRITE_REQUEST).
   * 응답 패킷을 생성할 때 요청 종류에 맞는 응답 타입(READ_REPLY / WRITE_REPLY)을
   * 결정하는 데 사용된다.
   * 설정자: TrafficManager::_Inject() — 요청 패킷의 flit 타입으로 설정.
   * 읽는 자: TrafficManager::_Eject() — 응답 패킷의 타입 결정 시 참조.
   * 값 범위: Flit::READ_REQUEST(0) 또는 Flit::WRITE_REQUEST(2).
   * 동기화: 단일 trafficmanager 스레드에서만 접근, 락 불필요. */

  /*
   * [한국어]
   * PacketReplyInfo::New - 풀에서 인스턴스를 꺼내거나 새로 할당하여 반환한다.
   *
   * @return: 초기화되지 않은 PacketReplyInfo 포인터.
   *          호출자가 source, time, record, type 필드를 직접 설정해야 한다.
   *
   * _free 스택이 비어 있으면 new로 새 인스턴스를 할당하고 _all에 등록한다.
   * _free 스택에 반환된 인스턴스가 있으면 이를 꺼내어 재사용한다.
   * 이를 통해 매 사이클마다 발생하는 동적 할당 비용을 줄인다.
   * 실행 컨텍스트: 호스트 CPU, 매 사이클 패킷 주입 단계.
   *
   * 호출 체인:
   *   TrafficManager::_Inject() → [PacketReplyInfo::New()] → 인스턴스 반환
   */
  static PacketReplyInfo* New();

  /*
   * [한국어]
   * PacketReplyInfo::Free - 이 인스턴스를 _free 풀에 반환한다.
   *
   * @return: 없음 (void).
   *
   * 메모리를 즉시 해제(delete)하지 않고 _free 스택에 push하여 나중에
   * New()가 재사용할 수 있게 한다. 이를 통해 동적 할당 빈도를 줄인다.
   * 실행 컨텍스트: 호스트 CPU, 매 사이클 패킷 도착 처리 단계.
   *
   * 호출 체인:
   *   TrafficManager::_Eject() → [pr->Free()] → _free.push(this)
   */
  void Free();

  /*
   * [한국어]
   * PacketReplyInfo::FreeAll - _all 풀의 모든 인스턴스를 delete로 해제한다.
   *
   * @return: 없음 (void).
   *
   * 시뮬레이션 종료 시 한 번 호출하여 New()로 생성된 모든 인스턴스의
   * 메모리를 해제한다. _all 스택이 빌 때까지 top()을 delete하고 pop()한다.
   * 이 함수 이후에는 _all 및 _free 스택 모두 비워진다.
   * 실행 컨텍스트: 호스트 CPU, 시뮬레이션 종료 정리 단계.
   *
   * 호출 체인:
   *   TrafficManager 소멸자 / 시뮬레이터 종료 코드 → [PacketReplyInfo::FreeAll()]
   */
  static void FreeAll();

private:

  static stack<PacketReplyInfo*> _all;
  /* [한국어] New()로 할당된 모든 PacketReplyInfo 인스턴스 포인터를 추적하는 정적 스택.
   * 역할: FreeAll() 호출 시 모든 인스턴스를 한 번에 delete하기 위한 전체 목록 관리.
   * 설정자: New()가 new 할당 시 _all.push(pr)로 등록.
   * 읽는 자: FreeAll()이 while(!_all.empty()) 루프로 순회하며 delete.
   * 값 범위: 0개 이상의 PacketReplyInfo* 포인터.
   * 동기화: 정적 멤버이므로 모든 인스턴스 공유; 단일 trafficmanager 스레드에서만 접근. */

  static stack<PacketReplyInfo*> _free;
  /* [한국어] 반환된(Free() 호출) 인스턴스 포인터를 보관하는 재사용 풀 정적 스택.
   * 역할: New()가 새 인스턴스를 할당하기 전에 이 스택에서 재사용 가능한 인스턴스를 꺼냄.
   * 설정자: Free()가 _free.push(this)로 반환된 인스턴스를 등록.
   * 읽는 자: New()가 _free.empty() 확인 후 비어 있지 않으면 _free.top()으로 꺼냄.
   * 값 범위: 0개 이상의 PacketReplyInfo* 포인터 (현재 미사용 인스턴스들).
   * 동기화: 정적 멤버이므로 모든 인스턴스 공유; 단일 trafficmanager 스레드에서만 접근. */

  /*
   * [한국어]
   * PacketReplyInfo() - private 기본 생성자. New() 정적 함수만이 호출 가능.
   *
   * 필드를 초기화하지 않으므로 New() 호출 후 반드시 source, time, record, type을 설정해야 한다.
   * private 선언으로 외부에서 new PacketReplyInfo()를 직접 호출하지 못하게 막는다.
   */
  PacketReplyInfo() {}

  /*
   * [한국어]
   * ~PacketReplyInfo() - private 소멸자. FreeAll()만이 호출 가능.
   *
   * 멤버 필드가 포인터나 복잡한 객체를 포함하지 않으므로 소멸자 본문은 비어 있다.
   * private 선언으로 외부에서 delete를 직접 호출하지 못하게 막는다.
   * FreeAll() 내에서만 delete를 통해 호출된다.
   */
  ~PacketReplyInfo() {}
};

#endif // [한국어] _PACKET_REPLY_INFO_HPP_ 헤더 가드 끝
