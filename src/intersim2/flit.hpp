// $Id: flit.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] NoC 전송 기본 단위 Flit 헤더 (flit.hpp)
 *
 * === 파일의 역할 ===
 * Flit(Flow Control Unit)은 BookSim2 NoC 시뮬레이터에서 라우터 간 전송의 기본 단위이다.
 * 하나의 mem_fetch 패킷은 크기에 따라 여러 개의 Flit으로 분할된다.
 * 각 Flit은 HEAD/BODY/TAIL 중 하나이며, 라우터는 HEAD flit의 목적지 정보를 보고
 * 라우팅을 수행한다. 이 파일은 Flit 구조체와 풀(pool) 기반 메모리 관리를 선언한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   InterconnectInterface::Push() → _GeneratePacket() → Flit::New()로 Flit 생성
 *   GPUTrafficManager::_Step() → 라우터 파이프라인 통과
 *   GPUTrafficManager::_RetireFlit() → Flit::Free()로 Flit 반환
 *   InterconnectInterface 소멸자 → Flit::FreeAll()로 전체 해제
 *
 * 실행 컨텍스트: GPGPU-Sim 메인 시뮬레이션 루프 (단일 스레드).
 *
 * === 타 모듈과의 연결 ===
 * - GPUTrafficManager: Flit을 생성하고 (_GeneratePacket), 소멸시킴 (_RetireFlit)
 * - VC (Virtual Channel): 각 VC의 버퍼에 Flit을 저장하는 deque<Flit*>
 * - FlitChannel: 라우터 간 링크를 통해 Flit을 물리 전송
 * - routefunc: HEAD flit의 type 필드로 VC 범위를 결정하여 라우팅
 * - InterconnectInterface: Flit.data(=mem_fetch*)로 원본 패킷 포인터 유지
 *
 * === 주요 함수/구조체 요약 ===
 * - Flit::New(): 풀에서 Flit 재사용 (없으면 new 할당) — 성능 최적화
 * - Flit::Free(): Flit을 _free 스택에 반환 (실제 delete 없음)
 * - Flit::FreeAll(): 시뮬레이션 종료 시 모든 Flit 실제 delete
 * - Flit::Reset(): 모든 필드를 기본값으로 초기화
 * - FlitType enum: READ_REQUEST/READ_REPLY/WRITE_REQUEST/WRITE_REPLY/ANY_TYPE
 *   → 라우팅 함수에서 VC 범위를 결정하는 데 사용
 */

#ifndef _FLIT_HPP_
#define _FLIT_HPP_

#include <iostream>  // [한국어] ostream: operator<< 정의에 사용
#include <stack>     // [한국어] stack<Flit*>: 풀 기반 메모리 재사용에 사용

#include "booksim.hpp"   // [한국어] BookSim 공통 헤더 (gN, gK 등 글로벌 변수)
#include "outputset.hpp" // [한국어] OutputSet: la_route_set 필드 타입

/*
 * [한국어]
 * Flit - NoC 전송의 기본 단위 (Flow Control Unit)
 *
 * 하나의 GPU 메모리 요청(mem_fetch)은 여러 개의 Flit으로 분할된다.
 * HEAD Flit은 목적지(dest)와 패킷 타입(type)을 가지며, 라우팅 결정이 이루어진다.
 * BODY Flit은 데이터만 운반하며, TAIL Flit은 패킷 종료를 표시한다.
 * 단일 Flit 패킷은 head=true && tail=true 상태이다.
 *
 * 메모리 관리: 동적 할당/해제 비용 절감을 위해 _all/_free 스택 기반 풀을 사용한다.
 * New()는 _free에서 꺼내 Reset() 후 반환하고, Free()는 _free에 push만 한다.
 * 실제 delete는 시뮬레이션 종료 시 FreeAll()에서만 수행된다.
 *
 * 동기화: GPGPU-Sim은 단일 스레드 시뮬레이션이므로 별도 락 불필요.
 */
class Flit {

public:

  const static int NUM_FLIT_TYPES = 5; // [한국어] FlitType 열거형의 값 수 (배열 크기 지정에 사용)

  /*
   * [한국어]
   * FlitType - Flit(=패킷)의 메모리 요청 종류를 구분하는 열거형
   *
   * 라우팅 함수(routefunc.cpp)는 Flit의 type에 따라 사용할 VC 범위를 결정한다.
   * 예: READ_REQUEST → gReadReqBeginVC..gReadReqEndVC (하위 VC)
   *     READ_REPLY   → gReadReplyBeginVC..gReadReplyEndVC (상위 VC)
   * 이 분리는 요청/응답 간 데드락(Deadlock)을 방지하기 위한 핵심 메커니즘이다.
   */
  enum FlitType { READ_REQUEST  = 0, // [한국어] L1 캐시 미스로 발생한 메모리 읽기 요청 (SM → 메모리)
		  READ_REPLY    = 1, // [한국어] 메모리 읽기 응답 데이터 (메모리 → SM)
		  WRITE_REQUEST = 2, // [한국어] 메모리 쓰기 요청 (SM → 메모리, write-invalidate 등)
		  WRITE_REPLY   = 3, // [한국어] 쓰기 완료 확인 응답 (메모리 → SM)
                  ANY_TYPE      = 4  // [한국어] 타입 미지정 (초기화 시 기본값; GPU 시뮬레이션에서는 실제로 사용하면 오류)
  };

  FlitType type;
  /* [한국어] 이 Flit(패킷)의 메모리 요청 종류.
   * 설정자: _GeneratePacket()이 mem_fetch 타입을 변환하여 설정.
   * 읽는 자: routefunc.cpp의 라우팅 함수들이 VC 범위 결정에 사용.
   *          _RetireFlit()이 _requestsOutstanding 감소 조건 판별에 사용.
   * 값 범위: FlitType 열거형 값 (0~4).
   * 동기화: 단일 스레드 전용. */

  int vc;
  /* [한국어] 이 Flit이 사용하는 가상 채널(Virtual Channel) 번호.
   * 설정자: _Step()의 VC 할당 로직이 head flit 처리 시 설정; body/tail은 head에서 전달.
   *         Reset() 시 -1로 초기화 (미할당 상태).
   * 읽는 자: 라우터의 VC 할당기, WriteOutBuffer()가 ejection 버퍼 인덱스로 사용.
   * 값 범위: 0 .. num_vcs-1, 또는 -1(미할당).
   * 동기화: 단일 스레드. */

  int cl;
  /* [한국어] 트래픽 클래스(Class) 번호.
   * 설정자: _GeneratePacket()이 cl 파라미터로 설정. GPU 시뮬레이션에서는 항상 0.
   * 읽는 자: _total_in_flight_flits[cl], _sent_flits[cl] 등 통계 배열 인덱스.
   * 값 범위: 0 .. _classes-1. GPU에서는 단일 클래스(0)만 사용.
   * 동기화: 단일 스레드. */

  bool head;
  /* [한국어] 이 Flit이 패킷의 첫 번째 flit(HEAD)인지 여부.
   * 설정자: _GeneratePacket()에서 i==0일 때 true로 설정.
   * 읽는 자: 라우터 라우팅 로직 (HEAD에서만 목적지 계산), Transfer2BoundaryBuffer.
   * 값 범위: true/false.
   * 동기화: 단일 스레드. */

  bool tail;
  /* [한국어] 이 Flit이 패킷의 마지막 flit(TAIL)인지 여부.
   * 설정자: _GeneratePacket()에서 i==(size-1)일 때 true로 설정.
   * 읽는 자: _BoundaryBufferItem이 패킷 완성 여부 판별, _RetireFlit()이 패킷 통계 기록.
   * 값 범위: true/false. head==true && tail==true이면 단일-flit 패킷.
   * 동기화: 단일 스레드. */

  int  ctime;
  /* [한국어] 패킷 생성 시각 (Creation time, NoC 사이클 단위).
   * 설정자: _GeneratePacket()이 호출 시점의 _traffic_manager->_time으로 설정.
   * 읽는 자: _RetireFlit()이 plat (packet latency) = atime - ctime 계산에 사용.
   * 값 범위: 0 이상 정수. Reset() 시 -1. */

  int  itime;
  /* [한국어] Flit이 실제로 NoC에 주입된 시각 (Injection time, NoC 사이클).
   * 설정자: _Step()에서 WriteFlit() 직전에 _time으로 설정.
   * 읽는 자: _RetireFlit()이 nlat (network latency) = atime - itime 계산.
   * 값 범위: 0 이상 정수. Reset() 시 -1. */

  int  atime;
  /* [한국어] Flit이 목적지 노드에 도착한 시각 (Arrival time, NoC 사이클).
   * 설정자: _Step()의 크레딧 반환 단계에서 f->atime = _time으로 설정.
   * 읽는 자: _RetireFlit()이 flat (flit latency) = atime - itime,
   *          plat = atime - ctime 계산에 사용.
   * 값 범위: 0 이상 정수. Reset() 시 -1. */

  unsigned long long  id;
  /* [한국어] 전역적으로 고유한 Flit 식별자.
   * 설정자: _GeneratePacket()이 _cur_id++로 순차 할당.
   * 읽는 자: _total_in_flight_flits 맵의 키, 디버그 출력.
   * 값 범위: 0 이상 monotonically increasing. Reset() 시 -1.
   * 동기화: _cur_id는 TrafficManager 멤버로 단일 스레드에서 증가. */

  unsigned long long  pid;
  /* [한국어] 이 Flit이 속하는 패킷의 고유 식별자 (Packet ID).
   * 설정자: _GeneratePacket()이 _cur_pid++로 패킷마다 할당; 같은 패킷의 모든 flit은 동일 pid.
   * 읽는 자: _RetireFlit()이 head flit을 _retired_packets 맵에서 찾는 키.
   *          VC::AddFlit()이 패킷 연속성 검증에 사용 (_expected_pid).
   * 값 범위: 0 이상 monotonically increasing. Reset() 시 -1. */

  bool record;
  /* [한국어] 이 Flit의 지연 통계를 측정/기록할지 여부.
   * 설정자: _GeneratePacket()이 _measure_stats[cl]에 따라 설정 (warming_up/running 상태).
   * 읽는 자: _RetireFlit()이 _measured_in_flight_flits 관리 및 통계 AddSample에 사용.
   * 값 범위: true(측정 구간)/false(warming up 이전).
   * 동기화: 단일 스레드. */

  int  src;
  /* [한국어] 이 Flit(패킷)의 출발 노드 icntID.
   * 설정자: _GeneratePacket()이 source 파라미터(icntID)로 설정.
   * 읽는 자: 디버그 출력, _pair_stats 통계 배열 인덱스(src*_nodes+dest).
   * 값 범위: 0 .. _nodes-1. Reset() 시 -1. */

  int  dest;
  /* [한국어] 이 Flit의 목적지 노드 icntID.
   * 설정자: _GeneratePacket()에서 head flit(i==0)에만 설정; body/tail은 -1.
   * 읽는 자: 라우팅 함수(routefunc.cpp)가 출구 포트 계산에 사용.
   *          _RetireFlit()이 도착 검증 (f->dest != dest이면 오류).
   * 값 범위: head flit이면 0.._nodes-1, body/tail이면 -1. Reset() 시 -1. */

  int  pri;
  /* [한국어] 이 Flit의 우선순위 값.
   * 설정자: _GeneratePacket()이 _pri_type에 따라 설정
   *         (class_based: 클래스 우선순위, age_based: INT_MAX - time).
   *         VC::UpdatePriority()가 VC 버퍼 내 우선순위 donating 수행.
   * 읽는 자: _Step()의 VC 선택 루프에서 우선순위 비교 (f->pri >= cf->pri 조건).
   * 값 범위: 0 이상 정수. Reset() 시 0. */

  int  hops;
  /* [한국어] 이 Flit이 통과한 라우터(hop) 수.
   * 설정자: 라우터 파이프라인에서 전달될 때마다 증가.
   * 읽는 자: _RetireFlit()이 _hop_stats에 AddSample. 경로 분석에 사용.
   * 값 범위: 0 이상 정수. Reset() 시 0. */

  bool watch;
  /* [한국어] 이 Flit을 상세 추적(gWatchOut 출력)할지 여부.
   * 설정자: _GeneratePacket()이 _flits_to_watch/packets_to_watch 집합 확인으로 설정.
   * 읽는 자: _Step(), FlitChannel::ReadInputs/WriteOutputs, VC::SetState 등 디버그 분기.
   * 값 범위: true/false. Reset() 시 false. */

  int  subnetwork;
  /* [한국어] 이 Flit이 속한 서브넷 번호.
   * 설정자: _GeneratePacket()에서 subnet 파라미터로 설정.
   * 읽는 자: _Step()에서 올바른 서브넷의 _input_queue에 삽입 확인.
   * 값 범위: 0 .. _subnets-1. */

  // intermediate destination (if any)
  // [한국어] Valiant/ROMM 등 다단계 라우팅의 중간 목적지 노드 icntID
  mutable int intm;
  /* [한국어] Valiant 라우팅 1단계 목적지 (랜덤 중간 노드).
   * 설정자: valiant_mesh(), romm_mesh() 등이 in_channel==2*gN 시 랜덤으로 설정.
   * 읽는 자: 같은 라우팅 함수가 ph==0일 때 목적지로 사용, ph==1로 전환 후 dest 사용.
   * 값 범위: 0 .. gNodes-1. Reset() 시 -1.
   * mutable: 라우팅 함수가 const Flit* 파라미터로 이 값을 수정해야 하므로. */

  // phase in multi-phase algorithms
  // [한국어] VALIANT/ROMM 등 다단계 라우팅에서의 현재 단계 번호
  mutable int ph;
  /* [한국어] 다단계 라우팅 알고리즘의 현재 페이즈.
   * 0: 중간 목적지(intm)로 이동 중.
   * 1: 실제 목적지(dest)로 이동 중.
   * 설정자: valiant_mesh() 등이 injection 시 0으로, intm 도착 시 1로 전환.
   * 읽는 자: 같은 라우팅 함수에서 현재 목표 노드 결정.
   * mutable: const Flit*에서도 수정 가능하도록. */

  // Fields for arbitrary data
  // [한국어] GPU 시뮬레이터에서 사용하는 실제 메모리 요청 포인터 (원본 mem_fetch*)
  void* data ;
  /* [한국어] 이 Flit이 운반하는 실제 데이터 포인터. GPGPU-Sim에서는 항상 mem_fetch*.
   * 설정자: _GeneratePacket()이 Push()로 전달된 data 파라미터를 모든 flit에 공유 저장.
   * 읽는 자: _BoundaryBufferItem::PushFlitData()가 tail flit의 data를 패킷 포인터로 반환.
   * 값 범위: 유효한 mem_fetch 포인터 (패킷의 모든 flit이 동일 포인터 공유).
   * 동기화: 단일 스레드. Flit 소멸 시에도 data 자체는 해제하지 않음(소유권 없음). */

  // Lookahead route info
  // [한국어] Lookahead 라우팅 최적화를 위한 사전 계산 경로 정보
  OutputSet la_route_set;
  /* [한국어] NOQ(Next-hop Output Queue) 등 lookahead 라우팅에서 다음 라우터의 출구 포트 집합.
   * 설정자: _Step()에서 _lookahead_routing이 활성화된 경우 _rf()로 사전 계산.
   * 읽는 자: 첫 번째 라우터가 이 정보를 사용하여 즉시 라우팅 (지연 없이).
   * 값 범위: OutputSet 객체 (AddRange/GetSet으로 접근).
   * 동기화: 단일 스레드. */

  /*
   * [한국어]
   * Reset() - Flit의 모든 필드를 기본값으로 초기화한다
   *
   * Flit::New()가 _free 풀에서 재사용 flit을 꺼낼 때 반드시 호출하여
   * 이전 패킷의 상태가 남지 않도록 한다. 기본 생성자도 이 함수를 호출한다.
   *
   * 호출 체인: Flit::New() → [이 함수] (재사용 시)
   *         또는 Flit() 생성자 → [이 함수]
   */
  void Reset();

  /*
   * [한국어]
   * New() - 풀에서 Flit을 할당하거나 새로 생성한다
   *
   * @return: 초기화된 Flit 포인터
   *
   * _free 스택이 비어있으면 new Flit을 생성하여 _all에 등록.
   * _free에 재사용 가능한 flit이 있으면 꺼내어 Reset() 후 반환.
   * 이 방식으로 시뮬레이션 중 동적 할당/해제 빈도를 대폭 줄인다.
   *
   * 호출 체인: _GeneratePacket() → [이 함수]
   */
  static Flit * New();

  /*
   * [한국어]
   * Free() - Flit을 _free 풀에 반환한다 (실제 delete 없음)
   *
   * 이 함수는 메모리를 실제로 해제하지 않고 _free 스택에 push한다.
   * 다음 New() 호출에서 재사용될 것이다.
   * 실제 delete는 FreeAll()에서만 일어난다.
   *
   * 호출 체인: _RetireFlit() → [이 함수]
   */
  void Free();

  /*
   * [한국어]
   * FreeAll() - _all 스택의 모든 Flit을 실제로 delete한다
   *
   * 시뮬레이션 종료 시 한번 호출하여 모든 Flit 객체를 해제한다.
   * _all에는 new로 생성된 모든 Flit이 등록되어 있다.
   *
   * 호출 체인: InterconnectInterface 소멸 → [이 함수]
   */
  static void FreeAll();

private:

  /*
   * [한국어]
   * Flit() - 기본 생성자 (private: 직접 생성 금지)
   *
   * Reset()을 호출하여 초기화. 외부에서는 반드시 Flit::New()를 사용해야 한다.
   */
  Flit();
  ~Flit() {} // [한국어] 소멸자: 별도 해제 없음 (FreeAll이 전담)

  static stack<Flit *> _all;
  /* [한국어] new로 생성된 모든 Flit 포인터를 추적하는 전역 스택.
   * 설정자: New()에서 new Flit 생성 시 push.
   * 읽는 자: FreeAll()이 전체 순회하며 delete.
   * 동기화: 단일 스레드 전용. */

  static stack<Flit *> _free;
  /* [한국어] 재사용 가능한 Flit 포인터를 저장하는 풀 스택.
   * 설정자: Free()가 반납된 Flit을 push.
   * 읽는 자: New()가 top()에서 꺼내 Reset() 후 재사용.
   * 동기화: 단일 스레드 전용. */

};

// [한국어] Flit 정보를 스트림에 출력하는 연산자 오버로드 (디버그용)
ostream& operator<<( ostream& os, const Flit& f );

#endif
