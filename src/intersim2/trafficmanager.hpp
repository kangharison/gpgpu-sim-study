// $Id: trafficmanager.hpp 5365 2012-11-25 02:09:59Z qtedq $

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
 * [한국어 설명] BookSim2 트래픽 매니저 기반 클래스 헤더 (trafficmanager.hpp)
 *
 * === 파일의 역할 ===
 * TrafficManager는 BookSim2의 핵심 시뮬레이션 루프를 정의하는 추상 기반 클래스다.
 * 네트워크 설정, 트래픽 패턴, 통계 수집, 패킷 생성/주입/배출/은퇴 등
 * NoC 시뮬레이션에 필요한 모든 공통 기반 기능을 제공한다.
 * GPUTrafficManager가 이 클래스를 상속하여 _GeneratePacket, _IssuePacket, _Step,
 * _RetireFlit를 GPU 시뮬레이션에 맞게 오버라이드한다.
 * GPGPU-Sim에서는 표준 Run()/_SingleSim() 루프 대신 Init()+_Step() 직접 호출 방식을 사용.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   TrafficManager::New() → GPUTrafficManager 생성 (sim_type=="gpgpusim")
 *   InterconnectInterface::Init() → GPUTrafficManager::Init()
 *   InterconnectInterface::Advance() → GPUTrafficManager::_Step()
 *   InterconnectInterface::Push() → GPUTrafficManager::_GeneratePacket()
 *
 * 실행 컨텍스트: GPGPU-Sim 메인 시뮬레이션 루프 (단일 스레드).
 *
 * === 타 모듈과의 연결 ===
 * - Network (network.hpp): _net[] 벡터로 실제 라우터 네트워크 참조
 * - Flit (flit.hpp): _total_in_flight_flits, _retired_packets 등에서 Flit* 관리
 * - BufferState (buffer_state.hpp): _buf_states[n][subnet]으로 VC 사용 가능 여부 추적
 * - Stats (stats.hpp): _plat_stats, _nlat_stats 등 통계 객체
 * - routefunc.hpp: _rf (라우팅 함수 포인터), tRoutingFunction
 * - OutputSet (outputset.hpp): 라우팅 결과 출구 포트 집합
 * - InjectionProcess (injection.hpp): 합성 트래픽 주입 패턴
 * - TrafficPattern (traffic.hpp): 합성 트래픽 목적지 생성 패턴
 *
 * === 주요 함수/구조체 요약 ===
 * - New(): 설정에 따라 GPUTrafficManager 또는 TrafficManager 생성
 * - Run(): 합성 트래픽 시뮬레이션 루프 (GPU에서는 사용 안 함)
 * - _Step(): 1 NoC 사이클 진행 (GPUTrafficManager에서 오버라이드)
 * - _RetireFlit(): 도달 Flit 통계 기록 및 Free() (GPUTrafficManager에서 오버라이드)
 * - _ClearStats(): 통계 초기화 (커널 실행 전 Init()에서 호출)
 * - _buf_states[n][subnet]: 노드 n이 보는 첫 라우터 VC 버퍼 상태 (주입 가능 여부 판단)
 * - _rf: 사용 중인 라우팅 함수 포인터 (dim_order_mesh 등, gRoutingFunctionMap에서 조회)
 */

#ifndef _TRAFFICMANAGER_HPP_
#define _TRAFFICMANAGER_HPP_

#include <list>     // [한국어] _partial_packets, _repliesPending 등에 사용
#include <map>      // [한국어] _total_in_flight_flits, _retired_packets, _stats 등에 사용
#include <set>      // [한국어] _flits_to_watch, _packets_to_watch 등에 사용
#include <cassert>  // [한국어] assert() 매크로

#include "module.hpp"        // [한국어] Module 기반 클래스 (FullName(), Error() 등 제공)
#include "config_utils.hpp"  // [한국어] Configuration 타입 (설정 파일 파싱 결과)
#include "network.hpp"       // [한국어] Network 타입 (라우터 네트워크 객체)
#include "flit.hpp"          // [한국어] Flit 타입 및 Flit::FlitType
#include "buffer_state.hpp"  // [한국어] BufferState 타입 (VC 크레딧/가용 여부 추적)
#include "stats.hpp"         // [한국어] Stats 타입 (레이턴시, 스루풋 등 통계 객체)
#include "traffic.hpp"       // [한국어] TrafficPattern 타입 (합성 목적지 생성 패턴)
#include "routefunc.hpp"     // [한국어] tRoutingFunction 타입 및 gRoutingFunctionMap
#include "outputset.hpp"     // [한국어] OutputSet 타입 (라우팅 결과 출구 포트 집합)
#include "injection.hpp"     // [한국어] InjectionProcess 타입 (합성 주입 패턴)

//register the requests to a node
// [한국어] PacketReplyInfo: 요청 패킷에 대한 응답 생성에 필요한 메타데이터를 임시 저장하는 클래스
// (BookSim 합성 트래픽용; GPU에서는 #if 0으로 비활성화됨)
class PacketReplyInfo;

/*
 * [한국어]
 * TrafficManager - BookSim2 NoC 시뮬레이션의 기반 클래스
 *
 * 합성 트래픽 시뮬레이션에서는 Run() → _SingleSim() → _Step() 루프로 동작.
 * GPU 시뮬레이션에서는 GPUTrafficManager가 상속하여 Init() + _Step() 직접 호출 방식 사용.
 *
 * 주요 상속 구조:
 *   Module (이름, 에러 처리)
 *   └── TrafficManager (공통 NoC 시뮬레이션 상태, 통계, 설정)
 *       └── GPUTrafficManager (GPU 특화: _input_queue, _GeneratePacket, _Step)
 */
class TrafficManager : public Module {

private:

  vector<vector<int> > _packet_size;
  /* [한국어] 클래스별/버킷별 패킷 크기 분포.
   * 설정자: 생성자에서 설정 파일 "packet_size" 파라미터로 초기화.
   * 읽는 자: _GetNextPacketSize()가 랜덤 샘플링.
   * 값 범위: flit 단위 양의 정수 (1 이상).
   * 동기화: 단일 스레드 전용. */

  vector<vector<int> > _packet_size_rate;
  /* [한국어] _packet_size와 대응하는 확률 비율 (가중치).
   * 설정자: 생성자에서 "packet_size_rate" 파라미터로 초기화.
   * 읽는 자: _GetNextPacketSize()에서 확률적 크기 선택에 사용.
   * 값 범위: 양의 정수 (비율, 합이 _packet_size_max_val).
   * 동기화: 단일 스레드 전용. */

  vector<int> _packet_size_max_val;
  /* [한국어] 각 클래스의 _packet_size_rate 합계 (누적 확률 최대값).
   * 설정자: 생성자에서 _packet_size_rate 합산.
   * 읽는 자: _GetNextPacketSize()에서 RandomInt() 범위 지정.
   * 값 범위: 양의 정수.
   * 동기화: 단일 스레드 전용. */

protected:
  int _nodes;
  /* [한국어] 네트워크의 총 단말 노드 수 (SM + 메모리 컨트롤러).
   * GPGPU-Sim 예: 28SM + 8메모리 = 36 노드.
   * 설정자: 생성자에서 config의 "k" × "n" (k-ary n-cube) 계산.
   * 읽는 자: 모든 노드 반복 루프, 통계 벡터 크기 결정.
   * 동기화: 단일 스레드 전용. */

  int _routers;
  /* [한국어] 네트워크 내 라우터(스위치) 총 수.
   * 설정자: 생성자에서 토폴로지 설정으로 결정.
   * 읽는 자: _router 벡터 크기, 디버그 출력.
   * 동기화: 단일 스레드 전용. */

  int _vcs;
  /* [한국어] 물리 포트당 가상 채널(VC) 수 (= gNumVCs).
   * GPGPU-Sim 기본값: 2 (요청용 VC, 응답용 VC).
   * 설정자: 생성자에서 config의 "num_vcs" 파라미터.
   * 읽는 자: _buf_states 크기, VC 범위 계산.
   * 동기화: 단일 스레드 전용. */

  vector<Network *> _net;
  /* [한국어] 서브넷별 Network 포인터 벡터.
   * 크기: [subnets] — subnets=1이면 단일 네트워크, subnets=2이면 요청/응답 분리.
   * 설정자: 생성자 파라미터로 전달받아 저장.
   * 읽는 자: _Step()에서 ReadFlit/WriteFlit/ReadCredit/WriteCredit/Evaluate/WriteOutputs.
   * 동기화: 단일 스레드 전용. */

  vector<vector<Router *> > _router;
  /* [한국어] 서브넷별/라우터별 Router 포인터 2D 벡터.
   * 크기: [subnets][routers].
   * 설정자: 생성자에서 _net[s]->GetRouter(i)로 채움.
   * 읽는 자: 라우팅 함수에서 라우터 정보 참조.
   * 동기화: 단일 스레드 전용. */

  // ============ Traffic ============

  int    _classes;
  /* [한국어] 트래픽 클래스 수 (GPGPU-Sim에서는 항상 1).
   * BookSim 합성 트래픽에서는 다중 클래스로 우선순위 차등 가능.
   * 설정자: 생성자에서 config의 "classes" 파라미터.
   * 읽는 자: 모든 클래스 반복 루프, 통계 벡터 크기 결정.
   * 동기화: 단일 스레드 전용. */

  vector<double> _load;
  /* [한국어] 클래스별 주입 부하 (offered load, 0.0~1.0).
   * 합성 트래픽에서 사이클당 패킷 주입 확률을 결정.
   * 설정자: 생성자에서 config의 "injection_rate" 파라미터.
   * 읽는 자: _IssuePacket()에서 RandomFloat() < _load[cl] 비교.
   * 동기화: 단일 스레드 전용. */

  vector<int> _use_read_write;
  /* [한국어] 클래스별 읽기/쓰기 구분 패킷 생성 여부 (0=비활성, 1=활성).
   * GPGPU-Sim에서는 0 (GPU가 타입을 직접 지정).
   * 설정자: 생성자에서 config의 "use_read_write" 파라미터.
   * 읽는 자: _GeneratePacket()에서 패킷 타입 결정.
   * 동기화: 단일 스레드 전용. */

  vector<double> _write_fraction;
  /* [한국어] 클래스별 쓰기 패킷의 비율 (0.0~1.0).
   * _use_read_write가 활성일 때 READ/WRITE 비율 결정.
   * 설정자: 생성자에서 config의 "write_fraction" 파라미터.
   * 읽는 자: _IssuePacket() 랜덤 패킷 타입 결정.
   * 동기화: 단일 스레드 전용. */

  vector<int> _read_request_size;
  /* [한국어] 클래스별 읽기 요청 패킷 크기 (flit 수).
   * 설정자: 생성자에서 config의 "read_request_size" 파라미터.
   * 읽는 자: _GeneratePacket()에서 READ_REQUEST 패킷 생성 시.
   * 동기화: 단일 스레드 전용. */

  vector<int> _read_reply_size;
  /* [한국어] 클래스별 읽기 응답 패킷 크기 (flit 수).
   * 설정자: 생성자에서 config의 "read_reply_size" 파라미터.
   * 읽는 자: _GeneratePacket()에서 READ_REPLY 패킷 생성 시.
   * 동기화: 단일 스레드 전용. */

  vector<int> _write_request_size;
  /* [한국어] 클래스별 쓰기 요청 패킷 크기 (flit 수).
   * 설정자: 생성자에서 config의 "write_request_size" 파라미터.
   * 읽는 자: _GeneratePacket()에서 WRITE_REQUEST 패킷 생성 시.
   * 동기화: 단일 스레드 전용. */

  vector<int> _write_reply_size;
  /* [한국어] 클래스별 쓰기 응답(ACK) 패킷 크기 (flit 수).
   * 설정자: 생성자에서 config의 "write_reply_size" 파라미터.
   * 읽는 자: _GeneratePacket()에서 WRITE_REPLY 패킷 생성 시.
   * 동기화: 단일 스레드 전용. */

  vector<string> _traffic;
  /* [한국어] 클래스별 합성 트래픽 패턴 이름 (예: "uniform", "hotspot").
   * 설정자: 생성자에서 config의 "traffic" 파라미터.
   * 읽는 자: TrafficPattern::New()에서 패턴 객체 생성.
   * 동기화: 단일 스레드 전용. */

  vector<int> _class_priority;
  /* [한국어] 클래스별 고정 우선순위 (class_based 우선순위 타입에 사용).
   * 설정자: 생성자에서 config의 "class_priority" 파라미터.
   * 읽는 자: _GeneratePacket()에서 pri_type==class_based이면 f->pri 설정.
   * 동기화: 단일 스레드 전용. */

  vector<vector<int> > _last_class;
  /* [한국어] 노드별/서브넷별 직전 사이클에 주입한 클래스 인덱스.
   * 크기: [nodes][subnets].
   * 라운드-로빈 공정성: (last_class + i) % _classes 순으로 클래스 탐색.
   * 설정자: _Step()이 flit 주입 후 갱신.
   * 읽는 자: _Step()의 클래스 선택 루프 시작점.
   * 동기화: 단일 스레드 전용. */

  vector<TrafficPattern *> _traffic_pattern;
  /* [한국어] 클래스별 합성 트래픽 목적지 생성 패턴 객체 (uniform random, hotspot 등).
   * 설정자: 생성자에서 TrafficPattern::New()로 생성.
   * 읽는 자: _IssuePacket()에서 목적지 노드 ID 샘플링.
   * 동기화: 단일 스레드 전용. */

  vector<InjectionProcess *> _injection_process;
  /* [한국어] 클래스별 합성 트래픽 주입 프로세스 객체 (bernoulli, on-off 등).
   * 설정자: 생성자에서 InjectionProcess::New()로 생성.
   * 읽는 자: _IssuePacket()에서 이번 사이클 주입 여부 결정.
   * 동기화: 단일 스레드 전용. */

  // ============ Message priorities ============

  /*
   * [한국어]
   * ePriority - Flit 우선순위 결정 방식 열거형
   *
   * class_based: 트래픽 클래스별 고정 우선순위 (_class_priority[] 값)
   * age_based: 패킷 생성 시각 기반 (MAX_INT - ctime; 오래될수록 높은 우선순위)
   * network_age_based: 네트워크 주입 시각 기반 (MAX_INT - itime; _Step()에서 갱신)
   * local_age_based: 큐 진입 시각 기반 (head flit ctime 기준)
   * queue_length_based: 큐 길이 기반 (긴 큐일수록 높은 우선순위)
   * hop_count_based: 남은 홉 수 기반 (많을수록 높은 우선순위)
   * sequence_based: 패킷 생성 순서 기반 (MAX_INT - _packet_seq_no[])
   * none: 우선순위 없음 (FIFO)
   */
  enum ePriority { class_based, age_based, network_age_based, local_age_based, queue_length_based, hop_count_based, sequence_based, none };

  ePriority _pri_type;
  /* [한국어] 현재 사용 중인 우선순위 결정 방식 (설정 파일의 "priority" 파라미터로 결정).
   * 설정자: 생성자에서 config의 "priority" 문자열 → 열거형으로 변환.
   * 읽는 자: _GeneratePacket()의 pri 계산 switch문, _Step()의 network_age_based 갱신.
   * 동기화: 단일 스레드 전용. */

  // ============ Injection VC states  ============

  vector<vector<BufferState *> > _buf_states;
  /* [한국어] 노드별/서브넷별 injection 포트의 다운스트림 VC 버퍼 상태 추적기.
   * 크기: [nodes][subnets]. 각 원소는 BufferState* (동적 할당).
   * 역할: 업스트림(injection 노드)에서 다운스트림(첫 라우터)의 VC 가용 여부를 추적.
   *       크레딧 기반 흐름 제어: 크레딧 수 > 0이면 전송 가능.
   * 설정자: 생성자에서 BufferState::new로 생성; _Step()의 ProcessCredit()으로 갱신.
   * 읽는 자: _Step()의 IsAvailableFor(vc)/IsFullFor(vc)/TakeBuffer()/SendingFlit().
   * 동기화: 단일 스레드 전용. */

#ifdef TRACK_FLOWS
  vector<vector<vector<int> > > _outstanding_credits;
  /* [한국어] 노드별/서브넷별/클래스별 미처리 크레딧 수.
   * 크기: [classes][subnets][nodes]. TRACK_FLOWS 모드에서만 활성화.
   * 설정자: _Step()이 flit 주입 시 ++, 크레딧 수신 시 --.
   * 읽는 자: 흐름 추적 통계 출력.
   * 동기화: 단일 스레드 전용. */

  vector<vector<vector<queue<int> > > > _outstanding_classes;
  /* [한국어] 노드별/서브넷별/VC별 미처리 클래스 큐.
   * 크기: [nodes][subnets][vcs]. TRACK_FLOWS 모드에서만 활성화.
   * 설정자: _Step()이 flit 주입 시 push(c), 크레딧 수신 시 pop().
   * 읽는 자: 크레딧 수신 시 해당 VC의 클래스를 특정하는데 사용.
   * 동기화: 단일 스레드 전용. */
#endif

  vector<vector<vector<int> > > _last_vc;
  /* [한국어] 노드별/서브넷별/클래스별 마지막으로 사용한 VC 번호.
   * 크기: [nodes][subnets][classes].
   * 라운드-로빈 VC 탐색 기준점: (last_vc에서 i번 다음) % vc_count 순으로 VC 탐색.
   * 설정자: _Step()이 head flit 주입 성공 시 _last_vc[n][subnet][c] = f->vc.
   * 읽는 자: _Step()의 VC 탐색 루프 시작점 결정.
   * 동기화: 단일 스레드 전용. */

  // ============ Routing ============

  tRoutingFunction _rf;
  /* [한국어] 현재 사용 중인 라우팅 함수 포인터 (예: dim_order_mesh).
   * 설정자: 생성자에서 gRoutingFunctionMap["routing_function"] 조회.
   * 읽는 자: _Step()에서 주입 시 inject=true 모드로 호출 (VC 범위 결정),
   *           lookahead 라우팅 시 inject=false 모드로 호출 (la_route_set 결정).
   * 동기화: 단일 스레드 전용. */

  bool _lookahead_routing;
  /* [한국어] lookahead 라우팅 활성화 여부.
   * 활성화 시: head flit 주입 전에 첫 라우터에서의 출구 포트를 미리 계산하여
   *            la_route_set에 저장. 라우터가 이를 이용해 스위치 할당 선행.
   * 설정자: 생성자에서 config의 "lookahead_routing" 파라미터.
   * 읽는 자: _Step()의 head flit 처리 로직.
   * 동기화: 단일 스레드 전용. */

  bool _noq;
  /* [한국어] Next-hop Output Queuing 활성화 여부.
   * 활성화 시: 주입 VC를 첫 홉의 출구 포트별로 분할하여 HOL 블로킹 감소.
   * 설정자: 생성자에서 config의 "noq" 파라미터.
   * 읽는 자: _Step()의 VC 탐색 로직 (분기 기준).
   * 동기화: 단일 스레드 전용. */

  // ============ Injection queues ============

  vector<vector<int> > _qtime;
  /* [한국어] 합성 트래픽의 클래스별/노드별 다음 패킷 생성 시각.
   * 크기: [classes][nodes]. 합성 트래픽 전용.
   * 설정자: _GeneratePacket() 완료 후 업데이트.
   * 읽는 자: _IssuePacket()에서 현재 시각과 비교.
   * 동기화: 단일 스레드 전용. */

  vector<vector<bool> > _qdrained;
  /* [한국어] 클래스별/노드별 주입 큐 소진 여부.
   * 크기: [classes][nodes]. draining 상태 진입 시 패킷 생성 중단 여부 추적.
   * 설정자: _Step()에서 draining 상태 시 갱신.
   * 읽는 자: _PacketsOutstanding() → drain 완료 판단.
   * 동기화: 단일 스레드 전용. */

  vector<vector<list<Flit *> > > _partial_packets;
  /* [한국어] 합성 트래픽의 클래스별/노드별 부분 완성 패킷 Flit 리스트.
   * 크기: [classes][nodes]. GPU에서는 _input_queue로 대체됨.
   * 역할: 멀티 flit 패킷이 한 사이클에 주입 완료되지 못할 때 나머지 flit 임시 저장.
   * 설정자: _GeneratePacket()이 push_back.
   * 읽는 자: _Inject()에서 pop_front()로 주입.
   * 동기화: 단일 스레드 전용. */

  vector<map<unsigned long long, Flit *> > _total_in_flight_flits;
  /* [한국어] 클래스별 현재 네트워크에 인플라이트 중인 모든 Flit 맵 (flit id → Flit*).
   * 크기: [classes]. 생성 시 삽입, _RetireFlit() 시 제거.
   * 데드락 감지: _Step()에서 flits_in_flight = !empty() 확인.
   * 설정자: _GeneratePacket()이 insert.
   * 읽는 자: _RetireFlit()이 erase; _Step()이 비었는지 확인.
   * 동기화: 단일 스레드 전용. */

  vector<map<unsigned long long, Flit *> > _measured_in_flight_flits;
  /* [한국어] 클래스별 통계 측정 대상 인플라이트 Flit 맵 (flit id → Flit*).
   * 크기: [classes]. f->record==true인 flit만 포함.
   * 설정자: _GeneratePacket()이 record==true이면 insert.
   * 읽는 자: _RetireFlit()이 erase.
   * 동기화: 단일 스레드 전용. */

  vector<map<unsigned long long, Flit *> > _retired_packets;
  /* [한국어] 클래스별 도달했지만 tail이 아직 오지 않은 패킷의 head flit 임시 저장 맵.
   * 크기: [classes]. key = pid, value = head Flit*.
   * 역할: tail flit이 도달했을 때 ctime 참조를 위해 head flit을 임시 보관.
   * 설정자: _RetireFlit()이 head이고 tail이 아닌 경우 insert; tail 도달 시 erase.
   * 읽는 자: _RetireFlit()의 tail 처리에서 head 복원.
   * 동기화: 단일 스레드 전용. */

  bool _empty_network;
  /* [한국어] 네트워크가 비어있는 상태인지 여부 (인플라이트 flit 없음).
   * 합성 트래픽에서 drain 완료 감지에 사용.
   * 설정자: _SingleSim()에서 갱신.
   * 읽는 자: _Step()의 _Inject() 호출 조건 (#if 0으로 GPU에서 비활성).
   * 동기화: 단일 스레드 전용. */

  bool _hold_switch_for_packet;
  /* [한국어] 패킷 전송 중 스위치 연결을 유지하는 설정.
   * true이면: body/tail flit이 계속 준비된 경우 같은 VC로 연속 주입 (스위치 전환 오버헤드 감소).
   * 설정자: 생성자에서 config의 "hold_switch_for_packet" 파라미터.
   * 읽는 자: _Step()의 class_limit 감소 로직.
   * 동기화: 단일 스레드 전용. */

  // ============ physical sub-networks ==========

  int _subnets;
  /* [한국어] 물리 서브넷 수 (1 또는 2).
   * subnets=1: 단일 네트워크 (요청/응답 공유, VC로 분리).
   * subnets=2: 요청(subnet 0)/응답(subnet 1) 별도 네트워크 (데드락 방지).
   * 설정자: 생성자에서 config의 "physical_subnetworks" 파라미터.
   * 읽는 자: _Step() 서브넷 루프, _input_queue 크기, 통계 벡터.
   * 동기화: 단일 스레드 전용. */

  vector<int> _subnet;
  /* [한국어] FlitType별 사용할 서브넷 인덱스 맵.
   * 크기: [Flit::NUM_FLIT_TYPES]. 예: READ_REQUEST→0, READ_REPLY→1.
   * GPGPU-Sim에서는 Push()가 직접 subnet을 지정하므로 참조 안 함.
   * 설정자: 생성자에서 config의 "subnet" 파라미터.
   * 읽는 자: _GeneratePacket() (BookSim 합성 모드에서만).
   * 동기화: 단일 스레드 전용. */

  // ============ deadlock ==========

  int _deadlock_timer;
  /* [한국어] 데드락 감시 타이머 (사이클 단위 카운터).
   * 인플라이트 flit이 있는데 _deadlock_warn_timeout 사이클 동안 배출이 없으면 경고.
   * 설정자: _Step()에서 flits_in_flight이면 ++; flit 배출 시(_RetireFlit) 0으로 리셋.
   * 읽는 자: _Step()의 임계값 비교.
   * 동기화: 단일 스레드 전용. */

  int _deadlock_warn_timeout;
  /* [한국어] 데드락 경고를 출력하는 임계값 사이클 수.
   * 설정자: 생성자에서 config의 "deadlock_warn_timeout" 파라미터.
   * 읽는 자: _Step()의 _deadlock_timer 비교.
   * 동기화: 단일 스레드 전용. */

  // ============ request & replies ==========================

  vector<int> _packet_seq_no;
  /* [한국어] 노드별 패킷 생성 순번 카운터 (sequence_based 우선순위에 사용).
   * 크기: [nodes]. 패킷 생성 시마다 ++.
   * 설정자: _GeneratePacket()에서 sequence_based 우선순위 계산 후 갱신.
   * 읽는 자: _GeneratePacket()의 pri 계산 (MAX_INT - _packet_seq_no[source]).
   * 동기화: 단일 스레드 전용. */

  vector<list<PacketReplyInfo*> > _repliesPending;
  /* [한국어] 노드별 응답 대기 중인 PacketReplyInfo 리스트.
   * 크기: [nodes]. 읽기/쓰기 요청 도달 시 추가, 응답 생성 시 제거.
   * GPGPU-Sim에서는 #if 0으로 비활성화됨 (메모리 컨트롤러가 직접 응답 생성).
   * 설정자: _RetireFlit() (요청 도달 시) — 비활성화됨.
   * 읽는 자: _GeneratePacket() (응답 생성 시) — 비활성화됨.
   * 동기화: 단일 스레드 전용. */

  vector<int> _requestsOutstanding;
  /* [한국어] 노드별 미처리 요청 수 (응답 대기 중인 READ/WRITE REQUEST 수).
   * 크기: [nodes]. 응답 도달 시 감소.
   * 설정자: _RetireFlit()에서 REPLY 도달 시 _requestsOutstanding[dest]--.
   * 읽는 자: 데드락 판정 보조 (패킷 outstanding 여부).
   * 동기화: 단일 스레드 전용. */

  // ============ Statistics ============

  vector<Stats *> _plat_stats;
  /* [한국어] 클래스별 패킷 레이턴시(plat) 통계 (ctime~atime 사이클).
   * 크기: [classes]. AddSample(f->atime - head->ctime) 호출.
   * 설정자: _RetireFlit()에서 tail flit 도달 시 AddSample.
   * 읽는 자: DisplayStats()에서 평균/최솟/최댓값 출력.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_min_plat;
  /* [한국어] 전체 시뮬레이션 누적 최소 패킷 레이턴시 (커널별 최솟값의 최솟값).
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 읽는 자: DisplayOverallStats()에서 출력.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_avg_plat;
  /* [한국어] 전체 시뮬레이션 누적 평균 패킷 레이턴시 (커널별 평균의 합 / 횟수).
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 읽는 자: DisplayOverallStats()에서 출력.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_max_plat;
  /* [한국어] 전체 시뮬레이션 누적 최대 패킷 레이턴시.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 읽는 자: DisplayOverallStats()에서 출력.
   * 동기화: 단일 스레드 전용. */

  vector<Stats *> _nlat_stats;
  /* [한국어] 클래스별 네트워크 레이턴시(nlat) 통계 (itime~atime 사이클; 주입 후 도착까지).
   * 설정자: _RetireFlit()에서 AddSample(f->atime - head->itime).
   * 읽는 자: DisplayStats()에서 출력.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_min_nlat;
  /* [한국어] 전체 누적 최소 네트워크 레이턴시.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_avg_nlat;
  /* [한국어] 전체 누적 평균 네트워크 레이턴시.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_max_nlat;
  /* [한국어] 전체 누적 최대 네트워크 레이턴시.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<Stats *> _flat_stats;
  /* [한국어] 클래스별 flit 레이턴시(flat) 통계 (itime~atime 사이클; flit 단위).
   * 설정자: _RetireFlit()에서 AddSample(f->atime - f->itime).
   * 읽는 자: DisplayStats()에서 출력.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_min_flat;
  /* [한국어] 전체 누적 최소 flit 레이턴시.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_avg_flat;
  /* [한국어] 전체 누적 평균 flit 레이턴시.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_max_flat;
  /* [한국어] 전체 누적 최대 flit 레이턴시.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<Stats *> _frag_stats;
  /* [한국어] 클래스별 패킷 단편화(frag) 통계.
   * frag = (tail.atime - head.atime) - (tail.id - head.id): 패킷 내 flit 도착 시간 분산.
   * 설정자: _RetireFlit()에서 tail 도달 시 AddSample.
   * 읽는 자: DisplayStats()에서 출력.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_min_frag;
  /* [한국어] 전체 누적 최소 패킷 단편화.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_avg_frag;
  /* [한국어] 전체 누적 평균 패킷 단편화.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_max_frag;
  /* [한국어] 전체 누적 최대 패킷 단편화.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<vector<Stats *> > _pair_plat;
  /* [한국어] 소스-목적지 쌍별 패킷 레이턴시 통계 2D 배열.
   * 크기: [classes][src*_nodes + dest].
   * 설정자: _RetireFlit()에서 _pair_stats 활성 시 AddSample.
   * 읽는 자: 쌍별 레이턴시 분포 분석.
   * 동기화: 단일 스레드 전용. */

  vector<vector<Stats *> > _pair_nlat;
  /* [한국어] 소스-목적지 쌍별 네트워크 레이턴시 통계 2D 배열.
   * 크기: [classes][src*_nodes + dest].
   * 설정자: _RetireFlit()에서 _pair_stats 활성 시 AddSample.
   * 동기화: 단일 스레드 전용. */

  vector<vector<Stats *> > _pair_flat;
  /* [한국어] 소스-목적지 쌍별 flit 레이턴시 통계 2D 배열.
   * 크기: [classes][src*_nodes + dest].
   * 설정자: _RetireFlit()에서 _pair_stats 활성 시 AddSample.
   * 동기화: 단일 스레드 전용. */

  vector<Stats *> _hop_stats;
  /* [한국어] 클래스별 패킷 홉 수(hop count) 통계.
   * 설정자: _RetireFlit()에서 tail 도달 시 AddSample(f->hops).
   * 읽는 자: DisplayStats()에서 평균 홉 수 출력.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_hop_stats;
  /* [한국어] 전체 누적 평균 홉 수.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 읽는 자: DisplayOverallStats()에서 출력.
   * 동기화: 단일 스레드 전용. */

  vector<vector<int> > _sent_packets;
  /* [한국어] 클래스별/노드별 전송한 패킷 수 (head flit 단위).
   * 크기: [classes][nodes].
   * 설정자: _Step()에서 f->head인 flit 주입 시 ++.
   * 읽는 자: DisplayStats()에서 처리량 계산.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_min_sent_packets;
  /* [한국어] 전체 누적 최소 전송 패킷 수.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_avg_sent_packets;
  /* [한국어] 전체 누적 평균 전송 패킷 수.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_max_sent_packets;
  /* [한국어] 전체 누적 최대 전송 패킷 수.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<vector<int> > _accepted_packets;
  /* [한국어] 클래스별/노드별 수신한 패킷 수 (tail flit 단위, ejection 기준).
   * 크기: [classes][nodes].
   * 설정자: _Step()에서 ejected_flit->tail이면 ++.
   * 읽는 자: DisplayStats()에서 수신 처리량 계산.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_min_accepted_packets;
  /* [한국어] 전체 누적 최소 수신 패킷 수.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_avg_accepted_packets;
  /* [한국어] 전체 누적 평균 수신 패킷 수.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_max_accepted_packets;
  /* [한국어] 전체 누적 최대 수신 패킷 수.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<vector<int> > _sent_flits;
  /* [한국어] 클래스별/노드별 전송한 flit 수 (처리량 측정의 기본 단위).
   * 크기: [classes][nodes].
   * 설정자: _Step()에서 flit 주입 시 ++.
   * 읽는 자: DisplayStats()에서 초당 flit 처리량 계산.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_min_sent;
  /* [한국어] 전체 누적 최소 전송 flit 수.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_avg_sent;
  /* [한국어] 전체 누적 평균 전송 flit 수.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_max_sent;
  /* [한국어] 전체 누적 최대 전송 flit 수.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<vector<int> > _accepted_flits;
  /* [한국어] 클래스별/노드별 수신한 flit 수 (ejection된 flit 수).
   * 크기: [classes][nodes].
   * 설정자: _Step()에서 ejected_flit이 있을 때 ++.
   * 읽는 자: DisplayStats()에서 수신 처리량 계산.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_min_accepted;
  /* [한국어] 전체 누적 최소 수신 flit 수.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_avg_accepted;
  /* [한국어] 전체 누적 평균 수신 flit 수.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_max_accepted;
  /* [한국어] 전체 누적 최대 수신 flit 수.
   * 설정자: _UpdateOverallStats()에서 갱신.
   * 동기화: 단일 스레드 전용. */

#ifdef TRACK_STALLS
  vector<vector<int> > _buffer_busy_stalls;
  /* [한국어] 클래스별/노드별 버퍼 busy(VC 예약 중) 정체 횟수. TRACK_STALLS 모드.
   * 설정자: _Step()에서 IsAvailableFor(vc)==false 시 ++.
   * 동기화: 단일 스레드 전용. */

  vector<vector<int> > _buffer_conflict_stalls;
  /* [한국어] 클래스별/노드별 버퍼 충돌 정체 횟수. TRACK_STALLS 모드.
   * 동기화: 단일 스레드 전용. */

  vector<vector<int> > _buffer_full_stalls;
  /* [한국어] 클래스별/노드별 버퍼 가득 참 정체 횟수. TRACK_STALLS 모드.
   * 설정자: _Step()에서 IsFullFor(vc)==true 시 ++.
   * 동기화: 단일 스레드 전용. */

  vector<vector<int> > _buffer_reserved_stalls;
  /* [한국어] 클래스별/노드별 버퍼 예약 정체 횟수. TRACK_STALLS 모드.
   * 동기화: 단일 스레드 전용. */

  vector<vector<int> > _crossbar_conflict_stalls;
  /* [한국어] 클래스별/노드별 크로스바 충돌 정체 횟수. TRACK_STALLS 모드.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_buffer_busy_stalls;
  /* [한국어] 전체 누적 버퍼 busy 정체 횟수 합계. TRACK_STALLS 모드.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_buffer_conflict_stalls;
  /* [한국어] 전체 누적 버퍼 충돌 정체 횟수 합계. TRACK_STALLS 모드.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_buffer_full_stalls;
  /* [한국어] 전체 누적 버퍼 가득 참 정체 횟수 합계. TRACK_STALLS 모드.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_buffer_reserved_stalls;
  /* [한국어] 전체 누적 버퍼 예약 정체 횟수 합계. TRACK_STALLS 모드.
   * 동기화: 단일 스레드 전용. */

  vector<double> _overall_crossbar_conflict_stalls;
  /* [한국어] 전체 누적 크로스바 충돌 정체 횟수 합계. TRACK_STALLS 모드.
   * 동기화: 단일 스레드 전용. */
#endif

  vector<int> _slowest_packet;
  /* [한국어] 클래스별 현재까지 가장 느린 패킷의 ID (디버깅용).
   * 크기: [classes]. 초기값 -1 (미설정).
   * 설정자: _RetireFlit()에서 _plat_stats->Max() 비교 후 갱신.
   * 읽는 자: DisplayStats()에서 최악 케이스 패킷 정보 출력.
   * 동기화: 단일 스레드 전용. */

  vector<int> _slowest_flit;
  /* [한국어] 클래스별 현재까지 가장 느린 flit의 ID (디버깅용).
   * 크기: [classes]. 초기값 -1 (미설정).
   * 설정자: _RetireFlit()에서 _flat_stats->Max() 비교 후 갱신.
   * 읽는 자: DisplayStats()에서 최악 케이스 flit 정보 출력.
   * 동기화: 단일 스레드 전용. */

  map<string, Stats *> _stats;
  /* [한국어] 이름 → Stats 포인터 맵 (외부에서 이름으로 통계 객체 조회 가능).
   * 예: _stats["plat"] = _plat_stats[0].
   * 설정자: 생성자에서 모든 Stats 객체를 이름으로 등록.
   * 읽는 자: InterconnectInterface::GetIcntStats()에서 이름으로 통계 조회.
   * 동기화: 단일 스레드 전용. */

  // ============ Simulation parameters ============

  /*
   * [한국어]
   * eSimState - 시뮬레이션 상태 머신 열거형
   *
   * warming_up: 워밍업 단계 (패킷 주입 시작, 통계 수집 시작 전)
   * running: 실행 중 (통계 수집 중, record==true인 패킷 생성)
   * draining: 소진 단계 (패킷 주입 중단, 남은 패킷 배출 대기)
   * done: 완료 (모든 패킷 배출 완료)
   */
  enum eSimState { warming_up, running, draining, done };

  eSimState _sim_state;
  /* [한국어] 현재 시뮬레이션 상태 (warming_up/running/draining/done).
   * 설정자: Init()에서 running으로 설정; _SingleSim()에서 단계 전환.
   * 읽는 자: _GeneratePacket()에서 record 결정; _RetireFlit()에서 통계 수집 여부.
   * 동기화: 단일 스레드 전용. */

  bool _measure_latency;
  /* [한국어] 레이턴시 통계 수집 활성화 여부.
   * false이면 레이턴시 계산을 건너뜀 (성능 최적화).
   * 설정자: 생성자에서 config의 "measure_latency" 파라미터.
   * 읽는 자: _RetireFlit()에서 통계 기록 조건.
   * 동기화: 단일 스레드 전용. */

  int   _reset_time;
  /* [한국어] 통계 리셋 시각 (warming_up → running 전환 기준점).
   * 합성 트래픽에서 warm-up 기간 후 측정 시작 시각.
   * 설정자: 생성자에서 config의 "sample_period" × "warmup_periods".
   * 읽는 자: _SingleSim()에서 _sim_state 전환 시 비교.
   * 동기화: 단일 스레드 전용. */

  int   _drain_time;
  /* [한국어] 드레인 시작 시각 (running → draining 전환 기준점).
   * 합성 트래픽에서 패킷 주입 중단 시각.
   * 설정자: _SingleSim()에서 수렴 확인 시 현재 시각으로 설정.
   * 읽는 자: _GeneratePacket()에서 record 결정 조건 (draining && time < drain_time).
   * 동기화: 단일 스레드 전용. */

  int   _total_sims;
  /* [한국어] 총 시뮬레이션 실행 횟수 (GPGPU-Sim에서는 CUDA 커널 수와 동일).
   * 설정자: GPUTrafficManager 생성자에서 0으로 초기화; _UpdateOverallStats()에서 ++.
   * 읽는 자: DisplayOverallStats()에서 평균 계산.
   * 동기화: 단일 스레드 전용. */

  int   _sample_period;
  /* [한국어] 통계 수집 주기 (사이클 단위).
   * 합성 트래픽에서 _SingleSim()이 샘플 수집하는 간격.
   * 설정자: 생성자에서 config의 "sample_period" 파라미터.
   * 읽는 자: _SingleSim()에서 사이클 진행 루프.
   * 동기화: 단일 스레드 전용. */

  int   _max_samples;
  /* [한국어] 최대 통계 샘플 수 (수렴 전 최대 반복 횟수).
   * 설정자: 생성자에서 config의 "max_samples" 파라미터.
   * 읽는 자: _SingleSim()에서 무한 루프 방지.
   * 동기화: 단일 스레드 전용. */

  int   _warmup_periods;
  /* [한국어] 워밍업 기간 수 (= warmup 사이클 / sample_period).
   * 설정자: 생성자에서 config의 "warmup_periods" 파라미터.
   * 읽는 자: _reset_time 계산.
   * 동기화: 단일 스레드 전용. */

  int   _include_queuing;
  /* [한국어] 레이턴시 계산 시 큐 대기 시간 포함 여부 (1=포함, 0=제외).
   * 1이면 ctime 기준(패킷 생성 포함), 0이면 itime 기준(주입 후만).
   * 설정자: 생성자에서 config의 "include_queuing" 파라미터.
   * 읽는 자: _RetireFlit()에서 plat/nlat 선택 기준.
   * 동기화: 단일 스레드 전용. */

  vector<int> _measure_stats;
  /* [한국어] 클래스별 통계 측정 활성화 여부 (0=비활성, 1=활성).
   * 크기: [classes].
   * 설정자: 생성자에서 config의 "measure_stats" 파라미터.
   * 읽는 자: _GeneratePacket()에서 record 결정 (record = _measure_stats[cl]).
   * 동기화: 단일 스레드 전용. */

  bool _pair_stats;
  /* [한국어] 소스-목적지 쌍별 상세 통계 수집 활성화 여부.
   * 활성화 시 _pair_plat/_pair_nlat/_pair_flat 통계도 기록 (메모리 오버헤드 증가).
   * 설정자: 생성자에서 config의 "pair_stats" 파라미터.
   * 읽는 자: _RetireFlit()에서 pair 통계 기록 조건.
   * 동기화: 단일 스레드 전용. */

  vector<double> _latency_thres;
  /* [한국어] 클래스별 레이턴시 임계값 (초과 시 _SingleSim 재시도).
   * 설정자: 생성자에서 config의 "latency_thres" 파라미터.
   * 읽는 자: _SingleSim()에서 수렴 판정.
   * 동기화: 단일 스레드 전용. */

  vector<double> _stopping_threshold;
  /* [한국어] 클래스별 수렴 임계값 (레이턴시 변화율 기준).
   * 합성 트래픽에서 안정 상태 판정.
   * 설정자: 생성자에서 config의 "stopping_threshold" 파라미터.
   * 읽는 자: _SingleSim()에서 수렴 판정.
   * 동기화: 단일 스레드 전용. */

  vector<double> _acc_stopping_threshold;
  /* [한국어] 클래스별 누적 수렴 임계값 (전체 평균 기준 수렴 조건).
   * 설정자: 생성자에서 config의 "acc_stopping_threshold" 파라미터.
   * 읽는 자: _SingleSim()에서 누적 수렴 판정.
   * 동기화: 단일 스레드 전용. */

  vector<double> _warmup_threshold;
  /* [한국어] 클래스별 워밍업 완료 판정 임계값 (레이턴시 안정화 기준).
   * 설정자: 생성자에서 config의 "warmup_threshold" 파라미터.
   * 읽는 자: _SingleSim()에서 warming_up → running 전환 시점 결정.
   * 동기화: 단일 스레드 전용. */

  vector<double> _acc_warmup_threshold;
  /* [한국어] 클래스별 누적 워밍업 완료 판정 임계값.
   * 설정자: 생성자에서 config의 "acc_warmup_threshold" 파라미터.
   * 읽는 자: _SingleSim()에서 누적 워밍업 판정.
   * 동기화: 단일 스레드 전용. */

  unsigned long long _cur_id;
  /* [한국어] 전역 단조 증가 flit ID 카운터 (시뮬레이션 내 모든 flit에 고유 ID 부여).
   * flit ID는 패킷 내에서 연속적: head.id, head.id+1, ..., tail.id.
   * 이 연속성을 이용해 frag 통계를 계산함 (tail.id - head.id = 패킷 내 flit 수 - 1).
   * 설정자: _GeneratePacket()에서 f->id = _cur_id++.
   * 읽는 자: _RetireFlit()의 frag 계산.
   * 동기화: 단일 스레드 전용. */

  unsigned long long _cur_pid;
  /* [한국어] 전역 단조 증가 패킷 ID 카운터 (시뮬레이션 내 모든 패킷에 고유 ID 부여).
   * 설정자: _GeneratePacket()에서 pid = _cur_pid++.
   * 읽는 자: _RetireFlit()에서 head 복원 키 (f->pid → _retired_packets 조회).
   * 동기화: 단일 스레드 전용. */

  int _time;
  /* [한국어] 현재 NoC 사이클 카운터 (Init()에서 0으로 초기화, _Step() 끝에서 ++).
   * 레이턴시 계산 기준: ctime/itime/atime 모두 이 _time 값.
   * 설정자: Init()에서 0, _Step() 끝에서 ++.
   * 읽는 자: _GeneratePacket()의 ctime 설정; _Step()의 itime/atime 설정; 통계 정규화.
   * 동기화: 단일 스레드 전용. */

  set<unsigned long long> _flits_to_watch;
  /* [한국어] 상세 추적할 flit ID 집합 (watch 출력 필터링).
   * 이 집합에 포함된 ID의 flit은 모든 이벤트를 gWatchOut에 출력.
   * 설정자: _LoadWatchList()에서 파일로부터 로드.
   * 읽는 자: _GeneratePacket()에서 f->watch = _flits_to_watch.count(f->id) > 0.
   * 동기화: 단일 스레드 전용. */

  set<unsigned long long> _packets_to_watch;
  /* [한국어] 상세 추적할 패킷 ID 집합 (watch 출력 필터링).
   * 이 집합에 포함된 ID의 패킷은 모든 flit의 이벤트를 출력.
   * 설정자: _LoadWatchList()에서 파일로부터 로드.
   * 읽는 자: _GeneratePacket()에서 watch = _packets_to_watch.count(pid) > 0.
   * 동기화: 단일 스레드 전용. */

  bool _print_csv_results;
  /* [한국어] 통계를 CSV 형식으로도 출력할지 여부.
   * 설정자: 생성자에서 config의 "print_csv_results" 파라미터.
   * 읽는 자: DisplayOverallStatsCSV() 호출 조건.
   * 동기화: 단일 스레드 전용. */

  //flits to watch
  ostream * _stats_out;
  /* [한국어] 통계 출력 스트림 포인터 (파일 또는 cout).
   * 설정자: 생성자에서 config의 "stats_out" 파라미터로 파일 열기 또는 &cout.
   * 읽는 자: DisplayStats()/WriteStats()에서 통계 출력.
   * 동기화: 단일 스레드 전용. */

#ifdef TRACK_FLOWS
  vector<vector<int> > _injected_flits;
  /* [한국어] 클래스별/노드별 주입된 flit 수. TRACK_FLOWS 모드.
   * 설정자: _Step()에서 주입 시 ++.
   * 동기화: 단일 스레드 전용. */

  vector<vector<int> > _ejected_flits;
  /* [한국어] 클래스별/노드별 배출된 flit 수. TRACK_FLOWS 모드.
   * 설정자: _Step()에서 ejection 시 ++.
   * 동기화: 단일 스레드 전용. */

  ostream * _injected_flits_out;
  /* [한국어] 주입 flit 통계 출력 스트림. TRACK_FLOWS 모드.
   * 동기화: 단일 스레드 전용. */

  ostream * _received_flits_out;
  /* [한국어] 수신 flit 통계 출력 스트림. TRACK_FLOWS 모드.
   * 동기화: 단일 스레드 전용. */

  ostream * _stored_flits_out;
  /* [한국어] 저장(인플라이트) flit 통계 출력 스트림. TRACK_FLOWS 모드.
   * 동기화: 단일 스레드 전용. */

  ostream * _sent_flits_out;
  /* [한국어] 전송 flit 통계 출력 스트림. TRACK_FLOWS 모드.
   * 동기화: 단일 스레드 전용. */

  ostream * _outstanding_credits_out;
  /* [한국어] 미처리 크레딧 통계 출력 스트림. TRACK_FLOWS 모드.
   * 동기화: 단일 스레드 전용. */

  ostream * _ejected_flits_out;
  /* [한국어] 배출 flit 통계 출력 스트림. TRACK_FLOWS 모드.
   * 동기화: 단일 스레드 전용. */

  ostream * _active_packets_out;
  /* [한국어] 활성(인플라이트) 패킷 통계 출력 스트림. TRACK_FLOWS 모드.
   * 동기화: 단일 스레드 전용. */
#endif

#ifdef TRACK_CREDITS
  ostream * _used_credits_out;
  /* [한국어] 사용 중인 크레딧 수 출력 스트림. TRACK_CREDITS 모드.
   * 동기화: 단일 스레드 전용. */

  ostream * _free_credits_out;
  /* [한국어] 남은 크레딧 수 출력 스트림. TRACK_CREDITS 모드.
   * 동기화: 단일 스레드 전용. */

  ostream * _max_credits_out;
  /* [한국어] 최대 크레딧 수 출력 스트림. TRACK_CREDITS 모드.
   * 동기화: 단일 스레드 전용. */
#endif

  // ============ Internal methods ============
protected:

  /*
   * [한국어]
   * _RetireFlit() - 목적지 도달 Flit 통계 기록 및 Free() [가상 함수]
   *
   * @f: 도달한 Flit 포인터
   * @dest: 도달 노드 icntID
   *
   * 기반 클래스 버전은 BookSim 합성 트래픽용 응답 생성 로직 포함.
   * GPUTrafficManager가 오버라이드하여 GPU 특화 버전 제공.
   *
   * 호출 체인: _Step() (크레딧 발행 단계) → [이 함수]
   */
  virtual void _RetireFlit( Flit *f, int dest );

  /*
   * [한국어]
   * _Inject() - 합성 트래픽 패킷을 주입한다 (GPU에서는 #if 0으로 비활성화)
   *
   * 각 노드에서 InjectionProcess에 따라 _IssuePacket() 호출.
   * GPUTrafficManager::_Step()에서 이 함수 호출이 #if 0으로 비활성화됨.
   *
   * 호출 체인: TrafficManager::_Step() → [이 함수] (GPU에서는 미호출)
   */
  void _Inject();

  /*
   * [한국어]
   * _Step() - 1 NoC 사이클 진행 [가상 함수]
   *
   * 기반 클래스는 합성 트래픽용 표준 BookSim 사이클 진행.
   * GPUTrafficManager가 오버라이드하여 GPU 시뮬레이션에 맞게 변경.
   *
   * 호출 체인: Run()/_SingleSim() → [이 함수]
   *            InterconnectInterface::Advance() → GPUTrafficManager::_Step() (GPU)
   */
  virtual void _Step( );

  /*
   * [한국어]
   * _PacketsOutstanding() - 아직 완료되지 않은 패킷이 있는지 확인
   *
   * @return: true이면 인플라이트 또는 큐에 패킷 잔존
   *
   * draining 상태에서 모든 패킷이 배출됐는지 확인하는데 사용.
   *
   * 호출 체인: _SingleSim()에서 drain 완료 판정 → [이 함수]
   */
  bool _PacketsOutstanding( ) const;

  /*
   * [한국어]
   * _IssuePacket() - 합성 트래픽 패킷 생성 [가상 함수]
   *
   * @source: 송신 노드
   * @cl: 트래픽 클래스
   * @return: 생성한 패킷 수 (0=미생성)
   *
   * GPUTrafficManager가 항상 0을 반환하는 버전으로 오버라이드.
   *
   * 호출 체인: _Inject() → [이 함수]
   */
  virtual int  _IssuePacket( int source, int cl );

  /*
   * [한국어]
   * _GeneratePacket() - Flit 생성 [가상 함수]
   *
   * @source: 송신 노드
   * @size: 패킷 크기 (flit 수)
   * @cl: 트래픽 클래스
   * @time: 생성 시각
   *
   * GPUTrafficManager가 확장 시그니처로 오버라이드
   * (packet_type, data, dest 파라미터 추가).
   *
   * 호출 체인: _IssuePacket() 또는 InterconnectInterface::Push() → [이 함수]
   */
  virtual void _GeneratePacket( int source, int size, int cl, int time );

  /*
   * [한국어]
   * _ClearStats() - 모든 통계를 초기화한다
   *
   * _plat_stats, _nlat_stats, _flat_stats 등 모든 Stats 객체 Clear().
   * _sent_flits, _accepted_flits 등 카운터 벡터를 0으로 초기화.
   * GPUTrafficManager::Init()이 커널 실행 전 호출.
   *
   * 호출 체인: Init() → [이 함수]
   */
  virtual void _ClearStats( );

  /*
   * [한국어]
   * _ComputeStats() - 노드 단위 통계 벡터에서 합계/최솟/최댓값 계산
   *
   * @stats: 노드별 카운터 벡터
   * @sum: 합계 (출력)
   * @min: 최솟값 (출력, NULL이면 계산 안 함)
   * @max: 최댓값 (출력, NULL이면 계산 안 함)
   * @min_pos: 최솟값 위치 (출력, NULL이면 계산 안 함)
   * @max_pos: 최댓값 위치 (출력, NULL이면 계산 안 함)
   *
   * 호출 체인: WriteStats()/UpdateStats() → [이 함수]
   */
  void _ComputeStats( const vector<int> & stats, int *sum, int *min = NULL, int *max = NULL, int *min_pos = NULL, int *max_pos = NULL ) const;

  /*
   * [한국어]
   * _SingleSim() - 단일 시뮬레이션 실행 (warming_up → running → draining → done)
   *
   * @return: true이면 수렴 성공 (통계 유효), false이면 실패
   *
   * 합성 트래픽에서만 사용. GPU에서는 Init() + _Step() 직접 호출 방식 사용.
   *
   * 호출 체인: Run() → [이 함수]
   */
  virtual bool _SingleSim( );

  /*
   * [한국어]
   * _DisplayRemaining() - 미완료 패킷 정보를 출력한다
   *
   * @os: 출력 스트림 (기본 cout)
   *
   * 데드락 또는 조기 종료 시 인플라이트/큐 잔존 패킷 정보 출력.
   *
   * 호출 체인: Run() 에러 처리 → [이 함수]
   */
  void _DisplayRemaining( ostream & os = cout ) const;

  /*
   * [한국어]
   * _LoadWatchList() - flit/패킷 watch 목록을 파일에서 로드
   *
   * @filename: watch list 파일 경로
   *
   * 파일에서 flit ID / 패킷 ID를 읽어 _flits_to_watch / _packets_to_watch에 삽입.
   *
   * 호출 체인: 생성자 → [이 함수]
   */
  void _LoadWatchList(const string & filename);

  /*
   * [한국어]
   * _UpdateOverallStats() - 현재 시뮬레이션의 통계를 overall 누적 통계에 반영
   *
   * _overall_avg_plat 등 누적 평균/최솟/최댓값 갱신.
   * ++_total_sims.
   *
   * 호출 체인: Run() → _SingleSim() 완료 후 → [이 함수]
   */
  virtual void _UpdateOverallStats();

  /*
   * [한국어]
   * _OverallStatsCSV() - 전체 누적 통계를 CSV 형식 문자열로 반환
   *
   * @c: 클래스 인덱스 (기본 0)
   * @return: "avg_plat,avg_nlat,..." 형식 문자열
   *
   * 호출 체인: DisplayOverallStatsCSV() → [이 함수]
   */
  virtual string _OverallStatsCSV(int c = 0) const;

  /*
   * [한국어]
   * _GetNextPacketSize() - 클래스의 다음 패킷 크기를 랜덤 샘플링
   *
   * @cl: 트래픽 클래스 인덱스
   * @return: flit 단위 패킷 크기
   *
   * _packet_size / _packet_size_rate 분포에서 가중 랜덤 샘플링.
   *
   * 호출 체인: _IssuePacket() → [이 함수]
   */
  int _GetNextPacketSize(int cl) const;

  /*
   * [한국어]
   * _GetAveragePacketSize() - 클래스의 평균 패킷 크기 계산
   *
   * @cl: 트래픽 클래스 인덱스
   * @return: 가중 평균 패킷 크기 (flit 단위, double)
   *
   * 호출 체인: DisplayStats() → [이 함수]
   */
  double _GetAveragePacketSize(int cl) const;

public:

  /*
   * [한국어]
   * New() - 설정에 따라 TrafficManager 또는 GPUTrafficManager를 생성하는 팩토리 함수
   *
   * @config: BookSim 설정 파일 파싱 결과
   * @net: 서브넷별 Network 포인터 벡터
   * @return: 생성된 TrafficManager* (실제로는 GPUTrafficManager*)
   *
   * config의 "sim_type" 파라미터를 확인:
   *   "gpgpusim" → new GPUTrafficManager()
   *   기타 → new TrafficManager() (합성 트래픽 시뮬레이션용)
   *
   * 호출 체인: InterconnectInterface::CreateInterconnect() → [이 함수]
   */
  static TrafficManager * New(Configuration const & config,
			      vector<Network *> const & net);

  /*
   * [한국어]
   * TrafficManager() - 기반 클래스 생성자
   *
   * @config: BookSim 설정 파일 파싱 결과
   * @net: 서브넷별 Network 포인터 벡터
   *
   * 설정 파일에서 모든 파라미터를 읽어 멤버 변수를 초기화:
   *   - 토폴로지: _nodes, _routers, _vcs, _subnets
   *   - 트래픽: _classes, _traffic, _load, _traffic_pattern, _injection_process
   *   - 우선순위: _pri_type, _class_priority
   *   - 라우팅: _rf, _lookahead_routing, _noq
   *   - 통계: _plat_stats, _nlat_stats, _flat_stats, _frag_stats, _hop_stats 등
   *   - VC 상태: _buf_states, _last_vc, _last_class
   *   - 시뮬레이션: _total_sims, _sample_period, _max_samples, _sim_state
   *
   * 호출 체인: TrafficManager::New() → [이 함수] 또는 GPUTrafficManager::GPUTrafficManager()
   */
  TrafficManager( const Configuration &config, const vector<Network *> & net );

  /*
   * [한국어]
   * ~TrafficManager() - 소멸자
   *
   * _net, _buf_states, Stats*, TrafficPattern*, InjectionProcess* 등 동적 자원 해제.
   *
   * 호출 체인: InterconnectInterface::~InterconnectInterface() → [이 함수]
   */
  virtual ~TrafficManager( );

  /*
   * [한국어]
   * Run() - 합성 트래픽 시뮬레이션 메인 루프 실행
   *
   * @return: true이면 수렴 성공
   *
   * _max_samples번 _SingleSim()을 반복 실행.
   * GPU에서는 사용하지 않음 (Advance()가 _Step() 직접 호출).
   *
   * 호출 체인: main() (표준 BookSim 모드) → [이 함수]
   */
  bool Run( );

  /*
   * [한국어]
   * WriteStats() - 통계를 스트림에 출력한다
   *
   * @os: 출력 스트림 (기본 cout)
   *
   * 현재 수집된 모든 통계(_plat_stats 등)를 텍스트 형식으로 출력.
   *
   * 호출 체인: InterconnectInterface::DisplayStats() → [이 함수]
   */
  virtual void WriteStats( ostream & os = cout ) const ;

  /*
   * [한국어]
   * UpdateStats() - 통계 객체를 사이클 단위로 업데이트한다
   *
   * 이동 평균 등 주기적 업데이트가 필요한 통계 갱신.
   *
   * 호출 체인: InterconnectInterface::DisplayStats() → [이 함수]
   */
  virtual void UpdateStats( ) ;

  /*
   * [한국어]
   * DisplayStats() - 현재 통계 요약을 출력한다
   *
   * @os: 출력 스트림 (기본 cout)
   *
   * 레이턴시, 스루풋, 홉 수 요약 출력.
   *
   * 호출 체인: InterconnectInterface::DisplayStats() → [이 함수]
   */
  virtual void DisplayStats( ostream & os = cout ) const ;

  /*
   * [한국어]
   * DisplayOverallStats() - 전체 누적 통계 요약을 출력한다
   *
   * @os: 출력 스트림 (기본 cout)
   *
   * _total_sims 횟수에 걸친 누적 평균/최솟/최댓값 출력.
   *
   * 호출 체인: InterconnectInterface::DisplayOverallStats() → [이 함수]
   */
  virtual void DisplayOverallStats( ostream & os = cout ) const ;

  /*
   * [한국어]
   * DisplayOverallStatsCSV() - 전체 누적 통계를 CSV 형식으로 출력한다
   *
   * @os: 출력 스트림 (기본 cout)
   *
   * _print_csv_results가 true일 때 호출.
   *
   * 호출 체인: InterconnectInterface::DisplayOverallStats() → [이 함수]
   */
  virtual void DisplayOverallStatsCSV( ostream & os = cout ) const ;

  /*
   * [한국어]
   * getTime() - 현재 NoC 사이클 시각 반환
   *
   * @return: _time (현재 사이클 번호)
   *
   * 호출 체인: InterconnectInterface::GetIcntTime() → [이 함수]
   */
  inline int getTime() { return _time;}

  /*
   * [한국어]
   * getStats() - 이름으로 통계 객체를 조회한다
   *
   * @name: 통계 이름 문자열 (예: "plat", "nlat")
   * @return: Stats* (없으면 NULL 또는 쓰레기 값)
   *
   * 호출 체인: InterconnectInterface::GetIcntStats() → [이 함수]
   */
  Stats * getStats(const string & name) { return _stats[name]; }

};

/*
 * [한국어]
 * operator<<(ostream&, vector<T>&) - 벡터를 쉼표 구분 문자열로 출력하는 템플릿 연산자
 *
 * @os: 출력 스트림
 * @v: 출력할 벡터
 * @return: os 참조 (체이닝)
 *
 * 예: {1, 2, 3} → "1,2,3" (마지막 원소 뒤 쉼표 없음)
 * DisplayStats()에서 노드별 카운터 출력에 사용.
 */
template<class T>
ostream & operator<<(ostream & os, const vector<T> & v) {
  for(size_t i = 0; i < v.size() - 1; ++i) { // [한국어] 마지막 원소를 제외하고 쉼표 뒤에 출력
    os << v[i] << ",";
  }
  os << v[v.size()-1]; // [한국어] 마지막 원소는 쉼표 없이 출력
  return os; // [한국어] 체이닝을 위해 스트림 반환
}

#endif
