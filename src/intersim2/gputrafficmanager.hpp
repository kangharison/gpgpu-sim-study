// Copyright (c) 2009-2013, Tor M. Aamodt, Dongdong Li, Ali Bakhoda
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/*
 * [한국어 설명] GPU 특화 트래픽 매니저 헤더 (gputrafficmanager.hpp)
 *
 * === 파일의 역할 ===
 * TrafficManager를 GPU 시뮬레이션 환경에 맞게 특화한 서브클래스를 선언한다.
 * 표준 BookSim2의 트래픽 생성/측정 메커니즘 대신 GPGPU-Sim이 직접 패킷을 생성하므로,
 * _IssuePacket() 등 일부 가상 함수를 GPU 특화 버전으로 오버라이드한다.
 * 핵심 차이점: _input_queue가 TrafficManager의 _partial_packets 대신 사용되며,
 * _GeneratePacket() 서명도 GPU 시뮬레이션에 맞게 변경되었다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   InterconnectInterface::Push() → _GeneratePacket() (Flit 생성 및 _input_queue 삽입)
 *   InterconnectInterface::Advance() → _Step() (1사이클 NoC 진행)
 *   _Step() 내부 → _RetireFlit() (목적지 도달 flit 통계 처리)
 *   _Step() 내부 → g_icnt_interface->WriteOutBuffer/Transfer2BoundaryBuffer
 *
 * 실행 컨텍스트: GPGPU-Sim 메인 시뮬레이션 루프 (단일 스레드).
 *
 * === 타 모듈과의 연결 ===
 * - TrafficManager (trafficmanager.hpp): 기반 클래스 (통계, 설정, 라우팅 등 상속)
 * - InterconnectInterface: friend 관계로 _input_queue 등 protected 멤버 직접 접근
 * - Network: _net[] 벡터 통해 실제 라우터 네트워크와 통신
 * - Flit: _GeneratePacket에서 Flit::New()로 생성, _RetireFlit에서 Free()
 *
 * === 주요 함수/구조체 요약 ===
 * - _input_queue[subnets][nodes][classes]: Push()로 삽입된 Flit 대기열
 * - _GeneratePacket(): mem_fetch → n_flits개 Flit 생성하여 _input_queue에 삽입
 * - _Step(): 1 NoC 사이클 전체 파이프라인 실행
 * - _RetireFlit(): 도착 Flit 통계 처리 및 Free()
 * - Init(): 시뮬레이션 상태 초기화 (_time=0, running)
 */

#ifndef _GPUTRAFFICMANAGER_HPP_
#define _GPUTRAFFICMANAGER_HPP_

#include <iostream>  // [한국어] 디버그 출력
#include <vector>    // [한국어] _input_queue 다차원 벡터
#include <list>      // [한국어] _input_queue 원소 타입 (list<Flit*>)

#include "config_utils.hpp"   // [한국어] Configuration 타입
#include "stats.hpp"          // [한국어] Stats 타입 (통계 객체)
#include "trafficmanager.hpp" // [한국어] TrafficManager 기반 클래스
#include "booksim.hpp"        // [한국어] BookSim 공통 정의
#include "booksim_config.hpp" // [한국어] BookSim 설정 파싱
#include "flit.hpp"           // [한국어] Flit 및 Flit::FlitType

/*
 * [한국어]
 * GPUTrafficManager - GPU 시뮬레이션 특화 트래픽 매니저
 *
 * TrafficManager의 서브클래스로, 합성 트래픽 생성 대신 GPGPU-Sim이 직접
 * 패킷을 _GeneratePacket()으로 주입하는 방식을 지원한다.
 *
 * 주요 변경사항:
 *   - _IssuePacket(): GPU 시뮬레이션에서는 사용하지 않으므로 항상 0 반환
 *   - _GeneratePacket(): GPU 시뮬레이션용 서명 (packet_type, data, dest 추가)
 *   - _input_queue: TrafficManager::_partial_packets 대신 직접 Flit 큐 관리
 *   - _Step(): g_icnt_interface 콜백으로 ejection/credit 처리 통합
 *   - Init(): _total_sims=0으로 초기화 (커널 단위 통계)
 */
class GPUTrafficManager : public TrafficManager {

protected:
  /*
   * [한국어]
   * _RetireFlit() - 목적지에 도달한 Flit을 처리하고 통계를 기록한다
   *
   * @f: 목적지 도달 Flit 포인터
   * @dest: 실제 도달한 노드 icntID
   *
   * 동작 순서:
   *   1. _deadlock_timer 리셋 (데드락 감시)
   *   2. _total_in_flight_flits에서 제거
   *   3. flat_stats (flit latency = atime - itime) 기록
   *   4. tail flit이면 plat/nlat/frag_stats (패킷 레이턴시) 기록
   *   5. READ_REPLY/WRITE_REPLY이면 _requestsOutstanding 감소
   *   6. tail flit이면 head flit Free(), 아니면 _retired_packets에 head 보관
   *   7. Flit::Free()로 풀에 반환
   *
   * 주의: GPGPU-Sim에서는 응답 생성을 메모리 컨트롤러가 처리하므로
   *       reply 생성 코드(#if 0 블록)는 비활성화됨.
   *
   * 호출 체인: _Step() (크레딧 반환 단계) → [이 함수]
   */
  virtual void _RetireFlit( Flit *f, int dest );

  /*
   * [한국어]
   * _GeneratePacket() - mem_fetch 패킷으로부터 Flit들을 생성하여 _input_queue에 삽입한다
   *
   * @source: 송신 노드 icntID
   * @stype: 예약 파라미터 (GPU에서는 -1, 0이면 assert 실패)
   * @cl: 트래픽 클래스 (GPU에서는 항상 0)
   * @time: 패킷 생성 시각 (ctime으로 저장)
   * @subnet: 사용할 서브넷 (0=요청, 1=응답)
   * @packet_size: 패킷의 flit 수
   * @packet_type: Flit::FlitType (READ_REQUEST 등)
   * @data: 원본 mem_fetch 포인터 (모든 Flit에 공유 저장)
   * @dest: 목적지 icntID (HEAD flit에만 설정)
   *
   * 동작 순서:
   *   1. 패킷 ID(pid) 및 각 Flit ID(id) 순차 할당
   *   2. packet_size개의 Flit::New()로 Flit 생성
   *   3. i==0: head=true, dest=packet_destination
   *   4. i==size-1: tail=true
   *   5. 우선순위 설정 (_pri_type에 따라)
   *   6. VC = -1로 미할당 상태
   *   7. _input_queue[subnet][source][cl]에 push_back
   *
   * 호출 체인: InterconnectInterface::Push() → [이 함수]
   */
  virtual void _GeneratePacket(int source, int stype, int cl, int time, int subnet, int package_size, const Flit::FlitType& packet_type, void* const data, int dest);

  /*
   * [한국어]
   * _IssuePacket() - 합성 패킷 생성 (GPU에서는 사용하지 않음)
   *
   * @source: 송신 노드 (사용 안 함)
   * @cl: 클래스 (사용 안 함)
   * @return: 항상 0
   *
   * TrafficManager의 기존 패킷 생성 메커니즘을 비활성화.
   * GPGPU-Sim에서는 InterconnectInterface::Push()가 직접 _GeneratePacket()을 호출.
   *
   * 호출 체인: TrafficManager::_Inject() → [이 함수] (실제 호출 안 됨, #if 0으로 막혀있음)
   */
  virtual int  _IssuePacket( int source, int cl );

  /*
   * [한국어]
   * _Step() - NoC 시뮬레이션 1사이클을 진행한다
   *
   * 매 사이클의 실행 순서:
   *   1. 데드락 감시 (_deadlock_timer 증가, 임계값 초과 시 경고)
   *   2. 모든 서브넷/노드에서 배출 flit 수집 → WriteOutBuffer()
   *   3. Transfer2BoundaryBuffer() → GetEjectedFlit() → 크레딧 반환 준비
   *   4. 네트워크 크레딧 읽기 → _buf_states[n][subnet]->ProcessCredit()
   *   5. _net[subnet]->ReadInputs() (모든 FlitChannel 입력 캡처)
   *   6. 각 노드에서 _input_queue → 라우팅/VC할당 → _net[subnet]->WriteFlit()
   *      - head flit: _rf()로 lookahead 라우팅 계산
   *      - VC 할당: 가용 VC를 라운드-로빈으로 탐색
   *   7. 크레딧 반환: Credit::New() + WriteCredit() + _RetireFlit()
   *   8. _net[subnet]->Evaluate() + WriteOutputs()
   *   9. ++_time
   *
   * 호출 체인: InterconnectInterface::Advance() → [이 함수]
   */
  virtual void _Step();

  // record size of _partial_packets for each subnet
  // [한국어] GPU 시뮬레이션용 입력 큐: 서브넷별/노드별/클래스별 Flit 대기 리스트
  // 크기: [subnets][nodes][classes] (TrafficManager의 _partial_packets 대체)
  vector<vector<vector<list<Flit *> > > > _input_queue;
  /* [한국어] Push()로 삽입된 Flit들이 _Step()에 의해 네트워크에 주입되기 전까지 대기하는 큐.
   * 설정자: _GeneratePacket()이 push_back으로 Flit 삽입.
   * 읽는 자: _Step()이 front()에서 Flit을 꺼내 네트워크에 WriteFlit().
   * 값 범위: list<Flit*>, 크기 <= _input_buffer_capacity.
   * 동기화: 단일 스레드 전용. */

public:

  /*
   * [한국어]
   * GPUTrafficManager() - GPU 트래픽 매니저 생성자
   *
   * @config: BookSim 설정 파싱 결과
   * @net: 서브넷별 Network 포인터 벡터
   *
   * TrafficManager 기반 클래스 초기화 후:
   *   - _total_sims = 0 (커널 실행 횟수 카운터 초기화)
   *   - _input_queue를 [subnets][nodes][classes] 형태로 초기화
   *
   * 호출 체인: TrafficManager::New() (sim_type=="gpgpusim") → [이 함수]
   */
  GPUTrafficManager( const Configuration &config, const vector<Network *> & net );

  /*
   * [한국어]
   * ~GPUTrafficManager() - 소멸자
   *
   * 기반 클래스(TrafficManager) 소멸자가 네트워크 및 통계 자원 해제.
   * GPUTrafficManager 고유 자원(vector 등)은 자동 소멸.
   *
   * 호출 체인: InterconnectInterface::~InterconnectInterface() → [이 함수]
   */
  virtual ~GPUTrafficManager( );

  // correspond to TrafficManger::Run/SingleSim
  /*
   * [한국어]
   * Init() - 시뮬레이션 상태를 초기화한다 (커널 실행 시작 전 호출)
   *
   * _time = 0, _sim_state = running으로 설정하고
   * _ClearStats()로 이전 커널의 통계를 초기화한다.
   * 각 CUDA 커널 실행 전에 호출되어야 한다.
   *
   * 호출 체인: InterconnectInterface::Init() → [이 함수]
   */
  void Init();

  // TODO: if it is not good...
  // [한국어] InterconnectInterface가 이 클래스의 protected 멤버
  // (_input_queue, _time, _total_in_flight_flits 등)를 직접 접근하기 위한 friend 선언
  friend class InterconnectInterface;



  //    virtual void WriteStats( ostream & os = cout ) const;
  //    virtual void DisplayStats( ostream & os = cout ) const;
  //    virtual void DisplayOverallStats( ostream & os = cout ) const;

};



#endif
