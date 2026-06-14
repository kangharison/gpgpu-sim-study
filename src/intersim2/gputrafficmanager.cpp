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
 * [한국어 설명] GPU 특화 트래픽 매니저 구현 (gputrafficmanager.cpp)
 *
 * === 파일의 역할 ===
 * GPUTrafficManager 클래스의 모든 메서드를 구현한다.
 * 핵심 함수는 _Step()으로, 매 NoC 사이클마다 InterconnectInterface::Advance()가 호출하여
 * flit 배출 → 크레딧 반환 → 주입(injection) → 라우팅/VC 할당 → 네트워크 평가 순으로 파이프라인을 진행한다.
 * 표준 BookSim2의 합성 트래픽 생성(_Inject) 대신, GPGPU-Sim이 Push()를 통해 직접 Flit을 생성하는
 * _GeneratePacket()을 사용한다. 따라서 _IssuePacket()은 항상 0을 반환하는 더미다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   GPGPU-Sim cycle() → InterconnectInterface::Advance() → GPUTrafficManager::_Step()
 *   GPGPU-Sim mem_fetch Push → InterconnectInterface::Push() → _GeneratePacket()
 *   _Step() → g_icnt_interface->WriteOutBuffer/Transfer2BoundaryBuffer/GetEjectedFlit
 *   _Step() → _RetireFlit() (통계 기록 및 Flit::Free())
 *
 * 실행 컨텍스트: GPGPU-Sim 메인 시뮬레이션 루프 (단일 스레드, 매 사이클 1회 호출).
 *
 * === 타 모듈과의 연결 ===
 * - InterconnectInterface (interconnect_interface.hpp): g_icnt_interface 전역을 통해
 *   WriteOutBuffer/Transfer2BoundaryBuffer/GetEjectedFlit 콜백 호출
 * - TrafficManager (trafficmanager.hpp): 기반 클래스로 _time, _subnets, _nodes, _classes,
 *   _net, _buf_states, _rf 등 모든 상태 상속
 * - Flit (flit.hpp): _GeneratePacket에서 Flit::New(), _RetireFlit에서 Flit::Free()
 * - Network (network.hpp): ReadFlit/WriteCredit/ReadCredit/WriteFlit/Evaluate/WriteOutputs
 * - Credit (credit.hpp): 배출 후 업스트림 크레딧 반환에 사용
 * - BufferState (buffer_state.hpp): 주입 전 VC 가용 여부 확인에 사용
 *
 * === 주요 함수/구조체 요약 ===
 * - GPUTrafficManager(): _total_sims=0, _input_queue 3D 초기화
 * - Init(): _time=0, _sim_state=running, 통계 초기화 (커널 단위)
 * - _GeneratePacket(): mem_fetch로부터 packet_size개 Flit 생성, _input_queue에 삽입
 * - _RetireFlit(): 도달 Flit 통계(flit/packet latency, hops) 기록 후 Free()
 * - _Step(): 1 NoC 사이클 전체 파이프라인 (배출→크레딧→주입→평가)
 */

#include <sstream>  // [한국어] ostringstream (에러 메시지 포맷)
#include <fstream>  // [한국어] 파일 출력 (통계 저장용, 직접 사용 없음)
#include <limits>   // [한국어] numeric_limits<int>::max() (우선순위 역전 기법)

#include "gputrafficmanager.hpp"    // [한국어] GPUTrafficManager 클래스 선언
#include "interconnect_interface.hpp" // [한국어] g_icnt_interface 전역 포인터 및 콜백 선언
#include "globals.hpp"              // [한국어] gWatchOut, gTrace, GetSimTime 등 전역 상태

/*
 * [한국어]
 * GPUTrafficManager() - GPU 트래픽 매니저 생성자
 *
 * @config: BookSim 설정 파일 파싱 결과 (subnets, nodes, classes 등)
 * @net: 서브넷별 Network 포인터 벡터
 *
 * TrafficManager 기반 클래스를 초기화한 뒤:
 *   1. _total_sims = 0 (CUDA 커널 실행 횟수 카운터; 통계 표준화에 사용)
 *   2. _input_queue를 [subnets][nodes][classes] 형태의 3차원 벡터로 초기화
 *      각 셀은 list<Flit*>이며 Push()로 삽입, _Step()으로 소비
 *
 * 주의: _input_queue는 TrafficManager::_partial_packets와 달리 클래스 개수까지 차원화되어 있다.
 *
 * 호출 체인: TrafficManager::New() (sim_type=="gpgpusim" 분기) → [이 함수]
 */
GPUTrafficManager::GPUTrafficManager( const Configuration &config, const vector<Network *> &net)
:TrafficManager(config, net)
{
  // The total simulations equal to number of kernels
  _total_sims = 0; // [한국어] 커널 실행 횟수를 0으로 초기화 (TrafficManager 기본값은 1)

  _input_queue.resize(_subnets); // [한국어] 서브넷 수만큼 outer 차원 초기화
  for ( int subnet = 0; subnet < _subnets; ++subnet) { // [한국어] 각 서브넷 반복
    _input_queue[subnet].resize(_nodes); // [한국어] 서브넷 내 노드 수만큼 초기화
    for ( int node = 0; node < _nodes; ++node ) { // [한국어] 각 노드 반복
      _input_queue[subnet][node].resize(_classes); // [한국어] 노드 내 트래픽 클래스 수만큼 초기화 (보통 1)
    }
  }
}

/*
 * [한국어]
 * ~GPUTrafficManager() - 소멸자
 *
 * 기반 클래스(TrafficManager) 소멸자가 _net, _buf_states, Stats* 등 동적 자원을 해제.
 * _input_queue의 Flit*들은 소멸 시 이미 Free() 됐어야 하므로 별도 정리 불필요.
 *
 * 호출 체인: InterconnectInterface::~InterconnectInterface() → delete _traffic_manager → [이 함수]
 */
GPUTrafficManager::~GPUTrafficManager()
{
  // [한국어] 빈 소멸자: 모든 자원은 기반 클래스(TrafficManager)에서 해제
}

/*
 * [한국어]
 * Init() - 시뮬레이션 상태를 초기화한다 (CUDA 커널 실행 시작 전 호출)
 *
 * 동작:
 *   1. _time = 0 → NoC 사이클 카운터 리셋 (커널마다 독립 측정)
 *   2. _sim_state = running → 통계 수집 시작 (warming_up 단계 없이 바로 running)
 *   3. _ClearStats() → 이전 커널의 통계 초기화
 *
 * 주의: _total_sims는 이 함수에서 증가시키지 않는다.
 *       TrafficManager::_SingleSim() 또는 Run()에서 관리하지만, GPU에서는 사용 안 함.
 *
 * 호출 체인: InterconnectInterface::Init() → [이 함수]
 */
void GPUTrafficManager::Init()
{
  _time = 0;              // [한국어] NoC 사이클 카운터를 0으로 리셋
  _sim_state = running;   // [한국어] 상태를 running으로 설정 (GPU는 warm-up 없이 바로 측정 시작)
  _ClearStats( );         // [한국어] 레이턴시/스루풋 등 이전 커널의 모든 통계를 초기화

}

/*
 * [한국어]
 * _RetireFlit() - 목적지 노드에 도달한 Flit의 통계를 기록하고 풀에 반환한다
 *
 * @f: 목적지 도달 Flit 포인터 (ejection buffer에서 꺼내진 Flit)
 * @dest: Flit이 도달한 노드의 icntID
 *
 * 동작 순서:
 *   1. _deadlock_timer = 0 → 데드락 감시 타이머 리셋 (flit이 계속 도달하면 데드락 아님)
 *   2. _total_in_flight_flits에서 이 flit 제거 (in-flight 카운터 감소)
 *   3. f->record이면 _measured_in_flight_flits에서도 제거 (측정 대상 카운터 감소)
 *   4. f->watch이면 배출 이벤트를 gWatchOut에 출력 (디버깅)
 *   5. head flit이 잘못된 목적지에 도달했으면 Error() 호출 (시뮬레이션 종료)
 *   6. _flat_stats (flit 레이턴시 = atime - itime) 기록
 *   7. f->tail이면 패킷 레이턴시 처리:
 *      - head flit을 _retired_packets에서 복원
 *      - _plat_stats (plat = atime - ctime: 생성~도착)
 *      - _nlat_stats (nlat = atime - itime: 주입~도착)
 *      - _frag_stats (패킷 내 flit 도착 분산)
 *      - _hop_stats (홉 수)
 *      - READ_REPLY/WRITE_REPLY이면 _requestsOutstanding[dest]-- (요청-응답 인플라이트 감소)
 *      - head flit Free() (head-tail이 다른 flit인 경우)
 *   8. head이고 tail이 아닌 flit은 _retired_packets에 저장 (tail 도착 시 복원)
 *   9. tail이거나 single-flit이면 Flit::Free() (풀 반환)
 *
 * 주의: READ_REQUEST/WRITE_REQUEST에 대한 reply 생성 코드는 #if 0으로 비활성화됨.
 *       GPGPU-Sim의 메모리 컨트롤러가 응답 패킷을 자체 생성하기 때문.
 *       ANY_TYPE은 GPU에서 사용하지 않으므로 오면 Error().
 *
 * 호출 체인: _Step() (크레딧 반환 단계, flits[subnet][n] 처리) → [이 함수]
 */
void GPUTrafficManager::_RetireFlit( Flit *f, int dest )
{
  _deadlock_timer = 0; // [한국어] flit이 도달했으므로 데드락 타이머를 리셋 (네트워크가 진행 중임을 확인)

  assert(_total_in_flight_flits[f->cl].count(f->id) > 0); // [한국어] 반드시 in-flight 목록에 있어야 함 (무결성 확인)
  _total_in_flight_flits[f->cl].erase(f->id); // [한국어] in-flight flit 카운터에서 제거 (전체 인플라이트 맵)

  if(f->record) { // [한국어] 이 flit이 통계 측정 대상인 경우 (running 상태에서 생성된 flit)
    assert(_measured_in_flight_flits[f->cl].count(f->id) > 0); // [한국어] 측정 목록에도 반드시 있어야 함
    _measured_in_flight_flits[f->cl].erase(f->id); // [한국어] 측정 인플라이트 맵에서도 제거
  }

  if ( f->watch ) { // [한국어] 이 flit을 추적(watch) 중인 경우 → 배출 이벤트 로그 출력
    *gWatchOut << GetSimTime() << " | "
    << "node" << dest << " | "
    << "Retiring flit " << f->id
    << " (packet " << f->pid
    << ", src = " << f->src
    << ", dest = " << f->dest
    << ", hops = " << f->hops
    << ", flat = " << f->atime - f->itime // [한국어] flit latency = 도착시각 - 주입시각
    << ")." << endl;
  }

  if ( f->head && ( f->dest != dest ) ) { // [한국어] head flit이 잘못된 노드에 도달한 경우 → 라우팅 오류
    ostringstream err; // [한국어] 에러 메시지 포맷팅
    err << "Flit " << f->id << " arrived at incorrect output " << dest;
    Error( err.str( ) ); // [한국어] BookSim Error() → 에러 출력 및 시뮬레이션 중단
  }

  if((_slowest_flit[f->cl] < 0) ||
     (_flat_stats[f->cl]->Max() < (f->atime - f->itime))) // [한국어] 이 flit이 현재까지 가장 느린 flit인지 확인
    _slowest_flit[f->cl] = f->id; // [한국어] 가장 느린 flit ID 갱신 (디버그 용도)

  _flat_stats[f->cl]->AddSample( f->atime - f->itime); // [한국어] flit latency 통계에 샘플 추가 (주입~도착 사이클)
  if(_pair_stats){ // [한국어] 소스-목적지 쌍별 통계를 기록하는 경우 (pair_stats 설정 활성화)
    _pair_flat[f->cl][f->src*_nodes+dest]->AddSample( f->atime - f->itime ); // [한국어] 소스-목적지 조합 인덱스로 통계 저장
  }

  if ( f->tail ) { // [한국어] tail flit이면 패킷이 완전히 도달 → 패킷 레이턴시 통계 처리
    Flit * head; // [한국어] tail에 대응하는 head flit 포인터 (ctime 참조용)
    if(f->head) { // [한국어] head == tail인 단일 flit 패킷
      head = f; // [한국어] 자기 자신이 head
    } else { // [한국어] 멀티 flit 패킷: head가 먼저 도달해 _retired_packets에 저장됨
      map<unsigned long long, Flit *>::iterator iter = _retired_packets[f->cl].find(f->pid);
      assert(iter != _retired_packets[f->cl].end()); // [한국어] head flit이 반드시 먼저 저장되어 있어야 함
      head = iter->second; // [한국어] head flit 포인터 복원
      _retired_packets[f->cl].erase(iter); // [한국어] head flit을 임시 저장소에서 제거
      assert(head->head); // [한국어] 복원된 flit이 실제 head인지 확인
      assert(f->pid == head->pid); // [한국어] 같은 패킷의 flit인지 확인
    }
    if ( f->watch ) { // [한국어] 패킷 단위 배출 이벤트 출력 (watch 활성 시)
      *gWatchOut << GetSimTime() << " | "
      << "node" << dest << " | "
      << "Retiring packet " << f->pid
      << " (plat = " << f->atime - head->ctime     // [한국어] 패킷 생성시각(ctime) ~ 도착시각(atime)
      << ", nlat = " << f->atime - head->itime     // [한국어] 첫 flit 주입시각(itime) ~ tail 도착시각
      << ", frag = " << (f->atime - head->atime) - (f->id - head->id) // NB: In the spirit of solving problems using ugly hacks, we compute the packet length by taking advantage of the fact that the IDs of flits within a packet are contiguous.
      // [한국어] frag = 패킷 내 flit 간 도착 시간 분산. (tail.atime - head.atime) - (tail.id - head.id).
      // flit ID가 패킷 내에서 연속적임을 이용한 해킹 방식
      << ", src = " << head->src
      << ", dest = " << head->dest
      << ")." << endl;
    }

// GPGPUSim: Memory will handle reply, do not need this
// [한국어] GPGPU-Sim에서는 메모리 컨트롤러가 응답(reply)을 직접 생성하므로
// BookSim의 _repliesPending/_requestsOutstanding 기반 응답 생성 코드 비활성화
#if 0
    //code the source of request, look carefully, its tricky ;)
    if (f->type == Flit::READ_REQUEST || f->type == Flit::WRITE_REQUEST) {
      PacketReplyInfo* rinfo = PacketReplyInfo::New();
      rinfo->source = f->src;
      rinfo->time = f->atime;
      rinfo->record = f->record;
      rinfo->type = f->type;
      _repliesPending[dest].push_back(rinfo);
    } else {
      if(f->type == Flit::READ_REPLY || f->type == Flit::WRITE_REPLY  ){
        _requestsOutstanding[dest]--;
      } else if(f->type == Flit::ANY_TYPE) {
        _requestsOutstanding[f->src]--;
      }

    }
#endif

    if(f->type == Flit::READ_REPLY || f->type == Flit::WRITE_REPLY  ){ // [한국어] 응답 패킷이 도달한 경우
      _requestsOutstanding[dest]--; // [한국어] 응답을 기다리는 요청 카운터 감소 (데드락 판정에 사용)
    } else if(f->type == Flit::ANY_TYPE) { // [한국어] GPU 시뮬레이션에서는 ANY_TYPE 패킷 불가
      ostringstream err;
      err << "Flit " << f->id << " cannot be ANY_TYPE" ;
      Error( err.str( ) ); // [한국어] ANY_TYPE은 GPU에서 사용 금지 → 시뮬레이션 중단
    }

    // Only record statistics once per packet (at tail)
    // and based on the simulation state
    // [한국어] warming_up 상태이거나 f->record가 true인 패킷만 통계 기록
    if ( ( _sim_state == warming_up ) || f->record ) {

      _hop_stats[f->cl]->AddSample( f->hops ); // [한국어] 패킷이 거친 홉 수 통계 기록

      if((_slowest_packet[f->cl] < 0) ||
         (_plat_stats[f->cl]->Max() < (f->atime - head->itime))) // [한국어] 현재까지 가장 느린 패킷 갱신
        _slowest_packet[f->cl] = f->pid; // [한국어] 가장 느린 패킷 ID 저장 (디버깅)
      _plat_stats[f->cl]->AddSample( f->atime - head->ctime); // [한국어] plat (패킷 레이턴시: ctime~atime) 통계
      _nlat_stats[f->cl]->AddSample( f->atime - head->itime); // [한국어] nlat (네트워크 레이턴시: itime~atime) 통계
      _frag_stats[f->cl]->AddSample( (f->atime - head->atime) - (f->id - head->id) ); // [한국어] frag 통계 (패킷 내 flit 분산)

      if(_pair_stats){ // [한국어] pair 통계 활성화된 경우: 소스-목적지 쌍별 패킷 레이턴시 기록
        _pair_plat[f->cl][f->src*_nodes+dest]->AddSample( f->atime - head->ctime ); // [한국어] 쌍별 plat
        _pair_nlat[f->cl][f->src*_nodes+dest]->AddSample( f->atime - head->itime ); // [한국어] 쌍별 nlat
      }
    }

    if(f != head) { // [한국어] 멀티 flit 패킷이면 head flit도 별도 Free() 필요
      head->Free(); // [한국어] head flit을 Flit 풀에 반환 (tail과는 별도 객체)
    }

  }

  if(f->head && !f->tail) { // [한국어] head이지만 tail이 아닌 flit → tail이 나중에 올 때 ctime 참조 필요
    _retired_packets[f->cl].insert(make_pair(f->pid, f)); // [한국어] tail 도착 시까지 head를 임시 보관
  } else { // [한국어] tail (단일 flit 포함) → 이 flit 자체 Free()
    f->Free(); // [한국어] Flit을 _free 스택에 반환 (Flit::Free())
  }
}

/*
 * [한국어]
 * _IssuePacket() - 합성 트래픽 패킷 생성 (GPU에서는 사용 안 함)
 *
 * @source: 송신 노드 (사용 안 함)
 * @cl: 트래픽 클래스 (사용 안 함)
 * @return: 항상 0 (패킷 미생성)
 *
 * TrafficManager::_Inject()가 호출하지만, GPU 시뮬레이션에서는
 * _Inject() 자체가 #if 0으로 비활성화되어 있으므로 이 함수는 실제로 불리지 않는다.
 *
 * 호출 체인: TrafficManager::_Inject() (#if 0 비활성화) → [이 함수] (실질적 무효)
 */
int  GPUTrafficManager::_IssuePacket( int source, int cl )
{
  return 0; // [한국어] 합성 트래픽 생성 없음; GPU는 직접 _GeneratePacket()을 호출
}

/*
 * [한국어]
 * _GeneratePacket() - mem_fetch 패킷으로부터 Flit들을 생성하여 _input_queue에 삽입한다
 *
 * @source: 송신 노드의 icntID
 * @stype: 예약 파라미터 (현재 GPU에서는 -1로 고정, 0이면 assert 실패)
 * @cl: 트래픽 클래스 (GPU에서는 항상 0)
 * @time: 패킷 생성 시각 (Flit::ctime에 저장)
 * @subnet: 사용할 서브넷 인덱스 (0=요청 네트워크, 1=응답 네트워크)
 * @packet_size: 패킷을 구성하는 flit 수 (= ceil(mem_fetch_size / flit_size))
 * @packet_type: Flit::FlitType (READ_REQUEST, WRITE_REQUEST, READ_REPLY, WRITE_REPLY)
 * @data: 원본 mem_fetch 포인터 (Flit::data에 저장, Pop() 시 복원용)
 * @dest: 목적지 노드의 icntID
 *
 * 동작 순서:
 *   1. stype != 0 확인 (assert)
 *   2. 패킷 고유 ID(pid) 할당: _cur_pid++ (순차적, 모든 패킷에 고유)
 *   3. 목적지 유효성 확인: 0 <= dest < _nodes
 *   4. record 결정: _sim_state==running이면 _measure_stats[cl]
 *   5. for i in range(packet_size):
 *      a. Flit::New() → 풀에서 Flit 할당
 *      b. id = _cur_id++ (flit 고유 ID, 패킷 내 연속)
 *      c. pid, subnetwork, src, ctime, record, cl, data 설정
 *      d. i==0: head=true, dest=packet_destination; else: head=false, dest=-1
 *      e. 우선순위(pri) 설정 (_pri_type에 따라):
 *         - class_based: _class_priority[cl]
 *         - age_based: MAX_INT - time (오래될수록 높은 우선순위)
 *         - sequence_based: MAX_INT - _packet_seq_no[source]
 *         - default: 0
 *      f. i==size-1: tail=true; else: tail=false
 *      g. vc = -1 (미할당; _Step()에서 VC 탐색)
 *      h. _input_queue[subnet][source][cl].push_back(f)
 *
 * 주의: 기존 BookSim의 reply 생성 코드(#if 0)는 비활성화됨.
 *       GPGPU-Sim이 직접 타입과 크기를 명시하여 호출.
 *
 * 호출 체인: InterconnectInterface::Push() → [이 함수]
 */
//TODO: Remove stype?
void GPUTrafficManager::_GeneratePacket(int source, int stype, int cl, int time, int subnet, int packet_size, const Flit::FlitType& packet_type, void* const data, int dest)
{
  assert(stype!=0); // [한국어] stype==0은 기존 BookSim reply 생성 경로 → GPU에서 사용 금지

  //  Flit::FlitType packet_type = Flit::ANY_TYPE;
  int size = packet_size; //input size // [한국어] 패킷의 flit 수 (입력 그대로 사용)
  unsigned long long pid = _cur_pid++; // [한국어] 새 패킷 ID 할당 (전역 단조 증가)
  assert(_cur_pid > 0); // [한국어] pid 오버플로 감지 (64비트지만 안전 확인)
  int packet_destination = dest; // [한국어] 목적지 icntID (head flit에만 설정됨)
  bool record = false; // [한국어] 기본값: 통계 측정 안 함
  bool watch = gWatchOut && (_packets_to_watch.count(pid) > 0); // [한국어] 이 패킷을 watch 목록에서 추적하는지 확인

  // In GPGPUSim, the core specified the packet_type and size
  // [한국어] GPGPU-Sim에서는 호출자가 packet_type과 size를 직접 명시함 (합성 트래픽 로직 불필요)

// [한국어] 기존 BookSim reply 생성 로직 비활성화:
// READ/WRITE REQUEST → _repliesPending 등록 → 다음 사이클에 REPLY 생성
// GPGPU-Sim에서는 메모리 컨트롤러가 응답 타이밍을 직접 결정하므로 이 경로 미사용
#if 0
  if(_use_read_write[cl]){
    if(stype > 0) {
      if (stype == 1) {
        packet_type = Flit::READ_REQUEST;
        size = _read_request_size[cl];
      } else if (stype == 2) {
        packet_type = Flit::WRITE_REQUEST;
        size = _write_request_size[cl];
      } else {
        ostringstream err;
        err << "Invalid packet type: " << packet_type;
        Error( err.str( ) );
      }
    } else {
      PacketReplyInfo* rinfo = _repliesPending[source].front();
      if (rinfo->type == Flit::READ_REQUEST) {//read reply
        size = _read_reply_size[cl];
        packet_type = Flit::READ_REPLY;
      } else if(rinfo->type == Flit::WRITE_REQUEST) {  //write reply
        size = _write_reply_size[cl];
        packet_type = Flit::WRITE_REPLY;
      } else {
        ostringstream err;
        err << "Invalid packet type: " << rinfo->type;
        Error( err.str( ) );
      }
      packet_destination = rinfo->source;
      time = rinfo->time;
      record = rinfo->record;
      _repliesPending[source].pop_front();
      rinfo->Free();
    }
  }
#endif

  if ((packet_destination <0) || (packet_destination >= _nodes)) { // [한국어] 목적지 유효성 검사
    ostringstream err; // [한국어] 잘못된 목적지 → 에러 메시지 구성
    err << "Incorrect packet destination " << packet_destination
    << " for stype " << packet_type;
    Error( err.str( ) ); // [한국어] 시뮬레이션 중단
  }

  if ( ( _sim_state == running ) ||
      ( ( _sim_state == draining ) && ( time < _drain_time ) ) ) { // [한국어] running 또는 draining 초기에 생성된 패킷은 통계 측정 대상
    record = _measure_stats[cl]; // [한국어] _measure_stats[cl]가 활성화되면 record=true
  }

  int subnetwork = subnet; // [한국어] 사용할 서브넷 인덱스 (0=요청, 1=응답)
  //                ((packet_type == Flit::ANY_TYPE) ?
  //                    RandomInt(_subnets-1) :
  //                    _subnet[packet_type]);
  // [한국어] 주석처리된 코드: _subnet[] 맵으로 packet_type에 따라 자동 서브넷 선택하는 BookSim 방식.
  // GPU에서는 Push()가 명시적으로 subnet을 지정하므로 직접 사용.

  if ( watch ) { // [한국어] 패킷을 watch 목록에서 추적 중이면 enqueue 이벤트 출력
    *gWatchOut << GetSimTime() << " | "
    << "node" << source << " | "
    << "Enqueuing packet " << pid
    << " at time " << time
    << "." << endl;
  }

  for ( int i = 0; i < size; ++i ) { // [한국어] 패킷을 구성하는 각 flit을 생성
    Flit * f  = Flit::New(); // [한국어] flit 풀(_free 스택)에서 Flit 할당 (힙 재할당 없이 재사용)
    f->id     = _cur_id++; // [한국어] 전역 단조 증가 flit ID 할당 (패킷 내에서 연속적)
    assert(_cur_id); // [한국어] 64비트 ID 오버플로 방지 확인
    f->pid    = pid; // [한국어] 이 flit이 속한 패킷 ID (head/tail 매핑에 사용)
    f->watch  = watch | (gWatchOut && (_flits_to_watch.count(f->id) > 0)); // [한국어] 패킷 watch OR 개별 flit watch 여부
    f->subnetwork = subnetwork; // [한국어] 이 flit이 속한 서브넷 (라우팅 시 사용)
    f->src    = source; // [한국어] 송신 노드 icntID
    f->ctime  = time; // [한국어] 패킷 생성(creation) 시각 (plat 계산 기준점)
    f->record = record; // [한국어] 통계 측정 대상 여부
    f->cl     = cl; // [한국어] 트래픽 클래스 (GPU에서는 항상 0)
    f->data = data; // [한국어] 원본 mem_fetch 포인터 (ejection 시 복원용, 모든 flit에 동일하게 설정)

    _total_in_flight_flits[f->cl].insert(make_pair(f->id, f)); // [한국어] 전체 인플라이트 맵에 등록 (데드락 감지용)
    if(record) { // [한국어] 측정 대상이면 측정 인플라이트 맵에도 등록
      _measured_in_flight_flits[f->cl].insert(make_pair(f->id, f));
    }

    if(gTrace){ // [한국어] gTrace 활성 시 새 flit 생성 이벤트 출력 (상세 디버깅)
      cout<<"New Flit "<<f->src<<endl;
    }
    f->type = packet_type; // [한국어] READ_REQUEST / WRITE_REQUEST / READ_REPLY / WRITE_REPLY 설정

    if ( i == 0 ) { // Head flit // [한국어] 첫 번째 flit = head: 목적지 정보 포함
      f->head = true; // [한국어] head 플래그 설정
      //packets are only generated to nodes smaller or equal to limit
      f->dest = packet_destination; // [한국어] 목적지 icntID 설정 (head flit만 유효한 dest를 가짐)
    } else { // [한국어] 두 번째 이후 flit: head 아님, dest는 -1 (head flit에서 전달받음)
      f->head = false; // [한국어] head 아님
      f->dest = -1; // [한국어] body/tail flit은 dest 필드 사용 안 함 (-1로 표시)
    }
    switch( _pri_type ) { // [한국어] 설정 파일의 priority_type에 따라 flit 우선순위 결정
      case class_based: // [한국어] 클래스별 고정 우선순위 (설정 파일의 class_priority)
        f->pri = _class_priority[cl];
        assert(f->pri >= 0); // [한국어] 우선순위는 음수 불가
        break;
      case age_based: // [한국어] 나이 기반 우선순위: 오래된 패킷일수록 높은 우선순위 (MAX_INT - time)
        f->pri = numeric_limits<int>::max() - time;
        assert(f->pri >= 0);
        break;
      case sequence_based: // [한국어] 순서 기반: 먼저 생성된 패킷일수록 높은 우선순위
        f->pri = numeric_limits<int>::max() - _packet_seq_no[source];
        assert(f->pri >= 0);
        break;
      default: // [한국어] 우선순위 없음 (none 등): 모두 동일 우선순위 0
        f->pri = 0;
    }
    if ( i == ( size - 1 ) ) { // Tail flit // [한국어] 마지막 flit = tail: 패킷 완성 신호
      f->tail = true; // [한국어] tail 플래그 설정
    } else { // [한국어] 중간 flit
      f->tail = false;
    }

    f->vc  = -1; // [한국어] VC 미할당 상태 (-1); _Step()이 head flit 처리 시 VC를 탐색하여 할당

    if ( f->watch ) { // [한국어] 이 flit을 watch 중이면 enqueue 이벤트 출력
      *gWatchOut << GetSimTime() << " | "
      << "node" << source << " | "
      << "Enqueuing flit " << f->id
      << " (packet " << f->pid
      << ") at time " << time
      << "." << endl;
    }

    _input_queue[subnet][source][cl].push_back( f ); // [한국어] 이 flit을 해당 서브넷/소스노드/클래스의 입력 큐에 삽입
  }
}

/*
 * [한국어]
 * _Step() - NoC 시뮬레이션 1 사이클을 진행한다
 *
 * 매 사이클마다 InterconnectInterface::Advance()가 호출하며, 실행 순서는 다음과 같다:
 *
 * [단계 1] 데드락 감시
 *   - 인플라이트 flit이 있으면 _deadlock_timer 증가
 *   - _deadlock_warn_timeout 초과 시 경고 출력 및 타이머 리셋
 *
 * [단계 2] 배출(Ejection) 처리 — 각 서브넷/노드
 *   - _net[subnet]->ReadFlit(n): 이미 도착한 Flit을 라우터 출구에서 읽음
 *   - g_icnt_interface->WriteOutBuffer(subnet, n, f): ejection 버퍼에 저장
 *   - g_icnt_interface->Transfer2BoundaryBuffer(subnet, n): ejection → boundary 버퍼 이동
 *   - g_icnt_interface->GetEjectedFlit(subnet, n): boundary에서 꺼낸 flit → flits[subnet][n] 맵에 저장
 *   - _accepted_flits / _accepted_packets 통계 증가
 *
 * [단계 3] 크레딧 수신 처리 — 각 서브넷/노드
 *   - _net[subnet]->ReadCredit(n): 다운스트림에서 올라온 크레딧 읽기
 *   - _buf_states[n][subnet]->ProcessCredit(c): 업스트림 버퍼 상태 업데이트
 *   - Credit::Free()로 크레딧 반환
 *
 * [단계 4] 네트워크 입력 캡처
 *   - _net[subnet]->ReadInputs(): 모든 FlitChannel 입력 캡처 (딜레이 큐 전진)
 *
 * [단계 5] 주입(Injection) — 각 서브넷/노드
 *   - _input_queue[subnet][n][c]에서 가장 우선순위 높은 Flit 선택
 *   - head flit이고 vc==-1이면: _rf()로 주입 VC 범위(inject mode) 계산, 라운드-로빈으로 가용 VC 탐색
 *   - lookahead routing: _net[subnet]->GetInject(n)->GetSink()로 첫 라우터 참조 → _rf()로 la_route_set 설정
 *   - dest_buf->TakeBuffer(vc), dest_buf->SendingFlit(f)으로 버퍼 상태 업데이트
 *   - _input_queue에서 pop_front(), _net[subnet]->WriteFlit(f, n) 으로 네트워크에 주입
 *   - tail이 아닌 경우 다음 flit에 vc 값 전달 ("Pass VC back")
 *   - _sent_flits / _sent_packets 통계 증가
 *
 * [단계 6] 크레딧 발행(Ejection Credit) — 각 서브넷/노드
 *   - flits[subnet][n]에서 꺼낸 ejected flit에 f->atime = _time 설정
 *   - Credit::New(), c->vc.insert(f->vc) → _net[subnet]->WriteCredit(c, n): 업스트림 크레딧 반환
 *   - _RetireFlit(f, n): 통계 기록 및 Free()
 *
 * [단계 7] 네트워크 평가 및 출력
 *   - _net[subnet]->Evaluate(): 라우터 내부 스위치 스케줄링 실행
 *   - _net[subnet]->WriteOutputs(): 출력 FlitChannel에 Flit 전달
 *
 * [단계 8] 시간 증가
 *   - ++_time
 *
 * 주의: _Inject() 호출은 #if 0으로 비활성화 (GPU는 _input_queue에서 직접 주입)
 *
 * 호출 체인: InterconnectInterface::Advance() → [이 함수]
 */
void GPUTrafficManager::_Step()
{
  bool flits_in_flight = false; // [한국어] 현재 네트워크에 인플라이트 flit이 있는지 여부
  for(int c = 0; c < _classes; ++c) { // [한국어] 모든 클래스 확인
    flits_in_flight |= !_total_in_flight_flits[c].empty(); // [한국어] 어느 클래스에라도 인플라이트 flit이 있으면 true
  }
  if(flits_in_flight && (_deadlock_timer++ >= _deadlock_warn_timeout)){ // [한국어] 인플라이트 flit이 있고 타이머가 임계값 초과
    _deadlock_timer = 0; // [한국어] 타이머 리셋 (반복 경고 방지)
    cout << "WARNING: Possible network deadlock.\n"; // [한국어] 데드락 가능성 경고 출력 (시뮬레이션은 계속)
  }

  vector<map<int, Flit *> > flits(_subnets); // [한국어] 이번 사이클에 ejection된 flit을 임시 저장 [subnet][node → flit]

  for ( int subnet = 0; subnet < _subnets; ++subnet ) { // [한국어] 각 서브넷 반복 (0=요청, 1=응답)
    for ( int n = 0; n < _nodes; ++n ) { // [한국어] 각 노드 반복 (SM + 메모리 컨트롤러)
      Flit * const f = _net[subnet]->ReadFlit( n ); // [한국어] 노드 n에 도달한 Flit 읽기 (해당 노드의 ejection port)
      if ( f ) { // [한국어] 이번 사이클에 도달한 Flit이 있는 경우
        if(f->watch) { // [한국어] 이 Flit을 watch 중이면 ejection 이벤트 출력
          *gWatchOut << GetSimTime() << " | "
          << "node" << n << " | "
          << "Ejecting flit " << f->id
          << " (packet " << f->pid << ")"
          << " from VC " << f->vc
          << "." << endl;
        }
        g_icnt_interface->WriteOutBuffer(subnet, n, f); // [한국어] ejection buffer에 Flit 저장 (_ejection_buffer[subnet][n]에 추가)
      }

      g_icnt_interface->Transfer2BoundaryBuffer(subnet, n); // [한국어] ejection buffer → boundary buffer로 완성된 패킷 이동
      Flit* const ejected_flit = g_icnt_interface->GetEjectedFlit(subnet, n); // [한국어] boundary buffer에서 꺼낸 tail flit 가져오기 (패킷 완성 확인)
      if (ejected_flit) { // [한국어] boundary에서 꺼낸 flit이 있는 경우 (완성된 패킷의 tail)
        if(ejected_flit->head) // [한국어] head flit이 ejection되면 반드시 dest가 현재 노드와 일치해야 함
          assert(ejected_flit->dest == n); // [한국어] 라우팅 정확도 확인
        if(ejected_flit->watch) { // [한국어] 이 flit을 watch 중이면 ejection buffer 출력 이벤트 로그
          *gWatchOut << GetSimTime() << " | "
          << "node" << n << " | "
          << "Ejected flit " << ejected_flit->id
          << " (packet " << ejected_flit->pid
          << " VC " << ejected_flit->vc << ")"
          << "from ejection buffer." << endl;
        }
        flits[subnet].insert(make_pair(n, ejected_flit)); // [한국어] 이번 사이클 ejected flit 맵에 저장 (크레딧 발행 단계에서 처리)
        if((_sim_state == warming_up) || (_sim_state == running)) { // [한국어] 통계 수집 상태이면
          ++_accepted_flits[ejected_flit->cl][n]; // [한국어] 노드 n에 도달한 flit 수 증가
          if(ejected_flit->tail) { // [한국어] tail flit이면 패킷 단위 카운터도 증가
            ++_accepted_packets[ejected_flit->cl][n]; // [한국어] 노드 n에 완전히 도달한 패킷 수 증가
          }
        }
      }

      // Processing the credit From the network
      // [한국어] 다운스트림 라우터가 보내온 크레딧(Credit) 처리
      Credit * const c = _net[subnet]->ReadCredit( n ); // [한국어] 노드 n에서 올라온 크레딧 읽기 (injection port의 업스트림 크레딧)
      if ( c ) { // [한국어] 크레딧이 있는 경우
#ifdef TRACK_FLOWS // [한국어] 흐름 추적 모드: 각 VC별 클래스 정보 업데이트
        for(set<int>::const_iterator iter = c->vc.begin(); iter != c->vc.end(); ++iter) {
          int const vc = *iter; // [한국어] 크레딧 반환된 VC 번호
          assert(!_outstanding_classes[n][subnet][vc].empty()); // [한국어] 해당 VC에 대기 중인 클래스가 있어야 함
          int cl = _outstanding_classes[n][subnet][vc].front(); // [한국어] 해당 VC의 클래스 확인
          _outstanding_classes[n][subnet][vc].pop(); // [한국어] 처리 완료로 큐에서 제거
          assert(_outstanding_credits[cl][subnet][n] > 0); // [한국어] 미처리 크레딧이 남아있어야 함
          --_outstanding_credits[cl][subnet][n]; // [한국어] 해당 클래스/서브넷/노드의 미처리 크레딧 감소
        }
#endif
        _buf_states[n][subnet]->ProcessCredit(c); // [한국어] 업스트림 버퍼 상태(BufferState) 갱신: 해당 VC 사용 가능 슬롯 +1
        c->Free(); // [한국어] Credit 객체를 풀에 반환
      }
    }
    _net[subnet]->ReadInputs( ); // [한국어] 서브넷의 모든 FlitChannel에서 입력 읽기 (딜레이 큐 전진, 이번 사이클 flit 전달)
  }

// GPGPUSim will generate/inject packets from interconnection interface
// [한국어] 합성 트래픽 생성 비활성화:
// GPGPU-Sim에서는 Push() → _GeneratePacket()으로 직접 _input_queue에 Flit을 삽입하므로
// TrafficManager의 _Inject() (합성 트래픽 생성) 경로는 사용하지 않음
#if 0
  if ( !_empty_network ) {
    _Inject();
  }
#endif

  for(int subnet = 0; subnet < _subnets; ++subnet) { // [한국어] 각 서브넷별 주입(injection) 처리

    for(int n = 0; n < _nodes; ++n) { // [한국어] 각 노드별 처리 (SM + 메모리 컨트롤러)

      Flit * f = NULL; // [한국어] 이번 사이클에 주입할 최고 우선순위 Flit (없으면 NULL)

      BufferState * const dest_buf = _buf_states[n][subnet]; // [한국어] 노드 n이 injection 포트를 통해 보는 다운스트림 VC 버퍼 상태

      int const last_class = _last_class[n][subnet]; // [한국어] 직전 사이클에 이 노드가 주입한 클래스 (공정성 구현)

      int class_limit = _classes; // [한국어] 이번 사이클에 탐색할 클래스 수

      if(_hold_switch_for_packet) { // [한국어] 패킷 중간에 스위치를 유지하는 설정 (HOL 블로킹 방지)
        list<Flit *> const & pp = _input_queue[subnet][n][last_class]; // [한국어] 이전 클래스의 큐 참조
        if(!pp.empty() && !pp.front()->head &&
           !dest_buf->IsFullFor(pp.front()->vc)) { // [한국어] 이전 클래스의 body/tail flit이 있고 VC에 자리 있으면 연속 주입
          f = pp.front(); // [한국어] 같은 클래스의 다음 flit 선택 (패킷 연속 전송)
          assert(f->vc == _last_vc[n][subnet][last_class]); // [한국어] VC 할당이 연속되었는지 확인

          // if we're holding the connection, we don't need to check that class
          // again in the for loop
          // [한국어] 이미 last_class의 flit을 선택했으므로 for 루프에서 last_class 다시 탐색 불필요
          --class_limit; // [한국어] 탐색 클래스 수 감소
        }
      }

      for(int i = 1; i <= class_limit; ++i) { // [한국어] 라운드-로빈으로 각 클래스 탐색 (last_class부터 순환)

        int const c = (last_class + i) % _classes; // [한국어] 라운드-로빈: last_class 다음 클래스부터 확인

        list<Flit *> const & pp = _input_queue[subnet][n][c]; // [한국어] 이 클래스의 입력 큐 참조

        if(pp.empty()) { // [한국어] 이 클래스의 큐가 비어있으면 다음 클래스로
          continue;
        }

        Flit * const cf = pp.front(); // [한국어] 큐의 맨 앞 Flit (Head-of-Line)
        assert(cf); // [한국어] 큐가 비어있지 않으면 front()는 반드시 유효
        assert(cf->cl == c); // [한국어] flit의 클래스가 큐 인덱스와 일치해야 함

        assert(cf->subnetwork == subnet); // [한국어] flit의 서브넷이 현재 처리 중인 서브넷과 일치해야 함

        if(f && (f->pri >= cf->pri)) { // [한국어] 이미 선택된 flit보다 우선순위가 낮거나 같으면 건너뜀
          continue;
        }

        if(cf->head && cf->vc == -1) { // Find first available VC // [한국어] head flit이고 아직 VC가 할당되지 않은 경우 → VC 탐색

          OutputSet route_set; // [한국어] 주입 VC 범위를 저장할 OutputSet
          _rf(NULL, cf, -1, &route_set, true); // [한국어] 라우팅 함수를 inject=true 모드로 호출: 이 flit 타입에 허용된 VC 범위 반환
          // NULL = 주입 포트 (라우터 아님), in_channel=-1 (주입), inject=true
          set<OutputSet::sSetElement> const & os = route_set.GetSet(); // [한국어] 가능한 출구 포트/VC 범위 집합
          assert(os.size() == 1); // [한국어] 주입 모드에서는 반드시 하나의 원소 (out_port=-1, vc_start/end)
          OutputSet::sSetElement const & se = *os.begin(); // [한국어] 유일한 원소
          assert(se.output_port == -1); // [한국어] 주입 모드의 출구 포트는 -1 (injection port)
          int vc_start = se.vc_start; // [한국어] 이 flit 타입에 허용된 VC 범위 시작
          int vc_end = se.vc_end;     // [한국어] 허용 VC 범위 끝
          int vc_count = vc_end - vc_start + 1; // [한국어] 사용 가능한 VC 총 수
          if(_noq) { // [한국어] NOQ(Next-hop Output Queuing) 활성화 시: 첫 홉 기반 VC 분할
            assert(_lookahead_routing); // [한국어] NOQ는 반드시 lookahead routing과 함께 사용
            const FlitChannel * inject = _net[subnet]->GetInject(n); // [한국어] 이 노드의 injection 채널
            const Router * router = inject->GetSink(); // [한국어] injection 채널과 연결된 첫 라우터
            assert(router); // [한국어] injection 채널이 연결된 라우터가 있어야 함
            int in_channel = inject->GetSinkPort(); // [한국어] 라우터 입구 채널 번호

            // NOTE: Because the lookahead is not for injection, but for the
            // first hop, we have to temporarily set cf's VC to be non-negative
            // in order to avoid seting of an assertion in the routing function.
            // [한국어] 라우팅 함수 내부 assert(vc >= 0) 우회를 위해 임시로 vc_start 설정
            cf->vc = vc_start; // [한국어] 임시 VC 설정 (assert 우회용)
            _rf(router, cf, in_channel, &cf->la_route_set, false); // [한국어] 첫 라우터에서의 lookahead 라우팅 계산
            cf->vc = -1; // [한국어] 임시 설정 원래대로 복원

            if(cf->watch) { // [한국어] watch 중이면 lookahead 생성 이벤트 출력
              *gWatchOut << GetSimTime() << " | "
              << "node" << n << " | "
              << "Generating lookahead routing info for flit " << cf->id
              << " (NOQ)." << endl;
            }
            set<OutputSet::sSetElement> const sl = cf->la_route_set.GetSet(); // [한국어] lookahead 라우팅 결과 집합
            assert(sl.size() == 1); // [한국어] lookahead 결과도 단일 출구 포트여야 함
            int next_output = sl.begin()->output_port; // [한국어] 첫 홉에서의 출구 포트 번호
            vc_count /= router->NumOutputs(); // [한국어] VC를 출구 포트 수만큼 분할 (NOQ 방식)
            vc_start += next_output * vc_count; // [한국어] 해당 출구 포트에 해당하는 VC 범위 시작점
            vc_end = vc_start + vc_count - 1; // [한국어] 해당 출구 포트에 해당하는 VC 범위 끝점
            assert(vc_start >= se.vc_start && vc_start <= se.vc_end); // [한국어] 원래 허용 범위 내에 있는지 확인
            assert(vc_end >= se.vc_start && vc_end <= se.vc_end); // [한국어] 범위 끝도 허용 범위 내인지 확인
            assert(vc_start <= vc_end); // [한국어] 범위가 유효한지 확인
          }
          if(cf->watch) { // [한국어] VC 탐색 과정을 watch 출력에 기록
            *gWatchOut << GetSimTime() << " | " << FullName() << " | "
            << "Finding output VC for flit " << cf->id
            << ":" << endl;
          }
          for(int i = 1; i <= vc_count; ++i) { // [한국어] 가용 VC 탐색 (라운드-로빈, vc_count번 시도)
            int const lvc = _last_vc[n][subnet][c]; // [한국어] 이 노드/서브넷/클래스에서 마지막으로 사용한 VC
            int const vc =
            (lvc < vc_start || lvc > vc_end) ? // [한국어] 마지막 VC가 허용 범위 밖이면 vc_start부터 시작
            vc_start :
            (vc_start + (lvc - vc_start + i) % vc_count); // [한국어] 라운드-로빈: (lvc에서 i번 다음) % vc_count
            assert((vc >= vc_start) && (vc <= vc_end)); // [한국어] 계산된 VC가 허용 범위 내인지 확인
            if(!dest_buf->IsAvailableFor(vc)) { // [한국어] 이 VC가 이미 다른 패킷에게 예약됨(head flit 처리 중)
              if(cf->watch) {
                *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                << "  Output VC " << vc << " is busy." << endl;
              }
            } else { // [한국어] 이 VC를 사용할 수 있음
              if(dest_buf->IsFullFor(vc)) { // [한국어] VC의 버퍼가 가득 찬 경우 (크레딧 부족)
                if(cf->watch) {
                  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                  << "  Output VC " << vc << " is full." << endl;
                }
              } else { // [한국어] VC 가용하고 버퍼도 여유 있음 → 이 VC 선택
                if(cf->watch) {
                  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                  << "  Selected output VC " << vc << "." << endl;
                }
                cf->vc = vc; // [한국어] head flit에 VC 할당
                break; // [한국어] VC 찾았으므로 탐색 종료
              }
            }
          }
        }

        if(cf->vc == -1) { // [한국어] 가용 VC를 찾지 못한 경우 → 이번 사이클 주입 불가
          if(cf->watch) {
            *gWatchOut << GetSimTime() << " | " << FullName() << " | "
            << "No output VC found for flit " << cf->id
            << "." << endl;
          }
        } else { // [한국어] VC가 할당된 경우 → 버퍼 가득 찬지 최종 확인
          if(dest_buf->IsFullFor(cf->vc)) { // [한국어] 할당된 VC가 이미 가득 찬 경우 (크레딧 부족)
            if(cf->watch) {
              *gWatchOut << GetSimTime() << " | " << FullName() << " | "
              << "Selected output VC " << cf->vc
              << " is full for flit " << cf->id
              << "." << endl;
            }
          } else { // [한국어] VC 가용하고 버퍼 여유 있음 → 이 flit을 주입 후보로 선택
            f = cf; // [한국어] 최종 주입 후보 flit 결정
          }
        }
      }

      if(f) { // [한국어] 이번 사이클에 주입할 flit이 결정된 경우

        assert(f->subnetwork == subnet); // [한국어] 무결성: flit의 서브넷과 현재 처리 서브넷 일치 확인

        int const c = f->cl; // [한국어] 선택된 flit의 클래스

        if(f->head) { // [한국어] head flit 처리: lookahead 라우팅 및 버퍼 예약

          if (_lookahead_routing) { // [한국어] lookahead 라우팅 설정 활성 시
            if(!_noq) { // [한국어] NOQ 비활성 시에만 이 시점에 lookahead 계산 (NOQ이면 VC 탐색 시 이미 계산됨)
              const FlitChannel * inject = _net[subnet]->GetInject(n); // [한국어] 이 노드의 injection 채널
              const Router * router = inject->GetSink(); // [한국어] injection 채널 끝의 첫 라우터
              assert(router); // [한국어] 라우터가 연결되어 있어야 함
              int in_channel = inject->GetSinkPort(); // [한국어] 입구 채널 번호
              _rf(router, f, in_channel, &f->la_route_set, false); // [한국어] 첫 라우터 기준 lookahead 라우팅 계산 (inject=false)
              if(f->watch) {
                *gWatchOut << GetSimTime() << " | "
                << "node" << n << " | "
                << "Generating lookahead routing info for flit " << f->id
                << "." << endl;
              }
            } else if(f->watch) { // [한국어] NOQ 활성 시에는 VC 탐색 단계에서 이미 계산됨
              *gWatchOut << GetSimTime() << " | "
              << "node" << n << " | "
              << "Already generated lookahead routing info for flit " << f->id
              << " (NOQ)." << endl;
            }
          } else { // [한국어] lookahead 라우팅 비활성: la_route_set 초기화
            f->la_route_set.Clear(); // [한국어] lookahead 라우팅 정보 초기화
          }

          dest_buf->TakeBuffer(f->vc); // [한국어] 첫 라우터의 해당 VC를 이 패킷이 예약 (다른 패킷 진입 차단)
          _last_vc[n][subnet][c] = f->vc; // [한국어] 마지막 사용 VC 갱신 (다음 head flit의 라운드-로빈 시작점)
        }

        _last_class[n][subnet] = c; // [한국어] 마지막 주입 클래스 갱신 (다음 사이클의 공정성 기준)

        _input_queue[subnet][n][c].pop_front(); // [한국어] 선택한 flit을 입력 큐에서 제거

#ifdef TRACK_FLOWS // [한국어] 흐름 추적: 이 VC로 전송 예정 크레딧 카운터 증가
        ++_outstanding_credits[c][subnet][n];
        _outstanding_classes[n][subnet][f->vc].push(c); // [한국어] 해당 VC로 전송한 클래스 기록
#endif

        dest_buf->SendingFlit(f); // [한국어] BufferState에 flit 전송 통보: 해당 VC 사용 가능 슬롯 -1

        if(_pri_type == network_age_based) { // [한국어] 네트워크 진입 시점 기준 우선순위 갱신
          f->pri = numeric_limits<int>::max() - _time; // [한국어] 주입 시각 기준 나이: 오래될수록 높은 우선순위
          assert(f->pri >= 0);
        }

        if(f->watch) { // [한국어] watch 중이면 주입 이벤트 출력
          *gWatchOut << GetSimTime() << " | "
          << "node" << n << " | "
          << "Injecting flit " << f->id
          << " into subnet " << subnet
          << " at time " << _time
          << " with priority " << f->pri
          << "." << endl;
        }
        f->itime = _time; // [한국어] 주입 시각(itime) 기록 (flat/nlat 레이턴시 계산 기준)

        // Pass VC "back"
        // [한국어] 연속 body/tail flit에 같은 VC 번호를 전달 (패킷 내 VC 일관성 유지)
        if(!_input_queue[subnet][n][c].empty() && !f->tail) { // [한국어] 큐에 다음 flit이 있고 현재가 tail이 아닌 경우
          Flit * const nf = _input_queue[subnet][n][c].front(); // [한국어] 다음 flit (body 또는 tail)
          nf->vc = f->vc; // [한국어] 같은 패킷의 다음 flit에 VC 번호 전달 (head에서 할당한 VC를 body/tail에 승계)
        }

        if((_sim_state == warming_up) || (_sim_state == running)) { // [한국어] 통계 수집 상태이면
          ++_sent_flits[c][n]; // [한국어] 이 노드에서 전송한 flit 수 증가
          if(f->head) { // [한국어] head flit이면 패킷 단위 카운터도 증가
            ++_sent_packets[c][n]; // [한국어] 이 노드에서 전송한 패킷 수 증가
          }
        }

#ifdef TRACK_FLOWS // [한국어] 흐름 추적: 이 노드에서 주입된 flit 카운터 증가
        ++_injected_flits[c][n];
#endif

        _net[subnet]->WriteFlit(f, n); // [한국어] Flit을 서브넷의 injection 포트를 통해 네트워크에 주입

      }
    }
  }
  //Send the credit To the network
  // [한국어] ejection된 flit에 대한 크레딧을 업스트림으로 전송
  for(int subnet = 0; subnet < _subnets; ++subnet) { // [한국어] 각 서브넷별 크레딧 발행
    for(int n = 0; n < _nodes; ++n) { // [한국어] 각 노드별 처리
      map<int, Flit *>::const_iterator iter = flits[subnet].find(n); // [한국어] 이번 사이클에 이 노드에서 ejection된 flit 조회
      if(iter != flits[subnet].end()) { // [한국어] 이번 사이클에 ejection된 flit이 있는 경우
        Flit * const f = iter->second; // [한국어] ejection된 flit 포인터

        f->atime = _time; // [한국어] 도착 시각(atime) 기록 (레이턴시 계산: atime - itime/ctime)
        if(f->watch) { // [한국어] watch 중이면 크레딧 발행 이벤트 출력
          *gWatchOut << GetSimTime() << " | "
          << "node" << n << " | "
          << "Injecting credit for VC " << f->vc
          << " into subnet " << subnet
          << "." << endl;
        }
        Credit * const c = Credit::New(); // [한국어] 크레딧 객체 풀에서 할당
        c->vc.insert(f->vc); // [한국어] 반환하는 VC 번호 기록 (업스트림 라우터가 이 VC의 슬롯을 +1)
        _net[subnet]->WriteCredit(c, n); // [한국어] 크레딧을 노드 n의 injection 포트로 전송 (업스트림 라우터에 전달)

#ifdef TRACK_FLOWS // [한국어] 흐름 추적: 이 노드에서 ejection된 flit 카운터 증가
        ++_ejected_flits[f->cl][n];
#endif

        _RetireFlit(f, n); // [한국어] flit 통계 기록 및 Flit::Free() 호출
      }
    }
    flits[subnet].clear(); // [한국어] 이번 사이클 ejection 맵 초기화 (다음 사이클 준비)
    // _InteralStep here
    _net[subnet]->Evaluate( ); // [한국어] 서브넷의 모든 라우터 내부 스위치 스케줄링 실행 (SA 알고리즘)
    _net[subnet]->WriteOutputs( ); // [한국어] 스케줄링 결과에 따라 출력 FlitChannel에 Flit 전달
  }

  ++_time; // [한국어] NoC 사이클 카운터 증가 (레이턴시 계산의 기준 시각)
  assert(_time); // [한국어] 오버플로 확인 (_time이 0으로 되돌아오면 에러)
  if(gTrace){ // [한국어] gTrace 활성 시 현재 시각 출력 (상세 디버깅)
    cout<<"TIME "<<_time<<endl;
  }

}
