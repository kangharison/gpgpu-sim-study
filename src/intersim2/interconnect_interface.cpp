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
 * [한국어 설명] GPGPU-Sim 인터커넥트 인터페이스 구현 (interconnect_interface.cpp)
 *
 * === 파일의 역할 ===
 * InterconnectInterface 클래스의 모든 메서드를 구현한다. 이 파일이
 * GPGPU-Sim 타이밍 모델과 BookSim2 NoC 시뮬레이터를 실질적으로 연결하는 핵심이다.
 * 특히 Push()는 mem_fetch를 Flit으로 분할하여 NoC에 삽입하고,
 * Pop()은 NoC를 통과한 Flit들로부터 원본 mem_fetch 포인터를 복원하여 반환한다.
 * _CreateNodeMap()은 메시 노드 배치 최적화를 위한 SM-메모리 매핑을 생성한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   gpu-sim.cc (cycle)
 *     → icnt_wrapper.cc (함수 포인터 g_icnt_interface)
 *       → [이 파일] InterconnectInterface::Push / Pop / Advance
 *         → GPUTrafficManager::_GeneratePacket / _Step (gputrafficmanager.cpp)
 *           → Network::WriteFlit / ReadFlit / Evaluate (booksim2 네트워크 코어)
 *
 * === 타 모듈과의 연결 ===
 * - gputrafficmanager.cpp: _traffic_manager를 통해 NoC 사이클 구동
 * - network.hpp / network.cpp: BookSim2 라우터 메시 네트워크 (Network::New, Evaluate)
 * - routefunc.cpp: InitializeRoutingMap()으로 라우팅 함수 등록
 * - power_module.hpp: 소멸 시 전력 분석 실행 (sim_power > 0)
 * - mem_fetch.h: Push()의 data 파라미터가 실제로 mem_fetch* 타입
 * - flit.hpp: NoC 내부 전송 단위; _GeneratePacket이 생성하고 Pop이 소비
 * - intersim_config.hpp: 설정 파일 파싱 (flit_size, num_vcs, subnets 등)
 *
 * === 주요 함수/구조체 요약 ===
 * - New(): 팩토리 메서드 - 설정 파싱 후 인스턴스 반환
 * - CreateInterconnect(): 전체 초기화 (네트워크, 버퍼, 노드 맵)
 * - Push(): mem_fetch → Flit 분할 → _input_queue 삽입
 * - Pop(): boundary_buffer에서 완성된 패킷(mem_fetch*) 반환
 * - Transfer2BoundaryBuffer(): ejection_buffer → boundary_buffer 이동
 * - _CreateNodeMap(): SM/메모리 노드의 최적 메시 배치 매핑 생성
 * - _BoundaryBufferItem::PopPacket(): tail flit까지 pop하여 패킷 반환
 */

#include <fstream>    // [한국어] watch_out 파일 출력 스트림 (ofstream)
#include <iostream>   // [한국어] cout 표준 출력
#include <sstream>    // [한국어] ostringstream: 네트워크 이름 "network_N" 문자열 생성
#include <iomanip>    // [한국어] setw: _DisplayMap에서 정렬된 출력
#include <cmath>      // [한국어] sqrt: _DisplayMap의 메시 차원 계산
#include <utility>    // [한국어] make_pair: 노드 매핑에서 pair<unsigned,unsigned> 생성
#include <algorithm>  // [한국어] copy: config_memory_node 벡터 복사에 사용

#include "interconnect_interface.hpp"  // [한국어] 이 파일의 클래스 선언
#include "routefunc.hpp"               // [한국어] InitializeRoutingMap: 라우팅 함수 등록
#include "globals.hpp"                 // [한국어] gPrintActivity, gTrace, gWatchOut 글로벌 변수
#include "trafficmanager.hpp"          // [한국어] TrafficManager::New: 트래픽 매니저 생성
#include "power_module.hpp"            // [한국어] Power_Module: 전력 분석 (소멸자에서 사용)
#include "mem_fetch.h"                 // [한국어] mem_fetch 타입 정의 (get_type() 호출에 필요)
#include "flit.hpp"                    // [한국어] Flit, Flit::FlitType 정의
#include "gputrafficmanager.hpp"       // [한국어] GPUTrafficManager 타입 (downcast에 필요)
#include "booksim.hpp"                 // [한국어] BookSim 공통 정의 (DPRINTF 등)
#include "intersim_config.hpp"         // [한국어] IntersimConfig 클래스
#include "network.hpp"                 // [한국어] Network::New, Network 타입
#include "trace.h"                     // [한국어] DPRINTF 매크로 정의

/*
 * [한국어]
 * InterconnectInterface::New() - 설정 파일을 파싱하여 인스턴스를 생성하는 팩토리 메서드
 *
 * @config_file: BookSim 설정 파일 경로 (gpgpusim.config에서 icnt_flit_size 등을 포함한 파일)
 * @return: 초기화된 InterconnectInterface 포인터
 *
 * 설정 파일 없이는 시뮬레이터가 동작할 수 없으므로 NULL이면 즉시 exit(-1).
 * 설정 파싱까지만 수행하며, 실제 네트워크 초기화는 CreateInterconnect()에서 별도로 한다.
 *
 * 호출 체인: icnt_wrapper_init() → [이 함수] → CreateInterconnect()
 */
InterconnectInterface* InterconnectInterface::New(const char* const config_file)
{
  if (! config_file ) { // [한국어] 설정 파일 경로가 NULL이면 실행 불가
    cout << "Interconnect Requires a configfile" << endl;
    exit (-1); // [한국어] 설정 없이는 NoC를 초기화할 수 없으므로 즉시 종료
  }
  InterconnectInterface* icnt_interface = new InterconnectInterface(); // [한국어] 빈 인터페이스 객체 할당
  icnt_interface->_icnt_config = new IntersimConfig(); // [한국어] 설정 파서 객체 할당
  icnt_interface->_icnt_config->ParseFile(config_file); // [한국어] gpgpusim.config에서 NoC 설정값 파싱

  return icnt_interface; // [한국어] 설정이 적재된 인터페이스 반환 (CreateInterconnect는 미호출 상태)
}

/*
 * [한국어]
 * InterconnectInterface::InterconnectInterface() - 기본 생성자
 *
 * 멤버 변수를 명시적으로 초기화하지 않는다 (CreateInterconnect에서 모두 설정).
 * New() 팩토리 메서드에서만 호출된다.
 *
 * 호출 체인: New() → [이 생성자]
 */
InterconnectInterface::InterconnectInterface()
{

}

/*
 * [한국어]
 * InterconnectInterface::~InterconnectInterface() - 소멸자
 *
 * 시뮬레이션 종료 시 모든 BookSim 자원을 해제하고, sim_power 설정이 있으면
 * 전력 분석 결과를 먼저 출력한다.
 *
 * 동작 순서:
 *   1. 각 서브넷에 대해 sim_power > 0이면 Power_Module을 생성하여 전력 리포트 출력
 *   2. Network 객체 delete
 *   3. _traffic_manager delete 및 NULL 설정
 *   4. _icnt_config delete
 *
 * 호출 체인: gpgpu_sim 소멸 → g_icnt_interface 소멸 → [이 소멸자]
 */
InterconnectInterface::~InterconnectInterface()
{
  for (int i=0; i<_subnets; ++i) { // [한국어] 모든 서브넷에 대해 정리
    ///Power analysis
    if(_icnt_config->GetInt("sim_power") > 0){ // [한국어] sim_power 설정 > 0이면 전력 분석 활성화
      Power_Module pnet(_net[i], *_icnt_config); // [한국어] 해당 서브넷의 전력 모델 객체 생성
      pnet.run(); // [한국어] 전력 분석 실행 및 결과 출력
    }
    delete _net[i]; // [한국어] BookSim Network 객체 메모리 해제
  }

  delete _traffic_manager; // [한국어] GPUTrafficManager 해제 (Flit 풀, 통계 포함)
  _traffic_manager = NULL; // [한국어] dangling pointer 방지를 위해 NULL로 설정
  delete _icnt_config; // [한국어] 설정 파서 객체 해제
}

/*
 * [한국어]
 * InterconnectInterface::CreateInterconnect() - NoC 전체 구조를 초기화한다
 *
 * @n_shader: SM 클러스터 수 (GPGPU-Sim config의 -gpgpu_n_clusters 값)
 * @n_mem: 메모리 파티션 수 (GPGPU-Sim config의 -gpgpu_n_mem 값)
 *
 * 동작 순서:
 *   1. SM 수와 메모리 수를 멤버 변수에 저장
 *   2. 라우팅 함수 맵 초기화 (VC 범위 설정 포함)
 *   3. 글로벌 디버그 플래그 설정 (print_activity, viewer_trace)
 *   4. watch_out 파일 설정 (디버그 flit 추적 출력 대상)
 *   5. 서브넷 수만큼 Network::New()로 BookSim 네트워크 생성
 *   6. GPUTrafficManager 생성 (sim_type="gpgpusim"이어야 함)
 *   7. 버퍼 용량 설정 (ejection, boundary, input)
 *   8. _CreateBuffer()로 버퍼 배열 할당
 *   9. _CreateNodeMap()으로 deviceID↔icntID 매핑 생성
 *
 * 호출 체인: icnt_wrapper_init() → New() → [이 함수]
 */
void InterconnectInterface::CreateInterconnect(unsigned n_shader, unsigned n_mem)
{
  _n_shader = n_shader; // [한국어] SM 클러스터 수 저장 (Push/Pop에서 서브넷 분기에 사용)
  _n_mem = n_mem; // [한국어] 메모리 파티션 수 저장

  InitializeRoutingMap(*_icnt_config); // [한국어] routefunc.cpp에 등록된 라우팅 함수 맵 초기화, VC 범위 설정

  gPrintActivity = (_icnt_config->GetInt("print_activity") > 0); // [한국어] 사이클별 활동 출력 플래그 설정
  gTrace = (_icnt_config->GetInt("viewer_trace") > 0); // [한국어] flit 추적 트레이스 출력 플래그 설정

  string watch_out_file = _icnt_config->GetStr( "watch_out" ); // [한국어] 디버그 flit 추적 출력 파일 경로 조회
  if(watch_out_file == "") { // [한국어] 설정 없으면 watch 출력 비활성화
    gWatchOut = NULL;
  } else if(watch_out_file == "-") { // [한국어] "-"이면 표준 출력(stdout)으로 추적
    gWatchOut = &cout;
  } else { // [한국어] 파일 경로 지정 시 파일에 추적 출력
    gWatchOut = new ofstream(watch_out_file.c_str());
  }

  _subnets = _icnt_config->GetInt("subnets"); // [한국어] 서브넷 수 읽기 (1=단일망, 2=요청/응답 분리)
  assert(_subnets); // [한국어] 서브넷이 0이면 초기화 불가

  /*To include a new network, must register the network here
   *add an else if statement with the name of the network
   */
  _net.resize(_subnets); // [한국어] 서브넷 수만큼 네트워크 벡터 크기 할당
  for (int i = 0; i < _subnets; ++i) { // [한국어] 각 서브넷에 BookSim 네트워크 생성
    ostringstream name;
    name << "network_" << i; // [한국어] 네트워크 이름: "network_0", "network_1" 등
    _net[i] = Network::New( *_icnt_config, name.str() ); // [한국어] 설정 파일에 따라 mesh/torus 등 네트워크 생성
  }

  assert(_icnt_config->GetStr("sim_type") == "gpgpusim"); // [한국어] GPGPU-Sim 전용 모드 확인
  _traffic_manager = static_cast<GPUTrafficManager*>(TrafficManager::New( *_icnt_config, _net )) ; // [한국어] GPUTrafficManager 생성 후 다운캐스트

  _flit_size = _icnt_config->GetInt( "flit_size" ); // [한국어] flit 크기(바이트) 읽기, Push에서 패킷 분할 크기

  // Config for interface buffers
  if (_icnt_config->GetInt("ejection_buffer_size")) { // [한국어] ejection 버퍼 크기가 명시적으로 설정되어 있으면
    _ejection_buffer_capacity = _icnt_config->GetInt( "ejection_buffer_size" ) ; // [한국어] 설정값 사용
  } else { // [한국어] 설정 없으면 VC 버퍼 크기와 동일하게 설정
    _ejection_buffer_capacity = _icnt_config->GetInt( "vc_buf_size" );
  }

  _boundary_buffer_capacity = _icnt_config->GetInt( "boundary_buffer_size" ) ; // [한국어] boundary 버퍼 flit 용량 설정
  assert(_boundary_buffer_capacity); // [한국어] 0이면 의미 없으므로 assert
  if (_icnt_config->GetInt("input_buffer_size")) { // [한국어] 입력 버퍼 크기 명시 설정 확인
    _input_buffer_capacity = _icnt_config->GetInt("input_buffer_size"); // [한국어] 설정값 사용
  } else { // [한국어] 기본값: 9 flit (실험적으로 결정된 기본값)
    _input_buffer_capacity = 9;
  }
  _vcs = _icnt_config->GetInt("num_vcs"); // [한국어] VC 수 읽기 (헤드-오브-라인 블로킹 방지)

  _CreateBuffer(); // [한국어] [subnets][nodes][vcs] 크기의 버퍼 배열 초기화
  _CreateNodeMap(_n_shader, _n_mem, _traffic_manager->_nodes, _icnt_config->GetInt("use_map")); // [한국어] deviceID↔icntID 매핑 생성
}

/*
 * [한국어]
 * InterconnectInterface::Init() - 트래픽 매니저를 초기 상태로 리셋한다
 *
 * 커널 실행 시작 전에 NoC 상태를 초기화한다.
 * _time=0, _sim_state=running, 통계 초기화를 수행.
 *
 * 호출 체인: icnt_init() → [이 함수] → GPUTrafficManager::Init()
 */
void InterconnectInterface::Init()
{
  _traffic_manager->Init(); // [한국어] GPUTrafficManager::Init()으로 시뮬레이션 상태 초기화
  // TODO: Should we init _round_robin_turn?
  //       _boundary_buffer, _ejection_buffer and _ejected_flit_queue should be cleared
}

/*
 * [한국어]
 * InterconnectInterface::Push() - mem_fetch 패킷을 Flit으로 분할하여 NoC에 삽입한다
 *
 * @input_deviceID: 송신 노드 device ID (SM이면 < _n_shader, 메모리이면 >= _n_shader)
 * @output_deviceID: 수신 노드 device ID
 * @data: 전송할 데이터 포인터 (실제로는 mem_fetch* 타입으로 캐스팅)
 * @size: 패킷 크기 (바이트), Flit 분할 계산에 사용
 *
 * 동작 순서:
 *   1. HasBuffer()로 입력 버퍼 여유 확인 (없으면 assert 실패)
 *   2. deviceID를 icntID로 변환 (_node_map 조회)
 *   3. flit 수 계산: ceil(size / _flit_size)
 *   4. 서브넷 결정: 단일망이면 0, 이중망이면 SM발신=0, 메모리발신=1
 *   5. mem_fetch->get_type()으로 Flit::FlitType 결정
 *   6. _GeneratePacket()으로 n_flits개의 Flit 객체 생성 및 _input_queue에 삽입
 *
 * 주의: data 포인터는 모든 Flit.data에 공유 저장됨 (소유권 이전 없음).
 *
 * 호출 체인: shader_core_ctx → icnt_push() → [이 함수] → _GeneratePacket()
 */
void InterconnectInterface::Push(unsigned input_deviceID, unsigned output_deviceID, void *data, unsigned int size)
{
  // it should have free buffer
  assert(HasBuffer(input_deviceID, size)); // [한국어] 호출자가 HasBuffer 확인 없이 Push하면 assert 실패

  DPRINTF(INTERCONNECT, "Sent %d bytes from %d to %d", size, input_deviceID, output_deviceID); // [한국어] 디버그 출력: 전송 정보

  int output_icntID = _node_map[output_deviceID]; // [한국어] 수신 deviceID → BookSim icntID 변환
  int input_icntID = _node_map[input_deviceID]; // [한국어] 송신 deviceID → BookSim icntID 변환

#if 0
  cout<<"Call interconnect push input: "<<input<<" output: "<<output<<endl;
#endif

  //TODO: move to _IssuePacket
  //TODO: create a Inject and wrap _IssuePacket and _GeneratePacket
  unsigned int n_flits = size / _flit_size + ((size % _flit_size)? 1:0); // [한국어] 패킷을 몇 개의 flit으로 분할할지 계산 (올림)
  int subnet;
  if (_subnets == 1) { // [한국어] 단일 서브넷 구성이면 모든 패킷이 subnet 0 사용
    subnet = 0;
  } else { // [한국어] 이중 서브넷: SM 발신=요청망(0), 메모리 발신=응답망(1)
    if (input_deviceID < _n_shader ) { // [한국어] SM이 보내는 패킷: 메모리 요청 방향 → subnet 0
      subnet = 0;
    } else { // [한국어] 메모리가 보내는 패킷: SM으로의 응답 방향 → subnet 1
      subnet = 1;
    }
  }

  //TODO: Remove mem_fetch to reduce dependency
  Flit::FlitType packet_type; // [한국어] NoC 패킷 타입 (라우팅 VC 선택에 영향)
  mem_fetch* mf = static_cast<mem_fetch*>(data); // [한국어] void* → mem_fetch*로 캐스팅하여 타입 정보 추출

  switch (mf->get_type()) { // [한국어] mem_fetch 타입을 Flit::FlitType으로 변환
    case READ_REQUEST:  packet_type = Flit::READ_REQUEST   ;break; // [한국어] L1 캐시 미스 읽기 요청
    case WRITE_REQUEST: packet_type = Flit::WRITE_REQUEST  ;break; // [한국어] 쓰기 요청 (write-invalidate 등)
    case READ_REPLY:    packet_type = Flit::READ_REPLY     ;break; // [한국어] 메모리 읽기 응답 데이터
    case WRITE_ACK:     packet_type = Flit::WRITE_REPLY    ;break; // [한국어] 쓰기 완료 확인 응답
    default:
    	{
    		cout<<"Type "<<mf->get_type()<<" is undefined!"<<endl;
    		assert (0 && "Type is undefined"); // [한국어] 알 수 없는 패킷 타입은 치명적 오류
    	}
  }

  //TODO: _include_queuing ?
  // [한국어] _GeneratePacket: n_flits개의 Flit 객체를 생성하여 _input_queue[subnet][source][cl]에 삽입
  // cl=0: 단일 트래픽 클래스 (GPU 시뮬레이션에서는 우선순위 클래스 미사용)
  _traffic_manager->_GeneratePacket( input_icntID, -1, 0 /*class*/, _traffic_manager->_time, subnet, n_flits, packet_type, data, output_icntID);

#if DOUB
  cout <<"Traffic[" << subnet << "] (mapped) sending form "<< input_icntID << " to " << output_icntID << endl;
#endif
//  }
}

/*
 * [한국어]
 * InterconnectInterface::Pop() - 수신 완료된 패킷을 boundary_buffer에서 꺼낸다
 *
 * @deviceID: 수신자 노드 device ID
 * @return: 완성된 패킷의 data 포인터(= mem_fetch*), 없으면 NULL
 *
 * 동작 순서:
 *   1. deviceID → icntID 변환
 *   2. SM이면 응답 서브넷(1), 메모리이면 요청 서브넷(0) 선택
 *   3. 라운드-로빈으로 VC를 순환하며 완성된 패킷 탐색
 *   4. 패킷 발견 시 _round_robin_turn 업데이트 및 반환
 *
 * 호출 체인: memory_partition_unit::cycle() → icnt_pop() → [이 함수]
 *         또는 shader_core_ctx::cycle() → icnt_pop() → [이 함수]
 */
void* InterconnectInterface::Pop(unsigned deviceID)
{
  int icntID = _node_map[deviceID]; // [한국어] deviceID를 BookSim icntID로 변환
#if 0
  cout<<"Call interconnect POP  " << output<<endl;
#endif

  void* data = NULL; // [한국어] 반환할 패킷 포인터 초기화 (없으면 NULL 반환)

  // 0-_n_shader-1 indicates reply(network 1), otherwise request(network 0)
  // [한국어] SM 노드(0..n_shader-1)가 Pop하면 응답 서브넷(1)에서 꺼냄
  // 메모리 노드(n_shader..)가 Pop하면 요청 서브넷(0)에서 꺼냄
  int subnet = 0;
  if (deviceID < _n_shader) // [한국어] SM이 응답 패킷을 수신하는 경우: subnet 1 사용
    subnet = 1;

  int turn = _round_robin_turn[subnet][icntID]; // [한국어] 이번 라운드에서 검색 시작할 VC 인덱스
  for (int vc=0;(vc<_vcs) && (data==NULL);vc++) { // [한국어] 모든 VC를 순환하면서 완성된 패킷 탐색
    if (_boundary_buffer[subnet][icntID][turn].HasPacket()) { // [한국어] 이 VC에 완성된 패킷이 있으면
      data = _boundary_buffer[subnet][icntID][turn].PopPacket(); // [한국어] 패킷(mem_fetch 포인터) 추출
    }
    turn++; // [한국어] 다음 VC로 이동
    if (turn == _vcs) turn = 0; // [한국어] VC 인덱스 순환 (마지막 VC에서 0으로)
  }
  if (data) { // [한국어] 패킷을 찾았으면 다음 Pop 호출을 위해 라운드-로빈 포인터 업데이트
    _round_robin_turn[subnet][icntID] = turn; // [한국어] 다음 Pop은 마지막 선택 VC의 다음부터 시작
  }

  return data; // [한국어] mem_fetch* 또는 NULL 반환

}

/*
 * [한국어]
 * InterconnectInterface::Advance() - NoC를 1사이클 진행한다
 *
 * GPUTrafficManager::_Step()을 호출하여 모든 라우터의 파이프라인 단계를 1사이클 실행.
 * GPGPU-Sim의 gpgpu_sim::cycle()이 매 시뮬레이션 사이클마다 이 함수를 호출한다.
 *
 * 호출 체인: gpgpu_sim::cycle() → icnt_transfer() → [이 함수] → _Step()
 */
void InterconnectInterface::Advance()
{
  _traffic_manager->_Step(); // [한국어] NoC 내 모든 라우터 파이프라인 1사이클 진행
}

/*
 * [한국어]
 * InterconnectInterface::Busy() - NoC가 여전히 활성 상태인지 확인한다
 *
 * @return: true = 처리 중인 flit 또는 미처리 패킷이 남아있음
 *
 * 시뮬레이션 종료 조건 판단에 사용.
 * in-flight flit이 없을 때도 _boundary_buffer에 패킷이 남아있으면 busy.
 *
 * 호출 체인: gpgpu_sim::active() → [이 함수]
 */
bool InterconnectInterface::Busy() const
{
  bool busy = !_traffic_manager->_total_in_flight_flits[0].empty(); // [한국어] 첫 번째 클래스 기준으로 in-flight flit 확인
  if (!busy) { // [한국어] in-flight flit이 없으면 _input_queue도 비어있어야 정상
    for (int s = 0; s < _subnets; ++s) { // [한국어] 모든 서브넷 확인
      for (unsigned n = 0; n < _n_shader+_n_mem; ++n) { // [한국어] 모든 노드 확인
        //FIXME: if this cannot make sure _partial_packets is empty
        assert(_traffic_manager->_input_queue[s][n][0].empty()); // [한국어] _input_queue도 비어야 함을 검증
      }
    }
  }
  else
    return true; // [한국어] in-flight flit이 있으면 즉시 busy 반환
  for (int s = 0; s < _subnets; ++s) { // [한국어] 모든 서브넷/노드/VC의 boundary_buffer 확인
    for (unsigned n=0; n < (_n_shader+_n_mem); ++n) { // [한국어] 모든 SM+메모리 노드 순회
      for (int vc=0; vc<_vcs; ++vc) { // [한국어] 모든 VC 순회
        if (_boundary_buffer[s][n][vc].HasPacket() ) { // [한국어] 아직 Pop되지 않은 완성 패킷이 있으면 busy
          return true;
        }
      }
    }
  }
  return false; // [한국어] in-flight flit도 없고 boundary_buffer도 비어있으면 idle
}

/*
 * [한국어]
 * InterconnectInterface::HasBuffer() - 패킷 삽입을 위한 버퍼 여유를 확인한다
 *
 * @deviceID: 송신자 device ID
 * @size: 삽입하려는 패킷 크기 (바이트)
 * @return: true = 충분한 입력 버퍼 공간 있음
 *
 * _input_queue의 현재 flit 수에 새 패킷 flit 수를 더한 값이
 * _input_buffer_capacity를 넘지 않아야 true.
 * backpressure 구현: SM이 Push 전에 이 함수를 반드시 확인해야 한다.
 *
 * 호출 체인: shader_core_ctx → [이 함수] → Push() (true일 때만)
 */
bool InterconnectInterface::HasBuffer(unsigned deviceID, unsigned int size) const
{
  bool has_buffer = false;
  unsigned int n_flits = size / _flit_size + ((size % _flit_size)? 1:0); // [한국어] 패킷을 flit 단위로 변환
  int icntID = _node_map.find(deviceID)->second; // [한국어] deviceID → icntID 변환 (map::find 사용, 범위 검사 겸)

  has_buffer = _traffic_manager->_input_queue[0][icntID][0].size() +n_flits <= _input_buffer_capacity; // [한국어] 기본 서브넷(0) 기준 여유 공간 확인

  if ((_subnets>1) && deviceID >= _n_shader) // deviceID is memory node
    // [한국어] 이중 서브넷이고 메모리 노드이면 응답 서브넷(1)의 _input_queue 확인
    has_buffer = _traffic_manager->_input_queue[1][icntID][0].size() +n_flits <= _input_buffer_capacity;

  return has_buffer;
}

/*
 * [한국어]
 * InterconnectInterface::DisplayStats() - 현재 측정 구간의 통계를 출력한다
 *
 * UpdateStats()로 내부 통계를 집계한 뒤 DisplayStats()로 출력.
 * 커널 실행 완료 시 호출되어 latency, throughput 등 NoC 성능 지표 보고.
 *
 * 호출 체인: gpu_print_stat() → [이 함수]
 */
void InterconnectInterface::DisplayStats() const
{
  _traffic_manager->UpdateStats(); // [한국어] 내부 플로팅 통계 집계 (평균, min, max 계산)
  _traffic_manager->DisplayStats(); // [한국어] 집계된 통계 출력 (flit latency, packet latency 등)
}

/*
 * [한국어]
 * InterconnectInterface::GetFlitSize() - flit 크기 반환
 *
 * @return: _flit_size (바이트 단위)
 *
 * 호출 체인: icnt_get_flit_size() → [이 함수]
 */
unsigned InterconnectInterface::GetFlitSize() const
{
  return _flit_size; // [한국어] 설정 파일에서 읽은 flit_size 값 반환
}

/*
 * [한국어]
 * InterconnectInterface::DisplayOverallStats() - 전체 시뮬레이션 통합 통계 출력
 *
 * BookSim의 _UpdateOverallStats()는 _drain_time과 _total_sims를 기반으로 계산하므로,
 * 이 함수는 GPGPU-Sim 맥락에 맞게 두 값을 수동으로 설정하는 해킹을 포함한다.
 * _drain_time = _time: 마지막 사이클을 drain 시점으로 가정
 * _total_sims += 1: 각 커널을 하나의 시뮬레이션으로 간주
 *
 * 호출 체인: gpu_print_stat() 마지막 → [이 함수]
 */
void InterconnectInterface::DisplayOverallStats() const
{
  // hack: booksim2 use _drain_time and calculate delta time based on it, but we don't, change this if you have a better idea
  // [한국어] BookSim은 _drain_time 기준으로 delta 시간을 계산하지만,
  // GPGPU-Sim은 drain 개념을 사용하지 않으므로 현재 시간을 drain 시점으로 설정
  _traffic_manager->_drain_time = _traffic_manager->_time; // [한국어] 현재 시뮬레이션 시간을 drain 완료 시점으로 설정
  // hack: also _total_sims equals to number of kernel calls
  _traffic_manager->_total_sims += 1; // [한국어] 커널 호출 횟수를 시뮬레이션 횟수로 카운트

  _traffic_manager->_UpdateOverallStats(); // [한국어] 전체 통합 통계 집계 (min/avg/max across all kernels)
  _traffic_manager->DisplayOverallStats(); // [한국어] 전체 통합 통계 출력
  if(_traffic_manager->_print_csv_results) { // [한국어] CSV 형식 출력이 설정되어 있으면
    _traffic_manager->DisplayOverallStatsCSV(); // [한국어] CSV 형식으로도 출력
  }
}

/*
 * [한국어]
 * InterconnectInterface::DisplayState() - NoC 현재 상태를 파일에 출력한다 (미구현)
 *
 * @fp: 출력 대상 FILE 포인터
 *
 * 현재는 "Under implementation" 문자열만 출력.
 * 이전 코드(주석 처리)에서는 in-flight 수, boundary 버퍼 상태 등을 출력했으나
 * 인터페이스 리팩토링으로 제거됨.
 *
 * 호출 체인: gpgpu_sim::print_stats() → [이 함수]
 */
void InterconnectInterface::DisplayState(FILE *fp) const
{
  fprintf(fp, "GPGPU-Sim uArch: ICNT:Display State: Under implementation\n"); // [한국어] 미구현 상태 메시지 출력
//  fprintf(fp,"GPGPU-Sim uArch: interconnect busy state\n");

//  for (unsigned i=0; i<net_c;i++) {
//    if (traffic[i]->_measured_in_flight)
//      fprintf(fp,"   Network %u has %u _measured_in_flight\n", i, traffic[i]->_measured_in_flight );
//  }
//
//  for (unsigned i=0 ;i<(_n_shader+_n_mem);i++ ) {
//    if( !traffic[0]->_partial_packets[i] [0].empty() )
//      fprintf(fp,"   Network 0 has nonempty _partial_packets[%u][0]\n", i);
//    if ( doub_net && !traffic[1]->_partial_packets[i] [0].empty() )
//      fprintf(fp,"   Network 1 has nonempty _partial_packets[%u][0]\n", i);
//    for (unsigned j=0;j<g_num_vcs;j++ ) {
//      if( !ejection_buf[i][j].empty() )
//        fprintf(fp,"   ejection_buf[%u][%u] is non-empty\n", i, j);
//      if( clock_boundary_buf[i][j].has_packet() )
//        fprintf(fp,"   clock_boundary_buf[%u][%u] has packet\n", i, j );
//    }
//  }
}

/*
 * [한국어]
 * InterconnectInterface::Transfer2BoundaryBuffer() - ejection_buffer → boundary_buffer 이동
 *
 * @subnet: 서브넷 인덱스
 * @output: 목적지 노드 icntID
 *
 * GPUTrafficManager::_Step()이 각 노드에 대해 이 함수를 호출한다.
 * 각 VC의 ejection_buffer에서 flit을 꺼내어 boundary_buffer의 PushFlitData()로 삽입.
 * 삽입된 flit은 _ejected_flit_queue에도 넣어 크레딧 반환 준비를 한다.
 *
 * 흐름 제어: _boundary_buffer[subnet][output][vc].Size() < _boundary_buffer_capacity일 때만 이동.
 * head flit이면 dest 필드가 output과 일치하는지 검증한다.
 *
 * 호출 체인: GPUTrafficManager::_Step() → [이 함수] → PushFlitData() + _ejected_flit_queue.push()
 */
void InterconnectInterface::Transfer2BoundaryBuffer(int subnet, int output)
{
  Flit* flit;
  int vc;
  for (vc=0; vc<_vcs;vc++) { // [한국어] 모든 VC에 대해 ejection → boundary 이동 시도

    if ( !_ejection_buffer[subnet][output][vc].empty() && _boundary_buffer[subnet][output][vc].Size() < _boundary_buffer_capacity ) {
      // [한국어] ejection 버퍼에 flit이 있고 boundary 버퍼에 공간이 있으면 이동
      flit = _ejection_buffer[subnet][output][vc].front(); // [한국어] 배출된 flit 참조
      assert(flit); // [한국어] NULL flit은 오류

      _ejection_buffer[subnet][output][vc].pop(); // [한국어] ejection 버퍼에서 제거
      _boundary_buffer[subnet][output][vc].PushFlitData( flit->data, flit->tail); // [한국어] flit 데이터와 tail 여부를 boundary 버퍼에 삽입

      _ejected_flit_queue[subnet][output].push(flit); //indicate this flit is already popped from ejection buffer and ready for credit return
      // [한국어] 크레딧 반환 준비 큐에 추가
      // 크레딧 반환: 이 flit을 수신한 것을 상류 라우터에 알려 버퍼 공간을 복원

      if ( flit->head ) { // [한국어] head flit이면 목적지 주소 검증
        assert (flit->dest == output); // [한국어] head flit의 dest가 현재 노드와 일치해야 함
      }
    }
  }
}

/*
 * [한국어]
 * InterconnectInterface::WriteOutBuffer() - Network에서 배출된 flit을 ejection_buffer에 저장
 *
 * @subnet: 서브넷 인덱스
 * @output_icntID: 목적지 노드 icntID
 * @flit: Network::ReadFlit()이 반환한 배출 flit
 *
 * flit의 vc 필드로 해당 VC의 ejection_buffer에 push한다.
 * 용량 초과 시 assert 실패 (ejection_buffer는 항상 여유가 있어야 함).
 *
 * 호출 체인: GPUTrafficManager::_Step() → [이 함수]
 */
void InterconnectInterface::WriteOutBuffer(int subnet, int output_icntID, Flit*  flit )
{
  int vc = flit->vc; // [한국어] flit이 사용하는 VC 번호로 해당 ejection 버퍼에 저장
  assert (_ejection_buffer[subnet][output_icntID][vc].size() < _ejection_buffer_capacity); // [한국어] 용량 초과 방지
  _ejection_buffer[subnet][output_icntID][vc].push(flit); // [한국어] 해당 VC의 ejection 버퍼에 flit 저장
}

/*
 * [한국어]
 * InterconnectInterface::GetIcntTime() - NoC 시뮬레이션 현재 시간을 반환한다
 *
 * @return: _traffic_manager->getTime() (NoC 사이클 카운터)
 *
 * 호출 체인: 통계/디버그 코드 → [이 함수]
 */
int InterconnectInterface::GetIcntTime() const
{
  return _traffic_manager->getTime(); // [한국어] TrafficManager::_time 값 반환
}

/*
 * [한국어]
 * InterconnectInterface::GetIcntStats() - 이름으로 BookSim 통계 객체를 반환한다
 *
 * @name: 통계 객체 이름
 * @return: Stats* 포인터 (없으면 NULL)
 *
 * 호출 체인: 외부 분석 코드 → [이 함수]
 */
Stats* InterconnectInterface::GetIcntStats(const string &name) const
{
  return _traffic_manager->getStats(name); // [한국어] TrafficManager::_stats 맵에서 이름으로 조회
}

/*
 * [한국어]
 * InterconnectInterface::GetEjectedFlit() - 크레딧 반환용 flit을 꺼낸다
 *
 * @subnet: 서브넷 인덱스
 * @node: 노드 icntID
 * @return: _ejected_flit_queue의 front flit, 없으면 NULL
 *
 * GPUTrafficManager::_Step()이 이 함수로 flit을 꺼내어 upstream에 크레딧을 반환한다.
 * 크레딧 반환은 "이 VC에 버퍼 공간이 생겼다"는 신호를 상류 라우터에 보내는 것.
 *
 * 호출 체인: GPUTrafficManager::_Step() → [이 함수] → Credit::New() 및 WriteCredit()
 */
Flit* InterconnectInterface::GetEjectedFlit(int subnet, int node)
{
  Flit* flit = NULL; // [한국어] 기본 반환값: NULL (큐가 비어있으면)
  if (!_ejected_flit_queue[subnet][node].empty()) { // [한국어] 반환 대기 flit이 있으면
    flit = _ejected_flit_queue[subnet][node].front(); // [한국어] 큐 맨 앞 flit 참조
    _ejected_flit_queue[subnet][node].pop(); // [한국어] 큐에서 제거
  }
  return flit; // [한국어] flit 포인터 반환 (호출자가 크레딧 반환 처리)
}

/*
 * [한국어]
 * InterconnectInterface::_CreateBuffer() - 모든 버퍼 배열을 할당한다
 *
 * _boundary_buffer, _ejection_buffer, _ejected_flit_queue, _round_robin_turn을
 * [subnets][nodes][vcs] 또는 [subnets][nodes] 형태로 초기화한다.
 * nodes는 Network::NumNodes()로 얻는 BookSim 내부 노드 수.
 *
 * 호출 체인: CreateInterconnect() → [이 함수]
 */
void InterconnectInterface::_CreateBuffer()
{
  unsigned nodes = _net[0]->NumNodes(); // [한국어] 전체 노드 수 (SM + 메모리 노드 합계)

  _boundary_buffer.resize(_subnets); // [한국어] 서브넷 차원 크기 설정
  _ejection_buffer.resize(_subnets); // [한국어] 서브넷 차원 크기 설정
  _round_robin_turn.resize(_subnets); // [한국어] 서브넷 차원 크기 설정
  _ejected_flit_queue.resize(_subnets); // [한국어] 서브넷 차원 크기 설정

  for (int subnet = 0; subnet < _subnets; ++subnet) { // [한국어] 각 서브넷에 대해 노드 차원 초기화
    _ejection_buffer[subnet].resize(nodes); // [한국어] 노드 차원 크기 설정
    _boundary_buffer[subnet].resize(nodes); // [한국어] 노드 차원 크기 설정
    _round_robin_turn[subnet].resize(nodes); // [한국어] 노드별 RR 포인터 (기본값 0으로 초기화됨)
    _ejected_flit_queue[subnet].resize(nodes); // [한국어] 노드별 배출 flit 큐

    for (unsigned node=0;node < nodes;++node){ // [한국어] 각 노드에 대해 VC 차원 초기화
      _ejection_buffer[subnet][node].resize(_vcs); // [한국어] VC 수만큼 ejection 버퍼 할당
      _boundary_buffer[subnet][node].resize(_vcs); // [한국어] VC 수만큼 boundary 버퍼 할당
    }
  }
}

/*
 * [한국어]
 * InterconnectInterface::_CreateNodeMap() - deviceID와 icntID 사이의 양방향 매핑 생성
 *
 * @n_shader: SM 수
 * @n_mem: 메모리 파티션 수
 * @n_node: 전체 NoC 노드 수 (Network::NumNodes())
 * @use_map: 1이면 최적 매핑 사용, 0이면 항등 매핑 (deviceID==icntID)
 *
 * use_map=1이면 SM-메모리 간 평균 홉 수를 최소화하는 사전 정의된 배치를 사용.
 * SM과 메모리를 메시에 인터리빙 배치하여 트래픽 편중을 방지한다.
 * 지원 구성: (8SM,8메모리), (28SM,8메모리), (56SM,8메모리), (110SM,11메모리)
 * 또는 memory_node_map 설정으로 커스텀 매핑 지정 가능.
 *
 * 호출 체인: CreateInterconnect() → [이 함수] → _DisplayMap()
 */
void InterconnectInterface::_CreateNodeMap(unsigned n_shader, unsigned n_mem, unsigned n_node, int use_map)
{
  if (use_map) { // [한국어] 최적 노드 배치 매핑 사용
    // The (<SM, Memory>, Memory Location Vector) map
    // [한국어] (SM수, 메모리수) 쌍에서 메모리 노드의 icntID 목록으로의 사전 정의 맵
    map<pair<unsigned,unsigned>, vector<unsigned> > preset_memory_map;

    // preset memory and shader map, optimized for mesh
    // good for 8 SMs and 8 memory ports, the map is as follows:
    // +--+--+--+--+
    // |C0|M0|C1|M1|
    // +--+--+--+--+
    // |M2|C2|M3|C3|
    // +--+--+--+--+
    // |C4|M4|C5|M5|
    // +--+--+--+--+
    // |M6|C6|M7|C7|
    // +--+--+--+--+
    // [한국어] 8SM/8메모리용: 4x4 메시에서 SM(C)과 메모리(M)를 체커보드 패턴으로 배치
    {
      unsigned memory_node[] = {1, 3, 4, 6, 9, 11, 12, 14}; // [한국어] 8개 메모리 노드의 icntID (메시 내 물리 위치)
      preset_memory_map[make_pair(8,8)] = vector<unsigned>(memory_node, memory_node+8);
    }

    // good for 28 SMs and 8 memory ports
    // [한국어] 28SM/8메모리용 최적 배치
    {
      unsigned memory_node[] = {3, 7, 10, 12, 23, 25, 28, 32};
      preset_memory_map[make_pair(28,8)] = vector<unsigned>(memory_node, memory_node+8);
    }

    // good for 56 SMs and 8 memory cores
    // [한국어] 56SM/8메모리용 최적 배치
    {
      unsigned memory_node[] = {3, 15, 17, 29, 36, 47, 49, 61};
      preset_memory_map[make_pair(56,8)] = vector<unsigned>(memory_node, memory_node+sizeof(memory_node)/sizeof(unsigned));
    }

    // good for 110 SMs and 11 memory cores
    // [한국어] 110SM/11메모리용 최적 배치 (대형 GPU 모델용)
    {
      unsigned memory_node[] = {12, 20, 25, 28, 57, 60, 63, 92, 95,100,108};
      preset_memory_map[make_pair(110, 11)] = vector<unsigned>(memory_node, memory_node+sizeof(memory_node)/sizeof(unsigned));
    }
    const vector<int> config_memory_node(_icnt_config->GetIntArray("memory_node_map")); // [한국어] 설정 파일에서 커스텀 memory_node_map 읽기
    if (!config_memory_node.empty()) { // [한국어] 커스텀 매핑이 지정된 경우
      if (config_memory_node.size() != _n_mem) { // [한국어] 메모리 수와 매핑 항목 수가 일치해야 함
        cerr << "Number of memory nodes in memory_node_map should equal to memory ports" << endl;
        assert( config_memory_node.size() == _n_mem);
      }
      vector<unsigned> t_memory_node(config_memory_node.size()); // [한국어] int 벡터를 unsigned 벡터로 변환
      copy(config_memory_node.begin(), config_memory_node.end(), t_memory_node.begin());
      preset_memory_map[make_pair(_n_shader, _n_mem)] = t_memory_node; // [한국어] 커스텀 매핑 등록
    }

    const vector<unsigned> &memory_node = preset_memory_map[make_pair(_n_shader, _n_mem)]; // [한국어] 현재 SM/메모리 구성에 해당하는 배치 조회
    if (memory_node.empty()) { // [한국어] 지원하지 않는 구성이면 오류 (use_map=0으로 변경 필요)
      cerr<<"ERROR!!! NO MAPPING IMPLEMENTED YET FOR THIS CONFIG"<<endl;
      assert(0);
    }

    // create node map
    // [한국어] SM들의 icntID를 메모리 노드 위치를 피해서 순차 할당
    unsigned next_node = 0; // [한국어] 다음 SM에 할당할 icntID 후보
    unsigned memory_node_index = 0; // [한국어] 메모리 노드 배열 현재 인덱스
    for (unsigned i = 0; i < n_shader; ++i) { // [한국어] 각 SM에 icntID 할당
      while (next_node == memory_node[memory_node_index]) { // [한국어] 다음 노드가 메모리 위치이면 건너뜀
        next_node += 1; // [한국어] 메모리 자리를 건너뛰어 다음 노드로
        memory_node_index += 1; // [한국어] 메모리 노드 인덱스 증가
      }
      _node_map[i] = next_node; // [한국어] SM deviceID i → icntID next_node 매핑
      next_node += 1; // [한국어] 다음 SM을 위해 icntID 증가
    }
    for (unsigned i = n_shader; i < n_shader+n_mem; ++i) { // [한국어] 메모리 노드들의 icntID 직접 매핑
      _node_map[i] = memory_node[i-n_shader]; // [한국어] 메모리 deviceID → 사전 정의된 icntID
    }
  } else { //not use preset map
    // [한국어] 최적 매핑 비사용: deviceID = icntID (항등 매핑)
    for (unsigned i=0;i<n_node;i++) {
      _node_map[i]=i; // [한국어] deviceID와 icntID를 동일하게 설정
    }
  }

  // [한국어] _node_map의 역(icntID → deviceID)인 _reverse_node_map 생성
  for (unsigned i = 0; i < n_node ; i++) { // [한국어] 모든 icntID에 대해
    for (unsigned j = 0; j< n_node ; j++) { // [한국어] _node_map에서 value == i인 j를 찾음
      if ( _node_map[j] == i ) { // [한국어] j번 device의 icntID가 i이면
        _reverse_node_map[i]=j; // [한국어] icntID i → deviceID j 역매핑 저장
        break; // [한국어] 하나 찾으면 종료 (1:1 매핑)
      }
    }
  }

  //FIXME: should compatible with non-square number
  _DisplayMap((int) sqrt(n_node), n_node); // [한국어] 매핑 결과를 메시 형태로 출력 (정사각형 메시 가정)

}

/*
 * [한국어]
 * InterconnectInterface::_DisplayMap() - 노드 매핑 테이블을 메시 형태로 출력한다
 *
 * @dim: 메시의 한 변 크기 (sqrt(n_node))
 * @count: 전체 노드 수
 *
 * deviceID→icntID, icntID→deviceID 두 방향의 매핑을 메시 격자 형태로 출력.
 * SM 노드와 메모리 노드의 물리적 배치를 시각적으로 확인할 수 있다.
 *
 * 호출 체인: _CreateNodeMap() → [이 함수]
 */
void InterconnectInterface::_DisplayMap(int dim,int count)
{
  cout << "GPGPU-Sim uArch: interconnect node map (shaderID+MemID to icntID)" << endl;
  cout << "GPGPU-Sim uArch: Memory nodes ID start from index: " << _n_shader << endl; // [한국어] 메모리 노드 시작 deviceID 출력
  cout << "GPGPU-Sim uArch: ";
  for (int i = 0;i < count; i++) { // [한국어] 모든 deviceID에 대해 icntID 출력
    cout << setw(4) << _node_map[i]; // [한국어] 4칸 너비로 정렬하여 icntID 출력
    if ((i+1)%dim == 0 && i != count-1) // [한국어] 한 행이 끝나면 줄바꿈 (마지막 행 제외)
      cout << endl << "GPGPU-Sim uArch: ";
  }
  cout << endl;

  cout << "GPGPU-Sim uArch: interconnect node reverse map (icntID to shaderID+MemID)" << endl;
  cout << "GPGPU-Sim uArch: Memory nodes start from ID: " << _n_shader << endl;
  cout << "GPGPU-Sim uArch: ";
  for (int i = 0;i < count; i++) { // [한국어] 모든 icntID에 대해 deviceID 출력
    cout << setw(4) << _reverse_node_map[i]; // [한국어] 4칸 너비로 정렬하여 deviceID 출력
    if ((i+1)%dim == 0 && i != count-1)
      cout << endl << "GPGPU-Sim uArch: ";
  }
  cout << endl;
}

/*
 * [한국어]
 * InterconnectInterface::_BoundaryBufferItem::PopPacket() - 완성된 패킷을 꺼낸다
 *
 * @return: tail flit의 data 포인터 (= mem_fetch*)
 *
 * 동작:
 *   1. _packet_n이 0이면 assert 실패 (완성된 패킷이 없는데 Pop 시도)
 *   2. _buffer와 _tail_flag를 동시에 pop하면서 tail이 나올 때까지 flit 버림
 *   3. tail flit의 data 포인터를 반환하고 _packet_n 감소
 *
 * 전제 조건: 같은 패킷의 모든 flit이 동일한 data 포인터를 가짐.
 * (PushFlitData에서 flit.data로 저장되며, 모든 flit이 원본 mem_fetch*를 공유)
 *
 * 호출 체인: Pop() → _BoundaryBufferItem::HasPacket() → [이 함수]
 */
void* InterconnectInterface::_BoundaryBufferItem::PopPacket()
{
  assert (_packet_n); // [한국어] 완성된 패킷이 있어야 Pop 가능
  void * data = NULL; // [한국어] 반환할 mem_fetch 포인터 초기화
  void * flit_data = _buffer.front(); // [한국어] 첫 flit 데이터 참조 (패킷 일관성 검증용)
  while (data == NULL) { // [한국어] tail flit을 찾을 때까지 반복
    assert(flit_data == _buffer.front()); //all flits must belong to the same packet
    // [한국어] 같은 패킷의 모든 flit은 동일한 data 포인터를 가져야 함
    if (_tail_flag.front()) { // [한국어] tail flit이면 이 flit의 data가 패킷 대표 포인터
      data = _buffer.front(); // [한국어] tail flit의 data = mem_fetch 포인터
      _packet_n--; // [한국어] 완성된 패킷 카운터 감소
    }
    _buffer.pop(); // [한국어] flit 데이터 큐에서 제거
    _tail_flag.pop(); // [한국어] tail 플래그 큐에서 제거 (항상 _buffer와 동시 pop)
  }
  return data; // [한국어] mem_fetch* 반환
}

/*
 * [한국어]
 * InterconnectInterface::_BoundaryBufferItem::TopPacket() - pop 없이 패킷 데이터 조회 (peek)
 *
 * @return: 완성된 패킷의 data 포인터
 *
 * 현재 코드에서는 사용되지 않는 함수.
 * 구현에 잠재적 무한루프 버그가 있음: while(data==NULL)에서
 * _tail_flag가 false인 경우 _buffer.pop()이 호출되지 않아 무한루프 가능.
 *
 * 호출 체인: (미사용)
 */
void* InterconnectInterface::_BoundaryBufferItem::TopPacket() const
{
  assert (_packet_n); // [한국어] 완성된 패킷 존재 확인
  void* data = NULL;
  void* temp_d = _buffer.front(); // [한국어] 현재 front flit 데이터 참조
  while (data==NULL) { // [한국어] 주의: 무한루프 잠재적 버그 - pop() 없는 반복
    if (_tail_flag.front()) { // [한국어] tail이면 데이터 반환
      data = _buffer.front();
    }
    assert(temp_d == _buffer.front()); //all flits must belong to the same packet
    // [한국어] 패킷 데이터 일관성 검증 (tail이 나올 때까지 front는 같아야 함)
  }
  return data;

}

/*
 * [한국어]
 * InterconnectInterface::_BoundaryBufferItem::PushFlitData() - flit 데이터 삽입
 *
 * @data: flit의 data 포인터 (= mem_fetch* 공유 포인터)
 * @is_tail: 이 flit이 패킷의 마지막 flit인지 여부
 *
 * _buffer와 _tail_flag에 동시 삽입하여 1:1 대응 유지.
 * is_tail이면 _packet_n을 증가시켜 완성된 패킷으로 등록.
 *
 * 호출 체인: Transfer2BoundaryBuffer() → [이 함수]
 */
void InterconnectInterface::_BoundaryBufferItem::PushFlitData(void* data,bool is_tail)
{
  _buffer.push(data); // [한국어] flit 데이터(mem_fetch 포인터) 큐에 추가
  _tail_flag.push(is_tail); // [한국어] tail 여부를 대응하는 큐에 동시 추가
  if (is_tail) { // [한국어] tail flit이면 패킷 완성 카운터 증가
    _packet_n++; // [한국어] HasPacket()이 true를 반환하도록 카운터 증가
  }
}
