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
 * [한국어 설명] GPGPU-Sim 인터커넥트 인터페이스 헤더 (interconnect_interface.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim 타이밍 모델과 BookSim 2 기반 NoC 시뮬레이터(intersim2) 사이의
 * 유일한 공개 API 경계를 정의한다. GPGPU-Sim의 gpu-sim.cc나 icnt_wrapper.cc는
 * 이 클래스의 메서드(Init, Push, Pop, Advance, HasBuffer)만을 통해 NoC와 통신한다.
 * 따라서 이 파일이 "NoC를 GPU 시뮬레이터에 붙이는 어댑터 레이어"의 선언부이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   gpgpu_sim::cycle()  →  icnt_wrapper (함수 포인터)
 *     → InterconnectInterface::Push()   (SM이 메모리에 요청 전송)
 *     → InterconnectInterface::Advance() (NoC 1사이클 진행)
 *     → InterconnectInterface::Pop()    (메모리가 SM으로부터 요청 수신)
 *
 * 실행 컨텍스트: GPGPU-Sim 메인 시뮬레이션 루프(단일 스레드), 사이클마다 호출.
 * NoC 내부는 GPUTrafficManager::_Step()이 시뮬레이션하며, 이 인터페이스가
 * 그 진입점 역할을 한다.
 *
 * === 타 모듈과의 연결 ===
 * - icnt_wrapper.cc: InterconnectInterface 포인터(g_icnt_interface)를 통해 이 API 호출
 * - GPUTrafficManager: 실제 NoC 사이클 시뮬레이션 담당 (_traffic_manager 멤버)
 * - Network (BookSim): 라우터 메시 네트워크 구조체, _net[] 벡터로 보유
 * - mem_fetch: Push()로 전달되는 데이터의 실제 타입 (void* data 캐스팅)
 * - Flit: NoC 내부 전송 단위; Push에서 mem_fetch를 분할하여 Flit으로 생성
 * 데이터 흐름: mem_fetch → Flit 분할 → _input_queue → NoC 라우팅 → ejection_buffer
 *              → _boundary_buffer → Pop()으로 재조립 → mem_fetch 반환
 *
 * === 주요 함수/구조체 요약 ===
 * - InterconnectInterface::New(): 설정 파일을 파싱하여 인스턴스 생성 (팩토리 메서드)
 * - CreateInterconnect(): NoC 네트워크, 버퍼, 노드 맵 전체 초기화
 * - Push(): mem_fetch를 Flit으로 분할하여 NoC 입력 큐에 삽입
 * - Pop(): 목적지 노드의 boundary_buffer에서 완료된 패킷(mem_fetch 포인터) 추출
 * - Advance(): GPUTrafficManager::_Step() 호출로 NoC 1사이클 진행
 * - HasBuffer(): 입력 버퍼 여유 공간 확인 (backpressure 구현)
 * - _BoundaryBufferItem: Flit들을 모아 패킷 단위로 Pop할 수 있게 하는 내부 버퍼
 */

#ifndef _INTERCONNECT_INTERFACE_HPP_
#define _INTERCONNECT_INTERFACE_HPP_

#include <vector>   // [한국어] _boundary_buffer, _ejection_buffer 등 다차원 벡터에 사용
#include <queue>    // [한국어] _EjectionBufferItem, _ejected_flit_queue 등 FIFO 큐에 사용
#include <iostream> // [한국어] 디버그 출력 및 오류 메시지 스트림
#include <map>      // [한국어] deviceID ↔ icntID 양방향 노드 매핑 테이블에 사용
using namespace std;


// Do not use #include since it will not compile in icnt_wrapper or change the makefile to make it
// [한국어] 순환 의존성 방지를 위해 전방 선언만 사용한다.
// interconnect_interface.hpp는 icnt_wrapper.cc에서도 포함되므로,
// BookSim 헤더를 직접 #include하면 컴파일 오류가 생길 수 있다.
class Flit;               // [한국어] NoC 전송 기본 단위 (flit.hpp에 정의)
class GPUTrafficManager;  // [한국어] GPU 특화 트래픽 매니저 (gputrafficmanager.hpp에 정의)
class IntersimConfig;     // [한국어] BookSim 설정 파싱 객체 (intersim_config.hpp에 정의)
class Network;            // [한국어] BookSim 라우터 네트워크 (network.hpp에 정의)
class Stats;              // [한국어] BookSim 통계 수집 객체 (stats.hpp에 정의)

//TODO: fixed_lat_icnt, add class support? support for signle network

/*
 * [한국어]
 * InterconnectInterface - GPGPU-Sim과 BookSim2 NoC 사이의 어댑터 클래스
 *
 * GPU 시뮬레이터(GPGPU-Sim)는 이 클래스의 가상 함수들만을 통해 NoC와 통신한다.
 * 내부적으로는 BookSim2의 Network, GPUTrafficManager, Flit 등 복잡한 NoC 자료구조를
 * 캡슐화하여 GPGPU-Sim 코드가 NoC 구현 세부사항을 알 필요 없게 만든다.
 *
 * 노드 번호 체계:
 *   deviceID 0 .. n_shader-1: SM 클러스터 (shader 코어)
 *   deviceID n_shader .. n_shader+n_mem-1: 메모리 파티션 노드
 *   icntID: BookSim 내부 노드 번호 (use_map=1이면 deviceID와 다를 수 있음)
 *
 * 서브넷(subnet) 체계:
 *   subnets=1이면 단일 네트워크로 요청/응답 모두 처리.
 *   subnets=2이면 subnet0 = SM→메모리(요청), subnet1 = 메모리→SM(응답)으로 분리
 *   → 데드락 방지를 위해 요청과 응답 패킷이 별도의 물리 네트워크를 사용.
 */
class InterconnectInterface {
public:
  /*
   * [한국어]
   * InterconnectInterface() - 기본 생성자
   *
   * New() 팩토리 메서드에 의해 호출된다. 멤버 변수 초기화는 CreateInterconnect()에서 수행.
   * 직접 생성자를 호출하면 설정이 없는 미완성 객체가 되므로, 반드시 New()를 사용해야 한다.
   *
   * 호출 체인: InterconnectInterface::New() → [이 생성자] → _icnt_config 할당
   */
  InterconnectInterface();

  /*
   * [한국어]
   * ~InterconnectInterface() - 소멸자
   *
   * Power_Module을 실행하여 전력 분석 결과를 출력한 뒤,
   * _net[] 벡터의 각 Network 객체, _traffic_manager, _icnt_config를 해제한다.
   * sim_power 설정이 0보다 크면 소멸 시점에 전력 리포트가 출력된다.
   *
   * 호출 체인: [시뮬레이션 종료] → gpgpu_sim 소멸 → [이 소멸자]
   */
  virtual ~InterconnectInterface();

  /*
   * [한국어]
   * New() - 설정 파일을 파싱하여 InterconnectInterface 인스턴스를 생성하는 팩토리 메서드
   *
   * @config_file: IntersimConfig에 파싱할 설정 파일 경로 (gpgpusim.config 내 지정)
   * @return: 초기화된 InterconnectInterface 포인터 (실패 시 exit(-1) 호출)
   *
   * 동작 과정:
   *   1. 설정 파일 경로 유효성 검사
   *   2. InterconnectInterface 객체 new 할당
   *   3. IntersimConfig::ParseFile()로 설정 파일 파싱
   *   4. 완성된 객체 반환 (CreateInterconnect는 별도 호출 필요)
   *
   * 호출 체인: icnt_wrapper_init() → [이 함수] → CreateInterconnect()
   */
  static InterconnectInterface* New(const char* const config_file);

  /*
   * [한국어]
   * CreateInterconnect() - NoC 네트워크와 버퍼 전체를 초기화한다
   *
   * @n_shader: SM 클러스터 수 (GPGPU-Sim의 n_simt_clusters)
   * @n_mem: 메모리 파티션 수 (gpgpu_n_mem)
   *
   * 동작 과정:
   *   1. 라우팅 맵 초기화 (InitializeRoutingMap)
   *   2. gPrintActivity, gTrace 글로벌 플래그 설정
   *   3. 서브넷 수만큼 Network::New()로 BookSim 네트워크 생성
   *   4. GPUTrafficManager 생성 (TrafficManager::New → GPUTrafficManager 다운캐스트)
   *   5. 버퍼 크기 설정 (ejection, boundary, input 버퍼)
   *   6. _CreateBuffer()로 버퍼 할당
   *   7. _CreateNodeMap()으로 deviceID ↔ icntID 매핑 생성
   *
   * 호출 체인: icnt_wrapper_init() → New() → [이 함수]
   */
  virtual void CreateInterconnect(unsigned n_shader,  unsigned n_mem);

  // =====================================================================
  // [한국어] 노드 측 공개 API (GPGPU-Sim이 직접 호출하는 함수들)
  // icnt_wrapper.cc의 함수 포인터를 통해 간접 호출됨
  // =====================================================================

  //node side functions
  /*
   * [한국어]
   * Init() - 트래픽 매니저를 초기 상태로 리셋한다
   *
   * _traffic_manager->Init()을 호출하여 시뮬레이션 시간(_time=0), 상태(running),
   * 통계 초기화를 수행한다. 커널 실행 시작 전에 호출되어야 한다.
   *
   * 호출 체인: icnt_init() → [이 함수] → GPUTrafficManager::Init()
   */
  virtual void Init();

  /*
   * [한국어]
   * Push() - 메모리 요청/응답 패킷을 NoC에 삽입한다
   *
   * @input_deviceID: 송신자 노드 ID (SM 또는 메모리 파티션)
   * @output_deviceID: 수신자 노드 ID
   * @data: 전송할 데이터 포인터 (실제로는 mem_fetch* 타입)
   * @size: 데이터 크기 (바이트 단위, flit 분할에 사용)
   *
   * 동작 과정:
   *   1. HasBuffer()로 입력 버퍼 여유 확인 (실패 시 assert)
   *   2. deviceID를 icntID로 변환
   *   3. 패킷 크기를 flit 수로 변환 (ceil(size / flit_size))
   *   4. 서브넷 결정 (단일망 또는 SM→메모리/메모리→SM 이중망)
   *   5. mem_fetch 타입에서 Flit::FlitType 결정
   *   6. GPUTrafficManager::_GeneratePacket()으로 Flit들을 생성하여 큐에 삽입
   *
   * 주의: data는 내부적으로 mem_fetch*로 캐스팅됨. flit에는 포인터만 저장.
   *
   * 호출 체인: shader_core_ctx::cycle() → mem_fetch 생성 → icnt_push()
   *            → [이 함수] → _GeneratePacket()
   */
  virtual void Push(unsigned input_deviceID, unsigned output_deviceID, void* data, unsigned int size);

  /*
   * [한국어]
   * Pop() - 수신 완료된 패킷(mem_fetch 포인터)을 꺼낸다
   *
   * @ouput_deviceID: 수신자 노드 ID (Pop을 요청하는 노드)
   * @return: 완료된 패킷의 데이터 포인터 (mem_fetch*), 없으면 NULL
   *
   * 동작 과정:
   *   1. deviceID → icntID 변환
   *   2. SM이면 subnet=1 (응답 네트워크), 메모리이면 subnet=0 (요청 네트워크)
   *   3. 라운드-로빈으로 VC를 순회하면서 완성된 패킷을 _boundary_buffer에서 꺼냄
   *   4. 패킷을 찾으면 다음 호출을 위해 _round_robin_turn 업데이트
   *
   * 호출 체인: memory_partition_unit::cycle() → icnt_pop() → [이 함수]
   *   또는: shader_core_ctx::cycle() → icnt_pop() → [이 함수]
   */
  virtual void* Pop(unsigned ouput_deviceID);

  /*
   * [한국어]
   * Advance() - NoC 시뮬레이션을 1사이클 진행한다
   *
   * GPUTrafficManager::_Step()을 호출하여 모든 라우터의 파이프라인 단계를 실행한다.
   * GPGPU-Sim의 gpu-sim cycle()이 매 시뮬레이션 사이클마다 이 함수를 호출한다.
   * NoC 주파수가 코어 주파수보다 낮으면 여러 코어 사이클마다 1번 호출될 수 있다.
   *
   * 호출 체인: gpgpu_sim::cycle() → icnt_transfer() → [이 함수] → GPUTrafficManager::_Step()
   */
  virtual void Advance();

  /*
   * [한국어]
   * Busy() - NoC 내에 아직 처리 중인 패킷이 있는지 확인한다
   *
   * @return: true = in-flight flit이나 boundary_buffer에 잔류 패킷이 있음
   *
   * 시뮬레이션 종료 조건 판단에 사용. in-flight flit이 없어도
   * _boundary_buffer에 미처리 패킷이 남아있으면 true를 반환한다.
   *
   * 호출 체인: gpgpu_sim::active() → [이 함수]
   */
  virtual bool Busy() const;

  /*
   * [한국어]
   * HasBuffer() - 지정 노드에 패킷을 삽입할 여유 버퍼가 있는지 확인한다
   *
   * @deviceID: 송신자 노드 ID
   * @size: 삽입하려는 패킷 크기 (바이트)
   * @return: true = 충분한 입력 버퍼 공간 있음
   *
   * _input_queue의 현재 flit 수 + 새로 삽입할 flit 수가
   * _input_buffer_capacity를 초과하지 않아야 true를 반환한다.
   * 이를 통해 NoC 입구에서 backpressure를 구현한다.
   *
   * 호출 체인: shader_core_ctx가 Push 전에 [이 함수] 호출 → Push()
   */
  virtual bool HasBuffer(unsigned deviceID, unsigned int size) const;

  /*
   * [한국어]
   * DisplayStats() - 현재 시뮬레이션 구간의 통계를 출력한다
   *
   * UpdateStats()로 통계 집계 후 DisplayStats()로 출력.
   * 각 커널 실행 완료 시 호출되어 latency, throughput 등의 NoC 성능 지표를 보고.
   *
   * 호출 체인: gpgpu_sim::gpu_print_stat() → icnt_display_stats() → [이 함수]
   */
  virtual void DisplayStats() const;

  /*
   * [한국어]
   * DisplayOverallStats() - 전체 시뮬레이션 통합 통계를 출력한다
   *
   * 모든 커널 실행에 걸친 누적 통계. _drain_time과 _total_sims를 설정하여
   * BookSim 내부의 overall stats 계산 방식(drain 기반 delta 시간)을 우회한다.
   *
   * 호출 체인: gpgpu_sim::gpu_print_stat() 마지막에 [이 함수] 호출
   */
  virtual void DisplayOverallStats() const;

  /*
   * [한국어]
   * GetFlitSize() - 설정된 flit 크기(바이트)를 반환한다
   *
   * @return: _flit_size (gpgpusim.config의 flit_size 설정값)
   *
   * icnt_wrapper.cc에서 flit 크기를 조회하여 mem_fetch를 분할할 때 사용.
   *
   * 호출 체인: icnt_get_flit_size() → [이 함수]
   */
  unsigned GetFlitSize() const;

  /*
   * [한국어]
   * DisplayState() - 현재 NoC 상태를 파일에 출력한다 (미구현)
   *
   * @fp: 출력 대상 FILE 포인터
   *
   * 현재는 "Under implementation" 메시지만 출력. 향후 상세 상태 덤프 예정.
   *
   * 호출 체인: gpgpu_sim::print_stats() → [이 함수]
   */
  virtual void DisplayState(FILE* fp) const;

  // =====================================================================
  // [한국어] BookSim 측 함수 (GPUTrafficManager::_Step()이 호출하는 콜백)
  // =====================================================================

  //booksim side functions
  /*
   * [한국어]
   * WriteOutBuffer() - BookSim 네트워크에서 배출된 flit을 ejection_buffer에 저장한다
   *
   * @subnet: 서브넷 인덱스
   * @output: 목적지 노드 icntID
   * @flit: 네트워크에서 배출된 Flit 포인터
   *
   * GPUTrafficManager::_Step()이 Network::ReadFlit()으로 배출 flit을 받아
   * 이 함수를 통해 InterconnectInterface의 _ejection_buffer에 저장한다.
   * _ejection_buffer_capacity를 초과하면 assert로 오류 처리.
   *
   * 호출 체인: GPUTrafficManager::_Step() → [이 함수] → _ejection_buffer에 push
   */
  void WriteOutBuffer( int subnet, int output, Flit* flit );

  /*
   * [한국어]
   * Transfer2BoundaryBuffer() - ejection_buffer의 flit을 boundary_buffer로 이동한다
   *
   * @subnet: 서브넷 인덱스
   * @output: 목적지 노드 icntID
   *
   * 각 VC의 ejection_buffer에서 flit을 꺼내어 _BoundaryBufferItem에 PushFlitData()로 삽입.
   * tail flit이 도착할 때 _packet_n이 증가하여 팻킷 완성을 표시한다.
   * 크레딧 반환을 위해 이동한 flit을 _ejected_flit_queue에도 추가한다.
   *
   * 호출 체인: GPUTrafficManager::_Step() → [이 함수] → _BoundaryBufferItem::PushFlitData()
   */
  void Transfer2BoundaryBuffer(int subnet, int output);

  /*
   * [한국어]
   * GetIcntTime() - 현재 NoC 시뮬레이션 시간(사이클)을 반환한다
   *
   * @return: _traffic_manager->getTime() (NoC 사이클 카운터)
   *
   * 호출 체인: 통계 출력 코드 → [이 함수]
   */
  int GetIcntTime() const;

  /*
   * [한국어]
   * GetIcntStats() - 이름으로 BookSim 통계 객체를 반환한다
   *
   * @name: 통계 이름 문자열 (예: "plat", "nlat", "flit_lat")
   * @return: 해당 Stats 포인터 (없으면 NULL)
   *
   * 호출 체인: 외부 분석 코드 → [이 함수]
   */
  Stats* GetIcntStats(const string & name) const;

  /*
   * [한국어]
   * GetEjectedFlit() - _ejected_flit_queue에서 크레딧 반환용 flit을 꺼낸다
   *
   * @subnet: 서브넷 인덱스
   * @node: 노드 icntID
   * @return: 배출된 Flit 포인터 (없으면 NULL)
   *
   * Transfer2BoundaryBuffer()가 _ejected_flit_queue에 넣어둔 flit을
   * GPUTrafficManager::_Step()이 이 함수로 꺼내어 크레딧을 반환한다.
   * 크레딧 반환은 상류 라우터의 버퍼 공간을 복원하여 흐름 제어를 구현한다.
   *
   * 호출 체인: GPUTrafficManager::_Step() → [이 함수] → Credit 반환
   */
  Flit* GetEjectedFlit(int subnet, int node);

protected:

  /*
   * [한국어]
   * _BoundaryBufferItem - Flit들을 모아 패킷 단위로 Pop할 수 있게 하는 내부 버퍼
   *
   * NoC에서 배출된 flit들은 VC별로 도착 순서가 보장되지 않을 수 있으므로,
   * 이 클래스가 flit들을 모아서 tail flit이 도착할 때 패킷 완성을 표시한다.
   * Pop()은 완성된 패킷(tail flit의 data 포인터 = mem_fetch*)만을 반환한다.
   *
   * 동기화: 단일 스레드 시뮬레이션에서 사용되므로 별도 락 불필요.
   */
  class _BoundaryBufferItem {
  public:
    /*
     * [한국어]
     * _BoundaryBufferItem() - 패킷 카운터를 0으로 초기화하는 생성자
     */
    _BoundaryBufferItem():_packet_n(0) {}

    /*
     * [한국어]
     * Size() - 버퍼에 저장된 flit 수를 반환한다
     *
     * @return: _buffer.size() (flit 단위, 패킷 단위가 아님)
     *
     * Transfer2BoundaryBuffer에서 _boundary_buffer_capacity와 비교하여 흐름 제어에 사용.
     */
    inline unsigned Size(void) const { return _buffer.size(); }

    /*
     * [한국어]
     * HasPacket() - 완성된 패킷이 하나 이상 있는지 반환한다
     *
     * @return: _packet_n > 0 이면 true (tail flit이 도착하여 완성된 패킷 존재)
     *
     * Pop() 호출 전에 확인하여 데이터 없는 Pop을 방지.
     */
    inline bool HasPacket() const { return _packet_n; }

    /*
     * [한국어]
     * PopPacket() - 완성된 패킷의 데이터 포인터를 꺼낸다
     *
     * @return: tail flit의 data 포인터 (= mem_fetch* 포인터)
     *
     * 동작: _buffer와 _tail_flag를 동시에 순회하면서 tail이 나올 때까지
     * flit들을 버린다. tail flit의 data 포인터가 패킷 전체를 대표한다.
     * (_BoundaryBufferItem 전체에 걸쳐 같은 data 포인터를 공유하는 설계)
     *
     * 호출 체인: InterconnectInterface::Pop() → [이 함수]
     */
    void* PopPacket();

    /*
     * [한국어]
     * TopPacket() - 버퍼 맨 앞 패킷의 데이터를 pop 없이 조회한다 (peek)
     *
     * @return: 완성된 패킷의 data 포인터
     *
     * 현재 코드에서는 사실상 사용되지 않지만, 미래 확장을 위해 선언됨.
     * 무한루프 가능성이 있는 구현 버그가 있으므로 주의 (while data==NULL).
     */
    void* TopPacket() const;

    /*
     * [한국어]
     * PushFlitData() - flit 데이터와 tail 여부를 버퍼에 추가한다
     *
     * @data: flit이 가진 mem_fetch* 포인터 (void*로 전달)
     * @is_tail: 이 flit이 패킷의 마지막 flit인지 여부
     *
     * tail이면 _packet_n을 증가시켜 패킷 완성을 알린다.
     *
     * 호출 체인: Transfer2BoundaryBuffer() → [이 함수]
     */
    void PushFlitData(void* data,bool is_tail);

  private:
    queue<void *> _buffer;
    /* [한국어] flit 데이터 포인터를 순서대로 저장하는 큐.
     * 설정자: PushFlitData()가 flit.data를 순서대로 push.
     * 읽는 자: PopPacket()이 tail까지 pop하면서 반환값을 찾음.
     * 값 범위: mem_fetch* 포인터 (비교: 같은 패킷의 모든 flit은 동일 data 포인터 공유).
     * 동기화: 단일 스레드 전용, 락 불필요. */

    queue<bool> _tail_flag;
    /* [한국어] _buffer의 각 항목이 tail flit인지를 1:1 대응하여 저장하는 큐.
     * 설정자: PushFlitData()가 is_tail 값을 순서대로 push.
     * 읽는 자: PopPacket()이 tail 위치를 판별하는 데 사용.
     * 값 범위: true(tail flit), false(head 또는 body flit).
     * 동기화: _buffer와 항상 같이 조작되므로 size()가 동일해야 함. */

    int _packet_n;
    /* [한국어] 완성된 패킷(tail flit 도착 완료) 수를 기록하는 카운터.
     * 설정자: PushFlitData()에서 is_tail==true일 때 ++.
     * 읽는 자: HasPacket()이 0 여부 확인, PopPacket()이 -- 후 반환.
     * 값 범위: 0 이상 정수; 0이면 완성된 패킷 없음.
     * 동기화: 단일 스레드 전용. */
  };

  // [한국어] ejection 버퍼 항목 타입: VC별로 배출된 Flit을 임시 저장하는 FIFO 큐
  typedef queue<Flit*> _EjectionBufferItem;

  /*
   * [한국어]
   * _CreateBuffer() - 모든 서브넷/노드/VC 조합에 대해 버퍼를 할당한다
   *
   * _boundary_buffer[subnets][nodes][vcs]
   * _ejection_buffer[subnets][nodes][vcs]
   * _ejected_flit_queue[subnets][nodes]
   * _round_robin_turn[subnets][nodes] 형태로 3차원 벡터를 초기화한다.
   *
   * 호출 체인: CreateInterconnect() → [이 함수]
   */
  void _CreateBuffer( );

  /*
   * [한국어]
   * _CreateNodeMap() - deviceID와 icntID 사이의 양방향 매핑을 생성한다
   *
   * @n_shader: SM 수
   * @n_mem: 메모리 파티션 수
   * @n_node: 전체 NoC 노드 수 (Network::NumNodes())
   * @use_map: 1이면 사전 정의된 최적 매핑 사용, 0이면 항등 매핑
   *
   * use_map=1이면 SM과 메모리 노드를 메시 위에 물리적으로 분산 배치하는
   * 사전 정의된 매핑(8SM/8메모리, 28SM/8메모리 등)을 사용한다.
   * 이 배치는 SM-메모리 간 평균 홉 수를 최소화하기 위한 최적화이다.
   *
   * 호출 체인: CreateInterconnect() → [이 함수] → _DisplayMap()
   */
  void _CreateNodeMap(unsigned n_shader, unsigned n_mem, unsigned n_node, int use_map);

  /*
   * [한국어]
   * _DisplayMap() - 노드 매핑 테이블을 메시 형태로 콘솔에 출력한다
   *
   * @dim: 메시의 한 변 크기 (sqrt(n_node))
   * @count: 전체 노드 수
   *
   * 시뮬레이션 시작 시 deviceID→icntID, icntID→deviceID 매핑을 시각적으로 출력.
   * 메시가 정사각형이 아닌 경우 올바르지 않을 수 있음(FIXME 주석 있음).
   *
   * 호출 체인: _CreateNodeMap() → [이 함수]
   */
  void _DisplayMap(int dim,int count);

  // size: [subnets][nodes][vcs]
  vector<vector<vector<_BoundaryBufferItem> > > _boundary_buffer;
  /* [한국어] 노드별/VC별 경계 버퍼: ejection_buffer에서 이동된 flit들을 패킷 단위로 조립.
   * 설정자: Transfer2BoundaryBuffer()가 flit 데이터를 PushFlitData()로 추가.
   * 읽는 자: Pop()이 HasPacket()/PopPacket()으로 완성 패킷 추출.
   * 값 범위: 각 _BoundaryBufferItem은 flit 수 기준으로 _boundary_buffer_capacity 제한.
   * 동기화: 단일 스레드 시뮬레이션 루프 내에서만 접근. */

  unsigned int _boundary_buffer_capacity;
  /* [한국어] boundary_buffer 항목당 최대 flit 수 (gpgpusim.config의 boundary_buffer_size).
   * 설정자: CreateInterconnect()에서 설정 파일 읽어 초기화.
   * 읽는 자: Transfer2BoundaryBuffer()가 삽입 가능 여부 확인에 사용.
   * 값 범위: 양의 정수, 0이면 assert 실패.
   * 동기화: 초기화 후 읽기 전용. */

  // size: [subnets][nodes][vcs]
  vector<vector<vector<_EjectionBufferItem> > > _ejection_buffer;
  /* [한국어] NoC에서 배출된 flit을 VC별로 임시 저장하는 ejection 버퍼.
   * 설정자: WriteOutBuffer()가 Network::ReadFlit() 결과를 push.
   * 읽는 자: Transfer2BoundaryBuffer()가 pop하여 boundary_buffer로 이동.
   * 값 범위: 항목 수 < _ejection_buffer_capacity (초과 시 assert 실패).
   * 동기화: 단일 스레드 전용. */

  // size:[subnets][nodes]
  vector<vector<queue<Flit* > > > _ejected_flit_queue;
  /* [한국어] boundary_buffer로 이동 완료된 flit들의 크레딧 반환 대기 큐.
   * Transfer2BoundaryBuffer()가 push, GetEjectedFlit()이 pop.
   * GPUTrafficManager::_Step()이 GetEjectedFlit()으로 꺼내어 upstream에 크레딧 반환.
   * 크레딧 반환은 상류 라우터의 버퍼 공간을 복원하여 흐름 제어를 구현한다. */

  unsigned int _ejection_buffer_capacity;
  /* [한국어] ejection_buffer 항목당 최대 flit 수 (gpgpusim.config의 ejection_buffer_size 또는 vc_buf_size).
   * WriteOutBuffer()에서 이 용량 초과 시 assert 실패. */

  unsigned int _input_buffer_capacity;
  /* [한국어] 노드별 입력 큐(_input_queue)의 최대 flit 수 (input_buffer_size 또는 기본값 9).
   * HasBuffer()가 이 값과 비교하여 backpressure를 구현한다. */

  vector<vector<int> > _round_robin_turn;
  /* [한국어] Pop() 시 VC를 공정하게 순환 선택하기 위한 라운드-로빈 상태 변수.
   * 크기: [subnets][nodes]. 각 원소는 마지막으로 선택된 VC 인덱스+1을 저장.
   * 설정자: Pop()이 데이터를 찾았을 때 업데이트.
   * 읽는 자: Pop()이 다음 순회 시작 VC를 결정하는 데 사용.
   * 값 범위: 0 .. _vcs-1.
   * 동기화: 단일 스레드 전용. */

  GPUTrafficManager* _traffic_manager;
  /* [한국어] BookSim2 NoC 사이클 시뮬레이션을 수행하는 트래픽 매니저 포인터.
   * 설정자: CreateInterconnect()에서 TrafficManager::New()로 생성 및 downcast.
   * 읽는 자: Advance(), DisplayStats(), Busy() 등 대부분의 공개 API.
   * 값 범위: GPUTrafficManager 인스턴스 포인터 (NULL이면 미초기화).
   * 동기화: 생성 후 단일 스레드에서 접근. */

  unsigned _flit_size;
  /* [한국어] flit 하나의 크기 (바이트 단위), gpgpusim.config의 flit_size 설정.
   * Push()에서 패킷을 몇 개 flit으로 분할할지 계산할 때 사용: ceil(size/flit_size). */

  IntersimConfig* _icnt_config;
  /* [한국어] BookSim 설정 파일 파서 결과를 담는 설정 객체.
   * New()에서 ParseFile()로 초기화되고, CreateInterconnect()에서 각종 설정값을 읽는다.
   * 소멸자에서 delete로 해제된다. */

  unsigned _n_shader;
  /* [한국어] SM 클러스터 수. 서브넷 분리 로직(Push, Pop)에서 SM/메모리 구분에 사용.
   * deviceID < _n_shader 이면 SM 노드, 그 이상이면 메모리 노드. */

  unsigned _n_mem;
  /* [한국어] 메모리 파티션 수. 노드 매핑 생성 및 Busy() 순회 범위 결정에 사용. */

  vector<Network *> _net;
  /* [한국어] 서브넷별 BookSim Network 객체 포인터 벡터. 크기: _subnets.
   * 각 Network는 라우터들의 연결 구조(mesh, torus 등)를 가진다.
   * CreateInterconnect()에서 Network::New()로 생성, 소멸자에서 delete. */

  int _vcs;
  /* [한국어] VC(Virtual Channel) 수. 설정 파일의 num_vcs.
   * 버퍼 할당 크기, Pop()의 라운드-로빈 순회 범위 등에 사용. */

  int _subnets;
  /* [한국어] 서브넷 수 (1=단일망, 2=요청/응답 분리망).
   * 설정 파일의 subnets 값. 데드락 방지를 위해 2-서브넷 구성을 권장. */

  //deviceID to icntID map
  //deviceID : Starts from 0 for shaders and then continues until mem nodes
  //which starts at location n_shader and then continues to n_shader+n_mem (last device)
  // [한국어] deviceID → icntID 순방향 매핑 테이블
  // deviceID: SM은 0부터 n_shader-1, 메모리는 n_shader부터 n_shader+n_mem-1
  // icntID: BookSim 내부 노드 번호 (use_map=1이면 물리적 위치 최적화됨)
  map<unsigned, unsigned> _node_map;
  /* [한국어] GPU device ID에서 BookSim 내부 icnt ID로의 변환 테이블.
   * 설정자: _CreateNodeMap()이 초기화.
   * 읽는 자: Push(), Pop(), HasBuffer()가 deviceID를 icntID로 변환.
   * 값 범위: key=0..n_shader+n_mem-1, value=0..n_node-1.
   * 동기화: 초기화 후 읽기 전용. */

  //icntID to deviceID map
  // [한국어] icntID → deviceID 역방향 매핑 테이블
  map<unsigned, unsigned> _reverse_node_map;
  /* [한국어] BookSim 내부 icnt ID에서 GPU device ID로의 역변환 테이블.
   * 설정자: _CreateNodeMap()이 _node_map의 역을 계산하여 초기화.
   * 읽는 자: 주로 디버그 및 DisplayMap에서 사용.
   * 값 범위: key=0..n_node-1, value=0..n_shader+n_mem-1. */

};

#endif


