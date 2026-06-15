// $Id: kncube.cpp 5516 2013-10-06 02:14:48Z dub $

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

/*kn.cpp
 *
 *Meshs, cube, torus
 *
 */

/*
 * [한국어 설명] k진 n-큐브(KNCube) 네트워크 토폴로지 구현 (kncube.cpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 intersim2(Booksim 기반 GPU NoC 시뮬레이터)에서 k-ary n-cube 토폴로지를
 * 구현한다. k-ary n-cube란 n개의 차원(dimension)을 가지며 각 차원에 k개의 라우터가
 * 일렬로 배열된 격자 구조이다. _mesh=true이면 각 차원의 끝 노드 간 링크가 없는
 * 메시(mesh), _mesh=false이면 끝 노드끼리 wrap-around 링크로 연결되는 토러스(torus)가 된다.
 * GPU NoC에서는 2D 메시(n=2, _mesh=true)가 가장 널리 쓰인다.
 * 이 파일은 라우터 생성, 채널 연결, 레이턴시 설정, 인접 노드/채널 인덱스 계산,
 * 그리고 레거시 랜덤 장애 삽입 기능을 모두 구현한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 NoC 계층(intersim2)에서 실제 물리 토폴로지를 담당한다.
 * 호출 체인 (위→아래):
 *   gpgpu_sim::cycle() [gpu-sim.cc]
 *   → icnt_push / icnt_pop [icnt_wrapper.cc]
 *   → LocalInterconnect / intersim2 Network 구동
 *   → KNCube 생성자(_ComputeSize → _Alloc → _BuildNet) [초기화 시 1회]
 *   → 매 사이클: Network::ReadInputs / Evaluate / WriteOutputs
 *     (KNCube 자체는 이 함수를 오버라이드하지 않고 Network 기반 클래스의 _timed_modules
 *      루프를 통해 _routers[i]와 _chan[i]가 간접적으로 업데이트됨)
 * 실행 컨텍스트: 시뮬레이터 메인 스레드(단일 스레드 진행). 사이클-레벨 타이밍 모델.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - booksim.hpp: 공통 매크로/타입 (Error, gK, gN 전역 변수)
 *   - network.hpp (Network 기반 클래스): _size/_nodes/_channels, _routers/_chan/_inject/_eject,
 *     _Alloc(), OutChannelFault(), AddInputChannel/AddOutputChannel 등
 *   - random_utils.hpp: RandomInt(), RandomSeed(), RandomIntLong() — InsertRandomFaults 사용
 *   - misc_utils.hpp: powi(k, n) — k^n 계산
 *   - Router::NewRouter(): 각 노드의 라우터 인스턴스 생성 (iq_router 등)
 *   - Configuration: gpgpusim.config의 "k", "n", "use_noc_latency",
 *                    "link_failures", "fail_seed" 옵션 소비
 * 이 모듈에 의존하는 모듈:
 *   - icnt_wrapper.cc: Network* 타입으로 KNCube 인스턴스를 보유·구동
 *   - local_interconnect.cc: LocalInterconnect 내부 망으로 사용 가능
 * 데이터 흐름:
 *   SM/L2 플릿 → _inject[src](FlitChannel) → 라우터 → _chan[i](내부 링크)
 *   → 라우터 → _eject[dst](FlitChannel) → SM/L2
 *
 * === 주요 함수/구조체 요약 ===
 * - KNCube(): 생성자. mesh/torus 선택, 크기 계산, 자원 할당, 네트워크 구축 순으로 초기화
 * - _ComputeSize(): "k"/"n" 설정 읽기, _size=k^n, _channels=2*n*size, gK/gN 전역 설정
 * - _BuildNet(): 모든 라우터 생성, 각 차원의 좌/우 채널 연결, use_noc_latency에 따른 레이턴시 설정
 * - _LeftChannel/_RightChannel(): 채널 인덱스 계산 (2*n*node + 2*dim+1 / 2*dim)
 * - _LeftNode/_RightNode(): 인접 노드 번호 계산, 차원 경계에서 wrap-around 처리
 * - GetN()/GetK(): _n/_k 읽기 전용 접근자
 * - Capacity(): 이론적 처리 용량 (mesh: k/8.0, torus: k/4.0)
 * - InsertRandomFaults(): 레거시 랜덤 링크 장애 삽입 (현재 GPU 시뮬에서 미사용)
 */

#include "booksim.hpp"   // [한국어] intersim2 공통 헤더: Error() 매크로, gK/gN 전역 변수 등 기반 선언
#include <vector>        // [한국어] std::vector — InsertRandomFaults의 fail_nodes 배열 등에 사용
#include <sstream>       // [한국어] std::ostringstream — _BuildNet에서 라우터 이름 문자열 조립에 사용
#include "kncube.hpp"    // [한국어] KNCube 클래스 선언 헤더 (이 파일에서 구현)
#include "random_utils.hpp"  // [한국어] RandomInt/RandomSeed/RandomIntLong — InsertRandomFaults 난수 생성
#include "misc_utils.hpp"    // [한국어] powi(x, y) — 정수 거듭제곱(k^n) 계산
 //#include "iq_router.hpp"  // [한국어] (주석 처리됨) IQRouter 직접 의존이 필요할 때의 흔적


/*
 * [한국어]
 * KNCube 생성자 - k진 n-큐브 네트워크 객체를 초기화하고 전체 토폴로지를 구축한다.
 *
 * @config: gpgpusim.config에서 읽은 설정 객체. "k", "n", "use_noc_latency" 등 포함.
 * @name:   이 네트워크 인스턴스의 이름 문자열 (디버그/로그 출력용).
 * @mesh:   true → 메시(끝단 연결 없음), false → 토러스(wrap-around 연결).
 * @return: 없음 (생성자).
 *
 * Network 기반 생성자(Network(config, name))를 먼저 호출하여 기반 자원을 준비한 뒤,
 * 1) _mesh 플래그 설정 → 2) _ComputeSize(config)로 _size/_channels 계산 →
 * 3) _Alloc()으로 _routers/_chan 등 배열 동적 할당 →
 * 4) _BuildNet(config)으로 실제 라우터 생성 및 채널 연결.
 * 이 생성자 이후에는 네트워크가 완전히 사용 가능한 상태가 된다.
 *
 * 실행 컨텍스트: 시뮬레이터 초기화 단계(1회 호출). 단일 스레드.
 *
 * 호출 체인:
 *   Network::NewNetwork() 또는 icnt_wrapper_init() → [KNCube()]
 *   → Network(config, name) [기반 클래스 생성자]
 *   → _ComputeSize(config)
 *   → _Alloc()       [Network 기반 클래스]
 *   → _BuildNet(config)
 */
KNCube::KNCube( const Configuration &config, const string & name, bool mesh ) :
Network( config, name ) // [한국어] Network 기반 클래스 생성자 호출: _size/_nodes/_channels를 -1로 초기화하고 _classes 등 공통 설정을 준비
{
  _mesh = mesh; // [한국어] 메시(true) 또는 토러스(false) 토폴로지 선택: 이후 _BuildNet의 레이턴시 결정 및 Capacity()에서 참조

  _ComputeSize( config ); // [한국어] "k"/"n" 옵션으로 _size=k^n, _channels=2*n*size, _nodes=_size, gK/gN 전역 변수 설정
  _Alloc( );              // [한국어] _size/_nodes/_channels에 맞춰 _routers/_chan/_inject/_eject 배열을 동적 할당하고 FlitChannel/CreditChannel 객체를 생성
  _BuildNet( config );    // [한국어] 모든 라우터 인스턴스를 생성하고 채널로 상호 연결하여 k-ary n-cube 토폴로지를 완성
}

/*
 * [한국어]
 * _ComputeSize - gpgpusim.config에서 k와 n을 읽어 네트워크 규모를 계산한다.
 *
 * @config: GPGPU-Sim 설정 객체. "k"(각 차원당 라우터 수), "n"(차원 수) 옵션 사용.
 * @return: 없음. Network 기반 클래스 멤버(_size, _nodes, _channels)와 전역 gK/gN을 직접 설정.
 *
 * k-ary n-cube의 토폴로지 매개변수를 결정하고 네트워크 자원 크기를 확정한다.
 * - _size = k^n: 전체 라우터(노드) 수. 예) 4-ary 2-cube → 4^2=16 라우터.
 * - _channels = 2*n*_size: 라우터 간 내부 채널 총수. 각 노드마다 n개 차원 × 좌/우 2개 = 2*n개 채널.
 * - _nodes = _size: 엔드포인트(주입/이젝션 포인트) 수 = 라우터 수와 동일.
 * 전역 gK/gN에도 복사하여 intersim2 라우팅 함수(xy routing 등)가 전역으로 참조 가능하게 한다.
 *
 * 실행 컨텍스트: KNCube 생성자에서 1회 호출. 단일 스레드.
 *
 * 호출 체인:
 *   KNCube() → [_ComputeSize()] → _Alloc() → _BuildNet()
 */
void KNCube::_ComputeSize( const Configuration &config )
{
  _k = config.GetInt( "k" ); // [한국어] 각 차원당 라우터 수(radix) 설정 파일에서 읽기. 예) 4-ary이면 _k=4
  _n = config.GetInt( "n" ); // [한국어] 차원 수(dimension count) 설정 파일에서 읽기. 예) 2D 메시이면 _n=2

  gK = _k; gN = _n;           // [한국어] intersim2 전역 변수 gK/gN에 동일 값 복사: 라우팅 함수(xy_yx_routing 등)가 전역 gK/gN을 직접 참조하기 때문에 필요
  _size     = powi( _k, _n ); // [한국어] 전체 라우터 수 = k^n (misc_utils.hpp의 powi로 정수 거듭제곱 계산). 예) k=4, n=2 → 16개 라우터
  _channels = 2*_n*_size;     // [한국어] 내부 채널 총수 = 각 라우터마다 n차원 × 좌/우 2방향: 16라우터×2차원×2=64개 채널

  _nodes = _size; // [한국어] 엔드포인트(SM/L2 연결 포인트) 수 = 라우터 수와 동일. _inject/_eject 배열 크기 결정
}

/*
 * [한국어]
 * RegisterRoutingFunctions - KNCube 전용 라우팅 함수를 전역 라우팅 테이블에 등록한다.
 *
 * @return: 없음.
 *
 * 현재 구현은 빈 함수(empty body)이다.
 * k-ary n-cube용 XY 라우팅, 차원 순서 라우팅 등이 등록될 예정이었으나,
 * 실제 라우팅 함수 등록은 별도의 routing_function 파일(kn_routing.cpp 등)에서
 * 수행되거나, Network 기반 클래스의 글로벌 함수 맵에 직접 등록되는 방식으로 분리된 것으로 보인다.
 *
 * 실행 컨텍스트: 시뮬레이터 초기화 시 Network 초기화 루틴에서 호출 가능.
 *
 * 호출 체인:
 *   booksim_main / Network 초기화 → [RegisterRoutingFunctions()]
 */
void KNCube::RegisterRoutingFunctions() {
  // [한국어] 현재 비어 있음: k-ary n-cube 전용 라우팅 함수가 여기에 등록될 예정이나
  // intersim2 구조에서 라우팅 함수는 별도 파일(routing 디렉터리)에 분산되어 있음
}

/*
 * [한국어]
 * _BuildNet - 모든 라우터 인스턴스를 생성하고 채널로 연결하여 k-ary n-cube 토폴로지를 완성한다.
 *
 * @config: GPGPU-Sim 설정 객체. "use_noc_latency" 옵션이 채널 레이턴시를 결정.
 * @return: 없음. _routers[], _chan[], _inject[], _eject[], _timed_modules 등을 채운다.
 *
 * _ComputeSize()와 _Alloc() 이후 호출된다.
 * 각 노드(node = 0 ~ _size-1)에 대해:
 *   1) 라우터 이름 조립: "router_<dim0좌표>_<dim1좌표>..." 형식.
 *   2) Router::NewRouter()로 라우터 인스턴스 생성. 포트 수 = 2*_n+1 (n차원 × 좌/우 + 인젝션).
 *   3) 각 차원(dim)에 대해:
 *      - 왼쪽 이웃(_LeftNode), 오른쪽 이웃(_RightNode) 계산.
 *      - 입력 채널: 오른쪽 이웃의 왼쪽 채널(_LeftChannel(right_node, dim))과
 *                   왼쪽 이웃의 오른쪽 채널(_RightChannel(left_node, dim)).
 *      - 출력 채널: 자신의 오른쪽 채널(_RightChannel(node, dim))과 왼쪽 채널(_LeftChannel(node, dim)).
 *   4) use_noc_latency=true이면 latency = _mesh ? 1 : 2 (torus wrap-around는 배선 지연이 더 큼).
 *      use_noc_latency=false이면 모든 채널 latency=1 (단순화).
 *   5) 인젝션/이젝션 채널(_inject, _eject)은 항상 latency=1.
 *
 * 채널 연결 방향 다이어그램:
 *   (L)eft node ---left_input(RightChannel of L)---> (N)ode <---right_input(LeftChannel of R)--- (R)ight node
 *   (L)eft node <--left_output(LeftChannel of N)--- (N)ode ---right_output(RightChannel of N)---> (R)ight node
 *
 * 실행 컨텍스트: 생성자에서 1회 호출. 단일 스레드.
 *
 * 호출 체인:
 *   KNCube() → _ComputeSize() → _Alloc() → [_BuildNet()]
 *   → Router::NewRouter() [각 노드마다]
 *   → _LeftNode(), _RightNode() [각 차원마다]
 *   → _LeftChannel(), _RightChannel() [채널 인덱스 계산]
 *   → Router::AddInputChannel() / AddOutputChannel() [채널 연결]
 *   → FlitChannel::SetLatency() / CreditChannel::SetLatency() [레이턴시 설정]
 */
void KNCube::_BuildNet( const Configuration &config )
{
  int left_node;   // [한국어] 현재 노드의 dim 차원에서 왼쪽 인접 라우터 번호
  int right_node;  // [한국어] 현재 노드의 dim 차원에서 오른쪽 인접 라우터 번호

  int right_input; // [한국어] 오른쪽 이웃 → 현재 노드로 들어오는 입력 채널 인덱스 (오른쪽 이웃의 왼쪽 채널)
  int left_input;  // [한국어] 왼쪽 이웃 → 현재 노드로 들어오는 입력 채널 인덱스 (왼쪽 이웃의 오른쪽 채널)

  int right_output; // [한국어] 현재 노드 → 오른쪽 이웃 방향 출력 채널 인덱스 (자신의 오른쪽 채널)
  int left_output;  // [한국어] 현재 노드 → 왼쪽 이웃 방향 출력 채널 인덱스 (자신의 왼쪽 채널)

  ostringstream router_name; // [한국어] 라우터 이름 문자열을 조립하기 위한 스트림. 예) "router_0_1" (2D, node=4, k=4)

  //latency type, noc or conventional network
  // [한국어] use_noc_latency: NoC 레이턴시 모드(true)이면 torus 링크에 latency=2, mesh에 latency=1 설정;
  //          false이면 모든 채널에 latency=1 (기존 전통적 네트워크 모드)
  bool use_noc_latency;
  use_noc_latency = (config.GetInt("use_noc_latency")==1); // [한국어] gpgpusim.config의 "use_noc_latency" 정수 옵션을 bool로 변환 (1이면 true)

  for ( int node = 0; node < _size; ++node ) { // [한국어] 모든 라우터 노드를 순차적으로 처리 (node=0 ~ _size-1 = k^n-1)

    router_name << "router"; // [한국어] 라우터 이름 문자열 시작: "router"

    if ( _k > 1 ) { // [한국어] k=1이면 단일 노드(trivial 토폴로지)이므로 좌표 접미사 불필요; k>1일 때만 각 차원 좌표를 이름에 추가
      for ( int dim_offset = _size / _k; dim_offset >= 1; dim_offset /= _k ) {
        // [한국어] 상위 차원부터 하위 차원 순서로 좌표 계산:
        //   dim_offset은 _size/_k(_n-1차원 stride)에서 시작하여 _k씩 나눠 가며 각 차원 stride를 순환.
        //   (node / dim_offset) % _k 로 해당 차원에서의 좌표를 추출.
        //   예) k=4, n=2, node=5: dim_offset=4 → (5/4)%4=1, dim_offset=1 → (5/1)%4=1 → "router_1_1"
	router_name << "_" << ( node / dim_offset ) % _k; // [한국어] 해당 차원 좌표를 "_<좌표>" 형태로 이름에 추가
      }
    }

    _routers[node] = Router::NewRouter( config, this, router_name.str( ),
					node, 2*_n + 1, 2*_n + 1 );
    // [한국어] 라우터 인스턴스 생성:
    //   - router_name.str(): 조립된 이름 (예: "router_0_1")
    //   - node: 라우터의 고유 ID (0 ~ _size-1)
    //   - 2*_n + 1: 입력 포트 수 = n차원 × 좌/우 2개 링크 + 1개 인젝션 포트
    //   - 2*_n + 1: 출력 포트 수 = 동일 (이젝션 포트 포함)
    //   Router::NewRouter()는 config의 "router" 옵션에 따라 IQRouter 등의 서브클래스를 생성

    _timed_modules.push_back(_routers[node]); // [한국어] 이 라우터를 _timed_modules 큐에 추가:
    // 매 사이클 Network::ReadInputs/Evaluate/WriteOutputs 호출 시 이 큐의 모든 모듈이 업데이트됨

    router_name.str(""); // [한국어] ostringstream 버퍼 초기화: 다음 노드의 이름 조립을 위해 기존 내용 삭제

    for ( int dim = 0; dim < _n; ++dim ) { // [한국어] 각 차원(dim=0 ~ _n-1)에 대해 좌/우 채널 연결 수행

      //find the neighbor
      left_node  = _LeftNode( node, dim );  // [한국어] dim 차원에서 현재 노드(node)의 왼쪽 인접 라우터 번호 계산 (torus시 wrap-around 포함)
      right_node = _RightNode( node, dim ); // [한국어] dim 차원에서 현재 노드(node)의 오른쪽 인접 라우터 번호 계산 (torus시 wrap-around 포함)

      //
      // Current (N)ode
      // (L)eft node
      // (R)ight node
      //
      //   L--->N<---R
      //   L<---N--->R
      //
      // [한국어] 채널 방향 설명:
      //   오른쪽(R) → 현재(N): R의 왼쪽 채널이 N의 입력이 됨 (right_input)
      //   왼쪽(L) → 현재(N): L의 오른쪽 채널이 N의 입력이 됨 (left_input)
      //   현재(N) → 오른쪽(R): N의 오른쪽 채널이 N의 출력이 됨 (right_output)
      //   현재(N) → 왼쪽(L): N의 왼쪽 채널이 N의 출력이 됨 (left_output)

      // torus channel is longer due to wrap around
      int latency = _mesh ? 1 : 2 ; // [한국어] 채널 레이턴시 결정:
      // 메시(_mesh=true)는 직선 링크이므로 latency=1,
      // 토러스(_mesh=false)는 wrap-around 링크가 물리적으로 더 길어 latency=2.
      // use_noc_latency=false이면 이 값은 무시되고 아래에서 강제 1로 덮어씌워짐

      //get the input channel number
      right_input = _LeftChannel( right_node, dim );
      // [한국어] 오른쪽 이웃(right_node)에서 현재 노드 방향(왼쪽)으로 오는 채널 인덱스.
      // right_node의 왼쪽 채널(_LeftChannel)이 현재 노드로 들어오는 입력이 됨.
      // 인덱스 = 2*_n*right_node + 2*dim + 1

      left_input  = _RightChannel( left_node, dim );
      // [한국어] 왼쪽 이웃(left_node)에서 현재 노드 방향(오른쪽)으로 오는 채널 인덱스.
      // left_node의 오른쪽 채널(_RightChannel)이 현재 노드로 들어오는 입력이 됨.
      // 인덱스 = 2*_n*left_node + 2*dim

      //add the input channel
      _routers[node]->AddInputChannel( _chan[right_input], _chan_cred[right_input] );
      // [한국어] 오른쪽 이웃 → 현재 노드 방향의 FlitChannel과 CreditChannel을 이 라우터의 입력 포트에 등록.
      // FlitChannel: 데이터 플릿이 오른쪽 이웃에서 현재 노드로 흐르는 채널.
      // CreditChannel: 현재 노드 → 오른쪽 이웃으로 역방향 크레딧 흐름 채널.

      _routers[node]->AddInputChannel( _chan[left_input], _chan_cred[left_input] );
      // [한국어] 왼쪽 이웃 → 현재 노드 방향의 FlitChannel/CreditChannel을 입력 포트에 등록.

      //set input channel latency
      if(use_noc_latency){ // [한국어] NoC 레이턴시 모드: 토폴로지 특성을 반영한 레이턴시 설정
	_chan[right_input]->SetLatency( latency );       // [한국어] 오른쪽 이웃 → 현재 노드 FlitChannel 레이턴시: mesh=1, torus=2
	_chan[left_input]->SetLatency( latency );        // [한국어] 왼쪽 이웃 → 현재 노드 FlitChannel 레이턴시: mesh=1, torus=2
	_chan_cred[right_input]->SetLatency( latency );  // [한국어] 크레딧 역방향 채널 레이턴시도 동일하게 설정 (크레딧 전파 지연도 링크 거리에 비례)
	_chan_cred[left_input]->SetLatency( latency );   // [한국어] 왼쪽 방향 크레딧 채널 레이턴시 설정
      } else { // [한국어] 전통적 네트워크 모드: 모든 채널 레이턴시를 1로 고정 (레이턴시 세분화 없음)
	_chan[left_input]->SetLatency( 1 );              // [한국어] 왼쪽 이웃 → 현재 FlitChannel latency=1 고정
	_chan_cred[right_input]->SetLatency( 1 );        // [한국어] 오른쪽 이웃 방향 CreditChannel latency=1 고정
	_chan_cred[left_input]->SetLatency( 1 );         // [한국어] 왼쪽 이웃 방향 CreditChannel latency=1 고정
	_chan[right_input]->SetLatency( 1 );             // [한국어] 오른쪽 이웃 → 현재 FlitChannel latency=1 고정
      }

      //get the output channel number
      right_output = _RightChannel( node, dim );
      // [한국어] 현재 노드(node)에서 오른쪽 이웃 방향으로 나가는 출력 채널 인덱스.
      // 인덱스 = 2*_n*node + 2*dim

      left_output  = _LeftChannel( node, dim );
      // [한국어] 현재 노드(node)에서 왼쪽 이웃 방향으로 나가는 출력 채널 인덱스.
      // 인덱스 = 2*_n*node + 2*dim + 1

      //add the output channel
      _routers[node]->AddOutputChannel( _chan[right_output], _chan_cred[right_output] );
      // [한국어] 현재 노드 → 오른쪽 이웃 방향의 FlitChannel/CreditChannel을 출력 포트에 등록.

      _routers[node]->AddOutputChannel( _chan[left_output], _chan_cred[left_output] );
      // [한국어] 현재 노드 → 왼쪽 이웃 방향의 FlitChannel/CreditChannel을 출력 포트에 등록.

      //set output channel latency
      if(use_noc_latency){ // [한국어] NoC 레이턴시 모드: 출력 채널에도 토폴로지에 맞는 레이턴시 설정
	_chan[right_output]->SetLatency( latency );      // [한국어] 현재 → 오른쪽 이웃 FlitChannel 레이턴시: mesh=1, torus=2
	_chan[left_output]->SetLatency( latency );       // [한국어] 현재 → 왼쪽 이웃 FlitChannel 레이턴시: mesh=1, torus=2
	_chan_cred[right_output]->SetLatency( latency ); // [한국어] 오른쪽 방향 출력의 크레딧 역방향 채널 레이턴시 설정
	_chan_cred[left_output]->SetLatency( latency );  // [한국어] 왼쪽 방향 출력의 크레딧 역방향 채널 레이턴시 설정
      } else { // [한국어] 전통적 모드: 출력 채널 레이턴시도 모두 1 고정
	_chan[right_output]->SetLatency( 1 ); // [한국어] 오른쪽 출력 FlitChannel latency=1
	_chan[left_output]->SetLatency( 1 );  // [한국어] 왼쪽 출력 FlitChannel latency=1
	_chan_cred[right_output]->SetLatency( 1 ); // [한국어] 오른쪽 출력 CreditChannel latency=1
	_chan_cred[left_output]->SetLatency( 1 );  // [한국어] 왼쪽 출력 CreditChannel latency=1

      }
    }
    //injection and ejection channel, always 1 latency
    // [한국어] 인젝션/이젝션 채널: SM/L2 ↔ 라우터 연결 채널은 링크 타입 무관하게 항상 latency=1
    _routers[node]->AddInputChannel( _inject[node], _inject_cred[node] );
    // [한국어] 인젝션 FlitChannel(_inject[node])과 CreditChannel(_inject_cred[node])을 입력 포트에 등록:
    // SM/L2(외부 엔드포인트)에서 이 라우터로 플릿을 주입하는 채널

    _routers[node]->AddOutputChannel( _eject[node], _eject_cred[node] );
    // [한국어] 이젝션 FlitChannel(_eject[node])과 CreditChannel(_eject_cred[node])을 출력 포트에 등록:
    // 이 라우터에서 SM/L2(목적지 엔드포인트)로 플릿을 배출하는 채널

    _inject[node]->SetLatency( 1 ); // [한국어] 인젝션 채널 레이턴시: 항상 1 사이클 (외부 주입 채널은 단순 1홉)
    _eject[node]->SetLatency( 1 );  // [한국어] 이젝션 채널 레이턴시: 항상 1 사이클 (외부 배출 채널도 단순 1홉)
  }
}

/*
 * [한국어]
 * _LeftChannel - 지정된 노드와 차원에서 왼쪽 방향 채널의 전역 인덱스를 반환한다.
 *
 * @node: 라우터 노드 번호 (0 ~ _size-1).
 * @dim:  차원 번호 (0 ~ _n-1).
 * @return: _chan[] 및 _chan_cred[] 배열에서의 채널 인덱스.
 *          계산식: 2*_n*node + 2*dim + 1
 *
 * 채널 인덱싱 규약:
 *   각 노드의 기저(base) 인덱스 = 2*_n*node.
 *   dim 차원의 오른쪽(+방향) 채널 = base + 2*dim.
 *   dim 차원의 왼쪽(-방향) 채널 = base + 2*dim + 1.
 * 따라서 "왼쪽 채널"은 해당 노드에서 왼쪽 이웃 방향으로 나가는 출력 채널이기도 하고,
 * 오른쪽 이웃이 현재 노드로 보내는 입력 채널(_chan[LeftChannel(right_node)])이기도 하다.
 *
 * 실행 컨텍스트: _BuildNet()에서 채널 인덱스 계산 시 호출. 단일 스레드.
 *
 * 호출 체인:
 *   _BuildNet() → [_LeftChannel(node, dim)] : 자신의 왼쪽 출력 채널 인덱스
 *   _BuildNet() → [_LeftChannel(right_node, dim)] : 오른쪽 이웃의 왼쪽 채널 → 현재 노드 입력
 *   InsertRandomFaults() → (간접) 채널 장애 주입 경로에서 참조
 */
int KNCube::_LeftChannel( int node, int dim )
{
  // The base channel for a node is 2*_n*node
  int base = 2*_n*node; // [한국어] 노드 node의 채널 기저 인덱스: node마다 2*_n개의 채널을 보유하므로 node × 2*_n이 시작점

  // The offset for a left channel is 2*dim + 1
  int off  = 2*dim + 1; // [한국어] dim 차원의 왼쪽(-방향) 채널 오프셋: 2*dim은 오른쪽 채널, +1이 왼쪽 채널을 구분하는 비트

  return ( base + off ); // [한국어] 최종 채널 인덱스 반환: _chan[base+off]가 이 노드-dim의 왼쪽 방향 FlitChannel
}

/*
 * [한국어]
 * _RightChannel - 지정된 노드와 차원에서 오른쪽 방향 채널의 전역 인덱스를 반환한다.
 *
 * @node: 라우터 노드 번호 (0 ~ _size-1).
 * @dim:  차원 번호 (0 ~ _n-1).
 * @return: _chan[] 및 _chan_cred[] 배열에서의 채널 인덱스.
 *          계산식: 2*_n*node + 2*dim
 *
 * _LeftChannel과 대칭적. 오프셋이 2*dim (짝수)이므로 왼쪽(2*dim+1, 홀수)보다 1 작다.
 * "오른쪽 채널"은 해당 노드에서 오른쪽 이웃 방향으로 나가는 출력 채널이기도 하고,
 * 왼쪽 이웃이 현재 노드로 보내는 입력 채널(_chan[RightChannel(left_node)])이기도 하다.
 *
 * 실행 컨텍스트: _BuildNet()에서 채널 인덱스 계산 시 호출. 단일 스레드.
 *
 * 호출 체인:
 *   _BuildNet() → [_RightChannel(node, dim)] : 자신의 오른쪽 출력 채널 인덱스
 *   _BuildNet() → [_RightChannel(left_node, dim)] : 왼쪽 이웃의 오른쪽 채널 → 현재 노드 입력
 */
int KNCube::_RightChannel( int node, int dim )
{
  // The base channel for a node is 2*_n*node
  int base = 2*_n*node; // [한국어] 노드 node의 채널 기저 인덱스 (노드당 2*_n개 채널 블록)

  // The offset for a right channel is 2*dim
  int off  = 2*dim; // [한국어] dim 차원의 오른쪽(+방향) 채널 오프셋: 짝수 오프셋 = 오른쪽, 홀수 오프셋 = 왼쪽

  return ( base + off ); // [한국어] 최종 채널 인덱스 반환: _chan[base+off]가 이 노드-dim의 오른쪽 방향 FlitChannel
}

/*
 * [한국어]
 * _LeftNode - 지정된 노드의 지정된 차원에서 왼쪽 인접 라우터 노드 번호를 반환한다.
 *
 * @node: 기준 라우터 노드 번호 (0 ~ _size-1).
 * @dim:  차원 번호 (0 ~ _n-1). 어느 차원에서 왼쪽 이웃을 찾을지 결정.
 * @return: 왼쪽 인접 노드 번호 (0 ~ _size-1). 차원 경계에서 wrap-around 처리 포함.
 *
 * 노드 번호는 혼합 기수법으로 인코딩되어 있다:
 *   dim 차원에서의 좌표 = (node / k^dim) % k
 *   k^dim = powi(_k, dim) = 해당 차원의 stride (dim 차원 좌표가 1 바뀔 때 node 번호 변화량)
 * 왼쪽 이웃 계산:
 *   loc_in_dim == 0이면 차원 경계의 왼쪽 끝 → 오른쪽 끝으로 wrap-around:
 *     left_node = node + (k-1)*k^dim (같은 차원의 반대편 끝 노드)
 *   loc_in_dim > 0이면 단순히 왼쪽으로 1칸:
 *     left_node = node - k^dim
 * 메시(_mesh=true)에서는 이 wrap-around 계산이 수행되지만, _BuildNet에서 latency=1로
 * 처리되고 실제로 경계 포트 연결이 의미 없는 구조이므로 라우팅 알고리즘이 메시 경계를
 * 인식하고 wrap-around 경로를 사용하지 않는 것이 필요하다.
 *
 * 실행 컨텍스트: _BuildNet()에서 각 노드-차원마다 호출. 단일 스레드.
 *
 * 호출 체인:
 *   _BuildNet() → [_LeftNode(node, dim)] → 입력 채널 연결 시 이웃 탐색
 *   InsertRandomFaults() → [_LeftNode(node, n)] → 장애 전파 시 이웃 노드 마킹
 */
int KNCube::_LeftNode( int node, int dim )
{
  int k_to_dim = powi( _k, dim ); // [한국어] k^dim: dim 차원의 stride. 이 값만큼 node 번호가 바뀌면 dim 차원 좌표가 1 변함. misc_utils.hpp의 powi()로 계산
  int loc_in_dim = ( node / k_to_dim ) % _k; // [한국어] 현재 노드의 dim 차원 좌표: (node / stride) % k. 0이면 왼쪽 끝, k-1이면 오른쪽 끝

  int left_node;
  // if at the left edge of the dimension, wraparound
  if ( loc_in_dim == 0 ) { // [한국어] dim 차원에서 왼쪽 경계(좌표=0)인 경우: wrap-around로 반대편 끝(좌표=k-1)으로 이동
    left_node = node + (_k-1)*k_to_dim; // [한국어] wrap-around 왼쪽 이웃: 같은 노드 번호에서 (k-1)*stride를 더하면 해당 차원의 반대편 끝 노드
  } else { // [한국어] 차원 내부(좌표>0)인 경우: 단순히 stride만큼 빼면 왼쪽 이웃
    left_node = node - k_to_dim; // [한국어] 왼쪽 이웃: dim 차원 좌표를 1 감소 → node 번호를 stride만큼 감소
  }

  return left_node; // [한국어] 계산된 왼쪽 인접 노드 번호 반환 (torus/mesh 공통)
}

/*
 * [한국어]
 * _RightNode - 지정된 노드의 지정된 차원에서 오른쪽 인접 라우터 노드 번호를 반환한다.
 *
 * @node: 기준 라우터 노드 번호 (0 ~ _size-1).
 * @dim:  차원 번호 (0 ~ _n-1).
 * @return: 오른쪽 인접 노드 번호 (0 ~ _size-1). 차원 경계에서 wrap-around 처리 포함.
 *
 * _LeftNode와 대칭적. 오른쪽 경계(loc_in_dim == k-1)에서 wrap-around:
 *   right_node = node - (k-1)*k^dim (반대편 왼쪽 끝 노드)
 * 내부에서는:
 *   right_node = node + k^dim (stride만큼 증가)
 *
 * 실행 컨텍스트: _BuildNet()에서 각 노드-차원마다 호출. 단일 스레드.
 *
 * 호출 체인:
 *   _BuildNet() → [_RightNode(node, dim)] → 입력 채널 연결 시 이웃 탐색
 *   InsertRandomFaults() → [_RightNode(node, n)] → 장애 전파 시 이웃 노드 마킹
 */
int KNCube::_RightNode( int node, int dim )
{
  int k_to_dim = powi( _k, dim ); // [한국어] k^dim: dim 차원의 stride. _LeftNode와 동일한 방식으로 계산
  int loc_in_dim = ( node / k_to_dim ) % _k; // [한국어] 현재 노드의 dim 차원 좌표: k-1이면 오른쪽 경계

  int right_node;
  // if at the right edge of the dimension, wraparound
  if ( loc_in_dim == ( _k-1 ) ) { // [한국어] dim 차원에서 오른쪽 경계(좌표=k-1)인 경우: wrap-around로 반대편 왼쪽 끝(좌표=0)으로 이동
    right_node = node - (_k-1)*k_to_dim; // [한국어] wrap-around 오른쪽 이웃: (k-1)*stride를 빼면 해당 차원의 왼쪽 끝 노드
  } else { // [한국어] 차원 내부(좌표<k-1)인 경우: stride만큼 더하면 오른쪽 이웃
    right_node = node + k_to_dim; // [한국어] 오른쪽 이웃: dim 차원 좌표를 1 증가 → node 번호를 stride만큼 증가
  }

  return right_node; // [한국어] 계산된 오른쪽 인접 노드 번호 반환
}

/*
 * [한국어]
 * GetN - 차원 수(_n)를 반환하는 읽기 전용 접근자.
 *
 * @return: _n (차원 수). 예) 2D 메시이면 2.
 *
 * 라우팅 알고리즘이나 외부 통계 코드에서 차원 수를 조회할 때 사용.
 * const 함수이므로 객체 상태를 변경하지 않는다.
 *
 * 호출 체인:
 *   라우팅 함수 / 통계 수집 코드 → [GetN()]
 */
int KNCube::GetN( ) const
{
  return _n; // [한국어] _n(차원 수) 반환: _ComputeSize()에서 "n" 옵션으로 설정된 값
}

/*
 * [한국어]
 * GetK - 차원당 라우터 수(_k)를 반환하는 읽기 전용 접근자.
 *
 * @return: _k (radix). 예) 4-ary이면 4.
 *
 * 라우팅 알고리즘이나 외부 통계 코드에서 radix를 조회할 때 사용.
 * const 함수이므로 객체 상태를 변경하지 않는다.
 *
 * 호출 체인:
 *   라우팅 함수 / 통계 수집 코드 → [GetK()]
 */
int KNCube::GetK( ) const
{
  return _k; // [한국어] _k(차원당 라우터 수) 반환: _ComputeSize()에서 "k" 옵션으로 설정된 값
}

/*legacy, not sure how this fits into the own scheme of things*/
/*
 * [한국어]
 * InsertRandomFaults - 랜덤 링크 장애를 네트워크에 삽입한다 (레거시 코드).
 *
 * @config: 설정 객체. "link_failures"(삽입할 장애 수), "fail_seed"(난수 시드) 옵션 사용.
 * @return: 없음.
 *
 * 레거시(legacy) 코드임을 원본 주석에서 명시한다. 현재 일반적인 GPU NoC 시뮬레이션에서는
 * 거의 사용되지 않으며, 장애 허용(fault-tolerance) 연구 목적으로 작성된 코드다.
 *
 * 동작 과정:
 *   1) config에서 link_failures(장애 수)와 fail_seed(시드)를 읽는다.
 *   2) 현재 난수 시드를 저장하고 fail_seed로 교체 (재현 가능한 랜덤 시퀀스).
 *   3) fail_nodes 배열로 각 노드의 "장애 삽입 불가" 여부를 추적.
 *      에지 노드(어떤 차원에서든 좌표가 0 또는 k-1인 노드)는 초기에 true(불가)로 설정.
 *   4) num_fails만큼 반복:
 *      a) 랜덤 노드(j)에서 시작하여 장애 삽입 가능한 내부 노드 탐색.
 *      b) 해당 노드의 랜덤 채널을 선택, 채널의 반대편 이웃이 장애 삽입 불가 노드이면 유효한 선택.
 *      c) OutChannelFault(node, chan)으로 채널 장애 설정.
 *      d) 장애 노드 및 그 n차원 이웃 모두를 fail_nodes=true로 마킹 (인접 노드 중복 장애 방지).
 *   5) 난수 시드를 원래대로 복원.
 *
 * 에지 노드 제외 이유: 에지 노드는 wrap-around 채널이 없거나 연결이 제한되어
 * 장애 삽입 시 네트워크 단절 위험이 높기 때문.
 *
 * 실행 컨텍스트: 초기화 단계에서 1회 호출 (장애 삽입 시험 시). 단일 스레드.
 *
 * 호출 체인:
 *   Network 초기화 / 장애 실험 코드 → [InsertRandomFaults(config)]
 *   → RandomIntLong() [현재 시드 저장]
 *   → RandomSeed(fail_seed) [시드 교체]
 *   → RandomInt() [랜덤 노드/채널 선택]
 *   → _LeftNode() / _RightNode() [이웃 노드 탐색]
 *   → OutChannelFault(node, chan) [Network 기반 클래스: 채널 장애 설정]
 *   → RandomSeed(prev_seed) [시드 복원]
 */
void KNCube::InsertRandomFaults( const Configuration &config )
{
  int num_fails;     // [한국어] 삽입할 링크 장애 총수: config의 "link_failures" 옵션에서 읽음
  unsigned long prev_seed; // [한국어] 장애 삽입 전 난수 시드 저장용: 함수 종료 시 복원하여 다른 난수 시퀀스에 영향 없게 함

  int node, chan = 0; // [한국어] 장애를 삽입할 노드 번호와 채널 번호. chan=0으로 초기화 (탐색 중 갱신됨)
  int i, j, t, n, c; // [한국어] 루프 인덱스: i=장애 삽입 횟수, j=시작 노드, t=탐색 시도 횟수, n=차원, c=채널 시작 오프셋
  bool available;    // [한국어] 유효한 장애 삽입 위치를 찾았는지 여부: true이면 장애 삽입 가능한 채널 발견

  bool edge; // [한국어] 현재 노드가 에지(경계) 노드인지 여부: 에지 노드는 장애 삽입 대상에서 제외

  num_fails = config.GetInt( "link_failures" ); // [한국어] config에서 "link_failures" 옵션 읽기: 삽입할 장애 수 결정

  if ( _size && num_fails ) { // [한국어] 네트워크가 존재하고(_size>0) 장애 삽입 요청이 있을 때만 실행
    prev_seed = RandomIntLong( ); // [한국어] 현재 난수 상태 저장: random_utils.hpp의 RandomIntLong()으로 현재 시드 추출
    RandomSeed( config.GetInt( "fail_seed" ) ); // [한국어] fail_seed로 난수 시드 교체: 재현 가능한(deterministic) 장애 패턴 생성

    vector<bool> fail_nodes(_size); // [한국어] 크기 _size의 bool 벡터: 각 노드의 장애 삽입 가능 여부(false=가능, true=불가) 추적

    for ( i = 0; i < _size; ++i ) { // [한국어] 모든 노드를 순회하며 에지 노드를 미리 불가(true)로 마킹
      node = i; // [한국어] 현재 검사 중인 노드 번호

      // edge test
      // [한국어] 에지 노드 판별: 각 차원에서 좌표가 0 또는 k-1이면 에지로 판단
      edge = false; // [한국어] 에지 여부 초기화
      for ( n = 0; n < _n; ++n ) { // [한국어] 모든 차원을 순회하며 에지 여부 확인
	if ( ( ( node % _k ) == 0 ) ||
	     ( ( node % _k ) == _k - 1 ) ) {
          // [한국어] node % _k로 현재 차원(n)에서의 좌표 추출:
          //   0이면 해당 차원의 왼쪽 끝, k-1이면 오른쪽 끝 → 에지 노드로 분류.
          //   에지 노드는 wrap-around 채널이 실제로 연결되지 않아 장애 삽입 시 네트워크 단절 위험
	  edge = true; // [한국어] 어느 차원이라도 에지이면 이 노드는 에지 노드로 마킹
	}
	node /= _k; // [한국어] 다음 차원의 좌표를 추출하기 위해 _k로 나눔 (혼합 기수법 디코딩)
      }

      if ( edge ) { // [한국어] 에지 노드이면: 장애 삽입 불가로 마킹
	fail_nodes[i] = true; // [한국어] 에지 노드는 장애 삽입 대상에서 제외 (초기에 불가 상태로 설정)
      } else { // [한국어] 내부 노드이면: 장애 삽입 가능 상태로 초기화
	fail_nodes[i] = false; // [한국어] 내부 노드는 장애 삽입 후보로 초기화
      }
    }

    for ( i = 0; i < num_fails; ++i ) { // [한국어] num_fails만큼 장애를 삽입하는 메인 루프
      j = RandomInt( _size - 1 ); // [한국어] 랜덤 시작 노드 j 선택: 0 ~ _size-1 범위에서 균일 난수
      available = false; // [한국어] 유효 위치 미발견 상태로 초기화

      for ( t = 0; ( t < _size ) && (!available); ++t ) { // [한국어] 최대 _size번 시도: j부터 순환 탐색하여 유효 노드 발견 시 중단
	node = ( j + t ) % _size; // [한국어] j에서 t칸 앞의 노드 (순환): 모든 노드를 순서대로 탐색하는 round-robin

	if ( !fail_nodes[node] ) { // [한국어] 현재 노드가 장애 삽입 가능 상태(내부 노드)인 경우에만 채널 탐색
	  // check neighbors
	  c = RandomInt( 2*_n - 1 ); // [한국어] 랜덤 채널 시작 오프셋 c: 0 ~ 2*_n-1 범위. 특정 채널로의 편향 없이 공정하게 탐색

	  for ( n = 0; ( n < 2*_n ) && (!available); ++n ) { // [한국어] 최대 2*_n개 채널 탐색: c부터 순환하며 유효 채널 발견 시 중단
	    chan = ( n + c ) % 2*_n; // [한국어] c에서 n칸 앞의 채널 인덱스 (순환): 채널 0 ~ 2*_n-1을 라운드로빈으로 검사

	    if ( chan % 1 ) { // [한국어] 주의: 원본 코드의 버그 가능성 — chan%1은 항상 0(false)이므로 else 분기만 실행됨.
	      //   의도는 chan%2로 홀수(왼쪽) 채널과 짝수(오른쪽) 채널을 구분하려 했을 가능성 있음 (레거시 코드의 알려진 버그)
	      available = fail_nodes[_LeftNode( node, chan/2 )]; // [한국어] (사실상 실행 안 됨) 왼쪽 이웃이 장애 불가 상태이면 이 채널이 유효한 장애 삽입 대상
	    } else {
	      available = fail_nodes[_RightNode( node, chan/2 )]; // [한국어] 채널 번호/2 = 차원 번호: 해당 차원의 오른쪽 이웃이 장애 불가 상태이면 유효한 장애 삽입 대상
	    }
	  }
	}

	if ( !available ) { // [한국어] 이 노드에서 유효한 채널을 찾지 못한 경우: 다음 노드로 넘어가기 전에 로그 출력
	  cout << "skipping " << node << endl; // [한국어] 탐색 건너뛰기 알림: 디버그용 출력 (표준 출력)
	}
      }

      if ( t == _size ) { // [한국어] _size번 시도 후에도 유효 위치를 찾지 못한 경우: 치명적 오류로 시뮬레이션 중단
	Error( "Could not find another possible fault channel" ); // [한국어] Error(): booksim.hpp의 오류 출력 매크로 → 시뮬레이션 종료
      }


      OutChannelFault( node, chan ); // [한국어] Network 기반 클래스의 OutChannelFault()로 실제 채널 장애 설정:
      // _routers[node]의 출력 채널 chan에 장애를 주입하여 플릿 전송 불가 상태로 만듦

      fail_nodes[node] = true; // [한국어] 장애가 삽입된 노드를 불가 상태로 마킹: 동일 노드에 이중 장애 삽입 방지

      for ( n = 0; ( n < _n ) && available ; ++n ) { // [한국어] 장애 노드의 모든 차원 이웃을 불가 상태로 마킹: 인접 노드 집중 장애 방지
	fail_nodes[_LeftNode( node, n )]  = true; // [한국어] dim n 방향 왼쪽 이웃 노드를 장애 불가로 마킹
	fail_nodes[_RightNode( node, n )] = true; // [한국어] dim n 방향 오른쪽 이웃 노드를 장애 불가로 마킹
      }

      cout << "failure at node " << node << ", channel "
	   << chan << endl; // [한국어] 장애 삽입 결과를 표준 출력에 기록: 어느 노드의 어느 채널에 장애가 삽입되었는지 보고
    }

    RandomSeed( prev_seed ); // [한국어] 난수 시드를 장애 삽입 전 상태로 복원: 이후 시뮬레이션의 난수 시퀀스가 장애 삽입에 의해 오염되지 않게 함
  }
}

/*
 * [한국어]
 * Capacity - 네트워크의 이론적 처리 용량(bisection bandwidth 기반)을 반환한다.
 *
 * @return: double. _mesh=true이면 (double)_k/8.0, _mesh=false이면 (double)_k/4.0.
 *
 * 이 값은 k-ary n-cube에서 bisection 단면을 지나는 링크 수를 전체 노드 수로 정규화한
 * 이론적 지표이다. torus는 양방향 wrap-around 링크로 메시 대비 bisection bandwidth가
 * 2배이므로 나누는 값이 4.0(torus) vs 8.0(mesh)으로 2배 차이.
 * 실제 처리량 상한(throughput bound)으로 시뮬레이션 결과와 비교하는 데 사용된다.
 *
 * 실행 컨텍스트: 통계/검증 코드에서 필요 시 호출. 상태 변경 없음(const).
 *
 * 호출 체인:
 *   Network 통계 출력 / 검증 코드 → [Capacity()]
 */
double KNCube::Capacity( ) const
{
  return (double)_k / ( _mesh ? 8.0 : 4.0 ); // [한국어] 이론적 처리 용량 계산:
  // _k를 8.0(메시) 또는 4.0(토러스)으로 나눔.
  // 메시: bisection 단면 링크 수가 적어 용량이 작음 → /8.0.
  // 토러스: wrap-around 링크로 bisection 용량이 2배 → /4.0.
  // (double) 캐스트: _k가 int이므로 정수 나눗셈이 아닌 부동소수점 나눗셈 수행
}
