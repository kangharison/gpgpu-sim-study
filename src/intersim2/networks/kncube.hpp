// $Id: kncube.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] k진 n-큐브(KNCube) 네트워크 토폴로지 클래스 선언 (kncube.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 intersim2(Booksim 기반 NoC 시뮬레이터)에서 사용되는 k진 n-큐브(k-ary n-cube)
 * 네트워크 토폴로지를 구현하는 KNCube 클래스를 선언한다. k-ary n-cube는 각 차원(dimension)마다
 * k개의 라우터가 배열된 n차원 격자(grid) 구조이며, 링크 연결 방식에 따라 메시(mesh, 끝단
 * 연결 없음)와 토러스(torus, 끝단 wrap-around 연결)의 두 가지 변형을 모두 지원한다.
 * GPU NoC(Network-on-Chip)에서는 2D 메시(n=2, _mesh=true)가 가장 흔히 사용된다.
 * 이 클래스는 Network 기반 클래스를 상속받아 라우터 생성, 채널 연결, 레이턴시 설정 등
 * 토폴로지 구축의 전 과정을 담당한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 NoC 시뮬레이션 레이어인 intersim2 안에 위치한다.
 * 호출 체인 (위→아래):
 *   gpu-sim.cc(gpgpu_sim::cycle) → icnt_wrapper.cc(icnt_push/icnt_pop) →
 *   LocalInterconnect (local_interconnect.cc) 또는
 *   intersim2 Network::NewNetwork() → KNCube 생성자 → _ComputeSize + _BuildNet
 * 실행 컨텍스트: 시뮬레이터 초기화 시(Network 구축) 및 매 사이클(ReadInputs/Evaluate/
 * WriteOutputs 호출, 단 이 파일에 직접 구현은 없고 Network 기반 클래스가 처리).
 * 이 헤더는 kncube.cpp에서만 구현되며, Network 기반 클래스(network.hpp)를 통해
 * icnt_wrapper에서 간접 참조된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - network.hpp (Network 기반 클래스): _size, _nodes, _channels, _routers,
 *     _chan, _chan_cred, _inject, _eject, _timed_modules 등 기반 자원 관리
 *   - Router::NewRouter() (iq_router 등): _BuildNet에서 각 노드의 라우터 인스턴스 생성
 *   - misc_utils.hpp: powi() — k^n 계산에 사용
 *   - random_utils.hpp: RandomInt/RandomSeed/RandomIntLong — InsertRandomFaults 사용
 *   - Configuration: gpgpusim.config의 "k", "n", "use_noc_latency", "link_failures",
 *     "fail_seed" 옵션을 읽어 토폴로지와 동작을 결정
 * 이 모듈에 의존하는 모듈:
 *   - icnt_wrapper.cc: Network* 포인터로 KNCube 인스턴스를 보유 및 구동
 *   - local_interconnect.cc: LocalInterconnect가 KNCube를 내부 망으로 사용 가능
 * 데이터 흐름: SM/L2의 플릿 → _inject[src] → (KNCube 내부 라우터 + 채널) → _eject[dst] → SM/L2
 *
 * === 주요 함수/구조체 요약 ===
 * - KNCube(config, name, mesh): 생성자. _mesh 설정, 크기 계산, 라우터·채널 할당·연결
 * - _ComputeSize(config): _k/_n을 읽어 _size=k^n, _channels=2*n*size 계산
 * - _BuildNet(config): 모든 라우터 생성, 각 차원의 좌/우 채널·크레딧 채널 연결, 레이턴시 설정
 * - _LeftChannel/RightChannel(node,dim): 주어진 노드·차원의 채널 인덱스 반환
 * - _LeftNode/_RightNode(node,dim): 인접 라우터 노드 번호 반환 (torus시 wrap-around 처리)
 * - GetN()/GetK(): 차원 수/차원당 라우터 수 반환 (읽기 전용 접근자)
 * - Capacity(): 이론적 처리 용량 반환 — mesh: k/8.0, torus: k/4.0
 * - InsertRandomFaults(config): 레거시 코드, 랜덤 링크 장애 삽입
 */

#ifndef _KNCUBE_HPP_
#define _KNCUBE_HPP_

#include "network.hpp" // [한국어] Network 기반 클래스: 라우터/채널 배열, _size/_nodes/_channels, ReadInputs/Evaluate/WriteOutputs 등 공통 인프라 제공

/*
 * [한국어] KNCube — k진 n-큐브(mesh 또는 torus) 토폴로지 클래스.
 * Network를 상속받아 k-ary n-cube 구조의 라우터·채널 네트워크를 구축한다.
 * _mesh=true 이면 메시(끝단 연결 없음), false 이면 토러스(wrap-around 연결).
 */
class KNCube : public Network {

  bool _mesh;
  /* [한국어] 토폴로지 종류를 결정하는 플래그.
   * 설정자: 생성자 KNCube(config, name, mesh)의 세 번째 인수로 전달되어 초기화.
   *         true → 메시(mesh): 각 차원의 양 끝 노드 간 링크가 없음 (선형 배열).
   *         false → 토러스(torus): 양 끝 노드가 wrap-around 링크로 연결됨 (링 구조).
   * 읽는 자: _BuildNet()에서 레이턴시 결정(mesh=1, torus=2),
   *          _LeftNode/_RightNode에서 wrap-around 여부 판단,
   *          Capacity()에서 이론 용량 계산.
   * 값 범위: true(메시) 또는 false(토러스). 생성 후 변경되지 않음(읽기 전용).
   * 동기화: 생성 후 불변이므로 별도 동기화 불필요. */

  int _k;
  /* [한국어] 각 차원당 라우터(노드) 수 — k-ary n-cube의 'k'(radix).
   * 설정자: _ComputeSize()가 gpgpusim.config의 "k" 옵션을 읽어 설정.
   *         전역 변수 gK에도 동일 값이 복사됨(intersim2 라우팅 함수 참조용).
   * 읽는 자: _BuildNet()에서 라우터 이름 생성, _LeftNode/_RightNode에서 위치 계산,
   *          InsertRandomFaults에서 에지 노드 판별, Capacity()에서 용량 계산.
   * 값 범위: 양의 정수. 일반적으로 GPU NoC에서는 4~8 (2D 4x4 ~ 8x8 메시).
   * 동기화: _ComputeSize() 이후 불변. */

  int _n;
  /* [한국어] 차원 수(dimension count) — k-ary n-cube의 'n'.
   * 설정자: _ComputeSize()가 gpgpusim.config의 "n" 옵션을 읽어 설정.
   *         전역 변수 gN에도 동일 값이 복사됨.
   * 읽는 자: _ComputeSize()에서 _size=k^n, _channels=2*n*size 계산,
   *          _BuildNet()에서 채널 포트 수(2*n) 결정 및 차원 루프,
   *          _LeftChannel/_RightChannel/_LeftNode/_RightNode의 차원 오프셋 계산,
   *          InsertRandomFaults의 에지 판별 루프.
   * 값 범위: 양의 정수. GPU NoC에서는 보통 2(2D).
   * 동기화: _ComputeSize() 이후 불변. */

  /*
   * [한국어]
   * _ComputeSize - gpgpusim.config에서 k, n을 읽어 네트워크 규모를 계산한다.
   *
   * @config: GPGPU-Sim 설정 객체. "k"(radix), "n"(차원 수) 옵션을 제공.
   * @return: 없음 (Network 기반 클래스의 _size/_nodes/_channels를 직접 설정).
   *
   * 이 함수는 KNCube 생성자에서 가장 먼저 호출되어 네트워크 크기를 확정한다.
   * _size = k^n (전체 라우터 수), _channels = 2*n*size (각 노드의 n개 차원 × 좌/우 2방향),
   * _nodes = _size (라우터 수 = 엔드포인트 수).
   * 전역 gK/gN에도 값을 복사하여 intersim2 라우팅 함수들이 참조할 수 있게 한다.
   *
   * 호출 체인:
   *   KNCube() 생성자 → [_ComputeSize()] → Network::_Alloc() → _BuildNet()
   */
  void _ComputeSize( const Configuration &config );

  /*
   * [한국어]
   * _BuildNet - 모든 라우터 인스턴스를 생성하고 채널로 연결하여 토폴로지를 구축한다.
   *
   * @config: GPGPU-Sim 설정 객체. "use_noc_latency" 등 채널 설정에 사용.
   * @return: 없음 (Network 기반 클래스의 _routers, _chan, _inject, _eject 등을 채운다).
   *
   * _ComputeSize()와 _Alloc() 이후 호출된다. 각 노드(라우터)에 대해:
   *   1) Router::NewRouter()로 라우터 인스턴스 생성 (포트 수 = 2*n + 1: n차원 × 좌/우 + 인젝션).
   *   2) 각 차원(dim)의 왼쪽·오른쪽 이웃 노드를 _LeftNode/_RightNode로 구한다.
   *   3) 이웃 노드의 반대편 채널을 입력, 자신의 채널을 출력으로 AddInputChannel/AddOutputChannel.
   *   4) use_noc_latency=true이면 torus 링크에 latency=2, mesh에 latency=1을 설정한다.
   *   5) 인젝션/이젝션 채널은 항상 latency=1.
   *
   * 호출 체인:
   *   KNCube() 생성자 → _ComputeSize() → _Alloc() → [_BuildNet()]
   */
  void _BuildNet( const Configuration &config );

  /*
   * [한국어]
   * _LeftChannel - 지정된 노드의 지정된 차원에서 왼쪽 방향 채널 인덱스를 반환한다.
   *
   * @node: 라우터 노드 번호 (0 ~ _size-1).
   * @dim:  차원 번호 (0 ~ _n-1).
   * @return: 전역 _chan 배열에서의 채널 인덱스.
   *          계산식: 2*_n*node + 2*dim + 1
   *
   * 채널 인덱싱 규약: 각 노드의 기저 채널 = 2*_n*node.
   * dim 차원의 오른쪽 채널 = +2*dim, 왼쪽 채널 = +2*dim+1.
   * '_BuildNet에서 입력/출력 채널을 연결할 때 사용한다.
   *
   * 호출 체인:
   *   _BuildNet() → [_LeftChannel()] : 자신의 왼쪽 출력 채널 번호 획득
   *   _BuildNet() → [_LeftChannel()] : 오른쪽 이웃의 입력 채널 번호 획득
   */
  int _LeftChannel( int node, int dim );

  /*
   * [한국어]
   * _RightChannel - 지정된 노드의 지정된 차원에서 오른쪽 방향 채널 인덱스를 반환한다.
   *
   * @node: 라우터 노드 번호 (0 ~ _size-1).
   * @dim:  차원 번호 (0 ~ _n-1).
   * @return: 전역 _chan 배열에서의 채널 인덱스.
   *          계산식: 2*_n*node + 2*dim
   *
   * 채널 인덱싱 규약은 _LeftChannel 주석 참조.
   * _BuildNet에서 입력/출력 채널을 연결할 때 사용한다.
   *
   * 호출 체인:
   *   _BuildNet() → [_RightChannel()] : 자신의 오른쪽 출력 채널 번호 획득
   *   _BuildNet() → [_RightChannel()] : 왼쪽 이웃의 입력 채널 번호 획득
   */
  int _RightChannel( int node, int dim );

  /*
   * [한국어]
   * _LeftNode - 지정된 노드의 지정된 차원에서 왼쪽 방향 인접 라우터 노드 번호를 반환한다.
   *
   * @node: 기준 라우터 노드 번호 (0 ~ _size-1).
   * @dim:  차원 번호 (0 ~ _n-1).
   * @return: 왼쪽 인접 노드 번호. 메시·토러스 공통으로 wrap-around 처리를 포함.
   *          dim 차원에서 loc=0이면 torus: node + (k-1)*k^dim (왼쪽 끝으로 wraparound).
   *          loc>0이면 node - k^dim.
   *
   * torus(_mesh=false)에서 이 함수는 차원 경계에서 반대편 끝 노드를 반환한다.
   * mesh(_mesh=true)에서도 동일한 wraparound 계산을 하지만, _BuildNet의 latency
   * 설정에서 메시 여부를 반영하고, 실제 메시에서는 엣지 포트가 연결되지 않는 것이 아니라
   * 동일 공식으로 처리한다 (intersim2 설계상 메시도 wrap-around 계산 사용).
   *
   * 호출 체인:
   *   _BuildNet() → [_LeftNode()] : 입력 채널 연결 시 이웃 노드 탐색
   *   InsertRandomFaults() → [_LeftNode()] : 인접 노드 장애 전파 시
   */
  int _LeftNode( int node, int dim );

  /*
   * [한국어]
   * _RightNode - 지정된 노드의 지정된 차원에서 오른쪽 방향 인접 라우터 노드 번호를 반환한다.
   *
   * @node: 기준 라우터 노드 번호 (0 ~ _size-1).
   * @dim:  차원 번호 (0 ~ _n-1).
   * @return: 오른쪽 인접 노드 번호.
   *          dim 차원에서 loc=k-1이면 node - (k-1)*k^dim (오른쪽 끝에서 wraparound).
   *          loc<k-1이면 node + k^dim.
   *
   * _LeftNode와 대칭적. 차원 경계에서의 wrap-around 처리 포함.
   *
   * 호출 체인:
   *   _BuildNet() → [_RightNode()] : 입력 채널 연결 시 이웃 노드 탐색
   *   InsertRandomFaults() → [_RightNode()] : 인접 노드 장애 전파 시
   */
  int _RightNode( int node, int dim );

public:
  /*
   * [한국어]
   * KNCube 생성자 - k진 n-큐브 네트워크를 초기화하고 구축한다.
   *
   * @config: GPGPU-Sim 설정 객체. "k", "n", "use_noc_latency" 등 옵션 포함.
   * @name:   이 네트워크 인스턴스의 문자열 이름 (디버그/출력용).
   * @mesh:   true → 메시 토폴로지, false → 토러스 토폴로지.
   * @return: 없음 (생성자).
   *
   * Network 기반 생성자를 먼저 호출한 뒤, _mesh 설정 → _ComputeSize → _Alloc → _BuildNet
   * 순서로 전체 네트워크를 구성한다.
   *
   * 호출 체인:
   *   Network::NewNetwork() 또는 icnt_wrapper_init() → [KNCube()]
   */
  KNCube( const Configuration &config, const string & name, bool mesh );

  /*
   * [한국어]
   * RegisterRoutingFunctions - KNCube 전용 라우팅 함수를 전역 라우팅 테이블에 등록한다.
   *
   * @return: 없음.
   *
   * 현재 구현은 비어 있음(empty body). k-ary n-cube용 XY 라우팅 등의 함수가
   * 여기에 등록될 예정이었으나 intersim2에서 별도 라우팅 파일로 분리된 것으로 보인다.
   * Network::RegisterRoutingFunctions() 호출 체인에서 이 함수가 불릴 수 있다.
   *
   * 호출 체인:
   *   booksim_main / Network 초기화 → [RegisterRoutingFunctions()]
   */
  static void RegisterRoutingFunctions();

  /*
   * [한국어]
   * GetN - 차원 수(_n)를 반환하는 읽기 전용 접근자.
   *
   * @return: _n (차원 수, 예: 2D 메시이면 2).
   *
   * 라우팅 알고리즘이나 통계 수집 코드에서 차원 수를 조회할 때 사용.
   * const 함수이므로 상태를 변경하지 않는다.
   *
   * 호출 체인:
   *   라우팅 함수 / 외부 통계 코드 → [GetN()]
   */
  int GetN( ) const;

  /*
   * [한국어]
   * GetK - 차원당 라우터 수(_k)를 반환하는 읽기 전용 접근자.
   *
   * @return: _k (각 차원의 radix, 예: 4x4 메시이면 4).
   *
   * 라우팅 알고리즘이나 통계 수집 코드에서 radix를 조회할 때 사용.
   * const 함수이므로 상태를 변경하지 않는다.
   *
   * 호출 체인:
   *   라우팅 함수 / 외부 통계 코드 → [GetK()]
   */
  int GetK( ) const;

  /*
   * [한국어]
   * Capacity - 네트워크의 이론적 처리 용량을 반환한다.
   *
   * @return: double. 메시이면 _k/8.0, 토러스이면 _k/4.0.
   *          이 값은 bisection bandwidth 등 이론적 지표를 나타내며,
   *          시뮬레이션 설정이나 통계 보고에서 참조된다.
   *
   * 반환 공식의 근거: k-ary n-cube에서 bisection link 수를 노드 수로 정규화한 값.
   * torus는 양 방향 wrap-around 링크가 있어 mesh 대비 2배 용량.
   *
   * 호출 체인:
   *   Network 통계/검증 코드 → [Capacity()]
   */
  double Capacity( ) const;

  /*
   * [한국어]
   * InsertRandomFaults - 랜덤 링크 장애를 네트워크에 삽입한다 (레거시 코드).
   *
   * @config: GPGPU-Sim 설정 객체. "link_failures"(장애 수), "fail_seed"(난수 시드) 옵션 사용.
   * @return: 없음.
   *
   * 레거시(legacy) 코드로, 현재 일반적인 GPU NoC 시뮬레이션에서는 사용되지 않는다.
   * 에지(edge) 노드(차원 경계 노드)를 제외한 내부 노드에서 랜덤하게 채널 장애를
   * OutChannelFault()로 설정하여 장애 허용(fault-tolerance) 시나리오를 시뮬레이션한다.
   * 장애가 삽입된 노드와 그 이웃 노드는 fail_nodes 배열로 추적하여 중복 삽입을 방지한다.
   *
   * 호출 체인:
   *   Network 초기화 / 장애 주입 실험 코드 → [InsertRandomFaults()]
   *   → OutChannelFault(node, chan) [Network 기반 클래스]
   */
  void InsertRandomFaults( const Configuration &config );

};

#endif
