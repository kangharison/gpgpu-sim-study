// $Id: network.cpp 5188 2012-08-30 00:31:31Z dub $

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

/*network.cpp
 *
 *This class is the basis of the entire network, it contains, all the routers
 *channels in the network, and is extended by all the network topologies
 *
 */

/*
 * [한국어 설명] NoC(네트워크온칩) 기반 클래스 구현 (network.cpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 BookSim2 NoC 시뮬레이터의 모든 토폴로지 구현이 상속하는 추상 기반 클래스
 * Network의 핵심 메서드들을 구현한다. Network 클래스는 라우터(Router) 객체들의 배열,
 * 주입(inject)/방출(eject) 채널, 라우터 간 내부 채널을 포함하며, 이들을 생성·관리·소멸시키는
 * 책임을 진다. 구체적인 토폴로지(mesh, torus, fattree 등)는 이 클래스를 상속받아 라우터
 * 연결 구조만 특화하면 되므로, 채널 생성·사이클 구동·트래픽 주입/수신 로직은 여기서
 * 모두 공통 처리된다. gpgpusim.config의 "topology" 키로 사용할 토폴로지를 선택한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 NoC 계층은 icnt_wrapper.cc가 intersim2 라이브러리를 감싸는 구조이다:
 *
 *   gpgpu-sim/icnt_wrapper.cc          ← SM↔L2 트래픽 주입/수신 진입점
 *     → intersim2/networks/network.cpp  ← 이 파일 (토폴로지 공통 인터페이스)
 *         → intersim2/networks/kncube.cpp, fattree.cpp 등 (구체 토폴로지)
 *             → intersim2/routers/*.cpp (라우터 단위 사이클 처리)
 *                 → intersim2/channels/flitchannel.cpp (FlitChannel: 레이턴시 FIFO)
 *                 → intersim2/channels/creditchannel.cpp (CreditChannel: 백프레셔 신호)
 *
 * 매 사이클마다 icnt_wrapper.cc가 ReadInputs → Evaluate → WriteOutputs 3단계를 순서대로
 * 호출하며, 이 파일의 각 메서드는 _timed_modules 큐에 등록된 모든 FlitChannel,
 * CreditChannel, Router에 그 호출을 전파한다. 실행 컨텍스트: CPU 호스트 스레드 (시뮬레이터
 * 메인 루프), GPU 디바이스 코드와는 무관.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - booksim.hpp / network.hpp: 기반 타입(Flit, Credit, TimedModule, Configuration)
 *   - kncube.hpp, fly.hpp, cmesh.hpp, flatfly_onchip.hpp, qtree.hpp, tree4.hpp,
 *     fattree.hpp, anynet.hpp, dragonfly.hpp: New()에서 토폴로지별 서브클래스 생성
 *   - FlitChannel (flitchannel.hpp): 레이턴시가 있는 플릿(flit) 전달 FIFO 채널
 *   - CreditChannel (creditchannel.hpp): 백프레셔(backpressure) 크레딧 전달 채널
 *   - Router (router.hpp): 각 노드(라우터)의 사이클별 패킷 라우팅
 *   - Configuration: gpgpusim.config 파싱 결과를 담은 설정 객체
 *
 * 이 모듈에 의존하는 모듈:
 *   - src/gpgpu-sim/icnt_wrapper.cc: intersim2_*() 래퍼 함수들이 Network 포인터를 통해
 *     WriteFlit/ReadFlit/ReadInputs/Evaluate/WriteOutputs를 호출
 *   - 모든 구체 토폴로지 클래스(KNCube, CMesh, KNFly, ...): 이 클래스를 public 상속
 *
 * 데이터 흐름:
 *   SM 코어 → icnt_wrapper::intersim2_send() → Network::WriteFlit() → _inject[s]->Send()
 *   → FlitChannel FIFO → 라우터 입력 포트 → 라우팅/스케줄링 → 라우터 출력 포트
 *   → _eject[d]->Receive() → Network::ReadFlit() → icnt_wrapper → L2 캐시 / 메모리 컨트롤러
 *
 * 공유 전역 변수:
 *   - gNodes (booksim.hpp): 네트워크 전체 노드 수. _Alloc()에서 _nodes로 초기화됨.
 *     라우팅 함수들이 이를 참조하므로 _Alloc() 이전에 사용해서는 안 된다.
 *
 * === 주요 함수/구조체 요약 ===
 * Network::Network()     - 기반 멤버 초기화(-1 센티널), classes 설정값 로드
 * Network::~Network()    - 라우터·채널 동적 객체 전체 해제 (메모리 누수 방지)
 * Network::New()         - "topology" 문자열로 구체 토폴로지 서브클래스를 생성·반환하는 팩토리
 * Network::_Alloc()      - FlitChannel/CreditChannel 벡터를 생성하고 _timed_modules에 등록
 * Network::ReadInputs()  - 사이클 3단계 1: 모든 timed_module의 입력을 래치(latch)
 * Network::Evaluate()    - 사이클 3단계 2: 라우터/채널 로직 연산 수행
 * Network::WriteOutputs()- 사이클 3단계 3: 연산 결과를 출력 레지스터에 반영
 * Network::WriteFlit()   - SM이 NoC에 플릿을 주입하는 외부 인터페이스
 * Network::ReadFlit()    - 목적지 노드가 NoC에서 플릿을 수신하는 외부 인터페이스
 */

#include <cassert>  // [한국어] assert() — 런타임 전제조건 검사용. 잘못된 노드 인덱스 등을 즉시 중단시킴
#include <sstream>  // [한국어] ostringstream — 채널 이름 문자열을 동적으로 조립할 때 사용

#include "booksim.hpp"          // [한국어] BookSim2 공통 타입 및 전역 변수 (gNodes, Flit, Credit 등) 정의
#include "network.hpp"          // [한국어] Network 클래스 및 TimedModule 기반 클래스 선언

#include "kncube.hpp"           // [한국어] k-ary n-cube 토폴로지 (mesh/torus): KNCube 클래스
#include "fly.hpp"              // [한국어] k-ary n-fly 버터플라이(Butterfly) 토폴로지: KNFly 클래스
#include "cmesh.hpp"            // [한국어] 집중(Concentrated) 메쉬 토폴로지: CMesh 클래스
#include "flatfly_onchip.hpp"   // [한국어] FlatFly 온칩 토폴로지 (GPU NoC에 자주 사용): FlatFlyOnChip 클래스
#include "qtree.hpp"            // [한국어] 쿼드트리(QuadTree) 토폴로지: QTree 클래스
#include "tree4.hpp"            // [한국어] 4-ary 트리 토폴로지: Tree4 클래스
#include "fattree.hpp"          // [한국어] Fat-Tree 토폴로지 (데이터센터 네트워크에서도 사용): FatTree 클래스
#include "anynet.hpp"           // [한국어] 임의(Arbitrary) 토폴로지 (파일 기술 방식): AnyNet 클래스
#include "dragonfly.hpp"        // [한국어] Dragonfly 토폴로지 (고성능 HPC 인터커넥트): DragonFlyNew 클래스


/*
 * [한국어]
 * Network::Network - Network 기반 클래스 생성자: 멤버 변수를 미초기화 센티널(-1)로 설정
 *
 * @config: gpgpusim.config 파싱 결과 객체. "classes" 키(트래픽 클래스 수)를 읽음.
 * @name:   이 네트워크 모듈의 문자열 식별자. TimedModule 계층에서 디버그 출력·로깅에 사용됨.
 * @return: (생성자, 반환값 없음)
 *
 * _size, _nodes, _channels를 -1로 초기화하는 이유: 서브클래스 생성자가 이 기반 생성자를
 * 호출한 직후 토폴로지 계산으로 올바른 값을 채우고, 그 뒤 _Alloc()을 호출해야 하는 설계
 * 계약(design contract)을 강제하기 위해서다. _Alloc() 내부의 assert()가 -1이 남아 있으면
 * 즉시 프로그램을 중단시켜 서브클래스가 초기화를 빠뜨린 버그를 조기에 포착한다.
 * _classes는 클래스별 통계 추적(FlitChannel 내부 per-class 카운터)에 필요하므로 여기서 로드.
 *
 * 실행 컨텍스트: icnt_wrapper_init() → Network::New() → 각 서브클래스 생성자 → 이 생성자
 * 순서로 호출됨. 단일 스레드 초기화 경로이며 동기화 불필요.
 *
 * 호출 체인:
 *   icnt_wrapper_init() → Network::New() → KNCube::KNCube() → Network::Network()
 */
Network::Network( const Configuration &config, const string & name ) :
  TimedModule( 0, name )  // [한국어] TimedModule 기반 생성자: 지연(latency) 0사이클, 모듈 이름 등록
{
  _size     = -1;  // [한국어] 라우터(스위치) 총 개수. 서브클래스 생성자가 토폴로지 크기 계산 후 덮어씀
  _nodes    = -1;  // [한국어] 엔드포인트(endpoint) 노드 총 개수 = SM 수 + L2 슬라이스 수. 서브클래스가 설정
  _channels = -1;  // [한국어] 라우터 간 내부 채널(링크) 총 개수. 서브클래스가 토폴로지 계산 후 설정
  _classes  = config.GetInt("classes");  // [한국어] 트래픽 클래스 수 (QoS 구분). 보통 1. FlitChannel의 per-class 통계 배열 크기 결정
}

/*
 * [한국어]
 * Network::~Network - Network 기반 클래스 소멸자: 동적 할당된 모든 채널·라우터 객체 해제
 *
 * @return: (소멸자, 반환값 없음)
 *
 * _routers, _inject, _inject_cred, _eject, _eject_cred, _chan, _chan_cred 벡터에
 * 저장된 포인터들은 모두 _Alloc() 또는 서브클래스 Build() 과정에서 new로 생성됐으므로
 * 여기서 delete해야 한다. nullptr 체크(if 조건)는 서브클래스가 일부 슬롯을 채우지 않은
 * 경우(예: 오류 상황 또는 희소 그래프)에 이중 해제(double-free)를 방지한다.
 * _timed_modules 큐는 포인터만 보관하며 소유권이 없으므로 별도 해제하지 않는다.
 *
 * 실행 컨텍스트: 시뮬레이터 종료 시 단일 스레드에서 호출. 동기화 불필요.
 *
 * 호출 체인:
 *   icnt_wrapper_init()의 스코프 종료 또는 프로그램 종료 → Network::~Network()
 */
Network::~Network( )
{
  for ( int r = 0; r < _size; ++r ) {  // [한국어] 라우터 배열 순회: 인덱스 0부터 (_size - 1)까지
    if ( _routers[r] ) delete _routers[r];  // [한국어] null이 아닐 때만 삭제 — 미초기화 슬롯 방어
  }
  for ( int s = 0; s < _nodes; ++s ) {  // [한국어] 주입(inject) 측 채널 순회: 노드 수만큼
    if ( _inject[s] ) delete _inject[s];            // [한국어] 노드 s의 FlitChannel 주입 채널 해제
    if ( _inject_cred[s] ) delete _inject_cred[s];  // [한국어] 노드 s의 CreditChannel 주입 크레딧 채널 해제
  }
  for ( int d = 0; d < _nodes; ++d ) {  // [한국어] 방출(eject) 측 채널 순회: 노드 수만큼
    if ( _eject[d] ) delete _eject[d];              // [한국어] 노드 d의 FlitChannel 방출 채널 해제
    if ( _eject_cred[d] ) delete _eject_cred[d];    // [한국어] 노드 d의 CreditChannel 방출 크레딧 채널 해제
  }
  for ( int c = 0; c < _channels; ++c ) {  // [한국어] 라우터 간 내부 채널 순회: 총 채널 수만큼
    if ( _chan[c] ) delete _chan[c];            // [한국어] 내부 FlitChannel(라우터 간 데이터 링크) 해제
    if ( _chan_cred[c] ) delete _chan_cred[c];  // [한국어] 내부 CreditChannel(라우터 간 백프레셔 링크) 해제
  }
}

/*
 * [한국어]
 * Network::New - 토폴로지 문자열로 구체 Network 서브클래스를 생성하는 정적 팩토리 함수
 *
 * @config: gpgpusim.config 파싱 결과 객체. "topology" 키로 토폴로지 종류를,
 *          "link_failures" 키로 링크 장애 삽입 여부를 결정.
 * @name:   생성될 네트워크 모듈의 문자열 식별자 (TimedModule 계층 로깅용).
 * @return: 생성된 구체 Network 서브클래스 포인터. 알 수 없는 토폴로지이면 NULL.
 *          호출자(icnt_wrapper_init)가 소유권을 가지며, 시뮬레이터 종료 시 delete 책임.
 *
 * 이 함수가 왜 필요한가: 토폴로지 선택 로직을 한 곳에 집중시켜, 상위 레이어(icnt_wrapper)가
 * 토폴로지 구체 클래스를 몰라도 되게 한다 (의존성 역전 원칙). 새로운 토폴로지를 추가할 때
 * 이 함수에 분기만 추가하면 되고 상위 레이어는 수정 불필요.
 *
 * 각 토폴로지별 RegisterRoutingFunctions() 호출 이유: 라우팅 함수는 전역 맵에 등록되는
 * 방식으로 동작한다. 라우터 객체가 생성되기 전에 반드시 등록되어야 하므로, 서브클래스 생성자
 * 호출 직전에 호출한다. 각 서브클래스가 지원하는 라우팅 알고리즘(최단경로, adaptive 등)
 * 이름 → 함수 포인터 매핑을 등록한다.
 *
 * link_failures 처리: 레거시 기능으로, 랜덤 링크 장애를 삽입해 내결함성 라우팅을 테스트.
 * 현재 대부분의 토폴로지에서 InsertRandomFaults()가 구현되어 있지 않아 Error()로 중단됨.
 *
 * 실행 컨텍스트: 시뮬레이터 초기화 단계(단일 스레드). 동기화 불필요.
 *
 * 호출 체인:
 *   icnt_wrapper_init() → Network::New() → [각 서브클래스 생성자] → Network::Network()
 *                                        → [서브클래스]::Build() → Network::_Alloc()
 */
Network * Network::New(const Configuration & config, const string & name)
{
  const string topo = config.GetStr( "topology" );  // [한국어] gpgpusim.config의 "topology" 값 읽기 (예: "mesh", "fattree", "flatfly")
  Network * n = NULL;  // [한국어] 결과 포인터 초기화 — 알 수 없는 토폴로지일 경우 NULL 반환 보장
  if ( topo == "torus" ) {
    // [한국어] k-ary n-cube 토러스(torus): 양방향 링이 차원별로 존재, 래핑(wrap-around) 연결 있음
    KNCube::RegisterRoutingFunctions() ;  // [한국어] torus용 라우팅 알고리즘(DOR, adaptive 등)을 전역 맵에 등록
    n = new KNCube( config, name, false );  // [한국어] false = torus 모드 (mesh이면 true). 래핑 연결 생성
  } else if ( topo == "mesh" ) {
    // [한국어] k-ary n-cube 메쉬: 래핑 연결 없는 직사각형(또는 다차원) 그리드 — GPGPU-Sim 기본 NoC 토폴로지
    KNCube::RegisterRoutingFunctions() ;  // [한국어] mesh용 라우팅 함수 등록 (torus와 동일 클래스, 모드만 다름)
    n = new KNCube( config, name, true );  // [한국어] true = mesh 모드. 래핑 없는 단순 격자 연결 생성
  } else if ( topo == "cmesh" ) {
    // [한국어] Concentrated Mesh: 여러 엔드포인트가 하나의 라우터에 집중(concentrate)되는 메쉬 변형
    CMesh::RegisterRoutingFunctions() ;  // [한국어] cmesh 전용 라우팅 함수 등록
    n = new CMesh( config, name );  // [한국어] CMesh 객체 생성 — 집중도(concentration) 파라미터는 config에서 읽음
  } else if ( topo == "fly" ) {
    // [한국어] k-ary n-fly (Butterfly 네트워크): 다단계 상호연결 네트워크, 균일한 bisection bandwidth
    KNFly::RegisterRoutingFunctions() ;  // [한국어] butterfly 라우팅 함수 등록
    n = new KNFly( config, name );  // [한국어] KNFly 객체 생성
  } else if ( topo == "qtree" ) {
    // [한국어] QuadTree 토폴로지: 4진 트리 구조의 계층적 인터커넥트
    QTree::RegisterRoutingFunctions() ;  // [한국어] qtree 라우팅 함수 등록
    n = new QTree( config, name );  // [한국어] QTree 객체 생성
  } else if ( topo == "tree4" ) {
    // [한국어] 4-ary 트리 토폴로지: 각 노드가 최대 4개 자식을 갖는 트리형 네트워크
    Tree4::RegisterRoutingFunctions() ;  // [한국어] tree4 라우팅 함수 등록
    n = new Tree4( config, name );  // [한국어] Tree4 객체 생성
  } else if ( topo == "fattree" ) {
    // [한국어] Fat-Tree 토폴로지: 상위 링크 대역폭이 하위보다 넓은 트리. 데이터센터·HPC에서 범용 사용
    FatTree::RegisterRoutingFunctions() ;  // [한국어] fattree 라우팅 함수 등록
    n = new FatTree( config, name );  // [한국어] FatTree 객체 생성
  } else if ( topo == "flatfly" ) {
    // [한국어] FlatFly 온칩 토폴로지: GPU 내부 NoC에 최적화된 저직경(low-diameter) 플랫 버터플라이 변형
    FlatFlyOnChip::RegisterRoutingFunctions() ;  // [한국어] flatfly 라우팅 함수 등록
    n = new FlatFlyOnChip( config, name );  // [한국어] FlatFlyOnChip 객체 생성
  } else if ( topo == "anynet"){
    // [한국어] AnyNet: 사용자 정의 파일로 임의 토폴로지를 기술하는 범용 백엔드
    AnyNet::RegisterRoutingFunctions() ;  // [한국어] anynet 라우팅 함수 등록
    n = new AnyNet(config, name);  // [한국어] AnyNet 객체 생성 — 연결 파일 경로는 config에서 읽음
  } else if ( topo == "dragonflynew"){
    // [한국어] Dragonfly(신형): 글로벌 링크와 로컬 링크의 2계층 구조 — 초대규모 HPC 인터커넥트
    DragonFlyNew::RegisterRoutingFunctions() ;  // [한국어] dragonfly 라우팅 함수 등록
    n = new DragonFlyNew(config, name);  // [한국어] DragonFlyNew 객체 생성
  } else {
    cerr << "Unknown topology: " << topo << endl;  // [한국어] 알 수 없는 토폴로지 이름 — 표준 에러 출력 후 n은 NULL 유지
  }

  /*legacy code that insert random faults in the networks
   *not sure how to use this
   */
  // [한국어] 레거시 랜덤 링크 장애 삽입 기능: "link_failures" > 0이면 노드 생성 후 랜덤으로 링크를 비활성화
  if ( n && ( config.GetInt( "link_failures" ) > 0 ) ) {
    // [한국어] n이 NULL이 아닌지 먼저 확인 (알 수 없는 토폴로지일 때 크래시 방지)
    n->InsertRandomFaults( config );  // [한국어] 대부분의 토폴로지에서 구현 안 됨 → Error()로 즉시 중단
  }
  return n;  // [한국어] 생성된 구체 Network 서브클래스 포인터 반환. 실패 시 NULL
}

/*
 * [한국어]
 * Network::_Alloc - FlitChannel·CreditChannel 배열을 할당하고 _timed_modules 큐에 등록
 *
 * @return: (void) 내부 상태 변경만 수행. 실패 시 assert()로 즉시 중단.
 *
 * 이 함수가 왜 필요한가: BookSim2 초기 설계에서는 채널을 단순 플릿 배열(1개짜리 FIFO)로
 * 구현했다. 링크 레이턴시를 정확히 모델링하기 위해 FlitChannel 클래스가 추가됐으며, 이는
 * 깊이(depth = 채널 레이턴시 사이클 수)를 가진 FIFO로 매 사이클마다 한 칸씩 시프트한다.
 * CreditChannel은 수신 측이 송신 측에 버퍼 여유를 알려주는 백프레셔 신호 채널이다.
 *
 * 채널 종류:
 *   1. _inject[s]      : 노드 s에서 NoC로 들어오는 FlitChannel (주입 방향 데이터)
 *   2. _inject_cred[s] : 노드 s로 나가는 CreditChannel (주입 방향 백프레셔)
 *   3. _eject[d]       : NoC에서 노드 d로 나가는 FlitChannel (방출 방향 데이터)
 *   4. _eject_cred[d]  : 노드 d에서 오는 CreditChannel (방출 방향 백프레셔)
 *   5. _chan[c]         : 라우터 간 내부 FlitChannel (라우팅 경로 데이터)
 *   6. _chan_cred[c]    : 라우터 간 내부 CreditChannel (라우팅 경로 백프레셔)
 *
 * _timed_modules 큐에 등록하는 이유: ReadInputs/Evaluate/WriteOutputs 3단계가 이 큐를
 * 순회하며 각 모듈의 동명 메서드를 호출한다. 등록 순서가 곧 사이클 내 업데이트 순서이므로
 * 데이터 전달 방향과 일치하게 등록해야 한다.
 *
 * 전제조건: 서브클래스 Build()가 _size, _nodes, _channels를 올바른 값으로 채운 뒤 호출.
 * assert()가 -1 센티널을 검사하여 이 전제조건을 강제한다.
 *
 * 실행 컨텍스트: 시뮬레이션 초기화 단계 (단일 스레드). 동기화 불필요.
 *
 * 호출 체인:
 *   Network::New() → [서브클래스]::Build() → Network::_Alloc()
 *   이후: icnt_wrapper.cc가 매 사이클 ReadInputs/Evaluate/WriteOutputs 호출
 */
void Network::_Alloc( )
{
  assert( ( _size != -1 ) &&
	  ( _nodes != -1 ) &&
	  ( _channels != -1 ) );
  // [한국어] 세 값이 모두 서브클래스에 의해 설정됐는지 검사 — 어느 하나라도 -1이면 서브클래스 초기화 버그

  _routers.resize(_size);  // [한국어] 라우터 포인터 벡터를 _size 크기로 확장. 서브클래스 Build()가 각 슬롯에 new Router 할당
  gNodes = _nodes;  // [한국어] 전역 변수 gNodes 설정 — BookSim2 라우팅 함수들이 노드 수를 이 전역 변수로 참조

  /*booksim used arrays of flits as the channels which makes have capacity of
   *one. To simulate channel latency, flitchannel class has been added
   *which are fifos with depth = channel latency and each cycle the channel
   *shifts by one
   *credit channels are the necessary counter part
   */
  // [한국어] 주입(ingress) 채널 할당 — 각 엔드포인트 노드 s에 대해 FlitChannel + CreditChannel 쌍을 생성
  _inject.resize(_nodes);       // [한국어] 주입 FlitChannel 포인터 벡터를 _nodes 크기로 확장
  _inject_cred.resize(_nodes);  // [한국어] 주입 CreditChannel 포인터 벡터를 _nodes 크기로 확장
  for ( int s = 0; s < _nodes; ++s ) {  // [한국어] 노드 0부터 (_nodes - 1)까지 순회하며 주입 채널 쌍 생성
    ostringstream name;  // [한국어] 채널 이름 문자열 스트림 — 로깅·디버깅용 고유 이름 생성
    name << Name() << "_fchan_ingress" << s;  // [한국어] 예: "net0_fchan_ingress3" — 네트워크 이름 + 주입 FlitChannel + 노드 번호
    _inject[s] = new FlitChannel(this, name.str(), _classes);  // [한국어] 레이턴시 FIFO FlitChannel 생성. this=부모 모듈, _classes=클래스 수
    _inject[s]->SetSource(NULL, s);  // [한국어] 소스를 NULL(외부 엔드포인트)로, 소스 포트를 노드 번호 s로 설정
    _timed_modules.push_back(_inject[s]);  // [한국어] 사이클 구동 큐에 등록 — ReadInputs/Evaluate/WriteOutputs가 이 채널을 처리
    name.str("");  // [한국어] ostringstream 버퍼 초기화 — 다음 이름 생성 전 비워야 함
    name << Name() << "_cchan_ingress" << s;  // [한국어] 예: "net0_cchan_ingress3" — 주입 CreditChannel 이름
    _inject_cred[s] = new CreditChannel(this, name.str());  // [한국어] 백프레셔 크레딧 채널 생성 (클래스 구분 없음, 크레딧은 단일 카운터)
    _timed_modules.push_back(_inject_cred[s]);  // [한국어] 크레딧 채널도 사이클 큐에 등록 — 매 사이클 크레딧 전파
  }
  // [한국어] 방출(egress) 채널 할당 — 각 엔드포인트 노드 d에 대해 FlitChannel + CreditChannel 쌍을 생성
  _eject.resize(_nodes);        // [한국어] 방출 FlitChannel 포인터 벡터를 _nodes 크기로 확장
  _eject_cred.resize(_nodes);   // [한국어] 방출 CreditChannel 포인터 벡터를 _nodes 크기로 확장
  for ( int d = 0; d < _nodes; ++d ) {  // [한국어] 노드 0부터 (_nodes - 1)까지 순회하며 방출 채널 쌍 생성
    ostringstream name;  // [한국어] 채널 이름 조립용 스트림 재사용
    name << Name() << "_fchan_egress" << d;  // [한국어] 예: "net0_fchan_egress5" — 방출 FlitChannel 이름
    _eject[d] = new FlitChannel(this, name.str(), _classes);  // [한국어] 방출 방향 레이턴시 FIFO FlitChannel 생성
    _eject[d]->SetSink(NULL, d);  // [한국어] 싱크를 NULL(외부 엔드포인트)로, 싱크 포트를 노드 번호 d로 설정
    _timed_modules.push_back(_eject[d]);  // [한국어] 사이클 구동 큐에 등록
    name.str("");  // [한국어] ostringstream 버퍼 초기화
    name << Name() << "_cchan_egress" << d;  // [한국어] 예: "net0_cchan_egress5" — 방출 CreditChannel 이름
    _eject_cred[d] = new CreditChannel(this, name.str());  // [한국어] 방출 방향 백프레셔 크레딧 채널 생성
    _timed_modules.push_back(_eject_cred[d]);  // [한국어] 사이클 큐에 등록
  }
  // [한국어] 라우터 간 내부 채널 할당 — 토폴로지 링크마다 FlitChannel + CreditChannel 쌍을 생성
  _chan.resize(_channels);       // [한국어] 내부 FlitChannel 포인터 벡터를 _channels 크기로 확장
  _chan_cred.resize(_channels);  // [한국어] 내부 CreditChannel 포인터 벡터를 _channels 크기로 확장
  for ( int c = 0; c < _channels; ++c ) {  // [한국어] 채널 0부터 (_channels - 1)까지 순회하며 내부 채널 쌍 생성
    ostringstream name;  // [한국어] 채널 이름 조립용 스트림
    name << Name() << "_fchan_" << c;  // [한국어] 예: "net0_fchan_12" — 내부 FlitChannel 이름
    _chan[c] = new FlitChannel(this, name.str(), _classes);  // [한국어] 라우터 간 레이턴시 FIFO FlitChannel 생성
    _timed_modules.push_back(_chan[c]);  // [한국어] 사이클 큐에 등록 — 라우터 간 데이터 전파에 참여
    name.str("");  // [한국어] ostringstream 버퍼 초기화
    name << Name() << "_cchan_" << c;  // [한국어] 예: "net0_cchan_12" — 내부 CreditChannel 이름
    _chan_cred[c] = new CreditChannel(this, name.str());  // [한국어] 라우터 간 백프레셔 크레딧 채널 생성
    _timed_modules.push_back(_chan_cred[c]);  // [한국어] 사이클 큐에 등록 — 크레딧 역방향 전파에 참여
  }
}

/*
 * [한국어]
 * Network::ReadInputs - 사이클 3단계 중 1단계: 모든 TimedModule의 입력값을 래치(latch)
 *
 * @return: (void)
 *
 * BookSim2는 디지털 회로 시뮬레이션 모델을 따른다. 매 사이클은 3단계로 분리된다:
 *   1. ReadInputs  — 이전 사이클의 출력 레지스터 값을 현재 입력 레지스터로 복사(래치)
 *   2. Evaluate    — 입력 레지스터 값으로 다음 상태 연산
 *   3. WriteOutputs— 연산 결과를 출력 레지스터에 기록
 * 이 3단계 분리 덕분에 같은 사이클 내 여러 모듈 간 업데이트 순서에 관계없이 일관된
 * 결과를 얻는다 (순서 의존성 제거). 이 함수는 _timed_modules 큐의 모든 채널과 라우터에
 * ReadInputs를 전파한다.
 *
 * 실행 컨텍스트: icnt_wrapper.cc의 intersim2_advance_time() (매 NoC 사이클 1회 호출).
 * 단일 스레드. 동기화 불필요.
 *
 * 호출 체인:
 *   icnt_wrapper.cc::intersim2_advance_time()
 *     → Network::ReadInputs()
 *         → [각 FlitChannel/CreditChannel/Router]::ReadInputs()
 */
void Network::ReadInputs( )
{
  for(deque<TimedModule *>::const_iterator iter = _timed_modules.begin();
      iter != _timed_modules.end();
      ++iter) {
    // [한국어] _timed_modules 큐를 처음부터 끝까지 순회하며 각 모듈의 ReadInputs() 호출
    // [한국어] 큐는 _Alloc()에서 채널 등록 순서대로 구성됨: inject 채널 → eject 채널 → 내부 채널
    (*iter)->ReadInputs( );  // [한국어] 포인터 역참조 후 가상 함수 ReadInputs() 디스패치 — 채널이면 FIFO 앞단 값 래치
  }
}

/*
 * [한국어]
 * Network::Evaluate - 사이클 3단계 중 2단계: 모든 TimedModule의 상태 연산 수행
 *
 * @return: (void)
 *
 * ReadInputs()로 래치된 입력을 바탕으로 각 모듈이 다음 출력값을 계산한다.
 * FlitChannel은 FIFO를 한 칸 시프트하는 연산을, 라우터는 플릿 라우팅·스케줄링 결정을 수행.
 * CreditChannel은 크레딧 카운터 갱신을 처리한다. 이 단계가 NoC 시뮬레이션의 핵심 연산부.
 *
 * 실행 컨텍스트: ReadInputs() 직후 동일 사이클 내에서 호출. 단일 스레드.
 *
 * 호출 체인:
 *   icnt_wrapper.cc::intersim2_advance_time()
 *     → Network::ReadInputs() (완료 후)
 *     → Network::Evaluate()
 *         → [각 FlitChannel/CreditChannel/Router]::Evaluate()
 */
void Network::Evaluate( )
{
  for(deque<TimedModule *>::const_iterator iter = _timed_modules.begin();
      iter != _timed_modules.end();
      ++iter) {
    // [한국어] _timed_modules 큐 전체 순회 — ReadInputs와 동일한 순서로 Evaluate() 전파
    (*iter)->Evaluate( );  // [한국어] 가상 함수 Evaluate() 디스패치 — FlitChannel FIFO 시프트, 라우터 스케줄링 실행
  }
}

/*
 * [한국어]
 * Network::WriteOutputs - 사이클 3단계 중 3단계: 연산 결과를 출력 레지스터에 기록
 *
 * @return: (void)
 *
 * Evaluate()에서 계산된 다음 상태 값을 실제 출력 레지스터(또는 공유 버퍼)에 기록한다.
 * 이 단계 이후 다음 사이클의 ReadInputs()가 이 출력값을 읽어가게 된다.
 * 3단계가 모두 완료되어야 비로소 1 NoC 사이클이 종료된다.
 *
 * 실행 컨텍스트: Evaluate() 직후 동일 사이클 내에서 호출. 단일 스레드.
 *
 * 호출 체인:
 *   icnt_wrapper.cc::intersim2_advance_time()
 *     → Network::Evaluate() (완료 후)
 *     → Network::WriteOutputs()
 *         → [각 FlitChannel/CreditChannel/Router]::WriteOutputs()
 */
void Network::WriteOutputs( )
{
  for(deque<TimedModule *>::const_iterator iter = _timed_modules.begin();
      iter != _timed_modules.end();
      ++iter) {
    // [한국어] _timed_modules 큐 전체 순회 — WriteOutputs() 전파로 1 사이클 완결
    (*iter)->WriteOutputs( );  // [한국어] 가상 함수 WriteOutputs() 디스패치 — FIFO 출력단 갱신, 라우터 출력 포트 업데이트
  }
}

/*
 * [한국어]
 * Network::WriteFlit - SM 코어(또는 L2 슬라이스)가 NoC에 플릿(flit)을 주입하는 인터페이스
 *
 * @f:      주입할 Flit 포인터. 헤더·바디·테일 플릿 구분 없이 동일 인터페이스 사용.
 *          호출자가 할당하며, 네트워크 내부에서 라우팅 후 목적지에서 ReadFlit()으로 수신됨.
 * @source: 플릿을 주입하는 소스 노드 번호 (0 이상, _nodes 미만). SM의 경우 SM 번호에 대응.
 * @return: (void)
 *
 * 이 함수는 NoC에 대한 트래픽 주입 진입점이다. 상위 레이어(icnt_wrapper)가 SM→L2 또는
 * SM→SM 패킷을 NoC에 넣을 때 호출한다. 내부적으로는 _inject[source] FlitChannel의
 * Send()를 호출하며, 이 채널이 매 사이클 시프트되어 라우터 입력 포트로 플릿을 전달한다.
 *
 * assert()는 source 범위를 검사한다. 잘못된 source 번호는 배열 범위 초과(out-of-bounds)
 * 접근을 유발하므로 개발 시 즉시 포착할 수 있도록 assert로 방어.
 *
 * 실행 컨텍스트: icnt_wrapper.cc::intersim2_send()가 호출. NoC 사이클 진행 전 또는 후에
 * 언제든 호출 가능. 단일 스레드(GPGPU-Sim 메인 시뮬레이션 루프 내).
 *
 * 호출 체인:
 *   SM 파이프라인 → icnt_wrapper.cc::intersim2_send()
 *     → Network::WriteFlit()
 *         → FlitChannel::Send() → FIFO 큐에 플릿 삽입
 */
void Network::WriteFlit( Flit *f, int source )
{
  assert( ( source >= 0 ) && ( source < _nodes ) );  // [한국어] source가 유효 범위[0, _nodes)인지 검사 — 범위 위반 즉시 중단
  _inject[source]->Send(f);  // [한국어] 소스 노드 source의 주입 FlitChannel에 플릿 f를 삽입 — 다음 사이클부터 FIFO를 통해 전파 시작
}

/*
 * [한국어]
 * Network::ReadFlit - 목적지 노드가 NoC에서 도착한 플릿을 수신하는 인터페이스
 *
 * @dest:   플릿을 수신할 목적지 노드 번호 (0 이상, _nodes 미만). L2 슬라이스의 경우 슬라이스 번호.
 * @return: 도착한 Flit 포인터. 이 사이클에 도착한 플릿이 없으면 NULL.
 *          호출자가 반환값을 NULL 체크 후 처리해야 함.
 *
 * 이 함수는 NoC 출력 수신 진입점이다. 상위 레이어(icnt_wrapper)가 L2→SM 또는 목적지 노드로의
 * 플릿 도착 여부를 확인할 때 호출한다. _eject[dest] FlitChannel의 Receive()를 통해
 * FIFO 출력단에서 플릿을 꺼낸다. 도착 플릿이 없으면 NULL 반환.
 *
 * 실행 컨텍스트: icnt_wrapper.cc::intersim2_receive()가 호출. WriteOutputs() 이후 호출해야
 * 현재 사이클의 도착 플릿을 올바르게 수신한다. 단일 스레드.
 *
 * 호출 체인:
 *   icnt_wrapper.cc::intersim2_receive()
 *     → Network::ReadFlit()
 *         → FlitChannel::Receive() → FIFO 출력단에서 플릿 반환
 *     → L2 캐시 / 메모리 컨트롤러로 데이터 전달
 */
Flit *Network::ReadFlit( int dest )
{
  assert( ( dest >= 0 ) && ( dest < _nodes ) );  // [한국어] dest가 유효 범위[0, _nodes)인지 검사 — 범위 위반 즉시 중단
  return _eject[dest]->Receive();  // [한국어] 목적지 노드 dest의 방출 FlitChannel에서 플릿 수신. 도착 플릿 없으면 NULL 반환
}

/*
 * [한국어]
 * Network::WriteCredit - 수신 측(목적지 노드)이 송신 측에 버퍼 여유를 알리는 크레딧 전송
 *
 * @c:    전송할 Credit 포인터. 크레딧 수(슬롯 여유)를 인코딩한 객체.
 * @dest: 크레딧을 수신할 목적지 노드 번호 (0 이상, _nodes 미만).
 *        (역방향 채널이므로, "dest"는 원래 데이터 흐름의 목적지 = 크레딧의 발원지)
 * @return: (void)
 *
 * 크레딧(credit) 기반 흐름 제어(flow control): 수신 버퍼 슬롯이 비면 크레딧을 송신 측으로
 * 역방향 전송한다. 송신 측은 크레딧이 있을 때만 플릿을 보낼 수 있으므로 버퍼 오버플로우를
 * 방지한다. _eject_cred[dest]는 방출 방향의 역방향 크레딧 채널이다.
 *
 * 실행 컨텍스트: icnt_wrapper.cc 또는 라우터 로직이 호출. 단일 스레드.
 *
 * 호출 체인:
 *   수신 노드 버퍼 소비 → Network::WriteCredit()
 *     → CreditChannel::Send() → 크레딧 역방향 전파
 */
void Network::WriteCredit( Credit *c, int dest )
{
  assert( ( dest >= 0 ) && ( dest < _nodes ) );  // [한국어] dest가 유효 범위[0, _nodes)인지 검사
  _eject_cred[dest]->Send(c);  // [한국어] 방출 방향 크레딧 채널(역방향)에 크레딧 c를 삽입 — 송신 측 버퍼 여유 통보
}

/*
 * [한국어]
 * Network::ReadCredit - 송신 측(소스 노드)이 크레딧 채널에서 크레딧을 수신하는 인터페이스
 *
 * @source: 크레딧을 확인할 소스 노드 번호 (0 이상, _nodes 미만).
 *          (역방향 채널이므로, "source"는 원래 데이터 흐름의 소스 = 크레딧의 수신자)
 * @return: 도착한 Credit 포인터. 이 사이클에 도착한 크레딧이 없으면 NULL.
 *
 * 소스 노드가 플릿을 보내기 전에 크레딧이 있는지 확인하거나, 크레딧 반환을 확인할 때 사용.
 * _inject_cred[source]는 주입 방향의 역방향 크레딧 채널이다.
 *
 * 실행 컨텍스트: icnt_wrapper.cc 또는 라우터 로직이 호출. 단일 스레드.
 *
 * 호출 체인:
 *   소스 노드 흐름 제어 확인 → Network::ReadCredit()
 *     → CreditChannel::Receive() → 크레딧 반환
 */
Credit *Network::ReadCredit( int source )
{
  assert( ( source >= 0 ) && ( source < _nodes ) );  // [한국어] source가 유효 범위[0, _nodes)인지 검사
  return _inject_cred[source]->Receive();  // [한국어] 주입 방향 역방향 크레딧 채널에서 크레딧 수신. 없으면 NULL
}

/*
 * [한국어]
 * Network::InsertRandomFaults - 랜덤 링크 장애를 삽입하는 레거시 기능 (기반 클래스 기본 구현)
 *
 * @config: 장애 삽입 파라미터(링크 장애 수 등)를 담은 설정 객체.
 * @return: (void) — 실제로는 Error()로 프로그램을 즉시 중단시킴.
 *
 * 기반 클래스의 이 구현은 항상 Error()를 호출하여 중단한다. 이는 대부분의 구체 토폴로지가
 * 이 기능을 구현하지 않았음을 명시적으로 알리기 위한 의도적 설계다. 내결함성 라우팅 연구를
 * 위해 특정 토폴로지가 이 메서드를 오버라이드하면 사용 가능.
 *
 * 실행 컨텍스트: Network::New()에서 "link_failures" > 0일 때만 호출.
 *
 * 호출 체인:
 *   Network::New() → Network::InsertRandomFaults() → Error() → 프로그램 중단
 */
void Network::InsertRandomFaults( const Configuration &config )
{
  Error( "InsertRandomFaults not implemented for this topology!" );
  // [한국어] 기반 클래스는 구현 없음을 Error()로 명시. 서브클래스가 오버라이드하지 않으면 항상 여기서 중단
}

/*
 * [한국어]
 * Network::OutChannelFault - 특정 라우터의 출력 채널에 장애 상태를 설정하는 인터페이스
 *
 * @r:     장애를 설정할 라우터 번호 (0 이상, _size 미만).
 * @c:     해당 라우터에서 장애를 설정할 출력 채널(포트) 번호.
 * @fault: true이면 장애 활성화(링크 비활성화), false이면 장애 해제(링크 복구).
 * @return: (void)
 *
 * InsertRandomFaults()와 함께 사용되는 링크 장애 시뮬레이션 인터페이스. 라우터의
 * OutChannelFault()를 위임 호출하여 해당 출력 포트를 비활성화/활성화한다.
 * 장애가 설정된 출력 포트로는 플릿을 전송할 수 없으며, 라우터가 대체 경로를 찾아야 한다.
 *
 * 실행 컨텍스트: 장애 삽입 초기화 단계에서 호출. 단일 스레드.
 *
 * 호출 체인:
 *   InsertRandomFaults() → Network::OutChannelFault()
 *     → _routers[r]->OutChannelFault(c, fault)
 */
void Network::OutChannelFault( int r, int c, bool fault )
{
  assert( ( r >= 0 ) && ( r < _size ) );  // [한국어] 라우터 번호가 유효 범위[0, _size)인지 검사 — 범위 위반 즉시 중단
  _routers[r]->OutChannelFault( c, fault );  // [한국어] 라우터 r의 OutChannelFault() 위임 호출 — 포트 c를 fault 상태로 설정
}

/*
 * [한국어]
 * Network::Capacity - 네트워크의 정규화된 이론적 용량을 반환하는 인터페이스
 *
 * @return: 1.0 (double). 기반 클래스 기본값. 서브클래스가 오버라이드하여 실제 용량 반환 가능.
 *
 * 네트워크 용량은 포화 지점에서의 주입률(injection rate)을 이론적 최대값으로 정규화한 값.
 * 기반 클래스는 1.0(100%)을 반환하며, 복잡한 토폴로지는 bisection bandwidth 등을 기반으로
 * 이 값을 오버라이드할 수 있다. 성능 분석 및 통계 출력에서 참조값으로 사용.
 *
 * 실행 컨텍스트: 시뮬레이션 통계 수집 단계. 읽기 전용, const 함수.
 *
 * 호출 체인:
 *   통계 수집 루틴 → Network::Capacity() → 1.0 반환
 */
double Network::Capacity( ) const
{
  return 1.0;  // [한국어] 이론적 용량 정규화 기본값 1.0 (100%) 반환 — 서브클래스가 필요 시 오버라이드
}

/* this function can be heavily modified to display any information
 * neceesary of the network, by default, call display on each router
 * and display the channel utilization rate
 */
/*
 * [한국어]
 * Network::Display - 모든 라우터의 상태 정보를 출력 스트림에 표시
 *
 * @os:    출력 대상 스트림 (기본: cout, 파일 스트림도 가능).
 * @return: (void)
 *
 * 시뮬레이션 중 또는 종료 후 디버깅 목적으로 각 라우터의 내부 상태(버퍼 점유, 포트 상태 등)를
 * 출력한다. 각 라우터의 Display()를 위임 호출하므로, 출력 형식은 라우터 구현에 따라 다르다.
 * 채널 활용률(channel utilization rate)도 라우터 Display()를 통해 출력될 수 있다.
 *
 * 실행 컨텍스트: 시뮬레이션 루프 외부 또는 디버그 출력 시점. const 함수(상태 변경 없음).
 *
 * 호출 체인:
 *   디버그 출력 루틴 → Network::Display()
 *     → _routers[r]->Display(os) (각 라우터의 가상 함수)
 */
void Network::Display( ostream & os ) const
{
  for ( int r = 0; r < _size; ++r ) {  // [한국어] 라우터 0부터 (_size - 1)까지 순회
    _routers[r]->Display( os );  // [한국어] 라우터 r의 상태 정보를 출력 스트림 os에 기록 — 가상 함수 디스패치
  }
}

/*
 * [한국어]
 * Network::DumpChannelMap - 전체 채널 연결 맵을 CSV 형식으로 덤프
 *
 * @os:     출력 대상 스트림 (파일 또는 cout).
 * @prefix: 각 출력 행 앞에 붙일 접두어 문자열 (들여쓰기나 식별자 구분에 사용).
 * @return: (void)
 *
 * 채널 맵은 "소스 라우터, 소스 포트, 목적지 라우터, 목적지 포트" 형식의 CSV로 출력된다.
 * 외부 노드(엔드포인트)는 라우터 ID -1로 표기된다. 출력 순서:
 *   1. 주입(inject) 채널: 소스 라우터 -1(외부) → 네트워크 진입 라우터
 *   2. 내부(internal) 채널: 라우터 간 링크
 *   3. 방출(eject) 채널: 네트워크 출구 라우터 → 목적지 -1(외부)
 * 이 덤프는 토폴로지 검증, 라우팅 경로 분석, 시뮬레이션 후 후처리에 활용된다.
 *
 * 실행 컨텍스트: 시뮬레이션 초기화 후 또는 종료 시 단일 스레드에서 호출. const 함수.
 *
 * 호출 체인:
 *   시뮬레이터 통계 출력 루틴 → Network::DumpChannelMap()
 *     → FlitChannel::GetSourcePort/GetSink/GetSinkPort (각 채널의 연결 정보 쿼리)
 */
void Network::DumpChannelMap( ostream & os, string const & prefix ) const
{
  os << prefix << "source_router,source_port,dest_router,dest_port" << endl;
  // [한국어] CSV 헤더 출력: 소스 라우터 ID, 소스 포트, 목적지 라우터 ID, 목적지 포트

  for(int c = 0; c < _nodes; ++c)
    // [한국어] 주입(inject) 채널 연결 정보 출력: 노드 0부터 (_nodes - 1)까지
    os << prefix
       << "-1,"                              // [한국어] 소스 라우터 = -1 (외부 엔드포인트, 라우터 아님)
       << _inject[c]->GetSourcePort() << ',' // [한국어] 소스 포트 = 노드 번호 c (주입 채널의 외부 포트 번호)
       << _inject[c]->GetSink()->GetID() << ',' // [한국어] 목적지 라우터 ID = 주입 채널이 연결된 첫 번째 라우터
       << _inject[c]->GetSinkPort() << endl; // [한국어] 목적지 포트 = 해당 라우터의 입력 포트 번호

  for(int c = 0; c < _channels; ++c)
    // [한국어] 내부 채널 연결 정보 출력: 라우터 간 링크 0부터 (_channels - 1)까지
    os << prefix
       << _chan[c]->GetSource()->GetID() << ',' // [한국어] 소스 라우터 ID = 이 내부 채널을 보내는 라우터
       << _chan[c]->GetSourcePort() << ','       // [한국어] 소스 포트 = 소스 라우터의 출력 포트 번호
       << _chan[c]->GetSink()->GetID() << ','    // [한국어] 목적지 라우터 ID = 이 내부 채널이 연결되는 라우터
       << _chan[c]->GetSinkPort() << endl;       // [한국어] 목적지 포트 = 목적지 라우터의 입력 포트 번호

  for(int c = 0; c < _nodes; ++c)
    // [한국어] 방출(eject) 채널 연결 정보 출력: 노드 0부터 (_nodes - 1)까지
    os << prefix
       << _eject[c]->GetSource()->GetID() << ',' // [한국어] 소스 라우터 ID = 방출 채널을 보내는 마지막 라우터
       << _eject[c]->GetSourcePort() << ','       // [한국어] 소스 포트 = 해당 라우터의 출력 포트 번호
       << "-1,"                                   // [한국어] 목적지 라우터 = -1 (외부 엔드포인트, 라우터 아님)
       << _eject[c]->GetSinkPort() << endl;       // [한국어] 목적지 포트 = 노드 번호 c (방출 채널의 외부 포트 번호)
}

/*
 * [한국어]
 * Network::DumpNodeMap - 각 엔드포인트 노드의 주입·방출 라우터 연결을 CSV 형식으로 덤프
 *
 * @os:     출력 대상 스트림.
 * @prefix: 각 출력 행 앞에 붙일 접두어 문자열.
 * @return: (void)
 *
 * 각 노드(엔드포인트)에 대해 "플릿을 방출하는(eject) 라우터 ID, 플릿을 주입받는(inject)
 * 라우터 ID" 쌍을 출력한다. 일반적으로 같은 노드의 주입·방출 라우터는 동일하지만,
 * 비대칭 토폴로지에서는 다를 수 있다. DumpChannelMap()과 함께 사용해 노드-라우터 매핑을
 * 검증하거나, 후처리 분석 도구에 입력으로 제공한다.
 *
 * 실행 컨텍스트: 시뮬레이션 종료 후 통계 출력 단계. const 함수.
 *
 * 호출 체인:
 *   시뮬레이터 통계 출력 루틴 → Network::DumpNodeMap()
 *     → FlitChannel::GetSource/GetSink (채널 연결 라우터 쿼리)
 */
void Network::DumpNodeMap( ostream & os, string const & prefix ) const
{
  os << prefix << "source_router,dest_router" << endl;
  // [한국어] CSV 헤더: source_router = 방출 측 라우터 ID, dest_router = 주입 측 라우터 ID

  for(int s = 0; s < _nodes; ++s)
    // [한국어] 노드 0부터 (_nodes - 1)까지 순회하며 각 노드의 라우터 연결 정보 출력
    os << prefix
       << _eject[s]->GetSource()->GetID() << ','  // [한국어] 노드 s로 플릿을 방출하는 라우터 ID (목적지 측 마지막 라우터)
       << _inject[s]->GetSink()->GetID() << endl;  // [한국어] 노드 s에서 플릿을 받는(주입) 라우터 ID (소스 측 첫 번째 라우터)
}
