// $Id: anynet.hpp 5354 2012-11-07 23:51:49Z qtedq $

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
 * [한국어 설명] AnyNet 임의 토폴로지 네트워크 헤더 (anynet.hpp)
 *
 * === 파일의 역할 ===
 * AnyNet은 사용자가 텍스트 파일로 네트워크 토폴로지를 자유롭게 정의할 수 있는
 * 범용 NoC(Network-on-Chip) 시뮬레이션 클래스이다. Mesh, Torus, Fat-Tree 등
 * 고정 토폴로지 클래스와 달리, 라우터 간 연결 관계와 채널 레이턴시를 외부 파일
 * ("network_file" 설정 키)에서 읽어 동적으로 구성한다.
 * 파일 파싱 후 Dijkstra 알고리즘으로 최단 경로 라우팅 테이블을 자동 계산하여
 * min_anynet 라우팅 함수에 제공한다.
 * 이 헤더는 AnyNet 클래스 선언 및 파일 외부에서 사용되는
 * min_anynet 라우팅 함수 프로토타입을 정의한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 타이밍 모델에서 SM 간 메모리 요청은 intersim2 NoC 시뮬레이터를
 * 통해 라우팅된다. AnyNet은 intersim2 내부의 Network 추상 클래스를 상속받아
 * 임의 토폴로지를 구현하는 구체 클래스이다.
 * 호출 체인:
 *   icnt_wrapper.cc → Network::New() → AnyNet::AnyNet()
 *   → _ComputeSize() → readFile() (토폴로지 파싱)
 *   → _Alloc() (채널/라우터 배열 할당)
 *   → _BuildNet() → buildRoutingTable() → route() (Dijkstra)
 * 실행 컨텍스트: 시뮬레이터 초기화 단계에서 단일 스레드로 실행된다.
 * 이후 매 사이클 step()에서 min_anynet 라우팅 함수가 호출된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - network.hpp: Network 기반 클래스 (라우터 배열, 채널 배열, _Alloc, step 등)
 *   - routefunc.hpp: gRoutingFunctionMap (전역 라우팅 함수 테이블), OutputSet
 *   - router.hpp: Router::NewRouter() — 라우터 객체 생성
 * 의존받는 모듈:
 *   - icnt_wrapper.cc: Network::New()를 통해 AnyNet 인스턴스를 생성함
 *   - anynet.cpp에 정의된 min_anynet: Router의 라우팅 함수 콜백으로 등록됨
 * 공유 자료구조:
 *   - global_routing_table (anynet.cpp에 정의된 전역 포인터):
 *     routing_table[0]을 가리키며, min_anynet이 라우터 ID와 목적지 노드로
 *     출력 포트를 조회하는 데 사용한다.
 *
 * === 주요 함수/구조체 요약 ===
 * AnyNet::AnyNet()          : 생성자 — 파일 파싱→크기 계산→할당→네트워크 빌드 순으로 초기화
 * AnyNet::readFile()        : 토폴로지 텍스트 파일을 상태 머신으로 파싱, node_list/router_list 구성
 * AnyNet::buildRoutingTable(): 모든 라우터에서 route() 호출로 Dijkstra 수행, global_routing_table 설정
 * AnyNet::route(r_start)    : r_start에서 전체 라우터 대상 Dijkstra 최단 경로 계산
 * min_anynet()              : 라우팅 함수 콜백 — global_routing_table 조회로 출력 포트 결정
 */

#ifndef _ANYNET_HPP_
#define _ANYNET_HPP_

#include "network.hpp"    // [한국어] Network 기반 클래스 — 라우터/채널 배열, _Alloc(), step() 제공
#include "routefunc.hpp"  // [한국어] gRoutingFunctionMap, OutputSet, Flit 등 라우팅 인프라
#include <cassert>        // [한국어] assert() — 파싱 오류/잘못된 토폴로지 즉시 중단용
#include <string>         // [한국어] file_name 저장에 사용하는 std::string
#include <map>            // [한국어] node_list, router_list, routing_table의 희소 매핑 컨테이너
#include <list>           // [한국어] (현재 직접 사용되지 않으나 파싱 보조 목적으로 포함)

/*
 * [한국어] AnyNet - 임의 토폴로지 NoC 네트워크 클래스
 *
 * Network 기반 클래스를 상속받아, 텍스트 파일로 정의된 임의의 라우터-노드
 * 연결 구조를 시뮬레이션한다. 고정 토폴로지 클래스(Mesh, QTree 등)와 달리
 * k, n 같은 파라미터가 없으며 GetN()/GetK()는 -1을 반환한다.
 *
 * 내부 자료구조 3종:
 *   1) node_list[node_id] = router_id
 *      : 터미널 노드가 어느 라우터에 직접 연결되어 있는지 저장
 *   2) router_list[type][src_id][dst_id] = (port, latency)
 *      : type=0 이면 라우터→노드 연결, type=1 이면 라우터→라우터 연결.
 *        port 필드는 _BuildNet() 단계에서 채워진다 (초기값 -1).
 *   3) routing_table[router_id][dest_node] = out_port
 *      : 각 라우터에서 각 목적지 노드로 향하는 출력 포트 (Dijkstra 결과)
 */
class AnyNet : public Network {

  string file_name;
  /* [한국어] 토폴로지를 정의하는 텍스트 파일 경로.
   * 설정자: _ComputeSize() 내에서 config.GetStr("network_file")로 설정.
   * 읽는 자: readFile()이 ifstream으로 열어 파싱.
   * 값 범위: 비어 있으면 즉시 exit(-1) — 필수 설정값.
   * 동기화: 초기화 단계에서만 쓰이며 이후 불변(read-only). */

  //associtation between  nodes and routers
  map<int, int > node_list;
  /* [한국어] 터미널 노드 ID → 연결된 라우터 ID 매핑 테이블.
   * 설정자: readFile() 내 BODY_ID 상태에서 "router X node Y" 또는
   *         "node X router Y" 구문을 처리할 때 node_list[node_id] = router_id로 채워짐.
   * 읽는 자: _ComputeSize()에서 _nodes 계산 (node_list.size()),
   *          _BuildNet()에서 inject/eject 채널 수 결정에 간접 사용.
   * 값 범위: 키(노드 ID)는 0부터 _nodes-1까지 순차 번호이어야 함 (readFile()에서 검증).
   *          값(라우터 ID)은 0부터 _size-1 사이.
   * 동기화: 초기화 단계(단일 스레드)에서만 수정. 이후 buildRoutingTable()/route()에서 읽기 전용. */

  //[link type][src router][dest router]=(port, latency)
  vector<map<int,  map<int, pair<int,int> > > > router_list;
  /* [한국어] 라우터 간 / 라우터-노드 간 연결 정보를 저장하는 2차원 희소 맵.
   * 인덱스 구조:
   *   router_list[0][router_id][node_id]   = (out_port, latency): 라우터→노드 방향
   *   router_list[1][router_id][router_id2] = (out_port, latency): 라우터→라우터 방향
   *   pair.first  = 해당 라우터의 출력 포트 번호 (초기값 -1, _BuildNet()에서 확정)
   *   pair.second = 채널 레이턴시 (사이클 단위, 기본값 1)
   * 설정자: readFile()이 파일 파싱 중 각 항목을 삽입/갱신.
   *          _BuildNet()이 out_port 필드(pair.first)를 실제 포트 번호로 채움.
   * 읽는 자: _ComputeSize()가 _size/_channels 계산,
   *          _BuildNet()이 채널 레이턴시 설정 및 라우터 포트 연결,
   *          route()가 Dijkstra 탐색 시 인접 라우터와 가중치(pair.second) 조회.
   * 값 범위: resize(2)로 크기 2 고정.
   * 동기화: 초기화 단계 단일 스레드에서만 수정. */

  //stores minimal routing information from every router to every node
  //[router][dest_node]=port
  vector<map<int, int> > routing_table;
  /* [한국어] 최소 경로 라우팅 테이블 — 각 라우터에서 각 목적지 노드까지의 출력 포트.
   * 인덱스: routing_table[router_id][dest_node_id] = out_port
   * 설정자: buildRoutingTable()이 resize(_size) 후 route() 호출로 채움.
   *          buildRoutingTable() 완료 후 global_routing_table = &routing_table[0]으로
   *          전역 포인터에 등록됨.
   * 읽는 자: min_anynet() 라우팅 콜백이 global_routing_table 포인터를 통해 조회.
   * 값 범위: 외부 포트 번호 (0 이상 radix 미만). 도달 불가 목적지는 항목 없음.
   * 동기화: 초기화 단계에서만 수정. 시뮬레이션 루프에서는 읽기 전용. */

  /*
   * [한국어] _ComputeSize - 네트워크 크기 파라미터(_size, _nodes, _channels) 계산
   *
   * @config: BooksimConfig — "network_file" 설정 키로 파일 경로를 읽음
   * @return: 없음 (멤버 변수 _size, _nodes, _channels를 직접 설정)
   *
   * readFile()을 호출하여 토폴로지 파일을 파싱한 후,
   * node_list와 router_list로부터 _nodes, _size, _channels를 계산한다.
   * Network::_Alloc()이 이 값들을 기반으로 채널/라우터 배열을 할당하므로
   * 반드시 _Alloc() 전에 호출되어야 한다.
   *
   * 호출 체인:
   *   AnyNet() 생성자 → [_ComputeSize()] → readFile() → _Alloc() → _BuildNet()
   */
  void _ComputeSize( const Configuration &config );

  /*
   * [한국어] _BuildNet - 라우터 객체 생성 및 채널 연결
   *
   * @config: BooksimConfig — Router::NewRouter()에 전달하여 라우터별 설정 적용
   * @return: 없음
   *
   * _Alloc()이 할당한 _routers[], _inject[], _eject[], _chan[] 배열을
   * router_list 정보에 따라 실제로 연결한다.
   * inject/eject 채널(노드→라우터)을 먼저 구성하고, 이후 라우터 간 채널을
   * 순서대로 할당한다. 모든 채널 연결 후 buildRoutingTable()을 호출한다.
   *
   * 호출 체인:
   *   AnyNet() 생성자 → _ComputeSize() → _Alloc() → [_BuildNet()] → buildRoutingTable()
   */
  void _BuildNet( const Configuration &config );

  /*
   * [한국어] readFile - 토폴로지 텍스트 파일을 상태 머신으로 파싱
   *
   * @return: 없음 (node_list, router_list 멤버를 채움)
   *
   * file_name이 가리키는 파일을 한 줄씩 읽어 HEAD_TYPE→HEAD_ID→BODY_TYPE→BODY_ID→LINK_WEIGHT
   * 5단계 상태 머신으로 파싱한다.
   * 파일 형식:
   *   "router 0 router 1 15 router 2" — 라우터0 → 라우터1 (15사이클), 라우터0 → 라우터2 (1사이클)
   *   "router 0 node 0 node 1 5"      — 라우터0 → 노드0 (1사이클), 라우터0 → 노드1 (5사이클)
   * 노드 ID는 0부터 시작하는 순차 번호여야 하며, 그렇지 않으면 assert 실패.
   * 노드가 두 개 이상의 라우터에 연결되면 오류로 처리한다.
   *
   * 호출 체인:
   *   _ComputeSize() → [readFile()]
   */
  void readFile();

  /*
   * [한국어] buildRoutingTable - 전체 라우터의 Dijkstra 라우팅 테이블 구축
   *
   * @return: 없음 (routing_table 멤버 및 전역 global_routing_table 설정)
   *
   * routing_table을 _size 크기로 초기화한 후, 각 라우터 인덱스 i에 대해
   * route(i)를 호출하여 해당 라우터의 Dijkstra 최단 경로를 계산한다.
   * 모든 라우터의 테이블 구축 완료 후 global_routing_table 전역 포인터를
   * routing_table[0] 주소로 설정하여 min_anynet 라우팅 함수가 접근 가능하도록 한다.
   *
   * 호출 체인:
   *   _BuildNet() → [buildRoutingTable()] → route(0), route(1), ..., route(_size-1)
   */
  void buildRoutingTable();

  /*
   * [한국어] route - r_start 라우터에서 모든 노드까지의 Dijkstra 최단 경로 계산
   *
   * @r_start: 출발 라우터 ID (0 이상 _size 미만)
   * @return: 없음 (routing_table[r_start][dest_node] = out_port 를 채움)
   *
   * 표준 Dijkstra 알고리즘을 사용하여 r_start에서 모든 라우터까지의
   * 최단 거리(dist[])와 이전 라우터(prev[])를 계산한다.
   * 거리 단위는 홉 수가 아니라 채널 레이턴시 합계(사이클)이다.
   * prev[] 역추적으로 r_start의 첫 번째 홉 이웃(neighbor)을 찾고,
   * 그 이웃에 대한 출력 포트를 routing_table[r_start][각 dest_node]에 기록한다.
   * 동일 라우터에 연결된 여러 노드는 같은 출력 포트에 매핑된다.
   *
   * 호출 체인:
   *   buildRoutingTable() → [route(r_start)]
   */
  void route(int r_start);

public:

  /*
   * [한국어] AnyNet 생성자 — 임의 토폴로지 네트워크 초기화
   *
   * @config: BooksimConfig — "network_file", "k", VC 수 등 설정 포함
   * @name: 네트워크 객체 이름 문자열 (로깅/디버그용)
   * @return: 없음 (생성자)
   *
   * Network 기반 클래스 초기화 후 _ComputeSize → _Alloc → _BuildNet 순으로
   * 네트워크를 완전히 구성한다. 이 생성자가 반환된 시점부터 시뮬레이션이 가능하다.
   * router_list는 크기 2로 미리 resize하여 [0]=노드 연결, [1]=라우터 연결 슬롯을 준비한다.
   *
   * 호출 체인:
   *   Network::New() → [AnyNet()] → _ComputeSize() → _Alloc() → _BuildNet()
   */
  AnyNet( const Configuration &config, const string & name );

  /*
   * [한국어] ~AnyNet 소멸자 — router_list 내부 맵 정리
   *
   * @return: 없음 (소멸자)
   *
   * router_list[0], router_list[1]의 각 항목(map<int, pair<int,int>>)을 clear()하여
   * 내부 맵 메모리를 해제한다. vector 자체는 자동 소멸된다.
   * global_routing_table 포인터가 routing_table[0]을 가리키므로,
   * 이 소멸자 호출 후 global_routing_table 접근은 undefined behavior이다.
   *
   * 호출 체인:
   *   시뮬레이션 종료 → Network 소멸자 → [~AnyNet()]
   */
  ~AnyNet();

  /*
   * [한국어] GetN - 네트워크 차원 수 반환 (AnyNet은 정의 없음)
   *
   * @return: -1 (임의 토폴로지는 고정 차원 개념이 없음을 나타냄)
   *
   * Mesh/Torus 등 정규 토폴로지의 n(차원 수) 개념이 AnyNet에는 적용되지 않는다.
   * 라우팅 함수나 통계 모듈이 GetN()을 호출하면 -1을 받으며, 호출자가 이를 인식해야 한다.
   *
   * 호출 체인:
   *   통계/라우팅 쿼리 → [GetN()]
   */
  int GetN( ) const{ return -1;}

  /*
   * [한국어] GetK - 네트워크 방사(radix) 반환 (AnyNet은 정의 없음)
   *
   * @return: -1 (임의 토폴로지는 고정 방사 개념이 없음을 나타냄)
   *
   * k-ary n-fly 등 정규 토폴로지의 k 값이 AnyNet에는 없다.
   * GetN()과 동일한 이유로 -1을 반환한다.
   *
   * 호출 체인:
   *   통계/라우팅 쿼리 → [GetK()]
   */
  int GetK( ) const{ return -1;}

  /*
   * [한국어] RegisterRoutingFunctions - "min_anynet" 라우팅 함수를 전역 맵에 등록
   *
   * @return: 없음
   *
   * gRoutingFunctionMap["min_anynet"] = &min_anynet 으로 등록한다.
   * Network::New()가 네트워크 유형별로 이 정적 함수를 호출하여
   * 설정 파일의 routing_function 값과 실제 함수 포인터를 연결한다.
   *
   * 호출 체인:
   *   Network::RegisterRoutingFunctions() → [AnyNet::RegisterRoutingFunctions()]
   */
  static void RegisterRoutingFunctions();

  /*
   * [한국어] Capacity - 네트워크 수용량 반환 (AnyNet은 정의 없음)
   *
   * @return: -1 (임의 토폴로지는 단일 수용량 메트릭이 없음)
   *
   * Capacity()는 bisection bandwidth 등의 개념적 메트릭으로, AnyNet은
   * 임의 구조이므로 일반적인 계산이 불가능하다.
   *
   * 호출 체인:
   *   통계 모듈 → [Capacity()]
   */
  double Capacity( ) const {return -1;}

  /*
   * [한국어] InsertRandomFaults - 임의 결함 삽입 (AnyNet은 미구현)
   *
   * @config: BooksimConfig (사용되지 않음)
   * @return: 없음
   *
   * 결함 주입 실험을 위한 인터페이스이나 AnyNet에서는 구현되지 않아 아무 동작도 하지 않는다.
   *
   * 호출 체인:
   *   시뮬레이션 설정 → [InsertRandomFaults()] (no-op)
   */
  void InsertRandomFaults( const Configuration &config ){}
};

/*
 * [한국어] min_anynet - AnyNet 전용 최소 경로 라우팅 함수 (콜백)
 *
 * @r:          현재 플릿이 위치한 라우터 포인터 — GetID()로 라우터 ID를 얻음
 * @f:          현재 라우팅할 플릿 포인터 — dest(목적지 노드 ID)와 type(패킷 종류) 조회
 * @in_channel: 플릿이 들어온 입력 채널 번호 (이 라우팅 함수에서는 사용하지 않음)
 * @outputs:    라우팅 결과를 기록할 OutputSet — AddRange()로 출력 포트+VC 범위 추가
 * @inject:     true이면 플릿이 inject 중이며 out_port 계산 불필요
 * @return:     없음 (outputs 객체를 통해 결과 전달)
 *
 * global_routing_table[r->GetID()][f->dest]를 조회하여 출력 포트를 결정한다.
 * 플릿의 타입(READ_REQUEST / WRITE_REQUEST / READ_REPLY / WRITE_REPLY)에 따라
 * gReadReqBeginVC~gWriteReplyEndVC 범위 내에서 VC(가상 채널)를 선택한다.
 * inject=true이면 out_port=-1로 두고(inject 경로는 별도 처리), VC 범위만 결정한다.
 *
 * 주의: global_routing_table은 anynet.cpp에 전역 포인터로 선언되어 있으며,
 * AnyNet 인스턴스가 살아있는 동안만 유효하다.
 *
 * 호출 체인:
 *   Router::Route() → [min_anynet()] → outputs->AddRange()
 */
void min_anynet( const Router *r, const Flit *f, int in_channel,
		      OutputSet *outputs, bool inject );
#endif
