// $Id: anynet.cpp 5354 2012-11-07 23:51:49Z qtedq $

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

/*anynet
 *
 *Network setup file format
 *example 1:
 *router 0 router 1 15 router 2
 *
 *Router 0 is connect to router 1 with a 15-cycle channel, and router 0 is connected to
 * router 2 with a 1-cycle channel, the channels latency are unidirectional, so channel
 * from router 1 back to router 0 is only single-cycle because it was not specified
 *
 *example 2:
 *router 0 node 0 node 1 5 node 2 5
 *
 *Router 0 is directly connected to node 0-2. Channel latency is 5cycles for 1 and 2. In
 * this case the latency specification is bidirectional, the injeciton and ejection lat
 * for node 1 and 2 are 5-cycle
 *
 *other notes:
 *
 *Router and node numbers must be sequential starting with 0
 *Credit channel latency follows the channel latency, even though it travels in revse
 * direction this might not be desired
 *
 */

/*
 * [한국어 설명] AnyNet 임의 토폴로지 네트워크 구현 (anynet.cpp)
 *
 * === 파일의 역할 ===
 * AnyNet 클래스의 전체 구현을 담는 파일이다. 텍스트 파일에서 임의의 네트워크 토폴로지를
 * 읽어 라우터 객체를 생성하고 채널로 연결한 뒤, Dijkstra 알고리즘으로 각 라우터의
 * 최단 경로 라우팅 테이블을 구축한다. 또한 min_anynet 라우팅 함수를 구현하여
 * 매 사이클 플릿 라우팅 결정에 사용될 콜백을 제공한다.
 * 파일 파싱은 5단계 상태 머신(HEAD_TYPE→HEAD_ID→BODY_TYPE→BODY_ID→LINK_WEIGHT)으로
 * 진행되며, 파싱 결과는 node_list와 router_list 자료구조에 저장된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 NoC 시뮬레이션 계층(intersim2)에서 구체적인 토폴로지를 제공하는 모듈이다.
 * icnt_wrapper.cc가 Network::New()를 통해 AnyNet을 생성하면, 초기화 시퀀스가 실행된다:
 *   1) readFile(): 토폴로지 파일 파싱 → node_list, router_list 구성
 *   2) _ComputeSize(): _size(라우터 수), _nodes(터미널 노드 수), _channels(채널 수) 계산
 *   3) _Alloc(): Network 기반 클래스가 _routers[], _inject[], _chan[] 배열 할당
 *   4) _BuildNet(): 라우터 객체 생성, 채널 레이턴시 설정, inject/eject 및 라우터간 채널 연결
 *   5) buildRoutingTable(): 모든 라우터에서 Dijkstra 실행, global_routing_table 설정
 * 이후 시뮬레이션 루프에서 step() 호출 시 min_anynet()이 플릿마다 호출된다.
 * 실행 컨텍스트: 초기화는 단일 메인 스레드; 시뮬레이션 루프는 gpgpusim의 시뮬 스레드.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - anynet.hpp: 클래스 선언 및 멤버 변수 정의
 *   - network.hpp: Network 기반 클래스 (라우터/채널 배열, _Alloc, step, _timed_modules)
 *   - routefunc.hpp: gRoutingFunctionMap, gNumVCs, gReadReqBeginVC 등 전역 VC 범위 변수
 *   - router.hpp: Router::NewRouter() — 각 라우터 인스턴스 생성
 * 공유 전역 변수:
 *   - global_routing_table (이 파일 선언): AnyNet 인스턴스의 routing_table[0] 주소를 보관.
 *     min_anynet 라우팅 콜백이 Router 객체에서 AnyNet 객체로 직접 접근하기 어려운 구조적
 *     한계를 우회하기 위해 전역 포인터를 사용한다 (코드 주석에서 "a hack"으로 명시).
 * 데이터 흐름:
 *   텍스트 파일 → readFile() → node_list/router_list → _BuildNet() → 라우터 연결
 *   router_list → buildRoutingTable() → route() → routing_table → global_routing_table
 *   플릿 라우팅 요청 → min_anynet() → global_routing_table 조회 → OutputSet에 포트 기록
 *
 * === 주요 함수/구조체 요약 ===
 * AnyNet()          : 생성자 — router_list 크기 2로 초기화 후 표준 3단계(ComputeSize/Alloc/BuildNet) 실행
 * ~AnyNet()         : 소멸자 — router_list 내부 맵 clear()
 * _ComputeSize()    : readFile() 호출 후 _size, _nodes, _channels 계산 및 디버그 출력
 * _BuildNet()       : 라우터 생성, 채널 레이턴시 설정, inject/router간 채널 연결, buildRoutingTable 호출
 * min_anynet()      : 라우팅 함수 콜백 — global_routing_table 조회로 out_port 결정, VC 범위 설정
 * buildRoutingTable(): 모든 라우터 route() 호출 후 global_routing_table 포인터 등록
 * route(r_start)    : Dijkstra로 r_start→모든 노드 최단 경로 계산, routing_table[r_start] 채움
 * readFile()        : 5단계 상태 머신으로 토폴로지 파일 파싱, node_list/router_list 구성 및 검증
 */

#include "anynet.hpp"   // [한국어] AnyNet 클래스 선언 및 멤버 구조체/타입 포함
#include <fstream>      // [한국어] ifstream — 토폴로지 텍스트 파일 열기/읽기
#include <sstream>      // [한국어] ostringstream — 라우터 이름 문자열 조합
#include <limits>       // [한국어] numeric_limits<int>::max() — Dijkstra 초기 거리 무한대 설정
#include <algorithm>    // [한국어] sort() — 노드 ID 순차성 검증에 사용

//this is a hack, I can't easily get the routing talbe out of the network
map<int, int>* global_routing_table;
/* [한국어] AnyNet 인스턴스의 routing_table[0] 주소를 보관하는 전역 포인터.
 * 설정자: AnyNet::buildRoutingTable()이 routing_table 구축 완료 후
 *          global_routing_table = &routing_table[0] 으로 설정.
 * 읽는 자: min_anynet() 라우팅 콜백이 global_routing_table[r->GetID()][f->dest] 형태로 조회.
 * 값 범위: 유효한 AnyNet::routing_table 벡터 원소 주소. AnyNet 소멸 후 dangling pointer.
 * 동기화: 초기화 단계에서 1회 쓰기, 이후 시뮬레이션 루프에서 읽기 전용.
 *          동시 접근 없으므로 별도 락 불필요.
 * 설계 의도: Router 콜백 함수(min_anynet)는 Router*와 Flit*만 인수로 받아
 *             AnyNet 인스턴스에 직접 접근할 수 없다. 이 전역 포인터가 그 간극을 메운다.
 *             (원코드 주석: "this is a hack") */

/*
 * [한국어]
 * AnyNet::AnyNet - 임의 토폴로지 네트워크 생성자
 *
 * @config: BooksimConfig — "network_file" 및 라우터 설정(VC 수, 버퍼 크기 등) 포함
 * @name: 네트워크 객체 이름 문자열 (디버그/로그 식별자)
 * @return: 없음 (생성자)
 *
 * Network 기반 클래스 생성자를 먼저 호출하여 기본 인프라를 초기화한다.
 * router_list를 크기 2로 미리 resize하여 [0]=노드 연결, [1]=라우터 연결 슬롯을 준비한다.
 * 이후 _ComputeSize → _Alloc → _BuildNet 순으로 네트워크를 완전히 구성한다.
 * 생성자가 반환되면 global_routing_table이 설정되어 시뮬레이션 준비가 완료된 상태이다.
 *
 * 호출 체인:
 *   Network::New() → [AnyNet()] → router_list.resize(2)
 *                                 → _ComputeSize() → readFile()
 *                                 → _Alloc() (Network 기반)
 *                                 → _BuildNet() → buildRoutingTable()
 */
AnyNet::AnyNet( const Configuration &config, const string & name )
  :  Network( config, name ){  // [한국어] Network 기반 클래스 생성자 호출 — _routers, _chan, _inject, _eject 포인터 초기화

  router_list.resize(2);          // [한국어] [0]=라우터→노드 연결, [1]=라우터→라우터 연결 두 슬롯으로 초기화
  _ComputeSize( config );         // [한국어] readFile()로 토폴로지 파싱 후 _size/_nodes/_channels 계산
  _Alloc( );                      // [한국어] Network::_Alloc() — _routers[], _chan[], _inject[], _eject[] 배열 동적 할당
  _BuildNet( config );            // [한국어] 라우터 객체 생성, 채널 연결, 라우팅 테이블 구축
}

/*
 * [한국어]
 * AnyNet::~AnyNet - AnyNet 소멸자
 *
 * @return: 없음 (소멸자)
 *
 * router_list[0], router_list[1]의 각 라우터 엔트리에 대한 내부 맵을 clear()하여
 * pair<int,int> 값들이 담긴 내부 map 메모리를 명시적으로 해제한다.
 * vector 자체와 map 키 레벨은 자동 소멸되지만, 중첩 map의 메모리를 확실히 정리하기 위해
 * 명시적 루프를 사용한다.
 * 주의: 이 소멸자 호출 후 global_routing_table이 routing_table[0]을 가리키고 있으면
 * dangling pointer가 되므로 시뮬레이터 종료 순서에 주의해야 한다.
 *
 * 호출 체인:
 *   시뮬레이션 종료 → Network 소멸자 호출 순서 → [~AnyNet()]
 */
AnyNet::~AnyNet(){
  for(int i = 0; i < 2; ++i) {  // [한국어] [0]=노드 연결 맵, [1]=라우터 연결 맵 순으로 순회
    for(map<int, map<int, pair<int,int> > >::iterator iter = router_list[i].begin();
	iter != router_list[i].end();
	++iter) {                  // [한국어] router_list[i]의 각 라우터(키=라우터 ID) 항목 순회
      iter->second.clear();       // [한국어] 각 라우터에 연결된 목적지 맵(map<int, pair<int,int>>)을 비워 메모리 해제
    }
  }
}

/*
 * [한국어]
 * AnyNet::_ComputeSize - 네트워크 크기 파라미터 계산 및 디버그 출력
 *
 * @config: BooksimConfig — "network_file" 키로 파일 경로 읽음
 * @return: 없음 (_size, _nodes, _channels 멤버 변수를 직접 설정)
 *
 * readFile()을 호출하여 토폴로지 파일을 파싱한 후, 파싱 결과로부터:
 *   - _nodes = node_list.size()    (터미널 노드 수)
 *   - _size  = router_list[1].size() (라우터 수 = 라우터 간 연결 맵의 라우터 수)
 *   - _channels = 라우터 간 방향성 링크 수 (router_list[1] 순회로 집계)
 * 를 설정한다. 또한 파싱된 노드/라우터 연결 정보를 콘솔에 출력하여 검증을 돕는다.
 * Network::_Alloc()이 이 값들을 기반으로 배열을 동적 할당하므로 _Alloc() 전에 반드시 호출.
 *
 * 에러 처리: network_file이 빈 문자열이면 오류 메시지 출력 후 exit(-1).
 *
 * 호출 체인:
 *   AnyNet() → [_ComputeSize()] → readFile()
 *                                  → (콘솔 출력) → _nodes/_size/_channels 설정
 */
void AnyNet::_ComputeSize( const Configuration &config ){
  file_name = config.GetStr("network_file");  // [한국어] 설정 파일의 "network_file" 값으로 토폴로지 파일 경로 설정
  if(file_name==""){                          // [한국어] 파일 경로가 지정되지 않은 경우 — 필수 설정 누락
    cout<<"No network file name provided"<<endl;  // [한국어] 오류 메시지 출력 후 즉시 종료
    exit(-1);                                 // [한국어] 복구 불가능한 설정 오류 — 시뮬레이터 강제 종료
  }
  //parse the network description file
  readFile();  // [한국어] 토폴로지 텍스트 파일 파싱 — node_list, router_list 구성

  _channels =0;  // [한국어] 라우터 간 채널 수 초기화 (이하 루프에서 집계)
  cout<<"========================Network File Parsed=================\n";  // [한국어] 파싱 완료 구분선 출력
  cout<<"******************node listing**********************\n";          // [한국어] 노드 목록 출력 시작
  map<int,  int >::iterator iter;
  for(iter = node_list.begin(); iter!=node_list.end(); iter++){  // [한국어] 모든 터미널 노드 순회
    cout<<"Node "<<iter->first;                                   // [한국어] 노드 ID 출력
    cout<<"\tRouter "<<iter->second<<endl;                        // [한국어] 해당 노드와 연결된 라우터 ID 출력
  }

  map<int,   map<int, pair<int,int> > >::iterator iter3;
  cout<<"\n****************router to node listing*************\n";  // [한국어] 라우터→노드 연결 목록 출력 시작
  for(iter3 = router_list[0].begin(); iter3!=router_list[0].end(); iter3++){  // [한국어] 라우터→노드 연결 맵 순회
    cout<<"Router "<<iter3->first<<endl;  // [한국어] 라우터 ID 출력
    map<int, pair<int,int> >::iterator iter2;
    for(iter2 = iter3->second.begin();
	iter2!=iter3->second.end();
	iter2++){                           // [한국어] 해당 라우터에 연결된 모든 노드 출력
      cout<<"\t Node "<<iter2->first<<" lat "<<iter2->second.second<<endl;  // [한국어] 노드 ID와 채널 레이턴시 출력
    }
  }

  cout<<"\n*****************router to router listing************\n";  // [한국어] 라우터→라우터 연결 목록 출력 시작
  for(iter3 = router_list[1].begin(); iter3!=router_list[1].end(); iter3++){  // [한국어] 라우터→라우터 연결 맵 순회
    cout<<"Router "<<iter3->first<<endl;  // [한국어] 라우터 ID 출력
    map<int, pair<int,int> >::iterator iter2;
    if(iter3->second.size() == 0){        // [한국어] 다른 라우터와 연결이 없는 고립 라우터 경고
      cout<<"Caution Router "<<iter3->first
	  <<" is not connected to any other Router\n"<<endl;  // [한국어] 경고 메시지 — 라우팅 불가 상태
    }
    for(iter2 = iter3->second.begin();
	iter2!=iter3->second.end();
	iter2++){                            // [한국어] 해당 라우터에서 나가는 모든 라우터 연결 순회
      cout<<"\t Router "<<iter2->first<<" lat "<<iter2->second.second<<endl;  // [한국어] 연결 라우터 ID와 레이턴시 출력
      _channels++;                        // [한국어] 방향성 라우터 간 채널 수 집계 (단방향 링크 1개 = 1 채널)
    }
  }

  _size = router_list[1].size();   // [한국어] 전체 라우터 수 = 라우터→라우터 맵의 항목 수 (모든 라우터가 등록됨)
  _nodes = node_list.size();       // [한국어] 전체 터미널 노드 수 = node_list 항목 수

}



/*
 * [한국어]
 * AnyNet::_BuildNet - 라우터 객체 생성, 채널 레이턴시 설정, 포트 연결
 *
 * @config: BooksimConfig — Router::NewRouter()에 전달하여 라우터 내부 설정 적용
 * @return: 없음
 *
 * 두 단계로 채널을 구성한다:
 *
 * 1단계 — Injection/Ejection 채널 (노드 ↔ 라우터):
 *   router_list[0] 순회. 각 라우터의 노드 연결 수와 라우터 연결 수를 합산하여
 *   해당 라우터의 radix(포트 수)를 결정하고 Router::NewRouter()로 라우터를 생성한다.
 *   각 노드에 대해 _inject/_inject_cred 채널을 라우터 입력으로, _eject/_eject_cred를
 *   출력으로 연결하며 채널 레이턴시를 router_list[0]의 latency 값으로 설정한다.
 *   outport[node] 카운터로 각 라우터의 다음 사용 가능한 출력 포트를 추적한다.
 *
 * 2단계 — 라우터 간 채널:
 *   router_list[1] 순회. 각 라우터→라우터 연결에 _chan/chan_cred 채널을 할당하고
 *   레이턴시를 설정한다. channel_count로 채널 번호를 순차 할당한다.
 *   src 라우터의 출력 채널과 dst 라우터의 입력 채널로 동시에 등록한다 (단방향).
 *
 * 모든 연결 완료 후 buildRoutingTable()을 호출하여 라우팅 테이블을 구축한다.
 *
 * 에러 처리: outport 배열은 malloc으로 할당하며 해제 없이 반환됨 (메모리 누수 존재).
 *
 * 호출 체인:
 *   AnyNet() → _Alloc() → [_BuildNet()] → buildRoutingTable()
 */
void AnyNet::_BuildNet( const Configuration &config ){


  //I need to keep track the output ports for each router during build
  int * outport = (int*)malloc(sizeof(int)*_size);  // [한국어] 각 라우터별 현재 할당된 출력 포트 수를 추적하는 배열 (크기=_size)
  for(int i = 0; i<_size; i++){outport[i] = 0;}    // [한국어] 모든 라우터의 출력 포트 카운터를 0으로 초기화

  cout<<"==========================Node to Router =====================\n";  // [한국어] 노드→라우터 연결 단계 시작 출력
  //adding the injection/ejection chanenls first
  map<int,   map<int, pair<int,int> > >::iterator niter;
  for(niter = router_list[0].begin(); niter!=router_list[0].end(); niter++){  // [한국어] 모든 라우터의 노드 연결 맵 순회
    map<int,   map<int, pair<int,int> > >::iterator riter = router_list[1].find(niter->first);  // [한국어] 동일 라우터의 라우터 간 연결 맵 탐색
    //calculate radix
    int radix = niter->second.size()+riter->second.size();  // [한국어] 라우터 radix = 노드 연결 수 + 라우터 연결 수 (총 입출력 포트 수)
    int node = niter->first;                                 // [한국어] 현재 처리 중인 라우터 ID (변수명 node는 혼동 여지 있으나 라우터 ID)
    cout<<"router "<<node<<" radix "<<radix<<endl;           // [한국어] 라우터 ID와 계산된 radix 디버그 출력
    //decalre the routers
    ostringstream router_name;
    router_name << "router";
    router_name << "_" <<  node ;                            // [한국어] "router_<ID>" 형태의 라우터 이름 문자열 생성
    _routers[node] = Router::NewRouter( config, this, router_name.str( ),
    				node, radix, radix );          // [한국어] Router 인스턴스 생성 — 입력/출력 모두 radix 포트로 설정
    _timed_modules.push_back(_routers[node]);                // [한국어] 타이밍 시뮬레이션 모듈 목록에 등록 — 매 사이클 step() 호출 대상
    //add injeciton ejection channels
    map<int, pair<int,int> >::iterator nniter;
    for(nniter = niter->second.begin();nniter!=niter->second.end(); nniter++){  // [한국어] 이 라우터에 연결된 모든 노드 순회
      int link = nniter->first;                              // [한국어] 연결 노드 ID
      //add the outport port assined to the map
      (niter->second)[link].first = outport[node];          // [한국어] router_list[0][라우터][노드].first(포트 번호)를 현재 출력 포트 카운터로 확정
      outport[node]++;                                       // [한국어] 다음 노드/라우터 연결을 위해 출력 포트 카운터 증가
      cout<<"\t connected to node "<<link<<" at outport "<<nniter->second.first
	  <<" lat "<<nniter->second.second<<endl;            // [한국어] 연결 노드, 할당된 출력 포트, 레이턴시 디버그 출력
      _inject[link]->SetLatency(nniter->second.second);     // [한국어] inject 채널(노드→라우터) 레이턴시 설정 (사이클 단위)
      _inject_cred[link]->SetLatency(nniter->second.second);// [한국어] inject 크레딧 역방향 채널 레이턴시 설정 (흐름 제어)
      _eject[link]->SetLatency(nniter->second.second);      // [한국어] eject 채널(라우터→노드) 레이턴시 설정
      _eject_cred[link]->SetLatency(nniter->second.second); // [한국어] eject 크레딧 역방향 채널 레이턴시 설정

      _routers[node]->AddInputChannel( _inject[link], _inject_cred[link] );   // [한국어] 라우터 입력 포트에 inject 채널 등록 (노드에서 들어오는 방향)
      _routers[node]->AddOutputChannel( _eject[link], _eject_cred[link] );    // [한국어] 라우터 출력 포트에 eject 채널 등록 (노드로 나가는 방향)
    }

  }

  cout<<"==========================Router to Router =====================\n";  // [한국어] 라우터 간 채널 연결 단계 시작 출력
  //add inter router channels
  //since there is no way to systematically number the channels we just start from 0
  //the map, is a mapping of output->input
  int channel_count = 0;  // [한국어] _chan[] 배열의 현재 할당 위치 — 라우터 간 채널에 순차 번호 부여
  for(niter = router_list[0].begin(); niter!=router_list[0].end(); niter++){   // [한국어] 모든 라우터 순회 (router_list[0]은 전체 라우터 목록 포함)
    map<int,   map<int, pair<int,int> > >::iterator riter = router_list[1].find(niter->first);  // [한국어] 현재 라우터의 라우터→라우터 연결 맵 탐색
    int node = niter->first;                              // [한국어] 현재 처리 중인 라우터 ID
    map<int, pair<int,int> >::iterator rriter;
    cout<<"router "<<node<<endl;                          // [한국어] 현재 라우터 ID 디버그 출력
    for(rriter = riter->second.begin();rriter!=riter->second.end(); rriter++){  // [한국어] 이 라우터에서 나가는 모든 라우터 연결 순회
      int other_node = rriter->first;                    // [한국어] 연결 대상 라우터 ID (변수명 "node"는 혼동 여지 있으나 라우터 ID)
      int link = channel_count;                          // [한국어] 이 단방향 연결에 할당할 _chan[] 채널 번호
      //add the outport port assined to the map
      (riter->second)[other_node].first = outport[node]; // [한국어] router_list[1][src][dst].first(포트 번호)를 현재 출력 포트 카운터로 확정
      outport[node]++;                                   // [한국어] 다음 연결을 위해 출력 포트 카운터 증가
      cout<<"\t connected to router "<<other_node<<" using link "<<link
	  <<" at outport "<<rriter->second.first
	  <<" lat "<<rriter->second.second<<endl;          // [한국어] 연결 라우터, 채널 번호, 포트, 레이턴시 디버그 출력

      _chan[link]->SetLatency(rriter->second.second);       // [한국어] 라우터 간 데이터 채널 레이턴시 설정
      _chan_cred[link]->SetLatency(rriter->second.second);  // [한국어] 라우터 간 크레딧 채널 레이턴시 설정 (역방향 흐름 제어)

      _routers[node]->AddOutputChannel( _chan[link], _chan_cred[link] );       // [한국어] src 라우터의 출력 포트에 채널 등록
      _routers[other_node]->AddInputChannel( _chan[link], _chan_cred[link]);   // [한국어] dst 라우터의 입력 포트에 동일 채널 등록 (단방향 링크)
      channel_count++;  // [한국어] 다음 라우터 간 연결을 위해 채널 번호 증가
    }
  }

  buildRoutingTable();  // [한국어] 모든 채널 연결 완료 후 Dijkstra 기반 라우팅 테이블 구축
}


/*
 * [한국어]
 * AnyNet::RegisterRoutingFunctions - "min_anynet" 라우팅 함수를 전역 맵에 등록
 *
 * @return: 없음
 *
 * intersim2의 전역 라우팅 함수 맵(gRoutingFunctionMap)에 "min_anynet" 키로
 * min_anynet 함수 포인터를 등록한다. 이후 설정 파일에서 routing_function=min_anynet으로
 * 지정하면 이 함수가 모든 라우터의 라우팅 결정에 사용된다.
 * 정적 함수이므로 인스턴스 없이 Network::RegisterRoutingFunctions()에서 호출 가능하다.
 *
 * 호출 체인:
 *   Network::New() 또는 초기화 단계 → [AnyNet::RegisterRoutingFunctions()]
 */
void AnyNet::RegisterRoutingFunctions() {
  gRoutingFunctionMap["min_anynet"] = &min_anynet;  // [한국어] "min_anynet" 문자열 키로 함수 포인터를 전역 라우팅 함수 맵에 등록
}

/*
 * [한국어]
 * min_anynet - AnyNet 전용 최소 경로 라우팅 콜백 함수
 *
 * @r:          현재 플릿이 위치한 라우터 포인터 — GetID()로 라우팅 테이블 조회 키 획득
 * @f:          라우팅할 플릿 포인터 — dest(목적지 노드 ID), type(READ/WRITE REQ/REPLY) 참조
 * @in_channel: 플릿이 들어온 입력 채널 번호 (이 함수에서 미사용 — 최소 경로는 입력 무관)
 * @outputs:    라우팅 결과를 기록할 OutputSet — AddRange()로 출력 포트와 VC 범위 추가
 * @inject:     true이면 플릿이 inject 단계이며 out_port=-1로 초기화된 채로 VC만 결정
 * @return:     없음 (outputs 객체를 통해 결과 전달)
 *
 * inject=false이면 global_routing_table[r->GetID()][f->dest]를 조회하여 out_port를 결정한다.
 * assert로 routing_table에 해당 목적지 항목이 반드시 존재함을 검증한다.
 * 플릿의 타입에 따라 VC 범위(vcBegin~vcEnd)를 결정하며, 기본값은 [0, gNumVCs-1]이다.
 * GPGPU-Sim에서는 READ_REQUEST/WRITE_REQUEST/READ_REPLY/WRITE_REPLY 4종의 패킷이 있으며
 * 각각 별도의 VC 범위를 사용하여 데드락을 방지한다.
 * 마지막으로 outputs->Clear() 후 AddRange(out_port, vcBegin, vcEnd)로 결과를 기록한다.
 *
 * 호출 체인:
 *   Router::Route() (매 사이클) → [min_anynet()] → global_routing_table 조회
 *                                                  → outputs->AddRange()
 */
void min_anynet( const Router *r, const Flit *f, int in_channel,
		 OutputSet *outputs, bool inject ){
  int out_port=-1;        // [한국어] 출력 포트 초기값 — inject=true이거나 조회 전 상태
  if(!inject){            // [한국어] inject 단계가 아닌 경우에만 라우팅 테이블 조회
    assert(global_routing_table[r->GetID()].count(f->dest)!=0);  // [한국어] routing_table에 dest 항목이 없으면 assert 실패 — 토폴로지 오류
    out_port=global_routing_table[r->GetID()][f->dest];          // [한국어] 현재 라우터에서 목적지 노드까지의 최단 경로 출력 포트 조회
  }


  int vcBegin = 0, vcEnd = gNumVCs-1;  // [한국어] 기본 VC 범위: 전체 VC (0부터 최대 VC-1)
  if ( f->type == Flit::READ_REQUEST ) {       // [한국어] 읽기 요청 플릿 — 전용 VC 범위로 제한
    vcBegin = gReadReqBeginVC;
    vcEnd   = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {  // [한국어] 쓰기 요청 플릿 — 전용 VC 범위
    vcBegin = gWriteReqBeginVC;
    vcEnd   = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {    // [한국어] 읽기 응답 플릿 — 전용 VC 범위
    vcBegin = gReadReplyBeginVC;
    vcEnd   = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {   // [한국어] 쓰기 응답 플릿 — 전용 VC 범위
    vcBegin = gWriteReplyBeginVC;
    vcEnd   = gWriteReplyEndVC;
  }

  outputs->Clear( );  // [한국어] 이전 라우팅 결과를 지워 새 결과를 기록할 준비

  outputs->AddRange( out_port , vcBegin, vcEnd );  // [한국어] 결정된 출력 포트와 VC 범위를 OutputSet에 추가 — Router가 이를 참조하여 스위치 할당
}

/*
 * [한국어]
 * AnyNet::buildRoutingTable - 전체 라우터의 Dijkstra 라우팅 테이블 구축
 *
 * @return: 없음 (routing_table 벡터 및 global_routing_table 전역 포인터 설정)
 *
 * routing_table을 _size 크기로 초기화한 후, 각 라우터 인덱스 i(0부터 _size-1)에 대해
 * route(i)를 호출하여 Dijkstra 최단 경로를 계산하고 routing_table[i]를 채운다.
 * 모든 라우터의 테이블 구축 완료 후 global_routing_table = &routing_table[0]으로
 * 전역 포인터를 설정한다. routing_table은 연속 메모리의 vector이므로 [0] 주소로
 * 전체 배열에 포인터 산술로 접근 가능하다.
 *
 * 호출 체인:
 *   _BuildNet() → [buildRoutingTable()] → route(0) ... route(_size-1)
 *                                        → global_routing_table = &routing_table[0]
 */
void AnyNet::buildRoutingTable(){
  cout<<"========================== Routing table  =====================\n";  // [한국어] 라우팅 테이블 구축 시작 구분선 출력
  routing_table.resize(_size);  // [한국어] 모든 라우터에 대한 routing_table 슬롯 초기화 (크기=라우터 수)
  for(int i = 0; i<_size; i++){  // [한국어] 각 라우터 i에서 출발하는 Dijkstra 계산
    route(i);                    // [한국어] 라우터 i를 출발점으로 하는 Dijkstra 최단 경로 계산 → routing_table[i] 채움
  }
  global_routing_table = &routing_table[0];  // [한국어] 전역 포인터를 routing_table 배열 시작점으로 설정 — min_anynet 콜백에서 접근 가능
}


//11/7/2012
//basically djistra's, tested on a large dragonfly anynet configuration
/*
 * [한국어]
 * AnyNet::route - r_start 라우터에서 전체 네트워크 대상 Dijkstra 최단 경로 계산
 *
 * @r_start: 출발 라우터 ID (0 이상 _size 미만)
 * @return: 없음 (routing_table[r_start][dest_node] = out_port 형태로 결과 저장)
 *
 * 표준 Dijkstra 알고리즘 구현:
 *   1) dist[] 초기화: r_start=0, 나머지=INT_MAX; rlist(미방문 집합)에 전체 라우터 삽입
 *   2) 반복: rlist에서 dist 최솟값 라우터 min_cand 선택 후 제거
 *   3) min_cand의 인접 라우터(router_list[1][min_cand])에 대해 거리 완화(relaxation)
 *      — 거리는 hop 수가 아닌 채널 레이턴시(사이클) 합계 사용
 *   4) prev[neighbor] 갱신으로 경로 역추적 정보 유지
 *
 * 경로 후처리 (routing_table 채우기):
 *   - prev[i]==-1 이면 i는 r_start 자신 → 로컬 노드의 출력 포트를 직접 기록
 *   - prev[i]!=-1 이면 prev 체인을 역추적하여 r_start의 첫 번째 홉 라우터(neighbor) 특정
 *     → r_start에서 neighbor로 나가는 출력 포트를 해당 라우터에 연결된 모든 노드에 기록
 *   - 같은 중간 라우터를 통해 도달하는 여러 노드는 동일 out_port를 공유
 *
 * 복잡도: O(_size^2) (rlist에서 선형 탐색으로 최솟값 선택)
 * 주의: dist와 prev 배열은 new[]로 할당하지만 delete[]로 해제하지 않음 (메모리 누수).
 *
 * 호출 체인:
 *   buildRoutingTable() → [route(r_start)] for each r_start in [0, _size)
 */
void AnyNet::route(int r_start){
  int* dist = new int[_size];   // [한국어] Dijkstra 거리 배열 — dist[i] = r_start에서 라우터 i까지 최단 거리(레이턴시 합)
  int* prev = new int[_size];   // [한국어] 최단 경로 역추적 배열 — prev[i] = i로 가는 최단 경로에서 i의 직전 라우터 ID
  set<int> rlist;               // [한국어] Dijkstra 미방문 라우터 집합 (set<int>으로 삽입/삭제 O(log n))
  for(int i = 0; i<_size; i++){
    dist[i] =  numeric_limits<int>::max();  // [한국어] 초기 거리: INT_MAX (도달 불가로 가정)
    prev[i] = -1;                           // [한국어] 초기 이전 라우터: -1 (없음 / r_start 자신)
    rlist.insert(i);                        // [한국어] 전체 라우터를 미방문 집합에 삽입
  }
  dist[r_start] = 0;  // [한국어] 출발 라우터 자신까지의 거리 = 0
  while(!rlist.empty()){    // [한국어] 미방문 라우터가 남아 있는 동안 반복
    //find min
    int min_dist = numeric_limits<int>::max();  // [한국어] 현재 반복에서 최소 거리 추적 변수
    int min_cand = -1;                          // [한국어] 현재 반복에서 최소 거리 라우터 ID
    for(set<int>::iterator i = rlist.begin();
	i!=rlist.end();
	i++){                                   // [한국어] 미방문 집합 선형 탐색으로 최소 거리 라우터 탐색
      if(dist[*i]<min_dist){                  // [한국어] 현재 최소보다 작은 거리 발견
	min_dist = dist[*i];                  // [한국어] 최소 거리 갱신
	min_cand = *i;                        // [한국어] 최소 거리 라우터 ID 갱신
      }
    }
    rlist.erase(min_cand);  // [한국어] 최소 거리 라우터를 미방문 집합에서 제거 (방문 처리)

    //neighbor
    for(map<int,pair<int,int> >::iterator i = router_list[1][min_cand].begin();
	i!=router_list[1][min_cand].end();
	i++){                              // [한국어] min_cand의 모든 인접 라우터(단방향 출력 연결) 순회
      int new_dist = dist[min_cand] + i->second.second;//distance is hops not cycles
      // [한국어] 완화 비용 계산: min_cand까지의 거리 + min_cand→인접 라우터 채널 레이턴시
      //         (주석의 "hops not cycles"는 오해 — 실제로는 레이턴시(사이클) 기반)
      if(new_dist < dist[i->first]){   // [한국어] 더 짧은 경로 발견 — 거리 완화(relaxation)
	dist[i->first] = new_dist;     // [한국어] 인접 라우터까지의 최단 거리 갱신
	prev[i->first] = min_cand;    // [한국어] 인접 라우터의 이전 라우터를 min_cand로 갱신
      }
    }
  }

  //post process from the prev list
  for(int i = 0; i<_size; i++){       // [한국어] 모든 라우터에 대해 경로 역추적으로 routing_table 채우기
    if(prev[i] ==-1){ //self           // [한국어] prev[i]==-1 이면 i는 r_start 자신 (자기 자신으로의 경로)
      assert(i == r_start);            // [한국어] r_start 외에 prev==-1인 라우터가 있으면 비연결 토폴로지 오류
      for(map<int, pair<int, int> >::iterator iter = router_list[0][i].begin();
	  iter!=router_list[0][i].end();
	  iter++){                       // [한국어] r_start에 직접 연결된 모든 노드 순회
	routing_table[r_start][iter->first]=iter->second.first;  // [한국어] 로컬 노드의 출력 포트를 routing_table에 직접 기록
	//cout<<"node "<<iter->first<<" port "<< iter->second.first<<endl;
      }
    } else {  // [한국어] r_start에서 라우터 i로의 경로가 존재하는 경우 (i != r_start)
      int distance=0;
      int neighbor=i;
      while(prev[neighbor]!=r_start){  // [한국어] prev 체인을 역추적하여 r_start의 첫 번째 홉 라우터(neighbor)를 찾음
	assert(router_list[1][neighbor].count(prev[neighbor])>0);  // [한국어] prev 링크가 실제 연결인지 검증
	distance+=router_list[1][prev[neighbor]][neighbor].second;//REVERSE lat
        // [한국어] 역추적 경로의 누적 레이턴시 합산 (REVERSE: 역방향 링크의 레이턴시 — 비대칭 네트워크 대응)
	neighbor= prev[neighbor];  // [한국어] 한 단계 r_start 방향으로 이동
      }
      distance+=router_list[1][prev[neighbor]][neighbor].second;//lat
      // [한국어] 마지막 홉(r_start→neighbor)의 레이턴시 추가

      assert( router_list[1][r_start].count(neighbor)!=0);  // [한국어] r_start에서 neighbor로의 직접 연결이 존재하는지 검증
      int port = router_list[1][r_start][neighbor].first;   // [한국어] r_start에서 neighbor로 나가는 출력 포트 번호 조회
      for(map<int, pair<int,int> >::iterator iter = router_list[0][i].begin();
	  iter!=router_list[0][i].end();
	  iter++){                          // [한국어] 라우터 i에 직접 연결된 모든 노드 순회
	routing_table[r_start][iter->first]=port;  // [한국어] r_start에서 이 노드로 가려면 port 방향으로 나가야 함을 기록
	//cout<<"node "<<iter->first<<" port "<< port<<" dist "<<distance<<endl;
      }
    }
  }
}


/*
 * [한국어]
 * AnyNet::readFile - 토폴로지 텍스트 파일을 5단계 상태 머신으로 파싱
 *
 * @return: 없음 (node_list, router_list[0], router_list[1] 멤버 구성)
 *
 * 파일을 한 줄씩 읽어 각 토큰을 5단계 상태 머신으로 파싱한다:
 *   HEAD_TYPE : "router" 또는 "node" — 라인의 주체 유형 결정
 *   HEAD_ID   : 주체의 ID 번호 파싱, router_list 초기 엔트리 생성
 *   BODY_TYPE : 연결 대상 유형("router"/"node") 결정
 *   BODY_ID   : 연결 대상 ID 파싱, node_list/router_list 갱신
 *   LINK_WEIGHT: 선택적 레이턴시 값 파싱 (없으면 기본값 1)
 *
 * LINK_WEIGHT 상태에서 다음 토큰이 "router"/"node"이면 새 연결 파싱으로 fall-through.
 * router-router 연결은 양방향으로 자동 등록 (명시된 방향만 레이턴시 지정, 역방향은 1).
 * node-node 연결은 오류로 처리.
 * 한 노드가 두 라우터에 연결되면 오류로 처리.
 *
 * 파싱 완료 후 검증:
 *   - router_list[0].size() == router_list[1].size() (모든 라우터가 두 맵에 등록되었는지)
 *   - 노드 ID가 0부터 시작하는 순차 번호인지 확인 (sort 후 i==node_check[i])
 *
 * 에러 처리: 파일 열기 실패 시 exit(-1); 파싱 오류 시 assert(false).
 *
 * 호출 체인:
 *   _ComputeSize() → [readFile()]
 */
void AnyNet::readFile(){

  ifstream network_list;   // [한국어] 토폴로지 파일을 읽기 위한 입력 파일 스트림
  string line;             // [한국어] 한 번에 읽는 한 줄 문자열
  enum ParseState{HEAD_TYPE=0,  // [한국어] 라인 시작 — 주체 유형("router"/"node") 파싱 대기
		  HEAD_ID,        // [한국어] 주체 ID 번호 파싱 대기
		  BODY_TYPE,      // [한국어] 연결 대상 유형 파싱 대기
		  BODY_ID,        // [한국어] 연결 대상 ID 번호 파싱 대기
		  LINK_WEIGHT};   // [한국어] 선택적 레이턴시 값 파싱 대기 (없으면 기본값 1)
  enum ParseType{NODE=0,        // [한국어] 파싱 유형 — 터미널 노드 (0 = router_list[NODE] 인덱스)
		 ROUTER,          // [한국어] 파싱 유형 — 라우터 (1 = router_list[ROUTER] 인덱스)
		 UNKNOWN};        // [한국어] 파싱 유형 — 아직 결정되지 않은 상태

  network_list.open(file_name.c_str());  // [한국어] file_name 경로의 토폴로지 파일 열기
  if(!network_list.is_open()){           // [한국어] 파일 열기 실패 — 경로 오류 또는 파일 없음
    cout<<"Anynet:can't open network file "<<file_name<<endl;  // [한국어] 오류 메시지 출력
    exit(-1);  // [한국어] 파일 없으면 시뮬레이션 불가 — 강제 종료
  }

  //loop through the entire file
  while(!network_list.eof()){    // [한국어] 파일 끝(EOF)까지 한 줄씩 처리
    getline(network_list,line);  // [한국어] 다음 줄을 line 버퍼로 읽기
    if(line==""){                // [한국어] 빈 줄 — 건너뜀 (토폴로지 파일의 빈 줄 허용)
      continue;
    }

    ParseState state=HEAD_TYPE;  // [한국어] 각 줄 파싱 시작 시 상태 머신을 HEAD_TYPE으로 리셋
    //position to parse out white sspace
    int pos = 0;          // [한국어] 현재 파싱 위치 (공백 기준 토큰 추출에 사용)
    int next_pos=-1;      // [한국어] 다음 공백 위치 — find()의 반환값
    string temp;          // [한국어] 현재 처리 중인 토큰 문자열
    //the first node and its type
    int head_id = -1;           // [한국어] 파싱된 주체 ID (라인의 첫 번째 router/node)
    ParseType head_type = UNKNOWN;  // [한국어] 파싱된 주체 유형 (ROUTER 또는 NODE)
    //stuff that head are linked to
    ParseType body_type = UNKNOWN;  // [한국어] 현재 파싱 중인 연결 대상 유형
    int body_id = -1;               // [한국어] 현재 파싱 중인 연결 대상 ID
    int link_weight = 1;            // [한국어] 현재 연결의 레이턴시 — 기본값 1사이클

    do{

      //skip empty spaces
      next_pos = line.find(" ",pos);           // [한국어] pos 이후 첫 번째 공백 위치 탐색
      temp = line.substr(pos,next_pos-pos);    // [한국어] [pos, next_pos) 범위의 토큰 추출
      pos = next_pos+1;                        // [한국어] 다음 토큰 시작 위치로 이동 (공백 1개 건너뜀)
      if(temp=="" || temp==" "){               // [한국어] 빈 토큰 또는 공백 토큰 — 건너뜀 (연속 공백 처리)
	continue;
      }

      switch(state){
      case HEAD_TYPE:  // [한국어] 라인의 첫 토큰 — 주체 유형 결정
	if(temp=="router"){
	  head_type = ROUTER;   // [한국어] 주체가 라우터
	} else if (temp == "node"){
	  head_type = NODE;     // [한국어] 주체가 터미널 노드
	} else {
	  cout<<"Anynet:Unknow head of line type "<<temp<<"\n";  // [한국어] "router"/"node" 외 토큰 — 파싱 오류
	  assert(false);  // [한국어] 알 수 없는 주체 유형 — 즉시 종료
	}
	state=HEAD_ID;  // [한국어] 다음 토큰은 ID 번호
	break;
      case HEAD_ID:  // [한국어] 주체 ID 번호 파싱
	//need better error check
	head_id = atoi(temp.c_str());  // [한국어] 토큰을 정수로 변환하여 주체 ID 설정

	//intialize router structures
	if(router_list[NODE].count(head_id) == 0){  // [한국어] 이 라우터의 노드 연결 맵이 없으면 생성
	  router_list[NODE][head_id] = map<int, pair<int,int> >();  // [한국어] 빈 노드 연결 맵 초기화
	}
	if(router_list[ROUTER].count(head_id) == 0){  // [한국어] 이 라우터의 라우터 연결 맵이 없으면 생성
	  router_list[ROUTER][head_id] = map<int, pair<int,int> >();  // [한국어] 빈 라우터 연결 맵 초기화
	}

	state=BODY_TYPE;  // [한국어] 다음 토큰은 연결 대상 유형
	break;
      case LINK_WEIGHT:  // [한국어] 선택적 레이턴시 값 파싱 — "router"/"node"이면 새 연결로 fall-through
	if(temp=="router"||
	   temp == "node"){
	  //ignore
	} else {  // [한국어] 숫자 토큰이면 레이턴시 값으로 파싱
	  link_weight= atoi(temp.c_str());  // [한국어] 레이턴시 값 (사이클) 파싱
	  router_list[head_type][head_id][body_id].second=link_weight;  // [한국어] 해당 연결의 레이턴시(pair.second) 갱신
	  break;
	}
	//intentionally letting it flow through
        // [한국어] 의도적 fall-through: LINK_WEIGHT 토큰이 "router"/"node"이면 새 BODY_TYPE으로 처리
      case BODY_TYPE:  // [한국어] 연결 대상 유형 파싱
	if(temp=="router"){
	  body_type = ROUTER;   // [한국어] 연결 대상이 라우터
	} else if (temp == "node"){
	  body_type = NODE;     // [한국어] 연결 대상이 터미널 노드
	} else {
	  cout<<"Anynet:Unknow body type "<<temp<<"\n";  // [한국어] 알 수 없는 대상 유형 — 오류
	  assert(false);
	}
	state=BODY_ID;  // [한국어] 다음 토큰은 연결 대상 ID
	break;
      case BODY_ID:  // [한국어] 연결 대상 ID 파싱 및 연결 정보 등록
	body_id = atoi(temp.c_str());	// [한국어] 연결 대상 ID 파싱
	//intialize router structures if necessary
	if(body_type==ROUTER){   // [한국어] 대상이 라우터이면 해당 라우터의 맵도 초기화 (아직 없는 경우)
	  if(router_list[NODE].count(body_id) ==0){
	    router_list[NODE][body_id] = map<int, pair<int,int> >();    // [한국어] 대상 라우터의 노드 연결 맵 초기화
	  }
	  if(router_list[ROUTER].count(body_id) == 0){
	    router_list[ROUTER][body_id] = map<int, pair<int,int> >();  // [한국어] 대상 라우터의 라우터 연결 맵 초기화
	  }
	}

	if(head_type==NODE && body_type==NODE){   // [한국어] 노드-노드 직접 연결 — 지원하지 않음

	  cout<<"Anynet:Cannot connect node to node "<<temp<<"\n";
	  assert(false);  // [한국어] 노드는 라우터를 통해서만 연결 가능

	} else if(head_type==NODE && body_type==ROUTER){  // [한국어] 노드가 라우터에 연결되는 경우

	  if(node_list.count(head_id)!=0 &&
	     node_list[head_id]!=body_id){  // [한국어] 이 노드가 이미 다른 라우터에 등록된 경우 — 오류
	    cout<<"Anynet:Node "<<body_id<<" trying to connect to multiple router "
		<<body_id<<" and "<<node_list[head_id]<<endl;
	    assert(false);  // [한국어] 하나의 노드는 하나의 라우터에만 연결 가능
	  }
	  node_list[head_id]=body_id;  // [한국어] node_list[노드 ID] = 연결 라우터 ID 등록
	  router_list[NODE][body_id][head_id]=pair<int, int>(-1,1);  // [한국어] 라우터의 노드 연결 맵에 추가 (포트=-1 임시, 레이턴시=1 기본)

	} else if(head_type==ROUTER && body_type==NODE){  // [한국어] 라우터가 노드에 연결되는 경우
	  //insert and check node
	  if(node_list.count(body_id) != 0 &&
	     node_list[body_id]!=head_id){  // [한국어] 이 노드가 이미 다른 라우터에 등록된 경우 — 오류
	    cout<<"Anynet:Node "<<body_id<<" trying to connect to multiple router "
		<<body_id<<" and "<<node_list[head_id]<<endl;
	    assert(false);  // [한국어] 동일 노드의 다중 라우터 연결 금지
	  }
	  node_list[body_id] = head_id;  // [한국어] node_list[노드 ID] = 연결 라우터 ID 등록
	  router_list[NODE][head_id][body_id]=pair<int, int>(-1,1);  // [한국어] 라우터의 노드 연결 맵에 추가 (포트=-1 임시, 레이턴시=1 기본)

	} else if(head_type==ROUTER && body_type==ROUTER){  // [한국어] 라우터-라우터 연결 (단방향)
	  router_list[ROUTER][head_id][body_id]=pair<int, int>(-1,1);  // [한국어] head→body 방향 링크 등록 (포트=-1, 레이턴시=1 기본)
	  if(router_list[ROUTER][body_id].count(head_id)==0){           // [한국어] 역방향 링크가 없으면 자동으로 등록 (양방향 자동 보완)
	    router_list[ROUTER][body_id][head_id]=pair<int, int>(-1,1); // [한국어] body→head 역방향 링크 등록 (레이턴시=1, 명시 시 나중에 갱신됨)
	  }
	}
	state=LINK_WEIGHT;  // [한국어] 다음 토큰은 선택적 레이턴시 값 또는 다음 대상 유형
	break ;
      default:
	cout<<"Anynet:Unknow parse state\n";  // [한국어] 도달 불가 상태 — 상태 머신 버그
	assert(false);
	break;
      }

    } while(pos!=0);  // [한국어] pos==0이 되면 더 이상 토큰 없음 (find가 npos 반환 → +1=0)
    if(state!=LINK_WEIGHT &&
       state!=BODY_TYPE){  // [한국어] 줄 파싱이 BODY_ID 이전에 끊기면 불완전한 라인 경고
      cout<<"Anynet:Incomplete parse of the line: "<<line<<endl;
    }

  }

  //map verification, make sure the information contained in bother maps
  //are the same
  assert(router_list[0].size() == router_list[1].size());  // [한국어] 노드 연결 맵과 라우터 연결 맵의 라우터 수 일치 검증

  //traffic generator assumes node list is sequenctial and starts at 0
  vector<int> node_check;  // [한국어] 노드 ID 목록을 수집하여 순차 번호 여부 검증
  for(map<int,int>::iterator i = node_list.begin();
      i!=node_list.end();
      i++){
    node_check.push_back(i->first);  // [한국어] 모든 노드 ID를 벡터에 수집
  }
  sort(node_check.begin(), node_check.end());  // [한국어] 노드 ID를 오름차순 정렬
  for(size_t i = 0; i<node_check.size(); i++){
    if(node_check[i] != (int)i){  // [한국어] 정렬된 i번째 ID가 i가 아니면 순차 번호 아님
      cout<<"Anynet:booksim trafficmanager assumes sequential node numbering starting at 0\n";
      assert(false);  // [한국어] 트래픽 제너레이터가 0부터 연속된 노드 번호를 가정하므로 필수 조건
    }
  }

}

