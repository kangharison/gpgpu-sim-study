// $Id: main.cpp 5487 2013-02-27 08:16:18Z qtedq $

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

/*main.cpp
 *
 *The starting point of the network simulator
 *-Include all network header files
 *-initilize the network
 *-initialize the traffic manager and set it to run
 *
 *
 */

/*
 * [한국어 설명] BookSim NoC 시뮬레이터 독립 실행형 메인 진입점 (main.cpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 BookSim NoC 시뮬레이터를 독립적인 명령줄 프로그램으로 실행할 때
 * 사용되는 main() 함수와 Simulate() 함수를 제공한다. 설정 파일을 파싱하고,
 * Network 객체들과 TrafficManager를 생성한 뒤 시뮬레이션을 실행한다.
 * GPGPU-Sim 연동 모드에서는 이 main() 대신 libcuda/libopencl 측에서
 * InterconnectInterface를 직접 생성/제어하므로 이 main()은 호출되지 않는다.
 * 단, CREATE_LIBRARY가 정의된 경우 이 main()은 컴파일에서 제외된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 독립 실행형 BookSim 시뮬레이터의 진입점. 라이브러리 모드(CREATE_LIBRARY) 시
 * intersim2 낸부의 Network/TrafficManager/GPUTrafficManager가 GPGPU-Sim 측
 * icnt_wrapper.cc에 의해 직접 구동된다.
 *
 * 실행 컨텍스트: 호스트 유저스페이스 — 단일 프로세스/단일 스레드.
 *
 * === 타 모듈과의 연결 ===
 * - booksim.hpp        : BookSim 공통 매크로/전방 선언.
 * - routefunc.hpp      : InitializeRoutingMap() — 라우팅 함수 등록.
 * - traffic.hpp        : 트래픽 패턴 등록.
 * - booksim_config.hpp : BookSimConfig 클래스 — 설정 파싱/저장.
 * - trafficmanager.hpp : TrafficManager — 시뮬레이션 메인 루프.
 * - interconnect_interface.hpp : GPGPU-Sim 연동 인터페이스 (독립 실행 시 미사용).
 *
 * === 주요 함수/구조체 요약 ===
 * - GetSimTime()      : 현재 시뮬레이션 시간 반환 (g_icnt_interface 우선).
 * - GetStats()        : 이름으로 Stats 객체 조회 (g_icnt_interface 우선).
 * - Simulate(config)  : Network/TrafficManager 생성 → Run() → 정리.
 * - main(argc, argv)  : 설정 파싱 → InitializeRoutingMap() → Simulate() 호출.
 */

#include <sys/time.h> /* [한국어] gettimeofday() — 시뮬레이션 실행 시간 측정용 */

#include <string>    /* [한국어] std::string — 설정값/파일명 처리 */
#include <cstdlib>   /* [한국어] exit(), atoi() 등 표준 라이브러리 */
#include <iostream>  /* [한국어] std::cout, std::cerr — 메시지 출력 */
#include <fstream>   /* [한국어] std::ofstream — watch_out 파일 출력 */

#include <sstream>   /* [한국어] std::ostringstream — 네트워크 이름 생성 */
#include "booksim.hpp"             /* [한국어] BookSim 공통 헤더 */
#include "routefunc.hpp"           /* [한국어] 라우팅 함수 맵 초기화 */
#include "traffic.hpp"             /* [한국어] 트래픽 패턴 등록 */
#include "booksim_config.hpp"      /* [한국어] BookSimConfig 설정 클래스 */
#include "trafficmanager.hpp"      /* [한국어] TrafficManager 시뮬레이션 루프 */
#include "random_utils.hpp"        /* [한국어] 난수 시드 초기화 */
#include "network.hpp"             /* [한국어] Network 클래스 */
#include "injection.hpp"           /* [한국어] 주입 프로세스 등록 */
#include "power_module.hpp"        /* [한국어] 전력 분석 모듈 */
#include "interconnect_interface.hpp" /* [한국어] GPGPU-Sim 연동 인터페이스 */



///////////////////////////////////////////////////////////////////////////////
//Global declarations
//////////////////////

// Interconnect Interface instance
InterconnectInterface *g_icnt_interface; /* [한국어] GPGPU-Sim 연동 시 icnt_wrapper에서 생성/할당하는 전역 인터페이스 포인터 */

/* the current traffic manager instance */
TrafficManager * trafficManager = NULL; /* [한국어] 독립 실행형 BookSim의 TrafficManager 전역 포인터. Simulate()에서 생성. */
#if 0

int GetSimTime() {
    return trafficManager->getTime();
}

class Stats;
Stats * GetStats(const std::string & name) {
    Stats* test =  trafficManager->getStats(name);
    if(test == 0){
        cout<<"warning statistics "<<name<<" not found"<<endl;
    }
    return test;
}
#else
/*
 * [한국어]
 * GetSimTime — 현재 NoC 시뮬레이션 사이클 번호 반환.
 * @return: 현재 시뮬레이션 시간.
 * GPGPU-Sim 연동 시 g_icnt_interface를 우선 사용하고, 없으면 trafficManager를 사용.
 */
int GetSimTime() {
  return g_icnt_interface->GetIcntTime(); /* [한국어] InterconnectInterface가 제공하는 NoC 시간 반환 */
}

class Stats;
/*
 * [한국어]
 * GetStats — 이름으로 BookSim Stats 객체를 조회.
 * @name: 통계 객체 이름 (예: "plat").
 * @return: Stats 포인터, 없으면 0(NULL).
 * GPGPU-Sim 연동 시 g_icnt_interface를 우선 사용.
 */
Stats * GetStats(const std::string & name) {
  Stats* test =  g_icnt_interface->GetIcntStats(name); /* [한국어] 인터커넥트 인터페이스에서 통계 객체 검색 */
  if(test == 0){
    cout<<"warning statistics "<<name<<" not found"<<endl; /* [한국어] 통계 이름이 없으면 경고 출력 */
  }
  return test;
}

#endif

/* printing activity factor*/
bool gPrintActivity; /* [한국어] 각 라우터/링크 활동(activity) 출력 플래그 — print_activity 설정 */

int gK;//radix /* [한국어] k-ary n-cube 토폴로지에서 한 차원의 라우터 수(radix) */
int gN;//dimension /* [한국어] 네트워크 차원 수 */
int gC;//concentration /* [한국어] 각 라우터에 연결된 노드 수(concentration) */

int gNodes; /* [한국어] 전체 네트워크 노드 수 */

//generate nocviewer trace
bool gTrace; /* [한국어] nocviewer 추적(trace) 파일 생성 플래그 — viewer_trace 설정 */

ostream * gWatchOut; /* [한국어] 디버그/워치 출력 대상 스트림 포인터 — watch_out 설정 */



///////////////////////////////////////////////////////////////////////////////

/*
 * [한국어]
 * Simulate — 독립 실행형 BookSim 시뮬레이션을 구성하고 실행.
 * @config: 파싱 완료된 BookSimConfig 객체.
 * @return: true = 시뮬레이션이 비정상 종료(예: 교착 상태 감지), false = 정상 종료.
 *
 * 동작 과정:
 *   1) subnets 수만큼 Network 객체 생성.
 *   2) TrafficManager 생성 및 Run() 호출.
 *   3) 시뮬레이션 wall-clock 시간 출력.
 *   4) sim_power 설정 시 각 서브넷에 대해 Power_Module 실행.
 *   5) Network/TrafficManager 메모리 해제.
 */
bool Simulate( BookSimConfig const & config )
{
  vector<Network *> net; /* [한국어] 서브넷별 Network 객체 포인터 벡터 */

  int subnets = config.GetInt("subnets"); /* [한국어] 설정 파일의 subnets 값으로 생성할 네트워크 수 결정 */
  /*To include a new network, must register the network here
   *add an else if statement with the name of the network
   */

  net.resize(subnets); /* [한국어] 벡터 크기를 서브넷 수만큼 확장 */
  for (int i = 0; i < subnets; ++i) {
    ostringstream name; /* [한국어] 각 서브넷 Network의 고유 이름 생성 */
    name << "network_" << i; /* [한국어] "network_0", "network_1" 형태 이름 */
    net[i] = Network::New( config, name.str() ); /* [한국어] 토폴로지/topology 설정에 따라 Network 객체 생성 */
  }

  /*tcc and characterize are legacy
   *not sure how to use them 
   */

  assert(trafficManager == NULL); /* [한국어] 중복 TrafficManager 생성 방지 */
  trafficManager = TrafficManager::New( config, net ) ; /* [한국어] 설정과 Network들을 기반으로 TrafficManager 생성 */

  /*Start the simulation run
   */

  double total_time; /* Amount of time we've run */
  struct timeval start_time, end_time; /* Time before/after user code */
  total_time = 0.0;
  gettimeofday(&start_time, NULL); /* [한국어] 시뮬레이션 시작 시각 기록 */

  bool result = trafficManager->Run() ; /* [한국어] 메인 시뮬레이션 루프 실행 */


  gettimeofday(&end_time, NULL); /* [한국어] 시뮬레이션 종료 시각 기록 */
  total_time = ((double)(end_time.tv_sec) + (double)(end_time.tv_usec)/1000000.0)
            - ((double)(start_time.tv_sec) + (double)(start_time.tv_usec)/1000000.0); /* [한국어] wall-clock 시간(초) 계산 */

  cout<<"Total run time "<<total_time<<endl; /* [한국어] 총 실행 시간 출력 */

  for (int i=0; i<subnets; ++i) {

    ///Power analysis
    if(config.GetInt("sim_power") > 0){ /* [한국어] sim_power 설정이 켜져 있으면 전력 분석 수행 */
      Power_Module pnet(net[i], config); /* [한국어] i번째 네트워크에 대한 Power_Module 생성 */
      pnet.run(); /* [한국어] 전력 분석 실행 및 결과 출력 */
    }

    delete net[i]; /* [한국어] Network 객체 메모리 해제 */
  }

  delete trafficManager; /* [한국어] TrafficManager 메모리 해제 */
  trafficManager = NULL; /* [한국어] 해제 후 NULL 설정 — 재실행/중복 해제 방지 */

  return result;
}

#ifdef CREATE_LIBRARY

#else
/*
 * [한국어]
 * main — 독립 실행형 BookSim 시뮬레이터의 명령줄 진입점.
 * @argc: 명령줄 인자 개수.
 * @argv: 인자 배열. argv[1]부터 설정 파일 및 param=value 쌍.
 * @return: 0 = 정상 종료, -1 = 시뮬레이션에서 문제 발생(Simulate() 반환 true).
 *
 * CREATE_LIBRARY이 정의되면 이 main()은 컴파일되지 않고 라이브러리 형태로 사용된다.
 */
int main( int argc, char **argv )
{

  BookSimConfig config; /* [한국어] 설정값을 저장할 BookSimConfig 객체 */


  if ( !ParseArgs( &config, argc, argv ) ) { /* [한국어] 명령줄 인자와 설정 파일을 파싱 */
    cerr << "Usage: " << argv[0] << " configfile... [param=value...]" << endl; /* [한국어] 파싱 실패 시 사용법 출력 */
    return 0;
 } 

  
  /*initialize routing, traffic, injection functions
   */
  InitializeRoutingMap( config ); /* [한국어] topology/routing 설정에 따른 라우팅 함수 등록 */

  gPrintActivity = (config.GetInt("print_activity") > 0); /* [한국어] print_activity 설정으로 활동 출력 플래그 초기화 */
  gTrace = (config.GetInt("viewer_trace") > 0); /* [한국어] viewer_trace 설정으로 trace 플래그 초기화 */
  
  string watch_out_file = config.GetStr( "watch_out" ); /* [한국어] watch_out 설정값(파일명) 읽기 */
  if(watch_out_file == "") {
    gWatchOut = NULL; /* [한국어] 파일명이 비어 있으면 워치 출력 비활성화 */
  } else if(watch_out_file == "-") {
    gWatchOut = &cout; /* [한국어] "-"이면 표준 출력에 워치 메시지 출력 */
  } else {
    gWatchOut = new ofstream(watch_out_file.c_str()); /* [한국어] 지정 파일로 워치 메시지 출력 */
  }
  

  /*configure and run the simulator
   */
  bool result = Simulate( config ); /* [한국어] 설정된 구성으로 시뮬레이션 실행 */
  return result ? -1 : 0; /* [한국어] Simulate()가 true(비정상)면 -1, false(정상)면 0 반환 */
}
#endif
