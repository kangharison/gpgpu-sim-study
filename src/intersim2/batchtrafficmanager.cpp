// $Id: batchtrafficmanager.cpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] BookSim 배치 트래픽 매니저 구현 (batchtrafficmanager.cpp)
 *
 * === 파일의 역할 ===
 * BatchTrafficManager 클래스의 모든 멤버 함수를 구현한다.
 * 이 클래스는 TrafficManager를 상속하여 "배치(batch)" 방식의 합성 트래픽 시뮬레이션을 제공한다.
 * 일반 시뮬레이터가 지속적 트래픽을 주입하는 데 반해, 이 구현은 다음 절차를 반복한다:
 *   1) 모든 노드에서 batch_size개의 패킷을 주입 완료할 때까지 Step()을 반복
 *   2) 주입 완료 후 모든 인-플라이트(in-flight) 플릿이 수신될 때까지 대기(드레인)
 *   3) 배치 소요 시간을 측정하여 Stats에 저장
 *   4) 위 과정을 batch_count번 반복
 * 이 방식은 GPU 메모리 서브시스템의 왕복 지연(RTT)과 처리율을 측정할 때 유용하다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 컨텍스트: 합성 트래픽(synthetic traffic) 시뮬레이션 전용.
 * GPGPU-Sim의 실제 GPU 시뮬레이션과는 별개이며, BookSim 독립 실행 시에만 사용된다.
 * 호출 체인:
 *   BookSim main()
 *     → TrafficManager::New() (config에 따라 BatchTrafficManager 선택)
 *     → BatchTrafficManager::BatchTrafficManager() ← [이 파일]
 *     → TrafficManager::Run()
 *         → _SingleSim() ← [이 파일의 핵심 루프]
 *             → _Step() (부모 TrafficManager, 1 사이클 진행)
 *             → _IssuePacket() ← [이 파일]
 *             → _RetireFlit() ← [이 파일]
 *     → DisplayOverallStats() ← [이 파일]
 *
 * === 타 모듈과의 연결 ===
 * - 의존: packet_reply_info.hpp (PacketReplyInfo, 응답 큐 항목),
 *         random_utils.hpp (RandomFloat, 50% 확률 read/write 결정),
 *         batchtrafficmanager.hpp (클래스 선언),
 *         trafficmanager.hpp (부모 클래스, _Step, _RetireFlit, _IssuePacket 등)
 * - 피의존: TrafficManager::New() 팩토리 함수 (sim_type="batch"일 때 이 클래스 선택)
 * - 부모로부터 상속받아 사용하는 핵심 멤버:
 *     _nodes: 총 노드 수 (루프 범위)
 *     _time: 현재 사이클 카운터
 *     _classes: 트래픽 클래스 수
 *     _packet_seq_no[node]: 노드별 주입 완료 패킷 시퀀스 번호
 *     _requestsOutstanding[node]: 노드별 미완료 요청 수
 *     _repliesPending[node]: 노드별 응답 대기 큐
 *     _total_in_flight_flits[cl]: 클래스별 비행 중 플릿 집합
 *     _use_read_write[cl]: 클래스별 read/write 모드 여부
 *     _sim_state: 시뮬레이션 상태 (running/draining)
 *     _drain_time: 드레인 시작 사이클
 *     _total_sims: 누적 시뮬레이션 실행 횟수
 *
 * === 주요 함수/구조체 요약 ===
 * BatchTrafficManager() — 설정 읽기, Stats/파일스트림 초기화
 * ~BatchTrafficManager() — Stats와 파일스트림 해제
 * _RetireFlit()  — 마지막 플릿 ID/PID 추적 후 부모 위임
 * _IssuePacket() — batch_size/max_outstanding 조건으로 주입 결정
 * _SingleSim()   — 배치 루프의 핵심: 주입→드레인→통계 반복
 * _UpdateOverallStats() — 배치 시간을 전역 누적 통계에 반영
 * _OverallStatsCSV()    — CSV 문자열에 배치 시간 min/avg/max 추가
 * WriteStats()          — 배치 평균 시간 기록
 * DisplayStats()        — 배치 min/avg/max 시간 콘솔 출력
 * DisplayOverallStats() — 전체 누적 배치 시간 콘솔 출력
 */

#include <limits>    // [한국어] std::numeric_limits — 정수 최대값 등 (직접 사용은 없으나 관련 헤더)
#include <sstream>   // [한국어] ostringstream — _OverallStatsCSV()에서 CSV 문자열 조립에 사용
#include <fstream>   // [한국어] ofstream — sent_packets_out 파일 스트림 생성에 사용

#include "packet_reply_info.hpp"     // [한국어] PacketReplyInfo — 응답 대기 큐(_repliesPending) 항목 타입
#include "random_utils.hpp"          // [한국어] RandomFloat() — _IssuePacket()에서 read/write 비율 결정
#include "batchtrafficmanager.hpp"   // [한국어] BatchTrafficManager 클래스 선언

/*
 * [한국어]
 * BatchTrafficManager::BatchTrafficManager - 배치 트래픽 매니저 생성자
 *
 * @param config: BookSimConfig 설정 객체 (batch_size, batch_count, max_outstanding_requests,
 *                sent_packets_out 파라미터를 읽음)
 * @param net: 시뮬레이션할 네트워크 포인터 벡터 (부모 TrafficManager로 전달)
 * @return: (생성자, 반환값 없음)
 *
 * 부모 TrafficManager를 초기화한 뒤 배치 전용 파라미터를 설정한다.
 * _last_id와 _last_pid를 -1로 초기화하여 "아직 완료된 플릿 없음"을 표시한다.
 * _batch_time Stats 객체를 생성하고 _stats 맵에 등록하여 공통 통계 인프라와 연동한다.
 * sent_packets_out 파일 경로가 설정된 경우 ofstream을 열어 배치 진행 로그를 기록한다.
 *
 * 실행 컨텍스트: 시뮬레이터 초기화 단계, 싱글스레드.
 * 호출 체인:
 *   TrafficManager::New() → [BatchTrafficManager::BatchTrafficManager()]
 */
BatchTrafficManager::BatchTrafficManager( const Configuration &config,
					  const vector<Network *> & net )
: TrafficManager(config, net), _last_id(-1), _last_pid(-1),
   _overall_min_batch_time(0), _overall_avg_batch_time(0),
   _overall_max_batch_time(0)
{
  // [한국어] 부모 TrafficManager 초기화 목록: _last_id=-1(미완료 표시), _last_pid=-1(동일),
  //          전체 누적 배치 시간 통계 변수를 0으로 초기화.

  _max_outstanding = config.GetInt ("max_outstanding_requests");
  /* [한국어] 노드당 동시 미완료 요청 최대 수를 설정에서 읽는다.
   * 0이면 무제한 주입, 양수이면 _IssuePacket()에서 흐름 제어에 사용. */

  _batch_size = config.GetInt( "batch_size" );
  /* [한국어] 한 배치에서 각 노드가 주입할 총 패킷 수를 읽는다.
   * _IssuePacket()에서 _packet_seq_no[source] < _batch_size 조건으로 주입을 허용한다. */

  _batch_count = config.GetInt( "batch_count" );
  /* [한국어] 총 배치 반복 횟수를 읽는다.
   * _SingleSim()의 while(batch_index < _batch_count) 루프 횟수를 결정한다. */

  _batch_time = new Stats( this, "batch_time", 1.0, 1000 );
  /* [한국어] 배치 소요 시간 통계 객체를 생성한다.
   * 파라미터: this(소유자), "batch_time"(이름), 1.0(빈 크기), 1000(히스토그램 빈 수).
   * 1.0 사이클 단위 해상도로 최대 1000 사이클 분포를 기록한다. */

  _stats["batch_time"] = _batch_time;
  /* [한국어] 부모 TrafficManager의 _stats 맵에 등록하여
   * TrafficManager::_ClearStats() 등 공통 통계 관리 로직과 연동한다.
   * "batch_time" 키로 접근 가능하다. */

  string sent_packets_out_file = config.GetStr( "sent_packets_out" );
  /* [한국어] 배치 진행 로그를 기록할 파일 경로를 설정에서 읽는다.
   * 빈 문자열이면 로그를 비활성화한다. */

  if(sent_packets_out_file == "") {
    /* [한국어] 파일 경로가 지정되지 않은 경우 — 로그 비활성화. */
    _sent_packets_out = NULL;         // [한국어] NULL로 설정하면 _SingleSim()에서 로그 출력을 건너뜀
  } else {
    /* [한국어] 파일 경로가 지정된 경우 — ofstream을 열어 _sent_packets_out에 저장. */
    _sent_packets_out = new ofstream(sent_packets_out_file.c_str());
    /* [한국어] c_str()은 C 스타일 문자열 포인터 변환 (구버전 ofstream 생성자 호환). */
  }
}

/*
 * [한국어]
 * BatchTrafficManager::~BatchTrafficManager - 소멸자
 *
 * @return: (소멸자, 반환값 없음)
 *
 * 생성자에서 동적으로 할당한 _batch_time Stats 객체와
 * _sent_packets_out 파일 스트림 객체를 해제한다.
 * 부모 TrafficManager 소멸자는 C++ 소멸 체인에 의해 자동 호출된다.
 *
 * 실행 컨텍스트: 시뮬레이터 종료 단계, 싱글스레드.
 * 호출 체인:
 *   main() 종료 → delete trafficmanager → [BatchTrafficManager::~BatchTrafficManager()]
 */
BatchTrafficManager::~BatchTrafficManager( )
{
  delete _batch_time;
  /* [한국어] 배치 시간 통계 객체를 해제한다.
   * _stats 맵에도 등록되어 있지만, 포인터 소유권은 이 클래스에 있으므로 여기서 해제. */

  if(_sent_packets_out) delete _sent_packets_out;
  /* [한국어] 파일 스트림이 열려 있으면 닫고 해제한다.
   * NULL 체크를 통해 파일이 열리지 않은 경우(sent_packets_out="")를 안전하게 처리. */
}

/*
 * [한국어]
 * BatchTrafficManager::_RetireFlit - 플릿 완료(retire) 처리 오버라이드
 *
 * @param f: 목적지에 도달한 플릿 포인터. f->id (플릿 ID)와 f->pid (패킷 ID)를 읽음.
 * @param dest: 목적지 노드 번호.
 * @return: (void)
 *
 * 부모 TrafficManager::_RetireFlit()을 호출하기 전에
 * _last_id와 _last_pid를 현재 플릿의 ID로 갱신한다.
 * 이를 통해 배치가 완료된 후 "마지막으로 수신된 플릿/패킷"을 진단 메시지에 출력할 수 있다.
 * 모든 플릿에 대해 호출되므로, 마지막에 호출된 플릿이 배치의 마지막 수신 플릿이 된다.
 *
 * 실행 컨텍스트: _Step() 내부에서 플릿이 수신될 때마다 호출, 싱글스레드.
 * 호출 체인:
 *   _Step() → Network::ReadFlit() → [_RetireFlit()] → TrafficManager::_RetireFlit()
 */
void BatchTrafficManager::_RetireFlit( Flit *f, int dest )
{
  _last_id = f->id;
  /* [한국어] 현재 retire된 플릿의 전역 ID로 갱신한다.
   * 배치 완료 후 진단 메시지 "last flit was <_last_id>"에 사용된다. */

  _last_pid = f->pid;
  /* [한국어] 현재 retire된 플릿이 속한 패킷의 ID로 갱신한다.
   * 배치 완료 후 진단 메시지 "last packet was <_last_pid>"에 사용된다. */

  TrafficManager::_RetireFlit(f, dest);
  /* [한국어] 부모 클래스의 실제 retire 로직을 수행한다.
   * 지연 통계 계산, _total_in_flight_flits에서 제거, 필요 시 응답 패킷 생성 등을 처리한다. */
}

/*
 * [한국어]
 * BatchTrafficManager::_IssuePacket - 패킷 주입 결정 함수
 *
 * @param source: 패킷을 주입할 소스 노드 번호 (0 ~ _nodes-1).
 * @param cl: 트래픽 클래스 번호 (0 ~ _classes-1).
 * @return: 주입할 패킷 크기(플릿 수). 0이면 이 사이클에 주입 없음.
 *          _use_read_write[cl]=1인 경우: -1=응답 패킷 주입, 1=write 요청, 2=read 요청.
 *          _use_read_write[cl]=0인 경우: _GetNextPacketSize(cl) 반환값.
 *
 * 배치 트래픽 모드에서 패킷 주입 여부를 결정한다.
 * 주입 허용 조건:
 *   1) _packet_seq_no[source] < _batch_size — 이 노드가 아직 배치 할당량을 채우지 않은 경우
 *   2) _max_outstanding <= 0 OR _requestsOutstanding[source] < _max_outstanding
 *      — 미완료 요청이 한도 미만인 경우
 * read/write 모드에서는 대기 중인 응답이 있으면 응답을 먼저 처리한다.
 *
 * 실행 컨텍스트: _Step() 내부에서 노드마다 매 사이클 호출, 싱글스레드.
 * 호출 체인:
 *   _Step() → [_IssuePacket(source, cl)] → (결과에 따라) _GeneratePacket()
 */
int BatchTrafficManager::_IssuePacket( int source, int cl )
{
  int result = 0;                    // [한국어] 반환값 초기화 — 기본은 "이 사이클에 주입 없음"

  if(_use_read_write[cl]) { //read write packets
    /* [한국어] read/write 요청-응답 쌍 모드가 활성화된 트래픽 클래스인 경우.
     * 응답 대기 큐를 먼저 확인하고, 요청 조건을 별도로 판단한다. */

    //check queue for waiting replies.
    //check to make sure it is on time yet
    if(!_repliesPending[source].empty()) {
      /* [한국어] 이 소스 노드에 대기 중인 응답 패킷이 있는 경우.
       * 요청에 대한 응답(reply)이 아직 전송되지 않았다는 의미이다. */

      if(_repliesPending[source].front()->time <= _time) {
        /* [한국어] 대기 큐의 첫 응답 패킷의 예정 전송 시간이 현재 사이클 이하인 경우.
         * 즉, 응답을 전송해야 할 시점이 도래했다. */
        result = -1;
        /* [한국어] -1은 응답(reply) 패킷을 주입하라는 신호.
         * 부모 TrafficManager::_Step()이 -1을 수신하면 응답 패킷을 생성한다. */
      }
    } else {
      /* [한국어] 대기 중인 응답이 없는 경우 — 새 요청 패킷 주입 가능 여부를 판단. */

      if((_packet_seq_no[source] < _batch_size) &&
	 ((_max_outstanding <= 0) ||
	  (_requestsOutstanding[source] < _max_outstanding))) {
        /* [한국어] 주입 허용 조건:
         *   (1) 이 노드의 배치 할당량이 아직 남아 있음 (_packet_seq_no < _batch_size)
         *   (2) max_outstanding 제한이 없거나, 현재 미완료 요청이 한도 미만임 */

	//coin toss to determine request type.
	result = (RandomFloat() < 0.5) ? 2 : 1;
        /* [한국어] 50% 확률로 read 요청(2) 또는 write 요청(1)을 결정한다.
         * RandomFloat()은 [0.0, 1.0) 균일 랜덤 값을 반환한다.
         * 2 = read request, 1 = write request (부모 TrafficManager 규약). */

	_requestsOutstanding[source]++;
        /* [한국어] 이 소스 노드의 미완료 요청 수를 1 증가시킨다.
         * 응답이 도착해 _repliesPending에서 처리될 때 감소한다. */
      }
    }
  } else { //normal
    /* [한국어] 일반 트래픽 모드 (read/write 구분 없음). */

    if((_packet_seq_no[source] < _batch_size) &&
       ((_max_outstanding <= 0) ||
	(_requestsOutstanding[source] < _max_outstanding))) {
      /* [한국어] 배치 할당량이 남아 있고 미완료 요청이 한도 미만인 경우 주입 허용. */

      result = _GetNextPacketSize(cl);
      /* [한국어] 트래픽 클래스 설정(packet_size 분포)에 따라 다음 패킷 크기를 결정한다.
       * 단일 크기이면 항상 같은 값, 확률 분포이면 랜덤 선택. */

      _requestsOutstanding[source]++;
      /* [한국어] 미완료 요청 수를 1 증가 — 패킷이 retire될 때 감소한다. */
    }
  }

  if(result != 0) {
    /* [한국어] 실제로 패킷을 주입하는 경우 (result가 0이 아닌 경우). */
    _packet_seq_no[source]++;
    /* [한국어] 이 소스 노드의 배치 내 주입 완료 패킷 시퀀스 번호를 증가시킨다.
     * _packet_seq_no[source] == _batch_size가 되면 이 노드의 배치 주입이 완료된다.
     * _SingleSim()에서 모든 노드의 _packet_seq_no >= _batch_size이면 배치 주입 완료로 판단. */
  }

  return result;  // [한국어] 0: 주입 없음, 양수: 패킷 크기, -1: 응답 패킷 주입
}

/*
 * [한국어]
 * BatchTrafficManager::_ClearStats - 통계 초기화
 *
 * @return: (void)
 *
 * 부모 TrafficManager::_ClearStats()를 먼저 호출하여 공통 통계를 초기화하고,
 * 이후 _batch_time 통계 객체를 별도로 초기화한다.
 * 새 시뮬레이션 실행(sim_count 루프)이 시작될 때 호출된다.
 *
 * 실행 컨텍스트: 싱글스레드, 각 시뮬레이션 실행 전.
 * 호출 체인:
 *   TrafficManager::Run() → [_ClearStats()] → TrafficManager::_ClearStats()
 */
void BatchTrafficManager::_ClearStats( )
{
  TrafficManager::_ClearStats();    // [한국어] 부모의 지연/처리율 통계를 초기화한다
  _batch_time->Clear( );            // [한국어] 배치 소요 시간 통계를 초기화한다 — 히스토그램과 누적값 초기화
}

/*
 * [한국어]
 * BatchTrafficManager::_SingleSim - 단일 시뮬레이션 실행 (배치 루프)
 *
 * @return: true (항상 성공적으로 완료됨을 의미)
 *
 * BatchTrafficManager의 핵심 제어 루프이다.
 * batch_count번에 걸쳐 다음 절차를 반복한다:
 *   [1] 주입 단계: 모든 노드에서 _packet_seq_no[i] >= _batch_size가 될 때까지 _Step() 반복
 *   [2] 드레인 단계: 모든 인-플라이트 플릿이 수신될 때까지 _Step() 반복
 *   [3] 통계 기록: 배치 소요 시간(start_time ~ 마지막 플릿 수신)을 _batch_time에 추가
 * 배치마다 _packet_seq_no[]를 0으로 리셋하여 독립적인 배치를 보장한다.
 * 모든 배치 완료 후 _sim_state를 draining으로 설정하고 true를 반환한다.
 *
 * 실행 컨텍스트: TrafficManager::Run()에서 호출, 싱글스레드.
 * 호출 체인:
 *   TrafficManager::Run() → [_SingleSim()] → _Step() (반복)
 *                                           → UpdateStats() → DisplayStats()
 */
bool BatchTrafficManager::_SingleSim( )
{
  int batch_index = 0;
  /* [한국어] 현재 실행 중인 배치의 인덱스 (0부터 시작). */

  while(batch_index < _batch_count) {
    /* [한국어] batch_count번 배치 루프를 실행한다.
     * 각 배치는 독립적으로 주입→드레인→통계 기록을 수행한다. */

    _packet_seq_no.assign(_nodes, 0);
    /* [한국어] 모든 노드의 패킷 시퀀스 번호를 0으로 리셋한다.
     * 이 배치에서 각 노드가 주입할 패킷을 새로 카운팅하기 위해 초기화한다.
     * assign(_nodes, 0)은 벡터를 _nodes개의 0으로 채운다. */

    _last_id = -1;
    /* [한국어] 마지막 완료 플릿 ID를 초기화 — 이 배치에서 아직 수신된 플릿 없음을 표시. */

    _last_pid = -1;
    /* [한국어] 마지막 완료 패킷 ID를 초기화 — 이 배치에서 아직 수신된 패킷 없음을 표시. */

    _sim_state = running;
    /* [한국어] 시뮬레이션 상태를 running으로 설정한다.
     * _IssuePacket() 등 내부 로직이 시뮬레이션이 진행 중임을 인식한다. */

    int start_time = _time;
    /* [한국어] 이 배치 시작 시점의 사이클 카운터를 기록한다.
     * 배치 완료 후 소요 시간 = _time - start_time으로 계산한다. */

    bool batch_complete;
    /* [한국어] 이 배치의 주입 완료 여부 플래그. */

    cout << "Sending batch " << batch_index + 1 << " (" << _batch_size << " packets)..." << endl;
    /* [한국어] 배치 시작 알림 메시지 출력 (1-indexed: batch_index+1). */

    do {
      _Step();
      /* [한국어] 1 사이클 시뮬레이션을 진행한다.
       * 내부에서 _IssuePacket()을 통해 조건을 만족하는 노드에서 패킷을 주입한다.
       * 부모 TrafficManager::_Step()이 라우터 파이프라인, 플릿 이동, 완료 처리를 수행한다. */

      batch_complete = true;
      /* [한국어] 완료 여부를 true로 가정한 뒤, 미완료 노드가 발견되면 false로 바꾼다. */

      for(int i = 0; i < _nodes; ++i) {
        /* [한국어] 모든 노드를 순회하며 배치 할당량 완료 여부를 확인한다. */

	if(_packet_seq_no[i] < _batch_size) {
          /* [한국어] 이 노드가 아직 배치 할당량(_batch_size)을 채우지 않은 경우. */
	  batch_complete = false;  // [한국어] 한 노드라도 미완료이면 배치 주입이 완료되지 않음
	  break;                   // [한국어] 조기 종료 — 더 이상 확인할 필요 없음
	}
      }

      if(_sent_packets_out) {
        /* [한국어] sent_packets_out 파일이 설정된 경우 현재 사이클의 주입 현황을 기록한다. */
	*_sent_packets_out << _packet_seq_no << endl;
        /* [한국어] _packet_seq_no 벡터를 스트림에 출력한다.
         * 각 사이클의 노드별 주입 완료 패킷 수를 파일에 기록하여 외부 분석에 활용한다. */
      }
    } while(!batch_complete);
    /* [한국어] 모든 노드가 배치 할당량을 채울 때까지 _Step()을 반복한다. */

    cout << "Batch injected. Time used is " << _time - start_time << " cycles." << endl;
    /* [한국어] 배치 주입 완료 메시지와 주입에 소요된 사이클 수를 출력한다. */

    int sent_time = _time;
    /* [한국어] 배치 주입 완료 시점의 사이클을 기록한다.
     * 드레인 소요 시간 = 드레인 완료 시 _time - sent_time으로 계산한다. */

    cout << "Waiting for batch to complete..." << endl;
    /* [한국어] 드레인 단계 시작 알림 — 주입된 모든 패킷이 수신될 때까지 대기함을 알림. */

    int empty_steps = 0;
    /* [한국어] 드레인 단계에서 실행된 사이클 수 카운터.
     * 1000 사이클마다 _DisplayRemaining()으로 남은 패킷 수를 출력한다. */

    bool packets_left = false;
    /* [한국어] 네트워크에 아직 비행 중인 플릿이 있는지 여부. */

    for(int c = 0; c < _classes; ++c) {
      /* [한국어] 모든 트래픽 클래스를 순회하며 비행 중 플릿 존재 여부를 확인한다. */
      packets_left |= !_total_in_flight_flits[c].empty();
      /* [한국어] 하나의 클래스라도 비행 중 플릿이 있으면 packets_left=true.
       * _total_in_flight_flits[c]는 클래스 c에서 아직 수신되지 않은 플릿의 집합(맵)이다. */
    }

    while( packets_left ) {
      /* [한국어] 모든 인-플라이트 플릿이 목적지에 도달할 때까지 사이클을 계속 진행한다.
       * 이 드레인 단계가 배치 총 소요 시간의 후반부를 차지한다. */

      _Step( );
      /* [한국어] 1 사이클 진행 — 비행 중인 플릿들이 라우터를 통해 이동한다.
       * 이 단계에서는 새 패킷 주입이 없다 (모든 노드의 _packet_seq_no >= _batch_size). */

      ++empty_steps;
      /* [한국어] 드레인 사이클 카운터를 증가시킨다. */

      if ( empty_steps % 1000 == 0 ) {
        /* [한국어] 1000 사이클마다 남은 패킷 상태를 출력하여 진행 상황을 모니터링한다.
         * 데드락 감지에도 유용하다 — 패킷이 계속 남아 있으면 네트워크 교착 가능성을 의심. */
	_DisplayRemaining( );
        /* [한국어] 아직 비행 중인 플릿/패킷 정보를 stdout에 출력하는 부모 함수 호출. */
	cout << ".";
        /* [한국어] 진행 중임을 나타내는 점(.) 출력 — 긴 드레인 시간에 사용자 피드백 제공. */
      }

      packets_left = false;        // [한국어] 매 사이클마다 재확인을 위해 false로 리셋
      for(int c = 0; c < _classes; ++c) {
        /* [한국어] 모든 클래스의 비행 중 플릿 존재 여부를 다시 확인한다. */
	packets_left |= !_total_in_flight_flits[c].empty();
        /* [한국어] 하나라도 남아 있으면 packets_left=true → 드레인 루프 계속. */
      }
    }

    cout << endl;
    cout << "Batch received. Time used is " << _time - sent_time << " cycles." << endl
	 << "Last packet was " << _last_pid << ", last flit was " << _last_id << "." << endl;
    /* [한국어] 드레인 완료 메시지 출력:
     * - 드레인 소요 시간 = _time - sent_time (마지막 주입 후 마지막 수신까지)
     * - 마지막으로 수신된 패킷 ID와 플릿 ID (배치의 완료를 확인하는 진단 정보) */

    _batch_time->AddSample(_time - start_time);
    /* [한국어] 이 배치의 총 소요 시간(주입 시작 ~ 마지막 플릿 수신)을 통계에 추가한다.
     * start_time은 배치 시작 시점, _time은 드레인 완료 시점이다. */

    cout << _sim_state << endl;
    /* [한국어] 현재 시뮬레이션 상태를 출력한다 (running 상태가 출력됨). */

    UpdateStats();
    /* [한국어] 부모 TrafficManager::UpdateStats()를 호출하여 이번 배치의 통계를 갱신한다.
     * 지연, 처리율 등 TrafficManager 공통 통계를 업데이트한다. */

    DisplayStats();
    /* [한국어] 이번 배치의 통계를 출력한다 (최소/평균/최대 배치 시간 포함). */

    ++batch_index;
    /* [한국어] 다음 배치로 진행하기 위해 배치 인덱스를 증가시킨다. */
  }

  _sim_state = draining;
  /* [한국어] 모든 배치가 완료된 후 시뮬레이션 상태를 draining으로 전환한다.
   * 부모 TrafficManager::Run()이 이 상태를 확인하여 최종 통계 수집 여부를 결정한다. */

  _drain_time = _time;
  /* [한국어] 드레인 시작 시각을 기록한다 (배치 모드에서는 모든 배치 완료 후를 가리킴). */

  return 1;
  /* [한국어] 시뮬레이션이 성공적으로 완료되었음을 반환한다 (true로 해석됨). */
}

/*
 * [한국어]
 * BatchTrafficManager::_UpdateOverallStats - 전체 누적 통계 갱신
 *
 * @return: (void)
 *
 * 현재 시뮬레이션(sim_count 중 하나)의 배치 시간 통계를 전체 누적 변수에 반영한다.
 * 부모 TrafficManager::_UpdateOverallStats()를 먼저 호출하여 공통 통계(지연, 처리율)를 누적한 뒤,
 * _batch_time의 Min/Average/Max를 각각의 누적 변수에 더한다.
 * DisplayOverallStats()에서 이 누적 값을 _total_sims로 나누어 전체 평균을 구한다.
 *
 * 실행 컨텍스트: TrafficManager::Run()에서 각 시뮬레이션 실행 완료 후 호출, 싱글스레드.
 * 호출 체인:
 *   TrafficManager::Run() → [_UpdateOverallStats()] → TrafficManager::_UpdateOverallStats()
 */
void BatchTrafficManager::_UpdateOverallStats() {
  TrafficManager::_UpdateOverallStats();
  /* [한국어] 부모의 지연/처리율 누적 통계를 먼저 갱신한다. */

  _overall_min_batch_time += _batch_time->Min();
  /* [한국어] 현재 시뮬레이션의 최소 배치 시간을 전체 누적 최소에 더한다.
   * 최종 출력 시 _total_sims로 나누어 평균화한다. */

  _overall_avg_batch_time += _batch_time->Average();
  /* [한국어] 현재 시뮬레이션의 평균 배치 시간을 전체 누적 평균에 더한다. */

  _overall_max_batch_time += _batch_time->Max();
  /* [한국어] 현재 시뮬레이션의 최대 배치 시간을 전체 누적 최대에 더한다. */
}

/*
 * [한국어]
 * BatchTrafficManager::_OverallStatsCSV - 전체 통계를 CSV 문자열로 반환
 *
 * @param c: 트래픽 클래스 번호 (기본값 0). 부모 CSV에 전달된다.
 * @return: "부모CSV,min_배치시간,avg_배치시간,max_배치시간" 형식의 CSV 문자열.
 *          각 값은 _total_sims로 나눈 시뮬레이션 간 평균이다.
 *
 * 부모 TrafficManager::_OverallStatsCSV()가 생성한 CSV 문자열 뒤에
 * min/avg/max 배치 시간을 쉼표로 이어 붙여 확장된 CSV를 반환한다.
 * print_csv_results=1 설정 시 결과를 스크립트로 파싱할 때 사용된다.
 *
 * 실행 컨텍스트: 시뮬레이션 완료 후 출력 단계, 싱글스레드.
 * 호출 체인:
 *   TrafficManager::Run() → _OverallStatsCSV() ← [이 함수]
 *                                               → TrafficManager::_OverallStatsCSV()
 */
string BatchTrafficManager::_OverallStatsCSV(int c) const
{
  ostringstream os;
  /* [한국어] 문자열 스트림을 생성하여 CSV를 조립한다. */

  os << TrafficManager::_OverallStatsCSV(c) << ','
     << _overall_min_batch_time / (double)_total_sims << ','
     << _overall_avg_batch_time / (double)_total_sims << ','
     << _overall_max_batch_time / (double)_total_sims;
  /* [한국어] 부모 CSV 뒤에 min/avg/max 배치 시간의 시뮬레이션 간 평균을 추가한다.
   * (double)_total_sims로 나누어 여러 번 실행의 평균값을 계산한다.
   * 형식: "<부모CSV>,<min>,<avg>,<max>" */

  return os.str();
  /* [한국어] 완성된 CSV 문자열을 반환한다. */
}

/*
 * [한국어]
 * BatchTrafficManager::WriteStats - 통계를 출력 스트림에 기록
 *
 * @param os: 통계를 기록할 출력 스트림 (기본값: cout).
 * @return: (void)
 *
 * 부모 TrafficManager::WriteStats()를 호출하여 공통 통계를 기록한 뒤,
 * 배치 평균 소요 시간("batch_time = <avg>;")을 추가로 기록한다.
 * stats_out 파일이나 stdout에 최종 통계를 저장할 때 사용된다.
 *
 * 실행 컨텍스트: 시뮬레이션 완료 후 통계 저장 단계, 싱글스레드.
 * 호출 체인:
 *   TrafficManager::Run() → [WriteStats(os)] → TrafficManager::WriteStats(os)
 */
void BatchTrafficManager::WriteStats(ostream & os) const
{
  TrafficManager::WriteStats(os);
  /* [한국어] 부모의 지연/처리율 통계를 스트림에 먼저 기록한다. */

  os << "batch_time = " << _batch_time->Average() << ";" << endl;
  /* [한국어] 배치 평균 소요 시간을 "batch_time = <값>;" 형식으로 기록한다.
   * 세미콜론(;)은 BookSim 통계 파일 형식의 구분자이다. */
}

/*
 * [한국어]
 * BatchTrafficManager::DisplayStats - 현재 시뮬레이션 배치 통계 출력
 *
 * @param os: 출력 스트림 (기본값: cout).
 * @return: (void)
 *
 * 현재 시뮬레이션(단일 실행)에서 측정된 배치 소요 시간의 최소/평균/최대를 출력한다.
 * 부모 TrafficManager::DisplayStats()를 먼저 호출하여 지연/처리율 통계도 함께 출력한다.
 * _SingleSim()에서 각 배치 완료 후 호출된다.
 *
 * 실행 컨텍스트: _SingleSim() 내 각 배치 완료 후 호출, 싱글스레드.
 * 호출 체인:
 *   _SingleSim() → [DisplayStats(os)] → TrafficManager::DisplayStats()
 */
void BatchTrafficManager::DisplayStats(ostream & os) const {
  TrafficManager::DisplayStats();
  /* [한국어] 부모의 지연/처리율 통계를 먼저 출력한다.
   * 주의: 부모 호출에 os 인수를 전달하지 않고 기본값(cout)을 사용한다.
   * 이는 의도적인 것일 수 있으나, os가 파일 스트림인 경우 불일치가 발생할 수 있다. */

  os << "Minimum batch duration = " << _batch_time->Min() << endl;
  /* [한국어] 이번 시뮬레이션에서 가장 짧게 소요된 배치 시간(사이클)을 출력한다. */

  os << "Average batch duration = " << _batch_time->Average() << endl;
  /* [한국어] 이번 시뮬레이션에서 모든 배치의 평균 소요 시간을 출력한다. */

  os << "Maximum batch duration = " << _batch_time->Max() << endl;
  /* [한국어] 이번 시뮬레이션에서 가장 오래 소요된 배치 시간(사이클)을 출력한다. */
}

/*
 * [한국어]
 * BatchTrafficManager::DisplayOverallStats - 전체 누적 배치 통계 출력
 *
 * @param os: 출력 스트림 (기본값: cout).
 * @return: (void)
 *
 * sim_count회의 독립적인 시뮬레이션 실행에 걸쳐 누적된 배치 시간 통계를 출력한다.
 * 누적값을 _total_sims로 나누어 시뮬레이션 간 평균을 계산한다.
 * 부모 TrafficManager::DisplayOverallStats()를 먼저 호출하여 공통 전체 통계도 출력한다.
 *
 * 주의: 코드에 "Overall min/avg/max" 레이블이 모두 "Overall min"으로 잘못 표시된 버그가 있다.
 * 실제 값은 _overall_min / _overall_avg / _overall_max 순서로 올바르게 출력된다.
 *
 * 실행 컨텍스트: 모든 시뮬레이션 완료 후 최종 결과 출력 단계, 싱글스레드.
 * 호출 체인:
 *   TrafficManager::Run() → [DisplayOverallStats(os)] → TrafficManager::DisplayOverallStats(os)
 */
void BatchTrafficManager::DisplayOverallStats(ostream & os) const {
  TrafficManager::DisplayOverallStats(os);
  /* [한국어] 부모의 전체 누적 지연/처리율 통계를 먼저 출력한다. */

  os << "Overall min batch duration = " << _overall_min_batch_time / (double)_total_sims
     << " (" << _total_sims << " samples)" << endl
     << "Overall min batch duration = " << _overall_avg_batch_time / (double)_total_sims
     << " (" << _total_sims << " samples)" << endl
     << "Overall min batch duration = " << _overall_max_batch_time / (double)_total_sims
     << " (" << _total_sims << " samples)" << endl;
  /* [한국어] 전체 시뮬레이션에 걸친 배치 시간의 누적 min/avg/max를 _total_sims로 나누어 출력한다.
   * 주의: 원본 코드의 버그 — 세 줄 모두 "Overall min batch duration"이라고 출력된다.
   *       실제로는 첫 번째가 min, 두 번째가 avg, 세 번째가 max를 출력하는 것이 의도이다.
   *       레이블 문자열은 의도적으로 수정하지 않고 원본 그대로 유지한다. */
}
