// $Id: batchtrafficmanager.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] BookSim 배치 트래픽 매니저 헤더 (batchtrafficmanager.hpp)
 *
 * === 파일의 역할 ===
 * TrafficManager를 상속하여 "배치(batch)" 기반 합성 트래픽 시뮬레이션을 구현하는
 * BatchTrafficManager 클래스를 선언한다.
 * 일반 TrafficManager가 지속적(steady-state) 트래픽을 주입하는 데 반해,
 * BatchTrafficManager는 정해진 수(batch_size)의 패킷을 한 배치로 묶어 주입한 뒤
 * 모든 패킷이 목적지에 도달할 때까지 대기하는 방식으로 동작한다.
 * 배치가 완료되면 소요 시간을 측정하고, 이를 batch_count회 반복한다.
 * GPU 메모리 서브시스템의 왕복 지연(RTT) 연구 등에서 유용하다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 컨텍스트: 합성 트래픽 시뮬레이션 전용 (GPGPU-Sim 실제 GPU 시뮬레이션과 별개).
 * 호출 체인:
 *   BookSim main() / trafficmanager_factory()
 *     → BatchTrafficManager::BatchTrafficManager() (설정 파라미터 로드)
 *     → BatchTrafficManager::_SingleSim() (배치 루프 실행)
 *         → _Step() (1 사이클 진행, 부모 TrafficManager)
 *         → _IssuePacket() (패킷 주입 결정)
 *         → _RetireFlit() (플릿 완료 처리)
 *     → DisplayStats() / DisplayOverallStats() (결과 출력)
 * sim_type이 "batch"로 설정된 경우에만 이 클래스가 인스턴스화된다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존: trafficmanager.hpp (TrafficManager 기반 클래스), stats.hpp (Stats 통계 객체),
 *         config_utils.hpp (Configuration 파라미터 읽기)
 * - 피의존: BookSim 팩토리 함수 (trafficmanager.cc의 TrafficManager::New())
 * - 부모 클래스 상속 멤버 활용:
 *     _nodes: 네트워크 노드 수
 *     _time: 현재 시뮬레이션 사이클
 *     _classes: 트래픽 클래스 수
 *     _packet_seq_no[]: 노드별 주입 패킷 시퀀스 번호
 *     _requestsOutstanding[]: 노드별 미완료 요청 수
 *     _repliesPending[]: 노드별 대기 중인 응답 큐
 *     _total_in_flight_flits[]: 클래스별 비행 중 플릿 맵
 *     _sim_state: 시뮬레이션 상태 (running / draining)
 *     _total_sims: 누적 시뮬레이션 실행 횟수
 * - 데이터 흐름: BatchTrafficManager는 _packet_seq_no로 주입량을 추적하고,
 *   _total_in_flight_flits로 드레인 완료를 판단한다.
 *
 * === 주요 함수/구조체 요약 ===
 * BatchTrafficManager() — 생성자: batch_size, batch_count, max_outstanding 등 로드,
 *                         _batch_time Stats 객체 초기화
 * _SingleSim()         — 핵심: batch_count번의 배치 루프 실행, 각 배치의 주입→드레인 제어
 * _IssuePacket()       — 노드별 패킷 주입 결정: seq_no < batch_size 이면 주입 허용
 * _RetireFlit()        — 플릿 완료 시 _last_id/_last_pid 갱신 후 부모 호출
 * _UpdateOverallStats()— 배치 시간(min/avg/max)을 누적 통계에 반영
 * DisplayStats()       — 현재 배치의 최소/평균/최대 소요 시간 출력
 */

#ifndef _BATCHTRAFFICMANAGER_HPP_
#define _BATCHTRAFFICMANAGER_HPP_

#include <iostream>         // [한국어] ostream — WriteStats/DisplayStats 출력 스트림

#include "config_utils.hpp" // [한국어] Configuration — BookSimConfig 파라미터 읽기
#include "stats.hpp"        // [한국어] Stats — 배치 소요 시간 분포 통계 객체
#include "trafficmanager.hpp" // [한국어] TrafficManager — 상속 기반 클래스 (Step, Retire 등)

/*
 * [한국어]
 * BatchTrafficManager - 배치(batch) 기반 합성 트래픽 시뮬레이션 관리자
 *
 * TrafficManager를 상속하며, 지속적(steady-state) 주입 대신
 * "N개 패킷 주입 → 전부 수신 확인 → 반복" 방식으로 동작한다.
 * 각 배치의 소요 시간(주입 시작~마지막 패킷 수신)을 Stats로 측정한다.
 *
 * 배치 완료 조건:
 *   주입 조건: 모든 노드에서 _packet_seq_no[i] >= batch_size
 *   드레인 조건: 모든 클래스에서 _total_in_flight_flits[c].empty()
 *
 * 실행 컨텍스트: 합성 트래픽 시뮬레이션, 싱글스레드.
 */
class BatchTrafficManager : public TrafficManager {

protected:

  int _max_outstanding;
  /* [한국어] 노드당 허용되는 최대 미완료(outstanding) 요청 수.
   * 설정자: 생성자에서 config.GetInt("max_outstanding_requests")로 읽음.
   * 읽는 자: _IssuePacket()에서 _requestsOutstanding[source] < _max_outstanding 조건 검사.
   * 값 범위: 0이면 제한 없음 (무제한 주입), 양수이면 해당 수만큼 동시 요청 제한.
   * 동기화: 싱글스레드 시뮬레이터이므로 별도 락 불필요. */

  int _batch_size;
  /* [한국어] 한 배치에서 각 노드가 주입해야 할 패킷 수 목표값.
   * 설정자: 생성자에서 config.GetInt("batch_size")로 읽음.
   * 읽는 자: _IssuePacket()에서 _packet_seq_no[source] < _batch_size 조건 검사.
   *          _SingleSim()에서 배치 완료 판정 시 동일 조건 사용.
   * 값 범위: 양의 정수 (보통 1000 이상). 기본값 1000.
   * 동기화: 불변값 (시뮬레이션 도중 변경 없음). */

  int _batch_count;
  /* [한국어] 수행할 배치 반복 횟수.
   * 설정자: 생성자에서 config.GetInt("batch_count")로 읽음.
   * 읽는 자: _SingleSim()의 while(batch_index < _batch_count) 루프 조건.
   * 값 범위: 양의 정수. 기본값 1.
   * 동기화: 불변값. */

  int _last_id;
  /* [한국어] 가장 최근에 완료(retired)된 플릿의 전역 ID.
   * 설정자: _RetireFlit()에서 f->id로 갱신됨.
   *          _SingleSim() 시작 시 -1로 초기화됨.
   * 읽는 자: _SingleSim()에서 배치 완료 후 진단 메시지 출력에 사용.
   * 값 범위: -1 (초기 또는 미수신) 또는 유효한 플릿 ID (0 이상).
   * 동기화: 싱글스레드, 락 불필요. */

  int _last_pid;
  /* [한국어] 가장 최근에 완료된 패킷의 ID (Packet ID, pid).
   * 설정자: _RetireFlit()에서 f->pid로 갱신됨.
   *          _SingleSim() 시작 시 -1로 초기화됨.
   * 읽는 자: _SingleSim()에서 배치 완료 후 "Last packet was <_last_pid>" 메시지 출력.
   * 값 범위: -1 (초기) 또는 유효한 패킷 ID (0 이상).
   * 동기화: 싱글스레드, 락 불필요. */

  Stats * _batch_time;
  /* [한국어] 배치별 소요 시간(사이클 수) 통계 객체.
   * 설정자: 생성자에서 new Stats(this, "batch_time", 1.0, 1000)으로 생성.
   *          _SingleSim()에서 각 배치 완료 후 AddSample(_time - start_time) 호출.
   * 읽는 자: DisplayStats()에서 Min/Average/Max 출력.
   *          WriteStats()에서 Average 기록.
   *          _UpdateOverallStats()에서 누적 통계에 반영.
   *          _ClearStats()에서 Clear() 초기화.
   * 값 범위: 배치 시작 사이클부터 마지막 플릿 수신 사이클까지의 차이.
   * 동기화: 싱글스레드, 락 불필요. 소멸자에서 delete됨. */

  double _overall_min_batch_time;
  /* [한국어] 전체 시뮬레이션(sim_count회 반복) 누적 최소 배치 시간.
   * 설정자: _UpdateOverallStats()에서 _batch_time->Min()을 누적 덧셈.
   * 읽는 자: DisplayOverallStats()에서 /(double)_total_sims로 평균화하여 출력.
   * 값 범위: 0.0 이상 (사이클 단위). 초기값 0.
   * 동기화: 싱글스레드. */

  double _overall_avg_batch_time;
  /* [한국어] 전체 시뮬레이션 누적 평균 배치 시간.
   * 설정자: _UpdateOverallStats()에서 _batch_time->Average()를 누적 덧셈.
   * 읽는 자: DisplayOverallStats()에서 평균화하여 출력.
   * 값 범위: 0.0 이상. 초기값 0.
   * 동기화: 싱글스레드. */

  double _overall_max_batch_time;
  /* [한국어] 전체 시뮬레이션 누적 최대 배치 시간.
   * 설정자: _UpdateOverallStats()에서 _batch_time->Max()를 누적 덧셈.
   * 읽는 자: DisplayOverallStats()에서 평균화하여 출력.
   * 값 범위: 0.0 이상. 초기값 0.
   * 동기화: 싱글스레드. */

  ostream * _sent_packets_out;
  /* [한국어] 배치 진행 중 각 사이클의 노드별 전송 패킷 시퀀스 번호를 기록하는 출력 스트림.
   * 설정자: 생성자에서 config.GetStr("sent_packets_out")가 비어 있으면 NULL,
   *          아니면 new ofstream(파일경로)으로 생성.
   * 읽는 자: _SingleSim()의 do-while 루프에서 매 사이클 *_sent_packets_out << _packet_seq_no.
   * 값 범위: NULL (비활성) 또는 유효한 ofstream 포인터.
   * 동기화: 싱글스레드. 소멸자에서 NULL이 아닐 때 delete됨. */

  virtual void _RetireFlit( Flit *f, int dest );
  /* [한국어] 목적지에 도달한 플릿을 처리(retire)하는 가상 함수.
   * 부모 TrafficManager::_RetireFlit()을 호출하기 전에 _last_id와 _last_pid를 갱신한다.
   * 이를 통해 배치의 마지막 플릿 정보를 추적한다. */

  virtual int _IssuePacket( int source, int cl );
  /* [한국어] 특정 소스 노드가 현재 사이클에 패킷을 주입할지 결정하는 가상 함수.
   * batch_size와 max_outstanding 조건을 기반으로 주입 여부와 패킷 크기를 반환한다. */

  virtual void _ClearStats( );
  /* [한국어] 통계 초기화 가상 함수. 부모 _ClearStats() 호출 후 _batch_time->Clear() 수행. */

  virtual bool _SingleSim( );
  /* [한국어] 단일 시뮬레이션 실행(batch_count번 배치 루프) 가상 함수.
   * 핵심 제어 루프: 주입 → 드레인 → 통계 갱신을 배치마다 반복한다. */

  virtual void _UpdateOverallStats( );
  /* [한국어] 개별 시뮬레이션 결과를 전체 누적 통계에 반영하는 가상 함수. */

  virtual string _OverallStatsCSV(int c = 0) const;
  /* [한국어] 전체 누적 통계를 CSV 문자열로 반환하는 가상 함수.
   * 부모의 CSV에 min/avg/max 배치 시간을 ','로 이어 붙여 반환한다. */

public:

  BatchTrafficManager( const Configuration &config, const vector<Network *> & net );
  /* [한국어] 생성자 — BatchTrafficManager를 초기화한다.
   * config에서 batch_size, batch_count, max_outstanding_requests를 읽고,
   * _batch_time Stats 객체를 생성하며, sent_packets_out 파일 스트림을 설정한다. */

  virtual ~BatchTrafficManager( );
  /* [한국어] 소멸자 — _batch_time과 _sent_packets_out을 해제한다. */

  virtual void WriteStats( ostream & os = cout ) const;
  /* [한국어] 현재 시뮬레이션의 배치 통계를 스트림에 기록한다.
   * 부모 WriteStats() 호출 후 "batch_time = <avg>" 행을 추가로 출력한다. */

  virtual void DisplayStats( ostream & os = cout ) const;
  /* [한국어] 현재 시뮬레이션의 배치 통계를 콘솔에 출력한다.
   * 최소/평균/최대 배치 소요 시간을 사람이 읽기 쉬운 형식으로 출력한다. */

  virtual void DisplayOverallStats( ostream & os = cout ) const;
  /* [한국어] 전체 시뮬레이션(sim_count회 누적) 배치 통계를 출력한다.
   * 부모 DisplayOverallStats() 호출 후 배치 시간 누적 min/avg/max를 출력한다. */

};

#endif
