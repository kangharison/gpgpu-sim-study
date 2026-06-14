// $Id: vc.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] 가상 채널(Virtual Channel) 헤더 (vc.hpp)
 *
 * === 파일의 역할 ===
 * VC(Virtual Channel)은 물리 포트를 시간 다중화하여 여러 독립적인 논리 채널을
 * 제공하는 NoC의 핵심 메커니즘이다. 각 VC는 독립적인 Flit 버퍼(deque)와
 * 상태 기계(idle → routing → vc_alloc → active)를 가진다.
 * HEAD-OF-LINE(HOL) 블로킹을 방지하여 NoC 처리량을 향상시키고,
 * 요청/응답 패킷을 별도 VC에 배정하여 데드락을 방지한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   Router 파이프라인 → VC::AddFlit() (패킷 버퍼 삽입)
 *   Router 라우팅 단계 → VC::Route() (라우팅 함수 호출)
 *   Router VC 할당 단계 → VC::SetOutput() (출구 포트/VC 설정)
 *   Router 전송 단계 → VC::RemoveFlit() (버퍼에서 flit 추출)
 *
 * 실행 컨텍스트: GPUTrafficManager::_Step() 내 _net[subnet]->Evaluate()에서
 * 라우터의 각 파이프라인 단계가 VC 메서드를 호출한다. 단일 스레드.
 *
 * === 타 모듈과의 연결 ===
 * - Router (router.hpp): 각 입구 포트마다 VC 배열을 보유
 * - Flit (flit.hpp): VC 버퍼에 저장되는 데이터 단위
 * - OutputSet (outputset.hpp): 라우팅 결과 출구 포트 집합 (_route_set)
 * - routefunc.hpp: Route()에서 호출하는 라우팅 함수 포인터 타입
 *
 * === 주요 함수/구조체 요약 ===
 * - AddFlit(): 버퍼에 flit 삽입 및 우선순위 업데이트, 패킷 연속성 검증
 * - RemoveFlit(): 버퍼 front에서 flit 추출
 * - Route(): 라우팅 함수를 호출하여 _route_set 계산
 * - SetOutput(): VC 할당기가 결정한 출구 포트/VC 저장
 * - UpdatePriority(): VC의 스케줄링 우선순위 갱신
 * - eVCState: idle→routing→vc_alloc→active 상태 기계
 */

#ifndef _VC_HPP_
#define _VC_HPP_

#include <deque>  // [한국어] deque<Flit*>: VC 버퍼 (양방향 큐, front pop이 잦음)

#include "flit.hpp"         // [한국어] Flit 타입 (버퍼 원소)
#include "outputset.hpp"    // [한국어] OutputSet: 라우팅 결과 출구 포트 집합
#include "routefunc.hpp"    // [한국어] tRoutingFunction: 라우팅 함수 포인터 타입
#include "config_utils.hpp" // [한국어] Configuration: 설정 파일 파싱 객체

/*
 * [한국어]
 * VC - 라우터 내 가상 채널 (Virtual Channel)
 *
 * 물리 입력 포트당 여러 VC를 두어 HOL 블로킹을 방지한다.
 * 각 VC는 독립적인 Flit 버퍼와 상태를 가지며, 서로 다른 패킷이
 * 같은 물리 포트를 통해 독립적으로 진행될 수 있다.
 *
 * 상태 기계:
 *   idle: VC 비어있음, 새 패킷 수락 가능
 *   routing: HEAD flit 수신 후 라우팅 계산 중
 *   vc_alloc: 출구 VC 할당 대기 중
 *   active: 출구 포트/VC 확정, 데이터 전송 중 (tail flit 전송 후 idle로 복귀)
 */
class VC : public Module {
public:
  /*
   * [한국어]
   * eVCState - VC 상태 기계의 상태 열거형
   *
   * idle: HEAD flit 미도착, 이 VC가 비어있거나 새 패킷 대기 중
   * routing: HEAD flit 도착 후 라우팅 함수로 출구 포트 계산 중
   * vc_alloc: 출구 포트 결정 후 해당 포트의 다운스트림 VC 할당 대기 중
   * active: 출구 포트와 다운스트림 VC가 확정되어 flit 전송 진행 중
   *         tail flit 전송 완료 시 idle로 복귀
   */
  enum eVCState { state_min = 0, idle = state_min, routing, vc_alloc, active,
		  state_max = active };

  /*
   * [한국어]
   * state_info_t - VC 상태별 사이클 수를 기록하는 구조체 (디버그/통계용)
   */
  struct state_info_t {
    int cycles; // [한국어] 해당 상태에 머문 사이클 수
  };

  // [한국어] 상태 이름 문자열 배열 (인덱스 = eVCState 값)
  // 예: VCSTATE[idle] = "idle", VCSTATE[routing] = "routing"
  static const char * const VCSTATE[];

private:

  deque<Flit *> _buffer;
  /* [한국어] 이 VC의 Flit 저장 버퍼 (선입선출 deque).
   * 설정자: AddFlit()이 back에 push.
   * 읽는 자: FrontFlit()이 front 참조, RemoveFlit()이 front pop.
   * 값 범위: NULL 항목 없음; 크기 <= 설정된 VC 버퍼 크기.
   * 동기화: 단일 스레드. */

  eVCState _state;
  /* [한국어] 이 VC의 현재 상태 (idle/routing/vc_alloc/active 중 하나).
   * 설정자: SetState()가 상태 전이 처리.
   * 읽는 자: 라우터 파이프라인이 각 단계에서 상태에 따라 동작 결정.
   * 값 범위: eVCState 열거형 값.
   * 동기화: 단일 스레드. */

  OutputSet *_route_set;
  /* [한국어] 라우팅 계산 결과: 가능한 출구 포트 집합.
   * 설정자: Route()가 라우팅 함수 호출 결과로 채움.
   *         lookahead 라우팅 시 NULL (la_route_set을 대신 사용).
   * 읽는 자: VC 할당기가 출구 포트/VC 결정에 사용.
   * 값 범위: OutputSet 포인터 또는 NULL(lookahead 모드).
   * 동기화: 단일 스레드. */

  int _out_port, _out_vc;
  /* [한국어] VC 할당 완료 후 결정된 출구 포트(_out_port)와 다운스트림 VC(_out_vc).
   * 설정자: SetOutput()이 VC 할당기 결과로 설정. 초기값: -1(미확정).
   * 읽는 자: 라우터 전송 단계에서 GetOutputPort()/GetOutputVC()로 조회.
   * 값 범위: _out_port: 0 .. num_outputs-1; _out_vc: 0 .. num_vcs-1; 미확정 시 -1.
   * 동기화: 단일 스레드. */

  /*
   * [한국어]
   * ePrioType - VC 내 Flit 우선순위 결정 방식 열거형
   *
   * local_age_based: 각 flit이 VC에 도착한 시간 기준 (오래된 flit 우선)
   * queue_length_based: VC 버퍼 길이 기준 (긴 큐 우선, 공정성)
   * hop_count_based: flit의 누적 홉 수 기준 (많이 이동한 flit 우선)
   * none: 우선순위 없음 (모두 동등)
   * other: 외부에서 설정된 우선순위 사용 (pri 필드 직접 참조)
   */
  enum ePrioType { local_age_based, queue_length_based, hop_count_based, none, other };

  ePrioType _pri_type;
  /* [한국어] 이 VC의 우선순위 결정 방식.
   * 설정자: VC 생성자가 설정 파일의 "priority" 문자열로 선택.
   * 읽는 자: UpdatePriority()가 _pri 값 계산 방식 결정.
   * 값 범위: ePrioType 열거형.
   * 동기화: 초기화 후 읽기 전용. */

  int _pri;
  /* [한국어] 이 VC의 현재 스케줄링 우선순위 값.
   * 설정자: UpdatePriority()가 _pri_type에 따라 재계산.
   * 읽는 자: 라우터 VC 할당기가 여러 VC 중 높은 우선순위 선택.
   * 값 범위: 0 이상 정수. 높을수록 우선순위 높음.
   * 동기화: 단일 스레드. */

  int _priority_donation;
  /* [한국어] 우선순위 기증(donation) 활성화 여부 (0=비활성, 1=활성).
   * 설정자: VC 생성자가 설정 파일의 "vc_priority_donation" 읽음.
   * 읽는 자: UpdatePriority()에서 버퍼 내 최고 우선순위 flit이
   *          front flit에게 우선순위를 "기증"하여 head-of-line 패킷 우선 처리.
   * 동기화: 초기화 후 읽기 전용. */

  bool _watched;
  /* [한국어] 이 VC를 상세 추적할지 여부.
   * 설정자: SetWatch()로 외부에서 설정.
   * 읽는 자: IsWatched()를 통해 조회.
   * 동기화: 단일 스레드. */

  int _expected_pid;
  /* [한국어] 현재 수신 중인 패킷의 예상 pid (패킷 연속성 검증).
   * 설정자: AddFlit()에서 non-tail flit 도착 시 f->pid로 설정; tail 도착 시 -1로 리셋.
   * 읽는 자: AddFlit()에서 다음 flit의 pid가 일치하는지 검증.
   * 값 범위: 유효한 pid 또는 -1(패킷 간 경계).
   * 목적: 같은 VC에 서로 다른 패킷의 flit이 섞이지 않았음을 보장. */

  int _last_id;
  /* [한국어] 이 VC에서 마지막으로 제거된 Flit의 id.
   * 설정자: RemoveFlit()에서 f->id로 업데이트.
   * 읽는 자: 디버그/추적 목적.
   * 동기화: 단일 스레드. */

  int _last_pid;
  /* [한국어] 이 VC에서 마지막으로 제거된 Flit의 패킷 id.
   * 설정자: RemoveFlit()에서 f->pid로 업데이트.
   * 읽는 자: 디버그/추적 목적.
   * 동기화: 단일 스레드. */

  bool _lookahead_routing;
  /* [한국어] Lookahead 라우팅 활성화 여부.
   * 설정자: VC 생성자가 설정 파일의 "routing_delay"가 0이면 true.
   * 읽는 자: 생성자에서 _route_set 할당 여부 결정
   *         (lookahead이면 _route_set = NULL, Flit::la_route_set 사용).
   * 동기화: 초기화 후 읽기 전용. */

public:

  /*
   * [한국어]
   * VC() - 가상 채널 초기화
   *
   * @config: BookSim 설정 파일 파싱 결과 (priority, vc_priority_donation, routing_delay)
   * @outputs: 이 VC가 있는 라우터의 출구 포트 수 (_route_set OutputSet 크기)
   * @parent: 부모 Module (라우터)
   * @name: VC 이름 문자열
   *
   * 초기 상태: idle, _out_port=-1, _out_vc=-1, _pri=0, _watched=false
   *
   * 호출 체인: Router 생성자 → (각 입구 포트마다) → [이 함수]
   */
  VC( const Configuration& config, int outputs,
      Module *parent, const string& name );

  /*
   * [한국어]
   * ~VC() - 소멸자
   *
   * lookahead 라우팅이 아닌 경우 _route_set을 delete.
   * lookahead 모드에서는 _route_set이 NULL이므로 delete 불필요.
   *
   * 호출 체인: Router 소멸 → [이 함수]
   */
  ~VC();

  /*
   * [한국어]
   * AddFlit() - VC 버퍼에 Flit을 추가한다
   *
   * @f: 추가할 Flit 포인터 (NULL 불가)
   *
   * 동작 순서:
   *   1. 패킷 연속성 검증 (_expected_pid)
   *   2. 우선순위 업데이트 (local_age_based, hop_count_based 처리)
   *   3. _buffer.push_back(f)으로 버퍼 뒤에 추가
   *   4. UpdatePriority() 호출
   *
   * 호출 체인: Router 파이프라인(입력 단계) → [이 함수]
   */
  void AddFlit( Flit *f );

  /*
   * [한국어]
   * FrontFlit() - 버퍼 맨 앞 Flit을 pop 없이 반환한다 (peek)
   *
   * @return: _buffer.front() 또는 NULL(버퍼 비어있을 때)
   *
   * 라우터 파이프라인에서 현재 처리 중인 flit을 확인할 때 사용.
   *
   * 호출 체인: Router 파이프라인 → [이 함수]
   */
  inline Flit *FrontFlit( ) const
  {
    return _buffer.empty() ? NULL : _buffer.front();
  }

  /*
   * [한국어]
   * RemoveFlit() - 버퍼 맨 앞 Flit을 꺼낸다
   *
   * @return: 버퍼 front의 Flit 포인터
   *
   * 버퍼가 비어있으면 Error() 호출. _last_id/_last_pid 업데이트 후 UpdatePriority().
   *
   * 호출 체인: 라우터 전송 단계 → [이 함수]
   */
  Flit *RemoveFlit( );


  /*
   * [한국어]
   * Empty() - VC 버퍼가 비어있는지 반환한다
   *
   * @return: true = 버퍼에 flit 없음
   *
   * 호출 체인: 라우터 파이프라인 → [이 함수]
   */
  inline bool Empty( ) const
  {
    return _buffer.empty( );
  }

  /*
   * [한국어]
   * GetState() - VC 현재 상태를 반환한다
   *
   * @return: eVCState 열거형 값 (idle/routing/vc_alloc/active)
   *
   * 호출 체인: 라우터 파이프라인, 버퍼 상태 관리 → [이 함수]
   */
  inline VC::eVCState GetState( ) const
  {
    return _state;
  }


  /*
   * [한국어]
   * SetState() - VC 상태를 전이시킨다
   *
   * @s: 새로운 상태
   *
   * watch 플래그가 있는 flit이 있으면 상태 전이 메시지를 gWatchOut에 출력.
   *
   * 호출 체인: 라우터 파이프라인 각 단계 → [이 함수]
   */
  void SetState( eVCState s );

  /*
   * [한국어]
   * GetRouteSet() - 라우팅 계산 결과 출구 포트 집합을 반환한다
   *
   * @return: _route_set 포인터 (lookahead 모드에서는 NULL)
   *
   * 호출 체인: 라우터 VC 할당기 → [이 함수]
   */
  const OutputSet *GetRouteSet( ) const;

  /*
   * [한국어]
   * SetRouteSet() - 라우팅 결과를 외부에서 설정한다 (lookahead 라우팅용)
   *
   * @output_set: 새로운 라우팅 결과 집합
   *
   * _out_port, _out_vc를 -1로 리셋하여 이전 할당 정보를 무효화.
   *
   * 호출 체인: 라우터 lookahead 처리 → [이 함수]
   */
  void SetRouteSet( OutputSet * output_set );

  /*
   * [한국어]
   * SetOutput() - VC 할당기가 결정한 출구 포트/VC를 저장한다
   *
   * @port: 결정된 출구 포트 번호
   * @vc: 결정된 다운스트림 VC 번호
   *
   * 호출 체인: 라우터 VC 할당기 → [이 함수]
   */
  void SetOutput( int port, int vc );

  /*
   * [한국어]
   * GetOutputPort() - 확정된 출구 포트 번호를 반환한다
   *
   * @return: _out_port (-1이면 미확정)
   *
   * 호출 체인: 라우터 전송 단계 → [이 함수]
   */
  inline int GetOutputPort( ) const
  {
    return _out_port;
  }


  /*
   * [한국어]
   * GetOutputVC() - 확정된 다운스트림 VC 번호를 반환한다
   *
   * @return: _out_vc (-1이면 미확정)
   *
   * 호출 체인: 라우터 전송 단계 → [이 함수]
   */
  inline int GetOutputVC( ) const
  {
    return _out_vc;
  }

  /*
   * [한국어]
   * UpdatePriority() - VC의 스케줄링 우선순위(_pri)를 재계산한다
   *
   * _pri_type에 따라:
   *   queue_length_based: _pri = _buffer.size()
   *   local_age_based: (AddFlit에서 직접 처리)
   *   hop_count_based: (AddFlit에서 직접 처리)
   *   other: priority_donation이 활성화면 버퍼 내 최고 pri flit의 값을 front에 기증
   *
   * 호출 체인: AddFlit(), RemoveFlit() → [이 함수]
   */
  void UpdatePriority();

  /*
   * [한국어]
   * GetPriority() - VC의 현재 우선순위 값을 반환한다
   *
   * @return: _pri (높을수록 우선순위 높음)
   *
   * 호출 체인: 라우터 VC 할당기 → [이 함수]
   */
  inline int GetPriority( ) const
  {
    return _pri;
  }

  /*
   * [한국어]
   * Route() - 라우팅 함수를 호출하여 출구 포트 집합을 계산한다
   *
   * @rf: 사용할 라우팅 함수 포인터 (예: dim_order_mesh, valiant_mesh)
   * @router: 현재 라우터
   * @f: 라우팅 대상 Flit (HEAD flit)
   * @in_channel: 이 VC의 입구 채널 번호
   *
   * rf(router, f, in_channel, _route_set, false) 호출로 _route_set 채움.
   * SetOutput()을 -1로 리셋하여 이전 할당 무효화.
   *
   * 호출 체인: 라우터 라우팅 단계 → [이 함수]
   */
  void Route( tRoutingFunction rf, const Router* router, const Flit* f, int in_channel );

  /*
   * [한국어]
   * GetOccupancy() - VC 버퍼의 현재 Flit 수를 반환한다
   *
   * @return: _buffer.size()
   *
   * 버퍼 점유율 통계, 흐름 제어 판단에 사용.
   *
   * 호출 체인: 통계 코드, 디버그 → [이 함수]
   */
  inline int GetOccupancy() const
  {
    return (int)_buffer.size();
  }

  // ==== Debug functions ====

  /*
   * [한국어]
   * SetWatch() - 이 VC의 watch 추적 여부를 설정한다
   *
   * @watch: true이면 이 VC 관련 이벤트를 gWatchOut에 출력
   *
   * 호출 체인: 외부 디버그 코드 → [이 함수]
   */
  void SetWatch( bool watch = true );

  /*
   * [한국어]
   * IsWatched() - watch 추적 여부를 반환한다
   *
   * @return: _watched
   *
   * 호출 체인: 외부 디버그 코드 → [이 함수]
   */
  bool IsWatched( ) const;

  /*
   * [한국어]
   * Display() - VC 상태를 스트림에 출력한다 (디버그)
   *
   * @os: 출력 스트림 (기본: cout)
   *
   * idle 상태가 아닌 VC의 상태, 출구 포트/VC, 버퍼 크기, front flit ID, 우선순위 출력.
   *
   * 호출 체인: Router::Display() → [이 함수]
   */
  void Display( ostream & os = cout ) const;
};

#endif
