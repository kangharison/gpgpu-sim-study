// $Id: vc.cpp 5188 2012-08-30 00:31:31Z dub $

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

/*vc.cpp
 *
 *this class describes a virtual channel in a router
 *it includes buffers and virtual channel state and controls
 *
 *This class calls the routing functions
 */

/*
 * [한국어 설명] 가상 채널(Virtual Channel) 구현 (vc.cpp)
 *
 * === 파일의 역할 ===
 * VC 클래스의 모든 메서드를 구현한다. 핵심은 Flit 버퍼 관리(AddFlit/RemoveFlit),
 * 상태 기계 전이(SetState), 라우팅 계산(Route), 우선순위 관리(UpdatePriority)이다.
 * 특히 UpdatePriority의 우선순위 기증(donation) 메커니즘은 HOL 패킷의
 * 스케줄링 우선순위를 높여 긴 패킷의 지연을 줄이는 최적화이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   Router 파이프라인 (vc.cpp 메서드 호출 순서):
 *     ReadInputs: AddFlit() → UpdatePriority()
 *     RouteCompute: Route() → _route_set 채움
 *     VCAlloc: GetRouteSet() → SetOutput()
 *     SwitchAlloc: RemoveFlit() → UpdatePriority()
 *
 * === 타 모듈과의 연결 ===
 * - routefunc.hpp: Route()에서 tRoutingFunction 함수 포인터 호출
 * - outputset.hpp: _route_set이 OutputSet 타입 (AddRange/GetSet)
 * - globals.hpp: gWatchOut (디버그 추적 출력 스트림)
 *
 * === 주요 함수/구조체 요약 ===
 * - VCSTATE[]: 상태 이름 문자열 배열
 * - VC(): 설정 파일 기반 초기화
 * - AddFlit(): 버퍼 삽입 + 패킷 연속성 검증 + 우선순위 업데이트
 * - RemoveFlit(): 버퍼 추출 + 우선순위 재계산
 * - UpdatePriority(): 우선순위 기증 포함 다중 정책 우선순위 계산
 * - Route(): 라우팅 함수 래퍼
 */

#include <limits>    // [한국어] numeric_limits<int>::max(): local_age_based 우선순위 계산에 사용
#include <sstream>   // [한국어] ostringstream: 오류 메시지 생성

#include "globals.hpp"  // [한국어] gWatchOut, GetSimTime() 전역 변수
#include "booksim.hpp"  // [한국어] BookSim 공통 정의 (Error 함수 등)
#include "vc.hpp"       // [한국어] VC 클래스 선언

// [한국어] VC 상태 이름 문자열 배열 (디버그 출력에서 상태를 가독성 있게 표시)
const char * const VC::VCSTATE[] = {"idle",    // [한국어] eVCState::idle
				    "routing",  // [한국어] eVCState::routing
				    "vc_alloc", // [한국어] eVCState::vc_alloc
				    "active"};  // [한국어] eVCState::active

/*
 * [한국어]
 * VC::VC() - 가상 채널 초기화
 *
 * @config: BookSim 설정 파일 파싱 결과
 * @outputs: 이 라우터의 출구 포트 수 (OutputSet 크기에 사용)
 * @parent: 부모 Module (라우터)
 * @name: VC 이름 문자열
 *
 * 설정 파일에서 읽는 값:
 *   - "routing_delay" == 0이면 lookahead 라우팅 → _route_set = NULL
 *   - "priority" 문자열로 _pri_type 결정
 *   - "vc_priority_donation" 값으로 우선순위 기증 활성화
 *
 * 초기 상태: idle, _out_port=-1, _out_vc=-1, _pri=0, _watched=false
 *
 * 호출 체인: Router 생성자 → (각 입구 포트 × VC 수) → [이 함수]
 */
VC::VC( const Configuration& config, int outputs,
	Module *parent, const string& name )
  : Module( parent, name ),
    _state(idle), _out_port(-1), _out_vc(-1), _pri(0), _watched(false),
    _expected_pid(-1), _last_id(-1), _last_pid(-1)
{
  _lookahead_routing = !config.GetInt("routing_delay"); // [한국어] routing_delay가 0이면 lookahead 모드
  _route_set = _lookahead_routing ? NULL : new OutputSet( ); // [한국어] lookahead이면 NULL, 아니면 OutputSet 할당

  string priority = config.GetStr( "priority" ); // [한국어] 우선순위 정책 문자열 읽기
  if ( priority == "local_age" ) { // [한국어] VC에 도착한 시간 기준 (오래된 flit 우선)
    _pri_type = local_age_based;
  } else if ( priority == "queue_length" ) { // [한국어] VC 버퍼 길이 기준
    _pri_type = queue_length_based;
  } else if ( priority == "hop_count" ) { // [한국어] 누적 홉 수 기준
    _pri_type = hop_count_based;
  } else if ( priority == "none" ) { // [한국어] 우선순위 없음 (균등 처리)
    _pri_type = none;
  } else { // [한국어] 그 외 (Flit의 pri 필드 직접 사용)
    _pri_type = other;
  }

  _priority_donation = config.GetInt("vc_priority_donation"); // [한국어] 우선순위 기증 활성화 여부
}

/*
 * [한국어]
 * VC::~VC() - 소멸자
 *
 * lookahead 라우팅이 아닌 경우 _route_set을 delete.
 * lookahead 모드에서는 _route_set이 NULL이므로 delete 불필요.
 *
 * 호출 체인: Router 소멸 → [이 함수]
 */
VC::~VC()
{
  if(!_lookahead_routing) { // [한국어] lookahead 모드가 아닐 때만 _route_set 해제
    delete _route_set; // [한국어] OutputSet 메모리 해제
  }
}

/*
 * [한국어]
 * VC::AddFlit() - VC 버퍼에 Flit을 추가한다
 *
 * @f: 추가할 Flit 포인터 (NULL이면 assert 실패)
 *
 * 동작 순서:
 *   1. 패킷 연속성 검증 (_expected_pid):
 *      - non-tail flit 도착 시 _expected_pid 설정
 *      - 이후 flit의 pid가 불일치하면 Error() 호출
 *      - tail 도착 시 _expected_pid = -1로 리셋
 *   2. local_age_based이면 f->pri = INT_MAX - 현재시간 (오래될수록 높은 우선순위)
 *   3. hop_count_based이면 f->pri = f->hops
 *   4. _buffer.push_back(f)
 *   5. UpdatePriority() 호출
 *
 * 호출 체인: Router 입력 단계 → [이 함수]
 */
void VC::AddFlit( Flit *f )
{
  assert(f); // [한국어] NULL flit은 허용되지 않음

  if(_expected_pid >= 0) { // [한국어] 현재 수신 중인 패킷이 있으면 연속성 검증
    if((long long int)f->pid != _expected_pid) { // [한국어] pid 불일치: 다른 패킷의 flit이 섞임
      ostringstream err;
      err << "Received flit " << f->id << " with unexpected packet ID: " << f->pid
	  << " (expected: " << _expected_pid << ")";
      Error(err.str()); // [한국어] 치명적 오류: VC 무결성 위반
    } else if(f->tail) { // [한국어] 예상된 pid의 tail flit 도착: 패킷 완료
      _expected_pid = -1; // [한국어] 다음 패킷 수락 준비 (경계 표시 리셋)
    }
  } else if(!f->tail) { // [한국어] 새 패킷의 non-tail flit 도착: 연속성 추적 시작
    _expected_pid = f->pid; // [한국어] 이 패킷의 pid를 예상 pid로 설정
  }

  // update flit priority before adding to VC buffer
  if(_pri_type == local_age_based) { // [한국어] 도착 시각 기반 우선순위: 오래된 flit일수록 높음
    f->pri = numeric_limits<int>::max() - GetSimTime(); // [한국어] 현재 시각 역수로 우선순위 설정
    assert(f->pri >= 0); // [한국어] 우선순위 음수 방지 (오버플로 방지)
  } else if(_pri_type == hop_count_based) { // [한국어] 홉 수 기반: 더 많이 이동한 flit 우선
    f->pri = f->hops; // [한국어] 누적 홉 수를 우선순위로 설정
    assert(f->pri >= 0);
  }

  _buffer.push_back(f); // [한국어] flit을 VC 버퍼 뒤에 추가 (FIFO 순서 유지)
  UpdatePriority(); // [한국어] 버퍼 변경 후 VC 스케줄링 우선순위 재계산
}

/*
 * [한국어]
 * VC::RemoveFlit() - VC 버퍼 맨 앞에서 Flit을 꺼낸다
 *
 * @return: _buffer.front()에서 꺼낸 Flit 포인터
 *
 * 버퍼가 비어있으면 Error() 호출 (잘못된 호출 감지).
 * _last_id, _last_pid 업데이트 후 UpdatePriority() 호출.
 *
 * 호출 체인: 라우터 전송 단계(SwitchAlloc 이후) → [이 함수]
 */
Flit *VC::RemoveFlit( )
{
  Flit *f = NULL;
  if ( !_buffer.empty( ) ) { // [한국어] 버퍼에 flit이 있으면
    f = _buffer.front( ); // [한국어] 맨 앞 flit 참조
    _buffer.pop_front( ); // [한국어] 버퍼에서 제거
    _last_id = f->id;     // [한국어] 마지막 제거 flit ID 기록 (디버그용)
    _last_pid = f->pid;   // [한국어] 마지막 제거 패킷 ID 기록
    UpdatePriority();     // [한국어] 버퍼 변경 후 우선순위 재계산
  } else { // [한국어] 비어있는 VC에서 RemoveFlit 시도: 오류
    Error("Trying to remove flit from empty buffer.");
  }
  return f; // [한국어] 꺼낸 flit 반환 (라우터가 FlitChannel::Send()로 전달)
}



/*
 * [한국어]
 * VC::SetState() - VC 상태를 전이시킨다
 *
 * @s: 새로운 eVCState 상태
 *
 * watch 플래그가 있는 front flit이 있으면 상태 전이 메시지를 gWatchOut에 출력.
 * 이후 _state = s로 업데이트.
 *
 * 호출 체인: 라우터 파이프라인 각 단계 → [이 함수]
 */
void VC::SetState( eVCState s )
{
  Flit * f = FrontFlit(); // [한국어] 현재 처리 중인 head flit 확인

  if(f && f->watch) // [한국어] watch 대상 flit이면 상태 전이 추적 메시지 출력
    *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		<< "Changing state from " << VC::VCSTATE[_state] // [한국어] 현재 상태 이름
		<< " to " << VC::VCSTATE[s] << "." << endl; // [한국어] 새 상태 이름

  _state = s; // [한국어] 상태 업데이트
}

/*
 * [한국어]
 * VC::GetRouteSet() - 라우팅 계산 결과 반환
 *
 * @return: _route_set 포인터 (lookahead 모드에서는 NULL)
 *
 * 호출 체인: 라우터 VC 할당기 → [이 함수]
 */
const OutputSet *VC::GetRouteSet( ) const
{
  return _route_set; // [한국어] 라우팅 함수가 채운 출구 포트 집합 반환
}

/*
 * [한국어]
 * VC::SetRouteSet() - 라우팅 결과를 외부에서 설정한다
 *
 * @output_set: 새로운 OutputSet 포인터
 *
 * _out_port, _out_vc를 -1로 리셋하여 이전 VC 할당 정보를 무효화.
 * lookahead 라우팅에서 Flit::la_route_set을 VC에 연결할 때 사용.
 *
 * 호출 체인: 라우터 라우팅 단계 → [이 함수]
 */
void VC::SetRouteSet( OutputSet * output_set )
{
  _route_set = output_set; // [한국어] 새 라우팅 결과 설정
  _out_port = -1; // [한국어] 이전 출구 포트 할당 무효화
  _out_vc = -1;   // [한국어] 이전 다운스트림 VC 할당 무효화
}

/*
 * [한국어]
 * VC::SetOutput() - 출구 포트와 다운스트림 VC를 설정한다
 *
 * @port: VC 할당기가 결정한 출구 포트 번호
 * @vc: VC 할당기가 결정한 다운스트림 VC 번호
 *
 * 호출 체인: 라우터 VC 할당기 → [이 함수]
 */
void VC::SetOutput( int port, int vc )
{
  _out_port = port; // [한국어] 확정된 출구 포트 저장
  _out_vc   = vc;   // [한국어] 확정된 다운스트림 VC 저장
}

/*
 * [한국어]
 * VC::UpdatePriority() - VC의 스케줄링 우선순위(_pri)를 재계산한다
 *
 * 버퍼가 비어있으면 즉시 반환.
 * _pri_type에 따라:
 *   queue_length_based: _pri = _buffer.size()
 *   other + priority_donation: 버퍼 전체를 순회하여 최고 pri flit을 찾아
 *     front flit에게 우선순위 "기증" (HOL 패킷이 높은 우선순위를 얻도록)
 *   기타: front flit의 pri 값을 _pri로 설정
 *
 * 우선순위 기증의 목적:
 *   큐의 뒤에 있는 중요 flit이 앞의 느린 패킷 때문에 지연되는 것을 방지.
 *   HOL 패킷이 뒤의 flit 우선순위를 빌려 더 높은 우선순위로 스케줄링됨.
 *
 * 호출 체인: AddFlit(), RemoveFlit() → [이 함수]
 */
void VC::UpdatePriority()
{
  if(_buffer.empty()) return; // [한국어] 버퍼 비어있으면 우선순위 계산 불필요
  if(_pri_type == queue_length_based) { // [한국어] 큐 길이 기반: 버퍼 크기가 우선순위
    _pri = _buffer.size(); // [한국어] 큰 큐 = 높은 우선순위 (공정성 향상)
  } else if(_pri_type != none) { // [한국어] none 외의 모든 정책: front flit의 pri 참조
    Flit * f = _buffer.front(); // [한국어] front flit (현재 HOL 패킷의 대표 flit)
    if((_pri_type != local_age_based) && _priority_donation) { // [한국어] 우선순위 기증 활성화
      Flit * df = f; // [한국어] 기증할 최고 우선순위 flit 초기화 (front flit으로 시작)
      for(size_t i = 1; i < _buffer.size(); ++i) { // [한국어] 버퍼 전체 순회하여 최고 pri flit 탐색
	Flit * bf = _buffer[i];
	if(bf->pri > df->pri) df = bf; // [한국어] 더 높은 우선순위 flit 발견 시 갱신
      }
      if((df != f) && (df->watch || f->watch)) { // [한국어] 기증이 발생하고 watch 대상이면 로그 출력
	*gWatchOut << GetSimTime() << " | " << FullName() << " | "
		    << "Flit " << df->id
		    << " donates priority to flit " << f->id // [한국어] 기증자(df) → 수혜자(f) 로그
		    << "." << endl;
      }
      f = df; // [한국어] 최고 우선순위 flit을 대표로 선택
    }
    if(f->watch) // [한국어] watch 대상이면 우선순위 설정 로그 출력
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
		  << "Flit " << f->id
		  << " sets priority to " << f->pri // [한국어] 설정될 우선순위 값
		  << "." << endl;
    _pri = f->pri; // [한국어] 선택된 flit의 우선순위를 VC 우선순위로 설정
  }
}


/*
 * [한국어]
 * VC::Route() - 라우팅 함수를 호출하여 출구 포트 집합을 계산한다
 *
 * @rf: 라우팅 함수 포인터 (routefunc.cpp에 등록된 함수)
 * @router: 현재 라우터 (위치 정보 제공)
 * @f: 라우팅 대상 HEAD Flit (목적지/타입 정보)
 * @in_channel: 이 VC의 입구 채널 번호 (DOR 라우팅에서 방향 결정에 사용)
 *
 * rf()는 _route_set(OutputSet)에 가능한 출구 포트와 VC 범위를 AddRange()로 추가.
 * 이후 _out_port = -1, _out_vc = -1로 리셋하여 VC 할당 대기 상태로 전환.
 *
 * 호출 체인: 라우터 라우팅 단계 → [이 함수] → rf(router, f, in_channel, _route_set, false)
 */
void VC::Route( tRoutingFunction rf, const Router* router, const Flit* f, int in_channel )
{
  rf( router, f, in_channel, _route_set, false ); // [한국어] 라우팅 함수 호출: _route_set에 출구 포트 집합 채움
  _out_port = -1; // [한국어] VC 할당 전: 출구 포트 미확정 상태로 리셋
  _out_vc = -1;   // [한국어] VC 할당 전: 다운스트림 VC 미확정 상태로 리셋
}

// ==== Debug functions ====

/*
 * [한국어]
 * VC::SetWatch() - VC 추적 여부 설정
 *
 * @watch: true이면 이 VC의 이벤트를 gWatchOut에 출력
 *
 * 호출 체인: 외부 디버그 코드 → [이 함수]
 */
void VC::SetWatch( bool watch )
{
  _watched = watch; // [한국어] watch 플래그 설정
}

/*
 * [한국어]
 * VC::IsWatched() - VC 추적 여부 반환
 *
 * @return: _watched
 *
 * 호출 체인: 외부 디버그 코드 → [이 함수]
 */
bool VC::IsWatched( ) const
{
  return _watched; // [한국어] _watched 값 반환
}

/*
 * [한국어]
 * VC::Display() - VC 상태를 스트림에 출력한다 (디버그)
 *
 * @os: 출력 스트림 (기본: cout)
 *
 * idle 상태가 아닌 VC만 출력한다 (idle은 무의미하므로 건너뜀).
 * 출력 정보: 상태, 출구 포트/VC (active 상태일 때), 버퍼 크기, front flit ID, 우선순위.
 *
 * 호출 체인: Router::Display() → [이 함수]
 */
void VC::Display( ostream & os ) const
{
  if ( _state != VC::idle ) { // [한국어] idle 상태가 아닌 VC만 출력 (idle은 표시 불필요)
    os << FullName() << ": "
       << " state: " << VCSTATE[_state]; // [한국어] 현재 상태 이름 출력
    if(_state == VC::active) { // [한국어] active 상태이면 확정된 출구 포트/VC도 출력
      os << " out_port: " << _out_port
	 << " out_vc: " << _out_vc;
    }
    os << " fill: " << _buffer.size(); // [한국어] 현재 버퍼에 저장된 flit 수
    if(!_buffer.empty()) {
      os << " front: " << _buffer.front()->id; // [한국어] 현재 처리 중인 front flit ID
    }
    os << " pri: " << _pri; // [한국어] 현재 스케줄링 우선순위
    os << endl;
  }
}
