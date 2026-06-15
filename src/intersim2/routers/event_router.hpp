// $Id: event_router.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] 이벤트 기반 NoC 라우터 선언 (event_router.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 BookSim2 NoC 시뮬레이터의 이벤트 기반 라우터(EventRouter)와
 * 그 보조 클래스(EventNextVCState)의 인터페이스를 선언한다.
 * EventRouter는 IQRouter(사이클마다 모든 버퍼를 폴링)와 달리,
 * 플릿(flit)이나 크레딧(credit)이 실제로 이동할 때만 "이벤트"를 생성해
 * 처리하는 이벤트 드리븐(event-driven) 플로우 컨트롤 모델을 구현한다.
 * 불필요한 폴링을 제거함으로써 대규모 NoC 시뮬레이션에서 시뮬레이션 속도를
 * 향상시키는 것이 주된 목적이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 시뮬레이션 계층:
 *   GPU 코어(SM) → 메모리 요청(mem_fetch) → icnt_wrapper
 *     → intersim2 / LocalInterconnect → [EventRouter 또는 IQRouter]
 *       → 출력 포트 → 다음 라우터 또는 메모리 컨트롤러
 *
 * EventRouter는 gpgpusim.config의 `router event` 옵션으로 활성화된다.
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 단일 스레드 시뮬레이션 루프.
 * 사이클마다 _InternalStep()이 호출되며, 이벤트가 발생한 경로만 처리한다.
 *
 * 2단계 파이프라인:
 *   Phase 1 (Arrival): 플릿 도착 → tArrivalEvent 생성 → arrival_pipe 지연
 *                      → ArrivalRequests() → ArrivalArb() → 출력 VC 소유권 확보
 *   Phase 2 (Transport): tTransportEvent 생성 → TransportRequests()
 *                        → TransportArb() → crossbar_pipe → OutputQueuing()
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - Router (router.hpp): 공통 라우터 기반 클래스 — 입출력 채널, 크레딧 채널 관리
 *   - Buffer (buffer.hpp): 입력 VC 버퍼, 라우팅 결과 저장
 *   - PipelineFIFO (pipefifo.hpp): 크로스바/크레딧/도착 지연 모델링
 *   - PriorityArbiter (prio_arb.hpp): 도착/전송 이벤트 중재
 *   - Flit / Credit: NoC 전송 단위 (flit) 및 역방향 흐름 제어 신호 (credit)
 *   - tRoutingFunction (routefunc.hpp): 라우팅 함수 포인터 타입
 *
 * 데이터 흐름:
 *   외부 입력 채널 → _input_buffer → _buf(VC 버퍼)
 *   → tArrivalEvent → _arrival_pipe → _arrival_queue
 *   → _transport_queue → _crossbar_pipe → _output_buffer → 외부 출력 채널
 *   크레딧 역방향: _out_cred_buffer → (상태 갱신) → _in_cred_buffer → 업스트림 전송
 *
 * === 주요 함수/구조체 요약 ===
 * - EventNextVCState: 각 출력 포트의 VC 상태(idle/busy/tail_pending), 크레딧,
 *   프레즌스, 대기열을 추적하는 상태 머신 클래스
 * - EventRouter: 이벤트 드리븐 NoC 라우터 메인 클래스
 * - EventRouter::_IncomingFlits(): 입력 버퍼의 플릿을 VC 버퍼로 이동하고
 *   tArrivalEvent를 생성해 arrival_pipe에 삽입
 * - EventRouter::_ArrivalArb(): 출력 VC 소유권을 중재하고 크레딧 처리,
 *   tTransportEvent를 생성
 * - EventRouter::_TransportArb(): 입력 포트 중재 후 플릿을 crossbar_pipe로 전송
 */

#ifndef _EVENT_ROUTER_HPP_
#define _EVENT_ROUTER_HPP_

#include <string>   // [한국어] 모듈 이름, 설정 키 문자열 처리에 사용
#include <queue>    // [한국어] 이벤트 큐(arrival_queue, transport_queue) 및 버퍼 큐에 사용
#include <vector>   // [한국어] 입출력 포트별 가변 배열 자료구조에 사용

#include "module.hpp"     // [한국어] 계층적 모듈 시스템 기반 — FullName(), Error() 등 제공
#include "router.hpp"     // [한국어] Router 기반 클래스 — 입출력 채널, 크레딧 채널 관리
#include "buffer.hpp"     // [한국어] VC별 입력 버퍼 (Buffer 클래스) — 플릿 저장 및 라우팅 정보 유지
#include "vc.hpp"         // [한국어] VC(Virtual Channel) 상태 정의 — VC_IDLE 등
#include "prio_arb.hpp"   // [한국어] 우선순위 중재기 (PriorityArbiter) — arrival/transport 이벤트 중재
#include "routefunc.hpp"  // [한국어] 라우팅 함수 포인터 타입(tRoutingFunction) 및 전역 맵
#include "outputset.hpp"  // [한국어] 라우팅 결과(OutputSet) — (포트, VC) 쌍 집합
#include "pipefifo.hpp"   // [한국어] 파이프라인 FIFO 지연 모델 (PipelineFIFO 템플릿)

/*
 * [한국어]
 * EventNextVCState - 하나의 출력 포트에 속한 모든 VC의 상태를 추적하는 클래스
 *
 * 이벤트 드리븐 라우터에서 각 출력 포트는 자신에게 연결된 다운스트림 라우터
 * 버퍼의 상태를 VC별로 추적해야 한다. EventNextVCState는 그 역할을 담당한다.
 *
 * 상태 전이:
 *   idle → busy: head 플릿 도착 시 ArrivalArb()에서 VC를 할당
 *   busy → idle: tail 크레딧 수신 후 대기 중인 연결이 없을 때 _ProcessWaiting()에서 전환
 *   busy → busy: tail 크레딧 수신 후 _waiting 큐에 다음 연결이 있으면 ownership 이전
 *   (tail_pending 상태는 선언되어 있으나 현재 로직에서는 주로 idle/busy를 사용)
 *
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, EventRouter::_InternalStep() 내에서만 접근.
 * 단일 스레드이므로 별도 동기화 불필요.
 *
 * 호출 체인:
 *   EventRouter::_ArrivalArb() → SetState/SetInput/SetInputVC/PushWaiting/...
 *   EventRouter::_ProcessWaiting() → GetCredits/SetCredits/PopWaiting/...
 */
class EventNextVCState : public Module {
public:
  /*
   * [한국어]
   * eNextVCState - 출력 VC의 현재 할당 상태를 나타내는 열거형
   *
   * 이 상태는 EventRouter가 각 출력 VC에 대해 새 패킷을 받을 수 있는지,
   * 이미 다른 입력이 점유 중인지 판단하는 데 사용된다.
   * 설정자: _ArrivalArb()와 _ProcessWaiting()이 SetState()로 전환.
   * 읽는 자: _ArrivalArb()에서 head/body 플릿 처리 경로 결정에 사용.
   * 동기화: 단일 스레드 시뮬레이션 — 잠금 불필요.
   */
  enum eNextVCState {
    idle,         /* [한국어] VC 유휴 상태 — 어떤 패킷도 이 VC를 점유하지 않음.
                   * head 플릿 도착 시 즉시 busy로 전환 가능. */
    busy,         /* [한국어] VC 사용 중 상태 — 특정 (input, src_vc) 쌍이 이 VC를 점유.
                   * _input[vc], _inputVC[vc]에 현재 소유자 기록.
                   * 이 상태에서 tail 크레딧 수신 → _ProcessWaiting() 호출. */
    tail_pending  /* [한국어] tail 전송 후 크레딧을 기다리는 중간 상태.
                   * 현재 구현에서는 주로 선언 목적으로 유지되며,
                   * 실제 tail 처리는 busy 상태에서 tail credit 수신 시 직접 처리. */
  };

  /*
   * [한국어]
   * tWaiting - 출력 VC가 busy 상태일 때 그 VC를 기다리는 입력 연결 정보
   *
   * head 플릿이 도착했는데 목표 출력 VC가 이미 busy 상태라면,
   * 해당 입력 연결의 정보를 tWaiting으로 패키징하여 _waiting[vc] 리스트에 삽입한다.
   * 출력 VC가 해제될 때 _ProcessWaiting()이 리스트의 front를 꺼내
   * 새 소유자로 설정하고 tTransportEvent를 생성한다.
   *
   * 설정자: _ArrivalArb()가 new tWaiting을 생성하고 PushWaiting()으로 삽입.
   * 읽는 자: _ProcessWaiting()이 PopWaiting()으로 꺼내 다음 소유자 결정.
   * 동기화: 단일 스레드, 잠금 불필요.
   */
  struct tWaiting {
    int  input;
    /* [한국어] 이 대기 연결의 업스트림 입력 포트 인덱스.
     * 설정자: _ArrivalArb()에서 aevt->input 값으로 설정.
     * 읽는 자: _ProcessWaiting()이 tevt->input으로 복사해 tTransportEvent 생성.
     * 값 범위: [0, _inputs) 내의 유효한 포트 인덱스. */

    int  vc;
    /* [한국어] 이 대기 연결의 업스트림 입력 VC 인덱스 (src_vc).
     * 설정자: _ArrivalArb()에서 aevt->src_vc 값으로 설정.
     * 읽는 자: _ProcessWaiting()이 tevt->src_vc로 복사.
     *         IncrWaiting()이 (input, vc) 쌍으로 기존 항목 탐색에 사용.
     * 값 범위: [0, _vcs) 내의 유효한 VC 인덱스. */

    int  id;
    /* [한국어] 이 대기 연결을 생성한 플릿의 고유 식별자 (디버그 목적).
     * 설정자: _ArrivalArb()에서 aevt->id 값으로 설정.
     * 읽는 자: _ProcessWaiting()에서 tevt->id로 복사.
     *         watch 플래그가 true일 때 cout 출력에 사용.
     * 값 범위: 시뮬레이터 전역 플릿 ID, 음수 불가. */

    int  pres;
    /* [한국어] 이 대기 연결이 입력 버퍼에 가지고 있는 플릿 수 (presence count).
     * head 플릿 도착 시 1로 초기화.
     * 이후 같은 (input, vc) 쌍의 body 플릿 도착 이벤트마다 IncrWaiting()으로 +1.
     * 설정자: _ArrivalArb()가 w->pres = 1로 초기화; IncrWaiting()이 (*match)->pres++ 증가.
     * 읽는 자: _ProcessWaiting()이 tevt 생성 시 w->pres - 1을 SetPresence()로 설정.
     * 값 범위: 1 이상 (0이면 대기 항목 자체가 없어야 함). */

    bool watch;
    /* [한국어] 이 대기 연결에 대한 디버그 추적 활성 여부.
     * 설정자: _ArrivalArb()에서 aevt->watch 값으로 복사.
     * 읽는 자: _ProcessWaiting()이 cout 출력 여부 결정에 사용.
     * 값 범위: true(추적 활성) / false(추적 비활성). */
  };

private:
  int _buf_size;
  /* [한국어] 각 VC의 다운스트림 버퍼 크기 (슬롯 수).
   * 설정자: 생성자에서 config.GetInt("vc_buf_size")로 초기화.
   * 읽는 자: 생성자에서 _credits[v] = _buf_size로 초기 크레딧 설정에 사용.
   * 값 범위: 1 이상의 양의 정수 (gpgpusim.config의 vc_buf_size 값).
   * 동기화: 초기화 후 읽기 전용 — 잠금 불필요. */

  int _vcs;
  /* [한국어] 이 출력 포트가 관리하는 VC의 총 개수.
   * 설정자: 생성자에서 config.GetInt("num_vcs")로 초기화.
   * 읽는 자: 모든 per-VC 벡터(_credits, _presence, _state 등) 크기 결정에 사용.
   *         각 접근자/설정자 함수에서 범위 검사(assert)에 사용.
   * 값 범위: 1 이상의 양의 정수.
   * 동기화: 초기화 후 읽기 전용 — 잠금 불필요. */

  vector<int> _credits;
  /* [한국어] VC별 남은 다운스트림 크레딧 수 (다운스트림 버퍼 가용 슬롯).
   * 각 원소 _credits[v]는 출력 VC v에 대해 이 라우터가 얼마나 많은 플릿을
   * 추가로 전송할 수 있는지를 나타낸다.
   * 설정자: 생성자에서 _buf_size로 초기화.
   *         _ArrivalArb()에서 크레딧 수신 시 SetCredits()로 +1.
   *         _SendTransport()/_ProcessWaiting()에서 플릿 전송 시 SetCredits()로 -1.
   * 읽는 자: _SendTransport()/_ProcessWaiting()이 크레딧 가용 여부 확인.
   * 값 범위: [0, _buf_size]. 음수 불가. */

  vector<int> _presence;
  /* [한국어] VC별 입력 버퍼 내 대기 플릿 수 (presence count).
   * 크레딧이 없어서 즉시 transport 이벤트를 생성할 수 없을 때,
   * 입력 버퍼에 플릿이 존재한다는 사실을 기록하는 카운터.
   * 크레딧이 나중에 반환되면 이 카운터를 보고 지연된 transport 이벤트를 생성한다.
   * 설정자: 생성자에서 0으로 초기화.
   *         _SendTransport()에서 크레딧 없을 시 SetPresence()로 +1.
   *         _ArrivalArb()에서 크레딧 반환 후 SetPresence()로 -1.
   *         _ProcessWaiting()에서 waiting 처리 시 업데이트.
   * 읽는 자: _ArrivalArb()에서 크레딧 반환 시 즉시 transport 이벤트 생성 여부 결정.
   * 값 범위: 0 이상. */

  vector<int> _input;
  /* [한국어] VC별 현재 소유자의 입력 포트 인덱스.
   * busy 상태의 VC에 대해, 어느 입력 포트가 이 VC를 점유하고 있는지 기록한다.
   * 설정자: _ArrivalArb()에서 SetInput()으로 설정 (head 플릿이 VC를 획득할 때).
   *         _ProcessWaiting()에서 SetInput()으로 교체 (대기 중인 연결로 소유권 이전 시).
   * 읽는 자: _ArrivalArb()에서 body 플릿이 현재 소유자 소속인지 확인.
   *          _ArrivalArb()에서 크레딧 반환 시 tevt->input 설정.
   * 값 범위: [0, _inputs) 범위의 유효 포트 인덱스. idle 상태에서는 미정의. */

  vector<int> _inputVC;
  /* [한국어] VC별 현재 소유자의 입력 VC 인덱스 (src_vc).
   * busy 상태의 VC에 대해, 어느 입력 VC가 이 출력 VC를 점유하는지 기록한다.
   * 설정자: _ArrivalArb()에서 SetInputVC()로 설정 (head 플릿이 VC를 획득할 때).
   *         _ProcessWaiting()에서 SetInputVC()로 교체.
   * 읽는 자: _ArrivalArb()에서 body 플릿 검증, tevt->src_vc 설정에 사용.
   * 값 범위: [0, _vcs) 범위의 유효 VC 인덱스. idle 상태에서는 미정의. */

  vector<list<tWaiting *> > _waiting;
  /* [한국어] VC별 대기 연결 리스트 — busy VC를 기다리는 연결들의 FIFO 큐.
   * _waiting[v]는 출력 VC v가 busy 상태일 때 이 VC를 기다리는 tWaiting 포인터 리스트.
   * list를 사용하는 이유: FIFO 순서로 pop_front()와 push_back()을 모두 O(1)에 수행,
   * 그리고 IncrWaiting()에서 중간 탐색이 필요하기 때문.
   * 설정자: PushWaiting()으로 삽입; PopWaiting()으로 front 제거.
   *         IncrWaiting()이 기존 항목의 pres 필드를 수정.
   * 읽는 자: IsWaiting()이 비어있는지 확인; IsInputWaiting()이 특정 항목 탐색.
   * 동기화: 단일 스레드 시뮬레이션 — 잠금 불필요.
   * 메모리: tWaiting 포인터를 저장하므로, PopWaiting() 후 호출자가 delete 책임. */

  vector<eNextVCState> _state;
  /* [한국어] VC별 현재 상태 (idle / busy / tail_pending).
   * 설정자: 생성자에서 idle로 초기화.
   *         SetState()를 통해 _ArrivalArb() 및 _ProcessWaiting()이 전환.
   * 읽는 자: _ArrivalArb()에서 head/body 플릿 처리 경로 결정.
   *          GetState()를 통해 크레딧 수신 처리 경로 결정.
   * 값 범위: eNextVCState 열거형의 세 가지 값.
   * 동기화: 단일 스레드 시뮬레이션 — 잠금 불필요. */

public:

  /*
   * [한국어]
   * EventNextVCState 생성자 - VC 상태 추적 객체 초기화
   *
   * @config: 시뮬레이터 전역 설정 — vc_buf_size, num_vcs 값을 읽음
   * @parent: 부모 모듈 (이 객체를 소유하는 EventRouter 인스턴스)
   * @name: 이 모듈의 이름 문자열 (디버그 출력에 사용)
   * @return: 없음 (생성자)
   *
   * config에서 vc_buf_size와 num_vcs를 읽어 per-VC 벡터들을 할당·초기화한다.
   * _credits[v]는 _buf_size로 초기화 (다운스트림 버퍼 full → 최대 크레딧 보유).
   * _presence[v]는 0으로 초기화 (아직 플릿 없음).
   * _state[v]는 idle로 초기화 (아무 패킷도 VC를 점유하지 않음).
   * 실행 컨텍스트: 시뮬레이션 초기화 단계 (EventRouter 생성자에서 호출).
   *
   * 호출 체인:
   *   EventRouter 생성자 → new EventNextVCState(...) → [이 생성자]
   */
  EventNextVCState( const Configuration& config,
		    Module *parent, const string& name );

  /*
   * [한국어]
   * GetState - 지정 VC의 현재 상태 반환
   *
   * @vc: 조회할 VC 인덱스 ([0, _vcs) 범위)
   * @return: eNextVCState — idle, busy, tail_pending 중 하나
   *
   * VC 인덱스 범위 검사(assert) 후 _state[vc]를 반환한다.
   * _ArrivalArb()에서 head/body 플릿 처리 분기 결정에 사용.
   * 실행 컨텍스트: EventRouter::_InternalStep() 내부, 단일 스레드.
   *
   * 호출 체인:
   *   EventRouter::_ArrivalArb() → [GetState]
   */
  eNextVCState GetState( int vc ) const;

  /*
   * [한국어]
   * GetPresence - 지정 VC에 대해 입력 버퍼에 대기 중인 플릿 수 반환
   *
   * @vc: 조회할 VC 인덱스 ([0, _vcs) 범위)
   * @return: int — 현재 프레즌스 카운트 (0 이상)
   *
   * 크레딧 없이 도착한 플릿 수를 추적하는 _presence[vc]를 반환한다.
   * _ArrivalArb()에서 크레딧 반환 시 지연된 transport 이벤트 생성 여부 결정에 사용.
   *
   * 호출 체인:
   *   EventRouter::_ArrivalArb() → [GetPresence]
   */
  int GetPresence( int vc ) const;

  /*
   * [한국어]
   * GetCredits - 지정 VC에 대한 남은 다운스트림 크레딧 수 반환
   *
   * @vc: 조회할 VC 인덱스 ([0, _vcs) 범위)
   * @return: int — 현재 크레딧 수 (0 이상, _buf_size 이하)
   *
   * _credits[vc]를 반환한다.
   * _SendTransport()와 _ProcessWaiting()에서 즉시 transport 이벤트를 생성할 수
   * 있는지 판단하는 데 사용 (credits > 0 이면 즉시 전송 가능).
   *
   * 호출 체인:
   *   EventRouter::_SendTransport() → [GetCredits]
   *   EventRouter::_ProcessWaiting() → [GetCredits]
   */
  int GetCredits( int vc ) const;

  /*
   * [한국어]
   * GetInput - 지정 출력 VC를 현재 점유하는 입력 포트 인덱스 반환
   *
   * @vc: 조회할 출력 VC 인덱스 ([0, _vcs) 범위)
   * @return: int — 현재 소유자의 입력 포트 인덱스 (busy 상태에서만 유효)
   *
   * _input[vc]를 반환한다.
   * _ArrivalArb()에서 body 플릿이 현재 VC 소유자와 같은 입력에서 왔는지 확인,
   * 크레딧 반환 시 tevt->input 설정에 사용.
   *
   * 호출 체인:
   *   EventRouter::_ArrivalArb() → [GetInput]
   */
  int GetInput( int vc ) const;

  /*
   * [한국어]
   * GetInputVC - 지정 출력 VC를 현재 점유하는 입력 VC 인덱스 반환
   *
   * @vc: 조회할 출력 VC 인덱스 ([0, _vcs) 범위)
   * @return: int — 현재 소유자의 입력 VC 인덱스 (busy 상태에서만 유효)
   *
   * _inputVC[vc]를 반환한다.
   * _ArrivalArb()에서 body 플릿 검증 및 tevt->src_vc 설정에 사용.
   *
   * 호출 체인:
   *   EventRouter::_ArrivalArb() → [GetInputVC]
   */
  int GetInputVC( int vc ) const;

  /*
   * [한국어]
   * IsWaiting - 지정 출력 VC의 대기 리스트가 비어있지 않은지 확인
   *
   * @vc: 확인할 출력 VC 인덱스 ([0, _vcs) 범위)
   * @return: bool — true이면 대기 중인 연결이 하나 이상 존재
   *
   * _waiting[vc].empty()의 부정을 반환한다.
   * _ProcessWaiting()에서 tail 처리 후 다음 소유자가 있는지 확인에 사용.
   *
   * 호출 체인:
   *   EventRouter::_ProcessWaiting() → [IsWaiting]
   */
  bool IsWaiting( int vc ) const;

  /*
   * [한국어]
   * IsInputWaiting - 지정 출력 VC의 대기 리스트에 특정 (input, vc) 쌍이 있는지 확인
   *
   * @vc: 확인할 출력 VC 인덱스
   * @w_input: 탐색할 입력 포트 인덱스
   * @w_vc: 탐색할 입력 VC 인덱스
   * @return: bool — 해당 (input, vc) 쌍의 tWaiting 항목이 리스트에 존재하면 true
   *
   * _waiting[vc] 리스트를 순차 탐색하여 (w_input, w_vc) 일치 항목을 찾는다.
   * _ArrivalArb()에서 body 플릿이 이미 waiting 리스트에 있는 연결 소속인지
   * 확인하는 데 사용 (같은 패킷의 head가 대기 중이고 body가 추가 도착하는 경우).
   *
   * 호출 체인:
   *   EventRouter::_ArrivalArb() → [IsInputWaiting]
   */
  bool IsInputWaiting( int vc, int w_input, int w_vc ) const;

  /*
   * [한국어]
   * PushWaiting - 지정 출력 VC의 대기 리스트에 새 항목 삽입
   *
   * @vc: 대기 리스트에 삽입할 출력 VC 인덱스
   * @w: 삽입할 tWaiting 포인터 (호출자가 new로 할당, 소유권 이전)
   * @return: 없음
   *
   * _waiting[vc].push_back(w)으로 FIFO 순서로 삽입한다.
   * watch 플래그가 true이면 디버그 로그를 출력한다.
   * _ArrivalArb()에서 head 플릿이 도착했지만 목표 VC가 busy일 때 호출.
   * 소유권: tWaiting 포인터의 소유권이 이 클래스로 이전되며,
   * PopWaiting()으로 꺼낸 후 호출자(EventRouter::_ProcessWaiting)가 delete.
   *
   * 호출 체인:
   *   EventRouter::_ArrivalArb() → [PushWaiting]
   */
  void PushWaiting( int vc, tWaiting *w );

  /*
   * [한국어]
   * IncrWaiting - 지정 (input, vc) 쌍의 대기 항목 프레즌스 카운트 증가
   *
   * @vc: 출력 VC 인덱스 (대기 리스트 선택)
   * @w_input: 탐색할 입력 포트 인덱스
   * @w_vc: 탐색할 입력 VC 인덱스
   * @return: 없음 (항목을 찾지 못하면 Error() 호출)
   *
   * _waiting[vc] 리스트에서 (w_input, w_vc) 쌍을 탐색하고,
   * 찾은 항목의 pres 필드를 +1 증가시킨다.
   * _ArrivalArb()에서 body 플릿이 도착했는데 해당 패킷이 이미 waiting 리스트에
   * head로 등록된 경우 호출 — body 플릿이 입력 버퍼에 존재한다는 사실을 기록.
   *
   * 호출 체인:
   *   EventRouter::_ArrivalArb() → [IncrWaiting]
   */
  void IncrWaiting( int vc, int w_input, int w_vc );

  /*
   * [한국어]
   * PopWaiting - 지정 출력 VC의 대기 리스트 맨 앞 항목 제거 및 반환
   *
   * @vc: 대기 리스트에서 꺼낼 출력 VC 인덱스
   * @return: tWaiting* — 제거된 항목 포인터 (호출자가 delete 책임)
   *
   * _waiting[vc].front()를 저장하고 pop_front()로 리스트에서 제거한 뒤 반환.
   * _ProcessWaiting()에서 tail 처리 후 다음 소유자를 얻기 위해 호출.
   * 반환된 포인터의 소유권이 호출자로 이전되므로, _ProcessWaiting()에서 delete.
   *
   * 호출 체인:
   *   EventRouter::_ProcessWaiting() → [PopWaiting] → (호출자가 delete)
   */
  tWaiting *PopWaiting( int vc );

  /*
   * [한국어]
   * SetState - 지정 출력 VC의 상태 설정
   *
   * @vc: 상태를 변경할 출력 VC 인덱스
   * @state: 새 상태값 (idle / busy / tail_pending)
   * @return: 없음
   *
   * _state[vc] = state로 직접 설정. 범위 검사 후 대입.
   * _ArrivalArb()에서 head 플릿 수신 시 idle → busy 전환,
   * _ProcessWaiting()에서 tail 처리 후 busy → idle 또는 busy 유지.
   *
   * 호출 체인:
   *   EventRouter::_ArrivalArb() → [SetState]
   *   EventRouter::_ProcessWaiting() → [SetState]
   */
  void SetState( int vc, eNextVCState state );

  /*
   * [한국어]
   * SetCredits - 지정 출력 VC의 크레딧 값 설정
   *
   * @vc: 크레딧을 변경할 출력 VC 인덱스
   * @value: 새 크레딧 값 (0 이상 _buf_size 이하)
   * @return: 없음
   *
   * _credits[vc] = value로 직접 설정.
   * _SendTransport()/_ProcessWaiting()에서 플릿 전송 시 크레딧 감소(credits-1),
   * _ArrivalArb()에서 크레딧 수신 시 증가(credits+1) 후 호출.
   *
   * 호출 체인:
   *   EventRouter::_SendTransport() → [SetCredits]
   *   EventRouter::_ArrivalArb() → [SetCredits]
   *   EventRouter::_ProcessWaiting() → [SetCredits]
   */
  void SetCredits( int vc, int value );

  /*
   * [한국어]
   * SetPresence - 지정 출력 VC의 프레즌스 카운트 설정
   *
   * @vc: 프레즌스를 변경할 출력 VC 인덱스
   * @value: 새 프레즌스 값 (0 이상)
   * @return: 없음
   *
   * _presence[vc] = value로 직접 설정.
   * _SendTransport()에서 크레딧 없을 시 +1,
   * _ArrivalArb()에서 크레딧 반환 후 프레즌스 소모 시 -1 후 호출.
   *
   * 호출 체인:
   *   EventRouter::_SendTransport() → [SetPresence]
   *   EventRouter::_ArrivalArb() → [SetPresence]
   *   EventRouter::_ProcessWaiting() → [SetPresence]
   */
  void SetPresence( int vc, int value );

  /*
   * [한국어]
   * SetInput - 지정 출력 VC의 현재 소유자 입력 포트 설정
   *
   * @vc: 소유자를 기록할 출력 VC 인덱스
   * @input: 소유자의 입력 포트 인덱스
   * @return: 없음
   *
   * _input[vc] = input으로 직접 설정.
   * _ArrivalArb()에서 head 플릿이 VC를 획득할 때 호출.
   * _ProcessWaiting()에서 대기 연결로 소유권 이전 시 호출.
   *
   * 호출 체인:
   *   EventRouter::_ArrivalArb() → [SetInput]
   *   EventRouter::_ProcessWaiting() → [SetInput]
   */
  void SetInput( int vc, int input );

  /*
   * [한국어]
   * SetInputVC - 지정 출력 VC의 현재 소유자 입력 VC 설정
   *
   * @vc: 소유자를 기록할 출력 VC 인덱스
   * @in_vc: 소유자의 입력 VC 인덱스 (src_vc)
   * @return: 없음
   *
   * _inputVC[vc] = in_vc로 직접 설정.
   * SetInput()과 항상 쌍으로 호출되어 (input, inputVC) 소유자 쌍을 갱신.
   *
   * 호출 체인:
   *   EventRouter::_ArrivalArb() → [SetInputVC]
   *   EventRouter::_ProcessWaiting() → [SetInputVC]
   */
  void SetInputVC( int vc, int in_vc );
};

/*
 * [한국어]
 * EventRouter - 이벤트 드리븐 NoC 라우터 메인 클래스
 *
 * IQRouter와의 핵심 차이점:
 *   IQRouter는 매 사이클마다 모든 입력 버퍼를 폴링하여 플릿 유무를 확인하지만,
 *   EventRouter는 플릿/크레딧이 실제로 이동할 때만 이벤트를 생성하고 처리한다.
 *   이를 통해 불필요한 폴링 오버헤드를 제거하여 시뮬레이션 효율을 높인다.
 *
 * 2단계 파이프라인:
 *   [Phase 1 - Arrival]
 *     _IncomingFlits(): 플릿을 VC 버퍼에 저장하고 tArrivalEvent를 _arrival_pipe에 삽입
 *     _arrival_pipe->Advance(): 라우팅 지연 시뮬레이션
 *     _ArrivalRequests(): _arrival_pipe에서 이벤트를 꺼내 arrival_arbiter에 요청 등록
 *     _ArrivalArb(): 출력 VC 소유권 중재, 크레딧 처리, tTransportEvent 생성
 *
 *   [Phase 2 - Transport]
 *     _TransportRequests(): _transport_queue의 이벤트를 입력 중재기에 요청 등록
 *     _TransportArb(): 입력 포트 중재, 플릿을 버퍼에서 꺼내 _crossbar_pipe에 삽입
 *     _crossbar_pipe->Advance(): 크로스바 전파 지연 시뮬레이션
 *     _OutputQueuing(): _crossbar_pipe에서 플릿을 꺼내 출력 버퍼에 삽입
 *
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 단일 스레드 시뮬레이션 루프.
 * gpgpusim.config의 `router event` 옵션으로 활성화.
 *
 * 호출 체인 (매 사이클):
 *   intersim2 주 루프 → ReadInputs() → _InternalStep() → WriteOutputs()
 */
class EventRouter : public Router {
  int _vcs;
  /* [한국어] 라우터가 지원하는 VC(Virtual Channel) 총 개수.
   * 설정자: 생성자에서 config.GetInt("num_vcs")로 초기화.
   * 읽는 자: _buf, _active, _output_state 등 per-VC 자료구조 초기화에 사용.
   *         _IncomingFlits()에서 VC 인덱스 범위 접근에 사용.
   * 값 범위: 1 이상의 양의 정수.
   * 동기화: 초기화 후 읽기 전용 — 잠금 불필요. */

  int _vct;
  /* [한국어] VCT(Virtual Cut-Through) 모드 활성 여부 플래그.
   * 0이면 store-and-forward 모드: 모든 플릿이 개별 arrival 이벤트를 생성.
   * 1이면 VCT 모드: head 플릿만 arrival 이벤트를 생성하여 패킷 전체를 대표.
   *   VCT 모드에서는 tail 플릿까지 한 번에 크로스바를 통과하므로
   *   head credit만이 VC 상태 전환을 유발한다.
   * 설정자: 생성자에서 config.GetInt("vct")로 초기화.
   * 읽는 자: _IncomingFlits()에서 arrival 이벤트 생성 조건 결정.
   *          _ArrivalArb()에서 크레딧 처리 로직 분기.
   *          _TransportArb()에서 tail 이후 상태 전환 방식 결정.
   * 동기화: 초기화 후 읽기 전용 — 잠금 불필요. */

  vector<Buffer *> _buf;
  /* [한국어] 입력 포트별 VC 버퍼 배열 (크기: _inputs).
   * _buf[i]는 입력 포트 i의 Buffer 객체로, VC별 플릿 큐와
   * 라우팅 결과(출력 포트, 출력 VC)를 저장한다.
   * 설정자: 생성자에서 new Buffer(...)로 할당; AddFlit(), Route(), SetOutput() 호출로 갱신.
   * 읽는 자: _IncomingFlits()에서 플릿 삽입 및 라우팅.
   *          _TransportArb()에서 플릿 제거 및 출력 VC 조회.
   *          Display()에서 상태 출력.
   * 메모리: 소멸자에서 delete _buf[i] 수행. */

  vector<vector<bool> > _active;
  /* [한국어] 입력 포트별·VC별 활성 상태 플래그 (크기: _inputs × _vcs).
   * _active[i][v]가 true이면 입력 포트 i의 VC v에 현재 패킷이 진행 중임을 의미.
   * false이면 다음 플릿은 head 플릿이어야 한다.
   * 설정자: 생성자에서 false로 초기화.
   *         _IncomingFlits()에서 head 플릿 도착 시 true로 설정.
   *         _TransportArb()에서 tail 플릿 전송 완료 시 false로 복구.
   * 읽는 자: _IncomingFlits()에서 비활성 VC에 non-head 플릿 도착 오류 감지.
   *          _TransportArb()에서 비활성 VC에 grant 수신 오류 감지. */

  tRoutingFunction   _rf;
  /* [한국어] 라우팅 함수 포인터 — 플릿의 목적지를 보고 출력 포트/VC를 결정.
   * 함수 시그니처: void (*tRoutingFunction)(const Router*, const Flit*, int, OutputSet*, bool)
   * 설정자: 생성자에서 gRoutingFunctionMap에서 "routingfunc_topology" 키로 조회.
   * 읽는 자: _IncomingFlits()에서 cur_buf->Route(vc, _rf, ...) 호출 시 전달.
   *          실제 라우팅 결정(출력 포트, 출력 VC)은 이 함수가 수행.
   * 값 범위: gRoutingFunctionMap에 등록된 유효 라우팅 함수 포인터. NULL 불가.
   * 동기화: 초기화 후 읽기 전용 — 잠금 불필요. */

  vector<EventNextVCState *> _output_state;
  /* [한국어] 출력 포트별 VC 상태 추적기 배열 (크기: _outputs).
   * _output_state[o]는 출력 포트 o에 대한 EventNextVCState 객체로,
   * 각 출력 VC의 상태(idle/busy/tail_pending), 크레딧, 프레즌스, 대기 리스트를 관리.
   * 설정자: 생성자에서 new EventNextVCState(...)로 할당.
   * 읽는 자: _ArrivalArb(), _SendTransport(), _ProcessWaiting()이 모두 접근.
   * 메모리: 소멸자에서 delete _output_state[o] 수행. */

  PipelineFIFO<Flit>   *_crossbar_pipe;
  /* [한국어] 크로스바 전파 지연을 모델링하는 파이프라인 FIFO.
   * _TransportArb()에서 Write(f, output)로 플릿을 삽입하면,
   * _crossbar_delay 사이클 후 _OutputQueuing()의 Read(output)에서 꺼낼 수 있다.
   * 설정자: 생성자에서 new PipelineFIFO<Flit>(..., _outputs, _crossbar_delay) 로 할당.
   *         매 사이클 WriteAll(0)으로 초기화, Advance()로 한 사이클 진행.
   * 읽는 자: _OutputQueuing()에서 Read()로 완료된 플릿 추출.
   * 메모리: 소멸자에서 delete _crossbar_pipe 수행. */

  PipelineFIFO<Credit> *_credit_pipe;
  /* [한국어] 크레딧 반환 지연을 모델링하는 파이프라인 FIFO.
   * _TransportArb()에서 Write(c, input)으로 크레딧을 삽입하면,
   * _credit_delay 사이클 후 _OutputQueuing()의 Read(input)에서 꺼낼 수 있다.
   * 설정자: 생성자에서 new PipelineFIFO<Credit>(..., _inputs, _credit_delay)로 할당.
   *         매 사이클 WriteAll(0)으로 초기화, Advance()로 진행.
   * 읽는 자: _OutputQueuing()에서 Read()로 완료된 크레딧 추출.
   * 메모리: 소멸자에서 delete _credit_pipe 수행. */

  vector<queue<Flit *> > _input_buffer;
  /* [한국어] 입력 포트별 플릿 수신 대기 큐 (크기: _inputs).
   * _ReceiveFlits()에서 채널로부터 수신한 플릿을 임시 보관.
   * _IncomingFlits()에서 front()를 꺼내 _buf[input]에 삽입하고 이벤트를 생성.
   * 설정자: _ReceiveFlits()에서 push().
   * 읽는 자: _IncomingFlits()에서 front()/pop() 호출. */

  vector<queue<Flit *> > _output_buffer;
  /* [한국어] 출력 포트별 플릿 전송 대기 큐 (크기: _outputs).
   * _OutputQueuing()에서 crossbar_pipe에서 꺼낸 플릿을 보관.
   * _SendFlits()에서 출력 채널로 전송.
   * 설정자: _OutputQueuing()에서 push().
   * 읽는 자: _SendFlits()에서 front()/pop() 호출. */

  vector<queue<Credit *> > _in_cred_buffer;
  /* [한국어] 입력 포트별 크레딧 전송 대기 큐 (크기: _inputs).
   * _OutputQueuing()에서 credit_pipe에서 꺼낸 크레딧을 보관.
   * _SendCredits()에서 업스트림(입력 채널 방향)으로 전송.
   * 설정자: _OutputQueuing()에서 push().
   * 읽는 자: _SendCredits()에서 front()/pop() 호출. */

  vector<queue<Credit *> > _out_cred_buffer;
  /* [한국어] 출력 포트별 다운스트림 크레딧 수신 대기 큐 (크기: _outputs).
   * _ReceiveCredits()에서 다운스트림 라우터가 반환한 크레딧을 임시 보관.
   * _ArrivalArb()에서 꺼내 _output_state[output]의 크레딧 카운터를 갱신.
   * 설정자: _ReceiveCredits()에서 push().
   * 읽는 자: _ArrivalArb()에서 front()/pop() 호출. */

  /*
   * [한국어]
   * tArrivalEvent - 플릿이 이 라우터에 도착하여 라우팅된 결과를 담는 이벤트 구조체
   *
   * _IncomingFlits()에서 생성되어 _arrival_pipe에 삽입된다.
   * arrival_pipe의 지연(라우팅/디코딩 시간) 후 _ArrivalRequests()에서 꺼내
   * _arrival_queue에 넣고, _ArrivalArb()에서 출력 VC 소유권을 결정하는 데 사용.
   * 이 이벤트 하나가 "이 플릿은 input에서 output으로, src_vc→dst_vc로 라우팅되었다"는
   * 정보를 완전하게 담고 있어, 이벤트를 처리하는 순간 폴링 없이 라우팅 결과를 안다.
   *
   * 메모리: new로 할당, _ArrivalArb()에서 처리 완료 후 delete.
   */
  struct tArrivalEvent {
    int  input;
    /* [한국어] 이 플릿이 도착한 입력 포트 인덱스.
     * 설정자: _IncomingFlits()에서 루프 변수 input 값으로 설정.
     * 읽는 자: _ArrivalArb()에서 arrival_arbiter에 등록된 입력 식별자로 사용.
     *          _SendTransport()에서 tevt->input으로 복사. */

    int  output;
    /* [한국어] 라우팅 결과로 결정된 출력 포트 인덱스.
     * 설정자: _IncomingFlits()에서 cur_buf->GetOutputPort(vc)로 설정.
     * 읽는 자: _ArrivalRequests()에서 어느 arrival_arbiter에 요청할지 결정.
     *          _ArrivalArb()에서 어느 _output_state를 접근할지 결정. */

    int  src_vc;
    /* [한국어] 이 플릿이 도착한 입력 VC 인덱스 (이 라우터 관점의 소스 VC).
     * 설정자: _IncomingFlits()에서 f->vc로 설정.
     * 읽는 자: _ArrivalArb()에서 SetInputVC(aevt->dst_vc, aevt->src_vc) 호출.
     *          _SendTransport()에서 tevt->src_vc로 복사. */

    int  dst_vc;
    /* [한국어] 라우팅 결과로 결정된 출력 VC 인덱스 (다운스트림 VC).
     * 설정자: _IncomingFlits()에서 cur_buf->GetOutputVC(vc)로 설정.
     * 읽는 자: _ArrivalArb()에서 _output_state[output]->GetState(aevt->dst_vc) 조회.
     *          _SendTransport()에서 tevt->dst_vc로 복사. */

    bool head;
    /* [한국어] 이 플릿이 패킷의 헤드 플릿인지 여부.
     * 설정자: _IncomingFlits()에서 f->head로 설정.
     * 읽는 자: _ArrivalArb()에서 head/body 처리 분기 결정.
     *          head이면 VC 소유권 획득 시도, body이면 기존 소유자 확인. */

    bool tail;
    /* [한국어] 이 플릿이 패킷의 테일 플릿인지 여부 (VCT 모드에서 사용).
     * 설정자: _IncomingFlits()에서 f->tail로 설정.
     * 읽는 자: VCT 모드에서 head+tail 동시 패킷 처리에 사용 가능.
     *          현재 구현에서는 체크되지 않으나 향후 확장용. */

    int  id;    // debug
    /* [한국어] 이 이벤트를 생성한 플릿의 고유 ID (디버그 추적용).
     * 설정자: _IncomingFlits()에서 f->id로 설정.
     * 읽는 자: watch가 true일 때 cout 출력에 사용. */

    bool watch; // debug
    /* [한국어] 이 이벤트에 대한 디버그 추적 활성 여부.
     * 설정자: _IncomingFlits()에서 f->watch로 설정.
     * 읽는 자: _ArrivalArb(), _SendTransport() 등에서 cout 출력 여부 결정. */
  };

  PipelineFIFO<tArrivalEvent> *_arrival_pipe;
  /* [한국어] 도착 이벤트 라우팅/디코딩 지연을 모델링하는 파이프라인 FIFO.
   * _IncomingFlits()에서 Write(aevt, input)로 도착 이벤트를 삽입하면,
   * 0 사이클 지연(현재 "FIX THIS EVENTUALLY" 주석 참고) 후
   * _ArrivalRequests()의 Read(input)에서 꺼낼 수 있다.
   * 설정자: 생성자에서 new PipelineFIFO<tArrivalEvent>(..., _inputs, 0)으로 할당.
   *         매 사이클 _IncomingFlits() 진입 시 WriteAll(0)으로 초기화, Advance()로 진행.
   * 읽는 자: _ArrivalRequests()에서 Read() 호출.
   * 메모리: 소멸자에서 delete _arrival_pipe 수행. */

  vector<queue<tArrivalEvent *> > _arrival_queue;
  /* [한국어] 입력 포트별 도착 이벤트 큐 (크기: _inputs).
   * _arrival_pipe에서 나온 이벤트를 보관하며, 아직 중재되지 않은 이벤트를 대기.
   * _ArrivalRequests()에서 push(), 중재 후 _ArrivalArb()에서 front()/pop().
   * 설정자: _ArrivalRequests()에서 push().
   * 읽는 자: _ArrivalRequests()에서 front() (중재 요청 등록).
   *          _ArrivalArb()에서 front()/pop() (처리). */

  vector<PriorityArbiter*> _arrival_arbiter;
  /* [한국어] 출력 포트별 도착 이벤트 중재기 배열 (크기: _outputs).
   * 여러 입력 포트가 같은 출력 포트의 VC를 동시에 요청할 때 중재.
   * _ArrivalRequests()에서 AddRequest(input) 등록,
   * _ArrivalArb()에서 Arbitrate() 후 Match()로 승자 결정.
   * 설정자: 생성자에서 new PriorityArbiter(...)로 할당.
   *         _InternalStep()에서 매 사이클 Clear()로 리셋.
   * 메모리: 소멸자에서 delete _arrival_arbiter[o] 수행. */

  /*
   * [한국어]
   * tTransportEvent - 플릿을 크로스바를 통해 실제 전송할 수 있는 권한을 나타내는 이벤트
   *
   * _ArrivalArb()나 _ProcessWaiting()에서 생성되며,
   * 크레딧이 있고 출력 VC 소유권이 확보된 상황에서 실제 플릿 전송을 지시한다.
   * _TransportRequests()에서 입력 중재기에 등록되고,
   * _TransportArb()에서 입력 포트를 선택해 플릿을 크로스바에 넣는다.
   *
   * tArrivalEvent와의 차이: tArrivalEvent는 라우팅 결과(어느 출력으로)를 담고,
   * tTransportEvent는 실제 전송 권한(크레딧 확보 완료 상태)을 담는다.
   *
   * 메모리: new로 할당, _TransportArb()에서 처리 완료 후 delete.
   */
  struct tTransportEvent {
    int  input;
    /* [한국어] 플릿을 꺼낼 입력 포트 인덱스.
     * 설정자: _ArrivalArb()/_ProcessWaiting()에서 tevt->input으로 설정.
     * 읽는 자: _TransportRequests()에서 어느 transport_arbiter에 요청할지 결정.
     *          _TransportArb()에서 실제 플릿 추출 시 _buf[input] 인덱스로 사용. */

    int  src_vc;
    /* [한국어] 플릿을 꺼낼 입력 VC 인덱스.
     * 설정자: _ArrivalArb()에서 tevt->src_vc = aevt->src_vc로 설정.
     * 읽는 자: _TransportArb()에서 cur_buf->RemoveFlit(vc)의 vc 인자로 사용.
     *          VC 검증: tevt->dst_vc != cur_buf->GetOutputVC(vc) 이면 Error(). */

    int  dst_vc;
    /* [한국어] 플릿이 전송될 목적지 출력 VC 인덱스 (다운스트림 VC).
     * 설정자: _ArrivalArb()에서 tevt->dst_vc = aevt->dst_vc로 설정.
     * 읽는 자: _TransportArb()에서 VC 일관성 검증 및 f->vc 갱신에 사용.
     *          검증: cur_buf->GetOutputVC(vc)와 일치해야 함. */

    int  id;    // debug
    /* [한국어] 이 이벤트를 생성한 플릿의 고유 ID (디버그 추적용). */

    bool watch; // debug
    /* [한국어] 이 이벤트에 대한 디버그 추적 활성 여부. */
  };

  vector<queue<tTransportEvent *> > _transport_queue;
  /* [한국어] 출력 포트별 전송 이벤트 큐 (크기: _outputs).
   * _ArrivalArb()/_ProcessWaiting()에서 생성된 tTransportEvent를 보관.
   * _TransportRequests()에서 front()로 요청 등록, _TransportArb()에서 pop()으로 제거.
   * 설정자: _ArrivalArb()/_ProcessWaiting()에서 push().
   * 읽는 자: _TransportRequests()에서 front() (요청 등록).
   *          _TransportArb()에서 front()/pop() (처리 후 제거). */

  vector<PriorityArbiter*> _transport_arbiter;
  /* [한국어] 입력 포트별 전송 이벤트 중재기 배열 (크기: _inputs).
   * 여러 출력 포트가 같은 입력 포트로부터 플릿을 요청할 때 중재.
   * _TransportRequests()에서 AddRequest(output) 등록,
   * _TransportArb()에서 Arbitrate() 후 Match()로 승자 결정.
   * 설정자: 생성자에서 new PriorityArbiter(...)로 할당.
   *         _InternalStep()에서 매 사이클 Clear()로 리셋.
   * 메모리: 소멸자에서 delete _transport_arbiter[i] 수행. */

  vector<bool> _transport_free;
  /* [한국어] 입력 포트별 전송 중재기 자유 여부 플래그 (크기: _inputs).
   * _transport_free[i]가 true이면 이번 사이클에 새 중재를 수행할 수 있음.
   * VCT 모드에서 tail 플릿까지 연속 전송할 때 false로 유지하여 매번 재중재 방지.
   * 설정자: 생성자에서 true로 초기화.
   *         _TransportArb()에서 VCT 모드 + tail 아닐 때 false 설정.
   *         VCT 모드에서 tail 도달 시 true 복구.
   *         Store-and-forward 모드에서는 매 플릿 후 true로 복구.
   * 읽는 자: _TransportArb()에서 새 중재 수행 여부 결정. */

  vector<int> _transport_match;
  /* [한국어] 입력 포트별 이전 중재 결과 저장 (크기: _inputs).
   * _transport_free[i]가 false일 때 이전 사이클의 중재 결과(출력 포트 인덱스)를 재사용.
   * VCT 모드에서 동일 출력으로 연속 플릿을 전송하는 동안 중재 결과를 유지.
   * 설정자: 생성자에서 -1로 초기화.
   *         _TransportArb()에서 VCT 모드 + tail 아닐 때 output 값으로 설정.
   *         VCT 모드에서 tail 도달 시 -1로 리셋.
   * 읽는 자: _TransportArb()에서 _transport_free가 false일 때 사용. */

  /*
   * [한국어]
   * _ReceiveFlits - 모든 입력 채널에서 플릿을 수신하여 입력 버퍼에 보관
   *
   * @return: 없음
   *
   * ReadInputs()에서 호출. 각 입력 채널(_input_channels[i])에서 Receive()를 호출하고
   * 플릿이 있으면 _input_buffer[i]에 push()한다.
   * 실제 VC 버퍼 삽입과 이벤트 생성은 다음 단계인 _IncomingFlits()가 담당.
   *
   * 호출 체인:
   *   ReadInputs() → [_ReceiveFlits]
   */
  void _ReceiveFlits( );

  /*
   * [한국어]
   * _ReceiveCredits - 모든 출력 채널로부터 다운스트림 크레딧을 수신하여 버퍼에 보관
   *
   * @return: 없음
   *
   * ReadInputs()에서 호출. 각 출력 크레딧 채널(_output_credits[o])에서 Receive()를
   * 호출하고 크레딧이 있으면 _out_cred_buffer[o]에 push()한다.
   * 크레딧 처리(상태 갱신, transport 이벤트 생성)는 _ArrivalArb()가 담당.
   *
   * 호출 체인:
   *   ReadInputs() → [_ReceiveCredits]
   */
  void _ReceiveCredits( );

  /*
   * [한국어]
   * _IncomingFlits - 입력 버퍼의 플릿을 VC 버퍼에 삽입하고 도착 이벤트 생성
   *
   * @return: 없음
   *
   * _InternalStep()의 첫 번째 단계. 각 입력 포트의 _input_buffer에서 플릿을 꺼내
   * _buf[input]의 해당 VC에 AddFlit()으로 삽입한다.
   * head 플릿 도착 시 라우팅 함수(_rf)를 호출하여 출력 포트/VC를 결정하고
   * _active[input][vc] = true로 활성화한다.
   * VCT 모드가 아니거나, VCT 모드에서 head 플릿인 경우 tArrivalEvent를 생성하여
   * _arrival_pipe에 Write()한다 (라우팅 지연 시뮬레이션 시작).
   *
   * 호출 체인:
   *   _InternalStep() → [_IncomingFlits] → _buf[i]->Route()/AddFlit() → _arrival_pipe->Write()
   */
  void _IncomingFlits( );

  /*
   * [한국어]
   * _ArrivalRequests - arrival_pipe에서 완료된 도착 이벤트를 큐에 넣고 중재 요청 등록
   *
   * @input: 처리할 입력 포트 인덱스
   * @return: 없음
   *
   * _arrival_pipe->Read(input)으로 지연 완료된 도착 이벤트를 꺼내 _arrival_queue[input]에
   * push()한다. _arrival_queue가 비어있지 않으면 큐 front의 출력 포트를 확인하여
   * _arrival_arbiter[output]->AddRequest(input)으로 중재 요청을 등록한다.
   *
   * 호출 체인:
   *   _InternalStep() → [_ArrivalRequests(input)] → _arrival_arbiter[output]->AddRequest()
   */
  void _ArrivalRequests( int input );

  /*
   * [한국어]
   * _ArrivalArb - 출력 포트별 도착 이벤트 중재 및 크레딧 처리, 전송 이벤트 생성
   *
   * @output: 중재할 출력 포트 인덱스
   * @return: 없음
   *
   * _InternalStep()에서 모든 출력 포트에 대해 호출되는 핵심 함수.
   * 두 가지 작업을 수행한다:
   *   (1) 다운스트림에서 반환된 크레딧 처리:
   *       _out_cred_buffer[output]에서 크레딧을 꺼내 _output_state[output]의
   *       크레딧 카운터를 +1한다. VCT 모드와 store-and-forward 모드에서 각각
   *       다른 방식으로 VC 상태 전환 및 transport 이벤트 생성을 처리한다.
   *   (2) 도착 이벤트 중재:
   *       _arrival_arbiter[output]->Arbitrate() 후 Match()로 승자 입력을 선택.
   *       head 플릿: 출력 VC가 idle이면 소유권 획득 + _SendTransport() 호출.
   *                 busy이면 tWaiting 구조체 생성 후 PushWaiting().
   *       body 플릿: 소유자 확인 후 _SendTransport() 또는 IncrWaiting() 호출.
   *
   * 호출 체인:
   *   _InternalStep() → [_ArrivalArb(output)]
   *     → _ProcessWaiting() (tail credit 수신 시)
   *     → _SendTransport() (크레딧 있을 때)
   *     → PushWaiting() / IncrWaiting() (크레딧 없거나 VC busy 시)
   */
  void _ArrivalArb( int output );

  /*
   * [한국어]
   * _SendTransport - 크레딧이 있으면 전송 이벤트 생성, 없으면 프레즌스 카운터 증가
   *
   * @input: 플릿이 위치한 입력 포트 인덱스
   * @output: 플릿이 전송될 출력 포트 인덱스
   * @aevt: 처리할 도착 이벤트 (src_vc, dst_vc, id, watch 정보 포함)
   * @return: 없음
   *
   * _output_state[output]->GetCredits(aevt->dst_vc)로 크레딧 가용성을 확인한다.
   * 크레딧 있음: 크레딧을 -1하고 tTransportEvent를 생성하여 _transport_queue[output]에 push().
   * 크레딧 없음: _output_state[output]->SetPresence()로 프레즌스 카운터를 +1.
   *             나중에 크레딧이 반환되면 _ArrivalArb()가 프레즌스를 보고 이벤트 생성.
   *
   * 호출 체인:
   *   _ArrivalArb() → [_SendTransport] → _transport_queue[output].push()
   */
  void _SendTransport( int input, int output, tArrivalEvent *aevt );

  /*
   * [한국어]
   * _ProcessWaiting - tail 크레딧 수신 후 대기 중인 연결로 VC 소유권 이전 처리
   *
   * @output: 처리할 출력 포트 인덱스
   * @out_vc: tail 크레딧이 도착한 출력 VC 인덱스
   * @return: 없음
   *
   * out_vc의 _waiting 리스트를 확인한다.
   *   대기 있음: PopWaiting()으로 다음 소유자(w)를 꺼내 SetState(busy), SetInput(),
   *              SetInputVC()로 소유권을 이전한다. 크레딧이 있으면 즉시 tTransportEvent를
   *              생성하여 _transport_queue에 push(); 없으면 SetPresence()로 w->pres를 기록.
   *   대기 없음: SetState(idle)로 VC를 유휴 상태로 전환.
   *
   * 호출 체인:
   *   _ArrivalArb() → [_ProcessWaiting] → PopWaiting() / SetState() / _transport_queue.push()
   */
  void _ProcessWaiting( int output, int out_vc );

  /*
   * [한국어]
   * _TransportRequests - 전송 이벤트 큐의 front를 입력 중재기에 요청으로 등록
   *
   * @output: 전송 이벤트를 확인할 출력 포트 인덱스
   * @return: 없음
   *
   * _transport_queue[output]이 비어있지 않으면 front()의 입력 포트를 확인하여
   * _transport_arbiter[tevt->input]->AddRequest(output)으로 요청을 등록한다.
   * 실제 플릿 전송은 _TransportArb()가 담당.
   *
   * 호출 체인:
   *   _InternalStep() → [_TransportRequests(output)] → _transport_arbiter[input]->AddRequest()
   */
  void _TransportRequests( int output );

  /*
   * [한국어]
   * _TransportArb - 입력 포트별 전송 이벤트 중재 및 실제 플릿 전송
   *
   * @input: 중재할 입력 포트 인덱스
   * @return: 없음
   *
   * _InternalStep()에서 모든 입력 포트에 대해 호출되는 Phase 2 핵심 함수.
   * _transport_free[input]가 true이면 _transport_arbiter[input]->Arbitrate()와
   * Match()로 출력 포트를 결정한다. false이면 이전 결과(_transport_match)를 재사용(VCT).
   * 매칭된 출력이 있으면:
   *   - _buf[input]에서 해당 VC의 플릿을 RemoveFlit()으로 꺼낸다.
   *   - 크레딧 객체를 생성하여 _credit_pipe->Write(c, input)에 삽입.
   *   - f->hops++, f->vc = 출력VC로 갱신 후 _crossbar_pipe->Write(f, output) 삽입.
   *   - VCT 모드 여부와 tail 여부에 따라 _transport_free/_transport_match/_active 갱신.
   *
   * 호출 체인:
   *   _InternalStep() → [_TransportArb(input)]
   *     → _buf[input]->RemoveFlit() → _credit_pipe->Write() → _crossbar_pipe->Write()
   */
  void _TransportArb( int input );

  /*
   * [한국어]
   * _OutputQueuing - crossbar_pipe와 credit_pipe에서 완료된 항목을 출력 버퍼에 이동
   *
   * @return: 없음
   *
   * _InternalStep()의 마지막 단계. _crossbar_pipe->Read(output)으로 크로스바를 통과한
   * 플릿을 _output_buffer[output]에 push()한다. _credit_pipe->Read(input)으로
   * 지연 완료된 크레딧을 _in_cred_buffer[input]에 push()한다.
   * 실제 채널 전송은 WriteOutputs() 단계의 _SendFlits()/_SendCredits()가 담당.
   *
   * 호출 체인:
   *   _InternalStep() → [_OutputQueuing] → _output_buffer.push() / _in_cred_buffer.push()
   */
  void _OutputQueuing( );

  /*
   * [한국어]
   * _SendFlits - 출력 버퍼의 플릿을 출력 채널로 전송
   *
   * @return: 없음
   *
   * WriteOutputs()에서 호출. 각 출력 포트의 _output_buffer[output]이 비어있지 않으면
   * front()의 플릿을 꺼내 _output_channels[output]->Send(f)로 다운스트림에 전송.
   *
   * 호출 체인:
   *   WriteOutputs() → [_SendFlits] → _output_channels[output]->Send()
   */
  void _SendFlits( );

  /*
   * [한국어]
   * _SendCredits - 크레딧 버퍼의 크레딧을 업스트림으로 전송
   *
   * @return: 없음
   *
   * WriteOutputs()에서 호출. 각 입력 포트의 _in_cred_buffer[input]이 비어있지 않으면
   * front()의 크레딧을 꺼내 _input_credits[input]->Send(c)로 업스트림에 전송.
   *
   * 호출 체인:
   *   WriteOutputs() → [_SendCredits] → _input_credits[input]->Send()
   */
  void _SendCredits( );

  /*
   * [한국어]
   * _InternalStep - 매 사이클 호출되는 라우터 내부 파이프라인 실행
   *
   * @return: 없음 (virtual)
   *
   * Router 기반 클래스의 순수 가상 함수를 오버라이드. 매 사이클 intersim2 주 루프에서
   * ReadInputs() → _InternalStep() → WriteOutputs() 순으로 호출된다.
   * 이 함수가 이벤트 드리븐 라우터의 전체 2단계 파이프라인을 순서대로 실행한다:
   *   1. _IncomingFlits() — 플릿 수신 및 도착 이벤트 생성
   *   2. _arrival_pipe->Advance() — 라우팅 지연 진행
   *   3. arrival_arbiter Clear() — 중재기 리셋
   *   4. _ArrivalRequests() for all inputs — 도착 이벤트 요청 등록
   *   5. _ArrivalArb() for all outputs — 도착 중재 + 크레딧 처리
   *   6. transport_arbiter Clear() — 중재기 리셋
   *   7. crossbar_pipe/credit_pipe WriteAll(0) — 파이프라인 초기화
   *   8. _TransportRequests() for all outputs — 전송 이벤트 요청 등록
   *   9. _TransportArb() for all inputs — 전송 중재 + 플릿 전송
   *   10. crossbar_pipe/credit_pipe Advance() — 파이프라인 진행
   *   11. _OutputQueuing() — 출력 버퍼로 이동
   *
   * 호출 체인:
   *   intersim2 주 루프 → [_InternalStep]
   */
  virtual void _InternalStep( );

public:
  /*
   * [한국어]
   * EventRouter 생성자 - 이벤트 드리븐 라우터 초기화
   *
   * @config: 시뮬레이터 전역 설정 (num_vcs, vct, routing_function, topology 등)
   * @parent: 부모 모듈 (네트워크 또는 시뮬레이터)
   * @name: 이 라우터의 이름 문자열 (디버그 출력에 사용)
   * @id: 라우터 고유 ID
   * @inputs: 입력 포트 수
   * @outputs: 출력 포트 수
   * @return: 없음 (생성자)
   *
   * Router 기반 클래스 초기화 후 다음을 수행한다:
   *   1. config에서 num_vcs, vct, routing_function 읽기
   *   2. 입력별 Buffer 및 _active 배열 할당
   *   3. 출력별 EventNextVCState 할당
   *   4. 출력별 arrival_arbiter, 입력별 transport_arbiter 할당
   *   5. crossbar_pipe, credit_pipe, arrival_pipe 할당
   *   6. 각종 큐 및 _transport_free/_transport_match 초기화
   *
   * 호출 체인:
   *   intersim2 네트워크 초기화 → Router 팩토리 → [EventRouter 생성자]
   */
  EventRouter( const Configuration& config,
	       Module *parent, const string & name, int id,
	       int inputs, int outputs );

  /*
   * [한국어]
   * ~EventRouter - 이벤트 드리븐 라우터 소멸자
   *
   * @return: 없음 (소멸자)
   *
   * 생성자에서 new로 할당한 모든 동적 객체를 해제한다:
   *   - _buf[i] (입력별 버퍼)
   *   - _output_state[o] (출력별 VC 상태)
   *   - _arrival_arbiter[o], _transport_arbiter[i] (중재기)
   *   - _crossbar_pipe, _credit_pipe, _arrival_pipe (파이프라인)
   *
   * 호출 체인:
   *   intersim2 네트워크 종료 → [~EventRouter]
   */
  virtual ~EventRouter( );

  /*
   * [한국어]
   * ReadInputs - 모든 입력 채널에서 플릿과 크레딧 수신
   *
   * @return: 없음 (virtual)
   *
   * Router 기반 클래스 인터페이스 구현. intersim2 주 루프에서 매 사이클 첫 번째로 호출.
   * _ReceiveFlits()와 _ReceiveCredits()를 순서대로 호출하여
   * 채널 데이터를 내부 버퍼(_input_buffer, _out_cred_buffer)에 저장한다.
   *
   * 호출 체인:
   *   intersim2 주 루프 → [ReadInputs] → _ReceiveFlits() + _ReceiveCredits()
   */
  virtual void ReadInputs( );

  /*
   * [한국어]
   * WriteOutputs - 출력 채널로 플릿과 크레딧 전송
   *
   * @return: 없음 (virtual)
   *
   * Router 기반 클래스 인터페이스 구현. intersim2 주 루프에서 매 사이클 마지막으로 호출.
   * _SendFlits()와 _SendCredits()를 순서대로 호출하여
   * 내부 출력 버퍼(_output_buffer, _in_cred_buffer)의 데이터를 채널로 전송.
   *
   * 호출 체인:
   *   intersim2 주 루프 → [WriteOutputs] → _SendFlits() + _SendCredits()
   */
  virtual void WriteOutputs( );

  /*
   * [한국어]
   * GetUsedCredit - 출력 포트의 사용된 크레딧 수 반환 (EventRouter에서는 미구현)
   *
   * @o: 출력 포트 인덱스 (사용 안 함)
   * @return: int — 항상 0 반환
   *
   * Router 기반 클래스의 순수 가상 함수 오버라이드.
   * EventRouter의 크레딧 추적 방식은 EventNextVCState에서 per-VC로 관리되므로
   * IQRouter 방식의 단순 합계는 의미가 없어 0을 반환한다.
   */
  virtual int GetUsedCredit(int o) const {return 0;}

  /*
   * [한국어]
   * GetBufferOccupancy - 입력 포트의 버퍼 점유율 반환 (EventRouter에서는 미구현)
   *
   * @i: 입력 포트 인덱스 (사용 안 함)
   * @return: int — 항상 0 반환
   *
   * Router 기반 클래스의 순수 가상 함수 오버라이드. 미구현 스텁.
   */
  virtual int GetBufferOccupancy(int i) const {return 0;}

#ifdef TRACK_BUFFERS
  /*
   * [한국어]
   * GetUsedCreditForClass - 클래스별 출력 크레딧 사용량 반환 (미구현 스텁)
   *
   * TRACK_BUFFERS 빌드 옵션이 활성화된 경우에만 컴파일된다.
   * EventRouter에서는 미구현으로 항상 0 반환.
   */
  virtual int GetUsedCreditForClass(int output, int cl) const {return 0;}

  /*
   * [한국어]
   * GetBufferOccupancyForClass - 클래스별 입력 버퍼 점유율 반환 (미구현 스텁)
   *
   * TRACK_BUFFERS 빌드 옵션이 활성화된 경우에만 컴파일된다.
   * EventRouter에서는 미구현으로 항상 0 반환.
   */
  virtual int GetBufferOccupancyForClass(int input, int cl) const {return 0;}
#endif

  /*
   * [한국어]
   * UsedCredits - 모든 출력 포트의 사용 크레딧 벡터 반환 (미구현 스텁)
   *
   * @return: vector<int> — 빈 벡터 반환 (EventRouter 미구현)
   *
   * 통계 수집 인터페이스의 일부. EventRouter는 per-VC 크레딧을 EventNextVCState에서
   * 관리하므로 이 인터페이스 형식과 맞지 않아 빈 벡터 반환.
   */
  virtual vector<int> UsedCredits() const { return vector<int>(); }

  /*
   * [한국어]
   * FreeCredits - 모든 출력 포트의 여유 크레딧 벡터 반환 (미구현 스텁)
   *
   * @return: vector<int> — 빈 벡터 반환 (EventRouter 미구현)
   */
  virtual vector<int> FreeCredits() const { return vector<int>(); }

  /*
   * [한국어]
   * MaxCredits - 모든 출력 포트의 최대 크레딧 벡터 반환 (미구현 스텁)
   *
   * @return: vector<int> — 빈 벡터 반환 (EventRouter 미구현)
   */
  virtual vector<int> MaxCredits() const { return vector<int>(); }

  /*
   * [한국어]
   * Display - 모든 입력 버퍼 상태를 출력 스트림에 출력
   *
   * @os: 출력 대상 스트림 (기본값: cout)
   * @return: 없음
   *
   * 디버그 목적으로 각 입력 포트의 _buf[input]->Display(os)를 호출하여
   * VC별 버퍼 내용을 출력한다.
   *
   * 호출 체인:
   *   (디버그/통계 수집 코드) → [Display]
   */
  void Display( ostream & os = cout ) const;
};

#endif
