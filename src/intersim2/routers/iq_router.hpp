// $Id: iq_router.hpp 5263 2012-09-20 23:40:33Z dub $

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
 * [한국어 설명] IQ(Input-Queued) 라우터 클래스 선언 (iq_router.hpp)
 *
 * === 파일의 역할 ===
 * GPGPU-Sim의 NoC(Network-on-Chip) 시뮬레이터(intersim2)에서 사용하는
 * IQRouter 클래스를 선언한다. IQRouter는 GPU 내부 인터커넥트의 핵심 라우팅
 * 엔진으로, 각 입력 포트에 버퍼(Input Queue)를 두고 5단계 파이프라인
 * (RC→VA→SA→ST→LT)으로 flit(플릿)을 처리한다.
 * 이 파일은 라우터의 모든 내부 상태, 파이프라인 단계별 deque, 할당자 포인터,
 * 스위치 홀드 벡터, NOQ(Next-Output Queuing) 룩어헤드 테이블을 선언한다.
 * 실제 구현은 iq_router.cpp에 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPU NoC 시뮬레이션 계층에서 가장 핵심적인 위치를 차지한다.
 * 호출 체인:
 *   gpgpu_sim::cycle() [gpu-sim.cc]
 *     → icnt_push() / icnt_pop() [icnt_wrapper.cc]
 *       → Interconnect::Advance() [intersim2/]
 *         → Network::_Step() [intersim2/networks/]
 *           → IQRouter::ReadInputs() → IQRouter::_InternalStep() → IQRouter::WriteOutputs()
 * 실행 컨텍스트: 호스트 CPU 스레드(유저스페이스), GPU 타이밍 시뮬레이션 루프 내에서
 * 매 사이클 1회 호출된다. GPU 커널/디바이스 코드가 아니라 순수 C++ 시뮬레이션 코드이다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - router.hpp / Router 기반 클래스: 공통 채널 배열(_input_channels, _output_channels)
 *     및 스피드업(speedup), 딜레이 파라미터 상속
 *   - buffer.hpp / Buffer: 입력 포트별 VC(Virtual Channel) FIFO 버퍼 (_buf[i])
 *   - buffer_state.hpp / BufferState: 다운스트림 라우터의 버퍼 점유 추적, 크레딧 기반
 *     흐름 제어 (_next_buf[o])
 *   - allocator.hpp / Allocator: VC 할당자(_vc_allocator), 스위치 할당자(_sw_allocator),
 *     투기적 스위치 할당자(_spec_sw_allocator)
 *   - routefunc.hpp: 라우팅 함수 포인터 타입 tRoutingFunction (_rf)
 *   - switch_monitor.hpp, buffer_monitor.hpp: 전력/성능 모니터링 객체
 * 이 모듈에 의존하는 모듈:
 *   - intersim2 Network 클래스: IQRouter 인스턴스를 생성하고 매 사이클 구동
 *   - icnt_wrapper.cc: intersim2 초기화 시 IQRouter가 포함된 네트워크 구성
 * 데이터 흐름:
 *   수신 flit → _in_queue_flits → _buf[i] (Buffer) →
 *   [RC단계] route_set 계산 → [VA단계] output VC 할당 → [SA단계] 크로스바 연결 할당 →
 *   [ST단계] _crossbar_flits → _output_buffer[o] → [LT단계] output_channels[o]로 전송
 *   크레딧: _output_credits[o] → _proc_credits → _next_buf[o]->ProcessCredit()
 *           _out_queue_credits → _credit_buffer[i] → _input_credits[i]->Send()
 *
 * === 주요 함수/구조체 요약 ===
 * - IQRouter(config, parent, name, id, inputs, outputs): 생성자. 모든 버퍼/할당자/
 *   파이프라인 deque 초기화. gpgpusim.config의 num_vcs, speculative, noq 등 옵션 적용.
 * - ReadInputs(): 매 사이클 _ReceiveFlits()/_ReceiveCredits() 호출, _active 플래그 갱신
 * - _InternalStep(): 5단계 파이프라인의 Evaluate/Update를 순서대로 구동하는 메인 루프
 * - WriteOutputs(): _SendFlits()/_SendCredits()로 이번 사이클에 처리 완료된 flit/크레딧 전송
 * - _SWAllocAddReq(input, vc, output): 스위치 할당자에 요청을 등록하는 헬퍼. RR 우선순위
 *   기반 교체(supersede) 로직 포함
 * - _UpdateNOQ(input, vc, f): 다음 홉의 라우팅을 미리 계산해 la_route_set에 저장(NOQ 모드)
 */

#ifndef _IQ_ROUTER_HPP_
#define _IQ_ROUTER_HPP_

#include <string>  // [한국어] 라우터 이름, 타입 문자열에 사용
#include <deque>   // [한국어] 파이프라인 단계별 대기열(_route_vcs, _vc_alloc_vcs 등) 구현
#include <queue>   // [한국어] 출력/크레딧 버퍼(_output_buffer, _credit_buffer) 구현
#include <set>     // [한국어] OutputSet::sSetElement 집합 순회에 사용
#include <map>     // [한국어] _in_queue_flits(입력 포트→flit), _out_queue_credits(포트→크레딧) 구현

#include "router.hpp"    // [한국어] Router 기반 클래스: 채널 배열, 딜레이 파라미터, Module 상속
#include "routefunc.hpp" // [한국어] tRoutingFunction 함수 포인터 타입 정의

using namespace std;

// [한국어] 전방 선언: 헤더 간 순환 의존을 피하기 위해 포인터로만 참조할 클래스들을 선언
class VC;           // [한국어] 가상 채널 상태 머신 (idle/routing/vc_alloc/active 상태 관리)
class Flit;         // [한국어] 파이프라인을 흐르는 기본 데이터 단위 (head/body/tail 구분)
class Credit;       // [한국어] 다운스트림 라우터가 업스트림으로 보내는 흐름 제어 신호
class Buffer;       // [한국어] 입력 포트별 VC FIFO 버퍼 (_buf[i]의 타입)
class BufferState;  // [한국어] 다운스트림 버퍼 점유 상태 추적 (_next_buf[o]의 타입)
class Allocator;    // [한국어] VC/스위치 할당 알고리즘 기반 클래스 (iSLIP 등)
class SwitchMonitor; // [한국어] 스위치 횡단 통계 수집 (전력 모델링용)
class BufferMonitor; // [한국어] 버퍼 읽기/쓰기 통계 수집 (전력 모델링용)

class IQRouter : public Router {

  // ─────────────────────────────────────────────
  // VC 설정 파라미터
  // ─────────────────────────────────────────────

  int _vcs;
  /* [한국어] 이 라우터가 지원하는 가상 채널(VC) 수.
   * 설정자: 생성자에서 config.GetInt("num_vcs")로 초기화.
   * 읽는 자: _vc_allocator 생성 시 차원(_vcs*_inputs × _vcs*_outputs), 인덱스 계산
   *          (input_and_vc = input*_vcs + vc) 전반에 사용.
   * 값 범위: 1 이상의 정수. NOQ 모드에서는 _vcs >= _outputs 조건 필수.
   * 동기화: 초기화 후 변경되지 않으므로 별도 동기화 불필요. */

  bool _vc_busy_when_full;
  /* [한국어] 다운스트림 VC의 크레딧이 모두 소진(full)되면 그 VC를 "사용 중(busy)"으로
   * 간주하는 정책 플래그.
   * 설정자: config.GetInt("vc_busy_when_full") > 0 이면 true.
   * 읽는 자: _VCAllocEvaluate()에서 IsFullFor() 결과를 STALL_BUFFER_FULL/RESERVED로
   *          매핑할지 여부 결정.
   * 값 범위: true/false.
   * 동기화: 초기화 후 불변. */

  bool _vc_prioritize_empty;
  /* [한국어] 비어 있는 출력 VC에 더 높은 우선순위를 부여하는 정책 플래그.
   * 설정자: config.GetInt("vc_prioritize_empty") > 0 이면 true.
   * 읽는 자: _VCAllocEvaluate()에서 IsEmptyFor()가 false이면 in_priority에
   *          numeric_limits<int>::min()을 더해 우선순위를 최저로 낮춤.
   *          _SWAllocUpdate()의 피기백 VC 할당 경로에서도 동일 적용.
   * 값 범위: true/false.
   * 동기화: 초기화 후 불변. */

  bool _vc_shuffle_requests;
  /* [한국어] VC 할당 요청을 (input*_vcs+vc) 대신 (vc*_inputs+input) 순서로 인덱싱하는
   * 셔플 정책 플래그. 공정성 개선 목적.
   * 설정자: config.GetInt("vc_shuffle_requests") > 0 이면 true.
   * 읽는 자: _VCAllocEvaluate()와 _SWAllocUpdate()에서 input_and_vc 계산 시 분기.
   * 값 범위: true/false.
   * 동기화: 초기화 후 불변. */

  // ─────────────────────────────────────────────
  // 투기적(speculative) SA 파라미터
  // ─────────────────────────────────────────────

  bool _speculative;
  /* [한국어] 투기적 스위치 할당 활성화 플래그.
   * true이면 VC 할당(VA)이 완료되기 전에 스위치 할당(SA)을 동시에 시도한다.
   * VA 결과와 SA 결과가 불일치하면 SA 그랜트를 폐기(misspeculation)한다.
   * 설정자: config.GetInt("speculative") > 0.
   * 읽는 자: _InputQueuing(), _RouteUpdate(), _VCAllocUpdate(), _SWAllocEvaluate(),
   *          _SWAllocUpdate() 전반에 걸쳐 VC::vc_alloc 상태의 SA 입찰 허용 여부 결정.
   * 값 범위: true/false.
   * 동기화: 초기화 후 불변. */

  bool _spec_check_elig;
  /* [한국어] 투기적 SA 요청 시 출력 포트에 적합한 VC가 존재하는지(eligible) 사전 확인
   * 여부 플래그. true이면 적합 VC가 없는 출력 포트에는 투기적 요청을 보내지 않는다.
   * 설정자: config.GetInt("spec_check_elig") > 0.
   * 읽는 자: _SWAllocEvaluate()의 speculative 경로.
   * 값 범위: true/false.
   * 동기화: 초기화 후 불변. */

  bool _spec_check_cred;
  /* [한국어] 투기적 SA 요청 시 출력 VC의 크레딧 가용성(credit)까지 확인하는 플래그.
   * _spec_check_elig가 true일 때만 의미 있다. true이면 크레딧이 없는 VC로의 투기적
   * 요청을 억제한다.
   * 설정자: config.GetInt("spec_check_cred") > 0.
   * 읽는 자: _SWAllocEvaluate()의 speculative 경로.
   * 값 범위: true/false.
   * 동기화: 초기화 후 불변. */

  bool _spec_mask_by_reqs;
  /* [한국어] 투기적 SA 그랜트를 비투기적 요청이 있는 출력 포트에서 차단하는 마스킹 정책.
   * true이면 _sw_allocator에 요청이 있는 expanded_output의 투기적 그랜트를 폐기.
   * false이면 _sw_allocator가 이미 그랜트를 발행한 경우에만 폐기.
   * 설정자: config.GetInt("spec_mask_by_reqs") > 0.
   * 읽는 자: _SWAllocEvaluate()의 _spec_sw_allocator 그랜트 확인 경로.
   * 값 범위: true/false.
   * 동기화: 초기화 후 불변. */

  bool _active;
  /* [한국어] 이 사이클에 처리할 작업(flit/크레딧/파이프라인 항목)이 있는지 나타내는 플래그.
   * false이면 _InternalStep()이 조기 반환하여 불필요한 연산을 스킵한다.
   * 설정자: ReadInputs()에서 have_flits || have_credits이면 true;
   *          _InternalStep() 말미에 파이프라인 deque가 모두 비었으면 false.
   * 읽는 자: _InternalStep() 최상단 가드 조건.
   * 값 범위: true/false.
   * 동기화: 단일 시뮬레이션 스레드에서만 접근하므로 별도 동기화 불필요. */

  // ─────────────────────────────────────────────
  // 파이프라인 단계 딜레이 (사이클 단위)
  // ─────────────────────────────────────────────

  int _routing_delay;
  /* [한국어] RC(Route Computation) 단계에 걸리는 사이클 수.
   * 0이면 lookahead 라우팅을 사용한다(flit이 이전 라우터에서 la_route_set을 계산해 도착).
   * NOQ 모드에서는 반드시 0이어야 한다.
   * 설정자: config.GetInt("routing_delay").
   * 읽는 자: _RouteEvaluate()에서 iter->first = GetSimTime() + _routing_delay - 1 계산.
   * 값 범위: 0 이상의 정수.
   * 동기화: 초기화 후 불변. */

  int _vc_alloc_delay;
  /* [한국어] VA(VC Allocation) 단계에 걸리는 사이클 수. 0이 될 수 없다(생성자에서 Error).
   * _vc_alloc_delay > 1이면 _VCAllocEvaluate()가 추가 유효성 검사 루프를 실행한다.
   * 설정자: config.GetInt("vc_alloc_delay").
   * 읽는 자: _VCAllocEvaluate()에서 iter->first = GetSimTime() + _vc_alloc_delay - 1.
   *          AddOutputChannel()에서 min_latency 계산에 포함.
   * 값 범위: 1 이상의 정수.
   * 동기화: 초기화 후 불변. */

  int _sw_alloc_delay;
  /* [한국어] SA(Switch Allocation) 단계에 걸리는 사이클 수. 0이 될 수 없다(생성자에서 Error).
   * 설정자: config.GetInt("sw_alloc_delay").
   * 읽는 자: _SWAllocEvaluate()에서 iter->first = GetSimTime() + _sw_alloc_delay - 1.
   *          AddOutputChannel()의 min_latency 계산에 포함.
   * 값 범위: 1 이상의 정수.
   * 동기화: 초기화 후 불변. */

  // ─────────────────────────────────────────────
  // 파이프라인 단계 간 데이터 운반 자료구조
  // 모든 deque 항목의 pair<int, ...>에서:
  //   first < 0  : 아직 스케줄되지 않은 신규 항목 (이번 사이클 Evaluate에서 처리 대상)
  //   first >= 0 : 처리 완료, first == GetSimTime() 이 되는 사이클에 Update에서 확정
  // ─────────────────────────────────────────────

  map<int, Flit *> _in_queue_flits;
  /* [한국어] 이번 사이클에 입력 채널로부터 수신된 flit의 임시 스테이징 맵.
   * 키: 입력 포트 번호(int), 값: flit 포인터.
   * 설정자: _ReceiveFlits()에서 _input_channels[input]->Receive()가 flit을 반환하면 삽입.
   * 읽는 자: _InputQueuing()에서 전체 맵을 순회하여 각 flit을 _buf[input]에 AddFlit하고
   *          파이프라인 deque에 등록. 처리 후 clear().
   * 값 범위: 포트당 최대 1개(같은 사이클에 같은 입력 포트에서 flit 2개 도착 불가).
   * 동기화: 단일 스레드, ReadInputs()→_InternalStep() 순서로만 접근. */

  deque<pair<int, pair<Credit *, int> > > _proc_credits;
  /* [한국어] 다운스트림으로부터 수신하여 _credit_delay 사이클 후 처리될 크레딧 대기열.
   * 항목 구조: pair<delivery_time, pair<Credit*, output_port>>.
   *   delivery_time: GetSimTime() + _credit_delay (수신 시 계산)
   *   Credit*: 처리 후 Free() 호출
   *   output_port: 크레딧이 해제하는 출력 포트 인덱스
   * 설정자: _ReceiveCredits()에서 output 크레딧 채널에서 수신 시 push_back.
   * 읽는 자: _InputQueuing()에서 front()의 time <= GetSimTime()이면 ProcessCredit() 호출.
   * 동기화: 단일 스레드. */

  deque<pair<int, pair<int, int> > > _route_vcs;
  /* [한국어] RC(Route Computation) 파이프라인 단계 대기열.
   * 항목 구조: pair<completion_time, pair<input_port, vc>>.
   *   completion_time < 0: 이번 사이클 _RouteEvaluate()가 처리해 시간을 설정해야 할 항목
   *   completion_time >= 0: 이미 RC 진행 중, GetSimTime()==time이 되면 _RouteUpdate()가 확정
   * 설정자: _InputQueuing()에서 _routing_delay > 0이고 VC가 idle→routing 전환 시 push_back.
   *          _SWHoldUpdate(), _SWAllocUpdate()에서 tail flit 이후 다음 패킷 head가 routing
   *          상태로 전환될 때 push_back.
   * 읽는 자: _RouteEvaluate(), _RouteUpdate().
   * 동기화: 단일 스레드. */

  deque<pair<int, pair<pair<int, int>, int> > > _vc_alloc_vcs;
  /* [한국어] VA(VC Allocation) 파이프라인 단계 대기열.
   * 항목 구조: pair<completion_time, pair<pair<input, vc>, result>>.
   *   result == -1: 아직 할당 결과 미결정(Evaluate 처리 전 또는 할당 시도 후 대기)
   *   result >= 0: output_and_vc (output_port * _vcs + out_vc) 할당 성공
   *   result < -1: STALL_BUFFER_BUSY / STALL_BUFFER_CONFLICT 등 실패 코드
   * 설정자: _InputQueuing(), _RouteUpdate(), _VCAllocUpdate()(재시도 시) push_back.
   * 읽는 자: _VCAllocEvaluate(), _VCAllocUpdate().
   * 동기화: 단일 스레드. */

  deque<pair<int, pair<pair<int, int>, int> > > _sw_hold_vcs;
  /* [한국어] "스위치 홀드" 경로의 대기열.
   * 패킷 내 연속 flit이 이미 할당받은 크로스바 경로를 재사용하는 fast-path.
   * _hold_switch_for_packet==true이고 스위치 홀드 상태인 VC의 비-head flit이 여기 들어온다.
   * 항목 구조: pair<completion_time, pair<pair<input, vc>, result>>.
   *   result == -1: 평가 전
   *   result >= 0: expanded_output (크로스바 경로 재사용 가능)
   *   result < -1: STALL_BUFFER_FULL/RESERVED (크레딧 부족으로 재사용 불가)
   * 설정자: _InputQueuing(), _SWHoldUpdate()(비-tail flit 처리 후 재등록) push_back.
   * 읽는 자: _SWHoldEvaluate(), _SWHoldUpdate().
   * 동기화: 단일 스레드. */

  deque<pair<int, pair<pair<int, int>, int> > > _sw_alloc_vcs;
  /* [한국어] SA(Switch Allocation) 파이프라인 단계 대기열.
   * 항목 구조: pair<completion_time, pair<pair<input, vc>, result>>.
   *   result == -1: 이번 사이클 _SWAllocEvaluate()가 처리해야 할 신규 항목
   *   result >= 0: expanded_output 할당 성공
   *   result == -1 (time >= 0): 투기적 misspeculation으로 VA 경로에서 카운트되는 실패
   *   result < -1: STALL_CROSSBAR_CONFLICT / STALL_BUFFER_* 실패 코드
   * 설정자: _InputQueuing(), _RouteUpdate(), _VCAllocUpdate(), _SWAllocUpdate()(재시도) push_back.
   * 읽는 자: _SWAllocEvaluate(), _SWAllocUpdate().
   * 동기화: 단일 스레드. */

  deque<pair<int, pair<Flit *, pair<int, int> > > > _crossbar_flits;
  /* [한국어] ST(Switch Traversal) 파이프라인 단계 대기열.
   * 크로스바 횡단 중인 flit을 _crossbar_delay 사이클 동안 보관한다.
   * 항목 구조: pair<completion_time, pair<Flit*, pair<expanded_input, expanded_output>>>.
   * 설정자: _SWHoldUpdate(), _SWAllocUpdate()에서 크로스바 진입 확정 시 push_back.
   * 읽는 자: _SwitchEvaluate(), _SwitchUpdate().
   * 동기화: 단일 스레드. */

  map<int, Credit *> _out_queue_credits;
  /* [한국어] 이번 사이클에 업스트림 라우터로 돌려보낼 크레딧의 임시 맵.
   * 키: 입력 포트 번호, 값: Credit 객체(여러 VC를 하나의 Credit에 묶어 전송 가능).
   * 설정자: _SWHoldUpdate()/_SWAllocUpdate()에서 flit을 버퍼에서 꺼낼 때
   *          input 포트에 대한 Credit을 생성/갱신하고 vc 번호를 vc 집합에 삽입.
   * 읽는 자: _OutputQueuing()에서 _credit_buffer[input]으로 이동. 이후 clear().
   * 동기화: 단일 스레드. */

  // ─────────────────────────────────────────────
  // 입력/출력 버퍼
  // ─────────────────────────────────────────────

  vector<Buffer *> _buf;
  /* [한국어] 각 입력 포트에 해당하는 VC 버퍼 배열 (크기: _inputs).
   * _buf[i]는 i번째 입력 포트의 모든 VC FIFO를 포함하는 Buffer 객체.
   * Buffer 내부는 _vcs개의 VC를 가지며, 각 VC는 flit FIFO + 상태(idle/routing/vc_alloc/active).
   * 설정자: 생성자에서 new Buffer(config, _outputs, ...) 로 초기화.
   * 읽는 자: 파이프라인 모든 단계에서 상태 확인 및 flit 추가/제거.
   *          소멸자에서 delete.
   * 동기화: 단일 스레드. */

  vector<BufferState *> _next_buf;
  /* [한국어] 각 출력 포트에 연결된 다운스트림 라우터의 버퍼 점유 상태를 추적하는 배열 (크기: _outputs).
   * _next_buf[o]는 출력 포트 o를 통해 연결된 다운스트림 라우터의 VC별 크레딧 카운터를 유지한다.
   * 이 정보를 기반으로 IsFullFor(vc), IsAvailableFor(vc), TakeBuffer(vc, owner) 등을 호출하여
   * 흐름 제어를 구현한다.
   * 설정자: 생성자에서 new BufferState(config, ...) 초기화. _InputQueuing()에서 ProcessCredit()으로 갱신.
   *          _SWHoldUpdate()/_SWAllocUpdate()에서 SendingFlit()으로 크레딧 차감.
   *          _VCAllocUpdate()에서 TakeBuffer()로 VC 소유 등록.
   * 읽는 자: VA/SA/SWHold Evaluate 및 Update 단계에서 full/available 확인.
   * 동기화: 단일 스레드. */

  // ─────────────────────────────────────────────
  // 할당자 (Allocator)
  // ─────────────────────────────────────────────

  Allocator *_vc_allocator;
  /* [한국어] VC 할당 알고리즘 구현 객체 (iSLIP 등).
   * 차원: (_vcs*_inputs) 입력 측 × (_vcs*_outputs) 출력 측.
   * NULL이면 "piggyback" 모드 — VC 할당을 SA와 결합해 수행.
   * 설정자: 생성자에서 Allocator::NewAllocator()로 생성.
   * 읽는 자: _VCAllocEvaluate()에서 AddRequest()/Allocate()/OutputAssigned().
   *           _SWAllocUpdate()의 piggyback 경로에서 OutputAssigned()로 VA 결과 확인.
   * 동기화: 단일 스레드. 매 사이클 _InternalStep() 시작 시 Clear() 후 재사용. */

  Allocator *_sw_allocator;
  /* [한국어] 스위치(크로스바) 할당 알고리즘 구현 객체.
   * 차원: (_inputs*_input_speedup) 입력 측 × (_outputs*_output_speedup) 출력 측.
   * 설정자: 생성자에서 Allocator::NewAllocator()로 생성.
   * 읽는 자: _SWAllocAddReq()에서 AddRequest()/RemoveRequest().
   *          _SWAllocEvaluate()에서 Allocate()/OutputAssigned().
   * 동기화: 단일 스레드. 매 사이클 _InternalStep()에서 Clear() 후 재사용. */

  Allocator *_spec_sw_allocator;
  /* [한국어] 투기적 SA 전용 별도 할당자 (speculative=true이고 spec_sw_allocator != "prio" 일 때).
   * NULL이면 _sw_allocator에서 낮은 우선순위로 투기적 요청을 처리한다.
   * 설정자: 생성자에서 _speculative && spec_sw_alloc_type != "prio" 조건으로 생성.
   * 읽는 자: _SWAllocAddReq(), _SWAllocEvaluate().
   * 동기화: 단일 스레드. 매 사이클 _InternalStep()에서 Clear() 후 재사용. */

  vector<int> _vc_rr_offset;
  /* [한국어] piggyback VC 할당 시 라운드로빈 시작 오프셋 배열.
   * 크기: _outputs * _classes. 각 (출력 포트, 클래스) 쌍마다 다음 할당 시작 VC를 기억.
   * 설정자: 생성자에서 -1로 초기화. _SWAllocUpdate()의 piggyback 경로에서 match_vc+1로 갱신.
   * 읽는 자: _SWAllocUpdate()의 piggyback VC 선택 RoundRobinArbiter::Supersedes() 호출 시.
   * 동기화: 단일 스레드. */

  vector<int> _sw_rr_offset;
  /* [한국어] 스위치 할당 라운드로빈 시작 오프셋 배열.
   * 크기: _inputs * _input_speedup. 각 expanded_input별로 마지막으로 허가된 VC 다음 VC를 기억.
   * 설정자: 생성자에서 i % _input_speedup으로 초기화. 할당 성공 시 (vc + _input_speedup) % _vcs로 갱신.
   * 읽는 자: _SWAllocAddReq()의 RoundRobinArbiter::Supersedes() 호출 시 현재 오프셋 전달.
   * 동기화: 단일 스레드. */

  // ─────────────────────────────────────────────
  // 라우팅 함수 포인터
  // ─────────────────────────────────────────────

  tRoutingFunction   _rf;
  /* [한국어] 이 라우터가 사용하는 라우팅 함수 포인터 (예: xy_iq, dim_order_mesh).
   * tRoutingFunction = void(*)(Router*, Flit*, int in_ch, OutputSet*, bool lookahead).
   * 설정자: 생성자에서 gRoutingFunctionMap[routing_function + "_" + topology]로 조회.
   * 읽는 자: _RouteUpdate()에서 cur_buf->Route(vc, _rf, this, f, input) 내부 호출.
   *          _SWHoldUpdate()/_SWAllocUpdate()에서 lookahead la_route_set 갱신 시 _rf(router, f, ...).
   *          _UpdateNOQ()에서 다음 홉 라우팅 계산 시 _rf(router, f, ...).
   * 동기화: 초기화 후 불변. */

  // ─────────────────────────────────────────────
  // 출력 버퍼 (ST 이후 LT 전 대기 큐)
  // ─────────────────────────────────────────────

  int _output_buffer_size;
  /* [한국어] 출력 버퍼 최대 flit 수 (-1이면 무제한).
   * 설정자: config.GetInt("output_buffer_size").
   * 읽는 자: _SWHoldUpdate(), _SWAllocUpdate()에서 출력 버퍼 용량 초과 시 스위치 통과 차단.
   *          _SwitchUpdate()의 assert에서 유효 범위 검증.
   * 동기화: 초기화 후 불변. */

  vector<queue<Flit *> > _output_buffer;
  /* [한국어] 크로스바를 통과한 flit이 output 채널로 나가기 전 대기하는 출력 버퍼 (크기: _outputs).
   * _output_buffer[o]는 o번 출력 포트의 FIFO 큐.
   * 설정자: _SwitchUpdate()에서 push().
   * 읽는 자: _SendFlits()에서 front()/pop()으로 꺼내 _output_channels[o]->Send().
   *          _SWHoldUpdate()/_SWAllocUpdate()에서 size()를 _output_buffer_size와 비교.
   * 동기화: 단일 스레드. */

  vector<queue<Credit *> > _credit_buffer;
  /* [한국어] 업스트림으로 돌려보낼 크레딧이 대기하는 입력 포트별 버퍼 (크기: _inputs).
   * 설정자: _OutputQueuing()에서 _out_queue_credits 맵을 이 버퍼로 이동.
   * 읽는 자: _SendCredits()에서 front()/pop()으로 꺼내 _input_credits[i]->Send().
   * 동기화: 단일 스레드. */

  // ─────────────────────────────────────────────
  // 스위치 홀드 (hold_switch_for_packet)
  // ─────────────────────────────────────────────

  bool _hold_switch_for_packet;
  /* [한국어] 패킷 내 모든 flit이 같은 크로스바 경로를 유지하는 "스위치 홀드" 정책 플래그.
   * true이면 head flit이 SA를 통과한 후 tail flit까지 동일 크로스바 경로를 예약한다.
   * 설정자: config.GetInt("hold_switch_for_packet") > 0.
   * 읽는 자: _InternalStep()에서 _SWHoldEvaluate/_SWHoldUpdate 호출 여부 결정.
   *          _SWAllocUpdate()에서 비-tail flit 처리 후 _sw_hold_vcs 또는 _sw_alloc_vcs 선택.
   * 값 범위: true/false.
   * 동기화: 초기화 후 불변. */

  vector<int> _switch_hold_in;
  /* [한국어] expanded_input → 현재 홀드 중인 expanded_output 매핑 (-1이면 홀드 없음).
   * 크기: _inputs * _input_speedup.
   * expanded_input = input * _input_speedup + vc % _input_speedup.
   * 설정자: _SWAllocUpdate()에서 hold 설정 시 expanded_output 기록.
   *          _SWHoldUpdate(), _SWAllocUpdate()에서 tail flit 이후 -1로 리셋.
   * 읽는 자: _SWAllocAddReq()에서 홀드 상태인 expanded_input에는 SA 요청 불가 조건.
   *          _SWHoldEvaluate()에서 홀드 경로 확인.
   *          _SWAllocEvaluate()의 후속 검증 루프에서 홀드 충돌 감지.
   * 동기화: 단일 스레드. */

  vector<int> _switch_hold_out;
  /* [한국어] expanded_output → 현재 홀드 중인 expanded_input 매핑 (-1이면 홀드 없음).
   * 크기: _outputs * _output_speedup.
   * 설정자: _SWAllocUpdate()에서 hold 설정 시 expanded_input 기록.
   *          _SWHoldUpdate(), _SWAllocUpdate()에서 tail flit 이후 -1로 리셋.
   * 읽는 자: _SWAllocAddReq()에서 홀드 상태인 expanded_output에 SA 요청 불가 조건.
   *          _SWAllocEvaluate()의 후속 검증 루프에서 홀드 충돌 감지.
   * 동기화: 단일 스레드. */

  vector<int> _switch_hold_vc;
  /* [한국어] expanded_input → 현재 스위치를 홀드하고 있는 VC 번호 (-1이면 없음).
   * 크기: _inputs * _input_speedup.
   * 설정자: _SWAllocUpdate()에서 홀드 설정 시 vc 기록.
   *          _SWHoldUpdate(), _SWAllocUpdate()에서 tail flit 이후 -1로 리셋.
   * 읽는 자: _InputQueuing()에서 active 상태 VC가 _sw_hold_vcs vs _sw_alloc_vcs 경로 선택.
   *          _SWHoldEvaluate()에서 expanded_input의 홀드 VC가 vc와 일치하는지 검증.
   *          _SWAllocEvaluate()에서 SA 요청 전 홀드 VC가 아님을 확인.
   * 동기화: 단일 스레드. */

  // ─────────────────────────────────────────────
  // NOQ (Next-Output Queuing) 룩어헤드 라우팅
  // ─────────────────────────────────────────────

  bool _noq;
  /* [한국어] NOQ(Next-Output Queuing) 모드 활성화 플래그.
   * true이면 flit이 현재 라우터를 통과할 때 다음 홉 라우팅 결과를 미리 계산해
   * flit의 la_route_set에 저장한다. 이를 통해 다음 라우터에서 RC 지연 없이 VA를 시작할 수 있다.
   * NOQ는 _routing_delay == 0이고 _vcs >= _outputs인 경우에만 활성화 가능.
   * 설정자: config.GetInt("noq") > 0.
   * 읽는 자: _InputQueuing(), _VCAllocEvaluate(), _SWAllocEvaluate(), _SWHoldUpdate(),
   *          _SWAllocUpdate(), _UpdateNOQ() 전반에서 NOQ 경로 분기.
   * 동기화: 초기화 후 불변. */

  vector<vector<int> > _noq_next_output_port;
  /* [한국어] NOQ 모드에서 (input, vc) 쌍에 대한 다음 홉 출력 포트 미리 계산 결과.
   * 크기: _inputs × _vcs. -1이면 미계산 또는 목적지에 도달(다음 라우터 없음).
   * 설정자: _UpdateNOQ()에서 _rf() 호출 결과로 next_output_port 기록.
   *          _SWHoldUpdate()/_SWAllocUpdate()에서 la_route_set에 반영 후 -1로 리셋.
   * 읽는 자: _VCAllocEvaluate(), _SWAllocEvaluate(), _SWAllocUpdate()에서 NOQ VC 범위 조회.
   * 동기화: 단일 스레드. */

  vector<vector<int> > _noq_next_vc_start;
  /* [한국어] NOQ 모드에서 (input, vc) 쌍에 대한 다음 홉 VC 범위의 시작 값.
   * 설정자/읽는 자: _noq_next_output_port와 동일.
   * 값 범위: 0 이상 _vcs 미만, 또는 -1.
   * 동기화: 단일 스레드. */

  vector<vector<int> > _noq_next_vc_end;
  /* [한국어] NOQ 모드에서 (input, vc) 쌍에 대한 다음 홉 VC 범위의 끝 값.
   * 설정자/읽는 자: _noq_next_output_port와 동일.
   * 값 범위: 0 이상 _vcs 미만 (_noq_next_vc_start 이상), 또는 -1.
   * 동기화: 단일 스레드. */

#ifdef TRACK_FLOWS
  vector<vector<queue<int> > > _outstanding_classes;
  /* [한국어] 출력 포트별·VC별 미처리 크레딧의 클래스 큐 (TRACK_FLOWS 조건부 컴파일).
   * 크기: _outputs × _vcs. 각 큐는 크레딧이 돌아올 때까지 해당 flit의 클래스를 FIFO로 보관.
   * 설정자: _SWHoldUpdate()/_SWAllocUpdate()에서 flit 전송 시 f->cl push.
   * 읽는 자: _InputQueuing()에서 ProcessCredit() 전 cl pop으로 _outstanding_credits 갱신.
   * 동기화: 단일 스레드. */
#endif

  // ─────────────────────────────────────────────
  // 파이프라인 단계별 private 메서드 (구현은 iq_router.cpp)
  // ─────────────────────────────────────────────

  bool _ReceiveFlits( );    // [한국어] 입력 채널에서 flit을 수신해 _in_queue_flits에 스테이징
  bool _ReceiveCredits( );  // [한국어] 출력 크레딧 채널에서 크레딧을 수신해 _proc_credits에 추가

  virtual void _InternalStep( ); // [한국어] 5단계 파이프라인 Evaluate/Update 시퀀스를 구동하는 메인 루프

  bool _SWAllocAddReq(int input, int vc, int output); // [한국어] SA 할당자에 요청 등록 헬퍼 (RR 기반 교체 로직 포함)

  void _InputQueuing( );  // [한국어] _in_queue_flits → _buf 이동, 크레딧 처리, 파이프라인 deque 진입 결정

  void _RouteEvaluate( ); // [한국어] RC 단계: 새 항목에 완료 시각 설정 (실제 라우팅은 Update에서)
  void _VCAllocEvaluate( ); // [한국어] VA 단계: VC 할당자에 요청 등록, Allocate() 호출, 결과 기록
  void _SWHoldEvaluate( ); // [한국어] 스위치 홀드 단계: 크레딧 잔여 확인, 재사용 가능 여부 result 기록
  void _SWAllocEvaluate( ); // [한국어] SA 단계: SA 요청 등록, Allocate() 호출, 그랜트 검증 후 result 기록
  void _SwitchEvaluate( ); // [한국어] ST 단계: 크로스바 진입 flit에 완료 시각 설정

  void _RouteUpdate( );   // [한국어] RC 완료 항목 처리: _rf() 호출, VC 상태 routing→vc_alloc 전환
  void _VCAllocUpdate( ); // [한국어] VA 완료 항목 처리: TakeBuffer(), SetOutput(), VC 상태 active 전환
  void _SWHoldUpdate( );  // [한국어] 스위치 홀드 완료 항목 처리: flit 버퍼 제거, 크로스바 진입, 홀드 갱신
  void _SWAllocUpdate( ); // [한국어] SA 완료 항목 처리: flit 버퍼 제거, 크로스바 진입, 홀드 설정, 재시도 등록
  void _SwitchUpdate( );  // [한국어] ST 완료 항목 처리: flit을 _output_buffer로 이동

  void _OutputQueuing( ); // [한국어] _out_queue_credits → _credit_buffer로 이동

  void _SendFlits( );     // [한국어] _output_buffer의 flit을 output 채널로 전송 (LT 단계)
  void _SendCredits( );   // [한국어] _credit_buffer의 크레딧을 input 크레딧 채널로 전송

  void _UpdateNOQ(int input, int vc, Flit const * f); // [한국어] NOQ 룩어헤드: 다음 홉 라우팅 결과를 _noq_next_* 배열에 저장

  // ─────────────────────────────────────────────
  // 전력 모델링용 모니터 (Router Power Modelling)
  // ─────────────────────────────────────────────

  SwitchMonitor * _switchMonitor ;
  /* [한국어] 크로스바 횡단 이벤트를 수집하는 전력 모델링 모니터.
   * 설정자: 생성자에서 new SwitchMonitor(inputs, outputs, _classes).
   *          _SwitchUpdate()에서 traversal(input, output, f) 호출로 횡단 기록.
   *          소멸자에서 delete.
   * 읽는 자: GetSwitchMonitor()를 통해 외부(전력 모델 등)에서 통계 접근.
   *          소멸자에서 gPrintActivity 시 *_switchMonitor 출력.
   * 동기화: 단일 스레드. */

  BufferMonitor * _bufferMonitor ;
  /* [한국어] 버퍼 읽기/쓰기 이벤트를 수집하는 전력 모델링 모니터.
   * 설정자: 생성자에서 new BufferMonitor(inputs, _classes).
   *          _InputQueuing()에서 write(input, f).
   *          _SWHoldUpdate()/_SWAllocUpdate()에서 read(input, f).
   *          소멸자에서 delete.
   * 읽는 자: GetBufferMonitor()를 통해 외부에서 통계 접근.
   *          소멸자에서 gPrintActivity 시 *_bufferMonitor 출력.
   * 동기화: 단일 스레드. */

public:

  /*
   * [한국어]
   * IQRouter - IQ 라우터 생성자
   *
   * @config: 시뮬레이터 설정 객체. num_vcs, speculative, noq, routing_delay 등 모든
   *          라우터 파라미터를 이 객체에서 읽는다.
   * @parent: 부모 Module (네트워크 계층 구조에서 이 라우터의 상위 객체).
   * @name: 디버그 출력에 사용될 라우터 이름 문자열.
   * @id: 네트워크 내 라우터의 고유 정수 ID.
   * @inputs: 입력 포트 수 (= _inputs).
   * @outputs: 출력 포트 수 (= _outputs).
   * @return: 없음 (생성자).
   *
   * 라우터에 필요한 모든 내부 자료구조를 초기화한다:
   *   1) VC 정책 파라미터(_vc_busy_when_full 등) 설정
   *   2) 라우팅 함수 포인터 gRoutingFunctionMap에서 조회
   *   3) _buf[i]: 입력별 Buffer 객체 생성 (VC FIFO + 상태 머신)
   *   4) _next_buf[j]: 출력별 BufferState 객체 생성 (크레딧 추적)
   *   5) _vc_allocator, _sw_allocator, _spec_sw_allocator 생성
   *   6) _switch_hold_* 벡터 -1로 초기화 (홀드 없음)
   *   7) NOQ 배열 초기화
   *   8) 모니터 객체 생성
   *
   * 호출 체인:
   *   Network::Network() [intersim2/networks/*.cpp]
   *     → IQRouter::IQRouter()
   */
  IQRouter( Configuration const & config,
	    Module *parent, string const & name, int id,
	    int inputs, int outputs );

  /*
   * [한국어]
   * ~IQRouter - 소멸자
   *
   * gPrintActivity가 true이면 bufferMonitor, switchMonitor 통계를 출력한다.
   * _buf[i], _next_buf[j], 할당자, 모니터 객체를 delete.
   */
  virtual ~IQRouter( );

  /*
   * [한국어]
   * AddOutputChannel - 출력 채널 및 역방향(크레딧) 채널 등록
   *
   * @channel: 출력 flit 채널 포인터.
   * @backchannel: 역방향 크레딧 채널 포인터.
   *
   * 기반 클래스 Router::AddOutputChannel()을 호출하기 전에 해당 출력 포트의
   * _next_buf의 MinLatency를 계산·설정한다.
   * MinLatency = 1(파이프 오버헤드) + crossbar_delay + channel latency +
   *              routing_delay + alloc_delay + backchannel latency + credit_delay.
   * 이 값은 크레딧이 너무 일찍 발행되어 흐름 제어가 깨지는 것을 방지하는 하한선이다.
   */
  virtual void AddOutputChannel(FlitChannel * channel, CreditChannel * backchannel);

  /*
   * [한국어]
   * ReadInputs - 매 사이클 입력 채널 읽기 (시뮬레이션 루프의 첫 번째 단계)
   *
   * _ReceiveFlits()와 _ReceiveCredits()를 호출하여 이번 사이클 도착한
   * flit/크레딧을 스테이징 자료구조에 저장한다. 수신된 데이터가 있으면 _active를 true로 설정.
   */
  virtual void ReadInputs( );

  /*
   * [한국어]
   * WriteOutputs - 매 사이클 출력 채널 쓰기 (시뮬레이션 루프의 마지막 단계)
   *
   * _SendFlits()와 _SendCredits()를 호출하여 이번 사이클에 파이프라인이 완료한
   * flit과 크레딧을 각각 출력 채널과 크레딧 채널로 전송한다.
   */
  virtual void WriteOutputs( );

  /*
   * [한국어]
   * Display - 모든 입력 버퍼 내용을 디버그 출력
   *
   * @os: 출력 스트림 (기본값: cout).
   * 모든 입력 포트의 _buf[input]->Display(os)를 호출한다. 디버깅/검사 목적.
   */
  void Display( ostream & os = cout ) const;

  /*
   * [한국어]
   * GetUsedCredit - 특정 출력 포트의 사용 중인 크레딧 수 반환
   *
   * @o: 출력 포트 인덱스.
   * @return: _next_buf[o]->Occupancy() — 다운스트림에서 소비된(아직 반환 안 된) 크레딧 수.
   * 외부 통계 수집 및 icnt_wrapper의 GetUsedCredit()에서 호출된다.
   */
  virtual int GetUsedCredit(int o) const;

  /*
   * [한국어]
   * GetBufferOccupancy - 특정 입력 포트의 버퍼 점유 flit 수 반환
   *
   * @i: 입력 포트 인덱스.
   * @return: _buf[i]->GetOccupancy() — 현재 해당 입력 포트의 모든 VC에 쌓인 flit 수.
   */
  virtual int GetBufferOccupancy(int i) const;

#ifdef TRACK_BUFFERS
  virtual int GetUsedCreditForClass(int output, int cl) const;   // [한국어] 출력 포트+클래스별 사용 크레딧 수 반환
  virtual int GetBufferOccupancyForClass(int input, int cl) const; // [한국어] 입력 포트+클래스별 버퍼 점유 수 반환
#endif

  /*
   * [한국어]
   * UsedCredits / FreeCredits / MaxCredits - 전체 출력 포트×VC 크레딧 상태 벡터 반환
   *
   * @return: _outputs*_vcs 크기의 벡터. 인덱스 o*_vcs+v에 포트 o, VC v의 값.
   * 외부(icnt_wrapper, 네트워크 통계)에서 전체 크레딧 상태를 한 번에 읽을 때 사용.
   */
  virtual vector<int> UsedCredits() const;
  virtual vector<int> FreeCredits() const;
  virtual vector<int> MaxCredits() const;

  // [한국어] 전력 모델링 모니터 접근자 — const 포인터의 const 포인터를 반환해 읽기 전용 접근 보장
  SwitchMonitor const * const GetSwitchMonitor() const {return _switchMonitor;}
  BufferMonitor const * const GetBufferMonitor() const {return _bufferMonitor;}

};

#endif
