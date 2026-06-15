// $Id: chaos_router.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] ChaosRouter 클래스 선언 (chaos_router.hpp)
 *
 * === 파일의 역할 ===
 * 이 파일은 BookSim2 NoC 시뮬레이터의 ChaosRouter 클래스를 선언한다.
 * ChaosRouter는 "멀티-큐(multi-queue)" 기반의 적응형(adaptive) 라우터로,
 * 기존의 VC(Virtual Channel) 크레딧 기반 흐름제어 대신 다운스트림 버퍼 수를
 * 직접 카운팅하는 방식으로 혼잡을 완화한다.
 * 입력 채널에서 수신된 패킷은 크로스바(crossbar)를 통해 직접 출력으로 전달되거나,
 * 일시적으로 멀티-큐(multi-queue)에 저장되어 나중에 다른 출력 포트로 우회(derouting)할
 * 수 있으므로 적응적 라우팅이 가능하다. GPGPU-Sim에서 주 GPU NoC 라우터는
 * IQRouter이며, ChaosRouter는 gpgpusim.config에서 `router chaos`로 설정할 때만
 * 활성화되는 대안적 라우터이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 전체 계층:
 *   CUDA 애플리케이션
 *     → libcuda (cuLaunchKernel 인터셉트)
 *       → gpgpusim_entrypoint.cc (시뮬레이터 진입)
 *         → gpgpu-sim/gpu-sim.cc (사이클-레벨 루프)
 *           → gpgpu-sim/shader.cc (SM 파이프라인, 메모리 요청 발행)
 *             → gpgpu-sim/icnt_wrapper.cc (NoC 인터페이스)
 *               → intersim2/ (BookSim2 NoC 시뮬레이터)
 *                 → [이 파일] ChaosRouter (적응형 라우터 구현)
 *                   → 메모리 파티션 (L2 캐시/DRAM)
 *
 * BookSim2 NoC 내부 계층:
 *   Network (네트워크 토폴로지 관리)
 *     → Channel (링크, 지연 모델링)
 *       → [이 파일] ChaosRouter (패킷 라우팅 및 버퍼 관리)
 *         → PipelineFIFO<Flit> (크로스바 파이프라인 지연 모델)
 *         → OutputSet (라우팅 결정 - 어느 출력 포트로 보낼지)
 *
 * 실행 컨텍스트: 호스트 CPU 유저스페이스, 단일 시뮬레이션 스레드.
 * 매 사이클마다 ReadInputs() → _InternalStep() → WriteOutputs() 순으로 호출된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - Router (router.hpp): 상위 추상 클래스, 입출력 채널/크레딧 채널 관리
 *   - Module (module.hpp): BookSim2 모듈 계층 최상위, 이름 및 부모-자식 관계
 *   - PipelineFIFO<Flit> (pipefifo.hpp): 크로스바 지연(crossbar_delay 사이클)을 모델링하는 FIFO
 *   - OutputSet (outputset.hpp): 라우팅 함수가 결정한 선호 출력 포트 집합
 *   - tRoutingFunction (routefunc.hpp): 라우팅 알고리즘 함수 포인터 타입
 *   - Flit (flitchannel.hpp): 단일 플릿(flit) 데이터 — head/tail 플래그, VC 번호, 페이로드
 *   - Credit (credit.hpp): 업스트림 라우터에게 버퍼 가용성을 알리는 크레딧 객체
 *   - BufferState (buffer_state.hpp): 다운스트림 버퍼 상태 추적 (ChaosRouter에서는 간접적 사용)
 *
 * 이 모듈에 의존하는 모듈:
 *   - Network (network.hpp): ChaosRouter를 생성하고 채널로 연결하는 상위 네트워크
 *   - icnt_wrapper.cc: intersim2_push/pop 함수를 통해 이 라우터에 패킷을 주입/추출
 *
 * 데이터 흐름:
 *   SM → (icnt_push) → 인젝션 채널(_inputs-1번) → _input_frame[]
 *     → 크로스바(_crossbar_pipe) 또는 _multi_queue[]
 *       → _output_frame[] → 이젝션 채널(_outputs-1번) → (icnt_pop) → 메모리 파티션
 *
 * === 주요 함수/구조체 요약 ===
 * ChaosRouter(config, parent, name, id, inputs, outputs):
 *   생성자. 버퍼, 멀티-큐, 크로스바 파이프, 라우팅 테이블을 초기화한다.
 *
 * ReadInputs():
 *   매 사이클 첫 번째 단계. 입력 채널에서 플릿을 읽어 _input_frame[]에 저장하고,
 *   출력 채널에서 크레딧을 받아 _next_queue_cnt[]를 갱신한다.
 *
 * _InternalStep():
 *   매 사이클 두 번째 단계. _NextInterestingChannel()로 이번 사이클에 처리할
 *   채널을 결정하고, _OutputAdvance()로 플릿을 크로스바 또는 멀티-큐로 이동시킨다.
 *
 * WriteOutputs():
 *   매 사이클 세 번째 단계. _SendFlits()로 크로스바를 통과한 플릿을 출력 채널에
 *   전송하고, _SendCredits()로 업스트림에 크레딧을 반환한다.
 *
 * eQState enum:
 *   입력 버퍼와 멀티-큐 슬롯의 6가지 상태를 표현한다. 슬롯이 동시에 두 패킷을
 *   부분적으로 보유할 수 있는 파이프라인 중첩(shared, cut_through) 상태를 포함한다.
 *
 * _NextInterestingChannel():
 *   라운드-로빈으로 처리할 채널을 선택하고 입력→출력 또는 입력→멀티-큐 매칭을 결정하는
 *   핵심 스케줄링 함수이다.
 *
 * _MultiQueueForOutput(output):
 *   주어진 출력 포트를 선호하는 가장 오래된 멀티-큐 슬롯을 반환한다. 선호 슬롯이 없고
 *   멀티-큐가 가득 찬 경우 임의 슬롯을 선택하는 "derouting" 경로를 포함한다.
 */

#ifndef _CHAOS_ROUTER_HPP_
#define _CHAOS_ROUTER_HPP_

#include <string>   // [한국어] std::string — 라우터 이름, 설정 키 조회에 사용
#include <queue>    // [한국어] std::queue — 입력/출력/멀티-큐 FIFO 버퍼, 크레딧 큐에 사용
#include <vector>   // [한국어] std::vector — 포트/슬롯 수만큼 동적 크기의 배열 컨테이너

#include "module.hpp"        // [한국어] BookSim2 모듈 계층 기반 클래스 (이름, 부모-자식 계층)
#include "router.hpp"        // [한국어] Router 추상 기반 클래스 — _inputs, _outputs, _input_channels[], _output_channels[] 등 포트 관리
#include "allocator.hpp"     // [한국어] BookSim2 할당자 인터페이스 (ChaosRouter는 직접 사용하지 않으나 Router 기반 클래스가 요구)
#include "routefunc.hpp"     // [한국어] tRoutingFunction 타입 정의 및 gRoutingFunctionMap 전역 맵 — 라우팅 알고리즘 함수 포인터 등록표
#include "outputset.hpp"     // [한국어] OutputSet 클래스 — 라우팅 결정 결과(선호 출력 포트 집합)를 표현
#include "buffer_state.hpp"  // [한국어] BufferState 클래스 — 다운스트림 버퍼 가용성 추적 (ChaosRouter에서는 _next_queue_cnt로 자체 추적)
#include "pipefifo.hpp"      // [한국어] PipelineFIFO<T> 템플릿 — 고정 지연 파이프라인 FIFO (크로스바 지연 모델링)
#include "vc.hpp"            // [한국어] VC(Virtual Channel) 관련 정의 — ChaosRouter는 단일 VC(VC 0)만 사용

/*
 * [한국어]
 * ChaosRouter - 멀티-큐 기반 적응형 NoC 라우터 클래스
 *
 * Router 추상 기반 클래스를 상속하며, BookSim2의 고전적인 VC 크레딧 흐름제어 대신
 * 멀티-큐(multi-queue)를 통한 패킷 우회(derouting)로 적응적 라우팅을 구현한다.
 *
 * 기본 동작 원리:
 *   1. ReadInputs(): 입력 채널에서 플릿을 받아 _input_frame[]에 저장.
 *      각 입력의 상태(eQState)를 갱신하고, head 플릿 도착 시 라우팅 함수(_rf) 호출.
 *   2. _InternalStep(): _NextInterestingChannel()로 이번 사이클 처리 채널을 선택.
 *      선택된 채널의 패킷을 출력 포트(크로스바) 또는 멀티-큐로 매칭.
 *      _OutputAdvance()로 플릿을 실제로 이동. 크로스바 파이프 Advance().
 *   3. WriteOutputs(): 크로스바를 빠져나온 플릿을 _output_frame[]에 적재 후
 *      다운스트림으로 전송. 업스트림에 크레딧 반환.
 *
 * IQRouter와의 차이:
 *   - IQRouter: VC별 독립 버퍼 + 크레딧 기반 흐름제어 (VC allocation, switch allocation)
 *   - ChaosRouter: 단일 입력 버퍼 + 멀티-큐 우회 + 다운스트림 큐 카운트로 흐름제어
 *
 * 설정 연동 (gpgpusim.config):
 *   - `router chaos`: 이 라우터를 선택
 *   - `vc_buf_size`: 입력/출력/멀티-큐 슬롯 크기 (_buffer_size)
 *   - `multi_queue_size`: 멀티-큐 슬롯 수 (_multi_queue_size)
 *   - `routing_function`: 라우팅 알고리즘 이름 (_rf 함수 포인터 결정)
 *   - `topology`: 토폴로지 이름 (라우팅 함수 이름 조합에 사용)
 *   - `crossbar_delay`: 크로스바 파이프라인 지연 사이클 수
 */
class ChaosRouter : public Router {

  tRoutingFunction   _rf;
  /* [한국어] 라우팅 알고리즘 함수 포인터.
   * 설정자: 생성자에서 gRoutingFunctionMap을 통해 config의 "routing_function"+"topology" 키로 조회하여 설정.
   * 읽는 자: ReadInputs()에서 head 플릿이 도착할 때, _OutputAdvance()에서 멀티-큐 head를 받을 때 호출.
   * 값 범위: 유효한 tRoutingFunction 함수 포인터 (NULL 불가, 없으면 Error() 호출).
   * 동기화: 단일 시뮬레이션 스레드에서만 접근하므로 락 불필요.
   * 호출 시그니처: _rf(router, flit, input_port, output_set, is_multipath)
   *   router: 현재 라우터 포인터
   *   flit: 라우팅할 플릿 (head 플릿에만 호출)
   *   input_port: 패킷이 도착한 입력 포트 번호
   *   output_set: 라우팅 결과를 저장할 OutputSet 포인터
   *   is_multipath: 멀티패스 라우팅 여부 (false = 단일 경로) */

  vector<OutputSet*> _input_route;
  /* [한국어] 각 입력 포트에 현재 수신 중인 패킷의 라우팅 결정(선호 출력 포트 집합).
   * 설정자: ReadInputs()에서 head 플릿 도착 시 _rf()를 호출하여 내용을 채운다.
   *   크기는 _inputs(입력 포트 수)이며, 생성자에서 resize 후 각 원소를 new OutputSet()으로 초기화.
   * 읽는 자: _NextInterestingChannel() → _InputForOutput()에서 입력 i가 특정 출력을 원하는지 확인.
   * 값 범위: 각 원소는 유효한 OutputSet 포인터. 패킷이 없을 때도 이전 결과가 남아있을 수 있으나
   *   _InputReady() 확인을 통해 유효 여부를 판단한다.
   * 동기화: 단일 스레드, 매 사이클 head 플릿 도착 시 덮어쓰기. */

  vector<OutputSet*> _mq_route;
  /* [한국어] 각 멀티-큐 슬롯에 저장된 패킷의 라우팅 결정(선호 출력 포트 집합).
   * 설정자: _OutputAdvance()에서 입력→멀티-큐 이동 시 head 플릿에 대해 _rf()를 호출하여 설정.
   *   크기는 _multi_queue_size이며, 생성자에서 resize 후 각 원소를 new OutputSet()으로 초기화.
   * 읽는 자: _NextInterestingChannel() → _MultiQueueForOutput()에서 특정 출력을 선호하는 슬롯 탐색.
   * 값 범위: 각 원소는 유효한 OutputSet 포인터. 슬롯이 비어있어도 이전 값이 남아있으므로
   *   _multi_state[]가 empty/leaving이 아닌 경우에만 참조해야 한다.
   * 동기화: 단일 스레드. */

  enum eQState {
    /* [한국어]
     * eQState - 입력 버퍼(_input_state[]) 및 멀티-큐 슬롯(_multi_state[])의 상태를 나타내는 열거형.
     *
     * ChaosRouter는 하나의 버퍼 슬롯이 동시에 두 패킷의 플릿을 부분적으로 보유할 수 있는
     * 파이프라인 중첩 동작(cut_through, shared)을 지원하므로 6가지 상태가 필요하다.
     *
     * 상태 전이도 (입력 버퍼 기준):
     *
     *   [empty]
     *     → head 도착 (단일 플릿 패킷 head+tail): → full
     *     → head 도착 (멀티 플릿 패킷 head만): → filling
     *
     *   [filling] (head 받음, tail 아직 도착 중)
     *     → tail 도착: → full
     *     → crossbar 매칭 시작: → cut_through
     *     → MQ 매칭 시작: → cut_through (이 경우 MQ 쪽)
     *
     *   [full] (패킷 전체 수신 완료, 전송 준비)
     *     → crossbar/MQ 매칭 시작: → leaving
     *
     *   [leaving] (head가 이미 출력으로 나갔고, tail은 아직 버퍼에 남음)
     *     → 새 head 도착 (다음 패킷): → shared
     *     → tail 이 crossbar/MQ로 나감: → empty
     *
     *   [cut_through] (head가 MQ로 나갔고, 나머지 플릿이 아직 입력에서 오는 중)
     *     → tail 도착: → leaving (tail도 MQ 이동 후 → empty)
     *
     *   [shared] (leaving 중 새 패킷의 head가 도착 — 두 패킷이 슬롯을 공유)
     *     → 이전 패킷 tail이 나감: → filling (새 패킷의 filling 단계)
     *
     * 멀티-큐 상태 전이도 (_multi_state[] 기준):
     *   [empty] → head 입력: → filling
     *   [filling] → tail 입력: → full
     *   [filling] → head 출력(crossbar): → cut_through
     *   [full] → head 출력(crossbar): → leaving
     *   [leaving] → 새 head 입력: → shared
     *   [leaving] → tail 출력: → empty
     *   [shared] → tail 출력: → filling
     *   [cut_through] → tail 입력: → leaving
     */

    empty,         //            input avail
    /* [한국어] 버퍼/슬롯이 완전히 비어있다. 새 패킷의 head 플릿을 받을 준비가 된 상태.
     * 초기값: 생성자에서 _input_state[]와 _multi_state[] 모두 empty로 초기화. */

    filling,       //    >**H    ready to send
    /* [한국어] head 플릿이 도착했지만 tail 플릿이 아직 오지 않은 상태. 패킷의 중간 플릿을 수신 중.
     * 주석의 ">**H"는 head(H)가 버퍼에 있고 나머지 플릿(**)이 뒤따라오는 모습.
     * _InputReady() 기준: filling 상태도 "준비됨(ready)"으로 간주 — cut-through 전송 허용.
     * 멀티-큐 filling: 이 슬롯에 head가 들어왔고 tail은 아직 이동 중임을 의미. */

    full,          //  T****H    ready to send
    /* [한국어] head부터 tail까지 패킷 전체가 버퍼에 수신 완료된 상태. 즉시 전송 가능.
     * 주석의 "T****H"는 tail(T)과 head(H) 사이의 모든 플릿(**)이 버퍼에 존재하는 모습.
     * _InputReady(): full도 "준비됨"으로 간주.
     * _NextInterestingChannel(): full 상태 입력 채널은 출력 포트와 무관하게 "interesting"으로 선택. */

    leaving,       //    T***>   input avail
    /* [한국어] head 플릿이 이미 크로스바/MQ로 전송되었고, 남은 플릿(tail 포함)이 아직 버퍼에 있는 상태.
     * 주석의 "T***>"는 tail(T)과 중간 플릿(***)이 버퍼에 있고 head(>)가 나간 모습.
     * 이 상태에서는 새 패킷의 head를 받을 수 있다 → shared 상태로 전이.
     * _InputReady(): leaving은 false (이미 매칭 완료, 추가 스케줄링 불필요). */

    cut_through,   //    >***>
    /* [한국어] head가 입력→MQ 또는 입력→크로스바로 cut-through 전송된 상태.
     * 동시에 입력 채널에서 나머지 플릿이 계속 들어오고 있다.
     * 주석의 ">***>"는 앞(>)과 뒤(>)가 모두 이동 중인 모습.
     * tail이 도착하면 → leaving 상태로 전이.
     * 이 상태에서 head 플릿이 새로 도착하면 Error() 발생 (버퍼 충돌). */

    shared         // >**HT**>
    /* [한국어] leaving 상태에서 새 패킷의 head가 도착하여 한 슬롯이 두 패킷을 부분 보유하는 상태.
     * 주석의 ">**HT**>"는 이전 패킷의 tail(T)과 새 패킷의 head(H)가 동시에 버퍼에 있는 모습.
     * (이전 패킷 head >는 나갔고, 이전 tail T는 아직 버퍼에, 새 head H가 방금 도착)
     * 이전 패킷 tail이 크로스바로 나가면 → filling(새 패킷만 남음) 상태로 전이.
     * 이 상태에서 추가 head가 도착하면 Error() 발생.
     * 멀티-큐 shared: 이전 패킷 leaving 중 새 head가 MQ로 들어온 경우. */
  };

  PipelineFIFO<Flit>   *_crossbar_pipe;
  /* [한국어] 크로스바 파이프라인 지연을 모델링하는 FIFO.
   * 설정자: 생성자에서 new PipelineFIFO<Flit>(this, "crossbar_pipeline", _outputs, _crossbar_delay)로 생성.
   *   _outputs는 출력 포트 수(채널 수), _crossbar_delay는 config의 crossbar_delay 설정값.
   * 읽는 자:
   *   - _OutputAdvance(): Write(f, output)로 플릿을 파이프라인에 주입.
   *   - _SendFlits(): Read(output)로 파이프라인을 빠져나온 플릿을 꺼냄.
   *   - _InternalStep(): Advance()로 매 사이클 파이프라인을 한 단계 진행.
   * 값 범위: 크로스바 지연(_crossbar_delay) 사이클 동안 플릿이 파이프라인에 머문 후 출력됨.
   * 동기화: 단일 스레드. 소멸자에서 delete. */

  int _multi_queue_size;
  /* [한국어] 멀티-큐 슬롯의 총 개수.
   * 설정자: 생성자에서 config.GetInt("multi_queue_size")로 설정.
   * 읽는 자: 생성자의 resize(), _FindAvailMultiQueue()의 루프 상한, _MultiQueueForOutput()의 루프 상한,
   *   _OutputAdvance()의 MQ 루프 상한, 소멸자의 삭제 루프.
   * 값 범위: 양의 정수. gpgpusim.config의 multi_queue_size 파라미터.
   * 동기화: 생성 후 불변(const처럼 동작). */

  int _buffer_size;
  /* [한국어] 입력 버퍼, 출력 버퍼, 멀티-큐 슬롯 각각의 최대 플릿 수.
   * 설정자: 생성자에서 config.GetInt("vc_buf_size")로 설정.
   *   assert: _buffer_size >= const_flits_per_packet (패킷 전체가 한 슬롯에 들어갈 수 있어야 함).
   * 읽는 자: _OutputFull(), _MultiQueueFull() — 버퍼/슬롯이 가득 찼는지 판단.
   *   _SendFlits() — 다운스트림 큐 카운트(_next_queue_cnt)와 비교하여 전송 여부 결정.
   * 값 범위: const_flits_per_packet 이상의 양의 정수.
   * 동기화: 생성 후 불변. */

  vector<queue<Flit *> > _input_frame;
  /* [한국어] 각 입력 포트의 수신 플릿 FIFO 큐.
   * 설정자: ReadInputs()에서 _input_channels[input]->Receive()로 받은 플릿을 push().
   * 읽는 자: _OutputAdvance()에서 front()로 꺼내 크로스바 또는 MQ로 이동.
   *   이동 성공 시 pop()으로 제거.
   * 값 범위: 크기는 _inputs. 각 큐에는 현재 수신 중인 패킷의 플릿들이 순서대로 저장됨.
   * 동기화: 단일 스레드. */

  vector<queue<Flit *> > _output_frame;
  /* [한국어] 각 출력 포트의 전송 대기 플릿 FIFO 큐.
   * 설정자: _SendFlits()에서 _crossbar_pipe->Read(output)로 받은 플릿을 push().
   * 읽는 자: _SendFlits()에서 front()를 _output_channels[output]->Send()로 전송 후 pop().
   *   _OutputAvail()에서 empty() 확인 — 비어있고 매칭 안 됐을 때만 출력 포트 "가용"으로 판단.
   *   _OutputFull()에서 size() 확인 — 크기 >= _buffer_size이면 "가득 참"으로 판단.
   * 값 범위: 크기는 _outputs. 각 큐에는 크로스바를 통과해 출력 대기 중인 플릿들이 저장됨.
   * 동기화: 단일 스레드. */

  vector<queue<Flit *> > _multi_queue;
  /* [한국어] 멀티-큐 슬롯별 플릿 저장 FIFO 큐.
   * 멀티-큐는 크로스바로 직접 보내지 못한 패킷을 임시 저장하는 버퍼이다.
   * 이후 _MultiQueueForOutput()에서 적합한 출력 포트가 생기면 크로스바로 전송된다.
   * 설정자: _OutputAdvance()에서 _input_mq_match[]가 설정된 경우 push()로 플릿 저장.
   * 읽는 자: _OutputAdvance()에서 _multi_match[]가 설정된 슬롯에서 front()/pop()으로 꺼내 크로스바에 Write().
   *   _MultiQueueFull()에서 size() 확인.
   * 값 범위: 크기는 _multi_queue_size. 각 큐에는 한 패킷의 플릿들이 순서대로 저장됨.
   * 동기화: 단일 스레드. */

  vector<int> _next_queue_cnt;
  /* [한국어] 각 출력 포트의 다운스트림(다음 라우터/노드) 버퍼에 현재 전송된 플릿 수 카운터.
   * ChaosRouter는 VC 크레딧 기반 흐름제어 대신 이 카운터로 다운스트림 혼잡을 판단한다.
   * 설정자:
   *   - _SendFlits()에서 플릿을 전송할 때 ++_next_queue_cnt[output].
   *   - ReadInputs()에서 다운스트림이 크레딧을 보내오면 --_next_queue_cnt[output].
   * 읽는 자: _SendFlits()에서 _next_queue_cnt[output] < _buffer_size 조건으로 전송 가능 여부 판단.
   *   (현재 _OutputAvail()의 조건에서는 주석처리되어 있으나 _SendFlits()에서 여전히 사용)
   * 값 범위: [0, _buffer_size]. 0이면 다운스트림 버퍼 비어있음, _buffer_size이면 가득 참.
   * 동기화: 단일 스레드. ReadInputs()에서 감소, _SendFlits()에서 증가 — 같은 사이클 내에서 순서 보장. */

  vector<queue<Credit *> > _credit_queue;
  /* [한국어] 각 입력 포트로 반환할 크레딧 객체 FIFO 큐.
   * 업스트림 라우터(또는 인젝션 소스)에게 이 라우터가 플릿을 소비했음을 알리는 크레딧.
   * 설정자: _OutputAdvance()에서 플릿이 성공적으로 크로스바 또는 MQ로 이동되면
   *   Credit::New()를 생성하여 push().
   * 읽는 자: _SendCredits()에서 front()/pop()으로 꺼내 _input_credits[input]->Send()로 전송.
   * 값 범위: 크기는 _inputs. 각 큐에는 전송할 크레딧 객체들이 저장됨.
   * 동기화: 단일 스레드. */

  vector<eQState> _input_state;
  /* [한국어] 각 입력 포트 버퍼의 현재 상태.
   * 설정자:
   *   - 생성자: 모두 empty로 초기화.
   *   - ReadInputs(): 플릿 도착 시 상태 전이 (empty→filling/full, filling→full, leaving→shared, cut_through→leaving 등).
   *   - _NextInterestingChannel(): full→leaving, filling→cut_through (크로스바/MQ 매칭 시).
   *   - _OutputAdvance(): leaving→empty, shared→filling (tail 플릿 이동 후).
   * 읽는 자: ReadInputs()의 switch(_input_state[input]), _InputReady(), _OutputAvail(),
   *   _NextInterestingChannel(), _OutputAdvance().
   * 값 범위: eQState 열거형 6가지 값 중 하나.
   * 동기화: 단일 스레드. */

  vector<eQState> _multi_state;
  /* [한국어] 각 멀티-큐 슬롯의 현재 상태.
   * 설정자:
   *   - 생성자: 모두 empty로 초기화.
   *   - _OutputAdvance(): 플릿이 MQ로 들어오거나 MQ에서 크로스바로 나갈 때 상태 전이.
   * 읽는 자: _MultiQueueForOutput(), _FindAvailMultiQueue(), _OutputAdvance(), Display().
   * 값 범위: eQState 열거형 6가지 값 중 하나.
   * 동기화: 단일 스레드. */

  vector<int> _input_output_match;
  /* [한국어] 각 입력 포트에 대해 이번 사이클에 매칭된 출력 포트 번호.
   * -1이면 이번 사이클에 크로스바 직접 매칭 없음을 의미한다.
   * 설정자:
   *   - 생성자: 모두 -1로 초기화.
   *   - _NextInterestingChannel(): 입력→출력 직접 매칭 시 _input_output_match[in_index] = _cur_channel.
   *   - _OutputAdvance(): tail 플릿이 이동된 후 -1로 리셋.
   * 읽는 자: _OutputAdvance()에서 _input_output_match[i] != -1이면 크로스바로 플릿 이동.
   * 값 범위: -1(미매칭) 또는 [0, _outputs-1].
   * 동기화: 단일 스레드. */

  vector<int> _input_mq_match;
  /* [한국어] 각 입력 포트에 대해 이번 사이클에 매칭된 멀티-큐 슬롯 번호.
   * -1이면 이번 사이클에 MQ 매칭 없음을 의미한다.
   * 설정자:
   *   - 생성자: 모두 -1로 초기화.
   *   - _NextInterestingChannel(): 입력→MQ 매칭 시 _input_mq_match[_cur_channel] = mq_avail.
   *   - _OutputAdvance(): tail 플릿이 MQ로 이동된 후 -1로 리셋.
   * 읽는 자: _OutputAdvance()에서 _input_mq_match[i] != -1이면 MQ로 플릿 이동.
   * 값 범위: -1(미매칭) 또는 [0, _multi_queue_size-1].
   * 동기화: 단일 스레드. */

  vector<int> _multi_match;
  /* [한국어] 각 멀티-큐 슬롯에 대해 이번 사이클에 매칭된 출력 포트 번호.
   * -1이면 이번 사이클에 이 MQ 슬롯이 출력 포트에 매칭되지 않음을 의미한다.
   * 설정자:
   *   - 생성자: 모두 -1로 초기화.
   *   - _NextInterestingChannel(): MQ→출력 매칭 시 _multi_match[mq_index] = _cur_channel.
   *   - _OutputAdvance(): MQ 슬롯의 tail 플릿이 크로스바로 나간 후 -1로 리셋.
   * 읽는 자: _OutputAdvance()에서 _multi_match[m] != -1이면 MQ에서 크로스바로 플릿 이동.
   * 값 범위: -1(미매칭) 또는 [0, _outputs-1].
   * 동기화: 단일 스레드. */

  vector<int> _mq_age;
  /* [한국어] 각 멀티-큐 슬롯에 패킷이 들어온 이후 경과한 사이클 수(나이).
   * 값이 클수록 더 오래된 패킷이다. _MultiQueueForOutput()에서 가장 오래된 슬롯을
   * 우선 선택하여 기아(starvation)를 방지한다(LRU/oldest-first 정책).
   * 설정자:
   *   - _OutputAdvance(): head 플릿이 MQ로 들어올 때 _mq_age[mq] = 0으로 리셋.
   *   - _OutputAdvance() MQ 루프 끝: 매 사이클 _mq_age[m]++ (모든 슬롯 나이 증가).
   * 읽는 자: _MultiQueueForOutput()에서 mq_age[i] > mq_age 비교로 가장 오래된 슬롯 선택.
   * 값 범위: 0 이상의 정수. MQ 슬롯이 비어있을 때도 계속 증가하나 해당 상태에서는 참조되지 않음.
   * 동기화: 단일 스레드. */

  vector<bool> _output_matched;
  /* [한국어] 각 출력 포트가 이번 사이클에 이미 패킷에 매칭되어 있는지 여부.
   * true이면 이 출력 포트는 이미 다른 입력/MQ에 할당되어 _OutputAvail()에서 false를 반환.
   * 설정자:
   *   - 생성자: 모두 false로 초기화.
   *   - _NextInterestingChannel(): MQ 또는 입력이 출력에 매칭될 때 true로 설정.
   *   - _OutputAdvance(): tail 플릿이 크로스바로 나간 후 false로 리셋.
   * 읽는 자: _OutputAvail()에서 !_output_matched[out] 조건으로 확인.
   * 값 범위: true/false.
   * 동기화: 단일 스레드. 같은 사이클 내 _NextInterestingChannel()과 _OutputAdvance()에서 순서대로 접근. */

  vector<bool> _mq_matched;
  /* [한국어] 각 멀티-큐 슬롯이 이번 사이클에 이미 입력 채널 매칭에 예약되어 있는지 여부.
   * true이면 이 MQ 슬롯은 _FindAvailMultiQueue()에서 사용 불가로 간주된다.
   * 설정자:
   *   - 생성자/초기화 루프: 모두 false로 초기화.
   *   - _NextInterestingChannel(): 입력→MQ 매칭 시 _mq_matched[mq_avail] = true.
   *   - _OutputAdvance(): 입력의 tail 플릿이 MQ에 이동된 후 _mq_matched[mq] = false로 리셋.
   * 읽는 자: _FindAvailMultiQueue()에서 !_mq_matched[i] 조건으로 확인.
   * 값 범위: true/false.
   * 동기화: 단일 스레드. */

  int _cur_channel;
  /* [한국어] _NextInterestingChannel()에서 현재 검사 중인 채널(입력 포트) 인덱스. 라운드-로빈 포인터.
   * 설정자:
   *   - 생성자: 0으로 초기화 (첫 번째 채널부터 시작).
   *   - _NextInterestingChannel(): "interesting"하지 않으면 (_cur_channel + 1) % _inputs로 증가.
   *     매칭 성공 또는 MQ 매칭 시 다음 사이클을 위해 다음 채널로 이동.
   *     채널이 스탈(stall) 중이면 이동하지 않고 재시도.
   * 읽는 자: _NextInterestingChannel()에서 루프 변수 및 매칭 기준 채널.
   * 값 범위: [0, _inputs-1]. 모듈러 연산으로 순환.
   * 동기화: 단일 스레드. */

  int _read_stall;
  /* [한국어] 인젝션 채널이 아닌 채널이 크로스바 출력 포트를 기다리며 스탈된 사이클 수 카운터.
   * 이 값이 양수이면 _NextInterestingChannel()에서 MQ 우회를 시도한다.
   * 즉, 출력 포트가 바빠서 직접 크로스바 전송이 안 될 때 MQ로 패킷을 돌려 채널 스탈을 해소한다.
   * 설정자:
   *   - 생성자: 0으로 초기화.
   *   - _NextInterestingChannel(): 비인젝션 채널이 interesting하지만 출력 포트가 없으면 ++_read_stall.
   *     매칭 성공 또는 MQ 우회 성공 시 0으로 리셋.
   * 읽는 자: _NextInterestingChannel()에서 _read_stall > 0이면 MQ 우회 경로 진입.
   * 값 범위: 0 이상의 정수. 정상 동작 시 0, 스탈 중에는 양수.
   * 동기화: 단일 스레드. */

  /*
   * [한국어]
   * _IsInjectionChan - 주어진 채널이 인젝션 채널인지 확인
   *
   * @param chan: 검사할 채널 번호 [0, _inputs-1]
   * @return: true이면 인젝션 채널(SM에서 NoC로 패킷을 주입하는 채널)
   *
   * GPGPU-Sim에서 인젝션/이젝션 채널은 관례적으로 마지막 입력/출력 포트 번호를 사용한다.
   * 인젝션 채널은 _read_stall 카운팅 대상에서 제외된다 — SM이 직접 패킷을 주입하는
   * 채널이므로 스탈 카운트와 관계없이 독립적으로 처리해야 하기 때문이다.
   *
   * 호출 체인: _NextInterestingChannel() → [이 함수]
   */
  bool _IsInjectionChan( int chan ) const;

  /*
   * [한국어]
   * _IsEjectionChan - 주어진 채널이 이젝션 채널인지 확인
   *
   * @param chan: 검사할 채널 번호 [0, _outputs-1]
   * @return: true이면 이젝션 채널(NoC에서 메모리 파티션으로 패킷을 내보내는 채널)
   *
   * _MultiQueueForOutput()에서 MQ 패킷을 이젝션 채널로 derouting(우회)하는 것을 금지하는 데 사용된다.
   * 이젝션 채널로의 derouting은 의미 없는 패킷 전달을 유발할 수 있기 때문이다.
   *
   * 호출 체인: _MultiQueueForOutput() → [이 함수]
   */
  bool _IsEjectionChan( int chan ) const;

  /*
   * [한국어]
   * _InputReady - 입력 포트가 전송 준비된 패킷을 보유하고 있는지 확인
   *
   * @param input: 검사할 입력 포트 번호 [0, _inputs-1]
   * @return: true이면 이 입력이 크로스바 전송 또는 MQ 이동 가능한 상태
   *
   * filling 상태(tail 미도착)도 ready로 간주하여 cut-through 전송을 허용한다.
   * cut-through: head가 도착하는 즉시 전송을 시작하고, 이후 플릿은 도착하는 대로 전달.
   *
   * 호출 체인: _InputForOutput() → [이 함수]
   */
  bool _InputReady( int input ) const;

  /*
   * [한국어]
   * _OutputFull - 출력 포트의 전송 대기 큐가 가득 찼는지 확인
   *
   * @param out: 검사할 출력 포트 번호 [0, _outputs-1]
   * @return: true이면 _output_frame[out] 크기 >= _buffer_size (더 이상 플릿을 받을 수 없음)
   *
   * 호출 체인: _OutputAvail()에서 참조(현재 주석처리된 조건에서), _MultiQueueForOutput()에서 간접 참조
   */
  bool _OutputFull( int out ) const;

  /*
   * [한국어]
   * _OutputAvail - 출력 포트가 새로운 패킷을 받을 수 있는지 확인
   *
   * @param out: 검사할 출력 포트 번호 [0, _outputs-1]
   * @return: true이면 아직 매칭되지 않았고 출력 큐가 비어있음 (전송 가능)
   *
   * 조건: !_output_matched[out] && _output_frame[out].empty()
   * 출력 큐가 비어있어야 하므로 실질적으로 한 번에 하나의 패킷만 출력 포트로 진입 가능.
   * (크로스바 지연 중 _output_frame이 비어있을 때만 새 패킷 수용 — 보수적 정책)
   *
   * 호출 체인: _NextInterestingChannel() → [이 함수]
   */
  bool _OutputAvail( int out ) const;

  /*
   * [한국어]
   * _MultiQueueFull - 멀티-큐 슬롯이 가득 찼는지 확인
   *
   * @param mq: 검사할 멀티-큐 슬롯 번호 [0, _multi_queue_size-1]
   * @return: true이면 _multi_queue[mq] 크기 >= _buffer_size
   *
   * 호출 체인: _FindAvailMultiQueue() → [이 함수], _OutputAdvance() → [이 함수]
   */
  bool _MultiQueueFull( int mq ) const;

  /*
   * [한국어]
   * _InputForOutput - 주어진 출력 포트를 선호하는 입력 포트 검색
   *
   * @param output: 찾고자 하는 출력 포트 번호 [0, _outputs-1]
   * @return: 해당 출력을 선호하는 ready 상태 입력 포트 번호, 없으면 -1
   *
   * 랜덤 오프셋(RandomInt)을 시작점으로 라운드-로빈으로 모든 입력을 검사하여 공평성 보장.
   * _InputReady(input) && !_input_route[input]->OutputEmpty(output) 조건으로 매칭.
   * 여러 입력이 같은 출력을 원할 때 먼저 발견된 입력을 반환 (랜덤 시작으로 공평성 확보).
   *
   * 호출 체인: _NextInterestingChannel() → [이 함수]
   */
  int  _InputForOutput( int output ) const;

  /*
   * [한국어]
   * _MultiQueueForOutput - 주어진 출력 포트로 보낼 최적 멀티-큐 슬롯 검색
   *
   * @param output: 찾고자 하는 출력 포트 번호 [0, _outputs-1]
   * @return: 해당 출력을 선호하는 가장 오래된 MQ 슬롯 번호, 없으면 -1
   *
   * 우선순위:
   *   1. 해당 output을 선호하고, 아직 매칭 안 된(_multi_match[i]==-1) filling/full 슬롯 중
   *      가장 오래된 것(_mq_age 최대) 반환.
   *   2. 선호 슬롯이 없고, MQ 전체가 가득 차있고(_multi_state가 filling/full/shared),
   *      이젝션 채널이 아닌 경우 → derouting: 임의의 filling/full 슬롯 반환.
   *      (derouting: 원래 목적지가 아닌 출력으로 패킷을 보내 MQ 공간 확보)
   *
   * 호출 체인: _NextInterestingChannel() → [이 함수]
   */
  int  _MultiQueueForOutput( int output ) const;

  /*
   * [한국어]
   * _FindAvailMultiQueue - 비어있는(또는 가득 차지 않은) 멀티-큐 슬롯 검색
   *
   * @return: 사용 가능한 MQ 슬롯 번호 (비어있고 매칭 안 됨), 없으면 -1
   *
   * 조건: !_MultiQueueFull(i) && !_mq_matched[i]
   * MQ 우회(derouting) 이전에 실제로 입력 패킷을 수용할 슬롯이 있는지 확인.
   *
   * 호출 체인: _NextInterestingChannel() → [이 함수]
   */
  int  _FindAvailMultiQueue( ) const;

  /*
   * [한국어]
   * _NextInterestingChannel - 이번 사이클에 처리할 채널을 선택하고 입력/MQ 매칭 수행
   *
   * @param: 없음
   * @return: 없음 (내부 상태 _input_output_match[], _input_mq_match[], _multi_match[],
   *          _output_matched[], _input_state[], _cur_channel, _read_stall 갱신)
   *
   * ChaosRouter의 핵심 스케줄링 함수. 매 사이클 _InternalStep()에서 첫 번째로 호출된다.
   *
   * "interesting" 채널의 정의:
   *   1. _cur_channel에 대응하는 출력 포트가 가용(_OutputAvail())하고,
   *      그 출력을 원하는 MQ 슬롯(_MultiQueueForOutput) 또는 입력(_InputForOutput)이 존재하거나
   *   2. _cur_channel의 입력 상태가 full (꽉 찬 패킷이 기다리고 있음)
   *
   * 매칭 우선순위 (interesting한 채널에 대해):
   *   1. MQ 슬롯이 출력을 선호 → _multi_match[mq_index] = _cur_channel (MQ→출력 매칭 우선)
   *   2. 입력이 출력을 선호 → _input_output_match[in_index] = _cur_channel (입력→출력 직접 매칭)
   *   (두 조건이 모두 해당되면 MQ가 우선 — MQ에 오래된 패킷이 있을 수 있으므로)
   *
   * 스탈(stall) 처리:
   *   - 비인젝션 채널이 interesting하지만 출력 포트가 없으면 _read_stall 증가.
   *   - _read_stall > 0이면 MQ 우회를 시도: _input_mq_match[_cur_channel] = mq_avail.
   *   - MQ 슬롯도 없으면 _read_stall 추가 증가 (완전 스탈).
   *
   * 호출 체인: _InternalStep() → [이 함수]
   */
  void _NextInterestingChannel( );

  /*
   * [한국어]
   * _OutputAdvance - _NextInterestingChannel()의 매칭 결과에 따라 플릿을 실제로 이동
   *
   * @param: 없음
   * @return: 없음 (내부 버퍼 상태 변경, 크레딧 생성)
   *
   * _NextInterestingChannel() 직후 _InternalStep()에서 호출되며, 3가지 경로로 플릿을 이동:
   *   경로 A (입력→크로스바): _input_output_match[i] != -1이면 _crossbar_pipe->Write(f, output)
   *   경로 B (입력→MQ): _input_mq_match[i] != -1이면 _multi_queue[mq].push(f)
   *   경로 C (MQ→크로스바): _multi_match[m] != -1이면 _crossbar_pipe->Write(f, output)
   *
   * 각 경로에서 tail 플릿 이동 후 매칭 정보(_input_output_match, _input_mq_match, _multi_match) 리셋.
   * 플릿 이동 성공 시 크레딧(Credit::New())을 생성하여 _credit_queue[i]에 push (업스트림 반환용).
   * 매 사이클 끝에 _mq_age[m]++ (모든 MQ 슬롯의 나이 증가).
   *
   * 호출 체인: _InternalStep() → [이 함수] → _crossbar_pipe->Write(), _multi_queue[].push()
   */
  void _OutputAdvance( );

  /*
   * [한국어]
   * _SendFlits - 크로스바 파이프라인을 통과한 플릿을 출력 채널로 전송
   *
   * @param: 없음
   * @return: 없음 (출력 채널로 플릿 전송, _next_queue_cnt 갱신)
   *
   * WriteOutputs()에서 호출. _crossbar_pipe->Read(output)으로 파이프라인을 빠져나온 플릿을
   * _output_frame[output]에 적재한다. 이후 _next_queue_cnt[output] < _buffer_size이면
   * _output_channels[output]->Send()로 다운스트림에 전송하고 카운터를 증가시킨다.
   * f->hops++로 패킷이 한 홉 더 이동했음을 기록한다.
   *
   * 호출 체인: WriteOutputs() → [이 함수]
   */
  void _SendFlits( );

  /*
   * [한국어]
   * _SendCredits - 입력 채널로 크레딧 반환
   *
   * @param: 없음
   * @return: 없음 (업스트림 라우터에 크레딧 전송)
   *
   * WriteOutputs()에서 호출. _credit_queue[input]에 대기 중인 크레딧을
   * _input_credits[input]->Send()로 업스트림에 전송한다.
   * 크레딧은 _OutputAdvance()에서 플릿 이동 성공 시 생성된다.
   *
   * 호출 체인: WriteOutputs() → [이 함수]
   */
  void _SendCredits( );

  /*
   * [한국어]
   * _InternalStep - 한 사이클의 내부 처리 단계 수행 (가상 함수 오버라이드)
   *
   * @param: 없음
   * @return: 없음
   *
   * Router 기반 클래스의 순수 가상 함수를 구현한다.
   * 순서: _NextInterestingChannel() → _OutputAdvance() → _crossbar_pipe->Advance()
   * 크로스바 파이프 Advance()는 마지막에 수행되어야 이번 사이클에 Write()된 플릿이
   * 다음 사이클부터 파이프라인에서 진행된다.
   *
   * 호출 체인: Network::Evaluate() → [이 함수]
   */
  virtual void _InternalStep( );

public:
  /*
   * [한국어]
   * ChaosRouter - 생성자
   *
   * @param config: BookSim2 설정 객체 (vc_buf_size, multi_queue_size, routing_function, topology 등)
   * @param parent: 부모 모듈 포인터 (Network 객체)
   * @param name: 이 라우터의 이름 문자열 (예: "router_0")
   * @param id: 이 라우터의 고유 ID 번호 (네트워크 내 라우터 인덱스)
   * @param inputs: 입력 포트 수 (출력 포트 수와 동일해야 함, 아니면 Error())
   * @param outputs: 출력 포트 수
   *
   * 초기화 순서:
   *   1. 기반 클래스 Router 초기화 (입출력 채널, 크레딧 채널 배열 할당).
   *   2. inputs == outputs 검증.
   *   3. _buffer_size(vc_buf_size), _multi_queue_size(multi_queue_size) 설정.
   *   4. _cur_channel=0, _read_stall=0 초기화.
   *   5. 라우팅 함수(_rf) 조회 및 설정.
   *   6. _input_route[], _mq_route[] OutputSet 객체 배열 할당.
   *   7. _crossbar_pipe PipelineFIFO 생성.
   *   8. 모든 버퍼/큐/상태 벡터 resize 및 초기화.
   *
   * 호출 체인: Network::AddRouter() 또는 Network 생성자 → [이 함수]
   */
  ChaosRouter( const Configuration& config,
	    Module *parent, const string & name, int id,
	    int inputs, int outputs );

  /*
   * [한국어]
   * ~ChaosRouter - 소멸자
   *
   * @param: 없음
   * @return: 없음
   *
   * _crossbar_pipe와 _input_route[], _mq_route[]의 OutputSet 객체를 delete.
   * _input_frame[], _output_frame[], _multi_queue[], _credit_queue[]의 큐 원소(Flit*, Credit*)는
   * 상위 레이어(Network)에서 관리하므로 여기서 개별 삭제하지 않는다.
   *
   * 호출 체인: Network 소멸 → [이 함수]
   */
  virtual ~ChaosRouter( );

  /*
   * [한국어]
   * ReadInputs - 입력 채널에서 플릿/크레딧을 읽는 단계 (순수 가상 함수 구현)
   *
   * @param: 없음
   * @return: 없음
   *
   * 매 사이클 Network::Evaluate()가 시작하기 전 Network::ReadInputs()가 모든 라우터에 대해 호출.
   * 모든 입력 채널(_input_channels[])을 순회하여 플릿을 _input_frame[]에 저장하고
   * 출력 채널(_output_credits[])에서 크레딧을 수신하여 _next_queue_cnt[]를 감소시킨다.
   *
   * 호출 체인: Network::ReadInputs() → [이 함수]
   */
  virtual void ReadInputs( );

  /*
   * [한국어]
   * WriteOutputs - 출력 채널로 플릿/크레딧을 전송하는 단계 (순수 가상 함수 구현)
   *
   * @param: 없음
   * @return: 없음
   *
   * 매 사이클 _InternalStep() 이후 Network::WriteOutputs()가 모든 라우터에 대해 호출.
   * _SendFlits()로 크로스바 파이프라인을 빠져나온 플릿을 출력 채널로 전송하고,
   * _SendCredits()로 업스트림에 크레딧을 반환한다.
   *
   * 호출 체인: Network::WriteOutputs() → [이 함수]
   */
  virtual void WriteOutputs( );

  /*
   * [한국어]
   * GetUsedCredit - 출력 포트의 사용된 크레딧 수 반환 (IQRouter 호환 인터페이스)
   *
   * @param out: 출력 포트 번호
   * @return: 항상 0 반환 (ChaosRouter는 VC 크레딧 기반 흐름제어를 사용하지 않으므로)
   *
   * GPGPU-Sim의 icnt_wrapper에서 통계 수집 목적으로 호출될 수 있으나,
   * ChaosRouter는 VC 구조가 없으므로 의미있는 값을 반환하지 않는다.
   */
  virtual int GetUsedCredit(int out) const {return 0;}

  /*
   * [한국어]
   * GetBufferOccupancy - 입력 포트의 버퍼 점유율 반환 (IQRouter 호환 인터페이스)
   *
   * @param i: 입력 포트 번호
   * @return: 항상 0 반환 (ChaosRouter에서는 의미있는 통계 미지원)
   */
  virtual int GetBufferOccupancy(int i) const {return 0;}

#ifdef TRACK_BUFFERS
  /* [한국어] TRACK_BUFFERS 컴파일 옵션이 활성화된 경우에만 포함되는 VC별/클래스별 통계 함수들.
   * ChaosRouter는 VC를 사용하지 않으므로 모두 0을 반환하는 스텁(stub)으로 구현. */
  virtual int GetUsedCreditForClass(int output, int cl) const {return 0;}
  virtual int GetBufferOccupancyForClass(int input, int cl) const {return 0;}
#endif

  /*
   * [한국어]
   * UsedCredits / FreeCredits / MaxCredits - VC별 크레딧 통계 벡터 반환
   *
   * @return: 빈 vector<int> (ChaosRouter는 VC 크레딧 구조 미사용)
   *
   * IQRouter와의 인터페이스 호환성을 위해 존재하는 스텁 함수들.
   * GPGPU-Sim의 icnt_wrapper나 통계 수집 코드에서 호출될 수 있으나 실질적 의미 없음.
   */
  virtual vector<int> UsedCredits() const { return vector<int>(); }
  virtual vector<int> FreeCredits() const { return vector<int>(); }
  virtual vector<int> MaxCredits() const { return vector<int>(); }

  /*
   * [한국어]
   * Display - 라우터 내부 상태를 출력 스트림에 덤프
   *
   * @param os: 출력 대상 스트림 (기본값: cout)
   * @return: 없음
   *
   * 현재 chaos_router.cpp의 구현체는 비어있음 (빈 함수 본체).
   * 디버깅 또는 상태 덤프 목적으로 호출될 수 있으나 ChaosRouter에서는 미구현.
   *
   * 호출 체인: 외부 디버그 코드 또는 Network::Display() → [이 함수]
   */
  void Display( ostream & os = cout ) const;
};

#endif
