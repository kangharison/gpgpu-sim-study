// $Id: buffer_state.hpp 5378 2013-01-10 03:40:11Z dub $

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
 * [한국어 설명] 다운스트림 라우터 버퍼 상태 추적 클래스 선언 (buffer_state.hpp)
 *
 * === 파일의 역할 ===
 * BufferState 클래스는 상류(upstream) 라우터가 하류(downstream) 라우터의
 * 입력 버퍼 잔여 공간을 크레딧 기반으로 추적하는 객체이다.
 * NoC의 크레딧 기반 흐름 제어(credit-based flow control)에서 송신 측은
 * 수신 측이 반환하는 크레딧(Credit 패킷)을 받아 BufferState를 갱신하며,
 * 이를 통해 다운스트림에 공간이 있을 때만 플릿을 전송한다.
 * 내부에 여러 버퍼 정책(private, shared, limited, dynamic, feedback 등)을
 * 전략 패턴(Strategy Pattern)으로 구현하여 설정에 따라 교체 가능하다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2 NoC 시뮬레이터의 흐름 제어 계층.
 * 라우터의 출력 포트별로 BufferState 인스턴스가 하나씩 존재한다.
 * 크레딧 처리 흐름:
 *   다운스트림 라우터 Buffer의 RemoveFlit() 시 크레딧 생성
 *   → CreditChannel로 전송 → 업스트림 라우터 ReadInputs()
 *   → BufferState::ProcessCredit() 호출 → occupancy 감소
 * 플릿 전송 흐름:
 *   라우터 SA 완료 → BufferState::SendingFlit() → occupancy 증가
 *   → IsFullFor(vc) 확인 → 공간 있을 때만 실제 전송
 * 실행 컨텍스트: 호스트 CPU 싱글 스레드, 매 사이클마다 호출됨.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - module.hpp: Module 기반 클래스 (계층형 이름, Error() 제공)
 *   - flit.hpp:   Flit 클래스 — 플릿의 vc, tail, cl 필드 참조
 *   - credit.hpp: Credit 클래스 — 크레딧 패킷 (vc 집합 포함)
 *   - config_utils.hpp: Configuration — num_vcs, buf_size, buffer_policy 등 읽기
 * 이 파일을 사용하는 모듈:
 *   - iqrouter.cc, kncrouter.cc: 출력 포트별 BufferState 생성·ProcessCredit·SendingFlit 호출
 *   - vcalloc.cc: IsFullFor(), IsAvailableFor(), AvailableFor() 호출해 VC 할당 결정
 *
 * === 주요 함수/구조체 요약 ===
 * BufferState::BufferState(): 생성자 — 크기/VCs 초기화, 버퍼 정책 객체 생성
 * ProcessCredit(c): 크레딧 패킷 처리 — 해당 VC의 occupancy 감소
 * SendingFlit(f):   플릿 송신 기록 — 해당 VC의 occupancy 증가
 * TakeBuffer(vc, tag): VC를 특정 패킷(tag)에 할당 (VC 전용 예약)
 * IsFullFor(vc):    지정 VC에 대해 다운스트림 버퍼가 가득 찼는지 조회
 * IsAvailableFor(vc): 지정 VC가 현재 다른 패킷에 사용 중인지 조회
 * BufferPolicy 중첩 클래스 계층: 7가지 버퍼 정책을 전략 패턴으로 구현
 */

#ifndef _BUFFER_STATE_HPP_
#define _BUFFER_STATE_HPP_

#include <vector> // [한국어] VC별 occupancy, in_use_by, tail_sent, last_id, last_pid 벡터에 사용
#include <queue>  // [한국어] FeedbackSharedBufferPolicy의 _flit_sent_time (queue<int>), TRACK_BUFFERS의 _outstanding_classes에 사용

#include "module.hpp"       // [한국어] Module 기반 클래스 — 계층형 이름, Error() 제공
#include "flit.hpp"         // [한국어] Flit 클래스 — vc, tail, cl, id, pid 필드 참조
#include "credit.hpp"       // [한국어] Credit 클래스 — 크레딧 패킷 (set<int> vc 집합)
#include "config_utils.hpp" // [한국어] Configuration 클래스 — num_vcs, buf_size, buffer_policy 등 읽기

/*
 * [한국어]
 * BufferState - 다운스트림 라우터 버퍼의 잔여 공간을 크레딧으로 추적하는 클래스
 *
 * 업스트림 라우터가 플릿을 보내거나 크레딧을 받을 때마다 내부 상태를 갱신한다.
 * 버퍼 정책(BufferPolicy)은 추상 기반 클래스 + 7가지 구체 클래스로 분리되어
 * 설정 파라미터 "buffer_policy"에 따라 런타임에 선택된다.
 *
 * 상속: Module — 계층형 이름 트리에 등록, Error()/FullName() 지원
 */
class BufferState : public Module {

  /*
   * [한국어]
   * BufferPolicy - 버퍼 가용성 정책의 추상 기반 클래스 (전략 패턴)
   *
   * 각 구체 정책 클래스는 이 클래스를 상속하여 IsFullFor/AvailableFor/LimitFor를 구현한다.
   * BufferState는 _buffer_policy 포인터를 통해 정책과 무관하게 동작한다.
   *
   * 상속: Module — 이름 트리에 등록됨
   */
  class BufferPolicy : public Module {
  protected:
    BufferState const * const _buffer_state;
    /* 이 정책 객체를 소유하는 BufferState에 대한 const 포인터.
     * 설정자: 생성자 초기화 리스트에서 parent 포인터로 설정.
     * 읽는 자: 각 구체 정책의 IsFullFor/AvailableFor 등이 OccupancyFor() 등 조회에 사용.
     * 값 범위: 항상 유효한 BufferState 포인터 (NULL 불가).
     * 동기화: 단일 스레드 — 불필요. */
  public:
    /*
     * [한국어]
     * BufferPolicy 생성자 — 부모 BufferState 포인터 저장
     *
     * @config: NoC 설정 객체
     * @parent: 소유자 BufferState (const 포인터로 저장)
     * @name:   모듈 이름 (예: "policy")
     */
    BufferPolicy(Configuration const & config, BufferState * parent,
		 const string & name);

    /*
     * [한국어]
     * SetMinLatency - 최소 레이턴시 설정 (Feedback 정책에서만 의미 있음)
     *
     * @min_latency: 측정된 최소 왕복 레이턴시 (사이클 단위)
     *
     * 기본 구현은 아무것도 하지 않는다 (가상 함수 빈 구현).
     * FeedbackSharedBufferPolicy에서 오버라이드하여 occupancy limit 계산에 활용.
     */
    virtual void SetMinLatency(int min_latency) {}

    /*
     * [한국어]
     * TakeBuffer - VC를 하나의 패킷 흐름에 할당할 때 호출 (가상)
     *
     * @vc: 할당할 VC 번호 (기본 0)
     *
     * 기본 구현은 아무것도 하지 않는다.
     * LimitedSharedBufferPolicy에서 오버라이드하여 _active_vcs를 증가시킨다.
     */
    virtual void TakeBuffer(int vc = 0);

    /*
     * [한국어]
     * SendingFlit - 플릿 한 개가 이 출력 포트로 전송될 때 호출 (가상)
     *
     * @f: 전송되는 플릿 포인터 (vc, tail, cl 필드 참조)
     *
     * 기본 구현은 아무것도 하지 않는다.
     * 각 구체 정책에서 오버라이드하여 private/shared buf occupancy 갱신.
     */
    virtual void SendingFlit(Flit const * const f);

    /*
     * [한국어]
     * FreeSlotFor - 다운스트림에서 슬롯 하나가 해제되었을 때 호출 (가상)
     *
     * @vc: 슬롯이 해제된 VC 번호 (기본 0)
     *
     * 기본 구현은 아무것도 하지 않는다.
     * SharedBufferPolicy에서 오버라이드하여 private/shared buf occupancy 감소.
     * ProcessCredit() → _buffer_policy->FreeSlotFor(vc)로 호출됨.
     */
    virtual void FreeSlotFor(int vc = 0);

    /*
     * [한국어]
     * IsFullFor - 지정 VC에 대해 다운스트림 버퍼가 가득 찼는지 반환 (순수 가상)
     *
     * @vc: 확인할 VC 번호 (기본 0)
     * @return: 가득 찼으면 true — 이 VC로 플릿 전송 불가
     */
    virtual bool IsFullFor(int vc = 0) const = 0;

    /*
     * [한국어]
     * AvailableFor - 지정 VC에 대해 사용 가능한 슬롯 수 반환 (순수 가상)
     *
     * @vc: 확인할 VC 번호 (기본 0)
     * @return: 사용 가능한 슬롯 수 (0 이상)
     */
    virtual int AvailableFor(int vc = 0) const = 0;

    /*
     * [한국어]
     * LimitFor - 지정 VC에 대한 최대 허용 슬롯 한도 반환 (순수 가상)
     *
     * @vc: 확인할 VC 번호 (기본 0)
     * @return: 이 VC가 사용 가능한 최대 슬롯 수
     */
    virtual int LimitFor(int vc = 0) const = 0;

    /*
     * [한국어]
     * New - 설정에 따라 적합한 BufferPolicy 구체 객체를 생성해 반환 (팩토리 메서드)
     *
     * @config: "buffer_policy" 문자열로 정책 종류 결정
     * @parent: 소유자 BufferState
     * @name:   모듈 이름
     * @return: 새로 생성된 BufferPolicy 파생 클래스 포인터
     *
     * "private" → PrivateBufferPolicy
     * "shared"  → SharedBufferPolicy
     * "limited" → LimitedSharedBufferPolicy
     * "dynamic" → DynamicLimitedSharedBufferPolicy
     * "shifting"→ ShiftingDynamicLimitedSharedBufferPolicy
     * "feedback"→ FeedbackSharedBufferPolicy
     * "simplefeedback" → SimpleFeedbackSharedBufferPolicy
     */
    static BufferPolicy * New(Configuration const & config,
			      BufferState * parent, const string & name);
  };

  /*
   * [한국어]
   * PrivateBufferPolicy - VC별 완전히 독립된 전용 버퍼 정책
   *
   * 각 VC가 독립적으로 _vc_buf_size 슬롯을 점유할 수 있다.
   * VC 간 버퍼 공유 없음 — 단순하고 예측 가능한 지연.
   * 설정: "buf_size" (양수이면 VC당 buf_size/num_vcs) 또는 "vc_buf_size" 직접 지정.
   */
  class PrivateBufferPolicy : public BufferPolicy {
  protected:
    int _vc_buf_size;
    /* VC 하나에 허용되는 최대 슬롯 수.
     * 설정자: 생성자에서 buf_size/num_vcs 또는 vc_buf_size로 결정.
     * 읽는 자: IsFullFor(), AvailableFor(), LimitFor(), SendingFlit() 오버플로 검사.
     * 값 범위: 양의 정수 (assert로 보장).
     * 동기화: 생성 후 변경 없음 — 읽기 전용. */
  public:
    PrivateBufferPolicy(Configuration const & config, BufferState * parent,
			const string & name);
    virtual void SendingFlit(Flit const * const f);
    virtual bool IsFullFor(int vc = 0) const;
    virtual int AvailableFor(int vc = 0) const;
    virtual int LimitFor(int vc = 0) const;
  };

  /*
   * [한국어]
   * SharedBufferPolicy - 전용(private) + 공유(shared) 혼합 버퍼 정책
   *
   * 각 VC(또는 VC 그룹)에 전용 영역(_private_buf_size)을 할당하고,
   * 전용 영역이 차면 공유 영역(_shared_buf_size)을 추가로 사용한다.
   * 설정: "private_bufs", "private_buf_size", "private_buf_start_vc", "private_buf_end_vc"
   * 공유 영역 = 전체(_buf_size) - 모든 전용 영역 합계.
   */
  class SharedBufferPolicy : public BufferPolicy {
  protected:
    int _buf_size;
    /* 버퍼 전체 슬롯 수 (전용 + 공유).
     * 설정자: 생성자에서 "buf_size" 또는 num_vcs * "vc_buf_size"로 결정.
     * 읽는 자: _shared_buf_size 계산, LimitFor(), AvailableFor().
     * 값 범위: 양의 정수.
     * 동기화: 생성 후 읽기 전용. */

    vector<int> _private_buf_vc_map;
    /* VC 번호 → 전용 버퍼 그룹 인덱스 매핑 테이블.
     * 설정자: 생성자에서 start_vc/end_vc 범위를 이용해 채워짐.
     * 읽는 자: SendingFlit(), FreeSlotFor(), IsFullFor(), AvailableFor(), LimitFor()가
     *          vc 번호로 해당 전용 버퍼 그룹을 찾을 때 사용.
     * 값 범위: 인덱스 0 이상 num_private_bufs 미만, 매핑 없는 VC는 -1.
     * 동기화: 생성 후 읽기 전용. */

    vector<int> _private_buf_size;
    /* 각 전용 버퍼 그룹의 슬롯 수.
     * 설정자: 생성자에서 "private_buf_size" 파라미터 배열로 결정.
     * 읽는 자: SharedBufferPolicy의 IsFullFor/ProcessFreeSlot/SendingFlit.
     * 값 범위: 각 원소 0 이상 _buf_size 이하.
     * 동기화: 생성 후 읽기 전용. */

    vector<int> _private_buf_occupancy;
    /* 각 전용 버퍼 그룹의 현재 점유 슬롯 수.
     * 설정자: SendingFlit()에서 증가, ProcessFreeSlot()에서 감소.
     * 읽는 자: IsFullFor(), ProcessFreeSlot()의 shared overflow 판단.
     * 값 범위: 0 이상 _private_buf_size[i] 이상도 가능(공유 영역 사용 시).
     * 동기화: 단일 스레드 — 불필요. */

    int _shared_buf_size;
    /* 공유 버퍼 영역의 총 슬롯 수 = _buf_size - 모든 전용 크기 합계.
     * 설정자: 생성자에서 계산(assert(_shared_buf_size >= 0)).
     * 읽는 자: IsFullFor(), AvailableFor(), LimitFor(), SendingFlit() 오버플로 검사.
     * 값 범위: 0 이상.
     * 동기화: 생성 후 읽기 전용. */

    int _shared_buf_occupancy;
    /* 현재 공유 버퍼 영역에서 사용 중인 슬롯 수.
     * 설정자: SendingFlit()에서 전용 영역 초과분 발생 시 증가,
     *         ProcessFreeSlot()에서 전용 영역이 포화였을 때 감소.
     * 읽는 자: IsFullFor(), AvailableFor() 에서 공유 영역 가용성 계산.
     * 값 범위: 0 이상 _shared_buf_size 이하.
     * 동기화: 단일 스레드 — 불필요. */

    vector<int> _reserved_slots;
    /* VC별 예약된 슬롯 수 — tail 이후 돌아오는 크레딧을 위한 예약.
     * 설정자: FreeSlotFor()에서 VC가 사용 중이면서 비어 있을 때 증가,
     *         SendingFlit()에서 tail 플릿 이후 잔여 예약 슬롯 해제 시 사용.
     * 읽는 자: IsFullFor()에서 _reserved_slots[vc] > 0이면 not full 판정,
     *          AvailableFor()에서 예약 슬롯도 가용으로 계산.
     * 값 범위: 0 이상 (이론적 상한 없음).
     * 동기화: 단일 스레드 — 불필요. */

    void ProcessFreeSlot(int vc = 0);
  public:
    SharedBufferPolicy(Configuration const & config, BufferState * parent,
		       const string & name);
    virtual void SendingFlit(Flit const * const f);
    virtual void FreeSlotFor(int vc = 0);
    virtual bool IsFullFor(int vc = 0) const;
    virtual int AvailableFor(int vc = 0) const;
    virtual int LimitFor(int vc = 0) const;
  };

  /*
   * [한국어]
   * LimitedSharedBufferPolicy - 활성 VC 수에 따른 최대 보유 슬롯 한도 정책
   *
   * SharedBufferPolicy를 상속하여, 추가로 VC당 _max_held_slots 한도를 적용한다.
   * 단일 VC가 전체 버퍼를 독점하지 못하도록 제한한다.
   * 설정: "max_held_slots" (음수이면 _buf_size 전체)
   */
  class LimitedSharedBufferPolicy : public SharedBufferPolicy {
  protected:
    int _vcs;
    /* 전체 VC 수.
     * 설정자: 생성자에서 "num_vcs"로 결정.
     * 읽는 자: TakeBuffer()의 _active_vcs 오버플로 검사.
     * 값 범위: 양의 정수.
     * 동기화: 생성 후 읽기 전용. */

    int _active_vcs;
    /* 현재 버퍼를 예약(TakeBuffer)한 VC 수 — 패킷 흐름이 진행 중인 VC 수.
     * 설정자: TakeBuffer()에서 증가, SendingFlit()에서 tail 플릿 처리 후 감소.
     * 읽는 자: DynamicLimitedSharedBufferPolicy의 _max_held_slots 재계산.
     * 값 범위: 0 이상 _vcs 이하.
     * 동기화: 단일 스레드 — 불필요. */

    int _max_held_slots;
    /* 단일 VC가 동시에 보유할 수 있는 최대 슬롯 수 한도.
     * 설정자: 생성자에서 "max_held_slots"로 결정(음수이면 _buf_size).
     *         DynamicLimited에서 _active_vcs 변화 시마다 재계산.
     * 읽는 자: IsFullFor(), AvailableFor(), LimitFor()에서 상위 정책 결과와 min() 취함.
     * 값 범위: 1 이상 _buf_size 이하 (assert로 보장).
     * 동기화: 단일 스레드 — 불필요. */
  public:
    LimitedSharedBufferPolicy(Configuration const & config,
			      BufferState * parent,
			      const string & name);
    virtual void TakeBuffer(int vc = 0);
    virtual void SendingFlit(Flit const * const f);
    virtual bool IsFullFor(int vc = 0) const;
    virtual int AvailableFor(int vc = 0) const;
    virtual int LimitFor(int vc = 0) const;
  };

  /*
   * [한국어]
   * DynamicLimitedSharedBufferPolicy - 활성 VC 수에 따라 _max_held_slots를 동적으로 조정
   *
   * LimitedSharedBufferPolicy를 상속. TakeBuffer/tail 처리 시마다
   * _max_held_slots = _buf_size / _active_vcs로 재계산한다.
   * 활성 VC가 적으면 더 많은 슬롯을 허용하고, 많으면 적게 허용한다.
   */
  class DynamicLimitedSharedBufferPolicy : public LimitedSharedBufferPolicy {
  public:
    DynamicLimitedSharedBufferPolicy(Configuration const & config,
				     BufferState * parent,
				     const string & name);
    virtual void TakeBuffer(int vc = 0);
    virtual void SendingFlit(Flit const * const f);
  };

  /*
   * [한국어]
   * ShiftingDynamicLimitedSharedBufferPolicy - 비트 시프트 기반 동적 슬롯 한도 정책
   *
   * DynamicLimitedSharedBufferPolicy를 상속. _active_vcs를 log2 비트 시프트로
   * 반올림하여 _max_held_slots를 2의 거듭제곱 값으로 유지한다.
   * 나눗셈 연산 없이 시프트 연산만으로 한도를 결정하는 최적화된 정책.
   */
  class ShiftingDynamicLimitedSharedBufferPolicy : public DynamicLimitedSharedBufferPolicy {
  public:
    ShiftingDynamicLimitedSharedBufferPolicy(Configuration const & config,
					     BufferState * parent,
					     const string & name);
    virtual void TakeBuffer(int vc = 0);
    virtual void SendingFlit(Flit const * const f);
  };

  /*
   * [한국어]
   * FeedbackSharedBufferPolicy - 왕복 지연(RTT) 측정 기반 적응형 버퍼 한도 정책
   *
   * SharedBufferPolicy를 상속. 플릿이 전송될 때 시각을 기록하고,
   * 크레딧이 돌아올 때 RTT(Round-Trip Time)를 계산한다.
   * RTT가 최소 RTT(_min_latency)보다 크면 occupancy limit을 줄여 혼잡을 완화한다.
   * 설정: "feedback_aging_scale", "feedback_offset"
   */
  class FeedbackSharedBufferPolicy : public SharedBufferPolicy {
  protected:
    /*
     * [한국어]
     * _ComputeRTT - RTT 이동 평균(moving average) 계산
     *
     * @vc: 계산할 VC 번호
     * @last_rtt: 가장 최근에 측정된 RTT 값 (사이클)
     * @return: 지수 이동 평균으로 갱신된 RTT 추정값
     *
     * rtt = ((rtt << _aging_scale) + last_rtt - rtt) >> _aging_scale
     * _aging_scale이 크면 과거 값에 더 많은 가중치를 두는 느린 이동 평균.
     */
    int _ComputeRTT(int vc, int last_rtt) const;

    /*
     * [한국어]
     * _ComputeLimit - 측정된 RTT로부터 occupancy limit 계산
     *
     * @rtt: 현재 RTT 추정값 (사이클)
     * @return: 새 occupancy limit = max(2*min_latency - rtt + offset, 1)
     *
     * RTT가 최소값(2*_min_latency)보다 클수록 limit가 줄어들어 흐름 억제.
     * _offset은 기준선 보정값.
     */
    int _ComputeLimit(int rtt) const;

    /*
     * [한국어]
     * _ComputeMaxSlots - VC의 현재 최대 허용 슬롯 수 계산
     *
     * @vc: 계산할 VC 번호
     * @return: 현재 _occupancy_limit[vc]와 RTT 기반 limit 중 최솟값
     *
     * 실시간 RTT 기반으로 슬롯 한도를 더 조일 수 있음.
     */
    int _ComputeMaxSlots(int vc) const;

    int _vcs;
    /* 전체 VC 수.
     * 설정자: 생성자에서 "num_vcs" 읽기.
     * 읽는 자: 초기화 시 벡터 크기 결정.
     * 값 범위: 양의 정수. 동기화: 읽기 전용. */

    vector<int> _occupancy_limit;
    /* VC별 현재 최대 허용 occupancy 슬롯 수.
     * 설정자: 생성자에서 _buf_size로 초기화; FreeSlotFor()에서 RTT 기반으로 갱신.
     * 읽는 자: _ComputeMaxSlots() — IsFullFor/AvailableFor 계산에 사용.
     * 값 범위: 1 이상 _buf_size 이하.
     * 동기화: 단일 스레드 — 불필요. */

    vector<int> _round_trip_time;
    /* VC별 현재 RTT 이동 평균 추정값 (사이클).
     * 설정자: 생성자에서 -1로 초기화(미측정); FreeSlotFor()에서 _ComputeRTT로 갱신.
     * 읽는 자: _ComputeRTT() — 이전 RTT 값과 새 측정값을 혼합.
     * 값 범위: -1(미초기화) 또는 1 이상.
     * 동기화: 단일 스레드 — 불필요. */

    vector<queue<int> > _flit_sent_time;
    /* VC별 플릿 전송 시각 기록 큐 — 각 원소는 해당 플릿이 전송된 사이클.
     * 설정자: SendingFlit()에서 GetSimTime()을 push.
     * 읽는 자: FreeSlotFor()에서 front()로 가장 오래된 전송 시각 꺼내 RTT 계산.
     *          _ComputeMaxSlots()에서 현재 진행 중인 RTT 실시간 추정에 사용.
     * 값 범위: 각 원소 0 이상 (시뮬레이션 시각).
     * 동기화: 단일 스레드 — 불필요. */

    int _min_latency;
    /* 측정된 최소 왕복 레이턴시 (사이클) — 혼잡 없는 이상적 RTT 기준선.
     * 설정자: SetMinLatency()로 외부에서 설정 (라우터가 최소 지연 관찰 후 업데이트).
     *         생성자에서 -1로 초기화 (미설정 상태).
     * 읽는 자: _ComputeLimit()에서 2*_min_latency 계산에 사용.
     * 값 범위: -1(미설정) 또는 양의 정수.
     * 동기화: 단일 스레드 — 불필요. */

    int _total_mapped_size;
    /* 모든 VC의 occupancy_limit 합계 — 전체 버퍼 매핑 공간 추적용.
     * 설정자: 생성자에서 _buf_size * _vcs로 초기화; FreeSlotFor()에서 limit 변화량 가산.
     * 읽는 자: 디버그 출력(DEBUG_FEEDBACK) — 통계 확인용.
     * 값 범위: 0 이상. 동기화: 단일 스레드 — 불필요. */

    int _aging_scale;
    /* RTT 이동 평균의 노화(aging) 스케일 파라미터 — 값이 클수록 과거 가중치 증가.
     * 설정자: 생성자에서 "feedback_aging_scale" 읽기.
     * 읽는 자: _ComputeRTT()에서 비트 시프트 연산에 사용.
     * 값 범위: 0 이상 (0이면 last_rtt 직접 사용). 동기화: 읽기 전용. */

    int _offset;
    /* occupancy limit 계산 시 보정 오프셋.
     * 설정자: 생성자에서 "feedback_offset" 읽기.
     * 읽는 자: _ComputeLimit()에서 기준선 보정: 2*min_latency - rtt + _offset.
     * 값 범위: 임의 정수 (음수 가능). 동기화: 읽기 전용. */

  public:
    FeedbackSharedBufferPolicy(Configuration const & config,
			       BufferState * parent, const string & name);
    virtual void SetMinLatency(int min_latency);
    virtual void SendingFlit(Flit const * const f);
    virtual void FreeSlotFor(int vc = 0);
    virtual bool IsFullFor(int vc = 0) const;
    virtual int AvailableFor(int vc = 0) const;
    virtual int LimitFor(int vc = 0) const;
  };

  /*
   * [한국어]
   * SimpleFeedbackSharedBufferPolicy - 단순화된 피드백 정책 (probe 플릿 방식)
   *
   * FeedbackSharedBufferPolicy를 상속. 매 플릿마다 RTT를 측정하지 않고,
   * 큐가 비어 있을 때 첫 번째 플릿만 "probe"로 삼아 RTT를 측정한다.
   * 나머지 플릿은 non-probe로 처리하여 불필요한 RTT 계산을 줄인다.
   */
  class SimpleFeedbackSharedBufferPolicy : public FeedbackSharedBufferPolicy {
  protected:
    vector<int> _pending_credits;
    /* VC별 아직 처리되지 않은 non-probe 크레딧 수.
     * 설정자: SendingFlit()에서 probe 플릿 전송 시 현재 occupancy-1로 초기화.
     * 읽는 자: FreeSlotFor()에서 _pending_credits[vc] > 0이면 non-probe 크레딧 처리.
     * 값 범위: 0 이상.
     * 동기화: 단일 스레드 — 불필요. */
  public:
    SimpleFeedbackSharedBufferPolicy(Configuration const & config,
				     BufferState * parent, const string & name);
    virtual void SendingFlit(Flit const * const f);
    virtual void FreeSlotFor(int vc = 0);
  };

  bool _wait_for_tail_credit;
  /* tail 플릿의 크레딧을 받을 때까지 VC를 사용 중으로 유지할지 여부.
   * 설정자: 생성자에서 config의 "wait_for_tail_credit"으로 결정.
   * 읽는 자: ProcessCredit()에서 VC 해제 시점 결정;
   *          SendingFlit()에서 tail 처리 후 VC 해제 시점 결정.
   * 값 범위: true(크레딧 대기) 또는 false(tail 전송 즉시 해제).
   * 동기화: 단일 스레드 — 불필요. */

  int  _size;
  /* 다운스트림 버퍼 전체 최대 슬롯 수 (크레딧 상한).
   * 설정자: 생성자에서 "buf_size" 또는 _vcs * "vc_buf_size"로 결정.
   * 읽는 자: IsFull(), 오버플로 검사(SendingFlit), BufferPolicy 생성 후 간접 사용.
   * 값 범위: 양의 정수. 동기화: 생성 후 읽기 전용. */

  int  _occupancy;
  /* 다운스트림 버퍼 전체의 현재 사용 중인 슬롯 수 추정값.
   * 설정자: SendingFlit()에서 ++, ProcessCredit()에서 --.
   * 읽는 자: IsFull(), Occupancy().
   * 값 범위: 0 이상 _size 이하. 동기화: 단일 스레드 — 불필요. */

  vector<int> _vc_occupancy;
  /* VC별 현재 사용 중인 슬롯 수 추정값.
   * 설정자: SendingFlit()에서 ++, ProcessCredit()에서 --.
   * 읽는 자: OccupancyFor(), IsEmptyFor(), BufferPolicy의 각 메서드.
   * 값 범위: 각 원소 0 이상. 동기화: 단일 스레드 — 불필요. */

  int  _vcs;
  /* 가상 채널(VC) 총 수.
   * 설정자: 생성자에서 "num_vcs"로 결정.
   * 읽는 자: 벡터 초기화, 범위 검사(assert), 루프.
   * 값 범위: 양의 정수. 동기화: 생성 후 읽기 전용. */

  BufferPolicy * _buffer_policy;
  /* 현재 사용 중인 버퍼 정책 객체 포인터 (전략 패턴).
   * 설정자: 생성자에서 BufferPolicy::New()가 반환한 구체 정책 객체 저장.
   * 읽는 자: IsFullFor/AvailableFor/LimitFor/SetMinLatency/TakeBuffer/SendingFlit/FreeSlotFor 위임.
   * 값 범위: 유효한 BufferPolicy 파생 클래스 포인터 (NULL 불가).
   * 동기화: 생성 후 읽기 전용. 소멸자에서 delete. */

  vector<int> _in_use_by;
  /* VC별 현재 사용 중인 패킷의 태그(tag) 값 (-1이면 사용 가능).
   * 설정자: TakeBuffer()에서 tag로 설정; ProcessCredit() 또는 SendingFlit()에서 -1로 해제.
   * 읽는 자: IsAvailableFor() — _in_use_by[vc] < 0이면 사용 가능.
   *          UsedBy() — 현재 점유 중인 tag 반환.
   * 값 범위: -1(사용 가능) 또는 0 이상 정수(패킷 식별 태그).
   * 동기화: 단일 스레드 — 불필요. */

  vector<bool> _tail_sent;
  /* VC별 tail 플릿 송신 여부 플래그.
   * 설정자: SendingFlit()에서 f->tail == true이면 true로 설정.
   *         TakeBuffer()에서 새 패킷 할당 시 false로 초기화.
   * 읽는 자: ProcessCredit()에서 _wait_for_tail_credit && !vc_occupancy && _tail_sent이면 VC 해제.
   * 값 범위: true 또는 false.
   * 동기화: 단일 스레드 — 불필요. */

  vector<int> _last_id;
  /* VC별 마지막으로 전송된 플릿의 고유 ID.
   * 설정자: SendingFlit()에서 f->id로 갱신.
   * 읽는 자: 디버그 출력 또는 상태 검사.
   * 값 범위: -1(초기값) 또는 0 이상 플릿 ID.
   * 동기화: 단일 스레드 — 불필요. */

  vector<int> _last_pid;
  /* VC별 마지막으로 전송된 플릿의 패킷 ID.
   * 설정자: SendingFlit()에서 f->pid로 갱신.
   * 읽는 자: 디버그 출력 또는 상태 검사.
   * 값 범위: -1(초기값) 또는 0 이상 패킷 ID.
   * 동기화: 단일 스레드 — 불필요. */

#ifdef TRACK_BUFFERS
  int _classes;
  /* 트래픽 클래스 수 (TRACK_BUFFERS 활성 시만 사용).
   * 설정자: 생성자에서 "classes" 읽기.
   * 읽는 자: _outstanding_classes 벡터 초기화, _class_occupancy 벡터 초기화.
   * 값 범위: 양의 정수. 동기화: 읽기 전용. */

  vector<queue<int> > _outstanding_classes;
  /* VC별 아직 크레딧이 돌아오지 않은 플릿들의 트래픽 클래스 큐.
   * 설정자: SendingFlit()에서 f->cl push; ProcessCredit()에서 front() pop.
   * 읽는 자: ProcessCredit()에서 크레딧 반환 시 해당 클래스 occupancy 감소.
   * 값 범위: 각 원소 0 이상 _classes 미만. 동기화: 단일 스레드. */

  vector<int> _class_occupancy;
  /* 클래스별 현재 다운스트림 버퍼 점유 수.
   * 설정자: SendingFlit()에서 ++, ProcessCredit()에서 --.
   * 읽는 자: OccupancyForClass() — 통계 수집.
   * 값 범위: 0 이상. 동기화: 단일 스레드. */
#endif

public:

  /*
   * [한국어]
   * BufferState 생성자 — 크기, VC 배열, 버퍼 정책 초기화
   *
   * @config: NoC 설정 객체 (num_vcs, buf_size, buffer_policy, wait_for_tail_credit 등)
   * @parent: 모듈 계층 부모 (라우터 객체)
   * @name:   모듈 이름 (예: "next_buf_0")
   */
  BufferState( const Configuration& config,
	       Module *parent, const string& name );

  /*
   * [한국어]
   * BufferState 소멸자 — 버퍼 정책 객체 해제
   */
  ~BufferState();

  /*
   * [한국어]
   * SetMinLatency - 최소 레이턴시를 버퍼 정책에 전달 (인라인)
   *
   * @min_latency: 측정된 최소 왕복 레이턴시 (사이클)
   *
   * Feedback 정책에서만 유효하며, 다른 정책에서는 빈 가상 함수로 처리됨.
   */
  inline void SetMinLatency(int min_latency) {
    _buffer_policy->SetMinLatency(min_latency); // [한국어] 정책에 최소 레이턴시 기준값 전달 (Feedback 정책에서 limit 계산에 사용)
  }

  /*
   * [한국어]
   * ProcessCredit - 다운스트림에서 받은 크레딧 패킷 처리
   *
   * @c: 수신된 크레딧 패킷 포인터 (set<int> vc 집합 포함)
   * @return: 없음
   *
   * 크레딧 패킷의 vc 집합 각각에 대해:
   *   - _occupancy, _vc_occupancy 감소
   *   - _wait_for_tail_credit && !vc_occupancy && tail_sent이면 VC 해제
   *   - _buffer_policy->FreeSlotFor(vc) 호출
   * 에러: _occupancy < 0 또는 _vc_occupancy[vc] < 0이면 Error() 호출.
   *
   * 호출 체인:
   *   IQRouter::ReadInputs() → [BufferState::ProcessCredit()] → BufferPolicy::FreeSlotFor()
   */
  void ProcessCredit( Credit const * const c );

  /*
   * [한국어]
   * SendingFlit - 다운스트림으로 플릿 전송 시 버퍼 상태 갱신
   *
   * @f: 전송되는 플릿 포인터 (vc, tail, cl, id, pid 필드 참조)
   * @return: 없음
   *
   * _occupancy, _vc_occupancy 증가, 정책의 SendingFlit() 호출.
   * f->tail이면 _tail_sent[vc] = true, !_wait_for_tail_credit이면 즉시 VC 해제.
   * TRACK_BUFFERS: 클래스 큐와 카운터 갱신.
   * 에러: _occupancy > _size이면 "Buffer overflow." 에러.
   *
   * 호출 체인:
   *   IQRouter::_SwitchTraversal() → [BufferState::SendingFlit()] → BufferPolicy::SendingFlit()
   */
  void SendingFlit( Flit const * const f );

  /*
   * [한국어]
   * TakeBuffer - 지정 VC를 특정 패킷(tag)에 전용 할당
   *
   * @vc:  할당할 VC 번호
   * @tag: 패킷을 식별하는 태그 값 (예: 패킷 ID 또는 입력 VC 번호)
   * @return: 없음
   *
   * _in_use_by[vc]가 이미 >= 0이면 "Buffer taken while in use" 에러.
   * 그렇지 않으면 _in_use_by[vc] = tag, _tail_sent[vc] = false로 초기화.
   * 정책의 TakeBuffer(vc)도 호출한다.
   *
   * 호출 체인:
   *   IQRouter::_VCAlloc() → [BufferState::TakeBuffer()] → BufferPolicy::TakeBuffer()
   */
  void TakeBuffer( int vc = 0, int tag = 0 );

  /*
   * [한국어]
   * IsFull - 다운스트림 버퍼 전체가 가득 찼는지 확인 (인라인)
   *
   * @return: _occupancy == _size이면 true (assert로 범위 검사)
   *
   * 전체 occupancy 기반 흐름 제어 판단. 특정 VC 무관.
   */
  inline bool IsFull() const {
    assert(_occupancy <= _size); // [한국어] occupancy가 size를 초과하지 않았는지 불변 조건 검사
    return (_occupancy == _size); // [한국어] 전체 버퍼 포화 상태이면 true
  }

  /*
   * [한국어]
   * IsFullFor - 지정 VC에 대해 버퍼 정책상 가득 찼는지 확인 (인라인)
   *
   * @vc: 확인할 VC 번호 (기본 0)
   * @return: 정책이 full이라고 판단하면 true
   *
   * VC할당기(VCAllocator)가 특정 VC에 플릿을 보낼 수 있는지 판단.
   */
  inline bool IsFullFor( int vc = 0 ) const {
    return _buffer_policy->IsFullFor(vc); // [한국어] 정책 객체에 지정 VC의 가득 찼는지 여부 위임
  }

  /*
   * [한국어]
   * AvailableFor - 지정 VC에 사용 가능한 슬롯 수 반환 (인라인)
   *
   * @vc: 확인할 VC 번호 (기본 0)
   * @return: 사용 가능한 슬롯 수 (0 이상)
   */
  inline int AvailableFor( int vc = 0 ) const {
    return _buffer_policy->AvailableFor(vc); // [한국어] 정책 객체에 지정 VC의 가용 슬롯 수 위임
  }

  /*
   * [한국어]
   * LimitFor - 지정 VC에 허용된 최대 슬롯 한도 반환 (인라인)
   *
   * @vc: 확인할 VC 번호 (기본 0)
   * @return: 최대 슬롯 한도
   */
  inline int LimitFor( int vc = 0 ) const {
    return _buffer_policy->LimitFor(vc); // [한국어] 정책 객체에 지정 VC의 슬롯 한도 위임
  }

  /*
   * [한국어]
   * IsEmptyFor - 지정 VC의 다운스트림 버퍼가 비어 있는지 확인 (인라인)
   *
   * @vc: 확인할 VC 번호 (기본 0)
   * @return: _vc_occupancy[vc] == 0이면 true
   *
   * SharedBufferPolicy에서 예약 슬롯 처리 시 사용.
   */
  inline bool IsEmptyFor(int vc = 0) const {
    assert((vc >= 0) && (vc < _vcs)); // [한국어] VC 번호 유효성 검사
    return (_vc_occupancy[vc] == 0);  // [한국어] 이 VC의 다운스트림 점유가 0이면 비어 있음
  }

  /*
   * [한국어]
   * IsAvailableFor - 지정 VC가 현재 유휴(사용 가능)한지 확인 (인라인)
   *
   * @vc: 확인할 VC 번호 (기본 0)
   * @return: _in_use_by[vc] < 0이면 true (현재 어떤 패킷도 할당되지 않음)
   *
   * VC 할당기가 미사용 VC를 선택할 때 사용.
   */
  inline bool IsAvailableFor( int vc = 0 ) const {
    assert( ( vc >= 0 ) && ( vc < _vcs ) ); // [한국어] VC 번호 유효성 검사
    return _in_use_by[vc] < 0; // [한국어] -1이면 이 VC는 현재 사용 가능한 상태
  }

  /*
   * [한국어]
   * UsedBy - 지정 VC를 현재 사용 중인 패킷의 태그 반환 (인라인)
   *
   * @vc: 확인할 VC 번호 (기본 0)
   * @return: 현재 VC를 점유 중인 패킷 태그 (-1이면 미사용)
   */
  inline int UsedBy(int vc = 0) const {
    assert( ( vc >= 0 ) && ( vc < _vcs ) ); // [한국어] VC 번호 유효성 검사
    return _in_use_by[vc]; // [한국어] 이 VC를 점유 중인 패킷 태그 반환 (-1이면 유휴)
  }

  /*
   * [한국어]
   * Occupancy - 다운스트림 버퍼 전체 점유 슬롯 수 반환 (인라인)
   *
   * @return: 현재 전체 occupancy
   */
  inline int Occupancy() const {
    return _occupancy; // [한국어] 다운스트림 버퍼 전체의 현재 점유 수 반환
  }

  /*
   * [한국어]
   * OccupancyFor - 지정 VC의 점유 슬롯 수 반환 (인라인)
   *
   * @vc: 확인할 VC 번호 (기본 0)
   * @return: 해당 VC의 현재 점유 수
   */
  inline int OccupancyFor( int vc = 0 ) const {
    assert((vc >= 0) && (vc < _vcs)); // [한국어] VC 번호 유효성 검사
    return _vc_occupancy[vc]; // [한국어] 이 VC의 다운스트림 점유 슬롯 수 반환
  }

#ifdef TRACK_BUFFERS
  /*
   * [한국어]
   * OccupancyForClass - 트래픽 클래스별 점유 수 반환 (인라인, TRACK_BUFFERS 한정)
   *
   * @c: 클래스 번호 (0 이상 _classes 미만)
   * @return: 해당 클래스의 현재 점유 수
   */
  inline int OccupancyForClass(int c) const {
    assert((c >= 0) && (c < _classes)); // [한국어] 클래스 번호 유효성 검사
    return _class_occupancy[c]; // [한국어] 해당 클래스의 현재 점유 수 반환
  }
#endif

  /*
   * [한국어]
   * Display - 현재 BufferState 전체 상태를 출력 스트림에 덤프 (디버그용)
   *
   * @os: 출력 스트림 (기본값 cout)
   * @return: 없음
   *
   * 전체 occupancy와 각 VC별 in_use_by, tail_sent, occupied 값을 출력한다.
   */
  void Display( ostream & os = cout ) const;
};

#endif
