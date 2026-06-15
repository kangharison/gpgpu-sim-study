// $Id: buffer_state.cpp 5188 2012-08-30 00:31:31Z dub $

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

/*buffer_state.cpp
 *
 * This class is the buffere state of the next router down the channel
 * tracks the credit and how much of the buffer is in use
 */

/*
 * [한국어 설명] 다운스트림 라우터 버퍼 상태 구현 (buffer_state.cpp)
 *
 * === 파일의 역할 ===
 * BufferState 클래스와 내부 BufferPolicy 계층 전체의 비인라인 멤버 함수를 구현한다.
 * 크레딧 기반 흐름 제어(credit-based flow control)의 송신 측 상태 머신으로서,
 * 업스트림 라우터가 다운스트림 라우터의 버퍼 가용 공간을 크레딧으로 추적한다.
 * 7가지 BufferPolicy 구체 클래스(private, shared, limited, dynamic, shifting,
 * feedback, simplefeedback)를 구현하여 다양한 버퍼 관리 전략을 제공한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * intersim2 NoC 시뮬레이터의 흐름 제어 계층.
 * 라우터의 각 출력 포트에 대응하는 BufferState 객체가 하나씩 존재하며,
 * 라우터의 ReadInputs() 단계에서 CreditChannel로부터 수신된 Credit 패킷이
 * ProcessCredit()을 통해 처리된다.
 * SA(스위치 할당) 완료 후 플릿이 전송될 때 SendingFlit()이 호출된다.
 * 실행 컨텍스트: 호스트 CPU 싱글 스레드, 매 사이클마다 호출됨.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - booksim.hpp:      Error() 매크로, 공통 타입 정의
 *   - buffer_state.hpp: BufferState 클래스 및 BufferPolicy 내부 클래스 선언
 *   - random_utils.hpp: RandomInt() 등 난수 유틸 (이 파일에서는 직접 미사용)
 *   - globals.hpp:      GetSimTime() — 현재 시뮬레이션 사이클 번호 반환
 * 이 파일을 사용하는 모듈:
 *   - iqrouter.cc, kncrouter.cc: ProcessCredit(), SendingFlit(), TakeBuffer() 직접 호출
 *   - allocator/ 계열: IsFullFor(), IsAvailableFor(), AvailableFor() 호출
 *
 * === 주요 함수/구조체 요약 ===
 * BufferPolicy::New(): 설정 "buffer_policy" 값에 따라 구체 정책 팩토리 생성
 * SharedBufferPolicy::ProcessFreeSlot(): 전용/공유 버퍼 점유 감소 로직 (FreeSlotFor 내부 호출)
 * SharedBufferPolicy::SendingFlit(): 예약 슬롯 처리 + 전용/공유 overflow 갱신
 * FeedbackSharedBufferPolicy: RTT 기반 동적 occupancy limit 조정
 * BufferState::ProcessCredit(): 크레딧 패킷 수신 처리 (occupancy 감소 + VC 해제)
 * BufferState::SendingFlit(): 플릿 전송 처리 (occupancy 증가 + tail 시 VC 해제 판단)
 * BufferState::TakeBuffer(): VC를 특정 패킷에 예약 할당
 */

#include <iostream>  // [한국어] cerr, cout 출력 — 디버그 메시지 및 에러 출력용
#include <sstream>   // [한국어] ostringstream — 에러 메시지 문자열 동적 생성용
#include <cstdlib>   // [한국어] exit() — 에러 발생 시 시뮬레이션 강제 종료
#include <cassert>   // [한국어] assert() — 불변 조건 검사 (디버그 모드에서 활성화)
#include <limits>    // [한국어] numeric_limits — 이론적으로 포함되나 직접 미사용

#include "booksim.hpp"       // [한국어] Error() 매크로, 공통 타입/헤더 포함
#include "buffer_state.hpp"  // [한국어] BufferState 및 모든 BufferPolicy 클래스 선언
#include "random_utils.hpp"  // [한국어] RandomInt() 등 — 이 파일에서 직접 사용되지 않으나 의존 포함
#include "globals.hpp"       // [한국어] GetSimTime() — 피드백 정책의 RTT 계산에 필요한 현재 사이클 반환

//#define DEBUG_FEEDBACK        // [한국어] Feedback 정책의 RTT/limit 변경을 cerr에 출력하는 디버그 매크로 (기본 비활성)
//#define DEBUG_SIMPLEFEEDBACK  // [한국어] SimpleFeedback 정책의 probe/non-probe 처리를 cerr에 출력 (기본 비활성)

/* =========================================================================
 * BufferPolicy (기반 클래스) 구현
 * ========================================================================= */

/*
 * [한국어]
 * BufferPolicy::BufferPolicy - BufferPolicy 기반 클래스 생성자
 *
 * @config: NoC 설정 객체 (파생 클래스에서 사용)
 * @parent: 소유자 BufferState 포인터 — _buffer_state const 포인터로 저장
 * @name:   이 정책 모듈의 이름 (예: "policy")
 * @return: 없음 (생성자)
 *
 * Module 기반 클래스 초기화와 _buffer_state 포인터 저장을 수행한다.
 * _buffer_state는 정책이 OccupancyFor() 등을 조회할 때 사용한다.
 *
 * 호출 체인:
 *   각 구체 정책 생성자 → [BufferPolicy::BufferPolicy()] → Module::Module()
 */
BufferState::BufferPolicy::BufferPolicy(Configuration const & config, BufferState * parent, const string & name)
: Module(parent, name), _buffer_state(parent) // [한국어] Module 계층 등록, 소유자 BufferState 포인터 저장
{
}

/*
 * [한국어]
 * BufferPolicy::TakeBuffer - VC 할당 이벤트 기본 처리 (아무 동작 없음)
 *
 * @vc: 할당된 VC 번호
 *
 * 기반 클래스의 기본 구현은 아무것도 하지 않는다.
 * LimitedSharedBufferPolicy에서 오버라이드하여 _active_vcs 증가.
 */
void BufferState::BufferPolicy::TakeBuffer(int vc) {
  // [한국어] 기본 구현: 아무 동작 없음 — 파생 클래스에서 필요 시 오버라이드
}

/*
 * [한국어]
 * BufferPolicy::SendingFlit - 플릿 전송 이벤트 기본 처리 (아무 동작 없음)
 *
 * @f: 전송되는 플릿 포인터
 *
 * 기반 클래스의 기본 구현은 아무것도 하지 않는다.
 * SharedBufferPolicy 이상에서 오버라이드하여 occupancy 갱신.
 */
void BufferState::BufferPolicy::SendingFlit(Flit const * const f) {
  // [한국어] 기본 구현: 아무 동작 없음 — 파생 클래스에서 필요 시 오버라이드
}

/*
 * [한국어]
 * BufferPolicy::FreeSlotFor - 슬롯 해제 이벤트 기본 처리 (아무 동작 없음)
 *
 * @vc: 슬롯이 해제된 VC 번호
 *
 * 기반 클래스의 기본 구현은 아무것도 하지 않는다.
 * SharedBufferPolicy에서 오버라이드하여 private/shared occupancy 감소.
 */
void BufferState::BufferPolicy::FreeSlotFor(int vc) {
  // [한국어] 기본 구현: 아무 동작 없음 — 파생 클래스에서 필요 시 오버라이드
}

/*
 * [한국어]
 * BufferPolicy::New - 설정에 따라 구체 BufferPolicy 객체 생성 (팩토리 메서드)
 *
 * @config: NoC 설정 — "buffer_policy" 문자열로 정책 종류 결정
 * @parent: 소유자 BufferState 포인터
 * @name:   정책 모듈 이름
 * @return: 생성된 구체 정책 포인터 (알 수 없는 정책이면 NULL 반환 후 cout 출력)
 *
 * 7가지 정책을 "buffer_policy" 문자열로 분기하여 생성한다.
 * 알 수 없는 정책 이름이면 에러 메시지 출력 후 NULL 반환.
 *
 * 호출 체인:
 *   BufferState::BufferState() → [BufferPolicy::New()] → 구체 정책 생성자
 */
BufferState::BufferPolicy * BufferState::BufferPolicy::New(Configuration const & config, BufferState * parent, const string & name)
{
  BufferPolicy * sp = NULL; // [한국어] 반환할 정책 포인터 초기화 (알 수 없는 정책이면 NULL 반환)
  string buffer_policy = config.GetStr("buffer_policy"); // [한국어] 설정에서 버퍼 정책 문자열 읽기
  if(buffer_policy == "private") { // [한국어] VC 전용 버퍼 정책 — VC별 독립 슬롯, 공유 없음
    sp = new PrivateBufferPolicy(config, parent, name);
  } else if(buffer_policy == "shared") { // [한국어] 전용+공유 혼합 버퍼 정책 — 전용 초과분은 공유 영역 사용
    sp = new SharedBufferPolicy(config, parent, name);
  } else if(buffer_policy == "limited") { // [한국어] 최대 보유 슬롯 고정 제한 정책
    sp = new LimitedSharedBufferPolicy(config, parent, name);
  } else if(buffer_policy == "dynamic") { // [한국어] 활성 VC 수에 따라 한도 동적 조정 정책
    sp = new DynamicLimitedSharedBufferPolicy(config, parent, name);
  } else if(buffer_policy == "shifting") { // [한국어] 비트 시프트 기반 동적 한도 정책 (2의 거듭제곱)
    sp = new ShiftingDynamicLimitedSharedBufferPolicy(config, parent, name);
  } else if(buffer_policy == "feedback") { // [한국어] RTT 기반 적응형 피드백 정책 — 모든 플릿 RTT 측정
    sp = new FeedbackSharedBufferPolicy(config, parent, name);
  } else if(buffer_policy == "simplefeedback") { // [한국어] probe 플릿만 RTT 측정하는 단순 피드백 정책
    sp = new SimpleFeedbackSharedBufferPolicy(config, parent, name);
  } else { // [한국어] 알 수 없는 정책 이름 — 에러 메시지 출력, NULL 반환으로 이후 크래시 유도
    cout << "Unknown buffer policy: " << buffer_policy << endl;
  }
  return sp; // [한국어] 생성된 정책 포인터 반환 (알 수 없는 경우 NULL)
}

/* =========================================================================
 * PrivateBufferPolicy 구현
 * ========================================================================= */

/*
 * [한국어]
 * PrivateBufferPolicy::PrivateBufferPolicy - VC 전용 버퍼 정책 생성자
 *
 * @config: "num_vcs", "buf_size", "vc_buf_size" 참조
 * @parent: 소유자 BufferState
 * @name:   모듈 이름
 * @return: 없음 (생성자)
 *
 * buf_size가 양수이면 VC당 슬롯 = buf_size / num_vcs,
 * 음수이면 "vc_buf_size"를 직접 사용한다.
 * _vc_buf_size > 0인지 assert로 보장한다.
 *
 * 호출 체인:
 *   BufferPolicy::New() → [PrivateBufferPolicy::PrivateBufferPolicy()] → BufferPolicy()
 */
BufferState::PrivateBufferPolicy::PrivateBufferPolicy(Configuration const & config, BufferState * parent, const string & name)
  : BufferPolicy(config, parent, name) // [한국어] 기반 클래스 생성자 호출 — Module 등록, _buffer_state 저장
{
  int const vcs = config.GetInt( "num_vcs" );  // [한국어] 전체 VC 수 읽기 — VC당 크기 분배에 사용
  int const buf_size = config.GetInt("buf_size"); // [한국어] 전체 버퍼 크기 읽기 — 양수이면 VC당 분배
  if(buf_size <= 0) { // [한국어] buf_size가 지정되지 않은 경우(음수 또는 0) — vc_buf_size로 직접 결정
    _vc_buf_size = config.GetInt("vc_buf_size"); // [한국어] VC당 전용 슬롯 수를 vc_buf_size로 직접 읽기
  } else { // [한국어] buf_size가 양수이면 균등 분배
    _vc_buf_size = buf_size / vcs; // [한국어] 전체 버퍼를 VC 수로 나누어 VC당 전용 슬롯 크기 결정
  }
  assert(_vc_buf_size > 0); // [한국어] VC당 슬롯이 최소 1 이상임을 보장 — 잘못된 설정 검출
}

/*
 * [한국어]
 * PrivateBufferPolicy::SendingFlit - 플릿 전송 시 VC별 버퍼 오버플로 검사
 *
 * @f: 전송되는 플릿 포인터 (vc 필드로 VC 번호 식별)
 * @return: 없음
 *
 * 다운스트림 VC의 occupancy가 _vc_buf_size를 초과했는지 검사한다.
 * 초과했으면 "Buffer overflow for VC N" 에러를 출력하고 종료한다.
 * 실제 occupancy 갱신은 BufferState::SendingFlit()이 이미 수행했으므로
 * 여기서는 검증만 한다.
 *
 * 호출 체인:
 *   BufferState::SendingFlit() → [PrivateBufferPolicy::SendingFlit()]
 */
void BufferState::PrivateBufferPolicy::SendingFlit(Flit const * const f)
{
  int const vc = f->vc; // [한국어] 플릿이 사용하는 VC 번호 — 해당 VC의 점유 수 검사에 사용
  if(_buffer_state->OccupancyFor(vc) > _vc_buf_size) { // [한국어] VC 점유가 허용 최대치 초과하면 오버플로
    ostringstream err; // [한국어] 에러 메시지 스트림 생성
    err << "Buffer overflow for VC " << vc; // [한국어] 어떤 VC에서 오버플로가 발생했는지 명시
    Error(err.str()); // [한국어] 에러 메시지 출력 후 시뮬레이션 종료
  }
}

/*
 * [한국어]
 * PrivateBufferPolicy::IsFullFor - 지정 VC의 전용 버퍼가 가득 찼는지 확인
 *
 * @vc: 확인할 VC 번호
 * @return: OccupancyFor(vc) >= _vc_buf_size이면 true
 *
 * VC 할당기가 이 VC에 새 플릿을 보낼 수 있는지 판단.
 *
 * 호출 체인:
 *   BufferState::IsFullFor() → [PrivateBufferPolicy::IsFullFor()]
 */
bool BufferState::PrivateBufferPolicy::IsFullFor(int vc) const
{
  return (_buffer_state->OccupancyFor(vc) >= _vc_buf_size); // [한국어] VC 점유가 전용 슬롯 한도에 도달했으면 가득 참
}

/*
 * [한국어]
 * PrivateBufferPolicy::AvailableFor - 지정 VC의 전용 버퍼 잔여 슬롯 수 반환
 *
 * @vc: 확인할 VC 번호
 * @return: _vc_buf_size - OccupancyFor(vc) (0 이상)
 *
 * 호출 체인:
 *   BufferState::AvailableFor() → [PrivateBufferPolicy::AvailableFor()]
 */
int BufferState::PrivateBufferPolicy::AvailableFor(int vc) const
{
  return _vc_buf_size - _buffer_state->OccupancyFor(vc); // [한국어] 전용 슬롯 한도에서 현재 점유를 뺀 잔여 공간
}

/*
 * [한국어]
 * PrivateBufferPolicy::LimitFor - 지정 VC의 최대 슬롯 한도 반환
 *
 * @vc: 확인할 VC 번호
 * @return: _vc_buf_size (모든 VC에 동일한 전용 한도)
 *
 * 호출 체인:
 *   BufferState::LimitFor() → [PrivateBufferPolicy::LimitFor()]
 */
int BufferState::PrivateBufferPolicy::LimitFor(int vc) const
{
  return _vc_buf_size; // [한국어] 모든 VC에 동일하게 적용되는 전용 슬롯 최대 한도
}

/* =========================================================================
 * SharedBufferPolicy 구현
 * ========================================================================= */

/*
 * [한국어]
 * SharedBufferPolicy::SharedBufferPolicy - 전용+공유 혼합 버퍼 정책 생성자
 *
 * @config: "num_vcs", "buf_size", "vc_buf_size", "private_bufs",
 *          "private_buf_size", "private_buf_start_vc", "private_buf_end_vc" 등
 * @parent: 소유자 BufferState
 * @name:   모듈 이름
 * @return: 없음 (생성자)
 *
 * 1. num_private_bufs(전용 버퍼 그룹 수) 결정
 * 2. 전체 버퍼 크기(_buf_size) 결정
 * 3. 각 그룹의 전용 크기(_private_buf_size) 결정
 * 4. start_vc/end_vc로 VC→그룹 매핑(_private_buf_vc_map) 설정
 * 5. 공유 영역 크기 = 전체 - 전용 합계
 * 6. _reserved_slots 초기화 (모두 0)
 *
 * 호출 체인:
 *   BufferPolicy::New() → [SharedBufferPolicy::SharedBufferPolicy()] → BufferPolicy()
 */
BufferState::SharedBufferPolicy::SharedBufferPolicy(Configuration const & config, BufferState * parent, const string & name)
  : BufferPolicy(config, parent, name), _shared_buf_occupancy(0) // [한국어] 기반 생성자 호출, 공유 영역 점유 0으로 초기화
{
  int const vcs = config.GetInt( "num_vcs" ); // [한국어] 전체 VC 수 읽기 — 매핑 벡터 크기 결정에 사용
  int num_private_bufs = config.GetInt("private_bufs"); // [한국어] 전용 버퍼 그룹 수 읽기 (-1이면 VC별 1개)
  if(num_private_bufs < 0) { // [한국어] -1이면 각 VC가 독립적인 전용 그룹을 가짐
    num_private_bufs = vcs; // [한국어] VC 수와 동일한 수의 전용 버퍼 그룹 생성
  } else if(num_private_bufs == 0) { // [한국어] 0이면 최소 1개의 전용 그룹 보장
    num_private_bufs = 1; // [한국어] 전용 그룹 0은 의미 없으므로 최소 1로 강제
  }

  _private_buf_occupancy.resize(num_private_bufs, 0); // [한국어] 전용 버퍼 그룹별 점유 수 벡터 초기화 (모두 0)

  _buf_size = config.GetInt("buf_size"); // [한국어] 전체 버퍼 크기 읽기 (음수이면 아래에서 재계산)
  if(_buf_size < 0) { // [한국어] buf_size가 지정되지 않은 경우 VC별 기본 크기로 계산
    _buf_size = vcs * config.GetInt("vc_buf_size"); // [한국어] 각 VC에 vc_buf_size 슬롯씩 할당한 합계
  }

  _private_buf_size = config.GetIntArray("private_buf_size"); // [한국어] 전용 그룹별 크기 배열 읽기 (빈 배열이면 스칼라 읽기)
  if(_private_buf_size.empty()) { // [한국어] 배열로 지정되지 않은 경우 스칼라 값 또는 균등 분배로 결정
    int const bs = config.GetInt("private_buf_size"); // [한국어] 스칼라 전용 크기 읽기 (-1이면 균등 분배)
    if(bs < 0) { // [한국어] 스칼라도 지정 안 됨 — 전체를 그룹 수로 균등 분배
      _private_buf_size.push_back(_buf_size / num_private_bufs); // [한국어] 균등 분배한 크기를 벡터에 추가
    } else { // [한국어] 스칼라 크기가 지정된 경우 — 모든 그룹에 동일 적용
      _private_buf_size.push_back(bs); // [한국어] 지정된 크기를 벡터에 추가
    }
  }
  _private_buf_size.resize(num_private_bufs, _private_buf_size.back()); // [한국어] 부족한 원소를 마지막 값으로 채워 그룹 수만큼 확장

  vector<int> start_vc = config.GetIntArray("private_buf_start_vc"); // [한국어] 각 전용 그룹의 시작 VC 배열 읽기
  if(start_vc.empty()) { // [한국어] 배열 미지정 — 스칼라 또는 균등 분배
    int const sv = config.GetInt("private_buf_start_vc"); // [한국어] 스칼라 시작 VC 읽기 (-1이면 균등 분배)
    if(sv < 0) { // [한국어] 자동 균등 분배 — i번 그룹의 시작 VC = i * vcs / num_private_bufs
      start_vc.resize(num_private_bufs); // [한국어] num_private_bufs 크기의 시작 VC 배열 할당
      for(int i = 0; i < num_private_bufs; ++i) {
	start_vc[i] = i * vcs / num_private_bufs; // [한국어] i번 그룹의 시작 VC 번호 균등 분배 계산
      }
    } else { // [한국어] 스칼라로 하나의 시작 VC만 지정된 경우
      start_vc.push_back(sv); // [한국어] 단일 시작 VC를 벡터에 추가
    }
  }

  vector<int> end_vc = config.GetIntArray("private_buf_end_vc"); // [한국어] 각 전용 그룹의 종료 VC 배열 읽기
  if(end_vc.empty()) { // [한국어] 배열 미지정 — 스칼라 또는 균등 분배
    int const ev = config.GetInt("private_buf_end_vc"); // [한국어] 스칼라 종료 VC 읽기 (-1이면 균등 분배)
    if(ev < 0) { // [한국어] 자동 균등 분배 — i번 그룹의 종료 VC = (i+1)*vcs/num_private_bufs - 1
      end_vc.resize(num_private_bufs); // [한국어] num_private_bufs 크기의 종료 VC 배열 할당
      for(int i = 0; i < num_private_bufs; ++i) {
	end_vc[i] = (i + 1) * vcs / num_private_bufs - 1; // [한국어] i번 그룹의 종료 VC 번호 균등 분배 계산
      }
    } else { // [한국어] 스칼라로 하나의 종료 VC만 지정된 경우
      end_vc.push_back(ev); // [한국어] 단일 종료 VC를 벡터에 추가
    }
  }

  _private_buf_vc_map.resize(vcs, -1); // [한국어] VC→그룹 매핑 벡터 초기화 (모두 -1: 미매핑)
  _shared_buf_size = _buf_size; // [한국어] 공유 영역 크기를 전체 크기에서 시작하여 전용 크기만큼 차감
  for(int i = 0; i < num_private_bufs; ++i) { // [한국어] 각 전용 버퍼 그룹에 대해 VC 매핑 수행
    _shared_buf_size -= _private_buf_size[i]; // [한국어] 전용 그룹 i의 크기만큼 공유 영역에서 차감
    assert(start_vc[i] <= end_vc[i]); // [한국어] 시작 VC가 종료 VC보다 작거나 같아야 함
    for(int v = start_vc[i]; v <= end_vc[i]; ++v) { // [한국어] 그룹 i에 속하는 모든 VC를 매핑
      assert(_private_buf_vc_map[v] < 0); // [한국어] VC가 이미 다른 그룹에 매핑되지 않아야 함 (중복 방지)
      _private_buf_vc_map[v] = i; // [한국어] VC v를 그룹 i에 매핑
    }
  }
  assert(_shared_buf_size >= 0); // [한국어] 전용 크기 합계가 전체를 초과하지 않아야 함

  _reserved_slots.resize(vcs, 0); // [한국어] VC별 예약 슬롯 카운터 초기화 (모두 0)
}

/*
 * [한국어]
 * SharedBufferPolicy::ProcessFreeSlot - 전용/공유 버퍼 점유 감소 로직 (내부 함수)
 *
 * @vc: 슬롯이 해제된 VC 번호
 * @return: 없음
 *
 * VC에 해당하는 전용 버퍼 그룹의 점유를 1 감소시킨다.
 * 감소 전 private occupancy가 private size 이상이었다면 (공유 영역을 사용 중이었다면)
 * 공유 버퍼 occupancy도 1 감소시킨다.
 * 에러: private 점유가 0 미만 또는 shared 점유가 0 미만이면 Error() 호출.
 *
 * 호출 체인:
 *   FreeSlotFor() → [ProcessFreeSlot()] 또는 SendingFlit()에서 tail 후 예약 소진 시
 */
void BufferState::SharedBufferPolicy::ProcessFreeSlot(int vc)
{
  int i = _private_buf_vc_map[vc]; // [한국어] VC v가 속한 전용 버퍼 그룹 인덱스 조회
  --_private_buf_occupancy[i]; // [한국어] 해당 전용 그룹의 점유 수 1 감소
  if(_private_buf_occupancy[i] < 0) { // [한국어] 점유가 음수가 되면 크레딧 처리 버그 의심
    ostringstream err;
    err << "Private buffer occupancy fell below zero for buffer " << i; // [한국어] 어떤 전용 버퍼 그룹에서 발생했는지 표시
    Error(err.str()); // [한국어] 에러 출력 후 종료
  } else if(_private_buf_occupancy[i] >= _private_buf_size[i]) {
    // [한국어] 감소 후에도 여전히 private 크기 이상이면 감소 전에 공유 영역도 사용 중이었음을 의미
    // (private_occupancy: size+1→size 감소는 공유 1슬롯 해제, size→size-1은 전용만 해제)
    --_shared_buf_occupancy; // [한국어] 공유 버퍼 점유 1 감소 — 공유 슬롯 하나가 반환됨
    if(_shared_buf_occupancy < 0) { // [한국어] 공유 점유가 음수가 되면 버그
      Error("Shared buffer occupancy fell below zero.");
    }
  }
}

/*
 * [한국어]
 * SharedBufferPolicy::SendingFlit - 플릿 전송 시 전용/공유 점유 갱신
 *
 * @f: 전송되는 플릿 포인터 (vc, tail 필드 참조)
 * @return: 없음
 *
 * 예약 슬롯(_reserved_slots[vc])이 있으면 먼저 소진한다 (실제 점유 증가 불필요).
 * 없으면 전용 버퍼 그룹 점유를 증가시키고, 전용 한도 초과 시 공유 점유도 증가.
 * tail 플릿이면 남은 예약 슬롯을 모두 ProcessFreeSlot()으로 해제한다.
 *
 * 호출 체인:
 *   BufferState::SendingFlit() → [SharedBufferPolicy::SendingFlit()]
 */
void BufferState::SharedBufferPolicy::SendingFlit(Flit const * const f)
{
  int const vc = f->vc; // [한국어] 이 플릿이 사용하는 VC 번호
  if(_reserved_slots[vc] > 0) { // [한국어] 이 VC에 예약 슬롯이 있으면 — tail 이후 돌아오는 크레딧을 위한 예약
    --_reserved_slots[vc]; // [한국어] 예약 슬롯 하나 소진 — 실제 private/shared 점유 증가는 하지 않음
  } else { // [한국어] 예약 슬롯 없음 — 실제 버퍼 점유 발생
    int i = _private_buf_vc_map[vc]; // [한국어] 이 VC가 속한 전용 버퍼 그룹 인덱스
    ++_private_buf_occupancy[i]; // [한국어] 전용 그룹 점유 1 증가
    if(_private_buf_occupancy[i] > _private_buf_size[i]) { // [한국어] 전용 한도 초과 — 공유 영역으로 넘침
      ++_shared_buf_occupancy; // [한국어] 공유 버퍼 점유 1 증가
      if(_shared_buf_occupancy > _shared_buf_size) { // [한국어] 공유 영역도 초과하면 전체 버퍼 오버플로
	Error("Shared buffer overflow."); // [한국어] 공유 버퍼 오버플로 에러 출력 후 종료
      }
    }
  }
  if(f->tail) { // [한국어] tail 플릿이면 이 패킷에 대한 점유가 종료되므로 남은 예약 슬롯 모두 정리
    while(_reserved_slots[vc]) { // [한국어] 남은 예약 슬롯이 있는 동안 반복
      --_reserved_slots[vc]; // [한국어] 예약 슬롯 하나 소진
      ProcessFreeSlot(vc);   // [한국어] 해당 슬롯을 실제로 해제하여 전용/공유 occupancy 감소
    }
  }
}

/*
 * [한국어]
 * SharedBufferPolicy::FreeSlotFor - 크레딧 수신 시 전용/공유 슬롯 해제
 *
 * @vc: 슬롯이 해제된 VC 번호
 * @return: 없음
 *
 * VC가 사용 중(!IsAvailableFor)이면서 비어 있는(IsEmptyFor) 경우:
 *   tail 이후 아직 VC가 해제되지 않은 상태에서 돌아오는 크레딧이므로
 *   _reserved_slots[vc]를 증가시켜 예약해 둔다.
 * 그 외 경우:
 *   ProcessFreeSlot(vc)를 직접 호출해 점유를 감소시킨다.
 *
 * 호출 체인:
 *   BufferState::ProcessCredit() → BufferPolicy::FreeSlotFor() → [SharedBufferPolicy::FreeSlotFor()]
 */
void BufferState::SharedBufferPolicy::FreeSlotFor(int vc)
{
  if(!_buffer_state->IsAvailableFor(vc) && _buffer_state->IsEmptyFor(vc)) {
    // [한국어] VC가 아직 점유 중이지만 플릿이 없음 — tail 이후 wait_for_tail_credit 모드의 크레딧
    // 이 경우 나중에 SendingFlit()의 tail 처리에서 소진되도록 예약 슬롯으로 관리
    ++_reserved_slots[vc]; // [한국어] 예약 슬롯 1 증가 — 이 크레딧은 바로 해제하지 않고 예약
  } else { // [한국어] 정상적인 크레딧 반환 — 즉시 점유 감소 처리
    ProcessFreeSlot(vc); // [한국어] 전용/공유 버퍼 점유 감소
  }
}

/*
 * [한국어]
 * SharedBufferPolicy::IsFullFor - 지정 VC의 전용+공유 버퍼가 가득 찼는지 확인
 *
 * @vc: 확인할 VC 번호
 * @return: 예약 슬롯도 없고 전용 한도에 도달했으며 공유도 포화 상태이면 true
 *
 * 조건: _reserved_slots[vc] == 0
 *    && _private_buf_occupancy[i] >= _private_buf_size[i]
 *    && _shared_buf_occupancy >= _shared_buf_size
 *
 * 호출 체인:
 *   BufferState::IsFullFor() → [SharedBufferPolicy::IsFullFor()]
 */
bool BufferState::SharedBufferPolicy::IsFullFor(int vc) const
{
  int i = _private_buf_vc_map[vc]; // [한국어] VC에 해당하는 전용 버퍼 그룹 인덱스
  return ((_reserved_slots[vc] == 0) && // [한국어] 예약 슬롯이 없고 (예약 슬롯이 있으면 사용 가능)
	  (_private_buf_occupancy[i] >= _private_buf_size[i]) && // [한국어] 전용 영역이 포화 상태이며
	  (_shared_buf_occupancy >= _shared_buf_size)); // [한국어] 공유 영역도 포화 상태이면 완전히 가득 참
}

/*
 * [한국어]
 * SharedBufferPolicy::AvailableFor - 지정 VC의 사용 가능한 슬롯 수 계산
 *
 * @vc: 확인할 VC 번호
 * @return: 예약 슬롯 + 전용 잔여 + 공유 잔여 합계
 *
 * 공식: _reserved_slots[vc]
 *     + max(_private_buf_size[i] - _private_buf_occupancy[i], 0)
 *     + (_shared_buf_size - _shared_buf_occupancy)
 *
 * 호출 체인:
 *   BufferState::AvailableFor() → [SharedBufferPolicy::AvailableFor()]
 */
int BufferState::SharedBufferPolicy::AvailableFor(int vc) const
{
  int i = _private_buf_vc_map[vc]; // [한국어] VC에 해당하는 전용 버퍼 그룹 인덱스
  return (_reserved_slots[vc] + // [한국어] 예약된 슬롯도 가용으로 계산
	  max(_private_buf_size[i] - _private_buf_occupancy[i], 0) + // [한국어] 전용 잔여 슬롯 (음수 방지를 위해 max 0 적용)
	  (_shared_buf_size - _shared_buf_occupancy)); // [한국어] 공유 영역 잔여 슬롯
}

/*
 * [한국어]
 * SharedBufferPolicy::LimitFor - 지정 VC의 최대 슬롯 한도 반환
 *
 * @vc: 확인할 VC 번호
 * @return: 전용 크기 + 공유 크기 (VC가 이론상 최대로 사용할 수 있는 슬롯)
 *
 * 호출 체인:
 *   BufferState::LimitFor() → [SharedBufferPolicy::LimitFor()]
 */
int BufferState::SharedBufferPolicy::LimitFor(int vc) const
{
  int i = _private_buf_vc_map[vc]; // [한국어] VC에 해당하는 전용 버퍼 그룹 인덱스
  return (_private_buf_size[i] + _shared_buf_size); // [한국어] 전용 슬롯 한도 + 공유 전체를 사용할 수 있는 이론적 최대값
}

/* =========================================================================
 * LimitedSharedBufferPolicy 구현
 * ========================================================================= */

/*
 * [한국어]
 * LimitedSharedBufferPolicy::LimitedSharedBufferPolicy - 최대 보유 슬롯 제한 정책 생성자
 *
 * @config: "num_vcs", "max_held_slots" 참조
 * @parent: 소유자 BufferState
 * @name:   모듈 이름
 * @return: 없음 (생성자)
 *
 * SharedBufferPolicy 생성자를 호출한 뒤 _vcs, _max_held_slots를 추가 초기화.
 * _max_held_slots < 0이면 _buf_size(전체 버퍼)로 설정.
 * _active_vcs는 0으로 초기화.
 *
 * 호출 체인:
 *   BufferPolicy::New() → [LimitedSharedBufferPolicy::LimitedSharedBufferPolicy()] → SharedBufferPolicy()
 */
BufferState::LimitedSharedBufferPolicy::LimitedSharedBufferPolicy(Configuration const & config, BufferState * parent, const string & name)
  : SharedBufferPolicy(config, parent, name), _active_vcs(0) // [한국어] SharedBufferPolicy 생성자 호출, 활성 VC 수 0으로 초기화
{
  _vcs = config.GetInt("num_vcs");          // [한국어] 전체 VC 수 — TakeBuffer()의 _active_vcs 오버플로 검사에 사용
  _max_held_slots = config.GetInt("max_held_slots"); // [한국어] 단일 VC가 보유 가능한 최대 슬롯 한도 읽기
  if(_max_held_slots < 0) { // [한국어] 지정되지 않은 경우(-1) — 전체 버퍼를 최대 한도로 설정 (사실상 제한 없음)
    _max_held_slots = _buf_size; // [한국어] 전체 버퍼 크기를 최대 한도로 사용
  }
}

/*
 * [한국어]
 * LimitedSharedBufferPolicy::TakeBuffer - VC 할당 시 활성 VC 수 증가
 *
 * @vc: 할당된 VC 번호
 * @return: 없음
 *
 * _active_vcs를 1 증가시키고 _vcs를 초과하면 에러.
 * DynamicLimited/ShiftingDynamic 파생 클래스에서 이를 오버라이드해
 * _max_held_slots를 재계산한다.
 *
 * 호출 체인:
 *   BufferState::TakeBuffer() → BufferPolicy::TakeBuffer() → [LimitedSharedBufferPolicy::TakeBuffer()]
 */
void BufferState::LimitedSharedBufferPolicy::TakeBuffer(int vc)
{
  ++_active_vcs; // [한국어] 새 VC가 버퍼를 예약했으므로 활성 VC 수 1 증가
  if(_active_vcs > _vcs) { // [한국어] 활성 VC 수가 전체 VC 수를 초과하면 논리 오류
    Error("Number of active VCs is too large."); // [한국어] 활성 VC 오버플로 에러 — 알고리즘 버그
  }
}

/*
 * [한국어]
 * LimitedSharedBufferPolicy::SendingFlit - 플릿 전송 시 tail이면 활성 VC 수 감소
 *
 * @f: 전송되는 플릿 포인터 (tail 필드 참조)
 * @return: 없음
 *
 * 상위 SharedBufferPolicy::SendingFlit()을 먼저 호출한 뒤,
 * f->tail이면 _active_vcs를 1 감소시킨다.
 * 감소 후 _active_vcs < 0이면 에러.
 *
 * 호출 체인:
 *   BufferState::SendingFlit() → [LimitedSharedBufferPolicy::SendingFlit()] → SharedBufferPolicy::SendingFlit()
 */
void BufferState::LimitedSharedBufferPolicy::SendingFlit(Flit const * const f)
{
  SharedBufferPolicy::SendingFlit(f); // [한국어] 상위 정책의 전용/공유 점유 갱신 먼저 수행
  if(f->tail) { // [한국어] tail 플릿이면 이 패킷의 마지막 플릿 — VC 사용 종료
    --_active_vcs; // [한국어] 활성 VC 수 1 감소 (이 패킷에 대한 VC 사용 완료)
    if(_active_vcs < 0) { // [한국어] 활성 VC가 음수이면 논리 오류
      Error("Number of active VCs fell below zero."); // [한국어] 활성 VC 언더플로 에러
    }
  }
}

/*
 * [한국어]
 * LimitedSharedBufferPolicy::IsFullFor - 공유 버퍼 포화 또는 최대 한도 초과 시 full 반환
 *
 * @vc: 확인할 VC 번호
 * @return: 공유 정책상 full이거나 VC 점유 >= _max_held_slots이면 true
 *
 * 호출 체인:
 *   BufferState::IsFullFor() → [LimitedSharedBufferPolicy::IsFullFor()] → SharedBufferPolicy::IsFullFor()
 */
bool BufferState::LimitedSharedBufferPolicy::IsFullFor(int vc) const
{
  return (SharedBufferPolicy::IsFullFor(vc) || // [한국어] 상위 정책(shared)이 이미 포화 상태이거나
	  (_buffer_state->OccupancyFor(vc) >= _max_held_slots)); // [한국어] 이 VC의 점유가 단일 VC 최대 한도에 도달했으면 full
}

/*
 * [한국어]
 * LimitedSharedBufferPolicy::AvailableFor - 공유 가용량과 VC 한도 잔여 중 최솟값 반환
 *
 * @vc: 확인할 VC 번호
 * @return: min(SharedBufferPolicy::AvailableFor(vc), _max_held_slots - OccupancyFor(vc))
 *
 * 호출 체인:
 *   BufferState::AvailableFor() → [LimitedSharedBufferPolicy::AvailableFor()]
 */
int BufferState::LimitedSharedBufferPolicy::AvailableFor(int vc) const
{
  return min(SharedBufferPolicy::AvailableFor(vc), // [한국어] 공유 정책상 가용 슬롯과
	     _max_held_slots - _buffer_state->OccupancyFor(vc)); // [한국어] VC 최대 한도 잔여 중 더 제한적인 값
}

/*
 * [한국어]
 * LimitedSharedBufferPolicy::LimitFor - 공유 한도와 VC 최대 한도 중 최솟값 반환
 *
 * @vc: 확인할 VC 번호
 * @return: min(SharedBufferPolicy::LimitFor(vc), _max_held_slots)
 *
 * 호출 체인:
 *   BufferState::LimitFor() → [LimitedSharedBufferPolicy::LimitFor()]
 */
int BufferState::LimitedSharedBufferPolicy::LimitFor(int vc) const
{
  return min(SharedBufferPolicy::LimitFor(vc), _max_held_slots); // [한국어] 공유 이론적 한도와 VC 최대 한도 중 더 작은 값
}

/* =========================================================================
 * DynamicLimitedSharedBufferPolicy 구현
 * ========================================================================= */

/*
 * [한국어]
 * DynamicLimitedSharedBufferPolicy::DynamicLimitedSharedBufferPolicy - 동적 한도 정책 생성자
 *
 * @config, @parent, @name: 상위 LimitedSharedBufferPolicy로 전달
 * @return: 없음 (생성자)
 *
 * _max_held_slots를 _buf_size 전체로 초기화한다.
 * 이후 TakeBuffer() 호출 시마다 _active_vcs에 따라 재계산된다.
 *
 * 호출 체인:
 *   BufferPolicy::New() → [DynamicLimitedSharedBufferPolicy::DynamicLimitedSharedBufferPolicy()] → LimitedSharedBufferPolicy()
 */
BufferState::DynamicLimitedSharedBufferPolicy::DynamicLimitedSharedBufferPolicy(Configuration const & config, BufferState * parent, const string & name)
  : LimitedSharedBufferPolicy(config, parent, name) // [한국어] LimitedShared 생성자 호출 — _vcs, _active_vcs, _max_held_slots 초기화
{
  _max_held_slots = _buf_size; // [한국어] 초기에는 전체 버퍼를 한도로 설정 — TakeBuffer()에서 동적 갱신됨
}

/*
 * [한국어]
 * DynamicLimitedSharedBufferPolicy::TakeBuffer - VC 할당 시 _max_held_slots 동적 재계산
 *
 * @vc: 할당된 VC 번호
 * @return: 없음
 *
 * 상위 LimitedShared::TakeBuffer()로 _active_vcs를 증가시킨 뒤,
 * _max_held_slots = _buf_size / _active_vcs로 재계산한다.
 * 활성 VC가 늘수록 VC당 허용 슬롯이 줄어든다.
 *
 * 호출 체인:
 *   BufferState::TakeBuffer() → [DynamicLimitedSharedBufferPolicy::TakeBuffer()] → LimitedSharedBufferPolicy::TakeBuffer()
 */
void BufferState::DynamicLimitedSharedBufferPolicy::TakeBuffer(int vc)
{
  LimitedSharedBufferPolicy::TakeBuffer(vc); // [한국어] _active_vcs 증가 및 오버플로 검사
  assert(_active_vcs > 0); // [한국어] 분모 0 방지 — 직전 호출로 _active_vcs가 1 이상임을 보장
  _max_held_slots = _buf_size / _active_vcs; // [한국어] 전체 버퍼를 현재 활성 VC 수로 균등 분배하여 최대 한도 재계산
  assert(_max_held_slots > 0); // [한국어] 한도가 0보다 커야 함 — buf_size < active_vcs이면 버그
}

/*
 * [한국어]
 * DynamicLimitedSharedBufferPolicy::SendingFlit - tail 처리 후 _max_held_slots 재계산
 *
 * @f: 전송되는 플릿 포인터 (tail 필드 참조)
 * @return: 없음
 *
 * 상위 LimitedShared::SendingFlit()을 호출(tail이면 _active_vcs 감소)한 뒤,
 * tail이고 _active_vcs > 0이면 _max_held_slots를 재계산한다.
 *
 * 호출 체인:
 *   BufferState::SendingFlit() → [DynamicLimitedSharedBufferPolicy::SendingFlit()] → LimitedSharedBufferPolicy::SendingFlit()
 */
void BufferState::DynamicLimitedSharedBufferPolicy::SendingFlit(Flit const * const f)
{
  LimitedSharedBufferPolicy::SendingFlit(f); // [한국어] 공유 점유 갱신 + tail이면 _active_vcs 감소
  if(f->tail && _active_vcs) { // [한국어] tail이고 아직 활성 VC가 남아 있으면 최대 한도 재계산
    _max_held_slots = _buf_size / _active_vcs; // [한국어] 줄어든 활성 VC 수로 VC당 허용 슬롯 재계산
  }
  assert(_max_held_slots > 0); // [한국어] 최대 한도가 여전히 양수여야 함
}

/* =========================================================================
 * ShiftingDynamicLimitedSharedBufferPolicy 구현
 * ========================================================================= */

/*
 * [한국어]
 * ShiftingDynamicLimitedSharedBufferPolicy::ShiftingDynamicLimitedSharedBufferPolicy - 비트 시프트 동적 한도 정책 생성자
 *
 * @config, @parent, @name: 상위 DynamicLimited 생성자로 전달
 * @return: 없음 (생성자)
 *
 * DynamicLimited 생성자를 그대로 호출하며, 추가 초기화 없음.
 * TakeBuffer/SendingFlit에서 비트 시프트 기반 _max_held_slots 계산을 사용한다.
 *
 * 호출 체인:
 *   BufferPolicy::New() → [ShiftingDynamicLimitedSharedBufferPolicy()] → DynamicLimited()
 */
BufferState::ShiftingDynamicLimitedSharedBufferPolicy::ShiftingDynamicLimitedSharedBufferPolicy(Configuration const & config, BufferState * parent, const string & name)
  : DynamicLimitedSharedBufferPolicy(config, parent, name) // [한국어] DynamicLimited 생성자 호출 — 추가 초기화 없음
{

}

/*
 * [한국어]
 * ShiftingDynamicLimitedSharedBufferPolicy::TakeBuffer - 비트 시프트 기반 _max_held_slots 설정
 *
 * @vc: 할당된 VC 번호
 * @return: 없음
 *
 * LimitedShared::TakeBuffer()로 _active_vcs를 증가시킨 뒤,
 * i = _active_vcs - 1로 시작해 i를 1비트 우시프트해 가며
 * _max_held_slots도 1비트 우시프트한다.
 * 결과: _max_held_slots = _buf_size >> floor(log2(_active_vcs))
 *   → 활성 VC 수를 2의 거듭제곱으로 올림하여 한도를 2의 거듭제곱 단위로 설정.
 *
 * 호출 체인:
 *   BufferState::TakeBuffer() → [ShiftingDynamic::TakeBuffer()] → LimitedShared::TakeBuffer()
 */
void BufferState::ShiftingDynamicLimitedSharedBufferPolicy::TakeBuffer(int vc)
{
  LimitedSharedBufferPolicy::TakeBuffer(vc); // [한국어] _active_vcs 증가 (Dynamic 아닌 Limited 직접 호출)
  assert(_active_vcs); // [한국어] _active_vcs > 0인지 보장 — 분모 0 방지
  int i = _active_vcs - 1; // [한국어] 비트 시프트 루프의 초기값: _active_vcs - 1 (0이면 루프 안 돌아 _max_held_slots = _buf_size)
  _max_held_slots = _buf_size; // [한국어] 최대 한도를 전체 버퍼에서 시작하여 시프트로 줄임
  while(i) { // [한국어] i가 0이 될 때까지 반복 — i와 _max_held_slots를 동시에 1비트 우시프트
    _max_held_slots >>= 1; // [한국어] 최대 한도를 절반으로 줄임 (2로 나눔)
    i >>= 1;               // [한국어] i도 절반으로 줄여 루프 카운터 역할
  }
  assert(_max_held_slots > 0); // [한국어] 시프트 결과가 1 이상이어야 함 (buf_size가 0이면 오류)
}

/*
 * [한국어]
 * ShiftingDynamicLimitedSharedBufferPolicy::SendingFlit - tail 처리 후 비트 시프트로 한도 재계산
 *
 * @f: 전송되는 플릿 포인터 (tail 필드 참조)
 * @return: 없음
 *
 * LimitedShared::SendingFlit()으로 tail이면 _active_vcs를 감소시킨 뒤,
 * tail이고 _active_vcs > 0이면 TakeBuffer()와 동일한 비트 시프트 로직으로 재계산.
 *
 * 호출 체인:
 *   BufferState::SendingFlit() → [ShiftingDynamic::SendingFlit()] → LimitedShared::SendingFlit()
 */
void BufferState::ShiftingDynamicLimitedSharedBufferPolicy::SendingFlit(Flit const * const f)
{
  LimitedSharedBufferPolicy::SendingFlit(f); // [한국어] 공유 점유 갱신 + tail이면 _active_vcs 감소
  if(f->tail && _active_vcs) { // [한국어] tail이고 아직 활성 VC가 남아 있으면 비트 시프트로 재계산
    int i = _active_vcs - 1; // [한국어] 비트 시프트 루프 초기값
    _max_held_slots = _buf_size; // [한국어] 전체 버퍼에서 시작
    while(i) { // [한국어] floor(log2(_active_vcs))회 시프트
      _max_held_slots >>= 1; // [한국어] 최대 한도를 절반으로
      i >>= 1;               // [한국어] 루프 카운터도 절반으로
    }
  }
  assert(_max_held_slots > 0); // [한국어] 재계산 후 최대 한도가 양수임을 보장
}

/* =========================================================================
 * FeedbackSharedBufferPolicy 구현
 * ========================================================================= */

/*
 * [한국어]
 * FeedbackSharedBufferPolicy::FeedbackSharedBufferPolicy - RTT 기반 피드백 정책 생성자
 *
 * @config: "feedback_aging_scale", "feedback_offset", "num_vcs" 참조
 * @parent: 소유자 BufferState
 * @name:   모듈 이름
 * @return: 없음 (생성자)
 *
 * SharedBufferPolicy 생성자 호출 후 RTT 관련 벡터 초기화:
 *   - _occupancy_limit: 모두 _buf_size (초기에는 최대 허용)
 *   - _round_trip_time: 모두 -1 (미측정)
 *   - _flit_sent_time: 빈 큐로 초기화
 *   - _total_mapped_size = _buf_size * _vcs
 *   - _min_latency = -1 (미설정)
 *
 * 호출 체인:
 *   BufferPolicy::New() → [FeedbackSharedBufferPolicy()] → SharedBufferPolicy()
 */
BufferState::FeedbackSharedBufferPolicy::FeedbackSharedBufferPolicy(Configuration const & config, BufferState * parent, const string & name)
  : SharedBufferPolicy(config, parent, name) // [한국어] SharedBufferPolicy 생성자 호출 — private/shared 구조 초기화
{
  _aging_scale = config.GetInt("feedback_aging_scale"); // [한국어] RTT 이동 평균의 노화 스케일 읽기 (값이 클수록 느린 평균)
  _offset = config.GetInt("feedback_offset");           // [한국어] limit 계산 보정값 읽기 — 2*min_latency - rtt + _offset
  _vcs = config.GetInt("num_vcs");                     // [한국어] 전체 VC 수 읽기 — 벡터 크기 결정

  _occupancy_limit.resize(_vcs, _buf_size); // [한국어] 모든 VC의 점유 한도를 전체 버퍼로 초기화 (RTT 측정 시작 전 최대 허용)
  _round_trip_time.resize(_vcs, -1);       // [한국어] 모든 VC의 RTT 추정값 -1로 초기화 (미측정 상태)
  _flit_sent_time.resize(_vcs);            // [한국어] VC별 플릿 전송 시각 큐 초기화 (모두 빈 큐)
  _total_mapped_size = _buf_size * _vcs;   // [한국어] 전체 매핑 크기 초기화 = 모든 VC에 최대 할당된 합계
  _min_latency = -1;                       // [한국어] 최소 레이턴시 미설정 상태 (-1) — SetMinLatency()로 설정됨
}

/*
 * [한국어]
 * FeedbackSharedBufferPolicy::SetMinLatency - 최소 왕복 레이턴시 설정
 *
 * @min_latency: 라우터가 측정한 최소 왕복 레이턴시 (사이클)
 * @return: 없음
 *
 * _ComputeLimit()의 기준선으로 사용될 _min_latency를 갱신한다.
 * DEBUG_FEEDBACK 활성 시 cerr에 새 값 출력.
 *
 * 호출 체인:
 *   BufferState::SetMinLatency() → [FeedbackSharedBufferPolicy::SetMinLatency()]
 */
void BufferState::FeedbackSharedBufferPolicy::SetMinLatency(int min_latency)
{
#ifdef DEBUG_FEEDBACK
  cerr << FullName() << ": Setting minimum latency to "
       << min_latency << "." << endl; // [한국어] 최소 레이턴시 변경 사실을 디버그 출력
#endif
  _min_latency = min_latency; // [한국어] 기준 최소 RTT 갱신 — 이후 _ComputeLimit()에서 참조됨
}

/*
 * [한국어]
 * FeedbackSharedBufferPolicy::SendingFlit - 플릿 전송 시각 기록
 *
 * @f: 전송되는 플릿 포인터 (vc 필드 참조)
 * @return: 없음
 *
 * 상위 SharedBufferPolicy::SendingFlit()을 호출해 점유를 갱신한 뒤,
 * 현재 시뮬레이션 시각을 _flit_sent_time[vc] 큐에 push한다.
 * 이 시각은 나중에 크레딧이 돌아올 때 FreeSlotFor()에서 RTT를 계산하는 데 사용된다.
 *
 * 호출 체인:
 *   BufferState::SendingFlit() → [FeedbackSharedBufferPolicy::SendingFlit()] → SharedBufferPolicy::SendingFlit()
 */
void BufferState::FeedbackSharedBufferPolicy::SendingFlit(Flit const * const f)
{
  SharedBufferPolicy::SendingFlit(f);    // [한국어] 상위 정책의 전용/공유 점유 갱신 먼저 수행
  _flit_sent_time[f->vc].push(GetSimTime()); // [한국어] 이 플릿의 전송 시각을 VC별 큐에 기록 — 크레딧 반환 시 RTT 계산용
}

/*
 * [한국어]
 * FeedbackSharedBufferPolicy::_ComputeRTT - RTT 지수 이동 평균 계산
 *
 * @vc: 계산할 VC 번호
 * @last_rtt: 가장 최근 측정된 RTT 값 (사이클)
 * @return: 갱신된 RTT 이동 평균 추정값
 *
 * 수식: new_rtt = ((old_rtt << s) + last_rtt - old_rtt) >> s
 *            = old_rtt + (last_rtt - old_rtt) / 2^s
 * old_rtt가 -1(미초기화)이면 last_rtt를 그대로 반환한다.
 * _aging_scale이 크면 과거 값에 더 많은 가중치를 두는 느린 평균.
 *
 * 호출 체인:
 *   FreeSlotFor() → [_ComputeRTT()] → 반환값으로 _round_trip_time 갱신
 */
int BufferState::FeedbackSharedBufferPolicy::_ComputeRTT(int vc, int last_rtt) const
{
  // compute moving average of round-trip time
  int rtt = _round_trip_time[vc]; // [한국어] 이 VC의 현재 RTT 이동 평균 추정값 읽기
  if(rtt < 0) { // [한국어] -1이면 아직 RTT 측정값이 없음 — 첫 측정값을 그대로 사용
    return last_rtt; // [한국어] 초기화: 첫 번째 RTT 측정값을 바로 추정값으로 채택
  }
  return ((rtt << _aging_scale) + last_rtt - rtt) >> _aging_scale;
  // [한국어] 지수 이동 평균: (rtt * 2^s + last_rtt - rtt) / 2^s
  //           = rtt + (last_rtt - rtt) / 2^_aging_scale
  //           _aging_scale이 클수록 last_rtt 반영 비율이 작아짐 (느린 평균)
}

/*
 * [한국어]
 * FeedbackSharedBufferPolicy::_ComputeLimit - RTT 기반 occupancy limit 계산
 *
 * @rtt: 현재 RTT 이동 평균 추정값 (사이클)
 * @return: 새 점유 한도 = max(2 * _min_latency - rtt + _offset, 1)
 *
 * RTT가 최소값(혼잡 없는 이상적 RTT = 2*_min_latency)보다 크면
 * 초과분만큼 한도를 줄인다. _offset으로 보정값을 추가한다.
 * 최솟값은 1 (0이 되면 전송 불가).
 * assert: _min_latency >= 0 (SetMinLatency()가 먼저 호출되어야 함).
 *
 * 호출 체인:
 *   FreeSlotFor() → [_ComputeLimit()] → _occupancy_limit 갱신
 */
int BufferState::FeedbackSharedBufferPolicy::_ComputeLimit(int rtt) const
{
  // for every cycle that the measured average round trip time exceeded the
  // observed minimum round trip time, reduce buffer occupancy limit by one
  assert(_min_latency >= 0); // [한국어] 최소 레이턴시가 아직 설정되지 않았으면 계산 불가 — 호출 순서 버그
  return max((_min_latency << 1) - rtt + _offset, 1);
  // [한국어] 이상적 RTT(2*min_latency)에서 현재 RTT를 뺀 값에 _offset을 더함
  // 결과가 1보다 작으면 1로 클램핑 — 최소 1슬롯은 허용해야 교착 방지
}

/*
 * [한국어]
 * FeedbackSharedBufferPolicy::_ComputeMaxSlots - VC의 현재 최대 허용 슬롯 수 실시간 계산
 *
 * @vc: 계산할 VC 번호
 * @return: min(_occupancy_limit[vc], limit_based_on_current_rtt)
 *
 * 이미 전송된 플릿이 있으면 가장 오래된 플릿의 전송 시각으로 현재 RTT를 추정해
 * 실시간 제한 값을 계산한다. 이를 _occupancy_limit과 비교해 더 작은 값을 반환.
 * 플릿이 없으면 _occupancy_limit만 반환.
 *
 * 호출 체인:
 *   IsFullFor/AvailableFor/LimitFor → [_ComputeMaxSlots()]
 */
int BufferState::FeedbackSharedBufferPolicy::_ComputeMaxSlots(int vc) const
{
  int max_slots = _occupancy_limit[vc]; // [한국어] 가장 최근 FreeSlotFor()에서 갱신된 한도에서 시작
  if(!_flit_sent_time[vc].empty()) { // [한국어] 아직 크레딧이 돌아오지 않은 플릿이 있으면 실시간 RTT 추정
    int min_rtt = GetSimTime() - _flit_sent_time[vc].front(); // [한국어] 가장 오래된 플릿의 현재 대기 시간 = 하한 RTT
    int rtt = _ComputeRTT(vc, min_rtt);    // [한국어] 현재 대기 시간으로 RTT 이동 평균 임시 계산
    int limit = _ComputeLimit(rtt);        // [한국어] 임시 RTT로 점유 한도 계산
    max_slots = min(max_slots, limit);     // [한국어] 실시간 추정 한도와 저장된 한도 중 더 제한적인 값 선택
  }
  return max_slots; // [한국어] 최종 허용 최대 슬롯 수 반환
}

/*
 * [한국어]
 * FeedbackSharedBufferPolicy::FreeSlotFor - 크레딧 수신 시 RTT 갱신 및 occupancy limit 재계산
 *
 * @vc: 슬롯이 해제된 VC 번호
 * @return: 없음
 *
 * 1. SharedBufferPolicy::FreeSlotFor(vc)로 공유 occupancy 감소
 * 2. _flit_sent_time[vc] 큐의 front()로 RTT 계산 후 pop
 * 3. _ComputeRTT()로 RTT 이동 평균 갱신
 * 4. _ComputeLimit()로 새 occupancy limit 계산
 * 5. _total_mapped_size와 _occupancy_limit[vc] 갱신
 *
 * 호출 체인:
 *   BufferState::ProcessCredit() → BufferPolicy::FreeSlotFor() → [FeedbackSharedBufferPolicy::FreeSlotFor()]
 */
void BufferState::FeedbackSharedBufferPolicy::FreeSlotFor(int vc)
{
  SharedBufferPolicy::FreeSlotFor(vc); // [한국어] 상위 정책의 공유/예약 슬롯 처리 먼저 수행
  assert(!_flit_sent_time[vc].empty()); // [한국어] 전송 시각 큐가 비어 있으면 크레딧 처리 순서 버그
  int const last_rtt = GetSimTime() - _flit_sent_time[vc].front(); // [한국어] 가장 오래된 전송 시각부터 현재까지 = 실제 RTT
#ifdef DEBUG_FEEDBACK
  cerr << FullName() << ": Probe for VC "
       << vc << " came back after "
       << last_rtt << " cycles."
       << endl; // [한국어] 이 크레딧의 실제 RTT를 디버그 출력
#endif
  _flit_sent_time[vc].pop(); // [한국어] 이 크레딧에 대응하는 전송 시각 큐에서 제거

  int rtt = _ComputeRTT(vc, last_rtt); // [한국어] 측정된 RTT로 이동 평균 갱신
#ifdef DEBUG_FEEDBACK
  int old_rtt = _round_trip_time[vc];
  if(rtt != old_rtt) {
    cerr << FullName() << ": Updating RTT estimate for VC "
	 << vc << " from "
	 << old_rtt << " to "
	 << rtt << " cycles."
	 << endl; // [한국어] RTT 추정값이 변경되면 디버그 출력
  }
#endif
  _round_trip_time[vc] = rtt; // [한국어] VC의 RTT 이동 평균 추정값을 새 값으로 갱신

  int limit = _ComputeLimit(rtt); // [한국어] 갱신된 RTT로 새 점유 한도 계산
#ifdef DEBUG_FEEDBACK
  int old_limit = _occupancy_limit[vc];
  int old_mapped_size = _total_mapped_size;
#endif
  _total_mapped_size += (limit - _occupancy_limit[vc]); // [한국어] 전체 매핑 크기를 한도 변화량만큼 조정
  _occupancy_limit[vc] = limit; // [한국어] 이 VC의 점유 한도를 새 값으로 갱신
#ifdef DEBUG_FEEDBACK
  if(limit != old_limit) {
    cerr << FullName() << ": Occupancy limit for VC "
	 << vc << " changed from "
	 << old_limit << " to "
	 << limit << " slots."
	 << endl; // [한국어] 한도 변경 시 디버그 출력
    cerr << FullName() << ": Total mapped buffer space changed from "
	 << old_mapped_size << " to "
	 << _total_mapped_size << " slots."
	 << endl; // [한국어] 전체 매핑 크기 변화도 디버그 출력
  }
#endif
}

/*
 * [한국어]
 * FeedbackSharedBufferPolicy::IsFullFor - RTT 기반 한도 포함하여 가득 찼는지 확인
 *
 * @vc: 확인할 VC 번호
 * @return: 공유 정책상 full이거나 VC 점유 >= _ComputeMaxSlots(vc)이면 true
 *
 * 호출 체인:
 *   BufferState::IsFullFor() → [FeedbackSharedBufferPolicy::IsFullFor()]
 */
bool BufferState::FeedbackSharedBufferPolicy::IsFullFor(int vc) const
{
  if(SharedBufferPolicy::IsFullFor(vc)) { // [한국어] 공유 정책(shared)이 이미 포화 상태이면 즉시 full 반환
    return true;
  }
  return (_buffer_state->OccupancyFor(vc) >= _ComputeMaxSlots(vc)); // [한국어] RTT 기반 동적 한도에도 도달했는지 확인
}

/*
 * [한국어]
 * FeedbackSharedBufferPolicy::AvailableFor - RTT 기반 한도를 적용한 가용 슬롯 수
 *
 * @vc: 확인할 VC 번호
 * @return: min(SharedBufferPolicy::AvailableFor(vc), _ComputeMaxSlots(vc) - OccupancyFor(vc))
 *
 * 호출 체인:
 *   BufferState::AvailableFor() → [FeedbackSharedBufferPolicy::AvailableFor()]
 */
int BufferState::FeedbackSharedBufferPolicy::AvailableFor(int vc) const
{
  return min(SharedBufferPolicy::AvailableFor(vc), // [한국어] 공유 정책 가용량과
	     _ComputeMaxSlots(vc) - _buffer_state->OccupancyFor(vc)); // [한국어] RTT 기반 한도 잔여 중 더 작은 값
}

/*
 * [한국어]
 * FeedbackSharedBufferPolicy::LimitFor - RTT 기반 한도를 적용한 최대 한도
 *
 * @vc: 확인할 VC 번호
 * @return: min(SharedBufferPolicy::LimitFor(vc), _ComputeMaxSlots(vc))
 *
 * 호출 체인:
 *   BufferState::LimitFor() → [FeedbackSharedBufferPolicy::LimitFor()]
 */
int BufferState::FeedbackSharedBufferPolicy::LimitFor(int vc) const
{
  return min(SharedBufferPolicy::LimitFor(vc), _ComputeMaxSlots(vc)); // [한국어] 공유 이론 한도와 RTT 기반 한도 중 더 제한적인 값
}

/* =========================================================================
 * SimpleFeedbackSharedBufferPolicy 구현
 * ========================================================================= */

/*
 * [한국어]
 * SimpleFeedbackSharedBufferPolicy::SimpleFeedbackSharedBufferPolicy - 단순 피드백 정책 생성자
 *
 * @config, @parent, @name: FeedbackSharedBufferPolicy 생성자로 전달
 * @return: 없음 (생성자)
 *
 * _pending_credits 벡터를 _vcs 크기로 초기화 (모두 0).
 *
 * 호출 체인:
 *   BufferPolicy::New() → [SimpleFeedbackSharedBufferPolicy()] → FeedbackSharedBufferPolicy()
 */
BufferState::SimpleFeedbackSharedBufferPolicy::SimpleFeedbackSharedBufferPolicy(Configuration const & config, BufferState * parent, const string & name)
  : FeedbackSharedBufferPolicy(config, parent, name) // [한국어] Feedback 생성자 호출 — RTT 벡터 초기화
{
  _pending_credits.resize(_vcs, 0); // [한국어] VC별 대기 중인 non-probe 크레딧 수 0으로 초기화
}

/*
 * [한국어]
 * SimpleFeedbackSharedBufferPolicy::SendingFlit - probe 여부에 따른 분기 처리
 *
 * @f: 전송되는 플릿 포인터 (vc 필드 참조)
 * @return: 없음
 *
 * _flit_sent_time[vc]가 비어 있으면 (진행 중인 RTT 측정 없음):
 *   이 플릿을 "probe"로 지정해 FeedbackSharedBufferPolicy::SendingFlit() 호출.
 *   _pending_credits[vc] = OccupancyFor(vc) - 1 (probe 이전에 이미 큐에 있던 플릿 수).
 * 그렇지 않으면: probe가 이미 전송 중이므로 그냥 SharedBufferPolicy::SendingFlit() 호출.
 *
 * 호출 체인:
 *   BufferState::SendingFlit() → [SimpleFeedbackSharedBufferPolicy::SendingFlit()]
 */
void BufferState::SimpleFeedbackSharedBufferPolicy::SendingFlit(Flit const * const f)
{
  int const & vc = f->vc; // [한국어] 이 플릿의 VC 번호 (const 레퍼런스)
  if(_flit_sent_time[vc].empty()) { // [한국어] 이 VC에 진행 중인 RTT 측정 없음 — 새 probe 플릿 전송
    assert(_buffer_state->OccupancyFor(vc) > 0); // [한국어] probe를 보내려면 이미 1개 이상 적재되어 있어야 함
    _pending_credits[vc] = _buffer_state->OccupancyFor(vc) - 1;
    // [한국어] probe보다 먼저 전송될 non-probe 크레딧 수
    // = 현재 VC occupancy - 1 (probe 플릿 자신 제외)
#ifdef DEBUG_SIMPLEFEEDBACK
    cerr << FullName() << ": Sending probe flit for VC "
	 << vc << "; "
	 << _pending_credits[vc] << " non-probe flits in flight."
	 << endl; // [한국어] probe 전송 및 non-probe 수 디버그 출력
#endif
    FeedbackSharedBufferPolicy::SendingFlit(f); // [한국어] probe: 전송 시각을 _flit_sent_time에 기록
    return; // [한국어] probe 처리 완료 후 조기 반환
  }
  SharedBufferPolicy::SendingFlit(f); // [한국어] non-probe: probe가 이미 전송 중이므로 공유 점유만 갱신
}

/*
 * [한국어]
 * SimpleFeedbackSharedBufferPolicy::FreeSlotFor - probe 크레딧과 non-probe 크레딧 구분 처리
 *
 * @vc: 슬롯이 해제된 VC 번호
 * @return: 없음
 *
 * _pending_credits[vc] == 0이고 _flit_sent_time이 비지 않음:
 *   probe 크레딧이 돌아온 것 → FeedbackSharedBufferPolicy::FreeSlotFor()로 RTT 측정.
 * _pending_credits[vc] > 0:
 *   non-probe 크레딧 → _pending_credits 감소 + SharedBufferPolicy::FreeSlotFor().
 * 그 외: 일반 SharedBufferPolicy::FreeSlotFor() 처리.
 *
 * 호출 체인:
 *   BufferState::ProcessCredit() → BufferPolicy::FreeSlotFor() → [SimpleFeedbackSharedBufferPolicy::FreeSlotFor()]
 */
void BufferState::SimpleFeedbackSharedBufferPolicy::FreeSlotFor(int vc)
{
  if(!_flit_sent_time[vc].empty() && _pending_credits[vc] == 0) {
    // [한국어] 진행 중인 RTT 측정이 있고 대기 non-probe 크레딧이 0 → probe 크레딧 도착
#ifdef DEBUG_SIMPLEFEEDBACK
    cerr << FullName() << ": Probe credit for VC "
	 << vc << " came back." << endl; // [한국어] probe 크레딧 반환 디버그 출력
#endif
    FeedbackSharedBufferPolicy::FreeSlotFor(vc); // [한국어] RTT 측정 + occupancy limit 갱신
    return; // [한국어] probe 처리 완료 후 조기 반환
  }
  if(_pending_credits[vc] > 0) { // [한국어] 아직 non-probe 크레딧이 남아 있음
    assert(!_flit_sent_time[vc].empty()); // [한국어] probe가 이미 전송 중이어야 함 (전송 시각 큐 비어 있으면 버그)
    --_pending_credits[vc]; // [한국어] non-probe 크레딧 카운터 감소
#ifdef DEBUG_SIMPLEFEEDBACK
    cerr << FullName() << ": Ignoring non-probe credit for VC "
	 << vc << "; "
	 << _pending_credits[vc] << " remaining."
	 << endl; // [한국어] non-probe 크레딧 무시 사실 디버그 출력
#endif
  }
  SharedBufferPolicy::FreeSlotFor(vc); // [한국어] 일반 공유 슬롯 해제 처리 (RTT 갱신 없음)
}

/* =========================================================================
 * BufferState 구현
 * ========================================================================= */

/*
 * [한국어]
 * BufferState::BufferState - 버퍼 상태 추적 객체 생성자
 *
 * @config: num_vcs, buf_size, buffer_policy, wait_for_tail_credit, classes 등 참조
 * @parent: 모듈 계층 부모 (라우터 객체)
 * @name:   모듈 이름 (예: "next_buf_0")
 * @return: 없음 (생성자)
 *
 * 초기화 순서:
 *   1. _occupancy = 0
 *   2. _vcs, _size 읽기 (buf_size < 0이면 vcs * vc_buf_size)
 *   3. BufferPolicy::New()로 구체 정책 객체 생성
 *   4. _wait_for_tail_credit 읽기
 *   5. _vc_occupancy, _in_use_by(-1), _tail_sent(false), _last_id(-1), _last_pid(-1) 초기화
 *   6. TRACK_BUFFERS: _outstanding_classes, _class_occupancy 초기화
 *
 * 호출 체인:
 *   IQRouter::IQRouter() → [BufferState::BufferState()] → BufferPolicy::New()
 */
BufferState::BufferState( const Configuration& config, Module *parent, const string& name ) :
  Module( parent, name ), _occupancy(0) // [한국어] Module 계층 등록, 전체 occupancy 0으로 초기화
{
  _vcs = config.GetInt( "num_vcs" ); // [한국어] 전체 VC 수 읽기 — 벡터 크기 결정
  _size = config.GetInt("buf_size"); // [한국어] 전체 버퍼 크기 읽기 (음수이면 아래에서 재계산)
  if(_size < 0) { // [한국어] buf_size 미지정(-1) — VC당 크기로 계산
    _size = _vcs * config.GetInt("vc_buf_size"); // [한국어] 각 VC에 vc_buf_size 슬롯씩 합산하여 전체 크기 결정
  }

  _buffer_policy = BufferPolicy::New(config, this, "policy"); // [한국어] 설정의 "buffer_policy"에 따라 적합한 정책 객체 생성

  _wait_for_tail_credit = config.GetInt( "wait_for_tail_credit" ); // [한국어] tail 크레딧 수신 후 VC 해제 여부 플래그 읽기

  _vc_occupancy.resize(_vcs, 0); // [한국어] VC별 점유 카운터 초기화 (모두 0)

  _in_use_by.resize(_vcs, -1);     // [한국어] VC별 사용 중인 패킷 태그 초기화 (-1: 미사용)
  _tail_sent.resize(_vcs, false);  // [한국어] VC별 tail 전송 여부 초기화 (false: 미전송)

  _last_id.resize(_vcs, -1);  // [한국어] VC별 마지막 전송 플릿 ID 초기화 (-1: 없음)
  _last_pid.resize(_vcs, -1); // [한국어] VC별 마지막 전송 패킷 ID 초기화 (-1: 없음)

#ifdef TRACK_BUFFERS
  _classes = config.GetInt("classes");          // [한국어] 트래픽 클래스 수 읽기
  _outstanding_classes.resize(_vcs);            // [한국어] VC별 미반환 크레딧의 클래스 큐 초기화 (빈 큐)
  _class_occupancy.resize(_classes, 0);         // [한국어] 클래스별 occupancy 카운터 초기화 (모두 0)
#endif
}

/*
 * [한국어]
 * BufferState::~BufferState - 버퍼 상태 추적 객체 소멸자
 *
 * @return: 없음 (소멸자)
 *
 * 생성자에서 BufferPolicy::New()로 할당된 정책 객체를 delete한다.
 *
 * 호출 체인:
 *   IQRouter::~IQRouter() 또는 시뮬레이터 종료 → [BufferState::~BufferState()]
 */
BufferState::~BufferState()
{
  delete _buffer_policy; // [한국어] 구체 정책 객체 메모리 해제 (파생 클래스 소멸자 자동 호출)
}

/*
 * [한국어]
 * BufferState::ProcessCredit - 다운스트림 라우터에서 받은 크레딧 패킷 처리
 *
 * @c: 수신된 크레딧 패킷 포인터 (set<int> vc 집합 포함, NULL 불가)
 * @return: 없음
 *
 * 크레딧 패킷에 포함된 각 VC 번호에 대해:
 *   1. _occupancy, _vc_occupancy[vc] 각각 1 감소
 *   2. _wait_for_tail_credit 모드에서 VC가 점유 중이지 않으면 에러
 *   3. _wait_for_tail_credit && !vc_occupancy && tail_sent이면 VC 해제(_in_use_by = -1)
 *   4. TRACK_BUFFERS: _outstanding_classes[vc] 큐에서 pop 후 클래스 occupancy 감소
 *   5. _buffer_policy->FreeSlotFor(vc) 호출
 * 에러: occupancy < 0 또는 vc_occupancy < 0이면 Error().
 *
 * 호출 체인:
 *   IQRouter::ReadInputs() → [BufferState::ProcessCredit()] → BufferPolicy::FreeSlotFor()
 */
void BufferState::ProcessCredit( Credit const * const c )
{
  assert( c ); // [한국어] 크레딧 포인터 NULL 방지 검사

  set<int>::iterator iter = c->vc.begin(); // [한국어] 크레딧에 포함된 VC 집합 순회 시작
  while(iter != c->vc.end()) { // [한국어] 크레딧에 포함된 모든 VC 번호 처리

    int const vc = *iter; // [한국어] 현재 처리 중인 VC 번호

    assert( ( vc >= 0 ) && ( vc < _vcs ) ); // [한국어] VC 번호 유효성 검사

    if ( ( _wait_for_tail_credit ) && // [한국어] tail_credit 대기 모드에서
	 ( _in_use_by[vc] < 0 ) ) {  // [한국어] VC가 이미 해제된 상태에서 크레딧이 오면 논리 오류
      ostringstream err;
      err << "Received credit for idle VC " << vc; // [한국어] 유휴 VC에 크레딧이 돌아온 에러 — TakeBuffer/SendingFlit 순서 버그
      Error( err.str() );
    }
    --_occupancy; // [한국어] 전체 점유 1 감소 — 다운스트림 슬롯 하나가 해제됨
    if(_occupancy < 0) { // [한국어] 전체 occupancy가 음수가 되면 크레딧 처리 버그
      Error("Buffer occupancy fell below zero.");
    }
    --_vc_occupancy[vc]; // [한국어] 이 VC의 점유 1 감소
    if(_vc_occupancy[vc] < 0) { // [한국어] VC별 occupancy도 음수 방지
      ostringstream err;
      err << "Buffer occupancy fell below zero for VC " << vc;
      Error(err.str());
    }
    if(_wait_for_tail_credit && !_vc_occupancy[vc] && _tail_sent[vc]) {
      // [한국어] tail_credit 대기 모드에서 VC 점유가 0이 되었고 tail도 전송됐으면 VC 해제 가능
      assert(_in_use_by[vc] >= 0); // [한국어] 이 시점에서 VC는 반드시 사용 중이어야 함
      _in_use_by[vc] = -1; // [한국어] VC를 사용 가능 상태로 해제 (새 패킷 할당 허용)
    }

#ifdef TRACK_BUFFERS
    assert(!_outstanding_classes[vc].empty()); // [한국어] 크레딧에 대응하는 클래스 기록이 있어야 함
    int cl = _outstanding_classes[vc].front(); // [한국어] 가장 오래된 전송 플릿의 클래스 번호 읽기
    _outstanding_classes[vc].pop();            // [한국어] 해당 클래스 기록 제거
    assert((cl >= 0) && (cl < _classes));      // [한국어] 클래스 번호 유효성 검사
    assert(_class_occupancy[cl] > 0);          // [한국어] 클래스 점유가 0 이상이어야 함
    --_class_occupancy[cl];                    // [한국어] 해당 클래스의 점유 수 1 감소
#endif

    _buffer_policy->FreeSlotFor(vc); // [한국어] 정책 객체에 슬롯 해제 통보 (shared/feedback 정책에서 내부 상태 갱신)

    ++iter; // [한국어] 다음 VC 번호로 이동
  }
}


/*
 * [한국어]
 * BufferState::SendingFlit - 다운스트림으로 플릿 전송 시 상태 갱신
 *
 * @f: 전송되는 플릿 포인터 (vc, tail, cl, id, pid 필드 참조, NULL 불가)
 * @return: 없음
 *
 * 1. _occupancy, _vc_occupancy[vc] 증가 (오버플로 검사 포함)
 * 2. _buffer_policy->SendingFlit(f) 호출
 * 3. TRACK_BUFFERS: 클래스 큐/카운터 갱신
 * 4. f->tail이면 _tail_sent[vc] = true
 *    !_wait_for_tail_credit이면 즉시 VC 해제 (_in_use_by[vc] = -1)
 * 5. _last_id[vc], _last_pid[vc] 갱신
 *
 * 호출 체인:
 *   IQRouter::_SwitchTraversal() 또는 WriteOutputs() → [BufferState::SendingFlit()] → BufferPolicy::SendingFlit()
 */
void BufferState::SendingFlit( Flit const * const f )
{
  int const vc = f->vc; // [한국어] 이 플릿이 사용하는 VC 번호

  assert( f && ( vc >= 0 ) && ( vc < _vcs ) ); // [한국어] 포인터 및 VC 번호 유효성 검사

  ++_occupancy; // [한국어] 다운스트림 전체 버퍼 점유 1 증가 (크레딧 소비)
  if(_occupancy > _size) { // [한국어] 전체 버퍼 크기 초과 — 흐름 제어 누락 또는 IsFullFor 검사 누락
    Error("Buffer overflow."); // [한국어] 버퍼 오버플로 에러 출력 후 종료
  }

  ++_vc_occupancy[vc]; // [한국어] 이 VC의 점유 1 증가

  _buffer_policy->SendingFlit(f); // [한국어] 정책 객체에 플릿 전송 통보 (private/shared 내부 점유 갱신)

#ifdef TRACK_BUFFERS
  _outstanding_classes[vc].push(f->cl); // [한국어] 이 플릿의 클래스를 VC별 미반환 큐에 기록
  ++_class_occupancy[f->cl];            // [한국어] 이 클래스의 전체 점유 카운터 1 증가
#endif

  if ( f->tail ) { // [한국어] tail 플릿이면 이 VC에서의 패킷 전송 완료
    _tail_sent[vc] = true; // [한국어] tail이 전송됐음을 기록 — 크레딧 대기 모드의 VC 해제 조건 중 하나

    if ( !_wait_for_tail_credit ) { // [한국어] tail_credit 대기 모드가 아니면 즉시 VC 해제
      assert(_in_use_by[vc] >= 0); // [한국어] tail 전송 시점에 VC는 반드시 사용 중이어야 함
      _in_use_by[vc] = -1; // [한국어] VC를 즉시 유휴 상태로 해제 (새 패킷 할당 허용)
    }
  }
  _last_id[vc] = f->id;   // [한국어] 마지막 전송 플릿의 고유 ID 기록 (디버그용)
  _last_pid[vc] = f->pid; // [한국어] 마지막 전송 패킷의 패킷 ID 기록 (디버그용)
}

/*
 * [한국어]
 * BufferState::TakeBuffer - 지정 VC를 특정 패킷(tag)에 전용 예약
 *
 * @vc:  예약할 VC 번호 (0 이상 _vcs 미만)
 * @tag: 이 VC를 점유할 패킷의 태그 (예: 입력 VC 번호 또는 패킷 ID)
 * @return: 없음
 *
 * _in_use_by[vc]가 이미 >= 0이면 "Buffer taken while in use" 에러.
 * 그렇지 않으면:
 *   - _in_use_by[vc] = tag (VC를 이 패킷에 할당)
 *   - _tail_sent[vc] = false (새 패킷 시작 — tail 미전송 상태)
 *   - _buffer_policy->TakeBuffer(vc) 호출
 * 실행 컨텍스트: VC 할당 완료 직후 호출됨.
 *
 * 호출 체인:
 *   IQRouter::_VCAlloc() → [BufferState::TakeBuffer()] → BufferPolicy::TakeBuffer()
 */
void BufferState::TakeBuffer( int vc, int tag )
{
  assert( ( vc >= 0 ) && ( vc < _vcs ) ); // [한국어] VC 번호 유효성 검사

  if ( _in_use_by[vc] >= 0 ) { // [한국어] 이미 사용 중인 VC에 또 예약하려는 경우 — 중복 할당 버그
    ostringstream err;
    err << "Buffer taken while in use for VC " << vc; // [한국어] 어떤 VC에서 이중 할당이 발생했는지 명시
    Error( err.str() ); // [한국어] 에러 출력 후 종료
  }
  _in_use_by[vc] = tag;      // [한국어] 이 VC를 tag 패킷에 할당 — IsAvailableFor()가 false 반환하게 됨
  _tail_sent[vc] = false;    // [한국어] 새 패킷 시작이므로 tail 전송 여부 초기화
  _buffer_policy->TakeBuffer(vc); // [한국어] 정책 객체에 VC 할당 통보 (LimitedShared 이상에서 _active_vcs 갱신)
}

/*
 * [한국어]
 * BufferState::Display - 전체 버퍼 상태를 출력 스트림에 덤프 (디버그용)
 *
 * @os: 출력 스트림 (기본값 cout)
 * @return: 없음
 *
 * 전체 occupancy와 각 VC별 in_use_by, tail_sent, occupied 값을 출력한다.
 * 시뮬레이션 중 크레딧 처리 이상이나 흐름 제어 버그 디버깅 시 사용.
 *
 * 호출 체인:
 *   라우터 디버그 또는 수동 호출 → [BufferState::Display()]
 */
void BufferState::Display( ostream & os ) const
{
  os << FullName() << " :" << endl; // [한국어] 이 BufferState 모듈의 전체 이름 출력
  os << " occupied = " << _occupancy << endl; // [한국어] 다운스트림 버퍼 전체 현재 점유 수 출력
  for ( int v = 0; v < _vcs; ++v ) { // [한국어] 모든 VC에 대해 개별 상태 출력
    os << "  VC " << v << ": "; // [한국어] VC 번호 출력
    os << "in_use_by = " << _in_use_by[v]   // [한국어] 이 VC를 점유 중인 패킷 태그 (-1이면 유휴)
       << ", tail_sent = " << _tail_sent[v]  // [한국어] tail 플릿 전송 여부
       << ", occupied = " << _vc_occupancy[v] << endl; // [한국어] 이 VC의 현재 점유 슬롯 수
  }
}
