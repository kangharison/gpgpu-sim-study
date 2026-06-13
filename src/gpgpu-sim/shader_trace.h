/*
 * [한국어 설명] 셰이더 코어(SM) 전용 디버그 트레이스 매크로 (shader_trace.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 셰이더 코어(SM, Streaming Multiprocessor) 내부 컴포넌트에서
 * 사이클 단위 동작을 선택적으로 출력하기 위한 디버그 트레이스 매크로를 정의한다.
 * 전역 트레이스 시스템(trace.h)의 DTRACE/DPRINTF 위에 "특정 SM만 모니터링"하는
 * 필터링 레이어를 추가함으로써, 수십~수백 개의 SM 중 관심 있는 하나의 SM(또는 전체)에
 * 대해서만 상세한 출력이 나오도록 한다. 빌드 시 TRACING_ON이 정의되지 않으면 모든
 * 매크로는 완전한 no-op으로 컴파일되어 성능 오버헤드가 전혀 없다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 타이밍 모델(gpgpu-sim/)의 셰이더 코어 컴포넌트 최하단 공통 인프라에 해당한다.
 * 호출 체인:
 *   shader.cc (shader_core_ctx, scheduler_unit 등 SM 내부 클래스)
 *     → SHADER_DPRINTF / SCHED_DPRINTF 매크로 호출
 *         → trace.h의 DTRACE + Trace::sampling_core 필터 평가
 *             → 조건 충족 시 printf로 시뮬레이션 사이클 + 채널명 + SM ID 출력
 * 실행 컨텍스트: 호스트 유저스페이스의 사이클-레벨 시뮬레이션 루프 내부에서 호출되므로
 * 멀티스레드 동기화는 불필요하다 (GPGPU-Sim은 단일 스레드 시뮬레이션).
 *
 * === 타 모듈과의 연결 ===
 * 의존:
 *   - ../trace.h : DTRACE(x), SIM_PRINT_STR, Trace::sampling_core,
 *                  Trace::trace_streams_str[] 등 전역 트레이스 인프라 제공.
 *                  채널 enum(WARP_SCHEDULER 등)도 trace.h에서 정의된다.
 * 사용처:
 *   - shader.cc / shader.h : shader_core_ctx, scheduler_unit, operand_collector 등
 *     SM 내부 모든 컴포넌트가 이 매크로를 사용해 디버그 출력을 남긴다.
 * 데이터 흐름:
 *   Trace::sampling_core 전역 변수 ← gpgpusim.config의 -gpgpu_trace_sampling_core 옵션
 *   → SHADER_DTRACE 필터 결과 → printf 출력 (stdout / 리다이렉트 파일)
 *
 * === 주요 함수/구조체 요약 ===
 * SHADER_DTRACE(x)     : 채널 x가 활성화되어 있고 현재 SM이 모니터링 대상일 때 true 반환.
 * SHADER_DPRINTF(x,..) : SHADER_DTRACE 조건 하에 "[사이클] [채널명] Core N - ..." 출력.
 * SCHED_DPRINTF(..)    : scheduler_unit 내부 전용. WARP_SCHEDULER 채널 고정,
 *                        SM ID에 더해 스케줄러 ID(m_id)까지 접두어로 출력.
 * SHADER_PRINT_STR     : SHADER_DPRINTF 출력 포맷 문자열 ("... Core %d - ").
 * SCHED_PRINT_STR      : SCHED_DPRINTF 출력 포맷 문자열 ("... Core %d - Scheduler %d - ").
 */

// Copyright (c) 2009-2011, Tor M. Aamodt, Tim Rogers
// George L. Yuan, Andrew Turner, Inderpreet Singh
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/* [한국어] 헤더 가드 시작 — 이 파일이 동일 번역 단위에 중복 포함되는 것을 방지한다. */
#ifndef __SHADER_TRACE_H__
#define __SHADER_TRACE_H__

/* [한국어] trace.h 포함 — 전역 트레이스 인프라(DTRACE, SIM_PRINT_STR,
 * Trace::sampling_core, Trace::trace_streams_str[]) 및 채널 enum을 가져온다.
 * 이 파일의 모든 매크로는 trace.h 없이는 동작하지 않는다. */
#include "../trace.h"

/* [한국어] TRACING_ON 전처리 분기 — 빌드 시 -DTRACING_ON=1 플래그로 결정된다.
 * 활성화(=1): 아래의 실제 트레이스 매크로들이 컴파일에 포함된다.
 * 비활성화(=0 또는 미정의): #else 블록의 no-op 매크로들이 대신 삽입되어
 * 트레이싱 코드가 완전히 제거되므로 릴리스/성능 측정 빌드에서 오버헤드가 없다. */
#if TRACING_ON

/* [한국어] SHADER_PRINT_STR — SHADER_DPRINTF가 사용하는 출력 접두어 포맷 문자열.
 * SIM_PRINT_STR(trace.h 정의, 예: "%llu ")에 "Core %d - "를 이어 붙인다.
 * 최종 형식: "<시뮬레이션 총 사이클수> [채널명] Core <SM_ID> - <사용자 메시지>"
 * %d 자리에는 get_sid()의 반환값(SM ID)이 대입된다. */
#define SHADER_PRINT_STR SIM_PRINT_STR "Core %d - "

/* [한국어] SCHED_PRINT_STR — SCHED_DPRINTF가 사용하는 출력 접두어 포맷 문자열.
 * SHADER_PRINT_STR에 "Scheduler %d - "를 추가로 이어 붙인다.
 * 최종 형식: "<사이클> [WARP_SCHEDULER] Core <SM_ID> - Scheduler <sched_ID> - <메시지>"
 * 두 번째 %d 자리에는 scheduler_unit::m_id(스케줄러 인스턴스 번호)가 대입된다.
 * SM당 여러 개의 워프 스케줄러가 있을 수 있으므로(예: GF100은 2개), 스케줄러 ID를
 * 별도로 출력해야 어느 스케줄러에서 발생한 이벤트인지 구별할 수 있다. */
#define SCHED_PRINT_STR SHADER_PRINT_STR "Scheduler %d - "

/* [한국어] SHADER_DTRACE(x) — SM 레벨 트레이스 활성화 조건 평가 매크로.
 *
 * 조건:
 *   1. DTRACE(x): trace.h에서 정의된 전역 조건 — 채널 x가 활성화되어 있고
 *      트레이싱 시스템 자체가 켜져 있는지 확인한다.
 *   2. Trace::sampling_core == (int)get_sid():
 *      현재 코드가 실행 중인 SM(get_sid() 반환)이 gpgpusim.config의
 *      -gpgpu_trace_sampling_core 옵션으로 지정된 특정 SM과 일치하는지 확인.
 *   3. Trace::sampling_core == -1:
 *      sampling_core가 -1로 설정된 경우 모든 SM의 출력을 허용한다.
 *      (기본값은 -1 — 전체 SM 출력)
 *
 * 전체 조건: DTRACE(x) && (현재_SM == 지정_SM || 지정_SM == -1)
 * 이 조건이 false이면 SHADER_DPRINTF 내부 printf가 호출되지 않아 출력이 억제된다.
 * get_sid()는 shader_core_ctx 내의 멤버 함수로, 이 매크로는 해당 클래스 범위에서만
 * 올바르게 동작한다. */
#define SHADER_DTRACE(x) \
  (DTRACE(x) &&          \
   (Trace::sampling_core == (int)get_sid() || Trace::sampling_core == -1))

// Intended to be called from inside components of a shader core.
// Depends on a get_sid() function
/* [한국어] SHADER_DPRINTF(x, ...) — 셰이더 코어 내부 컴포넌트용 포맷 출력 매크로.
 *
 * @x    : 트레이스 채널 enum 값 (예: WARP_SCHEDULER, CACHE, EXECUTE 등).
 *          trace.h의 Trace::trace_streams_str[] 배열 인덱스로도 사용된다.
 * @...  : printf 스타일 포맷 문자열 및 인수들 (__VA_ARGS__로 전달됨).
 *
 * 동작:
 *   1. SHADER_DTRACE(x)로 채널 활성화 + SM 필터를 동시에 확인한다.
 *   2. 조건 충족 시 SHADER_PRINT_STR 포맷으로 접두어를 출력한다:
 *      - m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle : 워밍업 포함 총 사이클 수.
 *        gpu_sim_cycle은 현재 커널 실행 중 경과 사이클,
 *        gpu_tot_sim_cycle은 이전 커널들까지 누적된 총 사이클이다.
 *      - Trace::trace_streams_str[Trace::x] : 채널 이름 문자열 (예: "WARP_SCHEDULER").
 *      - get_sid() : 현재 SM의 ID (0-based).
 *   3. 이후 사용자 포맷 문자열(__VA_ARGS__)을 두 번째 printf로 출력한다.
 *
 * do { ... } while(0) 패턴:
 *   단일 문장(statement)처럼 동작하게 하는 C 매크로 관용구.
 *   if (cond) SHADER_DPRINTF(...); else ... 와 같은 구문에서도 세미콜론이 올바르게
 *   처리되며, 빈 else 분기 문제가 발생하지 않는다.
 *
 * 의존: m_gpu 포인터(gpu_sim_cycle, gpu_tot_sim_cycle 접근)와 get_sid() 함수가
 * 호출 컨텍스트(클래스)에 존재해야 한다. shader_core_ctx 및 그 내부 클래스에서 사용.
 *
 * 호출 체인:
 *   shader.cc (shader_core_ctx 또는 내부 서브컴포넌트)
 *     → [SHADER_DPRINTF] → SHADER_DTRACE 평가 → printf 출력 */
#define SHADER_DPRINTF(x, ...)                                \
  do {                                                        \
    if (SHADER_DTRACE(x)) {                                   \
      /* [한국어] 접두어 출력: 총 사이클 + 채널명 + SM ID */   \
      printf(SHADER_PRINT_STR,                                \
             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, \
             Trace::trace_streams_str[Trace::x], get_sid());  \
      /* [한국어] 사용자 정의 메시지 출력 (가변 인수 그대로 전달) */ \
      printf(__VA_ARGS__);                                    \
    }                                                         \
  } while (0)

// Intended to be called from inside a scheduler_unit.
// Depends on a m_id member
/* [한국어] SCHED_DPRINTF(...) — scheduler_unit 내부 전용 포맷 출력 매크로.
 *
 * @... : printf 스타일 포맷 문자열 및 인수들.
 *
 * SHADER_DPRINTF와의 차이점:
 *   1. 채널이 WARP_SCHEDULER로 고정된다 — 워프 스케줄러 이벤트 전용 채널.
 *   2. SCHED_PRINT_STR를 사용하여 SM ID 외에 스케줄러 ID(m_id)도 접두어에 포함한다.
 *      형식: "<사이클> [WARP_SCHEDULER] Core <SM_ID> - Scheduler <sched_ID> - <메시지>"
 *
 * 동작:
 *   1. SHADER_DTRACE(WARP_SCHEDULER): WARP_SCHEDULER 채널 활성화 + 현재 SM 필터 확인.
 *   2. m_shader->get_gpu()->gpu_sim_cycle + m_shader->get_gpu()->gpu_tot_sim_cycle :
 *      scheduler_unit은 m_gpu 대신 m_shader(상위 shader_core_ctx 포인터)를 통해
 *      gpu 객체에 접근한다.
 *   3. Trace::trace_streams_str[Trace::WARP_SCHEDULER] : "WARP_SCHEDULER" 문자열.
 *   4. get_sid() : scheduler_unit이 속한 SM의 ID. scheduler_unit은 상위 클래스나
 *      멤버 함수를 통해 get_sid()를 제공받는다.
 *   5. m_id : scheduler_unit 인스턴스 고유 번호. SM당 여러 스케줄러가 있을 때
 *      어느 스케줄러에서 발생한 이벤트인지 식별하는 데 사용된다.
 *
 * 의존: m_shader 포인터, get_sid() 함수, m_id 멤버가 호출 범위에 있어야 한다.
 *       scheduler_unit 클래스 또는 그 내부에서만 사용 가능.
 *
 * 호출 체인:
 *   shader.cc (scheduler_unit::cycle())
 *     → [SCHED_DPRINTF] → SHADER_DTRACE(WARP_SCHEDULER) → printf 출력 */
#define SCHED_DPRINTF(...)                                               \
  do {                                                                   \
    if (SHADER_DTRACE(WARP_SCHEDULER)) {                                 \
      /* [한국어] 접두어 출력: 총 사이클 + "WARP_SCHEDULER" 채널명 + SM ID + 스케줄러 ID */ \
      printf(SCHED_PRINT_STR,                                            \
             m_shader->get_gpu()->gpu_sim_cycle +                        \
                 m_shader->get_gpu()->gpu_tot_sim_cycle,                 \
             Trace::trace_streams_str[Trace::WARP_SCHEDULER], get_sid(), \
             m_id);                                                      \
      /* [한국어] 사용자 정의 메시지 출력 (가변 인수 그대로 전달) */        \
      printf(__VA_ARGS__);                                               \
    }                                                                    \
  } while (0)

/* [한국어] TRACING_ON=0(또는 미정의) 빌드용 no-op 분기.
 * 아래 정의들은 트레이싱이 비활성화된 경우 모든 트레이스 매크로 호출을
 * 컴파일러가 완전히 제거할 수 있도록 빈 do-while(0) 또는 (false)로 치환한다.
 * 이로써 성능 측정/릴리스 빌드에서 트레이스 코드의 오버헤드가 전혀 없다. */
#else

/* [한국어] SHADER_DTRACE(x) no-op — 항상 false를 반환하여 DPRINTF 내부 printf가
 * 호출되지 않도록 한다. 컴파일러 최적화로 dead code가 제거된다. */
#define SHADER_DTRACE(x) (false)

/* [한국어] SHADER_DPRINTF(x, ...) no-op — 빈 do-while(0)으로 치환.
 * 세미콜론 처리 일관성을 위해 do-while(0) 패턴을 유지한다. */
#define SHADER_DPRINTF(x, ...) \
  do {                         \
  } while (0)

/* [한국어] SCHED_DPRINTF(x, ...) no-op — 빈 do-while(0)으로 치환.
 * TRACING_ON 빌드와 동일한 호출 위치에서 컴파일 오류 없이 치환된다. */
#define SCHED_DPRINTF(x, ...) \
  do {                        \
  } while (0)

/* [한국어] TRACING_ON 전처리 분기 종료 */
#endif

/* [한국어] 헤더 가드 종료 */
#endif
