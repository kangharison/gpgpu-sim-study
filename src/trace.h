// Copyright (c) 2009-2013, Tor M. Aamodt, Timothy Rogers,
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

// This file is inspired by the trace system in gem5.
// This is a highly simplified version adpated for gpgpusim

/*
 * [한국어 설명] GPGPU-Sim 트레이스/디버그 출력 시스템 헤더 (trace.h)
 *
 * === 파일의 역할 ===
 * gem5의 트레이스 시스템에서 영감을 받아 GPGPU-Sim에 맞게 단순화한 조건부 디버그
 * 출력 시스템을 정의한다. Trace 네임스페이스 안에 트레이스 스트림 열거형(enum),
 * 활성화 플래그 배열, 설정 문자열 등을 선언하고, DTRACE/DPRINTF/DPRINTFG 매크로를
 * 제공한다. TRACING_ON 컴파일 플래그에 따라 전체 트레이스 출력을 활성화/비활성화한다.
 * trace_streams.tup 파일에 정의된 스트림(WARP_SCHEDULER, SCOREBOARD, 등)별로
 * 개별 활성화가 가능하다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: gpgpusim.config의 Trace::config_str → Trace::init() → 각 스트림 활성화
 *   → DTRACE(x) 조건 검사 → DPRINTF(x, ...) 조건부 출력
 * DPRINTF는 m_gpu 포인터(멤버 함수 내)에서, DPRINTFG는 글로벌 변수 gpu_sim_cycle에서
 * 현재 사이클 번호를 읽는다. 실행 컨텍스트: 호스트 유저스페이스.
 *
 * === 타 모듈과의 연결 ===
 * trace_streams.tup: 트레이스 스트림 이름 목록 정의 (X-매크로 패턴으로 include)
 * trace.cc: Trace 네임스페이스 전역 변수 정의 및 init() 구현
 * 사용처: shader.cc (WARP_SCHEDULER 스트림), scoreboard.cc (SCOREBOARD 스트림),
 *          mem_fetch.cc (MEMORY_PARTITION_UNIT 등)에서 DTRACE/DPRINTF 호출
 * gpgpusim.config: "-gpgpu_shader_core_pipeline_debug_test" 등 옵션으로
 *                   Trace::config_str을 설정하여 스트림별 활성화
 *
 * === 주요 함수/구조체 요약 ===
 * Trace::init()              - config_str을 파싱하여 스트림별 enabled 플래그 설정
 * DTRACE(x)                  - 스트림 x의 활성화 여부 확인 (bool 표현식)
 * DPRINTF(x, ...)            - m_gpu 기반 사이클 번호 포함 조건부 printf (멤버 함수용)
 * DPRINTFG(x, ...)           - 글로벌 변수 기반 사이클 번호 포함 조건부 printf (전역 함수용)
 */

#ifndef __TRACE_H__
#define __TRACE_H__

namespace Trace {

/* [한국어] X-매크로 패턴으로 trace_streams.tup을 포함해 trace_streams_type enum을 생성.
 * TS_TUP_BEGIN/TS_TUP/TS_TUP_END 매크로를 각각 enum 시작/항목/끝으로 재정의하여
 * trace_streams.tup을 include하면 열거형이 자동 생성된다.
 * 결과: enum trace_streams_type { WARP_SCHEDULER, SCOREBOARD, ..., NUM_TRACE_STREAMS }; */
#define TS_TUP_BEGIN(X) enum X {   /* [한국어] enum 선언 시작: "enum trace_streams_type {" */
#define TS_TUP(X) X                /* [한국어] 각 스트림 이름을 enum 항목으로 */
#define TS_TUP_END(X) \
  }                   \
  ;                   /* [한국어] enum 선언 종료: "};" */
#include "trace_streams.tup"
/* [한국어] trace_streams.tup 포함으로 WARP_SCHEDULER, SCOREBOARD,
 * MEMORY_PARTITION_UNIT, MEMORY_SUBPARTITION_UNIT, INTERCONNECT,
 * LIVENESS, NUM_TRACE_STREAMS 열거값 생성 */
#undef TS_TUP_BEGIN
#undef TS_TUP
#undef TS_TUP_END
/* [한국어] 매크로 재사용 방지를 위해 undef. trace.cc에서 다시 다른 의미로 재정의 */

extern bool enabled;
/* [한국어] 전체 트레이스 시스템 마스터 활성화 플래그.
 * trace.cc에서 false로 초기화. Trace::init() 호출 시 config_str에 의해 설정.
 * 읽는 자: DTRACE 매크로가 스트림별 플래그와 AND하여 출력 여부 결정. */

extern int sampling_core;
/* [한국어] 트레이스 출력을 특정 SM(shader core)으로 제한할 때 사용하는 SM 번호.
 * 0으로 초기화. 특정 SM만 트레이스하는 선택적 필터링용. */

extern int sampling_memory_partition;
/* [한국어] 트레이스 출력을 특정 메모리 파티션으로 제한할 때 사용하는 파티션 번호.
 * -1로 초기화(전체 파티션 대상). */

extern const char* trace_streams_str[];
/* [한국어] 각 트레이스 스트림 enum 값에 대응하는 이름 문자열 배열.
 * trace.cc에서 X-매크로로 생성: {"WARP_SCHEDULER", "SCOREBOARD", ...}.
 * DPRINTF 매크로가 출력 시 스트림 이름으로 사용. */

extern bool trace_streams_enabled[NUM_TRACE_STREAMS];
/* [한국어] 스트림별 활성화 플래그 배열. 인덱스는 trace_streams_type enum 값.
 * trace.cc에서 모두 false로 초기화. init()에서 config_str 파싱으로 개별 설정.
 * DTRACE(x) 매크로가 이 배열의 x번 항목을 조회. */

extern const char* config_str;
/* [한국어] gpgpusim.config에서 읽은 트레이스 설정 문자열.
 * 어떤 스트림을 활성화할지 나타내는 공백/쉼표 구분 스트림 이름 목록.
 * init()이 이 문자열을 파싱하여 trace_streams_enabled 배열 설정. */

/*
 * [한국어]
 * Trace::init() - 트레이스 스트림 초기화 (config_str 파싱)
 *
 * config_str에서 각 스트림 이름을 검색하여 발견되면 해당 스트림의
 * trace_streams_enabled 플래그를 true로 설정한다.
 * 실행 컨텍스트: 시뮬레이터 초기화 시 1회 호출.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc 또는 옵션 파서 → [Trace::init()] → trace_streams_enabled 갱신
 */
void init();

}  // namespace Trace

#if TRACING_ON
/* [한국어] TRACING_ON 컴파일 플래그가 정의된 경우에만 실제 트레이스 출력 활성화.
 * 미정의 시 DTRACE/DPRINTF/DPRINTFG는 아무 동작 없는 no-op으로 컴파일됨. */

#define SIM_PRINT_STR "GPGPU-Sim Cycle %llu: %s - "
/* [한국어] 모든 트레이스 출력 앞에 붙는 공통 접두어 포맷 문자열.
 * %llu: 현재 시뮬레이션 사이클 번호 (gpu_sim_cycle + gpu_tot_sim_cycle)
 * %s: 트레이스 스트림 이름 (WARP_SCHEDULER, SCOREBOARD 등) */

#define DTRACE(x) ((Trace::trace_streams_enabled[Trace::x]) && Trace::enabled)
/* [한국어] 트레이스 스트림 x가 활성화되어 있는지 확인하는 bool 표현식.
 * 스트림별 enabled와 마스터 enabled 둘 다 true일 때만 true 반환.
 * 단독 조건 검사용 또는 DPRINTF 내부에서 사용. */

#define DPRINTF(x, ...)                                                      \
  do {                                                                       \
    if (DTRACE(x)) {                                                         \
      printf(SIM_PRINT_STR, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, \
             Trace::trace_streams_str[Trace::x]);                            \
      printf(__VA_ARGS__);                                                   \
    }                                                                        \
  } while (0)
/* [한국어] 멤버 함수 내에서 사용하는 조건부 트레이스 출력 매크로.
 * m_gpu 포인터(gpgpu_sim*)로 현재 사이클 번호를 읽는다.
 * 스트림 x가 활성화된 경우에만 "GPGPU-Sim Cycle N: STREAM - <메시지>" 형식 출력.
 * do-while(0) 패턴으로 if/else 문에서 안전하게 사용 가능. */

#define DPRINTFG(x, ...)                                       \
  do {                                                         \
    if (DTRACE(x)) {                                           \
      printf(SIM_PRINT_STR, gpu_sim_cycle + gpu_tot_sim_cycle, \
             Trace::trace_streams_str[Trace::x]);              \
      printf(__VA_ARGS__);                                     \
    }                                                          \
  } while (0)
/* [한국어] 글로벌 함수 또는 네임스페이스 스코프에서 사용하는 조건부 트레이스 출력 매크로.
 * DPRINTF와 동일하지만 m_gpu 대신 전역 변수 gpu_sim_cycle/gpu_tot_sim_cycle을 사용.
 * gpu-sim.cc의 전역 함수에서 사용. */

#else
/* [한국어] TRACING_ON 미정의 시 모든 트레이스 매크로를 no-op으로 대체.
 * 릴리스 빌드에서 트레이스 코드가 완전히 제거되어 성능 오버헤드 없음. */

#define DTRACE(x) (false)                /* [한국어] 항상 false — 트레이스 비활성 */
#define DPRINTF(x, ...) \
  do {                  \
  } while (0)           /* [한국어] 완전한 no-op — 컴파일러에 의해 제거됨 */
#define DPRINTFG(x, ...) \
  do {                   \
  } while (0)            /* [한국어] 완전한 no-op — 컴파일러에 의해 제거됨 */

#endif

#endif
