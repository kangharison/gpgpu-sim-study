/*
 * [한국어 설명] 메모리 파티션(L2 캐시 + DRAM) 전용 디버그 트레이스 매크로 (l2cache_trace.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 메모리 파티션(memory_partition_unit) 및 메모리 서브파티션
 * (memory_sub_partition) 내부 컴포넌트에서 사이클 단위 동작을 선택적으로 출력하기 위한
 * 디버그 트레이스 매크로를 정의한다. GPU에는 여러 개의 메모리 파티션(각각 L2 캐시 슬라이스
 * + DRAM 채널 하나씩)이 있으며, 이 파일은 전역 트레이스 시스템(trace.h) 위에 "특정
 * 메모리 파티션만 모니터링"하는 필터링 레이어를 추가한다. shader_trace.h가 SM 측 필터를
 * 담당하는 것과 대칭적으로, 이 파일은 메모리 측 필터를 담당한다. TRACING_ON이 비활성화된
 * 빌드에서는 모든 매크로가 no-op으로 컴파일되어 성능 오버헤드가 전혀 없다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 타이밍 모델(gpgpu-sim/)의 메모리 서브시스템 최하단 공통 인프라에 해당한다.
 * GPU의 메모리 계층 구조:
 *   SM (shader core) → L1 캐시 → 인터커넥트(ICNT/intersim2)
 *     → memory_partition_unit (L2 캐시 + DRAM 컨트롤러)
 *         ├── memory_sub_partition (L2 슬라이스 + 요청 큐)
 *         └── DRAM 타이밍 모델 (dram.cc)
 * 호출 체인:
 *   gpu-sim.cc (cycle()) → memory_partition_unit::cache_cycle() / dram_cycle()
 *     → MEMPART_DPRINTF / MEM_SUBPART_DPRINTF 매크로 호출
 *         → trace.h의 DTRACE + Trace::sampling_memory_partition 필터 평가
 *             → 조건 충족 시 printf로 사이클 + 채널명 + 파티션 ID 출력
 * 실행 컨텍스트: 호스트 유저스페이스의 사이클-레벨 시뮬레이션 루프에서 실행.
 * 단일 스레드 시뮬레이션이므로 별도 동기화는 불필요하다.
 *
 * === 타 모듈과의 연결 ===
 * 의존:
 *   - ../trace.h : DTRACE(x), SIM_PRINT_STR, Trace::sampling_memory_partition,
 *                  Trace::trace_streams_str[] 등 전역 트레이스 인프라 제공.
 *                  MEMORY_PARTITION_UNIT, MEMORY_SUBPARTITION_UNIT 채널 enum도
 *                  trace.h에서 정의된다.
 * 사용처:
 *   - gpu-cache.cc / l2cache.cc : memory_partition_unit, memory_sub_partition 클래스가
 *     L2 캐시 접근, 미스 처리, DRAM 요청 발행 시 이 매크로로 디버그 출력을 남긴다.
 *   - dram.cc : DRAM 컨트롤러에서 메모리 요청 스케줄링 및 완료 처리 시 사용.
 * 데이터 흐름:
 *   Trace::sampling_memory_partition ← gpgpusim.config의 해당 설정 옵션
 *   → MEMPART_DTRACE / MEM_SUBPART_DTRACE 필터 → printf 출력
 * 파티션 ID 접근 방법:
 *   - memory_partition_unit: get_mpid() 멤버 함수 반환값
 *   - memory_sub_partition: m_id 멤버 변수
 *
 * === 주요 함수/구조체 요약 ===
 * MEMPART_DTRACE(x)       : 채널 x 활성화 + 현재 메모리 파티션이 모니터링 대상일 때 true.
 * MEM_SUBPART_DTRACE(x)   : 채널 x 활성화 + 현재 메모리 서브파티션이 모니터링 대상일 때 true.
 * MEMPART_DPRINTF(...)    : memory_partition_unit 내부용 포맷 출력. get_mpid() 의존.
 * MEM_SUBPART_DPRINTF(..) : memory_sub_partition 내부용 포맷 출력. m_id 멤버 의존.
 * MEMPART_PRINT_STR       : MEMPART_DPRINTF 출력 포맷 문자열 ("... %d - ").
 * MEM_SUBPART_PRINT_STR   : MEM_SUBPART_DPRINTF 출력 포맷 문자열 ("... %d - ").
 */

// Copyright (c) 2009-2011, Tor M. Aamodt, Tim Rogers, Wilson W. L. Fung
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

/* [한국어] #pragma once — 헤더 중복 포함 방지. #ifndef 가드와 동일한 효과지만
 * 컴파일러 확장 지시어로, 현대 컴파일러(GCC, Clang, MSVC)에서 표준적으로 지원된다.
 * shader_trace.h가 #ifndef 가드를 사용하는 것과 달리 이 파일은 #pragma once를 사용한다. */
#pragma once

/* [한국어] trace.h 포함 — 전역 트레이스 인프라(DTRACE, SIM_PRINT_STR,
 * Trace::sampling_memory_partition, Trace::trace_streams_str[]) 및
 * 채널 enum(MEMORY_PARTITION_UNIT, MEMORY_SUBPARTITION_UNIT 등)을 가져온다.
 * 이 파일의 모든 매크로는 trace.h 없이는 동작하지 않는다. */
#include "../trace.h"

/* [한국어] TRACING_ON 전처리 분기 — 빌드 시 -DTRACING_ON=1 플래그로 결정.
 * 활성화(=1): 실제 트레이스 매크로가 컴파일에 포함된다.
 * 비활성화(=0 또는 미정의): #else 블록의 no-op 매크로들이 대신 삽입되어
 * 트레이싱 코드가 완전히 제거되므로 성능 측정 빌드에서 오버헤드가 없다. */
#if TRACING_ON

/* [한국어] MEMPART_PRINT_STR — MEMPART_DPRINTF가 사용하는 출력 접두어 포맷 문자열.
 * SIM_PRINT_STR(trace.h 정의, 예: "%llu ")에 " %d - "를 이어 붙인다.
 * 최종 형식: "<총_사이클> [채널명] <파티션_ID> - <사용자 메시지>"
 * %d 자리에는 get_mpid()(메모리 파티션 ID, 0-based)가 대입된다.
 * 주의: shader_trace.h의 SHADER_PRINT_STR과 달리 "Core" 같은 레이블 없이
 * 숫자만 출력하는 단순한 형식이다. */
#define MEMPART_PRINT_STR SIM_PRINT_STR " %d - "

/* [한국어] MEMPART_DTRACE(x) — 메모리 파티션 레벨 트레이스 활성화 조건 평가 매크로.
 *
 * 조건:
 *   1. DTRACE(x): 채널 x가 전역적으로 활성화되어 있는지 확인 (trace.h 정의).
 *   2. Trace::sampling_memory_partition == -1:
 *      모든 메모리 파티션의 출력을 허용하는 와일드카드 값 (기본값).
 *   3. Trace::sampling_memory_partition == (int)get_mpid():
 *      현재 파티션이 gpgpusim.config에서 지정한 특정 파티션과 일치하는지 확인.
 *      get_mpid()는 memory_partition_unit의 멤버 함수로 파티션 ID를 반환한다.
 *
 * 전체 조건: DTRACE(x) && (지정_파티션 == -1 || 현재_파티션 == 지정_파티션)
 * 이 매크로는 memory_partition_unit 또는 그 내부 컴포넌트 클래스 범위에서만 올바르게
 * 동작한다 (get_mpid() 멤버 함수 접근 필요). */
#define MEMPART_DTRACE(x)                                  \
  (DTRACE(x) && (Trace::sampling_memory_partition == -1 || \
                 Trace::sampling_memory_partition == (int)get_mpid()))

/* [한국어] MEM_SUBPART_PRINT_STR — MEM_SUBPART_DPRINTF가 사용하는 출력 접두어 포맷.
 * MEMPART_PRINT_STR과 동일한 형식: SIM_PRINT_STR + " %d - ".
 * %d 자리에는 memory_sub_partition::m_id(서브파티션 ID)가 대입된다.
 * 메모리 파티션과 서브파티션의 ID 체계가 다를 수 있으므로 별도 매크로로 분리되어 있다.
 * (각 memory_partition_unit 아래에 복수의 memory_sub_partition이 있을 수 있다.) */
#define MEM_SUBPART_PRINT_STR SIM_PRINT_STR " %d - "

/* [한국어] MEM_SUBPART_DTRACE(x) — 메모리 서브파티션 레벨 트레이스 활성화 조건 평가.
 *
 * 조건:
 *   1. DTRACE(x): 채널 x가 전역적으로 활성화되어 있는지 확인.
 *   2. Trace::sampling_memory_partition == -1: 와일드카드 — 모든 파티션 허용.
 *   3. Trace::sampling_memory_partition == (int)m_id:
 *      현재 서브파티션의 m_id가 지정된 파티션 ID와 일치하는지 확인.
 *      m_id는 memory_sub_partition의 직접 멤버 변수이다.
 *
 * MEMPART_DTRACE와의 차이: get_mpid() 대신 m_id를 사용한다.
 * memory_sub_partition은 상위 파티션의 인터페이스(get_mpid())에 직접 접근하지 않고
 * 자신의 m_id 멤버로 필터링한다. 이 매크로는 memory_sub_partition 클래스 범위에서만
 * 올바르게 동작한다. */
#define MEM_SUBPART_DTRACE(x)                              \
  (DTRACE(x) && (Trace::sampling_memory_partition == -1 || \
                 Trace::sampling_memory_partition == (int)m_id))

// Intended to be called from inside components of a memory partition
// Depends on a get_mpid() function
/* [한국어] MEMPART_DPRINTF(...) — memory_partition_unit 내부 컴포넌트용 포맷 출력 매크로.
 *
 * @... : printf 스타일 포맷 문자열 및 인수들 (__VA_ARGS__로 전달됨).
 *
 * 동작:
 *   1. MEMPART_DTRACE(MEMORY_PARTITION_UNIT): MEMORY_PARTITION_UNIT 채널 활성화 여부와
 *      현재 파티션이 모니터링 대상인지 동시에 확인한다.
 *   2. 조건 충족 시 MEMPART_PRINT_STR 포맷으로 접두어 출력:
 *      - m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle: 총 누적 시뮬레이션 사이클.
 *        gpu_sim_cycle: 현재 커널 사이클, gpu_tot_sim_cycle: 이전 커널까지 누적 사이클.
 *      - Trace::trace_streams_str[Trace::MEMORY_PARTITION_UNIT]: "MEMORY_PARTITION_UNIT" 문자열.
 *      - get_mpid(): 현재 memory_partition_unit의 파티션 ID (0-based 정수).
 *   3. 사용자 포맷 문자열(__VA_ARGS__)을 두 번째 printf로 출력한다.
 *
 * do { ... } while(0) 패턴:
 *   if-else 구문 등에서 세미콜론이 올바르게 처리되도록 하는 매크로 관용구.
 *
 * 의존: m_gpu 포인터와 get_mpid() 함수가 호출 범위에 있어야 한다.
 *       memory_partition_unit 클래스 또는 그 내부에서만 사용 가능.
 *
 * 호출 체인:
 *   gpu-sim.cc (cycle()) → memory_partition_unit::cache_cycle() / dram_cycle()
 *     → [MEMPART_DPRINTF] → MEMPART_DTRACE(MEMORY_PARTITION_UNIT) → printf */
#define MEMPART_DPRINTF(...)                                                   \
  do {                                                                         \
    if (MEMPART_DTRACE(MEMORY_PARTITION_UNIT)) {                               \
      /* [한국어] 접두어 출력: 총 사이클 + "MEMORY_PARTITION_UNIT" 채널명 + 파티션 ID */ \
      printf(                                                                  \
          MEMPART_PRINT_STR, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,  \
          Trace::trace_streams_str[Trace::MEMORY_PARTITION_UNIT], get_mpid()); \
      /* [한국어] 사용자 정의 메시지 출력 (가변 인수 그대로 전달) */              \
      printf(__VA_ARGS__);                                                     \
    }                                                                          \
  } while (0)

/* [한국어] MEM_SUBPART_DPRINTF(...) — memory_sub_partition 내부 전용 포맷 출력 매크로.
 *
 * @... : printf 스타일 포맷 문자열 및 인수들.
 *
 * MEMPART_DPRINTF와의 차이점:
 *   1. 활성화 조건: MEM_SUBPART_DTRACE(MEMORY_PARTITION_UNIT)를 사용한다.
 *      필터링은 동일하게 Trace::sampling_memory_partition과 m_id를 비교한다.
 *   2. 채널 이름 출력: Trace::MEMORY_SUBPARTITION_UNIT 채널 이름 문자열을 사용한다.
 *      (MEMPART_DPRINTF는 Trace::MEMORY_PARTITION_UNIT을 사용하는 것과 대조적이다.)
 *      이 차이로 인해 출력 로그에서 파티션 레벨과 서브파티션 레벨 이벤트를 구별할 수 있다.
 *   3. ID 출력: get_mpid() 대신 m_id 멤버 변수를 직접 사용한다.
 *
 * 동작:
 *   1. MEM_SUBPART_DTRACE(MEMORY_PARTITION_UNIT): 채널 + m_id 필터 확인.
 *   2. MEM_SUBPART_PRINT_STR 포맷으로 접두어 출력:
 *      - m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle: 총 누적 사이클.
 *      - Trace::trace_streams_str[Trace::MEMORY_SUBPARTITION_UNIT]: 채널 이름 문자열.
 *      - m_id: 현재 memory_sub_partition의 ID.
 *   3. 사용자 포맷 문자열 출력.
 *
 * 의존: m_gpu 포인터와 m_id 멤버 변수가 호출 범위에 있어야 한다.
 *       memory_sub_partition 클래스 또는 그 내부에서만 사용 가능.
 *
 * 호출 체인:
 *   memory_partition_unit → memory_sub_partition::cycle()
 *     → [MEM_SUBPART_DPRINTF] → MEM_SUBPART_DTRACE → printf */
#define MEM_SUBPART_DPRINTF(...)                                               \
  do {                                                                         \
    if (MEM_SUBPART_DTRACE(MEMORY_PARTITION_UNIT)) {                           \
      /* [한국어] 접두어 출력: 총 사이클 + "MEMORY_SUBPARTITION_UNIT" 채널명 + 서브파티션 ID */ \
      printf(MEM_SUBPART_PRINT_STR,                                            \
             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,                  \
             Trace::trace_streams_str[Trace::MEMORY_SUBPARTITION_UNIT], m_id); \
      /* [한국어] 사용자 정의 메시지 출력 (가변 인수 그대로 전달) */              \
      printf(__VA_ARGS__);                                                     \
    }                                                                          \
  } while (0)

/* [한국어] TRACING_ON=0(또는 미정의) 빌드용 no-op 분기.
 * 아래 정의들은 트레이싱이 비활성화된 경우 모든 매크로를 빈 표현식으로 치환한다.
 * 컴파일러가 dead code를 완전히 제거하므로 성능 측정 빌드에서 오버헤드가 없다. */
#else

/* [한국어] MEMPART_DTRACE(x) no-op — 항상 false 반환, MEMPART_DPRINTF 내부 printf 차단. */
#define MEMPART_DTRACE(x) (false)

/* [한국어] MEMPART_DPRINTF(x, ...) no-op — 빈 do-while(0)으로 치환.
 * 세미콜론 처리 일관성을 위해 do-while(0) 패턴을 유지한다. */
#define MEMPART_DPRINTF(x, ...) \
  do {                          \
  } while (0)

/* [한국어] MEM_SUBPART_DTRACE(x) no-op — 항상 false 반환. */
#define MEM_SUBPART_DTRACE(x) (false)

/* [한국어] MEM_SUBPART_DPRINTF(x, ...) no-op — 빈 do-while(0)으로 치환. */
#define MEM_SUBPART_DPRINTF(x, ...) \
  do {                              \
  } while (0)

/* [한국어] TRACING_ON 전처리 분기 종료 */
#endif
