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

/*
 * [한국어 설명] GPGPU-Sim 트레이스 시스템 구현 (trace.cc)
 *
 * === 파일의 역할 ===
 * trace.h에서 선언된 Trace 네임스페이스의 전역 변수를 정의하고 init() 함수를 구현한다.
 * X-매크로 패턴으로 trace_streams.tup을 재포함하여 스트림 이름 문자열 배열
 * trace_streams_str[]을 생성한다. init()은 gpgpusim.config에서 전달된 config_str을
 * 파싱하여 어떤 트레이스 스트림을 활성화할지 결정한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: gpgpusim.config 파싱 → Trace::config_str 설정 → Trace::init() 호출
 *   → trace_streams_enabled[] 갱신 → 각 모듈의 DTRACE/DPRINTF 매크로가 이를 참조
 * 실행 컨텍스트: 시뮬레이터 초기화 단계(1회), 이후는 읽기 전용.
 *
 * === 타 모듈과의 연결 ===
 * 의존: trace.h (전역 변수 선언), trace_streams.tup (스트림 이름 목록),
 *       string.h (strstr 문자열 검색)
 * 데이터 흐름: config_str(gpgpusim.config) → init() → trace_streams_enabled[]
 *   → DTRACE(x)/DPRINTF(x,...) 매크로 (shader.cc, scoreboard.cc, mem_fetch.cc 등)
 *
 * === 주요 함수/구조체 요약 ===
 * trace_streams_str[]        - 스트림 enum 값에 대응하는 이름 문자열 배열 (자동 생성)
 * enabled, sampling_core ... - Trace 네임스페이스 전역 상태 변수 정의
 * Trace::init()              - config_str 파싱으로 스트림별 활성화 플래그 설정
 */

#include "trace.h"  /* [한국어] Trace 네임스페이스 선언 및 매크로 정의 */
#include "string.h" /* [한국어] strstr() 문자열 부분 검색 함수 */

namespace Trace {

/* [한국어] X-매크로를 재정의하여 trace_streams.tup을 포함함으로써
 * trace_streams_str[] 배열을 생성. 각 스트림 이름이 문자열 리터럴로 변환됨.
 * 결과: const char* trace_streams_str[] = {"WARP_SCHEDULER", "SCOREBOARD", ...}; */
#define TS_TUP_BEGIN(X) const char* trace_streams_str[] = {
/* [한국어] 배열 선언 시작 */
#define TS_TUP(X) #X
/* [한국어] 각 스트림 이름을 문자열 리터럴로 변환 (#X = stringify) */
#define TS_TUP_END(X) \
  }                   \
  ;
/* [한국어] 배열 선언 종료 */
#include "trace_streams.tup"
/* [한국어] X-매크로 전개: {"WARP_SCHEDULER","SCOREBOARD","MEMORY_PARTITION_UNIT",
 *                           "MEMORY_SUBPARTITION_UNIT","INTERCONNECT","LIVENESS"} */
#undef TS_TUP_BEGIN
#undef TS_TUP
#undef TS_TUP_END
/* [한국어] 이후 코드에서 매크로 오염 방지를 위해 undef */

bool enabled = false;
/* [한국어] 트레이스 마스터 활성화 플래그. 기본값 false(비활성).
 * config_str에 스트림 이름이 하나라도 존재하면 init()에서 true로 설정될 수 있음.
 * 읽는 자: DTRACE(x) 매크로. 쓰는 자: init() 또는 gpgpusim_entrypoint. */

int sampling_core = 0;
/* [한국어] 트레이스 출력을 특정 SM으로 제한할 때의 SM(shader core) 번호.
 * 0으로 초기화. 현재 코드에서는 직접 사용하지 않으며 설정 목적으로만 선언됨. */

int sampling_memory_partition = -1;
/* [한국어] 트레이스 출력을 특정 메모리 파티션으로 제한할 때의 파티션 번호.
 * -1로 초기화(전체 파티션 대상). 현재 코드에서는 직접 사용하지 않음. */

bool trace_streams_enabled[NUM_TRACE_STREAMS] = {false};
/* [한국어] 스트림별 활성화 플래그 배열. 인덱스 = trace_streams_type enum 값.
 * 모두 false로 초기화. init()에서 config_str 파싱 결과에 따라 개별 설정.
 * 동기화: init() 호출 후에는 읽기 전용으로 접근되므로 락 불필요. */

const char* config_str;
/* [한국어] gpgpusim.config에서 읽어온 트레이스 설정 문자열 포인터.
 * 어떤 스트림을 활성화할지 나타내는 스트림 이름 목록 (예: "WARP_SCHEDULER SCOREBOARD").
 * init() 호출 전에 외부에서 설정되어야 함. */

/*
 * [한국어]
 * Trace::init() - 트레이스 스트림 초기화 (config_str 파싱)
 *
 * @없음 (Trace 네임스페이스 함수)
 *
 * config_str을 순회하면서 각 스트림 이름 문자열이 포함되어 있는지 strstr()로 검사하고,
 * 발견되면 해당 스트림의 trace_streams_enabled[i]를 true로 설정한다.
 * NUM_TRACE_STREAMS개의 스트림을 선형 탐색하므로 O(n) 시간 복잡도.
 * 시뮬레이터 초기화 시 1회 호출되며 이후 trace_streams_enabled[]는 읽기 전용.
 * 실행 컨텍스트: 시뮬레이터 초기화 단계, 호스트 유저스페이스, 단일 스레드.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc → [Trace::init()] → trace_streams_enabled[] 갱신
 */
void init() {
  for (unsigned i = 0; i < NUM_TRACE_STREAMS; ++i) {
    /* [한국어] 각 스트림 이름(0부터 NUM_TRACE_STREAMS-1까지) 순회 */
    if (strstr(config_str, trace_streams_str[i]) != NULL) {
      /* [한국어] config_str 내에 스트림 이름 문자열이 존재하면 활성화.
       * 예: config_str="WARP_SCHEDULER SCOREBOARD"이면 두 스트림 활성화. */
      trace_streams_enabled[i] = true; /* [한국어] 해당 스트림 활성화 */
    }
  }
}
}  // namespace Trace
