// Copyright (c) 2009-2011, Tor M. Aamodt, George L. Yuan
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
 * [한국어 설명] GPU 시뮬레이터 공통 유틸리티 구현 (gpu-misc.cc)
 *
 * === 파일의 역할 ===
 * gpu-misc.h에 선언된 LOGB2() 함수를 구현한다. LOGB2(v)는 양의 정수 v에 대해
 * floor(log2(v))를 분기 없이 비트 조작(bit manipulation)만으로 계산한다. 이는
 * 캐시 파라미터(셋 수, 캐시 라인 크기, 뱅크 수 등)가 항상 2의 거듭제곱으로
 * 설정되는 GPGPU-Sim에서 비트 인덱스 너비를 구하는 핵심 연산이다. 예를 들어
 * LOGB2(64) = 6 이면 캐시 인덱스 필드는 6비트라는 의미이다. 파일 자체의 크기는
 * 매우 작지만, 거의 모든 캐시/메모리 모델 초기화 경로에서 호출된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 이 파일은 gpgpu-sim/ 타이밍 모델 계층의 최하위 유틸리티에 해당한다. 시뮬레이터
 * 초기화 시 gpu-cache.cc의 cache_config::init(), addrdec.cc의 linear_to_raw_address()
 * 등이 LOGB2()를 호출하여 비트 오프셋과 마스크를 계산한다. LOGB2()가 반환하는 값은
 * 이후 주소 분해(address decode) 및 캐시 인덱싱 전반에 걸쳐 상수처럼 재사용된다.
 * 실행 컨텍스트: 호스트 유저스페이스 — GPU 커널 런치 전 설정 파싱 단계에서 한 번,
 * 이후 사이클 루프 중 필요 시 재호출된다.
 *
 * === 타 모듈과의 연결 ===
 * - 이 파일이 의존하는 모듈: gpu-misc.h (자신의 선언)
 * - 이 파일에 의존하는 모듈: gpu-cache.cc, addrdec.cc, shader.cc, dram.cc 등
 *   gpgpu-sim/ 하위 대부분의 구현 파일
 * - 데이터 흐름: 호출자가 2의 거듭제곱 정수를 넘기면 log2 값(비트 너비)을 반환한다.
 *   반환된 값은 비트 마스크, 시프트 양, 배열 인덱스 계산에 사용된다.
 *
 * === 주요 함수/구조체 요약 ===
 * - LOGB2(v): 32비트 정수 v에 대한 floor(log2(v)) 계산.
 *   알고리즘: 16비트 → 8비트 → 4비트 → 2비트 → 1비트 순으로 이진 탐색(binary search)
 *   방식의 비트 분할을 수행하여 최상위 비트 위치를 결정한다. 조건 분기 없이
 *   비교 결과를 정수로 강제 변환(bool→int 캐스트)하여 시프트와 누적으로만 구현된다.
 */

/* [한국어] gpu-misc.h 포함 — LOGB2 선언 및 DEBUGL1MISS, gs_min2, min3 매크로 정의 */
#include "gpu-misc.h"

/*
 * [한국어]
 * LOGB2 - 부호 없는 32비트 정수 v에 대해 floor(log2(v))를 분기 없이 계산한다.
 *
 * @v: 로그를 계산할 양의 정수. 반드시 2의 거듭제곱이어야 올바른 비트 너비가 반환된다.
 *     (2의 거듭제곱이 아닌 경우 floor 값이 반환되지만, 호출자는 이를 사용하지 않는다.)
 * @return: floor(log2(v)) — v를 2진수로 표현했을 때 최상위 비트의 위치(0-based 인덱스).
 *          예) LOGB2(1)=0, LOGB2(2)=1, LOGB2(4)=2, LOGB2(64)=6, LOGB2(1024)=10.
 *          호출자는 반환값을 비트 시프트 양 또는 비트 마스크 계산에 직접 사용한다.
 *
 * 이 함수는 캐시 셋 수(num_sets), 캐시 라인 크기(line_size), 뱅크 수(bank_num) 등
 * 반드시 2의 거듭제곱으로 설정되는 GPGPU-Sim 파라미터로부터 비트 필드 너비를
 * 추출하기 위해 존재한다. 예를 들어 L1 캐시 셋 수가 64이면 LOGB2(64)=6으로
 * 인덱스 필드가 6비트임을 알 수 있다.
 *
 * 알고리즘: 이진 탐색(binary search on bits) 방식으로 5단계에 걸쳐 최상위 비트 위치를
 * 결정한다. 각 단계에서 상위 절반 비트 범위에 1이 있으면 해당 비트 수만큼 r에 누적하고
 * v를 오른쪽으로 시프트한다. 조건 분기(if/else) 없이 비교 결과(0 or 1)를 비트
 * 시프트로 변환하여 누적하므로 분기 예측 미스가 없다.
 *
 * 실행 컨텍스트: 호스트 유저스페이스 단일 스레드. 재진입 가능 (전역 상태 없음).
 * 호출 체인: gpu-cache.cc::cache_config::init() → [LOGB2]
 *            addrdec.cc::linear_to_raw_address_translation() → [LOGB2]
 */
unsigned int LOGB2(unsigned int v) {
  unsigned int shift; /* [한국어] 현재 단계에서 r에 누적할 비트 시프트 양 (0 또는 해당 단계의 비트 수) */
  unsigned int r;     /* [한국어] 누적 결과값 — 각 단계의 shift를 OR 연산으로 합산하여 최종 log2 값이 된다 */

  r = 0; /* [한국어] 누적 결과를 0으로 초기화 — 이후 단계별로 비트를 OR하여 채운다 */

  /* [한국어] 1단계: 상위 16비트(비트 16~31) 범위에 1이 있으면 shift=16, 없으면 shift=0.
   * 0xFFFF0000 마스크로 상위 워드를 검사하고, 결과(0 or 1)를 4비트 좌시프트하여 16을 만든다. */
  shift = ((v & 0xFFFF0000) != 0) << 4;
  v >>= shift; /* [한국어] shift가 16이면 v를 16비트 오른쪽 시프트 — 상위 16비트를 제거하고 탐색 범위를 좁힌다 */
  r |= shift;  /* [한국어] shift를 r에 OR로 누적 — 최상위 비트가 상위 16비트에 있었음을 기록 */

  /* [한국어] 2단계: 남은 v의 상위 8비트(비트 8~15) 범위에 1이 있으면 shift=8, 없으면 0 */
  shift = ((v & 0xFF00) != 0) << 3;
  v >>= shift; /* [한국어] 8비트 시프트로 탐색 범위를 다시 절반으로 줄인다 */
  r |= shift;  /* [한국어] 8을 r에 누적 */

  /* [한국어] 3단계: 남은 v의 상위 4비트(비트 4~7) 범위에 1이 있으면 shift=4, 없으면 0 */
  shift = ((v & 0xF0) != 0) << 2;
  v >>= shift; /* [한국어] 4비트 시프트 */
  r |= shift;  /* [한국어] 4를 r에 누적 */

  /* [한국어] 4단계: 남은 v의 상위 2비트(비트 2~3) 범위에 1이 있으면 shift=2, 없으면 0.
   * 0xC = 0b1100 마스크로 비트 2~3을 검사한다. */
  shift = ((v & 0xC) != 0) << 1;
  v >>= shift; /* [한국어] 2비트 시프트 */
  r |= shift;  /* [한국어] 2를 r에 누적 */

  /* [한국어] 5단계(최종): 남은 v의 비트 1에 1이 있으면 shift=1, 없으면 0.
   * 이 시점에서 v는 최대 2비트 값(0~3)으로 좁혀진 상태이다. */
  shift = ((v & 0x2) != 0) << 0;
  v >>= shift; /* [한국어] 1비트 시프트 (이후 v는 사용되지 않음) */
  r |= shift;  /* [한국어] 1을 r에 누적 — 이로써 r = floor(log2(원래 v)) 계산 완료 */

  return r; /* [한국어] 최상위 비트 위치(0-based) 반환 — 호출자가 비트 너비로 사용 */
}
