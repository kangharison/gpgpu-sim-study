// Copyright (c) 2009-2011, Tor M. Aamodt, George L. Yuan, Andrew Turner,
// Ali Bakhoda
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
 * [한국어 설명] GPU 시뮬레이터 공통 유틸리티 헤더 (gpu-misc.h)
 *
 * === 파일의 역할 ===
 * GPGPU-Sim 타이밍 모델 전체에서 공통으로 사용하는 소규모 유틸리티 함수와
 * 매크로를 선언하는 헤더 파일이다. 주요 내용은 두 가지이다: (1) 2의 지수를
 * 빠르게 계산하는 LOGB2() 함수 — 캐시 인덱스/오프셋 비트 수 계산 등에 광범위하게
 * 사용된다; (2) 2~3개 인자의 최솟값을 구하는 매크로 (gs_min2, min3) — 사이클 수,
 * 길이, 크기 비교 등에 활용된다. 또한 L1 캐시 미스 및 MSHR 상태 변화를 상세히
 * 출력하는 디버그 플래그(DEBUGL1MISS)를 정의한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * gpgpu-sim/ 디렉토리 하위의 거의 모든 타이밍 모델 파일(shader.cc, gpu-cache.cc,
 * dram.cc, mem_fetch.cc, addrdec.cc 등)이 이 헤더를 포함한다. LOGB2()는 특히
 * gpu-cache.cc와 addrdec.cc에서 캐시 셋 수·웨이 수·뱅크 수가 2의 거듭제곱임을
 * 전제로 비트 수를 추출할 때 핵심적으로 쓰인다. 이 파일 자체는 다른 시뮬레이터
 * 파일에 의존하지 않는 최하위 유틸리티 계층이다.
 * 실행 컨텍스트: 호스트 유저스페이스 (시뮬레이터 초기화 및 사이클 루프 전반).
 *
 * === 타 모듈과의 연결 ===
 * - 이 파일이 의존하는 모듈: 없음 (독립적인 최하위 유틸리티)
 * - 이 파일에 의존하는 모듈: gpu-cache.h/cc, shader.h/cc, dram.h/cc,
 *   mem_fetch.h/cc, addrdec.h/cc, delayqueue.h 등 gpgpu-sim/ 전반
 * - 데이터 흐름: LOGB2()는 캐시 파라미터(셋 수, 블록 크기 등)를 2의 지수로 변환하여
 *   비트 필드 추출에 쓰이는 정수 값을 반환한다. 매크로는 두 정수값을 비교하여
 *   더 작은 값을 인라인으로 반환한다.
 *
 * === 주요 함수/구조체 요약 ===
 * - LOGB2(v): 정수 v에 대해 floor(log2(v))를 분기 없이 비트 조작으로 계산
 * - gs_min2(a,b): 두 값 중 최솟값 반환 매크로
 * - min3(x,y,z): 세 값 중 최솟값 반환 매크로 (gs_min2 기반)
 * - DEBUGL1MISS: L1 캐시 미스 및 MSHR 상태 변화 디버그 출력 활성화 플래그
 */

#ifndef GPU_MISC_H
#define GPU_MISC_H

// enables a verbose printout of all L1 cache misses and all MSHR status changes
// good for a single shader configuration
/* [한국어] L1 캐시 미스 전체와 MSHR(Miss Status Holding Register) 상태 변화를
 * 상세히 출력하는 디버그 모드 플래그.
 * - 0: 비활성 (기본값, 성능 측정 시 사용)
 * - 1: 활성 — SM이 하나인 단일 셰이더 구성에서 L1 미스 패턴 추적 시 활성화
 * 설정자: 개발자가 빌드 전에 소스 수정으로 설정; 런타임 변경 불가.
 * 읽는 자: gpu-cache.cc 내 L1 캐시 접근 및 MSHR 삽입/삭제 경로.
 * 값 범위: 0(비활성) 또는 1(활성). */
#define DEBUGL1MISS 0

/* [한국어] LOGB2 - 부호 없는 32비트 정수 v에 대해 floor(log2(v))를 계산한다.
 * 캐시 셋 수, 캐시 블록 크기, 뱅크 수 등이 반드시 2의 거듭제곱이어야 하는
 * GPGPU-Sim의 캐시/메모리 모델에서 비트 필드 너비를 구할 때 광범위하게 사용된다.
 * 구현은 분기 없이 비트 조작만으로 동작하므로 분기 예측 미스가 없다.
 * 정의는 gpu-misc.cc 참고. */
unsigned int LOGB2(unsigned int v);

/* [한국어] gs_min2(a, b) - 두 값 a, b 중 더 작은 값을 반환하는 인라인 매크로.
 * 타입 독립적으로 동작하며 사이클 수, 큐 길이, 크기 등 다양한 정수 비교에 사용.
 * 주의: a, b에 부작용(side-effect)이 있는 표현식을 넣으면 두 번 평가되므로 금지. */
#define gs_min2(a, b) (((a) < (b)) ? (a) : (b))

/* [한국어] min3(x, y, z) - 세 값 x, y, z 중 가장 작은 값을 반환하는 인라인 매크로.
 * 내부적으로 gs_min2를 재사용하여 구현한다. x < y && x < z이면 x를,
 * 그렇지 않으면 gs_min2(y, z)를 반환한다.
 * 주의: gs_min2와 동일하게 인자에 부작용 있는 표현식 사용 금지. */
#define min3(x, y, z) (((x) < (y) && (x) < (z)) ? (x) : (gs_min2((y), (z))))

#endif
