// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
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
 * [한국어 설명] C++03/TR1 unordered_map 컴파일러 호환성 추상화 헤더 (tr1_hash_map.h)
 *
 * === 파일의 역할 ===
 * 컴파일러 버전에 따라 std::unordered_map 또는 std::map을 선택하여 tr1_hash_map이라는
 * 단일 이름으로 추상화하는 호환성 헤더이다. GPGPU-Sim 코드베이스는 원래 C++03 환경에서
 * 작성되었으며, std::unordered_map은 C++11(TR1 경유로 GCC 4.3부터 사용 가능)에서
 * 표준화되었다. 이 헤더를 통해 컴파일러가 지원하면 O(1) 평균 조회의 unordered_map을,
 * 그렇지 않으면 O(log n) 조회의 map을 사용하도록 자동 전환된다. tr1_hash_map_ismap
 * 매크로로 어느 컨테이너가 선택되었는지 런타임에 구분할 수 있다.
 * 또한 _GLIBCXX_DEBUG 모드(디버그 STL)에서는 unordered_map이 오동작하는 알려진
 * 문제를 피하기 위해 강제로 std::map을 사용하도록 분기한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 전체 코드베이스에서 키-값 조회가 필요한 모든 곳(warp ID → warp 상태,
 * 주소 → 캐시 라인 등)에서 이 헤더를 포함하여 tr1_hash_map<K, V>를 사용한다.
 * 이 파일은 컴파일 시점에만 동작하며 런타임 오버헤드가 전혀 없다.
 * 실행 컨텍스트: 컴파일 전처리 단계 (#ifdef/#define).
 *
 * === 타 모듈과의 연결 ===
 * - 이 파일이 의존하는 모듈: 없음 (순수 전처리기 추상화)
 * - 이 파일에 의존하는 모듈: shader.h/cc, gpu-cache.h/cc, cuda-sim/ 등
 *   tr1_hash_map<K, V> 타입을 사용하는 모든 파일
 * - 데이터 흐름: 이 헤더 포함 후 tr1_hash_map<K, V>로 선언하면 컴파일러에 따라
 *   std::unordered_map<K, V> 또는 std::map<K, V>로 해석됨
 *
 * === 주요 함수/구조체 요약 ===
 * - tr1_hash_map: std::unordered_map (GCC >= 4.3, 비디버그 STL) 또는 std::map (그 외)
 * - tr1_hash_map_ismap: 0이면 unordered_map 선택됨, 1이면 map 선택됨
 */

/* [한국어] #pragma once — 중복 포함 방지 (include guard 대체). 모든 주요 컴파일러에서 지원됨. */
#pragma once

// detection and fallback for unordered_map in C++0x
/* [한국어] #ifdef __cplusplus — C++ 컴파일러에서만 이 블록을 활성화한다.
 * C 컴파일러로 이 헤더를 포함하면 아무것도 정의되지 않는다. */
#ifdef __cplusplus
// detect GCC 4.3 or later and use unordered map (part of C++0x)
// unordered map doesn't play nice with _GLIBCXX_DEBUG, just use a map if its
// enabled.
/* [한국어] 1단계 분기: GCC 컴파일러이고(_GLIBCXX_DEBUG 미정의인 경우에만).
 * _GLIBCXX_DEBUG는 STL 컨테이너의 디버그 모드 활성화 플래그인데, 이 모드에서
 * unordered_map은 알려진 호환성 문제가 있으므로 이 조건에서만 unordered_map을 허용한다. */
#if defined(__GNUC__) and not defined(_GLIBCXX_DEBUG)
/* [한국어] 2단계 분기: GCC 버전이 4.3 이상인지 확인.
 * GCC 4.3부터 <tr1/unordered_map>이, GCC 4.3+(C++0x 플래그)부터 <unordered_map>이 가능.
 * __GNUC__는 주 버전(예: 4), __GNUC_MINOR__는 부 버전(예: 3)을 나타낸다. */
#if __GNUC__ >= 4 && __GNUC_MINOR__ >= 3
/* [한국어] GCC >= 4.3 이고 _GLIBCXX_DEBUG 미정의: std::unordered_map 사용 (C++11/TR1 표준) */
#include <unordered_map>                   /* [한국어] <unordered_map> 포함 — std::unordered_map 정의 */
#define tr1_hash_map std::unordered_map    /* [한국어] tr1_hash_map을 std::unordered_map으로 매핑 — O(1) 평균 조회 */
#define tr1_hash_map_ismap 0               /* [한국어] 0: unordered_map이 선택됨을 나타내는 플래그 */
#else
/* [한국어] GCC < 4.3: unordered_map 미지원 — std::map으로 폴백 (O(log n) 조회) */
#include <map>                             /* [한국어] <map> 포함 — std::map 정의 (정렬 트리 기반) */
#define tr1_hash_map std::map              /* [한국어] tr1_hash_map을 std::map으로 매핑 */
#define tr1_hash_map_ismap 1               /* [한국어] 1: map이 선택됨을 나타내는 플래그 */
#endif
#else
/* [한국어] GCC가 아니거나 _GLIBCXX_DEBUG 정의된 경우: std::map으로 폴백.
 * Clang, MSVC 등 비GCC 컴파일러 또는 STL 디버그 모드에서는 안전한 std::map을 사용한다. */
#include <map>                             /* [한국어] <map> 포함 — 폴백용 std::map */
#define tr1_hash_map std::map              /* [한국어] tr1_hash_map을 std::map으로 매핑 */
#define tr1_hash_map_ismap 1               /* [한국어] 1: map 폴백 선택됨 */
#endif

#endif
