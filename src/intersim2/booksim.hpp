// $Id: booksim.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] BookSim NoC 시뮬레이터 전역 헤더 (booksim.hpp)
 *
 * === 파일의 역할 ===
 * BookSim2 NoC(네트워크온칩) 시뮬레이터 전체에서 공통으로 사용하는
 * 표준 라이브러리 헤더 포함과 전역 네임스페이스 설정을 담당하는 최상위 헤더이다.
 * 이 파일은 시뮬레이터의 "공통 기반(common base)" 역할을 하며,
 * 모든 소스 파일이 간접적으로 이 헤더에 의존한다.
 * 플랫폼별(Win32) 컴파일러 경고 억제도 이 파일에서 일괄 처리한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * BookSim2의 모든 헤더 파일 의존 관계의 최상단(root)에 위치한다.
 * 실행 컨텍스트: 시뮬레이터 호스트 유저스페이스에서만 실행된다.
 * 호출 체인: 이 파일 자체는 아무 것도 호출하지 않는 순수 포함 헤더이다.
 * booksim_config.hpp, trafficmanager.hpp, batchtrafficmanager.hpp 등
 * 모든 BookSim 헤더가 이 파일을 먼저 포함한다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존: C++ 표준 라이브러리 (<string>, <cstdlib>, <cstring>, <climits>, <cassert>)
 * - 피의존(이 파일에 의존하는 모듈): intersim2/ 내 모든 소스 파일
 * - GPGPU-Sim과의 연결: icnt_wrapper.cc가 BookSim 진입 시 이 파일을 포함한 헤더 체인을 사용
 * - 데이터 흐름: 타입/매크로/네임스페이스 선언만 제공하며, 런타임 데이터는 없음
 *
 * === 주요 함수/구조체 요약 ===
 * 이 파일에는 함수나 구조체 정의가 없다. 역할은 다음 세 가지이다:
 * 1. 표준 C++ 헤더 일괄 포함 — 시뮬레이터 전역에서 공통 타입을 사용 가능하게 함
 * 2. Win32 빌드 시 C4786 경고 억제 (#pragma warning) 및 <ostream> 보충 포함
 * 3. `using namespace std;` 전역 선언 — 시뮬레이터 소스 전체에서 std:: 접두사 생략
 */

#ifndef _BOOKSIM_HPP_
#define _BOOKSIM_HPP_

#include <string>   // [한국어] std::string — 설정 키/값, 파일 이름 등 문자열 전역 사용
#include <cstdlib>  // [한국어] malloc/free/exit/rand 등 C 표준 유틸리티 함수
#include <cstring>  // [한국어] memset/memcpy/strlen 등 C 문자열/메모리 조작 함수
#include <climits>  // [한국어] INT_MAX, UINT_MAX 등 정수형 한계값 상수 (지연/카운터 초기화에 사용)
#include <cassert>  // [한국어] assert() 매크로 — 시뮬레이터 내부 불변조건(invariant) 검증에 광범위 사용

#ifdef _WIN32_
/* [한국어] Win32(MSVC) 환경에서만 활성화되는 섹션.
 * C4786: "식별자가 디버그 정보 한도(255자)를 초과" 경고를 억제한다.
 * BookSim은 템플릿을 많이 사용하기 때문에 MSVC에서 긴 심볼 이름이 생성된다.
 * <ostream>은 Win32 빌드에서 일부 컴파일러가 자동으로 포함하지 않을 수 있어 명시적으로 추가한다. */
#pragma warning (disable: 4786)
#include <ostream>
#endif

/* [한국어] 표준 네임스페이스를 전역으로 개방한다.
 * BookSim 소스 전체에서 std::string, std::cout, std::vector 등을
 * std:: 접두사 없이 사용할 수 있도록 한다.
 * 이 선언은 헤더 파일에 위치하기 때문에 이 헤더를 포함하는 모든 번역 단위에 전파된다. */
using namespace std;

#endif
