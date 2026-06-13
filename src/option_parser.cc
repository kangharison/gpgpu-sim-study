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
 * [한국어 설명] GPGPU-Sim 설정 파서 구현 (option_parser.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim 시뮬레이터의 설정(configuration) 파싱 시스템을 구현한다.
 * gpgpusim.config 파일, 커맨드라인 인자, 구분자 기반 문자열 세 가지 입력 소스를
 * 통일된 인터페이스로 처리하여, 각 시뮬레이터 모듈이 등록한 C/C++ 변수에 직접
 * 설정 값을 바인딩한다. 타입-세이프한 C++ 템플릿 계층(OptionRegistry<T>)과
 * C 모듈에서 사용할 수 있는 불투명 포인터(option_parser_t) 기반 C 인터페이스를
 * 모두 제공하여 C/C++ 혼합 코드베이스 전체에서 일관된 설정 관리를 가능하게 한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 실행 흐름에서 가장 먼저 실행되는 초기화 단계에 위치한다.
 * 호출 체인:
 *   gpgpusim_entrypoint.cc (시뮬레이터 진입점)
 *     → 각 모듈의 reg_options() 함수 (option_parser_register 호출)
 *     → option_parser_cmdline() / option_parser_cfgfile() (파싱 실행)
 *     → 각 모듈 변수에 값 직접 대입 완료
 *     → 이후 모든 시뮬레이션 사이클 루프가 이 변수들을 읽어 동작 결정
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 단계(싱글 스레드).
 * 파싱이 완료된 이후에는 이 모듈의 코드가 다시 호출되지 않는다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈(이 파일이 사용):
 *   - option_parser.h: C 인터페이스 선언, option_dtype 열거형, option_parser_t 타입
 *   - C++ 표준 라이브러리: fstream(파일 읽기), sstream(토큰화), map/list(레지스트리 저장)
 * 이 파일에 의존하는 모듈(이 파일의 함수를 호출):
 *   - gpgpusim_entrypoint.cc: 시뮬레이터 전체 옵션 등록 및 파싱 진입점
 *   - gpgpu-sim/gpu-sim.cc: GPU 타이밍 모델 옵션 등록 (reg_options)
 *   - gpgpu-sim/shader.cc: SM 파이프라인 옵션 등록
 *   - gpgpu-sim/gpu-cache.cc: 캐시 계층 옵션 등록
 *   - gpgpu-sim/dram.cc: DRAM 타이밍 옵션 등록
 *   - 그 외 모든 시뮬레이터 서브모듈: reg_options() 패턴으로 등록
 * 데이터 흐름:
 *   gpgpusim.config 텍스트 파일
 *     → ParseFile() → ParseStringStream() → ParseCommandLine()
 *     → OptionRegistry<T>::fromString() → 각 모듈의 변수(참조)에 직접 대입
 * 공유 자료구조: option_parser_t (불투명 포인터, 실체는 OptionParser*)
 *
 * === 주요 함수/구조체 요약 ===
 * OptionRegistryInterface  - 모든 타입의 옵션을 다루는 순수 가상 베이스 클래스
 * OptionRegistry<T>        - 타입 T 변수에 대한 참조 바인딩 + 문자열 변환 템플릿
 * OptionParser             - 옵션 컬렉션(list) + 빠른 조회(map)를 통합 관리하는 파서
 * option_parser_register() - C 인터페이스: option_dtype으로 타입 분기 후 Register<T> 호출
 * option_parser_cfgfile()  - C 인터페이스: gpgpusim.config 파일을 파싱하는 진입점
 */

#include "option_parser.h"  /* [한국어] C 인터페이스 선언 (option_parser_t, option_dtype, C 함수 프로토타입) */
#include <assert.h>         /* [한국어] assert() 매크로 — bool 옵션 범위 검증(0 또는 1)에 사용 */
#include <stdio.h>          /* [한국어] fprintf(), printf() — 에러 메시지 및 파싱 결과 출력 */
#include <stdlib.h>         /* [한국어] exit(), calloc(), free() — 에러 시 프로세스 종료 및 임시 argv 배열 관리 */
#include <string.h>         /* [한국어] strcpy() — C 문자열 복사 (char* 옵션 특수화에서 사용) */
#include <fstream>          /* [한국어] ifstream — gpgpusim.config 파일을 줄 단위로 읽기 위해 필요 */
#include <iomanip>          /* [한국어] setw(), left, right, setbase() — Print() 출력 정렬 및 진법 변환 */
#include <iostream>         /* [한국어] cout, cerr — 디버그 출력 및 오류 스트림 */
#include <list>             /* [한국어] list<OptionRegistryInterface*> — 등록 순서를 유지하는 옵션 컬렉션 */
#include <map>              /* [한국어] map<string, OptionRegistryInterface*> — 이름으로 O(1) 조회하는 옵션 맵 */
#include <sstream>          /* [한국어] stringstream — 파일/문자열 내용을 토큰 단위로 파싱하는 버퍼 */
#include <string>           /* [한국어] std::string — 옵션 이름, 설명, 값을 문자열로 관리 */
#include <vector>           /* [한국어] vector<char*> — ParseStringStream에서 임시 argv 배열 구성에 사용 */

using namespace std;  /* [한국어] std:: 접두사 생략 — string, stringstream, map 등 STL 타입을 간결하게 사용 */

/*
 * [한국어]
 * OptionRegistryInterface - 타입에 무관한 옵션 레지스트리 베이스 클래스
 *
 * 모든 타입(int, float, bool, string, char* 등)의 옵션을 동일한 인터페이스로
 * 다루기 위한 순수 가상 클래스다. OptionParser가 이 클래스의 포인터 컬렉션을
 * 유지함으로써 타입을 모르는 상태에서도 fromString/toString/isFlag 등을
 * 다형적으로 호출할 수 있다. 실제 타입별 동작은 OptionRegistry<T>와 그
 * 특수화 버전들이 오버라이드하여 제공한다.
 *
 * 설계 이유:
 *   - 순수 가상 함수(=0)로 선언하여 인터페이스만 정의하고 구현은 강제하지 않음
 *   - m_isParsed 플래그로 "기본값만 설정된 상태" vs "실제 파싱된 상태"를 구분,
 *     Print() 시 파싱되지 않은 필수 옵션을 감지하여 오류 보고 가능
 */
// A generic option registry regardless of data type
class OptionRegistryInterface {
 public:
  /*
   * [한국어]
   * OptionRegistryInterface 생성자
   *
   * @optionName: 옵션 식별자 문자열 (예: "-gpgpu_n_clusters"). 커맨드라인/설정
   *              파일에서 이 문자열로 옵션을 검색한다.
   * @optionDesc: 옵션 설명 문자열. Print() 출력 시 # 뒤에 표시되어 설정 파일
   *              독자에게 각 옵션의 용도를 알려준다.
   *
   * m_isParsed를 false로 초기화하여, 이후 fromString() 또는 assignDefault()가
   * 호출될 때까지 "미파싱" 상태를 유지한다. Print()는 m_isParsed == false인
   * 옵션이 있으면 assert로 프로그램을 중단한다.
   *
   * 호출 체인: OptionRegistry<T> 생성자 → [이 생성자] (위임 생성자 패턴)
   */
  OptionRegistryInterface(const string optionName, const string optionDesc)
      : m_optionName(optionName), m_optionDesc(optionDesc), m_isParsed(false) {}

  /*
   * [한국어]
   * 가상 소멸자 — 파생 클래스(OptionRegistry<T>) 객체를 베이스 포인터로
   * delete할 때 파생 클래스의 소멸자가 올바르게 호출되도록 보장한다.
   * OptionParser::~OptionParser()에서 list의 각 원소를 delete (*i_option)으로
   * 해제할 때 이 가상 소멸자가 호출된다.
   */
  virtual ~OptionRegistryInterface() {}

  /*
   * [한국어]
   * GetName - 옵션 이름 문자열 반환
   *
   * @return: m_optionName의 const 참조. OptionParser::ParseCommandLine()에서
   *          argv[i]와 비교할 때 및 Print()에서 출력할 때 사용된다.
   *
   * 복사 없이 참조를 반환하므로 호출 비용이 없다.
   */
  const string &GetName() { return m_optionName; }

  /*
   * [한국어]
   * GetDesc - 옵션 설명 문자열 반환
   *
   * @return: m_optionDesc의 const 참조. Print()에서 # 뒤에 출력되어
   *          설정 파일 주석으로 활용된다.
   */
  const string &GetDesc() { return m_optionDesc; }

  /*
   * [한국어]
   * isParsed - 이 옵션이 실제로 파싱되었는지 여부 반환
   *
   * @return: fromString() 또는 assignDefault()가 성공적으로 호출되었으면 true.
   *          Print()가 모든 옵션에 대해 이 값을 확인하고, false이면 필수 옵션
   *          누락으로 판단하여 assert(0)으로 종료한다.
   */
  const bool isParsed() { return m_isParsed; }

  /*
   * [한국어]
   * toString - 현재 옵션 값을 문자열로 직렬화 (순수 가상)
   *
   * @return: 현재 m_variable 값의 문자열 표현. Print()에서 현재 설정값을
   *          출력할 때 사용된다.
   *
   * 파생 클래스(OptionRegistry<T>)가 stringstream을 통해 구현하며,
   * char* 특수화는 NULL 방어 코드를 포함한다.
   */
  virtual string toString() = 0;

  /*
   * [한국어]
   * fromString - 문자열을 파싱하여 옵션 변수에 대입 (순수 가상)
   *
   * @str: 파싱할 값 문자열 (예: "1024", "0x400", "true")
   * @return: 파싱 성공 시 true, 실패 시 false.
   *          ParseCommandLine()이 false를 받으면 에러 메시지 출력 후 exit(1).
   *
   * 각 타입별 특수화가 실제 변환을 담당하며, 성공 시 m_isParsed = true로 설정.
   */
  virtual bool fromString(const string str) = 0;

  /*
   * [한국어]
   * isFlag - 이 옵션이 플래그(값 없이 존재만으로 의미를 가지는) 옵션인지 반환
   *          (순수 가상)
   *
   * @return: bool 타입 옵션이면 true, 나머지는 false.
   *
   * ParseCommandLine()이 이 값을 확인하여 bool 옵션의 경우 다음 argv가 숫자이면
   * 소비하고 아니면 소비하지 않는 유연한 처리를 구현한다.
   * (예: "-someflag" 단독 또는 "-someflag 1" 둘 다 허용)
   */
  virtual bool isFlag() = 0;

  /*
   * [한국어]
   * assignDefault - 기본값 문자열을 변수에 대입 (순수 가상)
   *
   * @str: 기본값 문자열 (NULL 허용 — char* 옵션은 NULL 기본값을 가질 수 있음)
   * @return: 대입 성공 시 true.
   *
   * Register() 호출 시 즉시 호출되어 등록 즉시 기본값이 설정된다.
   * char* 특수화는 const_cast를 사용하여 NULL을 그대로 저장하는 반면,
   * 일반 타입은 fromString()에 위임한다.
   */
  virtual bool assignDefault(const char *str) = 0;

 protected:
  string m_optionName;
  /* [한국어] 옵션 식별자 문자열 (예: "-gpgpu_n_clusters").
   * 설정자: 생성자에서 한 번 설정 후 불변.
   * 읽는 자: GetName(), ParseCommandLine()의 m_optionMap 키로 사용.
   * 값 범위: 비어 있으면 안 됨 — 커맨드라인 파싱의 키가 되므로 유일해야 함.
   * 동기화: 초기화 이후 읽기 전용이므로 락 불필요. */

  string m_optionDesc;
  /* [한국어] 옵션 설명 문자열 (예: "number of shader clusters").
   * 설정자: 생성자에서 한 번 설정 후 불변.
   * 읽는 자: GetDesc(), Print()에서 # 뒤 주석으로 출력.
   * 값 범위: 빈 문자열도 허용되나 사용자 가독성을 위해 비우지 않는 것이 관행.
   * 동기화: 초기화 이후 읽기 전용이므로 락 불필요. */

  bool m_isParsed;  // true if the target variable has been updated by
                    // fromString()
  /* [한국어] 이 옵션이 실제로 파싱되었는지를 나타내는 상태 플래그.
   * 설정자: fromString() 또는 assignDefault() 성공 시 true로 설정.
   *         생성자에서 false로 초기화.
   * 읽는 자: OptionParser::Print()가 이 값이 false인 옵션을 발견하면
   *          "Missing option" 에러 메시지 출력 후 assert(0) 호출.
   * 값 범위: false(미파싱/기본값미설정) 또는 true(파싱 또는 기본값 설정 완료).
   * 동기화: 싱글 스레드 초기화 단계에서만 설정되므로 락 불필요. */
};

/*
 * [한국어]
 * OptionRegistry<T> - 타입 T 변수에 대한 참조 바인딩 옵션 레지스트리 템플릿
 *
 * OptionRegistryInterface를 상속하여 특정 C++ 타입 T의 변수에 대한 참조를
 * 보유하고, 문자열 ↔ T 값 변환을 stringstream을 통해 구현한다.
 * 핵심 설계: m_variable이 T&(참조)이므로 fromString() 호출 시 원본 변수가
 * 직접 수정된다. 별도의 복사나 포인터 역참조 없이 파싱 결과가 즉시 반영된다.
 *
 * 특수화 버전들:
 *   - OptionRegistry<string>::fromString  - stringstream 없이 직접 대입
 *   - OptionRegistry<char*>::fromString   - new[]로 동적 할당 후 strcpy
 *   - OptionRegistry<char*>::assignDefault- NULL 허용을 위한 const_cast
 *   - OptionRegistry<char*>::toString     - NULL 포인터 방어
 *   - OptionRegistry<bool>::fromString    - 0/1 정수만 허용
 *   - OptionRegistry<bool>::isFlag        - true 반환 (플래그 옵션임을 선언)
 *
 * 기본 구현(이 템플릿)은 int, float, double, unsigned int, long long,
 * unsigned long long, char 등 표준 산술 타입에 사용된다.
 */
// Template for option registry - class T = specify data type of the option
template <class T>
class OptionRegistry : public OptionRegistryInterface {
 public:
  /*
   * [한국어]
   * OptionRegistry<T> 생성자
   *
   * @name:     옵션 이름 (베이스 클래스 m_optionName에 저장)
   * @desc:     옵션 설명 (베이스 클래스 m_optionDesc에 저장)
   * @variable: 파싱 결과를 저장할 T 타입 변수의 참조.
   *            이 참조를 m_variable에 보관하여, fromString() 호출 시
   *            원본 변수를 직접 수정한다.
   *
   * 호출 체인: OptionParser::Register<T>() → [이 생성자]
   *            → OptionRegistryInterface 생성자 (위임)
   */
  OptionRegistry(const string name, const string desc, T &variable)
      : OptionRegistryInterface(name, desc), m_variable(variable) {}

  /*
   * [한국어]
   * 가상 소멸자 — 베이스 클래스 포인터로 delete 시 올바른 소멸자 호출 보장.
   * m_variable은 참조이므로 소멸 시 아무 것도 해제하지 않는다
   * (참조 대상 변수의 수명은 호출자가 관리).
   */
  virtual ~OptionRegistry() {}

  /*
   * [한국어]
   * toString - 현재 m_variable 값을 문자열로 직렬화
   *
   * @return: m_variable의 현재 값을 stringstream에 삽입하여 얻은 문자열.
   *          Print()에서 현재 설정 값을 출력하는 데 사용된다.
   *
   * stringstream의 operator<<는 T 타입에 대해 적절한 변환을 수행한다.
   * char* 타입은 NULL 처리를 위해 이 함수를 특수화한다.
   *
   * 호출 체인: OptionParser::Print() → [이 함수]
   */
  virtual string toString() {
    stringstream ss;     /* [한국어] 값을 문자열로 변환하기 위한 임시 스트림 버퍼 */
    ss << m_variable;    /* [한국어] T 타입에 맞는 operator<<로 m_variable을 문자열화 */
    return ss.str();     /* [한국어] 변환된 문자열 반환 */
  }

  /*
   * [한국어]
   * fromString (기본 템플릿) - 문자열을 T 타입으로 변환하여 m_variable에 대입
   *
   * @str: 파싱할 값 문자열. 10진수, 8진수(0 접두사), 16진수(0x 접두사) 지원.
   * @return: 파싱 성공 시 true, stringstream 변환 실패 시 false.
   *
   * 진법 자동 감지 로직:
   *   - "0x..." → 앞 2글자 skip 후 setbase(16)으로 16진수 파싱
   *   - "0..."  → 앞 1글자 skip 후 setbase(8)로 8진수 파싱
   *   - 기타    → setbase(10)으로 10진수 파싱
   * 변환 실패(예: 문자열이 숫자 형식이 아님)는 exception catch로 처리하여
   * false를 반환하며, 호출자(ParseCommandLine)가 에러 메시지를 출력한다.
   *
   * 호출 체인: ParseCommandLine() → p_option->fromString() → [이 함수]
   *            → m_variable(원본 변수) 직접 수정
   */
  virtual bool fromString(const string str) {
    stringstream ss(str);   /* [한국어] 입력 문자열을 파싱 스트림으로 래핑 */
    ss.exceptions(stringstream::failbit | stringstream::badbit);
    /* [한국어] failbit(변환 실패) 또는 badbit(스트림 오류) 발생 시 exception을
     *          throw하도록 설정 — 아래 try/catch로 파싱 실패를 감지하기 위함 */
    ss << setbase(10);      /* [한국어] 기본 진법을 10진수로 설정 (이후 조건에서 변경 가능) */
    if (str.size() > 1 && str[0] == '0') {  /* [한국어] 첫 글자가 '0'이면 8진수 또는 16진수 */
      if (str.size() > 2 && str[1] == 'x') {
        /* [한국어] "0x" 접두사 → 16진수: 두 글자(0x) skip 후 setbase(16) */
        ss.ignore(2);       /* [한국어] "0x" 두 글자를 스트림에서 건너뜀 */
        ss << setbase(16);  /* [한국어] 이후 >> 연산자가 16진수로 파싱하도록 설정 */
      } else {
        /* [한국어] "0" 접두사만 있으면 8진수: 한 글자(0) skip 후 setbase(8) */
        ss.ignore(1);       /* [한국어] 선행 "0" 한 글자를 스트림에서 건너뜀 */
        ss << setbase(8);   /* [한국어] 이후 >> 연산자가 8진수로 파싱하도록 설정 */
      }
    }
    try {
      ss >> m_variable;     /* [한국어] 설정된 진법으로 문자열을 T 타입으로 변환하여 원본 변수에 직접 대입 */
    } catch (exception &e) {
      return false;         /* [한국어] 변환 실패(failbit/badbit exception) → 호출자에게 false 반환 */
    }
    m_isParsed = true;      /* [한국어] 파싱 성공 — 이 옵션이 실제 값으로 갱신되었음을 표시 */
    return true;            /* [한국어] 파싱 성공 반환 */
  }

  /*
   * [한국어]
   * isFlag (기본 템플릿) - 이 옵션이 플래그 옵션인지 반환
   *
   * @return: 항상 false. 기본 타입(int, float 등)은 값이 필요한 일반 옵션이다.
   *
   * bool 타입만 true를 반환하도록 특수화되어 있다.
   * ParseCommandLine()은 isFlag()==true인 옵션에 대해 다음 argv 소비 여부를
   * 유연하게 처리한다.
   */
  virtual bool isFlag() { return false; }

  /*
   * [한국어]
   * assignDefault (기본 템플릿) - 기본값 문자열을 변수에 대입
   *
   * @str: 기본값 문자열 (NULL 불가 — char* 특수화에서만 NULL 허용)
   * @return: fromString()의 반환값 그대로 전달.
   *
   * 일반 타입은 fromString()에 완전히 위임한다.
   * char* 특수화는 NULL 포인터를 직접 저장해야 하므로 이 함수를 오버라이드.
   *
   * 호출 체인: OptionParser::Register<T>() → [이 함수] → fromString()
   */
  virtual bool assignDefault(const char *str) { return fromString(str); }

  /*
   * [한국어]
   * T로의 암묵적 변환 연산자
   *
   * @return: m_variable의 현재 값. OptionRegistry<T> 객체를 T 값처럼 직접
   *          사용할 수 있게 해주는 편의 연산자.
   *
   * 주로 UNIT_TEST 블록에서 파싱 결과를 직접 읽을 때 사용되며,
   * 시뮬레이터 본 코드에서는 각 모듈이 원본 변수 참조를 직접 보유하므로
   * 이 연산자는 거의 사용되지 않는다.
   */
  operator T() { return m_variable; }

 private:
  T &m_variable;
  /* [한국어] 파싱 결과를 저장할 원본 변수의 참조(lvalue reference).
   * 설정자: 생성자에서 초기화 후 불변 (참조는 재바인딩 불가).
   * 읽는 자: fromString()이 파싱 결과를 대입, toString()이 현재 값을 읽음.
   * 값 범위: T 타입이 허용하는 범위 전체. 파싱 전에는 Register() 호출자가
   *          초기화한 기본값 또는 assignDefault()가 설정한 기본값을 가짐.
   * 동기화: 싱글 스레드 초기화 단계에서만 수정되므로 락 불필요.
   *         참조 대상 변수의 수명은 호출자(각 시뮬레이터 모듈)가 관리. */
};

/*
 * [한국어]
 * OptionRegistry<string>::fromString 특수화
 * - string 타입 옵션의 문자열 직접 대입
 *
 * @str: 대입할 문자열 값.
 * @return: 항상 true (문자열 대입은 실패하지 않음).
 *
 * 기본 템플릿의 fromString은 stringstream을 통해 >> 연산자로 값을 추출하는데,
 * string에 대해 >> 연산자는 공백으로 토큰을 분리하므로 공백을 포함한 문자열을
 * 올바르게 처리하지 못한다. 이 특수화는 str 전체를 m_variable에 직접 대입하여
 * 공백 포함 문자열도 완전히 저장한다.
 * (실제 공백 포함 문자열은 ParseStringStream의 quote 처리를 통해 전달됨)
 *
 * 호출 체인: ParseCommandLine() → fromString() → [이 특수화] → m_variable 직접 대입
 */
// specialized parser for string-type options
template <>
bool OptionRegistry<string>::fromString(const string str) {
  m_variable = str;       /* [한국어] stringstream 거치지 않고 문자열 전체를 m_variable에 직접 대입 */
  m_isParsed = true;      /* [한국어] 파싱 성공 표시 */
  return true;            /* [한국어] 문자열 대입은 항상 성공 */
}

/*
 * [한국어]
 * OptionRegistry<char*>::fromString 특수화
 * - C 문자열(char*) 타입 옵션의 동적 할당 및 복사
 *
 * @str: 복사할 문자열 값.
 * @return: 항상 true (메모리 할당 실패 시 프로세스 종료).
 *
 * char* 변수는 포인터이므로 str.c_str()을 그대로 저장하면 str 소멸 후
 * 댕글링 포인터가 된다. 따라서 new char[]로 str.size()+1 바이트를 동적 할당하고
 * strcpy로 내용을 복사하여 독립적인 수명을 보장한다.
 * 단, 이 할당 메모리는 명시적으로 해제되지 않으므로 시뮬레이터 수명 동안
 * 유지된다 (gpgpusim_entrypoint 또는 프로세스 종료 시 OS가 회수).
 *
 * 호출 체인: ParseCommandLine() → fromString() → [이 특수화]
 *            → new char[] 동적 할당 → strcpy → m_variable(원본 char* 변수) 업데이트
 */
// specialized parser for c-string type options
template <>
bool OptionRegistry<char *>::fromString(const string str) {
  m_variable = new char[str.size() + 1];  /* [한국어] 문자열 길이+1(null terminator) 바이트 동적 할당 */
  strcpy(m_variable, str.c_str());        /* [한국어] str 내용을 새로 할당한 버퍼에 복사 */
  m_isParsed = true;                      /* [한국어] 파싱 성공 표시 */
  return true;                            /* [한국어] 항상 성공 (new 실패 시 std::bad_alloc으로 프로세스 종료) */
}

/*
 * [한국어]
 * OptionRegistry<char*>::assignDefault 특수화
 * - C 문자열 기본값 설정 (NULL 허용)
 *
 * @str: 기본값 문자열 포인터. NULL 가능 — 옵션에 기본값이 없음을 나타냄.
 * @return: 항상 true.
 *
 * 기본 템플릿의 assignDefault는 fromString()에 위임하는데, fromString은
 * new char[]를 할당하므로 str==NULL이면 크래시가 발생한다.
 * 이 특수화는 const_cast<char*>(str)로 const char* → char*로 변환하여
 * NULL 포함 원본 포인터를 m_variable에 직접 저장한다.
 * (주석 "c-string options are not meant to be edited anyway"가 의미하듯,
 * char* 옵션은 이후 수정되지 않으므로 const_cast는 안전하다)
 * 이후 실제 파싱 값이 들어오면 fromString()이 호출되어 new[]로 대체된다.
 *
 * 호출 체인: OptionParser::Register<char*>() → [이 특수화] → m_variable = str
 */
// specialized default assignment for c-string type option to allow NULL default
template <>
bool OptionRegistry<char *>::assignDefault(const char *str) {
  m_variable = const_cast<char *>(
      str);  // c-string options are not meant to be edited anyway
  /* [한국어] const_cast로 const char* → char*로 변환하여 m_variable에 직접 저장.
   *          str이 NULL이어도 안전하며, 기본값이 없는 char* 옵션을 허용한다. */
  m_isParsed = true;  /* [한국어] 기본값 대입도 "파싱 완료" 상태로 처리 */
  return true;        /* [한국어] 항상 성공 */
}

/*
 * [한국어]
 * OptionRegistry<char*>::toString 특수화
 * - NULL char* 포인터에 대한 방어 출력
 *
 * @return: m_variable이 NULL이 아니면 그 내용 문자열, NULL이면 "NULL" 문자열.
 *
 * 기본 템플릿의 toString은 ss << m_variable을 그냥 호출하는데,
 * char*가 NULL이면 미정의 동작(UB)이 발생한다. 이 특수화는 NULL 여부를
 * 먼저 검사하여 안전하게 "NULL"을 출력한다.
 * NULL 기본값을 가진 옵션이 파싱되지 않은 채 Print()가 호출될 때 크래시를 방지.
 *
 * 호출 체인: OptionParser::Print() → toString() → [이 특수화]
 */
// specialized default assignment for c-string type option to allow NULL default
template <>
string OptionRegistry<char *>::toString() {
  stringstream ss;          /* [한국어] 출력 버퍼 생성 */
  if (m_variable != NULL) { /* [한국어] NULL 포인터 역참조 방지: 유효한 포인터인지 먼저 확인 */
    ss << m_variable;       /* [한국어] 유효한 C 문자열이면 스트림에 삽입 */
  } else {
    ss << "NULL";           /* [한국어] NULL 포인터면 리터럴 "NULL" 문자열로 표현 */
  }
  return ss.str();          /* [한국어] 결과 문자열 반환 */
}

/*
 * [한국어]
 * OptionRegistry<bool>::fromString 특수화
 * - bool 타입 옵션의 0/1 정수 파싱
 *
 * @str: "0" 또는 "1" 문자열. 그 외 값은 assert 실패로 프로세스 종료.
 * @return: 파싱 성공 시 true, 숫자 변환 실패(예: 빈 문자열) 시 false.
 *
 * bool 옵션은 반드시 0(false) 또는 1(true) 값만 허용한다.
 * 설계 근거:
 *   - "true"/"false" 문자열을 허용하지 않고 0/1만 허용하는 것은 GPGPU-Sim
 *     설정 파일 포맷의 관행이다 (gpgpusim.config 내 모든 bool은 0/1로 표기).
 *   - assert(value == 0 or value == 1)로 잘못된 값을 즉시 감지한다.
 *   - isFlag()가 true를 반환하므로 ParseCommandLine에서 다음 토큰이 0 또는 1인
 *     경우에만 소비하고, 없으면 1로 간주하는 동작과 연동된다.
 * 구현:
 *   - value를 int로 파싱하여 0/1 범위 확인 후 m_variable에 대입.
 *   - 빈 문자열(플래그만 등장, 값 없음)은 stringstream::failure exception이
 *     발생하여 parsed=false, value=1(초기값)이 유지되며 m_variable=true가 된다.
 *
 * 호출 체인: ParseCommandLine() → fromString() → [이 특수화] → m_variable 대입
 */
// specialized parser for boolean options
template <>
bool OptionRegistry<bool>::fromString(const string str) {
  int value = 1;       /* [한국어] 기본값 1 — 빈 문자열(플래그 단독 사용) 시 true로 처리되도록 */
  bool parsed = true;  /* [한국어] 파싱 성공 여부 추적 플래그 */
  stringstream ss(str);  /* [한국어] 입력 문자열을 파싱 스트림으로 래핑 */
  ss.exceptions(stringstream::failbit | stringstream::badbit);
  /* [한국어] 변환 실패/스트림 오류 시 exception throw 설정 */
  try {
    ss >> value;  /* [한국어] 문자열을 int로 변환 시도 (빈 문자열이면 failbit → exception) */
  } catch (stringstream::failure &ep) {
    parsed = false;  /* [한국어] 숫자 변환 실패 — 빈 문자열 등으로 int를 읽지 못한 경우 */
  }
  assert(value == 0 or
         value ==
             1);  // sanity check for boolean options (it can only be 1 or 0)
  /* [한국어] 안전성 검사: bool 옵션은 반드시 0 또는 1이어야 함.
   *          2 이상의 정수가 들어오면 즉시 assert 실패로 프로세스 종료. */
  m_variable = (value != 0);  /* [한국어] 0이면 false, 1이면 true로 변환하여 bool 변수에 대입 */
  m_isParsed = true;          /* [한국어] 파싱 완료 표시 (parsed 결과와 무관하게 값은 설정됨) */
  return parsed;              /* [한국어] 숫자 변환 성공 여부 반환 (플래그 단독 사용 시 false) */
}

/*
 * [한국어]
 * OptionRegistry<bool>::isFlag 특수화
 * - bool 옵션이 플래그 옵션임을 선언
 *
 * @return: 항상 true.
 *
 * 플래그 옵션은 값 없이 이름만으로 사용할 수 있다 (예: "-someflag" 단독).
 * ParseCommandLine은 isFlag()==true인 옵션의 경우 다음 argv[i+1]로 fromString을
 * 먼저 시도하고, 실패하면(예: 다음 토큰이 다른 옵션 이름) i를 증가시키지 않는다.
 * 이를 통해 "-someflag", "-someflag 1", "-someflag 0" 세 가지 형식 모두 지원.
 *
 * 호출 체인: ParseCommandLine() → p_option->isFlag() → [이 특수화]
 */
// specializing a flag query function to identify boolean option
template <>
bool OptionRegistry<bool>::isFlag() {
  return true;  /* [한국어] bool 옵션은 플래그 옵션 — 값 없이 이름만으로도 true로 설정 가능 */
}

/*
 * [한국어]
 * OptionParser - 옵션 컬렉션을 관리하고 다양한 소스에서 파싱하는 파서 클래스
 *
 * 두 가지 자료구조를 동시에 유지한다:
 *   1. m_optionReg (list): 등록 순서를 보존 → Print() 시 등록 순서대로 출력
 *   2. m_optionMap (map): 이름으로 O(log n) 조회 → ParseCommandLine의 빠른 탐색
 * list와 map이 같은 OptionRegistryInterface* 포인터를 공유하므로 메모리 낭비 없음.
 *
 * 파싱 흐름:
 *   ParseFile(파일명) → ParseStringStream(스트림)
 *   ParseString(문자열, 구분자) → ParseStringStream(스트림)
 *   ParseCommandLine(argc, argv) ← ParseStringStream이 최종적으로 호출
 *
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 초기화 단계(싱글 스레드).
 * 파싱 완료 후에는 OptionParser 객체가 소멸되거나 유지되어도 시뮬레이션에
 * 영향을 주지 않는다 (각 모듈이 원본 변수를 직접 보유하기 때문).
 */
// class holding a collection of options and parse them from command
// line/configfile
class OptionParser {
 public:
  /*
   * [한국어]
   * OptionParser 기본 생성자 — 빈 컬렉션으로 초기화.
   * m_optionReg(list)와 m_optionMap(map)은 기본 생성자로 빈 상태 초기화됨.
   */
  OptionParser() {}

  /*
   * [한국어]
   * OptionParser 소멸자 — m_optionReg의 모든 OptionRegistryInterface* 포인터 해제
   *
   * m_optionReg의 각 원소가 new OptionRegistry<T>로 생성된 힙 객체이므로
   * 소멸 시 delete로 명시적으로 해제해야 한다. m_optionMap은 같은 포인터를
   * 공유하므로 별도 해제하지 않는다 (이미 list에서 해제됨).
   *
   * 호출 체인: option_parser_destroy() → delete OptionParser → [이 소멸자]
   */
  ~OptionParser() {
    OptionCollection::iterator i_option;  /* [한국어] list 순회를 위한 반복자 */
    for (i_option = m_optionReg.begin(); i_option != m_optionReg.end();
         ++i_option) {
      delete (*i_option);  /* [한국어] 각 OptionRegistry<T> 객체를 힙에서 해제 (가상 소멸자 경유) */
    }
  }

  /*
   * [한국어]
   * Register<T> - 타입 T 옵션을 파서에 등록
   *
   * @optionName:     옵션 식별자 (예: "-gpgpu_n_clusters")
   * @optionDesc:     옵션 설명 문자열
   * @optionVariable: 파싱 결과를 받을 T 타입 변수 참조 (원본 변수 직접 바인딩)
   * @optionDefault:  기본값 문자열 (NULL 허용 — char* 옵션의 경우)
   *
   * 동작:
   *   1. OptionRegistry<T> 객체를 힙에 생성하고 변수 참조를 바인딩
   *   2. m_optionReg(list)에 추가하여 등록 순서 유지
   *   3. m_optionMap에 이름 → 포인터 매핑 추가하여 O(log n) 조회 지원
   *   4. assignDefault()를 즉시 호출하여 기본값 설정 (m_isParsed = true로 초기화)
   *
   * 호출 체인: option_parser_register() (C 인터페이스)
   *         또는 각 모듈의 reg_options() → [이 함수]
   *         → OptionRegistry<T> 생성 → assignDefault()
   */
  template <class T>
  void Register(const string optionName, const string optionDesc,
                T &optionVariable, const char *optionDefault) {
    OptionRegistry<T> *p_option =
        new OptionRegistry<T>(optionName, optionDesc, optionVariable);
    /* [한국어] 힙에 OptionRegistry<T> 생성 — 이 포인터의 소유권은 m_optionReg가 가짐 */
    m_optionReg.push_back(p_option);   /* [한국어] 등록 순서 보존을 위해 list 뒤에 추가 */
    m_optionMap[optionName] = p_option; /* [한국어] 빠른 이름 조회를 위해 map에도 등록 */
    p_option->assignDefault(optionDefault);
    /* [한국어] 기본값 즉시 적용 — 파일/커맨드라인 파싱 전에도 유효한 값이 보장됨 */
  }

  /*
   * [한국어]
   * ParseCommandLine - argc/argv 배열을 파싱하여 각 옵션 변수에 값 대입
   *
   * @argc: argv 배열의 원소 수 (argv[0]는 프로그램 이름 또는 dummy)
   * @argv: 파싱할 인자 문자열 배열. argv[0]은 건너뛰고 argv[1]부터 처리.
   *
   * 처리 흐름:
   *   - argv[i]가 m_optionMap에 있으면 → argv[i+1]을 값으로 fromString() 호출
   *   - argv[i]가 "-config"이면 → argv[i+1]을 파일명으로 ParseFile() 재귀 호출
   *   - argv[i]가 map에도 없고 "-config"도 아니면 → 에러 출력 후 exit(1)
   * bool(flag) 옵션은 다음 토큰 소비를 유연하게 처리:
   *   - fromString 성공(다음 토큰이 0 또는 1)이면 i += 1로 소비
   *   - fromString 실패(다음 토큰이 다른 옵션 이름 등)이면 i 유지
   *
   * 실행 컨텍스트: 싱글 스레드 초기화 단계. ParseStringStream에서 재귀 호출됨.
   *
   * 호출 체인: option_parser_cmdline() 또는 ParseStringStream()
   *            → [이 함수] → m_optionMap 조회 → fromString()
   *                        → ParseFile() (−config 처리 시)
   */
  void ParseCommandLine(int argc, const char *const argv[]) {
    for (int i = 1; i < argc; i++) {  /* [한국어] argv[0](프로그램 이름/dummy)은 건너뛰고 argv[1]부터 순환 */
      OptionMap::iterator i_option;   /* [한국어] map 탐색 결과를 받을 반복자 */
      bool optionFound = false;       /* [한국어] 현재 argv[i]가 유효한 옵션으로 처리되었는지 추적 */

      i_option = m_optionMap.find(argv[i]);  /* [한국어] 옵션 이름으로 O(log n) map 탐색 */
      if (i_option != m_optionMap.end()) {   /* [한국어] map에 등록된 옵션이면 처리 */
        const char *argstr = (i + 1 < argc) ? argv[i + 1] : "";
        /* [한국어] 다음 토큰(옵션 값)을 읽음. argv 범위 초과 방지를 위해 빈 문자열로 fallback */
        OptionRegistryInterface *p_option = i_option->second;  /* [한국어] 매핑된 OptionRegistry 포인터 추출 */
        if (p_option->isFlag()) {
          /* [한국어] bool(flag) 옵션: 다음 토큰을 값으로 시도 — 실패하면 토큰 소비 안 함 */
          if (p_option->fromString(argstr) == true) {
            i += 1;  /* [한국어] 다음 토큰(0 또는 1)을 값으로 소비했으므로 인덱스 전진 */
          }
          /* [한국어] fromString 실패(빈 문자열 또는 다른 옵션 이름)이면 i 유지 — 플래그만 있어도 true 설정됨 */
        } else {
          /* [한국어] 일반 옵션: 다음 토큰은 반드시 값이어야 함 */
          if (p_option->fromString(argstr) == false) {
            fprintf(stderr,
                    "\n\nGPGPU-Sim ** ERROR: Cannot parse value '%s' for "
                    "option '%s'.\n",
                    argstr, argv[i]);
            /* [한국어] 값 파싱 실패 — 에러 메시지를 stderr에 출력 */
            exit(1);  /* [한국어] 복구 불가 에러: 시뮬레이터 즉시 종료 */
          }
          i += 1;  /* [한국어] 값 토큰(argv[i+1])을 소비했으므로 인덱스를 하나 더 전진 */
        }
        optionFound = true;  /* [한국어] 유효한 옵션 처리 완료 */
      } else if (string(argv[i]) == "-config") {
        /* [한국어] "-config" 특수 옵션: 다음 토큰을 설정 파일 경로로 처리 */
        if (i + 1 >= argc) {
          fprintf(stderr,
                  "\n\nGPGPU-Sim ** ERROR: Missing filename for option "
                  "'-config'.\n");
          /* [한국어] "-config" 뒤에 파일명이 없으면 에러 출력 */
          exit(1);  /* [한국어] 복구 불가 에러: 즉시 종료 */
        }

        ParseFile(argv[i + 1]);  /* [한국어] 지정된 config 파일을 파싱하여 추가 옵션 로드 */
        i += 1;                  /* [한국어] 파일명 토큰(argv[i+1])을 소비했으므로 인덱스 전진 */
        optionFound = true;      /* [한국어] "-config" 옵션 처리 완료 */
      }
      if (optionFound == false) {
        /* [한국어] map에도 없고 "-config"도 아닌 알 수 없는 옵션 — 에러 종료 */
        fprintf(stderr, "\n\nGPGPU-Sim ** ERROR: Unknown Option: '%s' \n",
                argv[i]);
        exit(1);  /* [한국어] 알 수 없는 옵션은 설정 오류이므로 즉시 종료 */
      }
    }
  }

  /*
   * [한국어]
   * ParseFile - gpgpusim.config 파일을 읽어 파싱
   *
   * @filename: 파싱할 설정 파일 경로 (예: "gpgpusim.config")
   *
   * 동작:
   *   1. 파일을 줄 단위로 읽으면서 '#' 이후 주석을 제거
   *   2. 모든 줄을 하나의 stringstream에 이어 붙임
   *   3. ParseStringStream()에 전달하여 토큰 단위 파싱
   *
   * gpgpusim.config 파일 포맷:
   *   -옵션이름 값   # 주석
   *   -gpgpu_n_clusters 28  # number of SM clusters
   *
   * stringstream 버퍼링 이유:
   *   줄마다 ParseCommandLine을 호출하면 줄 경계에서 옵션-값 쌍이 끊길 수 있다.
   *   모든 줄을 하나의 스트림으로 합치면 ParseStringStream이 연속된 토큰 스트림으로
   *   처리하므로 줄 경계에 무관하게 안전하게 파싱된다.
   *
   * 에러 처리: 파일 열기 실패 시 에러 메시지 출력 후 exit(1).
   *
   * 호출 체인: option_parser_cfgfile() 또는 ParseCommandLine()의 -config 처리
   *            → [이 함수] → ParseStringStream() → ParseCommandLine()
   */
  void ParseFile(const char *filename) {
    ifstream inputFile;    /* [한국어] 설정 파일을 읽기 위한 입력 파일 스트림 */
    stringstream args;     /* [한국어] 주석 제거된 모든 줄을 이어 붙여 저장하는 버퍼 스트림 */

    // open config file, stream every line into a continuous buffer
    // get rid of comments in the process
    inputFile.open(filename);  /* [한국어] 설정 파일 열기 시도 */
    if (!inputFile.good()) {
      /* [한국어] 파일 열기 실패 (경로 오류, 권한 없음 등) — 에러 출력 후 종료 */
      fprintf(stderr, "\n\nGPGPU-Sim ** ERROR: Cannot open config file '%s'\n",
              filename);
      exit(1);  /* [한국어] 설정 파일 없이는 시뮬레이터 실행 불가: 즉시 종료 */
    }
    while (inputFile.good()) {  /* [한국어] EOF에 도달하거나 스트림 오류가 발생할 때까지 줄 단위 반복 */
      string line;               /* [한국어] 현재 줄 내용을 저장하는 임시 문자열 */
      getline(inputFile, line);  /* [한국어] 한 줄 전체(개행 제외)를 line에 읽기 */
      size_t commentStart = line.find_first_of("#");
      /* [한국어] '#' 문자의 위치를 탐색 — gpgpusim.config의 주석 시작 기호 */
      if (commentStart != line.npos) {
        line.erase(commentStart);  /* [한국어] '#'부터 줄 끝까지 삭제하여 주석 제거 */
      }
      args << line << ' ';
      /* [한국어] 주석 제거된 줄을 버퍼 스트림에 추가. 줄 뒤에 공백을 붙여
       *          다음 줄의 첫 토큰과 이전 줄 마지막 토큰이 붙지 않도록 분리 */
    }
    inputFile.close();  /* [한국어] 파일 스트림 닫기 — 파일 디스크립터 해제 */

    ParseStringStream(args);  /* [한국어] 주석 제거된 전체 내용을 토큰 단위로 파싱 */
  }

  /*
   * [한국어]
   * ParseString - 구분자 기반 문자열을 파싱
   *
   * @inputString: 파싱할 옵션 문자열 (예: "ABC 1111; DEF 88; Mode A; Name out")
   * @delimiters:  토큰 구분자 문자 집합 (기본값: " ;" — 공백과 세미콜론)
   *
   * 동작:
   *   1. inputString의 모든 구분자 문자를 공백으로 치환
   *   2. stringstream으로 래핑하여 ParseStringStream() 호출
   *
   * 구분자를 공백으로 변환하는 이유:
   *   ParseStringStream은 공백으로 분리된 토큰을 처리한다. 세미콜론 등 다른
   *   구분자를 공백으로 통일하면 ParseStringStream을 수정 없이 재사용할 수 있다.
   *   "Name=dram;DEF=702" 형식에서 " =;" 구분자를 지정하면 공백 문자열로 변환된
   *   후 "Name dram DEF 702"처럼 처리된다.
   *
   * 호출 체인: option_parser_delimited_string() → [이 함수] → ParseStringStream()
   *                                                          → ParseCommandLine()
   */
  // parse the given string as tokens separated by a set of given delimiters
  void ParseString(string inputString, const string delimiters = string(" ;")) {
    // convert all delimiter characters into whitespaces
    for (unsigned t = 0; t < inputString.size(); t++) {
      /* [한국어] inputString의 각 문자를 순회하며 구분자 여부 확인 */
      for (unsigned d = 0; d < delimiters.size(); d++) {
        /* [한국어] 현재 문자가 구분자 집합의 어느 하나와 일치하는지 비교 */
        if (inputString[t] == delimiters.at(d)) {
          inputString[t] = ' ';  /* [한국어] 구분자 문자를 공백으로 치환 — ParseStringStream 재사용 가능 */
          break;                 /* [한국어] 하나의 구분자와 일치하면 나머지 구분자 비교 불필요 */
        }
      }
    }
    stringstream args(inputString);  /* [한국어] 구분자가 공백으로 통일된 문자열을 스트림으로 래핑 */
    ParseStringStream(args);         /* [한국어] 공백 분리된 토큰 파싱 실행 */
  }

  /*
   * [한국어]
   * ParseStringStream - 공백 구분된 stringstream을 argv 배열로 변환 후 파싱
   *
   * @args: 파싱할 공백 구분 토큰 스트림 (ParseFile 또는 ParseString이 구성)
   *
   * 동작:
   *   1. dummy argv[0] 추가 ("dummy" 문자열 — ParseCommandLine이 argv[0]을 건너뜀)
   *   2. 스트림에서 토큰을 하나씩 읽어 argv 배열에 추가
   *   3. '"'로 시작하는 토큰은 '"'로 끝나는 토큰까지 이어 붙여 하나의 토큰으로 처리
   *      (공백 포함 문자열 지원: "-sdata \"hello world\"")
   *   4. vector<char*>를 char**로 변환하여 ParseCommandLine() 호출
   *   5. 임시 메모리(new char[], calloc) 모두 해제
   *
   * dummy argv[0] 이유:
   *   ParseCommandLine은 argv[0]이 프로그램 이름이라고 가정하고 i=1부터 처리한다.
   *   ParseStringStream은 파일/문자열 내용만 처리하므로 argv[0] 자리에 "dummy"를
   *   넣어 실제 옵션 토큰들이 argv[1]부터 시작하도록 맞춘다.
   *
   * quote 처리 이유:
   *   공백은 토큰 구분자이므로 공백을 포함한 값(파일 경로, 문자열 옵션 등)은
   *   따옴표로 묶어야 한다. 예: -filename "path with spaces/config.txt"
   *
   * 호출 체인: ParseFile() 또는 ParseString() → [이 함수] → ParseCommandLine()
   */
  // parse the given stringstream as whitespace-separated tokens. drain the
  // stream in the process
  void ParseStringStream(stringstream &args) {
    // extract non-whitespace string tokens
    vector<char *> argv;             /* [한국어] ParseCommandLine에 전달할 임시 argv 배열 (동적 확장) */
    argv.push_back(new char[6]);     /* [한국어] argv[0] 슬롯을 위해 6바이트("dummy"+null) 할당 */
    strcpy(argv[0], "dummy");        /* [한국어] argv[0]에 "dummy" 저장 — ParseCommandLine이 [0]을 건너뜀 */
    while (args.good()) {
      /* [한국어] 스트림이 유효한 동안 토큰을 하나씩 추출 */
      string argNew;
      args >> argNew;  /* [한국어] 공백으로 구분된 다음 토큰 추출 */

      if (argNew.size() == 0) continue;  // this is probably the last token
      /* [한국어] 빈 토큰(스트림 끝 근처에서 발생 가능) 건너뜀 */

      if (argNew[0] == '"') {
        /* [한국어] '"'로 시작하는 토큰 — 공백 포함 문자열의 시작. '"'로 끝날 때까지 이어 붙임 */
        while (args.good() && argNew[argNew.size() - 1] != '"') {
          /* [한국어] 마지막 문자가 '"'가 될 때까지 다음 토큰들을 계속 이어 붙임 */
          string argCont;
          args >> argCont;          /* [한국어] 다음 토큰(공백 뒤 부분) 추출 */
          argNew += " " + argCont;  /* [한국어] 공백 복원하며 이어 붙임 — 원래 문자열 재구성 */
        }
        argNew.erase(0, 1);               /* [한국어] 시작 '"' 제거 */
        argNew.erase(argNew.size() - 1);  /* [한국어] 끝 '"' 제거 — 따옴표 없는 순수 문자열만 남김 */
      }

      char *c_argNew = new char[argNew.size() + 1];  /* [한국어] C 문자열을 위한 버퍼 동적 할당 */
      strcpy(c_argNew, argNew.c_str());               /* [한국어] std::string에서 C 문자열로 복사 */
      argv.push_back(c_argNew);                       /* [한국어] 완성된 토큰을 argv 배열에 추가 */
    }

    // pass the string token into normal commandline parser
    char **targv = (char **)calloc(argv.size(), sizeof(char *));
    /* [한국어] vector<char*>를 ParseCommandLine에 전달할 char** 배열로 변환.
     *          calloc으로 0 초기화된 배열 할당 (argv.size() * sizeof(char*) 바이트) */
    for (unsigned k = 0; k < argv.size(); k++) targv[k] = argv[k];
    /* [한국어] vector의 각 char* 포인터를 targv 배열에 복사 */
    ParseCommandLine(argv.size(), targv);
    /* [한국어] 구성된 argc/argv로 실제 옵션 파싱 실행 */
    free(targv);
    /* [한국어] calloc으로 할당한 포인터 배열 해제 (각 원소가 가리키는 문자열은 아직 살아있음) */
    for (size_t i = 0; i < argv.size(); i++) {
      delete[] argv[i];  /* [한국어] new char[]로 할당한 각 토큰 문자열 버퍼 해제 */
    }
  }

  /*
   * [한국어]
   * Print - 현재 등록된 모든 옵션의 이름, 값, 설명을 정렬하여 출력
   *
   * @fout: 출력 파일 스트림 (보통 stdout 또는 시뮬레이터 로그 파일)
   *
   * 동작:
   *   - m_optionReg(list)를 순서대로 순회하여 등록 순서대로 출력
   *   - 파싱되지 않은 옵션(m_isParsed == false) 발견 시 에러 메시지 + assert(0)
   *   - 출력 포맷: setw(20) 이름 + setw(20) 값 + " # " + 설명
   *
   * 출력 예:
   *   -gpgpu_n_clusters             28 # number of SM clusters
   *
   * 호출 체인: option_parser_print() → [이 함수]
   *           또는 시뮬레이터 초기화 완료 후 설정 요약 출력 시 직접 호출
   */
  void Print(FILE *fout) {
    OptionCollection::iterator i_option;  /* [한국어] list 순회를 위한 반복자 */
    for (i_option = m_optionReg.begin(); i_option != m_optionReg.end();
         ++i_option) {
      stringstream sout;                     /* [한국어] 한 줄 출력을 구성하는 임시 스트림 */
      if ((*i_option)->isParsed() == false) {
        /* [한국어] 기본값도 설정되지 않은 필수 옵션 감지 — 심각한 설정 오류 */
        cerr << "\n\nGPGPU-Sim ** ERROR: Missing option '"
             << (*i_option)->GetName() << "'\n";
        assert(0);  /* [한국어] 복구 불가 상태: 누락 옵션이 있으면 즉시 프로세스 중단 */
      }
      sout << setw(20) << left << (*i_option)->GetName() << " ";
      /* [한국어] 옵션 이름을 왼쪽 정렬, 20자 폭으로 출력 (열 정렬을 위한 padding) */
      sout << setw(20) << right << (*i_option)->toString() << " # ";
      /* [한국어] 옵션 값을 오른쪽 정렬, 20자 폭으로 출력 후 주석 구분자 " # " 추가 */
      sout << left << (*i_option)->GetDesc();
      /* [한국어] 옵션 설명을 왼쪽 정렬로 출력 */
      sout << std::endl;           /* [한국어] 줄 끝 개행 추가 */
      fprintf(fout, "%s", sout.str().c_str());
      /* [한국어] stringstream에 구성된 한 줄을 FILE* 스트림에 출력 */
    }
  }

 private:
  typedef list<OptionRegistryInterface *> OptionCollection;
  /* [한국어] 등록 순서를 보존하는 OptionRegistryInterface* 포인터 list 타입 별칭.
   *          list를 사용하는 이유: vector와 달리 중간 삽입이 O(1)이며, 포인터 안정성 보장.
   *          Print()가 이 list를 순서대로 순회하여 등록 순서대로 옵션을 출력한다. */
  OptionCollection m_optionReg;
  /* [한국어] 모든 등록된 옵션 포인터의 순서 보존 컬렉션.
   * 설정자: Register<T>()에서 push_back으로 추가.
   * 읽는 자: Print()와 소멸자가 이 list를 순회.
   * 값 범위: 각 원소는 힙에 생성된 OptionRegistry<T> 객체의 포인터.
   * 동기화: 싱글 스레드 초기화 단계에서만 수정되므로 락 불필요.
   * 소유권: 이 list가 각 포인터의 소유자 — 소멸자에서 delete 수행. */

  typedef map<string, OptionRegistryInterface *> OptionMap;
  /* [한국어] 옵션 이름 → 포인터 O(log n) 조회를 위한 map 타입 별칭.
   *          ParseCommandLine이 argv[i]로 빠르게 옵션을 찾는 데 사용.
   *          list와 같은 포인터를 공유하므로 별도 메모리 오버헤드 없음. */
  OptionMap m_optionMap;
  /* [한국어] 옵션 이름(string) → OptionRegistryInterface* 포인터 맵.
   * 설정자: Register<T>()에서 m_optionMap[optionName] = p_option으로 추가.
   * 읽는 자: ParseCommandLine()의 find() 호출 — argv[i]로 옵션 검색.
   * 값 범위: m_optionReg의 원소들과 동일한 포인터를 가리킴.
   * 동기화: 싱글 스레드 초기화 단계에서만 수정되므로 락 불필요.
   * 소유권: 포인터를 공유만 하며, 실제 해제는 m_optionReg(list)의 소멸자가 담당. */
};

#include "option_parser.h"  /* [한국어] C 인터페이스 선언 재포함 — C 함수들이 이 선언에 의존 */

/*
 * [한국어]
 * option_parser_create - OptionParser 객체를 힙에 생성하고 불투명 핸들 반환
 *
 * @return: 새로 생성된 OptionParser 객체를 option_parser_t(void*)로 변환한 핸들.
 *          C 모듈은 이 핸들을 모든 option_parser_* 함수에 전달해야 한다.
 *
 * C 인터페이스 설계 이유:
 *   C++ OptionParser 클래스를 C 코드에서 직접 사용할 수 없으므로,
 *   불투명 포인터(void* typedef인 option_parser_t)로 C++ 객체를 숨기고
 *   C 함수 인터페이스만 노출한다. reinterpret_cast로 void* ↔ OptionParser*
 *   변환을 수행하며, 이는 포인터 값이 변하지 않으므로 안전하다.
 *
 * 호출 체인: gpgpusim_entrypoint.cc 또는 각 모듈의 초기화 함수
 *            → [이 함수] → new OptionParser() → reinterpret_cast → 핸들 반환
 */
option_parser_t option_parser_create() {
  OptionParser *p_opr = new OptionParser();  /* [한국어] OptionParser 객체를 힙에 동적 생성 */
  return reinterpret_cast<option_parser_t>(p_opr);
  /* [한국어] OptionParser* → option_parser_t(void*)로 변환하여 반환.
   *          reinterpret_cast: C++에서 포인터 타입 간 비트 패턴 보존 변환 */
}

/*
 * [한국어]
 * option_parser_destroy - option_parser_t 핸들이 가리키는 OptionParser 객체 해제
 *
 * @opp: option_parser_create()가 반환한 핸들.
 *
 * reinterpret_cast로 void* → OptionParser*로 복원 후 delete 호출.
 * delete는 OptionParser::~OptionParser()를 호출하여 m_optionReg의 모든
 * OptionRegistry<T> 객체를 해제한다.
 *
 * 호출 체인: 시뮬레이터 종료 또는 C 인터페이스 테스트 코드 → [이 함수]
 *            → delete OptionParser → ~OptionParser() → 각 OptionRegistry<T> 해제
 */
void option_parser_destroy(option_parser_t opp) {
  OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
  /* [한국어] 불투명 핸들(void*) → OptionParser*로 복원 */
  delete p_opr;  /* [한국어] OptionParser 소멸자 호출 → m_optionReg 내 모든 옵션 객체 해제 */
}

/*
 * [한국어]
 * option_parser_register - C 인터페이스로 옵션 등록
 *
 * @opp:          option_parser_create()가 반환한 파서 핸들
 * @name:         옵션 이름 문자열 (예: "-gpgpu_n_clusters")
 * @type:         옵션 값의 데이터 타입 (option_dtype 열거형)
 * @variable:     파싱 결과를 저장할 변수의 주소 (void*로 전달)
 * @desc:         옵션 설명 문자열
 * @defaultvalue: 기본값 문자열 (NULL 허용 — OPT_CSTR 타입에서만 의미 있음)
 *
 * C에서는 템플릿을 사용할 수 없으므로, option_dtype 열거형으로 타입을 구분하고
 * switch로 각 타입에 맞는 Register<T> 호출로 분기한다.
 * void* variable을 각 타입에 맞게 캐스팅하여 역참조한다:
 *   OPT_INT32  → *(int*)variable
 *   OPT_CSTR   → *(char**)variable (포인터의 포인터 — 파싱 시 새 버퍼로 갱신됨)
 *
 * 호출 체인: 각 시뮬레이터 모듈의 reg_options() → [이 함수]
 *            → switch 분기 → p_opr->Register<T>()
 */
void option_parser_register(option_parser_t opp, const char *name,
                            enum option_dtype type, void *variable,
                            const char *desc, const char *defaultvalue) {
  OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
  /* [한국어] 불투명 핸들 → OptionParser* 복원 */
  switch (type) {
    case OPT_INT32:
      p_opr->Register<int>(name, desc, *(int *)variable, defaultvalue);
      /* [한국어] void* → int*로 캐스팅 후 역참조하여 int& 변수 참조 바인딩 */
      break;
    case OPT_UINT32:
      p_opr->Register<unsigned int>(name, desc, *(unsigned int *)variable,
                                    defaultvalue);
      /* [한국어] void* → unsigned int*로 캐스팅 후 역참조 */
      break;
    case OPT_INT64:
      p_opr->Register<long long>(name, desc, *(long long *)variable,
                                 defaultvalue);
      /* [한국어] void* → long long*로 캐스팅 후 역참조 (64비트 부호 있는 정수) */
      break;
    case OPT_UINT64:
      p_opr->Register<unsigned long long>(
          name, desc, *(unsigned long long *)variable, defaultvalue);
      /* [한국어] void* → unsigned long long*로 캐스팅 후 역참조 (64비트 부호 없는 정수) */
      break;
    case OPT_BOOL:
      p_opr->Register<bool>(name, desc, *(bool *)variable, defaultvalue);
      /* [한국어] void* → bool*로 캐스팅 후 역참조 — isFlag()==true인 플래그 옵션으로 등록 */
      break;
    case OPT_FLOAT:
      p_opr->Register<float>(name, desc, *(float *)variable, defaultvalue);
      /* [한국어] void* → float*로 캐스팅 후 역참조 (단정밀도 부동소수점) */
      break;
    case OPT_DOUBLE:
      p_opr->Register<double>(name, desc, *(double *)variable, defaultvalue);
      /* [한국어] void* → double*로 캐스팅 후 역참조 (배정밀도 부동소수점) */
      break;
    case OPT_CHAR:
      p_opr->Register<char>(name, desc, *(char *)variable, defaultvalue);
      /* [한국어] void* → char*로 캐스팅 후 역참조 (단일 문자) */
      break;
    case OPT_CSTR:
      p_opr->Register<char *>(name, desc, *(char **)variable, defaultvalue);
      /* [한국어] void* → char**로 캐스팅 후 역참조하여 char*& 바인딩.
       *          파싱 시 fromString 특수화가 new char[]로 새 버퍼를 할당하고
       *          원본 char* 포인터를 갱신한다. */
      break;
    default:
      fprintf(stderr,
              "\n\nGPGPU-Sim ** ERROR: option data type (%d) not supported!\n",
              type);
      /* [한국어] 지원하지 않는 타입 코드 — 에러 메시지 출력 */
      exit(1);  /* [한국어] 복구 불가 오류: 즉시 종료 */
      break;
  }
}

/*
 * [한국어]
 * option_parser_cmdline - C 인터페이스: 커맨드라인 인자 파싱
 *
 * @opp:  파서 핸들
 * @argc: 커맨드라인 인자 수
 * @argv: 커맨드라인 인자 문자열 배열
 *
 * OptionParser::ParseCommandLine()의 C 래퍼.
 * gpgpusim_entrypoint.cc에서 CUDA 런타임이 전달한 argc/argv를 처리할 때 사용.
 *
 * 호출 체인: gpgpusim_entrypoint.cc → [이 함수] → p_opr->ParseCommandLine()
 */
void option_parser_cmdline(option_parser_t opp, int argc, const char *argv[]) {
  OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
  /* [한국어] 불투명 핸들 → OptionParser* 복원 */
  return p_opr->ParseCommandLine(argc, argv);
  /* [한국어] C++ ParseCommandLine에 위임 — return void이므로 반환값 없음 */
}

/*
 * [한국어]
 * option_parser_cfgfile - C 인터페이스: 설정 파일 파싱
 *
 * @opp:      파서 핸들
 * @filename: 파싱할 설정 파일 경로 (예: "gpgpusim.config")
 *
 * OptionParser::ParseFile()의 C 래퍼.
 * gpgpusim_entrypoint.cc에서 gpgpusim.config 파일을 로드할 때 사용.
 *
 * 호출 체인: gpgpusim_entrypoint.cc → [이 함수] → p_opr->ParseFile()
 *                                               → ParseStringStream()
 *                                               → ParseCommandLine()
 */
void option_parser_cfgfile(option_parser_t opp, const char *filename) {
  OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
  /* [한국어] 불투명 핸들 → OptionParser* 복원 */
  p_opr->ParseFile(filename);  /* [한국어] 설정 파일 파싱 실행 */
}

/*
 * [한국어]
 * option_parser_delimited_string - C 인터페이스: 구분자 기반 문자열 파싱
 *
 * @opp:         파서 핸들
 * @inputstring: 파싱할 옵션 문자열 (예: "ABC 1111; DEF 88")
 * @delimiters:  토큰 구분자 문자 집합 (예: " ;" 또는 " =;")
 *
 * OptionParser::ParseString()의 C 래퍼.
 * AccelSim 트레이스 드라이버 등에서 설정 문자열을 직접 전달할 때 사용.
 *
 * 호출 체인: 트레이스 드라이버 또는 테스트 코드 → [이 함수] → p_opr->ParseString()
 */
void option_parser_delimited_string(option_parser_t opp,
                                    const char *inputstring,
                                    const char *delimiters) {
  OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
  /* [한국어] 불투명 핸들 → OptionParser* 복원 */
  p_opr->ParseString(inputstring, delimiters);
  /* [한국어] 구분자 기반 문자열 파싱 실행 — 구분자를 공백으로 치환 후 토큰 파싱 */
}

/*
 * [한국어]
 * option_parser_print - C 인터페이스: 현재 옵션 설정 출력
 *
 * @opp:  파서 핸들
 * @fout: 출력 파일 스트림 (stdout 또는 로그 파일)
 *
 * OptionParser::Print()의 C 래퍼.
 * 시뮬레이터 초기화 완료 후 최종 설정값을 로그에 기록할 때 사용.
 * 파싱되지 않은 필수 옵션이 있으면 assert(0)으로 즉시 종료.
 *
 * 호출 체인: gpgpusim_entrypoint.cc 또는 테스트 코드 → [이 함수] → p_opr->Print()
 */
void option_parser_print(option_parser_t opp, FILE *fout) {
  OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
  /* [한국어] 불투명 핸들 → OptionParser* 복원 */
  p_opr->Print(fout);  /* [한국어] 등록된 모든 옵션의 현재 값을 fout에 출력 */
}

// #define UNIT_TEST
/* [한국어] UNIT_TEST 매크로 정의 여부에 따라 아래 테스트 코드가 컴파일됨.
 *          기본적으로 주석 처리되어 있으며, 파서 기능 검증 시에만 활성화한다. */
#ifdef UNIT_TEST
/* [한국어] UNIT_TEST가 정의된 경우에만 컴파일되는 단위 테스트 블록.
 *          C++ 인터페이스, C 인터페이스, 구분자 기반 문자열 파싱을 각각 검증한다. */

/*
 * [한국어]
 * testtype - 파서 테스트용 데이터 컨테이너 구조체
 *
 * 다양한 타입의 옵션 변수를 하나의 구조체에 모아 옵션 파싱 결과를 검증한다.
 * 이 구조체의 각 필드는 Register() 호출 시 참조로 바인딩되어 파싱 결과가
 * 직접 저장된다.
 */
class testtype {
 public:
  int idata;
  /* [한국어] 정수형 테스트 옵션 변수.
   * 설정자: Register<int>("-idata", ...) 호출 후 파싱 시 fromString<int>가 수정.
   * 읽는 자: 테스트 코드의 Print() 출력 및 cout << c.idata 직접 읽기.
   * 값 범위: int 전체 범위. 기본값: -456 (Register 시 지정).
   * 동기화: 싱글 스레드 테스트 코드 — 락 불필요. */

  float fdata;
  /* [한국어] 단정밀도 부동소수점 테스트 옵션 변수.
   * 설정자: Register<float>("-fdata", ...) 후 파싱 시 수정.
   * 읽는 자: Print() 출력.
   * 값 범위: float 전체 범위. 기본값: 0.001.
   * 동기화: 싱글 스레드 — 락 불필요. */

  string sdata;
  /* [한국어] 문자열 테스트 옵션 변수.
   * 설정자: Register<string>("-sdata", ...) 후 string 특수화 fromString이 직접 대입.
   * 읽는 자: Print() 출력 및 cout << c.sdata.
   * 값 범위: 임의 문자열. 기본값: "hellow".
   * 동기화: 싱글 스레드 — 락 불필요. */

  unsigned long long ulldata;
  /* [한국어] 64비트 부호 없는 정수 테스트 옵션 변수.
   * 설정자: Register<unsigned long long>("-ulldata", ...) 후 파싱 시 수정.
   * 읽는 자: Print() 출력.
   * 값 범위: 0 ~ 2^64-1. 기본값: 0x123456789abcdef1 (16진수 파싱 테스트용).
   * 동기화: 싱글 스레드 — 락 불필요. */

  bool bdata;
  /* [한국어] bool 플래그 테스트 옵션 변수.
   * 설정자: Register<bool>("-someflag", ...) 후 bool 특수화 fromString이 수정.
   * 읽는 자: Print() 출력.
   * 값 범위: true 또는 false. 기본값: false(0).
   * 동기화: 싱글 스레드 — 락 불필요. */

  unsigned int boolint;
  /* [한국어] bool 타입이지만 unsigned int 변수에 저장하는 테스트 케이스.
   * 설정자: Register<bool>("-otherflag", ..., (bool&)c.boolint, ...)로 bool& 캐스팅.
   * 읽는 자: Print() 출력.
   * 값 범위: 0 또는 1. 기본값: 1(true).
   * 동기화: 싱글 스레드 — 락 불필요. */

  char *coption;
  /* [한국어] C 문자열 포인터 테스트 옵션 변수.
   * 설정자: Register<char*>("-coption", ..., NULL)로 NULL 기본값 설정 가능 여부 검증.
   *         파싱 시 char* 특수화 fromString이 new char[]로 새 버퍼를 할당하고 갱신.
   * 읽는 자: Print() 출력.
   * 값 범위: NULL(기본값) 또는 파싱된 문자열 포인터.
   * 동기화: 싱글 스레드 — 락 불필요. */

  /*
   * [한국어]
   * testtype 기본 생성자 — 각 필드를 명시적으로 초기화.
   * Register() 호출 시 defaultvalue로 덮어씌워지므로 여기서의 초기화는
   * assignDefault() 이전의 임시 값이다.
   */
  testtype() : idata(0), fdata(0.0f), sdata(""), ulldata(0), bdata(false) {}
};

/*
 * [한국어]
 * cppinterfacetest - C++ OptionParser 클래스 직접 사용 테스트
 *
 * @argc: 커맨드라인 인자 수
 * @argv: 커맨드라인 인자 배열
 * @return: 항상 0 (테스트 성공)
 *
 * 동작:
 *   1. testtype 변수들을 C++ OptionParser에 직접 Register
 *   2. 기본값 출력 (Print)
 *   3. 커맨드라인 파싱 결과 출력 (ParseCommandLine)
 *   4. "test.config" 파일 파싱 결과 출력 (ParseFile)
 *
 * 호출 체인: main() → [이 함수]
 */
int cppinterfacetest(int argc, const char *argv[]) {
  testtype c;               /* [한국어] 테스트용 옵션 변수 컨테이너 생성 */
  OptionParser optionparser;  /* [한국어] C++ OptionParser 직접 인스턴스화 */
  c.idata = 123;            /* [한국어] 초기값 설정 — Register 시 defaultvalue("-456")로 덮어씌워질 예정 */
  c.fdata = 3249586.333;    /* [한국어] float 초기값 — defaultvalue("0.001")로 교체될 예정 */
  c.sdata = string("haha"); /* [한국어] string 초기값 — defaultvalue("hellow")로 교체될 예정 */

  optionparser.Register<int>("-idata", "integer data", c.idata, "-456");
  /* [한국어] int 옵션 등록: c.idata 참조 바인딩, 기본값 -456 */
  optionparser.Register<float>("-fdata", "floating point data", c.fdata,
                               "0.001");
  /* [한국어] float 옵션 등록: c.fdata 참조 바인딩, 기본값 0.001 */
  optionparser.Register<string>("-sdata", "first string data", c.sdata,
                                "hellow");
  /* [한국어] string 옵션 등록: c.sdata 참조 바인딩, 기본값 "hellow" */
  optionparser.Register<unsigned long long>(
      "-ulldata", "unsigned long long data", c.ulldata, "0x123456789abcdef1");
  /* [한국어] uint64 옵션 등록: 16진수 기본값 파싱 테스트 */
  optionparser.Register<bool>("-someflag", "first flag", c.bdata, "0");
  /* [한국어] bool 플래그 등록: 기본값 false(0) */
  optionparser.Register<bool>("-otherflag", "second flag", (bool &)c.boolint,
                              "1");
  /* [한국어] unsigned int를 bool&로 캐스팅하여 등록 — bool 파싱 결과를 int 변수에 저장하는 테스트 */
  optionparser.Register<char *>("-coption", "char * data", c.coption, NULL);
  /* [한국어] char* 옵션 등록: NULL 기본값 허용 테스트 */

  cout << "Default: \n";          /* [한국어] 기본값 출력 시작 표시 */
  optionparser.Print(stdout);     /* [한국어] assignDefault() 직후의 기본값 출력 */

  optionparser.ParseCommandLine(argc, argv);  /* [한국어] 커맨드라인 인자 파싱 */

  cout << "Commandline Parse Results: \n";  /* [한국어] 커맨드라인 파싱 결과 출력 시작 */
  optionparser.Print(stdout);               /* [한국어] 파싱 후 변경된 값 출력 */

  optionparser.ParseFile("test.config");  /* [한국어] "test.config" 파일 파싱 */
  cout << "File Parse Results: \n";       /* [한국어] 파일 파싱 결과 출력 시작 */
  optionparser.Print(stdout);             /* [한국어] 파일 파싱 후 변경된 값 출력 */
  cout << c.sdata << ' ' << c.idata << endl;  /* [한국어] 파싱 결과를 원본 변수에서 직접 읽어 검증 */

  return 0;  /* [한국어] 테스트 성공 반환 */
}

/*
 * [한국어]
 * cinterfacetest - C 인터페이스(option_parser_*) 사용 테스트
 *
 * @argc: 커맨드라인 인자 수
 * @argv: 커맨드라인 인자 배열
 * @return: 항상 0 (테스트 성공)
 *
 * cppinterfacetest와 동일한 시나리오를 C 인터페이스로 검증한다.
 * option_parser_t 핸들을 통해 모든 작업이 이루어지며, 마지막에
 * option_parser_destroy()로 메모리를 해제한다.
 *
 * 호출 체인: main() → [이 함수]
 */
int cinterfacetest(int argc, const char *argv[]) {
  testtype c;                               /* [한국어] 테스트용 옵션 변수 컨테이너 */
  option_parser_t opp = option_parser_create();  /* [한국어] C 인터페이스로 파서 생성 */
  c.idata = 123;                            /* [한국어] int 초기값 */
  c.fdata = 3249586.333;                    /* [한국어] float 초기값 */
  c.sdata = string("haha");                 /* [한국어] string 초기값 */
  char *otherstr;                           /* [한국어] OPT_CSTR 옵션 결과를 받을 char* 변수 */

  option_parser_register(opp, "-idata", OPT_INT32, &c.idata, "integer data",
                         "-456");
  /* [한국어] int 옵션 C 인터페이스 등록: &c.idata(void*)를 변수 주소로 전달 */
  option_parser_register(opp, "-fdata", OPT_FLOAT, &c.fdata,
                         "floating point data", "0.001");
  /* [한국어] float 옵션 등록 */
  option_parser_register(opp, "-sdata", OPT_CSTR, &otherstr,
                         "first string data", "hellow");
  /* [한국어] C 문자열 옵션 등록: &otherstr(char**)를 전달 — 파싱 시 new[]로 갱신됨 */
  option_parser_register(opp, "-ulldata", OPT_UINT64, &c.ulldata,
                         "unsigend long long data", "0x123456789abcdef1");
  /* [한국어] uint64 옵션 등록 */
  option_parser_register(opp, "-someflag", OPT_BOOL, &c.bdata, "first flag",
                         "0");
  /* [한국어] bool 플래그 등록: OPT_BOOL 타입 */
  option_parser_register(opp, "-otherflag", OPT_BOOL, &c.boolint, "second flag",
                         "1");
  /* [한국어] unsigned int 변수를 OPT_BOOL로 등록 — C에서는 타입 캐스팅 없이 void*로 전달 */
  option_parser_register(opp, "-coption", OPT_CSTR, &c.coption, "char * data",
                         NULL);
  /* [한국어] char* 옵션 등록: NULL 기본값 테스트 */

  printf("Default: \n");            /* [한국어] 기본값 출력 시작 */
  option_parser_print(opp, stdout); /* [한국어] 기본값 출력 */

  option_parser_cmdline(opp, argc, argv);  /* [한국어] 커맨드라인 파싱 */

  printf("Commandline Parse Results: \n");  /* [한국어] 커맨드라인 파싱 결과 출력 시작 */
  option_parser_print(opp, stdout);         /* [한국어] 파싱 후 값 출력 */

  option_parser_cfgfile(opp, "test.config");  /* [한국어] 설정 파일 파싱 */
  printf("File Parse Results: \n");           /* [한국어] 파일 파싱 결과 출력 시작 */
  option_parser_print(opp, stdout);           /* [한국어] 파싱 후 값 출력 */
  printf("%s %d\n", otherstr, c.idata);       /* [한국어] 원본 변수에서 직접 값을 읽어 검증 */

  option_parser_destroy(opp);  /* [한국어] C 인터페이스로 파서 및 모든 옵션 객체 해제 */

  return 0;  /* [한국어] 테스트 성공 반환 */
}

/*
 * [한국어]
 * stringparsertest - 구분자 기반 문자열 파싱 테스트
 *
 * @return: 항상 0 (테스트 성공)
 *
 * option_parser_delimited_string()의 두 가지 포맷을 테스트한다:
 *   1. 공백+세미콜론 구분자: "ABC 1111; DEF 88; Mode A; Name out"
 *   2. 공백+등호+세미콜론 구분자: "Name=dram;DEF=702;Mode=B;ABC=-9573;"
 *
 * 호출 체인: main() → [이 함수]
 */
int stringparsertest() {
  int tABC;       /* [한국어] 정수형 테스트 옵션 변수 */
  int tDEF;       /* [한국어] 정수형 테스트 옵션 변수 */
  char tMode;     /* [한국어] 단일 문자 테스트 옵션 변수 */
  char *tName;    /* [한국어] C 문자열 포인터 테스트 옵션 변수 */

  option_parser_t opp = option_parser_create();  /* [한국어] 파서 생성 */
  option_parser_register(opp, "ABC", OPT_INT32, &tABC, "tABC", "34");
  /* [한국어] 정수 옵션 "ABC" 등록: 기본값 34 */
  option_parser_register(opp, "DEF", OPT_INT32, &tDEF, "tDEF", "-56");
  /* [한국어] 정수 옵션 "DEF" 등록: 기본값 -56 */
  option_parser_register(opp, "Mode", OPT_CHAR, &tMode, "tMode", "P");
  /* [한국어] 문자 옵션 "Mode" 등록: 기본값 'P' */
  option_parser_register(opp, "Name", OPT_CSTR, &tName, "tName", "Cache");
  /* [한국어] 문자열 옵션 "Name" 등록: 기본값 "Cache" */

  option_parser_delimited_string(opp, "ABC 1111; DEF 88; Mode A; Name out",
                                 " ;");
  /* [한국어] 공백+세미콜론 구분자로 문자열 파싱: ABC=1111, DEF=88, Mode=A, Name=out */
  printf("String Parse Results: \n");  /* [한국어] 첫 번째 문자열 파싱 결과 출력 시작 */
  option_parser_print(opp, stdout);    /* [한국어] 파싱 결과 출력 */

  option_parser_delimited_string(opp, "Name=dram;DEF=702;Mode=B;ABC=-9573;",
                                 " =;");
  /* [한국어] 공백+등호+세미콜론 구분자로 파싱: Name=dram, DEF=702, Mode=B, ABC=-9573 */
  printf("String Parse Results: \n");  /* [한국어] 두 번째 문자열 파싱 결과 출력 시작 */
  option_parser_print(opp, stdout);    /* [한국어] 파싱 결과 출력 */

  return 0;  /* [한국어] 테스트 성공 반환 */
}

/*
 * [한국어]
 * main - UNIT_TEST 진입점: 세 가지 파서 테스트 순차 실행
 *
 * @argc: 커맨드라인 인자 수 (cppinterfacetest/cinterfacetest에 전달)
 * @argv: 커맨드라인 인자 배열
 * @return: 항상 0 (모든 테스트 성공)
 *
 * 세 테스트를 순차 실행:
 *   1. cppinterfacetest: C++ 클래스 직접 사용 검증
 *   2. cinterfacetest: C 함수 인터페이스 검증
 *   3. stringparsertest: 구분자 기반 문자열 파싱 검증
 *
 * 호출 체인: OS → [이 함수] → cppinterfacetest(), cinterfacetest(), stringparsertest()
 */
int main(int argc, const char *argv[]) {
  cppinterfacetest(argc, argv);  /* [한국어] C++ 인터페이스 테스트 실행 */
  cinterfacetest(argc, argv);    /* [한국어] C 인터페이스 테스트 실행 */
  stringparsertest();            /* [한국어] 구분자 문자열 파싱 테스트 실행 */

  return 0;  /* [한국어] 모든 테스트 완료, 정상 종료 */
}

#endif
