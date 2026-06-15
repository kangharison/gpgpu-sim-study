// $Id: config_utils.cpp 5188 2012-08-30 00:31:31Z dub $
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

/*config_utils.cpp
 *
 *The configuration object which contained the parsed data from the
 *configuration file
 */

/*
 * [한국어 설명] NoC 시뮬레이터 설정 파서 및 Configuration 클래스 구현 (config_utils.cpp)
 *
 * === 파일의 역할 ===
 * Configuration 클래스의 모든 멤버 함수와 관련 C 인터페이스 래퍼 함수, 그리고
 * 중괄호 배열 토크나이저(tokenize_str/int/float)를 구현한다.
 * Lex/Yacc 파서(config.l / config.y)가 설정 파일을 읽으면서
 * config_assign_string/int/float() C 래퍼를 호출하고,
 * 이들이 Configuration::Assign()을 통해 내부 맵에 값을 저장한다.
 * yyparse()의 입력은 config_input() → Configuration::Input()을 통해
 * 파일 또는 문자열 소스에서 공급된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 계층: icnt_wrapper → intersim2 초기화
 *   → Configuration 객체 생성 → ParseFile/ParseString 호출
 *   → yyparse() → config_assign_* 콜백 → Assign() → 맵 저장
 *   → 모든 모듈의 생성자에서 GetInt/GetStr/GetFloat로 설정 읽기
 * 실행 컨텍스트: 시뮬레이터 초기화 단계 (시뮬레이션 루프 시작 전).
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - booksim.hpp:      Error 매크로, 공통 타입
 *   - config_utils.hpp: Configuration 클래스 선언, yyparse 선언
 *   - config.l/.y:      Lex/Yacc 파서 (config_error/assign/input C 래퍼를 호출)
 * 이 파일을 사용하는 모듈:
 *   - buffer.cpp, buffer_state.cpp: GetInt("num_vcs"), GetInt("buf_size") 등
 *   - icnt_wrapper.cc: ParseFile(), ParseString() 직접 호출
 *
 * === 주요 함수/구조체 요약 ===
 * Configuration::Assign(): 키-값 쌍을 해당 타입 맵에 저장 (overload 3개)
 * Configuration::GetInt/GetStr/GetFloat(): 키로 값 조회 (없으면 exit)
 * Configuration::ParseFile/ParseString(): Lex/Yacc 파서 실행
 * Configuration::Input(): yyparse의 입력 소스 콜백
 * config_assign_xxx / config_input(): C 링크 래퍼 (yyparse 에서 호출)
 * ParseArgs(): 커맨드라인 인수 파싱
 * tokenize_str/int/float(): {a,b,c} 형식 배열 파싱 공통 로직
 */

#include "booksim.hpp" // [한국어] Error 매크로, 공통 타입/헤더
#include <iostream>    // [한국어] cerr, cout — 에러 메시지 및 설정 파일 내용 출력
#include <cstring>     // [한국어] memcpy 등 — 직접 사용은 없으나 의존성 포함
#include <sstream>     // [한국어] ostringstream — 에러 메시지 동적 생성 (미사용이지만 포함)
#include <fstream>     // [한국어] ifstream — ParseArgs에서 설정 파일 내용을 cout으로 출력할 때 사용
#include <cstdlib>     // [한국어] exit(), atoi(), atof() — 에러 종료 및 문자열→숫자 변환

#include "config_utils.hpp" // [한국어] Configuration 클래스 선언 및 tokenize/ParseArgs 선언

/* =========================================================================
 * 전역 싱글턴 초기화
 * ========================================================================= */

Configuration *Configuration::theConfig = 0;
/* [한국어] 전역 Configuration 싱글턴 포인터 초기화.
 * 프로그램 시작 시 NULL로 초기화되고, 첫 번째 Configuration 생성자 호출 시 this로 갱신.
 * 설정자: Configuration() 생성자. 읽는 자: GetTheConfig() → C 래퍼들. */

/* =========================================================================
 * Configuration 생성자
 * ========================================================================= */

/*
 * [한국어]
 * Configuration::Configuration - 싱글턴 등록 및 파일 핸들 초기화
 *
 * @return: 없음 (생성자)
 *
 * theConfig = this 로 전역 포인터를 갱신해 C 래퍼 함수들이 이 객체를 찾을 수 있게 한다.
 * _config_file을 0(NULL)으로 초기화한다.
 *
 * 호출 체인:
 *   icnt_wrapper 초기화 → [Configuration::Configuration()]
 */
Configuration::Configuration()
{
  theConfig = this; // [한국어] 전역 싱글턴 포인터를 현재 인스턴스로 갱신 — C 래퍼가 이 객체를 참조
  _config_file = 0; // [한국어] 파일 파싱 시작 전 파일 핸들 NULL 초기화
}

/* =========================================================================
 * 설정 등록/갱신 함수 구현
 * ========================================================================= */

/*
 * [한국어]
 * Configuration::AddStrField - 문자열 타입 설정 키를 맵에 등록 (스키마 정의)
 *
 * @field: 등록할 키 이름
 * @value: 기본값 문자열
 * @return: 없음
 *
 * _str_map[field] = value 로 직접 삽입한다.
 * 이 함수로 먼저 등록되지 않은 키는 Assign()에서 에러가 발생하므로
 * 파싱 전에 모든 허용 키를 AddStrField()로 미리 등록해야 한다.
 *
 * 호출 체인:
 *   시뮬레이터 초기화 → [Configuration::AddStrField()]
 */
void Configuration::AddStrField(string const & field, string const & value)
{
  _str_map[field] = value; // [한국어] 문자열 설정 맵에 키=기본값 쌍을 추가 또는 덮어쓰기
}

/*
 * [한국어]
 * Configuration::Assign (문자열) - 이미 등록된 문자열 키의 값 갱신
 *
 * @field: 갱신할 키 이름
 * @value: 새 문자열 값
 * @return: 없음
 *
 * _str_map에서 field를 검색하여 있으면 값을 갱신하고,
 * 없으면 ParseError("Unknown string field: " + field)를 호출해 exit.
 *
 * 호출 체인:
 *   config_assign_string() → [Configuration::Assign(string, string)]
 */
void Configuration::Assign(string const & field, string const & value)
{
  map<string, string>::const_iterator match; // [한국어] _str_map 검색 결과 반복자

  match = _str_map.find(field); // [한국어] 이 키가 이미 등록된 문자열 설정인지 검색
  if(match != _str_map.end()) { // [한국어] 키가 존재하면 값 갱신
    _str_map[field] = value; // [한국어] 기존 값을 새 값으로 덮어쓰기
  } else { // [한국어] 등록되지 않은 키에 대한 Assign 시도 — 파서 에러
    ParseError("Unknown string field: " + field); // [한국어] 알 수 없는 문자열 설정 키 에러 출력 후 exit
  }
}

/*
 * [한국어]
 * Configuration::Assign (정수) - 이미 등록된 정수 키의 값 갱신
 *
 * @field: 갱신할 키 이름
 * @value: 새 정수 값
 * @return: 없음
 *
 * _int_map에서 field를 검색하여 있으면 갱신, 없으면 ParseError.
 *
 * 호출 체인:
 *   config_assign_int() → [Configuration::Assign(string, int)]
 */
void Configuration::Assign(string const & field, int value)
{
  map<string, int>::const_iterator match; // [한국어] _int_map 검색 결과 반복자

  match = _int_map.find(field); // [한국어] 정수 설정 맵에서 키 검색
  if(match != _int_map.end()) { // [한국어] 키가 존재하면 값 갱신
    _int_map[field] = value; // [한국어] 기존 정수 값을 새 값으로 덮어쓰기
  } else { // [한국어] 미등록 키 — 설정 파일에 정의되지 않은 정수 파라미터
    ParseError("Unknown integer field: " + field); // [한국어] 알 수 없는 정수 설정 키 에러 후 exit
  }
}

/*
 * [한국어]
 * Configuration::Assign (실수) - 이미 등록된 실수 키의 값 갱신
 *
 * @field: 갱신할 키 이름
 * @value: 새 double 값
 * @return: 없음
 *
 * _float_map에서 field를 검색하여 있으면 갱신, 없으면 ParseError.
 *
 * 호출 체인:
 *   config_assign_float() → [Configuration::Assign(string, double)]
 */
void Configuration::Assign(string const & field, double value)
{
  map<string, double>::const_iterator match; // [한국어] _float_map 검색 결과 반복자

  match = _float_map.find(field); // [한국어] 실수 설정 맵에서 키 검색
  if(match != _float_map.end()) { // [한국어] 키가 존재하면 값 갱신
    _float_map[field] = value; // [한국어] 기존 실수 값을 새 값으로 덮어쓰기
  } else { // [한국어] 미등록 키
    ParseError("Unknown double field: " + field); // [한국어] 알 수 없는 실수 설정 키 에러 후 exit
  }
}

/* =========================================================================
 * 설정 조회 함수 구현
 * ========================================================================= */

/*
 * [한국어]
 * Configuration::GetStr - 문자열 설정 값 반환
 *
 * @field: 조회할 키 이름
 * @return: 등록된 문자열 값
 *
 * _str_map에서 field를 검색해 있으면 값 반환, 없으면 ParseError 후 exit(-1).
 *
 * 호출 체인:
 *   Buffer/Router 등 초기화 → [Configuration::GetStr()] → match->second 반환
 */
string Configuration::GetStr(string const & field) const
{
  map<string, string>::const_iterator match; // [한국어] _str_map 검색 결과 반복자

  match = _str_map.find(field); // [한국어] 문자열 설정 맵에서 키 검색
  if(match != _str_map.end()) { // [한국어] 키가 존재하면 값 반환
    return match->second; // [한국어] 등록된 문자열 값 반환 (맵 원소의 value 부분)
  } else { // [한국어] 미등록 키 — 존재하지 않는 설정 조회 시도
    ParseError("Unknown string field: " + field); // [한국어] 에러 메시지 출력
    exit(-1); // [한국어] ParseError가 exit하지 않는 경우를 위한 명시적 종료 (도달하지 않음)
  }
}

/*
 * [한국어]
 * Configuration::GetInt - 정수 설정 값 반환
 *
 * @field: 조회할 키 이름
 * @return: 등록된 정수 값
 *
 * _int_map에서 field를 검색해 있으면 값 반환, 없으면 ParseError 후 exit(-1).
 * 가장 빈번하게 호출되는 함수 — num_vcs, buf_size, vc_buf_size 등.
 *
 * 호출 체인:
 *   Buffer/BufferState/VC/Router 등 생성자 → [Configuration::GetInt()] → match->second 반환
 */
int Configuration::GetInt(string const & field) const
{
  map<string, int>::const_iterator match; // [한국어] _int_map 검색 결과 반복자

  match = _int_map.find(field); // [한국어] 정수 설정 맵에서 키 검색
  if(match != _int_map.end()) { // [한국어] 키가 존재하면 값 반환
    return match->second; // [한국어] 등록된 정수 값 반환
  } else { // [한국어] 미등록 키
    ParseError("Unknown integer field: " + field); // [한국어] 에러 메시지 출력
    exit(-1); // [한국어] 명시적 종료 (컴파일러 경고 방지 및 안전 보장)
  }
}

/*
 * [한국어]
 * Configuration::GetFloat - 실수 설정 값 반환
 *
 * @field: 조회할 키 이름
 * @return: 등록된 double 값
 *
 * _float_map에서 field를 검색해 있으면 값 반환, 없으면 ParseError 후 exit(-1).
 *
 * 호출 체인:
 *   트래픽 생성기 등 → [Configuration::GetFloat()] → match->second 반환
 */
double Configuration::GetFloat(string const & field) const
{
  map<string,double>::const_iterator match; // [한국어] _float_map 검색 결과 반복자

  match = _float_map.find(field); // [한국어] 실수 설정 맵에서 키 검색
  if(match != _float_map.end()) { // [한국어] 키가 존재하면 값 반환
    return match->second; // [한국어] 등록된 실수 값 반환
  } else { // [한국어] 미등록 키
    ParseError("Unknown double field: " + field); // [한국어] 에러 메시지 출력
    exit(-1); // [한국어] 명시적 종료
  }
}

/*
 * [한국어]
 * Configuration::GetStrArray - 문자열 배열 설정 파싱 반환
 *
 * @field: 조회할 키 이름 (값이 "{a, b}" 형태)
 * @return: 파싱된 문자열 벡터
 *
 * GetStr()로 원시 문자열을 가져온 뒤 tokenize_str()에 위임한다.
 *
 * 호출 체인:
 *   SharedBufferPolicy 생성자 → [Configuration::GetStrArray()] → tokenize_str()
 */
vector<string> Configuration::GetStrArray(string const & field) const
{
  string const param_str = GetStr(field); // [한국어] 이 키의 원시 문자열 값 읽기 (예: "{dor, oblivious}")
  return tokenize_str(param_str);         // [한국어] 중괄호 배열 형식 파싱하여 문자열 벡터 반환
}

/*
 * [한국어]
 * Configuration::GetIntArray - 정수 배열 설정 파싱 반환
 *
 * @field: 조회할 키 이름 (값이 "{1, 2, 3}" 형태)
 * @return: 파싱된 정수 벡터 (빈 문자열이면 빈 벡터)
 *
 * GetStr()로 원시 문자열을 가져온 뒤 tokenize_int()에 위임한다.
 * SharedBufferPolicy에서 private_buf_size, start_vc, end_vc 등 읽기.
 *
 * 호출 체인:
 *   SharedBufferPolicy 생성자 → [Configuration::GetIntArray()] → tokenize_int()
 */
vector<int> Configuration::GetIntArray(string const & field) const
{
  string const param_str = GetStr(field); // [한국어] 이 키의 원시 문자열 값 읽기 (예: "{1, 2, 4}")
  return tokenize_int(param_str);         // [한국어] 중괄호 배열 형식 파싱하여 정수 벡터 반환
}

/*
 * [한국어]
 * Configuration::GetFloatArray - 실수 배열 설정 파싱 반환
 *
 * @field: 조회할 키 이름 (값이 "{0.1, 0.5}" 형태)
 * @return: 파싱된 double 벡터
 *
 * 호출 체인:
 *   트래픽 생성기 등 → [Configuration::GetFloatArray()] → tokenize_float()
 */
vector<double> Configuration::GetFloatArray(string const & field) const
{
  string const param_str = GetStr(field); // [한국어] 이 키의 원시 문자열 값 읽기
  return tokenize_float(param_str);       // [한국어] 중괄호 배열 형식 파싱하여 실수 벡터 반환
}

/* =========================================================================
 * 파싱 함수 구현
 * ========================================================================= */

/*
 * [한국어]
 * Configuration::ParseFile - 설정 파일을 열어 yyparse()로 파싱
 *
 * @filename: 설정 파일 경로
 * @return: 없음
 *
 * fopen 실패 시 cerr에 에러 메시지 출력 후 exit(-1).
 * 성공 시 _config_file 설정 → yyparse() 호출 → fclose → _config_file = 0.
 * yyparse()는 내부적으로 config_input()을 통해 _config_file에서 읽는다.
 *
 * 호출 체인:
 *   icnt_wrapper 또는 ParseArgs() → [Configuration::ParseFile()] → yyparse() → Assign() 반복
 */
void Configuration::ParseFile(string const & filename)
{
  if((_config_file = fopen(filename.c_str(), "r")) == 0) { // [한국어] 설정 파일 읽기 모드로 열기 (실패하면 0 반환)
    cerr << "Could not open configuration file " << filename << endl; // [한국어] 파일 열기 실패 에러 메시지
    exit(-1); // [한국어] 파일 없으면 시뮬레이션 진행 불가 — 강제 종료
  }

  yyparse(); // [한국어] Lex/Yacc 파서 실행 — 설정 파일 전체를 토큰화하여 Assign() 호출

  fclose(_config_file); // [한국어] 파싱 완료 후 파일 핸들 닫기
  _config_file = 0;     // [한국어] 파일 핸들 NULL로 초기화 — 이후 Input()이 파일 읽기 시도 방지
}

/*
 * [한국어]
 * Configuration::ParseString - 설정 문자열을 yyparse()로 파싱
 *
 * @str: 파싱할 설정 문자열 (예: "num_vcs = 4" 또는 "buf_size = 16")
 * @return: 없음
 *
 * str에 ';'를 붙여 _config_string에 저장 (Lex 파서가 세미콜론을 문장 종결로 인식).
 * yyparse() 호출 후 _config_string을 빈 문자열로 초기화.
 * 커맨드라인 오버라이드 파라미터 또는 GPGPU-Sim 인라인 설정 전달에 사용.
 *
 * 호출 체인:
 *   ParseArgs() 또는 icnt_wrapper → [Configuration::ParseString()] → yyparse()
 */
void Configuration::ParseString(string const & str)
{
  _config_string = str + ';'; // [한국어] 설정 문자열에 세미콜론 추가 — Lex 파서의 문장 종결자
  yyparse();                   // [한국어] Lex/Yacc 파서 실행 — _config_string에서 토큰화 및 Assign() 호출
  _config_string = "";          // [한국어] 파싱 완료 후 문자열 초기화 — Input()이 다시 시도하지 않도록
}

/*
 * [한국어]
 * Configuration::Input - Lex 파서의 입력 소스 콜백 (yyparse가 내부적으로 호출)
 *
 * @line:     읽어 들인 데이터를 저장할 버퍼 포인터
 * @max_size: 버퍼의 최대 크기 (바이트)
 * @return:   실제로 읽은 바이트 수 (0이면 입력 종료)
 *
 * _config_file이 NULL이 아니면 fread로 파일에서 최대 max_size 바이트 읽기.
 * NULL이면 _config_string의 내용을 line 버퍼에 복사 후 문자열 비우기.
 * Lex 파서가 YY_INPUT 매크로를 통해 이 함수를 입력 소스로 사용한다.
 *
 * 호출 체인:
 *   yyparse() → config_input() → [Configuration::Input()]
 */
int Configuration::Input(char * line, int max_size)
{
  int length = 0; // [한국어] 실제 읽은 바이트 수 초기화

  if(_config_file) { // [한국어] 파일 파싱 모드 — _config_file이 유효한 파일 핸들
    length = fread(line, 1, max_size, _config_file); // [한국어] 파일에서 최대 max_size 바이트 읽기 (C 표준 파일 I/O)
  } else { // [한국어] 문자열 파싱 모드 — _config_string에서 읽기
    length = _config_string.length(); // [한국어] 남은 문자열 길이 계산
    _config_string.copy(line, max_size); // [한국어] _config_string 내용을 line 버퍼에 복사
    _config_string.clear();             // [한국어] 복사 후 문자열 비우기 — 다음 Input() 호출 시 0 반환하여 입력 종료
  }

  return length; // [한국어] 읽은 바이트 수 반환 (0이면 Lex 파서가 EOF로 처리)
}

/*
 * [한국어]
 * Configuration::ParseError - 파싱 에러 메시지 출력 후 시뮬레이터 강제 종료
 *
 * @msg:    표시할 에러 메시지 문자열
 * @lineno: 에러 발생 라인 번호 (0이면 라인 정보 없이 출력)
 * @return: 없음 (exit(-1)으로 종료)
 *
 * lineno가 0이면 "Parse error : msg" 형식으로 출력.
 * 양수이면 "Parse error on line N : msg" 형식으로 출력.
 * 에러 후 exit(-1)로 강제 종료한다.
 *
 * 호출 체인:
 *   Assign()/GetStr()/GetInt() 등 → [Configuration::ParseError()] → exit(-1)
 */
void Configuration::ParseError(string const & msg, unsigned int lineno) const
{
  if(lineno) { // [한국어] 라인 번호가 있으면 (Lex 에러 등) 라인 정보 포함 출력
    cerr << "Parse error on line " << lineno << " : " << msg << endl; // [한국어] 라인 번호 포함 에러 출력
  } else { // [한국어] 라인 번호 없는 에러 (필드 미등록 등)
    cerr << "Parse error : " << msg << endl; // [한국어] 라인 번호 없는 에러 출력
  }

  exit( -1 ); // [한국어] 복구 불가능한 설정 오류 — 시뮬레이터 즉시 종료
}

/*
 * [한국어]
 * Configuration::GetTheConfig - 전역 싱글턴 포인터 반환 (정적 메서드)
 *
 * @return: theConfig — 현재 활성화된 Configuration 인스턴스 포인터
 *
 * C 링크 래퍼 함수들이 직접 Configuration* 포인터 없이
 * 이 메서드를 통해 현재 Configuration 객체에 접근한다.
 *
 * 호출 체인:
 *   config_assign_*() / config_input() / config_error() → [Configuration::GetTheConfig()]
 */
Configuration * Configuration::GetTheConfig()
{
  return theConfig; // [한국어] 전역 싱글턴 포인터 반환 — 생성자에서 설정된 인스턴스
}

/* =========================================================================
 * C 링크 래퍼 함수 구현 (Lex/Yacc 파서가 호출)
 * ========================================================================= */

/*
 * [한국어]
 * config_error - Lex/Yacc 파서의 에러 콜백 (C 링크)
 *
 * @msg:    에러 메시지 문자열 (char* C 스타일)
 * @lineno: 에러 발생 라인 번호
 * @return: 없음
 *
 * yyparse()가 구문 오류를 감지할 때 이 함수를 호출한다.
 * Configuration::ParseError()로 위임하여 처리.
 *
 * 호출 체인:
 *   yyparse() → [config_error()] → Configuration::ParseError()
 */
void config_error( char * msg, int lineno )
{
  Configuration::GetTheConfig( )->ParseError( msg, lineno ); // [한국어] 싱글턴에 에러 처리 위임 — 에러 출력 후 exit
}

/*
 * [한국어]
 * config_assign_string - 문자열 설정 값 할당 C 래퍼 (Lex/Yacc → C++ 연결)
 *
 * @field: 설정 키 이름 (C 스타일 문자열)
 * @value: 설정 값 (C 스타일 문자열)
 * @return: 없음
 *
 * Lex/Yacc 파서 액션에서 C 링크로 직접 호출 가능하도록 C 스타일 인터페이스 제공.
 * Configuration::Assign(string, string)으로 위임.
 *
 * 호출 체인:
 *   yyparse() (파서 액션) → [config_assign_string()] → Configuration::Assign()
 */
 void config_assign_string( char const * field, char const * value )
{
  Configuration::GetTheConfig()->Assign(field, value); // [한국어] C 스타일 인수를 std::string으로 변환하여 Assign 위임
}

/*
 * [한국어]
 * config_assign_int - 정수 설정 값 할당 C 래퍼
 *
 * @field: 설정 키 이름
 * @value: 정수 값
 * @return: 없음
 *
 * yyparse()의 파서 액션에서 정수 토큰 인식 시 호출.
 *
 * 호출 체인:
 *   yyparse() → [config_assign_int()] → Configuration::Assign(string, int)
 */
void config_assign_int( char const * field, int value )
{
  Configuration::GetTheConfig()->Assign(field, value); // [한국어] 정수 설정 값을 Assign에 위임
}

/*
 * [한국어]
 * config_assign_float - 실수 설정 값 할당 C 래퍼
 *
 * @field: 설정 키 이름
 * @value: double 실수 값
 * @return: 없음
 *
 * yyparse()의 파서 액션에서 실수 토큰 인식 시 호출.
 *
 * 호출 체인:
 *   yyparse() → [config_assign_float()] → Configuration::Assign(string, double)
 */
void config_assign_float( char const * field, double value )
{
  Configuration::GetTheConfig()->Assign(field, value); // [한국어] 실수 설정 값을 Assign에 위임
}

/*
 * [한국어]
 * config_input - Lex 파서의 입력 소스 C 래퍼
 *
 * @line:     읽어 들인 데이터를 저장할 버퍼
 * @max_size: 버퍼 최대 크기
 * @return:   실제 읽은 바이트 수
 *
 * YY_INPUT 매크로가 이 함수를 사용하여 입력을 공급한다.
 * Configuration::Input()으로 위임하여 파일 또는 문자열 소스에서 읽기.
 *
 * 호출 체인:
 *   yyparse() 내부 YY_INPUT → [config_input()] → Configuration::Input()
 */
int config_input(char * line, int max_size)
{
  return Configuration::GetTheConfig()->Input(line, max_size); // [한국어] 파일 또는 문자열 소스에서 파서 입력 공급
}

/* =========================================================================
 * 커맨드라인 인수 파싱
 * ========================================================================= */

/*
 * [한국어]
 * ParseArgs - 커맨드라인 인수를 파싱하여 설정 파일 로드 또는 파라미터 오버라이드
 *
 * @cf:   대상 Configuration 객체 포인터
 * @argc: 커맨드라인 인수 개수
 * @argv: 커맨드라인 인수 배열 (argv[0]은 프로그램 이름 — 건너뜀)
 * @return: true이면 설정 파일이 하나 이상 파싱됨 (시뮬레이션 실행 조건 충족)
 *
 * argv[1]부터 순서대로 처리:
 *   - '-'로 시작하면 무시 (다른 도구의 옵션 플래그)
 *   - '='를 포함하면 "key=value" 오버라이드 → ParseString()으로 즉시 적용
 *   - 그 외이면 설정 파일 경로 → ParseFile()로 파싱,
 *     파일 내용을 cout에도 출력 (로그 목적)
 *
 * 호출 체인:
 *   main() → [ParseArgs(cf, argc, argv)] → cf->ParseFile() 또는 cf->ParseString()
 */
bool ParseArgs(Configuration * cf, int argc, char * * argv)
{
  bool rc = false; // [한국어] 설정 파일 파싱 성공 여부 — 하나라도 파싱되면 true

  //all dashed variables are ignored by the arg parser
  for(int i = 1; i < argc; ++i) { // [한국어] argv[1]부터 마지막 인수까지 순회 (argv[0]은 프로그램 이름)
    string arg(argv[i]);           // [한국어] C 스타일 문자열을 std::string으로 변환
    size_t pos = arg.find('=');    // [한국어] '=' 위치 검색 — key=value 형태인지 판단
    bool dash = (argv[i][0] =='-'); // [한국어] '-'로 시작하면 옵션 플래그로 무시
    if(pos == string::npos && !dash) { // [한국어] '='도 없고 '-'도 없으면 설정 파일 경로
      // parse config file
      cf->ParseFile( argv[i] ); // [한국어] 설정 파일 파싱 — 파일 경로를 ParseFile에 전달
      ifstream in(argv[i]);     // [한국어] 설정 파일 내용을 cout에 출력하기 위해 ifstream 열기
      cout << "BEGIN Configuration File: " << argv[i] << endl; // [한국어] 설정 파일 내용 출력 시작 표시
      while (!in.eof()) { // [한국어] 파일 끝까지 문자 단위로 읽어 cout에 출력
	char c;
	in.get(c);   // [한국어] 파일에서 한 문자 읽기
	cout << c ;  // [한국어] 읽은 문자를 즉시 cout에 출력 (설정 파일 내용 로그)
      }
      cout << "END Configuration File: " << argv[i] << endl; // [한국어] 설정 파일 내용 출력 종료 표시
      rc = true; // [한국어] 설정 파일이 하나 이상 파싱됨 — true로 갱신
    } else if(pos != string::npos)  { // [한국어] '='가 있으면 key=value 오버라이드 파라미터
      // override individual parameter
      cout << "OVERRIDE Parameter: " << arg << endl; // [한국어] 오버라이드 파라미터 로그 출력
      cf->ParseString(argv[i]); // [한국어] "key=value;" 형태 문자열을 파서로 처리하여 설정 갱신
    }
    // [한국어] '-'로 시작하는 인수는 else 분기 없이 무시됨 (dash == true인 경우)
  }

  return rc; // [한국어] 설정 파일 파싱 여부 반환 (false이면 시뮬레이션 설정 없음)
}


//helpful for the GUI, write out nearly all variables contained in a config file.
//However, it can't and won't write out  empty strings since the booksim yacc
//parser won't be abled to parse blank strings

/*
 * [한국어]
 * Configuration::WriteFile - 현재 설정 전체를 파일에 기록 (GUI/분석 도구용)
 *
 * @filename: 출력 파일 경로
 * @return: 없음
 *
 * _str_map, _int_map, _float_map의 모든 항목을 "key = value;" 형식으로 기록한다.
 * 빈 문자열 값은 booksim Yacc 파서가 파싱할 수 없으므로 건너뛴다.
 * 완료 후 flush하고 동적 할당된 ofstream을 delete한다.
 * 이 함수는 정상 시뮬레이션 흐름에서는 호출되지 않으며 GUI 지원 목적임.
 *
 * 호출 체인:
 *   GUI 또는 분석 도구 → [Configuration::WriteFile()]
 */
void Configuration::WriteFile(string const & filename) {

  ostream *config_out= new ofstream(filename.c_str()); // [한국어] 출력 파일 스트림 동적 생성


  for(map<string,string>::const_iterator i = _str_map.begin();
      i!=_str_map.end();
      i++){ // [한국어] 문자열 설정 맵 전체 순회
    //the parser won't read empty strings
    if(i->second[0]!='\0'){ // [한국어] 빈 문자열은 파서가 처리 불가 — 건너뜀
      *config_out<<i->first<<" = "<<i->second<<";"<<endl; // [한국어] "key = value;" 형식으로 파일에 기록
    }
  }

  for(map<string, int>::const_iterator i = _int_map.begin();
      i!=_int_map.end();
      i++){ // [한국어] 정수 설정 맵 전체 순회
    *config_out<<i->first<<" = "<<i->second<<";"<<endl; // [한국어] 정수 설정 "key = value;" 형식으로 기록

  }

  for(map<string, double>::const_iterator i = _float_map.begin();
      i!=_float_map.end();
      i++){ // [한국어] 실수 설정 맵 전체 순회
    *config_out<<i->first<<" = "<<i->second<<";"<<endl; // [한국어] 실수 설정 "key = value;" 형식으로 기록

  }
  config_out->flush(); // [한국어] 버퍼에 남은 데이터를 파일에 강제 쓰기
  delete config_out;   // [한국어] 동적 할당된 ofstream 해제 (파일도 닫힘)

}



/*
 * [한국어]
 * Configuration::WriteMatlabFile - 현재 설정을 MATLAB 주석 형식으로 기록
 *
 * @config_out: 출력 스트림 포인터
 * @return: 없음
 *
 * 모든 설정 항목을 "% key = value;" 형식으로 출력한다.
 * MATLAB 스크립트에서 %로 시작하는 줄은 주석이므로 직접 실행 가능한 형태는 아님.
 * 빈 문자열 값은 건너뜀.
 *
 * 호출 체인:
 *   MATLAB 분석 스크립트 지원 도구 → [Configuration::WriteMatlabFile()]
 */
void Configuration::WriteMatlabFile(ostream * config_out) const {

  for(map<string,string>::const_iterator i = _str_map.begin();
      i!=_str_map.end();
      i++){ // [한국어] 문자열 설정 맵 전체 순회
    //the parser won't read blanks lolz
    if(i->second[0]!='\0'){ // [한국어] 빈 문자열 건너뜀
      *config_out<<"%"<<i->first<<" = \'"<<i->second<<"\';"<<endl; // [한국어] MATLAB 주석 형식으로 문자열 설정 출력 (작은따옴표로 감싸기)
    }
  }

  for(map<string, int>::const_iterator i = _int_map.begin();
      i!=_int_map.end();
      i++){ // [한국어] 정수 설정 맵 전체 순회
    *config_out<<"%"<<i->first<<" = "<<i->second<<";"<<endl; // [한국어] MATLAB 주석 형식으로 정수 설정 출력

  }

  for(map<string, double>::const_iterator i = _float_map.begin();
      i!=_float_map.end();
      i++){ // [한국어] 실수 설정 맵 전체 순회
    *config_out<<"%"<<i->first<<" = "<<i->second<<";"<<endl; // [한국어] MATLAB 주석 형식으로 실수 설정 출력

  }
  config_out->flush(); // [한국어] 출력 버퍼 강제 플러시

}

/* =========================================================================
 * 배열 토크나이저 구현 (세 가지 타입 공통 로직)
 * ========================================================================= */

/*
 * [한국어]
 * tokenize_str - 중괄호 배열 형식 문자열을 토크나이즈하여 문자열 벡터 반환
 *
 * @data: 파싱할 원시 문자열
 *        예) "{dor, oblivious}" → ["dor", "oblivious"]
 *        예) "dor" → ["dor"] (단일 원소, 중괄호 없음)
 *        예) "" → [] (빈 입력)
 * @return: 파싱된 문자열 원소들의 벡터
 *
 * 파싱 알고리즘:
 *   1. 빈 문자열이면 빈 벡터 반환
 *   2. '{'로 시작하지 않으면 단일 원소로 취급
 *   3. '{'로 시작하면: start=1에서 시작, {/,/} 문자를 탐색
 *      - '{': nested++ (중첩 깊이 증가)
 *      - '}' + nested>0: nested-- (닫힘)
 *      - ',' 또는 '}' + nested==0: 현재 위치 ~ start 범위가 하나의 토큰
 *        → token = data.substr(start, curr-start) 추출 후 push_back
 *   4. 루프 종료 후 assert(!nested) — 중괄호 쌍이 맞는지 검증
 *
 * 호출 체인:
 *   Configuration::GetStrArray() → [tokenize_str()]
 */
vector<string> tokenize_str(string const & data)
{
  vector<string> values; // [한국어] 파싱된 문자열 원소를 담을 벡터

  // no elements, no braces --> empty list
  if(data.empty()) { // [한국어] 빈 문자열이면 빈 벡터 반환
    return values;
  }

  // doesn't start with an opening brace --> treat as single element
  // note that this element can potentially contain nested lists
  if(data[0] != '{') { // [한국어] '{' 없이 단순 문자열이면 전체를 하나의 원소로 취급
    values.push_back(data); // [한국어] 전체 문자열을 단일 원소로 추가
    return values;
  }

  size_t start = 1; // [한국어] 현재 토큰 시작 위치 ('{' 다음 문자부터 시작)
  int nested = 0;   // [한국어] 현재 중첩 중괄호 깊이 카운터 (0이면 최상위 레벨)

  size_t curr = start; // [한국어] 현재 탐색 위치

  while(string::npos != (curr = data.find_first_of("{,}", curr))) {
    // [한국어] '{', ',', '}' 중 하나를 탐색 — 없으면 find_first_of가 npos 반환하여 루프 종료

    if(data[curr] == '{') { // [한국어] 중첩 중괄호 시작 — 깊이 증가
      ++nested; // [한국어] 중첩 레벨 1 증가 (이 '{' 이후의 ',', '}'는 토큰 구분자로 사용하지 않음)
    } else if((data[curr] == '}') && nested) { // [한국어] 중첩된 '}'이면 — 깊이 감소
      --nested; // [한국어] 중첩 레벨 1 감소 (가장 안쪽의 '}' 처리)
    } else if(!nested) { // [한국어] 최상위 레벨('}'나 ','을 만남) — 토큰 추출
      if(curr > start) { // [한국어] 현재 위치와 시작 위치가 다르면 (비어있지 않은 토큰)
	string token = data.substr(start, curr - start); // [한국어] start부터 curr까지의 부분 문자열을 토큰으로 추출
	values.push_back(token); // [한국어] 추출된 토큰을 결과 벡터에 추가
      }
      start = curr + 1; // [한국어] 다음 토큰 시작 위치를 현재 구분자 다음으로 이동
    }
    ++curr; // [한국어] 탐색 위치를 1 이동하여 현재 문자를 다시 탐색하지 않도록
  }
  assert(!nested); // [한국어] 루프 종료 후 중첩 레벨이 0이어야 함 — '{' 와 '}' 짝이 맞지 않으면 버그

  return values; // [한국어] 파싱된 문자열 원소 벡터 반환
}

/*
 * [한국어]
 * tokenize_int - 중괄호 배열 형식 문자열을 정수 벡터로 파싱
 *
 * @data: 파싱할 원시 문자열
 *        예) "{1, 2, 4}" → [1, 2, 4]
 *        예) "8" → [8]
 * @return: 파싱된 정수 원소들의 벡터
 *
 * tokenize_str()과 동일한 파싱 로직 사용. 단, 토큰을 atoi()로 정수 변환.
 * GetIntArray()에서 호출됨 (예: private_buf_size, private_buf_start_vc 등).
 *
 * 호출 체인:
 *   Configuration::GetIntArray() → [tokenize_int()]
 */
vector<int> tokenize_int(string const & data)
{
  vector<int> values; // [한국어] 파싱된 정수 원소를 담을 벡터

  // no elements, no braces --> empty list
  if(data.empty()) { // [한국어] 빈 문자열이면 빈 벡터 반환
    return values;
  }

  // doesn't start with an opening brace --> treat as single element
  // note that this element can potentially contain nested lists
  if(data[0] != '{') { // [한국어] '{' 없이 단순 문자열이면 단일 정수로 변환
    values.push_back(atoi(data.c_str())); // [한국어] 전체 문자열을 atoi()로 정수 변환 후 추가
    return values;
  }

  size_t start = 1; // [한국어] 현재 토큰 시작 위치
  int nested = 0;   // [한국어] 중첩 중괄호 깊이 카운터

  size_t curr = start; // [한국어] 현재 탐색 위치

  while(string::npos != (curr = data.find_first_of("{,}", curr))) {
    // [한국어] '{', ',', '}' 탐색

    if(data[curr] == '{') { // [한국어] 중첩 '{' — 깊이 증가
      ++nested;
    } else if((data[curr] == '}') && nested) { // [한국어] 중첩 '}' — 깊이 감소
      --nested;
    } else if(!nested) { // [한국어] 최상위 레벨 구분자 — 토큰 추출
      if(curr > start) { // [한국어] 비어있지 않은 토큰
	string token = data.substr(start, curr - start); // [한국어] 부분 문자열 추출
	values.push_back(atoi(token.c_str())); // [한국어] atoi()로 정수 변환 후 추가
      }
      start = curr + 1; // [한국어] 다음 토큰 시작 위치 이동
    }
    ++curr; // [한국어] 탐색 위치 이동
  }
  assert(!nested); // [한국어] 중괄호 쌍이 맞는지 검증

  return values; // [한국어] 파싱된 정수 벡터 반환
}

/*
 * [한국어]
 * tokenize_float - 중괄호 배열 형식 문자열을 실수 벡터로 파싱
 *
 * @data: 파싱할 원시 문자열
 *        예) "{0.1, 0.5, 0.9}" → [0.1, 0.5, 0.9]
 *        예) "1.0" → [1.0]
 * @return: 파싱된 double 원소들의 벡터
 *
 * tokenize_str()과 동일한 파싱 로직 사용. 토큰을 atof()로 실수 변환.
 *
 * 호출 체인:
 *   Configuration::GetFloatArray() → [tokenize_float()]
 */
vector<double> tokenize_float(string const & data)
{
  vector<double> values; // [한국어] 파싱된 실수 원소를 담을 벡터

  // no elements, no braces --> empty list
  if(data.empty()) { // [한국어] 빈 문자열이면 빈 벡터 반환
    return values;
  }

  // doesn't start with an opening brace --> treat as single element
  // note that this element can potentially contain nested lists
  if(data[0] != '{') { // [한국어] '{' 없이 단순 문자열이면 단일 실수로 변환
    values.push_back(atof(data.c_str())); // [한국어] 전체 문자열을 atof()로 실수 변환 후 추가
    return values;
  }

  size_t start = 1; // [한국어] 현재 토큰 시작 위치
  int nested = 0;   // [한국어] 중첩 중괄호 깊이 카운터

  size_t curr = start; // [한국어] 현재 탐색 위치

  while(string::npos != (curr = data.find_first_of("{,}", curr))) {
    // [한국어] '{', ',', '}' 탐색

    if(data[curr] == '{') { // [한국어] 중첩 '{' — 깊이 증가
      ++nested;
    } else if((data[curr] == '}') && nested) { // [한국어] 중첩 '}' — 깊이 감소
      --nested;
    } else if(!nested) { // [한국어] 최상위 레벨 구분자 — 토큰 추출
      if(curr > start) { // [한국어] 비어있지 않은 토큰
	string token = data.substr(start, curr - start); // [한국어] 부분 문자열 추출
	values.push_back(atof(token.c_str())); // [한국어] atof()로 실수 변환 후 추가
      }
      start = curr + 1; // [한국어] 다음 토큰 시작 위치 이동
    }
    ++curr; // [한국어] 탐색 위치 이동
  }
  assert(!nested); // [한국어] 중괄호 쌍이 맞는지 검증

  return values; // [한국어] 파싱된 실수 벡터 반환
}
