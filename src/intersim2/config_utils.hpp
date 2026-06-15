// $Id: config_utils.hpp 5188 2012-08-30 00:31:31Z dub $

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
 * [한국어 설명] NoC 시뮬레이터 설정 파일 파서 및 Configuration 클래스 선언 (config_utils.hpp)
 *
 * === 파일의 역할 ===
 * Configuration 클래스는 Booksim2/intersim2 NoC 시뮬레이터의 설정 파일
 * (booksimrc 또는 GPGPU-Sim의 icnt_config 섹션)을 파싱하여 키=값 형태의
 * 설정 데이터를 세 가지 타입(문자열/정수/실수) 맵에 저장한다.
 * Lex/Yacc 기반 파서(config.l / config.y)가 파일을 토크나이즈하고
 * Configuration 객체의 Assign() 메서드를 호출해 값을 채운다.
 * 파싱된 설정은 시뮬레이터 초기화 시 Buffer, BufferState, Router, Network 등
 * 모든 구성 요소가 GetInt/GetStr/GetFloat로 읽어 사용한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 계층: gpu-sim.cc → icnt_wrapper → intersim2
 *   → gpgpusim_entrypoint.cc 또는 main에서 Configuration 객체 생성
 *   → ParseFile() 또는 ParseString()으로 설정 로드
 *   → 모든 intersim2 모듈이 Configuration const & config를 받아 초기화에 사용
 * 실행 컨텍스트: 시뮬레이터 초기화 단계 (시뮬레이션 루프 시작 전).
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - booksim.hpp:  공통 타입 및 매크로
 *   - config.l/.y:  Lex/Yacc 파서 — yyparse()가 Assign()을 호출
 * 이 파일을 사용하는 모듈:
 *   - buffer.hpp, buffer_state.hpp, vc.hpp: num_vcs, buf_size, vc_buf_size 등 읽기
 *   - dram 등 GPGPU-Sim 모듈: 간접적으로 option_parser와 병렬 사용
 *   - icnt_wrapper.cc: ParseFile/ParseString으로 설정 로드
 * 공유 자원:
 *   - theConfig: 전역 싱글턴 포인터 — GetTheConfig()로 어디서든 접근 가능
 *
 * === 주요 함수/구조체 요약 ===
 * Configuration():          생성자 — theConfig = this 등록, 파일 핸들 초기화
 * AddStrField/Assign():     설정 키-값 등록 및 수정
 * GetInt/GetStr/GetFloat(): 이미 등록된 키의 값 반환 (없으면 에러 후 exit)
 * GetIntArray/GetStrArray/GetFloatArray(): 중괄호 배열 형식 파싱
 * ParseFile/ParseString():  설정 파일 또는 문자열을 yyparse()로 파싱
 * ParseArgs():              커맨드라인 인수 파싱 (파일 경로 또는 key=value 오버라이드)
 * tokenize_str/int/float(): 중괄호 {a, b, c} 형태 배열 문자열 파싱 유틸리티
 */

#ifndef _CONFIG_UTILS_HPP_
#define _CONFIG_UTILS_HPP_

#include "booksim.hpp" // [한국어] intersim2 공통 타입 및 Error 매크로

#include<cstdio>   // [한국어] FILE*, fopen, fclose, fread — 설정 파일 읽기에 사용
#include<string>   // [한국어] std::string — 키/값 저장 타입
#include<map>      // [한국어] std::map — 키=값 설정 저장소 (_str_map, _int_map, _float_map)
#include<vector>   // [한국어] std::vector — GetIntArray 등 배열 반환 타입

/*
 * [한국어]
 * yyparse - Lex/Yacc 생성 파서의 파싱 진입점 (C 링크)
 *
 * @return: 0이면 파싱 성공, 1이면 구문 오류
 *
 * config.y(Yacc)와 config.l(Lex)가 생성하는 파서 함수.
 * ParseFile() 또는 ParseString() 내부에서 호출된다.
 * 파서는 설정 파일에서 "key = value;" 형태를 인식하여
 * config_assign_string/int/float() C 래퍼를 통해 Configuration::Assign()을 호출한다.
 */
int yyparse();

/*
 * [한국어]
 * Configuration - NoC 시뮬레이터 설정 키-값 저장소 (싱글턴 패턴)
 *
 * 모든 설정 항목을 문자열/정수/실수 세 가지 타입의 map에 분류하여 저장한다.
 * 생성자에서 전역 포인터 theConfig에 this를 등록하여 어디서든
 * Configuration::GetTheConfig()로 접근 가능한 싱글턴처럼 동작한다.
 * (단, 여러 Configuration 객체를 만들면 마지막 것이 theConfig가 됨에 주의.)
 */
class Configuration {
  static Configuration * theConfig;
  /* 전역 싱글턴 포인터 — 현재 활성화된 Configuration 인스턴스.
   * 설정자: 생성자에서 this로 초기화.
   * 읽는 자: GetTheConfig() — C 래퍼(config_assign_*, config_input) 등에서 사용.
   * 값 범위: NULL(초기화 전) 또는 유효한 Configuration 포인터.
   * 동기화: 단일 스레드 초기화 — 파싱 중 변경 없음. */

  FILE * _config_file;
  /* 파싱 중인 설정 파일의 파일 포인터.
   * 설정자: ParseFile()에서 fopen으로 열기; 파싱 완료 후 fclose 및 0으로 초기화.
   * 읽는 자: Input() — yyparse()의 입력 소스로 사용.
   * 값 범위: NULL(파일 파싱 중 아님) 또는 유효한 FILE 포인터.
   * 동기화: 단일 스레드 — 파싱은 순차적으로 수행됨. */

  string _config_string;
  /* 파싱 중인 설정 문자열 (ParseString() 호출 시 사용).
   * 설정자: ParseString()에서 str + ';'로 설정; 파싱 완료 후 빈 문자열로 초기화.
   * 읽는 자: Input() — _config_file이 NULL일 때 이 문자열에서 읽기.
   * 값 범위: 빈 문자열(파일 파싱 중) 또는 설정 문자열.
   * 동기화: 단일 스레드 — 불필요. */

protected:
  map<string,string> _str_map;
  /* 문자열 타입 설정 저장소 (키 → 문자열 값).
   * 설정자: AddStrField()로 키 등록; Assign(string,string)으로 값 갱신.
   * 읽는 자: GetStr(), GetStrArray() — 라우팅/토폴로지 설정 등.
   * 값 범위: 등록된 키에 대한 임의 문자열.
   * 동기화: 초기화 후 읽기 전용 (파싱 완료 후). */

  map<string,int>    _int_map;
  /* 정수 타입 설정 저장소 (키 → 정수 값).
   * 설정자: Assign(string,int)으로 값 갱신 (yyparse가 파서 액션에서 호출).
   * 읽는 자: GetInt() — 대부분의 모듈이 버퍼 크기, VC 수 등을 읽음.
   * 값 범위: 각 키에 대한 정수 범위 (설정 파일 의존).
   * 동기화: 초기화 후 읽기 전용. */

  map<string,double> _float_map;
  /* 실수 타입 설정 저장소 (키 → double 값).
   * 설정자: Assign(string,double)으로 값 갱신.
   * 읽는 자: GetFloat() — 트래픽 생성 파라미터 등.
   * 값 범위: 각 키에 대한 실수 범위 (설정 파일 의존).
   * 동기화: 초기화 후 읽기 전용. */

public:
  /*
   * [한국어]
   * Configuration 생성자 — 싱글턴 등록 및 파일 핸들 초기화
   *
   * @return: 없음 (생성자)
   *
   * theConfig = this 로 전역 싱글턴 포인터를 갱신하고, _config_file = 0으로 초기화.
   * 세 가지 맵(_str_map, _int_map, _float_map)은 기본 생성으로 빈 상태.
   *
   * 호출 체인:
   *   icnt_wrapper.cc 또는 main() → [Configuration::Configuration()]
   */
  Configuration();

  /*
   * [한국어]
   * AddStrField - 문자열 타입 설정 키를 맵에 등록 (초기 스키마 정의)
   *
   * @field: 등록할 키 이름 (예: "topology", "routing_function")
   * @value: 기본값 (예: "mesh", "dor")
   * @return: 없음
   *
   * _str_map[field] = value 로 키를 등록한다.
   * 이후 Assign()으로만 값 변경 가능 (키가 없으면 Assign이 에러를 출력).
   * 보통 시뮬레이터 초기화 코드에서 모든 설정 키의 기본값을 먼저 등록.
   *
   * 호출 체인:
   *   초기화 루틴 → [Configuration::AddStrField()] → _str_map 갱신
   */
  void AddStrField(string const & field, string const & value);

  /*
   * [한국어]
   * Assign (문자열) - 이미 등록된 문자열 설정 키의 값 갱신
   *
   * @field: 갱신할 키 이름
   * @value: 새 값
   * @return: 없음
   *
   * _str_map에 field가 없으면 "Unknown string field: field" 에러.
   * yyparse()가 파서 액션에서 config_assign_string()을 통해 간접 호출.
   *
   * 호출 체인:
   *   config_assign_string() → [Configuration::Assign(string,string)]
   */
  void Assign(string const & field, string const & value);

  /*
   * [한국어]
   * Assign (정수) - 이미 등록된 정수 설정 키의 값 갱신
   *
   * @field: 갱신할 키 이름
   * @value: 새 정수 값
   * @return: 없음
   *
   * _int_map에 field가 없으면 "Unknown integer field: field" 에러.
   *
   * 호출 체인:
   *   config_assign_int() → [Configuration::Assign(string,int)]
   */
  void Assign(string const & field, int value);

  /*
   * [한국어]
   * Assign (실수) - 이미 등록된 실수 설정 키의 값 갱신
   *
   * @field: 갱신할 키 이름
   * @value: 새 double 값
   * @return: 없음
   *
   * _float_map에 field가 없으면 "Unknown double field: field" 에러.
   *
   * 호출 체인:
   *   config_assign_float() → [Configuration::Assign(string,double)]
   */
  void Assign(string const & field, double value);

  /*
   * [한국어]
   * GetStr - 문자열 설정 값 반환
   *
   * @field: 조회할 키 이름
   * @return: 해당 키의 문자열 값
   *
   * _str_map에 field가 없으면 "Unknown string field: field" 에러 후 exit(-1).
   *
   * 호출 체인:
   *   Buffer/Router 등 초기화 → [Configuration::GetStr()]
   */
  string GetStr(string const & field) const;

  /*
   * [한국어]
   * GetInt - 정수 설정 값 반환
   *
   * @field: 조회할 키 이름
   * @return: 해당 키의 정수 값
   *
   * _int_map에 field가 없으면 "Unknown integer field: field" 에러 후 exit(-1).
   * num_vcs, buf_size 등 대부분의 NoC 설정이 이 함수로 읽힌다.
   *
   * 호출 체인:
   *   Buffer/BufferState/VC 등 초기화 → [Configuration::GetInt()]
   */
  int GetInt(string const & field) const;

  /*
   * [한국어]
   * GetFloat - 실수 설정 값 반환
   *
   * @field: 조회할 키 이름
   * @return: 해당 키의 double 값
   *
   * _float_map에 field가 없으면 에러 후 exit(-1).
   *
   * 호출 체인:
   *   트래픽 생성기 등 초기화 → [Configuration::GetFloat()]
   */
  double GetFloat(string const & field) const;

  /*
   * [한국어]
   * GetStrArray - 중괄호 배열 형식의 문자열 설정 값 파싱하여 벡터로 반환
   *
   * @field: 조회할 키 이름 (값이 "{a, b, c}" 형태)
   * @return: 파싱된 문자열 벡터
   *
   * GetStr(field)로 원시 문자열을 가져온 뒤 tokenize_str()에 위임.
   *
   * 호출 체인:
   *   SharedBufferPolicy 생성자 등 → [Configuration::GetStrArray()]
   */
  vector<string> GetStrArray(const string & field) const;

  /*
   * [한국어]
   * GetIntArray - 중괄호 배열 형식의 정수 설정 값 파싱하여 벡터로 반환
   *
   * @field: 조회할 키 이름 (값이 "{1, 2, 3}" 형태)
   * @return: 파싱된 정수 벡터 (빈 배열이면 빈 벡터)
   *
   * GetStr(field)로 원시 문자열을 가져온 뒤 tokenize_int()에 위임.
   * SharedBufferPolicy에서 private_buf_size, private_buf_start_vc 등 읽기에 사용.
   *
   * 호출 체인:
   *   SharedBufferPolicy 생성자 → [Configuration::GetIntArray()] → tokenize_int()
   */
  vector<int> GetIntArray(const string & field) const;

  /*
   * [한국어]
   * GetFloatArray - 중괄호 배열 형식의 실수 설정 값 파싱하여 벡터로 반환
   *
   * @field: 조회할 키 이름 (값이 "{0.1, 0.2}" 형태)
   * @return: 파싱된 double 벡터
   *
   * 호출 체인:
   *   트래픽 생성기 등 → [Configuration::GetFloatArray()] → tokenize_float()
   */
  vector<double> GetFloatArray(const string & field) const;

  /*
   * [한국어]
   * ParseFile - 설정 파일을 열어 Lex/Yacc 파서로 파싱
   *
   * @filename: 설정 파일 경로 문자열
   * @return: 없음
   *
   * fopen으로 파일 열기 실패 시 cerr 출력 후 exit(-1).
   * _config_file 설정 후 yyparse() 호출로 파싱 수행.
   * 완료 후 fclose 및 _config_file = 0으로 초기화.
   * yyparse()는 Input()을 통해 _config_file에서 데이터를 읽는다.
   *
   * 호출 체인:
   *   ParseArgs() 또는 icnt_wrapper → [Configuration::ParseFile()] → yyparse() → Assign() 반복
   */
  void ParseFile(string const & filename);

  /*
   * [한국어]
   * ParseString - 설정 문자열을 Lex/Yacc 파서로 파싱
   *
   * @str: 파싱할 설정 문자열 (예: "num_vcs = 4;")
   * @return: 없음
   *
   * str에 세미콜론을 붙여 _config_string에 저장 후 yyparse() 호출.
   * 완료 후 _config_string 초기화.
   * 커맨드라인 "key=value" 오버라이드나 GPGPU-Sim의 인라인 설정에 사용.
   *
   * 호출 체인:
   *   ParseArgs() 또는 icnt_wrapper → [Configuration::ParseString()] → yyparse()
   */
  void ParseString(string const & str);

  /*
   * [한국어]
   * Input - Lex 파서의 입력 소스 함수 (yyparse()가 내부적으로 호출)
   *
   * @line:     읽어 들인 데이터를 저장할 버퍼
   * @max_size: 버퍼의 최대 크기
   * @return:   실제로 읽은 바이트 수
   *
   * _config_file이 NULL이 아니면 fread로 파일에서 읽기.
   * NULL이면 _config_string에서 복사 후 문자열 비우기.
   * Lex 파서가 이 함수를 입력 콜백으로 사용한다.
   *
   * 호출 체인:
   *   yyparse() → config_input() → [Configuration::Input()]
   */
  int  Input(char * line, int max_size);

  /*
   * [한국어]
   * ParseError - 파싱 에러 출력 후 시뮬레이터 종료
   *
   * @msg:    에러 메시지 문자열
   * @lineno: 에러 발생 라인 번호 (0이면 라인 정보 없음)
   * @return: 없음 (exit(-1)으로 종료)
   *
   * 알 수 없는 필드를 Assign/Get할 때 또는 Lex/Yacc 구문 에러 시 호출.
   * config_error() C 래퍼 함수가 yyparse() 에러 처리에서 이 함수를 호출.
   *
   * 호출 체인:
   *   Assign/GetStr/GetInt 등 → [Configuration::ParseError()] → exit(-1)
   */
  void ParseError(string const & msg, unsigned int lineno = 0) const;

  /*
   * [한국어]
   * WriteFile - 현재 설정 전체를 파일에 기록 (GUI 도구용)
   *
   * @filename: 출력 파일 경로
   * @return: 없음
   *
   * _str_map, _int_map, _float_map의 모든 항목을 "key = value;" 형식으로 기록.
   * 빈 문자열 값은 파서가 파싱할 수 없으므로 건너뜀.
   *
   * 호출 체인:
   *   GUI 또는 분석 도구 → [Configuration::WriteFile()]
   */
  void WriteFile(string const & filename);

  /*
   * [한국어]
   * WriteMatlabFile - 현재 설정을 MATLAB 주석 형식으로 기록
   *
   * @o: 출력 스트림 포인터
   * @return: 없음
   *
   * 모든 설정 항목을 "% key = value;" 형식으로 출력.
   * MATLAB 분석 스크립트에서 설정을 참조할 때 사용.
   *
   * 호출 체인:
   *   분석 도구 → [Configuration::WriteMatlabFile()]
   */
  void WriteMatlabFile(ostream * o) const;

  /*
   * [한국어]
   * GetStrMap - 문자열 설정 맵 전체에 대한 const 레퍼런스 반환 (인라인)
   *
   * @return: _str_map const 레퍼런스
   *
   * 설정 항목 전체를 순회하거나 외부로 노출할 때 사용.
   */
  inline const map<string, string> & GetStrMap() const {
    return _str_map; // [한국어] 문자열 설정 맵 전체를 읽기 전용으로 반환
  }

  /*
   * [한국어]
   * GetIntMap - 정수 설정 맵 전체에 대한 const 레퍼런스 반환 (인라인)
   *
   * @return: _int_map const 레퍼런스
   */
  inline const map<string, int> & GetIntMap() const {
    return _int_map; // [한국어] 정수 설정 맵 전체를 읽기 전용으로 반환
  }

  /*
   * [한국어]
   * GetFloatMap - 실수 설정 맵 전체에 대한 const 레퍼런스 반환 (인라인)
   *
   * @return: _float_map const 레퍼런스
   */
  inline const map<string, double> & GetFloatMap() const {
    return _float_map; // [한국어] 실수 설정 맵 전체를 읽기 전용으로 반환
  }

  /*
   * [한국어]
   * GetTheConfig - 전역 싱글턴 포인터 반환 (정적 메서드)
   *
   * @return: 현재 활성화된 Configuration 인스턴스 포인터
   *
   * C 래퍼 함수(config_assign_*, config_input, config_error)가
   * 직접 Configuration 포인터 없이 이 메서드로 현재 객체를 얻는다.
   *
   * 호출 체인:
   *   config_assign_string/int/float(), config_input(), config_error() → [Configuration::GetTheConfig()]
   */
  static Configuration * GetTheConfig();

};

/*
 * [한국어]
 * ParseArgs - 커맨드라인 인수를 파싱하여 설정 파일 로드 또는 파라미터 오버라이드
 *
 * @cf:   대상 Configuration 객체 포인터
 * @argc: 커맨드라인 인수 개수
 * @argv: 커맨드라인 인수 배열
 * @return: true이면 설정 파일이 하나 이상 파싱됨
 *
 * '-'로 시작하는 인수는 무시(다른 용도의 플래그).
 * '='를 포함하면 "key=value" 오버라이드 파라미터 → ParseString()으로 처리.
 * 그 외이면 설정 파일 경로 → ParseFile()로 로드 및 파일 내용 cout 출력.
 *
 * 호출 체인:
 *   main() 또는 icnt_wrapper 초기화 → [ParseArgs()] → cf->ParseFile() 또는 cf->ParseString()
 */
bool ParseArgs(Configuration * cf, int argc, char **argv);

/*
 * [한국어]
 * tokenize_str - 중괄호 배열 형식 문자열을 토크나이즈하여 벡터로 반환
 *
 * @data: 파싱할 문자열 (예: "{a, b, {c, d}}" 또는 단순 "a")
 * @return: 파싱된 문자열 원소들의 벡터
 *
 * '{' 로 시작하지 않으면 단일 원소로 처리.
 * '{' 로 시작하면 중첩 중괄호 레벨을 추적하며 쉼표/닫는 괄호로 원소 분리.
 * 중첩된 {a, {b, c}, d} 형식도 지원 (nested 카운터로 추적).
 *
 * 호출 체인:
 *   Configuration::GetStrArray() → [tokenize_str()]
 */
vector<string> tokenize_str(string const & data);

/*
 * [한국어]
 * tokenize_int - 중괄호 배열 형식 문자열을 정수 벡터로 파싱
 *
 * @data: 파싱할 문자열 (예: "{1, 2, 3}" 또는 단순 "4")
 * @return: 파싱된 정수 원소들의 벡터
 *
 * tokenize_str()과 동일한 파싱 로직 사용, atoi()로 정수 변환.
 * GetIntArray() 내부에서 사용됨.
 *
 * 호출 체인:
 *   Configuration::GetIntArray() → [tokenize_int()]
 */
vector<int> tokenize_int(string const & data);

/*
 * [한국어]
 * tokenize_float - 중괄호 배열 형식 문자열을 실수 벡터로 파싱
 *
 * @data: 파싱할 문자열 (예: "{0.1, 0.5}" 또는 단순 "1.0")
 * @return: 파싱된 double 원소들의 벡터
 *
 * tokenize_str()과 동일한 파싱 로직 사용, atof()로 실수 변환.
 *
 * 호출 체인:
 *   Configuration::GetFloatArray() → [tokenize_float()]
 */
vector<double> tokenize_float(string const & data);

#endif
