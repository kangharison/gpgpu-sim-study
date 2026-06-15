/*
 * [한국어 설명] PTX(BIN) Bison/Yacc 문법 파서 (ptx.y)
 *
 * === 파일의 역할 ===
 * cuobjdump가 출력한 PTX 텍스트(특히 텍스처, 상수 메모리, 함수 파라미터
 * 등의 헤더 선언 부분)를 파싱하여 ptx_parser.h에 정의된 전역 함수들을
 * 호출하는 Bison 문법 분석기 정의 파일이다. 실제 PTXPlus 변환기의
 * 핵심은 sass.y와 cuobjdumpInst.cc에 있으며, 이 파일은 주로 텍스처
 * 리스트 등 부가 정보를 구성하는 보조 파서 역할을 한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   cuobjdump_to_ptxplus.cc
 *     → ptx_parse() (이 파일이 생성)
 *         → ptx.l (토큰 생성)
 *             → 이 파일의 규칙에 따라 headerList/텍스처 정보 구축
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - ptx_parser.h : 의미 규칙에서 호출하는 전역 헤더 처리 함수
 *   - ptx.l        : PTX 토큰 제공
 * 이 파일에 의존하는 모듈:
 *   - 빌드 시스템  : bison ptx.y → ptx_parser.hh/ptx.tab.cc 생성
 *
 * === 주요 함수/구조체 요약 ===
 * program           : 선언(decl)들의 집합
 * version_decl      : .version <major> <minor>
 * target_decl       : .target <target>
 * address_size_decl : .address_size <bits>
 * file_decl         : .file <id> "<path>"
 * function_decl     : function_directive + optional (param_list) + block
 * function_directive: .entry/.func <name>
 * function_info     : in/out directives + maxntid/minnctapersm + stmt_list
 * directive_spec    : .reg/.global/.local/.param/.shared/.const/.tex
 *                     + type + vector_size + identifier
 * type_spec         : PTX scalar/vector/predicate types
 * directive         : directive_spec + optional array_spec
 * param_list        : .param declarations
 * statement         : single_statement + SEMICOLON | block
 */

// Copyright (c) 2009-2012, Jimmy Kwa, Andrew Boktor
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ptx_parser.h"

/* [한국어] Bison 푸시 파서용 인자. */
#define YYPARSE_PARAM scanner
#define YYLEX_PARAM   scanner

/* [한국어] bison 에러 핸들러. */
void ptx_error(YYLTYPE * locp, const char* s);
int ptx_lex( ... );

%}

%locations
%pure-parser
%error-verbose

%union {
  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  void * ptr_value;
}


%token <string_value> IDENTIFIER
%token <string_value> HEXLITERAL
%token <string_value> OCTLITERAL
%token <string_value> BINLITERAL
%token <string_value> DECLITERAL
%token <string_value> REGISTER_DECL
%token <string_value> VECTOR_REGISTER_DECL
%token <string_value> PRED_REGISTER_DECL
%token <string_value> SPECIAL_REGISTER

%token VERSION TARGET ADDRESS_SIZE ENTRY FUNC FILE LOC
%token SREG REG GLOBAL LOCAL PARAM SHARED CONST TEX SURF
%token V2 V4 V8
%token PRED
%token B8 B16 B32 B64 B128
%token U8 U16 U32 U64
%token S8 S16 S32 S64
%token F16 F32 F64
%token ALIGN
%token CALLPROTOTYPE CALLTARGETS PARAMETRICCALL

%token SEMICOLON COMMA LBRACE RBRACE LBRACKET RBRACKET LPAREN RPAREN LANGLE RANGLE
%token ATSIGN EXCLAMATION PIPE LEQ GEQ EQ NEQ OROR ANDAND SHL SHR PLUS MINUS STAR
%token DIV MOD AND XOR ASSIGN TILDE PERIOD

%type <string_value> function_identifier

%%

/* [한국어] 최상위 프로그램: version/target/address_size/file/decl/entry/func 등이
 * 0개 이상 반복될 수 있다. */
program:
					| program version_decl
					| program target_decl
					| program address_size_decl
					| program file_decl
					| program declaration
					| program entry_decl
					| program func_decl
					;

/* [한국어] .version <major> <minor>; add_version_info 호출. */
version_decl:
					VERSION DECLITERAL DECLITERAL SEMICOLON
					{
						add_version_info($2,$3);
					}
					;

/* [한국어] .target <target_name>; target_header 호출. */
target_decl:
					TARGET IDENTIFIER SEMICOLON
					{
						target_header($2);
					}
					;

/* [한국어] .address_size <bits>; address_size_header 호출. */
address_size_decl:
					ADDRESS_SIZE DECLITERAL SEMICOLON
					{
						address_size_header($2);
					}
					;

/* [한국어] .file <id> "<path>"; 현재 아무 동작도 하지 않음. */
file_decl:
					FILE DECLITERAL IDENTIFIER SEMICOLON
					;

/* [한국어] 전역 선언: directive_spec SEMICOLON. */
declaration:
					directive_spec SEMICOLON
					;

/* [한국어] directive_spec: 주소 공간 + 타입 + (벡터) + 식별자.
 * add_space_spec/add_scalar_type_spec/add_vector_spec/add_identifier
 * 등을 호출하여 헤더 정보를 누적한다. */
directive_spec:
					space_spec type_spec identifier_spec
					;

space_spec:
					GLOBAL { add_space_spec("global"); }
					| LOCAL { add_space_spec("local"); }
					| PARAM { add_space_spec("param"); }
					| SHARED { add_space_spec("shared"); }
					| CONST { add_space_spec("const"); }
					| REG { add_space_spec("reg"); }
					| SREG { add_space_spec("sreg"); }
					| TEX { add_space_spec("tex"); }
					| SURF { add_space_spec("surf"); }
					;

/* [한국어] 타입 지정: 스칼라/벡터/PRED. */
type_spec:
					scalar_type_spec
					| vector_type_spec
					| PRED { add_scalar_type_spec("pred"); }
					;

scalar_type_spec:
					B8 { add_scalar_type_spec("b8"); }
					| B16 { add_scalar_type_spec("b16"); }
					| B32 { add_scalar_type_spec("b32"); }
					| B64 { add_scalar_type_spec("b64"); }
					| B128 { add_scalar_type_spec("b128"); }
					| U8 { add_scalar_type_spec("u8"); }
					| U16 { add_scalar_type_spec("u16"); }
					| U32 { add_scalar_type_spec("u32"); }
					| U64 { add_scalar_type_spec("u64"); }
					| S8 { add_scalar_type_spec("s8"); }
					| S16 { add_scalar_type_spec("s16"); }
					| S32 { add_scalar_type_spec("s32"); }
					| S64 { add_scalar_type_spec("s64"); }
					| F16 { add_scalar_type_spec("f16"); }
					| F32 { add_scalar_type_spec("f32"); }
					| F64 { add_scalar_type_spec("f64"); }
					;

vector_type_spec:
					V2 scalar_type_spec { add_vector_spec(2); }
					| V4 scalar_type_spec { add_vector_spec(4); }
					| V8 scalar_type_spec { add_vector_spec(8); }
					;

/* [한국어] 식별자: 변수명 또는 레지스터 선언. */
identifier_spec:
					IDENTIFIER { add_identifier($1); }
					| REGISTER_DECL { add_identifier($1); }
					| PRED_REGISTER_DECL { add_identifier($1); }
					;

/* [한국어] 엔트리(.entry) 또는 함수(.func) 선언. */
entry_decl:
					ENTRY function_identifier { add_function_name($2); } optional_param_list function_info
					;

func_decl:
					FUNC function_identifier { add_function_name($2); } optional_param_list function_info
					;

function_identifier:
					IDENTIFIER { $$ = $1; }
					;

/* [한국어] 선택적 파라미터 목록: (.param ... ). */
optional_param_list:
					LPAREN param_list RPAREN
					|
					;

param_list:
					param_decl
					| param_list COMMA param_decl
					;

param_decl:
					PARAM type_spec identifier_spec
					;

/* [한국어] 함수 본체/정보: in/out 지시자, maxntid, minnctapersm,
 * perf_directive, stmt_list 등(현재 대부분 no-op). */
function_info:
					LBRACE perf_directive stmt_list RBRACE
					;

perf_directive:
					maxntid_directive
					| maxntid_directive minnctapersm_directive
					|
					;

maxntid_directive:
					/* maxntid */
					;

minnctapersm_directive:
					/* minnctapersm */
					;

stmt_list:
					stmt
					| stmt_list stmt
					;

stmt:
					single_statement SEMICOLON
					| block
					;

single_statement:
					/* instruction */
					;

block:
					LBRACE stmt_list RBRACE
					;

%%

/*
 * [한국어]
 * ptx_error - PTX 파서에서 문법 오류가 발생했을 때 호출
 *
 * @locp : 오류 발생 위치(YYLTYPE)
 * @s    : 오류 메시지
 *
 * 파일명, 줄 번호, 근처 텍스트와 함께 오류 메시지를 출력하고
 * exit(1)로 변환을 중단한다.
 */
void ptx_error(YYLTYPE * locp, const char* s)
{
	extern char *ptx_lex_current_line( void );
	extern const char *g_filename;
	printf("GPGPU-Sim PTX: Error (%s:%u) lexical error near \"%s\": %s\n",
		g_filename, locp->first_line, ptx_lex_current_line(), s );
	exit(1);
}
