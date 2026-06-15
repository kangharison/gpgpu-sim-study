/*
 * [한국어 설명] ptxas 리소스 사용 정보 문법 파서 (ptxinfo.y)
 *
 * === 파일의 역할 ===
 * 이 파일은 Flex 토크나이저(ptxinfo.l)가 생성한 토큰 스트림을 받아,
 * NVIDIA ptxas가 출력하는 커널별 리소스 사용량 정보를 해석하는 Bison(Yacc)
 * 문법 파일이다. "Used N registers", "M bytes lmem", "K bytes smem[bank]",
 * "X bytes gmem" 등의 ptxas info 라인을 파싱하여 function_info에 등록할
 * 레지스터 수, 로컬/공유/상수/전역 메모리 크기를 추출한다. 빌드 시 Bison이
 * ptxinfo.tab.c/ptxinfo.tab.h를 생성하고, ptxinfo_parse() 진입점이 여기서 나온다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CUDA App → libcuda 인터셉트 → gpgpusim_entrypoint.cc
 *   → ptxas 출력 캡처 → ptxinfo.l (ptxinfo_lex) → [이 파일: ptxinfo_parse()]
 *       → ptxinfo_* 콜백 (ptx_loader.cc 등) → function_info 리소스 설정
 *           → gpgpu-sim/ 타이밍 시뮬레이션 (점유율, 레지스터/공유메모리 할당)
 * 실행 컨텍스트: 호스트 유저스페이스, 커널 실행 전 초기화 단계.
 * 사이클-레벨 시뮬레이션과 무관한 전처리 단계이다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존하는 모듈: ptxinfo.l (Flex lexer, 토큰 스트림 공급), ptx_loader.h
 *   (ptxinfo_data 구조체), gpgpu_context.h (g_ptxinfo_error_detected 등)
 * - 이 파일에 의존하는 모듈: 빌드 시스템(Bison이 ptxinfo.tab.c/h 생성),
 *   ptx_loader.cc (ptxinfo_function, ptxinfo_regs, ptxinfo_lmem 등 콜백 구현),
 *   ptx_parser.cc/init_parser 경로에서 간접적으로 사용
 * - 데이터 흐름: ptxas 출력 문자열 → ptxinfo.l 토큰 → ptxinfo.y 문법 reduce →
 *   ptxinfo_function()/ptxinfo_regs()/ptxinfo_lmem() 등 콜백 → function_info
 *
 * === 주요 문법 규칙 요약 ===
 * - input: 최상위; line 반복
 * - line: HEADER INFO COLON line_info | HEADER WARNING | HEADER FATAL 등
 * - line_info: function_name | function_info | gmem_info
 * - function_name: FUNC QUOTE IDENTIFIER QUOTE (선택적 FOR ...)
 * - function_info: info (COMMA info 반복)
 * - info: 레지스터/로컬/공유/상수/전역 메모리/텍스처 사용량
 * - tuple: declared + system bytes (lmem/smem의 declared/system 구분)
 * - duplicate: 중복 정의된 함수/변수 이름
 */

/*
Copyright (c) 2009-2011, Tor M. Aamodt
The University of British Columbia
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
Neither the name of The University of British Columbia nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

%{
typedef void * yyscan_t;       // [한국어] Flex 재진입 스캐너 핸들 타입
#include "ptx_loader.h"        // [한국어] ptxinfo_data 및 관련 함수 선언
%}

%define api.pure full          // [한국어] 순수 재진입 파서 생성
%parse-param {yyscan_t scanner} // [한국어] yyparse 첫 번째 인자: Flex 스캐너 핸들
%parse-param {ptxinfo_data* ptxinfo} // [한국어] yyparse 두 번째 인자: ptxinfo_data 포인터
%lex-param {yyscan_t scanner}   // [한국어] yylex 호출 시 스캐너 전달
%lex-param {ptxinfo_data* ptxinfo} // [한국어] yylex 호출 시 ptxinfo_data 전달

%union {
  int    int_value;            // [한국어] 정수 리터럴
  char * string_value;         // [한국어] IDENTIFIER, FUNCTION, VARIABLE, WARNING 문자열
}

%token <int_value> INT_OPERAND // [한국어] 정수 상수
%token HEADER                  // [한국어] "ptxas"
%token INFO                    // [한국어] "info"
%token FUNC                    // [한국어] "Compiling entry function"
%token USED                    // [한국어] "Used"
%token REGS                    // [한국어] "registers"
%token BYTES                   // [한국어] "bytes"
%token LMEM                    // [한국어] "lmem"
%token SMEM                    // [한국어] "smem"
%token CMEM                    // [한국어] "cmem"
%token GMEM                    // [한국어] "gmem"
%token <string_value> IDENTIFIER // [한국어] 함수/변수 이름
%token PLUS                    // [한국어] +
%token COMMA                   // [한국어] ,
%token LEFT_SQUARE_BRACKET     // [한국어] [
%token RIGHT_SQUARE_BRACKET    // [한국어] ]
%token COLON                   // [한국어] :
%token SEMICOLON               // [한국어] ;
%token QUOTE                   // [한국어] '
%token LINE                    // [한국어] "line"
%token <string_value> WARNING  // [한국어] warning 메시지
%token FOR                     // [한국어] "for"
%token TEXTURES                // [한국어] "textures"
%token DUPLICATE               // [한국어] "Duplicate definition of"
%token <string_value> FUNCTION // [한국어] "function"
%token <string_value> VARIABLE // [한국어] "variable"
%token FATAL                   // [한국어] "fatal"

%{
	#include <stdlib.h>
	#include <string.h>
	
	static unsigned g_declared;  // [한국어] 선언된 메모리 크기 (lmem/smem)
	static unsigned g_system;    // [한국어] 시스템 추가 메모리 크기 (lmem/smem)
	int ptxinfo_lex(YYSTYPE * yylval_param, yyscan_t yyscanner, ptxinfo_data* ptxinfo);
	void yyerror(yyscan_t yyscanner, ptxinfo_data* ptxinfo, const char* msg);
	void ptxinfo_function(const char *fname );
	void ptxinfo_regs( unsigned nregs );
	void ptxinfo_lmem( unsigned declared, unsigned system );
	void ptxinfo_gmem( unsigned declared, unsigned system );
	void ptxinfo_smem( unsigned declared, unsigned system );
	void ptxinfo_cmem( unsigned nbytes, unsigned bank );
	void ptxinfo_linenum( unsigned );
	void ptxinfo_dup_type( const char* );
%}

%%

input:	/* empty */             // [한국어] 빈 입력 허용
	| input line              // [한국어] ptxas info 라인 반복
	;

line: 	HEADER INFO COLON line_info
	// [한국어] "ptxas info : ..." 형태의 정보 라인
	| HEADER IDENTIFIER COMMA LINE INT_OPERAND SEMICOLON WARNING
	// [한국어] warning이 포함된 헤더 라인
	| HEADER WARNING { printf("GPGPU-Sim: ptxas %s\n", $2); }
	// [한국어] 경고 메시지 출력
	| HEADER IDENTIFIER COMMA LINE INT_OPERAND SEMICOLON DUPLICATE duplicate { ptxinfo_linenum($5); }
	// [한국어] 중복 정의 오류 및 라인 번호 기록
	| HEADER FATAL
	// [한국어] fatal 오류 헤더
	;

line_info: function_name
	| function_info { ptxinfo->ptxinfo_addinfo(); }
	// [한국어] function_info 파싱 후 ptxinfo_addinfo()로 function_info에 반영
	| gmem_info
	;

function_name:	FUNC QUOTE IDENTIFIER QUOTE { ptxinfo_function($3); }
	// [한국어] "Compiling entry function 'kernel_name'"
	|  FUNC QUOTE IDENTIFIER QUOTE FOR QUOTE IDENTIFIER QUOTE { ptxinfo_function($3); }
	// [한국어] "Compiling entry function 'kernel_name' for 'sm_xx'"
	;
		
function_info: info
	| function_info COMMA info
	;

gmem_info: INT_OPERAND BYTES GMEM
	// [한국어] "N bytes gmem" 형태의 전역 메모리 사용량
	;

info: 	  USED INT_OPERAND REGS { ptxinfo_regs($2); }
	// [한국어] "Used N registers" → 레지스터 수 기록
	| tuple LMEM { ptxinfo_lmem(g_declared,g_system); }
	// [한국어] "declared+system bytes lmem" → 로컬 메모리
	| tuple SMEM { ptxinfo_smem(g_declared,g_system); }
	// [한국어] "declared+system bytes smem" → 공유 메모리
	| INT_OPERAND BYTES CMEM LEFT_SQUARE_BRACKET INT_OPERAND RIGHT_SQUARE_BRACKET { ptxinfo_cmem($1,$5); }
	// [한국어] "N bytes cmem[bank]" → 상수 메모리 뱅크별 크기
	| INT_OPERAND BYTES GMEM { ptxinfo_gmem($1,0); }
	// [한국어] "N bytes gmem" → 전역 메모리
	| INT_OPERAND BYTES LMEM { ptxinfo_lmem($1,0); }
	// [한국어] "N bytes lmem" → 로컬 메모리 (system 없음)
	| INT_OPERAND BYTES SMEM { ptxinfo_smem($1,0); }
	// [한국어] "N bytes smem" → 공유 메모리 (system 없음)
	| INT_OPERAND BYTES CMEM { ptxinfo_cmem($1,0); }
	// [한국어] "N bytes cmem" → 상수 메모리 (bank 0)
	| INT_OPERAND REGS { ptxinfo_regs($1); }
	// [한국어] "N registers" 축약 형태
	| INT_OPERAND TEXTURES {}
	// [한국어] "N textures"는 현재 무시
	;

tuple: INT_OPERAND PLUS INT_OPERAND BYTES { g_declared=$1; g_system=$3; }
	// [한국어] "declared + system bytes"에서 declared와 system 값을 분리 저장
	;

duplicate:	FUNCTION QUOTE IDENTIFIER QUOTE { ptxinfo_dup_type($1); }
	// [한국어] 중복 정의 대상이 함수
	| VARIABLE QUOTE IDENTIFIER QUOTE { ptxinfo_dup_type($1); }
	// [한국어] 중복 정의 대상이 변수
	;

%%

