/*
 * [한국어 설명] ptxas 리소스 사용 정보 파싱용 Lexer (ptxinfo.l)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim이 NVIDIA ptxas(NVCC의 PTX 어셈블러)가 출력하는
 * "ptxas info" 메시지에서 커널별 리소스 사용량(레지스터 수, 공유/로컬/상수/전역
 * 메모리 크기, 텍스처 수 등)을 추출하기 위한 Flex(Lex) 어휘 분석기를 정의한다.
 * ptxas가 생성하는 텍스트 로그를 토큰화하여 ptxinfo.y(Bison 파서)가 정형화된
 * 리소스 정보로 변환할 수 있도록 한다. 이 정보는 function_info에 반영되어
 * 스케줄러가 CTA 점유율(occupancy)을 계산하거나 레지스터 파일/공유 메모리
 * 할당량을 결정하는 데 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CUDA App → libcuda 인터셉트 → gpgpusim_entrypoint.cc
 *   → ptxas 실행 또는 ptxas 출력 캡처 → [이 파일: ptxinfo_lex()] → ptxinfo.y
 *       → ptxinfo_* 콜백 (ptx_loader.cc 등) → function_info 리소스 설정
 *           → gpgpu-sim/ 타이밍 시뮬레이션 (점유율, 레지스터/공유메모리 할당)
 * 실행 컨텍스트: 호스트 유저스페이스, 커널 실행 전 초기화 단계.
 *
 * === 타 모듈과의 연결 ===
 * - 의존하는 모듈: ptxinfo.tab.h (Bison 생성 토큰 번호), ptx_loader.h
 *   (ptxinfo_data 구조체), gpgpu_context.h (g_ptxinfo_error_detected 등)
 * - 이 파일에 의존하는 모듈: ptxinfo.y (ptxinfo_parse()가 ptxinfo_lex 호출),
 *   ptx_loader.cc (ptxinfo_function, ptxinfo_regs, ptxinfo_lmem 등 콜백 구현)
 * - Flex/Bison 생성 산출물: ptxinfo_lexer.c 등이 빌드 시 생성되며
 *   ptxinfo_lex(), ptxinfo_error() 구현이 들어간다.
 *
 * === 주요 함수/구조체 요약 ===
 * - ptxinfo_lex(): Flex가 생성하는 토큰화 루프 진입점.
 * - ptxinfo_error(): ptxinfo 파싱 중 문법 오류 발생 시 파일/라인/컬럼 정보를
 *   출력하고 g_ptxinfo_error_detected를 설정한 뒤 exit(43)으로 종료.
 * - %option prefix="ptxinfo_": ptx.l의 ptx_ 심볼들과 충돌하지 않게 접두사 분리.
 * - ptxinfo_data: ptxas 정보 파싱 상태를 담는 구조체 (linebuf, col, gpgpu_ctx 등).
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


%option noyywrap
%option yylineno
%option prefix="ptxinfo_"

%option bison-bridge
%option reentrant

%{
#include "ptx_loader.h"         // [한국어] ptxinfo_data 구조체 및 로더 인터페이스
#include "ptxinfo.tab.h"        // [한국어] Bison 생성 토큰 번호 정의
#include <string.h>             // [한국어] strdup, strlen 등
#include "../../libcuda/gpgpu_context.h" // [한국어] 시뮬레이터 전역 컨텍스트

#define LINEBUF_SIZE 1024       // [한국어] 현재 파싱 중인 ptxas 출력 라인 버퍼 크기
#define TC if( (yylineno == 1) && (ptxinfo->col + strlen(yytext) < LINEBUF_SIZE) ) { \
		strncpy(ptxinfo->linebuf+ptxinfo->col,yytext,strlen(yytext)); \
	   } \
	   ptxinfo->col+=strlen(yytext);
                            // [한국어] TC(Token Column): linebuf에 토큰 텍스트 누적(첫 줄 한정) 및 컬럼 증가
#define YY_DECL int ptxinfo_lex \
	       (YYSTYPE * yylval_param , yyscan_t yyscanner, ptxinfo_data* ptxinfo)
                            // [한국어] ptxinfo_lex 함수 선언: Bison-bridge + reentrant + ptxinfo_data 인자
%}

%%
"ptxas info    : Function properties for"[^\n]*\n[^\n]*
                            // [한국어] "Function properties for ..." 헤더 라인 전체 무시
"warning"[^\n]* TC; return WARNING; // [한국어] ptxas warning 메시지
"ptxas"	TC; 	return HEADER;  // [한국어] ptxas 출력 헤더 토큰
"info"	TC; 	return INFO;    // [한국어] info 레이블 토큰
"Compiling entry function"	TC; return FUNC; // [한국어] "Compiling entry function '...'"
"Used"		TC; return USED;   // [한국어] "Used N registers"
"registers"	TC; return REGS;   // [한국어] 레지스터 단위
"bytes"		TC; return BYTES;  // [한국어] 바이트 단위
"lmem"		TC; return LMEM;   // [한국어] local memory
"smem"		TC; return SMEM;   // [한국어] shared memory
"cmem"		TC; return CMEM;   // [한국어] constant memory
"gmem"		TC; return GMEM;   // [한국어] global memory
"line"		TC; return LINE;   // [한국어] 소스 라인 번호
"for"		TC; return FOR;    // [한국어] "for function..."
"textures"	TC; return TEXTURES; // [한국어] 텍스처 개수
"error   : Duplicate definition of"	TC; return DUPLICATE; // [한국어] 중복 정의 오류
"function"	TC; yylval->string_value = strdup(yytext); return FUNCTION; // [한국어] 중복 정의 대상이 함수일 때
"variable"	TC; yylval->string_value = strdup(yytext); return VARIABLE; // [한국어] 중복 정의 대상이 변수일 때
"fatal   : Ptx assembly aborted due to errors"	TC; return FATAL; // [한국어] ptxas fatal 오류

[_A-Za-z$%][_0-9A-Za-z$]*  TC; yylval->string_value = strdup(yytext); return IDENTIFIER; // [한국어] 함수/변수 이름 식별자
[-]{0,1}[0-9]+	 TC; yylval->int_value =  atoi(yytext); return INT_OPERAND; // [한국어] 부호 있는 정수 리터럴

"+"	TC; return PLUS;        // [한국어] 더하기 기호
","     TC; return COMMA;     // [한국어] 쉼표
"["     TC; return LEFT_SQUARE_BRACKET;  // [한국어] 왼쪽 대괄호
"]"	TC; return RIGHT_SQUARE_BRACKET; // [한국어] 오른쪽 대괄호
":"	TC; return COLON;       // [한국어] 콜론
";"	TC; return SEMICOLON;   // [한국어] 세미콜론
"'"	TC; return QUOTE;       // [한국어] 작은따옴표
" " TC;                    // [한국어] 공백 무시
"\t" TC;                   // [한국어] 탭 무시

\n.*  ptxinfo->col=0; strncpy(ptxinfo->linebuf, yytext + 1, 1024); yyless( 1 );
                           // [한국어] 줄바꿈 시 컬럼 초기화 및 다음 줄을 linebuf에 복사

%%

int ptxinfo_error(yyscan_t yyscanner, ptxinfo_data* ptxinfo, const char* msg)
{
    struct yyguts_t * yyg = (struct yyguts_t *)yyscanner;
	int i;
	ptxinfo->gpgpu_ctx->func_sim->g_ptxinfo_error_detected = 1;
	fflush(stdout);
	printf("GPGPU-Sim: ERROR while parsing output of ptxas (used to capture resource usage information)\n");
	if( msg != NULL )
		printf("GPGPU-Sim:     %s (%s:%u) Syntax error:\n\n", ptxinfo->gpgpu_ctx->g_filename, ptxinfo->g_ptxinfo_filename, yylineno );
	printf("   %s\n", ptxinfo->linebuf );
	printf("   ");
	for( i=0; i < ptxinfo->col-1; i++ ) {
		if( ptxinfo->linebuf[i] == '\t' ) printf("\t");
		else printf(" ");
	}
			
	printf("^\n\n");
	fflush(stdout);
	exit(43);
	return 0;
}
