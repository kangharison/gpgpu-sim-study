/*
 * [한국어 설명] 원본 PTX 헤더 Lex/Flex 어휘 분석기 (header.l)
 *
 * === 파일의 역할 ===
 * cuobjdump_to_ptxplus가 원본 PTX 파일을 파싱할 때, Flex가 생성하는
 * 어휘 분석기(lexer) 정의 파일이다. GPGPU-Sim 메인 PTX 파서(ptx.l)와
 * 달리, 이 lexer는 커널 본문({...})과 .file/.extern 등 대부분을
 * 무시(discard)하고, .version, .target, .entry 선언과 그 파라미터,
 * .tex 선언 등 **헤더 부분만** 토큰화한다. 생성된 header_parser.hh는
 * Bison(header.y)과 공유된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   cuobjdump_to_ptxplus.cc
 *     → header_parse() (header.y가 생성)
 *         → header.l (이 파일)이 먼저 토큰을 생성
 *             → header.y가 문법 규칙에 따라 g_headerList 구축
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - header_parser.hh : Bison이 생성한 토큰 정의 및 yylval 공용체
 * 이 파일에 의존하는 모듈:
 *   - header.y : 이 lexer가 반환하는 토큰들을 문법 규칙에서 사용
 *   - 빌드 시스템 : flex header.l → header.yy.cc 생성
 *
 * === 주요 함수/구조체 요약 ===
 * Flex 옵션:
 *   noyywrap    : 입력이 끝나도 yywrap 호출하지 않음
 *   yylineno    : 줄 번호 자동 추적
 *   prefix="header_" : 생성된 함수/변수명에 header_ 접두사 부여
 *   nounput      : unput 함수 생성 안 함
 * START 조건: 없음(단일 INITIAL 상태)
 * 주요 토큰:
 *   DOTVERSION/DOTTARGET/DOTENTRY : 컴파일러 지시어
 *   DOTPARAM                      : 파라미터 공간
 *   DOTU64/DOTU32/DOTU16/DOTB32/DOTF32 : scalar 타입
 *   IDENTIFER/DECLITERAL          : 식별자와 10진수 리터럴
 *   LEFTPAREN/RIGHTPAREN          : 괄호
 */

/*
// Copyright (c) 2009-2011, Jimmy Kwa,
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
*/

%option noyywrap
%option yylineno
%option prefix="header_"
%option nounput
%{
#include <iostream>
#include "header_parser.hh"
#include <string.h>

/* not sure if this is needed*/
#define	YY_NEVER_INTERACTIVE	1

/* [한국어] header.l 전용 오류 처리 함수 선언. */
void header_error(const char*);
/* [한국어] 출력 함수(output)는 cuobjdump_to_ptxplus.cc에 정의되어 있다. */
extern void output(const char * text);
%}


/*Regular expresions go here*/
alpha		[a-zA-Z]
numeric		[0-9]
hexalpha	[A-Fa-f]
alphanum	({alpha}|{numeric})
hexdigit	({numeric}|{hexalpha})
whitespace	[ \t]
newline		[\n]
anycharbutnewline [^\n]
anychar		({newline}|{anycharbutnewline})
anycharbutrightbrace [^"}"]

identifier	({alpha}({alphanum}|["_$"])*)|([_$%]({alphanum}|["_$"])+)
hexliteral	0[xX]{hexdigit}+U?
octliteral	0[0-7]+U?
binliteral	0[bB][01]+U?
decliteral	([1-9][0-9]*U?)|0U?

%%
	/*Translation rules*/

	/*Compiler directives*/
\.version	return DOTVERSION;
\.target	return DOTTARGET;
\.entry	return DOTENTRY;

	/*State spaces*/
\.param	return DOTPARAM;

	/*operand types*/
\.u64	return DOTU64;
\.u32	return DOTU32;
\.u16	return DOTU16;
\.b32	return DOTB32;
\.f32	return DOTF32;


	/*identifier + literals*/
{identifier}	header_lval.string_value = strdup(yytext); return IDENTIFER;
{decliteral}	header_lval.string_value = strdup(yytext); return DECLITERAL; //change to int later?

	/*braces and parenthesis*/
\(	return LEFTPAREN;
\)	return RIGHTPAREN;

	/*discard newline*/
{newline}+


	/*discard commas and periods*/
\,
\.
	/*discard whitespace*/
{whitespace}+
	/*discard comments*/
"//"{anycharbutnewline}*{newline}

	/*discard other header stuff*/
\.file{anycharbutnewline}*{newline}
\.extern{anycharbutnewline}*{newline}

	/*discard kernel body*/
\{{anycharbutrightbrace}*\}

	/*reached end of file*/
<<EOF>>	return 0;

	/*None of the rules matched. Throw error*/
.			header_error("Invalid token");

%%

/*
 * [한국어]
 * header_error - lexer가 인식할 수 없는 토큰을 만났을 때 호출
 *
 * @message: 오류 메시지
 *
 * 현재 입력 토큰과 줄 번호를 output()를 통해 출력 파일에 기록한다.
 */
void header_error(const char* message)
{
	output("   "); output(message); output(" near \""); output(header_text); output("\"");
	output(" on line "); 
	char line[5]; sprintf(line, "%i", header_lineno); output(line);
	output("\n");
}
