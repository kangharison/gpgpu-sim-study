/*
 * [한국어 설명] cuobjdump ELF 출력 Lex/Flex 어휘 분석기 (elf.l)
 *
 * === 파일의 역할 ===
 * cuobjdump가 출력한 ELF 섹션 정보에서 상수 메모리(.nv.constant0,
 * .nv.constant1, .nv.constant14), 로컬 메모리(.nv.local), 그리고
 * SYMTAB의 전역/상수 심볼 정보를 토큰으로 추출하는 Flex lexer 정의 파일이다.
 * START 조건(cmem, cmem14, symtab)을 사용하여 각 섹션에 맞는 규칙만
 * 활성화하고, 불필요한 라인은 무시한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   cuobjdump_to_ptxplus.cc
 *     → elf_parse() (elf.y가 생성)
 *         → elf.l (이 파일)이 토큰 생성
 *             → elf.y가 g_instList의 상수/전역/로컬 메모리 정보 채움
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - elf_parser.hh : Bison이 생성한 토큰 정의
 * 이 파일에 의존하는 모듈:
 *   - elf.y         : lexer 토큰을 문법 규칙에서 사용
 *   - 빌드 시스템   : flex elf.l → elf.yy.cc 생성
 *
 * === 주요 함수/구조체 요약 ===
 * Flex 옵션:
 *   noyywrap  : yywrap 사용 안 함
 *   yylineno  : 줄 번호 추적
 *   nounput   : unput 함수 비생성
 * START 조건:
 *   cmem      : .nv.constant0/.nv.constant1 섹션 내용 파싱
 *   cmem14    : .rel.nv.constant14 재배치 섹션 파싱
 *   symtab    : .symtab 섹션 헤더/내용 파싱
 * 주요 토큰:
 *   C1BEGIN/C0BEGIN/C14BEGIN : 상수 메모리 섹션 시작
 *   LOCALMEM                 : 로컬 메모리 섹션 시작
 *   STBEGIN/STHEADER         : SYMTAB 시작/헤더
 *   CMEMVAL                  : 0xXXXXXXXX 형태 상수 값
 *   NUMBER                   : 16진수 숫자
 *   IDENTIFIER               : 심볼/섹션 이름
 *   SPACE2                   : 상수 메모리 내 구분자("  ")
 */

/*
// Copyright (c) 2011-2012, Andrew Boktor
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
	/*Lex file for elf files output by cuobjdump*/
%{
#include <stdio.h>
#include <string.h>
#include "elf_parser.hh"

#define yylval elf_lval

/* [한국어] ELF lexer 오류 처리 함수 선언. */
void elf_error(const char*);
%}

%option noyywrap
%option yylineno
%option nounput

/* [한국어] START 조건: 상수 메모리(cmem), 상수14 재배치(cmem14), SYMTAB(symtab). */
%x cmem
%x symtab
%x cmem14

whitespace	[ \t]
newline		[\n]
notnewline	[^\n]
newlines	{newline}+
ident		[[:alnum:]_\.]
hexdigit	[[:digit:]abcdef]
%%

/* [한국어] .nv.constant1 섹션 시작. cmem 조건으로 전환하고 C1BEGIN 토큰 반환. */
<cmem,cmem14,INITIAL>\.nv\.constant1\.{notnewline}*{newline}	{
	BEGIN(cmem);
	yylval.string_value = strdup(yytext); 
	return C1BEGIN;
}

/* [한국어] .nv.local 섹션 시작. INITIAL로 되돌리고 LOCALMEM 토큰 반환. */
<cmem,cmem14,INITIAL>\.nv\.local{notnewline}*{newline}	{
	BEGIN(INITIAL);
	yylval.string_value = strdup(yytext); 
	return LOCALMEM;
}

/* [한국어] .nv.constant0 섹션 시작. cmem 조건으로 전환하고 C0BEGIN 토큰 반환. */
<cmem,cmem14,INITIAL>\.nv\.constant0{notnewline}*{newline}	{
	BEGIN(cmem);
	return C0BEGIN;
}

/* [한국어] .rel.nv.constant14 섹션 시작. cmem14 조건으로 전환하고 C14BEGIN 토큰 반환. */
<cmem,cmem14,INITIAL>\.section" "\.rel\.nv\.constant14{notnewline}*{newline}	{
	BEGIN(cmem14);
	return C14BEGIN;
}

/* [한국어] .symtab 섹션 시작. symtab 조건으로 전환하고 STBEGIN 토큰 반환. */
\.section" "\.symtab\tSYMTAB{newline}	{
	BEGIN(symtab);
	return STBEGIN;
}

/* [한국어] SYMTAB 이전의 다른 .section 라인들은 무시. */
\.section{notnewline}*{newline}	{
	//These are the lines before the symtab
}

/* [한국어] cmem/cmem14 상태에서 .으로 시작하는 다른 라인은 INITIAL로 되돌림. */
<cmem,cmem14,INITIAL>\.{notnewline}*{newline}	{
	BEGIN(INITIAL);
}

/* [한국어] 그 외 일반 라인은 무시(주석/구분선 등). */
{notnewline}*{newline} {
	//printf("Discarding line %s", yytext);
}

/* [한국어] cmem 상태에서 두 칸 공백(상수값 그룹 구분자)을 SPACE2 토큰으로 반환. */
<cmem>"  "{newline}?	{
	//printf("#SPACE#");
	return SPACE2;
}

/* [한국어] cmem 상태에서 0xXXXXXXXX 형태의 32비트 상수값을 CMEMVAL 토큰으로 반환. */
<cmem>0x[0-9a-fA-F]{8}	{
	//printf("CMEM VAL: %s \n", yytext);
	yylval.string_value = strdup(yytext); 
	return CMEMVAL;
}

/* [한국어] symtab 상태에서 SYMTAB 헤더 라인(7개 알파벳 필드)을 STHEADER로 반환. */
<symtab>{whitespace}*[[:alpha:]]+{whitespace}+[[:alpha:]]+{whitespace}+[[:alpha:]]+{whitespace}+[[:alpha:]]+{whitespace}+[[:alpha:]]+{whitespace}+[[:alpha:]]+{whitespace}+[[:alpha:]]+{whitespace}+{newline} {
	return STHEADER;
}

/* [한국어] symtab/cmem14 상태에서 공백과 개행은 무시(토큰으로 반환하지 않음). */
<symtab,cmem14>{whitespace}+ {}

<symtab,cmem14>{newline} {}

/* [한국어] symtab/cmem14 상태에서 16진수 숫자를 NUMBER 토큰으로 반환. */
<symtab,cmem14>{hexdigit}+	{
	yylval.string_value = strdup(yytext);
	return NUMBER;
}

/* [한국어] symtab/cmem14 상태에서 식별자를 IDENTIFIER 토큰으로 반환. */
<symtab,cmem14>{ident}+	{
	yylval.string_value = strdup(yytext);
	return IDENTIFIER;
}

/* [한국어] symtab 상태에서 빈 줄 두 개가 나오면 INITIAL로 되돌림(섹션 종료). */
<symtab>{newline}{whitespace}*{newline}{whitespace}* {
	BEGIN(INITIAL);
}
	/*Reached end of file*/
<<EOF>> return 0;

	/*No other rule matched. Throw an error*/
.	elf_error("Invalid token");

%%

/*
 * [한국어]
 * elf_error - lexer가 인식할 수 없는 토큰을 만났을 때 호출
 *
 * @message: 오류 메시지
 *
 * 메시지, 현재 토큰(yytext), 줄 번호를 stdout에 출력하고 abort()로 종료한다.
 */
void elf_error(const char* message)
{
	printf("   ");
	printf("%s",message);
	printf(" near \"");
	printf("%s",yytext);
	printf("\"");
	printf(" on line "); 
	char line[5];
	sprintf(line, "%i", yylineno);
	printf("%s",line);
	printf("\n");
	abort();
}
