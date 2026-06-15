/*
 * [한국어 설명] 원본 PTX 헤더 Bison/Yacc 문법 파서 (header.y)
 *
 * === 파일의 역할 ===
 * cuobjdump_to_ptxplus가 원본 PTX 파일의 헤더 부분을 파싱할 때,
 * Bison이 생성하는 문법 분석기 정의 파일이다. header.l이 반환한
 * 토큰(DOTVERSION, DOTTARGET, DOTENTRY, DOTPARAM, 타입 토큰 등)을
 * 받아서 .version, .target, .entry/.func 선언과 파라미터 목록을
 * g_headerList(cuobjdumpInstList)에 저장한다. PTX 전체 문법이 아닌
 * 헤더에 해당하는 최소한의 문법만 정의되어 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   cuobjdump_to_ptxplus.cc
 *     → header_parse() (이 파일이 생성)
 *         → header.l (토큰 생성)
 *             → ptx_parser.h의 액션 함수 호출
 *                 → g_headerList 구축
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - cuobjdumpInstList.h : g_headerList 타입
 *   - ptx_parser.h        : add_version_info, func_header 등 액션 함수
 *   - header.l            : 토큰 정의
 * 이 파일에 의존하는 모듈:
 *   - 빌드 시스템 : bison header.y → header_parser.hh/header.tab.cc 생성
 *
 * === 주요 함수/구조체 요약 ===
 * program       : statementList
 * statementList : statement 반복
 * statement     : compilerDirective + 리터럴/식별자/파라미터 목록
 * compilerDirective : DOTVERSION | DOTTARGET | DOTENTRY
 * identifierList: IDENTIFER 반복
 * parameterList : parameter 반복
 * parameter     : stateSpace opTypes IDENTIFER
 * stateSpace    : DOTPARAM
 * opTypes       : DOTU64 | DOTU32 | DOTU16 | DOTB32 | DOTF32
 * literal       : DECLITERAL
 */

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

%{
#include <iostream>
#include "cuobjdumpInstList.h"

int yylex(void);
void yyerror(const char*);

/* [한국어] 원본 PTX 헤더 정보를 저장하는 전역 리스트. */
extern cuobjdumpInstList *g_headerList;
/* [한국어] 출력 함수는 cuobjdump_to_ptxplus.cc에 정의되어 있다. */
extern void output(const char * text);
%}


%union {
  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  void * ptr_value;
}

%token DOTVERSION
%token DOTTARGET
%token DOTENTRY

%token DOTPARAM

%token DOTU64
%token DOTU32
%token DOTU16
%token DOTB32
%token DOTF32

%token <string_value> IDENTIFER

	/*change these 4 to int later?*/
%token <string_value> DECLITERAL

%token LEFTPAREN
%token RIGHTPAREN


%%

	/*translation rules*/
program		: statementList			{ output("No parsing errors\n");  }
		;

statementList	: statementList statement	{ output("\n"); }
		| statement			{ output("\n"); }
		;

statement	: compilerDirective literal literal	{}
		| compilerDirective identifierList		{}
		| compilerDirective identifierList LEFTPAREN parameterList RIGHTPAREN	{}
		;

/* [한국어] compilerDirective 규칙: .version/.target/.entry 토큰에 따라
 * g_headerList에 새 cuobjdumpInst를 추가하고 기본 문자열을 출력한다. */
compilerDirective	: DOTVERSION	{ output(".version"); cuobjdumpInst *instEntry = new cuobjdumpInst(); instEntry->setBase(".version"); g_headerList->add(instEntry);}
			| DOTTARGET	{ output(".target"); cuobjdumpInst *instEntry = new cuobjdumpInst(); instEntry->setBase(".target"); g_headerList->add(instEntry);}
			| DOTENTRY	{ output(".entry"); cuobjdumpInst *instEntry = new cuobjdumpInst(); instEntry->setBase(".entry"); g_headerList->add(instEntry);}
			;

identifierList	: identifierList IDENTIFER	{ output(" "); output($2); g_headerList->getListEnd().addOperand($2); }
		| IDENTIFER			{ output(" "); output($1); g_headerList->getListEnd().addOperand($1); }
		;

parameterList	: parameterList parameter
		| parameter
		;

parameter	: stateSpace opTypes IDENTIFER	{ output(" "); output($3); g_headerList->getListEnd().addOperand($3);}
			;

/* [한국어] stateSpace 규칙: .param 공간에 대해 g_headerList에 .param inst 추가. */
stateSpace	: DOTPARAM	{ output("\n.param"); cuobjdumpInst *instEntry = new cuobjdumpInst(); instEntry->setBase(".param"); g_headerList->add(instEntry); }
			;

/* [한국어] opTypes 규칙: 각 타입 토큰에 대해 g_headerList 마지막 inst에
 * 타입 수정자를 추가한다. */
opTypes		: DOTU64		{ output(".u64"); g_headerList->getListEnd().addTypeModifier(".u64");}
		| DOTU32		{ output(".u32"); g_headerList->getListEnd().addTypeModifier(".u32");}
		| DOTU16		{ output(".u16"); g_headerList->getListEnd().addTypeModifier(".u16");}
		| DOTB32		{ output(".b32"); g_headerList->getListEnd().addTypeModifier(".b32");}
		| DOTF32		{ output(".f32"); g_headerList->getListEnd().addTypeModifier(".f32");}
		;

literal		: DECLITERAL	{ output(" "); output($1); g_headerList->getListEnd().addOperand($1); }
			;

%%

/*support c++ functions go here*/
