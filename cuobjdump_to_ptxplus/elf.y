/*
 * [한국어 설명] cuobjdump ELF 출력 Bison/Yacc 문법 파서 (elf.y)
 *
 * === 파일의 역할 ===
 * cuobjdump가 출력한 ELF 디스어셈블리에서 상수 메모리(.nv.constant0,
 * .nv.constant1), 로컬 메모리(.nv.local), .rel.nv.constant14 재배치
 * 정보, 그리고 .symtab의 전역/상수 심볼 정보를 추출하여 g_instList에
 * 저장하는 Bison 문법 분석기 정의 파일이다. 생성된 토큰은 elf.l이
 * 제공한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   cuobjdump_to_ptxplus.cc
 *     → elf_parse() (이 파일이 생성)
 *         → elf.l (토큰 생성)
 *             → 이 파일의 규칙에 따라 g_instList의 메모리 정보 채움
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - cuobjdumpInstList.h : g_instList 타입 및 메모리 추가 API
 *   - elf.l               : C1BEGIN, CMEMVAL, NUMBER 등 토큰 제공
 * 이 파일에 의존하는 모듈:
 *   - 빌드 시스템 : bison elf.y → elf_parser.hh/elf.tab.cc 생성
 *
 * === 주요 함수/구조체 요약 ===
 * elffile        : symtab + program
 * symtab         : STBEGIN STHEADER stcontent
 * stcontent/stline: SYMTAB 항목(7개 필드) 처리
 * program        : cmemsection | localmemsection | cmem14section
 * localmemsection: LOCALMEM 처리
 * cmemsection    : C1BEGIN cmemvals | C0BEGIN cmemvals
 * cmemvals       : CMEMVAL SPACE2 반복
 * cmem14section  : C14BEGIN c14content (전역 메모리 오프셋 업데이트)
 */

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

	/*Yacc file for elf files output by cuobjdump*/
%{
#include <stdio.h>
#include "cuobjdumpInstList.h"

int yylex(void);
void yyerror(const char*);
//void addEntryConstMemory(int, int);
//void setConstMemoryType(const char*);
//void addConstMemoryValue(const char*);
/* [한국어] SASS/ELF 변환 IR 컨테이너. */
extern cuobjdumpInstList *g_instList;
/* [한국어] 상수 메모리 뱅크 인덱스 카운터. */
int cmemcount=1;
/* [한국어] 로컬 메모리 인덱스 카운터. */
int lmemcount=1;
/* [한국어] 마지막으로 파싱한 상수 메모리 종류(false=constant0, true=constant1). */
bool lastcmem = false;// false = constrant0, true = constant1
%}
%union {
	char* string_value;
}
%token <string_value> C1BEGIN C14BEGIN CMEMVAL SPACE2 C0BEGIN STBEGIN STHEADER
%token <string_value> NUMBER HEXNUMBER IDENTIFIER LOCALMEM

%%

/* [한국어] 전체 ELF 파일: SYMTAB 다음 상수/로컬 메모리 섹션들. */
elffile	:	symtab program
		;

/* [한국어] SYMTAB: 시작 토큰, 헤더, 내용. */
symtab	:	STBEGIN STHEADER stcontent
		;

stcontent	:	stcontent stline
			|	stline;

/* [한국어]
 * SYMTAB 항목 한 줄 처리.
 * NUMBER가 7개, IDENTIFIER가 1개 있는 형태.
 *   - $7(이름)이 ".nv.global"로 시작하면 전역 변수 섹션 인덱스 기록
 *   - $4가 "11"(STB_GLOBAL 또는 관련 섹션 타입)이면:
 *       - $6(섹션 인덱스)가 전역 변수 섹스와 같으면 전역 메모리 심볼 추가
 *       - 아니면 상수 메모리 포인터 추가
 */
stline	:	NUMBER NUMBER NUMBER NUMBER NUMBER NUMBER IDENTIFIER {
					if(strncmp( $7, ".nv.global", 10)==0) {
						g_instList->setglobalVarShndx($6);
					}
					if (strcmp($4, "11")==0) {
						if(atoi($6) == g_instList->getglobalVarShndx()) {
							g_instList->addGlobalMemoryID($3, $7);
						} else {
							g_instList->addConstMemoryPtr($2, $3, $7);
						}
					}
				}
		|	NUMBER NUMBER NUMBER NUMBER NUMBER NUMBER {}
			;

/* [한국어] program: 상수 메모리/로컬 메모리/cmem14 섹션들의 반복. */
program	:	program cmemsection
		|	program localmemsection
		|	program cmem14section
		|	{
					g_instList->setKernelCount(cmemcount-1);
				};

/* [한국어] 로컬 메모리 섹션: 엔트리별 로컬 메모리 슬롯 추가 및 맵 갱신. */
localmemsection	:	LOCALMEM {
							g_instList->addEntryLocalMemory(0, lmemcount);
							g_instList->setLocalMemoryMap($1, lmemcount);
							lmemcount++;
						};

/* [한국어] 상수 메모리 섹션: constant1(엔트리별) 또는 constant0(전역). */
cmemsection	:	C1BEGIN {
						g_instList->addEntryConstMemory2($1);
						g_instList->setConstMemoryType2(".u32");
						cmemcount++;
						lastcmem = true;
					} cmemvals
			|	C0BEGIN {
						g_instList->addConstMemory(0);
						g_instList->setConstMemoryType(".u32");
						lastcmem = false;
					} cmemvals;

/* [한국어] 상수 메모리 값들: CMEMVAL SPACE2 반복. lastcmem에 따라
 * constant1 또는 constant0 리스트에 값 추가. */
cmemvals	:	cmemvals CMEMVAL SPACE2 {
						if (lastcmem)
							g_instList->addConstMemoryValue2($3);
						else
							g_instList->addConstMemoryValue($3);
					}
			|	;

/* [한국어] constant14 재배치 섹션: 전역 메모리 오프셋 업데이트에 사용. */
cmem14section	:	C14BEGIN c14content
				|	C14BEGIN
					;

c14content	:	c14content c14line
			|	c14line;

/* [한국어] cmem14 한 줄: NUMBER(오프셋) IDENTIFIER(이름) IDENTIFIER(재배치 타입)
 * 형태에서 첫 번째 NUMBER를 이름으로 전역 메모리 오프셋 업데이트. */
c14line	:	NUMBER IDENTIFIER IDENTIFIER {
					g_instList->updateGlobalMemoryID($1, $2);
				};
%%
