/*
 * [한국어 설명] cuobjdump 출력용 Bison/Yacc 문법 파서 (cuobjdump.y)
 *
 * === 파일의 역할 ===
 * 이 파일은 NVIDIA cuobjdump 도구가 fat binary를 텍스트로 덤프한 출력을
 * Bison(Yacc) 문법으로 파싱하여 PTX/ELF 섹션 객체(cuobjdumpSection)를 생성한다.
 * cuobjdump.l(Flex)가 반환한 토큰(PTXHEADER, ELFHEADER, H_ARCH, IDENTIFIER 등)을
 * 받아 "Fatbin ptx/elf code:" 헤더, 메타데이터(arch, code version, producer, host,
 * compile_size), identifier, 그리고 PTX/ELF/SASS 본문을 구조화된 객체로 변환한다.
 * 파싱 결과는 cuobjdumpSectionList에 누적되며, 이후 cuda-sim/ 기능 시뮬레이션이
 * 해당 리스트에서 시뮬레이션 대상 SM 버전의 PTX 파일을 선택하여 로드한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CUDA 애플리케이션 초기화 흐름의 libcuda/ 레이어에서 한 번 실행된다.
 *   CUDA 바이너리
 *     → cuobjdumpParseBinary() (libcuda/cuda_runtime_api.cc / cuobjdump.cc)
 *       → cuobjdump 서브프로세스 실행 → 텍스트 출력
 *         → [cuobjdump.l Flex 스캐너] → [cuobjdump.y Bison 파서]
 *           → cuobjdumpPTXSection / cuobjdumpELFSection 객체 생성
 *             → gpgpu_ptx_sim_load_ptx_from_filename() → cuda-sim PTX IR
 *               → gpgpu_sim 타이밍 시뮬레이션
 * 실행 컨텍스트: CPU 호스트, CUDA 런타임 초기화 단계에서 단일 스레드로 호출.
 * %define api.pure full 및 %parse-param/%lex-param를 사용하여 재진입 가능한
 * pure parser를 생성하므로 전역 상태 오염 없이 파싱할 수 있다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈/헤더:
 *   - cuobjdump.h: cuobjdumpSection 파생 클래스, cuobjdump_parser 상태 구조체,
 *     yyscan_t 타입 정의
 *   - cuobjdump.l: PTXHEADER, ELFHEADER, H_ARCH, IDENTIFIER 등 토큰 생성
 *   - cuobjdump.cc: Bison 액션에서 호출하는 addCuobjdumpSection(),
 *     setCuobjdumparch(), setCuobjdumpidentifier(), setCuobjdumpptxfilename(),
 *     setCuobjdumpelffilename(), setCuobjdumpsassfilename() 구현
 * 이 파일이 생성하는 데이터:
 *   - std::list<cuobjdumpSection*>& cuobjdumpSectionList: 파싱된 모든 섹션 저장
 *   - struct cuobjdump_parser* parser: 임시 파일 포인터(ptxfile/elffile/sassfile)
 *     및 serial 카운터(ptxserial/elfserial) 공유
 *
 * === 주요 함수/구조체 요약 ===
 * - program: 최상위 문법 규칙. 빈 줄을 걸러낸 뒤 하나 이상의 section을 파싱.
 * - section: PTXHEADER/ELFHEADER를 만나면 cuobjdumpSection 객체를 생성하고
 *   임시 파일을 열어 본문을 기록한다. PTX는 .ptx 파일, ELF는 .elf와 .sass 파일로
 *   각각 추출한다.
 * - headerinfo: "================" 구분선 이후의 arch, code version, producer,
 *   host, compile_size 필드를 파싱하고 마지막에 setCuobjdumparch()로 SM 버전 설정.
 * - identifier: "identifier = <name>" 또는 기본값 "default"로 섹션 식별자 설정.
 * - compressedkeyword: "compressed" 키워드를 선택적으로 소비.
 * - ptxcode / elfcode / sasscode: 본문 라인들을 반복적으로 읽어 parser의
 *   ptxfile / elffile / sassfile에 기록.
 * - addCuobjdumpSection(): 0=PTX 섹션, 1=ELF 섹션 객체를 리스트 끝에 추가.
 * - setCuobjdumparch/identifier/*filename(): 리스트의 마지막 섹션 객체 필드 설정.
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

	/*Yacc file for output of cuobjdump*/
%{
/* [한국어] 표준 입출력 — printf() 디버그 출력용 */
#include <stdio.h>

/* [한국어]
 * yyscan_t를 void*로 typedef: cuobjdump.h가 Flex 헤더를 포함하기 전에
 * 재진입 스캐너 핸들 타입을 단순화하여 사용한다.
 */
typedef void * yyscan_t;
/* [한국어] cuobjdumpSection, cuobjdump_parser, yyscan_t 정의 */
#include "cuobjdump.h"

/* [한국어]
 * cuobjdump.cc에서 구현되는 외부 함수 선언.
 * Bison 액션이 파싱 중 생성한 섹션 리스트의 마지막 객체에 필드를 설정할 때
 * 이 함수들을 호출한다.
 */
extern void addCuobjdumpSection(int sectiontype, std::list<cuobjdumpSection*> &cuobjdumpSectionList);
void setCuobjdumparch(const char* arch, std::list<cuobjdumpSection*> &cuobjdumpSectionList);
void setCuobjdumpidentifier(const char* identifier, std::list<cuobjdumpSection*> &cuobjdumpSectionList);
void setCuobjdumpptxfilename(const char* filename, std::list<cuobjdumpSection*> &cuobjdumpSectionList);
void setCuobjdumpelffilename(const char* filename, std::list<cuobjdumpSection*> &cuobjdumpSectionList);
void setCuobjdumpsassfilename(const char* filename, std::list<cuobjdumpSection*> &cuobjdumpSectionList);
%}

/* [한국어] pure 재진입 파서 생성: yyparse가 전역 변수 대신 인자로 scanner/parser/리스트를 받음 */
%define api.pure full

/* [한국어] yyparse()의 추가 매개변수: 재진입 Flex 핸들, 파서 상태, 섹션 리스트 */
%parse-param {yyscan_t scanner}
%parse-param {struct cuobjdump_parser* parser}
%parse-param {std::list<cuobjdumpSection*> &cuobjdumpSectionList}

/* [한국어] yylex()의 추가 매개변수: Flex 스캐너가 yylex 호출 시 전달받을 인자들 */
%lex-param {yyscan_t scanner}
%lex-param {struct cuobjdump_parser* parser}
%lex-param {std::list<cuobjdumpSection*> &cuobjdumpSectionList}

/* [한국어]
 * 토큰 의미값(semantic value)의 공용체.
 * cuobjdump 출력에서 대부분의 값은 문자열이므로 char*만 사용한다.
 */
%union {
	char* string_value;
}
%{
/* [한국어] Flex가 생성한 cuobjdump_lex() 함수 선언 (Bison에 의해 yylex로 연결됨) */
int yylex(YYSTYPE * yylval_param, yyscan_t yyscanner, struct cuobjdump_parser* parser, std::list<cuobjdumpSection*> &cuobjdumpSectionList);
/* [한국어] Bison 파서 오류 처리 함수 선언 */
void yyerror(yyscan_t yyscanner, struct cuobjdump_parser* parser, std::list<cuobjdumpSection*> &cuobjdumpSectionList, const char* msg);
%}

/* [한국어] cuobjdump 헤더 메타데이터 토큰 — 모두 string_value 타입 의미값 사용 */
%token <string_value> H_SEPARATOR H_ARCH H_CODEVERSION H_PRODUCER H_HOST H_COMPILESIZE H_IDENTIFIER H_UNKNOWN H_COMPRESSED
/* [한국어] code version 값 "[N,M]" 토큰 */
%token <string_value> CODEVERSION
/* [한국어] 일반 문자열 토큰 (현재 문법에서는 직접 사용되지 않음) */
%token <string_value> STRING
/* [한국어] 섹션 식별자(커널/모듈명) 값 토큰 */
%token <string_value> FILENAME
/* [한국어] 10진 정수 토큰 (예: arch 값) */
%token <string_value> DECIMAL
/* [한국어] "Fatbin ptx code:" / "Fatbin elf code:" 헤더 토큰 */
%token <string_value> PTXHEADER ELFHEADER
/* [한국어] PTX 본문 한 줄 토큰 */
%token <string_value> PTXLINE
/* [한국어] ELF 본문 한 줄 토큰 */
%token <string_value> ELFLINE
/* [한국어] SASS 본문 한 줄 토큰 */
%token <string_value> SASSLINE
/* [한국어] 헤더의 일반 식별자(arch, producer, host, compile_size 값 등) 토큰 */
%token <string_value> IDENTIFIER
/* [한국어] 빈 줄용 개행 토큰 */
%token <string_value> NEWLINE

%%


/* [한국어]
 * program — cuobjdump 출력 전체의 최상위 문법 규칙.
 * 파서 시작 시 디버그 헤더를 출력하고, 앞부분의 빈 줄(emptylines)을 무시한 뒤
 * 하나 이상의 section(PTX 또는 ELF 섹션)을 반복적으로 파싱한다.
 */
program :	{printf("######### cuobjdump parser ########\n");}
			emptylines section
		|	program section;

/* [한국어]
 * emptylines — 시작 부분 또는 섹션 사이의 불필요한 빈 줄(개행)들을 0개 이상 소비.
 * NEWLINE 토큰을 반복적으로 받아들여 유연한 공백 처리를 제공한다.
 */
emptylines	:	emptylines NEWLINE
			|	;

/* [한국어]
 * section — PTX 섹션 또는 ELF 섹션 하나를 파싱.
 *   - PTXHEADER: cuobjdumpPTXSection(0)을 리스트에 추가하고,
 *     "_cuobjdump_N.ptx" 임시 파일을 쓰기 모드로 연 뒤
 *     headerinfo, compressedkeyword, identifier, ptxcode를 파싱.
 *     ptxcode가 끝나면 ptxfile을 닫는다.
 *   - ELFHEADER: cuobjdumpELFSection(1)을 리스트에 추가하고,
 *     "_cuobjdump_N.elf" 임시 파일을 열어 ELF 헤더/본문을 기록한 뒤 닫는다.
 *     이어서 "_cuobjdump_N.sass" 임시 파일을 열어 SASS 본문을 기록하고 닫는다.
 * ptxserial/elfserial은 parser 구조체에 있으며, 각 섹션마다 고유한 파일명 생성에 사용.
 */
section :	PTXHEADER {
				addCuobjdumpSection(0, cuobjdumpSectionList);
				snprintf(parser->filename, 1024, "_cuobjdump_%d.ptx", parser->ptxserial++);
				parser->ptxfile = fopen(parser->filename, "w");
				setCuobjdumpptxfilename(parser->filename, cuobjdumpSectionList);
			} headerinfo compressedkeyword identifier ptxcode {
				fclose(parser->ptxfile);
			}
		|	ELFHEADER {
				addCuobjdumpSection(1, cuobjdumpSectionList);
				snprintf(parser->filename, 1024, "_cuobjdump_%d.elf", parser->elfserial);
				parser->elffile = fopen(parser->filename, "w");
				setCuobjdumpelffilename(parser->filename, cuobjdumpSectionList);
			} headerinfo compressedkeyword identifier elfcode {
				fclose(parser->elffile);
				snprintf(parser->filename, 1024, "_cuobjdump_%d.sass", parser->elfserial++);
				parser->sassfile = fopen(parser->filename, "w");
				setCuobjdumpsassfilename(parser->filename, cuobjdumpSectionList);
			} sasscode { 
				fclose(parser->sassfile);
			};

/* [한국어]
 * headerinfo — PTX/ELF 섹션 공통 메타데이터 헤더 파싱.
 * "================" 구분선 이후
 *   arch = <IDENTIFIER>
 *   code version = <CODEVERSION>
 *   producer = <UNKNOWN|IDENTIFIER>
 *   host = <IDENTIFIER>
 *   compile_size = <IDENTIFIER>
 * 순서로 나타나며, 마지막 액션에서 $4(네 번째 심볼, arch 값)를
 * setCuobjdumparch()로 마지막 섹션에 저장한다.
 * producer가 "<unknown>"인 경우와 일반 식별자인 경우 두 가지 규칙이 있다.
 */
headerinfo :	H_SEPARATOR NEWLINE
				H_ARCH IDENTIFIER NEWLINE
				H_CODEVERSION CODEVERSION NEWLINE
				H_PRODUCER H_UNKNOWN NEWLINE
				H_HOST IDENTIFIER NEWLINE
				H_COMPILESIZE IDENTIFIER  {setCuobjdumparch($4, cuobjdumpSectionList);};
			|   H_SEPARATOR NEWLINE
				H_ARCH IDENTIFIER NEWLINE
				H_CODEVERSION CODEVERSION NEWLINE
				H_PRODUCER IDENTIFIER NEWLINE
				H_HOST IDENTIFIER NEWLINE
				H_COMPILESIZE IDENTIFIER {setCuobjdumparch($4, cuobjdumpSectionList);};

/* [한국어]
 * identifier — 섹션의 식별자(보통 커널/모듈명)를 파싱.
 *   - "identifier = <FILENAME>" 형식이 있으면 해당 값을 setCuobjdumpidentifier()로 저장.
 *   - identifier 필드가 없으면 "default"를 기본 식별자로 저장.
 */
identifier : H_IDENTIFIER FILENAME emptylines {setCuobjdumpidentifier($2, cuobjdumpSectionList);}
			 |	{setCuobjdumpidentifier("default", cuobjdumpSectionList);};

/* [한국어]
 * compressedkeyword — "compressed" 키워드를 선택적으로 소비.
 * 압축된 섹션 표시가 있을 수도 있고 없을 수도 있으므로 ε 규칙을 포함한다.
 */
compressedkeyword : H_COMPRESSED emptylines
                    | ;

/* [한국어]
 * ptxcode — PTX 본문 라인들을 0개 이상 반복 파싱.
 * 각 PTXLINE 토큰의 문자열($2)을 parser->ptxfile에 그대로 기록하여
 * 임시 .ptx 파일을 완성한다.
 */
ptxcode :	ptxcode PTXLINE {fprintf(parser->ptxfile, "%s", $2);}
		|	;

/* [한국어]
 * elfcode — ELF 본문(헤더) 라인들을 0개 이상 반복 파싱.
 * 각 ELFLINE 토큰의 문자열($2)을 parser->elffile에 기록한다.
 */
elfcode :	elfcode ELFLINE {fprintf(parser->elffile, "%s", $2);}
		|	;

/* [한국어]
 * sasscode — SASS 어셈블리 본문 라인들을 0개 이상 반복 파싱.
 * 각 SASSLINE 토큰의 문자열($2)을 parser->sassfile에 기록한다.
 */
sasscode :	sasscode SASSLINE {fprintf(parser->sassfile, "%s", $2);}
		 |	;


%%
