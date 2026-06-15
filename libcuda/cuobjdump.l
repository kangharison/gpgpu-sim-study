/*
 * [한국어 설명] cuobjdump 출력용 Flex/Lex 어휘 분석기 (cuobjdump.l)
 *
 * === 파일의 역할 ===
 * 이 파일은 NVIDIA cuobjdump 도구가 fat binary를 텍스트로 덤프한 출력을
 * Flex(Lex)로 토큰화하는 어휘 분석기를 정의한다.
 * cuobjdump 출력은 "Fatbin ptx code:" / "Fatbin elf code:" 헤더와
 * 그 뒤에 이어지는 PTX/ELF/SASS 텍스트 본문, 그리고 각 섹션의 메타데이터
 * (arch, identifier, code version, producer, host, compile_size 등)로
 * 구성된다. 이 Lex 규칙들은 해당 출력을 PTXHEADER, ELFHEADER, PTXLINE,
 * ELFLINE, SASSLINE, H_ARCH, H_IDENTIFIER 등의 토큰으로 잘라내어
 * cuobjdump.y (Bison 파서)로 전달한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CUDA 애플리케이션 초기화 흐름의 일부로, libcuda/ 레이어에서 한 번 실행된다.
 *   CUDA 바이너리
 *     → cuobjdumpParseBinary() (libcuda/cuda_runtime_api.cc / cuobjdump.cc)
 *       → cuobjdump 도구 서브프로세스 실행
 *         → [cuobjdump.l Flex 스캐너] + [cuobjdump.y Bison 파서]
 *           → cuobjdumpPTXSection / cuobjdumpELFSection 객체 생성
 *             → cuda-sim/ PTX 로드 → 기능/타이밍 시뮬레이션
 * 실행 컨텍스트: CPU 호스트, CUDA 런타임 초기화 단계에서 단일 스레드로 호출.
 * 재진입(reentrant) 모드이므로 전역 스캐너 상태 오염 없이 파싱할 수 있다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 헤더:
 *   - cuobjdump.h: cuobjdumpSection, cuobjdump_parser, yyscan_t 타입 정의
 *   - cuobjdump_parser.h: Bison이 생성한 토큰/YYSTYPE 선언
 * 이 파일이 생성하는 토큰을 사용하는 모듈:
 *   - cuobjdump.y: Bison 문법 규칙에서 토큰들의 의미 조합
 *   - cuobjdump.cc: Bison 액션에서 호출하는 addCuobjdumpSection(),
 *     setCuobjdumparch() 등의 외부 함수 구현
 * 공유 데이터:
 *   - struct cuobjdump_parser* parser: Flex/Bison이 공유하는 임시 파일 포인터와
 *     serial 카운터를 담는 구조체
 *   - std::list<cuobjdumpSection*>& cuobjdumpSectionList: 파싱된 섹션 객체를
 *     저장하는 리스트 (Bison 액션이 채움)
 *
 * === 주요 함수/구조체 요약 ===
 * - cuobjdump_lex(): Flex가 생성하는 yylex() 대신 재진입 래퍼.
 *   scanner, parser, cuobjdumpSectionList를 인자로 받아 토큰을 반환한다.
 * - cuobjdump_error(): Flex 규칙이 매칭되지 않거나 Bison 파서 오류 발생 시
 *   호출되는 오류 보고 함수. 현재 라인 번호(yylineno)와 오류 메시지를 출력.
 * - 스타트 조건(START conditions): ptxcode, elfcode, sasscode, elfheader,
 *   ptxheader, endheader, identifier, conidentifier, endidentifier.
 *   cuobjdump 출력의 "섹션 본문/헤더/식별자" 단계별로 활성화/비활성화되며,
 *   yy_push_state()/yy_pop_state()/BEGIN()으로 전환된다.
 * - 토큰: PTXHEADER, ELFHEADER, PTXLINE, ELFLINE, SASSLINE, IDENTIFIER,
 *   FILENAME, DECIMAL, H_ARCH, H_CODEVERSION, H_PRODUCER, H_HOST,
 *   H_COMPILESIZE, H_IDENTIFIER, H_COMPRESSED, H_UNKNOWN, CODEVERSION,
 *   NEWLINE.
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
	/*Lex file for output of cuobjdump*/
%{
#include <stdio.h>              /* [한국어] 표준 입출력 — cuobjdump_error()의 printf()용 */
#include <string.h>             /* [한국어] strdup(), yytext 길이 처리 등 문자열 함수용 */
#include "cuobjdump.h"          /* [한국어] cuobjdump_parser, cuobjdumpSection, yyscan_t 타입 정의 */
#include "cuobjdump_parser.h"   /* [한국어] Bison이 생성한 토큰 enum, YYSTYPE, yylex 프로토타입 */

#define YY_NEVER_INTERACTIVE 1  /* [한국어] Flex가 터미널 대화형 모드로 동작하지 않도록 고정 — batch 입력 가정 */
#define YY_NO_INPUT             /* [한국어] input() 함수 생성 생략 — 파일 입력만 사용 */
#define YY_NO_TOP_STATE         /* [한국어] YY_START/top_state() 함수 생성 생략 — 불필요 */

#define YYDEBUG 1               /* [한국어] Flex 디버깅 정보 활성화 (빌드 시 -d 옵션과 함께 사용) */

/* [한국어]
 * cuobjdump_error — Flex/Bison 파싱 중 오류 발생 시 출력하는 함수.
 * scanner, parser, 섹션 리스트, 오류 메시지를 받아 yytext와 현재 라인 번호를
 * 함께 출력한다. Flex 규칙 마지막의 폴스루(.) 규칙에서 호출된다.
 */
void cuobjdump_error(yyscan_t yyscanner, struct cuobjdump_parser* parser, std::list<cuobjdumpSection*> &cuobjdumpSectionList, const char* msg);

/* [한국어]
 * YY_DECL 매크로: Flex가 생성하는 스캐너 함수의 시그니처를 재정의한다.
 * yylval(토큰 의미값), scanner(재진입 핸들), parser(파서 상태),
 * cuobjdumpSectionList(섹션 리스트)를 인자로 받는 cuobjdump_lex()를 생성.
 * %option bison-bridge 덕분에 yylval_param이 YYSTYPE*로 전달된다.
 */
#define YY_DECL int cuobjdump_lex \
	       (YYSTYPE * yylval_param , yyscan_t yyscanner, struct cuobjdump_parser* parser, std::list<cuobjdumpSection*> &cuobjdumpSectionList)
%}

/* [한국어] 스타트 조건(start conditions)을 스택으로 push/pop할 수 있게 함 — 헤더→식별자→본문 순서 진입/복귀 */
%option stack
/* [한국어] EOF 도달 시 yywrap() 호출 없이 0 반환 — 단일 파일 입력 가정 */
%option noyywrap
/* [한국어] 입력 소스의 현재 행 번호를 yylineno에 자동 기록 — 오류 메시지에 사용 */
%option yylineno
/* [한국어] yyunput() 함수 생성 안 함 — 사용하지 않음으로 바이너리 크기 감소 */
%option nounput

/* [한국어] Bison pure parser와 통합: yylval을 직접 인자로 전달받음 */
%option bison-bridge
/* [한국어] 재진입 가능한 스캐너 생성: 전역 변수 대신 yyscan_t 핸들 사용 */
%option reentrant

/* [한국어]
 * 스타트 조건(Start Conditions) 정의.
 * 각 조건은 cuobjdump 출력의 특정 구역에서만 활성화되는 규칙 그룹이다.
 *   INITIAL     : 초기 상태 (헤더 시작 키워드 대기)
 *   ptxcode     : PTX 어셈블리 본문 읽기
 *   sasscode    : SASS 어셈블리 본문 읽기
 *   elfcode     : ELF 섹션의 추가 헤더(예: "code for sm_XX") 읽기
 *   elfheader   : ELF 섹션의 메타데이터(arch, producer, host 등) 읽기
 *   ptxheader   : PTX 섹션의 메타데이터 읽기
 *   endheader   : 메타데이터 마지막 "compile_size = ..." 이후의 식별자 읽기
 *   identifier  : "identifier = " 필드 읽기 준비
 *   conidentifier : "compressed"가 나타난 후의 identifier 읽기 준비
 *   endidentifier : identifier 값(FILENAME) 읽기
 * %x로 선언된 조합(identifier, conidentifier, endidentifier)은 명시적으로
 * 활성화되지 않으면 규칙이 매칭되지 않는 exclusive 조건이다.
 */
%s ptxcode
%s sasscode
%s elfcode
%s elfheader
%s ptxheader
%s endheader
%x identifier
%x conidentifier
%s endidentifier

alpha		[a-zA-Z]            /* [한국어] 영문 알파벳 문자 클래스 */
numeric		[0-9]               /* [한국어] 0~9 숫자 문자 클래스 */
whitespace	[ \t]               /* [한국어] 공백과 탭 문자 클래스 */
newline		[\n]                /* [한국어] 개행 문자 클래스 */
notnewline	[^\n]               /* [한국어] 개행을 제외한 임의의 한 문자 클래스 */
newlines	{newline}+          /* [한국어] 하나 이상의 연속된 개행 문자 */
%%


/* [한국어] "ptxasOptions"으로 시작하는 한 줄 — cuobjdump 출력의 어셈블러 옵션 라인; 파서에 토큰으로 전달하지 않고 무시 */
"ptxasOptions"{notnewline}*{newline}

/* [한국어] 0이 아닌 시작을 가진 10진 정수 — SASS 헤더 등에서 arch 번호로 사용; yylval->string_value에 복사하여 DECIMAL 토큰 반환 */
[1-9]{numeric}* yylval->string_value = strdup(yytext); return DECIMAL;

/* [한국어] "has debug info" 라인 — 디버그 정보 유무를 알리는 cuobjdump 출력; 토큰화하지 않고 무시 */
"has debug info"{newline}


/* [한국어]
 * PTX 섹션 시작: "Fatbin ptx code:" 다음에 개행이 오면
 * 1) ptxcode 스타트 조건 push(PTX 본문 읽기)
 * 2) identifier 스타트 조건 push(식별자 추출)
 * 3) ptxheader 스타트 조건 push(PTX 메타데이터 읽기)
 * 순서대로 스택에 쌓고, PTXHEADER 토큰을 Bison에 반환한다.
 */
"Fatbin ptx code:"{newline}	{
	yy_push_state(ptxcode, yyscanner);
	yy_push_state(identifier, yyscanner);
	yy_push_state(ptxheader, yyscanner);
	yylval->string_value = strdup(yytext);
	return PTXHEADER;
}

/* [한국어]
 * ELF 섹션 시작: "Fatbin elf code:" 다음에 개행이 오면
 * 1) elfcode 스타트 조건 push(ELF 본문/SASS 헤더 읽기)
 * 2) identifier 스타트 조건 push(식별자 추출)
 * 3) elfheader 스타트 조건 push(ELF 메타데이터 읽기)
 * 순서대로 스택에 쌓고, ELFHEADER 토큰을 반환한다.
 */
"Fatbin elf code:"{newline}	{
	yy_push_state(elfcode, yyscanner);
	yy_push_state(identifier, yyscanner);
	yy_push_state(elfheader, yyscanner);
	yylval->string_value = strdup(yytext);
	return ELFHEADER;
}

	/*PTX code tokens*/
/* [한국어] ptxcode 상태에서 개행으로 끝나는 한 줄 전체를 PTXLINE 토큰으로 반환.
 * cuobjdump.y의 ptxcode 규칙이 이 토큰을 받아 parser->ptxfile에 기록한다. */
<ptxcode>{notnewline}*{newline}	yylval->string_value = strdup(yytext); return PTXLINE;

	/*ELF code tokens*/
/* [한국어]
 * elfcode 상태에서 "code for sm_NN" 패턴을 만나면 SASS 본문이 시작된 것이므로
 * sasscode 상태로 전환하고 SASSLINE 토큰으로 반환한다.
 * 이 라인은 ELF 헤더의 arch 정보가 아닌 SASS 본문 직전의 안내 라인이다.
 */
<elfcode>{whitespace}*"code for sm_"{numeric}+{newline}	{
	BEGIN(sasscode); 
	yylval->string_value = strdup(yytext);
	return SASSLINE;
}
/* [한국어] elfcode 상태에서 "code for sm_"이 아닌 다른 모든 줄은 ELFLINE 토큰으로
 * 반환하여 parser->elffile에 기록된다. */
<elfcode>{notnewline}*{newline} yylval->string_value = strdup(yytext); return ELFLINE;

	/*SASS code tokens*/
/* [한국어] sasscode 상태에서 개행으로 끝나는 한 줄 전체를 SASSLINE 토큰으로 반환.
 * cuobjdump.y의 sasscode 규칙이 parser->sassfile에 기록한다. */
<sasscode>{notnewline}*{newline} yylval->string_value = strdup(yytext); return SASSLINE;

/* [한국어]
 * identifier 상태에서 "compressed"가 단독으로 나타나면 압축된 섹션임을 표시.
 * conidentifier 상태로 전환하고 H_COMPRESSED 토큰을 반환한다.
 */
<identifier>{newline}"compressed"{newline}	BEGIN(conidentifier); return H_COMPRESSED;

/* [한국어]
 * identifier 상태에서 "identifier = "를 만나면 identifier 값을 읽기 위해
 * endidentifier 상태로 전환하고 H_IDENTIFIER 토큰을 반환한다.
 */
<identifier>{newline}"identifier = "	BEGIN(endidentifier); return H_IDENTIFIER;

/* [한국어]
 * identifier 상태에서 빈 줄(연속 개행)을 만나면 identifier 필드가 없는 것으로
 * 보고 현재 스타트 조건을 pop하여 이전 상태로 복귀한다.
 */
<identifier>{newline}{newline} yy_pop_state(yyscanner);


/* [한국어]
 * conidentifier(압축 섹션) 상태에서 "identifier = "를 만나면
 * endidentifier로 전환하고 H_IDENTIFIER를 반환한다.
 */
<conidentifier>"identifier = "	BEGIN(endidentifier); return H_IDENTIFIER;

/* [한국어]
 * conidentifier 상태에서 "identifier = "가 아닌 단순 개행을 만나면
 * 압축 마커만 있고 identifier가 없는 것으로 보고 스타트 조건을 pop한다.
 */
<conidentifier>{newline} yy_pop_state(yyscanner);


/* [한국어]
 * endidentifier 상태에서 개행을 제외한 한 줄 전체를 identifier 값으로 보고
 * FILENAME 토큰으로 반환한다. 식별자를 읽은 뒤에는 스타트 조건을 pop하여
 * header 파싱 단계로 돌아간다.
 */
<endidentifier>{notnewline}+	yylval->string_value = strdup(yytext); yy_pop_state(yyscanner); return FILENAME;



	/*Header tokens*/
/* [한국어]
 * elfheader 상태에서 알파벳/숫자/밑줄로 이루어진 토큰은 IDENTIFIER로 반환.
 * arch 값("sm_" 뒤 숫자가 아닌 "60" 등), producer, host, compile_size 값 등에 사용.
 */
<elfheader>[[:alnum:]_]+ yylval->string_value = strdup(yytext); return IDENTIFIER;
/* [한국어] ELF 섹션 헤더의 구분선 "================" — H_SEPARATOR 토큰 반환 */
<elfheader>"================"	return H_SEPARATOR;
/* [한국어] "arch = " 필드 시작 — H_ARCH 토큰 반환, Bison이 다음 IDENTIFIER를 arch로 저장 */
<elfheader>"arch = "		return H_ARCH;
/* [한국어] "code version = " 필드 시작 — H_CODEVERSION 토큰 반환 */
<elfheader>"code version = "	return H_CODEVERSION;
/* [한국어] "[N,M]" 형태의 code version 값 — CODEVERSION 토큰으로 반환 */
<elfheader>\[{numeric},{numeric}\]	return CODEVERSION;
/* [한국어] "producer = " 필드 시작 — H_PRODUCER 토큰 반환 */
<elfheader>"producer = "		return H_PRODUCER;
/* [한국어] producer가 "<unknown>"인 경우 — H_UNKNOWN 토큰 반환 */
<elfheader>"<unknown>"			return H_UNKNOWN;
/* [한국어] "host = " 필드 시작 — H_HOST 토큰 반환 */
<elfheader>"host = "			return H_HOST;
/* [한국어] "compile_size = " 필드 시작 — H_COMPILESIZE 토큰 반환, endheader 상태로 전환하여 값(식별자) 읽기 */
<elfheader>"compile_size = "	BEGIN(endheader); return H_COMPILESIZE;


	/*Header tokens*/
/* [한국어] ptxheader 상태에서 알파벳/숫자/밑줄 토큰 — IDENTIFIER 반환 */
<ptxheader>[[:alnum:]_]+ yylval->string_value = strdup(yytext); return IDENTIFIER;
/* [한국어] PTX 섹션 헤더 구분선 — H_SEPARATOR 반환 */
<ptxheader>"================"	return H_SEPARATOR;
/* [한국어] PTX 섹션 arch 필드 시작 — H_ARCH 반환 */
<ptxheader>"arch = "		return H_ARCH;
/* [한국어] PTX 섹션 code version 필드 시작 — H_CODEVERSION 반환 */
<ptxheader>"code version = "	return H_CODEVERSION;
/* [한국어] PTX code version 값 "[N,M]" — CODEVERSION 반환 */
<ptxheader>\[{numeric},{numeric}\]	return CODEVERSION;
/* [한국어] PTX producer 필드 시작 — H_PRODUCER 반환 */
<ptxheader>"producer = "		return H_PRODUCER;
/* [한국어] PTX host 필드 시작 — H_HOST 반환 */
<ptxheader>"host = "			return H_HOST;
/* [한국어] PTX compile_size 필드 시작 — H_COMPILESIZE 반환, endheader로 전환 */
<ptxheader>"compile_size = "	BEGIN(endheader); return H_COMPILESIZE;





	/* Looking for the identifier (filename) then the header is done */
/* [한국어]
 * endheader 상태에서 개행을 제외한 한 줄 전체를 헤더의 마지막 값(compile_size 값)
 * 또는 섹션 식별자로 보고 IDENTIFIER 토큰으로 반환한다.
 * 값을 읽은 뒤에는 스타트 조건을 pop하여 헤더 파싱을 마친다.
 */
<endheader>{notnewline}+	yylval->string_value = strdup(yytext); yy_pop_state(yyscanner); return IDENTIFIER;



/* [한국어] 위의 특정 상태 규칙에 걸리지 않은 단일 개행은 NEWLINE 토큰으로 반환.
 * emptylines 규칙에서 빈 줄들을 소비하는 데 사용된다. */
{newline}	return NEWLINE;

	/*Reached end of file*/
/* [한국어] EOF 도달 시 INITIAL 상태로 되돌리고 0을 반환하여 Flex/Bison에게
 * 입력 종료를 알린다. */
<<EOF>> BEGIN(INITIAL);return 0;

	/*No other rule matched. Throw an error*/
/* [한국어] 어떤 규칙도 매칭되지 않는 문자를 만나면 cuobjdump_error()를 호출하여
 * "Invalid token" 오류를 출력한다. 이는 cuobjdump 출력 형식이 예상과 다를 때
 * 도달하는 폴스루 경보다. */
.	cuobjdump_error(yyscanner, parser, cuobjdumpSectionList, "Invalid token");

%%

/* [한국어]
 * cuobjdump_error — Flex/Bison 파싱 오류 보고 함수.
 * @yyscanner: 재진입 Flex 스캐너 핸들 (yyguts_t* 캐스팅하여 yytext/yylineno 접근)
 * @parser:    cuobjdump_parser 상태 구조체 (현재는 오류 출력에 직접 사용하지 않음)
 * @cuobjdumpSectionList: 파싱 중인 섹션 리스트 (오류 출력에 직접 사용하지 않음)
 * @msg:       오류 메시지 문자열
 * @return:    없음
 *
 * Flex의 폴스루(.) 규칙에서 호출되며, 예상치 못한 문자가 나타났을 때
 * 해당 문자(yytext)와 현재 라인 번호(yylineno)를 함께 출력하여
 * cuobjdump 출력 형식 변경 또는 Lex 규칙 누락을 디버깅할 수 있게 한다.
 */
void cuobjdump_error(yyscan_t yyscanner, struct cuobjdump_parser* parser, std::list<cuobjdumpSection*> &cuobjdumpSectionList, const char* msg)
{
    struct yyguts_t * yyg = (struct yyguts_t*)yyscanner;
	printf("   %s near \"%s\"",msg, yytext);
	printf(" on line %i\n",yylineno);
}
