/*
 * [한국어 설명] SASS 디스어셈블리 Lex/Flex 어휘 분석기 (sass.l)
 *
 * === 파일의 역할 ===
 * cuobjdump가 출력한 SASS(Shader Assembly) 디스어셈블리 텍스트를
 * 읽어 Bison(sass.y)이 사용할 토큰으로 변환하는 Flex lexer 정의 파일이다.
 * SASS 명령어 니모닉, 수정자, 레지스터, 메모리 위치, 상수값, 라벨,
 * 프레디케이트 등을 토큰화한다. GPGPU-Sim 메인 PTX lexer(ptx.l)와는
 * 달리 SASS 문법에 특화되어 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   cuobjdump_to_ptxplus.cc
 *     → sass_parse() (sass.y가 생성)
 *         → sass.l (이 파일)이 SASS 텍스트를 토큰화
 *             → sass.y가 cuobjdumpInstList(IR) 구축
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - sass_parser.hh : Bison이 생성한 토큰 정의
 * 이 파일에 의존하는 모듈:
 *   - sass.y         : lexer 토큰을 문법 규칙에서 사용
 *   - 빌드 시스템    : flex sass.l → sass.yy.cc 생성
 *
 * === 주요 함수/구조체 요약 ===
 * Flex 옵션:
 *   noyywrap : 입력 종료 시 yywrap 호출 안 함
 *   yylineno : 줄 번호 추적
 *   nounput  : unput 함수 비생성
 * 주요 토큰 범주:
 *   - SASS 명령어: ADA, BRA, IADD, GLD, GST, MOV 등
 *   - 비교/수정자: EQ, GE, DOTTRUNC, DOTANY 등
 *   - 타입 수정자: DOTF32, DOTS32, DOTU64 등
 *   - 레지스터: REGISTER, REGISTERLO, REGISTERHI, OFFSETREGISTER 등
 *   - 메모리 위치: SMEMLOCATION, GMEMLOCATION, CMEMLOCATION, LMEMLOCATION
 *   - 프레디케이트: PREDREGISTER, PREDREGISTER2, PREDREGISTER3
 *   - 라벨/헤더: LABEL, VERSIONHEADER, FUNCTIONHEADER 등
 */

/*
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
*/
%{
#include <stdio.h>
#include "sass_parser.hh"
#include <string.h>


/* Not sure if this is needed*/
#define	YY_NEVER_INTERACTIVE	1

#define yylval sass_lval

/* [한국어] SASS lexer 오류 처리 함수 선언. */
void sass_error(const char*);
%}
%option noyywrap
%option yylineno
%option nounput

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
pipe		\|

identifier	({alpha}({alphanum}|["_$"])*)|([_$%]({alphanum}|["_$"])+)
hexliteral	0[xX]{hexdigit}+U?
octliteral	0[0-7]+U?
binliteral	0[bB][01]+U?
decliteral	([1-9][0-9]*U?)|0U?
fltliteral	[0-9]+\.[0-9]+

%%
	/*Translation rules*/

	/*Compiler directives*/

	/*constant memory directives*/

	/*guard predicate*/

	/*label*/

"[" yylval.string_value = strdup(yytext); return OSQBRACKET;
"]" yylval.string_value = strdup(yytext); return CSQBRACKET; 

	/*base instructions*/
/* [한국어] SASS 기본 명령어 니모닉들을 토큰으로 반환. */
"BAR.ARV.WAIT b0, 0xfff"	yylval.string_value = strdup(yytext); return BAR;
ADA	yylval.string_value = strdup(yytext); return ADA;
BRA	yylval.string_value = strdup(yytext); return BRA;
BRX	yylval.string_value = strdup(yytext); return BRX;
CAL	yylval.string_value = strdup(yytext); return CAL;
COS	yylval.string_value = strdup(yytext); return COS;
DADD	yylval.string_value = strdup(yytext); return DADD;
DMIN	yylval.string_value = strdup(yytext); return DMIN;
DMAX	yylval.string_value = strdup(yytext); return DMAX;
DFMA	yylval.string_value = strdup(yytext); return DFMA;
DMUL	yylval.string_value = strdup(yytext); return DMUL;
EX2	yylval.string_value = strdup(yytext); return EX2;
F2F	yylval.string_value = strdup(yytext); return F2F;
F2I	yylval.string_value = strdup(yytext); return F2I;
FADD	yylval.string_value = strdup(yytext); return FADD;
FADD32	yylval.string_value = strdup(yytext); return FADD32;
FADD32I	yylval.string_value = strdup(yytext); return FADD32I;
FMAD	yylval.string_value = strdup(yytext); return FMAD;
FMAD32I	yylval.string_value = strdup(yytext); return FMAD32I;
FMUL	yylval.string_value = strdup(yytext); return FMUL;
FMUL32	yylval.string_value = strdup(yytext); return FMUL32;
FMUL32I	yylval.string_value = strdup(yytext); return FMUL32I;
FSET	yylval.string_value = strdup(yytext); return FSET;
DSET	yylval.string_value = strdup(yytext); return DSET;
G2R	yylval.string_value = strdup(yytext); return G2R;
GLD	yylval.string_value = strdup(yytext); return GLD;
GST	yylval.string_value = strdup(yytext); return GST;
I2F	yylval.string_value = strdup(yytext); return I2F;
I2I	yylval.string_value = strdup(yytext); return I2I;
S2R	yylval.string_value = strdup(yytext); return S2R;
STS	yylval.string_value = strdup(yytext); return STS;
LD	yylval.string_value = strdup(yytext); return SASS_LD;
ST	yylval.string_value = strdup(yytext); return SASS_ST;
IADD	yylval.string_value = strdup(yytext); return IADD;
IADD\.CARRY{numeric}	yylval.string_value = strdup(yytext); return IADDCARRY;
IADD32	yylval.string_value = strdup(yytext); return IADD32;
IADD32I	yylval.string_value = strdup(yytext); return IADD32I;
IMAD	yylval.string_value = strdup(yytext); return IMAD;
ISAD	yylval.string_value = strdup(yytext); return ISAD;
IMAD\.U24	yylval.string_value = strdup(yytext); return IMAD24;
IMAD\.S24	yylval.string_value = strdup(yytext); return IMAD24;
IMAD32	yylval.string_value = strdup(yytext); return IMAD32;
IMAD32I	yylval.string_value = strdup(yytext); return IMAD32I;
IMUL	yylval.string_value = strdup(yytext); return IMUL;
IMUL\.U24\.U24	yylval.string_value = strdup(yytext); return IMUL24;
IMUL\.S24\.S24	yylval.string_value = strdup(yytext); return IMULS24;
IMUL\.HI\.U24\.U24	yylval.string_value = strdup(yytext); return IMUL24H;
IMUL32	yylval.string_value = strdup(yytext); return IMUL32;
IMUL32\.S24\.S24	yylval.string_value = strdup(yytext); return IMUL32S24;
IMUL32\.U24\.U24	yylval.string_value = strdup(yytext); return IMUL32U24;
IMUL32I	yylval.string_value = strdup(yytext); return IMUL32I;
IMUL32I\.S24\.S24	yylval.string_value = strdup(yytext); return IMUL32IS24;
IMUL32I\.U24\.U24	yylval.string_value = strdup(yytext); return IMUL32I24;
ISET	yylval.string_value = strdup(yytext); return ISET;
LG2	yylval.string_value = strdup(yytext); return LG2;
LLD	yylval.string_value = strdup(yytext); return LLD;
LOP\.AND	yylval.string_value = strdup(yytext); return AND;
LOP\.OR	yylval.string_value = strdup(yytext); return OR;
LOP\.PASS_B	yylval.string_value = strdup(yytext); return NOT;
LOP\.XOR	yylval.string_value = strdup(yytext); return XOR;
LOP\.S\.AND	yylval.string_value = strdup(yytext); return ANDS;
LOP\.S\.OR	yylval.string_value = strdup(yytext); return ORS;
LOP\.S\.PASS_B	yylval.string_value = strdup(yytext); return NOTS;
LOP\.S\.XOR	yylval.string_value = strdup(yytext); return XORS;
LST	yylval.string_value = strdup(yytext); return LST;
MOV	yylval.string_value = strdup(yytext); return MOV;
MOV32	yylval.string_value = strdup(yytext); return MOV32;
MVC	yylval.string_value = strdup(yytext); return MVC;
MVI	yylval.string_value = strdup(yytext); return MVI;
NOP	yylval.string_value = strdup(yytext); return NOP;
R2A	yylval.string_value = strdup(yytext); return R2A;
R2G	yylval.string_value = strdup(yytext); return R2G;
R2G.U16.U8	yylval.string_value = strdup(yytext); return R2GU16U8;
RCP	yylval.string_value = strdup(yytext); return RCP;
RCP32	yylval.string_value = strdup(yytext); return RCP32;
RET	yylval.string_value = strdup(yytext); return RET;
RRO	yylval.string_value = strdup(yytext); return RRO;
RSQ	yylval.string_value = strdup(yytext); return RSQ;
SIN	yylval.string_value = strdup(yytext); return SIN;
SHL	yylval.string_value = strdup(yytext); return SHL;
SHR	yylval.string_value = strdup(yytext); return SHR;
SSY	yylval.string_value = strdup(yytext); return SSY;
BAR	yylval.string_value = strdup(yytext); return BAR;
LDS	yylval.string_value = strdup(yytext); return LDS;
IMIN	yylval.string_value = strdup(yytext); return IMIN;
IMAX	yylval.string_value = strdup(yytext); return IMAX;
FMAX	yylval.string_value = strdup(yytext); return FMAX;
FMIN	yylval.string_value = strdup(yytext); return FMIN;
C2R	yylval.string_value = strdup(yytext); return C2R;
R2C	yylval.string_value = strdup(yytext); return R2C;
A2R	yylval.string_value = strdup(yytext); return A2R;
TEX	yylval.string_value = strdup(yytext); return TEX;
TEX32	yylval.string_value = strdup(yytext); return TEX32;
EXIT	yylval.string_value = strdup(yytext); return EXIT;
GRED	yylval.string_value = strdup(yytext); return GRED;
GATOM	yylval.string_value = strdup(yytext); return GATOM;
PBK	yylval.string_value = strdup(yytext); return PBK;
BRK	yylval.string_value = strdup(yytext); return BRK;
VOTE	yylval.string_value = strdup(yytext); return VOTE;

	/*instruction modifiers*/
/* [한국어] 명령어 수정자(비교 연산자, 반올림, 기타 플래그) 토큰. */
EQ	yylval.string_value = strdup(yytext); return EQ;
EQU	yylval.string_value = strdup(yytext); return EQU;
GE	yylval.string_value = strdup(yytext); return GE;
GEU	yylval.string_value = strdup(yytext); return GEU;
GT	yylval.string_value = strdup(yytext); return GT;
GTU	yylval.string_value = strdup(yytext); return GTU;
LE	yylval.string_value = strdup(yytext); return LE;
LEU	yylval.string_value = strdup(yytext); return LEU;
LT	yylval.string_value = strdup(yytext); return LT;
LTU	yylval.string_value = strdup(yytext); return LTU;
NE	yylval.string_value = strdup(yytext); return NE;
NEU	yylval.string_value = strdup(yytext); return NEU;
\.BEXT	yylval.string_value = strdup(yytext); return DOTBEXT;
\.S	yylval.string_value = strdup(yytext); return DOTS;
\.SFU	yylval.string_value = strdup(yytext); return DOTSFU;
\.TRUNC	yylval.string_value = strdup(yytext); return DOTTRUNC;
\.CEIL	yylval.string_value = strdup(yytext); return DOTCEIL;
\.FLOOR	yylval.string_value = strdup(yytext); return DOTFLOOR;
\.IR	yylval.string_value = strdup(yytext); /*unknown*/
\.UN	yylval.string_value = strdup(yytext); return DOTUN;
\.NODEP	yylval.string_value = strdup(yytext); return DOTNODEP;
\.SAT	yylval.string_value = strdup(yytext); /*unknown*/
\.ANY	yylval.string_value = strdup(yytext); return DOTANY;
\.ALL	yylval.string_value = strdup(yytext); return DOTALL;
\.MAD	yylval.string_value = strdup(yytext); /*unknown*/



	/* comparison operators */
\.EQ	yylval.string_value = strdup(yytext); return DOTEQ;
\.EQU	yylval.string_value = strdup(yytext); return DOTEQU;
\.FALSE	yylval.string_value = strdup(yytext); return DOTFALSE;
\.GE	yylval.string_value = strdup(yytext); return DOTGE;
\.GEU	yylval.string_value = strdup(yytext); return DOTGEU;
\.GT	yylval.string_value = strdup(yytext); return DOTGT;
\.GTU	yylval.string_value = strdup(yytext); return DOTGTU;
\.LE	yylval.string_value = strdup(yytext); return DOTLE;
\.LEU	yylval.string_value = strdup(yytext); return DOTLEU;
\.LT	yylval.string_value = strdup(yytext); return DOTLT;
\.LTU	yylval.string_value = strdup(yytext); return DOTLTU;
\.NE	yylval.string_value = strdup(yytext); return DOTNE;
\.NEU	yylval.string_value = strdup(yytext); return DOTNEU;
\.NOT_SIGN	yylval.string_value = strdup(yytext); return DOTNSF;
\.SIGN	yylval.string_value = strdup(yytext); return DOTSF;
\.CARRY	yylval.string_value = strdup(yytext); return DOTCARRY;
\.CARRY{numeric}	yylval.string_value = strdup(yytext); /*unknown*/
\.CC	yylval.string_value = strdup(yytext); return DOTCC;
\.X		yylval.string_value = strdup(yytext); return DOTX;
\.E		yylval.string_value = strdup(yytext); return DOTE;
\.RED	yylval.string_value = strdup(yytext); return DOTRED;
\.POPC	yylval.string_value = strdup(yytext); return DOTPOPC;

	/*operand types*/
\.F16	yylval.string_value = strdup(yytext); return DOTF16;
\.F32	yylval.string_value = strdup(yytext); return DOTF32;
\.F64	yylval.string_value = strdup(yytext); return DOTF64;
\.S8	yylval.string_value = strdup(yytext); return DOTS8;
\.S16	yylval.string_value = strdup(yytext); return DOTS16;
\.S32	yylval.string_value = strdup(yytext); return DOTS32;
\.S64	yylval.string_value = strdup(yytext); return DOTS64;
\.S128	yylval.string_value = strdup(yytext); return DOTS128;
\.U8	yylval.string_value = strdup(yytext); return DOTU8;
\.U16	yylval.string_value = strdup(yytext); return DOTU16;
\.U32	yylval.string_value = strdup(yytext); return DOTU32;
\.U24	yylval.string_value = strdup(yytext); return DOTU24;
\.U64	yylval.string_value = strdup(yytext); return DOTU64;
\.HI	yylval.string_value = strdup(yytext); return DOTHI;
\.NOINC	yylval.string_value = strdup(yytext); return DOTNOINC;

	/*geometries*/

	/*register operands*/
/* [한국어] SASS 레지스터 피연산자들. R#(일반), R#L/R#H(lo/hi), A#(offset),
 * .C#(predicate), o[0x#](predicate2), C#(predicate3), SR_*(특수 레지스터). */
[\-]?R{numeric}+	yylval.string_value = strdup(yytext); return REGISTER;
[\-]?R{numeric}+L	yylval.string_value = strdup(yytext); return REGISTERLO;
[\-]?R{numeric}+H    yylval.string_value = strdup(yytext); return REGISTERHI;
A{numeric}+	yylval.string_value = strdup(yytext); return OFFSETREGISTER;
\.C{numeric}	yylval.string_value = strdup(yytext); return PREDREGISTER;
"o ["{hexliteral}\]	yylval.string_value = strdup(yytext); return PREDREGISTER2;
C{numeric}	 yylval.string_value = strdup(yytext); return PREDREGISTER3;
"SR_"[^\.; \t]+	 yylval.string_value = strdup(yytext); return SREGISTER;

	/*tex operand*/

	/*vector operands*/

	/*special registers*/

	/* cuobjdump header */
/* [한국어] cuobjdump 출력의 섹션 헤더들(무시 또는 토큰 반환). */
"Fatbin ptx code:" yylval.string_value = strdup(yytext); return PTXHEADER;
"Fatbin elf code:" yylval.string_value = strdup(yytext); return ELFHEADER;
"================"
"arch = " yylval.string_value = strdup(yytext); return INFOARCHVERSION;
"code version = " yylval.string_value = strdup(yytext); return INFOCODEVERSION_HEADER;
\[{numeric},{numeric}\] yylval.string_value = strdup(yytext); return INFOCODEVERSION;
"producer = " yylval.string_value = strdup(yytext); return INFOPRODUCER;
"host = " yylval.string_value = strdup(yytext); return INFOHOST;
{numeric}{numeric}"bit" yylval.string_value = strdup(yytext); return INFOCOMPILESIZE;
"compile_size = " yylval.string_value = strdup(yytext); return INFOCOMPILESIZE_HEADER;
"identifier = " yylval.string_value = strdup(yytext); return INFOIDENTIFIER;
"code for "	yylval.string_value = strdup(yytext); return VERSIONHEADER;
"Function : "	yylval.string_value = strdup(yytext); return FUNCTIONHEADER;
"/*"{hexliteral}{whitespace}*"*/" 	yylval.string_value = strdup(yytext); return INSTHEX;
"." yylval.string_value = strdup(yytext); return DOT;

	/*memory location operands*/
/* [한국어] 메모리 위치 피연산자들.
 * ABSSMEMLOCATION: |g| [..] 형태(절대 공유 메모리)
 * SMEMLOCATION:    g [..] 형태(공유 메모리)
 * GMEMLOCATION:    global14 [..] 형태(전역 메모리)
 * CMEMLOCATION:    c [..] [..] 형태(상수 메모리)
 * LMEMLOCATION:    local [..] 형태(로컬 메모리)
 */
[\-]?"|g| ["{hexliteral}"]"".U16"?	yylval.string_value = strdup(yytext); return ABSSMEMLOCATION;
[\-]?"|g| [R"{numeric}+\]".U16"?	yylval.string_value = strdup(yytext); return ABSSMEMLOCATION;
[\-]?"|g| [A"{numeric}+\+{hexliteral}\]".U16"?	yylval.string_value = strdup(yytext); return ABSSMEMLOCATION;
[\-]?"|g| [A"{numeric}+\+\+\+{hexliteral}\]".U16"?	yylval.string_value = strdup(yytext); return ABSSMEMLOCATION;

[\-]?"g ["{hexliteral}"]"".U16"?	yylval.string_value = strdup(yytext); return SMEMLOCATION;
[\-]?"g [R"{numeric}+\]".U16"?	yylval.string_value = strdup(yytext); return SMEMLOCATION;
[\-]?"g [A"{numeric}+\+{hexliteral}\]".U16"?	yylval.string_value = strdup(yytext); return SMEMLOCATION;
[\-]?"g [A"{numeric}+\+\+\+{hexliteral}\]".U16"?	yylval.string_value = strdup(yytext); return SMEMLOCATION;

[\-]?"global14 ["{hexliteral}"]"".U16"?	yylval.string_value = strdup(yytext); return GMEMLOCATION;
[\-]?"global14 [R"{numeric}+\]".U16"?	yylval.string_value = strdup(yytext); return GMEMLOCATION;
[\-]?"global14 [A"{numeric}+\+{hexliteral}\]".U16"?	yylval.string_value = strdup(yytext); return GMEMLOCATION;
[\-]?"global14 [A"{numeric}+\+\+\+{hexliteral}\]".U16"?	yylval.string_value = strdup(yytext); return GMEMLOCATION;

[\-]?"c ["{hexliteral}"] ["{hexliteral}"]"".U16"?	yylval.string_value = strdup(yytext); return CMEMLOCATION;
[\-]?"c ["{hexliteral}"] [R"{numeric}+\]".U16"?	yylval.string_value = strdup(yytext); return CMEMLOCATION;
[\-]?"c ["{hexliteral}"] [A"{numeric}+\+{hexliteral}\]".U16"?	yylval.string_value = strdup(yytext); return CMEMLOCATION;
[\-]?"c ["{hexliteral}"] [A"{numeric}+\+\+\+{hexliteral}\]".U16"?	yylval.string_value = strdup(yytext); return CMEMLOCATION;

[\-]?"local ["{hexliteral}"]"".U16"?	yylval.string_value = strdup(yytext); return LMEMLOCATION;
[\-]?"local [R"{numeric}+\]".U16"?	yylval.string_value = strdup(yytext); return LMEMLOCATION;
[\-]?"local [A"{numeric}+\+{hexliteral}\]".U16"?	yylval.string_value = strdup(yytext); return LMEMLOCATION;
[\-]?"local [A"{numeric}+\+\+\+{hexliteral}\]".U16"?	yylval.string_value = strdup(yytext); return LMEMLOCATION;

	/*identifier + literals*/
		
{identifier}	yylval.string_value = strdup(yytext); return IDENTIFIER;
[\-]?{hexliteral}	yylval.string_value = strdup(yytext); return HEXLITERAL; //change to int later?

	/*label ends*/
"/*"	yylval.string_value = strdup(yytext); return LABELSTART;
"*/"	yylval.string_value = strdup(yytext); return LABELEND;
{hexdigit}{hexdigit}{hexdigit}{hexdigit} yylval.string_value = strdup(yytext); return LABEL;

	/*braces*/

	/*brackets*/
\(	yylval.string_value = strdup(yytext); return LEFTBRACKET;
\)	yylval.string_value = strdup(yytext); return RIGHTBRACKET;

	/* pound sign*/

	/* comma*/
","	/*yylval.string_value = strdup(yytext); return COMMA;*/

	/* underscore sign*/

	/* pipe*/
"|"	yylval.string_value = strdup(yytext); return PIPE;
"~"	yylval.string_value = strdup(yytext); return TILDE;

	/*Need to create to newline tokens*/
{newline}+	return NEWLINE;

	/* semicolon */
;	return SEMICOLON;

	/*discard whitespace*/
{whitespace}+
	/*discard comments*/
"//"{anycharbutnewline}*
	/*discard dividers*/
"-----"[\-]+
"....."[\.]+


	/*reached end of file*/
<<EOF>>	return 0;

	/*None of the rules matched. Throw error*/
.			sass_error("Invalid token");

%%

/*
 * [한국어]
 * sass_error - lexer가 인식할 수 없는 토큰을 만났을 때 호출
 *
 * @message: 오류 메시지
 *
 * 메시지, 현재 토큰, 줄 번호를 출력하고 exit(1)로 종료한다.
 */
void sass_error(const char* message)
{
	printf("   "); printf("%s",message); printf(" near \""); printf("%s",yytext); printf("\"");
	printf(" on line "); 
	char line[5]; sprintf(line, "%i", yylineno); printf("%s",line);
	printf("\n");
	exit(1);
}
