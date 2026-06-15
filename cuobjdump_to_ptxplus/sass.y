/*
 * [한국어 설명] SASS 디스어셈블리 Bison/Yacc 문법 파서 (sass.y)
 *
 * === 파일의 역할 ===
 * cuobjdump가 출력한 SASS 디스어셈블리를 파싱하여 cuobjdumpInstList
 * 형태의 중간 표현(IR)으로 변환하는 Bison 문법 분석기 정의 파일이다.
 * sass.l이 반환한 토큰을 받아 "Function : <이름>" 블록별로 엔트리를
 * 생성하고, 각 SASS 명령어를 cuobjdumpInst로 변환한다. 분기/호출,
 * 메모리 피연산자, 레지스터, 프레디케이트 등의 변환 규칙을 포함한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   cuobjdump_to_ptxplus.cc
 *     → sass_parse() (이 파일이 생성)
 *         → sass.l (토큰 생성)
 *             → 이 파일의 규칙에 따라 g_instList 구축
 *                 → printCuobjdumpPtxPlusList()로 PTXPlus 출력
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - cuobjdumpInstList.h/.cc : IR 컨테이너 및 레지스터/메모리 처리 API
 *   - sass.l                  : SASS 토큰 제공
 * 이 파일에 의존하는 모듈:
 *   - 빌드 시스템 : bison sass.y → sass_parser.hh/sass.tab.cc 생성
 *
 * === 주요 함수/구조체 요약 ===
 * program              : sassCode 반복
 * sassCode             : VERSIONHEADER + IDENTIFIER + functionList
 * functionList/function: FUNCTIONHEADER IDENTIFIER + statementList
 * statement            : instructionLabel + instructionHex + assemblyInstruction
 * instructionLabel     : LABELSTART LABEL LABELEND → l0x... 형태 레이블 생성
 * assemblyInstruction  : baseInstruction + modifierList + operandList
 * baseInstruction      : simpleInstructions | GRED/GATOM DOT simpleInstructions
 * branchInstructions   : BRA/CAL + 선택적 predicate + HEXLITERAL(목적지 레이블)
 * pbkInstruction       : PBK + HEXLITERAL(브레이크 타겟)
 * modifier             : opTypes | DOTBEXT | DOTTRUNC 등
 * operand              : registerlocation | memorylocation | immediateValue |
 *                        operandPredicate | preOperand 등
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
#include "cuobjdumpInstList.h"

int yylex(void);
void yyerror(const char*);
/* [한국어] 디버그 출력용 함수(기본적으로 비활성화). */
void debug_print( const char *s );

/* [한국어] SASS 변환 IR 컨테이너. */
extern cuobjdumpInstList *g_instList;

/* [한국어] 현재 파싱 중인 SASS 명령어 IR 객체. */
cuobjdumpInst *instEntry;
%}


%union {
  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  void * ptr_value;
}

%token <string_value> BAR
%token <string_value> ADA AND ANDS BRA BRX CAL COS DADD DMIN DMAX DFMA DMUL EX2 F2F F2I FADD
%token <string_value> FADD32 FADD32I FMAD FMAD32I FMUL FMUL32 FMUL32I FSET DSET G2R
%token <string_value> GLD GST I2F I2I IADD IADD32 IADD32I IMAD ISAD IMAD24 IMAD32I IMAD32 IADDCARRY
%token <string_value> IMUL IMUL24 IMUL24H IMULS24 IMUL32 IMUL32S24 IMUL32U24 IMUL32I IMUL32I24 IMUL32IS24
%token <string_value> ISET LG2 LLD LST MOV MOV32 MVC MVI NOP NOT NOTS OR ORS
%token <string_value> R2A R2G R2GU16U8 RCP RCP32 RET RRO RSQ SIN SHL SHR SSY XOR XORS
%token <string_value> S2R SASS_LD STS LDS SASS_ST IMIN IMAX A2R FMAX FMIN TEX TEX32 C2R EXIT
%token <string_value> GRED PBK BRK R2C GATOM VOTE

%token <string_value> EQ EQU GE GEU GT GTU LE LEU LT LTU NE NEU
%token <string_value> DOTBEXT DOTS DOTSFU
%token <string_value> DOTTRUNC DOTCEIL DOTFLOOR DOTIR DOTUN DOTNODEP DOTSAT DOTANY DOTALL
%token <string_value> DOTF16 DOTF32 DOTF64 DOTS8 DOTS16 DOTS32 DOTS64 DOTS128 DOTU8 DOTU16 DOTU32 DOTU24 DOTU64
%token <string_value> DOTHI DOTNOINC
%token <string_value> DOTEQ DOTEQU DOTFALSE DOTGE DOTGEU DOTGT DOTGTU DOTLE DOTLEU DOTLT DOTLTU DOTNE DOTNEU DOTNSF DOTSF DOTCARRY
%token <string_value> DOTCC DOTX DOTE DOTRED DOTPOPC
%token <string_value> REGISTER REGISTERLO REGISTERHI OFFSETREGISTER
%token <string_value> PREDREGISTER PREDREGISTER2 PREDREGISTER3 SREGISTER
%token <string_value> VERSIONHEADER FUNCTIONHEADER
%token <string_value> SMEMLOCATION ABSSMEMLOCATION GMEMLOCATION CMEMLOCATION LMEMLOCATION
%token <string_value> IDENTIFIER
%token <string_value> HEXLITERAL
%token <string_value> LEFTBRACKET RIGHTBRACKET
%token <string_value> PIPE TILDE
%token <string_value> NEWLINE SEMICOLON /*COMMA*/
%token <string_value> LABEL LABELSTART LABELEND
%token <string_value> PTXHEADER ELFHEADER
%token <string_value> INFOARCHVERSION
%token <string_value> INFOCODEVERSION_HEADER INFOCODEVERSION
%token <string_value> INFOPRODUCER
%token <string_value> INFOHOST
%token <string_value> INFOCOMPILESIZE_HEADER INFOCOMPILESIZE
%token <string_value> INFOIDENTIFIER DOT
%token <string_value> INSTHEX
%token <string_value> OSQBRACKET CSQBRACKET
	/* set types for rules */
%type<string_value> simpleInstructions
%type<string_value> predicateModifier
%type<string_value> opTypes

%%

	/*translation rules*/
/* [한국어] 최상위 프로그램: 여러 개의 sassCode 블록(SM 아키텍처별) 반복. */
program		: program sassCode
				| sassCode;

/* [한국어] SASS 코드 블록 하나: "code for <arch>" + 함수 목록. */
sassCode	: VERSIONHEADER IDENTIFIER NEWLINE functionList			{ debug_print($1); debug_print($2); debug_print(" No parsing errors\n\n");  }
			| NEWLINE VERSIONHEADER IDENTIFIER NEWLINE functionList	{ debug_print($2); debug_print($3); debug_print(" No parsing errors\n\n");  }
			| VERSIONHEADER IDENTIFIER NEWLINE;

functionList	: functionList function
					| function
					;
					
/* [한국어] 함수(엔트리) 정의: "Function : <이름>" + statementList. */
function	:	FUNCTIONHEADER IDENTIFIER {
						debug_print($1); 
						debug_print($2);
						debug_print("\n");
						g_instList->addEntry($2);
						instEntry = new cuobjdumpInst();
						instEntry->setBase(".entry");
						g_instList->add(instEntry);
						g_instList->getListEnd().addOperand($2);} statementList NEWLINE
						;


statementList	: statementList statement NEWLINE	{ debug_print("\n"); }
			| statementList statement SEMICOLON NEWLINE	{ debug_print(";\n"); }
			| statement NEWLINE			{ debug_print("\n"); }
			| statement SEMICOLON NEWLINE			{ debug_print(";\n"); }
			| NEWLINE	{}
			;

/* [한국어] statement: 새 cuobjdumpInst를 할당한 뒤 라벨과 명령어 파싱. */
statement	: { instEntry = new cuobjdumpInst(); } instructionLabel statementend
				;

statementend	: instructionHex assemblyInstruction
			| /*blank*/ {instEntry->setBase("NOP"); g_instList->add(instEntry); debug_print("NOP");}
			;

instructionHex	: INSTHEX
					;

// [한국어] 명령어 라벨: /* hex */ 형태의 인스트럭션 헥스를 l0xXXXXXXXX 형태로 변환.
instructionLabel	: LABELSTART LABEL LABELEND	{ char* tempInput = $2;
								  char* tempLabel = new char[12];
								  tempLabel[0] = 'l';
								  tempLabel[1] = '0';
								  tempLabel[2] = 'x';
								  for(int i=0; i<(8-strlen(tempInput)); i++)
								  {
									tempLabel[3+i] = '0';
								  }
								  for(int i=(11-strlen(tempInput)); i<11; i++)
								  {
									tempLabel[i] = tempInput[i-(11-strlen(tempInput))];
								  }
								  tempLabel[11] = '\0';
								  instEntry->setLabel(tempLabel); }
				;

assemblyInstruction	: baseInstruction modifierList operandList	{ }
						/*| baseInstruction operandList			{ }*/
						/*| baseInstruction modifierList			{ }*/
						/*| baseInstruction				{ }*/
						;

/* [한국어] 기본 명령어: simpleInstructions 토큰을 base로 설정하고 IR에 추가.
 * GRED/GATOM은 .(dot) 뒤의 연산을 base modifier로 추가. */
baseInstruction : simpleInstructions	{ debug_print($1); instEntry->setBase($1); g_instList->add(instEntry);}
			| branchInstructions
			| GRED DOT simpleInstructions	{ debug_print($1); instEntry->setBase($1); g_instList->add(instEntry); g_instList->getListEnd().addBaseModifier($3);}
			| GATOM DOT simpleInstructions	{ debug_print($1); instEntry->setBase($1); g_instList->add(instEntry); g_instList->getListEnd().addBaseModifier($3);}
			| pbkInstruction
			;

simpleInstructions	: ADA | AND | ANDS | BRX | COS | DADD | DMIN | DMAX | DFMA | DMUL | EX2 | F2F 
						| F2I | FADD | FADD32 | FADD32I | FMAD | FMAD32I | FMUL 
						| FMUL32 | FMUL32I | FSET | DSET | G2R | GLD | GST | I2F | I2I 
						| IADD | IADD32 | IADD32I | IMAD | ISAD | IMAD24 | IMAD32I | IMAD32 | IMUL 
						| IMUL24 | IMUL24H | IMULS24 | IMUL32 | IMUL32S24 | IMUL32I | IMUL32I24 | IMUL32IS24
						| IMUL32U24
						| ISET | LG2 | LLD | LST | MOV | MOV32 | MVC | MVI | NOP 
						| NOT | NOTS | OR | ORS | R2A | R2G | R2GU16U8 | RCP | RCP32 | RET | RRO 
						| RSQ | SHL | SHR | SIN | SSY | XOR | XORS | S2R | SASS_LD | STS 
						| LDS | SASS_ST | EXIT | BAR | IMIN | IMAX | A2R | FMAX | FMIN 
						| TEX | TEX32 | C2R | BRK | R2C | IADDCARRY | VOTE
						;

/* [한국어] PBK(Pre-Break) 명령어: 뒤따르는 16진수를 브레이크 타겟 레이블로 저장. */
pbkInstruction	:	PBK {
							debug_print($1); instEntry->setBase($1); g_instList->add(instEntry);
						} HEXLITERAL {
							char* tempInput = $3;
							char* tempLabel = new char[12];
							tempLabel[0] = 'l';
							tempLabel[1] = '0';
							tempLabel[2] = 'x';
							for(int i=0; i<(10-strlen(tempInput)); i++)
							{
								tempLabel[3+i] = '0';
							}
							for(int i=(13-strlen(tempInput)); i<11; i++)
							{
								tempLabel[i] = tempInput[i-(11-strlen(tempInput))];
							}
							tempLabel[11] = '\0';
							g_instList->getListEnd().addOperand(tempLabel);
							g_instList->addCubojdumpLabel(tempLabel);
						}
					;

/* [한국어] 분기/호출 명령어: BRA/CAL 뒤의 16진수를 l0x... 형태 레이블로 변환. */
branchInstructions	: BRA {debug_print($1); instEntry->setBase($1); g_instList->add(instEntry);} instructionPredicate HEXLITERAL
					{ debug_print($4);
					  char* tempInput = $4;
					  char* tempLabel = new char[12];
					  tempLabel[0] = 'l';
					  tempLabel[1] = '0';
					  tempLabel[2] = 'x';
					  for(int i=0; i<(10-strlen(tempInput)); i++)
					  {
						tempLabel[3+i] = '0';
					  }
					  for(int i=(13-strlen(tempInput)); i<11; i++)
					  {
						tempLabel[i] = tempInput[i-(11-strlen(tempInput))];
					  }
					  tempLabel[11] = '\0';
					  g_instList->getListEnd().addOperand(tempLabel);
					  g_instList->addCubojdumpLabel(tempLabel);}
				| BRA {debug_print($1); instEntry->setBase($1); g_instList->add(instEntry);} HEXLITERAL
					{ debug_print($3);
					  char* tempInput = $3;
					  char* tempLabel = new char[12];
					  tempLabel[0] = 'l';
					  tempLabel[1] = '0';
					  tempLabel[2] = 'x';
					  for(int i=0; i<(10-strlen(tempInput)); i++)
					  {
						tempLabel[3+i] = '0';
					  }
					  for(int i=(13-strlen(tempInput)); i<11; i++)
					  {
						tempLabel[i] = tempInput[i-(11-strlen(tempInput))];
					  }
					  tempLabel[11] = '\0';
					  g_instList->getListEnd().addOperand(tempLabel);
					  g_instList->addCubojdumpLabel(tempLabel);}
				| CAL {debug_print($1); instEntry->setBase($1); g_instList->add(instEntry);} HEXLITERAL
					{ debug_print($3);
					  char* tempInput = $3;
					  char* tempLabel = new char[12];
					  tempLabel[0] = 'l';
					  tempLabel[1] = '0';
					  tempLabel[2] = 'x';
					  for(int i=0; i<(10-strlen(tempInput)); i++)
					  {
						tempLabel[3+i] = '0';
					  }
					  for(int i=(13-strlen(tempInput)); i<11; i++)
					  {
						tempLabel[i] = tempInput[i-(11-strlen(tempInput))];
					  }
					  tempLabel[11] = '\0';
					  g_instList->getListEnd().addOperand(tempLabel);
					  g_instList->addCubojdumpLabel(tempLabel);}
				
				| CAL {debug_print($1); instEntry->setBase($1); g_instList->add(instEntry);} DOTNOINC HEXLITERAL
					{ debug_print($4);
					  char* tempInput = $4;
					  char* tempLabel = new char[12];
					  tempLabel[0] = 'l';
					  tempLabel[1] = '0';
					  tempLabel[2] = 'x';
					  for(int i=0; i<(10-strlen(tempInput)); i++)
					  {
						tempLabel[3+i] = '0';
					  }
					  for(int i=(13-strlen(tempInput)); i<11; i++)
					  {
						tempLabel[i] = tempInput[i-(11-strlen(tempInput))];
					  }
					  tempLabel[11] = '\0';
					  g_instList->getListEnd().addOperand(tempLabel);
					  g_instList->addCubojdumpLabel(tempLabel);}

				;

modifierList	: modifier modifierList
					/*| modifier */
					|
					;

/* [한국어] 수정자: 타입 수정자 또는 다양한 베이스 수정자를 IR에 추가. */
modifier	: opTypes	{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			| DOTBEXT		{ g_instList->getListEnd().addBaseModifier(".bext"); }
			| DOTS			{ g_instList->getListEnd().addBaseModifier(".s"); }
			| DOTSFU		{ g_instList->getListEnd().addBaseModifier(".sfu"); }
			| DOTTRUNC		{ g_instList->getListEnd().addBaseModifier(".rz"); }
			| DOTCEIL		{ g_instList->getListEnd().addBaseModifier(".rp"); }
			| DOTFLOOR		{ g_instList->getListEnd().addBaseModifier(".rm"); }
			| DOTX			{ g_instList->getListEnd().addBaseModifier(".x"); }
			| DOTE			{ g_instList->getListEnd().addBaseModifier(".e"); }
			| DOTRED		{ g_instList->getListEnd().addBaseModifier(".red"); }
			| DOTPOPC		{ g_instList->getListEnd().addBaseModifier(".popc"); }
			| DOTIR			{ g_instList->getListEnd().addBaseModifier(".ir"); }
			| DOTUN			{ /*g_instList->getListEnd().addBaseModifier(".un"); */}
			| DOTNODEP		{ /*g_instList->getListEnd().addBaseModifier(".nodep"); */}
			| DOTANY		{ g_instList->getListEnd().addBaseModifier(".any"); }
			| DOTALL		{ g_instList->getListEnd().addBaseModifier(".all"); }
			;

/* [한국어] 타입 수정자: opTypes 규칙에 매칭되면 addTypeModifier가 호출됨
 * (실제 동작은 sass.l에서 이미 처리되도록 주석 처리되어 있으나, 규칙은 유지). */
opTypes		: DOTF16	//{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			| DOTF32	//{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			| DOTF64	//{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			| DOTS8		//{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			| DOTS16	//{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			| DOTS32	//{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			| DOTS64	//{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			| DOTS128	//{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			| DOTU8		//{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			| DOTU16	//{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			| DOTU32	//{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			| DOTU24	//{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			| DOTU64	//{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			| DOTHI		//{ debug_print($1); g_instList->getListEnd().addTypeModifier($1);}
			;

operandList	: operandList { debug_print(" "); } /*COMMA*/ operand	{}
				/*| { debug_print(" "); } operand		{}*/
				|
				;

/* [한국어] 피연산자: 레지스터, 메모리, 즉시값, 프레디케이트, 추가 수정자 등. */
operand		: registerlocation
			| PIPE registerlocation PIPE	{ g_instList->getListEnd().addBaseModifier(".abs"); }
			| TILDE registerlocation
			| LEFTBRACKET instructionPredicate RIGHTBRACKET
			| memorylocation opTypes { debug_print($2); g_instList->getListEnd().addTypeModifier($2);}
			| memorylocation
			| immediateValue
			| extraModifier
			| operandPredicate
			| preOperand
			;
	/* Register of the format [R0] will be converted to R0 */
	/* regMod will be also ignored */
registerlocation	: REGISTER regMod	{ debug_print($1); g_instList->addCuobjdumpRegister($1);}
				| OSQBRACKET REGISTER CSQBRACKET	{ debug_print($1); debug_print($2); debug_print($3); g_instList->addCuobjdumpRegister($2);}
				| REGISTERLO	{ debug_print($1); g_instList->addCuobjdumpRegister($1,true);}
				| REGISTERHI	{ debug_print($1); g_instList->addCuobjdumpRegister($1,true);}
				| SREGISTER		{ debug_print($1); g_instList->addCuobjdumpRegister($1,false);}
				| OFFSETREGISTER	{ debug_print($1); g_instList->addCuobjdumpRegister($1);}
				| PREDREGISTER PREDREGISTER2	{ debug_print($1); debug_print(" "); debug_print($2); g_instList->addCuobjdumpDoublePredReg($1, $2);}
				| PREDREGISTER REGISTER	{ debug_print($1); debug_print(" "); debug_print($2); g_instList->addCuobjdumpDoublePredReg($1, $2);}
				/*| REGISTER PREDREGISTER3 { debug_print($1); debug_print(" "); debug_print($2); g_instList->addCuobjdumpRegister($1); debug_print("WEIRD CASE\n");}*/
				;

regMod		: DOTCC
				|
				;


/* [한국어] 메모리 위치 피연산자: 공유/절대공유/전역/상수/로컬 메모리.
 * ABSSMEMLOCATION은 |g|를 제거하고 .abs 수정자 추가. */
memorylocation	: SMEMLOCATION	{ debug_print($1); g_instList->addCuobjdumpMemoryOperand($1,1);}
			|	ABSSMEMLOCATION {
					debug_print($1);
					char* input = $1;
					char* temp = new char[99];
					temp[0] = input[1];
					unsigned i=1;
					while (i < strlen(input)-2) {
						temp[i] = input[i+2];
						i++;
					}
					g_instList->addCuobjdumpMemoryOperand(temp,1);
					g_instList->getListEnd().addBaseModifier(".abs");
				}
			| GMEMLOCATION	{ debug_print($1); g_instList->addCuobjdumpMemoryOperand($1,2);}
			| CMEMLOCATION	{ debug_print($1); g_instList->addCuobjdumpMemoryOperand($1,0);}
			| LMEMLOCATION	{ debug_print($1); g_instList->addCuobjdumpMemoryOperand($1,3);}
			;

/* [한국어] 즉시값 피연산자: 식별자(라벨 등) 또는 16진수 리터럴. */
immediateValue	: IDENTIFIER { debug_print($1); g_instList->getListEnd().addOperand($1);}
				| HEXLITERAL { debug_print($1); g_instList->getListEnd().addOperand($1);}
				;

/* [한국어] 추가 비교 수정자: EQ/EQU/GE/... 를 base modifier로 추가. */
extraModifier	: EQ	{ debug_print($1); g_instList->getListEnd().addBaseModifier($1);} 
			| EQU	{ debug_print($1); g_instList->getListEnd().addBaseModifier($1);}
			| GE	{ debug_print($1); g_instList->getListEnd().addBaseModifier($1);}
			| GEU	{ debug_print($1); g_instList->getListEnd().addBaseModifier($1);}
			| GT	{ debug_print($1); g_instList->getListEnd().addBaseModifier($1);}
			| GTU	{ debug_print($1); g_instList->getListEnd().addBaseModifier($1);}
			| LE	{ debug_print($1); g_instList->getListEnd().addBaseModifier($1);}
			| LEU	{ debug_print($1); g_instList->getListEnd().addBaseModifier($1);}
			| LT	{ debug_print($1); g_instList->getListEnd().addBaseModifier($1);}
			| LTU	{ debug_print($1); g_instList->getListEnd().addBaseModifier($1);}
			| NE	{ debug_print($1); g_instList->getListEnd().addBaseModifier($1);}
			| NEU	{ debug_print($1); g_instList->getListEnd().addBaseModifier($1);}
			;

/* [한국어] 명령어 predicate: PREDREGISTER3 + 선택적 predicate modifier. */
instructionPredicate	: PREDREGISTER3	predicateModifier {debug_print($1); debug_print($2);
									g_instList->getListEnd().setPredicate($1);
									g_instList->getListEnd().addPredicateModifier($2);}
							| PREDREGISTER3 {debug_print($1); g_instList->getListEnd().setPredicate($1);}
				;

/* [한국어] 피연산자 위치의 predicate(드문 경우). */
operandPredicate	:	PREDREGISTER3	predicateModifier {
								debug_print($1); 
								debug_print($2);
								//g_instList->getListEnd().addOperand($1);
								g_instList->getListEnd().setPredicate($1);
								g_instList->getListEnd().addPredicateModifier($2);
								/*May be the modifier needs to be added too*/
							}
						|	PREDREGISTER3 {
								debug_print("HELLO: "); 
								debug_print($1); 
								g_instList->getListEnd().addOperand($1);
							}
						;


/* [한국어] 사전 피연산자 수정자: EX2/SIN/COS를 base modifier로 추가(SFU 관련). */
preOperand	: EX2	{ debug_print($1); g_instList->getListEnd().addBaseModifier("ex2");}
			| SIN	{ debug_print($1); g_instList->getListEnd().addBaseModifier("sin");}
			| COS	{ debug_print($1); g_instList->getListEnd().addBaseModifier("cos");}
			;

/* [한국어] predicate modifier 토큰들(일부는 현재 무시됨). */
predicateModifier	: DOTEQ	{ }
				| DOTEQU	{ }
				| DOTFALSE	{ }
				| DOTGE	{ }
				| DOTGEU	{ }
				| DOTGT	{ }
				| DOTGTU	{ }
				| DOTLE	{ }
				| DOTLEU	{ }
				| DOTLT	{ }
				| DOTLTU	{ }
				| DOTNE	{ }
				| DOTNEU	{ }
				| DOTNSF	{ }
				| DOTSF	{ }
				| DOTCARRY	{ }
				;

%%

/*support c++ functions go here*/

/*
 * [한국어]
 * debug_print - 파서 디버그 출력 함수
 *
 * @s: 출력할 문자열
 *
 * 기본적으로 주석 처리되어 아무것도 출력하지 않는다.
 * 필요시 printf 주석을 해제하여 파싱 과정을 추적할 수 있다.
 */
void debug_print( const char *s )
{
	// uncomment to debug
	// printf("%s",s);
}
