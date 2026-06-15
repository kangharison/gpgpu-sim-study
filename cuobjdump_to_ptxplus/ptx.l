/*
 * [한국어 설명] PTX(BIN) Lex/Flex 어휘 분석기 (ptx.l)
 *
 * === 파일의 역할 ===
 * cuobjdump가 출력한 PTX 이진 코드(또는 .ptx 텍스트)를 읽어
 * Bison(ptx.y)이 사용할 토큰으로 변환하는 Flex lexer 정의 파일이다.
 * GPGPU-Sim 메인 PTX lexer(src/cuda-sim/ptx.l)를 기반으로 하되,
 * cuobjdump_to_ptxplus 용으로 축소/변형되어 있으며, SASS 변환 과정에서
 * 보조적인 PTX 헤더 정보를 추출하는 데 사용된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 *   cuobjdump_to_ptxplus.cc
 *     → ptx_parse() (ptx.y가 생성)
 *         → ptx.l (이 파일)이 PTX 텍스트를 토큰화
 *             → ptx.y가 텍스처/상수 등 선언 처리
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - ptx_parser.hh : Bison이 생성한 토큰 정의
 * 이 파일에 의존하는 모듈:
 *   - ptx.y         : lexer 토큰을 문법 규칙에서 사용
 *   - 빌드 시스템    : flex ptx.l → ptx.yy.cc 생성
 *
 * === 주요 함수/구조체 요약 ===
 * Flex 옵션:
 *   noyywrap  : 입력 종료 시 yywrap 호출 안 함
 *   yylineno  : 줄 번호 추적
 *   never-interactive : 버퍼링 설정
 *   nounput   : unput 함수 비생성
 *   bison-bridge : Bison %pure-parser/push-parser와의 연동
 *   reentrant : 재진입 가능한 scanner 생성
 * 주요 토큰 범주:
 *   - PTX 키워드: VERSION, TARGET, ADDRESS_SIZE, ENTRY, FUNC, FILE, etc.
 *   - 메모리/주소 공간: REG, SREG, GLOBAL, LOCAL, PARAM, SHARED, CONST, etc.
 *   - 데이터 타입: PRED, B8..B128, U8..U64, S8..S64, F16..F64, etc.
 *   - 벡터 타입: V2, V4, V8
 *   - 레지스터: REGISTER_DECL, VECTOR_REGISTER_DECL, etc.
 *   - 주석/공백/개행 무시
 */

%{
#include "ptx_parser.h"
#include "ptx_parser.hh"
#include <string.h>
#include <stdio.h>

	/* macros to enable bison to use flex semantic value type */
#define yylval ptx_lval

#define LINEBUF_SIZE 4096
char linebuf[LINEBUF_SIZE];

extern int ptx_error( const char *s );

%}
%option noyywrap
%option yylineno
%option never-interactive
%option nounput
%option bison-bridge
%option reentrant

%{
#define YY_USER_ACTION { char *c; strncpy(linebuf, yytext, LINEBUF_SIZE);
                       if ((c=strchr(linebuf,'\n'))) *c = 0; }
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
pipe		\|

identifier	({alpha}({alphanum}|["_$"])*)|([_$%]({alphanum}|["_$"])+)
hexliteral	0[xX]{hexdigit}+U?
octliteral	0[0-7]+U?
binliteral	0[bB][01]+U?
decliteral	([1-9][0-9]*U?)|0U?

%%

	/* PTX key words */
".version"			return VERSION;
".target"			return TARGET;
".address_size"		return ADDRESS_SIZE;
".entry"			return ENTRY;
".func"				return FUNC;
".file"				return FILE;
".loc"				return LOC;

	/* data types */
".sreg"				return SREG;
".reg"				return REG;
".global"			return GLOBAL;
".local"			return LOCAL;
".param"			return PARAM;
".shared"			return SHARED;
".const"			return CONST;
".tex"				return TEX;
".surf"				return SURF;

".v2"				return V2;
".v4"				return V4;
".v8"				return V8;

".predicate"|	".pred" return PRED;

".b8"				return B8;
".b16"				return B16;
".b32"				return B32;
".b64"				return B64;
".b128"				return B128;

".u8"				return U8;
".u16"				return U16;
".u32"				return U32;
".u64"				return U64;

".s8"				return S8;
".s16"				return S16;
".s32"				return S32;
".s64"				return S64;

".f16"				return F16;
".f32"				return F32;
".f64"				return F64;

".align" 			return ALIGN;

".callprototype" return CALLPROTOTYPE;
".calltargets" 	 return CALLTARGETS;
".parametriccall" return PARAMETRICCALL;

	/* identifiers */
{identifier}	{ yylval->string_value = strdup(yytext); return IDENTIFIER; }

	/* literals */
{hexliteral}	{ yylval->string_value = strdup(yytext); return HEXLITERAL; }
{octliteral}	{ yylval->string_value = strdup(yytext); return OCTLITERAL; }
{binliteral}	{ yylval->string_value = strdup(yytext); return BINLITERAL; }
{decliteral}	{ yylval->string_value = strdup(yytext); return DECLITERAL; }

	/* registers */
%r{numeric}+			{ yylval->string_value = strdup(yytext); return REGISTER_DECL; }
%r{numeric}+<1>			{ yylval->string_value = strdup(yytext); return VECTOR_REGISTER_DECL; }
%r{numeric}+<2>			{ yylval->string_value = strdup(yytext); return VECTOR_REGISTER_DECL; }
%r{numeric}+<3>			{ yylval->string_value = strdup(yytext); return VECTOR_REGISTER_DECL; }
%r{numeric}+<{numeric}+>	{ yylval->string_value = strdup(yytext); return VECTOR_REGISTER_DECL; }
%p{numeric}+			{ yylval->string_value = strdup(yytext); return PRED_REGISTER_DECL; }
%pd{numeric}+			{ yylval->string_value = strdup(yytext); return PRED_REGISTER_DECL; }

	/* special registers */
"%tid"|	"%ntid"|	"%ctaId"|	"%nctaId"|	"%smid"|	"%nwarpId"|	"%warpId"|
"%laneId"|	"%clock"|	"%clock64"|	"%pm0"|	"%pm1"|	"%pm2"|	"%pm3" { yylval->string_value = strdup(yytext); return SPECIAL_REGISTER; }


	/* punctuation */
";" return SEMICOLON;
"," return COMMA;
"{" return LBRACE;
"}" return RBRACE;
"[" return LBRACKET;
"]" return RBRACKET;
"(" return LPAREN;
")" return RPAREN;
"<" return LANGLE;
">" return RANGLE;
"@" return ATSIGN;
"!" return EXCLAMATION;
"|" return PIPE;

"<=" return LEQ;
">=" return GEQ;
"==" return EQ;
"!=" return NEQ;
"||" return OROR;
"&&" return ANDAND;
"<<" return SHL;
">>" return SHR;

"+" return PLUS;
"-" return MINUS;
"*" return STAR;
"/" return DIV;
"%" return MOD;
"&" return AND;
"^" return XOR;
"=" return ASSIGN;

"~" return TILDE;
"." return PERIOD;


	/*Discard spaces and tabs*/
{whitespace}+

	/*Discard comments*/
"//"{anycharbutnewline}*

	/*Discard newlines*/
{newline}+

<<EOF>>	return 0;

. ptx_error("Invalid token");

%%

/* [한국어] 사용자 정의 lexer 보조 함수 영역. 현재는 별도의 추가 함수가 없다. */
