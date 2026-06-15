/*
 * [한국어 설명] BookSim 설정 파일용 Flex/Lex 어휘 분석기 (config.l)
 *
 * === 파일의 역할 ===
 * 이 파일은 BookSim NoC 시뮬레이터의 설정 파일(예: gpgpusim.config)을
 * 읽어 들여 토큰(token) 단위로 분해하는 Flex 어휘 분석기(lexer)를 정의한다.
 * 정수, 실수, 문자열, 식별자, 주석, 중괄호 등을 구분하여
 * config.y의 Bison 문법 분석기(parser)가 이해할 수 있는 토큰으로 변환한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 설정 파일 파싱 파이프라인의 최전방:
 *   gpgpusim.config 텍스트 → [config.l lexer] → 토큰 시퀀스
 *                                         → [config.y parser] → Configuration 객체
 *                                         → Network/Router 초기화
 * 실행 컨텍스트: 시뮬레이터 초기화 단계, 1회 실행.
 *
 * === 타 모듈과의 연결 ===
 * - config.y         : Bison parser — lexer가 반환한 토큰들로 문법 분석 수행.
 * - config_utils.hpp : Configuration 클래스 — parser가 파싱 결과를 저장하는 저장소.
 * - booksim_config.hpp/IntersimConfig : 파싱된 설정값을 사용하는 소비자.
 * - icnt_wrapper.cc  : IntersimConfig::ParseFile() 호출로 설정 파일 파싱 시작.
 *
 * === 주요 토큰/규칙 요약 ===
 * - INT:    10진/16진 정수 (예: 32, 0x20)
 * - FLOAT:  실수 (예: 3.14, 2.5e-3)
 * - STR:    큰따옴표 또는 작은따옴표로 감싼 문자열
 * - ID:     알파벳/숫자/밑줄로 구성된 식별자 (파라미터 이름)
 * - [,;={}] : 리스트/블록 구조용 구분자
 * - #, //  : 주석 (lexer가 무시)
 * - EOF:    파일 종료
 */

%{

#include "y.tab.h"

static unsigned int lineno = 1; /* [한국어] 현재 파싱 중인 설정 파일의 줄 번호 (에러 보고용) */

void config_error(char * msg, int lineno);
void yyerror(char * msg);

extern int config_input(char *, int);
#undef YY_INPUT
/* [한국어] Flex의 기본 입력 매크로를 BookSim 커스텀 config_input()으로 교체 */
#define YY_INPUT(b, r, ms) (r = config_input(b, ms))

%}

/* [한국어] 10진 숫자 하나의 정규식 정의 */
Digit [0-9]
/* [한국어] 실수의 지수 부분 (e/E + 부호 + 숫자) */
Exponent [eE][+-]?{Digit}+
/* [한국어] 소수점 또는 지수를 포함한 실수 상수 */
DblConst ({Digit}*\.)?{Digit}+{Exponent}?
/* [한국어] 파라미터 이름/문자열 상수: 첫 글자는 문자/밑줄/특수문자, 이후 영숫자/특수문자 */
StrConst [A-Za-z_\-/\.][A-Za-z0-9_\-/,\.+(\{\,)\}]*

%%

   /* Ignore comments and all spaces */
   /* [한국어] 주석과 공백 문자는 모두 무시(토큰 반환 안 함) */

/* [한국어] //로 시작하는 한 줄 주석 무시 */
\/\/[^\n]* ;
/* [한국어] 스페이스/탭/캐리지 리턴 무시 */
[ \t\r]*   ;

/* [한국어] 개행 문자를 만나면 줄 번호 증가 — 에러 메시지에서 사용 */
\n         { lineno++; }

   /* Commands */
   /* [한국어] 설정 파일에서 등장하는 각종 토큰 규칙 */

/* [한국어] 중괄호로 둘러싼 리스트(예: {a,b,c})를 STR 토큰으로 반환 */
\{[A-Za-z0-9_\-\.(\{\,)\]+(\,[A-Za-z0-9_\-\.(\{\,)\]+)*\} { yylval.name = strdup( yytext ); return STR; }

/* [한국어] 부호 있는 정수를 NUM 토큰으로 반환 (yylval.num에 저장) */
-?[0-9]+     { yylval.num = atoi( yytext ); return NUM; }

/* [한국어] 소수점 형태 실수를 FNUM 토큰으로 반환 */
-?[0-9]*\.[0-9]+ { yylval.fnum = atof( yytext ); return FNUM; }

/* [한국어] 지수 포함 등 DblConst 형태의 실수를 FNUM 토큰으로 반환 */
-?{DblConst}  { yylval.fnum = atof( yytext ); return FNUM;}

/* [한국어] 식별자/문자열 상수를 STR 토큰으로 반환 */
{StrConst} { yylval.name = strdup( yytext ); return STR; }

/* [한국어] 위 규칙에 매칭되지 않는 단일 문자는 그대로 ASCII 토큰으로 반환(구분자 [,;={}] 등) */
.          { return yytext[0]; }

%%

/*
 * [한국어]
 * yyerror — lexer/parser에서 오류 발생 시 호출되는 에러 핸들러.
 * @msg: Bison/Flex에서 전달하는 에러 메시지.
 * 현재 줄 번호(lineno)를 config_error()에 넘겨 설정 파일 내 위치를 출력한다.
 */
void yyerror( char * msg )
{
  config_error( msg, lineno ); /* [한국어] 현재 lexer가 기록한 줄 번호를 함께 에러 처리 */
}

/*
 * [한국어]
 * yywrap — Flex가 입력 파일 끝(EOF)에 도달했을 때 호출.
 * @return: 1이면 입력 종료(더 이상 파일 없음), 0이면 추가 입력 있음.
 * BookSim 설정 파일은 단일 파일만 파싱하므로 항상 1을 반환한다.
 */
int yywrap()
{
  return 1; /* [한국어] EOF — 추가 입력 없음 */
}
