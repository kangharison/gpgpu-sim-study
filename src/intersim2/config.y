/*
 * [한국어 설명] BookSim 설정 파일용 Bison 문법 분석기 (config.y)
 *
 * === 파일의 역할 ===
 * 이 파일은 config.l lexer가 생성한 토큰(STR, NUM, FNUM)들을 받아
 * BookSim 설정 파일의 문법을 분석하는 Bison parser를 정의한다.
 * 지원하는 설정 문장은 다음 세 가지 형태뿐이다:
 *   parameter = "string_value" ;
 *   parameter = 42 ;
 *   parameter = 3.14 ;
 * 파싱 결과는 config_assign_string/int/float() 함수를 통해
 * Configuration 객체의 _str_map / _int_map / _float_map에 저장된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 설정 파일 파싱 파이프라인의 두 번째 단계:
 *   gpgpusim.config → [config.l lexer] → 토큰 → [config.y parser]
 *                                              → Configuration 낸부 맵 저장
 *                                              → Network/Router 초기화
 * 실행 컨텍스트: 시뮬레이터 초기화 단계, 1회 실행.
 *
 * === 타 모듈과의 연결 ===
 * - config.l         : Flex lexer — NUM, FNUM, STR 토큰과 yylval 값을 제공.
 * - config_utils.cpp : config_assign_string/int/float() 구현 — Configuration에 값 저장.
 * - config_utils.hpp : Configuration 클래스 선언 — _str_map, _int_map, _float_map.
 * - icnt_wrapper.cc  : IntersimConfig::ParseFile() 호출로 파싱 시작.
 *
 * === 주요 문법/함수 요약 ===
 * - commands : command의 반복(재귀) — 여러 줄의 설정을 순차 처리.
 * - command  : STR '=' STR ';'  → config_assign_string()
 *              STR '=' NUM ';'  → config_assign_int()
 *              STR '=' FNUM ';' → config_assign_float()
 * - yylex()  : config.l에 의해 생성된 lexer 진입점.
 * - yyerror(): 파싱 에러 발생 시 config.l의 yyerror()를 통해 config_error() 호출.
 */

%{

int  yylex(void); /* [한국어] config.l(Flex)에 의해 생성된 lexer 진입점 선언 */
void yyerror(char * msg); /* [한국어] Bison 파싱 에러 시 호출될 에러 핸들러 선언 */

/* [한국어] 파싱된 값을 Configuration 객체의 해당 맵에 저장하는 외부 함수들 */
void config_assign_string( char const * field, char const * value ); /* [한국어] 문자열 값 저장: _str_map[field] = value */
void config_assign_int( char const * field, int value );             /* [한국어] 정수 값 저장: _int_map[field] = value */
void config_assign_float( char const * field, double value );        /* [한국어] 실수 값 저장: _float_map[field] = value */

#ifdef _WIN32
#pragma warning ( disable : 4102 ) /* [한국어] MSVC: 레이블 경고 억제 */
#pragma warning ( disable : 4244 ) /* [한국어] MSVC: 형 변환 경고 억제 */
#endif

%}

/* [한국어] yylval 공용체: lexer가 반환할 토큰 값의 타입 정의 */
%union {
  char   *name; /* [한국어] STR 토큰에 사용 — 문자열(식별자/문자열 상수) */
  int    num;   /* [한국어] NUM 토큰에 사용 — 정수 값 */
  double fnum;  /* [한국어] FNUM 토큰에 사용 — 실수 값 */
}

%token <name> STR /* [한국어] 문자열/식별자 토큰. yylval.name으로 접근. */
%token <num>  NUM /* [한국어] 정수 토큰. yylval.num으로 접근. */
%token <fnum> FNUM /* [한국어] 실수 토큰. yylval.fnum으로 접근. */

%%

/* [한국어] commands: 0개 이상의 command를 재귀적으로 파싱 */
commands : commands command /* [한국어] 이미 파싱된 commands 뒤에 새 command 추가 */
         | command         /* [한국어] 첫 번째 command */
;

/* [한국어] command: "이름 = 값 ;" 형태의 한 줄 설정 */
command : STR '=' STR ';'   { config_assign_string( $1, $3 ); free( $1 ); free( $3 ); } /* [한국어] 문자열 파라미터 저장 후 동적 문자열 해제 */
        | STR '=' NUM ';'   { config_assign_int( $1, $3 ); free( $1 ); }               /* [한국어] 정수 파라미터 저장 후 키 문자열 해제 */
        | STR '=' FNUM ';'  { config_assign_float( $1, $3 ); free( $1 ); }             /* [한국어] 실수 파라미터 저장 후 키 문자열 해제 */
;

%%
