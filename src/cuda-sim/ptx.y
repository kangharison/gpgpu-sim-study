/*
 * [한국어 설명] PTX 문법 파서(Bison/Yacc) 정의 (ptx.y)
 *
 * === 파일의 역할 ===
 * 이 파일은 Flex 토크나이저(ptx.l)가 생성한 토큰 스트림을 받아 PTX(Parallel Thread
 * eXecution) 어셈블리의 문법 구조를 해석하는 Bison(Yacc) 문법 파일이다. 변수 선언,
 * 함수 정의, 명령어, 피연산자, 옵션 등의 문법 규칙을 정의하고, 각 규칙이 reduce될
 * 때마다 ptx_recognizer(ptx_parser.cc)의 semantic action 함수를 호출하여 PTX IR
 * (function_info, symbol_table, ptx_instruction 등)를 조립한다. 빌드 시 Bison이
 * ptx.tab.c/ptx.tab.h를 생성하며, ptx_parse() 진입점이 여기서 나온다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CUDA Application → libcuda 인터셉트 → gpgpusim_entrypoint.cc
 *   → gpgpu_context::init_parser()
 *       → ptx_lex_init() → ptx.l (ptx_lex) → [이 파일: ptx_parse()] → Yacc reduce 액션
 *           → ptx_recognizer (ptx_parser.cc) → function_info / symbol_table / ptx_instruction
 *               → cuda-sim/ 기능 시뮬레이션 → gpgpu-sim/ 타이밍 시뮬레이션
 * 실행 컨텍스트: 호스트 유저스페이스, 커널 실행 전 초기화 단계에서 1회 수행.
 * 사이클-레벨 시뮬레이션과 무관한 전처리 단계이다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존하는 모듈: ptx.l (Flex lexer, 토큰 스트림 공급), ptx_parser.h/ptx_parser.cc
 *   (semantic action 구현체 ptx_recognizer), ptx_ir.h (생성되는 PTX IR 타입),
 *   opcodes.h (opcode enum), gpgpu_context.h (전역 상태)
 * - 이 파일에 의존하는 모듈: 빌드 시스템(Bison이 ptx.tab.c/h 생성), ptx_parser.cc
 *   (ptx.tab.h의 토큰 번호 사용), cuda-sim.cc (최종 PTX IR 사용)
 * - 데이터 흐름: PTX 소스 문자열 → ptx.l 토큰 → ptx.y 문법 reduce →
 *   recognizer->add_*() 호출 → g_instructions / symbol_table 축적 →
 *   end_function()에서 function_info::add_inst() → gpgpu_ptx_assemble()
 *
 * === 주요 문법 규칙 요약 ===
 * - input: 최상위 규칙; directive_statement와 function_defn을 반복
 * - function_defn: 함수 선언 + 문법 본문; start_function / end_function 경계
 * - function_decl: 함수 헤더(.entry/.func/.extern) 파싱; param_list 처리
 * - statement_list: 변수 선언(directive_statement)과 명령어(instruction_statement) 반복
 * - variable_declaration: 변수 선언 + 선택적 초기화
 * - instruction_statement: 레이블 또는 프레디케이트가 붙은 명령어
 * - operand / vector_operand / memory_operand / literal_operand / address_expression:
 *   다양한 PTX 피연산자 형태 처리
 * - option_list / compare_spec / rounding_mode / atomic_operation_spec:
 *   명령어 수식어(옵션) 처리
 */

/*
Copyright (c) 2009-2011, Tor M. Aamodt
The University of British Columbia
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
Neither the name of The University of British Columbia nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

%{
typedef void * yyscan_t;           // [한국어] Flex 재진입 스캐너 핸들 타입
class ptx_recognizer;              // [한국어] semantic action을 담당하는 클래스 전방 선언
#include "../../libcuda/gpgpu_context.h"  // [한국어] gpgpu_context 전역 상태
%}

%define api.pure full              // [한국어] 순수(pure) 재진입 파서 생성
%parse-param {yyscan_t scanner}    // [한국어] yyparse의 첫 번째 인자: Flex 스캐너 핸들
%parse-param {ptx_recognizer* recognizer} // [한국어] yyparse의 두 번째 인자: ptx_recognizer 포인터
%lex-param {yyscan_t scanner}      // [한국어] yylex 호출 시 Flex 스캐너 전달
%lex-param {ptx_recognizer* recognizer}   // [한국어] yylex 호출 시 recognizer 전달

%union {
  double double_value;             // [한국어] double 리터럴 / .version 번호 등
  float  float_value;              // [한국어] float 리터럴
  int    int_value;                // [한국어] 정수 리터럴 / opcode / 옵션 / 토큰 값
  char * string_value;             // [한국어] IDENTIFIER, STRING 등 문자열 포인터
  void * ptr_value;                // [한국어] symbol_table 포인터 등 불투명 포인터
}

%token <string_value> STRING       // [한국어] 문자열 리터럴 토큰
%token <int_value>  OPCODE         // [한국어] PTX opcode 토큰 (ADD_OP, LD_OP 등)
%token <int_value>  WMMA_DIRECTIVE // [한국어] WMMA 지시어 토큰
%token <int_value>  LAYOUT         // [한국어] WMMA 레이아웃(ROW/COL) 토큰
%token <int_value>  CONFIGURATION  // [한국어] WMMA 설정(M16N16K16 등) 토큰
%token  ALIGN_DIRECTIVE            // [한국어] .align
%token  BRANCHTARGETS_DIRECTIVE    // [한국어] .branchtargets
%token  BYTE_DIRECTIVE             // [한국어] .byte
%token  CALLPROTOTYPE_DIRECTIVE    // [한국어] .callprototype
%token  CALLTARGETS_DIRECTIVE      // [한국어] .calltargets
%token  <int_value> CONST_DIRECTIVE // [한국어] .const[bank]
%token  CONSTPTR_DIRECTIVE         // [한국어] .constptr
%token  PTR_DIRECTIVE              // [한국어] .ptr
%token  ENTRY_DIRECTIVE            // [한국어] .entry
%token  EXTERN_DIRECTIVE           // [한국어] .extern
%token  FILE_DIRECTIVE             // [한국어] .file
%token  FUNC_DIRECTIVE             // [한국어] .func
%token  GLOBAL_DIRECTIVE           // [한국어] .global
%token  LOCAL_DIRECTIVE            // [한국어] .local
%token  LOC_DIRECTIVE              // [한국어] .loc
%token  MAXNCTAPERSM_DIRECTIVE     // [한국어] .maxnctapersm
%token  MAXNNREG_DIRECTIVE         // [한국어] .maxnreg
%token  MAXNTID_DIRECTIVE          // [한국어] .maxntid
%token  MINNCTAPERSM_DIRECTIVE     // [한국어] .minnctapersm
%token  PARAM_DIRECTIVE            // [한국어] .param
%token  PRAGMA_DIRECTIVE           // [한국어] .pragma
%token  REG_DIRECTIVE              // [한국어] .reg
%token  REQNTID_DIRECTIVE          // [한국어] .reqntid
%token  SECTION_DIRECTIVE          // [한국어] .section
%token  SHARED_DIRECTIVE           // [한국어] .shared
%token  SREG_DIRECTIVE             // [한국어] .sreg
%token	SSTARR_DIRECTIVE           // [한국어] .sstarr
%token  STRUCT_DIRECTIVE           // [한국어] .struct
%token  SURF_DIRECTIVE             // [한국어] .surf
%token  TARGET_DIRECTIVE           // [한국어] .target
%token  TEX_DIRECTIVE              // [한국어] .tex
%token  UNION_DIRECTIVE            // [한국어] .union
%token  VERSION_DIRECTIVE          // [한국어] .version
%token  ADDRESS_SIZE_DIRECTIVE     // [한국어] .address_size
%token  VISIBLE_DIRECTIVE          // [한국어] .visible
%token  WEAK_DIRECTIVE             // [한국어] .weak
%token  <string_value> IDENTIFIER  // [한국어] 사용자 정의 식별자
%token  <int_value> INT_OPERAND    // [한국어] 정수 상수
%token  <float_value> FLOAT_OPERAND // [한국어] float 상수
%token  <double_value> DOUBLE_OPERAND // [한국어] double 상수
%token  S8_TYPE                    // [한국어] .s8
%token  S16_TYPE                   // [한국어] .s16
%token  S32_TYPE                   // [한국어] .s32
%token  S64_TYPE                   // [한국어] .s64
%token  U8_TYPE                    // [한국어] .u8
%token  U16_TYPE                   // [한국어] .u16
%token  U32_TYPE                   // [한국어] .u32
%token  U64_TYPE                   // [한국어] .u64
%token  F16_TYPE                   // [한국어] .f16
%token  F32_TYPE                   // [한국어] .f32
%token  F64_TYPE                   // [한국어] .f64
%token  FF64_TYPE                  // [한국어] .ff64
%token  B8_TYPE                    // [한국어] .b8
%token  B16_TYPE                   // [한국어] .b16
%token  B32_TYPE                   // [한국어] .b32
%token  B64_TYPE                   // [한국어] .b64
%token  BB64_TYPE                  // [한국어] .bb64
%token  BB128_TYPE                 // [한국어] .bb128
%token  PRED_TYPE                  // [한국어] .pred
%token  TEXREF_TYPE                // [한국어] .texref
%token  SAMPLERREF_TYPE            // [한국어] .samplerref
%token  SURFREF_TYPE               // [한국어] .surfref
%token  V2_TYPE                    // [한국어] .v2
%token  V3_TYPE                    // [한국어] .v3
%token  V4_TYPE                    // [한국어] .v4
%token  COMMA                      // [한국어] ,
%token  PRED                       // [한국어] @ (프레디케이트 접두사)
%token  HALF_OPTION                // [한국어] .half
%token  EXTP_OPTION                // [한국어] .cc
%token  EQ_OPTION                  // [한국어] .eq
%token  NE_OPTION                  // [한국어] .ne
%token  LT_OPTION                  // [한국어] .lt
%token  LE_OPTION                  // [한국어] .le
%token  GT_OPTION                  // [한국어] .gt
%token  GE_OPTION                  // [한국어] .ge
%token  LO_OPTION                  // [한국어] .lo
%token  LS_OPTION                  // [한국어] .ls
%token  HI_OPTION                  // [한국어] .hi
%token  HS_OPTION                  // [한국어] .hs
%token  EQU_OPTION                 // [한국어] .equ
%token  NEU_OPTION                 // [한국어] .neu
%token  LTU_OPTION                 // [한국어] .ltu
%token  LEU_OPTION                 // [한국어] .leu
%token  GTU_OPTION                 // [한국어] .gtu
%token  GEU_OPTION                 // [한국어] .geu
%token  NUM_OPTION                 // [한국어] .num
%token  NAN_OPTION                 // [한국어] .nan
%token  CF_OPTION                  // [한국어] .cf
%token  SF_OPTION                  // [한국어] .sf
%token  NSF_OPTION                 // [한국어] .nsf
%token  LEFT_SQUARE_BRACKET        // [한국어] [
%token  RIGHT_SQUARE_BRACKET       // [한국어] ]
%token  WIDE_OPTION                // [한국어] .wide
%token  <int_value> SPECIAL_REGISTER // [한국어] %tid, %ctaid 등 특수 레지스터
%token  MINUS                      // [한국어] -
%token  PLUS                       // [한국어] +
%token  COLON                      // [한국어] :
%token  SEMI_COLON                 // [한국어] ;
%token  EXCLAMATION                // [한국어] !
%token  PIPE                       // [한국어] |
%token	RIGHT_BRACE                // [한국어] }
%token	LEFT_BRACE                 // [한국어] {
%token	EQUALS                     // [한국어] =
%token  PERIOD                     // [한국어] .
%token  BACKSLASH                  // [한국어] /
%token <int_value> DIMENSION_MODIFIER // [한국어] .x/.y/.z/.0/.1/.2
%token RN_OPTION                   // [한국어] .rn
%token RZ_OPTION                   // [한국어] .rz
%token RM_OPTION                   // [한국어] .rm
%token RP_OPTION                   // [한국어] .rp
%token RNI_OPTION                  // [한국어] .rni
%token RZI_OPTION                  // [한국어] .rzi
%token RMI_OPTION                  // [한국어] .rmi
%token RPI_OPTION                  // [한국어] .rpi
%token UNI_OPTION                  // [한국어] .uni
%token GEOM_MODIFIER_1D            // [한국어] .1d
%token GEOM_MODIFIER_2D            // [한국어] .2d
%token GEOM_MODIFIER_3D            // [한국어] .3d
%token SAT_OPTION                  // [한국어] .sat
%token FTZ_OPTION                  // [한국어] .ftz
%token NEG_OPTION                  // [한국어] .neg
%token SYNC_OPTION                 // [한국어] .sync
%token RED_OPTION                  // [한국어] .red
%token ARRIVE_OPTION               // [한국어] .arrive
%token ATOMIC_POPC                 // [한국어] .popc
%token ATOMIC_AND                  // [한국어] .and
%token ATOMIC_OR                   // [한국어] .or
%token ATOMIC_XOR                  // [한국어] .xor
%token ATOMIC_CAS                  // [한국어] .cas
%token ATOMIC_EXCH                 // [한국어] .exch
%token ATOMIC_ADD                  // [한국어] .add
%token ATOMIC_INC                  // [한국어] .inc
%token ATOMIC_DEC                  // [한국어] .dec
%token ATOMIC_MIN                  // [한국어] .min
%token ATOMIC_MAX                  // [한국어] .max
%token  LEFT_ANGLE_BRACKET         // [한국어] <
%token  RIGHT_ANGLE_BRACKET        // [한국어] >
%token  LEFT_PAREN                 // [한국어] (
%token  RIGHT_PAREN                // [한국어] )
%token  APPROX_OPTION              // [한국어] .approx
%token  FULL_OPTION                // [한국어] .full
%token  ANY_OPTION                 // [한국어] .any
%token  ALL_OPTION                 // [한국어] .all
%token  BALLOT_OPTION              // [한국어] .ballot
%token  GLOBAL_OPTION              // [한국어] .gl
%token  CTA_OPTION                 // [한국어] .cta
%token  SYS_OPTION                 // [한국어] .sys
%token  EXIT_OPTION                // [한국어] .exit
%token  ABS_OPTION                 // [한국어] .abs
%token  TO_OPTION                  // [한국어] .to
%token  CA_OPTION;
%token  CG_OPTION;
%token  CS_OPTION;
%token  LU_OPTION;
%token  CV_OPTION;
%token  WB_OPTION;
%token  WT_OPTION;
%token	NC_OPTION;
%token	UP_OPTION;
%token	DOWN_OPTION;
%token	BFLY_OPTION;
%token	IDX_OPTION;
%token	PRMT_F4E_MODE;
%token	PRMT_B4E_MODE;
%token	PRMT_RC8_MODE;
%token	PRMT_RC16_MODE;
%token	PRMT_ECL_MODE;
%token	PRMT_ECR_MODE;
%token	WRAP_OPTION;
%token	CLAMP_OPTION;
%token	LEFT_OPTION;
%token	RIGHT_OPTION;

%type <int_value> function_decl_header
%type <ptr_value> function_decl

%{
  	#include "ptx_parser.h"      // [한국어] ptx_recognizer 클래스 정의
	#include <stdlib.h>
	#include <string.h>
	#include <math.h>
	void syntax_not_implemented(yyscan_t yyscanner, ptx_recognizer* recognizer);
	int ptx_lex(YYSTYPE * yylval_param, yyscan_t yyscanner, ptx_recognizer* recognizer);
	int ptx_error( yyscan_t yyscanner, ptx_recognizer* recognizer, const char *s );
%}

%%

input:	/* empty */                // [한국어] 빈 입력도 허용
	| input directive_statement  // [한국어] 지시어(변수 선언 등) 반복
	| input function_defn        // [한국어] 함수 정의 반복
	| input function_decl        // [한국어] 함수 선언(프로토타입) 반복
	;

function_defn: function_decl { recognizer->set_symtab($1); recognizer->func_header(".skip"); } statement_block { recognizer->end_function(); }
	// [한국어] 함수 정의: 함수 헤더 + 심볼 테이블 설정 + 함수 본문 + end_function()
	| function_decl { recognizer->set_symtab($1); } block_spec_list { recognizer->func_header(".skip"); } statement_block { recognizer->end_function(); }
	// [한국어] .maxntid 등 블록 스펙이 있는 함수 정의
	;

block_spec: MAXNTID_DIRECTIVE INT_OPERAND COMMA INT_OPERAND COMMA INT_OPERAND {recognizer->func_header_info_int(".maxntid", $2);
										recognizer->func_header_info_int(",", $4);
										recognizer->func_header_info_int(",", $6);
                                                                                recognizer->maxnt_id($2, $4, $6);}
	// [한국어] .maxntid x, y, z: CTA당 최대 스레드 수
	| MINNCTAPERSM_DIRECTIVE INT_OPERAND { recognizer->func_header_info_int(".minnctapersm", $2); printf("GPGPU-Sim: Warning: .minnctapersm ignored. \n"); }
	// [한국어] .minnctapersm: 현재 무시
	| MAXNCTAPERSM_DIRECTIVE INT_OPERAND { recognizer->func_header_info_int(".maxnctapersm", $2); printf("GPGPU-Sim: Warning: .maxnctapersm ignored. \n"); }
	// [한국어] .maxnctapersm: 현재 무시
	;

block_spec_list: block_spec
	| block_spec_list block_spec
	;

function_decl: function_decl_header LEFT_PAREN { recognizer->start_function($1); recognizer->func_header_info("(");} param_entry RIGHT_PAREN {recognizer->func_header_info(")");} function_ident_param { $$ = recognizer->reset_symtab(); }
	// [한국어] (.param ...) 형태의 함수 헤더 + 이름: 커널/함수 파라미터 처리
	| function_decl_header { recognizer->start_function($1); } function_ident_param { $$ = recognizer->reset_symtab(); }
	// [한국어] 파라미터 없는 함수 헤더
	| function_decl_header { recognizer->start_function($1); recognizer->add_function_name(""); recognizer->g_func_decl=0; $$ = recognizer->reset_symtab(); }
	// [한국어] 이름 없는 함수 선언(예: .entry에 이름 생략 시)
	;

function_ident_param: IDENTIFIER { recognizer->add_function_name($1); } LEFT_PAREN {recognizer->func_header_info("(");} param_list RIGHT_PAREN { recognizer->g_func_decl=0; recognizer->func_header_info(")"); }
	// [한국어] 함수 이름 + 파라미터 목록
	| IDENTIFIER { recognizer->add_function_name($1); recognizer->g_func_decl=0; }
	// [한국어] 함수 이름만 (파라미터 없음)
	;

function_decl_header: ENTRY_DIRECTIVE { $$ = 1; recognizer->g_func_decl=1; recognizer->func_header(".entry"); }
	// [한국어] .entry: 커널 엔트리 포인트 (entry_point=1)
	| VISIBLE_DIRECTIVE ENTRY_DIRECTIVE { $$ = 1; recognizer->g_func_decl=1; recognizer->func_header(".entry"); }
	| WEAK_DIRECTIVE ENTRY_DIRECTIVE { $$ = 1; recognizer->g_func_decl=1; recognizer->func_header(".entry"); }
	| FUNC_DIRECTIVE { $$ = 0; recognizer->g_func_decl=1; recognizer->func_header(".func"); }
	// [한국어] .func: 일반 device 함수 (entry_point=0)
	| VISIBLE_DIRECTIVE FUNC_DIRECTIVE { $$ = 0; recognizer->g_func_decl=1; recognizer->func_header(".func"); }
	| WEAK_DIRECTIVE FUNC_DIRECTIVE { $$ = 0; recognizer->g_func_decl=1; recognizer->func_header(".func"); }
	| EXTERN_DIRECTIVE FUNC_DIRECTIVE { $$ = 2; recognizer->g_func_decl=1; recognizer->func_header(".func"); }
	// [한국어] .extern .func: 외부 함수 선언 (entry_point=2)
	| WEAK_DIRECTIVE FUNC_DIRECTIVE { $$ = 0; recognizer->g_func_decl=1; recognizer->func_header(".func"); }
	;

param_list: /*empty*/
	| param_entry { recognizer->add_directive(); }
	| param_list COMMA {recognizer->func_header_info(",");} param_entry { recognizer->add_directive(); }

param_entry: PARAM_DIRECTIVE { recognizer->add_space_spec(param_space_unclassified,0); } variable_spec ptr_spec identifier_spec { recognizer->add_function_arg(); }
	// [한국어] .param으로 선언된 함수 파라미터
	| REG_DIRECTIVE { recognizer->add_space_spec(reg_space,0); } variable_spec identifier_spec { recognizer->add_function_arg(); }
	// [한국어] .reg으로 선언된 함수 파라미터

ptr_spec: /*empty*/
        | PTR_DIRECTIVE ptr_space_spec ptr_align_spec
        | PTR_DIRECTIVE ptr_align_spec

ptr_space_spec: GLOBAL_DIRECTIVE { recognizer->add_ptr_spec(global_space); }
              | LOCAL_DIRECTIVE  { recognizer->add_ptr_spec(local_space); }
              | SHARED_DIRECTIVE { recognizer->add_ptr_spec(shared_space); }
				  | CONST_DIRECTIVE { recognizer->add_ptr_spec(global_space); }
				  // [한국어] .ptr .const는 global_space로 매핑

ptr_align_spec: ALIGN_DIRECTIVE INT_OPERAND

statement_block: LEFT_BRACE statement_list RIGHT_BRACE 

statement_list: directive_statement { recognizer->add_directive(); }
    | statement_list prototype_block {printf("Prototype statement detected. WARNING: this is not supported yet on GPGPU-SIM\n"); }
	| instruction_statement { recognizer->add_instruction(); }
	| statement_list directive_statement { recognizer->add_directive(); }
	| statement_list instruction_statement { recognizer->add_instruction(); }
	| statement_list {recognizer->start_inst_group();} statement_block {recognizer->end_inst_group();}
	// [한국어] CDP 명령어 그룹 중첩
	| {recognizer->start_inst_group();} statement_block {recognizer->end_inst_group();}
	;

directive_statement: variable_declaration SEMI_COLON
	| VERSION_DIRECTIVE DOUBLE_OPERAND { recognizer->add_version_info($2, 0); }
	// [한국어] .version 번호
	| VERSION_DIRECTIVE DOUBLE_OPERAND PLUS { recognizer->add_version_info($2,1); }
	| ADDRESS_SIZE_DIRECTIVE INT_OPERAND {/*Do nothing*/}
	// [한국어] .address_size는 무시
	| TARGET_DIRECTIVE IDENTIFIER COMMA IDENTIFIER { recognizer->target_header2($2,$4); }
	| TARGET_DIRECTIVE IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER { recognizer->target_header3($2,$4,$6); }
	| TARGET_DIRECTIVE IDENTIFIER { recognizer->target_header($2); }
	| FILE_DIRECTIVE INT_OPERAND STRING { recognizer->add_file($2,$3); }
	| FILE_DIRECTIVE INT_OPERAND STRING COMMA INT_OPERAND COMMA INT_OPERAND { recognizer->add_file($2,$3); }
	| LOC_DIRECTIVE INT_OPERAND INT_OPERAND INT_OPERAND 
	| PRAGMA_DIRECTIVE STRING SEMI_COLON { recognizer->add_pragma($2); }
	| function_decl SEMI_COLON {/*Do nothing*/}
	// [한국어] 함수 선언 프로토타입만 있는 경우
	;

variable_declaration: variable_spec identifier_list { recognizer->add_variables(); }
	| variable_spec identifier_spec EQUALS initializer_list { recognizer->add_variables(); }
	| variable_spec identifier_spec EQUALS literal_operand { recognizer->add_variables(); }
	| CONSTPTR_DIRECTIVE IDENTIFIER COMMA IDENTIFIER COMMA INT_OPERAND { recognizer->add_constptr($2, $4, $6); }
	// [한국어] .constptr 상수 포인터 재배치
	;

variable_spec: var_spec_list { recognizer->set_variable_type(); }
	// [한국어] 공간/타입/정렬 지시어 모음이 모이면 type_info 생성

identifier_list: identifier_spec
	| identifier_list COMMA identifier_spec;

identifier_spec: IDENTIFIER { recognizer->add_identifier($1,0,NON_ARRAY_IDENTIFIER); recognizer->func_header_info($1);}
	// [한국어] 스칼라 식별자
	| IDENTIFIER LEFT_ANGLE_BRACKET INT_OPERAND RIGHT_ANGLE_BRACKET { recognizer->func_header_info($1); recognizer->func_header_info_int("<", $3); recognizer->func_header_info(">");
		int i,lbase,l;
		char *id = NULL;
		lbase = strlen($1);
		for( i=0; i < $3; i++ ) { 
			l = lbase + (int)log10(i+1)+10;
			id = (char*) malloc(l);
			snprintf(id,l,"%s%u",$1,i);
			recognizer->add_identifier(id,0,NON_ARRAY_IDENTIFIER);
		}
		free($1);
	}
	// [한국어] 벡터 형태 식별자: %r<4> → %r0, %r1, %r2, %r3로 확장
	| IDENTIFIER LEFT_SQUARE_BRACKET RIGHT_SQUARE_BRACKET { recognizer->add_identifier($1,0,ARRAY_IDENTIFIER_NO_DIM); recognizer->func_header_info($1); recognizer->func_header_info("["); recognizer->func_header_info("]");}
	// [한국어] 크기 미지정 배열
	| IDENTIFIER LEFT_SQUARE_BRACKET INT_OPERAND RIGHT_SQUARE_BRACKET { recognizer->add_identifier($1,$3,ARRAY_IDENTIFIER); recognizer->func_header_info($1); recognizer->func_header_info_int("[",$3); recognizer->func_header_info("]");}
	// [한국어] 크기 지정 배열
	;

var_spec_list: var_spec 
		 | var_spec_list var_spec;

var_spec: space_spec 
	| type_spec
	| align_spec
	| VISIBLE_DIRECTIVE
	| EXTERN_DIRECTIVE { recognizer->add_extern_spec(); }
    | WEAK_DIRECTIVE
	;

align_spec: ALIGN_DIRECTIVE INT_OPERAND { recognizer->add_alignment_spec($2); }

space_spec: REG_DIRECTIVE {  recognizer->add_space_spec(reg_space,0); }
	| SREG_DIRECTIVE  {  recognizer->add_space_spec(reg_space,0); }
	| addressable_spec
	;

addressable_spec: CONST_DIRECTIVE {  recognizer->add_space_spec(const_space,$1); }
	| GLOBAL_DIRECTIVE 	  {  recognizer->add_space_spec(global_space,0); }
	| LOCAL_DIRECTIVE 	  {  recognizer->add_space_spec(local_space,0); }
	| PARAM_DIRECTIVE 	  {  recognizer->add_space_spec(param_space_unclassified,0); }
	| SHARED_DIRECTIVE 	  {  recognizer->add_space_spec(shared_space,0); }
	| SSTARR_DIRECTIVE    {  recognizer->add_space_spec(sstarr_space,0); }
	| SURF_DIRECTIVE 	  {  recognizer->add_space_spec(surf_space,0); }
	| TEX_DIRECTIVE 	  {  recognizer->add_space_spec(tex_space,0); }
	;

type_spec: scalar_type 
	|  vector_spec scalar_type 
	;

vector_spec:  V2_TYPE {  recognizer->add_option(V2_TYPE); recognizer->func_header_info(".v2");}
	| V3_TYPE     {  recognizer->add_option(V3_TYPE); recognizer->func_header_info(".v3");}
	| V4_TYPE     {  recognizer->add_option(V4_TYPE); recognizer->func_header_info(".v4");}
	;

scalar_type: S8_TYPE { recognizer->add_scalar_type_spec( S8_TYPE ); }
	| S16_TYPE   { recognizer->add_scalar_type_spec( S16_TYPE ); }
	| S32_TYPE   { recognizer->add_scalar_type_spec( S32_TYPE ); }
	| S64_TYPE   { recognizer->add_scalar_type_spec( S64_TYPE ); }
	| U8_TYPE    { recognizer->add_scalar_type_spec( U8_TYPE ); }
	| U16_TYPE   { recognizer->add_scalar_type_spec( U16_TYPE ); }
	| U32_TYPE   { recognizer->add_scalar_type_spec( U32_TYPE ); }
	| U64_TYPE   { recognizer->add_scalar_type_spec( U64_TYPE ); }
	| F16_TYPE   { recognizer->add_scalar_type_spec( F16_TYPE ); }
	| F32_TYPE   { recognizer->add_scalar_type_spec( F32_TYPE ); }
	| F64_TYPE   { recognizer->add_scalar_type_spec( F64_TYPE ); }
	| FF64_TYPE   { recognizer->add_scalar_type_spec( FF64_TYPE ); }
	| B8_TYPE    { recognizer->add_scalar_type_spec( B8_TYPE );  }
	| B16_TYPE   { recognizer->add_scalar_type_spec( B16_TYPE ); }
	| B32_TYPE   { recognizer->add_scalar_type_spec( B32_TYPE ); }
	| B64_TYPE   { recognizer->add_scalar_type_spec( B64_TYPE ); }
	| BB64_TYPE   { recognizer->add_scalar_type_spec( BB64_TYPE ); }
	| BB128_TYPE   { recognizer->add_scalar_type_spec( BB128_TYPE ); }
	| PRED_TYPE  { recognizer->add_scalar_type_spec( PRED_TYPE ); }
	| TEXREF_TYPE  { recognizer->add_scalar_type_spec( TEXREF_TYPE ); }
	| SAMPLERREF_TYPE  { recognizer->add_scalar_type_spec( SAMPLERREF_TYPE ); }
	| SURFREF_TYPE  { recognizer->add_scalar_type_spec( SURFREF_TYPE ); }
	;

initializer_list: LEFT_BRACE literal_list RIGHT_BRACE { recognizer->add_array_initializer(); }
	| LEFT_BRACE initializer_list RIGHT_BRACE { syntax_not_implemented(scanner, recognizer); }

literal_list: literal_operand
	| literal_list COMMA literal_operand;

// TODO: This is currently hardcoded to handle and ignore one specific case
// that all prototype statements follow in the PTX from Pytorch. As a
// workaround, this parses and ignores both the prototype declaration 
// and calling of the prototype (which conveniently comes right after the 
// declaration for all cases.) This should be changed to handle both 
// declaring the prototype, and actually calling it.
prototype_block: prototype_decl prototype_call

prototype_decl: IDENTIFIER COLON CALLPROTOTYPE_DIRECTIVE LEFT_PAREN prototype_param RIGHT_PAREN IDENTIFIER LEFT_PAREN prototype_param RIGHT_PAREN SEMI_COLON 
		      
prototype_call: OPCODE LEFT_PAREN IDENTIFIER RIGHT_PAREN COMMA operand COMMA LEFT_PAREN IDENTIFIER RIGHT_PAREN COMMA IDENTIFIER SEMI_COLON
	      | OPCODE IDENTIFIER COMMA LEFT_PAREN IDENTIFIER RIGHT_PAREN COMMA IDENTIFIER SEMI_COLON

prototype_param: /* empty */
	       | PARAM_DIRECTIVE B64_TYPE IDENTIFIER
	       | PARAM_DIRECTIVE B32_TYPE IDENTIFIER

instruction_statement:  instruction SEMI_COLON
	| IDENTIFIER COLON { recognizer->add_label($1); }
	// [한국어] 레이블 정의: "loop:"
	| pred_spec instruction SEMI_COLON;
	// [한국어] 프레디케이트가 붙은 명령어: "@p bra loop"

instruction: opcode_spec LEFT_PAREN operand RIGHT_PAREN { recognizer->set_return(); } COMMA operand COMMA LEFT_PAREN operand_list RIGHT_PAREN
	// [한국어] call (ret), func, (args) 형태
	| opcode_spec operand COMMA LEFT_PAREN operand_list RIGHT_PAREN
	| opcode_spec operand COMMA LEFT_PAREN RIGHT_PAREN
	| opcode_spec operand_list 
	| opcode_spec
	// [한국어] 오퍼랜드 없는 명령어 (예: exit, ret)
	;

opcode_spec: OPCODE { recognizer->add_opcode($1); } option_list
	| OPCODE { recognizer->add_opcode($1); }

pred_spec: PRED IDENTIFIER  { recognizer->add_pred($2,0, -1); }
	// [한국어] @%p
	| PRED EXCLAMATION IDENTIFIER { recognizer->add_pred($3,1, -1); }
	// [한국어] @!%p
	| PRED IDENTIFIER LT_OPTION  { recognizer->add_pred($2,0,1); }
	| PRED IDENTIFIER EQ_OPTION  { recognizer->add_pred($2,0,2); }
	| PRED IDENTIFIER LE_OPTION  { recognizer->add_pred($2,0,3); }
	| PRED IDENTIFIER NE_OPTION  { recognizer->add_pred($2,0,5); }
	| PRED IDENTIFIER GE_OPTION  { recognizer->add_pred($2,0,6); }
	| PRED IDENTIFIER EQU_OPTION  { recognizer->add_pred($2,0,10); }
	| PRED IDENTIFIER GTU_OPTION  { recognizer->add_pred($2,0,12); }
	| PRED IDENTIFIER NEU_OPTION  { recognizer->add_pred($2,0,13); }
	| PRED IDENTIFIER CF_OPTION  { recognizer->add_pred($2,0,17); }
	| PRED IDENTIFIER SF_OPTION  { recognizer->add_pred($2,0,19); }
	| PRED IDENTIFIER NSF_OPTION  { recognizer->add_pred($2,0,28); }
	;

option_list: option
	| option option_list ;

option: type_spec
	| compare_spec
	| addressable_spec
	| rounding_mode
	| wmma_spec 
	| prmt_spec 
	| SYNC_OPTION { recognizer->add_option(SYNC_OPTION); }
	| ARRIVE_OPTION { recognizer->add_option(ARRIVE_OPTION); }
	| RED_OPTION { recognizer->add_option(RED_OPTION); }
	| UNI_OPTION { recognizer->add_option(UNI_OPTION); }
	| WIDE_OPTION { recognizer->add_option(WIDE_OPTION); }
	| ANY_OPTION { recognizer->add_option(ANY_OPTION); }
	| ALL_OPTION { recognizer->add_option(ALL_OPTION); }
	| BALLOT_OPTION { recognizer->add_option(BALLOT_OPTION); }
	| GLOBAL_OPTION { recognizer->add_option(GLOBAL_OPTION); }
	| CTA_OPTION { recognizer->add_option(CTA_OPTION); }
	| SYS_OPTION { recognizer->add_option(SYS_OPTION); }
	| GEOM_MODIFIER_1D { recognizer->add_option(GEOM_MODIFIER_1D); }
	| GEOM_MODIFIER_2D { recognizer->add_option(GEOM_MODIFIER_2D); }
	| GEOM_MODIFIER_3D { recognizer->add_option(GEOM_MODIFIER_3D); }
	| SAT_OPTION { recognizer->add_option(SAT_OPTION); }
	| FTZ_OPTION { recognizer->add_option(FTZ_OPTION); }
	| NEG_OPTION { recognizer->add_option(NEG_OPTION); }
	| APPROX_OPTION { recognizer->add_option(APPROX_OPTION); }
	| FULL_OPTION { recognizer->add_option(FULL_OPTION); }
	| EXIT_OPTION { recognizer->add_option(EXIT_OPTION); }
	| ABS_OPTION { recognizer->add_option(ABS_OPTION); }
	| atomic_operation_spec ;
	| TO_OPTION { recognizer->add_option(TO_OPTION); }
	| HALF_OPTION { recognizer->add_option(HALF_OPTION); }
	| EXTP_OPTION { recognizer->add_option(EXTP_OPTION); }
	| CA_OPTION { recognizer->add_option(CA_OPTION); }
	| CG_OPTION { recognizer->add_option(CG_OPTION); }
	| CS_OPTION { recognizer->add_option(CS_OPTION); }
	| LU_OPTION { recognizer->add_option(LU_OPTION); }
	| CV_OPTION { recognizer->add_option(CV_OPTION); }
	| WB_OPTION { recognizer->add_option(WB_OPTION); }
	| WT_OPTION { recognizer->add_option(WT_OPTION); }
	| NC_OPTION { recognizer->add_option(NC_OPTION); }
	| UP_OPTION { recognizer->add_option(UP_OPTION); }
	| DOWN_OPTION { recognizer->add_option(DOWN_OPTION); }
	| BFLY_OPTION { recognizer->add_option(BFLY_OPTION); }
	| IDX_OPTION { recognizer->add_option(IDX_OPTION); }
	| WRAP_OPTION { recognizer->add_option(WRAP_OPTION); }
	| CLAMP_OPTION { recognizer->add_option(CLAMP_OPTION); }
	| LEFT_OPTION { recognizer->add_option(LEFT_OPTION); }
	| RIGHT_OPTION { recognizer->add_option(RIGHT_OPTION); }
	;

atomic_operation_spec: ATOMIC_AND { recognizer->add_option(ATOMIC_AND); }
	| ATOMIC_POPC { recognizer->add_option(ATOMIC_POPC); }
	| ATOMIC_OR { recognizer->add_option(ATOMIC_OR); }
	| ATOMIC_XOR { recognizer->add_option(ATOMIC_XOR); }
	| ATOMIC_CAS { recognizer->add_option(ATOMIC_CAS); }
	| ATOMIC_EXCH { recognizer->add_option(ATOMIC_EXCH); }
	| ATOMIC_ADD { recognizer->add_option(ATOMIC_ADD); }
	| ATOMIC_INC { recognizer->add_option(ATOMIC_INC); }
	| ATOMIC_DEC { recognizer->add_option(ATOMIC_DEC); }
	| ATOMIC_MIN { recognizer->add_option(ATOMIC_MIN); }
	| ATOMIC_MAX { recognizer->add_option(ATOMIC_MAX); }
	;

rounding_mode: floating_point_rounding_mode
	| integer_rounding_mode;


floating_point_rounding_mode: RN_OPTION { recognizer->add_option(RN_OPTION); }
	| RZ_OPTION { recognizer->add_option(RZ_OPTION); }
	| RM_OPTION { recognizer->add_option(RM_OPTION); }
	| RP_OPTION { recognizer->add_option(RP_OPTION); }
	;

integer_rounding_mode: RNI_OPTION { recognizer->add_option(RNI_OPTION); }
	| RZI_OPTION { recognizer->add_option(RZI_OPTION); }
	| RMI_OPTION { recognizer->add_option(RMI_OPTION); }
	| RPI_OPTION { recognizer->add_option(RPI_OPTION); }
	;

compare_spec:EQ_OPTION { recognizer->add_option(EQ_OPTION); }
	| NE_OPTION { recognizer->add_option(NE_OPTION); }
	| LT_OPTION { recognizer->add_option(LT_OPTION); }
	| LE_OPTION { recognizer->add_option(LE_OPTION); }
	| GT_OPTION { recognizer->add_option(GT_OPTION); }
	| GE_OPTION { recognizer->add_option(GE_OPTION); }
	| LO_OPTION { recognizer->add_option(LO_OPTION); }
	| LS_OPTION { recognizer->add_option(LS_OPTION); }
	| HI_OPTION { recognizer->add_option(HI_OPTION); }
	| HS_OPTION  { recognizer->add_option(HS_OPTION); }
	| EQU_OPTION { recognizer->add_option(EQU_OPTION); }
	| NEU_OPTION { recognizer->add_option(NEU_OPTION); }
	| LTU_OPTION { recognizer->add_option(LTU_OPTION); }
	| LEU_OPTION { recognizer->add_option(LEU_OPTION); }
	| GTU_OPTION { recognizer->add_option(GTU_OPTION); }
	| GEU_OPTION { recognizer->add_option(GEU_OPTION); }
	| NUM_OPTION { recognizer->add_option(NUM_OPTION); }
	| NAN_OPTION { recognizer->add_option(NAN_OPTION); }
	;

prmt_spec: PRMT_F4E_MODE { recognizer->add_option( PRMT_F4E_MODE); }
	|  PRMT_B4E_MODE { recognizer->add_option( PRMT_B4E_MODE); }
	|  PRMT_RC8_MODE { recognizer->add_option( PRMT_RC8_MODE); }
	|  PRMT_RC16_MODE{ recognizer->add_option( PRMT_RC16_MODE);}
	|  PRMT_ECL_MODE { recognizer->add_option( PRMT_ECL_MODE); }
	|  PRMT_ECR_MODE { recognizer->add_option( PRMT_ECR_MODE); }
	;

wmma_spec: WMMA_DIRECTIVE LAYOUT CONFIGURATION{recognizer->add_space_spec(global_space,0);recognizer->add_ptr_spec(global_space); recognizer->add_wmma_option($1);recognizer->add_wmma_option($2);recognizer->add_wmma_option($3);}
	// [한국어] WMMA load/store: .load/.store .row/.col .m16n16k16
	| WMMA_DIRECTIVE LAYOUT LAYOUT CONFIGURATION{recognizer->add_wmma_option($1);recognizer->add_wmma_option($2);recognizer->add_wmma_option($3);recognizer->add_wmma_option($4);}
	// [한국어] WMMA mma: .mma .row .col .m16n16k16
	;

vp_spec: WMMA_DIRECTIVE LAYOUT CONFIGURATION{recognizer->add_space_spec(global_space,0);recognizer->add_ptr_spec(global_space);recognizer->add_wmma_option($1);recognizer->add_wmma_option($2);recognizer->add_wmma_option($3);}
	| WMMA_DIRECTIVE LAYOUT LAYOUT CONFIGURATION{recognizer->add_wmma_option($1);recognizer->add_wmma_option($2);recognizer->add_wmma_option($3);recognizer->add_wmma_option($4);}
	;



operand_list: operand
	| operand COMMA operand_list;

operand: IDENTIFIER  { recognizer->add_scalar_operand( $1 ); }
	// [한국어] 스칼라 레지스터/변수 피연산자
	| EXCLAMATION IDENTIFIER { recognizer->add_neg_pred_operand( $2 ); }
	// [한국어] 부정 프레디케이트 피연산자
	| MINUS IDENTIFIER  { recognizer->add_scalar_operand( $2 ); recognizer->change_operand_neg(); }
	// [한국어] 음수 레지스터/변수
	| memory_operand
	| literal_operand
	| builtin_operand
	| vector_operand
	| MINUS vector_operand { recognizer->change_operand_neg(); }
	| tex_operand
	| IDENTIFIER PLUS INT_OPERAND { recognizer->add_address_operand($1,$3); }
	// [한국어] symbol + offset 주소
	| IDENTIFIER LO_OPTION { recognizer->add_scalar_operand( $1 ); recognizer->change_operand_lohi(1);}
	// [한국어] 하위 16비트
	| MINUS IDENTIFIER LO_OPTION { recognizer->add_scalar_operand( $2 ); recognizer->change_operand_lohi(1); recognizer->change_operand_neg();}
	| IDENTIFIER HI_OPTION { recognizer->add_scalar_operand( $1 ); recognizer->change_operand_lohi(2);}
	// [한국어] 상위 16비트
	| MINUS IDENTIFIER HI_OPTION { recognizer->add_scalar_operand( $2 ); recognizer->change_operand_lohi(2); recognizer->change_operand_neg();}
	| IDENTIFIER PIPE IDENTIFIER { recognizer->add_2vector_operand($1,$3); recognizer->change_double_operand_type(-1);}
	// [한국어] reg | reg 형태 (set/cvt double destination)
	| IDENTIFIER PIPE IDENTIFIER LO_OPTION { recognizer->add_2vector_operand($1,$3); recognizer->change_double_operand_type(-1); recognizer->change_operand_lohi(1);}
	| IDENTIFIER PIPE IDENTIFIER HI_OPTION { recognizer->add_2vector_operand($1,$3); recognizer->change_double_operand_type(-1); recognizer->change_operand_lohi(2);}
	| IDENTIFIER BACKSLASH IDENTIFIER { recognizer->add_2vector_operand($1,$3); recognizer->change_double_operand_type(-3);}
	| IDENTIFIER BACKSLASH IDENTIFIER LO_OPTION { recognizer->add_2vector_operand($1,$3); recognizer->change_double_operand_type(-3); recognizer->change_operand_lohi(1);}
	| IDENTIFIER BACKSLASH IDENTIFIER HI_OPTION { recognizer->add_2vector_operand($1,$3); recognizer->change_double_operand_type(-3); recognizer->change_operand_lohi(2);}
	;

vector_operand: LEFT_BRACE IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE { recognizer->add_2vector_operand($2,$4); }
			| LEFT_BRACE IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE { recognizer->add_3vector_operand($2,$4,$6); }
			| LEFT_BRACE IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE { recognizer->add_4vector_operand($2,$4,$6,$8); }
			| LEFT_BRACE IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER RIGHT_BRACE { recognizer->add_8vector_operand($2,$4,$6,$8,$10,$12,$14,$16); }
			| LEFT_BRACE IDENTIFIER RIGHT_BRACE { recognizer->add_1vector_operand($2); }
		;

tex_operand: LEFT_SQUARE_BRACKET IDENTIFIER COMMA { recognizer->add_scalar_operand($2); }
			vector_operand 
		     RIGHT_SQUARE_BRACKET
		;

builtin_operand: SPECIAL_REGISTER DIMENSION_MODIFIER { recognizer->add_builtin_operand($1,$2); }
	        | SPECIAL_REGISTER { recognizer->add_builtin_operand($1,-1); }
		;

memory_operand : LEFT_SQUARE_BRACKET address_expression RIGHT_SQUARE_BRACKET { recognizer->add_memory_operand(); }
	// [한국어] [address]
	| IDENTIFIER LEFT_SQUARE_BRACKET address_expression RIGHT_SQUARE_BRACKET { recognizer->add_memory_operand(); recognizer->change_memory_addr_space($1); }
	// [한국어] g[address] 같은 메모리 공간 한정자
	| IDENTIFIER LEFT_SQUARE_BRACKET literal_operand RIGHT_SQUARE_BRACKET { recognizer->change_memory_addr_space($1); }
	| IDENTIFIER LEFT_SQUARE_BRACKET twin_operand RIGHT_SQUARE_BRACKET { recognizer->change_memory_addr_space($1); recognizer->add_memory_operand();}
        | MINUS memory_operand { recognizer->change_operand_neg(); }
		;

twin_operand : IDENTIFIER PLUS IDENTIFIER { recognizer->add_double_operand($1,$3); recognizer->change_double_operand_type(1); }
	| IDENTIFIER PLUS IDENTIFIER LO_OPTION { recognizer->add_double_operand($1,$3); recognizer->change_double_operand_type(1); recognizer->change_operand_lohi(1); }
	| IDENTIFIER PLUS IDENTIFIER HI_OPTION { recognizer->add_double_operand($1,$3); recognizer->change_double_operand_type(1); recognizer->change_operand_lohi(2); }
	| IDENTIFIER PLUS EQUALS IDENTIFIER  { recognizer->add_double_operand($1,$4); recognizer->change_double_operand_type(2); }
	| IDENTIFIER PLUS EQUALS IDENTIFIER LO_OPTION { recognizer->add_double_operand($1,$4); recognizer->change_double_operand_type(2); recognizer->change_operand_lohi(1); }
	| IDENTIFIER PLUS EQUALS IDENTIFIER HI_OPTION { recognizer->add_double_operand($1,$4); recognizer->change_double_operand_type(2); recognizer->change_operand_lohi(2); }
	| IDENTIFIER PLUS EQUALS INT_OPERAND  { recognizer->add_address_operand($1,$4); recognizer->change_double_operand_type(3); }
	;

literal_operand : INT_OPERAND { recognizer->add_literal_int($1); }
	| FLOAT_OPERAND { recognizer->add_literal_float($1); }
	| DOUBLE_OPERAND { recognizer->add_literal_double($1); }
	;

address_expression: IDENTIFIER { recognizer->add_address_operand($1,0); }
	| IDENTIFIER LO_OPTION { recognizer->add_address_operand($1,0); recognizer->change_operand_lohi(1);}
	| IDENTIFIER HI_OPTION { recognizer->add_address_operand($1,0); recognizer->change_operand_lohi(2); }
	| IDENTIFIER PLUS INT_OPERAND { recognizer->add_address_operand($1,$3); }
	| INT_OPERAND { recognizer->add_address_operand2($1); }
	// [한국어] 즉치 주소 (예: [256])
	;

%%

void syntax_not_implemented(yyscan_t yyscanner, ptx_recognizer* recognizer)
{
	printf("Parse error (%s): this syntax is not (yet) implemented:\n", recognizer->gpgpu_ctx->g_filename);
	ptx_error(yyscanner, recognizer, NULL);
	abort();
}
