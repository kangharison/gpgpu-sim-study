/*
 * [한국어 설명] PTX 파서 클래스 선언 (ptx_parser.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 PTX(Parallel Thread eXecution) 어셈블리 파서의 핵심
 * 클래스인 `ptx_recognizer`를 선언한다. Lex/Yacc로 생성된 `ptx.l`/`ptx.y` 파서가
 * PTX 텍스트를 토큰화하면, `ptx_recognizer`는 그 토큰 스트림을 받아 PTX IR
 * (Intermediate Representation) 구조체로 조립하는 "의미 분석(semantic action)"
 * 집합체이다. 즉, Yacc의 각 문법 규칙 액션에서 이 클래스의 메서드를 호출하여
 * 함수 정의, 변수 선언, 명령어, 오퍼랜드 등을 PTX IR로 축적한다.
 * 파싱 완료 후 `g_global_symbol_table`에 모든 PTX 함수/변수가 등록된 상태가 된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CUDA Application → libcuda 인터셉트 → gpgpusim_entrypoint
 *   → [이 파일: PTX 텍스트 파싱] → PTX IR (ptx_ir.h의 function_info 등)
 *   → cuda-sim/ 기능 시뮬레이션 → gpgpu-sim/ 타이밍 시뮬레이션
 * 실행 컨텍스트: 호스트 유저스페이스, 커널 실행 이전 초기화 단계에서 단 1회 호출됨.
 * 사이클-레벨 시뮬레이션과는 무관하게, 커널 바이너리를 IR로 준비하는 전처리 단계다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존하는 모듈: ptx.l, ptx.y (Lex/Yacc 생성 파서), ptx_ir.h (function_info,
 *   symbol_table, ptx_instruction, operand_info 등), abstract_hardware_model.h
 * - 이 파일에 의존하는 모듈: gpgpusim_entrypoint.cc (gpgpu_context::init_parser),
 *   ptx_parser.cc (모든 semantic action 구현)
 * - 핵심 자료구조 흐름: ptx.y의 문법 액션 → ptx_recognizer의 g_instructions 리스트
 *   → function_info::add_inst() → gpgpu_ptx_assemble() → 완성된 PTX IR
 * - gpgpu_context를 통해 func_sim(cuda_sim), g_filename 등 전역 상태와 연결됨
 *
 * === 주요 함수/구조체 요약 ===
 * - ptx_recognizer 클래스: 파서 상태 전체를 보유하는 핵심 클래스
 * - init_directive_state(): 지시어(variable/type 선언) 파싱 상태 초기화
 * - init_instruction_state(): 명령어 파싱 상태 초기화 (매 명령어마다 호출)
 * - add_instruction(): 완성된 명령어를 g_instructions 리스트에 추가
 * - end_function(): 함수 파싱 완료 시 IR 조립 및 어셈블 수행
 * - add_identifier(): 변수 식별자를 심볼 테이블에 등록하고 메모리 공간을 할당
 */

// Copyright (c) 2009-2011, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ptx_parser_INCLUDED  // [한국어] include 가드: 중복 포함 방지
#define ptx_parser_INCLUDED  // [한국어] 헤더 파일이 한 번만 처리되도록 정의

#include "../abstract_hardware_model.h"  // [한국어] warp, core_config 등 하드웨어 추상화 타입 포함
#include "ptx_ir.h"  // [한국어] PTX IR의 핵심 타입들: function_info, symbol_table, ptx_instruction, operand_info 등

class gpgpu_context;           // [한국어] 시뮬레이터 전체 컨텍스트 전방 선언 (순환 의존 방지)
typedef void *yyscan_t;        // [한국어] Flex(Lex) 재진입 가능 스캐너 핸들 타입; void*로 opaque하게 다룸

/*
 * [한국어]
 * ptx_recognizer - PTX 파서의 의미 분석기(semantic recognizer) 클래스
 *
 * @param ctx: 시뮬레이터 전체 컨텍스트 포인터; g_filename, func_sim 등 전역 상태 접근
 *
 * 이 클래스는 Yacc(Bison) 문법 파일 ptx.y에서 각 문법 규칙이 reduce될 때
 * 호출되는 "semantic action"의 집합체다. Flex 토크나이저가 PTX 텍스트를 토큰으로
 * 분해하면, Yacc 파서가 문법에 따라 reduce하며 이 클래스의 메서드를 순서대로 호출한다.
 * 그 결과로 function_info, symbol_table, ptx_instruction 등 PTX IR이 완성된다.
 *
 * 실행 컨텍스트: 호스트 유저스페이스, 커널 실행 전 한 번 수행되는 초기화 단계.
 * 스레드 안전성 없음 — 단일 파서 인스턴스를 단일 스레드에서 사용한다.
 *
 * 호출 체인:
 *   gpgpu_context::init_parser() → ptx_parse() [Yacc 파서 진입]
 *     → ptx.y 문법 규칙 액션들 → [이 클래스의 메서드들] → PTX IR 완성
 */
class ptx_recognizer {
 public:
  /*
   * [한국어]
   * ptx_recognizer 생성자 - 파서 상태를 안전한 초기값으로 설정
   *
   * @param ctx: gpgpu_context 포인터; g_filename, func_sim 연결에 필요
   *
   * g_return_var를 ctx로 초기화하는 이유: operand_info 생성자가 gpgpu_context를 필요로 하므로
   * 초기화 리스트에서 먼저 처리한다. 이후 모든 멤버를 "미설정" 상태를 나타내는
   * 센티넬 값(-1, NULL, undefined_space, false)으로 초기화한다.
   * 이 값들은 파싱 중 각 상태 초기화 함수(init_directive_state 등)에서도 재사용된다.
   *
   * 호출 체인:
   *   gpgpu_context 생성자 → [이 생성자] (gpgpu_context가 ptx_recognizer를 멤버로 보유)
   */
  ptx_recognizer(gpgpu_context *ctx) : g_return_var(ctx) {
    scanner = NULL;                             // [한국어] Flex 스캐너 핸들; init_parser에서 ptx_lex_init으로 초기화됨
    g_size = -1;                                // [한국어] 현재 파싱 중인 타입의 크기(바이트); -1은 미설정
    g_add_identifier_cached__identifier = NULL; // [한국어] 반환값 변수 이름 캐시; 함수 이름 파싱 전에 먼저 등장하는 경우 임시 저장
    g_alignment_spec = -1;                      // [한국어] .align 지시어 값; -1은 정렬 미지정
    g_var_type = NULL;                          // [한국어] 현재 파싱 중인 변수의 type_info; set_variable_type에서 설정
    g_opcode = -1;                              // [한국어] 현재 명령어 오피코드; -1은 레이블(label) 또는 미설정
    g_space_spec = undefined_space;             // [한국어] 메모리 공간 지시어(.global, .shared 등); undefined=미설정
    g_ptr_spec = undefined_space;               // [한국어] .ptr 지시어의 대상 공간; undefined=포인터 없음
    g_scalar_type_spec = -1;                    // [한국어] 스칼라 타입(.s32, .f32 등)의 enum 값; -1=미설정
    g_vector_spec = -1;                         // [한국어] .v2/.v4 벡터 지시어; -1=벡터 없음
    g_extern_spec = 0;                          // [한국어] .extern 지시어 여부; 0=일반 선언
    g_func_decl = 0;                            // [한국어] 현재 함수 선언 파싱 중 여부; 0=비활성
    g_ident_add_uid = 0;                        // [한국어] 식별자 추가 순서 카운터 (디버그/추적용)
    g_const_alloc = 1;                          // [한국어] 상수 메모리 할당 인덱스; 1부터 시작 (0=예약)
    g_max_regs_per_thread = 0;                  // [한국어] 현재 함수에서 사용된 최대 레지스터 번호 추적
    g_global_symbol_table = NULL;               // [한국어] 파일 전체 전역 심볼 테이블; init_parser에서 생성
    g_current_symbol_table = NULL;              // [한국어] 현재 파싱 컨텍스트의 심볼 테이블 (함수 진입 시 변경)
    g_last_symbol = NULL;                       // [한국어] 마지막으로 추가된 심볼; add_identifier가 설정
    g_error_detected = 0;                       // [한국어] 파싱 오류 발생 여부; parse_error_impl에서 1로 설정
    g_entry_func_param_index = 0;               // [한국어] 커널 파라미터 인덱스; add_identifier에서 증가
    g_func_info = NULL;                         // [한국어] 현재 파싱 중인 함수의 function_info; add_function_name에서 설정
    g_debug_ir_generation = false;              // [한국어] PTX_SIM_DEBUG 환경변수로 활성화; 파싱 과정 printf 출력
    gpgpu_ctx = ctx;                            // [한국어] 역방향 포인터 저장; g_filename 접근에 필수
  }
  // global list
  yyscan_t scanner;
  /* [한국어] Flex 재진입 가능(reentrant) 스캐너의 불투명 핸들.
   * 설정자: gpgpu_context::init_parser에서 ptx_lex_init(&scanner)으로 초기화.
   * 읽는 자: ptx_get_lineno(scanner)로 현재 소스 라인 번호 조회, PTX_PARSE_DPRINTF 매크로.
   * 값 범위: 유효한 Flex 스캐너 포인터 (초기화 후 NULL 불가).
   * 동기화: 단일 스레드 사용; 여러 파일을 순차 파싱한다. */
#define PTX_LINEBUF_SIZE (4 * 1024)  // [한국어] 파서가 한 줄로 읽을 수 있는 최대 버퍼 크기: 4KB
  char linebuf[PTX_LINEBUF_SIZE];
  /* [한국어] 현재 파서가 처리 중인 소스 라인의 텍스트 버퍼.
   * 설정자: ptx.l의 Flex 규칙에서 각 라인을 읽을 때 갱신.
   * 읽는 자: add_instruction에서 ptx_instruction 생성 시 소스 위치 정보로 전달.
   * 값 범위: PTX_LINEBUF_SIZE(4KB) 이내의 null-종료 문자열.
   * 동기화: 단일 스레드 파싱 중에만 사용; 완료 후 해제됨. */
  unsigned col;
  /* [한국어] 현재 파서 위치의 컬럼(열) 번호.
   * 설정자: Flex 스캐너 규칙에서 문자 진행에 따라 증가.
   * 읽는 자: 오류 메시지 출력 시 위치 정보로 사용.
   * 값 범위: 0 이상 (줄 시작=0, 오른쪽으로 증가).
   * 동기화: 단일 스레드 파싱 중에만 유효. */
  int g_size;
  /* [한국어] 현재 파싱 중인 타입(type specifier)의 크기 (바이트 단위).
   * 설정자: add_scalar_type_spec에서 타입에 따라 1/2/4/8/16 중 하나로 설정.
   *         add_identifier에서 배열이면 배열 원소 수를 곱해 전체 크기로 갱신.
   * 읽는 자: add_identifier (공간 할당량 계산), add_function_arg (정렬 계산).
   * 값 범위: 1/2/4/8/16 또는 배열이면 그 배수; -1은 미설정.
   * 동기화: 지시어 파싱 시작(init_directive_state)마다 -1로 리셋. */
  char *g_add_identifier_cached__identifier;
  /* [한국어] 반환값 변수 이름의 임시 캐시 포인터.
   * 배경: PTX .func 선언에서 반환값 변수는 함수 이름보다 먼저 등장할 수 있다.
   *       그때 g_func_info가 아직 NULL이므로 나중에 처리하기 위해 여기에 임시 저장.
   * 설정자: add_identifier에서 g_func_decl==1이고 g_func_info==NULL일 때 strdup.
   * 읽는 자: add_function_name에서 NULL이 아니면 다시 add_identifier를 재호출.
   * 값 범위: 유효한 strdup 문자열 또는 NULL (캐시 없음).
   * 동기화: 반드시 add_function_name에서 free 후 NULL로 클리어해야 함. */
  int g_add_identifier_cached__array_dim;
  /* [한국어] 임시 캐시된 반환값 변수의 배열 차원 수.
   * 설정자: add_identifier에서 g_add_identifier_cached__identifier와 함께 저장.
   * 읽는 자: add_function_name에서 캐시 재처리 시 add_identifier에 전달.
   * 값 범위: 배열 선언이면 원소 수, 아니면 0.
   * 동기화: g_add_identifier_cached__identifier와 함께 사용됨. */
  int g_add_identifier_cached__array_ident;
  /* [한국어] 임시 캐시된 반환값 변수의 배열 식별자 종류 (NON_ARRAY_IDENTIFIER 등).
   * 설정자: add_identifier에서 g_add_identifier_cached__identifier와 함께 저장.
   * 읽는 자: add_function_name에서 캐시 재처리 시 add_identifier에 전달.
   * 값 범위: NON_ARRAY_IDENTIFIER(1), ARRAY_IDENTIFIER_NO_DIM(2), ARRAY_IDENTIFIER(3).
   * 동기화: g_add_identifier_cached__identifier와 함께 사용됨. */
  int g_alignment_spec;
  /* [한국어] .align 지시어의 정렬 값(바이트 단위).
   * 배경: PTX .align 8은 8바이트 경계로 정렬하라는 의미.
   * 설정자: add_alignment_spec에서 Yacc 파서가 .align 토큰을 읽을 때 설정.
   * 읽는 자: add_function_arg에서 함수 파라미터 정렬 값 결정에 사용.
   * 값 범위: 1/2/4/8/16 등 유효한 정렬값, 또는 -1(미설정).
   * 동기화: init_directive_state에서 -1로 리셋. */
  // variable declaration stuff:
  type_info *g_var_type;
  /* [한국어] 현재 파싱 중인 변수 선언의 type_info 포인터.
   * 배경: PTX 변수 선언 ".reg .s32 %r0"에서 ".s32"에 해당하는 타입 객체.
   * 설정자: set_variable_type에서 g_space_spec + g_scalar_type_spec로 symbol_table::add_type 호출.
   * 읽는 자: add_identifier에서 symbol_table::add_variable에 타입 정보 전달.
   * 값 범위: 유효한 type_info* 또는 NULL(미설정 — set_variable_type 호출 전).
   * 동기화: init_directive_state에서 NULL 리셋; 각 변수 선언마다 한 번씩 설정됨. */
  // instruction definition stuff:
  const symbol *g_pred;
  /* [한국어] 현재 명령어의 프레디케이트 레지스터 심볼.
   * 배경: PTX "@%p bra label"에서 %p가 프레디케이트. 조건부 실행 제어에 사용.
   * 설정자: add_pred에서 심볼 테이블에서 식별자를 조회하여 설정.
   * 읽는 자: add_instruction에서 ptx_instruction 생성자에 전달.
   * 값 범위: 유효한 symbol* 또는 NULL(프레디케이트 없는 명령어).
   * 동기화: init_instruction_state에서 NULL 리셋. */
  int g_neg_pred;
  /* [한국어] 프레디케이트 부정 여부 (0=정상, 1=부정(@!%p)).
   * 배경: "@!%p"는 %p가 false일 때 실행하라는 의미.
   * 설정자: add_pred의 neg 파라미터로 설정.
   * 읽는 자: add_instruction에서 ptx_instruction 생성자에 전달.
   * 값 범위: 0(부정 없음) 또는 1(부정 있음).
   * 동기화: init_instruction_state에서 0 리셋. */
  int g_pred_mod;
  /* [한국어] 프레디케이트 수식어(predicate modifier) 값.
   * 배경: 일부 명령어에서 프레디케이트에 추가 수식어가 붙을 수 있음.
   * 설정자: add_pred의 predModifier 파라미터로 설정.
   * 읽는 자: add_instruction에서 ptx_instruction 생성자에 전달.
   * 값 범위: -1(없음) 또는 특정 수식어 enum 값.
   * 동기화: init_instruction_state에서 -1 리셋. */
  symbol *g_label;
  /* [한국어] 현재 명령어에 붙은 레이블(분기 대상 주소 마커) 심볼.
   * 배경: PTX에서 "loop:" 같은 레이블은 분기 명령어의 대상이 됨.
   * 설정자: add_label에서 심볼 테이블에 레이블을 등록하여 설정.
   * 읽는 자: add_instruction에서 ptx_instruction 생성자에 전달.
   * 값 범위: 유효한 symbol* 또는 NULL(레이블 없는 명령어).
   * 동기화: init_instruction_state에서 NULL 리셋. */
  int g_opcode;
  /* [한국어] 현재 파싱 중인 PTX 명령어의 오피코드 정수 값.
   * 배경: ptx_ir.h의 enum에 정의된 ADD_OP, LD_OP, BRA_OP 등이 사용됨.
   * 설정자: add_opcode에서 Yacc 파서가 명령어 토큰을 읽을 때 설정.
   * 읽는 자: add_instruction, set_return, add_scalar_type_spec (다중 타입 허용 여부 확인).
   * 값 범위: ptx_ir.h의 opcode enum 값, 또는 -1(레이블 전용, 명령어 없음).
   * 동기화: init_instruction_state에서 -1 리셋. */
  std::list<operand_info> g_operands;
  /* [한국어] 현재 명령어의 오퍼랜드(피연산자) 목록.
   * 배경: PTX "add.s32 %r0, %r1, %r2"에서 %r0, %r1, %r2가 오퍼랜드.
   * 설정자: add_scalar_operand, add_address_operand, add_literal_int 등 각종 add_*_operand에서 push_back.
   * 읽는 자: add_instruction에서 ptx_instruction 생성자에 전달, set_return에서 front().
   * 값 범위: 0~다수의 operand_info 엔트리.
   * 동기화: init_directive_state(인라인)와 init_instruction_state에서 clear(). */
  std::list<int> g_options;
  /* [한국어] 현재 명령어의 옵션(수식어) 목록 (예: .lo, .hi, .ca 등).
   * 배경: 많은 PTX 명령어는 .ftz(flush-to-zero), .rn(round-nearest) 같은 수식어를 가짐.
   * 설정자: add_option에서 push_back.
   * 읽는 자: add_instruction에서 ptx_instruction 생성자에 전달.
   * 값 범위: 각 옵션의 정수 enum 값 목록.
   * 동기화: init_instruction_state에서 clear(). */
  std::list<int> g_wmma_options;
  /* [한국어] WMMA(Warp Matrix Multiply-Accumulate) 명령어 전용 옵션 목록.
   * 배경: Tensor Core 연산 명령어(wmma.load, wmma.mma 등)는 일반 옵션과 분리된 추가 옵션이 있음.
   * 설정자: add_wmma_option에서 push_back.
   * 읽는 자: add_instruction에서 ptx_instruction 생성자에 전달.
   * 값 범위: WMMA 관련 enum 값 목록.
   * 동기화: init_instruction_state에서 clear(). */
  std::list<int> g_scalar_type;
  /* [한국어] 현재 명령어/선언의 스칼라 타입 목록 (.s32, .f32 등).
   * 배경: cvt 명령어처럼 소스/목적지 타입이 다른 경우 두 타입이 모두 저장됨.
   * 설정자: add_scalar_type_spec에서 push_back.
   * 읽는 자: add_instruction에서 ptx_instruction 생성자에 전달.
   * 값 범위: 타입 enum 값 목록(B8~BB128 등); 보통 1개, cvt 등은 2개.
   * 동기화: init_directive_state에서 clear(). */
  // type specifier stuff:
  memory_space_t g_space_spec;
  /* [한국어] 현재 선언/명령어의 메모리 공간 지시어 (.reg, .global, .shared 등).
   * 배경: PTX에서 모든 변수는 반드시 메모리 공간을 명시해야 함.
   * 설정자: add_space_spec에서 파싱된 공간 enum으로 설정. param은 맥락에 따라 분류.
   * 읽는 자: set_variable_type, add_identifier의 switch-case에서 공간별 메모리 할당.
   * 값 범위: memory_space_t enum (undefined_space, reg_space, global_space, shared_space 등).
   * 동기화: init_directive_state에서 undefined_space로 리셋. */
  memory_space_t g_ptr_spec;
  /* [한국어] .ptr 지시어의 대상 메모리 공간 (포인터가 가리키는 공간).
   * 배경: PTX의 ".param .u64 .ptr .global ptr"에서 .global이 ptr_spec.
   * 설정자: add_ptr_spec에서 설정; global/local/shared만 유효.
   * 읽는 자: add_identifier에서 파라미터 포인터 여부 판단에 사용.
   * 값 범위: undefined_space(포인터 없음), global_space, local_space, shared_space.
   * 동기화: init_directive_state에서 undefined_space로 리셋. */
  int g_scalar_type_spec;
  /* [한국어] 마지막으로 파싱된 스칼라 타입 지시어의 enum 값.
   * 배경: g_scalar_type 리스트의 마지막 원소와 같으나 빠른 접근을 위해 별도 유지.
   * 설정자: add_scalar_type_spec에서 매번 갱신.
   * 읽는 자: set_variable_type에서 type_info 생성에 사용.
   * 값 범위: ptx_ir.h의 type enum 값 (B8~BB128 등), -1은 미설정.
   * 동기화: init_directive_state에서 -1 리셋. */
  int g_vector_spec;
  /* [한국어] 벡터 지시어 (.v2, .v4) 값.
   * 배경: PTX ".reg .v4 .f32 %f"는 4개짜리 float 벡터 레지스터 선언.
   * 설정자: add_vector_spec에서 설정; 중복 지정 시 parse_assert 실패.
   * 읽는 자: set_variable_type에서 type_info 생성에 사용.
   * 값 범위: 2/4 등 벡터 원소 수, 또는 -1(벡터 없음).
   * 동기화: init_directive_state에서 -1 리셋. */
  int g_extern_spec;
  /* [한국어] .extern 지시어 여부 플래그.
   * 배경: PTX .extern은 다른 모듈에 정의된 외부 변수/함수 선언을 나타냄.
   * 설정자: add_extern_spec에서 1로 설정.
   * 읽는 자: set_variable_type에서 type_info 생성에 extern 플래그 전달.
   * 값 범위: 0(일반 선언) 또는 1(외부 선언).
   * 동기화: init_directive_state에서 0 리셋. */
  int g_func_decl;
  /* [한국어] 현재 함수 선언 파싱 중임을 나타내는 플래그.
   * 배경: 함수 선언 컨텍스트에서는 param 공간을 kernel_param으로 분류해야 함.
   * 설정자: start_function에서 entry_point 값에 따라 설정.
   * 읽는 자: add_space_spec(param 분류), add_identifier(반환값 캐싱), add_function_arg.
   * 값 범위: 0(일반 코드) 또는 1(함수 선언 중).
   * 동기화: start_function 호출 시 변경, end_function에서 자동 리셋. */
  int g_ident_add_uid;
  /* [한국어] 식별자 추가 순서 카운터 (디버그/검사 목적).
   * 배경: 각 add_identifier 호출마다 1씩 증가하여 처리 순서를 추적.
   * 설정자: add_identifier에서 ++로 증가.
   * 읽는 자: 현재는 디버그 출력(PTX_PARSE_DPRINTF)에서만 사용.
   * 값 범위: 0 이상 (파서 생성 시 0, 파싱 진행에 따라 증가).
   * 동기화: 생성자에서 0으로 초기화; 파일 전체 파싱 동안 누적. */
  unsigned g_const_alloc;
  /* [한국어] 상수 메모리 할당 인스턴스 카운터 (디버그 출력용).
   * 배경: add_identifier에서 const_space 변수마다 할당 로그를 출력할 때 일련 번호로 사용.
   * 설정자: add_identifier의 const_space case에서 g_const_alloc++ 후 출력.
   * 읽는 자: printf 출력에서 할당 번호 표시.
   * 값 범위: 1 이상 (1부터 시작, 각 상수 할당마다 증가).
   * 동기화: 생성자에서 1로 초기화; 파일 파싱 전체에서 누적됨. */
  unsigned g_max_regs_per_thread;
  /* [한국어] 현재 함수에서 사용된 최대 레지스터 번호 (레지스터 수 추적).
   * 배경: PTX .maxnreg 지시어 또는 실제 사용량에서 결정; 스케줄러가 이 값을 활용.
   * 설정자: end_function에서 mymax(g_max_regs_per_thread, next_reg_num()-1)로 갱신.
   * 읽는 자: end_function 이후 외부에서 쿼리되어 function_info에 저장.
   * 값 범위: 0 이상; 레지스터를 쓰지 않으면 0.
   * 동기화: 생성자에서 0 초기화; 파일 전체에서 누적 최대값 유지. */
  symbol_table *g_global_symbol_table;
  /* [한국어] 파일 전체를 아우르는 전역 심볼 테이블 포인터.
   * 배경: 모든 .global, .const 변수와 함수 이름이 이 테이블에 등록됨.
   * 설정자: gpgpu_context::init_parser에서 "global_allfiles" 테이블을 생성하여 설정.
   * 읽는 자: add_function_name (함수 등록), end_function (현재 테이블 복구), 기타 다수.
   * 값 범위: 유효한 symbol_table* (init_parser 이후 NULL 불가).
   * 동기화: 단일 스레드 파싱; 파싱 완료 후 호출자가 반환값으로 수신. */
  symbol_table *g_current_symbol_table;
  /* [한국어] 현재 파싱 스코프의 심볼 테이블 포인터.
   * 배경: 함수 진입 시 함수 전용 테이블로 교체, 함수 종료 시 전역으로 복귀.
   * 설정자: add_function_name에서 함수별 테이블로 교체; end_function에서 전역으로 복구.
   * 읽는 자: add_identifier (변수 추가), add_pred/add_label (심볼 조회) 등.
   * 값 범위: 전역 또는 현재 함수의 유효한 symbol_table* (init_parser 이후 NULL 불가).
   * 동기화: 단일 스레드; 함수 진입/종료 시 명시적으로 교체됨. */
  symbol *g_last_symbol;
  /* [한국어] 가장 최근에 추가된 심볼 포인터.
   * 배경: add_identifier 후 add_variables나 add_function_arg에서 이 심볼에 추가 정보를 설정.
   * 설정자: add_identifier에서 symbol_table::add_variable 반환값으로 설정.
   * 읽는 자: add_variables (초기값 설정), add_function_arg (함수 인자 등록),
   *          add_function_name (반환값 설정).
   * 값 범위: 유효한 symbol* 또는 NULL (초기화 전 또는 init_directive_state 후).
   * 동기화: init_directive_state에서 NULL 리셋. */
  std::list<ptx_instruction *> g_instructions;
  /* [한국어] 현재 함수의 파싱 중인 PTX 명령어 목록 (동적 축적).
   * 배경: 함수 파싱 중 add_instruction에서 하나씩 추가되다가, end_function에서
   *       function_info::add_inst에 전달된 후 clear.
   * 설정자: add_instruction에서 push_back.
   * 읽는 자: end_function에서 g_func_info->add_inst(g_instructions).
   * 값 범위: 0~N개의 ptx_instruction* (N = 함수 내 명령어 수).
   * 동기화: end_function에서 clear; 단일 스레드 파싱. */
  int g_error_detected;
  /* [한국어] 파싱 오류 발생 여부 플래그.
   * 설정자: parse_error_impl에서 1로 설정.
   * 읽는 자: 현재는 내부 상태 추적용; 오류 발생 시 abort() 호출로 즉시 종료됨.
   * 값 범위: 0(정상) 또는 1(오류 발생).
   * 동기화: 생성자에서 0 초기화; 파싱 중 오류 시 abort로 인해 이후 상태 무의미. */
  unsigned g_entry_func_param_index;
  /* [한국어] 커널 엔트리 함수의 파라미터 인덱스 카운터.
   * 배경: add_identifier에서 param_space_kernel 파라미터마다 증가하여 위치 정보 기록.
   * 설정자: start_function에서 0으로 리셋; add_identifier의 param_space_kernel 분기에서 ++.
   * 읽는 자: function_info::add_param_name_type_size에 index 전달.
   * 값 범위: 0 이상 (파라미터 수에 따라 증가).
   * 동기화: 함수마다 start_function에서 리셋. */
  function_info *g_func_info;
  /* [한국어] 현재 파싱 중인 함수의 function_info 포인터.
   * 배경: 함수 파싱 시작 시 생성되어 명령어/파라미터/프레임 크기 등이 이 객체에 쌓임.
   * 설정자: add_function_name에서 add_function_decl 반환값으로 설정.
   * 읽는 자: add_identifier(local 프레임 크기), add_function_arg, end_function 등.
   * 값 범위: 유효한 function_info* (함수 파싱 중), NULL (함수 밖 컨텍스트).
   * 동기화: start_function에서 NULL 리셋, add_function_name에서 설정. */
  operand_info g_return_var;
  /* [한국어] 현재 명령어(주로 call)의 반환값 오퍼랜드 정보.
   * 배경: "call (%rd0), func, (args);"에서 (%rd0)에 해당.
   * 설정자: set_return에서 g_operands.front()를 복사 후 set_return 표시.
   * 읽는 자: add_instruction에서 ptx_instruction 생성자에 전달.
   * 값 범위: 유효한 operand_info (call 명령어 시), 기본 생성자 상태(그 외).
   * 동기화: init_instruction_state에서 operand_info(gpgpu_ctx)로 리셋. */
  bool g_debug_ir_generation;
  /* [한국어] PTX IR 생성 과정 디버그 출력 활성화 여부.
   * 배경: 환경변수 PTX_SIM_DEBUG >= 30이면 활성화; 매 파싱 단계를 printf로 출력.
   * 설정자: read_parser_environment_variables에서 환경변수 확인 후 설정.
   * 읽는 자: PTX_PARSE_DPRINTF 매크로에서 이 플래그가 true일 때만 출력.
   * 값 범위: false(기본, 출력 없음) 또는 true(상세 파싱 로그 출력).
   * 동기화: 생성 시 false, 이후 불변. */
  int g_entry_point;
  /* [한국어] 현재 파싱 중인 함수의 진입점 종류.
   * 값 범위: 0=일반 device 함수, 1=.entry(커널 엔트리), 2=.extern 선언.
   * 설정자: start_function에서 Yacc 파서로부터 전달받은 entry_point로 설정.
   * 읽는 자: add_function_name에서 디버그 출력, add_space_spec에서 param 분류.
   * 동기화: start_function 호출마다 갱신. */
  const struct core_config *g_shader_core_config;
  /* [한국어] 셰이더 코어(SM) 설정 포인터; 워프 크기 등 하드웨어 파라미터 포함.
   * 배경: ptx_instruction 생성 시 워프 크기(warp_size)를 알아야 하는 경우가 있음.
   * 설정자: set_ptx_warp_size에서 외부 코어 설정으로 설정.
   * 읽는 자: add_instruction에서 ptx_instruction 생성자에 전달.
   * 값 범위: 유효한 core_config* (set_ptx_warp_size 호출 후); assert(!=0) 보호됨.
   * 동기화: 파싱 전에 한 번 설정 후 불변. */
  std::map<std::string, std::map<unsigned, const ptx_instruction *> >
      g_inst_lookup;
  /* [한국어] 소스 파일명 + 라인 번호 → ptx_instruction* 매핑 테이블.
   * 배경: 디버그 시 소스 위치로 PTX 명령어를 역조회하는 데 사용.
   * 설정자: add_instruction에서 파일명/라인번호 키로 삽입.
   * 읽는 자: ptx_instruction_lookup 함수에서 외부 조회용.
   * 값 범위: 파일명별로 라인번호→명령어 중첩 맵.
   * 동기화: 단일 스레드 파싱 중 축적; 완료 후 read-only 조회. */
  // the program intermediate representation...
  std::map<std::string, symbol_table *> g_sym_name_to_symbol_table;
  /* [한국어] 변수 이름 → 심볼 테이블 매핑 (전역/상수 변수용).
   * 배경: add_identifier에서 global/const 공간 변수를 등록 시 어떤 심볼 테이블에
   *       속하는지 기록; memcpy_symbol에서 이름으로 심볼 테이블을 찾을 때 사용.
   * 설정자: add_identifier의 global_space/const_space 분기에서 삽입.
   * 읽는 자: cuda_sim::gpgpu_ptx_sim_memcpy_symbol 등에서 이름으로 테이블 조회.
   * 값 범위: 변수 이름(string) → symbol_table* 매핑.
   * 동기화: 파싱 완료 후 read-only 사용. */
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  /* [한국어] 시뮬레이터 전체 컨텍스트로의 역방향 포인터.
   * 배경: g_filename 접근, func_sim->g_constants 등록 등에 필요.
   * 설정자: 생성자에서 ctx 인자로 초기화.
   * 읽는 자: add_identifier (g_globals/g_constants 등록), add_function_name 등 다수.
   * 값 범위: 유효한 gpgpu_context* (생성자 이후 NULL 불가).
   * 동기화: 생성 시 설정, 이후 불변. */

  // member function list
  /*
   * [한국어]
   * init_directive_state - 지시어(변수/타입 선언) 파싱 상태를 초기값으로 리셋
   *
   * PTX에서 지시어란 ".global .s32 %r0" 같은 변수 선언 구문이다.
   * 새 지시어 파싱을 시작할 때마다 이전 파싱의 잔여 상태를 지워
   * g_space_spec, g_ptr_spec, g_scalar_type_spec, g_vector_spec, g_opcode,
   * g_alignment_spec, g_size, g_extern_spec, g_scalar_type, g_operands,
   * g_last_symbol을 모두 초기 센티넬 값으로 되돌린다.
   *
   * 호출 체인:
   *   init_instruction_state → [이 함수]
   *   start_function → init_directive_state
   *   add_directive → init_directive_state
   *   end_function → init_directive_state
   */
  void init_directive_state();

  /*
   * [한국어]
   * init_instruction_state - 명령어 파싱 상태를 초기값으로 리셋
   *
   * PTX 명령어 파싱을 새로 시작할 때마다 이전 명령어의 잔여 상태를 제거한다.
   * g_pred, g_neg_pred, g_pred_mod, g_label, g_opcode, g_options,
   * g_wmma_options, g_return_var을 초기화하고, init_directive_state도 함께 호출한다.
   * Yacc의 각 명령어 규칙 시작 부분에서 호출된다.
   *
   * 호출 체인:
   *   ptx.y의 instruction 규칙 시작 → [이 함수] → init_directive_state
   */
  void init_instruction_state();

  /*
   * [한국어]
   * start_function - 새 함수 파싱 시작을 알리는 함수
   *
   * @param entry_point: 함수 종류 (0=device, 1=entry/커널, 2=extern)
   *
   * 지시어/명령어 상태를 초기화하고, g_entry_point와 g_entry_func_param_index를 설정한다.
   * g_func_info를 NULL로 리셋하여 add_function_name 이전 상태를 보장한다.
   *
   * 호출 체인:
   *   ptx.y의 func_decl 규칙 → [이 함수]
   */
  void start_function(int entry_point);

  /*
   * [한국어]
   * add_function_name - 파싱된 함수 이름을 전역 심볼 테이블에 등록
   *
   * @param fname: PTX 소스에서 읽은 함수 이름 문자열
   *
   * g_global_symbol_table에 함수를 선언하고, g_func_info와 g_current_symbol_table을
   * 함수 전용 값으로 갱신한다. 반환값 변수가 캐시되어 있으면(g_add_identifier_cached)
   * 재처리하여 함수의 반환값으로 등록한다. 이미 선언된 함수(prior_decl)이면 인자를 제거하고 재정의한다.
   *
   * 호출 체인:
   *   ptx.y의 func_name 규칙 → [이 함수] → symbol_table::add_function_decl
   */
  void add_function_name(const char *fname);

  /*
   * [한국어]
   * add_directive - 현재 지시어 파싱 완료를 알리는 더미 리셋 함수
   *
   * init_directive_state를 호출하여 지시어 상태를 초기화하는 단순 래퍼.
   * Yacc에서 ';'로 끝나는 지시어 규칙이 reduce될 때 호출된다.
   *
   * 호출 체인:
   *   ptx.y의 directive 규칙 → [이 함수] → init_directive_state
   */
  void add_directive();

  /*
   * [한국어]
   * end_function - 함수 파싱 완료 처리; IR 조립 및 어셈블 수행
   *
   * 1) 지시어/명령어 상태 초기화
   * 2) g_max_regs_per_thread 갱신 (레지스터 사용량 반영)
   * 3) g_func_info에 g_instructions 목록 전달 (add_inst)
   * 4) gpgpu_ptx_assemble로 PC 주소 할당 (바이너리화)
   * 5) g_current_symbol_table을 전역으로 복귀
   *
   * 호출 체인:
   *   ptx.y의 func_body 규칙 → [이 함수] → gpgpu_ptx_assemble
   */
  void end_function();

  /*
   * [한국어]
   * add_identifier - 파싱된 식별자를 심볼 테이블에 등록하고 메모리 공간 할당
   *
   * @param s: 식별자 이름 문자열
   * @param array_dim: 배열 원소 수 (배열이 아니면 0)
   * @param array_ident: ARRAY_IDENTIFIER/ARRAY_IDENTIFIER_NO_DIM/NON_ARRAY_IDENTIFIER
   *
   * 타입 정보(g_var_type)와 공간(g_space_spec)을 바탕으로 심볼을 생성하고,
   * 메모리 공간에 따라 레지스터 번호 할당, shared/local/global/const 메모리 공간 할당 등을 수행한다.
   * 중복 선언은 경고 후 무시한다.
   *
   * 호출 체인:
   *   ptx.y의 identifier_list 규칙 → [이 함수] → symbol_table::add_variable
   */
  void add_identifier(const char *s, int array_dim, unsigned array_ident);

  /*
   * [한국어]
   * add_function_arg - 마지막 심볼을 함수 인자로 등록
   *
   * g_last_symbol을 g_func_info의 인자 목록에 추가하고,
   * 크기/정렬 정보를 add_config_param으로 등록한다.
   * g_func_info가 NULL이면 (함수 외부에서 호출 시) 아무것도 하지 않는다.
   *
   * 호출 체인:
   *   ptx.y의 func_param 규칙 → [이 함수] → function_info::add_arg
   */
  void add_function_arg();

  /*
   * [한국어]
   * add_scalar_type_spec - 스칼라 타입 지시어(.s32, .f32 등)를 파싱 상태에 추가
   *
   * @param type_spec: 타입 enum 값 (B8_TYPE, S32_TYPE, F32_TYPE 등)
   *
   * 타입에 따라 g_size를 바이트 단위로 설정하고 g_scalar_type 리스트에 추가한다.
   * g_scalar_type_spec에 마지막 타입을 저장한다.
   * cvt/set/slct/tex/mma/dp4a 외 명령어에서 두 번 이상 타입이 오면 parse_assert 실패.
   *
   * 호출 체인:
   *   ptx.y의 type_spec 규칙 → [이 함수]
   */
  void add_scalar_type_spec(int type_spec);

  /*
   * [한국어]
   * add_scalar_operand - 스칼라 레지스터/변수 오퍼랜드를 파싱 상태에 추가
   *
   * @param identifier: 오퍼랜드 식별자 이름 (%r0, %f1 등)
   *
   * 심볼 테이블에서 identifier를 조회하고, 없으면 분기/callp 대상(forward decl)으로 처리한다.
   * 조회된 심볼로 operand_info를 생성하여 g_operands에 push_back한다.
   *
   * 호출 체인:
   *   ptx.y의 operand 규칙 → [이 함수]
   */
  void add_scalar_operand(const char *identifier);

  /*
   * [한국어]
   * add_neg_pred_operand - 부정(negated) 프레디케이트 오퍼랜드 추가
   *
   * @param identifier: 프레디케이트 식별자 이름
   *
   * 심볼 테이블에서 조회(없으면 생성)한 심볼로 operand_info를 만들고
   * set_neg_pred()로 부정 표시 후 g_operands에 추가한다.
   *
   * 호출 체인:
   *   ptx.y의 neg_pred_operand 규칙 → [이 함수]
   */
  void add_neg_pred_operand(const char *identifier);

  /*
   * [한국어]
   * add_variables - 변수 선언에 초기화 값을 설정하고 지시어 상태 리셋
   *
   * g_operands에 초기화 값이 있으면 g_last_symbol에 add_initializer로 설정.
   * 완료 후 init_directive_state로 파싱 상태를 초기화한다.
   *
   * 호출 체인:
   *   ptx.y의 variable_declaration 규칙 → [이 함수]
   */
  void add_variables();

  /*
   * [한국어]
   * set_variable_type - 현재 공간/타입 지시어로 type_info 객체를 생성
   *
   * g_space_spec과 g_scalar_type_spec이 유효한지 확인(parse_assert)하고,
   * symbol_table::add_type으로 type_info를 생성하여 g_var_type에 저장한다.
   *
   * 호출 체인:
   *   ptx.y의 type_qualifier 규칙 → [이 함수] → symbol_table::add_type
   */
  void set_variable_type();

  /*
   * [한국어]
   * add_opcode - 현재 명령어 오피코드를 g_opcode에 저장
   *
   * @param opcode: ptx_ir.h의 opcode enum 값
   *
   * 단순 대입; Yacc에서 명령어 토큰을 읽을 때 즉시 호출된다.
   *
   * 호출 체인:
   *   ptx.y의 opcode 규칙 → [이 함수]
   */
  void add_opcode(int opcode);

  /*
   * [한국어]
   * add_pred - 프레디케이트 레지스터를 현재 명령어에 설정
   *
   * @param identifier: 프레디케이트 레지스터 이름 (%p0 등)
   * @param negate: 부정 여부 (0=정상, 1=부정)
   * @param predModifier: 프레디케이트 수식어
   *
   * 심볼 테이블에서 identifier를 조회하고 g_pred, g_neg_pred, g_pred_mod를 설정한다.
   * 선언되지 않은 프레디케이트는 parse_error를 발생시킨다.
   *
   * 호출 체인:
   *   ptx.y의 pred_spec 규칙 → [이 함수]
   */
  void add_pred(const char *identifier, int negate, int predModifier);

  void add_1vector_operand(const char *d1);  // [한국어] 1-원소 벡터 오퍼랜드 추가 (tex.1d용)
  void add_2vector_operand(const char *d1, const char *d2);  // [한국어] .v2 벡터 오퍼랜드 추가
  void add_3vector_operand(const char *d1, const char *d2, const char *d3);  // [한국어] .v3 벡터 오퍼랜드 추가
  void add_4vector_operand(const char *d1, const char *d2, const char *d3,
                           const char *d4);  // [한국어] .v4 벡터 오퍼랜드 추가 (가장 일반적인 벡터 크기)
  void add_8vector_operand(const char *d1, const char *d2, const char *d3,
                           const char *d4, const char *d5, const char *d6,
                           const char *d7, const char *d8);  // [한국어] .v8 벡터 오퍼랜드 추가 (WMMA 8원소용)
  void add_option(int option);  // [한국어] 명령어 옵션 하나를 g_options에 추가
  void add_wmma_option(int option);  // [한국어] WMMA 전용 옵션을 g_wmma_options에 추가
  void add_builtin_operand(int builtin, int dim_modifier);  // [한국어] 내장 레지스터(%tid, %ctaid 등) 오퍼랜드 추가
  void add_memory_operand();  // [한국어] 마지막 오퍼랜드를 메모리 간접 참조로 변경 (ld/st의 [addr] 표기)
  void add_literal_int(int value);  // [한국어] 정수 리터럴 상수 오퍼랜드 추가
  void add_literal_float(float value);  // [한국어] 단정밀도 부동소수점 리터럴 오퍼랜드 추가
  void add_literal_double(double value);  // [한국어] 배정밀도 부동소수점 리터럴 오퍼랜드 추가
  void add_address_operand(const char *identifier, int offset);  // [한국어] 심볼+오프셋 주소 오퍼랜드 추가
  void add_address_operand2(int offset);  // [한국어] 즉치 오프셋만으로 구성된 주소 오퍼랜드 추가

  /*
   * [한국어]
   * add_label - PTX 레이블을 심볼 테이블에 등록하고 g_label에 저장
   *
   * @param identifier: 레이블 이름 (분기 대상; 예: "loop:", "end:")
   *
   * 심볼 테이블에 이미 있으면 기존 심볼 재사용(forward reference 처리),
   * 없으면 새로 add_variable로 생성한다.
   *
   * 호출 체인:
   *   ptx.y의 label 규칙 → [이 함수]
   */
  void add_label(const char *idenfiier);

  void add_vector_spec(int spec);  // [한국어] .v2/.v4 벡터 지시어 값을 g_vector_spec에 저장
  void add_space_spec(enum _memory_space_t spec, int value);  // [한국어] 메모리 공간 지시어 설정; param은 맥락에 따라 kernel/local로 분류
  void add_ptr_spec(enum _memory_space_t spec);  // [한국어] .ptr 대상 공간 설정 (global/local/shared만 허용)
  void add_extern_spec();  // [한국어] .extern 지시어 설정; g_extern_spec = 1

  /*
   * [한국어]
   * add_instruction - 현재 파싱 상태로 ptx_instruction을 생성하여 g_instructions에 추가
   *
   * g_opcode, g_pred, g_operands, g_options 등 현재 명령어 파싱 상태 전체를 모아
   * ptx_instruction 객체를 new로 생성하고, g_instructions 리스트 끝에 추가한다.
   * 소스 파일/라인 정보를 g_inst_lookup에 등록하고, init_instruction_state를 호출하여 상태 초기화.
   *
   * 호출 체인:
   *   ptx.y의 instruction 규칙 → [이 함수] → new ptx_instruction(...)
   */
  void add_instruction();

  /*
   * [한국어]
   * set_return - call 명령어의 반환값 오퍼랜드를 g_return_var에 설정
   *
   * CALL_OP 또는 CALLP_OP에서만 호출됨(parse_assert 검증).
   * g_operands.front()에 set_return 표시를 하고 g_return_var에 복사한다.
   *
   * 호출 체인:
   *   ptx.y의 call_return 규칙 → [이 함수]
   */
  void set_return();

  void add_alignment_spec(int spec);  // [한국어] .align 정렬 값 설정; g_alignment_spec에 저장
  void add_array_initializer();  // [한국어] 배열 초기화 값을 g_last_symbol에 설정
  void add_file(unsigned num, const char *filename);  // [한국어] .file 지시어 처리; g_filename 추론 및 심볼 테이블 복귀
  void add_version_info(float ver, unsigned ext);  // [한국어] .version 지시어; 전역 심볼 테이블에 PTX 버전 설정
  void *reset_symtab();  // [한국어] 현재 심볼 테이블을 전역으로 복귀하고 이전 테이블 포인터 반환
  void set_symtab(void *);  // [한국어] 지정된 심볼 테이블을 현재 테이블로 설정 (CDP 지원용)
  void add_pragma(const char *str);  // [한국어] .pragma 지시어 무시 경고 출력 (미구현)
  void func_header(const char *a);  // [한국어] 함수 헤더 처리 더미 함수 (현재 비어 있음)
  void func_header_info(const char *a);  // [한국어] 함수 헤더 정보 더미 함수
  void func_header_info_int(const char *a, int b);  // [한국어] 정수 인자를 받는 함수 헤더 정보 더미 함수

  /*
   * [한국어]
   * add_constptr - 상수 포인터 변수의 주소를 다른 상수 기준으로 재배치
   *
   * @param identifier1: 재배치할 상수 변수 이름
   * @param identifier2: 기준 상수 변수 이름
   * @param offset: 기준 변수로부터의 바이트 오프셋
   *
   * s1의 주소를 s2의 주소 + offset으로 변경한다.
   * PTX의 alias/layout 제어 지시어 처리에 사용된다.
   *
   * 호출 체인:
   *   ptx.y의 constptr 규칙 → [이 함수]
   */
  void add_constptr(const char *identifier1, const char *identifier2,
                    int offset);

  void target_header(char *a);  // [한국어] .target 지시어 (SM 버전 하나) 처리; set_sm_target 호출
  void target_header2(char *a, char *b);  // [한국어] .target 지시어 (SM 버전 + extension) 처리
  void target_header3(char *a, char *b, char *c);  // [한국어] .target 지시어 (SM 버전 + extension 2개) 처리
  void add_double_operand(const char *d1, const char *d2);  // [한국어] 두 심볼을 조합한 더블 오퍼랜드 추가 (s[$ofs+$r0] 형식)
  void change_memory_addr_space(const char *identifier);  // [한국어] 마지막 오퍼랜드의 메모리 주소 공간 변경 (g/s/c/l 접두사 처리)
  void change_operand_lohi(int lohi);  // [한국어] 마지막 오퍼랜드의 하위(lo)/상위(hi) 비트 선택 설정
  void change_double_operand_type(int addr_type);  // [한국어] 더블 오퍼랜드의 연산 종류 설정 (reg+reg, reg+=reg 등)
  void change_operand_neg();  // [한국어] 마지막 오퍼랜드에 부정(-) 연산 플래그 설정
  void set_immediate_operand_type();  // [한국어] 마지막 오퍼랜드를 즉치 주소(immediate address) 타입으로 변경
  void version_header(double a);  // [한국어] .version 헤더 처리 더미 함수 (g_global_symbol_table의 set_ptx_version으로 위임됨)
  void maxnt_id(int x, int y, int z);  // [한국어] .maxnctapersm/.maxntid 지시어; 최대 스레드 수를 function_info에 설정

  /*
   * [한국어]
   * parse_error_impl - 파싱 오류 처리 함수
   *
   * @param file: 오류가 발생한 C++ 소스 파일명 (__FILE__)
   * @param line: 오류가 발생한 C++ 소스 라인 번호 (__LINE__)
   * @param msg: printf 형식의 오류 메시지
   *
   * 오류 메시지를 포맷팅하여 출력하고, g_error_detected를 1로 설정한 후 abort로 종료한다.
   * 직접 호출보다는 parse_error 매크로를 통해 자동으로 __FILE__, __LINE__이 삽입된다.
   *
   * 호출 체인:
   *   parse_error 매크로 → [이 함수] → ptx_error → abort
   */
  void parse_error_impl(const char *file, unsigned line, const char *msg, ...);

  /*
   * [한국어]
   * parse_assert_impl - 조건 검사를 수행하고 실패 시 parse_error_impl 호출
   *
   * @param test_value: 검사할 조건 값 (0이면 실패)
   * @param file, line, msg: parse_error_impl과 동일
   *
   * test_value == 0이면 parse_error_impl을 호출한다. C의 assert와 유사하나
   * PTX 파싱 오류 맥락에서 적절한 오류 메시지를 출력한다는 점이 다르다.
   *
   * 호출 체인:
   *   parse_assert 매크로 → [이 함수] → parse_error_impl
   */
  void parse_assert_impl(int test_value, const char *file, unsigned line,
                         const char *msg, ...);

  // Jin: handle instructino group for cdp
  /*
   * [한국어]
   * start_inst_group - CDP(CUDA Dynamic Parallelism) 명령어 그룹 시작
   *
   * 현재 심볼 테이블의 새 자식 그룹 스코프로 전환한다.
   * CDP 커널 내에서 디바이스 측 커널 런치를 위한 명령어 그룹 지원.
   *
   * 호출 체인:
   *   ptx.y의 inst_group_start 규칙 → [이 함수] → symbol_table::start_inst_group
   */
  void start_inst_group();

  /*
   * [한국어]
   * end_inst_group - CDP 명령어 그룹 종료; 부모 스코프로 복귀
   *
   * 호출 체인:
   *   ptx.y의 inst_group_end 규칙 → [이 함수] → symbol_table::end_inst_group
   */
  void end_inst_group();

  /*
   * [한국어]
   * check_for_duplicates - 현재 심볼 테이블에 식별자가 이미 있는지 확인
   *
   * @param identifier: 확인할 식별자 이름
   * @return: true면 이미 선언된 중복 식별자
   *
   * add_identifier에서 중복 선언 감지에 사용된다.
   *
   * 호출 체인:
   *   add_identifier → [이 함수] → symbol_table::lookup
   */
  bool check_for_duplicates(const char *identifier);

  /*
   * [한국어]
   * read_parser_environment_variables - 환경변수에서 파서 설정을 읽음
   *
   * PTX_SIM_KERNELFILE: 시뮬레이션할 PTX 파일 경로를 g_filename에 설정.
   * PTX_SIM_DEBUG: 디버그 레벨이 30 이상이면 g_debug_ir_generation을 true로 설정.
   *
   * 호출 체인:
   *   gpgpu_context::init_parser 이전 → [이 함수]
   */
  void read_parser_environment_variables();

  /*
   * [한국어]
   * set_ptx_warp_size - 파서에 셰이더 코어 설정(워프 크기 등)을 연결
   *
   * @param warp_size: 코어 설정 구조체 포인터
   *
   * g_shader_core_config를 설정하여 add_instruction에서 ptx_instruction 생성 시
   * 하드웨어 파라미터를 사용할 수 있게 한다.
   *
   * 호출 체인:
   *   gpgpu_context::init_parser 후 → [이 함수]
   */
  void set_ptx_warp_size(const struct core_config *warp_size);

  /*
   * [한국어]
   * ptx_instruction_lookup - 소스 파일명과 라인 번호로 PTX 명령어 역조회
   *
   * @param filename: PTX 소스 파일 이름
   * @param linenumber: 소스 파일 내 라인 번호
   * @return: 해당 위치의 ptx_instruction*, 없으면 NULL
   *
   * g_inst_lookup 맵을 두 단계로 탐색한다: 파일명 → 라인번호.
   * 디버거나 프로파일러에서 소스 레벨 매핑에 사용된다.
   *
   * 호출 체인:
   *   디버그/프로파일 코드 → [이 함수] → g_inst_lookup 맵 조회
   */
  const class ptx_instruction *ptx_instruction_lookup(const char *filename,
                                                      unsigned linenumber);
};

/*
 * [한국어]
 * decode_token - 정수 토큰 타입 값을 사람이 읽을 수 있는 문자열로 변환
 *
 * @param type: Yacc/Flex가 사용하는 토큰 타입 정수 값
 * @return: g_ptx_token_decode 맵에 등록된 해당 토큰의 문자열 (없으면 빈 문자열)
 *
 * 파싱 오류 메시지나 디버그 출력에서 토큰 번호 대신 이름을 표시하는 데 사용됨.
 * g_ptx_token_decode는 init_parser에서 ptx_parser_decode.def를 통해 채워진다.
 *
 * 호출 체인:
 *   parse_error, PTX_PARSE_DPRINTF 내부 → [이 함수] → g_ptx_token_decode[type]
 */
const char *decode_token(int type);

/*
 * [한국어]
 * read_parser_environment_variables - 전역 수준 파서 환경변수 읽기 래퍼
 *
 * ptx_recognizer::read_parser_environment_variables의 비멤버 함수 버전.
 * 일부 코드에서 recognizer 객체 없이 환경변수를 초기화할 때 사용.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc 등 → [이 함수]
 */
void read_parser_environment_variables();

/* [한국어] 배열이 아닌 일반 식별자 (스칼라 변수)를 나타내는 상수
 * add_identifier의 array_ident 파라미터로 전달됨 */
#define NON_ARRAY_IDENTIFIER 1

/* [한국어] 크기가 명시되지 않은 배열 식별자 (예: .u32 arr[])
 * 나중에 초기화 값에서 크기를 유추하거나 외부에서 설정해야 함 */
#define ARRAY_IDENTIFIER_NO_DIM 2

/* [한국어] 크기가 명시된 배열 식별자 (예: .u32 arr[16])
 * add_identifier에서 array_dim * g_size를 전체 크기로 계산 */
#define ARRAY_IDENTIFIER 3

#endif  // [한국어] ptx_parser_INCLUDED include 가드 끝
