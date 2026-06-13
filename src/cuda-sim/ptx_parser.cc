/*
 * [한국어 설명] PTX 파서 의미 분석 구현 (ptx_parser.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 ptx_parser.h에서 선언된 `ptx_recognizer` 클래스의 모든 멤버 함수를 구현한다.
 * PTX 어셈블리 텍스트를 파싱하는 과정에서 Yacc(Bison) 문법 파일(ptx.y)의 각 규칙이
 * reduce될 때 호출되는 semantic action 함수들이 여기에 정의된다.
 * 또한 `gpgpu_context::init_parser`가 이 파일에서 구현되어, PTX 파일을 열고
 * Lex/Yacc 파서 파이프라인을 구동하는 진입점 역할을 담당한다.
 * 파싱 완료 후에는 각 함수의 `function_info`가 `gpgpu_ptx_assemble`을 통해
 * PC 주소를 할당받아 실행 가능한 PTX IR 상태가 된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CUDA App → libcuda → gpgpusim_entrypoint → [이 파일: gpgpu_context::init_parser]
 *   → ptx_lex_init → ptx_parse (Yacc) → ptx_recognizer의 add_* 함수들
 *   → function_info::add_inst → gpgpu_ptx_assemble → PTX IR 완성
 *   → cuda-sim/ 기능 시뮬레이션, gpgpu-sim/ 타이밍 시뮬레이션
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 시작 전 단 1회 수행됨.
 * 사이클 모델과 무관; 타이밍 시뮬레이션의 전처리 단계다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존하는 모듈: ptx_parser.h, ptx_ir.h (function_info, symbol_table, ptx_instruction,
 *   operand_info), ptx.tab.h (Yacc 생성 토큰/파서 선언), gpgpu_context.h
 * - 이 파일에 의존하는 모듈: gpgpusim_entrypoint.cc (init_parser 호출),
 *   libcuda 런타임 인터셉트 레이어
 * - 데이터 흐름: PTX 텍스트 파일 → Lex 토큰 스트림 → Yacc reduce 액션 →
 *   g_instructions 리스트 → function_info → gpgpu_ptx_assemble → PTX IR
 * - 공유 자료구조: gpgpu_context::g_global_allfiles_symbol_table (모든 함수/변수 등록),
 *   ptx_recognizer의 각 g_* 상태 변수들
 *
 * === 주요 함수/구조체 요약 ===
 * - gpgpu_context::init_parser: PTX 파일을 열고 Lex/Yacc 파이프라인 구동
 * - ptx_recognizer::end_function: 함수 파싱 완료; IR 조립 및 어셈블 수행
 * - ptx_recognizer::add_identifier: 변수를 심볼 테이블에 등록하고 메모리 공간 할당
 * - ptx_recognizer::add_instruction: 파싱된 명령어를 ptx_instruction 객체로 생성
 * - pad_address: 주소 정렬에 필요한 패딩 바이트 수를 계산하는 유틸리티 함수
 */

// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, Wilson W.L. Fung
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

#include "ptx_parser.h"  // [한국어] ptx_recognizer 클래스 선언 포함
#include "../../libcuda/gpgpu_context.h"  // [한국어] 시뮬레이터 전체 컨텍스트 (g_filename, ptx_parser 멤버 접근)
#include "ptx_ir.h"  // [한국어] PTX IR 핵심 타입들: function_info, symbol_table, ptx_instruction

typedef void *yyscan_t;  // [한국어] Flex 재진입 가능 스캐너 핸들; opaque 포인터로 선언
#include <stdarg.h>  // [한국어] va_list, va_start, vsnprintf 등 가변 인자 함수 지원
#include "ptx.tab.h"  // [한국어] Bison(Yacc) 생성 헤더: 토큰 번호 enum, YYSTYPE 정의 포함

/* [한국어] Flex/Bison 생성 함수들의 외부 선언.
 * 이 함수들은 ptx.l과 ptx.y에서 생성된 ptx.tab.c/ptx_lexer.c에 실제 구현이 있다. */
extern int ptx_get_lineno(yyscan_t yyscanner);  // [한국어] 현재 Lex 스캐너가 처리 중인 소스 라인 번호 반환
extern YYSTYPE *ptx_get_lval(yyscan_t yyscanner);  // [한국어] 현재 Lex 토큰의 의미 값(semantic value) 포인터 반환
extern int ptx_error(yyscan_t yyscanner, ptx_recognizer *recognizer,
                     const char *s);  // [한국어] Bison 오류 핸들러; 오류 메시지를 출력하고 파싱 중단
extern int ptx_lex_init(yyscan_t *scanner);  // [한국어] Flex 스캐너 초기화; scanner 핸들 생성
extern void ptx_set_in(FILE *_in_str, yyscan_t yyscanner);  // [한국어] Flex 스캐너의 입력 파일 설정
extern FILE *ptx_get_in(yyscan_t yyscanner);  // [한국어] 현재 Flex 스캐너의 입력 파일 포인터 반환
extern int ptx_parse(yyscan_t scanner, ptx_recognizer *recognizer);  // [한국어] Bison 파서 진입점; 전체 PTX 파일을 파싱
extern int ptx_lex_destroy(yyscan_t scanner);  // [한국어] Flex 스캐너 해제; 할당된 내부 자원 반환

/*
 * [한국어]
 * ptx_recognizer::set_ptx_warp_size - 셰이더 코어 설정을 파서에 연결
 *
 * @param warp_size: 코어 설정 구조체 포인터 (워프 크기, 레지스터 파일 크기 등 포함)
 *
 * add_instruction에서 ptx_instruction 생성 시 하드웨어 파라미터가 필요하므로,
 * 파싱 시작 전에 이 함수를 통해 코어 설정 포인터를 전달해둔다.
 * 파싱 후 어셈블 단계에서도 이 설정을 참조한다.
 *
 * 호출 체인:
 *   gpgpu_context::init_parser → [이 함수] (파싱 전 설정 단계)
 */
void ptx_recognizer::set_ptx_warp_size(const struct core_config *warp_size) {
  g_shader_core_config = warp_size;  // [한국어] 코어 설정 포인터 저장; add_instruction에서 assert(!=0)로 검증됨
}

/*
 * [한국어] PTX_PARSE_DPRINTF 매크로
 *
 * g_debug_ir_generation이 true일 때만 파싱 진행 상황을 stdout에 출력한다.
 * 환경변수 PTX_SIM_DEBUG >= 30으로 활성화된다.
 * 출력 형식: "파일명:라인번호 =>   (소스파일:라인) 메시지"
 * fflush(stdout)으로 즉시 출력 보장 (버퍼링 없이 실시간 확인 가능).
 */
#define PTX_PARSE_DPRINTF(...)                                            \
  if (g_debug_ir_generation) {                                            \
    printf(" %s:%u => ", gpgpu_ctx->g_filename, ptx_get_lineno(scanner)); \
    printf("   (%s:%u) ", __FILE__, __LINE__);                            \
    printf(__VA_ARGS__);                                                  \
    printf("\n");                                                         \
    fflush(stdout);                                                       \
  }

/* [한국어] 정수 토큰 번호 → 문자열 이름 매핑 테이블.
 * 파일-정적(static) 변수이므로 이 파일 내에서만 접근 가능.
 * init_parser에서 ptx_parser_decode.def를 통해 모든 토큰 이름이 등록됨.
 * decode_token 함수를 통해 외부에서 접근 가능. */
static std::map<unsigned, std::string> g_ptx_token_decode;

/*
 * [한국어]
 * decode_token - 정수 토큰 타입을 사람이 읽을 수 있는 문자열로 변환
 *
 * @param type: Yacc 토큰 정수 값 또는 memory_space_t enum 값
 * @return: g_ptx_token_decode에 등록된 토큰 이름 문자열의 C 포인터
 *
 * g_ptx_token_decode가 static이므로 이 래퍼 함수로만 외부 접근 가능.
 * 파싱 오류 메시지나 디버그 출력에서 토큰 번호 대신 이름을 표시한다.
 *
 * 호출 체인:
 *   parse_error_impl, PTX_PARSE_DPRINTF → [이 함수]
 */
const char *decode_token(int type) { return g_ptx_token_decode[type].c_str(); }

/*
 * [한국어]
 * ptx_recognizer::read_parser_environment_variables - 환경변수에서 파서 설정 읽기
 *
 * PTX_SIM_KERNELFILE: 파싱할 PTX 파일 경로를 gpgpu_ctx->g_filename에 설정.
 *   이 값이 설정되면 파싱 중 소스 위치 정보(파일명)로 사용됨.
 * PTX_SIM_DEBUG: 정수 디버그 레벨을 읽어 30 이상이면 g_debug_ir_generation을 true로 활성화.
 *   이후 PTX_PARSE_DPRINTF 매크로가 모든 파싱 단계를 출력한다.
 *
 * 호출 체인:
 *   gpgpu_context::init_parser (파싱 시작 전) → [이 함수]
 */
void ptx_recognizer::read_parser_environment_variables() {
  gpgpu_ctx->g_filename = getenv("PTX_SIM_KERNELFILE");  // [한국어] PTX 파일 경로 환경변수 읽기; NULL이면 .file 지시어에서 추론
  char *dbg_level = getenv("PTX_SIM_DEBUG");             // [한국어] 디버그 레벨 환경변수 읽기
  if (dbg_level && strlen(dbg_level)) {                  // [한국어] 환경변수가 설정되어 있고 비어있지 않으면
    int debug_execution = 0;                             // [한국어] 디버그 레벨 정수값 저장 변수
    sscanf(dbg_level, "%d", &debug_execution);           // [한국어] 문자열을 정수로 파싱
    if (debug_execution >= 30) g_debug_ir_generation = true;  // [한국어] 레벨 30 이상이면 IR 생성 디버그 출력 활성화
  }
}

/*
 * [한국어]
 * ptx_recognizer::init_directive_state - 지시어 파싱 상태를 초기값으로 리셋
 *
 * 각 지시어(.reg, .global, .param 등) 파싱을 새로 시작할 때 호출된다.
 * 이전 지시어의 잔여 상태(타입, 공간, 크기 등)가 다음 지시어에 영향을 주지 않도록
 * 모든 관련 파서 상태를 안전한 센티넬 값으로 초기화한다.
 *
 * 초기화 항목: g_space_spec, g_ptr_spec, g_scalar_type_spec, g_vector_spec,
 *   g_opcode, g_alignment_spec, g_size, g_extern_spec, g_scalar_type 리스트,
 *   g_operands 리스트, g_last_symbol
 *
 * 호출 체인:
 *   init_instruction_state → [이 함수]
 *   start_function → [이 함수]
 *   add_directive → [이 함수]
 *   end_function → [이 함수]
 *   add_variables → [이 함수]
 */
void ptx_recognizer::init_directive_state() {
  PTX_PARSE_DPRINTF("init_directive_state");  // [한국어] 디버그 모드에서 상태 초기화 로그 출력
  g_space_spec = undefined_space;  // [한국어] 메모리 공간 지시어 초기화 (.global/.shared 등 미설정 상태)
  g_ptr_spec = undefined_space;    // [한국어] 포인터 대상 공간 초기화 (포인터 없음)
  g_scalar_type_spec = -1;         // [한국어] 스칼라 타입 초기화 (타입 미설정)
  g_vector_spec = -1;              // [한국어] 벡터 지시어 초기화 (벡터 없음)
  g_opcode = -1;                   // [한국어] 오피코드 초기화 (-1은 레이블 전용 또는 미설정)
  g_alignment_spec = -1;           // [한국어] 정렬 지시어 초기화 (미설정)
  g_size = -1;                     // [한국어] 타입 크기 초기화 (add_scalar_type_spec에서 설정됨)
  g_extern_spec = 0;               // [한국어] extern 지시어 플래그 초기화 (외부 선언 아님)
  g_scalar_type.clear();           // [한국어] 스칼라 타입 목록 초기화 (이전 파싱 잔여 제거)
  g_operands.clear();              // [한국어] 오퍼랜드 목록 초기화 (이전 명령어 오퍼랜드 제거)
  g_last_symbol = NULL;            // [한국어] 마지막 심볼 포인터 초기화
}

/*
 * [한국어]
 * ptx_recognizer::init_instruction_state - 명령어 파싱 상태를 초기값으로 리셋
 *
 * 새 PTX 명령어 파싱을 시작할 때 호출된다. 프레디케이트, 레이블, 옵션 등
 * 명령어별 상태를 초기화하고, init_directive_state를 호출하여 공통 파서 상태도 초기화한다.
 *
 * 초기화 항목: g_pred, g_neg_pred, g_pred_mod, g_label, g_opcode(재초기화),
 *   g_options, g_wmma_options, g_return_var
 * 이후 init_directive_state를 호출하여 타입/공간/오퍼랜드 상태도 초기화.
 *
 * 호출 체인:
 *   ptx.y의 instruction 규칙 시작 → [이 함수] → init_directive_state
 *   add_instruction 완료 후에도 [이 함수] 호출하여 다음 명령어 준비
 */
void ptx_recognizer::init_instruction_state() {
  PTX_PARSE_DPRINTF("init_instruction_state");  // [한국어] 디버그 모드에서 명령어 상태 초기화 로그 출력
  g_pred = NULL;                       // [한국어] 프레디케이트 레지스터 심볼 초기화 (조건부 실행 없음)
  g_neg_pred = 0;                      // [한국어] 프레디케이트 부정 플래그 초기화 (부정 없음)
  g_pred_mod = -1;                     // [한국어] 프레디케이트 수식어 초기화 (없음)
  g_label = NULL;                      // [한국어] 레이블 심볼 초기화 (이 명령어에 레이블 없음)
  g_opcode = -1;                       // [한국어] 오피코드 초기화 (명령어 토큰 파싱 전)
  g_options.clear();                   // [한국어] 명령어 옵션 목록 초기화 (.ftz, .rn 등 제거)
  g_wmma_options.clear();              // [한국어] WMMA 전용 옵션 목록 초기화
  g_return_var = operand_info(gpgpu_ctx);  // [한국어] 반환값 오퍼랜드를 기본 상태로 초기화 (call 이전 상태)
  init_directive_state();              // [한국어] 타입/공간/크기/오퍼랜드 공통 상태도 초기화
}

/*
 * [한국어]
 * gpgpu_context::init_parser - PTX 파일을 열고 Lex/Yacc 파서 파이프라인 구동
 *
 * @param ptx_filename: 파싱할 PTX 파일 경로 (null-종료 문자열)
 * @return: 파싱 완료 후 모든 함수/변수가 등록된 전역 심볼 테이블 포인터
 *
 * 이 함수는 PTX 텍스트 파싱의 최상위 진입점이다. 아래 과정을 수행한다:
 * 1) g_filename을 PTX 파일 이름으로 설정 (오류 메시지/소스 위치 추적용)
 * 2) 전역 심볼 테이블(g_global_allfiles_symbol_table)이 없으면 생성
 *    (여러 PTX 파일을 파싱할 때는 한 테이블을 공유)
 * 3) g_ptx_token_decode 맵에 토큰 번호→이름 매핑 등록 (디버그/오류 출력용)
 * 4) Flex 스캐너 초기화(ptx_lex_init), 파서 상태 초기화
 * 5) PTX 파일 열기(fopen), 스캐너에 입력 연결(ptx_set_in)
 * 6) Yacc 파서 실행(ptx_parse): 전체 파일을 파싱하여 PTX IR 생성
 * 7) 스캐너 해제(ptx_lex_destroy), 파일 닫기(fclose)
 * 8) 전역 심볼 테이블 반환
 *
 * 실행 컨텍스트: 호스트 유저스페이스, 커널 실행 전 초기화 단계 (1회).
 * 에러 발생 시: ptx_recognizer::parse_error_impl에서 abort() 호출로 즉시 종료.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc (PTX 파일 로딩) → [이 함수]
 *     → ptx_lex_init → ptx_parse → ptx.y 규칙들 → add_instruction/add_identifier 등
 *     → gpgpu_ptx_assemble → PTX IR 완성
 */
symbol_table *gpgpu_context::init_parser(const char *ptx_filename) {
  g_filename = strdup(ptx_filename);  // [한국어] PTX 파일 이름을 복사하여 전역에 저장; 오류 메시지/소스 위치 추적에 사용
  if (g_global_allfiles_symbol_table == NULL) {  // [한국어] 전역 심볼 테이블이 아직 없으면 (첫 번째 PTX 파일 파싱)
    g_global_allfiles_symbol_table =
        new symbol_table("global_allfiles", 0, NULL, this);  // [한국어] "global_allfiles" 이름의 최상위 심볼 테이블 생성
    ptx_parser->g_global_symbol_table = ptx_parser->g_current_symbol_table =
        g_global_allfiles_symbol_table;  // [한국어] 파서의 전역/현재 심볼 테이블을 새로 만든 최상위 테이블로 설정
  }
  /* 주석 처리된 코드: 여러 파일을 별도 심볼 테이블로 파싱하는 방식은 현재 사용 안 함.
     모든 PTX 파일은 하나의 global_allfiles 테이블에 통합됨. */

  /* [한국어] ptx_parser_decode.def 파일의 DEF(토큰번호, 문자열) 매크로를
   * 이용하여 모든 Yacc 토큰 이름을 g_ptx_token_decode에 등록.
   * 이 과정은 한 번만 수행되면 충분하므로 init_parser 첫 호출 시 처리. */
#define DEF(X, Y) g_ptx_token_decode[X] = Y;  // [한국어] 토큰 번호 X에 이름 문자열 Y를 등록하는 매크로 인스턴스화
#include "ptx_parser_decode.def"  // [한국어] 모든 PTX 토큰 정의 파일 포함; DEF 매크로를 각 토큰에 대해 확장
#undef DEF  // [한국어] DEF 매크로 해제; 이후 다른 곳에서 충돌 방지

  /* [한국어] memory_space_t enum 값들은 별도로 등록; ptx_parser_decode.def에 없는 항목들 */
  g_ptx_token_decode[undefined_space] = "undefined_space";         // [한국어] 0=미정의 공간 (초기값 센티넬)
  g_ptx_token_decode[undefined_space] = "undefined_space=0";       // [한국어] 중복 등록이지만 더 명확한 설명으로 덮어쓰기
  g_ptx_token_decode[reg_space] = "reg_space";                     // [한국어] 레지스터 공간 (.reg)
  g_ptx_token_decode[local_space] = "local_space";                 // [한국어] 로컬 메모리 (.local; 스레드 고유 스택 공간)
  g_ptx_token_decode[shared_space] = "shared_space";               // [한국어] 공유 메모리 (.shared; CTA 내 스레드 공유)
  g_ptx_token_decode[param_space_unclassified] = "param_space_unclassified";  // [한국어] 맥락 미결정 파라미터 공간
  g_ptx_token_decode[param_space_kernel] = "param_space_kernel";   // [한국어] 커널 파라미터 공간 (호스트→디바이스 전달)
  g_ptx_token_decode[param_space_local] = "param_space_local";     // [한국어] 로컬 파라미터 공간 (함수 호출 인자)
  g_ptx_token_decode[const_space] = "const_space";                 // [한국어] 상수 메모리 (.const; 읽기 전용, 캐시됨)
  g_ptx_token_decode[tex_space] = "tex_space";                     // [한국어] 텍스처 메모리 (.tex; 하드웨어 보간 가능)
  g_ptx_token_decode[surf_space] = "surf_space";                   // [한국어] 서피스 메모리 (.surf; 쓰기 가능 텍스처)
  g_ptx_token_decode[global_space] = "global_space";               // [한국어] 전역 메모리 (.global; 모든 스레드 공유 DRAM)
  g_ptx_token_decode[generic_space] = "generic_space";             // [한국어] 제네릭 주소 공간 (런타임에 공간 결정)
  g_ptx_token_decode[instruction_space] = "instruction_space";     // [한국어] 명령어 공간 (PC 주소 영역)

  ptx_lex_init(&(ptx_parser->scanner));  // [한국어] Flex 재진입 가능 스캐너 초기화; scanner 핸들 할당
  ptx_parser->init_directive_state();    // [한국어] 지시어 파서 상태 초기화 (파싱 전 깨끗한 상태 보장)
  ptx_parser->init_instruction_state();  // [한국어] 명령어 파서 상태 초기화 (프레디케이트, 옵션 등 초기화)

  FILE *ptx_in;  // [한국어] PTX 파일 스트림 포인터
  ptx_in = fopen(ptx_filename, "r");  // [한국어] PTX 파일을 읽기 모드로 열기; 실패 시 NULL (에러 처리 없음 — 주의!)
  ptx_set_in(ptx_in, ptx_parser->scanner);  // [한국어] Flex 스캐너의 입력 파일을 PTX 파일 스트림으로 설정
  ptx_parse(ptx_parser->scanner, ptx_parser);  // [한국어] Bison 파서 실행; PTX 전체 파일을 파싱하고 PTX IR 생성
  ptx_in = ptx_get_in(ptx_parser->scanner);  // [한국어] 파싱 후 스캐너에서 파일 포인터 회수 (fclose를 위해)
  ptx_lex_destroy(ptx_parser->scanner);  // [한국어] Flex 스캐너가 할당한 내부 자원 해제
  fclose(ptx_in);  // [한국어] PTX 파일 닫기
  return ptx_parser->g_global_symbol_table;  // [한국어] 모든 함수/변수가 등록된 전역 심볼 테이블 반환
}

/*
 * [한국어]
 * ptx_recognizer::start_function - 새 함수 파싱 시작을 알리는 진입 함수
 *
 * @param entry_point: 함수 종류 플래그
 *   0 = 일반 device 함수 (.func)
 *   1 = 커널 엔트리 포인트 (.entry; 호스트에서 직접 런치 가능)
 *   2 = extern 선언 (.extern; 다른 모듈에 구현됨)
 *
 * 파서 상태를 초기화하고 이 함수 파싱에 필요한 변수들을 설정한다.
 * g_func_info를 NULL로 초기화하여 add_function_name 이전 상태를 명확히 한다.
 * g_entry_func_param_index를 0으로 리셋하여 파라미터 인덱싱을 처음부터 시작한다.
 *
 * 호출 체인:
 *   ptx.y의 func_decl 규칙 → [이 함수] → init_directive_state, init_instruction_state
 */
void ptx_recognizer::start_function(int entry_point) {
  PTX_PARSE_DPRINTF("start_function");  // [한국어] 디버그 모드에서 함수 파싱 시작 로그 출력
  init_directive_state();               // [한국어] 이전 지시어 잔여 상태 제거
  init_instruction_state();             // [한국어] 이전 명령어 잔여 상태 제거
  g_entry_point = entry_point;          // [한국어] 함수 종류 저장 (add_space_spec에서 param 분류에 사용)
  g_func_info = NULL;                   // [한국어] 함수 정보 초기화; add_function_name에서 실제 할당됨
  g_entry_func_param_index = 0;         // [한국어] 파라미터 인덱스를 0부터 시작
}

/*
 * [한국어]
 * ptx_recognizer::add_function_name - 파싱된 함수 이름을 전역 심볼 테이블에 등록
 *
 * @param name: PTX 소스의 함수 이름 문자열 (예: "matrixMul", "_Z9kernelFunci")
 *
 * 이 함수가 하는 일:
 * 1) g_global_symbol_table::add_function_decl으로 함수를 선언하고
 *    g_func_info(function_info)와 g_current_symbol_table을 함수 전용으로 교체
 * 2) 반환값 변수가 캐시되어 있으면(g_add_identifier_cached__identifier != NULL)
 *    그것을 재처리하여 function_info의 반환값으로 등록; 캐시 해제 후 NULL로 클리어
 * 3) 이미 선언된 함수(prior_decl)이면 기존 인자 목록 제거 후 재정의
 * 4) g_global_symbol_table::add_function으로 소스 위치와 함께 최종 등록
 *
 * 실행 순서 배경: PTX .func 선언에서 반환값 변수가 함수 이름보다 먼저 파싱될 수 있다.
 * 그때 g_func_info가 아직 NULL이므로 반환값 식별자를 캐시했다가 여기서 처리한다.
 *
 * 호출 체인:
 *   ptx.y의 func_name 규칙 → [이 함수]
 *     → symbol_table::add_function_decl (g_func_info 설정)
 *     → (캐시된 반환값이 있으면) add_identifier, function_info::add_return_var
 *     → symbol_table::add_function (최종 등록)
 */
void ptx_recognizer::add_function_name(const char *name) {
  PTX_PARSE_DPRINTF(
      "add_function_name %s %s", name,
      ((g_entry_point == 1) ? "(entrypoint)"
                            : ((g_entry_point == 2) ? "(extern)" : "")));
  // [한국어] 전역 심볼 테이블에 함수 선언; g_func_info와 g_current_symbol_table을 함수 전용으로 교체
  bool prior_decl = g_global_symbol_table->add_function_decl(
      name, g_entry_point, &g_func_info, &g_current_symbol_table);
  if (g_add_identifier_cached__identifier) {  // [한국어] 반환값 변수 캐시가 있으면 (함수 이름 전에 반환값이 먼저 파싱된 경우)
    add_identifier(g_add_identifier_cached__identifier,  // [한국어] 캐시된 반환값 식별자를 이제 처리
                   g_add_identifier_cached__array_dim,
                   g_add_identifier_cached__array_ident);
    free(g_add_identifier_cached__identifier);  // [한국어] strdup으로 할당된 캐시 문자열 메모리 해제
    g_add_identifier_cached__identifier = NULL; // [한국어] 캐시 포인터를 NULL로 클리어 (use-after-free 방지)
    g_func_info->add_return_var(g_last_symbol); // [한국어] 처리된 심볼을 함수의 반환값으로 등록
    init_directive_state();                     // [한국어] 반환값 처리 후 지시어 상태 초기화
  }
  if (prior_decl) {  // [한국어] 이미 선언된 함수라면 (forward declaration 또는 재정의)
    g_func_info->remove_args();  // [한국어] 이전 선언의 인자 목록 제거; 이번 선언으로 덮어씀
  }
  g_global_symbol_table->add_function(g_func_info, gpgpu_ctx->g_filename,
                                      ptx_get_lineno(scanner));  // [한국어] 함수를 소스 위치와 함께 전역 심볼 테이블에 최종 등록
}

// Jin: handle instruction group for cdp
/*
 * [한국어]
 * ptx_recognizer::start_inst_group - CDP 명령어 그룹 스코프 시작
 *
 * CDP(CUDA Dynamic Parallelism): GPU 커널 내에서 또 다른 커널을 런치하는 기능.
 * CDP 관련 명령어들은 별도 스코프(자식 심볼 테이블)에서 처리된다.
 * symbol_table::start_inst_group이 새 자식 스코프를 생성하고 반환한다.
 *
 * 호출 체인:
 *   ptx.y의 inst_group_start 규칙 → [이 함수] → symbol_table::start_inst_group
 */
void ptx_recognizer::start_inst_group() {
  PTX_PARSE_DPRINTF("start_instruction_group");  // [한국어] 디버그 모드에서 그룹 시작 로그 출력
  g_current_symbol_table = g_current_symbol_table->start_inst_group();  // [한국어] 현재 테이블의 자식 스코프로 진입
}

/*
 * [한국어]
 * ptx_recognizer::end_inst_group - CDP 명령어 그룹 스코프 종료; 부모 스코프 복귀
 *
 * start_inst_group으로 들어간 자식 스코프에서 빠져나와 부모 심볼 테이블로 복귀한다.
 *
 * 호출 체인:
 *   ptx.y의 inst_group_end 규칙 → [이 함수] → symbol_table::end_inst_group
 */
void ptx_recognizer::end_inst_group() {
  PTX_PARSE_DPRINTF("end_instruction_group");  // [한국어] 디버그 모드에서 그룹 종료 로그 출력
  g_current_symbol_table = g_current_symbol_table->end_inst_group();  // [한국어] 부모 심볼 테이블로 복귀
}

/*
 * [한국어]
 * ptx_recognizer::add_directive - 지시어 파싱 완료를 알리는 단순 리셋 함수
 *
 * Yacc에서 지시어 규칙이 ';'로 완성될 때 호출된다.
 * init_directive_state를 호출하여 파서 상태를 초기화하는 것이 전부다.
 * 이 함수의 존재 이유: Yacc 문법에서 동일한 시점에 항상 호출되는 명확한 훅 포인트 제공.
 *
 * 호출 체인:
 *   ptx.y의 directive 규칙 → [이 함수] → init_directive_state
 */
void ptx_recognizer::add_directive() {
  PTX_PARSE_DPRINTF("add_directive");  // [한국어] 디버그 모드에서 지시어 완료 로그 출력
  init_directive_state();              // [한국어] 지시어 파서 상태 초기화
}

/* [한국어] mymax 매크로: 두 값 중 큰 값 반환. 표준 max를 피하기 위한 로컬 매크로.
 * end_function에서 레지스터 수 최대값 추적에 사용됨. */
#define mymax(a, b) ((a) > (b) ? (a) : (b))

/*
 * [한국어]
 * ptx_recognizer::end_function - 함수 파싱 완료 처리; PTX IR 조립 및 어셈블
 *
 * 이 함수는 함수 파싱의 최종 단계를 수행한다:
 * 1) 파서 상태 초기화 (다음 함수 파싱 준비)
 * 2) g_max_regs_per_thread 갱신: 현재 함수의 레지스터 사용량을 최대값으로 누적
 *    (next_reg_num()-1이 현재 함수에서 사용된 마지막 레지스터 번호)
 * 3) g_func_info에 g_instructions 목록 전달 (function_info::add_inst)
 * 4) g_instructions 리스트 초기화 (메모리 반환)
 * 5) gpgpu_ptx_assemble 호출: 각 명령어에 PC 주소 할당 → 실행 가능 IR로 변환
 * 6) g_current_symbol_table을 전역 심볼 테이블로 복귀
 *
 * 호출 체인:
 *   ptx.y의 func_body 규칙 → [이 함수]
 *     → g_func_info->add_inst(g_instructions) → g_instructions 저장
 *     → gpgpu_ptx_assemble(name, g_func_info) → PC 주소 할당
 */
void ptx_recognizer::end_function() {
  PTX_PARSE_DPRINTF("end_function");  // [한국어] 디버그 모드에서 함수 완료 로그 출력

  init_directive_state();    // [한국어] 지시어 파서 상태 초기화 (잔여 상태 제거)
  init_instruction_state();  // [한국어] 명령어 파서 상태 초기화 (잔여 명령어 상태 제거)
  g_max_regs_per_thread = mymax(g_max_regs_per_thread,
                                (g_current_symbol_table->next_reg_num() - 1));
  // [한국어] 현재 함수의 최대 레지스터 번호를 누적 최대값으로 갱신.
  //          next_reg_num()-1이 이 함수에서 사용된 마지막 레지스터 번호임.
  //          이 값은 스케줄러나 AccelWattch 전력 모델에서 레지스터 파일 점유량 계산에 사용됨.
  g_func_info->add_inst(g_instructions);  // [한국어] 파싱된 명령어 목록을 function_info에 전달
  g_instructions.clear();                 // [한국어] 명령어 목록 초기화 (다음 함수 파싱 준비; 메모리는 function_info가 관리)
  gpgpu_ptx_assemble(g_func_info->get_name(), g_func_info);  // [한국어] 각 명령어에 PC 주소 할당; 실행 가능한 PTX IR로 변환
  g_current_symbol_table = g_global_symbol_table;  // [한국어] 심볼 테이블을 전역으로 복귀 (함수 스코프 종료)

  PTX_PARSE_DPRINTF("function %s, PC = %llu\n", g_func_info->get_name().c_str(),
                    g_func_info->get_start_PC());  // [한국어] 디버그 모드에서 함수 시작 PC 출력
}

/* [한국어] parse_error 매크로: parse_error_impl을 __FILE__, __LINE__과 함께 호출.
 * 직접 parse_error_impl을 부르는 대신 이 매크로를 쓰면 오류 발생 위치(C++ 소스 파일:라인)를
 * 자동으로 포함하므로 디버깅이 용이하다. */
#define parse_error(msg, ...) \
  parse_error_impl(__FILE__, __LINE__, msg, ##__VA_ARGS__)

/* [한국어] parse_assert 매크로: cond가 0이면 parse_error_impl을 호출하는 조건 검사.
 * C의 assert와 유사하지만 파싱 오류 맥락에서 적절한 오류 메시지를 출력한다.
 * parse_error_impl 내부에서 abort()를 호출하므로 실패 시 즉시 종료된다. */
#define parse_assert(cond, msg, ...) \
  parse_assert_impl((cond), __FILE__, __LINE__, msg, ##__VA_ARGS__)

/*
 * [한국어]
 * ptx_recognizer::parse_error_impl - PTX 파싱 오류 처리 함수
 *
 * @param file: 오류가 발생한 C++ 소스 파일명 (parse_error 매크로가 __FILE__ 전달)
 * @param line: 오류가 발생한 C++ 소스 라인 번호 (parse_error 매크로가 __LINE__ 전달)
 * @param msg: printf 형식의 오류 메시지 문자열
 * @param ...: printf 가변 인자
 *
 * PTX 파싱 중 오류가 발생했을 때 사람이 읽을 수 있는 오류 메시지를 출력하고
 * g_error_detected를 1로 설정한 후 파싱을 중단(abort)한다.
 * 출력 형식: "PTX파일:라인번호: Parse error: 메시지 (C++소스:C++라인)"
 * va_list 사용: printf처럼 가변 인자를 받아 vsnprintf로 포맷팅.
 *
 * 호출 체인:
 *   parse_error 매크로 → [이 함수] → ptx_error(scanner) → abort() → exit(1)
 *   (abort 이후 exit(1)은 도달하지 않음; 방어적 코딩)
 */
void ptx_recognizer::parse_error_impl(const char *file, unsigned line,
                                      const char *msg, ...) {
  va_list ap;          // [한국어] 가변 인자 목록 핸들
  char buf[1024];      // [한국어] 포맷팅된 오류 메시지 버퍼 (1KB 충분)
  va_start(ap, msg);   // [한국어] 가변 인자 목록 초기화; msg 이후 인자들을 ap에 연결
  vsnprintf(buf, 1024, msg, ap);  // [한국어] printf 형식 메시지를 buf에 포맷팅; 오버플로우 방지 위해 snprintf 사용
  va_end(ap);          // [한국어] 가변 인자 목록 정리

  g_error_detected = 1;  // [한국어] 파싱 오류 발생 표시 (상위 코드에서 확인 가능)
  printf("%s:%u: Parse error: %s (%s:%u)\n\n", gpgpu_ctx->g_filename,
         ptx_get_lineno(scanner), buf, file, line);
  // [한국어] 오류 메시지 출력: "PTX파일:라인 에러: 내용 (C++파일:라인)"
  //          gpgpu_ctx->g_filename: PTX 소스 파일 이름
  //          ptx_get_lineno(scanner): PTX 소스 라인 번호
  //          buf: 포맷팅된 오류 메시지
  //          file, line: 이 오류를 감지한 C++ 소스 위치
  ptx_error(scanner, this, NULL);  // [한국어] Bison 오류 핸들러 호출; 파서 상태 정리
  abort();   // [한국어] 즉시 프로세스 강제 종료 (코어 덤프 생성; 디버깅 목적)
  exit(1);   // [한국어] abort 이후 도달 불가능; 컴파일러 경고 억제 목적
}

/*
 * [한국어]
 * ptx_recognizer::parse_assert_impl - 조건 검사 및 실패 시 parse_error_impl 호출
 *
 * @param test_value: 검사할 조건값 (0이면 실패, 0 아니면 통과)
 * @param file, line, msg: parse_error_impl과 동일한 오류 정보
 *
 * test_value == 0이면 parse_error_impl을 호출하여 오류 처리한다.
 * 이 함수의 가변 인자(msg 포맷팅)는 현재 parse_error_impl에 그대로 msg만 전달하는
 * 단순 버전으로 구현되어 있다 (가변 인자가 실제로 msg 내에 포함됨).
 *
 * 호출 체인:
 *   parse_assert 매크로 → [이 함수] → parse_error_impl (test_value == 0인 경우)
 */
void ptx_recognizer::parse_assert_impl(int test_value, const char *file,
                                       unsigned line, const char *msg, ...) {
  va_list ap;          // [한국어] 가변 인자 목록 (현재는 사용되지 않지만 인터페이스 통일성을 위해 존재)
  char buf[1024];      // [한국어] 포맷팅 버퍼 (현재는 사용되지 않음)
  va_start(ap, msg);   // [한국어] 가변 인자 목록 초기화
  vsnprintf(buf, 1024, msg, ap);  // [한국어] 메시지 포맷팅 (결과는 사용되지 않음)
  va_end(ap);          // [한국어] 가변 인자 목록 정리

  if (test_value == 0) parse_error_impl(file, line, msg);  // [한국어] 조건 실패 시 parse_error_impl로 오류 처리 위임
}

/*
 * [한국어]
 * ptx_recognizer::set_return - call 명령어의 반환값 오퍼랜드를 g_return_var에 설정
 *
 * PTX "call (%rd0), func, (args);" 형식에서 (%rd0)에 해당하는 반환값 오퍼랜드를 처리한다.
 * CALL_OP 또는 CALLP_OP에서만 유효하며, 다른 명령어에서 호출되면 parse_assert 실패.
 * g_operands.front()를 반환값으로 표시(set_return)하고 g_return_var에 복사한다.
 *
 * 호출 체인:
 *   ptx.y의 call_return 규칙 → [이 함수]
 */
void ptx_recognizer::set_return() {
  parse_assert((g_opcode == CALL_OP || g_opcode == CALLP_OP),
               "only call can have return value");  // [한국어] call/callp 명령어만 반환값 가능; 다른 명령어 시 파싱 오류
  g_operands.front().set_return();  // [한국어] 첫 번째 오퍼랜드를 반환값으로 표시 (callee → caller 데이터 전달 대상)
  g_return_var = g_operands.front();  // [한국어] g_return_var에 복사; add_instruction에서 ptx_instruction 생성 시 전달
}

/*
 * [한국어]
 * ptx_recognizer::ptx_instruction_lookup - 소스 파일명과 라인 번호로 PTX 명령어 역조회
 *
 * @param filename: 조회할 PTX 소스 파일 이름
 * @param linenumber: 조회할 소스 라인 번호
 * @return: 해당 위치의 ptx_instruction 포인터; 없으면 NULL
 *
 * g_inst_lookup 중첩 맵에서 2단계 탐색(파일명 → 라인번호)으로 명령어를 찾는다.
 * 디버거나 프로파일러에서 소스 레벨 매핑에 사용된다.
 *
 * 호출 체인:
 *   디버그/프로파일 코드 → [이 함수] → g_inst_lookup 맵 조회
 */
const ptx_instruction *ptx_recognizer::ptx_instruction_lookup(
    const char *filename, unsigned linenumber) {
  std::map<std::string, std::map<unsigned, const ptx_instruction *> >::iterator
      f = g_inst_lookup.find(filename);  // [한국어] 파일명으로 외부 맵 탐색
  if (f == g_inst_lookup.end()) return NULL;  // [한국어] 해당 파일 이름이 없으면 NULL 반환
  std::map<unsigned, const ptx_instruction *>::iterator l =
      f->second.find(linenumber);  // [한국어] 찾은 파일 내에서 라인 번호로 탐색
  if (l == f->second.end()) return NULL;  // [한국어] 해당 라인 번호의 명령어가 없으면 NULL 반환
  return l->second;  // [한국어] 해당 위치의 ptx_instruction* 반환
}

/*
 * [한국어]
 * ptx_recognizer::add_instruction - 파싱 상태를 바탕으로 ptx_instruction 객체 생성
 *
 * 현재까지 파싱된 오피코드, 프레디케이트, 오퍼랜드, 옵션, 타입, 공간 지시어 등을
 * 모두 모아 ptx_instruction 객체를 new로 동적 생성하고 g_instructions 리스트에 추가한다.
 * 동시에 g_inst_lookup에 소스 위치(파일명/라인번호) → 명령어 매핑도 등록한다.
 * 완료 후 init_instruction_state를 호출하여 다음 명령어 파싱을 위해 상태를 초기화한다.
 *
 * assert(g_shader_core_config != 0): 워프 크기 설정이 반드시 먼저 되어 있어야 함.
 *
 * 호출 체인:
 *   ptx.y의 instruction 규칙 완성 → [이 함수]
 *     → new ptx_instruction(...) → g_instructions.push_back
 *     → g_inst_lookup 등록 → init_instruction_state
 */
void ptx_recognizer::add_instruction() {
  PTX_PARSE_DPRINTF("add_instruction: %s",
                    ((g_opcode > 0) ? g_opcode_string[g_opcode] : "<label>"));
  // [한국어] 디버그 모드에서 추가되는 명령어 이름 출력; g_opcode <= 0이면 레이블 전용 항목
  assert(g_shader_core_config != 0);  // [한국어] ptx_instruction 생성 전에 코어 설정이 반드시 있어야 함
  ptx_instruction *i = new ptx_instruction(
      g_opcode, g_pred, g_neg_pred, g_pred_mod, g_label, g_operands,
      g_return_var, g_options, g_wmma_options, g_scalar_type, g_space_spec,
      gpgpu_ctx->g_filename, ptx_get_lineno(scanner), linebuf,
      g_shader_core_config, gpgpu_ctx);
  // [한국어] 현재 명령어 파싱 상태 전체를 ptx_instruction 객체로 생성.
  //          g_opcode: 명령어 종류, g_pred/g_neg_pred/g_pred_mod: 프레디케이트
  //          g_label: 레이블, g_operands: 피연산자, g_return_var: call 반환값
  //          g_options/g_wmma_options: 수식어, g_scalar_type: 타입 목록
  //          g_space_spec: 메모리 공간, g_filename/linebuf: 소스 위치 디버그 정보
  //          g_shader_core_config: 워프 크기 등 하드웨어 파라미터
  g_instructions.push_back(i);  // [한국어] 현재 함수의 명령어 목록에 추가; end_function에서 function_info에 전달됨
  g_inst_lookup[gpgpu_ctx->g_filename][ptx_get_lineno(scanner)] = i;  // [한국어] 소스 위치 → 명령어 역방향 맵 등록 (디버그/프로파일용)
  init_instruction_state();  // [한국어] 명령어 파싱 상태 초기화; 다음 명령어 파싱 준비
}

/*
 * [한국어]
 * ptx_recognizer::add_variables - 변수 선언에 초기화 값 설정 및 지시어 상태 리셋
 *
 * 현재 오퍼랜드 목록(g_operands)이 비어있지 않으면, 마지막으로 추가된 심볼(g_last_symbol)에
 * 초기화 값을 설정한다. PTX에서 ".global .s32 arr[] = {1, 2, 3};" 같은 초기화 선언을 처리한다.
 * 완료 후 init_directive_state로 파서 상태를 초기화한다.
 *
 * 호출 체인:
 *   ptx.y의 variable_declaration 규칙 → [이 함수] → symbol::add_initializer
 */
void ptx_recognizer::add_variables() {
  PTX_PARSE_DPRINTF("add_variables");  // [한국어] 디버그 모드에서 변수 처리 로그 출력
  if (!g_operands.empty()) {  // [한국어] 초기화 값이 있으면 (배열 초기화 등)
    assert(g_last_symbol != NULL);  // [한국어] 초기화 값이 있으면 반드시 심볼이 먼저 등록되어 있어야 함
    g_last_symbol->add_initializer(g_operands);  // [한국어] 심볼에 초기화 값 목록 설정
  }
  init_directive_state();  // [한국어] 지시어 파서 상태 초기화 (다음 선언 준비)
}

/*
 * [한국어]
 * ptx_recognizer::set_variable_type - 현재 공간/타입 지시어로 type_info 객체 생성
 *
 * g_space_spec과 g_scalar_type_spec이 모두 유효한지 검증(parse_assert)하고,
 * symbol_table::add_type을 호출하여 type_info를 생성한 후 g_var_type에 저장한다.
 * 이후 add_identifier에서 변수를 심볼 테이블에 등록할 때 이 type_info가 사용된다.
 *
 * 검증: g_space_spec != undefined_space, g_scalar_type_spec != -1
 *
 * 호출 체인:
 *   ptx.y의 type_qualifier 규칙 → [이 함수] → symbol_table::add_type → g_var_type
 */
void ptx_recognizer::set_variable_type() {
  PTX_PARSE_DPRINTF("set_variable_type space_spec=%s scalar_type_spec=%s",
                    g_ptx_token_decode[g_space_spec.get_type()].c_str(),
                    g_ptx_token_decode[g_scalar_type_spec].c_str());
  // [한국어] 디버그 모드에서 설정 중인 타입 이름과 공간 이름 출력
  parse_assert(g_space_spec != undefined_space,
               "variable has no space specification");  // [한국어] 메모리 공간 미지정 오류 검사
  parse_assert(
      g_scalar_type_spec != -1,
      "variable has no type information");  // need to extend for structs?
  // [한국어] 스칼라 타입 미지정 오류 검사 (구조체 확장은 TODO 상태)
  g_var_type = g_current_symbol_table->add_type(
      g_space_spec, g_scalar_type_spec, g_vector_spec, g_alignment_spec,
      g_extern_spec);
  // [한국어] 공간, 스칼라 타입, 벡터 지시어, 정렬, extern 여부로 type_info 객체 생성하여 g_var_type에 저장
  //          add_identifier에서 이 type_info를 사용하여 변수 심볼을 생성함
}

/*
 * [한국어]
 * ptx_recognizer::check_for_duplicates - 식별자 중복 선언 여부 확인
 *
 * @param identifier: 확인할 식별자 이름
 * @return: true면 현재 스코프에 이미 선언된 중복 식별자
 *
 * 현재 심볼 테이블에서 identifier를 조회하여 이미 있으면 true를 반환한다.
 * add_identifier에서 중복 선언 경고를 출력하거나 재사용 결정에 사용됨.
 *
 * 호출 체인:
 *   add_identifier → [이 함수] → symbol_table::lookup
 */
bool ptx_recognizer::check_for_duplicates(const char *identifier) {
  const symbol *s = g_current_symbol_table->lookup(identifier);  // [한국어] 현재 스코프 심볼 테이블에서 이름 조회
  return (s != NULL);  // [한국어] 조회 결과가 있으면(NULL이 아니면) 중복 선언
}

// Returns padding that needs to be inserted ahead of address to make it aligned
// to min(size, maxalign)
/*
 * @param address the address in bytes
 * @param size the size of the memory to be allocated in bytes
 * @param maximum alignment in bytes. i.e. if size is too big then align to this
 * instead
 */
/*
 * [한국어]
 * pad_address - 주소 정렬을 위한 패딩 바이트 수를 계산하는 유틸리티 함수
 *
 * @param address: 현재 할당 주소 (바이트 단위)
 * @param size: 할당할 메모리 크기 (바이트)
 * @param maxalign: 최대 정렬 바이트 (이보다 큰 정렬은 적용하지 않음)
 * @return: address 앞에 삽입해야 할 패딩 바이트 수 (0 이상)
 *
 * 배경: 메모리 정렬(alignment)은 하드웨어가 데이터를 효율적으로 읽기 위해 필요하다.
 * 예를 들어 4바이트 정수는 4의 배수 주소에 있어야 한다.
 * PTX 변수를 shared/global/local 메모리에 배치할 때 이 함수로 앞에 삽입할 빈 공간을 계산한다.
 *
 * 정렬 규칙: min(size, maxalign)으로 정렬. 단, size가 2의 거듭제곱일 때만 size로 정렬;
 * size가 2의 거듭제곱이 아니거나 size >= maxalign이면 maxalign으로 정렬.
 * 수식: alignto ? ((alignto - (address % alignto)) % alignto) : 0
 *   address % alignto: 현재 주소의 정렬 여분
 *   alignto - ...: 다음 정렬 경계까지의 거리
 *   외부 % alignto: address가 이미 정렬된 경우(여분=0) 0을 반환하기 위한 처리
 *
 * 호출 체인:
 *   add_identifier (shared/const/global/local 공간 할당) → [이 함수]
 */
int pad_address(new_addr_type address, unsigned size, unsigned maxalign) {
  assert(size >= 0);      // [한국어] 크기는 반드시 0 이상이어야 함
  assert(maxalign > 0);   // [한국어] 최대 정렬 값은 0보다 커야 함 (0 정렬은 의미 없음)
  int alignto = maxalign; // [한국어] 기본 정렬 단위를 maxalign으로 설정
  if (size < maxalign && (size & (size - 1)) == 0) {  // size is a power of 2
    // [한국어] size가 maxalign보다 작고 2의 거듭제곱이면 size로 정렬
    // (size & (size-1)) == 0 은 size가 2의 거듭제곱인지 확인하는 비트 연산
    // 예: size=4 → 0b100 & 0b011 = 0 → 2의 거듭제곱
    alignto = size;  // [한국어] 더 작은 size로 정렬 (불필요한 패딩 방지)
  }
  return alignto ? ((alignto - (address % alignto)) % alignto) : 0;
  // [한국어] 반환값 계산:
  //   alignto == 0이면 0 반환 (정렬 불필요 경우, 실제로는 assert로 방지됨)
  //   address % alignto: 현재 주소를 alignto로 나눈 나머지 (정렬 여분)
  //   alignto - ...: 다음 alignto 배수 주소까지 몇 바이트가 필요한지
  //   외부 % alignto: 나머지가 0이면(이미 정렬됨) 결과도 0이 되도록
  //   예: address=5, alignto=4 → 5%4=1, 4-1=3, 3%4=3 → 3바이트 패딩 필요
  //   예: address=8, alignto=4 → 8%4=0, 4-0=4, 4%4=0 → 패딩 불필요
}

/*
 * [한국어]
 * ptx_recognizer::add_identifier - 파싱된 식별자를 심볼 테이블에 등록하고 메모리 공간 할당
 *
 * @param identifier: PTX 소스의 식별자 이름 (%r0, arr, myConst 등)
 * @param array_dim: 배열 원소 수 (크기 있는 배열), 또는 0
 * @param array_ident: ARRAY_IDENTIFIER(크기 지정 배열), ARRAY_IDENTIFIER_NO_DIM(크기 미지정),
 *                     NON_ARRAY_IDENTIFIER(스칼라)
 *
 * 이 함수는 PTX 변수 선언의 핵심 처리기이다. 아래 과정을 수행한다:
 * 1) 배열이면 g_size *= array_dim으로 전체 크기 계산
 * 2) 반환값 캐싱: 함수 이름 전에 먼저 파싱되는 반환값 변수는 캐시에 저장하고 즉시 반환
 * 3) 중복 선언 검사: 이미 있으면 기존 심볼 재사용하고 경고 출력
 * 4) 배열 타입 처리: ARRAY_IDENTIFIER이면 배열 타입으로 변환
 * 5) 메모리 공간별 할당:
 *    - reg_space: 레지스터 번호(regnum) 할당, 아키텍처 레지스터 번호(arch_regnum) 추출
 *    - shared_space: 공유 메모리 공간에 128바이트 정렬 할당
 *    - sstarr_space: sstarr 공간 할당 (shared 유사)
 *    - const_space: 전역 메모리 주소에 상수로 할당; g_constants에 등록
 *    - global_space: 전역 메모리 할당; g_globals에 등록
 *    - local_space: 함수 밖이면 로컬 공간, 함수 안이면 스택 프레임 공간 할당
 *    - param_space_local: 로컬 스택 파라미터 할당; 스택 프레임 크기 갱신
 *    - param_space_kernel: 커널 파라미터; 별도 할당 없음 (호스트가 전달)
 * 6) param_space_kernel이면 function_info::add_param_name_type_size로 파라미터 정보 등록
 *
 * 호출 체인:
 *   ptx.y의 identifier_list 규칙 → [이 함수] → symbol_table::add_variable
 *     → (공간에 따라) set_regno, set_address, alloc_shared/global/local
 */
void ptx_recognizer::add_identifier(const char *identifier, int array_dim,
                                    unsigned array_ident) {
  if (array_ident == ARRAY_IDENTIFIER) {  // [한국어] 크기가 명시된 배열이면
    g_size *= array_dim;  // [한국어] 원소 수 × 원소 크기 = 전체 배열 크기 (바이트)
  }
  if (g_func_decl && (g_func_info == NULL)) {  // [한국어] 함수 선언 중이고 함수 이름이 아직 파싱되지 않았으면
    // return variable decl...
    // [한국어] 반환값 변수는 함수 이름보다 먼저 파싱될 수 있음; 캐시에 저장 후 나중에 처리
    assert(g_add_identifier_cached__identifier == NULL);  // [한국어] 반환값 캐시는 하나만 허용
    g_add_identifier_cached__identifier = strdup(identifier);  // [한국어] 식별자 이름을 캐시에 복사 (strdup: 힙 할당)
    g_add_identifier_cached__array_dim = array_dim;    // [한국어] 배열 차원 정보 캐시
    g_add_identifier_cached__array_ident = array_ident; // [한국어] 배열 종류 정보 캐시
    return;  // [한국어] add_function_name에서 처리될 것이므로 지금은 즉시 반환
  }
  PTX_PARSE_DPRINTF("add_identifier \"%s\" (%u)", identifier, g_ident_add_uid);
  // [한국어] 디버그 모드에서 처리 중인 식별자 이름과 처리 순서 번호 출력
  g_ident_add_uid++;  // [한국어] 식별자 처리 순서 카운터 증가 (디버그/추적용)
  type_info *type = g_var_type;  // [한국어] 현재 set_variable_type으로 설정된 타입 정보 참조
  type_info_key ti = type->get_key();  // [한국어] 타입 키 추출 (메모리 공간, 스칼라 타입 등 조회용)
  int basic_type;      // [한국어] 기본 타입 분류 (정수/부동소수점/비트 등)
  int regnum;          // [한국어] reg_space 변수에 할당할 레지스터 번호
  size_t num_bits;     // [한국어] 변수 크기 (비트 단위)
  unsigned addr_pad;   // [한국어] 정렬을 위해 앞에 추가할 패딩 바이트 수
  new_addr_type addr;  // [한국어] 현재 할당 위치 (메모리 공간의 다음 빈 주소)
  ti.type_decode(num_bits, basic_type);  // [한국어] 타입 키에서 비트 크기와 기본 타입 분류를 추출

  bool duplicates = check_for_duplicates(identifier);  // [한국어] 현재 스코프에 같은 이름이 이미 있는지 확인
  if (duplicates) {  // [한국어] 중복 선언이 있으면
    symbol *s = g_current_symbol_table->lookup(identifier);  // [한국어] 기존 심볼 조회
    g_last_symbol = s;  // [한국어] g_last_symbol을 기존 심볼로 설정 (후속 처리에서 재사용)
    if (g_func_decl) return;  // [한국어] 함수 선언 컨텍스트에서는 경고 없이 재사용 허용
    std::string msg = std::string(identifier) + " was declared previous at " +
                      s->decl_location() + " skipping new declaration";  // [한국어] 경고 메시지 조합
    printf("GPGPU-Sim PTX: Warning %s\n", msg.c_str());  // [한국어] 중복 선언 경고 출력 (오류가 아님)
    return;  // [한국어] 기존 심볼을 유지하고 새 선언 무시
  }

  assert(g_var_type != NULL);  // [한국어] set_variable_type이 반드시 먼저 호출되어 타입이 설정되어야 함
  switch (array_ident) {  // [한국어] 배열 여부에 따라 타입 조정
    case ARRAY_IDENTIFIER:  // [한국어] 크기 명시 배열
      type = g_current_symbol_table->get_array_type(type, array_dim);  // [한국어] 배열 타입 객체 생성 (원소 타입 + 원소 수)
      num_bits = array_dim * num_bits;  // [한국어] 전체 배열 크기를 비트 단위로 계산
      break;
    case ARRAY_IDENTIFIER_NO_DIM:  // [한국어] 크기 미지정 배열 (초기화 값에서 나중에 크기 결정)
      type = g_current_symbol_table->get_array_type(type, (unsigned)-1);  // [한국어] -1은 크기 미정 표시
      num_bits = 0;  // [한국어] 크기 0으로 설정 (나중에 초기화 값으로 결정됨)
      break;
    default:  // [한국어] NON_ARRAY_IDENTIFIER: 스칼라 변수, 타입 변경 없음
      break;
  }
  g_last_symbol = g_current_symbol_table->add_variable(
      identifier, type, num_bits / 8, gpgpu_ctx->g_filename,
      ptx_get_lineno(scanner));
  // [한국어] 심볼 테이블에 변수 등록; 크기를 바이트 단위로 변환(num_bits/8)하여 저장
  //          g_last_symbol은 이후 add_variables, add_function_arg 등에서 참조됨
  switch (ti.get_memory_space().get_type()) {  // [한국어] 메모리 공간 종류에 따른 추가 처리
    case reg_space: {  // [한국어] 레지스터 공간: 번호 할당
      regnum = g_current_symbol_table->next_reg_num();  // [한국어] 다음 레지스터 번호 할당 (0부터 순차 증가)
      int arch_regnum = -1;  // [한국어] 아키텍처 레지스터 번호 (PTX 식별자에서 숫자 부분 추출; -1은 미설정)
      for (int d = 0; d < strlen(identifier); d++) {  // [한국어] 식별자에서 첫 숫자를 찾아 아키텍처 레지스터 번호로 사용
        if (isdigit(identifier[d])) {  // [한국어] 숫자 문자를 만나면
          sscanf(identifier + d, "%d", &arch_regnum);  // [한국어] 그 위치부터 정수를 파싱하여 arch_regnum에 저장
          break;  // [한국어] 첫 숫자만 처리 후 반복 종료
        }
      }
      if (strcmp(identifier, "%sp") == 0) {  // [한국어] %sp는 스택 포인터 레지스터; 특별히 arch_regnum=0으로 설정
        arch_regnum = 0;
      }
      g_last_symbol->set_regno(regnum, arch_regnum);  // [한국어] 심볼에 가상 레지스터 번호와 아키텍처 번호 설정
    } break;
    case shared_space:  // [한국어] 공유 메모리 (.shared): CTA 내 스레드들이 공유하는 고속 메모리
      printf("GPGPU-Sim PTX: allocating shared region for \"%s\" ", identifier);
      fflush(stdout);  // [한국어] 즉시 출력 보장 (파싱 중 버퍼링 없이 진행 상황 확인)
      assert((num_bits % 8) == 0);  // [한국어] 비트 크기가 바이트의 배수여야 함 (8비트 = 1바이트)
      addr = g_current_symbol_table->get_shared_next();  // [한국어] 현재 공유 메모리의 다음 빈 주소 조회
      addr_pad = pad_address(addr, num_bits / 8, 128);  // [한국어] 128바이트 정렬에 필요한 패딩 계산
      printf("from 0x%llx to 0x%llx (shared memory space)\n", addr + addr_pad,
             addr + addr_pad + num_bits / 8);
      fflush(stdout);
      g_last_symbol->set_address(addr + addr_pad);  // [한국어] 패딩 후 정렬된 주소를 심볼에 설정
      g_current_symbol_table->alloc_shared(num_bits / 8 + addr_pad);  // [한국어] 패딩 포함 전체 크기만큼 공유 메모리 공간 예약
      break;
    case sstarr_space:  // [한국어] sstarr 공간: shared 유사 특수 공간 (구체적 용도는 구현에 따라 다름)
      printf("GPGPU-Sim PTX: allocating sstarr region for \"%s\" ", identifier);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      addr = g_current_symbol_table->get_sstarr_next();  // [한국어] sstarr 공간의 다음 빈 주소
      addr_pad = pad_address(addr, num_bits / 8, 128);  // [한국어] 128바이트 정렬 패딩 계산
      printf("from 0x%llx to 0x%llx (sstarr memory space)\n", addr + addr_pad,
             addr + addr_pad + num_bits / 8);
      fflush(stdout);
      g_last_symbol->set_address(addr + addr_pad);
      g_current_symbol_table->alloc_sstarr(num_bits / 8 + addr_pad);  // [한국어] sstarr 공간 예약
      break;
    case const_space:  // [한국어] 상수 메모리 (.const): 읽기 전용, 모든 스레드 공유, L1 캐시됨
      if (array_ident == ARRAY_IDENTIFIER_NO_DIM) {  // [한국어] 크기 미지정 배열이면 크기를 알 수 없어 할당 보류
        printf(
            "GPGPU-Sim PTX: deferring allocation of constant region for \"%s\" "
            "(need size information)\n",
            identifier);
        // [한국어] 크기 정보가 필요하여 나중에 처리 예정 (초기화 값 분석 후)
      } else {  // [한국어] 크기가 명시되어 있으면 즉시 전역 메모리 영역에 할당
        printf("GPGPU-Sim PTX: allocating constant region for \"%s\" ",
               identifier);
        fflush(stdout);
        assert((num_bits % 8) == 0);
        addr = g_current_symbol_table->get_global_next();  // [한국어] 상수는 전역 메모리 영역에 배치
        addr_pad = pad_address(addr, num_bits / 8, 128);  // [한국어] 128바이트 정렬 패딩 계산
        printf("from 0x%llx to 0x%llx (global memory space) %u\n",
               addr + addr_pad, addr + addr_pad + num_bits / 8,
               g_const_alloc++);  // [한국어] g_const_alloc++로 상수 할당 인스턴스 번호 증가 출력
        fflush(stdout);
        g_last_symbol->set_address(addr + addr_pad);  // [한국어] 정렬된 주소를 심볼에 설정
        g_current_symbol_table->alloc_global(num_bits / 8 + addr_pad);  // [한국어] 전역 메모리 공간 예약
      }
      if (g_current_symbol_table == g_global_symbol_table) {  // [한국어] 전역 스코프의 상수이면
        gpgpu_ctx->func_sim->g_constants.insert(identifier);  // [한국어] 전역 상수 집합에 이름 등록
      }
      assert(g_current_symbol_table != NULL);
      g_sym_name_to_symbol_table[identifier] = g_current_symbol_table;  // [한국어] 이름 → 심볼테이블 역방향 맵 등록 (memcpy_symbol에서 사용)
      break;
    case global_space:  // [한국어] 전역 메모리 (.global): 모든 스레드가 공유하는 DRAM 영역
      printf("GPGPU-Sim PTX: allocating global region for \"%s\" ", identifier);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      addr = g_current_symbol_table->get_global_next();  // [한국어] 전역 메모리의 다음 빈 주소
      addr_pad = pad_address(addr, num_bits / 8, 128);  // [한국어] 128바이트 정렬 패딩 계산
      printf("from 0x%llx to 0x%llx (global memory space)\n", addr + addr_pad,
             addr + addr_pad + num_bits / 8);
      fflush(stdout);
      g_last_symbol->set_address(addr + addr_pad);  // [한국어] 정렬된 전역 주소를 심볼에 설정
      g_current_symbol_table->alloc_global(num_bits / 8 + addr_pad);  // [한국어] 전역 메모리 공간 예약
      gpgpu_ctx->func_sim->g_globals.insert(identifier);  // [한국어] 전역 변수 집합에 이름 등록 (cudaMemcpyToSymbol 등에서 조회)
      assert(g_current_symbol_table != NULL);
      g_sym_name_to_symbol_table[identifier] = g_current_symbol_table;  // [한국어] 이름 → 심볼테이블 역방향 맵 등록
      break;
    case local_space:  // [한국어] 로컬 메모리 (.local): 스레드 고유 메모리 (느린 DRAM 기반)
      if (g_func_info == NULL) {  // [한국어] 함수 밖(전역 스코프)의 로컬 변수 선언 (드문 경우)
        printf("GPGPU-Sim PTX: allocating local region for \"%s\" ",
               identifier);
        fflush(stdout);
        assert((num_bits % 8) == 0);
        addr = g_current_symbol_table->get_local_next();  // [한국어] 로컬 메모리의 다음 빈 주소
        addr_pad = pad_address(addr, num_bits / 8, 128);  // [한국어] 128바이트 정렬 패딩 계산
        printf("from 0x%llx to 0x%llx (local memory space)\n", addr + addr_pad,
               addr + addr_pad + num_bits / 8);
        fflush(stdout);
        g_last_symbol->set_address(addr + addr_pad);
        g_current_symbol_table->alloc_local(num_bits / 8 + addr_pad);  // [한국어] 로컬 메모리 공간 예약
      } else {  // [한국어] 함수 내부의 .local: 스택 프레임(call frame)에 할당
        printf(
            "GPGPU-Sim PTX: allocating stack frame region for .local \"%s\" ",
            identifier);
        fflush(stdout);
        assert((num_bits % 8) == 0);
        addr = g_current_symbol_table->get_local_next();  // [한국어] 현재 스택 프레임의 다음 빈 위치
        addr_pad = pad_address(addr, num_bits / 8, 128);  // [한국어] 정렬 패딩 계산
        printf("from 0x%llx to 0x%llx\n", addr + addr_pad,
               addr + addr_pad + num_bits / 8);
        fflush(stdout);
        g_last_symbol->set_address(addr + addr_pad);
        g_current_symbol_table->alloc_local(num_bits / 8 + addr_pad);  // [한국어] 스택 프레임 공간 예약
        g_func_info->set_framesize(g_current_symbol_table->get_local_next());  // [한국어] 함수의 스택 프레임 크기 갱신 (로컬 할당마다 업데이트)
      }
      break;
    case tex_space:  // [한국어] 텍스처 공간 (.tex): 텍스처 메모리 선언; 시뮬레이터에서는 정보만 기록
      printf("GPGPU-Sim PTX: encountered texture directive %s.\n", identifier);
      // [한국어] 텍스처 변수는 별도 할당 없이 이름만 기록; 텍스처 접근은 tex 명령어에서 처리
      break;
    case param_space_local:  // [한국어] 로컬 파라미터 (.param 함수 내부): 스택 프레임에 파라미터 공간 할당
      printf(
          "GPGPU-Sim PTX: allocating stack frame region for .param \"%s\" from "
          "0x%llx to 0x%llx\n",
          identifier, g_current_symbol_table->get_local_next(),
          g_current_symbol_table->get_local_next() + num_bits / 8);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      g_last_symbol->set_address(g_current_symbol_table->get_local_next());  // [한국어] 패딩 없이 현재 로컬 위치에 파라미터 배치
      g_current_symbol_table->alloc_local(num_bits / 8);  // [한국어] 파라미터 크기만큼 로컬 공간 예약
      g_func_info->set_framesize(g_current_symbol_table->get_local_next());  // [한국어] 함수 스택 프레임 크기 갱신
      break;
    case param_space_kernel:  // [한국어] 커널 파라미터 (.param 커널 입력): 호스트에서 전달; 별도 메모리 할당 없음
      break;  // [한국어] 커널 파라미터는 호스트가 메모리를 관리하므로 디바이스 측 할당 불필요
    default:  // [한국어] 알 수 없는 메모리 공간: 처리 불가
      abort();  // [한국어] 지원하지 않는 메모리 공간 종류; 즉시 종료 (개발 중 발견해야 할 버그)
      break;
  }

  assert(!ti.is_param_unclassified());  // [한국어] 파라미터 공간이 분류되지 않은 상태(param_space_unclassified)로 남아있으면 오류
  if (ti.is_param_kernel()) {  // [한국어] 커널 파라미터이면 function_info에 파라미터 메타데이터 등록
    bool is_ptr = (g_ptr_spec != undefined_space);  // [한국어] .ptr 지시어가 있으면 포인터 파라미터
    g_func_info->add_param_name_type_size(g_entry_func_param_index, identifier,
                                          ti.scalar_type(), num_bits, is_ptr,
                                          g_ptr_spec);
    // [한국어] 파라미터 인덱스, 이름, 스칼라 타입, 크기(비트), 포인터 여부, 포인터 대상 공간을 등록
    //          ptx_sim_init_thread에서 커널 파라미터를 스레드에 바인딩할 때 이 정보 사용
    g_entry_func_param_index++;  // [한국어] 다음 파라미터를 위해 인덱스 증가
  }
}

/*
 * [한국어]
 * ptx_recognizer::add_constptr - 상수 포인터 변수의 주소를 다른 상수 기준으로 재배치
 *
 * @param identifier1: 재배치할 상수 변수 이름 (주소가 변경될 변수)
 * @param identifier2: 기준 상수 변수 이름 (이 변수의 주소를 기준으로 사용)
 * @param offset: 기준 주소로부터의 바이트 오프셋
 *
 * PTX의 .constptr 지시어를 처리한다. identifier1의 주소를 identifier2의 주소 + offset으로 변경.
 * 이 지시어는 상수 변수들의 메모리 레이아웃을 재정의할 때 사용되며, CUDA 컴파일러가
 * 여러 상수 변수를 하나의 연속된 블록으로 정렬할 때 생성할 수 있다.
 *
 * 호출 체인:
 *   ptx.y의 constptr_directive 규칙 → [이 함수]
 */
void ptx_recognizer::add_constptr(const char *identifier1,
                                  const char *identifier2, int offset) {
  symbol *s1 = g_current_symbol_table->lookup(identifier1);  // [한국어] 재배치할 심볼 조회
  const symbol *s2 = g_current_symbol_table->lookup(identifier2);  // [한국어] 기준 심볼 조회
  parse_assert(s1 != NULL, "'from' constant identifier does not exist.");  // [한국어] s1이 없으면 파싱 오류
  parse_assert(s1 != NULL, "'to' constant identifier does not exist.");  // [한국어] 실제로는 s2 확인 의도였으나 s1으로 중복됨 (버그 가능성)

  unsigned addr = s2->get_address();  // [한국어] 기준 변수의 현재 주소 조회

  printf("GPGPU-Sim PTX: moving \"%s\" from 0x%llx to 0x%x (%s+%d)\n",
         identifier1, s1->get_address(), addr + offset, identifier2, offset);
  // [한국어] 주소 변경 내역 출력: "이전 주소" → "기준 변수+오프셋"

  s1->set_address(addr + offset);  // [한국어] identifier1의 주소를 identifier2의 주소 + offset으로 변경
}

/*
 * [한국어]
 * ptx_recognizer::add_function_arg - 마지막 심볼(g_last_symbol)을 함수 인자로 등록
 *
 * 현재 파싱된 마지막 변수(g_last_symbol)를 g_func_info의 인자(arg) 목록에 추가하고,
 * 크기(g_size)와 정렬(alignment) 정보를 add_config_param으로 기록한다.
 * g_func_info가 NULL이면 (함수 외부에서 호출 시) 아무것도 하지 않는다.
 * alignment는 .align 지시어가 있으면 그 값, 없으면 변수 크기(g_size)를 사용한다.
 *
 * 호출 체인:
 *   ptx.y의 func_arg 규칙 → [이 함수] → function_info::add_arg, add_config_param
 */
void ptx_recognizer::add_function_arg() {
  assert(g_size > 0);  // [한국어] 인자 크기가 반드시 양수여야 함 (add_scalar_type_spec에서 설정)
  if (g_func_info) {  // [한국어] 함수 컨텍스트에서만 처리 (NULL이면 전역/비함수 컨텍스트)
    PTX_PARSE_DPRINTF("add_function_arg \"%s\"", g_last_symbol->name().c_str());
    // [한국어] 디버그 모드에서 추가되는 함수 인자 이름 출력
    g_func_info->add_arg(g_last_symbol);  // [한국어] 함수의 인자 목록에 심볼 추가
    unsigned alignment = (g_alignment_spec == -1) ? g_size : g_alignment_spec;
    // [한국어] .align 지시어가 있으면 그 값, 없으면 변수 크기를 정렬 기준으로 사용
    assert(alignment == 1 || alignment == 2 || alignment == 4 ||
           alignment == 8 || alignment == 16);  // known valid alignment values
    // [한국어] 정렬 값이 1/2/4/8/16 중 하나여야 함 (GPU 하드웨어가 지원하는 정렬)
    g_func_info->add_config_param(g_size, alignment);  // [한국어] 파라미터 크기와 정렬 정보를 function_info에 기록
  }
}

/*
 * [한국어]
 * ptx_recognizer::add_extern_spec - .extern 지시어 처리; g_extern_spec 설정
 *
 * .extern 지시어는 이 변수/함수가 현재 모듈이 아닌 다른 모듈에 정의되어 있음을 의미한다.
 * set_variable_type에서 extern 정보를 type_info에 포함시키기 위해 g_extern_spec = 1로 설정.
 *
 * 호출 체인:
 *   ptx.y의 extern_spec 규칙 → [이 함수]
 */
void ptx_recognizer::add_extern_spec() {
  PTX_PARSE_DPRINTF("add_extern_spec");  // [한국어] 디버그 모드에서 extern 지시어 처리 로그 출력
  g_extern_spec = 1;  // [한국어] extern 플래그 설정; set_variable_type에서 type_info 생성 시 사용됨
}

/*
 * [한국어]
 * ptx_recognizer::add_alignment_spec - .align 정렬 지시어 값 설정
 *
 * @param spec: 정렬 바이트 수 (1/2/4/8/16 등)
 *
 * 같은 변수 선언에 .align이 두 번 오면 parse_assert 실패.
 * 설정된 값은 add_function_arg에서 파라미터 정렬로, 메모리 할당 시에는 pad_address 기준으로 사용된다.
 *
 * 호출 체인:
 *   ptx.y의 align_spec 규칙 → [이 함수]
 */
void ptx_recognizer::add_alignment_spec(int spec) {
  PTX_PARSE_DPRINTF("add_alignment_spec");  // [한국어] 디버그 모드에서 정렬 지시어 처리 로그 출력
  parse_assert(
      g_alignment_spec == -1,
      "multiple .align specifiers per variable declaration not allowed.");
  // [한국어] 한 선언에 .align이 두 번 이상 오면 파싱 오류
  g_alignment_spec = spec;  // [한국어] 정렬 값 저장; add_function_arg에서 파라미터 정렬로 사용
}

/*
 * [한국어]
 * ptx_recognizer::add_ptr_spec - .ptr 포인터 대상 공간 지시어 설정
 *
 * @param spec: 포인터가 가리키는 메모리 공간 (global_space, local_space, shared_space만 허용)
 *
 * PTX ".param .u64 .ptr .global p"에서 .global에 해당하는 부분을 처리한다.
 * 중복 .ptr 지시어나 허용되지 않는 공간은 parse_assert 실패.
 *
 * 호출 체인:
 *   ptx.y의 ptr_spec 규칙 → [이 함수]
 */
void ptx_recognizer::add_ptr_spec(enum _memory_space_t spec) {
  PTX_PARSE_DPRINTF("add_ptr_spec \"%s\"", g_ptx_token_decode[spec].c_str());
  // [한국어] 디버그 모드에서 포인터 대상 공간 이름 출력
  parse_assert(g_ptr_spec == undefined_space,
               "multiple ptr space specifiers not allowed.");  // [한국어] 중복 .ptr 지시어 금지
  parse_assert(
      spec == global_space or spec == local_space or spec == shared_space,
      "invalid space for ptr directive.");  // [한국어] 포인터는 global/local/shared만 가능 (reg/const 등 불가)
  g_ptr_spec = spec;  // [한국어] 포인터 대상 공간 저장; add_identifier에서 is_ptr 판단에 사용
}

/*
 * [한국어]
 * ptx_recognizer::add_space_spec - 메모리 공간 지시어를 파싱 상태에 설정
 *
 * @param spec: 파싱된 메모리 공간 enum 값 (.global, .shared, .local, .param 등)
 * @param value: const_space에서 뱅크 번호 (그 외는 0)
 *
 * .param의 경우 컨텍스트에 따라 실제 공간을 결정한다:
 *   - g_func_decl && g_entry_point==1 → param_space_kernel (커널 진입점 파라미터)
 *   - g_func_decl && 그 외 → param_space_local (함수 호출 파라미터)
 *   - 함수 밖 → param_space_unclassified (현재 거의 사용 안 됨)
 * const_space인 경우 뱅크 번호(value)를 set_bank로 설정한다.
 *
 * 호출 체인:
 *   ptx.y의 space_spec 규칙 → [이 함수]
 */
void ptx_recognizer::add_space_spec(enum _memory_space_t spec, int value) {
  PTX_PARSE_DPRINTF("add_space_spec \"%s\"", g_ptx_token_decode[spec].c_str());
  // [한국어] 디버그 모드에서 설정 중인 공간 이름 출력
  parse_assert(g_space_spec == undefined_space,
               "multiple space specifiers not allowed.");  // [한국어] 한 선언에 공간 지시어는 하나만 허용
  if (spec == param_space_unclassified) {  // [한국어] .param 지시어: 컨텍스트에 따라 실제 공간 분류
    if (g_func_decl) {  // [한국어] 함수 선언 컨텍스트이면
      if (g_entry_point == 1)  // [한국어] 커널 엔트리 포인트(.entry)이면
        g_space_spec = param_space_kernel;  // [한국어] 커널 파라미터 (호스트→디바이스 전달)
      else  // [한국어] 일반 device 함수(.func)이면
        g_space_spec = param_space_local;  // [한국어] 로컬 파라미터 (스택 기반 함수 인자)
    } else  // [한국어] 함수 밖(전역 컨텍스트)이면 분류 보류
      g_space_spec = param_space_unclassified;  // [한국어] 나중에 컨텍스트 확정 후 처리
  } else {  // [한국어] .param 외의 공간 지시어: 그대로 설정
    g_space_spec = spec;  // [한국어] 파싱된 공간을 그대로 저장
    if (g_space_spec == const_space) g_space_spec.set_bank((unsigned)value);  // [한국어] 상수 메모리는 뱅크 번호도 함께 설정
  }
}

/*
 * [한국어]
 * ptx_recognizer::add_vector_spec - .v2/.v4 벡터 지시어 값 설정
 *
 * @param spec: 벡터 원소 수 (2=.v2, 4=.v4 등)
 *
 * 중복 벡터 지시어가 오면 parse_assert 실패.
 * set_variable_type에서 type_info 생성 시 벡터 차원으로 사용된다.
 *
 * 호출 체인:
 *   ptx.y의 vector_spec 규칙 → [이 함수]
 */
void ptx_recognizer::add_vector_spec(int spec) {
  PTX_PARSE_DPRINTF("add_vector_spec");  // [한국어] 디버그 모드에서 벡터 지시어 처리 로그 출력
  parse_assert(g_vector_spec == -1, "multiple vector specifiers not allowed.");  // [한국어] 중복 벡터 지시어 금지
  g_vector_spec = spec;  // [한국어] 벡터 원소 수 저장 (2 또는 4; set_variable_type에서 사용)
}

/*
 * [한국어]
 * ptx_recognizer::add_scalar_type_spec - 스칼라 타입 지시어를 파싱 상태에 추가
 *
 * @param type_spec: 타입 enum 값 (B8_TYPE, S16_TYPE, U32_TYPE, F32_TYPE 등)
 *
 * 타입에 따라 g_size를 바이트 단위로 설정하고 g_scalar_type 리스트에 추가한다.
 * g_scalar_type_spec을 마지막 타입으로 갱신한다.
 * cvt/set/slct/tex/mma/dp4a/vmin/vmax 외 명령어에서 두 번 이상 타입이 오면 parse_assert 실패.
 * (이 명령어들은 소스/목적지 타입이 달라 두 개의 타입 지시어를 가진다)
 *
 * 타입별 크기:
 *   B8/S8/U8 → 1바이트, B16/S16/U16/F16 → 2바이트
 *   B32/S32/U32/F32 → 4바이트, B64/BB64/S64/U64/F64/FF64 → 8바이트
 *   BB128 → 16바이트 (WMMA Tensor Core 행렬용)
 *
 * 호출 체인:
 *   ptx.y의 type_spec 규칙 → [이 함수]
 */
void ptx_recognizer::add_scalar_type_spec(int type_spec) {
  // save size of parameter
  // [한국어] 파라미터/변수 크기를 타입에 맞게 설정
  switch (type_spec) {
    case B8_TYPE:   // [한국어] 8비트 비트형
    case S8_TYPE:   // [한국어] 8비트 부호 있는 정수
    case U8_TYPE:   // [한국어] 8비트 부호 없는 정수
      g_size = 1;  // [한국어] 1바이트 = 8비트
      break;
    case B16_TYPE:  // [한국어] 16비트 비트형
    case S16_TYPE:  // [한국어] 16비트 부호 있는 정수
    case U16_TYPE:  // [한국어] 16비트 부호 없는 정수
    case F16_TYPE:  // [한국어] 16비트 반정밀도 부동소수점 (half float)
      g_size = 2;  // [한국어] 2바이트 = 16비트
      break;
    case B32_TYPE:  // [한국어] 32비트 비트형
    case S32_TYPE:  // [한국어] 32비트 부호 있는 정수
    case U32_TYPE:  // [한국어] 32비트 부호 없는 정수
    case F32_TYPE:  // [한국어] 32비트 단정밀도 부동소수점 (float; GPU에서 가장 일반적인 타입)
      g_size = 4;  // [한국어] 4바이트 = 32비트
      break;
    case B64_TYPE:  // [한국어] 64비트 비트형
    case BB64_TYPE: // [한국어] 64비트 비트형 (alternate)
    case S64_TYPE:  // [한국어] 64비트 부호 있는 정수
    case U64_TYPE:  // [한국어] 64비트 부호 없는 정수
    case F64_TYPE:  // [한국어] 64비트 배정밀도 부동소수점 (double)
    case FF64_TYPE: // [한국어] 64비트 부동소수점 (alternate)
      g_size = 8;  // [한국어] 8바이트 = 64비트
      break;
    case BB128_TYPE: // [한국어] 128비트 비트형 (Tensor Core WMMA 연산의 행렬 원소용)
      g_size = 16; // [한국어] 16바이트 = 128비트
      break;
  }
  PTX_PARSE_DPRINTF("add_scalar_type_spec \"%s\"",
                    g_ptx_token_decode[type_spec].c_str());
  // [한국어] 디버그 모드에서 처리 중인 타입 이름 출력
  g_scalar_type.push_back(type_spec);  // [한국어] 타입 목록에 추가; add_instruction에서 ptx_instruction에 전달됨
  if (g_scalar_type.size() > 1) {  // [한국어] 두 번째 이상 타입 지시어이면 명령어 제한 검사
    parse_assert(
        (g_opcode == -1) || (g_opcode == CVT_OP) || (g_opcode == SET_OP) ||
            (g_opcode == SLCT_OP) || (g_opcode == TEX_OP) ||
            (g_opcode == MMA_OP) || (g_opcode == DP4A_OP) ||
            (g_opcode == VMIN_OP) || (g_opcode == VMAX_OP),
        "only cvt, set, slct, tex, vmin, vmax and dp4a can have more than one "
        "type specifier.");
    // [한국어] 두 타입 지시어를 허용하는 명령어:
    //   cvt: 타입 변환 (src 타입 → dst 타입)
    //   set/setp: 비교 결과 타입과 비교 대상 타입이 다를 수 있음
    //   slct: select 명령어; 선택 기준 타입과 출력 타입이 다를 수 있음
    //   tex: 텍스처 좌표 타입과 결과 타입이 다를 수 있음
    //   mma: Tensor Core 행렬 곱셈; 입력/출력/누산 타입이 다를 수 있음
    //   dp4a, vmin, vmax: 특수 정수 연산
  }
  g_scalar_type_spec = type_spec;  // [한국어] 마지막으로 파싱된 타입을 g_scalar_type_spec에 저장 (set_variable_type에서 사용)
}

/*
 * [한국어]
 * ptx_recognizer::add_label - PTX 레이블을 심볼 테이블에 등록하고 g_label에 설정
 *
 * @param identifier: 레이블 이름 (분기 목적지; 예: "loop_start:", "end_kernel:")
 *
 * 레이블은 분기 명령어(bra, brx 등)의 대상이 되는 코드 위치 표시자다.
 * 이미 심볼 테이블에 있으면(forward reference: 분기 명령어가 레이블보다 먼저 파싱됨) 재사용하고,
 * 없으면 새 심볼로 등록한다.
 *
 * 호출 체인:
 *   ptx.y의 label 규칙 → [이 함수] → symbol_table::add_variable (없을 때)
 */
void ptx_recognizer::add_label(const char *identifier) {
  PTX_PARSE_DPRINTF("add_label");  // [한국어] 디버그 모드에서 레이블 처리 로그 출력
  symbol *s = g_current_symbol_table->lookup(identifier);  // [한국어] 심볼 테이블에서 이미 등록되어 있는지 조회
  if (s != NULL) {  // [한국어] 이미 있으면 (forward reference: 분기 명령어가 레이블보다 먼저 파싱됨)
    g_label = s;   // [한국어] 기존 심볼 재사용 (이미 forward 선언된 것을 레이블로 확정)
  } else {  // [한국어] 처음 등장하는 레이블이면 새로 등록
    g_label = g_current_symbol_table->add_variable(
        identifier, NULL, 0, gpgpu_ctx->g_filename, ptx_get_lineno(scanner));
    // [한국어] 타입=NULL, 크기=0으로 레이블 심볼 생성 (레이블은 타입/크기 없이 위치만 나타냄)
  }
}

/*
 * [한국어]
 * ptx_recognizer::add_opcode - 명령어 오피코드를 g_opcode에 설정
 *
 * @param opcode: ptx_ir.h의 opcode enum 값 (ADD_OP, LD_OP, BRA_OP 등)
 *
 * Yacc에서 명령어 토큰을 읽을 때 즉시 호출되는 단순 설정 함수.
 *
 * 호출 체인:
 *   ptx.y의 opcode 규칙 → [이 함수]
 */
void ptx_recognizer::add_opcode(int opcode) { g_opcode = opcode; }  // [한국어] 현재 명령어 오피코드 저장; add_instruction에서 사용됨

/*
 * [한국어]
 * ptx_recognizer::add_pred - 현재 명령어의 프레디케이트 레지스터 설정
 *
 * @param identifier: 프레디케이트 레지스터 이름 (%p0, %p1 등)
 * @param neg: 부정 여부 (0=정상 "@%p", 1=부정 "@!%p")
 * @param predModifier: 프레디케이트 수식어 (-1=없음)
 *
 * 심볼 테이블에서 identifier를 조회하고, 없으면 parse_error 발생.
 * 조회된 심볼과 부정/수식어 플래그를 g_pred, g_neg_pred, g_pred_mod에 저장한다.
 *
 * 호출 체인:
 *   ptx.y의 pred_spec 규칙 → [이 함수]
 */
void ptx_recognizer::add_pred(const char *identifier, int neg,
                              int predModifier) {
  PTX_PARSE_DPRINTF("add_pred");  // [한국어] 디버그 모드에서 프레디케이트 처리 로그 출력
  const symbol *s = g_current_symbol_table->lookup(identifier);  // [한국어] 프레디케이트 레지스터를 심볼 테이블에서 조회
  if (s == NULL) {  // [한국어] 선언되지 않은 프레디케이트 레지스터는 오류
    std::string msg =
        std::string("predicate \"") + identifier + "\" has no declaration.";  // [한국어] 오류 메시지 조합
    parse_error(msg.c_str());  // [한국어] 파싱 오류 처리 (abort로 종료)
  }
  g_pred = s;              // [한국어] 프레디케이트 레지스터 심볼 저장
  g_neg_pred = neg;        // [한국어] 부정 플래그 저장 (0=정상, 1=부정)
  g_pred_mod = predModifier; // [한국어] 프레디케이트 수식어 저장
}

/*
 * [한국어]
 * ptx_recognizer::add_option - 명령어 수식어 옵션을 g_options에 추가
 *
 * @param option: 옵션 enum 값 (.ftz, .rn, .lo, .hi, .ca 등)
 *
 * 단순 push_back; add_instruction에서 ptx_instruction 생성 시 전달됨.
 *
 * 호출 체인:
 *   ptx.y의 instruction_option 규칙 → [이 함수]
 */
void ptx_recognizer::add_option(int option) {
  PTX_PARSE_DPRINTF("add_option");  // [한국어] 디버그 모드에서 옵션 추가 로그 출력
  g_options.push_back(option);  // [한국어] 일반 명령어 옵션 목록에 추가
}

/*
 * [한국어]
 * ptx_recognizer::add_wmma_option - WMMA 전용 옵션을 g_wmma_options에 추가
 *
 * @param option: WMMA 옵션 enum 값 (.row, .col, .m16n16k16 등)
 *
 * Tensor Core WMMA 명령어(wmma.load, wmma.store, wmma.mma)에만 사용되는 옵션.
 * 일반 옵션(g_options)과 분리하여 관리한다.
 *
 * 호출 체인:
 *   ptx.y의 wmma_option 규칙 → [이 함수]
 */
void ptx_recognizer::add_wmma_option(int option) {
  PTX_PARSE_DPRINTF("add_option");  // [한국어] 디버그 출력 (add_option과 동일 메시지; 의도적인 것으로 보임)
  g_wmma_options.push_back(option);  // [한국어] WMMA 전용 옵션 목록에 추가
}

/*
 * [한국어]
 * ptx_recognizer::add_double_operand - 두 심볼을 조합한 더블 오퍼랜드 추가
 *
 * @param d1, d2: 조합할 두 심볼의 이름
 *
 * 두 레지스터/변수를 조합하는 오퍼랜드를 생성한다. 예: s[$ofs1+$r0], g[$ofs1+=$r0].
 * 두 심볼 모두 심볼 테이블에 있어야 하며, 없으면 parse_assert 실패.
 * TODO: 두 개의 목적지에 저장하는 용도로도 사용할지 미결 상태.
 *
 * 호출 체인:
 *   ptx.y의 double_operand 규칙 → [이 함수]
 */
void ptx_recognizer::add_double_operand(const char *d1, const char *d2) {
  // operands that access two variables.
  // eg. s[$ofs1+$r0], g[$ofs1+=$r0]
  // TODO: Not sure if I'm going to use this for storing to two destinations or
  // not.
  // [한국어] 두 변수를 조합한 오퍼랜드 (예: 공유 메모리 인덱싱 s[오프셋+레지스터])

  PTX_PARSE_DPRINTF("add_double_operand");  // [한국어] 디버그 모드에서 더블 오퍼랜드 처리 로그 출력
  const symbol *s1 = g_current_symbol_table->lookup(d1);  // [한국어] 첫 번째 심볼 조회
  const symbol *s2 = g_current_symbol_table->lookup(d2);  // [한국어] 두 번째 심볼 조회
  parse_assert(s1 != NULL && s2 != NULL, "component(s) missing declarations.");  // [한국어] 둘 다 선언되어 있어야 함
  g_operands.push_back(operand_info(s1, s2, gpgpu_ctx));  // [한국어] 두 심볼을 하나의 오퍼랜드로 묶어 목록에 추가
}

/*
 * [한국어]
 * ptx_recognizer::add_1vector_operand - 단일 원소 벡터 오퍼랜드 추가 (tex.1d용)
 *
 * @param d1: 벡터의 유일한 원소 심볼 이름
 *
 * tex.1d 명령어처럼 {%v1} 형식의 단일 원소 벡터 오퍼랜드를 처리한다.
 * 나머지 3개 슬롯은 NULL로 채워진다.
 *
 * 호출 체인:
 *   ptx.y의 1vector_operand 규칙 → [이 함수]
 */
void ptx_recognizer::add_1vector_operand(const char *d1) {
  // handles the single element vector operand ({%v1}) found in tex.1d
  // instructions
  // [한국어] tex.1d에서 단일 원소 벡터 오퍼랜드 ({%v1}) 처리
  PTX_PARSE_DPRINTF("add_1vector_operand");  // [한국어] 디버그 모드에서 로그 출력
  const symbol *s1 = g_current_symbol_table->lookup(d1);  // [한국어] 벡터 원소 심볼 조회
  parse_assert(s1 != NULL, "component(s) missing declarations.");  // [한국어] 선언되어 있어야 함
  g_operands.push_back(operand_info(s1, NULL, NULL, NULL, gpgpu_ctx));  // [한국어] 1개 원소만 있는 벡터 오퍼랜드 생성 (나머지 3개 NULL)
}

/*
 * [한국어]
 * ptx_recognizer::add_2vector_operand - .v2 벡터 오퍼랜드 추가
 *
 * @param d1, d2: 2원소 벡터의 각 원소 심볼 이름
 *
 * {%v0, %v1} 형식의 2원소 벡터 오퍼랜드를 처리한다.
 *
 * 호출 체인:
 *   ptx.y의 2vector_operand 규칙 → [이 함수]
 */
void ptx_recognizer::add_2vector_operand(const char *d1, const char *d2) {
  PTX_PARSE_DPRINTF("add_2vector_operand");  // [한국어] 디버그 모드에서 로그 출력
  const symbol *s1 = g_current_symbol_table->lookup(d1);  // [한국어] 첫 번째 원소 심볼 조회
  const symbol *s2 = g_current_symbol_table->lookup(d2);  // [한국어] 두 번째 원소 심볼 조회
  parse_assert(s1 != NULL && s2 != NULL,
               "v2 component(s) missing declarations.");  // [한국어] 두 원소 모두 선언되어 있어야 함
  g_operands.push_back(operand_info(s1, s2, NULL, NULL, gpgpu_ctx));  // [한국어] 2원소 벡터 오퍼랜드 생성 (나머지 2개 NULL)
}

/*
 * [한국어]
 * ptx_recognizer::add_3vector_operand - .v3 벡터 오퍼랜드 추가
 *
 * @param d1, d2, d3: 3원소 벡터의 각 원소 심볼 이름
 *
 * {%v0, %v1, %v2} 형식의 3원소 벡터 오퍼랜드를 처리한다 (비교적 드물게 사용).
 *
 * 호출 체인:
 *   ptx.y의 3vector_operand 규칙 → [이 함수]
 */
void ptx_recognizer::add_3vector_operand(const char *d1, const char *d2,
                                         const char *d3) {
  PTX_PARSE_DPRINTF("add_3vector_operand");  // [한국어] 디버그 모드에서 로그 출력
  const symbol *s1 = g_current_symbol_table->lookup(d1);  // [한국어] 첫 번째 원소 심볼 조회
  const symbol *s2 = g_current_symbol_table->lookup(d2);  // [한국어] 두 번째 원소 심볼 조회
  const symbol *s3 = g_current_symbol_table->lookup(d3);  // [한국어] 세 번째 원소 심볼 조회
  parse_assert(s1 != NULL && s2 != NULL && s3 != NULL,
               "v3 component(s) missing declarations.");  // [한국어] 세 원소 모두 선언되어 있어야 함
  g_operands.push_back(operand_info(s1, s2, s3, NULL, gpgpu_ctx));  // [한국어] 3원소 벡터 오퍼랜드 생성 (마지막 슬롯 NULL)
}

/*
 * [한국어]
 * ptx_recognizer::add_4vector_operand - .v4 벡터 오퍼랜드 추가
 *
 * @param d1, d2, d3, d4: 4원소 벡터의 각 원소 심볼 이름
 *
 * {%v0, %v1, %v2, %v3} 형식의 4원소 벡터 오퍼랜드를 처리한다.
 * GPU에서 가장 일반적인 벡터 크기로, ld.v4, st.v4 등에서 사용된다.
 * "_" (와일드카드)로 선언된 원소는 NULL로 처리하여 사용하지 않음을 표시한다.
 *
 * 호출 체인:
 *   ptx.y의 4vector_operand 규칙 → [이 함수]
 */
void ptx_recognizer::add_4vector_operand(const char *d1, const char *d2,
                                         const char *d3, const char *d4) {
  PTX_PARSE_DPRINTF("add_4vector_operand");  // [한국어] 디버그 모드에서 로그 출력
  const symbol *s1 = g_current_symbol_table->lookup(d1);  // [한국어] 첫 번째 원소 심볼 조회
  const symbol *s2 = g_current_symbol_table->lookup(d2);  // [한국어] 두 번째 원소 심볼 조회
  const symbol *s3 = g_current_symbol_table->lookup(d3);  // [한국어] 세 번째 원소 심볼 조회
  const symbol *s4 = g_current_symbol_table->lookup(d4);  // [한국어] 네 번째 원소 심볼 조회
  parse_assert(s1 != NULL && s2 != NULL && s3 != NULL && s4 != NULL,
               "v4 component(s) missing declarations.");  // [한국어] 네 원소 모두 심볼 테이블에 있어야 함
  const symbol *null_op = g_current_symbol_table->lookup("_");  // [한국어] "_" 와일드카드 심볼 조회 (사용하지 않는 원소 표시)
  if (s2 == null_op) s2 = NULL;  // [한국어] "_"이면 NULL로 교체 (이 원소는 사용 안 함)
  if (s3 == null_op) s3 = NULL;  // [한국어] "_"이면 NULL로 교체
  if (s4 == null_op) s4 = NULL;  // [한국어] "_"이면 NULL로 교체
  g_operands.push_back(operand_info(s1, s2, s3, s4, gpgpu_ctx));  // [한국어] 4원소 벡터 오퍼랜드 생성
}
/*
 * [한국어]
 * ptx_recognizer::add_8vector_operand - 8원소 벡터 오퍼랜드 추가
 *
 * @param d1..d8: 벡터의 각 원소 레지스터 이름 (8개). "_"이면 NULL 원소 (미사용).
 *
 * WMMA(Warp Matrix Multiply Accumulate) 명령어에서 사용되는 8원소 벡터 오퍼랜드를 처리한다.
 * 8개의 심볼을 모두 조회하고, 모두 선언되어 있어야 한다.
 * "_"에 해당하는 심볼은 NULL로 교체한다.
 *
 * 호출 체인:
 *   ptx.y의 v8_operand 규칙 → [이 함수]
 */
void ptx_recognizer::add_8vector_operand(const char *d1, const char *d2,
                                         const char *d3, const char *d4,
                                         const char *d5, const char *d6,
                                         const char *d7, const char *d8) {
  PTX_PARSE_DPRINTF("add_8vector_operand");  // [한국어] 디버그 모드에서 8원소 벡터 오퍼랜드 처리 로그 출력
  const symbol *s1 = g_current_symbol_table->lookup(d1);  // [한국어] 첫 번째 원소 심볼 조회
  const symbol *s2 = g_current_symbol_table->lookup(d2);  // [한국어] 두 번째 원소 심볼 조회
  const symbol *s3 = g_current_symbol_table->lookup(d3);  // [한국어] 세 번째 원소 심볼 조회
  const symbol *s4 = g_current_symbol_table->lookup(d4);  // [한국어] 네 번째 원소 심볼 조회
  const symbol *s5 = g_current_symbol_table->lookup(d5);  // [한국어] 다섯 번째 원소 심볼 조회
  const symbol *s6 = g_current_symbol_table->lookup(d6);  // [한국어] 여섯 번째 원소 심볼 조회
  const symbol *s7 = g_current_symbol_table->lookup(d7);  // [한국어] 일곱 번째 원소 심볼 조회
  const symbol *s8 = g_current_symbol_table->lookup(d8);  // [한국어] 여덟 번째 원소 심볼 조회
  parse_assert(s1 != NULL && s2 != NULL && s3 != NULL && s4 != NULL &&
                   s5 != NULL && s6 != NULL && s7 != NULL && s8 != NULL,
               "v4 component(s) missing declarations.");
  // [한국어] 8개 원소 모두 선언되어 있어야 함; 아니면 파싱 오류
  const symbol *null_op = g_current_symbol_table->lookup("_");  // [한국어] "_" = null 오퍼랜드 심볼 (unused slot 표시)
  if (s2 == null_op) s2 = NULL;  // [한국어] "_"면 NULL로 교체
  if (s3 == null_op) s3 = NULL;  // [한국어] "_"면 NULL로 교체
  if (s4 == null_op) s4 = NULL;  // [한국어] "_"면 NULL로 교체
  if (s5 == null_op) s5 = NULL;  // [한국어] "_"면 NULL로 교체
  if (s6 == null_op) s6 = NULL;  // [한국어] "_"면 NULL로 교체
  if (s7 == null_op) s7 = NULL;  // [한국어] "_"면 NULL로 교체
  if (s8 == null_op) s8 = NULL;  // [한국어] "_"면 NULL로 교체
  g_operands.push_back(operand_info(s1, s2, s3, s4, s5, s6, s7, s8, gpgpu_ctx));  // [한국어] 8원소 벡터 오퍼랜드 생성하여 목록에 추가
}

/*
 * [한국어]
 * ptx_recognizer::add_builtin_operand - 빌트인(내장) 오퍼랜드 추가
 *
 * @param builtin: 빌트인 식별자 (예: TIDX, TIDY, NTID, CTAID 등의 열거값)
 * @param dim_modifier: 차원 수정자 (x=0, y=1, z=2)
 *
 * PTX의 빌트인 특수 레지스터(예: %tid.x, %ntid.y, %ctaid.z)를 오퍼랜드로 추가한다.
 * 빌트인 레지스터는 워프의 스레드 ID, 블록 ID 등 GPU 실행 컨텍스트 정보를 제공한다.
 *
 * 호출 체인:
 *   ptx.y의 builtin_operand 규칙 → [이 함수]
 */
void ptx_recognizer::add_builtin_operand(int builtin, int dim_modifier) {
  PTX_PARSE_DPRINTF("add_builtin_operand");  // [한국어] 디버그 모드에서 빌트인 오퍼랜드 처리 로그 출력
  g_operands.push_back(operand_info(builtin, dim_modifier, gpgpu_ctx));  // [한국어] 빌트인 ID와 차원 수정자로 operand_info 생성하여 추가
}

/*
 * [한국어]
 * ptx_recognizer::add_memory_operand - 마지막 오퍼랜드를 메모리 오퍼랜드로 변환
 *
 * ld/st 명령어에서 []로 감싸인 메모리 주소 오퍼랜드를 처리한다.
 * 이미 추가된 주소 오퍼랜드에 메모리 접근 표시를 추가하는 역할이다.
 * 예: ld.global.f32 %f0, [%rd0]; 에서 [%rd0] 처리.
 *
 * 호출 체인:
 *   ptx.y의 memory_operand 규칙 → [이 함수]
 */
void ptx_recognizer::add_memory_operand() {
  PTX_PARSE_DPRINTF("add_memory_operand");  // [한국어] 디버그 모드에서 메모리 오퍼랜드 변환 로그 출력
  assert(!g_operands.empty());  // [한국어] 변환할 오퍼랜드가 반드시 있어야 함
  g_operands.back().make_memory_operand();  // [한국어] 마지막 오퍼랜드를 메모리 접근 형태로 변환 (주소 → 메모리 참조)
}

/*
 * [한국어]
 * ptx_recognizer::change_memory_addr_space - 메모리 오퍼랜드의 주소 공간 설정
 *
 * @param identifier: 메모리 주소 공간 식별자 문자열
 *   "g" → global_space (전역 메모리)
 *   "s" → shared_space (공유 메모리)
 *   "c*" → const_space (상수 메모리; 'c'로 시작)
 *   "l*" → local_space (로컬 메모리; 'l'로 시작)
 *
 * PTX의 메모리 한정자(e.g. .global, .shared, .const, .local)가 없이
 * 직접 주소 공간 식별자로 지정되는 경우를 처리한다.
 * const 메모리는 심볼 테이블에서 해당 상수의 오프셋도 추가로 설정한다.
 * 인식하지 못한 메모리 타입이면 parse_assert로 오류 발생.
 *
 * TODO: 다른 메모리 공간(texture, surface, param 등) 추가 필요
 *
 * 호출 체인:
 *   ptx.y의 memory_addr_space 규칙 → [이 함수]
 */
/*TODO: add other memory locations*/
void ptx_recognizer::change_memory_addr_space(const char *identifier) {
  /*0 = N/A, not reading from memory
   *1 = global memory
   *2 = shared memory
   *3 = const memory segment
   *4 = local memory segment
   */

  bool recognizedType = false;  // [한국어] 인식된 메모리 타입 여부 추적 플래그

  PTX_PARSE_DPRINTF("change_memory_addr_space");  // [한국어] 디버그 모드에서 주소 공간 변경 로그 출력
  assert(!g_operands.empty());  // [한국어] 변경할 오퍼랜드가 반드시 있어야 함
  if (!strcmp(identifier, "g")) {  // [한국어] "g" = global 메모리
    g_operands.back().set_addr_space(global_space);  // [한국어] 마지막 오퍼랜드의 주소 공간을 전역으로 설정
    recognizedType = true;  // [한국어] 타입 인식 완료
  }
  if (!strcmp(identifier, "s")) {  // [한국어] "s" = shared 메모리
    g_operands.back().set_addr_space(shared_space);  // [한국어] 마지막 오퍼랜드의 주소 공간을 공유 메모리로 설정
    recognizedType = true;  // [한국어] 타입 인식 완료
  }
  // For constants, check if the first character is 'c'
  // [한국어] 상수 메모리 뱅크는 c0, c1, ... 형식 (첫 글자가 'c')
  char c[2];
  strncpy(c, identifier, 1);  // [한국어] 첫 문자만 추출
  c[1] = '\0';  // [한국어] null 종료
  if (!strcmp(c, "c")) {  // [한국어] 첫 문자가 'c'이면 상수 메모리
    g_operands.back().set_addr_space(const_space);  // [한국어] 주소 공간을 상수 메모리로 설정
    parse_assert(g_current_symbol_table->lookup(identifier) != NULL,
                 "Constant was not defined.");
    // [한국어] 상수 뱅크 심볼이 반드시 선언되어 있어야 함
    g_operands.back().set_const_mem_offset(
        g_current_symbol_table->lookup(identifier)->get_address());
    // [한국어] 상수 심볼의 기준 주소를 오퍼랜드의 const 오프셋으로 설정
    recognizedType = true;  // [한국어] 타입 인식 완료
  }
  // For local memory, check if the first character is 'l'
  // [한국어] 로컬 메모리는 'l'로 시작
  char l[2];
  strncpy(l, identifier, 1);  // [한국어] 첫 문자만 추출
  l[1] = '\0';  // [한국어] null 종료
  if (!strcmp(l, "l")) {  // [한국어] 첫 문자가 'l'이면 로컬 메모리
    g_operands.back().set_addr_space(local_space);  // [한국어] 주소 공간을 로컬 메모리로 설정
    // parse_assert(g_current_symbol_table->lookup(identifier) != NULL, "Local
    // memory segment was not defined.");
    // g_operands.back().set_const_mem_offset(g_current_symbol_table->lookup(identifier)->get_address());
    // [한국어] 로컬 메모리 오프셋 설정은 미구현 (주석 처리됨)
    recognizedType = true;  // [한국어] 타입 인식 완료
  }

  parse_assert(recognizedType, "Error: unrecognized memory type.");  // [한국어] 위 중 하나도 해당 없으면 파싱 오류
}

/*
 * [한국어]
 * ptx_recognizer::change_operand_lohi - 마지막 오퍼랜드의 하위/상위 비트 선택 설정
 *
 * @param lohi: 0=전체 오퍼랜드 읽기, 1=하위(lo) 비트, 2=상위(hi) 비트
 *
 * PTX 명령어에서 레지스터의 하위 16비트나 상위 16비트만 선택적으로 읽는 경우에 사용.
 * 예: cvt 명령어에서 64비트 → 32비트 변환 시 lo/hi 절반만 읽는 경우.
 *
 * 호출 체인:
 *   ptx.y의 operand_lohi_modifier 규칙 → [이 함수]
 */
void ptx_recognizer::change_operand_lohi(int lohi) {
  /*0 = N/A, read entire operand
   *1 = lo, reading from lowest bits
   *2 = hi, reading from highest bits
   */
  /* [한국어] lohi 값 의미:
   * 0 = N/A: 오퍼랜드 전체 읽기 (기본)
   * 1 = lo: 가장 낮은 비트(하위 절반) 읽기
   * 2 = hi: 가장 높은 비트(상위 절반) 읽기 */

  PTX_PARSE_DPRINTF("change_operand_lohi");  // [한국어] 디버그 모드에서 lo/hi 변경 로그 출력
  assert(!g_operands.empty());  // [한국어] 변경할 오퍼랜드가 반드시 있어야 함

  g_operands.back().set_operand_lohi(lohi);  // [한국어] 마지막 오퍼랜드의 lo/hi 선택 플래그 설정
}

/*
 * [한국어]
 * ptx_recognizer::set_immediate_operand_type - 마지막 오퍼랜드를 즉치 주소 타입으로 변경
 *
 * 주소 값이 리터럴로 직접 제공되는 경우(즉치 주소)를 나타내는 플래그를 설정한다.
 * 예: ld.global.u32 %r0, [0x1234];에서 [0x1234] 부분 처리.
 *
 * 호출 체인:
 *   ptx.y의 immediate_addr 규칙 → [이 함수]
 */
void ptx_recognizer::set_immediate_operand_type() {
  PTX_PARSE_DPRINTF("set_immediate_operand_type");  // [한국어] 디버그 모드에서 즉치 주소 타입 설정 로그 출력
  assert(!g_operands.empty());  // [한국어] 변경할 오퍼랜드가 반드시 있어야 함
  g_operands.back().set_immediate_addr();  // [한국어] 마지막 오퍼랜드를 즉치 주소로 표시
}

/*
 * [한국어]
 * ptx_recognizer::change_double_operand_type - 더블 오퍼랜드의 연산 종류 설정
 *
 * @param operand_type: 더블 오퍼랜드 조합 방식
 *   -3 = reg / reg (set 명령어에서 두 목적지에 같은 값 저장)
 *   -2 = reg | reg (cvt 명령어에서 두 레지스터 분리)
 *   -1 = reg | reg (set 명령어에서 두 레지스터 분리)
 *    0 = N/A, 기본값
 *    1 = reg + reg (덧셈 기반 주소 계산)
 *    2 = reg += reg (증감 주소 계산)
 *    3 = reg += immediate (즉치 증감)
 *
 * set/setp 명령어는 -1 또는 -2를 -1로 처리하고, 그 외는 -2로 처리한다.
 * -3은 set 또는 mad 명령어에서만 허용된다.
 *
 * 호출 체인:
 *   ptx.y의 double_operand_type 규칙 → [이 함수]
 */
void ptx_recognizer::change_double_operand_type(int operand_type) {
  /*
   *-3 = reg / reg (set instruction, but both get same value)
   *-2 = reg | reg (cvt instruction)
   *-1 = reg | reg (set instruction)
   *0 = N/A, default
   *1 = reg + reg
   *2 = reg += reg
   *3 = reg += immediate
   */

  PTX_PARSE_DPRINTF("change_double_operand_type");  // [한국어] 디버그 모드에서 더블 오퍼랜드 타입 변경 로그 출력
  assert(!g_operands.empty());  // [한국어] 변경할 오퍼랜드가 반드시 있어야 함

  // For double destination operands, ensure valid instruction
  // [한국어] 더블 목적지 오퍼랜드는 특정 명령어에서만 허용됨
  if (operand_type == -1 || operand_type == -2) {  // [한국어] set 또는 cvt 스타일 분리 오퍼랜드
    if ((g_opcode == SET_OP) || (g_opcode == SETP_OP))  // [한국어] set/setp 명령어면
      g_operands.back().set_double_operand_type(-1);  // [한국어] set 스타일(-1)로 처리
    else  // [한국어] 그 외(cvt 등)면
      g_operands.back().set_double_operand_type(-2);  // [한국어] cvt 스타일(-2)로 처리
  } else if (operand_type == -3) {  // [한국어] 두 목적지에 같은 값을 저장하는 경우
    if (g_opcode == SET_OP || g_opcode == MAD_OP)  // [한국어] set 또는 mad 명령어에서만 허용
      g_operands.back().set_double_operand_type(operand_type);  // [한국어] -3 그대로 설정
    else
      parse_assert(0, "Error: Unsupported use of double destination operand.");  // [한국어] 지원하지 않는 명령어에서 사용 시 오류
  } else {  // [한국어] 일반 더블 오퍼랜드 타입 (0=기본, 1=reg+reg 등)
    g_operands.back().set_double_operand_type(operand_type);  // [한국어] 지정된 타입 그대로 설정
  }
}

/*
 * [한국어]
 * ptx_recognizer::change_operand_neg - 마지막 오퍼랜드에 부정(negation) 플래그 설정
 *
 * 피연산자에 '-' 부호가 붙는 경우를 처리한다. 예: mul.f32 %r0, %r1, -3.14;에서 -3.14 처리.
 *
 * 호출 체인:
 *   ptx.y의 neg_operand 규칙 → [이 함수]
 */
void ptx_recognizer::change_operand_neg() {
  PTX_PARSE_DPRINTF("change_operand_neg");  // [한국어] 디버그 모드에서 오퍼랜드 부정 설정 로그 출력
  assert(!g_operands.empty());  // [한국어] 부정할 오퍼랜드가 반드시 있어야 함

  g_operands.back().set_operand_neg();  // [한국어] 마지막 오퍼랜드에 부정 플래그 설정
}

/*
 * [한국어]
 * ptx_recognizer::add_literal_int - 정수 리터럴 상수 오퍼랜드 추가
 *
 * @param value: 정수 리터럴 값 (예: 0, 1, -1, 0xFF 등)
 *
 * PTX 명령어에서 즉치 정수 값을 오퍼랜드로 사용하는 경우를 처리한다.
 * 예: add.s32 %r0, %r1, 42;에서 42 처리.
 *
 * 호출 체인:
 *   ptx.y의 int_literal 규칙 → [이 함수]
 */
void ptx_recognizer::add_literal_int(int value) {
  PTX_PARSE_DPRINTF("add_literal_int");  // [한국어] 디버그 모드에서 정수 리터럴 추가 로그 출력
  g_operands.push_back(operand_info(value, gpgpu_ctx));  // [한국어] 정수 값으로 operand_info 생성하여 오퍼랜드 목록에 추가
}

/*
 * [한국어]
 * ptx_recognizer::add_literal_float - 단정밀도 부동소수점 리터럴 오퍼랜드 추가
 *
 * @param value: float 리터럴 값 (예: 1.0, 3.14, 0f3F800000 형식의 16진수 등)
 *
 * 호출 체인:
 *   ptx.y의 float_literal 규칙 → [이 함수]
 */
void ptx_recognizer::add_literal_float(float value) {
  PTX_PARSE_DPRINTF("add_literal_float");  // [한국어] 디버그 모드에서 float 리터럴 추가 로그 출력
  g_operands.push_back(operand_info(value, gpgpu_ctx));  // [한국어] float 값으로 operand_info 생성하여 추가
}

/*
 * [한국어]
 * ptx_recognizer::add_literal_double - 배정밀도 부동소수점 리터럴 오퍼랜드 추가
 *
 * @param value: double 리터럴 값 (예: 0d3FF0000000000000 형식의 16진수 등)
 *
 * 호출 체인:
 *   ptx.y의 double_literal 규칙 → [이 함수]
 */
void ptx_recognizer::add_literal_double(double value) {
  PTX_PARSE_DPRINTF("add_literal_double");  // [한국어] 디버그 모드에서 double 리터럴 추가 로그 출력
  g_operands.push_back(operand_info(value, gpgpu_ctx));  // [한국어] double 값으로 operand_info 생성하여 추가
}

/*
 * [한국어]
 * ptx_recognizer::add_scalar_operand - 스칼라 레지스터/변수 오퍼랜드 추가
 *
 * @param identifier: 오퍼랜드 식별자 이름 (%r0, %f1, myVar 등)
 *
 * 심볼 테이블에서 identifier를 조회한다:
 * - 있으면: 그 심볼로 operand_info 생성하여 추가
 * - 없고 bra/callp 명령어면: forward reference(분기 대상이 아직 선언 전)로 간주하여 새 심볼 생성
 * - 없고 다른 명령어면: parse_error 발생
 *
 * 호출 체인:
 *   ptx.y의 scalar_operand 규칙 → [이 함수]
 */
void ptx_recognizer::add_scalar_operand(const char *identifier) {
  PTX_PARSE_DPRINTF("add_scalar_operand");  // [한국어] 디버그 모드에서 스칼라 오퍼랜드 처리 로그 출력
  const symbol *s = g_current_symbol_table->lookup(identifier);  // [한국어] 심볼 테이블에서 식별자 조회
  if (s == NULL) {  // [한국어] 심볼 테이블에 없으면
    if (g_opcode == BRA_OP || g_opcode == CALLP_OP) {  // [한국어] 분기 명령어의 대상은 forward reference 허용
      // forward branch target...
      // [한국어] 레이블이 분기 명령어보다 나중에 파싱되는 경우; 임시 심볼 생성 후 나중에 레이블로 연결됨
      s = g_current_symbol_table->add_variable(
          identifier, NULL, 0, gpgpu_ctx->g_filename, ptx_get_lineno(scanner));
      // [한국어] 타입=NULL, 크기=0으로 임시 심볼 생성 (나중에 add_label에서 확정됨)
    } else {  // [한국어] 분기 명령어가 아닌데 선언되지 않은 오퍼랜드이면 오류
      std::string msg =
          std::string("operand \"") + identifier + "\" has no declaration.";  // [한국어] 오류 메시지 조합
      parse_error(msg.c_str());  // [한국어] 파싱 오류 처리 (abort로 종료)
    }
  }
  g_operands.push_back(operand_info(s, gpgpu_ctx));  // [한국어] 심볼로 operand_info 생성하여 목록에 추가
}

/*
 * [한국어]
 * ptx_recognizer::add_neg_pred_operand - 부정 프레디케이트 오퍼랜드 추가
 *
 * @param identifier: 프레디케이트 레지스터 이름
 *
 * 부정 프레디케이트 오퍼랜드(setp.p, lop3 등의 명령어에서 사용되는 ~p 형식)를 처리한다.
 * 선언되지 않은 심볼이면 임시로 생성한다 (size=1로 생성하여 비트 타입 암시).
 * 생성된 operand_info에 set_neg_pred로 부정 표시 후 추가한다.
 *
 * 호출 체인:
 *   ptx.y의 neg_pred_operand 규칙 → [이 함수]
 */
void ptx_recognizer::add_neg_pred_operand(const char *identifier) {
  PTX_PARSE_DPRINTF("add_neg_pred_operand");  // [한국어] 디버그 모드에서 부정 프레디케이트 처리 로그 출력
  const symbol *s = g_current_symbol_table->lookup(identifier);  // [한국어] 프레디케이트 심볼 조회
  if (s == NULL) {  // [한국어] 없으면 임시 생성 (선언 없이 사용되는 경우)
    s = g_current_symbol_table->add_variable(
        identifier, NULL, 1, gpgpu_ctx->g_filename, ptx_get_lineno(scanner));
    // [한국어] size=1로 생성; 1바이트 비트 타입 암시
  }
  operand_info op(s, gpgpu_ctx);  // [한국어] 심볼로 operand_info 생성
  op.set_neg_pred();  // [한국어] 부정 프레디케이트 표시 (~p)
  g_operands.push_back(op);  // [한국어] 오퍼랜드 목록에 추가
}

/*
 * [한국어]
 * ptx_recognizer::add_address_operand - 심볼+오프셋 주소 오퍼랜드 추가
 *
 * @param identifier: 기준 심볼 이름 (변수/레지스터)
 * @param offset: 기준 심볼 주소로부터의 바이트 오프셋
 *
 * ld/st 명령어의 [%rd0+8] 형식 오퍼랜드를 처리한다.
 * 예: ld.global.f32 %f0, [%rd0+16];에서 [%rd0+16] 처리.
 *
 * 호출 체인:
 *   ptx.y의 address_operand 규칙 → [이 함수]
 */
void ptx_recognizer::add_address_operand(const char *identifier, int offset) {
  PTX_PARSE_DPRINTF("add_address_operand");  // [한국어] 디버그 모드에서 주소 오퍼랜드 처리 로그 출력
  const symbol *s = g_current_symbol_table->lookup(identifier);  // [한국어] 기준 심볼 조회
  if (s == NULL) {  // [한국어] 선언되지 않은 심볼로 주소를 만들려 하면 오류
    std::string msg =
        std::string("operand \"") + identifier + "\" has no declaration.";  // [한국어] 오류 메시지 조합
    parse_error(msg.c_str());  // [한국어] 파싱 오류 처리
  }
  g_operands.push_back(operand_info(s, offset, gpgpu_ctx));  // [한국어] 심볼과 오프셋으로 주소 오퍼랜드 생성
}

/*
 * [한국어]
 * ptx_recognizer::add_address_operand2 - 즉치 오프셋만으로 구성된 주소 오퍼랜드 추가
 *
 * @param offset: 즉치 주소 오프셋 값
 *
 * 심볼 없이 즉치 오프셋만으로 구성된 주소 오퍼랜드를 처리한다.
 * 예: ld.global.f32 %f0, [256];에서 [256] 처리.
 * unsigned로 캐스트하여 음수 오프셋도 큰 양수로 표현한다.
 *
 * 호출 체인:
 *   ptx.y의 immediate_address_operand 규칙 → [이 함수]
 */
void ptx_recognizer::add_address_operand2(int offset) {
  PTX_PARSE_DPRINTF("add_address_operand");  // [한국어] 디버그 모드에서 즉치 주소 오퍼랜드 처리 로그 출력 (동일 메시지 의도적)
  g_operands.push_back(operand_info((unsigned)offset, gpgpu_ctx));  // [한국어] 즉치 오프셋을 unsigned로 변환하여 주소 오퍼랜드 생성
}

/*
 * [한국어]
 * ptx_recognizer::add_array_initializer - 배열 초기화 값을 g_last_symbol에 설정
 *
 * g_operands에 축적된 초기화 값 목록을 g_last_symbol에 전달한다.
 * 예: .global .s32 arr[] = {1, 2, 3};에서 {1, 2, 3} 처리.
 *
 * 호출 체인:
 *   ptx.y의 array_initializer 규칙 → [이 함수] → symbol::add_initializer
 */
void ptx_recognizer::add_array_initializer() {
  g_last_symbol->add_initializer(g_operands);  // [한국어] 오퍼랜드 목록(초기화 값들)을 마지막 심볼에 설정
}

/*
 * [한국어]
 * ptx_recognizer::add_version_info - PTX 버전 정보를 전역 심볼 테이블에 설정
 *
 * @param ver: PTX 버전 번호 (예: 6.5, 7.0)
 * @param ext: 확장 플래그 비트마스크
 *
 * .version 지시어를 처리하여 전역 심볼 테이블에 PTX 버전을 기록한다.
 * 이 정보는 PTX 기능 검사(텍스처 모드, fp16 지원 여부 등)에 사용된다.
 *
 * 호출 체인:
 *   ptx.y의 version_directive 규칙 → [이 함수] → symbol_table::set_ptx_version
 */
void ptx_recognizer::add_version_info(float ver, unsigned ext) {
  g_global_symbol_table->set_ptx_version(ver, ext);  // [한국어] 전역 심볼 테이블에 PTX 버전과 확장 정보 저장
}

/*
 * [한국어]
 * ptx_recognizer::add_file - .file 지시어 처리; g_filename 추론 및 심볼 테이블 복귀
 *
 * @param num: 파일 번호 (디버그 정보에서 사용)
 * @param filename: 소스 파일 이름 ("xxx.cu" 또는 "xxx.ptx" 형식)
 *
 * g_filename이 아직 설정되지 않은 경우, .file 지시어에서 소스 파일 이름을 추론하여 설정한다.
 * 파일 이름 추론 로직:
 * 1) 전체 경로에서 마지막 '/' 이후 파일명 부분 추출
 * 2) '.' 앞의 기본 이름(stem) 추출
 * 3) ".cu" 확장자이면 ".ptx"로 변환하여 g_filename으로 설정
 * .file 처리 후 g_current_symbol_table을 전역으로 복귀한다.
 *
 * 호출 체인:
 *   ptx.y의 file_directive 규칙 → [이 함수]
 */
void ptx_recognizer::add_file(unsigned num, const char *filename) {
  if (gpgpu_ctx->g_filename == NULL) {  // [한국어] g_filename이 아직 설정되지 않은 경우에만 추론 시도
    char *b = strdup(filename);  // [한국어] 파일 이름을 힙에 복사 (strtok 등이 원본 수정하므로)
    char *l = b;   // [한국어] 파일명 시작 포인터 (경로 제거 후 갱신될 예정)
    char *n = b;   // [한국어] 순회 포인터
    while (*n != '\0') {  // [한국어] 문자열 끝까지 순회
      if (*n == '/') l = n + 1;  // [한국어] '/'를 만날 때마다 l을 다음 문자로 갱신 → 마지막 '/' 이후가 파일명
      n++;
    }

    char *p = strtok(l, ".");  // [한국어] '.'으로 분리하여 첫 부분(확장자 없는 기본 이름) 추출
    char buf[1024];
    snprintf(buf, 1024, "%s.ptx", p);  // [한국어] 기본 이름 + ".ptx" 조합

    char *q = strtok(NULL, ".");  // [한국어] '.' 이후 확장자 부분 추출
    if (q && !strcmp(q, "cu")) {  // [한국어] 확장자가 ".cu"이면 CUDA 소스 파일; ".ptx"로 변환 적용
      gpgpu_ctx->g_filename = strdup(buf);  // [한국어] "stem.ptx" 형태로 g_filename 설정
    }

    free(b);  // [한국어] strdup으로 할당한 복사본 해제
  }

  g_current_symbol_table = g_global_symbol_table;  // [한국어] .file 처리 후 전역 심볼 테이블로 복귀 (함수 스코프 밖으로)
}

/*
 * [한국어]
 * ptx_recognizer::reset_symtab - 현재 심볼 테이블을 전역으로 복귀하고 이전 테이블 반환
 *
 * @return: 전환 전의 g_current_symbol_table 포인터
 *
 * 현재 스코프의 심볼 테이블을 저장하고 전역 테이블로 전환한다.
 * CDP(CUDA Dynamic Parallelism)에서 스코프 전환 시 사용된다.
 * 반환된 포인터는 set_symtab으로 나중에 복원할 수 있다.
 *
 * 호출 체인:
 *   CDP 관련 ptx.y 규칙 → [이 함수]
 */
void *ptx_recognizer::reset_symtab() {
  void *result = g_current_symbol_table;  // [한국어] 현재 심볼 테이블 저장 (호출자에게 반환하기 위해)
  g_current_symbol_table = g_global_symbol_table;  // [한국어] 전역 심볼 테이블로 복귀
  return result;  // [한국어] 저장한 이전 심볼 테이블 반환 (void*로 반환; set_symtab에서 캐스트)
}

/*
 * [한국어]
 * ptx_recognizer::set_symtab - 지정된 심볼 테이블을 현재 스코프로 설정
 *
 * @param symtab: 설정할 심볼 테이블 포인터 (void*로 전달; symbol_table*로 캐스트)
 *
 * reset_symtab가 반환한 포인터를 다시 활성화하는 쌍(pair) 함수다.
 * CDP 명령어 그룹 처리에서 스코프를 명시적으로 전환할 때 사용.
 *
 * 호출 체인:
 *   CDP 관련 ptx.y 규칙 → [이 함수]
 */
void ptx_recognizer::set_symtab(void *symtab) {
  g_current_symbol_table = (symbol_table *)symtab;  // [한국어] void* → symbol_table*로 캐스트하여 현재 심볼 테이블로 설정
}

/*
 * [한국어]
 * ptx_recognizer::add_pragma - .pragma 지시어 무시 경고 출력
 *
 * @param str: pragma 내용 문자열
 *
 * GPGPU-Sim은 .pragma 지시어를 현재 구현하지 않았다.
 * 파싱은 하되 무시하며 경고 메시지를 출력한다.
 *
 * 호출 체인:
 *   ptx.y의 pragma_directive 규칙 → [이 함수]
 */
void ptx_recognizer::add_pragma(const char *str) {
  printf("GPGPU-Sim PTX: Warning -- ignoring pragma '%s'\n", str);  // [한국어] .pragma 지시어 무시 경고 출력 (현재 미구현)
}

/*
 * [한국어]
 * ptx_recognizer::version_header - .version 헤더 처리 더미 함수
 *
 * @param a: 버전 번호 (double; 사용되지 않음)
 *
 * 의도적 더미 함수. 실제 버전 정보는 add_version_info에서 처리됨.
 * Yacc 규칙에서 일관성 있는 호출 구조를 유지하기 위해 존재한다.
 */
void ptx_recognizer::version_header(double a) {}  // intentional dummy function
// [한국어] 의도적 빈 함수; 버전 처리는 add_version_info에서 수행됨

/*
 * [한국어]
 * ptx_recognizer::target_header - .target 지시어 (SM 버전 1개) 처리
 *
 * @param a: SM 버전 문자열 (예: "sm_80", "sm_70")
 *
 * 전역 심볼 테이블에 SM 버전을 설정한다.
 * 시뮬레이션 대상 GPU 아키텍처를 결정하는 중요한 지시어다.
 *
 * 호출 체인:
 *   ptx.y의 target_directive 규칙 → [이 함수] → symbol_table::set_sm_target
 */
void ptx_recognizer::target_header(char *a) {
  g_global_symbol_table->set_sm_target(a, NULL, NULL);  // [한국어] SM 버전만 설정, 확장 없음
}

/*
 * [한국어]
 * ptx_recognizer::target_header2 - .target 지시어 (SM 버전 + 확장 1개) 처리
 *
 * @param a: SM 버전 문자열
 * @param b: 첫 번째 확장 문자열 (예: "texmode_unified", "map_f64_to_f32")
 *
 * 호출 체인:
 *   ptx.y의 target_directive 규칙 → [이 함수] → symbol_table::set_sm_target
 */
void ptx_recognizer::target_header2(char *a, char *b) {
  g_global_symbol_table->set_sm_target(a, b, NULL);  // [한국어] SM 버전과 확장 1개 설정
}

/*
 * [한국어]
 * ptx_recognizer::target_header3 - .target 지시어 (SM 버전 + 확장 2개) 처리
 *
 * @param a: SM 버전 문자열
 * @param b: 첫 번째 확장 문자열
 * @param c: 두 번째 확장 문자열
 *
 * 호출 체인:
 *   ptx.y의 target_directive 규칙 → [이 함수] → symbol_table::set_sm_target
 */
void ptx_recognizer::target_header3(char *a, char *b, char *c) {
  g_global_symbol_table->set_sm_target(a, b, c);  // [한국어] SM 버전과 확장 2개 모두 설정
}

/*
 * [한국어]
 * ptx_recognizer::maxnt_id - .maxnctapersm/.maxntid 지시어; 최대 스레드 수 설정
 *
 * @param x, y, z: 각 차원의 최대 스레드 수
 *
 * 커널의 최대 활성 스레드 수를 function_info에 설정한다.
 * x * y * z로 총 스레드 수를 계산하여 set_maxnt_id로 전달한다.
 * 이 값은 GPGPU-Sim의 스케줄러가 CTA 점유율(occupancy) 계산 시 참조한다.
 *
 * 호출 체인:
 *   ptx.y의 maxnt_directive 규칙 → [이 함수] → function_info::set_maxnt_id
 */
void ptx_recognizer::maxnt_id(int x, int y, int z) {
  g_func_info->set_maxnt_id(x * y * z);  // [한국어] 3차원 스레드 블록 크기의 곱 = 총 스레드 수를 function_info에 설정
}

/*
 * [한국어]
 * ptx_recognizer::func_header - 함수 헤더 처리 더미 함수
 *
 * @param a: 함수 헤더 문자열 (사용되지 않음)
 *
 * 의도적 더미 함수. Yacc 규칙의 일관성을 위해 존재하며 실제 처리는 없다.
 */
void ptx_recognizer::func_header(const char *a) {}  // intentional dummy
                                                    // function
// [한국어] 의도적 빈 함수; 함수 헤더 처리는 다른 함수들이 담당

/*
 * [한국어]
 * ptx_recognizer::func_header_info - 문자열 인자를 받는 함수 헤더 정보 더미 함수
 * @param a: 문자열 인자 (사용되지 않음)
 */
void ptx_recognizer::func_header_info(const char *a) {
}  // intentional dummy function
// [한국어] 의도적 빈 함수

/*
 * [한국어]
 * ptx_recognizer::func_header_info_int - 문자열+정수 인자를 받는 함수 헤더 정보 더미 함수
 * @param a: 문자열 인자 (사용되지 않음)
 * @param b: 정수 인자 (사용되지 않음)
 */
void ptx_recognizer::func_header_info_int(const char *a, int b) {
}  // intentional dummy function
// [한국어] 의도적 빈 함수
