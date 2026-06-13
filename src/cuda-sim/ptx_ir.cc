// Copyright (c) 2009-2021, Tor M. Aamodt, Ali Bakhoda, Wilson W.L. Fung,
// George L. Yuan, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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

/*
 * [한국어 설명] PTX IR 자료구조 메서드 구현 (ptx_ir.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 기능 시뮬레이션(functional simulation) 계층에서
 * PTX(Parallel Thread eXecution) 중간 표현(IR)을 구성하는 핵심 클래스들의
 * 메서드를 구현한다. symbol, symbol_table, operand_info, ptx_instruction,
 * function_info, type_info_key 등 PTX IR 구성요소의 생성, 조회, 조립(assemble),
 * CFG(Control Flow Graph) 분석, 지배자(dominator) 분석, 재수렴(reconvergence)
 * 쌍 계산 등을 담당한다. PTX 어셈블러가 파서에서 전달한 명령어 목록을 받아
 * 실제 시뮬레이션에 사용할 수 있는 형태로 변환하는 마지막 준비 단계이다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CUDA/OpenCL 애플리케이션 → libcuda(인터셉트) → gpgpusim_entrypoint.cc(진입)
 *   → cuda-sim/ptx_parser.cc(PTX 파싱) → [이 파일: IR 구조 구성 및 CFG 분석]
 *   → cuda-sim/ptx_sim.cc(사이클별 기능 실행) → gpgpu-sim/shader.cc(타이밍 모델)
 *
 * PTX 파서(ptx.l/ptx.y)가 소스를 파싱하면서 ptx_instruction 객체들을 생성하고
 * function_info에 추가한다. 이 파일의 gpgpu_ptx_assemble() 함수가 호출되면
 * function_info::ptx_assemble()을 통해 PC 할당과 CFG 분석(do_pdom())이 수행된다.
 * 이후 타이밍 시뮬레이터(shader.cc의 워프 스케줄러)가 pc_to_instruction()으로
 * 명령어를 조회하고, SIMT 스택이 reconvergence 쌍을 사용해 워프 분기를 관리한다.
 *
 * 실행 컨텍스트: 호스트 CPU 유저스페이스 (시뮬레이션 초기화 단계, 싱글스레드)
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - ptx_ir.h: 이 파일이 구현하는 모든 클래스 선언 (symbol, symbol_table,
 *               operand_info, ptx_instruction, function_info, type_info_key)
 *   - ptx_parser.h / ptx.tab.h: PTX 파서 토큰 정의 및 yyscan_t 타입
 *   - opcodes.h: PTX opcode 상수 정의 (BRA_OP, RET_OP, EXIT_OP, BREAK_OP 등)
 *   - cuda-sim.h: ptx_thread_info, ptx_reg_t, g_debug_execution 등 실행 컨텍스트
 *   - libcuda/gpgpu_context.h: gpgpu_context (전역 UID 카운터, s_g_pc_to_insn 테이블)
 *   - abstract_hardware_model.h: warp_inst_t (ptx_instruction의 부모 클래스)
 *
 * 이 파일에 의존하는 모듈:
 *   - cuda-sim/ptx_sim.cc: ptx_instruction을 꺼내 기능 실행 (execute())
 *   - gpgpu-sim/shader.cc: pc_to_instruction()으로 명령어 조회, reconvergence 쌍 사용
 *   - libcuda/: register_ptx_function()을 통해 함수 등록 정보 수신
 *
 * 공유 자료구조:
 *   - gpgpu_context::s_g_pc_to_insn: PC → ptx_instruction* 전역 테이블
 *   - gpgpu_context::symbol_sm_next_uid, operand_info_sm_next_uid,
 *     function_info_sm_next_uid, g_num_ptx_inst_uid: 전역 UID 카운터들
 *
 * 데이터 흐름:
 *   PTX 소스 파싱 → ptx_instruction 객체 생성 → function_info에 추가
 *   → ptx_assemble()로 PC 할당 → do_pdom()으로 CFG/dominator/ipdom 계산
 *   → s_g_pc_to_insn 테이블 등록 → 타이밍 시뮬레이터가 조회
 *
 * === 주요 함수/구조체 요약 ===
 * pc_to_instruction()         : PC 값으로 ptx_instruction* 를 전역 테이블에서 조회
 * symbol::get_uid()           : 전역 카운터를 후위증가시켜 고유 UID 반환
 * symbol_table::add_function_decl(): 함수 선언 처리 — function_info + symbol_table 생성 및 등록
 * symbol_table::lookup()      : 심볼 이름으로 현재 스코프 → 부모 스코프 순차 탐색
 * function_info::do_pdom()    : CFG 분석 파이프라인 전체 실행 (BB 생성→연결→지배자→ipdom→pre_decode)
 * function_info::create_basic_blocks(): 리더 기반 기본 블록 식별 알고리즘
 * function_info::find_dominators()   : Muchnick Fig 7.14 반복 데이터플로우 알고리즘
 * function_info::find_ipostdominators(): Fig 7.15 즉각 후위지배자 — SIMT 재수렴점 결정에 사용
 * function_info::get_reconvergence_pairs(): BRA 명령어별 (분기 PC, ipdom PC) 쌍 배열 생성
 * ptx_instruction::ptx_instruction(): 대형 생성자 — opcode/pred/operand/옵션 파싱 및 저장
 * type_info_key::type_decode()       : PTX 타입 토큰 → (비트 크기, 기본 타입 범주) 변환
 * copy_arg_to_buffer() / copy_buffer_to_frame(): CALL 명령어 인자 복사 유틸리티
 */

#include "ptx_ir.h"          // [한국어] 이 파일이 구현하는 모든 PTX IR 클래스 선언 포함
#include "ptx_parser.h"      // [한국어] PTX 파서 인터페이스 (yyscan_t 등 파서 내부 타입)
typedef void *yyscan_t;      // [한국어] Flex/Bison 재진입 스캐너 핸들 타입 — void* 로 전방 선언
#include <assert.h>          // [한국어] assert() 매크로 — 불변 조건 위반 시 프로세스 중단
#include <stdio.h>           // [한국어] fprintf/printf/snprintf/fflush/popen/pclose 등 C I/O
#include <stdlib.h>          // [한국어] abort() — 복구 불가 오류 시 비정상 종료
#include <algorithm>         // [한국어] std::remove — 탭 문자 제거 (m_source 정제)
#include <list>              // [한국어] std::list — 명령어 목록, 오퍼랜드 목록 컨테이너
#include "assert.h"          // [한국어] GPGPU-Sim 자체 assert 래퍼 (메시지 확장 버전)
#include "opcodes.h"         // [한국어] PTX opcode 정수 상수 (BRA_OP, RET_OP, CALL_OP 등 100+개)
#include "ptx.tab.h"         // [한국어] Bison이 생성한 토큰 번호 정의 (SYNC_OPTION 등 옵션 상수)

#include "../../libcuda/gpgpu_context.h"  // [한국어] gpgpu_context 클래스 — 전역 UID 카운터,
                                          //         s_g_pc_to_insn PC→명령어 테이블 접근
#include "cuda-sim.h"        // [한국어] 기능 시뮬레이션 공용 선언 (ptx_thread_info, g_debug_execution 등)

#define STR_SIZE 1024        // [한국어] 문자열 버퍼 최대 크기 — to_string(), get_insn_str()에서 사용

/*
 * [한국어]
 * pc_to_instruction - PC 값으로 PTX 명령어 객체를 조회하는 전역 접근자
 *
 * @pc: 조회할 프로그램 카운터 값. function_info::ptx_assemble()에서 할당된
 *      시뮬레이터 내부 PC (실제 GPU PC가 아니라 s_g_pc_to_insn 배열 인덱스).
 * @return: 해당 PC에 등록된 ptx_instruction 포인터.
 *          pc가 테이블 범위를 벗어나면 NULL 반환.
 *
 * s_g_pc_to_insn는 ptx_assemble() 단계에서 채워지는 전역 벡터로,
 * 타이밍 시뮬레이터(shader.cc)가 워프 스케줄러를 통해 명령어를 fetch할 때
 * 이 함수를 호출하여 ptx_instruction*을 얻어 기능 시뮬레이션(cuda-sim)에 넘긴다.
 * 실행 컨텍스트: 사이클마다 호출 (싱글스레드 시뮬레이터 루프).
 *
 * 호출 체인:
 *   shader.cc(워프 스케줄러 fetch) → [pc_to_instruction] → 반환된 ptx_instruction*
 *   → ptx_sim.cc(execute)
 */
const ptx_instruction *gpgpu_context::pc_to_instruction(unsigned pc) {
  if (pc < s_g_pc_to_insn.size())   // [한국어] PC가 등록된 명령어 테이블 범위 내인지 확인
    return s_g_pc_to_insn[pc];       // [한국어] 테이블에서 해당 PC의 명령어 포인터 반환
  else
    return NULL;                     // [한국어] 범위 초과 — 아직 어셈블되지 않은 PC이거나 잘못된 PC
}

/*
 * [한국어]
 * symbol::get_uid - 이 심볼에 할당할 고유 UID를 전역 카운터에서 채번
 *
 * @return: 새로 할당된 UID (unsigned). 후위증가이므로 현재 값을 반환 후 증가.
 *
 * gpgpu_context::symbol_sm_next_uid는 시뮬레이터 전체에서 공유되는 단조증가 카운터이다.
 * 심볼 생성자(symbol::symbol)에서 호출되어 각 심볼이 고유한 숫자 ID를 갖게 한다.
 * 이 UID는 디버그 출력과 내부 식별에 사용된다. 싱글스레드 초기화 단계에서만
 * 호출되므로 별도 동기화는 불필요하다.
 *
 * 호출 체인:
 *   symbol::symbol() (생성자) → [get_uid] → gpgpu_ctx->symbol_sm_next_uid++
 */
unsigned symbol::get_uid() {
  unsigned result = (gpgpu_ctx->symbol_sm_next_uid)++;  // [한국어] 전역 카운터 현재 값을 UID로 채번 후 카운터 증가
  return result;                                         // [한국어] 채번된 UID 반환 (이 심볼의 고유 식별자)
}

/*
 * [한국어]
 * symbol::add_initializer - 전역/상수 변수의 초기값 리스트를 심볼에 저장
 *
 * @init: PTX .global/.const 변수 선언부의 초기화 값 목록 (operand_info 리스트).
 *        파서가 "{1, 2, 3}" 형태의 초기화 구문을 파싱하여 넘겨주는 값이다.
 * @return: 없음
 *
 * 파서가 전역/상수 변수 선언을 처리할 때 초기값이 있으면 이 함수를 호출한다.
 * 저장된 m_initializer는 function_info::finalize()에서 실제 메모리(global_mem,
 * const_mem)에 기록될 때 사용된다. 레지스터나 지역 변수는 초기값을 갖지 않으므로
 * 이 함수가 호출되지 않는다.
 *
 * 호출 체인:
 *   ptx_parser.cc(변수 선언 처리) → [add_initializer] → m_initializer 저장
 *   → function_info::finalize() 에서 실제 메모리 기록 시 참조
 */
void symbol::add_initializer(const std::list<operand_info> &init) {
  m_initializer = init;  // [한국어] 초기화 값 목록을 심볼 내부에 복사 저장
}

/*
 * [한국어]
 * symbol::print_info - 디버그용 심볼 속성 덤프 출력
 *
 * @fp: 출력 대상 FILE 포인터 (보통 stdout).
 * @return: 없음
 *
 * 심볼의 UID, 선언 위치, 타입 포인터, 주소 유효성, 속성 플래그들을
 * 한 줄로 출력하는 디버그 함수이다. symbol_table::dump()가 모든 심볼을
 * 순회하며 이 함수를 호출한다. 시뮬레이터 디버그 레벨이 높을 때만 실제로 사용된다.
 *
 * 호출 체인:
 *   symbol_table::dump() → [print_info] → fprintf()
 */
void symbol::print_info(FILE *fp) const {
  fprintf(fp, "uid:%u, decl:%s, type:%p, ", m_uid, m_decl_location.c_str(),
          m_type);                                          // [한국어] UID, 선언 위치(파일:라인), 타입 포인터 출력
  if (m_address_valid) fprintf(fp, "<address valid>, ");   // [한국어] 메모리 주소가 유효하게 할당된 심볼인 경우 표시
  if (m_is_label) fprintf(fp, " is_label ");               // [한국어] PTX 레이블 심볼 (분기 대상 이름)
  if (m_is_shared) fprintf(fp, " is_shared ");             // [한국어] .shared 메모리 공간의 심볼
  if (m_is_const) fprintf(fp, " is_const ");               // [한국어] .const 상수 메모리 공간의 심볼
  if (m_is_global) fprintf(fp, " is_global ");             // [한국어] .global 전역 메모리 공간의 심볼
  if (m_is_local) fprintf(fp, " is_local ");               // [한국어] .local 스레드 지역 메모리 공간의 심볼
  if (m_is_tex) fprintf(fp, " is_tex ");                   // [한국어] 텍스처 레퍼런스 심볼
  if (m_is_func_addr) fprintf(fp, " is_func_addr ");       // [한국어] 함수 주소를 저장하는 심볼 (간접 호출용)
  if (m_function) fprintf(fp, " %p ", m_function);         // [한국어] 연결된 function_info 포인터 주소 출력
}

/*
 * [한국어]
 * symbol_table::symbol_table (기본 생성자) - 직접 사용 금지 생성자
 *
 * @return: 없음 (생성자)
 *
 * 이 기본 생성자는 절대 직접 호출되어서는 안 된다. symbol_table 객체는
 * 반드시 스코프 이름, 진입점 여부, 부모 테이블, gpgpu_context 포인터를 갖는
 * 파라미터 생성자를 통해서만 올바르게 초기화된다.
 * assert(0)으로 즉시 중단하여 잘못된 사용을 컴파일 타임이 아닌 런타임에 방지한다.
 *
 * 호출 체인: (호출되면 안 됨) → assert(0) 중단
 */
symbol_table::symbol_table() { assert(0); }  // [한국어] 기본 생성자 — 직접 사용 금지, 런타임 assert로 오용 방지

/*
 * [한국어]
 * symbol_table::symbol_table (파라미터 생성자) - 새 심볼 테이블 스코프 초기화
 *
 * @scope_name: 이 스코프의 이름 (보통 함수명 또는 빈 문자열).
 * @entry_point: 진입점 종류 — 1=커널 엔트리, 2=extern 함수, 3=CDP 명령어 그룹,
 *               0=일반 디바이스 함수.
 * @parent: 부모 심볼 테이블 포인터. NULL이면 최상위(파일 스코프).
 *          .global/.shared 주소 카운터를 부모로부터 상속받는다.
 * @ctx: 시뮬레이터 전역 컨텍스트 포인터 (UID 카운터 등 공유 상태 접근용).
 * @return: 없음 (생성자)
 *
 * 모든 주소 할당 카운터를 초기화하고, 부모가 존재하면 .shared와 .global 카운터를
 * 부모로부터 이어받는다. 이는 자식 스코프(함수)가 부모의 전역 메모리 레이아웃을
 * 이어서 사용해야 하기 때문이다. .local과 .reg는 스코프마다 독립적으로 시작한다.
 * m_global_next가 0x100부터 시작하는 이유는 0~0xFF를 NULL/특수값으로 예약하기 위함.
 *
 * 호출 체인:
 *   symbol_table::add_function_decl() → new symbol_table("", entry_point, this, ctx)
 *   symbol_table::start_inst_group() → new symbol_table(name, 3, this, ctx)
 */
symbol_table::symbol_table(const char *scope_name, unsigned entry_point,
                           symbol_table *parent, gpgpu_context *ctx) {
  gpgpu_ctx = ctx;                        // [한국어] 전역 컨텍스트 포인터 저장 — UID 카운터 접근에 사용
  m_scope_name = std::string(scope_name); // [한국어] 스코프 이름 저장 (디버그 출력 및 식별용)
  m_reg_allocator = 0;                    // [한국어] 레지스터 번호 할당기 초기화 — 0번부터 새 레지스터 할당
  m_shared_next = 0;                      // [한국어] .shared 메모리 다음 빈 오프셋 — 0바이트부터 시작
  m_const_next = 0;                       // [한국어] .const 상수 메모리 다음 빈 오프셋
  m_global_next = 0x100;                  // [한국어] .global 전역 메모리 다음 빈 주소 — 0x100부터 시작
                                          //         (0~0xFF는 NULL 등 특수 용도로 예약)
  m_local_next = 0;                       // [한국어] .local 스레드 지역 메모리 다음 빈 오프셋
  m_tex_next = 0;                         // [한국어] 텍스처 레퍼런스 다음 인덱스

  // Jin: handle instruction group for cdp
  m_inst_group_id = 0;                    // [한국어] CDP(CUDA Dynamic Parallelism) 명령어 그룹 ID — 0부터 시작

  m_parent = parent;                      // [한국어] 부모 심볼 테이블 포인터 저장 (NULL이면 최상위 스코프)
  if (m_parent) {                         // [한국어] 부모가 존재하면 .shared/.global 주소 카운터를 상속
    m_shared_next = m_parent->m_shared_next;  // [한국어] 부모의 .shared 주소 이어받기 — 자식이 부모와 같은 공유 메모리 레이아웃 사용
    m_global_next = m_parent->m_global_next;  // [한국어] 부모의 .global 주소 이어받기 — 전역 변수 주소 연속 할당 보장
  }
}

/*
 * [한국어]
 * symbol_table::set_name - 스코프 이름을 나중에 설정하는 세터
 *
 * @name: 새로 설정할 스코프 이름 (보통 함수명).
 * @return: 없음
 *
 * add_function_decl()에서 빈 문자열("")로 심볼 테이블을 먼저 생성한 후,
 * 함수 이름이 확정되면 이 함수로 이름을 갱신한다.
 *
 * 호출 체인:
 *   symbol_table::add_function_decl() → [set_name(name)] → m_scope_name 갱신
 */
void symbol_table::set_name(const char *name) {
  m_scope_name = std::string(name);  // [한국어] 스코프 이름을 C 문자열에서 std::string으로 변환하여 저장
}

/*
 * [한국어]
 * symbol_table::get_ptx_version - PTX 버전 정보를 재귀적으로 루트 스코프에서 조회
 *
 * @return: ptx_version 상수 참조. .version 지시어가 선언된 파일 스코프(루트)의 버전.
 *
 * PTX .version 지시어는 파일 최상위 스코프에만 선언되므로, 자식 스코프에서
 * 버전을 조회할 때는 부모를 따라 루트까지 재귀 탐색한다. 이를 통해 함수 스코프에서도
 * PTX 버전 조건 분기가 가능하다 (예: ver >= 6.0이면 wmma 지원).
 *
 * 호출 체인:
 *   ptx_parser.cc(버전 조건 분기) → [get_ptx_version] → 루트 m_ptx_version 반환
 */
const ptx_version &symbol_table::get_ptx_version() const {
  if (m_parent == NULL)             // [한국어] 부모가 없으면 이 스코프가 루트 — 자신의 버전 반환
    return m_ptx_version;
  else
    return m_parent->get_ptx_version();  // [한국어] 부모를 따라 재귀 탐색 — 루트에 .version 지시어가 있음
}

/*
 * [한국어]
 * symbol_table::get_sm_target - SM(Streaming Multiprocessor) 타겟 아키텍처 번호 조회
 *
 * @return: SM 타겟 번호 (예: sm_70은 700, sm_80은 800). gpgpusim.config의 -gpgpu_ptx_force_max_capability와 비교에 사용.
 *
 * .target 지시어는 파일 루트 스코프에 있으므로, get_ptx_version()과 동일하게
 * 루트까지 재귀 탐색한다. 타이밍 모델 설정(wmma 지원 여부 등)에 영향을 준다.
 *
 * 호출 체인:
 *   ptx_parser.cc / cuda-sim.cc → [get_sm_target] → 루트 m_ptx_version.target()
 */
unsigned symbol_table::get_sm_target() const {
  if (m_parent == NULL)                  // [한국어] 루트 스코프 — .target 지시어가 여기 저장됨
    return m_ptx_version.target();       // [한국어] ptx_version 객체에서 SM 타겟 번호 반환
  else
    return m_parent->get_sm_target();    // [한국어] 부모 방향으로 재귀 탐색
}

/*
 * [한국어]
 * symbol_table::set_ptx_version - PTX 버전 정보를 파일 루트 스코프에 저장
 *
 * @ver: PTX 버전 실수 (예: 6.5f → PTX ISA 6.5).
 * @ext: 확장 버전 플래그 (버전 관련 부가 정보, 보통 0).
 * @return: 없음
 *
 * 파서가 ".version X.Y" 지시어를 만나면 현재(루트) 심볼 테이블에 이 함수를 호출한다.
 * ptx_version 객체를 새로 생성하여 m_ptx_version에 저장한다.
 *
 * 호출 체인:
 *   ptx_parser.cc(".version" 지시어 처리) → [set_ptx_version]
 */
void symbol_table::set_ptx_version(float ver, unsigned ext) {
  m_ptx_version = ptx_version(ver, ext);  // [한국어] PTX 버전 객체 생성 및 저장 (예: ver=6.5, ext=0)
}

/*
 * [한국어]
 * symbol_table::set_sm_target - SM 타겟 아키텍처를 .target 문자열로 설정
 *
 * @target: 기본 타겟 문자열 (예: "sm_70", "sm_80").
 * @ext: 첫 번째 확장 옵션 (예: "texmode_unified", NULL 가능).
 * @ext2: 두 번째 확장 옵션 (NULL 가능).
 * @return: 없음
 *
 * 파서가 ".target sm_XX" 지시어를 만나면 호출된다. ptx_version 내부의
 * set_target() 메서드로 위임하여 SM 번호와 텍스처 모드 등 옵션을 파싱한다.
 *
 * 호출 체인:
 *   ptx_parser.cc(".target" 지시어 처리) → [set_sm_target] → m_ptx_version.set_target()
 */
void symbol_table::set_sm_target(const char *target, const char *ext,
                                 const char *ext2) {
  m_ptx_version.set_target(target, ext, ext2);  // [한국어] ptx_version 객체에 타겟 문자열 파싱 위임
}

/*
 * [한국어]
 * symbol_table::lookup - 식별자 이름으로 심볼을 현재 → 부모 스코프 순서로 탐색
 *
 * @identifier: 탐색할 심볼 이름 (PTX 식별자 문자열).
 * @return: 발견한 symbol 포인터. 모든 스코프에서 찾지 못하면 NULL.
 *
 * C/C++ 언어의 이름 탐색 규칙과 동일하게, 현재 스코프의 m_symbols 맵을 먼저
 * 검색하고, 없으면 m_parent->lookup()을 재귀 호출하여 부모(파일 스코프)까지
 * 탐색한다. PTX 파서가 오퍼랜드 이름을 분석할 때 호출하며, 반환된 symbol*로
 * 레지스터/메모리 공간/주소를 결정한다. NULL이면 "undefined symbol" 오류.
 *
 * 호출 체인:
 *   ptx_parser.cc(오퍼랜드 분석) → [lookup] → m_symbols.find() or m_parent->lookup()
 */
symbol *symbol_table::lookup(const char *identifier) {
  std::string key(identifier);                           // [한국어] C 문자열을 std::string 키로 변환 (맵 탐색용)
  std::map<std::string, symbol *>::iterator i = m_symbols.find(key);  // [한국어] 현재 스코프 심볼 맵에서 이름 탐색
  if (i != m_symbols.end()) {                            // [한국어] 현재 스코프에서 발견 — 즉시 반환
    return i->second;
  }
  if (m_parent) {                                        // [한국어] 현재 스코프에 없으면 부모 스코프로 탐색 범위 확장
    return m_parent->lookup(identifier);                 // [한국어] 재귀 호출 — 루트 스코프까지 탐색
  }
  return NULL;                                           // [한국어] 루트까지 탐색했으나 없음 — 미정의 심볼
}

/*
 * [한국어]
 * symbol_table::lookup_by_addr - 메모리 주소 값으로 심볼을 역방향 탐색
 *
 * @addr: 찾고자 하는 메모리 주소 값 (addr_t).
 * @return: 해당 주소가 할당된 symbol 포인터. 없으면 NULL.
 *
 * 주소→심볼 역방향 조회 함수로, 디버그나 역참조 시 사용된다.
 * m_symbols 맵 전체를 선형 탐색하므로 심볼 수에 비례하는 O(N) 비용이 발생한다.
 * 레지스터(is_reg())는 주소가 없으므로 건너뛰고, 주소가 유효한(has_valid_address())
 * 심볼만 대상으로 비교한다. 현재 스코프에서 못 찾으면 부모 스코프로 재귀한다.
 *
 * 호출 체인:
 *   cuda-sim.cc(역참조 디버그) → [lookup_by_addr] → 선형 탐색 → 부모 재귀
 */
symbol *symbol_table::lookup_by_addr(addr_t addr) {
  for (auto it = m_symbols.begin(); it != m_symbols.end(); ++it) {  // [한국어] 현재 스코프 심볼 맵 전체 선형 탐색
    symbol *sym = it->second;                                        // [한국어] 맵 값(symbol 포인터) 추출

    // check if symbol has the addr to be found
    if ((!sym->is_reg()) && (sym->has_valid_address()) &&           // [한국어] 레지스터 제외, 주소 유효성 확인
        (sym->get_address() == addr)) {                             // [한국어] 주소 값 일치 여부 확인
      return sym;                                                    // [한국어] 일치하는 심볼 발견 — 반환
    }
  }
  if (m_parent) {                                                    // [한국어] 현재 스코프에서 못 찾으면 부모로 확장
    return m_parent->lookup_by_addr(addr);                           // [한국어] 부모 스코프에서 재귀 탐색
  }
  return NULL;                                                       // [한국어] 모든 스코프에서 해당 주소 심볼 없음
}

/*
 * [한국어]
 * symbol_table::add_variable - 새 변수/레지스터 심볼을 현재 스코프에 등록
 *
 * @identifier: PTX 식별자 이름 (예: "%r0", "myGlobalVar").
 * @type: 변수의 타입 정보 포인터 (메모리 공간, 스칼라 타입, 배열 차원 등 포함).
 * @size: 바이트 단위 크기 (0이면 타입에서 추론).
 * @filename: 이 선언이 위치한 PTX 소스 파일명 (디버그 정보용).
 * @line: 소스 파일 내 줄 번호 (디버그 정보용).
 * @return: 새로 생성된 symbol 포인터.
 *
 * 파서가 변수 선언을 처리할 때 호출된다. 이미 같은 이름의 심볼이 있으면
 * assert(0)으로 중단한다 (PTX는 스코프 내 중복 선언 불허). 타입이 .global이면
 * m_globals 목록에, .const이면 m_consts 목록에 추가로 등록한다.
 * 이 목록은 finalize() 단계에서 전역/상수 메모리 초기화에 사용된다.
 *
 * 호출 체인:
 *   ptx_parser.cc(변수 선언) → [add_variable] → new symbol() 생성 → m_symbols 등록
 *   → (조건부) m_globals / m_consts 추가
 */
symbol *symbol_table::add_variable(const char *identifier,
                                   const type_info *type, unsigned size,
                                   const char *filename, unsigned line) {
  char buf[1024];                                    // [한국어] "파일명:줄번호" 형태 문자열 버퍼
  std::string key(identifier);                       // [한국어] 식별자를 std::string 키로 변환
  assert(m_symbols.find(key) == m_symbols.end());   // [한국어] 중복 선언 방지 — 이미 있으면 assert(0) 중단
  snprintf(buf, 1024, "%s:%u", filename, line);      // [한국어] "파일:라인" 형식의 선언 위치 문자열 생성
  symbol *s = new symbol(identifier, type, buf, size, gpgpu_ctx);  // [한국어] 새 심볼 객체 동적 할당 (생성자 내부에서 UID 채번)
  m_symbols[key] = s;                                // [한국어] 현재 스코프 심볼 맵에 등록 (이름 → 심볼 매핑)

  if (type != NULL && type->get_key().is_global()) { // [한국어] .global 공간 변수인 경우
    m_globals.push_back(s);                          // [한국어] 전역 변수 목록에 추가 — finalize()에서 global_mem 초기화에 사용
  }
  if (type != NULL && type->get_key().is_const()) {  // [한국어] .const 공간 변수인 경우
    m_consts.push_back(s);                           // [한국어] 상수 변수 목록에 추가 — finalize()에서 const_mem 초기화에 사용
  }

  return s;                                          // [한국어] 생성된 심볼 포인터 반환 (파서가 추가 속성 설정에 사용)
}

/*
 * [한국어]
 * symbol_table::add_function - 함수 이름을 심볼로 심볼 테이블에 등록
 *
 * @func: 등록할 함수의 function_info 포인터.
 * @filename: 이 함수 선언이 위치한 PTX 소스 파일명.
 * @linenumber: 소스 파일 내 줄 번호.
 * @return: 없음
 *
 * 함수명을 심볼 테이블에 등록하여 PTX 코드에서 함수를 참조(CALL 등)할 때
 * lookup()으로 찾을 수 있게 한다. 이미 등록된 함수면 조용히 반환(중복 등록 방지).
 * 함수 타입(is_func=true)의 type_info를 만들고 symbol에 func 포인터를 연결한다.
 * add_function_decl()과 달리 이 함수는 symbol_table과 function_info를 새로
 * 생성하지 않으며, 단순히 이름→함수 매핑만 심볼 테이블에 추가한다.
 *
 * 호출 체인:
 *   ptx_parser.cc(함수 정의 완료 시) → [add_function] → add_type(func) → new symbol()
 */
void symbol_table::add_function(function_info *func, const char *filename,
                                unsigned linenumber) {
  std::map<std::string, symbol *>::iterator i =
      m_symbols.find(func->get_name());        // [한국어] 이미 등록된 함수 심볼인지 확인
  if (i != m_symbols.end()) return;            // [한국어] 이미 등록됨 — 중복 등록 방지를 위해 조용히 반환
  char buf[1024];                              // [한국어] 선언 위치 문자열 버퍼
  snprintf(buf, 1024, "%s:%u", filename, linenumber);  // [한국어] "파일:라인" 선언 위치 문자열 생성
  type_info *type = add_type(func);            // [한국어] 함수 타입의 type_info 생성 (is_func=true 플래그 설정)
  symbol *s = new symbol(func->get_name().c_str(), type, buf, 0, gpgpu_ctx);  // [한국어] 함수명 심볼 생성 (크기=0, 주소=미할당)
  s->set_function(func);                       // [한국어] 심볼에 function_info 포인터 연결 — 심볼에서 함수 구현체 접근 가능
  m_symbols[func->get_name()] = s;            // [한국어] 함수명을 키로 심볼 테이블에 등록
}

// Jin: handle instruction group for cdp
/*
 * [한국어]
 * symbol_table::start_inst_group - CDP 명령어 그룹을 위한 자식 심볼 테이블 생성
 *
 * @return: 새로 생성된 자식 symbol_table 포인터.
 *
 * CDP(CUDA Dynamic Parallelism)에서는 디바이스 코드에서 직접 새 커널을 런치할 수 있다.
 * 이 경우 PTX에서 특수한 명령어 그룹(inst_group)이 생성되며, 이 그룹은 별도의
 * 심볼 테이블 스코프를 가진다. 새 자식 심볼 테이블을 생성하고, 부모의 모든 주소
 * 카운터를 완전히 복사하여 연속적인 주소 할당이 유지되도록 한다.
 * 이미 같은 이름의 그룹이 있으면 assert(0)으로 중단한다 (각 그룹은 유일해야 함).
 *
 * 호출 체인:
 *   ptx_parser.cc(CDP 명령어 그룹 시작) → [start_inst_group] → new symbol_table(3)
 */
symbol_table *symbol_table::start_inst_group() {
  char inst_group_name[4096];                                  // [한국어] 명령어 그룹 이름 버퍼 (4096바이트)
  snprintf(inst_group_name, 4096, "%s_inst_group_%u", m_scope_name.c_str(),
           m_inst_group_id);  // [한국어] "<스코프명>_inst_group_<번호>" 형태의 유일한 이름 생성

  // previous added
  assert(m_inst_group_symtab.find(std::string(inst_group_name)) ==
         m_inst_group_symtab.end());  // [한국어] 같은 이름의 그룹이 이미 존재하면 중단 (중복 생성 방지)
  symbol_table *sym_table =
      new symbol_table(inst_group_name, 3 /*inst group*/, this, gpgpu_ctx);
  // [한국어] entry_point=3(CDP 명령어 그룹)으로 자식 심볼 테이블 생성, 부모=this

  sym_table->m_global_next = m_global_next;    // [한국어] 부모의 전역 메모리 할당 위치 동기화
  sym_table->m_shared_next = m_shared_next;    // [한국어] 부모의 공유 메모리 할당 위치 동기화
  sym_table->m_local_next = m_local_next;      // [한국어] 부모의 지역 메모리 할당 위치 동기화
  sym_table->m_reg_allocator = m_reg_allocator;// [한국어] 부모의 레지스터 번호 할당기 동기화
  sym_table->m_tex_next = m_tex_next;          // [한국어] 부모의 텍스처 인덱스 동기화
  sym_table->m_const_next = m_const_next;      // [한국어] 부모의 상수 메모리 할당 위치 동기화

  m_inst_group_symtab[std::string(inst_group_name)] = sym_table;  // [한국어] 부모의 그룹 맵에 자식 테이블 등록

  return sym_table;  // [한국어] 새 자식 심볼 테이블 반환 — 파서가 이후 심볼 등록의 현재 컨텍스트로 사용
}

/*
 * [한국어]
 * symbol_table::end_inst_group - CDP 명령어 그룹 종료 및 부모 심볼 테이블로 복귀
 *
 * @return: 부모 symbol_table 포인터.
 *
 * CDP 명령어 그룹 처리가 끝나면 이 함수를 호출하여 자식에서 부모 스코프로 복귀한다.
 * 자식 심볼 테이블에서 변경된 모든 주소 카운터를 부모에 역전파(write-back)하여
 * 부모가 그룹 이후의 심볼을 연속된 주소에 할당할 수 있게 한다.
 * m_inst_group_id를 증가시켜 다음 그룹이 다른 이름을 갖게 한다.
 *
 * 호출 체인:
 *   ptx_parser.cc(CDP 명령어 그룹 종료) → [end_inst_group] → 부모 카운터 갱신 → 부모 반환
 */
symbol_table *symbol_table::end_inst_group() {
  symbol_table *sym_table = m_parent;                   // [한국어] 부모 심볼 테이블 포인터 가져오기

  sym_table->m_global_next = m_global_next;             // [한국어] 그룹 내 할당된 전역 메모리 위치를 부모에 반영
  sym_table->m_shared_next = m_shared_next;             // [한국어] 그룹 내 할당된 공유 메모리 위치를 부모에 반영
  sym_table->m_local_next = m_local_next;               // [한국어] 그룹 내 할당된 지역 메모리 위치를 부모에 반영
  sym_table->m_reg_allocator = m_reg_allocator;         // [한국어] 그룹 내 할당된 레지스터 번호를 부모에 반영
  sym_table->m_tex_next = m_tex_next;                   // [한국어] 그룹 내 할당된 텍스처 인덱스를 부모에 반영
  sym_table->m_const_next = m_const_next;               // [한국어] 그룹 내 할당된 상수 메모리 위치를 부모에 반영
  sym_table->m_inst_group_id++;                         // [한국어] 부모의 그룹 ID 증가 — 다음 그룹은 다른 이름을 가짐

  return sym_table;  // [한국어] 부모 심볼 테이블 반환 — 파서가 이후 심볼 등록의 현재 컨텍스트로 복귀
}

/*
 * [한국어]
 * register_ptx_function - PTX 함수를 런타임에 등록하는 외부 함수 (libcuda 또는 libopencl 구현)
 *
 * @name: 등록할 함수 이름 (PTX 함수명, C++ mangled 가능).
 * @impl: 이 함수의 구현을 담은 function_info 포인터.
 *
 * libcuda 또는 libopencl에서 구현되며, CUDA 런타임이 커널 런치 요청 시
 * 함수 이름으로 function_info를 찾을 수 있도록 전역 테이블에 등록한다.
 * add_function_decl()의 마지막 단계에서 호출된다.
 */
void register_ptx_function(const char *name,
                           function_info *impl);  // either libcuda or libopencl
// [한국어] libcuda/libopencl에서 정의 — PTX 함수명을 전역 함수 레지스트리에 등록

/*
 * [한국어]
 * symbol_table::add_function_decl - 함수 선언을 처리하여 function_info와 symbol_table을 생성·등록
 *
 * @name: 함수 이름 (PTX .func 또는 .entry 지시어의 함수명).
 * @entry_point: 진입점 종류 — 1=커널 엔트리, 2=extern, 0=디바이스 함수.
 * @func_info: [출력] 생성되거나 기존 function_info 포인터를 여기에 반환.
 * @sym_table: [출력] 생성되거나 기존 symbol_table 포인터를 여기에 반환.
 * @return: true이면 이미 이전에 선언된 함수 (forward declaration 존재),
 *          false이면 이번에 처음 선언됨.
 *
 * PTX 파서가 .func/.entry 지시어를 만날 때마다 호출된다. 함수 이름을 키로
 * m_function_info_lookup과 m_function_symtab_lookup을 검색하여, 이미 선언된
 * 함수면 기존 객체를 반환하고 (forward decl 이후 정의 시), 처음이면 새로 생성한다.
 * 새로 생성하는 경우 특수 레지스터 "_"(비아키텍처 레지스터)를 심볼 테이블에 추가하고
 * register_ptx_function()을 호출하여 런타임에 등록한다.
 *
 * "_" 레지스터: PTX에서 오퍼랜드를 사용하지 않을 때 쓰는 더미 레지스터.
 * 파서는 이를 유효한 레지스터로 인식하지만, 타이밍 시뮬레이터는 무시한다
 * (non_arch_reg 플래그).
 *
 * 호출 체인:
 *   ptx_parser.cc(.func/.entry 지시어 처리) → [add_function_decl]
 *   → new function_info() + new symbol_table() + add_variable("_")
 *   → register_ptx_function()
 */
bool symbol_table::add_function_decl(const char *name, int entry_point,
                                     function_info **func_info,
                                     symbol_table **sym_table) {
  std::string key = std::string(name);          // [한국어] 함수명을 맵 탐색 키로 변환
  bool prior_decl = false;                      // [한국어] 이전 선언 존재 여부 플래그
  if (m_function_info_lookup.find(key) != m_function_info_lookup.end()) {
    // [한국어] 이미 function_info가 등록된 함수 — forward declaration 이후 정의 또는 중복 선언
    *func_info = m_function_info_lookup[key];   // [한국어] 기존 function_info 반환
    prior_decl = true;                          // [한국어] 이전 선언 존재 플래그 설정
  } else {
    // [한국어] 처음 보는 함수 — 새 function_info 생성
    *func_info = new function_info(entry_point, gpgpu_ctx);  // [한국어] function_info 동적 할당 (진입점 플래그, 컨텍스트 포함)
    (*func_info)->set_name(name);               // [한국어] 함수 이름 설정
    (*func_info)->set_maxnt_id(0);              // [한국어] 최대 스레드 수 초기값 0 설정 (나중에 .maxntid 지시어로 갱신)
    m_function_info_lookup[key] = *func_info;   // [한국어] 이름 → function_info 맵에 등록
  }

  if (m_function_symtab_lookup.find(key) != m_function_symtab_lookup.end()) {
    // [한국어] 이미 심볼 테이블이 있는 함수 — function_info 맵과 일관성 검증
    assert(prior_decl);                         // [한국어] 심볼 테이블이 있으면 function_info도 이전에 만들어졌어야 함
    *sym_table = m_function_symtab_lookup[key]; // [한국어] 기존 심볼 테이블 반환
  } else {
    // [한국어] 처음 보는 함수 — 새 심볼 테이블 생성
    assert(!prior_decl);                        // [한국어] 심볼 테이블 없으면 function_info도 새로 만든 것이어야 함
    *sym_table = new symbol_table("", entry_point, this, gpgpu_ctx);
    // [한국어] 빈 이름("")으로 함수 스코프 심볼 테이블 생성, 부모=현재 파일 스코프

    // Initial setup code to support a register represented as "_".
    // This register is used when an instruction operand is
    // not read or written.  However, the parser must recognize it
    // as a legitimate register but we do not want to pass
    // it to the micro-architectural register to the performance simulator.
    // For this purpose we add a symbol to the symbol table but
    // mark it as a non_arch_reg so it does not effect the performance sim.
    type_info_key null_key(reg_space, 0, 0, 0, 0, 0);  // [한국어] .reg 공간의 0비트 타입 키 생성 (최소 타입)
    null_key.set_is_non_arch_reg();             // [한국어] 비아키텍처 레지스터 플래그 설정 — 타이밍 시뮬레이터가 무시
    // First param is null - which is bad.
    // However, the first parameter is actually unread in the constructor...
    // TODO - remove the symbol_table* from type_info
    type_info *null_type_info = new type_info(NULL, null_key);
    // [한국어] NULL 심볼 테이블로 type_info 생성 (생성자에서 first param 미사용, 기술 부채 주석 참고)
    symbol *null_reg =
        (*sym_table)->add_variable("_", null_type_info, 0, "", 0);
    // [한국어] "_" 이름으로 더미 레지스터 심볼 등록 (크기=0, 파일/라인 정보 없음)
    null_reg->set_regno(0, 0);                  // [한국어] 레지스터 번호 0으로 설정 — 실제 레지스터 파일에 영향 없음

    (*sym_table)->set_name(name);               // [한국어] 빈 이름("")으로 생성한 심볼 테이블에 실제 함수명 설정
    (*func_info)->set_symtab(*sym_table);        // [한국어] function_info에 자신의 심볼 테이블 연결
    m_function_symtab_lookup[key] = *sym_table; // [한국어] 이름 → 심볼 테이블 맵에 등록
    assert((*func_info)->get_symtab() == *sym_table);  // [한국어] 연결 일관성 검증
    register_ptx_function(name, *func_info);    // [한국어] libcuda/libopencl 전역 함수 레지스트리에 등록
  }
  return prior_decl;  // [한국어] 이전 선언 존재 여부 반환 — 파서가 forward decl vs 정의를 구분하는 데 사용
}

/*
 * [한국어]
 * symbol_table::lookup_function - 함수 이름으로 function_info를 조회 (반드시 존재해야 함)
 *
 * @name: 조회할 함수 이름 (PTX 함수명).
 * @return: 해당 함수의 function_info 포인터.
 *
 * m_function_info_lookup 맵에서 이름을 찾아 function_info를 반환한다.
 * lookup()과 달리 부모 스코프로 탐색하지 않으며, 없으면 assert(0)으로 중단한다.
 * 함수가 반드시 등록되어 있어야 하는 상황(예: CALL 명령어 처리 시)에서 사용한다.
 *
 * 호출 체인:
 *   ptx_parser.cc(CALL 처리) / cuda-sim.cc → [lookup_function] → m_function_info_lookup
 */
function_info *symbol_table::lookup_function(std::string name) {
  std::string key = std::string(name);                  // [한국어] std::string 키로 변환
  std::map<std::string, function_info *>::iterator it =
      m_function_info_lookup.find(key);                 // [한국어] 함수 이름→function_info 맵 탐색
  assert(it != m_function_info_lookup.end());           // [한국어] 미등록 함수면 즉시 중단 — 반드시 존재해야 함
  return it->second;                                    // [한국어] 등록된 function_info 포인터 반환
}

/*
 * [한국어]
 * symbol_table::add_type (변수용) - 변수의 메모리 공간/타입 조합으로 type_info 생성
 *
 * @space_spec: 메모리 공간 종류 (reg_space, global_space, shared_space, param_space_local 등).
 *              param_space_unclassified이면 자동으로 param_space_local로 변환한다.
 * @scalar_type_spec: 스칼라 타입 토큰 (S32_TYPE, F32_TYPE, B8_TYPE 등).
 * @vector_spec: 벡터 너비 (V2_TYPE, V4_TYPE, 0=스칼라).
 * @alignment_spec: 정렬 요구사항 (바이트 단위, 0=기본).
 * @extern_spec: extern 여부 플래그 (0 또는 1).
 * @return: 새로 생성된 type_info 포인터.
 *
 * 파서가 변수 선언의 타입 지정자를 처리할 때 호출하여 type_info 객체를 얻는다.
 * 이 객체는 add_variable()에 전달되어 심볼의 타입으로 연결된다.
 * param_space_unclassified는 파싱 중 공간이 명확하지 않은 .param의 중간 상태이며,
 * 이 시점에서 param_space_local로 확정한다.
 *
 * 호출 체인:
 *   ptx_parser.cc(타입 지정자 처리) → [add_type(변수용)] → new type_info()
 */
type_info *symbol_table::add_type(memory_space_t space_spec,
                                  int scalar_type_spec, int vector_spec,
                                  int alignment_spec, int extern_spec) {
  if (space_spec == param_space_unclassified) space_spec = param_space_local;
  // [한국어] 분류되지 않은 .param 공간을 .param(local)으로 확정 — 디바이스 함수의 지역 param
  type_info_key t(space_spec, scalar_type_spec, vector_spec, alignment_spec,
                  extern_spec, 0);                      // [한국어] 타입 키 생성 (공간, 스칼라타입, 벡터, 정렬, extern, 배열차원)
  type_info *pt;
  pt = new type_info(this, t);                          // [한국어] 이 심볼 테이블을 소유자로 type_info 동적 할당
  return pt;                                            // [한국어] 생성된 type_info 반환 — add_variable()에 전달됨
}

/*
 * [한국어]
 * symbol_table::add_type (함수용) - 함수 타입의 type_info 생성
 *
 * @func: 타입을 생성할 함수의 function_info 포인터 (현재 미사용, 시그니처용).
 * @return: is_func 플래그가 설정된 새 type_info 포인터.
 *
 * 함수 심볼의 타입을 나타내는 type_info를 생성한다. set_is_func()로 함수 타입임을
 * 표시하며, 이 type_info를 가진 심볼은 add_function()에서 함수 주소 심볼로 처리된다.
 *
 * 호출 체인:
 *   symbol_table::add_function() → [add_type(함수용)] → new type_info(is_func=true)
 */
type_info *symbol_table::add_type(function_info *func) {
  type_info_key t;                 // [한국어] 기본값으로 type_info_key 생성
  type_info *pt;
  t.set_is_func();                 // [한국어] 함수 타입 플래그 설정 — is_reg()나 is_global() 등과 구별됨
  pt = new type_info(this, t);     // [한국어] 함수 타입의 type_info 동적 할당
  return pt;                       // [한국어] 생성된 type_info 반환
}

/*
 * [한국어]
 * symbol_table::get_array_type - 기본 타입에 배열 차원을 추가하여 배열 타입 생성
 *
 * @base_type: 배열 원소의 기본 type_info (예: .u32 타입).
 * @array_dim: 배열 원소 개수 (예: 16이면 .u32[16]).
 * @return: 배열 차원이 설정된 새 type_info 포인터.
 *
 * PTX에서 배열 변수 선언 시 사용한다 (예: .global .u32 arr[16]).
 * 기본 타입의 키를 복사하고 배열 차원을 덮어써서 새 type_info를 만든다.
 * 주의: m_types 맵 등록은 세그폴트 문제로 비활성화된 상태 (TODO 주석 참고).
 *
 * 호출 체인:
 *   ptx_parser.cc(배열 변수 선언) → [get_array_type] → new type_info(배열 차원 포함)
 */
type_info *symbol_table::get_array_type(type_info *base_type,
                                        unsigned array_dim) {
  type_info_key t = base_type->get_key();   // [한국어] 기본 타입의 키 복사 (공간, 스칼라 타입 등 그대로)
  t.set_array_dim(array_dim);               // [한국어] 배열 차원 설정 (예: 16이면 16원소 배열)
  type_info *pt = new type_info(this, t);   // [한국어] 배열 타입의 type_info 동적 할당
  // Where else is m_types being used? As of now, I dont find any use of it and
  // causing seg fault. So disabling m_types.
  // TODO: find where m_types can be used in future and solve the seg fault.
  // pt = m_types[t] = new type_info(this,t);
  // [한국어] m_types 맵 등록은 세그폴트 문제로 비활성화 — 향후 수정 필요 (기술 부채)
  return pt;  // [한국어] 배열 타입 반환
}

/*
 * [한국어]
 * symbol_table::set_label_address - 레이블 심볼에 PC 주소를 설정
 *
 * @label: 주소를 설정할 레이블 심볼 포인터 (이름 키로 맵에서 찾음).
 * @addr: 이 레이블에 해당하는 명령어의 PC 값 (ptx_assemble()에서 할당된 값).
 * @return: 없음
 *
 * ptx_assemble() 단계에서 각 레이블이 어떤 명령어 PC에 해당하는지 확정된 후
 * 이 함수를 호출하여 레이블 심볼에 PC 값을 저장한다. 이후 BRA/BREAK 명령어가
 * 레이블 이름으로 분기 대상 PC를 조회할 때 사용된다. assert로 레이블이 반드시
 * 심볼 테이블에 있어야 함을 검증한다.
 *
 * 호출 체인:
 *   function_info::ptx_assemble()(PC 할당 후) → [set_label_address] → symbol::set_label_address()
 */
void symbol_table::set_label_address(const symbol *label, unsigned addr) {
  std::map<std::string, symbol *>::iterator i = m_symbols.find(label->name());
  // [한국어] 레이블 이름으로 심볼 테이블에서 탐색
  assert(i != m_symbols.end());        // [한국어] 레이블이 심볼 테이블에 없으면 중단 (파서가 등록해야 함)
  symbol *s = i->second;               // [한국어] 레이블 심볼 포인터 추출
  s->set_label_address(addr);          // [한국어] 레이블 심볼에 PC 주소 저장
}

/*
 * [한국어]
 * symbol_table::dump - 디버그용으로 이 스코프의 모든 심볼 정보를 출력
 *
 * @return: 없음
 *
 * 현재 심볼 테이블의 스코프 이름과 모든 심볼(이름 + print_info() 출력)을
 * stdout에 덤프한다. 시뮬레이터 디버그 레벨이 높을 때 문제 진단에 사용되며
 * 일반 시뮬레이션에서는 호출되지 않는다.
 *
 * 호출 체인:
 *   (디버그 목적 직접 호출) → [dump] → symbol::print_info() for each symbol
 */
void symbol_table::dump() {
  printf("\n\n");                                    // [한국어] 가독성을 위한 빈 줄 출력
  printf("Symbol table for \"%s\":\n", m_scope_name.c_str());  // [한국어] 스코프 이름 출력
  std::map<std::string, symbol *>::iterator i;
  for (i = m_symbols.begin(); i != m_symbols.end(); i++) {  // [한국어] 모든 심볼 순회
    printf("%30s : ", i->first.c_str());               // [한국어] 30자 폭으로 심볼 이름 출력
    if (i->second)
      i->second->print_info(stdout);                   // [한국어] 심볼 속성 출력 (UID, 타입, 플래그 등)
    else
      printf(" <no symbol object> ");                  // [한국어] 심볼 포인터가 NULL인 비정상 상태
    printf("\n");
  }
  printf("\n");
}

/*
 * [한국어]
 * operand_info::get_uid - 오퍼랜드에 할당할 고유 UID를 전역 카운터에서 채번
 *
 * @return: 새로 할당된 UID (unsigned). 후위증가이므로 현재 값을 반환 후 증가.
 *
 * symbol::get_uid()와 동일한 패턴으로 동작하지만, 오퍼랜드 UID 전용 카운터
 * operand_info_sm_next_uid를 사용한다. operand_info 생성자에서 호출되어
 * 각 오퍼랜드 객체가 고유한 UID를 가지게 한다. 싱글스레드 초기화 단계.
 *
 * 호출 체인:
 *   operand_info::operand_info() (생성자) → [get_uid] → gpgpu_ctx->operand_info_sm_next_uid++
 */
unsigned operand_info::get_uid() {
  unsigned result = (gpgpu_ctx->operand_info_sm_next_uid)++;  // [한국어] 전역 오퍼랜드 UID 카운터 현재 값 채번 후 증가
  return result;                                              // [한국어] 채번된 고유 UID 반환
}

/*
 * [한국어]
 * function_info::find_next_real_instruction - 레이블 명령어를 건너뛰고 다음 실제 명령어 이터레이터 반환
 *
 * @i: 탐색 시작 이터레이터 (m_instructions 리스트의 이터레이터).
 * @return: 레이블이 아닌 다음 ptx_instruction을 가리키는 이터레이터.
 *          실제 명령어가 없으면 m_instructions.end() 반환.
 *
 * PTX에서 레이블은 명령어 목록에 is_label()=true인 더미 ptx_instruction으로
 * 삽입된다. create_basic_blocks()에서 리더 식별 시 레이블만 있는 구간을 건너뛰어
 * 실제 실행 가능한 명령어를 찾는 데 사용한다. 연속된 레이블도 모두 건너뛴다.
 *
 * 호출 체인:
 *   function_info::create_basic_blocks() → [find_next_real_instruction] → 이터레이터 반환
 */
std::list<ptx_instruction *>::iterator
function_info::find_next_real_instruction(
    std::list<ptx_instruction *>::iterator i) {
  while ((i != m_instructions.end()) && (*i)->is_label()) i++;  // [한국어] 레이블인 동안 이터레이터 전진 (연속 레이블 건너뜀)
  return i;  // [한국어] 레이블이 아닌 첫 명령어 이터레이터 반환 (또는 end())
}

/*
 * [한국어]
 * function_info::create_basic_blocks - 리더(leader) 기반 기본 블록 식별 알고리즘 수행
 *
 * @return: 없음 (m_basic_blocks 벡터에 결과 저장)
 *
 * 컴파일러 이론의 표준 리더 기반 기본 블록 식별 알고리즘을 구현한다.
 * 리더(basic block의 첫 명령어)를 다음 규칙으로 결정한다:
 *   1) 함수의 첫 명령어는 리더
 *   2) BRA/RET/EXIT/RETP/BREAK 직후의 명령어는 리더
 *   3) 레이블이 붙은 명령어는 리더 (분기 대상이므로)
 *   4) 술어가 있는(predicated) CALL/CALLP 직후의 명령어는 리더
 * 마지막으로 특수 exit basic_block_t를 추가하여 RET/EXIT의 후계자로 사용한다.
 * 결과는 m_basic_blocks 벡터에 저장되며 do_pdom()이 호출하는 첫 단계이다.
 *
 * 실행 컨텍스트: 시뮬레이션 초기화 단계 (싱글스레드), ptx_assemble() → do_pdom() 내부.
 *
 * 호출 체인:
 *   function_info::do_pdom() → [create_basic_blocks] → basic_block_t 생성들
 *   → connect_basic_blocks()로 이어짐
 */
void function_info::create_basic_blocks() {
  std::list<ptx_instruction *> leaders;        // [한국어] 식별된 리더 명령어 목록 (각 기본 블록의 시작)
  std::list<ptx_instruction *>::iterator i, l; // [한국어] i: 전체 명령어 순회, l: 리더 목록 순회 이터레이터

  // first instruction is a leader
  i = m_instructions.begin();           // [한국어] 명령어 목록의 첫 명령어로 이터레이터 설정
  leaders.push_back(*i);               // [한국어] 규칙 1: 함수 첫 명령어는 무조건 리더
  i++;                                 // [한국어] 다음 명령어로 이동
  while (i != m_instructions.end()) {  // [한국어] 모든 명령어를 순회하며 리더 식별
    ptx_instruction *pI = *i;          // [한국어] 현재 명령어 추출
    if (pI->is_label()) {              // [한국어] 규칙 3: 레이블은 분기 대상 — 리더
      leaders.push_back(pI);           // [한국어] 레이블을 리더로 등록
      i = find_next_real_instruction(++i);  // [한국어] 레이블 뒤 연속 레이블들 건너뛰고 다음 실제 명령어로
    } else {
      switch (pI->get_opcode()) {
        case BRA_OP:                   // [한국어] 분기 명령어 — 다음 명령어가 새 기본 블록 시작
        case RET_OP:                   // [한국어] 함수 반환 — 다음 명령어가 새 기본 블록 시작
        case EXIT_OP:                  // [한국어] 스레드 종료 — 다음 명령어가 새 기본 블록 시작
        case RETP_OP:                  // [한국어] 술어부 반환 — 다음 명령어가 새 기본 블록 시작
        case BREAK_OP:                 // [한국어] 루프 탈출 — 다음 명령어가 새 기본 블록 시작
          i++;                         // [한국어] 분기/반환 명령어 다음 명령어로 이동
          if (i != m_instructions.end()) leaders.push_back(*i);  // [한국어] 규칙 2: 분기 뒤 명령어를 리더로 등록
          i = find_next_real_instruction(i);  // [한국어] 연속 레이블 건너뜀
          break;
        case CALL_OP:                  // [한국어] 함수 호출 — 술어가 있을 때만 다음이 새 블록 시작
        case CALLP_OP:
          if (pI->has_pred()) {        // [한국어] 술어(predicated) call: 조건부이므로 다음 명령어도 도달 가능
            printf("GPGPU-Sim PTX: Warning found predicated call\n");  // [한국어] 술어 call 경고 — 드문 패턴
            i++;
            if (i != m_instructions.end()) leaders.push_back(*i);  // [한국어] 술어 call 뒤 명령어를 리더로 등록
            i = find_next_real_instruction(i);
          } else
            i++;                       // [한국어] 무조건 call: 연속적이므로 다음 명령어가 같은 블록에 속함
          break;
        default:
          i++;                         // [한국어] 일반 명령어: 리더 조건 없음, 단순히 다음으로
      }
    }
  }

  if (leaders.empty()) {               // [한국어] 리더가 없으면 기본 블록도 없는 빈 함수
    printf("GPGPU-Sim PTX: Function \'%s\' has no basic blocks\n",
           m_name.c_str());            // [한국어] 경고 출력 (빈 함수 또는 레이블만 있는 함수)
    return;                            // [한국어] m_basic_blocks를 비운 채 조기 반환
  }

  unsigned bb_id = 0;                  // [한국어] 기본 블록 ID 카운터 (0부터 순차 할당)
  l = leaders.begin();                  // [한국어] 리더 목록의 첫 번째 리더로 이터레이터 설정
  i = m_instructions.begin();           // [한국어] 전체 명령어 목록의 시작으로 이터레이터 설정
  m_basic_blocks.push_back(
      new basic_block_t(bb_id++, *find_next_real_instruction(i), NULL, 1, 0));
  // [한국어] 첫 번째 기본 블록 생성: bb_id=0, 시작=첫 실제 명령어, 끝=NULL(나중에 설정),
  //         is_entry=1(엔트리 블록), is_exit=0
  ptx_instruction *last_real_inst = *(l++);  // [한국어] 첫 리더를 last_real_inst로 초기화, 다음 리더로 l 전진

  for (; i != m_instructions.end(); i++) {  // [한국어] 모든 명령어를 순회하며 각 명령어를 기본 블록에 할당
    ptx_instruction *pI = *i;               // [한국어] 현재 명령어 추출
    if (l != leaders.end() && *i == *l) {   // [한국어] 현재 명령어가 다음 리더이면 → 새 기본 블록 시작
      // found start of next basic block
      m_basic_blocks.back()->ptx_end = last_real_inst;  // [한국어] 이전 블록의 마지막 실제 명령어 확정
      if (find_next_real_instruction(i) !=
          m_instructions.end()) {  // if not bogus trailing label
        // [한국어] 마지막 레이블이 실제 명령어 뒤에 오는 "가짜 trailing 레이블"이 아닌 경우만 새 블록 생성
        m_basic_blocks.push_back(new basic_block_t(
            bb_id++, *find_next_real_instruction(i), NULL, 0, 0));
        // [한국어] 새 기본 블록 생성: 시작=다음 실제 명령어, is_entry=0, is_exit=0
        last_real_inst = *find_next_real_instruction(i);  // [한국어] 새 블록의 첫 실제 명령어를 last_real_inst로 갱신
      }
      // start search for next leader
      l++;                                 // [한국어] 리더 이터레이터를 다음 리더로 전진
    }
    pI->assign_bb(m_basic_blocks.back()); // [한국어] 현재 명령어를 현재 기본 블록에 소속시킴
    if (!pI->is_label()) last_real_inst = pI;  // [한국어] 실제 명령어이면 last_real_inst 갱신 (레이블은 무시)
  }
  m_basic_blocks.back()->ptx_end = last_real_inst;  // [한국어] 마지막 기본 블록의 끝 명령어 확정
  m_basic_blocks.push_back(
      /*exit basic block*/ new basic_block_t(bb_id, NULL, NULL, 0, 1));
  // [한국어] 특수 exit 기본 블록 추가: ptx_begin=NULL, ptx_end=NULL, is_entry=0, is_exit=1
  //         RET/EXIT 명령어의 후계자가 이 블록을 가리키게 됨
}

/*
 * [한국어]
 * function_info::print_basic_blocks - 기본 블록별 명령어 목록과 요약을 출력 (디버그)
 *
 * @return: 없음
 *
 * 두 가지 정보를 출력한다:
 *   1) 명령어별로 소속 기본 블록 ID와 명령어 내용을 출력
 *   2) 기본 블록별 요약 (bb_id, 첫 명령어 opcode, 마지막 명령어 opcode)
 * g_debug_execution >= 50일 때 do_pdom()이 호출한다. 일반 시뮬레이션에서 호출 안 됨.
 *
 * 호출 체인:
 *   function_info::do_pdom() (디버그 레벨 ≥50) → [print_basic_blocks]
 */
void function_info::print_basic_blocks() {
  printf("Printing basic blocks for function \'%s\':\n", m_name.c_str());  // [한국어] 함수명 제목 출력
  std::list<ptx_instruction *>::iterator ptx_itr;
  unsigned last_bb = 0;                           // [한국어] 이전에 출력한 기본 블록 ID (블록 전환 시 빈 줄 출력용)
  for (ptx_itr = m_instructions.begin(); ptx_itr != m_instructions.end();
       ptx_itr++) {                               // [한국어] 모든 명령어를 순서대로 순회
    if ((*ptx_itr)->get_bb()) {                   // [한국어] 기본 블록에 소속된 명령어만 출력 (exit BB 제외)
      if ((*ptx_itr)->get_bb()->bb_id != last_bb) {  // [한국어] 새 기본 블록으로 전환 시
        printf("\n");                             // [한국어] 블록 구분 빈 줄
        last_bb = (*ptx_itr)->get_bb()->bb_id;   // [한국어] 현재 블록 ID 갱신
      }
      printf("bb_%02u\t: ", (*ptx_itr)->get_bb()->bb_id);  // [한국어] 기본 블록 ID 출력 (2자리)
      (*ptx_itr)->print_insn();                 // [한국어] 명령어 내용 출력 (PC, 소스 파일:라인, 소스 텍스트)
      printf("\n");
    }
  }
  printf("\nSummary of basic blocks for \'%s\':\n", m_name.c_str());  // [한국어] 기본 블록 요약 섹션
  std::vector<basic_block_t *>::iterator bb_itr;
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {                              // [한국어] 기본 블록 벡터 순회
    printf("bb_%02u\t:", (*bb_itr)->bb_id);    // [한국어] 기본 블록 ID 출력
    if ((*bb_itr)->ptx_begin)
      printf(" first: %s\t", ((*bb_itr)->ptx_begin)->get_opcode_cstr());  // [한국어] 첫 명령어 opcode 문자열
    else
      printf(" first: NULL\t");               // [한국어] exit BB 등 시작 명령어 없는 경우
    if ((*bb_itr)->ptx_end) {
      printf(" last: %s\t", ((*bb_itr)->ptx_end)->get_opcode_cstr());    // [한국어] 마지막 명령어 opcode 문자열
    } else
      printf(" last: NULL\t");               // [한국어] exit BB 등 끝 명령어 없는 경우
    printf("\n");
  }
  printf("\n");
}

/*
 * [한국어]
 * function_info::print_basic_block_links - 기본 블록의 선행자/후계자 링크를 출력 (디버그)
 *
 * @return: 없음
 *
 * connect_basic_blocks() 수행 후 각 기본 블록의 predecessor_ids와 successor_ids를
 * 텍스트로 출력한다. CFG 엣지가 올바르게 구성되었는지 검증하는 디버그 도구.
 * g_debug_execution >= 50일 때 do_pdom()이 호출한다.
 *
 * 호출 체인:
 *   function_info::do_pdom() (디버그 레벨 ≥50) → [print_basic_block_links]
 */
void function_info::print_basic_block_links() {
  printf("Printing basic blocks links for function \'%s\':\n", m_name.c_str());  // [한국어] 함수명 제목 출력
  std::vector<basic_block_t *>::iterator bb_itr;
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {                             // [한국어] 모든 기본 블록 순회
    printf("ID: %d\t:", (*bb_itr)->bb_id);   // [한국어] 기본 블록 ID 출력
    if (!(*bb_itr)->predecessor_ids.empty()) {  // [한국어] 선행자가 있는 블록만 출력
      printf("Predecessors:");
      std::set<int>::iterator p;
      for (p = (*bb_itr)->predecessor_ids.begin();
           p != (*bb_itr)->predecessor_ids.end(); p++) {
        printf(" %d", *p);                    // [한국어] 각 선행자 블록 ID 출력
      }
      printf("\t");
    }
    if (!(*bb_itr)->successor_ids.empty()) {   // [한국어] 후계자가 있는 블록만 출력
      printf("Successors:");
      std::set<int>::iterator s;
      for (s = (*bb_itr)->successor_ids.begin();
           s != (*bb_itr)->successor_ids.end(); s++) {
        printf(" %d", *s);                    // [한국어] 각 후계자 블록 ID 출력
      }
    }
    printf("\n");
  }
}
/*
 * [한국어]
 * function_info::find_break_target - BREAK 명령어의 탈출 목적지 오퍼랜드를 도미네이터 트리로 탐색
 *
 * @p_break_insn: 탈출 목적지를 찾을 BREAK 명령어의 ptx_instruction 포인터.
 * @return: BREAKADDR 명령어의 목적지 operand_info 포인터 (레이블 이름 포함).
 *          찾지 못하면 assert(0) (예외 경로는 dead code).
 *
 * PTX의 BREAK 명령어는 loop-break 탈출을 의미하며, 탈출 대상 레이블은
 * BREAKADDR 명령어(loop 헤더 기본 블록에 존재)로 지정된다.
 * BREAK 명령어가 속한 기본 블록에서 출발하여 즉각 도미네이터(immediatedominator_id)
 * 방향으로 트리를 거슬러 올라가면서 각 블록을 역방향으로 탐색하여
 * BREAKADDR_OP를 가진 첫 번째 명령어를 반환한다.
 * 도미네이터 방향으로 탐색하는 이유: BREAKADDR는 BREAK를 지배하는 블록
 * (루프 헤더나 loop 시작 부분)에 위치하기 때문이다.
 *
 * 실행 컨텍스트: do_pdom() 내 connect_break_targets() 호출 시 (초기화 단계).
 *
 * 호출 체인:
 *   function_info::connect_break_targets() → [find_break_target]
 *   → 도미네이터 트리 탐색 → BREAKADDR_OP 반환
 */
operand_info *function_info::find_break_target(
    ptx_instruction *p_break_insn)  // find the target of a break instruction
{
  const basic_block_t *break_bb = p_break_insn->get_bb();  // [한국어] BREAK 명령어가 속한 기본 블록 가져오기
  // go through the dominator tree
  for (const basic_block_t *p_bb = break_bb; p_bb->immediatedominator_id != -1;
       p_bb = m_basic_blocks[p_bb->immediatedominator_id]) {
    // [한국어] BREAK 블록에서 출발하여 즉각 도미네이터 방향으로 트리를 거슬러 올라감
    //         immediatedominator_id == -1이면 엔트리 블록 (루트) 도달
    // reverse search through instructions in basic block for breakaddr
    // instruction
    unsigned insn_addr = p_bb->ptx_end->get_m_instr_mem_index();  // [한국어] 현재 블록의 마지막 명령어 인덱스부터 역방향 탐색
    while (insn_addr >= p_bb->ptx_begin->get_m_instr_mem_index()) {  // [한국어] 블록의 첫 명령어까지 역순 탐색
      ptx_instruction *pI = m_instr_mem[insn_addr];   // [한국어] 해당 PC의 명령어 가져오기
      insn_addr -= 1;                                  // [한국어] 이전 명령어 인덱스로 이동 (역방향)
      if (pI == NULL)
        continue;  // temporary solution for variable size instructions
      // [한국어] NULL 엔트리는 가변 크기 명령어의 두 번째 슬롯 — 건너뜀
      if (pI->get_opcode() == BREAKADDR_OP) {          // [한국어] BREAKADDR 명령어 발견
        return &(pI->dst());                           // [한국어] BREAKADDR의 목적지 오퍼랜드(레이블) 반환
      }
    }
  }

  assert(0);  // [한국어] 도미네이터 트리 전체를 탐색했으나 BREAKADDR를 못 찾음 — 구조적 오류

  // lazy fallback: just traverse backwards?
  // [한국어] 아래 코드는 assert(0) 이후 dead code — 컴파일러 경고 억제 목적으로 남겨둠
  for (int insn_addr = p_break_insn->get_m_instr_mem_index(); insn_addr >= 0;
       insn_addr--) {                                  // [한국어] fallback: BREAK에서 역방향 선형 탐색
    ptx_instruction *pI = m_instr_mem[insn_addr];
    if (pI->get_opcode() == BREAKADDR_OP) {
      return &(pI->dst());                             // [한국어] fallback에서 BREAKADDR 발견 (실제로는 도달 안 함)
    }
  }

  return NULL;  // [한국어] 컴파일러 경고 억제용 반환 (실제로는 도달 안 함)
}
/*
 * [한국어]
 * function_info::connect_basic_blocks - CFG(Control Flow Graph) 엣지 연결
 *
 * @return: 없음 (각 basic_block_t의 successor_ids, predecessor_ids 필드를 채움)
 *
 * create_basic_blocks()가 만든 기본 블록들 사이에 제어 흐름 엣지를 연결한다.
 * 각 기본 블록의 마지막 명령어(ptx_end) 종류에 따라 후계자를 결정:
 *   - RET_OP / RETP_OP / EXIT_OP: exit 기본 블록(맨 뒤)으로 엣지
 *     - 술어(predicated)이면 조건 미충족 시 다음 블록으로도 엣지
 *   - BRA_OP: 레이블 테이블(labels 맵)에서 분기 대상 블록으로 엣지
 *     - 무조건(unpredicated) BRA가 아니면 다음 블록으로도 엣지 (fall-through)
 *   - 그 외(CALL, 일반 명령어): 다음 기본 블록으로 fall-through 엣지
 * BREAK_OP의 엣지는 connect_break_targets()에서 별도 처리 (도미네이터 계산 후).
 *
 * 실행 컨텍스트: do_pdom() 초기화 단계 (싱글스레드).
 *
 * 호출 체인:
 *   function_info::do_pdom() → [connect_basic_blocks]
 *   → (이후) find_dominators() → connect_break_targets()
 */
void function_info::connect_basic_blocks()  // iterate across m_basic_blocks of
                                            // function, connecting basic blocks
                                            // together
{
  std::vector<basic_block_t *>::iterator bb_itr;          // [한국어] 기본 블록 순회 이터레이터
  std::vector<basic_block_t *>::iterator bb_target_itr;   // [한국어] 분기 대상 블록 탐색용 (현재 미사용)
  basic_block_t *exit_bb = m_basic_blocks.back();         // [한국어] 마지막 블록 = exit 기본 블록 (RET/EXIT의 후계자)

  // start from first basic block, which we know is the entry point
  bb_itr = m_basic_blocks.begin();                        // [한국어] 첫 번째(entry) 기본 블록부터 시작
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {                                         // [한국어] 모든 기본 블록 순회
    ptx_instruction *pI = (*bb_itr)->ptx_end;              // [한국어] 현재 블록의 마지막 명령어 (블록 종류 결정)
    if ((*bb_itr)->is_exit)  // reached last basic block, no successors to link
      continue;              // [한국어] exit 기본 블록은 후계자 없음 — 건너뜀
    if (pI->get_opcode() == RETP_OP || pI->get_opcode() == RET_OP ||
        pI->get_opcode() == EXIT_OP) {
      // [한국어] 반환/종료 명령어로 끝나는 블록 — exit 기본 블록으로 엣지 추가
      (*bb_itr)->successor_ids.insert(exit_bb->bb_id);        // [한국어] 이 블록의 후계자에 exit 블록 등록
      exit_bb->predecessor_ids.insert((*bb_itr)->bb_id);      // [한국어] exit 블록의 선행자에 이 블록 등록
      if (pI->has_pred()) {                                    // [한국어] 술어부(predicated) RET/EXIT인 경우
        printf("GPGPU-Sim PTX: Warning detected predicated return/exit.\n");
        // [한국어] 술어가 false이면 다음 명령어로 fall-through 가능 — 추가 엣지 필요
        // if predicated, add link to next block
        unsigned next_addr = pI->get_m_instr_mem_index() + pI->inst_size();  // [한국어] 다음 명령어의 PC 계산
        if (next_addr < m_instr_mem_size && m_instr_mem[next_addr]) {        // [한국어] 다음 명령어 존재 확인
          basic_block_t *next_bb = m_instr_mem[next_addr]->get_bb();         // [한국어] 다음 명령어가 속한 블록
          (*bb_itr)->successor_ids.insert(next_bb->bb_id);                   // [한국어] fall-through 엣지 추가
          next_bb->predecessor_ids.insert((*bb_itr)->bb_id);
        }
      }
      continue;  // [한국어] RET/EXIT 처리 완료 — 다음 블록으로
    } else if (pI->get_opcode() == BRA_OP) {
      // find successor and link that basic_block to this one
      operand_info &target = pI->dst();          // get operand, e.g. target name
      // [한국어] BRA 명령어의 목적지 오퍼랜드(레이블 이름) 가져오기
      unsigned addr = labels[target.name()];     // [한국어] 레이블 이름으로 PC 주소 조회 (labels 맵: 이름→PC)
      ptx_instruction *target_pI = m_instr_mem[addr];  // [한국어] 분기 대상 PC의 명령어 가져오기
      basic_block_t *target_bb = target_pI->get_bb();  // [한국어] 분기 대상 명령어가 속한 기본 블록
      (*bb_itr)->successor_ids.insert(target_bb->bb_id);        // [한국어] 분기 대상 블록을 후계자로 등록
      target_bb->predecessor_ids.insert((*bb_itr)->bb_id);      // [한국어] 분기 대상 블록의 선행자로 이 블록 등록
    }

    if (!(pI->get_opcode() == BRA_OP && (!pI->has_pred()))) {
      // if basic block does not end in an unpredicated branch,
      // then next basic block is also successor
      // (this is better than testing for .uni)
      // [한국어] 무조건 분기(BRA without pred)가 아니면 fall-through 엣지도 추가해야 함
      //         - 조건부 BRA: 조건 미충족 시 다음 블록으로 fall-through
      //         - CALL, 일반 명령어: 다음 블록이 항상 후계자
      //         - .uni(uniforme branch)도 이 검사를 통과하지 않으므로 fall-through 추가
      unsigned next_addr = pI->get_m_instr_mem_index() + pI->inst_size();  // [한국어] 다음 명령어 PC (현재 PC + 명령어 크기)
      basic_block_t *next_bb = m_instr_mem[next_addr]->get_bb();           // [한국어] 다음 명령어 소속 기본 블록
      (*bb_itr)->successor_ids.insert(next_bb->bb_id);                     // [한국어] fall-through 후계자 등록
      next_bb->predecessor_ids.insert((*bb_itr)->bb_id);                   // [한국어] fall-through 선행자 등록
    } else
      assert(pI->get_opcode() == BRA_OP);  // [한국어] 불변 조건 검증 — 여기 도달하면 반드시 BRA_OP
  }
}
/*
 * [한국어]
 * function_info::connect_break_targets - BREAK 명령어의 CFG 엣지를 올바른 목적지로 재연결
 *
 * @return: true이면 이번 호출에서 엣지가 변경됨 (do_pdom이 재시도해야 함),
 *          false이면 변경 없음 (안정 상태 도달).
 *
 * connect_basic_blocks()는 BREAK 명령어의 목적지를 알 수 없어 임시로 연결한다.
 * do_pdom()에서 도미네이터를 계산한 후 이 함수를 호출하면, find_break_target()으로
 * BREAKADDR 명령어를 찾아 BREAK의 실제 후계자를 재연결한다.
 * 연결이 변경되면 도미네이터가 달라질 수 있으므로 do_pdom()은 안정될 때까지
 * find_dominators() + connect_break_targets()를 반복 호출한다 (고정점 반복).
 * 술어부(predicated) BREAK는 조건 미충족 시 다음 블록으로도 fall-through 엣지를 추가한다.
 *
 * 실행 컨텍스트: do_pdom() 내 반복 루프 (싱글스레드).
 *
 * 호출 체인:
 *   function_info::do_pdom() (반복) → [connect_break_targets]
 *   → find_break_target() → 엣지 재연결
 */
bool function_info::connect_break_targets()  // connecting break instructions
                                             // with proper targets
{
  std::vector<basic_block_t *>::iterator bb_itr;       // [한국어] 기본 블록 순회 이터레이터
  std::vector<basic_block_t *>::iterator bb_target_itr; // [한국어] 대상 블록 탐색용 (현재 미사용)
  bool modified = false;                               // [한국어] 이번 호출에서 엣지 변경 여부 추적

  // start from first basic block, which we know is the entry point
  bb_itr = m_basic_blocks.begin();
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {                                     // [한국어] 모든 기본 블록 순회
    basic_block_t *p_bb = *bb_itr;                    // [한국어] 현재 기본 블록 포인터
    ptx_instruction *pI = p_bb->ptx_end;              // [한국어] 블록의 마지막 명령어
    if (p_bb->is_exit)  // reached last basic block, no successors to link
      continue;         // [한국어] exit 블록은 후계자 없음 — 건너뜀
    if (pI->get_opcode() == BREAK_OP) {               // [한국어] BREAK 명령어로 끝나는 블록만 처리
      // backup existing successor_ids for stability check
      std::set<int> orig_successor_ids = p_bb->successor_ids;
      // [한국어] 변경 전 후계자 집합 백업 — 이번 재연결이 이전과 다른지 비교용

      // erase the previous linkage with old successors
      for (std::set<int>::iterator succ_ids = p_bb->successor_ids.begin();
           succ_ids != p_bb->successor_ids.end(); ++succ_ids) {  // [한국어] 기존 후계자 엣지 모두 제거
        basic_block_t *successor_bb = m_basic_blocks[*succ_ids];
        successor_bb->predecessor_ids.erase(p_bb->bb_id);  // [한국어] 기존 후계자의 선행자 목록에서 이 블록 제거
      }
      p_bb->successor_ids.clear();  // [한국어] 이 블록의 후계자 집합 초기화 (새로 채울 준비)

      // find successor and link that basic_block to this one
      // successor of a break is set by an preceeding breakaddr instruction
      operand_info *target = find_break_target(pI);       // [한국어] 도미네이터 트리를 거슬러 BREAKADDR 오퍼랜드 탐색
      unsigned addr = labels[target->name()];              // [한국어] BREAKADDR 레이블 이름으로 PC 주소 조회
      ptx_instruction *target_pI = m_instr_mem[addr];     // [한국어] 분기 대상 PC의 명령어
      basic_block_t *target_bb = target_pI->get_bb();     // [한국어] 분기 대상 명령어가 속한 기본 블록
      p_bb->successor_ids.insert(target_bb->bb_id);       // [한국어] BREAK의 실제 후계자(루프 탈출 대상 블록) 등록
      target_bb->predecessor_ids.insert(p_bb->bb_id);     // [한국어] 대상 블록의 선행자에 BREAK 블록 등록

      if (pI->has_pred()) {                               // [한국어] 술어부(predicated) BREAK: 조건 미충족 시 fall-through
        // predicated break - add link to next basic block
        unsigned next_addr = pI->get_m_instr_mem_index() + pI->inst_size();  // [한국어] BREAK 다음 명령어 PC
        basic_block_t *next_bb = m_instr_mem[next_addr]->get_bb();           // [한국어] 다음 명령어 소속 블록
        p_bb->successor_ids.insert(next_bb->bb_id);                          // [한국어] fall-through 후계자 등록
        next_bb->predecessor_ids.insert(p_bb->bb_id);                        // [한국어] fall-through 선행자 등록
      }

      modified = modified || (orig_successor_ids != p_bb->successor_ids);
      // [한국어] 이번에 설정한 후계자가 이전과 다르면 modified=true (도미네이터 재계산 필요)
    }
  }

  return modified;  // [한국어] 변경 있으면 true 반환 → do_pdom() 반복 루프가 재시도
}
/*
 * [한국어]
 * function_info::do_pdom - CFG 분석 파이프라인 전체 실행 (PTX 어셈블 후 준비 완료 단계)
 *
 * @return: 없음 (m_assembled=true로 설정하고 완료)
 *
 * ptx_assemble()이 PC 할당을 완료한 뒤 호출되는 CFG 분석 파이프라인이다.
 * 다음 단계를 순서대로 수행한다:
 *   1) create_basic_blocks(): 리더 기반 기본 블록 식별
 *   2) connect_basic_blocks(): CFG 엣지 연결 (BRA/RET/EXIT/CALL 처리)
 *   3) 고정점 반복: find_dominators() → find_idominators() → connect_break_targets()
 *      BREAK 명령어의 대상이 도미네이터에 의존하므로, 안정될 때까지 반복
 *   4) (디버그) CFG 출력
 *   5) find_postdominators(): 역방향 CFG에서 후위지배자 집합 계산
 *   6) find_ipostdominators(): 즉각 후위지배자(ipdom) 계산 — SIMT 재수렴점
 *   7) (디버그) 후위지배자 출력
 *   8) 모든 명령어에 pre_decode() 호출 — 타이밍 시뮬레이터용 분기 대상 PC 캐시
 *   9) m_assembled = true 설정 → 이후 시뮬레이션 실행 가능 상태
 *
 * gpgpusim.config의 g_debug_execution 값으로 출력 레벨 제어:
 *   ≥2: 지배자 출력, ≥50: 기본 블록/링크/DOT/후위지배자 출력
 *
 * 실행 컨텍스트: 시뮬레이션 초기화 단계 (싱글스레드), function_info::ptx_assemble() 내부.
 *
 * 호출 체인:
 *   function_info::ptx_assemble() → [do_pdom]
 *   → create_basic_blocks → connect_basic_blocks → find_dominators → ...
 *   → find_ipostdominators → pre_decode (모든 명령어)
 */
void function_info::do_pdom() {
  create_basic_blocks();          // [한국어] 단계 1: 리더 기반 기본 블록 식별 및 생성
  connect_basic_blocks();         // [한국어] 단계 2: CFG 엣지 연결 (BRA/RET/EXIT fall-through)
  bool modified = false;          // [한국어] BREAK 타겟 재연결 고정점 반복 종료 조건
  do {
    find_dominators();            // [한국어] 단계 3a: Muchnick 7.14 지배자 집합 계산
    find_idominators();           // [한국어] 단계 3b: Muchnick 7.15 즉각 지배자 계산 (BREAK 탐색에 필요)
    modified = connect_break_targets();  // [한국어] 단계 3c: 도미네이터 사용해 BREAK 엣지 재연결
  } while (modified == true);     // [한국어] 고정점 도달(변경 없음)까지 반복

  if (g_debug_execution >= 50) {  // [한국어] 상세 디버그 레벨(≥50): 기본 블록 정보 출력
    print_basic_blocks();         // [한국어] 기본 블록별 명령어 목록 출력
    print_basic_block_links();    // [한국어] CFG 선행자/후계자 링크 출력
    print_basic_block_dot();      // [한국어] GraphViz DOT 형식 CFG 출력
  }
  if (g_debug_execution >= 2) {   // [한국어] 기본 디버그 레벨(≥2): 지배자 집합 출력
    print_dominators();           // [한국어] 각 블록의 지배자 집합 출력
  }
  find_postdominators();          // [한국어] 단계 5: 역방향 CFG에서 후위지배자 집합 계산
  find_ipostdominators();         // [한국어] 단계 6: 즉각 후위지배자(ipdom) 계산 — SIMT 재수렴점 결정
  if (g_debug_execution >= 50) {  // [한국어] 상세 디버그: 후위지배자 정보 출력
    print_postdominators();       // [한국어] 각 블록의 후위지배자 집합 출력
    print_ipostdominators();      // [한국어] 각 블록의 즉각 후위지배자 ID 출력
  }
  printf("GPGPU-Sim PTX: pre-decoding instructions for \'%s\'...\n",
         m_name.c_str());         // [한국어] pre-decode 시작 알림 (커널 이름 포함)
  for (unsigned ii = 0; ii < m_n;
       ii += m_instr_mem[ii]->inst_size()) {  // handle branch instructions
    // [한국어] 모든 명령어 슬롯을 inst_size() 단위로 순회 (가변 크기 명령어 지원)
    ptx_instruction *pI = m_instr_mem[ii];   // [한국어] 해당 PC의 명령어 가져오기
    pI->pre_decode();  // [한국어] 명령어별 사전 디코딩: 분기 대상 PC 캐시, operand 타입 검증 등
  }
  printf("GPGPU-Sim PTX: ... done pre-decoding instructions for \'%s\'.\n",
         m_name.c_str());         // [한국어] pre-decode 완료 알림
  fflush(stdout);                 // [한국어] 출력 버퍼 강제 플러시 — 로그 손실 방지
  m_assembled = true;             // [한국어] 어셈블/CFG 분석 완료 플래그 설정 — 이후 시뮬레이션 실행 허가
}
/*
 * [한국어]
 * intersect - 집합 A를 A∩B 교집합으로 인플레이스 갱신
 *
 * @A: [입/출력] 교집합 결과가 저장될 집합. B에 없는 원소가 제거됨.
 * @B: 교집합의 다른 피연산자. 변경되지 않음.
 * @return: 없음
 *
 * 지배자/후위지배자 알고리즘(find_dominators, find_postdominators)에서
 * Dom(n) = {n} ∪ ∩Dom(predecessor of n) 계산 시 교집합을 수행하는 보조 함수.
 * 반복 중 A에서 원소를 삭제하므로, 삭제 시 이터레이터가 무효화되지 않도록
 * a_next를 미리 저장한 후 삭제한다.
 *
 * 호출 체인:
 *   function_info::find_dominators() / find_postdominators() → [intersect]
 */
void intersect(std::set<int> &A, const std::set<int> &B) {
  // return intersection of A and B in A
  // [한국어] A를 순회하면서 B에 없는 원소를 A에서 제거 → 결과가 A∩B
  for (std::set<int>::iterator a = A.begin(); a != A.end();) {
    std::set<int>::iterator a_next = a;  // [한국어] 삭제 후 이터레이터 무효화 방지를 위해 다음 이터레이터 미리 저장
    a_next++;
    if (B.find(*a) == B.end()) {         // [한국어] *a가 B에 없으면 교집합에 포함 안 됨
      A.erase(*a);                       // [한국어] A에서 해당 원소 제거
      a = a_next;                        // [한국어] 삭제된 이터레이터 대신 미리 저장한 다음 이터레이터 사용
    } else
      a++;                               // [한국어] B에도 있으면 교집합 원소이므로 유지, 다음으로
  }
}

/*
 * [한국어]
 * is_equal - 두 정수 집합이 동일한지 비교
 *
 * @A: 비교할 첫 번째 집합.
 * @B: 비교할 두 번째 집합.
 * @return: 두 집합이 동일하면 true, 다르면 false.
 *
 * find_dominators()의 고정점 반복에서 "변경이 있었는가"를 판단하는 데 사용된다.
 * 크기가 다르면 즉시 false, 같으면 B의 모든 원소가 A에 있는지 확인한다.
 *
 * 호출 체인:
 *   function_info::find_dominators() / find_postdominators() → [is_equal]
 */
bool is_equal(const std::set<int> &A, const std::set<int> &B) {
  if (A.size() != B.size()) return false;  // [한국어] 크기가 다르면 즉시 false (빠른 경로)
  for (std::set<int>::iterator b = B.begin(); b != B.end(); b++)
    if (A.find(*b) == A.end()) return false;  // [한국어] B의 원소가 A에 없으면 서로 다른 집합
  return true;  // [한국어] 모든 원소가 A에도 있고 크기도 같으면 동일
}

/*
 * [한국어]
 * print_set - 정수 집합의 모든 원소를 공백 구분으로 출력 (디버그)
 *
 * @A: 출력할 정수 집합.
 * @return: 없음
 *
 * 지배자/후위지배자 집합의 내용을 디버그 출력할 때 사용하는 단순 유틸리티.
 * print_dominators() / print_postdominators() 등에서 활용된다.
 *
 * 호출 체인:
 *   (디버그 목적 직접 호출) → [print_set] → printf
 */
void print_set(const std::set<int> &A) {
  std::set<int>::iterator a;
  for (a = A.begin(); a != A.end(); a++) {  // [한국어] 집합의 모든 원소를 오름차순 순회 (set은 정렬됨)
    printf("%d ", (*a));                    // [한국어] 원소 값을 공백으로 구분하여 출력
  }
  printf("\n");  // [한국어] 줄바꿈
}

/*
 * [한국어]
 * function_info::find_dominators - 반복 데이터플로우 알고리즘으로 지배자 집합 계산
 *
 * @return: 없음 (각 basic_block_t::dominator_ids 필드에 결과 저장)
 *
 * Muchnick "Advanced Compiler Design & Implementation" Fig 7.14 알고리즘 구현.
 * 지배자(dominator): 블록 n의 지배자 집합 Dom(n)은 CFG에서 엔트리에서 n까지의
 * 모든 경로가 반드시 통과하는 블록들의 집합이다.
 *
 * 알고리즘:
 *   초기화: Dom(entry) = {entry}, Dom(n) = 모든 블록 집합 (n ≠ entry)
 *   반복: Dom(n) = {n} ∪ ∩{Dom(p) | p ∈ predecessors(n)}
 *         변화가 없을 때까지 반복 (고정점)
 *   마무리: 선행자가 없는 블록(도달 불가 블록)의 지배자 집합 초기화
 *
 * 결과는 find_idominators()와 connect_break_targets()에서 사용된다.
 * find_ipostdominators() 결과는 SIMT 스택 재수렴점 결정에 사용된다.
 *
 * 실행 컨텍스트: do_pdom() 내 반복 루프 (싱글스레드).
 *
 * 호출 체인:
 *   function_info::do_pdom() → [find_dominators] → intersect() / is_equal()
 *   → find_idominators() → connect_break_targets()
 */
void function_info::find_dominators() {
  // find dominators using algorithm of Muchnick's Adv. Compiler Design &
  // Implemmntation Fig 7.14
  printf("GPGPU-Sim PTX: Finding dominators for \'%s\'...\n", m_name.c_str());  // [한국어] 진행 상황 로그
  fflush(stdout);                         // [한국어] 버퍼 강제 플러시 (장시간 반복 중 로그 확인용)
  assert(m_basic_blocks.size() >= 2);  // must have a distinquished entry block
  // [한국어] 최소 엔트리 + exit 블록 2개 필요
  std::vector<basic_block_t *>::iterator bb_itr = m_basic_blocks.begin();
  (*bb_itr)->dominator_ids.insert(
      (*bb_itr)->bb_id);  // the only dominator of the entry block is the entry
  // [한국어] 엔트리 블록의 지배자는 자기 자신뿐 — Dom(entry) = {entry}
  // copy all basic blocks to all dominator lists EXCEPT for the entry block
  for (++bb_itr; bb_itr != m_basic_blocks.end(); bb_itr++) {  // [한국어] 엔트리 제외 모든 블록 초기화
    for (unsigned i = 0; i < m_basic_blocks.size(); i++)
      (*bb_itr)->dominator_ids.insert(i);  // [한국어] 초기값: Dom(n) = 모든 블록 집합 (가장 보수적인 시작점)
  }
  bool change = true;                     // [한국어] 고정점 반복 종료 조건 플래그
  while (change) {                        // [한국어] 변화가 없을 때까지 반복
    change = false;                       // [한국어] 이번 순회에서 변화 없다고 가정
    for (int h = 1 /*skip entry*/; h < m_basic_blocks.size(); ++h) {
      // [한국어] 엔트리 블록(h=0) 제외하고 모든 블록에 대해 Dom 갱신
      assert(m_basic_blocks[h]->bb_id == (unsigned)h);  // [한국어] 블록 벡터 인덱스 = bb_id 일관성 검증
      std::set<int> T;
      for (unsigned i = 0; i < m_basic_blocks.size(); i++) T.insert(i);
      // [한국어] T를 모든 블록 집합으로 초기화 (교집합 연산의 시작점)
      for (std::set<int>::iterator s =
               m_basic_blocks[h]->predecessor_ids.begin();
           s != m_basic_blocks[h]->predecessor_ids.end(); s++)
        intersect(T, m_basic_blocks[*s]->dominator_ids);
      // [한국어] 모든 선행자의 Dom 집합과 교집합 계산: T = ∩Dom(predecessors of h)
      T.insert(h);  // [한국어] Dom(h) = {h} ∪ T — 자기 자신은 항상 자신의 지배자
      if (!is_equal(T, m_basic_blocks[h]->dominator_ids)) {  // [한국어] 이전 Dom(h)와 다르면 갱신
        change = true;                                        // [한국어] 변화 발생 — 다음 순회도 필요
        m_basic_blocks[h]->dominator_ids = T;                // [한국어] Dom(h) 갱신
      }
    }
  }
  // clean the basic block of dominators of it has no predecessors -- except for
  // entry block
  // [한국어] 선행자가 없는 블록(도달 불가 블록)의 지배자 집합 정리 — 엔트리 제외
  bb_itr = m_basic_blocks.begin();
  for (++bb_itr; bb_itr != m_basic_blocks.end(); bb_itr++) {
    if ((*bb_itr)->predecessor_ids.empty()) (*bb_itr)->dominator_ids.clear();
    // [한국어] 선행자 없는 블록은 도달 불가 — 지배자 집합 비워서 오류 전파 방지
  }
}

/*
 * [한국어]
 * function_info::find_postdominators - 역방향 데이터플로우로 후위지배자 집합 계산
 *
 * @return: 없음 (각 basic_block_t::postdominator_ids 필드에 결과 저장)
 *
 * find_dominators()와 동일한 Muchnick Fig 7.14 알고리즘이지만, CFG를 역방향으로
 * (후계자 방향으로) 처리하여 후위지배자(post-dominator)를 계산한다.
 * 후위지배자: 블록 n의 후위지배자 집합 PDom(n)은 CFG에서 n에서 exit까지의
 * 모든 경로가 반드시 통과하는 블록들의 집합이다.
 *
 * 알고리즘:
 *   초기화: PDom(exit) = {exit}, PDom(n) = 모든 블록 집합 (n ≠ exit)
 *   반복: PDom(n) = {n} ∪ ∩{PDom(s) | s ∈ successors(n)}
 *         선행자 → 후계자 방향으로 교집합 계산
 *
 * 결과는 find_ipostdominators()에서 즉각 후위지배자(ipdom) 계산에 사용되며,
 * ipdom은 SIMT 스택의 재수렴점(reconvergence point) 결정에 최종 사용된다.
 *
 * 호출 체인:
 *   function_info::do_pdom() → [find_postdominators] → find_ipostdominators()
 */
void function_info::find_postdominators() {
  // find postdominators using algorithm of Muchnick's Adv. Compiler Design &
  // Implemmntation Fig 7.14
  printf("GPGPU-Sim PTX: Finding postdominators for \'%s\'...\n",
         m_name.c_str());    // [한국어] 진행 상황 로그
  fflush(stdout);            // [한국어] 버퍼 강제 플러시
  assert(m_basic_blocks.size() >= 2);  // must have a distinquished exit block
  // [한국어] 최소 엔트리 + exit 블록 2개 필요
  std::vector<basic_block_t *>::reverse_iterator bb_itr =
      m_basic_blocks.rbegin();  // [한국어] 역방향 이터레이터 — exit 블록(맨 뒤)부터 시작
  (*bb_itr)->postdominator_ids.insert(
      (*bb_itr)
          ->bb_id);  // the only postdominator of the exit block is the exit
  // [한국어] PDom(exit) = {exit} — exit 블록의 후위지배자는 자기 자신뿐
  for (++bb_itr; bb_itr != m_basic_blocks.rend();
       bb_itr++) {  // copy all basic blocks to all postdominator lists EXCEPT
                    // for the exit block
    // [한국어] exit 제외 모든 블록 초기화: PDom(n) = 모든 블록 집합
    for (unsigned i = 0; i < m_basic_blocks.size(); i++)
      (*bb_itr)->postdominator_ids.insert(i);
  }
  bool change = true;   // [한국어] 고정점 반복 종료 조건
  while (change) {      // [한국어] 변화가 없을 때까지 반복
    change = false;     // [한국어] 이번 순회에서 변화 없다고 가정
    for (int h = m_basic_blocks.size() - 2 /*skip exit*/; h >= 0; --h) {
      // [한국어] exit 블록(마지막) 제외하고 역방향으로 모든 블록 처리
      assert(m_basic_blocks[h]->bb_id == (unsigned)h);  // [한국어] 인덱스 일관성 검증
      std::set<int> T;
      for (unsigned i = 0; i < m_basic_blocks.size(); i++) T.insert(i);
      // [한국어] T를 모든 블록 집합으로 초기화
      for (std::set<int>::iterator s = m_basic_blocks[h]->successor_ids.begin();
           s != m_basic_blocks[h]->successor_ids.end(); s++)
        intersect(T, m_basic_blocks[*s]->postdominator_ids);
      // [한국어] 모든 후계자의 PDom 교집합 계산: T = ∩PDom(successors of h)
      T.insert(h);  // [한국어] PDom(h) = {h} ∪ T — 자기 자신은 항상 자신의 후위지배자
      if (!is_equal(T, m_basic_blocks[h]->postdominator_ids)) {  // [한국어] 이전 PDom과 다르면 갱신
        change = true;                                            // [한국어] 변화 발생 — 다음 순회 필요
        m_basic_blocks[h]->postdominator_ids = T;                // [한국어] PDom(h) 갱신
      }
    }
  }
}

/*
 * [한국어]
 * function_info::find_ipostdominators - 즉각 후위지배자(ipdom) 계산
 *
 * @return: 없음 (각 basic_block_t::immediatepostdominator_id 필드에 결과 저장)
 *
 * Muchnick Fig 7.15 알고리즘으로 각 기본 블록의 즉각 후위지배자(immediate
 * post-dominator, ipdom)를 찾는다. ipdom(n)은 PDom(n)-{n} 중에서 다른 어떤
 * 후위지배자에도 후위지배되지 않는 유일한 원소이다.
 *
 * 알고리즘 (Tmp 집합 정제법):
 *   초기화: Tmp(n) = PDom(n) - {n}  (자기 자신 제외)
 *   정제: 모든 n에 대해, Tmp(n)의 원소 s와 t를 비교하여,
 *         t가 PDom(s)에 있으면 t를 Tmp(n)에서 제거
 *         (s가 t의 후위지배자이면 t는 즉각 후위지배자가 될 수 없음)
 *   결과: |Tmp(n)| = 1이면 그 원소가 ipdom(n)
 *
 * 이 결과가 SIMT 스택의 재수렴점(reconvergence point)으로 사용된다:
 * BRA 명령어를 만난 워프가 분기되면, ipdom(BRA 블록)의 첫 명령어에서 재수렴한다.
 * get_reconvergence_pairs()가 이 ipdom을 수집하여 타이밍 시뮬레이터에 제공한다.
 *
 * 실행 컨텍스트: do_pdom() 내부 (싱글스레드, 초기화 단계).
 *
 * 호출 체인:
 *   function_info::do_pdom() → [find_ipostdominators]
 *   결과 → get_reconvergence_pairs() → shader.cc SIMT 스택
 */
void function_info::find_ipostdominators() {
  // find immediate postdominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.15
  printf("GPGPU-Sim PTX: Finding immediate postdominators for \'%s\'...\n",
         m_name.c_str());   // [한국어] 진행 상황 로그
  fflush(stdout);           // [한국어] 버퍼 강제 플러시
  assert(m_basic_blocks.size() >= 2);  // must have a distinquished exit block
  // [한국어] 최소 2개 블록 (엔트리 + exit) 필요
  for (unsigned i = 0; i < m_basic_blocks.size();
       i++) {  // initialize Tmp(n) to all pdoms of n except for n
    // [한국어] 각 블록의 Tmp 집합을 PDom(n) - {n} 으로 초기화
    m_basic_blocks[i]->Tmp_ids = m_basic_blocks[i]->postdominator_ids;  // [한국어] PDom(n) 복사
    assert(m_basic_blocks[i]->bb_id == i);   // [한국어] 인덱스 = bb_id 일관성 검증
    m_basic_blocks[i]->Tmp_ids.erase(i);     // [한국어] 자기 자신 제거: Tmp(n) = PDom(n) - {n}
  }
  for (int n = m_basic_blocks.size() - 2; n >= 0; --n) {
    // point iterator to basic block before the exit
    // [한국어] exit 블록 제외하고 역방향으로 모든 블록 처리
    for (std::set<int>::iterator s = m_basic_blocks[n]->Tmp_ids.begin();
         s != m_basic_blocks[n]->Tmp_ids.end(); s++) {  // [한국어] Tmp(n)의 각 후위지배자 s 순회
      int bb_s = *s;              // [한국어] 후위지배자 블록 ID
      for (std::set<int>::iterator t = m_basic_blocks[n]->Tmp_ids.begin();
           t != m_basic_blocks[n]->Tmp_ids.end();) {  // [한국어] s와 비교할 다른 후위지배자 t 순회
        std::set<int>::iterator t_next = t;
        t_next++;  // might erase thing pointed to be t, invalidating iterator t
        // [한국어] t가 가리키는 원소를 erase할 수 있으므로 다음 이터레이터 미리 저장
        if (*s == *t) {           // [한국어] s == t이면 자기 자신 비교 — 건너뜀
          t = t_next;
          continue;
        }
        int bb_t = *t;            // [한국어] 비교 대상 블록 ID
        if (m_basic_blocks[bb_s]->postdominator_ids.find(bb_t) !=
            m_basic_blocks[bb_s]->postdominator_ids.end())
          m_basic_blocks[n]->Tmp_ids.erase(bb_t);
        // [한국어] s가 t를 후위지배하면(t ∈ PDom(s)), t는 n의 즉각 후위지배자가 될 수 없음
        //         → Tmp(n)에서 t 제거 (s가 더 가까운 후위지배자)
        t = t_next;               // [한국어] 다음 t로 이동 (erase 후 이터레이터 무효화 방지)
      }
    }
  }
  unsigned num_ipdoms = 0;        // [한국어] ipdom이 확정된 블록 수 카운터
  for (int n = m_basic_blocks.size() - 1; n >= 0; --n) {  // [한국어] 모든 블록 순회하며 ipdom 확정
    assert(m_basic_blocks[n]->Tmp_ids.size() <= 1);
    // if the above assert fails we have an error in either postdominator
    // computation, the flow graph does not have a unique exit, or some other
    // error
    // [한국어] Tmp_ids.size() > 1이면 알고리즘 오류 또는 exit 블록이 유일하지 않음
    if (!m_basic_blocks[n]->Tmp_ids.empty()) {
      m_basic_blocks[n]->immediatepostdominator_id =
          *m_basic_blocks[n]->Tmp_ids.begin();  // [한국어] Tmp_ids의 유일한 원소를 ipdom으로 설정
      num_ipdoms++;                              // [한국어] ipdom 확정 블록 수 증가
    }
    // [한국어] Tmp_ids가 비어 있으면 ipdom 없음 (exit 블록의 경우)
  }
  assert(num_ipdoms == m_basic_blocks.size() - 1);
  // the exit node does not have an immediate post dominator, but everyone else
  // should
  // [한국어] exit 블록을 제외한 모든 블록은 ipdom을 가져야 함 (구조 검증)
}

/*
 * [한국어]
 * function_info::find_idominators - 즉각 지배자(idom) 계산
 *
 * @return: 없음 (각 basic_block_t::immediatedominator_id 필드에 결과 저장)
 *
 * find_ipostdominators()와 대칭적인 알고리즘으로, PDom 대신 Dom 집합을 사용하여
 * 즉각 지배자(immediate dominator, idom)를 계산한다. idom(n)은 Dom(n)-{n} 중에서
 * 다른 어떤 지배자에도 지배되지 않는 유일한 원소이다.
 *
 * 이 결과는 find_break_target()에서 BREAK 명령어의 탈출 대상을 찾을 때
 * 도미네이터 트리를 거슬러 올라가는 데 사용된다.
 *
 * 알고리즘: find_ipostdominators()와 동일 구조, Dom/idom으로 대체.
 *
 * 호출 체인:
 *   function_info::do_pdom() → [find_idominators]
 *   결과 → find_break_target() → connect_break_targets()
 */
void function_info::find_idominators() {
  // find immediate dominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.15
  printf("GPGPU-Sim PTX: Finding immediate dominators for \'%s\'...\n",
         m_name.c_str());  // [한국어] 진행 상황 로그
  fflush(stdout);          // [한국어] 버퍼 강제 플러시
  assert(m_basic_blocks.size() >= 2);  // must have a distinquished entry block
  // [한국어] 최소 2개 블록 필요
  for (unsigned i = 0; i < m_basic_blocks.size();
       i++) {  // initialize Tmp(n) to all doms of n except for n
    // [한국어] 각 블록의 Tmp 집합을 Dom(n) - {n} 으로 초기화
    m_basic_blocks[i]->Tmp_ids = m_basic_blocks[i]->dominator_ids;  // [한국어] Dom(n) 복사
    assert(m_basic_blocks[i]->bb_id == i);  // [한국어] 인덱스 일관성 검증
    m_basic_blocks[i]->Tmp_ids.erase(i);    // [한국어] 자기 자신 제거: Tmp(n) = Dom(n) - {n}
  }
  for (int n = 0; n < m_basic_blocks.size(); ++n) {
    // point iterator to basic block before the exit
    // [한국어] 순방향으로 모든 블록 처리 (idom은 엔트리 방향이므로 순방향)
    for (std::set<int>::iterator s = m_basic_blocks[n]->Tmp_ids.begin();
         s != m_basic_blocks[n]->Tmp_ids.end(); s++) {  // [한국어] Tmp(n)의 각 지배자 s 순회
      int bb_s = *s;             // [한국어] 지배자 블록 ID
      for (std::set<int>::iterator t = m_basic_blocks[n]->Tmp_ids.begin();
           t != m_basic_blocks[n]->Tmp_ids.end();) {
        std::set<int>::iterator t_next = t;
        t_next++;  // might erase thing pointed to be t, invalidating iterator t
        // [한국어] erase 시 이터레이터 무효화 방지
        if (*s == *t) {          // [한국어] 자기 비교 건너뜀
          t = t_next;
          continue;
        }
        int bb_t = *t;           // [한국어] 비교 대상 지배자 블록 ID
        if (m_basic_blocks[bb_s]->dominator_ids.find(bb_t) !=
            m_basic_blocks[bb_s]->dominator_ids.end())
          m_basic_blocks[n]->Tmp_ids.erase(bb_t);
        // [한국어] s가 t를 지배하면(t ∈ Dom(s)), t는 n의 즉각 지배자가 될 수 없음
        //         → Tmp(n)에서 t 제거 (s가 더 가까운 지배자)
        t = t_next;              // [한국어] 다음 t로 이동
      }
    }
  }
  unsigned num_idoms = 0;        // [한국어] idom 확정 블록 수
  unsigned num_nopred = 0;       // [한국어] 선행자 없는(도달 불가) 블록 수
  for (int n = 0; n < m_basic_blocks.size(); ++n) {
    // assert( m_basic_blocks[n]->Tmp_ids.size() <= 1 );
    // if the above assert fails we have an error in either dominator
    // computation, the flow graph does not have a unique entry, or some other
    // error
    // [한국어] 주석처리된 assert: 도달 불가 블록이 여러 idom 후보를 가질 수 있어서 비활성화
    if (!m_basic_blocks[n]->Tmp_ids.empty()) {
      m_basic_blocks[n]->immediatedominator_id =
          *m_basic_blocks[n]->Tmp_ids.begin();  // [한국어] Tmp_ids의 유일한 원소를 idom으로 설정
      num_idoms++;                               // [한국어] idom 확정 블록 수 증가
    } else if (m_basic_blocks[n]->predecessor_ids.empty()) {
      num_nopred += 1;  // [한국어] 선행자 없는 블록 카운트 (엔트리 블록 또는 도달 불가 블록)
    }
  }
  assert(num_idoms == m_basic_blocks.size() - num_nopred);
  // the entry node does not have an immediate dominator, but everyone else
  // should
  // [한국어] 선행자 없는 블록을 제외한 모든 블록은 idom을 가져야 함 (구조 검증)
}

/*
 * [한국어]
 * function_info::print_dominators - 각 기본 블록의 지배자 집합을 출력 (디버그)
 *
 * @return: 없음
 * g_debug_execution >= 2일 때 do_pdom()이 호출한다.
 *
 * 호출 체인: function_info::do_pdom() (디버그 ≥2) → [print_dominators]
 */
void function_info::print_dominators() {
  printf("Printing dominators for function \'%s\':\n", m_name.c_str());  // [한국어] 함수명 제목
  std::vector<int>::iterator bb_itr;   // [한국어] (미사용 선언 — 레거시)
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {  // [한국어] 모든 블록 순회
    printf("ID: %d\t:", i);            // [한국어] 블록 ID 출력
    for (std::set<int>::iterator j = m_basic_blocks[i]->dominator_ids.begin();
         j != m_basic_blocks[i]->dominator_ids.end(); j++)
      printf(" %d", *j);              // [한국어] 각 지배자 블록 ID 공백 구분 출력
    printf("\n");
  }
}

/*
 * [한국어]
 * function_info::print_postdominators - 각 기본 블록의 후위지배자 집합을 출력 (디버그)
 *
 * @return: 없음
 * g_debug_execution >= 50일 때 do_pdom()이 호출한다.
 *
 * 호출 체인: function_info::do_pdom() (디버그 ≥50) → [print_postdominators]
 */
void function_info::print_postdominators() {
  printf("Printing postdominators for function \'%s\':\n", m_name.c_str());  // [한국어] 함수명 제목
  std::vector<int>::iterator bb_itr;   // [한국어] (미사용 선언 — 레거시)
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {  // [한국어] 모든 블록 순회
    printf("ID: %d\t:", i);            // [한국어] 블록 ID 출력
    for (std::set<int>::iterator j =
             m_basic_blocks[i]->postdominator_ids.begin();
         j != m_basic_blocks[i]->postdominator_ids.end(); j++)
      printf(" %d", *j);              // [한국어] 각 후위지배자 블록 ID 출력
    printf("\n");
  }
}

/*
 * [한국어]
 * function_info::print_ipostdominators - 각 기본 블록의 즉각 후위지배자 ID를 출력 (디버그)
 *
 * @return: 없음
 * g_debug_execution >= 50일 때 do_pdom()이 호출한다.
 *
 * 호출 체인: function_info::do_pdom() (디버그 ≥50) → [print_ipostdominators]
 */
void function_info::print_ipostdominators() {
  printf("Printing immediate postdominators for function \'%s\':\n",
         m_name.c_str());              // [한국어] 함수명 제목
  std::vector<int>::iterator bb_itr;   // [한국어] (미사용 선언 — 레거시)
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {  // [한국어] 모든 블록 순회
    printf("ID: %d\t:", i);            // [한국어] 블록 ID 출력
    printf("%d\n", m_basic_blocks[i]->immediatepostdominator_id);
    // [한국어] ipdom ID 출력 (-1이면 exit 블록처럼 ipdom 없음)
  }
}

/*
 * [한국어]
 * function_info::print_idominators - 각 기본 블록의 즉각 지배자 ID를 출력 (디버그)
 *
 * @return: 없음
 * (현재 do_pdom()에서 직접 호출되지는 않지만 디버그 목적으로 제공)
 *
 * 호출 체인: (디버그 목적 직접 호출) → [print_idominators]
 */
void function_info::print_idominators() {
  printf("Printing immediate dominators for function \'%s\':\n",
         m_name.c_str());              // [한국어] 함수명 제목
  std::vector<int>::iterator bb_itr;   // [한국어] (미사용 선언 — 레거시)
  for (unsigned i = 0; i < m_basic_blocks.size(); i++) {  // [한국어] 모든 블록 순회
    printf("ID: %d\t:", i);            // [한국어] 블록 ID 출력
    printf("%d\n", m_basic_blocks[i]->immediatedominator_id);
    // [한국어] idom ID 출력 (-1이면 엔트리 블록처럼 idom 없음)
  }
}

/*
 * [한국어]
 * function_info::get_num_reconvergence_pairs - 이 함수에서 필요한 재수렴 쌍의 수를 반환
 *
 * @return: BRA_OP로 끝나는 기본 블록의 수 (각 분기마다 재수렴 쌍 1개 필요).
 *
 * 캐시된 num_reconvergence_pairs 값을 반환하며, 처음 호출 시 기본 블록을
 * 순회하여 계산한다. exit 기본 블록(마지막)은 분기를 가질 수 없으므로 제외한다.
 * 타이밍 시뮬레이터(shader.cc)가 SIMT 스택의 재수렴 쌍 배열 크기를 결정할 때 사용한다.
 *
 * 호출 체인:
 *   shader.cc(SIMT 스택 초기화) → [get_num_reconvergence_pairs]
 *   → num_reconvergence_pairs 캐시 또는 새로 계산
 */
unsigned function_info::get_num_reconvergence_pairs() {
  if (!num_reconvergence_pairs) {                    // [한국어] 아직 계산 안 됨 — 초기값 0
    if (m_basic_blocks.size() == 0) return 0;        // [한국어] 기본 블록 없으면 0 반환
    for (unsigned i = 0; i < (m_basic_blocks.size() - 1);
         i++) {  // last basic block containing exit obviously won't have a pair
      // [한국어] exit 블록(마지막) 제외하고 모든 블록 검사
      if (m_basic_blocks[i]->ptx_end->get_opcode() == BRA_OP) {  // [한국어] BRA로 끝나는 블록 카운트
        num_reconvergence_pairs++;  // [한국어] 재수렴 쌍 수 증가
      }
    }
  }
  return num_reconvergence_pairs;  // [한국어] 캐시된 값 반환
}

/*
 * [한국어]
 * function_info::get_reconvergence_pairs - 재수렴 쌍 배열을 채워 타이밍 시뮬레이터에 제공
 *
 * @recon_points: [출력] gpgpu_recon_t 배열. 호출자가 get_num_reconvergence_pairs() 크기로
 *               미리 할당해야 한다.
 * @return: 없음
 *
 * 각 BRA_OP 기본 블록에 대해 재수렴 쌍을 채운다:
 *   - source_pc / source_inst: BRA 명령어의 PC와 ptx_instruction*
 *   - target_pc / target_inst: ipdom 블록의 첫 명령어 PC와 ptx_instruction*
 *     (ipdom 블록이 exit BB이면 target_pc = -2, target_inst = NULL)
 * 타이밍 시뮬레이터(shader.cc)는 이 배열을 SIMT 스택에 저장하여,
 * 분기 명령어를 만날 때 해당 재수렴 PC를 스택에 push한다.
 *
 * 실행 컨텍스트: 시뮬레이션 초기화 단계 (싱글스레드).
 *
 * 호출 체인:
 *   shader.cc(커널 초기화) → [get_reconvergence_pairs]
 *   → recon_points 배열 채움 → SIMT 스택에 저장
 */
void function_info::get_reconvergence_pairs(gpgpu_recon_t *recon_points) {
  unsigned idx = 0;  // array index
  // [한국어] recon_points 배열의 현재 쓰기 인덱스
  if (m_basic_blocks.size() == 0) return;  // [한국어] 기본 블록 없으면 조기 반환
  for (unsigned i = 0; i < (m_basic_blocks.size() - 1);
       i++) {  // last basic block containing exit obviously won't have a pair
    // [한국어] exit 블록(마지막) 제외하고 모든 블록 검사
#ifdef DEBUG_GET_RECONVERG_PAIRS
    printf("i=%d\n", i);   // [한국어] 디버그 매크로 활성화 시: 현재 블록 인덱스 출력
    fflush(stdout);
#endif
    if (m_basic_blocks[i]->ptx_end->get_opcode() == BRA_OP) {
      // [한국어] BRA 명령어로 끝나는 블록 — 재수렴 쌍 생성
#ifdef DEBUG_GET_RECONVERG_PAIRS
      printf("\tbranch!\n");
      printf("\tbb_id=%d; ipdom=%d\n", m_basic_blocks[i]->bb_id,
             m_basic_blocks[i]->immediatepostdominator_id);
      printf("\tm_instr_mem index=%d\n",
             m_basic_blocks[i]->ptx_end->get_m_instr_mem_index());
      fflush(stdout);
#endif
      recon_points[idx].source_pc = m_basic_blocks[i]->ptx_end->get_PC();
      // [한국어] BRA 명령어의 PC를 재수렴 쌍의 source_pc로 설정
      recon_points[idx].source_inst = m_basic_blocks[i]->ptx_end;
      // [한국어] BRA 명령어의 ptx_instruction* 포인터 저장
#ifdef DEBUG_GET_RECONVERG_PAIRS
      printf("\trecon_points[idx].source_pc=%d\n", recon_points[idx].source_pc);
#endif
      if (m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
              ->ptx_begin) {
        // [한국어] ipdom 블록의 첫 명령어가 있으면 (exit BB가 아닌 실제 BB)
        recon_points[idx].target_pc =
            m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
                ->ptx_begin->get_PC();
        // [한국어] ipdom 블록의 첫 명령어 PC를 재수렴 대상으로 설정
        recon_points[idx].target_inst =
            m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
                ->ptx_begin;
        // [한국어] ipdom 블록의 첫 명령어 포인터 저장
      } else {
        // reconverge after function return
        // [한국어] ipdom 블록이 exit BB (ptx_begin=NULL) — 함수 반환 후 재수렴
        recon_points[idx].target_pc = -2;       // [한국어] -2는 "함수 반환 후 재수렴" 특수 값
        recon_points[idx].target_inst = NULL;   // [한국어] 명령어 없음
      }
#ifdef DEBUG_GET_RECONVERG_PAIRS
      m_basic_blocks[m_basic_blocks[i]->immediatepostdominator_id]
          ->ptx_begin->print_insn();
      printf("\trecon_points[idx].target_pc=%d\n", recon_points[idx].target_pc);
      fflush(stdout);
#endif
      idx++;  // [한국어] 다음 재수렴 쌍 슬롯으로 이동
    }
  }
}

// interface with graphviz (print the graph in DOT language) for plotting
/*
 * [한국어]
 * function_info::print_basic_block_dot - CFG를 GraphViz DOT 언어 형식으로 출력 (시각화)
 *
 * @return: 없음
 *
 * 기본 블록의 CFG 엣지를 GraphViz DOT 언어로 stdout에 출력한다.
 * 출력을 파일로 저장하고 "dot -Tpng"로 시각화하면 CFG 그래프를 이미지로 볼 수 있다.
 * g_debug_execution >= 50일 때 do_pdom()이 호출한다.
 *
 * 출력 형식: "digraph 함수명 { n -> m; n -> k; ... }"
 *
 * 호출 체인:
 *   function_info::do_pdom() (디버그 ≥50) → [print_basic_block_dot]
 */
void function_info::print_basic_block_dot() {
  printf("Basic Block in DOT\n");                      // [한국어] DOT 출력 시작 알림
  printf("digraph %s {\n", m_name.c_str());            // [한국어] GraphViz digraph 선언 (함수명을 그래프 이름으로)
  std::vector<basic_block_t *>::iterator bb_itr;
  for (bb_itr = m_basic_blocks.begin(); bb_itr != m_basic_blocks.end();
       bb_itr++) {                                     // [한국어] 모든 기본 블록 순회
    printf("\t");                                      // [한국어] 들여쓰기
    std::set<int>::iterator s;
    for (s = (*bb_itr)->successor_ids.begin();
         s != (*bb_itr)->successor_ids.end(); s++) {   // [한국어] 현재 블록의 모든 후계자 순회
      unsigned succ_bb = *s;                           // [한국어] 후계자 블록 ID
      printf("%d -> %d; ", (*bb_itr)->bb_id, succ_bb);  // [한국어] "source -> target;" DOT 엣지 출력
    }
    printf("\n");
  }
  printf("}\n");                                       // [한국어] DOT digraph 닫기
}

/*
 * [한국어]
 * ptx_kernel_shmem_size - 커널의 정적 공유 메모리 크기를 반환하는 C 인터페이스 래퍼
 *
 * @kernel_impl: function_info* 포인터를 void*로 캐스팅한 값.
 *               libcuda의 C 인터페이스가 void*로 전달.
 * @return: 이 커널이 사용하는 정적 .shared 메모리 바이트 수 (smem).
 *
 * libcuda에서 cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES) 구현에 사용된다.
 * function_info::m_kernel_info.smem 값을 반환하며, PTX 어셈블 후 확정된다.
 *
 * 호출 체인:
 *   libcuda/cuda_runtime_api.cc(cuFuncGetAttribute) → [ptx_kernel_shmem_size]
 *   → f->get_kernel_info()->smem
 */
unsigned ptx_kernel_shmem_size(void *kernel_impl) {
  function_info *f = (function_info *)kernel_impl;    // [한국어] void*를 function_info*로 다운캐스팅
  const struct gpgpu_ptx_sim_info *kernel_info = f->get_kernel_info();  // [한국어] 커널 리소스 정보 구조체 가져오기
  return kernel_info->smem;                           // [한국어] 정적 공유 메모리 크기(바이트) 반환
}

/*
 * [한국어]
 * ptx_kernel_nregs - 커널이 사용하는 레지스터 수를 반환하는 C 인터페이스 래퍼
 *
 * @kernel_impl: function_info* 포인터를 void*로 캐스팅한 값.
 * @return: 이 커널이 스레드당 사용하는 레지스터 수 (regs).
 *
 * libcuda의 cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS) 구현에 사용된다.
 * ptx_kernel_shmem_size()와 동일한 패턴의 void* C 인터페이스 래퍼이다.
 *
 * 호출 체인:
 *   libcuda/cuda_runtime_api.cc(cuFuncGetAttribute) → [ptx_kernel_nregs]
 *   → f->get_kernel_info()->regs
 */
unsigned ptx_kernel_nregs(void *kernel_impl) {
  function_info *f = (function_info *)kernel_impl;    // [한국어] void*를 function_info*로 다운캐스팅
  const struct gpgpu_ptx_sim_info *kernel_info = f->get_kernel_info();  // [한국어] 커널 리소스 정보 가져오기
  return kernel_info->regs;                           // [한국어] 스레드당 레지스터 수 반환
}

/*
 * [한국어]
 * type_info_key::type_decode (인스턴스 메서드) - 이 타입 키의 스칼라 타입을 디코딩
 *
 * @size: [출력] 이 타입의 비트 크기.
 * @basic_type: [출력] 기본 타입 범주 (1=부호 정수, 0=무부호 정수, -1=부동소수점, 2=pred, 3=텍스처).
 * @return: 0~16 범위의 타입 인덱스.
 *
 * 이 타입 키가 저장하고 있는 scalar_type() 값을 꺼내 정적 type_decode(int, ...) 에 위임한다.
 *
 * 호출 체인:
 *   cuda-sim.cc / ptx_sim.cc (타입 크기 결정) → [type_decode 인스턴스]
 *   → type_decode(int, size, basic_type)
 */
unsigned type_info_key::type_decode(size_t &size, int &basic_type) const {
  int type = scalar_type();           // [한국어] 이 타입 키의 스칼라 타입 토큰 (S32_TYPE, F32_TYPE 등)
  return type_decode(type, size, basic_type);  // [한국어] 정적 메서드에 위임
}

/*
 * [한국어]
 * type_info_key::type_decode (정적 메서드) - PTX 타입 토큰을 비트 크기와 기본 타입 범주로 변환
 *
 * @type: PTX 타입 토큰 정수 (S8_TYPE, U32_TYPE, F32_TYPE, PRED_TYPE 등 — ptx.tab.h 정의).
 * @size: [출력] 비트 크기 (8/16/32/64/128/1).
 * @basic_type: [출력] 기본 타입 범주:
 *              1 = 부호 정수 (S8~S64)
 *              0 = 무부호 정수 또는 비트 타입 (U8~U64, B8~B128)
 *             -1 = 부동소수점 (F16, F32, F64, FF64)
 *              2 = 술어(predicate) 레지스터 (PRED)
 *              3 = 텍스처/샘플러/서피스 레퍼런스 (TEXREF, SAMPLERREF, SURFREF)
 * @return: 0~16 범위의 타입 인덱스 (배열 인덱싱이나 switch 대체에 사용).
 *
 * 기능 시뮬레이터(ptx_sim.cc, instructions.cc)가 각 명령어의 데이터 타입을 결정할 때
 * 사용한다. 반환 인덱스는 타입별 연산 분기에 사용된다.
 *
 * 알려지지 않은 타입이면 에러 메시지 출력 후 assert(0)으로 중단한다.
 *
 * 호출 체인:
 *   instructions.cc (명령어 실행) → [type_decode(int, ...)] → (size, basic_type) 반환
 */
unsigned type_info_key::type_decode(int type, size_t &size, int &basic_type) {
  switch (type) {
    case S8_TYPE:                  // [한국어] .s8: 8비트 부호 정수
      size = 8;
      basic_type = 1;              // [한국어] 부호 정수 범주
      return 0;                    // [한국어] 타입 인덱스 0
    case S16_TYPE:                 // [한국어] .s16: 16비트 부호 정수
      size = 16;
      basic_type = 1;
      return 1;
    case S32_TYPE:                 // [한국어] .s32: 32비트 부호 정수 (가장 자주 쓰임)
      size = 32;
      basic_type = 1;
      return 2;
    case S64_TYPE:                 // [한국어] .s64: 64비트 부호 정수
      size = 64;
      basic_type = 1;
      return 3;
    case U8_TYPE:                  // [한국어] .u8: 8비트 무부호 정수
      size = 8;
      basic_type = 0;              // [한국어] 무부호 정수 범주
      return 4;
    case U16_TYPE:                 // [한국어] .u16: 16비트 무부호 정수
      size = 16;
      basic_type = 0;
      return 5;
    case U32_TYPE:                 // [한국어] .u32: 32비트 무부호 정수
      size = 32;
      basic_type = 0;
      return 6;
    case U64_TYPE:                 // [한국어] .u64: 64비트 무부호 정수
      size = 64;
      basic_type = 0;
      return 7;
    case F16_TYPE:                 // [한국어] .f16: 16비트 반정밀도 부동소수점 (IEEE 754 half)
      size = 16;
      basic_type = -1;             // [한국어] 부동소수점 범주
      return 8;
    case F32_TYPE:                 // [한국어] .f32: 32비트 단정밀도 부동소수점 (IEEE 754 float)
      size = 32;
      basic_type = -1;
      return 9;
    case F64_TYPE:                 // [한국어] .f64: 64비트 배정밀도 부동소수점 (IEEE 754 double)
      size = 64;
      basic_type = -1;
      return 10;
    case FF64_TYPE:                // [한국어] .ff64: PTXPlus 64비트 FP 쌍 (F32x2) — F64와 동일 인덱스
      size = 64;
      basic_type = -1;
      return 10;                   // [한국어] F64와 같은 인덱스 반환 (동일 처리)
    case PRED_TYPE:                // [한국어] .pred: 1비트 술어(predicate) 레지스터
      size = 1;
      basic_type = 2;             // [한국어] 술어 타입 범주
      return 11;
    case B8_TYPE:                  // [한국어] .b8: 8비트 비트 패턴 (타입 없는 바이트)
      size = 8;
      basic_type = 0;             // [한국어] 비트 타입도 무부호 범주로 취급
      return 12;
    case B16_TYPE:                 // [한국어] .b16: 16비트 비트 패턴
      size = 16;
      basic_type = 0;
      return 13;
    case B32_TYPE:                 // [한국어] .b32: 32비트 비트 패턴 (원자 연산에 자주 사용)
      size = 32;
      basic_type = 0;
      return 14;
    case B64_TYPE:                 // [한국어] .b64: 64비트 비트 패턴
      size = 64;
      basic_type = 0;
      return 15;
    case BB64_TYPE:                // [한국어] .bb64: PTXPlus 64비트 비트쌍 (B32x2) — B64와 동일 인덱스
      size = 64;
      basic_type = 0;
      return 15;                   // [한국어] B64와 같은 인덱스 반환
    case BB128_TYPE:               // [한국어] .bb128: 128비트 비트 패턴 (PTXPlus 확장 — B32x4)
      size = 128;
      basic_type = 0;
      return 16;
    case TEXREF_TYPE:              // [한국어] .texref: 텍스처 레퍼런스 오브젝트 (32비트 핸들)
    case SAMPLERREF_TYPE:          // [한국어] .samplerref: 샘플러 레퍼런스 오브젝트
    case SURFREF_TYPE:             // [한국어] .surfref: 서피스 레퍼런스 오브젝트
      size = 32;
      basic_type = 3;             // [한국어] 텍스처/레퍼런스 범주
      return 16;
    default:
      printf("ERROR ** type_decode() does not know about \"%s\"\n",
             decode_token(type));  // [한국어] 알 수 없는 타입 토큰 오류 메시지 (decode_token으로 이름 출력)
      assert(0);                   // [한국어] 즉시 중단 — 알 수 없는 타입은 지원하지 않음
      return 0xDEADBEEF;           // [한국어] 컴파일러 경고 억제 (실제로는 도달 안 함)
  }
}

/*
 * [한국어]
 * copy_arg_to_buffer - CALL 명령어의 실제 인자를 arg_buffer_t로 읽어오기
 *
 * @thread: 현재 실행 중인 PTX 스레드 (레지스터 파일과 지역 메모리 접근용).
 * @actual_param_op: CALL 명령어에서 전달된 실제 인자 오퍼랜드 (레지스터 또는 .param 지역).
 * @formal_param: 피호출 함수의 형식 매개변수 심볼 (크기 정보 포함).
 * @return: 값이 복사된 arg_buffer_t 객체 (레지스터 값 또는 메모리 버퍼 포함).
 *
 * CALL 명령어 실행 시, 호출자 스레드의 실제 인자를 임시 버퍼(arg_buffer_t)로 읽는다.
 * 두 가지 케이스를 처리한다:
 *   1) 레지스터 인자 (is_reg()): get_reg()로 레지스터 값 직접 읽기
 *   2) .param 지역 인자 (is_param_local()): 지역 메모리 스택에서 메모리 읽기
 *      (스택 포인터 + param 오프셋 = 실제 주소)
 * 이 버퍼는 copy_buffer_to_frame()이 피호출 함수의 스택 프레임에 쓸 때 사용된다.
 * 1024바이트 임시 버퍼를 사용하므로 param 크기가 1024 미만이어야 한다 (assert).
 *
 * 실행 컨텍스트: CALL 명령어 실행 시 스레드별로 호출 (기능 시뮬레이션).
 *
 * 호출 체인:
 *   copy_args_into_buffer_list() → [copy_arg_to_buffer]
 *   → thread->get_reg() 또는 thread->m_local_mem->read()
 */
arg_buffer_t copy_arg_to_buffer(ptx_thread_info *thread,
                                operand_info actual_param_op,
                                const symbol *formal_param) {
  if (actual_param_op.is_reg()) {                        // [한국어] 실제 인자가 레지스터인 경우
    ptx_reg_t value = thread->get_reg(actual_param_op.get_symbol());
    // [한국어] 호출자 스레드의 레지스터 파일에서 인자 심볼의 값 읽기
    return arg_buffer_t(formal_param, actual_param_op, value);
    // [한국어] 형식 매개변수 심볼, 실제 오퍼랜드, 레지스터 값으로 arg_buffer_t 생성
  } else if (actual_param_op.is_param_local()) {         // [한국어] 실제 인자가 .param(지역) 공간인 경우
    unsigned size = formal_param->get_size_in_bytes();   // [한국어] 형식 매개변수의 바이트 크기
    addr_t frame_offset = actual_param_op.get_symbol()->get_address();
    // [한국어] .param 심볼의 스택 프레임 내 오프셋
    addr_t from_addr = thread->get_local_mem_stack_pointer() + frame_offset;
    // [한국어] 실제 지역 메모리 주소 = 스택 포인터 + 오프셋
    char buffer[1024];                                   // [한국어] 인자 값을 담을 임시 스택 버퍼 (1024바이트 제한)
    assert(size < 1024);                                 // [한국어] 1024바이트 초과 인자는 지원 불가 (간단한 구현 제약)
    thread->m_local_mem->read(from_addr, size, buffer);  // [한국어] 지역 메모리에서 인자 값을 버퍼로 읽기
    return arg_buffer_t(formal_param, actual_param_op, buffer, size);
    // [한국어] 메모리 버퍼 내용으로 arg_buffer_t 생성
  } else {
    printf(
        "GPGPU-Sim PTX: ERROR ** need to add support for this operand type in "
        "call/return\n");  // [한국어] 미지원 오퍼랜드 타입 에러 — 레지스터와 param_local만 지원
    abort();               // [한국어] 즉시 비정상 종료 (미구현 케이스)
  }
}

/*
 * [한국어]
 * copy_args_into_buffer_list - CALL 명령어의 모든 실제 인자를 arg_buffer_list에 복사
 *
 * @pI: 실행 중인 CALL ptx_instruction (실제 인자 오퍼랜드 포함).
 * @thread: 현재 실행 스레드 (레지스터/메모리 읽기용).
 * @target_func: 피호출 함수의 function_info (형식 매개변수 개수와 심볼 접근용).
 * @arg_values: [출력] 읽어온 인자들을 담는 arg_buffer_list_t (vector).
 * @return: 없음
 *
 * CALL 명령어 실행의 첫 단계로, 호출자 스레드의 실제 인자를 모두 임시 버퍼 목록에 저장한다.
 * pI의 오퍼랜드 배열에서 반환값(n_return)과 함수 주소(+1) 다음 오퍼랜드부터 인자가 시작된다.
 * 각 인자에 대해 copy_arg_to_buffer()를 호출하고 결과를 arg_values에 추가한다.
 * 중간 버퍼가 필요한 이유: 인자를 읽는 시점과 피호출 함수 프레임에 쓰는 시점이 분리되어야
 * (스택 포인터 이동 후 쓰기) 하기 때문이다.
 *
 * 실행 컨텍스트: CALL 명령어 실행 시 스레드별로 호출.
 *
 * 호출 체인:
 *   instructions.cc(call_impl) → [copy_args_into_buffer_list]
 *   → copy_arg_to_buffer() (각 인자마다)
 *   → copy_buffer_list_into_frame() (스택 프레임에 쓰기)
 */
void copy_args_into_buffer_list(const ptx_instruction *pI,
                                ptx_thread_info *thread,
                                const function_info *target_func,
                                arg_buffer_list_t &arg_values) {
  unsigned n_return = target_func->has_return();         // [한국어] 반환값 오퍼랜드 개수 (0 또는 1)
  unsigned n_args = target_func->num_args();             // [한국어] 피호출 함수의 형식 매개변수 수
  for (unsigned arg = 0; arg < n_args; arg++) {          // [한국어] 각 형식 매개변수에 대해 실제 인자 읽기
    const operand_info &actual_param_op =
        pI->operand_lookup(n_return + 1 + arg);
    // [한국어] CALL 오퍼랜드 배열 인덱스: [반환값들] [함수주소] [인자0] [인자1] ...
    //         n_return개의 반환값 오퍼랜드 + 1(함수 주소) + arg 인덱스
    const symbol *formal_param = target_func->get_arg(arg);  // [한국어] 피호출 함수의 arg번째 형식 매개변수 심볼
    arg_values.push_back(
        copy_arg_to_buffer(thread, actual_param_op, formal_param));
    // [한국어] 실제 인자 값을 읽어 버퍼 목록에 추가
  }
}

/*
 * [한국어]
 * copy_buffer_to_frame - 하나의 arg_buffer_t를 피호출 함수의 스택 프레임에 쓰기
 *
 * @thread: 피호출 함수 실행 컨텍스트의 스레드 (스택 포인터와 레지스터 파일 접근용).
 * @a: 쓸 인자 버퍼 (레지스터 값 또는 메모리 버퍼).
 * @return: 없음
 *
 * arg_buffer_t에서 값을 꺼내 피호출 함수의 새 스택 프레임에 저장한다.
 * 두 가지 케이스:
 *   1) 레지스터 매개변수: 피호출 함수의 레지스터 파일에 직접 set_reg()
 *   2) .param 지역 매개변수: 피호출 함수의 지역 메모리에 write()
 *      (새 스택 포인터 + 형식 매개변수의 오프셋)
 * 이 함수는 스택 포인터가 피호출 함수의 프레임으로 이미 이동한 후 호출된다.
 *
 * 실행 컨텍스트: CALL 명령어 실행 시, 스택 프레임 전환 후 스레드별로 호출.
 *
 * 호출 체인:
 *   copy_buffer_list_into_frame() → [copy_buffer_to_frame]
 *   → thread->set_reg() 또는 thread->m_local_mem->write()
 */
void copy_buffer_to_frame(ptx_thread_info *thread, const arg_buffer_t &a) {
  if (a.is_reg()) {                                      // [한국어] 레지스터 매개변수인 경우
    ptx_reg_t value = a.get_reg();                       // [한국어] 버퍼에서 레지스터 값 꺼내기
    operand_info dst_reg =
        operand_info(a.get_dst(), thread->get_gpu()->gpgpu_ctx);
    // [한국어] 형식 매개변수 심볼로 오퍼랜드 생성 (레지스터 번호 결정용)
    thread->set_reg(dst_reg.get_symbol(), value);        // [한국어] 피호출 함수 레지스터 파일에 값 저장
  } else {                                               // [한국어] .param 지역 매개변수인 경우
    const void *buffer = a.get_param_buffer();           // [한국어] 인자 값이 담긴 메모리 버퍼 포인터
    size_t size = a.get_param_buffer_size();             // [한국어] 인자 바이트 크기
    const symbol *dst = a.get_dst();                     // [한국어] 형식 매개변수 심볼 (오프셋 정보)
    addr_t frame_offset = dst->get_address();            // [한국어] 피호출 함수 스택 프레임 내 매개변수 오프셋
    addr_t to_addr = thread->get_local_mem_stack_pointer() + frame_offset;
    // [한국어] 실제 지역 메모리 주소 = 새 스택 포인터 + 오프셋
    thread->m_local_mem->write(to_addr, size, buffer, NULL, NULL);
    // [한국어] 지역 메모리에 인자 값 쓰기 (마지막 두 NULL: warp/lane 정보 미사용)
  }
}

/*
 * [한국어]
 * copy_buffer_list_into_frame - 모든 인자 버퍼를 피호출 함수의 스택 프레임에 일괄 쓰기
 *
 * @thread: 피호출 함수 실행 컨텍스트의 스레드.
 * @arg_values: copy_args_into_buffer_list()가 채운 인자 버퍼 목록.
 * @return: 없음
 *
 * arg_values 목록을 순회하며 각 원소에 copy_buffer_to_frame()을 호출한다.
 * CALL 명령어 실행의 최종 단계로, 모든 인자를 피호출 함수 프레임에 기록한다.
 *
 * 실행 컨텍스트: CALL 명령어 실행 시, 스택 프레임 전환 후 스레드별로 호출.
 *
 * 호출 체인:
 *   instructions.cc(call_impl) → [copy_buffer_list_into_frame]
 *   → copy_buffer_to_frame() (각 인자마다)
 */
void copy_buffer_list_into_frame(ptx_thread_info *thread,
                                 arg_buffer_list_t &arg_values) {
  arg_buffer_list_t::iterator a;
  for (a = arg_values.begin(); a != arg_values.end(); a++) {  // [한국어] 모든 인자 버퍼 순회
    copy_buffer_to_frame(thread, *a);  // [한국어] 각 인자를 피호출 함수 스택 프레임에 복사
  }
}

/*
 * [한국어]
 * check_operands - PTX 명령어 생성자에서 오퍼랜드 타입을 검증/변환하는 내부 유틸리티
 *
 * @opcode: PTX opcode 정수 (CVT_OP, SET_OP 등 — opcodes.h 정의).
 * @scalar_type: 명령어에 지정된 스칼라 타입 목록 (보통 0~1개).
 * @operands: 파서가 전달한 원래 오퍼랜드 목록.
 * @ctx: gpgpu_context 포인터 (새 operand_info 생성 시 필요).
 * @return: 검증/변환된 오퍼랜드 목록 (대부분의 경우 입력과 동일).
 *
 * 두 가지 처리를 수행한다:
 *   1) 두 타입 명령어(CVT, SET, SLCT, TEX, MMA, DP4A, VMIN, VMAX):
 *      리터럴 오퍼랜드 경고 메시지만 출력 (변환 없음).
 *      이 명령어들은 두 개의 타입 지정자를 갖는 PTX 명령어로 리터럴 처리가 복잡하다.
 *   2) 단일 타입 명령어 (대부분):
 *      리터럴이 double 타입이지만 명령어가 F32_TYPE을 요구하면,
 *      double 리터럴을 float으로 다운캐스팅하여 새 operand_info로 대체한다.
 *      PTX 소스에서 "1.0"이 double로 파싱되지만 .f32 명령어에서는 float이어야 하는 경우.
 *
 * 실행 컨텍스트: ptx_instruction 생성자에서 호출 (파싱 단계, 싱글스레드).
 *
 * 호출 체인:
 *   ptx_instruction::ptx_instruction() → [check_operands] → 변환된 오퍼랜드 목록 반환
 */
static std::list<operand_info> check_operands(
    int opcode, const std::list<int> &scalar_type,
    const std::list<operand_info> &operands, gpgpu_context *ctx) {
  static int g_warn_literal_operands_two_type_inst;
  // [한국어] 두 타입 명령어의 리터럴 경고를 한 번만 출력하기 위한 static 플래그
  if ((opcode == CVT_OP) || (opcode == SET_OP) || (opcode == SLCT_OP) ||
      (opcode == TEX_OP) || (opcode == MMA_OP) || (opcode == DP4A_OP) ||
      (opcode == VMIN_OP) || (opcode == VMAX_OP)) {
    // [한국어] 두 타입 지정자를 갖는 특수 명령어 (CVT=변환, SET=비교설정, SLCT=선택,
    //         TEX=텍스처, MMA=행렬곱, DP4A=dot product, VMIN/VMAX=벡터 min/max)
    // just make sure these do not have have const operands...
    // [한국어] 이 명령어들에 리터럴 오퍼랜드가 있으면 경고 (한 번만)
    if (!g_warn_literal_operands_two_type_inst) {        // [한국어] 아직 경고를 출력하지 않은 경우
      std::list<operand_info>::const_iterator o;
      for (o = operands.begin(); o != operands.end(); o++) {  // [한국어] 모든 오퍼랜드 검사
        const operand_info &op = *o;
        if (op.is_literal()) {                           // [한국어] 리터럴 상수 오퍼랜드 발견
          printf(
              "GPGPU-Sim PTX: PTX uses two scalar type intruction with literal "
              "operand.\n");                             // [한국어] 두 타입 명령어에 리터럴 경고 출력
          g_warn_literal_operands_two_type_inst = 1;    // [한국어] 경고 출력 완료 플래그 설정
        }
      }
    }
  } else {                                               // [한국어] 단일 타입 명령어 처리
    assert(scalar_type.size() < 2);                     // [한국어] 단일 타입 명령어는 타입 지정자가 0 또는 1개
    if (scalar_type.size() == 1) {                      // [한국어] 타입 지정자가 1개 있는 경우만 변환 수행
      std::list<operand_info> result;                   // [한국어] 변환된 오퍼랜드를 담을 새 목록
      int inst_type = scalar_type.front();              // [한국어] 명령어의 스칼라 타입 (예: F32_TYPE)
      std::list<operand_info>::const_iterator o;
      for (o = operands.begin(); o != operands.end(); o++) {  // [한국어] 모든 오퍼랜드 순회
        const operand_info &op = *o;
        if (op.is_literal()) {                          // [한국어] 리터럴 상수 오퍼랜드인 경우
          if ((op.get_type() == double_op_t) && (inst_type == F32_TYPE)) {
            // [한국어] double 리터럴인데 명령어가 F32를 요구하는 경우 — 다운캐스팅 필요
            ptx_reg_t v = op.get_literal_value();       // [한국어] 리터럴 값을 ptx_reg_t 유니온으로 읽기
            float u = (float)v.f64;                     // [한국어] double(f64)을 float으로 명시적 다운캐스팅
            operand_info n(u, ctx);                     // [한국어] float 값으로 새 operand_info 생성
            result.push_back(n);                        // [한국어] 변환된 float 리터럴 오퍼랜드 추가
          } else {
            result.push_back(op);   // [한국어] double 외 타입이거나 명령어 타입이 F32가 아닌 경우 — 원본 그대로 추가
          }
        } else {
          result.push_back(op);     // [한국어] 리터럴이 아닌 오퍼랜드 — 변환 없이 그대로 추가
        }
      }
      return result;                // [한국어] 변환된 오퍼랜드 목록 반환
    }
  }
  return operands;  // [한국어] 변환 불필요한 경우 원본 오퍼랜드 목록 그대로 반환
}

/*
 * [한국어]
 * ptx_instruction::ptx_instruction - PTX 명령어 객체의 대형 생성자
 *
 * @opcode: PTX 명령어 opcode 정수 (ADD_OP, LD_OP, BRA_OP 등 — opcodes.h).
 * @pred: 술어 레지스터 심볼 포인터 (없으면 NULL). "@p bra" 에서 p.
 * @neg_pred: 술어 부정 여부 (1이면 "@!p").
 * @pred_mod: 술어 수정자 플래그.
 * @label: 이 명령어가 레이블이면 레이블 심볼 포인터 (일반 명령어는 NULL).
 * @operands: 파서가 전달한 원본 오퍼랜드 목록 (check_operands() 통과 후 저장).
 * @return_var: CALL 명령어의 반환값 오퍼랜드.
 * @options: 명령어 옵션 토큰 목록 (SAT_OPTION, RN_OPTION, V4_TYPE 등).
 * @wmma_options: wmma(Warp Matrix Multiply-Accumulate) 전용 옵션 목록.
 * @scalar_type: 명령어의 스칼라 타입 목록 (보통 1개 또는 0개).
 * @space_spec: 메모리 공간 지정 (LD/ST 명령어용).
 * @file: 소스 파일명 (디버그 정보용).
 * @line: 소스 줄 번호.
 * @source: 소스 코드 텍스트 (탭 제거 후 m_source에 저장).
 * @config: SM 코어 설정 (부모 warp_inst_t 생성자에 전달).
 * @ctx: gpgpu_context 포인터.
 * @return: 없음 (생성자)
 *
 * PTX 파서(ptx.y)가 명령어를 파싱할 때 생성자를 호출하여 ptx_instruction 객체를 만든다.
 * 초기화 순서:
 *   1) warp_inst_t(config) 부모 생성자 (타이밍 시뮬레이션용 기본 필드)
 *   2) UID, PC, opcode, pred, label, operands 저장
 *   3) 모든 옵션 플래그를 기본값(false/0/RN_OPTION 등)으로 초기화
 *   4) wmma_options 파싱: WMMA 명령어(행렬 곱) 전용 옵션 설정
 *   5) options 파싱: 공통 옵션 100+가지를 switch-case로 처리
 *      (barrier, compare, saturation, rounding, hi/lo/wide, uni,
 *       geometry, vector, atomic, vote, membar, cache, shfl, prmt 등)
 *   6) 메모리 공간 결정: LD/ST/LDU의 undefined_space → generic_space
 *   7) 오퍼랜드에서 메모리 공간 오버라이드
 *   8) 소스 파일/줄/텍스트 저장, 탭 제거
 *   9) CALL opcode이면 함수 이름 확인: printf → m_is_printf, CDP → m_is_cdp
 *
 * 실행 컨텍스트: PTX 파싱 단계 (싱글스레드, 시뮬레이션 초기화 전).
 *
 * 호출 체인:
 *   ptx_parser.cc(명령어 완성 시) → [ptx_instruction()] → check_operands()
 *   → 옵션 파싱 switch-case들
 *   → function_info::add_instruction()으로 명령어 목록에 추가
 */
ptx_instruction::ptx_instruction(
    int opcode, const symbol *pred, int neg_pred, int pred_mod, symbol *label,
    const std::list<operand_info> &operands, const operand_info &return_var,
    const std::list<int> &options, const std::list<int> &wmma_options,
    const std::list<int> &scalar_type, memory_space_t space_spec,
    const char *file, unsigned line, const char *source,
    const core_config *config, gpgpu_context *ctx)
    : warp_inst_t(config), m_return_var(ctx) {
  // [한국어] warp_inst_t(config): 타이밍 시뮬레이터용 부모 클래스 초기화 (SM 코어 설정 전달)
  // [한국어] m_return_var(ctx): 반환값 오퍼랜드 초기화 (컨텍스트 포인터 필요)
  gpgpu_ctx = ctx;                           // [한국어] 전역 컨텍스트 포인터 저장
  m_uid = ++(ctx->g_num_ptx_inst_uid);       // [한국어] 전역 명령어 UID 선위증가 후 저장 (전체 시뮬레이션에서 유일)
  m_PC = 0;                                  // [한국어] PC는 아직 미할당 (ptx_assemble()에서 설정됨)
  m_opcode = opcode;                         // [한국어] PTX opcode 저장 (ADD_OP, BRA_OP 등)
  m_pred = pred;                             // [한국어] 술어 레지스터 심볼 포인터 (NULL이면 무조건 실행)
  m_neg_pred = neg_pred;                     // [한국어] 술어 부정 플래그 (1이면 "@!p" 형태)
  m_pred_mod = pred_mod;                     // [한국어] 술어 수정자 (추가 조건)
  m_label = label;                           // [한국어] 레이블 심볼 (레이블 명령어는 non-NULL, 일반은 NULL)
  const std::list<operand_info> checked_operands =
      check_operands(opcode, scalar_type, operands, ctx);
  // [한국어] 오퍼랜드 타입 검증/변환 (double→float 다운캐스팅 등)
  m_operands.insert(m_operands.begin(), checked_operands.begin(),
                    checked_operands.end());
  // [한국어] 검증된 오퍼랜드 목록을 m_operands 벡터에 복사 삽입
  m_return_var = return_var;                 // [한국어] CALL 명령어의 반환값 오퍼랜드 저장
  m_options = options;                       // [한국어] 전체 옵션 목록 저장 (나중에 접근 가능하도록)
  m_wmma_options = wmma_options;             // [한국어] WMMA 전용 옵션 목록 저장
  m_wide = false;                            // [한국어] wide 연산 플래그 초기화 (MUL.WIDE 등)
  m_hi = false;                              // [한국어] high-part 연산 플래그 초기화 (MUL.HI 등)
  m_lo = false;                              // [한국어] low-part 연산 플래그 초기화 (MUL.LO 등)
  m_uni = false;                             // [한국어] uniform 분기 플래그 초기화 (BRA.UNI: 워프 전체 같은 방향)
  m_exit = false;                            // [한국어] exit 플래그 초기화 (CALLP.BRANCH.EXIT 등)
  m_abs = false;                             // [한국어] 절댓값 플래그 초기화 (ABS 옵션)
  m_neg = false;                             // [한국어] 부정 플래그 초기화 (NEG 옵션)
  m_to_option = false;                       // [한국어] TO 옵션 플래그 초기화 (CVT.TO 등)
  m_cache_option = 0;                        // [한국어] 캐시 힌트 옵션 초기화 (CA/CG/CS/LU/CV/WB/WT)
  m_rounding_mode = RN_OPTION;              // [한국어] 반올림 모드 기본값: RN (round to nearest even)
  m_compare_op = -1;                         // [한국어] 비교 연산자 초기화: -1 = 없음
  m_saturation_mode = 0;                     // [한국어] 포화 연산 모드 초기화: 0 = 없음
  m_clamp_mode = 0;                          // [한국어] 클램프 모드 초기화: 0 = WRAP
  m_left_mode = 0;                           // [한국어] 방향 모드 초기화: 0 = RIGHT
  m_geom_spec = 0;                           // [한국어] 텍스처 기하 스펙 초기화: 0 = 없음 (1D/2D/3D 선택 전)
  m_vector_spec = 0;                         // [한국어] 벡터 너비 스펙 초기화: 0 = 스칼라
  m_atomic_spec = 0;                         // [한국어] 원자 연산 종류 초기화: 0 = 없음
  m_membar_level = 0;                        // [한국어] 메모리 배리어 범위 초기화: 0 = 없음
  m_inst_size = 8;  // bytes                 // [한국어] 명령어 크기 기본값: 8바이트 (PTX 기본)
  int rr = 0;                                // [한국어] wmma_layout 배열 인덱스 (행/열 레이아웃용)
  std::list<int>::const_iterator i;          // [한국어] 옵션 목록 순회 이터레이터
  unsigned n = 1;                            // [한국어] 옵션 순번 카운터 (현재 미사용)
  for (i = wmma_options.begin(); i != wmma_options.end(); i++, n++) {
    // [한국어] WMMA(Warp Matrix Multiply-Accumulate) 전용 옵션 파싱
    //         wmma.load/store/mma 명령어의 행렬 조각 타입과 레이아웃 결정
    int last_ptx_inst_option = *i;           // [한국어] 현재 WMMA 옵션 토큰
    switch (last_ptx_inst_option) {
      case SYNC_OPTION:                      // [한국어] wmma.sync: 동기화 버전 (Volta+)
      case LOAD_A:                           // [한국어] wmma.load.a: A 행렬 로드
      case LOAD_B:                           // [한국어] wmma.load.b: B 행렬 로드
      case LOAD_C:                           // [한국어] wmma.load.c: C(accumulator) 행렬 로드
      case STORE_D:                          // [한국어] wmma.store.d: D(result) 행렬 저장
      case MMA:                              // [한국어] wmma.mma: 행렬 곱셈-누적 수행
        m_wmma_type = last_ptx_inst_option;  // [한국어] WMMA 연산 종류 저장
        break;
      case ROW:                              // [한국어] 행 우선(row-major) 레이아웃
      case COL:                              // [한국어] 열 우선(column-major) 레이아웃
        m_wmma_layout[rr++] = last_ptx_inst_option;  // [한국어] 행렬 레이아웃 배열에 저장 (A, B, C/D 순)
        break;
      case M16N16K16:                        // [한국어] 16x16x16 타일 크기
      case M32N8K16:                         // [한국어] 32x8x16 타일 크기
      case M8N32K16:                         // [한국어] 8x32x16 타일 크기
        break;                               // [한국어] 타일 크기는 별도 저장 없이 pass (현재 미사용)
      default:
        assert(0);                           // [한국어] 알 수 없는 WMMA 옵션 — 즉시 중단
        break;
    }
  }
  rr = 0;                                    // [한국어] rr 재사용 (이하 일반 옵션에서는 사용 안 함)
  n = 1;                                     // [한국어] n 카운터 재설정
  for (i = options.begin(); i != options.end(); i++, n++) {
    // [한국어] 일반 명령어 옵션 파싱 (100+가지 옵션 토큰을 switch-case로 처리)
    int last_ptx_inst_option = *i;           // [한국어] 현재 옵션 토큰
    switch (last_ptx_inst_option) {
      case SYNC_OPTION:                      // [한국어] bar.sync: 배리어 동기화
      case ARRIVE_OPTION:                    // [한국어] bar.arrive: 배리어 도착 통지
      case RED_OPTION:                       // [한국어] bar.red: 배리어 리덕션
        m_barrier_op = last_ptx_inst_option; // [한국어] 배리어 연산 종류 저장
        break;
      case EQU_OPTION:                       // [한국어] ==U (unsigned equal or unordered FP)
      case NEU_OPTION:                       // [한국어] !=U
      case LTU_OPTION:                       // [한국어] <U
      case LEU_OPTION:                       // [한국어] <=U
      case GTU_OPTION:                       // [한국어] >U
      case GEU_OPTION:                       // [한국어] >=U
      case EQ_OPTION:                        // [한국어] == (signed/ordered FP)
      case NE_OPTION:                        // [한국어] !=
      case LT_OPTION:                        // [한국어] <
      case LE_OPTION:                        // [한국어] <=
      case GT_OPTION:                        // [한국어] >
      case GE_OPTION:                        // [한국어] >=
      case LS_OPTION:                        // [한국어] <= (unsigned low-or-same, SASS 전용)
      case HS_OPTION:                        // [한국어] >= (unsigned high-or-same)
        m_compare_op = last_ptx_inst_option; // [한국어] 비교 연산자 종류 저장 (SET/SETP 명령어용)
        break;
      case NUM_OPTION:                       // [한국어] NUM: 두 피연산자 모두 NaN이 아닌 경우 (ordered)
      case NAN_OPTION:                       // [한국어] NAN: 어느 한쪽이 NaN인 경우 (unordered)
        m_compare_op = last_ptx_inst_option;
        // assert(0); // finish this
        // [한국어] 향후 완전 구현 필요 (TODO 주석) — 현재는 비교 연산자만 저장
        break;
      case SAT_OPTION:                       // [한국어] .sat: 결과를 [0,1] 또는 [-128,127] 등으로 포화
        m_saturation_mode = 1;              // [한국어] 포화 모드 활성화
        break;
      case WRAP_OPTION:                      // [한국어] .wrap: 텍스처 래핑 모드
        m_clamp_mode = 0;                   // [한국어] 클램프 없음 (순환 래핑)
        break;
      case CLAMP_OPTION:                     // [한국어] .clamp: 텍스처 클램프 모드
        m_clamp_mode = 1;                   // [한국어] 클램프 모드 활성화 (경계에서 포화)
        break;
      case LEFT_OPTION:                      // [한국어] 좌측 방향 옵션
        m_left_mode = 1;                    // [한국어] 좌측 모드 활성화
        break;
      case RIGHT_OPTION:                     // [한국어] 우측 방향 옵션
        m_left_mode = 0;                    // [한국어] 우측 모드 (기본값)
        break;
      case RNI_OPTION:                       // [한국어] round to nearest even, int 대상
      case RZI_OPTION:                       // [한국어] round toward zero, int 대상
      case RMI_OPTION:                       // [한국어] round toward minus infinity, int 대상
      case RPI_OPTION:                       // [한국어] round toward plus infinity, int 대상
      case RN_OPTION:                        // [한국어] round to nearest even (FP → FP)
      case RZ_OPTION:                        // [한국어] round toward zero
      case RM_OPTION:                        // [한국어] round toward minus infinity
      case RP_OPTION:                        // [한국어] round toward plus infinity
        m_rounding_mode = last_ptx_inst_option;  // [한국어] 반올림 모드 저장 (FMA/CVT/SQRT 등에서 사용)
        break;
      case HI_OPTION:                        // [한국어] .hi: 곱셈 결과의 상위 절반 사용 (MUL.HI)
        m_compare_op = last_ptx_inst_option; // [한국어] HI를 compare_op 필드에도 저장 (호환성)
        m_hi = true;                         // [한국어] HI 플래그 설정
        assert(!m_lo);                       // [한국어] LO와 동시에 설정 불가 (상호 배제)
        assert(!m_wide);                     // [한국어] WIDE와도 동시 설정 불가 (상호 배제)
        break;
      case LO_OPTION:                        // [한국어] .lo: 곱셈 결과의 하위 절반 사용 (MUL.LO)
        m_compare_op = last_ptx_inst_option;
        m_lo = true;                         // [한국어] LO 플래그 설정
        assert(!m_hi);                       // [한국어] HI와 동시에 설정 불가
        assert(!m_wide);                     // [한국어] WIDE와도 동시 설정 불가
        break;
      case WIDE_OPTION:                      // [한국어] .wide: 두 배 너비 결과 (MUL.WIDE: 32→64비트)
        m_wide = true;                       // [한국어] WIDE 플래그 설정
        assert(!m_lo);                       // [한국어] LO와 동시에 설정 불가
        assert(!m_hi);                       // [한국어] HI와도 동시 설정 불가
        break;
      case UNI_OPTION:
        m_uni = true;  // don't care... < now we DO care when constructing
                       // flowgraph>
        // [한국어] .uni: uniform branch — 워프의 모든 활성 스레드가 동일한 방향으로 분기
        //         CFG 구성 시 fall-through 엣지 추가 여부 결정에 사용됨
        break;
      case GEOM_MODIFIER_1D:                 // [한국어] 텍스처 1D 기하 (1차원 텍스처)
      case GEOM_MODIFIER_2D:                 // [한국어] 텍스처 2D 기하 (2차원 텍스처)
      case GEOM_MODIFIER_3D:                 // [한국어] 텍스처 3D 기하 (3차원 텍스처)
        m_geom_spec = last_ptx_inst_option;  // [한국어] 텍스처 기하 스펙 저장 (TEX/SURED/SULD)
        break;
      case V2_TYPE:                          // [한국어] .v2: 2-원소 벡터 (LD.v2, ST.v2)
      case V3_TYPE:                          // [한국어] .v3: 3-원소 벡터
      case V4_TYPE:                          // [한국어] .v4: 4-원소 벡터 (가장 흔함)
        m_vector_spec = last_ptx_inst_option; // [한국어] 벡터 너비 스펙 저장
        break;
      case ATOMIC_AND:                       // [한국어] 원자 AND 연산
      case ATOMIC_OR:                        // [한국어] 원자 OR 연산
      case ATOMIC_XOR:                       // [한국어] 원자 XOR 연산
      case ATOMIC_CAS:                       // [한국어] 원자 CAS(Compare-And-Swap) 연산
      case ATOMIC_EXCH:                      // [한국어] 원자 EXCH(Exchange) 연산
      case ATOMIC_ADD:                       // [한국어] 원자 ADD 연산
      case ATOMIC_INC:                       // [한국어] 원자 INC(Increment, 0~N 순환) 연산
      case ATOMIC_DEC:                       // [한국어] 원자 DEC(Decrement, 0~N 순환) 연산
      case ATOMIC_MIN:                       // [한국어] 원자 MIN 연산
      case ATOMIC_MAX:                       // [한국어] 원자 MAX 연산
        m_atomic_spec = last_ptx_inst_option; // [한국어] 원자 연산 종류 저장 (ATOM 명령어용)
        break;
      case APPROX_OPTION:                    // [한국어] .approx: 근사 연산 (RCP.APPROX, SQRT.APPROX 등)
        break;                               // [한국어] 현재 별도 처리 없음 (pass)
      case FULL_OPTION:                      // [한국어] .full: 완전 정밀도 연산
        break;                               // [한국어] 현재 별도 처리 없음
      case ANY_OPTION:                       // [한국어] vote.any: 어느 스레드든 true이면 true
        m_vote_mode = vote_any;              // [한국어] vote 모드: any
        break;
      case ALL_OPTION:                       // [한국어] vote.all: 모든 스레드가 true여야 true
        m_vote_mode = vote_all;              // [한국어] vote 모드: all
        break;
      case BALLOT_OPTION:                    // [한국어] vote.ballot: 각 스레드의 술어를 비트마스크로 수집
        m_vote_mode = vote_ballot;           // [한국어] vote 모드: ballot
        break;
      case GLOBAL_OPTION:                    // [한국어] .global: membar.gl — 전역 메모리 배리어
        m_membar_level = GLOBAL_OPTION;      // [한국어] 메모리 배리어 범위: 글로벌 (모든 SM)
        break;
      case CTA_OPTION:                       // [한국어] .cta: membar.cta — CTA(스레드 블록) 내 배리어
        m_membar_level = CTA_OPTION;         // [한국어] 메모리 배리어 범위: CTA
        break;
      case SYS_OPTION:                       // [한국어] .sys: membar.sys — 시스템 전체 배리어 (CPU-GPU 포함)
        m_membar_level = SYS_OPTION;         // [한국어] 메모리 배리어 범위: 시스템 전체
        break;
      case FTZ_OPTION:                       // [한국어] .ftz: flush-to-zero (비정규화 수를 0으로 처리)
        break;                               // [한국어] 별도 플래그 없음 (시뮬레이터에서 FTZ 별도 처리)
      case EXIT_OPTION:                      // [한국어] .exit: CALLP에서 탈출 분기 옵션
        m_exit = true;                       // [한국어] exit 플래그 설정
        break;
      case ABS_OPTION:                       // [한국어] .abs: 절댓값 결과
        m_abs = true;                        // [한국어] 절댓값 플래그 설정
        break;
      case NEG_OPTION:                       // [한국어] .neg: 부정(negate) 결과
        m_neg = true;                        // [한국어] 부정 플래그 설정
        break;
      case TO_OPTION:                        // [한국어] .to: 역방향 CVT 옵션
        m_to_option = true;                  // [한국어] TO 옵션 플래그 설정
        break;
      case CA_OPTION:                        // [한국어] .ca: cache all (L1/L2 캐시 모두 사용)
      case CG_OPTION:                        // [한국어] .cg: cache at global (L2만 사용)
      case CS_OPTION:                        // [한국어] .cs: cache streaming (evict-first)
      case LU_OPTION:                        // [한국어] .lu: last use 힌트
      case CV_OPTION:                        // [한국어] .cv: cache volatile (항상 L2)
      case WB_OPTION:                        // [한국어] .wb: write-back
      case WT_OPTION:                        // [한국어] .wt: write-through
        m_cache_option = last_ptx_inst_option;  // [한국어] 캐시 힌트 저장
        break;
      case HALF_OPTION:                      // [한국어] PTXPlus 반 크기(4바이트) 명령어
        m_inst_size = 4;  // bytes           // [한국어] 명령어 크기를 4바이트로 설정
        break;
      case EXTP_OPTION:                      // [한국어] .extp: 확장 파라미터 (CDP용)
        break;                               // [한국어] 별도 처리 없음
      case NC_OPTION:                        // [한국어] .nc: non-coherent read-only 캐시
        m_cache_option = last_ptx_inst_option;  // [한국어] NC 옵션 저장
        break;
      case UP_OPTION:                        // [한국어] shfl.up: 낮은 레인에서 수신
      case DOWN_OPTION:                      // [한국어] shfl.down: 높은 레인에서 수신
      case BFLY_OPTION:                      // [한국어] shfl.bfly: XOR butterfly 셔플
      case IDX_OPTION:                       // [한국어] shfl.idx: 특정 레인 인덱스
        m_shfl_op = last_ptx_inst_option;    // [한국어] warp shuffle 연산 종류 저장
        break;
      case PRMT_F4E_MODE:                    // [한국어] prmt.f4e: forward 4-extract
      case PRMT_B4E_MODE:                    // [한국어] prmt.b4e: backward 4-extract
      case PRMT_RC8_MODE:                    // [한국어] prmt.rc8: replicate 8
      case PRMT_ECL_MODE:                    // [한국어] prmt.ecl: edge clamp left
      case PRMT_ECR_MODE:                    // [한국어] prmt.ecr: edge clamp right
      case PRMT_RC16_MODE:                   // [한국어] prmt.rc16: replicate 16
        m_prmt_op = last_ptx_inst_option;    // [한국어] PRMT(permute) 모드 저장
        break;
      default:
        assert(0);                           // [한국어] 알 수 없는 옵션 — 파서-코드 불일치 즉시 중단
        break;
    }
  }
  m_scalar_type = scalar_type;              // [한국어] 명령어 스칼라 타입 목록 저장 (F32_TYPE 등)
  m_space_spec = space_spec;               // [한국어] 파서가 전달한 메모리 공간 초기 설정
  if ((opcode == ST_OP || opcode == LD_OP || opcode == LDU_OP) &&
      (space_spec == undefined_space)) {    // [한국어] LD/ST/LDU에서 공간 미지정인 경우
    m_space_spec = generic_space;           // [한국어] generic(범용) 공간으로 설정 (PTX 5.0+ 기본)
  }
  for (std::vector<operand_info>::const_iterator i = m_operands.begin();
       i != m_operands.end(); ++i) {        // [한국어] 오퍼랜드에서 메모리 공간 오버라이드 확인
    const operand_info &op = *i;
    if (op.get_addr_space() != undefined_space)
      m_space_spec =
          op.get_addr_space();  // TODO: can have more than one memory space for
                                // ptxplus (g8x) inst
      // [한국어] 오퍼랜드가 특정 공간 주소를 지정하면 해당 공간으로 덮어쓰기
  }
  if (opcode == TEX_OP) m_space_spec = tex_space;
  // [한국어] TEX는 항상 텍스처 공간 — 오퍼랜드 공간 오버라이드보다 우선

  m_source_file = file ? file : "<unknown>"; // [한국어] 소스 파일명 저장 (NULL이면 "<unknown>")
  m_source_line = line;                      // [한국어] 소스 줄 번호 저장
  m_source = source;                         // [한국어] 소스 코드 텍스트 저장
  // Trim tabs
  m_source.erase(std::remove(m_source.begin(), m_source.end(), '\t'),
                 m_source.end());
  // [한국어] std::remove로 탭을 뒤로 이동 후 erase로 제거 — 소스 텍스트 정제

  if (opcode == CALL_OP) {                   // [한국어] CALL 명령어인 경우 호출 대상 함수 이름으로 특수 처리
    const operand_info &target = func_addr();  // [한국어] CALL의 함수 주소 오퍼랜드
    assert(target.is_function_address());    // [한국어] 반드시 함수 주소여야 함
    const symbol *func_addr = target.get_symbol();  // [한국어] 함수 주소 심볼
    const function_info *target_func = func_addr->get_pc();  // [한국어] function_info 가져오기
    std::string fname = target_func->get_name();  // [한국어] 호출 대상 함수 이름

    if (fname == "vprintf") {               // [한국어] 디바이스 printf 내부 함수
      m_is_printf = true;                   // [한국어] printf 플래그 설정 — 특수 실행 경로 활성화
    }
    if (fname == "cudaStreamCreateWithFlags") m_is_cdp = 1;
    // [한국어] CDP: 디바이스에서 CUDA 스트림 생성 → m_is_cdp=1
    if (fname == "cudaGetParameterBufferV2") m_is_cdp = 2;
    // [한국어] CDP: 커널 파라미터 버퍼 요청 → m_is_cdp=2
    if (fname == "cudaLaunchDeviceV2") m_is_cdp = 4;
    // [한국어] CDP: 디바이스에서 새 커널 런치 (핵심 CDP 연산) → m_is_cdp=4
  }
}

/*
 * [한국어]
 * ptx_instruction::print_insn (인자 없음) - 표준 출력으로 이 명령어 정보 출력
 *
 * @return: 없음
 *
 * stdout에 to_string() 결과를 출력하고 fflush로 버퍼를 비운다.
 * 디버그 상황에서 명령어를 즉시 확인하기 위한 편의 래퍼.
 *
 * 호출 체인:
 *   print_basic_blocks() / 디버그 직접 호출 → [print_insn()] → print_insn(stdout) → fflush
 */
void ptx_instruction::print_insn() const {
  print_insn(stdout);   // [한국어] stdout을 인자로 FILE* 버전 위임
  fflush(stdout);       // [한국어] 버퍼 강제 플러시 — 즉시 출력 보장
}

/*
 * [한국어]
 * ptx_instruction::print_insn (FILE*) - 지정된 FILE 스트림에 명령어 정보 출력
 *
 * @fp: 출력 대상 FILE 포인터 (stdout, stderr, 또는 파일).
 * @return: 없음
 *
 * to_string()으로 생성한 문자열을 그대로 fprintf로 출력하는 단순 래퍼.
 *
 * 호출 체인:
 *   function_info::print_insn() / print_basic_blocks() → [print_insn(fp)] → to_string()
 */
void ptx_instruction::print_insn(FILE *fp) const {
  fprintf(fp, "%s", to_string().c_str());  // [한국어] to_string() 결과를 FILE 스트림에 출력
}

/*
 * [한국어]
 * ptx_instruction::to_string - 명령어 정보를 문자열로 변환
 *
 * @return: "PC=0x... (파일:라인) 소스텍스트" 형태의 std::string.
 *          레이블이면 PC 없이 공백 패딩.
 *
 * 명령어 디버그 출력의 핵심 함수. STR_SIZE(1024) 버퍼에 PC 주소, 소스 위치,
 * 소스 텍스트를 순서대로 snprintf로 구성한다.
 * 레이블 명령어는 PC가 없으므로 공백 16자로 패딩하여 정렬을 맞춘다.
 *
 * 호출 체인:
 *   print_insn() → [to_string] → std::string 반환
 *   function_info::get_insn_str() → [to_string]
 */
std::string ptx_instruction::to_string() const {
  char buf[STR_SIZE];                    // [한국어] STR_SIZE(1024)바이트 출력 버퍼
  unsigned used_bytes = 0;               // [한국어] 버퍼에 이미 사용된 바이트 수 (경계 안전 snprintf용)
  if (!is_label()) {                     // [한국어] 일반 실행 명령어인 경우 PC 출력
    used_bytes += snprintf(buf + used_bytes, STR_SIZE - used_bytes,
                           " PC=0x%03llx ", m_PC);
    // [한국어] 16진수 3자리 PC 출력 (예: " PC=0x01a ")
  } else {
    used_bytes +=
        snprintf(buf + used_bytes, STR_SIZE - used_bytes, "                ");
    // [한국어] 레이블은 PC 없음 — 16자 공백으로 정렬 패딩
  }
  used_bytes +=
      snprintf(buf + used_bytes, STR_SIZE - used_bytes, "(%s:%d) %s",
               m_source_file.c_str(), m_source_line, m_source.c_str());
  // [한국어] "(파일명:라인) 소스코드텍스트" 형태로 소스 위치 + 내용 추가
  return std::string(buf);               // [한국어] char 버퍼를 std::string으로 변환하여 반환
}

/*
 * [한국어]
 * ptx_instruction::get_pred - 이 명령어의 술어 레지스터를 operand_info로 반환
 *
 * @return: 술어 레지스터 심볼로 생성된 operand_info 객체.
 *          m_pred가 NULL이면 빈 operand_info (술어 없는 명령어).
 *
 * 타이밍 시뮬레이터나 기능 시뮬레이터가 명령어 실행 전 술어를 평가할 때 사용한다.
 * operand_info 생성자가 gpgpu_ctx를 필요로 하므로 ctx를 전달한다.
 *
 * 호출 체인:
 *   ptx_sim.cc(명령어 실행 전 술어 평가) → [get_pred] → operand_info(m_pred, ctx)
 */
operand_info ptx_instruction::get_pred() const {
  return operand_info(m_pred, gpgpu_ctx);  // [한국어] m_pred 심볼로 operand_info 생성 후 반환
}

/*
 * [한국어]
 * function_info::function_info - 함수 정보 객체의 생성자
 *
 * @entry_point: 진입점 종류:
 *               1 = 커널 엔트리 포인트 (cuLaunchKernel의 대상)
 *               2 = extern 함수 선언 (구현 없음, PTX 외부)
 *               0 = 일반 디바이스 함수 (device 함수, 커널 아님)
 * @ctx: gpgpu_context 포인터 (전역 UID 카운터 접근용).
 * @return: 없음 (생성자)
 *
 * add_function_decl()에서 새 함수를 처음 만날 때 호출된다. 모든 멤버를
 * 기본값으로 초기화하고, m_kernel_info(cmem/lmem/regs/smem)는 0으로 초기화한다.
 * 실제 값은 PTX 파싱 중 .maxntid, .param 등 지시어를 처리하면서 채워진다.
 * m_args_aligned_size = -1은 "아직 계산 안 됨"을 의미한다.
 * pdom_done = false: do_pdom()이 아직 실행되지 않았음 (어셈블 전 상태).
 *
 * 실행 컨텍스트: PTX 파싱 단계 (싱글스레드, 시뮬레이션 초기화 전).
 *
 * 호출 체인:
 *   symbol_table::add_function_decl() → new function_info(entry_point, ctx)
 */
function_info::function_info(int entry_point, gpgpu_context *ctx) {
  gpgpu_ctx = ctx;                                    // [한국어] 전역 컨텍스트 포인터 저장
  m_uid = (gpgpu_ctx->function_info_sm_next_uid)++;   // [한국어] 전역 function_info UID 채번 (후위증가)
  m_entry_point = (entry_point == 1) ? true : false;  // [한국어] entry_point==1이면 커널 엔트리 플래그 설정
  m_extern = (entry_point == 2) ? true : false;       // [한국어] entry_point==2이면 extern 함수 플래그 설정
  num_reconvergence_pairs = 0;                        // [한국어] 재수렴 쌍 수 초기화 (처음 조회 시 계산됨)
  m_symtab = NULL;                                    // [한국어] 함수 스코프 심볼 테이블 — add_function_decl에서 설정
  m_assembled = false;                                // [한국어] 어셈블 완료 플래그 — ptx_assemble() 후 true
  m_return_var_sym = NULL;                            // [한국어] 반환값 심볼 — .funcname 선언에서 설정
  m_kernel_info.cmem = 0;                             // [한국어] 상수 메모리 사용량 초기화 (bytes)
  m_kernel_info.lmem = 0;                             // [한국어] 지역(스택) 메모리 사용량 초기화 (bytes)
  m_kernel_info.regs = 0;                             // [한국어] 스레드당 레지스터 사용량 초기화
  m_kernel_info.smem = 0;                             // [한국어] 공유 메모리 사용량 초기화 (bytes)
  m_local_mem_framesize = 0;                          // [한국어] 지역 메모리 스택 프레임 크기 초기화
  m_args_aligned_size = -1;                           // [한국어] 인자 정렬 크기 캐시 초기화: -1 = 미계산
  pdom_done = false;  // initialize it to false       // [한국어] 후위지배자 분석 완료 플래그 초기화
}

/*
 * [한국어]
 * function_info::print_insn - 지정 PC의 명령어를 함수명(demangle)과 함께 출력
 *
 * @pc: 출력할 명령어의 프로그램 카운터 값.
 * @fp: 출력 대상 FILE 포인터.
 * @return: 출력한 명령어의 크기(바이트). 알 수 없으면 1.
 *
 * c++filt -p를 popen()으로 실행하여 m_name의 C++ mangled 이름을 demangled 이름으로 변환한다.
 * 이후 해당 PC의 ptx_instruction::print_insn()을 호출하여 명령어 내용을 출력한다.
 * pc가 범위 밖이거나 해당 슬롯이 NULL이면 에러 메시지를 출력한다.
 * 타이밍 시뮬레이터의 디버그 출력에서 "함수명 + 명령어" 형태로 로그를 남길 때 사용한다.
 *
 * 실행 컨텍스트: 디버그/오류 출력 시 (싱글스레드).
 *
 * 호출 체인:
 *   shader.cc(디버그) / cuda-sim.cc → [function_info::print_insn(pc, fp)]
 *   → popen("c++filt") → ptx_instruction::print_insn(fp)
 */
unsigned function_info::print_insn(unsigned pc, FILE *fp) const {
  unsigned inst_size = 1;  // return offset to next instruction or 1 if unknown
  // [한국어] 반환할 명령어 크기 기본값: 1 (알 수 없는 경우)
  unsigned index = pc - m_start_PC;         // [한국어] m_instr_mem 배열 인덱스 = PC - 함수 시작 PC
  char command[1024];                        // [한국어] popen 명령어 문자열 버퍼
  char buffer[1024];                         // [한국어] c++filt 출력을 담는 버퍼
  memset(command, 0, 1024);                  // [한국어] 명령어 버퍼 0으로 초기화
  memset(buffer, 0, 1024);                   // [한국어] 출력 버퍼 0으로 초기화
  snprintf(command, 1024, "c++filt -p %s", m_name.c_str());
  // [한국어] "c++filt -p <mangled_name>" 명령어 생성 — -p는 파라미터 타입 생략 옵션
  FILE *p = popen(command, "r");             // [한국어] c++filt 프로세스 실행 후 출력 파이프 열기
  buffer[0] = 0;                             // [한국어] 버퍼 첫 바이트 명시적 초기화
  assert(fgets(buffer, 1023, p) != NULL);   // [한국어] c++filt 출력 한 줄 읽기 (NULL이면 중단)
  // Remove trailing "\n" in buffer
  char *c;
  if ((c = strchr(buffer, '\n')) != NULL) *c = '\0';
  // [한국어] c++filt 출력 끝의 개행 문자 제거 (strchr로 찾아 NUL 대체)
  fprintf(fp, "%s", buffer);                // [한국어] demangled 함수명 출력
  if (index >= m_instr_mem_size) {           // [한국어] PC가 이 함수의 명령어 메모리 범위를 초과
    fprintf(fp, "<past last instruction (max pc=%u)>",
            m_start_PC + m_instr_mem_size - 1);  // [한국어] 범위 초과 에러 메시지 출력
  } else {
    if (m_instr_mem[index] != NULL) {        // [한국어] 해당 PC에 명령어가 있는 경우
      m_instr_mem[index]->print_insn(fp);   // [한국어] 명령어 내용 출력
      inst_size = m_instr_mem[index]->isize;  // [한국어] 실제 명령어 크기 반환용으로 저장
    } else
      fprintf(fp, "<no instruction at pc = %u>", pc);  // [한국어] NULL 슬롯 — 가변 크기 명령어의 두 번째 슬롯
  }
  pclose(p);                                 // [한국어] c++filt 프로세스 닫기
  return inst_size;                          // [한국어] 명령어 크기 반환 (1이면 알 수 없음)
}

/*
 * [한국어]
 * function_info::get_insn_str - 지정 PC의 명령어 내용을 문자열로 반환
 *
 * @pc: 조회할 프로그램 카운터 값.
 * @return: 해당 PC 명령어의 to_string() 결과.
 *          범위 초과나 NULL 슬롯이면 에러 메시지 문자열 반환.
 *
 * function_info::print_insn()의 문자열 버전. 파이프 없이 m_instr_mem[index]->to_string()을
 * 직접 반환한다. 함수명 demangle 없이 순수 명령어 내용만 필요할 때 사용한다.
 *
 * 호출 체인:
 *   cuda-sim.cc / shader.cc(디버그) → [get_insn_str] → m_instr_mem[index]->to_string()
 */
std::string function_info::get_insn_str(unsigned pc) const {
  unsigned index = pc - m_start_PC;          // [한국어] m_instr_mem 배열 인덱스 계산
  if (index >= m_instr_mem_size) {            // [한국어] 범위 초과 확인
    char buff[STR_SIZE];
    buff[STR_SIZE - 1] = '\0';               // [한국어] 버퍼 마지막 바이트 NUL 설정 (오버플로우 방지)
    snprintf(buff, STR_SIZE, "<past last instruction (max pc=%u)>",
             m_start_PC + m_instr_mem_size - 1);  // [한국어] 범위 초과 에러 메시지 포맷
    return std::string(buff);                // [한국어] 에러 메시지 문자열 반환
  } else {
    if (m_instr_mem[index] != NULL) {        // [한국어] 유효한 명령어 슬롯인 경우
      return m_instr_mem[index]->to_string();  // [한국어] 명령어 to_string() 반환
    } else {
      char buff[STR_SIZE];
      buff[STR_SIZE - 1] = '\0';             // [한국어] 버퍼 NUL 초기화
      snprintf(buff, STR_SIZE, "<no instruction at pc = %u>", pc);
      // [한국어] NULL 슬롯 에러 메시지 (가변 크기 명령어의 두 번째 슬롯)
      return std::string(buff);              // [한국어] 에러 메시지 반환
    }
  }
}

/*
 * [한국어]
 * gpgpu_ptx_assemble - 커널 이름으로 PTX 어셈블리를 시작하는 전역 진입 함수
 *
 * @kname: 어셈블할 커널 함수 이름 (std::string).
 * @kinfo: function_info* 포인터를 void*로 캐스팅한 값.
 * @return: 없음
 *
 * libcuda가 커널을 실행하기 전 각 커널에 대해 이 함수를 한 번 호출한다.
 * 기본 검사를 수행한 후 function_info::ptx_assemble()에 위임한다:
 *   - kinfo == NULL: 함수 정의가 없는 경우 경고 후 반환
 *   - is_extern(): extern 선언만 있고 구현이 없는 함수 — 어셈블 불필요
 *   - 정상: func_info->ptx_assemble() 호출 → PC 할당 + CFG 분석 (do_pdom())
 *
 * 이 함수가 반환한 후 function_info는 시뮬레이션 실행 준비 완료 상태(m_assembled=true).
 *
 * 실행 컨텍스트: 첫 번째 커널 런치 전 초기화 단계 (싱글스레드).
 *
 * 호출 체인:
 *   libcuda/cuda_runtime_api.cc(cuLaunchKernel 준비) → [gpgpu_ptx_assemble]
 *   → function_info::ptx_assemble() → do_pdom()
 */
void gpgpu_ptx_assemble(std::string kname, void *kinfo) {
  function_info *func_info = (function_info *)kinfo;  // [한국어] void*를 function_info*로 다운캐스팅
  if ((function_info *)kinfo == NULL) {               // [한국어] kinfo가 NULL인 경우 — 등록되지 않은 함수
    printf("GPGPU-Sim PTX: Warning - missing function definition \'%s\'\n",
           kname.c_str());                            // [한국어] 함수 정의 없음 경고 출력
    return;                                           // [한국어] 어셈블 불가 — 조기 반환
  }
  if (func_info->is_extern()) {                       // [한국어] extern 선언만 있는 함수 (구현 없음)
    printf(
        "GPGPU-Sim PTX: skipping assembly for extern declared function "
        "\'%s\'\n",
        func_info->get_name().c_str());               // [한국어] extern 함수 어셈블 생략 알림
    return;                                           // [한국어] extern은 어셈블 필요 없음 — 조기 반환
  }
  func_info->ptx_assemble();  // [한국어] 실제 어셈블 실행: PC 할당 + CFG 분석(do_pdom()) + s_g_pc_to_insn 등록
}
