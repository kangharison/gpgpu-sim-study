/*
 * [한국어 설명] PTX 기능 시뮬레이션 — CTA/warp/스레드 상태 관리 구현 (ptx_sim.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 기능 시뮬레이션(functional simulation) 레이어에서
 * CTA(Cooperative Thread Array, CUDA thread block), warp, 개별 스레드(ptx_thread_info)의
 * 생명주기와 상태를 관리하는 클래스 메서드를 구현한다. 구체적으로는 CTA 생성/소멸 추적,
 * 스레드 종료 등록, 배리어(__syncthreads) 카운터 관리, PTX 스레드의 콜스택(callstack)
 * push/pop, 레지스터 파일 덤프, 특수 레지스터(%tid, %smid 등) 값 제공 등을 담당한다.
 * 기능 시뮬레이션 레이어는 PTX 명령어의 의미론적 결과(레지스터/메모리 값 변화)를 계산하며,
 * 사이클-레벨 타이밍(지연, 스톨, 파이프라인 흐름)은 gpgpu-sim/shader.cc가 담당한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim의 2-레이어 구조에서 기능 시뮬레이션 레이어(cuda-sim/)에 속한다.
 *
 *   CUDA Application
 *     → libcuda (런타임 인터셉트)
 *         → gpgpusim_entrypoint.cc (시뮬레이터 진입)
 *             → gpgpu-sim/shader.cc (타이밍 모델, 사이클-레벨 파이프라인)
 *                 → cuda-sim/ptx_sim.cc (기능 시뮬레이션: CTA/스레드 상태)
 *                     → cuda-sim/instructions.cc (PTX 명령어별 실행 시맨틱)
 *
 * 실행 컨텍스트: 호스트 유저스페이스, 단일 시뮬레이션 스레드.
 * 각 GPU 스레드(ptx_thread_info)는 시뮬레이터 내에서 C++ 객체로 표현되며,
 * 실제 병렬 실행이 아니라 호스트 CPU에서 순차적으로 시뮬레이션된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존(이 파일이 사용하는):
 *   - ptx_sim.h : ptx_cta_info, ptx_warp_info, ptx_thread_info 클래스 선언
 *   - ptx_ir.h  : ptx_instruction, function_info, symbol_table, symbol, type_info 등 PTX IR 타입
 *   - gpu-sim.h : gpgpu_sim (gpu_sim_cycle 접근용), gpgpu_context
 *   - shader.h  : shader_core_ctx (get_warp_size(), get_kernel_info() 등)
 *   - gpgpu_context.h : g_ptx_cta_info_sm_idx_used, g_ptx_thread_info_uid_next 전역 상태
 *   - ptx.tab.h : PTX 파서 심볼 (feature_not_implemented 등)
 * 피의존(이 파일에 의존하는):
 *   - cuda-sim/instructions.cc : ptx_exec_inst() 내부에서 ptx_thread_info 메서드 호출
 *   - gpgpu-sim/shader.cc : ptx_sim_init_thread(), execute_warp_inst_t() 등에서 ptx_thread_info 객체를 생성·관리
 *   - gpgpu-sim/gpu-sim.cc : CTA 할당/해제 시 ptx_cta_info 생성·소멸 호출
 * 데이터 흐름:
 *   shader.cc가 warp 실행을 구동 → ptx_thread_info::get_builtin()으로 특수 레지스터 제공
 *   → instructions.cc가 각 PTX 명령어 실행 → 결과가 ptx_thread_info 레지스터 파일에 저장
 *   → 스레드 종료 시 ptx_cta_info::register_thread_exit() → CTA 완료 감지
 *
 * === 주요 함수/구조체 요약 ===
 * ptx_cta_info::ptx_cta_info()         : CTA 객체 생성, SM 인덱스 등록 및 UID 할당.
 * ptx_cta_info::check_cta_thread_status_and_reset(): 모든 스레드 종료 검증 후 CTA 재설정.
 * ptx_thread_info::ptx_thread_info()   : 스레드 상태 전체 초기화 (PC, 레지스터, 콜스택).
 * ptx_thread_info::get_builtin()       : %tid, %ctaid, %smid 등 PTX 특수 레지스터 값 제공.
 * ptx_thread_info::callstack_push/pop(): 함수 호출/반환 시 콜스택과 레지스터 프레임 관리.
 * ptx_thread_info::dump_regs()         : 디버그용 레지스터 파일 전체 출력.
 */

// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda
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

#include "ptx_sim.h"      // [한국어] ptx_cta_info, ptx_warp_info, ptx_thread_info 클래스 선언 포함
#include <string>         // [한국어] std::string — snprintf 결과 저장, 심볼 이름 처리에 사용
#include "ptx_ir.h"       // [한국어] PTX IR 타입 (ptx_instruction, function_info, symbol_table, type_info 등) 포함
class ptx_recognizer;     // [한국어] PTX Lex 파서의 전방 선언 — 전체 정의 없이 yyscan_t typedef를 위해 필요
typedef void *yyscan_t;   // [한국어] Flex 재진입 스캐너 핸들 타입 — PTX 렉서가 사용하는 불투명 포인터
#include "../../libcuda/gpgpu_context.h" // [한국어] gpgpu_context 전역 시뮬레이터 컨텍스트 포함 — g_ptx_cta_info_sm_idx_used 등 전역 상태 접근
#include "../gpgpu-sim/gpu-sim.h"        // [한국어] gpgpu_sim 클래스 포함 — gpu_sim_cycle, gpu_tot_sim_cycle 필드 접근
#include "../gpgpu-sim/shader.h"         // [한국어] shader_core_ctx 포함 — get_warp_size(), get_kernel_info() 등 SM 인터페이스 접근
#include "ptx.tab.h"      // [한국어] PTX yacc 파서가 생성한 헤더 — 토큰 상수 및 feature_not_implemented 선언 포함

void feature_not_implemented(const char *f); // [한국어] 미구현 PTX 특수 레지스터 접근 시 에러 출력 후 종료하는 함수 전방 선언

/*
 * [한국어]
 * ptx_cta_info::ptx_cta_info - CTA(thread block) 시뮬레이션 정보 객체 생성자
 *
 * @sm_idx : 이 CTA가 할당된 SM(Streaming Multiprocessor)의 인덱스.
 *           shader_core_ctx::m_sid 값과 대응하며, 0부터 시작하는 정수이다.
 * @ctx    : 시뮬레이터 전역 컨텍스트 포인터 (gpgpu_context).
 *           g_ptx_cta_info_sm_idx_used 집합과 g_ptx_cta_info_uid 카운터 접근에 사용.
 *
 * 이 생성자는 새로운 CTA가 SM에 할당될 때 gpu-sim.cc의 issue_block2core()에서 호출된다.
 * SM 인덱스 중복 사용을 assert로 검증하고, 전역 UID 카운터를 증가시켜 이 CTA에 고유 ID를 부여한다.
 * 배리어(__syncthreads) 카운터 m_bar_threads를 0으로 초기화하여 동기화 상태를 시작 상태로 설정한다.
 *
 * 실행 컨텍스트: 호스트 CPU, 단일 시뮬레이션 스레드, CTA 할당 시점 (cycle 함수 내부).
 * 에러 경로: sm_idx가 이미 사용 중이면 assert 실패 → 프로그램 종료 (시뮬레이터 내부 버그).
 *
 * 호출 체인:
 *   gpu-sim.cc (issue_block2core) → [ptx_cta_info 생성자] → g_ptx_cta_info_sm_idx_used 등록
 */
ptx_cta_info::ptx_cta_info(unsigned sm_idx, gpgpu_context *ctx) {
  // [한국어] 이 SM 인덱스가 아직 사용 중이지 않음을 검증한다.
  // 동일한 SM에 두 CTA가 동시에 할당되는 것은 시뮬레이터 논리 오류이므로 assert로 보호한다.
  assert(ctx->func_sim->g_ptx_cta_info_sm_idx_used.find(sm_idx) ==
         ctx->func_sim->g_ptx_cta_info_sm_idx_used.end());
  ctx->func_sim->g_ptx_cta_info_sm_idx_used.insert(sm_idx); // [한국어] SM 인덱스를 사용 중 집합에 등록하여 이후 중복 할당을 방지한다

  m_sm_idx = sm_idx;                       // [한국어] 이 CTA가 실행 중인 SM의 인덱스를 저장한다
  m_uid = (ctx->g_ptx_cta_info_uid)++;     // [한국어] 전역 CTA UID 카운터에서 고유 ID를 할당하고 카운터를 증가시킨다. 후위 증가(++)이므로 현재값이 먼저 할당된다
  m_bar_threads = 0;                        // [한국어] 배리어(__syncthreads)에 도달한 스레드 수를 0으로 초기화한다. 모든 스레드가 배리어에 도달해야 실행이 재개된다
  gpgpu_ctx = ctx;                          // [한국어] 전역 시뮬레이터 컨텍스트 포인터를 저장한다. 이후 check_cta_thread_status_and_reset() 등에서 전역 상태 접근에 사용된다
}

/*
 * [한국어]
 * ptx_cta_info::add_thread - CTA에 스레드를 등록한다
 *
 * @thd : 이 CTA에 속하는 ptx_thread_info 포인터.
 *
 * CTA가 생성된 후 소속 스레드들을 m_threads_in_cta 집합에 추가한다.
 * 이 집합은 이후 check_cta_thread_status_and_reset()에서 모든 스레드가 종료했는지
 * 검증하는 데 사용된다.
 * 실행 컨텍스트: 호스트 CPU, CTA 할당 직후 스레드 초기화 단계.
 * 호출 체인:
 *   shader.cc (ptx_sim_init_thread) → ptx_cta_info::add_thread
 */
void ptx_cta_info::add_thread(ptx_thread_info *thd) {
  m_threads_in_cta.insert(thd); // [한국어] 스레드 포인터를 CTA 소속 집합에 삽입한다. std::set이므로 중복 삽입은 무시된다
}

/*
 * [한국어]
 * ptx_cta_info::num_threads - CTA에 등록된 스레드 수를 반환한다
 *
 * @return : m_threads_in_cta 집합의 크기 — 이 CTA에 속한 총 스레드 수.
 *
 * 호출 체인:
 *   shader.cc (CTA 완료 확인 시) → ptx_cta_info::num_threads
 */
unsigned ptx_cta_info::num_threads() const { return m_threads_in_cta.size(); } // [한국어] std::set의 size()를 반환한다. CTA 크기는 생성 후 변하지 않으므로 이 값도 고정이다

/*
 * [한국어]
 * ptx_cta_info::check_cta_thread_status_and_reset - CTA 완료 검증 및 상태 초기화
 *
 * 이 함수는 CTA가 완전히 실행을 마치고 SM에서 해제(deallocation)되기 전에 호출된다.
 * 두 가지 검사를 수행한다:
 *   1. m_threads_that_have_exited 수 == m_threads_in_cta 수 (모든 스레드가 종료했는지)
 *   2. 개별 스레드 객체의 is_done() 상태가 참인지 (이중 확인)
 * 검사 실패 시 아직 실행 중인 스레드 목록을 출력한 후 abort()로 종료한다.
 * 두 검사 모두 통과하면 세 집합(m_threads_in_cta, m_threads_that_have_exited,
 * m_dangling_pointers)을 모두 clear()하여 이 CTA 객체가 재사용 가능한 상태가 된다.
 *
 * 실행 컨텍스트: 호스트 CPU, CTA 해제 시점 (gpu-sim.cc의 CTA 완료 처리 중).
 * 에러 경로: 스레드 미완료 시 printf 진단 출력 후 abort() → 코어 덤프.
 *
 * 호출 체인:
 *   gpu-sim.cc (shader_core_ctx::register_cta_thread_exit / CTA 해제) → [이 함수]
 */
void ptx_cta_info::check_cta_thread_status_and_reset() {
  bool fail = false;                        // [한국어] 첫 번째 검사(스레드 종료 수 불일치) 실패 플래그

  // [한국어] 검사 1: 종료 등록된 스레드 수가 CTA 전체 스레드 수와 일치하는지 확인
  if (m_threads_that_have_exited.size() != m_threads_in_cta.size()) {
    printf("\n\n");
    printf(
        "Execution error: Some threads still running in CTA during CTA "
        "reallocation! (1)\n");            // [한국어] CTA 재할당 시 아직 실행 중인 스레드가 있음을 보고
    printf("   CTA uid = %Lu (sm_idx = %u) : %lu running out of %zu total\n",
           m_uid, m_sm_idx,
           (m_threads_in_cta.size() - m_threads_that_have_exited.size()), // [한국어] 아직 실행 중인 스레드 수 = 전체 - 종료된 수
           m_threads_in_cta.size());        // [한국어] 이 CTA의 전체 스레드 수
    printf("   These are the threads that are still running:\n");
    std::set<ptx_thread_info *>::iterator t_iter;
    for (t_iter = m_threads_in_cta.begin(); t_iter != m_threads_in_cta.end();
         ++t_iter) {                        // [한국어] CTA의 모든 스레드를 순회하며 종료하지 않은 스레드를 찾는다
      ptx_thread_info *t = *t_iter;
      if (m_threads_that_have_exited.find(t) ==
          m_threads_that_have_exited.end()) { // [한국어] 이 스레드가 종료 집합에 없으면 아직 실행 중이다
        if (m_dangling_pointers.find(t) != m_dangling_pointers.end()) {
          printf("       <thread deleted>\n"); // [한국어] 이미 삭제된 스레드의 포인터(댕글링 포인터)인 경우 — 메모리 해제 후 포인터가 남아있는 비정상 상황
        } else {
          printf("       [done=%c] : ", (t->is_done() ? 'Y' : 'N')); // [한국어] 스레드의 is_done() 상태를 출력한다
          t->print_insn(t->get_pc(), stdout);  // [한국어] 현재 PC의 PTX 명령어를 출력하여 어디서 멈췄는지 확인
          printf("\n");
        }
      }
    }
    printf("\n\n");
    fail = true;                            // [한국어] 첫 번째 검사 실패 표시
  }
  if (fail) {
    abort();                               // [한국어] 첫 번째 검사 실패 시 즉시 종료 — 시뮬레이션 무결성 위반
  }

  bool fail2 = false;                      // [한국어] 두 번째 검사(개별 is_done() 상태) 실패 플래그
  std::set<ptx_thread_info *>::iterator t_iter;
  for (t_iter = m_threads_in_cta.begin(); t_iter != m_threads_in_cta.end();
       ++t_iter) {                          // [한국어] 각 스레드 객체를 직접 확인하는 이중 검증
    ptx_thread_info *t = *t_iter;
    if (m_dangling_pointers.find(t) == m_dangling_pointers.end()) { // [한국어] 삭제된 포인터는 역참조하지 않고 건너뛴다
      if (!t->is_done()) {                  // [한국어] is_done()이 false이면 스레드가 아직 실행 중이다
        if (!fail2) {
          printf(
              "Execution error: Some threads still running in CTA during CTA "
              "reallocation! (2)\n");       // [한국어] 두 번째 검사에서 미완료 스레드 발견 시 헤더 메시지 출력
          printf("   CTA uid = %Lu (sm_idx = %u) :\n", m_uid, m_sm_idx);
          fail2 = true;                     // [한국어] 헤더 메시지는 한 번만 출력하도록 플래그 설정
        }
        printf("       ");
        t->print_insn(t->get_pc(), stdout); // [한국어] 미완료 스레드의 현재 PTX 명령어 위치를 출력
        printf("\n");
      }
    }
  }
  if (fail2) {
    abort();                               // [한국어] 두 번째 검사 실패 시 즉시 종료
  }

  // [한국어] 두 검사 모두 통과: CTA 상태를 초기화하여 이 객체가 재사용될 준비를 완료한다
  m_threads_in_cta.clear();               // [한국어] CTA 소속 스레드 집합을 비운다
  m_threads_that_have_exited.clear();     // [한국어] 종료 등록 스레드 집합을 비운다
  m_dangling_pointers.clear();            // [한국어] 댕글링 포인터 집합을 비운다
}

/*
 * [한국어]
 * ptx_cta_info::register_thread_exit - 스레드 종료를 CTA에 등록한다
 *
 * @thd : 실행을 마친 ptx_thread_info 포인터.
 *
 * 스레드가 PTX의 exit 명령어를 실행하거나 함수의 끝에 도달하면 이 함수가 호출된다.
 * 동일한 스레드가 두 번 종료 등록되는 것을 assert로 방지한다.
 * m_threads_that_have_exited 집합에 추가하여 check_cta_thread_status_and_reset()의
 * 검증이 가능하도록 한다.
 *
 * 실행 컨텍스트: 호스트 CPU, PTX exit 명령어 실행 시점 (instructions.cc).
 * 호출 체인:
 *   instructions.cc (exit_impl) → ptx_cta_info::register_thread_exit
 */
void ptx_cta_info::register_thread_exit(ptx_thread_info *thd) {
  assert(m_threads_that_have_exited.find(thd) ==
         m_threads_that_have_exited.end()); // [한국어] 같은 스레드가 두 번 종료 등록되는 버그를 방지한다
  m_threads_that_have_exited.insert(thd);   // [한국어] 스레드를 종료 집합에 등록한다
}

/*
 * [한국어]
 * ptx_cta_info::register_deleted_thread - 삭제된 스레드 포인터를 댕글링 집합에 등록한다
 *
 * @thd : 메모리에서 해제된(deleted) ptx_thread_info의 포인터.
 *
 * 스레드 객체가 delete되었지만 m_threads_in_cta 집합에 포인터가 남아있을 때,
 * 해당 포인터를 역참조하면 undefined behavior가 발생한다.
 * 이 함수는 삭제된 포인터를 m_dangling_pointers 집합에 기록하여
 * check_cta_thread_status_and_reset()에서 안전하게 건너뛸 수 있게 한다.
 *
 * 호출 체인:
 *   shader.cc (스레드 객체 소멸 시) → ptx_cta_info::register_deleted_thread
 */
void ptx_cta_info::register_deleted_thread(ptx_thread_info *thd) {
  m_dangling_pointers.insert(thd); // [한국어] 삭제된 포인터를 댕글링 포인터 집합에 등록한다. 이후 이 포인터는 역참조 없이 존재 확인만 한다
}

/*
 * [한국어]
 * ptx_cta_info::get_sm_idx - 이 CTA가 할당된 SM 인덱스를 반환한다
 *
 * @return : m_sm_idx — 생성 시 등록된 SM 인덱스 (0-based).
 *
 * 호출 체인:
 *   shader.cc / gpu-sim.cc (CTA-SM 매핑 확인) → ptx_cta_info::get_sm_idx
 */
unsigned ptx_cta_info::get_sm_idx() const { return m_sm_idx; } // [한국어] 생성자에서 설정된 SM 인덱스를 반환한다

/*
 * [한국어]
 * ptx_cta_info::get_bar_threads - 현재 배리어에 도달한 스레드 수를 반환한다
 *
 * @return : m_bar_threads — __syncthreads() 배리어에 도달한 스레드 수.
 *
 * 이 값이 CTA의 총 스레드 수와 같아질 때 배리어가 해제된다.
 * 호출 체인:
 *   shader.cc (배리어 완료 조건 확인) → ptx_cta_info::get_bar_threads
 */
unsigned ptx_cta_info::get_bar_threads() const { return m_bar_threads; } // [한국어] 배리어 도달 스레드 카운터를 반환한다

/*
 * [한국어]
 * ptx_cta_info::inc_bar_threads - 배리어 도달 스레드 수를 1 증가시킨다
 *
 * __syncthreads() PTX 명령어(bar.sync)를 실행한 스레드가 호출한다.
 * 호출 체인:
 *   instructions.cc (bar_impl) → ptx_cta_info::inc_bar_threads
 */
void ptx_cta_info::inc_bar_threads() { m_bar_threads++; } // [한국어] 배리어 도달 카운터를 원자적이지 않게 증가시킨다. 단일 스레드 시뮬레이션이므로 경쟁 조건이 없다

/*
 * [한국어]
 * ptx_cta_info::reset_bar_threads - 배리어 완료 후 카운터를 0으로 리셋한다
 *
 * 모든 스레드가 배리어에 도달하여 배리어가 해제된 후 카운터를 초기화한다.
 * 다음 __syncthreads() 호출을 위한 준비 단계이다.
 * 호출 체인:
 *   shader.cc (배리어 해제 시) → ptx_cta_info::reset_bar_threads
 */
void ptx_cta_info::reset_bar_threads() { m_bar_threads = 0; } // [한국어] 배리어 카운터를 0으로 초기화한다. 다음 배리어 사이클을 위한 준비이다

/*
 * [한국어]
 * ptx_warp_info::ptx_warp_info - warp 정보 객체 기본 생성자
 *
 * warp 내에서 실행을 완료한 스레드 수(m_done_threads)를 0으로 초기화한다.
 * 호출 체인:
 *   shader.cc (warp 초기화) → ptx_warp_info 생성자
 */
ptx_warp_info::ptx_warp_info() { reset_done_threads(); } // [한국어] 생성자에서 reset_done_threads()를 호출하여 m_done_threads = 0으로 초기화한다

/*
 * [한국어]
 * ptx_warp_info::get_done_threads - 완료된 스레드 수를 반환한다
 *
 * @return : m_done_threads — 이 warp에서 실행을 마친 스레드 수.
 *
 * 호출 체인:
 *   shader.cc (warp 완료 조건 확인) → ptx_warp_info::get_done_threads
 */
unsigned ptx_warp_info::get_done_threads() const { return m_done_threads; } // [한국어] warp 내 완료 스레드 수를 반환한다

/*
 * [한국어]
 * ptx_warp_info::inc_done_threads - 완료 스레드 카운터를 1 증가시킨다
 *
 * 호출 체인:
 *   instructions.cc (exit_impl) → ptx_warp_info::inc_done_threads
 */
void ptx_warp_info::inc_done_threads() { m_done_threads++; } // [한국어] 스레드가 exit할 때마다 호출된다. 단일 스레드 시뮬레이션이므로 원자 연산 불필요

/*
 * [한국어]
 * ptx_warp_info::reset_done_threads - 완료 스레드 카운터를 0으로 초기화한다
 *
 * warp가 새 CTA 스레드들로 재활용될 때 호출된다.
 * 호출 체인:
 *   shader.cc (warp 재초기화) → ptx_warp_info::reset_done_threads
 */
void ptx_warp_info::reset_done_threads() { m_done_threads = 0; } // [한국어] 완료 카운터를 0으로 리셋한다. 이 warp가 새 스레드들을 실행하기 위한 초기화 단계이다

/*
 * [한국어]
 * ptx_thread_info::~ptx_thread_info - PTX 스레드 정보 객체 소멸자
 *
 * 스레드 객체가 delete될 때 전역 삭제 카운터를 증가시킨다.
 * 이 카운터는 디버그 목적으로 시뮬레이션 중 생성/삭제된 스레드 수를 추적한다.
 *
 * 실행 컨텍스트: 호스트 CPU, 스레드 객체 delete 시점.
 * 주의: 소멸자에서 m_gpu를 통해 gpgpu_ctx에 접근하므로, m_gpu가 유효한 상태에서
 *       소멸자가 호출되어야 한다.
 * 호출 체인:
 *   shader.cc (스레드 객체 소멸 시 delete) → ptx_thread_info 소멸자
 */
ptx_thread_info::~ptx_thread_info() {
  m_gpu->gpgpu_ctx->func_sim->g_ptx_thread_info_delete_count++; // [한국어] 전역 스레드 삭제 카운터를 증가시킨다. 생성 카운터(g_ptx_thread_info_uid_next)와 함께 사용하여 메모리 누수를 감지한다
}

/*
 * [한국어]
 * ptx_thread_info::ptx_thread_info - PTX 스레드 상태 객체 생성자
 *
 * @kernel : 이 스레드가 속한 커널의 kernel_info_t 참조.
 *           함수 정보(function_info), 심볼 테이블, PTX 버전 등을 가져오는 데 사용한다.
 *
 * 이 생성자는 하나의 GPU 스레드에 대한 기능 시뮬레이션 상태를 전부 초기화한다.
 * 초기화하는 항목:
 *   - 전역 UID 카운터에서 고유 ID 할당
 *   - 모든 포인터를 NULL로 설정 (m_core, m_shared_mem, m_local_mem 등)
 *   - 실행 상태 플래그 초기화 (m_valid=false, m_thread_done=false 등)
 *   - 레지스터 프레임 스택(m_regs), 콜스택(m_callstack), 디버그 추적 스택 생성
 *   - 복귀 PC(m_RPC=-1), 로컬 메모리 스택 포인터 초기화
 *
 * 실행 컨텍스트: 호스트 CPU, CTA 초기화 단계 (shader.cc의 ptx_sim_init_thread).
 * 호출 체인:
 *   shader.cc (ptx_sim_init_thread) → ptx_thread_info 생성자
 */
ptx_thread_info::ptx_thread_info(kernel_info_t &kernel) : m_kernel(kernel) {
  m_uid = kernel.entry()->gpgpu_ctx->func_sim->g_ptx_thread_info_uid_next++; // [한국어] 전역 UID 카운터에서 이 스레드의 고유 ID를 할당하고 카운터를 증가시킨다. 후위 증가이므로 현재값이 먼저 할당된다
  m_core = NULL;                            // [한국어] 소속 SM(shader_core_ctx) 포인터 — 아직 SM에 배치되지 않았으므로 NULL
  m_barrier_num = -1;                       // [한국어] 현재 대기 중인 배리어 번호 (-1: 배리어 없음). bar.sync 명령어 실행 시 설정된다
  m_at_barrier = false;                     // [한국어] 배리어 대기 중 플래그 — false: 배리어 미대기, true: bar.sync에서 대기 중
  m_valid = false;                          // [한국어] 스레드 상태 유효성 — false: 아직 초기화 미완료. set_info() 호출 후 true가 된다
  m_gridid = 0;                             // [한국어] 이 스레드가 속한 grid ID (커널 launch 고유 번호). 기본값 0
  m_thread_done = false;                    // [한국어] 스레드 실행 완료 플래그 — false: 실행 중, true: exit 명령어 도달
  m_cycle_done = 0;                         // [한국어] 스레드가 완료된 시뮬레이션 사이클 번호. 완료 시 gpu_sim_cycle로 설정
  m_PC = 0;                                 // [한국어] 현재 PTX 프로그램 카운터(PC). set_info()에서 함수 시작 PC로 설정된다
  m_icount = 0;                             // [한국어] 이 스레드가 실행한 명령어 수 누적 카운터 (instruction count)
  m_last_effective_address = 0;             // [한국어] 마지막 메모리 접근의 유효 주소. 디버그 및 메모리 트레이스에 사용
  m_last_memory_space = undefined_space;    // [한국어] 마지막 접근한 메모리 공간 (shared/global/local 등). undefined_space: 아직 메모리 접근 없음
  m_branch_taken = 0;                       // [한국어] 마지막 분기 명령어에서 분기가 취해졌는지 여부 (0: not taken, 1: taken)
  m_shared_mem = NULL;                      // [한국어] 공유 메모리(shared memory) 시뮬레이션 객체 포인터 — SM 배치 시 설정
  m_sstarr_mem = NULL;                      // [한국어] SST(Software Stack Top) 배열 메모리 포인터 — PTXPlus 관련
  m_warp_info = NULL;                       // [한국어] 소속 warp 정보(ptx_warp_info) 포인터 — SM 배치 시 설정
  m_cta_info = NULL;                        // [한국어] 소속 CTA 정보(ptx_cta_info) 포인터 — CTA 초기화 시 설정
  m_local_mem = NULL;                       // [한국어] 로컬 메모리(레지스터 스필 공간) 시뮬레이션 객체 포인터 — SM 배치 시 설정
  m_symbol_table = NULL;                    // [한국어] 현재 실행 중인 함수의 심볼 테이블 포인터 — set_info() 또는 callstack_push()에서 설정
  m_func_info = NULL;                       // [한국어] 현재 실행 중인 함수의 PTX IR 정보(function_info) 포인터 — set_info()에서 설정
  m_hw_tid = -1;                            // [한국어] 하드웨어 스레드 ID (SM 내 물리적 슬롯 번호). -1: 아직 배치 전
  m_hw_wid = -1;                            // [한국어] 하드웨어 warp ID (SM 내 물리적 warp 슬롯 번호). -1: 아직 배치 전
  m_hw_sid = -1;                            // [한국어] 하드웨어 SM ID (시뮬레이터 SM 인덱스). -1: 아직 배치 전
  m_last_dram_callback.function = NULL;     // [한국어] DRAM 접근 완료 콜백 함수 포인터 초기화 — 아직 등록된 콜백 없음
  m_last_dram_callback.instruction = NULL;  // [한국어] 콜백 대상 PTX 명령어 포인터 초기화 — NULL: 콜백 없음
  m_regs.push_back(reg_map_t());            // [한국어] 첫 번째 레지스터 프레임을 생성한다. 함수 호출 시 callstack_push()에서 추가 프레임이 쌓인다
  m_debug_trace_regs_modified.push_back(reg_map_t()); // [한국어] 명령어별 수정된 레지스터 추적용 맵의 첫 프레임 생성. 디버그 모드에서 레지스터 변화를 기록한다
  m_debug_trace_regs_read.push_back(reg_map_t());     // [한국어] 명령어별 읽힌 레지스터 추적용 맵의 첫 프레임 생성. 디버그 모드에서 소스 오퍼랜드 접근을 기록한다
  m_callstack.push_back(stack_entry());     // [한국어] 콜스택의 첫 번째 엔트리(최하위 프레임)를 생성한다. 기본 생성 stack_entry는 m_valid=false인 진입점 프레임이다
  m_RPC = -1;                               // [한국어] 복귀 PC(Return PC) — -1: 현재 복귀 주소 없음(최상위 커널 함수에서는 복귀 불필요)
  m_RPC_updated = false;                    // [한국어] RPC 업데이트 플래그 — false: RPC가 최근에 변경되지 않음
  m_last_was_call = false;                  // [한국어] 직전 명령어가 함수 호출이었는지 여부 — callstack_push() 후 true가 됨
  m_enable_debug_trace = false;             // [한국어] 디버그 레지스터 추적 활성화 플래그 — 기본값 false (비활성)
  m_local_mem_stack_pointer = 0;            // [한국어] 로컬 메모리(스필 공간) 스택 포인터 — 함수 호출 시 framesize만큼 증가하고 반환 시 감소
  m_gpu = NULL;                             // [한국어] gpgpu_sim 포인터 — SM 배치 후 shader.cc에서 설정된다
  m_last_set_operand_value = ptx_reg_t();   // [한국어] 마지막으로 설정된 오퍼랜드 값 — 기본 초기화 (모든 필드 0)
}

/*
 * [한국어]
 * ptx_thread_info::get_ptx_version - 현재 실행 중인 함수의 PTX ISA 버전을 반환한다
 *
 * @return : m_func_info->get_ptx_version() — PTX 버전 정보 참조.
 *
 * PTX 버전에 따라 일부 명령어의 시맨틱이 다르며, instructions.cc에서 버전 분기가 필요할 때 사용.
 * 호출 체인:
 *   instructions.cc (명령어 버전 분기) → ptx_thread_info::get_ptx_version
 */
const ptx_version &ptx_thread_info::get_ptx_version() const {
  return m_func_info->get_ptx_version(); // [한국어] 현재 실행 중인 함수(m_func_info)의 PTX 버전 정보를 반환한다
}

/*
 * [한국어]
 * ptx_thread_info::set_done - 스레드 실행 완료를 표시한다
 *
 * PTX의 exit 명령어 실행 시 호출되어 스레드를 완료 상태로 전환한다.
 * m_at_barrier가 true인 상태(배리어 대기 중)에서는 exit할 수 없으므로 assert로 검증.
 * m_cycle_done에 현재 시뮬레이션 사이클을 기록하여 종료 시점을 추적한다.
 *
 * 실행 컨텍스트: 호스트 CPU, instructions.cc의 exit_impl() 내부.
 * 호출 체인:
 *   instructions.cc (exit_impl) → ptx_thread_info::set_done
 */
void ptx_thread_info::set_done() {
  assert(!m_at_barrier);                   // [한국어] 배리어 대기 중인 스레드가 exit하는 것은 논리 오류 — 배리어 완료 전 exit 불가
  m_thread_done = true;                    // [한국어] 스레드 완료 플래그를 true로 설정한다. 이후 is_done()은 true를 반환한다
  m_cycle_done = m_gpu->gpu_sim_cycle;     // [한국어] 스레드 완료 시점의 시뮬레이션 사이클을 기록한다. 성능 분석(스레드별 실행 시간)에 활용 가능
}

/*
 * [한국어]
 * ptx_thread_info::get_builtin - PTX 특수 레지스터의 현재 값을 반환한다
 *
 * @builtin_id : 특수 레지스터 종류 (opcodes.h의 special_regs enum 값).
 *               하위 16비트가 레지스터 종류, 상위 16비트가 배열 인덱스(예: %envreg<n>의 n).
 * @dim_mod    : 차원 수식어 (0=x, 1=y, 2=z). %tid, %ctaid, %ntid, %nctaid 등에 사용.
 * @return     : 해당 특수 레지스터의 현재 값 (unsigned).
 *
 * PTX ISA는 %tid, %ctaid, %smid 등의 읽기 전용 특수 레지스터를 정의하며,
 * 이 함수는 instructions.cc의 operand fetch 단계에서 해당 레지스터 값을 CPU 상태에서 조합한다.
 * 각 레지스터의 의미는 opcodes.h의 special_regs 주석 참조.
 * 미구현 레지스터(%lanemask_*, %nwarpid, %pm, %smid)는 feature_not_implemented()로 abort.
 *
 * 실행 컨텍스트: 호스트 CPU, PTX 명령어 오퍼랜드 fetch 단계 (instructions.cc).
 * 에러 경로: 미구현 레지스터 접근 → feature_not_implemented() → printf + abort().
 *
 * 호출 체인:
 *   instructions.cc (get_operand_value_로 시작하는 함수) → ptx_thread_info::get_builtin
 */
unsigned ptx_thread_info::get_builtin(int builtin_id, unsigned dim_mod) {
  assert(m_valid);                         // [한국어] 스레드 상태가 유효한지 확인한다. m_valid=false이면 set_info()가 호출되지 않은 상태로 특수 레지스터 접근이 불가하다
  switch ((builtin_id & 0xFFFF)) {         // [한국어] 하위 16비트에서 레지스터 종류를 추출한다. 상위 16비트는 %envreg<n>의 n 인덱스 등에 사용된다
    case CLOCK_REG:
      // [한국어] %clock: 32비트 시뮬레이션 사이클 카운터. 워밍업 사이클(gpu_tot_sim_cycle)을 포함한다
      return (unsigned)(m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    case CLOCK64_REG:
      // [한국어] %clock64: 64비트 확장 클럭. 현재 구현되지 않아 abort() 후 fallthrough
      abort();  // change return value to unsigned long long?
                // GPGPUSim clock is 4 times slower - multiply by 4
      return (m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) * 4; // [한국어] 실제 GPU 클럭 대비 4배 보정 — GPGPU-Sim의 시뮬레이션 클럭이 실제 GPU 클럭보다 4배 느리기 때문
    case HALFCLOCK_ID:
      // GPGPUSim clock is 4 times slower - multiply by 4
      // Hardware clock counter is incremented at half the shader clock
      // frequency - divide by 2 (Henry '10)
      // [한국어] %halfclock: 실제 HW는 셰이더 클럭의 절반 속도로 증가. GPGPU-Sim은 4배 느리므로 4/2=2를 곱해 보정
      return (m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) * 2;
    case CTAID_REG:
      // [한국어] %ctaid.{x,y,z}: 그리드 내 이 CTA의 좌표. dim_mod(0,1,2)로 x/y/z 선택
      assert(dim_mod < 3);                 // [한국어] 차원 수식어는 0,1,2(x,y,z)만 유효하다
      if (dim_mod == 0) return m_ctaid.x;  // [한국어] %ctaid.x: blockIdx.x에 해당하는 CTA x좌표
      if (dim_mod == 1) return m_ctaid.y;  // [한국어] %ctaid.y: blockIdx.y에 해당하는 CTA y좌표
      if (dim_mod == 2) return m_ctaid.z;  // [한국어] %ctaid.z: blockIdx.z에 해당하는 CTA z좌표
      abort();                             // [한국어] dim_mod가 3 이상이면 도달 불가 — abort()
      break;
    case ENVREG_REG: {                     // [한국어] %envreg<n>: 환경 레지스터. 상위 16비트에서 n 인덱스를 추출한다
      int index = builtin_id >> 16;        // [한국어] builtin_id 상위 16비트에서 환경 레지스터 번호(0~31)를 추출한다
      dim3 gdim = this->get_core()->get_kernel_info()->get_grid_dim(); // [한국어] 현재 커널의 그리드 차원(gridDim)을 가져온다
      switch (index) {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
          return 0;                        // [한국어] envreg 0~5: GPGPU-Sim에서 정의되지 않은 환경 레지스터 — 0을 반환한다
          break;
        case 6:
          return gdim.x;                   // [한국어] envreg 6: gridDim.x (그리드의 x방향 블록 수)
        case 7:
          return gdim.y;                   // [한국어] envreg 7: gridDim.y (그리드의 y방향 블록 수)
        case 8:
          return gdim.z;                   // [한국어] envreg 8: gridDim.z (그리드의 z방향 블록 수)
        case 9:
          // [한국어] envreg 9: 그리드 차원 수 (1D=1, 2D=2, 3D=3)
          if (gdim.z == 1 && gdim.y == 1)
            return 1;                      // [한국어] z=1, y=1이면 1D 그리드
          else if (gdim.z == 1)
            return 2;                      // [한국어] z=1이면 2D 그리드
          else
            return 3;                      // [한국어] z!=1이면 3D 그리드
          break;
        default:
          break;                           // [한국어] 나머지 envreg 번호: 현재 미구현, 0으로 낙하(fall-through 후 switch 블록 종료)
      }
    }
    case GRIDID_REG:
      return m_gridid;                     // [한국어] %gridid: 이 커널 launch의 고유 ID를 반환한다. cuLaunchKernel마다 증가하는 정수
    case LANEID_REG:
      return get_hw_tid() % m_core->get_warp_size(); // [한국어] %laneid: warp 내 스레드 위치. HW 스레드 ID를 warp 크기로 나눈 나머지가 lane 번호이다
    case LANEMASK_EQ_REG:
      feature_not_implemented("%lanemask_eq"); // [한국어] %lanemask_eq: 미구현 — printf 출력 후 abort()
      return 0;
    case LANEMASK_LE_REG:
      feature_not_implemented("%lanemask_le"); // [한국어] %lanemask_le: 미구현
      return 0;
    case LANEMASK_LT_REG:
      feature_not_implemented("%lanemask_lt"); // [한국어] %lanemask_lt: 미구현
      return 0;
    case LANEMASK_GE_REG:
      feature_not_implemented("%lanemask_ge"); // [한국어] %lanemask_ge: 미구현
      return 0;
    case LANEMASK_GT_REG:
      feature_not_implemented("%lanemask_gt"); // [한국어] %lanemask_gt: 미구현
      return 0;
    case NCTAID_REG:
      // [한국어] %nctaid.{x,y,z}: 그리드의 블록 수 (gridDim.{x,y,z}에 해당)
      assert(dim_mod < 3);
      if (dim_mod == 0) return m_nctaid.x; // [한국어] %nctaid.x: gridDim.x
      if (dim_mod == 1) return m_nctaid.y; // [한국어] %nctaid.y: gridDim.y
      if (dim_mod == 2) return m_nctaid.z; // [한국어] %nctaid.z: gridDim.z
      abort();
      break;
    case NTID_REG:
      // [한국어] %ntid.{x,y,z}: 블록 내 스레드 수 (blockDim.{x,y,z}에 해당)
      assert(dim_mod < 3);
      if (dim_mod == 0) return m_ntid.x;   // [한국어] %ntid.x: blockDim.x
      if (dim_mod == 1) return m_ntid.y;   // [한국어] %ntid.y: blockDim.y
      if (dim_mod == 2) return m_ntid.z;   // [한국어] %ntid.z: blockDim.z
      abort();
      break;
    case NWARPID_REG:
      feature_not_implemented("%nwarpid"); // [한국어] %nwarpid: SM의 최대 warp 수 — 미구현
      return 0;
    case PM_REG:
      feature_not_implemented("%pm");      // [한국어] %pm<n>: 성능 모니터 카운터 — 미구현
      return 0;
    case SMID_REG:
      feature_not_implemented("%smid");    // [한국어] %smid: 현재 SM ID — 미구현. 실제로는 m_hw_sid를 반환해야 하나 아직 구현되지 않음
      return 0;
    case TID_REG:
      // [한국어] %tid.{x,y,z}: 블록 내 스레드 ID (threadIdx.{x,y,z}에 해당)
      assert(dim_mod < 3);
      if (dim_mod == 0) return m_tid.x;    // [한국어] %tid.x: threadIdx.x
      if (dim_mod == 1) return m_tid.y;    // [한국어] %tid.y: threadIdx.y
      if (dim_mod == 2) return m_tid.z;    // [한국어] %tid.z: threadIdx.z
      abort();
      break;
    case WARPSZ_REG:
      return m_core->get_warp_size();      // [한국어] %WARP_SZ: warp 크기 (항상 32). SM의 warp 크기 설정값을 반환한다
    default:
      assert(0);                           // [한국어] 알 수 없는 특수 레지스터 ID — 시뮬레이터 내부 버그이므로 assert로 종료
  }
  return 0;                                // [한국어] 컴파일러 경고 억제용 unreachable return
}

/*
 * [한국어]
 * ptx_thread_info::set_info - 스레드를 특정 PTX 함수 실행을 위해 초기화한다
 *
 * @func : 실행할 PTX 함수의 function_info 포인터.
 *         심볼 테이블, 시작 PC, 함수 메타데이터를 포함한다.
 *
 * 생성자에서 NULL로 초기화된 m_symbol_table, m_func_info를 실제 함수 정보로 연결하고,
 * m_PC를 함수의 시작 PC로 설정한다. 이 함수 호출 후 m_valid=true가 되어 get_builtin()이
 * 올바르게 동작할 수 있다.
 *
 * 실행 컨텍스트: 호스트 CPU, 스레드 초기화 단계 (shader.cc의 ptx_sim_init_thread).
 * 호출 체인:
 *   shader.cc (ptx_sim_init_thread) → ptx_thread_info::set_info
 */
void ptx_thread_info::set_info(function_info *func) {
  m_symbol_table = func->get_symtab();    // [한국어] 함수의 심볼 테이블을 연결한다. 레지스터 이름→값 매핑, 타입 정보 조회에 사용
  m_func_info = func;                     // [한국어] 실행할 함수의 PTX IR 정보를 연결한다
  m_PC = func->get_start_PC();            // [한국어] 프로그램 카운터를 함수의 첫 번째 명령어 주소로 설정한다
}

/*
 * [한국어]
 * ptx_thread_info::cpy_tid_to_reg - %tid를 구형 PTX의 $r0 레지스터에 복사한다
 *
 * @tid : 복사할 3D 스레드 ID (x/y/z 각 성분).
 *
 * 구형 PTX (1.x 시절)에서는 %tid 특수 레지스터 대신 $r0 범용 레지스터에
 * tid.x + (tid.y<<16) + (tid.z<<26) 형태로 스레드 ID를 인코딩하여 저장했다.
 * 커널이 $r0을 사용하지 않으면 심볼 테이블 조회에서 NULL이 반환되어 건너뛴다.
 *
 * 실행 컨텍스트: 호스트 CPU, CTA 초기화 단계.
 * 호출 체인:
 *   shader.cc (ptx_sim_init_thread) → ptx_thread_info::cpy_tid_to_reg
 */
void ptx_thread_info::cpy_tid_to_reg(dim3 tid) {
  // copies %tid.x, %tid.y and %tid.z into $r0
  ptx_reg_t data;                          // [한국어] 레지스터 값 컨테이너 — 여러 타입(s32, u32, f32 등)을 union으로 보유
  data.s64 = 0;                            // [한국어] 64비트 전체를 0으로 초기화하여 상위 비트 쓰레기값을 제거한다

  data.u32 = (tid.x + (tid.y << 16) + (tid.z << 26)); // [한국어] tid.x를 비트 0~15, tid.y를 비트 16~25, tid.z를 비트 26~31에 패킹한다. 구형 PTX $r0 인코딩 방식

  const symbol *r0 = m_symbol_table->lookup("$r0"); // [한국어] 심볼 테이블에서 "$r0" 심볼을 조회한다. 커널이 $r0을 참조하지 않으면 NULL이 반환된다
  if (r0) {
    // No need to set pid if kernel doesn't use it
    set_reg(r0, data);                     // [한국어] $r0이 존재하면 인코딩된 tid 값을 레지스터 파일에 쓴다
  }
}

/*
 * [한국어]
 * ptx_thread_info::print_insn - 지정된 PC의 PTX 명령어를 파일에 출력한다
 *
 * @pc : 출력할 PTX 명령어의 프로그램 카운터 주소.
 * @fp : 출력 대상 파일 스트림 (보통 stdout).
 *
 * 디버그 출력 시 어떤 명령어에서 스레드가 멈췄는지 확인하는 데 사용된다.
 * 실제 출력은 function_info::print_insn()에 위임한다.
 * 호출 체인:
 *   ptx_cta_info::check_cta_thread_status_and_reset → ptx_thread_info::print_insn
 */
void ptx_thread_info::print_insn(unsigned pc, FILE *fp) const {
  m_func_info->print_insn(pc, fp); // [한국어] 현재 함수(m_func_info)의 PC에 해당하는 PTX 명령어를 fp에 출력한다
}

/*
 * [한국어]
 * print_reg (static) - 심볼 이름과 값을 타입에 맞게 파일에 출력하는 내부 헬퍼 함수
 *
 * @fp     : 출력 대상 파일 스트림.
 * @name   : 레지스터/심볼 이름 문자열 (예: "%r0", "%f1").
 * @value  : 레지스터 값 (ptx_reg_t union — 여러 타입을 보유하는 공용체).
 * @symtab : 심볼 테이블 포인터 — 이름으로 타입 정보를 조회하는 데 사용.
 *
 * 심볼 테이블에서 해당 이름의 타입을 조회하고, 타입에 따라 적절한 형식으로 값을 출력한다.
 * 타입 불명시 시 16진수 64비트로 출력한다.
 * 이 함수는 파일 범위(static) 함수로, dump_regs()와 dump_modifiedregs()에서만 사용된다.
 *
 * 실행 컨텍스트: 호스트 CPU, 디버그 출력 시점.
 * 호출 체인:
 *   dump_regs / dump_modifiedregs → print_reg(fp, ...) → fprintf
 */
static void print_reg(FILE *fp, std::string name, ptx_reg_t value,
                      symbol_table *symtab) {
  const symbol *sym = symtab->lookup(name.c_str()); // [한국어] 심볼 테이블에서 이름으로 심볼 객체를 조회한다
  fprintf(fp, "  %8s   ", name.c_str());   // [한국어] 레지스터 이름을 8칸 폭으로 우측 정렬하여 출력한다
  if (sym == NULL) {                        // [한국어] 심볼을 찾지 못한 경우 — 정의되지 않은 레지스터
    fprintf(fp, "<unknown type> 0x%llx\n", (unsigned long long)value.u64); // [한국어] 타입 불명이므로 64비트 16진수로 원시 비트 패턴을 출력한다
    return;
  }
  const type_info *t = sym->type();        // [한국어] 심볼의 타입 정보를 가져온다 (PTX 타입: s32, f32, u64 등)
  if (t == NULL) {                          // [한국어] 타입 정보가 없는 경우
    fprintf(fp, "<unknown type> 0x%llx\n", (unsigned long long)value.u64); // [한국어] 타입 불명이므로 64비트 16진수로 출력한다
    return;
  }
  type_info_key ti = t->get_key();          // [한국어] 타입 키 추출 — scalar_type(), vector_type() 등 분류 정보를 포함한다

  switch (ti.scalar_type()) {               // [한국어] 스칼라 타입에 따라 적절한 출력 포맷을 선택한다
    case S8_TYPE:
      fprintf(fp, ".s8  %d\n", value.s8);   // [한국어] 8비트 부호있는 정수로 출력
      break;
    case S16_TYPE:
      fprintf(fp, ".s16 %d\n", value.s16);  // [한국어] 16비트 부호있는 정수로 출력
      break;
    case S32_TYPE:
      fprintf(fp, ".s32 %d\n", value.s32);  // [한국어] 32비트 부호있는 정수로 출력 (가장 흔한 정수 레지스터 타입)
      break;
    case S64_TYPE:
      fprintf(fp, ".s64 %Ld\n", value.s64); // [한국어] 64비트 부호있는 정수로 출력 (%Ld: long long 포맷)
      break;
    case U8_TYPE:
      fprintf(fp, ".u8  %u [0x%02x]\n", value.u8, (unsigned)value.u8); // [한국어] 8비트 부호없는 정수를 10진수와 16진수로 출력
      break;
    case U16_TYPE:
      fprintf(fp, ".u16 %u [0x%04x]\n", value.u16, (unsigned)value.u16); // [한국어] 16비트 부호없는 정수를 10진수와 16진수로 출력
      break;
    case U32_TYPE:
      fprintf(fp, ".u32 %u [0x%08x]\n", value.u32, (unsigned)value.u32); // [한국어] 32비트 부호없는 정수를 10진수와 8자리 16진수로 출력
      break;
    case U64_TYPE:
      fprintf(fp, ".u64 %llu [0x%llx]\n", value.u64, value.u64); // [한국어] 64비트 부호없는 정수를 10진수와 16진수로 출력
      break;
    case F16_TYPE:
      fprintf(fp, ".f16 %f [0x%04x]\n", static_cast<float>(value.f16),
              (unsigned)value.u16);         // [한국어] 16비트 반정밀도 float을 32비트로 변환하여 출력하고, 원시 16비트 값도 병기한다
      break;
    case F32_TYPE:
      fprintf(fp, ".f32 %.15lf [0x%08x]\n", value.f32, value.u32); // [한국어] 32비트 단정밀도 float을 15자리 소수로 출력하고, 원시 비트 패턴(IEEE 754)도 병기한다
      break;
    case F64_TYPE:
      fprintf(fp, ".f64 %.15le [0x%016llx]\n", value.f64, value.u64); // [한국어] 64비트 배정밀도 float을 15자리 과학 표기로 출력하고, 원시 비트 패턴도 병기한다
      break;
    case B8_TYPE:
      fprintf(fp, ".b8  0x%02x\n", (unsigned)value.u8); // [한국어] 8비트 비트 패턴을 16진수로 출력
      break;
    case B16_TYPE:
      fprintf(fp, ".b16 0x%04x\n", (unsigned)value.u16); // [한국어] 16비트 비트 패턴을 16진수로 출력
      break;
    case B32_TYPE:
      fprintf(fp, ".b32 0x%08x\n", (unsigned)value.u32); // [한국어] 32비트 비트 패턴을 8자리 16진수로 출력
      break;
    case B64_TYPE:
      fprintf(fp, ".b64 0x%llx\n", (unsigned long long)value.u64); // [한국어] 64비트 비트 패턴을 16진수로 출력
      break;
    case PRED_TYPE:
      fprintf(fp, ".pred %u\n", (unsigned)value.pred); // [한국어] 서술자(predicate) 레지스터 값을 0 또는 1로 출력
      break;
    default:
      fprintf(fp, "non-scalar type\n");    // [한국어] 벡터 타입 등 비스칼라 타입은 아직 출력 미구현
      break;
  }
  fflush(fp);                              // [한국어] 파일 버퍼를 즉시 비운다. 디버그 출력이 프로그램 abort() 전에 화면에 표시되도록 보장한다
}

/*
 * [한국어]
 * print_reg (static, stdout 버전) - stdout에 레지스터를 출력하는 편의 래퍼
 *
 * @name   : 레지스터 이름 문자열.
 * @value  : 레지스터 값.
 * @symtab : 심볼 테이블.
 *
 * print_reg(fp, ...) 의 stdout 특화 버전이다. fp=stdout으로 고정하여 호출한다.
 * 호출 체인:
 *   dump_modifiedregs → print_reg(name, value, symtab) → print_reg(stdout, ...)
 */
static void print_reg(std::string name, ptx_reg_t value, symbol_table *symtab) {
  print_reg(stdout, name, value, symtab); // [한국어] fp를 stdout으로 고정하여 파일 포인터 버전을 호출한다
}

/*
 * [한국어]
 * ptx_thread_info::callstack_push - 함수 호출 시 콜스택 프레임을 추가한다
 *
 * @pc             : 호출 명령어 다음의 복귀 주소 (반환 시 실행을 재개할 PC).
 * @rpc            : 재귀 복귀 PC (중첩 호출 지원용).
 * @return_var_src : 피호출 함수(callee)에서 반환값을 읽을 심볼 (없으면 NULL).
 * @return_var_dst : 호출 함수(caller)에서 반환값을 쓸 심볼 (없으면 NULL).
 * @call_uid       : 이 함수 호출의 고유 ID (디버그 추적용).
 *
 * PTX call 명령어 실행 시 호출되어:
 *   1. m_RPC를 -1로 재설정하고 m_RPC_updated=true, m_last_was_call=true로 설정
 *   2. 현재 함수 정보(심볼 테이블, function_info)와 복귀 정보를 콜스택에 쌓는다
 *   3. 새 레지스터 프레임(reg_map_t)과 디버그 추적 프레임을 각 스택에 추가한다
 *   4. 로컬 메모리 스택 포인터를 호출된 함수의 프레임 크기만큼 전진시킨다
 *
 * 실행 컨텍스트: 호스트 CPU, PTX call 명령어 실행 시점 (instructions.cc).
 * 호출 체인:
 *   instructions.cc (call_impl) → ptx_thread_info::callstack_push
 */
void ptx_thread_info::callstack_push(unsigned pc, unsigned rpc,
                                     const symbol *return_var_src,
                                     const symbol *return_var_dst,
                                     unsigned call_uid) {
  m_RPC = -1;                              // [한국어] 복귀 PC를 일시적으로 무효화한다. 피호출 함수 실행 중에는 복귀 PC가 필요 없다
  m_RPC_updated = true;                    // [한국어] RPC가 업데이트됨을 표시하여 타이밍 모델(shader.cc)이 RPC 변경을 인지할 수 있도록 한다
  m_last_was_call = true;                  // [한국어] 직전 명령어가 call이었음을 표시한다. SIMT 스택 업데이트에 사용된다
  assert(m_func_info != NULL);             // [한국어] 현재 실행 중인 함수가 유효해야 한다 — NULL이면 콜스택 저장 불가
  m_callstack.push_back(stack_entry(m_symbol_table, m_func_info, pc, rpc,
                                    return_var_src, return_var_dst, call_uid)); // [한국어] 현재 함수의 심볼 테이블, function_info, 복귀 PC, 반환값 심볼 정보를 콜스택에 저장한다
  m_regs.push_back(reg_map_t());           // [한국어] 새 함수를 위한 빈 레지스터 프레임을 추가한다. 피호출 함수의 지역 레지스터가 이 프레임에 저장된다
  m_debug_trace_regs_modified.push_back(reg_map_t()); // [한국어] 피호출 함수용 수정 레지스터 추적 프레임 추가
  m_debug_trace_regs_read.push_back(reg_map_t());     // [한국어] 피호출 함수용 읽기 레지스터 추적 프레임 추가
  m_local_mem_stack_pointer += m_func_info->local_mem_framesize(); // [한국어] 피호출 함수의 로컬 메모리 프레임 크기만큼 스택 포인터를 전진시킨다. 레지스터 스필 공간을 확보하는 효과
}

/*
 * [한국어]
 * ptx_thread_info::callstack_push_plus - PTXPlus용 함수 호출 콜스택 프레임 추가
 *
 * @pc             : 복귀 주소.
 * @rpc            : 재귀 복귀 PC.
 * @return_var_src : 반환값 소스 심볼.
 * @return_var_dst : 반환값 목적지 심볼.
 * @call_uid       : 호출 고유 ID.
 *
 * callstack_push()와 유사하지만 PTXPlus(SASS와 유사한 하위 레벨 IR)를 위한 버전이다.
 * 중요한 차이: PTXPlus는 레지스터 파일을 함수 호출 간에 공유하므로
 * m_regs, m_debug_trace_regs_modified, m_debug_trace_regs_read에 새 프레임을
 * 추가하지 않는다 (주석 처리된 라인 참조).
 *
 * 실행 컨텍스트: 호스트 CPU, PTXPlus call 명령어 실행 시점.
 * 호출 체인:
 *   instructions.cc (callp_impl 등) → ptx_thread_info::callstack_push_plus
 */
// ptxplus version of callstack_push.
void ptx_thread_info::callstack_push_plus(unsigned pc, unsigned rpc,
                                          const symbol *return_var_src,
                                          const symbol *return_var_dst,
                                          unsigned call_uid) {
  m_RPC = -1;                              // [한국어] 복귀 PC 무효화
  m_RPC_updated = true;                    // [한국어] RPC 업데이트 플래그 설정
  m_last_was_call = true;                  // [한국어] call 직후임을 표시
  assert(m_func_info != NULL);             // [한국어] 현재 함수 유효성 검사
  m_callstack.push_back(stack_entry(m_symbol_table, m_func_info, pc, rpc,
                                    return_var_src, return_var_dst, call_uid)); // [한국어] 콜스택에 현재 함수 컨텍스트를 저장한다
  // m_regs.push_back( reg_map_t() );
  // [한국어] PTXPlus는 레지스터 파일을 호출자/피호출자가 공유하므로 새 레지스터 프레임을 생성하지 않는다
  // m_debug_trace_regs_modified.push_back( reg_map_t() );
  // [한국어] PTXPlus에서는 디버그 추적 프레임도 추가하지 않는다
  // m_debug_trace_regs_read.push_back( reg_map_t() );
  m_local_mem_stack_pointer += m_func_info->local_mem_framesize(); // [한국어] 로컬 메모리 스택 포인터는 PTXPlus에서도 전진시킨다
}

/*
 * [한국어]
 * ptx_thread_info::callstack_pop - 함수 반환 시 콜스택 프레임을 제거한다
 *
 * @return : 콜스택이 비어 있으면 true (최상위 커널 함수로 복귀), 아니면 false.
 *
 * PTX ret 명령어 실행 시 호출되어:
 *   1. 반환값을 피호출 함수 레지스터 프레임에서 arg_buffer로 읽는다 (rv_src 존재 시)
 *   2. 콜스택 최상위 엔트리에서 호출자 함수 정보(심볼 테이블, function_info, PC, RPC)를 복원한다
 *   3. 로컬 메모리 스택 포인터를 피호출 함수 프레임 크기만큼 되돌린다
 *   4. 콜스택, 레지스터 프레임, 디버그 추적 프레임을 pop한다
 *   5. 반환값을 호출자 레지스터 프레임에 쓴다 (rv_dst 존재 시)
 *
 * 실행 컨텍스트: 호스트 CPU, PTX ret 명령어 실행 시점 (instructions.cc).
 * 에러 경로: rv_src와 rv_dst 중 하나만 NULL이면 assert 실패 (호출자/피호출자 반환값 불일치).
 * 호출 체인:
 *   instructions.cc (ret_impl) → ptx_thread_info::callstack_pop
 */
bool ptx_thread_info::callstack_pop() {
  const symbol *rv_src = m_callstack.back().m_return_var_src; // [한국어] 피호출 함수에서 반환값을 읽을 심볼 (callee에서의 반환 레지스터)
  const symbol *rv_dst = m_callstack.back().m_return_var_dst; // [한국어] 호출자 함수에서 반환값을 쓸 심볼 (caller에서의 결과 레지스터)
  assert(!((rv_src != NULL) ^
           (rv_dst != NULL)));  // ensure caller and callee agree on whether
                                // there is a return value
  // [한국어] rv_src와 rv_dst 중 정확히 하나만 NULL이면 XOR이 true → 호출자/피호출자 반환값 정보 불일치 → assert 실패

  // read return value from callee frame
  arg_buffer_t buffer(m_gpu->gpgpu_ctx);   // [한국어] 반환값을 임시 저장할 인수 버퍼를 생성한다. gpgpu_context가 메모리 할당자를 제공한다
  if (rv_src != NULL)
    buffer = copy_arg_to_buffer(this, operand_info(rv_src, m_gpu->gpgpu_ctx),
                                rv_dst);    // [한국어] 피호출 함수의 반환 레지스터(rv_src)에서 호출자의 대상 심볼(rv_dst) 형식에 맞게 값을 버퍼로 복사한다

  m_symbol_table = m_callstack.back().m_symbol_table; // [한국어] 호출자 함수의 심볼 테이블을 복원한다
  m_NPC = m_callstack.back().m_PC;          // [한국어] 다음 실행 PC를 콜스택에서 복원한다 (call 다음 명령어 주소)
  m_RPC_updated = true;                     // [한국어] RPC가 복원됨을 표시한다
  m_last_was_call = false;                  // [한국어] 직전 명령어가 ret이었음을 표시한다
  m_RPC = m_callstack.back().m_RPC;         // [한국어] 재귀 복귀 PC를 콜스택에서 복원한다
  m_func_info = m_callstack.back().m_func_info; // [한국어] 호출자 함수의 function_info를 복원한다
  if (m_func_info) {
    assert(m_local_mem_stack_pointer >= m_func_info->local_mem_framesize()); // [한국어] 스택 포인터가 줄어들 공간이 있는지 검증한다 — 언더플로우 방지
    m_local_mem_stack_pointer -= m_func_info->local_mem_framesize(); // [한국어] 피호출 함수의 로컬 메모리 프레임을 해제한다 (스택 포인터를 되돌림)
  }
  m_callstack.pop_back();                   // [한국어] 콜스택에서 피호출 함수 엔트리를 제거한다
  m_regs.pop_back();                        // [한국어] 피호출 함수의 레지스터 프레임을 제거한다
  m_debug_trace_regs_modified.pop_back();   // [한국어] 피호출 함수의 수정 레지스터 추적 프레임을 제거한다
  m_debug_trace_regs_read.pop_back();       // [한국어] 피호출 함수의 읽기 레지스터 추적 프레임을 제거한다

  // write return value into caller frame
  if (rv_dst != NULL) copy_buffer_to_frame(this, buffer); // [한국어] 버퍼에 저장한 반환값을 호출자 레지스터 프레임의 rv_dst 심볼에 쓴다

  return m_callstack.empty();               // [한국어] 콜스택이 비어 있으면 true 반환 — 최상위 커널 함수로 복귀했음을 의미하며, 스레드 실행이 완료된다
}

/*
 * [한국어]
 * ptx_thread_info::callstack_pop_plus - PTXPlus용 함수 반환 시 콜스택 프레임 제거
 *
 * @return : 콜스택이 비어 있으면 true.
 *
 * callstack_pop()과 유사하지만 PTXPlus용이다.
 * PTXPlus는 레지스터 파일을 공유하므로 m_regs, m_debug_trace_regs_* 스택을
 * pop하지 않는다 (주석 처리된 라인 참조).
 *
 * 실행 컨텍스트: 호스트 CPU, PTXPlus ret 명령어 실행 시점.
 * 호출 체인:
 *   instructions.cc (retp_impl 등) → ptx_thread_info::callstack_pop_plus
 */
// ptxplus version of callstack_pop
bool ptx_thread_info::callstack_pop_plus() {
  const symbol *rv_src = m_callstack.back().m_return_var_src; // [한국어] 피호출 함수의 반환값 소스 심볼
  const symbol *rv_dst = m_callstack.back().m_return_var_dst; // [한국어] 호출자의 반환값 목적지 심볼
  assert(!((rv_src != NULL) ^
           (rv_dst != NULL)));  // ensure caller and callee agree on whether
                                // there is a return value
  // [한국어] 반환값 유무 일관성 검사 — callstack_pop과 동일

  // read return value from callee frame
  arg_buffer_t buffer(m_gpu->gpgpu_ctx);   // [한국어] 반환값 임시 버퍼 생성
  if (rv_src != NULL)
    buffer = copy_arg_to_buffer(this, operand_info(rv_src, m_gpu->gpgpu_ctx),
                                rv_dst);    // [한국어] 피호출 함수의 반환값을 버퍼로 복사

  m_symbol_table = m_callstack.back().m_symbol_table; // [한국어] 호출자 심볼 테이블 복원
  m_NPC = m_callstack.back().m_PC;          // [한국어] 복귀 PC 복원
  m_RPC_updated = true;                     // [한국어] RPC 업데이트 표시
  m_last_was_call = false;                  // [한국어] ret 명령어 처리 중임을 표시
  m_RPC = m_callstack.back().m_RPC;         // [한국어] 재귀 복귀 PC 복원
  m_func_info = m_callstack.back().m_func_info; // [한국어] 호출자 function_info 복원
  if (m_func_info) {
    assert(m_local_mem_stack_pointer >= m_func_info->local_mem_framesize()); // [한국어] 언더플로우 방지 검사
    m_local_mem_stack_pointer -= m_func_info->local_mem_framesize(); // [한국어] 로컬 메모리 프레임 해제
  }
  m_callstack.pop_back();                   // [한국어] 피호출 함수 콜스택 엔트리 제거
  // m_regs.pop_back();
  // [한국어] PTXPlus는 레지스터 프레임을 공유하므로 pop하지 않는다
  // m_debug_trace_regs_modified.pop_back();
  // m_debug_trace_regs_read.pop_back();

  // write return value into caller frame
  if (rv_dst != NULL) copy_buffer_to_frame(this, buffer); // [한국어] 반환값을 호출자 프레임에 씀

  return m_callstack.empty();               // [한국어] 최상위 커널 함수로 복귀 여부 반환
}

/*
 * [한국어]
 * ptx_thread_info::dump_callstack - 현재 콜스택 상태를 stdout에 출력한다 (디버그용)
 *
 * 시뮬레이션 오류 발생 시 스레드의 함수 호출 흐름을 추적하는 데 사용한다.
 * 콜스택과 레지스터 스택을 동시에 순회하며 각 프레임의 함수 이름, PC, 반환값 심볼,
 * 레지스터 수를 출력한다. 두 스택 크기가 불일치하면 경고 메시지를 추가로 출력한다.
 *
 * 실행 컨텍스트: 호스트 CPU, 디버그/오류 처리 시점.
 * 호출 체인:
 *   시뮬레이터 디버그 코드 → ptx_thread_info::dump_callstack
 */
void ptx_thread_info::dump_callstack() const {
  std::list<stack_entry>::const_iterator c = m_callstack.begin(); // [한국어] 콜스택의 시작(최하위 프레임)을 가리키는 이터레이터
  std::list<reg_map_t>::const_iterator r = m_regs.begin();        // [한국어] 레지스터 스택의 시작을 가리키는 이터레이터

  printf("\n\n");
  printf("Call stack for thread uid = %u (sc=%u, hwtid=%u)\n", m_uid, m_hw_sid,
         m_hw_tid);                         // [한국어] 스레드 UID, SM ID(sc), HW 스레드 ID를 출력하여 어느 스레드의 콜스택인지 식별한다
  while (c != m_callstack.end() && r != m_regs.end()) { // [한국어] 콜스택과 레지스터 스택을 동시에 순회한다
    const stack_entry &c_e = *c;            // [한국어] 현재 콜스택 엔트리 참조
    const reg_map_t &regs = *r;             // [한국어] 현재 레지스터 프레임 참조
    if (!c_e.m_valid) {                     // [한국어] m_valid=false이면 최하위 진입점 프레임(커널 함수 자체)
      printf("  <entry>                              #regs = %zu\n",
             regs.size());                  // [한국어] 진입점 프레임은 함수 이름 없이 레지스터 수만 출력
    } else {
      printf("  %20s  PC=%3u RV= (callee=\'%s\',caller=\'%s\') #regs = %zu\n",
             c_e.m_func_info->get_name().c_str(), c_e.m_PC,
             c_e.m_return_var_src->name().c_str(),
             c_e.m_return_var_dst->name().c_str(), regs.size()); // [한국어] 함수 이름, 복귀 PC, 반환값 심볼(callee/caller), 레지스터 수를 출력
    }
    c++;                                    // [한국어] 다음 콜스택 엔트리로 이동
    r++;                                    // [한국어] 다음 레지스터 프레임으로 이동
  }
  if (c != m_callstack.end() || r != m_regs.end()) {
    printf("  *** mismatch in m_regs and m_callstack sizes ***\n"); // [한국어] 두 스택 크기가 다르면 내부 버그 — 경고 메시지 출력
  }
  printf("\n\n");
}

/*
 * [한국어]
 * ptx_thread_info::get_location - 현재 실행 중인 PTX 소스 위치를 문자열로 반환한다
 *
 * @return : "파일명:라인번호" 형식의 문자열 (예: "kernel.ptx:42").
 *
 * PTX 명령어에는 원본 CUDA 소스의 파일명과 줄 번호 정보가 포함되어 있다.
 * 이 함수는 현재 PC의 PTX 명령어에서 그 정보를 추출하여 반환한다.
 * 디버그 오류 메시지나 실행 추적에 사용된다.
 * 호출 체인:
 *   오류 진단 코드 → ptx_thread_info::get_location
 */
std::string ptx_thread_info::get_location() const {
  const ptx_instruction *pI = m_func_info->get_instruction(m_PC); // [한국어] 현재 PC에 해당하는 PTX 명령어 객체를 가져온다
  char buf[1024];
  snprintf(buf, 1024, "%s:%u", pI->source_file(), pI->source_line()); // [한국어] "소스파일명:줄번호" 형식의 문자열을 1024바이트 버퍼에 안전하게 포맷한다
  return std::string(buf);                  // [한국어] 버퍼 내용을 std::string으로 변환하여 반환한다
}

/*
 * [한국어]
 * ptx_thread_info::get_inst - 현재 PC의 PTX 명령어를 반환한다
 *
 * @return : 현재 m_PC에 해당하는 ptx_instruction 포인터.
 *
 * 호출 체인:
 *   instructions.cc / shader.cc → ptx_thread_info::get_inst
 */
const ptx_instruction *ptx_thread_info::get_inst() const {
  return m_func_info->get_instruction(m_PC); // [한국어] 현재 함수에서 m_PC 위치의 PTX 명령어 객체를 반환한다
}

/*
 * [한국어]
 * ptx_thread_info::get_inst (PC 지정 버전) - 지정된 PC의 PTX 명령어를 반환한다
 *
 * @pc     : 조회할 PTX 프로그램 카운터 주소.
 * @return : 해당 PC의 ptx_instruction 포인터.
 *
 * 현재 PC가 아닌 특정 PC의 명령어를 조회할 때 사용한다.
 * 호출 체인:
 *   디버그/분석 코드 → ptx_thread_info::get_inst(addr_t)
 */
const ptx_instruction *ptx_thread_info::get_inst(addr_t pc) const {
  return m_func_info->get_instruction(pc); // [한국어] 지정된 pc 주소의 PTX 명령어 객체를 반환한다
}

/*
 * [한국어]
 * ptx_thread_info::dump_regs - 현재 레지스터 파일 전체를 파일에 출력한다 (디버그용)
 *
 * @fp : 출력 대상 파일 스트림.
 *
 * 현재 함수 프레임(m_regs의 마지막 요소)의 모든 레지스터를 순회하며 이름과 값을 출력한다.
 * 레지스터 파일이 비어 있거나 현재 프레임이 비어 있으면 아무것도 출력하지 않는다.
 * 시뮬레이션 오류 발생 시 스레드 상태를 덤프하는 데 사용된다.
 * 호출 체인:
 *   오류 진단 코드 → ptx_thread_info::dump_regs
 */
void ptx_thread_info::dump_regs(FILE *fp) {
  if (m_regs.empty()) return;               // [한국어] 레지스터 스택이 비어 있으면 출력 불가 — 조기 반환
  if (m_regs.back().empty()) return;        // [한국어] 현재 프레임(가장 최근 함수)에 레지스터가 없으면 조기 반환
  fprintf(fp, "Register File Contents:\n"); // [한국어] 레지스터 파일 덤프 헤더 출력
  fflush(fp);                               // [한국어] 버퍼 즉시 비우기 — abort() 전 출력 보장
  reg_map_t::const_iterator r;
  for (r = m_regs.back().begin(); r != m_regs.back().end(); ++r) { // [한국어] 현재 함수 프레임의 모든 레지스터를 순회한다
    const symbol *sym = r->first;           // [한국어] 레지스터 심볼 포인터 (이름과 타입 정보 포함)
    ptx_reg_t value = r->second;            // [한국어] 레지스터 현재 값 (union 타입)
    std::string name = sym->name();         // [한국어] 심볼 이름 추출 (예: "%r0", "%f1")
    print_reg(fp, name, value, m_symbol_table); // [한국어] 타입에 맞게 포맷하여 출력
  }
}

/*
 * [한국어]
 * ptx_thread_info::dump_modifiedregs - 가장 최근 명령어에서 수정/읽힌 레지스터를 출력한다
 *
 * @fp : 출력 대상 파일 스트림.
 *
 * 디버그 트레이스 모드(m_enable_debug_trace=true)에서 각 명령어 실행 후 호출되어
 * 해당 명령어가 수정한(출력) 레지스터와 읽은(입력) 레지스터를 모두 출력한다.
 * 명령어 단위의 레지스터 변화를 추적하는 데 사용된다.
 * 출력 후에도 m_debug_trace_regs_* 스택은 비워지지 않는다.
 *
 * 호출 체인:
 *   shader.cc 또는 instructions.cc (디버그 모드) → ptx_thread_info::dump_modifiedregs
 */
void ptx_thread_info::dump_modifiedregs(FILE *fp) {
  if (!(m_debug_trace_regs_modified.empty() ||
        m_debug_trace_regs_modified.back().empty())) { // [한국어] 수정 레지스터 추적 스택이 비어있지 않고 현재 프레임에 데이터가 있는 경우에만 출력한다
    fprintf(fp, "Output Registers:\n");     // [한국어] 출력 레지스터 섹션 헤더 — 명령어가 값을 쓴 레지스터들
    fflush(fp);
    reg_map_t::iterator r;
    for (r = m_debug_trace_regs_modified.back().begin();
         r != m_debug_trace_regs_modified.back().end(); ++r) { // [한국어] 수정된 레지스터 프레임을 순회한다
      const symbol *sym = r->first;         // [한국어] 수정된 레지스터 심볼
      std::string name = sym->name();       // [한국어] 레지스터 이름
      ptx_reg_t value = r->second;          // [한국어] 명령어 실행 후 레지스터 값
      print_reg(fp, name, value, m_symbol_table); // [한국어] 타입에 맞게 포맷하여 출력
    }
  }
  if (!(m_debug_trace_regs_read.empty() ||
        m_debug_trace_regs_read.back().empty())) { // [한국어] 읽기 레지스터 추적 스택이 비어있지 않고 현재 프레임에 데이터가 있는 경우에만 출력한다
    fprintf(fp, "Input Registers:\n");      // [한국어] 입력 레지스터 섹션 헤더 — 명령어가 값을 읽은 소스 레지스터들
    fflush(fp);
    reg_map_t::iterator r;
    for (r = m_debug_trace_regs_read.back().begin();
         r != m_debug_trace_regs_read.back().end(); ++r) { // [한국어] 읽힌 레지스터 프레임을 순회한다
      const symbol *sym = r->first;         // [한국어] 읽힌 레지스터 심볼
      std::string name = sym->name();       // [한국어] 레지스터 이름
      ptx_reg_t value = r->second;          // [한국어] 명령어 실행 전 소스 레지스터 값
      print_reg(fp, name, value, m_symbol_table); // [한국어] 타입에 맞게 포맷하여 출력
    }
  }
}

/*
 * [한국어]
 * ptx_thread_info::push_breakaddr - 브레이크 주소를 스택에 추가한다
 *
 * @breakaddr : 브레이크포인트 대상 오퍼랜드 정보.
 *
 * PTXPlus의 brk(break) 명령어 구현에서 사용된다.
 * break는 루프 내에서 루프를 빠져나오는 분기로, 빠져나올 대상 주소를 스택에 쌓아
 * pop_breakaddr()로 꺼내 사용한다.
 * 호출 체인:
 *   instructions.cc (brk_impl 또는 유사 함수) → ptx_thread_info::push_breakaddr
 */
void ptx_thread_info::push_breakaddr(const operand_info &breakaddr) {
  m_breakaddrs.push(breakaddr); // [한국어] 브레이크 주소를 스택(std::stack)에 push한다. 중첩된 루프에서 각 루프의 break 대상이 별도로 쌓인다
}

/*
 * [한국어]
 * ptx_thread_info::pop_breakaddr - 브레이크 주소 스택에서 가장 최근 주소를 꺼낸다
 *
 * @return : 가장 최근에 push된 브레이크 대상 오퍼랜드 정보 참조.
 *
 * 스택이 비어 있는 상태에서 pop을 시도하면 에러 메시지 출력 후 assert로 종료한다.
 * 호출 체인:
 *   instructions.cc (break 명령어 실행 시) → ptx_thread_info::pop_breakaddr
 */
const operand_info &ptx_thread_info::pop_breakaddr() {
  if (m_breakaddrs.empty()) {               // [한국어] 브레이크 주소 스택이 비어 있으면 — break 명령어가 루프 바깥에서 사용된 오류
    printf("empty breakaddrs stack");
    assert(0);                             // [한국어] assert 실패로 프로그램을 종료한다
  }
  operand_info &breakaddr = m_breakaddrs.top(); // [한국어] 스택 최상위(가장 최근 push된) 브레이크 주소 참조를 가져온다
  m_breakaddrs.pop();                       // [한국어] 스택에서 최상위 항목을 제거한다
  return breakaddr;                         // [한국어] 이미 pop된 항목의 참조를 반환 — 이 참조는 pop 이후에도 유효해야 하며, 지역 변수가 아닌 스택 원소였으므로 주의가 필요하다
}

/*
 * [한국어]
 * ptx_thread_info::set_npc - 다음 실행 PC를 함수 시작점으로 설정한다 (함수 포인터 호출용)
 *
 * @f : 호출할 함수의 function_info 포인터.
 *
 * 간접 함수 호출(indirect call, 함수 포인터를 통한 호출) 시 사용된다.
 * 일반 call 명령어와 달리 callstack_push() 없이 단순히 다음 PC와 현재 함수 컨텍스트를 변경한다.
 * m_func_info와 m_symbol_table도 새 함수로 교체한다.
 *
 * 실행 컨텍스트: 호스트 CPU, 간접 함수 호출 처리 시점.
 * 호출 체인:
 *   instructions.cc (간접 call 처리) → ptx_thread_info::set_npc
 */
void ptx_thread_info::set_npc(const function_info *f) {
  m_NPC = f->get_start_PC();               // [한국어] 다음 실행 PC를 호출할 함수의 시작 주소로 설정한다
  m_func_info = const_cast<function_info *>(f); // [한국어] 현재 함수 정보를 새 함수로 교체한다. const_cast는 function_info의 비상수 멤버에 접근하기 위해 필요하다
  m_symbol_table = m_func_info->get_symtab(); // [한국어] 새 함수의 심볼 테이블로 교체한다
}

/*
 * [한국어]
 * feature_not_implemented - 미구현 PTX 기능 접근 시 에러 메시지 출력 후 종료
 *
 * @f : 미구현 기능의 이름 문자열 (예: "%smid", "%lanemask_eq").
 *
 * PTX의 특수 레지스터나 명령어 중 GPGPU-Sim에서 아직 구현되지 않은 것에 접근할 때 호출된다.
 * 에러 메시지를 stdout에 출력한 뒤 abort()로 프로그램을 즉시 종료한다.
 * 이 함수는 파일 상단에 전방 선언되어 get_builtin() 내에서 사용된다.
 *
 * 실행 컨텍스트: 호스트 CPU, 미구현 레지스터 접근 시점.
 * 에러 경로: printf → abort() → SIGABRT → 코어 덤프.
 * 호출 체인:
 *   ptx_thread_info::get_builtin → [feature_not_implemented] → abort()
 */
void feature_not_implemented(const char *f) {
  printf("GPGPU-Sim: feature '%s' not supported\n", f); // [한국어] 미구현 기능 이름을 포함한 에러 메시지를 출력한다. stdout에 출력되어 디버그 로그에 남는다
  abort();                                 // [한국어] SIGABRT 신호로 즉시 종료한다. 코어 덤프가 생성되어 디버거(gdb)에서 호출 스택 분석이 가능하다
}
