// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ivan Sham,
// Andrew Turner, Ali Bakhoda, The University of British Columbia
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

/*
 * [한국어 설명] GPGPU-Sim 시뮬레이터 진입점 구현 (gpgpusim_entrypoint.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim 시뮬레이터의 최상위 진입점(entrypoint) 구현체이다.
 * CUDA 런타임 API(cuLaunchKernel, cudaMemcpy 등)가 libcuda를 통해 인터셉트된 후
 * 이 파일의 함수들로 제어가 넘어온다. 시뮬레이터 초기화(gpgpu_ptx_sim_init_perf),
 * 시뮬레이션 쓰레드 생성(start_sim_thread), 사이클-레벨 실행 루프
 * (gpgpu_sim_thread_concurrent / gpgpu_sim_thread_sequential),
 * 동기화(synchronize / synchronize_check), 종료(exit_simulation) 등
 * 시뮬레이터 생명주기 전반을 담당한다.
 * SST(Sandia Structural Simulation Toolkit) 연동 모드도 지원하며,
 * SST_Cycle() 함수를 통해 외부 시스템 시뮬레이터가 GPGPU-Sim에 클럭을 주입할 수 있다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름 최상위 계층에 위치한다:
 *   CUDA Application
 *     → libcuda (libcuda/cuda_runtime_api.cc) — CUDA API 인터셉트
 *         → [이 파일] gpgpusim_entrypoint.cc — 시뮬레이터 생명주기 관리
 *             ├── gpgpu_ptx_sim_init_perf() → 옵션 파싱, GPU 객체 생성
 *             ├── start_sim_thread()        → 시뮬레이션 전용 쓰레드 생성
 *             ├── gpgpu_sim_thread_concurrent() → 메인 사이클 루프 (별도 쓰레드)
 *             │     ├── stream_manager::operation() — 스트림 큐에서 작업 꺼냄
 *             │     ├── gpgpu_cuda_ptx_sim_main_func() — PTX 기능 시뮬레이션
 *             │     └── gpgpu_sim::cycle() — 타이밍 시뮬레이션 1 사이클 전진
 *             └── synchronize() / exit_simulation() → 동기화 및 종료
 * 실행 컨텍스트: 호스트 사용자 공간(user-space), 멀티쓰레드 (호스트 쓰레드 + 시뮬레이션 쓰레드).
 *
 * === 타 모듈과의 연결 ===
 * 의존(사용)하는 모듈:
 *   - libcuda/gpgpu_context.h   : GPGPU-Sim 전역 컨텍스트(gpgpu_context) 및 GPGPU_Context() 싱글톤
 *   - cuda-sim/cuda-sim.h       : PTX 기능 시뮬레이션(cuda_sim::gpgpu_cuda_ptx_sim_main_func)
 *   - cuda-sim/ptx_ir.h         : PTX IR 파싱 초기화 (ptx_parser)
 *   - cuda-sim/ptx_parser.h     : 파서 환경 변수 읽기 (read_parser_environment_variables)
 *   - gpgpu-sim/gpu-sim.h       : 타이밍 시뮬레이터 핵심(exec_gpgpu_sim, sst_gpgpu_sim, gpgpu_sim)
 *   - gpgpu-sim/icnt_wrapper.h  : 네트워크온칩(ICNT/Booksim) 옵션 등록 (icnt_reg_options)
 *   - option_parser.h           : gpgpusim.config 파싱 프레임워크
 *   - stream_manager.h          : CUDA 스트림 큐 관리 (stream_manager)
 * 공유 자료구조:
 *   - GPGPUsim_ctx (gpgpusim_entrypoint.h): 시뮬레이션 상태 + 세마포어 + 핵심 객체 포인터
 *   - gpgpu_context (libcuda/gpgpu_context.h): 전체 컨텍스트 싱글톤
 *
 * === 주요 함수/구조체 요약 ===
 * - gpgpu_ptx_sim_init_perf()       : 옵션 파싱 → GPU 객체(gpgpu_sim) + 스트림 관리자 생성 → 세마포어 초기화
 * - start_sim_thread()              : 시뮬레이션 전용 pthred 생성 (concurrent 또는 sequential 모드)
 * - gpgpu_sim_thread_concurrent()   : 비동기 커널 동시 실행 지원하는 메인 사이클 루프 (별도 쓰레드)
 * - gpgpu_sim_thread_sequential()   : 한 번에 커널 하나씩만 실행하는 순차 사이클 루프 (별도 쓰레드)
 * - SST_Cycle()                     : SST 외부 시뮬레이터가 호출하는 단일 사이클 진행 함수
 * - synchronize()                   : 호스트가 GPU 시뮬레이션 완료를 바쁜 대기(busy-wait)로 기다림
 * - exit_simulation()               : g_sim_done = true 설정 후 시뮬레이션 쓰레드 종료 대기
 * - print_simulation_time()         : 시뮬레이션 경과 시간 및 속도(inst/cycle per sec) 출력
 */

#include "gpgpusim_entrypoint.h"   /* [한국어] GPGPUsim_ctx 클래스 선언 — 시뮬레이터 상태/세마포어/핵심 객체 포인터 */
#include <stdio.h>                 /* [한국어] 표준 입출력 (printf, fprintf, fflush) */

#include "../libcuda/gpgpu_context.h"  /* [한국어] gpgpu_context 싱글톤 및 GPGPU_Context() 접근자 */
#include "cuda-sim/cuda-sim.h"         /* [한국어] PTX 기능 시뮬레이션 (cuda_sim, gpgpu_cuda_ptx_sim_main_func) */
#include "cuda-sim/ptx_ir.h"           /* [한국어] PTX IR 파싱 구조체 및 초기화 */
#include "cuda-sim/ptx_parser.h"       /* [한국어] PTX 파서 환경 변수 읽기 */
#include "gpgpu-sim/gpu-sim.h"         /* [한국어] 타이밍 시뮬레이터 핵심(gpgpu_sim, exec_gpgpu_sim, sst_gpgpu_sim) */
#include "gpgpu-sim/icnt_wrapper.h"    /* [한국어] Booksim NoC(네트워크온칩) 옵션 등록 */
#include "option_parser.h"             /* [한국어] gpgpusim.config 파싱 프레임워크 */
#include "stream_manager.h"            /* [한국어] CUDA 스트림 큐 관리 */

/* [한국어] 두 값 중 더 큰 값을 반환하는 매크로.
 * 경과 시간 계산 시 분모가 0이 되지 않도록 최솟값을 1로 보정할 때 사용한다. */
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

/* [한국어] gpgpusim.config 파일을 읽기 위한 가짜 argc/argv.
 * 시뮬레이터는 실제 프로그램 인수 없이 내부적으로 "-config gpgpusim.config"를
 * 고정 인수로 사용하여 option_parser_cmdline()을 호출한다.
 * sg_argc = 3: 프로그램명("") + "-config" + "gpgpusim.config" 총 3개.
 * sg_argv: 순서대로 프로그램명(빈 문자열), 옵션 키, 옵션 값. */
static int sg_argc = 3;
static const char *sg_argv[] = {"", "-config", "gpgpusim.config"};

/*
 * [한국어]
 * GPGPUsim_ctx_ptr - 현재 GPGPU 컨텍스트의 GPGPUsim_ctx 포인터 반환
 *
 * @return: GPGPU_Context() 싱글톤에서 the_gpgpusim 필드 포인터.
 *          NULL인 경우 gpgpu_ptx_sim_init_perf()가 아직 호출되지 않은 것이다.
 *
 * SST 모드에서 긴 화살표 체인(GPGPU_Context()->the_gpgpusim->...)을
 * 반복 타이핑하지 않도록 래핑한 헬퍼 함수이다.
 * 실행 컨텍스트: SST 외부 클럭 스레드에서 호출된다.
 *
 * 호출 체인:
 *   SST_Cycle() / SST helper funcs → [GPGPUsim_ctx_ptr] → GPGPU_Context()->the_gpgpusim
 */
// Help funcs to avoid multiple '->' for SST
GPGPUsim_ctx *GPGPUsim_ctx_ptr() { return GPGPU_Context()->the_gpgpusim; }

/*
 * [한국어]
 * g_the_gpu - SST 모드 전용 GPU 시뮬레이터(sst_gpgpu_sim) 포인터 반환
 *
 * @return: GPGPUsim_ctx의 g_the_gpu를 sst_gpgpu_sim*로 다운캐스트한 포인터.
 *          SST 모드에서만 유효하며, 비-SST 모드에서 호출하면 타입이 맞지 않는다.
 *
 * SST 연동 시 sst_gpgpu_sim 고유 메서드(SST_cycle 등)를 호출하기 위해
 * static_cast를 통해 기반 클래스 포인터를 파생 클래스로 내려받는다.
 * 실행 컨텍스트: SST 클럭 스레드.
 *
 * 호출 체인:
 *   SST_Cycle() → [g_the_gpu] → sst_gpgpu_sim::SST_cycle() 등
 */
class sst_gpgpu_sim *g_the_gpu() {
  return static_cast<sst_gpgpu_sim *>(GPGPUsim_ctx_ptr()->g_the_gpu);
}

/*
 * [한국어]
 * g_stream_manager - 전역 스트림 관리자 포인터 반환
 *
 * @return: GPGPUsim_ctx의 g_stream_manager 포인터.
 *          NULL이면 gpgpu_ptx_sim_init_perf()가 아직 완료되지 않은 것이다.
 *
 * SST 모드의 헬퍼로, 스트림 관리자 접근을 단순화한다.
 * 실행 컨텍스트: SST 클럭 스레드.
 *
 * 호출 체인:
 *   SST_Cycle() / SST_callback_xxx → [g_stream_manager] → stream_manager 메서드
 */
class stream_manager *g_stream_manager() {
  return GPGPUsim_ctx_ptr()->g_stream_manager;
}

/* [한국어] SST 연동 시 cudaThreadSynchronize 완료를 SST에 통지하는 콜백 함수 선언.
 * __attribute__((weak))는 "링크 시 강한 정의(strong definition)가 있으면 그것을 우선 사용하고,
 * 없으면 이 빈 구현을 사용하라"는 GCC 약한 심볼(weak symbol) 지시자이다.
 * SST 없이 단독 실행 시에는 빈 함수가 호출되어 아무 일도 하지 않는다. */
// SST callback
extern void SST_callback_cudaThreadSynchronize_done();
__attribute__((weak)) void SST_callback_cudaThreadSynchronize_done() {}

/*
 * [한국어]
 * gpgpu_sim_thread_sequential - 순차 시뮬레이션 전용 쓰레드 함수 (OpenCL 등 단일 커널 모드)
 *
 * @ctx_ptr: gpgpu_context* 를 void*로 전달받은 포인터.
 *           pthread_create()에서 넘긴 (void*)this가 여기로 들어온다.
 * @return: 항상 NULL. pthread 쓰레드 함수 규약.
 *
 * 순차 모드(sequential mode)에서는 한 번에 커널 하나만 실행된다.
 * 동작 흐름:
 *   1) g_sim_signal_start 세마포어를 기다린다.
 *      → 메인 쓰레드(gpgpu_opencl_ptx_sim_main_perf)가 sem_post로 신호를 보낼 때 깨어난다.
 *   2) 아직 실행할 CTA(Cooperative Thread Array, CUDA의 스레드 블록)가 있으면(get_more_cta_left):
 *      a) GPU를 초기화(init)하고
 *      b) GPU가 active 상태인 동안 cycle()을 한 사이클씩 전진시키며 deadlock을 감시한다.
 *      c) 커널이 완료되면 통계를 출력하고 경과 시간을 프린트한다.
 *   3) g_sim_signal_finish 세마포어로 메인 쓰레드에게 완료를 통보한다.
 *   4) done이 되면 루프를 탈출하고 g_sim_signal_exit로 종료를 통보한다.
 * 실행 컨텍스트: 시뮬레이션 전용 별도 쓰레드. 메인 쓰레드와 세마포어로 동기화한다.
 * 이 쓰레드에서는 gpgpu_sim::cycle()이 호출되므로 타이밍 시뮬레이션이 실제로 수행된다.
 *
 * 호출 체인:
 *   pthread_create(gpgpu_sim_thread_sequential) ← start_sim_thread(api=2)
 *   [이 함수] → g_the_gpu->init() / cycle() / deadlock_check() / print_stats() / update_stats()
 */
void *gpgpu_sim_thread_sequential(void *ctx_ptr) {
  gpgpu_context *ctx = (gpgpu_context *)ctx_ptr;  /* [한국어] void*를 gpgpu_context*로 복원 */
  // at most one kernel running at a time
  bool done;  /* [한국어] 루프 탈출 조건: 더 이상 실행할 CTA가 없으면 true */
  do {
    sem_wait(&(ctx->the_gpgpusim->g_sim_signal_start));
    /* [한국어] 메인 쓰레드(gpgpu_opencl_ptx_sim_main_perf)가 sem_post를 호출할 때까지 블록.
     * 새로운 커널 실행 요청이 올 때마다 이 세마포어를 통해 깨어난다. */

    done = true;  /* [한국어] 낙관적으로 true로 설정 — CTA가 없으면 그대로 루프 종료 */

    if (ctx->the_gpgpusim->g_the_gpu->get_more_cta_left()) {
      /* [한국어] 아직 GPU에 올릴 CTA(스레드 블록)가 남아 있으면 시뮬레이션 진행.
       * get_more_cta_left()는 현재 커널의 모든 CTA가 디스패치되었는지 확인한다. */

      done = false;  /* [한국어] 아직 할 일이 있으니 루프 계속 */

      ctx->the_gpgpusim->g_the_gpu->init();
      /* [한국어] GPU 타이밍 모델 초기화: SM 상태, 캐시 초기화, 커널 dispatch 큐 세팅.
       * 새 커널을 시작하기 전에 반드시 호출해야 한다. */

      while (ctx->the_gpgpusim->g_the_gpu->active()) {
        /* [한국어] GPU가 아직 활성 상태(실행 중인 warp 또는 대기 중인 메모리 요청이 있음)인 동안 반복 */

        ctx->the_gpgpusim->g_the_gpu->cycle();
        /* [한국어] 타이밍 시뮬레이션 1 사이클 전진:
         * SM 파이프라인(fetch/decode/execute), 캐시, DRAM, NoC를 모두 1 사이클씩 전진시킨다. */

        ctx->the_gpgpusim->g_the_gpu->deadlock_check();
        /* [한국어] 교착 상태(deadlock) 감지: 일정 사이클 이상 진전이 없으면 assert로 종료.
         * 시뮬레이터 버그나 잘못된 설정으로 인한 무한 루프를 방지한다. */
      }

      ctx->the_gpgpusim->g_the_gpu->print_stats(
          ctx->the_gpgpusim->g_the_gpu->last_streamID);
      /* [한국어] 커널 완료 후 성능 통계(IPC, cache hit rate, DRAM bandwidth 등)를 stdout에 출력.
       * last_streamID: 방금 완료된 커널이 속한 CUDA 스트림 ID. */

      ctx->the_gpgpusim->g_the_gpu->update_stats();
      /* [한국어] 누적 통계(gpu_tot_sim_insn, gpu_tot_sim_cycle 등)를 업데이트.
       * 여러 커널을 순차 실행할 때 전체 통계를 누적하기 위해 커널 완료마다 호출한다. */

      ctx->print_simulation_time();
      /* [한국어] 시뮬레이션 경과 시간(wall-clock time)과 시뮬레이션 속도(inst/sec, cycle/sec)를 출력. */
    }

    sem_post(&(ctx->the_gpgpusim->g_sim_signal_finish));
    /* [한국어] 메인 쓰레드(gpgpu_opencl_ptx_sim_main_perf)에게 이 단계가 완료되었음을 통보.
     * 메인 쓰레드는 sem_wait(&g_sim_signal_finish)에서 블록 중이므로 이 신호로 깨어난다. */

  } while (!done);
  /* [한국어] done이 false(CTA 있음)이면 루프를 계속하여 다음 신호를 기다린다. */

  sem_post(&(ctx->the_gpgpusim->g_sim_signal_exit));
  /* [한국어] 시뮬레이션 완전 종료 통보. exit_simulation()이 sem_wait(&g_sim_signal_exit)에서 기다린다. */

  return NULL;  /* [한국어] pthread 쓰레드 함수 반환값 (사용하지 않음) */
}

/*
 * [한국어]
 * termination_callback - 프로세스 종료 시 호출되는 atexit 콜백
 *
 * @return: void
 *
 * atexit()에 등록되어, 프로세스가 정상 종료될 때 "exit detected" 메시지를 출력한다.
 * 시뮬레이션이 비정상 종료(예: 최대 사이클 도달로 exit(1) 호출)될 때에도 이 메시지가 출력되므로
 * 로그 파일에서 정상/비정상 종료 여부를 판단하는 데 쓸 수 있다.
 * gpgpu_sim_thread_concurrent() 내에서 atexit(termination_callback)로 등록된다.
 * 실행 컨텍스트: 프로세스 종료 시 C 런타임이 호출 — 어느 쓰레드든 exit() 호출 시 실행.
 *
 * 호출 체인:
 *   atexit 등록 ← gpgpu_sim_thread_concurrent()
 *   [이 함수] ← C 런타임 atexit 메커니즘 (exit() 시점)
 */
static void termination_callback() {
  printf("GPGPU-Sim: *** exit detected ***\n");  /* [한국어] 종료 감지 메시지 출력 */
  fflush(stdout);  /* [한국어] 버퍼 플러시: 프로세스가 종료되기 전에 출력이 실제로 기록되도록 강제 */
}

/*
 * [한국어]
 * gpgpu_sim_thread_concurrent - 동시 커널 실행(concurrent kernel execution) 시뮬레이션 쓰레드
 *
 * @ctx_ptr: gpgpu_context* 를 void*로 전달받은 포인터.
 *           start_sim_thread(api=1)의 pthread_create()에서 (void*)this로 전달된다.
 * @return: 항상 NULL (pthread 쓰레드 규약).
 *
 * GPGPU-Sim의 가장 일반적인 시뮬레이션 모드 쓰레드이다.
 * 여러 CUDA 스트림에 걸쳐 커널이 동시에 실행될 수 있는 GPU 동작을 시뮬레이션한다.
 * 동작 흐름:
 *   [외부 루프] g_sim_done이 될 때까지 반복
 *     1. 스트림 큐가 비어있는 동안 바쁜 대기(busy-wait) — 새 작업이 들어올 때까지 스핀
 *     2. g_sim_lock 잠금 후 g_sim_active = true 설정
 *     3. GPU 초기화(init())
 *     [내부 루프] active 상태이고 g_sim_done이 아닌 동안 반복
 *       a. stream_manager::operation(): 스트림 큐 front에서 작업 꺼내어 실행 시도.
 *          커널이 완료되었고 GPU도 비활성이면 break (bug 147: 커널 간 재초기화 필요)
 *       b. 기능 시뮬레이션(PTX): is_functional_sim()이면 gpgpu_cuda_ptx_sim_main_func() 호출
 *       c. 타이밍 시뮬레이션: GPU active이면 cycle() 1 사이클 전진 + deadlock_check()
 *       d. 최대 사이클/명령 한도 초과 시 g_sim_done = true, break_limit = true
 *     4. 통계 출력 및 g_sim_active = false 설정
 *   [종료] g_sim_signal_exit 세마포어 post → exit_simulation() 깨어남
 *          break_limit이면 exit(1)로 프로세스 강제 종료
 * 실행 컨텍스트: 시뮬레이션 전용 쓰레드. 호스트 쓰레드와 g_sim_lock(뮤텍스)으로 동기화한다.
 * 주의: 이 루프는 바쁜 대기(spin-wait)를 포함하므로 CPU 코어를 계속 점유한다.
 *
 * 호출 체인:
 *   pthread_create(gpgpu_sim_thread_concurrent) ← start_sim_thread(api=1)
 *   [이 함수] → stream_manager::operation() → stream_operation::do_operation()
 *            → gpgpu_cuda_ptx_sim_main_func() (기능 시뮬레이션)
 *            → gpgpu_sim::cycle() (타이밍 시뮬레이션)
 *            → gpgpu_sim::deadlock_check()
 */
void *gpgpu_sim_thread_concurrent(void *ctx_ptr) {
  gpgpu_context *ctx = (gpgpu_context *)ctx_ptr;  /* [한국어] void*에서 gpgpu_context*로 복원 */
  atexit(termination_callback);
  /* [한국어] 프로세스 종료 시 termination_callback을 실행하도록 등록.
   * exit(1)로 강제 종료될 때에도 "exit detected" 메시지가 출력된다. */

  // concurrent kernel execution simulation thread
  do {
    if (g_debug_execution >= 3) {
      /* [한국어] g_debug_execution >= 3이면 디버그 레벨 3 이상으로 설정된 것 (gpgpusim.config의 -gpgpu_runtime_stat 등).
       * 스트림 대기 중임을 알리는 디버그 메시지를 출력한다. */
      printf(
          "GPGPU-Sim: *** simulation thread starting and spinning waiting for "
          "work ***\n");
      fflush(stdout);  /* [한국어] 버퍼 플러시 — 출력이 지연되지 않도록 즉시 기록 */
    }

    while (ctx->the_gpgpusim->g_stream_manager->empty_protected() &&
           !ctx->the_gpgpusim->g_sim_done)
      ;
    /* [한국어] 스트림 큐가 비어있고 아직 g_sim_done이 아니면 바쁜 대기(spin-wait).
     * empty_protected()는 m_lock을 잡고 안전하게 비어있는지 확인한다.
     * 호스트 쓰레드가 stream_manager::push()로 새 작업을 넣으면 empty가 false가 되어 루프 탈출. */

    if (g_debug_execution >= 3) {
      printf("GPGPU-Sim: ** START simulation thread (detected work) **\n");
      ctx->the_gpgpusim->g_stream_manager->print(stdout);
      /* [한국어] 현재 스트림 관리자 상태(각 스트림의 pending 작업 목록)를 디버그 출력 */
      fflush(stdout);
    }

    pthread_mutex_lock(&(ctx->the_gpgpusim->g_sim_lock));
    /* [한국어] g_sim_active 변경 전 잠금. synchronize()가 동시에 g_sim_active를 읽으므로
     * 데이터 레이스(data race)를 방지해야 한다. */
    ctx->the_gpgpusim->g_sim_active = true;
    /* [한국어] 시뮬레이션이 현재 실행 중임을 표시.
     * synchronize()의 done 조건 검사에서 이 값이 false일 때만 완료로 본다. */
    pthread_mutex_unlock(&(ctx->the_gpgpusim->g_sim_lock));

    bool active = false;     /* [한국어] 내부 루프 계속 여부: GPU 활성 또는 스트림 큐 비어있지 않음 */
    bool sim_cycles = false; /* [한국어] 이 번 외부 루프에서 실제로 사이클이 진행되었는지 여부 — 통계 출력 조건 */

    ctx->the_gpgpusim->g_the_gpu->init();
    /* [한국어] 새로운 커널 세션 시작 전 GPU 타이밍 모델 초기화.
     * SM 상태, warp 스케줄러, 캐시 등이 초기 상태로 리셋된다. */

    do {
      // check if a kernel has completed
      // launch operation on device if one is pending and can be run

      // Need to break this loop when a kernel completes. This was a
      // source of non-deterministic behaviour in GPGPU-Sim (bug 147).
      // If another stream operation is available, g_the_gpu remains active,
      // causing this loop to not break. If the next operation happens to be
      // another kernel, the gpu is not re-initialized and the inter-kernel
      // behaviour may be incorrect. Check that a kernel has finished and
      // no other kernel is currently running.
      if (ctx->the_gpgpusim->g_stream_manager->operation(&sim_cycles) &&
          !ctx->the_gpgpusim->g_the_gpu->active())
        break;
      /* [한국어] operation()은 스트림 큐 front에서 작업을 꺼내어 실행 시도하고,
       * 커널이 완료(grid_uid 등록 해제)되었으면 true를 반환한다.
       * 커널 완료 직후 GPU도 idle 상태(active() == false)이면 내부 루프를 탈출한다.
       * 이유: 다음 커널을 시작하기 전에 init()을 다시 호출해야 하기 때문이다(bug 147).
       * sim_cycles: operation() 안에서 커널이 실제로 실행됐으면 true로 설정된다. */

      // functional simulation
      if (ctx->the_gpgpusim->g_the_gpu->is_functional_sim()) {
        /* [한국어] 기능 시뮬레이션(functional simulation) 모드인 경우:
         * PTX 명령어의 "의미(semantics)"만 실행하고, 타이밍은 무시한다.
         * 주로 correctness 검증 또는 tracing 목적으로 사용된다. */
        kernel_info_t *kernel =
            ctx->the_gpgpusim->g_the_gpu->get_functional_kernel();
        /* [한국어] 현재 기능 시뮬레이션 대상 커널 정보(kernel_info_t) 포인터 획득 */
        assert(kernel);  /* [한국어] 기능 시뮬레이션 모드인데 커널이 NULL이면 내부 버그 */
        ctx->the_gpgpusim->gpgpu_ctx->func_sim->gpgpu_cuda_ptx_sim_main_func(
            *kernel);
        /* [한국어] PTX 명령어를 실제로 해석·실행하는 기능 시뮬레이션 진입점 호출.
         * cuda-sim/cuda-sim.cc에 구현되어 있으며, 모든 스레드의 PTX를 순차 실행한다. */
        ctx->the_gpgpusim->g_the_gpu->finish_functional_sim(kernel);
        /* [한국어] 기능 시뮬레이션 완료 처리: 커널 상태를 "기능 시뮬레이션 완료"로 전환 */
      }

      // performance simulation
      if (ctx->the_gpgpusim->g_the_gpu->active()) {
        /* [한국어] GPU에 아직 실행 중인 warp 또는 미완료 메모리 요청이 있으면 사이클 전진 */
        ctx->the_gpgpusim->g_the_gpu->cycle();
        /* [한국어] 타이밍 시뮬레이션 1 사이클: SM 파이프라인, 캐시, DRAM, NoC를 전진시킨다. */
        sim_cycles = true;  /* [한국어] 이 사이클에서 실제 시뮬레이션이 수행되었음을 기록 */
        ctx->the_gpgpusim->g_the_gpu->deadlock_check();
        /* [한국어] 교착 상태 감지: 일정 사이클 이상 진전 없으면 에러 출력 후 abort. */
      } else {
        /* [한국어] GPU가 비활성 상태(active() == false): 실행 중인 warp가 없음 */
        if (ctx->the_gpgpusim->g_the_gpu->cycle_insn_cta_max_hit()) {
          /* [한국어] gpgpusim.config의 -gpgpu_max_cycle, -gpgpu_max_insn, -gpgpu_max_cta 설정값에 도달 */
          ctx->the_gpgpusim->g_stream_manager->stop_all_running_kernels();
          /* [한국어] 현재 실행 중인 모든 커널을 강제 종료하고 스트림 큐를 정리 */
          ctx->the_gpgpusim->g_sim_done = true;
          /* [한국어] 외부 루프 종료 조건 설정: 더 이상 새 커널을 받지 않는다 */
          ctx->the_gpgpusim->break_limit = true;
          /* [한국어] 최대 한도 도달로 인한 종료임을 표시 — 이후 exit(1) 호출 여부 결정에 사용 */
        }
      }

      active = ctx->the_gpgpusim->g_the_gpu->active() ||
               !(ctx->the_gpgpusim->g_stream_manager->empty_protected());
      /* [한국어] 내부 루프 계속 조건:
       * GPU 자체가 아직 실행 중이거나(active()) 스트림 큐에 대기 중인 작업이 있으면 계속. */

    } while (active && !ctx->the_gpgpusim->g_sim_done);
    /* [한국어] active이고 아직 g_sim_done이 아닌 동안 내부 루프 반복 */

    if (g_debug_execution >= 3) {
      printf("GPGPU-Sim: ** STOP simulation thread (no work) **\n");
      fflush(stdout);
    }

    if (sim_cycles) {
      /* [한국어] 이번 외부 루프 반복에서 실제로 사이클이 진행된 경우에만 통계를 출력.
       * sim_cycles == false이면 아무 일도 안 한 것이므로 통계 출력이 불필요하다. */
      ctx->the_gpgpusim->g_the_gpu->print_stats(
          ctx->the_gpgpusim->g_the_gpu->last_streamID);
      /* [한국어] 커널 완료 후 IPC, cache hit/miss, DRAM bandwidth 등 성능 통계 출력 */
      ctx->the_gpgpusim->g_the_gpu->update_stats();
      /* [한국어] 누적 통계 업데이트 (여러 커널에 걸친 total 통계 집계) */
      ctx->print_simulation_time();
      /* [한국어] wall-clock 기준 시뮬레이션 경과 시간과 시뮬레이션 속도 출력 */
    }

    pthread_mutex_lock(&(ctx->the_gpgpusim->g_sim_lock));
    ctx->the_gpgpusim->g_sim_active = false;
    /* [한국어] 시뮬레이션이 일시 정지 상태임을 표시.
     * synchronize()가 이 값을 확인하여 GPU가 idle인지 판단한다. */
    pthread_mutex_unlock(&(ctx->the_gpgpusim->g_sim_lock));

  } while (!ctx->the_gpgpusim->g_sim_done);
  /* [한국어] g_sim_done이 될 때까지 외부 루프 반복.
   * exit_simulation()이 g_sim_done = true로 설정하면 루프를 탈출한다. */

  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  fflush(stdout);

  if (ctx->the_gpgpusim->break_limit) {
    /* [한국어] 최대 사이클/명령 한도 초과로 인한 종료:
     * 일반 종료(sem_post)가 아닌 exit(1)로 프로세스 자체를 종료한다. */
    printf(
        "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
        "instructions) **\n");
    exit(1);  /* [한국어] 비정상 종료 코드 1 반환 — exit 시 atexit(termination_callback)가 실행된다 */
  }

  sem_post(&(ctx->the_gpgpusim->g_sim_signal_exit));
  /* [한국어] 정상 종료 통보: exit_simulation()이 sem_wait(&g_sim_signal_exit)에서 기다리고 있다. */
  return NULL;  /* [한국어] pthread 쓰레드 함수 반환값 (사용하지 않음) */
}

/* [한국어] SST 연동 시 사이클 단위로 사용된 사이클 여부 추적 플래그.
 * gpgpu_sim_thread_concurrent의 sim_cycles에 해당하는 SST 전용 전역 변수.
 * SST_Cycle()이 반환되어도 상태를 유지해야 하므로 전역으로 선언된다. */
bool sst_sim_cycles = false;

/*
 * [한국어]
 * SST_Cycle - SST 외부 시뮬레이터가 GPGPU-Sim에 1 사이클을 주입하는 인터페이스
 *
 * @return: true이면 시뮬레이션 한도(break_limit) 도달로 SST에게 종료를 알림.
 *          false이면 정상 처리 완료 또는 처리할 작업 없음.
 *
 * SST(Sandia Structural Simulation Toolkit) 모드에서는 별도 시뮬레이션 쓰레드를 생성하지 않고,
 * SST가 외부에서 이 함수를 매 사이클마다 호출하여 GPGPU-Sim을 구동한다.
 * gpgpu_sim_thread_concurrent()의 내부 루프와 같은 역할을 하되, SST 클럭에 묶인다.
 * 동작 흐름:
 *   1. 이전에 cudaThreadSynchronize 요청이 있었고 지금 완료 조건이면 SST 콜백 호출
 *   2. 스트림 큐 비어있고 GPU idle이면 g_sim_active = false 후 false 반환(아무 일 없음)
 *   3. stream_manager::operation()으로 작업 처리; 커널 완료+GPU idle이면 false 반환
 *   4. g_sim_active = true 설정
 *   5. 기능 시뮬레이션 처리
 *   6. 타이밍 시뮬레이션: SST_cycle() (gpgpu_sim::cycle()의 SST 버전)
 *   7. 최대 한도 도달 시 g_sim_done, break_limit 설정 후 true 반환
 *   8. GPU 비활성이면 통계 출력
 * 실행 컨텍스트: SST 메인 루프 쓰레드. 멀티쓰레드가 아니므로 g_sim_lock 잠금은 불필요.
 * 주의: 이 함수는 gpgpu_sim_thread_concurrent()와 달리 루프가 아니라 단일 사이클만 처리한다.
 *
 * 호출 체인:
 *   SST 외부 시뮬레이터 클럭 → [SST_Cycle]
 *     → stream_manager::operation() → stream_operation::do_operation()
 *     → gpgpu_cuda_ptx_sim_main_func() (기능 시뮬레이션)
 *     → sst_gpgpu_sim::SST_cycle() (타이밍 시뮬레이션)
 */
bool SST_Cycle() {
  // Check if Synchronize is done when SST previously requested
  // cudaThreadSynchronize
  if (GPGPU_Context()->requested_synchronize &&
      ((g_stream_manager()->empty() && !GPGPUsim_ctx_ptr()->g_sim_active) ||
       GPGPUsim_ctx_ptr()->g_sim_done)) {
    /* [한국어] 이전 사이클에서 cudaThreadSynchronize가 요청된 상태에서,
     * 지금 스트림 큐가 비어있고 GPU가 idle이거나, 또는 g_sim_done이면
     * 동기화가 완료된 것으로 판단하고 SST에 콜백으로 통지한다. */
    SST_callback_cudaThreadSynchronize_done();
    /* [한국어] SST에게 cudaThreadSynchronize 완료를 통보 (weak 심볼 — SST 없으면 no-op) */
    GPGPU_Context()->requested_synchronize = false;
    /* [한국어] 동기화 요청 플래그 해제 — 다음 사이클부터는 이 조건을 다시 체크하지 않는다 */
  }

  if (g_stream_manager()->empty_protected() &&
      !GPGPUsim_ctx_ptr()->g_sim_done && !g_the_gpu()->active()) {
    /* [한국어] 스트림 큐가 비어있고(더 보낼 작업 없음) + g_sim_done도 아니고 + GPU도 idle:
     * 이번 사이클에는 처리할 것이 없다. */
    GPGPUsim_ctx_ptr()->g_sim_active = false;
    /* [한국어] GPU가 유휴 상태임을 표시 */
    // printf("stream is empty %d \n",  g_stream_manager->empty());
    return false;  /* [한국어] SST에게 "이번 사이클 작업 없음"을 알림 */
  }

  if (g_stream_manager()->operation(&sst_sim_cycles) &&
      !g_the_gpu()->active()) {
    /* [한국어] 스트림 작업을 처리했고(커널 완료 등) GPU도 idle이면:
     * 다음 커널을 위해 init()이 필요하므로 이번 사이클은 여기서 반환한다. */
    if (sst_sim_cycles) {
      sst_sim_cycles = false;
      /* [한국어] 이전에 사이클이 진행된 적이 있으면 초기화.
       * 커널 완료 직후에는 통계 출력 전 상태를 리셋한다. */
    }
    return false;  /* [한국어] 커널 완료 처리 완료, 다음 사이클에서 재시작 */
  }

  // printf("GPGPU-Sim: Give GPU Cycle\n");
  GPGPUsim_ctx_ptr()->g_sim_active = true;
  /* [한국어] 이번 사이클에서 실제로 GPU를 구동할 것임을 표시 */

  // functional simulation
  if (g_the_gpu()->is_functional_sim()) {
    /* [한국어] 기능 시뮬레이션 모드: PTX 명령어 의미론적 실행 (타이밍 없음) */
    kernel_info_t *kernel = g_the_gpu()->get_functional_kernel();
    assert(kernel);  /* [한국어] 기능 시뮬레이션 중에는 커널이 반드시 존재해야 함 */
    GPGPUsim_ctx_ptr()->gpgpu_ctx->func_sim->gpgpu_cuda_ptx_sim_main_func(
        *kernel);
    /* [한국어] PTX 명령어를 해석하여 실제 계산을 수행 — 결과값은 시뮬레이터 메모리에 반영 */
    g_the_gpu()->finish_functional_sim(kernel);
    /* [한국어] 기능 시뮬레이션 완료 처리 */
  }

  // performance simulation
  if (g_the_gpu()->active()) {
    /* [한국어] GPU가 활성 상태(실행 중인 warp 또는 대기 중인 메모리 요청 존재) */
    g_the_gpu()->SST_cycle();
    /* [한국어] SST 전용 타이밍 시뮬레이션 1 사이클 전진.
     * 일반 gpgpu_sim::cycle()과 동일하게 SM/캐시/DRAM/NoC를 전진시키지만
     * SST 클럭 도메인에 맞게 래핑되어 있다. */
    sst_sim_cycles = true;  /* [한국어] 이번 사이클에서 실제로 시뮬레이션이 수행됨 */
    g_the_gpu()->deadlock_check();
    /* [한국어] 교착 상태 감지 — 일정 사이클 이상 아무 진전 없으면 abort */
  } else {
    /* [한국어] GPU가 비활성 상태 */
    if (g_the_gpu()->cycle_insn_cta_max_hit()) {
      /* [한국어] 최대 사이클/명령/CTA 한도에 도달 */
      g_stream_manager()->stop_all_running_kernels();
      /* [한국어] 실행 중인 모든 커널 강제 종료 */
      GPGPUsim_ctx_ptr()->g_sim_done = true;
      /* [한국어] 시뮬레이션 종료 플래그 설정 */
      GPGPUsim_ctx_ptr()->g_sim_active = false;
      /* [한국어] GPU 비활성 표시 */
      GPGPUsim_ctx_ptr()->break_limit = true;
      /* [한국어] 한도 도달로 인한 종료임을 표시 */
    }
  }

  if (!g_the_gpu()->active()) {
    /* [한국어] GPU가 이번 사이클 후 idle 상태가 되었으면 커널 단위 통계 출력 */
    g_the_gpu()->print_stats(GPGPUsim_ctx_ptr()->g_the_gpu->last_streamID);
    /* [한국어] 성능 통계 출력 (IPC, cache, DRAM bandwidth 등) */
    g_the_gpu()->update_stats();
    /* [한국어] 누적 통계 업데이트 */
    GPGPU_Context()->print_simulation_time();
    /* [한국어] 경과 시간 및 시뮬레이션 속도 출력 */
  }

  if (GPGPUsim_ctx_ptr()->break_limit) {
    /* [한국어] 최대 한도 도달: SST에게 true를 반환하여 시뮬레이션 종료를 알림 */
    printf(
        "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
        "instructions) **\n");
    return true;  /* [한국어] SST는 이 true를 받아 시뮬레이션 루프를 종료한다 */
  }

  return false;  /* [한국어] 정상 사이클 처리 완료 — SST는 다음 사이클을 계속 진행 */
}

/*
 * [한국어]
 * gpgpu_context::synchronize - 호스트가 GPU 시뮬레이션 완료를 바쁜 대기(busy-wait)로 기다림
 *
 * @return: void
 *
 * CUDA의 cudaDeviceSynchronize() / cudaThreadSynchronize() 에 해당하는 시뮬레이터 구현이다.
 * 호스트 쓰레드가 GPU 시뮬레이션 쓰레드가 모든 작업을 마칠 때까지 블록된다.
 * 완료 조건 (둘 중 하나):
 *   1) 스트림 큐가 비어있고(empty()) g_sim_active == false (GPU idle)
 *   2) g_sim_done == true (시뮬레이션 전체 종료)
 * g_sim_lock 뮤텍스로 g_sim_active를 안전하게 읽는다 (데이터 레이스 방지).
 * 현재 구현은 바쁜 대기(spin-wait)이므로 대기 시간이 길면 CPU 사이클을 낭비한다.
 * (주석 처리된 sem_wait/sem_post는 세마포어 기반 구현의 이전 버전 흔적이다.)
 * 실행 컨텍스트: 호스트 쓰레드 (libcuda에서 cudaDeviceSynchronize 호출 시).
 *
 * 호출 체인:
 *   libcuda/cuda_runtime_api.cc의 cudaDeviceSynchronize() → [synchronize]
 *   [이 함수] → stream_manager::empty(), g_sim_active 검사 (g_sim_lock 보호)
 */
void gpgpu_context::synchronize() {
  printf("GPGPU-Sim: synchronize waiting for inactive GPU simulation\n");
  the_gpgpusim->g_stream_manager->print(stdout);
  /* [한국어] 현재 스트림 큐 상태를 출력하여 어떤 작업이 아직 남아있는지 확인 가능하게 함 */
  fflush(stdout);
  //    sem_wait(&g_sim_signal_finish);
  /* [한국어] 이전에는 세마포어로 기다렸으나, 현재는 뮤텍스 기반 폴링으로 변경됨 */
  bool done = false;  /* [한국어] 대기 루프 탈출 조건 */
  do {
    pthread_mutex_lock(&(the_gpgpusim->g_sim_lock));
    /* [한국어] g_sim_active와 g_sim_done을 안전하게 읽기 위해 잠금.
     * 시뮬레이션 쓰레드도 같은 뮤텍스로 g_sim_active를 수정한다. */
    done = (the_gpgpusim->g_stream_manager->empty() &&
            !the_gpgpusim->g_sim_active) ||
           the_gpgpusim->g_sim_done;
    /* [한국어] 완료 조건 평가:
     * (큐 비었고 GPU idle) 또는 (전체 종료 플래그) 중 하나가 참이면 done = true */
    pthread_mutex_unlock(&(the_gpgpusim->g_sim_lock));
  } while (!done);
  /* [한국어] done이 될 때까지 폴링(spin-wait). CPU 코어를 계속 소비하는 바쁜 대기이다. */

  printf("GPGPU-Sim: detected inactive GPU simulation thread\n");
  fflush(stdout);
  //    sem_post(&g_sim_signal_start);
  /* [한국어] 이전 세마포어 방식의 흔적 — 현재는 사용하지 않음 */
}

/*
 * [한국어]
 * gpgpu_context::synchronize_check - SST 모드 전용 비차단(non-blocking) 동기화 완료 확인
 *
 * @return: true이면 GPU 시뮬레이션이 완료 상태 (동기화 조건 충족),
 *          false이면 아직 진행 중 (SST는 다음 사이클에 다시 확인해야 함).
 *
 * synchronize()가 블록 대기(busy-wait)인 것과 달리, 이 함수는 즉시 반환한다.
 * SST 모드에서는 SST의 클럭 루프 안에서 매 사이클마다 동기화 완료 여부를 확인해야 하므로,
 * 블록되지 않는 이 함수가 사용된다.
 * requested_synchronize = true로 설정하면, SST_Cycle()이 매 사이클에서 이 플래그를 보고
 * 완료 조건 충족 시 SST_callback_cudaThreadSynchronize_done()을 호출한다.
 * 실행 컨텍스트: 호스트 쓰레드(libcuda에서 cudaDeviceSynchronize 호출 시, SST 모드).
 *
 * 호출 체인:
 *   libcuda(SST 모드) → [synchronize_check]
 *   → 이후 SST_Cycle() 내에서 requested_synchronize 플래그 확인 → SST_callback_cudaThreadSynchronize_done()
 */
bool gpgpu_context::synchronize_check() {
  // printf("GPGPU-Sim: synchronize checking for inactive GPU simulation\n");
  requested_synchronize = true;
  /* [한국어] 동기화 요청 플래그를 true로 설정.
   * SST_Cycle()이 이 플래그를 확인하여 완료 조건이 충족되면 콜백을 호출한다. */

  the_gpgpusim->g_stream_manager->print(stdout);
  /* [한국어] 현재 스트림 큐 상태를 출력 — 어떤 작업이 남아있는지 진단 가능 */
  fflush(stdout);
  //    sem_wait(&g_sim_signal_finish);
  /* [한국어] 이전 세마포어 기반 구현의 흔적 — 현재 SST 모드에서는 사용하지 않음 */

  bool done = false;  /* [한국어] 동기화 완료 여부 */

  pthread_mutex_lock(&(the_gpgpusim->g_sim_lock));
  /* [한국어] g_sim_active를 안전하게 읽기 위해 잠금 */
  done = (the_gpgpusim->g_stream_manager->empty() &&
          !the_gpgpusim->g_sim_active) ||
         the_gpgpusim->g_sim_done;
  /* [한국어] synchronize()와 동일한 완료 조건 평가:
   * (큐 비었고 GPU idle) 또는 (전체 종료 플래그) */
  pthread_mutex_unlock(&(the_gpgpusim->g_sim_lock));

  if (done) {
    /* [한국어] 이미 완료 상태이면 즉시 메시지 출력 후 true 반환 */
    printf(
        "GPGPU-Sim: synchronize checking: detected inactive GPU simulation "
        "thread\n");
  }
  fflush(stdout);
  return done;  /* [한국어] 완료이면 true, 아직 진행 중이면 false */
}

/*
 * [한국어]
 * gpgpu_context::exit_simulation - 시뮬레이션 전체 종료 요청 및 쓰레드 종료 대기
 *
 * @return: void
 *
 * CUDA 애플리케이션이 모든 커널을 실행한 후 시뮬레이터를 종료할 때 호출된다.
 * 동작 흐름:
 *   1) g_sim_done = true 설정 → 시뮬레이션 쓰레드의 외부 루프 탈출 유도
 *   2) sem_wait(&g_sim_signal_exit)으로 시뮬레이션 쓰레드가 실제로 종료될 때까지 블록
 *   3) 시뮬레이션 쓰레드가 sem_post(&g_sim_signal_exit)를 호출하면 깨어나 반환
 * sequential 모드에서는 gpgpu_sim_thread_sequential()이 마지막에 sem_post를 호출하고,
 * concurrent 모드에서는 gpgpu_sim_thread_concurrent()가 루프 탈출 후 sem_post를 호출한다.
 * break_limit이면 exit(1)로 이미 프로세스가 종료되므로 이 함수가 호출되지 않을 수 있다.
 * 실행 컨텍스트: 호스트 쓰레드 (libcuda에서 시뮬레이터 종료 시).
 *
 * 호출 체인:
 *   libcuda/cuda_runtime_api.cc → [exit_simulation]
 *   [이 함수] → g_sim_done = true → 시뮬레이션 쓰레드 루프 탈출
 *   시뮬레이션 쓰레드 → sem_post(g_sim_signal_exit) → [이 함수] sem_wait 깨어남
 */
void gpgpu_context::exit_simulation() {
  the_gpgpusim->g_sim_done = true;
  /* [한국어] 시뮬레이션 종료 신호: 시뮬레이션 쓰레드의 외부 do-while 루프 탈출 조건 */
  printf("GPGPU-Sim: exit_simulation called\n");
  fflush(stdout);
  sem_wait(&(the_gpgpusim->g_sim_signal_exit));
  /* [한국어] 시뮬레이션 쓰레드가 실제로 종료(sem_post)될 때까지 블록.
   * 이 세마포어는 gpgpu_ptx_sim_init_perf()에서 sem_init으로 0으로 초기화되었다. */
  printf("GPGPU-Sim: simulation thread signaled exit\n");
  fflush(stdout);
}

/*
 * [한국어]
 * gpgpu_context::gpgpu_ptx_sim_init_perf - 타이밍 시뮬레이터 전체 초기화
 *
 * @return: 생성된 gpgpu_sim* 포인터 (exec_gpgpu_sim 또는 sst_gpgpu_sim 인스턴스).
 *          호출자(libcuda)는 이 포인터를 통해 GPU 시뮬레이터를 제어한다.
 *
 * 시뮬레이터 최초 실행 시 한 번 호출되며, 시뮬레이션에 필요한 모든 객체를 생성하고 초기화한다.
 * 동작 흐름:
 *   1) 난수 시드 고정(srand(1)) → 재현 가능한 시뮬레이션을 위해
 *   2) 스플래시 화면 출력 (버전 정보)
 *   3) 환경 변수 읽기 (func_sim, ptx_parser)
 *   4) option_parser 생성 → PTX/기능 시뮬레이션/NoC/GPU 마이크로아키텍처 옵션 등록
 *   5) gpgpusim.config 파일 파싱 (option_parser_cmdline)
 *   6) 로케일을 "C"로 설정 → 소수점 구분자가 "."이 되도록 (환경 독립성)
 *   7) GPU 설정 초기화 (gpgpu_sim_config::init)
 *   8) SST 모드이면 sst_gpgpu_sim, 아니면 exec_gpgpu_sim 객체 생성
 *   9) 스트림 관리자(stream_manager) 생성
 *   10) 시뮬레이션 시작 시간 기록
 *   11) 세마포어 3개(start/finish/exit) 초기화 (값 0으로 시작 → 신호 올 때까지 wait 블록)
 * 실행 컨텍스트: 호스트 쓰레드. 시뮬레이터 전체 수명 중 단 한 번 호출된다.
 *
 * 호출 체인:
 *   libcuda/cuda_runtime_api.cc의 첫 CUDA API 호출 → [gpgpu_ptx_sim_init_perf]
 *     → option_parser_create / option_parser_cmdline (option_parser.cc)
 *     → new exec_gpgpu_sim / sst_gpgpu_sim (gpgpu-sim/gpu-sim.cc)
 *     → new stream_manager (stream_manager.cc)
 */
gpgpu_sim *gpgpu_context::gpgpu_ptx_sim_init_perf() {
  srand(1);
  /* [한국어] 난수 생성기 시드를 1로 고정.
   * 시뮬레이션 재현성(reproducibility)을 위해 매 실행마다 동일한 난수 시퀀스를 사용한다. */

  print_splash();
  /* [한국어] GPGPU-Sim 버전 및 저작권 정보를 stdout에 출력하는 스플래시 화면 */

  func_sim->read_sim_environment_variables();
  /* [한국어] CUDA 기능 시뮬레이션 관련 환경 변수 읽기
   * (예: PTX_SIM_MODE, GPGPU_PTX_SIM_API_PRINT 등) */

  ptx_parser->read_parser_environment_variables();
  /* [한국어] PTX 파서 관련 환경 변수 읽기 (예: PTX_PARSER_PRINT_IR 등) */

  option_parser_t opp = option_parser_create();
  /* [한국어] gpgpusim.config 파일을 파싱하기 위한 옵션 파서 생성.
   * option_parser_t는 등록된 옵션들을 관리하는 핸들이다. */

  ptx_reg_options(opp);
  /* [한국어] PTX 실행 관련 옵션을 파서에 등록
   * (예: -gpgpu_ptx_instruction_classification, -gpgpu_ptx_sim_mode 등) */

  func_sim->ptx_opcocde_latency_options(opp);
  /* [한국어] PTX 명령어 레이턴시(예: DP FP, SP FP, INT 연산 레이턴시) 옵션 등록
   * 참고: "opcocde"는 "opcode"의 오타이나 원본 코드를 유지함 */

  icnt_reg_options(opp);
  /* [한국어] Booksim NoC(네트워크온칩) 관련 옵션 등록
   * (예: -network_mode, -inter_config_file 등 icnt_wrapper.h의 설정들) */

  the_gpgpusim->g_the_gpu_config = new gpgpu_sim_config(this);
  /* [한국어] GPU 마이크로아키텍처 설정 객체 생성.
   * gpgpu_sim_config는 SM 수, 캐시 크기, DRAM 타이밍 등 모든 하드웨어 파라미터를 담는다. */

  the_gpgpusim->g_the_gpu_config->reg_options(
      opp);  // register GPU microrachitecture options
  /* [한국어] GPU 마이크로아키텍처 옵션을 파서에 등록
   * (예: -gpgpu_n_clusters, -gpgpu_n_cores_per_cluster, -gpgpu_cache:il1 등) */

  option_parser_cmdline(opp, sg_argc, sg_argv);  // parse configuration options
  /* [한국어] sg_argv = {"", "-config", "gpgpusim.config"}를 파싱하여
   * gpgpusim.config 파일을 읽고 등록된 모든 옵션 값을 설정한다. */

  fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
  option_parser_print(opp, stdout);
  /* [한국어] 파싱된 모든 설정 값을 stdout에 출력 — 재현성을 위해 실제 사용된 설정을 기록 */

  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));
  /* [한국어] 수치 로케일을 표준 C 로케일로 설정.
   * 일부 유럽 시스템은 소수점으로 쉼표(,)를 사용하는데, 이 경우 "3.14"를 올바르게 파싱하지 못한다.
   * "C" 로케일은 항상 점(.)을 소수점으로 사용하므로 OS 환경에 독립적이다. */

  the_gpgpusim->g_the_gpu_config->init();
  /* [한국어] 파싱된 설정값으로 GPU 설정 객체를 완전히 초기화.
   * SM 개수 계산, 캐시 크기 검증, DRAM 채널 구성 등 파생 값들을 계산한다. */

  if (the_gpgpusim->g_the_gpu_config->is_SST_mode()) {
    // Create SST specific GPGPUSim
    the_gpgpusim->g_the_gpu =
        new sst_gpgpu_sim(*(the_gpgpusim->g_the_gpu_config), this);
    /* [한국어] SST 모드: SST와 연동하는 특수 GPU 시뮬레이터 생성.
     * sst_gpgpu_sim은 exec_gpgpu_sim을 상속하되 SST_cycle() 등 SST 전용 인터페이스를 추가한다. */
  } else {
    the_gpgpusim->g_the_gpu =
        new exec_gpgpu_sim(*(the_gpgpusim->g_the_gpu_config), this);
    /* [한국어] 일반(execution-driven) 모드: 표준 GPGPU-Sim 타이밍 시뮬레이터 생성.
     * exec_gpgpu_sim은 gpgpu_sim을 상속하여 완전한 사이클-레벨 시뮬레이션을 수행한다. */
  }

  the_gpgpusim->g_stream_manager = new stream_manager(
      (the_gpgpusim->g_the_gpu), func_sim->g_cuda_launch_blocking);
  /* [한국어] 스트림 관리자 생성:
   * g_the_gpu: 스트림 작업 실행 시 실제 GPU 시뮬레이터에 명령을 내릴 포인터
   * g_cuda_launch_blocking: true이면 stream 0 동작(커널 완료 후 다음 진행). */

  the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);
  /* [한국어] 시뮬레이션 시작 wall-clock 시간 기록 (Unix timestamp).
   * print_simulation_time()에서 경과 시간 계산 시 이 값을 기준으로 한다. */

  sem_init(&(the_gpgpusim->g_sim_signal_start), 0, 0);
  /* [한국어] g_sim_signal_start 세마포어를 초기값 0으로 초기화.
   * 초기값 0: sem_wait()이 즉시 블록됨 → 처음에는 시뮬레이션 시작 신호를 기다린다.
   * 두 번째 인수 0: 프로세스 내 쓰레드 간 공유 (0이 아니면 프로세스 간 공유) */

  sem_init(&(the_gpgpusim->g_sim_signal_finish), 0, 0);
  /* [한국어] g_sim_signal_finish 세마포어를 초기값 0으로 초기화.
   * gpgpu_opencl_ptx_sim_main_perf()에서 커널 완료 통보를 기다리는 데 사용된다. */

  sem_init(&(the_gpgpusim->g_sim_signal_exit), 0, 0);
  /* [한국어] g_sim_signal_exit 세마포어를 초기값 0으로 초기화.
   * exit_simulation()에서 시뮬레이션 쓰레드 종료를 기다리는 데 사용된다. */

  return the_gpgpusim->g_the_gpu;
  /* [한국어] 생성된 GPU 시뮬레이터 포인터 반환 — 호출자(libcuda)가 이후 커널 실행에 사용한다. */
}

/*
 * [한국어]
 * gpgpu_context::start_sim_thread - 시뮬레이션 전용 쓰레드 생성 또는 SST 초기화
 *
 * @api: 시뮬레이션 API 유형.
 *       1 = concurrent 모드 (gpgpu_sim_thread_concurrent, 동시 커널 지원)
 *       2 = sequential 모드 (gpgpu_sim_thread_sequential, OpenCL 등 순차 커널)
 * @return: void
 *
 * 커널 launch 요청이 처음 들어오거나 이전 시뮬레이션이 완료된 후(g_sim_done == true)에만
 * 새 시뮬레이션 쓰레드를 생성한다. 이미 실행 중(g_sim_done == false)이면 아무 일도 하지 않는다.
 * SST 모드에서는 별도 쓰레드 대신 GPU를 직접 init()하고 클럭은 SST_Cycle()이 외부에서 공급한다.
 * 실행 컨텍스트: 호스트 쓰레드 (libcuda에서 커널 launch 시).
 *
 * 호출 체인:
 *   libcuda/cuda_runtime_api.cc → [start_sim_thread]
 *     → pthread_create(gpgpu_sim_thread_concurrent)  (api == 1, 비-SST)
 *     → pthread_create(gpgpu_sim_thread_sequential)  (api != 1, 비-SST)
 *     → g_the_gpu()->init()                          (SST 모드)
 */
void gpgpu_context::start_sim_thread(int api) {
  if (the_gpgpusim->g_sim_done) {
    /* [한국어] 이전 시뮬레이션이 완전히 종료된 상태에서만 새 쓰레드를 시작한다.
     * g_sim_done이 false이면 아직 시뮬레이션 쓰레드가 실행 중이므로 새 쓰레드를 만들지 않는다. */

    the_gpgpusim->g_sim_done = false;
    /* [한국어] 시뮬레이션 진행 중 상태로 전환.
     * 이 값을 false로 설정해야 시뮬레이션 쓰레드의 루프가 종료되지 않고 실행된다. */

    if (the_gpgpusim->g_the_gpu_config->is_SST_mode()) {
      // Do not create concurrent thread in SST mode
      g_the_gpu()->init();
      /* [한국어] SST 모드: 별도 쓰레드 없이 GPU를 직접 초기화.
       * SST가 SST_Cycle()을 호출하여 클럭을 외부에서 공급하므로 내부 쓰레드가 필요 없다.
       * init()은 SM 상태, warp 스케줄러, 캐시 등을 초기화한다. */
    } else {
      if (api == 1) {
        pthread_create(&(the_gpgpusim->g_simulation_thread), NULL,
                       gpgpu_sim_thread_concurrent, (void *)this);
        /* [한국어] CUDA concurrent 모드: 동시 커널 실행을 지원하는 시뮬레이션 쓰레드 생성.
         * g_simulation_thread: 생성된 쓰레드의 식별자(ID) 저장.
         * NULL: 기본 쓰레드 속성 사용.
         * (void*)this: gpgpu_context 포인터를 쓰레드 함수에 전달. */
      } else {
        pthread_create(&(the_gpgpusim->g_simulation_thread), NULL,
                       gpgpu_sim_thread_sequential, (void *)this);
        /* [한국어] 순차 모드(OpenCL 등): 커널 하나씩 순서대로 실행하는 시뮬레이션 쓰레드 생성.
         * gpgpu_opencl_ptx_sim_main_perf()가 세마포어로 이 쓰레드와 동기화한다. */
      }
    }
  }
}

/*
 * [한국어]
 * gpgpu_context::print_simulation_time - 시뮬레이션 경과 시간과 시뮬레이션 속도 출력
 *
 * @return: void
 *
 * 커널 완료 시 또는 시뮬레이터 종료 시 호출되어 다음 통계를 stdout에 출력한다:
 *   - 경과 wall-clock 시간 (일/시/분/초 및 총 초)
 *   - 시뮬레이션 속도 (명령어/초, 사이클/초)
 *   - Silicon slowdown: 실제 GPU 클럭 대비 시뮬레이터 속도 비율
 *     예) 실제 GPU 1GHz에서 시뮬레이터가 초당 1000 사이클이면 slowdown = 1,000,000x
 * MAX 매크로를 사용해 difference가 최소 1이 되도록 보장 → division by zero 방지.
 * 실행 컨텍스트: 시뮬레이션 쓰레드 (gpgpu_sim_thread_concurrent/sequential) 또는 SST 클럭 쓰레드.
 *
 * 호출 체인:
 *   gpgpu_sim_thread_concurrent() → [print_simulation_time]
 *   gpgpu_sim_thread_sequential() → [print_simulation_time]
 *   SST_Cycle() → GPGPU_Context()->print_simulation_time() → [이 함수]
 */
void gpgpu_context::print_simulation_time() {
  time_t current_time, difference, d, h, m, s;
  /* [한국어] time_t: Unix timestamp 타입 (1970-01-01 00:00:00 UTC 기준 경과 초 수) */

  current_time = time((time_t *)NULL);
  /* [한국어] 현재 wall-clock 시간 획득. NULL을 전달하면 반환값만 사용한다. */

  difference = MAX(current_time - the_gpgpusim->g_simulation_starttime, 1);
  /* [한국어] 시뮬레이션 시작 이후 경과 초 수 계산.
   * MAX(..., 1)로 최솟값을 1초로 보장 → 이후 나눗셈에서 division by zero 방지.
   * g_simulation_starttime은 gpgpu_ptx_sim_init_perf()에서 기록되었다. */

  d = difference / (3600 * 24);              /* [한국어] 경과 일 수 (86400초 = 1일) */
  h = difference / 3600 - 24 * d;           /* [한국어] 일 단위 제외 후 경과 시간 수 */
  m = difference / 60 - 60 * (h + 24 * d);  /* [한국어] 시간 단위 제외 후 경과 분 수 */
  s = difference - 60 * (m + 60 * (h + 24 * d));  /* [한국어] 분 단위 제외 후 경과 초 수 */

  fflush(stderr);  /* [한국어] stderr 버퍼 플러시 — 경고/에러 메시지가 시간 통계보다 먼저 출력되도록 */

  printf(
      "\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u sec)\n",
      (unsigned)d, (unsigned)h, (unsigned)m, (unsigned)s, (unsigned)difference);
  /* [한국어] 경과 시간을 일/시/분/초와 총 초 수로 출력 */

  printf("gpgpu_simulation_rate = %u (inst/sec)\n",
         (unsigned)(the_gpgpusim->g_the_gpu->gpu_tot_sim_insn / difference));
  /* [한국어] 초당 시뮬레이션 명령어 수 출력.
   * gpu_tot_sim_insn: 시뮬레이션된 총 명령어 수 (모든 커널 누적) */

  const unsigned cycles_per_sec =
      (unsigned)(the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle / difference);
  /* [한국어] 초당 시뮬레이션 사이클 수 계산.
   * gpu_tot_sim_cycle: 시뮬레이션된 총 사이클 수 (모든 커널 누적) */

  printf("gpgpu_simulation_rate = %u (cycle/sec)\n", cycles_per_sec);
  /* [한국어] 초당 시뮬레이션 사이클 수 출력 */

  if (cycles_per_sec == 0) {
    /* [한국어] 1초도 안 걸렸거나 사이클이 0이면 slowdown 계산 불가 */
    printf("gpgpu_silicon_slowdown = Nan\n");
  } else {
    printf("gpgpu_silicon_slowdown = %ux\n",
           the_gpgpusim->g_the_gpu->shader_clock() * 1000 / cycles_per_sec);
    /* [한국어] 실리콘 slowdown 배율 출력:
     * shader_clock()은 gpgpusim.config의 -gpgpu_clock_domains 설정에서 읽은
     * 시뮬레이션 대상 GPU의 코어 클럭 속도 (MHz 단위).
     * shader_clock() * 1000: MHz → kHz 변환 (사이클/초와 단위 맞춤)
     * slowdown = (시뮬레이션 목표 클럭 kHz) / (실제 시뮬레이터 사이클/초)
     * 예) 목표 1GHz, 시뮬레이터 1000 cycle/sec → slowdown = 1,000,000x */
  }
  fflush(stdout);  /* [한국어] 버퍼 플러시 — 출력이 지연 없이 기록되도록 */
}

/*
 * [한국어]
 * gpgpu_context::gpgpu_opencl_ptx_sim_main_perf - OpenCL 커널의 타이밍 시뮬레이션 진입점
 *
 * @grid: 실행할 OpenCL 커널의 정보를 담은 kernel_info_t 포인터.
 *        커널 이름, 그리드/블록 차원, 인수(arguments) 등이 포함된다.
 * @return: 0 (성공). 에러 반환 경로는 없으며, 실패 시 내부 assert로 종료된다.
 *
 * OpenCL clEnqueueNDRangeKernel() API가 libopencl을 통해 이 함수로 리다이렉트된다.
 * 순차(sequential) 시뮬레이션 모드 전용이다:
 *   1) g_the_gpu->launch(grid): GPU 시뮬레이터에 커널을 등록 (CTA 디스패치 큐에 추가)
 *   2) sem_post(&g_sim_signal_start): 시뮬레이션 쓰레드에 "시작" 신호 전송
 *      → gpgpu_sim_thread_sequential()이 sem_wait에서 깨어나 커널을 실행하기 시작
 *   3) sem_wait(&g_sim_signal_finish): 커널 완료 신호를 기다리며 블록
 *      → 시뮬레이션 쓰레드가 커널 완료 후 sem_post하면 깨어나 반환
 * 실행 컨텍스트: 호스트 쓰레드 (OpenCL 런타임에서 커널 실행 요청 시).
 *
 * 호출 체인:
 *   libopencl → [gpgpu_opencl_ptx_sim_main_perf]
 *     → g_the_gpu->launch(grid)          — 커널 등록
 *     → sem_post(g_sim_signal_start)     — 시뮬레이션 쓰레드 깨우기
 *     → sem_wait(g_sim_signal_finish)    — 커널 완료 대기
 */
int gpgpu_context::gpgpu_opencl_ptx_sim_main_perf(kernel_info_t *grid) {
  the_gpgpusim->g_the_gpu->launch(grid);
  /* [한국어] GPU 시뮬레이터에 커널을 등록.
   * launch()는 kernel_info_t를 GPU 내부 커널 실행 큐에 추가하고
   * CTA(스레드 블록) 디스패치를 준비한다. */

  sem_post(&(the_gpgpusim->g_sim_signal_start));
  /* [한국어] 시뮬레이션 시작 신호를 시뮬레이션 쓰레드로 전송.
   * gpgpu_sim_thread_sequential()이 sem_wait(&g_sim_signal_start)에서 블록 중이므로
   * 이 post로 깨어나 커널 실행을 시작한다. */

  sem_wait(&(the_gpgpusim->g_sim_signal_finish));
  /* [한국어] 커널 완료 신호를 기다리며 블록.
   * 시뮬레이션 쓰레드가 커널 실행을 마치고 sem_post(&g_sim_signal_finish)를
   * 호출하면 깨어나 반환한다. 이 동안 호스트 쓰레드는 아무 일도 하지 않는다. */

  return 0;  /* [한국어] 성공 반환. OpenCL API는 clEnqueueNDRangeKernel의 반환값을 확인한다. */
}

//! Functional simulation of OpenCL
/*!
 * This function call the CUDA PTX functional simulator
 */
/*
 * [한국어]
 * cuda_sim::gpgpu_opencl_ptx_sim_main_func - OpenCL 커널의 기능 시뮬레이션 진입점
 *
 * @grid: 실행할 OpenCL 커널의 kernel_info_t 포인터.
 * @return: 0 (항상 성공).
 *
 * OpenCL 커널을 PTX 기능 시뮬레이터(CUDA PTX 시뮬레이터)로 실행한다.
 * gpgpu_cuda_ptx_sim_main_func()를 is_opencl=true 플래그와 함께 호출한다.
 * 이 플래그가 필요한 이유:
 *   CUDA 커널은 진입/종료 시 커널 등록/해제 절차가 있는데,
 *   OpenCL 커널은 처음부터 이 절차가 없었으므로 종료 시에도 해제 처리를 하지 않아야 한다.
 *   is_opencl=true로 이 차이를 구분한다.
 * 실행 컨텍스트: 기능 시뮬레이션 모드(타이밍 없이 PTX 명령어만 실행).
 *
 * 호출 체인:
 *   libopencl → [gpgpu_opencl_ptx_sim_main_func]
 *     → gpgpu_cuda_ptx_sim_main_func(*grid, true) — PTX 기능 시뮬레이터 (cuda-sim/cuda-sim.cc)
 */
int cuda_sim::gpgpu_opencl_ptx_sim_main_func(kernel_info_t *grid) {
  // calling the CUDA PTX simulator, sending the kernel by reference and a flag
  // set to true, the flag used by the function to distinguish OpenCL calls from
  // the CUDA simulation calls which it is needed by the called function to not
  // register the exit the exit of OpenCL kernel as it doesn't register entering
  // in the first place as the CUDA kernels does
  gpgpu_cuda_ptx_sim_main_func(*grid, true);
  /* [한국어] PTX 기능 시뮬레이터 호출.
   * *grid: kernel_info_t 객체를 참조로 전달 (복사 없이 커널 정보 접근).
   * true: is_opencl 플래그 — CUDA와 달리 OpenCL 커널 종료 처리 방식을 사용하도록 지시. */

  return 0;  /* [한국어] 성공 반환. */
}
