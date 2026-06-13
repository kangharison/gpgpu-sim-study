/*
 * [한국어 설명] GPGPU-Sim 시뮬레이터 진입점 헤더 (gpgpusim_entrypoint.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPUsim_ctx 클래스를 정의하며, GPGPU-Sim 시뮬레이터의
 * 전체 생명주기(초기화 → 실행 → 동기화 → 종료)에 필요한 상태 변수들을
 * 하나의 구조로 묶는다. 이 클래스는 "시뮬레이터 제어 블록"으로, 시뮬레이션
 * 쓰레드와 호스트 쓰레드 사이의 동기화 객체(세마포어, 뮤텍스), 시뮬레이터의
 * 실행 상태 플래그(g_sim_active, g_sim_done, break_limit), 그리고 핵심 서브시스템
 * (GPU 시뮬레이터, 스트림 관리자, CUDA 디바이스/컨텍스트)의 포인터를 보관한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 이 클래스는 CUDA 애플리케이션과 GPGPU-Sim 내부 시뮬레이션 엔진 사이의
 * "중간 관제탑" 역할을 한다:
 *   CUDA Application
 *     → libcuda (libcuda/cuda_runtime_api.cc) — CUDA API 인터셉트
 *         → gpgpu_context (libcuda/gpgpu_context.h) — 전역 컨텍스트 싱글톤
 *             ├── GPGPUsim_ctx (이 파일) — 시뮬레이터 상태/세마포어/핵심 객체 포인터
 *             │     ├── gpgpu_sim_config* — GPU 마이크로아키텍처 설정
 *             │     ├── gpgpu_sim* (exec/sst) — 사이클-레벨 타이밍 시뮬레이터
 *             │     └── stream_manager* — CUDA 스트림 큐 관리
 *             └── gpgpusim_entrypoint.cc — 시뮬레이터 생명주기 구현
 * 실행 컨텍스트: 호스트 사용자 공간(user-space). GPGPUsim_ctx 인스턴스는
 * gpgpu_context 싱글톤이 소유하며, 프로세스 종료까지 유지된다.
 *
 * === 타 모듈과의 연결 ===
 * 의존(include)하는 헤더:
 *   - pthread.h           : pthread_t(쓰레드 ID), pthread_mutex_t(뮤텍스) 타입
 *   - semaphore.h         : sem_t(POSIX 세마포어) 타입
 *   - time.h              : time_t(Unix timestamp) 타입
 *   - abstract_hardware_model.h : GPGPU-Sim 전반의 하드웨어 추상 모델
 * 이 파일을 include하는 모듈:
 *   - gpgpusim_entrypoint.cc : GPGPUsim_ctx 멤버 초기화 및 생명주기 구현
 *   - libcuda/gpgpu_context.h : gpgpu_context가 GPGPUsim_ctx* the_gpgpusim 포인터 보유
 * 공유 자료구조:
 *   - gpgpu_context (libcuda/gpgpu_context.h): 이 파일의 gpgpu_ctx 필드가 가리킴
 *   - gpgpu_sim (gpgpu-sim/gpu-sim.h): g_the_gpu 포인터로 접근
 *   - stream_manager (stream_manager.h): g_stream_manager 포인터로 접근
 *
 * === 주요 함수/구조체 요약 ===
 * - GPGPUsim_ctx 생성자 : 모든 상태 플래그와 포인터를 초기 안전 값(false/NULL)으로 초기화
 * - g_sim_signal_start  : 호스트→시뮬레이션 쓰레드 "커널 시작" 신호 세마포어
 * - g_sim_signal_finish : 시뮬레이션 쓰레드→호스트 "커널 완료" 신호 세마포어
 * - g_sim_signal_exit   : 시뮬레이션 쓰레드→호스트 "쓰레드 종료" 신호 세마포어
 * - g_sim_active        : 현재 사이클이 진행 중인지 여부 (g_sim_lock으로 보호)
 * - g_sim_done          : 시뮬레이션 전체 종료 조건 플래그
 * - break_limit         : 최대 사이클/명령 한도 도달로 인한 강제 종료 플래그
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

#ifndef GPGPUSIM_ENTRYPOINT_H_INCLUDED  /* [한국어] 인클루드 가드 시작: 이 헤더가 여러 번 포함되는 것을 방지 */
#define GPGPUSIM_ENTRYPOINT_H_INCLUDED  /* [한국어] 이 매크로를 정의하여 두 번째 include 시 내용을 건너뜀 */

#include <pthread.h>
/* [한국어] POSIX 쓰레드 라이브러리.
 * 사용 이유: GPGPU-Sim은 시뮬레이션을 별도 쓰레드(gpgpu_sim_thread_concurrent 등)에서
 * 실행한다. 이 헤더에서 필요한 타입:
 *   - pthread_t          : 시뮬레이션 쓰레드 식별자 (g_simulation_thread 필드 타입)
 *   - pthread_mutex_t    : 시뮬레이션 상태 보호 뮤텍스 (g_sim_lock 필드 타입)
 *   - PTHREAD_MUTEX_INITIALIZER : 정적 뮤텍스 초기화 매크로 */

#include <semaphore.h>
/* [한국어] POSIX 세마포어(counting semaphore) 라이브러리.
 * 사용 이유: 호스트 쓰레드와 시뮬레이션 쓰레드 사이의 단방향 신호 전달에 사용된다.
 * 세마포어는 뮤텍스와 달리 "소유권(ownership)" 개념이 없어, 한 쓰레드가 post하고
 * 다른 쓰레드가 wait하는 교차 쓰레드 신호 전달에 적합하다.
 * 이 헤더에서 필요한 타입:
 *   - sem_t : 세마포어 객체 타입 (g_sim_signal_start/finish/exit 필드 타입) */

#include <time.h>
/* [한국어] C 표준 시간 라이브러리.
 * 사용 이유: 시뮬레이션 경과 시간(wall-clock time)을 측정하기 위해 필요하다.
 * 이 헤더에서 필요한 타입:
 *   - time_t : Unix timestamp 타입 (g_simulation_starttime 필드 타입)
 * 사용 방법: time(NULL)로 현재 시간을 얻고 시작 시간과 빼면 경과 초 수를 구한다. */

#include "abstract_hardware_model.h"
/* [한국어] GPGPU-Sim의 하드웨어 추상 계층 헤더.
 * 사용 이유: GPGPUsim_ctx의 생성자 매개변수인 gpgpu_context 전방 선언과
 * 시뮬레이터 전반에서 공유하는 기본 타입/상수를 제공하기 때문이다.
 * 이 헤더를 통해 warp_inst_t, kernel_info_t 등 핵심 자료구조도 간접적으로 가져온다. */

// extern time_t g_simulation_starttime;
/* [한국어] 이전에는 g_simulation_starttime을 전역 변수로 선언하려 했으나,
 * 현재는 GPGPUsim_ctx 클래스의 멤버 변수로 이동하여 필요 없게 되었다.
 * 전역 변수 대신 클래스 멤버로 두면 여러 시뮬레이터 인스턴스를 독립적으로
 * 관리할 수 있다는 장점이 있다. (현재는 단일 인스턴스이지만 구조상 더 깔끔하다.) */

class gpgpu_context;
/* [한국어] gpgpu_context 클래스의 전방 선언(forward declaration).
 * 이 헤더에서 gpgpu_context의 완전한 정의는 필요 없고 포인터만 사용한다.
 * 포인터 타입은 크기가 고정(8바이트, 64비트 아키텍처)이므로 완전한 정의 없이
 * 컴파일할 수 있다. 실제 정의는 libcuda/gpgpu_context.h에 있으며,
 * gpgpusim_entrypoint.cc에서 그 헤더를 include한다.
 * 이렇게 전방 선언을 사용하면 헤더 파일 간의 순환 의존성을 피할 수 있다. */

/*
 * [한국어]
 * GPGPUsim_ctx - GPGPU-Sim 시뮬레이터 제어 블록 (Simulator Control Block)
 *
 * 이 클래스는 GPGPU-Sim 시뮬레이터의 전체 생명주기 상태를 하나의 구조체로 묶는다.
 * gpgpu_context 싱글톤이 GPGPUsim_ctx* the_gpgpusim 포인터로 이 객체를 소유한다.
 * 모든 멤버가 public이므로 gpgpusim_entrypoint.cc와 libcuda 어디서든 직접 접근 가능하다.
 *
 * 설계 의도: CUDA 런타임 인터셉트 레이어(libcuda)와 내부 시뮬레이션 엔진(gpu-sim, stream_manager)
 * 사이의 중간 제어 객체로, 동기화 메커니즘과 핵심 서브시스템 포인터를 한곳에 집약한다.
 */
class GPGPUsim_ctx {
 public:
  /*
   * [한국어]
   * GPGPUsim_ctx 생성자 - 시뮬레이터 제어 블록 초기화
   *
   * @ctx: GPGPU-Sim 전체 컨텍스트 싱글톤 포인터. gpgpu_context::the_gpgpusim을
   *       설정하는 시점에 (void*)this로 전달된다.
   *
   * 모든 상태 플래그를 "아직 시뮬레이션 없음" 초기 상태로 설정하고,
   * 모든 서브시스템 포인터를 NULL로 초기화한다.
   * gpgpu_ptx_sim_init_perf()가 호출되기 전까지 NULL 포인터는 역참조해서는 안 된다.
   * 실행 컨텍스트: 호스트 쓰레드, gpgpu_context 생성 시점에 단 한 번 호출된다.
   *
   * 호출 체인:
   *   libcuda 초기화 → gpgpu_context 생성 → [GPGPUsim_ctx(ctx)]
   */
  GPGPUsim_ctx(gpgpu_context *ctx) {
    g_sim_active = false;
    /* [한국어] 현재 사이클이 진행 중이지 않음 (초기 상태).
     * 설정자: gpgpu_sim_thread_concurrent()가 내부 루프 전/후로 true/false 전환.
     * 읽는 자: synchronize(), synchronize_check()가 GPU idle 여부 판단에 사용.
     * 값 범위: true(사이클 진행 중) / false(idle 또는 미시작).
     * 동기화: g_sim_lock 뮤텍스로 보호 — 호스트/시뮬레이션 쓰레드가 동시 접근 가능. */

    g_sim_done = true;
    /* [한국어] 시뮬레이션이 "완료/미시작" 상태임을 나타냄 (초기값 true).
     * 초기값이 true인 이유: start_sim_thread()는 g_sim_done == true일 때만 새 쓰레드를
     * 생성하므로, 첫 번째 커널 launch 전 상태에서 쓰레드 생성이 가능해야 한다.
     * 설정자: start_sim_thread()가 false로, exit_simulation()/한도 초과 시 true로 설정.
     * 읽는 자: 시뮬레이션 쓰레드의 외부 루프 종료 조건으로 사용.
     * 값 범위: true(종료/미시작) / false(진행 중).
     * 동기화: bool 단일 워드 읽기는 아키텍처적으로 원자적이나, g_sim_active와 함께
     *         검사할 때는 g_sim_lock을 잡는다 (synchronize() 참조). */

    break_limit = false;
    /* [한국어] 최대 사이클/명령/CTA 한도 미도달 (초기 상태).
     * 설정자: gpgpu_sim_thread_concurrent() 또는 SST_Cycle()에서
     *         cycle_insn_cta_max_hit() == true일 때 true로 설정.
     * 읽는 자: 시뮬레이션 쓰레드 종료 직전에 true이면 exit(1)로 프로세스 강제 종료.
     * 값 범위: false(정상) / true(한도 도달 → 강제 종료 예정).
     * 동기화: 단일 시뮬레이션 쓰레드에서만 설정되므로 별도 잠금 불필요. */

    g_sim_lock = PTHREAD_MUTEX_INITIALIZER;
    /* [한국어] g_sim_active 읽기/쓰기를 보호하는 뮤텍스 정적 초기화.
     * PTHREAD_MUTEX_INITIALIZER는 pthread_mutex_init() 없이 선언 시 초기화하는 매크로이다.
     * 설정자: 이 생성자에서 한 번만 초기화되고 이후 pthread_mutex_lock/unlock으로 사용.
     * 읽는 자/쓰는 자: synchronize(), synchronize_check(), gpgpu_sim_thread_concurrent()
     * 값 범위: 잠금(locked) / 해제(unlocked) 내부 상태 — 직접 읽지 않음.
     * 동기화: 뮤텍스 자체가 동기화 수단이므로 별도 보호 불필요. */

    g_the_gpu_config = NULL;
    /* [한국어] GPU 마이크로아키텍처 설정 객체 미생성 상태.
     * 설정자: gpgpu_ptx_sim_init_perf()에서 new gpgpu_sim_config(ctx)로 생성 후 저장.
     * 읽는 자: start_sim_thread()가 is_SST_mode() 확인에 사용.
     *          gpgpu_sim 생성자에서 설정값 전달에 사용.
     * 값 범위: NULL(미초기화) / 유효한 gpgpu_sim_config 객체 포인터.
     * 동기화: 초기화 이후에는 읽기 전용이므로 별도 잠금 불필요. */

    g_the_gpu = NULL;
    /* [한국어] GPU 타이밍 시뮬레이터 핵심 객체 미생성 상태.
     * 설정자: gpgpu_ptx_sim_init_perf()에서 new exec_gpgpu_sim / sst_gpgpu_sim으로 생성.
     * 읽는 자: gpgpu_sim_thread_concurrent/sequential()이 매 사이클마다 cycle() 호출.
     *          synchronize(), exit_simulation() 등 모든 시뮬레이션 제어 경로.
     * 값 범위: NULL(미초기화) / exec_gpgpu_sim* 또는 sst_gpgpu_sim* (gpgpu_sim*로 다형성 사용).
     * 동기화: 포인터 자체는 불변. 내부 상태는 시뮬레이션 쓰레드만 변경하며
     *         launch()는 원자적으로 처리된다. */

    g_stream_manager = NULL;
    /* [한국어] CUDA 스트림 큐 관리자 미생성 상태.
     * 설정자: gpgpu_ptx_sim_init_perf()에서 new stream_manager(g_the_gpu, ...)로 생성.
     * 읽는 자:
     *   - 호스트 쓰레드(libcuda): push()로 커널/memcpy 작업을 큐에 등록.
     *   - 시뮬레이션 쓰레드: operation()으로 큐 front에서 작업을 꺼내어 실행.
     *   - synchronize(): empty()로 큐가 비었는지 확인.
     * 값 범위: NULL(미초기화) / 유효한 stream_manager 포인터.
     * 동기화: stream_manager 내부의 m_lock 뮤텍스가 큐 접근을 보호한다. */

    the_cude_device = NULL;
    /* [한국어] 시뮬레이션 대상 CUDA 디바이스 정보 미초기화 상태.
     * 설정자: libcuda에서 첫 cuDeviceGet() 또는 cudaGetDevice() 호출 시 초기화.
     * 읽는 자: libcuda가 장치 속성(디바이스 ID, 컴퓨트 캐파빌리티 등) 조회 시 사용.
     * 값 범위: NULL(미초기화) / _cuda_device_id 구조체 포인터.
     * 동기화: 초기화 이후 읽기 전용 — 별도 잠금 불필요.
     * 참고: "cude"는 "cuda"의 오타이나 원본 코드를 유지한다. */

    the_context = NULL;
    /* [한국어] CUDA 컨텍스트(CUctx_st) 미초기화 상태.
     * CUDA 컨텍스트는 GPU 메모리 할당 맵, 로드된 모듈(PTX 바이너리), 스트림 등을 담는다.
     * 설정자: libcuda에서 cuCtxCreate() 호출 시 초기화.
     * 읽는 자: libcuda의 메모리 할당, 커널 로드, 스트림 생성 등 모든 CUDA API 경로.
     * 값 범위: NULL(미초기화) / CUctx_st 구조체 포인터.
     * 동기화: 단일 CUDA 컨텍스트는 단일 호스트 쓰레드에서 사용 가정 — 별도 잠금 불필요. */

    gpgpu_ctx = ctx;
    /* [한국어] GPGPU-Sim 전역 컨텍스트 포인터 저장.
     * 생성자 매개변수로 받은 gpgpu_context*를 멤버에 보관한다.
     * 설정자: 이 생성자에서 단 한 번 설정 후 불변(immutable).
     * 읽는 자: gpgpusim_entrypoint.cc의 gpgpu_ptx_sim_init_perf() 등에서
     *          func_sim, ptx_parser 등 gpgpu_context의 다른 서브시스템에 접근하는 데 사용.
     * 값 범위: 항상 유효한 gpgpu_context 포인터 (NULL 불가 — 생성자 호출 시점에 존재).
     * 동기화: 불변값이므로 별도 잠금 불필요. */
  }

  // struct gpgpu_ptx_sim_arg *grid_params;
  /* [한국어] PTX 커널 그리드 매개변수 포인터 (사용 안 함, 비활성화된 코드).
   * 원래 커널 launch 인수를 여기서 관리하려 했으나 현재는 kernel_info_t를 통해
   * gpgpu_sim::launch()에 직접 전달하는 방식으로 변경되어 불필요해졌다. */

  sem_t g_sim_signal_start;
  /* [한국어] 호스트 쓰레드 → 시뮬레이션 쓰레드 방향의 "커널 시작" 신호 세마포어.
   * 설정자: gpgpu_ptx_sim_init_perf()에서 sem_init(..., 0, 0)으로 초기화(값 0).
   *         gpgpu_opencl_ptx_sim_main_perf()에서 sem_post()로 신호 전송.
   * 읽는 자: gpgpu_sim_thread_sequential()의 루프 상단에서 sem_wait()으로 블록.
   * 값 범위: 0 (대기 중) / 1 이상 (신호 수신됨).
   * 동기화: POSIX 세마포어는 스레드 안전(thread-safe) — 별도 잠금 불필요.
   * 참고: concurrent 모드에서는 이 세마포어를 사용하지 않고 스트림 큐로 대신한다. */

  sem_t g_sim_signal_finish;
  /* [한국어] 시뮬레이션 쓰레드 → 호스트 쓰레드 방향의 "커널 완료" 신호 세마포어.
   * 설정자: gpgpu_ptx_sim_init_perf()에서 sem_init(..., 0, 0)으로 초기화.
   *         gpgpu_sim_thread_sequential()이 커널 완료 후 sem_post()로 통보.
   * 읽는 자: gpgpu_opencl_ptx_sim_main_perf()에서 sem_wait()으로 커널 완료를 기다림.
   * 값 범위: 0 (진행 중) / 1 이상 (완료 신호 수신됨).
   * 동기화: POSIX 세마포어 내부적으로 원자적 처리.
   * 참고: concurrent 모드에서는 사용하지 않음 (스트림 큐로 대체됨). */

  sem_t g_sim_signal_exit;
  /* [한국어] 시뮬레이션 쓰레드 → 호스트 쓰레드 방향의 "쓰레드 종료 완료" 신호 세마포어.
   * 설정자: gpgpu_ptx_sim_init_perf()에서 sem_init(..., 0, 0)으로 초기화.
   *         gpgpu_sim_thread_sequential()과 gpgpu_sim_thread_concurrent()가
   *         루프 탈출 후 sem_post()로 종료를 통보한다.
   * 읽는 자: exit_simulation()에서 sem_wait()으로 쓰레드 종료를 기다림.
   * 값 범위: 0 (쓰레드 실행 중) / 1 (종료됨).
   * 동기화: POSIX 세마포어. break_limit이면 exit(1)로 프로세스 자체가 종료되므로
   *         이 세마포어가 post되지 않을 수도 있다. */

  time_t g_simulation_starttime;
  /* [한국어] 시뮬레이션 시작 시점의 Unix timestamp (wall-clock 기준).
   * 설정자: gpgpu_ptx_sim_init_perf()에서 time(NULL)로 기록.
   * 읽는 자: print_simulation_time()에서 현재 시간과의 차이로 경과 시간 계산.
   * 값 범위: 양수 Unix timestamp (1970-01-01 00:00:00 UTC 이후 경과 초 수).
   * 동기화: 단 한 번 쓰고 이후 읽기 전용 — 별도 잠금 불필요. */

  pthread_t g_simulation_thread;
  /* [한국어] 시뮬레이션 전용 쓰레드의 POSIX 쓰레드 식별자(handle).
   * 설정자: start_sim_thread()에서 pthread_create() 호출 시 OS가 할당하는 값이 저장됨.
   * 읽는 자: 현재 코드에서는 pthread_join() 등으로 직접 사용하지 않음.
   *          (시뮬레이션 쓰레드 종료는 세마포어로 감지함)
   * 값 범위: OS가 할당한 불투명(opaque) 쓰레드 ID.
   * 동기화: start_sim_thread()만 이 값을 쓰므로 별도 잠금 불필요.
   * 참고: SST 모드에서는 별도 쓰레드를 생성하지 않으므로 이 값이 사용되지 않는다. */

  class gpgpu_sim_config *g_the_gpu_config;
  /* [한국어] GPU 마이크로아키텍처 설정 객체 포인터 (gpgpu_sim_config 전방 선언).
   * gpgpusim.config 파일에서 파싱된 모든 하드웨어 파라미터를 보관한다:
   *   예) -gpgpu_n_clusters 20, -gpgpu_n_cores_per_cluster 1,
   *       -gpgpu_cache:dl1 16:128:4,L:L:m:N:H,A:32:8,8,0,
   *       -gpgpu_dram_timing_opt 등.
   * 설정자: gpgpu_ptx_sim_init_perf()에서 new gpgpu_sim_config(ctx)로 생성.
   * 읽는 자: GPU 시뮬레이터 생성 시 참조. start_sim_thread()에서 is_SST_mode() 확인.
   * 값 범위: NULL(미초기화) / 유효한 gpgpu_sim_config 포인터.
   * 동기화: 생성 후 읽기 전용 — 별도 잠금 불필요. */

  class gpgpu_sim *g_the_gpu;
  /* [한국어] GPU 타이밍 시뮬레이터 최상위 객체 포인터 (gpgpu_sim 전방 선언).
   * 실제 타입은 exec_gpgpu_sim(일반 모드) 또는 sst_gpgpu_sim(SST 모드)이며,
   * 다형성(polymorphism)으로 gpgpu_sim* 포인터를 통해 접근한다.
   * 이 객체 안에 SM(shader core), 캐시 계층, DRAM 컨트롤러, NoC 인터페이스가 있다.
   * 설정자: gpgpu_ptx_sim_init_perf()에서 new exec_gpgpu_sim / sst_gpgpu_sim으로 생성.
   * 읽는 자: 시뮬레이션 쓰레드의 cycle(), active(), deadlock_check() 등 매 사이클 호출.
   *          libcuda의 launch() (커널 등록), synchronize() (idle 대기) 등.
   * 값 범위: NULL(미초기화) / exec_gpgpu_sim* 또는 sst_gpgpu_sim* (gpgpu_sim*로 다형성 사용).
   * 동기화: 포인터 자체는 불변. 내부 상태(SM 파이프라인, 캐시 등)는 시뮬레이션 쓰레드만 변경.
   *         stream_manager::operation()이 launch()를 호출하는 것은 g_sim_lock 없이 이루어지며
   *         launch()가 원자적으로 처리한다. */

  class stream_manager *g_stream_manager;
  /* [한국어] CUDA 스트림 큐 관리자 포인터 (stream_manager 전방 선언).
   * 모든 활성 CUstream_st 객체와 그 안의 보류 중인 stream_operation을 관리한다.
   * 설정자: gpgpu_ptx_sim_init_perf()에서 new stream_manager(g_the_gpu, ...)로 생성.
   * 읽는 자:
   *   - 호스트 쓰레드(libcuda): push()로 커널/memcpy 작업을 큐에 등록.
   *   - 시뮬레이션 쓰레드: operation()으로 큐 front에서 작업을 꺼내어 실행.
   *   - synchronize(): empty()로 큐가 비었는지 확인.
   * 값 범위: NULL(미초기화) / 유효한 stream_manager 포인터.
   * 동기화: stream_manager 내부의 m_lock 뮤텍스가 큐 접근을 보호한다. */

  struct _cuda_device_id *the_cude_device;
  /* [한국어] 시뮬레이션 대상 CUDA 디바이스 정보 구조체 포인터.
   * _cuda_device_id는 장치 ID, 컴퓨트 캐파빌리티 버전 등 장치 식별 정보를 담는다.
   * 설정자: libcuda에서 첫 GPU 장치 열거(cuDeviceGet) 또는 선택 시 초기화.
   * 읽는 자: libcuda의 cudaGetDeviceProperties(), cuDeviceGetAttribute() 등 장치 속성 쿼리.
   * 값 범위: NULL(미초기화) / 유효한 _cuda_device_id 포인터.
   * 동기화: 초기화 후 읽기 전용 — 별도 잠금 불필요.
   * 참고: 필드명 "cude"는 "cuda"의 오타이나 원본 코드를 유지한다. */

  struct CUctx_st *the_context;
  /* [한국어] CUDA 실행 컨텍스트(CUctx_st, CUcontext) 포인터.
   * CUctx_st는 단일 GPU 사용 세션의 상태를 나타내며, CUDA 드라이버 API의 CUcontext에 해당한다.
   * 내용: GPU 가상 메모리 할당 맵, 로드된 cubin/PTX 모듈, 이벤트, 스트림 목록 등.
   * 설정자: libcuda에서 cuCtxCreate() 또는 암묵적 컨텍스트 생성 시 초기화.
   * 읽는 자: libcuda 전반의 메모리 관리, 커널 로드, 스트림 작업 등 모든 API 경로.
   * 값 범위: NULL(미초기화) / 유효한 CUctx_st 포인터.
   * 동기화: 단일 호스트 쓰레드에서 사용 가정(CUDA 컨텍스트는 기본적으로 단일 쓰레드 소유). */

  gpgpu_context *gpgpu_ctx;
  /* [한국어] GPGPU-Sim 전역 컨텍스트 싱글톤 역참조 포인터.
   * GPGPUsim_ctx는 gpgpu_context의 the_gpgpusim 포인터로 소유되는데,
   * 반대 방향으로도 gpgpu_context에 접근해야 하므로 이 역참조 포인터를 저장한다.
   * 설정자: 생성자에서 단 한 번 설정 후 불변.
   * 읽는 자: gpgpu_ptx_sim_init_perf()에서 ctx->func_sim, ctx->ptx_parser 등 접근.
   *          gpgpu_sim_thread_concurrent()에서 gpgpu_ctx->func_sim->gpgpu_cuda_ptx_sim_main_func() 호출.
   * 값 범위: 항상 유효한 gpgpu_context 포인터 (NULL 불가).
   * 동기화: 불변값 — 별도 잠금 불필요. */

  pthread_mutex_t g_sim_lock;
  /* [한국어] g_sim_active 필드의 읽기/쓰기를 보호하는 뮤텍스.
   * 호스트 쓰레드(synchronize, synchronize_check)와 시뮬레이션 쓰레드
   * (gpgpu_sim_thread_concurrent)가 g_sim_active를 동시에 읽고 쓸 수 있으므로
   * 데이터 레이스(data race)를 방지하기 위해 이 뮤텍스로 보호한다.
   * 설정자: 생성자에서 PTHREAD_MUTEX_INITIALIZER로 정적 초기화.
   * 읽는 자/쓰는 자: synchronize(), synchronize_check(), gpgpu_sim_thread_concurrent().
   * 값 범위: LOCKED / UNLOCKED 내부 상태 (pthread 내부 관리).
   * 동기화: 뮤텍스 자체가 동기화 수단 — 중첩 잠금(deadlock) 주의:
   *         이 뮤텍스를 잡은 상태에서 stream_manager의 m_lock을 잡으면 안 된다. */

  bool g_sim_active;
  /* [한국어] 현재 시뮬레이션 쓰레드가 사이클을 진행 중인지 나타내는 플래그.
   * 설정자:
   *   - true : gpgpu_sim_thread_concurrent()가 내부 루프 진입 전 (g_sim_lock 보호)
   *   - false: gpgpu_sim_thread_concurrent()가 내부 루프 탈출 후 (g_sim_lock 보호)
   *            SST_Cycle()에서도 일부 경로에서 직접 설정.
   * 읽는 자: synchronize(), synchronize_check()가 GPU idle 조건 판단에 사용.
   * 값 범위: true(사이클 진행 중) / false(idle 또는 대기 중).
   * 동기화: 반드시 g_sim_lock 뮤텍스 잠금 하에 읽고 써야 한다.
   * g_sim_done과의 관계: g_sim_done == true이면 g_sim_active는 의미 없다. */

  bool g_sim_done;
  /* [한국어] 시뮬레이션 전체 종료 여부를 나타내는 플래그.
   * 설정자:
   *   - true(초기값): 생성자에서 "아직 시작 안 함" 상태로.
   *   - false        : start_sim_thread()에서 새 시뮬레이션 시작 시.
   *   - true(종료)   : exit_simulation()에서 외부 종료 요청 시,
   *                    또는 최대 한도 도달(cycle_insn_cta_max_hit) 시.
   * 읽는 자: 시뮬레이션 쓰레드의 외부 do-while 루프 탈출 조건.
   *          start_sim_thread()에서 새 쓰레드 생성 가능 여부 판단.
   * 값 범위: true(종료/미시작) / false(진행 중).
   * 동기화: bool 단일 워드 읽기는 원자적이나, g_sim_active와 함께 검사할 때는
   *         g_sim_lock을 잡는다 (synchronize() 참조). */

  bool break_limit;
  /* [한국어] 최대 사이클/명령/CTA 한도 도달로 인한 강제 종료 플래그.
   * gpgpusim.config의 -gpgpu_max_cycle, -gpgpu_max_insn, -gpgpu_max_cta 옵션 중
   * 하나라도 초과하면 true로 설정된다.
   * 설정자: gpgpu_sim_thread_concurrent() 또는 SST_Cycle()에서 cycle_insn_cta_max_hit() 시.
   * 읽는 자: 시뮬레이션 쓰레드 루프 탈출 후 true이면 exit(1)로 프로세스 강제 종료.
   *          SST_Cycle()은 true이면 return true하여 SST에 종료를 통보.
   * 값 범위: false(정상 실행 중) / true(한도 도달, 강제 종료 예정).
   * 동기화: 단일 시뮬레이션 쓰레드에서만 설정 — 별도 잠금 불필요.
   * 주의: break_limit = true이면 g_sim_done = true도 함께 설정된다. */
};

#endif  // GPGPUSIM_ENTRYPOINT_H_INCLUDED
/* [한국어] 인클루드 가드 종료. 파일 최상단의 #ifndef GPGPUSIM_ENTRYPOINT_H_INCLUDED와 짝을 이룬다. */
