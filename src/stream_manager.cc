/*
 * [한국어 설명] CUDA 스트림 관리자 구현 (stream_manager.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 stream_manager.h에서 선언된 CUstream_st, stream_operation,
 * stream_manager 클래스의 모든 메서드를 구현한다.
 * CUDA 스트림의 큐잉, 스케줄링, 실행, 완료 처리의 전체 생명 주기를 담당한다.
 * 호스트 쓰레드와 시뮬레이션 쓰레드 사이의 작업 전달을 뮤텍스로 안전하게 조율하며,
 * SST 모드에서는 memcpy 완료 후 SST 프레임워크에 콜백으로 알린다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CUDA 비동기 API 호출 흐름:
 *   호스트 쓰레드 (libcuda) → stream_manager::push(op)
 *       → CUstream_st::push() — m_operations 큐에 작업 추가
 *   시뮬레이션 쓰레드 (gpgpu_sim_thread_concurrent) → stream_manager::operation(&sim)
 *       → stream_manager::front() — 라운드로빈으로 다음 스트림 작업 선택
 *       → stream_operation::do_operation(gpu) — 실제 memcpy/kernel launch 수행
 *       → check_finished_kernel() → register_finished_kernel()
 *           → CUstream_st::record_next_done() — 완료 작업 큐에서 제거
 * 실행 컨텍스트: 두 쓰레드가 동시에 접근하므로 CUstream_st::m_lock과
 *               stream_manager::m_lock이 항상 올바른 순서로 잡혀야 한다.
 *
 * === 타 모듈과의 연결 ===
 * 포함하는 헤더:
 *   - stream_manager.h              : 이 파일이 구현하는 모든 클래스 선언
 *   - ../libcuda/gpgpu_context.h    : gpgpu_context (func_sim 포인터 제공)
 *   - cuda-sim/cuda-sim.h           : gpgpu_ptx_sim_memcpy_symbol 등 PTX 기능 시뮬레이션 API
 *   - gpgpu-sim/gpu-sim.h           : gpgpu_sim 클래스 (memcpy_to_gpu, launch, finished_kernel 등)
 *   - gpgpusim_entrypoint.h         : g_debug_execution 전역 변수
 * 데이터 흐름:
 *   libcuda → [stream_operation 생성] → CUstream_st::m_operations 큐
 *   → stream_operation::do_operation() → gpgpu_sim (타이밍 모델) / cuda_sim (기능 모델)
 *   → 커널 완료 시 gpu->finished_kernel() → register_finished_kernel() → 큐 정리
 *
 * === 주요 함수/구조체 요약 ===
 * - CUstream_st::push/next/record_next_done: 큐 FIFO 연산 (뮤텍스 보호)
 * - stream_operation::do_operation()        : 작업 타입 switch — 핵심 실행 분기
 * - stream_manager::operation()             : 시뮬레이션 쓰레드에서 매 사이클 호출되는 주 진입점
 * - stream_manager::front()                 : 라운드로빈 스케줄러 — 다음 실행 작업 선택
 * - stream_manager::push()                  : blocking/non-blocking 모드 분기 + spin-wait
 * - stream_manager::register_finished_kernel(): 커널 완료 후 스트림 큐 정리 및 kernel 해제
 */

// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
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

#include "stream_manager.h"           /* [한국어] 이 파일이 구현하는 모든 클래스 선언 */
#include "../libcuda/gpgpu_context.h" /* [한국어] gpgpu_context 전역 싱글턴 — func_sim 포인터 접근용 */
#include "cuda-sim/cuda-sim.h"        /* [한국어] PTX 기능 시뮬레이션 API (gpgpu_ptx_sim_memcpy_symbol 등) */
#include "gpgpu-sim/gpu-sim.h"        /* [한국어] gpgpu_sim 클래스 — memcpy_to_gpu, launch, finished_kernel 등 타이밍 시뮬레이터 API */
#include "gpgpusim_entrypoint.h"      /* [한국어] g_debug_execution 전역 변수 (디버그 출력 레벨 제어) */

unsigned CUstream_st::sm_next_stream_uid = 0;
/* [한국어] CUstream_st 정적 멤버 변수 초기화. 전역 스트림 UID 카운터를 0으로 설정.
 * 첫 번째 생성되는 스트림은 m_uid = 0이 된다. */

// SST memcpy callbacks
/* [한국어] SST(Sandia Structural Simulation Toolkit) 연동 시 memcpy 완료를 SST 프레임워크에
 * 알리기 위한 콜백 함수 선언. SST와 함께 빌드할 때는 SST 측에서 이 심볼들을 강력한(strong)
 * 심볼로 제공하고, GPGPU-Sim 단독 실행 시에는 아래 __attribute__((weak)) 빈 구현이 사용된다. */
extern void SST_callback_memcpy_H2D_done();        /* [한국어] H2D 복사 완료 SST 콜백 선언 */
extern void SST_callback_memcpy_D2H_done();        /* [한국어] D2H 복사 완료 SST 콜백 선언 */
extern void SST_callback_memcpy_to_symbol_done();  /* [한국어] to_symbol 복사 완료 SST 콜백 선언 */
extern void SST_callback_memcpy_from_symbol_done();/* [한국어] from_symbol 복사 완료 SST 콜백 선언 */

__attribute__((weak)) void SST_callback_memcpy_H2D_done() {}
/* [한국어] SST 없이 빌드 시 사용되는 빈 약한 심볼 구현.
 * SST가 링크되면 SST 측의 강한(strong) 심볼로 교체되어 실제 콜백이 발동된다. */
__attribute__((weak)) void SST_callback_memcpy_D2H_done() {}
/* [한국어] D2H 완료 콜백 — SST 없을 때 빈 no-op. */
__attribute__((weak)) void SST_callback_memcpy_to_symbol_done() {}
/* [한국어] to_symbol 완료 콜백 — SST 없을 때 빈 no-op. */
__attribute__((weak)) void SST_callback_memcpy_from_symbol_done() {}
/* [한국어] from_symbol 완료 콜백 — SST 없을 때 빈 no-op. */

/*
 * [한국어]
 * CUstream_st::CUstream_st - CUDA 스트림 생성자
 *
 * @param (없음)
 * @return (생성자)
 *
 * m_pending = false로 초기화하고, 전역 카운터 sm_next_stream_uid에서 고유 ID를 부여한다.
 * pthread_mutex_t를 기본 속성(NULL)으로 초기화한다.
 * 큐(m_operations)는 std::list의 기본 생성자로 자동 초기화된다.
 *
 * 실행 컨텍스트: 호스트 쓰레드 (cudaStreamCreate() 시 new CUstream_st() 호출).
 *
 * 호출 체인:
 *   libcuda::cudaStreamCreate() → new CUstream_st() → [이 생성자]
 */
CUstream_st::CUstream_st() {
  m_pending = false;                    /* [한국어] 처리 중인 front 작업 없음으로 초기화 */
  m_uid = sm_next_stream_uid++;         /* [한국어] 후위증가로 현재 카운터 값을 UID로 저장하고 카운터 증가 */
  pthread_mutex_init(&m_lock, NULL);    /* [한국어] 기본 속성의 뮤텍스 초기화. NULL = 기본 뮤텍스 속성 (재진입 불가) */
}

/*
 * [한국어]
 * CUstream_st::empty - 이 스트림의 작업 큐가 비어있는지 확인 (뮤텍스 보호)
 *
 * @return: m_operations가 비어있으면 true, 아니면 false.
 *
 * m_lock을 잡고 확인하므로 호스트/시뮬레이션 쓰레드 모두에서 안전하게 호출 가능.
 * destroy_stream()의 spin-wait과 synchronize()의 완료 조건 확인에 사용된다.
 *
 * 호출 체인:
 *   CUstream_st::synchronize(), CUstream_st::busy(),
 *   stream_manager::concurrent_streams_empty() → [empty]
 */
bool CUstream_st::empty() {
  pthread_mutex_lock(&m_lock);              /* [한국어] m_operations 접근 전 락 획득 */
  bool empty = m_operations.empty();        /* [한국어] 작업 큐가 비어있는지 확인 */
  pthread_mutex_unlock(&m_lock);            /* [한국어] 확인 즉시 락 해제 (최소 임계 구간) */
  return empty;                             /* [한국어] 비어있으면 true 반환 */
}

/*
 * [한국어]
 * CUstream_st::busy - front 작업이 현재 처리 중인지 확인 (뮤텍스 보호)
 *
 * @return: m_pending이 true(처리 중)이면 true, false(유휴)이면 false.
 *
 * m_pending은 next()가 호출된 후 record_next_done() 또는 cancel_front()가
 * 호출되기 전까지 true이다.
 * stream_manager::front()에서 이 스트림에서 새 작업을 꺼낼 수 있는지 결정한다.
 *
 * 호출 체인:
 *   stream_manager::front() → CUstream_st::busy()
 */
bool CUstream_st::busy() {
  pthread_mutex_lock(&m_lock);              /* [한국어] m_pending 읽기 전 락 획득 */
  bool pending = m_pending;                 /* [한국어] 처리 중 여부 읽기 */
  pthread_mutex_unlock(&m_lock);            /* [한국어] 락 즉시 해제 */
  return pending;                           /* [한국어] 처리 중이면 true 반환 */
}

/*
 * [한국어]
 * CUstream_st::synchronize - 이 스트림의 모든 작업 완료를 바쁜 대기(spin-wait)로 기다림
 *
 * @param (없음)
 * @return (없음)
 *
 * m_operations.empty()가 true가 될 때까지 m_lock을 반복적으로 잡고 확인한다.
 * CPU를 지속 점유하는 바쁜 대기(spin-wait)이므로 CPU 코어 1개를 소진한다.
 * cudaStreamSynchronize(stream)을 구현하기 위해 호스트 쓰레드에서 사용된다.
 *
 * 실행 컨텍스트: 호스트 쓰레드.
 *
 * 호출 체인:
 *   libcuda::cudaStreamSynchronize() → [CUstream_st::synchronize]
 */
void CUstream_st::synchronize() {
  // called by host thread
  bool done = false;                    /* [한국어] 완료 여부 로컬 변수 초기화 */
  do {
    pthread_mutex_lock(&m_lock);        /* [한국어] 큐 접근 전 락 획득 */
    done = m_operations.empty();        /* [한국어] 큐가 비면 완료 */
    pthread_mutex_unlock(&m_lock);      /* [한국어] 락 즉시 해제하여 시뮬레이션 쓰레드가 작업 처리 가능하게 함 */
  } while (!done);                      /* [한국어] 완료될 때까지 반복 (spin-wait) */
}

/*
 * [한국어]
 * CUstream_st::push - 스트림 큐 끝에 새 작업 추가 (호스트 쓰레드 전용)
 *
 * @op: 추가할 stream_operation 객체 (값으로 복사됨).
 * @return (없음)
 *
 * m_lock을 잡고 m_operations 리스트 끝에 op를 추가한다.
 * 이후 시뮬레이션 쓰레드가 next()로 꺼내어 처리한다.
 *
 * 실행 컨텍스트: 호스트 쓰레드 (libcuda 비동기 API 처리 중).
 *
 * 호출 체인:
 *   stream_manager::push() → CUstream_st::push()
 */
void CUstream_st::push(const stream_operation &op) {
  // called by host thread
  pthread_mutex_lock(&m_lock);          /* [한국어] 큐 접근 전 락 획득 */
  m_operations.push_back(op);           /* [한국어] 큐 끝에 작업 추가 — FIFO 순서 보장 */
  pthread_mutex_unlock(&m_lock);        /* [한국어] 락 해제 */
}

/*
 * [한국어]
 * CUstream_st::record_next_done - front 작업 완료 처리 (GPU 쓰레드 전용)
 *
 * @param (없음)
 * @return (없음)
 *
 * m_lock을 잡고 m_operations.pop_front()로 완료된 작업을 제거하고
 * m_pending = false로 설정하여 다음 작업을 꺼낼 수 있게 한다.
 * assert(m_pending)으로 next()가 먼저 호출되었는지 확인한다
 * (pending 상태가 아닌데 done을 기록하는 것은 논리 오류).
 *
 * 실행 컨텍스트: 시뮬레이션 쓰레드 (stream_operation::do_operation() 성공 후).
 *
 * 호출 체인:
 *   stream_operation::do_operation() [성공 시] → stream->record_next_done()
 *   register_finished_kernel() → stream->record_next_done()
 */
void CUstream_st::record_next_done() {
  // called by gpu thread
  pthread_mutex_lock(&m_lock);          /* [한국어] 큐 수정 전 락 획득 */
  assert(m_pending);                    /* [한국어] next()가 먼저 호출되어 pending 상태여야 함 — 아니면 버그 */
  m_operations.pop_front();             /* [한국어] 완료된 front 작업 큐에서 제거 */
  m_pending = false;                    /* [한국어] 처리 중 플래그 해제 — busy()가 false 반환하게 됨 */
  pthread_mutex_unlock(&m_lock);        /* [한국어] 락 해제 */
}

/*
 * [한국어]
 * CUstream_st::next - front 작업을 "처리 중"으로 마킹하고 반환 (GPU 쓰레드 전용)
 *
 * @return: m_operations.front()의 값 복사본.
 *
 * m_pending = true로 설정하여 이 작업이 진행 중임을 표시한다.
 * 이후 do_operation() 성공 시 record_next_done()이 호출될 때까지 busy()는 true이다.
 * 큐가 비어있는 상태에서 호출하면 UB(m_operations.front() on empty list) 발생 가능 —
 * stream_manager::front()가 항상 !stream->empty() && !stream->busy() 조건 확인 후 호출.
 *
 * 실행 컨텍스트: 시뮬레이션 쓰레드.
 *
 * 호출 체인:
 *   stream_manager::front() → CUstream_st::next()
 */
stream_operation CUstream_st::next() {
  // called by gpu thread
  pthread_mutex_lock(&m_lock);                  /* [한국어] 큐 접근 전 락 획득 */
  m_pending = true;                             /* [한국어] 처리 중 플래그 설정 */
  stream_operation result = m_operations.front();  /* [한국어] front 작업 값 복사 */
  pthread_mutex_unlock(&m_lock);                /* [한국어] 락 해제 (result는 이미 복사됨) */
  return result;                                /* [한국어] 복사된 작업 반환 */
}

/*
 * [한국어]
 * CUstream_st::cancel_front - pending 중인 front 작업 취소 (GPU 쓰레드 전용)
 *
 * @param (없음)
 * @return (없음)
 *
 * do_operation()이 false를 반환하여 작업 실행을 미룰 때 사용한다.
 * m_pending = false로 되돌려 다음 사이클에 다시 front()로 같은 작업을 시도할 수 있게 한다.
 * assert(m_pending)으로 next()가 먼저 호출된 상태인지 확인한다.
 *
 * 실행 컨텍스트: 시뮬레이션 쓰레드 (stream_manager::operation()에서 do_operation 실패 시).
 *
 * 호출 체인:
 *   stream_manager::operation() [do_operation 실패 시] → op.get_stream()->cancel_front()
 */
void CUstream_st::cancel_front() {
  pthread_mutex_lock(&m_lock);          /* [한국어] m_pending 수정 전 락 획득 */
  assert(m_pending);                    /* [한국어] next() 이후에만 호출 가능 — 아니면 논리 오류 */
  m_pending = false;                    /* [한국어] pending 플래그 해제 — 다음 사이클에 재시도 가능 */
  pthread_mutex_unlock(&m_lock);        /* [한국어] 락 해제 */
}

/*
 * [한국어]
 * CUstream_st::print - 이 스트림의 현재 상태를 파일 스트림에 출력
 *
 * @fp: 출력 대상 FILE 포인터 (보통 stdout).
 * @return (없음)
 *
 * m_lock을 잡고 스트림 UID와 작업 수를 출력한 후,
 * m_operations의 각 작업에 대해 stream_operation::print()를 호출한다.
 * 디버그 출력 및 stream_manager::print_impl()에서 사용된다.
 *
 * 호출 체인:
 *   stream_manager::print_impl() → CUstream_st::print(fp)
 */
void CUstream_st::print(FILE *fp) {
  pthread_mutex_lock(&m_lock);          /* [한국어] m_operations 순회 중 다른 쓰레드의 수정 방지 */
  fprintf(fp, "GPGPU-Sim API:    stream %u has %zu operations\n", m_uid,
          m_operations.size());         /* [한국어] 스트림 UID와 대기 중인 작업 수 출력 */
  std::list<stream_operation>::iterator i;  /* [한국어] 작업 목록 순회 반복자 */
  unsigned n = 0;                           /* [한국어] 작업 인덱스 카운터 */
  for (i = m_operations.begin(); i != m_operations.end(); i++) {
    /* [한국어] 모든 대기 작업을 순서대로 출력 */
    stream_operation &op = *i;              /* [한국어] 현재 작업에 대한 참조 */
    fprintf(fp, "GPGPU-Sim API:       %u : ", n++);  /* [한국어] 작업 인덱스 출력 */
    op.print(fp);                           /* [한국어] 작업 타입 출력 (stream_operation::print) */
    fprintf(fp, "\n");                      /* [한국어] 줄바꿈 */
  }
  pthread_mutex_unlock(&m_lock);        /* [한국어] 출력 완료 후 락 해제 */
}

/*
 * [한국어]
 * stream_operation::do_operation - 스트림 작업을 실제로 실행하는 핵심 함수
 *
 * @gpu: 타이밍/기능 시뮬레이터 객체 포인터. memcpy/launch 등을 수행한다.
 * @return: true이면 이 작업이 완료되어 m_done = true로 설정됨.
 *          false이면 아직 실행 조건 미충족 (커널 launch latency 대기, 이벤트 미완료).
 *
 * m_type에 따른 switch 분기:
 *   H2D: gpu->memcpy_to_gpu() → record_next_done() → SST 콜백
 *   D2H: gpu->memcpy_from_gpu() → record_next_done() → SST 콜백
 *   D2D: gpu->memcpy_gpu_to_gpu() → record_next_done()
 *   to_symbol: gpgpu_ptx_sim_memcpy_symbol(dir=1) → record_next_done() → SST 콜백
 *   from_symbol: gpgpu_ptx_sim_memcpy_symbol(dir=0) → record_next_done() → SST 콜백
 *   kernel_launch (기능 시뮬): gpu->functional_launch(m_kernel)
 *   kernel_launch (타이밍 시뮬): gpu->can_start_kernel() && m_launch_latency==0이면 gpu->launch()
 *                                 아니면 m_launch_latency-- 후 return false (재시도 예약)
 *   stream_event: m_event->update() → record_next_done()
 *   stream_wait_event: m_event->num_updates() >= m_cnt이면 record_next_done(), 아니면 return false
 *
 * 실행 컨텍스트: 시뮬레이션 쓰레드 (stream_manager::operation()에서 호출).
 *
 * 호출 체인:
 *   stream_manager::operation() → [do_operation]
 *     → gpu->memcpy_to_gpu() / gpu->launch() / gpu->functional_launch()
 *     → stream->record_next_done() [완료 시]
 */
bool stream_operation::do_operation(gpgpu_sim *gpu) {
  if (is_noop()) return true;           /* [한국어] 빈 작업은 즉시 완료 반환 */

  assert(!m_done && m_stream);          /* [한국어] 이미 완료된 작업을 재실행하거나 스트림 없는 작업을 실행하면 버그 */
  if (g_debug_execution >= 3)           /* [한국어] 디버그 레벨 3 이상일 때 작업 실행 정보 출력 */
    printf("GPGPU-Sim API: stream %u performing ", m_stream->get_uid());  /* [한국어] 스트림 UID 출력 */
  switch (m_type) {
    case stream_memcpy_host_to_device:
      /* [한국어] 호스트→디바이스 메모리 복사:
       * gpu->memcpy_to_gpu()가 시뮬레이터의 GPU 메모리 공간에 데이터를 복사한다.
       * 완료 후 record_next_done()으로 이 작업을 큐에서 제거.
       * SST 모드이면 SST 프레임워크에 완료 콜백을 보낸다. */
      if (g_debug_execution >= 3) printf("memcpy host-to-device\n");  /* [한국어] 디버그 출력 */
      gpu->memcpy_to_gpu(m_device_address_dst, m_host_address_src, m_cnt);
      /* [한국어] 시뮬레이터 GPU 메모리에 m_cnt 바이트를 복사.
       * m_device_address_dst: 대상 GPU 가상 주소, m_host_address_src: 소스 호스트 포인터. */
      m_stream->record_next_done();     /* [한국어] 작업 완료 — 큐에서 제거하고 m_pending 해제 */
      if (gpu->is_SST_mode()) SST_callback_memcpy_H2D_done();
      /* [한국어] SST 모드일 때 H2D 완료를 SST 외부 시뮬레이터에 통지 (weak 심볼 — SST 없으면 no-op) */
      break;
    case stream_memcpy_device_to_host:
      /* [한국어] 디바이스→호스트 메모리 복사:
       * gpu->memcpy_from_gpu()가 시뮬레이터 GPU 메모리에서 호스트 버퍼로 데이터를 복사한다. */
      if (g_debug_execution >= 3) printf("memcpy device-to-host\n");   /* [한국어] 디버그 출력 */
      gpu->memcpy_from_gpu(m_host_address_dst, m_device_address_src, m_cnt);
      /* [한국어] GPU 메모리에서 호스트 버퍼로 m_cnt 바이트 복사 */
      m_stream->record_next_done();     /* [한국어] 작업 완료 처리 */
      if (gpu->is_SST_mode()) SST_callback_memcpy_D2H_done();
      /* [한국어] SST 모드 D2H 완료 콜백 */
      break;
    case stream_memcpy_device_to_device:
      /* [한국어] 디바이스→디바이스 메모리 복사:
       * GPU 내부 메모리 복사로, 호스트를 거치지 않는다. */
      if (g_debug_execution >= 3) printf("memcpy device-to-device\n"); /* [한국어] 디버그 출력 */
      gpu->memcpy_gpu_to_gpu(m_device_address_dst, m_device_address_src, m_cnt);
      /* [한국어] GPU 메모리 내 m_cnt 바이트를 src에서 dst로 복사 */
      m_stream->record_next_done();     /* [한국어] 작업 완료 처리 */
      break;                            /* [한국어] D2D는 SST 콜백 없음 */
    case stream_memcpy_to_symbol:
      /* [한국어] 호스트→GPU 상수 메모리 심볼 복사:
       * gpgpu_ptx_sim_memcpy_symbol()의 dir=1 호출로 호스트 데이터를
       * GPU의 상수 메모리 심볼에 기록한다. */
      if (g_debug_execution >= 3) printf("memcpy to symbol\n");        /* [한국어] 디버그 출력 */
      gpu->gpgpu_ctx->func_sim->gpgpu_ptx_sim_memcpy_symbol(
          m_symbol, m_host_address_src, m_cnt, m_offset, 1, gpu);
      /* [한국어] 인수: symbol이름, 소스 호스트 주소, 바이트 수, 심볼 오프셋, dir=1(to), gpu포인터.
       * func_sim: 기능 시뮬레이션 객체 (PTX 상수 메모리 맵을 관리). */
      m_stream->record_next_done();     /* [한국어] 작업 완료 처리 */
      if (gpu->is_SST_mode()) SST_callback_memcpy_to_symbol_done();
      /* [한국어] SST 모드 to_symbol 완료 콜백 */
      break;
    case stream_memcpy_from_symbol:
      /* [한국어] GPU 상수 메모리 심볼→호스트 복사:
       * gpgpu_ptx_sim_memcpy_symbol()의 dir=0 호출로 GPU 심볼 데이터를 호스트 버퍼로 읽어온다. */
      if (g_debug_execution >= 3) printf("memcpy from symbol\n");      /* [한국어] 디버그 출력 */
      gpu->gpgpu_ctx->func_sim->gpgpu_ptx_sim_memcpy_symbol(
          m_symbol, m_host_address_dst, m_cnt, m_offset, 0, gpu);
      /* [한국어] dir=0 → from_symbol (GPU→호스트 방향) */
      m_stream->record_next_done();     /* [한국어] 작업 완료 처리 */
      if (gpu->is_SST_mode()) SST_callback_memcpy_from_symbol_done();
      /* [한국어] SST 모드 from_symbol 완료 콜백 */
      break;
    case stream_kernel_launch:
      /* [한국어] GPU 커널 실행:
       * m_sim_mode가 true이면 기능 시뮬레이션(PTX 의미론적 실행, 타이밍 무시),
       * false이면 타이밍 시뮬레이션(사이클-레벨 마이크로아키텍처 모델). */
      if (m_sim_mode) {  // Functional Sim
        /* [한국어] 기능 시뮬레이션 경로:
         * gpu->functional_launch()는 타이밍 없이 커널 로직만 즉시 실행한다.
         * 이 경우 record_next_done()은 여기서 호출하지 않는다 —
         * 기능 시뮬레이션은 즉시 완료되지 않고 별도로 처리될 수 있음. */
        if (g_debug_execution >= 3) {
          printf("kernel %d: \'%s\' transfer to GPU hardware scheduler\n",
                 m_kernel->get_uid(), m_kernel->name().c_str());
          /* [한국어] 커널 UID와 함수 이름 출력 */
          m_kernel->print_parent_info();
          /* [한국어] CDP(CUDA Dynamic Parallelism) 사용 시 부모 커널 정보 출력 */
        }
        gpu->set_cache_config(m_kernel->name());
        /* [한국어] 이 커널의 캐시 구성(L1 크기, 셰어드 메모리 크기 등)을 GPU에 설정.
         * gpgpusim.config에서 커널 이름별로 다른 캐시 설정을 지원. */
        gpu->functional_launch(m_kernel);
        /* [한국어] 기능 시뮬레이션 모드로 커널 시작 — 타이밍 없이 PTX 명령어 의미 실행 */
      } else {  // Performance Sim
        /* [한국어] 타이밍 시뮬레이션 경로:
         * 두 가지 조건이 모두 충족되어야 launch 가능:
         *   1) gpu->can_start_kernel(): GPU에 빈 SM 슬롯이 있어야 함
         *   2) m_kernel->m_launch_latency == 0: 인위적 launch 지연이 0이어야 함
         * 조건 미충족 시 return false로 다음 사이클에 재시도. */
        if (gpu->can_start_kernel() && m_kernel->m_launch_latency == 0) {
          /* [한국어] 커널을 GPU 하드웨어 스케줄러에 전달할 조건 충족 */
          if (g_debug_execution >= 3) {
            printf("kernel %d: \'%s\' transfer to GPU hardware scheduler\n",
                   m_kernel->get_uid(), m_kernel->name().c_str());
            /* [한국어] 커널 정보 디버그 출력 */
            m_kernel->print_parent_info();
            /* [한국어] 부모 커널 정보 출력 (CDP) */
          }
          gpu->set_cache_config(m_kernel->name());
          /* [한국어] 캐시 구성 설정 (기능 시뮬레이션 경로와 동일) */
          gpu->launch(m_kernel);
          /* [한국어] 타이밍 시뮬레이션 모드로 커널 launch — GPU의 실행 큐에 추가.
           * 이 커널의 완료는 gpu->finished_kernel()로 비동기 감지.
           * record_next_done()은 register_finished_kernel()에서 나중에 호출된다. */
        } else {
          /* [한국어] 아직 실행 조건 미충족 — 다음 사이클에 재시도 */
          if (m_kernel->m_launch_latency) m_kernel->m_launch_latency--;
          /* [한국어] launch latency가 남아있으면 1 감소 (카운트다운).
           * m_launch_latency가 0이 되면 다음 사이클에 can_start_kernel 조건만 확인. */
          if (g_debug_execution >= 3)
            printf(
                "kernel %d: \'%s\', latency %u not ready to transfer to GPU "
                "hardware scheduler\n",
                m_kernel->get_uid(), m_kernel->name().c_str(),
                m_kernel->m_launch_latency);
          /* [한국어] 아직 대기 중인 커널 상태 디버그 출력 */
          return false;
          /* [한국어] false 반환 — stream_manager::operation()이 cancel_front()를 호출하여
           * 다음 사이클에 이 커널을 다시 시도하게 됨 */
        }
      }
      break;
    case stream_event: {
      /* [한국어] 이벤트 기록:
       * 현재 시뮬레이션 사이클과 wall-clock 시간을 이벤트에 타임스탬프로 기록한다.
       * cudaEventRecord()에 대응한다. */
      printf("event update\n");                     /* [한국어] 이벤트 처리 디버그 출력 */
      time_t wallclock = time((time_t *)NULL);       /* [한국어] 현재 wall-clock 시간 획득 (Unix timestamp) */
      m_event->update(gpu->gpu_tot_sim_cycle, wallclock);
      /* [한국어] 이벤트 객체에 현재 누적 시뮬레이션 사이클과 wall-clock 시간 기록 */
      m_stream->record_next_done();                 /* [한국어] 이벤트 기록 완료 — 큐에서 제거 */
    } break;
    case stream_wait_event:
      /* [한국어] 이벤트 대기 장벽:
       * m_event->num_updates() >= m_cnt이면 이벤트가 완료된 것으로 판단하여 큐를 진행.
       * m_cnt는 생성 시점의 m_event->num_issued() 스냅샷이다.
       * 조건 미충족 시 return false — 다음 사이클에 재시도. */
      // only allows next op to go if event is done
      // otherwise stays in the stream queue
      printf("stream wait event processing...\n");  /* [한국어] 이벤트 대기 처리 디버그 출력 */
      if (m_event->num_updates() >= m_cnt) {
        /* [한국어] 이벤트 update 횟수가 생성 시점 issue 수 이상이면 — 이벤트가 GPU를 통과했음 */
        printf("stream wait event done\n");         /* [한국어] 이벤트 완료 디버그 출력 */
        m_stream->record_next_done();               /* [한국어] 대기 작업 완료 — 큐에서 제거 */
      } else {
        return false;
        /* [한국어] 이벤트 미완료 — 이 스트림 작업을 다시 큐 앞에 두고 다음 사이클 재시도 */
      }
      break;
    default:
      abort();
      /* [한국어] 알 수 없는 작업 타입 — 프로그램 즉시 중단 (SIGABRT).
       * 새 stream_operation_type이 추가되었는데 여기 case가 없으면 이 분기에 진입한다. */
  }
  m_done = true;    /* [한국어] 작업 정상 완료 표시 (break로 switch 탈출한 모든 경우) */
  fflush(stdout);   /* [한국어] 출력 버퍼 즉시 비우기 — 시뮬레이터 종료 전 로그가 잘리지 않도록 */
  return true;      /* [한국어] 완료 반환 */
}

/*
 * [한국어]
 * stream_operation::print - 작업 타입을 파일 스트림에 출력
 *
 * @fp: 출력 대상 FILE 포인터.
 * @return (없음)
 *
 * m_type을 switch로 판별하여 대응하는 한국어 문자열을 fprintf로 출력한다.
 * CUstream_st::print()와 stream_manager::push() 디버그 경로에서 사용된다.
 *
 * 호출 체인:
 *   CUstream_st::print() → stream_operation::print()
 *   stream_manager::push() [g_debug_execution >= 3] → print_impl() → stream_operation::print()
 */
void stream_operation::print(FILE *fp) const {
  fprintf(fp, " stream operation ");               /* [한국어] 고정 접두사 출력 */
  switch (m_type) {
    case stream_event:
      fprintf(fp, "event");                        /* [한국어] 이벤트 기록 타입 */
      break;
    case stream_kernel_launch:
      fprintf(fp, "kernel");                       /* [한국어] 커널 실행 타입 */
      break;
    case stream_memcpy_device_to_device:
      fprintf(fp, "memcpy device-to-device");      /* [한국어] D2D 복사 타입 */
      break;
    case stream_memcpy_device_to_host:
      fprintf(fp, "memcpy device-to-host");        /* [한국어] D2H 복사 타입 */
      break;
    case stream_memcpy_host_to_device:
      fprintf(fp, "memcpy host-to-device");        /* [한국어] H2D 복사 타입 */
      break;
    case stream_memcpy_to_symbol:
      fprintf(fp, "memcpy to symbol");             /* [한국어] 상수 메모리 심볼 쓰기 타입 */
      break;
    case stream_memcpy_from_symbol:
      fprintf(fp, "memcpy from symbol");           /* [한국어] 상수 메모리 심볼 읽기 타입 */
      break;
    case stream_no_op:
      fprintf(fp, "no-op");                        /* [한국어] 빈 작업 타입 */
      break;
    default:
      break;                                       /* [한국어] stream_wait_event 등 미처리 타입은 출력 없음 */
  }
}

/*
 * [한국어]
 * stream_manager::stream_manager - 스트림 관리자 생성자
 *
 * @gpu: GPU 시뮬레이터 포인터. do_operation()에서 memcpy/launch에 사용.
 * @cuda_launch_blocking: true이면 blocking 모드 — 모든 작업이 스트림 0으로 직렬화됨.
 * @return (생성자)
 *
 * m_gpu와 m_cuda_launch_blocking을 저장하고, m_service_stream_zero = false로 초기화한다.
 * m_lock을 기본 속성으로 초기화하고, m_last_stream을 m_streams.begin()으로 설정한다.
 * (m_streams가 비어있으므로 m_last_stream = end()와 동일)
 *
 * 실행 컨텍스트: 호스트 쓰레드 (gpgpu_ptx_sim_init_perf() 중 new stream_manager() 호출).
 *
 * 호출 체인:
 *   gpgpu_ptx_sim_init_perf() → new stream_manager(gpu, blocking) → [이 생성자]
 */
stream_manager::stream_manager(gpgpu_sim *gpu, bool cuda_launch_blocking) {
  m_gpu = gpu;                                    /* [한국어] GPU 시뮬레이터 포인터 저장 */
  m_service_stream_zero = false;                  /* [한국어] 스트림 0 우선 서비스 초기값 false */
  m_cuda_launch_blocking = cuda_launch_blocking;  /* [한국어] blocking 모드 플래그 저장 */
  pthread_mutex_init(&m_lock, NULL);              /* [한국어] 스트림 목록 보호 뮤텍스 초기화 */
  m_last_stream = m_streams.begin();              /* [한국어] 라운드로빈 시작점 설정 (현재는 end()와 동일) */
}

/*
 * [한국어]
 * stream_manager::operation - 시뮬레이션 사이클마다 호출되는 스트림 처리 주 함수
 *
 * @sim: out 파라미터. 커널이 이번 사이클에 실행되었으면 true로 설정됨.
 * @return: true이면 커널이 방금 완료됨 (gpgpu_sim_thread_concurrent의 break 조건).
 *
 * 처리 순서:
 *   1) check_finished_kernel()로 이미 완료된 커널이 있으면 스트림 큐에서 제거
 *   2) m_lock 획득 후 front()로 다음 실행할 작업 선택
 *   3) op.do_operation(m_gpu)로 작업 실행 시도
 *   4) 실행 실패(false 반환) 시:
 *      - 커널 작업이었다면 m_grid_id_to_stream에서 해당 grid 제거
 *      - op.get_stream()->cancel_front()로 pending 상태 해제 (다음 사이클 재시도)
 *   5) m_lock 해제 후 check 결과(커널 완료 여부) 반환
 *
 * 실행 컨텍스트: 시뮬레이션 쓰레드 (gpgpu_sim_thread_concurrent의 내부 루프).
 *
 * 호출 체인:
 *   gpgpu_sim_thread_concurrent() → [operation]
 *     → check_finished_kernel() → register_finished_kernel() → record_next_done()
 *     → front() → CUstream_st::next()
 *     → stream_operation::do_operation()
 */
bool stream_manager::operation(bool *sim) {
  bool check = check_finished_kernel();
  /* [한국어] 이미 GPU에서 완료된 커널이 있으면 스트림 큐에서 제거하고 true 반환.
   * 완료된 커널이 없으면 false. 이 결과를 최종 반환값으로 사용한다. */
  pthread_mutex_lock(&m_lock);                /* [한국어] m_streams, m_grid_id_to_stream 접근 전 락 획득 */
  //    if(check)m_gpu->print_stats();
  /* [한국어] 주석 처리된 디버그 코드: 커널 완료 시 통계 출력 (성능 오버헤드로 비활성화) */
  stream_operation op = front();              /* [한국어] 다음 실행할 스트림 작업 선택 (라운드로빈) */
  if (!op.do_operation(m_gpu))  // not ready to execute
  {
    /* [한국어] 작업이 아직 실행 준비되지 않음 (커널 launch latency, 이벤트 미완료 등)
     * 작업을 취소하고 다음 사이클에 재시도한다. */
    // cancel operation
    if (op.is_kernel()) {
      /* [한국어] 커널 작업이 실패한 경우:
       * front()에서 m_grid_id_to_stream에 등록했던 grid_uid를 제거해야 한다.
       * 그렇지 않으면 finish 검사 시 잘못된 스트림을 참조하게 된다. */
      unsigned grid_uid = op.get_kernel()->get_uid();  /* [한국어] 커널의 grid UID 얻기 */
      m_grid_id_to_stream.erase(grid_uid);             /* [한국어] 맵에서 제거 — 재시도 시 다시 등록 */
    }
    op.get_stream()->cancel_front();
    /* [한국어] 이 작업의 스트림에서 pending 상태 해제.
     * cancel_front()가 m_pending = false로 설정하여 다음 사이클 front()에서 다시 선택 가능. */
  }
  pthread_mutex_unlock(&m_lock);              /* [한국어] 락 해제 */
  // pthread_mutex_lock(&m_lock);
  // simulate a clock cycle on the GPU
  /* [한국어] 주석 처리된 코드: 이전 버전에서 GPU 사이클 시뮬레이션을 여기서 했을 가능성.
   * 현재는 gpgpu_sim_thread_concurrent()의 내부 루프에서 사이클 진행. */
  return check;
  /* [한국어] 커널 완료 여부 반환 — true이면 gpgpu_sim_thread_concurrent()가 루프를 탈출. */
}

/*
 * [한국어]
 * stream_manager::check_finished_kernel - GPU에서 완료된 커널 확인 및 처리
 *
 * @return: 커널이 완료 처리되었으면 true, 아니면 false.
 *
 * gpu->finished_kernel()은 완료된 커널의 grid_uid를 반환한다 (없으면 0).
 * 반환된 uid를 register_finished_kernel()에 전달하여 스트림 큐를 정리한다.
 *
 * 실행 컨텍스트: 시뮬레이션 쓰레드 (operation()에서 매 사이클 호출).
 *
 * 호출 체인:
 *   operation() → [check_finished_kernel] → register_finished_kernel()
 */
bool stream_manager::check_finished_kernel() {
  unsigned grid_uid = m_gpu->finished_kernel();
  /* [한국어] GPU 시뮬레이터에서 완료된 커널의 grid UID 얻기.
   * finished_kernel()은 내부 완료 큐에서 uid를 pop하여 반환 (없으면 0). */
  bool check = register_finished_kernel(grid_uid);
  /* [한국어] 완료된 커널에 대한 스트림 정리 수행 */
  return check;
  /* [한국어] 커널 완료 처리 여부 반환 */
}

/*
 * [한국어]
 * stream_manager::register_finished_kernel - 완료된 커널의 스트림 큐 정리 및 메모리 해제
 *
 * @grid_uid: 완료된 커널의 grid UID. 0이면 완료된 커널 없음을 의미한다.
 * @return: 실제로 커널이 완료 처리되었으면 true, 아니면 false.
 *
 * 처리 순서:
 *   1) grid_uid > 0인지 확인 (유효한 커널 UID)
 *   2) m_grid_id_to_stream[grid_uid]로 이 커널이 속한 스트림 찾기
 *   3) stream->front().get_kernel()로 해당 커널 메타데이터 포인터 얻기
 *   4) kernel->is_finished()로 CDP(CUDA Dynamic Parallelism) 자식 커널 포함 완료 확인
 *   5) 완료되었으면:
 *      - stream->record_next_done(): 큐에서 커널 작업 제거
 *      - m_grid_id_to_stream.erase(grid_uid): 맵에서 제거
 *      - kernel->notify_parent_finished(): 부모 커널(CDP)에 완료 통지
 *      - delete kernel: 커널 메타데이터 해제
 *
 * CDP(CUDA Dynamic Parallelism): GPU 커널 안에서 또 다른 커널을 launch하는 기능.
 * 부모 커널이 완료되려면 자식 커널도 모두 완료되어야 한다.
 *
 * 실행 컨텍스트: 시뮬레이션 쓰레드 (check_finished_kernel() 경유).
 *
 * 호출 체인:
 *   check_finished_kernel() → [register_finished_kernel]
 *     → stream->record_next_done()
 *     → kernel->notify_parent_finished()
 *     → delete kernel
 */
bool stream_manager::register_finished_kernel(unsigned grid_uid) {
  // called by gpu simulation thread
  if (grid_uid > 0) {
    /* [한국어] grid_uid가 0이 아닌 유효한 커널 완료 신호 */
    CUstream_st *stream = m_grid_id_to_stream[grid_uid];
    /* [한국어] 이 커널이 속한 스트림 포인터 조회.
     * m_grid_id_to_stream에 등록되지 않은 uid가 오면 새 키(NULL stream)가 삽입될 수 있음 — 버그. */
    kernel_info_t *kernel = stream->front().get_kernel();
    /* [한국어] 스트림 front 작업에서 커널 메타데이터 포인터 얻기.
     * 이 커널이 m_grid_id_to_stream에 등록된 커널과 일치해야 한다. */
    assert(grid_uid == kernel->get_uid());
    /* [한국어] grid_uid와 스트림 front 커널 UID가 일치해야 함.
     * 불일치하면 m_grid_id_to_stream과 스트림 큐 상태가 맞지 않는 버그. */

    // Jin: should check children kernels for CDP
    if (kernel->is_finished()) {
      /* [한국어] is_finished(): 이 커널의 모든 스레드와 CDP 자식 커널이 완료되었는지 확인.
       * 자식 커널이 아직 실행 중이면 부모 커널은 완료로 처리하지 않는다. */
      //            std::ofstream kernel_stat("kernel_stat.txt",
      //            std::ofstream::out | std::ofstream::app); kernel_stat<< "
      //            kernel " << grid_uid << ": " << kernel->name();
      //            if(kernel->get_parent())
      //                kernel_stat << ", parent " <<
      //                kernel->get_parent()->get_uid() <<
      //                ", launch " << kernel->launch_cycle;
      //            kernel_stat<< ", start " << kernel->start_cycle <<
      //                ", end " << kernel->end_cycle << ", retire " <<
      //                gpu_sim_cycle + gpu_tot_sim_cycle << "\n";
      //            printf("kernel %d finishes, retires from stream %d\n",
      //            grid_uid, stream->get_uid()); kernel_stat.flush();
      //            kernel_stat.close();
      /* [한국어] 주석 처리된 코드: 커널 통계(시작/종료 사이클)를 파일에 기록하는 디버그 코드.
       * 성능 오버헤드로 비활성화되어 있다. */
      stream->record_next_done();
      /* [한국어] 스트림 큐에서 이 커널 작업 제거 및 m_pending 해제 */
      m_grid_id_to_stream.erase(grid_uid);
      /* [한국어] grid → 스트림 맵에서 이 항목 제거 — 이후 같은 grid_uid 재사용 시 충돌 방지 */
      kernel->notify_parent_finished();
      /* [한국어] CDP 사용 시 부모 커널에 "자식이 완료됨"을 알림.
       * 부모 커널이 is_finished() 조건을 재평가하도록 트리거한다. */
      delete kernel;
      /* [한국어] 커널 메타데이터 해제 — 이후 kernel 포인터 접근 금지 (dangling pointer) */
      return true;      /* [한국어] 커널 완료 처리 완료 */
    }
  }

  return false;
  /* [한국어] grid_uid == 0이거나 kernel->is_finished()가 false이면 완료 처리 안 함 */
}

/*
 * [한국어]
 * stream_manager::stop_all_running_kernels - 모든 실행 중인 커널 강제 종료 및 정리
 *
 * @param (없음)
 * @return (없음)
 *
 * 최대 사이클/명령/CTA 한도 초과 시 gpgpu_sim_thread_concurrent()에서 호출된다.
 * 처리 순서:
 *   1) m_lock 획득
 *   2) m_gpu->get_running_kernels()로 실행 중인 커널 목록 수집
 *   3) 각 커널의 streamID 기록 (통계 출력 시 사용)
 *   4) m_gpu->stop_all_running_kernels()로 GPU에 모든 커널 중단 명령
 *   5) check_finished_kernel() 루프로 완료된 커널들을 스트림 큐에서 모두 제거
 *   6) 완료된 각 스트림에 대해 m_gpu->print_stats(streamID) 통계 출력
 *   7) m_lock 해제
 *
 * 실행 컨텍스트: 시뮬레이션 쓰레드.
 *
 * 호출 체인:
 *   gpgpu_sim_thread_concurrent() [한도 초과 시] → [stop_all_running_kernels]
 *     → m_gpu->get_running_kernels()
 *     → m_gpu->stop_all_running_kernels()
 *     → check_finished_kernel() (반복)
 *     → m_gpu->print_stats()
 */
void stream_manager::stop_all_running_kernels() {
  pthread_mutex_lock(&m_lock);          /* [한국어] 스트림/커널 맵 수정 전 락 획득 */

  std::vector<unsigned long long> finished_streams;
  /* [한국어] 종료시킬 커널들의 streamID 목록 — 나중에 통계 출력에 사용 */
  std::vector<kernel_info_t *> running_kernels = m_gpu->get_running_kernels();
  /* [한국어] GPU에서 현재 실행 중인 모든 커널의 메타데이터 포인터 목록 가져오기 */
  for (kernel_info_t *k : running_kernels) {
    /* [한국어] 실행 중인 각 커널의 streamID 수집 */
    if (k != NULL) {
      /* [한국어] NULL 항목 방어적 검사 (실행 큐에 빈 슬롯이 있을 수 있음) */
      finished_streams.push_back(k->get_streamID());
      /* [한국어] 이 커널이 속한 스트림 ID를 목록에 추가 */
    }
  }

  // Signal m_gpu to stop all running kernels
  m_gpu->stop_all_running_kernels();
  /* [한국어] GPU 시뮬레이터에 모든 커널 중단 명령.
   * 내부적으로 실행 중인 모든 커널의 완료 큐에 즉시 완료 신호를 보낸다. */

  // Clean up all streams waiting on running kernels
  int count = 0;           /* [한국어] 완료 처리된 커널 수 카운터 (현재 출력 없음) */
  while (check_finished_kernel()) {
    count++;
    /* [한국어] 완료된 커널이 더 이상 없을 때까지 반복하여 스트림 큐 정리.
     * stop_all_running_kernels()로 강제 완료되었으므로 모두 처리될 때까지 반복. */
  }

  // If any kernels completed, print out the current stats
  for (unsigned long long streamID : finished_streams) {
    m_gpu->print_stats(streamID);
    /* [한국어] 강제 종료된 각 커널의 최종 성능 통계 출력.
     * streamID를 기반으로 해당 스트림에서 실행된 커널의 통계를 출력. */
  }

  pthread_mutex_unlock(&m_lock);        /* [한국어] 모든 정리 완료 후 락 해제 */
}

/*
 * [한국어]
 * stream_manager::front - 다음 실행할 스트림 작업을 선택하는 라운드로빈 스케줄러
 *
 * @return: 선택된 stream_operation 복사본.
 *          처리할 작업이 없으면 stream_no_op 타입의 기본 stream_operation 반환.
 *
 * 스케줄링 우선순위:
 *   1) 항상 m_service_stream_zero = true로 강제 설정하여 스트림 0을 먼저 확인.
 *      (원래 주석된 concurrent_streams_empty() 조건부 설정이 있었으나 현재 무조건 true)
 *   2) 스트림 0이 비어있지 않고 busy하지 않으면 스트림 0의 next() 반환.
 *   3) 스트림 0 서비스 불가이면 m_service_stream_zero = false로 전환.
 *   4) m_last_stream 다음부터 시작하여 m_streams를 순환하며
 *      !busy() && !empty()인 첫 번째 스트림의 next() 반환.
 * 선택된 커널 작업은 m_grid_id_to_stream에 등록된다.
 *
 * 실행 컨텍스트: 시뮬레이션 쓰레드. operation()이 m_lock을 잡은 상태에서 호출.
 *
 * 호출 체인:
 *   operation() [m_lock 보유 중] → [front] → CUstream_st::next()
 */
stream_operation stream_manager::front() {
  // called by gpu simulation thread
  stream_operation result;              /* [한국어] 반환할 작업 (기본값: stream_no_op, m_done=true) */
  //    if( concurrent_streams_empty() )
  m_service_stream_zero = true;
  /* [한국어] 주석된 코드: 원래는 concurrent_streams_empty()이면 스트림 0 서비스.
   * 현재는 무조건 true로 설정하여 스트림 0을 항상 먼저 검사한다.
   * TODO: 이로 인해 사용자 스트림이 스트림 0보다 항상 후순위가 됨. */
  if (m_service_stream_zero) {
    /* [한국어] 스트림 0 서비스 시도 */
    if (!m_stream_zero.empty() && !m_stream_zero.busy()) {
      /* [한국어] 스트림 0에 작업이 있고 현재 처리 중인 작업이 없는 경우 */
      result = m_stream_zero.next();    /* [한국어] 스트림 0의 front 작업을 꺼내어 pending 설정 */
      if (result.is_kernel()) {
        /* [한국어] 커널 작업이면 grid → 스트림 맵에 등록 (완료 시 스트림 찾기용) */
        unsigned grid_id = result.get_kernel()->get_uid();  /* [한국어] 커널의 grid UID */
        m_grid_id_to_stream[grid_id] = &m_stream_zero;     /* [한국어] 스트림 0 포인터 등록 */
      }
    } else {
      m_service_stream_zero = false;
      /* [한국어] 스트림 0이 비어있거나 busy하면 사용자 스트림으로 전환 */
    }
  }
  if (!m_service_stream_zero) {
    /* [한국어] 스트림 0 서비스 불가 — 사용자 스트림 라운드로빈 탐색 */
    std::list<struct CUstream_st *>::iterator s = m_last_stream;
    /* [한국어] 마지막으로 서비스한 스트림 위치에서 시작 */
    if (m_last_stream == m_streams.end()) {
      s = m_streams.begin();
      /* [한국어] 마지막 위치가 end()이면 처음으로 순환 */
    } else {
      s++;
      /* [한국어] 마지막 서비스 스트림 다음부터 탐색 (라운드로빈) */
    }
    for (size_t ii = 0; ii < m_streams.size(); ii++, s++) {
      /* [한국어] 모든 스트림을 최대 한 바퀴 순환하며 탐색 */
      if (s == m_streams.end()) {
        s = m_streams.begin();
        /* [한국어] 리스트 끝에 도달하면 처음으로 순환 (원형 탐색) */
      }
      m_last_stream = s;                /* [한국어] 현재 탐색 위치를 마지막 서비스 위치로 업데이트 */
      CUstream_st *stream = *s;         /* [한국어] 현재 스트림 포인터 역참조 */
      if (!stream->busy() && !stream->empty()) {
        /* [한국어] 처리 중이 아니고 대기 중인 작업이 있는 스트림 발견 */
        result = stream->next();        /* [한국어] front 작업 꺼내기 및 pending 설정 */
        if (result.is_kernel()) {
          /* [한국어] 커널 작업이면 grid → 스트림 맵 등록 */
          unsigned grid_id = result.get_kernel()->get_uid();  /* [한국어] 커널 grid UID */
          m_grid_id_to_stream[grid_id] = stream;             /* [한국어] 이 스트림 포인터 등록 */
        }
        break;
        /* [한국어] 첫 번째로 발견된 처리 가능한 스트림 선택 후 탐색 종료 */
      }
    }
  }
  return result;
  /* [한국어] 선택된 작업 반환. 처리할 작업이 없으면 기본 stream_no_op 작업 반환. */
}

/*
 * [한국어]
 * stream_manager::add_stream - 새 CUDA 스트림을 관리 목록에 추가
 *
 * @stream: 추가할 CUstream_st 포인터 (new CUstream_st()로 생성된 객체).
 * @return (없음)
 *
 * m_lock을 잡고 m_streams 리스트 끝에 스트림 포인터를 추가한다.
 *
 * 실행 컨텍스트: 호스트 쓰레드 (cudaStreamCreate() 시).
 *
 * 호출 체인:
 *   libcuda::cudaStreamCreate() → g_stream_manager->add_stream(stream)
 */
void stream_manager::add_stream(struct CUstream_st *stream) {
  // called by host thread
  pthread_mutex_lock(&m_lock);          /* [한국어] m_streams 수정 전 락 획득 */
  m_streams.push_back(stream);          /* [한국어] 스트림 포인터를 리스트 끝에 추가 */
  pthread_mutex_unlock(&m_lock);        /* [한국어] 락 해제 */
}

/*
 * [한국어]
 * stream_manager::destroy_stream - CUDA 스트림 제거 및 메모리 해제
 *
 * @stream: 제거할 CUstream_st 포인터.
 * @return (없음)
 *
 * m_lock을 잡고 스트림이 비어있을 때까지 spin-wait하며 대기한다.
 * 이후 m_streams 리스트에서 제거하고 delete stream으로 메모리 해제한다.
 * m_last_stream을 m_streams.begin()으로 리셋하여 라운드로빈 상태 초기화.
 *
 * 주의: m_lock을 잡은 채로 stream->empty()를 스핀하면 시뮬레이션 쓰레드가
 *       m_lock을 잡으려 할 때 데드락이 발생할 수 있다.
 *       현재 구현의 while(!stream->empty())는 m_lock 안에서 실행되어
 *       시뮬레이션 쓰레드의 operation()이 m_lock 획득을 무한 대기하는 상황이 된다.
 *       → 이는 잠재적 데드락 버그이나 원본 코드를 그대로 유지한다.
 *
 * 실행 컨텍스트: 호스트 쓰레드 (cudaStreamDestroy() 시).
 *
 * 호출 체인:
 *   libcuda::cudaStreamDestroy() → g_stream_manager->destroy_stream(stream)
 */
void stream_manager::destroy_stream(CUstream_st *stream) {
  // called by host thread
  pthread_mutex_lock(&m_lock);          /* [한국어] m_streams 수정 전 락 획득 */
  while (!stream->empty())
    ;
  /* [한국어] 스트림의 모든 작업이 완료될 때까지 대기 (바쁜 대기).
   * stream->empty()는 CUstream_st::m_lock을 내부적으로 잡아 확인한다.
   * 잠재적 데드락: stream_manager::m_lock → CUstream_st::m_lock 순서로 이중 잠금 발생. */
  std::list<CUstream_st *>::iterator s;  /* [한국어] m_streams 탐색 반복자 */
  for (s = m_streams.begin(); s != m_streams.end(); s++) {
    /* [한국어] 제거할 스트림 포인터를 리스트에서 탐색 */
    if (*s == stream) {
      m_streams.erase(s);               /* [한국어] 리스트에서 해당 스트림 제거 */
      break;                            /* [한국어] 찾으면 즉시 탐색 종료 */
    }
  }
  delete stream;                        /* [한국어] CUstream_st 객체 메모리 해제 */
  m_last_stream = m_streams.begin();    /* [한국어] 라운드로빈 시작점 리셋 (erase 후 반복자 무효화 방지) */
  pthread_mutex_unlock(&m_lock);        /* [한국어] 락 해제 */
}

/*
 * [한국어]
 * stream_manager::concurrent_streams_empty - 사용자 스트림들이 모두 비어있는지 확인
 *
 * @return: m_streams의 모든 스트림이 empty()이면 true.
 *
 * 잠금 없이 각 스트림을 순회한다 (단, 각 stream->empty()는 내부적으로 m_lock 잠금).
 * 호출자가 stream_manager::m_lock을 잡은 상태에서 호출해야 안전하다.
 *
 * 실행 컨텍스트: 주로 시뮬레이션 쓰레드 (operation(), empty_protected()).
 *
 * 호출 체인:
 *   empty_protected() → [concurrent_streams_empty]
 *   empty() → [concurrent_streams_empty]
 */
bool stream_manager::concurrent_streams_empty() {
  bool result = true;                   /* [한국어] 초기값 true (빈 것으로 가정) */
  if (m_streams.empty()) return true;   /* [한국어] 사용자 스트림이 하나도 없으면 즉시 true 반환 */
  // called by gpu simulation thread
  std::list<struct CUstream_st *>::iterator s;  /* [한국어] 스트림 목록 순회 반복자 */
  for (s = m_streams.begin(); s != m_streams.end(); ++s) {
    /* [한국어] 각 사용자 스트림을 순회하며 비어있는지 확인 */
    struct CUstream_st *stream = *s;    /* [한국어] 현재 스트림 포인터 */
    if (!stream->empty()) {
      // stream->print(stdout);
      /* [한국어] 주석 처리된 디버그 코드: 비어있지 않은 스트림 상태 출력 (성능상 비활성화) */
      result = false;                   /* [한국어] 비어있지 않은 스트림 발견 — 결과 false */
      break;                            /* [한국어] 하나라도 비어있지 않으면 탐색 조기 종료 */
    }
  }
  return result;
  /* [한국어] 모든 스트림이 비어있으면 true, 하나라도 작업 남아있으면 false */
}

/*
 * [한국어]
 * stream_manager::empty_protected - 뮤텍스 보호하에 모든 스트림이 비어있는지 확인
 *
 * @return: m_streams와 m_stream_zero 모두 비어있으면 true.
 *
 * m_lock을 잡고 concurrent_streams_empty()와 m_stream_zero.empty()를 확인한다.
 * 두 조건 중 하나라도 false이면 result = false.
 * gpgpu_sim_thread_concurrent()에서 모든 작업 완료를 판단하는 데 사용된다.
 *
 * 실행 컨텍스트: 시뮬레이션 쓰레드.
 *
 * 호출 체인:
 *   gpgpu_sim_thread_concurrent() 루프 조건 → [empty_protected]
 */
bool stream_manager::empty_protected() {
  bool result = true;                   /* [한국어] 초기값 true (모두 비어있다고 가정) */
  pthread_mutex_lock(&m_lock);          /* [한국어] 스트림 상태 확인 전 락 획득 */
  if (!concurrent_streams_empty()) result = false;
  /* [한국어] 사용자 스트림 중 하나라도 비어있지 않으면 false */
  if (!m_stream_zero.empty()) result = false;
  /* [한국어] 스트림 0도 비어있어야 함 */
  pthread_mutex_unlock(&m_lock);        /* [한국어] 락 해제 */
  return result;
  /* [한국어] 모두 비어있으면 true (시뮬레이션 쓰레드가 유휴 상태로 전환 조건) */
}

/*
 * [한국어]
 * stream_manager::empty - 잠금 없이 모든 스트림이 비어있는지 확인
 *
 * @return: 모든 스트림이 비어있으면 true.
 *
 * m_lock을 잡지 않으므로 호출자가 이미 락을 잡았거나 단일 쓰레드 맥락에서 사용해야 한다.
 * push()에서 blocking 대기 종료 조건 확인, synchronize()에서 사용된다.
 *
 * 호출 체인:
 *   push() [blocking 대기] → [empty]
 *   gpgpu_context::synchronize() → g_stream_manager->empty()
 */
bool stream_manager::empty() {
  bool result = true;                   /* [한국어] 초기값 true */
  if (!concurrent_streams_empty()) result = false;
  /* [한국어] 사용자 스트림 비어있는지 확인 */
  if (!m_stream_zero.empty()) result = false;
  /* [한국어] 스트림 0 비어있는지 확인 */
  return result;
  /* [한국어] 모두 비어있어야 true */
}

/*
 * [한국어]
 * stream_manager::print - 스트림 관리자 상태를 파일 스트림에 출력 (잠금 포함)
 *
 * @fp: 출력 대상 FILE 포인터.
 * @return (없음)
 *
 * m_lock을 잡고 print_impl()을 호출한다.
 * 디버그/로깅 목적으로 외부에서 사용된다.
 *
 * 호출 체인:
 *   gpgpu_context::synchronize() [디버그] → g_stream_manager->print(stdout)
 */
void stream_manager::print(FILE *fp) {
  pthread_mutex_lock(&m_lock);          /* [한국어] 출력 중 다른 쓰레드의 수정 방지 */
  print_impl(fp);                       /* [한국어] 실제 출력 수행 */
  pthread_mutex_unlock(&m_lock);        /* [한국어] 락 해제 */
}

/*
 * [한국어]
 * stream_manager::print_impl - 스트림 상태 출력의 실제 구현 (잠금 없음)
 *
 * @fp: 출력 대상 FILE 포인터.
 * @return (없음)
 *
 * "GPGPU-Sim API: Stream Manager State" 헤더를 출력하고,
 * m_streams의 각 비어있지 않은 스트림에 대해 CUstream_st::print()를 호출한다.
 * m_stream_zero가 비어있지 않으면 스트림 0도 출력한다.
 *
 * 주의: 이 함수는 m_lock을 잡지 않는다. 반드시 호출자가 잠금 상태에서 호출해야 한다.
 *
 * 호출 체인:
 *   print() [m_lock 보유 중] → [print_impl]
 *   push() [m_lock 보유 중, g_debug_execution >= 3] → [print_impl]
 */
void stream_manager::print_impl(FILE *fp) {
  fprintf(fp, "GPGPU-Sim API: Stream Manager State\n");
  /* [한국어] 스트림 관리자 상태 출력 헤더 */
  std::list<struct CUstream_st *>::iterator s;  /* [한국어] 스트림 목록 순회 반복자 */
  for (s = m_streams.begin(); s != m_streams.end(); ++s) {
    /* [한국어] 각 사용자 스트림 순회 */
    struct CUstream_st *stream = *s;            /* [한국어] 현재 스트림 포인터 */
    if (!stream->empty()) stream->print(fp);    /* [한국어] 비어있지 않은 스트림만 출력 */
  }
  if (!m_stream_zero.empty()) m_stream_zero.print(fp);
  /* [한국어] 스트림 0도 비어있지 않으면 출력 */
}

/*
 * [한국어]
 * stream_manager::push - 스트림 작업을 적절한 스트림 큐에 추가 (가장 복잡한 함수)
 *
 * @op: 추가할 stream_operation 객체 (값 복사).
 * @return (없음)
 *
 * 처리 순서:
 *   1) op.get_stream() == NULL이거나 m_cuda_launch_blocking이면 block = true.
 *      block == true인 동안 concurrent_streams_empty()가 true가 될 때까지 spin-wait.
 *      → 스트림 0(또는 blocking 모드)은 다른 스트림의 모든 작업이 끝난 뒤에만 시작.
 *   2) m_lock 획득 후 m_gpu->cycle_insn_cta_max_hit() 확인:
 *      - false(한도 미도달): 작업 허용
 *        * stream != NULL && !blocking: stream->push(op)
 *        * 그 외(스트림 0 또는 blocking): op.set_stream(&m_stream_zero) 후 m_stream_zero.push(op)
 *      - true(한도 도달): 작업 무시하고 경고 메시지 출력
 *   3) g_debug_execution >= 3이면 print_impl(stdout)으로 현재 상태 출력
 *   4) m_lock 해제
 *   5) SST 모드가 아니고 blocking/스트림 0 작업이면:
 *      empty()가 true가 될 때까지 usleep() 기반 exponential backoff 대기.
 *      (wait_amount: 100μs → 200μs → ... → 최대 100ms)
 *      → CUDA blocking 모드에서 커널/memcpy 완료를 CPU가 기다리는 구현.
 *
 * 실행 컨텍스트: 호스트 쓰레드.
 *
 * 호출 체인:
 *   libcuda::cuLaunchKernel / cudaMemcpyAsync / cudaEventRecord → [push]
 *     → CUstream_st::push() 또는 m_stream_zero.push()
 */
void stream_manager::push(stream_operation op) {
  struct CUstream_st *stream = op.get_stream();
  /* [한국어] 이 작업의 소속 스트림 포인터 얻기 (NULL이면 스트림 0 대상) */

  // block if stream 0 (or concurrency disabled) and pending concurrent
  // operations exist
  bool block = !stream || m_cuda_launch_blocking;
  /* [한국어] block = true 조건:
   *   1) stream == NULL: 스트림 0에 추가하는 작업 (cudaMemcpy 동기 버전 등)
   *   2) m_cuda_launch_blocking: 전체 blocking 모드로 설정된 경우
   * blocking이 필요한 경우, 다른 스트림의 모든 작업이 완료될 때까지 대기 후 push. */
  while (block) {
    /* [한국어] 다른 스트림의 모든 작업이 완료될 때까지 spin-wait */
    pthread_mutex_lock(&m_lock);            /* [한국어] concurrent_streams_empty 확인 전 락 */
    block = !concurrent_streams_empty();    /* [한국어] 사용자 스트림이 모두 비어있으면 block 해제 */
    pthread_mutex_unlock(&m_lock);          /* [한국어] 즉시 락 해제하여 시뮬레이션 쓰레드 작업 가능하게 */
  };

  pthread_mutex_lock(&m_lock);              /* [한국어] 작업 큐에 추가하기 위한 락 획득 */
  if (!m_gpu->cycle_insn_cta_max_hit()) {
    /* [한국어] 최대 사이클/명령/CTA 한도에 도달하지 않은 경우 — 작업 허용 */
    // Accept the stream operation if the maximum cycle/instruction/cta counts
    // are not triggered
    if (stream && !m_cuda_launch_blocking) {
      /* [한국어] 유효한 사용자 스트림이 있고 blocking 모드가 아닌 경우 — 해당 스트림에 직접 추가 */
      stream->push(op);
      /* [한국어] op를 지정된 사용자 스트림의 큐 끝에 추가 */
    } else {
      /* [한국어] 스트림 0 대상이거나 blocking 모드인 경우 */
      op.set_stream(&m_stream_zero);
      /* [한국어] 작업의 소속 스트림을 스트림 0으로 변경
       * (원래 stream이 NULL이었거나, blocking 모드에서 강제 스트림 0 사용) */
      m_stream_zero.push(op);
      /* [한국어] 스트림 0의 큐 끝에 추가 */
    }
  } else {
    /* [한국어] 한도 도달 — 새 작업 무시 */
    // Otherwise, ignore operation and continue
    printf(
        "GPGPU-Sim API: Maximum cycle, instruction, or CTA count hit. "
        "Skipping:");
    /* [한국어] 한도 초과로 작업이 무시됨을 알리는 경고 메시지 출력 */
    op.print(stdout);                 /* [한국어] 무시되는 작업의 타입 출력 */
    printf("\n");
  }
  if (g_debug_execution >= 3) print_impl(stdout);
  /* [한국어] 디버그 레벨 3 이상이면 추가 후 전체 스트림 상태 출력 */
  pthread_mutex_unlock(&m_lock);        /* [한국어] 큐 추가 완료 후 락 해제 */
  if (!m_gpu->is_SST_mode() && (m_cuda_launch_blocking || stream == NULL)) {
    /* [한국어] SST 모드가 아니고 (blocking 모드이거나 스트림 0 작업)인 경우:
     * 작업이 완료될 때까지 호스트 쓰레드를 blocking한다. */
    unsigned int wait_amount = 100;     /* [한국어] 초기 대기 시간: 100 마이크로초 */
    unsigned int wait_cap = 100000;  // 100ms
    /* [한국어] 최대 대기 시간: 100,000 마이크로초 = 100 밀리초 */
    while (!empty()) {
      /* [한국어] 모든 스트림이 비어있을 때까지 대기 (작업 완료 대기) */
      // sleep to prevent CPU hog by empty spin
      // sleep time increased exponentially ensure fast response when needed
      usleep(wait_amount);
      /* [한국어] CPU 점유를 줄이기 위해 sleep. 빈 spin보다 CPU 친화적.
       * usleep: POSIX 마이크로초 단위 슬립 */
      wait_amount *= 2;
      /* [한국어] 지수 증가: 100 → 200 → 400 → ... μs.
       * 작업이 빨리 완료될 때는 빠른 응답, 오래 걸릴 때는 CPU 오버헤드 최소화. */
      if (wait_amount > wait_cap) wait_amount = wait_cap;
      /* [한국어] 100ms 상한선 초과 방지 — 무한 지수 증가 제한 */
    }
  }
}

/*
 * [한국어]
 * stream_manager::pushCudaStreamWaitEventToAllStreams - 모든 스트림에 이벤트 대기 작업 추가
 *
 * @e: 대기할 CUDA 이벤트 포인터.
 * @flags: cudaStreamWaitEvent() flags (현재 사용 안 함, 0만 지원).
 * @return (없음)
 *
 * cudaStreamWaitEvent(cudaStreamAll, event, flags)의 효과를 구현한다.
 * m_streams의 모든 사용자 스트림에 stream_wait_event 작업을 push()한다.
 * 각 스트림은 이벤트가 완료되기 전까지 다음 작업을 실행할 수 없게 된다.
 *
 * 실행 컨텍스트: 호스트 쓰레드.
 *
 * 호출 체인:
 *   libcuda::cudaStreamWaitEvent(cudaStreamAll, ...) → [pushCudaStreamWaitEventToAllStreams]
 *     → push() → stream->push(op) [각 스트림에 대해]
 */
void stream_manager::pushCudaStreamWaitEventToAllStreams(CUevent_st *e,
                                                         unsigned int flags) {
  std::list<CUstream_st *>::iterator s;     /* [한국어] m_streams 순회 반복자 */
  for (s = m_streams.begin(); s != m_streams.end(); s++) {
    /* [한국어] 모든 사용자 스트림에 이벤트 대기 작업 추가 */
    stream_operation op(*s, e, flags);
    /* [한국어] stream_wait_event 타입의 작업 생성.
     * 생성자: stream_operation(CUstream_st*, CUevent_st*, unsigned int).
     * m_cnt = e->num_issued()로 현재 시점의 이벤트 issue 수 스냅샷 저장. */
    push(op);
    /* [한국어] 해당 스트림의 큐 끝에 작업 추가.
     * push()는 스트림이 NULL이 아니고 non-blocking이면 스트림에 직접 추가. */
  }
}
