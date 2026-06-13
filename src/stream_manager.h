/*
 * [한국어 설명] CUDA 스트림 및 작업 관리 헤더 (stream_manager.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 CUDA 스트림(CUstream_st), 스트림 작업(stream_operation),
 * 이벤트(CUevent_st), 스트림 관리자(stream_manager) 클래스를 정의한다.
 * CUDA 스트림은 GPU에서 비동기로 수행되는 작업들의 순서를 보장하는 큐로,
 * 이 파일은 GPGPU-Sim 내에서 그 동작을 소프트웨어적으로 구현한다.
 * 호스트 쓰레드(libcuda)가 push()로 작업을 등록하고,
 * 시뮬레이션 쓰레드가 operation()/front()로 작업을 꺼내어 실행한다.
 * 스트림 0(m_stream_zero)과 사용자 생성 스트림(m_streams)을 구분하여 관리하며,
 * 동시 실행(concurrent) 모드와 블로킹(blocking) 모드를 모두 지원한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * CUDA 애플리케이션의 비동기 호출 흐름:
 *   cuLaunchKernel(stream) / cudaMemcpyAsync(stream)
 *     → libcuda → stream_manager::push(op)    [호스트 쓰레드]
 *         → m_streams[N]->push(op) 또는 m_stream_zero.push(op)
 *   gpgpu_sim_thread_concurrent()              [시뮬레이션 쓰레드]
 *     → stream_manager::operation(&sim_cycles)
 *         → stream_manager::front()           — 실행할 작업 선택 (라운드로빈)
 *         → stream_operation::do_operation()  — 실제 커널 launch 또는 memcpy 수행
 *             ├── gpu->launch(kernel)          — 타이밍 시뮬레이션 커널 등록
 *             ├── gpu->memcpy_to_gpu()         — 호스트→디바이스 메모리 복사
 *             └── gpgpu_cuda_ptx_sim_main_func() — 기능 시뮬레이션 실행
 * 실행 컨텍스트: 호스트 쓰레드(push)와 시뮬레이션 쓰레드(operation/front)가 동시에 접근.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 헤더:
 *   - pthread.h               : pthread_mutex_t (CUstream_st::m_lock, stream_manager::m_lock)
 *   - time.h                  : time_t (CUevent_st::m_wallclock)
 *   - list                    : std::list (m_operations, m_streams)
 *   - abstract_hardware_model.h : kernel_info_t (커널 메타데이터)
 * 이 파일에 의존하는 모듈:
 *   - gpgpusim_entrypoint.cc  : stream_manager 생성 및 시뮬레이션 루프에서 사용
 *   - stream_manager.cc       : 이 헤더의 모든 클래스 구현
 *   - libcuda/cuda_runtime_api.cc : cudaMemcpyAsync, cuLaunchKernel 등 API 처리
 * 공유 자료구조:
 *   - gpgpu_sim* (gpu-sim.h): stream_operation::do_operation()이 GPU에 직접 명령
 *   - kernel_info_t (abstract_hardware_model.h): 커널 실행 메타데이터
 *
 * === 주요 함수/구조체 요약 ===
 * - CUevent_st           : CUDA 이벤트 — 타임스탬프 기록, 스트림 간 동기화 장벽
 * - stream_operation_type: 스트림 작업 종류 enum (kernel_launch, memcpy_*, event, wait_event)
 * - stream_operation      : 단일 스트림 작업 — 오버로드 생성자로 7가지 작업 타입 지원
 * - CUstream_st           : 단일 CUDA 스트림 — push/next/record_next_done으로 큐 조작
 * - stream_manager        : 모든 스트림 통합 관리 — 라운드로빈 스케줄링, 커널 완료 처리
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

#ifndef STREAM_MANAGER_H_INCLUDED  /* [한국어] 인클루드 가드 시작 */
#define STREAM_MANAGER_H_INCLUDED

#include <pthread.h>    /* [한국어] POSIX 뮤텍스: CUstream_st::m_lock, stream_manager::m_lock 타입 */
#include <time.h>       /* [한국어] time_t: CUevent_st::m_wallclock의 wall-clock 타임스탬프 타입 */
#include <list>         /* [한국어] std::list: 스트림 작업 큐(m_operations)와 스트림 목록(m_streams) 컨테이너 */
#include "abstract_hardware_model.h"  /* [한국어] kernel_info_t(커널 실행 메타데이터) 정의 포함 */

// class stream_barrier {
// public:
//    stream_barrier() { m_pending_streams=0; }
//    void inc() { m_pending_streams++; }
//    void dec() { assert(m_pending_streams); m_pending_streams--; }
//    unsigned value() const { return m_pending_streams; }
// private:
//    unsigned m_pending_streams;
//};
/* [한국어] stream_barrier 클래스 (비활성화된 코드).
 * 여러 스트림이 공통 장벽(barrier)을 기다리는 동기화 메커니즘을 구현하려 했으나
 * 현재는 사용하지 않는다. CUevent_st + stream_wait_event로 유사 기능을 구현한다. */

/*
 * [한국어]
 * CUevent_st - CUDA 이벤트 객체 (cudaEvent_t에 대응하는 시뮬레이터 구현)
 *
 * CUDA 이벤트는 두 가지 용도로 사용된다:
 *   1) 타임스탬프: cudaEventRecord() 후 cudaEventElapsedTime()으로 GPU 실행 시간 측정
 *   2) 스트림 간 동기화: cudaStreamWaitEvent()로 "스트림 A가 이벤트 E를 지나쳐야
 *      스트림 B의 다음 작업이 시작된다"는 의존성 표현
 * 시뮬레이터에서는 실제 GPU 하드웨어 타이머 없이 gpu_tot_sim_cycle로 시간을 추적한다.
 *
 * 설계: m_issued는 이 이벤트가 몇 번 스트림에 등록(issue)되었는지 카운트하고,
 * m_updates는 실제로 GPU가 이 이벤트 지점을 통과한 횟수이다.
 * done() = (m_updates == m_issued)이면 이벤트가 완료된 것이다.
 */
struct CUevent_st {
 public:
  /*
   * [한국어]
   * CUevent_st 생성자 - CUDA 이벤트 초기화
   *
   * @blocking: true이면 cudaEventBlockingSync 플래그 — 이벤트 완료 대기 시 CPU spin 대신 sleep.
   *            false이면 기본 polling 모드.
   *
   * 전역 카운터 m_next_event_uid를 증가시켜 이 이벤트에 고유 ID를 부여한다.
   * 모든 카운터를 0으로 초기화하고 m_done = false로 설정한다.
   *
   * 호출 체인:
   *   libcuda의 cudaEventCreate() → [CUevent_st(blocking)]
   */
  CUevent_st(bool blocking) {
    m_uid = ++m_next_event_uid;
    /* [한국어] 전역 카운터를 선증가(pre-increment)하여 고유 ID 부여.
     * m_next_event_uid는 정적 멤버로 모든 이벤트 인스턴스가 공유한다. */
    m_blocking = blocking;    /* [한국어] blocking 모드 플래그 저장 */
    m_updates = 0;            /* [한국어] GPU가 이 이벤트 지점을 통과한 횟수 초기화 */
    m_wallclock = 0;          /* [한국어] 이벤트 기록 시점의 wall-clock 시간 초기화 */
    m_gpu_tot_sim_cycle = 0;  /* [한국어] 이벤트 기록 시점의 누적 시뮬레이션 사이클 초기화 */
    m_issued = 0;             /* [한국어] 이 이벤트가 스트림에 등록된 횟수 초기화 */
    m_done = false;           /* [한국어] 아직 GPU가 이 이벤트를 처리하지 않음 */
  }

  /*
   * [한국어]
   * update - GPU가 이 이벤트 지점을 통과했을 때 타임스탬프 기록
   *
   * @cycle: 현재 누적 시뮬레이션 사이클 수 (gpu_tot_sim_cycle).
   * @clk: 현재 wall-clock 시간 (time(NULL)).
   *
   * stream_operation::do_operation()의 stream_event 케이스에서 호출된다.
   * m_updates를 증가시키고 타임스탬프를 기록하여 이벤트를 "완료" 상태로 전환한다.
   *
   * 호출 체인:
   *   stream_operation::do_operation() [stream_event 케이스] → [update]
   */
  void update(double cycle, time_t clk) {
    m_updates++;              /* [한국어] GPU가 이 이벤트를 처리한 횟수 증가 */
    m_wallclock = clk;        /* [한국어] wall-clock 타임스탬프 기록 */
    m_gpu_tot_sim_cycle = cycle;  /* [한국어] 시뮬레이션 사이클 타임스탬프 기록 */
    m_done = true;            /* [한국어] 이벤트 처리 완료 표시 */
  }

  // void set_done() { assert(!m_done); m_done=true; }
  /* [한국어] set_done()은 이전 버전의 완료 처리 함수이나 현재 update()로 대체됨 */

  /*
   * [한국어]
   * get_uid - 이벤트 고유 식별자 반환
   * @return: 생성 순서로 부여된 양수 정수 ID.
   */
  int get_uid() const { return m_uid; }

  /*
   * [한국어]
   * num_updates - GPU가 이 이벤트를 처리한 횟수 반환
   * @return: update() 호출 횟수. 0이면 아직 GPU가 처리하지 않음.
   */
  unsigned num_updates() const { return m_updates; }

  /*
   * [한국어]
   * done - 이벤트 완료 여부 확인
   * @return: m_updates == m_issued이면 true (등록된 횟수만큼 GPU가 처리 완료).
   *
   * stream_wait_event 처리 시 이 이벤트가 완료되었는지 확인하는 데 사용된다.
   */
  bool done() const { return m_updates == m_issued; }

  /*
   * [한국어]
   * clock - 이벤트 기록 시점의 wall-clock 시간 반환
   * @return: update()에서 기록된 time_t 값.
   */
  time_t clock() const { return m_wallclock; }

  /*
   * [한국어]
   * issue - 이 이벤트를 스트림에 등록할 때 카운터 증가
   *
   * cudaStreamWaitEvent()나 cudaEventRecord() 호출 시 호출된다.
   * done() 조건이 m_updates == m_issued이므로,
   * issue()를 호출한 만큼 update()가 완료되어야 done이 된다.
   */
  void issue() { m_issued++; }

  /*
   * [한국어]
   * num_issued - 이 이벤트가 스트림에 등록된 횟수 반환
   * @return: issue() 호출 횟수.
   * stream_wait_event 생성자에서 현재 issued 수를 스냅샷으로 저장하는 데 사용된다.
   */
  unsigned int num_issued() const { return m_issued; }

 private:
  int m_uid;
  /* [한국어] 이벤트 고유 식별자.
   * 설정자: 생성자에서 ++m_next_event_uid로 할당.
   * 읽는 자: get_uid()로 외부 코드가 이벤트를 식별할 때 사용.
   * 값 범위: 1 이상의 양수 정수.
   * 동기화: 생성 시 한 번만 쓰고 이후 읽기 전용 — 별도 잠금 불필요. */

  bool m_blocking;
  /* [한국어] blocking sync 모드 플래그.
   * true이면 cudaEventSynchronize()가 CPU를 sleep하며 대기 (OS 스케줄링 양보).
   * false이면 spin-wait (CPU를 계속 점유하며 대기).
   * 설정자: 생성자에서 한 번 설정 후 불변.
   * 현재 코드에서는 이 플래그를 읽어 동작을 변경하는 부분이 없어 미구현 상태. */

  bool m_done;
  /* [한국어] 이벤트가 GPU에서 처리 완료되었는지 나타내는 단순 플래그.
   * update()에서 true로 설정되며, done() 메서드의 m_updates == m_issued 조건의
   * 보조 플래그로 사용된다. 현재는 done()이 주로 쓰이고 이 필드는 직접 사용 빈도 낮음. */

  unsigned int m_updates;
  /* [한국어] GPU가 이 이벤트 지점을 통과하여 update()를 호출한 횟수.
   * 설정자: update()에서 증가.
   * done() 조건: m_updates == m_issued. */

  unsigned int m_issued;
  /* [한국어] 이 이벤트가 스트림에 등록(issue())된 총 횟수.
   * 설정자: issue()에서 증가.
   * stream_operation(stream_wait_event 생성자)에서 현재 m_issued 값을 스냅샷으로
   * 저장하여 해당 시점 이후의 update 완료를 감지하는 데 사용. */

  time_t m_wallclock;
  /* [한국어] update() 시점의 wall-clock Unix timestamp.
   * 설정자: update(cycle, clk)에서 clk 값 저장.
   * 읽는 자: clock() 메서드로 경과 시간 계산에 사용.
   * 값 범위: 0(미기록) / 양수 Unix timestamp. */

  double m_gpu_tot_sim_cycle;
  /* [한국어] update() 시점의 누적 시뮬레이션 사이클 수.
   * 설정자: update(cycle, clk)에서 cycle 값 저장.
   * 읽는 자: 현재 코드에서는 직접 접근하는 getter가 없으나,
   *          향후 GPU 실행 시간 측정(두 이벤트 간 사이클 차이)에 활용 가능. */

  static int m_next_event_uid;
  /* [한국어] 다음에 생성될 이벤트에 부여할 UID 카운터 (정적 클래스 변수).
   * 모든 CUevent_st 인스턴스가 공유하며, 생성자에서 선증가 후 m_uid에 할당한다.
   * 설정자: 각 CUevent_st 생성자에서 ++m_next_event_uid로 증가.
   * 초기값: stream_manager.cc에서 int CUevent_st::m_next_event_uid = 0으로 정의.
   * 동기화: 멀티쓰레드 환경에서 동시 이벤트 생성은 현재 libcuda에서 단일 쓰레드
   *         순차 호출로 처리되어 별도 잠금 없음. */
};

/*
 * [한국어]
 * stream_operation_type - CUDA 스트림 작업의 종류를 나타내는 열거형
 *
 * CUDA에서 스트림에 넣을 수 있는 작업의 7가지 카테고리를 정의한다.
 * stream_operation 클래스가 내부에서 어떤 작업인지를 이 타입으로 구분한다.
 */
enum stream_operation_type {
  stream_no_op,                   /* [한국어] 빈 작업 (default 생성자에서 사용) */
  stream_memcpy_host_to_device,   /* [한국어] 호스트→디바이스 메모리 복사 (cudaMemcpyAsync H2D) */
  stream_memcpy_device_to_host,   /* [한국어] 디바이스→호스트 메모리 복사 (cudaMemcpyAsync D2H) */
  stream_memcpy_device_to_device, /* [한국어] 디바이스→디바이스 메모리 복사 (cudaMemcpyAsync D2D) */
  stream_memcpy_to_symbol,        /* [한국어] 호스트 데이터→GPU 상수 메모리 심볼 복사 (cudaMemcpyToSymbol) */
  stream_memcpy_from_symbol,      /* [한국어] GPU 상수 메모리 심볼→호스트 복사 (cudaMemcpyFromSymbol) */
  stream_kernel_launch,           /* [한국어] GPU 커널 실행 요청 (cuLaunchKernel) */
  stream_event,                   /* [한국어] CUDA 이벤트 기록 (cudaEventRecord) */
  stream_wait_event               /* [한국어] 스트림이 이벤트 완료를 기다리는 장벽 (cudaStreamWaitEvent) */
};

/*
 * [한국어]
 * stream_operation - 단일 CUDA 스트림 작업을 나타내는 클래스
 *
 * 하나의 CUDA 비동기 API 호출(memcpy, kernel launch, event 등)에 대응하는
 * 작업 레코드이다. stream_manager와 CUstream_st가 이 객체들을 큐에 보관한다.
 * 작업 타입에 따라 서로 다른 오버로드 생성자로 초기화되며,
 * do_operation()이 실제 시뮬레이터에 작업을 전달한다.
 *
 * 동시성: 이 객체 자체는 immutable(생성 후 내부 데이터 변경 없음)이므로
 * 별도 잠금 없이 큐에서 꺼내어 처리할 수 있다.
 */
class stream_operation {
 public:
  /*
   * [한국어]
   * 기본 생성자 - 빈(no-op) 작업 생성
   *
   * stream_manager::front()가 처리할 작업이 없을 때 반환하는 기본 값으로 사용된다.
   * m_type = stream_no_op, m_done = true로 초기화된다.
   */
  stream_operation() {
    m_kernel = NULL;          /* [한국어] 커널 포인터 없음 */
    m_type = stream_no_op;    /* [한국어] 빈 작업 타입 */
    m_stream = NULL;          /* [한국어] 소속 스트림 없음 */
    m_done = true;            /* [한국어] 빈 작업은 즉시 완료 상태 */
  }

  /*
   * [한국어]
   * 심볼에 메모리 복사(host→constant memory) 생성자
   *
   * @src: 복사할 소스 호스트 메모리 포인터
   * @symbol: 대상 GPU 상수 메모리 심볼 이름 문자열
   * @count: 복사할 바이트 수
   * @offset: 심볼 내 오프셋 (바이트)
   * @stream: 이 작업이 속하는 CUDA 스트림
   *
   * cudaMemcpyToSymbol()에 대응한다. GPU 상수 메모리(constant memory)에
   * 호스트 데이터를 복사할 때 사용된다.
   */
  stream_operation(const void *src, const char *symbol, size_t count,
                   size_t offset, struct CUstream_st *stream) {
    m_kernel = NULL;                 /* [한국어] 커널 포인터 없음 (memcpy 작업) */
    m_stream = stream;               /* [한국어] 소속 스트림 저장 */
    m_type = stream_memcpy_to_symbol;  /* [한국어] 심볼 복사 작업 타입 설정 */
    m_host_address_src = src;        /* [한국어] 소스 호스트 메모리 주소 */
    m_symbol = symbol;               /* [한국어] 대상 심볼 이름 문자열 */
    m_cnt = count;                   /* [한국어] 복사 바이트 수 */
    m_offset = offset;               /* [한국어] 심볼 내 오프셋 */
    m_done = false;                  /* [한국어] 아직 실행되지 않음 */
  }

  /*
   * [한국어]
   * 심볼에서 메모리 복사(constant memory→host) 생성자
   *
   * @symbol: 소스 GPU 상수 메모리 심볼 이름
   * @dst: 복사 대상 호스트 메모리 포인터
   * @count: 복사할 바이트 수
   * @offset: 심볼 내 오프셋 (바이트)
   * @stream: 소속 CUDA 스트림
   *
   * cudaMemcpyFromSymbol()에 대응한다.
   */
  stream_operation(const char *symbol, void *dst, size_t count, size_t offset,
                   struct CUstream_st *stream) {
    m_kernel = NULL;                   /* [한국어] 커널 포인터 없음 */
    m_stream = stream;                 /* [한국어] 소속 스트림 저장 */
    m_type = stream_memcpy_from_symbol;  /* [한국어] 심볼 읽기 타입 설정 */
    m_host_address_dst = dst;          /* [한국어] 대상 호스트 메모리 주소 */
    m_symbol = symbol;                 /* [한국어] 소스 심볼 이름 */
    m_cnt = count;                     /* [한국어] 복사 바이트 수 */
    m_offset = offset;                 /* [한국어] 심볼 내 오프셋 */
    m_done = false;                    /* [한국어] 아직 실행되지 않음 */
  }

  /*
   * [한국어]
   * 커널 실행 작업 생성자
   *
   * @kernel: 실행할 커널의 메타데이터(kernel_info_t) 포인터.
   *          그리드 차원, 블록 차원, 공유 메모리 크기, 인수 등이 포함된다.
   * @sim_mode: true이면 기능 시뮬레이션(PTX semantic), false이면 타이밍 시뮬레이션.
   * @stream: 소속 CUDA 스트림.
   *
   * cuLaunchKernel() 호출에 대응한다.
   */
  stream_operation(kernel_info_t *kernel, bool sim_mode,
                   struct CUstream_st *stream) {
    m_type = stream_kernel_launch;  /* [한국어] 커널 실행 타입 설정 */
    m_kernel = kernel;              /* [한국어] 커널 메타데이터 포인터 저장 */
    m_sim_mode = sim_mode;          /* [한국어] 기능/타이밍 시뮬레이션 모드 */
    m_stream = stream;              /* [한국어] 소속 스트림 */
    m_done = false;                 /* [한국어] 아직 실행되지 않음 */
  }

  /*
   * [한국어]
   * 이벤트 기록 작업 생성자
   *
   * @e: 기록할 CUevent_st 객체 포인터.
   * @stream: 소속 CUDA 스트림.
   *
   * cudaEventRecord(event, stream)에 대응한다.
   * GPU가 이 작업을 처리할 때 e->update()를 호출하여 타임스탬프를 기록한다.
   */
  stream_operation(struct CUevent_st *e, struct CUstream_st *stream) {
    m_kernel = NULL;         /* [한국어] 커널 없음 */
    m_type = stream_event;   /* [한국어] 이벤트 기록 타입 */
    m_event = e;             /* [한국어] 기록할 이벤트 객체 포인터 */
    m_stream = stream;       /* [한국어] 소속 스트림 */
    m_done = false;          /* [한국어] 아직 처리되지 않음 */
  }

  /*
   * [한국어]
   * 스트림 이벤트 대기 작업 생성자
   *
   * @stream: 대기할 스트림 (이 스트림의 다음 작업이 이벤트 완료 후 실행됨)
   * @e: 기다릴 이벤트
   * @flags: cudaStreamWaitEvent() flags (현재 0만 지원)
   *
   * cudaStreamWaitEvent(stream, event, flags)에 대응한다.
   * m_cnt = e->num_issued()로 현재 시점의 이벤트 issue 수를 스냅샷으로 저장하여
   * 이 생성자 호출 이후의 update()가 완료되었는지 감지한다.
   */
  stream_operation(struct CUstream_st *stream, class CUevent_st *e,
                   unsigned int flags) {
    m_kernel = NULL;              /* [한국어] 커널 없음 */
    m_type = stream_wait_event;   /* [한국어] 이벤트 대기 타입 */
    m_event = e;                  /* [한국어] 대기할 이벤트 포인터 */
    m_cnt = m_event->num_issued();  /* [한국어] 이 시점의 issue 수 스냅샷 — 이후 update를 감지하기 위함 */
    m_stream = stream;            /* [한국어] 소속 스트림 */
    m_done = false;               /* [한국어] 아직 완료되지 않음 */
  }

  /*
   * [한국어]
   * 호스트→디바이스 메모리 복사 작업 생성자
   *
   * @host_address_src: 소스 호스트 메모리 포인터
   * @device_address_dst: 대상 GPU 디바이스 메모리 주소 (시뮬레이터 가상 주소)
   * @cnt: 복사할 바이트 수
   * @stream: 소속 CUDA 스트림
   *
   * cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream)에 대응한다.
   */
  stream_operation(const void *host_address_src, size_t device_address_dst,
                   size_t cnt, struct CUstream_st *stream) {
    m_kernel = NULL;                            /* [한국어] 커널 없음 */
    m_type = stream_memcpy_host_to_device;      /* [한국어] H2D 복사 타입 */
    m_host_address_src = host_address_src;      /* [한국어] 소스 호스트 주소 */
    m_device_address_dst = device_address_dst;  /* [한국어] 대상 디바이스 주소 */
    m_host_address_dst = NULL;                  /* [한국어] 호스트 대상 주소 없음 (H2D이므로) */
    m_device_address_src = 0;                   /* [한국어] 디바이스 소스 주소 없음 (H2D이므로) */
    m_cnt = cnt;                                /* [한국어] 복사 바이트 수 */
    m_stream = stream;                          /* [한국어] 소속 스트림 */
    m_sim_mode = false;                         /* [한국어] memcpy는 기능/타이밍 구분 없음 */
    m_done = false;                             /* [한국어] 아직 실행되지 않음 */
  }

  /*
   * [한국어]
   * 디바이스→호스트 메모리 복사 작업 생성자
   *
   * @device_address_src: 소스 GPU 디바이스 메모리 주소
   * @host_address_dst: 대상 호스트 메모리 포인터
   * @cnt: 복사할 바이트 수
   * @stream: 소속 CUDA 스트림
   *
   * cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream)에 대응한다.
   */
  stream_operation(size_t device_address_src, void *host_address_dst,
                   size_t cnt, struct CUstream_st *stream) {
    m_kernel = NULL;                            /* [한국어] 커널 없음 */
    m_type = stream_memcpy_device_to_host;      /* [한국어] D2H 복사 타입 */
    m_device_address_src = device_address_src;  /* [한국어] 소스 디바이스 주소 */
    m_host_address_dst = host_address_dst;      /* [한국어] 대상 호스트 주소 */
    m_device_address_dst = 0;                   /* [한국어] 디바이스 대상 없음 (D2H이므로) */
    m_host_address_src = NULL;                  /* [한국어] 호스트 소스 없음 (D2H이므로) */
    m_cnt = cnt;                                /* [한국어] 복사 바이트 수 */
    m_stream = stream;                          /* [한국어] 소속 스트림 */
    m_sim_mode = false;                         /* [한국어] memcpy는 기능/타이밍 구분 없음 */
    m_done = false;                             /* [한국어] 아직 실행되지 않음 */
  }

  /*
   * [한국어]
   * 디바이스→디바이스 메모리 복사 작업 생성자
   *
   * @device_address_src: 소스 GPU 디바이스 메모리 주소
   * @device_address_dst: 대상 GPU 디바이스 메모리 주소
   * @cnt: 복사할 바이트 수
   * @stream: 소속 CUDA 스트림
   *
   * cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream)에 대응한다.
   */
  stream_operation(size_t device_address_src, size_t device_address_dst,
                   size_t cnt, struct CUstream_st *stream) {
    m_kernel = NULL;                            /* [한국어] 커널 없음 */
    m_type = stream_memcpy_device_to_device;    /* [한국어] D2D 복사 타입 */
    m_device_address_src = device_address_src;  /* [한국어] 소스 디바이스 주소 */
    m_device_address_dst = device_address_dst;  /* [한국어] 대상 디바이스 주소 */
    m_host_address_src = NULL;                  /* [한국어] 호스트 소스 없음 */
    m_host_address_dst = NULL;                  /* [한국어] 호스트 대상 없음 */
    m_cnt = cnt;                                /* [한국어] 복사 바이트 수 */
    m_stream = stream;                          /* [한국어] 소속 스트림 */
    m_sim_mode = false;                         /* [한국어] memcpy는 기능/타이밍 구분 없음 */
    m_done = false;                             /* [한국어] 아직 실행되지 않음 */
  }

  /*
   * [한국어]
   * is_kernel - 이 작업이 커널 실행인지 확인
   * @return: m_type == stream_kernel_launch이면 true.
   *
   * stream_manager::front()와 register_finished_kernel()에서 커널 작업을
   * grid_id_to_stream 맵에 등록/해제할 때 사용한다.
   */
  bool is_kernel() const { return m_type == stream_kernel_launch; }

  /*
   * [한국어]
   * is_mem - 이 작업이 메모리 복사인지 확인
   * @return: H2D 또는 D2H 타입이면 true.
   * 주의: D2D 타입이 누락된 버그가 있음 (두 번 H2D를 검사하고 D2D는 미검사).
   */
  bool is_mem() const {
    return m_type == stream_memcpy_host_to_device ||
           m_type == stream_memcpy_device_to_host ||
           m_type == stream_memcpy_host_to_device;  /* [한국어] 주의: 원본 코드에 D2D 누락 버그 있음 */
  }

  /*
   * [한국어]
   * is_noop - 이 작업이 빈 작업인지 확인
   * @return: m_type == stream_no_op이면 true.
   * stream_manager::front()가 처리할 작업이 없을 때 반환하는 기본값.
   */
  bool is_noop() const { return m_type == stream_no_op; }

  /*
   * [한국어]
   * is_done - 이 작업이 완료되었는지 확인
   * @return: m_done이 true이면 작업 완료.
   * do_operation()이 성공하면 m_done = true로 설정한다.
   */
  bool is_done() const { return m_done; }

  /*
   * [한국어]
   * get_kernel - 이 작업의 커널 정보 포인터 반환
   * @return: kernel_info_t* — is_kernel()이 false이면 NULL일 수 있음.
   *
   * stream_manager에서 커널 완료 시 kernel->get_uid()로 grid_id_to_stream 맵에서
   * 제거하는 데 사용한다.
   */
  kernel_info_t *get_kernel() { return m_kernel; }

  /*
   * [한국어]
   * do_operation - 이 스트림 작업을 실제로 실행
   *
   * @gpu: 시뮬레이터 GPU 객체. memcpy/launch 등을 실제로 수행한다.
   * @return: true이면 작업 완료(m_done = true 설정 포함),
   *          false이면 아직 준비되지 않음 (예: 커널 launch latency 대기, 이벤트 미완료).
   *
   * stream_manager::operation()에서 호출된다.
   * 작업 타입에 따라 switch 분기:
   *   - H2D/D2H/D2D: gpu->memcpy_*() 호출 후 stream->record_next_done()
   *   - to_symbol/from_symbol: gpgpu_ptx_sim_memcpy_symbol() 호출
   *   - kernel_launch: m_sim_mode에 따라 functional_launch() 또는 launch()
   *   - event: m_event->update() 후 record_next_done()
   *   - wait_event: 이벤트 update 횟수가 m_cnt에 도달하면 record_next_done()
   *
   * 호출 체인:
   *   stream_manager::operation() → [do_operation] → gpu->launch/memcpy/...
   */
  bool do_operation(gpgpu_sim *gpu);

  /*
   * [한국어]
   * print - 이 작업의 타입을 파일 스트림에 출력
   *
   * @fp: 출력 대상 FILE 포인터 (보통 stdout).
   * 디버그 출력 및 stream_manager::print_impl()에서 사용된다.
   */
  void print(FILE *fp) const;

  /*
   * [한국어]
   * get_stream - 이 작업이 속한 스트림 포인터 반환
   * @return: CUstream_st* — NULL이면 스트림 0(default stream)에 속함.
   */
  struct CUstream_st *get_stream() {
    return m_stream;
  }

  /*
   * [한국어]
   * set_stream - 이 작업의 소속 스트림을 변경
   *
   * @stream: 새로운 소속 스트림 포인터.
   * stream_manager::push()에서 blocking 모드일 때 m_stream_zero로 스트림을 교체할 때 사용.
   */
  void set_stream(CUstream_st *stream) { m_stream = stream; }

 private:
  struct CUstream_st *m_stream;
  /* [한국어] 이 작업이 속한 CUDA 스트림 포인터.
   * 설정자: 각 생성자에서 설정. set_stream()으로 나중에 변경 가능.
   * 읽는 자: do_operation()에서 stream->record_next_done() 호출 시,
   *          stream_manager::front()에서 grid_id_to_stream 맵에 스트림 등록 시.
   * 값 범위: NULL(기본 스트림 = stream 0) / 유효한 CUstream_st 포인터.
   * 동기화: 작업이 큐에 있는 동안은 CUstream_st::m_lock으로 큐 전체가 보호됨. */

  bool m_done;
  /* [한국어] 작업 완료 여부.
   * 설정자: 기본 생성자에서 true, 나머지 생성자에서 false로 초기화.
   *         do_operation() 성공 시 true로 설정.
   * 읽는 자: is_done()으로 외부 코드가 확인.
   * 동기화: do_operation()은 시뮬레이션 쓰레드에서만 호출되므로 별도 잠금 불필요. */

  stream_operation_type m_type;
  /* [한국어] 이 작업의 종류 (stream_operation_type enum 값).
   * 설정자: 각 생성자에서 작업 타입에 맞게 설정.
   * 읽는 자: do_operation()의 switch, is_kernel(), is_mem(), is_noop(). */

  size_t m_device_address_dst;  /* [한국어] 디바이스 대상 주소 (H2D, D2D 복사에서 사용) */
  size_t m_device_address_src;  /* [한국어] 디바이스 소스 주소 (D2H, D2D 복사에서 사용) */
  void *m_host_address_dst;     /* [한국어] 호스트 대상 주소 (D2H, from_symbol에서 사용) */
  const void *m_host_address_src;  /* [한국어] 호스트 소스 주소 (H2D, to_symbol에서 사용) */
  size_t m_cnt;
  /* [한국어] 복사할 바이트 수 또는 stream_wait_event에서 이벤트 issue 스냅샷.
   * memcpy 계열: 복사 크기(바이트).
   * stream_wait_event: 이 작업 생성 시점의 m_event->num_issued() 값.
   *   do_operation()에서 m_event->num_updates() >= m_cnt이면 이벤트 완료로 판단. */

  const char *m_symbol;         /* [한국어] cudaMemcpyToSymbol/FromSymbol에서 사용하는 심볼 이름 문자열 */
  size_t m_offset;              /* [한국어] 심볼 내 시작 오프셋 (바이트, to_symbol/from_symbol에서 사용) */

  bool m_sim_mode;
  /* [한국어] 커널 실행 시뮬레이션 모드 플래그.
   * true: 기능 시뮬레이션 (PTX 명령어 의미론적 실행, 타이밍 없음)
   *       → gpu->functional_launch(m_kernel) 호출
   * false: 타이밍 시뮬레이션 (사이클-레벨 마이크로아키텍처 모델)
   *        → gpu->launch(m_kernel) 호출
   * 설정자: 커널 실행 생성자에서 설정. memcpy 생성자에서는 false로 초기화(미사용). */

  kernel_info_t *m_kernel;
  /* [한국어] 커널 실행 작업에서 사용하는 커널 메타데이터 포인터.
   * 설정자: 커널 실행 생성자에서 설정. 나머지 생성자에서는 NULL.
   * 읽는 자: do_operation()의 stream_kernel_launch 케이스에서 gpu->launch(m_kernel).
   *          get_kernel()을 통해 stream_manager가 커널 완료 시 정리에 사용.
   * 값 범위: NULL(비커널 작업) / 유효한 kernel_info_t 포인터.
   * 동기화: 커널이 GPU에서 실행 완료되면 register_finished_kernel()이 delete kernel을 호출.
   *         이후 이 포인터는 dangling — 커널 완료 후 접근하면 안 된다. */

  struct CUevent_st *m_event;
  /* [한국어] 이벤트 기록(stream_event) 또는 이벤트 대기(stream_wait_event)에서 사용하는 이벤트 포인터.
   * 설정자: 이벤트 관련 생성자에서 설정. 나머지 생성자에서는 미초기화.
   * 읽는 자: do_operation()의 stream_event/stream_wait_event 케이스. */
};

/*
 * [한국어]
 * CUstream_st - 단일 CUDA 스트림 큐 (cudaStream_t에 대응하는 시뮬레이터 구현)
 *
 * CUDA 스트림은 GPU에서 작업들이 순서대로 실행되도록 보장하는 순서 큐이다.
 * 같은 스트림의 작업들은 항상 FIFO 순서로 실행되고,
 * 서로 다른 스트림의 작업들은 가능하면 동시에 실행된다.
 *
 * 내부적으로 stream_operation 객체들의 std::list를 보관하며,
 * m_lock으로 호스트 쓰레드(push)와 시뮬레이션 쓰레드(next/record_next_done)의
 * 동시 접근을 보호한다.
 *
 * m_pending 플래그: front() 작업이 시작되었지만 아직 완료되지 않음을 표시.
 * m_pending == true이면 새 작업을 꺼낼 수 없다(한 번에 하나씩만 실행).
 */
struct CUstream_st {
 public:
  /*
   * [한국어]
   * CUstream_st 생성자 - 스트림 초기화
   *
   * 스트림 고유 ID 부여, m_pending = false 초기화, m_lock 초기화를 수행한다.
   *
   * 호출 체인:
   *   libcuda의 cudaStreamCreate() → new CUstream_st() → [이 생성자]
   */
  CUstream_st();

  /*
   * [한국어]
   * empty - 스트림 큐가 비어있는지 확인 (잠금 포함)
   *
   * @return: m_operations.empty()이면 true.
   *
   * m_lock을 잡고 안전하게 확인한다. 호스트 쓰레드와 시뮬레이션 쓰레드 모두에서 호출 가능.
   */
  bool empty();

  /*
   * [한국어]
   * busy - 스트림 front 작업이 진행 중인지 확인 (잠금 포함)
   *
   * @return: m_pending이 true이면 busy.
   *
   * stream_manager::front()에서 스트림에서 다음 작업을 꺼낼 수 있는지 판단할 때 사용.
   * m_pending == true이면 이미 꺼낸 작업이 완료 처리 대기 중이므로 새 작업 불가.
   */
  bool busy();

  /*
   * [한국어]
   * synchronize - 이 스트림의 모든 작업 완료를 바쁜 대기(spin-wait)로 기다림
   *
   * m_operations.empty()가 될 때까지 폴링한다.
   * 호스트 쓰레드에서 cudaStreamSynchronize() 호출 시 사용된다.
   * 실행 컨텍스트: 호스트 쓰레드.
   */
  void synchronize();

  /*
   * [한국어]
   * push - 스트림 큐 끝에 새 작업 추가
   *
   * @op: 추가할 stream_operation 객체.
   *
   * m_lock을 잡고 m_operations.push_back(op)으로 큐 뒤에 추가한다.
   * 실행 컨텍스트: 호스트 쓰레드 (libcuda에서 비동기 API 호출 시).
   *
   * 호출 체인:
   *   stream_manager::push() → CUstream_st::push() → [이 함수]
   */
  void push(const stream_operation &op);

  /*
   * [한국어]
   * record_next_done - front 작업 완료 처리 (pop + m_pending 해제)
   *
   * m_lock을 잡고 m_operations.pop_front()로 완료된 작업을 제거하고
   * m_pending = false로 설정한다.
   * 실행 컨텍스트: 시뮬레이션 쓰레드 (do_operation 성공 후).
   *
   * 호출 체인:
   *   stream_operation::do_operation() → stream->record_next_done()
   */
  void record_next_done();

  /*
   * [한국어]
   * next - front 작업을 "처리 중" 상태로 마킹하고 반환
   *
   * @return: m_operations.front()의 복사본.
   *
   * m_pending = true로 설정하여 "이 작업이 진행 중"임을 표시한다.
   * record_next_done()이 호출되기 전까지 busy()는 true를 반환한다.
   * 실행 컨텍스트: 시뮬레이션 쓰레드.
   *
   * 호출 체인:
   *   stream_manager::front() → CUstream_st::next()
   */
  stream_operation next();

  /*
   * [한국어]
   * cancel_front - front 작업의 pending 상태 취소
   *
   * do_operation()이 false를 반환하여 실행을 미룰 때 사용한다.
   * m_pending = false로 되돌려 다음 사이클에 다시 시도할 수 있게 한다.
   * 실행 컨텍스트: 시뮬레이션 쓰레드.
   *
   * 호출 체인:
   *   stream_manager::operation() [do_operation 실패 시] → op.get_stream()->cancel_front()
   */
  void cancel_front();  // front operation fails, cancle the pending status

  /*
   * [한국어]
   * front - 큐의 첫 번째 작업 참조 반환 (잠금 없음)
   *
   * @return: m_operations.front()에 대한 참조.
   * 주의: 호출자가 m_lock을 잡은 상태에서 사용해야 한다. 이 함수 자체는 잠금하지 않음.
   */
  stream_operation &front() { return m_operations.front(); }

  /*
   * [한국어]
   * print - 이 스트림의 현재 상태를 출력
   *
   * @fp: 출력 대상 FILE 포인터.
   * m_lock을 잡고 스트림 ID와 모든 pending 작업 목록을 출력한다.
   */
  void print(FILE *fp);

  /*
   * [한국어]
   * get_uid - 스트림 고유 ID 반환
   * @return: 생성 순서로 부여된 스트림 ID.
   */
  unsigned get_uid() const { return m_uid; }

 private:
  unsigned m_uid;
  /* [한국어] 스트림 고유 식별자.
   * 설정자: 생성자에서 sm_next_stream_uid++로 할당.
   * 읽는 자: get_uid()로 외부 코드, print() 출력에서 사용.
   * 값 범위: 0 이상의 정수 (최초 생성된 스트림이 0).
   * 동기화: 생성 후 불변. */

  static unsigned sm_next_stream_uid;
  /* [한국어] 다음 스트림 UID 카운터 (모든 인스턴스 공유 정적 변수).
   * 초기값: stream_manager.cc에서 unsigned CUstream_st::sm_next_stream_uid = 0으로 정의.
   * 설정자: 각 CUstream_st 생성자에서 m_uid = sm_next_stream_uid++.
   * 동기화: libcuda에서 스트림 생성은 단일 호스트 쓰레드에서 순차 호출 가정 — 별도 잠금 없음. */

  std::list<stream_operation> m_operations;
  /* [한국어] 이 스트림에 대기 중인 작업들의 FIFO 큐.
   * 설정자: push()에서 push_back()으로 작업 추가 (호스트 쓰레드).
   * 읽는 자: next()에서 front()로 작업 꺼냄, record_next_done()에서 pop_front() (시뮬레이션 쓰레드).
   * 동기화: 모든 접근은 m_lock을 잡고 수행해야 한다. */

  bool m_pending;
  /* [한국어] front 작업이 시작되었지만 아직 완료(record_next_done)되지 않음.
   * true: next()가 호출된 후 record_next_done() 또는 cancel_front() 전.
   *       busy()가 true를 반환하여 새 작업을 꺼낼 수 없다.
   * false: 현재 처리 중인 작업 없음 — 다음 작업을 꺼낼 수 있음.
   * 설정자: next()에서 true, record_next_done()/cancel_front()에서 false.
   * 동기화: m_lock 보호. */

  pthread_mutex_t m_lock;  // ensure only one host or gpu manipulates stream
                           // operation at one time
  /* [한국어] 스트림 큐(m_operations)와 m_pending에 대한 호스트/시뮬레이션 쓰레드 동시 접근 보호.
   * 설정자: 생성자에서 pthread_mutex_init(&m_lock, NULL)으로 초기화.
   * 읽는 자/쓰는 자: push(), empty(), busy(), next(), record_next_done(), cancel_front(), print().
   * 동기화: 이 뮤텍스를 잡은 채로 stream_manager::m_lock을 잡으면 안 된다 (데드락 주의). */
};

/*
 * [한국어]
 * stream_manager - 모든 CUDA 스트림을 통합 관리하는 최상위 관리자
 *
 * 이 클래스는 시뮬레이터 내의 모든 활성 CUDA 스트림(m_streams)과
 * 스트림 0(m_stream_zero)을 관리한다.
 * 호스트 쓰레드가 push()로 작업을 등록하고,
 * 시뮬레이션 쓰레드가 operation()으로 작업을 꺼내어 실행한다.
 *
 * 핵심 설계:
 *   1) 스트림 0: m_stream_zero (블로킹 또는 기본 스트림)
 *   2) 사용자 스트림: m_streams 리스트 (라운드로빈 스케줄링)
 *   3) 커널 완료 추적: m_grid_id_to_stream 맵으로 어느 스트림의 커널이 완료됐는지 추적
 *   4) m_service_stream_zero 플래그로 스트림 0 우선 서비스 여부 결정
 */
class stream_manager {
 public:
  /*
   * [한국어]
   * stream_manager 생성자 - 스트림 관리자 초기화
   *
   * @gpu: GPU 시뮬레이터 포인터. do_operation()에서 실제 작업을 수행할 때 사용.
   * @cuda_launch_blocking: true이면 blocking 모드 — 모든 커널이 스트림 0으로 직렬화됨.
   *
   * m_lock 초기화, m_last_stream을 m_streams.begin()으로 설정한다.
   *
   * 호출 체인:
   *   gpgpu_ptx_sim_init_perf() → new stream_manager(g_the_gpu, ...) → [이 생성자]
   */
  stream_manager(gpgpu_sim *gpu, bool cuda_launch_blocking);

  /*
   * [한국어]
   * register_finished_kernel - 완료된 커널을 스트림 큐에서 제거
   *
   * @grid_uid: 완료된 커널의 grid UID (gpgpu_sim::finished_kernel()이 반환).
   * @return: true이면 커널이 실제로 완료 처리됨, false이면 grid_uid가 0이거나 아직 미완료.
   *
   * m_grid_id_to_stream 맵에서 grid_uid에 해당하는 스트림을 찾아 front 작업을 pop한다.
   * kernel_info_t::is_finished()가 true일 때만(CDP를 고려한 자식 커널 완료 포함) 처리.
   * 완료 후 kernel->notify_parent_finished()와 delete kernel을 수행한다.
   *
   * 호출 체인:
   *   check_finished_kernel() → [register_finished_kernel]
   *   stop_all_running_kernels() → [register_finished_kernel]
   */
  bool register_finished_kernel(unsigned grid_uid);

  /*
   * [한국어]
   * check_finished_kernel - GPU에서 완료된 커널이 있으면 스트림 처리
   *
   * @return: register_finished_kernel()의 반환값 (커널 완료 처리 여부).
   *
   * gpgpu_sim::finished_kernel()로 완료된 커널 UID를 조회하고
   * register_finished_kernel()에 전달한다.
   * operation()이 매 사이클 첫 번째로 이 함수를 호출한다.
   *
   * 호출 체인:
   *   operation() → [check_finished_kernel] → register_finished_kernel()
   */
  bool check_finished_kernel();

  /*
   * [한국어]
   * front - 다음에 실행할 스트림 작업 선택 및 반환
   *
   * @return: 선택된 stream_operation. 처리할 작업이 없으면 기본 생성자(is_noop() == true) 반환.
   *
   * 스케줄링 정책:
   *   1) m_service_stream_zero == true이고 스트림 0이 비어있지 않고 busy하지 않으면
   *      스트림 0의 next()를 반환.
   *   2) 아니면 m_last_stream에서 시작하여 라운드로빈으로 사용자 스트림을 순회하며
   *      busy하지 않고 비어있지 않은 첫 번째 스트림의 next()를 반환.
   * 선택된 커널 작업은 m_grid_id_to_stream 맵에 등록된다.
   *
   * 실행 컨텍스트: 시뮬레이션 쓰레드. m_lock이 이미 잡힌 상태에서 호출된다.
   *
   * 호출 체인:
   *   operation() → [front] → CUstream_st::next()
   */
  stream_operation front();

  /*
   * [한국어]
   * add_stream - 새 CUDA 스트림을 m_streams 리스트에 추가
   *
   * @stream: 추가할 CUstream_st 포인터.
   *
   * m_lock을 잡고 m_streams.push_back(stream)을 수행한다.
   * 실행 컨텍스트: 호스트 쓰레드 (cudaStreamCreate() 시).
   *
   * 호출 체인:
   *   libcuda::cudaStreamCreate() → [add_stream]
   */
  void add_stream(CUstream_st *stream);

  /*
   * [한국어]
   * destroy_stream - CUDA 스트림 제거 및 메모리 해제
   *
   * @stream: 제거할 CUstream_st 포인터.
   *
   * m_lock을 잡고 스트림이 완전히 비어있을 때까지 대기한 후
   * m_streams에서 제거하고 delete stream을 수행한다.
   * m_last_stream을 m_streams.begin()으로 리셋한다.
   * 실행 컨텍스트: 호스트 쓰레드 (cudaStreamDestroy() 시).
   *
   * 호출 체인:
   *   libcuda::cudaStreamDestroy() → [destroy_stream]
   */
  void destroy_stream(CUstream_st *stream);

  /*
   * [한국어]
   * concurrent_streams_empty - 사용자 생성 스트림(m_streams)이 모두 비어있는지 확인
   *
   * @return: m_streams의 모든 스트림이 empty()이면 true.
   *
   * 잠금 없이 확인하므로 호출자가 m_lock을 잡아야 한다.
   * empty_protected()와 empty()에서 호출된다.
   *
   * 호출 체인:
   *   empty_protected() / empty() → [concurrent_streams_empty]
   */
  bool concurrent_streams_empty();

  /*
   * [한국어]
   * empty_protected - 뮤텍스 보호하에 모든 스트림이 비어있는지 확인
   *
   * @return: m_streams와 m_stream_zero 모두 비어있으면 true.
   *
   * m_lock을 잡고 concurrent_streams_empty()와 m_stream_zero.empty()를 확인한다.
   * gpgpu_sim_thread_concurrent()의 스핀 대기 루프에서 사용된다.
   *
   * 호출 체인:
   *   gpgpu_sim_thread_concurrent() 스핀 루프 → [empty_protected]
   */
  bool empty_protected();

  /*
   * [한국어]
   * empty - 잠금 없이 모든 스트림이 비어있는지 확인
   *
   * @return: 모든 스트림이 비어있으면 true.
   *
   * synchronize()에서 완료 조건 확인에 사용된다.
   * 주의: m_lock 없이 접근하므로 호출자가 적절히 동기화해야 한다.
   *
   * 호출 체인:
   *   synchronize() / synchronize_check() → [empty]
   */
  bool empty();

  /*
   * [한국어]
   * print - 스트림 관리자 현재 상태 출력 (잠금 포함)
   *
   * @fp: 출력 대상 FILE 포인터.
   * m_lock을 잡고 print_impl()을 호출하여 모든 스트림 상태를 출력한다.
   *
   * 호출 체인:
   *   gpgpu_sim_thread_concurrent() 디버그 출력 / synchronize() → [print]
   */
  void print(FILE *fp);

  /*
   * [한국어]
   * push - 새 스트림 작업을 적절한 스트림 큐에 추가
   *
   * @op: 추가할 stream_operation 객체 (값으로 복사됨).
   *
   * 가장 복잡한 함수 중 하나:
   *   1) blocking 모드 또는 stream == NULL이면: 모든 concurrent 스트림이 비어있을 때까지 spin-wait
   *   2) m_lock을 잡고 최대 한도 미도달 시:
   *      - 비-blocking 모드: stream->push(op)로 해당 스트림에 추가
   *      - blocking 모드: op.set_stream(&m_stream_zero) 후 m_stream_zero.push(op)
   *   3) 최대 한도 도달 시: 작업 무시하고 메시지 출력
   *   4) blocking/stream 0 모드이면 작업 완료까지 usleep()으로 대기
   *      (exponential backoff, 최대 100ms)
   *
   * 실행 컨텍스트: 호스트 쓰레드 (libcuda API 처리 중).
   *
   * 호출 체인:
   *   libcuda::cuLaunchKernel / cudaMemcpyAsync → [push]
   *     → CUstream_st::push() 또는 m_stream_zero.push()
   */
  void push(stream_operation op);

  /*
   * [한국어]
   * pushCudaStreamWaitEventToAllStreams - 모든 스트림에 이벤트 대기 작업 추가
   *
   * @e: 대기할 이벤트 포인터.
   * @flags: cudaStreamWaitEvent() flags.
   *
   * cudaStreamWaitEvent(cudaStreamAll, event, flags)에 해당하는 기능.
   * m_streams의 모든 스트림에 stream_wait_event 작업을 push()한다.
   *
   * 호출 체인:
   *   libcuda::cudaStreamWaitEvent(ALL_STREAMS) → [pushCudaStreamWaitEventToAllStreams]
   */
  void pushCudaStreamWaitEventToAllStreams(CUevent_st *e, unsigned int flags);

  /*
   * [한국어]
   * operation - 시뮬레이션 사이클에서 스트림 작업 처리 (핵심 함수)
   *
   * @sim: out 파라미터. 커널이 실행되었으면 true로 설정됨.
   * @return: true이면 커널이 방금 완료됨 (gpgpu_sim_thread_concurrent에서 break 조건).
   *
   * 동작:
   *   1) check_finished_kernel()로 완료된 커널 처리
   *   2) m_lock을 잡고 front()로 다음 작업 선택
   *   3) op.do_operation(m_gpu) 호출
   *   4) do_operation() 실패 시 커널이면 m_grid_id_to_stream에서 제거하고 cancel_front()
   *
   * 실행 컨텍스트: 시뮬레이션 쓰레드 (gpgpu_sim_thread_concurrent의 내부 루프).
   *
   * 호출 체인:
   *   gpgpu_sim_thread_concurrent() → [operation]
   *     → check_finished_kernel() → register_finished_kernel()
   *     → front() → CUstream_st::next()
   *     → stream_operation::do_operation()
   */
  bool operation(bool *sim);

  /*
   * [한국어]
   * stop_all_running_kernels - 모든 실행 중인 커널을 강제 종료
   *
   * 최대 사이클/명령 한도 초과 시 호출된다.
   * m_gpu->get_running_kernels()로 실행 중인 커널 목록을 가져와
   * m_gpu->stop_all_running_kernels()로 GPU에 중단 명령을 내린다.
   * 이후 check_finished_kernel() 루프로 모든 스트림 큐를 정리하고
   * 각 완료된 스트림의 통계를 출력한다.
   *
   * 호출 체인:
   *   gpgpu_sim_thread_concurrent() [한도 초과 시] → [stop_all_running_kernels]
   */
  void stop_all_running_kernels();

  /*
   * [한국어]
   * size - 현재 활성 사용자 스트림 수 반환
   * @return: m_streams.size(). 스트림 0은 포함하지 않는다.
   */
  unsigned size() { return m_streams.size(); };

  /*
   * [한국어]
   * is_blocking - blocking 모드 여부 반환
   * @return: m_cuda_launch_blocking이 true이면 blocking 모드.
   */
  bool is_blocking() { return m_cuda_launch_blocking; };

 private:
  /*
   * [한국어]
   * print_impl - 잠금 없이 스트림 상태를 출력하는 내부 구현
   *
   * @fp: 출력 대상 FILE 포인터.
   * print()가 m_lock을 잡은 후 이 함수를 호출한다.
   * push()에서 g_debug_execution >= 3일 때도 직접 호출된다.
   */
  void print_impl(FILE *fp);

  bool m_cuda_launch_blocking;
  /* [한국어] CUDA launch blocking 모드 플래그.
   * true이면 모든 커널/memcpy가 스트림 0으로 직렬화되어 동기적으로 완료될 때까지 대기.
   * 설정자: 생성자의 cuda_launch_blocking 매개변수.
   * 읽는 자: push()에서 blocking 여부 결정. is_blocking()으로 외부 접근.
   * 값 범위: 설정 파일 -gpgpu_runtime_stat의 g_cuda_launch_blocking 변수에서 초기화. */

  gpgpu_sim *m_gpu;
  /* [한국어] GPU 시뮬레이터 포인터. do_operation()에서 실제 memcpy/launch를 수행하기 위해 필요.
   * 설정자: 생성자에서 설정 후 불변.
   * 읽는 자: operation(), check_finished_kernel(), stop_all_running_kernels()에서 사용.
   * 동기화: 불변 포인터 — 별도 잠금 불필요. */

  std::list<CUstream_st *> m_streams;
  /* [한국어] 사용자가 생성한 모든 CUDA 스트림(스트림 0 제외)의 포인터 목록.
   * 설정자: add_stream()에서 push_back, destroy_stream()에서 erase.
   * 읽는 자: front()에서 라운드로빈 스케줄링, concurrent_streams_empty()에서 비어있는지 확인.
   * 동기화: 모든 접근 시 m_lock을 잡아야 한다. */

  std::map<unsigned, CUstream_st *> m_grid_id_to_stream;
  /* [한국어] 실행 중인 커널의 grid_uid → 소속 스트림 매핑.
   * 커널 launch 시 front()에서 등록되고, register_finished_kernel()에서 커널 완료 시 제거된다.
   * 설정자: front()에서 is_kernel() 작업 처리 시 grid_id 등록.
   *         register_finished_kernel() / operation() [도 operation 취소 시]에서 제거.
   * 읽는 자: register_finished_kernel()에서 완료된 커널의 스트림 찾기.
   * 동기화: m_lock 보호. */

  CUstream_st m_stream_zero;
  /* [한국어] 스트림 0 (default stream) 객체. 포인터가 아닌 값 객체로 직접 포함.
   * CUDA의 "null stream" 또는 "스트림 0" — cudaMemcpy (비동기 아님)나
   * blocking 모드에서 사용되는 기본 스트림.
   * 설정자: add_stream()을 통하지 않고 직접 초기화 (생성자 자동 호출).
   * 읽는 자: front()에서 우선 서비스, push()에서 blocking 모드 시 대상 스트림.
   * 동기화: CUstream_st 내부의 m_lock으로 보호. */

  bool m_service_stream_zero;
  /* [한국어] 다음 사이클에 스트림 0을 서비스할지 여부 플래그.
   * front()에서 스트림 0이 우선 처리되었으면 false로 전환하여
   * 다음 호출에서는 사용자 스트림을 처리한다.
   * 설정자: front() 내부에서 제어.
   * 현재 구현에서는 항상 true로 설정되어 스트림 0이 항상 우선 처리됨 (TODO: 버그?). */

  pthread_mutex_t m_lock;
  /* [한국어] 스트림 목록(m_streams), 스트림 0, m_grid_id_to_stream 등
   * 공유 데이터에 대한 호스트/시뮬레이션 쓰레드 동시 접근 보호 뮤텍스.
   * 설정자: 생성자에서 pthread_mutex_init으로 초기화.
   * 읽는 자/쓰는 자: operation(), push(), add_stream(), destroy_stream(),
   *                  empty_protected(), print().
   * 주의: CUstream_st::m_lock과의 잠금 순서: stream_manager::m_lock → CUstream_st::m_lock
   *       (반대 순서로 잡으면 데드락 위험). */

  std::list<struct CUstream_st *>::iterator m_last_stream;
  /* [한국어] 라운드로빈 스케줄링 상태 유지를 위한 이전 서비스 스트림 반복자.
   * front()에서 다음 사이클에 어느 스트림부터 탐색할지 결정하는 데 사용된다.
   * 설정자: 생성자에서 m_streams.begin()으로 초기화.
   *         destroy_stream()에서 m_streams.begin()으로 리셋.
   *         front()에서 서비스한 스트림의 반복자로 업데이트.
   * 동기화: m_lock 보호 (front()는 m_lock을 잡은 상태에서만 호출됨). */
};

#endif  // STREAM_MANAGER_H_INCLUDED
/* [한국어] 인클루드 가드 종료. 파일 최상단의 #ifndef STREAM_MANAGER_H_INCLUDED와 짝을 이룬다. */
