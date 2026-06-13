/*
 * [한국어 설명] CUDA Dynamic Parallelism(CDP) 디바이스 런타임 에뮬레이션 인터페이스 (cuda_device_runtime.h)
 *
 * === 파일의 역할 ===
 * CUDA Dynamic Parallelism(CDP)을 에뮬레이션하기 위한 자료구조와 인터페이스를 정의한다.
 * CDP는 디바이스 코드(커널)에서 자식 커널을 동적으로 런칭하는 기능이다 (CUDA 5.0+).
 * 이 파일은 자식 커널 런칭 설정(device_launch_config_t), 런칭 연산
 * (device_launch_operation_t), 그리고 런타임 API 에뮬레이션 메서드를 가진
 * cuda_device_runtime 클래스를 정의한다.
 * CUDART_VERSION >= 5000 (CUDA 5.0 이상)인 경우에만 실질적으로 컴파일된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: PTX 기능 시뮬레이션 중 cudaGetParameterBufferV2 호출 →
 *   gpgpusim_cuda_getParameterBufferV2 (파라미터 버퍼 할당)
 *   → 이후 cudaLaunchDeviceV2 호출 → gpgpusim_cuda_launchDeviceV2 (자식 커널 등록)
 *   → launch_all_device_kernels (스트림 매니저에 전달)
 * CDP를 사용하지 않으면(g_cdp_enabled == false) 이 클래스의 메서드는 호출되지 않음.
 * 실행 컨텍스트: 기능 시뮬레이션 (호스트 CPU 단일 스레드).
 *
 * === 타 모듈과의 연결 ===
 * 의존: abstract_hardware_model.h (dim3, kernel_info_t), ptx_ir.h (function_info),
 *   stream_manager.h (stream_operation), gpgpu_context.h
 * 의존받음: cuda-sim/instructions.cc (CDP API 콜 디스패치),
 *   gpgpu-sim/gpu-sim.cc (launch_all_device_kernels 호출)
 * 공유 구조: g_cuda_device_launch_param_map (파라미터 버퍼 → 런칭 설정),
 *   g_cuda_device_launch_op (런칭 대기 큐)
 *
 * === 주요 함수/구조체 요약 ===
 * device_launch_config_t: 자식 커널 런칭 설정 (grid/block 크기, smem, 커널 엔트리)
 * device_launch_operation_t: 런칭 대기 연산 (kernel_info_t + CUstream)
 * cuda_device_runtime: CDP 런타임 API 에뮬레이션 클래스
 * gpgpusim_cuda_getParameterBufferV2(): 자식 커널 파라미터 버퍼 할당 + 설정 저장
 * gpgpusim_cuda_launchDeviceV2(): 자식 커널 생성 및 런칭 큐에 추가
 * launch_all_device_kernels(): 큐의 모든 자식 커널을 스트림 매니저에 전달
 */
#ifndef __cuda_device_runtime_h__ /* [한국어] 헤더 중복 포함 방지 가드 */
#define __cuda_device_runtime_h__
// Jin: cuda_device_runtime.h
// Defines CUDA device runtime APIs for CDP support
/* [한국어] CUDA 5.0부터 지원되는 Dynamic Parallelism(CDP)용 디바이스 런타임 API 에뮬레이션 정의 */

/*
 * [한국어]
 * device_launch_config_t - 자식 커널(child kernel) 런칭 설정 저장 클래스
 *
 * cudaGetParameterBufferV2 호출 시 디바이스 스레드가 지정한 자식 커널의
 * 실행 설정(grid/block 크기, shared memory, 커널 엔트리)을 보관한다.
 * 파라미터 버퍼 주소를 키로 하는 맵(g_cuda_device_launch_param_map)의 값으로 저장되며,
 * 이후 cudaLaunchDeviceV2 호출 시 이 설정을 꺼내 실제 kernel_info_t를 생성한다.
 */
class device_launch_config_t {
 public:
  device_launch_config_t() {} /* [한국어] 기본 생성자 — 맵 내 기본값 생성 시 사용 */

  /*
   * [한국어]
   * 파라미터 생성자 - 자식 커널 런칭 설정 초기화
   *
   * @_grid_dim: 자식 커널 그리드 차원 (x,y,z)
   * @_block_dim: 자식 커널 블록(CTA) 차원 (x,y,z)
   * @_shared_mem: 자식 커널당 동적 shared memory 크기 (바이트)
   * @_entry: 자식 커널 함수 정보 포인터 (function_info)
   */
  device_launch_config_t(dim3 _grid_dim, dim3 _block_dim,
                         unsigned int _shared_mem, function_info* _entry)
      : grid_dim(_grid_dim),
        block_dim(_block_dim),
        shared_mem(_shared_mem),
        entry(_entry) {}

  dim3 grid_dim;           /* [한국어] 자식 커널 그리드 차원 (x × y × z 개의 CTA).
                             * 설정자: gpgpusim_cuda_getParameterBufferV2()에서 PTX 파라미터로부터 읽어 초기화.
                             * 읽는 자: gpgpusim_cuda_launchDeviceV2()에서 kernel_info_t 생성 시.
                             * 동기화: 생성 후 불변. */
  dim3 block_dim;          /* [한국어] 자식 커널 블록(CTA) 차원 (블록당 x × y × z 스레드).
                             * 설정자: gpgpusim_cuda_getParameterBufferV2()에서 초기화.
                             * 읽는 자: gpgpusim_cuda_launchDeviceV2().
                             * 동기화: 생성 후 불변. */
  unsigned int shared_mem; /* [한국어] 자식 커널당 동적 shared memory 요청 크기(바이트).
                             * 설정자: gpgpusim_cuda_getParameterBufferV2()에서 초기화.
                             * 읽는 자: gpgpusim_cuda_launchDeviceV2() — kernel_info_t 생성에 사용.
                             * 동기화: 생성 후 불변. */
  function_info* entry;    /* [한국어] 자식 커널의 PTX 함수 정보 포인터.
                             * 설정자: gpgpusim_cuda_getParameterBufferV2()에서 초기화.
                             * 읽는 자: gpgpusim_cuda_launchDeviceV2() — 커널 이름, 파라미터 크기, PDOM 분석.
                             * 동기화: 생성 후 불변. */
};

/*
 * [한국어]
 * device_launch_operation_t - 자식 커널 런칭 대기 연산 단위
 *
 * cudaLaunchDeviceV2 호출 시 생성된 자식 커널(kernel_info_t)과 그 스트림 정보를
 * 묶어서 보관하는 구조체. g_cuda_device_launch_op 큐에 저장되며,
 * 이후 launch_one_device_kernel()이 이를 꺼내 스트림 매니저에 전달한다.
 */
class device_launch_operation_t {
 public:
  device_launch_operation_t() {} /* [한국어] 기본 생성자 — 기본값 초기화 */
  device_launch_operation_t(kernel_info_t* _grid, CUstream_st* _stream)
      : grid(_grid), stream(_stream) {} /* [한국어] grid와 stream으로 초기화하는 생성자 */

  kernel_info_t* grid;   // a new child grid
  /* [한국어] 생성된 자식 커널 실행 정보 포인터.
   * 설정자: gpgpusim_cuda_launchDeviceV2()에서 new kernel_info_t로 생성.
   * 읽는 자: launch_one_device_kernel() → stream_manager에 전달.
   * 동기화: 큐에 push/pop — 단일 스레드 접근이므로 락 불필요. */

  CUstream_st* stream;   /* [한국어] 자식 커널을 실행할 CUDA 스트림 포인터.
                           * 설정자: gpgpusim_cuda_launchDeviceV2()에서 인자로부터 읽거나 CTA 기본 스트림 사용.
                           * 읽는 자: launch_one_device_kernel() → stream_operation 생성 시.
                           * 동기화: 생성 후 불변. */
};

class gpgpu_context; /* [한국어] 전방 선언 — gpgpu_ctx 역방향 포인터 타입 */

/*
 * [한국어]
 * cuda_device_runtime - CUDA Dynamic Parallelism 런타임 API 에뮬레이션 클래스
 *
 * gpgpu_context의 멤버로 존재하며, CUDA 5.0+ CDP API의 디바이스 측 구현을 에뮬레이션한다.
 * 주요 기능:
 *   1. cudaGetParameterBufferV2: 자식 커널 파라미터 버퍼를 global memory에 할당하고
 *      설정을 g_cuda_device_launch_param_map에 저장
 *   2. cudaLaunchDeviceV2: 자식 kernel_info_t를 생성하고 g_cuda_device_launch_op 큐에 추가
 *   3. cudaStreamCreateWithFlags: 디바이스 측 스트림 생성 에뮬레이션
 *   4. launch_all_device_kernels: 큐의 모든 자식 커널을 stream_manager에 전달
 * g_cdp_enabled=false이면 CDP 기능이 비활성화되어 이 클래스의 메서드는 호출되지 않음.
 * 실행 컨텍스트: 기능 시뮬레이션 (호스트 CPU 단일 스레드).
 */
class cuda_device_runtime {
 public:
  /*
   * [한국어] 생성자 - CDP 런타임 상태 초기화
   *
   * @ctx: 상위 gpgpu_context 역방향 포인터
   */
  cuda_device_runtime(gpgpu_context* ctx) {
    g_total_param_size = 0;     /* [한국어] 누적 파라미터 버퍼 크기 0으로 초기화 */
    g_max_total_param_size = 0; /* [한국어] 최대 파라미터 버퍼 크기 0으로 초기화 */
    gpgpu_ctx = ctx;            /* [한국어] gpgpu_context 역방향 포인터 저장 */
  }
  unsigned long long g_total_param_size;         /* [한국어] 현재까지 할당된 자식 커널 파라미터 버퍼 총 크기(바이트).
                                                   * 설정자: gpgpusim_cuda_getParameterBufferV2()에서 += 크기 (256바이트 정렬).
                                                   * 읽는 자: g_max_total_param_size 업데이트 시 비교.
                                                   * 동기화: 기능 시뮬레이션 단일 스레드. */
  std::map<void*, device_launch_config_t> g_cuda_device_launch_param_map; /* [한국어] 파라미터 버퍼 주소 → 자식 커널 런칭 설정 맵.
                                                   * 설정자: gpgpusim_cuda_getParameterBufferV2()에서 삽입.
                                                   * 읽는 자: gpgpusim_cuda_launchDeviceV2()에서 설정 꺼낸 후 erase.
                                                   * 동기화: 단일 스레드. */
  std::list<device_launch_operation_t> g_cuda_device_launch_op;          /* [한국어] 자식 커널 런칭 대기 큐(FIFO).
                                                   * 설정자: gpgpusim_cuda_launchDeviceV2()에서 push_back.
                                                   * 읽는 자: launch_one_device_kernel()에서 front 꺼낸 후 pop_front.
                                                   * 동기화: 단일 스레드 (기능 시뮬레이션에서만 push, 타이밍에서 pop). */
  unsigned g_kernel_launch_latency;              /* [한국어] 자식 커널 런칭 지연 사이클 수 (gpgpusim.config로 설정).
                                                   * 설정자: option_parser를 통해 설정.
                                                   * 읽는 자: 타이밍 시뮬레이션 (CDP 커널 런칭 모델링).
                                                   * 동기화: 옵션 파싱 후 불변. */
  unsigned g_TB_launch_latency;                  /* [한국어] 스레드 블록(CTA) 런칭 지연 사이클 수 (gpgpusim.config로 설정).
                                                   * 설정자: option_parser를 통해 설정.
                                                   * 읽는 자: 타이밍 시뮬레이션.
                                                   * 동기화: 옵션 파싱 후 불변. */
  unsigned long long g_max_total_param_size;     /* [한국어] 시뮬레이션 전체에서 파라미터 버퍼 최대 누적 크기(바이트).
                                                   * 설정자: gpgpusim_cuda_getParameterBufferV2()에서 g_total_param_size > max일 때 갱신.
                                                   * 읽는 자: 통계 출력 시.
                                                   * 동기화: 단일 스레드. */
  bool g_cdp_enabled;                            /* [한국어] CDP(Dynamic Parallelism) 활성화 여부 (-gpgpu_cuda_cdp_enabled 옵션).
                                                   * 설정자: option_parser를 통해 설정.
                                                   * 읽는 자: ptxas 플래그 결정, PDOM 분석, CDP API 콜 디스패치.
                                                   * 동기화: 옵션 파싱 후 불변. */

  // backward pointer
  class gpgpu_context* gpgpu_ctx;               /* [한국어] 상위 gpgpu_context 역방향 포인터.
                                                   * 설정자: 생성자에서 초기화.
                                                   * 읽는 자: launch_one_device_kernel()에서 stream_manager 접근.
                                                   * 동기화: 생성 후 불변. */
#if (CUDART_VERSION >= 5000) /* [한국어] CUDA 5.0 이상에서만 CDP API 지원 */
#pragma once /* [한국어] (참고: 이 위치의 #pragma once는 클래스 내부에 있어 효과 없음 — 헤더 가드는 파일 상단에서 처리) */
  /*
   * [한국어]
   * gpgpusim_cuda_launchDeviceV2 - cudaLaunchDeviceV2 에뮬레이션
   *
   * @pI: 현재 PTX call 명령어
   * @thread: 호출 스레드
   * @target_func: cudaLaunchDeviceV2 함수 정보
   *
   * 파라미터 버퍼 주소로 런칭 설정을 찾아 kernel_info_t 생성, PDOM 분석,
   * 파라미터 복사 후 g_cuda_device_launch_op 큐에 추가.
   */
  void gpgpusim_cuda_launchDeviceV2(const ptx_instruction* pI,
                                    ptx_thread_info* thread,
                                    const function_info* target_func);
  /*
   * [한국어]
   * gpgpusim_cuda_streamCreateWithFlags - cudaStreamCreateWithFlags 에뮬레이션
   *
   * @pI: 현재 PTX call 명령어
   * @thread: 호출 스레드
   * @target_func: cudaStreamCreateWithFlags 함수 정보
   *
   * CTA 단위 스트림 생성 — cudaStreamNonBlocking 플래그만 지원.
   */
  void gpgpusim_cuda_streamCreateWithFlags(const ptx_instruction* pI,
                                           ptx_thread_info* thread,
                                           const function_info* target_func);
  /*
   * [한국어]
   * gpgpusim_cuda_getParameterBufferV2 - cudaGetParameterBufferV2 에뮬레이션
   *
   * @pI: 현재 PTX call 명령어
   * @thread: 호출 스레드
   * @target_func: cudaGetParameterBufferV2 함수 정보
   *
   * 자식 커널 파라미터 버퍼를 global memory에 할당하고 런칭 설정을 맵에 저장.
   * 버퍼 주소를 반환값으로 스레드 로컬 메모리에 기록.
   */
  void gpgpusim_cuda_getParameterBufferV2(const ptx_instruction* pI,
                                          ptx_thread_info* thread,
                                          const function_info* target_func);
  /*
   * [한국어]
   * launch_all_device_kernels - 큐의 모든 자식 커널을 스트림 매니저에 전달
   *
   * g_cuda_device_launch_op 큐가 빌 때까지 launch_one_device_kernel 반복 호출.
   * gpu-sim.cc 사이클 루프에서 주기적으로 호출.
   */
  void launch_all_device_kernels();
  /*
   * [한국어]
   * launch_one_device_kernel - 큐 앞 자식 커널 하나를 스트림 매니저에 전달
   *
   * g_cuda_device_launch_op 큐에서 첫 번째 항목을 꺼내 stream_operation으로 변환하여
   * g_stream_manager->push() 호출. 큐가 비면 아무 동작 없음.
   */
  void launch_one_device_kernel();
#endif
};

#endif /* __cuda_device_runtime_h__  */
