/*
 * [한국어 설명] CUDA Dynamic Parallelism(CDP) 런타임 API 에뮬레이션 구현 (cuda_device_runtime.cc)
 *
 * === 파일의 역할 ===
 * cuda_device_runtime.h에서 선언된 CDP API 에뮬레이션 메서드를 구현한다.
 * CUDA 5.0+의 Dynamic Parallelism은 GPU 커널에서 직접 다른 커널을 런칭하는 기능으로,
 * GPGPU-Sim에서는 이를 기능 시뮬레이션 단계에서 소프트웨어로 에뮬레이션한다.
 * 자식 커널 파라미터 버퍼 할당(getParameterBufferV2), 자식 커널 생성 및 큐 등록
 * (launchDeviceV2), 디바이스 스트림 생성(streamCreateWithFlags), 큐 실행
 * (launch_all/one_device_kernel)의 전체 CDP 런칭 파이프라인을 구현한다.
 * CUDART_VERSION < 5000인 경우 이 파일 전체가 컴파일되지 않는다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: PTX 명령어 실행(instructions.cc) → CDP API 디스패치
 *   → gpgpusim_cuda_getParameterBufferV2 (파라미터 버퍼 할당)
 *   → gpgpusim_cuda_launchDeviceV2 (자식 커널 생성 + 큐 등록)
 *   → (부모 커널 완료 후) gpu-sim.cc → launch_all_device_kernels
 *   → stream_manager → 자식 커널 스케줄링
 * 실행 컨텍스트: 기능 시뮬레이션 단계 (호스트 CPU 단일 스레드).
 *
 * === 타 모듈과의 연결 ===
 * 의존: cuda_device_runtime.h, ptx_ir.h, libcuda/gpgpu_context.h,
 *   gpgpu-sim/gpu-sim.h (gpu_malloc), stream_manager.h (stream_operation),
 *   gpgpusim_entrypoint.h
 * 의존받음: cuda-sim/instructions.cc (CDP API call 디스패치),
 *   gpgpu-sim/gpu-sim.cc (launch_all_device_kernels 호출)
 * 공유 자료구조: g_cuda_device_launch_param_map, g_cuda_device_launch_op,
 *   g_total_param_size, g_max_total_param_size
 *
 * === 주요 함수/구조체 요약 ===
 * gpgpusim_cuda_getParameterBufferV2(): 자식 커널 파라미터 버퍼 global memory 할당
 * gpgpusim_cuda_launchDeviceV2(): 자식 kernel_info_t 생성, PDOM 분석, 큐 등록
 * gpgpusim_cuda_streamCreateWithFlags(): CTA 단위 스트림 생성 에뮬레이션
 * launch_one_device_kernel(): 큐 앞 자식 커널을 stream_manager에 전달
 * launch_all_device_kernels(): 큐 전체 flush
 * DEV_RUNTIME_REPORT: 디버그 출력 매크로 (g_debug_execution 활성 시)
 */
// Jin: cuda_device_runtime.cc
// Defines CUDA device runtime APIs for CDP support
/* [한국어] CUDA 5.0+ CDP 디바이스 런타임 API 에뮬레이션 구현 */

#include <iostream> /* [한국어] std::cout — DEV_RUNTIME_REPORT 디버그 출력 */
#include <map>      /* [한국어] std::map — (cuda_device_runtime.h를 통해 간접 사용) */

#if (CUDART_VERSION >= 5000) /* [한국어] CUDA 5.0 이상에서만 CDP 지원 — 이하 코드 전체 조건부 컴파일 */
#define __CUDA_RUNTIME_API_H__ /* [한국어] CUDA 런타임 헤더 중복 포함 방지 — builtin_types.h가 내부적으로 cuda_runtime_api.h를 포함하지 않도록 */

#include <builtin_types.h>   /* [한국어] dim3, cudaError_t 등 CUDA 빌트인 타입 */
#include <driver_types.h>    /* [한국어] cudaStream_t, CUstream_st 등 드라이버 타입 */
#include "../../libcuda/gpgpu_context.h" /* [한국어] gpgpu_context — stream_manager, func_sim 등 접근 */
#include "../gpgpu-sim/gpu-sim.h"        /* [한국어] gpgpu_t::gpu_malloc() — 파라미터 버퍼 global memory 할당 */
#include "../gpgpusim_entrypoint.h"      /* [한국어] gpgpusim_entrypoint 전역 변수 접근 */
#include "../stream_manager.h"           /* [한국어] stream_operation, g_stream_manager — 자식 커널 스케줄링 */
#include "cuda-sim.h"                    /* [한국어] GLOBAL_HEAP_START, generic_to_local() 등 */
#include "cuda_device_runtime.h"         /* [한국어] cuda_device_runtime 클래스 선언 */
#include "ptx_ir.h"                      /* [한국어] ptx_instruction, ptx_thread_info, function_info, kernel_info_t 등 */

#define DEV_RUNTIME_REPORT(a)                                       \
  if (g_debug_execution) {                                          \
    std::cout << __FILE__ << ", " << __LINE__ << ": " << a << "\n"; \
    std::cout.flush();                                              \
  }
/* [한국어] DEV_RUNTIME_REPORT - CDP 디버그 출력 매크로
 * g_debug_execution이 true(0이 아님)일 때만 파일명, 줄 번호, 메시지를 stdout에 출력.
 * 인자 a는 << 연산자로 이어지는 출력 스트림 표현식. 예: DEV_RUNTIME_REPORT("child kernel: " << name) */

// Handling device runtime api:
// void * cudaGetParameterBufferV2(void *func, dim3 gridDimension, dim3
// blockDimension, unsigned int sharedMemSize)
/*
 * [한국어]
 * cuda_device_runtime::gpgpusim_cuda_getParameterBufferV2 - cudaGetParameterBufferV2 에뮬레이션
 *
 * @pI: PTX call 명령어 — 4개 실제 파라미터 피연산자 포함
 * @thread: 호출 스레드 — 로컬 메모리에서 파라미터 읽기
 * @target_func: cudaGetParameterBufferV2 함수 정보 — 4개 인자, 1개 리턴값
 *
 * CUDA API: void* cudaGetParameterBufferV2(void *func, dim3 gridDim, dim3 blockDim, unsigned sharedMem)
 * 처리 순서:
 *   1. PTX 파라미터에서 자식 커널 엔트리, grid/block 크기, shared memory 크기를 읽음
 *   2. 자식 커널 파라미터 크기만큼 global memory 버퍼 할당 (gpu_malloc)
 *   3. 파라미터 버퍼 주소 → 런칭 설정 맵(g_cuda_device_launch_param_map) 등록
 *   4. 버퍼 주소를 반환값으로 스레드 로컬 메모리에 기록
 * g_total_param_size: 누적 파라미터 버퍼 크기 추적 (256바이트 정렬).
 * 실행 컨텍스트: 기능 시뮬레이션 (CDP 커널 내 cudaGetParameterBufferV2 호출 시).
 *
 * 호출 체인:
 *   instructions.cc (call 명령어 → CDP API 디스패치) → [이 함수]
 */
void cuda_device_runtime::gpgpusim_cuda_getParameterBufferV2(
    const ptx_instruction *pI, ptx_thread_info *thread,
    const function_info *target_func) {
  DEV_RUNTIME_REPORT("Calling cudaGetParameterBufferV2"); /* [한국어] 디버그 모드에서 API 진입 로그 */

  unsigned n_return = target_func->has_return(); /* [한국어] void* 반환값 있음 — 피연산자 인덱스 계산용 */
  assert(n_return);                              /* [한국어] 반환값 없으면 버그 */
  unsigned n_args = target_func->num_args();     /* [한국어] 인자 수 확인 */
  assert(n_args == 4);                           /* [한국어] func, gridDim, blockDim, sharedMem = 정확히 4개 */

  function_info *child_kernel_entry = NULL; /* [한국어] 자식 커널 함수 정보 포인터 */
  struct dim3 grid_dim, block_dim;          /* [한국어] 자식 커널 grid/block 차원 */
  unsigned int shared_mem;                  /* [한국어] 자식 커널 shared memory 크기 */

  for (unsigned arg = 0; arg < n_args; arg++) { /* [한국어] 4개 인자 순서대로 읽기 */
    const operand_info &actual_param_op =
        pI->operand_lookup(n_return + 1 + arg);  // param#
    /* [한국어] PTX 명령어에서 arg번째 실제 파라미터 피연산자 조회 (n_return은 리턴값 피연산자 오프셋) */
    const symbol *formal_param =
        target_func->get_arg(arg);  // cudaGetParameterBufferV2_param_#
    /* [한국어] 함수 정의에서 arg번째 형식 파라미터 정보 조회 */
    unsigned size = formal_param->get_size_in_bytes(); /* [한국어] 이 파라미터의 크기(바이트) */
    assert(formal_param->is_param_local());   /* [한국어] 파라미터는 로컬 param 공간이어야 함 */
    assert(actual_param_op.is_param_local()); /* [한국어] 실제 피연산자도 param_local이어야 함 */
    addr_t from_addr = actual_param_op.get_symbol()->get_address(); /* [한국어] 파라미터의 로컬 메모리 주소 */

    if (arg == 0) {  // function_info* for the child kernel
      /* [한국어] 인자 0: 자식 커널 함수 포인터 (function_info*) */
      unsigned long long buf;              /* [한국어] 포인터 값을 읽을 임시 버퍼 */
      assert(size == sizeof(function_info *)); /* [한국어] 포인터 크기 검증 */
      thread->m_local_mem->read(from_addr, size, &buf); /* [한국어] 로컬 메모리에서 function_info* 읽기 */
      child_kernel_entry = (function_info *)buf; /* [한국어] 읽은 주소를 function_info*로 해석 */
      assert(child_kernel_entry);          /* [한국어] 유효한 커널 함수여야 함 */
      DEV_RUNTIME_REPORT("child kernel name "
                         << child_kernel_entry->get_name()); /* [한국어] 자식 커널 이름 디버그 출력 */
    } else if (arg == 1) {  // dim3 grid_dim for the child kernel
      /* [한국어] 인자 1: 자식 커널 그리드 차원 (dim3) */
      assert(size == sizeof(struct dim3)); /* [한국어] dim3 크기 검증 */
      thread->m_local_mem->read(from_addr, size, &grid_dim); /* [한국어] 그리드 차원 읽기 */
      DEV_RUNTIME_REPORT("grid (" << grid_dim.x << ", " << grid_dim.y << ", "
                                  << grid_dim.z << ")"); /* [한국어] 그리드 차원 디버그 출력 */
    } else if (arg == 2) {  // dim3 block_dim for the child kernel
      /* [한국어] 인자 2: 자식 커널 블록 차원 (dim3) */
      assert(size == sizeof(struct dim3)); /* [한국어] dim3 크기 검증 */
      thread->m_local_mem->read(from_addr, size, &block_dim); /* [한국어] 블록 차원 읽기 */
      DEV_RUNTIME_REPORT("block (" << block_dim.x << ", " << block_dim.y << ", "
                                   << block_dim.z << ")"); /* [한국어] 블록 차원 디버그 출력 */
    } else if (arg == 3) {  // unsigned int shared_mem
      /* [한국어] 인자 3: 동적 shared memory 크기 (unsigned int) */
      assert(size == sizeof(unsigned int)); /* [한국어] unsigned int 크기 검증 */
      thread->m_local_mem->read(from_addr, size, &shared_mem); /* [한국어] shared memory 크기 읽기 */
      DEV_RUNTIME_REPORT("shared memory " << shared_mem); /* [한국어] shared memory 크기 디버그 출력 */
    }
  }

  // get total child kernel argument size and malloc buffer in global memory
  /* [한국어] 자식 커널 파라미터 버퍼 크기 계산 및 global memory 할당 */
  unsigned child_kernel_arg_size = child_kernel_entry->get_args_aligned_size(); /* [한국어] 자식 커널의 파라미터 총 크기 (정렬 포함) */
  void *param_buffer = thread->get_gpu()->gpu_malloc(child_kernel_arg_size);    /* [한국어] global memory에 파라미터 버퍼 할당 — gpu_malloc은 GLOBAL_HEAP에서 할당 */
  g_total_param_size += ((child_kernel_arg_size + 255) / 256 * 256);           /* [한국어] 256바이트 정렬한 크기를 누적 — 메모리 사용량 통계 */
  DEV_RUNTIME_REPORT("child kernel arg size total "
                     << child_kernel_arg_size
                     << ", parameter buffer allocated at " << param_buffer); /* [한국어] 파라미터 버퍼 크기 및 위치 디버그 출력 */
  if (g_total_param_size > g_max_total_param_size)
    g_max_total_param_size = g_total_param_size; /* [한국어] 최대 파라미터 버퍼 크기 갱신 */

  // store param buffer address and launch config
  /* [한국어] 파라미터 버퍼 주소 → 런칭 설정 맵에 저장 */
  device_launch_config_t device_launch_config(grid_dim, block_dim, shared_mem,
                                              child_kernel_entry); /* [한국어] 자식 커널 런칭 설정 객체 생성 */
  assert(g_cuda_device_launch_param_map.find(param_buffer) ==
         g_cuda_device_launch_param_map.end()); /* [한국어] 같은 버퍼 주소가 이미 등록되어 있으면 버그 */
  g_cuda_device_launch_param_map[param_buffer] = device_launch_config; /* [한국어] 버퍼 주소 → 런칭 설정 맵에 저장 */

  // copy the buffer address to retval0
  /* [한국어] 반환값으로 파라미터 버퍼 주소를 스레드 로컬 메모리에 기록 */
  const operand_info &actual_return_op = pI->operand_lookup(0);  // retval0
  /* [한국어] PTX 명령어의 반환값 피연산자(retval0) 조회 */
  const symbol *formal_return = target_func->get_return_var();   // void *
  /* [한국어] 함수 리턴 변수 정보 (void* 타입) */
  unsigned int return_size = formal_return->get_size_in_bytes(); /* [한국어] void* 크기 (보통 8바이트) */
  DEV_RUNTIME_REPORT("cudaGetParameterBufferV2 return value has size of "
                     << return_size); /* [한국어] 반환값 크기 디버그 출력 */
  assert(actual_return_op.is_param_local()); /* [한국어] 반환값 피연산자가 param_local이어야 함 */
  assert(actual_return_op.get_symbol()->get_size_in_bytes() == return_size &&
         return_size == sizeof(void *)); /* [한국어] 반환값 크기가 void* 크기와 일치해야 함 */
  addr_t ret_param_addr = actual_return_op.get_symbol()->get_address(); /* [한국어] 반환값 저장 위치(로컬 메모리 오프셋) */
  thread->m_local_mem->write(ret_param_addr, return_size, &param_buffer, NULL,
                             NULL); /* [한국어] 할당된 파라미터 버퍼 주소를 반환값으로 기록 */
}

// Handling device runtime api:
// cudaError_t cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream)
/*
 * [한국어]
 * cuda_device_runtime::gpgpusim_cuda_launchDeviceV2 - cudaLaunchDeviceV2 에뮬레이션
 *
 * @pI: PTX call 명령어 — 2개 실제 파라미터 피연산자 포함
 * @thread: 호출 스레드 — 부모 커널/CTA/스레드 식별
 * @target_func: cudaLaunchDeviceV2 함수 정보 — 2개 인자, cudaError_t 리턴값
 *
 * CUDA API: cudaError_t cudaLaunchDeviceV2(void *parameterBuffer, cudaStream_t stream)
 * 처리 순서:
 *   1. 파라미터 버퍼 주소 읽기 → g_cuda_device_launch_param_map에서 런칭 설정 조회
 *   2. 자식 커널 PDOM(Post-DOMinator) 분석 (아직 안 된 경우)
 *   3. 파라미터 버퍼에서 자식 커널 param 메모리로 데이터 복사 (4바이트 단위)
 *   4. 스트림 처리: NULL이면 CTA 기본 스트림 사용, 아니면 부모 커널의 CTA 스트림 확인
 *   5. g_cuda_device_launch_op 큐에 추가, 맵에서 해당 항목 제거
 *   6. 반환값 cudaSuccess를 스레드 로컬 메모리에 기록
 * 실행 컨텍스트: 기능 시뮬레이션.
 *
 * 호출 체인:
 *   instructions.cc (call 명령어 → CDP API 디스패치) → [이 함수]
 *     → launch_all_device_kernels → stream_manager
 */
void cuda_device_runtime::gpgpusim_cuda_launchDeviceV2(
    const ptx_instruction *pI, ptx_thread_info *thread,
    const function_info *target_func) {
  DEV_RUNTIME_REPORT("Calling cudaLaunchDeviceV2"); /* [한국어] 디버그 모드에서 API 진입 로그 */

  unsigned n_return = target_func->has_return(); /* [한국어] cudaError_t 반환값 있음 */
  assert(n_return);                              /* [한국어] 반환값 없으면 버그 */
  unsigned n_args = target_func->num_args();     /* [한국어] 인자 수 확인 */
  assert(n_args == 2);                           /* [한국어] parameterBuffer + stream = 정확히 2개 */

  kernel_info_t *device_grid = NULL;         /* [한국어] 생성될 자식 커널 실행 정보 */
  function_info *device_kernel_entry = NULL; /* [한국어] 자식 커널 함수 정보 */
  void *parameter_buffer;                    /* [한국어] 파라미터 버퍼 주소 (global memory) */
  struct CUstream_st *child_stream;          /* [한국어] 자식 커널을 실행할 스트림 */
  device_launch_config_t config;             /* [한국어] 파라미터 맵에서 조회한 런칭 설정 */
  device_launch_operation_t device_launch_op; /* [한국어] 큐에 추가할 런칭 연산 */

  for (unsigned arg = 0; arg < n_args; arg++) { /* [한국어] 2개 인자 순서대로 처리 */
    const operand_info &actual_param_op =
        pI->operand_lookup(n_return + 1 + arg);  // param#
    /* [한국어] PTX 명령어에서 arg번째 실제 파라미터 피연산자 조회 */
    const symbol *formal_param =
        target_func->get_arg(arg);  // cudaLaunchDeviceV2_param_#
    /* [한국어] 함수 정의에서 arg번째 형식 파라미터 정보 */
    unsigned size = formal_param->get_size_in_bytes(); /* [한국어] 파라미터 크기 */
    assert(formal_param->is_param_local());   /* [한국어] param_local이어야 함 */
    assert(actual_param_op.is_param_local()); /* [한국어] param_local이어야 함 */
    addr_t from_addr = actual_param_op.get_symbol()->get_address(); /* [한국어] 파라미터 로컬 메모리 주소 */

    if (arg == 0) {  // paramter buffer for child kernel (in global memory)
      // get parameter_buffer from the cudaLaunchDeviceV2_param0
      /* [한국어] 인자 0: 파라미터 버퍼 주소 (getParameterBufferV2가 반환한 global memory 주소) */
      assert(size == sizeof(void *)); /* [한국어] void* 크기 검증 */
      thread->m_local_mem->read(from_addr, size, &parameter_buffer); /* [한국어] 로컬 메모리에서 버퍼 주소 읽기 */
      assert((size_t)parameter_buffer >= GLOBAL_HEAP_START); /* [한국어] 유효한 global memory 주소여야 함 */
      DEV_RUNTIME_REPORT("Parameter buffer locating at global memory "
                         << parameter_buffer); /* [한국어] 버퍼 주소 디버그 출력 */

      // get child grid info through parameter_buffer address
      /* [한국어] 파라미터 버퍼 주소로 런칭 설정 맵에서 자식 커널 정보 조회 */
      assert(g_cuda_device_launch_param_map.find(parameter_buffer) !=
             g_cuda_device_launch_param_map.end()); /* [한국어] 맵에 없으면 버그 (getParameterBufferV2 호출 전 launchDeviceV2 호출) */
      config = g_cuda_device_launch_param_map[parameter_buffer]; /* [한국어] 런칭 설정 조회 */
      // device_grid = op.grid;
      device_kernel_entry = config.entry; /* [한국어] 자식 커널 함수 정보 추출 */
      DEV_RUNTIME_REPORT("find device kernel "
                         << device_kernel_entry->get_name()); /* [한국어] 자식 커널 이름 디버그 출력 */

      // PDOM analysis is done for Parent kernel but not for child kernel.
      /* [한국어] 자식 커널에 대한 PDOM(Post-DOMinator) 분석 수행 — 워프 재합류 포인트 결정 */
      if (device_kernel_entry->is_pdom_set()) { /* [한국어] 이미 PDOM 분석된 경우 (같은 커널을 여러 번 런칭) */
        printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
               device_kernel_entry->get_name().c_str()); /* [한국어] 재사용 알림 */
      } else { /* [한국어] 처음 런칭되는 커널 — PDOM 분석 필요 */
        printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
               device_kernel_entry->get_name().c_str()); /* [한국어] PDOM 분석 시작 알림 */
        /*
         * Some of the instructions like printf() gives the gpgpusim the wrong
         * impression that it is a function call. As printf() doesnt have a body
         * like functions do, doing pdom analysis for printf() causes a crash.
         */
        /* [한국어] printf 같은 빌트인 함수는 함수 본체가 없어 PDOM 분석 불가 — 크기 > 0 확인 */
        if (device_kernel_entry->get_function_size() > 0)
          device_kernel_entry->do_pdom(); /* [한국어] SIMT 재합류(reconvergence) 포인트 분석 */
        device_kernel_entry->set_pdom();  /* [한국어] PDOM 분석 완료 플래그 설정 */
      }

      // copy data in parameter_buffer to device kernel param memory
      /* [한국어] global memory 파라미터 버퍼에서 자식 커널의 param 메모리로 4바이트 단위 복사 */
      unsigned device_kernel_arg_size =
          device_kernel_entry->get_args_aligned_size(); /* [한국어] 자식 커널 파라미터 총 크기 */
      DEV_RUNTIME_REPORT("device_kernel_arg_size " << device_kernel_arg_size); /* [한국어] 크기 디버그 출력 */
      memory_space *device_kernel_param_mem; /* [한국어] 자식 커널의 param 메모리 공간 포인터 */

      // create child kernel_info_t and index it with parameter_buffer address
      /* [한국어] 자식 커널의 kernel_info_t 객체 생성 — 그리드/블록 차원, 함수 정보 포함 */
      gpgpu_t *gpu = thread->get_gpu(); /* [한국어] 현재 스레드가 실행 중인 gpgpu_t 포인터 */
      device_grid = new kernel_info_t(
          config.grid_dim, config.block_dim, device_kernel_entry,
          gpu->getNameArrayMapping(), gpu->getNameInfoMapping());
      device_grid->launch_cycle = gpu->gpu_sim_cycle + gpu->gpu_tot_sim_cycle;
      kernel_info_t &parent_grid = thread->get_kernel();
      DEV_RUNTIME_REPORT(
          "child kernel launched by "
          << parent_grid.name() << ", cta (" << thread->get_ctaid().x << ", "
          << thread->get_ctaid().y << ", " << thread->get_ctaid().z
          << "), thread (" << thread->get_tid().x << ", " << thread->get_tid().y
          << ", " << thread->get_tid().z << ")");
      device_grid->set_parent(&parent_grid, thread->get_ctaid(),
                              thread->get_tid());
      device_launch_op = device_launch_operation_t(device_grid, NULL);
      device_kernel_param_mem = device_grid->get_param_memory();  // kernel
                                                                  // param
      size_t param_start_address = 0;
      // copy in word
      for (unsigned n = 0; n < device_kernel_arg_size; n += 4) {
        unsigned int oneword;
        thread->get_gpu()->get_global_memory()->read(
            (size_t)parameter_buffer + n, 4, &oneword);
        device_kernel_param_mem->write(param_start_address + n, 4, &oneword,
                                       NULL, NULL);
      }
    } else if (arg == 1) {  // cudaStream for the child kernel

      assert(size == sizeof(cudaStream_t));
      thread->m_local_mem->read(from_addr, size, &child_stream);

      kernel_info_t &parent_kernel = thread->get_kernel();
      if (child_stream == 0) {  // default stream on device for current CTA
        child_stream =
            parent_kernel.get_default_stream_cta(thread->get_ctaid());
        DEV_RUNTIME_REPORT("launching child kernel "
                           << device_grid->get_uid()
                           << " to default stream of the cta "
                           << child_stream->get_uid() << ": " << child_stream);
      } else {
        assert(parent_kernel.cta_has_stream(thread->get_ctaid(), child_stream));
        DEV_RUNTIME_REPORT("launching child kernel "
                           << device_grid->get_uid() << " to stream "
                           << child_stream->get_uid() << ": " << child_stream);
      }

      device_launch_op.stream = child_stream;
    }
  }

  // launch child kernel
  g_cuda_device_launch_op.push_back(device_launch_op);
  g_cuda_device_launch_param_map.erase(parameter_buffer);

  // set retval0
  const operand_info &actual_return_op = pI->operand_lookup(0);  // retval0
  const symbol *formal_return = target_func->get_return_var();   // cudaError_t
  unsigned int return_size = formal_return->get_size_in_bytes();
  DEV_RUNTIME_REPORT("cudaLaunchDeviceV2 return value has size of "
                     << return_size);
  assert(actual_return_op.is_param_local());
  assert(actual_return_op.get_symbol()->get_size_in_bytes() == return_size &&
         return_size == sizeof(cudaError_t));
  cudaError_t error = cudaSuccess;
  addr_t ret_param_addr = actual_return_op.get_symbol()->get_address();
  thread->m_local_mem->write(ret_param_addr, return_size, &error, NULL, NULL);
}

// Handling device runtime api:
// cudaError_t cudaStreamCreateWithFlags ( cudaStream_t* pStream, unsigned int
// flags) flags can only be cudaStreamNonBlocking
void cuda_device_runtime::gpgpusim_cuda_streamCreateWithFlags(
    const ptx_instruction *pI, ptx_thread_info *thread,
    const function_info *target_func) {
  DEV_RUNTIME_REPORT("Calling cudaStreamCreateWithFlags");

  unsigned n_return = target_func->has_return();
  assert(n_return);
  unsigned n_args = target_func->num_args();
  assert(n_args == 2);

  size_t generic_pStream_addr;
  addr_t pStream_addr = 0;
  unsigned int flags;
  for (unsigned arg = 0; arg < n_args; arg++) {
    const operand_info &actual_param_op =
        pI->operand_lookup(n_return + 1 + arg);  // param#
    const symbol *formal_param =
        target_func->get_arg(arg);  // cudaStreamCreateWithFlags_param_#
    unsigned size = formal_param->get_size_in_bytes();
    assert(formal_param->is_param_local());
    assert(actual_param_op.is_param_local());
    addr_t from_addr = actual_param_op.get_symbol()->get_address();

    if (arg == 0) {  // cudaStream_t * pStream, address of cudaStream_t
      assert(size == sizeof(cudaStream_t *));
      thread->m_local_mem->read(from_addr, size, &generic_pStream_addr);

      // pStream should be non-zero address in local memory
      pStream_addr = generic_to_local(
          thread->get_hw_sid(), thread->get_hw_tid(), generic_pStream_addr);

      DEV_RUNTIME_REPORT("pStream locating at local memory " << pStream_addr);
    } else if (arg ==
               1) {  // unsigned int flags, should be cudaStreamNonBlocking
      assert(size == sizeof(unsigned int));
      thread->m_local_mem->read(from_addr, size, &flags);
      assert(flags == cudaStreamNonBlocking);
    }
  }

  // create stream and write back to param0
  CUstream_st *stream =
      thread->get_kernel().create_stream_cta(thread->get_ctaid());
  DEV_RUNTIME_REPORT("Create stream " << stream->get_uid() << ": " << stream);
  thread->m_local_mem->write(pStream_addr, sizeof(cudaStream_t), &stream, NULL,
                             NULL);

  // set retval0
  const operand_info &actual_return_op = pI->operand_lookup(0);  // retval0
  const symbol *formal_return = target_func->get_return_var();   // cudaError_t
  unsigned int return_size = formal_return->get_size_in_bytes();
  DEV_RUNTIME_REPORT("cudaStreamCreateWithFlags return value has size of "
                     << return_size);
  assert(actual_return_op.is_param_local());
  assert(actual_return_op.get_symbol()->get_size_in_bytes() == return_size &&
         return_size == sizeof(cudaError_t));
  cudaError_t error = cudaSuccess;
  addr_t ret_param_addr = actual_return_op.get_symbol()->get_address();
  thread->m_local_mem->write(ret_param_addr, return_size, &error, NULL, NULL);
}

void cuda_device_runtime::launch_one_device_kernel() {
  if (!g_cuda_device_launch_op.empty()) {
    device_launch_operation_t &op = g_cuda_device_launch_op.front();

    stream_operation stream_op = stream_operation(
        op.grid, gpgpu_ctx->func_sim->g_ptx_sim_mode, op.stream);
    gpgpu_ctx->the_gpgpusim->g_stream_manager->push(stream_op);
    g_cuda_device_launch_op.pop_front();
  }
}

void cuda_device_runtime::launch_all_device_kernels() {
  while (!g_cuda_device_launch_op.empty()) {
    launch_one_device_kernel();
  }
}
#endif
