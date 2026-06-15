/*
 * [한국어 설명] GPGPU-Sim OpenCL 런타임 API 구현 (opencl_runtime_api.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 OpenCL 1.0/1.1 API 함수들을 GPGPU-Sim 시뮬레이터 위에 구현한다.
 * OpenCL 애플리케이션이 clEnqueueNDRangeKernel, clCreateBuffer, clBuildProgram 등을
 * 호출하면, 실제 OpenCL ICD(Installable Client Driver) 대신 이 파일의 함수들이 실행된다.
 * 핵심 역할은 두 가지이다: (1) OpenCL C 소스를 PTX로 컴파일하여 GPGPU-Sim에 로드하고,
 * (2) OpenCL 커널 실행 요청(clEnqueueNDRangeKernel)을 GPGPU-Sim의 타이밍 시뮬레이션
 * 루프(gpgpu_opencl_ptx_sim_main_perf) 또는 기능 시뮬레이션(gpgpu_opencl_ptx_sim_main_func)으로 전달한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * OpenCL 애플리케이션 → libopencl.so(이 파일) → GPGPUSim_Init()으로 시뮬레이터 생성
 *   → clBuildProgram → nvopencl_wrapper(NVIDIA 드라이버로 CL→PTX 변환)
 *   → clEnqueueNDRangeKernel → gpgpusim_entrypoint.cc → gpu-sim.cc(사이클 루프)
 * 실행 컨텍스트: 호스트 CPU 유저스페이스. GPU 디바이스 코드와는 무관하며,
 * 시뮬레이션 스레드(start_sim_thread(2))는 별도 스레드에서 사이클 루프를 실행한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - src/abstract_hardware_model.h: warp_inst_t, kernel_info_t 등 추상 HW 모델
 *   - src/cuda-sim/cuda-sim.h: PTX 시뮬레이션 함수 (gpgpu_opencl_ptx_sim_main_func)
 *   - src/cuda-sim/ptx_loader.h: PTX 로딩 (gpgpu_ptx_sim_load_ptx_from_string)
 *   - src/cuda-sim/ptx_ir.h: PTX IR (function_info, symbol, type_info_key)
 *   - src/gpgpusim_entrypoint.h: GPGPUSim_Context, start_sim_thread
 *   - src/gpgpu-sim/gpu-sim.h: gpgpu_sim, gpgpu_t (gpu_malloc, memcpy_to_gpu 등)
 *   - libcuda/gpgpu_context.h: gpgpu_context (전역 시뮬레이터 컨텍스트)
 *   - CL/cl.h: OpenCL 표준 타입 및 에러 코드 (cl_int, cl_mem, CL_SUCCESS 등)
 * 데이터 흐름: OpenCL 소스 → _cl_program::Build() → PTX 문자열 → gpgpu_ptx_sim_load_ptx_from_string
 *   → function_info* → _cl_kernel → clEnqueueNDRangeKernel → kernel_info_t → 사이클 시뮬레이션
 *
 * === 주요 함수/구조체 요약 ===
 * GPGPUSim_Init():        시뮬레이터 싱글톤 초기화 — gpgpu_sim* 생성, 시뮬 스레드 시작
 * _cl_program::Build():   OpenCL C → PTX 컴파일 (nvopencl_wrapper 또는 사전 추출 PTX)
 * clEnqueueNDRangeKernel(): 가장 중요한 함수 — work 차원 → CUDA dim3 변환, PDOM 분석, 시뮬 디스패치
 * _cl_mem::_cl_mem():    GPU 메모리 할당(gpu_malloc) 또는 호스트 메모리 래핑
 * _cl_context::CreateBuffer(): _cl_mem 생성 및 hostptr/devptr 맵 등록
 * _cl_kernel::bind_args(): OpenCL 커널 인자를 gpgpu_ptx_sim_arg_list_t로 변환
 * clGetDeviceInfo():      GPGPU-Sim의 설정 값을 OpenCL 표준 쿼리 인터페이스로 노출
 */

/*
 * opencl_runtime_api.cc
 *
 * Copyright © 2009 by Tor M. Aamodt and the University of British Columbia,
 * Vancouver, BC V6T 1Z4, All Rights Reserved.
 * 
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
 * benchmarks/template/ are derived from the CUDA SDK available from 
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
 * src/intersim/ are derived from Booksim (a simulator provided with the 
 * textbook "Principles and Practices of Interconnection Networks" available 
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA 
 * which is distributed seperately by NVIDIA under separate terms and 
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
 *  
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 * 
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
 * Vancouver, BC V6T 1Z4
 */

#include <stdlib.h>  // [한국어] malloc/free/getenv/exit/abort 등 표준 C 라이브러리 — GPU 메모리 할당 래퍼 및 프로세스 종료에 필요
#include <stdio.h>   // [한국어] printf/fprintf/fopen/fclose/fread/fwrite 등 — 에러 메시지 출력과 OpenCL→PTX 임시 파일 I/O에 필요
#include <string.h>  // [한국어] strlen/memcpy/snprintf/strncmp 등 — PTX 소스 문자열 처리 및 명령줄 구성에 필요
#include <assert.h>  // [한국어] assert() 매크로 — 런타임 불변 조건 검사 (예: 커널명 중복 방지)
#include <time.h>    // [한국어] time() 등 — 현재는 직접 사용하지 않으나 헤더 의존성 체인에 포함됨
#ifdef OPENGL_SUPPORT // [한국어] OpenGL 인터롭(interop) 지원이 빌드 옵션으로 활성화된 경우에만 포함
#define GL_GLEXT_PROTOTYPES // [한국어] OpenGL 확장 함수 프로토타입을 직접 선언 — libGL.so 없이도 헤더만으로 사용 가능하게 함
#include <GL/gl.h>  // [한국어] OpenGL 기본 헤더 — 현재 OpenCL-OpenGL 인터롭 함수들을 위해 필요
#endif

/* [한국어] CUDA 런타임 API 헤더(cuda_runtime_api.h)의 include guard를 미리 정의하여
 * CUDA 런타임 헤더가 중복 include되는 것을 방지한다. OpenCL 파일이지만 GPGPU-Sim 내부 헤더들이
 * CUDA 타입을 참조하기 때문에 이 트릭이 필요하다. */
#define __CUDA_RUNTIME_API_H__
#include "host_defines.h"    // [한국어] CUDA 호스트 함수 선언 보조 매크로 (__host__, __device__ 등) — 내부 헤더 의존성 해결
#include "builtin_types.h"   // [한국어] dim3, cudaMemcpyKind 등 CUDA 빌트인 타입 — clEnqueueNDRangeKernel에서 dim3 GridDim/BlockDim 변환에 필요
#if (CUDART_VERSION < 8000)   // [한국어] CUDA 8.0 미만 버전에서는 fatbinary 포맷 헤더가 별도 파일로 존재
#include "__cudaFatFormat.h" // [한국어] CUDA fat binary 포맷 구조체 정의 — CUDA 8.0 이후는 불필요
#endif
#include "../src/abstract_hardware_model.h"   // [한국어] warp_inst_t, kernel_info_t, gpgpu_t 등 GPU 추상 HW 모델 — 커널 실행 및 메모리 접근에 필요
#include "../src/cuda-sim/cuda-sim.h"         // [한국어] PTX 기능 시뮬레이션 함수 선언 — gpgpu_opencl_ptx_sim_main_func 등
#include "../src/cuda-sim/ptx_loader.h"       // [한국어] PTX 문자열 로딩 함수 — gpgpu_ptx_sim_load_ptx_from_string 사용
#include "../src/cuda-sim/ptx_ir.h"           // [한국어] PTX IR 타입 — function_info, symbol, type_info_key (커널 인자 타입 정보 추출)
#include "../src/gpgpusim_entrypoint.h"       // [한국어] GPGPU_Context() 전역 접근자, start_sim_thread() — 시뮬레이터 초기화 및 시작
#include "../src/gpgpu-sim/gpu-sim.h"         // [한국어] gpgpu_sim, memory_config 등 타이밍 시뮬레이터 — gpu_malloc, shader_clock, num_shader 등
#include "../src/gpgpu-sim/shader.h"          // [한국어] shader_core_config — threads_per_core, num_registers_per_core 쿼리에 필요
#include "../libcuda/gpgpu_context.h"         // [한국어] gpgpu_context 클래스 — func_sim, ptxinfo 등 전역 시뮬레이터 상태 접근

/* [한국어] __my_func__: 현재 함수 이름을 나타내는 매크로. 에러 메시지 출력 시 어느 함수에서
 * 오류가 발생했는지 식별하기 위해 사용된다. C++/GNUC 버전에 따라 __func__ (C99/GCC 2.6+) 또는
 * NULL (구형 컴파일러)로 fallback된다. __PRETTY_FUNCTION__은 주석 처리되어 있어 현재 비활성. */
//#   define __my_func__    __PRETTY_FUNCTION__
# if defined __cplusplus ? __GNUC_PREREQ (2, 6) : __GNUC_PREREQ (2, 4)
#   define __my_func__    __func__  // [한국어] C++에서 GCC 2.6+, 또는 순수 C에서 GCC 2.4+ 이상이면 표준 __func__ 사용
# else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define __my_func__    __my_func__  // [한국어] C99 이상이면 __func__ 사용 (재귀적이나 실제로는 컴파일러가 처리)
#  else
#   define __my_func__    ((__const char *) 0)  // [한국어] 구형 컴파일러: NULL 포인터로 fallback — 함수명 없이 NULL 출력
#  endif
# endif

/* [한국어] CL_USE_DEPRECATED_OPENCL_1_0_APIS: OpenCL 1.0에서 deprecated된 함수들
 * (clSetCommandQueueProperty 등)을 사용할 수 있도록 허용하는 매크로. GPGPU-Sim이
 * 오래된 OpenCL 1.0 API를 구현하기 때문에 이 매크로가 필요하다. */
#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
/* [한국어] CL/cl.h: Khronos Group의 공식 OpenCL 헤더. cl_int, cl_mem, cl_context,
 * CL_SUCCESS, CL_INVALID_VALUE 등 모든 OpenCL 타입과 에러 코드를 정의한다.
 * 이 구현 파일은 이 헤더의 함수 시그니처에 맞는 구현체를 제공한다. */
#include <CL/cl.h>

#include <map>     // [한국어] std::map — _cl_context의 hostptr→cl_mem/devptr→cl_mem 매핑, _cl_program의 소스 번호→pgm_info 매핑, _cl_kernel의 인자 인덱스→arg_info 매핑에 사용
#include <string>  // [한국어] std::string — 커널 이름, PTX 소스/어셈블리 문자열 저장에 사용

/*
 * [한국어]
 * setErrCode - OpenCL API 에러 코드를 안전하게 반환 포인터에 기록하는 헬퍼
 *
 * @errcode_ret: 호출자가 제공한 에러 코드 반환 포인터. NULL이면 기록하지 않음
 * @err_code:    기록할 OpenCL 에러 코드 (CL_SUCCESS, CL_INVALID_VALUE 등)
 * @return:      없음 (void)
 *
 * OpenCL API 규약상 errcode_ret 파라미터는 NULL을 허용해야 하므로, 모든 API 함수가
 * 이 래퍼를 통해 에러 코드를 기록한다. 직접 *errcode_ret = err를 쓰면 NULL 역참조가 발생한다.
 *
 * 호출 체인:
 *   clCreateBuffer / clCreateContext / clCreateKernel 등 → [setErrCode] → (반환 포인터에 기록)
 */
static void setErrCode(cl_int *errcode_ret, cl_int err_code) {
   if ( errcode_ret ) {   // [한국어] errcode_ret가 NULL이 아닌 경우에만 기록 — OpenCL 규약: NULL 전달 허용
      *errcode_ret = err_code;  // [한국어] 호출자가 요청한 에러 코드를 출력 포인터에 기록
   }
}

/*
 * [한국어] _cl_context: OpenCL 컨텍스트 객체 — 하나의 시뮬레이션 GPU 사용 세션을 나타낸다.
 *
 * OpenCL 컨텍스트는 디바이스, 메모리 객체, 커맨드 큐, 프로그램, 커널의 공유 공간이다.
 * GPGPU-Sim에서는 단일 GPU(_cl_device_id)만 지원하므로 m_gpu가 유일한 디바이스이다.
 * 메모리 객체 추적을 위해 두 개의 역방향 맵을 유지하여 호스트 포인터 또는 디바이스
 * 포인터로부터 cl_mem 객체를 조회할 수 있다.
 */
struct _cl_context {
   _cl_context( cl_device_id gpu );   // [한국어] 생성자: GPGPUSim_Init()이 반환한 _cl_device_id*로 초기화
   cl_device_id get_first_device();   // [한국어] 연결된 첫 번째 GPU 디바이스 반환 — GPGPU-Sim은 단일 GPU만 지원
   cl_mem CreateBuffer(               // [한국어] 버퍼 메모리 객체 생성 — GPU 메모리 또는 호스트 메모리 할당
               cl_mem_flags flags,
               size_t       size ,
               void *       host_ptr,
               cl_int *     errcode_ret );
   cl_mem lookup_mem( cl_mem m );     // [한국어] 디바이스 포인터 또는 호스트 포인터로 cl_mem 객체 조회 — clEnqueueCopyBuffer에서 사용
private:
   unsigned m_uid;
   /* [한국어] 이 컨텍스트의 고유 식별자.
    * 설정자: 생성자에서 sm_context_uid++ 값으로 할당.
    * 읽는 자: 현재 디버깅/추적 목적으로만 유용하며, 공개 API에서 직접 노출되지 않음.
    * 값 범위: 0부터 시작하는 단조 증가 정수. OpenCL 앱 실행 중 충돌하지 않음.
    * 동기화: sm_context_uid는 static이며 단일 스레드(호스트 초기화 시)에서만 접근됨. */

   cl_device_id m_gpu;
   /* [한국어] 이 컨텍스트에 연결된 GPGPU-Sim 시뮬레이션 GPU 디바이스 포인터.
    * 설정자: 생성자(_cl_context(cl_device_id gpu))에서 m_gpu = gpu로 초기화.
    * 읽는 자: get_first_device()가 반환. CreateBuffer()에서 _cl_mem 생성 시 전달됨.
    * 값 범위: GPGPUSim_Init()이 반환한 유효한 _cl_device_id* 포인터. NULL 불가.
    * 동기화: 생성 이후 변경되지 않으므로 별도 락 불필요. */

   static unsigned sm_context_uid;
   /* [한국어] 컨텍스트 UID 전역 카운터 (static). 새 컨텍스트 생성마다 1씩 증가.
    * 설정자: _cl_context 생성자에서 m_uid = sm_context_uid++로 소비.
    * 읽는 자: m_uid 필드 (각 인스턴스에 고유값 부여).
    * 값 범위: 0부터 시작하는 단조 증가. 초기값은 파일 하단 `unsigned _cl_context::sm_context_uid = 0;` 참조.
    * 동기화: 단일 호스트 초기화 스레드에서만 접근 → 락 불필요. */

   std::map<void*/*host_ptr*/,cl_mem> m_hostptr_to_cl_mem;
   /* [한국어] 호스트 포인터 → cl_mem 객체 역방향 맵.
    * 설정자: CreateBuffer()에서 host_ptr가 있는 경우 m_hostptr_to_cl_mem[host_ptr] = result 삽입.
    * 읽는 자: lookup_mem()에서 디바이스 포인터 조회 실패 시 이 맵으로 fallback.
    * 값 범위: host_ptr가 제공된 cl_mem만 포함. CL_MEM_USE_HOST_PTR인 경우 항상 존재.
    * 동기화: 호스트 스레드에서만 접근. GPU 시뮬레이션 스레드는 이 맵을 접근하지 않음. */

   std::map<cl_mem/*device ptr*/,cl_mem> m_devptr_to_cl_mem;
   /* [한국어] 디바이스 포인터(GPU 주소) → cl_mem 객체 역방향 맵.
    * 설정자: CreateBuffer()에서 m_devptr_to_cl_mem[result->device_ptr()] = result 삽입.
    * 읽는 자: lookup_mem()에서 먼저 이 맵을 검색 → 실패 시 m_hostptr_to_cl_mem 시도.
    * 값 범위: GPU 메모리에 할당된 cl_mem만 포함 (CL_MEM_USE_HOST_PTR이 아닌 경우).
    * 동기화: 호스트 스레드 전용. */
};

/*
 * [한국어] _cl_device_id: GPGPU-Sim의 시뮬레이션 GPU 디바이스를 나타내는 OpenCL 디바이스 객체.
 *
 * OpenCL cl_device_id 타입의 실제 구현체이다. GPGPU-Sim에서는 단일 디바이스만 지원하며,
 * m_gpgpu가 실제 시뮬레이터 인스턴스(gpgpu_sim*)를 가리킨다. m_next는 멀티-GPU 연결 리스트용이나
 * 현재는 사용되지 않는다. GPGPUSim_Init()이 싱글톤으로 하나만 생성한다.
 */
struct _cl_device_id {
   _cl_device_id(gpgpu_sim* gpu) {m_id = 0; m_next = NULL; m_gpgpu=gpu;}  // [한국어] 생성자: gpgpu_sim* 포인터를 받아 디바이스 객체 초기화
   struct _cl_device_id *next() { return m_next; }      // [한국어] 연결 리스트의 다음 디바이스 반환 — 현재는 항상 NULL (단일 GPU 지원)
   gpgpu_sim *the_device() const { return m_gpgpu; }    // [한국어] 실제 GPGPU-Sim 시뮬레이터 인스턴스 반환 — 메모리 연산 및 설정 쿼리에 사용
private:
   unsigned m_id;
   /* [한국어] 디바이스 인덱스 번호 (현재 항상 0).
    * 설정자: 생성자에서 m_id = 0으로 하드코딩.
    * 읽는 자: 현재 외부에서 직접 접근하지 않음.
    * 값 범위: 0 (단일 GPU 지원이므로 항상 0).
    * 동기화: 불변값이므로 동기화 불필요. */

   gpgpu_sim *m_gpgpu;
   /* [한국어] GPGPU-Sim의 최상위 시뮬레이터 인스턴스 포인터.
    * 설정자: GPGPUSim_Init() → new _cl_device_id(the_gpu) 에서 초기화.
    * 읽는 자: the_device() 게터. clGetDeviceInfo, clEnqueueNDRangeKernel, _cl_mem 생성자 등에서 사용.
    * 값 범위: gpgpu_ptx_sim_init_perf()가 반환한 유효한 gpgpu_sim* 포인터. NULL 불가.
    * 동기화: 생성 이후 불변. GPU 시뮬레이션 스레드와 호스트 스레드가 모두 읽지만 변경하지 않음. */

   struct _cl_device_id *m_next;
   /* [한국어] 연결 리스트의 다음 디바이스 포인터 (멀티-GPU 확장용).
    * 설정자: 생성자에서 m_next = NULL로 초기화.
    * 읽는 자: next() 게터. clGetContextInfo의 CL_CONTEXT_DEVICES 처리 루프에서 순회.
    * 값 범위: NULL (현재 단일 GPU 지원만 구현됨).
    * 동기화: 불변값이므로 동기화 불필요. */
};

/*
 * [한국어] _cl_command_queue: OpenCL 커맨드 큐 객체.
 *
 * OpenCL cl_command_queue 타입의 실제 구현체이다. GPGPU-Sim의 시뮬레이션 모델은 동기적이므로
 * 실제 큐잉(비동기 실행)을 구현하지 않는다. 커맨드 큐는 단순히 컨텍스트와 디바이스 정보를
 * 보유하는 컨테이너 역할만 한다. clEnqueueNDRangeKernel 등 enqueue 함수가 호출되면
 * 즉시 동기적으로 시뮬레이션을 실행한다.
 */
struct _cl_command_queue
{
   _cl_command_queue( cl_context context, cl_device_id device, cl_command_queue_properties properties )  // [한국어] 생성자: 컨텍스트, 디바이스, 속성 플래그로 큐 초기화
   {
      m_valid = true;       // [한국어] 유효한 큐임을 표시 — is_valid() 검사에서 사용
      m_context = context;  // [한국어] 연결된 OpenCL 컨텍스트 저장
      m_device = device;    // [한국어] 연결된 GPU 디바이스 저장
      m_properties = properties;  // [한국어] 큐 속성 플래그 저장 (CL_QUEUE_PROFILING_ENABLE 등)
   }
   bool is_valid() { return m_valid; }  // [한국어] 큐의 유효성 검사 — clEnqueueCopyBuffer 등에서 NULL/유효성 확인
   cl_context get_context() { return m_context; }    // [한국어] 큐에 연결된 컨텍스트 반환 — enqueue 함수들에서 메모리 조회에 사용
   cl_device_id get_device() { return m_device; }    // [한국어] 큐에 연결된 디바이스 반환 — memcpy 함수들에서 GPU 접근에 사용
   cl_command_queue_properties get_properties() { return m_properties; }  // [한국어] 큐 속성 반환 — clGetCommandQueueInfo에서 사용
private:
   bool m_valid;
   /* [한국어] 이 커맨드 큐가 유효하게 초기화되었는지 나타내는 플래그.
    * 설정자: 생성자에서 true로 초기화. 현재 false로 설정하는 경로 없음.
    * 읽는 자: is_valid(). clEnqueueCopyBuffer에서 NULL 큐 방어에 사용.
    * 값 범위: true (정상 생성된 큐), false (비정상 상태, 현재 미사용).
    * 동기화: 생성 후 불변이므로 락 불필요. */

   cl_context                     m_context;
   /* [한국어] 이 커맨드 큐가 속한 OpenCL 컨텍스트 포인터.
    * 설정자: 생성자 파라미터로 수신.
    * 읽는 자: get_context(). clGetCommandQueueInfo (CL_QUEUE_CONTEXT 쿼리).
    * 값 범위: clCreateCommandQueue 호출자가 제공한 유효한 _cl_context*. NULL 불가.
    * 동기화: 불변이므로 락 불필요. */

   cl_device_id                   m_device;
   /* [한국어] 이 커맨드 큐가 연결된 GPU 디바이스 포인터.
    * 설정자: 생성자 파라미터로 수신.
    * 읽는 자: get_device(). clEnqueueReadBuffer/WriteBuffer/CopyBuffer에서 GPU memcpy 수행 시.
    * 값 범위: GPGPUSim_Init()이 반환한 유효한 _cl_device_id*. NULL 불가.
    * 동기화: 불변이므로 락 불필요. */

   cl_command_queue_properties    m_properties;
   /* [한국어] 커맨드 큐 생성 시 지정된 속성 비트마스크.
    * 설정자: 생성자 파라미터로 수신.
    * 읽는 자: get_properties(). clGetCommandQueueInfo (CL_QUEUE_PROPERTIES 쿼리).
    * 값 범위: CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE 조합.
    *          GPGPU-Sim은 두 속성 모두 경고 후 무시한다.
    * 동기화: 불변이므로 락 불필요. */
};

/*
 * [한국어] _cl_mem: OpenCL 메모리 객체. GPU 디바이스 메모리 또는 호스트 메모리 영역을 나타낸다.
 *
 * clCreateBuffer()로 생성되며, flags에 따라 세 가지 동작을 한다:
 *   - CL_MEM_USE_HOST_PTR 또는 CL_MEM_ALLOC_HOST_PTR: 호스트 메모리를 직접 사용 (m_is_on_host=true)
 *   - 그 외: GPGPU-Sim의 gpu_malloc()으로 GPU 시뮬레이션 메모리 할당 (m_device_ptr에 주소 저장)
 * device_ptr()은 GPU 메모리 주소를, host_ptr()은 호스트 메모리 주소를 반환한다.
 */
struct _cl_mem {
   _cl_mem( cl_mem_flags flags, size_t size , void *host_ptr, cl_int *errcode_ret, cl_device_id gpu );  // [한국어] 생성자: flags에 따라 호스트/GPU 메모리 할당 및 초기 데이터 복사
   cl_mem device_ptr();   // [한국어] GPU 메모리 주소를 cl_mem으로 캐스팅하여 반환 — memcpy_to/from_gpu에 전달
   void* host_ptr();      // [한국어] 호스트 메모리 포인터 반환 — CL_MEM_USE_HOST_PTR인 경우에만 유효
   bool is_on_host() { return m_is_on_host; }  // [한국어] 이 버퍼가 호스트 메모리에 있는지 여부 — clEnqueueCopyBuffer에서 memcpy 방향 결정에 사용
private:
   bool m_is_on_host;
   /* [한국어] 이 메모리 객체가 호스트 측에 존재하는지 여부.
    * 설정자: 생성자에서 CL_MEM_USE_HOST_PTR 또는 CL_MEM_ALLOC_HOST_PTR 플래그 시 true.
    * 읽는 자: is_on_host(). clEnqueueCopyBuffer에서 memcpy 방향(HtoD/DtoH/DtoD) 결정.
    * 값 범위: true (호스트 메모리 기반), false (GPU 디바이스 메모리 기반).
    * 동기화: 생성 후 불변. */

   size_t m_device_ptr;
   /* [한국어] GPU 시뮬레이션 디바이스 메모리 주소 (size_t로 저장되는 정수 주소).
    * 설정자: 생성자에서 gpu->the_device()->gpu_malloc(size) 반환값으로 설정.
    *         호스트 메모리 기반인 경우 0으로 유지.
    * 읽는 자: device_ptr()이 (cl_mem)(void*)m_device_ptr로 캐스팅하여 반환.
    *          clEnqueueReadBuffer/WriteBuffer/CopyBuffer에서 GPU 메모리 접근 주소로 사용.
    * 값 범위: gpu_malloc()이 반환한 유효한 시뮬레이션 메모리 주소 (0이 아님). 호스트 기반이면 0.
    * 동기화: 생성 후 불변. GPU 시뮬레이션 스레드가 이 주소를 메모리 접근 대상으로 사용. */

   void *m_host_ptr;
   /* [한국어] 호스트 측 메모리 포인터.
    * 설정자: CL_MEM_USE_HOST_PTR이면 사용자가 제공한 host_ptr로, CL_MEM_ALLOC_HOST_PTR이면
    *         malloc(size)로 할당된 포인터. 그 외에는 사용자 제공 host_ptr (초기 데이터 복사용).
    * 읽는 자: host_ptr(). clEnqueueMapBuffer에서 반환.
    * 값 범위: 유효한 호스트 포인터 또는 NULL (GPU 전용 버퍼인 경우).
    * 동기화: 생성 후 불변. 호스트 스레드에서만 접근. */

   cl_mem_flags m_flags;
   /* [한국어] 버퍼 생성 시 지정된 OpenCL 메모리 플래그 비트마스크.
    * 설정자: 생성자 파라미터 flags로 초기화.
    * 읽는 자: 현재 생성자 내부 로직에서만 사용. 이후 조회 용도로 보존.
    * 값 범위: CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR |
    *          CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY 조합.
    * 동기화: 생성 후 불변. */

   size_t m_size;
   /* [한국어] 이 메모리 객체의 크기 (바이트 단위).
    * 설정자: 생성자 파라미터 size로 초기화.
    * 읽는 자: 현재 직접 외부 접근 없음. 생성 시 gpu_malloc, memcpy_to_gpu에 전달.
    * 값 범위: 1 이상의 바이트 수.
    * 동기화: 생성 후 불변. */
};

/*
 * [한국어] pgm_info: 하나의 OpenCL 소스 파일(또는 문자열)에 대응하는 컴파일 정보 구조체.
 *
 * _cl_program은 여러 소스 문자열을 받을 수 있으며, 각 소스에 하나의 pgm_info가 대응한다.
 * Build() 호출 후 m_asm(PTX 어셈블리)과 m_symtab, m_kernels가 채워진다.
 */
struct pgm_info {
   std::string   m_source;
   /* [한국어] OpenCL C 소스 코드 문자열.
    * 설정자: _cl_program 생성자에서 clCreateProgramWithSource로 전달된 strings[]를 복사.
    * 읽는 자: _cl_program::Build()에서 임시 .cl 파일로 기록.
    * 값 범위: 유효한 OpenCL C 소스 텍스트.
    * 동기화: Build() 호출 이전에 설정되고 이후 불변. */

   std::string   m_asm;
   /* [한국어] nvopencl_wrapper가 생성한 PTX 어셈블리 문자열.
    * 설정자: _cl_program::Build()에서 PTX 파일을 읽어 info.m_asm = tmp로 설정.
    * 읽는 자: _cl_program::get_ptx(), get_ptx_size()에서 전체 PTX 반환 시 연결.
    * 값 범위: 유효한 PTX 소스 텍스트 ('}' 이후 트레일링 문자 제거됨).
    * 동기화: Build() 내에서 설정, 이후 불변. */

   class symbol_table *m_symtab;
   /* [한국어] PTX 심볼 테이블 포인터 — PTX 파싱 후 변수/함수 심볼 정보 보관.
    * 설정자: gpgpu_ptx_sim_load_ptx_from_string()이 반환한 symbol_table* 포인터.
    * 읽는 자: 현재 외부에서 직접 접근하지 않음 (CreateKernel이 m_kernels를 사용).
    * 값 범위: 유효한 symbol_table* (PTX 파싱 성공 시), NULL이면 오류.
    * 동기화: Build() 내에서 설정, 이후 불변. */

   std::map<std::string,function_info*> m_kernels;
   /* [한국어] 커널 이름 → function_info* 매핑. PTX 파싱으로 발견된 모든 커널 함수.
    * 설정자: register_ptx_function()에서 sg_info->m_kernels[name] = impl로 삽입.
    *         ptxinfo_addinfo()가 ptxinfo_opencl_addinfo()를 통해 메타데이터 보강.
    * 읽는 자: _cl_program::CreateKernel()에서 kernel_name으로 조회.
    * 값 범위: PTX 내의 .visible .entry 함수들이 등록됨. 빈 맵은 빌드 실패를 의미.
    * 동기화: Build() 내에서 설정, CreateKernel() 호출 전 불변. */
};

/*
 * [한국어] _cl_program: OpenCL 프로그램 객체. 소스 코드 수집 → PTX 컴파일 → 커널 생성을 관리.
 *
 * clCreateProgramWithSource()로 생성하여 소스를 m_pgm에 저장하고, clBuildProgram()이
 * Build()를 호출하면 nvopencl_wrapper(NVIDIA 드라이버 기반 변환 도구)를 통해 PTX를 생성한다.
 * 이후 clCreateKernel()이 CreateKernel()을 호출하여 특정 커널 함수의 _cl_kernel 객체를 반환한다.
 */
struct _cl_program {
   _cl_program( cl_context context,
                cl_uint           count,
             const char **     strings,
             const size_t *    lengths );   // [한국어] 생성자: OpenCL 소스 문자열 count개를 m_pgm에 저장
   void Build(const char *options);         // [한국어] OpenCL C → PTX 컴파일 실행. nvopencl_wrapper 또는 사전 추출 PTX 사용
   cl_kernel CreateKernel( const char *kernel_name, cl_int *errcode_ret );  // [한국어] 빌드된 PTX에서 특정 커널 이름으로 _cl_kernel 객체 생성
   cl_context get_context() { return m_context; }  // [한국어] 이 프로그램이 속한 컨텍스트 반환
   char *get_ptx();        // [한국어] 빌드된 전체 PTX 문자열 반환 (힙 할당, 호출자가 해제 책임)
   size_t get_ptx_size();  // [한국어] 빌드된 전체 PTX 문자열의 바이트 크기 반환

private:
   cl_context m_context;
   /* [한국어] 이 프로그램이 속한 OpenCL 컨텍스트.
    * 설정자: 생성자 파라미터 context로 초기화.
    * 읽는 자: get_context(). clGetProgramInfo(CL_PROGRAM_CONTEXT) 쿼리 처리 시.
    * 값 범위: 유효한 _cl_context* 포인터. NULL 불가.
    * 동기화: 생성 후 불변. */

   std::map<cl_uint,pgm_info> m_pgm;
   /* [한국어] 소스 번호(m_kernels_compiled 시점 값) → pgm_info 매핑.
    * 설정자: 생성자에서 각 소스 문자열마다 m_pgm[m_kernels_compiled] = info 삽입.
    * 읽는 자: Build()에서 각 소스를 순회하며 PTX 컴파일.
    *          get_ptx()/get_ptx_size()에서 모든 PTX를 연결.
    *          CreateKernel()에서 m_pgm 순회하며 커널 검색.
    * 값 범위: 1개 이상의 pgm_info 항목. 각 항목은 소스 문자열 하나에 대응.
    * 동기화: 생성자에서 삽입, Build() 호출 전까지 불변. Build() 내에서 m_asm 등 채워짐. */

   static unsigned m_kernels_compiled;
   /* [한국어] 전체 _cl_program 인스턴스에 걸쳐 컴파일된 소스 수 (전역 카운터).
    * 설정자: 생성자에서 소스 하나 처리 후 ++m_kernels_compiled.
    * 읽는 자: 각 pgm_info의 키 값 (m_pgm의 인덱스로 사용).
    *          Build()에서 ptx_fname 생성 시 source_num으로 사용 (PTX_SIM_USE_PTX_FILE 경로).
    * 값 범위: 0부터 시작하는 단조 증가 정수. 여러 _cl_program 인스턴스가 이 카운터를 공유.
    * 동기화: 단일 호스트 초기화 흐름에서만 접근 → 락 불필요. */
};

/*
 * [한국어] _cl_kernel: OpenCL 커널 객체. PTX 함수 구현체(function_info*)와 커널 인자 목록을 보관.
 *
 * clCreateKernel()로 생성하여 _cl_program에서 커널 이름으로 찾은 function_info*를 저장한다.
 * clSetKernelArg()가 SetKernelArg()를 호출하여 인자를 m_args에 저장하고,
 * clEnqueueNDRangeKernel()이 bind_args()를 호출하여 gpgpu_ptx_sim_arg_list_t로 변환한다.
 */
struct _cl_kernel {
   _cl_kernel( cl_program prog, const char* kernel_name, class function_info *kernel_impl );  // [한국어] 생성자: 프로그램, 커널 이름, PTX function_info* 포인터로 초기화
   void SetKernelArg(         // [한국어] clSetKernelArg 대응: arg_index 위치에 인자 크기와 값을 m_args에 저장
      cl_uint      arg_index,
      size_t       arg_size,
      const void * arg_value );
   cl_int bind_args( gpgpu_ptx_sim_arg_list_t &arg_list );  // [한국어] m_args를 순차 검사하여 gpgpu_ptx_sim_arg_list_t로 변환. 오프셋 정렬 처리
   std::string name() const { return m_kernel_name; }            // [한국어] 커널 이름 문자열 반환 — clEnqueueNDRangeKernel 로그 출력에 사용
   size_t get_workgroup_size(cl_device_id device);               // [한국어] 레지스터 수 기반 최적 워크그룹 크기 계산 — CL_KERNEL_WORK_GROUP_SIZE 쿼리 응답
   cl_program get_program() { return m_prog; }                   // [한국어] 이 커널이 속한 프로그램 반환
   class function_info *get_implementation() { return m_kernel_impl; }  // [한국어] PTX IR 함수 객체 반환 — clEnqueueNDRangeKernel에서 PTX 버전 확인, PDOM 분석에 사용
private:
   unsigned m_uid;
   /* [한국어] 이 커널 객체의 고유 식별자.
    * 설정자: 생성자에서 sm_context_uid++ 값으로 할당.
    * 읽는 자: 현재 외부에서 직접 접근하지 않음. 디버깅/추적 목적.
    * 값 범위: 0부터 시작하는 단조 증가 정수.
    * 동기화: 단일 호스트 스레드에서만 접근 → 락 불필요. */

   static unsigned sm_context_uid;
   /* [한국어] _cl_kernel 인스턴스 전체에 걸친 UID 카운터 (static). _cl_context::sm_context_uid와 별개.
    * 설정자: 생성자에서 m_uid = sm_context_uid++로 소비.
    * 읽는 자: m_uid 필드에 복사됨.
    * 값 범위: 0부터 단조 증가. 초기값은 파일 하단 `unsigned _cl_kernel::sm_context_uid = 0;` 참조.
    * 동기화: 단일 호스트 스레드 → 락 불필요. */

   cl_program m_prog;
   /* [한국어] 이 커널이 소속된 _cl_program 객체 포인터.
    * 설정자: 생성자 파라미터 prog로 초기화.
    * 읽는 자: get_program().
    * 값 범위: 유효한 _cl_program* 포인터. NULL 불가.
    * 동기화: 불변. */

   std::string m_kernel_name;
   /* [한국어] 커널 함수의 이름 문자열 (PTX .entry 함수 이름과 동일).
    * 설정자: 생성자에서 std::string(kernel_name)으로 초기화.
    * 읽는 자: name(). clEnqueueNDRangeKernel의 printf 로그 메시지에서 사용.
    * 값 범위: 유효한 C 식별자 문자열 (PTX 커널 이름).
    * 동기화: 불변. */

   struct arg_info {
      size_t m_arg_size;
      /* [한국어] 이 커널 인자의 바이트 크기.
       * 설정자: SetKernelArg()에서 arg.m_arg_size = arg_size로 설정.
       * 읽는 자: bind_args()에서 오프셋 계산 및 gpgpu_ptx_sim_arg 생성 시 사용.
       * 값 범위: 4(int/float), 8(double/포인터), 1~16 등 실제 인자 타입에 따라 다름.
       * 동기화: SetKernelArg() 후 bind_args() 전까지 불변. 호스트 스레드 전용. */

      const void *m_arg_value;
      /* [한국어] 이 커널 인자의 값을 가리키는 포인터 (호스트 메모리).
       * 설정자: SetKernelArg()에서 arg.m_arg_value = arg_value로 설정. 원본 포인터 저장.
       * 읽는 자: bind_args()에서 gpgpu_ptx_sim_arg(arg.m_arg_value, ...) 생성 시 전달.
       * 값 범위: 사용자가 제공한 인자 값의 주소 (cl_mem 포인터, int, float 등). NULL이면 로컬 메모리 인자.
       * 동기화: SetKernelArg() 이후 불변. 원본 데이터의 수명은 호출자 책임. */
   };

   std::map<unsigned, arg_info> m_args;
   /* [한국어] 커널 인자 인덱스 → arg_info 매핑. clSetKernelArg()가 호출될 때마다 삽입/갱신.
    * 설정자: SetKernelArg()에서 m_args[arg_index] = arg로 삽입.
    * 읽는 자: bind_args()에서 인덱스 순서대로 순회하며 인자를 PTX 시뮬레이터 형식으로 변환.
    * 값 범위: 0부터 (커널 인자 개수 - 1)까지의 연속 인덱스. 연속적이지 않으면 CL_INVALID_KERNEL_ARGS 반환.
    * 동기화: 호스트 스레드에서 순차적으로 설정. bind_args() 호출 전에 모든 인자가 설정되어야 함. */

   class function_info *m_kernel_impl;
   /* [한국어] 이 커널의 PTX IR 함수 객체 포인터 (기능 시뮬레이션 엔진의 내부 표현).
    * 설정자: 생성자 파라미터 kernel_impl로 초기화 (_cl_program::CreateKernel이 m_kernels에서 찾은 값).
    * 읽는 자: get_implementation(). clEnqueueNDRangeKernel에서 PTX 버전 확인, PDOM 분석, 커널 실행에 사용.
    *          bind_args()에서 get_arg()로 인자 심볼 타입 정보 획득.
    *          get_workgroup_size()에서 ptx_kernel_nregs()로 레지스터 수 조회.
    * 값 범위: _cl_program::Build() 후 유효한 function_info* 포인터. NULL이면 커널 미존재.
    * 동기화: 불변. GPU 시뮬레이션 스레드도 읽지만 쓰기는 없음. */
};

/*
 * [한국어] _cl_platform_id: OpenCL 플랫폼 ID 객체. GPGPU-Sim은 단일 플랫폼만 제공한다.
 *
 * clGetPlatformIDs()가 반환하는 cl_platform_id는 이 구조체의 전역 인스턴스(g_gpgpu_sim_platform_id)를 가리킨다.
 * m_uid=0으로 GPGPU-Sim의 유일한 플랫폼을 식별한다. clCreateContext에서 properties를 검증할 때 사용.
 */
struct _cl_platform_id {
   static const unsigned m_uid = 0;
   /* [한국어] GPGPU-Sim 플랫폼의 고유 ID (항상 0). static const이므로 인스턴스 없이도 접근 가능.
    * 설정자: 컴파일 시 상수로 정의됨.
    * 읽는 자: clGetPlatformInfo()에서 platform->m_uid != 0 검증.
    *          clCreateContext()에서 properties[1] == &g_gpgpu_sim_platform_id 검증.
    *          clGetDeviceIDs()에서 platform->m_uid != 0 검증.
    * 값 범위: 항상 0 (GPGPU-Sim은 플랫폼 1개만 지원).
    * 동기화: 컴파일 시 상수이므로 동기화 불필요. */
};

/* [한국어] g_gpgpu_sim_platform_id: GPGPU-Sim의 유일한 OpenCL 플랫폼 객체 전역 인스턴스.
 * clGetPlatformIDs()가 platforms[0] = &g_gpgpu_sim_platform_id를 반환하여
 * 애플리케이션이 이 주소를 cl_platform_id로 사용한다. */
struct _cl_platform_id g_gpgpu_sim_platform_id;

/*
 * [한국어]
 * gpgpusim_exit - GPGPU-Sim 시뮬레이터 비정상 종료 함수
 *
 * @return: 없음 (noreturn — abort()는 절대 반환하지 않음)
 *
 * 복구 불가능한 에러 발생 시 프로세스를 즉시 종료한다. SIGABRT 시그널을 발생시켜
 * core dump를 생성하므로 디버깅 시 스택 추적이 가능하다.
 *
 * 호출 체인:
 *   gpgpusim_opencl_error → [gpgpusim_exit] → abort()
 */
void gpgpusim_exit()
{
   abort();  // [한국어] SIGABRT 발생 — core dump 생성, 프로세스 즉시 종료
}

/*
 * [한국어]
 * gpgpusim_opencl_warning - OpenCL API 경고 메시지 출력 (실행 계속)
 *
 * @func: 경고가 발생한 함수 이름 (보통 __my_func__ 매크로로 전달)
 * @line: 경고 발생 소스 라인 번호 (__LINE__ 매크로로 전달)
 * @desc: 경고 설명 문자열
 * @return: 없음
 *
 * 미구현이지만 무시 가능한 기능(비동기 I/O, 프로파일링 등)에 대해 경고를 출력하고
 * 실행을 계속한다. 에러와 달리 프로세스를 종료하지 않는다.
 *
 * 호출 체인:
 *   clCreateCommandQueue / clEnqueueReadBuffer / clEnqueueWriteBuffer 등 → [gpgpusim_opencl_warning]
 */
void gpgpusim_opencl_warning( const char* func, unsigned line, const char *desc )
{
   printf("GPGPU-Sim OpenCL API: Warning (%s:%u) ** %s\n", func,line,desc);  // [한국어] 함수명, 라인 번호, 설명을 포함한 경고 메시지 출력
}

/*
 * [한국어]
 * gpgpusim_opencl_error - OpenCL API 치명적 에러 출력 및 프로세스 종료
 *
 * @func: 에러가 발생한 함수 이름
 * @line: 에러 발생 소스 라인 번호
 * @desc: 에러 설명 문자열
 * @return: 없음 (noreturn)
 *
 * 복구 불가능한 OpenCL API 에러 발생 시 메시지를 출력하고 gpgpusim_exit()를 호출한다.
 * OpenCL 에러 코드로 표현할 수 없는 내부 오류나 미지원 기능 호출 시 사용된다.
 *
 * 호출 체인:
 *   (다양한 OpenCL API 함수) → [gpgpusim_opencl_error] → gpgpusim_exit → abort()
 */
void gpgpusim_opencl_error( const char* func, unsigned line, const char *desc )
{
   printf("GPGPU-Sim OpenCL API: ERROR (%s:%u) ** %s\n", func,line,desc);  // [한국어] 에러 위치와 설명을 stderr 대신 stdout에 출력 (flush 없이)
   gpgpusim_exit();  // [한국어] abort()를 통해 프로세스 즉시 종료 (core dump 생성)
}

/*
 * [한국어]
 * _cl_kernel::_cl_kernel - OpenCL 커널 객체 생성자
 *
 * @prog:        이 커널이 속한 _cl_program 객체
 * @kernel_name: PTX .entry 함수 이름 (null-terminated C 문자열)
 * @kernel_impl: PTX 파싱으로 생성된 function_info 포인터 (실제 커널 구현)
 * @return:      없음 (생성자)
 *
 * _cl_program::CreateKernel()에서 호출된다. 고유 UID를 할당하고 커널 이름과
 * function_info 포인터를 보관한다.
 *
 * 호출 체인:
 *   clCreateKernel → _cl_program::CreateKernel → [_cl_kernel 생성자]
 */
_cl_kernel::_cl_kernel( cl_program prog, const char* kernel_name, class function_info *kernel_impl )
{
   m_uid = sm_context_uid++;                  // [한국어] 전역 카운터에서 고유 UID 할당 후 카운터 증가
   m_kernel_name = std::string(kernel_name);  // [한국어] C 문자열을 std::string으로 복사 — 원본 포인터의 수명과 독립
   m_kernel_impl = kernel_impl;               // [한국어] PTX function_info* 포인터 저장 — 커널 실행 및 인자 정보 접근에 사용
   m_prog = prog;                             // [한국어] 소속 프로그램 포인터 저장
}

/*
 * [한국어]
 * _cl_kernel::SetKernelArg - 커널 인자 저장
 *
 * @arg_index: 커널 함수 파라미터 인덱스 (0부터 시작)
 * @arg_size:  인자의 바이트 크기
 * @arg_value: 인자 값을 가리키는 호스트 포인터. NULL이면 로컬 메모리 인자
 * @return:    없음 (void)
 *
 * clSetKernelArg()의 내부 구현. m_args에 arg_index → {size, value} 쌍을 저장한다.
 * 실제 인자 값을 복사하지 않고 포인터만 보관하므로, 원본 데이터는 clEnqueueNDRangeKernel
 * 호출 전까지 유효해야 한다.
 *
 * 호출 체인:
 *   clSetKernelArg → [SetKernelArg] → m_args[arg_index] = arg
 */
void _cl_kernel::SetKernelArg(
      cl_uint      arg_index,
      size_t       arg_size,
      const void * arg_value )
{
   arg_info arg;                    // [한국어] 로컬 arg_info 구조체 임시 생성
   arg.m_arg_size = arg_size;       // [한국어] 인자 바이트 크기 저장
   arg.m_arg_value = arg_value;     // [한국어] 인자 값 포인터 저장 (값 복사 아님)
   m_args[arg_index] = arg;         // [한국어] 인덱스를 키로 m_args에 삽입 또는 갱신
}

/*
 * [한국어]
 * _cl_kernel::bind_args - 저장된 커널 인자를 PTX 시뮬레이터 형식으로 변환
 *
 * @arg_list: 출력 파라미터 — gpgpu_ptx_sim_arg 항목들의 링크드 리스트 (비어있어야 함)
 * @return:   CL_SUCCESS (인자 목록 정상 변환),
 *            CL_INVALID_KERNEL_ARGS (인자 인덱스가 불연속적인 경우)
 *
 * m_args를 인덱스 순서대로 순회하며, 각 인자의 타입 정보(PTX 심볼 테이블에서 획득)로
 * 정렬(alignment) 패딩을 계산하고, gpgpu_ptx_sim_arg를 arg_list 앞쪽에 삽입한다.
 * push_front를 사용하므로 arg_list는 역순이 된다 (PTX 시뮬레이터가 역순으로 처리).
 *
 * 호출 체인:
 *   clEnqueueNDRangeKernel → kernel->bind_args(params) → [bind_args] → arg_list 완성
 *                          → gpgpu_opencl_ptx_sim_init_grid(params) → PTX 실행 엔진
 */
cl_int _cl_kernel::bind_args( gpgpu_ptx_sim_arg_list_t &arg_list )
{
   size_t offset = 0;  // [한국어] 파라미터 메모리 영역 내 현재 오프셋 (바이트 단위)

   assert( arg_list.empty() );  // [한국어] 빈 리스트로 시작해야 함을 보장 — 중복 호출 방지
   unsigned k=0;                // [한국어] 예상 인자 인덱스 카운터 (0부터 연속적이어야 함)
   std::map<unsigned, arg_info>::iterator i;
   for( i = m_args.begin(); i!=m_args.end(); i++ ) {  // [한국어] m_args를 인덱스 오름차순으로 순회
      if( i->first != k )                              // [한국어] 인덱스가 불연속이면 오류 — clSetKernelArg가 연속 호출되지 않은 경우
         return CL_INVALID_KERNEL_ARGS;                // [한국어] OpenCL 스펙: 모든 인자가 설정되어야 커널 실행 가능

      arg_info arg = i->second;                                     // [한국어] 현재 인자의 크기와 값 포인터 복사
      const symbol *sym = m_kernel_impl->get_arg(i->first);         // [한국어] PTX 심볼 테이블에서 i->first 번째 파라미터의 심볼 객체 획득
      const type_info_key &t = sym->type()->get_key();              // [한국어] 파라미터의 타입 정보 키 획득 — 정렬 스펙 포함

      /* [한국어] 정렬 요구사항 계산: PTX 타입에서 alignment_spec이 지정된 경우 사용,
       * 지정되지 않은 경우(-1) 인자 크기 자체를 정렬 기준으로 사용 (자연 정렬). */
      int align = (t.get_alignment_spec() == -1) ? arg.m_arg_size : t.get_alignment_spec();
      if( offset % align )                              // [한국어] 현재 오프셋이 정렬 경계에 있지 않으면
         offset += (align - (offset % align));          // [한국어] 다음 정렬 경계까지 패딩 추가

      gpgpu_ptx_sim_arg param( arg.m_arg_value, arg.m_arg_size, offset );  // [한국어] PTX 시뮬레이터용 인자 패킷 생성: (값 포인터, 크기, 오프셋)
      arg_list.push_front( param );  // [한국어] 리스트 앞쪽에 삽입 — PTX 시뮬레이터는 역순 리스트로 인자를 처리

      offset += arg.m_arg_size;  // [한국어] 다음 인자를 위한 오프셋 전진
      k++;                       // [한국어] 예상 인덱스 카운터 증가
   }
   return CL_SUCCESS;  // [한국어] 모든 인자 정상 변환 완료
}

/* [한국어] min(a,b): 두 값 중 작은 값을 반환하는 매크로.
 * 표준 헤더의 min이 C++에서 항상 사용 가능하지 않고, <algorithm>의 std::min과 충돌을
 * 피하기 위해 로컬에 정의한다. get_workgroup_size()에서만 사용. */
#define min(a,b) ((a<b)?(a):(b))

/*
 * [한국어]
 * _cl_kernel::get_workgroup_size - 커널의 최적 워크그룹 크기 계산
 *
 * @device: 실행 대상 GPU 디바이스 — SM당 레지스터 수와 스레드 수 쿼리에 사용
 * @return: 권장 로컬 워크그룹 크기 (size_t). clGetKernelWorkGroupInfo(CL_KERNEL_WORK_GROUP_SIZE) 응답에 사용
 *
 * PTX 커널이 사용하는 레지스터 수(nregs)를 기반으로 SM에 동시 상주 가능한 최대 스레드 수를 계산한다.
 * result_regs = SM 총 레지스터 / (nregs를 4의 배수로 올림) 으로 레지스터 제한 워크그룹 크기를 구하고,
 * SM의 물리적 최대 스레드 수와의 최솟값을 반환한다.
 *
 * 호출 체인:
 *   clGetKernelWorkGroupInfo(CL_KERNEL_WORK_GROUP_SIZE) → [get_workgroup_size]
 */
size_t _cl_kernel::get_workgroup_size(cl_device_id device)
{
   unsigned nregs = ptx_kernel_nregs( m_kernel_impl );   // [한국어] 이 커널의 PTX 함수가 사용하는 레지스터 수 조회
   unsigned result_regs = (unsigned)-1;                   // [한국어] 레지스터 제한 없음을 나타내는 초기값 (최대 unsigned)
   if( nregs > 0 )
      /* [한국어] SM당 레지스터 총수를 (nregs를 4의 배수로 올림)으로 나눠 최대 스레드 수 계산.
       * (nregs+3)&~3 는 nregs를 4의 배수로 올림 (예: 5 → 8, 4 → 4). GPU 레지스터 파일은
       * 4개 단위로 할당되므로 이 올림이 필요하다. */
      result_regs = device->the_device()->num_registers_per_core() / ((nregs+3)&~3);
   unsigned result = device->the_device()->threads_per_core();  // [한국어] SM의 물리적 최대 스레드 수 (gpgpusim.config의 n_thread_per_shader)
   result = min(result, result_regs);  // [한국어] 레지스터 제한값과 물리적 최대값 중 더 작은 값 선택
   return (size_t)result;              // [한국어] 권장 워크그룹 크기 반환
}

/*
 * [한국어]
 * _cl_mem::device_ptr - GPU 디바이스 메모리 주소를 cl_mem 타입으로 반환
 *
 * @return: m_device_ptr를 (cl_mem)(void*)로 캐스팅한 값.
 *          GPU 메모리가 할당되지 않은 경우 NULL (호스트 메모리 기반 버퍼).
 *
 * GPGPU-Sim에서 cl_mem은 실제로 GPU 시뮬레이션 메모리의 정수 주소를 포인터로 재해석한 값이다.
 *
 * 호출 체인:
 *   _cl_context::CreateBuffer → [device_ptr] → m_devptr_to_cl_mem 등록
 *   clEnqueueReadBuffer / clEnqueueWriteBuffer → buffer → [device_ptr] 암묵적 사용
 */
cl_mem _cl_mem::device_ptr()
{
   cl_mem result = (cl_mem)(void*)m_device_ptr;  // [한국어] size_t 정수 주소를 void*를 경유해 cl_mem 포인터로 재해석
   return result;
}

/*
 * [한국어]
 * _cl_mem::host_ptr - 호스트 메모리 포인터 반환
 *
 * @return: m_host_ptr. CL_MEM_USE_HOST_PTR 또는 CL_MEM_ALLOC_HOST_PTR인 경우 유효.
 *          GPU 전용 버퍼인 경우 NULL.
 *
 * 호출 체인:
 *   clEnqueueMapBuffer → lookup_mem → [host_ptr]
 *   clEnqueueCopyBuffer → src/dst->host_ptr() (호스트↔GPU 방향 결정)
 */
void* _cl_mem::host_ptr()
{
   return m_host_ptr;  // [한국어] 호스트 메모리 포인터 직접 반환
}

/*
 * [한국어]
 * _cl_mem::_cl_mem - OpenCL 메모리 객체 생성자
 *
 * @flags:       CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR 등의 조합
 * @size:        메모리 영역 크기 (바이트)
 * @host_ptr:    호스트 메모리 포인터 (CL_MEM_USE_HOST_PTR이면 직접 사용, CL_MEM_COPY_HOST_PTR이면 데이터 복사)
 * @errcode_ret: 에러 코드 출력 포인터 (NULL 가능)
 * @gpu:         시뮬레이션 GPU 디바이스 — gpu_malloc(), memcpy_to_gpu() 호출에 사용
 * @return:      없음 (생성자)
 *
 * flags에 따라 세 가지 경로로 처리된다:
 *   1. CL_MEM_USE_HOST_PTR: m_host_ptr = host_ptr, m_is_on_host = true
 *   2. CL_MEM_ALLOC_HOST_PTR: m_host_ptr = malloc(size), m_is_on_host = true
 *   3. 그 외: m_device_ptr = gpu_malloc(size), host_ptr 있으면 memcpy_to_gpu()
 *
 * 호출 체인:
 *   _cl_context::CreateBuffer → [_cl_mem 생성자] → gpu->the_device()->gpu_malloc()
 */
_cl_mem::_cl_mem(
   cl_mem_flags flags,
   size_t       size ,
   void *       host_ptr,
   cl_int *     errcode_ret,
   cl_device_id gpu )
{
   setErrCode( errcode_ret, CL_SUCCESS );  // [한국어] 초기 에러 코드를 CL_SUCCESS로 설정

   m_is_on_host = false;      // [한국어] 기본값: GPU 메모리 기반 (아래에서 flags에 따라 갱신)
   m_flags = flags;           // [한국어] 메모리 플래그 저장
   m_size = size;             // [한국어] 크기 저장
   m_host_ptr = host_ptr;     // [한국어] 초기 호스트 포인터 저장 (NULL일 수 있음)
   m_device_ptr = 0;          // [한국어] 디바이스 포인터 초기화 (0 = 미할당)

   /* [한국어] 에러 검사: CL_MEM_USE_HOST_PTR 또는 CL_MEM_COPY_HOST_PTR을 지정했는데
    * host_ptr이 NULL인 경우 — OpenCL 스펙 §5.2.1 위반 */
   if( (flags & (CL_MEM_USE_HOST_PTR|CL_MEM_COPY_HOST_PTR)) && host_ptr == NULL ) {
      setErrCode( errcode_ret, CL_INVALID_HOST_PTR );  // [한국어] host_ptr이 필요한데 NULL → CL_INVALID_HOST_PTR 반환
      return;
   }
   /* [한국어] 에러 검사: CL_MEM_COPY_HOST_PTR과 CL_MEM_USE_HOST_PTR은 상호 배타적 — 동시 사용 불가 */
   if( (flags & CL_MEM_COPY_HOST_PTR) && (flags & CL_MEM_USE_HOST_PTR) ) {
      setErrCode( errcode_ret, CL_INVALID_VALUE );  // [한국어] 두 플래그 동시 사용 → CL_INVALID_VALUE 반환
      return;
   }
   /* [한국어] CL_MEM_ALLOC_HOST_PTR: GPGPU-Sim이 호스트 메모리를 새로 할당하여 버퍼로 사용.
    * host_ptr와 함께 사용하는 경우는 현재 미지원이므로 에러 출력. */
   if( flags & CL_MEM_ALLOC_HOST_PTR ) {
      if( host_ptr )
         gpgpusim_opencl_error(__my_func__,__LINE__," CL_MEM_ALLOC_HOST_PTR -- not yet supported/tested.\n");  // [한국어] host_ptr + CL_MEM_ALLOC_HOST_PTR 조합은 미구현
      m_host_ptr = malloc(size);  // [한국어] 호스트 측 메모리를 size 바이트 할당
   }

   /* [한국어] m_is_on_host 결정:
    * CL_MEM_USE_HOST_PTR 또는 CL_MEM_ALLOC_HOST_PTR이면 호스트 메모리 기반,
    * 그 외에는 GPU 시뮬레이션 메모리 할당. */
   if( flags & (CL_MEM_USE_HOST_PTR|CL_MEM_ALLOC_HOST_PTR) ) {
      m_is_on_host = true;   // [한국어] 호스트 메모리 직접 사용/할당 경로 — GPU malloc 불필요
   } else {
      m_is_on_host = false;  // [한국어] GPU 메모리 할당 경로 (아래에서 처리)
   }
   /* [한국어] GPU 메모리 할당 및 초기 데이터 복사:
    * USE_HOST_PTR/ALLOC_HOST_PTR이 아닌 경우에만 GPU 메모리를 할당한다.
    * CL_MEM_COPY_HOST_PTR인 경우 host_ptr의 데이터를 GPU 메모리로 복사한다. */
   if( !(flags & (CL_MEM_USE_HOST_PTR|CL_MEM_ALLOC_HOST_PTR)) ) {
      // if not allocating on host, then allocate GPU memory and make a copy
      m_device_ptr = (size_t) gpu->the_device()->gpu_malloc(size);  // [한국어] 시뮬레이션 GPU 메모리 할당 — 반환값은 시뮬레이션 주소 정수
      if( host_ptr )
         gpu->the_device()->memcpy_to_gpu( m_device_ptr, host_ptr, size );  // [한국어] CL_MEM_COPY_HOST_PTR인 경우: host_ptr에서 GPU 주소로 데이터 복사
   }
}

/*
 * [한국어]
 * _cl_context::_cl_context - OpenCL 컨텍스트 생성자
 *
 * @gpu: GPGPUSim_Init()이 반환한 _cl_device_id 포인터 — GPGPU-Sim의 시뮬레이션 GPU
 * @return: 없음 (생성자)
 *
 * clCreateContext / clCreateContextFromType에서 호출된다. 고유 UID를 할당하고
 * 디바이스 포인터를 m_gpu에 저장한다.
 *
 * 호출 체인:
 *   clCreateContext / clCreateContextFromType → GPGPUSim_Init() → [_cl_context 생성자]
 */
_cl_context::_cl_context( struct _cl_device_id *gpu )
{
   m_uid = sm_context_uid++;  // [한국어] 전역 카운터에서 고유 UID 할당 후 증가
   m_gpu = gpu;               // [한국어] 연결된 GPU 디바이스 포인터 저장
}

/*
 * [한국어]
 * _cl_context::get_first_device - 이 컨텍스트의 첫 번째 (유일한) GPU 디바이스 반환
 *
 * @return: m_gpu (_cl_device_id*). GPGPU-Sim에서는 항상 단일 디바이스.
 *
 * 호출 체인:
 *   clGetContextInfo(CL_CONTEXT_DEVICES) → [get_first_device] → device 리스트 구성
 */
cl_device_id _cl_context::get_first_device()
{
   return m_gpu;  // [한국어] 이 컨텍스트에 연결된 유일한 GPU 디바이스 포인터 반환
}

/*
 * [한국어]
 * _cl_context::CreateBuffer - OpenCL 버퍼 메모리 객체 생성 및 등록
 *
 * @flags:       CL_MEM_READ_WRITE 등 메모리 플래그
 * @size:        버퍼 크기 (바이트)
 * @host_ptr:    초기 데이터 또는 호스트 메모리 포인터 (NULL 가능)
 * @errcode_ret: 에러 코드 출력 포인터 (NULL 가능)
 * @return:      cl_mem — GPU 디바이스 메모리 주소(정수 주소 캐스팅) 또는 host_ptr.
 *               _cl_mem* 포인터 자체가 아님에 주의.
 *
 * _cl_mem 객체를 힙에 생성하고, 두 역방향 맵(devptr/hostptr)에 등록한다.
 * 반환값은 애플리케이션이 clEnqueueReadBuffer 등에서 buffer 파라미터로 전달하는 핸들이다.
 *
 * 호출 체인:
 *   clCreateBuffer → [CreateBuffer] → new _cl_mem → gpu_malloc
 */
cl_mem _cl_context::CreateBuffer(
               cl_mem_flags flags,
               size_t       size ,
               void *       host_ptr,
               cl_int *     errcode_ret )
{
   /* [한국어] 동일한 호스트 포인터로 이미 버퍼가 생성된 경우 경고 출력.
    * 중복 생성은 에러는 아니지만 메모리 낭비 또는 불일치를 유발할 수 있다. */
   if( host_ptr && (m_hostptr_to_cl_mem.find(host_ptr) != m_hostptr_to_cl_mem.end()) ) {
      printf("GPGPU-Sim OpenCL API: WARNING ** clCreateBuffer - buffer already created for this host variable\n");  // [한국어] 같은 host_ptr로 중복 버퍼 생성 경고
   }
   cl_mem result = new _cl_mem(flags,size,host_ptr,errcode_ret,m_gpu);  // [한국어] _cl_mem 객체 힙 할당 — 내부에서 gpu_malloc 또는 호스트 메모리 설정
   m_devptr_to_cl_mem[result->device_ptr()] = result;                   // [한국어] 디바이스 주소 → _cl_mem* 역방향 맵 등록 (lookup_mem에서 사용)
   if( host_ptr )
      m_hostptr_to_cl_mem[host_ptr] = result;                           // [한국어] 호스트 포인터 → _cl_mem* 역방향 맵 등록 (호스트 기반 버퍼 조회용)
   if( result->device_ptr() )
      return (cl_mem) result->device_ptr();  // [한국어] GPU 메모리가 할당된 경우: 디바이스 주소를 cl_mem 핸들로 반환
   else
      return (cl_mem) host_ptr;              // [한국어] 호스트 메모리 기반 버퍼: host_ptr을 cl_mem 핸들로 반환
}

/*
 * [한국어]
 * _cl_context::lookup_mem - cl_mem 핸들로 _cl_mem 객체 포인터 조회
 *
 * @m: clCreateBuffer가 반환한 cl_mem 핸들 (GPU 주소 또는 host_ptr)
 * @return: 대응하는 _cl_mem* 포인터. 없으면 NULL.
 *
 * 먼저 m_devptr_to_cl_mem(디바이스 포인터 맵)에서 검색하고, 없으면
 * m_hostptr_to_cl_mem(호스트 포인터 맵)에서 검색한다.
 * clEnqueueCopyBuffer에서 src/dst 버퍼의 실제 _cl_mem 객체를 찾기 위해 사용된다.
 *
 * 호출 체인:
 *   clEnqueueCopyBuffer → context->lookup_mem(src_buffer/dst_buffer) → [lookup_mem]
 *   clEnqueueMapBuffer → get_context()->lookup_mem(buffer) → [lookup_mem]
 */
cl_mem _cl_context::lookup_mem( cl_mem m )
{
   std::map<cl_mem/*device ptr*/,cl_mem>::iterator i=m_devptr_to_cl_mem.find(m);  // [한국어] 디바이스 포인터 맵에서 먼저 검색
   if( i == m_devptr_to_cl_mem.end() ) {  // [한국어] 디바이스 포인터 맵에 없는 경우 → 호스트 포인터 맵 시도
      void *t = (void*)m;                  // [한국어] cl_mem을 void* (호스트 포인터)로 재해석
      std::map<void*/*host_ptr*/,cl_mem>::iterator j = m_hostptr_to_cl_mem.find(t);  // [한국어] 호스트 포인터 맵 검색
      if( j == m_hostptr_to_cl_mem.end() )
         return NULL;        // [한국어] 양쪽 맵 모두에서 찾지 못함 — 유효하지 않은 cl_mem 핸들
      else
         return j->second;   // [한국어] 호스트 포인터 맵에서 _cl_mem* 반환
   } else {
      return i->second;      // [한국어] 디바이스 포인터 맵에서 _cl_mem* 반환
   }
}

/* [한국어] _cl_program::m_kernels_compiled: 전체 _cl_program 인스턴스에 걸쳐
 * 누적된 소스 파일 컴파일 수. 각 소스에 고유한 인덱스를 부여하기 위한 전역 카운터.
 * 초기값 0에서 시작. */
unsigned _cl_program::m_kernels_compiled = 0;

/*
 * [한국어]
 * _cl_program::_cl_program - OpenCL 프로그램 객체 생성자
 *
 * @context: 이 프로그램이 속할 OpenCL 컨텍스트
 * @count:   소스 문자열 개수
 * @strings: OpenCL C 소스 문자열 배열 (count개)
 * @lengths: 각 문자열의 길이 배열. NULL이거나 길이=0이면 strlen() 사용
 * @return:  없음 (생성자)
 *
 * clCreateProgramWithSource()의 내부 구현. 소스 문자열들을 복사하여 m_pgm에 저장한다.
 * 실제 컴파일(PTX 생성)은 clBuildProgram() → Build()에서 수행된다.
 *
 * 호출 체인:
 *   clCreateProgramWithSource → [_cl_program 생성자] → m_pgm에 소스 저장
 */
_cl_program::_cl_program( cl_context        context,
                          cl_uint           count,
                          const char **     strings,
                          const size_t *    lengths )
{
   m_context = context;  // [한국어] 연결된 OpenCL 컨텍스트 저장
   for( cl_uint i=0; i<count; i++ ) {  // [한국어] 각 소스 문자열을 처리
      unsigned len;
      /* [한국어] 문자열 길이 결정: lengths 배열이 제공되고 0보다 크면 그 값 사용,
       * 그렇지 않으면 null-terminated 문자열로 가정하고 strlen() 사용. */
      if(lengths != NULL and lengths[i] > 0)
          len = lengths[i];             // [한국어] 명시적 길이 사용 (null terminator가 없을 수 있음)
      else
          len = strlen(strings[i]);     // [한국어] null terminator 이전까지의 길이 계산
      char *tmp = (char*)malloc(len+1); // [한국어] 소스 문자열 + null terminator를 위한 메모리 할당
      memcpy(tmp,strings[i],len);       // [한국어] 소스 문자열 복사 (null terminator 미포함)
      tmp[len] = 0;                     // [한국어] null terminator 수동 추가
      m_pgm[m_kernels_compiled].m_source = tmp;  // [한국어] 전역 카운터 값을 키로 m_pgm에 소스 저장
      ++m_kernels_compiled;             // [한국어] 전역 소스 카운터 증가 (다음 소스에 다른 인덱스 부여)
      free(tmp);                        // [한국어] 임시 버퍼 해제 (m_source에 std::string으로 이미 복사됨)
   }
}

/* [한국어] sg_info: register_ptx_function()과 ptxinfo_addinfo()가 현재 처리 중인
 * pgm_info를 가리키는 전역 포인터. Build() 내에서 각 소스를 처리하기 전에 설정된다.
 * 단일 스레드에서 순차적으로 실행되므로 락 불필요. */
static pgm_info *sg_info;

/*
 * [한국어]
 * register_ptx_function - PTX 파싱 완료 후 커널 함수를 현재 pgm_info에 등록하는 콜백
 *
 * @name: PTX .entry 함수 이름 (커널 이름)
 * @impl: 파싱된 함수의 function_info* 포인터
 * @return: 없음
 *
 * PTX 파서(ptx_loader.cc의 gpgpu_ptx_sim_load_ptx_from_string)가 .entry를 발견할 때
 * 이 함수를 콜백으로 호출한다. sg_info가 현재 처리 중인 pgm_info를 가리킨다.
 *
 * 호출 체인:
 *   Build → gpgpu_ptx_sim_load_ptx_from_string → (PTX 파서) → [register_ptx_function]
 */
void register_ptx_function( const char *name, function_info *impl )
{
   sg_info->m_kernels[name] = impl;  // [한국어] 전역 sg_info가 가리키는 pgm_info의 m_kernels 맵에 커널 함수 등록
}

/*
 * [한국어]
 * ptxinfo_addinfo - PTX 컴파일 정보(레지스터 수 등 메타데이터)를 현재 pgm_info의 커널에 추가
 *
 * @return: 없음
 *
 * ptxinfo 파일 파싱 완료 후 호출되어, 각 커널의 레지스터 사용량, 공유 메모리 크기 등
 * 메타데이터를 function_info에 기록한다. sg_info→m_kernels를 통해 커널들에 접근한다.
 *
 * 호출 체인:
 *   Build → gpgpu_ptxinfo_load_from_string → (파싱) → [ptxinfo_addinfo]
 */
 void ptxinfo_addinfo()
{
   ptxinfo_opencl_addinfo( sg_info->m_kernels );  // [한국어] sg_info의 커널 맵에 ptxinfo 메타데이터 적용
}

/*
 * [한국어]
 * _cl_program::Build - OpenCL C 소스를 PTX로 컴파일하고 GPGPU-Sim에 로드
 *
 * @options: clBuildProgram에서 전달된 컴파일 옵션 문자열 (NULL이면 빈 문자열로 처리)
 * @return:  없음 (에러 시 exit(1))
 *
 * 전체 컴파일 흐름 (두 가지 경로):
 *
 * [경로 1] PTX_SIM_USE_PTX_FILE 환경변수 미설정 (일반 경로):
 *   ① 소스를 임시 .cl 파일로 기록 (mkstemp → write)
 *   ② OPENCL_REMOTE_GPU_HOST가 설정된 경우: ssh/rsync로 원격 서버에 파일 전송 후
 *      원격에서 nvopencl_wrapper 실행 (NVIDIA OpenCL→PTX 변환)
 *   ③ 로컬인 경우: LD_LIBRARY_PATH를 NVOPENCL_LIBDIR로 설정 후 nvopencl_wrapper 로컬 실행
 *   ④ 임시 .cl 파일 정리 (g_keep_intermediate_files 미설정 시)
 *
 * [경로 2] PTX_SIM_USE_PTX_FILE 설정 (사전 추출 PTX 사용):
 *   ① _{source_num}.ptx 파일명으로 기존 PTX 파일 사용 (컴파일 단계 건너뜀)
 *
 * [공통 후속 처리]:
 *   ⑤ PTX 파일 읽기 (끝의 '}' 이후 trailing 문자 제거)
 *   ⑥ PTX 파일 정리 (경로 1만)
 *   ⑦ gpgpu_ptx_sim_load_ptx_from_string()으로 PTX 로드 → register_ptx_function 콜백
 *   ⑧ gpgpu_ptxinfo_load_from_string()으로 레지스터 수 등 메타데이터 로드 → ptxinfo_addinfo 콜백
 *
 * 필요한 환경변수: NVOPENCL_LIBDIR, GPGPUSIM_ROOT, GPGPUSIM_CONFIG (경로 1 필수)
 * 선택 환경변수: OPENCL_REMOTE_GPU_HOST, OPENCL_REMOTE_DIRECTORY (원격 컴파일),
 *               PTX_SIM_USE_PTX_FILE (사전 추출 PTX 사용)
 *
 * 호출 체인:
 *   clBuildProgram → [Build] → nvopencl_wrapper(외부 프로세스)
 *                            → gpgpu_ptx_sim_load_ptx_from_string → register_ptx_function
 *                            → gpgpu_ptxinfo_load_from_string → ptxinfo_addinfo
 */
void _cl_program::Build(const char *options)
{
    gpgpu_context *ctx;
    ctx = GPGPU_Context();                                          // [한국어] 전역 GPGPU-Sim 컨텍스트 획득 — PTX 로더 및 설정 접근용
   printf("GPGPU-Sim OpenCL API: compiling OpenCL kernels...\n");  // [한국어] 컴파일 시작 알림 메시지
   std::map<cl_uint,pgm_info>::iterator i;
   for( i = m_pgm.begin(); i!= m_pgm.end(); i++ ) {  // [한국어] m_pgm의 각 소스 파일(pgm_info)에 대해 순차 컴파일
      pgm_info &info=i->second;     // [한국어] 현재 처리 중인 pgm_info 참조
      sg_info = &info;              // [한국어] 전역 sg_info를 현재 pgm_info로 설정 — register_ptx_function 콜백이 이를 통해 커널 등록
      unsigned source_num=i->first; // [한국어] 현재 소스의 전역 번호 (PTX_SIM_USE_PTX_FILE 경로에서 파일명 생성에 사용)
      char ptx_fname[1024];         // [한국어] 생성될 또는 읽을 PTX 파일의 경로 버퍼

      /* [한국어] PTX_SIM_USE_PTX_FILE: 이 환경변수가 설정되면 nvopencl_wrapper를 실행하지 않고
       * 사전에 추출해 둔 _{source_num}.ptx 파일을 직접 읽는다. 개발/디버깅 시 유용. */
      char *use_extracted_ptx = getenv("PTX_SIM_USE_PTX_FILE");
      if( use_extracted_ptx == NULL ) {  // [한국어] 일반 경로: nvopencl_wrapper로 OpenCL → PTX 컴파일
         char *nvopencl_libdir = getenv("NVOPENCL_LIBDIR");  // [한국어] NVIDIA의 libOpenCL.so가 있는 디렉토리 경로 (필수)
         /* [한국어] GPGPU-Sim 빌드 디렉토리 경로 구성: $GPGPUSIM_ROOT/build/$GPGPUSIM_CONFIG/
          * 이 경로에 nvopencl_wrapper 바이너리가 존재해야 한다. */
         const std::string gpgpu_opencl_path_str = std::string(getenv("GPGPUSIM_ROOT"))
            + "/build/" + std::string(getenv("GPGPUSIM_CONFIG"));
         bool error = false;
         /* [한국어] 필수 환경변수 검사 1: NVOPENCL_LIBDIR이 없으면 컴파일 불가 */
         if( nvopencl_libdir == NULL ) {
            printf("GPGPU-Sim OpenCL API: Please set your NVOPENCL_LIBDIR environment variable to\n"
                   "                      the location of NVIDIA's libOpenCL.so file on your system.\n");  // [한국어] NVOPENCL_LIBDIR 환경변수 설정 안내
            error = true;
         }
         /* [한국어] 필수 환경변수 검사 2: GPGPUSIM_ROOT와 GPGPUSIM_CONFIG 둘 다 필요 */
         if( getenv("GPGPUSIM_ROOT") == NULL || getenv("GPGPUSIM_CONFIG") == NULL ) {
            fprintf(stderr,"GPGPU-Sim OpenCL API: Please set your GPGPUSIM_ROOT environment variable\n");  // [한국어] GPGPUSIM_ROOT 미설정 에러
            fprintf(stderr,"                      to point to the location of your GPGPU-Sim installation\n");
            error = true;
         }
         if( error )
            exit(1);  // [한국어] 필수 환경변수 누락 → 즉시 프로세스 종료

         char cl_fname[1024];                // [한국어] OpenCL C 소스를 기록할 임시 파일 경로 버퍼
         const char *source = info.m_source.c_str();  // [한국어] 현재 소스의 C 문자열 포인터

         // call wrapper
         char *ld_library_path_orig = getenv("LD_LIBRARY_PATH");  // [한국어] 원래 LD_LIBRARY_PATH 저장 — 컴파일 후 복원을 위해

         // create temporary filenames
         snprintf(cl_fname,1024,"_cl_XXXXXX");   // [한국어] mkstemp 템플릿: XXXXXX가 고유 문자로 치환됨
         snprintf(ptx_fname,1024,"_ptx_XXXXXX"); // [한국어] PTX 출력 임시 파일 템플릿
         int fd=mkstemp(cl_fname);  // [한국어] cl_fname을 고유 임시 파일명으로 변환하고 파일 디스크립터 반환
         close(fd);                 // [한국어] mkstemp가 열어준 fd 닫기 — 이후 fopen으로 다시 열 것임
         fd=mkstemp(ptx_fname);    // [한국어] ptx_fname 임시 파일 생성
         close(fd);                 // [한국어] ptx fd 닫기

         // write OpenCL source to file
         FILE *fp = fopen(cl_fname,"w");  // [한국어] .cl 임시 파일을 쓰기 모드로 오픈
         if( fp == NULL ) {              // [한국어] 파일 생성 실패 — 쓰기 권한 없는 경우
            printf("GPGPU-Sim OpenCL API: ERROR ** could not create temporary files required for generating PTX\n");
            printf("                      Ensure you have write permission to the simulation directory\n");
            exit(1);                     // [한국어] 임시 파일 생성 실패 → 즉시 종료
         }
         fputs(source,fp);  // [한국어] OpenCL C 소스 문자열을 임시 .cl 파일에 기록
         fclose(fp);        // [한국어] .cl 파일 닫기 (nvopencl_wrapper가 읽을 준비 완료)

         char commandline[1024];  // [한국어] 실행할 시스템 명령줄 버퍼
         const char *opt = options?options:"";  // [한국어] 컴파일 옵션 — NULL이면 빈 문자열로 대체

         /* [한국어] 원격 컴파일 설정: OPENCL_REMOTE_GPU_HOST가 설정된 경우
          * NVIDIA GPU가 있는 원격 서버에서 OpenCL→PTX 변환을 수행한다.
          * 시뮬레이션 호스트에 NVIDIA GPU가 없을 때 사용하는 경로이다. */
         const char* remote_dir = getenv( "OPENCL_REMOTE_DIRECTORY" );  // [한국어] 원격 서버의 작업 디렉토리
         const char* local_pwd = getenv( "PWD" );                        // [한국어] 로컬 현재 작업 디렉토리
         if ( !remote_dir || strncmp( remote_dir, "", 1 ) == 0 ) {
             remote_dir = local_pwd;  // [한국어] OPENCL_REMOTE_DIRECTORY 미설정 시 로컬 디렉토리 사용
         }
         const char* remote_host = getenv( "OPENCL_REMOTE_GPU_HOST" );  // [한국어] 원격 GPU 서버 호스트명 (예: "gpu-server.example.com")
         if ( remote_host && remote_dir ) {
            // create same directory on OpenCL to PTX server
            snprintf(commandline,1024,"ssh %s mkdir -p %s", remote_host, remote_dir );  // [한국어] 원격 서버에 작업 디렉토리 생성 명령
            printf("GPGPU-Sim OpenCL API: OpenCL wrapper command line \'%s\'\n", commandline);
            fflush(stdout);                                        // [한국어] 명령 실행 전 stdout 버퍼 플러시 — 출력 순서 보장
            int result = system(commandline);                      // [한국어] ssh mkdir -p 원격 실행
            if( result ) { printf("GPGPU-Sim OpenCL API: ERROR (%d)\n", result ); exit(1); }  // [한국어] 실패 시 에러 코드 출력 후 종료

            // copy input OpenCL file to OpenCL to PTX server
            snprintf(commandline,1024,"rsync -t %s/%s %s:%s/%s", local_pwd, cl_fname, remote_host, remote_dir, cl_fname );  // [한국어] 타임스탬프 보존(-t) rsync로 .cl 파일 전송
            printf("GPGPU-Sim OpenCL API: OpenCL wrapper command line \'%s\'\n", commandline);
            fflush(stdout);
            result = system(commandline);  // [한국어] rsync 실행으로 .cl 파일을 원격 서버로 복사
            if( result ) { printf("GPGPU-Sim OpenCL API: ERROR (%d)\n", result ); exit(1); }

            // copy the nvopencl_wrapper file to the remote server
            snprintf(commandline,1024,"rsync -t %s/libopencl/bin/nvopencl_wrapper %s:%s/nvopencl_wrapper", gpgpu_opencl_path_str.c_str(), remote_host, remote_dir );  // [한국어] nvopencl_wrapper 바이너리를 원격 서버로 전송
            printf("GPGPU-Sim OpenCL API: OpenCL wrapper command line \'%s\'\n", commandline);
            fflush(stdout);
            result = system(commandline);  // [한국어] nvopencl_wrapper 바이너리 전송
            if( result ) { printf("GPGPU-Sim OpenCL API: ERROR (%d)\n", result ); exit(1); }

            // convert OpenCL to PTX on remote server
            /* [한국어] 원격 서버에서 nvopencl_wrapper 실행:
             * LD_LIBRARY_PATH를 NVOPENCL_LIBDIR(NVIDIA libOpenCL.so 경로)로 설정하고
             * nvopencl_wrapper가 .cl → .ptx 변환을 수행한다. */
            snprintf(commandline,1024,"ssh %s \"export LD_LIBRARY_PATH=%s; %s/nvopencl_wrapper %s/%s %s/%s %s\"",
                    remote_host, nvopencl_libdir, remote_dir, remote_dir, cl_fname, remote_dir, ptx_fname, opt );  // [한국어] 원격 ssh 명령: nvopencl_wrapper {cl파일} {ptx파일} {옵션}
            printf("GPGPU-Sim OpenCL API: OpenCL wrapper command line \'%s\'\n", commandline);
            fflush(stdout);
            result = system(commandline);  // [한국어] 원격 nvopencl_wrapper 실행 — NVIDIA 드라이버를 통해 CL→PTX 변환
            if( result ) { printf("GPGPU-Sim OpenCL API: ERROR (%d)\n", result ); exit(1); }

            // copy output PTX from OpenCL to PTX server back to simulation directory
            snprintf(commandline,1024,"rsync -t %s:%s/%s %s/%s", remote_host, remote_dir, ptx_fname, local_pwd, ptx_fname );  // [한국어] 생성된 PTX 파일을 원격 서버에서 로컬로 다시 복사
            printf("GPGPU-Sim OpenCL API: OpenCL wrapper command line \'%s\'\n", commandline);
            fflush(stdout);
            result = system(commandline);  // [한국어] rsync로 PTX 파일 로컬로 복사
            if( result ) { printf("GPGPU-Sim OpenCL API: ERROR (%d)\n", result ); exit(1); }
         } else {
            /* [한국어] 로컬 컴파일 경로: OPENCL_REMOTE_GPU_HOST가 없는 경우
             * LD_LIBRARY_PATH를 NVOPENCL_LIBDIR로 임시 변경하여 로컬에서 nvopencl_wrapper 실행 */
            setenv("LD_LIBRARY_PATH",nvopencl_libdir,1);  // [한국어] LD_LIBRARY_PATH를 NVIDIA libOpenCL.so 경로로 임시 변경
            snprintf(commandline,1024,"%s/libopencl/bin/nvopencl_wrapper %s %s %s",
                   gpgpu_opencl_path_str.c_str(), cl_fname, ptx_fname, opt );  // [한국어] 로컬 nvopencl_wrapper 명령: {wrapper경로} {cl파일} {ptx파일} {옵션}
            printf("GPGPU-Sim OpenCL API: OpenCL wrapper command line \'%s\'\n", commandline);
            fflush(stdout);
            int result = system(commandline);                    // [한국어] 로컬 nvopencl_wrapper 실행 — NVIDIA 드라이버 호출하여 PTX 생성
            setenv("LD_LIBRARY_PATH",ld_library_path_orig,1);  // [한국어] LD_LIBRARY_PATH를 원래 값으로 복원
            if( result != 0 ) {
               printf("GPGPU-Sim OpenCL API: ERROR ** while calling NVIDIA driver to convert OpenCL to PTX (%u)\n",
                      result );  // [한국어] nvopencl_wrapper 실패 — exit code 출력
               printf("GPGPU-Sim OpenCL API: LD_LIBRARY_PATH was \'%s\'\n", nvopencl_libdir);   // [한국어] 사용한 LD_LIBRARY_PATH 디버깅 정보
               printf("GPGPU-Sim OpenCL API: command line was \'%s\'\n", commandline);           // [한국어] 실패한 명령줄 디버깅 정보
               exit(1);                                                                           // [한국어] 컴파일 실패 → 즉시 종료
            }
         }
         /* [한국어] 임시 .cl 파일 정리: g_keep_intermediate_files가 설정된 경우 보존.
          * 보존 설정은 gpgpusim.config의 옵션으로 디버깅 시 유용하다. */
         if( !ctx->ptxinfo->g_keep_intermediate_files ) {
            // clean up files...
            snprintf(commandline,1024,"rm -f %s", cl_fname );  // [한국어] 임시 .cl 파일 삭제 명령 구성
            int result = system(commandline);                   // [한국어] rm -f {cl_fname} 실행
            if( result != 0 )
               printf("GPGPU-Sim OpenCL API: could not remove temporary files generated while generating PTX\n");  // [한국어] 정리 실패는 치명적 에러가 아님 — 경고만 출력
         }
      } else {
         /* [한국어] 사전 추출 PTX 경로: PTX_SIM_USE_PTX_FILE 환경변수가 설정된 경우.
          * _{source_num}.ptx 형식의 파일명으로 기존 PTX 파일을 사용한다. */
         snprintf(ptx_fname,1024,"_%u.ptx", source_num);  // [한국어] 사전 추출 PTX 파일명: _{소스번호}.ptx
      }

      // read in PTX generated by wrapper
      FILE *fp = fopen(ptx_fname,"r");  // [한국어] 생성된(또는 기존) PTX 파일 읽기 모드로 오픈
      if( fp == NULL ) {                // [한국어] PTX 파일이 없는 경우
         printf("GPGPU-Sim OpenCL API: ERROR ** could not open PTX file \'%s\' for reading\n", ptx_fname );
         if( use_extracted_ptx != NULL )
            printf("                      Ensure PTX files are in simulation directory.\n");  // [한국어] 사전 추출 PTX 경로: 파일 위치 안내
         exit(1);  // [한국어] PTX 파일 열기 실패 → 즉시 종료
      }
      fseek(fp,0,SEEK_END);  // [한국어] 파일 끝으로 이동 — 파일 크기 측정 준비
      unsigned len = ftell(fp);  // [한국어] 현재 위치(=파일 크기) 반환 (바이트 단위)
      if( len == 0 ) {           // [한국어] 빈 PTX 파일인 경우 — nvopencl_wrapper가 빈 출력 생성
         exit(1);                // [한국어] 빈 PTX 파일은 치명적 오류 → 즉시 종료
      }
      fseek(fp,0,SEEK_SET);                    // [한국어] 파일 처음으로 되돌아가기
      char *tmp = (char*)calloc(len+1,1);      // [한국어] PTX 전체 + null terminator를 위한 메모리 할당 (0으로 초기화)
      fread(tmp,1,len,fp);                     // [한국어] PTX 파일 전체 내용을 tmp 버퍼로 읽기
      fclose(fp);                              // [한국어] PTX 파일 닫기
      if( use_extracted_ptx == NULL ) {
         // clean up files...
         char commandline[1024];
         snprintf(commandline,1024,"rm -f %s", ptx_fname );  // [한국어] 임시 PTX 파일 삭제 명령 구성
         int result = system(commandline);                    // [한국어] rm -f {ptx_fname} 실행
         if( result != 0 )
            printf("GPGPU-Sim OpenCL API: could not remove temporary files generated while generating PTX\n");
         // remove any trailing characters from string
         /* [한국어] PTX 파일 끝 정리: nvopencl_wrapper가 '}' 이후에 불필요한 문자를 추가할 수 있다.
          * PTX 파일의 마지막 유효 문자는 '}' (마지막 .entry 함수 닫기 괄호)이므로
          * 그 이후의 모든 문자를 null로 교체하여 제거한다. */
         while( len > 0 && tmp[len] != '}' ) {
            tmp[len] = 0;  // [한국어] '}' 이후 문자를 null로 교체 (문자열 단축)
            len--;         // [한국어] 앞으로 이동하며 검색
         }
      }
      info.m_asm = tmp;  // [한국어] PTX 어셈블리 문자열을 pgm_info에 저장
      /* [한국어] PTX를 GPGPU-Sim의 PTX 로더로 파싱 및 로드:
       * register_ptx_function 콜백이 각 .entry 함수마다 호출되어 sg_info->m_kernels에 등록됨. */
      info.m_symtab = ctx->gpgpu_ptx_sim_load_ptx_from_string( tmp, source_num );
      /* [한국어] PTX info(레지스터 수, 공유 메모리 크기 등 메타데이터) 로드:
       * ptxinfo_addinfo 콜백이 호출되어 m_kernels의 function_info에 메타데이터 적용됨. */
      ctx->gpgpu_ptxinfo_load_from_string( tmp, source_num );
      free(tmp);  // [한국어] PTX 문자열 버퍼 해제 (이미 info.m_asm와 심볼 테이블에 복사됨)
   }
   printf("GPGPU-Sim OpenCL API: finished compiling OpenCL kernels.\n");  // [한국어] 모든 소스 파일 컴파일 완료 메시지
}

/*
 * [한국어]
 * _cl_program::CreateKernel - 빌드된 PTX에서 특정 이름의 커널 함수로 _cl_kernel 객체 생성
 *
 * @kernel_name: 찾을 OpenCL 커널 함수 이름 (PTX .entry 함수 이름과 동일)
 * @errcode_ret: 에러 코드 출력 포인터 (NULL 가능)
 * @return:      성공 시 new _cl_kernel(...) 포인터, 커널 미발견 시 NULL
 *
 * m_pgm의 모든 pgm_info의 m_kernels 맵에서 kernel_name을 검색한다.
 * 서로 다른 .cl 소스 파일에 동일 이름의 커널이 있으면 assert 실패한다 (의도적 제약).
 *
 * 호출 체인:
 *   clCreateKernel → [CreateKernel] → new _cl_kernel → (커널 실행 준비 완료)
 */
cl_kernel _cl_program::CreateKernel( const char *kernel_name, cl_int *errcode_ret )
{
   cl_kernel result = NULL;            // [한국어] 반환할 커널 객체 (기본 NULL)
   class function_info *finfo=NULL;    // [한국어] 찾은 커널의 PTX function_info 포인터 (기본 NULL)
   std::map<cl_uint,pgm_info>::iterator f;
   for( f = m_pgm.begin(); f!= m_pgm.end(); f++ ) {  // [한국어] 모든 소스 파일의 pgm_info를 순회
      pgm_info &info=f->second;        // [한국어] 현재 소스의 pgm_info 참조
      std::map<std::string,function_info*>::iterator k = info.m_kernels.find(kernel_name);  // [한국어] 이 소스의 커널 맵에서 이름 검색
      if( k != info.m_kernels.end() ) {  // [한국어] 이 소스에서 커널 발견
         assert( finfo == NULL ); // kernels with same name in different .cl files  // [한국어] 이미 다른 소스에서 같은 이름의 커널을 찾은 경우 → 중복 오류
         finfo = k->second;  // [한국어] 발견된 커널의 function_info 저장
      }
   }

   if( finfo == NULL )
      setErrCode( errcode_ret, CL_INVALID_PROGRAM_EXECUTABLE );  // [한국어] 커널 미발견 → CL_INVALID_PROGRAM_EXECUTABLE 에러
   else{
      result = new _cl_kernel(this,kernel_name,finfo);  // [한국어] 커널 발견 → _cl_kernel 객체 힙 할당
      setErrCode( errcode_ret, CL_SUCCESS );            // [한국어] 성공 에러 코드 설정
   }
   return result;  // [한국어] 커널 객체 포인터 반환 (미발견 시 NULL)
}

/*
 * [한국어]
 * _cl_program::get_ptx - 이 프로그램의 모든 PTX 어셈블리를 연결하여 반환
 *
 * @return: 모든 소스의 PTX 문자열을 연결한 힙 할당 버퍼. 호출자가 free() 해야 함.
 *          빌드 전 호출 시 abort().
 *
 * 주로 clGetProgramInfo(CL_PROGRAM_BINARIES)에서 PTX 바이너리 반환 시 사용된다.
 *
 * 호출 체인:
 *   clGetProgramInfo(CL_PROGRAM_BINARIES) → [get_ptx] → memcpy PTX → 호출자 버퍼로 복사
 */
char *_cl_program::get_ptx()
{
   if( m_pgm.empty() ) {  // [한국어] Build() 호출 전에 get_ptx() 호출 — 잘못된 사용 감지
      printf("GPGPU-Sim PTX OpenCL API: Cannot get PTX before building program\n");
      abort();  // [한국어] 빌드 전 호출 → abort로 프로세스 종료
   }
   size_t buffer_length= get_ptx_size();               // [한국어] 전체 PTX 크기 계산
   char *tmp = (char*)calloc(buffer_length + 1,1);     // [한국어] PTX 전체 + null terminator를 위한 버퍼 할당 (0으로 초기화)
   tmp[ buffer_length ] = '\0';                        // [한국어] 명시적 null terminator 설정
   unsigned n=0;                                       // [한국어] 현재 복사 위치 (바이트 오프셋)
   std::map<cl_uint,pgm_info>::iterator p;
   for( p=m_pgm.begin(); p != m_pgm.end(); p++ ) {    // [한국어] 각 소스의 PTX를 순서대로 연결
      const char *ptx = p->second.m_asm.c_str();      // [한국어] 이 소스의 PTX 문자열 포인터
      unsigned len = strlen( ptx );                    // [한국어] 이 PTX 문자열의 길이
      assert( (n+len) <= buffer_length );              // [한국어] 버퍼 오버플로 방지 검사
      memcpy(tmp+n,ptx,len);                           // [한국어] PTX를 버퍼의 현재 위치에 복사
      n+=len;                                          // [한국어] 다음 PTX를 복사할 위치로 전진
   }
   assert( n == buffer_length );  // [한국어] 최종 복사량이 예상 총 크기와 일치하는지 검증
   return tmp;                    // [한국어] 연결된 PTX 버퍼 반환 (호출자가 free() 책임)
}

/*
 * [한국어]
 * _cl_program::get_ptx_size - 빌드된 전체 PTX 어셈블리의 바이트 크기 계산
 *
 * @return: 모든 소스의 PTX 문자열 길이 합계 (바이트). clGetProgramInfo(CL_PROGRAM_BINARY_SIZES)에서 사용.
 *
 * 호출 체인:
 *   clGetProgramInfo(CL_PROGRAM_BINARY_SIZES) → [get_ptx_size]
 *   get_ptx() → [get_ptx_size] → buffer 크기 결정
 */
size_t _cl_program::get_ptx_size()
{
   size_t buffer_length=0;                          // [한국어] 전체 PTX 크기 누적 변수
   std::map<cl_uint,pgm_info>::iterator p;
   for( p=m_pgm.begin(); p != m_pgm.end(); p++ ) { // [한국어] 각 소스의 PTX 크기를 누적
      buffer_length += p->second.m_asm.length();    // [한국어] 이 소스의 PTX 문자열 길이 추가
   }
   return buffer_length;  // [한국어] 전체 PTX 바이트 크기 반환
}

/* [한국어] 정적 멤버 변수 초기화: 전역 UID 카운터들을 0으로 초기화.
 * C++에서 static 멤버 변수는 클래스 외부에서 별도로 정의/초기화해야 한다. */
unsigned _cl_context::sm_context_uid = 0;  // [한국어] _cl_context UID 카운터 초기화
unsigned _cl_kernel::sm_context_uid = 0;   // [한국어] _cl_kernel UID 카운터 초기화

/*
 * [한국어]
 * GPGPUSim_Init - GPGPU-Sim GPU 디바이스 싱글톤 초기화 및 반환
 *
 * @return: static _cl_device_id* 포인터 (단일 인스턴스). 항상 동일한 포인터 반환.
 *
 * 최초 호출 시 gpgpu_ptx_sim_init_perf()로 gpgpu_sim 인스턴스를 생성하고
 * _cl_device_id로 래핑한다. 이후 호출에서는 기존 the_device를 재사용한다.
 * 매 호출마다 start_sim_thread(2)를 호출하여 시뮬레이션 스레드를 시작/재시작한다.
 * (2는 OpenCL 시뮬레이션 모드를 나타내는 상수)
 *
 * 호출 체인:
 *   clGetDeviceIDs / clCreateContext / clCreateContextFromType / clGetDeviceInfo
 *     → [GPGPUSim_Init] → gpgpu_ptx_sim_init_perf() (최초 1회)
 *                       → start_sim_thread(2) (매 호출)
 */
class _cl_device_id *GPGPUSim_Init()
{
   static _cl_device_id *the_device = NULL;   // [한국어] 싱글톤 GPU 디바이스 포인터 (static — 함수 호출 간 유지)
   gpgpu_context *ctx;
   ctx = GPGPU_Context();                     // [한국어] 전역 GPGPU-Sim 컨텍스트 획득
   if( !the_device ) {                        // [한국어] 최초 호출 시에만 GPU 초기화 수행
      gpgpu_sim *the_gpu = ctx->gpgpu_ptx_sim_init_perf();  // [한국어] gpgpu_sim 인스턴스 생성 및 성능 모델 초기화
      the_device = new _cl_device_id(the_gpu);              // [한국어] gpgpu_sim*를 _cl_device_id로 래핑
   }
   ctx->start_sim_thread(2);  // [한국어] 시뮬레이션 스레드 시작 (2 = OpenCL 모드 플래그)
   return the_device;         // [한국어] 싱글톤 디바이스 포인터 반환
}

/*
 * [한국어]
 * opencl_not_implemented - 미구현 OpenCL API 함수 호출 시 에러 출력 및 종료
 *
 * @func: 미구현 함수 이름 (__my_func__ 매크로로 전달)
 * @line: 호출 소스 라인 번호 (__LINE__ 매크로로 전달)
 * @return: 없음 (noreturn)
 *
 * GPGPU-Sim에서 아직 구현하지 않은 OpenCL API 함수의 stub에서 호출된다.
 * stdout/stderr를 플러시하여 출력 버퍼가 소실되지 않도록 하고 abort()로 종료한다.
 *
 * 호출 체인:
 *   clGetProgramInfo(CL_PROGRAM_SOURCE) 등 미구현 함수 → [opencl_not_implemented] → abort()
 */
void opencl_not_implemented( const char* func, unsigned line )
{
   fflush(stdout);  // [한국어] stdout 버퍼 플러시 — 이전 출력이 소실되지 않도록
   fflush(stderr);  // [한국어] stderr 버퍼 플러시
   printf("\n\nGPGPU-Sim PTX: Execution error: OpenCL API function \"%s()\" has not been implemented yet.\n"
         "                 [$GPGPUSIM_ROOT/libcuda/%s around line %u]\n\n\n",
         func,__FILE__, line );  // [한국어] 미구현 함수명, 파일, 라인 번호를 포함한 에러 메시지 출력
   fflush(stdout);  // [한국어] printf 출력 플러시 — abort() 전 마지막 출력 보장
   abort();         // [한국어] SIGABRT로 프로세스 종료 (core dump 생성)
}

/*
 * [한국어]
 * opencl_not_finished - 부분 구현된 OpenCL API 함수 호출 시 에러 출력 및 종료
 *
 * @func: 미완성 함수 이름
 * @line: 호출 소스 라인 번호
 * @return: 없음 (noreturn)
 *
 * opencl_not_implemented와 달리, 이 함수는 "구현이 시작되었으나 완료되지 않은" API에서 호출된다.
 * 예: clGetContextInfo에서 CL_CONTEXT_REFERENCE_COUNT 처리가 아직 미완성인 경우.
 *
 * 호출 체인:
 *   clGetContextInfo(CL_CONTEXT_REFERENCE_COUNT) 등 → [opencl_not_finished] → abort()
 */
void opencl_not_finished( const char* func, unsigned line )
{
   fflush(stdout);  // [한국어] stdout 버퍼 플러시
   fflush(stderr);  // [한국어] stderr 버퍼 플러시
   printf("\n\nGPGPU-Sim PTX: Execution error: OpenCL API function \"%s()\" has not been completed yet.\n"
         "                 [$GPGPUSIM_ROOT/libopencl/%s around line %u]\n\n\n",
         func,__FILE__, line );  // [한국어] 미완성 함수명, 파일, 라인 번호를 포함한 에러 메시지
   fflush(stdout);  // [한국어] 마지막 출력 플러시
   abort();         // [한국어] 프로세스 종료
}

/*
 * [한국어]
 * clCreateContextFromType - 디바이스 타입으로 OpenCL 컨텍스트 생성
 *
 * @properties:   컨텍스트 속성 배열 (NULL이면 무시, 현재 속성 처리 미구현)
 * @device_type:  원하는 디바이스 타입 (CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL 등)
 * @pfn_notify:   에러 콜백 함수 포인터 (현재 미사용)
 * @user_data:    콜백 사용자 데이터 (현재 미사용)
 * @errcode_ret:  에러 코드 출력 포인터 (NULL 가능)
 * @return:       생성된 cl_context 객체. 실패 시 NULL.
 *
 * clCreateContext와 달리 명시적 디바이스 목록 대신 디바이스 타입으로 컨텍스트를 생성한다.
 * GPGPU-Sim은 GPU/ACCELERATOR/DEFAULT/ALL 타입을 지원하며, CPU 타입은 거부한다.
 * properties는 현재 무시된다 (AMD 샘플 앱 호환성을 위해 exit(1) 주석 처리).
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clCreateContextFromType] → GPGPUSim_Init → new _cl_context
 */
extern CL_API_ENTRY cl_context CL_API_CALL
clCreateContextFromType(const cl_context_properties * properties,
                        cl_device_type          device_type,
                        void (*pfn_notify)(const char *, const void *, size_t, void *),
                        void *                  user_data,
                        cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
   _cl_device_id *gpu = GPGPUSim_Init();  // [한국어] GPGPU-Sim 싱글톤 GPU 디바이스 초기화 또는 획득

   /* [한국어] 디바이스 타입 검사: GPGPU-Sim은 GPU, ACCELERATOR, DEFAULT, ALL 타입으로 분류 가능.
    * CPU 타입은 지원하지 않으므로 default 케이스에서 거부한다. */
   switch (device_type) {
   case CL_DEVICE_TYPE_GPU:           // [한국어] GPU 타입 — GPGPU-Sim은 GPU 시뮬레이터이므로 지원
   case CL_DEVICE_TYPE_ACCELERATOR:   // [한국어] 가속기 타입 — GPGPU-Sim을 가속기로 분류하는 경우
   case CL_DEVICE_TYPE_DEFAULT:       // [한국어] 기본 타입 — 시스템의 기본 디바이스 (여기서는 GPGPU-Sim)
   case CL_DEVICE_TYPE_ALL:           // [한국어] 모든 타입 — CPU를 제외한 모든 디바이스 (GPGPU-Sim 포함)
      break; // GPGPU-Sim qualifies as these types of device.
   default:
      printf("GPGPU-Sim OpenCL API: unsupported device type %lx\n", device_type );  // [한국어] CPU 등 미지원 타입 요청 경고
      setErrCode( errcode_ret, CL_DEVICE_NOT_FOUND );  // [한국어] 해당 타입의 디바이스 없음 에러
      return NULL;                                       // [한국어] 컨텍스트 생성 실패 반환
      break;
   }

   if( properties != NULL ) {
      printf("GPGPU-Sim OpenCL API: do not know how to use properties in %s\n", __my_func__ );  // [한국어] 속성 처리 미구현 경고 출력 (하지만 계속 실행)
      //exit(1); // Temporarily commented out to allow the AMD Sample applications to run.  // [한국어] AMD 샘플 앱 호환성을 위해 주석 처리됨 — 실제 속성 처리 필요
   }

   setErrCode( errcode_ret, CL_SUCCESS );  // [한국어] 성공 에러 코드 설정
   cl_context ctx = new _cl_context(gpu);  // [한국어] 새 OpenCL 컨텍스트 객체 생성 (힙 할당)
   return ctx;                             // [한국어] 생성된 컨텍스트 반환
}

/*
 * [한국어] 미구현 stub 함수들 — OpenCL API 함수 중 GPGPU-Sim이 아직 구현하지 않은 것들.
 * 호출 시 opencl_not_finished() 또는 경고를 통해 처리되며, 일부는 CL_SUCCESS만 반환한다.
 */
/***************************** Unimplemented shell functions *******************************************/
/*
 * [한국어]
 * clCreateProgramWithBinary - 사전 컴파일된 바이너리(cubin/PTX)로 프로그램 생성 (미구현)
 *
 * @return: 미구현. opencl_not_finished()로 abort().
 *
 * GPGPU-Sim은 소스 컴파일 경로만 지원하며, 사전 컴파일된 바이너리 로딩은 미구현.
 */
extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBinary(cl_context                     /* context */,
                          cl_uint                        /* num_devices */,
                          const cl_device_id *           /* device_list */,
                          const size_t *                 /* lengths */,
                          const unsigned char **         /* binaries */,
                          cl_int *                       /* binary_status */,
                          cl_int *                       /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0 {

	opencl_not_finished(__my_func__, __LINE__ );  // [한국어] 미완성 기능 → abort()
	return cl_program();                          // [한국어] unreachable — 컴파일러 경고 방지
}

/*
 * [한국어]
 * clGetEventProfilingInfo - 이벤트 프로파일링 정보 조회 (미구현, CL_SUCCESS 반환)
 *
 * @return: CL_SUCCESS (경고 출력 후 성공 반환)
 *
 * GPGPU-Sim의 시뮬레이션 모델은 실제 타이밍 측정이 아닌 사이클 시뮬레이션이므로
 * OpenCL 이벤트 프로파일링(나노초 단위 타임스탬프)을 지원하지 않는다.
 * 미구현이지만 CL_SUCCESS를 반환하여 애플리케이션이 계속 실행될 수 있게 한다.
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetEventProfilingInfo(cl_event            /* event */,
                        cl_profiling_info   /* param_name */,
                        size_t              /* param_value_size */,
                        void *              /* param_value */,
                        size_t *            /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0{
	gpgpusim_opencl_warning(__my_func__,__LINE__, "GPGPUsim - OpenCLFunction is not implemented. Returning CL_SUCCESS");  // [한국어] 미구현 경고 출력 후 계속
	return CL_SUCCESS;  // [한국어] 실패를 에러로 처리하지 않고 CL_SUCCESS 반환 (앱 호환성 유지)
}
/*******************************************************************************************************/


/*
 * [한국어]
 * clCreateContext - 명시적 디바이스 목록으로 OpenCL 컨텍스트 생성
 *
 * @properties:  컨텍스트 속성 배열. CL_CONTEXT_PLATFORM 항목이 있으면 GPGPU-Sim 플랫폼인지 검증.
 * @num_devices: 디바이스 수 (현재 무시 — GPGPU-Sim은 단일 GPU만 지원)
 * @devices:     디바이스 ID 배열 (현재 무시 — GPGPUSim_Init()이 반환한 유일한 디바이스 사용)
 * @pfn_notify:  에러 콜백 (현재 미사용)
 * @user_data:   콜백 사용자 데이터 (현재 미사용)
 * @errcode_ret: 에러 코드 출력 포인터 (NULL 가능)
 * @return:      생성된 cl_context. 플랫폼 불일치 시 NULL.
 *
 * clCreateContextFromType과 달리 디바이스 목록을 직접 지정하지만, GPGPU-Sim에서는
 * 사실상 GPGPUSim_Init()의 싱글톤 디바이스만 사용한다.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clCreateContext] → GPGPUSim_Init → new _cl_context
 */
extern CL_API_ENTRY cl_context CL_API_CALL
clCreateContext(  const cl_context_properties * properties,
                  cl_uint num_devices,
                  const cl_device_id *devices,
                  void (*pfn_notify)(const char *, const void *, size_t, void *),
                  void *                  user_data,
                  cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
   struct _cl_device_id *gpu = GPGPUSim_Init();  // [한국어] 싱글톤 GPU 디바이스 획득
   if( properties != NULL ) {
      /* [한국어] properties 검증: CL_CONTEXT_PLATFORM 속성이 있으면 GPGPU-Sim 플랫폼과 일치하는지 확인.
       * properties 배열 형식: {CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0} */
      if( properties[0] != CL_CONTEXT_PLATFORM || properties[1] != (cl_context_properties)&g_gpgpu_sim_platform_id ) {
         setErrCode( errcode_ret, CL_INVALID_PLATFORM );  // [한국어] 다른 플랫폼(NVIDIA, AMD 등)의 플랫폼 ID → 거부
         return NULL;                                      // [한국어] 잘못된 플랫폼으로 컨텍스트 생성 실패
      }
   }
   setErrCode( errcode_ret, CL_SUCCESS );  // [한국어] 성공 에러 코드 설정
   cl_context ctx = new _cl_context(gpu);  // [한국어] 새 OpenCL 컨텍스트 객체 생성
   return ctx;                             // [한국어] 생성된 컨텍스트 반환
}

/*
 * [한국어]
 * clGetContextInfo - OpenCL 컨텍스트 속성 정보 조회
 *
 * @context:            조회할 컨텍스트
 * @param_name:         조회할 속성 (CL_CONTEXT_DEVICES 등)
 * @param_value_size:   param_value 버퍼 크기
 * @param_value:        결과를 기록할 버퍼 (NULL이면 크기만 반환)
 * @param_value_size_ret: 실제 필요한 크기를 기록할 포인터 (NULL 가능)
 * @return:             CL_SUCCESS 또는 CL_INVALID_CONTEXT
 *
 * CL_CONTEXT_DEVICES만 구현됨. 연결된 디바이스 목록(_cl_device_id 연결 리스트)을 반환.
 * CL_CONTEXT_REFERENCE_COUNT, CL_CONTEXT_PROPERTIES는 opencl_not_finished()로 미구현 처리.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clGetContextInfo] → context->get_first_device()
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetContextInfo(cl_context         context,
                 cl_context_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0
{
   if( context == NULL ) return CL_INVALID_CONTEXT;  // [한국어] NULL 컨텍스트 체크
   switch( param_name ) {
   case CL_CONTEXT_DEVICES: {  // [한국어] 컨텍스트에 연결된 디바이스 목록 반환
      unsigned ngpu=0;                                        // [한국어] 발견된 GPU 개수 카운터
      cl_device_id device_id = context->get_first_device();  // [한국어] 첫 번째 디바이스 획득
      while ( device_id != NULL ) {                           // [한국어] 디바이스 연결 리스트 순회 (현재는 1개만 존재)
         if( param_value )
            ((cl_device_id*)param_value)[ngpu] = device_id;  // [한국어] 디바이스 ID를 출력 배열에 기록
         device_id = device_id->next();                       // [한국어] 다음 디바이스로 이동 (항상 NULL)
         ngpu++;                                              // [한국어] 디바이스 카운터 증가
      }
      if( param_value_size_ret ) *param_value_size_ret = ngpu * sizeof(cl_device_id);  // [한국어] 필요한 버퍼 크기 반환
      break;
   }
   case CL_CONTEXT_REFERENCE_COUNT:
      opencl_not_finished(__my_func__,__LINE__);  // [한국어] 참조 카운팅 미구현
      break;
   case CL_CONTEXT_PROPERTIES:
      opencl_not_finished(__my_func__,__LINE__);  // [한국어] 속성 쿼리 미구현
      break;
   default:
      opencl_not_finished(__my_func__,__LINE__);  // [한국어] 기타 알 수 없는 속성
   }
   return CL_SUCCESS;  // [한국어] 성공 반환 (CL_CONTEXT_DEVICES 처리 완료)
}

/*
 * [한국어]
 * clCreateCommandQueue - OpenCL 커맨드 큐 생성
 *
 * @context:    커맨드 큐가 속할 컨텍스트
 * @device:     커맨드 큐가 연결될 디바이스 (현재 검증 없이 저장만 함)
 * @properties: 큐 속성 플래그 (CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE 등)
 * @errcode_ret: 에러 코드 출력 포인터 (NULL 가능)
 * @return:     생성된 cl_command_queue. 실패 시 NULL.
 *
 * GPGPU-Sim의 커맨드 큐는 실제 비동기 큐를 구현하지 않는다.
 * CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE와 CL_QUEUE_PROFILING_ENABLE은 경고 후 무시된다.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clCreateCommandQueue] → new _cl_command_queue
 */
extern CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context                     context,
                     cl_device_id                   device,
                     cl_command_queue_properties    properties,
                     cl_int *                       errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
   if( !context ) { setErrCode( errcode_ret, CL_INVALID_CONTEXT );   return NULL; }  // [한국어] NULL 컨텍스트 에러
   gpgpusim_opencl_warning(__my_func__,__LINE__, "assuming device_id is in context");  // [한국어] device가 context에 속하는지 검증하지 않음을 경고
   if( (properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) )
      gpgpusim_opencl_warning(__my_func__,__LINE__, "ignoring command queue property");  // [한국어] 비순서 실행 모드 무시 경고 — GPGPU-Sim은 동기적으로만 실행
   if( (properties & CL_QUEUE_PROFILING_ENABLE) )
      gpgpusim_opencl_warning(__my_func__,__LINE__, "ignoring command queue property");  // [한국어] 프로파일링 모드 무시 경고 — 타임스탬프 미지원
   setErrCode( errcode_ret, CL_SUCCESS );                        // [한국어] 성공 에러 코드 설정
   return new _cl_command_queue(context,device,properties);      // [한국어] 새 커맨드 큐 객체 생성 및 반환
}

/*
 * [한국어]
 * clCreateBuffer - OpenCL 버퍼 메모리 객체 생성
 *
 * @context:    버퍼가 속할 컨텍스트
 * @flags:      메모리 플래그 (CL_MEM_READ_WRITE, CL_MEM_USE_HOST_PTR 등)
 * @size:       버퍼 크기 (바이트)
 * @host_ptr:   호스트 데이터 포인터 (NULL 가능)
 * @errcode_ret: 에러 코드 출력 포인터 (NULL 가능)
 * @return:     생성된 cl_mem 핸들. 실패 시 NULL.
 *
 * 내부적으로 context->CreateBuffer()를 위임하며, GPU 메모리 할당 또는 호스트 메모리 래핑을 수행.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clCreateBuffer] → _cl_context::CreateBuffer → _cl_mem 생성자 → gpu_malloc
 */
extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer(cl_context   context,
               cl_mem_flags flags,
               size_t       size ,
               void *       host_ptr,
               cl_int *     errcode_ret ) CL_API_SUFFIX__VERSION_1_0
{
   if( !context ) { setErrCode( errcode_ret, CL_INVALID_CONTEXT );   return NULL; }  // [한국어] NULL 컨텍스트 에러
   return context->CreateBuffer(flags,size,host_ptr,errcode_ret);  // [한국어] 컨텍스트에 버퍼 생성 위임
}

/*
 * [한국어]
 * clCreateProgramWithSource - OpenCL C 소스 문자열로 프로그램 객체 생성
 *
 * @context:    프로그램이 속할 컨텍스트
 * @count:      소스 문자열 수
 * @strings:    OpenCL C 소스 문자열 배열
 * @lengths:    각 문자열 길이 배열 (NULL이면 strlen 사용)
 * @errcode_ret: 에러 코드 출력 포인터 (NULL 가능)
 * @return:     생성된 cl_program. 실패 시 NULL.
 *
 * 소스를 _cl_program 객체에 저장만 한다. 실제 컴파일은 clBuildProgram()에서 수행.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clCreateProgramWithSource] → new _cl_program
 */
extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context        context,
                          cl_uint           count,
                          const char **     strings,
                          const size_t *    lengths,
                          cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
   if( !context ) { setErrCode( errcode_ret, CL_INVALID_CONTEXT );   return NULL; }  // [한국어] NULL 컨텍스트 에러
   setErrCode( errcode_ret, CL_SUCCESS );                              // [한국어] 성공 에러 코드 설정
   return new _cl_program(context,count,strings,lengths);              // [한국어] 소스를 보관하는 _cl_program 객체 생성 반환
}


/*
 * [한국어]
 * clBuildProgram - OpenCL 프로그램을 컴파일(빌드)
 *
 * @program:     빌드할 cl_program 객체
 * @num_devices: 빌드 대상 디바이스 수 (현재 무시)
 * @device_list: 빌드 대상 디바이스 목록 (현재 무시)
 * @options:     컴파일 옵션 문자열 (nvopencl_wrapper에 전달됨)
 * @pfn_notify:  빌드 완료 콜백 (현재 미사용 — 동기 빌드만 지원)
 * @user_data:   콜백 사용자 데이터 (현재 미사용)
 * @return:      CL_SUCCESS 또는 CL_INVALID_PROGRAM
 *
 * 내부적으로 _cl_program::Build()를 호출하여 OpenCL C → PTX 컴파일을 수행한다.
 * Build()는 에러 시 exit(1)을 호출하므로 이 함수는 에러 코드를 반환하지 않는다.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clBuildProgram] → program->Build() → nvopencl_wrapper(외부 프로세스)
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program           program,
               cl_uint              num_devices,
               const cl_device_id * device_list,
               const char *         options,
               void (*pfn_notify)(cl_program /* program */, void * /* user_data */),
               void *               user_data ) CL_API_SUFFIX__VERSION_1_0
{
   if( !program ) return CL_INVALID_PROGRAM;  // [한국어] NULL 프로그램 에러
   program->Build(options);                   // [한국어] OpenCL C → PTX 컴파일 수행 (실패 시 exit(1))
   return CL_SUCCESS;                         // [한국어] 빌드 성공 반환
}

/*
 * [한국어]
 * clCreateKernel - 빌드된 프로그램에서 특정 이름의 커널 객체 생성
 *
 * @program:     빌드된 cl_program 객체
 * @kernel_name: 커널 함수 이름 (PTX .entry 함수 이름)
 * @errcode_ret: 에러 코드 출력 포인터 (NULL 가능)
 * @return:      생성된 cl_kernel 객체. 커널 미발견 시 NULL.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clCreateKernel] → program->CreateKernel → new _cl_kernel
 */
extern CL_API_ENTRY cl_kernel CL_API_CALL
clCreateKernel(cl_program      program,
               const char *    kernel_name,
               cl_int *        errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
   if( kernel_name == NULL ) {  // [한국어] NULL 커널 이름 에러 체크
      setErrCode( errcode_ret, CL_INVALID_KERNEL_NAME );  // [한국어] 커널 이름 NULL → CL_INVALID_KERNEL_NAME
      return NULL;
   }
   cl_kernel kobj = program->CreateKernel(kernel_name,errcode_ret);  // [한국어] 프로그램에서 커널 이름으로 _cl_kernel 생성
   return kobj;  // [한국어] 생성된 커널 객체 반환 (미발견 시 NULL)
}

/*
 * [한국어]
 * clSetKernelArg - 커널 인자 값 설정
 *
 * @kernel:    인자를 설정할 cl_kernel 객체
 * @arg_index: 인자 인덱스 (0부터 시작, 커널 파라미터 순서)
 * @arg_size:  인자 크기 (바이트)
 * @arg_value: 인자 값 포인터 (cl_mem 타입은 cl_mem* 포인터로 전달됨)
 * @return:    CL_SUCCESS (항상)
 *
 * 내부적으로 _cl_kernel::SetKernelArg()로 위임하여 m_args에 저장.
 * clEnqueueNDRangeKernel 호출 전에 모든 인자가 설정되어야 한다.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clSetKernelArg] → kernel->SetKernelArg → m_args[arg_index] = {size, value}
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArg(cl_kernel    kernel,
               cl_uint      arg_index,
               size_t       arg_size,
               const void * arg_value ) CL_API_SUFFIX__VERSION_1_0
{
   kernel->SetKernelArg(arg_index,arg_size,arg_value);  // [한국어] 커널 인자를 m_args에 저장 위임
   return CL_SUCCESS;                                    // [한국어] 항상 성공 반환
}

/*
 * [한국어]
 * clEnqueueNDRangeKernel - OpenCL 커널을 ND-Range로 실행 (GPGPU-Sim에서 가장 핵심 함수)
 *
 * @command_queue:          커널을 실행할 커맨드 큐 (디바이스 및 컨텍스트 접근용)
 * @kernel:                 실행할 cl_kernel 객체
 * @work_dim:               작업 차원 수 (1~3). 0 또는 4이상이면 CL_INVALID_WORK_DIMENSION
 * @global_work_offset:     각 차원의 전역 작업 ID 시작 오프셋 (현재 비 0 값은 abort)
 * @global_work_size:       각 차원의 전체 work-item 수 (OpenCL의 스레드 격자 크기)
 * @local_work_size:        각 차원의 work-group 크기 (CUDA의 threadBlock 크기). NULL이면 자동 선택
 * @num_events_in_wait_list: 대기할 이벤트 수 (현재 무시)
 * @event_wait_list:        대기 이벤트 목록 (현재 무시)
 * @event:                  완료 이벤트 출력 (현재 미사용)
 * @return:                 CL_SUCCESS, CL_INVALID_WORK_DIMENSION, CL_INVALID_WORK_GROUP_SIZE 등
 *
 * [전체 처리 흐름]
 * 1) PTX_SIM_MODE_FUNC 환경변수로 기능/성능 시뮬레이션 모드 결정
 * 2) work_dim 유효성 검사 (1~3만 허용)
 * 3) local_work_size 결정:
 *    - 제공된 경우: 그대로 사용
 *    - NULL인 경우 자동 선택 알고리즘:
 *      * d==0: global_work_size[0]이 SM 최대 스레드 수 이하면 그대로,
 *              그렇지 않으면 SM 최대 스레드 수에서 64씩 감소하며 global_work_size를 나눌 수 있는 값 탐색
 *      * d>0: 항상 1
 * 4) global_work_size가 local_work_size의 배수인지 검증
 * 5) global_work_offset가 비 0이면 abort (현재 미지원)
 * 6) OpenCL work 크기 → CUDA dim3 변환:
 *    GridDim = global_work_size / local_work_size (각 차원별)
 *    BlockDim = local_work_size
 * 7) 커널 인자 bind_args()로 gpgpu_ptx_sim_arg_list_t 생성
 * 8) PTX 버전 < 3.0이면 OpenCL 내장 변수(%_global_size 등)를 GPU 심볼로 복사
 * 9) gpgpu_opencl_ptx_sim_init_grid()로 kernel_info_t(실행 격자) 생성
 * 10) PDOM(Post-Dominator) 분석: warp 재합류 포인트 계산 (미수행 시에만)
 * 11) 시뮬레이션 실행:
 *    - 기능 모드: gpgpu_opencl_ptx_sim_main_func(grid) — 정확한 기능 검증, 타이밍 무시
 *    - 성능 모드: gpgpu_opencl_ptx_sim_main_perf(grid) — 사이클-레벨 타이밍 시뮬레이션
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clEnqueueNDRangeKernel]
 *     → gpgpu_opencl_ptx_sim_init_grid → kernel_info_t 생성
 *     → do_pdom (PDOM 분석)
 *     → gpgpu_opencl_ptx_sim_main_perf / gpgpu_opencl_ptx_sim_main_func (시뮬레이션)
 *       → gpu-sim.cc의 사이클 루프
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel(cl_command_queue command_queue,
                       cl_kernel        kernel,
                       cl_uint          work_dim,
                       const size_t *   global_work_offset,
                       const size_t *   global_work_size,
                       const size_t *   local_work_size,
                       cl_uint          num_events_in_wait_list,
                       const cl_event * event_wait_list,
                       cl_event *       event) CL_API_SUFFIX__VERSION_1_0
{
    gpgpu_context *ctx;
    ctx = GPGPU_Context();                                // [한국어] 전역 GPGPU-Sim 컨텍스트 획득
   int _global_size[3];                                   // [한국어] 전역 work-item 수를 int 배열로 변환 (PTX 내장 변수용)
   int zeros[3] = { 0, 0, 0};                            // [한국어] 오프셋 기본값 (현재 오프셋 미지원 → 항상 0)
   printf("\n\n\n");                                      // [한국어] 커널 실행 시작을 시각적으로 구분하기 위한 빈 줄 출력
   char *mode = getenv("PTX_SIM_MODE_FUNC");             // [한국어] PTX_SIM_MODE_FUNC 환경변수로 시뮬레이션 모드 제어
   if ( mode )
      sscanf(mode,"%u", &(ctx->func_sim->g_ptx_sim_mode));  // [한국어] 환경변수 값을 파싱하여 g_ptx_sim_mode 설정 (0=성능, 1=기능)
   printf("GPGPU-Sim OpenCL API: clEnqueueNDRangeKernel '%s' (mode=%s)\n", kernel->name().c_str(),
          (ctx->func_sim->g_ptx_sim_mode)?"functional simulation":"performance simulation");  // [한국어] 실행할 커널 이름과 시뮬레이션 모드 출력
   if ( !work_dim || work_dim > 3 ) return CL_INVALID_WORK_DIMENSION;  // [한국어] work_dim은 1~3만 유효 (OpenCL 스펙 §3.3)
   size_t _local_size[3];               // [한국어] 로컬 work-group 크기 (각 차원) — CUDA threadBlock 크기에 대응
   if( local_work_size != NULL ) {
      for ( unsigned d=0; d < work_dim; d++ )
         _local_size[d]=local_work_size[d];  // [한국어] 호출자가 지정한 local_work_size 그대로 사용
   } else {
      /* [한국어] local_work_size 자동 선택:
       * OpenCL 스펙은 NULL이면 구현체가 적절한 크기를 자동 선택하도록 허용한다.
       * GPGPU-Sim은 SM의 최대 스레드 수에서 시작하여 64씩 감소하며 global을 나눌 수 있는 값을 탐색. */
      printf("GPGPU-Sim OpenCL API: clEnqueueNDRangeKernel automatic local work size selection:\n");
      for ( unsigned d=0; d < work_dim; d++ ) {
          if( d==0 ) {  // [한국어] 첫 번째 차원(x)만 실제로 계산, 나머지는 1로 고정
             if( global_work_size[d] <= command_queue->get_device()->the_device()->threads_per_core() ) {
                _local_size[d] = global_work_size[d];  // [한국어] 총 work-item 수가 SM 최대 스레드 수 이하 → 전체를 하나의 블록으로
             } else {
                // start with the maximum number of thread that a core may hold,
                // and decrement by 64 threadsuntil there is a local_work_size
                // that can perfectly divide the global_work_size.
                /* [한국어] global_work_size[d]를 균등 분할할 수 있는 최대 블록 크기 탐색:
                 * SM 최대 스레드 수에서 64씩 감소하며 나머지가 0인 값 탐색.
                 * 64 단위는 CUDA 워프 크기(32)의 2배 — 워프 정렬을 위해 사용. */
                unsigned n_thread_per_core = command_queue->get_device()->the_device()->threads_per_core();  // [한국어] SM당 최대 스레드 수 (gpgpusim.config의 n_thread_per_shader)
                size_t local_size_attempt = n_thread_per_core;   // [한국어] 최대 블록 크기부터 시작
                while (local_size_attempt > 1 and (n_thread_per_core % 64 == 0)) {  // [한국어] 64의 배수인 SM 크기에서만 64씩 감소
                   if (global_work_size[d] % local_size_attempt == 0) {
                      break;  // [한국어] 나머지가 0인 블록 크기 발견 → 탐색 종료
                   }
                   local_size_attempt -= 64;  // [한국어] 64씩 감소하며 다음 후보 시도
                }
                if (local_size_attempt == 0) local_size_attempt = 1;  // [한국어] 0이 되면 최소값 1로 fallback
                _local_size[d] = local_size_attempt;  // [한국어] 계산된 로컬 크기 적용
             }
          } else {
             _local_size[d] = 1;  // [한국어] 2차원, 3차원 자동 선택: 항상 1로 고정
          }
          printf("GPGPU-Sim OpenCL API: clEnqueueNDRangeKernel global_work_size[%u] = %zu\n", d, global_work_size[d] );  // [한국어] 각 차원의 전역 크기 출력
          printf("GPGPU-Sim OpenCL API: clEnqueueNDRangeKernel local_work_size[%u]  = %zu\n", d, _local_size[d] );       // [한국어] 각 차원의 로컬 크기 출력
      }
   }
   /* [한국어] 유효성 검사: global_work_size가 local_work_size의 배수인지 확인.
    * OpenCL 스펙: global_work_size는 local_work_size로 나누어 떨어져야 함 (§5.8). */
   for ( unsigned d=0; d < work_dim; d++ ) {
      _global_size[d] = (int)global_work_size[d];             // [한국어] int 배열로 변환 (PTX 내장 변수용)
      if ( (global_work_size[d] % _local_size[d]) != 0 )
         return CL_INVALID_WORK_GROUP_SIZE;                    // [한국어] 나눠떨어지지 않으면 에러
   }
   /* [한국어] global_work_offset 검사: 현재 비 0 오프셋은 지원하지 않는다.
    * 오프셋 지원은 구현이 복잡하여 현재 생략됨 — 발견 시 abort(). */
   if (global_work_offset != NULL){
	   for ( unsigned d=0; d < work_dim; d++ ) {
		   if (global_work_offset[d] != 0){
			   printf("GPGPU-Sim: global id offset is not supported\n");  // [한국어] 비 0 오프셋 미지원 알림
			   abort();                                                    // [한국어] 지원하지 않는 기능 → 즉시 종료
		   }
	   }
   }
   assert( global_work_size[0] == _local_size[0] * (global_work_size[0]/_local_size[0]) ); // i.e., we can divide into equal CTAs  // [한국어] 균등 분할 가능성 재확인 assert
   /* [한국어] OpenCL ND-Range → CUDA dim3 변환:
    * OpenCL의 "work-group" = CUDA의 "thread block" (dim3 BlockDim)
    * OpenCL의 "ND-Range / work-group" = CUDA의 "grid" (dim3 GridDim) */
   dim3 GridDim;
   GridDim.x = global_work_size[0]/_local_size[0];                               // [한국어] x 차원 그리드 크기 = 전체 work-items / work-group 크기
   GridDim.y = (work_dim < 2)?1:(global_work_size[1]/_local_size[1]);            // [한국어] y 차원: work_dim < 2이면 1 (단일 행)
   GridDim.z = (work_dim < 3)?1:(global_work_size[2]/_local_size[2]);            // [한국어] z 차원: work_dim < 3이면 1 (단일 깊이)
   dim3 BlockDim;
   BlockDim.x = _local_size[0];                                                   // [한국어] x 차원 블록 크기 = local_work_size[0]
   BlockDim.y = (work_dim < 2)?1:_local_size[1];                                 // [한국어] y 차원: work_dim < 2이면 1
   BlockDim.z = (work_dim < 3)?1:_local_size[2];                                 // [한국어] z 차원: work_dim < 3이면 1

   gpgpu_ptx_sim_arg_list_t params;           // [한국어] PTX 시뮬레이터용 커널 인자 리스트 (빈 리스트로 시작)
   cl_int err_val = kernel->bind_args(params); // [한국어] m_args를 gpgpu_ptx_sim_arg_list_t로 변환
   if ( err_val != CL_SUCCESS )
      return err_val;  // [한국어] 인자 변환 실패 (인덱스 불연속 등) → 에러 반환

   gpgpu_t *gpu = command_queue->get_device()->the_device();  // [한국어] 시뮬레이션 GPU 인스턴스 획득 (gpgpu_sim*)
   /* [한국어] PTX 버전 < 3.0 호환성 처리:
    * 구형 PTX(3.0 미만)에서는 OpenCL 내장 변수(%_global_size, %_work_dim 등)가
    * PTX 전역 심볼로 선언되어 있으므로 GPU 심볼 공간에 실제 값을 복사해야 한다.
    * PTX 3.0 이상에서는 이 변수들이 특수 레지스터로 처리되어 별도 복사 불필요. */
   if (kernel->get_implementation()->get_ptx_version().ver() <3.0){
	   ctx->func_sim->gpgpu_ptx_sim_memcpy_symbol( "%_global_size", _global_size, 3 * sizeof(int), 0, 1, gpu );          // [한국어] 전체 work-item 수 (3차원) → PTX %_global_size 심볼에 복사
	   ctx->func_sim->gpgpu_ptx_sim_memcpy_symbol( "%_work_dim", &work_dim, 1 * sizeof(int), 0, 1, gpu  );               // [한국어] 작업 차원 수 → PTX %_work_dim 심볼에 복사
	   ctx->func_sim->gpgpu_ptx_sim_memcpy_symbol( "%_global_num_groups", &GridDim, 3 * sizeof(int), 0, 1, gpu );        // [한국어] 그리드 차원 (work-group 수) → PTX %_global_num_groups 심볼에 복사
	   ctx->func_sim->gpgpu_ptx_sim_memcpy_symbol( "%_global_launch_offset", zeros, 3 * sizeof(int), 0, 1, gpu );        // [한국어] 전역 시작 오프셋 (0, 0, 0) → PTX %_global_launch_offset 심볼에 복사
	   ctx->func_sim->gpgpu_ptx_sim_memcpy_symbol( "%_global_block_offset", zeros, 3 * sizeof(int), 0, 1, gpu );         // [한국어] 블록 오프셋 (0, 0, 0) → PTX %_global_block_offset 심볼에 복사
   }
   /* [한국어] 커널 실행 격자(kernel_info_t) 초기화:
    * function_info, 인자 목록, GridDim, BlockDim, gpu를 결합하여 실행 준비된 격자 생성. */
   kernel_info_t *grid = ctx->func_sim->gpgpu_opencl_ptx_sim_init_grid(kernel->get_implementation(),params,GridDim,BlockDim,gpu);

   //do dynamic PDOM analysis for performance simulation scenario
   /* [한국어] PDOM(Post-Dominator) 분석:
    * 워프 내 분기(divergence) 처리를 위한 재합류 포인트를 커널 CFG에서 계산한다.
    * 성능 시뮬레이션(타이밍 모델)에서 SIMT 스택 처리에 필수이다.
    * 동일 커널을 여러 번 실행하는 경우 재분석을 피하기 위해 is_pdom_set()으로 캐시 확인. */
   std::string kname = grid->name();                         // [한국어] 커널 이름 문자열 획득
   function_info *kernel_func_info = grid->entry();          // [한국어] PTX function_info* 획득
   if (kernel_func_info->is_pdom_set()) {
      printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n", kname.c_str() );  // [한국어] 이미 분석 완료 — 재분석 스킵
   } else {
      printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n", kname.c_str() );  // [한국어] PDOM 분석 시작 알림
      kernel_func_info->do_pdom();  // [한국어] 커널 CFG에서 후위 지배자 트리 계산 — 재합류 포인트 결정
      kernel_func_info->set_pdom(); // [한국어] PDOM 분석 완료 플래그 설정 (다음 호출 시 재분석 방지)
   }
   /* [한국어] 시뮬레이션 모드에 따른 커널 실행 분기:
    * 기능 시뮬레이션(g_ptx_sim_mode=1): 타이밍 모델 없이 PTX 명령어를 순차 실행 — 기능 검증용
    * 성능 시뮬레이션(g_ptx_sim_mode=0, 기본): SM 파이프라인, 캐시, DRAM 타이밍 포함 사이클 시뮬레이션 */
   if ( ctx->func_sim->g_ptx_sim_mode )
      ctx->func_sim->gpgpu_opencl_ptx_sim_main_func( grid );  // [한국어] 기능 시뮬레이션 실행 (cuda-sim/ 엔진)
   else
      ctx->gpgpu_opencl_ptx_sim_main_perf( grid );             // [한국어] 성능 시뮬레이션 실행 (gpu-sim/ 사이클 루프)
   return CL_SUCCESS;  // [한국어] 커널 실행 완료 (동기 실행이므로 이 시점에 시뮬레이션 완료)
}

/*
 * [한국어]
 * clEnqueueReadBuffer - GPU 디바이스 메모리에서 호스트 메모리로 데이터 읽기
 *
 * @command_queue:          커맨드 큐 (GPU 디바이스 접근용)
 * @buffer:                 읽을 GPU 메모리 cl_mem 핸들 (실제로는 GPU 주소 정수)
 * @blocking_read:          CL_TRUE이면 동기 읽기, CL_FALSE이면 비동기 (현재 항상 동기 처리)
 * @offset:                 버퍼 내 시작 오프셋 (현재 무시됨)
 * @cb:                     읽을 바이트 수
 * @ptr:                    데이터를 저장할 호스트 메모리 포인터
 * @num_events_in_wait_list: 대기 이벤트 수 (현재 무시)
 * @event_wait_list:        대기 이벤트 목록 (현재 무시)
 * @event:                  완료 이벤트 출력 (현재 미사용)
 * @return:                 CL_SUCCESS
 *
 * GPU 시뮬레이션 메모리에서 호스트 메모리로 데이터를 복사한다.
 * buffer는 실제로 GPU 시뮬레이션 메모리 주소(정수)이며, (size_t)buffer로 캐스팅하여 사용.
 * offset은 현재 적용되지 않으므로 buffer가 정확한 시작 주소를 가리켜야 한다.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clEnqueueReadBuffer] → gpu->memcpy_from_gpu(ptr, GPU주소, cb)
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer(cl_command_queue    command_queue,
                    cl_mem              buffer,
                    cl_bool             blocking_read,
                    size_t              offset,
                    size_t              cb,
                    void *              ptr,
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event ) CL_API_SUFFIX__VERSION_1_0
{
   if( !blocking_read )
      gpgpusim_opencl_warning(__my_func__,__LINE__, "non-blocking read treated as blocking read");  // [한국어] 비동기 읽기 요청 → 경고 후 동기로 처리 (이벤트 완료 기다림 없음)
   gpgpu_t *gpu = command_queue->get_device()->the_device();    // [한국어] 시뮬레이션 GPU 인스턴스 획득
   gpu->memcpy_from_gpu( ptr, (size_t)buffer, cb );             // [한국어] GPU 시뮬레이션 메모리[buffer주소 .. buffer주소+cb] → 호스트 ptr로 복사
   return CL_SUCCESS;                                           // [한국어] 복사 완료 반환
}

/*
 * [한국어]
 * clEnqueueWriteBuffer - 호스트 메모리에서 GPU 디바이스 메모리로 데이터 쓰기
 *
 * @command_queue:   커맨드 큐 (GPU 디바이스 접근용)
 * @buffer:          쓸 GPU 메모리 cl_mem 핸들 (GPU 주소 정수)
 * @blocking_write:  CL_TRUE이면 동기 쓰기 (현재 항상 동기)
 * @offset:          버퍼 내 시작 오프셋 (현재 무시됨)
 * @cb:              쓸 바이트 수
 * @ptr:             읽을 호스트 메모리 포인터
 * @return:          CL_SUCCESS
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clEnqueueWriteBuffer] → gpu->memcpy_to_gpu(GPU주소, ptr, cb)
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBuffer(cl_command_queue   command_queue,
                     cl_mem             buffer,
                     cl_bool            blocking_write,
                     size_t             offset,
                     size_t             cb,
                     const void *       ptr,
                     cl_uint            num_events_in_wait_list,
                     const cl_event *   event_wait_list,
                     cl_event *         event ) CL_API_SUFFIX__VERSION_1_0
{
   if( !blocking_write )
      gpgpusim_opencl_warning(__my_func__,__LINE__, "non-blocking write treated as blocking write");  // [한국어] 비동기 쓰기 요청 → 경고 후 동기로 처리
   gpgpu_t *gpu = command_queue->get_device()->the_device();  // [한국어] 시뮬레이션 GPU 인스턴스 획득
   gpu->memcpy_to_gpu( (size_t)buffer, ptr, cb );             // [한국어] 호스트 ptr → GPU 시뮬레이션 메모리[buffer주소..buffer주소+cb]로 복사
   return CL_SUCCESS;                                         // [한국어] 복사 완료 반환
}

/*
 * [한국어]
 * clReleaseMemObject - OpenCL 메모리 객체 해제 (현재 미구현, CL_SUCCESS만 반환)
 *
 * @return: CL_SUCCESS (항상)
 *
 * 실제 구현에서는 참조 카운트를 감소시키고 0이 되면 _cl_mem을 delete해야 한다.
 * GPGPU-Sim에서는 메모리 객체가 명시적으로 해제되지 않는다 (메모리 누수).
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseMemObject(cl_mem /* memobj */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;  // [한국어] 참조 카운트 관리 미구현 — 메모리 누수 허용 (시뮬레이션 목적)
}

/*
 * [한국어]
 * clReleaseKernel - OpenCL 커널 객체 해제 (현재 미구현, CL_SUCCESS만 반환)
 *
 * @return: CL_SUCCESS (항상)
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseKernel(cl_kernel   /* kernel */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;  // [한국어] _cl_kernel 해제 미구현 — 시뮬레이션 수명 동안 유지
}

/*
 * [한국어]
 * clReleaseProgram - OpenCL 프로그램 객체 해제 (현재 미구현, CL_SUCCESS만 반환)
 *
 * @return: CL_SUCCESS (항상)
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseProgram(cl_program /* program */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;  // [한국어] _cl_program 해제 미구현 — PTX 어셈블리와 커널 맵을 계속 보유
}

/*
 * [한국어]
 * clReleaseCommandQueue - OpenCL 커맨드 큐 해제 (현재 미구현, CL_SUCCESS만 반환)
 *
 * @return: CL_SUCCESS (항상)
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseCommandQueue(cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;  // [한국어] _cl_command_queue 해제 미구현
}

/*
 * [한국어]
 * clReleaseContext - OpenCL 컨텍스트 해제 (현재 미구현, CL_SUCCESS만 반환)
 *
 * @return: CL_SUCCESS (항상)
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseContext(cl_context /* context */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;  // [한국어] _cl_context 해제 미구현 — 메모리 맵과 디바이스 참조 유지
}

/*
 * [한국어]
 * clGetPlatformIDs - 사용 가능한 OpenCL 플랫폼 목록 조회
 *
 * @num_entries:   platforms 배열의 최대 항목 수
 * @platforms:     플랫폼 ID를 저장할 배열 (NULL이면 개수만 반환)
 * @num_platforms: 발견된 플랫폼 수를 저장할 포인터 (NULL 가능)
 * @return:        CL_SUCCESS 또는 CL_INVALID_VALUE
 *
 * GPGPU-Sim은 단 하나의 플랫폼(g_gpgpu_sim_platform_id)만 제공한다.
 * num_entries==0이면서 platforms!=NULL이거나, 두 출력 포인터가 모두 NULL이면 에러.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clGetPlatformIDs] → g_gpgpu_sim_platform_id 반환
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms ) CL_API_SUFFIX__VERSION_1_0
{
   /* [한국어] OpenCL 스펙 §4.1: 다음 두 경우 CL_INVALID_VALUE 반환:
    * 1) num_entries==0인데 platforms 포인터가 NULL이 아닌 경우
    * 2) platforms와 num_platforms 둘 다 NULL인 경우 (반환 값 없음) */
   if( ((num_entries == 0) && (platforms != NULL)) ||
       ((num_platforms == NULL) && (platforms == NULL)) )
      return CL_INVALID_VALUE;
   if( (platforms != NULL) && (num_entries > 0) )
      platforms[0] = &g_gpgpu_sim_platform_id;  // [한국어] GPGPU-Sim의 단일 플랫폼 ID 반환
   if( num_platforms )
      *num_platforms = 1;  // [한국어] 항상 플랫폼 1개 반환
   return CL_SUCCESS;
}

/*
 * [한국어] clGetPlatformInfo / clGetDeviceInfo / clGetProgramInfo 등에서 사용하는
 * switch-case 응답 생성 매크로들. 세 가지 공통 동작을 캡슐화한다:
 * 1) param_value 버퍼가 너무 작으면 CL_INVALID_VALUE 반환
 * 2) param_value가 NULL이 아니면 결과 기록
 * 3) param_value_size_ret가 NULL이 아니면 실제 필요한 크기 기록
 */

/* [한국어] CL_STRING_CASE: 문자열(const char*) 응답 생성 매크로.
 * S 문자열과 null terminator를 포함한 크기로 버퍼 크기 검사 및 snprintf로 복사.
 * buf는 (char*)param_value로 선언되어야 함. */
#define CL_STRING_CASE( S ) \
      if( param_value && (param_value_size < strlen(S)+1) ) return CL_INVALID_VALUE; /* [한국어] 버퍼 크기 부족 체크 */ \
      if( param_value ) snprintf(buf,strlen(S)+1,S); /* [한국어] null terminator 포함하여 문자열 복사 */ \
      if( param_value_size_ret ) *param_value_size_ret = strlen(S)+1;  /* [한국어] 필요한 바이트 수 (null 포함) 반환 */

/* [한국어] CL_INT_CASE: cl_int 값 응답 생성 매크로. N을 cl_int로 캐스팅하여 기록. */
#define CL_INT_CASE( N ) \
      if( param_value && param_value_size < sizeof(cl_int) ) return CL_INVALID_VALUE; /* [한국어] sizeof(cl_int) 크기 체크 */ \
      if( param_value ) *((cl_int*)param_value) = (N); /* [한국어] cl_int 값 기록 */ \
      if( param_value_size_ret ) *param_value_size_ret = sizeof(cl_int); /* [한국어] sizeof(cl_int) 반환 */

/* [한국어] CL_UINT_CASE: cl_uint 값 응답 생성 매크로. cl_int와 동일하나 부호 없는 타입. */
#define CL_UINT_CASE( N ) \
      if( param_value && param_value_size < sizeof(cl_uint) ) return CL_INVALID_VALUE; /* [한국어] sizeof(cl_uint) 크기 체크 */ \
      if( param_value ) *((cl_uint*)param_value) = (N); /* [한국어] cl_uint 값 기록 */ \
      if( param_value_size_ret ) *param_value_size_ret = sizeof(cl_uint); /* [한국어] sizeof(cl_uint) 반환 */

/* [한국어] CL_ULONG_CASE: cl_ulong(64비트 부호없는 정수) 값 응답 생성 매크로.
 * 메모리 크기 등 큰 값을 반환할 때 사용 (CL_DEVICE_GLOBAL_MEM_SIZE 등). */
#define CL_ULONG_CASE( N ) \
      if( param_value && param_value_size < sizeof(cl_ulong) ) return CL_INVALID_VALUE; /* [한국어] sizeof(cl_ulong) 크기 체크 */ \
      if( param_value ) *((cl_ulong*)param_value) = (N); /* [한국어] cl_ulong 값 기록 */ \
      if( param_value_size_ret ) *param_value_size_ret = sizeof(cl_ulong); /* [한국어] sizeof(cl_ulong) 반환 */

/* [한국어] CL_BOOL_CASE: cl_bool(OpenCL 불리언) 값 응답 생성 매크로.
 * CL_TRUE/CL_FALSE를 반환하는 속성에 사용 (CL_DEVICE_AVAILABLE 등). */
#define CL_BOOL_CASE( N ) \
      if( param_value && param_value_size < sizeof(cl_bool) ) return CL_INVALID_VALUE; /* [한국어] sizeof(cl_bool) 크기 체크 */ \
      if( param_value ) *((cl_bool*)param_value) = (N); /* [한국어] cl_bool 값 기록 */ \
      if( param_value_size_ret ) *param_value_size_ret = sizeof(cl_bool); /* [한국어] sizeof(cl_bool) 반환 */

/* [한국어] CL_SIZE_CASE: size_t 값 응답 생성 매크로.
 * 워크그룹 크기 등 플랫폼 포인터 크기에 의존하는 값에 사용 (CL_KERNEL_WORK_GROUP_SIZE 등). */
#define CL_SIZE_CASE( N ) \
      if( param_value && param_value_size < sizeof(size_t) ) return CL_INVALID_VALUE; /* [한국어] sizeof(size_t) 크기 체크 */ \
      if( param_value ) *((size_t*)param_value) = (N); /* [한국어] size_t 값 기록 */ \
      if( param_value_size_ret ) *param_value_size_ret = sizeof(size_t); /* [한국어] sizeof(size_t) 반환 */

/* [한국어] CL_CASE(T, N): 임의 타입 T의 값 N을 응답 생성하는 범용 매크로.
 * cl_device_type, cl_build_status 등 특수 타입에 사용. */
#define CL_CASE( T, N ) \
      if( param_value && param_value_size < sizeof(T) ) return CL_INVALID_VALUE; /* [한국어] sizeof(T) 크기 체크 */ \
      if( param_value ) *((T*)param_value) = (N); /* [한국어] T 타입으로 캐스팅하여 값 N 기록 */ \
      if( param_value_size_ret ) *param_value_size_ret = sizeof(T); /* [한국어] sizeof(T) 반환 */

/*
 * [한국어]
 * clGetPlatformInfo - OpenCL 플랫폼 속성 조회
 *
 * @platform:          조회할 플랫폼 ID (g_gpgpu_sim_platform_id만 유효)
 * @param_name:        조회할 속성 (CL_PLATFORM_PROFILE 등)
 * @param_value_size:  param_value 버퍼 크기
 * @param_value:       결과를 기록할 버퍼 (NULL이면 크기만 반환)
 * @param_value_size_ret: 실제 필요한 크기 출력 포인터 (NULL 가능)
 * @return:            CL_SUCCESS, CL_INVALID_PLATFORM, CL_INVALID_VALUE
 *
 * GPGPU-Sim 플랫폼의 속성:
 *   PROFILE="FULL_PROFILE", VERSION="OpenCL 1.0", NAME="GPGPU-Sim",
 *   VENDOR="GPGPU-Sim.org", EXTENSIONS=" " (확장 없음)
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clGetPlatformInfo]
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformInfo(cl_platform_id   platform,
                  cl_platform_info param_name,
                  size_t           param_value_size,
                  void *           param_value,
                  size_t *         param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0
{
   if( platform == NULL || platform->m_uid != 0 )
      return CL_INVALID_PLATFORM;  // [한국어] NULL이거나 GPGPU-Sim 플랫폼이 아닌 경우 에러
   char *buf = (char*)param_value;  // [한국어] CL_STRING_CASE 매크로에서 사용할 char* 버퍼 포인터
   switch( param_name ) {
   case CL_PLATFORM_PROFILE:    CL_STRING_CASE("FULL_PROFILE"); break;  // [한국어] 플랫폼 프로파일: 전체 OpenCL 기능 지원
   case CL_PLATFORM_VERSION:    CL_STRING_CASE("OpenCL 1.0"); break;    // [한국어] OpenCL 버전: 1.0 (구형 API 지원)
   case CL_PLATFORM_NAME:       CL_STRING_CASE("GPGPU-Sim"); break;     // [한국어] 플랫폼 이름: "GPGPU-Sim"
   case CL_PLATFORM_VENDOR:     CL_STRING_CASE("GPGPU-Sim.org"); break; // [한국어] 벤더: "GPGPU-Sim.org"
   case CL_PLATFORM_EXTENSIONS: CL_STRING_CASE(" "); break;             // [한국어] 확장 목록: 빈 문자열 (확장 없음, 공백 하나로 대체)
   default:
      return CL_INVALID_VALUE;  // [한국어] 알 수 없는 param_name → CL_INVALID_VALUE
   }
   return CL_SUCCESS;
}

/* [한국어] NUM_DEVICES: GPGPU-Sim이 제공하는 디바이스 수. 단일 GPU 시뮬레이션이므로 항상 1. */
#define NUM_DEVICES 1

/*
 * [한국어]
 * clGetDeviceIDs - OpenCL 플랫폼에서 사용 가능한 디바이스 목록 조회
 *
 * @platform:    플랫폼 ID (g_gpgpu_sim_platform_id만 유효)
 * @device_type: 원하는 디바이스 타입 (CL_DEVICE_TYPE_GPU 등)
 * @num_entries: devices 배열의 최대 항목 수
 * @devices:     디바이스 ID를 저장할 배열 (NULL이면 개수만 반환)
 * @num_devices: 발견된 디바이스 수 출력 포인터 (NULL 가능)
 * @return:      CL_SUCCESS, CL_INVALID_PLATFORM, CL_INVALID_VALUE, CL_DEVICE_NOT_FOUND 등
 *
 * CPU 타입은 거부하며, GPU/ACCELERATOR/DEFAULT/ALL은 GPGPUSim_Init()의 단일 디바이스를 반환.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clGetDeviceIDs] → GPGPUSim_Init() (GPU 미초기화 시)
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id   platform,
               cl_device_type   device_type,
               cl_uint          num_entries,
               cl_device_id *   devices,
               cl_uint *        num_devices ) CL_API_SUFFIX__VERSION_1_0
{
   if( platform == NULL || platform->m_uid != 0 )
      return CL_INVALID_PLATFORM;  // [한국어] NULL이거나 GPGPU-Sim 플랫폼이 아닌 경우 에러
   if( (num_entries == 0 && devices != NULL) ||
       (num_devices == NULL && devices == NULL) )
      return CL_INVALID_VALUE;     // [한국어] OpenCL 스펙 §4.2: 유효하지 않은 파라미터 조합

   switch( device_type ) {
   case CL_DEVICE_TYPE_CPU:
      // Some benchmarks (e.g. ComD benchmark from Mantevo package) looks for CPU and GPU to choose among, so it is not wise to abort execution because of GPGPUsim is not a CPU !.
      // [한국어] CPU 타입: GPGPU-Sim은 GPU 시뮬레이터이므로 CPU 디바이스 없음.
      //          abort 대신 CL_DEVICE_NOT_FOUND 반환 — Mantevo ComD 벤치마크 호환성 유지.
      printf("GPGPU-Sim OpenCL API: unsupported device type %lx\n", device_type );
      return CL_DEVICE_NOT_FOUND;  // [한국어] CPU 디바이스 미지원 → 발견되지 않음 반환
      break;
   case CL_DEVICE_TYPE_DEFAULT:     // [한국어] 기본 디바이스 타입
   case CL_DEVICE_TYPE_GPU:         // [한국어] GPU 타입 — GPGPU-Sim 지원
   case CL_DEVICE_TYPE_ACCELERATOR: // [한국어] 가속기 타입 — GPGPU-Sim 지원
   case CL_DEVICE_TYPE_ALL:         // [한국어] 모든 타입 — GPGPU-Sim 포함
      if( devices != NULL )
         devices[0] = GPGPUSim_Init();  // [한국어] 싱글톤 GPU 디바이스를 devices[0]에 기록
      if( num_devices )
         *num_devices = NUM_DEVICES;   // [한국어] 디바이스 수 = 1 기록
      break;
   default:
      return CL_INVALID_DEVICE_TYPE;  // [한국어] 알 수 없는 device_type → CL_INVALID_DEVICE_TYPE
   }
   return CL_SUCCESS;
}

/*
 * [한국어]
 * clGetDeviceInfo - OpenCL 디바이스 속성 조회
 *
 * @device:           조회할 디바이스 (GPGPUSim_Init()이 반환한 유일한 디바이스만 유효)
 * @param_name:       조회할 속성 (CL_DEVICE_NAME, CL_DEVICE_GLOBAL_MEM_SIZE 등)
 * @param_value_size: param_value 버퍼 크기
 * @param_value:      결과 저장 버퍼 (NULL이면 크기만 반환)
 * @param_value_size_ret: 실제 필요 크기 출력 포인터 (NULL 가능)
 * @return:           CL_SUCCESS, CL_INVALID_DEVICE, CL_INVALID_VALUE
 *
 * GPGPU-Sim 시뮬레이션 GPU의 속성을 반환한다. 실제 GPU 속성은 gpgpusim.config 설정에서
 * 읽어와서 반환하며, 일부는 하드코딩된 값을 사용한다.
 * 주요 반환값:
 *   - CL_DEVICE_MAX_COMPUTE_UNITS: num_shader() (SM 수, 설정 파일로 결정)
 *   - CL_DEVICE_MAX_CLOCK_FREQUENCY: shader_clock() (MHz, 설정 파일로 결정)
 *   - CL_DEVICE_GLOBAL_MEM_SIZE: 1GB (하드코딩)
 *   - CL_DEVICE_LOCAL_MEM_SIZE: shared_mem_size() (설정 파일로 결정)
 *   - CL_DEVICE_ADDRESS_BITS: 32 (하드코딩 — GPGPU-Sim의 32비트 주소 모델)
 *   - CL_DEVICE_MAX_WORK_GROUP_SIZE: threads_per_core() (SM당 최대 스레드 수)
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clGetDeviceInfo] → device->the_device()->get_config() 등
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceInfo(cl_device_id    device,
                cl_device_info  param_name,
                size_t          param_value_size,
                void *          param_value,
                size_t *        param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
   if( device != GPGPUSim_Init() )
      return CL_INVALID_DEVICE;   // [한국어] GPGPU-Sim의 싱글톤 디바이스가 아닌 경우 → 잘못된 디바이스 에러
   char *buf = (char*)param_value;  // [한국어] CL_STRING_CASE 매크로에서 사용할 char* 버퍼
   switch( param_name ) {
   case CL_DEVICE_NAME: CL_STRING_CASE( "GPGPU-Sim" ); break;  // [한국어] 디바이스 이름: "GPGPU-Sim"
   case CL_DEVICE_GLOBAL_MEM_SIZE: CL_ULONG_CASE( 1024*1024*1024 ); break;  // [한국어] 전역 메모리 크기: 1GB (하드코딩 — 실제 값은 설정 파일로 결정)
   case CL_DEVICE_MAX_COMPUTE_UNITS: CL_UINT_CASE( device->the_device()->get_config().num_shader() ); break;  // [한국어] 컴퓨트 유닛 수 = SM 수 (gpgpusim.config의 gpu_n_shader)
   case CL_DEVICE_MAX_CLOCK_FREQUENCY: CL_UINT_CASE( device->the_device()->shader_clock() ); break;  // [한국어] 최대 클럭 주파수 (MHz) = 설정된 SM 클럭 속도
   case CL_DEVICE_VENDOR:CL_STRING_CASE("GPGPU-Sim.org"); break;  // [한국어] 벤더: "GPGPU-Sim.org"
   case CL_DEVICE_VERSION: CL_STRING_CASE("OpenCL 1.0"); break;   // [한국어] OpenCL 버전: 1.0
   case CL_DRIVER_VERSION: CL_STRING_CASE("1.0"); break;          // [한국어] 드라이버 버전: "1.0" (GPGPU-Sim 드라이버)
   case CL_DEVICE_TYPE: CL_CASE(cl_device_type, CL_DEVICE_TYPE_GPU); break;  // [한국어] 디바이스 타입: GPU
   case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: CL_INT_CASE( 3 ); break;  // [한국어] 최대 work 차원: 3 (x, y, z)
   case CL_DEVICE_MAX_WORK_ITEM_SIZES:  // [한국어] 각 차원별 최대 work-item 수
      if( param_value && param_value_size < 3*sizeof(size_t) ) return CL_INVALID_VALUE;  // [한국어] 3*sizeof(size_t) 크기 체크
      if( param_value ) {
         unsigned n_thread_per_shader = device->the_device()->threads_per_core();  // [한국어] SM당 최대 스레드 수 조회
         ((size_t*)param_value)[0] = n_thread_per_shader;  // [한국어] x 차원 최대 work-item 수
         ((size_t*)param_value)[1] = n_thread_per_shader;  // [한국어] y 차원 최대 work-item 수
         ((size_t*)param_value)[2] = n_thread_per_shader;  // [한국어] z 차원 최대 work-item 수
      }
      if( param_value_size_ret ) *param_value_size_ret = 3*sizeof(cl_uint);  // [한국어] 필요 크기: 3 * sizeof(cl_uint)
      break;
   case CL_DEVICE_MAX_WORK_GROUP_SIZE: CL_INT_CASE( device->the_device()->threads_per_core() ); break;  // [한국어] 최대 work-group 크기 = SM당 최대 스레드 수
   case CL_DEVICE_ADDRESS_BITS: CL_INT_CASE( 32 ); break;  // [한국어] 주소 비트 수: 32 (GPGPU-Sim의 32비트 시뮬레이션 주소 공간)
   case CL_DEVICE_AVAILABLE: CL_BOOL_CASE( CL_TRUE ); break;          // [한국어] 디바이스 사용 가능: true
   case CL_DEVICE_COMPILER_AVAILABLE: CL_BOOL_CASE( CL_TRUE ); break; // [한국어] 컴파일러 가용: true (nvopencl_wrapper 사용 가능)
   case CL_DEVICE_IMAGE_SUPPORT: CL_INT_CASE( CL_TRUE ); break;       // [한국어] 이미지 지원: true (기본 이미지 형식 목록 제공)
   case CL_DEVICE_MAX_READ_IMAGE_ARGS: CL_INT_CASE( 128 ); break;     // [한국어] 최대 읽기 이미지 인자 수: 128
   case CL_DEVICE_MAX_WRITE_IMAGE_ARGS: CL_INT_CASE( 8 ); break;      // [한국어] 최대 쓰기 이미지 인자 수: 8
   case CL_DEVICE_IMAGE2D_MAX_HEIGHT: CL_INT_CASE( 8192 ); break;     // [한국어] 2D 이미지 최대 높이: 8192 픽셀
   case CL_DEVICE_IMAGE2D_MAX_WIDTH: CL_INT_CASE( 8192 ); break;      // [한국어] 2D 이미지 최대 너비: 8192 픽셀
   case CL_DEVICE_IMAGE3D_MAX_HEIGHT: CL_INT_CASE( 2048 ); break;     // [한국어] 3D 이미지 최대 높이: 2048
   case CL_DEVICE_IMAGE3D_MAX_WIDTH: CL_INT_CASE( 2048 ); break;      // [한국어] 3D 이미지 최대 너비: 2048
   case CL_DEVICE_IMAGE3D_MAX_DEPTH: CL_INT_CASE( 2048 ); break;      // [한국어] 3D 이미지 최대 깊이: 2048
   case CL_DEVICE_MAX_MEM_ALLOC_SIZE: CL_INT_CASE( 128*1024*1024 ); break;  // [한국어] 단일 할당 최대 크기: 128MB (하드코딩)
   case CL_DEVICE_ERROR_CORRECTION_SUPPORT: CL_INT_CASE( 0 ); break;  // [한국어] ECC 지원: 없음 (시뮬레이터이므로)
   case CL_DEVICE_LOCAL_MEM_TYPE: CL_INT_CASE( CL_LOCAL ); break;     // [한국어] 로컬 메모리 타입: CL_LOCAL (실제 하드웨어 공유 메모리)
   case CL_DEVICE_LOCAL_MEM_SIZE: CL_ULONG_CASE( device->the_device()->shared_mem_size() ); break;  // [한국어] 로컬(공유) 메모리 크기: gpgpusim.config의 gpgpu_shmem_size
   case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: CL_ULONG_CASE( 64 * 1024 ); break;  // [한국어] 상수 버퍼 최대 크기: 64KB (일반 NVIDIA GPU 기준값)
   case CL_DEVICE_QUEUE_PROPERTIES: CL_INT_CASE( CL_QUEUE_PROFILING_ENABLE ); break;  // [한국어] 지원 큐 속성: 프로파일링 가능 (실제로는 미구현이지만 지원 선언)
   case CL_DEVICE_EXTENSIONS:  // [한국어] 확장 목록: 없음 (빈 문자열 하나 = null terminator만)
      if( param_value && (param_value_size < 1) ) return CL_INVALID_VALUE;  // [한국어] 최소 1바이트 필요
      if( param_value ) buf[0]=0;                                            // [한국어] 빈 문자열 (null terminator)
      if( param_value_size_ret ) *param_value_size_ret = 1;                 // [한국어] 크기 = 1 (null terminator만)
      break;
   case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:   CL_INT_CASE(1); break;  // [한국어] char 벡터 너비 선호: 1 (스칼라 연산만)
   case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:  CL_INT_CASE(1); break;  // [한국어] short 벡터 너비 선호: 1
   case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:    CL_INT_CASE(1); break;  // [한국어] int 벡터 너비 선호: 1
   case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:   CL_INT_CASE(1); break;  // [한국어] long 벡터 너비 선호: 1
   case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:  CL_INT_CASE(1); break;  // [한국어] float 벡터 너비 선호: 1
   case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: CL_INT_CASE(0); break;  // [한국어] double 벡터 너비 선호: 0 (double 미지원)
   case CL_DEVICE_SINGLE_FP_CONFIG: CL_INT_CASE(0); break;               // [한국어] 단정밀도 FP 설정: 0 (특별한 FP 기능 없음)
   case CL_DEVICE_MEM_BASE_ADDR_ALIGN: CL_INT_CASE(256*8); break;        // [한국어] 메모리 기준 주소 정렬: 256바이트 * 8비트/바이트 = 2048비트 (256바이트 정렬)
   default:
      opencl_not_implemented(__my_func__,__LINE__);  // [한국어] 알 수 없는 속성 → 미구현 처리 (abort)
   }
   return CL_SUCCESS;
}

/*
 * [한국어]
 * clFinish - 커맨드 큐의 모든 커맨드 완료 대기 (현재 no-op)
 *
 * @return: CL_SUCCESS (항상)
 *
 * GPGPU-Sim에서 clEnqueueNDRangeKernel이 동기적으로 실행되므로 이 함수는 불필요.
 * 호환성을 위해 CL_SUCCESS만 반환한다.
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clFinish(cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;  // [한국어] 동기 실행 모델이므로 별도 대기 불필요 → 항상 즉시 완료
}

/*
 * [한국어]
 * clGetProgramInfo - OpenCL 프로그램 객체의 속성 조회
 *
 * @program:              조회할 cl_program 객체
 * @param_name:           조회할 속성 (CL_PROGRAM_REFERENCE_COUNT, CL_PROGRAM_BINARIES 등)
 * @param_value_size:     param_value 버퍼 크기
 * @param_value:          결과를 기록할 버퍼 (NULL이면 크기만 반환)
 * @param_value_size_ret: 실제 필요한 크기 출력 포인터 (NULL 가능)
 * @return:               CL_SUCCESS, CL_INVALID_PROGRAM, CL_INVALID_VALUE
 *
 * 주요 케이스:
 *   - CL_PROGRAM_REFERENCE_COUNT: 참조 카운트 (항상 1 반환 — 참조 카운팅 미구현)
 *   - CL_PROGRAM_CONTEXT: 프로그램이 속한 컨텍스트 포인터
 *   - CL_PROGRAM_NUM_DEVICES: 디바이스 수 (NUM_DEVICES = 1)
 *   - CL_PROGRAM_DEVICES: 디바이스 ID 목록 (GPGPUSim_Init()의 단일 디바이스)
 *   - CL_PROGRAM_SOURCE: 미구현 (abort)
 *   - CL_PROGRAM_BINARY_SIZES: 빌드된 PTX 크기 배열 (NUM_DEVICES 개)
 *   - CL_PROGRAM_BINARIES: 빌드된 PTX 바이너리 (빌드 후 호출해야 함)
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clGetProgramInfo] → program->get_ptx()/get_ptx_size()
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetProgramInfo(cl_program         program,
                 cl_program_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0
{
   if( program == NULL )
      return CL_INVALID_PROGRAM;   // [한국어] NULL 프로그램 에러
   char *tmp=NULL;                 // [한국어] CL_PROGRAM_BINARIES에서 get_ptx() 버퍼 임시 저장용
   size_t len=0;                   // [한국어] PTX 바이너리 길이 임시 변수
   switch( param_name ) {
   case CL_PROGRAM_REFERENCE_COUNT:
      CL_INT_CASE(1);                // [한국어] 참조 카운트: 항상 1 (참조 카운팅 미구현)
      break;
   case CL_PROGRAM_CONTEXT:         // [한국어] 프로그램이 속한 컨텍스트 반환
      if( param_value && param_value_size < sizeof(cl_context)) return CL_INVALID_VALUE;  // [한국어] 버퍼 크기 검사
      if( param_value ) *((cl_context*)param_value) = program->get_context();              // [한국어] 컨텍스트 포인터 기록
      if( param_value_size_ret ) *param_value_size_ret = sizeof(cl_context);               // [한국어] sizeof(cl_context) 반환
      break;
   case CL_PROGRAM_NUM_DEVICES:
      CL_INT_CASE(NUM_DEVICES);      // [한국어] 디바이스 수: NUM_DEVICES(=1)
      break;
   case CL_PROGRAM_DEVICES:          // [한국어] 디바이스 ID 목록 반환 (GPGPUSim 단일 디바이스)
      if( param_value && param_value_size < NUM_DEVICES * sizeof(cl_device_id) )
         return CL_INVALID_VALUE;    // [한국어] NUM_DEVICES * sizeof(cl_device_id) 크기 검사
      if( param_value ) {
         assert( NUM_DEVICES == 1 );  // [한국어] 단일 디바이스 가정 검증
         ((cl_device_id*)param_value)[0] = GPGPUSim_Init();  // [한국어] 싱글톤 디바이스 ID 기록
      }
      if( param_value_size_ret ) *param_value_size_ret = sizeof(cl_device_id);  // [한국어] 1개 디바이스 ID 크기
      break;
   case CL_PROGRAM_SOURCE:
      opencl_not_implemented(__my_func__,__LINE__);  // [한국어] OpenCL C 소스 반환 미구현 → abort
      break;
   case CL_PROGRAM_BINARY_SIZES:    // [한국어] PTX 바이너리 크기 배열 반환 (NUM_DEVICES 개의 size_t)
      if( param_value && param_value_size < NUM_DEVICES * sizeof(size_t) ) return CL_INVALID_VALUE;  // [한국어] NUM_DEVICES * sizeof(size_t) 크기 검사
      if( param_value ) *((size_t*)param_value) = program->get_ptx_size();  // [한국어] PTX 전체 크기 기록 (단일 디바이스이므로 배열[0]에만)
      if( param_value_size_ret ) *param_value_size_ret = NUM_DEVICES*sizeof(size_t);  // [한국어] size_t 배열 크기 반환
      break;
   case CL_PROGRAM_BINARIES:         // [한국어] PTX 바이너리 내용 반환
      len = program->get_ptx_size();  // [한국어] 전체 PTX 크기 계산
      tmp = program->get_ptx();       // [한국어] 연결된 PTX 버퍼 획득 (힙 할당 — 여기서는 free 하지 않음: 메모리 누수)
      if( param_value ) memcpy( ((char**)param_value)[0], tmp, len );  // [한국어] param_value[0] 버퍼에 PTX 복사 (OpenCL 스펙: param_value는 char** 타입)
      if( param_value_size_ret ) *param_value_size_ret = len;           // [한국어] PTX 길이 반환
      break;
   default:
      return CL_INVALID_VALUE;  // [한국어] 알 수 없는 param_name → CL_INVALID_VALUE
      break;
   }
   return CL_SUCCESS;
}

/*
 * [한국어]
 * clGetProgramBuildInfo - OpenCL 프로그램 빌드 정보 조회
 *
 * @program:              조회할 cl_program 객체
 * @device:               빌드 대상 디바이스 (현재 무시)
 * @param_name:           조회할 속성 (CL_PROGRAM_BUILD_STATUS/OPTIONS/LOG/BINARY_TYPE)
 * @param_value_size:     param_value 버퍼 크기
 * @param_value:          결과를 기록할 버퍼 (NULL이면 크기만 반환)
 * @param_value_size_ret: 실제 필요한 크기 출력 포인터 (NULL 가능)
 * @return:               CL_SUCCESS, CL_INVALID_VALUE
 *
 * GPGPU-Sim에서는 nvopencl_wrapper가 OpenCL C → PTX 컴파일 성공 여부만 반환하므로
 * BUILD_STATUS는 항상 CL_BUILD_SUCCESS, BUILD_LOG/OPTIONS는 빈 문자열,
 * BINARY_TYPE은 CL_PROGRAM_BINARY_TYPE_EXECUTABLE로 고정 반환한다.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clGetProgramBuildInfo]
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetProgramBuildInfo (cl_program            program,
                       cl_device_id          device,
                       cl_program_build_info param_name,
                       size_t                param_value_size,
                       void *                param_value,
                       size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
   char *buf = (char*)param_value;  // [한국어] CL_STRING_CASE 매크로에서 사용할 char* 버퍼 포인터

   switch( param_name ) {
   case CL_PROGRAM_BUILD_STATUS:
      CL_CASE( cl_build_status, CL_BUILD_SUCCESS );  // [한국어] 빌드 항상 성공으로 보고
      break;
   case CL_PROGRAM_BUILD_OPTIONS:
   case CL_PROGRAM_BUILD_LOG:
      CL_STRING_CASE( "" );  // [한국어] 빌드 옵션/로그는 빈 문자열 반환
      break;
   case CL_PROGRAM_BINARY_TYPE:
      CL_CASE( cl_program_binary_type, CL_PROGRAM_BINARY_TYPE_EXECUTABLE );  // [한국어] 실행 가능한 바이너리(PTX)로 표시
      break;
   default:
      return CL_INVALID_VALUE;  // [한국어] 알 수 없는 속성
      break;
   }

   return CL_SUCCESS;
}

/*
 * [한국어]
 * clEnqueueCopyBuffer - GPU 버퍼 간 메모리 복사
 *
 * @command_queue:          복사를 수행할 커맨드 큐
 * @src_buffer:             원본 버퍼 cl_mem 핸들
 * @dst_buffer:             대상 버퍼 cl_mem 핸들
 * @src_offset:             원본 버퍼 내 시작 오프셋 (바이트)
 * @dst_offset:             대상 버퍼 내 시작 오프셋 (바이트)
 * @cb:                     복사할 바이트 수
 * @num_events_in_wait_list: 대기 이벤트 수 (현재 0만 지원, 0 초과 시 미구현)
 * @event_wait_list:        대기 이벤트 목록 (현재 무시)
 * @event:                  완료 이벤트 출력 (현재 미사용)
 * @return:                 CL_SUCCESS 또는 에러 코드
 *
 * src/dst 버퍼가 호스트 메모리 기반인지 GPU 메모리 기반인지에 따라 세 가지 경로로 분기:
 *   - H→D: memcpy_to_gpu
 *   - D→H: memcpy_from_gpu
 *   - D→D: memcpy_gpu_to_gpu
 * H→H 복사는 미구현이다.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clEnqueueCopyBuffer] → lookup_mem → gpu->memcpy_*()
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBuffer(cl_command_queue    command_queue, 
                    cl_mem              src_buffer,
                    cl_mem              dst_buffer, 
                    size_t              src_offset,
                    size_t              dst_offset,
                    size_t              cb, 
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event ) CL_API_SUFFIX__VERSION_1_0
{
   if( num_events_in_wait_list > 0 ) 
      opencl_not_implemented(__my_func__,__LINE__);  // [한국어] 이벤트 대기 복사는 미구현
   if( command_queue == NULL || !command_queue->is_valid() ) 
      return CL_INVALID_COMMAND_QUEUE;  // [한국어] NULL 또는 유효하지 않은 커맨드 큐
   cl_context context = command_queue->get_context();           // [한국어] 커맨드 큐에서 컨텍스트 획득
   cl_mem src = context->lookup_mem( src_buffer );              // [한국어] src_buffer로 _cl_mem* 조회
   cl_mem dst = context->lookup_mem( dst_buffer );              // [한국어] dst_buffer로 _cl_mem* 조회
   if( src == NULL || dst == NULL ) 
      return CL_INVALID_MEM_OBJECT;  // [한국어] 유효하지 않은 cl_mem 핸들

   gpgpu_t *gpu = command_queue->get_device()->the_device();    // [한국어] 시뮬레이션 GPU 인스턴스 획득
   /* [한국어] 버퍼 위치 조합에 따른 복사 방향 결정:
    * is_on_host()가 true이면 호스트 메모리, false이면 GPU 메모리 */
   if( src->is_on_host() && !dst->is_on_host() )
      gpu->memcpy_to_gpu( ((size_t)dst->device_ptr())+dst_offset, ((char*)src->host_ptr())+src_offset, cb );      // [한국어] 호스트 → GPU
   else if( !src->is_on_host() && dst->is_on_host() ) 
      gpu->memcpy_from_gpu( ((char*)dst->host_ptr())+dst_offset, ((size_t)src->device_ptr())+src_offset, cb );    // [한국어] GPU → 호스트
   else if( !src->is_on_host() && !dst->is_on_host() ) 
      gpu->memcpy_gpu_to_gpu( ((size_t)dst->device_ptr())+dst_offset, ((size_t)src->device_ptr())+src_offset, cb );  // [한국어] GPU → GPU
   else
      opencl_not_implemented(__my_func__,__LINE__);  // [한국어] 호스트 → 호스트 복사 미구현
   return CL_SUCCESS;
}

/*
 * [한국어]
 * clGetKernelWorkGroupInfo - 커널 워크그룹 관련 정보 조회
 *
 * @kernel:                 조회할 cl_kernel 객체
 * @device:                 조회 대상 디바이스
 * @param_name:             조회할 속성 (CL_KERNEL_WORK_GROUP_SIZE 등)
 * @param_value_size:       param_value 버퍼 크기
 * @param_value:            결과를 기록할 버퍼 (NULL이면 크기만 반환)
 * @param_value_size_ret:   실제 필요한 크기 출력 포인터 (NULL 가능)
 * @return:                 CL_SUCCESS, CL_INVALID_KERNEL, CL_INVALID_VALUE
 *
 * CL_KERNEL_WORK_GROUP_SIZE만 완전히 구현되어 있으며, _cl_kernel::get_workgroup_size()가
 * PTX 커널의 레지스터 사용량과 SM 물리적 최대 스레드 수를 기반으로 계산한다.
 * CL_KERNEL_COMPILE_WORK_GROUP_SIZE와 CL_KERNEL_LOCAL_MEM_SIZE는 미구현 처리 후
 * 공유 메모리 크기를 반환한다 (비표준 fallback).
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clGetKernelWorkGroupInfo] → kernel->get_workgroup_size()
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetKernelWorkGroupInfo(cl_kernel                  kernel,
                         cl_device_id               device,
                         cl_kernel_work_group_info  param_name,
                         size_t                     param_value_size,
                         void *                     param_value,
                         size_t *                   param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0
{
   if( kernel == NULL ) 
      return CL_INVALID_KERNEL;  // [한국어] NULL 커널 에러
   switch( param_name ) {
   case CL_KERNEL_WORK_GROUP_SIZE:
      CL_SIZE_CASE( kernel->get_workgroup_size(device) );  // [한국어] 레지스터 기반 권장 워크그룹 크기 반환
      break;
   case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
   case CL_KERNEL_LOCAL_MEM_SIZE:
      opencl_not_implemented(__my_func__,__LINE__);  // [한국어] 미구현 속성 — abort 대신 아래 fallback 수행
      *(size_t *)param_value = device->the_device()->shared_mem_size();  // [한국어] 비표준 fallback: 공유 메모리 크기 반환
      break;
   default:
      return CL_INVALID_VALUE;  // [한국어] 알 수 없는 속성
      break;
   }
   return CL_SUCCESS;
}

/*
 * [한국어]
 * clWaitForEvents - 지정한 이벤트들이 완료될 때까지 대기 (현재 no-op)
 *
 * @return: CL_SUCCESS (항상)
 *
 * GPGPU-Sim에서 clEnqueueNDRangeKernel은 동기적으로 실행되므로 이벤트 대기가 의미 없다.
 * 호환성을 위해 즉시 CL_SUCCESS를 반환한다.
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clWaitForEvents(cl_uint             /* num_events */,
                const cl_event *    /* event_list */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;  // [한국어] 동기 실행 모델이므로 즉시 반환
}

/*
 * [한국어]
 * clReleaseEvent - OpenCL 이벤트 객체 해제 (현재 미구현, CL_SUCCESS만 반환)
 *
 * @return: CL_SUCCESS (항상)
 *
 * 참조 카운팅 기반 이벤트 해제는 현재 구현되어 있지 않다.
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clReleaseEvent(cl_event /* event */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;  // [한국어] 이벤트 해제 미구현
}

/*
 * [한국어]
 * clGetCommandQueueInfo - OpenCL 커맨드 큐 속성 조회
 *
 * @command_queue:        조회할 cl_command_queue 객체
 * @param_name:           조회할 속성 (CL_QUEUE_CONTEXT/DEVICE/REFERENCE_COUNT/PROPERTIES)
 * @param_value_size:     param_value 버퍼 크기
 * @param_value:          결과를 기록할 버퍼 (NULL이면 크기만 반환)
 * @param_value_size_ret: 실제 필요한 크기 출력 포인터 (NULL 가능)
 * @return:               CL_SUCCESS, CL_INVALID_COMMAND_QUEUE, CL_INVALID_VALUE
 *
 * 커맨드 큐 생성 시 저장핸 context, device, properties를 반환하며,
 * 참조 카운트는 참조 카운팅 미구현으로 항상 1을 반환한다.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clGetCommandQueueInfo] → command_queue->get_*()
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetCommandQueueInfo(cl_command_queue      command_queue,
                      cl_command_queue_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0
{
   if( command_queue == NULL ) 
      return CL_INVALID_COMMAND_QUEUE;  // [한국어] NULL 커맨드 큐 에러
   switch( param_name ) {
   case CL_QUEUE_CONTEXT: CL_CASE(cl_context, command_queue->get_context()); break;        // [한국어] 큐의 컨텍스트 반환
   case CL_QUEUE_DEVICE: CL_CASE(cl_device_id, command_queue->get_device()); break;        // [한국어] 큐의 디바이스 반환
   case CL_QUEUE_REFERENCE_COUNT: CL_CASE(cl_uint,1); break;                               // [한국어] 참조 카운트: 항상 1
   case CL_QUEUE_PROPERTIES: CL_CASE(cl_command_queue_properties, command_queue->get_properties()); break;  // [한국어] 큐 생성 속성 반환
   default:
      return CL_INVALID_VALUE;  // [한국어] 알 수 없는 속성
   }
   return CL_SUCCESS;
}

/*
 * [한국어]
 * clFlush - 커맨드 큐의 모든 명령을 디바이스에 제출 (현재 no-op)
 *
 * @return: CL_SUCCESS (항상)
 *
 * GPGPU-Sim의 clEnqueueNDRangeKernel은 즉시 동기 실행되므로 flush가 의미 없다.
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clFlush(cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0
{
   return CL_SUCCESS;  // [한국어] 동기 실행 모델이므로 즉시 반환
}

/*
 * [한국어]
 * clGetSupportedImageFormats - 디바이스에서 지원하는 이미지 포맷 목록 조회
 *
 * @context:             조회할 컨텍스트
 * @flags:               메모리 플래그 (CL_MEM_READ_ONLY만 지원)
 * @image_type:          이미지 객체 타입 (CL_MEM_OBJECT_IMAGE2D만 지원)
 * @num_entries:         image_formats 배열의 최대 항목 수
 * @image_formats:       포맷 목록을 저장할 배열 (NULL이면 개수만 반환)
 * @num_image_formats:   발견된 포맷 수 출력 포인터 (NULL 가능)
 * @return:              CL_SUCCESS, CL_INVALID_CONTEXT, CL_INVALID_VALUE
 *
 * CL_MEM_READ_ONLY | CL_MEM_OBJECT_IMAGE2D 조합에 대해 71개의 이미지 포맷을 하드코딩하여 반환.
 * CL_MEM_READ_ONLY이 아니거나 2D 이미지가 아닌 경우 미구현 또는 에러 처리.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clGetSupportedImageFormats]
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetSupportedImageFormats(cl_context           context,
                           cl_mem_flags         flags,
                           cl_mem_object_type   image_type,
                           cl_uint              num_entries,
                           cl_image_format *    image_formats,
                           cl_uint *            num_image_formats) CL_API_SUFFIX__VERSION_1_0
{
   if( !context ) 
      return CL_INVALID_CONTEXT;  // [한국어] NULL 컨텍스트 에러
   if( flags == CL_MEM_READ_ONLY ) {
      // [한국어] 2D 이미지 타입 지원 (조건이 중복되었으나 기존 코드 유지)
      if( image_type == CL_MEM_OBJECT_IMAGE2D || image_type == CL_MEM_OBJECT_IMAGE2D ) {
         if( num_entries == 0 || image_formats == NULL ) {
            if( num_image_formats != NULL ) 
               *num_image_formats = 71;  // [한국어] 포맷 수만 반환
         } else {
            if( num_entries != 71 ) 
               opencl_not_implemented(__my_func__,__LINE__);  // [한국어] 71개가 아닌 버퍼는 미구현
            /* [한국어] 하드코딩된 71개 이미지 포맷: R/A/RG/RA/RGBA/BGRA/ARGB/INTENSITY/LUMINANCE
             * 각 채널 순서별로 FLOAT/HALF_FLOAT/UNORM_INT8/UNORM_INT16/SNORM_INT16/SIGNED_INT8/16/32/UNSIGNED_INT8/16/32 등 조합 */
            image_formats[0].image_channel_order = CL_R;                        image_formats[0].image_channel_data_type = CL_FLOAT               ;
            image_formats[1].image_channel_order = CL_R;                        image_formats[1].image_channel_data_type = CL_HALF_FLOAT          ;
            image_formats[2].image_channel_order = CL_R;                        image_formats[2].image_channel_data_type = CL_UNORM_INT8          ;
            image_formats[3].image_channel_order = CL_R;                        image_formats[3].image_channel_data_type = CL_UNORM_INT16         ;
            image_formats[4].image_channel_order = CL_R;                        image_formats[4].image_channel_data_type = CL_SNORM_INT16         ;
            image_formats[5].image_channel_order = CL_R;                        image_formats[5].image_channel_data_type = CL_SIGNED_INT8         ;
            image_formats[6].image_channel_order = CL_R;                        image_formats[6].image_channel_data_type = CL_SIGNED_INT16        ;
            image_formats[7].image_channel_order = CL_R;                        image_formats[7].image_channel_data_type = CL_SIGNED_INT32        ;
            image_formats[8].image_channel_order = CL_R;                        image_formats[8].image_channel_data_type = CL_UNSIGNED_INT8       ;
            image_formats[9].image_channel_order = CL_R;                        image_formats[9].image_channel_data_type = CL_UNSIGNED_INT16      ;
            image_formats[10].image_channel_order = CL_R;                       image_formats[10].image_channel_data_type = CL_UNSIGNED_INT32     ;
            image_formats[11].image_channel_order = CL_A;                       image_formats[11].image_channel_data_type = CL_FLOAT              ;
            image_formats[12].image_channel_order = CL_A;                       image_formats[12].image_channel_data_type = CL_HALF_FLOAT         ;
            image_formats[13].image_channel_order = CL_A;                       image_formats[13].image_channel_data_type = CL_UNORM_INT8         ;
            image_formats[14].image_channel_order = CL_A;                       image_formats[14].image_channel_data_type = CL_UNORM_INT16        ;
            image_formats[15].image_channel_order = CL_A;                       image_formats[15].image_channel_data_type = CL_SNORM_INT16        ;
            image_formats[16].image_channel_order = CL_A;                       image_formats[16].image_channel_data_type = CL_SIGNED_INT8        ;
            image_formats[17].image_channel_order = CL_A;                       image_formats[17].image_channel_data_type = CL_SIGNED_INT16       ;
            image_formats[18].image_channel_order = CL_A;                       image_formats[18].image_channel_data_type = CL_SIGNED_INT32       ;
            image_formats[19].image_channel_order = CL_A;                       image_formats[19].image_channel_data_type = CL_UNSIGNED_INT8      ;
            image_formats[20].image_channel_order = CL_A;                       image_formats[20].image_channel_data_type = CL_UNSIGNED_INT16     ;
            image_formats[21].image_channel_order = CL_A;                       image_formats[21].image_channel_data_type = CL_UNSIGNED_INT32     ;
            image_formats[22].image_channel_order = CL_RG;                      image_formats[22].image_channel_data_type = CL_FLOAT              ;
            image_formats[23].image_channel_order = CL_RG;                      image_formats[23].image_channel_data_type = CL_HALF_FLOAT         ;
            image_formats[24].image_channel_order = CL_RG;                      image_formats[24].image_channel_data_type = CL_UNORM_INT8         ;
            image_formats[25].image_channel_order = CL_RG;                      image_formats[25].image_channel_data_type = CL_UNORM_INT16        ;
            image_formats[26].image_channel_order = CL_RG;                      image_formats[26].image_channel_data_type = CL_SNORM_INT16        ;
            image_formats[27].image_channel_order = CL_RG;                      image_formats[27].image_channel_data_type = CL_SIGNED_INT8        ;
            image_formats[28].image_channel_order = CL_RG;                      image_formats[28].image_channel_data_type = CL_SIGNED_INT16       ;
            image_formats[29].image_channel_order = CL_RG;                      image_formats[29].image_channel_data_type = CL_SIGNED_INT32       ;
            image_formats[30].image_channel_order = CL_RG;                      image_formats[30].image_channel_data_type = CL_UNSIGNED_INT8      ;
            image_formats[31].image_channel_order = CL_RG;                      image_formats[31].image_channel_data_type = CL_UNSIGNED_INT16     ;
            image_formats[32].image_channel_order = CL_RG;                      image_formats[32].image_channel_data_type = CL_UNSIGNED_INT32     ;
            image_formats[33].image_channel_order = CL_RA;                      image_formats[33].image_channel_data_type = CL_FLOAT              ;
            image_formats[34].image_channel_order = CL_RA;                      image_formats[34].image_channel_data_type = CL_HALF_FLOAT         ;
            image_formats[35].image_channel_order = CL_RA;                      image_formats[35].image_channel_data_type = CL_UNORM_INT8         ;
            image_formats[36].image_channel_order = CL_RA;                      image_formats[36].image_channel_data_type = CL_UNORM_INT16        ;
            image_formats[37].image_channel_order = CL_RA;                      image_formats[37].image_channel_data_type = CL_SNORM_INT16        ;
            image_formats[38].image_channel_order = CL_RA;                      image_formats[38].image_channel_data_type = CL_SIGNED_INT8        ;
            image_formats[39].image_channel_order = CL_RA;                      image_formats[39].image_channel_data_type = CL_SIGNED_INT16       ;
            image_formats[40].image_channel_order = CL_RA;                      image_formats[40].image_channel_data_type = CL_SIGNED_INT32       ;
            image_formats[41].image_channel_order = CL_RA;                      image_formats[41].image_channel_data_type = CL_UNSIGNED_INT8      ;
            image_formats[42].image_channel_order = CL_RA;                      image_formats[42].image_channel_data_type = CL_UNSIGNED_INT16     ;
            image_formats[43].image_channel_order = CL_RA;                      image_formats[43].image_channel_data_type = CL_UNSIGNED_INT32     ;
            image_formats[44].image_channel_order = CL_RGBA;                    image_formats[44].image_channel_data_type = CL_FLOAT              ;
            image_formats[45].image_channel_order = CL_RGBA;                    image_formats[45].image_channel_data_type = CL_HALF_FLOAT         ;
            image_formats[46].image_channel_order = CL_RGBA;                    image_formats[46].image_channel_data_type = CL_UNORM_INT8         ;
            image_formats[47].image_channel_order = CL_RGBA;                    image_formats[47].image_channel_data_type = CL_UNORM_INT16        ;
            image_formats[48].image_channel_order = CL_RGBA;                    image_formats[48].image_channel_data_type = CL_SNORM_INT16        ;
            image_formats[49].image_channel_order = CL_RGBA;                    image_formats[49].image_channel_data_type = CL_SIGNED_INT8        ;
            image_formats[50].image_channel_order = CL_RGBA;                    image_formats[50].image_channel_data_type = CL_SIGNED_INT16       ;
            image_formats[51].image_channel_order = CL_RGBA;                    image_formats[51].image_channel_data_type = CL_SIGNED_INT32       ;
            image_formats[52].image_channel_order = CL_RGBA;                    image_formats[52].image_channel_data_type = CL_UNSIGNED_INT8      ;
            image_formats[53].image_channel_order = CL_RGBA;                    image_formats[53].image_channel_data_type = CL_UNSIGNED_INT16     ;
            image_formats[54].image_channel_order = CL_RGBA;                    image_formats[54].image_channel_data_type = CL_UNSIGNED_INT32     ;
            image_formats[55].image_channel_order = CL_BGRA;                    image_formats[55].image_channel_data_type = CL_UNORM_INT8         ;
            image_formats[56].image_channel_order = CL_BGRA;                    image_formats[56].image_channel_data_type = CL_SIGNED_INT8        ;
            image_formats[57].image_channel_order = CL_BGRA;                    image_formats[57].image_channel_data_type = CL_UNSIGNED_INT8      ;
            image_formats[58].image_channel_order = CL_ARGB;                    image_formats[58].image_channel_data_type = CL_UNORM_INT8         ;
            image_formats[59].image_channel_order = CL_ARGB;                    image_formats[59].image_channel_data_type = CL_SIGNED_INT8        ;
            image_formats[60].image_channel_order = CL_ARGB;                    image_formats[60].image_channel_data_type = CL_UNSIGNED_INT8      ;
            image_formats[61].image_channel_order = CL_INTENSITY;               image_formats[61].image_channel_data_type = CL_FLOAT              ;
            image_formats[62].image_channel_order = CL_INTENSITY;               image_formats[62].image_channel_data_type = CL_HALF_FLOAT         ;
            image_formats[63].image_channel_order = CL_INTENSITY;               image_formats[63].image_channel_data_type = CL_UNORM_INT8         ;
            image_formats[64].image_channel_order = CL_INTENSITY;               image_formats[64].image_channel_data_type = CL_UNORM_INT16        ;
            image_formats[65].image_channel_order = CL_INTENSITY;               image_formats[65].image_channel_data_type = CL_SNORM_INT16        ;
            image_formats[66].image_channel_order = CL_LUMINANCE;               image_formats[66].image_channel_data_type = CL_FLOAT              ;
            image_formats[67].image_channel_order = CL_LUMINANCE;               image_formats[67].image_channel_data_type = CL_HALF_FLOAT         ;
            image_formats[68].image_channel_order = CL_LUMINANCE;               image_formats[68].image_channel_data_type = CL_UNORM_INT8         ;
            image_formats[69].image_channel_order = CL_LUMINANCE;               image_formats[69].image_channel_data_type = CL_UNORM_INT16        ;
            image_formats[70].image_channel_order = CL_LUMINANCE;               image_formats[70].image_channel_data_type = CL_SNORM_INT16        ;
         }
      } else return CL_INVALID_VALUE;  // [한국어] 2D 이미지가 아닌 경우
   } else {
      opencl_not_implemented(__my_func__,__LINE__);  // [한국어] READ_ONLY 외 플래그는 미구현
   }
   return CL_SUCCESS;
}

/*
 * [한국어]
 * clEnqueueMapBuffer - 버퍼를 호스트 메모리에 매핑하여 포인터 반환
 *
 * @command_queue:          커맨드 큐
 * @buffer:                 매핑할 cl_mem 핸들
 * @blocking_map:           동기 매핑 여부 (현재 무시)
 * @map_flags:              매핑 플래그 (현재 무시)
 * @offset:                 버퍼 내 오프셋 (현재 무시)
 * @cb:                     매핑할 바이트 수 (현재 무시)
 * @num_events_in_wait_list: 대기 이벤트 수 (현재 무시)
 * @event_wait_list:        대기 이벤트 목록 (현재 무시)
 * @event:                  완료 이벤트 출력 (현재 미사용)
 * @errcode_ret:            에러 코드 출력 포인터 (NULL 가능)
 * @return:                 호스트 메모리 포인터
 *
 * GPGPU-Sim에서는 CL_MEM_USE_HOST_PTR/ALLOC_HOST_PTR 버퍼만 매핑 가능하며,
 * 매핑 결과는 납부된 host_ptr() 그대로 반환한다. GPU 전용 버퍼 매핑은 assert 실패.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clEnqueueMapBuffer] → lookup_mem → host_ptr()
 */
extern CL_API_ENTRY void * CL_API_CALL
clEnqueueMapBuffer(cl_command_queue command_queue,
                   cl_mem           buffer,
                   cl_bool          blocking_map, 
                   cl_map_flags     map_flags,
                   size_t           offset,
                   size_t           cb,
                   cl_uint          num_events_in_wait_list,
                   const cl_event * event_wait_list,
                   cl_event *       event,
                   cl_int *         errcode_ret ) CL_API_SUFFIX__VERSION_1_0
{
   _cl_mem *mem = command_queue->get_context()->lookup_mem(buffer);  // [한국어] buffer로 _cl_mem* 조회
   assert( mem->is_on_host() );  // [한국어] 호스트 메모리 버퍼만 매핑 가능 (GPU 버퍼는 미지원)
   return mem->host_ptr();       // [한국어] 호스트 메모리 포인터 반환
}


/*
 * [한국어]
 * clSetCommandQueueProperty - OpenCL 1.0 deprecated 커맨드 큐 속성 설정
 *
 * @command_queue:    속성을 변경할 커맨드 큐
 * @properties:       설정할 속성 비트마스크
 * @enable:           true면 설정, false면 해제
 * @old_properties:   이전 속성 값을 저장할 포인터 (현재 무시)
 * @return:           CL_SUCCESS (항상)
 *
 * OpenCL 1.0에서 deprecated된 함수이며 GPGPU-Sim에서는 실제 동작 없이 CL_SUCCESS만 반환.
 * TODO 주석이 남아 있어 향후 속성 변경이 필요할 수 있음.
 *
 * 호출 체인:
 *   OpenCL 애플리케이션 → [clSetCommandQueueProperty]
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clSetCommandQueueProperty( cl_command_queue command_queue,
                              cl_command_queue_properties properties,
                              cl_bool enable,
                              cl_command_queue_properties *old_properties
                           ) CL_API_SUFFIX__VERSION_1_0
{
   // TODO: do something here  // [한국어] 실제 속성 변경 로직은 미구현
   return CL_SUCCESS;
}

