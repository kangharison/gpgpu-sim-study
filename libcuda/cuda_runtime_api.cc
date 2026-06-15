// This file created from cuda_runtime_api.h distributed with CUDA 1.1
// Changes Copyright 2009,  Tor M. Aamodt, Ali Bakhoda and George L. Yuan
// University of British Columbia

/*
 * cuda_runtime_api.cc
 *
 * Copyright © 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda,
 * George L. Yuan and the University of British Columbia, Vancouver,
 * BC V6T 1Z4, All Rights Reserved.
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
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use
 * only.
 *
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 *
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung,
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia,
 * Vancouver, BC V6T 1Z4
 */

/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
 * [한국어 설명] CUDA 런타임 API 인터셉트 레이어 (cuda_runtime_api.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 가장 핵심적인 인터셉트 레이어로, NVIDIA의 실제
 * libcudart.so 대신 링크되는 가짜 CUDA 런타임 라이브러리의 본체이다.
 * CUDA 애플리케이션이 cudaMalloc, cudaMemcpy, cudaLaunchKernel 등 모든
 * CUDA 런타임 API를 호출할 때, 이 파일에 구현된 stub 함수들이 대신
 * 실행되어 GPGPU-Sim 시뮬레이터로 요청을 전달한다.
 * 또한 cuobjdump 파서 콜백, PTX 정보 등록, 디바이스 속성 초기화 등
 * 시뮬레이터 부트스트래핑에 필요한 모든 초기화 코드를 포함한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 실행 흐름의 최상위 진입점이다:
 *   CUDA Application
 *     → libcuda.so (이 파일) ← 링크 시 NVIDIA libcudart.so 대신 치환
 *         → GPGPUSim_Context() → GPGPUSim_Init()  (시뮬레이터 싱글톤 초기화)
 *             → gpgpu_ptx_sim_init_perf()          (타이밍 모델 gpgpu_sim 생성)
 *             → cuda-sim/ (PTX 기능 시뮬레이션)
 *             → gpgpu-sim/ (사이클-레벨 타이밍 시뮬레이션)
 * 즉, 이 파일은 사용자 공간(user-space) 호스트 측에서 실행되며,
 * 실제 GPU 드라이버/하드웨어 없이 시뮬레이터가 CUDA 애플리케이션을
 * 투명하게 가로채는 첫 번째 관문이다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - src/gpgpusim_entrypoint.{cc,h}: GPGPUSim_ctx, start_sim_thread 등 시뮬레이터 생애주기
 *   - src/gpgpu-sim/gpu-sim.{cc,h}: 타이밍 모델 gpgpu_sim (사이클 루프 본체)
 *   - src/cuda-sim/cuda-sim.h: PTX 기능 시뮬레이션 인터페이스
 *   - src/cuda-sim/ptx_ir.h: function_info 등 PTX IR 자료구조
 *   - src/cuda-sim/ptx_loader.h: PTX 바이너리 파싱/로딩
 *   - src/cuda-sim/ptx_parser.h: ptxinfo_data, PTX 파서 콜백
 *   - src/stream_manager.{cc,h}: CUDA 스트림(CUstream_st) 관리
 *   - src/abstract_hardware_model.h: kernel_info_t, warp_inst_t 등 추상 HW 모델
 *   - libcuda/cuda_api_object.h: CUctx_st, _cuda_device_id 등 CUDA 객체 정의
 *   - libcuda/gpgpu_context.h: gpgpu_context 전역 컨텍스트 (시뮬레이터 상태 소유자)
 * 데이터 흐름:
 *   - CUDA 앱의 API 호출 → 이 파일의 stub → CUctx_st/gpgpu_sim으로 전달
 *   - cuobjdump 파서 → addCuobjdumpSection/setCuobjdump* 콜백 → cuobjdumpSectionList
 *   - PTX 파서 → ptxinfo_data::ptxinfo_addinfo() → CUctx_st::add_ptxinfo()
 *
 * === 주요 함수/구조체 요약 ===
 * - GPGPUSim_Init()        : 시뮬레이터 싱글톤 초기화; gpgpu_sim 생성, cudaDeviceProp 채움
 * - GPGPUSim_Context()     : CUctx_st 싱글톤 반환 (없으면 생성); 모든 API 함수의 출발점
 * - GPGPU_Context()        : gpgpu_context 전역 싱글톤 반환 (없으면 heap 할당)
 * - ptxinfo_addinfo()      : PTX 파서가 .section ptxinfo 지시자 파싱 후 호출하는 콜백
 * - cuda_not_implemented() : 미구현 CUDA API 호출 시 에러 메시지를 출력하고 abort
 * - addCuobjdumpSection()  : cuobjdump lex/yacc 파서 콜백; PTX/ELF 섹션 노드 추가
 * - get_app_binary()       : /proc/self/exe 심볼릭 링크를 통해 현재 실행 바이너리 경로 반환
 * - cudaArray              : 2D/3D 텍스처/배열을 위한 디바이스 메모리 래퍼 구조체
 */

#include <assert.h>   // [한국어] assert() 매크로 — 내부 불변 조건(invariant) 위반 시 즉시 abort
#include <stdarg.h>   // [한국어] va_list/va_start/va_end — 가변 인수 함수(printf-style) 구현에 필요
#include <stdio.h>    // [한국어] printf/fprintf/fflush/fopen/fgets — 콘솔 출력 및 파일 I/O
#include <stdlib.h>   // [한국어] malloc/calloc/free/abort/system/mkstemp — 동적 메모리 및 프로세스 제어
#include <string.h>   // [한국어] strcmp/strdup/strtok/memcpy — C 문자열 조작
#include <time.h>     // [한국어] time() — 타이머 이벤트 시뮬레이션용 클럭 기준값
#include <fstream>    // [한국어] std::ifstream/ofstream — PTX 파일 로딩 등 파일 스트림
#include <functional> // [한국어] std::function — 콜백 래퍼 (스트림 오퍼레이션 dispatch에 사용)
#include <iostream>   // [한국어] std::cout/cerr — C++ 스타일 콘솔 출력
#include <regex>      // [한국어] std::regex — CUDA 버전 문자열 파싱 및 섹션 헤더 매칭
#include <sstream>    // [한국어] std::stringstream — /proc/self/exe 경로 조합 등 문자열 빌더
#include <string>     // [한국어] std::string — C++ 문자열; API 이름, 파일 경로 저장에 광범위 사용
#ifdef OPENGL_SUPPORT
#define GL_GLEXT_PROTOTYPES  // [한국어] GL 확장 함수 프로토타입을 헤더에서 노출 (OpenGL interop 지원 시)
#ifdef __APPLE__
#include <GLUT/glut.h>  // Apple's version of GLUT is here
// [한국어] macOS: GLUT가 시스템 프레임워크 경로(/System/Library/Frameworks/GLUT.framework)에 위치
#else
#include <GL/gl.h>
// [한국어] Linux/Windows: 표준 OpenGL 헤더 (GL 컨텍스트와 CUDA 메모리 공유 지원)
#endif
#endif

#define __CUDA_RUNTIME_API_H__
// [한국어] CUDA 런타임 API 헤더의 중복 포함 방지 가드를 먼저 정의.
// 이후 include되는 NVIDIA 헤더들이 CUDA 런타임 함수 원형을 재선언하는 것을 막아
// GPGPU-Sim이 정의하는 stub 함수들과의 충돌을 방지한다.

// clang-format off
#include "host_defines.h"
// [한국어] __host__, __device__, __global__ 등 CUDA 한정자(qualifier) 매크로 정의.
// 시뮬레이터 환경에서는 이 한정자들이 빈 매크로로 확장되어 호스트 코드로 컴파일된다.
#include "builtin_types.h"
// [한국어] dim3, cudaError_t, size_t 등 CUDA 기본 타입 정의.
// CUDA 앱이 사용하는 모든 기본 자료형의 공통 기반이다.
#include "driver_types.h"
// [한국어] cudaMemcpyKind, cudaChannelFormatDesc, cudaDeviceProp 등
// CUDA 드라이버 수준의 enum/구조체 정의. 이 파일에서 cudaDeviceProp 채움에 직접 사용된다.
#include "cuda_api.h"
// [한국어] CUDA API 선언부 (libcuda 내부 전용). CUctx_st, _cuda_device_id 등
// GPGPU-Sim 전용 CUDA 객체 계층의 인터페이스를 포함한다.
#include "cudaProfiler.h"
// [한국어] cudaProfilerStart/Stop 등 CUDA 프로파일러 API 선언.
// GPGPU-Sim에서는 stub으로만 구현된다.
// clang-format on
#if (CUDART_VERSION < 8000)
#include "__cudaFatFormat.h"
// [한국어] CUDA 8.0 이전의 fat binary 포맷(__cudaFatCudaBinary 구조체) 정의.
// CUDA 8.0부터는 fat binary 포맷이 변경되었으므로 구버전 호환성을 위한 분기이다.
#endif
#include "../src/abstract_hardware_model.h"
// [한국어] warp_inst_t, kernel_info_t, core_t 등 GPU 추상 하드웨어 모델.
// CUDA 커널 실행 요청을 시뮬레이터 내부 자료구조로 변환할 때 사용된다.
#include "../src/cuda-sim/cuda-sim.h"
// [한국어] PTX 기능 시뮬레이션 인터페이스 (gpgpu_ptx_sim_init_perf 등).
// 타이밍 모델(gpgpu-sim/)과 기능 모델(cuda-sim/)을 연결하는 중간 인터페이스다.
#include "../src/cuda-sim/ptx_ir.h"
// [한국어] function_info, ptx_instruction 등 PTX IR(중간 표현) 자료구조.
// register_ptx_function에서 커널 함수를 등록할 때 사용된다.
#include "../src/cuda-sim/ptx_loader.h"
// [한국어] PTX 바이너리 파싱/로딩 인터페이스. fat binary에서 PTX를 추출하여
// 파서에게 넘기는 역할을 한다.
#include "../src/cuda-sim/ptx_parser.h"
// [한국어] ptxinfo_data 클래스 정의. PTX 어셈블러(ptxas)가 출력하는
// 레지스터/공유메모리 사용량 정보를 파싱한 결과를 담는다.
#include "../src/gpgpu-sim/gpu-sim.h"
// [한국어] gpgpu_sim 클래스 및 gpgpu_sim_config 정의.
// 사이클-레벨 타이밍 시뮬레이터의 최상위 클래스; GPGPUSim_Init에서 생성된다.
#include "../src/gpgpusim_entrypoint.h"
// [한국어] GPGPUsim_ctx, start_sim_thread 등 시뮬레이터 생애주기 함수.
// 시뮬레이션 스레드를 시작하고 시뮬레이터 전역 상태를 관리한다.
#include "../src/stream_manager.h"
// [한국어] CUstream_st, stream_manager 정의.
// cudaMemcpyAsync, cudaLaunchKernel 등 비동기 CUDA 오퍼레이션의 큐잉 메커니즘.
#include "cuda_api_object.h"
// [한국어] CUctx_st(_cuda_device_id *), CUevent_st 등 CUDA 객체 구현체.
// 이 파일의 GPGPUSim_Context()가 생성/반환하는 핵심 객체들이 여기 정의된다.
#include "gpgpu_context.h"
// [한국어] gpgpu_context 전역 컨텍스트 클래스 — 시뮬레이터 전체 상태의 소유자.
// GPGPU_Context() 함수가 반환하는 싱글톤 객체의 타입이다.

#include <pthread.h>   // [한국어] pthread_t, pthread_create 등 POSIX 스레드 API.
                       // 시뮬레이션 루프를 별도 스레드로 구동하기 위해 필요하다.
#include <semaphore.h> // [한국어] sem_t, sem_post/sem_wait — 호스트-시뮬레이터 스레드 간 동기화.
                       // CUDA 스트림 오퍼레이션 완료를 알리는 데 사용된다.

#ifdef __APPLE__
#include <mach-o/dyld.h>
// [한국어] macOS 전용: _NSGetExecutablePath() 함수 선언.
// Linux의 /proc/self/exe 대신 macOS에서 현재 실행 바이너리 경로를 얻는 API.
#endif

// SST cycle
/* [한국어] SST 통합 모드에서 GPU 시뮬레이션을 1사이클 진행 */
extern bool SST_Cycle();
// [한국어] SST(Structural Simulation Toolkit) 통합 모드에서 사이클을 진행시키는 외부 함수.
// GPGPU-Sim을 SST 프레임워크 내 컴포넌트로 연결할 때 이 함수를 통해 동기화한다.

/*DEVICE_BUILTIN*/
/*
 * [한국어] cudaArray — CUDA 2D/3D 배열 (텍스처/서피스 메모리 래퍼) 구조체.
 * cudaMallocArray(), cudaMemcpy2DToArray() 등의 API가 이 구조체를 통해
 * 디바이스 메모리 내 2D/3D 배열 레이아웃을 표현한다.
 * GPGPU-Sim에서는 실제 GPU DRAM 대신 시뮬레이터 내부 메모리 공간에 매핑된다.
 */
struct cudaArray {
  void *devPtr;
  /* [한국어] 디바이스 메모리 포인터 (64비트 주소).
   * 설정자: cudaMallocArray() 등 배열 할당 API가 시뮬레이터 메모리 할당 후 저장.
   * 읽는 자: cudaMemcpy2DToArray() 등 배열 접근 API가 실제 데이터 복사 대상 주소로 사용.
   * 값 범위: 시뮬레이터 가상 주소 공간 내 유효한 포인터 (NULL이면 미초기화).
   * 동기화: 배열 하나는 단일 CUDA 컨텍스트에서 생성·사용되며 별도 락 없음. */

  int devPtr32;
  /* [한국어] 32비트 환경 또는 레거시 CUDA 드라이버 호환성을 위한 32비트 디바이스 포인터.
   * 설정자: CUDA 1.x 시절 32비트 주소 공간에서 배열 할당 시 저장.
   * 읽는 자: 32비트 경로의 접근 API (현재 GPGPU-Sim에서는 거의 사용되지 않음).
   * 값 범위: 유효한 32비트 디바이스 주소 또는 0 (64비트 환경에서는 0).
   * 동기화: devPtr과 동일, 별도 락 없음. */

  struct cudaChannelFormatDesc desc;
  /* [한국어] 채널 포맷 기술자 (Channel Format Descriptor).
   * 각 채널(x,y,z,w)의 비트 수와 데이터 종류(int/uint/float)를 기술한다.
   * 설정자: cudaCreateChannelDesc() 또는 cudaMallocArray() 호출 시 인수로 전달된 값.
   * 읽는 자: 텍스처 바인딩(cudaBindTextureToArray) 및 배열 복사 시 포맷 검증에 사용.
   * 값 범위: x,y,z,w 각각 0~32 비트, kind는 cudaChannelFormatKindSigned/Unsigned/Float.
   * 동기화: 생성 시 1회 설정, 이후 read-only. */

  int width;
  /* [한국어] 배열의 너비 (x 방향 요소 수).
   * 설정자: cudaMallocArray() 호출 시 width 인수 값.
   * 읽는 자: 경계 검사, 메모리 크기 계산, 텍스처 좌표 정규화에 사용.
   * 값 범위: 1 이상의 양의 정수; SM 컴퓨트 능력에 따라 최대값 제한.
   * 동기화: 생성 후 불변. */

  int height;
  /* [한국어] 배열의 높이 (y 방향 요소 수). 1D 배열의 경우 0 또는 1.
   * 설정자: cudaMallocArray() 호출 시 height 인수 값.
   * 읽는 자: 2D/3D 텍스처 참조(texref) 설정 및 경계 검사에 사용.
   * 값 범위: 0 (1D 배열) 또는 1 이상 (2D/3D 배열).
   * 동기화: 생성 후 불변. */

  int size;  // in bytes
  /* [한국어] 배열 전체 크기 (바이트 단위).
   * 설정자: 배열 할당 함수가 width * height * desc 채널 크기로 계산하여 저장.
   * 읽는 자: cudaMemcpy2DToArray() 등 데이터 복사 시 전체 크기 검증에 사용.
   * 값 범위: 양의 정수 (0이면 미초기화 또는 할당 실패).
   * 동기화: 생성 후 불변. */

  unsigned dimensions;
  /* [한국어] 배열의 차원 수 (1, 2, 또는 3).
   * 설정자: cudaMalloc3DArray() 등 할당 API가 요청된 차원으로 설정.
   * 읽는 자: 텍스처 바인딩 API가 배열 타입(1D/2D/3D)을 구분하는 데 사용.
   * 값 범위: 1(1D), 2(2D), 3(3D).
   * 동기화: 생성 후 불변. */
};

#if !defined(__dv)
#if defined(__cplusplus)
#define __dv(v) = v
/* [한국어] C++ 모드: __dv(v)를 C++ 기본 인수(default argument) '= v'로 확장.
 * CUDA API 헤더에서 선택적 파라미터(예: cudaMemcpy의 kind 기본값)를 지원하기 위해 사용. */
#else /* __cplusplus */
#define __dv(v)
/* [한국어] C 모드: __dv(v)를 빈 문자열로 확장.
 * C언어는 기본 인수를 지원하지 않으므로 매크로를 무시하여 순수 C 호환성을 유지한다. */
#endif /* __cplusplus */
#endif /* !__dv */

cudaError_t g_last_cudaError = cudaSuccess;
/* [한국어] 가장 최근 CUDA API 호출의 에러 코드를 저장하는 전역 변수.
 * 설정자: 이 파일의 모든 CUDA API stub 함수들이 반환 직전에 'g_last_cudaError = <에러코드>'로 갱신.
 * 읽는 자: cudaGetLastError() / cudaPeekAtLastError()가 이 값을 반환.
 * 값 범위: cudaSuccess(0) ~ 정의된 cudaError_t enum 값 중 하나.
 * 동기화: CUDA 스트림/컨텍스트는 단일 호스트 스레드에서 사용하므로 별도 락 없이 안전하다.
 *          단, 멀티스레드 환경에서 동시 API 호출 시 경쟁 조건 가능 — CUDA 스펙상 동일 컨텍스트
 *          동시 호출은 정의되지 않은 동작이므로 현재 구현은 이를 보호하지 않는다. */

/*
 * [한국어]
 * register_ptx_function - PTX 함수 등록 (현재 미사용 stub)
 *
 * @name: 등록하려는 PTX 커널 함수의 이름 문자열.
 * @impl: 해당 커널의 function_info 포인터 (PTX IR 표현).
 * @return: 없음 (void).
 *
 * 예전 GPGPU-Sim 버전에서는 PTX 커널을 이름으로 등록하는 역할을 했으나,
 * 현재 버전에서는 PTX 로더(ptx_loader.cc)가 직접 커널 함수를 관리하므로
 * 이 함수는 아무 동작도 하지 않는 빈 stub으로 남아 있다.
 * 하위 호환성을 위해 함수 시그니처는 유지된다.
 *
 * 호출 체인:
 *   (이전) cudaRegisterFunction() 경로 → [register_ptx_function] → (미사용)
 */
void register_ptx_function(const char *name, function_info *impl) {
  // no longer need this
  // [한국어] 과거 PTX 함수 등록 코드; 현재 버전에서는 ptx_loader가 직접 처리하므로 비워둠
}

#if defined __APPLE__
#define __my_func__ __PRETTY_FUNCTION__
// [한국어] macOS/Clang 환경: __PRETTY_FUNCTION__은 클래스::메서드(인자타입) 형태의 상세 함수명 제공
#else
#if defined __cplusplus ? __GNUC_PREREQ(2, 6) : __GNUC_PREREQ(2, 4)
#define __my_func__ __PRETTY_FUNCTION__
// [한국어] GCC 2.6+(C++) 또는 GCC 2.4+(C): __PRETTY_FUNCTION__으로 시그니처 포함 함수명 제공
#else
#if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#define __my_func__ __func__
// [한국어] C99 이상: __func__는 현재 함수명 문자열 리터럴 (단순 이름, 시그니처 없음)
#else
#define __my_func__ ((__const char *)0)
// [한국어] 구형 컴파일러: 함수명 제공 불가 — NULL 포인터로 정의하여 printf에서 "(null)" 출력
#endif
#endif
#endif
// [한국어] __my_func__: announce_call()이 어떤 CUDA API가 호출되었는지 로그로 출력할 때
// 컴파일러/플랫폼에 관계없이 현재 함수명을 가져오기 위한 이식성 래퍼 매크로.

/*
 * [한국어]
 * gpgpu_context::GPGPUSim_Init - GPGPU-Sim 시뮬레이터 싱글톤 초기화
 *
 * @return: 초기화된 _cuda_device_id 포인터.
 *          cudaGetDeviceProperties() 등의 API가 장치 속성을 조회할 때 반환된다.
 *
 * 이 함수는 GPGPU-Sim 시뮬레이터의 핵심 초기화 루틴이다.
 * 최초 호출 시(the_cude_device == NULL) gpgpu_sim 타이밍 모델 객체를 생성하고,
 * gpgpusim.config에 설정된 GPU 아키텍처 파라미터를 읽어 cudaDeviceProp 구조체를
 * 채운 뒤, _cuda_device_id 싱글톤을 생성한다.
 * 이후 start_sim_thread(1)을 호출하여 사이클-레벨 시뮬레이션 루프를 실행하는
 * 별도의 POSIX 스레드를 시작한다.
 * 두 번째 이후 호출에서는 이미 초기화된 the_cude_device를 즉시 반환한다.
 *
 * 실행 컨텍스트: 호스트 스레드 (CUDA API를 처음 호출하는 애플리케이션 스레드).
 * 동시성: GPGPUSim_Context()가 한 번만 호출하도록 보호하므로 재진입 없음.
 *
 * 호출 체인:
 *   GPGPUSim_Context() → [GPGPUSim_Init] → gpgpu_ptx_sim_init_perf()
 *                                         → start_sim_thread(1)
 */
struct _cuda_device_id *gpgpu_context::GPGPUSim_Init() {
  _cuda_device_id *the_device = the_gpgpusim->the_cude_device; // [한국어] 기존 싱글톤 장치 포인터 로드 (두 번째 이후 호출 시 NULL이 아님)
  if (!the_device) { // [한국어] 아직 초기화되지 않은 경우에만 시뮬레이터를 생성 (최초 1회 진입)
    gpgpu_sim *the_gpu = gpgpu_ptx_sim_init_perf();
    // [한국어] gpgpusim.config를 파싱하여 타이밍 모델(gpgpu_sim) 객체 생성.
    // 이 함수 내부에서 SM 수, 캐시 크기, DRAM 파라미터 등이 모두 초기화된다.

    cudaDeviceProp *prop = (cudaDeviceProp *)calloc(sizeof(cudaDeviceProp), 1);
    // [한국어] cudaGetDeviceProperties()가 반환할 cudaDeviceProp 구조체를 heap에 할당.
    // calloc을 사용하여 모든 필드를 0으로 초기화한 뒤 명시적으로 채운다.

    snprintf(prop->name, 256, "GPGPU-Sim_v%s", g_gpgpusim_version_string);
    // [한국어] 장치 이름을 "GPGPU-Sim_v<버전>" 형태로 설정.
    // cudaGetDeviceProperties()로 이름을 조회하는 앱이 시뮬레이터임을 인식할 수 있게 한다.

    prop->major = the_gpu->compute_capability_major();
    // [한국어] CUDA 컴퓨트 능력 메이저 버전 (예: SM 7.0 → 7) — gpgpusim.config의 sm 설정에서 읽음
    prop->minor = the_gpu->compute_capability_minor();
    // [한국어] CUDA 컴퓨트 능력 마이너 버전 (예: SM 7.0 → 0)
    prop->totalGlobalMem = 0x80000000 /* 2 GB */;
    // [한국어] 전체 글로벌 메모리 크기를 2GB로 고정. 실제 GPU처럼 다양한 용량을 지원하지 않고
    // 시뮬레이터는 고정 값을 사용한다. cudaMalloc의 범위 검사 기준이 된다.
    prop->memPitch = 0;
    // [한국어] 2D 메모리 pitch (행 간격) — 시뮬레이터에서는 pitch 정렬 최적화를 하지 않으므로 0.

    if (prop->major >= 2) {
      // [한국어] Fermi(SM 2.x) 이상: 스레드 블록당 최대 1024개 스레드 허용
      prop->maxThreadsPerBlock = 1024; // [한국어] 블록당 최대 스레드 수 (1024 = 32 warp × 32)
      prop->maxThreadsDim[0] = 1024;  // [한국어] x 방향 최대 스레드 수
      prop->maxThreadsDim[1] = 1024;  // [한국어] y 방향 최대 스레드 수
    } else {
      // [한국어] Tesla(SM 1.x): 블록당 최대 512개 스레드 (16 warp × 32)
      prop->maxThreadsPerBlock = 512; // [한국어] Tesla SM의 블록당 스레드 한계
      prop->maxThreadsDim[0] = 512;  // [한국어] Tesla x 방향 최대 스레드 수
      prop->maxThreadsDim[1] = 512;  // [한국어] Tesla y 방향 최대 스레드 수
    }

    prop->maxThreadsDim[2] = 64;          // [한국어] z 방향 최대 스레드 수 (SM 버전 무관 64로 제한)
    prop->maxGridSize[0] = 0x40000000;    // [한국어] 그리드 x 방향 최대 블록 수 (1G = 2^30)
    prop->maxGridSize[1] = 0x40000000;    // [한국어] 그리드 y 방향 최대 블록 수
    prop->maxGridSize[2] = 0x40000000;    // [한국어] 그리드 z 방향 최대 블록 수
    prop->totalConstMem = 0x40000000;     // [한국어] 상수 메모리 총 크기 1GB (실제 GPU는 64KB; 시뮬레이터 단순화)
    prop->textureAlignment = 0;           // [한국어] 텍스처 주소 정렬 바이트 수 — 시뮬레이터는 정렬 제약 없으므로 0
    //        * TODO: Update the .config and xml files of all GPU config files
    //        with new value of sharedMemPerBlock and regsPerBlock
    prop->sharedMemPerBlock = the_gpu->shared_mem_per_block();
    // [한국어] 블록당 공유 메모리 크기 (bytes) — gpgpusim.config의 gpgpu_shmem_size 값에서 읽음
#if (CUDART_VERSION > 5050)
    prop->regsPerMultiprocessor = the_gpu->num_registers_per_core();
    // [한국어] CUDA 5.5 이후: SM 당 레지스터 수 (예: Kepler = 65536, Maxwell = 65536)
    prop->sharedMemPerMultiprocessor = the_gpu->shared_mem_size();
    // [한국어] CUDA 5.5 이후: SM 당 전체 공유 메모리 크기 (bytes)
#endif
    prop->sharedMemPerBlock = the_gpu->shared_mem_per_block();
    // [한국어] 블록당 공유 메모리 재설정 (위 CUDART_VERSION 분기 후 무조건 적용)
    prop->regsPerBlock = the_gpu->num_registers_per_block();
    // [한국어] 블록당 레지스터 수 — 컴파일러가 레지스터 압력(register spilling) 판단에 사용
    prop->warpSize = the_gpu->wrp_size();
    // [한국어] warp 크기 (NVIDIA GPU 표준 = 32) — PTX 스케줄러와 SIMT 스택 크기의 기준
    prop->clockRate = the_gpu->shader_clock();
    // [한국어] SM 클럭 속도 (kHz 단위) — gpgpusim.config의 gpgpu_clock_gated_el_cta 등에서 파생
#if (CUDART_VERSION >= 2010)
    prop->multiProcessorCount = the_gpu->get_config().num_shader();
    // [한국어] CUDA 2.1 이후: GPU 내 SM(Streaming Multiprocessor) 총 수
    // num_shader()는 gpgpusim.config의 gpgpu_n_clusters * gpgpu_n_cores_per_cluster에서 계산
#endif
#if (CUDART_VERSION >= 4000)
    prop->maxThreadsPerMultiProcessor = the_gpu->threads_per_core();
    // [한국어] CUDA 4.0 이후: SM당 동시 활성 스레드 최대 수 (Fermi=1536, Kepler=2048 등)
#endif
    the_gpu->set_prop(prop);
    // [한국어] 방금 채운 cudaDeviceProp을 gpgpu_sim 객체 내부에 저장.
    // 이후 cudaGetDeviceProperties() 호출 시 이 포인터를 반환한다.

    the_gpgpusim->the_cude_device = new _cuda_device_id(the_gpu);
    // [한국어] gpgpu_sim 포인터를 래핑하는 _cuda_device_id 싱글톤 생성.
    // 이 객체가 CUctx_st(CUDA 컨텍스트)에 바인딩되는 "장치 핸들"이다.

    the_device = the_gpgpusim->the_cude_device;
    // [한국어] 로컬 변수에도 새 장치 포인터를 저장하여 함수 끝에서 반환할 준비
  }
  start_sim_thread(1);
  // [한국어] 사이클-레벨 시뮬레이션 루프를 실행하는 POSIX 스레드를 시작한다.
  // 인수 1은 스레드를 바로 활성화(detach하지 않고 join 가능)하는 플래그.
  // 이 스레드가 gpgpu_sim::cycle()을 반복 호출하며 GPU 타이밍을 시뮬레이션한다.

  return the_device; // [한국어] 초기화된 (또는 기존의) 장치 싱글톤 반환
}

/*
 * [한국어]
 * GPGPUSim_Context - CUDA 컨텍스트(CUctx_st) 싱글톤 반환
 *
 * @ctx: GPGPU-Sim 전역 시뮬레이터 컨텍스트 포인터 (GPGPU_Context()가 반환한 값).
 * @return: 초기화된 CUctx_st 포인터 (CUDA 컨텍스트 객체, NULL이 반환되는 경우 없음).
 *
 * 이 함수는 GPGPU-Sim에서 CUDA 컨텍스트를 나타내는 CUctx_st 싱글톤을 관리한다.
 * 모든 CUDA API stub 함수(cudaMalloc, cudaLaunchKernel 등)는 이 함수를 통해
 * 현재 활성 컨텍스트를 얻은 뒤 시뮬레이터에 요청을 전달한다.
 * 최초 호출 시 GPGPUSim_Init()을 호출하여 시뮬레이터 전체를 초기화하고,
 * 이후 호출에서는 already-created 컨텍스트를 즉시 반환한다.
 * 실행 컨텍스트: 호스트 스레드. 동시성: ctx->the_context가 단일 스레드에서 설정됨.
 *
 * 호출 체인:
 *   (모든 CUDA API stub) → [GPGPUSim_Context] → GPGPUSim_Init()
 *                                              → new CUctx_st(the_gpu)
 */
CUctx_st *GPGPUSim_Context(gpgpu_context *ctx) {
  // static CUctx_st *the_context = NULL;
  CUctx_st *the_context = ctx->the_gpgpusim->the_context;
  // [한국어] gpgpu_context 내부에 저장된 기존 CUDA 컨텍스트 포인터 읽기.
  // 이전에 이미 GPGPUSim_Context()가 호출됐다면 NULL이 아닌 값이 반환된다.

  if (the_context == NULL) {
    // [한국어] 최초 호출: 아직 CUDA 컨텍스트가 없으므로 시뮬레이터 전체를 초기화
    _cuda_device_id *the_gpu = ctx->GPGPUSim_Init();
    // [한국어] GPGPUSim_Init()으로 gpgpu_sim 객체와 cudaDeviceProp을 초기화하고
    // _cuda_device_id 싱글톤을 얻는다. 이 안에서 시뮬레이션 스레드도 시작된다.

    ctx->the_gpgpusim->the_context = new CUctx_st(the_gpu);
    // [한국어] _cuda_device_id를 감싸는 CUctx_st(CUDA 컨텍스트 객체) 생성.
    // CUctx_st는 커널 등록 테이블, ptxinfo 맵, 스트림 목록을 소유한다.

    the_context = ctx->the_gpgpusim->the_context;
    // [한국어] 로컬 변수도 갱신하여 함수 끝의 반환에서 사용
  }
  return the_context; // [한국어] 초기화된(또는 기존의) CUDA 컨텍스트 싱글톤 반환
}

/*
 * [한국어]
 * GPGPU_Context - gpgpu_context 전역 싱글톤 반환
 *
 * @return: 전역 gpgpu_context 포인터. 항상 동일한 객체가 반환된다.
 *
 * gpgpu_context는 GPGPU-Sim 시뮬레이터의 모든 전역 상태를 소유하는 최상위 컨테이너다.
 * 이 함수는 프로세스 내에서 단 하나의 gpgpu_context 인스턴스만 존재하도록 보장하는
 * 고전적인 lazy-initialization 싱글톤 패턴을 구현한다.
 * libcuda.so가 로드될 때 자동으로 생성되지 않고, 첫 번째 CUDA API 호출 시
 * 이 함수가 호출되어 비로소 시뮬레이터가 초기화되기 시작한다.
 * 실행 컨텍스트: 호스트 스레드. 동시성: 단일 스레드 가정 (CUDA 컨텍스트 생성은 thread-safe 아님).
 *
 * 호출 체인:
 *   (모든 CUDA API stub) → [GPGPU_Context] → GPGPUSim_Context() → GPGPUSim_Init()
 */
gpgpu_context *GPGPU_Context() {
  static gpgpu_context *gpgpu_ctx = NULL;
  // [한국어] 함수-스코프 static: 최초 호출 시 한 번만 초기화되는 싱글톤 포인터.
  // C++11부터 function-local static 초기화는 thread-safe하지만,
  // 여기서는 명시적 NULL 체크 방식을 사용한다.

  if (gpgpu_ctx == NULL) {
    // [한국어] 최초 호출: gpgpu_context 객체를 힙에 할당 (이후 프로세스 종료까지 유지)
    gpgpu_ctx = new gpgpu_context();
    // [한국어] gpgpu_context 생성자에서 api, the_gpgpusim 등 하위 객체가 초기화된다.
    // 이 시점에서는 아직 gpgpu_sim이나 CUDA 컨텍스트는 생성되지 않는다.
  }
  return gpgpu_ctx; // [한국어] 전역 gpgpu_context 싱글톤 반환
}

/*
 * [한국어]
 * ptxinfo_data::ptxinfo_addinfo - PTX 어셈블러 정보를 CUDA 컨텍스트에 등록
 *
 * @return: 없음 (void).
 *
 * 이 함수는 CUDA fat binary를 처리하는 과정에서 ptxinfo 섹션을 파싱할 때
 * PTX 파서(ptx_parser.y)가 호출하는 콜백이다.
 * ptxinfo 섹션에는 ptxas(PTX 어셈블러)가 컴파일한 각 커널의 레지스터 수,
 * 공유 메모리 사용량, 상수 메모리 사용량 등이 기록되어 있다.
 * 이 정보는 GPGPUSim_Context()를 통해 현재 CUDA 컨텍스트(CUctx_st)에 등록되어
 * 이후 커널 실행 시 스케줄러와 타이밍 모델이 사용한다.
 * CUDA 5.0 이후에는 커널별 정보 외에 바이너리 전체에 적용되는 전역 정보
 * (gmem, cmem 등)도 ptxinfo 섹션에 포함되므로 커널 이름 없는 경우를 별도 처리한다.
 * 실행 컨텍스트: 호스트 스레드 (fat binary 등록 시 동기적으로 실행).
 *
 * 호출 체인:
 *   cuobjdump 파서 / PTX 파서 → [ptxinfo_addinfo] → CUctx_st::add_ptxinfo()
 */
void ptxinfo_data::ptxinfo_addinfo() {
  CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
  // [한국어] 현재 활성 CUDA 컨텍스트를 가져온다. 없으면 이 시점에 초기화됨.
  // gpgpu_ctx는 ptxinfo_data가 참조하는 전역 gpgpu_context 포인터다.

  if (!get_ptxinfo_kname()) {
    /* This info is not per kernel (since CUDA 5.0 some info (e.g. gmem, and
     * cmem) is added at the beginning for the whole binary ) */
    // [한국어] 커널 이름이 NULL인 경우: CUDA 5.0 이후의 바이너리 전체 ptxinfo (gmem/cmem).
    // 특정 커널에 귀속되지 않는 전역 메모리 사용량 정보를 컨텍스트에 등록한다.
    print_ptxinfo();                    // [한국어] ptxinfo 내용을 stdout에 디버그 출력
    context->add_ptxinfo(get_ptxinfo()); // [한국어] 커널 이름 없이 전역 ptxinfo를 컨텍스트에 추가
    clear_ptxinfo();                    // [한국어] 현재 ptxinfo_data 내부 상태 초기화 (다음 파싱 준비)
    return;                             // [한국어] 전역 정보 처리 완료, 이하 커널별 처리 불필요
  }
  if (!strcmp("__cuda_dummy_entry__", get_ptxinfo_kname())) {
    // this string produced by ptxas for empty ptx files (e.g., bandwidth test)
    // [한국어] ptxas가 빈 PTX 파일(커널이 없는 경우, 예: bandwidth test)에
    // 더미 엔트리로 생성하는 특수 이름. 실제 커널이 아니므로 정보를 버린다.
    clear_ptxinfo(); // [한국어] 더미 엔트리 정보 폐기 후 상태 초기화
    return;          // [한국어] 더미 엔트리 처리 완료, 등록 불필요
  }
  print_ptxinfo();
  // [한국어] 커널명과 함께 ptxinfo 내용을 stdout에 출력 (디버그/로그용)

  context->add_ptxinfo(get_ptxinfo_kname(), get_ptxinfo());
  // [한국어] 커널 이름(kname)을 키로 하여 ptxinfo를 CUDA 컨텍스트의 맵에 등록.
  // 이후 cudaLaunchKernel 시 해당 커널의 레지스터/메모리 사용량을 조회할 수 있게 된다.

  clear_ptxinfo();
  // [한국어] ptxinfo_data 내부 상태 초기화 — 다음 파싱 사이클을 위해 비워둠
}

/*
 * [한국어]
 * cuda_not_implemented - 미구현 CUDA API 호출 시 에러 출력 후 프로세스 종료
 *
 * @func: 미구현 CUDA API 함수의 이름 문자열 (예: "cudaBindSurfaceToArray").
 * @line: 이 함수를 호출한 stub 코드의 소스 라인 번호 (디버그 위치 추적용).
 * @return: 없음 (abort()로 프로세스가 종료되므로 실제로 반환되지 않음).
 *
 * GPGPU-Sim은 모든 CUDA API를 구현하지 않는다. 구현되지 않은 API의 stub 함수가
 * 호출될 때 이 함수를 통해 명확한 에러 메시지를 출력하고 시뮬레이션을 종료한다.
 * stdout/stderr를 명시적으로 flush하는 이유는 버퍼링된 출력이 abort() 이전에
 * 반드시 화면에 표시되도록 보장하기 위해서다.
 * 실행 컨텍스트: 호스트 스레드. 동시성: abort() 이후 모든 스레드 종료.
 *
 * 호출 체인:
 *   (미구현 CUDA API stub) → [cuda_not_implemented] → abort()
 */
void cuda_not_implemented(const char *func, unsigned line) {
  fflush(stdout); // [한국어] stdout 버퍼 강제 플러시 — abort() 전에 이전 출력이 모두 보이도록
  fflush(stderr); // [한국어] stderr 버퍼 강제 플러시 — 에러 메시지가 유실되지 않도록
  printf(
      "\n\nGPGPU-Sim PTX: Execution error: CUDA API function \"%s()\" has not "
      "been implemented yet.\n"
      "                 [$GPGPUSIM_ROOT/libcuda/%s around line %u]\n\n\n",
      func, __FILE__, line);
  // [한국어] 미구현 API 이름과 소스 파일/라인을 포함한 에러 메시지 출력.
  // __FILE__은 "cuda_runtime_api.cc"로 확장되어 GPGPU-Sim 루트 경로를 안내한다.
  fflush(stdout); // [한국어] printf 출력이 abort() 전에 화면에 표시되도록 재차 플러시
  abort();        // [한국어] SIGABRT 신호 발생으로 프로세스 즉시 종료 (코어 덤프 생성 가능)
}

/*
 * [한국어]
 * announce_call - CUDA API 호출 디버그 로그 출력
 *
 * @func: 호출된 CUDA API 함수의 이름 문자열.
 * @return: 없음 (void).
 *
 * g_debug_execution >= 3 조건에서 각 CUDA API stub이 호출될 때 이 함수를 통해
 * 함수명을 stdout에 출력한다. GPU 디버거가 없는 시뮬레이터 환경에서 API 호출
 * 시퀀스를 추적하기 위한 경량 트레이싱 유틸리티다.
 * 실행 컨텍스트: 호스트 스레드. 동시성: printf는 FILE 락을 사용하므로 출력 혼용 없음.
 *
 * 호출 체인:
 *   (CUDA API stub, g_debug_execution >= 3인 경우) → [announce_call]
 */
void announce_call(const char *func) {
  printf("\n\nGPGPU-Sim PTX: CUDA API function \"%s\" has been called.\n",
         func);
  // [한국어] 호출된 CUDA API 함수명을 표준 형식으로 출력 (디버그 트레이싱용)
  fflush(stdout); // [한국어] 출력을 즉시 화면에 반영 — 버퍼링으로 인한 지연 출력 방지
}

#define gpgpusim_ptx_error(msg, ...) \
  gpgpusim_ptx_error_impl(__func__, __FILE__, __LINE__, msg, ##__VA_ARGS__)
// [한국어] gpgpusim_ptx_error 매크로: 호출 지점의 함수명(__func__), 파일(__FILE__),
// 라인(__LINE__)을 자동으로 캡처하여 gpgpusim_ptx_error_impl에 전달한다.
// 이를 통해 에러 메시지에 정확한 소스 위치가 포함된다.

#define gpgpusim_ptx_assert(cond, msg, ...)                           \
  gpgpusim_ptx_assert_impl((cond), __func__, __FILE__, __LINE__, msg, \
                           ##__VA_ARGS__)
// [한국어] gpgpusim_ptx_assert 매크로: 조건(cond)이 false일 때 에러를 발생시키는
// assert 래퍼. cond, 함수명, 파일, 라인을 gpgpusim_ptx_assert_impl에 전달한다.
// 표준 assert()와 달리 NDEBUG 여부에 무관하게 항상 검사가 활성화된다.

/*
 * [한국어]
 * gpgpusim_ptx_error_impl - PTX/CUDA API 에러 메시지 출력 후 abort
 *
 * @func: 에러가 발생한 함수 이름 (gpgpusim_ptx_error 매크로에서 __func__로 전달).
 * @file: 에러가 발생한 소스 파일 이름 (__FILE__).
 * @line: 에러가 발생한 소스 라인 번호 (__LINE__).
 * @msg:  printf 스타일 포맷 문자열 (가변 인수 지원).
 * @...:  포맷 문자열에 대응하는 가변 인수.
 * @return: 없음 (abort()로 프로세스 종료).
 *
 * PTX 시뮬레이션 또는 CUDA API 처리 중 복구 불가능한 에러가 발생할 때 사용된다.
 * printf-style 포맷으로 상세한 에러 메시지를 출력하고 프로세스를 종료한다.
 * va_list를 직접 사용하므로 gpgpusim_ptx_error 매크로를 통해 간접 호출된다.
 * 실행 컨텍스트: 호스트 스레드. 동시성: abort()로 즉시 종료.
 *
 * 호출 체인:
 *   gpgpusim_ptx_error 매크로 → [gpgpusim_ptx_error_impl] → abort()
 *   gpgpusim_ptx_assert_impl (조건 실패 시) → [gpgpusim_ptx_error_impl] → abort()
 */
void gpgpusim_ptx_error_impl(const char *func, const char *file, unsigned line,
                             const char *msg, ...) {
  va_list ap;                       // [한국어] 가변 인수 목록 핸들 선언
  char buf[1024];                   // [한국어] 포맷팅된 에러 메시지를 담을 스택 버퍼 (최대 1023자)
  va_start(ap, msg);                // [한국어] msg 이후의 가변 인수 순회 시작
  vsnprintf(buf, 1024, msg, ap);    // [한국어] 포맷 문자열과 가변 인수를 buf에 안전하게 스프린트 (버퍼 오버플로 방지)
  va_end(ap);                       // [한국어] 가변 인수 목록 순회 종료 (스택 정리)

  printf("GPGPU-Sim CUDA API: %s\n", buf);
  // [한국어] 포맷팅된 에러 메시지 출력
  printf("                    [%s:%u : %s]\n", file, line, func);
  // [한국어] 에러 발생 위치(파일:라인 : 함수명)를 두 번째 줄에 정렬하여 출력
  abort(); // [한국어] SIGABRT로 프로세스 즉시 종료 (코어 덤프 허용)
}

/*
 * [한국어]
 * gpgpusim_ptx_assert_impl - 조건부 에러 검사 (assert 구현체)
 *
 * @test_value: 검사할 조건값. 0이면 에러(조건 실패), 0이 아니면 정상 통과.
 * @func: 에러가 발생한 함수 이름.
 * @file: 에러가 발생한 소스 파일.
 * @line: 에러가 발생한 소스 라인.
 * @msg:  에러 시 출력할 printf 스타일 포맷 문자열.
 * @...:  포맷 문자열 가변 인수.
 * @return: 없음 (test_value가 0이면 abort(), 아니면 정상 반환).
 *
 * gpgpusim_ptx_assert 매크로의 실제 구현체다.
 * test_value가 0(조건 실패)이면 gpgpusim_ptx_error_impl을 호출하여 abort한다.
 * va_list를 파싱하지만 실제로는 msg 자체를 gpgpusim_ptx_error_impl에 넘기므로
 * 가변 인수는 에러 구현체에서 재처리된다.
 * 실행 컨텍스트: 호스트 스레드. 동시성: 조건 실패 시 abort().
 *
 * 호출 체인:
 *   gpgpusim_ptx_assert 매크로 → [gpgpusim_ptx_assert_impl]
 *                                  → gpgpusim_ptx_error_impl() (조건 실패 시)
 */
void gpgpusim_ptx_assert_impl(int test_value, const char *func,
                              const char *file, unsigned line, const char *msg,
                              ...) {
  va_list ap;                    // [한국어] 가변 인수 목록 핸들 선언
  char buf[1024];                // [한국어] 포맷팅 버퍼 (현재 assert에서는 직접 사용하지 않음)
  va_start(ap, msg);             // [한국어] 가변 인수 순회 시작
  vsnprintf(buf, 1024, msg, ap); // [한국어] 포맷팅 (결과 buf는 현재 직접 출력에 사용되지 않음)
  va_end(ap);                    // [한국어] 가변 인수 순회 종료

  if (test_value == 0) gpgpusim_ptx_error_impl(func, file, line, msg);
  // [한국어] test_value가 0(조건 실패)이면 에러 출력 후 abort.
  // msg를 포맷 문자열로 직접 넘기므로 가변 인수의 실제 처리는 error_impl에서 수행된다.
  // 조건이 참(0이 아님)이면 이 줄은 실행되지 않고 정상 반환된다.
}

typedef std::map<unsigned, CUevent_st *> event_tracker_t;
// [한국어] event_tracker_t: CUDA 이벤트 UID(unsigned) → CUevent_st 포인터 맵 타입.
// cudaEventCreate/Record/Destroy 구현에서 이벤트 핸들을 정수 UID로 관리하기 위해 사용된다.

int CUevent_st::m_next_event_uid;
// [한국어] CUevent_st 클래스의 static 멤버 정의 (선언은 cuda_api_object.h).
// 새 CUDA 이벤트가 생성될 때마다 단조 증가하는 전역 UID 카운터.
// 동기화: 이벤트 생성은 단일 호스트 스레드에서 수행 가정 (별도 락 없음).

event_tracker_t g_timer_events;
// [한국어] 전역 이벤트 추적 맵: UID → CUevent_st* 연관 저장.
// cudaEventRecord, cudaEventSynchronize, cudaEventElapsedTime 등이 이 맵으로
// 이벤트 핸들을 조회한다.

extern int cuobjdump_lex_init(yyscan_t *scanner);
// [한국어] cuobjdump lex 스캐너 초기화 함수 (cuobjdump.l에서 자동 생성).
// 재진입 가능(reentrant) 스캐너를 초기화하여 scanner 포인터에 저장한다.

extern void cuobjdump_set_in(FILE *_in_str, yyscan_t yyscanner);
// [한국어] cuobjdump 스캐너의 입력 스트림을 설정한다.
// cuobjdump의 출력 파일을 파이프로 열어 이 함수에 전달하면 파서가 읽는다.

extern int cuobjdump_parse(yyscan_t scanner, struct cuobjdump_parser *parser,
                           std::list<cuobjdumpSection *> &cuobjdumpSectionList);
// [한국어] cuobjdump lex/yacc 파서의 파싱 엔트리포인트 (cuobjdump.y에서 자동 생성).
// 파싱 결과로 cuobjdumpSectionList에 PTX/ELF 섹션 노드들이 추가된다.

extern int cuobjdump_lex_destroy(yyscan_t scanner);
// [한국어] cuobjdump lex 스캐너 자원 해제 (파싱 완료 후 반드시 호출해야 메모리 누수 없음).

enum cuobjdumpSectionType { PTXSECTION = 0, ELFSECTION };
// [한국어] cuobjdump 섹션 타입 열거형.
// PTXSECTION(0): PTX(가상 ISA) 코드를 담는 섹션.
// ELFSECTION(1): ELF 형식의 SASS(실제 GPU ISA) 코드를 담는 섹션.
// addCuobjdumpSection()의 sectiontype 파라미터가 이 값을 참조한다.

// sectiontype: 0 for ptx, 1 for elf
/*
 * [한국어]
 * addCuobjdumpSection - cuobjdump 파서 콜백: 새 PTX/ELF 섹션 노드 추가
 *
 * @sectiontype: 섹션 타입. 0이면 PTX(cuobjdumpPTXSection), 1이면 ELF(cuobjdumpELFSection).
 * @cuobjdumpSectionList: 섹션 노드들을 축적하는 리스트. 파싱이 진행되면서 채워진다.
 * @return: 없음 (void).
 *
 * cuobjdump lex/yacc 파서가 fat binary 내에서 새 섹션 헤더를 인식할 때 호출된다.
 * 섹션 타입에 따라 cuobjdumpPTXSection 또는 cuobjdumpELFSection 객체를 생성하고
 * 리스트의 앞(front)에 삽입한다. 이후 setCuobjdump* 콜백들이 이 front() 노드의
 * 속성(arch, identifier, filename)을 채운다.
 * 실행 컨텍스트: fat binary 등록 시 호스트 스레드에서 동기적으로 실행.
 *
 * 호출 체인:
 *   cuobjdump_parse() → [addCuobjdumpSection] → cuobjdumpSectionList.push_front()
 */
void addCuobjdumpSection(int sectiontype,
                         std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  if (sectiontype)                                                       // [한국어] ELF 섹션인 경우 (sectiontype == 1)
    cuobjdumpSectionList.push_front(new cuobjdumpELFSection());          // [한국어] ELF 섹션 노드를 리스트 앞에 삽입 (이후 콜백이 front()로 접근)
  else
    cuobjdumpSectionList.push_front(new cuobjdumpPTXSection());          // [한국어] PTX 섹션 노드를 리스트 앞에 삽입
  printf("## Adding new section %s\n", sectiontype ? "ELF" : "PTX");    // [한국어] 섹션 추가 로그 출력 (타입 이름 표시)
}

/*
 * [한국어]
 * setCuobjdumparch - cuobjdump 파서 콜백: 현재 섹션의 GPU 아키텍처 설정
 *
 * @arch: SM 아키텍처 문자열 (예: "sm_70", "sm_86"). "sm_%u" 형식.
 * @cuobjdumpSectionList: 섹션 리스트. front()가 현재 파싱 중인 섹션이다.
 * @return: 없음 (void).
 *
 * cuobjdump 출력에서 각 섹션의 GPU 아키텍처 지시자(예: .arch sm_70)를 파싱할 때
 * 호출된다. "sm_%u" 형식의 문자열에서 숫자 부분을 추출하여 섹션 객체에 저장한다.
 * 이 아키텍처 번호는 GPGPU-Sim이 시뮬레이션할 GPU 아키텍처와 일치하는 섹션을
 * 선택하는 데 사용된다.
 * 실행 컨텍스트: 호스트 스레드. 동시성: 단일 파싱 스레드.
 *
 * 호출 체인:
 *   cuobjdump_parse() → [setCuobjdumparch] → cuobjdumpSection::setArch()
 */
void setCuobjdumparch(const char *arch,
                      std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  unsigned archnum;                                  // [한국어] "sm_%u"에서 추출할 숫자 부분 (예: "sm_70" → 70)
  sscanf(arch, "sm_%u", &archnum);                   // [한국어] arch 문자열에서 SM 번호 파싱 (예: "sm_70" → archnum=70)
  assert(archnum && "cannot have sm_0");             // [한국어] archnum=0은 유효하지 않은 아키텍처 — 파싱 실패 감지
  printf("Adding arch: %s\n", arch);                 // [한국어] 아키텍처 설정 로그 출력
  cuobjdumpSectionList.front()->setArch(archnum);    // [한국어] 현재 섹션(front)에 SM 아키텍처 번호 저장
}

/*
 * [한국어]
 * setCuobjdumpidentifier - cuobjdump 파서 콜백: 현재 섹션의 식별자(커널 이름) 설정
 *
 * @identifier: 섹션 식별자 문자열 (일반적으로 커널 함수 이름 또는 모듈 이름).
 * @cuobjdumpSectionList: 섹션 리스트. front()가 현재 파싱 중인 섹션이다.
 * @return: 없음 (void).
 *
 * cuobjdump 출력에서 섹션 식별자(커널 심볼 이름 또는 모듈 식별 문자열)를
 * 파싱할 때 호출된다. 이 식별자로 어떤 커널이 이 섹션에 속하는지 구분한다.
 * 실행 컨텍스트: 호스트 스레드. 동시성: 단일 파싱 스레드.
 *
 * 호출 체인:
 *   cuobjdump_parse() → [setCuobjdumpidentifier] → cuobjdumpSection::setIdentifier()
 */
void setCuobjdumpidentifier(
    const char *identifier,
    std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  printf("Adding identifier: %s\n", identifier);                     // [한국어] 식별자 설정 로그 출력
  cuobjdumpSectionList.front()->setIdentifier(identifier);           // [한국어] 현재 섹션(front)에 식별자 문자열 저장
}

/*
 * [한국어]
 * setCuobjdumpptxfilename - cuobjdump 파서 콜백: PTX 섹션의 파일명 설정
 *
 * @filename: cuobjdump가 추출한 PTX 파일의 임시 경로 문자열.
 * @cuobjdumpSectionList: 섹션 리스트. front()가 현재 파싱 중인 PTX 섹션이어야 한다.
 * @return: 없음 (void). front()가 PTX 섹션이 아니면 assert로 abort.
 *
 * cuobjdump가 fat binary에서 PTX 코드를 임시 파일로 추출했을 때 그 파일 경로를
 * 현재 섹션에 등록하는 파서 콜백이다. 반드시 PTX 섹션에만 호출되어야 하므로
 * dynamic_cast로 타입을 검증한다.
 * 이 파일명은 나중에 ptx_loader가 파일을 열어 PTX 코드를 파싱하는 데 사용된다.
 * 실행 컨텍스트: 호스트 스레드. 동시성: 단일 파싱 스레드.
 *
 * 호출 체인:
 *   cuobjdump_parse() → [setCuobjdumpptxfilename] → cuobjdumpPTXSection::setPTXfilename()
 */
void setCuobjdumpptxfilename(
    const char *filename, std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  printf("Adding ptx filename: %s\n", filename);      // [한국어] PTX 파일명 설정 로그 출력
  cuobjdumpSection *x = cuobjdumpSectionList.front(); // [한국어] 현재 파싱 중인 섹션 포인터 획득
  if (dynamic_cast<cuobjdumpPTXSection *>(x) == NULL) {
    // [한국어] front() 섹션이 PTX가 아니라 ELF 섹션인 경우 — 파서 상태 불일치, 즉시 abort
    assert(0 &&
           "You shouldn't be trying to add a ptxfilename to an elf section");
  }
  (dynamic_cast<cuobjdumpPTXSection *>(x))->setPTXfilename(filename);
  // [한국어] 타입 검증 통과 후 PTX 섹션 객체에 파일명 저장.
  // dynamic_cast는 위에서 이미 NULL 체크를 했으므로 여기서는 안전하게 사용된다.
}

/*
 * [한국어]
 * setCuobjdumpelffilename - cuobjdump 파서 콜백: ELF 섹션의 파일명 설정
 *
 * @filename: cuobjdump가 추출한 ELF 파일의 임시 경로 문자열.
 * @cuobjdumpSectionList: 섹션 리스트. front()가 현재 파싱 중인 ELF 섹션이어야 한다.
 * @return: 없음 (void). front()가 ELF 섹션이 아니면 assert로 abort.
 *
 * cuobjdump가 fat binary에서 ELF(SASS) 코드를 임시 파일로 추출했을 때 경로를
 * 현재 ELF 섹션에 등록한다. dynamic_cast로 ELF 섹션임을 검증한다.
 * 실행 컨텍스트: 호스트 스레드. 동시성: 단일 파싱 스레드.
 *
 * 호출 체인:
 *   cuobjdump_parse() → [setCuobjdumpelffilename] → cuobjdumpELFSection::setELFfilename()
 */
void setCuobjdumpelffilename(
    const char *filename, std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  if (dynamic_cast<cuobjdumpELFSection *>(cuobjdumpSectionList.front()) ==
      NULL) {
    // [한국어] front() 섹션이 PTX 섹션인 경우 — ELF 파일명을 PTX 섹션에 추가하려는 잘못된 상태, abort
    assert(0 &&
           "You shouldn't be trying to add a elffilename to an ptx section");
  }
  (dynamic_cast<cuobjdumpELFSection *>(cuobjdumpSectionList.front()))
      ->setELFfilename(filename);
  // [한국어] 타입 검증 통과 후 ELF 섹션 객체에 ELF 파일 경로를 저장.
  // 이 경로는 ELF 바이너리 로더가 SASS 코드를 추출할 때 사용된다.
}

/*
 * [한국어]
 * setCuobjdumpsassfilename - cuobjdump 파서 콜백: ELF 섹션의 SASS 파일명 설정
 *
 * @filename: cuobjdump가 추출한 SASS(디스어셈블된 기계어) 파일의 임시 경로.
 * @cuobjdumpSectionList: 섹션 리스트. front()가 현재 파싱 중인 ELF 섹션이어야 한다.
 * @return: 없음 (void). front()가 ELF 섹션이 아니면 assert로 abort.
 *
 * cuobjdump --dump-sass로 추출된 SASS 코드의 파일 경로를 현재 ELF 섹션에 등록한다.
 * SASS는 NVIDIA GPU의 실제 ISA로 PTXPlus(PTXPlus emulation) 경로에서 사용된다.
 * dynamic_cast로 ELF 섹션임을 검증한다.
 * 실행 컨텍스트: 호스트 스레드. 동시성: 단일 파싱 스레드.
 *
 * 호출 체인:
 *   cuobjdump_parse() → [setCuobjdumpsassfilename] → cuobjdumpELFSection::setSASSfilename()
 */
void setCuobjdumpsassfilename(
    const char *filename, std::list<cuobjdumpSection *> &cuobjdumpSectionList) {
  if (dynamic_cast<cuobjdumpELFSection *>(cuobjdumpSectionList.front()) ==
      NULL) {
    // [한국어] front() 섹션이 PTX인 경우 — SASS 파일명을 PTX 섹션에 추가하려는 잘못된 상태, abort
    assert(0 &&
           "You shouldn't be trying to add a sassfilename to an ptx section");
  }
  (dynamic_cast<cuobjdumpELFSection *>(cuobjdumpSectionList.front()))
      ->setSASSfilename(filename);
  // [한국어] 타입 검증 통과 후 ELF 섹션 객체에 SASS 파일 경로를 저장.
  // SASS 파일은 PTXPlus 시뮬레이션 경로에서 로드되어 기계어 수준 시뮬레이션에 사용된다.
}

//! Return the executable file of the process containing the PTX/SASS code
//!
//! This Function returns the executable file ran by the process.  This
//! executable is supposed to contain the PTX/SASS code.  It provides workaround
//! for processes running on valgrind by dereferencing /proc/<pid>/exe within
//! the GPGPU-Sim process before calling cuobjdump to extract PTX/SASS.  This is
//! needed because valgrind uses x86 emulation to detect memory leak.  Other
//! processes (e.g. cuobjdump) reading /proc/<pid>/exe will see the emulator
//! executable instead of the application binary.
//!
// In SST need the string to pass the binary information
// as we cannot get it from /proc/self/exe
/*
 * [한국어]
 * get_app_binary(const char *fn) - SST 통합 모드용: 직접 지정된 바이너리 경로 반환
 *
 * @fn: 실행 바이너리의 경로 문자열 (SST 프레임워크가 명시적으로 전달).
 * @return: fn을 그대로 std::string으로 래핑하여 반환.
 *
 * SST(Structural Simulation Toolkit) 연동 모드에서는 /proc/self/exe로 실행 파일을
 * 자동 탐지하는 것이 불가능하므로 (SST 컴포넌트로 실행되기 때문), 바이너리 경로를
 * 외부에서 명시적으로 전달받아 반환한다. 이 오버로드는 아래의 무인수 버전과
 * 쌍을 이루며, 호출 컨텍스트에 따라 둘 중 하나가 선택된다.
 * 실행 컨텍스트: 호스트 스레드 (fat binary 등록 또는 시뮬레이터 초기화 시).
 *
 * 호출 체인:
 *   get_app_cuda_version(fn) → [get_app_binary(fn)] → get_app_cuda_version_internal()
 */
std::string get_app_binary(const char *fn) {
  printf("self exe links to: %s\n", fn); // [한국어] 전달받은 바이너리 경로를 로그로 출력
  return fn;                             // [한국어] 전달받은 경로를 std::string으로 변환하여 반환
}

/*
 * [한국어]
 * get_app_binary() - /proc/self/exe 또는 macOS API로 현재 실행 바이너리 경로 반환
 *
 * @return: 현재 프로세스의 실행 파일 절대 경로 문자열. 실패 시 exit(1).
 *
 * Linux에서는 /proc/self/exe 심볼릭 링크를 readlink로 역참조하여 절대 경로를 얻는다.
 * valgrind 환경에서는 cuobjdump 같은 자식 프로세스가 /proc/<pid>/exe를 읽으면
 * valgrind 에뮬레이터 바이너리를 가리키므로, GPGPU-Sim 프로세스 내에서 미리
 * readlink로 실제 앱 경로를 확인하는 워크어라운드이다.
 * macOS에서는 _NSGetExecutablePath()를 사용한다 (Linux /proc이 없음).
 * 실행 컨텍스트: 호스트 스레드 (fat binary 등록 시).
 *
 * 호출 체인:
 *   get_app_cuda_version() → [get_app_binary()] → get_app_cuda_version_internal()
 *   get_app_binary_name() 의 입력으로도 사용된다.
 */
std::string get_app_binary() {
  char self_exe_path[1025]; // [한국어] 실행 파일 경로를 담을 버퍼 (최대 1024자 + NULL)
#ifdef __APPLE__
  uint32_t size = sizeof(self_exe_path);              // [한국어] 버퍼 크기를 uint32_t로 전달 (macOS API 요구사항)
  if (_NSGetExecutablePath(self_exe_path, &size) != 0) {
    // [한국어] 버퍼가 너무 작아 경로를 담지 못한 경우 — size에 필요한 크기가 채워지지만 여기서는 abort
    printf("GPGPU-Sim ** ERROR: _NSGetExecutablePath input buffer too small\n");
    exit(1); // [한국어] 복구 불가능한 초기화 오류 — 시뮬레이터 종료
  }
#else
  std::stringstream exec_link;         // [한국어] /proc/self/exe 경로 조합용 스트림
  exec_link << "/proc/self/exe";       // [한국어] Linux에서 현재 프로세스의 실행 파일을 가리키는 심볼릭 링크 경로

  ssize_t path_length = readlink(exec_link.str().c_str(), self_exe_path, 1024);
  // [한국어] /proc/self/exe 심볼릭 링크를 역참조하여 실제 실행 파일 절대 경로를 얻음.
  // 반환값은 실제 쓰여진 바이트 수 (NULL 종료 문자 미포함). 실패 시 -1.
  assert(path_length != -1);           // [한국어] readlink 실패(심볼릭 링크 없음, 권한 오류 등)는 복구 불가 — abort
  self_exe_path[path_length] = '\0';   // [한국어] readlink는 NULL을 추가하지 않으므로 수동으로 종료 문자 추가
#endif

  printf("self exe links to: %s\n", self_exe_path); // [한국어] 탐지된 실행 파일 경로 로그 출력
  return self_exe_path;                              // [한국어] 경로 문자열을 std::string으로 변환하여 반환
}

// above func gives abs path whereas this give just the name of application.
/*
 * [한국어]
 * get_app_binary_name - 절대 경로에서 애플리케이션 파일 이름(확장자 제외)만 추출
 *
 * @abs_path: get_app_binary()가 반환한 실행 파일 절대 경로 (예: "/path/to/matrixMul").
 * @return: 경로에서 디렉토리와 확장자를 제거한 순수 파일명 (예: "matrixMul").
 *          반환 포인터는 내부 strdup 버퍼를 가리키므로 호출자가 free해야 한다.
 *
 * '/' 구분자로 경로를 순회하여 마지막 토큰(파일명 부분)만 추출하고,
 * '.' 구분자로 다시 토큰화하여 확장자를 제거한다. 결과는 애플리케이션 이름으로
 * 임시 파일 생성이나 설정 파일 검색 등에 사용된다.
 * macOS는 미테스트 상태이므로 abort()로 안전하게 차단한다.
 * 실행 컨텍스트: 호스트 스레드. 동시성: strtok는 내부 static 포인터를 사용하므로 비재진입성 주의.
 *
 * 호출 체인:
 *   (fat binary 처리 경로) → [get_app_binary_name] (get_app_binary() 결과를 입력으로 사용)
 */
char *get_app_binary_name(std::string abs_path) {
  char *self_exe_path = NULL;         // [한국어] 최종 결과(파일명) 포인터 초기화
#ifdef __APPLE__
  // TODO: get apple device and check the result.
  printf("WARNING: not tested for Apple-mac devices \n"); // [한국어] macOS 미테스트 경고 출력
  abort();                            // [한국어] macOS에서 안전하게 차단 (미구현)
#else
  char *buf = strdup(abs_path.c_str());
  // [한국어] abs_path를 heap에 복사 (strtok가 원본 문자열을 수정하기 때문에 복사본 필요)
  char *token = strtok(buf, "/");     // [한국어] '/'로 경로 분리 시작 (첫 번째 토큰)
  while (token != NULL) {
    self_exe_path = token;            // [한국어] 매 반복마다 마지막 토큰을 갱신 — 루프 종료 시 파일명이 됨
    token = strtok(NULL, "/");        // [한국어] 다음 '/' 토큰 획득 (NULL이면 루프 종료)
  }
#endif
  self_exe_path = strtok(self_exe_path, ".");
  // [한국어] 파일명에서 '.' 이후(확장자)를 제거. 예: "matrixMul.exe" → "matrixMul".
  // strtok는 내부 static 상태를 사용하므로 위의 strtok 루프와 독립적으로 사용 가능.
  printf("self exe links to: %s\n", self_exe_path); // [한국어] 추출된 앱 이름 로그 출력
  return self_exe_path;                              // [한국어] 순수 앱 파일명 반환 (buf 내부 포인터)
}

/*
 * [한국어]
 * get_app_cuda_version_internal - ldd/strings 명령으로 앱이 링크한 CUDA 버전 탐지 (내부 구현)
 *
 * @app_binary: 탐색할 실행 파일 경로 (get_app_binary()가 반환한 값).
 * @return: 앱이 링크한 CUDA 런타임 버전 정수 (예: 10020 = CUDA 10.2). 실패 시 exit(1).
 *
 * ldd 명령으로 앱이 의존하는 libcudart.so 버전을 추출하고, strings 명령으로
 * Balar/Vanadis SST 통합 바이너리 내 libcudart_vanadis.a 버전 문자열도 함께 검색한다.
 * 결과를 임시 파일(mkstemp)에 저장하고 atoi로 정수 버전으로 변환한다.
 * 이 버전 정보는 GPGPU-Sim이 어떤 CUDA API 집합을 활성화할지 결정하는 데 사용된다.
 * 실행 컨텍스트: 호스트 스레드. 동시성: mkstemp/system/fopen은 단일 호출 보장.
 *
 * 호출 체인:
 *   get_app_cuda_version() / get_app_cuda_version(fn) → [get_app_cuda_version_internal]
 */
static int get_app_cuda_version_internal(std::string app_binary) {
  int app_cuda_version = 0;           // [한국어] 탐지된 CUDA 버전 (초기값 0: 탐지 실패 표시)
  char fname[1024];                   // [한국어] 임시 파일 이름 버퍼
  snprintf(fname, 1024, "_app_cuda_version_XXXXXX");
  // [한국어] mkstemp용 임시 파일 이름 템플릿. "XXXXXX"는 mkstemp가 유일한 문자로 치환한다.
  int fd = mkstemp(fname);            // [한국어] 임시 파일 생성 및 파일 디스크립터 반환 (fname이 실제 이름으로 갱신됨)
  close(fd);                          // [한국어] 파일 디스크립터 즉시 닫기 — 이후 shell 명령이 파일에 직접 쓸 것임
  // Weili: Add way to extract CUDA version information from Balar Vanadis
  // binary (stored as a const string)
  std::string app_cuda_version_command =
      "ldd " + app_binary +
      " | grep libcudart.so | sed  's/.*libcudart.so.\\(.*\\) =>.*/\\1/' > " +
      fname + " && strings " + app_binary +
      " | grep libcudart_vanadis.a | sed  "
      "'s/.*libcudart_vanadis.a.\\(.*\\)/\\1/' >> " +
      fname;
  // [한국어] 쉘 명령 문자열 구성:
  //   1) ldd <바이너리> | grep libcudart.so: 동적 링크된 CUDA 런타임 라이브러리 버전 탐지
  //   2) sed: "libcudart.so.<버전> =>" 형식에서 버전 부분만 추출
  //   3) > fname: 결과를 임시 파일에 저장
  //   4) strings <바이너리> | grep libcudart_vanadis.a: SST Balar 통합 바이너리 내
  //      정적으로 임베딩된 CUDA 버전 문자열 탐지
  //   5) >> fname: 결과를 임시 파일에 추가 (ldd 결과 다음에 이어붙임)
  int res = system(app_cuda_version_command.c_str());
  // [한국어] 위 쉘 명령을 실행. system()은 쉘을 fork하여 명령을 실행하고 종료를 기다린다.
  // 반환값 -1은 fork 실패 또는 쉘 실행 불가 상태를 의미한다.
  if (res == -1) {
    // [한국어] system() 실패 — fork/exec 오류. CUDA 버전을 탐지할 수 없으므로 종료.
    printf("Error - Cannot detect the app's CUDA version. Command: %s\n",
           app_cuda_version_command.c_str());
    exit(1); // [한국어] 복구 불가능한 초기화 오류 — 시뮬레이터 종료
  }
  FILE *cmd = fopen(fname, "r");      // [한국어] 임시 파일을 읽기 모드로 열어 버전 문자열 읽기
  char buf[256];                      // [한국어] 파일 한 줄을 읽을 버퍼
  while (fgets(buf, sizeof(buf), cmd) != 0) {
    // [한국어] 임시 파일에서 한 줄씩 읽기. 여러 줄이 있을 경우 마지막 줄이 최종 버전이 됨.
    std::cout << buf;                       // [한국어] 읽은 내용 stdout 출력 (디버그용)
    app_cuda_version = atoi(buf);           // [한국어] 줄 문자열을 정수 버전으로 변환 (예: "10020\n" → 10020)
  }
  fclose(cmd);                        // [한국어] 임시 파일 닫기 (이후 임시 파일 삭제는 별도 처리 필요)
  if (app_cuda_version == 0) {
    // [한국어] 파일이 비어 있거나 변환 결과가 0 — 버전 탐지 실패
    printf("Error - Cannot detect the app's CUDA version. Command: %s\n",
           app_cuda_version_command.c_str());
    exit(1); // [한국어] 버전 없이 시뮬레이터를 실행하면 API 호환성 판단 불가 — 종료
  }
  return app_cuda_version; // [한국어] 탐지된 CUDA 버전 정수 반환
}

/*
 * [한국어]
 * get_app_cuda_version(const char *fn) - SST 통합 모드용: 지정 바이너리의 CUDA 버전 탐지
 *
 * @fn: 탐색할 실행 바이너리 경로 (SST가 명시적으로 전달).
 * @return: 앱이 링크한 CUDA 런타임 버전 정수. 실패 시 exit(1).
 *
 * SST 통합 모드에서 /proc/self/exe 자동 탐지 대신 fn으로 바이너리 경로를 전달하는 오버로드.
 * get_app_binary(fn)으로 경로를 확인하고 공통 구현체에 위임한다.
 * 실행 컨텍스트: 호스트 스레드. 동시성: 단일 초기화 경로.
 *
 * 호출 체인:
 *   (SST 초기화 경로) → [get_app_cuda_version(fn)] → get_app_binary(fn)
 *                                                    → get_app_cuda_version_internal()
 */
static int get_app_cuda_version(const char *fn) {
  // Use for other simulator integration
  std::string app_binary = get_app_binary(fn);            // [한국어] 전달받은 경로를 std::string으로 변환
  return get_app_cuda_version_internal(app_binary);        // [한국어] 공통 구현체에 위임하여 CUDA 버전 탐지
}

/*
 * [한국어]
 * get_app_cuda_version() - /proc/self/exe로 현재 앱 바이너리의 CUDA 버전 자동 탐지
 *
 * @return: 앱이 링크한 CUDA 런타임 버전 정수. 실패 시 exit(1).
 *
 * 일반 실행 환경(SST 없음)에서 /proc/self/exe를 통해 실행 바이너리 경로를 자동으로
 * 탐지하고 CUDA 버전을 추출한다. valgrind 환경 지원을 위해 get_app_binary()에서
 * 미리 경로를 역참조한다 (상세는 get_app_binary() 주석 참고).
 * 실행 컨텍스트: 호스트 스레드. 동시성: 단일 초기화 경로.
 *
 * 호출 체인:
 *   (fat binary 등록 경로, 일반 모드) → [get_app_cuda_version()] → get_app_binary()
 *                                                                  → get_app_cuda_version_internal()
 */
static int get_app_cuda_version() {
  std::string app_binary = get_app_binary();              // [한국어] /proc/self/exe로 현재 실행 파일 경로 자동 탐지
  return get_app_cuda_version_internal(app_binary);        // [한국어] 공통 구현체에 위임하여 CUDA 버전 탐지
}

/*
 * [한국어]
 * cuobjdumpRegisterFatBinary - fat binary 핸들과 소스 파일명의 매핑을 등록한다
 *
 * @handle:   NVCC가 생성한 fatbin 핸들 번호 (1부터 시작하는 단조 증가 정수).
 *            cudaRegisterFatBiaryInternal_impl()에서 next_fat_bin_handle로 발급된 값.
 * @filename: fat binary가 생성된 원본 소스 파일명 (CUDA < 6.0: 포인터 산술로 추출,
 *            CUDA >= 6.0: "default" 문자열).
 * @context:  현재 CUctx_st 컨텍스트 (현재 구현에서는 사용하지 않음, 확장 여지).
 * @return:   없음 (void).
 *
 * 이 함수는 cuobjdump가 출력하는 섹션 정보에서 파일명으로 PTX 코드를 찾아야 할 때
 * 어떤 핸들이 어떤 파일명에 대응하는지를 fatbinmap에 기록한다.
 * cudaLaunch() 또는 cudaRegisterFunction() 이후, cuobjdumpParseBinary() 내부에서
 * 핸들 → 파일명 매핑을 이 맵에서 조회하여 PTX/SASS 섹션을 특정한다.
 *
 * 실행 컨텍스트: CPU 호스트 스레드 (앱 시작 시 __cudaRegisterFatBinary 체인에서 호출).
 * 동기화: 단일 스레드 초기화 시점에 호출되므로 락 불필요.
 *
 * 호출 체인:
 *   cudaRegisterFatBiaryInternal_impl() → ctx->api->cuobjdumpRegisterFatBinary()
 */
//! Keep track of the association between filename and cubin handle
void cuda_runtime_api::cuobjdumpRegisterFatBinary(unsigned int handle,
                                                  const char *filename,
                                                  CUctx_st *context) {
  fatbinmap[handle] = filename; // [한국어] handle → filename 매핑을 fatbinmap STL map에 삽입 (이미 존재하면 덮어씀)
}

/*******************************************************************************
 * Add internal cuda runtime API call to accept gpgpu_context *
 *******************************************************************************/
/*
 * [한국어]
 * cudaSetDeviceInternal - GPGPU-Sim 시뮬레이터에서 활성 GPU 장치 번호를 설정
 *
 * @device: 활성화할 장치 인덱스. GPGPU-Sim은 단일 장치만 시뮬레이션하므로
 *          보통 0을 지정하며, 0 이상 num_devices()-1 이하만 유효.
 * @gpgpu_ctx: 외부에서 전달한 gpgpu_context 포인터. NULL이면 GPGPU_Context()로
 *             전역 싱글톤을 획득.
 * @return: 성공 시 cudaSuccess, 잘못된 장치 번호 시 cudaErrorInvalidDevice.
 *
 * CUDA 런타임의 cudaSetDevice()에 대응하는 납부 구현. GPGPUSim_Init()으로
 * 시뮬레이터를 초기화한 뒤 ctx->api->g_active_device을 갱신한다.
 */
cudaError_t cudaSetDeviceInternal(int device, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;  // [한국어] 외부에서 이미 획득한 컨텍스트 사용 (SST 등)
  } else {
    ctx = GPGPU_Context();  // [한국어] 전역 싱글톤 컨텍스트 획득
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // set the active device to run cuda
  // [한국어] 요청한 device 번호가 시뮬레이터가 제공하는 장치 수 범위 내인지 검사
  if (device <= ctx->GPGPUSim_Init()->num_devices()) {
    ctx->api->g_active_device = device;  // [한국어] 활성 장치 번호 갱신
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorInvalidDevice;
  }
}

/*
 * [한국어]
 * cudaGetDeviceInternal - 현재 활성화된 GPU 장치 번호를 조회
 *
 * @device: 결과를 저장할 int 포인터.
 * @gpgpu_ctx: 외부 컨텍스트 포인터. NULL이면 전역 싱글톤 사용.
 * @return: 항상 cudaSuccess. GPGPU-Sim은 장치 0 하나만 제공.
 */
cudaError_t cudaGetDeviceInternal(int *device,
                                  gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;  // [한국어] 외부 컨텍스트 사용
  } else {
    ctx = GPGPU_Context();  // [한국어] 전역 싱글톤 컨텍스트 획득
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *device = ctx->api->g_active_device;  // [한국어] cudaSetDeviceInternal에서 설정한 값 반환
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * cudaDeviceGetLimitInternal - CUDA 장치 리소스 한계값을 조회
 *
 * @pValue: 한계값 결과를 저장할 size_t 포인터.
 * @limit:  조회할 한계 종류 (cudaLimitStackSize, cudaLimitMallocHeapSize 등).
 * @gpgpu_ctx: 외부 컨텍스트 포인터.
 * @return: 성공 시 cudaSuccess, 미지원 limit 시 abort.
 *
 * gpgpusim.config의 -stack_limit, -heap_limit, -sync_depth_limit,
 * -pending_launch_count_limit 옵션에 의해 값이 결정된다.
 */
__host__ cudaError_t CUDARTAPI cudaDeviceGetLimitInternal(
    size_t *pValue, cudaLimit limit, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  const struct cudaDeviceProp *prop = dev->get_prop();  // [한국어] cudaDeviceProp은 GPGPUSim_Init()에서 채워짐
  const gpgpu_sim_config &config = dev->get_gpgpu()->get_config();  // [한국어] gpgpusim.config 파싱 결과
  switch (limit) {
    case 0:  // cudaLimitStackSize
      // [한국어] 스레드당 스택 한계 — gpgpusim.config -stack_limit 옵션
      *pValue = config.stack_limit();
      break;
    case 2:  // cudaLimitMallocHeapSize
      // [한국어] 디바이스 힙 크기 한계 — gpgpusim.config -heap_limit 옵션
      *pValue = config.heap_limit();
      break;
#if (CUDART_VERSION > 5050)
    case 3:  // cudaLimitDevRuntimeSyncDepth
      // [한국어] CDP(CUDA Dynamic Parallelism) 동기화 깊이 한계 — Fermi 이상에서만 지원
      if (prop->major > 2) {
        *pValue = config.sync_depth_limit();
        break;
      } else {
        printf("ERROR:Limit %d is not supported on this architecture \n",
               limit);
        abort();
      }
    case 4:  // cudaLimitDevRuntimePendingLaunchCount
      // [한국어] CDP 미완료 커널 최대 개수 — Fermi 이상에서만 지원
      if (prop->major > 2) {
        *pValue = config.pending_launch_count_limit();
        break;
      } else {
        printf("ERROR:Limit %d is not supported on this architecture \n",
               limit);
        abort();
      }
#endif
    default:
      printf("ERROR:Limit %d unimplemented \n", limit);
      abort();
  }
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * cudaRegisterFatBiaryInternal_impl - NVCC가 생성한 fat binary를 GPGPU-Sim에 등록
 *
 * @fatCubin: NVCC가 생성한 fat binary 핸들(실제로는 납부 구조체 포인터).
 * @gpgpu_ctx: gpgpu_context 포인터.
 * @app_binary_path: 현재 실행 중인 애플리케이션 바이너리 경로.
 * @app_cuda_version: 앱이 링크한 CUDA 런타임 주요 버전 (ldd/strings로 탐지).
 * @ctx_cuobjdumpInit_func: cuobjdump 초기화 래퍼 람다.
 * @return: fat_cubin_handle을 void**로 캐스팅하여 반환. 이후 cudaLaunch에서
 *          커널 함수를 찾을 때 사용.
 *
 * 두 가지 경로:
 *   1) cuobjdump 사용 (CUDA 4.0+ 권장): fat binary에서 PTX/SASS 섹션을
 *      cuobjdump로 추출하여 파싱. -gpgpu_ptx_use_cuobjdump 옵션으로 활성화.
 *   2) 레거시 fat binary 직접 파싱 (CUDA < 8.0): __cudaFatCudaBinary 구조체에서
 *      PTX 문자열을 직접 선택하여 ptx_loader에 전달.
 *
 * gpgpusim.config 관련: -gpgpu_ptx_use_cuobjdump, -gpgpu_ptx_force_max_capability,
 *                       -gpgpu_ptx_convert_to_ptxplus
 */
// Internal implementation for cudaRegisterFatBiaryInternal
void **cudaRegisterFatBiaryInternal_impl(
    void *fatCubin, gpgpu_context *gpgpu_ctx, std::string &app_binary_path,
    int app_cuda_version,
    std::function<void(gpgpu_context *)> ctx_cuobjdumpInit_func) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#if (CUDART_VERSION < 2010)
  printf(
      "GPGPU-Sim PTX: ERROR ** this version of GPGPU-Sim requires CUDA 2.1 or "
      "higher\n");
  exit(1);
#endif
  CUctx_st *context = GPGPUSim_Context(ctx);
  static unsigned next_fat_bin_handle = 1;
  // [한국어] cuobjdump 기반 fat binary 처리 경로 (권장)
  if (context->get_device()->get_gpgpu()->get_config().use_cuobjdump()) {
    // The following workaround has only been verified on 64-bit systems.
    if (sizeof(void *) == 4)
      printf(
          "GPGPU-Sim PTX: FatBin file name extraction has not been tested on "
          "32-bit system.\n");

    // This code will get the CUDA version the app was compiled with.
    // We need this to determine how to handle the parsing of the binary.
    // Making this a runtime variable based on the app, enables GPGPU-Sim
    // compiled with a newer version of CUDA to run apps compiled with older
    // versions of CUDA. This is especially useful for PTXPLUS execution.
    // Skip cuda version check for pytorch application
    // [한국어] PyTorch 바이너리는 python이라는 서브스트링을 포함하며, CUDA 버전
    //        체크를 생략 (PyTorch는 자체 CUDA 런타임을 내장)
    int pos = app_binary_path.find("python");
    if (pos == std::string::npos) {
      // Not pytorch app : checking cuda version
      // [한국어] 앱과 시뮬레이터의 CUDA 주요 버전이 일치해야 PTX 문법/ABI 호환
      assert(
          app_cuda_version == CUDART_VERSION / 1000 &&
          "The app must be compiled with same major version as the simulator.");
    }

    // int app_cuda_version = get_app_cuda_version();
    // assert( app_cuda_version == CUDART_VERSION / 1000  && "The app must be
    // compiled with same major version as the simulator." );
    const char *filename;
#if CUDART_VERSION < 6000
    // FatBin handle from the .fatbin.c file (one of the intermediate files
    // generated by NVCC)
    // [한국어] CUDA 6.0 미만: NVCC 중간 파일의 fatbin 구조체에서 소스 파일명 추출
    typedef struct {
      int m;
      int v;
      const unsigned long long *d;
      char *f;
    } __fatDeviceText __attribute__((aligned(8)));
    __fatDeviceText *fatDeviceText = (__fatDeviceText *)fatCubin;

    // Extract the source code file name that generate the given FatBin.
    // - Obtains the pointer to the actual fatbin structure from the FatBin
    // handle (fatCubin).
    // - An integer inside the fatbin structure contains the relative offset to
    // the source code file name.
    // - This offset differs among different CUDA and GCC versions.
    char *pfatbin = (char *)fatDeviceText->d;
    int offset = *((int *)(pfatbin + 48));
    filename = (pfatbin + 16 + offset);
#else
    filename = "default";  // [한국어] CUDA 6.0 이상: cuobjdump가 파일명을 출력하므로 "default"로 통일
#endif

    // The extracted file name is associated with a fat_cubin_handle passed
    // into cudaLaunch().  Inside cudaLaunch(), the associated file name is
    // used to find the PTX/SASS section from cuobjdump, which contains the
    // PTX/SASS code for the launched kernel function.
    // This allows us to work around the fact that cuobjdump only outputs the
    // file name associated with each section.
    // [한국어] fat_cubin_handle은 1부터 시작하는 단조 증가 정수. 커널 실행 시
    //        fatbinmap을 통해 파일명을 역조회하여 해당 PTX 섹션 선택.
    unsigned long long fat_cubin_handle = next_fat_bin_handle;
    next_fat_bin_handle++;
    printf(
        "GPGPU-Sim PTX: __cudaRegisterFatBinary, fat_cubin_handle = %llu, "
        "filename=%s\n",
        fat_cubin_handle, filename);
    /*!
     * This function extracts all data from all files in first call
     * then for next calls, only returns the appropriate number
     */
    assert(fat_cubin_handle >= 1);
    // [한국어] 첫 번째 fat binary 등록 시에만 cuobjdump 초기화 수행
    if (fat_cubin_handle == 1) ctx_cuobjdumpInit_func(ctx);
    ctx->api->cuobjdumpRegisterFatBinary(fat_cubin_handle, filename, context);

    return (void **)fat_cubin_handle;  // [한국어] 정수 핸들을 void**로 캐스팅하여 NVCC 런타임에 반환
  }
#if (CUDART_VERSION < 8000)
  else {
    // [한국어] 레거시 경로: cuobjdump를 사용하지 않고 __cudaFatCudaBinary에서
    //        직접 PTX 버전을 선택하여 로드 (CUDA 8.0 미만에서만 컴파일됨)
    static unsigned source_num = 1;  // [한국어] PTX 소스 번호 (ptxinfo 식별용)
    unsigned long long fat_cubin_handle = next_fat_bin_handle++;
    __cudaFatCudaBinary *info = (__cudaFatCudaBinary *)fatCubin;
    assert(info->version >= 3);
    unsigned num_ptx_versions = 0;
    unsigned max_capability = 0;
    unsigned selected_capability = 0;
    bool found = false;
    unsigned forced_max_capability = context->get_device()
                                         ->get_gpgpu()
                                         ->get_config()
                                         .get_forced_max_capability();
    if (!info->ptx) {
      printf(
          "ERROR: Cannot find ptx code in cubin file\n"
          "\tIf you are using CUDA 4.0 or higher, please enable "
          "-gpgpu_ptx_use_cuobjdump or downgrade to CUDA 3.1\n");
      exit(1);
    }
    while (info->ptx[num_ptx_versions].gpuProfileName != NULL) {
      unsigned capability = 0;
      sscanf(info->ptx[num_ptx_versions].gpuProfileName, "compute_%u",
             &capability);
      printf(
          "GPGPU-Sim PTX: __cudaRegisterFatBinary found PTX versions for "
          "'%s', ",
          info->ident);
      printf("capability = %s\n", info->ptx[num_ptx_versions].gpuProfileName);
      // [한국어] forced_max_capability가 설정된 경우 그 이하의 최고 capability 선택
      if (forced_max_capability) {
        if (capability > max_capability &&
            capability <= forced_max_capability) {
          found = true;
          max_capability = capability;
          selected_capability = num_ptx_versions;
        }
      } else {
        if (capability > max_capability) {
          found = true;
          max_capability = capability;
          selected_capability = num_ptx_versions;
        }
      }
      num_ptx_versions++;
    }
    if (found) {
      printf("GPGPU-Sim PTX: Loading PTX for %s, capability = %s\n",
             info->ident, info->ptx[selected_capability].gpuProfileName);
      symbol_table *symtab;
      const char *ptx = info->ptx[selected_capability].ptx;
      if (context->get_device()
              ->get_gpgpu()
              ->get_config()
              .convert_to_ptxplus()) {
        // [한국어] 레거시 경로에서는 PTXPlus(SASS→PTXPlus 변환)를 지원하지 않음
        printf(
            "GPGPU-Sim PTX: ERROR ** PTXPlus is only supported through "
            "cuobjdump\n"
            "\tEither enable cuobjdump or disable PTXPlus in your "
            "configuration file\n");
        exit(1);
      } else {
        symtab = ctx->gpgpu_ptx_sim_load_ptx_from_string(ptx, source_num);
        context->add_binary(symtab, fat_cubin_handle);
        ctx->gpgpu_ptxinfo_load_from_string(ptx, source_num, max_capability,
                                            context->no_of_ptx);
      }
      source_num++;
      // [한국어] 정적 전역 변수/상수 메모리 초기화 (PTX .global/.const)
      ctx->api->load_static_globals(symtab, STATIC_ALLOC_LIMIT, 0xFFFFFFFF,
                                    context->get_device()->get_gpgpu());
      ctx->api->load_constants(symtab, STATIC_ALLOC_LIMIT,
                               context->get_device()->get_gpgpu());
    } else {
      printf(
          "GPGPU-Sim PTX: warning -- did not find an appropriate PTX in "
          "cubin\n");
    }
    return (void **)fat_cubin_handle;
  }
#else
  else {
    // [한국어] CUDA 8.0 이상에서는 cuobjdump를 사용하지 않는 경로를 지원하지 않음
    printf("ERROR **  __cudaRegisterFatBinary() needs to be updated\n");
    abort();
  }
#endif
}

/*
 * [한국어]
 * cudaRegisterFatBinaryInternal(SST용) - SST 통합 모드에서 fat binary 등록
 *
 * @fn: 실행 바이너리 경로 (SST가 명시적으로 전달).
 * @fatCubin: NVCC fat binary 핸들.
 * @gpgpu_ctx: gpgpu_context 포인터.
 * @return: fat_cubin_handle.
 */
void **cudaRegisterFatBinaryInternal(const char *fn, void *fatCubin,
                                     gpgpu_context *gpgpu_ctx = NULL) {
  std::string app_binary_path = get_app_binary(fn);
  int app_cuda_version = get_app_cuda_version(fn);
  auto ctx_cuobjdumpInit = [=](gpgpu_context *ctx) {
    ctx->api->cuobjdumpInit(fn);
  };
  return cudaRegisterFatBiaryInternal_impl(fatCubin, gpgpu_ctx, app_binary_path,
                                           app_cuda_version, ctx_cuobjdumpInit);
}

/*
 * [한국어]
 * cudaRegisterFatBinaryInternal(일반용) - /proc/self/exe 기반 fat binary 등록
 *
 * @fatCubin: NVCC fat binary 핸들.
 * @gpgpu_ctx: gpgpu_context 포인터.
 * @return: fat_cubin_handle.
 */
void **cudaRegisterFatBinaryInternal(void *fatCubin,
                                     gpgpu_context *gpgpu_ctx = NULL) {
  std::string app_binary_path = get_app_binary();     // [한국어] /proc/self/exe로 실행 파일 경로 획득
  int app_cuda_version = get_app_cuda_version();      // [한국어] ldd/strings로 CUDA 버전 탐지
  auto ctx_cuobjdumpInit = [](gpgpu_context *ctx) {
    ctx->api->cuobjdumpInit();  // [한국어] cuobjdump 초기화: 섹션 추출 및 파싱
  };
  return cudaRegisterFatBiaryInternal_impl(fatCubin, gpgpu_ctx, app_binary_path,
                                           app_cuda_version, ctx_cuobjdumpInit);
}

/*
 * [한국어]
 * cudaRegisterFunctionInternal - NVCC가 생성한 커널 함수 심볼을 GPGPU-Sim에 등록
 *
 * @fatCubinHandle: __cudaRegisterFatBinary()가 반환한 fat_cubin_handle.
 * @hostFun: 호스트측 커널 함수 포인터. cudaLaunch(hostFun) 시 이 값으로 검색.
 * @deviceFun: PTX에서의 커널 함수 이름.
 * @deviceName: 커널의 장치측 심볼 이름.
 * @thread_limit, tid, bid, bDim, gDim: NVCC 런타임에서 전달되는 추가 정보
 *                                       (GPGPU-Sim에서는 사용하지 않음).
 * @gpgpu_ctx: gpgpu_context 포인터.
 *
 * 이 함수는 CUDA 런타임의 __cudaRegisterFunction()에 대응. fatCubinHandle에
 * 해당하는 fat binary가 아직 파싱되지 않았으면 cuobjdumpParseBinary()를 호출하여
 * PTX를 로드한 뒤, hostFun ↔ deviceFun 매핑을 CUctx_st에 등록한다.
 */
void cudaRegisterFunctionInternal(void **fatCubinHandle, const char *hostFun,
                                  char *deviceFun, const char *deviceName,
                                  int thread_limit, uint3 *tid, uint3 *bid,
                                  dim3 *bDim, dim3 *gDim,
                                  gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  unsigned fat_cubin_handle = (unsigned)(unsigned long long)fatCubinHandle;
  printf(
      "GPGPU-Sim PTX: __cudaRegisterFunction %s : hostFun 0x%p, "
      "fat_cubin_handle = %u\n",
      deviceFun, hostFun, fat_cubin_handle);
  // [한국어] cuobjdump 사용 시 필요한 fat binary를 아직 파싱하지 않았다면 파싱
  if (context->get_device()->get_gpgpu()->get_config().use_cuobjdump())
    ctx->cuobjdumpParseBinary(fat_cubin_handle);
  // [한국어] hostFun 포인터를 키로, deviceFun 이름을 값으로 컨텍스트에 등록
  context->register_function(fat_cubin_handle, hostFun, deviceFun);
}

/*
 * [한국어]
 * cudaRegisterVarInternal - 정적 전역/상수 변수를 GPGPU-Sim에 등록
 *
 * @fatCubinHandle: fat_cubin_handle.
 * @hostVar: 호스트측 변수 포인터.
 * @deviceAddress: PTX에서의 변수 이름.
 * @deviceName: 동일한 변수 이름.
 * @ext, @size, @constant, @global: 변수 속성 플래그.
 * @gpgpu_ctx: gpgpu_context 포인터.
 *
 * constant && !global && !ext 이면 상수 메모리(constant memory) 변수로 등록하고,
 * !constant && !global && !ext 이면 글로벌 메모리(global memory) 변수로 등록한다.
 * 그 외 조합은 미지원.
 */
void cudaRegisterVarInternal(
    void **fatCubinHandle,
    char *hostVar,           // pointer to...something
    char *deviceAddress,     // name of variable
    const char *deviceName,  // name of variable (same as above)
    int ext, int size, int constant, int global,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf(
      "GPGPU-Sim PTX: __cudaRegisterVar: hostVar = %p; deviceAddress = %s; "
      "deviceName = %s\n",
      hostVar, deviceAddress, deviceName);
  printf(
      "GPGPU-Sim PTX: __cudaRegisterVar: Registering const memory space of %d "
      "bytes\n",
      size);
  // [한국어] cuobjdump 사용 시 fat binary 파싱 보장
  if (GPGPUSim_Context(ctx)
          ->get_device()
          ->get_gpgpu()
          ->get_config()
          .use_cuobjdump())
    ctx->cuobjdumpParseBinary((unsigned)(unsigned long long)fatCubinHandle);
  fflush(stdout);
  // [한국어] 상수 메모리 변수 등록 (__constant__)
  if (constant && !global && !ext) {
    ctx->func_sim->gpgpu_ptx_sim_register_const_variable(hostVar, deviceName,
                                                         size);
  } else if (!constant && !global && !ext) {  // [한국어] 글로벌 메모리 변수 등록 (__device__)
    ctx->func_sim->gpgpu_ptx_sim_register_global_variable(hostVar, deviceName,
                                                          size);
  } else
    cuda_not_implemented(__my_func__, __LINE__);
}

/*
 * [한국어]
 * cudaConfigureCallInternal - 커널 실행 구성(gridDim, blockDim, sharedMem, stream)을 스택에 저장
 *
 * @gridDim, @blockDim: 커널 그리드/블록 차원.
 * @sharedMem: 블록당 동적 공유 메모리 크기(bytes).
 * @stream: 실행할 CUDA 스트림.
 * @gpgpu_ctx: gpgpu_context 포인터.
 * @return: cudaSuccess.
 *
 * cudaConfigureCall()의 납부 구현. 이후 cudaSetupArgumentInternal()로 인수를
 * 채우고 cudaLaunchInternal()로 실제 발행한다.
 */
cudaError_t cudaConfigureCallInternal(dim3 gridDim, dim3 blockDim,
                                      size_t sharedMem, cudaStream_t stream,
                                      gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  struct CUstream_st *s = (struct CUstream_st *)stream;
  // [한국어] launch 설정을 스택에 푸시 (중첩 launch 미지원, 보통 1개)
  ctx->api->g_cuda_launch_stack.push_back(
      kernel_config(gridDim, blockDim, sharedMem, s));
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * cudaGetDeviceCountInternal - 시뮬레이션 가능한 GPU 장치 수를 반환
 *
 * GPGPU-Sim은 현재 단일 장치만 시뮬레이션하므로 *count = 1.
 */
__host__ cudaError_t CUDARTAPI
cudaGetDeviceCountInternal(int *count, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  *count = dev->num_devices();  // [한국어] GPGPU-Sim은 1개 장치만 시뮬레이션
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * cudaGetDevicePropertiesInternal - 지정 장치의 속성을 cudaDeviceProp에 복사
 *
 * @prop: 결과를 저장할 cudaDeviceProp 구조체 포인터.
 * @device: 장치 인덱스. 0만 유효.
 * @gpgpu_ctx: gpgpu_context 포인터.
 * @return: 유효한 장치면 cudaSuccess, 아니면 cudaErrorInvalidDevice.
 *
 * GPGPUSim_Init()에서 채운 cudaDeviceProp을 복사하여 반환. SM 수, 메모리 크기,
 * warp 크기 등 모두 gpgpusim.config에서 초기화된 값.
 */
__host__ cudaError_t CUDARTAPI cudaGetDevicePropertiesInternal(
    struct cudaDeviceProp *prop, int device, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  if (device <= dev->num_devices()) {
    *prop = *dev->get_prop();  // [한국어] GPGPUSim_Init()에서 생성된 속성 구조체 복사
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorInvalidDevice;
  }
}

/*
 * [한국어]
 * cudaChooseDeviceInternal - 요구 속성에 맞는 장치를 선택
 *
 * GPGPU-Sim은 장치가 하나뿐이므로 항상 0을 반환.
 */
__host__ cudaError_t CUDARTAPI
/*
 * [한국어]
 * CUDA 런타임 API cudaChooseDevice()의 GPGPU-Sim 납부 구현
 */
cudaChooseDeviceInternal(int *device, const struct cudaDeviceProp *prop,
                         gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  *device = dev->get_id();
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * cudaSetupArgumentInternal - cudaConfigureCall로 설정된 launch에 커널 인수 추가
 *
 * @arg: 호스트 메모리의 인수 데이터 포인터.
 * @size: 인수 크기(bytes).
 * @offset: 인수 스택 내 오프셋.
 * @gpgpu_ctx: gpgpu_context 포인터.
 *
 * launch 스택의 최상위 kernel_config에 인수를 복사/기록. 실제 전송은
 * cudaLaunchInternal에서 이뤄진다.
 */
cudaError_t cudaSetupArgumentInternal(const void *arg, size_t size,
                                      size_t offset,
                                      gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  gpgpusim_ptx_assert(!ctx->api->g_cuda_launch_stack.empty(),
                      "empty launch stack");
  kernel_config &config = ctx->api->g_cuda_launch_stack.back();
  // [한국어] 인수 데이터를 kernel_config에 기록 (실제 GPU 메모리 복사는 launch 시)
  config.set_arg(arg, size, offset);
  printf(
      "GPGPU-Sim PTX: Setting up arguments for %zu bytes starting at "
      "0x%llx..\n",
      size, (unsigned long long)arg);

  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * cudaLaunchInternal - 구성/인수가 완료된 커널을 GPGPU-Sim에 발행
 *
 * @hostFun: 호스트 측 커널 함수 심볼(포인터). GPGPU-Sim 낸에서
 *           register_function()으로 deviceFun과 매핑됨.
 * @gpgpu_ctx: gpgpu_context 포인터.
 * @return: cudaSuccess 또는 cudaErrorInvalidConfiguration.
 *
 * 이 함수는 cudaLaunch()의 실제 구현. launch 스택에서 설정을 pop하고,
 * kernel_info_t를 생성한 뒤 stream_manager로 전달하여 GPU 시뮬레이션 코어의
 * 스케줄러가 실행하도록 한다. 또한 체크포인트 재개(resume) 옵션을 처리한다.
 */
cudaError_t cudaLaunchInternal(const char *hostFun,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  // [한국어] 환경 변수 PTX_SIM_MODE_FUNC로 functional(1)/performance(0) 모드 강제 설정
  char *mode = getenv("PTX_SIM_MODE_FUNC");
  if (mode) sscanf(mode, "%u", &(ctx->func_sim->g_ptx_sim_mode));
  gpgpusim_ptx_assert(!ctx->api->g_cuda_launch_stack.empty(),
                      "empty launch stack");
  kernel_config config = ctx->api->g_cuda_launch_stack.back();
  {
    // [한국어] 빈 gridDim/blockDim은 launch 불가
    dim3 gridDim = config.grid_dim();
    dim3 blockDim = config.block_dim();
    if (gridDim.x * gridDim.y * gridDim.z == 0 ||
        blockDim.x * blockDim.y * blockDim.z == 0) {
      // can't launch
      printf("can't launch a empty kernel\n");
      ctx->api->g_cuda_launch_stack.pop_back();
      return g_last_cudaError = cudaErrorInvalidConfiguration;
    }
  }
  struct CUstream_st *stream = config.get_stream();

  printf("\nGPGPU-Sim PTX: cudaLaunch for 0x%p (mode=%s) on stream %u\n",
         hostFun,
         (ctx->func_sim->g_ptx_sim_mode) ? "functional simulation"
                                         : "performance simulation",
         stream ? stream->get_uid() : 0);
  // [한국어] kernel_info_t 생성: 커널 함수 엔트리, 인수, grid/block 차원 설정
  kernel_info_t *grid = ctx->api->gpgpu_cuda_ptx_sim_init_grid(
      hostFun, config.get_args(), config.grid_dim(), config.block_dim(),
      context);
  // do dynamic PDOM analysis for performance simulation scenario
  // [한국어] 동적 PDOM(post-dominator) 복수점 분석: warp 발산/수렴 처리용
  std::string kname = grid->name();
  function_info *kernel_func_info = grid->entry();
  if (kernel_func_info->is_pdom_set()) {
    printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
           kname.c_str());
  } else {
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           kname.c_str());
    kernel_func_info->do_pdom();
    kernel_func_info->set_pdom();
  }
  dim3 gridDim = config.grid_dim();
  dim3 blockDim = config.block_dim();

  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  checkpoint *g_checkpoint;
  g_checkpoint = new checkpoint();
  class memory_space *global_mem;
  global_mem = gpu->get_global_memory();

  // [한국어] 체크포인트 재개: 지정한 커널/CTA부터 이어서 실행
  if (gpu->resume_option == 1 && (grid->get_uid() == gpu->resume_kernel)) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/global_mem_%d.txt",
             grid->get_uid());

    g_checkpoint->load_global_mem(global_mem, f1name);
    for (int i = 0; i < gpu->resume_CTA; i++) grid->increment_cta_id();
  }
  // [한국어] 이전 커널은 메모리 상태를 복원만 하고 스킵
  if (gpu->resume_option == 1 && (grid->get_uid() < gpu->resume_kernel)) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/global_mem_%d.txt",
             grid->get_uid());

    g_checkpoint->load_global_mem(global_mem, f1name);
    printf("Skipping kernel %d as resuming from kernel %d\n", grid->get_uid(),
           gpu->resume_kernel);
    ctx->api->g_cuda_launch_stack.pop_back();
    return g_last_cudaError = cudaSuccess;
  }
  // [한국어] 체크포인트 생성 대상 이후의 커널은 스킵
  if (gpu->checkpoint_option == 1 &&
      (grid->get_uid() > gpu->checkpoint_kernel)) {
    printf("Skipping kernel %d as checkpoint from kernel %d\n", grid->get_uid(),
           gpu->checkpoint_kernel);
    ctx->api->g_cuda_launch_stack.pop_back();
    return g_last_cudaError = cudaSuccess;
  }
  printf(
      "GPGPU-Sim PTX: pushing kernel \'%s\' to stream %u, gridDim= (%u,%u,%u) "
      "blockDim = (%u,%u,%u) \n",
      kname.c_str(), stream ? stream->get_uid() : 0, gridDim.x, gridDim.y,
      gridDim.z, blockDim.x, blockDim.y, blockDim.z);
  // [한국어] stream_manager에 커널 launch operation을 push (비동기 실행)
  stream_operation op(grid, ctx->func_sim->g_ptx_sim_mode, stream);
  ctx->the_gpgpusim->g_stream_manager->push(op);
  ctx->api->g_cuda_launch_stack.pop_back();
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * cudaMallocInternal - GPU 글로벌 메모리 할당
 *
 * @devPtr: 할당된 디바이스 포인터를 저장할 포인터.
 * @size: 요청 크기(bytes).
 * @gpgpu_ctx: gpgpu_context 포인터.
 * @return: 성공 시 cudaSuccess, 실패 시 cudaErrorMemoryAllocation.
 */
cudaError_t cudaMallocInternal(void **devPtr, size_t size,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  *devPtr = context->get_device()->get_gpgpu()->gpu_malloc(size);
  if (g_debug_execution >= 3) {
    printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n",
           size, (unsigned long long)*devPtr);
    ctx->api->g_mallocPtr_Size[(unsigned long long)*devPtr] = size;
  }
  if (*devPtr) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

/*
 * [한국어]
 * cudaMallocHostInternal - 페이지 고정(pinning)된 호스트 메모리 할당
 *
 * @ptr: 할당된 호스트 포인터를 저장할 포인터.
 * @size: 요청 크기(bytes).
 * @gpgpu_ctx: gpgpu_context 포인터.
 * @return: 성공 시 cudaSuccess, 실패 시 cudaErrorMemoryAllocation.
 *
 * 실제 메모리는 malloc()으로 할당되며, pinned_memory_size 맵에 기록하여
 * cudaHostGetDevicePointerInternal() 시 GPU 메모리도 동일 크기로 할당한다.
 */
cudaError_t cudaMallocHostInternal(void **ptr, size_t size,
                                   gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *ptr = malloc(size);
  if (*ptr) {
    // track pinned memory size allocated in the host so that same amount of
    // memory is also allocated in GPU.
    // [한국어] 호스트 핀드 메모리 크기 추적 -> 이후 GPU 측 동일 크기 할당에 사용
    ctx->api->pinned_memory_size[*ptr] = size;
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

/*
 * [한국어]
 * cudaMallocHostSSTInternal - SST/Vanadis 환경에서 이미 할당된 호스트 주소를 기록
 *
 * SST 시뮬레이터가 호스트 메모리 할당을 대행하므로 libcuda는 주소만
 * pinned_memory_size 맵에 등록한다.
 */
// SST malloc done by vanadis, we just need to record the memory addr
cudaError_t CUDARTAPI cudaMallocHostSSTInternal(
    void *addr, size_t size, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // track pinned memory size allocated in the host so that same amount of
  // memory is also allocated in GPU.
  ctx->api->pinned_memory_size[addr] = size;
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * cudaMallocPitchInternal - 2D 텍스처/배열용 align된 글로벌 메모리 할당
 *
 * GPGPU-Sim은 현재 pitch 정렬을 수행하지 않고 width * height 크기만큼
 * gpu_malloc()으로 할당한 뒤 pitch = width를 반환한다.
 */
__host__ cudaError_t CUDARTAPI
/*
 * [한국어]
 * CUDA 런타임 API cudaMallocPitch()의 GPGPU-Sim 납부 구현
 */
cudaMallocPitchInternal(void **devPtr, size_t *pitch, size_t width,
                        size_t height, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  unsigned malloc_width_inbytes = width;
  printf("GPGPU-Sim PTX: cudaMallocPitch (width = %d)\n", malloc_width_inbytes);
  CUctx_st *context = GPGPUSim_Context(ctx);
  *devPtr = context->get_device()->get_gpgpu()->gpu_malloc(
      malloc_width_inbytes * height);
  pitch[0] = malloc_width_inbytes;
  if (*devPtr) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

/*
 * [한국어]
 * cudaHostGetDevicePointerInternal - 핀드 호스트 메모리에 대응하는 GPU 디바이스 포인터 획득
 *
 * cudaHostAlloc/cudaMallocHost로 기록된 크기만큼 GPU 메모리를 새로 할당하고,
 * 현재 CPU 메모리 내용을 GPU로 복사한다.
 */
cudaError_t cudaHostGetDevicePointerInternal(void **pDevice, void *pHost,
                                             unsigned int flags,
                                             gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // only cpu memory allocation happens in cudaHostAlloc. Linking with device
  // pointer to pinned memory happens here.
  // TODO: once kernel is executed, the contents in global pointer of GPU must
  // be copied back to CPU host pointer!
  flags = 0;
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  // [한국어] pinned_memory_size 맵에서 호스트 주소에 대응하는 크기 검색
  std::map<void *, size_t>::const_iterator i =
      ctx->api->pinned_memory_size.find(pHost);
  assert(i != ctx->api->pinned_memory_size.end());
  size_t size = i->second;
  *pDevice = gpu->gpu_malloc(size);
  if (g_debug_execution >= 3) {
    printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n",
           size, (unsigned long long)*pDevice);
    ctx->api->g_mallocPtr_Size[(unsigned long long)*pDevice] = size;
  }
  if (*pDevice) {
    ctx->api->pinned_memory[pHost] = pDevice;
    // Copy contents in cpu to gpu
    // [한국어] 핀드 메모리 초기 내용을 GPU 메모리로 복사
    gpu->memcpy_to_gpu((size_t)*pDevice, pHost, size);
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

/*
 * [한국어]
 * cudaMallocArrayInternal - CUDA 배열(cudaArray)용 메모리 할당
 *
 * cudaChannelFormatDesc에 명시된 채널 비트 수를 바탕으로 요소 크기를 계산하고,
 * gpu_mallocarray()로 GPU 메모리를 할당한다.
 */
__host__ cudaError_t CUDARTAPI cudaMallocArrayInternal(
    struct cudaArray **array, const struct cudaChannelFormatDesc *desc,
    size_t width, size_t height __dv(1), gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // [한국어] 채널 비트 수를 바이트로 환산하여 전체 배열 크기 산출
  unsigned size =
      width * height * ((desc->x + desc->y + desc->z + desc->w) / 8);
  CUctx_st *context = GPGPUSim_Context(ctx);
  (*array) = (struct cudaArray *)malloc(sizeof(struct cudaArray));
  (*array)->desc = *desc;
  (*array)->width = width;
  (*array)->height = height;
  (*array)->size = size;
  (*array)->dimensions = 2;
  ((*array)->devPtr32) =
      (int)(long long)context->get_device()->get_gpgpu()->gpu_mallocarray(size);
  printf("GPGPU-Sim PTX: cudaMallocArray: devPtr32 = %d\n",
         ((*array)->devPtr32));
  ((*array)->devPtr) = (void *)(long long)((*array)->devPtr32);
  if (((*array)->devPtr)) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

/*
 * [한국어]
 * cudaMemcpyInternal - GPU/호스트 간 메모리 복사
 *
 * @kind: cudaMemcpyHostToDevice, DeviceToHost, DeviceToDevice, Default.
 *        Default는 주소값(GLOBAL_HEAP_START 이상)으로 방향을 추론.
 *
 * 모든 복사는 stream_manager에 stream_operation으로 push되어 비동기 처리됨.
 */
__host__ cudaError_t CUDARTAPI
/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpy()의 GPGPU-Sim 납부 구현
 */
cudaMemcpyInternal(void *dst, const void *src, size_t count,
                   enum cudaMemcpyKind kind, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // CUctx_st *context = GPGPUSim_Context();
  // gpgpu_t *gpu = context->get_device()->get_gpgpu();
  if (g_debug_execution >= 3)
    printf("GPGPU-Sim PTX: cudaMemcpy(): devPtr = %p\n", dst);
  // [한국어] 복사 방향에 따라 stream_manager에 적절한 stream_operation 추가
  if (kind == cudaMemcpyHostToDevice)
    ctx->the_gpgpusim->g_stream_manager->push(
        stream_operation(src, (size_t)dst, count, 0));
  else if (kind == cudaMemcpyDeviceToHost)
    ctx->the_gpgpusim->g_stream_manager->push(
        stream_operation((size_t)src, dst, count, 0));
  else if (kind == cudaMemcpyDeviceToDevice)
    ctx->the_gpgpusim->g_stream_manager->push(
        stream_operation((size_t)src, (size_t)dst, count, 0));
  else if (kind == cudaMemcpyDefault) {
    // [한국어] 주소가 GLOBAL_HEAP_START 이상이면 GPU 메모리로 간주
    if ((size_t)src >= GLOBAL_HEAP_START) {
      if ((size_t)dst >= GLOBAL_HEAP_START)
        ctx->the_gpgpusim->g_stream_manager->push(stream_operation(
            (size_t)src, (size_t)dst, count, 0));  // device to device
      else
        ctx->the_gpgpusim->g_stream_manager->push(
            stream_operation((size_t)src, dst, count, 0));  // device to host
    } else {
      if ((size_t)dst >= GLOBAL_HEAP_START)
        ctx->the_gpgpusim->g_stream_manager->push(
            stream_operation(src, (size_t)dst, count, 0));
      else {
        printf(
            "GPGPU-Sim PTX: cudaMemcpy - ERROR : unsupported transfer: host to "
            "host\n");
        abort();
      }
    }
  } else {
    printf("GPGPU-Sim PTX: cudaMemcpy - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * cudaMemcpyToArrayInternal - cudaArray로 메모리 복사
 */
__host__ cudaError_t CUDARTAPI cudaMemcpyToArrayInternal(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t count, enum cudaMemcpyKind kind, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  size_t size = count;
  printf("GPGPU-Sim PTX: cudaMemcpyToArray\n");
  if (kind == cudaMemcpyHostToDevice)
    gpu->memcpy_to_gpu((size_t)(dst->devPtr), src, size);
  else if (kind == cudaMemcpyDeviceToHost)
    gpu->memcpy_from_gpu(dst->devPtr, (size_t)src, size);
  else if (kind == cudaMemcpyDeviceToDevice)
    gpu->memcpy_gpu_to_gpu((size_t)(dst->devPtr), (size_t)src, size);
  else {
    printf(
        "GPGPU-Sim PTX: cudaMemcpyToArray - ERROR : unsupported "
        "cudaMemcpyKind\n");
    abort();
  }
  dst->devPtr32 = (unsigned)(size_t)(dst->devPtr);
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * cudaMemcpy2DInternal - 2D 메모리 복사 (동일 pitch 가정)
 */
__host__ cudaError_t CUDARTAPI cudaMemcpy2DInternal(
    void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
    size_t height, enum cudaMemcpyKind kind, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  size_t size = spitch * height;
  gpgpusim_ptx_assert((dpitch == spitch),
                      "different src and dst pitch not supported yet");
  if (kind == cudaMemcpyHostToDevice)
    gpu->memcpy_to_gpu((size_t)dst, src, size);
  else if (kind == cudaMemcpyDeviceToHost)
    gpu->memcpy_from_gpu(dst, (size_t)src, size);
  else if (kind == cudaMemcpyDeviceToDevice)
    gpu->memcpy_gpu_to_gpu((size_t)dst, (size_t)src, size);
  else {
    printf(
        "GPGPU-Sim PTX: cudaMemcpy2D - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * cudaMemcpy2DToArrayInternal - 2D 배열로 메모리 복사
 *
 * 현재는 전체 배열 복사, wOffset/hOffset=0, spitch==width만 지원.
 */
__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayInternal(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  size_t size = spitch * height;
  size_t channel_size = dst->desc.w + dst->desc.x + dst->desc.y + dst->desc.z;
  gpgpusim_ptx_assert(
      ((channel_size % 8) == 0),
      "none byte multiple destination channel size not supported (sz=%u)",
      channel_size);
  unsigned elem_size = channel_size / 8;
  // [한국어] 아직 부분 복사나 1D/3D 배열, pitch 불일치는 지원하지 않음
  gpgpusim_ptx_assert((dst->dimensions == 2),
                      "copy to none 2D array not supported");
  gpgpusim_ptx_assert((wOffset == 0), "non-zero wOffset not yet supported");
  gpgpusim_ptx_assert((hOffset == 0), "non-zero hOffset not yet supported");
  gpgpusim_ptx_assert((dst->height == (int)height),
                      "partial copy not supported");
  gpgpusim_ptx_assert((elem_size * dst->width == width),
                      "partial copy not supported");
  gpgpusim_ptx_assert((spitch == width), "spitch != width not supported");
  if (kind == cudaMemcpyHostToDevice)
    gpu->memcpy_to_gpu((size_t)(dst->devPtr), src, size);
  else if (kind == cudaMemcpyDeviceToHost)
    gpu->memcpy_from_gpu(dst->devPtr, (size_t)src, size);
  else if (kind == cudaMemcpyDeviceToDevice)
    gpu->memcpy_gpu_to_gpu((size_t)dst->devPtr, (size_t)src, size);
  else {
    printf(
        "GPGPU-Sim PTX: cudaMemcpy2D - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }
  dst->devPtr32 = (unsigned)(size_t)(dst->devPtr);
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * cudaMemcpyToSymbolInternal - PTX 심볼(전역/상수 변수)에 데이터 복사
 *
 * 복사는 stream_manager의 symbol stream_operation으로 push되어 비동기 수행.
 */
__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolInternal(
    const char *symbol, const void *src, size_t count, size_t offset __dv(0),
    enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice),
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // CUctx_st *context = GPGPUSim_Context();
  assert(kind == cudaMemcpyHostToDevice);
  printf("GPGPU-Sim PTX: cudaMemcpyToSymbol: symbol = %p\n", symbol);
  // stream_operation( const char *symbol, const void *src, size_t count, size_t
  // offset )
  // [한국어] 심볼 이름 기반 비동기 복사 operation push
  ctx->the_gpgpusim->g_stream_manager->push(
      stream_operation(src, symbol, count, offset, 0));
  // gpgpu_ptx_sim_memcpy_symbol(symbol,src,count,offset,1,context->get_device()->get_gpgpu());
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpyFromSymbol()의 GPGPU-Sim 납부 구현
 */
__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolInternal(
    void *dst, const char *symbol, size_t count, size_t offset __dv(0),
    enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost),
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // CUctx_st *context = GPGPUSim_Context();
  assert(kind == cudaMemcpyDeviceToHost);
  printf("GPGPU-Sim PTX: cudaMemcpyFromSymbol: symbol = %p\n", symbol);
  ctx->the_gpgpusim->g_stream_manager->push(
      stream_operation(symbol, dst, count, offset, 0));
  // gpgpu_ptx_sim_memcpy_symbol(symbol,dst,count,offset,0,context->get_device()->get_gpgpu());
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpyAsync()의 GPGPU-Sim 납부 구현
 */
__host__ cudaError_t CUDARTAPI cudaMemcpyAsyncInternal(
    void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
    cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  struct CUstream_st *s = (struct CUstream_st *)stream;
  switch (kind) {
    case cudaMemcpyHostToDevice:
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation(src, (size_t)dst, count, s));
      break;
    case cudaMemcpyDeviceToHost:
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation((size_t)src, dst, count, s));
      break;
    case cudaMemcpyDeviceToDevice:
      ctx->the_gpgpusim->g_stream_manager->push(
          stream_operation((size_t)src, (size_t)dst, count, s));
      break;
    default:
      abort();
  }
  return g_last_cudaError = cudaSuccess;
}

#if (CUDART_VERSION >= 8000)
cudaError_t CUDARTAPI
/*
 * [한국어]
 * CUDA 런타임 API cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags()의 GPGPU-Sim 납부 구현 (GPGPU-Sim에서 현재 미구현/미지원)
 */
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsInternal(
    int *numBlocks, const char *hostFunc, int blockSize, size_t dynamicSMemSize,
    unsigned int flags, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  printf(
      "GPGPU-Sim PTX: cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags "
      "%p\n",
      hostFunc);
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFunc);
  printf(
      "Calculate Maxium Active Block with function ptr=%p, blockSize=%d, "
      "SMemSize=%lu\n",
      hostFunc, blockSize, dynamicSMemSize);
  if (flags == cudaOccupancyDefault) {
    // create kernel_info based on entry
    dim3 gridDim(context->get_device()->get_gpgpu()->max_cta_per_core() *
                 context->get_device()->get_gpgpu()->get_config().num_shader());
    dim3 blockDim(blockSize);
    // because this fuction is only checking for resource requirements, we do
    // not care which stream this kernel runs at, just picked -1
    kernel_info_t result(gridDim, blockDim, entry, -1);
    // if(entry == NULL){
    //	*numBlocks = 1;
    //	return g_last_cudaError = cudaErrorUnknown;
    //}
    *numBlocks = context->get_device()->get_gpgpu()->get_max_cta(result);
    printf("Maximum size is %d with gridDim %d and blockDim %d\n", *numBlocks,
           gridDim.x, blockDim.x);
    return g_last_cudaError = cudaSuccess;
  } else {
    cuda_not_implemented(__my_func__, __LINE__);
    return g_last_cudaError = cudaErrorUnknown;
  }
}

#endif
/*
 * [한국어]
 * CUDA 런타임 API cudaMemset()의 GPGPU-Sim 납부 구현
 */
__host__ cudaError_t CUDARTAPI cudaMemsetInternal(
    void *mem, int c, size_t count, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  gpu->gpu_memset((size_t)mem, c, count);
  return g_last_cudaError = cudaSuccess;
}

// memset operation is done but i think its not async?
__host__ cudaError_t CUDARTAPI
/*
 * [한국어]
 * CUDA 런타임 API cudaMemsetAsync()의 GPGPU-Sim 납부 구현
 */
cudaMemsetAsyncInternal(void *mem, int c, size_t count, cudaStream_t stream = 0,
                        gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: WARNING: Asynchronous memset not supported (%s)\n",
         __my_func__);
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  gpu->gpu_memset((size_t)mem, c, count);
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaGLMapBufferObject()의 GPGPU-Sim 납부 구현
 */
cudaError_t cudaGLMapBufferObjectInternal(void **devPtr, GLuint bufferObj,
                                          gpgpu_context *gpgpu_ctx = NULL) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#ifdef OPENGL_SUPPORT
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  GLint buffer_size = 0;
  CUctx_st *context = GPGPUSim_Context(ctx);

  glbmap_entry_t *p = ctx->api->g_glbmap;
  while (p && p->m_bufferObj != bufferObj) p = p->m_next;
  if (p == NULL) {
    glBindBuffer(GL_ARRAY_BUFFER, bufferObj);
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &buffer_size);
    assert(buffer_size != 0);
    *devPtr = context->get_device()->get_gpgpu()->gpu_malloc(buffer_size);

    // create entry and insert to front of list
    glbmap_entry_t *n = (glbmap_entry_t *)calloc(1, sizeof(glbmap_entry_t));
    n->m_next = ctx->api->g_glbmap;
    ctx->api->g_glbmap = n;

    // initialize entry
    n->m_bufferObj = bufferObj;
    n->m_devPtr = *devPtr;
    n->m_size = buffer_size;

    p = n;
  } else {
    buffer_size = p->m_size;
    *devPtr = p->m_devPtr;
  }

  if (*devPtr) {
    char *data = (char *)calloc(p->m_size, 1);
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, buffer_size, data);
    memcpy_to_gpu((size_t)*devPtr, data, buffer_size);
    free(data);
    printf(
        "GPGPU-Sim PTX: cudaGLMapBufferObject %zu bytes starting at 0x%llx..\n",
        (size_t)buffer_size, (unsigned long long)*devPtr);
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }

  return g_last_cudaError = cudaSuccess;
#else
  fflush(stdout);
  fflush(stderr);
  printf(
      "GPGPU-Sim PTX: GPGPU-Sim support for OpenGL integration disabled -- "
      "exiting\n");
  fflush(stdout);
  exit(50);
#endif
}

#if CUDART_VERSION >= 6050
/*
 * [한국어]
 * CUDA 런타임 API cuLinkAddFile()의 GPGPU-Sim 납부 구현
 */
CUresult cuLinkAddFileInternal(CUlinkState state, CUjitInputType type,
                               const char *path, unsigned int numOptions,
                               CUjit_option *options, void **optionValues,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  static bool addedFile = false;
  if (addedFile) {
    printf(
        "GPGPU-Sim PTX: ERROR: cuLinkAddFile does not support multiple "
        "files\n");
    abort();
  }

  // blocking
  assert(type == CU_JIT_INPUT_PTX);
  CUctx_st *context = GPGPUSim_Context(ctx);
  char *file = getenv("PTX_JIT_PATH");
  if (file == NULL) {
    printf("GPGPU-Sim PTX: ERROR: PTX_JIT_PATH has not been set\n");
    abort();
  }
  strcat(file, "/");
  strcat(file, path);
  symbol_table *symtab = ctx->gpgpu_ptx_sim_load_ptx_from_filename(file);
  std::string fname(path);
  ctx->api->name_symtab[fname] = symtab;
  context->add_binary(symtab, 1);
  ctx->api->load_static_globals(symtab, STATIC_ALLOC_LIMIT, 0xFFFFFFFF,
                                context->get_device()->get_gpgpu());
  ctx->api->load_constants(symtab, STATIC_ALLOC_LIMIT,
                           context->get_device()->get_gpgpu());
  addedFile = true;
  return CUDA_SUCCESS;
}
#endif

#if (CUDART_VERSION >= 2010)
/*
 * [한국어]
 * CUDA 런타임 API cudaHostAlloc()의 GPGPU-Sim 납부 구현
 */
cudaError_t cudaHostAllocInternal(void **pHost, size_t bytes,
                                  unsigned int flags,
                                  gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *pHost = malloc(bytes);
  // need to track the size allocated so that cudaHostGetDevicePointer() can
  // function properly.
  // TODO: vary this function behavior based on flags value (following nvidia
  // documentation)
  ctx->api->pinned_memory_size[*pHost] = bytes;
  if (*pHost)
    return g_last_cudaError = cudaSuccess;
  else
    return g_last_cudaError = cudaErrorMemoryAllocation;
}

#endif
/*
 * [한국어]
 * 커널 속성과 장치 속성으로 블록당 최대 스레드 계산
 */
size_t getMaxThreadsPerBlock(struct cudaFuncAttributes *attr,
                             gpgpu_context *ctx) {
  _cuda_device_id *dev = ctx->GPGPUSim_Init();
  struct cudaDeviceProp prop;

  prop = *dev->get_prop();

  size_t max = prop.maxThreadsPerBlock;

  if (attr->numRegs && (prop.regsPerBlock / attr->numRegs) < max)
    max = prop.regsPerBlock / attr->numRegs;

  if (attr->sharedSizeBytes &&
      (prop.sharedMemPerBlock / attr->sharedSizeBytes) < max)
    max = prop.sharedMemPerBlock / attr->sharedSizeBytes;

  return max;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaFuncGetAttributes()의 GPGPU-Sim 납부 구현
 */
cudaError_t CUDARTAPI cudaFuncGetAttributesInternal(
    struct cudaFuncAttributes *attr, const char *hostFun,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFun);
  if (entry) {
    const struct gpgpu_ptx_sim_info *kinfo = entry->get_kernel_info();
    attr->sharedSizeBytes = kinfo->smem;
    attr->constSizeBytes = kinfo->cmem;
    attr->localSizeBytes = kinfo->lmem;
    attr->numRegs = kinfo->regs;
    if (kinfo->maxthreads > 0)
      attr->maxThreadsPerBlock = kinfo->maxthreads;
    else
      attr->maxThreadsPerBlock = getMaxThreadsPerBlock(attr, ctx);
#if CUDART_VERSION >= 3000
    attr->ptxVersion = kinfo->ptx_version;
    attr->binaryVersion = kinfo->sm_target;
#endif
  }
  return g_last_cudaError = cudaSuccess;
}

#if (CUDART_VERSION > 5000)
__host__ cudaError_t CUDARTAPI
/*
 * [한국어]
 * CUDA 런타임 API cudaDeviceGetAttribute()의 GPGPU-Sim 납부 구현
 */
cudaDeviceGetAttributeInternal(int *value, enum cudaDeviceAttr attr, int device,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  const struct cudaDeviceProp *prop;
  _cuda_device_id *dev = ctx->GPGPUSim_Init();

  if (device <= dev->num_devices()) {
    prop = dev->get_prop();
    switch (attr) {
      case 1:
        *value = prop->maxThreadsPerBlock;
        break;
      case 2:
        *value = prop->maxThreadsDim[0];
        break;
      case 3:
        *value = prop->maxThreadsDim[1];
        break;
      case 4:
        *value = prop->maxThreadsDim[2];
        break;
      case 5:
        *value = prop->maxGridSize[0];
        break;
      case 6:
        *value = prop->maxGridSize[1];
        break;
      case 7:
        *value = prop->maxGridSize[2];
        break;
      case 8:
        *value = prop->sharedMemPerBlock;
        break;
      case 9:
        *value = prop->totalConstMem;
        break;
      case 10:
        *value = prop->warpSize;
        break;
      case 11:
        *value = 16;  // dummy value
        break;
      case 12:
        *value = prop->regsPerBlock;
        break;
      case 13:
        *value = 1480000;  // for 1080ti
        break;
      case 14:
        *value = prop->textureAlignment;
        break;
      case 15:
        *value = 0;
        break;
      case 16:
        *value = prop->multiProcessorCount;
        break;
      case 17:
      case 18:
      case 19:
        *value = 0;
        break;
      case 21:
      case 22:
      case 23:
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 42:
      case 45:
      case 46:
      case 47:
      case 48:
      case 49:
      case 52:
      case 53:
      case 55:
      case 56:
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
      case 62:
      case 63:
      case 64:
      case 66:
      case 67:
      case 69:
      case 70:
      case 71:
      case 73:
      case 74:
      case 77:
        *value = 1000;  // dummy value
        break;
      case 29:
      case 43:
      case 54:
      case 65:
      case 68:
      case 72:
        *value = 10;  // dummy value
        break;
      case 30:
      case 51:
        *value = 128;  // dummy value
        break;
      case 31:
        *value = 1;
        break;
      case 32:
        *value = 0;
        break;
      case 33:
      case 50:
        *value = 0;  // dummy value
        break;
      case 34:
        *value = 0;
        break;
      case 35:
        *value = 0;
        break;
      case 36:
        *value = 1250000;  // CK value for 1080ti
        break;
      case 37:
        *value = 352;  // value for 1080ti
        break;
      case 38:
        *value = 3000000;  // value for 1080ti
        break;
      case 39:
        *value = dev->get_gpgpu()->threads_per_core();
        break;
      case 40:
        *value = 0;
        break;
      case 41:
        *value = 0;
        break;
      case 75:  // cudaDevAttrComputeCapabilityMajor
        *value = prop->major;
        break;
      case 76:  // cudaDevAttrComputeCapabilityMinor
        *value = prop->minor;
        break;
      case 78:
        *value = 0;  // TODO: as of now, we dont support stream priorities.
        break;
      case 79:
        *value = 0;
        break;
      case 80:
        *value = 0;
        break;
#if (CUDART_VERSION > 5050)
      case 81:
        *value = prop->sharedMemPerMultiprocessor;
        break;
      case 82:
        *value = prop->regsPerMultiprocessor;
        break;
#endif
      case 83:
      case 84:
      case 85:
      case 86:
        *value = 0;
        break;
      case 87:
        *value = 4;  // dummy value
        break;
      case 88:
      case 89:
        *value = 0;
        break;
      default:
        printf("ERROR: Attribute number %d unimplemented \n", attr);
        abort();
    }
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorInvalidDevice;
  }
}
#endif
/*
 * [한국어]
 * CUDA 런타임 API cudaBindTexture()의 GPGPU-Sim 납부 구현
 */
__host__ cudaError_t CUDARTAPI cudaBindTextureInternal(
    size_t *offset, const struct textureReference *texref, const void *devPtr,
    const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX),
    gpgpu_context *gpgpu_ctx = NULL) {
#if (CUDART_VERSION <= 1200)
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf(
      "GPGPU-Sim PTX: in cudaBindTexture: sizeof(struct textureReference) = "
      "%zu\n",
      sizeof(struct textureReference));
  struct cudaArray *array;
  array = (struct cudaArray *)malloc(sizeof(struct cudaArray));
  array->desc = *desc;
  array->size = size;
  array->width = size;
  array->height = 1;
  array->dimensions = 1;
  array->devPtr = (void *)devPtr;
  array->devPtr32 = (int)(long long)devPtr;
  offset = 0;
  printf("GPGPU-Sim PTX:   size = %zu\n", size);
  printf("GPGPU-Sim PTX:   texref = %p, array = %p\n", texref, array);
  printf("GPGPU-Sim PTX:   devPtr32 = %x\n", array->devPtr32);
  printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n",
         gpu->gpgpu_ptx_sim_findNamefromTexture(texref));
  printf("GPGPU-Sim PTX:   ChannelFormatDesc: x=%d, y=%d, z=%d, w=%d\n",
         desc->x, desc->y, desc->z, desc->w);
  printf("GPGPU-Sim PTX:   Texture Normalized? = %d\n", texref->normalized);
  gpu->gpgpu_ptx_sim_bindTextureToArray(texref, array);
  devPtr = (void *)(long long)array->devPtr32;
  printf("GPGPU-Sim PTX: devPtr = %p\n", devPtr);
#endif
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaBindTextureToArray()의 GPGPU-Sim 납부 구현
 */
__host__ cudaError_t CUDARTAPI cudaBindTextureToArrayInternal(
    const struct textureReference *texref, const struct cudaArray *array,
    const struct cudaChannelFormatDesc *desc, gpgpu_context *gpgpu_ctx = NULL) {
#if (CUDART_VERSION <= 1200)
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf("GPGPU-Sim PTX: in cudaBindTextureToArray: %p %p\n", texref, array);
  printf("GPGPU-Sim PTX:   devPtr32 = %x\n", array->devPtr32);
  printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n",
         gpu->gpgpu_ptx_sim_findNamefromTexture(texref));
  printf("GPGPU-Sim PTX:   Texture Normalized? = %d\n", texref->normalized);
  gpu->gpgpu_ptx_sim_bindTextureToArray(texref, array);
#endif
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaUnbindTexture()의 GPGPU-Sim 납부 구현
 */
__host__ cudaError_t CUDARTAPI cudaUnbindTextureInternal(
    const struct textureReference *texref, gpgpu_context *gpgpu_ctx = NULL) {
#if (CUDART_VERSION <= 1200)
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf(
      "GPGPU-Sim PTX: in cudaUnbindTexture: sizeof(struct textureReference) = "
      "%zu\n",
      sizeof(struct textureReference));
  printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n",
         gpu->gpgpu_ptx_sim_findNamefromTexture(texref));

  gpu->gpgpu_ptx_sim_unbindTexture(texref);
#endif
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaLaunchKernel()의 GPGPU-Sim 납부 구현 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaLaunchKernelInternal(
    const char *hostFun, dim3 gridDim, dim3 blockDim, const void **args,
    size_t sharedMem, cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }

  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFun);
#if CUDART_VERSION < 10000
  cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream, ctx);
#endif
  for (unsigned i = 0; i < entry->num_args(); i++) {
    std::pair<size_t, unsigned> p = entry->get_param_config(i);
    cudaSetupArgumentInternal(args[i], p.first, p.second);
  }

  cudaLaunchInternal(hostFun);
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaStreamCreate()의 GPGPU-Sim 납부 구현
 */
__host__ cudaError_t CUDARTAPI cudaStreamCreateInternal(
    cudaStream_t *stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: cudaStreamCreate\n");
#if (CUDART_VERSION >= 3000)
  *stream = new struct CUstream_st();
  ctx->the_gpgpusim->g_stream_manager->add_stream(*stream);
#else
  *stream = 0;
  printf(
      "GPGPU-Sim PTX: WARNING: Asynchronous kernel execution not supported "
      "(%s)\n",
      __my_func__);
#endif
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaStreamDestroy()의 GPGPU-Sim 납부 구현
 */
__host__ cudaError_t CUDARTAPI cudaStreamDestroyInternal(
    cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#if (CUDART_VERSION >= 3000)
  // per-stream synchronization required for application using external
  // libraries without explicit synchronization in the code to avoid the
  // stream_manager from spinning forever to destroy non-empty streams without
  // making any forward progress.
  stream->synchronize();
  ctx->the_gpgpusim->g_stream_manager->destroy_stream(stream);
#endif
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaStreamSynchronize()의 GPGPU-Sim 납부 구현
 */
__host__ cudaError_t CUDARTAPI cudaStreamSynchronizeInternal(
    cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#if (CUDART_VERSION >= 3000)
  if (stream == NULL) ctx->synchronize();
  return g_last_cudaError = cudaSuccess;
  stream->synchronize();
#else
  printf(
      "GPGPU-Sim PTX: WARNING: Asynchronous kernel execution not supported "
      "(%s)\n",
      __my_func__);
#endif
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * CUDA 런타임 API __cudaRegisterTexture()의 GPGPU-Sim 납부 구현
 */
void __cudaRegisterTextureInternal(
    void **fatCubinHandle, const struct textureReference *hostVar,
    const void **deviceAddress, const char *deviceName, int dim, int norm,
    int ext,
    gpgpu_context *gpgpu_ctx =
        NULL)  // passes in a newly created textureReference
{
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  std::string devStr(deviceName);
#if (CUDART_VERSION > 4020)
  if (devStr.size() > 2 && devStr.data()[0] == ':' && devStr.data()[1] == ':')
    devStr = devStr.replace(0, 2, "");
#endif
  CUctx_st *context = GPGPUSim_Context(ctx);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf("GPGPU-Sim PTX: in __cudaRegisterTexture:\n");
  gpu->gpgpu_ptx_sim_bindNameToTexture(devStr.data(), hostVar, dim, norm, ext);
  printf("GPGPU-Sim PTX:   int dim = %d\n", dim);
  printf("GPGPU-Sim PTX:   int norm = %d\n", norm);
  printf("GPGPU-Sim PTX:   int ext = %d\n", ext);
  printf(
      "GPGPU-Sim PTX:   Execution warning: Not finished implementing \"%s\"\n",
      __my_func__);
}
/*
 * [한국어]
 * CUDA 런타임 API cudaGLUnmapBufferObject()의 GPGPU-Sim 납부 구현
 */
cudaError_t cudaGLUnmapBufferObjectInternal(GLuint bufferObj,
                                            gpgpu_context *gpgpu_ctx = NULL) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#ifdef OPENGL_SUPPORT
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  CUctx_st *ctx = GPGPUSim_Context(ctx);
  glbmap_entry_t *p = ctx->api->g_glbmap;
  while (p && p->m_bufferObj != bufferObj) p = p->m_next;
  if (p == NULL) return g_last_cudaError = cudaErrorUnknown;

  char *data = (char *)calloc(p->m_size, 1);
  memcpy_from_gpu(data, (size_t)p->m_devPtr, p->m_size);
  glBufferSubData(GL_ARRAY_BUFFER, 0, p->m_size, data);
  free(data);

  return g_last_cudaError = cudaSuccess;
#else
  fflush(stdout);
  fflush(stderr);
  printf("GPGPU-Sim PTX: support for OpenGL integration disabled -- exiting\n");
  fflush(stdout);
  exit(50);
#endif
}

#if CUDART_VERSION >= 3000

__host__ cudaError_t CUDARTAPI
/*
 * [한국어]
 * CUDA 런타임 API cudaFuncSetCacheConfig()의 GPGPU-Sim 납부 구현
 */
cudaFuncSetCacheConfigInternal(const char *func, enum cudaFuncCache cacheConfig,
                               gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(ctx);
  context->get_device()->get_gpgpu()->set_cache_config(
      context->get_kernel(func)->get_name(), (FuncCache)cacheConfig);
  return g_last_cudaError = cudaSuccess;
}

#endif

#if CUDART_VERSION >= 4000
/*
 * [한국어]
 * CUDA 런타임 API cuLaunchKernel()의 GPGPU-Sim 납부 구현 — 실제 동작은 *Internal() 납부 함수로 전달
 */
CUresult CUDAAPI cuLaunchKernelInternal(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (extra != NULL) {
    printf(
        "GPGPU-Sim CUDA DRIVER API: ERROR: Currently do not support void** "
        "extra.\n");
    abort();
  }
  const char *hostFun = (const char *)f;
  CUctx_st *context = GPGPUSim_Context(ctx);
  function_info *entry = context->get_kernel(hostFun);
  cudaConfigureCallInternal(dim3(gridDimX, gridDimY, gridDimZ),
                            dim3(blockDimX, blockDimY, blockDimZ),
                            sharedMemBytes, (cudaStream_t)hStream, ctx);
  for (unsigned i = 0; i < entry->num_args(); i++) {
    std::pair<size_t, unsigned> p = entry->get_param_config(i);
    cudaSetupArgumentInternal(kernelParams[i], p.first, p.second, ctx);
  }
  cudaLaunchInternal(hostFun, ctx);
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

CUevent_st *get_event(cudaEvent_t event) {
  unsigned event_uid;
#if CUDART_VERSION >= 3000
  event_uid = event->get_uid();
#else
  event_uid = event;
#endif
  event_tracker_t::iterator e = g_timer_events.find(event_uid);
  if (e == g_timer_events.end()) return NULL;
  return e->second;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaEventRecord()의 GPGPU-Sim 납부 구현
 */
__host__ cudaError_t CUDARTAPI cudaEventRecordInternal(
    cudaEvent_t event, cudaStream_t stream, gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = get_event(event);
  if (!e) return g_last_cudaError = cudaErrorUnknown;
  struct CUstream_st *s = (struct CUstream_st *)stream;
  stream_operation op(e, s);
  e->issue();
  ctx->the_gpgpusim->g_stream_manager->push(op);
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaStreamWaitEvent()의 GPGPU-Sim 납부 구현
 */
__host__ cudaError_t CUDARTAPI cudaStreamWaitEventInternal(
    cudaStream_t stream, cudaEvent_t event, unsigned int flags,
    gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // reference:
  // https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__STREAM_gfe68d207dc965685d92d3f03d77b0876.html
  CUevent_st *e = get_event(event);
  if (!e) {
    printf(
        "GPGPU-Sim API: Error at cudaStreamWaitEvent. Event is not created "
        ".\n");
    return g_last_cudaError = cudaErrorInvalidResourceHandle;
  } else if (e->num_issued() == 0) {
    printf(
        "GPGPU-Sim API: Warning: cudaEventRecord has not been called on event "
        "before calling cudaStreamWaitEvent.\nNothin    g to be done.\n");
    return g_last_cudaError = cudaSuccess;
  }
  if (!stream) {
    ctx->the_gpgpusim->g_stream_manager->pushCudaStreamWaitEventToAllStreams(
        e, flags);
  } else {
    struct CUstream_st *s = (struct CUstream_st *)stream;
    stream_operation op(s, e, flags);
    ctx->the_gpgpusim->g_stream_manager->push(op);
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaThreadExitInternal(gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  ctx->exit_simulation();
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI
cudaThreadSynchronizeInternal(gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // Called on host side
  ctx->synchronize();
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI
cudaDeviceSynchronizeInternal(gpgpu_context *gpgpu_ctx = NULL) {
  gpgpu_context *ctx;
  if (gpgpu_ctx) {
    ctx = gpgpu_ctx;
  } else {
    ctx = GPGPU_Context();
  }
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // Blocks until the device has completed all preceding requested tasks
  ctx->synchronize();
  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

/*******************************************************************************
 *                                                                              *
 *   SST Specific functions, used by Balar *
 *                                                                              *
 *******************************************************************************/

/**
 * @brief Custom function to get CUDA function parameter size and offset
 *        from PTX parsing result
 *
 * @param hostFun
 * @param index
 * @return std::tuple<cudaError_t, size_t, unsigned>
 */
std::tuple<cudaError_t, size_t, unsigned> SST_cudaGetParamConfig(
    uint64_t hostFun, unsigned index) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUctx_st *context = GPGPUSim_Context(GPGPU_Context());
  function_info *entry = context->get_kernel((char *)hostFun);
  cudaError_t result = cudaSuccess;
  size_t size = 0;
  unsigned alignment = 0;
  if (index >= entry->num_args()) {
    result = cudaErrorAssert;
  } else {
    std::pair<size_t, unsigned> p = entry->get_param_config(index);
    size = p.first;
    alignment = p.second;
  }
  return std::tuple<cudaError_t, size_t, unsigned>(result, size, alignment);
}

extern "C" {
void SST_receive_mem_reply(unsigned core_id, void *mem_req) {
  CUctx_st *context = GPGPUSim_Context(GPGPU_Context());
  static_cast<sst_gpgpu_sim *>(context->get_device()->get_gpgpu())
      ->SST_receive_mem_reply(core_id, mem_req);
  // printf("GPGPU-sim: Recived Request\n");
}

/* [한국어] SST GPU 코어 1사이클 진행 래퍼 */
bool SST_gpu_core_cycle() { return SST_Cycle(); }

void SST_gpgpusim_numcores_equal_check(unsigned sst_numcores) {
  CUctx_st *context = GPGPUSim_Context(GPGPU_Context());
  static_cast<sst_gpgpu_sim *>(context->get_device()->get_gpgpu())
      ->SST_gpgpusim_numcores_equal_check(sst_numcores);
}
/*
 * [한국어]
 * SST 모드에서 GPU 메모리 할당 주소 기록
 */
uint64_t cudaMallocSST(void **devPtr, size_t size) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  void *test_malloc;
  test_malloc = (void *)malloc(size);
  void **test_malloc2 = &test_malloc;
  CUctx_st *context = GPGPUSim_Context(GPGPU_Context());
  *test_malloc2 = context->get_device()->get_gpgpu()->gpu_malloc(size);
  printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n", size,
         (unsigned long long)*test_malloc2);
  if (g_debug_execution >= 3)
    printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n",
           size, (unsigned long long)*test_malloc2);
  return (uint64_t)*test_malloc2;
}

__host__ cudaError_t CUDARTAPI cudaMallocHostSST(void *addr, size_t size) {
  return cudaMallocHostSSTInternal(addr, size);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaPeekAtLastError()의 GPGPU-Sim 구현. 납부 구현(cudaPeekAtLastErrorInternal)에 위임
 */
cudaError_t cudaPeekAtLastError(void) { return g_last_cudaError; }

__host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
  return cudaMallocInternal(devPtr, size);
}

__host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
  return cudaMallocHostInternal(ptr, size);
}
/*
 * [한국어]
 * CUDA 런타임 API cudaMallocPitch()의 GPGPU-Sim 구현. 납부 구현(cudaMallocPitchInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch,
                                               size_t width, size_t height) {
  return cudaMallocPitchInternal(devPtr, pitch, width, height);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMallocArray()의 GPGPU-Sim 구현. 납부 구현(cudaMallocArrayInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaMallocArray(
    struct cudaArray **array, const struct cudaChannelFormatDesc *desc,
    size_t width, size_t height __dv(1)) {
  return cudaMallocArrayInternal(array, desc, width, height);
}
/*
 * [한국어]
 * CUDA 런타임 API cudaFree()의 GPGPU-Sim 구현
 */
__host__ cudaError_t CUDARTAPI cudaFree(void *devPtr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // TODO...  manage g_global_mem space?
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaFreeHost()의 GPGPU-Sim 구현
 */
__host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  free(ptr);  // this will crash the system if called twice
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaFreeArray()의 GPGPU-Sim 구현
 */
__host__ cudaError_t CUDARTAPI cudaFreeArray(struct cudaArray *array) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // TODO...  manage g_global_mem space?
  return g_last_cudaError = cudaSuccess;
};

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpy()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpyInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src,
                                          size_t count,
                                          enum cudaMemcpyKind kind) {
  return cudaMemcpyInternal(dst, src, count, kind);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpyToArray()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpyToArrayInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaMemcpyToArray(struct cudaArray *dst,
                                                 size_t wOffset, size_t hOffset,
                                                 const void *src, size_t count,
                                                 enum cudaMemcpyKind kind) {
  return cudaMemcpyToArrayInternal(dst, wOffset, hOffset, src, count, kind);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpyFromArray()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpyFromArrayInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst,
                                                   const struct cudaArray *src,
                                                   size_t wOffset,
                                                   size_t hOffset, size_t count,
                                                   enum cudaMemcpyKind kind) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpyArrayToArray()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpyArrayToArrayInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(
    struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst,
    const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc,
    size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpy2D()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpy2DInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch,
                                            const void *src, size_t spitch,
                                            size_t width, size_t height,
                                            enum cudaMemcpyKind kind) {
  return cudaMemcpy2DInternal(dst, dpitch, src, spitch, width, height, kind);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpy2DToArray()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpy2DToArrayInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
  return cudaMemcpy2DToArrayInternal(dst, wOffset, hOffset, src, spitch, width,
                                     height, kind);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpy2DFromArray()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpy2DFromArrayInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(
    void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset,
    size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpy2DArrayToArray()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpy2DArrayToArrayInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(
    struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst,
    const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc,
    size_t width, size_t height,
    enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpyToSymbol()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpyToSymbolInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(
    const char *symbol, const void *src, size_t count, size_t offset __dv(0),
    enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice)) {
  return cudaMemcpyToSymbolInternal(symbol, src, count, offset, kind);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpyFromSymbol()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpyFromSymbolInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(
    void *dst, const char *symbol, size_t count, size_t offset __dv(0),
    enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost)) {
  return cudaMemcpyFromSymbolInternal(dst, symbol, count, offset, kind);
}
/*
 * [한국어]
 * CUDA 런타임 API cudaMemGetInfo()의 GPGPU-Sim 구현
 */
__host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // placeholder; should interact with cudaMalloc and cudaFree?
  *free = 10000000000;
  *total = 10000000000;

  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpyAsync()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpyAsyncInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src,
                                               size_t count,
                                               enum cudaMemcpyKind kind,
                                               cudaStream_t stream) {
  return cudaMemcpyAsyncInternal(dst, src, count, kind, stream);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpyToArrayAsync()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpyToArrayAsyncInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpyFromArrayAsync()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpyFromArrayAsyncInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(
    void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset,
    size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpy2DAsync()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpy2DAsyncInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch,
                                                 const void *src, size_t spitch,
                                                 size_t width, size_t height,
                                                 enum cudaMemcpyKind kind,
                                                 cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpy2DToArrayAsync()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpy2DToArrayAsyncInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(
    struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
    size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind,
    cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemcpy2DFromArrayAsync()의 GPGPU-Sim 구현. 납부 구현(cudaMemcpy2DFromArrayAsyncInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(
    void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset,
    size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind,
    cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

#if (CUDART_VERSION >= 8000)
/*
 * [한국어]
 * CUDA 런타임 API cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags()의 GPGPU-Sim 구현. 납부 구현(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const char *hostFunc, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlagsInternal(
      numBlocks, hostFunc, blockSize, dynamicSMemSize, flags);
}

#endif

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMemset(void *mem, int c, size_t count) {
  return cudaMemsetInternal(mem, c, count);
}

// memset operation is done but i think its not async?
/*
 * [한국어]
 * CUDA 런타임 API cudaMemsetAsync()의 GPGPU-Sim 구현. 납부 구현(cudaMemsetAsyncInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaMemsetAsync(void *mem, int c, size_t count,
                                               cudaStream_t stream = 0) {
  return cudaMemsetAsyncInternal(mem, c, count, stream = 0);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaMemset2D()의 GPGPU-Sim 구현. 납부 구현(cudaMemset2DInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaMemset2D(void *mem, size_t pitch, int c,
                                            size_t width, size_t height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

/*
 * [한국어]
 * CUDA 런타임 API cudaGetSymbolAddress()의 GPGPU-Sim 구현. 납부 구현(cudaGetSymbolAddressInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr,
                                                    const char *symbol) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaGetSymbolSize()의 GPGPU-Sim 구현. 납부 구현(cudaGetSymbolSizeInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size,
                                                 const char *symbol) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
__host__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
  return cudaGetDeviceCountInternal(count);
}

__host__ cudaError_t CUDARTAPI
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
  return cudaGetDevicePropertiesInternal(prop, device);
}

#if (CUDART_VERSION > 5000)
/*
 * [한국어]
 * CUDA 런타임 API cudaDeviceGetAttribute()의 GPGPU-Sim 구현. 납부 구현(cudaDeviceGetAttributeInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value,
                                                      enum cudaDeviceAttr attr,
                                                      int device) {
  return cudaDeviceGetAttributeInternal(value, attr, device);
}
#endif

__host__ cudaError_t CUDARTAPI
cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
  return cudaChooseDeviceInternal(device, prop);
}

__host__ cudaError_t CUDARTAPI cudaSetDevice(int device) {
  return cudaSetDeviceInternal(device);
}

__host__ cudaError_t CUDARTAPI cudaGetDevice(int *device) {
  return cudaGetDeviceInternal(device);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaDeviceGetLimit()의 GPGPU-Sim 구현. 납부 구현(cudaDeviceGetLimitInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue,
                                                  cudaLimit limit) {
  return cudaDeviceGetLimitInternal(pValue, limit);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaStreamGetPriority()의 GPGPU-Sim 구현. 납부 구현(cudaStreamGetPriorityInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t hStream,
                                                     int *priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaDeviceGetPCIBusId()의 GPGPU-Sim 구현. 납부 구현(cudaDeviceGetPCIBusIdInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len,
                                                     int device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaIpcGetMemHandle()의 GPGPU-Sim 구현. 납부 구현(cudaIpcGetMemHandleInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle,
                                                   void *devPtr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaIpcOpenMemHandle()의 GPGPU-Sim 구현. 납부 구현(cudaIpcOpenMemHandleInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t cudaIpcOpenMemHandle(void **devPtr,
                                          cudaIpcMemHandle_t handle,
                                          unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI
/*
 * [한국어]
 * CUDA 런타임 API cudaDestroyTextureObject()의 GPGPU-Sim 구현. 납부 구현(cudaDestroyTextureObjectInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
cudaDestroyTextureObject(cudaTextureObject_t texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

/*
 * [한국어]
 * CUDA 런타임 API cudaBindTexture()의 GPGPU-Sim 구현. 납부 구현(cudaBindTextureInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaBindTexture(
    size_t *offset, const struct textureReference *texref, const void *devPtr,
    const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX)) {
  return cudaBindTextureInternal(offset, texref, devPtr, desc,
                                 size __dv(UINT_MAX));
}

/*
 * [한국어]
 * CUDA 런타임 API cudaBindTextureToArray()의 GPGPU-Sim 구현. 납부 구현(cudaBindTextureToArrayInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaBindTextureToArray(
    const struct textureReference *texref, const struct cudaArray *array,
    const struct cudaChannelFormatDesc *desc) {
  return cudaBindTextureToArrayInternal(texref, array, desc);
}

__host__ cudaError_t CUDARTAPI
cudaUnbindTexture(const struct textureReference *texref) {
  return cudaUnbindTextureInternal(texref);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaGetTextureAlignmentOffset()의 GPGPU-Sim 구현. 납부 구현(cudaGetTextureAlignmentOffsetInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(
    size_t *offset, const struct textureReference *texref) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaGetTextureReference()의 GPGPU-Sim 구현. 납부 구현(cudaGetTextureReferenceInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaGetTextureReference(
    const struct textureReference **texref, const char *symbol) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaGetChannelDesc()의 GPGPU-Sim 구현. 납부 구현(cudaGetChannelDescInternal)에 위임
 */
__host__ cudaError_t CUDARTAPI cudaGetChannelDesc(
    struct cudaChannelFormatDesc *desc, const struct cudaArray *array) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *desc = array->desc;
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaCreateChannelDesc()의 GPGPU-Sim 구현. 납부 구현(cudaCreateChannelDescInternal)에 위임
 */
__host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(
    int x, int y, int z, int w, enum cudaChannelFormatKind f) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  struct cudaChannelFormatDesc dummy;
  dummy.x = x;
  dummy.y = y;
  dummy.z = z;
  dummy.w = w;
  dummy.f = f;
  return dummy;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaGetLastError()의 GPGPU-Sim 구현. 납부 구현(cudaGetLastErrorInternal)에 위임
 */
__host__ cudaError_t CUDARTAPI cudaGetLastError(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError;
}

__host__ const char *cudaGetErrorName(cudaError_t error) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return NULL;
}

__host__ const char *CUDARTAPI cudaGetErrorString(cudaError_t error) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  if (g_last_cudaError == cudaSuccess) return "no error";
  char buf[1024];
  snprintf(buf, 1024, "<<GPGPU-Sim PTX: there was an error (code = %d)>>",
           g_last_cudaError);
  return strdup(buf);
}

// SST specific cuda apis
/*
 * [한국어]
 * CUDA 런타임 API cudaSetupArgumentSST()의 GPGPU-Sim 구현. 납부 구현(cudaSetupArgumentSSTInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaSetupArgumentSST(uint64_t arg,
                                                    uint8_t value[200],
                                                    size_t size,
                                                    size_t offset) {
  void *local_value;
  local_value = (void *)malloc(size);

  if (arg) {
    memcpy(local_value, (void *)&arg, size);
  } else {
    memcpy(local_value, value, size);
  }
  return cudaSetupArgumentInternal(local_value, size, offset);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaSetupArgument()의 GPGPU-Sim 구현. 납부 구현(cudaSetupArgumentInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size,
                                                 size_t offset) {
  return cudaSetupArgumentInternal(arg, size, offset);
}

// SST specific cuda apis
__host__ cudaError_t CUDARTAPI cudaLaunchSST(uint64_t hostFun) {
  return cudaLaunchInternal((char *)hostFun);
}

__host__ cudaError_t CUDARTAPI cudaLaunch(const char *hostFun) {
  return cudaLaunchInternal(hostFun);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaLaunchKernel()의 GPGPU-Sim 구현. 납부 구현(cudaLaunchKernelInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaLaunchKernel(const char *hostFun,
                                                dim3 gridDim, dim3 blockDim,
                                                const void **args,
                                                size_t sharedMem,
                                                cudaStream_t stream) {
  return cudaLaunchKernelInternal(hostFun, gridDim, blockDim, args, sharedMem,
                                  stream);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *stream) {
  return cudaStreamCreateInternal(stream);
}

// TODO: introduce priorities
/*
 * [한국어]
 * CUDA 런타임 API cudaStreamCreateWithPriority()의 GPGPU-Sim 구현. 납부 구현(cudaStreamCreateWithPriorityInternal)에 위임
 */
__host__ cudaError_t CUDARTAPI cudaStreamCreateWithPriority(
    cudaStream_t *stream, unsigned int flags, int priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaStreamCreate(stream);
}

__host__ cudaError_t CUDARTAPI
/*
 * [한국어]
 * CUDA 런타임 API cudaDeviceGetStreamPriorityRange()의 GPGPU-Sim 구현. 납부 구현(cudaDeviceGetStreamPriorityRangeInternal)에 위임
 */
cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaSuccess;
}

__host__ __device__ cudaError_t CUDARTAPI
/*
 * [한국어]
 * CUDA 런타임 API cudaStreamCreateWithFlags()의 GPGPU-Sim 구현. 납부 구현(cudaStreamCreateWithFlagsInternal)에 위임
 */
cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaStreamCreate(stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
  return cudaStreamDestroyInternal(stream);
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
  return cudaStreamSynchronizeInternal(stream);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaStreamQuery()의 GPGPU-Sim 구현. 납부 구현(cudaStreamQueryInternal)에 위임
 */
__host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
#if (CUDART_VERSION >= 3000)
  if (stream == NULL) return g_last_cudaError = cudaErrorInvalidResourceHandle;
  return g_last_cudaError = stream->empty() ? cudaSuccess : cudaErrorNotReady;
#else
  printf(
      "GPGPU-Sim PTX: WARNING: Asynchronous kernel execution not supported "
      "(%s)\n",
      __my_func__);
  return g_last_cudaError = cudaSuccess;  // it is always success because all
                                          // cuda calls are synchronous
#endif
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

/*
 * [한국어]
 * CUDA 런타임 API cudaEventCreate()의 GPGPU-Sim 구현. 납부 구현(cudaEventCreateInternal)에 위임
 */
__host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = new CUevent_st(false);
  g_timer_events[e->get_uid()] = e;
#if CUDART_VERSION >= 3000
  *event = e;
#else
  *event = e->get_uid();
#endif
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaEventRecord()의 GPGPU-Sim 구현. 납부 구현(cudaEventRecordInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event,
                                               cudaStream_t stream) {
  return cudaEventRecordInternal(event, stream);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaStreamWaitEvent()의 GPGPU-Sim 구현. 납부 구현(cudaStreamWaitEventInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream,
                                                   cudaEvent_t event,
                                                   unsigned int flags) {
  return cudaStreamWaitEventInternal(stream, event, flags);
}
/*
 * [한국어]
 * CUDA 런타임 API cudaEventQuery()의 GPGPU-Sim 구현
 */
__host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = get_event(event);
  if (e == NULL) {
    return g_last_cudaError = cudaErrorInvalidValue;
  } else if (e->done()) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorNotReady;
  }
}

/*
 * [한국어]
 * CUDA 런타임 API cudaEventSynchronize()의 GPGPU-Sim 구현. 납부 구현(cudaEventSynchronizeInternal)에 위임
 */
__host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim API: cudaEventSynchronize ** waiting for event\n");
  fflush(stdout);
  CUevent_st *e = (CUevent_st *)event;
  while (!e->done())
    ;
  printf("GPGPU-Sim API: cudaEventSynchronize ** event detected\n");
  fflush(stdout);
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaEventDestroy()의 GPGPU-Sim 구현. 납부 구현(cudaEventDestroyInternal)에 위임
 */
__host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  CUevent_st *e = get_event(event);
  unsigned event_uid = e->get_uid();
  event_tracker_t::iterator pe = g_timer_events.find(event_uid);
  if (pe == g_timer_events.end())
    return g_last_cudaError = cudaErrorInvalidValue;
  g_timer_events.erase(pe);
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaEventElapsedTime()의 GPGPU-Sim 구현. 납부 구현(cudaEventElapsedTimeInternal)에 위임
 */
__host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms,
                                                    cudaEvent_t start,
                                                    cudaEvent_t end) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  time_t elapsed_time;
  CUevent_st *s = get_event(start);
  CUevent_st *e = get_event(end);
  if (s == NULL || e == NULL) return g_last_cudaError = cudaErrorUnknown;
  elapsed_time = e->clock() - s->clock();
  *ms = 1000 * elapsed_time;
  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaThreadExit(void) {
  return cudaThreadExitInternal();
}

__host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void) {
  return cudaThreadSynchronizeInternal();
}

__host__ cudaError_t CUDARTAPI cudaThreadSynchronizeSST(void) {
  // For SST, perform a one-time check and let SST_Cycle()
  // do the polling test and invoke callback to SST
  // to signal ThreadSynchonize done
  gpgpu_context *ctx = GPGPU_Context();
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }

  // Called on host side
  bool thread_sync_done = ctx->synchronize_check();
  g_last_cudaError = cudaSuccess;
  if (thread_sync_done) {
    // We are already done, so no need to poll for sync done
    ctx->requested_synchronize = false;
    return cudaSuccess;
  } else {
    return cudaErrorNotReady;
  }
}

/* [한국어] NVCC CUDA 런타임 콜백 __cudaSynchronizeThreads() */
int CUDARTAPI __cudaSynchronizeThreads(void **, void *) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaThreadExit();
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

#if (CUDART_VERSION >= 3010)
/* [한국어] CUDA 런타임 더미 심볼 0 */
int dummy0() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return 0;
}
/*
 * [한국어]
 * dummy1()
 */
int dummy1() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return 2 << 20;
}

typedef int (*ExportedFunction)();

static ExportedFunction exportTable[3] = {&dummy0, &dummy0, &dummy0};

/*
 * [한국어]
 * CUDA 런타임 API cudaGetExportTable()의 GPGPU-Sim 구현. 납부 구현(cudaGetExportTableInternal)에 위임
 */
__host__ cudaError_t CUDARTAPI cudaGetExportTable(
    const void **ppExportTable, const cudaUUID_t *pExportTableId) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("cudaGetExportTable: UUID = ");
  for (int s = 0; s < 16; s++) {
    printf("%#2x ", (unsigned char)(pExportTableId->bytes[s]));
  }
  *ppExportTable = &exportTable;

  printf("\n");
  return g_last_cudaError = cudaSuccess;
}

#endif

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

//#include "../../cuobjdump_to_ptxplus/cuobjdump_parser.h"

// extracts all ptx files from binary and dumps into
// prog_name.unique_no.sm_<>.ptx files
/* [한국어] cuobjdump로부터 PTX 파일 추출(납부 구현) */
void cuda_runtime_api::extract_ptx_files_using_cuobjdump_internal(
    CUctx_st *context, std::string &app_binary) {
  char command[1000];
  char *pytorch_bin = getenv("PYTORCH_BIN");

  char ptx_list_file_name[1024];
  snprintf(ptx_list_file_name, 1024, "_cuobjdump_list_ptx_XXXXXX");
  int fd2 = mkstemp(ptx_list_file_name);
  close(fd2);

  if (pytorch_bin != NULL && strlen(pytorch_bin) != 0) {
    app_binary = std::string(pytorch_bin);
  }

  // only want file names
  snprintf(command, 1000,
           "$CUDA_INSTALL_PATH/bin/cuobjdump -lptx %s  | cut -d \":\" -f 2 | "
           "awk '{$1=$1}1' > %s",
           app_binary.c_str(), ptx_list_file_name);
  if (system(command) != 0) {
    printf("WARNING: Failed to execute cuobjdump to get list of ptx files \n");
    exit(0);
  }
  if (!gpgpu_ctx->device_runtime->g_cdp_enabled) {
    // based on the list above, dump ptx files individually. Format of dumped
    // ptx file is prog_name.unique_no.sm_<>.ptx

    std::ifstream infile(ptx_list_file_name);
    std::string line;
    while (std::getline(infile, line)) {
      // int pos = line.find(std::string(get_app_binary_name(app_binary)));
      const char *ptx_file = line.c_str();
      printf("Extracting specific PTX file named %s \n", ptx_file);
      snprintf(command, 1000, "$CUDA_INSTALL_PATH/bin/cuobjdump -xptx %s %s",
               ptx_file, app_binary.c_str());
      if (system(command) != 0) {
        printf("ERROR: command: %s failed \n", command);
        exit(0);
      }
      context->no_of_ptx++;
    }
  }

  if (!context->no_of_ptx) {
    printf(
        "WARNING: Number of ptx in the executable file are 0. One of the "
        "reasons might be\n");
    printf("\t1. CDP is enabled\n");
    printf("\t2. When using PyTorch, PYTORCH_BIN is not set correctly\n");
  }

  std::ifstream infile(ptx_list_file_name);
  std::string line;
  while (std::getline(infile, line)) {
    // int pos = line.find(std::string(get_app_binary_name(app_binary)));
    int pos1 = line.find("sm_");
    int pos2 = line.find_last_of(".");
    if (pos1 == std::string::npos && pos2 == std::string::npos) {
      printf("ERROR: PTX list is not in correct format");
      exit(0);
    }
    std::string vstr = line.substr(pos1 + 3, pos2 - pos1 - 3);
    int version = atoi(vstr.c_str());
    if (version_filename.find(version) == version_filename.end()) {
      version_filename[version] = std::set<std::string>();
    }
    version_filename[version].insert(line);
  }
}
/*
 * [한국어]
 * cuobjdump로부터 PTX 파일 추출
 */
void cuda_runtime_api::extract_ptx_files_using_cuobjdump(CUctx_st *context,
                                                         const char *fn) {
  std::string app_binary = get_app_binary(fn);
  this->extract_ptx_files_using_cuobjdump_internal(context, app_binary);
}

void cuda_runtime_api::extract_ptx_files_using_cuobjdump(CUctx_st *context) {
  std::string app_binary = get_app_binary();
  this->extract_ptx_files_using_cuobjdump_internal(context, app_binary);
}

//! Call cuobjdump to extract everything (-elf -sass -ptx)
/*!
 *	This Function extract the whole PTX (for all the files) using cuobjdump
 *	to _cuobjdump_complete_output_XXXXXX then runs a parser to chop it up
 *with each binary in its own file It is also responsible for extracting the
 *libraries linked to the binary if the option is enabled
 * */
/* [한국어] cuobjdump로부터 ELF/SASS/PTX 코드 추출(납부 구현) */
void cuda_runtime_api::extract_code_using_cuobjdump_internal(
    CUctx_st *context, std::string &app_binary,
    std::function<void(CUctx_st *)> ctx_extract_ptx_func) {
  // prevent the dumping by cuobjdump everytime we execute the code!
  const char *override_cuobjdump = getenv("CUOBJDUMP_SIM_FILE");
  char command[1000];
  // Running cuobjdump using dynamic link to current process
  snprintf(command, 1000, "md5sum %s ", app_binary.c_str());
  printf("Running md5sum using \"%s\"\n", command);
  if (system(command)) {
    std::cout << "Failed to execute: " << command << std::endl;
    exit(1);
  }
  // Running cuobjdump using dynamic link to current process
  // Needs the option '-all' to extract PTX from CDP-enabled binary

  // dump ptx for all individial ptx files into sepearte files which is later
  // used by ptxas.
  int result = 0;
#if (CUDART_VERSION >= 6000)
  ctx_extract_ptx_func(context);
  return;
#endif
  // TODO: redundant to dump twice. how can it be prevented?
  // dump only for specific arch
  char fname[1024];
  if ((override_cuobjdump == NULL) || (strlen(override_cuobjdump) == 0)) {
    snprintf(fname, 1024, "_cuobjdump_complete_output_XXXXXX");
    int fd = mkstemp(fname);
    close(fd);
    if (!gpgpu_ctx->device_runtime->g_cdp_enabled)
      snprintf(command, 1000,
               "$CUDA_INSTALL_PATH/bin/cuobjdump -ptx -elf -sass %s > %s",
               app_binary.c_str(), fname);
    else
      snprintf(command, 1000,
               "$CUDA_INSTALL_PATH/bin/cuobjdump -ptx -elf -sass -all %s > %s",
               app_binary.c_str(), fname);
    bool parse_output = true;
    result = system(command);
    if (result) {
      if (context->get_device()
              ->get_gpgpu()
              ->get_config()
              .experimental_lib_support() &&
          (result == 65280)) {
        // Some CUDA application may exclusively use kernels provided by CUDA
        // libraries (e.g. CUBLAS).  Skipping cuobjdump extraction from the
        // executable for this case.
        // 65280 is the return code from cuobjdump denoting the specific error
        // (tested on CUDA 4.0/4.1/4.2)
        printf("WARNING: Failed to execute: %s\n", command);
        printf("         Executable binary does not contain any GPU kernel.\n");
        parse_output = false;
      } else {
        printf("ERROR: Failed to execute: %s\n", command);
        exit(1);
      }
    }

    if (parse_output) {
      printf("Parsing file %s\n", fname);
      FILE *cuobjdump_in;
      cuobjdump_in = fopen(fname, "r");

      struct cuobjdump_parser parser;
      parser.elfserial = 1;
      parser.ptxserial = 1;
      cuobjdump_lex_init(&(parser.scanner));
      cuobjdump_set_in(cuobjdump_in, (parser.scanner));
      cuobjdump_parse(parser.scanner, &parser, cuobjdumpSectionList);
      cuobjdump_lex_destroy(parser.scanner);
      fclose(cuobjdump_in);
      printf("Done parsing!!!\n");
    } else {
      printf("Parsing skipped for %s\n", fname);
    }

    if (context->get_device()
            ->get_gpgpu()
            ->get_config()
            .experimental_lib_support()) {
      // Experimental library support
      // Currently only for cufft

      std::stringstream cmd;
      cmd << "ldd " << app_binary
          << " | grep $CUDA_INSTALL_PATH | awk \'{print $3}\' > _tempfile_.txt";
      int result = system(cmd.str().c_str());
      if (result) {
        std::cout << "Failed to execute: " << cmd.str() << std::endl;
        exit(1);
      }
      std::ifstream libsf;
      libsf.open("_tempfile_.txt");
      if (!libsf.is_open()) {
        std::cout << "Failed to open: _tempfile_.txt" << std::endl;
        exit(1);
      }

      // Save the original section list
      std::list<cuobjdumpSection *> tmpsl = cuobjdumpSectionList;
      cuobjdumpSectionList.clear();

      std::string line;
      std::getline(libsf, line);
      std::cout << "DOING: " << line << std::endl;
      int cnt = 1;
      while (libsf.good()) {
        std::stringstream libcodfn;
        libcodfn << "_cuobjdump_complete_lib_" << cnt << "_";
        cmd.str("");  // resetting
        cmd << "$CUDA_INSTALL_PATH/bin/cuobjdump -ptx -elf -sass ";
        cmd << line;
        cmd << " > ";
        cmd << libcodfn.str();
        std::cout << "Running cuobjdump on " << line << std::endl;
        std::cout << "Using command: " << cmd.str() << std::endl;
        result = system(cmd.str().c_str());
        if (result) {
          printf("ERROR: Failed to execute: %s\n", command);
          exit(1);
        }
        std::cout << "Done" << std::endl;

        std::cout << "Trying to parse " << libcodfn.str() << std::endl;
        FILE *cuobjdump_in;
        cuobjdump_in = fopen(libcodfn.str().c_str(), "r");
        struct cuobjdump_parser parser;
        parser.elfserial = 1;
        parser.ptxserial = 1;
        cuobjdump_lex_init(&(parser.scanner));
        cuobjdump_set_in(cuobjdump_in, (parser.scanner));
        cuobjdump_parse(parser.scanner, &parser, cuobjdumpSectionList);
        cuobjdump_lex_destroy(parser.scanner);
        fclose(cuobjdump_in);
        std::getline(libsf, line);
      }
      libSectionList = cuobjdumpSectionList;

      // Restore the original section list
      cuobjdumpSectionList = tmpsl;
    }
  } else {
    printf(
        "GPGPU-Sim PTX: overriding cuobjdump with '%s' (CUOBJDUMP_SIM_FILE is "
        "set)\n",
        override_cuobjdump);
    snprintf(fname, 1024, "%s", override_cuobjdump);
  }
}

void cuda_runtime_api::extract_code_using_cuobjdump(const char *fn) {
  CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
  std::string app_binary = get_app_binary(fn);
  auto ctx_extract_ptx_func = [=](CUctx_st *context) {
    extract_ptx_files_using_cuobjdump(context, fn);
  };
  extract_code_using_cuobjdump_internal(context, app_binary,
                                        ctx_extract_ptx_func);
}

void cuda_runtime_api::extract_code_using_cuobjdump() {
  CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
  std::string app_binary = get_app_binary();
  auto ctx_extract_ptx_func = [=](CUctx_st *context) {
    extract_ptx_files_using_cuobjdump(context);
  };
  extract_code_using_cuobjdump_internal(context, app_binary,
                                        ctx_extract_ptx_func);
}

//! Read file into char*
// TODO: convert this to C++ streams, will be way cleaner
char *readfile(const std::string filename) {
  assert(filename != "");
  FILE *fp = fopen(filename.c_str(), "r");
  if (!fp) {
    std::cout << "ERROR: Could not open file %s for reading\n"
              << filename << std::endl;
    assert(0);
  }
  // finding size of the file
  int filesize = 0;
  fseek(fp, 0, SEEK_END);

  filesize = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  // allocate and copy the entire ptx
  char *ret = (char *)malloc((filesize + 1) * sizeof(char));
  int num = fread(ret, 1, filesize, fp);
  if (num == 0) {
    std::cout << "ERROR: Could not read data from file %s\n"
              << filename << std::endl;
    assert(0);
  }
  ret[filesize] = '\0';
  fclose(fp);
  return ret;
}

//! Function that helps debugging
void printSectionList(std::list<cuobjdumpSection *> sl) {
  std::list<cuobjdumpSection *>::iterator iter;
  for (iter = sl.begin(); iter != sl.end(); iter++) {
    (*iter)->print();
  }
}

//! Remove unecessary sm versions from the section list
std::list<cuobjdumpSection *> cuda_runtime_api::pruneSectionList(
    CUctx_st *context) {
  unsigned forced_max_capability = context->get_device()
                                       ->get_gpgpu()
                                       ->get_config()
                                       .get_forced_max_capability();

  // For ptxplus, force the max capability to 19 if it's higher or
  // unspecified(0)
  if (context->get_device()->get_gpgpu()->get_config().convert_to_ptxplus()) {
    if ((forced_max_capability == 0) || (forced_max_capability >= 20)) {
      printf(
          "GPGPU-Sim: WARNING: Capability >= 20 are not supported in "
          "PTXPlus\n\tSetting forced_max_capability to 19\n");
      forced_max_capability = 19;
    }
  }

  std::list<cuobjdumpSection *> prunedList;

  // Find the highest capability (that is lower than the forced maximum) for
  // each cubin file and set it in cuobjdumpSectionMap. Do this only for ptx
  // sections
  std::map<std::string, unsigned> cuobjdumpSectionMap;
  int min_ptx_capability_found = 0;
  for (std::list<cuobjdumpSection *>::iterator iter =
           cuobjdumpSectionList.begin();
       iter != cuobjdumpSectionList.end(); iter++) {
    unsigned capability = (*iter)->getArch();
    if (dynamic_cast<cuobjdumpPTXSection *>(*iter) != NULL) {
      if (capability < min_ptx_capability_found ||
          min_ptx_capability_found == 0)
        min_ptx_capability_found = capability;
      if (capability <= forced_max_capability || forced_max_capability == 0) {
        if ((cuobjdumpSectionMap.find((*iter)->getIdentifier()) ==
             cuobjdumpSectionMap.end()) ||
            (cuobjdumpSectionMap[(*iter)->getIdentifier()] < capability))
          cuobjdumpSectionMap[(*iter)->getIdentifier()] = capability;
      }
    }
  }

  // Throw away the sections with the lower capabilites and push those with the
  // highest in the pruned list
  for (std::list<cuobjdumpSection *>::iterator iter =
           cuobjdumpSectionList.begin();
       iter != cuobjdumpSectionList.end(); iter++) {
    unsigned capability = (*iter)->getArch();
    if (capability == cuobjdumpSectionMap[(*iter)->getIdentifier()]) {
      prunedList.push_back(*iter);
    } else {
      delete *iter;
    }
  }
  if (prunedList.empty()) {
    printf(
        "Error: No PTX sections found with sm capability that is lower than "
        "current forced maximum capability \n minimum ptx capability found = "
        "%u, maximum forced ptx capability = %u \n User might want to change "
        "either the forced maximum capability from gpgpusim configuration or "
        "update the compilation to generate the required PTX version\n",
        min_ptx_capability_found, forced_max_capability);
    abort();
  }
  return prunedList;
}

//! Merge all PTX sections that have a specific identifier into one file
std::list<cuobjdumpSection *> cuda_runtime_api::mergeMatchingSections(
    std::string identifier) {
  const char *ptxcode = "";
  std::list<cuobjdumpSection *>::iterator old_iter;
  cuobjdumpPTXSection *old_ptxsection = NULL;
  cuobjdumpPTXSection *ptxsection;
  std::list<cuobjdumpSection *> mergedList;

  for (std::list<cuobjdumpSection *>::iterator iter =
           cuobjdumpSectionList.begin();
       iter != cuobjdumpSectionList.end(); iter++) {
    if ((ptxsection = dynamic_cast<cuobjdumpPTXSection *>(*iter)) != NULL &&
        strcmp(ptxsection->getIdentifier().c_str(), identifier.c_str()) == 0) {
      // Read and remove the last PTX section
      if (old_ptxsection != NULL) {
        ptxcode = readfile(old_ptxsection->getPTXfilename());
        // remove ptx file?
        delete *old_iter;
      }

      // Append all the PTX from the last PTX section into the current PTX
      // section Add 50 to ptxcode to ignore the information regarding
      // version/target/address_size
      if (strlen(ptxcode) >= 50) {
        FILE *ptxfile = fopen((ptxsection->getPTXfilename()).c_str(), "a");
        fprintf(ptxfile, "%s", ptxcode + 50);
        fclose(ptxfile);
      }

      old_iter = iter;
      old_ptxsection = ptxsection;
    }
    // Store all non-PTX sections and PTX sections with non-matching identifiers
    else {
      mergedList.push_back(*iter);
    }
  }

  // Store the final PTX section
  mergedList.push_back(*old_iter);

  return mergedList;
}

//! Merge any PTX sections with matching identifiers
std::list<cuobjdumpSection *> cuda_runtime_api::mergeSections() {
  std::vector<std::string> identifier;
  cuobjdumpPTXSection *ptxsection;

  // Add all identifiers present in PTX sections to a vector
  for (std::list<cuobjdumpSection *>::iterator iter =
           cuobjdumpSectionList.begin();
       iter != cuobjdumpSectionList.end(); iter++) {
    if ((ptxsection = dynamic_cast<cuobjdumpPTXSection *>(*iter)) != NULL) {
      std::string current_id = ptxsection->getIdentifier();

      // If we haven't yet seen a given identifier, add it to the vector
      if (std::find(identifier.begin(), identifier.end(), current_id) ==
          identifier.end()) {
        identifier.push_back(current_id);
      }
    }
  }

  // Call mergeMatchingSections on all identifiers in the vector
  for (std::vector<std::string>::iterator iter = identifier.begin();
       iter != identifier.end(); iter++) {
    cuobjdumpSectionList = mergeMatchingSections(*iter);
  }

  return cuobjdumpSectionList;
}

//! Within the section list, find the ELF section corresponding to a given
//! identifier
cuobjdumpELFSection *findELFSectionInList(
    std::list<cuobjdumpSection *> sectionlist, const std::string identifier) {
  std::list<cuobjdumpSection *>::iterator iter;
  for (iter = sectionlist.begin(); iter != sectionlist.end(); iter++) {
    cuobjdumpELFSection *elfsection;
    if ((elfsection = dynamic_cast<cuobjdumpELFSection *>(*iter)) != NULL) {
      if (elfsection->getIdentifier() == identifier) return elfsection;
    }
  }
  return NULL;
}

//! Find an ELF section in all the known lists
cuobjdumpELFSection *cuda_runtime_api::findELFSection(
    const std::string identifier) {
  cuobjdumpELFSection *sec =
      findELFSectionInList(cuobjdumpSectionList, identifier);
  if (sec != NULL) return sec;
  sec = findELFSectionInList(libSectionList, identifier);
  if (sec != NULL) return sec;
  std::cout << "Could not find " << identifier << std::endl;
  assert(0 && "Could not find the required ELF section");
  return NULL;
}

//! Within the section list, find the PTX section corresponding to a given
//! identifier
cuobjdumpPTXSection *cuda_runtime_api::findPTXSectionInList(
    std::list<cuobjdumpSection *> &sectionlist, const std::string identifier) {
  std::list<cuobjdumpSection *>::iterator iter;
  for (iter = sectionlist.begin(); iter != sectionlist.end(); iter++) {
    cuobjdumpPTXSection *ptxsection;
    if ((ptxsection = dynamic_cast<cuobjdumpPTXSection *>(*iter)) != NULL) {
      if (ptxsection->getIdentifier() == identifier)
        return ptxsection;
      else {
        if (gpgpu_ctx->device_runtime->g_cdp_enabled) {
          printf(
              "Warning: __cudaRegisterFatBinary needs %s, but find PTX section "
              "with %s\n",
              identifier.c_str(), ptxsection->getIdentifier().c_str());
          return ptxsection;
        }
      }
    }
  }
  return NULL;
}

//! Find an PTX section in all the known lists
cuobjdumpPTXSection *cuda_runtime_api::findPTXSection(
    const std::string identifier) {
  cuobjdumpPTXSection *sec =
      findPTXSectionInList(cuobjdumpSectionList, identifier);
  if (sec != NULL) return sec;
  sec = findPTXSectionInList(libSectionList, identifier);
  if (sec != NULL) return sec;
  std::cout << "Could not find " << identifier << std::endl;
  assert(0 && "Could not find the required PTX section");
  return NULL;
}

//! Extract the code using cuobjdump and remove unnecessary sections
void cuda_runtime_api::cuobjdumpInit_internal(
    std::function<void()> ctx_extract_code_func) {
  CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
  ctx_extract_code_func();  // extract all the output of cuobjdump to
                            // _cuobjdump_*.*
  const char *pre_load = getenv("CUOBJDUMP_SIM_FILE");
  if (pre_load == NULL || strlen(pre_load) == 0) {
    cuobjdumpSectionList = pruneSectionList(context);
    cuobjdumpSectionList = mergeSections();
  }
}

void cuda_runtime_api::cuobjdumpInit(const char *fn) {
  auto ctx_extract_code_func = [=]() { extract_code_using_cuobjdump(fn); };
  cuobjdumpInit_internal(ctx_extract_code_func);
}

void cuda_runtime_api::cuobjdumpInit() {
  auto ctx_extract_code_func = [=]() { extract_code_using_cuobjdump(); };
  cuobjdumpInit_internal(ctx_extract_code_func);
}

//! Either submit PTX for simulation or convert SASS to PTXPlus and submit it
void gpgpu_context::cuobjdumpParseBinary(unsigned int handle) {
  CUctx_st *context = GPGPUSim_Context(this);
  if (api->fatbin_registered[handle]) return;
  api->fatbin_registered[handle] = true;
  std::string fname = api->fatbinmap[handle];

  if (api->name_symtab.find(fname) != api->name_symtab.end()) {
    symbol_table *symtab = api->name_symtab[fname];
    context->add_binary(symtab, handle);
    return;
  }
  symbol_table *symtab = NULL;

#if (CUDART_VERSION >= 6000)
  // loops through all ptx files from smallest sm version to largest
  std::map<unsigned, std::set<std::string> >::iterator itr_m;
  for (itr_m = api->version_filename.begin();
       itr_m != api->version_filename.end(); itr_m++) {
    std::set<std::string>::iterator itr_s;
    for (itr_s = itr_m->second.begin(); itr_s != itr_m->second.end(); itr_s++) {
      std::string ptx_filename = *itr_s;
      printf("GPGPU-Sim PTX: Parsing %s\n", ptx_filename.c_str());
      symtab = gpgpu_ptx_sim_load_ptx_from_filename(ptx_filename.c_str());
    }
  }
  api->name_symtab[fname] = symtab;
  context->add_binary(symtab, handle);
  api->load_static_globals(symtab, STATIC_ALLOC_LIMIT, 0xFFFFFFFF,
                           context->get_device()->get_gpgpu());
  api->load_constants(symtab, STATIC_ALLOC_LIMIT,
                      context->get_device()->get_gpgpu());
  for (itr_m = api->version_filename.begin();
       itr_m != api->version_filename.end(); itr_m++) {
    std::set<std::string>::iterator itr_s;
    for (itr_s = itr_m->second.begin(); itr_s != itr_m->second.end(); itr_s++) {
      std::string ptx_filename = *itr_s;
      printf("GPGPU-Sim PTX: Loading PTXInfo from %s\n", ptx_filename.c_str());
      gpgpu_ptx_info_load_from_filename(ptx_filename.c_str(), itr_m->first);
    }
  }
  return;
#endif

  unsigned max_capability = 0;
  for (std::list<cuobjdumpSection *>::iterator iter =
           api->cuobjdumpSectionList.begin();
       iter != api->cuobjdumpSectionList.end(); iter++) {
    unsigned capability = (*iter)->getArch();
    if (capability > max_capability) max_capability = capability;
  }
  if (max_capability > 20)
    printf("WARNING: No guarantee that PTX will be parsed for SM version %u\n",
           max_capability);
  if (max_capability == 0)
    max_capability = context->get_device()
                         ->get_gpgpu()
                         ->get_config()
                         .get_forced_max_capability();

  cuobjdumpPTXSection *ptx = NULL;
  const char *pre_load = getenv("CUOBJDUMP_SIM_FILE");
  if (pre_load == NULL || strlen(pre_load) == 0)
    ptx = api->findPTXSection(fname);
  char *ptxcode;
  const char *override_ptx_name = getenv("PTX_SIM_KERNELFILE");
  if (override_ptx_name == NULL or getenv("PTX_SIM_USE_PTX_FILE") == NULL or
      strlen(getenv("PTX_SIM_USE_PTX_FILE")) == 0) {
    ptxcode = readfile(ptx->getPTXfilename());
  } else {
    printf(
        "GPGPU-Sim PTX: overriding embedded ptx with '%s' "
        "(PTX_SIM_USE_PTX_FILE is set)\n",
        override_ptx_name);
    ptxcode = readfile(override_ptx_name);
  }
  if (context->get_device()->get_gpgpu()->get_config().convert_to_ptxplus()) {
    cuobjdumpELFSection *elfsection = api->findELFSection(ptx->getIdentifier());
    assert(elfsection != NULL);
    char *ptxplus_str = ptxinfo->gpgpu_ptx_sim_convert_ptx_and_sass_to_ptxplus(
        ptx->getPTXfilename(), elfsection->getELFfilename(),
        elfsection->getSASSfilename());
    symtab = gpgpu_ptx_sim_load_ptx_from_string(ptxplus_str, handle);
    printf("Adding %s with cubin handle %u\n", ptx->getPTXfilename().c_str(),
           handle);
    context->add_binary(symtab, handle);
    gpgpu_ptxinfo_load_from_string(ptxcode, handle, max_capability,
                                   context->no_of_ptx);
    delete[] ptxplus_str;
  } else {
    symtab = gpgpu_ptx_sim_load_ptx_from_string(ptxcode, handle);
    // if CUOBJDUMP_SIM_FILE is not set, ptx is NULL. So comment below.
    // printf("Adding %s with cubin handle %u\n", ptx->getPTXfilename().c_str(),
    // handle);
    context->add_binary(symtab, handle);
    gpgpu_ptxinfo_load_from_string(ptxcode, handle, max_capability,
                                   context->no_of_ptx);
  }
  api->load_static_globals(symtab, STATIC_ALLOC_LIMIT, 0xFFFFFFFF,
                           context->get_device()->get_gpgpu());
  api->load_constants(symtab, STATIC_ALLOC_LIMIT,
                      context->get_device()->get_gpgpu());
  api->name_symtab[fname] = symtab;

  // TODO: Remove temporarily files as per configurations
}
}

extern "C" {

void **CUDARTAPI __cudaRegisterFatBinarySST(const char *fn) {
  return cudaRegisterFatBinaryInternal(fn, NULL);
}

void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return cudaRegisterFatBinaryInternal(fatCubin);
}
/*
 * [한국어]
 * NVCC 런타임 콜백 __cudaRegisterFatBinaryEnd()
 */
void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
}
/*
 * [한국어]
 * NVCC CDP 콜백 __cudaPushCallConfiguration() — 실제 동작은 *Internal() 납부 함수로 전달
 */
unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                               size_t sharedMem = 0,
                                               struct CUstream_st *stream = 0) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream);
  return 0;
}
/*
 * [한국어]
 * NVCC CDP 콜백 __cudaPopCallConfiguration()
 */
cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                 size_t *sharedMem,
                                                 void *stream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * SST 통합용 NVCC 런타임 콜백 __cudaRegisterFunctionSST() — 실제 동작은 *Internal() 납부 함수로 전달
 */
void CUDARTAPI __cudaRegisterFunctionSST(unsigned fatCubinHandle,
                                         uint64_t hostFun,
                                         char deviceFun[512]) {
  cudaRegisterFunctionInternal((void **)fatCubinHandle, (const char *)hostFun,
                               (char *)deviceFun, NULL, NULL, NULL, NULL, NULL,
                               NULL);
}
/*
 * [한국어]
 * NVCC 런타임 콜백 __cudaRegisterFunction() — 실제 동작은 *Internal() 납부 함수로 전달
 */
void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle,
                                      const char *hostFun, char *deviceFun,
                                      const char *deviceName, int thread_limit,
                                      uint3 *tid, uint3 *bid, dim3 *bDim,
                                      dim3 *gDim) {
  cudaRegisterFunctionInternal(fatCubinHandle, hostFun, deviceFun, deviceName,
                               thread_limit, tid, bid, bDim, gDim);
}
/*
 * [한국어]
 * NVCC 런타임 콜백 __cudaRegisterVar() — 실제 동작은 *Internal() 납부 함수로 전달
 */
extern void __cudaRegisterVar(
    void **fatCubinHandle,
    char *hostVar,           // pointer to...something
    char *deviceAddress,     // name of variable
    const char *deviceName,  // name of variable (same as above)
    int ext, int size, int constant, int global) {
  cudaRegisterVarInternal(fatCubinHandle, hostVar, deviceAddress, deviceName,
                          ext, size, constant, global);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaConfigureCall()의 GPGPU-Sim 구현. 납부 구현(cudaConfigureCallInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
__host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                                 size_t sharedMem,
                                                 cudaStream_t stream) {
  return cudaConfigureCallInternal(gridDim, blockDim, sharedMem, stream);
}
/*
 * [한국어]
 * NVCC 런타임 콜백 __cudaUnregisterFatBinary()
 */
void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
}

/*
 * [한국어]
 * CUDA 런타임 API cudaDeviceReset()의 GPGPU-Sim 구현. 납부 구현(cudaDeviceResetInternal)에 위임
 */
cudaError_t cudaDeviceReset(void) {
  // Should reset the simulated GPU
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceSynchronize(void) {
  return cudaDeviceSynchronizeInternal();
}
/*
 * [한국어]
 * NVCC 런타임 콜백 __cudaRegisterShared()
 */
void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // we don't do anything here
  printf("GPGPU-Sim PTX: __cudaRegisterShared\n");
}
/*
 * [한국어]
 * NVCC 런타임 콜백 __cudaRegisterSharedVar()
 */
void CUDARTAPI __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
                                       size_t size, size_t alignment,
                                       int storage) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // we don't do anything here
  printf("GPGPU-Sim PTX: __cudaRegisterSharedVar\n");
}
/*
 * [한국어]
 * NVCC 런타임 콜백 __cudaRegisterTexture() — 실제 동작은 *Internal() 납부 함수로 전달
 */
void __cudaRegisterTexture(
    void **fatCubinHandle, const struct textureReference *hostVar,
    const void **deviceAddress, const char *deviceName, int dim, int norm,
    int ext)  // passes in a newly created textureReference
{
  __cudaRegisterTextureInternal(fatCubinHandle, hostVar, deviceAddress,
                                deviceName, dim, norm, ext);
}
/*
 * [한국어]
 * NVCC 런타임 콜백 __cudaInitModule() (GPGPU-Sim에서 현재 미구현/미지원)
 */
char __cudaInitModule(void **fatCubinHandle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaGLRegisterBufferObject()의 GPGPU-Sim 구현. 납부 구현(cudaGLRegisterBufferObjectInternal)에 위임
 */
cudaError_t cudaGLRegisterBufferObject(GLuint bufferObj) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n",
         __my_func__);
  return g_last_cudaError = cudaSuccess;
}

cudaError_t cudaGLMapBufferObject(void **devPtr, GLuint bufferObj) {
  return cudaGLMapBufferObjectInternal(devPtr, bufferObj);
}

cudaError_t cudaGLUnmapBufferObject(GLuint bufferObj) {
  return cudaGLUnmapBufferObjectInternal(bufferObj);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaGLUnregisterBufferObject()의 GPGPU-Sim 구현. 납부 구현(cudaGLUnregisterBufferObjectInternal)에 위임
 */
cudaError_t cudaGLUnregisterBufferObject(GLuint bufferObj) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n",
         __my_func__);
  return g_last_cudaError = cudaSuccess;
}

#if (CUDART_VERSION >= 2010)

/*
 * [한국어]
 * CUDA 런타임 API cudaHostAlloc()의 GPGPU-Sim 구현. 납부 구현(cudaHostAllocInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t bytes,
                                    unsigned int flags) {
  return cudaHostAllocInternal(pHost, bytes, flags);
}

/*
 * [한국어]
 * CUDA 런타임 API cudaHostGetDevicePointer()의 GPGPU-Sim 구현. 납부 구현(cudaHostGetDevicePointerInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                               unsigned int flags) {
  return cudaHostGetDevicePointerInternal(pDevice, pHost, flags);
}

__host__ cudaError_t CUDARTAPI
/*
 * [한국어]
 * CUDA 런타임 API cudaPointerGetAttributes()의 GPGPU-Sim 구현. 납부 구현(cudaPointerGetAttributesInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
cudaPointerGetAttributes(cudaPointerAttributes *attributes, const void *ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaDeviceCanAccessPeer()의 GPGPU-Sim 구현. 납부 구현(cudaDeviceCanAccessPeerInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer,
                                                       int device,
                                                       int peerDevice) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaDeviceEnablePeerAccess()의 GPGPU-Sim 구현. 납부 구현(cudaDeviceEnablePeerAccessInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice,
                                                          unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaSetValidDevices()의 GPGPU-Sim 구현. 납부 구현(cudaSetValidDevicesInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
cudaError_t CUDARTAPI cudaSetValidDevices(int *device_arr, int len) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaSetDeviceFlags()의 GPGPU-Sim 구현. 납부 구현(cudaSetDeviceFlagsInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
cudaError_t CUDARTAPI cudaSetDeviceFlags(int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // This flag is implicitly always on (unless you are using the driver API). It
  // is safe for GPGPU-Sim to just ignore it.
  if (cudaDeviceMapHost == flags) {
    return g_last_cudaError = cudaSuccess;
  } else {
    cuda_not_implemented(__my_func__, __LINE__);
    return g_last_cudaError = cudaErrorUnknown;
  }
}

/*
 * [한국어]
 * CUDA 런타임 API cudaFuncGetAttributes()의 GPGPU-Sim 구현. 납부 구현(cudaFuncGetAttributesInternal)에 위임 — 실제 동작은 *Internal() 납부 함수로 전달
 */
cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
                                            const char *hostFun) {
  return cudaFuncGetAttributesInternal(attr, hostFun);
}

cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, int flags) {
  CUevent_st *e = new CUevent_st(flags == cudaEventBlockingSync);
  g_timer_events[e->get_uid()] = e;
#if CUDART_VERSION >= 3000
  *event = e;
#else
  *event = e->get_uid();
#endif
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaDriverGetVersion()의 GPGPU-Sim 구현. 납부 구현(cudaDriverGetVersionInternal)에 위임
 */
cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *driverVersion = CUDART_VERSION;
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaRuntimeGetVersion()의 GPGPU-Sim 구현. 납부 구현(cudaRuntimeGetVersionInternal)에 위임
 */
cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *runtimeVersion = CUDART_VERSION;
  return g_last_cudaError = cudaSuccess;
}

#if CUDART_VERSION >= 3000
__host__ cudaError_t CUDARTAPI
cudaFuncSetCacheConfig(const char *func, enum cudaFuncCache cacheConfig) {
  return cudaFuncSetCacheConfigInternal(func, cacheConfig);
}

// Jin: hack for cdp
/*
 * [한국어]
 * CUDA 런타임 API cudaDeviceSetLimit()의 GPGPU-Sim 구현. 납부 구현(cudaDeviceSetLimitInternal)에 위임
 */
__host__ cudaError_t CUDARTAPI cudaDeviceSetLimit(enum cudaLimit limit,
                                                  size_t value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  return g_last_cudaError = cudaSuccess;
}

//#if CUDART_VERSION >= 9000
//__host__  cudaError_t cudaFuncSetAttribute ( const void* func, enum
// cudaFuncAttribute attr, int value ) {

// ignore this Attribute for now, and the default is that carveout =
// cudaSharedmemCarveoutDefault;   //  (-1)
//	return g_last_cudaError = cudaSuccess;
//}

#endif

#endif

#if CUDART_VERSION >= 9000
/**
 * \brief Set attributes for a given function
 *
 * This function sets the attributes of a function specified via \p entry.
 * The parameter \p entry must be a pointer to a function that executes
 * on the device. The parameter specified by \p entry must be declared as a \p
 * __global__ function. The enumeration defined by \p attr is set to the value
 * defined by \p value If the specified function does not exist, then
 * ::cudaErrorInvalidDeviceFunction is returned. If the specified attribute
 * cannot be written, or if the value is incorrect, then ::cudaErrorInvalidValue
 * is returned.
 *
 * Valid values for \p attr are:
 * ::cuFuncAttrMaxDynamicSharedMem - Maximum size of dynamic shared memory per
 * block
 * ::cudaFuncAttributePreferredSharedMemoryCarveout - Preferred shared memory-L1
 * cache split ratio
 *
 * \param entry - Function to get attributes of
 * \param attr  - Attribute to set
 * \param value - Value to set
 *
 * \return
 * ::cudaSuccess,
 * ::cudaErrorInitializationError,
 * ::cudaErrorInvalidDeviceFunction,
 * ::cudaErrorInvalidValue
 * \notefnerr
 *
 * \ref ::cudaLaunchKernel(const T *func, dim3 gridDim, dim3 blockDim, void
 * **args, size_t sharedMem, cudaStream_t stream) "cudaLaunchKernel (C++ API)",
 * \ref ::cudaFuncSetCacheConfig(T*, enum cudaFuncCache) "cudaFuncSetCacheConfig
 * (C++ API)", \ref ::cudaFuncGetAttributes(struct cudaFuncAttributes*, const
 * void*) "cudaFuncGetAttributes (C API)",
 * ::cudaSetDoubleForDevice,
 * ::cudaSetDoubleForHost,
 * \ref ::cudaSetupArgument(T, size_t) "cudaSetupArgument (C++ API)"
 */
/*
 * [한국어]
 * CUDA 런타임 API cudaFuncSetAttribute()의 GPGPU-Sim 구현. 납부 구현(cudaFuncSetAttributeInternal)에 위임
 */
cudaError_t CUDARTAPI cudaFuncSetAttribute(const void *func,
                                           enum cudaFuncAttribute attr,
                                           int value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf(
      "GPGPU-Sim PTX: Execution warning: ignoring call to \"%s ( func=%p, "
      "attr=%d, value=%d )\"\n",
      __my_func__, func, attr, value);
  return g_last_cudaError = cudaSuccess;
}
#endif

/*
 * [한국어]
 * CUDA 런타임 API cudaGLSetGLDevice()의 GPGPU-Sim 구현. 납부 구현(cudaGLSetGLDeviceInternal)에 위임
 */
cudaError_t CUDARTAPI cudaGLSetGLDevice(int device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n",
         __my_func__);
  return g_last_cudaError = cudaErrorUnknown;
}

typedef void *HGPUNV;

/*
 * [한국어]
 * CUDA 런타임 API cudaWGLGetDevice()의 GPGPU-Sim 구현. 납부 구현(cudaWGLGetDeviceInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
cudaError_t CUDARTAPI cudaWGLGetDevice(int *device, HGPUNV hGpu) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}
/*
 * [한국어]
 * NVCC 런타임 뮤텍스 콜백 (미구현)
 */
void CUDARTAPI __cudaMutexOperation(int lock) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}
/*
 * [한국어]
 * NVCC 런타임 텍스처 페치 콜백 (미구현)
 */
void CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer,
                                  void *val) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}
}

namespace cuda_math {
/*
 * [한국어]
 * NVCC 런타임 뮤텍스 콜백 (미구현)
 */
void CUDARTAPI __cudaMutexOperation(int lock) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}
/*
 * [한국어]
 * NVCC 런타임 텍스처 페치 콜백 (미구현)
 */
void CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer,
                                  void *val) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
}
/*
 * [한국어]
 * NVCC 런타임 스레드 동기화 콜백
 */
int CUDARTAPI __cudaSynchronizeThreads(void **, void *) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // TODO This function should syncronize if we support Asyn kernel calls
  return g_last_cudaError = cudaSuccess;
}

}  // namespace cuda_math

////////

/// static functions
int cuda_runtime_api::load_static_globals(symbol_table *symtab,
                                          unsigned min_gaddr,
                                          unsigned max_gaddr, gpgpu_t *gpu) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: loading globals with explicit initializers... \n");
  fflush(stdout);
  int ng_bytes = 0;
  symbol_table::iterator g = symtab->global_iterator_begin();

  for (; g != symtab->global_iterator_end(); g++) {
    symbol *global = *g;
    if (global->has_initializer()) {
      printf("GPGPU-Sim PTX:     initializing '%s' ... ",
             global->name().c_str());
      unsigned addr = global->get_address();
      const type_info *type = global->type();
      type_info_key ti = type->get_key();
      size_t size;
      int t;
      ti.type_decode(size, t);
      int nbytes = size / 8;
      int offset = 0;
      std::list<operand_info> init_list = global->get_initializer();
      for (std::list<operand_info>::iterator i = init_list.begin();
           i != init_list.end(); i++) {
        operand_info op = *i;
        ptx_reg_t value = op.get_literal_value();
        assert((addr + offset + nbytes) <
               min_gaddr);  // min_gaddr is start of "heap" for cudaMalloc
        gpu->get_global_memory()->write(addr + offset, nbytes, &value, NULL,
                                        NULL);  // assuming little endian here
        offset += nbytes;
        ng_bytes += nbytes;
      }
      printf(" wrote %u bytes\n", offset);
    }
  }
  printf("GPGPU-Sim PTX: finished loading globals (%u bytes total).\n",
         ng_bytes);
  fflush(stdout);
  return ng_bytes;
}
/*
 * [한국어]
 * PTX 심볼 테이블의 상수 변수를 GPU 상수 메모리에 로드
 */
int cuda_runtime_api::load_constants(symbol_table *symtab, addr_t min_gaddr,
                                     gpgpu_t *gpu) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("GPGPU-Sim PTX: loading constants with explicit initializers... ");
  fflush(stdout);
  int nc_bytes = 0;
  symbol_table::iterator g = symtab->const_iterator_begin();

  for (; g != symtab->const_iterator_end(); g++) {
    symbol *constant = *g;
    if (constant->is_const() && constant->has_initializer()) {
      // get the constant element data size
      int basic_type;
      size_t num_bits;
      constant->type()->get_key().type_decode(num_bits, basic_type);

      std::list<operand_info> init_list = constant->get_initializer();
      int nbytes_written = 0;
      for (std::list<operand_info>::iterator i = init_list.begin();
           i != init_list.end(); i++) {
        operand_info op = *i;
        ptx_reg_t value = op.get_literal_value();
        int nbytes = num_bits / 8;
        switch (op.get_type()) {
          case int_t:
            assert(nbytes >= 1);
            break;
          case float_op_t:
            assert(nbytes == 4);
            break;
          case double_op_t:
            assert(nbytes >= 4);
            break;  // account for double DEMOTING
          default:
            abort();
        }
        unsigned addr = constant->get_address() + nbytes_written;
        assert(addr + nbytes < min_gaddr);

        gpu->get_global_memory()->write(
            addr, nbytes, &value, NULL,
            NULL);  // assume little endian (so u8 is the first byte in u32)
        nc_bytes += nbytes;
        nbytes_written += nbytes;
      }
    }
  }
  printf(" done.\n");
  fflush(stdout);
  return nc_bytes;
}

kernel_info_t *cuda_runtime_api::gpgpu_cuda_ptx_sim_init_grid(
    const char *hostFun, gpgpu_ptx_sim_arg_list_t args, struct dim3 gridDim,
    struct dim3 blockDim, CUctx_st *context) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  function_info *entry = context->get_kernel(hostFun);
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  /*
  Passing a snapshot of the GPU's current texture mapping to the kernel's info
  as kernels should use texture bindings present at the time of their launch.
  */
  kernel_info_t *result =
      new kernel_info_t(gridDim, blockDim, entry, gpu->getNameArrayMapping(),
                        gpu->getNameInfoMapping());
  if (entry == NULL) {
    printf(
        "GPGPU-Sim PTX: ERROR launching kernel -- no PTX implementation found "
        "for %p\n",
        hostFun);
    abort();
  }
  unsigned argcount = args.size();
  unsigned argn = 1;
  for (gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end();
       a++) {
    entry->add_param_data(argcount - argn, &(*a));
    argn++;
  }

  entry->finalize(result->get_param_memory());
  gpgpu_ctx->func_sim->g_ptx_kernel_count++;
  fflush(stdout);

  if (g_debug_execution >= 4) {
    entry->ptx_jit_config(g_mallocPtr_Size, result->get_param_memory(),
                          (gpgpu_t *)context->get_device()->get_gpgpu(),
                          gridDim, blockDim);
  }

  return result;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
//***extra api for pytorch***

/* [한국어] CUDA 드라이버 API cuGetErrorString()의 GPGPU-Sim 구현 */
CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuGetErrorName()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuInit()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuInit(unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuDriverGetVersion()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDriverGetVersion(int *driverVersion) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t e = cudaDriverGetVersion(driverVersion);
  assert(e == cudaSuccess);
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuDeviceGet()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  int deviceI = -1;
  cudaError_t e = cudaGetDevice(&deviceI);
  assert(e == cudaSuccess);
  assert(deviceI != -1);
  *device = deviceI;
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuDeviceGetCount()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDeviceGetCount(int *count) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t e = cudaGetDeviceCount(count);
  assert(e == cudaSuccess);
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuDeviceGetName()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  assert(len >= 10);
  strcpy(name, "GPGPU-Sim");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
/*
 * [한국어]
 * CUDA 드라이버 API cuDeviceTotalMem()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  *bytes = 20000000000;  // dummy value
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */
#if (CUDART_VERSION > 5000)
/*
 * [한국어]
 * CUDA 드라이버 API cuDeviceGetAttribute()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                                      CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t e = cudaDeviceGetAttribute(pi, (cudaDeviceAttr)attrib, dev);
  assert(e == cudaSuccess);

  return CUDA_SUCCESS;
}
#endif
/*
 * [한국어]
 * CUDA 드라이버 API cuDeviceGetProperties()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuDeviceComputeCapability()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDeviceComputeCapability(int *major, int *minor,
                                           CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 7000
/*
 * [한국어]
 * CUDA 드라이버 API cuDevicePrimaryCtxRetain()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuDevicePrimaryCtxRelease()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuDevicePrimaryCtxSetFlags()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuDevicePrimaryCtxGetState()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags,
                                            int *active) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuDevicePrimaryCtxReset()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 7000 */

#if CUDART_VERSION >= 3020
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxCreate()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags,
                             CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

#if CUDART_VERSION >= 4000
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxDestroy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 4000
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxPushCurrent()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxPopCurrent()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxSetCurrent()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxGetCurrent()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxGetDevice()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxGetDevice(CUdevice *device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 7000
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxGetFlags()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxGetFlags(unsigned int *flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 7000 */
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxSynchronize()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxSynchronize(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxSetLimit()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxGetLimit()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxGetCacheConfig()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxSetCacheConfig()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 4020
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxGetSharedMemConfig()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxSetSharedMemConfig()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxSetSharedMemConfig(CUsharedconfig config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxGetApiVersion()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxGetStreamPriorityRange()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxGetStreamPriorityRange(int *leastPriority,
                                             int *greatestPriority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxAttach()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxAttach(CUcontext *pctx, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxDetach()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxDetach(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuModuleLoad()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuModuleLoadData()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuModuleLoadDataEx()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image,
                                    unsigned int numOptions,
                                    CUjit_option *options,
                                    void **optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuModuleLoadFatBinary()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuModuleUnload()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuModuleGetFunction()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                     const char *name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
/*
 * [한국어]
 * CUDA 드라이버 API cuModuleGetGlobal()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                                   CUmodule hmod, const char *name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */
/*
 * [한국어]
 * CUDA 드라이버 API cuModuleGetTexRef()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod,
                                   const char *name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuModuleGetSurfRef()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod,
                                    const char *name) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 6050
/*
 * [한국어]
 * CUDA 드라이버 API cuLinkCreate()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                              void **optionValues, CUlinkState *stateOut) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // currently do not support options or multiple CUlinkStates
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuLinkAddData()의 GPGPU-Sim 구현 (GPGPU-Sim에서 현재 미구현/미지원)
 */
CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type,
                               void *data, size_t size, const char *name,
                               unsigned int numOptions, CUjit_option *options,
                               void **optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  assert(type == CU_JIT_INPUT_PTX);
  cuda_not_implemented(__my_func__, __LINE__);
  return CUDA_ERROR_UNKNOWN;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuLinkAddFile()의 GPGPU-Sim 구현 — 실제 동작은 *Internal() 납부 함수로 전달
 */
CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                               const char *path, unsigned int numOptions,
                               CUjit_option *options, void **optionValues) {
  return cuLinkAddFileInternal(state, type, path, numOptions, options,
                               optionValues);
}
#endif

#if CUDART_VERSION >= 5050
/*
 * [한국어]
 * CUDA 드라이버 API cuLinkComplete()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuLinkComplete(CUlinkState state, void **cubinOut,
                                size_t *sizeOut) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // all cuLink* function are implemented to block until completion so nothing
  // to do here
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuLinkDestroy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuLinkDestroy(CUlinkState state) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  // currently do not support options or multiple CUlinkStates
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 5050 */

#if CUDART_VERSION >= 3020
/*
 * [한국어]
 * CUDA 드라이버 API cuMemGetInfo()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemGetInfo(size_t *free, size_t *total) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemAlloc()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemAllocPitch()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                                 size_t WidthInBytes, size_t Height,
                                 unsigned int ElementSizeBytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemFree()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemGetAddressRange()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize,
                                      CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemAllocHost()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemAllocHost(void **pp, size_t bytesize) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */
/*
 * [한국어]
 * CUDA 드라이버 API cuMemFreeHost()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemFreeHost(void *p) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemHostAlloc()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize,
                                unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
/*
 * [한국어]
 * CUDA 드라이버 API cuMemHostGetDevicePointer()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p,
                                           unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */
/*
 * [한국어]
 * CUDA 드라이버 API cuMemHostGetFlags()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemHostGetFlags(unsigned int *pFlags, void *p) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 6000
/*
 * [한국어]
 * CUDA 드라이버 API cuMemAllocManaged()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize,
                                   unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 6000 */

#if CUDART_VERSION >= 4010
/*
 * [한국어]
 * CUDA 드라이버 API cuDeviceGetByPCIBusId()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuDeviceGetPCIBusId()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuIpcGetEventHandle()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuIpcOpenEventHandle()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuIpcOpenEventHandle(CUevent *phEvent,
                                      CUipcEventHandle handle) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuIpcGetMemHandle()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuIpcOpenMemHandle()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle,
                                    unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuIpcCloseMemHandle()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 4010 */

#if CUDART_VERSION >= 6050
/*
 * [한국어]
 * CUDA 드라이버 API cuMemHostRegister()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 런타임 API cudaHostRegister()의 GPGPU-Sim 구현. 납부 구현(cudaHostRegisterInternal)에 위임
 */
__host__ cudaError_t cudaHostRegister(void *ptr, size_t size,
                                      unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaProfilerStart()의 GPGPU-Sim 구현. 납부 구현(cudaProfilerStartInternal)에 위임
 */
__host__ cudaError_t cudaProfilerStart() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return g_last_cudaError = cudaSuccess;
}

/*
 * [한국어]
 * CUDA 런타임 API cudaProfilerStop()의 GPGPU-Sim 구현. 납부 구현(cudaProfilerStopInternal)에 위임
 */
__host__ cudaError_t cudaProfilerStop() {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return g_last_cudaError = cudaSuccess;
}

#endif
#if CUDART_VERSION >= 4000
/*
 * [한국어]
 * CUDA 드라이버 API cuMemHostUnregister()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemHostUnregister(void *p) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyPeer()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                              CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 3020
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyHtoD()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyDtoH()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyDtoD()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyDtoA()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyDtoA(CUarray dstArray, size_t dstOffset,
                              CUdeviceptr srcDevice, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyAtoD()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray,
                              size_t srcOffset, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyHtoA()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyHtoA(CUarray dstArray, size_t dstOffset,
                              const void *srcHost, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyAtoH()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyAtoH(void *dstHost, CUarray srcArray, size_t srcOffset,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyAtoA()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyAtoA(CUarray dstArray, size_t dstOffset,
                              CUarray srcArray, size_t srcOffset,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy2D()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy2DUnaligned()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy3D()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

#if CUDART_VERSION >= 4000
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy3DPeer()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                               size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyPeerAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 3020
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyHtoDAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyDtoHAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyDtoDAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyHtoAAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset,
                                   const void *srcHost, size_t ByteCount,
                                   CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyAtoHAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray,
                                   size_t srcOffset, size_t ByteCount,
                                   CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy2DAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy3DAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

#if CUDART_VERSION >= 4000
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy3DPeerAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy,
                                     CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 3020
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD8()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD16()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD16(CUdeviceptr dstDevice, unsigned short us,
                             size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD32()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD2D8()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch,
                              unsigned char uc, size_t Width, size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD2D16()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch,
                               unsigned short us, size_t Width, size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD2D32()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch,
                               unsigned int ui, size_t Width, size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD8Async()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                                 size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD16Async()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD32Async()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD2D8Async()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                                   unsigned char uc, size_t Width,
                                   size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD2D16Async()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned short us, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD2D32Async()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned int ui, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuArrayCreate()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuArrayCreate(CUarray *pHandle,
                               const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuArrayGetDescriptor()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor,
                                      CUarray hArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */
/*
 * [한국어]
 * CUDA 드라이버 API cuArrayDestroy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuArrayDestroy(CUarray hArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
/*
 * [한국어]
 * CUDA 드라이버 API cuArray3DCreate()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuArray3DCreate(
    CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuArray3DGetDescriptor()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuArray3DGetDescriptor(
    CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */

#if CUDART_VERSION >= 5000

CUresult CUDAAPI
/*
 * [한국어]
 * CUDA 드라이버 API cuMipmappedArrayCreate()의 GPGPU-Sim 구현
 */
cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                       unsigned int numMipmapLevels) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMipmappedArrayGetLevel()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMipmappedArrayGetLevel(CUarray *pLevelArray,
                                          CUmipmappedArray hMipmappedArray,
                                          unsigned int level) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMipmappedArrayDestroy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 5000 */

/** @} */ /* END CUDA_MEM */

#if CUDART_VERSION >= 4000
/* [한국어] CUDA 드라이버 API cuPointerGetAttribute()의 GPGPU-Sim 구현 */
CUresult CUDAAPI cuPointerGetAttribute(void *data,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

#if CUDART_VERSION >= 8000
/*
 * [한국어]
 * CUDA 런타임 API cudaCreateTextureObject()의 GPGPU-Sim 구현. 납부 구현(cudaCreateTextureObjectInternal)에 위임 (GPGPU-Sim에서 현재 미구현/미지원)
 */
__host__ cudaError_t CUDARTAPI cudaCreateTextureObject(
    cudaTextureObject_t *pTexObject, const cudaResourceDesc *pResDesc,
    const cudaTextureDesc *pTexDesc, const cudaResourceViewDesc *pResViewDesc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cuda_not_implemented(__my_func__, __LINE__);
  return g_last_cudaError = cudaSuccess;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemPrefetchAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                    CUdevice dstDevice, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemAdvise()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemAdvise(CUdeviceptr devPtr, size_t count,
                             CUmem_advise advice, CUdevice device) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemRangeGetAttribute()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemRangeGetAttribute(void *data, size_t dataSize,
                                        CUmem_range_attribute attribute,
                                        CUdeviceptr devPtr, size_t count) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemRangeGetAttributes()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemRangeGetAttributes(void **data, size_t *dataSizes,
                                         CUmem_range_attribute *attributes,
                                         size_t numAttributes,
                                         CUdeviceptr devPtr, size_t count) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 8000 */

#if CUDART_VERSION >= 6000
/*
 * [한국어]
 * CUDA 드라이버 API cuPointerSetAttribute()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuPointerSetAttribute(const void *value,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 6000 */

#if CUDART_VERSION >= 7000
/*
 * [한국어]
 * CUDA 드라이버 API cuPointerGetAttributes()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes,
                                        CUpointer_attribute *attributes,
                                        void **data, CUdeviceptr ptr) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 7000 */

/** @} */ /* END CUDA_UNIFIED */

/* [한국어] CUDA 드라이버 API cuStreamCreate()의 GPGPU-Sim 구현 */
CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamCreateWithPriority()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *phStream,
                                            unsigned int flags, int priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamGetPriority()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamGetFlags()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamWaitEvent()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamAddCallback()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamAddCallback(CUstream hStream,
                                     CUstreamCallback callback, void *userData,
                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 6000
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamAttachMemAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                        size_t length, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 6000 */
CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamSynchronize()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 4000
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamDestroy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamDestroy(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */

/** @} */ /* END CUDA_STREAM */

/* [한국어] CUDA 드라이버 API cuEventCreate()의 GPGPU-Sim 구현 */
CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuEventRecord()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuEventQuery()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuEventQuery(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuEventSynchronize()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuEventSynchronize(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 4000
/*
 * [한국어]
 * CUDA 드라이버 API cuEventDestroy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuEventDestroy(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 4000 */
CUresult CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart,
                                    CUevent hEnd) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 8000
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamWaitValue32()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                                     cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamWriteValue32()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                                      cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamBatchMemOp()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                    CUstreamBatchMemOpParams *paramArray,
                                    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 8000 */

/** @} */ /* END CUDA_EVENT */

/* [한국어] CUDA 드라이버 API cuFuncGetAttribute()의 GPGPU-Sim 구현 */
CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                                    CUfunction hfunc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuFuncSetCacheConfig()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 4020
/*
 * [한국어]
 * CUDA 드라이버 API cuFuncSetSharedMemConfig()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuFuncSetSharedMemConfig(CUfunction hfunc,
                                          CUsharedconfig config) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif

#if CUDART_VERSION >= 4000
/*
 * [한국어]
 * CUDA 드라이버 API cuLaunchKernel()의 GPGPU-Sim 구현 — 실제 동작은 *Internal() 납부 함수로 전달
 */
CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void **kernelParams, void **extra) {
  return cuLaunchKernelInternal(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                blockDimY, blockDimZ, sharedMemBytes, hStream,
                                kernelParams, extra);
}
#endif /* CUDART_VERSION >= 4000 */

/** @} */ /* END CUDA_EXEC */

/* [한국어] CUDA 드라이버 API cuFuncSetBlockShape()의 GPGPU-Sim 구현 */
CUresult CUDAAPI cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuFuncSetSharedSize()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuParamSetSize()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuParamSeti()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuParamSetf()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuParamSetf(CUfunction hfunc, int offset, float value) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuParamSetv()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuParamSetv(CUfunction hfunc, int offset, void *ptr,
                             unsigned int numbytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuLaunch()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuLaunch(CUfunction f) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuLaunchGrid()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuLaunchGridAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuLaunchGridAsync(CUfunction f, int grid_width,
                                   int grid_height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuParamSetTexRef()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuParamSetTexRef(CUfunction hfunc, int texunit,
                                  CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/** @} */ /* END CUDA_EXEC_DEPRECATED */

#if CUDART_VERSION >= 6050

/* [한국어] CUDA 드라이버 API cuOccupancyMaxActiveBlocksPerMultiprocessor()의 GPGPU-Sim 구현 */
CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/*
 * [한국어]
 * CUDA 드라이버 API cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuOccupancyMaxPotentialBlockSize()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuOccupancyMaxPotentialBlockSizeWithFlags()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_OCCUPANCY */
#endif    /* CUDART_VERSION >= 6050 */

/* [한국어] CUDA 드라이버 API cuTexRefSetArray()의 GPGPU-Sim 구현 */
CUresult CUDAAPI cuTexRefSetArray(CUtexref hTexRef, CUarray hArray,
                                  unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefSetMipmappedArray()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefSetMipmappedArray(CUtexref hTexRef,
                                           CUmipmappedArray hMipmappedArray,
                                           unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefSetAddress()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef,
                                    CUdeviceptr dptr, size_t bytes) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefSetAddress2D()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefSetAddress2D(CUtexref hTexRef,
                                      const CUDA_ARRAY_DESCRIPTOR *desc,
                                      CUdeviceptr dptr, size_t Pitch) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */
CUresult CUDAAPI cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt,
                                   int NumPackedComponents) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefSetAddressMode()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefSetAddressMode(CUtexref hTexRef, int dim,
                                        CUaddress_mode am) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefSetFilterMode()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefSetMipmapFilterMode()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefSetMipmapFilterMode(CUtexref hTexRef,
                                             CUfilter_mode fm) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefSetMipmapLevelBias()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefSetMipmapLevelClamp()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefSetMipmapLevelClamp(CUtexref hTexRef,
                                             float minMipmapLevelClamp,
                                             float maxMipmapLevelClamp) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefSetMaxAnisotropy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefSetMaxAnisotropy(CUtexref hTexRef,
                                          unsigned int maxAniso) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefSetBorderColor()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefSetFlags()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 3020
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefGetAddress()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefGetArray()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefGetMipmappedArray()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray,
                                           CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefGetAddressMode()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef,
                                        int dim) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefGetFilterMode()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefGetFormat()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels,
                                   CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefGetMipmapFilterMode()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm,
                                             CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefGetMipmapLevelBias()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefGetMipmapLevelClamp()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp,
                                             float *pmaxMipmapLevelClamp,
                                             CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefGetMaxAnisotropy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefGetBorderColor()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefGetFlags()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefCreate()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefCreate(CUtexref *pTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefDestroy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefDestroy(CUtexref hTexRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuSurfRefSetArray()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuSurfRefGetArray()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_SURFREF */

#if CUDART_VERSION >= 5000
CUresult CUDAAPI
/* [한국어] CUDA 드라이버 API cuTexObjectCreate()의 GPGPU-Sim 구현 */
cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc,
                  const CUDA_TEXTURE_DESC *pTexDesc,
                  const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexObjectDestroy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexObjectDestroy(CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexObjectGetResourceDesc()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                            CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexObjectGetTextureDesc()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc,
                                           CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuTexObjectGetResourceViewDesc()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexObjectGetResourceViewDesc(
    CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_TEXOBJECT */

/* [한국어] CUDA 드라이버 API cuSurfObjectCreate()의 GPGPU-Sim 구현 */
CUresult CUDAAPI cuSurfObjectCreate(CUsurfObject *pSurfObject,
                                    const CUDA_RESOURCE_DESC *pResDesc) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuSurfObjectDestroy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuSurfObjectDestroy(CUsurfObject surfObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuSurfObjectGetResourceDesc()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                             CUsurfObject surfObject) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 5000 */

#if CUDART_VERSION >= 4000
/*
 * [한국어]
 * CUDA 드라이버 API cuDeviceCanAccessPeer()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                                       CUdevice peerDev) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuDeviceGetP2PAttribute()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuDeviceGetP2PAttribute(int *value,
                                         CUdevice_P2PAttribute attrib,
                                         CUdevice srcDevice,
                                         CUdevice dstDevice) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxEnablePeerAccess()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peerContext,
                                       unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxDisablePeerAccess()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxDisablePeerAccess(CUcontext peerContext) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_PEER_ACCESS */
#endif    /* CUDART_VERSION >= 4000 */

/* [한국어] CUDA 드라이버 API cuGraphicsUnregisterResource()의 GPGPU-Sim 구현 */
CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuGraphicsSubResourceGetMappedArray()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuGraphicsSubResourceGetMappedArray(
    CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex,
    unsigned int mipLevel) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#if CUDART_VERSION >= 5000
/*
 * [한국어]
 * CUDA 드라이버 API cuGraphicsResourceGetMappedMipmappedArray()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuGraphicsResourceGetMappedMipmappedArray(
    CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

#endif /* CUDART_VERSION >= 5000 */

#if CUDART_VERSION >= 3020
/*
 * [한국어]
 * CUDA 드라이버 API cuGraphicsResourceGetMappedPointer()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(
    CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION >= 3020 */
CUresult CUDAAPI cuGraphicsResourceSetMapFlags(CUgraphicsResource resource,
                                               unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuGraphicsMapResources()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                        CUgraphicsResource *resources,
                                        CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuGraphicsUnmapResources()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                          CUgraphicsResource *resources,
                                          CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

/** @} */ /* END CUDA_GRAPHICS */

/* [한국어] CUDA 드라이버 API cuGetExportTable()의 GPGPU-Sim 구현 */
CUresult CUDAAPI cuGetExportTable(const void **ppExportTable,
                                  const CUuuid *pExportTableId) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  cudaError_t e = cudaGetExportTable(ppExportTable, pExportTableId);
  assert(e == cudaSuccess);
  return CUDA_SUCCESS;
}

#if defined(CUDART_VERSION_INTERNAL) || \
    (CUDART_VERSION >= 4000 && CUDART_VERSION < 6050)
/*
 * [한국어]
 * CUDA 드라이버 API cuMemHostRegister()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* defined(CUDART_VERSION_INTERNAL) || (CUDART_VERSION >= 4000 && \
          CUDART_VERSION < 6050) */

#if defined(CUDART_VERSION_INTERNAL) || \
    (CUDART_VERSION >= 5050 && CUDART_VERSION < 6050)
/*
 * [한국어]
 * CUDA 드라이버 API cuLinkCreate()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                              void **optionValues, CUlinkState *stateOut) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuLinkAddData()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuLinkAddData(CUlinkState state, CUjitInputType type,
                               void *data, size_t size, const char *name,
                               unsigned int numOptions, CUjit_option *options,
                               void **optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuLinkAddFile()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuLinkAddFile(CUlinkState state, CUjitInputType type,
                               const char *path, unsigned int numOptions,
                               CUjit_option *options, void **optionValues) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION_INTERNAL || (CUDART_VERSION >= 5050 && CUDART_VERSION \
          < 6050) */

#if defined(CUDART_VERSION_INTERNAL) || \
    (CUDART_VERSION >= 3020 && CUDART_VERSION < 4010)
/*
 * [한국어]
 * CUDA 드라이버 API cuTexRefSetAddress2D_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuTexRefSetAddress2D_v2(CUtexref hTexRef,
                                         const CUDA_ARRAY_DESCRIPTOR *desc,
                                         CUdeviceptr dptr, size_t Pitch) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION_INTERNAL || (CUDART_VERSION >= 3020 && CUDART_VERSION \
          < 4010) */

#if defined(CUDART_VERSION_INTERNAL) || CUDART_VERSION < 4000
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxDestroy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxPopCurrent()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuCtxPushCurrent()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamDestroy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamDestroy(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuEventDestroy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuEventDestroy(CUevent hEvent) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif /* CUDART_VERSION_INTERNAL || CUDART_VERSION < 4000 */

#if defined(CUDART_VERSION_INTERNAL)
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyHtoD_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyDtoH_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyDtoD_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyDtoA_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset,
                                 CUdeviceptr srcDevice, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyAtoD_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray,
                                 size_t srcOffset, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyHtoA_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset,
                                 const void *srcHost, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyAtoH_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray,
                                 size_t srcOffset, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyAtoA_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset,
                                 CUarray srcArray, size_t srcOffset,
                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyHtoAAsync_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset,
                                      const void *srcHost, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyAtoHAsync_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray,
                                      size_t srcOffset, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy2D_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy2DUnaligned_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy3D_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyHtoDAsync_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                                      const void *srcHost, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyDtoHAsync_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                                      size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyDtoDAsync_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice,
                                      CUdeviceptr srcDevice, size_t ByteCount,
                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy2DAsync_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy,
                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy3DAsync_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy,
                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD8_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc,
                               size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD16_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us,
                                size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD32_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui,
                                size_t N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD2D8_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                 unsigned char uc, size_t Width,
                                 size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD2D16_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                  unsigned short us, size_t Width,
                                  size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD2D32_v2()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch,
                                  unsigned int ui, size_t Width,
                                  size_t Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                               size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyPeer()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                              CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpyPeerAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy3DPeer()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy3DPeerAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy,
                                     CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD8Async()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                                 size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD16Async()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD32Async()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                                  size_t N, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD2D8Async()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                                   unsigned char uc, size_t Width,
                                   size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD2D16Async()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned short us, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemsetD2D32Async()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                                    unsigned int ui, size_t Width,
                                    size_t Height, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamGetPriority()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamGetFlags()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamWaitEvent()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamAddCallback()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamAddCallback(CUstream hStream,
                                     CUstreamCallback callback, void *userData,
                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamAttachMemAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                        size_t length, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamQuery()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamQuery(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamSynchronize()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuEventRecord()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuLaunchKernel()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void **kernelParams, void **extra) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuGraphicsMapResources()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                        CUgraphicsResource *resources,
                                        CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuGraphicsUnmapResources()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                          CUgraphicsResource *resources,
                                          CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuMemPrefetchAsync()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                    CUdevice dstDevice, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamWriteValue32()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                                      cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamWaitValue32()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                                     cuuint32_t value, unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuStreamBatchMemOp()의 GPGPU-Sim 구현
 */
CUresult CUDAAPI cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                    CUstreamBatchMemOpParams *paramArray,
                                    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
#endif
/*
 * [한국어]
 * CUDA 드라이버 API cuProfilerInitialize()의 GPGPU-Sim 구현
 */
CUresult cuProfilerInitialize(const char *configFile, const char *outputFile,
                              CUoutput_mode outputMode) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuProfilerStart()의 GPGPU-Sim 구현
 */
CUresult cuProfilerStart(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
/*
 * [한국어]
 * CUDA 드라이버 API cuProfilerStop()의 GPGPU-Sim 구현
 */
CUresult cuProfilerStop(void) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

//_ptds

extern "C" CUresult CUDAAPI cuMemcpy_ptds(CUdeviceptr dst, CUdeviceptr src,
                                          size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyPeer_ptds(CUdeviceptr dstDevice,
                                              CUcontext dstContext,
                                              CUdeviceptr srcDevice,
                                              CUcontext srcContext,
                                              size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyHtoD_v2_ptds(CUdeviceptr dstDevice,
                                                 const void *srcHost,
                                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoH_v2_ptds(void *dstHost,
                                                 CUdeviceptr srcDevice,
                                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoD_v2_ptds(CUdeviceptr dstDevice,
                                                 CUdeviceptr srcDevice,
                                                 size_t ByteCount) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy2DUnaligned_v2_ptds()의 GPGPU-Sim 구현
 */
cuMemcpy2DUnaligned_v2_ptds(const CUDA_MEMCPY2D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy3D_v2_ptds(const CUDA_MEMCPY3D *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy3DPeer_ptds()의 GPGPU-Sim 구현
 */
cuMemcpy3DPeer_ptds(const CUDA_MEMCPY3D_PEER *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD8_v2_ptds(CUdeviceptr dstDevice,
                                               unsigned char uc,
                                               unsigned int N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD16_v2_ptds(CUdeviceptr dstDevice,
                                                unsigned short us,
                                                unsigned int N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD32_v2_ptds(CUdeviceptr dstDevice,
                                                unsigned int ui,
                                                unsigned int N) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D8_v2_ptds(CUdeviceptr dstDevice,
                                                 unsigned int dstPitch,
                                                 unsigned char uc,
                                                 unsigned int Width,
                                                 unsigned int Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D16_v2_ptds(CUdeviceptr dstDevice,
                                                  unsigned int dstPitch,
                                                  unsigned short us,
                                                  unsigned int Width,
                                                  unsigned int Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D32_v2_ptds(CUdeviceptr dstDevice,
                                                  unsigned int dstPitch,
                                                  unsigned int ui,
                                                  unsigned int Width,
                                                  unsigned int Height) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

//_ptsz
extern "C" CUresult CUDAAPI
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy3DPeer_ptsz()의 GPGPU-Sim 구현
 */
cuMemcpy3DPeer_ptsz(const CUDA_MEMCPY3D_PEER *pCopy) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr dst, CUdeviceptr src,
                                               size_t ByteCount,
                                               CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemcpyPeerAsync_ptsz(
    CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
    CUcontext srcContext, size_t ByteCount, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyHtoAAsync_v2_ptsz(CUarray dstArray,
                                                      size_t dstOffset,
                                                      const void *srcHost,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyAtoHAsync_v2_ptsz(void *dstHost,
                                                      CUarray srcArray,
                                                      size_t srcOffset,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr dstDevice,
                                                      const void *srcHost,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2_ptsz(void *dstHost,
                                                      CUdeviceptr srcDevice,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2_ptsz(CUdeviceptr dstDevice,
                                                      CUdeviceptr srcDevice,
                                                      size_t ByteCount,
                                                      CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy2DAsync_v2_ptsz(const CUDA_MEMCPY2D *pCopy,
                                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpy3DAsync_v2_ptsz(const CUDA_MEMCPY3D *pCopy,
                                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI
/*
 * [한국어]
 * CUDA 드라이버 API cuMemcpy3DPeerAsync_ptsz()의 GPGPU-Sim 구현
 */
cuMemcpy3DPeerAsync_ptsz(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemsetD8Async_ptsz(CUdeviceptr dstDevice,
                                                 unsigned char uc, size_t N,
                                                 CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemsetD2D8Async_ptsz(CUdeviceptr dstDevice,
                                                   size_t dstPitch,
                                                   unsigned char uc,
                                                   size_t Width, size_t Height,
                                                   CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuLaunchKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams, void **extra) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuEventRecord_ptsz(CUevent hEvent,
                                               CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamWriteValue32_ptsz(CUstream stream,
                                                      CUdeviceptr addr,
                                                      cuuint32_t value,
                                                      unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamWaitValue32_ptsz(CUstream stream,
                                                     CUdeviceptr addr,
                                                     cuuint32_t value,
                                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamBatchMemOp_ptsz(
    CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray,
    unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamGetPriority_ptsz(CUstream hStream,
                                                     int *priority) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamGetFlags_ptsz(CUstream hStream,
                                                  unsigned int *flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamWaitEvent_ptsz(CUstream hStream,
                                                   CUevent hEvent,
                                                   unsigned int Flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamAddCallback_ptsz(CUstream hStream,
                                                     CUstreamCallback callback,
                                                     void *userData,
                                                     unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamQuery_ptsz(CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamAttachMemAsync_ptsz(CUstream hStream,
                                                        CUdeviceptr dptr,
                                                        size_t length,
                                                        unsigned int flags) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsMapResources_ptsz(
    unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsUnmapResources_ptsz(
    unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemPrefetchAsync_ptsz(CUdeviceptr devPtr,
                                                    size_t count,
                                                    CUdevice dstDevice,
                                                    CUstream hStream) {
  if (g_debug_execution >= 3) {
    announce_call(__my_func__);
  }
  printf("WARNING: this function has not been implemented yet.");
  return CUDA_SUCCESS;
}
