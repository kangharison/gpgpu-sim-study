/*
 * [한국어 설명] CUDA GPU 수학 함수 호스트 에뮬레이션 구현 (cuda-math.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 CUDA GPU `__device__` 수학 함수들을 호스트(CPU) 환경에서 소프트웨어로 에뮬레이션하기
 * 위한 헤더이다. GPGPU-Sim은 GPU 커널을 CPU에서 기능 시뮬레이션(functional simulation)으로
 * 실행하는데, 이때 PTX 명령어가 참조하는 CUDA 내장 수학 함수들(예: __int2float_rn, __saturatef)이
 * 호스트 컴파일러에서 정의되어 있어야 한다. 이 파일은 그 정의를 제공함으로써 기능 시뮬레이터가
 * GPU와 동일한 의미론적 결과(반올림 모드 포함)를 CPU에서 재현할 수 있도록 한다.
 * CUDA 런타임 버전(CUDART_VERSION)에 따라 두 가지 구현 경로가 제공되며,
 * 최신 경로(>=3000)는 C99 fenv.h를 활용해 IEEE 754 반올림 모드를 CPU FPU에 임시 적용한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 실행 흐름에서 이 파일은 기능 시뮬레이션 계층(cuda-sim/)의 일부이다.
 *
 *   CUDA Application
 *     → libcuda (런타임 인터셉트)
 *         → gpgpusim_entrypoint.cc (시뮬레이터 진입)
 *             → cuda-sim/ptx_ir.cc, instructions.cc (PTX 명령어 시맨틱 실행)
 *                 → [이 파일] cuda-math.h (수학 내장 함수 에뮬레이션)
 *
 * 실행 컨텍스트: 호스트 유저스페이스 — CPU 단일 스레드에서 PTX 명령어를 해석하고 실행.
 * GPU 디바이스 코드가 아니며, 사이클-레벨 타이밍 모델(gpgpu-sim/)과는 무관하다.
 * instructions.cc가 PTX 수학 명령어(예: CVT.RN.F32.S32)를 처리할 때 이 헤더의 함수를 호출한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존(이 파일이 사용하는 것):
 *   - <cmath>: std::isnan, powf, truncf, nearbyintf, floorf, ceilf 등 표준 C 수학 함수
 *   - <fenv.h>: fegetround(), fesetround(), FE_TONEAREST 등 C99 FP 환경 제어 API
 *   - <device_types.h>: cudaRoundMode enum (cudaRoundNearest/Zero/MinInf/PosInf) 정의
 *   - <math_functions.h>: NVIDIA CUDA 수학 함수 선언/정의 (매크로 트릭으로 활성화)
 * 피의존(이 파일을 사용하는 것):
 *   - cuda-sim/instructions.cc: PTX CVT/FMA/MUFU 등 수학 명령어 시맨틱 구현 시 참조
 *   - cuda-sim/ptx_ir.cc: CUDA 내장 함수 심볼 해석 과정에서 간접 참조
 * 데이터 흐름:
 *   PTX 명령어의 소스 레지스터 값(정수 또는 float) → 이 헤더의 에뮬레이션 함수 →
 *   지정된 반올림 모드가 적용된 float 결과 → 목적지 레지스터에 저장
 *
 * === 주요 함수/구조체 요약 ===
 * __int2float_rn/rz/ru/rd  : 32비트 정수 → float 변환, 각 반올림 모드별 4종
 * __uint2float_rn/rz/ru/rd : 32비트 부호없는 정수 → float 변환, 4종
 * __ll2float_rn/rz/ru/rd   : 64비트 정수(long long) → float 변환, 4종
 * __ull2float_rn/rz/ru/rd  : 64비트 부호없는 정수 → float 변환, 4종
 * float2int / float2uint   : float → 정수/부호없는 정수 변환, cudaRoundMode 인자로 분기
 * __saturatef              : float를 [0.0, 1.0] 범위로 클램프 (NaN → 0.0)
 * __powf                   : float 거듭제곱 (powf 래퍼)
 * __signbitd (macOS 전용)  : double의 부호 비트 추출 (Mac GCC 누락 함수 보완)
 * isnanf (macOS 전용)      : float NaN 판별 (Mac GCC 누락 함수 보완)
 */

// This file created from vector_types.h distributed with CUDA 1.1
// (see original copyright notice below)
//
// Changes Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
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

/* [한국어] 인클루드 가드: 이 헤더가 여러 번 포함되어 중복 정의가 발생하지 않도록 방지한다.
 * GPGPU-Sim 빌드 시 다수의 .cc 파일이 cuda-math.h를 포함할 수 있어 필수적이다. */
#ifndef CUDA_MATH
#define CUDA_MATH

/* [한국어] 표준 C++ 수학 라이브러리를 포함한다.
 * std::isnan, powf, truncf, nearbyintf, floorf, ceilf 등 에뮬레이션 구현에 필요한
 * 호스트 수학 함수들을 제공한다. GPU 함수가 아니라 CPU에서 실행되는 일반 C 함수이다. */
#include <cmath>

// cuda math implementations
/* [한국어] C++ 표준 헤더 또는 일부 플랫폼 헤더(특히 Windows의 <windows.h>)는 max/min을
 * 매크로로 정의하는 경우가 있다. CUDA 헤더(math_functions.h 등) 내부에서 max/min이
 * 함수 이름으로 사용되면 매크로 치환으로 인해 컴파일 오류가 발생한다.
 * 이를 방지하기 위해 네임스페이스 진입 전에 매크로 정의를 해제한다. */
#undef max
#undef min

/* [한국어] 모든 CUDA 수학 에뮬레이션 구현을 cuda_math 네임스페이스로 격리한다.
 * 이렇게 함으로써 GPGPU-Sim 내부 심볼과 NVIDIA CUDA 헤더의 심볼이 충돌하는 것을
 * 방지한다. instructions.cc 등에서 이 함수들을 호출할 때는 cuda_math:: 접두사를 쓰거나
 * using namespace cuda_math; 선언을 통해 사용한다. */
namespace cuda_math {

/* [한국어] __attribute__((...)) 매크로를 빈 정의로 재정의하여 경고를 억제한다.
 * NVIDIA의 math_functions.h 내부에는 __attribute__((device)), __attribute__((host)) 등
 * GCC/Clang 확장 속성이 붙어 있는데, 호스트 컴파일러 환경에서 CUDA 컴파일러(__nvcc__)
 * 없이 컴파일하면 이 속성들이 인식되지 않아 경고 또는 오류가 발생한다.
 * 빈 매크로로 override하면 속성 자체가 전처리 단계에서 제거된다. */
#define __attribute__(a)  // to remove warnings inside math_functions.h

/* [한국어] <climits> 또는 플랫폼 헤더에서 정의된 INT_MAX 매크로를 해제한다.
 * CUDA의 device_types.h나 math_functions.h 내부에서 INT_MAX를 자체적으로
 * 재정의하거나 사용하는 경우 기존 정의와 충돌이 발생할 수 있기 때문이다. */
#undef INT_MAX

/* [한국어] CUDART_VERSION이 3000 미만인 경우의 레거시(구형 CUDA 2.x) 구현 경로.
 * CUDA 2.x는 벡터 타입(int4, float4 등)을 별도 헤더 없이 직접 정의해야 했으며,
 * math_functions.h의 포함 방식도 달랐다. GPGPU-Sim이 구 버전 CUDA와 호환되도록
 * 유지하기 위한 분기이다. 현재 대부분의 빌드 환경은 CUDART_VERSION >= 3000이다. */
#if CUDART_VERSION < 3000
// DEVICE_BUILTIN
/* [한국어] CUDA의 4-컴포넌트 32비트 정수 벡터 타입.
 * GPU PTX에서는 int4 타입이 하나의 레지스터 그룹(4개 레지스터)에 매핑된다.
 * CUDA 2.x 헤더에서는 이 타입이 vector_types.h에 정의되어 있었으나,
 * GPGPU-Sim이 구형 환경에서 빌드될 때 해당 헤더가 없을 수 있어 직접 정의한다. */
struct int4 {
  int x;
  /* [한국어] CUDA 벡터 타입의 첫 번째 컴포넌트 (레인 0에 해당).
   * PTX에서 .x 접미사로 접근하며, threadIdx.x 등 1D 좌표 값을 담기도 한다.
   * 설정자: float4/int4를 리턴하는 PTX 명령어 또는 CUDA 커널 코드.
   * 읽는 자: PTX 명령어의 소스 오퍼랜드로 컴포넌트 단위 접근.
   * 값 범위: 전 범위의 int (-2^31 ~ 2^31-1).
   * 동기화: 각 스레드가 독립적인 레지스터 파일을 가지므로 동기화 불필요. */
  int y;
  /* [한국어] CUDA 벡터 타입의 두 번째 컴포넌트 (레인 1에 해당).
   * .y 접미사로 접근하며, 2D 그리드에서 threadIdx.y 값을 담기도 한다.
   * 설정자: int4 초기화 또는 PTX 벡터 명령어.
   * 읽는 자: PTX 소스 오퍼랜드, struct 멤버 접근.
   * 값 범위: 전 범위의 int.
   * 동기화: 스레드별 독립 레지스터이므로 불필요. */
  int z;
  /* [한국어] CUDA 벡터 타입의 세 번째 컴포넌트 (레인 2에 해당).
   * .z 접미사로 접근하며, 3D 그리드에서 threadIdx.z 값을 담기도 한다.
   * 설정자: int4 초기화 또는 PTX 벡터 명령어.
   * 읽는 자: PTX 소스 오퍼랜드.
   * 값 범위: 전 범위의 int.
   * 동기화: 스레드별 독립 레지스터이므로 불필요. */
  int w;
  /* [한국어] CUDA 벡터 타입의 네 번째 컴포넌트 (레인 3에 해당).
   * .w 접미사로 접근하며, 동차 좌표계(homogeneous coordinates)의 w 성분이나
   * 패딩 값으로 사용되기도 한다.
   * 설정자: int4 초기화 또는 PTX 벡터 명령어.
   * 읽는 자: PTX 소스 오퍼랜드.
   * 값 범위: 전 범위의 int.
   * 동기화: 스레드별 독립 레지스터이므로 불필요. */
};

/* [한국어] CUDA의 4-컴포넌트 32비트 부호없는 정수 벡터 타입.
 * PTX에서 u32x4 타입에 해당하며, 부호없는 인덱스나 비트패턴 조작에 사용된다.
 * int4와 동일한 구조이나 각 필드가 unsigned int임이 차이점이다. */
struct uint4 {
  unsigned int x;
  /* [한국어] 부호없는 정수 벡터의 첫 번째 컴포넌트.
   * 설정자: uint4 초기화 또는 atomicAdd 등 부호없는 산술 PTX 명령어.
   * 읽는 자: 텍스처 좌표, 비트 마스크 연산 등에서 .x 접미사로 접근.
   * 값 범위: 0 ~ 2^32-1 (전 범위의 unsigned int).
   * 동기화: 스레드별 독립 레지스터이므로 불필요. */
  unsigned int y;
  /* [한국어] 부호없는 정수 벡터의 두 번째 컴포넌트.
   * 설정자: uint4 초기화.
   * 읽는 자: .y 접미사 접근.
   * 값 범위: 0 ~ 2^32-1.
   * 동기화: 불필요. */
  unsigned int z;
  /* [한국어] 부호없는 정수 벡터의 세 번째 컴포넌트.
   * 설정자: uint4 초기화.
   * 읽는 자: .z 접미사 접근.
   * 값 범위: 0 ~ 2^32-1.
   * 동기화: 불필요. */
  unsigned int w;
  /* [한국어] 부호없는 정수 벡터의 네 번째 컴포넌트.
   * 설정자: uint4 초기화.
   * 읽는 자: .w 접미사 접근.
   * 값 범위: 0 ~ 2^32-1.
   * 동기화: 불필요. */
};

/* [한국어] CUDA의 4-컴포넌트 단정밀도 부동소수점 벡터 타입.
 * GLSL/HLSL의 vec4와 유사하며, CUDA 그래픽스/컴퓨팅 커널에서 RGBA 색상,
 * 3D 좌표(동차), 4차원 부동소수점 데이터를 하나의 레지스터 그룹으로 다룰 때 사용된다. */
struct float4 {
  float x;
  /* [한국어] 단정밀도 float 벡터의 첫 번째 컴포넌트 (R 채널 또는 X 좌표).
   * 설정자: float4 초기화, PTX fadd/fmul 등 연산 결과.
   * 읽는 자: .x 멤버 접근, 텍스처 샘플링 결과 소비.
   * 값 범위: IEEE 754 단정밀도 부동소수점 전 범위 (NaN, Inf 포함).
   * 동기화: 스레드별 독립 레지스터이므로 불필요. */
  float y;
  /* [한국어] 단정밀도 float 벡터의 두 번째 컴포넌트 (G 채널 또는 Y 좌표).
   * 설정자: float4 초기화 또는 PTX 벡터 연산.
   * 읽는 자: .y 멤버 접근.
   * 값 범위: IEEE 754 단정밀도 전 범위.
   * 동기화: 불필요. */
  float z;
  /* [한국어] 단정밀도 float 벡터의 세 번째 컴포넌트 (B 채널 또는 Z 좌표).
   * 설정자: float4 초기화 또는 PTX 벡터 연산.
   * 읽는 자: .z 멤버 접근.
   * 값 범위: IEEE 754 단정밀도 전 범위.
   * 동기화: 불필요. */
  float w;
  /* [한국어] 단정밀도 float 벡터의 네 번째 컴포넌트 (A 채널 또는 동차 좌표 W).
   * 그래픽스 파이프라인에서 alpha 값이나 원근 분할용 w 좌표로 쓰이며,
   * 순수 컴퓨팅에서는 네 번째 데이터 차원으로 사용된다.
   * 설정자: float4 초기화.
   * 읽는 자: .w 멤버 접근.
   * 값 범위: IEEE 754 단정밀도 전 범위.
   * 동기화: 불필요. */
};

/* [한국어] CUDA의 2-컴포넌트 단정밀도 부동소수점 벡터 타입.
 * float4의 2차원 버전으로, 2D 텍스처 좌표(u, v), 복소수(실수+허수),
 * 또는 2D 벡터 연산에 사용된다. 메모리 정렬 측면에서 8바이트이다. */
struct float2 {
  float x;
  /* [한국어] 2D float 벡터의 첫 번째 컴포넌트 (U 텍스처 좌표 또는 실수부).
   * 설정자: float2 초기화, 2D 좌표 계산 결과.
   * 읽는 자: .x 멤버 접근, 텍스처 샘플러에 전달되는 좌표.
   * 값 범위: IEEE 754 단정밀도 전 범위.
   * 동기화: 스레드별 독립이므로 불필요. */
  float y;
  /* [한국어] 2D float 벡터의 두 번째 컴포넌트 (V 텍스처 좌표 또는 허수부).
   * 설정자: float2 초기화.
   * 읽는 자: .y 멤버 접근.
   * 값 범위: IEEE 754 단정밀도 전 범위.
   * 동기화: 불필요. */
};

/* [한국어] CUDA 벡터 타입에 대한 C 스타일 typedef 정의.
 * C++에서는 struct 태그만으로도 타입으로 사용할 수 있지만,
 * C 코드 호환성을 위해 typedef를 별도로 선언한다.
 * CUDA 1.1 시절의 vector_types.h 스타일을 그대로 유지한 것이다. */
typedef struct int4 int4;    /* [한국어] int4 구조체의 C 스타일 타입 별칭 */
typedef struct uint4 uint4;  /* [한국어] uint4 구조체의 C 스타일 타입 별칭 */
typedef struct float4 float4;/* [한국어] float4 구조체의 C 스타일 타입 별칭 */
typedef struct float2 float2;/* [한국어] float2 구조체의 C 스타일 타입 별칭 */

/* [한국어] rsqrtf (역제곱근: 1/sqrt(x)) 함수의 외부 선언.
 * CUDA 2.3 beta에서 추가된 함수로, GPU에서는 하드웨어 명령어로 구현되지만
 * 호스트 에뮬레이션 환경에서는 libm의 rsqrtf를 사용한다.
 * extern 선언만 있고 이 파일에서 정의하지 않으므로, 링크 시 libm이 제공해야 한다. */
extern float rsqrtf(float);  // CUDA 2.3 beta

/* [한국어] CUDA 부동소수점 수학 함수들을 활성화하는 매크로를 정의한다.
 * 뒤에 포함되는 math_functions.h가 이 매크로의 존재 여부를 확인하여
 * float 버전의 수학 함수 정의(예: sinf, cosf, expf)를 내보낼지 결정한다. */
#define CUDA_FLOAT_MATH_FUNCTIONS

/* [한국어] CUDA 내부 컴파일 진입 매크로를 정의하여 math_functions.h의 내용을 활성화한다.
 * math_functions.h는 __CUDA_INTERNAL_COMPILATION__이 정의되어 있을 때만
 * 실제 함수 정의(선언이 아닌 구현)를 출력하도록 설계되어 있다.
 * 시뮬레이터가 이 매크로를 임시로 정의하고 헤더를 포함한 후 즉시 해제한다. */
#include <device_types.h>
#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.h>   /* [한국어] CUDA 수학 함수 정의를 포함한다. __CUDA_INTERNAL_COMPILATION__ 매크로가 활성화된 상태이므로 실제 구현 코드가 삽입된다. */
#undef __CUDA_INTERNAL_COMPILATION__ /* [한국어] 매크로를 즉시 해제하여 이후 코드에서 CUDA 내부 컴파일 모드가 켜진 것처럼 오동작하지 않도록 한다. */
#undef __attribute__          /* [한국어] 이전에 빈 매크로로 재정의했던 __attribute__를 해제하여 이후 코드에서는 컴파일러 본래의 __attribute__ 의미를 복원한다. */

/*
 * [한국어]
 * float2int - float를 지정된 반올림 모드로 32비트 정수로 변환 (CUDA 2.x 경로)
 *
 * @a   : 변환할 float 값. PTX CVT 명령어의 소스 레지스터 값에 해당.
 * @mode: cudaRoundMode enum — 반올림 방식을 지정 (cudaRoundZero/Nearest/MinInf/PosInf).
 * @return: 반올림 모드가 적용된 int 값. PTX 목적지 레지스터에 저장된다.
 *
 * CUDA 2.x 경로에서 float→int 변환을 처리하기 위한 함수.
 * 실제 구현은 math_functions.h 내부의 __internal_float2uint를 재사용한다
 * (구 버전에서는 __internal_float2uint가 int와 uint 모두를 담당하는 통합 구현이었다).
 * 실행 컨텍스트: 호스트 CPU, PTX 기능 시뮬레이션 실행 중 단일 스레드.
 * 호출 체인: instructions.cc (CVT PTX 명령어 처리) → [이 함수] → __internal_float2uint
 */
// float to integer conversion
int float2int(float a, enum cudaRoundMode mode) {
  return __internal_float2uint(a, mode); /* [한국어] CUDA 2.x math_functions.h가 제공하는 내부 함수로 실제 변환을 위임한다. 구형 API에서는 int/uint 변환을 같은 함수가 처리했다. */
}

/*
 * [한국어]
 * float2uint - float를 지정된 반올림 모드로 32비트 부호없는 정수로 변환 (CUDA 2.x 경로)
 *
 * @a   : 변환할 float 값.
 * @mode: cudaRoundMode — 반올림 방식.
 * @return: 반올림 모드가 적용된 unsigned int 값.
 *
 * float2int의 부호없는 버전. CUDA 2.x에서는 __internal_float2uint가 양쪽을 담당한다.
 * 실행 컨텍스트: 호스트 CPU 단일 스레드, 기능 시뮬레이션 중.
 * 호출 체인: instructions.cc → [이 함수] → __internal_float2uint
 */
// float to unsigned integer conversion
unsigned int float2uint(float a, enum cudaRoundMode mode) {
  return __internal_float2uint(a, mode); /* [한국어] 부호없는 버전도 동일하게 __internal_float2uint로 위임한다. 반환 타입만 unsigned int로 다르다. */
}

/*
 * [한국어]
 * __ll2float_rz - 64비트 정수를 float으로 변환, 0 방향 절사(round toward zero)
 *
 * @a   : 변환할 long long int 값. PTX 64비트 정수 레지스터에 해당.
 * @return: 0 방향으로 절사된 float 값.
 *
 * PTX 명령어 CVT.RZ.F32.S64에 대응한다. CPU에서 이 변환을 에뮬레이션할 때,
 * 기본 float 캐스트는 현재 CPU FPU 반올림 모드(보통 FE_TONEAREST)를 따르므로
 * 원하는 반올림 모드(RZ = round toward zero)와 결과가 다를 수 있다.
 * 따라서 fegetround()로 현재 모드를 저장하고, fesetround(FE_TOWARDZERO)로
 * 변경한 후 변환을 수행하고, 원래 모드를 복원하는 패턴을 사용한다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드 — 전역 FPU 상태를 임시 변경하므로
 * 멀티스레드 환경에서는 주의가 필요하다(각 스레드는 독립 FPU 상태를 가짐).
 * 호출 체인: instructions.cc (CVT.RZ.F32.S64) → [이 함수] → fegetround/fesetround
 */
float __ll2float_rz(long long int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 CPU FPU 반올림 모드를 저장한다 (C99 fenv.h API). 일반적으로 FE_TONEAREST(0)가 기본값이다. 변환 후 이 값으로 복원한다. */
  fesetround(FE_TOWARDZERO);           /* [한국어] FPU 반올림 모드를 FE_TOWARDZERO(0 방향 절사)로 설정한다. PTX .rz 수식어(round toward zero)에 해당하는 IEEE 754 모드이다. */
  float b = a;                         /* [한국어] long long → float 암묵적 변환을 수행한다. 이 시점의 CPU는 FE_TOWARDZERO 모드이므로, 64비트 정수를 float으로 변환할 때 0 방향으로 절사된다. */
  fesetround(orig_rnd_mode);           /* [한국어] FPU 반올림 모드를 이전에 저장한 값으로 복원한다. 이후 다른 부동소수점 연산이 의도치 않게 영향받는 것을 방지한다. */
  return b;                            /* [한국어] 0 방향 절사가 적용된 float 결과를 반환한다. */
}

/*
 * [한국어]
 * __ll2float_ru - 64비트 정수를 float으로 변환, 양의 무한대 방향 올림(round up)
 *
 * @a   : 변환할 long long int 값.
 * @return: 양의 무한대 방향으로 올림된 float 값.
 *
 * PTX 명령어 CVT.RU.F32.S64에 대응한다. FE_UPWARD는 ceiling 연산으로,
 * 수학적 정확한 값보다 크거나 같은 최소 부동소수점 값을 선택한다.
 * __ll2float_rz와 동일한 save/set/convert/restore 패턴을 사용한다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RU.F32.S64) → [이 함수]
 */
float __ll2float_ru(long long int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_UPWARD);               /* [한국어] FPU 반올림 모드를 FE_UPWARD(양의 무한대 방향 올림)로 설정한다. PTX .ru 수식어(round up, toward +∞)에 해당한다. */
  float b = a;                         /* [한국어] FE_UPWARD 모드에서 long long → float 변환을 수행한다. 정확한 수학 값보다 크거나 같은 최소 float 값이 선택된다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 올림이 적용된 float 결과를 반환한다. */
}

/*
 * [한국어]
 * __ll2float_rd - 64비트 정수를 float으로 변환, 음의 무한대 방향 내림(round down)
 *
 * @a   : 변환할 long long int 값.
 * @return: 음의 무한대 방향으로 내림된 float 값.
 *
 * PTX 명령어 CVT.RD.F32.S64에 대응한다. FE_DOWNWARD는 floor 연산으로,
 * 수학적 정확한 값보다 작거나 같은 최대 부동소수점 값을 선택한다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RD.F32.S64) → [이 함수]
 */
float __ll2float_rd(long long int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_DOWNWARD);             /* [한국어] FPU 반올림 모드를 FE_DOWNWARD(음의 무한대 방향 내림)로 설정한다. PTX .rd 수식어(round down, toward -∞)에 해당한다. */
  float b = a;                         /* [한국어] FE_DOWNWARD 모드에서 long long → float 변환을 수행한다. 정확한 수학 값보다 작거나 같은 최대 float 값이 선택된다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 내림이 적용된 float 결과를 반환한다. */
}

/* [한국어] CUDART_VERSION >= 3000 경로: 현대적인 CUDA 3.x 이상에서 사용되는 구현.
 * 구형 경로와 달리 벡터 타입을 직접 정의하지 않고, 정수→float 변환 내장 함수들을
 * 반올림 모드별로 명시적으로 구현한다. fenv.h를 활용하여 CPU FPU 모드를 직접 제어한다. */
#else

/* [한국어] CUDA 부동소수점 수학 함수 그룹을 활성화하는 매크로를 정의한다.
 * math_functions.h가 float 버전의 수학 함수(sinf, cosf 등)를 포함하도록 한다. */
#define CUDA_FLOAT_MATH_FUNCTIONS

/* [한국어] CUDA 컴파일러(__nvcc__) 존재를 나타내는 매크로를 임시로 정의한다.
 * device_types.h와 math_functions.h 일부는 __CUDACC__가 정의되어 있을 때만
 * GPU 관련 타입(cudaRoundMode enum 등)과 함수를 활성화한다.
 * 시뮬레이터가 호스트 컴파일러(g++)로 빌드되므로 이 매크로를 수동으로 정의해야 한다.
 * math_functions.h 포함 후 즉시 해제한다(파일 하단의 #undef __CUDACC__ 참조). */
#define __CUDACC__

// implementing int to float intrinsics with different rounding modes
/* [한국어] cudaRoundMode enum(cudaRoundNearest/Zero/MinInf/PosInf)을 정의하는
 * CUDA 디바이스 타입 헤더를 포함한다. float2int, float2uint 함수의 mode 인자 타입으로 사용된다. */
#include <device_types.h>

/* [한국어] C99 부동소수점 환경 제어 헤더를 포함한다.
 * fegetround(): 현재 FPU 반올림 모드를 반환하는 함수 (FE_TONEAREST 등의 int 값)
 * fesetround(): FPU 반올림 모드를 설정하는 함수
 * FE_TONEAREST, FE_TOWARDZERO, FE_UPWARD, FE_DOWNWARD: 반올림 모드 상수
 * 이 헤더 없이는 CPU FPU 반올림 모드를 프로그램적으로 제어할 수 없으므로 필수이다. */
#include <fenv.h>

// 32-bit integer to float
/*
 * [한국어]
 * __int2float_rn - 32비트 부호있는 정수를 float으로 변환, 최근접 짝수 반올림(round to nearest even)
 *
 * @a   : 변환할 32비트 부호있는 정수. PTX s32 타입 레지스터 값에 해당.
 * @return: 최근접 짝수 반올림이 적용된 float 값.
 *
 * PTX 명령어 CVT.RN.F32.S32에 대응한다. .rn(round nearest)은 IEEE 754 기본 모드이며,
 * 정확한 결과에 가장 가까운 float 값을 선택한다. 두 값이 등거리일 경우 짝수 비트(ULP)를
 * 가진 쪽을 선택하는 "은행가 반올림(banker's rounding)"이다.
 * 32비트 정수는 최대 2^31-1 ≈ 2.1×10^9인데, float의 가수부는 23비트이므로
 * 약 2^24 이상의 값에서 정밀도 손실이 발생할 수 있다. 이 함수는 그 손실 방식을
 * GPU와 동일하게 FE_TONEAREST 모드로 보장한다.
 * 실행 컨텍스트: 호스트 CPU, PTX 기능 시뮬레이션 중 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RN.F32.S32 PTX 명령어) → [이 함수]
 */
float __int2float_rn(int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 CPU FPU 반올림 모드(보통 FE_TONEAREST)를 저장한다. 이 값으로 나중에 복원한다. */
  fesetround(FE_TONEAREST);            /* [한국어] FPU 반올림 모드를 FE_TONEAREST(최근접 짝수 반올림)로 명시적으로 설정한다. PTX .rn 수식어에 해당하는 IEEE 754 default 모드이다. */
  float b = a;                         /* [한국어] int → float 암묵적 변환을 수행한다. FE_TONEAREST 모드에서 가장 가까운 float 값으로 반올림된다. */
  fesetround(orig_rnd_mode);           /* [한국어] FPU 반올림 모드를 원래 저장된 값으로 복원하여 이후 코드에 영향을 주지 않는다. */
  return b;                            /* [한국어] 최근접 반올림이 적용된 float 값을 반환한다. */
}

/*
 * [한국어]
 * __int2float_rz - 32비트 부호있는 정수를 float으로 변환, 0 방향 절사(round toward zero)
 *
 * @a   : 변환할 32비트 부호있는 정수.
 * @return: 0 방향으로 절사된 float 값.
 *
 * PTX 명령어 CVT.RZ.F32.S32에 대응한다. .rz(round toward zero)는 truncate와 동일하며,
 * 양수는 floor, 음수는 ceiling 방향으로 절사된다. 정수→float에서는 소수점 이하가 없으므로
 * RN과 RZ의 차이는 float 표현 정밀도에서만 나타난다 (예: 2^24+1은 float에서 표현 불가).
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RZ.F32.S32) → [이 함수]
 */
float __int2float_rz(int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_TOWARDZERO);           /* [한국어] FPU 반올림 모드를 FE_TOWARDZERO(0 방향 절사)로 설정한다. PTX .rz 수식어에 해당한다. */
  float b = a;                         /* [한국어] FE_TOWARDZERO 모드에서 int → float 변환을 수행한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 0 방향 절사가 적용된 float 값을 반환한다. */
}

/*
 * [한국어]
 * __int2float_ru - 32비트 부호있는 정수를 float으로 변환, 양의 무한대 방향 올림(round up)
 *
 * @a   : 변환할 32비트 부호있는 정수.
 * @return: 양의 무한대 방향으로 올림된 float 값.
 *
 * PTX 명령어 CVT.RU.F32.S32에 대응한다. FE_UPWARD = ceiling 방향.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RU.F32.S32) → [이 함수]
 */
float __int2float_ru(int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_UPWARD);               /* [한국어] FPU 반올림 모드를 FE_UPWARD(양의 무한대 방향 올림)로 설정한다. PTX .ru 수식어에 해당한다. */
  float b = a;                         /* [한국어] FE_UPWARD 모드에서 int → float 변환을 수행한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 올림이 적용된 float 값을 반환한다. */
}

/*
 * [한국어]
 * __int2float_rd - 32비트 부호있는 정수를 float으로 변환, 음의 무한대 방향 내림(round down)
 *
 * @a   : 변환할 32비트 부호있는 정수.
 * @return: 음의 무한대 방향으로 내림된 float 값.
 *
 * PTX 명령어 CVT.RD.F32.S32에 대응한다. FE_DOWNWARD = floor 방향.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RD.F32.S32) → [이 함수]
 */
float __int2float_rd(int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_DOWNWARD);             /* [한국어] FPU 반올림 모드를 FE_DOWNWARD(음의 무한대 방향 내림)로 설정한다. PTX .rd 수식어에 해당한다. */
  float b = a;                         /* [한국어] FE_DOWNWARD 모드에서 int → float 변환을 수행한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 내림이 적용된 float 값을 반환한다. */
}

// 32-bit unsigned integer to float
/*
 * [한국어]
 * __uint2float_rn - 32비트 부호없는 정수를 float으로 변환, 최근접 짝수 반올림
 *
 * @a   : 변환할 32비트 부호없는 정수. PTX u32 타입 레지스터 값.
 * @return: 최근접 짝수 반올림이 적용된 float 값.
 *
 * PTX 명령어 CVT.RN.F32.U32에 대응한다. 부호없는 정수는 최대 2^32-1 ≈ 4.3×10^9이며,
 * float의 23비트 가수부로는 약 2^24 이상에서 정밀도 손실이 발생한다.
 * __int2float_rn과 동일한 패턴이지만 인자 타입이 unsigned int이다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RN.F32.U32) → [이 함수]
 */
float __uint2float_rn(unsigned int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_TONEAREST);            /* [한국어] FPU 반올림 모드를 FE_TONEAREST로 설정한다. */
  float b = a;                         /* [한국어] unsigned int → float 변환을 FE_TONEAREST 모드에서 수행한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 최근접 반올림이 적용된 float 값을 반환한다. */
}

/*
 * [한국어]
 * __uint2float_rz - 32비트 부호없는 정수를 float으로 변환, 0 방향 절사
 *
 * @a   : 변환할 32비트 부호없는 정수.
 * @return: 0 방향으로 절사된 float 값.
 *
 * PTX 명령어 CVT.RZ.F32.U32에 대응한다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RZ.F32.U32) → [이 함수]
 */
float __uint2float_rz(unsigned int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_TOWARDZERO);           /* [한국어] FPU 반올림 모드를 FE_TOWARDZERO로 설정한다. */
  float b = a;                         /* [한국어] FE_TOWARDZERO 모드에서 unsigned int → float 변환을 수행한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 0 방향 절사가 적용된 float 값을 반환한다. */
}

/*
 * [한국어]
 * __uint2float_ru - 32비트 부호없는 정수를 float으로 변환, 양의 무한대 방향 올림
 *
 * @a   : 변환할 32비트 부호없는 정수.
 * @return: 양의 무한대 방향으로 올림된 float 값.
 *
 * PTX 명령어 CVT.RU.F32.U32에 대응한다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RU.F32.U32) → [이 함수]
 */
float __uint2float_ru(unsigned int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_UPWARD);               /* [한국어] FPU 반올림 모드를 FE_UPWARD로 설정한다. */
  float b = a;                         /* [한국어] FE_UPWARD 모드에서 unsigned int → float 변환을 수행한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 올림이 적용된 float 값을 반환한다. */
}

/*
 * [한국어]
 * __uint2float_rd - 32비트 부호없는 정수를 float으로 변환, 음의 무한대 방향 내림
 *
 * @a   : 변환할 32비트 부호없는 정수.
 * @return: 음의 무한대 방향으로 내림된 float 값.
 *
 * PTX 명령어 CVT.RD.F32.U32에 대응한다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RD.F32.U32) → [이 함수]
 */
float __uint2float_rd(unsigned int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_DOWNWARD);             /* [한국어] FPU 반올림 모드를 FE_DOWNWARD로 설정한다. */
  float b = a;                         /* [한국어] FE_DOWNWARD 모드에서 unsigned int → float 변환을 수행한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 내림이 적용된 float 값을 반환한다. */
}

// 64-bit integer to float
/*
 * [한국어]
 * __ll2float_rn - 64비트 부호있는 정수를 float으로 변환, 최근접 짝수 반올림
 *
 * @a   : 변환할 long long int (64비트 부호있는 정수). PTX s64 타입 레지스터 값.
 * @return: 최근접 짝수 반올림이 적용된 float 값.
 *
 * PTX 명령어 CVT.RN.F32.S64에 대응한다. 64비트 정수는 최대 2^63-1 ≈ 9.2×10^18이며,
 * float의 23비트 가수부와의 정밀도 차이가 크므로 반올림 모드 설정이 더욱 중요하다.
 * 예를 들어 2^24+1 = 16777217을 float으로 변환하면 RN에서는 16777216.0f가 되고
 * RU에서는 16777218.0f가 된다. 이 함수는 GPU와 동일한 결과를 보장한다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RN.F32.S64) → [이 함수]
 */
float __ll2float_rn(long long int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_TONEAREST);            /* [한국어] FPU 반올림 모드를 FE_TONEAREST(최근접 짝수 반올림)로 설정한다. PTX .rn 수식어에 해당한다. */
  float b = a;                         /* [한국어] FE_TONEAREST 모드에서 long long → float 변환을 수행한다. 64비트 정수 범위에서는 가수부 정밀도(23비트) 이상의 값이 많으므로 반올림이 자주 발생한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 최근접 반올림이 적용된 float 값을 반환한다. */
}

/*
 * [한국어]
 * __ll2float_rz - 64비트 부호있는 정수를 float으로 변환, 0 방향 절사
 *
 * @a   : 변환할 long long int.
 * @return: 0 방향으로 절사된 float 값.
 *
 * PTX 명령어 CVT.RZ.F32.S64에 대응한다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RZ.F32.S64) → [이 함수]
 */
float __ll2float_rz(long long int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_TOWARDZERO);           /* [한국어] FPU 반올림 모드를 FE_TOWARDZERO로 설정한다. */
  float b = a;                         /* [한국어] FE_TOWARDZERO 모드에서 long long → float 변환을 수행한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 0 방향 절사가 적용된 float 값을 반환한다. */
}

/*
 * [한국어]
 * __ll2float_ru - 64비트 부호있는 정수를 float으로 변환, 양의 무한대 방향 올림
 *
 * @a   : 변환할 long long int.
 * @return: 양의 무한대 방향으로 올림된 float 값.
 *
 * PTX 명령어 CVT.RU.F32.S64에 대응한다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RU.F32.S64) → [이 함수]
 */
float __ll2float_ru(long long int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_UPWARD);               /* [한국어] FPU 반올림 모드를 FE_UPWARD로 설정한다. */
  float b = a;                         /* [한국어] FE_UPWARD 모드에서 long long → float 변환을 수행한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 올림이 적용된 float 값을 반환한다. */
}

/*
 * [한국어]
 * __ll2float_rd - 64비트 부호있는 정수를 float으로 변환, 음의 무한대 방향 내림
 *
 * @a   : 변환할 long long int.
 * @return: 음의 무한대 방향으로 내림된 float 값.
 *
 * PTX 명령어 CVT.RD.F32.S64에 대응한다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RD.F32.S64) → [이 함수]
 */
float __ll2float_rd(long long int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_DOWNWARD);             /* [한국어] FPU 반올림 모드를 FE_DOWNWARD로 설정한다. */
  float b = a;                         /* [한국어] FE_DOWNWARD 모드에서 long long → float 변환을 수행한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 내림이 적용된 float 값을 반환한다. */
}

// 64-bit unsigned integer to float
/*
 * [한국어]
 * __ull2float_rn - 64비트 부호없는 정수를 float으로 변환, 최근접 짝수 반올림
 *
 * @a   : 변환할 unsigned long long int (64비트 부호없는 정수). PTX u64 타입 레지스터 값.
 * @return: 최근접 짝수 반올림이 적용된 float 값.
 *
 * PTX 명령어 CVT.RN.F32.U64에 대응한다. 64비트 부호없는 정수는 최대 2^64-1 ≈ 1.8×10^19이며,
 * float 가수부(23비트)와의 정밀도 차이가 매우 크다. 특히 GPU 메모리 주소 계산이나
 * 타임스탬프 변환 등에서 이 함수가 사용된다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RN.F32.U64) → [이 함수]
 */
float __ull2float_rn(unsigned long long int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_TONEAREST);            /* [한국어] FPU 반올림 모드를 FE_TONEAREST로 설정한다. */
  float b = a;                         /* [한국어] FE_TONEAREST 모드에서 unsigned long long → float 변환을 수행한다. 64비트 범위에서는 대부분의 값에서 반올림이 발생한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 최근접 반올림이 적용된 float 값을 반환한다. */
}

/*
 * [한국어]
 * __ull2float_rz - 64비트 부호없는 정수를 float으로 변환, 0 방향 절사
 *
 * @a   : 변환할 unsigned long long int.
 * @return: 0 방향으로 절사된 float 값.
 *
 * PTX 명령어 CVT.RZ.F32.U64에 대응한다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RZ.F32.U64) → [이 함수]
 */
float __ull2float_rz(unsigned long long int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_TOWARDZERO);           /* [한국어] FPU 반올림 모드를 FE_TOWARDZERO로 설정한다. */
  float b = a;                         /* [한국어] FE_TOWARDZERO 모드에서 unsigned long long → float 변환을 수행한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 0 방향 절사가 적용된 float 값을 반환한다. */
}

/*
 * [한국어]
 * __ull2float_ru - 64비트 부호없는 정수를 float으로 변환, 양의 무한대 방향 올림
 *
 * @a   : 변환할 unsigned long long int.
 * @return: 양의 무한대 방향으로 올림된 float 값.
 *
 * PTX 명령어 CVT.RU.F32.U64에 대응한다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RU.F32.U64) → [이 함수]
 */
float __ull2float_ru(unsigned long long int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_UPWARD);               /* [한국어] FPU 반올림 모드를 FE_UPWARD로 설정한다. */
  float b = a;                         /* [한국어] FE_UPWARD 모드에서 unsigned long long → float 변환을 수행한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 올림이 적용된 float 값을 반환한다. */
}

/*
 * [한국어]
 * __ull2float_rd - 64비트 부호없는 정수를 float으로 변환, 음의 무한대 방향 내림
 *
 * @a   : 변환할 unsigned long long int.
 * @return: 음의 무한대 방향으로 내림된 float 값.
 *
 * PTX 명령어 CVT.RD.F32.U64에 대응한다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT.RD.F32.U64) → [이 함수]
 */
float __ull2float_rd(unsigned long long int a) {
  int orig_rnd_mode = fegetround();    /* [한국어] 현재 FPU 반올림 모드를 저장한다. */
  fesetround(FE_DOWNWARD);             /* [한국어] FPU 반올림 모드를 FE_DOWNWARD로 설정한다. */
  float b = a;                         /* [한국어] FE_DOWNWARD 모드에서 unsigned long long → float 변환을 수행한다. */
  fesetround(orig_rnd_mode);           /* [한국어] 반올림 모드를 원래 값으로 복원한다. */
  return b;                            /* [한국어] 내림이 적용된 float 값을 반환한다. */
}

/*
 * [한국어]
 * float2int - float를 지정된 반올림 모드로 32비트 부호있는 정수로 변환
 *
 * @a   : 변환할 float 값. PTX f32 타입 레지스터 값.
 * @mode: cudaRoundMode enum — 반올림 방식을 지정.
 *         cudaRoundZero    : 0 방향 절사 (truncate, PTX .rz)
 *         cudaRoundNearest : 최근접 짝수 반올림 (PTX .rn)
 *         cudaRoundMinInf  : 음의 무한대 방향 내림 (floor, PTX .rd)
 *         cudaRoundPosInf  : 양의 무한대 방향 올림 (ceiling, PTX .ru)
 * @return: 반올림 모드가 적용된 int 값. 오버플로우 동작은 C99 표준 함수(truncf 등)를 따른다.
 *
 * PTX 명령어 CVT.{RZ|RN|RD|RU}.S32.F32에 대응한다.
 * 반올림 모드를 CPU FPU 상태가 아닌 switch-case로 처리하는 이유:
 * C99 표준 함수(truncf, nearbyintf, floorf, ceilf)가 각 반올림 모드에 해당하는
 * 변환을 직접 제공하므로, fesetround 패턴보다 간결하고 오버헤드가 적다.
 * 정수 변환에서 float→int는 C 정수형 범위를 초과하는 경우 undefined behavior이므로,
 * 실제 GPU PTX에서도 포화(saturation) 플래그와 함께 사용하는 경우가 많다.
 * 실행 컨텍스트: 호스트 CPU, PTX 기능 시뮬레이션 중 단일 스레드.
 * 호출 체인: instructions.cc (CVT PTX 명령어) → [이 함수]
 *            또는 __internal_float2int → [이 함수]
 */
// float to integer conversion
int float2int(float a, enum cudaRoundMode mode) {
  int tmp;                             /* [한국어] 변환 결과를 임시 저장할 변수. switch-case에서 각 반올림 모드에 따라 값이 채워진다. */
  switch (mode) {                      /* [한국어] cudaRoundMode enum 값에 따라 적합한 C99 수학 함수를 선택한다. */
    case cudaRoundZero:                /* [한국어] cudaRoundZero: 0 방향 절사 (PTX .rz, truncate). 양수→floor, 음수→ceiling 방향으로 소수점 이하를 버린다. */
      tmp = truncf(a);                 /* [한국어] truncf(): 소수점 이하를 버리는 C99 함수. 예: truncf(1.7)=1.0, truncf(-1.7)=-1.0. */
      break;
    case cudaRoundNearest:             /* [한국어] cudaRoundNearest: 최근접 짝수 반올림 (PTX .rn). 현재 FPU 모드 무관하게 nearbyintf를 사용한다. */
      tmp = nearbyintf(a);             /* [한국어] nearbyintf(): 현재 FPU 반올림 모드(기본: FE_TONEAREST)를 따르는 반올림. feraiseexcept를 발생시키지 않아 신호 안전하다. */
      break;
    case cudaRoundMinInf:              /* [한국어] cudaRoundMinInf: 음의 무한대 방향 내림 (PTX .rd, floor). -∞ 방향으로 내림한다. */
      tmp = floorf(a);                 /* [한국어] floorf(): 음의 무한대 방향으로 내림하는 C99 함수. 예: floorf(1.7)=1.0, floorf(-1.2)=-2.0. */
      break;
    case cudaRoundPosInf:              /* [한국어] cudaRoundPosInf: 양의 무한대 방향 올림 (PTX .ru, ceiling). +∞ 방향으로 올림한다. */
      tmp = ceilf(a);                  /* [한국어] ceilf(): 양의 무한대 방향으로 올림하는 C99 함수. 예: ceilf(1.2)=2.0, ceilf(-1.7)=-1.0. */
      break;
    default:                           /* [한국어] 알 수 없는 반올림 모드가 전달된 경우: 시뮬레이터 내부 오류이므로 즉시 프로그램을 중단시킨다. */
      abort();                         /* [한국어] POSIX abort(): SIGABRT 신호를 발생시켜 코어 덤프와 함께 프로그램을 종료한다. 디버깅 시 발생 지점을 추적하기 위한 용도이다. */
  }
  return tmp;                          /* [한국어] 선택된 반올림 모드로 변환된 int 결과를 반환한다. */
}

/*
 * [한국어]
 * __internal_float2int - float2int의 내부 API 래퍼
 *
 * @a   : 변환할 float 값.
 * @mode: 반올림 모드 (cudaRoundMode).
 * @return: float2int(a, mode)의 결과를 그대로 반환.
 *
 * CUDA의 math_functions.h 내부에서 __internal_float2int라는 이름으로 float→int 변환을
 * 참조하는 코드가 있다. 시뮬레이터의 float2int 구현을 그 이름으로 노출하기 위한 래퍼이다.
 * 코드 중복 없이 하나의 구현(float2int)을 두 이름으로 제공하는 패턴이다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: math_functions.h 내부 코드 → [이 함수] → float2int
 */
int __internal_float2int(float a, enum cudaRoundMode mode) {
  return float2int(a, mode); /* [한국어] float2int로 직접 위임한다. 래퍼이므로 추가 로직이 없다. */
}

/*
 * [한국어]
 * float2uint - float를 지정된 반올림 모드로 32비트 부호없는 정수로 변환
 *
 * @a   : 변환할 float 값.
 * @mode: cudaRoundMode — 반올림 방식.
 * @return: 반올림 모드가 적용된 unsigned int 값.
 *
 * float2int의 부호없는(unsigned) 버전이다. PTX 명령어 CVT.{RZ|RN|RD|RU}.U32.F32에 대응한다.
 * 음수 float를 unsigned int로 변환하는 경우 C99 표준은 undefined behavior이지만,
 * GPU PTX는 0으로 포화(saturate)하는 동작이 명시되어 있다. 이 구현은 표준 C 함수를
 * 사용하므로 음수 입력 시 GPU와 정확히 동일한 결과를 보장하지 않을 수 있다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (CVT PTX 명령어) → [이 함수]
 *            또는 __internal_float2uint → [이 함수]
 */
// float to unsigned integer conversion
unsigned int float2uint(float a, enum cudaRoundMode mode) {
  unsigned int tmp;                    /* [한국어] 변환 결과를 임시 저장할 부호없는 정수 변수. */
  switch (mode) {                      /* [한국어] 반올림 모드에 따라 적합한 C99 수학 함수를 선택한다. */
    case cudaRoundZero:                /* [한국어] 0 방향 절사 (PTX .rz). */
      tmp = truncf(a);                 /* [한국어] truncf()로 소수점 이하를 버린 후 unsigned int로 변환한다. 음수 입력 시 C99 undefined behavior 주의. */
      break;
    case cudaRoundNearest:             /* [한국어] 최근접 짝수 반올림 (PTX .rn). */
      tmp = nearbyintf(a);             /* [한국어] nearbyintf()로 가장 가까운 정수로 반올림한 후 unsigned int로 변환한다. */
      break;
    case cudaRoundMinInf:              /* [한국어] 음의 무한대 방향 내림 (PTX .rd). */
      tmp = floorf(a);                 /* [한국어] floorf()로 내림한 후 unsigned int로 변환한다. */
      break;
    case cudaRoundPosInf:              /* [한국어] 양의 무한대 방향 올림 (PTX .ru). */
      tmp = ceilf(a);                  /* [한국어] ceilf()로 올림한 후 unsigned int로 변환한다. */
      break;
    default:                           /* [한국어] 알 수 없는 반올림 모드: 즉시 프로그램 종료. */
      abort();                         /* [한국어] SIGABRT로 코어 덤프와 함께 종료한다. */
  }
  return tmp;                          /* [한국어] 반올림 모드가 적용된 unsigned int 결과를 반환한다. */
}

/*
 * [한국어]
 * __internal_float2uint - float2uint의 내부 API 래퍼
 *
 * @a   : 변환할 float 값.
 * @mode: 반올림 모드 (cudaRoundMode).
 * @return: float2uint(a, mode)의 결과를 그대로 반환.
 *
 * __internal_float2int와 대칭적으로, math_functions.h 내부에서 참조하는
 * __internal_float2uint 이름으로 float2uint 구현을 노출하기 위한 래퍼이다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: math_functions.h 내부 → [이 함수] → float2uint
 */
unsigned int __internal_float2uint(float a, enum cudaRoundMode mode) {
  return float2uint(a, mode); /* [한국어] float2uint로 직접 위임한다. 추가 로직 없는 순수 래퍼이다. */
}

/*
 * [한국어]
 * fdividef - float 나눗셈 내장 함수
 *
 * @a: 피제수(분자) float 값.
 * @b: 제수(분모) float 값.
 * @return: a / b의 단정밀도 부동소수점 나눗셈 결과.
 *
 * PTX 명령어 DIV.F32 또는 CUDA 내장 함수 __fdividef(a, b)에 대응한다.
 * GPU의 __fdividef는 정밀도를 약간 낮추는 대신 빠른 하드웨어 역수 곱셈을 사용하지만,
 * 이 호스트 에뮬레이션은 단순히 C++ / 연산자를 사용하여 IEEE 754 완전 정밀도로 계산한다.
 * GPU와 호스트 결과 사이에 마지막 비트(ULP) 수준의 미세한 차이가 발생할 수 있다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (FMUL/FDIV PTX 명령어) → [이 함수]
 */
// intrinsic for division
float fdividef(float a, float b) { return (a / b); } /* [한국어] C++ / 연산자를 사용한 단순 나눗셈이다. b=0이면 ±Inf 또는 NaN이 반환되는데 이는 IEEE 754 표준 동작이다. */

/*
 * [한국어]
 * __internal_accurate_fdividef - 정밀도 보장 float 나눗셈의 내부 API 래퍼
 *
 * @a: 피제수 float 값.
 * @b: 제수 float 값.
 * @return: fdividef(a, b)의 결과 — 완전 정밀도 float 나눗셈.
 *
 * CUDA의 math_functions.h 내부에서 __internal_accurate_fdividef라는 이름으로
 * 정밀한 float 나눗셈을 참조하는 경우가 있다. fdividef와 동일한 구현이지만
 * "accurate"라는 이름으로 구별되어, GPU에서 빠른 근사 나눗셈과 구별되는 개념이다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: math_functions.h 내부 → [이 함수] → fdividef
 */
float __internal_accurate_fdividef(float a, float b) { return fdividef(a, b); } /* [한국어] fdividef로 직접 위임한다. 호스트에서는 근사 나눗셈과 정밀 나눗셈이 구별되지 않으므로 동일한 구현을 사용한다. */

/*
 * [한국어]
 * __saturatef - float 값을 [0.0, 1.0] 범위로 클램프(포화 연산)
 *
 * @a   : 클램프할 float 값. 임의의 IEEE 754 값 (NaN, -Inf, +Inf 포함 가능).
 * @return: [0.0, 1.0] 범위로 클램프된 float 값.
 *         NaN → 0.0f (GPU PTX saturate 동작과 일치)
 *         a >= 1.0f → 1.0f
 *         a <= 0.0f → 0.0f
 *         0.0f < a < 1.0f → a (변경 없음)
 *
 * PTX 명령어의 .sat(saturate) 수식어에 대응한다. GPU에서 포화 연산은 하드웨어로
 * 지원되며, FMA나 MUL 등의 결과를 [0.0, 1.0]으로 즉시 클램프한다.
 * 그래픽스 셰이더에서 색상 값이 0~1 범위를 벗어나지 않도록 하거나,
 * 시그모이드 활성화 함수 출력을 클램프하는 데 자주 사용된다.
 * NaN이 0.0으로 처리되는 이유: GPU PTX 스펙에서 saturate 연산의 NaN 동작이
 * 0.0 반환으로 정의되어 있기 때문이다. 이는 std::clamp와는 다른 동작이다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (PTX .sat 수식어 처리) → [이 함수]
 */
// intrinsic for saturate  (clamp values beyond 0 and 1)
float __saturatef(float a) {
  float b;                             /* [한국어] 클램프 결과를 저장할 임시 변수. if-else 분기에서 값이 채워진다. */
  if (std::isnan(a))                   /* [한국어] NaN(Not a Number) 판별: std::isnan은 IEEE 754 NaN 비트 패턴을 검사한다. NaN은 모든 비교 연산에서 false를 반환하므로 별도로 처리해야 한다. */
    b = 0.0f;                          /* [한국어] GPU PTX saturate 스펙에 따라 NaN 입력을 0.0으로 처리한다. */
  else if (a >= 1.0f)                  /* [한국어] 1.0 이상의 값(+Inf 포함)을 1.0으로 클램프한다. */
    b = 1.0f;                          /* [한국어] 상한(ceiling)을 1.0으로 설정한다. */
  else if (a <= 0.0f)                  /* [한국어] 0.0 이하의 값(-Inf, 음수 포함)을 0.0으로 클램프한다. */
    b = 0.0f;                          /* [한국어] 하한(floor)을 0.0으로 설정한다. -0.0f도 이 조건에 해당하여 0.0f로 변환된다. */
  else                                 /* [한국어] 0.0 < a < 1.0 범위: 값을 그대로 유지한다. */
    b = a;                             /* [한국어] 이미 [0.0, 1.0] 범위 내의 값은 변경하지 않는다. */
  return b;                            /* [한국어] 클램프된 결과를 반환한다. */
}

/*
 * [한국어]
 * __powf - float 거듭제곱 내장 함수
 *
 * @a: 밑(base) float 값.
 * @b: 지수(exponent) float 값.
 * @return: a^b의 단정밀도 부동소수점 결과.
 *
 * PTX 명령어 MUFU.{EX2, LG2} 조합 또는 CUDA 내장 함수 __powf(a, b)에 대응한다.
 * GPU의 __powf는 근사 연산(약 2 ULP 오차)이지만, 호스트 에뮬레이션은 libm의
 * 완전 정밀도 powf를 사용하므로 미세한 수치 차이가 발생할 수 있다.
 * 실행 컨텍스트: 호스트 CPU, 단일 스레드.
 * 호출 체인: instructions.cc (PTX 거듭제곱 명령어) → [이 함수] → libm powf
 */
// intrinsic for power
float __powf(float a, float b) { return powf(a, b); } /* [한국어] C 표준 라이브러리의 powf(float base, float exp)를 직접 호출한다. a<0이고 b가 정수가 아닌 경우 NaN을 반환한다. */

/* [한국어] macOS(Apple) 전용 조건부 컴파일 분기.
 * Mac OS X의 GCC 버전에서는 __signbitd 함수가 누락되어 있어,
 * GPGPU-Sim 코드나 포함되는 CUDA 헤더에서 이 함수를 호출하면 링크 오류가 발생한다.
 * 이를 방지하기 위해 macOS 빌드 시에만 소프트웨어 에뮬레이션 구현을 제공한다.
 * Linux나 Windows에서는 이 블록 전체가 전처리 단계에서 제거된다. */
// math functions missing in Mac OSX GCC
#ifdef __APPLE__
/*
 * [한국어]
 * __signbitd - double 값의 부호 비트 추출 (macOS 전용)
 *
 * @d   : 부호 비트를 확인할 double 값. 음수, +0.0, -0.0, NaN, Inf 모두 처리 가능.
 * @return: 부호 비트가 1이면 (음수 또는 -0.0이면) 비영값(참), 아니면 0(거짓).
 *
 * IEEE 754 double 형식의 최상위 비트(MSB, 비트 63)가 부호 비트이다.
 * Mac GCC에서 누락된 이 함수를 비트 연산으로 직접 구현한다.
 * 포인터 캐스트를 통해 double의 비트 패턴을 unsigned long long으로 재해석하여
 * 부호 비트를 마스킹하는 방식은 C99의 "type punning via pointer"이며,
 * 엄밀히는 undefined behavior이지만 GPGPU-Sim 빌드 환경(GCC/Clang)에서는 안전하게 동작한다.
 * 실행 컨텍스트: 호스트 CPU, macOS 빌드 환경에서만 활성화.
 * 호출 체인: CUDA 수학 함수 내부 → [이 함수]
 */
int __signbitd(double d) {
  unsigned long long int u = *((unsigned long long int*)&d); /* [한국어] double d의 메모리 주소를 unsigned long long 포인터로 재해석한 후 역참조한다. 이를 통해 double의 64비트 IEEE 754 비트 패턴을 정수로 얻는다. 포인터 캐스트를 사용한 type punning이다. */
  return ((u & 0x8000000000000000ULL) != 0); /* [한국어] 0x8000000000000000ULL은 64비트 값의 MSB(비트 63)만 1인 마스크이다. 부호 비트(MSB)가 1이면(음수 또는 -0.0이면) 참을 반환한다. double NaN의 경우도 부호 비트 패턴에 따라 결과가 달라진다. */
}
#endif /* [한국어] __APPLE__ 조건부 컴파일 블록 종료. */

/* [한국어] __CUDACC__ 매크로를 해제한다.
 * math_functions.h 포함을 위해 앞서 임시로 정의했으나, 포함 직전에 undef하고
 * 별도의 __CUDA_INTERNAL_COMPILATION__ 매크로로 전환한다.
 * 이 시점에서 __CUDACC__를 해제하여 이후 호스트 코드가 CUDA 컴파일러처럼
 * 동작하는 것을 방지한다. */
#undef __CUDACC__

/* [한국어] CUDA 내부 컴파일 매크로를 정의하여 math_functions.h의 구현부를 활성화한다.
 * math_functions.h는 __CUDA_INTERNAL_COMPILATION__이 정의된 상태에서만 실제
 * 함수 정의(구현)를 출력한다. 이 매크로 없이는 선언부만 포함되어 링크 오류가 발생한다. */
#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.h>   /* [한국어] NVIDIA CUDA 수학 함수 정의(sinf, cosf, expf, logf, rsqrtf 등 GPU 내장 함수의 호스트 에뮬레이션)를 포함한다. 이 파일 앞부분에서 정의한 float2int, fdividef 등이 math_functions.h 내부에서 참조된다. */
#undef __CUDA_INTERNAL_COMPILATION__ /* [한국어] CUDA 내부 컴파일 매크로를 즉시 해제한다. 이후 코드에서는 CUDA 내부 컴파일 모드가 비활성화된다. */
#undef __attribute__          /* [한국어] 파일 상단에서 빈 매크로로 재정의했던 __attribute__를 해제한다. 이제 컴파일러 본래의 __attribute__ 의미가 복원된다. */

/* [한국어] CUDART_VERSION 조건부 분기 종료.
 * 여기까지가 CUDART_VERSION >= 3000 경로의 구현이다. */
#endif

}  // namespace cuda_math
/* [한국어] cuda_math 네임스페이스 종료.
 * 이 이후의 코드는 전역 네임스페이스에 속한다.
 * macOS 전용 isnanf 함수는 cuda_math 바깥에 정의됨으로써
 * 전역 함수로 사용 가능하다. */

/* [한국어] macOS(Apple) 전용 isnanf 함수 정의 (전역 네임스페이스).
 * Mac OS X의 GCC에서는 float 버전의 isnanf()가 누락되어 있어
 * CUDA 헤더나 시뮬레이터 코드에서 이 함수를 호출하면 링크 오류가 발생한다.
 * cuda_math 네임스페이스 밖에 전역 함수로 정의하여 다른 코드에서도 사용 가능하다.
 * Linux나 Windows 빌드에서는 이 블록이 전처리 단계에서 제거된다. */
// math functions missing in Mac OSX GCC
#ifdef __APPLE__
/*
 * [한국어]
 * isnanf - float NaN 판별 함수 (macOS 전용, 전역 네임스페이스)
 *
 * @a   : NaN 여부를 판별할 float 값.
 * @return: a가 NaN이면 비영값(참), 아니면 0(거짓).
 *
 * Mac GCC에서 누락된 isnanf를 std::isnan을 활용하여 구현한다.
 * std::isnan은 <cmath>에서 제공되며, float 인자에 대해 올바르게 동작한다.
 * GPU PTX에서 TESTP.NOTANUMBER 명령어나 SETP 명령어의 NaN 처리에 대응하는
 * 호스트 에뮬레이션에서 이 함수가 사용된다.
 * 실행 컨텍스트: 호스트 CPU, macOS 빌드 환경에서만 활성화.
 * 호출 체인: CUDA 수학 함수 또는 instructions.cc → [이 함수] → std::isnan
 */
int isnanf(float a) { return (std::isnan(a)); } /* [한국어] std::isnan(a)는 IEEE 754 NaN 비트 패턴(지수부 전부 1, 가수부 비영)을 검사한다. 결과를 int로 반환하여 C 스타일 불리언으로 사용 가능하다. */
#endif /* [한국어] __APPLE__ 조건부 컴파일 블록 종료. */

/* [한국어] CUDA_MATH 인클루드 가드 종료.
 * 파일 상단의 #ifndef CUDA_MATH와 쌍을 이루며, 이 헤더의 중복 포함을 방지한다. */
#endif
