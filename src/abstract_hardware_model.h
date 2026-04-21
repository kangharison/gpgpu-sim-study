/*
 * ============================================================================
 * abstract_hardware_model.h
 * ============================================================================
 * 이 파일은 GPGPU-Sim이라는 GPU 시뮬레이터의 핵심 헤더 파일입니다.
 * "헤더 파일"이란 다른 코드 파일들이 공유해서 사용하는 설계도 같은 것입니다.
 *
 * GPU(Graphics Processing Unit, 그래픽 처리 장치)는 원래 게임 화면을 그리기
 * 위해 만들어졌지만, 지금은 인공지능(AI), 과학 계산 등에도 많이 사용됩니다.
 * GPU는 수천 개의 작은 계산기(코어)를 가지고 있어서 동시에 많은 계산을 할 수 있습니다.
 *
 * 이 파일에는 GPU의 하드웨어(물리적 부품)를 소프트웨어로 흉내내기 위한
 * 클래스, 열거형(enum), 구조체(struct) 등이 정의되어 있습니다.
 * ============================================================================
 */

// Copyright (c) 2009-2021, Tor M. Aamodt, Inderpreet Singh, Vijay Kandiah,
// 저작권 표시: 이 코드를 만든 사람들의 이름입니다 (2009~2021년)
// Nikos Hardavellas, Mahmoud Khairy, Junrui Pan, Timothy G. Rogers The
// University of British Columbia, Northwestern University, Purdue University
// 이 코드를 만든 대학교들: 브리티시컬럼비아 대학, 노스웨스턴 대학, 퍼듀 대학
// All rights reserved.
// 모든 권리 보유 - 이 코드의 권리는 위 사람들/기관에게 있다는 뜻
//
// Redistribution and use in source and binary forms, with or without
// 재배포 및 사용은 소스 코드와 바이너리(실행 파일) 형태로, 수정 여부에 관계없이
// modification, are permitted provided that the following conditions are met:
// 다음 조건을 충족하면 허용됩니다:
//
// 1. Redistributions of source code must retain the above copyright notice,
// 1. 소스 코드를 재배포할 때는 위의 저작권 표시를 유지해야 합니다
// this
//    list of conditions and the following disclaimer;
//    이 조건 목록과 아래의 면책 조항도 함께요
// 2. Redistributions in binary form must reproduce the above copyright notice,
// 2. 바이너리(실행 파일) 형태로 재배포할 때는 저작권 표시를 문서에 포함해야 합니다
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
// 3. 대학교 이름을 허가 없이 홍보에 사용할 수 없습니다
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// 이 소프트웨어는 "있는 그대로" 제공됩니다 (보증 없음)
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
// 위 내용은 "BSD 라이선스"라는 오픈소스 라이선스입니다.
// 쉽게 말해, "이 코드를 자유롭게 쓸 수 있지만, 문제가 생겨도 우리 책임이 아닙니다"라는 뜻

/*
 * #ifndef / #define / #endif는 "인클루드 가드(include guard)"라고 합니다.
 * 같은 헤더 파일이 여러 번 포함(include)되는 것을 방지합니다.
 * 마치 "이미 읽은 책은 다시 안 읽는다"와 같은 원리입니다.
 */
#ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED  // 이 파일이 아직 포함되지 않았다면
#define ABSTRACT_HARDWARE_MODEL_INCLUDED  // 이 파일이 포함되었다고 표시

/*
 * 전방 선언(Forward declarations):
 * "이런 클래스가 있을 거야"라고 컴파일러(코드를 실행 파일로 바꾸는 프로그램)에게
 * 미리 알려주는 것입니다. 자세한 내용은 나중에 다른 파일에서 정의합니다.
 * 마치 "나중에 소개할 친구가 있어"라고 미리 말해두는 것과 비슷합니다.
 */
// Forward declarations
class gpgpu_sim;       // GPU 시뮬레이터 전체를 나타내는 클래스 (나중에 정의됨)
class kernel_info_t;   // GPU에서 실행할 프로그램(커널) 정보를 담는 클래스
class gpgpu_context;   // GPU 시뮬레이터의 전체 상태(맥락)를 담는 클래스

/*
 * #define은 "매크로"라고 하며, 코드에서 특정 이름을 값으로 바꿔치기합니다.
 * 예: MAX_CTA_PER_SHADER를 쓰면 컴파일러가 자동으로 32로 바꿉니다.
 *
 * CTA(Cooperative Thread Array)란?
 *   - GPU에서 함께 협력하는 스레드(실행 단위)들의 묶음입니다.
 *   - CUDA에서는 "블록(block)"이라고도 부릅니다.
 *   - 하나의 CTA 안의 스레드들은 공유 메모리를 함께 사용할 수 있습니다.
 *
 * Shader(셰이더)란?
 *   - GPU의 계산 유닛(처리기)을 말합니다.
 *   - NVIDIA에서는 SM(Streaming Multiprocessor)이라고 부릅니다.
 */
// Set a hard limit of 32 CTAs per shader [cuda only has 8]
#define MAX_CTA_PER_SHADER 32   // 하나의 셰이더(SM)에서 최대 32개 CTA를 실행 가능 (CUDA 기본은 8개)
#define MAX_BARRIERS_PER_CTA 16 // 하나의 CTA에서 사용할 수 있는 최대 배리어 수 (배리어: 스레드들이 서로 기다리는 동기화 지점)

/*
 * 명령어(instruction)가 사용할 수 있는 입력/출력 값의 최대 개수
 * 벡터 연산(여러 값을 한 번에 처리)을 확장한 후의 최대 개수입니다.
 */
// After expanding the vector input and output operands
#define MAX_INPUT_VALUES 24   // 하나의 명령어가 읽을 수 있는 최대 입력 값 개수
#define MAX_OUTPUT_VALUES 8   // 하나의 명령어가 쓸 수 있는 최대 출력 값 개수

/*
 * enum(열거형)이란?
 *   관련 있는 상수들을 하나의 그룹으로 모아놓은 것입니다.
 *   예를 들어, "요일"을 enum으로 만들면: 월=0, 화=1, 수=2, ...
 *
 * _memory_space_t: GPU의 다양한 메모리 공간(종류)을 나타냅니다.
 * GPU는 여러 종류의 메모리를 가지고 있고, 각각 속도와 용도가 다릅니다.
 * 마치 집에 서랍장, 책상, 창고가 있는 것처럼, GPU도 다양한 저장 공간이 있습니다.
 */
enum _memory_space_t {
  undefined_space = 0,        // 정의되지 않은 메모리 공간 (아직 어떤 종류인지 모름)
  reg_space,                  // 레지스터 공간: CPU/GPU에서 가장 빠른 저장소. 계산할 때 즉시 사용하는 값을 보관
  local_space,                // 로컬 메모리: 각 스레드가 개인적으로 사용하는 메모리 (느림, 스레드마다 별도)
  shared_space,               // 공유 메모리: 같은 CTA(블록) 안의 스레드들이 함께 사용하는 빠른 메모리
  sstarr_space,               // 공유 메모리 배열 공간 (특수 용도)
  param_space_unclassified,   // 분류되지 않은 파라미터(매개변수) 공간
  param_space_kernel,         // 커널 파라미터 공간: 모든 스레드가 읽을 수 있지만 쓸 수 없음 (읽기 전용) /* global to all threads in a kernel : read-only */
  param_space_local,          // 로컬 파라미터 공간: 각 스레드가 읽고 쓸 수 있음 /* local to a thread : read-writable */
  const_space,                // 상수 메모리: 변하지 않는 값을 저장, 모든 스레드가 읽을 수 있음 (빠른 캐시 지원)
  tex_space,                  // 텍스처 메모리: 이미지(텍스처) 데이터를 위한 특별한 메모리 (캐시 최적화됨)
  surf_space,                 // 서피스 메모리: 텍스처와 비슷하지만 읽기/쓰기 모두 가능
  global_space,               // 글로벌 메모리: GPU의 가장 크지만 가장 느린 메모리. 모든 스레드가 접근 가능
  generic_space,              // 일반(제네릭) 주소 공간: 실제 메모리 종류는 실행 시 결정
  instruction_space            // 명령어 공간: GPU 프로그램의 명령어(코드)가 저장되는 곳
};

/*
 * #ifndef COEFF_STRUCT / #define COEFF_STRUCT: 이것도 인클루드 가드입니다.
 * 이 구조체가 중복 정의되는 것을 방지합니다.
 *
 * struct(구조체)란?
 *   여러 개의 관련된 변수들을 하나로 묶어놓은 것입니다.
 *   마치 학생 정보(이름, 나이, 학번)를 하나의 카드에 적는 것과 비슷합니다.
 */
#ifndef COEFF_STRUCT   // COEFF_STRUCT가 아직 정의되지 않았다면
#define COEFF_STRUCT   // 정의되었다고 표시

/*
 * PowerscalingCoefficients: 전력(파워) 스케일링 계수를 담는 구조체
 * GPU 시뮬레이션에서 각 연산이 얼마나 전력을 소비하는지 계산할 때 사용합니다.
 * double은 소수점이 있는 숫자(실수)를 저장하는 타입입니다.
 * 계수(coefficient)란 곱해지는 숫자를 의미합니다. 예: 전력 = 계수 x 연산횟수
 */
struct PowerscalingCoefficients {
  double int_coeff;        // 정수(integer) 연산의 전력 계수
  double int_mul_coeff;    // 정수 곱셈(multiply) 연산의 전력 계수
  double int_mul24_coeff;  // 24비트 정수 곱셈의 전력 계수 (24비트: 숫자 크기가 작은 곱셈)
  double int_mul32_coeff;  // 32비트 정수 곱셈의 전력 계수 (32비트: 일반적인 크기의 곱셈)
  double int_div_coeff;    // 정수 나눗셈(divide) 연산의 전력 계수
  double fp_coeff;         // 단정밀도 부동소수점(float, 32비트 실수) 연산의 전력 계수
  double dp_coeff;         // 배정밀도 부동소수점(double, 64비트 실수) 연산의 전력 계수 (더 정확하지만 더 느림)
  double fp_mul_coeff;     // 단정밀도 부동소수점 곱셈의 전력 계수
  double fp_div_coeff;     // 단정밀도 부동소수점 나눗셈의 전력 계수
  double dp_mul_coeff;     // 배정밀도 부동소수점 곱셈의 전력 계수
  double dp_div_coeff;     // 배정밀도 부동소수점 나눗셈의 전력 계수
  double sqrt_coeff;       // 제곱근(square root, 루트) 연산의 전력 계수
  double log_coeff;        // 로그(logarithm) 연산의 전력 계수
  double sin_coeff;        // 사인(sin, 삼각함수) 연산의 전력 계수
  double exp_coeff;        // 지수(exponential, e^x) 연산의 전력 계수
  double tensor_coeff;     // 텐서 코어 연산의 전력 계수 (텐서 코어: AI 연산을 빠르게 하는 특수 장치)
  double tex_coeff;        // 텍스처 연산의 전력 계수
};
#endif  // COEFF_STRUCT 인클루드 가드 끝

/*
 * FuncCache: 함수별 캐시 설정 방식
 * 캐시(cache)란?
 *   자주 사용하는 데이터를 빠르게 접근할 수 있도록 임시로 저장하는 작은 메모리입니다.
 *   마치 책상 위에 자주 보는 책을 올려두는 것과 같습니다.
 *
 * GPU에는 L1 캐시와 공유 메모리가 같은 물리적 공간을 나눠 쓰는데,
 * 어떤 것에 더 많은 공간을 줄지 선택할 수 있습니다.
 */
enum FuncCache {
  FuncCachePreferNone = 0,     // 특별한 선호 없음: 기본 설정 사용
  FuncCachePreferShared = 1,   // 공유 메모리를 더 많이 사용하도록 설정
  FuncCachePreferL1 = 2        // L1 캐시를 더 많이 사용하도록 설정
};

/*
 * AdaptiveCache: 캐시를 고정 크기로 쓸지, 상황에 따라 자동 조절할지 선택
 */
enum AdaptiveCache {
  FIXED = 0,            // 고정: 캐시 크기를 바꾸지 않음
  ADAPTIVE_CACHE = 1    // 적응형: 프로그램 상황에 따라 캐시 크기를 자동 조절
};

/*
 * #ifdef __cplusplus: C++ 컴파일러로 컴파일할 때만 아래 코드를 포함합니다.
 * C와 C++은 비슷하지만 다른 프로그래밍 언어이고,
 * 이 부분은 C++에서만 사용 가능한 기능(클래스 등)을 포함합니다.
 */
#ifdef __cplusplus

#include <stdio.h>    // 화면에 출력하거나 파일을 읽고 쓰는 기능 (printf, fprintf 등)
#include <string.h>   // 문자열(텍스트) 처리 기능 (memset, strcmp 등)
#include <set>        // std::set 컨테이너: 중복 없이 값을 정렬해서 저장하는 자료구조

/*
 * typedef는 "타입에 별명을 붙이는 것"입니다.
 * unsigned long long은 아주 큰 양수를 저장할 수 있는 숫자 타입입니다 (최소 64비트).
 * 메모리 주소를 나타낼 때 사용합니다.
 */
typedef unsigned long long new_addr_type;        // 새로운 주소 타입: 메모리 주소를 나타냄
typedef unsigned long long cudaTextureObject_t;  // CUDA 텍스처 객체 식별자
typedef unsigned long long address_type;         // 주소 타입: 프로그램 카운터(PC) 등에 사용
typedef unsigned long long addr_t;               // 주소 타입 (짧은 이름)

/*
 * 타이밍 모델(시간 시뮬레이션)에서 볼 수 있는 연산(operation) 종류들
 * SPECIALIZED_UNIT_NUM: 특수 실행 유닛의 개수 (8개)
 * SPEC_UNIT_START_ID: 특수 유닛 ID가 시작하는 번호 (100번부터)
 */
// the following are operations the timing model can see
#define SPECIALIZED_UNIT_NUM 8     // 특수 연산 유닛 최대 8개
#define SPEC_UNIT_START_ID 100     // 특수 유닛의 시작 ID 번호

/*
 * uarch_op_t: 마이크로아키텍처(uarch) 연산 타입
 * "마이크로아키텍처"란 CPU/GPU 내부의 실제 하드웨어 구조를 의미합니다.
 *
 * GPU가 실행할 수 있는 모든 종류의 명령어(연산)를 나열한 것입니다.
 * 마치 계산기의 버튼 종류(+, -, x, ÷ 등)를 나열한 것과 비슷합니다.
 */
enum uarch_op_t {
  NO_OP = -1,              // 연산 없음: 아무것도 하지 않는 명령어
  ALU_OP = 1,              // ALU 연산: 산술논리장치(더하기, 빼기, 비교 등 기본 계산)
  SFU_OP,                  // SFU 연산: 특수함수장치(sin, cos, 제곱근 등 복잡한 수학 계산)
  TENSOR_CORE_OP,          // 텐서 코어 연산: 행렬 곱셈 등 AI/딥러닝에 사용되는 특수 연산
  DP_OP,                   // 배정밀도(Double Precision) 연산: 64비트 실수 계산
  SP_OP,                   // 단정밀도(Single Precision) 연산: 32비트 실수 계산
  INTP_OP,                 // 정수(Integer) 연산: 정수 계산
  ALU_SFU_OP,              // ALU와 SFU를 함께 사용하는 연산
  LOAD_OP,                 // 로드 연산: 메모리에서 데이터를 읽어오는 것 (메모리 → 레지스터)
  TENSOR_CORE_LOAD_OP,     // 텐서 코어 로드: 텐서 코어 연산을 위해 데이터를 읽어옴
  TENSOR_CORE_STORE_OP,    // 텐서 코어 스토어: 텐서 코어 연산 결과를 메모리에 저장
  STORE_OP,                // 스토어 연산: 데이터를 메모리에 저장하는 것 (레지스터 → 메모리)
  BRANCH_OP,               // 분기 연산: 조건에 따라 다른 코드로 점프 (if/else와 비슷)
  BARRIER_OP,              // 배리어 연산: 스레드들이 서로 기다리는 동기화 명령
  MEMORY_BARRIER_OP,       // 메모리 배리어: 메모리 접근 순서를 보장하는 명령
  CALL_OPS,                // 함수 호출 연산: 다른 함수를 부르는 명령
  RET_OPS,                 // 리턴 연산: 함수에서 돌아오는 명령
  EXIT_OPS,                // 종료 연산: 스레드가 실행을 끝내는 명령
  SPECIALIZED_UNIT_1_OP = SPEC_UNIT_START_ID,  // 특수 유닛 1 연산 (ID: 100)
  SPECIALIZED_UNIT_2_OP,   // 특수 유닛 2 연산 (ID: 101)
  SPECIALIZED_UNIT_3_OP,   // 특수 유닛 3 연산 (ID: 102)
  SPECIALIZED_UNIT_4_OP,   // 특수 유닛 4 연산 (ID: 103)
  SPECIALIZED_UNIT_5_OP,   // 특수 유닛 5 연산 (ID: 104)
  SPECIALIZED_UNIT_6_OP,   // 특수 유닛 6 연산 (ID: 105)
  SPECIALIZED_UNIT_7_OP,   // 특수 유닛 7 연산 (ID: 106)
  SPECIALIZED_UNIT_8_OP    // 특수 유닛 8 연산 (ID: 107)
};
typedef enum uarch_op_t op_type;  // uarch_op_t에 "op_type"이라는 짧은 별명을 붙임

/*
 * uarch_bar_t: 배리어(barrier) 종류
 * 배리어란 "여기서 다 같이 기다려!"라는 동기화 명령입니다.
 * 학교에서 모든 학생이 도착할 때까지 출발하지 않는 것과 비슷합니다.
 */
enum uarch_bar_t {
  NOT_BAR = -1,  // 배리어가 아님
  SYNC = 1,      // 동기화(SYNC): 모든 스레드가 이 지점에 도달할 때까지 기다림
  ARRIVE,        // 도착(ARRIVE): "나는 도착했어"라고 알리지만 기다리지는 않음
  RED            // 리덕션(Reduction): 여러 스레드의 값을 하나로 합치면서 동기화
};
typedef enum uarch_bar_t barrier_type;  // "barrier_type"이라는 별명

/*
 * uarch_red_t: 리덕션(reduction) 연산 종류
 * 리덕션이란 여러 값을 하나의 값으로 줄이는 연산입니다.
 * 예: 100명의 점수를 모두 더해서 합계를 구하는 것
 */
enum uarch_red_t {
  NOT_RED = -1,    // 리덕션이 아님
  POPC_RED = 1,    // 팝카운트(POPC) 리덕션: 1인 비트의 개수를 셈
  AND_RED,         // AND 리덕션: 모든 값을 AND(논리곱) 연산으로 합침
  OR_RED           // OR 리덕션: 모든 값을 OR(논리합) 연산으로 합침
};
typedef enum uarch_red_t reduction_type;  // "reduction_type"이라는 별명

/*
 * uarch_operand_type_t: 피연산자(operand)의 데이터 타입
 * 피연산자란 연산에 사용되는 값입니다. 예: "3 + 5"에서 3과 5가 피연산자
 */
enum uarch_operand_type_t {
  UN_OP = -1,   // 알 수 없는 타입
  INT_OP,       // 정수(Integer) 타입 연산
  FP_OP         // 부동소수점(Floating Point, 실수) 타입 연산
};
typedef enum uarch_operand_type_t types_of_operands;  // "types_of_operands"라는 별명

/*
 * special_operations_t: 특수 연산의 세부 종류
 * GPU 전력 모델(power model)에서 어떤 종류의 연산인지 구분하기 위해 사용합니다.
 * 연산 종류에 따라 소비하는 전력이 다르기 때문입니다.
 */
enum special_operations_t {
  OTHER_OP,        // 기타 연산
  INT__OP,         // 정수 일반 연산
  INT_MUL24_OP,    // 24비트 정수 곱셈
  INT_MUL32_OP,    // 32비트 정수 곱셈
  INT_MUL_OP,      // 정수 곱셈 (일반)
  INT_DIV_OP,      // 정수 나눗셈
  FP_MUL_OP,       // 단정밀도 부동소수점 곱셈
  FP_DIV_OP,       // 단정밀도 부동소수점 나눗셈
  FP__OP,          // 단정밀도 부동소수점 일반 연산
  FP_SQRT_OP,      // 제곱근(루트) 연산
  FP_LG_OP,        // 로그 연산
  FP_SIN_OP,       // 사인 연산
  FP_EXP_OP,       // 지수 연산
  DP_MUL_OP,       // 배정밀도 부동소수점 곱셈
  DP_DIV_OP,       // 배정밀도 부동소수점 나눗셈
  DP___OP,         // 배정밀도 부동소수점 일반 연산
  TENSOR__OP,      // 텐서 코어 연산
  TEX__OP          // 텍스처 연산
};

typedef enum special_operations_t
    special_ops;  // "special_ops"라는 별명 - 전력 모델에서 연산 종류를 식별하는 데 필요

/*
 * operation_pipeline_t: 연산이 사용하는 파이프라인(처리 경로)
 * GPU 내부에는 여러 종류의 파이프라인이 있고, 명령어 종류에 따라
 * 어떤 파이프라인을 사용할지 결정됩니다.
 * 파이프라인이란 세탁기처럼 여러 단계를 거쳐 작업을 처리하는 방식입니다.
 */
enum operation_pipeline_t {
  UNKOWN_OP,         // 알 수 없는 파이프라인
  SP__OP,            // SP(Single Precision) 파이프라인: 단정밀도 실수 계산용
  DP__OP,            // DP(Double Precision) 파이프라인: 배정밀도 실수 계산용
  INTP__OP,          // INT 파이프라인: 정수 계산용
  SFU__OP,           // SFU 파이프라인: 특수 함수 계산용 (sin, cos, 루트 등)
  TENSOR_CORE__OP,   // 텐서 코어 파이프라인: AI 행렬 연산용
  MEM__OP,           // 메모리 파이프라인: 메모리 읽기/쓰기 처리용
  SPECIALIZED__OP,   // 특수 파이프라인: 사용자 정의 특수 연산용
};
typedef enum operation_pipeline_t operation_pipeline;  // "operation_pipeline"이라는 별명

/*
 * mem_operation_t: 메모리 연산이 텍스처 관련인지 아닌지 구분
 */
enum mem_operation_t {
  NOT_TEX,   // 텍스처가 아닌 일반 메모리 연산
  TEX        // 텍스처 메모리 연산
};
typedef enum mem_operation_t mem_operation;  // "mem_operation"이라는 별명

/*
 * _memory_op_t: 메모리 연산의 종류 (읽기/쓰기/없음)
 */
enum _memory_op_t {
  no_memory_op = 0,   // 메모리 연산 없음
  memory_load,        // 메모리 로드(읽기): 메모리에서 데이터를 가져옴
  memory_store        // 메모리 스토어(쓰기): 메모리에 데이터를 저장함
};

/*
 * 표준 C/C++ 라이브러리 헤더 파일들을 포함합니다.
 * 라이브러리란 미리 만들어진 유용한 기능 모음입니다.
 */
#include <assert.h>     // assert: 조건이 거짓이면 프로그램을 멈추는 디버깅(오류 찾기) 도구
#include <stdlib.h>     // 메모리 할당(malloc, free), 난수 생성 등의 기본 기능
#include <algorithm>    // 정렬, 검색 등 알고리즘 모음 (std::sort, std::find 등)
#include <bitset>       // std::bitset: 비트(0 또는 1) 배열을 다루는 자료구조. 워프 마스크에 사용
#include <deque>        // std::deque: 양쪽 끝에서 추가/삭제할 수 있는 자료구조 (덱)
#include <list>         // std::list: 연결 리스트. 중간에 요소를 추가/삭제하기 쉬운 자료구조
#include <map>          // std::map: 키-값 쌍으로 데이터를 저장하는 사전(딕셔너리) 자료구조
#include <vector>       // std::vector: 크기가 자동으로 변하는 배열 (가장 많이 쓰는 자료구조)

/*
 * vector_types.h: CUDA의 dim3 같은 벡터 타입을 정의하는 헤더
 * dim3은 3차원 좌표(x, y, z)를 나타내는 구조체입니다.
 * GPU 프로그래밍에서 그리드(grid)와 블록(block)의 크기를 지정할 때 사용합니다.
 */
#if !defined(__VECTOR_TYPES_H__)  // vector_types.h가 아직 포함되지 않았다면
#include "vector_types.h"         // dim3 등의 벡터 타입 정의를 포함
#endif

/*
 * dim3comp: dim3 구조체를 비교하는 함수 객체(functor)
 * std::map에서 dim3을 키(key)로 사용하려면 크기 비교가 필요합니다.
 * 이 구조체는 두 dim3 값의 순서를 z → y → x 순으로 비교합니다.
 *
 * operator()를 정의하면 이 구조체를 함수처럼 사용할 수 있습니다.
 * 예: dim3comp comp; comp(a, b); 처럼 호출 가능
 */
struct dim3comp {
  bool operator()(const dim3 &a, const dim3 &b) const {  // 두 dim3 값을 비교하는 함수
    if (a.z < b.z)       // z값을 먼저 비교
      return true;        // a의 z가 작으면 a가 "작다"고 판단
    else if (a.y < b.y)  // z가 같으면 y값 비교
      return true;        // a의 y가 작으면 a가 "작다"고 판단
    else if (a.x < b.x)  // y도 같으면 x값 비교
      return true;        // a의 x가 작으면 a가 "작다"고 판단
    else
      return false;       // 모두 같거나 a가 더 크면 false
  }
};

/*
 * increment_x_then_y_then_z: dim3 값을 x → y → z 순서로 증가시키는 함수
 * 마치 시계에서 초 → 분 → 시 순서로 올라가는 것과 같습니다.
 * x가 최대에 도달하면 y를 1 증가시키고 x를 0으로, y가 최대면 z를 증가...
 */
void increment_x_then_y_then_z(dim3 &i, const dim3 &bound);

// Jin: child kernel information for CDP
// Jin이라는 개발자가 추가한 부분: CDP(CUDA Dynamic Parallelism) 지원
// CDP란 GPU 프로그램 안에서 또 다른 GPU 프로그램을 실행하는 기능입니다.
// 마치 수업 중에 또 다른 수업을 시작하는 것과 비슷합니다.
#include "stream_manager.h"     // 스트림 관리자 헤더 포함 (스트림: GPU 작업의 대기열)
class stream_manager;           // 스트림 관리자 클래스 전방 선언
struct CUstream_st;             // CUDA 스트림 구조체 전방 선언
// extern stream_manager * g_stream_manager;  // 주석 처리됨: 전역 스트림 관리자 (사용 안 함)

/*
 * extern: "이 변수는 다른 파일에서 정의되어 있어"라고 알려주는 키워드
 * pinned_memory: 고정된(pinned) 메모리의 매핑 정보
 * "고정 메모리"란 운영체제가 위치를 바꾸지 않는 메모리로, CPU↔GPU 간 데이터 전송이 빠릅니다.
 */
// support for pinned memories added
extern std::map<void *, void **> pinned_memory;       // 고정 메모리 주소 매핑 (가상 주소 → 실제 주소)
extern std::map<void *, size_t> pinned_memory_size;   // 고정 메모리 크기 매핑 (주소 → 크기)

/*
 * ==========================================================================
 * kernel_info_t 클래스
 * ==========================================================================
 * 커널(kernel)이란 GPU에서 실행되는 하나의 프로그램(함수)입니다.
 * 이 클래스는 커널에 대한 모든 정보를 담고 있습니다:
 * - 그리드 크기 (총 블록 수)
 * - 블록 크기 (블록 당 스레드 수)
 * - 다음에 실행할 CTA/스레드 ID
 * - 실행 상태 (실행 중, 완료 등)
 *
 * class는 변수(데이터)와 함수(동작)를 하나로 묶은 것입니다.
 * public: 외부에서 접근 가능한 부분
 * private: 클래스 내부에서만 접근 가능한 부분 (캡슐화)
 */
class kernel_info_t {
 public:  // 외부에서 접근 가능한 영역
  //   kernel_info_t()              // 주석 처리된 기본 생성자 (사용 안 함)
  //   {
  //      m_valid=false;
  //      m_kernel_entry=NULL;
  //      m_uid=0;
  //      m_num_cores_running=0;
  //      m_param_mem=NULL;
  //   }

  /*
   * 생성자(constructor): 객체가 만들어질 때 자동으로 호출되는 특별한 함수
   * 커널 정보를 초기화합니다.
   * gridDim: 그리드 크기 (CTA가 몇 개인지, 3차원으로 표현)
   * blockDim: 블록 크기 (각 CTA에 스레드가 몇 개인지, 3차원으로 표현)
   * entry: 커널 함수의 진입점 (어디서부터 실행을 시작할지)
   * streamID: 이 커널이 속한 스트림 번호
   */
  kernel_info_t(dim3 gridDim, dim3 blockDim, class function_info *entry,
                unsigned long long streamID);
  /*
   * 두 번째 생성자: 텍스처 매핑 정보도 함께 받는 버전
   * 텍스처(texture)는 이미지 데이터를 GPU에서 효율적으로 읽기 위한 방식입니다.
   */
  kernel_info_t(
      dim3 gridDim, dim3 blockDim, class function_info *entry,
      std::map<std::string, const struct cudaArray *> nameToCudaArray,       // 텍스처 이름 → CUDA 배열 매핑
      std::map<std::string, const struct textureInfo *> nameToTextureInfo);  // 텍스처 이름 → 텍스처 정보 매핑
  ~kernel_info_t();  // 소멸자(destructor): 객체가 삭제될 때 메모리를 정리하는 함수

  /*
   * 코어(SM) 실행 관리 함수들
   * GPU의 SM(Streaming Multiprocessor)이 이 커널을 실행하기 시작하거나 끝날 때 호출
   */
  void inc_running() { m_num_cores_running++; }  // 이 커널을 실행하는 코어 수를 1 증가
  void dec_running() {                            // 이 커널을 실행하는 코어 수를 1 감소
    assert(m_num_cores_running > 0);              // 실행 중인 코어가 0보다 커야 함 (안전 검사)
    m_num_cores_running--;                        // 코어 수 감소
  }
  bool running() const { return m_num_cores_running > 0; }  // 아직 실행 중인 코어가 있는지 확인
  bool done() const { return no_more_ctas_to_run() && !running(); }  // 커널이 완전히 끝났는지: 더 실행할 CTA가 없고 실행 중인 코어도 없으면 완료

  /*
   * entry(): 커널 함수의 진입점(시작 정보)을 반환
   * const 버전은 읽기 전용으로만 접근
   */
  class function_info *entry() {
    return m_kernel_entry;                        // 커널 함수 진입점 반환
  }
  const class function_info *entry() const { return m_kernel_entry; }  // 읽기 전용 버전

  /*
   * num_blocks(): 커널의 총 블록(CTA) 수를 계산
   * 3차원이므로 x * y * z로 계산합니다.
   * 예: gridDim이 (4, 2, 1)이면 총 4*2*1 = 8개 블록
   */
  size_t num_blocks() const {
    return m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;
  }

  /*
   * threads_per_cta(): 하나의 CTA(블록) 안에 있는 스레드 수를 계산
   * 예: blockDim이 (256, 1, 1)이면 256개 스레드
   */
  size_t threads_per_cta() const {
    return m_block_dim.x * m_block_dim.y * m_block_dim.z;
  }

  dim3 get_grid_dim() const { return m_grid_dim; }   // 그리드 크기(dim3)를 반환
  dim3 get_cta_dim() const { return m_block_dim; }   // CTA(블록) 크기(dim3)를 반환

  /*
   * increment_cta_id(): 다음에 실행할 CTA의 ID를 증가시킴
   * x를 먼저 증가시키고, x가 최대에 도달하면 y를, y가 최대면 z를 증가
   * 새 CTA를 시작하므로 스레드 ID는 (0,0,0)으로 초기화
   */
  void increment_cta_id() {
    increment_x_then_y_then_z(m_next_cta, m_grid_dim);  // 다음 CTA 좌표 증가
    m_next_tid.x = 0;  // 새 CTA이므로 스레드 x 좌표를 0으로 초기화
    m_next_tid.y = 0;  // 스레드 y 좌표도 0으로
    m_next_tid.z = 0;  // 스레드 z 좌표도 0으로
  }
  dim3 get_next_cta_id() const { return m_next_cta; }  // 다음에 실행할 CTA의 3차원 ID를 반환

  /*
   * get_next_cta_id_single(): 3차원 CTA ID를 1차원 숫자로 변환
   * 마치 2차원 좌표 (행, 열)을 하나의 번호로 바꾸는 것:
   * 번호 = x + (가로크기 * y) + (가로크기 * 세로크기 * z)
   */
  unsigned get_next_cta_id_single() const {
    return m_next_cta.x + m_grid_dim.x * m_next_cta.y +
           m_grid_dim.x * m_grid_dim.y * m_next_cta.z;
  }

  /*
   * no_more_ctas_to_run(): 더 실행할 CTA가 없는지 확인
   * 다음 CTA ID가 그리드 크기를 넘어서면 더 이상 실행할 CTA가 없음
   */
  bool no_more_ctas_to_run() const {
    return (m_next_cta.x >= m_grid_dim.x || m_next_cta.y >= m_grid_dim.y ||
            m_next_cta.z >= m_grid_dim.z);
  }

  /*
   * 스레드 ID 관련 함수들
   * CTA 내에서 다음에 실행할 스레드의 ID를 관리합니다.
   */
  void increment_thread_id() {                                  // 다음 스레드 ID 증가
    increment_x_then_y_then_z(m_next_tid, m_block_dim);
  }
  dim3 get_next_thread_id_3d() const { return m_next_tid; }     // 다음 스레드의 3차원 ID 반환
  unsigned get_next_thread_id() const {                         // 다음 스레드의 1차원 ID 반환
    return m_next_tid.x + m_block_dim.x * m_next_tid.y +
           m_block_dim.x * m_block_dim.y * m_next_tid.z;
  }
  bool more_threads_in_cta() const {                            // 현재 CTA에 더 실행할 스레드가 있는지
    return m_next_tid.z < m_block_dim.z && m_next_tid.y < m_block_dim.y &&
           m_next_tid.x < m_block_dim.x;
  }
  unsigned get_uid() const { return m_uid; }                    // 커널의 고유 ID를 반환
  unsigned long long get_streamID() const { return m_streamID; }  // 커널이 속한 스트림 ID를 반환
  std::string get_name() const { return name(); }               // 커널 이름을 반환
  std::string name() const;                                     // 커널 이름을 반환하는 실제 구현 (다른 파일에 있음)

  /*
   * active_threads(): 현재 활성(실행 중인) 스레드 목록을 반환
   * std::list: 연결 리스트 (요소를 중간에 추가/삭제하기 쉬운 자료구조)
   */
  std::list<class ptx_thread_info *> &active_threads() {
    return m_active_threads;
  }
  /*
   * get_param_memory(): 커널의 파라미터(매개변수) 메모리를 반환
   * 커널에 전달된 인자(argument)들이 저장된 메모리
   */
  class memory_space *get_param_memory() {
    return m_param_mem;
  }

  // The following functions access texture bindings present at the kernel's
  // launch
  // 아래 함수들은 커널이 실행될 때(launch) 설정된 텍스처 바인딩 정보에 접근합니다.
  // 텍스처 바인딩: 텍스처 이름과 실제 메모리 데이터를 연결하는 것

  /*
   * get_texarray(): 텍스처 이름으로 CUDA 배열(실제 데이터)을 찾아 반환
   * assert: 찾지 못하면 프로그램을 멈춤 (오류 상황)
   */
  const struct cudaArray *get_texarray(const std::string &texname) const {
    std::map<std::string, const struct cudaArray *>::const_iterator t =  // 맵에서 텍스처 이름으로 검색
        m_NameToCudaArray.find(texname);
    assert(t != m_NameToCudaArray.end());  // 텍스처를 찾지 못하면 오류 (반드시 있어야 함)
    return t->second;                       // 찾은 CUDA 배열 포인터를 반환
  }

  /*
   * get_texinfo(): 텍스처 이름으로 텍스처 정보(크기, 형식 등)를 찾아 반환
   */
  const struct textureInfo *get_texinfo(const std::string &texname) const {
    std::map<std::string, const struct textureInfo *>::const_iterator t =
        m_NameToTextureInfo.find(texname);
    assert(t != m_NameToTextureInfo.end());  // 찾지 못하면 오류
    return t->second;                         // 텍스처 정보 반환
  }

 private:  // 비공개 영역: 클래스 외부에서 접근 불가
  kernel_info_t(const kernel_info_t &);   // 복사 생성자 비활성화: 커널 정보를 복사하는 것을 금지
  void operator=(const kernel_info_t &);  // 복사 대입 연산자 비활성화: = 연산으로 복사하는 것을 금지

  class function_info *m_kernel_entry;  // 커널 함수의 진입점 (어떤 함수를 실행할지)

  unsigned m_uid;  // Kernel ID: 각 커널을 구분하는 고유 번호
  unsigned long long m_streamID;  // 이 커널이 속한 CUDA 스트림의 ID

  // These maps contain the snapshot of the texture mappings at kernel launch
  // 커널이 실행될 때의 텍스처 매핑 정보를 스냅샷(사진 찍듯이 저장)으로 보관
  std::map<std::string, const struct cudaArray *> m_NameToCudaArray;       // 텍스처 이름 → CUDA 배열
  std::map<std::string, const struct textureInfo *> m_NameToTextureInfo;   // 텍스처 이름 → 텍스처 정보

  dim3 m_grid_dim;    // 그리드 크기: CTA가 3차원으로 몇 개인지 (예: 4x2x1 = 8개 CTA)
  dim3 m_block_dim;   // 블록 크기: 각 CTA 안에 스레드가 3차원으로 몇 개인지 (예: 256x1x1)
  dim3 m_next_cta;    // 다음에 실행할 CTA의 3차원 좌표
  dim3 m_next_tid;    // 다음에 실행할 스레드의 3차원 좌표

  unsigned m_num_cores_running;  // 현재 이 커널을 실행 중인 코어(SM)의 수

  std::list<class ptx_thread_info *> m_active_threads;  // 현재 활성 스레드들의 목록
  class memory_space *m_param_mem;                       // 커널 파라미터 메모리 포인터

 public:  // 다시 공개 영역
  // Jin: parent and child kernel management for CDP
  // CDP(CUDA Dynamic Parallelism) 지원: 부모-자식 커널 관계 관리
  // GPU 프로그램 안에서 또 다른 GPU 프로그램(자식 커널)을 실행할 수 있습니다.

  void set_parent(kernel_info_t *parent, dim3 parent_ctaid, dim3 parent_tid);  // 부모 커널 정보 설정
  void set_child(kernel_info_t *child);      // 자식 커널 추가
  void remove_child(kernel_info_t *child);   // 자식 커널 제거
  bool is_finished();                        // 이 커널이 끝났는지 확인
  bool children_all_finished();              // 모든 자식 커널이 끝났는지 확인
  void notify_parent_finished();             // 부모 커널에게 "나 끝났어"라고 알림
  CUstream_st *create_stream_cta(dim3 ctaid);           // 특정 CTA에서 새 스트림 생성
  CUstream_st *get_default_stream_cta(dim3 ctaid);      // 특정 CTA의 기본 스트림 가져오기
  bool cta_has_stream(dim3 ctaid, CUstream_st *stream); // 특정 CTA에 해당 스트림이 있는지 확인
  void destroy_cta_streams();                // CTA에서 만든 스트림들을 모두 삭제
  void print_parent_info();                  // 부모 커널 정보를 화면에 출력 (디버깅용)
  kernel_info_t *get_parent() { return m_parent_kernel; }  // 부모 커널 포인터 반환

 private:  // 비공개 영역
  kernel_info_t *m_parent_kernel;  // 부모 커널 포인터 (CDP로 이 커널을 실행한 커널)
  dim3 m_parent_ctaid;             // 부모 커널에서 이 커널을 실행한 CTA의 ID
  dim3 m_parent_tid;               // 부모 커널에서 이 커널을 실행한 스레드의 ID
  std::list<kernel_info_t *> m_child_kernels;  // 이 커널이 실행한 자식 커널들의 목록
  std::map<dim3, std::list<CUstream_st *>, dim3comp>
      m_cta_streams;  // 각 CTA에서 생성한 스트림 목록 (dim3comp로 정렬)

  // Jin: kernel timing
  // 커널 실행 시간 측정용 변수들
 public:
  unsigned long long launch_cycle;  // 커널이 "실행 시작 명령"을 받은 사이클(시점)
  unsigned long long start_cycle;   // 커널이 실제로 GPU에서 실행을 시작한 사이클
  unsigned long long end_cycle;     // 커널 실행이 완료된 사이클
  unsigned m_launch_latency;        // 커널 실행 시작까지의 지연 시간 (사이클 단위)

  mutable bool cache_config_set;    // 캐시 설정이 되었는지 여부 (mutable: const 함수에서도 변경 가능)

  unsigned m_kernel_TB_latency;  // CPU에서 GPU로 커널을 보내는 데 걸리는 지연 시간
                                 // this used for any CPU-GPU kernel latency and
                                 // counted in the gpu_cycle
};

/*
 * ==========================================================================
 * core_config 클래스
 * ==========================================================================
 * GPU 코어(SM)의 설정값들을 담는 클래스입니다.
 * 워프 크기, 공유 메모리 크기, 캐시 설정 등 하드웨어 특성을 정의합니다.
 *
 * virtual(가상)이란?
 *   이 클래스를 상속(extends)받는 자식 클래스에서 함수를 다시 정의할 수 있게 해줍니다.
 *   마치 "기본 설계도를 주지만, 세부 사항은 네가 바꿀 수 있어"라는 뜻입니다.
 *
 * "= 0"은 순수 가상 함수: 자식 클래스에서 반드시 구현해야 합니다.
 */
class core_config {
 public:
  /*
   * 생성자: 기본값으로 초기화
   * ctx: GPU 시뮬레이터의 전체 상태를 담는 컨텍스트 포인터
   */
  core_config(gpgpu_context *ctx) {
    gpgpu_ctx = ctx;                           // 컨텍스트 포인터 저장
    m_valid = false;                           // 아직 유효한 설정이 아님
    num_shmem_bank = 16;                       // 공유 메모리 뱅크 수: 16개 (기본값)
    shmem_limited_broadcast = false;           // 공유 메모리 제한 브로드캐스트: 비활성
    gpgpu_shmem_sizeDefault = (unsigned)-1;    // 기본 공유 메모리 크기: 미설정 (-1)
    gpgpu_shmem_sizePrefL1 = (unsigned)-1;     // L1 선호 시 공유 메모리 크기: 미설정
    gpgpu_shmem_sizePrefShared = (unsigned)-1; // 공유 메모리 선호 시 크기: 미설정
  }
  /*
   * 순수 가상 함수(pure virtual function): 자식 클래스가 반드시 구현해야 함
   * init(): 설정을 초기화하는 함수
   */
  virtual void init() = 0;

  bool m_valid;          // 이 설정이 유효한지 여부
  unsigned warp_size;    // 워프(warp) 크기: 동시에 실행되는 스레드 묶음의 크기
  /*
   * 워프(warp)란?
   *   GPU에서 동시에 같은 명령어를 실행하는 32개 스레드의 묶음입니다.
   *   32명의 학생이 선생님의 말에 동시에 같은 동작을 하는 것과 비슷합니다.
   *   NVIDIA GPU의 기본 워프 크기는 32입니다.
   */

  // backward pointer
  class gpgpu_context *gpgpu_ctx;  // GPU 시뮬레이터 컨텍스트를 가리키는 포인터 (역방향 참조)

  // off-chip memory request architecture parameters
  // 오프칩(off-chip) 메모리 요청 아키텍처 파라미터
  // 오프칩 메모리: GPU 칩 밖에 있는 메모리(DRAM 등). 느리지만 크기가 큼
  int gpgpu_coalesce_arch;  // 메모리 합치기(coalescing) 아키텍처 설정
  // 메모리 합치기(coalescing)란?
  //   여러 스레드의 메모리 요청을 하나로 합쳐서 효율적으로 처리하는 기술

  // shared memory bank conflict checking parameters
  // 공유 메모리 뱅크 충돌 검사 파라미터
  // 뱅크 충돌(bank conflict): 여러 스레드가 동시에 같은 메모리 뱅크에 접근하면 순서대로 처리해야 해서 느려짐
  bool shmem_limited_broadcast;          // 제한된 브로드캐스트(동시 전달) 모드 여부
  static const address_type WORD_SIZE = 4;  // 워드(word) 크기: 4바이트 (32비트). 기본 데이터 단위
  unsigned num_shmem_bank;               // 공유 메모리 뱅크 수

  /*
   * shmem_bank_func: 주소에서 어떤 뱅크에 접근하는지 계산
   * (주소 / 4바이트) % 뱅크수 = 뱅크 번호
   */
  unsigned shmem_bank_func(address_type addr) const {
    return ((addr / WORD_SIZE) % num_shmem_bank);
  }
  unsigned mem_warp_parts;               // 메모리 워프 분할 수
  mutable unsigned gpgpu_shmem_size;     // 현재 사용 중인 공유 메모리 크기 (mutable: 변경 가능)
  char *gpgpu_shmem_option;             // 공유 메모리 설정 옵션 문자열
  std::vector<unsigned> shmem_opt_list;  // 사용 가능한 공유 메모리 크기 옵션 목록
  unsigned gpgpu_shmem_sizeDefault;      // 기본 공유 메모리 크기 설정
  unsigned gpgpu_shmem_sizePrefL1;       // L1 캐시 선호 시 공유 메모리 크기
  unsigned gpgpu_shmem_sizePrefShared;   // 공유 메모리 선호 시 공유 메모리 크기
  unsigned mem_unit_ports;               // 메모리 유닛의 포트(접속 통로) 수

  // texture and constant cache line sizes (used to determine number of memory
  // accesses)
  // 텍스처 및 상수 캐시 라인 크기 (메모리 접근 횟수를 계산하는 데 사용)
  // 캐시 라인: 캐시가 한 번에 가져오는 데이터 블록의 크기
  unsigned gpgpu_cache_texl1_linesize;     // 텍스처 L1 캐시 라인 크기 (바이트 단위)
  unsigned gpgpu_cache_constl1_linesize;   // 상수 L1 캐시 라인 크기 (바이트 단위)

  unsigned gpgpu_max_insn_issue_per_warp;  // 워프당 한 사이클에 발행(issue)할 수 있는 최대 명령어 수
  bool gmem_skip_L1D;  // true면 글로벌 메모리 접근 시 L1 데이터 캐시를 건너뜀
  // on = global memory access always skip the L1 cache

  bool adaptive_cache_config;  // 적응형 캐시 설정 사용 여부
};

/*
 * ==========================================================================
 * SIMT 스택 관련 타입 정의
 * ==========================================================================
 * SIMT(Single Instruction, Multiple Threads): 하나의 명령어로 여러 스레드를 동시에 실행
 * GPU의 핵심 실행 방식입니다.
 *
 * 워프 안의 스레드들이 if/else 같은 분기를 만나면
 * 어떤 스레드는 if를, 어떤 스레드는 else를 실행해야 합니다.
 * 이때 SIMT 스택을 사용하여 분기를 관리하고, 나중에 다시 합칩니다.
 * 이것을 "분기 발산(divergence)"과 "재수렴(reconvergence)"이라고 합니다.
 */

// bounded stack that implements simt reconvergence using pdom mechanism from
// MICRO'07 paper
// MICRO'07 논문의 포스트 도미네이터(pdom) 메커니즘을 사용한 SIMT 재수렴 스택
const unsigned MAX_WARP_SIZE = 32;  // 최대 워프 크기: 32 스레드 (NVIDIA GPU 표준)

/*
 * active_mask_t: 워프 내에서 어떤 스레드가 활성(실행 중)인지를 나타내는 비트마스크
 * 32비트 중 각 비트가 하나의 스레드를 나타냄 (1=활성, 0=비활성)
 * 예: 11111111111111111111111111111111 → 32개 스레드 모두 활성
 */
typedef std::bitset<MAX_WARP_SIZE> active_mask_t;

#define MAX_WARP_SIZE_SIMT_STACK MAX_WARP_SIZE  // SIMT 스택용 최대 워프 크기 (32)
typedef std::bitset<MAX_WARP_SIZE_SIMT_STACK> simt_mask_t;  // SIMT 스택에서 사용하는 스레드 마스크
typedef std::vector<address_type> addr_vector_t;  // 주소들의 배열 (각 스레드의 다음 PC 주소 등)

/*
 * ==========================================================================
 * simt_stack 클래스
 * ==========================================================================
 * SIMT 스택: GPU의 분기 발산(divergence)을 관리하는 스택 자료구조
 *
 * 스택(stack)이란?
 *   접시를 쌓는 것처럼 나중에 넣은 것을 먼저 꺼내는 자료구조 (LIFO: Last In, First Out)
 *
 * 워프의 스레드들이 if/else를 만나면:
 * 1. 스택에 "합류 지점(reconvergence point)"을 저장
 * 2. if를 실행하는 스레드들만 활성화하고 나머지는 대기
 * 3. if 실행 후, else를 실행하는 스레드들을 활성화
 * 4. 합류 지점에 도달하면 모든 스레드가 다시 함께 실행
 */
class simt_stack {
 public:
  /*
   * 생성자: 워프 ID, 워프 크기, GPU 포인터를 받아 초기화
   */
  simt_stack(unsigned wid, unsigned warpSize, class gpgpu_sim *gpu);

  void reset();  // 스택을 초기 상태로 리셋

  /*
   * launch: 새 프로그램을 시작할 때 호출. 시작 PC와 활성 스레드 마스크를 설정
   * start_pc: 프로그램 시작 주소
   * active_mask: 어떤 스레드가 활성인지
   */
  void launch(address_type start_pc, const simt_mask_t &active_mask);

  /*
   * update: 분기 명령어를 실행한 후 스택을 업데이트
   * thread_done: 실행이 끝난 스레드들
   * next_pc: 각 스레드의 다음 PC (프로그램 카운터)
   * recvg_pc: 재수렴 지점 (다시 합류하는 곳)의 PC
   */
  void update(simt_mask_t &thread_done, addr_vector_t &next_pc,
              address_type recvg_pc, op_type next_inst_op,
              unsigned next_inst_size, address_type next_inst_pc);

  const simt_mask_t &get_active_mask() const;  // 현재 활성 스레드 마스크를 반환
  void get_pdom_stack_top_info(unsigned *pc, unsigned *rpc) const;  // 스택 맨 위의 PC와 재수렴 PC를 반환
  unsigned get_rp() const;   // 재수렴 지점(reconvergence point)을 반환
  void print(FILE *fp) const;  // 스택 내용을 파일에 출력 (디버깅용)
  void resume(char *fname);   // 체크포인트에서 스택 상태를 복원
  void print_checkpoint(FILE *fout) const;  // 체크포인트용으로 스택 상태를 출력

 protected:  // 보호 영역: 이 클래스와 자식 클래스에서만 접근 가능
  unsigned m_warp_id;    // 이 스택이 관리하는 워프의 ID
  unsigned m_warp_size;  // 워프 크기 (보통 32)

  /*
   * stack_entry_type: 스택 항목의 종류
   * 일반 항목과 함수 호출 항목을 구분합니다.
   */
  enum stack_entry_type {
    STACK_ENTRY_TYPE_NORMAL = 0,  // 일반 분기(if/else)를 위한 항목
    STACK_ENTRY_TYPE_CALL         // 함수 호출(call)을 위한 항목
  };

  /*
   * simt_stack_entry: SIMT 스택의 각 항목(entry)에 저장되는 정보
   * 분기가 발생할 때마다 새 항목이 스택에 추가됩니다.
   */
  struct simt_stack_entry {
    address_type m_pc;                    // 현재 프로그램 카운터(PC): 지금 실행 중인 명령어 주소
    unsigned int m_calldepth;             // 함수 호출 깊이: 함수 안에서 또 함수를 부른 횟수
    simt_mask_t m_active_mask;            // 이 항목에서 활성인 스레드들의 마스크
    address_type m_recvg_pc;              // 재수렴 PC: 분기 후 다시 합류하는 명령어 주소
    unsigned long long m_branch_div_cycle; // 분기 발산이 일어난 사이클 (시점)
    stack_entry_type m_type;              // 이 항목의 타입 (일반/함수호출)
    simt_stack_entry()                    // 기본 생성자: 모든 값을 기본값으로 초기화
        : m_pc(-1),                       // PC를 -1로 초기화 (유효하지 않은 주소)
          m_calldepth(0),                 // 호출 깊이 0
          m_active_mask(),                // 빈 활성 마스크 (모든 스레드 비활성)
          m_recvg_pc(-1),                 // 재수렴 PC도 -1로 (유효하지 않음)
          m_branch_div_cycle(0),          // 분기 발산 사이클 0
          m_type(STACK_ENTRY_TYPE_NORMAL){};  // 일반 타입으로 설정
  };

  std::deque<simt_stack_entry> m_stack;  // SIMT 스택을 deque(덱)로 구현. 양쪽에서 추가/삭제 가능

  class gpgpu_sim *m_gpu;  // GPU 시뮬레이터 객체를 가리키는 포인터
};

/*
 * ==========================================================================
 * GPU 메모리 레이아웃 상수들
 * ==========================================================================
 * GPU의 다양한 메모리 영역의 시작 주소와 크기를 정의합니다.
 * 메모리 레이아웃이란 메모리 공간을 어떻게 나눠 쓸지 정하는 것입니다.
 * 마치 학교 건물의 층별 배치도와 같습니다.
 *
 * constexpr: 컴파일 시간에 계산되는 상수 (C++11 기능)
 */
// Let's just upgrade to C++11 so we can use constexpr here...
// start allocating from this address (lower values used for allocating globals
// in .ptx file)
const unsigned long long GLOBAL_HEAP_START = 0xC0000000;  // 글로벌 힙(동적 메모리) 시작 주소: 약 3GB 위치
// Volta max shmem size is 96kB
const unsigned long long SHARED_MEM_SIZE_MAX = 96 * (1 << 10);  // 공유 메모리 최대 크기: 96KB (Volta GPU 기준)
// Volta max local mem is 16kB
const unsigned long long LOCAL_MEM_SIZE_MAX = 1 << 14;  // 로컬 메모리 최대 크기: 16KB (1 << 14 = 16384바이트)
// Volta Titan V has 80 SMs
const unsigned MAX_STREAMING_MULTIPROCESSORS = 80;  // 최대 SM(스트리밍 멀티프로세서) 수: 80개 (Volta Titan V 기준)
// Max 2048 threads / SM
const unsigned MAX_THREAD_PER_SM = 1 << 11;  // SM당 최대 스레드 수: 2048개 (1 << 11 = 2048)
// MAX 64 warps / SM
const unsigned MAX_WARP_PER_SM = 1 << 6;  // SM당 최대 워프 수: 64개 (2048 스레드 / 32 = 64 워프)

/*
 * 전체 메모리 크기 및 주소 범위 계산
 * 시뮬레이터에서 각 메모리 영역의 시작 주소를 계산합니다.
 */
const unsigned long long TOTAL_LOCAL_MEM_PER_SM =
    MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;  // SM당 총 로컬 메모리: 2048 * 16KB = 32MB
const unsigned long long TOTAL_SHARED_MEM =
    MAX_STREAMING_MULTIPROCESSORS * SHARED_MEM_SIZE_MAX;  // 전체 공유 메모리: 80 * 96KB = 7.5MB
const unsigned long long TOTAL_LOCAL_MEM =
    MAX_STREAMING_MULTIPROCESSORS * MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;  // 전체 로컬 메모리: 80 * 32MB = 2.5GB
const unsigned long long SHARED_GENERIC_START =
    GLOBAL_HEAP_START - TOTAL_SHARED_MEM;  // 공유 메모리 제네릭 주소 시작점: 글로벌 힙 바로 아래
const unsigned long long LOCAL_GENERIC_START =
    SHARED_GENERIC_START - TOTAL_LOCAL_MEM;  // 로컬 메모리 제네릭 주소 시작점: 공유 메모리 아래
const unsigned long long STATIC_ALLOC_LIMIT =
    GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM + TOTAL_SHARED_MEM);  // 정적 할당 한계: 로컬+공유 메모리 아래

/*
 * CUDA 런타임 API가 이미 정의되어 있지 않을 때만 cudaArray를 정의
 * cudaArray: CUDA에서 텍스처 데이터를 저장하는 배열 구조체
 */
#if !defined(__CUDA_RUNTIME_API_H__)

#include "builtin_types.h"  // CUDA 기본 타입 정의 포함

/*
 * cudaArray: CUDA 배열 구조체
 * 텍스처 메모리에 저장되는 이미지/데이터의 정보를 담습니다.
 */
struct cudaArray {
  void *devPtr;              // 디바이스(GPU) 메모리에서의 포인터 (데이터 위치)
  int devPtr32;              // 32비트 디바이스 포인터
  struct cudaChannelFormatDesc desc;  // 채널 형식 설명: 데이터의 비트 수와 타입 (예: 8비트 RGB)
  int width;                 // 배열의 가로 크기 (픽셀 수)
  int height;                // 배열의 세로 크기 (픽셀 수)
  int size;  // in bytes    // 배열의 전체 크기 (바이트 단위)
  unsigned dimensions;       // 차원 수 (1D, 2D, 3D 중 하나)
};

#endif

/*
 * textureReferenceAttr: 텍스처 참조(textureReference)의 추가 속성을 기록하는 구조체
 * __cudaRegisterTexture() 함수를 통해 전달되는 속성들입니다.
 * 텍스처는 GPU에서 이미지를 효율적으로 읽기 위한 특별한 메모리 접근 방식입니다.
 */
// Struct that record other attributes in the textureReference declaration
// - These attributes are passed thru __cudaRegisterTexture()
struct textureReferenceAttr {
  const struct textureReference *m_texref;  // 텍스처 참조 포인터
  int m_dim;                                 // 텍스처 차원 (1D, 2D, 3D)
  enum cudaTextureReadMode m_readmode;       // 텍스처 읽기 모드 (정규화된 float 또는 원래 타입)
  int m_ext;                                 // 확장 플래그
  /*
   * 생성자: 초기화 리스트를 사용하여 모든 멤버를 초기화
   * 초기화 리스트(:)는 멤버 변수를 효율적으로 초기화하는 C++ 방식
   */
  textureReferenceAttr(const struct textureReference *texref, int dim,
                       enum cudaTextureReadMode readmode, int ext)
      : m_texref(texref), m_dim(dim), m_readmode(readmode), m_ext(ext) {}
};

/*
 * ==========================================================================
 * gpgpu_functional_sim_config 클래스
 * ==========================================================================
 * GPU 기능 시뮬레이션(functional simulation)의 설정을 담는 클래스
 * "기능 시뮬레이션"이란 연산의 결과가 올바른지 확인하는 시뮬레이션입니다.
 * (타이밍/성능은 신경 쓰지 않고 결과만 확인)
 *
 * PTX(Parallel Thread Execution)란?
 *   NVIDIA GPU용 중간 언어(어셈블리와 비슷)입니다.
 *   CUDA 코드가 컴파일되면 PTX로 변환되고, 이것이 GPU에서 실행됩니다.
 */
class gpgpu_functional_sim_config {
 public:
  void reg_options(class OptionParser *opp);  // 설정 옵션을 등록하는 함수

  void ptx_set_tex_cache_linesize(unsigned linesize);  // 텍스처 캐시 라인 크기 설정

  /*
   * 아래는 각종 설정값을 반환하는 getter 함수들입니다.
   * const: 이 함수는 객체의 상태를 변경하지 않음을 보장
   */
  unsigned get_forced_max_capability() const {    // 강제 설정된 최대 GPU 능력(compute capability) 반환
    return m_ptx_force_max_capability;
  }
  bool convert_to_ptxplus() const { return m_ptx_convert_to_ptxplus; }  // PTXPlus로 변환할지 여부
  bool use_cuobjdump() const { return m_ptx_use_cuobjdump; }           // cuobjdump 도구 사용 여부
  bool experimental_lib_support() const { return m_experimental_lib_support; }  // 실험적 라이브러리 지원 여부

  int get_ptx_inst_debug_to_file() const { return g_ptx_inst_debug_to_file; }  // PTX 명령어 디버그를 파일에 출력할지
  const char *get_ptx_inst_debug_file() const { return g_ptx_inst_debug_file; }  // 디버그 파일 경로
  int get_ptx_inst_debug_thread_uid() const {       // 디버그할 스레드의 고유 ID
    return g_ptx_inst_debug_thread_uid;
  }
  unsigned get_texcache_linesize() const { return m_texcache_linesize; }  // 텍스처 캐시 라인 크기 반환

  /*
   * 체크포인트(checkpoint) 관련 설정 getter들
   * 체크포인트란 시뮬레이션 중간 상태를 저장해서 나중에 이어서 실행할 수 있게 하는 기능
   * 마치 게임의 "세이브 포인트"와 같습니다.
   */
  int get_checkpoint_option() const { return checkpoint_option; }     // 체크포인트 옵션
  int get_checkpoint_kernel() const { return checkpoint_kernel; }     // 체크포인트를 저장할 커널 번호
  int get_checkpoint_CTA() const { return checkpoint_CTA; }           // 체크포인트를 저장할 CTA 번호
  int get_resume_option() const { return resume_option; }             // 재개(resume) 옵션
  int get_resume_kernel() const { return resume_kernel; }             // 재개할 커널 번호
  int get_resume_CTA() const { return resume_CTA; }                   // 재개할 CTA 번호
  int get_checkpoint_CTA_t() const { return checkpoint_CTA_t; }       // 체크포인트 CTA 시간
  int get_checkpoint_insn_Y() const { return checkpoint_insn_Y; }     // 체크포인트 명령어 Y값

 private:  // 비공개 멤버 변수들
  // PTX options
  int m_ptx_convert_to_ptxplus;       // PTXPlus 변환 여부 (0 또는 1)
  int m_ptx_use_cuobjdump;            // cuobjdump 사용 여부
  int m_experimental_lib_support;     // 실험적 라이브러리 지원 여부
  unsigned m_ptx_force_max_capability; // 강제 최대 컴퓨트 능력
  int checkpoint_option;               // 체크포인트 옵션
  int checkpoint_kernel;               // 체크포인트 커널 번호
  int checkpoint_CTA;                  // 체크포인트 CTA 번호
  unsigned resume_option;              // 재개 옵션
  unsigned resume_kernel;              // 재개 커널 번호
  unsigned resume_CTA;                 // 재개 CTA 번호
  unsigned checkpoint_CTA_t;           // 체크포인트 CTA 시간
  int checkpoint_insn_Y;               // 체크포인트 명령어 Y값
  int g_ptx_inst_debug_to_file;       // PTX 명령어 디버그 파일 출력 플래그
  char *g_ptx_inst_debug_file;        // PTX 명령어 디버그 파일 경로
  int g_ptx_inst_debug_thread_uid;    // 디버그할 스레드 고유 ID

  unsigned m_texcache_linesize;        // 텍스처 캐시 라인 크기 (바이트)
};

/*
 * ==========================================================================
 * gpgpu_t 클래스
 * ==========================================================================
 * GPU 시뮬레이터의 기본 기능을 제공하는 클래스
 * 메모리 할당, 복사, 텍스처 바인딩 등 GPU의 기본 동작을 시뮬레이션합니다.
 *
 * 상속(inheritance)을 위한 기반 클래스로 사용됩니다.
 * gpgpu_sim 클래스가 이 클래스를 상속받아 확장합니다.
 */
class gpgpu_t {
 public:
  /*
   * 생성자: 기능 시뮬레이션 설정과 컨텍스트를 받아 초기화
   */
  gpgpu_t(const gpgpu_functional_sim_config &config, gpgpu_context *ctx);
  // backward pointer
  class gpgpu_context *gpgpu_ctx;  // GPU 시뮬레이터 컨텍스트 (역방향 포인터)

  /*
   * 체크포인트/재개 관련 변수들 (위의 config에서 복사해서 사용)
   */
  int checkpoint_option;
  int checkpoint_kernel;
  int checkpoint_CTA;
  unsigned resume_option;
  unsigned resume_kernel;
  unsigned resume_CTA;
  unsigned checkpoint_CTA_t;
  int checkpoint_insn_Y;

  // Move some cycle core stats here instead of being global
  // 전역 변수 대신 여기에 사이클 통계를 저장
  unsigned long long gpu_sim_cycle;      // 현재 GPU 시뮬레이션 사이클 수 (현재 커널)
  unsigned long long gpu_tot_sim_cycle;  // 전체 GPU 시뮬레이션 사이클 수 (모든 커널 누적)

  /*
   * GPU 메모리 관련 함수들
   * 실제 GPU처럼 메모리를 할당하고, 데이터를 복사하는 기능
   */
  void *gpu_malloc(size_t size);                // GPU 메모리 할당 (malloc과 비슷)
  void *gpu_mallocarray(size_t count);          // GPU 배열 메모리 할당
  void gpu_memset(size_t dst_start_addr, int c, size_t count);  // GPU 메모리를 특정 값으로 채움

  /*
   * memcpy 함수들: 데이터를 복사하는 함수
   * CPU → GPU, GPU → CPU, GPU → GPU 방향의 복사를 시뮬레이션
   */
  void memcpy_to_gpu(size_t dst_start_addr, const void *src, size_t count);    // CPU → GPU 메모리 복사
  void memcpy_from_gpu(void *dst, size_t src_start_addr, size_t count);        // GPU → CPU 메모리 복사
  void memcpy_gpu_to_gpu(size_t dst, size_t src, size_t count);                // GPU → GPU 메모리 복사

  /*
   * 각 메모리 공간에 접근하는 getter 함수들
   */
  class memory_space *get_global_memory() {   // 글로벌 메모리 포인터 반환
    return m_global_mem;
  }
  class memory_space *get_tex_memory() {      // 텍스처 메모리 포인터 반환
    return m_tex_mem;
  }
  class memory_space *get_surf_memory() {     // 서피스 메모리 포인터 반환
    return m_surf_mem;
  }

  /*
   * 텍스처 바인딩 관련 함수들
   * 바인딩(binding): 텍스처 이름/참조를 실제 데이터(배열)에 연결하는 것
   * 마치 변수 이름에 값을 할당하는 것과 비슷합니다.
   */
  void gpgpu_ptx_sim_bindTextureToArray(const struct textureReference *texref,
                                        const struct cudaArray *array);         // 텍스처를 배열에 바인딩
  void gpgpu_ptx_sim_bindNameToTexture(const char *name,
                                       const struct textureReference *texref,
                                       int dim, int readmode, int ext);         // 텍스처 이름을 텍스처 참조에 바인딩
  void gpgpu_ptx_sim_unbindTexture(const struct textureReference *texref);     // 텍스처 바인딩 해제
  const char *gpgpu_ptx_sim_findNamefromTexture(
      const struct textureReference *texref);                                   // 텍스처 참조로 이름 찾기

  /*
   * get_texref: 텍스처 이름으로 텍스처 참조(textureReference)를 찾아 반환
   */
  const struct textureReference *get_texref(const std::string &texname) const {
    std::map<std::string,
             std::set<const struct textureReference *> >::const_iterator t =
        m_NameToTextureRef.find(texname);   // 이름으로 텍스처 참조 검색
    assert(t != m_NameToTextureRef.end());  // 찾지 못하면 오류
    return *(t->second.begin());            // 첫 번째 텍스처 참조 반환
  }

  /*
   * get_texarray: 텍스처 이름으로 CUDA 배열을 찾아 반환
   */
  const struct cudaArray *get_texarray(const std::string &texname) const {
    std::map<std::string, const struct cudaArray *>::const_iterator t =
        m_NameToCudaArray.find(texname);
    assert(t != m_NameToCudaArray.end());
    return t->second;
  }

  /*
   * get_texinfo: 텍스처 이름으로 텍스처 정보를 찾아 반환
   */
  const struct textureInfo *get_texinfo(const std::string &texname) const {
    std::map<std::string, const struct textureInfo *>::const_iterator t =
        m_NameToTextureInfo.find(texname);
    assert(t != m_NameToTextureInfo.end());
    return t->second;
  }

  /*
   * get_texattr: 텍스처 이름으로 텍스처 속성을 찾아 반환
   */
  const struct textureReferenceAttr *get_texattr(
      const std::string &texname) const {
    std::map<std::string, const struct textureReferenceAttr *>::const_iterator
        t = m_NameToAttribute.find(texname);
    assert(t != m_NameToAttribute.end());
    return t->second;
  }

  /*
   * get_config: 기능 시뮬레이션 설정을 반환
   * const &: 참조로 반환하여 복사를 피하고, 수정도 불가능하게 함
   */
  const gpgpu_functional_sim_config &get_config() const {
    return m_function_model_config;
  }
  FILE *get_ptx_inst_debug_file() { return ptx_inst_debug_file; }  // PTX 디버그 파일 포인터 반환

  //  These maps return the current texture mappings for the GPU at any given
  //  time.
  //  현재 시점에서 GPU의 텍스처 매핑 정보를 반환하는 함수들
  std::map<std::string, const struct cudaArray *> getNameArrayMapping() {
    return m_NameToCudaArray;          // 이름 → CUDA 배열 매핑 반환
  }
  std::map<std::string, const struct textureInfo *> getNameInfoMapping() {
    return m_NameToTextureInfo;        // 이름 → 텍스처 정보 매핑 반환
  }

  /*
   * virtual ~gpgpu_t(): 가상 소멸자
   * virtual을 붙이면 자식 클래스의 소멸자가 올바르게 호출됩니다.
   * 상속을 사용할 때는 항상 가상 소멸자를 만드는 것이 좋은 습관입니다.
   */
  virtual ~gpgpu_t() {}

 protected:  // 보호 영역: 이 클래스와 자식 클래스에서만 접근 가능
  const gpgpu_functional_sim_config &m_function_model_config;  // 기능 시뮬레이션 설정 참조
  FILE *ptx_inst_debug_file;                                    // PTX 명령어 디버그 파일 포인터

  class memory_space *m_global_mem;  // 글로벌 메모리 객체 포인터
  class memory_space *m_tex_mem;     // 텍스처 메모리 객체 포인터
  class memory_space *m_surf_mem;    // 서피스 메모리 객체 포인터

  unsigned long long m_dev_malloc;   // GPU 메모리 할당기의 현재 위치 (다음 할당 주소)

  //  These maps contain the current texture mappings for the GPU at any given
  //  time.
  //  현재 GPU의 텍스처 매핑 정보를 저장하는 맵(사전)들
  std::map<std::string, std::set<const struct textureReference *> >
      m_NameToTextureRef;                                           // 이름 → 텍스처 참조 집합
  std::map<const struct textureReference *, std::string> m_TextureRefToName;  // 텍스처 참조 → 이름 (역방향)
  std::map<std::string, const struct cudaArray *> m_NameToCudaArray;          // 이름 → CUDA 배열
  std::map<std::string, const struct textureInfo *> m_NameToTextureInfo;      // 이름 → 텍스처 정보
  std::map<std::string, const struct textureReferenceAttr *> m_NameToAttribute; // 이름 → 텍스처 속성
};

/*
 * ==========================================================================
 * gpgpu_ptx_sim_info 구조체
 * ==========================================================================
 * 커널의 자원 사용량 정보를 담는 구조체
 * PTX 컴파일러가 생성한 .ptxinfo 파일에서 읽어온 정보입니다.
 */
struct gpgpu_ptx_sim_info {
  // Holds properties of the kernel (Kernel's resource use).
  // 커널의 속성(자원 사용량)을 저장합니다.
  // These will be set to zero if a ptxinfo file is not present.
  // ptxinfo 파일이 없으면 모두 0으로 설정됩니다.
  int lmem;             // 로컬 메모리 사용량 (바이트): 각 스레드가 개인적으로 사용하는 메모리
  int smem;             // 공유 메모리 사용량 (바이트): CTA 내 스레드들이 공유하는 메모리
  int cmem;             // 상수 메모리 사용량 (바이트): 변하지 않는 데이터
  int gmem;             // 글로벌 메모리 사용량 (바이트): 모든 스레드가 접근 가능한 메모리
  int regs;             // 레지스터 사용량: 각 스레드가 사용하는 레지스터 수
  unsigned maxthreads;  // 최대 스레드 수: 이 커널이 지원하는 블록당 최대 스레드 수
  unsigned ptx_version; // PTX 버전: 사용된 PTX ISA(명령어 집합) 버전
  unsigned sm_target;   // 대상 SM 버전: 이 커널이 타겟하는 GPU 아키텍처 버전
};

/*
 * gpgpu_ptx_sim_arg: 커널에 전달하는 인자(argument) 하나를 나타내는 구조체
 * 커널을 호출할 때 전달하는 매개변수 정보입니다.
 * 예: kernel<<<grid, block>>>(arg1, arg2); 에서 arg1, arg2의 정보
 */
struct gpgpu_ptx_sim_arg {
  gpgpu_ptx_sim_arg() { m_start = NULL; }  // 기본 생성자: 시작 포인터를 NULL로
  gpgpu_ptx_sim_arg(const void *arg, size_t size, size_t offset) {  // 인자 정보를 받는 생성자
    m_start = arg;      // 인자 데이터의 시작 주소
    m_nbytes = size;    // 인자 데이터의 크기 (바이트)
    m_offset = offset;  // 파라미터 메모리에서의 위치(오프셋)
  }
  const void *m_start;  // 인자 데이터의 시작 포인터 (데이터가 어디에 있는지)
  size_t m_nbytes;       // 인자의 바이트 수 (크기)
  size_t m_offset;       // 파라미터 메모리 내에서의 오프셋(위치)
};

/*
 * gpgpu_ptx_sim_arg_list_t: 커널 인자들의 리스트(목록)
 * 커널에 전달하는 모든 인자를 연결 리스트로 저장
 */
typedef std::list<gpgpu_ptx_sim_arg> gpgpu_ptx_sim_arg_list_t;

/*
 * ==========================================================================
 * memory_space_t 클래스
 * ==========================================================================
 * 메모리 공간 타입과 뱅크 번호를 함께 저장하는 클래스
 * 단순한 enum보다 더 풍부한 정보를 담을 수 있습니다.
 * 예: ".const[2]"처럼 상수 메모리의 2번 뱅크를 나타낼 수 있습니다.
 */
class memory_space_t {
 public:
  memory_space_t() {              // 기본 생성자: 미정의 상태로 초기화
    m_type = undefined_space;     // 메모리 타입: 정의되지 않음
    m_bank = 0;                   // 뱅크 번호: 0
  }
  memory_space_t(const enum _memory_space_t &from) {  // enum에서 변환하는 생성자
    m_type = from;                // 전달받은 메모리 타입 설정
    m_bank = 0;                   // 뱅크는 기본 0
  }
  /*
   * 비교 연산자들: 두 memory_space_t를 비교할 수 있게 해줌
   * == : 같은지 비교
   * != : 다른지 비교
   * <  : 순서 비교 (std::map에서 키로 사용하기 위해 필요)
   */
  bool operator==(const memory_space_t &x) const {
    return (m_bank == x.m_bank) && (m_type == x.m_type);  // 뱅크와 타입이 모두 같으면 같음
  }
  bool operator!=(const memory_space_t &x) const { return !(*this == x); }  // ==의 반대
  bool operator<(const memory_space_t &x) const {  // 순서 비교 (타입 먼저, 그다음 뱅크)
    if (m_type < x.m_type)
      return true;
    else if (m_type > x.m_type)
      return false;
    else if (m_bank < x.m_bank)
      return true;
    return false;
  }
  enum _memory_space_t get_type() const { return m_type; }  // 메모리 타입 반환
  void set_type(enum _memory_space_t t) { m_type = t; }      // 메모리 타입 설정
  unsigned get_bank() const { return m_bank; }               // 뱅크 번호 반환
  void set_bank(unsigned b) { m_bank = b; }                  // 뱅크 번호 설정

  /*
   * 메모리 종류를 확인하는 편의 함수들
   */
  bool is_const() const {   // 상수 메모리인지 확인 (const_space 또는 커널 파라미터)
    return (m_type == const_space) || (m_type == param_space_kernel);
  }
  bool is_local() const {   // 로컬 메모리인지 확인 (local_space 또는 로컬 파라미터)
    return (m_type == local_space) || (m_type == param_space_local);
  }
  bool is_global() const { return (m_type == global_space); }  // 글로벌 메모리인지 확인

 private:
  enum _memory_space_t m_type;  // 메모리 공간 타입
  unsigned m_bank;  // n in ".const[n]"; note .const == .const[0] (see PTX 2.1
                    // manual, sec. 5.1.3)
                    // 뱅크 번호: ".const[n]"에서 n에 해당. .const는 .const[0]과 같음
};

/*
 * ==========================================================================
 * 메모리 접근(Memory Access) 관련 타입 정의
 * ==========================================================================
 * 메모리 접근 시 어떤 바이트/섹터에 접근하는지 추적하기 위한 타입들
 */
const unsigned MAX_MEMORY_ACCESS_SIZE = 128;  // 최대 메모리 접근 크기: 128바이트
typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;  // 바이트 마스크: 128비트, 각 비트가 1바이트를 나타냄

const unsigned SECTOR_CHUNCK_SIZE = 4;  // four sectors  // 섹터 청크 크기: 4개 섹터
const unsigned SECTOR_SIZE = 32;        // sector is 32 bytes width  // 섹터 크기: 32바이트
typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;  // 섹터 마스크: 4비트, 각 비트가 하나의 섹터

#define NO_PARTIAL_WRITE (mem_access_byte_mask_t())  // 부분 쓰기 없음: 빈 바이트 마스크

/*
 * MEM_ACCESS_TYPE_TUP_DEF: 메모리 접근 타입을 정의하는 매크로
 * 매크로를 사용하여 enum을 자동 생성하는 기법입니다.
 * 이 방법을 사용하면 enum 값의 이름을 문자열로 변환하기 쉽습니다.
 *
 * 각 메모리 접근 타입의 의미:
 *   GLOBAL_ACC_R: 글로벌 메모리 읽기
 *   LOCAL_ACC_R: 로컬 메모리 읽기
 *   CONST_ACC_R: 상수 메모리 읽기
 *   TEXTURE_ACC_R: 텍스처 메모리 읽기
 *   GLOBAL_ACC_W: 글로벌 메모리 쓰기
 *   LOCAL_ACC_W: 로컬 메모리 쓰기
 *   L1_WRBK_ACC: L1 캐시 쓰기 되돌리기 (write-back)
 *   L2_WRBK_ACC: L2 캐시 쓰기 되돌리기
 *   INST_ACC_R: 명령어 읽기
 *   L1_WR_ALLOC_R: L1 쓰기 할당 읽기
 *   L2_WR_ALLOC_R: L2 쓰기 할당 읽기
 */
#define MEM_ACCESS_TYPE_TUP_DEF                                         \
  MA_TUP_BEGIN(mem_access_type)                                         \
  MA_TUP(GLOBAL_ACC_R), MA_TUP(LOCAL_ACC_R), MA_TUP(CONST_ACC_R),       \
      MA_TUP(TEXTURE_ACC_R), MA_TUP(GLOBAL_ACC_W), MA_TUP(LOCAL_ACC_W), \
      MA_TUP(L1_WRBK_ACC), MA_TUP(L2_WRBK_ACC), MA_TUP(INST_ACC_R),     \
      MA_TUP(L1_WR_ALLOC_R), MA_TUP(L2_WR_ALLOC_R),                     \
      MA_TUP(NUM_MEM_ACCESS_TYPE) MA_TUP_END(mem_access_type)

/*
 * 아래 매크로들은 위의 MEM_ACCESS_TYPE_TUP_DEF를 enum으로 변환합니다.
 * MA_TUP_BEGIN(X) → enum X {
 * MA_TUP(X) → X (단순히 이름 그대로)
 * MA_TUP_END(X) → };
 */
#define MA_TUP_BEGIN(X) enum X {
#define MA_TUP(X) X
#define MA_TUP_END(X) \
  }                   \
  ;
MEM_ACCESS_TYPE_TUP_DEF  // 위에서 정의한 매크로를 실행하여 mem_access_type enum 생성
#undef MA_TUP_BEGIN       // 매크로 정의 해제 (다른 곳에서 다시 정의할 수 있도록)
#undef MA_TUP             // 매크로 정의 해제
#undef MA_TUP_END         // 매크로 정의 해제

/*
 * mem_access_type_str: 메모리 접근 타입을 읽을 수 있는 문자열로 변환
 * 예: GLOBAL_ACC_R → "GLOBAL_ACC_R"
 * 디버깅이나 로그 출력에 유용합니다.
 */
const char *mem_access_type_str(enum mem_access_type access_type);

/*
 * cache_operator_type: 캐시 연산 타입
 * PTX 명령어에서 캐시 동작 방식을 지정하는 힌트(hint)입니다.
 * .ca, .lu, .cv 등은 PTX 어셈블리 명령어의 접미사입니다.
 */
enum cache_operator_type {
  CACHE_UNDEFINED,     // 캐시 연산 미정의

  // loads (로드/읽기 전용)
  CACHE_ALL,       // .ca: 모든 캐시 레벨에 캐시 (기본 동작)
  CACHE_LAST_USE,  // .lu: 마지막 사용 - 이 데이터를 캐시에서 곧 제거해도 됨
  CACHE_VOLATILE,  // .cv: 휘발성 - 캐시를 무시하고 항상 메모리에서 직접 읽음
  CACHE_L1,        // .nc: L1 캐시에만 캐시 (비일관성 허용)

  // loads and stores (로드/스토어 모두)
  CACHE_STREAMING,  // .cs: 스트리밍 - 한 번만 쓰는 데이터에 최적화
  CACHE_GLOBAL,     // .cg: 글로벌 - L2에만 캐시, L1은 건너뜀

  // stores (스토어/쓰기 전용)
  CACHE_WRITE_BACK,    // .wb: 쓰기 되돌리기 - 캐시에만 쓰고 나중에 메모리에 반영
  CACHE_WRITE_THROUGH  // .wt: 쓰기 관통 - 캐시와 메모리에 동시에 씀
};

/*
 * ==========================================================================
 * mem_access_t 클래스
 * ==========================================================================
 * 하나의 메모리 접근 요청을 나타내는 클래스
 * 메모리 주소, 크기, 읽기/쓰기 여부, 접근 타입 등의 정보를 담습니다.
 * 이 정보는 캐시와 메모리 시스템 시뮬레이션에 사용됩니다.
 */
class mem_access_t {
 public:
  mem_access_t(gpgpu_context *ctx) { init(ctx); }  // 기본 생성자: 컨텍스트로 초기화

  /*
   * 간단한 생성자: 타입, 주소, 크기, 읽기/쓰기 여부만 지정
   */
  mem_access_t(mem_access_type type, new_addr_type address, unsigned size,
               bool wr, gpgpu_context *ctx) {
    init(ctx);          // 기본 초기화
    m_type = type;      // 메모리 접근 타입 (글로벌 읽기, 로컬 쓰기 등)
    m_addr = address;   // 접근할 메모리 주소
    m_req_size = size;  // 요청 크기 (바이트)
    m_write = wr;       // true면 쓰기, false면 읽기
  }

  /*
   * 완전한 생성자: 워프 마스크, 바이트 마스크, 섹터 마스크도 함께 지정
   * 이 마스크들은 어떤 스레드가 어떤 바이트에 접근하는지 추적합니다.
   */
  mem_access_t(mem_access_type type, new_addr_type address, unsigned size,
               bool wr, const active_mask_t &active_mask,
               const mem_access_byte_mask_t &byte_mask,
               const mem_access_sector_mask_t &sector_mask, gpgpu_context *ctx)
      : m_warp_mask(active_mask),    // 이 접근에 참여하는 스레드들의 마스크
        m_byte_mask(byte_mask),      // 접근하는 바이트들의 마스크
        m_sector_mask(sector_mask) { // 접근하는 섹터들의 마스크
    init(ctx);
    m_type = type;
    m_addr = address;
    m_req_size = size;
    m_write = wr;
  }

  /*
   * Getter 함수들: 멤버 변수의 값을 반환
   */
  new_addr_type get_addr() const { return m_addr; }         // 메모리 주소 반환
  void set_addr(new_addr_type addr) { m_addr = addr; }       // 메모리 주소 설정
  unsigned get_size() const { return m_req_size; }            // 요청 크기 반환
  const active_mask_t &get_warp_mask() const { return m_warp_mask; }  // 워프 마스크 반환
  bool is_write() const { return m_write; }                   // 쓰기 연산인지 여부
  enum mem_access_type get_type() const { return m_type; }    // 접근 타입 반환
  mem_access_byte_mask_t get_byte_mask() const { return m_byte_mask; }      // 바이트 마스크 반환
  mem_access_sector_mask_t get_sector_mask() const { return m_sector_mask; } // 섹터 마스크 반환

  /*
   * print: 메모리 접근 정보를 파일에 출력 (디버깅용)
   * 주소, 읽기/쓰기, 크기, 접근 타입을 출력합니다.
   */
  void print(FILE *fp) const {
    fprintf(fp, "addr=0x%llx, %s, size=%u, ", m_addr,
            m_write ? "store" : "load ", m_req_size);  // 주소, 종류, 크기 출력
    switch (m_type) {                                   // 접근 타입에 따라 문자열 출력
      case GLOBAL_ACC_R:
        fprintf(fp, "GLOBAL_R");    // 글로벌 읽기
        break;
      case LOCAL_ACC_R:
        fprintf(fp, "LOCAL_R ");    // 로컬 읽기
        break;
      case CONST_ACC_R:
        fprintf(fp, "CONST   ");   // 상수 읽기
        break;
      case TEXTURE_ACC_R:
        fprintf(fp, "TEXTURE ");   // 텍스처 읽기
        break;
      case GLOBAL_ACC_W:
        fprintf(fp, "GLOBAL_W");   // 글로벌 쓰기
        break;
      case LOCAL_ACC_W:
        fprintf(fp, "LOCAL_W ");   // 로컬 쓰기
        break;
      case L2_WRBK_ACC:
        fprintf(fp, "L2_WRBK ");  // L2 캐시 쓰기 되돌리기
        break;
      case INST_ACC_R:
        fprintf(fp, "INST    ");   // 명령어 읽기
        break;
      case L1_WRBK_ACC:
        fprintf(fp, "L1_WRBK ");  // L1 캐시 쓰기 되돌리기
        break;
      default:
        fprintf(fp, "unknown ");   // 알 수 없는 타입
        break;
    }
  }

  gpgpu_context *gpgpu_ctx;  // GPU 시뮬레이터 컨텍스트 포인터

 private:
  void init(gpgpu_context *ctx);  // 초기화 함수 (구현은 .cc 파일에)

  unsigned m_uid;              // 이 메모리 접근의 고유 ID
  new_addr_type m_addr;        // request address - 요청 메모리 주소
  bool m_write;                // true: 쓰기, false: 읽기
  unsigned m_req_size;         // bytes - 요청 크기 (바이트 단위)
  mem_access_type m_type;      // 메모리 접근 타입 (글로벌 읽기/쓰기, 로컬 등)
  active_mask_t m_warp_mask;   // 이 접근에 참여하는 스레드들의 마스크
  mem_access_byte_mask_t m_byte_mask;    // 접근하는 바이트들의 마스크
  mem_access_sector_mask_t m_sector_mask; // 접근하는 섹터들의 마스크
};

/*
 * mem_fetch: 메모리 페치(가져오기) 요청을 나타내는 클래스 (전방 선언)
 * 메모리 시스템을 통해 이동하는 요청 패킷입니다.
 */
class mem_fetch;

/*
 * ==========================================================================
 * mem_fetch_interface 클래스 (추상 인터페이스)
 * ==========================================================================
 * 메모리 페치를 주고받는 인터페이스(약속)를 정의합니다.
 *
 * 인터페이스란?
 *   "이런 기능을 반드시 제공해야 해"라는 약속입니다.
 *   실제 구현은 이 인터페이스를 상속받는 클래스에서 합니다.
 *   순수 가상 함수(= 0)만 있으므로 이 클래스의 객체는 직접 만들 수 없습니다.
 */
class mem_fetch_interface {
 public:
  virtual bool full(unsigned size, bool write) const = 0;  // 버퍼가 가득 찼는지 확인 (순수 가상)
  virtual void push(mem_fetch *mf) = 0;                     // 메모리 페치 요청을 버퍼에 추가 (순수 가상)
};

/*
 * ==========================================================================
 * mem_fetch_allocator 클래스 (추상 인터페이스)
 * ==========================================================================
 * 메모리 페치 객체를 생성(할당)하는 인터페이스
 * 여러 종류의 alloc 함수를 제공하여 다양한 상황에서 mem_fetch를 만들 수 있습니다.
 */
class mem_fetch_allocator {
 public:
  /*
   * alloc: 메모리 페치 객체를 생성하는 팩토리 함수들
   * 여러 오버로드(같은 이름, 다른 매개변수)를 제공합니다.
   */
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned long long streamID) const = 0;  // 기본 할당
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const = 0;      // 명령어 기반 할당
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned wid, unsigned sid, unsigned tpc,
                           mem_fetch *original_mf,
                           unsigned long long streamID) const = 0;   // 상세 정보 포함 할당
};

// the maximum number of destination, source, or address uarch operands in a
// instruction
// 하나의 명령어에서 사용할 수 있는 최대 레지스터 피연산자 수
#define MAX_REG_OPERANDS 32

/*
 * dram_callback_t: DRAM(메인 메모리) 콜백 구조체
 * 콜백(callback)이란 "나중에 이 함수를 불러줘"라고 미리 등록해두는 것입니다.
 * 메모리 요청이 완료되면 콜백 함수가 호출됩니다.
 */
struct dram_callback_t {
  dram_callback_t() {        // 기본 생성자: 모든 포인터를 NULL로
    function = NULL;          // 콜백 함수: 아직 등록 안 됨
    instruction = NULL;       // 관련 명령어: 없음
    thread = NULL;            // 관련 스레드: 없음
  }
  void (*function)(const class inst_t *, class ptx_thread_info *);  // 콜백 함수 포인터
  // 함수 포인터란 함수를 변수처럼 저장할 수 있는 것입니다.

  const class inst_t *instruction;        // 이 콜백과 관련된 명령어
  class ptx_thread_info *thread;          // 이 콜백과 관련된 스레드
};

/*
 * ==========================================================================
 * inst_t 클래스
 * ==========================================================================
 * 하나의 GPU 명령어(instruction)를 나타내는 클래스
 * 명령어란 GPU가 실행하는 하나의 동작입니다.
 * 예: "두 숫자를 더해라", "메모리에서 데이터를 읽어라"
 *
 * 이 클래스는 명령어의 모든 속성을 담고 있습니다:
 * - 어떤 종류의 연산인지 (더하기, 곱하기, 메모리 읽기 등)
 * - 어떤 레지스터를 사용하는지
 * - 실행에 몇 사이클이 걸리는지
 * - 어떤 메모리에 접근하는지
 */
class inst_t {
 public:
  /*
   * 기본 생성자: 모든 멤버를 기본값으로 초기화
   */
  inst_t() {
    m_decoded = false;                   // 아직 디코딩(해석)되지 않은 상태
    pc = (address_type)-1;               // 프로그램 카운터: 유효하지 않은 값 (-1)
    reconvergence_pc = (address_type)-1; // 재수렴 PC: 유효하지 않은 값
    op = NO_OP;                          // 연산 종류: 없음
    bar_type = NOT_BAR;                  // 배리어 타입: 배리어 아님
    red_type = NOT_RED;                  // 리덕션 타입: 리덕션 아님
    bar_id = (unsigned)-1;               // 배리어 ID: 유효하지 않음
    bar_count = (unsigned)-1;            // 배리어 카운트: 유효하지 않음
    oprnd_type = UN_OP;                  // 피연산자 타입: 알 수 없음
    sp_op = OTHER_OP;                    // 특수 연산: 기타
    op_pipe = UNKOWN_OP;                 // 파이프라인: 알 수 없음
    mem_op = NOT_TEX;                    // 메모리 연산: 텍스처 아님
    const_cache_operand = 0;             // 상수 캐시 피연산자: 없음
    num_operands = 0;                    // 피연산자 수: 0
    num_regs = 0;                        // 레지스터 수: 0
    memset(out, 0, sizeof(unsigned));    // 출력 레지스터 배열을 0으로 초기화
    memset(in, 0, sizeof(unsigned));     // 입력 레지스터 배열을 0으로 초기화
    is_vectorin = 0;                     // 벡터 입력: 아님
    is_vectorout = 0;                    // 벡터 출력: 아님
    space = memory_space_t();            // 메모리 공간: 기본값(미정의)
    cache_op = CACHE_UNDEFINED;          // 캐시 연산: 미정의
    latency = 1;                         // 지연 시간: 1 사이클 (기본)
    initiation_interval = 1;             // 발행 간격: 1 사이클 (기본)
    for (unsigned i = 0; i < MAX_REG_OPERANDS; i++) {  // 아키텍처 레지스터 초기화
      arch_reg.src[i] = -1;             // 소스 레지스터: 유효하지 않음
      arch_reg.dst[i] = -1;             // 목적지 레지스터: 유효하지 않음
    }
    isize = 0;                           // 명령어 크기: 0바이트
  }

  bool valid() const { return m_decoded; }  // 이 명령어가 유효한(디코딩된) 명령어인지 확인

  /*
   * print_insn: 명령어 정보를 파일에 출력 (디버깅용)
   * virtual: 자식 클래스에서 재정의(오버라이드) 가능
   */
  virtual void print_insn(FILE *fp) const {
    fprintf(fp, " [inst @ pc=0x%04llx] ", pc);  // 명령어의 PC 주소를 출력
  }

  /*
   * 명령어 종류를 확인하는 함수들
   * 각 함수는 이 명령어가 특정 종류인지 bool(true/false)로 반환합니다.
   */
  bool is_load() const {      // 메모리 로드(읽기) 명령어인지 확인
    return (op == LOAD_OP || op == TENSOR_CORE_LOAD_OP ||
            memory_op == memory_load);
  }
  bool is_store() const {     // 메모리 스토어(쓰기) 명령어인지 확인
    return (op == STORE_OP || op == TENSOR_CORE_STORE_OP ||
            memory_op == memory_store);
  }

  bool is_fp() const { return ((sp_op == FP__OP)); }        // 단정밀도 부동소수점 연산인지
  bool is_fpdiv() const { return ((sp_op == FP_DIV_OP)); }  // 단정밀도 나눗셈인지
  bool is_fpmul() const { return ((sp_op == FP_MUL_OP)); }  // 단정밀도 곱셈인지
  bool is_dp() const { return ((sp_op == DP___OP)); }       // 배정밀도 부동소수점 연산인지
  bool is_dpdiv() const { return ((sp_op == DP_DIV_OP)); }  // 배정밀도 나눗셈인지
  bool is_dpmul() const { return ((sp_op == DP_MUL_OP)); }  // 배정밀도 곱셈인지
  bool is_imul() const { return ((sp_op == INT_MUL_OP)); }  // 정수 곱셈인지
  bool is_imul24() const { return ((sp_op == INT_MUL24_OP)); }  // 24비트 정수 곱셈인지
  bool is_imul32() const { return ((sp_op == INT_MUL32_OP)); }  // 32비트 정수 곱셈인지
  bool is_idiv() const { return ((sp_op == INT_DIV_OP)); }      // 정수 나눗셈인지
  bool is_sfu() const {   // SFU(특수함수장치) 연산인지 확인 (sin, cos, 루트, 지수, 텐서)
    return ((sp_op == FP_SQRT_OP) || (sp_op == FP_LG_OP) ||
            (sp_op == FP_SIN_OP) || (sp_op == FP_EXP_OP) ||
            (sp_op == TENSOR__OP));
  }
  bool is_alu() const { return (sp_op == INT__OP); }  // 정수 ALU 연산인지

  unsigned get_num_operands() const { return num_operands; }  // 피연산자 수 반환
  unsigned get_num_regs() const { return num_regs; }          // 레지스터 수 반환
  void set_num_regs(unsigned num) { num_regs = num; }         // 레지스터 수 설정
  void set_num_operands(unsigned num) { num_operands = num; } // 피연산자 수 설정
  void set_bar_id(unsigned id) { bar_id = id; }               // 배리어 ID 설정
  void set_bar_count(unsigned count) { bar_count = count; }   // 배리어 카운트 설정

  /*
   * 명령어의 주요 속성 멤버 변수들 (public으로 직접 접근 가능)
   */
  address_type pc;  // program counter address of instruction - 이 명령어의 메모리 주소 (PC)
  unsigned isize;   // size of instruction in bytes - 명령어 크기 (바이트 단위)
  op_type op;       // opcode (uarch visible) - 연산 코드: 마이크로아키텍처에서 보이는 연산 종류

  barrier_type bar_type;     // 배리어 종류 (동기화, 도착, 리덕션)
  reduction_type red_type;   // 리덕션 종류 (POPC, AND, OR)
  unsigned bar_id;           // 배리어 ID: 어떤 배리어인지 식별
  unsigned bar_count;        // 배리어에 참여하는 스레드 수

  types_of_operands oprnd_type;  // code (uarch visible) identify if the
                                 // operation is an interger or a floating point
                                 // 피연산자 타입: 정수인지 부동소수점인지
  special_ops
      sp_op;  // code (uarch visible) identify if int_alu, fp_alu, int_mul ....
              // 특수 연산 종류: 정수ALU, 실수ALU, 정수곱셈 등
  operation_pipeline op_pipe;  // code (uarch visible) identify the pipeline of
                               // the operation (SP, SFU or MEM)
                               // 이 명령어가 사용하는 파이프라인 (SP, SFU, MEM 등)
  mem_operation mem_op;        // code (uarch visible) identify memory type
                               // 메모리 연산 종류 (텍스처인지 아닌지)
  bool const_cache_operand;    // has a load from constant memory as an operand
                               // 상수 메모리에서 로드하는 피연산자가 있는지
  _memory_op_t memory_op;     // memory_op used by ptxplus
                               // PTXPlus에서 사용하는 메모리 연산 타입
  unsigned num_operands;       // 피연산자 수
  unsigned num_regs;  // count vector operand as one register operand
                      // 레지스터 수 (벡터 피연산자는 하나로 셈)

  address_type reconvergence_pc;  // -1 => not a branch, -2 => use function
                                  // return address
                                  // 재수렴 PC: -1이면 분기 아님, -2면 함수 반환 주소 사용

  unsigned out[8];             // 출력 레지스터 번호 배열 (최대 8개)
  unsigned outcount;           // 출력 레지스터 개수
  unsigned in[24];             // 입력 레지스터 번호 배열 (최대 24개)
  unsigned incount;            // 입력 레지스터 개수
  unsigned char is_vectorin;   // 벡터 입력 여부 (0: 아님, 1: 벡터)
  unsigned char is_vectorout;  // 벡터 출력 여부
  int pred;  // predicate register number - 조건 레지스터 번호
  // 조건 레지스터(predicate): 명령어를 실행할지 말지를 결정하는 조건 값
  int ar1, ar2;                // 주소 레지스터 번호 (메모리 접근 주소 계산에 사용)

  // register number for bank conflict evaluation
  // 뱅크 충돌 평가를 위한 레지스터 번호
  struct {
    int dst[MAX_REG_OPERANDS];  // 목적지(결과를 저장할) 레지스터 번호 배열
    int src[MAX_REG_OPERANDS];  // 소스(입력값을 읽을) 레지스터 번호 배열
  } arch_reg;
  // int arch_reg[MAX_REG_OPERANDS]; // 이전 버전 (주석 처리됨)

  unsigned latency;  // operation latency - 이 연산이 완료되는 데 걸리는 사이클 수
  unsigned initiation_interval;  // 같은 파이프라인에 다음 명령어를 보낼 수 있을 때까지의 사이클 수

  unsigned data_size;  // what is the size of the word being operated on?
                       // 연산하는 데이터의 크기 (바이트 단위)
  memory_space_t space;          // 이 명령어가 접근하는 메모리 공간의 종류
  cache_operator_type cache_op;  // 캐시 연산 타입 (캐시 힌트)

 protected:  // 보호 영역: 이 클래스와 자식 클래스에서만 접근 가능
  bool m_decoded;  // 이 명령어가 디코딩(해석)되었는지 여부
  virtual void pre_decode() {}  // 디코딩 전 처리 함수 (자식 클래스에서 재정의 가능)
};

/*
 * divergence_support_t: GPU의 분기 발산(divergence) 처리 방식
 * POST_DOMINATOR: 포스트 도미네이터 기반 재수렴 (MICRO'07 논문)
 *   - 분기 후 다시 합류하는 지점(포스트 도미네이터)을 찾아서 재수렴
 */
enum divergence_support_t { POST_DOMINATOR = 1, NUM_SIMD_MODEL };

/*
 * MAX_ACCESSES_PER_INSN_PER_THREAD: 하나의 명령어에서 하나의 스레드가
 * 수행할 수 있는 최대 메모리 접근 수 (8번)
 * 예: 32바이트 접근을 4바이트씩 8번에 나눠서 할 수 있음
 */
const unsigned MAX_ACCESSES_PER_INSN_PER_THREAD = 8;

/*
 * ==========================================================================
 * warp_inst_t 클래스 (inst_t 상속)
 * ==========================================================================
 * 워프 수준의 명령어를 나타내는 클래스
 * inst_t를 상속(extends)받아 워프 실행에 필요한 정보를 추가합니다.
 *
 * 상속(inheritance)이란?
 *   기존 클래스(부모, inst_t)의 모든 기능을 물려받고,
 *   새로운 기능을 추가하는 것입니다.
 *   마치 "자동차" 클래스를 만들고, "전기 자동차" 클래스가 이를 상속받는 것과 같습니다.
 *
 * inst_t: 하나의 명령어 정보 (개별 스레드 수준)
 * warp_inst_t: 워프(32개 스레드) 전체의 명령어 실행 정보
 */
class warp_inst_t : public inst_t {  // inst_t를 public으로 상속
 public:
  // constructors (생성자들)

  /*
   * 기본 생성자: 설정 없이 기본값으로 초기화
   */
  warp_inst_t() {
    m_uid = 0;                                  // 고유 ID: 0
    m_streamID = (unsigned long long)-1;        // 스트림 ID: 유효하지 않음
    m_empty = true;                             // 비어있는 상태 (명령어 없음)
    m_config = NULL;                            // 설정 포인터: 없음

    // Ni: (Ni라는 개발자가 추가한 필드들)
    m_is_ldgsts = false;     // ldgsts 명령어인지: 아님 (비동기 글로벌→공유 메모리 복사)
    m_is_ldgdepbar = false;  // ldgdepbar 명령어인지: 아님 (의존성 배리어)
    m_is_depbar = false;     // depbar 명령어인지: 아님 (의존성 배리어)

    m_depbar_group_no = 0;   // 의존성 배리어 그룹 번호: 0
  }

  /*
   * 설정을 받는 생성자: 코어 설정으로 초기화
   */
  warp_inst_t(const core_config *config) {
    m_uid = 0;
    m_streamID = (unsigned long long)-1;
    assert(config->warp_size <= MAX_WARP_SIZE);  // 워프 크기가 최대를 넘지 않는지 확인
    m_config = config;                            // 코어 설정 저장
    m_empty = true;                               // 비어있는 상태
    m_isatomic = false;                           // 원자적(atomic) 연산 아님
    // 원자적 연산: 여러 스레드가 동시에 같은 데이터에 접근해도 안전한 연산
    m_per_scalar_thread_valid = false;            // 스레드별 정보가 아직 유효하지 않음
    m_mem_accesses_created = false;               // 메모리 접근이 아직 생성되지 않음
    m_cache_hit = false;                          // 캐시 히트(적중) 아님
    m_is_printf = false;                          // printf 명령어 아님
    m_is_cdp = 0;                                 // CDP(동적 병렬성) 관련 아님
    should_do_atomic = true;                      // 원자적 연산을 수행해야 하는지

    // Ni:
    m_is_ldgsts = false;
    m_is_ldgdepbar = false;
    m_is_depbar = false;

    m_depbar_group_no = 0;
  }
  virtual ~warp_inst_t() {}  // 가상 소멸자: 자식 클래스가 안전하게 삭제되도록

  // modifiers (상태를 변경하는 함수들)

  void broadcast_barrier_reduction(const active_mask_t &access_mask);  // 배리어 리덕션 결과를 브로드캐스트(전파)
  void do_atomic(bool forceDo = false);                                 // 원자적 연산 수행
  void do_atomic(const active_mask_t &access_mask, bool forceDo = false);  // 특정 스레드들의 원자적 연산 수행
  void clear() { m_empty = true; }  // 명령어를 비움 (재사용을 위해)

  /*
   * issue: 명령어를 발행(실행 시작)
   * mask: 실행할 스레드들의 활성 마스크
   * warp_id: 이 워프의 ID
   * cycle: 발행 시점의 사이클
   * dynamic_warp_id: 동적 워프 ID
   * sch_id: 이 명령어를 발행한 스케줄러 ID
   * streamID: 스트림 ID
   */
  void issue(const active_mask_t &mask, unsigned warp_id,
             unsigned long long cycle, int dynamic_warp_id, int sch_id,
             unsigned long long streamID);

  const active_mask_t &get_active_mask() const { return m_warp_active_mask; }  // 현재 활성 마스크 반환
  void completed(unsigned long long cycle)
      const;  // stat collection: called when the instruction is completed
              // 명령어 완료 시 호출: 통계 수집용

  /*
   * set_addr: 특정 스레드의 메모리 접근 주소를 설정
   * n: 스레드 번호 (워프 내에서의 위치, 0~31)
   * addr: 메모리 주소
   */
  void set_addr(unsigned n, new_addr_type addr) {
    if (!m_per_scalar_thread_valid) {                    // 스레드별 정보가 아직 없으면
      m_per_scalar_thread.resize(m_config->warp_size);   // 워프 크기만큼 배열 할당
      m_per_scalar_thread_valid = true;                  // 유효하게 표시
    }
    m_per_scalar_thread[n].memreqaddr[0] = addr;         // 첫 번째 메모리 요청 주소 설정
  }
  /*
   * set_addr (오버로드): 여러 주소를 한번에 설정 (다중 접근용)
   * num_addrs: 주소의 개수
   */
  void set_addr(unsigned n, new_addr_type *addr, unsigned num_addrs) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(m_config->warp_size);
      m_per_scalar_thread_valid = true;
    }
    assert(num_addrs <= MAX_ACCESSES_PER_INSN_PER_THREAD);  // 최대 접근 수 초과 방지
    for (unsigned i = 0; i < num_addrs; i++)
      m_per_scalar_thread[n].memreqaddr[i] = addr[i];       // 각 주소를 설정
  }

  /*
   * print_m_accessq: 생성된 메모리 접근 요청들을 출력 (디버깅용)
   */
  void print_m_accessq() {
    if (accessq_empty())            // 접근 큐가 비어있으면 아무것도 안 함
      return;
    else {
      printf("Printing mem access generated\n");  // "생성된 메모리 접근 출력" 메시지
      std::list<mem_access_t>::iterator it;        // 리스트 반복자
      for (it = m_accessq.begin(); it != m_accessq.end(); ++it) {  // 모든 접근을 순회
        printf("MEM_TXN_GEN:%s:%llx, Size:%d \n",
               mem_access_type_str(it->get_type()), it->get_addr(),  // 타입, 주소, 크기 출력
               it->get_size());
      }
    }
  }

  /*
   * transaction_info: 메모리 트랜잭션(거래) 정보 구조체
   * 메모리 합치기(coalescing) 과정에서 생성되는 정보입니다.
   * 여러 스레드의 메모리 요청이 하나의 트랜잭션으로 합쳐집니다.
   */
  struct transaction_info {
    std::bitset<4> chunks;  // bitmask: 32-byte chunks accessed - 접근된 32바이트 청크 비트마스크
    mem_access_byte_mask_t bytes;  // 접근된 바이트들의 마스크
    active_mask_t active;  // threads in this transaction - 이 트랜잭션에 참여하는 스레드들

    /*
     * test_bytes: 특정 범위의 바이트가 접근되었는지 테스트
     */
    bool test_bytes(unsigned start_bit, unsigned end_bit) {
      for (unsigned i = start_bit; i <= end_bit; i++)
        if (bytes.test(i)) return true;   // 하나라도 접근되었으면 true
      return false;                        // 아무것도 접근되지 않았으면 false
    }
  };

  /*
   * 메모리 접근 생성 및 합치기(coalescing) 관련 함수들
   * 메모리 합치기: 워프 내 여러 스레드의 메모리 요청을 하나로 합쳐서
   * 효율적으로 처리하는 GPU의 중요한 최적화 기법
   */
  void generate_mem_accesses();  // 각 스레드의 메모리 접근을 생성
  void memory_coalescing_arch(bool is_write, mem_access_type access_type);  // 메모리 합치기 수행
  void memory_coalescing_arch_atomic(bool is_write,
                                     mem_access_type access_type);          // 원자적 연산의 메모리 합치기
  void memory_coalescing_arch_reduce_and_send(bool is_write,
                                              mem_access_type access_type,
                                              const transaction_info &info,
                                              new_addr_type addr,
                                              unsigned segment_size);       // 합치기 결과를 메모리에 전송

  /*
   * add_callback: 특정 스레드(lane)에 콜백 함수를 등록
   * 메모리 요청이 완료되면 이 콜백이 호출됩니다.
   * lane_id: 워프 내 스레드 번호 (0~31)
   */
  void add_callback(unsigned lane_id,
                    void (*function)(const class inst_t *,
                                     class ptx_thread_info *),
                    const inst_t *inst, class ptx_thread_info *thread,
                    bool atomic) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(m_config->warp_size);
      m_per_scalar_thread_valid = true;
      if (atomic) m_isatomic = true;  // 원자적 연산이면 플래그 설정
    }
    m_per_scalar_thread[lane_id].callback.function = function;      // 콜백 함수 등록
    m_per_scalar_thread[lane_id].callback.instruction = inst;       // 관련 명령어
    m_per_scalar_thread[lane_id].callback.thread = thread;          // 관련 스레드
  }
  void set_active(const active_mask_t &active);     // 활성 마스크 설정

  void clear_active(const active_mask_t &inactive); // 비활성 스레드 제거
  void set_not_active(unsigned lane_id);             // 특정 스레드를 비활성화

  // accessors (읽기 전용 접근 함수들)

  /*
   * print_insn: 명령어 정보와 활성 마스크를 출력 (디버깅용)
   * inst_t의 print_insn을 오버라이드(재정의)합니다.
   */
  virtual void print_insn(FILE *fp) const {
    fprintf(fp, " [inst @ pc=0x%04llx] ", pc);
    for (int i = (int)m_config->warp_size - 1; i >= 0; i--)  // 워프의 각 스레드에 대해
      fprintf(fp, "%c", ((m_warp_active_mask[i]) ? '1' : '0'));  // 활성이면 '1', 비활성이면 '0' 출력
  }
  bool active(unsigned thread) const { return m_warp_active_mask.test(thread); }  // 특정 스레드가 활성인지
  unsigned active_count() const { return m_warp_active_mask.count(); }  // 활성 스레드 수
  unsigned issued_count() const {       // 발행된 스레드 수 (명령어 카운팅용)
    assert(m_empty == false);           // 비어있으면 안 됨
    return m_warp_issued_mask.count();
  }  // for instruction counting
  bool empty() const { return m_empty; }  // 이 명령어 슬롯이 비어있는지
  unsigned warp_id() const {              // 워프 ID 반환
    assert(!m_empty);                      // 비어있으면 안 됨 (안전 검사)
    return m_warp_id;
  }
  unsigned warp_id_func() const  // to be used in functional simulations only
  // 기능 시뮬레이션에서만 사용하는 워프 ID (assert 없음)
  {
    return m_warp_id;
  }
  unsigned dynamic_warp_id() const {  // 동적 워프 ID 반환
    assert(!m_empty);
    return m_dynamic_warp_id;
  }
  /*
   * has_callback: 특정 스레드에 콜백이 등록되어 있는지 확인
   */
  bool has_callback(unsigned n) const {
    return m_warp_active_mask[n] && m_per_scalar_thread_valid &&
           (m_per_scalar_thread[n].callback.function != NULL);
  }
  /*
   * get_addr: 특정 스레드의 메모리 접근 주소를 반환
   */
  new_addr_type get_addr(unsigned n) const {
    assert(m_per_scalar_thread_valid);
    return m_per_scalar_thread[n].memreqaddr[0];
  }

  bool isatomic() const { return m_isatomic; }  // 원자적 연산인지 확인

  unsigned warp_size() const { return m_config->warp_size; }  // 워프 크기 반환

  /*
   * 메모리 접근 큐(accessq) 관련 함수들
   * 접근 큐: 생성된 메모리 접근 요청들이 대기하는 큐(줄)
   */
  bool accessq_empty() const { return m_accessq.empty(); }    // 접근 큐가 비어있는지
  unsigned accessq_count() const { return m_accessq.size(); }  // 접근 큐에 있는 요청 수
  const mem_access_t &accessq_back() { return m_accessq.back(); }  // 큐의 마지막 요청
  void accessq_pop_back() { m_accessq.pop_back(); }                // 큐에서 마지막 요청 제거

  /*
   * dispatch_delay: 발행 지연을 처리
   * cycles가 0보다 크면 1 감소시키고, 아직 지연 중이면 true 반환
   * 이 함수는 initiation_interval을 시뮬레이션합니다.
   */
  bool dispatch_delay() {
    if (cycles > 0) cycles--;       // 남은 사이클 1 감소
    return cycles > 0;               // 아직 지연 중이면 true
  }

  bool has_dispatch_delay() { return cycles > 0; }  // 발행 지연이 남아있는지

  void print(FILE *fout) const;                      // 명령어 정보를 파일에 출력
  unsigned get_uid() const { return m_uid; }         // 고유 ID 반환
  unsigned long long get_streamID() const { return m_streamID; }  // 스트림 ID 반환
  unsigned get_schd_id() const { return m_scheduler_id; }  // 스케줄러 ID 반환
  active_mask_t get_warp_active_mask() const { return m_warp_active_mask; }  // 활성 마스크 반환

 protected:  // 보호 영역
  unsigned m_uid;                       // 이 명령어의 고유 ID (생성 순서)
  unsigned long long m_streamID;        // 소속 스트림 ID
  bool m_empty;                         // 이 슬롯이 비어있는지 (true: 명령어 없음)
  bool m_cache_hit;                     // 캐시 히트(적중) 여부
  unsigned long long issue_cycle;       // 이 명령어가 발행된 사이클
  unsigned cycles;  // used for implementing initiation interval delay
                    // 발행 간격 지연을 구현하기 위한 카운터
  bool m_isatomic;                      // 원자적(atomic) 연산인지
  bool should_do_atomic;                // 원자적 연산을 실행해야 하는지
  bool m_is_printf;                     // printf 명령어인지
  unsigned m_warp_id;                   // 이 명령어가 속한 워프의 ID
  unsigned m_dynamic_warp_id;           // 동적 워프 ID (스케줄링에 사용)
  const core_config *m_config;          // 코어 설정 포인터

  active_mask_t m_warp_active_mask;  // dynamic active mask for timing model
                                     // (after predication)
                                     // 타이밍 모델용 동적 활성 마스크 (조건 판정 후)
  active_mask_t
      m_warp_issued_mask;  // active mask at issue (prior to predication test)
                           // -- for instruction counting
                           // 발행 시점의 활성 마스크 (조건 판정 전) -- 명령어 카운팅용

  /*
   * per_thread_info: 워프 내 각 스레드(스칼라 스레드)의 정보
   * 스레드마다 메모리 요청 주소와 콜백 정보를 저장합니다.
   */
  struct per_thread_info {
    per_thread_info() {
      for (unsigned i = 0; i < MAX_ACCESSES_PER_INSN_PER_THREAD; i++)
        memreqaddr[i] = 0;  // 모든 메모리 요청 주소를 0으로 초기화
    }
    dram_callback_t callback;  // 메모리 요청 완료 시 호출할 콜백
    new_addr_type
        memreqaddr[MAX_ACCESSES_PER_INSN_PER_THREAD];  // effective address,
                                                       // upto 8 different
                                                       // requests (to support
                                                       // 32B access in 8 chunks
                                                       // of 4B each)
                                                       // 실제 메모리 주소: 최대 8개
                                                       // (32바이트를 4바이트씩 8번에 나눠 접근)
  };
  bool m_per_scalar_thread_valid;                  // 스레드별 정보가 유효한지
  std::vector<per_thread_info> m_per_scalar_thread; // 각 스레드의 정보 배열
  bool m_mem_accesses_created;                     // 메모리 접근이 생성되었는지
  std::list<mem_access_t> m_accessq;               // 메모리 접근 요청 큐 (대기 중인 요청들)

  unsigned m_scheduler_id;  // the scheduler that issues this inst
                            // 이 명령어를 발행한 스케줄러의 ID

  // Jin: cdp support (CDP 지원)
 public:
  int m_is_cdp;  // CDP(CUDA Dynamic Parallelism) 관련 플래그
  // CDP: GPU 프로그램 안에서 또 다른 GPU 프로그램을 실행하는 기능

  // Ni: add boolean to indicate whether the instruction is ldgsts
  // Ni가 추가: ldgsts(비동기 로드) 명령어인지 나타내는 플래그
  bool m_is_ldgsts;     // ldgsts 명령어인지 (글로벌 → 공유 메모리 비동기 복사)
  bool m_is_ldgdepbar;  // ldgdepbar 명령어인지 (로드 의존성 배리어)
  bool m_is_depbar;     // depbar 명령어인지 (의존성 배리어)

  unsigned int m_depbar_group_no;  // 의존성 배리어 그룹 번호
};

/*
 * move_warp: 워프 명령어를 한 슬롯에서 다른 슬롯으로 이동
 * 파이프라인에서 명령어가 다음 단계로 넘어갈 때 사용합니다.
 */
void move_warp(warp_inst_t *&dst, warp_inst_t *&src);

/*
 * get_kernel_code_size: 커널 코드의 크기(바이트)를 반환
 */
size_t get_kernel_code_size(class function_info *entry);

/*
 * ==========================================================================
 * checkpoint 클래스
 * ==========================================================================
 * 시뮬레이션 상태를 저장하고 복원하는 체크포인트 기능
 * 오래 걸리는 시뮬레이션을 중간에 저장하고 나중에 이어서 할 수 있습니다.
 * 마치 게임의 세이브/로드 기능과 같습니다.
 */
class checkpoint {
 public:
  checkpoint();                                    // 생성자
  ~checkpoint() { printf("clasfsfss destructed\n"); }  // 소멸자: 삭제 시 메시지 출력 (디버깅용, 오타가 있음)

  void load_global_mem(class memory_space *temp_mem, char *f1name);           // 파일에서 글로벌 메모리 로드
  void store_global_mem(class memory_space *mem, char *fname, char *format);  // 글로벌 메모리를 파일에 저장
  unsigned radnom;  // 랜덤 값 (오타: random이 맞음)
};

/*
 * ==========================================================================
 * core_t 클래스 (추상 기반 클래스)
 * ==========================================================================
 * GPU 코어(SM)의 기본 기능을 정의하는 추상 클래스
 * "추상 클래스"란 직접 객체를 만들 수 없고, 자식 클래스가 상속받아 구현해야 하는 클래스입니다.
 *
 * 이 클래스는 기능 시뮬레이션과 성능 시뮬레이션 모두의 기반이 됩니다.
 * - 워프 실행
 * - SIMT 스택 관리
 * - 스레드 관리
 * - 배리어 리덕션
 */
/*
 * This abstract class used as a base for functional and performance and
 * simulation, it has basic functional simulation data structures and
 * procedures.
 */
class core_t {
 public:
  /*
   * 생성자: GPU, 커널 정보, 워프 크기, SM당 스레드 수를 받아 초기화
   * 초기화 리스트(:)를 사용하여 멤버 변수를 효율적으로 초기화
   */
  core_t(gpgpu_sim *gpu, kernel_info_t *kernel, unsigned warp_size,
         unsigned threads_per_shader)
      : m_gpu(gpu),            // GPU 시뮬레이터 포인터
        m_kernel(kernel),      // 실행할 커널 정보
        m_simt_stack(NULL),    // SIMT 스택 배열: 아직 없음
        m_thread(NULL),        // 스레드 배열: 아직 없음
        m_warp_size(warp_size) {  // 워프 크기 설정
    m_warp_count = threads_per_shader / m_warp_size;  // 워프 수 = 총 스레드 수 / 워프 크기
    // Handle the case where the number of threads is not a
    // multiple of the warp size
    // 스레드 수가 워프 크기의 배수가 아닌 경우 처리
    if (threads_per_shader % m_warp_size != 0) {
      m_warp_count += 1;  // 나머지가 있으면 워프 하나 추가 (부분적으로 찬 워프)
    }
    assert(m_warp_count * m_warp_size > 0);  // 총 스레드 수가 0보다 큰지 확인

    /*
     * calloc: 메모리를 할당하고 0으로 초기화하는 함수
     * 모든 스레드의 포인터를 저장할 배열을 할당
     */
    m_thread = (ptx_thread_info **)calloc(m_warp_count * m_warp_size,
                                          sizeof(ptx_thread_info *));
    initilizeSIMTStack(m_warp_count, m_warp_size);  // 모든 워프의 SIMT 스택 초기화

    /*
     * 리덕션 저장소 초기화
     * 각 CTA와 배리어에 대해 리덕션 결과를 저장하는 2차원 배열
     */
    for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) {
      for (unsigned j = 0; j < MAX_BARRIERS_PER_CTA; j++) {
        reduction_storage[i][j] = 0;  // 모든 리덕션 값을 0으로 초기화
      }
    }
  }
  virtual ~core_t() { free(m_thread); }  // 소멸자: 스레드 배열 메모리 해제

  /*
   * 순수 가상 함수들: 자식 클래스에서 반드시 구현해야 합니다.
   * = 0은 "구현이 없음, 자식이 해줘야 함"을 의미
   */
  virtual void warp_exit(unsigned warp_id) = 0;  // 워프가 실행을 종료할 때 호출
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const = 0;  // 워프가 배리어에서 대기 중인지
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) = 0;  // 실행 상태 확인 및 업데이트

  class gpgpu_sim *get_gpu() {   // GPU 시뮬레이터 포인터 반환
    return m_gpu;
  }

  /*
   * execute_warp_inst_t: 워프 명령어를 실행 (기능 시뮬레이션)
   * 실제로 각 스레드에서 명령어를 수행합니다.
   */
  void execute_warp_inst_t(warp_inst_t &inst, unsigned warpId = (unsigned)-1);
  bool ptx_thread_done(unsigned hw_thread_id) const;  // 특정 스레드가 실행을 완료했는지

  /*
   * SIMT 스택 관련 함수들
   */
  virtual void updateSIMTStack(unsigned warpId, warp_inst_t *inst);  // SIMT 스택 업데이트 (분기 처리)
  void initilizeSIMTStack(unsigned warp_count, unsigned warps_size); // SIMT 스택 초기화
  void deleteSIMTStack();                                            // SIMT 스택 삭제
  warp_inst_t getExecuteWarp(unsigned warpId);                       // 실행할 워프 명령어 가져오기
  void get_pdom_stack_top_info(unsigned warpId, unsigned *pc,
                               unsigned *rpc) const;                 // 스택 맨 위 정보 가져오기
  kernel_info_t *get_kernel_info() { return m_kernel; }              // 커널 정보 반환
  class ptx_thread_info **get_thread_info() {                        // 스레드 배열 반환
    return m_thread;
  }
  unsigned get_warp_size() const { return m_warp_size; }  // 워프 크기 반환

  /*
   * 리덕션 연산 함수들
   * 배리어에서 여러 스레드의 값을 하나로 합치는 연산
   * ctaid: CTA ID, barid: 배리어 ID, value: 이 스레드의 값
   */
  void and_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] &= value;  // AND 연산: 모든 값이 true여야 결과가 true
  }
  void or_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] |= value;   // OR 연산: 하나라도 true면 결과가 true
  }
  void popc_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] += value;   // POPC 연산: true인 값의 개수를 셈
  }
  unsigned get_reduction_value(unsigned ctaid, unsigned barid) {
    return reduction_storage[ctaid][barid];      // 리덕션 결과값 반환
  }

 protected:  // 보호 영역
  class gpgpu_sim *m_gpu;           // GPU 시뮬레이터 포인터
  kernel_info_t *m_kernel;          // 현재 실행 중인 커널 정보
  simt_stack **m_simt_stack;  // pdom based reconvergence context for each warp
                              // 각 워프의 SIMT 스택 (포스트 도미네이터 기반 재수렴)
  class ptx_thread_info **m_thread;  // 스레드 정보 배열 (각 스레드의 상태)
  unsigned m_warp_size;              // 워프 크기 (보통 32)
  unsigned m_warp_count;             // 이 코어의 총 워프 수
  unsigned reduction_storage[MAX_CTA_PER_SHADER][MAX_BARRIERS_PER_CTA];  // 리덕션 결과 저장 배열
  // [CTA ID][배리어 ID] → 리덕션 결과값
};

/*
 * ==========================================================================
 * register_set 클래스
 * ==========================================================================
 * 여러 명령어를 담을 수 있는 레지스터 세트
 * GPU 파이프라인의 각 단계 사이에 있는 "파이프라인 레지스터"를 나타냅니다.
 *
 * 파이프라인 레지스터란?
 *   파이프라인의 각 단계 사이에서 명령어를 임시로 저장하는 공간입니다.
 *   마치 공장의 컨베이어 벨트에서 물건을 임시로 놓는 칸과 같습니다.
 *   하나의 레지스터 세트에 여러 명령어를 담을 수 있어서,
 *   여러 워프의 명령어를 동시에 처리할 수 있습니다.
 */
// register that can hold multiple instructions.
class register_set {
 public:
  /*
   * 생성자: 레지스터 수와 이름을 받아 초기화
   * num: 이 세트에 포함되는 레지스터(슬롯) 수
   * name: 이 레지스터 세트의 이름 (디버깅용, 예: "ID_OC" = 디코드→실행 사이)
   */
  register_set(unsigned num, const char *name) {
    for (unsigned i = 0; i < num; i++) {
      regs.push_back(new warp_inst_t());  // 각 슬롯에 빈 warp_inst_t 객체 생성
    }
    m_name = name;  // 레지스터 세트 이름 저장
  }
  const char *get_name() { return m_name; }  // 이름 반환

  /*
   * has_free: 빈 슬롯(비어있는 레지스터)이 있는지 확인
   * 새 명령어를 넣을 공간이 있는지 확인할 때 사용
   */
  bool has_free() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->empty()) {    // 비어있는 슬롯을 찾으면
        return true;              // 빈 슬롯이 있음
      }
    }
    return false;                  // 빈 슬롯이 없음 (모두 사용 중)
  }
  /*
   * has_free (서브코어 모델용): 특정 레지스터 ID의 슬롯이 비어있는지 확인
   * 서브코어 모델에서는 각 스케줄러가 지정된 레지스터만 사용합니다.
   */
  bool has_free(bool sub_core_model, unsigned reg_id) {
    // in subcore model, each sched has a one specific reg to use (based on
    // sched id)
    // 서브코어 모델에서 각 스케줄러는 특정 레지스터 하나만 사용 (스케줄러 ID 기반)
    if (!sub_core_model) return has_free();  // 서브코어 모델이 아니면 일반 방식

    assert(reg_id < regs.size());  // 레지스터 ID가 유효한지 확인
    return regs[reg_id]->empty();  // 해당 레지스터가 비어있는지
  }

  /*
   * has_ready: 실행 준비된 명령어가 있는지 확인
   * 비어있지 않은(명령어가 있는) 슬롯이 있으면 준비됨
   */
  bool has_ready() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {  // 비어있지 않은 슬롯을 찾으면
        return true;                // 준비된 명령어가 있음
      }
    }
    return false;                    // 준비된 명령어가 없음
  }
  bool has_ready(bool sub_core_model, unsigned reg_id) {  // 서브코어 모델용
    if (!sub_core_model) return has_ready();
    assert(reg_id < regs.size());
    return (not regs[reg_id]->empty());
  }

  /*
   * get_ready_reg_id: 준비된 명령어 중 가장 오래된 것의 레지스터 ID를 반환
   * 가장 오래된 명령어를 우선 처리하기 위해 UID가 가장 작은 것을 찾습니다.
   * (UID가 작을수록 먼저 생성된 명령어)
   */
  unsigned get_ready_reg_id() {
    // for sub core model we need to figure which reg_id has the ready warp
    // this function should only be called if has_ready() was true
    assert(has_ready());             // 준비된 명령어가 있어야 호출 가능
    warp_inst_t **ready;
    ready = NULL;
    unsigned reg_id = 0;
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
          // ready is oldest: 현재 ready가 더 오래됨 → 유지
        } else {
          ready = &regs[i];   // 더 오래된(또는 첫 번째) 준비된 명령어를 선택
          reg_id = i;
        }
      }
    }
    return reg_id;  // 가장 오래된 준비된 명령어의 레지스터 ID 반환
  }
  unsigned get_schd_id(unsigned reg_id) {  // 특정 레지스터의 스케줄러 ID 반환
    assert(not regs[reg_id]->empty());
    return regs[reg_id]->get_schd_id();
  }

  /*
   * move_in: 명령어를 이 레지스터 세트의 빈 슬롯에 넣음
   * 파이프라인의 이전 단계에서 다음 단계로 명령어를 이동시킵니다.
   */
  void move_in(warp_inst_t *&src) {
    warp_inst_t **free = get_free();   // 빈 슬롯 찾기
    move_warp(*free, src);             // 명령어 이동 (src → free)
  }
  // void copy_in( warp_inst_t* src ){  // 주석 처리됨: 복사 방식
  //   src->copy_contents_to(*get_free());
  //}
  /*
   * move_in (서브코어 모델용): 특정 레지스터에 명령어를 넣음
   */
  void move_in(bool sub_core_model, unsigned reg_id, warp_inst_t *&src) {
    warp_inst_t **free;
    if (!sub_core_model) {
      free = get_free();                           // 일반 모델: 아무 빈 슬롯
    } else {
      assert(reg_id < regs.size());
      free = get_free(sub_core_model, reg_id);     // 서브코어: 지정된 슬롯
    }
    move_warp(*free, src);
  }

  /*
   * move_out_to: 준비된 명령어를 이 레지스터 세트에서 꺼냄
   * 파이프라인의 다음 단계로 보내기 위해 명령어를 추출합니다.
   */
  void move_out_to(warp_inst_t *&dest) {
    warp_inst_t **ready = get_ready();   // 준비된 명령어 찾기
    move_warp(dest, *ready);             // 명령어 이동 (ready → dest)
  }
  void move_out_to(bool sub_core_model, unsigned reg_id, warp_inst_t *&dest) {
    if (!sub_core_model) {
      return move_out_to(dest);                    // 일반 모델
    }
    warp_inst_t **ready = get_ready(sub_core_model, reg_id);  // 서브코어: 지정된 슬롯
    assert(ready != NULL);                         // 준비된 명령어가 있어야 함
    move_warp(dest, *ready);
  }

  /*
   * get_ready: 준비된 명령어 중 가장 오래된 것을 찾아 포인터를 반환
   * 가장 오래된 명령어(UID가 가장 작은)를 우선 처리
   */
  warp_inst_t **get_ready() {
    warp_inst_t **ready;
    ready = NULL;
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
          // ready is oldest: 현재 선택된 것이 더 오래됨 → 유지
        } else {
          ready = &regs[i];   // 새로 찾은 것이 더 오래되었거나 첫 번째
        }
      }
    }
    return ready;  // 가장 오래된 준비된 명령어의 포인터 반환 (없으면 NULL)
  }
  warp_inst_t **get_ready(bool sub_core_model, unsigned reg_id) {  // 서브코어 모델용
    if (!sub_core_model) return get_ready();
    warp_inst_t **ready;
    ready = NULL;
    assert(reg_id < regs.size());
    if (not regs[reg_id]->empty()) ready = &regs[reg_id];
    return ready;
  }

  /*
   * print: 레지스터 세트의 내용을 파일에 출력 (디버깅용)
   * 각 슬롯에 있는 명령어 정보를 출력합니다.
   */
  void print(FILE *fp) const {
    fprintf(fp, "%s : @%p\n", m_name, this);  // 세트 이름과 메모리 주소 출력
    for (unsigned i = 0; i < regs.size(); i++) {
      fprintf(fp, "     ");
      regs[i]->print(fp);    // 각 슬롯의 명령어 정보 출력
      fprintf(fp, "\n");
    }
  }

  /*
   * get_free: 빈 슬롯을 찾아 포인터를 반환
   * 명령어를 넣을 수 있는 빈 공간을 찾습니다.
   */
  warp_inst_t **get_free() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->empty()) {    // 비어있는 슬롯 발견
        return &regs[i];          // 그 슬롯의 포인터 반환
      }
    }
    assert(0 && "No free registers found");  // 빈 슬롯이 없으면 프로그램 중단 (오류)
    return NULL;
  }

  /*
   * get_free (서브코어 모델용): 특정 레지스터 ID의 슬롯이 비어있으면 반환
   */
  warp_inst_t **get_free(bool sub_core_model, unsigned reg_id) {
    // in subcore model, each sched has a one specific reg to use (based on
    // sched id)
    if (!sub_core_model) return get_free();

    assert(reg_id < regs.size());
    if (regs[reg_id]->empty()) {
      return &regs[reg_id];
    }
    assert(0 && "No free register found");  // 해당 슬롯이 비어있지 않으면 오류
    return NULL;
  }

  unsigned get_size() { return regs.size(); }  // 레지스터 세트의 슬롯 수 반환

 private:
  std::vector<warp_inst_t *> regs;  // 레지스터(슬롯) 배열: 각 원소는 하나의 명령어를 담을 수 있음
  const char *m_name;               // 이 레지스터 세트의 이름 (예: "ID_OC", "OC_EX")
};

#endif  // #ifdef __cplusplus  // C++ 전용 코드 영역 끝

#endif  // #ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED  // 인클루드 가드 끝
