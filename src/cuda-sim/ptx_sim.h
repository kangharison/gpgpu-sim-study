// Copyright (c) 2009-2011, The University of British Columbia
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
 * [한국어 설명] PTX 기능 시뮬레이션 핵심 자료구조 선언 (ptx_sim.h)
 *
 * === 파일의 역할 ===
 * GPGPU-Sim의 PTX 기능 시뮬레이션(functional simulation)에서 사용되는 핵심 타입과
 * 클래스들을 선언하는 헤더 파일이다. 이 파일에서 선언된 자료구조들은 PTX 코드를
 * 스레드 단위로 실행하면서 각 스레드의 상태(레지스터 파일, 프로그램 카운터, 콜스택,
 * 메모리 공간)를 관리한다. 기능 시뮬레이션은 타이밍 시뮬레이션(gpgpu-sim/)과 분리되어
 * 명령어 의미(semantics)만 정확히 실행하는 역할을 담당한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * GPGPU-Sim 기능 시뮬레이션 레이어(cuda-sim/)의 중심 헤더.
 * 실행 흐름: cuda application → libcuda → gpgpusim_entrypoint →
 *            functionalCoreSim → ptx_thread_info::ptx_exec_inst → [각 명령어 시맨틱]
 * ptx_thread_info는 각 GPU 스레드의 실행 상태를 캡슐화하는 최상위 클래스이다.
 * 타이밍 시뮬레이션(shader.cc)은 ptx_exec_inst를 호출하여 실제 명령어 결과를 얻는다.
 *
 * === 타 모듈과의 연결 ===
 * 의존 모듈:
 *   - abstract_hardware_model.h: kernel_info_t, warp_inst_t, core_t, gpgpu_t 등 상위 추상
 *   - cuda-sim/memory.h: memory_space (로컬/공유/전역 메모리 시뮬레이션)
 *   - cuda-sim/opcodes.h: PTX 명령어 opcode 열거형
 *   - half.h: fp16 타입 지원
 * 의존받는 모듈:
 *   - cuda-sim/instructions.cc: 각 PTX 명령어 시맨틱 구현 (ptx_thread_info 사용)
 *   - cuda-sim/ptx_sim.cc: ptx_thread_info, ptx_cta_info, ptx_warp_info 구현
 *   - gpgpu-sim/shader.cc: ptx_exec_inst 호출하여 타이밍 모델과 기능 모델 연결
 * 핵심 데이터 흐름:
 *   kernel launch → ptx_thread_info 생성 → ptx_exec_inst (명령어 실행) →
 *   레지스터/메모리 업데이트 → 타이밍 모델에 결과 반영
 *
 * === 주요 함수/구조체 요약 ===
 * ptx_reg_t       : PTX 레지스터 값을 모든 지원 타입으로 표현하는 union
 * ptx_cta_info    : CTA(Cooperative Thread Array, 스레드 블록)의 스레드 집합 및 배리어 관리
 * ptx_warp_info   : warp 내 완료된 스레드 수 추적 (barrier 처리에 활용)
 * stack_entry     : 함수 호출 스택 엔트리 (call/return 시 PC 및 심볼 테이블 복원)
 * ptx_version     : .version/.target 지시어에서 파싱된 PTX 버전 및 SM 버전 정보
 * ptx_thread_info : 개별 GPU 스레드의 전체 실행 상태 (레지스터 파일, PC, 콜스택, 메모리)
 * generic_to_*    : 제네릭 주소 ↔ 특정 메모리 공간 주소 변환 함수들
 */

#ifndef ptx_sim_h_INCLUDED
#define ptx_sim_h_INCLUDED

#include <stdlib.h>      // [한국어] malloc/free/abort 등 C 표준 유틸리티 — 메모리 할당에 사용
#include "../abstract_hardware_model.h"  // [한국어] kernel_info_t, warp_inst_t, core_t, gpgpu_t 등 상위 추상 모델 정의
#include "../tr1_hash_map.h"  // [한국어] tr1_hash_map<K,V> — 레지스터 파일 구현에 사용하는 해시맵 래퍼
#include "half.h"  // [한국어] half_float::half — fp16(반정밀도 부동소수점) 타입 제공

#include <assert.h>  // [한국어] assert 매크로 — 불변조건 검사
#include "opcodes.h"  // [한국어] PTX 명령어 opcode 열거형 및 정의

#include <list>    // [한국어] std::list — 레지스터 스택, 콜스택에 사용
#include <map>     // [한국어] std::map — 심볼→값 매핑에 사용
#include <set>     // [한국어] std::set — 스레드 집합(ptx_cta_info) 관리에 사용
#include <string>  // [한국어] std::string — 파일명/심볼명 저장에 사용

#include "memory.h"  // [한국어] memory_space 클래스 — 로컬/공유/전역 메모리 시뮬레이션

#define GCC_VERSION \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
// [한국어] GCC 버전 계산 매크로: 주버전*10000 + 부버전*100 + 패치버전
// fp16 지원 여부 분기(GCC 4.7.0 이상)에 사용됨

/*
 * [한국어]
 * struct param_t - 커널 파라미터 단일 항목 서술자
 *
 * CUDA 커널 론칭 시 호스트에서 GPU로 전달되는 파라미터 하나를 나타낸다.
 * cuLaunchKernel/cuParamSet* 등의 런타임 API가 파라미터를 파싱하여 이 구조체로 기록한다.
 * ptx_thread_info가 param 메모리 공간에서 커널 인수를 읽을 때 참조한다.
 */
struct param_t {
  const void *pdata;
  /* [한국어] 파라미터 값에 대한 포인터.
   * 설정자: cuParamSetv/cuLaunchKernel을 통해 호스트 측에서 설정.
   * 읽는 자: 파라미터를 메모리에 복사하는 초기화 코드.
   * 값 범위: 유효한 호스트 메모리 포인터 (NULL 불가 — 파라미터 크기>0이면).
   * 동기화: 커널 론칭 전에 설정되고 이후 읽기 전용으로 사용됨. */

  int type;
  /* [한국어] 파라미터 타입 코드 (PTX 타입 열거값; s32/u32/f32 등).
   * 설정자: 커널 서명 파싱 시 설정.
   * 읽는 자: 파라미터 복사 시 크기/정렬 결정에 사용.
   * 값 범위: PTX 타입 enum 값.
   * 동기화: 설정 후 읽기 전용. */

  size_t size;
  /* [한국어] 파라미터 값의 바이트 크기.
   * 설정자: cuParamSet*에서 파라미터 크기 기록.
   * 읽는 자: param 메모리 공간에 복사할 때 바이트 수 결정.
   * 값 범위: 1~16 바이트 (스칼라 타입 기준).
   * 동기화: 설정 후 읽기 전용. */

  size_t offset;
  /* [한국어] 커널 파라미터 블록 내 이 파라미터의 바이트 오프셋.
   * 설정자: cuParamSetv 시 누적 오프셋으로 설정.
   * 읽는 자: param 메모리 공간에서 이 파라미터가 저장된 위치 계산.
   * 값 범위: [0, 총 파라미터 블록 크기).
   * 동기화: 설정 후 읽기 전용. */
};

#include <stack>  // [한국어] std::stack — breakaddr 스택(ptx_thread_info::m_breakaddrs)에 사용

#include "memory.h"  // [한국어] memory_space 이중 include (guard 보호됨); .h 순서 안전성 확보용

using half_float::half;  // [한국어] half_float::half를 half로 사용 가능하게 함 (fp16 레지스터 타입에 사용)

/*
 * [한국어]
 * union ptx_reg_t - PTX 레지스터 값의 다형 union
 *
 * PTX 레지스터는 명령어에 따라 부호있는/없는 정수, 부동소수점, 비트 패턴, 프레디케이트,
 * 128비트 등 다양한 타입으로 해석된다. 이 union은 동일한 8바이트 메모리 영역을
 * 모든 PTX 기본 타입으로 접근할 수 있게 한다.
 * 레지스터 값을 읽고 쓸 때 명령어의 타입 한정자(type qualifier)에 따라
 * 해당 필드(s32, u64, f32 등)를 사용한다.
 * 동기화: 단일 스레드에서만 접근 (ptx_thread_info는 스레드 단위로 독립 존재).
 */
union ptx_reg_t {
  /*
   * [한국어]
   * ptx_reg_t() - 기본 생성자; 모든 필드를 0으로 초기화
   *
   * union의 모든 오버래핑 필드를 명시적으로 0으로 초기화한다.
   * C++ union은 기본적으로 0 초기화를 보장하지 않으므로
   * 각 필드를 개별적으로 0으로 설정한다.
   */
  ptx_reg_t() {
    bits.ms = 0;   // [한국어] 상위 32비트 0으로 초기화
    bits.ls = 0;   // [한국어] 하위 32비트 0으로 초기화
    u128.low = 0;      // [한국어] 128비트 값의 low 워드 초기화
    u128.lowest = 0;   // [한국어] 128비트 값의 lowest 워드 초기화
    u128.highest = 0;  // [한국어] 128비트 값의 highest 워드 초기화
    u128.high = 0;     // [한국어] 128비트 값의 high 워드 초기화
    s8 = 0;    // [한국어] 8비트 부호 정수 초기화
    s16 = 0;   // [한국어] 16비트 부호 정수 초기화
    s32 = 0;   // [한국어] 32비트 부호 정수 초기화
    s64 = 0;   // [한국어] 64비트 부호 정수 초기화
    u8 = 0;    // [한국어] 8비트 부호없는 정수 초기화
    u16 = 0;   // [한국어] 16비트 부호없는 정수 초기화
    u64 = 0;   // [한국어] 64비트 부호없는 정수 초기화
    f16 = 0;   // [한국어] 16비트 부동소수점 초기화
    f32 = 0;   // [한국어] 32비트 부동소수점 초기화
    f64 = 0;   // [한국어] 64비트 부동소수점 초기화
    pred = 0;  // [한국어] 4비트 프레디케이트 레지스터 초기화
  }
  /*
   * [한국어]
   * ptx_reg_t(unsigned x) - 부호없는 32비트 정수 값으로 초기화하는 생성자
   *
   * @param x: 초기화할 u32 값
   *
   * 모든 필드를 0으로 초기화 후 u32에 x를 설정한다.
   * 정수 상수나 주소 값으로 레지스터를 초기화할 때 사용.
   */
  ptx_reg_t(unsigned x) {
    bits.ms = 0;   // [한국어] 상위 32비트 0으로 초기화
    bits.ls = 0;   // [한국어] 하위 32비트 0으로 초기화
    u128.low = 0;      // [한국어] 128비트 low 워드 초기화
    u128.lowest = 0;   // [한국어] 128비트 lowest 워드 초기화
    u128.highest = 0;  // [한국어] 128비트 highest 워드 초기화
    u128.high = 0;     // [한국어] 128비트 high 워드 초기화
    s8 = 0;    // [한국어] s8 초기화
    s16 = 0;   // [한국어] s16 초기화
    s32 = 0;   // [한국어] s32 초기화
    s64 = 0;   // [한국어] s64 초기화
    u8 = 0;    // [한국어] u8 초기화
    u16 = 0;   // [한국어] u16 초기화
    u64 = 0;   // [한국어] u64 초기화
    f16 = 0;   // [한국어] f16 초기화
    f32 = 0;   // [한국어] f32 초기화
    f64 = 0;   // [한국어] f64 초기화
    pred = 0;  // [한국어] pred 초기화
    u32 = x;   // [한국어] u32에 지정된 값 설정
  }
  operator unsigned int() { return u32; }          // [한국어] unsigned int로 묵시적 변환 — u32 반환
  operator unsigned short() { return u16; }        // [한국어] unsigned short로 묵시적 변환 — u16 반환
  operator unsigned char() { return u8; }          // [한국어] unsigned char로 묵시적 변환 — u8 반환
  operator unsigned long long() { return u64; }    // [한국어] unsigned long long으로 묵시적 변환 — u64 반환

  /*
   * [한국어]
   * mask_and - 상위/하위 32비트에 AND 마스크 적용
   *
   * @param ms: 상위 32비트(bits.ms)에 AND할 마스크
   * @param ls: 하위 32비트(bits.ls)에 AND할 마스크
   *
   * bfe/bfi(비트 필드 추출/삽입) 등 비트 연산 명령어에서 특정 비트 범위를 추출할 때 사용.
   */
  void mask_and(unsigned ms, unsigned ls) {
    bits.ms &= ms;  // [한국어] 상위 32비트에 ms 마스크 AND 적용
    bits.ls &= ls;  // [한국어] 하위 32비트에 ls 마스크 AND 적용
  }

  /*
   * [한국어]
   * mask_or - 상위/하위 32비트에 OR 마스크 적용
   *
   * @param ms: 상위 32비트(bits.ms)에 OR할 마스크
   * @param ls: 하위 32비트(bits.ls)에 OR할 마스크
   *
   * 비트 필드 삽입(bfi) 등에서 특정 비트 위치에 값을 쓸 때 사용.
   */
  void mask_or(unsigned ms, unsigned ls) {
    bits.ms |= ms;  // [한국어] 상위 32비트에 ms 마스크 OR 적용
    bits.ls |= ls;  // [한국어] 하위 32비트에 ls 마스크 OR 적용
  }

  /*
   * [한국어]
   * get_bit - 지정된 비트 위치의 값을 반환 (0 또는 1)
   *
   * @param bit: 조회할 비트 위치 (0=최하위 비트, 63=최상위 비트)
   * @return: 해당 비트 값 (0 또는 1)
   *
   * 64비트 레지스터 값에서 특정 비트를 추출한다.
   * 비트 위치가 32 미만이면 bits.ls(하위 워드), 32 이상이면 bits.ms(상위 워드)에서 추출.
   */
  int get_bit(unsigned bit) {
    if (bit < 32)  // [한국어] 비트 위치가 하위 32비트 범위이면
      return (bits.ls >> bit) & 1;  // [한국어] 하위 워드에서 해당 비트 추출
    else  // [한국어] 상위 32비트 범위이면
      return (bits.ms >> (bit - 32)) & 1;  // [한국어] 상위 워드에서 비트 추출 (32 오프셋 적용)
  }

  // [한국어] === union 필드들 ===
  // 아래 필드들은 모두 동일한 8바이트 메모리 영역을 다른 타입으로 해석한다.
  // 명령어 타입 한정자(e.g. .s32, .f64, .pred)에 따라 적절한 필드를 선택하여 읽고 쓴다.

  signed char s8;
  /* [한국어] 8비트 부호있는 정수 해석 (PTX .s8 타입).
   * 설정자: cvt.s8 등 명령어의 결과 저장.
   * 읽는 자: cvt, add 등에서 .s8 타입으로 오퍼랜드를 읽을 때.
   * 값 범위: -128 ~ 127.
   * 동기화: 단일 스레드 접근. */

  signed short s16;
  /* [한국어] 16비트 부호있는 정수 해석 (PTX .s16 타입).
   * 설정자: cvt.s16, mul.s16 등 명령어 결과.
   * 읽는 자: .s16 타입 오퍼랜드 접근 시.
   * 값 범위: -32768 ~ 32767.
   * 동기화: 단일 스레드 접근. */

  signed int s32;
  /* [한국어] 32비트 부호있는 정수 해석 (PTX .s32 타입, 가장 흔한 정수 타입).
   * 설정자: add.s32, mul.s32, ld.global.s32 등 결과.
   * 읽는 자: 산술/논리 명령어에서 s32 오퍼랜드 접근 시.
   * 값 범위: -2^31 ~ 2^31-1.
   * 동기화: 단일 스레드 접근. */

  signed long long s64;
  /* [한국어] 64비트 부호있는 정수 해석 (PTX .s64 타입).
   * 설정자: mul.wide.s32 (32x32→64비트), ld.global.s64 등.
   * 읽는 자: 64비트 산술/메모리 명령어에서 .s64 오퍼랜드 접근 시.
   * 값 범위: -2^63 ~ 2^63-1.
   * 동기화: 단일 스레드 접근. */

  unsigned char u8;
  /* [한국어] 8비트 부호없는 정수 해석 (PTX .u8 타입).
   * 설정자: cvt.u8, ld.global.u8 등.
   * 읽는 자: .u8 타입 오퍼랜드 접근 시.
   * 값 범위: 0 ~ 255.
   * 동기화: 단일 스레드 접근. */

  unsigned short u16;
  /* [한국어] 16비트 부호없는 정수 해석 (PTX .u16 타입).
   * 설정자: cvt.u16, mul.u16 등.
   * 읽는 자: .u16 타입 오퍼랜드 접근 시.
   * 값 범위: 0 ~ 65535.
   * 동기화: 단일 스레드 접근. */

  unsigned int u32;
  /* [한국어] 32비트 부호없는 정수 해석 (PTX .u32 타입, 주소 계산에도 사용).
   * 설정자: add.u32, cvt.u32, mov.u32 등; ptx_reg_t(unsigned x) 생성자.
   * 읽는 자: u32 타입 오퍼랜드 및 unsigned int 캐스트 연산자.
   * 값 범위: 0 ~ 2^32-1.
   * 동기화: 단일 스레드 접근. */

  unsigned long long u64;
  /* [한국어] 64비트 부호없는 정수 해석 (PTX .u64 타입, 64비트 주소/데이터).
   * 설정자: mul.wide.u32 (32x32→64비트), ld.global.u64 등.
   * 읽는 자: u64 타입 오퍼랜드 및 unsigned long long 캐스트 연산자.
   * 값 범위: 0 ~ 2^64-1.
   * 동기화: 단일 스레드 접근. */

// gcc 4.7.0
#if GCC_VERSION >= 40700
  half f16;
  /* [한국어] 16비트 반정밀도 부동소수점 (PTX .f16 타입).
   * GCC 4.7.0 이상에서만 half_float::half 타입 사용 (IEEE 754 fp16).
   * 설정자: cvt.f16, ld.global.f16 등.
   * 읽는 자: .f16 타입 오퍼랜드 접근 시.
   * 값 범위: IEEE 754 fp16 범위 (~±65504).
   * 동기화: 단일 스레드 접근. */
#else
  float f16;
  /* [한국어] 구 GCC(4.7.0 미만)에서 fp16을 float로 에뮬레이션.
   * 완전한 fp16 의미론 미보장 — 정밀도 손실 가능.
   * 동기화: 단일 스레드 접근. */
#endif

  float f32;
  /* [한국어] 32비트 단정밀도 부동소수점 (PTX .f32 타입, GPU 연산의 주 타입).
   * 설정자: mul.f32, fma.f32, ld.global.f32 등.
   * 읽는 자: .f32 타입 오퍼랜드 접근 시.
   * 값 범위: IEEE 754 fp32 범위.
   * 동기화: 단일 스레드 접근. */

  double f64;
  /* [한국어] 64비트 배정밀도 부동소수점 (PTX .f64 타입).
   * 설정자: mul.f64, fma.f64, ld.global.f64 등.
   * 읽는 자: .f64 타입 오퍼랜드 접근 시.
   * 값 범위: IEEE 754 fp64 범위.
   * 동기화: 단일 스레드 접근. */

  struct {
    unsigned ls;
    /* [한국어] 64비트 비트 패턴의 하위 32비트 (PTX .b64에서 비트 조작 시 사용).
     * mask_and/mask_or/get_bit에서 하위 워드로 접근.
     * 값 범위: 0 ~ 2^32-1. */
    unsigned ms;
    /* [한국어] 64비트 비트 패턴의 상위 32비트.
     * mask_and/mask_or/get_bit에서 상위 워드로 접근.
     * 값 범위: 0 ~ 2^32-1. */
  } bits;
  /* [한국어] 64비트 값을 두 개의 32비트 워드로 분리하여 비트 조작하는 구조체.
   * 설정자: mask_and, mask_or에서 직접 조작.
   * 읽는 자: get_bit에서 비트 위치에 따라 ls/ms 선택.
   * 동기화: 단일 스레드 접근. */

  struct {
    unsigned int lowest;
    /* [한국어] 128비트 값의 최하위 32비트 워드 (비트 [31:0]).
     * WMMA 명령어(행렬 곱셈 누산)에서 128비트 누산기 레지스터 접근에 사용.
     * 값 범위: 0 ~ 2^32-1. */
    unsigned int low;
    /* [한국어] 128비트 값의 두 번째 32비트 워드 (비트 [63:32]).
     * 값 범위: 0 ~ 2^32-1. */
    unsigned int high;
    /* [한국어] 128비트 값의 세 번째 32비트 워드 (비트 [95:64]).
     * 값 범위: 0 ~ 2^32-1. */
    unsigned int highest;
    /* [한국어] 128비트 값의 최상위 32비트 워드 (비트 [127:96]).
     * 값 범위: 0 ~ 2^32-1. */
  } u128;
  /* [한국어] 128비트 레지스터 값 (PTX .b128 타입, WMMA 명령어 누산기용).
   * WMMA(wmma.mma.sync)에서 출력 행렬 타일이 128비트 레지스터로 저장됨.
   * 설정자: wmma 명령어 실행 후 결과 저장.
   * 읽는 자: wmma 명령어에서 누산기 입력 읽기.
   * 동기화: 단일 스레드 접근. */

  unsigned pred : 4;
  /* [한국어] 4비트 프레디케이트 레지스터 값 (PTX .pred 타입).
   * 조건 분기(@p bra, @!p bra) 및 조건부 실행의 조건 값을 저장.
   * setp 명령어의 결과 (참=1, 거짓=0)가 여기 저장됨.
   * 값 범위: 0 (거짓) 또는 1 (참); 4비트이므로 0~15 가능하나 0/1만 사용.
   * 동기화: 단일 스레드 접근. */
};

class ptx_instruction;
class operand_info;
class symbol_table;
class function_info;
class ptx_thread_info;

class ptx_cta_info {
 public:
  ptx_cta_info(unsigned sm_idx, gpgpu_context *ctx);
  void add_thread(ptx_thread_info *thd);
  unsigned num_threads() const;
  void check_cta_thread_status_and_reset();
  void register_thread_exit(ptx_thread_info *thd);
  void register_deleted_thread(ptx_thread_info *thd);
  unsigned get_sm_idx() const;
  unsigned get_bar_threads() const;
  void inc_bar_threads();
  void reset_bar_threads();

 private:
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  unsigned m_bar_threads;
  unsigned long long m_uid;
  unsigned m_sm_idx;
  std::set<ptx_thread_info *> m_threads_in_cta;
  std::set<ptx_thread_info *> m_threads_that_have_exited;
  std::set<ptx_thread_info *> m_dangling_pointers;
};

class ptx_warp_info {
 public:
  ptx_warp_info();  // add get_core or something, or threads?
  unsigned get_done_threads() const;
  void inc_done_threads();
  void reset_done_threads();

 private:
  unsigned m_done_threads;
};

class symbol;

struct stack_entry {
  stack_entry() {
    m_symbol_table = NULL;
    m_func_info = NULL;
    m_PC = 0;
    m_RPC = -1;
    m_return_var_src = NULL;
    m_return_var_dst = NULL;
    m_call_uid = 0;
    m_valid = false;
  }
  stack_entry(symbol_table *s, function_info *f, unsigned pc, unsigned rpc,
              const symbol *return_var_src, const symbol *return_var_dst,
              unsigned call_uid) {
    m_symbol_table = s;
    m_func_info = f;
    m_PC = pc;
    m_RPC = rpc;
    m_return_var_src = return_var_src;
    m_return_var_dst = return_var_dst;
    m_call_uid = call_uid;
    m_valid = true;
  }

  bool m_valid;
  symbol_table *m_symbol_table;
  function_info *m_func_info;
  unsigned m_PC;
  unsigned m_RPC;
  const symbol *m_return_var_src;
  const symbol *m_return_var_dst;
  unsigned m_call_uid;
};

class ptx_version {
 public:
  ptx_version() {
    m_valid = false;
    m_ptx_version = 0;
    m_ptx_extensions = 0;
    m_sm_version_valid = false;
    m_texmode_unified = true;
    m_map_f64_to_f32 = true;
  }
  ptx_version(float ver, unsigned extensions) {
    m_valid = true;
    m_ptx_version = ver;
    m_ptx_extensions = extensions;
    m_sm_version_valid = false;
    m_texmode_unified = true;
  }
  void set_target(const char *sm_ver, const char *ext, const char *ext2) {
    assert(m_valid);
    m_sm_version_str = sm_ver;
    check_target_extension(ext);
    check_target_extension(ext2);
    sscanf(sm_ver, "%u", &m_sm_version);
    m_sm_version_valid = true;
  }
  float ver() const {
    assert(m_valid);
    return m_ptx_version;
  }
  unsigned target() const {
    assert(m_valid && m_sm_version_valid);
    return m_sm_version;
  }
  unsigned extensions() const {
    assert(m_valid);
    return m_ptx_extensions;
  }

 private:
  void check_target_extension(const char *ext) {
    if (ext) {
      if (!strcmp(ext, "texmode_independent"))
        m_texmode_unified = false;
      else if (!strcmp(ext, "texmode_unified"))
        m_texmode_unified = true;
      else if (!strcmp(ext, "map_f64_to_f32"))
        m_map_f64_to_f32 = true;
      else
        abort();
    }
  }

  bool m_valid;
  float m_ptx_version;
  unsigned m_sm_version_valid;
  std::string m_sm_version_str;
  bool m_texmode_unified;
  bool m_map_f64_to_f32;
  unsigned m_sm_version;
  unsigned m_ptx_extensions;
};

class ptx_thread_info {
 public:
  ~ptx_thread_info();
  ptx_thread_info(kernel_info_t &kernel);

  void init(gpgpu_t *gpu, core_t *core, unsigned sid, unsigned cta_id,
            unsigned wid, unsigned tid, bool fsim) {
    m_gpu = gpu;
    m_core = core;
    m_hw_sid = sid;
    m_hw_ctaid = cta_id;
    m_hw_wid = wid;
    m_hw_tid = tid;
    m_functionalSimulationMode = fsim;
  }

  void ptx_fetch_inst(inst_t &inst) const;
  void ptx_exec_inst(warp_inst_t &inst, unsigned lane_id);

  const ptx_version &get_ptx_version() const;
  void set_reg(const symbol *reg, const ptx_reg_t &value);
  void print_reg_thread(char *fname);
  void resume_reg_thread(char *fname, symbol_table *symtab);
  ptx_reg_t get_reg(const symbol *reg);
  ptx_reg_t get_operand_value(const operand_info &op, operand_info dstInfo,
                              unsigned opType, ptx_thread_info *thread,
                              int derefFlag);
  void set_operand_value(const operand_info &dst, const ptx_reg_t &data,
                         unsigned type, ptx_thread_info *thread,
                         const ptx_instruction *pI);
  void set_operand_value(const operand_info &dst, const ptx_reg_t &data,
                         unsigned type, ptx_thread_info *thread,
                         const ptx_instruction *pI, int overflow, int carry);
  void get_vector_operand_values(const operand_info &op, ptx_reg_t *ptx_regs,
                                 unsigned num_elements);
  void set_vector_operand_values(const operand_info &dst,
                                 const ptx_reg_t &data1, const ptx_reg_t &data2,
                                 const ptx_reg_t &data3,
                                 const ptx_reg_t &data4);
  void set_wmma_vector_operand_values(
      const operand_info &dst, const ptx_reg_t &data1, const ptx_reg_t &data2,
      const ptx_reg_t &data3, const ptx_reg_t &data4, const ptx_reg_t &data5,
      const ptx_reg_t &data6, const ptx_reg_t &data7, const ptx_reg_t &data8);

  function_info *func_info() { return m_func_info; }
  void print_insn(unsigned pc, FILE *fp) const;
  void set_info(function_info *func);
  unsigned get_uid() const { return m_uid; }

  dim3 get_ctaid() const { return m_ctaid; }
  dim3 get_tid() const { return m_tid; }
  dim3 get_ntid() const { return m_ntid; }
  class gpgpu_sim *get_gpu() {
    return (gpgpu_sim *)m_gpu;
  }
  unsigned get_hw_tid() const { return m_hw_tid; }
  unsigned get_hw_ctaid() const { return m_hw_ctaid; }
  unsigned get_hw_wid() const { return m_hw_wid; }
  unsigned get_hw_sid() const { return m_hw_sid; }
  core_t *get_core() { return m_core; }

  unsigned get_icount() const { return m_icount; }
  void set_valid() { m_valid = true; }
  addr_t last_eaddr() const { return m_last_effective_address; }
  memory_space_t last_space() const { return m_last_memory_space; }
  dram_callback_t last_callback() const { return m_last_dram_callback; }
  unsigned long long get_cta_uid() { return m_cta_info->get_sm_idx(); }

  void set_single_thread_single_block() {
    m_ntid.x = 1;
    m_ntid.y = 1;
    m_ntid.z = 1;
    m_ctaid.x = 0;
    m_ctaid.y = 0;
    m_ctaid.z = 0;
    m_tid.x = 0;
    m_tid.y = 0;
    m_tid.z = 0;
    m_nctaid.x = 1;
    m_nctaid.y = 1;
    m_nctaid.z = 1;
    m_gridid = 0;
    m_valid = true;
  }
  void set_tid(dim3 tid) { m_tid = tid; }
  void cpy_tid_to_reg(dim3 tid);
  void set_ctaid(dim3 ctaid) { m_ctaid = ctaid; }
  void set_ntid(dim3 tid) { m_ntid = tid; }
  void set_nctaid(dim3 cta_size) { m_nctaid = cta_size; }

  unsigned get_builtin(int builtin_id, unsigned dim_mod);

  void set_done();
  bool is_done() { return m_thread_done; }
  unsigned donecycle() const { return m_cycle_done; }

  unsigned next_instr() {
    m_icount++;
    m_branch_taken = false;
    return m_PC;
  }
  bool branch_taken() const { return m_branch_taken; }
  unsigned get_pc() const { return m_PC; }
  void set_npc(unsigned npc) { m_NPC = npc; }
  void set_npc(const function_info *f);
  void callstack_push(unsigned npc, unsigned rpc, const symbol *return_var_src,
                      const symbol *return_var_dst, unsigned call_uid);
  bool callstack_pop();
  void callstack_push_plus(unsigned npc, unsigned rpc,
                           const symbol *return_var_src,
                           const symbol *return_var_dst, unsigned call_uid);
  bool callstack_pop_plus();
  void dump_callstack() const;
  std::string get_location() const;
  const ptx_instruction *get_inst() const;
  const ptx_instruction *get_inst(addr_t pc) const;
  bool rpc_updated() const { return m_RPC_updated; }
  bool last_was_call() const { return m_last_was_call; }
  unsigned get_rpc() const { return m_RPC; }
  void clearRPC() {
    m_RPC = -1;
    m_RPC_updated = false;
    m_last_was_call = false;
  }
  unsigned get_return_PC() { return m_callstack.back().m_PC; }
  void update_pc() { m_PC = m_NPC; }
  void dump_regs(FILE *fp);
  void dump_modifiedregs(FILE *fp);
  void clear_modifiedregs() {
    m_debug_trace_regs_modified.back().clear();
    m_debug_trace_regs_read.back().clear();
  }
  function_info *get_finfo() { return m_func_info; }
  const function_info *get_finfo() const { return m_func_info; }
  void push_breakaddr(const operand_info &breakaddr);
  const operand_info &pop_breakaddr();
  void enable_debug_trace() { m_enable_debug_trace = true; }
  unsigned get_local_mem_stack_pointer() const {
    return m_local_mem_stack_pointer;
  }

  memory_space *get_global_memory() { return m_gpu->get_global_memory(); }
  memory_space *get_tex_memory() { return m_gpu->get_tex_memory(); }
  memory_space *get_surf_memory() { return m_gpu->get_surf_memory(); }
  memory_space *get_param_memory() { return m_kernel.get_param_memory(); }
  const gpgpu_functional_sim_config &get_config() const {
    return m_gpu->get_config();
  }
  bool isInFunctionalSimulationMode() { return m_functionalSimulationMode; }
  void exitCore() {
    // m_core is not used in case of functional simulation mode
    if (!m_functionalSimulationMode) m_core->warp_exit(m_hw_wid);
  }

  void registerExit() { m_cta_info->register_thread_exit(this); }
  unsigned get_reduction_value(unsigned ctaid, unsigned barid) {
    return m_core->get_reduction_value(ctaid, barid);
  }
  void and_reduction(unsigned ctaid, unsigned barid, bool value) {
    m_core->and_reduction(ctaid, barid, value);
  }
  void or_reduction(unsigned ctaid, unsigned barid, bool value) {
    m_core->or_reduction(ctaid, barid, value);
  }
  void popc_reduction(unsigned ctaid, unsigned barid, bool value) {
    m_core->popc_reduction(ctaid, barid, value);
  }

  // Jin: get corresponding kernel grid for CDP purpose
  kernel_info_t &get_kernel() { return m_kernel; }

  // Weili: access symbol_table
  symbol_table *get_symbol_table() { return m_symbol_table; }

 public:
  addr_t m_last_effective_address;
  bool m_branch_taken;
  memory_space_t m_last_memory_space;
  dram_callback_t m_last_dram_callback;
  memory_space *m_shared_mem;
  memory_space *m_sstarr_mem;
  memory_space *m_local_mem;
  ptx_warp_info *m_warp_info;
  ptx_cta_info *m_cta_info;
  ptx_reg_t m_last_set_operand_value;

 private:
  bool m_functionalSimulationMode;
  unsigned m_uid;
  kernel_info_t &m_kernel;
  core_t *m_core;
  gpgpu_t *m_gpu;
  bool m_valid;
  dim3 m_ntid;
  dim3 m_tid;
  dim3 m_nctaid;
  dim3 m_ctaid;
  unsigned m_gridid;
  bool m_thread_done;
  unsigned m_hw_sid;
  unsigned m_hw_tid;
  unsigned m_hw_wid;
  unsigned m_hw_ctaid;

  unsigned m_icount;
  unsigned m_PC;
  unsigned m_NPC;
  unsigned m_RPC;
  bool m_RPC_updated;
  bool m_last_was_call;
  unsigned m_cycle_done;

  int m_barrier_num;
  bool m_at_barrier;

  symbol_table *m_symbol_table;
  function_info *m_func_info;

  std::list<stack_entry> m_callstack;
  unsigned m_local_mem_stack_pointer;

  typedef tr1_hash_map<const symbol *, ptx_reg_t> reg_map_t;
  std::list<reg_map_t> m_regs;
  std::list<reg_map_t> m_debug_trace_regs_modified;
  std::list<reg_map_t> m_debug_trace_regs_read;
  bool m_enable_debug_trace;

  std::stack<class operand_info, std::vector<operand_info> > m_breakaddrs;
};

addr_t generic_to_local(unsigned smid, unsigned hwtid, addr_t addr);
addr_t generic_to_shared(unsigned smid, addr_t addr);
addr_t generic_to_global(addr_t addr);
addr_t local_to_generic(unsigned smid, unsigned hwtid, addr_t addr);
addr_t shared_to_generic(unsigned smid, addr_t addr);
addr_t global_to_generic(addr_t addr);
bool isspace_local(unsigned smid, unsigned hwtid, addr_t addr);
bool isspace_shared(unsigned smid, addr_t addr);
bool isspace_global(addr_t addr);
memory_space_t whichspace(addr_t addr);

extern unsigned g_ptx_thread_info_uid_next;

#endif
