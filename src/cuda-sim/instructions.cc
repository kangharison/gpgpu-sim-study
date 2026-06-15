// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// Jimmy Kwa, George L. Yuan, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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
 * [한국어 설명] PTX 명령어 시맨틱 구현 (instructions.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 기능 시뮬레이터(functional simulator)에서 PTX/PTXPlus 명령어
 * 각각의 실행 시맨틱을 구현한다. 모든 PTX 연산(정수/부동소수점 산술, 논리, 메모리 접근,
 * 제어 흐름, Tensor Core wmma 등)의 실제 계산 로직이 이 파일에 집약되어 있다.
 * 각 PTX opcode에 대응하는 `xxx_impl()` 함수가 존재하며, 이 함수들이 시뮬레이트된
 * 스레드 레지스터 파일과 메모리 공간(전역/공유/로컬/상수)을 직접 읽고 쓴다.
 * 기능 시뮬레이션은 타이밍 모델과 분리되어 있으며, 이 파일은 순수하게 명령어의
 * 결과값이 무엇인지를 결정하는 역할을 담당한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 기능 시뮬레이션 계층(cuda-sim/)에 속한다. GPGPU-Sim의 실행 흐름은 다음과 같다:
 *   CUDA 애플리케이션 → libcuda 인터셉트 → gpgpusim_entrypoint.cc
 *   → shader.cc (타이밍 모델, warp 스케줄링)
 *   → cuda-sim.cc (ptx_exec_inst) → [이 파일] (명령어별 xxx_impl 함수 호출)
 * ptx_exec_inst()는 opcode를 보고 g_opcode_string[] 및 opcodes.def X-매크로
 * 디스패치 테이블을 통해 각 xxx_impl 함수 포인터를 찾아 호출한다.
 * 실행 컨텍스트는 GPU 기능 시뮬레이션 — 호스트 유저스페이스에서 동작하며
 * 시뮬레이터가 각 warp의 각 스레드를 순차적으로 대리 실행한다.
 * 타이밍(사이클 카운팅)은 shader.cc/gpu-sim.cc에서 담당하고,
 * 이 파일은 "이 명령어가 실행된 결과 레지스터/메모리가 어떻게 바뀌는가"에만 집중한다.
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - ptx_ir.h / ptx_sim.h: ptx_instruction, ptx_thread_info, operand_info,
 *     ptx_reg_t, symbol, type_info_key 등 핵심 자료구조 정의
 *   - opcodes.h / opcodes.def: opcode enum과 g_opcode_string[] 정의 (X-매크로)
 *   - abstract_hardware_model.h: GPU 메모리 공간(memory_space) 추상화
 *   - cuda-math.h: GPU 수학 에뮬레이션 함수
 *   - half.h / half.hpp: FP16(반정밀도 부동소수점) 에뮬레이션 라이브러리
 *   - cuda_device_runtime.h: CDP(CUDA Dynamic Parallelism) 지원
 *   - libcuda/gpgpu_context.h: 시뮬레이터 전역 컨텍스트(gpgpu_context)
 * 이 파일에 의존하는 모듈:
 *   - cuda-sim/cuda-sim.cc: ptx_exec_inst()가 xxx_impl 함수 포인터를 호출
 *   - opcodes.def: OP_DEF/OP_W_DEF 매크로로 함수 포인터 테이블 생성
 * 데이터 흐름:
 *   ptx_thread_info (레지스터 파일, 메모리 공간) → get_operand_value() 경유
 *   → xxx_impl() 에서 계산 수행 → set_operand_value() 경유 → ptx_thread_info 갱신
 *
 * === 주요 함수/구조체 요약 ===
 * - thread_group_offset()   : wmma(Tensor Core) 16x16 타일 내 스레드별 행렬 오프셋 매핑
 * - acc_float_offset()      : wmma float 누산기 인덱스를 행렬 메모리 오프셋으로 변환
 * - ptx_thread_info::set_reg()            : 레지스터 파일에 값 기록 (디버그 트레이스 포함)
 * - ptx_thread_info::get_reg()            : 레지스터 파일에서 값 읽기 (미초기화 경고 포함)
 * - ptx_thread_info::get_operand_value()  : 피연산자 타입(레지스터/내장/메모리/리터럴 등)에
 *                                           따른 복잡한 값 읽기 디스패치 로직
 * - ptx_thread_info::set_operand_value()  : 결과값을 레지스터 또는 메모리에 기록;
 *                                           carry/overflow 플래그를 predicate 레지스터에 저장
 * - abs_impl()  : PTX abs 명령어 — 정수/부동소수점 절댓값
 * - addp_impl() : PTXPlus add.cc 명령어 — carry-in 포함 덧셈, carry/overflow 플래그 갱신
 * - add_impl()  : PTX add 명령어 — 전 타입 덧셈, 부동소수점 반올림 모드 적용
 * - addc_impl() : PTX addc 명령어 — 미구현 (inst_not_implemented 호출)
 */

#include "instructions.h"    // [한국어] 이 파일의 헤더 — xxx_impl 함수 시그니처, 공용 상수, 전방선언
#include "half.h"             // [한국어] FP16(반정밀도) 연산용 C 인터페이스 헤더 (half_float 래퍼)
#include "half.hpp"           // [한국어] FP16 C++ 라이브러리 본체 — half_float::half 타입과 산술 연산자 정의
#include "opcodes.h"          // [한국어] PTX opcode enum과 OP_DEF/OP_W_DEF X-매크로 정의; g_opcode_string[] 생성에 사용
#include "ptx_ir.h"           // [한국어] PTX IR 자료구조 — ptx_instruction, operand_info, symbol, type_info_key 등
#include "ptx_sim.h"          // [한국어] 기능 시뮬레이터 상태 — ptx_thread_info, ptx_reg_t, reg_map_t, memory_space 등
typedef void *yyscan_t;       // [한국어] PTX Flex 렉서의 스캐너 핸들 타입 — ptx.tab.h 포함 전에 정의해야 링크 오류 방지
class ptx_recognizer;         // [한국어] PTX 파서 클래스 전방선언 — ptx.tab.h에서 실제 정의; 헤더 순환 의존 회피용
#include <assert.h>           // [한국어] assert() — 불변 조건 위반 시 즉시 프로세스 중단; 시뮬레이터 디버깅 필수
#include <fenv.h>             // [한국어] IEEE 754 부동소수점 환경 제어 — fegetround()/fesetround()로 GPU 반올림 모드 에뮬레이션
#include <math.h>             // [한국어] C 수학 함수(fabs, sqrt, sin 등) — PTX 부동소수점 명령어 시맨틱 구현에 사용
#include <stdio.h>            // [한국어] 표준 I/O (printf, fopen 등) — 오류 메시지 출력 및 레지스터 덤프/복구 파일 I/O
#include <stdlib.h>           // [한국어] abort(), atoi() 등 — 미구현 명령어 호출 시 시뮬레이터 즉시 종료
#include <string.h>           // [한국어] strtok() 등 C 문자열 함수 — 레지스터 체크포인트 파일 파싱에 사용
#include <cmath>              // [한국어] C++ 수학 함수(std::isnan, std::isinf 등) — NaN/Inf 판별 로직 지원
#include <map>                // [한국어] std::map — reg_map_t 내부적으로 레지스터 심볼→값 매핑에 사용
#include <sstream>            // [한국어] std::stringstream — 오류 메시지 문자열 조합
#include <string>             // [한국어] std::string — 심볼 이름, 위치 정보 등 문자열 처리
#include "../abstract_hardware_model.h"  // [한국어] memory_space 추상화, new_addr_type, GPU 메모리 공간 상수(global_space 등)
#include "../gpgpu-sim/gpu-sim.h"        // [한국어] gpgpu_sim 클래스 — 전역 시뮬레이터 상태 접근(설정, 클럭, 통계)
#include "../gpgpu-sim/shader.h"         // [한국어] shader_core_ctx — SM(Streaming Multiprocessor) 타이밍 모델 참조
#include "cuda-math.h"                   // [한국어] CUDA 내장 수학 함수 에뮬레이션(정수→float 변환, __saturatef 등)
#include "cuda_device_printf.h"          // [한국어] GPU 디바이스 printf(vprintf_impl) 지원 헤더
#include "ptx.tab.h"                     // [한국어] Bison 파서 생성 토큰 정의 — ROW, COL, LOAD_A 등 PTX 문법 심볼
#include "ptx_loader.h"                  // [한국어] PTX 모듈 로딩 인터페이스 — 커널 바이너리 파싱/등록 함수

// Jin: include device runtime for CDP
#include "cuda_device_runtime.h"   // [한국어] CDP(CUDA Dynamic Parallelism) 런타임 — 디바이스에서 커널 재귀 실행 지원

#include <stdarg.h>                     // [한국어] va_list 등 가변인수 매크로 — printf 계열 구현에 필요
#include "../../libcuda/gpgpu_context.h"  // [한국어] gpgpu_context — 시뮬레이터 전역 싱글톤, g_the_gpu 등 접근

using half_float::half;  // [한국어] FP16 타입 half를 네임스페이스 없이 사용 — F16_TYPE 명령어 구현 편의

/* [한국어] g_opcode_string[] — PTX opcode 열거값 인덱스에서 문자열("add", "mul" 등)로의 매핑 테이블.
 * NUM_OPCODES 크기의 const char* 배열로, opcodes.def의 OP_DEF/OP_W_DEF X-매크로를 이용해 자동 생성된다.
 * cuda-sim.cc의 ptx_exec_inst()가 오류 메시지 출력 시 opcode 이름을 출력하는 데 사용한다.
 * 함수 포인터 디스패치 테이블(g_func_table[])과 1:1 대응 인덱스를 가진다. */
const char *g_opcode_string[NUM_OPCODES] = {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION) STR,    // [한국어] 일반 PTX 명령어: opcode 열거값→문자열(STR) 추출
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION) STR,  // [한국어] PTXPlus 확장 명령어(carry/overflow 변형)의 문자열 추출
#include "opcodes.def"  // [한국어] opcodes.def 전개 — OP_DEF/OP_W_DEF 매크로로 enum 값마다 STR 문자열 삽입
#undef OP_DEF           // [한국어] 매크로 정의 해제 — 이후 코드에서 동일 매크로를 다른 목적으로 재정의하기 위해 클린업
#undef OP_W_DEF         // [한국어] OP_W_DEF 매크로 정의 해제
};
// Using profiled information::check the TensorCoreMatrixArrangement.xls for
// details

/*
 * [한국어]
 * thread_group_offset - wmma 행렬 타일 내 스레드 인덱스를 메모리 오프셋으로 변환
 *
 * @thread     : warp 내 스레드 번호 (0~31); 4개씩 묶어 thread_group(0~7) 형성
 * @wmma_type  : 행렬 피연산자 종류 — LOAD_A, LOAD_B, LOAD_C, STORE_D 중 하나
 * @wmma_layout: 행렬 저장 순서 — ROW(행 우선) 또는 COL(열 우선)
 * @type       : 원소 데이터 타입 — F16_TYPE(FP16) 또는 F32_TYPE(FP32)
 * @stride     : 행렬 행(row) 간격 (열 개수); 선형 주소 변환 시 행 오프셋 계산에 사용
 * @return     : 해당 스레드가 담당하는 행렬 원소의 1차원 배열 오프셋
 *
 * wmma(Warp Matrix Multiply Accumulate) 명령어는 Tensor Core를 사용하는 16×16 행렬 연산이다.
 * NVIDIA GPU에서 warp 내 32개 스레드가 각자 다른 행렬 원소를 담당하는데,
 * 어떤 스레드가 어떤 원소를 처리하는지는 하드웨어 내부 배치(layout)에 따라 결정된다.
 * 이 함수는 실제 NVIDIA GPU에서 프로파일링하여 얻은 스레드→행렬 원소 매핑 테이블
 * (TensorCoreMatrixArrangement.xls 참조)을 시뮬레이터에서 재현한다.
 * load_a_row[], load_b_col[] 등의 테이블 값은 모두 하드웨어 관측 결과이므로
 * 임의로 변경하면 안 된다.
 * 주목: load_b_row[]와 load_b_col[]은 A 행렬과 반대로 배치되어 있다
 * (B 행렬에서 ROW는 열 방향 밀집, COL은 행 방향 밀집).
 *
 * 실행 컨텍스트: cuda-sim.cc에서 wmma.load/store/mma impl 함수 내부에서 호출됨.
 * 단일 스레드 시뮬레이션 루프에서 호출되므로 재진입 안전.
 *
 * 호출 체인:
 *   wmma_load_impl / wmma_store_impl / wmma_mma_impl → [thread_group_offset] → (반환값 사용)
 */
unsigned thread_group_offset(int thread, unsigned wmma_type,
                             unsigned wmma_layout, unsigned type, int stride) {
  unsigned offset;  // [한국어] 최종 계산될 1차원 배열 오프셋 (함수 반환값)

  /* [한국어] 이하 8개 원소 배열은 thread_group 인덱스(0~7)별 기저 오프셋.
   * 실제 NVIDIA GPU에서 프로파일링한 값으로, A 행렬 ROW/COL 레이아웃,
   * B 행렬 ROW/COL, C/D 행렬 FP32/FP16 레이아웃 각각을 담는다.
   * 값 단위: 행렬 원소 개수 (바이트 아님). */
  unsigned load_a_row[8] = {0, 128, 0, 128, 64, 192, 64, 192};  // [한국어] A 행렬, ROW 레이아웃: thread_group별 기저 오프셋
  unsigned load_a_col[8] = {0, 8, 0, 8, 4, 12, 4, 12};          // [한국어] A 행렬, COL 레이아웃: thread_group별 기저 오프셋
  unsigned load_b_row[8] = {0, 8, 0, 8, 4, 12, 4, 12};          // [한국어] B 행렬, ROW 레이아웃 (A의 COL과 동일값 — B는 전치 방향)
  unsigned load_b_col[8] = {0, 128, 0, 128, 64, 192, 64, 192};  // [한국어] B 행렬, COL 레이아웃 (A의 ROW와 동일값)
  unsigned load_c_float_row[8] = {0, 128, 8, 136, 64, 192, 72, 200};  // [한국어] C/D FP32, ROW 레이아웃
  unsigned load_c_float_col[8] = {0, 8, 128, 136, 4, 12, 132, 140};   // [한국어] C/D FP32, COL 레이아웃
  unsigned load_c_half_row[8] = {0, 128, 8, 136, 64, 192, 72, 200};   // [한국어] C/D FP16, ROW 레이아웃 (FP32와 동일 기저)
  unsigned load_c_half_col[8] = {0, 8, 128, 136, 4, 12, 132, 140};    // [한국어] C/D FP16, COL 레이아웃

  unsigned thread_group = thread / 4;  // [한국어] 스레드를 4개씩 묶은 그룹 번호 (0~7); 행렬 원소 묶음 단위
  unsigned in_tg_index = thread % 4;   // [한국어] thread_group 내 스레드 위치 (0~3); 같은 묶음 내 세부 오프셋 결정

  switch (wmma_type) {
    case LOAD_A:  // [한국어] wmma.load 피연산자 A 행렬 (MxK; M=16, K=16 기본)
      if (wmma_layout == ROW)  // [한국어] A가 행 우선 저장: 연속 메모리가 같은 행 원소들
        offset = load_a_row[thread_group] + 16 * in_tg_index;  // [한국어] 기저+그룹 내 선형 오프셋 (16원소 단위)
      else  // [한국어] COL 레이아웃: 연속 메모리가 같은 열 원소들
        offset = load_a_col[thread_group] + 16 * in_tg_index;  // [한국어] 열 우선 기저+그룹 내 오프셋
      break;

    case LOAD_B:  // [한국어] wmma.load 피연산자 B 행렬 (KxN; K=16, N=16 기본)
      if (wmma_layout == ROW)  // [한국어] B가 행 우선: A와 테이블 값이 반대로 되어 있음(전치 배치)
        offset = load_b_row[thread_group] + 16 * in_tg_index;
      else
        offset = load_b_col[thread_group] + 16 * in_tg_index;
      break;

    case LOAD_C:   // [한국어] wmma.load 피연산자 C (누산기 초기값) — MxN 행렬
    case STORE_D:  // [한국어] wmma.store 결과 D (연산 결과) — LOAD_C와 동일 레이아웃 공유
      if (type == F16_TYPE) {  // [한국어] C/D 원소가 FP16인 경우 — 더 촘촘한 오프셋 간격
        if (wmma_layout == ROW)
          offset = load_c_half_row[thread_group] + 16 * in_tg_index;  // [한국어] FP16 행 우선: 16 단위 선형
        else
          offset = load_c_half_col[thread_group] + in_tg_index;  // [한국어] FP16 열 우선: 1 단위 선형 (FP32보다 밀집)
      } else {  // [한국어] FP32(F32_TYPE) 또는 기타 타입 — 원소 크기가 2배이므로 오프셋 계산이 다름
        if (wmma_layout == ROW)  // [한국어] FP32 행 우선: thread_group 기저만 적용
          offset = load_c_float_row[thread_group];
        else  // [한국어] FP32 열 우선
          offset = load_c_float_col[thread_group];

        /* [한국어] in_tg_index(0~3)에 따라 같은 thread_group 내에서 오프셋 세분화.
         * FP32는 원소 크기가 커서 같은 그룹의 4개 스레드가 비연속 원소를 처리한다.
         * 각 case 값은 프로파일링 결과(TensorCoreMatrixArrangement.xls)에서 도출됨. */
        switch (in_tg_index) {
          case 0:  // [한국어] 그룹 내 첫 번째 스레드: 기저 오프셋 그대로 사용
            break;
          case 1:  // [한국어] 그룹 내 두 번째 스레드
            if (wmma_layout == ROW)
              offset += 16;  // [한국어] 행 우선: 같은 열, 다음 두 번째 행 묶음(+16 원소)
            else
              offset += 1;   // [한국어] 열 우선: 같은 행, 바로 다음 열(+1 원소)
            break;
          case 2:  // [한국어] 그룹 내 세 번째 스레드
            if (wmma_layout == ROW)
              offset += 2;   // [한국어] 행 우선: 같은 행, 두 번째 열 그룹(+2)
            else
              offset += 32;  // [한국어] 열 우선: 열 축 이동(+32)
            break;
          case 3:  // [한국어] 그룹 내 네 번째 스레드
            if (wmma_layout == ROW)
              offset += 18;  // [한국어] 행 우선: 16+2 복합 이동
            else
              offset += 33;  // [한국어] 열 우선: 32+1 복합 이동
            break;
          default:
            abort();  // [한국어] in_tg_index는 0~3만 가능; 그 외는 하드웨어 불가 — 시뮬레이터 오류
        }
      }
      break;

    default:
      abort();  // [한국어] 알 수 없는 wmma_type: 정의되지 않은 행렬 피연산자 — 즉시 종료
  }
  /* [한국어] 선형 오프셋을 2차원 행렬의 (행, 열)로 분해한 후 stride를 적용해 1차원 메모리 오프셋으로 재합산.
   * offset / 16 : 행렬 내 행 번호 (16열 단위 블록으로 분리)
   * offset % 16 : 행 내 열 번호
   * stride : 실제 행렬의 열 개수(leading dimension); 메모리에서 행 간 거리
   * 공식: linear_addr = row * stride + col */
  offset = (offset / 16) * stride + offset % 16;
  return offset;  // [한국어] 이 스레드가 담당하는 행렬 원소의 1차원 배열 오프셋 반환
}

/*
 * [한국어]
 * acc_float_offset - wmma FP32 누산기 인덱스를 행렬 메모리 오프셋으로 변환
 *
 * @index      : wmma 누산기 레지스터 인덱스 (0~7); warp 내 스레드가 담당하는 C/D 원소 번호
 * @wmma_layout: 행렬 저장 순서 — ROW(행 우선) 또는 COL(열 우선)
 * @stride     : 행렬 행(row) 간격 (열 개수); 선형 주소 변환에 사용
 * @return     : 해당 누산기 인덱스가 가리키는 FP32 행렬 원소의 1차원 배열 오프셋
 *
 * wmma.mma 명령어를 실행하면 각 스레드는 C/D 행렬에서 최대 8개 원소를 담당한다.
 * 이 함수는 그 8개 원소 각각의 행렬 내 오프셋을 반환한다.
 * c_row_offset[], c_col_offset[] 값은 하드웨어에서 프로파일링한 결과이며,
 * thread_group_offset()과 쌍으로 사용된다 (thread_group_offset은 그룹 기저,
 * acc_float_offset은 그룹 내 누산기 인덱스 오프셋을 제공).
 *
 * 실행 컨텍스트: wmma_mma_impl() 내부에서 호출됨. 단일 스레드 시뮬레이션 루프.
 *
 * 호출 체인:
 *   wmma_mma_impl → [acc_float_offset] → (반환값으로 행렬 메모리 주소 계산)
 */
int acc_float_offset(int index, int wmma_layout, int stride) {
  /* [한국어] FP32 누산기 인덱스(0~7)별 행렬 오프셋 테이블.
   * ROW 레이아웃: 원소가 행 방향으로 연속 배치됨 → 오프셋이 열 단위로 증가
   * COL 레이아웃: 원소가 열 방향으로 연속 배치됨 → 오프셋이 행 단위로 증가
   * 값은 하드웨어 프로파일링 결과이므로 수정 금지. */
  int c_row_offset[] = {0, 1, 32, 33, 4, 5, 36, 37};   // [한국어] ROW 레이아웃: 인덱스별 원소 오프셋 (열 방향 밀집)
  int c_col_offset[] = {0, 16, 2, 18, 64, 80, 66, 82};  // [한국어] COL 레이아웃: 인덱스별 원소 오프셋 (행 방향 밀집)
  int offset;  // [한국어] 최종 계산될 1차원 배열 오프셋

  if (wmma_layout == ROW)  // [한국어] 행 우선 저장: ROW 테이블로 오프셋 조회
    offset = c_row_offset[index];
  else if (wmma_layout == COL)  // [한국어] 열 우선 저장: COL 테이블로 오프셋 조회
    offset = c_col_offset[index];
  else {  // [한국어] ROW/COL 이외의 레이아웃은 wmma에서 미지원 — 오류
    printf("wrong layout");  // [한국어] 잘못된 레이아웃 오류 출력
    abort();  // [한국어] 복구 불가 오류: 즉시 시뮬레이터 종료
  }
  /* [한국어] thread_group_offset()과 동일한 2D→1D 변환 공식 적용.
   * offset / 16 : 행 번호, offset % 16 : 열 번호
   * 행렬의 실제 stride(열 개수)를 곱해 1차원 주소로 환산 */
  offset = (offset / 16) * stride + offset % 16;
  return offset;  // [한국어] 누산기 인덱스에 해당하는 행렬 원소의 1차원 오프셋 반환
}

/* [한국어] 전방선언 — 이 파일 내에서 함수 정의보다 먼저 호출될 수 있는 함수들을 미리 선언.
 * C++ 컴파일러가 호출 지점에서 시그니처를 알 수 있도록 하기 위해 필요하다. */

/*
 * [한국어]
 * inst_not_implemented - 구현되지 않은 PTX 명령어 처리 공통 함수
 *
 * @pI: 실행 중인 PTX 명령어
 *
 * 아직 구현되지 않은 명령어를 만났을 때 오류 메시지를 출력하고 abort()로 시뮬레이션을 중단한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 */

void inst_not_implemented(const ptx_instruction *pI);  // [한국어] 미구현 명령어 처리 — 오류 메시지 출력 후 abort

/*
 * [한국어]
 * srcOperandModifiers - 소스 피연산자 수정자 처리 (메모리/lohi/neg)
 *
 * @opData: 원본 소스 피연산자 값
 * @opInfo: 소스 오퍼랜드 정보
 * @dstInfo: 목적지 오퍼랜드 정보
 * @type: PTX 데이터 타입
 * @thread: 현재 실행 중인 PTX 스레드
 * @return: 수정자가 적용된 피연산자 값
 *
 * 소스 오퍼랜드가 메모리 참조(global/shared/const)일 경우 메모리에서 값을 읽고,
 * .lo/.hi 수식어에 따라 16비트를 추출하거나, .neg 수식어 시 부동소수점 부호를 반전한다.
 * 실행 컨텍스트: 일부 명령어의 피연산자 준비 단계 (현재는 주석 처리된 곳이 많음).
 */

ptx_reg_t srcOperandModifiers(ptx_reg_t opData, operand_info opInfo,
                              operand_info dstInfo, unsigned type,
                              ptx_thread_info *thread);  // [한국어] 소스 피연산자 수식자 적용 (abs, neg 등) 후 값 반환

/*
 * [한국어]
 * video_mem_instruction - PTX 비디오 메모리 명령어(vmax/vmin)의 공통 구현
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 * @op_code: VMAX(0) 또는 VMIN(1)
 *
 * PTX 비디오 메모리 연산 vmax/vmin을 S32 타입에 대해 처리한다.
 * 두 피연산자의 max/min을 취한 뒤, atomic 옵션(max/min)이 있으면 c와 다시 비교한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   vmax_impl/vmin_impl → [video_mem_instruction] → set_operand_value()
 */

void video_mem_instruction(const ptx_instruction *pI, ptx_thread_info *thread,
                           int op_code);  // [한국어] 비디오 명령어(vabsdiff, vmin 등) 디스패처

void sign_extend(ptx_reg_t &data, unsigned src_size, const operand_info &dst);  // [한국어] 부호 확장 — 좁은 부호 정수를 목적지 크기로 확장

/*
 * [한국어]
 * ptx_thread_info::set_reg - 시뮬레이트된 스레드의 레지스터 파일에 값을 기록
 *
 * @reg  : 기록 대상 레지스터 심볼 포인터; ptx_ir.h의 symbol 클래스로 레지스터 이름/타입/uid 보유
 * @value: 기록할 값; ptx_reg_t 유니온(u8/u16/u32/u64/s8/.../f16/f32/f64/pred/bits/u128)
 * @return: void
 *
 * PTX 명령어 실행 시 결과를 목적지 레지스터에 저장하는 핵심 메서드.
 * m_regs는 스택 구조(함수 호출 깊이별 스코프)이며, back()이 현재 함수의 레지스터 파일을 가리킨다.
 * 디버그 트레이스가 활성화된 경우(m_enable_debug_trace) 수정된 레지스터를
 * m_debug_trace_regs_modified에도 기록하여 사이클별 레지스터 변화를 추적한다.
 * 실행 컨텍스트: 기능 시뮬레이터(cuda-sim.cc) 내 단일 스레드 실행 루프;
 * 각 스레드는 순차적으로 처리되므로 레지스터 파일 동시 접근 없음 (락 불필요).
 *
 * 호출 체인:
 *   xxx_impl() → set_operand_value() → [set_reg] → m_regs.back() 갱신
 */
void ptx_thread_info::set_reg(const symbol *reg, const ptx_reg_t &value) {
  assert(reg != NULL);  // [한국어] NULL 심볼로 쓰기 시도는 구현 버그 — 즉시 중단
  if (reg->name() == "_") return;  // [한국어] "_"는 PTX에서 결과 버리기(discard) 목적지 — 실제 쓰기 불필요
  assert(!m_regs.empty());  // [한국어] 레지스터 스택이 비어있으면 함수 진입 상태 오류
  assert(reg->uid() > 0);   // [한국어] 유효한 심볼 uid(고유 ID) 확인 — uid=0은 미초기화 심볼
  m_regs.back()[reg] = value;  // [한국어] 현재 함수 스코프의 레지스터 파일(해시맵)에 심볼→값 저장
  if (m_enable_debug_trace) m_debug_trace_regs_modified.back()[reg] = value;  // [한국어] 디버그 모드: 수정 레지스터 로그에도 동일값 기록 (사이클별 diff 분석용)
  m_last_set_operand_value = value;  // [한국어] 마지막으로 쓴 값 캐시 — 이후 conditional 코드에서 참조 가능
}

/*
 * [한국어]
 * ptx_thread_info::print_reg_thread - 현재 스레드의 레지스터 파일을 파일로 덤프
 *
 * @fname: 출력할 파일 경로; 기존 파일이 있으면 덮어쓴다
 * @return: void
 *
 * 시뮬레이터 체크포인트/디버깅 목적으로 현재 스레드의 모든 레지스터 값을 텍스트 파일에 기록한다.
 * 출력 형식: "<레지스터명> <값(u64)> <선언위치> <크기(바이트)>\n" 한 줄씩.
 * resume_reg_thread()와 쌍을 이루며, 저장한 상태를 복원할 수 있다.
 * 실행 컨텍스트: 디버깅/체크포인트 목적으로만 호출; 정상 시뮬레이션 루프에서는 미호출.
 * 주의: 주석 처리된 m_regs.pop_back()은 의도적으로 비활성화 — 스코프 파괴 없이 덤프만 수행.
 *
 * 호출 체인:
 *   (디버그/체크포인트 트리거) → [print_reg_thread] → fprintf(fp, ...) → fclose
 */
void ptx_thread_info::print_reg_thread(char *fname) {
  FILE *fp = fopen(fname, "w");  // [한국어] 레지스터 덤프 파일 쓰기 모드로 열기
  assert(fp != NULL);  // [한국어] 파일 열기 실패(권한/경로 오류) 시 즉시 중단

  int size = m_regs.size();  // [한국어] 현재 레지스터 스택 깊이(함수 호출 레벨 수)

  if (size > 0) {  // [한국어] 레지스터 스택이 비어있지 않을 때만 덤프 — 미진입 스레드 방어
    reg_map_t reg = m_regs.back();  // [한국어] 현재(최상위) 함수 스코프의 레지스터 맵 복사

    reg_map_t::const_iterator it;  // [한국어] 레지스터 맵 순회 이터레이터 (symbol* → ptx_reg_t)
    for (it = reg.begin(); it != reg.end(); ++it) {  // [한국어] 모든 레지스터를 순회하며 파일에 기록
      const std::string &name = it->first->name();          // [한국어] 레지스터 심볼 이름 (예: "$r0", "$p1")
      const std::string &dec = it->first->decl_location();  // [한국어] PTX 소스에서 이 레지스터가 선언된 위치 (파일:라인)
      unsigned size = it->first->get_size_in_bytes();        // [한국어] 레지스터 크기 (바이트); 타입별로 1/2/4/8
      fprintf(fp, "%s %llu %s %d\n", name.c_str(), it->second, dec.c_str(),
              size);  // [한국어] "이름 값(u64진수) 선언위치 크기" 형식으로 한 줄 출력; it->second는 ptx_reg_t (암묵적 u64 변환)
    }
    // m_regs.pop_back();  // [한국어] 의도적으로 비활성화: 스코프 스택을 파괴하지 않고 덤프만 수행
  }
  fclose(fp);  // [한국어] 파일 핸들 닫기 — 버퍼 플러시 및 OS 리소스 반납
}

/*
 * [한국어]
 * ptx_thread_info::resume_reg_thread - 파일에서 레지스터 상태를 복원
 *
 * @fname  : print_reg_thread()로 저장된 레지스터 덤프 파일 경로
 * @symtab : 이 스레드가 속한 PTX 커널의 심볼 테이블; 레지스터 이름으로 symbol* 조회에 사용
 * @return : void
 *
 * print_reg_thread()의 역연산. 덤프 파일을 한 줄씩 읽어 레지스터 심볼을 찾고 값을 복원한다.
 * 시뮬레이터 체크포인트 재개(resume) 시나리오에서 사용된다.
 * 파일 형식: "<이름> <값> <선언위치> <크기>" 한 줄씩;
 * 이름으로 symtab에서 symbol*를 조회하고, 값(atoi)을 ptx_reg_t에 저장한다.
 * 주의: atoi()로 정수 변환 시 64비트 값이 잘릴 수 있음; 디버그/소규모 테스트 용도로만 사용.
 * 실행 컨텍스트: 체크포인트 복원 시 단일 스레드 초기화 루프에서 호출.
 *
 * 호출 체인:
 *   (체크포인트 복원 트리거) → [resume_reg_thread] → symtab->lookup() + m_regs.back() 갱신
 */
void ptx_thread_info::resume_reg_thread(char *fname, symbol_table *symtab) {
  FILE *fp2 = fopen(fname, "r");  // [한국어] 레지스터 덤프 파일 읽기 모드로 열기
  assert(fp2 != NULL);  // [한국어] 파일 열기 실패 시 즉시 중단 (경로/권한 오류)
  // m_regs.push_back( reg_map_t() );  // [한국어] 의도적으로 비활성화: 새 스코프 추가 없이 현재 스코프에 복원
  char line[200];  // [한국어] 한 줄 입력 버퍼 (레지스터 한 항목); 200바이트로 긴 이름도 처리 가능
  while (fgets(line, sizeof line, fp2) != NULL) {  // [한국어] 파일 끝까지 한 줄씩 읽기
    symbol *reg;   // [한국어] 이름으로 조회한 레지스터 심볼 포인터
    char *pch;     // [한국어] strtok 파싱 포인터 — 현재 토큰 위치 추적
    pch = strtok(line, " ");  // [한국어] 공백 기준 첫 번째 토큰 = 레지스터 이름
    char *name = pch;         // [한국어] 레지스터 이름 문자열 포인터 저장
    reg = symtab->lookup(name);  // [한국어] 심볼 테이블에서 이름으로 symbol* 조회
    ptx_reg_t data;  // [한국어] 복원할 레지스터 값 임시 보관
    pch = strtok(NULL, " ");  // [한국어] 두 번째 토큰 = 레지스터 값(십진수 문자열)
    data = atoi(pch);         // [한국어] 문자열→정수 변환 후 ptx_reg_t에 대입; 32비트 제한 주의
    pch = strtok(NULL, " ");  // [한국어] 세 번째 토큰 = 선언 위치 (무시)
    pch = strtok(NULL, " ");  // [한국어] 네 번째 토큰 = 크기 (무시)
    m_regs.back()[reg] = data;  // [한국어] 현재 스코프 레지스터 파일에 복원된 값 저장
  }
  fclose(fp2);  // [한국어] 파일 핸들 닫기 및 OS 리소스 반납
}

/*
 * [한국어]
 * ptx_thread_info::get_reg - 시뮬레이트된 스레드의 레지스터 파일에서 값을 읽기
 *
 * @reg   : 읽을 레지스터 심볼 포인터; ptx_ir.h의 symbol 클래스
 * @return: 레지스터 현재 값 (ptx_reg_t 유니온); 미초기화 시 0x00000000 반환 + 경고 출력
 *
 * PTX 명령어 실행 시 소스 피연산자 레지스터 값을 읽는 핵심 메서드.
 * m_regs.back()은 현재 함수 스코프의 레지스터 파일(symbol*→ptx_reg_t 해시맵)을 가리킨다.
 * 레지스터가 아직 한 번도 쓰여지지 않은 경우(find == end()), 미초기화 경고를 출력하고
 * 0x0으로 초기화한 후 반환한다 — 실제 GPU 하드웨어에서도 초기화 전 레지스터 읽기는 미정의 동작.
 * unfound_register_warned 플래그로 동일 경고를 한 번만 출력 (로그 폭주 방지).
 * 디버그 트레이스 활성 시 읽은 레지스터를 m_debug_trace_regs_read에도 기록.
 * 실행 컨텍스트: 기능 시뮬레이터 단일 스레드 루프; 재진입 안전.
 *
 * 호출 체인:
 *   xxx_impl() → get_operand_value() → [get_reg] → m_regs.back() 조회
 */
ptx_reg_t ptx_thread_info::get_reg(const symbol *reg) {
  static bool unfound_register_warned = false;  // [한국어] 미초기화 레지스터 경고를 한 번만 출력하기 위한 정적 플래그
  assert(reg != NULL);  // [한국어] NULL 심볼로 읽기 시도는 구현 버그 — 즉시 중단
  assert(!m_regs.empty());  // [한국어] 레지스터 스택이 비어있으면 함수 미진입 상태 오류
  reg_map_t::iterator regs_iter = m_regs.back().find(reg);  // [한국어] 현재 스코프 레지스터 파일에서 심볼로 값 검색
  if (regs_iter == m_regs.back().end()) {  // [한국어] 레지스터가 아직 쓰여진 적 없음 — 미초기화 상태
    assert(reg->type()->get_key().is_reg());  // [한국어] 심볼이 실제 레지스터 타입인지 확인 (파라미터/전역변수와 구분)
    const std::string &name = reg->name();        // [한국어] 경고 메시지에 출력할 레지스터 이름
    unsigned call_uid = m_callstack.back().m_call_uid;  // [한국어] 현재 함수 호출의 고유 ID — 어떤 호출 인스턴스인지 식별
    ptx_reg_t uninit_reg;    // [한국어] 미초기화 레지스터에 줄 기본값 컨테이너
    uninit_reg.u32 = 0x0;    // [한국어] 초기값 0x00000000 설정 — 실제 HW는 미정의이나 시뮬레이터에서 재현 가능하게 0으로 통일
    set_reg(reg, uninit_reg);  // give it a value since we are going to warn the
                               // user anyway
    // [한국어] 경고 발행 전 레지스터에 0 값을 먼저 기록해 이후 find()가 성공하도록 처리
    std::string file_loc = get_location();  // [한국어] 현재 실행 중인 PTX 명령어의 소스 위치(파일:라인) 문자열
    if (!unfound_register_warned) {  // [한국어] 첫 번째 미초기화 레지스터 경고만 출력 (이후는 무시)
      printf(
          "GPGPU-Sim PTX: WARNING (%s) ** reading undefined register \'%s\' "
          "(cuid:%u). Setting to 0X00000000. This is okay if you are "
          "simulating the native ISA"
          "\n",
          file_loc.c_str(), name.c_str(), call_uid);
      // [한국어] 경고 메시지: 소스 위치, 레지스터 이름, 호출 uid 출력
      unfound_register_warned = true;  // [한국어] 경고 플래그 세트 — 이후 동일 경고 억제
    }
    regs_iter = m_regs.back().find(reg);  // [한국어] set_reg() 후 재검색 — 이제 반드시 find 성공
  }
  if (m_enable_debug_trace)
    m_debug_trace_regs_read.back()[reg] = regs_iter->second;  // [한국어] 디버그 트레이스 모드: 읽은 레지스터와 값을 로그에 기록 (사이클별 read set 분석용)
  return regs_iter->second;  // [한국어] 레지스터 현재 값 반환 (ptx_reg_t 유니온)
}

/*
 * [한국어]
 * ptx_thread_info::get_operand_value - 피연산자 종류에 따라 값을 읽는 복합 디스패처
 *
 * @op       : 읽을 피연산자 정보 (operand_info); 레지스터/내장변수/즉시 주소/메모리/리터럴/레이블 등
 * @dstInfo  : 목적지 피연산자 정보; 부호 확장(sign_extend) 및 lo/hi 비트 처리에 사용
 * @opType   : 피연산자 데이터 타입 (S8_TYPE, U32_TYPE, F32_TYPE, BB128_TYPE 등)
 * @thread   : 현재 실행 중인 시뮬레이트 스레드 — 메모리 공간 접근에 사용
 * @derefFlag: 1이면 메모리 역참조 수행; 0이면 주소값만 반환 (주소 계산 모드)
 * @return   : 읽어온 피연산자 값 (ptx_reg_t 유니온); 메모리 접근 시 실제 메모리 내용 반환
 *
 * PTX 명령어의 소스 피연산자 종류는 매우 다양하여 이 함수가 모든 경우를 통합 처리한다:
 *   1) 단순 레지스터 ($r0, $p1 등) → get_reg()
 *   2) 내장 변수 (threadIdx.x 등) → get_builtin()
 *   3) 즉시 주소 (mov 등에서 사용) → 직접 offset 값 반환
 *   4) 메모리 피연산자 (ld 명령어에서 [addr+offset]) → 주소 계산 후 derefFlag에 따라 역참조
 *      - 레지스터 기반 주소 / 커널 파라미터 / 전역/로컬/공유/상수 메모리 등 분기
 *   5) 리터럴 상수, 레이블 주소, 공유 메모리 변수, 함수 주소, 커널 파라미터 등
 *   6) BB128_TYPE: 4개 레지스터 → u128 조합
 *   7) BB64_TYPE/FF64_TYPE: 2개 레지스터 → bits.ls/ms 조합
 *   8) double 피연산자 타입 1: s[reg1 + reg2] — 두 레지스터 합산 주소
 *   9) double 피연산자 타입 2: s[reg1 += reg2] — 포스트-인크리먼트
 *   10) double 피연산자 타입 3: s[reg += immediate] — 포스트-인크리먼트(즉시값)
 * lo/hi 16비트 추출, 부정 연산도 이 함수에서 처리한다.
 * 실제 메모리 접근(derefFlag=1) 시 m_last_effective_address와 m_last_memory_space를 갱신.
 *
 * 실행 컨텍스트: 기능 시뮬레이터 단일 스레드 루프; xxx_impl()에서 직접 호출.
 * 에러 경로: 알 수 없는 피연산자 타입 시 printf 후 abort().
 *
 * 호출 체인:
 *   xxx_impl() → [get_operand_value] → get_reg() / get_builtin() / memory_space::read()
 */
ptx_reg_t ptx_thread_info::get_operand_value(const operand_info &op,
                                             operand_info dstInfo,
                                             unsigned opType,
                                             ptx_thread_info *thread,
                                             int derefFlag) {
  ptx_reg_t result, tmp;  // [한국어] result: 최종 반환될 피연산자 값; tmp: 임시 계산용 (현재 미사용)

  if (op.get_double_operand_type() == 0) {  // [한국어] 단일 피연산자 (가장 일반적인 경우)
    if (((opType != BB128_TYPE) && (opType != BB64_TYPE) &&
         (opType != FF64_TYPE)) ||
        (op.get_addr_space() != undefined_space)) {
      // [한국어] BB128/BB64/FF64 타입이 아니거나, 주소 공간이 명시된 경우 — 일반 단일값 읽기
      if (op.is_reg()) {  // [한국어] 일반 레지스터 피연산자 ($r0, $f1, $p2 등)
        result = get_reg(op.get_symbol());  // [한국어] 레지스터 파일에서 심볼로 값 조회
      } else if (op.is_builtin()) {  // [한국어] GPU 내장 변수 (threadIdx.x/y/z, blockIdx.x 등)
        result.u32 = get_builtin(op.get_int(), op.get_addr_offset());  // [한국어] 내장 변수 ID와 offset으로 threadIdx/blockIdx 등 32비트 값 얻기
      } else if (op.is_immediate_address()) {  // [한국어] 즉시 주소 피연산자 (mov.u64 $r, addr 형태)
        result.u64 = op.get_addr_offset();  // [한국어] 즉시값(주소)을 그대로 64비트 결과로 사용
      } else if (op.is_memory_operand()) {  // [한국어] 메모리 피연산자 — ld/st의 [base+offset] 형태; derefFlag에 따라 주소만 또는 값까지 반환
        // a few options here...
        const symbol *sym = op.get_symbol();     // [한국어] 메모리 심볼 (레지스터 또는 전역/로컬/공유 변수)
        const type_info *type = sym->type();     // [한국어] 심볼의 타입 정보 (어떤 메모리 공간에 속하는지)
        const type_info_key &info = type->get_key();  // [한국어] 타입 키 — is_reg/is_global/is_local 등 공간 판별

        if (info.is_reg()) {  // [한국어] 레지스터 간접 주소 — [reg+offset]: 레지스터 값이 기저 주소
          const symbol *name = op.get_symbol();  // [한국어] 기저 주소 레지스터 심볼
          result.u64 = get_reg(name).u64 + op.get_addr_offset();  // [한국어] 레지스터 값 + 즉시 오프셋 = 유효 주소
        } else if (info.is_param_kernel()) {  // [한국어] 커널 파라미터 메모리 — .param 공간에 있는 커널 인수
          result.u64 = sym->get_address() + op.get_addr_offset();  // [한국어] 파라미터 기저 주소 + 오프셋
        } else if (info.is_param_local()) {  // [한국어] 로컬 파라미터 — 함수 내부 .param (callseq에서 전달된 인수)
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (info.is_global()) {  // [한국어] 전역 메모리 심볼 주소 — ld.global의 경우 오프셋 없이 심볼 주소 그대로
          assert(op.get_addr_offset() == 0);  // [한국어] 전역 변수는 오프셋 없이 직접 주소 사용 — 다른 경우면 버그
          result.u64 = sym->get_address();  // [한국어] 전역 변수의 시뮬레이터 내 절대 주소
        } else if (info.is_local()) {  // [한국어] 로컬 메모리 변수 — 스레드 전용 스택 공간
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (info.is_const()) {  // [한국어] 상수 메모리 변수 — .const 공간
          result.u64 = sym->get_address() + op.get_addr_offset();
        } else if (op.is_shared()) {  // [한국어] 공유 메모리 — CTA(스레드 블록) 내 공유 .shared 변수
          result.u64 = op.get_symbol()->get_address() + op.get_addr_offset();  // [한국어] 공유 메모리 기저 + 오프셋
        } else if (op.is_sstarr()) {  // [한국어] 공유 메모리 배열 (sstarr = shared static array)
          result.u64 = op.get_symbol()->get_address() + op.get_addr_offset();
        } else {  // [한국어] 알 수 없는 메모리 피연산자 타입 — 지원되지 않는 경우
          const char *name = op.name().c_str();  // [한국어] 오류 메시지용 피연산자 이름
          printf(
              "GPGPU-Sim PTX: ERROR ** get_operand_value : unknown memory "
              "operand type for %s\n",
              name);
          abort();  // [한국어] 복구 불가 오류: 즉시 시뮬레이터 종료
        }

      } else if (op.is_literal()) {  // [한국어] 리터럴 상수 피연산자 (예: 0x1234, 3.14f)
        result = op.get_literal_value();  // [한국어] 파싱된 리터럴 값을 ptx_reg_t로 직접 반환
      } else if (op.is_label()) {  // [한국어] 레이블 주소 피연산자 — bra/call의 분기 목적지
        result.u64 = op.get_symbol()->get_address();  // [한국어] 레이블 심볼의 시뮬레이터 내 PC 주소
      } else if (op.is_shared()) {  // [한국어] 공유 메모리 변수 — non-memory_operand 형태의 공유 변수 참조
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_sstarr()) {  // [한국어] 공유 메모리 배열 non-deref 참조
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_const()) {  // [한국어] 상수 메모리 변수 non-deref 참조
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_global()) {  // [한국어] 전역 메모리 변수 non-deref 참조 (주소만 반환)
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_local()) {  // [한국어] 로컬 메모리 변수 non-deref 참조
        result.u64 = op.get_symbol()->get_address();
      } else if (op.is_function_address()) {  // [한국어] 함수 포인터 주소 (CDP 커널 launch, 간접 call 등)
        result.u64 = (size_t)op.get_symbol()->get_pc();  // [한국어] 심볼이 가리키는 PTX 함수의 PC(프로그램 카운터) 주소
      } else if (op.is_param_kernel()) {  // [한국어] 커널 파라미터 심볼 non-deref 참조
        result.u64 = op.get_symbol()->get_address();
      } else {  // [한국어] 위 모든 경우에 해당하지 않는 피연산자 — 최후 분기
        const char *name = op.name().c_str();  // [한국어] 오류 메시지용 피연산자 이름
        const symbol *sym2 = op.get_symbol();  // [한국어] 심볼 포인터로 타입 재조회
        const type_info *type2 = sym2->type();
        const type_info_key &info2 = type2->get_key();
        if (info2.is_param_kernel()) {  // [한국어] 심볼 타입이 커널 파라미터인 경우 — offset 포함 주소 반환
          result.u64 = sym2->get_address() + op.get_addr_offset();
        } else {  // [한국어] 완전히 알 수 없는 피연산자 — 구현 누락 또는 잘못된 PTX
          printf(
              "GPGPU-Sim PTX: ERROR ** get_operand_value : unknown operand "
              "type for %s\n",
              name);
          assert(0);  // [한국어] 구현 버그: assert로 즉시 종료
        }
      }

      if (op.get_operand_lohi() == 1)  // [한국어] lo 수식자: 하위 16비트만 추출
        result.u64 = result.u64 & 0xFFFF;  // [한국어] 상위 비트 마스크 제거 — PTXPlus의 .lo 접미사
      else if (op.get_operand_lohi() == 2)  // [한국어] hi 수식자: 상위 16비트를 하위로 이동
        result.u64 = (result.u64 >> 16) & 0xFFFF;  // [한국어] 16비트 오른쪽 시프트 후 마스크 — PTXPlus의 .hi 접미사
    } else if (opType == BB128_TYPE) {
      // b128  // [한국어] 128비트 타입: 4개의 32비트 레지스터를 u128 구조체로 조합
      result.u128.lowest  = get_reg(op.vec_symbol(0)).u32;  // [한국어] vec[0] → u128.lowest (최하위 32비트)
      result.u128.low     = get_reg(op.vec_symbol(1)).u32;  // [한국어] vec[1] → u128.low (하위 32비트)
      result.u128.high    = get_reg(op.vec_symbol(2)).u32;  // [한국어] vec[2] → u128.high (상위 32비트)
      result.u128.highest = get_reg(op.vec_symbol(3)).u32;  // [한국어] vec[3] → u128.highest (최상위 32비트)
    } else {
      // bb64 or ff64  // [한국어] 64비트 타입(BB64/FF64): 인접한 2개 레지스터를 bits.ls/ms로 조합
      result.bits.ls = get_reg(op.vec_symbol(0)).u32;  // [한국어] vec[0] → bits.ls (하위 32비트, least-significant)
      result.bits.ms = get_reg(op.vec_symbol(1)).u32;  // [한국어] vec[1] → bits.ms (상위 32비트, most-significant)
    }
  } else if (op.get_double_operand_type() == 1) {  // [한국어] double 피연산자 타입 1: s[reg1 + reg2] — 주소를 두 레지스터 합산으로 계산
    ptx_reg_t firstHalf, secondHalf;  // [한국어] 두 레지스터 값을 각각 저장할 임시 변수
    firstHalf.u64  = get_reg(op.vec_symbol(0)).u64;  // [한국어] 기저 주소 레지스터 값
    secondHalf.u64 = get_reg(op.vec_symbol(1)).u64;  // [한국어] 오프셋 레지스터 값
    if (op.get_operand_lohi() == 1)  // [한국어] secondHalf에 lo 수식자: 하위 16비트만 오프셋으로 사용
      secondHalf.u64 = secondHalf.u64 & 0xFFFF;
    else if (op.get_operand_lohi() == 2)  // [한국어] secondHalf에 hi 수식자: 상위 16비트를 오프셋으로 사용
      secondHalf.u64 = (secondHalf.u64 >> 16) & 0xFFFF;
    result.u64 = firstHalf.u64 + secondHalf.u64;  // [한국어] 두 레지스터 합산 = 최종 유효 주소
  } else if (op.get_double_operand_type() == 2) {
    // s[reg1 += reg2]
    // reg1 is incremented after value is returned: the value returned is
    // s[reg1]
    // [한국어] 포스트-인크리먼트 이중 레지스터 주소: s[reg1 += reg2]
    // 반환값은 인크리먼트 이전 reg1 값; 이후 reg1 = reg1 + reg2로 갱신
    ptx_reg_t firstHalf, secondHalf;  // [한국어] 두 레지스터 값 임시 저장
    firstHalf.u64  = get_reg(op.vec_symbol(0)).u64;  // [한국어] reg1 현재 값 — 반환할 주소이자 갱신 대상
    secondHalf.u64 = get_reg(op.vec_symbol(1)).u64;  // [한국어] reg2 현재 값 — reg1에 더할 인크리먼트
    if (op.get_operand_lohi() == 1)  // [한국어] reg2의 lo 16비트만 오프셋으로 사용
      secondHalf.u64 = secondHalf.u64 & 0xFFFF;
    else if (op.get_operand_lohi() == 2)  // [한국어] reg2의 hi 16비트만 오프셋으로 사용
      secondHalf.u64 = (secondHalf.u64 >> 16) & 0xFFFF;
    result.u64 = firstHalf.u64;  // [한국어] 인크리먼트 이전 reg1 값을 반환값으로 저장 (포스트 인크리먼트)
    firstHalf.u64 = firstHalf.u64 + secondHalf.u64;  // [한국어] reg1 = reg1 + reg2 (갱신)
    set_reg(op.vec_symbol(0), firstHalf);  // [한국어] 갱신된 reg1 값을 레지스터 파일에 기록
  } else if (op.get_double_operand_type() == 3) {
    // s[reg += immediate]
    // reg is incremented after value is returned: the value returned is s[reg]
    // [한국어] 포스트-인크리먼트 즉시값 주소: s[reg += immediate]
    // 반환값은 인크리먼트 이전 reg 값; 이후 reg = reg + immediate로 갱신
    ptx_reg_t firstHalf;  // [한국어] 레지스터 현재 값 임시 저장
    firstHalf.u64 = get_reg(op.get_symbol()).u64;  // [한국어] 레지스터 현재 값 읽기 (반환할 주소)
    result.u64 = firstHalf.u64;  // [한국어] 인크리먼트 이전 값을 반환값으로 저장
    firstHalf.u64 = firstHalf.u64 + op.get_addr_offset();  // [한국어] reg = reg + immediate (즉시값으로 갱신)
    set_reg(op.get_symbol(), firstHalf);  // [한국어] 갱신된 레지스터 값을 레지스터 파일에 기록
  }

  /* [한국어] 위 분기에서 result에는 유효 메모리 주소(또는 직접 값)가 저장됨.
   * 이제 derefFlag에 따라 실제 메모리에서 값을 읽거나(derefFlag=1),
   * 주소값을 그대로 반환(derefFlag=0)한다. */
  ptx_reg_t finalResult;   // [한국어] 실제로 반환될 최종 값 (메모리 읽기 결과 또는 result 그대로)
  memory_space *mem = NULL; // [한국어] 접근할 메모리 공간 포인터 (전역/공유/로컬/상수)
  size_t size = 0;          // [한국어] 피연산자 타입에 따른 읽기 크기(비트); type_decode()가 설정
  int t = 0;                // [한국어] type_decode() 내부 용도 (부호 여부 등); 이 함수에서는 직접 사용 안 함
  finalResult.u64 = 0;      // [한국어] 초기화 — 부분 읽기(예: 1바이트) 시 상위 비트가 쓰레기값이 되지 않도록

  // complete other cases for reading from memory, such as reading from other
  // const memory
  if ((op.get_addr_space() == global_space) && (derefFlag)) {
    // global memory - g[4], g[$r0]  // [한국어] 전역 메모리 실제 읽기 (ld.global)
    mem = thread->get_global_memory();  // [한국어] 전역 메모리 공간 포인터 획득
    type_info_key::type_decode(opType, size, t);  // [한국어] opType에서 읽기 크기(비트) 추출
    mem->read(result.u32, size / 8, &finalResult.u128);  // [한국어] 전역 메모리 result.u32 주소에서 (size/8)바이트 읽기
    thread->m_last_effective_address = result.u32;  // [한국어] 마지막 유효 주소 기록 (타이밍 모델에서 캐시 히트/미스 판단에 사용)
    thread->m_last_memory_space = global_space;     // [한국어] 마지막 접근 메모리 공간 기록

    if (opType == S16_TYPE || opType == S32_TYPE)  // [한국어] 부호 있는 정수 타입: 좁은 값을 목적지 폭으로 부호 확장
      sign_extend(finalResult, size, dstInfo);
  } else if ((op.get_addr_space() == shared_space) && (derefFlag)) {
    // shared memory - s[4], s[$r0]  // [한국어] 공유 메모리 실제 읽기 (ld.shared)
    mem = thread->m_shared_mem;  // [한국어] 공유 메모리 공간 포인터 (CTA 당 하나)
    type_info_key::type_decode(opType, size, t);
    mem->read(result.u32, size / 8, &finalResult.u128);  // [한국어] 공유 메모리에서 읽기
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = shared_space;

    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else if ((op.get_addr_space() == const_space) && (derefFlag)) {
    // const memory - ce0c1[4], ce0c1[$r0]  // [한국어] 상수 메모리 실제 읽기 (ld.const)
    mem = thread->get_global_memory();  // [한국어] 상수 메모리는 전역 메모리 공간과 동일 물리 공간에서 읽음
    type_info_key::type_decode(opType, size, t);
    mem->read((result.u32 + op.get_const_mem_offset()), size / 8,
              &finalResult.u128);  // [한국어] 상수 메모리 뱅크 오프셋(get_const_mem_offset)을 추가해 정확한 주소 계산
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = const_space;
    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else if ((op.get_addr_space() == local_space) && (derefFlag)) {
    // local memory - l0[4], l0[$r0]  // [한국어] 로컬 메모리 실제 읽기 (ld.local)
    mem = thread->m_local_mem;  // [한국어] 로컬 메모리 공간 포인터 (스레드 전용)
    type_info_key::type_decode(opType, size, t);
    mem->read(result.u32, size / 8, &finalResult.u128);  // [한국어] 로컬 메모리에서 읽기
    thread->m_last_effective_address = result.u32;
    thread->m_last_memory_space = local_space;
    if (opType == S16_TYPE || opType == S32_TYPE)
      sign_extend(finalResult, size, dstInfo);
  } else {  // [한국어] 메모리 접근이 아닌 경우(레지스터/리터럴/주소) 또는 derefFlag=0: result를 그대로 반환
    finalResult = result;
  }

  if ((op.get_operand_neg() == true) && (derefFlag)) {  // [한국어] 부정 수식자(neg)가 있고 역참조 모드: 값에 단항 음수 적용
    switch (opType) {
      // Default to f32 for now, need to add support for others
      // [한국어] 타입별로 부호 변환: s8/s16/s32/s64 범위 유지, float은 부호 비트 반전
      case S8_TYPE:
      case U8_TYPE:
      case B8_TYPE:
        finalResult.s8 = -finalResult.s8;  // [한국어] 8비트 부호 반전
        break;
      case S16_TYPE:
      case U16_TYPE:
      case B16_TYPE:
        finalResult.s16 = -finalResult.s16;  // [한국어] 16비트 부호 반전
        break;
      case S32_TYPE:
      case U32_TYPE:
      case B32_TYPE:
        finalResult.s32 = -finalResult.s32;  // [한국어] 32비트 부호 반전
        break;
      case S64_TYPE:
      case U64_TYPE:
      case B64_TYPE:
        finalResult.s64 = -finalResult.s64;  // [한국어] 64비트 부호 반전
        break;
      case F16_TYPE:
        finalResult.f16 = -finalResult.f16;  // [한국어] FP16 부호 비트 반전
        break;
      case F32_TYPE:
        finalResult.f32 = -finalResult.f32;  // [한국어] FP32 부호 비트 반전
        break;
      case F64_TYPE:
      case FF64_TYPE:
        finalResult.f64 = -finalResult.f64;  // [한국어] FP64 부호 비트 반전
        break;
      default:
        assert(0);  // [한국어] neg 수식자를 지원하지 않는 타입 — 구현 버그
    }
  }

  return finalResult;  // [한국어] 최종 피연산자 값 반환 (메모리 읽기 결과 또는 레지스터/즉시값)
}

/*
 * [한국어]
 * get_operand_nbits - 레지스터 피연산자의 비트 폭을 반환
 *
 * @op    : 비트 폭을 알고 싶은 피연산자 (operand_info); 현재 레지스터 타입만 지원
 * @return: 피연산자의 비트 폭 (1/8/16/32/64); 미지원 타입은 abort
 *
 * sign_extend()가 목적지 레지스터의 폭을 알아야 부호 확장 범위를 결정하기 위해 사용한다.
 * 레지스터 심볼의 타입 정보(type_info_key)에서 스칼라 타입을 추출하고
 * PTX 표준에 따른 비트 폭을 반환한다.
 * 현재는 레지스터 피연산자(op.is_reg())만 구현되어 있고,
 * 다른 피연산자 타입(메모리, 리터럴 등)은 미지원 오류를 출력하고 abort한다.
 *
 * 실행 컨텍스트: sign_extend()에서 호출; 기능 시뮬레이터 단일 스레드 루프.
 *
 * 호출 체인:
 *   get_operand_value() / set_operand_value() → sign_extend() → [get_operand_nbits]
 */
unsigned get_operand_nbits(const operand_info &op) {
  if (op.is_reg()) {  // [한국어] 레지스터 피연산자인 경우만 구현됨
    const symbol *sym = op.get_symbol();  // [한국어] 레지스터 심볼 포인터 획득
    const type_info *typ = sym->type();   // [한국어] 심볼의 타입 정보 객체 획득
    type_info_key t = typ->get_key();     // [한국어] 타입 키 — 스칼라 타입 enum 접근에 사용
    switch (t.scalar_type()) {  // [한국어] PTX 스칼라 타입에 따라 비트 폭 반환
      case PRED_TYPE:  // [한국어] 조건자(predicate) 레지스터: 1비트
        return 1;
      case B8_TYPE:
      case S8_TYPE:
      case U8_TYPE:    // [한국어] 8비트 타입 (비트/부호/무부호)
        return 8;
      case S16_TYPE:
      case U16_TYPE:
      case F16_TYPE:
      case B16_TYPE:   // [한국어] 16비트 타입 (부호/무부호/FP16/비트)
        return 16;
      case S32_TYPE:
      case U32_TYPE:
      case F32_TYPE:
      case B32_TYPE:   // [한국어] 32비트 타입 (부호/무부호/FP32/비트)
        return 32;
      case S64_TYPE:
      case U64_TYPE:
      case F64_TYPE:
      case B64_TYPE:   // [한국어] 64비트 타입 (부호/무부호/FP64/비트)
        return 64;
      default:
        printf("ERROR: unknown register type\n");  // [한국어] 알 수 없는 타입: 오류 출력
        fflush(stdout);  // [한국어] 버퍼 즉시 출력 — abort 전 메시지 보장
        abort();  // [한국어] 알 수 없는 레지스터 타입: 시뮬레이터 즉시 종료
    }
  } else {  // [한국어] 레지스터가 아닌 피연산자(메모리, 리터럴 등) — 미구현
    printf(
        "ERROR: Need to implement get_operand_nbits() for currently "
        "unsupported operand_info type\n");  // [한국어] 미구현 피연산자 타입 오류 출력
    fflush(stdout);  // [한국어] abort 전 버퍼 플러시
    abort();  // [한국어] 미구현 경로: 즉시 종료
  }

  return 0;  // [한국어] 도달 불가 코드; 컴파일러 경고 억제용
}

/*
 * [한국어]
 * ptx_thread_info::get_vector_operand_values - 벡터 피연산자에서 여러 레지스터 값을 배열로 읽기
 *
 * @op          : 벡터 피연산자 (operand_info); 내부적으로 num_elements개의 스칼라 레지스터 심볼을 가짐
 * @ptx_regs    : 읽은 레지스터 값을 저장할 배열; 호출자가 할당 (크기 >= num_elements)
 * @num_elements: 읽을 레지스터 개수 (2~8); PTX ld.v2/v4/v8 명령어 등에서 결정됨
 * @return      : void; 결과는 ptx_regs 배열에 저장
 *
 * PTX의 벡터 로드 명령어(ld.v2, ld.v4 등)에서 여러 목적지 레지스터를 한 번에 읽을 때 사용.
 * 역방향(num_elements-1 → 0)으로 순회하여 인덱스 순서를 맞춘다.
 * "_"(discard) 레지스터는 건너뛰어 읽지 않는다.
 * 벡터 피연산자는 반드시 op.is_vector()가 참이어야 하며, 원소 수는 8 이하로 제한된다.
 *
 * 실행 컨텍스트: ld.v2/v4/v8 impl 함수에서 호출; 기능 시뮬레이터 단일 스레드 루프.
 *
 * 호출 체인:
 *   ld_impl (벡터 로드) → [get_vector_operand_values] → m_regs.back() 다중 조회
 */
void ptx_thread_info::get_vector_operand_values(const operand_info &op,
                                                ptx_reg_t *ptx_regs,
                                                unsigned num_elements) {
  assert(op.is_vector());      // [한국어] 벡터 피연산자 타입 확인 — 스칼라 피연산자로 호출 시 버그
  assert(num_elements <= 8);   // [한국어] PTX 벡터 피연산자는 최대 8원소 (.v2/.v4/.v8 등)

  for (int idx = num_elements - 1; idx >= 0; --idx) {  // [한국어] 역방향 순회로 모든 원소 인덱스 처리
    const symbol *sym = NULL;            // [한국어] 현재 인덱스의 벡터 원소 레지스터 심볼
    sym = op.vec_symbol(idx);            // [한국어] 벡터 피연산자에서 idx번째 스칼라 레지스터 심볼 획득
    if (strcmp(sym->name().c_str(), "_") != 0) {  // [한국어] "_"(discard)가 아닌 경우만 읽기
      reg_map_t::iterator reg_iter = m_regs.back().find(sym);  // [한국어] 레지스터 파일에서 심볼 조회
      assert(reg_iter != m_regs.back().end());  // [한국어] 레지스터가 존재해야 함 — 미초기화 벡터 원소는 버그
      ptx_regs[idx] = reg_iter->second;          // [한국어] 해당 인덱스에 레지스터 값 저장
    }
  }
}

/*
 * [한국어]
 * sign_extend - 부호 있는 좁은 정수값을 목적지 레지스터 폭으로 부호 확장
 *
 * @data    : 확장할 값 (ptx_reg_t, 참조); 함수 내에서 in-place 수정됨
 * @src_size: 소스 값의 비트 폭 (예: S16 → 16, S32 → 32)
 * @dst     : 목적지 피연산자; 레지스터인 경우 get_operand_nbits()로 목적지 폭 조회
 * @return  : void; data가 in-place로 수정됨
 *
 * PTX ld/cvt 명령어에서 좁은 부호 정수(예: S16)를 넓은 목적지(예: S32)에 로드할 때
 * 부호 비트를 상위 비트로 복사해야 올바른 음수 표현이 유지된다.
 * 알고리즘:
 *   1) 목적지가 레지스터가 아니거나 src_size >= dst_size이면 확장 불필요 → 즉시 반환
 *   2) 소스 부호 비트(bit src_size-1) 검사: 0이면 양수 → 확장 불필요
 *   3) 음수인 경우: dst_size-src_size 크기의 마스크를 만들어 src_size 비트 위치부터
 *      data.u64에 OR-설정 → 상위 비트를 모두 1로 채움
 *
 * 실행 컨텍스트: get_operand_value()의 메모리 읽기 경로에서 S16/S32 타입에 대해 호출.
 *
 * 호출 체인:
 *   get_operand_value() → [sign_extend] → get_operand_nbits()
 */
void sign_extend(ptx_reg_t &data, unsigned src_size, const operand_info &dst) {
  if (!dst.is_reg()) return;  // [한국어] 목적지가 레지스터가 아니면 부호 확장 불필요 → 즉시 반환
  unsigned dst_size = get_operand_nbits(dst);  // [한국어] 목적지 레지스터의 비트 폭 조회
  if (src_size >= dst_size) return;  // [한국어] 소스가 목적지보다 같거나 크면 확장 불필요
  // src_size < dst_size  // [한국어] 이 지점: 소스가 목적지보다 좁음 → 확장 필요 가능성
  unsigned long long mask = 1;  // [한국어] 부호 비트 검사용 마스크 생성 시작
  mask <<= (src_size - 1);  // [한국어] mask = 1 << (src_size-1): 소스의 최상위 비트(부호 비트) 위치
  if ((mask & data.u64) == 0) {  // [한국어] 부호 비트가 0 = 양수: 확장해도 상위 비트는 0 → 불필요
    // no need to sign extend
    return;
  }
  // need to sign extend  // [한국어] 부호 비트가 1 = 음수: 상위 비트를 모두 1로 설정해야 함
  mask = 1;  // [한국어] 확장 마스크 새로 계산 시작
  mask <<= dst_size - src_size;  // [한국어] 확장할 비트 수만큼 시프트 (dst-src 개의 비트)
  mask -= 1;   // [한국어] 연속된 1 비트 생성: 예) 4비트 확장이면 0b1111
  mask <<= src_size;  // [한국어] 소스 비트 위치 위로 이동: 상위 dst-src 비트 위치에 마스크 배치
  data.u64 |= mask;  // [한국어] 마스크를 OR-설정: 상위 비트를 모두 1로 채워 부호 확장 완료
}

/*
 * [한국어]
 * ptx_thread_info::set_operand_value (carry/overflow 오버로드) -
 *   결과값을 기록하고 carry/overflow 플래그를 predicate 레지스터에 저장
 *
 * @dst     : 목적지 피연산자; double 타입 -2이면 predicate|result 쌍
 * @data    : 기록할 결과값 (ptx_reg_t)
 * @type    : 결과 데이터 타입 (S32_TYPE, U32_TYPE 등)
 * @thread  : 현재 실행 중인 시뮬레이트 스레드
 * @pI      : 현재 실행 중인 PTX 명령어 객체
 * @overflow: 오버플로 발생 여부 (1=발생, 0=미발생); 비트 3에 저장
 * @carry   : 캐리 발생 여부 (1=발생, 0=미발생); 비트 2에 저장
 * @return  : void
 *
 * PTXPlus의 carry/overflow 추적 기능을 위한 오버로드.
 * 먼저 일반 set_operand_value(5인수)를 호출해 결과값을 쓰고,
 * 이중 목적지(-2 타입: pred|reg)인 경우 predicate 레지스터의
 * 비트 2(carry)와 비트 3(overflow)을 갱신한다.
 * PTX 스펙: addc, subc, addp 등 carry-chain 명령어에서 사용.
 * predValue.u64 비트 레이아웃:
 *   bit 0: zero flag (기존 값 유지)
 *   bit 1: negative/sign flag (기존 값 유지)
 *   bit 2: carry flag (이 함수에서 갱신)
 *   bit 3: overflow flag (이 함수에서 갱신)
 *
 * 실행 컨텍스트: add_impl/addp_impl/sub_impl 등 산술 명령어 구현에서 호출.
 *
 * 호출 체인:
 *   add_impl / addp_impl → [set_operand_value(overflow,carry)] → set_operand_value(no-flags) + set_reg
 */
void ptx_thread_info::set_operand_value(const operand_info &dst,
                                        const ptx_reg_t &data, unsigned type,
                                        ptx_thread_info *thread,
                                        const ptx_instruction *pI, int overflow,
                                        int carry) {
  thread->set_operand_value(dst, data, type, thread, pI);  // [한국어] 먼저 일반 경로로 결과값 기록 (레지스터 또는 메모리)

  if (dst.get_double_operand_type() == -2) {  // [한국어] 이중 목적지 타입 -2: pred|reg — predicate도 갱신 필요
    ptx_reg_t predValue;  // [한국어] 갱신할 predicate 레지스터 값

    const symbol *sym = dst.vec_symbol(0);  // [한국어] 이중 목적지에서 첫 번째 = predicate 레지스터 심볼
    predValue.u64 = (m_regs.back()[sym].u64) & ~(0x0C);  // [한국어] 기존 predicate 값에서 비트 2,3(carry/overflow) 클리어; 나머지(bit 0,1) 보존
    predValue.u64 |= ((overflow & 0x01) << 3);  // [한국어] overflow 비트(1비트) → pred 비트 3에 설정
    predValue.u64 |= ((carry & 0x01) << 2);     // [한국어] carry 비트(1비트) → pred 비트 2에 설정

    set_reg(sym, predValue);  // [한국어] 갱신된 predicate 값을 레지스터 파일에 기록
  } else if (dst.get_double_operand_type() == 0) {  // [한국어] 단일 목적지 (일반 레지스터만): predicate 갱신 불필요
    // intentionally do nothing  // [한국어] 이미 위에서 set_operand_value로 기록 완료
  } else {  // [한국어] 예상치 못한 이중 목적지 타입 — 구현 버그
    printf("Unexpected double destination\n");
    assert(0);
  }
}

/*
 * [한국어]
 * ptx_thread_info::set_operand_value (5인수 기본 오버로드) -
 *   결과값을 목적지 피연산자(레지스터 또는 메모리)에 기록하는 핵심 디스패처
 *
 * @dst   : 목적지 피연산자; 레지스터, 전역/공유/로컬 메모리, 이중 목적지 등 다양한 형태
 * @data  : 기록할 결과값 (ptx_reg_t)
 * @type  : 결과 데이터 타입 (S8/U32/F32/BB128/BB64/FF64 등)
 * @thread: 현재 실행 중인 시뮬레이트 스레드 (메모리 공간 접근에 사용)
 * @pI    : 현재 실행 중인 PTX 명령어 (메모리 쓰기 시 타이밍 모델에 전달)
 * @return: void
 *
 * PTX 명령어 실행 결과를 목적지에 저장하는 모든 경우를 처리하는 핵심 함수:
 *   - undefined_space (레지스터 공간):
 *       * 이중 목적지 -1: p|p 쌍 (set 명령어; 두 번째는 첫 번째의 부정)
 *       * 이중 목적지 -2/-3: pred|reg 쌍 (cvt/mul 등; pred=result!=0)
 *       * BB128_TYPE: 4개 레지스터에 분산 저장
 *       * BB64_TYPE/FF64_TYPE: 2개 레지스터에 분산 저장
 *       * lo/hi 16비트 부분 기록
 *       * 단일 레지스터 기록 (일반 경우)
 *   - global_space: 전역 메모리 쓰기
 *   - shared_space: 공유 메모리 쓰기
 *   - local_space: 로컬 메모리 쓰기
 *
 * 실행 컨텍스트: 기능 시뮬레이터 단일 스레드 루프; xxx_impl()에서 직접 호출.
 * 에러 경로: 알 수 없는 메모리 공간 → printf + assert(0).
 *
 * 호출 체인:
 *   xxx_impl() → [set_operand_value] → set_reg() / memory_space::write()
 */
void ptx_thread_info::set_operand_value(const operand_info &dst,
                                        const ptx_reg_t &data, unsigned type,
                                        ptx_thread_info *thread,
                                        const ptx_instruction *pI) {
  ptx_reg_t dstData;        // [한국어] 메모리 쓰기 시 목적지 주소를 임시 저장
  memory_space *mem = NULL; // [한국어] 접근할 메모리 공간 포인터 (전역/공유/로컬)
  size_t size;              // [한국어] 쓰기 크기(비트); type_decode()로 설정
  int t;                    // [한국어] type_decode() 내부 용도 (부호 여부)

  type_info_key::type_decode(type, size, t);  // [한국어] type에서 크기(비트)와 부호 여부 추출

  /*complete this section for other cases*/
  if (dst.get_addr_space() == undefined_space) {  // [한국어] 레지스터 공간 (undefined_space = 레지스터 파일)
    ptx_reg_t setValue;    // [한국어] 실제로 레지스터에 기록할 값
    setValue.u64 = data.u64;  // [한국어] 결과값을 setValue에 복사 (이후 lo/hi 처리 등으로 수정 가능)

    // Double destination in set instruction ($p0|$p1) - second is negation of
    // first
    if (dst.get_double_operand_type() == -1) {  // [한국어] 이중 목적지 -1: set 명령어의 $p0|$p1 패턴
      ptx_reg_t setValue2;              // [한국어] 두 번째 목적지에 저장할 값 (첫 번째의 논리 부정)
      const symbol *name1 = dst.vec_symbol(0);  // [한국어] 첫 번째 predicate 레지스터 심볼
      const symbol *name2 = dst.vec_symbol(1);  // [한국어] 두 번째 predicate 레지스터 심볼 (부정값)

      if ((type == F16_TYPE) || (type == F32_TYPE) || (type == F64_TYPE) ||
          (type == FF64_TYPE)) {  // [한국어] 부동소수점 타입: 0이면 1.0f(참), 아니면 0.0f(거짓)
        setValue2.f32 = (setValue.u64 == 0) ? 1.0f : 0.0f;  // [한국어] 첫 번째가 false(0)이면 두 번째는 true(1.0f)
      } else {  // [한국어] 정수 타입: 0이면 0xFFFFFFFF(모든 비트 참), 아니면 0
        setValue2.u32 = (setValue.u64 == 0) ? 0xFFFFFFFF : 0;  // [한국어] 비트 패턴으로 true/false 표현
      }

      set_reg(name1, setValue);   // [한국어] 첫 번째 predicate에 원래 값 저장
      set_reg(name2, setValue2);  // [한국어] 두 번째 predicate에 부정값 저장
    }

    // Double destination in cvt,shr,mul,etc. instruction ($p0|$r4) - second
    // register operand receives data, first predicate operand is set as
    // $p0=($r4!=0) Also for Double destination in set instruction ($p0/$r1)
    else if ((dst.get_double_operand_type() == -2) ||
             (dst.get_double_operand_type() == -3)) {
      // [한국어] 이중 목적지 -2/-3: $p0|$r4 또는 $p0/$r1 패턴
      // pred 레지스터($p0)는 결과가 0인지/음수인지로 설정; reg($r4)는 결과값 그대로 저장
      ptx_reg_t predValue;               // [한국어] 첫 번째(predicate) 목적지 값
      const symbol *predName = dst.vec_symbol(0);  // [한국어] predicate 레지스터 심볼
      const symbol *regName  = dst.vec_symbol(1);  // [한국어] 일반 레지스터 심볼 (결과값 저장)
      predValue.u64 = 0;  // [한국어] predicate 초기화

      /* [한국어] zero flag(bit 0) 설정: 결과값이 부호 비트 제외 0인 경우
       * 부호 있는 타입(Sxx)은 부호 비트를 제외한 하위 비트들이 0인지 검사 */
      switch (type) {
        case S8_TYPE:
          if ((setValue.s8 & 0x7F) == 0) predValue.u64 |= 1;  // [한국어] 부호 제외 7비트가 0이면 zero flag
          break;
        case S16_TYPE:
          if ((setValue.s16 & 0x7FFF) == 0) predValue.u64 |= 1;  // [한국어] 부호 제외 15비트가 0
          break;
        case S32_TYPE:
          if ((setValue.s32 & 0x7FFFFFFF) == 0) predValue.u64 |= 1;  // [한국어] 부호 제외 31비트가 0
          break;
        case S64_TYPE:
          if ((setValue.s64 & 0x7FFFFFFFFFFFFFFF) == 0) predValue.u64 |= 1;  // [한국어] 부호 제외 63비트가 0
          break;
        case U8_TYPE:
        case B8_TYPE:
          if (setValue.u8 == 0) predValue.u64 |= 1;  // [한국어] 무부호 8비트가 0이면 zero flag
          break;
        case U16_TYPE:
        case B16_TYPE:
          if (setValue.u16 == 0) predValue.u64 |= 1;  // [한국어] 무부호 16비트가 0
          break;
        case U32_TYPE:
        case B32_TYPE:
          if (setValue.u32 == 0) predValue.u64 |= 1;  // [한국어] 무부호 32비트가 0
          break;
        case U64_TYPE:
        case B64_TYPE:
          if (setValue.u64 == 0) predValue.u64 |= 1;  // [한국어] 무부호 64비트가 0
          break;
        case F16_TYPE:
          if (setValue.f16 == 0) predValue.u64 |= 1;  // [한국어] FP16이 0이면 zero flag
          break;
        case F32_TYPE:
          if (setValue.f32 == 0) predValue.u64 |= 1;  // [한국어] FP32가 0이면 zero flag
          break;
        case F64_TYPE:
        case FF64_TYPE:
          if (setValue.f64 == 0) predValue.u64 |= 1;  // [한국어] FP64가 0이면 zero flag
          break;
        default:
          assert(0);  // [한국어] 지원하지 않는 타입 — 구현 버그
          break;
      }

      /* [한국어] negative/sign flag(bit 1) 설정:
       * 정수 타입: size-1 번째 비트(부호 비트)가 1이면 음수
       * FP32: 음수이면 1 */
      if ((type == S8_TYPE) || (type == S16_TYPE) || (type == S32_TYPE) ||
          (type == S64_TYPE) || (type == U8_TYPE) || (type == U16_TYPE) ||
          (type == U32_TYPE) || (type == U64_TYPE) || (type == B8_TYPE) ||
          (type == B16_TYPE) || (type == B32_TYPE) || (type == B64_TYPE)) {
        if ((setValue.u32 & (1 << (size - 1))) != 0) predValue.u64 |= 1 << 1;  // [한국어] 부호(MSB)가 1이면 negative flag(bit 1) 설정
      }
      if (type == F32_TYPE) {
        if (setValue.f32 < 0) predValue.u64 |= 1 << 1;  // [한국어] FP32 음수 → negative flag 설정
      }

      /* [한국어] lo/hi 16비트 부분 쓰기 처리:
       * lo(1): 기존 레지스터 상위 비트 유지, 하위 16비트만 갱신
       * hi(2): 기존 레지스터 하위 비트 유지, 상위 16비트만 갱신 */
      if (dst.get_operand_lohi() == 1) {  // [한국어] lo 수식자: 하위 16비트에만 data.u64 기록
        setValue.u64 =
            ((m_regs.back()[regName].u64) & (~(0xFFFF))) + (data.u64 & 0xFFFF);  // [한국어] 기존 상위 비트 유지 + 새 하위 16비트
      } else if (dst.get_operand_lohi() == 2) {  // [한국어] hi 수식자: 상위 16비트에만 data.u64 기록
        setValue.u64 = ((m_regs.back()[regName].u64) & (~(0xFFFF0000))) +
                       ((data.u64 << 16) & 0xFFFF0000);  // [한국어] 기존 하위 비트 유지 + data를 16비트 시프트해 상위 기록
      }

      set_reg(predName, predValue);  // [한국어] predicate 레지스터에 zero/negative flag 기록
      set_reg(regName, setValue);    // [한국어] 일반 레지스터에 결과값(또는 lo/hi 갱신값) 기록
    } else if (type == BB128_TYPE) {
      // b128 stuff here.  // [한국어] BB128_TYPE: 128비트를 4개 32비트 레지스터에 분산 저장
      ptx_reg_t setValue2, setValue3, setValue4;  // [한국어] 4개 레지스터에 저장할 값들
      setValue.u64 = 0;   // [한국어] setValue 초기화 (상위 32비트 클리어)
      setValue2.u64 = 0;
      setValue3.u64 = 0;
      setValue4.u64 = 0;
      setValue.u32  = data.u128.lowest;   // [한국어] u128.lowest → 첫 번째 레지스터 (최하위 32비트)
      setValue2.u32 = data.u128.low;      // [한국어] u128.low → 두 번째 레지스터
      setValue3.u32 = data.u128.high;     // [한국어] u128.high → 세 번째 레지스터
      setValue4.u32 = data.u128.highest;  // [한국어] u128.highest → 네 번째 레지스터 (최상위 32비트)

      const symbol *name1, *name2, *name3, *name4 = NULL;  // [한국어] 4개 레지스터 심볼 포인터

      name1 = dst.vec_symbol(0);  // [한국어] 벡터 목적지에서 각 인덱스의 스칼라 레지스터 심볼 획득
      name2 = dst.vec_symbol(1);
      name3 = dst.vec_symbol(2);
      name4 = dst.vec_symbol(3);

      set_reg(name1, setValue);   // [한국어] 최하위 32비트 → 첫 번째 레지스터에 기록
      set_reg(name2, setValue2);  // [한국어] 두 번째 32비트 → 두 번째 레지스터
      set_reg(name3, setValue3);  // [한국어] 세 번째 32비트 → 세 번째 레지스터
      set_reg(name4, setValue4);  // [한국어] 최상위 32비트 → 네 번째 레지스터
    } else if (type == BB64_TYPE || type == FF64_TYPE) {
      // ptxplus version of storing 64 bit values to registers stores to two
      // adjacent registers
      // [한국어] BB64/FF64: 64비트를 2개 인접 32비트 레지스터에 분산 저장 (PTXPlus 방식)
      ptx_reg_t setValue2;  // [한국어] 두 번째 레지스터에 저장할 상위 32비트
      setValue.u32 = 0;   // [한국어] 상위 비트 클리어
      setValue2.u32 = 0;
      setValue.u32  = data.bits.ls;  // [한국어] bits.ls(least-significant) → 첫 번째 레지스터 (하위 32비트)
      setValue2.u32 = data.bits.ms;  // [한국어] bits.ms(most-significant) → 두 번째 레지스터 (상위 32비트)

      const symbol *name1, *name2 = NULL;  // [한국어] 2개 레지스터 심볼 포인터

      name1 = dst.vec_symbol(0);  // [한국어] 첫 번째(하위) 레지스터 심볼
      name2 = dst.vec_symbol(1);  // [한국어] 두 번째(상위) 레지스터 심볼

      set_reg(name1, setValue);   // [한국어] 하위 32비트 기록
      set_reg(name2, setValue2);  // [한국어] 상위 32비트 기록
    } else {  // [한국어] 단일 레지스터 기록 (BB128/BB64/FF64가 아닌 일반 타입)
      if (dst.get_operand_lohi() == 1) {  // [한국어] lo 수식자: 기존 레지스터 상위 비트 유지, 하위 16비트만 갱신
        setValue.u64 = ((m_regs.back()[dst.get_symbol()].u64) & (~(0xFFFF))) +
                       (data.u64 & 0xFFFF);  // [한국어] 기존 상위 비트 마스크 + 새 하위 16비트
      } else if (dst.get_operand_lohi() == 2) {  // [한국어] hi 수식자: 하위 비트 유지, 상위 16비트만 갱신
        setValue.u64 =
            ((m_regs.back()[dst.get_symbol()].u64) & (~(0xFFFF0000))) +
            ((data.u64 << 16) & 0xFFFF0000);  // [한국어] 기존 하위 비트 + data를 16비트 시프트한 상위 16비트
      }
      set_reg(dst.get_symbol(), setValue);  // [한국어] 단일 레지스터에 최종값 기록
    }
  }

  // global memory - g[4], g[$r0]
  else if (dst.get_addr_space() == global_space) {  // [한국어] 전역 메모리 쓰기 (st.global)
    dstData = thread->get_operand_value(dst, dst, type, thread, 0);  // [한국어] derefFlag=0: 주소만 계산 (역참조 안 함)
    mem = thread->get_global_memory();  // [한국어] 전역 메모리 공간 포인터 획득
    type_info_key::type_decode(type, size, t);  // [한국어] 쓰기 크기(비트) 추출

    mem->write(dstData.u32, size / 8, &data.u128, thread, pI);  // [한국어] dstData.u32 주소에 (size/8)바이트 쓰기
    thread->m_last_effective_address = dstData.u32;  // [한국어] 마지막 유효 주소 기록 (타이밍 모델 캐시 분석용)
    thread->m_last_memory_space = global_space;      // [한국어] 마지막 접근 메모리 공간 기록
  }

  // shared memory - s[4], s[$r0]
  else if (dst.get_addr_space() == shared_space) {  // [한국어] 공유 메모리 쓰기 (st.shared)
    dstData = thread->get_operand_value(dst, dst, type, thread, 0);  // [한국어] 공유 메모리 주소 계산
    mem = thread->m_shared_mem;  // [한국어] 공유 메모리 공간 포인터
    type_info_key::type_decode(type, size, t);

    mem->write(dstData.u32, size / 8, &data.u128, thread, pI);  // [한국어] 공유 메모리에 쓰기
    thread->m_last_effective_address = dstData.u32;
    thread->m_last_memory_space = shared_space;
  }

  // local memory - l0[4], l0[$r0]
  else if (dst.get_addr_space() == local_space) {  // [한국어] 로컬 메모리 쓰기 (st.local)
    dstData = thread->get_operand_value(dst, dst, type, thread, 0);  // [한국어] 로컬 메모리 주소 계산
    mem = thread->m_local_mem;  // [한국어] 로컬 메모리 공간 포인터 (스레드 전용)
    type_info_key::type_decode(type, size, t);

    mem->write(dstData.u32, size / 8, &data.u128, thread, pI);  // [한국어] 로컬 메모리에 쓰기
    thread->m_last_effective_address = dstData.u32;
    thread->m_last_memory_space = local_space;
  }

  else {  // [한국어] 알 수 없는 메모리 공간 — 구현 누락 또는 잘못된 PTX
    printf("Destination stores to unknown location.");  // [한국어] 알 수 없는 목적지 공간 오류 출력
    assert(0);  // [한국어] 복구 불가 오류: 즉시 종료
  }
}

/*
 * [한국어]
 * ptx_thread_info::set_vector_operand_values - 벡터 목적지에 최대 4개 스칼라 값을 기록
 *
 * @dst  : 벡터 목적지 피연산자; .v2/.v4 벡터 레지스터 그룹
 * @data1~data4: 각 벡터 원소에 기록할 값; num_elements보다 많은 인수는 무시됨
 * @return: void
 *
 * PTX 벡터 스토어(st.v2, st.v4) 또는 벡터 연산 결과 저장에 사용.
 * dst.get_vect_nelem()으로 실제 원소 수를 결정하고, 그 수만큼만 set_reg를 호출한다.
 * m_last_set_operand_value는 data1로 설정 (벡터의 첫 원소를 마지막 기록값으로 간주).
 *
 * 실행 컨텍스트: st.v2/v4 impl, 또는 wmma 이외의 벡터 결과 저장에서 호출.
 *
 * 호출 체인:
 *   st_impl (벡터 스토어) → [set_vector_operand_values] → set_reg() × N
 */
void ptx_thread_info::set_vector_operand_values(const operand_info &dst,
                                                const ptx_reg_t &data1,
                                                const ptx_reg_t &data2,
                                                const ptx_reg_t &data3,
                                                const ptx_reg_t &data4) {
  unsigned num_elements = dst.get_vect_nelem();  // [한국어] 목적지 벡터의 실제 원소 수 (.v2=2, .v4=4 등)
  if (num_elements > 0) {  // [한국어] 최소 1개 원소가 있는 경우만 기록
    set_reg(dst.vec_symbol(0), data1);  // [한국어] 첫 번째 원소 레지스터에 data1 기록
    if (num_elements > 1) {  // [한국어] 원소 2개 이상: data2도 기록
      set_reg(dst.vec_symbol(1), data2);
      if (num_elements > 2) {  // [한국어] 원소 3개 이상: data3도 기록
        set_reg(dst.vec_symbol(2), data3);
        if (num_elements > 3) {  // [한국어] 원소 4개: data4도 기록 (.v4 최대)
          set_reg(dst.vec_symbol(3), data4);
        }
      }
    }
  }

  m_last_set_operand_value = data1;  // [한국어] 벡터의 첫 원소를 마지막 쓰기값으로 캐시
}

/*
 * [한국어]
 * ptx_thread_info::set_wmma_vector_operand_values - wmma.mma 결과를 8개 레지스터에 기록
 *
 * @dst        : 8원소 벡터 목적지 피연산자 (.v8 레지스터 그룹)
 * @data1~data8: wmma.mma 연산 결과의 8개 누산기 원소 값
 * @return     : void
 *
 * wmma.mma 명령어는 Tensor Core 행렬 곱 결과를 8개 레지스터에 저장한다.
 * 이 함수는 정확히 8개 원소인 경우만 처리하며, 그 외는 오류 메시지 출력.
 * set_vector_operand_values()와 달리 원소 수 분기 없이 8개 고정으로 set_reg 호출.
 * m_last_set_operand_value는 마지막 원소 data8로 설정.
 *
 * 실행 컨텍스트: wmma_mma_impl() 내에서 호출; 기능 시뮬레이터 단일 스레드 루프.
 *
 * 호출 체인:
 *   wmma_mma_impl → [set_wmma_vector_operand_values] → set_reg() × 8
 */
void ptx_thread_info::set_wmma_vector_operand_values(
    const operand_info &dst, const ptx_reg_t &data1, const ptx_reg_t &data2,
    const ptx_reg_t &data3, const ptx_reg_t &data4, const ptx_reg_t &data5,
    const ptx_reg_t &data6, const ptx_reg_t &data7, const ptx_reg_t &data8) {
  unsigned num_elements = dst.get_vect_nelem();  // [한국어] 목적지 벡터 원소 수 — wmma.mma에서는 반드시 8이어야 함
  if (num_elements == 8) {  // [한국어] 정확히 8원소인 경우만 정상 처리 (wmma D 행렬)
    set_reg(dst.vec_symbol(0), data1);  // [한국어] 누산기 원소 0 → 레지스터 0
    set_reg(dst.vec_symbol(1), data2);  // [한국어] 누산기 원소 1 → 레지스터 1
    set_reg(dst.vec_symbol(2), data3);
    set_reg(dst.vec_symbol(3), data4);
    set_reg(dst.vec_symbol(4), data5);
    set_reg(dst.vec_symbol(5), data6);
    set_reg(dst.vec_symbol(6), data7);
    set_reg(dst.vec_symbol(7), data8);  // [한국어] 누산기 원소 7 → 레지스터 7
  } else {  // [한국어] 8원소가 아닌 경우 — 구현 버그 또는 잘못된 wmma 피연산자
    printf("error:set_wmma_vector_operands");  // [한국어] 오류 출력 (개행 없이 — 원본 유지)
  }

  m_last_set_operand_value = data8;  // [한국어] 마지막 원소(data8)를 마지막 쓰기값으로 캐시
}

/* [한국어] 명령어 구현에서 반복적으로 사용되는 산술/논리 헬퍼 매크로들.
 * 전처리기 매크로로 정의되어 타입에 무관하게 다형적으로 동작한다.
 * PTX atomic 연산(atom.add, atom.max, atom.cas 등)과 일반 산술 명령어에서 공용으로 사용. */

#define my_abs(a) (((a) < 0) ? (-a) : (a))  // [한국어] 절댓값: 음수면 부호 반전, 양수면 그대로. abs_impl에서 사용

#define MY_MAX_I(a, b) (a > b) ? a : b  // [한국어] 정수 최댓값: NaN 처리 없이 단순 비교
#define MY_MAX_F(a, b) isNaN(a) ? b : isNaN(b) ? a : (a > b) ? a : b  // [한국어] 부동소수점 최댓값: NaN이면 다른 쪽 반환 (IEEE 754 minNum/maxNum 시맨틱)

#define MY_MIN_I(a, b) (a < b) ? a : b  // [한국어] 정수 최솟값: 단순 비교
#define MY_MIN_F(a, b) isNaN(a) ? b : isNaN(b) ? a : (a < b) ? a : b  // [한국어] 부동소수점 최솟값: NaN 처리 포함

#define MY_INC_I(a, b) (a >= b) ? 0 : a + 1  // [한국어] 원형 증가(circular increment): a가 b 이상이면 0으로 wrap-around, 아니면 +1. atom.inc에서 사용
#define MY_DEC_I(a, b) ((a == 0) || (a > b)) ? b : a - 1  // [한국어] 원형 감소: a가 0이거나 b 초과이면 b로 설정, 아니면 -1. atom.dec에서 사용

#define MY_CAS_I(a, b, c) (a == b) ? c : a  // [한국어] CAS(Compare-And-Swap): a==b이면 c 반환, 아니면 a 유지. atom.cas에서 사용

#define MY_EXCH(a, b) b  // [한국어] 교환(exchange): 항상 b 반환 (a는 무시). atom.exch에서 사용; a의 기존값은 호출 측에서 이미 저장됨

/*
 * [한국어]
 * abs_impl - PTX `abs` 명령어 구현 — 절댓값 연산
 *
 * @pI    : 실행할 PTX 명령어 객체; dst, src1, 타입 정보 보유
 * @thread: 현재 실행 중인 시뮬레이트 스레드; 레지스터 파일 접근
 * @return: void; 결과는 dst 레지스터에 기록
 *
 * PTX abs 명령어: d = |a| (절댓값)
 * 지원 타입: S16/S32/S64 (부호 있는 정수), U16/U32/U64 (무부호 정수), F32/F64/FF64 (부동소수점)
 * 무부호 정수(U16/U32/U64)는 항상 양수이므로 절댓값이 자기 자신이나,
 * 결과 저장 타입을 맞추기 위해 my_abs를 통과시킨다.
 * PTX 스펙: abs 명령어는 S8_TYPE을 지원하지 않음 (PTX ISA 제약).
 *
 * 실행 컨텍스트: cuda-sim.cc의 ptx_exec_inst()에서 opcode 디스패치로 호출.
 * 단일 스레드 기능 시뮬레이션; 재진입 안전.
 *
 * 호출 체인:
 *   ptx_exec_inst() → [abs_impl] → get_operand_value() → my_abs() → set_operand_value()
 */
void abs_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;  // [한국어] a: 소스 값, d: 절댓값 결과
  const operand_info &dst  = pI->dst();   // [한국어] 목적지 피연산자 (결과 레지스터)
  const operand_info &src1 = pI->src1();  // [한국어] 소스 피연산자 (절댓값을 구할 값)

  unsigned i_type = pI->get_type();  // [한국어] 명령어 데이터 타입 (S16/S32/F32 등)
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);  // [한국어] 소스 레지스터에서 값 읽기 (derefFlag=1: 역참조)

  switch (i_type) {  // [한국어] 타입별로 적절한 my_abs 적용
    case S16_TYPE:
      d.s16 = my_abs(a.s16);  // [한국어] S16: 부호 있는 16비트 절댓값
      break;
    case S32_TYPE:
      d.s32 = my_abs(a.s32);  // [한국어] S32: 부호 있는 32비트 절댓값
      break;
    case S64_TYPE:
      d.s64 = my_abs(a.s64);  // [한국어] S64: 부호 있는 64비트 절댓값
      break;
    case U16_TYPE:
      d.s16 = my_abs(a.u16);  // [한국어] U16: 무부호이므로 항상 비음수; s16 필드에 저장 (타입 호환)
      break;
    case U32_TYPE:
      d.s32 = my_abs(a.u32);  // [한국어] U32: 무부호 32비트; s32 필드에 저장
      break;
    case U64_TYPE:
      d.s64 = my_abs(a.u64);  // [한국어] U64: 무부호 64비트; s64 필드에 저장
      break;
    case F32_TYPE:
      d.f32 = my_abs(a.f32);  // [한국어] F32: 부동소수점 절댓값 (부호 비트 클리어)
      break;
    case F64_TYPE:
    case FF64_TYPE:
      d.f64 = my_abs(a.f64);  // [한국어] F64/FF64: 64비트 부동소수점 절댓값
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");  // [한국어] 지원하지 않는 타입 오류 출력
      assert(0);  // [한국어] 미지원 타입: 구현 버그 또는 잘못된 PTX
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);  // [한국어] 절댓값 결과를 목적지 레지스터에 기록
}

/*
 * [한국어]
 * addp_impl - PTXPlus `add.cc` 명령어 구현 — carry-in 포함 덧셈
 *
 * @pI    : 실행할 PTX 명령어; dst, src1, src2, src3(carry 레지스터), 타입, 반올림 모드 보유
 * @thread: 현재 실행 중인 시뮬레이트 스레드
 * @return: void; 결과는 dst에, carry/overflow 플래그는 predicate 레지스터에 기록
 *
 * PTXPlus 전용 carry 연산: d = src1 + src2 + carry_in
 * carry_in은 src3 predicate 레지스터의 비트 2(0x4 마스크)에서 읽는다.
 * 덧셈 후 carry와 overflow를 검출해 set_operand_value(overflow,carry)로 predicate에 기록.
 * carry/overflow 검출 방법:
 *   - 피연산자를 더 넓은 타입(s64/u64)으로 확장해 더한 후 해당 타입의 범위를 넘는 비트를 확인
 *   - overflow(부호 있는 오버플로): 두 입력 부호가 같을 때 결과 부호가 다르면 발생
 *   - carry(무부호 올림): 결과의 타입 비트 폭을 초과한 비트가 1이면 발생
 * 부동소수점 타입(F16/F32/F64)은 carry/overflow 미지원 (0으로 유지).
 * 반올림 모드(RN/RZ)를 fesetround()로 C 라이브러리에 적용하여 GPU 행동 에뮬레이션.
 *
 * 실행 컨텍스트: PTXPlus 코드에서 addc 체인을 표현하기 위해 사용.
 * cuda-sim.cc ptx_exec_inst()에서 opcode 디스패치로 호출.
 *
 * 호출 체인:
 *   ptx_exec_inst() → [addp_impl] → get_operand_value() × 3 → set_operand_value(carry/overflow)
 */
void addp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // PTXPlus add instruction with carry (carry is kept in a predicate) register
  ptx_reg_t src1_data, src2_data, src3_data, data;  // [한국어] src1/src2: 덧셈 피연산자, src3: carry predicate, data: 결과
  int overflow = 0;  // [한국어] 오버플로 플래그: 부호 있는 덧셈에서 결과가 표현 범위를 초과하면 1
  int carry = 0;     // [한국어] 캐리 플래그: 무부호 덧셈에서 자릿수 올림이 발생하면 1

  const operand_info &dst =
      pI->dst();  // get operand info of sources and destination
  // [한국어] 목적지 피연산자 (이중: pred|reg 가능)
  const operand_info &src1 =
      pI->src1();  // use them to determine that they are of type 'register'
  // [한국어] 첫 번째 소스 피연산자
  const operand_info &src2 = pI->src2();  // [한국어] 두 번째 소스 피연산자
  const operand_info &src3 = pI->src3();  // [한국어] 세 번째 피연산자: carry-in을 담은 predicate 레지스터

  unsigned i_type = pI->get_type();  // [한국어] 명령어 데이터 타입
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);  // [한국어] src1 값 읽기
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);  // [한국어] src2 값 읽기
  src3_data = thread->get_operand_value(src3, dst, i_type, thread, 1);  // [한국어] src3 pred 값 읽기 (carry_in은 src3_data.pred & 0x4)

  unsigned rounding_mode = pI->rounding_mode();  // [한국어] PTX 반올림 모드 수식자 (RN/RZ/RP/RM)
  int orig_rm = fegetround();  // [한국어] 현재 C 라이브러리 반올림 모드 저장 (복원을 위해)
  switch (rounding_mode) {
    case RN_OPTION:  // [한국어] RN(round to nearest even): C 라이브러리 기본값 → 변경 불필요
      break;
    case RZ_OPTION:  // [한국어] RZ(round toward zero): fesetround로 C 라이브러리 모드 변경
      fesetround(FE_TOWARDZERO);
      break;
    default:
      assert(0);  // [한국어] 미지원 반올림 모드 — 구현 버그
      break;
  }

  // performs addition. Sets carry and overflow if needed.
  // src3_data.pred&0x4 is the carry flag
  // [한국어] 타입별 덧셈 수행: 64비트로 확장해 계산하여 carry/overflow를 비트 연산으로 검출
  switch (i_type) {
    case S8_TYPE:
      /* [한국어] S8: 8비트 마스크(0xFF)로 각 입력 마스킹 후 64비트 덧셈
       * carry_in = src3_data.pred & 0x4 (pred 비트 2)
       * overflow: 두 입력 부호 비트(bit7)가 같은데 결과 부호가 다르면 발생 */
      data.s64 = (src1_data.s64 & 0x0000000FF) + (src2_data.s64 & 0x0000000FF) +
                 (src3_data.pred & 0x4);  // [한국어] carry_in을 더함 (pred 비트2 = 0x4)
      if (((src1_data.s64 & 0x80) - (src2_data.s64 & 0x80)) == 0) {  // [한국어] src1/src2 부호 비트 동일한 경우에만 overflow 검사
        overflow = ((src1_data.s64 & 0x80) - (data.s64 & 0x80)) == 0 ? 0 : 1;  // [한국어] 결과 부호가 다르면 overflow=1
      }
      carry = (data.u64 & 0x000000100) >> 8;  // [한국어] 비트8(9번째 비트)이 1이면 carry 발생 → 비트 추출
      break;
    case S16_TYPE:
      /* [한국어] S16: 16비트 마스크(0xFFFF)로 마스킹, overflow 비트 판정은 bit15 기준 */
      data.s64 = (src1_data.s64 & 0x00000FFFF) + (src2_data.s64 & 0x00000FFFF) +
                 (src3_data.pred & 0x4);
      if (((src1_data.s64 & 0x8000) - (src2_data.s64 & 0x8000)) == 0) {
        overflow =
            ((src1_data.s64 & 0x8000) - (data.s64 & 0x8000)) == 0 ? 0 : 1;
      }
      carry = (data.u64 & 0x000010000) >> 16;  // [한국어] 비트16이 1이면 carry
      break;
    case S32_TYPE:
      /* [한국어] S32: 32비트 마스크(0xFFFFFFFF)로 마스킹, overflow는 bit31 기준 */
      data.s64 = (src1_data.s64 & 0x0FFFFFFFF) + (src2_data.s64 & 0x0FFFFFFFF) +
                 (src3_data.pred & 0x4);
      if (((src1_data.s64 & 0x80000000) - (src2_data.s64 & 0x80000000)) == 0) {
        overflow = ((src1_data.s64 & 0x80000000) - (data.s64 & 0x80000000)) == 0
                       ? 0
                       : 1;
      }
      carry = (data.u64 & 0x100000000) >> 32;  // [한국어] 비트32가 1이면 carry
      break;
    case S64_TYPE:
      /* [한국어] S64: 64비트 이상 확장 없이 직접 덧셈; carry/overflow 감지 제한
       * carry_in(pred&0x4)을 더하나, 64비트 overflow/carry는 별도 처리 없음 */
      data.s64 = src1_data.s64 + src2_data.s64 + (src3_data.pred & 0x4);
      break;
    case U8_TYPE:
      /* [한국어] U8: 무부호 8비트; overflow 없음 (무부호), carry만 감지 */
      data.u64 = (src1_data.u64 & 0xFF) + (src2_data.u64 & 0xFF) +
                 (src3_data.pred & 0x4);
      carry = (data.u64 & 0x100) >> 8;  // [한국어] 비트8이 carry
      break;
    case U16_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFF) + (src2_data.u64 & 0xFFFF) +
                 (src3_data.pred & 0x4);
      carry = (data.u64 & 0x10000) >> 16;  // [한국어] 비트16이 carry
      break;
    case U32_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFFFFFF) + (src2_data.u64 & 0xFFFFFFFF) +
                 (src3_data.pred & 0x4);
      carry = (data.u64 & 0x100000000) >> 32;  // [한국어] 비트32가 carry
      break;
    case U64_TYPE:
      /* [한국어] U64: 64비트 무부호; s64 필드 사용 (u64와 동일 비트 패턴) */
      data.s64 = src1_data.s64 + src2_data.s64 + (src3_data.pred & 0x4);
      break;
    case F16_TYPE:
      data.f16 = src1_data.f16 + src2_data.f16;  // [한국어] FP16 덧셈; carry_in 미지원
      break;  // assert(0); break;  // [한국어] 원래 미구현으로 할 예정이었으나 현재는 허용
    case F32_TYPE:
      data.f32 = src1_data.f32 + src2_data.f32;  // [한국어] FP32 덧셈; 반올림 모드는 fesetround로 적용됨
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = src1_data.f64 + src2_data.f64;  // [한국어] FP64 덧셈
      break;
    default:
      assert(0);  // [한국어] 미지원 타입 — 구현 버그
      break;
  }
  fesetround(orig_rm);  // [한국어] 반올림 모드를 원래 값으로 복원 (다른 명령어에 영향 없도록)

  thread->set_operand_value(dst, data, i_type, thread, pI, overflow, carry);  // [한국어] 결과와 carry/overflow 플래그를 목적지에 기록
}

/*
 * [한국어]
 * add_impl - PTX `add` 명령어 구현 — 표준 덧셈 연산
 *
 * @pI    : 실행할 PTX 명령어; dst, src1, src2, 타입, 반올림 모드 보유
 * @thread: 현재 실행 중인 시뮬레이트 스레드
 * @return: void; 결과는 dst에, carry/overflow는 predicate에 기록
 *
 * PTX add 명령어: d = a + b
 * addp_impl과 거의 동일하나 carry_in(src3)이 없는 표준 이진 덧셈.
 * 지원 타입: S8/S16/S32/S64/U8/U16/U32/U64/F16/F32/F64/FF64
 * carry/overflow 검출은 addp_impl과 동일한 방식(64비트 확장 후 초과 비트 검출).
 * 부동소수점 타입에서는 fesetround()로 GPU 반올림 모드를 에뮬레이션한다:
 *   RN(round to nearest even) → C 기본값 유지
 *   RZ(round toward zero) → FE_TOWARDZERO 설정 후 복원
 *
 * 실행 컨텍스트: cuda-sim.cc ptx_exec_inst()에서 opcode 디스패치로 호출.
 * 단일 스레드 기능 시뮬레이션; 재진입 안전.
 *
 * 호출 체인:
 *   ptx_exec_inst() → [add_impl] → get_operand_value() × 2 → set_operand_value(carry/overflow)
 */
void add_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;  // [한국어] src1/src2: 덧셈 피연산자, data: 덧셈 결과
  int overflow = 0;  // [한국어] 오버플로 플래그 (부호 있는 산술)
  int carry = 0;     // [한국어] 캐리 플래그 (무부호 산술)

  const operand_info &dst =
      pI->dst();  // get operand info of sources and destination
  // [한국어] 목적지 피연산자
  const operand_info &src1 =
      pI->src1();  // use them to determine that they are of type 'register'
  // [한국어] 첫 번째 소스 피연산자
  const operand_info &src2 = pI->src2();  // [한국어] 두 번째 소스 피연산자

  unsigned i_type = pI->get_type();  // [한국어] 명령어 데이터 타입
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);  // [한국어] src1 값 읽기
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);  // [한국어] src2 값 읽기

  unsigned rounding_mode = pI->rounding_mode();  // [한국어] 반올림 모드 수식자
  int orig_rm = fegetround();  // [한국어] 현재 C 라이브러리 반올림 모드 저장
  switch (rounding_mode) {
    case RN_OPTION:  // [한국어] RN: 기본값 유지
      break;
    case RZ_OPTION:  // [한국어] RZ: 0쪽으로 반올림 설정
      fesetround(FE_TOWARDZERO);
      break;
    default:
      assert(0);  // [한국어] 미지원 반올림 모드
      break;
  }

  // performs addition. Sets carry and overflow if needed.
  // [한국어] 타입별 덧셈: addp_impl과 동일 로직이나 carry_in 없음
  switch (i_type) {
    case S8_TYPE:
      /* [한국어] S8: 8비트 마스크로 피연산자 마스킹 후 64비트로 덧셈
       * overflow: 입력 부호 동일 & 결과 부호 다르면 발생 */
      data.s64 = (src1_data.s64 & 0x0000000FF) + (src2_data.s64 & 0x0000000FF);
      if (((src1_data.s64 & 0x80) - (src2_data.s64 & 0x80)) == 0) {
        overflow = ((src1_data.s64 & 0x80) - (data.s64 & 0x80)) == 0 ? 0 : 1;
      }
      carry = (data.u64 & 0x000000100) >> 8;  // [한국어] 비트8이 carry
      break;
    case S16_TYPE:
      /* [한국어] S16: 16비트 마스킹, overflow/carry 검출 */
      data.s64 = (src1_data.s64 & 0x00000FFFF) + (src2_data.s64 & 0x00000FFFF);
      if (((src1_data.s64 & 0x8000) - (src2_data.s64 & 0x8000)) == 0) {
        overflow =
            ((src1_data.s64 & 0x8000) - (data.s64 & 0x8000)) == 0 ? 0 : 1;
      }
      carry = (data.u64 & 0x000010000) >> 16;  // [한국어] 비트16이 carry
      break;
    case S32_TYPE:
      /* [한국어] S32: 32비트 마스킹, overflow/carry 검출 */
      data.s64 = (src1_data.s64 & 0x0FFFFFFFF) + (src2_data.s64 & 0x0FFFFFFFF);
      if (((src1_data.s64 & 0x80000000) - (src2_data.s64 & 0x80000000)) == 0) {
        overflow = ((src1_data.s64 & 0x80000000) - (data.s64 & 0x80000000)) == 0
                       ? 0
                       : 1;
      }
      carry = (data.u64 & 0x100000000) >> 32;  // [한국어] 비트32가 carry
      break;
    case S64_TYPE:
      data.s64 = src1_data.s64 + src2_data.s64;  // [한국어] S64: 64비트 직접 덧셈; carry/overflow 별도 처리 없음
      break;
    case U8_TYPE:
      data.u64 = (src1_data.u64 & 0xFF) + (src2_data.u64 & 0xFF);  // [한국어] U8 마스킹 후 덧셈
      carry = (data.u64 & 0x100) >> 8;  // [한국어] 비트8이 carry
      break;
    case U16_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFF) + (src2_data.u64 & 0xFFFF);
      carry = (data.u64 & 0x10000) >> 16;  // [한국어] 비트16이 carry
      break;
    case U32_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFFFFFF) + (src2_data.u64 & 0xFFFFFFFF);
      carry = (data.u64 & 0x100000000) >> 32;  // [한국어] 비트32가 carry
      break;
    case U64_TYPE:
      data.u64 = src1_data.u64 + src2_data.u64;  // [한국어] U64: 64비트 무부호 직접 덧셈
      break;
    case F16_TYPE:
      data.f16 = src1_data.f16 + src2_data.f16;  // [한국어] FP16 덧셈 (반올림은 fesetround 적용됨)
      break;  // assert(0); break;
    case F32_TYPE:
      data.f32 = src1_data.f32 + src2_data.f32;  // [한국어] FP32 덧셈
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = src1_data.f64 + src2_data.f64;  // [한국어] FP64 덧셈
      break;
    default:
      assert(0);  // [한국어] 미지원 타입 — 구현 버그
      break;
  }
  fesetround(orig_rm);  // [한국어] 반올림 모드 원복 — 다른 명령어에 영향 없도록

  thread->set_operand_value(dst, data, i_type, thread, pI, overflow, carry);  // [한국어] 덧셈 결과와 carry/overflow를 목적지에 기록
}

/*
 * [한국어]
 * addc_impl - PTX `addc` 명령어 구현 — carry-in 포함 덧셈 (미구현)
 *
 * @pI    : 실행할 PTX 명령어
 * @thread: 현재 실행 중인 시뮬레이트 스레드 (미사용)
 * @return: void (실제로는 inst_not_implemented → abort)
 *
 * PTX addc 명령어: d = a + b + CC.CF (carry flag에서 carry-in 사용)
 * PTX ISA에서 addc는 명시적 carry 레지스터(CC.CF)를 사용하는 표준 PTX 명령어이나,
 * GPGPU-Sim에서는 현재 미구현 상태이다.
 * PTXPlus에서는 addp_impl이 유사한 역할을 담당한다.
 * 이 명령어가 실행되면 inst_not_implemented()가 오류 메시지를 출력하고 abort한다.
 *
 * 실행 컨텍스트: cuda-sim.cc ptx_exec_inst()에서 addc opcode 디스패치로 호출됨.
 *
 * 호출 체인:
 *   ptx_exec_inst() → [addc_impl] → inst_not_implemented() → abort()
 */
void addc_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);  // [한국어] 미구현 명령어 알림 — 오류 메시지 출력 후 시뮬레이터 종료
}

/*
 * [한국어]
 * and_impl - PTX `and` 명령어 구현: 비트 AND 연산
 *
 * @pI: 실행 중인 PTX 명령어 객체. 피연산자 타입(i_type), dst/src1/src2 정보 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 레지스터 파일 읽기/쓰기에 사용.
 *
 * PTX `and` 명령어는 두 피연산자에 대해 비트 단위 AND를 수행한다.
 * 피연산자 타입이 PRED_TYPE(술어 레지스터)인 경우 특수 처리가 필요하다:
 *   PTXPlus의 술어 레지스터 관례에서 1=false, 0=true (정상 부울 논리와 반전).
 *   따라서 AND(a, b)를 PTXPlus 공간에서 올바르게 표현하려면 드 모르간의 법칙을 적용해야 한다:
 *   AND(a, b) = NOT(NOT(a) AND NOT(b)) → PTXPlus 표현: ~(~src1 & ~src2).
 *   이 변환 없이 단순 & 연산을 하면 반전된 관례로 인해 잘못된 결과가 나온다.
 * PRED_TYPE이 아닌 경우(B16/B32/B64)는 64비트 필드에서 직접 & 연산을 수행한다.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: get_operand_value(), set_operand_value() — 레지스터 파일 접근.
 * 에러 경로: 없음 (타입 체크는 PTX 파서 수준에서 완료됨).
 *
 * 호출 체인:
 *   ptx_exec_inst() → [and_impl] → get_operand_value() / set_operand_value()
 */
void and_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data; // [한국어] 소스1, 소스2, 결과를 저장할 레지스터 값 변수

  const operand_info &dst = pI->dst();   // [한국어] 결과를 저장할 목적지 피연산자 정보
  const operand_info &src1 = pI->src1(); // [한국어] 첫 번째 소스 피연산자 정보
  const operand_info &src2 = pI->src2(); // [한국어] 두 번째 소스 피연산자 정보

  unsigned i_type = pI->get_type(); // [한국어] 명령어 피연산자 타입 (PRED_TYPE, B16/B32/B64 등)
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] src1 레지스터 값 읽기
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1); // [한국어] src2 레지스터 값 읽기

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)
    // [한국어] 술어 타입: PTXPlus의 반전 관례(1=false) 때문에 드 모르간 법칙 적용.
    //   ~(~a & ~b) = a OR b (PTXPlus 공간에서 AND를 구현하려면 이 변환 필요)
    data.pred = ~(~(src1_data.pred) & ~(src2_data.pred));
  else
    // [한국어] 비트 타입(B16/B32/B64): 64비트 필드에서 직접 비트 AND 수행
    data.u64 = src1_data.u64 & src2_data.u64;

  thread->set_operand_value(dst, data, i_type, thread, pI); // [한국어] 연산 결과를 dst 레지스터에 기록
}

/*
 * [한국어]
 * andn_impl - PTXPlus `andn` 명령어 구현: src2를 NOT한 뒤 src1과 AND
 *
 * @pI: 실행 중인 PTX 명령어 객체. 피연산자 타입과 dst/src1/src2 정보 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 레지스터 파일 읽기/쓰기에 사용.
 *
 * PTXPlus 전용 명령어로, 두 번째 피연산자(src2)를 비트 반전(NOT)한 뒤
 * 첫 번째 피연산자(src1)와 비트 AND를 수행한다: d = src1 & ~src2.
 * B16/B32/B64 타입만 지원하며, PRED_TYPE은 지원하지 않는다.
 * 이 명령어는 특정 비트를 선택적으로 클리어(clear)하는 마스킹 패턴에 활용된다.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: get_operand_value(), set_operand_value() — 레지스터 파일 접근.
 * 에러 경로: 지원하지 않는 타입이 오면 printf 출력 후 assert(0)으로 중단.
 *
 * 호출 체인:
 *   ptx_exec_inst() → [andn_impl] → get_operand_value() / set_operand_value()
 */
void andn_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data; // [한국어] 소스1, 소스2(NOT 적용 대상), 결과 변수

  const operand_info &dst = pI->dst();   // [한국어] 결과를 저장할 목적지 피연산자 정보
  const operand_info &src1 = pI->src1(); // [한국어] 첫 번째 소스 피연산자 정보
  const operand_info &src2 = pI->src2(); // [한국어] NOT 연산을 적용할 두 번째 소스 피연산자 정보

  unsigned i_type = pI->get_type(); // [한국어] 명령어 피연산자 타입 (B16/B32/B64만 허용)
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] src1 레지스터 값 읽기
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1); // [한국어] src2 레지스터 값 읽기

  switch (i_type) {
    case B16_TYPE:
      src2_data.u16 = ~src2_data.u16; // [한국어] src2의 16비트 값을 비트 반전 — andn 연산의 핵심
      break;
    case B32_TYPE:
      src2_data.u32 = ~src2_data.u32; // [한국어] src2의 32비트 값을 비트 반전
      break;
    case B64_TYPE:
      src2_data.u64 = ~src2_data.u64; // [한국어] src2의 64비트 값을 비트 반전
      break;
    default:
      printf("Execution error: type mismatch with instruction\n"); // [한국어] 지원하지 않는 타입 오류 출력
      assert(0); // [한국어] andn은 B16/B32/B64만 지원하므로 다른 타입은 버그로 처리
      break;
  }

  data.u64 = src1_data.u64 & src2_data.u64; // [한국어] src1 AND NOT(src2) 결과 계산 (64비트 필드로 통합)

  thread->set_operand_value(dst, data, i_type, thread, pI); // [한국어] 연산 결과를 dst 레지스터에 기록
}

/*
 * [한국어]
 * bar_callback - CTA 배리어 완료 후 타이밍 모델이 호출하는 콜백 함수
 *
 * @inst: 완료된 배리어 명령어의 기반 클래스 포인터. bar_id 필드를 통해 배리어 ID 접근.
 * @thread: 배리어를 완료한 스레드 상태. CTA ID, 리덕션 결과 읽기 및 레지스터 쓰기에 사용.
 *
 * PTX `bar.red` 명령어(배리어 기반 CTA 레벨 리덕션)가 완료된 후 타이밍 모델이
 * 이 콜백을 호출하여 리덕션 결과를 dst 레지스터에 기록한다.
 * bar_impl()이 thread->m_last_dram_callback.function = bar_callback 으로 등록하고,
 * 타이밍 모델의 mem_ctrl_pop 단계에서 실제 호출이 발생한다.
 * 리덕션 값(POPC/AND/OR 연산 결과)은 ptx_thread_info::get_reduction_value()로 읽는다.
 * 반환 타입은 항상 U32_TYPE으로 고정된다.
 *
 * 실행 컨텍스트: 타이밍 시뮬레이션(gpgpu-sim) 단계, 콜백 호출로 진입.
 * 호출자: gpu-sim.cc의 mem_ctrl_pop() — 배리어 완료 이벤트 처리.
 * 피호출자: get_reduction_value(), set_operand_value() — CTA 리덕션 결과 읽기/레지스터 쓰기.
 * 에러 경로: 없음 (타입은 U32_TYPE으로 고정).
 *
 * 호출 체인:
 *   mem_ctrl_pop() → [bar_callback] → get_reduction_value() / set_operand_value()
 */
void bar_callback(const inst_t *inst, ptx_thread_info *thread) {
  unsigned ctaid = thread->get_cta_uid(); // [한국어] 이 스레드가 속한 CTA(Cooperative Thread Array)의 고유 ID
  unsigned barid = inst->bar_id;          // [한국어] 완료된 배리어의 식별자 (bar_impl에서 set_bar_id로 설정된 값)
  unsigned value = thread->get_reduction_value(ctaid, barid); // [한국어] CTA 단위 리덕션 결과값 읽기 (POPC/AND/OR)
  const ptx_instruction *pI = dynamic_cast<const ptx_instruction *>(inst); // [한국어] 기반 클래스 inst_t를 파생 클래스 ptx_instruction*로 다운캐스트
  const operand_info &dst = pI->dst(); // [한국어] 리덕션 결과를 저장할 목적지 레지스터 정보
  ptx_reg_t data;     // [한국어] 결과값을 담을 레지스터 유니온 변수
  data.u32 = value;   // [한국어] 리덕션 결과를 u32 필드에 설정 (bar.red 결과는 항상 32비트 부호 없는 정수)
  thread->set_operand_value(dst, value, U32_TYPE, thread, pI); // [한국어] dst 레지스터에 리덕션 결과 기록
}

/*
 * [한국어]
 * atom_callback - 원자적 메모리 연산 완료 후 타이밍 모델이 호출하는 콜백 함수
 *
 * @inst: 완료된 atom 명령어의 기반 클래스 포인터. 피연산자/타입/연산 종류 정보 포함.
 * @thread: 원자 연산을 수행한 스레드 상태. 메모리 접근 및 레지스터 기록에 사용.
 *
 * PTX `atom` 명령어의 실제 메모리 읽기-수정-쓰기를 수행하는 콜백이다.
 * atom_impl()이 thread->m_last_dram_callback에 이 함수를 등록하고,
 * 타이밍 모델의 mem_ctrl_pop 단계에서 실제 호출이 발생한다.
 * 처리 순서:
 *   1) 출력 타입(to_type) 디코드 → 크기(바이트 수) 결정
 *   2) src1(주소)에서 현재 메모리 값 읽기
 *   3) 현재 값을 dst 레지스터에 기록 (원자 op 전 값 반환)
 *   4) 원자 연산 수행 (AND/OR/XOR/CAS/EXCH/ADD/INC/DEC/MIN/MAX)
 *   5) 연산 결과를 동일 주소에 다시 기록
 * CAS는 src3(비교값)도 읽는다. dst가 '_'(와일드카드)이면 반환값 기록을 생략한다.
 * 지원 메모리 공간: global_space, shared_space (generic 주소는 변환 후 처리).
 *
 * 실행 컨텍스트: 타이밍 시뮬레이션(gpgpu-sim) 단계, 콜백 호출로 진입.
 * 호출자: gpu-sim.cc의 mem_ctrl_pop() — 메모리 완료 이벤트 처리.
 * 피호출자: get_operand_value(), mem->read(), set_operand_value(), mem->write().
 * 에러 경로: 지원하지 않는 타입이나 메모리 공간이면 printf 출력 후 assert(0)/abort().
 *
 * 호출 체인:
 *   mem_ctrl_pop() → [atom_callback] → mem->read() / mem->write() / set_operand_value()
 */
void atom_callback(const inst_t *inst, ptx_thread_info *thread) {
  const ptx_instruction *pI = dynamic_cast<const ptx_instruction *>(inst); // [한국어] 기반 클래스 inst_t를 파생 클래스 ptx_instruction*로 다운캐스트

  // "Decode" the output type
  unsigned to_type = pI->get_type(); // [한국어] 원자 연산의 피연산자 타입 (U32/S32/U64/B32/B64/F32 등)
  size_t size; // [한국어] 피연산자의 바이트 크기 (type_decode로 결정)
  int t;       // [한국어] 타입의 부호 여부 등 보조 정보 (type_decode 출력)
  type_info_key::type_decode(to_type, size, t); // [한국어] to_type에서 size(비트 수)와 t(부호) 추출

  // Set up operand variables
  ptx_reg_t data;       // d  // [한국어] 메모리에서 읽은 현재 값 (연산 전 값 → dst로 반환)
  ptx_reg_t src1_data;  // a  // [한국어] src1: 메모리 주소를 담는 피연산자
  ptx_reg_t src2_data;  // b  // [한국어] src2: 원자 연산의 두 번째 피연산자 (비교값 또는 연산값)
  ptx_reg_t op_result;  // temp variable to hold operation result  // [한국어] 원자 연산 결과를 임시 저장

  bool data_ready = false; // [한국어] 연산 결과가 준비되었는지 추적하는 플래그 (연산 후 true로 설정)

  // Get operand info of sources and destination
  const operand_info &dst = pI->dst();    // d  // [한국어] 결과(연산 전 값)를 기록할 목적지 레지스터 정보
  const operand_info &src1 = pI->src1();  // a  // [한국어] 원자 연산 대상 메모리 주소를 담은 소스1 피연산자
  const operand_info &src2 = pI->src2();  // b  // [한국어] 원자 연산의 operand b (AND/OR/XOR/ADD 등에서 두 번째 값)

  // Get operand values
  src1_data = thread->get_operand_value(src1, src1, to_type, thread, 1);  // a  // [한국어] src1(메모리 주소) 값 읽기
  if (dst.get_symbol()->type()) {
    // [한국어] dst가 유효한 레지스터 심볼인 경우: src2를 dst 타입 기준으로 읽음
    src2_data = thread->get_operand_value(src2, dst, to_type, thread, 1);  // b
  } else {
    // This is the case whent he first argument (dest) is '_'
    // [한국어] dst가 '_'(와일드카드, 반환값 무시)인 경우: src2를 src1 타입 기준으로 읽음
    src2_data = thread->get_operand_value(src2, src1, to_type, thread, 1);  // b
  }

  // Check state space
  addr_t effective_address = src1_data.u64; // [한국어] src1에서 읽은 실제 메모리 주소
  memory_space_t space = pI->get_space();   // [한국어] 명령어에 명시된 메모리 공간 (global/shared/undefined)
  if (space == undefined_space) {
    // generic space - determine space via address
    // [한국어] undefined_space = 제네릭 주소 공간: 주소 값을 분석해 실제 공간 결정
    if (whichspace(effective_address) == global_space) {
      effective_address = generic_to_global(effective_address); // [한국어] 제네릭 주소 → 글로벌 주소로 변환
      space = global_space; // [한국어] 메모리 공간을 글로벌로 확정
    } else if (whichspace(effective_address) == shared_space) {
      unsigned smid = thread->get_hw_sid();                               // [한국어] 이 스레드가 실행 중인 SM(Streaming Multiprocessor) ID
      effective_address = generic_to_shared(smid, effective_address);     // [한국어] 제네릭 주소 → 공유 메모리 주소로 변환 (SM별 베이스 오프셋 적용)
      space = shared_space; // [한국어] 메모리 공간을 공유 메모리로 확정
    } else {
      abort(); // [한국어] 글로벌도 공유도 아닌 제네릭 주소 — 원자 연산 미지원 공간이므로 중단
    }
  }
  assert(space == global_space || space == shared_space); // [한국어] 원자 연산은 글로벌/공유 메모리에서만 허용됨을 확인

  memory_space *mem = NULL; // [한국어] 실제 메모리 객체 포인터 (글로벌 또는 공유 메모리)
  if (space == global_space)
    mem = thread->get_global_memory(); // [한국어] 글로벌 메모리 공간 객체 가져오기
  else if (space == shared_space)
    mem = thread->m_shared_mem;        // [한국어] 공유 메모리 공간 객체 가져오기 (SM 내 공유)
  else
    abort(); // [한국어] 도달 불가 경로 — 위 assert를 통과했으므로 이 분기는 방어 코드

  // Copy value pointed to in operand 'a' into register 'd'
  // (i.e. copy src1_data to dst)
  mem->read(effective_address, size / 8, &data.s64); // [한국어] 유효 주소에서 size/8 바이트를 읽어 data에 저장 (원자 연산 전 현재 값)
  if (dst.get_symbol()->type()) {
    thread->set_operand_value(dst, data, to_type, thread,
                              pI);  // Write value into register 'd'  // [한국어] 원자 연산 전 값을 dst 레지스터에 기록 (PTX 스펙: atom은 이전 값을 반환)
  }

  // Get the atomic operation to be performed
  unsigned m_atomic_spec = pI->get_atomic(); // [한국어] 수행할 원자 연산 종류 (ATOMIC_AND/OR/XOR/CAS/EXCH/ADD/INC/DEC/MIN/MAX)

  switch (m_atomic_spec) {
    // AND
    case ATOMIC_AND: {
      // [한국어] ATOMIC_AND: 메모리 현재 값과 src2를 비트 AND하여 결과 생성
      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = data.u32 & src2_data.u32; // [한국어] 32비트 비트 AND 연산
          data_ready = true; // [한국어] 결과 준비 완료 플래그 설정
          break;
        case S32_TYPE:
          op_result.s32 = data.s32 & src2_data.s32; // [한국어] 부호 있는 32비트 비트 AND 연산
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.AND "
              "only accepts b32\n",
              to_type); // [한국어] atom.AND는 b32/u32/s32만 지원 — 타입 불일치 오류 출력
          assert(0); // [한국어] 지원하지 않는 타입으로는 실행 불가
          break;
      }

      break;
    }
      // OR
    case ATOMIC_OR: {
      // [한국어] ATOMIC_OR: 메모리 현재 값과 src2를 비트 OR하여 결과 생성
      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = data.u32 | src2_data.u32; // [한국어] 32비트 비트 OR 연산
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = data.s32 | src2_data.s32; // [한국어] 부호 있는 32비트 비트 OR 연산
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.OR "
              "only accepts b32\n",
              to_type); // [한국어] atom.OR는 b32/u32/s32만 지원 — 타입 불일치 오류 출력
          assert(0);
          break;
      }

      break;
    }
      // XOR
    case ATOMIC_XOR: {
      // [한국어] ATOMIC_XOR: 메모리 현재 값과 src2를 비트 XOR하여 결과 생성
      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = data.u32 ^ src2_data.u32; // [한국어] 32비트 비트 XOR 연산
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = data.s32 ^ src2_data.s32; // [한국어] 부호 있는 32비트 비트 XOR 연산
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.XOR "
              "only accepts b32\n",
              to_type); // [한국어] atom.XOR는 b32/u32/s32만 지원 — 타입 불일치 오류 출력
          assert(0);
          break;
      }

      break;
    }
      // CAS
    case ATOMIC_CAS: {
      // [한국어] ATOMIC_CAS: Compare-And-Swap — data(현재값) == src2이면 src3으로 교체
      //   MY_CAS_I(old, cmp, new): old == cmp ? new : old
      ptx_reg_t src3_data; // [한국어] CAS의 새 값(new value)을 담는 세 번째 피연산자
      const operand_info &src3 = pI->src3(); // [한국어] CAS의 새 값 피연산자 정보
      src3_data = thread->get_operand_value(src3, dst, to_type, thread, 1); // [한국어] src3(새 값) 읽기

      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = MY_CAS_I(data.u32, src2_data.u32, src3_data.u32); // [한국어] 32비트 CAS: 현재값 == src2이면 src3으로 교체
          data_ready = true;
          break;
        case B64_TYPE:
        case U64_TYPE:
          op_result.u64 = MY_CAS_I(data.u64, src2_data.u64, src3_data.u64); // [한국어] 64비트 CAS: 현재값 == src2이면 src3으로 교체
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = MY_CAS_I(data.s32, src2_data.s32, src3_data.s32); // [한국어] 부호 있는 32비트 CAS
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.CAS "
              "only accepts b32 and b64\n",
              to_type); // [한국어] atom.CAS는 b32/u32/s32/b64/u64만 지원 — 타입 불일치 오류 출력
          assert(0);
          break;
      }

      break;
    }
      // EXCH
    case ATOMIC_EXCH: {
      // [한국어] ATOMIC_EXCH: Exchange — 현재 메모리 값을 src2로 무조건 교체
      //   MY_EXCH(old, new): new (항상 src2 값으로 교체)
      switch (to_type) {
        case B32_TYPE:
        case U32_TYPE:
          op_result.u32 = MY_EXCH(data.u32, src2_data.u32); // [한국어] 32비트 교체: 결과는 항상 src2 값
          data_ready = true;
          break;
        case B64_TYPE:
        case U64_TYPE:
          op_result.u64 = MY_EXCH(data.u64, src2_data.u64); // [한국어] 64비트 교체
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = MY_EXCH(data.s32, src2_data.s32); // [한국어] 부호 있는 32비트 교체
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch (%x) with instruction\natom.EXCH "
              "only accepts b32\n",
              to_type); // [한국어] atom.EXCH 타입 불일치 오류 출력
          assert(0);
          break;
      }

      break;
    }
      // ADD
    case ATOMIC_ADD: {
      // [한국어] ATOMIC_ADD: 현재 메모리 값과 src2를 덧셈하여 결과 저장
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = data.u32 + src2_data.u32; // [한국어] 부호 없는 32비트 덧셈
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = data.s32 + src2_data.s32; // [한국어] 부호 있는 32비트 덧셈
          data_ready = true;
          break;
        case U64_TYPE:
          op_result.u64 = data.u64 + src2_data.u64; // [한국어] 부호 없는 64비트 덧셈
          data_ready = true;
          break;
        case F32_TYPE:
          op_result.f32 = data.f32 + src2_data.f32; // [한국어] 단정밀도 부동소수점 덧셈
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.ADD only "
              "accepts u32, s32, u64, and f32\n"); // [한국어] atom.ADD 타입 불일치 오류 출력
          assert(0);
          break;
      }

      break;
    }
      // INC
    case ATOMIC_INC: {
      // [한국어] ATOMIC_INC: 증가 연산 — data < src2이면 data+1, 아니면 0으로 설정
      //   MY_INC_I(val, limit): val < limit ? val+1 : 0 (환형 카운터 패턴)
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = MY_INC_I(data.u32, src2_data.u32); // [한국어] 환형 증가: src2를 상한으로 하는 카운터 증가
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.INC only "
              "accepts u32 and s32\n"); // [한국어] atom.INC는 u32만 지원 — 타입 불일치 오류 출력
          assert(0);
          break;
      }

      break;
    }
      // DEC
    case ATOMIC_DEC: {
      // [한국어] ATOMIC_DEC: 감소 연산 — data > 0이면 data-1, 아니면 src2로 설정
      //   MY_DEC_I(val, limit): val > 0 ? val-1 : limit (환형 카운터 패턴)
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = MY_DEC_I(data.u32, src2_data.u32); // [한국어] 환형 감소: 0 이하이면 src2(상한)로 리셋
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.DEC only "
              "accepts u32 and s32\n"); // [한국어] atom.DEC는 u32만 지원 — 타입 불일치 오류 출력
          assert(0);
          break;
      }

      break;
    }
      // MIN
    case ATOMIC_MIN: {
      // [한국어] ATOMIC_MIN: 현재 메모리 값과 src2 중 최솟값을 결과로 저장
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = MY_MIN_I(data.u32, src2_data.u32); // [한국어] 부호 없는 32비트 최솟값
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = MY_MIN_I(data.s32, src2_data.s32); // [한국어] 부호 있는 32비트 최솟값
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.MIN only "
              "accepts u32 and s32\n"); // [한국어] atom.MIN 타입 불일치 오류 출력
          assert(0);
          break;
      }

      break;
    }
      // MAX
    case ATOMIC_MAX: {
      // [한국어] ATOMIC_MAX: 현재 메모리 값과 src2 중 최댓값을 결과로 저장
      switch (to_type) {
        case U32_TYPE:
          op_result.u32 = MY_MAX_I(data.u32, src2_data.u32); // [한국어] 부호 없는 32비트 최댓값
          data_ready = true;
          break;
        case S32_TYPE:
          op_result.s32 = MY_MAX_I(data.s32, src2_data.s32); // [한국어] 부호 있는 32비트 최댓값
          data_ready = true;
          break;
        default:
          printf(
              "Execution error: type mismatch with instruction\natom.MAX only "
              "accepts u32 and s32\n"); // [한국어] atom.MAX 타입 불일치 오류 출력
          assert(0);
          break;
      }

      break;
    }
      // DEFAULT
    default: {
      assert(0); // [한국어] 알 수 없는 원자 연산 종류 — 구현되지 않은 연산이므로 중단
      break;
    }
  }

  // Write operation result into  memory
  // (i.e. copy src1_data to dst)
  if (data_ready) {
    mem->write(effective_address, size / 8, &op_result.s64, thread, pI); // [한국어] 원자 연산 결과를 유효 주소에 다시 기록 (읽기-수정-쓰기 완료)
  } else {
    printf("Execution error: data_ready not set\n"); // [한국어] data_ready가 설정되지 않은 경우 — 연산 결과 없이 쓰기 시도하는 버그
    assert(0); // [한국어] 논리 오류: 모든 switch 분기에서 data_ready를 설정해야 함
  }
}

// atom_impl will now result in a callback being called in mem_ctrl_pop
// (gpu-sim.c)
/*
 * [한국어]
 * atom_impl - PTX `atom` 명령어의 기능 시뮬레이션 진입 함수
 *
 * @pI: 실행 중인 PTX atom 명령어 객체. 메모리 공간, 피연산자 타입, 연산 종류 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 유효 주소 계산 및 콜백 등록에 사용.
 *
 * PTX `atom.space.operation.type d, a, b[, c]` 명령어의 기능 시뮬레이션 측 구현.
 * 이 함수는 실제 메모리 읽기-수정-쓰기를 직접 수행하지 않는다.
 * 대신 유효 주소를 계산하고, atom_callback 함수를 thread->m_last_dram_callback에 등록한다.
 * 타이밍 모델의 mem_ctrl_pop 단계(gpu-sim.cc)에서 메모리 요청이 완료될 때
 * 등록된 atom_callback이 호출되어 실제 원자 연산을 수행한다.
 *
 * 제네릭 주소(undefined_space)인 경우 whichspace()로 실제 메모리 공간을 판별하고
 * generic_to_global() 또는 generic_to_shared()로 구체적인 주소로 변환한다.
 * 지원 메모리 공간: global_space, shared_space.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: get_operand_value(), whichspace(), generic_to_global/shared().
 * 에러 경로: 지원하지 않는 메모리 공간이면 abort().
 *
 * 호출 체인:
 *   ptx_exec_inst() → [atom_impl] → (콜백 등록) → mem_ctrl_pop() → atom_callback()
 */
void atom_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // SYNTAX
  // atom.space.operation.type d, a, b[, c]; (now read in callback)

  // obtain memory space of the operation
  memory_space_t space = pI->get_space(); // [한국어] 명령어에 명시된 메모리 공간 (global/shared/undefined)

  // get the memory address
  const operand_info &src1 = pI->src1(); // [한국어] 원자 연산 대상 메모리 주소를 담은 피연산자 정보
  // const operand_info &dst  = pI->dst();  // not needed for effective address
  // calculation
  unsigned i_type = pI->get_type(); // [한국어] 피연산자 타입 — 주소 읽기 시 타입 맥락 제공
  ptx_reg_t src1_data;
  src1_data = thread->get_operand_value(src1, src1, i_type, thread, 1); // [한국어] src1에서 메모리 주소 값 읽기
  addr_t effective_address = src1_data.u64; // [한국어] 읽은 값을 64비트 주소로 해석

  addr_t effective_address_final; // [한국어] 제네릭 주소 변환 후 최종 유효 주소

  // handle generic memory space by converting it to global
  if (space == undefined_space) {
    // [한국어] 제네릭 주소 공간: 주소 값 분석으로 실제 공간 판별 후 변환
    if (whichspace(effective_address) == global_space) {
      effective_address_final = generic_to_global(effective_address); // [한국어] 제네릭 → 글로벌 주소 변환
      space = global_space; // [한국어] 실제 메모리 공간을 글로벌로 확정
    } else if (whichspace(effective_address) == shared_space) {
      unsigned smid = thread->get_hw_sid();                               // [한국어] 이 스레드가 실행 중인 SM ID
      effective_address_final = generic_to_shared(smid, effective_address); // [한국어] 제네릭 → SM별 공유 메모리 주소 변환
      space = shared_space; // [한국어] 실제 메모리 공간을 공유 메모리로 확정
    } else {
      abort(); // [한국어] 글로벌/공유 이외의 제네릭 주소 — atom 미지원 공간
    }
  } else {
    assert(space == global_space || space == shared_space); // [한국어] 명시적 공간은 글로벌/공유만 허용
    effective_address_final = effective_address; // [한국어] 명시적 공간이면 주소 변환 없이 그대로 사용
  }

  // Check state space
  assert(space == global_space || space == shared_space); // [한국어] 최종적으로 허용된 메모리 공간인지 재확인

  thread->m_last_effective_address = effective_address_final; // [한국어] 타이밍 모델이 메모리 접근을 추적하기 위해 유효 주소 저장
  thread->m_last_memory_space = space;                        // [한국어] 타이밍 모델이 공간별 처리에 사용할 메모리 공간 저장
  thread->m_last_dram_callback.function = atom_callback;      // [한국어] 메모리 완료 시 타이밍 모델이 호출할 콜백 함수 등록
  thread->m_last_dram_callback.instruction = pI;              // [한국어] 콜백에서 피연산자 정보를 참조하기 위해 명령어 포인터 저장
}

/*
 * [한국어]
 * bar_impl - PTX `bar` 명령어 구현: CTA 수준 배리어 동기화 및 리덕션
 *
 * @pIin: 실행 중인 PTX bar 명령어 객체 (const). 배리어 종류/피연산자/리덕션 종류 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. CTA ID, 리덕션 집계, 콜백 등록에 사용.
 *
 * PTX `bar.sync/arrive/red` 명령어를 처리한다.
 * - SYNC_OPTION: 배리어 ID(및 선택적 스레드 수)를 pI에 설정 후 타이밍 모델에 위임.
 * - ARRIVE_OPTION: 배리어 ID와 스레드 수를 설정하고 도달 신호를 보내는 변형.
 * - RED_OPTION: POPC(참인 스레드 수)/AND/OR 리덕션 연산을 CTA 단위로 누적.
 *   피연산자3(술어)의 PTXPlus 반전 관례(0=true, 1=false)를 `!(op.pred & 1)`로 변환.
 *   피연산자 수 > 3이면 스레드 수 상한이 명시, 아니면 전체 CTA 스레드 수 사용.
 *
 * pI는 const_cast로 비const 포인터를 얻어 set_bar_id/set_bar_count를 호출한다.
 * 이 설계는 pI가 런타임 정보를 쓰기 위해 비const가 필요하기 때문이다.
 * 완료 콜백(bar_callback)을 m_last_dram_callback에 등록하여
 * 타이밍 모델이 배리어 완료 시 결과를 dst 레지스터에 기록하게 한다.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: get_operand_value(), popc_reduction(), and_reduction(), or_reduction().
 * 에러 경로: 미지원 bar_op 또는 red_op이면 abort().
 *
 * 호출 체인:
 *   ptx_exec_inst() → [bar_impl] → popc/and/or_reduction() → (콜백 등록) → bar_callback()
 */
void bar_impl(const ptx_instruction *pIin, ptx_thread_info *thread) {
  ptx_instruction *pI = const_cast<ptx_instruction *>(pIin); // [한국어] set_bar_id/set_bar_count 호출을 위해 const 제거 (런타임 bar 정보 기록 목적)
  unsigned bar_op = pI->barrier_op(); // [한국어] 배리어 종류: SYNC_OPTION / ARRIVE_OPTION / RED_OPTION
  unsigned red_op = pI->get_atomic(); // [한국어] 리덕션 연산 종류: ATOMIC_POPC / ATOMIC_AND / ATOMIC_OR (RED_OPTION에서만 유효)
  unsigned ctaid = thread->get_cta_uid(); // [한국어] 이 스레드가 속한 CTA의 고유 ID (리덕션 집계 공간 식별)

  switch (bar_op) {
    case SYNC_OPTION: {
      // [한국어] bar.sync: 배리어 ID와 선택적 스레드 수를 설정하여 동기화 배리어 등록
      if (pI->get_num_operands() > 1) {
        // [한국어] 피연산자가 2개인 경우: bar.sync barID, threadCount 형식
        const operand_info &op0 = pI->dst();   // [한국어] op0: 배리어 ID 피연산자
        const operand_info &op1 = pI->src1();  // [한국어] op1: 배리어 참여 스레드 수 피연산자
        ptx_reg_t op0_data;
        ptx_reg_t op1_data;
        op0_data = thread->get_operand_value(op0, op0, U32_TYPE, thread, 1); // [한국어] 배리어 ID 읽기
        op1_data = thread->get_operand_value(op1, op1, U32_TYPE, thread, 1); // [한국어] 참여 스레드 수 읽기
        pI->set_bar_id(op0_data.u32);    // [한국어] 명령어 객체에 배리어 ID 기록 (타이밍 모델이 사용)
        pI->set_bar_count(op1_data.u32); // [한국어] 명령어 객체에 스레드 수 기록 (몇 개 스레드가 모일 때 배리어 해제될지)
      } else {
        // [한국어] 피연산자가 1개인 경우: bar.sync barID 형식 (스레드 수는 기본값)
        const operand_info &op0 = pI->dst(); // [한국어] op0: 배리어 ID 피연산자
        ptx_reg_t op0_data;
        op0_data = thread->get_operand_value(op0, op0, U32_TYPE, thread, 1); // [한국어] 배리어 ID 읽기
        pI->set_bar_id(op0_data.u32); // [한국어] 배리어 ID만 설정 (스레드 수는 생략 — 전체 CTA)
      }
      break;
    }
    case ARRIVE_OPTION: {
      // [한국어] bar.arrive: 배리어에 도달만 신고하고 기다리지 않는 변형
      const operand_info &op0 = pI->dst();   // [한국어] op0: 배리어 ID 피연산자
      const operand_info &op1 = pI->src1();  // [한국어] op1: 배리어 참여 스레드 수 피연산자
      ptx_reg_t op0_data;
      ptx_reg_t op1_data;
      op0_data = thread->get_operand_value(op0, op0, U32_TYPE, thread, 1); // [한국어] 배리어 ID 읽기
      op1_data = thread->get_operand_value(op1, op1, U32_TYPE, thread, 1); // [한국어] 참여 스레드 수 읽기
      pI->set_bar_id(op0_data.u32);    // [한국어] 명령어 객체에 배리어 ID 기록
      pI->set_bar_count(op1_data.u32); // [한국어] 명령어 객체에 스레드 수 기록
      break;
    }
    case RED_OPTION: {
      // [한국어] bar.red: 배리어 + 리덕션 — 모든 스레드의 술어 값을 POPC/AND/OR로 집계
      if (pI->get_num_operands() > 3) {
        // [한국어] 피연산자가 4개인 경우: bar.red.op barID, threadCount, pred 형식
        const operand_info &op1 = pI->src1(); // [한국어] op1: 배리어 ID 피연산자
        const operand_info &op2 = pI->src2(); // [한국어] op2: 참여 스레드 수 피연산자
        const operand_info &op3 = pI->src3(); // [한국어] op3: 리덕션에 참여하는 술어 값 피연산자
        ptx_reg_t op1_data;
        ptx_reg_t op2_data;
        ptx_reg_t op3_data;
        op1_data = thread->get_operand_value(op1, op1, U32_TYPE, thread, 1);   // [한국어] 배리어 ID 읽기
        op2_data = thread->get_operand_value(op2, op2, U32_TYPE, thread, 1);   // [한국어] 참여 스레드 수 읽기
        op3_data = thread->get_operand_value(op3, op3, PRED_TYPE, thread, 1);  // [한국어] 술어 레지스터 값 읽기 (PTXPlus 반전 관례)
        op3_data.u32 = !(op3_data.pred & 0x0001); // [한국어] PTXPlus 반전 변환: bit0(0=true,1=false)를 일반 부울로 변환
        pI->set_bar_id(op1_data.u32);    // [한국어] 배리어 ID 기록
        pI->set_bar_count(op2_data.u32); // [한국어] 참여 스레드 수 기록
        switch (red_op) {
          case ATOMIC_POPC:
            thread->popc_reduction(ctaid, op1_data.u32, op3_data.u32); // [한국어] POPC: 이 스레드의 술어 값(0/1)을 CTA 누적 카운터에 합산
            break;
          case ATOMIC_AND:
            thread->and_reduction(ctaid, op1_data.u32, op3_data.u32);  // [한국어] AND: 이 스레드 술어로 CTA 누적 AND 값 업데이트
            break;
          case ATOMIC_OR:
            thread->or_reduction(ctaid, op1_data.u32, op3_data.u32);   // [한국어] OR: 이 스레드 술어로 CTA 누적 OR 값 업데이트
            break;
          default:
            abort(); // [한국어] bar.red에서 지원하지 않는 리덕션 연산
            break;
        }
      } else {
        // [한국어] 피연산자가 3개 이하인 경우: bar.red.op barID, pred 형식 (스레드 수 = 전체 CTA)
        const operand_info &op1 = pI->src1(); // [한국어] op1: 배리어 ID 피연산자
        const operand_info &op2 = pI->src2(); // [한국어] op2: 리덕션 술어 값 피연산자
        ptx_reg_t op1_data;
        ptx_reg_t op2_data;
        op1_data = thread->get_operand_value(op1, op1, U32_TYPE, thread, 1);  // [한국어] 배리어 ID 읽기
        op2_data = thread->get_operand_value(op2, op2, PRED_TYPE, thread, 1); // [한국어] 술어 레지스터 값 읽기
        op2_data.u32 = !(op2_data.pred & 0x0001); // [한국어] PTXPlus 반전 변환: bit0을 일반 부울로 변환
        pI->set_bar_id(op1_data.u32); // [한국어] 배리어 ID 기록
        pI->set_bar_count(thread->get_ntid().x * thread->get_ntid().y *
                          thread->get_ntid().z); // [한국어] 스레드 수 = CTA 전체 스레드 수 (x×y×z)
        switch (red_op) {
          case ATOMIC_POPC:
            thread->popc_reduction(ctaid, op1_data.u32, op2_data.u32); // [한국어] POPC 리덕션: 참인 스레드 수 집계
            break;
          case ATOMIC_AND:
            thread->and_reduction(ctaid, op1_data.u32, op2_data.u32);  // [한국어] AND 리덕션: 모든 스레드 술어의 AND
            break;
          case ATOMIC_OR:
            thread->or_reduction(ctaid, op1_data.u32, op2_data.u32);   // [한국어] OR 리덕션: 모든 스레드 술어의 OR
            break;
          default:
            abort(); // [한국어] 지원하지 않는 리덕션 연산
            break;
        }
      }
      break;
    }
    default:
      abort(); // [한국어] 미지원 배리어 종류 — SYNC/ARRIVE/RED 이외의 값은 버그
      break;
  }

  thread->m_last_dram_callback.function = bar_callback;    // [한국어] 배리어 완료 시 타이밍 모델이 호출할 콜백 함수 등록
  thread->m_last_dram_callback.instruction = pIin;         // [한국어] 콜백에서 피연산자 참조용으로 명령어 포인터 저장
}

/*
 * [한국어]
 * bfe_impl - PTX `bfe` 명령어 구현: 비트 필드 추출 (Bit Field Extract)
 *
 * @pI: 실행 중인 PTX 명령어 객체. 피연산자 타입(U32/U64/S32/S64), dst/src1~3 정보 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 레지스터 파일 읽기/쓰기에 사용.
 *
 * PTX `bfe.type d, a, b, c` 명령어:
 *   소스 값(a)에서 bit position b(pos)부터 len(c)개의 비트를 추출한다.
 *   - U 타입: 추출된 비트를 오른쪽 정렬하고 상위 비트는 0으로 채움 (zero-extension).
 *   - S 타입: 추출된 비트를 오른쪽 정렬하고 부호 비트로 상위를 채움 (sign-extension).
 *   pos와 len은 8비트로 클램프됨 (& 0xFF). len==0이면 결과는 0.
 * 추출 알고리즘:
 *   1) 소스를 pos만큼 오른쪽 시프트
 *   2) len 길이의 마스크로 상위 비트 제거 (U 타입)
 *      또는 추출된 최상위 비트(sbit)로 부호 확장 (S 타입)
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: get_operand_value(), set_operand_value() — 레지스터 파일 접근.
 * 에러 경로: 지원하지 않는 타입이면 printf 출력 후 abort().
 *
 * 호출 체인:
 *   ptx_exec_inst() → [bfe_impl] → get_operand_value() / set_operand_value()
 */
void bfe_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  unsigned i_type = pI->get_type(); // [한국어] 피연산자 타입 (U32/U64/S32/S64)
  unsigned msb = (i_type == U32_TYPE || i_type == S32_TYPE) ? 31 : 63; // [한국어] 타입별 최상위 비트 위치: 32비트=31, 64비트=63
  const operand_info &dst = pI->dst();   // [한국어] 추출 결과를 저장할 목적지 피연산자
  const operand_info &src1 = pI->src1(); // [한국어] 비트 추출 대상 소스 값 피연산자 (a)
  const operand_info &src2 = pI->src2(); // [한국어] 추출 시작 비트 위치 피연산자 (b = pos)
  const operand_info &src3 = pI->src3(); // [한국어] 추출할 비트 수 피연산자 (c = len)
  ptx_reg_t src = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] 소스 값(a) 읽기
  ptx_reg_t b = thread->get_operand_value(src2, dst, i_type, thread, 1);   // [한국어] pos 값(b) 읽기
  ptx_reg_t c = thread->get_operand_value(src3, dst, i_type, thread, 1);   // [한국어] len 값(c) 읽기
  ptx_reg_t data; // [한국어] 추출 결과를 담을 레지스터 변수
  unsigned pos = b.u32 & 0xFF; // [한국어] 추출 시작 비트 위치를 8비트로 클램프 (0~255 범위)
  unsigned len = c.u32 & 0xFF; // [한국어] 추출할 비트 수를 8비트로 클램프 (0이면 결과는 0)
  switch (i_type) {
    case U32_TYPE: {
      // [한국어] 부호 없는 32비트: 오른쪽 시프트 후 len 길이 마스크로 제로 확장
      unsigned mask;
      data.u32 = src.u32 >> pos;            // [한국어] pos부터 추출하기 위해 오른쪽 시프트
      mask = 0xFFFFFFFF >> (32 - len);      // [한국어] len 비트짜리 마스크 생성 (상위 비트 제거용)
      data.u32 &= mask;                     // [한국어] 마스크 적용 — len 비트만 남기고 나머지 0으로 클리어
      break;
    }
    case U64_TYPE: {
      // [한국어] 부호 없는 64비트: 오른쪽 시프트 후 len 길이 마스크로 제로 확장
      unsigned long mask;
      data.u64 = src.u64 >> pos;                    // [한국어] pos부터 추출하기 위해 오른쪽 시프트
      mask = 0xFFFFFFFFFFFFFFFF >> (64 - len);      // [한국어] 64비트 len 마스크 생성
      data.u64 &= mask;                             // [한국어] 마스크 적용 — len 비트만 보존
      break;
    }
    case S32_TYPE: {
      // [한국어] 부호 있는 32비트: 오른쪽 시프트 후 부호 비트로 sign-extension
      unsigned mask;
      unsigned min = MY_MIN_I(pos + len - 1, msb);  // [한국어] 추출 범위의 최상위 비트 위치 (msb=31을 초과하지 않도록 클램프)
      unsigned sbit = len == 0 ? 0 : (src.s32 >> min) & 0x1; // [한국어] 추출된 필드의 최상위 비트 (부호 비트 결정)
      data.s32 = src.s32 >> pos; // [한국어] pos부터 추출하기 위해 오른쪽 시프트
      if (sbit > 0) {
        // [한국어] 추출된 부호 비트가 1 → 음수: 상위 비트를 모두 1로 채워 부호 확장
        mask = 0xFFFFFFFF << len;
        data.s32 |= mask;
      } else {
        // [한국어] 추출된 부호 비트가 0 → 양수: 상위 비트를 모두 0으로 제거
        mask = 0xFFFFFFFF >> (32 - len);
        data.s32 &= mask;
      }
      break;
    }
    case S64_TYPE: {
      // [한국어] 부호 있는 64비트: 오른쪽 시프트 후 부호 비트로 sign-extension
      unsigned long mask;
      unsigned min = MY_MIN_I(pos + len - 1, msb);  // [한국어] 추출 범위의 최상위 비트 위치 클램프 (msb=63)
      unsigned sbit = len == 0 ? 0 : (src.s64 >> min) & 0x1; // [한국어] 추출된 필드의 최상위 비트 (부호 결정)
      data.s64 = src.s64 >> pos; // [한국어] pos부터 추출하기 위해 오른쪽 시프트
      if (sbit > 0) {
        // [한국어] 음수 부호 확장: 상위 비트를 모두 1로 설정
        mask = 0xFFFFFFFFFFFFFFFF << len;
        data.s64 |= mask;
      } else {
        // [한국어] 양수: 상위 비트를 모두 0으로 제거
        mask = 0xFFFFFFFFFFFFFFFF >> (64 - len);
        data.s64 &= mask;
      }
      break;
    }
    default:
      printf("Operand type not supported for BFE instruction.\n"); // [한국어] bfe는 U32/U64/S32/S64만 지원
      abort();
      return;
  }
  thread->set_operand_value(dst, data, i_type, thread, pI); // [한국어] 추출 결과를 dst 레지스터에 기록
}

/*
 * [한국어]
 * bfi_impl - PTX `bfi` 명령어 구현: 비트 필드 삽입 (Bit Field Insert)
 *
 * @pI: 실행 중인 PTX 명령어 객체. 피연산자 타입(B32/B64), dst/src1~4 정보 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 레지스터 파일 읽기/쓰기에 사용.
 *
 * PTX `bfi.type d, a, b, c, d` 명령어:
 *   src2(b)에 src1(a)의 하위 len(d)비트를 bit position src3(c)(pos)부터 삽입한다.
 *   d = src2; d[pos+i] = src1[i] for i in 0..len-1.
 * 처리 방식: 비트 단위 루프로 src2의 해당 비트를 클리어한 뒤 src1의 비트를 OR로 삽입.
 * pos+i가 max(32 또는 64)를 초과하면 삽입을 중단한다.
 * B32/B64 타입만 지원 (부호 없는 비트 연산).
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: get_operand_value(), set_operand_value() — 레지스터 파일 접근.
 * 에러 경로: 지원하지 않는 타입이면 printf 출력 후 assert(0).
 *
 * 호출 체인:
 *   ptx_exec_inst() → [bfi_impl] → get_operand_value() / set_operand_value()
 */
void bfi_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  int i, max; // [한국어] i: 루프 인덱스, max: 타입별 비트 폭 (32 또는 64)
  ptx_reg_t src1_data, src2_data; // [한국어] src1(삽입할 비트 소스), src2(삽입 대상 베이스 값)
  ptx_reg_t src3_data, src4_data, data; // [한국어] src3(pos), src4(len), data(결과 값)

  const operand_info &dst =
      pI->dst();  // get operand info of sources and destination  // [한국어] 결과를 저장할 목적지 피연산자
  const operand_info &src1 =
      pI->src1();  // use them to determine that they are of type 'register'  // [한국어] 삽입할 비트 값 소스(a) 피연산자
  const operand_info &src2 = pI->src2(); // [한국어] 삽입 대상 베이스 값(b) 피연산자
  const operand_info &src3 = pI->src3(); // [한국어] 삽입 시작 비트 위치(c = pos) 피연산자
  const operand_info &src4 = pI->src4(); // [한국어] 삽입할 비트 수(d = len) 피연산자

  unsigned i_type = pI->get_type(); // [한국어] 피연산자 타입 (B32 또는 B64만 지원)
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] 삽입할 비트 소스 값(a) 읽기
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1); // [한국어] 삽입 대상 베이스 값(b) 읽기
  src3_data = thread->get_operand_value(src3, dst, i_type, thread, 1); // [한국어] pos 값(c) 읽기
  src4_data = thread->get_operand_value(src4, dst, i_type, thread, 1); // [한국어] len 값(d) 읽기

  switch (i_type) {
    case B32_TYPE:
      max = 32; // [한국어] 32비트 타입: 비트 폭 상한 32
      break;
    case B64_TYPE:
      max = 64; // [한국어] 64비트 타입: 비트 폭 상한 64
      break;
    default:
      printf("Execution error: type mismatch with instruction\n"); // [한국어] bfi는 B32/B64만 지원 — 타입 불일치 오류
      assert(0);
      break;
  }
  data = src2_data; // [한국어] 결과 초기값은 src2(베이스 값)으로 설정 — 삽입 범위 외 비트는 src2 값 유지
  unsigned pos = src3_data.u32 & 0xFF; // [한국어] 삽입 시작 비트 위치를 8비트로 클램프
  unsigned len = src4_data.u32 & 0xFF; // [한국어] 삽입할 비트 수를 8비트로 클램프
  for (i = 0; i < len && pos + i < max; i++) {
    // [한국어] 비트 단위 삽입 루프: src2의 (pos+i)번째 비트를 클리어하고 src1의 i번째 비트를 삽입
    data.u32 = (~((0x00000001) << (pos + i))) & data.u32;       // [한국어] 삽입 위치(pos+i)의 비트를 클리어 (해당 비트만 0으로)
    data.u32 = data.u32 | ((src1_data.u32 & ((0x00000001) << (i))) << (pos)); // [한국어] src1의 i번째 비트를 (pos+i) 위치에 삽입
  }
  thread->set_operand_value(dst, data, i_type, thread, pI); // [한국어] 비트 삽입 결과를 dst 레지스터에 기록
}
/*
 * [한국어]
 * bfind_impl - PTX `bfind` 명령어 구현: 최상위 1 비트 위치 탐색 (Bit Find)
 *
 * @pI: 실행 중인 PTX 명령어 객체. 피연산자 타입(U32/U64/S32/S64), dst/src1 정보 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 레지스터 파일 읽기/쓰기에 사용.
 *
 * PTX `bfind.type d, a` 명령어:
 *   소스 a에서 MSB(Most Significant Bit) 방향에서 가장 먼저 만나는 1 비트의 위치를 반환한다.
 *   결과는 항상 U32_TYPE으로 저장된다. 1 비트가 없으면 0xFFFFFFFF를 반환.
 * 부호 있는 타입(S32/S64)에서 음수 입력은 비트 반전(~a)하여 MSB를 탐색한다:
 *   음수의 이진 보수에서 MSB(가장 유의미한 1 비트) 위치를 찾기 위해 반전.
 * 탐색 방향: msb(31 또는 63)부터 0 방향으로 내려오며 최초로 1인 비트 위치를 반환.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: get_operand_value(), set_operand_value() — 레지스터 파일 접근.
 * 에러 경로: 지원하지 않는 타입이면 assert(false) 후 abort().
 *
 * 호출 체인:
 *   ptx_exec_inst() → [bfind_impl] → get_operand_value() / set_operand_value()
 */
void bfind_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();   // [한국어] 결과를 저장할 목적지 피연산자 (항상 U32 결과)
  const operand_info &src1 = pI->src1(); // [한국어] MSB 탐색 대상 소스 값 피연산자
  const unsigned i_type = pI->get_type(); // [한국어] 피연산자 타입 (U32/U64/S32/S64)

  const ptx_reg_t src1_data =
      thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] 소스 값 읽기
  const int msb = (i_type == U32_TYPE || i_type == S32_TYPE) ? 31 : 63; // [한국어] 타입별 최상위 비트 위치 (32비트=31, 64비트=63)

  unsigned long a = 0; // [한국어] 타입별 부호 변환 후 탐색에 사용할 64비트 작업 변수
  switch (i_type) {
    case S32_TYPE:
      a = src1_data.s32; // [한국어] 부호 있는 32비트 값을 unsigned long으로 가져옴 (음수 처리 위해 후속 반전)
      break;
    case U32_TYPE:
      a = src1_data.u32; // [한국어] 부호 없는 32비트 값 그대로 사용
      break;
    case S64_TYPE:
      a = src1_data.s64; // [한국어] 부호 있는 64비트 값을 unsigned long으로 가져옴
      break;
    case U64_TYPE:
      a = src1_data.u64; // [한국어] 부호 없는 64비트 값 그대로 사용
      break;
    default:
      assert(false); // [한국어] bfind는 U32/U64/S32/S64만 지원 — 다른 타입은 버그
      abort();
  }

  // negate negative signed inputs
  if ((i_type == S32_TYPE || i_type == S64_TYPE) && (a & (1 << msb))) {
    // [한국어] 부호 있는 타입에서 음수(MSB가 1)이면 비트 반전 — 음수의 절댓값에 해당하는 패턴으로 탐색
    a = ~a;
  }
  uint32_t d_data = 0xffffffff; // [한국어] 초기값 0xFFFFFFFF — 1 비트를 못 찾은 경우 반환할 값
  for (uint32_t i = msb; i >= 0; i--) {
    // [한국어] MSB부터 0 방향으로 내려오며 최초 1 비트 위치 탐색
    if (a & (1 << i)) {
      d_data = i; // [한국어] 1 비트 발견: 위치(0-based)를 결과로 저장
      break;      // [한국어] 최상위 1 비트를 찾았으므로 탐색 종료
    }
  }

  // if (.shiftamt && d != 0xffffffff)  { d = msb - d; }

  // store d
  thread->set_operand_value(dst, d_data, U32_TYPE, thread, pI); // [한국어] MSB 위치(또는 0xFFFFFFFF)를 U32 타입으로 dst 레지스터에 기록
}

/*
 * [한국어]
 * bra_impl - PTX `bra` 명령어 구현: 직접 분기 (Branch)
 *
 * @pI: 실행 중인 PTX 명령어 객체. 분기 목적지 레이블/주소를 dst 피연산자로 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 분기 플래그 설정 및 다음 PC 변경에 사용.
 *
 * PTX `bra label` 명령어: 목적지 레이블의 PC를 읽어 스레드의 다음 실행 PC로 설정한다.
 * SIMT 스택 기반 분기 처리에서 이 함수는 분기 실행 의사를 m_branch_taken으로 표시하고,
 * set_npc()로 다음 PC를 갱신한다. 실제 warp 수준의 분기 처리(SIMT stack 업데이트)는
 * 상위 레이어(warp scheduler / SIMT stack)에서 active mask와 함께 처리된다.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: get_operand_value(), thread->set_npc() — 다음 PC 설정.
 * 에러 경로: 없음.
 *
 * 호출 체인:
 *   ptx_exec_inst() → [bra_impl] → get_operand_value() / set_npc()
 */
void bra_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &target = pI->dst(); // [한국어] 분기 목적지 레이블 피연산자 정보
  ptx_reg_t target_pc =
      thread->get_operand_value(target, target, U32_TYPE, thread, 1); // [한국어] 목적지 레이블의 PC 값 읽기 (PTX 기능 시뮬레이션에서 레이블은 PC로 변환됨)

  thread->m_branch_taken = true; // [한국어] 이 스레드가 분기를 실행함을 표시 (SIMT 스택이 참조)
  thread->set_npc(target_pc);    // [한국어] 스레드의 다음 실행 PC를 목적지 주소로 설정
}

/*
 * [한국어]
 * brx_impl - PTXPlus `brx` 명령어 구현: 간접 분기 (Indirect Branch)
 *
 * @pI: 실행 중인 PTX 명령어 객체. 분기 목적지 주소를 담은 레지스터를 dst로 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 분기 플래그 설정 및 다음 PC 변경에 사용.
 *
 * PTXPlus `brx reg` 명령어: bra_impl과 동일한 동작이지만 직접 레이블 대신
 * 레지스터에 저장된 주소로 간접 분기한다.
 * dst 피연산자에서 레지스터 값을 읽어 목적지 PC로 사용한다.
 * 이 명령어는 함수 포인터, 가상 디스패치 등의 동적 분기를 시뮬레이션할 때 사용된다.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: get_operand_value(), thread->set_npc() — 다음 PC 설정.
 * 에러 경로: 없음.
 *
 * 호출 체인:
 *   ptx_exec_inst() → [brx_impl] → get_operand_value() / set_npc()
 */
void brx_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &target = pI->dst(); // [한국어] 간접 분기 목적지 주소를 담은 레지스터 피연산자 정보
  ptx_reg_t target_pc =
      thread->get_operand_value(target, target, U32_TYPE, thread, 1); // [한국어] 레지스터에서 목적지 PC 주소 값 읽기

  thread->m_branch_taken = true; // [한국어] 이 스레드가 분기를 실행함을 표시
  thread->set_npc(target_pc);    // [한국어] 다음 실행 PC를 레지스터에서 읽은 목적지 주소로 설정
}

/*
 * [한국어]
 * break_impl - PTX `break` 명령어 구현: 루프 탈출 분기
 *
 * @pI: 실행 중인 PTX 명령어 객체 (break는 피연산자 없음 — 사용하지 않음).
 * @thread: 시뮬레이션 중인 스레드 상태. 브레이크 주소 스택 팝 및 PC 변경에 사용.
 *
 * PTX `break` 명령어: 스레드의 브레이크 주소 스택에서 최상위 목적지 주소를 팝하고
 * 해당 주소로 분기한다. breakaddr_impl()이 루프 탈출 목적지를 미리 스택에 푸시해 두면
 * break_impl()이 실행될 때 이를 팝하여 루프 밖으로 점프한다.
 * 이 메커니즘은 GPU 루프 구조에서 `__break` 키워드를 시뮬레이션한다.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: thread->pop_breakaddr(), get_operand_value(), thread->set_npc().
 * 에러 경로: 스택이 비어있으면 pop_breakaddr에서 오류 발생.
 *
 * 호출 체인:
 *   ptx_exec_inst() → [break_impl] → pop_breakaddr() / set_npc()
 */
void break_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &target = thread->pop_breakaddr(); // [한국어] 브레이크 주소 스택에서 루프 탈출 목적지 피연산자 팝
  ptx_reg_t target_pc =
      thread->get_operand_value(target, target, U32_TYPE, thread, 1); // [한국어] 팝한 피연산자에서 목적지 PC 값 읽기

  thread->m_branch_taken = true; // [한국어] 이 스레드가 분기(루프 탈출)를 실행함을 표시
  thread->set_npc(target_pc);    // [한국어] 다음 실행 PC를 루프 탈출 목적지로 설정
}

/*
 * [한국어]
 * breakaddr_impl - PTX `breakaddr` 명령어 구현: 루프 탈출 주소 스택 푸시
 *
 * @pI: 실행 중인 PTX 명령어 객체. 루프 탈출 목적지 레이블을 dst 피연산자로 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 브레이크 주소 스택에 목적지 푸시.
 *
 * PTX `breakaddr label` 명령어: 루프 탈출 목적지 레이블을 스레드의 브레이크 주소 스택에 푸시한다.
 * 이후 `break` 명령어(break_impl)가 실행될 때 이 주소를 팝하여 루프 밖으로 점프한다.
 * PDOM(Post-Dominator) 분석이 이 명령어를 술어 실행(predicated)으로 처리하지 못하기 때문에
 * has_pred() == false를 assert로 강제한다 — 조건부 breakaddr는 지원하지 않는다.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: thread->push_breakaddr() — 브레이크 주소 스택 푸시.
 * 에러 경로: 술어 실행이면 assert 실패.
 *
 * 호출 체인:
 *   ptx_exec_inst() → [breakaddr_impl] → push_breakaddr()
 */
void breakaddr_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &target = pI->dst(); // [한국어] 루프 탈출 목적지 레이블 피연산자 정보
  thread->push_breakaddr(target);         // [한국어] 브레이크 주소 스택에 목적지 피연산자 푸시 (break_impl이 팝하여 사용)
  assert(
      pI->has_pred() ==
      false);  // pdom analysis cannot handle if this instruction is predicated
  // [한국어] breakaddr가 술어 실행이면 PDOM 분석이 올바르게 동작하지 않으므로 반드시 비술어 명령어여야 함
}

/*
 * [한국어]
 * brev_impl - PTX `brev` 명령어 구현: 비트 순서 반전 (Bit Reverse)
 *
 * @pI: 실행 중인 PTX 명령어 객체. 피연산자 타입(B32/B64), dst/src1 정보 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 레지스터 파일 읽기/쓰기에 사용.
 *
 * PTX `brev.type d, a` 명령어: 소스 값(a)의 비트 순서를 완전히 반전하여 dst에 저장한다.
 * 즉, bit[0]↔bit[31] (B32) 또는 bit[0]↔bit[63] (B64) 등 대칭 위치의 비트를 교환.
 * 구현 방식: i번째 비트가 1이면 결과의 (msb-i)번째 비트를 1로 설정하는 루프.
 * 주로 CRC 계산, 해시, 암호화 등 비트 조작 알고리즘에서 사용된다.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: get_operand_value(), set_operand_value() — 레지스터 파일 접근.
 * 에러 경로: B32/B64 이외의 타입이면 assert(0).
 *
 * 호출 체인:
 *   ptx_exec_inst() → [brev_impl] → get_operand_value() / set_operand_value()
 */
void brev_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, data; // [한국어] 소스 값과 비트 반전 결과를 저장할 레지스터 변수
  const operand_info &dst = pI->dst();   // [한국어] 반전 결과를 저장할 목적지 피연산자
  const operand_info &src1 = pI->src1(); // [한국어] 비트 반전 대상 소스 피연산자
  unsigned i_type = pI->get_type(); // [한국어] 피연산자 타입 (B32 또는 B64만 지원)
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] 소스 값 읽기

  unsigned msb; // [한국어] 타입별 최상위 비트 위치 (B32=31, B64=63)
  switch (i_type) {
    case B32_TYPE:
      msb = 31; // [한국어] 32비트 타입: 최상위 비트 위치
      for (unsigned i = 0; i <= msb; i++) {
        // [한국어] i번째 비트가 1이면 결과의 대칭 위치(msb-i)에 1 설정
        if ((src1_data.u32 & (1 << i))) data.u32 |= 1 << (msb - i);
      }
      break;
    case B64_TYPE:
      msb = 63; // [한국어] 64비트 타입: 최상위 비트 위치
      for (unsigned i = 0; i <= msb; i++) {
        // [한국어] 64비트 비트 반전: i번째 비트를 (msb-i) 위치에 설정
        if ((src1_data.u64 & (1 << i))) data.u64 |= 1 << (msb - i);
      }
      break;
    default:
      assert(0); // [한국어] brev는 B32/B64만 지원 — 다른 타입은 버그
  }
  thread->set_operand_value(dst, data, i_type, thread, pI); // [한국어] 비트 반전 결과를 dst 레지스터에 기록
}
/*
 * [한국어]
 * brkpt_impl - PTX `brkpt` 명령어 구현: 디버그 브레이크포인트 (미지원)
 *
 * @pI: 실행 중인 PTX 명령어 객체.
 * @thread: 시뮬레이션 중인 스레드 상태 (이 함수에서는 사용하지 않음).
 *
 * PTX `brkpt` 명령어는 GPU 디버거 브레이크포인트를 설정하는 명령어이다.
 * GPGPU-Sim 기능 시뮬레이션에서는 이 명령어를 지원하지 않으며,
 * inst_not_implemented()를 호출하여 미지원 명령어임을 보고한다.
 * 실제 GPU 하드웨어에서는 CUDA 디버거가 이 지점에서 실행을 중단한다.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: inst_not_implemented() — 미지원 명령어 처리 (abort 가능).
 * 에러 경로: inst_not_implemented() 내부에서 오류 출력 및 abort.
 *
 * 호출 체인:
 *   ptx_exec_inst() → [brkpt_impl] → inst_not_implemented()
 */
void brkpt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI); // [한국어] brkpt는 GPGPU-Sim에서 미구현 — 미지원 명령어 처리 함수 호출
}

/*
 * [한국어]
 * trunc - 부호 없는 정수를 주어진 비트 정밀도로 절삭 (내부 헬퍼 함수)
 *
 * @num: 절삭 대상 부호 없는 정수 값.
 * @precision: 유지할 비트 수 (목표 정밀도). 이 비트 수를 초과하는 상위 비트를 제거.
 * @return: precision 비트 이내로 절삭된 결과값.
 *
 * PTX 명령어 구현 함수가 아닌 내부 헬퍼 함수로, wmma(Tensor Core MMA) 연산에서
 * FP16 값을 특정 비트 정밀도로 맞출 때 사용된다.
 * 알고리즘:
 *   1) 전체 비트를 순회하며 최상위 1 비트의 위치(latest_one)를 찾는다.
 *   2) latest_one >= precision이면 precision+1 비트로 맞추기 위해 오른쪽 시프트.
 * 주석 처리된 반올림 코드는 더 정교한 반올림을 의도했으나 단순 절삭으로 대체됨.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, mma_impl()에서 호출.
 * 호출자: wmma 관련 코드 (현재는 직접 호출 없음 — 미래 확장용 또는 참고용).
 * 피호출자: 없음 (순수 비트 조작 연산).
 * 에러 경로: 없음.
 *
 * 호출 체인:
 *   mma_impl() 등 wmma 관련 함수 → [trunc]
 */
unsigned trunc(unsigned num, unsigned precision) {
  int mask = 1, latest_one = -1; // [한국어] mask: 비트 추출 마스크, latest_one: 발견된 최상위 1 비트 위치 (-1=없음)
  unsigned data = num; // [한국어] 비트 순회를 위해 num의 복사본 (루프에서 오른쪽 시프트)
  for (unsigned j = 0; j < sizeof(unsigned) * 8; j++) {
    // [한국어] 모든 비트를 순회하며 최상위 1 비트 위치 추적
    int bit = data & mask; // [한국어] 현재 최하위 비트 추출 (마스크 = 1)
    if (bit == 1) latest_one = j; // [한국어] 1 비트 발견 시 그 위치(j)를 최신 1 비트로 업데이트
    data >>= 1; // [한국어] 다음 비트 위치로 이동 (오른쪽 시프트로 비트 순환)
  }
  if (latest_one >= precision) {
    // round_up is 1 if the most significant truncated digit is a 1, otherwise
    // it is 0
    // int round_up = (num & (1 << (latest_one-precision))) >>
    // (latest_one-precision); unsigned shifted_output = num >>
    // (latest_one-precision+1);
    // if shifted_output is a number like 1111, don't round up
    // if (shifted_output == (pow(2,precision)-1)) round_up = 0;
    // num = shifted_output + round_up;
    // [한국어] 최상위 1 비트가 precision을 초과하면 precision 비트 폭으로 축소하기 위해 시프트
    num >>= (latest_one - precision + 1); // [한국어] 오른쪽 시프트로 precision 비트 이내로 절삭 (반올림 없이 단순 truncation)
  }
  return num; // [한국어] 절삭된 결과 반환 (precision 비트 이내)
}
/*
 * [한국어]
 * mapping - wmma 연산에서 스레드 ID와 피연산자 인덱스를 행렬 (row, col) 좌표로 변환
 *
 * @thread: warp 내 스레드 번호 (0~31). 행렬 원소 소유권 결정에 사용.
 * @wmma_type: wmma 피연산자 종류 (LOAD_A / LOAD_B / LOAD_C). 행렬 역할 구분.
 * @wmma_layout: 행렬 배치 방식 (ROW = 행 우선 / COL = 열 우선).
 * @type: 원소 데이터 타입 (F16_TYPE / F32_TYPE 등). C 행렬 레이아웃 결정에 사용.
 * @index: 이 스레드가 담당하는 피연산자 내 원소 인덱스 (0~7 또는 0~nelem-1).
 * @stride: 행렬의 실제 행 폭 (행렬 메모리 배치에서의 stride, 보통 16).
 * @row: [출력] 계산된 행렬 행 좌표 (0~15).
 * @col: [출력] 계산된 행렬 열 좌표 (0~15).
 * @assg_offset: [출력] 스레드 레지스터 내 원소 할당 오프셋 (레지스터 벡터 인덱스).
 *
 * NVIDIA Tensor Core(wmma)는 16×16 행렬 연산에서 warp의 32개 스레드가
 * 행렬 원소를 분산 저장한다. 이 함수는 특정 스레드의 특정 레지스터 원소가
 * 16×16 행렬의 어느 (row, col) 위치에 해당하는지 계산한다.
 * - LOAD_A: 행(ROW) 또는 열(COL) 우선으로 A 행렬 원소 위치 결정
 * - LOAD_B: A와 반대 레이아웃으로 B 행렬 원소 위치 결정
 * - LOAD_C: F16/F32 타입별로 하드코딩된 오프셋 테이블로 C 행렬 원소 위치 결정
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, mma_impl()의 warp 루프 내 호출.
 * 호출자: mma_impl() — 행렬 집계/산란 단계.
 * 피호출자: thread_group_offset() — 스레드 그룹별 기반 오프셋 계산.
 * 에러 경로: 없음 (잘못된 wmma_type이면 row/col/assg_offset이 미초기화).
 *
 * 호출 체인:
 *   mma_impl() → [mapping] → thread_group_offset()
 */
void mapping(int thread, int wmma_type, int wmma_layout, int type, int index,
             int stride, int &row, int &col, int &assg_offset) {
  int offset; // [한국어] 행렬 원소의 선형 오프셋 (행렬 메모리 상 위치)
  // [한국어] C 행렬 원소 위치 결정을 위한 하드코딩된 오프셋 테이블들
  //   스레드 그룹(thread/4)별로 C 행렬에서의 행/열 기반 오프셋을 제공
  int c_row_offset[] = {0, 8, 0, 8, 4, 12, 4, 12};         // [한국어] thread/4에 따른 C 행렬 행 기반 오프셋
  int c_col_offset[] = {0, 0, 8, 8, 0, 0, 8, 8};            // [한국어] thread/4에 따른 C 행렬 열 기반 오프셋
  int c_tg_inside_row_offset[] = {0, 1, 0, 1};              // [한국어] 스레드 그룹 내 위치(thread%4)에 따른 행 오프셋
  int c_tg_inside_col_offset[] = {0, 0, 2, 2};              // [한국어] 스레드 그룹 내 위치(thread%4)에 따른 열 오프셋
  int c_inside_row_offset[] = {0, 0, 2, 2, 0, 0, 2, 2};     // [한국어] 원소 인덱스(index)에 따른 C 행렬 내부 행 오프셋
  int c_inside_col_offset[] = {0, 1, 0, 1, 4, 5, 4, 5};     // [한국어] 원소 인덱스(index)에 따른 C 행렬 내부 열 오프셋

  offset = thread_group_offset(thread, wmma_type, wmma_layout, type, stride); // [한국어] 스레드 그룹의 기반 선형 오프셋 계산

  if (wmma_type == LOAD_A) {
    // [한국어] A 행렬 원소 위치 계산
    if (wmma_layout == ROW) {
      // [한국어] 행 우선 배치: index와 상위 하프 warp 오프셋으로 선형 위치 계산
      offset += index + 8 * ((thread % 16) / 8);
    } else {
      // [한국어] 열 우선 배치: index를 4단위 그룹으로 나눠 열 우선 선형 위치 계산
      offset += 64 * (index / 4) + index % 4 + 128 * ((thread % 16) / 8);
    }
    offset = (offset / 16) * stride + offset % 16; // [한국어] 선형 오프셋을 stride를 적용한 실제 메모리 위치로 변환
    assg_offset = index + 8 * ((thread % 16) / 8); // [한국어] 레지스터 벡터 내 이 원소의 할당 오프셋
  } else if (wmma_type == LOAD_B) {
    // [한국어] B 행렬 원소 위치 계산 (A와 레이아웃 역전)
    if (wmma_layout == ROW) {
      // [한국어] B의 행 우선: A의 열 우선과 동일한 공식 적용 (행렬 곱 전치 관계)
      offset += 64 * (index / 4) + index % 4 + 128 * ((thread % 16) / 8);
    } else {
      // [한국어] B의 열 우선: A의 행 우선과 동일한 공식 적용
      offset += index + 8 * ((thread % 16) / 8);
    }
    offset = (offset / 16) * stride + offset % 16; // [한국어] stride 적용한 실제 메모리 위치로 변환
    assg_offset = index + 8 * ((thread % 16) / 8); // [한국어] 레지스터 벡터 내 이 원소의 할당 오프셋
  } else if (wmma_type == LOAD_C) {
    // [한국어] C(출력) 행렬 원소 위치 계산 — 하드코딩 테이블로 결정
    if (type == F16_TYPE) {
      // [한국어] F16 타입: 스레드 그룹(thread/4)과 그룹 내 위치(thread%4)로 행/열 결정
      row = c_row_offset[thread / 4] + thread % 4;  // [한국어] F16: 행 = 그룹 기반 오프셋 + 그룹 내 위치
      col = c_col_offset[thread / 4] + index;        // [한국어] F16: 열 = 그룹 기반 오프셋 + 원소 인덱스
    } else {
      // [한국어] F32 타입: 스레드 그룹 + 그룹 내 위치 + 원소 인덱스로 더 세밀한 행/열 결정
      row = c_row_offset[thread / 4] + c_tg_inside_row_offset[thread % 4] +
            c_inside_row_offset[index]; // [한국어] F32 행: 그룹 기반 + 그룹 내 행 + 원소 내부 행 오프셋
      col = c_col_offset[thread / 4] + c_tg_inside_col_offset[thread % 4] +
            c_inside_col_offset[index]; // [한국어] F32 열: 그룹 기반 + 그룹 내 열 + 원소 내부 열 오프셋
    }
    assg_offset = index; // [한국어] C 행렬의 레지스터 벡터 내 할당 오프셋은 원소 인덱스와 동일
  }

  if (wmma_type == LOAD_A || wmma_type == LOAD_B) {
    // [한국어] A/B 행렬의 경우 선형 오프셋에서 16×16 행렬의 (row, col) 좌표로 변환
    if (wmma_layout == ROW) {
      // [한국어] 행 우선 배치: 선형 오프셋 / 16 = 행, % 16 = 열
      row = offset / 16;
      col = offset % 16;
    } else {
      // [한국어] 열 우선 배치: 선형 오프셋 / 16 = 열, % 16 = 행 (전치)
      col = offset / 16;
      row = offset % 16;
    }
  }
}

/*
 * [한국어]
 * mma_impl - PTX wmma.mma 명령어 구현: Tensor Core 행렬 곱 누산 (D = A×B + C)
 *
 * @pI: 실행 중인 PTX 명령어 객체. wmma 레이아웃(A/B), 결과 타입(type), 입력 타입(type2), 피연산자 정보 포함.
 * @core: 이 warp를 실행 중인 SM 코어 객체. 스레드 정보 배열 접근에 사용.
 * @inst: warp 명령어 객체. warp ID와 warp 크기 정보 포함.
 *
 * NVIDIA Tensor Core(wmma API)의 16×16×16 행렬 곱 누산 연산을 시뮬레이션한다:
 *   D[i][j] = sum_k(A[i][k] * B[k][j]) + C[i][j]
 * warp의 32개 스레드가 A, B, C 행렬의 원소를 분산 저장하므로 이 함수는
 * 다른 impl 함수들과 달리 단일 스레드가 아니라 warp 전체를 처리한다.
 *
 * 처리 단계:
 *   1) 집계(Gather): 32개 스레드에서 A, B, C 행렬 원소를 16×16 행렬로 수집
 *      - A, B 원소: FP16 두 개가 하나의 32비트 레지스터에 팩킹 → 언팩킹 후 nw_v에 저장
 *      - C 원소: type2에 따라 F16 또는 F32로 읽기
 *      - mapping()으로 각 스레드의 원소가 행렬의 어느 (row,col)에 해당하는지 결정
 *   2) 계산: FP16 중간 누산으로 D = A×B 수행 후 C를 type 조합에 따라 누산
 *      - type/type2 조합: (F16,F16), (F32,F16), (F16,F32), (F32,F32)
 *   3) 산란(Scatter): 결과 행렬 D의 원소를 다시 32개 스레드의 레지스터에 기록
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, warp 단위로 한 번 호출(스레드 루프 내부).
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 (warp 레벨 처리).
 * 피호출자: get_vector_operand_values(), mapping(), set_wmma_vector_operand_values(), set_vector_operand_values().
 * 에러 경로: 잘못된 type 조합이면 printf 후 abort().
 *
 * 호출 체인:
 *   ptx_exec_inst() → [mma_impl] → mapping() / get/set_vector_operand_values()
 */
void mma_impl(const ptx_instruction *pI, core_t *core, warp_inst_t inst) {
  int i, j, k, thrd; // [한국어] 루프 인덱스: i=행, j=열, k=내적 차원, thrd=스레드 번호
  int row, col, offset; // [한국어] mapping() 출력: 행렬 좌표 및 레지스터 오프셋
  ptx_reg_t matrix_a[16][16]; // [한국어] 16×16 A 행렬 버퍼 — 32개 스레드에서 집계한 원소
  ptx_reg_t matrix_b[16][16]; // [한국어] 16×16 B 행렬 버퍼
  ptx_reg_t matrix_c[16][16]; // [한국어] 16×16 C 행렬 버퍼 (누산 초기값)
  ptx_reg_t matrix_d[16][16]; // [한국어] 16×16 D 결과 행렬 버퍼 (A×B+C 결과)
  ptx_reg_t src_data;         // [한국어] 소스 피연산자 읽기용 임시 변수
  ptx_thread_info *thread;    // [한국어] 현재 처리 중인 스레드의 상태 포인터

  unsigned a_layout = pI->get_wmma_layout(0); // [한국어] A 행렬의 메모리 배치 (ROW 또는 COL)
  unsigned b_layout = pI->get_wmma_layout(1); // [한국어] B 행렬의 메모리 배치
  unsigned type = pI->get_type();   // [한국어] 결과 D의 원소 타입 (F16 또는 F32)
  unsigned type2 = pI->get_type2(); // [한국어] 누산기 C의 원소 타입 (F16 또는 F32)
  int tid;                          // [한국어] warp의 첫 번째 스레드 전역 ID
  const operand_info &dst = pI->operand_lookup(0); // [한국어] 결과 D를 저장할 목적지 피연산자

  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size(); // [한국어] 기능 시뮬레이션 모드: 기능 시뮬용 warp ID로 스레드 기반 계산
  else
    tid = inst.warp_id() * core->get_warp_size(); // [한국어] 타이밍 시뮬레이션 모드: 타이밍용 warp ID로 스레드 기반 계산
  float temp;  // [한국어] half→float 변환 임시 변수 (디버그 출력 및 타입 변환용)
  half temp2;  // [한국어] FP16 중간 누산 임시 변수

  for (thrd = 0; thrd < core->get_warp_size(); thrd++) {
    // [한국어] 단계 1: warp의 32개 스레드에서 A, B, C 행렬 원소를 집계
    thread = core->get_thread_info()[tid + thrd]; // [한국어] thrd번째 스레드의 상태 포인터 가져오기
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
      printf("THREAD=%d\n:", thrd); // [한국어] 디버그 모드: 현재 처리 스레드 번호 출력
    for (int operand_num = 1; operand_num <= 3; operand_num++) {
      // [한국어] 피연산자 1(A), 2(B), 3(C) 순서로 각 스레드의 레지스터 벡터에서 원소 읽기
      const operand_info &src_a = pI->operand_lookup(operand_num); // [한국어] 해당 피연산자 정보 조회
      unsigned nelem = src_a.get_vect_nelem(); // [한국어] 이 피연산자의 벡터 원소 수 (보통 4 또는 8)
      ptx_reg_t v[8]; // [한국어] 이 스레드의 레지스터 벡터 값 (최대 8원소)
      thread->get_vector_operand_values(src_a, v, nelem); // [한국어] 레지스터 파일에서 벡터 원소 읽기
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
        printf("Thread%d_Iteration=%d\n:", thrd, operand_num); // [한국어] 디버그: 스레드/피연산자 번호
        for (k = 0; k < nelem; k++) {
          printf("%llx ", v[k].u64); // [한국어] 디버그: 레지스터 값 16진수 출력
        }
        printf("\n");
      }
      ptx_reg_t nw_v[16]; // [한국어] FP16 언팩킹 결과 — 한 레지스터에 팩킹된 2개의 FP16을 분리
      int hex_val;         // [한국어] FP16 언팩킹 시 임시 16비트 값

      if (!((operand_num == 3) && (type2 == F32_TYPE))) {
        // [한국어] A, B 피연산자 및 F16 타입 C 피연산자: 32비트 레지스터에서 FP16 두 개 언팩킹
        for (k = 0; k < 2 * nelem; k++) {
          if (k % 2 == 1)
            hex_val = (v[k / 2].s64 & 0xffff);             // [한국어] 짝수 FP16: 하위 16비트 추출
          else
            hex_val = ((v[k / 2].s64 & 0xffff0000) >> 16); // [한국어] 홀수 FP16: 상위 16비트를 하위로 이동
          nw_v[k].f16 = *(reinterpret_cast<half *>(hex_val)); // [한국어] 16비트 정수를 FP16(half)으로 재해석
        }
      }
      if (!((operand_num == 3) && (type2 == F32_TYPE))) {
        // [한국어] 디버그: 언팩킹된 FP16 값을 float으로 변환하여 출력
        for (k = 0; k < 2 * nelem; k++) {
          temp = nw_v[k].f16; // [한국어] FP16 → float 변환 (디버그 출력용)
          if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
            printf("%.2f ", temp);
        }
        if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) printf("\n");
      } else {
        // [한국어] F32 타입 C 피연산자: 그대로 float 값 출력 (언팩킹 불필요)
        if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
          for (k = 0; k < 8; k++) {
            printf("%.2f ", v[k].f32); // [한국어] 디버그: F32 C 원소 출력
          }
          printf("\n");
        }
      }
      switch (operand_num) {
        case 1:  // operand 1
          // [한국어] 피연산자 1 = A 행렬: 이 스레드의 8개 원소를 16×16 A 행렬에 배치
          for (k = 0; k < 8; k++) {
            mapping(thrd, LOAD_A, a_layout, F16_TYPE, k, 16, row, col, offset); // [한국어] 스레드 thrd의 k번째 원소가 A 행렬의 (row,col) 위치에 해당하는지 결정
            if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
              printf("A:thread=%d,row=%d,col=%d,offset=%d\n", thrd, row, col,
                     offset); // [한국어] 디버그: A 행렬 원소 위치 출력
            matrix_a[row][col] = nw_v[offset]; // [한국어] 결정된 (row,col) 위치에 FP16 원소 저장
          }
          break;
        case 2:  // operand 2
          // [한국어] 피연산자 2 = B 행렬: 이 스레드의 8개 원소를 16×16 B 행렬에 배치
          for (k = 0; k < 8; k++) {
            mapping(thrd, LOAD_B, b_layout, F16_TYPE, k, 16, row, col, offset); // [한국어] B 행렬 내 (row,col) 위치 결정
            if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
              printf("B:thread=%d,row=%d,col=%d,offset=%d\n", thrd, row, col,
                     offset); // [한국어] 디버그: B 행렬 원소 위치 출력
            matrix_b[row][col] = nw_v[offset]; // [한국어] B 행렬 (row,col) 위치에 FP16 원소 저장
          }
          break;
        case 3:  // operand 3
          // [한국어] 피연산자 3 = C 행렬: 이 스레드의 8개 원소를 16×16 C 행렬에 배치
          for (k = 0; k < 8; k++) {
            mapping(thrd, LOAD_C, ROW, type2, k, 16, row, col, offset); // [한국어] C 행렬 내 (row,col) 위치 결정 (type2로 레이아웃 결정)
            if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
              printf("C:thread=%d,row=%d,col=%d,offset=%d\n", thrd, row, col,
                     offset); // [한국어] 디버그: C 행렬 원소 위치 출력
            if (type2 != F16_TYPE) {
              matrix_c[row][col] = v[offset];   // [한국어] F32 타입: 언팩킹 전 원본 레지스터 값 사용
            } else {
              matrix_c[row][col] = nw_v[offset]; // [한국어] F16 타입: 언팩킹된 FP16 원소 사용
            }
          }
          break;
        default:
          printf("Invalid Operand Index\n"); // [한국어] 피연산자 번호가 1~3 이외 — 버그
      }
    }
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) printf("\n"); // [한국어] 디버그: 스레드 간 구분 빈 줄
  }
  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    // [한국어] 디버그: 집계된 A, B, C 행렬 내용 출력
    printf("MATRIX_A\n");
    for (i = 0; i < 16; i++) {
      for (j = 0; j < 16; j++) {
        temp = matrix_a[i][j].f16; // [한국어] A 행렬 원소를 float으로 변환하여 출력
        printf("%.2f ", temp);
      }
      printf("\n");
    }
    printf("MATRIX_B\n");
    for (i = 0; i < 16; i++) {
      for (j = 0; j < 16; j++) {
        temp = matrix_b[i][j].f16; // [한국어] B 행렬 원소를 float으로 변환하여 출력
        printf("%.2f ", temp);
      }
      printf("\n");
    }
    printf("MATRIX_C\n");
    for (i = 0; i < 16; i++) {
      for (j = 0; j < 16; j++) {
        if (type2 == F16_TYPE) {
          temp = matrix_c[i][j].f16; // [한국어] F16 C: float으로 변환하여 출력
          printf("%.2f ", temp);
        } else
          printf("%.2f ", matrix_c[i][j].f32); // [한국어] F32 C: 직접 float 출력
      }
      printf("\n");
    }
  }
  // [한국어] 단계 2: D 행렬 초기화 (FP16으로 초기화 — 내적 누산은 FP16 중간 결과로 수행)
  for (i = 0; i < 16; i++) {
    for (j = 0; j < 16; j++) {
      matrix_d[i][j].f16 = 0; // [한국어] D[i][j]를 FP16 0으로 초기화
    }
  }

  // [한국어] 단계 2 (계속): D = A×B + C 계산
  for (i = 0; i < 16; i++) {
    for (j = 0; j < 16; j++) {
      for (k = 0; k < 16; k++) {
        // [한국어] FP16 중간 누산: D[i][j] += A[i][k] * B[k][j] (행렬 곱 내적 차원 k)
        matrix_d[i][j].f16 =
            matrix_d[i][j].f16 + matrix_a[i][k].f16 * matrix_b[k][j].f16;
      }
      // [한국어] C 누산: type/type2 조합에 따라 D에 C를 더함
      if ((type == F16_TYPE) && (type2 == F16_TYPE))
        matrix_d[i][j].f16 += matrix_c[i][j].f16; // [한국어] F16 출력 + F16 C: FP16으로 직접 누산
      else if ((type == F32_TYPE) && (type2 == F16_TYPE)) {
        // [한국어] F32 출력 + F16 C: FP16 중간값을 F32로 변환 후 C와 합산
        temp2 = matrix_d[i][j].f16 + matrix_c[i][j].f16; // [한국어] FP16으로 중간 합산
        temp = temp2;                                      // [한국어] half → float 변환
        matrix_d[i][j].f32 = temp;                        // [한국어] F32 결과에 저장
      } else if ((type == F16_TYPE) && (type2 == F32_TYPE)) {
        // [한국어] F16 출력 + F32 C: float으로 합산 후 FP16으로 변환
        temp = matrix_d[i][j].f16; // [한국어] FP16 중간값을 float으로 변환
        temp += matrix_c[i][j].f32; // [한국어] F32 C와 float 합산
        matrix_d[i][j].f16 = half(temp); // [한국어] float 결과를 FP16으로 변환하여 저장
      } else {
        // [한국어] F32 출력 + F32 C: float으로 합산
        temp = matrix_d[i][j].f16;  // [한국어] FP16 중간값을 float으로 변환
        temp += matrix_c[i][j].f32; // [한국어] F32 C와 float 합산
        matrix_d[i][j].f32 = temp;  // [한국어] F32 결과에 저장
      }
    }
  }
  if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
    // [한국어] 디버그: 계산된 D 행렬 내용 출력
    printf("MATRIX_D\n");
    for (i = 0; i < 16; i++) {
      for (j = 0; j < 16; j++) {
        if (type == F16_TYPE) {
          temp = matrix_d[i][j].f16; // [한국어] F16 결과를 float으로 변환하여 출력
          printf("%.2f ", temp);
        } else
          printf("%.2f ", matrix_d[i][j].f32); // [한국어] F32 결과 직접 출력
      }
      printf("\n");
    }
  }
  // [한국어] 단계 3: D 행렬 원소를 다시 32개 스레드의 레지스터에 산란(Scatter)
  for (thrd = 0; thrd < core->get_warp_size(); thrd++) {
    int row_t[8]; // [한국어] 이 스레드가 담당하는 8개 D 원소의 행 좌표 배열
    int col_t[8]; // [한국어] 이 스레드가 담당하는 8개 D 원소의 열 좌표 배열
    for (k = 0; k < 8; k++) {
      mapping(thrd, LOAD_C, ROW, type, k, 16, row_t[k], col_t[k], offset); // [한국어] D 결과 행렬에서 이 스레드의 k번째 원소 위치(row_t, col_t) 결정
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
        printf("mma:store:row:%d,col%d\n", row_t[k], col_t[k]); // [한국어] 디버그: 저장 위치 출력
    }
    thread = core->get_thread_info()[tid + thrd]; // [한국어] 이 스레드의 상태 포인터 다시 가져오기 (산란 단계용)

    if (type == F32_TYPE) {
      // [한국어] F32 결과: 8개 D 원소를 wmma 전용 F32 벡터 레지스터에 직접 저장
      thread->set_wmma_vector_operand_values(
          dst, matrix_d[row_t[0]][col_t[0]], matrix_d[row_t[1]][col_t[1]],
          matrix_d[row_t[2]][col_t[2]], matrix_d[row_t[3]][col_t[3]],
          matrix_d[row_t[4]][col_t[4]], matrix_d[row_t[5]][col_t[5]],
          matrix_d[row_t[6]][col_t[6]], matrix_d[row_t[7]][col_t[7]]); // [한국어] 이 스레드의 8개 F32 D 원소를 레지스터 벡터에 저장

      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
        printf("thread%d:", thrd); // [한국어] 디버그: 스레드 번호 출력
        for (k = 0; k < 8; k++) {
          printf("%.2f ", matrix_d[row_t[k]][col_t[k]].f32); // [한국어] 디버그: F32 결과 원소 출력
        }
        printf("\n");
      }
    } else if (type == F16_TYPE) {
      // [한국어] F16 결과: 2개의 FP16을 하나의 32비트 레지스터에 팩킹하여 저장
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
        printf("thread%d:", thrd); // [한국어] 디버그: 스레드 번호 출력
        for (k = 0; k < 8; k++) {
          temp = matrix_d[row_t[k]][col_t[k]].f16; // [한국어] FP16 → float 변환 (디버그 출력용)
          printf("%.2f ", temp);
        }
        printf("\n");

        printf("thread%d:", thrd);
        for (k = 0; k < 8; k++) {
          printf("%x ", (unsigned int)matrix_d[row_t[k]][col_t[k]].f16); // [한국어] 디버그: FP16 원시 16진수 출력
        }
        printf("\n");
      }
      // [한국어] FP16 팩킹: 두 개의 FP16 값을 하나의 32비트 레지스터에 팩킹
      //   하위 16비트 = 짝수 인덱스 FP16, 상위 16비트 = 홀수 인덱스 FP16
      ptx_reg_t nw_data1, nw_data2, nw_data3, nw_data4;
      nw_data1.s64 = ((matrix_d[row_t[0]][col_t[0]].s64 & 0xffff)) |
                     ((matrix_d[row_t[1]][col_t[1]].s64 & 0xffff) << 16); // [한국어] D[0]을 하위 16비트, D[1]을 상위 16비트에 팩킹
      nw_data2.s64 = ((matrix_d[row_t[2]][col_t[2]].s64 & 0xffff)) |
                     ((matrix_d[row_t[3]][col_t[3]].s64 & 0xffff) << 16); // [한국어] D[2], D[3] 팩킹
      nw_data3.s64 = ((matrix_d[row_t[4]][col_t[4]].s64 & 0xffff)) |
                     ((matrix_d[row_t[5]][col_t[5]].s64 & 0xffff) << 16); // [한국어] D[4], D[5] 팩킹
      nw_data4.s64 = ((matrix_d[row_t[6]][col_t[6]].s64 & 0xffff)) |
                     ((matrix_d[row_t[7]][col_t[7]].s64 & 0xffff) << 16); // [한국어] D[6], D[7] 팩킹
      thread->set_vector_operand_values(dst, nw_data1, nw_data2, nw_data3,
                                        nw_data4); // [한국어] 팩킹된 4개의 32비트 레지스터를 dst 벡터 피연산자에 저장
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
        printf("thread%d=%llx,%llx,%llx,%llx", thrd, nw_data1.s64, nw_data2.s64,
               nw_data3.s64, nw_data4.s64); // [한국어] 디버그: 팩킹된 레지스터 값 출력

    } else {
      printf("wmma:mma:wrong type\n"); // [한국어] F16/F32 이외의 결과 타입 — 지원하지 않는 조합
      abort(); // [한국어] 지원하지 않는 wmma 결과 타입으로는 실행 불가
    }
  }
}

/*
 * [한국어]
 * call_impl - PTX `call` 명령어 구현: 함수 호출
 *
 * @pI: 실행 중인 PTX 명령어 객체. 호출 목적지 함수 주소, 인수/반환 피연산자 정보 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 호출 스택 관리, 인수 복사, PC 변경에 사용.
 *
 * PTX `call [ret,] func [, args]` 명령어의 기능 시뮬레이션 구현.
 * 처리 단계:
 *   1) PDOM(Post-Dominator) 분석: 피호출 함수가 아직 분석되지 않았으면 do_pdom() 수행.
 *      printf()처럼 함수 본체가 없는 인트린식은 함수 크기를 확인 후 건너뜀.
 *   2) 유효성 검사: 반환값 수, 인수 수가 함수 선언과 일치하는지 확인.
 *   3) 인트린식 처리: 함수 이름을 확인하여 특수 처리:
 *      - "vprintf": CUDA printf 에뮬레이션 → gpgpusim_cuda_vprintf()
 *      - "cudaGetParameterBufferV2": CDP(CUDA Dynamic Parallelism) 파라미터 버퍼
 *      - "cudaLaunchDeviceV2": 디바이스 측 커널 런치
 *      - "cudaStreamCreateWithFlags": 디바이스 측 스트림 생성
 *   4) 인수 복사: 호출 측 인수를 버퍼에 읽어 피호출 함수 프레임으로 복사.
 *   5) 콜스택 푸시: POST_DOMINATOR 모델에서 복귀 주소(callee_pc+inst_size)와 rpc를 스택에 저장.
 *   6) NPC 설정: 다음 실행 PC를 피호출 함수의 진입 주소로 설정.
 *
 * call_uid_next: 각 call 인스턴스에 고유 ID를 부여하는 정적 카운터.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: do_pdom(), copy_args_into_buffer_list(), callstack_push(), set_npc().
 * 에러 경로: 반환값/인수 수 불일치이면 printf 후 abort().
 *
 * 호출 체인:
 *   ptx_exec_inst() → [call_impl] → do_pdom() / callstack_push() / set_npc()
 */
void call_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  static unsigned call_uid_next = 1; // [한국어] 호출 인스턴스별 고유 ID 카운터 (정적 — 프로그램 실행 동안 단조 증가)

  const operand_info &target = pI->func_addr(); // [한국어] 호출 목적지 함수 주소 피연산자 정보
  assert(target.is_function_address()); // [한국어] 목적지가 함수 주소인지 확인 (일반 레지스터이면 버그)
  const symbol *func_addr = target.get_symbol(); // [한국어] 목적지 심볼 (함수 이름, PC 포함)
  function_info *target_func = func_addr->get_pc(); // [한국어] 함수 심볼에서 function_info 객체 획득 (PDOM 분석 상태, 인수 정보 포함)
  if (target_func->is_pdom_set()) {
    // [한국어] PDOM 분석이 이미 완료된 함수 — 재분석 불필요
    printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
           target_func->get_name().c_str());
  } else {
    // [한국어] PDOM 분석이 아직 수행되지 않은 함수 — 재합류 지점 분석 수행
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           target_func->get_name().c_str());
    /*
     * Some of the instructions like printf() gives the gpgpusim the wrong
     * impression that it is a function call. As printf() doesnt have a body
     * like functions do, doing pdom analysis for printf() causes a crash.
     */
    // [한국어] printf처럼 함수 본체가 없는 인트린식은 PDOM 분석 건너뜀 (함수 크기 > 0인 경우만 분석)
    if (target_func->get_function_size() > 0) target_func->do_pdom(); // [한국어] 실제 함수 본체가 있는 경우 PDOM(Post-Dominator) 분석 수행
    target_func->set_pdom(); // [한국어] PDOM 분석 완료 표시 (재분석 방지)
  }

  // check that number of args and return match function requirements
  if (pI->has_return() ^ target_func->has_return()) {
    // [한국어] 호출 명령어의 반환값 여부와 함수 선언의 반환값 여부가 다른 경우
    printf(
        "GPGPU-Sim PTX: Execution error - mismatch in number of return values "
        "between\n"
        "               call instruction and function declaration\n");
    abort(); // [한국어] 반환값 수 불일치는 심각한 오류 — 시뮬레이션 중단
  }
  unsigned n_return = target_func->has_return(); // [한국어] 함수가 반환값을 가지는지 여부 (0 또는 1)
  unsigned n_args = target_func->num_args();     // [한국어] 함수가 받는 인수 수
  unsigned n_operands = pI->get_num_operands();  // [한국어] call 명령어의 총 피연산자 수 (ret + func + args)

  if (n_operands != (n_return + 1 + n_args)) {
    // [한국어] call 명령어의 피연산자 수가 함수 선언과 맞지 않는 경우
    printf(
        "GPGPU-Sim PTX: Execution error - mismatch in number of arguements "
        "between\n"
        "               call instruction and function declaration\n");
    abort(); // [한국어] 인수 수 불일치는 심각한 오류 — 시뮬레이션 중단
  }

  // handle intrinsic functions
  std::string fname = target_func->get_name(); // [한국어] 피호출 함수 이름 (인트린식 판별에 사용)
  if (fname == "vprintf") {
    // [한국어] CUDA printf 인트린식: gpgpusim_cuda_vprintf()로 에뮬레이션
    gpgpusim_cuda_vprintf(pI, thread, target_func);
    return; // [한국어] 인트린식 처리 후 즉시 반환 (콜스택 푸시 불필요)
  }
#if (CUDART_VERSION >= 5000)
  // Jin: handle device runtime apis for CDP
  else if (fname == "cudaGetParameterBufferV2") {
    // [한국어] CDP(CUDA Dynamic Parallelism): 디바이스 커널 런치를 위한 파라미터 버퍼 획득
    target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_getParameterBufferV2(
        pI, thread, target_func);
    return; // [한국어] CDP 인트린식 처리 후 즉시 반환
  } else if (fname == "cudaLaunchDeviceV2") {
    // [한국어] CDP: 디바이스 측에서 새 커널 런치 에뮬레이션
    target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_launchDeviceV2(
        pI, thread, target_func);
    return; // [한국어] CDP 인트린식 처리 후 즉시 반환
  } else if (fname == "cudaStreamCreateWithFlags") {
    // [한국어] CDP: 디바이스 측 스트림 생성 에뮬레이션
    target_func->gpgpu_ctx->device_runtime->gpgpusim_cuda_streamCreateWithFlags(
        pI, thread, target_func);
    return; // [한국어] CDP 인트린식 처리 후 즉시 반환
  }
#endif

  // read source arguements into register specified in declaration of function
  arg_buffer_list_t arg_values; // [한국어] 인수 값 목록: 호출 측에서 읽어 피호출 측으로 전달할 인수 버퍼
  copy_args_into_buffer_list(pI, thread, target_func, arg_values); // [한국어] call 명령어의 인수 피연산자 값을 버퍼로 읽어옴

  // record local for return value (we only support a single return value)
  const symbol *return_var_src = NULL; // [한국어] 피호출 함수의 반환 변수 심볼 (함수 내부 return 변수)
  const symbol *return_var_dst = NULL; // [한국어] 호출 측의 반환값 수신 변수 심볼 (call 명령어 dst)
  if (target_func->has_return()) {
    // [한국어] 반환값이 있는 경우: 호출 측 dst와 피호출 측 return_var를 연결
    return_var_dst = pI->dst().get_symbol();       // [한국어] 호출 측에서 반환값을 받을 레지스터 심볼
    return_var_src = target_func->get_return_var(); // [한국어] 피호출 함수에서 반환값을 저장하는 변수 심볼
  }

  gpgpu_sim *gpu = thread->get_gpu(); // [한국어] 최상위 GPU 시뮬레이터 객체 (SIMD 모델 확인용)
  unsigned callee_pc = 0, callee_rpc = 0; // [한국어] 복귀 PC와 재합류 PC 초기화
  if (gpu->simd_model() == POST_DOMINATOR) {
    // [한국어] POST_DOMINATOR 모델: PDOM 스택에서 현재 PC와 재합류 PC 조회
    thread->get_core()->get_pdom_stack_top_info(thread->get_hw_wid(),
                                                &callee_pc, &callee_rpc); // [한국어] warp ID로 PDOM 스택 상단 PC/RPC 읽기
    assert(callee_pc == thread->get_pc()); // [한국어] PDOM 스택 PC와 현재 스레드 PC가 일치해야 함 (일관성 확인)
  }

  thread->callstack_push(callee_pc + pI->inst_size(), callee_rpc,
                         return_var_src, return_var_dst, call_uid_next++); // [한국어] 콜스택에 복귀 정보 푸시: 복귀 PC(현재 명령어 다음), 재합류 PC, 반환 변수, 고유 call ID

  copy_buffer_list_into_frame(thread, arg_values); // [한국어] 버퍼에 저장된 인수 값을 피호출 함수의 스택 프레임에 복사

  thread->set_npc(target_func); // [한국어] 다음 실행 PC를 피호출 함수의 진입 주소로 설정 (함수 진입)
}

// Ptxplus version of call instruction. Jumps to a label not a different Kernel.
/*
 * [한국어]
 * callp_impl - PTXPlus `callp` 명령어 구현: 레이블로 점프하는 PTXPlus 전용 함수 호출
 *
 * @pI: 실행 중인 PTX 명령어 객체. 목적지 레이블의 PC 값을 dst 피연산자로 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 콜스택 관리 및 PC 변경에 사용.
 *
 * PTXPlus 전용 call 명령어로, call_impl()과 달리 다른 커널이 아니라
 * 동일 함수 내의 레이블(label)로 점프한다. 즉, 소규모 서브루틴 호출 패턴.
 * call_impl()과의 차이점:
 *   - PDOM 분석, 인트린식 처리, 인수/반환 검증 없음
 *   - callstack_push_plus()를 사용 (PTXPlus용 콜스택 푸시)
 *   - 목적지는 레이블 PC 값을 직접 레지스터로 읽음
 *
 * call_uid_next: 각 callp 인스턴스에 고유 ID를 부여하는 정적 카운터.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: get_operand_value(), callstack_push_plus(), set_npc().
 * 에러 경로: 없음 (PTXPlus는 타입 검사를 별도로 수행하지 않음).
 *
 * 호출 체인:
 *   ptx_exec_inst() → [callp_impl] → callstack_push_plus() / set_npc()
 */
void callp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  static unsigned call_uid_next = 1; // [한국어] callp 인스턴스별 고유 ID 카운터

  const operand_info &target = pI->dst(); // [한국어] 점프 목적지 레이블 피연산자 정보
  ptx_reg_t target_pc =
      thread->get_operand_value(target, target, U32_TYPE, thread, 1); // [한국어] 목적지 레이블의 PC 값 읽기

  const symbol *return_var_src = NULL; // [한국어] PTXPlus callp는 반환 변수 없음 (NULL로 초기화)
  const symbol *return_var_dst = NULL; // [한국어] PTXPlus callp는 반환 수신 변수 없음 (NULL로 초기화)

  gpgpu_sim *gpu = thread->get_gpu(); // [한국어] GPU 시뮬레이터 객체 (SIMD 모델 확인용)
  unsigned callee_pc = 0, callee_rpc = 0; // [한국어] 복귀 PC와 재합류 PC 초기화
  if (gpu->simd_model() == POST_DOMINATOR) {
    // [한국어] POST_DOMINATOR 모델: PDOM 스택에서 현재 PC/재합류 PC 조회
    thread->get_core()->get_pdom_stack_top_info(thread->get_hw_wid(),
                                                &callee_pc, &callee_rpc); // [한국어] warp ID로 PDOM 스택 상단 정보 읽기
    assert(callee_pc == thread->get_pc()); // [한국어] PDOM 스택 PC와 현재 스레드 PC 일치 확인
  }

  thread->callstack_push_plus(callee_pc + pI->inst_size(), callee_rpc,
                              return_var_src, return_var_dst, call_uid_next++); // [한국어] PTXPlus용 콜스택에 복귀 정보 푸시
  thread->set_npc(target_pc); // [한국어] 다음 실행 PC를 목적지 레이블 주소로 설정 (서브루틴 진입)
}

/*
 * [한국어]
 * clz_impl - PTX `clz` 명령어 구현: 선행 0 비트 수 계산 (Count Leading Zeros)
 *
 * @pI: 실행 중인 PTX 명령어 객체. 피연산자 타입(B32/B64), dst/src1 정보 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 레지스터 파일 읽기/쓰기에 사용.
 *
 * PTX `clz.type d, a` 명령어:
 *   소스 값(a)의 MSB(최상위 비트)부터 세어 처음 1이 나타나기 전까지의 0 비트 수를 반환.
 *   - B32: 32비트 값, 최대 결과 32 (모두 0이면 32)
 *   - B64: 64비트 값, 최대 결과 64 (모두 0이면 64)
 * 결과는 항상 B32_TYPE(32비트 부호 없는 정수)로 저장된다.
 * 알고리즘: MSB 마스크(0x80000000 또는 0x8000000000000000)로 상위 비트를 확인하면서
 * 왼쪽 시프트를 반복해 0인 비트를 카운팅한다.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: get_operand_value(), set_operand_value() — 레지스터 파일 접근.
 * 에러 경로: B32/B64 이외의 타입이면 printf 후 assert(0).
 *
 * 호출 체인:
 *   ptx_exec_inst() → [clz_impl] → get_operand_value() / set_operand_value()
 */
void clz_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d; // [한국어] a: 소스 값, d: 선행 0 비트 카운트 결과
  const operand_info &dst = pI->dst();   // [한국어] 결과를 저장할 목적지 피연산자 (B32 타입으로 기록)
  const operand_info &src1 = pI->src1(); // [한국어] 선행 0 탐색 대상 소스 피연산자

  unsigned i_type = pI->get_type(); // [한국어] 피연산자 타입 (B32 또는 B64만 지원)
  a = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] 소스 값 읽기

  int max; // [한국어] 타입별 비트 폭 (B32=32, B64=64) — 최대 카운트 상한
  unsigned long long mask; // [한국어] MSB 확인용 마스크 (B32=0x80000000, B64=0x8000000000000000)
  d.u64 = 0; // [한국어] 결과 카운터를 0으로 초기화

  switch (i_type) {
    case B32_TYPE:
      max = 32;           // [한국어] 32비트: 최대 32개의 선행 0
      mask = 0x80000000;  // [한국어] 비트 31(MSB) 확인 마스크
      break;
    case B64_TYPE:
      max = 64;                       // [한국어] 64비트: 최대 64개의 선행 0
      mask = 0x8000000000000000;      // [한국어] 비트 63(MSB) 확인 마스크
      break;
    default:
      printf("Execution error: type mismatch with instruction\n"); // [한국어] clz는 B32/B64만 지원 — 타입 불일치 오류
      assert(0);
      break;
  }

  while ((d.u32 < max) && ((a.u64 & mask) == 0)) {
    // [한국어] MSB가 0인 동안: 카운터 증가하고 왼쪽 시프트로 다음 비트를 MSB 위치로 이동
    d.u32++;         // [한국어] 선행 0 카운터 증가
    a.u64 = a.u64 << 1; // [한국어] 왼쪽 시프트로 다음 비트를 MSB 위치로 이동 (순차 탐색)
  }

  thread->set_operand_value(dst, d, B32_TYPE, thread, pI); // [한국어] 선행 0 카운트를 B32 타입으로 dst 레지스터에 기록
}

/*
 * [한국어]
 * cnot_impl - PTX `cnot` 명령어 구현: 논리 NOT / 보수 (Complement NOT)
 *
 * @pI: 실행 중인 PTX 명령어 객체. 피연산자 타입(PRED/B16/B32/B64), dst/src1 정보 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 레지스터 파일 읽기/쓰기에 사용.
 *
 * PTX `cnot.type d, a` 명령어:
 *   소스 값이 0이면 1, 0이 아니면 0을 반환하는 논리 NOT(보수) 연산.
 *   비트 반전(brev)이 아닌 "0/1 상태 반전"이다.
 *   - PRED_TYPE: 술어 레지스터의 bit0(true/false 비트)을 검사. bit0==0(true)이면 1, 아니면 0.
 *   - B16/B32/B64: 전체 값이 0이면 1, 아니면 0.
 * 주로 조건 분기를 위한 술어 반전이나 마스크 생성에 사용된다.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: get_operand_value(), set_operand_value() — 레지스터 파일 접근.
 * 에러 경로: PRED/B16/B32/B64 이외의 타입이면 printf 후 assert(0).
 *
 * 호출 체인:
 *   ptx_exec_inst() → [cnot_impl] → get_operand_value() / set_operand_value()
 */
void cnot_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d; // [한국어] a: 소스 값, b: 미사용 변수, d: cnot 결과값
  const operand_info &dst = pI->dst();   // [한국어] 결과를 저장할 목적지 피연산자
  const operand_info &src1 = pI->src1(); // [한국어] NOT 연산 대상 소스 피연산자

  unsigned i_type = pI->get_type(); // [한국어] 피연산자 타입 (PRED/B16/B32/B64)
  a = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] 소스 값 읽기

  switch (i_type) {
    case PRED_TYPE:
      // [한국어] 술어 타입: bit0==0(PTXPlus에서 0=true)이면 결과 1, 아니면 0
      //   PTXPlus 술어 관례: bit0=0은 "참(true)", bit0=1은 "거짓(false)"
      d.pred = ((a.pred & 0x0001) == 0) ? 1 : 0;
      break;
    case B16_TYPE:
      d.u16 = (a.u16 == 0) ? 1 : 0; // [한국어] 16비트 논리 NOT: 0이면 1, 아니면 0
      break;
    case B32_TYPE:
      d.u32 = (a.u32 == 0) ? 1 : 0; // [한국어] 32비트 논리 NOT: 0이면 1, 아니면 0
      break;
    case B64_TYPE:
      d.u64 = (a.u64 == 0) ? 1 : 0; // [한국어] 64비트 논리 NOT: 0이면 1, 아니면 0
      break;
    default:
      printf("Execution error: type mismatch with instruction\n"); // [한국어] cnot는 PRED/B16/B32/B64만 지원
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI); // [한국어] 논리 NOT 결과를 dst 레지스터에 기록
}

/*
 * [한국어]
 * cos_impl - PTX `cos` 명령어 구현: 단정밀도 부동소수점 코사인
 *
 * @pI: 실행 중인 PTX 명령어 객체. 피연산자 타입(F32만 지원), dst/src1 정보 포함.
 * @thread: 시뮬레이션 중인 스레드 상태. 레지스터 파일 읽기/쓰기에 사용.
 *
 * PTX `cos.approx.f32 d, a` 명령어:
 *   소스 값(a)의 코사인 값을 계산하여 dst에 저장한다.
 *   GPGPU-Sim에서는 C 표준 라이브러리의 cos() 함수를 직접 호출하여 에뮬레이션한다.
 *   실제 GPU 하드웨어에서 `cos.approx.f32`는 근사 코사인 연산 (MUFU 단계)이지만,
 *   시뮬레이터에서는 정확한 수학 함수로 대체된다 — 이 차이로 인한 수치 오차는 무시한다.
 *   F32 타입만 지원하며, F64는 지원하지 않는다.
 *
 * 실행 컨텍스트: 기능 시뮬레이션(cuda-sim) 단계, 단일 스레드 단위로 호출.
 * 호출자: ptx_thread_info::ptx_exec_inst() — 명령어 디스패치 루프.
 * 피호출자: get_operand_value(), C cos(), set_operand_value().
 * 에러 경로: F32 이외의 타입이면 printf 후 assert(0).
 *
 * 호출 체인:
 *   ptx_exec_inst() → [cos_impl] → get_operand_value() / cos() / set_operand_value()
 */
void cos_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d; // [한국어] a: 소스 값(라디안 단위 입력 각도), d: 코사인 결과값
  const operand_info &dst = pI->dst();   // [한국어] 코사인 결과를 저장할 목적지 피연산자
  const operand_info &src1 = pI->src1(); // [한국어] 코사인 계산 대상 소스 피연산자 (라디안 단위 각도)

  unsigned i_type = pI->get_type(); // [한국어] 피연산자 타입 (F32만 지원)
  a = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] 소스 각도 값 읽기

  switch (i_type) {
    case F32_TYPE:
      d.f32 = cos(a.f32); // [한국어] C 표준 cos() 함수로 단정밀도 코사인 계산 (GPU MUFU 근사값 대신 정확한 값 사용)
      break;
    default:
      printf("Execution error: type mismatch with instruction\n"); // [한국어] PTX cos는 F32만 지원 — F64는 미지원
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI); // [한국어] 코사인 결과를 dst 레지스터에 기록
}

/*
 * [한국어]
 * chop - ptx_reg_t 값을 to_width 비트로 잘라내기(마스킹)
 *
 * @x: 변환 대상 레지스터 값 (ptx_reg_t 공용체)
 * @from_width: 원본 데이터 비트 폭 (사용되지 않음 — 마스킹은 to_width 기준)
 * @to_width: 목적지 비트 폭 (8/16/32/64 중 하나)
 * @to_sign: 목적지 부호 (이 함수에서는 사용하지 않음)
 * @rounding_mode: 반올림 모드 (이 함수에서는 사용하지 않음)
 * @saturation_mode: 포화 모드 (이 함수에서는 사용하지 않음)
 * @return: 상위 비트가 0으로 마스킹된 ptx_reg_t
 *
 * PTX cvt 명령어에서 정수 타입을 좁은 폭으로 변환할 때 사용되는 헬퍼 함수.
 * 넓은 타입(예: S32)을 좁은 타입(예: U8)으로 변환 시 상위 비트를 제거하여
 * to_width 하위 비트만 보존한다. 부호 확장 없이 단순 비트 마스킹을 수행한다.
 * 64비트 케이스에서는 마스킹이 불필요하므로 그대로 통과시킨다.
 *
 * 호출 체인:
 *   cvt_impl() → g_cvt_fn[src_fmt][dst_fmt]() → [chop] → (mask_and)
 */
ptx_reg_t chop(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
               int rounding_mode, int saturation_mode) {
  switch (to_width) { // [한국어] 목적지 비트 폭에 따라 마스킹 분기
    case 8:
      x.mask_and(0, 0xFF); // [한국어] 상위 56비트를 0으로 클리어, 하위 8비트만 보존
      break;
    case 16:
      x.mask_and(0, 0xFFFF); // [한국어] 상위 48비트를 0으로 클리어, 하위 16비트만 보존
      break;
    case 32:
      x.mask_and(0, 0xFFFFFFFF); // [한국어] 상위 32비트를 0으로 클리어, 하위 32비트만 보존
      break;
    case 64:
      break; // [한국어] 64비트는 전체 폭이므로 마스킹 불필요 — 그대로 통과
    default:
      assert(0); // [한국어] 지원하지 않는 비트 폭 — 구현 오류
  }
  return x; // [한국어] 마스킹된 레지스터 값 반환
}

/*
 * [한국어]
 * sext - from_width 비트에서 부호 확장(sign extension)
 *
 * @x: 변환 대상 레지스터 값
 * @from_width: 원본 데이터 비트 폭 (8/16/32/64) — 이 비트 폭의 부호 비트를 기준으로 확장
 * @to_width: 목적지 비트 폭 (이 함수에서는 사용되지 않음 — from_width 기준)
 * @to_sign: 목적지 부호 (이 함수에서는 사용하지 않음)
 * @rounding_mode: 반올림 모드 (하위 chop에 전달)
 * @saturation_mode: 포화 모드 (하위 chop에 전달)
 * @return: 부호 확장된 ptx_reg_t
 *
 * PTX cvt 명령어에서 signed 정수 타입을 더 넓은 타입으로 변환할 때 사용.
 * from_width의 MSB(최상위 비트, 부호 비트)가 1이면 상위 비트를 0xFF...로 채워
 * 음수 값을 올바르게 보존한다. 먼저 chop으로 from_width 비트 이상을 클리어한 뒤
 * 부호 비트를 검사하여 필요 시 마스크를 OR한다.
 *
 * 호출 체인:
 *   cvt_impl() → g_cvt_fn[src_fmt][dst_fmt]() → [sext] → chop()
 */
ptx_reg_t sext(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
               int rounding_mode, int saturation_mode) {
  x = chop(x, 0, from_width, 0, rounding_mode, saturation_mode); // [한국어] 먼저 from_width 비트로 잘라 상위 쓰레기 비트를 제거
  switch (from_width) { // [한국어] 원본 비트 폭에 따라 부호 비트 위치를 결정하여 확장 분기
    case 8:
      if (x.get_bit(7)) x.mask_or(0xFFFFFFFF, 0xFFFFFF00); // [한국어] 비트 7(부호)이 1이면 상위 56비트를 모두 1로 채워 음수 보존
      break;
    case 16:
      if (x.get_bit(15)) x.mask_or(0xFFFFFFFF, 0xFFFF0000); // [한국어] 비트 15(부호)가 1이면 상위 48비트를 모두 1로 채워 음수 보존
      break;
    case 32:
      if (x.get_bit(31)) x.mask_or(0xFFFFFFFF, 0x00000000); // [한국어] 비트 31(부호)이 1이면 상위 32비트를 0xFFFFFFFF로 채움 (64비트 레지스터 상위절반)
      break;
    case 64:
      break; // [한국어] 64비트 원본은 이미 최대 폭 — 확장 불필요
    default:
      assert(0); // [한국어] 지원하지 않는 비트 폭 — 구현 오류
  }
  return x; // [한국어] 부호 확장된 레지스터 값 반환
}

/*
 * [한국어]
 * sexd - 목적지 비트 폭 기준 부호 확장 (CUDA 4.2 SobelFilter 호환 핵)
 *
 * @x: 변환 대상 레지스터 값
 * @from_width: 원본 데이터 비트 폭 (chop에 전달하여 상위 비트 클리어에 사용)
 * @to_width: 목적지 비트 폭 — 이 함수에서 부호 비트 위치 결정에 사용 (sext와의 차이)
 * @to_sign: 목적지 부호 (이 함수에서는 사용하지 않음)
 * @rounding_mode: 반올림 모드 (하위 chop에 전달)
 * @saturation_mode: 포화 모드 (하위 chop에 전달)
 * @return: to_width 기준으로 부호 확장된 ptx_reg_t
 *
 * sext()와 거의 동일하지만 부호 비트 위치를 from_width가 아닌 to_width 기준으로
 * 판단한다는 점이 다르다. CUDA 4.2의 SobelFilter 커널이 특정 cvt 조합(S16→S32 등)에서
 * from_width가 아닌 to_width 기준 부호 확장을 기대하는 버그/특이 동작을 재현하기 위한
 * 임시(hack) 함수이다. g_cvt_fn 테이블에서 (S32→S16) 변환 셀에 배치되어 있다.
 *
 * 호출 체인:
 *   cvt_impl() → g_cvt_fn[2][1]() → [sexd] → chop()
 */
// sign extend depending on the destination register size - hack to get
// SobelFilter working in CUDA 4.2
ptx_reg_t sexd(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
               int rounding_mode, int saturation_mode) {
  x = chop(x, 0, from_width, 0, rounding_mode, saturation_mode); // [한국어] from_width 비트 초과 상위 비트를 먼저 클리어
  switch (to_width) { // [한국어] sext()와 달리 to_width(목적지 폭) 기준으로 부호 비트 위치 결정 — 이것이 핵심 차이
    case 8:
      if (x.get_bit(7)) x.mask_or(0xFFFFFFFF, 0xFFFFFF00); // [한국어] to_width=8 기준 비트 7을 부호 비트로 보고 상위 비트 채움
      break;
    case 16:
      if (x.get_bit(15)) x.mask_or(0xFFFFFFFF, 0xFFFF0000); // [한국어] to_width=16 기준 비트 15을 부호 비트로 보고 상위 비트 채움
      break;
    case 32:
      if (x.get_bit(31)) x.mask_or(0xFFFFFFFF, 0x00000000); // [한국어] to_width=32 기준 비트 31을 부호 비트로 보고 상위 32비트 채움
      break;
    case 64:
      break; // [한국어] 64비트는 이미 최대 폭 — 확장 불필요
    default:
      assert(0); // [한국어] 지원하지 않는 목적지 비트 폭 — 구현 오류
  }
  return x; // [한국어] to_width 기준 부호 확장된 레지스터 값 반환
}

/*
 * [한국어]
 * zext - 제로 확장(zero extension): from_width 이상의 상위 비트를 0으로 채움
 *
 * @x: 변환 대상 레지스터 값
 * @from_width: 원본 데이터 비트 폭 (chop에 전달하여 마스킹 기준으로 사용)
 * @to_width: 목적지 비트 폭 (이 함수에서 직접 사용하지 않음 — chop에 위임)
 * @to_sign: 목적지 부호 (사용하지 않음)
 * @rounding_mode: 반올림 모드 (chop에 전달)
 * @saturation_mode: 포화 모드 (chop에 전달)
 * @return: 상위 비트가 0으로 채워진 ptx_reg_t
 *
 * unsigned 정수 타입을 더 넓은 타입으로 변환할 때 사용하는 헬퍼.
 * sext()와 달리 부호 비트를 검사하지 않고 단순히 chop()으로 상위 비트를 0으로 클리어한다.
 * 실질적으로 chop()의 래퍼이며, g_cvt_fn 테이블에서 unsigned→wider unsigned 셀에 배치된다.
 *
 * 호출 체인:
 *   cvt_impl() → g_cvt_fn[src_fmt][dst_fmt]() → [zext] → chop()
 */
ptx_reg_t zext(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
               int rounding_mode, int saturation_mode) {
  return chop(x, 0, from_width, 0, rounding_mode, saturation_mode); // [한국어] chop()으로 from_width 이상의 상위 비트를 0 클리어하여 제로 확장 수행
}

/*
 * [한국어]
 * saturatei(int, int, int) - 부호있는 정수 값을 [min, max] 범위로 클램핑
 *
 * @a: 클램핑 대상 정수 값
 * @max: 허용 최대값 (포함)
 * @min: 허용 최소값 (포함)
 * @return: [min, max] 범위 내로 클램핑된 값
 *
 * f2x()에서 float→signed int 변환 시 to_width < 32인 경우(8비트, 16비트 대상)
 * 포화 모드가 활성화되었을 때 범위를 초과하는 값을 경계값으로 고정한다.
 * IEEE 754 포화 변환 규격: 범위 초과 시 최대/최소 정수값으로 포화.
 *
 * 호출 체인:
 *   f2x() → [saturatei(int,int,int)]
 */
int saturatei(int a, int max, int min) {
  if (a > max) // [한국어] 값이 최대값을 초과하면 최대값으로 포화
    a = max;
  else if (a < min) // [한국어] 값이 최소값 미만이면 최소값으로 포화
    a = min;
  return a; // [한국어] 클램핑된 값 반환 (범위 내이면 원본 그대로)
}

/*
 * [한국어]
 * saturatei(unsigned int, unsigned int) - 부호없는 정수 값을 [0, max] 범위로 클램핑
 *
 * @a: 클램핑 대상 unsigned 정수 값
 * @max: 허용 최대값 (포함); 최소값은 암시적으로 0
 * @return: [0, max] 범위 내로 클램핑된 값 (최소값 0은 unsigned 특성상 보장됨)
 *
 * f2x()에서 float→unsigned int 변환 시 to_width < 32인 경우 포화 모드 활성화 시 사용.
 * unsigned 타입이므로 하한은 0으로 고정, 상한만 검사한다.
 *
 * 호출 체인:
 *   f2x() → [saturatei(unsigned int, unsigned int)]
 */
unsigned int saturatei(unsigned int a, unsigned int max) {
  if (a > max) a = max; // [한국어] 값이 최대값 초과 시 최대값으로 포화; unsigned이므로 하한 검사 불필요
  return a; // [한국어] 클램핑된 unsigned 값 반환
}

/*
 * [한국어]
 * f2x - float32(또는 float16) 값을 정수 또는 다른 float 타입으로 변환
 *
 * @x: 변환 대상 레지스터 (f32 또는 f16 필드를 사용)
 * @from_width: 원본 비트 폭 (16이면 f16→f32, 그 외 f32 기준으로 동작)
 * @to_width: 목적지 비트 폭 (8/16/32/64)
 * @to_sign: 목적지 부호: 1=signed int, 0=unsigned int, -1(또는 그 외)=float
 * @rounding_mode: PTX 반올림 모드 (RZI/RNI/RMI/RPI → cudaRoundMode로 변환)
 * @saturation_mode: 1이면 to_width 범위를 벗어난 값을 경계값으로 포화
 * @return: 변환된 결과를 담은 ptx_reg_t
 *
 * PTX cvt 명령어에서 float 소스 타입을 처리하는 핵심 변환 헬퍼.
 * g_cvt_fn 테이블의 f32 행(행 9)과 f16 행(행 8)에서 호출된다.
 * 동작 분기: (1) to_sign==1이면 float→signed int 변환,
 *            (2) to_sign==0이면 float→unsigned int 변환,
 *            (3) 그 외(to_sign==-1 등)이면 float→float(다른 폭) 변환.
 * 비정규 부동소수점(denormal, 지수부=0)은 0으로 반올림하는 GPU 특성을 구현한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드 — 단일 스레드에서 호출되므로 동기화 불필요.
 *
 * 호출 체인:
 *   cvt_impl() → g_cvt_fn[8or9][dst_fmt]() → [f2x] → cuda_math::float2int/float2uint
 */
ptx_reg_t f2x(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  half mytemp; // [한국어] half 타입 임시 변수 (현재는 사용되지 않음 — 과거 코드 흔적)
  half_float::half tmp_h; // [한국어] half_float 라이브러리 임시 변수 (현재는 사용되지 않음)
  // assert( from_width == 32);

  enum cudaRoundMode mode = cudaRoundZero; // [한국어] 기본 반올림 모드를 0 방향(RZ)으로 초기화
  switch (rounding_mode) { // [한국어] PTX 반올림 옵션을 CUDA 수학 라이브러리의 cudaRoundMode 열거값으로 변환
    case RZI_OPTION:
      mode = cudaRoundZero; // [한국어] RZI: 0 방향(truncate) — 절댓값이 작아지는 방향
      break;
    case RNI_OPTION:
      mode = cudaRoundNearest; // [한국어] RNI: 가장 가까운 짝수 방향(round to nearest even)
      break;
    case RMI_OPTION:
      mode = cudaRoundMinInf; // [한국어] RMI: 음의 무한대 방향(floor)
      break;
    case RPI_OPTION:
      mode = cudaRoundPosInf; // [한국어] RPI: 양의 무한대 방향(ceil)
      break;
    default:
      break; // [한국어] 반올림 모드 미지정 시 cudaRoundZero 기본값 유지
  }

  ptx_reg_t y; // [한국어] 변환 결과를 담을 레지스터 초기화
  if (to_sign == 1) {  // convert to 64-bit number first?
    // [한국어] to_sign==1: 목적지가 signed 정수 타입인 경우 처리
    int tmp = cuda_math::float2int(x.f32, mode); // [한국어] float32를 32비트 signed int로 변환 (지정된 반올림 모드 적용)
    if ((x.u32 & 0x7f800000) == 0) tmp = 0;  // round denorm. FP to 0
    // [한국어] 지수부 비트(비트 23~30)가 모두 0인 비정규 부동소수점(denormal)은 GPU HW 특성상 0으로 강제 변환
    if (saturation_mode && to_width < 32) { // [한국어] 포화 모드이고 목적지가 32비트 미만인 경우에만 범위 클램핑 필요
      tmp = saturatei(tmp, (1 << to_width) - 1, -(1 << to_width)); // [한국어] to_width 비트 signed 범위 [-(1<<to_width), (1<<to_width)-1]로 클램핑
    }
    switch (to_width) { // [한국어] 목적지 비트 폭에 맞는 signed 정수 필드에 결과 저장
      case 8:
        y.s8 = (char)tmp; // [한국어] 8비트 signed int(char)로 저장 — 상위 비트 자동 절단
        break;
      case 16:
        y.s16 = (short)tmp; // [한국어] 16비트 signed int(short)로 저장
        break;
      case 32:
        y.s32 = (int)tmp; // [한국어] 32비트 signed int로 저장
        break;
      case 64:
        y.s64 = (long long)tmp; // [한국어] 64비트 signed int로 저장 (32→64 부호 확장)
        break;
      default:
        assert(0); // [한국어] 지원하지 않는 목적지 폭 — 구현 오류
        break;
    }
  } else if (to_sign == 0) {
    // [한국어] to_sign==0: 목적지가 unsigned 정수 타입인 경우 처리
    unsigned int tmp = cuda_math::float2uint(x.f32, mode); // [한국어] float32를 32비트 unsigned int로 변환
    if ((x.u32 & 0x7f800000) == 0) tmp = 0;  // round denorm. FP to 0
    // [한국어] 비정규 부동소수점(denormal)은 GPU HW 특성상 0으로 강제 변환
    if (saturation_mode && to_width < 32) { // [한국어] 포화 모드이고 목적지가 32비트 미만인 경우에만 범위 클램핑 필요
      tmp = saturatei(tmp, (1 << to_width) - 1); // [한국어] to_width 비트 unsigned 범위 [0, (1<<to_width)-1]로 클램핑
    }
    switch (to_width) { // [한국어] 목적지 비트 폭에 맞는 unsigned 정수 필드에 결과 저장
      case 8:
        y.u8 = (unsigned char)tmp; // [한국어] 8비트 unsigned int(byte)로 저장
        break;
      case 16:
        y.u16 = (unsigned short)tmp; // [한국어] 16비트 unsigned int로 저장
        break;
      case 32:
        y.u32 = (unsigned int)tmp; // [한국어] 32비트 unsigned int로 저장
        break;
      case 64:
        y.u64 = (unsigned long long)tmp; // [한국어] 64비트 unsigned int로 저장 (제로 확장)
        break;
      default:
        assert(0); // [한국어] 지원하지 않는 목적지 폭 — 구현 오류
        break;
    }
  } else {
    // [한국어] to_sign이 1도 0도 아닌 경우: 목적지가 float 타입인 경우 (f16, f32, f64)
    switch (to_width) { // [한국어] 목적지 float 비트 폭에 따라 변환 분기
      case 16:
        y.f16 = half_float::half_cast<half,
                                      std::numeric_limits<float>::round_style>(
            x.f32);  // mytemp;
        // [한국어] float32를 float16으로 변환: half_float 라이브러리의 half_cast 사용, float의 기본 반올림 스타일 적용
        break;
      case 32:
        y.f32 = float(x.f16); // [한국어] float16을 float32로 확장 (from_width==16 경우; 정밀도 손실 없음)
        break;  // handled by f2f
      case 64:
        y.f64 = x.f32; // [한국어] float32를 float64(double)로 확장 — 정밀도 손실 없음
        break;
      default:
        assert(0); // [한국어] 지원하지 않는 목적지 float 폭 — 구현 오류
        break;
    }
  }
  return y; // [한국어] 변환된 결과 레지스터 반환
}

/*
 * [한국어]
 * saturated2i - double 값을 [min, max] 범위로 클램핑
 *
 * @a: 클램핑 대상 double 값
 * @max: 허용 최대값 (포함)
 * @min: 허용 최소값 (포함)
 * @return: [min, max] 범위 내로 클램핑된 double 값
 *
 * d2x()에서 float64→정수 변환 시 to_width 범위를 초과하는 값을 경계값으로
 * 포화시키기 위해 사용하는 헬퍼. saturatei(int)의 double 버전이며, 넓은
 * 범위의 max/min 값(예: 2^31-1, -2^31)을 double로 전달하여 정밀도를 보존한다.
 *
 * 호출 체인:
 *   d2x() → [saturated2i]
 */
double saturated2i(double a, double max, double min) {
  if (a > max) // [한국어] double 값이 최대값 초과 시 최대값으로 포화
    a = max;
  else if (a < min) // [한국어] double 값이 최소값 미만 시 최소값으로 포화
    a = min;
  return a; // [한국어] 클램핑된 double 값 반환
}

/*
 * [한국어]
 * d2x - float64(double) 값을 정수 또는 다른 float 타입으로 변환
 *
 * @x: 변환 대상 레지스터 (f64 필드를 사용)
 * @from_width: 원본 비트 폭 (반드시 64이어야 함 — assert로 검증)
 * @to_width: 목적지 비트 폭 (8/16/32/64)
 * @to_sign: 목적지 부호: 1=signed int, 0=unsigned int, 그 외=float
 * @rounding_mode: PTX 반올림 모드 (RZI/RNI/RMI/RPI)
 * @saturation_mode: 포화 모드 (이 함수에서는 암묵적으로 항상 클램핑 수행)
 * @return: 변환된 결과를 담은 ptx_reg_t
 *
 * g_cvt_fn 테이블의 f64 행(행 10)에서 호출되는 float64 변환 헬퍼.
 * f2x()와 구조가 같으나 double 정밀도로 먼저 반올림 후 정수/float 목적지에 저장.
 * 정수 변환 시 saturated2i()로 to_width 범위를 벗어나는 값을 클램핑한다.
 * float64→float32 변환(to_sign==-1, to_width==32)은 d2d 대신 이 경로로도 처리됨.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드 — 단일 스레드, 동기화 불필요.
 *
 * 호출 체인:
 *   cvt_impl() → g_cvt_fn[10][dst_fmt]() → [d2x] → saturated2i()
 */
ptx_reg_t d2x(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  assert(from_width == 64); // [한국어] d2x는 반드시 64비트(double) 소스에서만 호출되어야 함

  double tmp; // [한국어] 반올림 적용 후 중간 double 값을 보관할 임시 변수
  switch (rounding_mode) { // [한국어] PTX 반올림 모드에 따라 C 표준 수학 함수로 double을 반올림
    case RZI_OPTION:
      tmp = trunc(x.f64); // [한국어] 0 방향 절단(truncate toward zero)
      break;
    case RNI_OPTION:
      tmp = nearbyint(x.f64); // [한국어] 가장 가까운 정수로 반올림(nearest, 현재 FP 환경의 반올림 모드 적용)
      break;
    case RMI_OPTION:
      tmp = floor(x.f64); // [한국어] 음의 무한대 방향(내림, floor)
      break;
    case RPI_OPTION:
      tmp = ceil(x.f64); // [한국어] 양의 무한대 방향(올림, ceil)
      break;
    default:
      tmp = x.f64; // [한국어] 반올림 모드 미지정 시 원본 double 값 그대로 사용
      break;
  }

  ptx_reg_t y; // [한국어] 변환 결과를 담을 레지스터 선언
  if (to_sign == 1) {
    // [한국어] 목적지가 signed 정수인 경우: 먼저 범위 클램핑 후 목적지 폭 정수로 저장
    tmp = saturated2i(tmp, ((1 << (to_width - 1)) - 1), (1 << (to_width - 1)));
    // [한국어] signed to_width 비트 최대 = 2^(to_width-1)-1, 최소 = -2^(to_width-1) — 범위 클램핑
    // [한국어] 주의: (1 << (to_width-1))은 음수를 의도했으나 실제론 양수값 — 설계 의도 재검토 필요
    switch (to_width) { // [한국어] 클램핑된 double을 목적지 비트 폭에 맞는 signed 필드에 저장
      case 8:
        y.s8 = (char)tmp; // [한국어] 8비트 signed int로 저장 (캐스트로 truncate)
        break;
      case 16:
        y.s16 = (short)tmp; // [한국어] 16비트 signed int로 저장
        break;
      case 32:
        y.s32 = (int)tmp; // [한국어] 32비트 signed int로 저장
        break;
      case 64:
        y.s64 = (long long)tmp; // [한국어] 64비트 signed int로 저장
        break;
      default:
        assert(0); // [한국어] 지원하지 않는 목적지 폭
        break;
    }
  } else if (to_sign == 0) {
    // [한국어] 목적지가 unsigned 정수인 경우: 0 이상 최대값 이하로 클램핑 후 저장
    tmp = saturated2i(tmp, ((1 << (to_width - 1)) - 1), 0);
    // [한국어] unsigned to_width 비트 범위로 클램핑 (최소=0, 최대=2^(to_width-1)-1 — 약간의 범위 오류 있음)
    switch (to_width) { // [한국어] 클램핑된 double을 목적지 비트 폭에 맞는 unsigned 필드에 저장
      case 8:
        y.u8 = (unsigned char)tmp; // [한국어] 8비트 unsigned int로 저장
        break;
      case 16:
        y.u16 = (unsigned short)tmp; // [한국어] 16비트 unsigned int로 저장
        break;
      case 32:
        y.u32 = (unsigned int)tmp; // [한국어] 32비트 unsigned int로 저장
        break;
      case 64:
        y.u64 = (unsigned long long)tmp; // [한국어] 64비트 unsigned int로 저장
        break;
      default:
        assert(0); // [한국어] 지원하지 않는 목적지 폭
        break;
    }
  } else {
    // [한국어] 목적지가 float 타입인 경우 (f32 또는 f64)
    switch (to_width) { // [한국어] 목적지 float 비트 폭에 따라 변환 분기
      case 16:
        assert(0); // [한국어] double→float16 변환은 미구현
        break;
      case 32:
        y.f32 = x.f64; // [한국어] double을 float32로 축소 변환 (정밀도 손실 발생 가능)
        break;
      case 64:
        y.f64 = x.f64;  // should be handled by d2d
        // [한국어] double→double은 원래 d2d()가 처리해야 하나, 이 경로로도 통과 가능 (no-op)
        break;
      default:
        assert(0); // [한국어] 지원하지 않는 목적지 float 폭
        break;
    }
  }
  return y; // [한국어] 변환된 결과 레지스터 반환
}

/*
 * [한국어]
 * s2f - 부호있는 정수(signed int)를 float32 또는 float64로 변환
 *
 * @x: 변환 대상 레지스터 (s8/s16/s32/s64 필드를 사용)
 * @from_width: 원본 비트 폭 (8/16/32/64)
 * @to_width: 목적지 float 비트 폭 (32=F32, 64=F64)
 * @to_sign: 목적지 부호 (이 함수에서는 사용하지 않음 — 항상 float 출력)
 * @rounding_mode: PTX 반올림 모드 (RZ/RN/RM/RP — RZI/RNI 변형은 cvt에서 정수 출력 시 사용)
 * @saturation_mode: 포화 모드 (이 함수에서는 적용하지 않음 — 정수→float 포화는 없음)
 * @return: 변환된 float 값을 담은 ptx_reg_t
 *
 * g_cvt_fn 테이블에서 signed 정수 행(행 0~3)의 float 목적지 열(열 8, 9, 10)에서 호출.
 * from_width < 64이면 먼저 sext()로 32비트 signed int로 부호 확장 후 변환,
 * from_width == 64이면 64비트 long long에서 직접 변환한다.
 * cuda_math의 __int2float_r*, __ll2float_r* 함수가 IEEE 754 반올림 모드를 정확히 구현한다.
 * s64→f64 변환은 정밀도 손실 없이 직접 할당으로 처리.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   cvt_impl() → g_cvt_fn[0~3][8or9or10]() → [s2f] → sext(), cuda_math::__int2float_r*
 */
ptx_reg_t s2f(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  ptx_reg_t y; // [한국어] 변환 결과를 담을 레지스터

  if (from_width < 64) {  // 32-bit conversion
    // [한국어] 원본이 32비트 미만인 경우: 먼저 32비트 signed int로 부호 확장
    y = sext(x, from_width, 32, 0, rounding_mode, saturation_mode); // [한국어] sext()로 from_width 비트 signed 값을 32비트로 부호 확장

    switch (to_width) { // [한국어] 목적지 float 비트 폭에 따라 변환 함수 선택
      case 16:
        assert(0); // [한국어] s32→f16 변환은 미구현
        break;
      case 32:
        switch (rounding_mode) { // [한국어] PTX 반올림 모드에 따라 cuda_math int→float 변환 함수 선택
          case RZ_OPTION:
            y.f32 = cuda_math::__int2float_rz(y.s32); // [한국어] RZ(Round to Zero): 절댓값이 작아지는 방향으로 반올림
            break;
          case RN_OPTION:
            y.f32 = cuda_math::__int2float_rn(y.s32); // [한국어] RN(Round Nearest even): 가장 가까운 짝수 방향
            break;
          case RM_OPTION:
            y.f32 = cuda_math::__int2float_rd(y.s32); // [한국어] RM(Round Minus inf): 음의 무한대 방향(내림)
            break;
          case RP_OPTION:
            y.f32 = cuda_math::__int2float_ru(y.s32); // [한국어] RP(Round Plus inf): 양의 무한대 방향(올림)
            break;
          default:
            break; // [한국어] 반올림 모드 미지정 시 변환 없이 이전 y 값 유지 (비정상 케이스)
        }
        break;
      case 64:
        y.f64 = y.s32; // [한국어] s32→f64 변환: double의 표현 범위가 충분하므로 반올림 불필요
        break;  // no rounding needed
      default:
        assert(0); // [한국어] 지원하지 않는 목적지 폭
        break;
    }
  } else {
    // [한국어] from_width == 64인 경우: 64비트 long long(s64)에서 직접 변환
    switch (to_width) { // [한국어] 목적지 float 비트 폭에 따라 분기
      case 16:
        assert(0); // [한국어] s64→f16 변환은 미구현
        break;
      case 32:
        switch (rounding_mode) { // [한국어] PTX 반올림 모드에 따라 ll→float 변환 함수 선택
          case RZ_OPTION:
            y.f32 = cuda_math::__ll2float_rz(y.s64); // [한국어] 64비트 signed int→float32, 0 방향 반올림
            break;
          case RN_OPTION:
            y.f32 = cuda_math::__ll2float_rn(y.s64); // [한국어] 64비트 signed int→float32, 최근접 짝수 반올림
            break;
          case RM_OPTION:
            y.f32 = cuda_math::__ll2float_rd(y.s64); // [한국어] 64비트 signed int→float32, 내림 반올림
            break;
          case RP_OPTION:
            y.f32 = cuda_math::__ll2float_ru(y.s64); // [한국어] 64비트 signed int→float32, 올림 반올림
            break;
          default:
            break; // [한국어] 반올림 모드 미지정 시 변환 없이 유지
        }
        break;
      case 64:
        y.f64 = y.s64; // [한국어] s64→f64 변환: 내부 구현 없음 — 암묵적 캐스트로 처리 (일부 정밀도 손실 가능)
        break;  // no internal implementation found
      default:
        assert(0); // [한국어] 지원하지 않는 목적지 폭
        break;
    }
  }

  // saturating an integer to 1 or 0?
  return y; // [한국어] 변환된 float 값이 담긴 레지스터 반환
}

/*
 * [한국어]
 * u2f - 부호없는 정수(unsigned int)를 float32 또는 float64로 변환
 *
 * @x: 변환 대상 레지스터 (u8/u16/u32/u64 필드를 사용)
 * @from_width: 원본 비트 폭 (8/16/32/64)
 * @to_width: 목적지 float 비트 폭 (32=F32, 64=F64)
 * @to_sign: 목적지 부호 (이 함수에서는 사용하지 않음 — 항상 float 출력)
 * @rounding_mode: PTX 반올림 모드 (RZ/RN/RM/RP)
 * @saturation_mode: 포화 모드 (이 함수에서는 적용하지 않음)
 * @return: 변환된 float 값을 담은 ptx_reg_t
 *
 * g_cvt_fn 테이블에서 unsigned 정수 행(행 4~7)의 float 목적지 열(열 8, 9, 10)에서 호출.
 * s2f()와 구조가 동일하나 부호 없는 타입을 다룬다.
 * from_width < 64이면 먼저 zext()로 32비트 unsigned int로 확장 후 변환.
 * 주의: u64→f32 변환 시 모든 반올림 모드에서 동일하게 __ull2float_rn()을 사용 — 구현 미비.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   cvt_impl() → g_cvt_fn[4~7][8or9or10]() → [u2f] → zext(), cuda_math::__uint2float_r*
 */
ptx_reg_t u2f(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  ptx_reg_t y; // [한국어] 변환 결과를 담을 레지스터

  if (from_width < 64) {  // 32-bit conversion
    // [한국어] 원본이 32비트 미만인 경우: 먼저 32비트 unsigned int로 제로 확장
    y = zext(x, from_width, 32, 0, rounding_mode, saturation_mode); // [한국어] zext()로 from_width 비트 unsigned 값을 32비트로 제로 확장

    switch (to_width) { // [한국어] 목적지 float 비트 폭에 따라 변환 함수 선택
      case 16:
        assert(0); // [한국어] u32→f16 변환은 미구현
        break;
      case 32:
        switch (rounding_mode) { // [한국어] PTX 반올림 모드에 따라 cuda_math uint→float 변환 함수 선택
          case RZ_OPTION:
            y.f32 = cuda_math::__uint2float_rz(y.u32); // [한국어] RZ(Round to Zero): 0 방향으로 반올림
            break;
          case RN_OPTION:
            y.f32 = cuda_math::__uint2float_rn(y.u32); // [한국어] RN(Round Nearest): 최근접 짝수로 반올림
            break;
          case RM_OPTION:
            y.f32 = cuda_math::__uint2float_rd(y.u32); // [한국어] RM(Round Minus): 음의 무한대 방향(내림)
            break;
          case RP_OPTION:
            y.f32 = cuda_math::__uint2float_ru(y.u32); // [한국어] RP(Round Plus): 양의 무한대 방향(올림)
            break;
          default:
            break; // [한국어] 반올림 모드 미지정 시 변환 없이 유지
        }
        break;
      case 64:
        y.f64 = y.u32; // [한국어] u32→f64 변환: double 표현 범위가 충분하므로 반올림 불필요
        break;  // no rounding needed
      default:
        assert(0); // [한국어] 지원하지 않는 목적지 폭
        break;
    }
  } else {
    // [한국어] from_width == 64인 경우: 64비트 unsigned int(u64)에서 직접 변환
    switch (to_width) { // [한국어] 목적지 float 비트 폭에 따라 분기
      case 16:
        assert(0); // [한국어] u64→f16 변환은 미구현
        break;
      case 32:
        switch (rounding_mode) { // [한국어] 반올림 모드별로 분기하나 모두 동일 함수 사용 — 구현 미비
          case RZ_OPTION:
            y.f32 = cuda_math::__ull2float_rn(y.u64); // [한국어] RZ이지만 __ull2float_rn(최근접 반올림) 사용 — 의도적 단순화
            break;
          case RN_OPTION:
            y.f32 = cuda_math::__ull2float_rn(y.u64); // [한국어] RN: 최근접 반올림으로 u64→f32 변환
            break;
          case RM_OPTION:
            y.f32 = cuda_math::__ull2float_rn(y.u64); // [한국어] RM이지만 __ull2float_rn으로 처리 — 정밀한 반올림 미구현
            break;
          case RP_OPTION:
            y.f32 = cuda_math::__ull2float_rn(y.u64); // [한국어] RP이지만 __ull2float_rn으로 처리 — 정밀한 반올림 미구현
            break;
          default:
            break; // [한국어] 반올림 모드 미지정 시 변환 없이 유지
        }
        break;
      case 64:
        y.f64 = y.u64; // [한국어] u64→f64 변환: 암묵적 캐스트로 처리 (일부 정밀도 손실 가능)
        break;  // no internal implementation found
      default:
        assert(0); // [한국어] 지원하지 않는 목적지 폭
        break;
    }
  }

  // saturating an integer to 1 or 0?
  return y; // [한국어] 변환된 float 값이 담긴 레지스터 반환
}

/*
 * [한국어]
 * f2f - float 타입을 동일하거나 다른 float 타입으로 변환 (반올림/NaN/포화 처리 포함)
 *
 * @x: 변환 대상 레지스터 (f16 또는 f32 필드를 사용)
 * @from_width: 원본 float 비트 폭 (16이면 f16→f32, 그 외 f32 처리)
 * @to_width: 목적지 float 비트 폭 (이 함수에서는 직접 사용하지 않음)
 * @to_sign: 목적지 부호 (이 함수에서는 사용하지 않음)
 * @rounding_mode: PTX 반올림 모드 (RZI/RNI/RMI/RPI = 정수로 반올림하는 변형)
 * @saturation_mode: 1이면 결과를 [0.0, 1.0]으로 클램핑
 * @return: 반올림/포화 처리된 float 값을 담은 ptx_reg_t
 *
 * g_cvt_fn 테이블에서 f32 소스의 f32 목적지(행 9, 열 9) 또는 f16 소스의 f32 목적지에서 호출.
 * from_width==16이면 f16→f32 확장(반올림 없음), 그 외에는 f32→f32 반올림을 수행한다.
 * 비정규 부동소수점(지수부=0)은 부호 비트를 보존한 채 0으로 처리 (GPU HW 특성 모방).
 * NaN 결과는 0x7fffffff(quiet NaN)으로 정규화.
 * 포화 모드 활성화 시 음수→0, 1 초과→1로 클램핑하여 [0,1] 범위를 보장.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   cvt_impl() → g_cvt_fn[8or9][9]() → [f2f] → truncf/nearbyintf/floorf/ceilf
 */
ptx_reg_t f2f(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  ptx_reg_t y; // [한국어] 변환 결과를 담을 레지스터
  if (from_width == 16) {
    // [한국어] float16(f16)에서 float32(f32)로 확장하는 경우
    half_float::detail::uint16 val = x.u16; // [한국어] f16 비트 패턴을 uint16으로 읽어 half_float 라이브러리에 전달
    y.f32 = half_float::detail::half2float<float>(val); // [한국어] half_float 라이브러리로 f16→f32 변환 (정밀도 손실 없음)
  } else {
    // [한국어] from_width != 16인 경우: f32→f32 반올림 처리 (cvt.rn.f32.f32 등)
    switch (rounding_mode) { // [한국어] PTX 반올림 모드에 따라 f32 반올림 적용
      case RZI_OPTION:
        y.f32 = truncf(x.f32); // [한국어] RZI: 0 방향(truncate)으로 정수 경계까지 반올림
        break;
      case RNI_OPTION:
#if CUDART_VERSION >= 3000
        y.f32 = nearbyintf(x.f32); // [한국어] RNI(CUDART >= 3000): 현재 반올림 모드(기본=RN)로 정수 경계까지 반올림
#else
        y.f32 = cuda_math::__internal_nearbyintf(x.f32); // [한국어] RNI(CUDART < 3000): GPGPU-Sim 내부 nearbyintf 에뮬레이션
#endif
        break;
      case RMI_OPTION:
        if ((x.u32 & 0x7f800000) == 0) {
          y.u32 = x.u32 & 0x80000000;  // round denorm. FP to 0, keeping sign
          // [한국어] 비정규 부동소수점(지수부=0): GPU HW 특성상 0으로 변환, 단 부호 비트는 보존 (0x80000000 마스크)
        } else {
          y.f32 = floorf(x.f32); // [한국어] RMI: 음의 무한대 방향(내림)으로 정수 경계까지 반올림
        }
        break;
      case RPI_OPTION:
        if ((x.u32 & 0x7f800000) == 0) {
          y.u32 = x.u32 & 0x80000000;  // round denorm. FP to 0, keeping sign
          // [한국어] 비정규 부동소수점: 0으로 처리, 부호 비트 보존
        } else {
          y.f32 = ceilf(x.f32); // [한국어] RPI: 양의 무한대 방향(올림)으로 정수 경계까지 반올림
        }
        break;
      default:
        // [한국어] 반올림 모드 없음(단순 f32→f32 복사 또는 포화만 적용): 비정규 처리 후 복사
        if ((x.u32 & 0x7f800000) == 0) {
          y.u32 = x.u32 & 0x80000000;  // round denorm. FP to 0, keeping sign
          // [한국어] 비정규 부동소수점: 0으로 처리, 부호 비트 보존
        } else {
          y.f32 = x.f32; // [한국어] 일반 f32: 그대로 복사
        }
        break;
    }
#if CUDART_VERSION >= 3000
    if (isnanf(y.f32)) // [한국어] CUDART >= 3000: C 표준 isnanf()로 NaN 검사
#else
    if (cuda_math::__cuda___isnanf(y.f32)) // [한국어] CUDART < 3000: GPGPU-Sim 내부 NaN 검사 함수 사용
#endif
    {
      y.u32 = 0x7fffffff; // [한국어] NaN 결과를 quiet NaN 표준 비트 패턴 0x7fffffff으로 정규화 (PTX 규격)
    } else if (saturation_mode) {
      y.f32 = cuda_math::__saturatef(y.f32); // [한국어] 포화 모드: __saturatef()로 결과를 [0.0f, 1.0f] 범위로 클램핑
    }
  }

  return y; // [한국어] 반올림/포화 처리된 결과 레지스터 반환
}

/*
 * [한국어]
 * d2d - float64(double)를 float64로 변환 (반올림/NaN/포화 처리 포함)
 *
 * @x: 변환 대상 레지스터 (f64 필드를 사용)
 * @from_width: 원본 float 비트 폭 (반드시 64)
 * @to_width: 목적지 float 비트 폭 (반드시 64)
 * @to_sign: 목적지 부호 (사용하지 않음)
 * @rounding_mode: PTX 반올림 모드 (RZI/RNI/RMI/RPI)
 * @saturation_mode: 1이면 결과를 [0.0, 1.0]으로 클램핑
 * @return: 반올림/포화 처리된 double 값을 담은 ptx_reg_t
 *
 * g_cvt_fn 테이블의 f64 행(행 10), f64 열(열 10)에서 호출되는 double→double 변환 헬퍼.
 * f2f()의 double 버전이며, cvt.rni.f64.f64 같은 명령어를 처리한다.
 * NaN 결과는 0xfff8000000000000(double quiet NaN 표준값)으로 정규화.
 * 포화 모드는 __saturatef()에 double을 전달하여 [0.0, 1.0] 클램핑.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   cvt_impl() → g_cvt_fn[10][10]() → [d2d] → trunc/nearbyint/floor/ceil
 */
ptx_reg_t d2d(ptx_reg_t x, unsigned from_width, unsigned to_width, int to_sign,
              int rounding_mode, int saturation_mode) {
  ptx_reg_t y; // [한국어] 변환 결과를 담을 레지스터
  switch (rounding_mode) { // [한국어] PTX 반올림 모드에 따라 double 반올림 적용
    case RZI_OPTION:
      y.f64 = trunc(x.f64); // [한국어] RZI: 0 방향 절단(truncate toward zero)
      break;
    case RNI_OPTION:
#if CUDART_VERSION >= 3000
      y.f64 = nearbyint(x.f64); // [한국어] RNI(CUDART >= 3000): 현재 FP 환경의 반올림 모드로 정수 경계까지 반올림
#else
      y.f64 = cuda_math::__internal_nearbyintf(x.f64); // [한국어] RNI(CUDART < 3000): GPGPU-Sim 내부 nearbyint 에뮬레이션
#endif
      break;
    case RMI_OPTION:
      y.f64 = floor(x.f64); // [한국어] RMI: 음의 무한대 방향(내림, floor)
      break;
    case RPI_OPTION:
      y.f64 = ceil(x.f64); // [한국어] RPI: 양의 무한대 방향(올림, ceil)
      break;
    default:
      y.f64 = x.f64; // [한국어] 반올림 모드 없음: 원본 double 값 그대로 복사
      break;
  }
  if (std::isnan(y.f64)) { // [한국어] 반올림 결과가 NaN인지 검사
    y.u64 = 0xfff8000000000000ull; // [한국어] double quiet NaN 표준 비트 패턴으로 정규화 (PTX 규격 준수)
  } else if (saturation_mode) {
    y.f64 = cuda_math::__saturatef(y.f64); // [한국어] 포화 모드: 결과를 [0.0, 1.0] 범위로 클램핑
  }
  return y; // [한국어] 반올림/포화 처리된 double 결과 레지스터 반환
}

/*
 * [한국어]
 * g_cvt_fn - PTX cvt 명령어 타입 변환 디스패치 테이블 (11×11 함수 포인터 2D 배열)
 *
 * 행(row) = 소스 타입 인덱스 (type_info_key::type_decode()가 반환하는 src_fmt):
 *   0=S8,  1=S16, 2=S32, 3=S64  — signed 정수
 *   4=U8,  5=U16, 6=U32, 7=U64  — unsigned 정수
 *   8=F16, 9=F32, 10=F64         — float (f16은 행8, f32는 행9, f64는 행10)
 *
 * 열(col) = 목적지 타입 인덱스 (dst_fmt):
 *   0=S8,  1=S16, 2=S32, 3=S64  — signed 정수
 *   4=U8,  5=U16, 6=U32, 7=U64  — unsigned 정수
 *   8=F16, 9=F32, 10=F64         — float
 *
 * 각 셀에는 해당 변환을 수행하는 함수 포인터가 저장된다:
 *   NULL   = 소스와 목적지 타입이 동일 (no-op; cvt_impl이 g_cvt_fn==NULL이면 변환 생략)
 *   chop   = 상위 비트 절단 (넓은→좁은 정수)
 *   sext   = 부호 확장 (좁은 signed→넓은 signed/unsigned)
 *   sexd   = 목적지 폭 기준 부호 확장 (S32→S16 특수 케이스, SobelFilter 핵)
 *   zext   = 제로 확장 (좁은 unsigned→넓은)
 *   s2f    = signed 정수→float
 *   u2f    = unsigned 정수→float
 *   f2x    = float(32/16)→정수 또는 다른 float 폭
 *   f2f    = float→float 동일 또는 반올림
 *   d2x    = double→정수 또는 float
 *   d2d    = double→double 반올림
 *
 * cvt_impl()에서 src_fmt/dst_fmt를 인덱스로 이 테이블을 조회하여 변환 함수를 결정한다.
 * 설정자: 정적 초기화 (프로그램 시작 시 한 번만 초기화).
 * 읽는 자: cvt_impl() — PTX cvt 명령어 실행 시 매번 호출.
 */
ptx_reg_t (*g_cvt_fn[11][11])(ptx_reg_t x, unsigned from_width,
                              unsigned to_width, int to_sign, int rounding_mode,
                              int saturation_mode) = {
    // [한국어] 행 0 (소스=S8): S8 소스를 각 목적지로 변환 — 같은 타입은 NULL, 넓어지면 sext, float이면 s2f
    {NULL, sext, sext, sext, NULL, sext, sext, sext, s2f, s2f, s2f},
    // [한국어] 행 1 (소스=S16): S16 소스 — 좁아지면 chop, 같은 부호 확장이면 sext, float이면 s2f
    {chop, NULL, sext, sext, chop, NULL, sext, sext, s2f, s2f, s2f},
    // [한국어] 행 2 (소스=S32): S32 소스 — sexd는 S32→S16 특수(SobelFilter hack), 나머지 chop/sext
    {chop, sexd, NULL, sext, chop, chop, NULL, sext, s2f, s2f, s2f},
    // [한국어] 행 3 (소스=S64): S64 소스 — 모두 chop(좁아짐), 같은 타입 NULL, float이면 s2f
    {chop, chop, chop, NULL, chop, chop, chop, NULL, s2f, s2f, s2f},
    // [한국어] 행 4 (소스=U8): U8 소스 — 넓어지면 zext, 같은 타입 NULL, float이면 u2f
    {NULL, zext, zext, zext, NULL, zext, zext, zext, u2f, u2f, u2f},
    // [한국어] 행 5 (소스=U16): U16 소스 — 좁아지면 chop, 넓어지면 zext, float이면 u2f
    {chop, NULL, zext, zext, chop, NULL, zext, zext, u2f, u2f, u2f},
    // [한국어] 행 6 (소스=U32): U32 소스 — 좁아지면 chop, 넓어지면 zext, float이면 u2f
    {chop, chop, NULL, zext, chop, chop, NULL, zext, u2f, u2f, u2f},
    // [한국어] 행 7 (소스=U64): U64 소스 — 모두 chop(좁아짐), 같은 타입 NULL, float이면 u2f
    {chop, chop, chop, NULL, chop, chop, chop, NULL, u2f, u2f, u2f},
    // [한국어] 행 8 (소스=F16): F16 소스 — 정수 목적지는 f2x, F32 목적지는 f2f(확장), F64 목적지는 f2x(내부에서 분기)
    {f2x, f2x, f2x, f2x, f2x, f2x, f2x, f2x, NULL, f2f, f2x},
    // [한국어] 행 9 (소스=F32): F32 소스 — 정수 목적지는 f2x, F32→F32는 f2f(반올림/포화), F64는 f2x
    {f2x, f2x, f2x, f2x, f2x, f2x, f2x, f2x, f2x, f2f, f2x},
    // [한국어] 행 10 (소스=F64): F64 소스 — 정수/F32 목적지는 d2x, F64→F64는 d2d(반올림/포화)
    {d2x, d2x, d2x, d2x, d2x, d2x, d2x, d2x, d2x, d2x, d2d}};

/*
 * [한국어]
 * ptx_round - ptx_reg_t 데이터에 PTX 반올림 모드를 적용 (인플레이스 수정)
 *
 * @data: 반올림 대상 레지스터 값 (참조로 전달, 인플레이스 수정)
 * @rounding_mode: PTX 반올림 옵션 (RN_OPTION/RZI_OPTION/RNI_OPTION/RMI_OPTION/RPI_OPTION)
 * @type: PTX 데이터 타입 (F16_TYPE/F32_TYPE/F64_TYPE/FF64_TYPE 등)
 *
 * PTX 명령어(add, mul, fma 등)의 반올림 모드 수식어(.rn/.rz/.rm/.rp)를 처리하는
 * 인플레이스 반올림 적용 함수. cvt_impl이 아닌 다른 연산 impl에서 후처리로 호출될 수 있다.
 * RN_OPTION(가장 가까운 짝수)은 C 기본 부동소수점 반올림과 동일하므로 조기 반환.
 * 정수 타입에 반올림 모드를 적용하려 하면 에러(assert)를 발생시킨다 — PTX 규격 위반.
 * NaN 결과는 F32→0x7fffffff, F64→0xfff8000000000000으로 정규화하여 PTX 규격 준수.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드 — 단일 스레드, 동기화 불필요.
 *
 * 호출 체인:
 *   (여러 PTX impl 함수들) → [ptx_round] → truncf/nearbyintf/floorf/ceilf
 */
void ptx_round(ptx_reg_t &data, int rounding_mode, int type) {
  if (rounding_mode == RN_OPTION) { // [한국어] RN(가장 가까운 짝수) 모드는 C 기본 FP 반올림과 동일하므로 처리 불필요 — 조기 반환
    return;
  }
  switch (rounding_mode) { // [한국어] RN 이외의 반올림 모드에 따라 분기 처리
    case RZI_OPTION:
      // [한국어] RZI(Round to Zero, Integer): 0 방향 절단 — truncate toward zero
      switch (type) { // [한국어] 타입에 따라 적절한 truncate 함수 선택
        case S8_TYPE:
        case S16_TYPE:
        case S32_TYPE:
        case S64_TYPE:
        case U8_TYPE:
        case U16_TYPE:
        case U32_TYPE:
        case U64_TYPE:
          printf("Trying to round an integer??\n"); // [한국어] 정수 타입에 반올림 모드 적용 — PTX 규격 위반
          assert(0); // [한국어] 정수 반올림 시도는 항상 오류
          break;
        case F16_TYPE:
          data.f16 = truncf(data.f16); // [한국어] F16: 0 방향으로 절단하여 가장 가까운 정수 경계로 반올림
          break;  // assert(0); break;
        case F32_TYPE:
          data.f32 = truncf(data.f32); // [한국어] F32: 0 방향으로 절단
          break;
        case F64_TYPE:
        case FF64_TYPE:
          if (data.f64 < 0)
            data.f64 = ceil(data.f64);  // negative
          // [한국어] F64 음수: 0 방향 절단은 ceil(올림)과 동일 — 예: -1.7 → -1.0
          else
            data.f64 = floor(data.f64);  // positive
          // [한국어] F64 양수: 0 방향 절단은 floor(내림)과 동일 — 예: 1.7 → 1.0
          break;
        default:
          assert(0); // [한국어] 지원하지 않는 타입
          break;
      }
      break;
    case RNI_OPTION:
      // [한국어] RNI(Round Nearest, Integer): 가장 가까운 정수로 반올림 (banker's rounding)
      switch (type) { // [한국어] 타입에 따라 nearbyint 계열 함수 선택
        case S8_TYPE:
        case S16_TYPE:
        case S32_TYPE:
        case S64_TYPE:
        case U8_TYPE:
        case U16_TYPE:
        case U32_TYPE:
        case U64_TYPE:
          printf("Trying to round an integer??\n"); // [한국어] 정수 타입에 반올림 적용 — PTX 규격 위반
          assert(0);
          break;
        case F16_TYPE:  // assert(0); break;
#if CUDART_VERSION >= 3000
          data.f16 = nearbyintf(data.f16); // [한국어] CUDART >= 3000: C 표준 nearbyintf로 F16 최근접 반올림
#else
          data.f16 = cuda_math::__cuda_nearbyintf(data.f16); // [한국어] CUDART < 3000: GPGPU-Sim 내부 구현 사용
#endif
          break;
        case F32_TYPE:
#if CUDART_VERSION >= 3000
          data.f32 = nearbyintf(data.f32); // [한국어] CUDART >= 3000: C 표준 nearbyintf로 F32 최근접 반올림
#else
          data.f32 = cuda_math::__cuda_nearbyintf(data.f32); // [한국어] CUDART < 3000: GPGPU-Sim 내부 구현 사용
#endif
          break;
        case F64_TYPE:
        case FF64_TYPE:
          data.f64 = round(data.f64); // [한국어] F64: C 표준 round()로 최근접 정수로 반올림 (0.5는 절댓값 큰 방향)
          break;
        default:
          assert(0); // [한국어] 지원하지 않는 타입
          break;
      }
      break;
    case RMI_OPTION:
      // [한국어] RMI(Round Minus Infinity, Integer): 음의 무한대 방향(내림, floor)으로 반올림
      switch (type) { // [한국어] 타입에 따라 floorf 계열 함수 선택
        case S8_TYPE:
        case S16_TYPE:
        case S32_TYPE:
        case S64_TYPE:
        case U8_TYPE:
        case U16_TYPE:
        case U32_TYPE:
        case U64_TYPE:
          printf("Trying to round an integer??\n"); // [한국어] 정수 타입에 반올림 적용 — PTX 규격 위반
          assert(0);
          break;
        case F16_TYPE:
          data.f16 = floorf(data.f16); // [한국어] F16: 음의 무한대 방향(내림)으로 반올림
          break;  // assert(0); break;
        case F32_TYPE:
          data.f32 = floorf(data.f32); // [한국어] F32: floorf()로 내림 반올림
          break;
        case F64_TYPE:
        case FF64_TYPE:
          data.f64 = floor(data.f64); // [한국어] F64: floor()로 내림 반올림
          break;
        default:
          assert(0); // [한국어] 지원하지 않는 타입
          break;
      }
      break;
    case RPI_OPTION:
      // [한국어] RPI(Round Plus Infinity, Integer): 양의 무한대 방향(올림, ceil)으로 반올림
      switch (type) { // [한국어] 타입에 따라 ceilf 계열 함수 선택
        case S8_TYPE:
        case S16_TYPE:
        case S32_TYPE:
        case S64_TYPE:
        case U8_TYPE:
        case U16_TYPE:
        case U32_TYPE:
        case U64_TYPE:
          printf("Trying to round an integer??\n"); // [한국어] 정수 타입에 반올림 적용 — PTX 규격 위반
          assert(0);
          break;
        case F16_TYPE:
          data.f16 = ceilf(data.f16); // [한국어] F16: 양의 무한대 방향(올림)으로 반올림
          break;  // assert(0); break;
        case F32_TYPE:
          data.f32 = ceilf(data.f32); // [한국어] F32: ceilf()로 올림 반올림
          break;
        case F64_TYPE:
        case FF64_TYPE:
          data.f64 = ceil(data.f64); // [한국어] F64: ceil()로 올림 반올림
          break;
        default:
          assert(0); // [한국어] 지원하지 않는 타입
          break;
      }
      break;
    default:
      break; // [한국어] 그 외 반올림 모드(예: 미지정)는 아무것도 하지 않음
  }

  if (type == F32_TYPE) { // [한국어] F32 결과에 대한 NaN 정규화 검사
#if CUDART_VERSION >= 3000
    if (isnanf(data.f32)) // [한국어] CUDART >= 3000: C 표준 isnanf()로 F32 NaN 검사
#else
    if (cuda_math::__cuda___isnanf(data.f32)) // [한국어] CUDART < 3000: GPGPU-Sim 내부 NaN 검사
#endif
    {
      data.u32 = 0x7fffffff; // [한국어] F32 NaN을 표준 quiet NaN 비트 패턴으로 정규화
    }
  }
  if ((type == F64_TYPE) || (type == FF64_TYPE)) { // [한국어] F64/FF64 결과에 대한 NaN 정규화 검사
    if (std::isnan(data.f64)) { // [한국어] C++ 표준 std::isnan()으로 double NaN 검사
      data.u64 = 0xfff8000000000000ull; // [한국어] F64 NaN을 표준 quiet NaN 비트 패턴으로 정규화
    }
  }
}

/*
 * [한국어]
 * ptx_saturate - ptx_reg_t 데이터에 PTX 포화(saturation) 모드를 인플레이스 적용
 *
 * @data: 포화 적용 대상 레지스터 값 (참조로 전달, 인플레이스 수정)
 * @saturation_mode: 0이면 포화 비활성(즉시 반환), 1이면 포화 적용
 * @type: PTX 데이터 타입 (F16_TYPE/F32_TYPE/F64_TYPE 등)
 *
 * PTX 명령어의 .sat 수식어를 처리하는 인플레이스 포화 함수.
 * ptx_round() 이후에 호출되어 [0.0, 1.0] 범위 클램핑을 수행한다.
 * 정수 타입에 포화를 적용하려 하면 에러(assert)를 발생시킨다 — PTX에서 정수 sat는
 * 명령어별(예: mad.sat)로 별도 처리하며 이 함수를 통하지 않는다.
 * float 타입은 단순 비교로 [0.0f, 1.0f] 경계 적용 (NaN에 대한 보장은 ptx_round 후 이미 처리됨).
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   (여러 PTX impl 함수들) → [ptx_saturate]
 */
void ptx_saturate(ptx_reg_t &data, int saturation_mode, int type) {
  if (!saturation_mode) { // [한국어] 포화 모드가 비활성(0)이면 아무것도 하지 않고 즉시 반환
    return;
  }
  switch (type) { // [한국어] 타입에 따라 포화 처리 분기
    case S8_TYPE:
    case S16_TYPE:
    case S32_TYPE:
    case S64_TYPE:
    case U8_TYPE:
    case U16_TYPE:
    case U32_TYPE:
    case U64_TYPE:
      printf("Trying to clamp an integer to 1??\n"); // [한국어] 정수 타입에 이 함수로 포화를 시도하는 것은 PTX 규격 위반
      assert(0); // [한국어] 정수 포화는 mad.sat 등 명령어별 별도 경로로 처리해야 함
      break;
    case F16_TYPE:                           // assert(0); break;
      if (data.f16 > 1.0f) data.f16 = 1.0f;  // negative
      // [한국어] F16 값이 1.0 초과 시 1.0으로 클램핑 (주석의 "negative"는 오기 — 상한 처리)
      if (data.f16 < 0.0f) data.f16 = 0.0f;  // positive
      // [한국어] F16 값이 0.0 미만 시 0.0으로 클램핑 (주석의 "positive"는 오기 — 하한 처리)
      break;
    case F32_TYPE:
      if (data.f32 > 1.0f) data.f32 = 1.0f;  // negative
      // [한국어] F32 상한: 1.0 초과 시 1.0으로 포화
      if (data.f32 < 0.0f) data.f32 = 0.0f;  // positive
      // [한국어] F32 하한: 0.0 미만 시 0.0으로 포화
      break;
    case F64_TYPE:
    case FF64_TYPE:
      if (data.f64 > 1.0f) data.f64 = 1.0f;  // negative
      // [한국어] F64 상한: 1.0 초과 시 1.0으로 포화
      if (data.f64 < 0.0f) data.f64 = 0.0f;  // positive
      // [한국어] F64 하한: 0.0 미만 시 0.0으로 포화
      break;
    default:
      assert(0); // [한국어] 지원하지 않는 타입 — 구현 오류
      break;
  }
}

/*
 * [한국어]
 * cvt_impl - PTX `cvt` 명령어 구현: 소스 타입에서 목적지 타입으로의 변환 디스패치
 *
 * @pI: 실행 중인 PTX 명령어 객체 (목적지/소스 오퍼랜드, to_type, from_type, 반올림/포화 모드 포함)
 * @thread: 현재 실행 중인 PTX 스레드 정보 (레지스터 파일, 메모리 등 접근)
 *
 * PTX `cvt.{rounding}.{saturation}.{to_type}.{from_type}` 명령어의 기능 시뮬레이션 구현.
 * 동작 흐름:
 *   1. 명령어에서 to_type, from_type, 반올림 모드, 포화 모드를 추출
 *   2. type_info_key::type_decode()로 타입을 11개 인덱스(src_fmt/dst_fmt)로 변환
 *   3. 소스 오퍼랜드 값을 레지스터에서 읽음
 *   4. .neg 수식어가 있으면 소스 값을 부정(negation)
 *   5. g_cvt_fn[src_fmt][dst_fmt]를 조회하여 실제 변환 함수 호출
 *   6. 변환 결과를 목적지 레지스터에 기록
 * NULL 셀(동일 타입)은 변환 없이 원본 값을 그대로 목적지에 기록.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드 — ptx_thread_info::execute()에서 1회 호출.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [cvt_impl] → g_cvt_fn[src][dst]() → thread->set_operand_value()
 */
void cvt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst(); // [한국어] 목적지 오퍼랜드 (변환 결과가 기록될 레지스터)
  const operand_info &src1 = pI->src1(); // [한국어] 소스 오퍼랜드 (변환할 원본 값의 레지스터)
  unsigned to_type = pI->get_type(); // [한국어] 목적지 PTX 타입 (예: S32_TYPE, F32_TYPE 등)
  unsigned from_type = pI->get_type2(); // [한국어] 소스 PTX 타입 (cvt 명령어는 두 개의 타입을 가짐)
  unsigned rounding_mode = pI->rounding_mode(); // [한국어] PTX 반올림 모드 수식어 (RN/RZ/RM/RP/RNI/RZI 등)
  unsigned saturation_mode = pI->saturation_mode(); // [한국어] PTX 포화 모드 수식어 (.sat)

  //   if ( to_type == F16_TYPE || from_type == F16_TYPE )
  //      abort();

  int to_sign, from_sign; // [한국어] 목적지/소스 부호 여부 (type_decode가 설정)
  size_t from_width, to_width; // [한국어] 소스/목적지 비트 폭 (type_decode가 설정)
  unsigned src_fmt =
      type_info_key::type_decode(from_type, from_width, from_sign);
  // [한국어] 소스 타입을 g_cvt_fn 테이블 행 인덱스(0~10)로 변환 + from_width/from_sign 설정
  unsigned dst_fmt = type_info_key::type_decode(to_type, to_width, to_sign);
  // [한국어] 목적지 타입을 g_cvt_fn 테이블 열 인덱스(0~10)로 변환 + to_width/to_sign 설정

  ptx_reg_t data = thread->get_operand_value(src1, dst, from_type, thread, 1);
  // [한국어] 소스 레지스터에서 from_type 타입으로 값을 읽어 data에 저장

  if (pI->is_neg()) { // [한국어] .neg 수식어가 있으면 소스 값을 부정(negation) 적용
    switch (from_type) {
      // Default to f32 for now, need to add support for others
      case S8_TYPE:
      case U8_TYPE:
      case B8_TYPE:
        data.s8 = -data.s8; // [한국어] 8비트 값 부정 (s8 필드 사용)
        break;
      case S16_TYPE:
      case U16_TYPE:
      case B16_TYPE:
        data.s16 = -data.s16; // [한국어] 16비트 값 부정
        break;
      case S32_TYPE:
      case U32_TYPE:
      case B32_TYPE:
        data.s32 = -data.s32; // [한국어] 32비트 값 부정
        break;
      case S64_TYPE:
      case U64_TYPE:
      case B64_TYPE:
        data.s64 = -data.s64; // [한국어] 64비트 값 부정
        break;
      case F16_TYPE:
        data.f16 = -data.f16; // [한국어] F16 값 부정 (부호 비트 반전)
        break;
      case F32_TYPE:
        data.f32 = -data.f32; // [한국어] F32 값 부정 (부호 비트 반전)
        break;
      case F64_TYPE:
      case FF64_TYPE:
        data.f64 = -data.f64; // [한국어] F64 값 부정 (부호 비트 반전)
        break;
      default:
        assert(0); // [한국어] 지원하지 않는 타입에 .neg 적용 — 구현 오류
    }
  }

  if (g_cvt_fn[src_fmt][dst_fmt] != NULL) { // [한국어] 동일 타입(NULL 셀)이 아닌 경우에만 변환 함수 호출
    ptx_reg_t result = g_cvt_fn[src_fmt][dst_fmt](
        data, from_width, to_width, to_sign, rounding_mode, saturation_mode);
    // [한국어] g_cvt_fn 테이블에서 src_fmt/dst_fmt 조합에 맞는 변환 함수를 호출하여 결과 획득
    data = result; // [한국어] 변환 결과를 data에 갱신
  }

  thread->set_operand_value(dst, data, to_type, thread, pI);
  // [한국어] 변환된 데이터를 to_type 타입으로 목적지 레지스터에 기록
}

/*
 * [한국어]
 * cvta_impl - PTX `cvta` 명령어 구현: 제네릭 주소와 특정 메모리 공간 주소 간 변환
 *
 * @pI: 실행 중인 PTX 명령어 객체 (공간 타입, .to 수식어, 오퍼랜드 포함)
 * @thread: 현재 실행 중인 PTX 스레드 (SM ID, HW 스레드 ID, 레지스터 파일 접근)
 *
 * PTX `cvta.{to.}space.type` 명령어를 구현한다.
 * 두 가지 방향으로 동작:
 *   (1) .to 수식어 있음 (to_non_generic=true): 제네릭 주소 → 특정 공간 주소 변환
 *       shared/local/global 제네릭 주소를 해당 공간의 직접 주소로 변환
 *   (2) .to 수식어 없음 (to_non_generic=false): 특정 공간 주소 → 제네릭 주소 변환
 *       shared/local/global 직접 주소를 제네릭 주소로 변환
 * local 공간의 경우 제네릭→직접 변환 시 스택 포인터를 더해 실제 주소를 계산한다.
 * 변환된 주소는 u64 필드에 저장되어 목적지 레지스터에 기록된다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [cvta_impl] → generic_to_shared/local/global
 */
void cvta_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t data; // [한국어] 사용되지 않는 임시 변수 (레거시 코드 잔재)

  const operand_info &dst = pI->dst(); // [한국어] 변환된 주소가 기록될 목적지 레지스터
  const operand_info &src1 = pI->src1(); // [한국어] 변환할 원본 주소를 담은 소스 레지스터
  memory_space_t space = pI->get_space(); // [한국어] 명령어가 지정한 메모리 공간 (shared/local/global)
  bool to_non_generic = pI->is_to(); // [한국어] .to 수식어 유무 — true이면 제네릭→특정, false이면 특정→제네릭

  unsigned i_type = pI->get_type(); // [한국어] 주소 값의 정수 타입 (U32 또는 U64)
  ptx_reg_t from_addr = thread->get_operand_value(src1, dst, i_type, thread, 1);
  // [한국어] 소스 레지스터에서 원본 주소 값을 읽음
  addr_t from_addr_hw = (addr_t)from_addr.u64; // [한국어] ptx_reg_t에서 HW 주소 타입으로 변환
  addr_t to_addr_hw = 0; // [한국어] 변환된 목적지 주소 초기화
  unsigned smid = thread->get_hw_sid(); // [한국어] 현재 SM(Streaming Multiprocessor) ID — shared 공간 주소 계산에 필요
  unsigned hwtid = thread->get_hw_tid(); // [한국어] HW 스레드 ID — local 공간 주소 계산에 필요

  if (to_non_generic) {
    // [한국어] .to 수식어: 제네릭 주소 → 특정 메모리 공간의 직접 주소로 변환
    switch (space.get_type()) { // [한국어] 목적지 공간 타입에 따라 변환 함수 선택
      case shared_space:
        to_addr_hw = generic_to_shared(smid, from_addr_hw); // [한국어] 제네릭→공유메모리 직접 주소 (SM ID 기반 오프셋)
        break;
      case local_space:
        to_addr_hw = generic_to_local(smid, hwtid, from_addr_hw); // [한국어] 제네릭→로컬메모리 직접 주소 (SM ID + HW 스레드 ID 기반)
        break;
      case global_space:
        to_addr_hw = generic_to_global(from_addr_hw); // [한국어] 제네릭→글로벌메모리 직접 주소 (현재 구현에서는 no-op에 가까움)
        break;
      default:
        abort(); // [한국어] 지원하지 않는 공간 타입 — 구현 오류
    }
  } else {
    // [한국어] .to 수식어 없음: 특정 메모리 공간의 직접 주소 → 제네릭 주소로 변환
    switch (space.get_type()) { // [한국어] 소스 공간 타입에 따라 제네릭 변환 함수 선택
      case shared_space:
        to_addr_hw = shared_to_generic(smid, from_addr_hw); // [한국어] 공유메모리→제네릭 주소 변환
        break;
      case local_space:
        to_addr_hw = local_to_generic(smid, hwtid, from_addr_hw) +
                     thread->get_local_mem_stack_pointer();
        // [한국어] 로컬메모리→제네릭 주소 변환 + 스택 포인터 추가
        // [한국어] 스택 포인터를 더해야 함수 인자로 포인터를 전달할 때 올바른 주소가 됨
        break;  // add stack ptr here so that it can be passed as a pointer at
                // function call
      case global_space:
        to_addr_hw = global_to_generic(from_addr_hw); // [한국어] 글로벌메모리→제네릭 주소 변환
        break;
      default:
        abort(); // [한국어] 지원하지 않는 공간 타입 — 구현 오류
    }
  }

  ptx_reg_t to_addr; // [한국어] 변환된 주소를 담을 레지스터 선언
  to_addr.u64 = to_addr_hw; // [한국어] HW 주소를 u64 필드에 저장
  thread->set_reg(dst.get_symbol(), to_addr); // [한국어] 변환된 주소를 목적지 레지스터에 기록
}

/*
 * [한국어]
 * div_impl - PTX `div` 명령어 구현: 모든 타입의 나눗셈
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, 소스1, 소스2, 타입 정보 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `div.type d, a, b` 명령어를 구현한다. d = a / b.
 * signed, unsigned, bitfield, float 모든 타입을 지원하며 타입별로 해당 C 나눗셈 연산자를 사용.
 * 0으로 나누기, 오버플로우, NaN 처리는 C 런타임/IEEE 754에 위임한다 (시뮬레이터가 별도 처리하지 않음).
 * F16 나눗셈은 assert(0)이 주석 처리되어 있어 현재 활성화 상태 — half_float 연산자 사용.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [div_impl] → thread->set_operand_value()
 */
void div_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t data; // [한국어] 나눗셈 결과를 담을 레지스터

  const operand_info &dst = pI->dst(); // [한국어] 목적지 레지스터 (d)
  const operand_info &src1 = pI->src1(); // [한국어] 피제수(dividend) 오퍼랜드 (a)
  const operand_info &src2 = pI->src2(); // [한국어] 제수(divisor) 오퍼랜드 (b)

  unsigned i_type = pI->get_type(); // [한국어] PTX 데이터 타입

  ptx_reg_t src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] 피제수 값을 레지스터에서 읽음
  ptx_reg_t src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1); // [한국어] 제수 값을 레지스터에서 읽음

  switch (i_type) { // [한국어] PTX 타입에 따라 해당 C 타입의 나눗셈 수행
    case S8_TYPE:
      data.s8 = src1_data.s8 / src2_data.s8; // [한국어] signed 8비트 정수 나눗셈
      break;
    case S16_TYPE:
      data.s16 = src1_data.s16 / src2_data.s16; // [한국어] signed 16비트 정수 나눗셈
      break;
    case S32_TYPE:
      data.s32 = src1_data.s32 / src2_data.s32; // [한국어] signed 32비트 정수 나눗셈
      break;
    case S64_TYPE:
      data.s64 = src1_data.s64 / src2_data.s64; // [한국어] signed 64비트 정수 나눗셈
      break;
    case U8_TYPE:
      data.u8 = src1_data.u8 / src2_data.u8; // [한국어] unsigned 8비트 정수 나눗셈
      break;
    case U16_TYPE:
      data.u16 = src1_data.u16 / src2_data.u16; // [한국어] unsigned 16비트 정수 나눗셈
      break;
    case U32_TYPE:
      data.u32 = src1_data.u32 / src2_data.u32; // [한국어] unsigned 32비트 정수 나눗셈
      break;
    case U64_TYPE:
      data.u64 = src1_data.u64 / src2_data.u64; // [한국어] unsigned 64비트 정수 나눗셈
      break;
    case B8_TYPE:
      data.u8 = src1_data.u8 / src2_data.u8; // [한국어] 비트필드 8비트 — unsigned로 처리
      break;
    case B16_TYPE:
      data.u16 = src1_data.u16 / src2_data.u16; // [한국어] 비트필드 16비트 — unsigned로 처리
      break;
    case B32_TYPE:
      data.u32 = src1_data.u32 / src2_data.u32; // [한국어] 비트필드 32비트 — unsigned로 처리
      break;
    case B64_TYPE:
      data.u64 = src1_data.u64 / src2_data.u64; // [한국어] 비트필드 64비트 — unsigned로 처리
      break;
    case F16_TYPE:
      data.f16 = src1_data.f16 / src2_data.f16; // [한국어] F16 부동소수점 나눗셈 (half_float 라이브러리의 연산자)
      break;  // assert(0); break;
    case F32_TYPE:
      data.f32 = src1_data.f32 / src2_data.f32; // [한국어] F32 IEEE 754 부동소수점 나눗셈
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = src1_data.f64 / src2_data.f64; // [한국어] F64 IEEE 754 배정밀도 부동소수점 나눗셈
      break;
    default:
      assert(0); // [한국어] 지원하지 않는 타입 — 구현 오류
      break;
  }
  thread->set_operand_value(dst, data, i_type, thread, pI); // [한국어] 나눗셈 결과를 목적지 레지스터에 기록
}

/*
 * [한국어]
 * dp4a_impl - PTX `dp4a` 명령어 구현 (미구현 스텁)
 *
 * @pI: 실행 중인 PTX 명령어 (미사용)
 * @thread: 현재 실행 중인 PTX 스레드 (미사용)
 *
 * PTX `dp4a` (Dot Product of 4 8-bit integers Accumulated to 32-bit integer) 명령어.
 * Volta 이상 GPU에서 INT8 행렬 연산 가속에 사용되는 명령어이나, GPGPU-Sim에서 미구현.
 * 호출 시 에러 메시지를 출력하고 assert(0)으로 시뮬레이션을 종료한다.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [dp4a_impl] → assert(0)
 */
void dp4a_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  printf("DP4A instruction not implemented yet"); // [한국어] dp4a 명령어 미구현 — 실행 시 오류 메시지 출력
  assert(0); // [한국어] 미구현 명령어 실행 시 시뮬레이션 강제 종료
}

/*
 * [한국어]
 * ex2_impl - PTX `ex2` 명령어 구현: 2의 x제곱 근사 (F32만 지원)
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, 소스1, 타입 정보 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `ex2.approx.f32 d, a` 명령어를 구현한다. d = 2^a (F32).
 * GPU HW는 SFU(Special Function Unit)에서 ex2를 근사값으로 계산하는데,
 * GPGPU-Sim은 cuda_math::__powf(2.0, a)로 소프트웨어 에뮬레이션한다.
 * F32 이외 타입은 지원하지 않아 오류를 발생시킨다.
 * AccelWattch 전력 모델에서 SFU 연산으로 카운팅될 수 있다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [ex2_impl] → cuda_math::__powf
 */
void ex2_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data; // [한국어] src2_data는 미사용 — 레거시 변수 선언
  const operand_info &dst = pI->dst(); // [한국어] 결과가 기록될 목적지 레지스터
  const operand_info &src1 = pI->src1(); // [한국어] 지수(exponent) 소스 오퍼랜드 (a)

  unsigned i_type = pI->get_type(); // [한국어] PTX 데이터 타입 (F32_TYPE만 유효)

  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] 지수 소스 값을 레지스터에서 읽음

  switch (i_type) { // [한국어] 타입 확인 — F32만 지원
    case F32_TYPE:
      data.f32 = cuda_math::__powf(2.0, src1_data.f32); // [한국어] 2^a 계산: cuda_math의 __powf 사용 (GPU SFU ex2 에뮬레이션)
      break;
    default:
      printf("Execution error: type mismatch with instruction\n"); // [한국어] F32 이외 타입에 ex2 적용 — 타입 오류
      assert(0); // [한국어] 지원하지 않는 타입으로 시뮬레이션 강제 종료
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI); // [한국어] 2^a 결과를 목적지 레지스터에 기록
}

/*
 * [한국어]
 * exit_impl - PTX `exit` 명령어 구현: 스레드를 시뮬레이션에서 종료
 *
 * @pI: 실행 중인 PTX 명령어 (미사용)
 * @thread: 현재 실행 중인 PTX 스레드 (종료 상태를 설정할 대상)
 *
 * PTX `exit` 명령어를 구현한다. 해당 스레드의 실행을 종료시킨다.
 * GPU에서 스레드가 exit을 실행하면 이후 명령어를 더 이상 fetch/execute하지 않는다.
 * GPGPU-Sim에서는 세 단계로 처리:
 *   1. set_done(): 스레드를 완료 상태로 표시 — 이후 cycle()에서 이 스레드를 실행하지 않음
 *   2. exitCore(): SM(shader core)에 스레드 종료를 통보 — warp의 active mask 업데이트
 *   3. registerExit(): 전체 시뮬레이션에 종료된 스레드 수를 보고 — 커널 완료 판단에 사용
 * 워프 내 모든 스레드가 exit을 실행하면 해당 워프가 완료되고, 모든 워프가 완료되면 커널이 종료된다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [exit_impl] → thread->set_done() → thread->exitCore() → thread->registerExit()
 */
void exit_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  thread->set_done(); // [한국어] 스레드를 완료 상태로 표시 — 이후 cycle()에서 이 스레드를 실행 스케줄에서 제외
  thread->exitCore(); // [한국어] SM에 스레드 종료 통보 — 해당 워프의 SIMT active mask에서 이 스레드 비트를 클리어
  thread->registerExit(); // [한국어] 전역 완료 카운터 업데이트 — 모든 스레드 종료 시 커널 완료를 판단하는 데 사용
}

void mad_def(const ptx_instruction *pI, ptx_thread_info *thread,
             bool use_carry = false);

/*
 * [한국어]
 * fma_impl - PTX `fma` 명령어 구현: Fused Multiply-Add
 *
 * @pI: 실행 중인 PTX 명령어 (a, b, c 오퍼랜드, 반올림 모드, 포화 모드 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `fma.rn.f32 d, a, b, c` 등 명령어를 구현한다. d = a*b + c.
 * fma와 mad는 기능적으로 동일하므로(use_carry=false로 mad_def 호출) mad_def에 위임한다.
 * GPU HW에서 fma는 단일 연산으로 반올림 오류 없이 곱셈과 덧셈을 수행하는 정밀 연산이나,
 * GPGPU-Sim의 기능 시뮬레이션에서는 mad_def와 동일한 소프트웨어 경로를 사용한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [fma_impl] → mad_def(use_carry=false)
 */
void fma_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  mad_def(pI, thread); // [한국어] fma는 mad와 동일한 구현(use_carry=false) — mad_def에 위임
}

/*
 * [한국어]
 * isspacep_impl - PTX `isspacep` 명령어 구현: 주소가 특정 메모리 공간에 속하는지 검사
 *
 * @pI: 실행 중인 PTX 명령어 (목적지 pred 레지스터, 소스 주소, 공간 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `isspacep.shared p, a` 등 명령어를 구현한다.
 * 주어진 주소(a)가 명령어에 지정된 메모리 공간(shared/local/global)에 속하면 p=1, 아니면 p=0.
 * 결과는 predicate 레지스터에 저장되어 이후 분기 명령어 등에서 사용된다.
 * 주의: switch 문에 break가 없어 fall-through가 발생 — 버그 가능성 있음.
 *       실제로는 global_space 케이스까지 항상 실행되어 t = isspace_global(addr)으로 덮어쓰임.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [isspacep_impl] → isspace_shared/local/global
 */
void isspacep_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a; // [한국어] 검사할 주소를 담을 레지스터
  bool t = false; // [한국어] 공간 검사 결과: 해당 공간이면 true, 아니면 false

  const operand_info &dst = pI->dst(); // [한국어] 결과가 기록될 predicate 목적지 레지스터
  const operand_info &src1 = pI->src1(); // [한국어] 검사할 주소를 담은 소스 레지스터
  memory_space_t space = pI->get_space(); // [한국어] 명령어에 지정된 메모리 공간 (shared/local/global)

  a = thread->get_reg(src1.get_symbol()); // [한국어] 소스 레지스터에서 주소 값을 직접 읽음 (get_operand_value 대신 get_reg 사용)
  addr_t addr = (addr_t)a.u64; // [한국어] ptx_reg_t의 u64 필드를 HW 주소 타입으로 변환
  unsigned smid = thread->get_hw_sid(); // [한국어] SM ID — shared/local 공간 검사에 필요
  unsigned hwtid = thread->get_hw_tid(); // [한국어] HW 스레드 ID — local 공간 검사에 필요

  switch (space.get_type()) { // [한국어] 명령어에 지정된 메모리 공간에 따라 검사 함수 선택
    // [한국어] 주의: 각 case에 break가 없어 fall-through 발생 — 아래 케이스까지 모두 실행됨 (버그 의심)
    case shared_space:
      t = isspace_shared(smid, addr); // [한국어] 주소가 이 SM의 공유메모리 범위에 속하는지 검사
    case local_space:
      t = isspace_local(smid, hwtid, addr); // [한국어] 주소가 이 스레드의 로컬메모리 범위에 속하는지 검사 (fall-through로 항상 실행)
    case global_space:
      t = isspace_global(addr); // [한국어] 주소가 글로벌메모리 범위에 속하는지 검사 (fall-through로 항상 마지막에 실행 — t를 덮어씀)
    default:
      abort(); // [한국어] 지원하지 않는 공간 타입 — 구현 오류
  }

  ptx_reg_t p; // [한국어] predicate 결과를 담을 레지스터 선언
  p.pred = t ? 1 : 0; // [한국어] bool 결과를 predicate 필드(1/0)로 변환

  thread->set_reg(dst.get_symbol(), p); // [한국어] 검사 결과를 목적지 predicate 레지스터에 기록
}

/*
 * ============================================================================
 * [한국어 설명] src/cuda-sim/instructions.cc lines 5001-8677: 잔여 PTX 명령어 구현
 * ============================================================================
 *
 * 이 범위는 GPGPU-Sim 기능 시뮬레이션의 PTX 명령어 시맨틱 중 후반부를 담당한다.
 * 앞선 범위(1-5000)에서 산술/비트/분기/변환/메모리 헬퍼 등을 다루었다면,
 * 여기서는 다음 명령어들과 각종 보조 함수들을 구현한다:
 *
 *   - 메모리/Tensor Core: decode_space, ld_exec, ld/ldu, mma_st/mma_ld
 *   - 수학/곱셈: lg2, mad24, mad/madp/madc/mad_def, mul24, mul, rcp, rsqrt,
 *              sad, sin, sqrt
 *   - 선택/비교: max, min, selp, setp, set, slct
 *   - 데이터 이동/비트: mov, neg, nandn, norn, not, or, orn, popc, prmt, shf,
 *                     shl, shr, xor
 *   - 제어/동기화: membar, ret/retp, ssy, sst, st, sub/subc, nop
 *   - 서피스/텍스처/비디오/투표: suld/sured/sust/suq, tex/txq,
 *                              vabsdiff/vadd/vmad/vmax/vmin/vset/vshl/vshr/vsub,
 *                              vote, activemask
 *   - 공통 헬퍼: isFloat, CmpOp, isNaN, read_byte, prmt_mode_present,
 *              reduce_precision, wrap/clamp, tex_linf_sampling,
 *              textureNormalizeOutput, srcOperandModifiers,
 *              video_mem_instruction, inst_not_implemented
 *
 * 대부분의 명령어는 단일 스레드 관점에서 기능적 결과를 생성하며,
 * 실제 사이클-레벨 동작(파이프라인 레이턴시, 메모리 타이밍, 베리어 동기화 등)은
 * 상위 타이밍 모델(src/gpgpu-sim/shader.cc 등)에서 처리한다.
 * 명령어별 레이턴스는 gpgpusim.config의 -ptx_opcode_latency_* / -ptx_opcode_initiation_*
 * 옵션군으로 설정된다.
 * ============================================================================
 */

/*
 * [한국어]
 * decode_space - 메모리 공간 타입을 실제 memory_space 객체와 HW 주소로 해석
 *
 * @space: 입출력 메모리 공간 타입 — 미분류(param_space_unclassified) 또는 제네릭(generic_space)인 경우
 *         이 함수가 실제 공간으로 해석하여 갱신한다
 * @thread: 현재 실행 중인 PTX 스레드 (메모리 객체 접근 및 SM/HW 스레드 ID 제공)
 * @op: 주소를 제공한 오퍼랜드 정보 — param_space_unclassified 해석 시 심볼 타입 확인에 사용
 * @mem: 출력 — 해당 공간에 대응하는 memory_space 포인터 (예: 공유메모리, 글로벌메모리 등)
 * @addr: 입출력 HW 주소 — 로컬/제네릭 공간의 경우 실제 HW 주소로 변환되어 갱신됨
 *
 * ld/st/cvta/isspacep 등 메모리 접근 명령어에서 공통으로 사용하는 주소 디코딩 헬퍼.
 * 주요 처리:
 *   1. param_space_unclassified: 오퍼랜드 심볼의 타입 정보를 보고 kernel/local param 판별
 *   2. 각 공간 타입 → 해당 memory_space 포인터 설정:
 *      - global → thread->get_global_memory()
 *      - local/param_local → thread->m_local_mem + 스택 포인터 추가
 *      - shared → thread->m_shared_mem
 *      - tex/surf/param_kernel/sstarr/const → 각 전용 memory_space
 *   3. generic_space(PTX 2.0 이상): whichspace()로 실제 공간 판별 후 직접 주소로 변환
 * local 공간은 addr에 스택 포인터를 더하여 스레드별 로컬메모리 영역의 올바른 주소를 계산한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ld_exec() / mma_st_impl() / mma_ld_impl() → [decode_space] → whichspace() / generic_to_*()
 */
void decode_space(memory_space_t &space, ptx_thread_info *thread,
                  const operand_info &op, memory_space *&mem, addr_t &addr) {
  unsigned smid = thread->get_hw_sid(); // [한국어] SM ID — shared/local 공간 주소 변환에 필요
  unsigned hwtid = thread->get_hw_tid(); // [한국어] HW 스레드 ID — local 공간 주소 변환에 필요

  if (space == param_space_unclassified) {
    // need to op to determine whether it refers to a kernel param or local
    // param
    // [한국어] param_space_unclassified: 오퍼랜드 심볼의 타입을 보고 커널 파라미터인지 로컬 파라미터인지 판별
    const symbol *s = op.get_symbol(); // [한국어] 오퍼랜드의 심볼 객체 획득
    const type_info *t = s->type(); // [한국어] 심볼의 타입 정보 획득
    type_info_key ti = t->get_key(); // [한국어] 타입 정보에서 키(분류 정보) 추출
    if (ti.is_param_kernel())
      space = param_space_kernel; // [한국어] 커널 파라미터 공간으로 분류 (cuLaunchKernel 인자)
    else if (ti.is_param_local()) {
      space = param_space_local; // [한국어] 함수 내 로컬 파라미터 공간으로 분류
    }
    // mov r1, param-label
    else if (ti.is_reg()) {
      space = param_space_kernel; // [한국어] 레지스터 타입이지만 param 컨텍스트 — 커널 파라미터로 처리
    } else {
      printf("GPGPU-Sim PTX: ERROR ** cannot resolve .param space for '%s'\n",
             s->name().c_str()); // [한국어] 알 수 없는 param 공간 분류 — 오류 메시지 출력
      abort(); // [한국어] 해석 불가한 공간 타입으로 시뮬레이션 강제 종료
    }
  }
  switch (space.get_type()) { // [한국어] 확정된 메모리 공간 타입에 따라 memory_space 포인터와 주소 설정
    case global_space:
      mem = thread->get_global_memory(); // [한국어] 글로벌 메모리 공간 포인터 설정 (전체 GPU가 공유)
      break;
    case param_space_local:
    case local_space:
      mem = thread->m_local_mem; // [한국어] 스레드별 로컬 메모리 공간 포인터 설정
      addr += thread->get_local_mem_stack_pointer(); // [한국어] 스택 포인터를 더해 이 스레드의 로컬 메모리 실제 HW 주소 계산
      break;
    case tex_space:
      mem = thread->get_tex_memory(); // [한국어] 텍스처 메모리 공간 포인터 설정
      break;
    case surf_space:
      mem = thread->get_surf_memory(); // [한국어] 서피스 메모리 공간 포인터 설정
      break;
    case param_space_kernel:
      mem = thread->get_param_memory(); // [한국어] 커널 파라미터 메모리 공간 포인터 설정
      break;
    case shared_space:
      mem = thread->m_shared_mem; // [한국어] SM 내 공유메모리 공간 포인터 설정 (워프 내 스레드가 공유)
      break;
    case sstarr_space:
      mem = thread->m_sstarr_mem; // [한국어] SSTARR(Static Shared To Array) 메모리 공간 포인터 설정
      break;
    case const_space:
      mem = thread->get_global_memory(); // [한국어] 상수 공간은 글로벌메모리로 매핑 (읽기 전용이나 물리적으로 동일 위치)
      break;
    case generic_space:
      if (thread->get_ptx_version().ver() >= 2.0) { // [한국어] PTX 2.0 이상에서만 제네릭 주소 지원
        // convert generic address to memory space address
        space = whichspace(addr); // [한국어] 주소 범위로 실제 메모리 공간을 판별 (whichspace는 주소 범위를 조회)
        switch (space.get_type()) { // [한국어] 판별된 실제 공간에 따라 포인터와 직접 주소 설정
          case global_space:
            mem = thread->get_global_memory(); // [한국어] 글로벌 메모리 포인터 설정
            addr = generic_to_global(addr); // [한국어] 제네릭→글로벌 직접 주소 변환
            break;
          case local_space:
            mem = thread->m_local_mem; // [한국어] 로컬 메모리 포인터 설정
            addr = generic_to_local(smid, hwtid, addr); // [한국어] 제네릭→로컬 직접 주소 변환 (SM+스레드 기반)
            break;
          case shared_space:
            mem = thread->m_shared_mem; // [한국어] 공유 메모리 포인터 설정
            addr = generic_to_shared(smid, addr); // [한국어] 제네릭→공유 직접 주소 변환 (SM 기반)
            break;
          default:
            abort(); // [한국어] 제네릭 주소가 알 수 없는 공간을 가리킴 — 구현 오류
        }
      } else {
        abort(); // [한국어] PTX 2.0 미만에서는 제네릭 주소 미지원 — 시뮬레이션 강제 종료
      }
      break;
    case param_space_unclassified:
    case undefined_space:
    default:
      abort(); // [한국어] 분류 불가한 공간 타입 — 구현 오류로 시뮬레이션 강제 종료
  }
}

/*
 * [한국어]
 * ld_exec - PTX `ld` / `ldu` 명령어 핵심 로직: 메모리에서 값을 읽어 레지스터에 저장
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, 소스(주소), 타입, 메모리 공간, 벡터 수식어 포함)
 * @thread: 현재 실행 중인 PTX 스레드 (메모리 접근, 레지스터 파일, m_last_effective_address 설정)
 *
 * PTX `ld.{space}.{type} d, [a]` 명령어의 실질적 구현.
 * 동작 흐름:
 *   1. 소스 오퍼랜드에서 메모리 주소를 읽음 (src1_data.u32)
 *   2. decode_space()로 메모리 공간과 실제 HW 주소를 해석
 *   3. 타입으로 접근 크기(size/8 바이트)를 결정
 *   4. 벡터 수식어 없으면 단일 mem->read(), 있으면 V2/V3/V4 요소 수만큼 반복 read()
 *   5. S16/S32 타입은 부호 확장(sign_extend) 후 목적지에 저장
 *   6. m_last_effective_address와 m_last_memory_space를 기록 — 타이밍 모델이 이 값을 읽어
 *      실제 캐시(L1/L2)/DRAM 접근을 시뮬레이션하는 데 사용함
 * 핵심 설계: 기능 시뮬레이션에서 실제 데이터를 읽은 뒤, 타이밍 모델이
 * m_last_effective_address를 참조하여 캐시 히트/미스/DRAM 지연을 별도로 계산한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ld_impl() / ldu_impl() → [ld_exec] → decode_space() → mem->read() → thread->set_operand_value()
 */
void ld_exec(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst(); // [한국어] 로드된 값이 저장될 목적지 레지스터
  const operand_info &src1 = pI->src1(); // [한국어] 읽을 메모리 주소를 담은 소스 오퍼랜드

  unsigned type = pI->get_type(); // [한국어] PTX 데이터 타입 (읽을 데이터의 타입 및 크기 결정)

  ptx_reg_t src1_data = thread->get_operand_value(src1, dst, type, thread, 1);
  // [한국어] 소스 오퍼랜드(주소 레지스터)에서 메모리 주소 값을 읽음
  ptx_reg_t data; // [한국어] 메모리에서 읽어온 데이터를 담을 레지스터
  memory_space_t space = pI->get_space(); // [한국어] 접근할 메모리 공간 타입 (global/shared/local 등)
  unsigned vector_spec = pI->get_vector(); // [한국어] 벡터 수식어 (0=스칼라, V2_TYPE/V3_TYPE/V4_TYPE)

  memory_space *mem = NULL; // [한국어] 실제 접근할 memory_space 객체 포인터 (decode_space가 설정)
  addr_t addr = src1_data.u32; // [한국어] 주소 값을 HW addr_t 타입으로 추출 (u32 사용 — 32비트 주소 가정)

  decode_space(space, thread, src1, mem, addr);
  // [한국어] space/mem/addr를 실제 HW 메모리 공간과 직접 주소로 해석 및 갱신

  size_t size; // [한국어] 타입 크기 (비트 단위, 바이트 변환 시 /8)
  int t; // [한국어] 타입 부호 정보 (sign flag, type_decode가 설정)
  data.u64 = 0; // [한국어] 읽기 전 결과 레지스터를 0으로 초기화 (좁은 타입 읽기 시 상위 바이트 보장)
  type_info_key::type_decode(type, size, t); // [한국어] PTX 타입에서 비트 크기와 부호 정보 추출
  if (!vector_spec) {
    // [한국어] 스칼라 로드: 단일 요소를 한 번의 mem->read()로 읽음
    mem->read(addr, size / 8, &data.s64); // [한국어] 메모리에서 size/8 바이트를 읽어 data.s64에 저장 (s64를 raw buffer로 사용)
    if (type == S16_TYPE || type == S32_TYPE) sign_extend(data, size, dst);
    // [한국어] S16/S32 타입은 메모리에서 읽은 후 부호 확장이 필요 — 목적지 레지스터 크기에 맞게 확장
    thread->set_operand_value(dst, data, type, thread, pI); // [한국어] 읽어온 값을 목적지 레지스터에 기록
  } else {
    // [한국어] 벡터 로드: 연속 주소에서 여러 요소를 읽어 벡터 레지스터에 저장
    ptx_reg_t data1, data2, data3, data4; // [한국어] 벡터 각 요소를 담을 레지스터 (최대 4개)
    mem->read(addr, size / 8, &data1.s64); // [한국어] 첫 번째 요소 읽기 (addr + 0)
    mem->read(addr + size / 8, size / 8, &data2.s64); // [한국어] 두 번째 요소 읽기 (addr + 1*elementSize)
    if (vector_spec != V2_TYPE) {  // either V3 or V4
      // [한국어] V3 또는 V4: 세 번째 요소도 읽음
      mem->read(addr + 2 * size / 8, size / 8, &data3.s64); // [한국어] 세 번째 요소 읽기 (addr + 2*elementSize)
      if (vector_spec != V3_TYPE) {  // v4
        // [한국어] V4: 네 번째 요소까지 읽음
        mem->read(addr + 3 * size / 8, size / 8, &data4.s64); // [한국어] 네 번째 요소 읽기 (addr + 3*elementSize)
        thread->set_vector_operand_values(dst, data1, data2, data3, data4); // [한국어] 4개 요소를 벡터 레지스터에 기록
      } else  // v3
        thread->set_vector_operand_values(dst, data1, data2, data3, data3); // [한국어] V3: 네 번째 자리에 data3를 중복 사용 (4요소 레지스터 구조 유지)
    } else  // v2
      thread->set_vector_operand_values(dst, data1, data2, data2, data2); // [한국어] V2: 세/네 번째 자리에 data2를 중복 사용
  }
  thread->m_last_effective_address = addr; // [한국어] 실제 접근한 HW 주소를 기록 — 타이밍 모델이 이 주소로 캐시/DRAM 시뮬레이션
  thread->m_last_memory_space = space; // [한국어] 접근한 메모리 공간을 기록 — 타이밍 모델이 올바른 캐시 레벨 선택에 사용
}

/*
 * [한국어]
 * ld_impl - PTX `ld` 명령어 구현: 메모리 로드 (ld_exec에 위임)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `ld.{space}.{type} d, [a]` 명령어의 진입점.
 * 실제 로직은 ld_exec()에 위임한다.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [ld_impl] → ld_exec()
 */
void ld_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ld_exec(pI, thread); // [한국어] 실제 메모리 로드 로직은 ld_exec()에 위임
}

/*
 * [한국어]
 * ldu_impl - PTX `ldu` 명령어 구현: 유니폼 메모리 로드 (ld_exec에 위임)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `ldu.{space}.{type} d, [a]` 명령어의 진입점.
 * `ldu`는 워프 내 모든 스레드가 동일한 주소를 읽는 유니폼 로드임을 나타내지만,
 * 기능 시뮬레이션에서는 ld와 동일하게 ld_exec()에 위임한다.
 * 타이밍 모델에서는 유니폼 로드 특성을 활용하여 최적화할 수 있다.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [ldu_impl] → ld_exec()
 */
void ldu_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ld_exec(pI, thread); // [한국어] ldu는 기능적으로 ld와 동일 — ld_exec()에 위임
}

/*
 * [한국어]
 * mma_st_impl - PTX `wmma.store.d` 명령어 구현: Tensor Core D 행렬 결과를 메모리에 저장
 *
 * @pI: 실행 중인 PTX 명령어 (목적지 주소, D 행렬 소스, stride, wmma_type/layout 포함)
 * @core: SM(Shader Core) 객체 — 워프 내 스레드 정보 및 GPU 컨텍스트 접근
 * @inst: warp_inst_t 참조 — 각 스레드의 메모리 트랜잭션 주소를 기록하여 타이밍 모델에 전달
 *
 * PTX `wmma.store.d.sync.aligned.{layout}.m16n16k16.{type}` 명령어를 구현한다.
 * 워프(32 스레드)가 집합적으로 16x16 D 행렬을 메모리에 기록한다.
 * 각 스레드는 행렬의 일부(최대 8개 요소)를 담당하며, thread_group_offset()으로
 * 이 스레드가 담당하는 행렬 내 오프셋을 계산한다.
 * 타입별 처리:
 *   - F32: 각 스레드가 8개 F32 요소를 acc_float_offset()이 정의한 위치에 저장
 *   - F16: 레이아웃(ROW/COL)에 따라 연속 또는 stride 간격으로 F16 쌍을 저장
 * 메모리 트랜잭션 주소를 inst.set_addr()에 기록하여 타이밍 모델이 캐시/DRAM 시뮬을 수행하게 함.
 * 실행 컨텍스트: 기능 시뮬레이션 — 워프 내 32 스레드 순서대로 처리.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [mma_st_impl] → decode_space() → mem->write()
 */
void mma_st_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst) {
  size_t size; // [한국어] PTX 타입의 요소 크기 (비트 단위)
  unsigned smid; // [한국어] SM ID — 공유메모리 주소 변환에 사용
  int t; // [한국어] 타입 부호 정보 (type_decode가 설정)
  int thrd, k; // [한국어] 스레드 루프 인덱스(thrd)와 요소 루프 인덱스(k)
  ptx_thread_info *thread; // [한국어] 현재 처리 중인 스레드 포인터

  const operand_info &src = pI->operand_lookup(1); // [한국어] D 행렬 소스 오퍼랜드 (벡터 레지스터)
  const operand_info &src1 = pI->operand_lookup(0); // [한국어] 저장 목적지 메모리 주소 오퍼랜드
  const operand_info &src2 = pI->operand_lookup(2); // [한국어] stride 오퍼랜드 (행 간격, 요소 수 단위)
  int tid; // [한국어] 워프의 첫 번째 스레드 전역 ID
  unsigned type = pI->get_type(); // [한국어] PTX 데이터 타입 (F32_TYPE 또는 F16_TYPE)
  unsigned wmma_type = pI->get_wmma_type(); // [한국어] WMMA 오퍼랜드 타입 (STORE_D 등)
  unsigned wmma_layout = pI->get_wmma_layout(0); // [한국어] 행렬 레이아웃 (ROW 또는 COL)
  int stride; // [한국어] 행렬의 행 간격 (요소 수 단위, src2에서 읽음)

  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size(); // [한국어] 기능 시뮬레이션 모드: 기능 시뮬 워프 ID 사용
  else
    tid = inst.warp_id() * core->get_warp_size(); // [한국어] 타이밍 시뮬레이션 모드: 타이밍 워프 ID 사용

  _memory_op_t insn_memory_op =
      pI->has_memory_read() ? memory_load : memory_store;
  // [한국어] 명령어 메모리 연산 방향 결정 (wmma.store는 memory_store) — inst의 memory_op 검증에 사용
  for (thrd = 0; thrd < core->get_warp_size(); thrd++) { // [한국어] 워프 내 32 스레드 각각에 대해 저장 수행
    thread = core->get_thread_info()[tid + thrd]; // [한국어] 현재 처리할 스레드 객체 포인터 획득
    ptx_reg_t addr_reg = thread->get_operand_value(src1, src, type, thread, 1);
    // [한국어] 이 스레드의 저장 목적지 기준 주소를 레지스터에서 읽음
    ptx_reg_t src2_data = thread->get_operand_value(src2, src, type, thread, 1);
    // [한국어] stride 값을 레지스터에서 읽음 (행 간격)
    const operand_info &src_a = pI->operand_lookup(1); // [한국어] D 행렬 값을 담은 벡터 오퍼랜드 (src와 동일)
    unsigned nelem = src_a.get_vect_nelem(); // [한국어] 벡터 오퍼랜드의 요소 수 (F32이면 8, F16이면 4)
    ptx_reg_t *v = new ptx_reg_t[8]; // [한국어] D 행렬 요소 값을 담을 임시 배열 (최대 8개)
    thread->get_vector_operand_values(src_a, v, nelem); // [한국어] D 행렬 레지스터 값을 v 배열에 읽어옴
    stride = src2_data.u32; // [한국어] stride: 행렬의 행 간격 (열 수, 예: 16이면 16요소 간격)

    memory_space_t space = pI->get_space(); // [한국어] 저장할 메모리 공간 타입 (global 또는 shared)

    memory_space *mem = NULL; // [한국어] 실제 접근할 memory_space 포인터 (decode_space가 설정)
    addr_t addr = addr_reg.u32; // [한국어] 기준 메모리 주소 (u32로 추출)

    new_addr_type mem_txn_addr[MAX_ACCESSES_PER_INSN_PER_THREAD]; // [한국어] 이 스레드의 메모리 트랜잭션 주소 배열 — 타이밍 모델에 전달
    int num_mem_txn = 0; // [한국어] 이 스레드가 수행한 메모리 트랜잭션 수

    smid = thread->get_hw_sid(); // [한국어] 이 스레드가 실행 중인 SM의 ID
    if (whichspace(addr) == shared_space) { // [한국어] 주소가 공유메모리 범위인지 확인
      addr = generic_to_shared(smid, addr); // [한국어] 제네릭 주소를 공유메모리 직접 주소로 변환
      space = shared_space; // [한국어] 메모리 공간을 공유메모리로 명시적으로 설정
    }
    decode_space(space, thread, src1, mem, addr);
    // [한국어] 메모리 공간과 실제 HW 주소를 해석하여 mem 포인터와 addr 갱신

    type_info_key::type_decode(type, size, t); // [한국어] PTX 타입에서 요소 크기(size, 비트 단위)와 부호 정보(t) 추출
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
      printf("mma_st: thrd=%d, addr=%x, fp(size=%zu), stride=%d\n", thrd,
             addr_reg.u32, size, src2_data.u32); // [한국어] Tensor Core 디버그 모드: 각 스레드의 저장 정보 출력
    addr_t new_addr =
        addr + thread_group_offset(thrd, wmma_type, wmma_layout, type, stride) *
                   size / 8;
    // [한국어] 이 스레드가 담당하는 행렬 내 오프셋(thread_group_offset) * 요소 바이트 크기를 더해
    // [한국어] 이 스레드가 저장해야 할 행렬 시작 주소 계산
    addr_t push_addr; // [한국어] 각 요소를 저장할 실제 주소

    ptx_reg_t nw_v[8]; // [한국어] F16 요소를 재포맷한 배열 (4개 packed F16 쌍 → 8개 분리된 F16)
    for (k = 0; k < 8; k++) { // [한국어] v[] 배열의 F16 쌍을 상위/하위 16비트로 분리
      if (k % 2 == 0)
        nw_v[k].s64 = (v[k / 2].s64 & 0xffff); // [한국어] 짝수 인덱스: 레지스터의 하위 16비트 (첫 번째 F16)
      else
        nw_v[k].s64 = ((v[k / 2].s64 & 0xffff0000) >> 16); // [한국어] 홀수 인덱스: 레지스터의 상위 16비트를 하위로 이동 (두 번째 F16)
    }

    for (k = 0; k < 8; k++) { // [한국어] 8개 요소 각각을 계산된 주소에 저장
      if (type == F32_TYPE) {
        // mem->write(new_addr+4*acc_float_offset(k,wmma_layout,stride),size/8,&v[k].s64,thread,pI);
        push_addr = new_addr + 4 * acc_float_offset(k, wmma_layout, stride);
        // [한국어] F32 요소 k의 저장 주소: acc_float_offset이 레이아웃과 stride를 고려하여 오프셋 계산, ×4(바이트)
        mem->write(push_addr, size / 8, &v[k].s64, thread, pI); // [한국어] F32 요소를 메모리에 기록 (size/8 = 4바이트)
        mem_txn_addr[num_mem_txn++] = push_addr; // [한국어] 트랜잭션 주소 기록 — 타이밍 모델에서 캐시 시뮬에 사용

        if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) {
          printf(
              "wmma:store:thread%d=%llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx\n",
              thrd, v[0].s64, v[1].s64, v[2].s64, v[3].s64, v[4].s64, v[5].s64,
              v[6].s64, v[7].s64); // [한국어] 디버그: 16진수로 저장 값 출력
          float temp;
          int l;
          printf("thread=%d:", thrd);
          for (l = 0; l < 8; l++) {
            temp = v[l].f32;
            printf("%.2f", temp); // [한국어] 디버그: 부동소수점으로 저장 값 출력
          }
          printf("\n");
        }
      } else if (type == F16_TYPE) {
        if (wmma_layout == ROW) {
          // mem->write(new_addr+k*2,size/8,&nw_v[k].s64,thread,pI);
          push_addr = new_addr + k * 2; // [한국어] ROW 레이아웃: 각 F16 요소가 2바이트 간격으로 연속 배치
          mem->write(push_addr, size / 8, &nw_v[k].s64, thread, pI); // [한국어] F16 요소(2바이트) 저장
          if (k % 2 == 0) mem_txn_addr[num_mem_txn++] = push_addr; // [한국어] 짝수 인덱스만 트랜잭션으로 기록 (F16 쌍 단위)
        } else if (wmma_layout == COL) {
          // mem->write(new_addr+k*2*stride,size/8,&nw_v[k].s64,thread,pI);
          push_addr = new_addr + k * 2 * stride; // [한국어] COL 레이아웃: 각 F16 요소가 2*stride 바이트 간격으로 배치 (열 방향)
          mem->write(push_addr, size / 8, &nw_v[k].s64, thread, pI); // [한국어] F16 요소(2바이트) 저장
          mem_txn_addr[num_mem_txn++] = push_addr; // [한국어] 모든 COL 주소를 트랜잭션으로 기록 (산포(scattered) 접근)
        }

        if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
          printf(
              "wmma:store:thread%d=%llx,%llx,%llx,%llx,%llx,%llx,%llx,%llx\n",
              thrd, nw_v[0].s64, nw_v[1].s64, nw_v[2].s64, nw_v[3].s64,
              nw_v[4].s64, nw_v[5].s64, nw_v[6].s64, nw_v[7].s64); // [한국어] 디버그: F16 분리된 요소 값 출력
      }
    }

    delete[] v; // [한국어] new[]로 할당한 D 행렬 요소 배열 해제
    inst.space = space; // [한국어] warp_inst_t에 접근 공간 설정 — 타이밍 모델이 캐시 레벨 결정에 사용
    inst.set_addr(thrd, (new_addr_type *)mem_txn_addr, num_mem_txn);
    // [한국어] 이 스레드의 메모리 트랜잭션 주소 배열을 warp_inst_t에 기록 — 타이밍 모델에 전달

    if ((type == F16_TYPE) &&
        (wmma_layout == COL))  // check the profiling xls for details
      inst.data_size = 2;      // 2 byte transaction
    // [한국어] F16 COL 레이아웃: 산포(scattered) 접근으로 2바이트 단위 트랜잭션
    else
      inst.data_size = 4;  // 4 byte transaction
    // [한국어] F32 및 F16 ROW: 4바이트 단위 트랜잭션

    assert(inst.memory_op == insn_memory_op); // [한국어] 명령어가 예상하는 메모리 연산 방향(store)과 일치하는지 검증
    // thread->m_last_effective_address = addr;
    // thread->m_last_memory_space = space;
    // [한국어] wmma 명령어는 m_last_effective_address를 갱신하지 않음 — 타이밍 모델에서 inst.addr로 직접 처리
  }
}

/*
 * [한국어]
 * mma_ld_impl - PTX `wmma.load` 명령어 구현: Tensor Core A/B/C 행렬을 메모리에서 레지스터로 로드
 *
 * @pI: 실행 중인 PTX 명령어 (목적지 레지스터, 소스 주소, stride, wmma_type/layout 포함)
 * @core: SM(Shader Core) 객체 — 워프 내 스레드 정보 및 GPU 컨텍스트 접근
 * @inst: warp_inst_t 참조 — 각 스레드의 메모리 트랜잭션 주소를 기록하여 타이밍 모델에 전달
 *
 * PTX `wmma.load.{a|b|c}.sync.aligned.{layout}.m16n16k16.{type}` 명령어를 구현한다.
 * 워프(32 스레드)가 집합적으로 16x16 행렬(A, B, 또는 C)을 메모리에서 레지스터로 읽는다.
 * wmma_type에 따라 읽는 요소 수와 배치가 다르다:
 *   - LOAD_A (행렬 A): 스레드당 16개 F16 요소, ROW이면 연속, COL이면 stride 간격
 *   - LOAD_B (행렬 B): 스레드당 16개 F16 요소, COL이면 연속, ROW이면 stride 간격 (A와 반대)
 *   - LOAD_C (누산기 C): 스레드당 8개 요소, F16이면 ROW/COL 레이아웃, F32이면 acc_float_offset
 * 읽은 데이터는 F16의 경우 2개씩 합쳐 하나의 레지스터에 packed 형태로 저장.
 * 메모리 트랜잭션 주소를 inst.set_addr()에 기록하여 타이밍 모델에 전달.
 * 실행 컨텍스트: 기능 시뮬레이션 — 워프 내 32 스레드 순서대로 처리.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [mma_ld_impl] → decode_space() → mem->read()
 *     → thread->set_wmma_vector_operand_values()
 */
void mma_ld_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst) {
  size_t size; // [한국어] PTX 타입의 요소 크기 (비트 단위)
  int t, i; // [한국어] 타입 부호 정보(t)와 요소 루프 인덱스(i)
  unsigned smid; // [한국어] SM ID — 공유메모리 주소 변환에 사용
  const operand_info &dst = pI->dst(); // [한국어] 행렬 데이터를 저장할 목적지 벡터 레지스터
  const operand_info &src1 = pI->src1(); // [한국어] 읽을 메모리의 기준 주소 오퍼랜드
  const operand_info &src2 = pI->src2(); // [한국어] stride 오퍼랜드 (행 간격, 요소 수 단위)

  unsigned type = pI->get_type(); // [한국어] PTX 데이터 타입 (F16_TYPE 또는 F32_TYPE)
  unsigned wmma_type = pI->get_wmma_type(); // [한국어] WMMA 오퍼랜드 타입 (LOAD_A/LOAD_B/LOAD_C)
  unsigned wmma_layout = pI->get_wmma_layout(0); // [한국어] 행렬 레이아웃 (ROW 또는 COL)
  int tid; // [한국어] 워프의 첫 번째 스레드 전역 ID
  int thrd, stride; // [한국어] 스레드 루프 인덱스(thrd)와 행 간격(stride)
  ptx_thread_info *thread; // [한국어] 현재 처리 중인 스레드 포인터

  if (core->get_gpu()->is_functional_sim())
    tid = inst.warp_id_func() * core->get_warp_size(); // [한국어] 기능 시뮬레이션 모드: 기능 시뮬 워프 ID 사용
  else
    tid = inst.warp_id() * core->get_warp_size(); // [한국어] 타이밍 시뮬레이션 모드: 타이밍 워프 ID 사용

  _memory_op_t insn_memory_op =
      pI->has_memory_read() ? memory_load : memory_store;
  // [한국어] 명령어 메모리 연산 방향 결정 (wmma.load는 memory_load) — inst 검증에 사용

  for (thrd = 0; thrd < core->get_warp_size(); thrd++) { // [한국어] 워프 내 32 스레드 각각에 대해 로드 수행
    thread = core->get_thread_info()[tid + thrd]; // [한국어] 현재 처리할 스레드 객체 포인터 획득
    ptx_reg_t src1_data =
        thread->get_operand_value(src1, dst, U32_TYPE, thread, 1);
    // [한국어] 기준 메모리 주소를 U32 타입으로 읽음 (주소는 항상 32비트)
    ptx_reg_t src2_data =
        thread->get_operand_value(src2, dst, U32_TYPE, thread, 1);
    // [한국어] stride 값을 U32 타입으로 읽음
    stride = src2_data.u32; // [한국어] stride: 행렬의 행 간격 (요소 수)
    memory_space_t space = pI->get_space(); // [한국어] 읽을 메모리 공간 타입

    memory_space *mem = NULL; // [한국어] 실제 접근할 memory_space 포인터 (decode_space가 설정)
    addr_t addr = src1_data.u32; // [한국어] 기준 메모리 주소
    smid = thread->get_hw_sid(); // [한국어] 이 스레드가 실행 중인 SM의 ID
    if (whichspace(addr) == shared_space) { // [한국어] 주소가 공유메모리 범위인지 확인
      addr = generic_to_shared(smid, addr); // [한국어] 제네릭 주소를 공유메모리 직접 주소로 변환
      space = shared_space; // [한국어] 메모리 공간을 공유메모리로 명시
    }

    decode_space(space, thread, src1, mem, addr);
    // [한국어] 메모리 공간과 실제 HW 주소를 해석하여 mem 포인터와 addr 갱신
    type_info_key::type_decode(type, size, t); // [한국어] PTX 타입에서 요소 크기(size, 비트 단위)와 부호 정보(t) 추출

    ptx_reg_t data[16]; // [한국어] 메모리에서 읽어온 원시(raw) 데이터 배열 (A/B는 최대 16개, C는 최대 8개)
    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore)
      printf("mma_ld: thrd=%d,addr=%x, fpsize=%zu, stride=%d\n", thrd,
             src1_data.u32, size, src2_data.u32); // [한국어] Tensor Core 디버그 모드: 로드 정보 출력

    addr_t new_addr =
        addr + thread_group_offset(thrd, wmma_type, wmma_layout, type, stride) *
                   size / 8;
    // [한국어] 이 스레드가 담당하는 행렬 내 오프셋(thread_group_offset) * 요소 바이트 크기를 더해
    // [한국어] 이 스레드가 읽어야 할 행렬 시작 주소 계산
    addr_t fetch_addr; // [한국어] 각 요소를 읽을 실제 주소
    new_addr_type mem_txn_addr[MAX_ACCESSES_PER_INSN_PER_THREAD]; // [한국어] 이 스레드의 메모리 트랜잭션 주소 배열
    int num_mem_txn = 0; // [한국어] 이 스레드가 수행한 메모리 트랜잭션 수

    if (wmma_type == LOAD_A) {
      // [한국어] LOAD_A: 행렬 A 로드 — 스레드당 16개 F16 요소 읽기
      for (i = 0; i < 16; i++) { // [한국어] 16개 F16 요소 각각을 읽음
        if (wmma_layout == ROW) {
          // mem->read(new_addr+2*i,size/8,&data[i].s64);
          fetch_addr = new_addr + 2 * i; // [한국어] ROW 레이아웃: 요소 i는 기준 주소에서 2*i 바이트 오프셋 (연속 배치)
          mem->read(fetch_addr, size / 8, &data[i].s64); // [한국어] F16 요소(2바이트) 읽기
        } else if (wmma_layout == COL) {
          // mem->read(new_addr+2*(i%4)+2*stride*4*(i/4),size/8,&data[i].s64);
          fetch_addr = new_addr + 2 * (i % 4) + 2 * stride * 4 * (i / 4);
          // [한국어] COL 레이아웃: 행(i/4)마다 stride*4*2 바이트 간격, 열(i%4)마다 2 바이트 간격
          mem->read(fetch_addr, size / 8, &data[i].s64); // [한국어] F16 요소(2바이트) 읽기
        } else {
          printf("mma_ld:wrong_layout_type\n"); // [한국어] 잘못된 레이아웃 타입 오류
          abort();
        }
        if (i % 2 == 0) mem_txn_addr[num_mem_txn++] = fetch_addr; // [한국어] 짝수 인덱스만 트랜잭션으로 기록 (F16 쌍 단위)
      }
    } else if (wmma_type == LOAD_B) {
      // [한국어] LOAD_B: 행렬 B 로드 — 스레드당 16개 F16 요소 읽기 (A와 레이아웃 방향이 반대)
      for (i = 0; i < 16; i++) { // [한국어] 16개 F16 요소 각각을 읽음
        if (wmma_layout == COL) {
          // mem->read(new_addr+2*i,size/8,&data[i].s64);
          fetch_addr = new_addr + 2 * i; // [한국어] COL 레이아웃 B: 연속 배치 (A의 ROW 레이아웃과 동일 패턴)
          mem->read(fetch_addr, size / 8, &data[i].s64); // [한국어] F16 요소(2바이트) 읽기
        } else if (wmma_layout == ROW) {
          // mem->read(new_addr+2*(i%4)+2*stride*4*(i/4),size/8,&data[i].s64);
          fetch_addr = new_addr + 2 * (i % 4) + 2 * stride * 4 * (i / 4);
          // [한국어] ROW 레이아웃 B: stride 간격 배치 (A의 COL 레이아웃과 동일 패턴)
          mem->read(fetch_addr, size / 8, &data[i].s64); // [한국어] F16 요소(2바이트) 읽기
        } else {
          printf("mma_ld:wrong_layout_type\n"); // [한국어] 잘못된 레이아웃 타입 오류
          abort();
        }
        if (i % 2 == 0) mem_txn_addr[num_mem_txn++] = fetch_addr; // [한국어] 짝수 인덱스만 트랜잭션으로 기록
      }
    } else if (wmma_type == LOAD_C) {
      // [한국어] LOAD_C: 누산기 C 로드 — 스레드당 8개 요소 읽기 (F16 또는 F32)
      for (i = 0; i < 8; i++) { // [한국어] 8개 요소 각각을 읽음
        if (type == F16_TYPE) {
          // [한국어] C 행렬이 F16 타입인 경우
          if (wmma_layout == ROW) {
            // mem->read(new_addr+2*i,size/8,&data[i].s64);
            fetch_addr = new_addr + 2 * i; // [한국어] ROW 레이아웃: F16 요소 i는 2*i 바이트 오프셋
            mem->read(fetch_addr, size / 8, &data[i].s64); // [한국어] F16 요소(2바이트) 읽기
            if (i % 2 == 0) mem_txn_addr[num_mem_txn++] = fetch_addr; // [한국어] 짝수 인덱스만 트랜잭션 기록
          } else if (wmma_layout == COL) {
            // mem->read(new_addr+2*stride*i,size/8,&data[i].s64);
            fetch_addr = new_addr + 2 * stride * i; // [한국어] COL 레이아웃: 요소 i는 2*stride*i 바이트 오프셋 (산포)
            mem->read(fetch_addr, size / 8, &data[i].s64); // [한국어] F16 요소(2바이트) 읽기
            mem_txn_addr[num_mem_txn++] = fetch_addr; // [한국어] 모든 COL 주소를 트랜잭션으로 기록
          } else {
            printf("mma_ld:wrong_type\n"); // [한국어] 잘못된 레이아웃 타입 오류
            abort();
          }
        } else if (type == F32_TYPE) {
          // [한국어] C 행렬이 F32 타입인 경우: acc_float_offset으로 주소 계산
          // mem->read(new_addr+4*acc_float_offset(i,wmma_layout,stride),size/8,&data[i].s64);
          fetch_addr = new_addr + 4 * acc_float_offset(i, wmma_layout, stride);
          // [한국어] F32 C 요소 i의 주소: acc_float_offset이 레이아웃과 stride를 고려한 오프셋 반환, ×4(바이트)
          mem->read(fetch_addr, size / 8, &data[i].s64); // [한국어] F32 요소(4바이트) 읽기
          mem_txn_addr[num_mem_txn++] = fetch_addr; // [한국어] 모든 F32 C 주소를 트랜잭션으로 기록
        } else {
          printf("wrong type"); // [한국어] 지원하지 않는 타입 오류
          abort();
        }
      }
    } else {
      printf("wrong wmma type\n"); // [한국어] 알 수 없는 WMMA 오퍼랜드 타입 오류
      ;
      abort();
    }
    // generate timing memory request
    inst.space = space; // [한국어] warp_inst_t에 접근 공간 설정 — 타이밍 모델 캐시 레벨 결정에 사용
    inst.set_addr(thrd, (new_addr_type *)mem_txn_addr, num_mem_txn);
    // [한국어] 이 스레드의 메모리 트랜잭션 주소 배열을 warp_inst_t에 기록 — 타이밍 모델에 전달

    if ((wmma_type == LOAD_C) && (type == F16_TYPE) &&
        (wmma_layout == COL))  // memory address is scattered, check the
                               // profiling xls for more detail.
      inst.data_size = 2;      // 2 byte transaction
    // [한국어] F16 C COL 레이아웃: 산포(scattered) 접근으로 2바이트 단위 트랜잭션
    else
      inst.data_size = 4;  // 4 byte transaction
    // [한국어] 그 외(F32, F16 ROW 등): 4바이트 단위 트랜잭션
    assert(inst.memory_op == insn_memory_op); // [한국어] 명령어가 예상하는 메모리 연산 방향(load)과 일치하는지 검증

    if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) { // [한국어] Tensor Core 디버그 모드: 로드된 데이터 출력
      if (type == F16_TYPE) {
        printf("\nmma_ld:thread%d= ", thrd);
        for (i = 0; i < 16; i++) {
          printf("%llx ", data[i].u64); // [한국어] F16 타입: 16개 요소를 16진수로 출력
        }
        printf("\n");

        printf("\nmma_ld:thread%d= ", thrd);
        float temp;
        for (i = 0; i < 16; i++) {
          temp = data[i].f16;
          printf("%.2f ", temp); // [한국어] F16 타입: 16개 요소를 float로 변환하여 출력
        }
        printf("\n");
      } else {
        printf("\nmma_ld:thread%d= ", thrd);
        for (i = 0; i < 8; i++) {
          printf("%.2f ", data[i].f32); // [한국어] F32 타입: 8개 요소를 float으로 출력
        }
        printf("\n");
        printf("\nmma_ld:thread%d= ", thrd);
        for (i = 0; i < 8; i++) {
          printf("%llx ", data[i].u64); // [한국어] F32 타입: 8개 요소를 16진수로 출력
        }
        printf("\n");
      }
    }

    if ((wmma_type == LOAD_C) && (type == F32_TYPE)) {
      // [한국어] LOAD_C + F32: 8개 F32 요소를 직접 wmma 벡터 레지스터에 기록
      thread->set_wmma_vector_operand_values(dst, data[0], data[1], data[2],
                                             data[3], data[4], data[5], data[6],
                                             data[7]);
      // [한국어] F32 C 행렬 8개 요소를 목적지 벡터 레지스터(8요소)에 기록
    } else {
      // [한국어] F16 타입 (A, B, 또는 F16 C): 인접한 F16 쌍을 하나의 레지스터에 packed 형태로 결합
      ptx_reg_t nw_data[8]; // [한국어] packed F16 쌍을 담을 레지스터 배열
      int num_reg; // [한국어] 최종 레지스터 수 (LOAD_C이면 4개, A/B이면 8개)

      if (wmma_type == LOAD_C)
        num_reg = 4; // [한국어] F16 C: 8개 F16 → 4개 packed 레지스터 (2 F16씩 결합)
      else
        num_reg = 8; // [한국어] F16 A/B: 16개 F16 → 8개 packed 레지스터 (2 F16씩 결합)

      for (i = 0; i < num_reg; i++) {
        nw_data[i].s64 = ((data[2 * i].s64 & 0xffff) << 16) |
                         ((data[2 * i + 1].s64 & 0xffff));
        // [한국어] 두 F16 요소를 하나의 32비트 필드에 pack: 첫 번째를 상위 16비트, 두 번째를 하위 16비트에 배치
      }

      if (wmma_type == LOAD_C)
        thread->set_vector_operand_values(dst, nw_data[0], nw_data[1],
                                          nw_data[2], nw_data[3]);
        // [한국어] F16 C: 4개 packed 레지스터를 일반 벡터 레지스터에 기록
      else
        thread->set_wmma_vector_operand_values(
            dst, nw_data[0], nw_data[1], nw_data[2], nw_data[3], nw_data[4],
            nw_data[5], nw_data[6], nw_data[7]);
        // [한국어] F16 A/B: 8개 packed 레지스터를 wmma 전용 벡터 레지스터에 기록
      if (core->get_gpu()->gpgpu_ctx->debug_tensorcore) { // [한국어] 디버그 모드: packed 결과 출력
        printf(
            "mma_ld:data[0].s64=%llx,data[1].s64=%llx,new_data[0].s64=%llx\n",
            data[0].u64, data[1].u64, nw_data[0].u64);
        printf(
            "mma_ld:data[2].s64=%llx,data[3].s64=%llx,new_data[1].s64=%llx\n",
            data[2].u64, data[3].u64, nw_data[1].u64);
        printf(
            "mma_ld:data[4].s64=%llx,data[5].s64=%llx,new_data[2].s64=%llx\n",
            data[4].u64, data[5].u64, nw_data[2].u64);
        printf(
            "mma_ld:data[6].s64=%llx,data[7].s64=%llx,new_data[3].s64=%llx\n",
            data[6].u64, data[7].u64, nw_data[3].u64);
        if (wmma_type != LOAD_C) { // [한국어] A/B 행렬은 4개 추가 packed 요소도 출력
          printf(
              "mma_ld:data[8].s64=%llx,data[9].s64=%llx,new_data[4].s64=%llx\n",
              data[8].u64, data[9].u64, nw_data[4].s64);
          printf(
              "mma_ld:data[10].s64=%llx,data[11].s64=%llx,new_data[5].s64=%"
              "llx\n",
              data[10].u64, data[11].u64, nw_data[5].u64);
          printf(
              "mma_ld:data[12].s64=%llx,data[13].s64=%llx,new_data[6].s64=%"
              "llx\n",
              data[12].u64, data[13].u64, nw_data[6].u64);
          printf(
              "mma_ld:data[14].s64=%llx,data[15].s64=%llx,new_data[7].s64=%"
              "llx\n",
              data[14].u64, data[15].u64, nw_data[3].u64);
        }
      }
    }

    // thread->m_last_effective_address = addr;
    // thread->m_last_memory_space = space;
    // [한국어] wmma 명령어는 m_last_effective_address를 갱신하지 않음 — 타이밍 모델에서 inst.addr로 직접 처리
  }
}

/*
 * [한국어]
 * lg2_impl - PTX `lg2` 명령어 구현: 2를 밑으로 하는 로그 근사 (F32만 지원)
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, 소스1, 타입 정보 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `lg2.approx.f32 d, a` 명령어를 구현한다. d = log2(a) (F32).
 * GPU HW는 SFU(Special Function Unit)에서 lg2를 근사값으로 계산하는데,
 * GPGPU-Sim은 log(a)/log(2)로 소프트웨어 에뮬레이션한다 (double 정밀도 사용).
 * F32 이외 타입은 지원하지 않아 오류를 발생시킨다.
 * ex2_impl()과 역함수 관계 (lg2는 ex2의 역).
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [lg2_impl] → log(), log(2)
 */
void lg2_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d; // [한국어] a=소스 레지스터, d=결과 레지스터
  const operand_info &dst = pI->dst(); // [한국어] log2 결과가 저장될 목적지 레지스터
  const operand_info &src1 = pI->src1(); // [한국어] 로그를 계산할 입력값 소스 오퍼랜드

  unsigned i_type = pI->get_type(); // [한국어] PTX 데이터 타입 (F32_TYPE만 유효)

  a = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] 소스 레지스터에서 입력값을 읽음

  switch (i_type) { // [한국어] 타입 확인 — F32만 지원
    case F32_TYPE:
      d.f32 = log(a.f32) / log(2); // [한국어] log2(a) = log(a)/log(2) — 소프트웨어 에뮬레이션 (double log 사용 후 float으로 저장)
      break;
    default:
      printf("Execution error: type mismatch with instruction\n"); // [한국어] F32 이외 타입에 lg2 적용 — 타입 오류
      assert(0); // [한국어] 지원하지 않는 타입으로 시뮬레이션 강제 종료
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI); // [한국어] log2(a) 결과를 목적지 레지스터에 기록
}

/*
 * [한국어]
 * mad24_impl - PTX `mad24` 명령어 구현: 24비트 곱셈 후 덧셈
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1/2/3, 타입, .hi/.lo, 포화 모드 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `mad24.{hi|lo}.{sat}.{s32|u32} d, a, b, c` 명령어를 구현한다.
 * 실제 GPU HW는 하위 24비트만 사용한 고속 곱셈을 수행하지만, GPGPU-Sim은 32비트 전체 곱셈으로 에뮬레이션한다.
 * 두 가지 결과 선택 방식:
 *   - .hi: 64비트 곱셈 결과의 상위 절반(t >> 16)에 c를 더함
 *     주의: 정확한 mad24.hi는 t >> 16이 아니라 t >> 24이어야 할 수 있음 — 구현 주의 필요
 *   - .lo: 32비트 곱셈 결과의 하위 절반(t.s32)에 c를 더함
 * S32 .hi 모드에서 포화 모드 활성화 시 S32 범위로 클램핑.
 * .wide 모드는 지원하지 않음 (assert로 검증).
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [mad24_impl] → thread->set_operand_value()
 */
void mad24_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst(); // [한국어] 결과를 저장할 목적지 레지스터
  const operand_info &src1 = pI->src1(); // [한국어] 첫 번째 피연산자 (a)
  const operand_info &src2 = pI->src2(); // [한국어] 두 번째 피연산자 (b)
  const operand_info &src3 = pI->src3(); // [한국어] 덧셈 피연산자 (c)
  ptx_reg_t d, t; // [한국어] d=최종 결과, t=중간 곱셈 결과

  unsigned i_type = pI->get_type(); // [한국어] PTX 데이터 타입 (S32_TYPE 또는 U32_TYPE)
  ptx_reg_t a = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] 소스 a 읽기
  ptx_reg_t b = thread->get_operand_value(src2, dst, i_type, thread, 1); // [한국어] 소스 b 읽기
  ptx_reg_t c = thread->get_operand_value(src3, dst, i_type, thread, 1); // [한국어] 소스 c 읽기

  unsigned sat_mode = pI->saturation_mode(); // [한국어] 포화 모드 (.sat 수식어)

  assert(!pI->is_wide()); // [한국어] mad24는 .wide 모드를 지원하지 않음 — PTX 규격 확인

  switch (i_type) { // [한국어] 타입에 따라 곱셈/덧셈 수행
    case S32_TYPE:
      t.s64 = a.s32 * b.s32; // [한국어] 32비트 signed 곱셈 — 결과는 64비트에 저장 (오버플로우 방지)
      if (pI->is_hi()) {
        d.s64 = (t.s64 >> 16) + c.s32; // [한국어] .hi: 64비트 곱 상위 16비트 위치에서 추출 후 c 덧셈
        // [한국어] 주의: mad24.hi는 보통 t >> 24이어야 정확한 24비트 곱의 상위 절반인데 >> 16을 사용 — 동작 차이 가능
        if (sat_mode) { // [한국어] .sat 수식어: S32 범위 [0x80000000, 0x7FFFFFFF]로 클램핑
          if (d.s64 > (int)0x7FFFFFFF)
            d.s64 = (int)0x7FFFFFFF; // [한국어] 상한 클램핑: S32 최대값
          else if (d.s64 < (int)0x80000000)
            d.s64 = (int)0x80000000; // [한국어] 하한 클램핑: S32 최소값
        }
      } else if (pI->is_lo())
        d.s64 = t.s32 + c.s32; // [한국어] .lo: 곱 결과의 하위 32비트(t.s32)에 c 덧셈
      else
        assert(0); // [한국어] .hi도 .lo도 아닌 경우 — PTX 규격 위반
      break;
    case U32_TYPE:
      t.u64 = a.u32 * b.u32; // [한국어] 32비트 unsigned 곱셈 — 64비트 결과에 저장
      if (pI->is_hi())
        d.u64 = (t.u64 >> 16) + c.u32; // [한국어] .hi: 64비트 곱의 상위 절반(>> 16) 추출 후 c 덧셈
      else if (pI->is_lo())
        d.u64 = t.u32 + c.u32; // [한국어] .lo: 곱 결과 하위 32비트에 c 덧셈
      else
        assert(0); // [한국어] .hi도 .lo도 아닌 경우 — PTX 규격 위반
      break;
    default:
      assert(0); // [한국어] S32/U32 외 타입은 mad24 미지원
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI); // [한국어] 계산된 결과를 목적지 레지스터에 기록
}

/*
 * [한국어]
 * mad_impl - PTX `mad` 명령어 구현: 곱셈 후 덧셈 (캐리 없음)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `mad.{hi|lo|wide}.{type} d, a, b, c` 명령어를 구현한다. d = a*b + c.
 * use_carry=false로 mad_def()에 위임한다 — 캐리 비트를 사용하지 않는 일반 곱셈 덧셈.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [mad_impl] → mad_def(use_carry=false)
 */
void mad_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  mad_def(pI, thread, false); // [한국어] 일반 곱셈 덧셈(carry 없음) — mad_def에 위임
}

/*
 * [한국어]
 * madp_impl - PTX `madp` 명령어 구현: 캐리 포함 곱셈 덧셈
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `madp` 명령어를 구현한다. use_carry=true로 mad_def()에 위임한다.
 * 캐리 비트(4번째 오퍼랜드)를 곱셈 덧셈에 포함한다.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [madp_impl] → mad_def(use_carry=true)
 */
void madp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  mad_def(pI, thread, true); // [한국어] 캐리 포함 곱셈 덧셈 — mad_def에 위임 (use_carry=true)
}

/*
 * [한국어]
 * madc_impl - PTX `madc` 명령어 구현: 캐리 포함 곱셈 덧셈 (madp와 동일)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `madc` 명령어를 구현한다. madp_impl과 동일하게 use_carry=true로 mad_def()에 위임한다.
 * PTX에서 madc는 carry-in이 있는 곱셈 덧셈(multiply-add with carry)을 의미한다.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [madc_impl] → mad_def(use_carry=true)
 */
void madc_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  mad_def(pI, thread, true); // [한국어] 캐리 포함 곱셈 덧셈 — mad_def에 위임 (use_carry=true)
}

/*
 * [한국어]
 * mad_def - PTX mad/fma/madp/madc 명령어의 실질적 곱셈-누산 구현
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1/2/3, 타입, .hi/.lo/.wide, 반올림/포화 모드 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 * @use_carry: true이면 4번째 오퍼랜드에서 캐리 비트를 읽어 덧셈에 포함 (madp/madc)
 *
 * d = a*b + c (+carry_bit) 연산을 모든 PTX 정수/float 타입에 대해 구현한다.
 * mad_impl(), madp_impl(), madc_impl(), fma_impl()에서 호출되는 공통 구현 함수.
 *
 * 동작 분기:
 *   - 정수 타입(S16/S32/S64/U16/U32/U64): .hi/.lo/.wide에 따라 곱 결과의 어느 절반을 사용할지 결정
 *     .wide: 곱 결과의 전체를 wider 타입으로 저장
 *     .hi: 곱 결과 상위 절반 사용 (예: S32*.hi → t >> 32)
 *     .lo: 곱 결과 하위 절반 사용 (예: S32*.lo → t.s32)
 *   - float 타입(F16/F32/F64): fegetround/fesetround로 C FP 환경의 반올림 모드를 일시 변경 후
 *     a*b + c 수행, 포화 모드 적용, 원래 반올림 모드 복원
 * 캐리 비트: use_carry=true이면 4번째 오퍼랜드(predicate 레지스터)의 bit[2]를 캐리로 사용.
 * 오버플로우/캐리 결과는 set_operand_value()의 overflow/carry 인자로 전달되어 CC 레지스터 업데이트.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드 — fesetround 호출이 스레드 로컬 FP 환경을 변경하므로
 *   다중 스레드 동시 실행 시 주의 필요 (GPGPU-Sim은 단일 스레드로 시뮬레이션하므로 안전).
 *
 * 호출 체인:
 *   mad_impl/madp_impl/madc_impl/fma_impl → [mad_def] → fesetround() → thread->set_operand_value()
 */
void mad_def(const ptx_instruction *pI, ptx_thread_info *thread,
             bool use_carry) {
  const operand_info &dst = pI->dst(); // [한국어] 곱셈-덧셈 결과를 저장할 목적지 레지스터
  const operand_info &src1 = pI->src1(); // [한국어] 첫 번째 피연산자 (a)
  const operand_info &src2 = pI->src2(); // [한국어] 두 번째 피연산자 (b)
  const operand_info &src3 = pI->src3(); // [한국어] 누산 피연산자 (c)
  ptx_reg_t d, t; // [한국어] d=최종 결과, t=중간 곱셈 결과 (overflow 방지를 위해 넓은 타입 사용)

  int carry = 0; // [한국어] 덧셈 캐리 아웃 (set_operand_value에 전달하여 CC 레지스터 업데이트)
  int overflow = 0; // [한국어] 오버플로우 플래그 (현재 항상 0 — 미구현)

  unsigned i_type = pI->get_type(); // [한국어] PTX 데이터 타입
  ptx_reg_t a = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] 소스 a 읽기
  ptx_reg_t b = thread->get_operand_value(src2, dst, i_type, thread, 1); // [한국어] 소스 b 읽기
  ptx_reg_t c = thread->get_operand_value(src3, dst, i_type, thread, 1); // [한국어] 소스 c 읽기

  // take the carry bit, it should be the 4th operand
  // [한국어] 캐리 비트: use_carry가 true이면 4번째 오퍼랜드(predicate 레지스터)에서 캐리 비트를 읽음
  ptx_reg_t carry_bit; // [한국어] 캐리 입력 비트를 담을 레지스터
  carry_bit.u64 = 0; // [한국어] 캐리 비트 초기화 (0 = no carry)
  if (use_carry) {
    const operand_info &carry = pI->operand_lookup(4); // [한국어] 4번째 오퍼랜드(carry predicate 레지스터) 획득
    carry_bit = thread->get_operand_value(carry, dst, PRED_TYPE, thread, 0);
    // [한국어] carry predicate 레지스터 값을 읽음 (derefFlag=0: 직접 읽기)
    carry_bit.pred &= 0x4; // [한국어] predicate의 bit[2]만 추출 — 이 비트가 carry flag
    carry_bit.pred >>= 2; // [한국어] bit[2]를 bit[0] 위치로 이동하여 0 또는 1 값 얻음
  }

  unsigned rounding_mode = pI->rounding_mode(); // [한국어] PTX 반올림 모드 (float 타입에서 fesetround 호출에 사용)

  switch (i_type) { // [한국어] PTX 타입에 따라 곱셈-덧셈 수행
    case S16_TYPE:
      t.s32 = a.s16 * b.s16; // [한국어] 16비트 signed 곱셈 — 32비트에 저장 (오버플로우 방지)
      if (pI->is_wide())
        d.s32 = t.s32 + c.s32 + carry_bit.pred; // [한국어] .wide: 32비트 전체 결과를 32비트 목적지에 저장
      else if (pI->is_hi())
        d.s16 = (t.s32 >> 16) + c.s16 + carry_bit.pred; // [한국어] .hi: 32비트 곱의 상위 16비트 + c + carry
      else if (pI->is_lo())
        d.s16 = t.s16 + c.s16 + carry_bit.pred; // [한국어] .lo: 32비트 곱의 하위 16비트 + c + carry
      else
        assert(0); // [한국어] .hi/.lo/.wide 중 하나 이어야 함
      carry =
          ((long long int)(t.s32 + c.s32 + carry_bit.pred) & 0x100000000) >> 32;
      // [한국어] 캐리 아웃 계산: 합이 2^32를 초과하면 캐리 발생 — bit[32]를 추출하여 carry 설정
      break;
    case S32_TYPE:
      t.s64 = a.s32 * b.s32; // [한국어] 32비트 signed 곱셈 — 64비트에 저장 (오버플로우 방지)
      if (pI->is_wide())
        d.s64 = t.s64 + c.s64 + carry_bit.pred; // [한국어] .wide: 64비트 전체 결과를 64비트 목적지에 저장
      else if (pI->is_hi())
        d.s32 = (t.s64 >> 32) + c.s32 + carry_bit.pred; // [한국어] .hi: 64비트 곱의 상위 32비트 + c + carry
      else if (pI->is_lo())
        d.s32 = t.s32 + c.s32 + carry_bit.pred; // [한국어] .lo: 64비트 곱의 하위 32비트 + c + carry
      else
        assert(0); // [한국어] .hi/.lo/.wide 중 하나 이어야 함
      break;
    case S64_TYPE:
      t.s64 = a.s64 * b.s64; // [한국어] 64비트 signed 곱셈 (결과가 64비트를 초과하면 오버플로우 발생 가능)
      assert(!pI->is_wide()); // [한국어] S64.wide는 미지원
      assert(!pI->is_hi()); // [한국어] S64.hi는 미지원
      assert(use_carry == false); // [한국어] S64에서는 캐리 모드 미지원
      if (pI->is_lo())
        d.s64 = t.s64 + c.s64 + carry_bit.pred; // [한국어] .lo: 64비트 곱의 하위 64비트 + c (캐리 항상 0)
      else
        assert(0); // [한국어] .lo만 지원
      break;
    case U16_TYPE:
      t.u32 = a.u16 * b.u16; // [한국어] 16비트 unsigned 곱셈 — 32비트에 저장
      if (pI->is_wide())
        d.u32 = t.u32 + c.u32 + carry_bit.pred; // [한국어] .wide: 32비트 전체 결과
      else if (pI->is_hi())
        d.u16 = (t.u32 + c.u16 + carry_bit.pred) >> 16; // [한국어] .hi: 합의 상위 16비트 추출
      else if (pI->is_lo())
        d.u16 = t.u16 + c.u16 + carry_bit.pred; // [한국어] .lo: 하위 16비트 + c + carry
      else
        assert(0); // [한국어] .hi/.lo/.wide 중 하나 이어야 함
      carry = ((long long int)((long long int)t.u32 + c.u32 + carry_bit.pred) &
               0x100000000) >>
              32;
      // [한국어] 캐리 아웃 계산: 32비트 합이 2^32를 초과하면 캐리 — bit[32] 추출
      break;
    case U32_TYPE:
      t.u64 = a.u32 * b.u32; // [한국어] 32비트 unsigned 곱셈 — 64비트에 저장
      if (pI->is_wide())
        d.u64 = t.u64 + c.u64 + carry_bit.pred; // [한국어] .wide: 64비트 전체 결과
      else if (pI->is_hi())
        d.u32 = (t.u64 + c.u32 + carry_bit.pred) >> 32; // [한국어] .hi: 64비트 합의 상위 32비트
      else if (pI->is_lo())
        d.u32 = t.u32 + c.u32 + carry_bit.pred; // [한국어] .lo: 하위 32비트 + c + carry
      else
        assert(0); // [한국어] .hi/.lo/.wide 중 하나 이어야 함
      break;
    case U64_TYPE:
      t.u64 = a.u64 * b.u64; // [한국어] 64비트 unsigned 곱셈 (결과 오버플로우 가능)
      assert(!pI->is_wide()); // [한국어] U64.wide 미지원
      assert(!pI->is_hi()); // [한국어] U64.hi 미지원
      assert(use_carry == false); // [한국어] U64에서는 캐리 모드 미지원
      if (pI->is_lo())
        d.u64 = t.u64 + c.u64 + carry_bit.pred; // [한국어] .lo: 64비트 곱 + c (캐리 항상 0)
      else
        assert(0); // [한국어] .lo만 지원
      break;
    case F16_TYPE: {
      // assert(0);
      // break;
      // [한국어] F16 fma/mad: float16 정밀도로 a*b + c 수행
      assert(use_carry == false); // [한국어] float 타입에서는 캐리 미지원
      int orig_rm = fegetround(); // [한국어] 현재 FP 반올림 모드를 저장 (복원을 위해)
      switch (rounding_mode) { // [한국어] PTX 반올림 모드에 따라 C FP 환경 변경
        case RN_OPTION:
          break; // [한국어] RN: 기본 반올림 모드(최근접 짝수) — 변경 불필요
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO); // [한국어] RZ: 0 방향 반올림으로 FP 환경 변경
          break;
        default:
          assert(0); // [한국어] F16에서 RN/RZ 이외 반올림 모드 미지원
          break;
      }
      d.f16 = a.f16 * b.f16 + c.f16; // [한국어] F16 곱셈 덧셈 수행 (현재 FP 환경의 반올림 모드 적용)
      if (pI->saturation_mode()) { // [한국어] .sat 수식어: 결과를 [0, 1] 범위로 클램핑
        if (d.f16 < 0)
          d.f16 = 0; // [한국어] 하한 포화: 0 미만이면 0으로
        else if (d.f16 > 1.0f)
          d.f16 = 1.0f; // [한국어] 상한 포화: 1.0 초과이면 1.0으로
      }
      fesetround(orig_rm); // [한국어] 원래 FP 반올림 환경 복원
      break;
    }
    case F32_TYPE: {
      // [한국어] F32 fma/mad: float32 정밀도로 a*b + c 수행
      assert(use_carry == false); // [한국어] float 타입에서는 캐리 미지원
      int orig_rm = fegetround(); // [한국어] 현재 FP 반올림 모드를 저장
      switch (rounding_mode) { // [한국어] PTX 반올림 모드에 따라 C FP 환경 변경
        case RN_OPTION:
          break; // [한국어] RN: 변경 불필요
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO); // [한국어] RZ: 0 방향 반올림 환경으로 변경
          break;
        default:
          // assert(0);
          break; // [한국어] F32에서 RN/RZ 이외 모드 — assert 해제, 그냥 통과
      }
      d.f32 = a.f32 * b.f32 + c.f32; // [한국어] F32 곱셈 덧셈 수행 (현재 FP 환경 적용)
      if (pI->saturation_mode()) { // [한국어] .sat 수식어: 결과를 [0, 1] 범위로 클램핑
        if (d.f32 < 0)
          d.f32 = 0; // [한국어] 하한 포화
        else if (d.f32 > 1.0f)
          d.f32 = 1.0f; // [한국어] 상한 포화
      }
      fesetround(orig_rm); // [한국어] 원래 FP 반올림 환경 복원
      break;
    }
    case F64_TYPE:
    case FF64_TYPE: {
      // [한국어] F64/FF64 fma/mad: double 정밀도로 a*b + c 수행
      assert(use_carry == false); // [한국어] float 타입에서는 캐리 미지원
      int orig_rm = fegetround(); // [한국어] 현재 FP 반올림 모드를 저장
      switch (rounding_mode) { // [한국어] PTX 반올림 모드에 따라 C FP 환경 변경
        case RN_OPTION:
          break; // [한국어] RN: 변경 불필요
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO); // [한국어] RZ: 0 방향 반올림 환경으로 변경
          break;
        default:
          assert(0); // [한국어] F64에서 RN/RZ 이외 모드 미지원
          break;
      }
      d.f64 = a.f64 * b.f64 + c.f64; // [한국어] F64 곱셈 덧셈 수행 (현재 FP 환경 적용)
      if (pI->saturation_mode()) { // [한국어] .sat 수식어: 결과를 [0, 1] 범위로 클램핑
        if (d.f64 < 0)
          d.f64 = 0; // [한국어] 하한 포화
        else if (d.f64 > 1.0f)
          d.f64 = 1.0; // [한국어] 상한 포화 (1.0f를 double로 암묵 변환)
      }
      fesetround(orig_rm); // [한국어] 원래 FP 반올림 환경 복원
      break;
    }
    default:
      assert(0); // [한국어] 지원하지 않는 타입 — 구현 오류
      break;
  }
  thread->set_operand_value(dst, d, i_type, thread, pI, overflow, carry);
  // [한국어] 계산된 결과를 목적지 레지스터에 기록; overflow와 carry 값도 전달하여 CC 레지스터 업데이트
}

/*
 * [한국어]
 * isNaN(float) - float 값이 NaN인지 검사하는 인라인 헬퍼
 *
 * @x: 검사할 float 값
 * @return: NaN이면 true, 아니면 false
 *
 * C++ std::isnan()의 float 오버로드 래퍼. 다른 명령어 impl 함수에서 NaN 처리 분기 시 사용.
 * 함수 오버로딩을 통해 float/double 타입을 동일한 함수명으로 처리 가능하게 한다.
 *
 * 호출 체인:
 *   (여러 impl 함수들) → [isNaN(float)] → std::isnan(float)
 */
bool isNaN(float x) { return std::isnan(x); } // [한국어] float NaN 검사: C++ std::isnan(float) 위임

/*
 * [한국어]
 * isNaN(double) - double 값이 NaN인지 검사하는 인라인 헬퍼
 *
 * @x: 검사할 double 값
 * @return: NaN이면 true, 아니면 false
 *
 * C++ std::isnan()의 double 오버로드 래퍼. isNaN(float)와 오버로딩 쌍을 이루어
 * 템플릿 없이도 float/double 구분 없이 사용 가능하게 한다.
 *
 * 호출 체인:
 *   (여러 impl 함수들) → [isNaN(double)] → std::isnan(double)
 */
bool isNaN(double x) { return std::isnan(x); } // [한국어] double NaN 검사: C++ std::isnan(double) 위임

/*
 * [한국어]
 * max_impl - PTX `max` 명령어 구현: 두 값 중 최대값 반환
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `max.type d, a, b` 명령어를 구현한다. d = max(a, b).
 * 모든 정수 타입(U16/U32/U64/S16/S32/S64)과 float 타입(F32/F64)을 지원한다.
 * MY_MAX_I 매크로는 정수 최대값, MY_MAX_F 매크로는 float 최대값을 계산한다.
 * MY_MAX_F는 NaN 처리를 포함할 수 있다 — 정의에 따라 다름.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [max_impl] → MY_MAX_I / MY_MAX_F 매크로
 */
void max_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d; // [한국어] a=첫 번째 피연산자, b=두 번째 피연산자, d=최대값 결과
  const operand_info &dst = pI->dst(); // [한국어] 최대값이 저장될 목적지 레지스터
  const operand_info &src1 = pI->src1(); // [한국어] 첫 번째 소스 오퍼랜드 (a)
  const operand_info &src2 = pI->src2(); // [한국어] 두 번째 소스 오퍼랜드 (b)

  unsigned i_type = pI->get_type(); // [한국어] PTX 데이터 타입
  a = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] 소스 a를 레지스터에서 읽음
  b = thread->get_operand_value(src2, dst, i_type, thread, 1); // [한국어] 소스 b를 레지스터에서 읽음

  switch (i_type) { // [한국어] 타입에 따라 적절한 최대값 매크로 적용
    case U16_TYPE:
      d.u16 = MY_MAX_I(a.u16, b.u16); // [한국어] unsigned 16비트 최대값
      break;
    case U32_TYPE:
      d.u32 = MY_MAX_I(a.u32, b.u32); // [한국어] unsigned 32비트 최대값
      break;
    case U64_TYPE:
      d.u64 = MY_MAX_I(a.u64, b.u64); // [한국어] unsigned 64비트 최대값
      break;
    case S16_TYPE:
      d.s16 = MY_MAX_I(a.s16, b.s16); // [한국어] signed 16비트 최대값
      break;
    case S32_TYPE:
      d.s32 = MY_MAX_I(a.s32, b.s32); // [한국어] signed 32비트 최대값
      break;
    case S64_TYPE:
      d.s64 = MY_MAX_I(a.s64, b.s64); // [한국어] signed 64비트 최대값
      break;
    case F32_TYPE:
      d.f32 = MY_MAX_F(a.f32, b.f32); // [한국어] float32 최대값 (NaN 처리는 MY_MAX_F 정의에 따름)
      break;
    case F64_TYPE:
    case FF64_TYPE:
      d.f64 = MY_MAX_F(a.f64, b.f64); // [한국어] float64/FF64 최대값
      break;
    default:
      printf("Execution error: type mismatch with instruction\n"); // [한국어] 지원하지 않는 타입 — 타입 오류
      assert(0); // [한국어] 시뮬레이션 강제 종료
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI); // [한국어] 최대값 결과를 목적지 레지스터에 기록
}

/*
 * [한국어]
 * membar_impl - PTX `membar` 명령어 구현: 메모리 배리어 (타이밍 시뮬레이터 처리)
 *
 * @pI: 실행 중인 PTX 명령어 (배리어 레벨: .cta/.gl/.sys 수식어 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `membar.{cta|gl|sys}` 명령어를 구현한다.
 * 메모리 배리어는 이전 메모리 연산이 완료되기 전에 이후 연산이 관측되지 않도록 보장한다:
 *   - .cta: CTA(스레드 블록) 내 모든 스레드의 메모리 연산 완료 보장
 *   - .gl: 전체 GPU(global memory)의 메모리 연산 완료 보장
 *   - .sys: 시스템 전체(CPU + GPU) 메모리 연산 완료 보장
 * 기능 시뮬레이션에서는 메모리 순서가 이미 순차적이므로 아무것도 수행하지 않는다.
 * 타이밍 시뮬레이션(shader.cc)에서 이 명령어를 인식하여 메모리 펜스를 처리한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [membar_impl] → (아무것도 안 함, 타이밍 시뮬레이터가 처리)
 */
void membar_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // handled by timing simulator
  // [한국어] 메모리 배리어는 기능 시뮬레이션에서는 no-op — 타이밍 시뮬레이터(shader.cc)가 실제 처리
}

/*
 * [한국어]
 * min_impl - PTX `min` 명령어 구현: 두 값 중 최소값 반환
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `min.type d, a, b` 명령어를 구현한다. d = min(a, b).
 * max_impl()과 대칭적인 구조이며 MY_MAX_I/MY_MAX_F 대신 MY_MIN_I/MY_MIN_F를 사용한다.
 * 모든 정수 타입(U16/U32/U64/S16/S32/S64)과 float 타입(F32/F64)을 지원한다.
 * MY_MIN_F는 NaN 처리를 포함할 수 있다 — 정의에 따라 다름.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [min_impl] → MY_MIN_I / MY_MIN_F 매크로
 */
void min_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d; // [한국어] a=첫 번째 피연산자, b=두 번째 피연산자, d=최소값 결과
  const operand_info &dst = pI->dst(); // [한국어] 최소값이 저장될 목적지 레지스터
  const operand_info &src1 = pI->src1(); // [한국어] 첫 번째 소스 오퍼랜드 (a)
  const operand_info &src2 = pI->src2(); // [한국어] 두 번째 소스 오퍼랜드 (b)

  unsigned i_type = pI->get_type(); // [한국어] PTX 데이터 타입
  a = thread->get_operand_value(src1, dst, i_type, thread, 1); // [한국어] 소스 a를 레지스터에서 읽음
  b = thread->get_operand_value(src2, dst, i_type, thread, 1); // [한국어] 소스 b를 레지스터에서 읽음

  switch (i_type) { // [한국어] 타입에 따라 적절한 최소값 매크로 적용
    case U16_TYPE:
      d.u16 = MY_MIN_I(a.u16, b.u16); // [한국어] unsigned 16비트 최소값
      break;
    case U32_TYPE:
      d.u32 = MY_MIN_I(a.u32, b.u32); // [한국어] unsigned 32비트 최소값
      break;
    case U64_TYPE:
      d.u64 = MY_MIN_I(a.u64, b.u64); // [한국어] unsigned 64비트 최소값
      break;
    case S16_TYPE:
      d.s16 = MY_MIN_I(a.s16, b.s16); // [한국어] signed 16비트 최소값
      break;
    case S32_TYPE:
      d.s32 = MY_MIN_I(a.s32, b.s32); // [한국어] signed 32비트 최소값
      break;
    case S64_TYPE:
      d.s64 = MY_MIN_I(a.s64, b.s64); // [한국어] signed 64비트 최소값
      break;
    case F32_TYPE:
      d.f32 = MY_MIN_F(a.f32, b.f32); // [한국어] float32 최소값 (NaN 처리는 MY_MIN_F 정의에 따름)
      break;
    case F64_TYPE:
    case FF64_TYPE:
      d.f64 = MY_MIN_F(a.f64, b.f64); // [한국어] float64/FF64 최소값
      break;
    default:
      printf("Execution error: type mismatch with instruction\n"); // [한국어] 지원하지 않는 타입 — 타입 오류
      assert(0); // [한국어] 시뮬레이션 강제 종료
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI); // [한국어] 최소값 결과를 목적지 레지스터에 기록
}

/*
 * [한국어]
 * mov_impl - PTX `mov` 명령어 구현: 레지스터/상수/벡터 간 데이터 이동 및 pack/unpack
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, 소스, 타입, 벡터 수식어 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `mov.type d, a` 명령어를 구현한다. 레지스터 값 복사, 상수 로드, 벡터 pack/unpack,
 * predicate 리터럴 변환을 처리한다.
 *   - 일반 mov: 소스 값을 그대로 목적지 레지스터에 복사한다.
 *   - 벡터 mov: 여러 좁은 정수 요소(B8/B16/B32)를 하나의 B16/B32/B64 레지스터로 pack하거나
 *     반대로 unpack한다.
 *   - predicate 리터럴: PTX에서 0=false, 1=true이나 PTXPlus zero-flag 규칙에 따라 반전한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [mov_impl] → get_operand_value() / set_operand_value()
 */

void mov_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t data;  // [한국어] 결과를 저장할 레지스터 선언

  const operand_info &dst = pI->dst();  // [한국어] 목적지 레지스터
  const operand_info &src1 = pI->src1();  // [한국어] 소스 오퍼랜드
  unsigned i_type = pI->get_type();  // [한국어] PTX 데이터 타입
  assert(src1.is_param_local() == 0);  // [한국어] 로컬 param 직접 이동은 미지원

  if ((src1.is_vector() || dst.is_vector()) && (i_type != BB64_TYPE) &&  // [한국어] 벡터 pack/unpack 분기
      (i_type != BB128_TYPE) && (i_type != FF64_TYPE)) {
    // pack or unpack operation  // [한국어] 벡터 pack 또는 unpack 연산
    unsigned nbits_to_move;
    ptx_reg_t tmp_bits;

    switch (pI->get_type()) {  // [한국어] pack/unpack 요소 크기 분기
      case B16_TYPE:
        nbits_to_move = 16;
        break;
      case B32_TYPE:
        nbits_to_move = 32;
        break;
      case B64_TYPE:
        nbits_to_move = 64;
        break;
      default:
        printf(
            "Execution error: mov pack/unpack with unsupported type "
            "qualifier\n");
        assert(0);
        break;
    }

    if (src1.is_vector()) {  // [한국어] 소스가 벡터인 경우 pack
      unsigned nelem = src1.get_vect_nelem();  // [한국어] 소스 벡터 요소 수
      ptx_reg_t v[4];
      thread->get_vector_operand_values(src1, v, nelem);  // [한국어] 벡터 요소 읽기

      unsigned bits_per_src_elem = nbits_to_move / nelem;  // [한국어] 소스 요소당 비트 수
      for (unsigned i = 0; i < nelem; i++) {
        switch (bits_per_src_elem) {
          case 8:
            tmp_bits.u64 |= ((unsigned long long)(v[i].u8) << (8 * i));  // [한국어] 요소를 누적 비트에 삽입
            break;
          case 16:
            tmp_bits.u64 |= ((unsigned long long)(v[i].u16) << (16 * i));  // [한국어] 요소를 누적 비트에 삽입
            break;
          case 32:
            tmp_bits.u64 |= ((unsigned long long)(v[i].u32) << (32 * i));  // [한국어] 요소를 누적 비트에 삽입
            break;
          default:
            printf(
                "Execution error: mov pack/unpack with unsupported source/dst "
                "size ratio (src)\n");
            assert(0);
            break;
        }
      }
    } else {
      data = thread->get_operand_value(src1, dst, i_type, thread, 1);  // [한국어] 스칼라 소스 값 읽기

      switch (pI->get_type()) {  // [한국어] pack/unpack 요소 크기 분기
        case B16_TYPE:
          tmp_bits.u16 = data.u16;
          break;
        case B32_TYPE:
          tmp_bits.u32 = data.u32;
          break;
        case B64_TYPE:
          tmp_bits.u64 = data.u64;
          break;
        default:
          assert(0);
          break;
      }
    }

    if (dst.is_vector()) {  // [한국어] 목적지가 벡터인 경우 unpack
      unsigned nelem = dst.get_vect_nelem();
      ptx_reg_t v[4];
      unsigned bits_per_dst_elem = nbits_to_move / nelem;
      for (unsigned i = 0; i < nelem; i++) {
        switch (bits_per_dst_elem) {
          case 8:
            v[i].u8 = (tmp_bits.u64 >> (8 * i)) & ((unsigned long long)0xFF);
            break;
          case 16:
            v[i].u16 =
                (tmp_bits.u64 >> (16 * i)) & ((unsigned long long)0xFFFF);
            break;
          case 32:
            v[i].u32 =
                (tmp_bits.u64 >> (32 * i)) & ((unsigned long long)0xFFFFFFFF);
            break;
          default:
            printf(
                "Execution error: mov pack/unpack with unsupported source/dst "
                "size ratio (dst)\n");
            assert(0);
            break;
        }
      }
      thread->set_vector_operand_values(dst, v[0], v[1], v[2], v[3]);  // [한국어] 벡터 레지스터에 기록
    } else {
      thread->set_operand_value(dst, tmp_bits, i_type, thread, pI);
    }
  } else if (i_type == PRED_TYPE and src1.is_literal() == true) {
    // in ptx, literal input translate to predicate as 0 = false and 1 = true
    // we have adopted the opposite to simplify implementation of zero flags in
    // ptxplus
    data = thread->get_operand_value(src1, dst, i_type, thread, 1);  // [한국어] 스칼라 소스 값 읽기

    ptx_reg_t finaldata;
    finaldata.pred = (data.u32 == 0) ? 1 : 0;  // setting zero-flag in predicate
    thread->set_operand_value(dst, finaldata, i_type, thread, pI);  // [한국어] predicate 결과 기록
  } else {
    data = thread->get_operand_value(src1, dst, i_type, thread, 1);  // [한국어] 스칼라 소스 값 읽기

    thread->set_operand_value(dst, data, i_type, thread, pI);
  }
}

/*
 * [한국어]
 * mul24_impl - PTX `mul24` 명령어 구현: 24비트 곱셈
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, .hi/.lo, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `mul24.{hi|lo}.{s32|u32} d, a, b` 명령어를 구현한다.
 * 실제 GPU는 24비트 입력을 사용하는 고속 곱셈 유닛을 사용하지만,
 * GPGPU-Sim은 하위 24비트를 마스크한 뒤 32/64비트 곱셈으로 에뮬레이션한다.
 * S32의 경우 부호 확장을 위해 비트 23을 기준으로 상위 비트를 채운다.
 * .hi 모드는 결과를 16비트 우시프트, .lo 모드는 하위 32비트를 반환한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [mul24_impl] → mask_and/mask_or → set_operand_value()
 */

void mul24_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // src1_data = srcOperandModifiers(src1_data, src1, dst, i_type, thread);
  // src2_data = srcOperandModifiers(src2_data, src2, dst, i_type, thread);

  src1_data.mask_and(0, 0x00FFFFFF);  // [한국어] 하위 24비트 마스크
  src2_data.mask_and(0, 0x00FFFFFF);  // [한국어] 하위 24비트 마스크

  switch (i_type) {
    case S32_TYPE:
      if (src1_data.get_bit(23)) src1_data.mask_or(0xFFFFFFFF, 0xFF000000);  // [한국어] S32 부호 확장
      if (src2_data.get_bit(23)) src2_data.mask_or(0xFFFFFFFF, 0xFF000000);  // [한국어] S32 부호 확장
      data.s64 = src1_data.s64 * src2_data.s64;  // [한국어] 24비트 signed 곱셈
      break;
    case U32_TYPE:
      data.u64 = src1_data.u64 * src2_data.u64;  // [한국어] 24비트 unsigned 곱셈
      break;
    default:
      printf(
          "GPGPU-Sim PTX: Execution error - type mismatch with instruction\n");
      assert(0);
      break;
  }

  if (pI->is_hi()) {  // [한국어] 상위 16비트 선택
    data.u64 = data.u64 >> 16;  // [한국어] 상위 16비트 추출
    data.mask_and(0, 0xFFFFFFFF);  // [한국어] 하위 32비트 마스크
  } else if (pI->is_lo()) {
    data.mask_and(0, 0xFFFFFFFF);  // [한국어] 하위 32비트 마스크
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

/*
 * [한국어]
 * mul_impl - PTX `mul` 명령어 구현: 곱셈
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, 타입, .hi/.lo/.wide, 반올림/포화 모드 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `mul.{hi|lo|wide}.{type} d, a, b` 명령어를 구현한다. d = a * b.
 * 정수 타입(S16/S32/S64/U16/U32/U64)은 .hi/.lo/.wide에 따라 곱 결과의 일부를 반환하고,
 * 부동소수 타입(F16/F32/F64)은 C FP 반올림 모드를 일시적으로 PTX 모드로 변경 후 곱셈을 수행한다.
 * .sat 수식어 시 결과를 [0,1]로 포화(clamp)한다.
 * 타이밍 모델에서 이 명령어의 레이턴시는 gpgpusim.config의
 * -ptx_opcode_latency_int/fp/dp/sfu 옵션으로 설정된다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [mul_impl] → fesetround() → set_operand_value()
 */

void mul_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  ptx_reg_t d, t;

  unsigned i_type = pI->get_type();
  ptx_reg_t a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  ptx_reg_t b = thread->get_operand_value(src2, dst, i_type, thread, 1);

  unsigned rounding_mode = pI->rounding_mode();  // [한국어] PTX 반올림 모드

  switch (i_type) {
    case S16_TYPE:  // [한국어] S16 곱셈 분기
      t.s32 = ((int)a.s16) * ((int)b.s16);
      if (pI->is_wide())
        d.s32 = t.s32;
      else if (pI->is_hi())
        d.s16 = (t.s32 >> 16);
      else if (pI->is_lo())
        d.s16 = t.s16;
      else
        assert(0);
      break;
    case S32_TYPE:  // [한국어] S32 곱셈 분기
      t.s64 = ((long long)a.s32) * ((long long)b.s32);
      if (pI->is_wide())
        d.s64 = t.s64;
      else if (pI->is_hi())
        d.s32 = (t.s64 >> 32);
      else if (pI->is_lo())
        d.s32 = t.s32;
      else
        assert(0);
      break;
    case S64_TYPE:  // [한국어] S64 곱셈 분기
      t.s64 = a.s64 * b.s64;
      assert(!pI->is_wide());
      // assert(!pI->is_hi());
      d.s64 = t.s64;
      break;
    case U16_TYPE:  // [한국어] U16 곱셈 분기
      t.u32 = ((unsigned)a.u16) * ((unsigned)b.u16);
      if (pI->is_wide())
        d.u32 = t.u32;
      else if (pI->is_lo())
        d.u16 = t.u16;
      else if (pI->is_hi())
        d.u16 = (t.u32 >> 16);
      else
        assert(0);
      break;
    case U32_TYPE:  // [한국어] U32 곱셈 분기
      t.u64 = ((unsigned long long)a.u32) * ((unsigned long long)b.u32);
      if (pI->is_wide())
        d.u64 = t.u64;
      else if (pI->is_lo())
        d.u32 = t.u32;
      else if (pI->is_hi())
        d.u32 = (t.u64 >> 32);
      else
        assert(0);
      break;
    case U64_TYPE:  // [한국어] U64 곱셈 분기
      t.u64 = a.u64 * b.u64;
      assert(!pI->is_wide());
      assert(!pI->is_hi());
      if (pI->is_lo())
        d.u64 = t.u64;
      else
        assert(0);
      break;
    case F16_TYPE: {  // [한국어] F16 곱셈 분기
      // assert(0);
      // break;
      int orig_rm = fegetround();  // [한국어] 현재 FP 반올림 모드 저장
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);  // [한국어] RZ(0 방향) 반올림 모드 설정
          break;
        default:
          assert(0);
          break;
      }

      d.f16 = a.f16 * b.f16;  // [한국어] F16 곱셈 수행

      if (pI->saturation_mode()) {  // [한국어] 포화(.sat) 모드 분기
        if (d.f16 < 0)
          d.f16 = 0;
        else if (d.f16 > 1.0f)
          d.f16 = 1.0f;
      }
      fesetround(orig_rm);  // [한국어] 원래 FP 반올림 모드 복원
      break;
    }
    case F32_TYPE: {  // [한국어] F32 곱셈 분기
      int orig_rm = fegetround();  // [한국어] 현재 FP 반올림 모드 저장
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);  // [한국어] RZ(0 방향) 반올림 모드 설정
          break;
        default:
          assert(0);
          break;
      }

      d.f32 = a.f32 * b.f32;  // [한국어] F32 곱셈 수행

      if (pI->saturation_mode()) {  // [한국어] 포화(.sat) 모드 분기
        if (d.f32 < 0)
          d.f32 = 0;
        else if (d.f32 > 1.0f)
          d.f32 = 1.0f;
      }
      fesetround(orig_rm);  // [한국어] 원래 FP 반올림 모드 복원
      break;
    }
    case F64_TYPE:  // [한국어] F64 곱셈 분기
    case FF64_TYPE: {
      int orig_rm = fegetround();  // [한국어] 현재 FP 반올림 모드 저장
      switch (rounding_mode) {
        case RN_OPTION:
          break;
        case RZ_OPTION:
          fesetround(FE_TOWARDZERO);  // [한국어] RZ(0 방향) 반올림 모드 설정
          break;
        default:
          assert(0);
          break;
      }
      d.f64 = a.f64 * b.f64;  // [한국어] F64 곱셈 수행
      if (pI->saturation_mode()) {  // [한국어] 포화(.sat) 모드 분기
        if (d.f64 < 0)
          d.f64 = 0;
        else if (d.f64 > 1.0f)
          d.f64 = 1.0;
      }
      fesetround(orig_rm);  // [한국어] 원래 FP 반올림 모드 복원
      break;
    }
    default:
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

/*
 * [한국어]
 * neg_impl - PTX `neg` 명령어 구현: 부호 반전
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, 소스, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `neg.type d, a` 명령어를 구현한다. d = -a.
 * signed 정수와 부동소수 타입만 지원하며, unsigned 정수는 정의되지 않아 assert로 처리한다.
 * 정수의 경우 0에서 빼는 방식으로 부호를 반전한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [neg_impl] → set_operand_value()
 */

void neg_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned to_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, to_type, thread, 1);

  switch (to_type) {
    case S8_TYPE:  // [한국어] signed 정수 분기
    case S16_TYPE:
    case S32_TYPE:
    case S64_TYPE:
      data.s64 = 0 - src1_data.s64;  // [한국어] 0에서 빼서 부호 반전
      break;  // seems buggy, but not (just ignore higher bits)
    case U8_TYPE:  // [한국어] unsigned 정수는 미지원
    case U16_TYPE:
    case U32_TYPE:
    case U64_TYPE:
      assert(0);  // [한국어] unsigned neg는 정의되지 않음
      break;
    case F16_TYPE:
      data.f16 = 0.0f - src1_data.f16;  // [한국어] F16 부호 반전
      break;  // assert(0); break;
    case F32_TYPE:
      data.f32 = 0.0f - src1_data.f32;  // [한국어] F32 부호 반전
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = 0.0f - src1_data.f64;  // [한국어] F64 부호 반전
      break;
    default:
      assert(0);  // [한국어] unsigned neg는 정의되지 않음
      break;
  }

  thread->set_operand_value(dst, data, to_type, thread, pI);
}

// nandn bitwise negates second operand then bitwise nands with the first
// operand

/*
 * [한국어]
 * nandn_impl - PTX `nandn` 명령어 구현: NOT-AND (두 번째 피연산자를 NOT 후 AND)
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `nandn.type d, a, b` 명령어를 구현한다. d = ~(a & ~b).
 * predicate 타입인 경우 PTXPlus 규칙(1=false, 0=true)에 맞춰 비트 연산을 수행한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [nandn_impl] → set_operand_value()
 */

void nandn_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)  // [한국어] predicate 타입 분기
    data.pred = (~src1_data.pred & src2_data.pred);  // [한국어] PTXPlus 규칙으로 NANDN
  else
    data.u64 = ~(src1_data.u64 & ~src2_data.u64);  // [한국어] 비트 NANDN

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

// norn bitwise negates first operand then bitwise ands with the second operand

/*
 * [한국어]
 * norn_impl - PTX `norn` 명령어 구현: NOT-OR (첫 번째 피연산자를 NOT 후 OR)
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `norn.type d, a, b` 명령어를 구현한다. d = ~a | b.
 * predicate 타입은 PTXPlus zero-flag 규칙에 따라 처리한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [norn_impl] → set_operand_value()
 */

void norn_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)  // [한국어] predicate 타입 분기
    data.pred = ~(src1_data.pred & ~(src2_data.pred));  // [한국어] PTXPlus 규칙으로 NORN
  else
    data.u64 = ~(src1_data.u64) & src2_data.u64;  // [한국어] 비트 NORN

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

/*
 * [한국어]
 * not_impl - PTX `not` 명령어 구현: 비트/프레디케이트 NOT
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, 소스, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `not.type d, a` 명령어를 구현한다. d = ~a.
 * predicate, B16, B32, B64 타입을 지원한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [not_impl] → set_operand_value()
 */

void not_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case PRED_TYPE:  // [한국어] predicate NOT 분기
      d.pred = (~(a.pred) & 0x000F);  // [한국어] predicate 비트 반전
      break;
    case B16_TYPE:
      d.u16 = ~a.u16;  // [한국어] B16 비트 반전
      break;
    case B32_TYPE:
      d.u32 = ~a.u32;  // [한국어] B32 비트 반전
      break;
    case B64_TYPE:
      d.u64 = ~a.u64;  // [한국어] B64 비트 반전
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

/*
 * [한국어]
 * or_impl - PTX `or` 명령어 구현: 비트/프레디케이트 OR
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `or.type d, a, b` 명령어를 구현한다. d = a | b.
 * predicate 타입은 PTXPlus zero-flag 규칙(1=false, 0=true)을 고려하여 처리한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [or_impl] → set_operand_value()
 */

void or_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)  // [한국어] predicate 타입 분기
    data.pred = ~(~(src1_data.pred) | ~(src2_data.pred));  // [한국어] PTXPlus 규칙으로 OR
  else
    data.u64 = src1_data.u64 | src2_data.u64;  // [한국어] 비트 OR

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

/*
 * [한국어]
 * orn_impl - PTX `orn` 명령어 구현: OR-NOT (두 번째 피연산자를 NOT 후 OR)
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `orn.type d, a, b` 명령어를 구현한다. d = a | ~b.
 * predicate 타입은 PTXPlus zero-flag 규칙에 따라 처리한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [orn_impl] → set_operand_value()
 */

void orn_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)  // [한국어] predicate 타입 분기
    data.pred = ~(~(src1_data.pred) | (src2_data.pred));  // [한국어] PTXPlus 규칙으로 ORN
  else
    data.u64 = src1_data.u64 | ~src2_data.u64;  // [한국어] 비트 ORN

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

/*
 * [한국어]
 * pmevent_impl - PTX `pmevent` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `pmevent`는 성능 모니터링 이벤트를 기록하는 명령어이다.
 * 현재 GPGPU-Sim에서는 구현되어 있지 않아 inst_not_implemented()를 호출한다.
 */

void pmevent_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * popc_impl - PTX `popc` 명령어 구현: 1비트 개수 세기
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, 소스, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `popc.type d, a` 명령어를 구현한다. d = population_count(a).
 * B32/B64 타입을 지원하며 std::bitset::count()로 1의 개수를 센다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [popc_impl] → set_operand_value()
 */

void popc_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src = pI->src1();

  unsigned i_type = pI->get_type();
  src_data = thread->get_operand_value(src, dst, i_type, thread, 1);

  switch (i_type) {
    case B32_TYPE: {
      std::bitset<32> mask(src_data.u32);
      data.u32 = mask.count();
    } break;
    case B64_TYPE: {
      std::bitset<64> mask(src_data.u64);
      data.u32 = mask.count();
    } break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

/*
 * [한국어]
 * prefetch_impl - PTX `prefetch` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `prefetch`는 메모리 프리페치 힌트 명령어이다.
 * 현재 GPGPU-Sim에서는 구현되어 있지 않아 inst_not_implemented()를 호출한다.
 */

void prefetch_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * prefetchu_impl - PTX `prefetchu` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `prefetchu`는 uniform 프리페치 힌트 명령어이다.
 * 현재 GPGPU-Sim에서는 구현되어 있지 않아 inst_not_implemented()를 호출한다.
 */

void prefetchu_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * prmt_mode_present - prmt 명령어의 특수 모드 여부 확인 헬퍼
 *
 * @mode: prmt 모드 값
 * @return: 특수 모드(F4E/B4E/RC8/RC16/ECL/ECR)이면 1, 아니면 0
 *
 * PTX `prmt`의 제어 워드가 특수 모드를 지정하는지 확인한다.
 * 실행 컨텍스트: prmt_impl 낶부.
 */

int prmt_mode_present(int mode) {
  int returnval = 0;
  switch (mode) {
    case PRMT_F4E_MODE:
    case PRMT_B4E_MODE:
    case PRMT_RC8_MODE:
    case PRMT_RC16_MODE:
    case PRMT_ECL_MODE:
    case PRMT_ECR_MODE:
      returnval = 1;
      break;
    default:
      break;
  }
  return returnval;
}

/*
 * [한국어]
 * read_byte - prmt 명령어를 위한 1바이트 선택/변환 헬퍼
 *
 * @mode: prmt 모드
 * @control: 4비트 선택 제어 값
 * @d_sel_index: 목적지 바이트 인덱스(0~3)
 * @value: 64비트 소스 값 (src1 | (src2 << 32))
 * @return: 선택된 바이트를 d_sel_index 위치로 시프트한 32비트 값
 *
 * prmt의 다양한 모드(F4E, B4E, RC8, RC16, ECL, ECR)에 따라 8비트 값을 선택하거나
 * 모드별 테이블로 변환한다. 일반 모드에서는 control이 직접 바이트 선택자로 사용된다.
 * 실행 컨텍스트: prmt_impl 낶부.
 */

int read_byte(int mode, int control, int d_sel_index, signed long long value) {
  int returnval = 0;
  int prmt_f4e_mode[4][4] = {
      {0, 1, 2, 3}, {1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}};
  int prmt_b4e_mode[4][4] = {
      {0, 7, 6, 5}, {1, 0, 7, 6}, {2, 1, 0, 7}, {3, 2, 1, 0}};
  int prmt_rc8_mode[4][4] = {
      {0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}};
  int prmt_ecl_mode[4][4] = {
      {0, 1, 2, 3}, {1, 1, 2, 3}, {2, 2, 2, 3}, {3, 3, 3, 3}};
  int prmt_ecr_mode[4][4] = {
      {0, 0, 0, 0}, {0, 1, 1, 1}, {0, 1, 2, 2}, {0, 1, 2, 3}};
  int prmt_rc16_mode[4][4] = {
      {0, 1, 0, 1}, {2, 3, 2, 3}, {0, 1, 0, 1}, {2, 3, 2, 3}};

  if (!prmt_mode_present(mode)) {
    if (control & 0x8) {
      returnval = 0xff;
    } else {
      returnval = (value >> (8 * control)) & 0xff;
    }
  } else {
    switch (mode) {
      case PRMT_F4E_MODE:
        returnval = prmt_f4e_mode[control][d_sel_index];
        break;
      case PRMT_B4E_MODE:
        returnval = prmt_b4e_mode[control][d_sel_index];
        break;
      case PRMT_RC8_MODE:
        returnval = prmt_rc8_mode[control][d_sel_index];
        break;
      case PRMT_ECL_MODE:
        returnval = prmt_ecl_mode[control][d_sel_index];
        break;
      case PRMT_ECR_MODE:
        returnval = prmt_ecr_mode[control][d_sel_index];
        break;
      case PRMT_RC16_MODE:
        returnval = prmt_rc16_mode[control][d_sel_index];
        break;
        // Change the default from printing "ERROR" to just asserting
      default:
        assert(false);
    }
  }
  return (returnval << 8 * d_sel_index);
}

/*
 * [한국어]
 * prmt_impl - PTX `prmt` 명령어 구현: 바이트 순열(permute)
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, src3(제어), 모드 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `prmt.mode.b32 d, a, b, c` 명령어를 구현한다.
 * 두 32비트 소스 a, b를 64비트로 연결한 뒤, c의 4비트 필드 4개로 각 목적지 바이트를 선택한다.
 * 특수 모드(mode)가 있으면 제어 값의 하위 2비트를 모든 바이트 선택에 사용한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [prmt_impl] → read_byte() → set_operand_value()
 */

void prmt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, src3_data, tmpdata, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  unsigned mode = pI->prmt_op();
  unsigned i_type = pI->get_type();

  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);
  src3_data = thread->get_operand_value(src3, dst, i_type, thread, 1);

  tmpdata.s64 = src1_data.s32 | (src2_data.s64 << 32);
  int ctl[4];

  if (!prmt_mode_present(mode)) {
    ctl[0] = (src3_data.s32 >> 0) & 0xf;
    ctl[1] = (src3_data.s32 >> 4) & 0xf;
    ctl[2] = (src3_data.s32 >> 8) & 0xf;
    ctl[3] = (src3_data.s32 >> 12) & 0xf;
  } else {
    ctl[0] = ctl[1] = ctl[2] = ctl[3] = (src3_data.s32 >> 0) & 0x3;
  }

  data.s32 = 0;
  data.s32 = data.s32 | read_byte(mode, ctl[0], 0, tmpdata.s64);  // First
                                                                  // byte-0
  data.s32 =
      data.s32 | read_byte(mode, ctl[1], 1, tmpdata.s64);  // Second byte-1
  data.s32 = data.s32 | read_byte(mode, ctl[2], 2, tmpdata.s64);  // Third
                                                                  // byte-2
  data.s32 =
      data.s32 | read_byte(mode, ctl[3], 3, tmpdata.s64);  // Fourth byte-3

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

/*
 * [한국어]
 * rcp_impl - PTX `rcp` 명령어 구현: 역수 (reciprocal)
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, 소스, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `rcp.approx.{f32|f64} d, a` 명령어를 구현한다. d = 1 / a.
 * SFU(Special Function Unit)에서 근사적으로 계산되며, 타이밍 모델의 레이턴시는
 * gpgpusim.config의 -ptx_opcode_latency_sfu 옵션으로 설정된다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [rcp_impl] → set_operand_value()
 */

void rcp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:  // [한국어] F32 역수
      data.f32 = 1.0f / src1_data.f32;  // [한국어] F32 역수 계산
      break;
    case F64_TYPE:  // [한국어] F64 역수
    case FF64_TYPE:
      data.f64 = 1.0f / src1_data.f64;  // [한국어] F64 역수 계산
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

/*
 * [한국어]
 * red_impl - PTX `red` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `red`는 메모리상 원자적 축소(atomic reduction) 명령어이다.
 * 현재 GPGPU-Sim에서는 구현되어 있지 않아 inst_not_implemented()를 호출한다.
 */

void red_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * rem_impl - PTX `rem` 명령어 구현: 나머지
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `rem.type d, a, b` 명령어를 구현한다. d = a % b.
 * S32/S64/U32/U64 타입을 지원한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [rem_impl] → set_operand_value()
 */

void rem_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case S32_TYPE:
      data.s32 = src1_data.s32 % src2_data.s32;
      break;
    case S64_TYPE:
      data.s64 = src1_data.s64 % src2_data.s64;
      break;
    case U32_TYPE:
      data.u32 = src1_data.u32 % src2_data.u32;
      break;
    case U64_TYPE:
      data.u64 = src1_data.u64 % src2_data.u64;
      break;
    default:
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

/*
 * [한국어]
 * ret_impl - PTX `ret` 명령어 구현: 서브루틴 복귀
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `ret` 명령어를 구현한다. 현재 스레드의 호출 스택에서 한 프레임을 pop하고,
 * 스택이 비면 스레드를 종료 처리한다(set_done/exitCore/registerExit).
 * 이는 SIMT 함수 호출의 복귀 메커니즘을 지원한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [ret_impl] → callstack_pop()
 */

void ret_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  bool empty = thread->callstack_pop();  // [한국어] 호출 스택에서 한 프레임 pop
  if (empty) {  // [한국어] 호출 스택이 비었으면 스레드 종료
    thread->set_done();  // [한국어] 스레드 완료 표시
    thread->exitCore();  // [한국어] 코어에서 스레드 제거
    thread->registerExit();  // [한국어] 스레드 종료 등록
  }
}

// Ptxplus version of ret instruction.

/*
 * [한국어]
 * retp_impl - PTXPlus `ret` 명령어 구현: PTXPlus 서브루틴 복귀
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTXPlus 버전의 ret 명령어로, callstack_pop_plus()를 사용하여 복귀한다.
 * 호출 스택이 비면 스레드를 종료 처리한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [retp_impl] → callstack_pop_plus()
 */

void retp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  bool empty = thread->callstack_pop_plus();  // [한국어] PTXPlus 호출 스택 pop
  if (empty) {
    thread->set_done();
    thread->exitCore();
    thread->registerExit();
  }
}

/*
 * [한국어]
 * rsqrt_impl - PTX `rsqrt` 명령어 구현: 역제곱근
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, 소스, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `rsqrt.approx.{f32|f64} d, a` 명령어를 구현한다. d = 1 / sqrt(a).
 * 음수 입력 시 NaN, 0 입력 시 +Inf를 반환한다.
 * SFU 연산으로, 타이밍 레이턴시는 -ptx_opcode_latency_sfu 옵션으로 설정된다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [rsqrt_impl] → set_operand_value()
 */

void rsqrt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      if (a.f32 < 0) {  // [한국어] 음수 입력 처리
        d.u64 = 0;
        d.u64 = 0x7fc00000;  // NaN  // [한국어] 음수 입력 시 NaN 반환
      } else if (a.f32 == 0) {  // [한국어] 0 입력 처리
        d.u64 = 0;
        d.u32 = 0x7f800000;  // Inf  // [한국어] 0 입력 시 Inf 반환
      } else
        d.f32 = cuda_math::__internal_accurate_fdividef(1.0f, sqrtf(a.f32));  // [한국어] F32 역제곱근 근사
      break;
    case F64_TYPE:
    case FF64_TYPE:
      if (a.f32 < 0) {  // [한국어] 음수 입력 처리
        d.u64 = 0;
        d.u32 = 0x7fc00000;  // NaN
        float x = d.f32;
        d.f64 = (double)x;
      } else if (a.f32 == 0) {  // [한국어] 0 입력 처리
        d.u64 = 0;
        d.u32 = 0x7f800000;  // Inf  // [한국어] 0 입력 시 Inf 반환
        float x = d.f32;
        d.f64 = (double)x;
      } else
        d.f64 = 1.0 / sqrt(a.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

#define SAD(d, a, b, c) d = c + ((a < b) ? (b - a) : (a - b))

/*
 * [한국어]
 * sad_impl - PTX `sad` 명령어 구현: 절대차 합 (Sum of Absolute Difference)
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, src3, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `sad.type d, a, b, c` 명령어를 구현한다. d = c + |a - b|.
 * 정수/부동소수 타입을 모두 지원하며, 영상 처리/머신러닝에서 자주 사용된다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [sad_impl] → SAD 매크로 → set_operand_value()
 */

void sad_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, c, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  switch (i_type) {
    case U16_TYPE:
      SAD(d.u16, a.u16, b.u16, c.u16);  // [한국어] U16 SAD
      break;
    case U32_TYPE:
      SAD(d.u32, a.u32, b.u32, c.u32);  // [한국어] U32 SAD
      break;
    case U64_TYPE:
      SAD(d.u64, a.u64, b.u64, c.u64);
      break;
    case S16_TYPE:
      SAD(d.s16, a.s16, b.s16, c.s16);
      break;
    case S32_TYPE:
      SAD(d.s32, a.s32, b.s32, c.s32);
      break;
    case S64_TYPE:
      SAD(d.s64, a.s64, b.s64, c.s64);
      break;
    case F32_TYPE:
      SAD(d.f32, a.f32, b.f32, c.f32);  // [한국어] F32 SAD
      break;
    case F64_TYPE:
    case FF64_TYPE:
      SAD(d.f64, a.f64, b.f64, c.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

/*
 * [한국어]
 * selp_impl - PTX `selp` 명령어 구현: 조건 선택
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, src3(조건 predicate) 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `selp.type d, a, b, p` 명령어를 구현한다. p가 true이면 d=a, 아니면 d=b.
 * PTXPlus zero-flag 규칙에 따라 predicate의 최하위 비트를 반전하여 평가한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [selp_impl] → set_operand_value()
 */

void selp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  ptx_reg_t a, b, c, d;

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  // predicate value was changed so the lowest bit being set means the zero flag
  // is set. As a result, the value of c.pred must be inverted to get proper
  // behavior
  d = (!(c.pred & 0x0001)) ? a : b;  // [한국어] predicate 반전 후 조건 선택

  thread->set_operand_value(dst, d, PRED_TYPE, thread, pI);
}

/*
 * [한국어]
 * isFloat - 타입이 부동소수점 타입인지 확인하는 헬퍼
 *
 * @type: PTX 데이터 타입 enum
 * @return: F16/F32/F64/FF64 타입이면 true, 아니면 false
 *
 * set_impl 등에서 목적지 타입에 따라 1.0f/0xFFFFFFFF 등의 결과 형태를 결정할 때 사용한다.
 * 실행 컨텍스트: 비교/설정 명령어 낶부.
 */

bool isFloat(int type) {
  switch (type) {
    case F16_TYPE:
    case F32_TYPE:
    case F64_TYPE:
    case FF64_TYPE:
      return true;
    default:
      return false;
  }
}

/*
 * [한국어]
 * CmpOp - PTX set/setp 명령어용 비교 연산 헬퍼
 *
 * @type: 피연산자의 PTX 데이터 타입
 * @a: 첫 번째 피연산자 값
 * @b: 두 번째 피연산자 값
 * @cmpop: 비교 연산 옵션 (EQ/NE/LT/LE/GT/GE/EQU/NEU/LTU/LEU/GTU/GEU/NUM/NAN/LO/LS/HI/HS)
 * @return: 비교 결과 bool
 *
 * 정수/비트/부동소수점 타입에 따라 다양한 비교 연산을 수행한다.
 * 부동소수점은 NaN 처리가 필요하여 unordered 비교 옵션(EQU/NEU/...)과 NUM/NAN을 별도 처리한다.
 * 정수 unsigned 비교는 LO/LS/HI/HS 옵션을 추가로 지원한다.
 * B16/B32/B64은 EQ/NE만 지원한다.
 * 실행 컨텍스트: setp_impl, set_impl 낶부.
 */

bool CmpOp(int type, ptx_reg_t a, ptx_reg_t b, unsigned cmpop) {
  bool t = false;

  switch (type) {
    case B16_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u16 == b.u16);
          break;
        case NE_OPTION:
          t = (a.u16 != b.u16);
          break;
        default:
          assert(0);
      }

    case B32_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u32 == b.u32);
          break;
        case NE_OPTION:
          t = (a.u32 != b.u32);
          break;
        default:
          assert(0);
      }
    case B64_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u64 == b.u64);
          break;
        case NE_OPTION:
          t = (a.u64 != b.u64);
          break;
        default:
          assert(0);
      }
      break;
    case S8_TYPE:
    case S16_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.s16 == b.s16);
          break;
        case NE_OPTION:
          t = (a.s16 != b.s16);
          break;
        case LT_OPTION:
          t = (a.s16 < b.s16);
          break;
        case LE_OPTION:
          t = (a.s16 <= b.s16);
          break;
        case GT_OPTION:
          t = (a.s16 > b.s16);
          break;
        case GE_OPTION:
          t = (a.s16 >= b.s16);
          break;
        default:
          assert(0);
      }
      break;
    case S32_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.s32 == b.s32);
          break;
        case NE_OPTION:
          t = (a.s32 != b.s32);
          break;
        case LT_OPTION:
          t = (a.s32 < b.s32);
          break;
        case LE_OPTION:
          t = (a.s32 <= b.s32);
          break;
        case GT_OPTION:
          t = (a.s32 > b.s32);
          break;
        case GE_OPTION:
          t = (a.s32 >= b.s32);
          break;
        default:
          assert(0);
      }
      break;
    case S64_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.s64 == b.s64);
          break;
        case NE_OPTION:
          t = (a.s64 != b.s64);
          break;
        case LT_OPTION:
          t = (a.s64 < b.s64);
          break;
        case LE_OPTION:
          t = (a.s64 <= b.s64);
          break;
        case GT_OPTION:
          t = (a.s64 > b.s64);
          break;
        case GE_OPTION:
          t = (a.s64 >= b.s64);
          break;
        default:
          assert(0);
      }
      break;
    case U8_TYPE:
    case U16_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u16 == b.u16);
          break;
        case NE_OPTION:
          t = (a.u16 != b.u16);
          break;
        case LT_OPTION:
          t = (a.u16 < b.u16);
          break;
        case LE_OPTION:
          t = (a.u16 <= b.u16);
          break;
        case GT_OPTION:
          t = (a.u16 > b.u16);
          break;
        case GE_OPTION:
          t = (a.u16 >= b.u16);
          break;
        case LO_OPTION:
          t = (a.u16 < b.u16);
          break;
        case LS_OPTION:
          t = (a.u16 <= b.u16);
          break;
        case HI_OPTION:
          t = (a.u16 > b.u16);
          break;
        case HS_OPTION:
          t = (a.u16 >= b.u16);
          break;
        default:
          assert(0);
      }
      break;
    case U32_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u32 == b.u32);
          break;
        case NE_OPTION:
          t = (a.u32 != b.u32);
          break;
        case LT_OPTION:
          t = (a.u32 < b.u32);
          break;
        case LE_OPTION:
          t = (a.u32 <= b.u32);
          break;
        case GT_OPTION:
          t = (a.u32 > b.u32);
          break;
        case GE_OPTION:
          t = (a.u32 >= b.u32);
          break;
        case LO_OPTION:
          t = (a.u32 < b.u32);
          break;
        case LS_OPTION:
          t = (a.u32 <= b.u32);
          break;
        case HI_OPTION:
          t = (a.u32 > b.u32);
          break;
        case HS_OPTION:
          t = (a.u32 >= b.u32);
          break;
        default:
          assert(0);
      }
      break;
    case U64_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.u64 == b.u64);
          break;
        case NE_OPTION:
          t = (a.u64 != b.u64);
          break;
        case LT_OPTION:
          t = (a.u64 < b.u64);
          break;
        case LE_OPTION:
          t = (a.u64 <= b.u64);
          break;
        case GT_OPTION:
          t = (a.u64 > b.u64);
          break;
        case GE_OPTION:
          t = (a.u64 >= b.u64);
          break;
        case LO_OPTION:
          t = (a.u64 < b.u64);
          break;
        case LS_OPTION:
          t = (a.u64 <= b.u64);
          break;
        case HI_OPTION:
          t = (a.u64 > b.u64);
          break;
        case HS_OPTION:
          t = (a.u64 >= b.u64);
          break;
        default:
          assert(0);
      }
      break;
    case F16_TYPE:
      assert(0);
      break;
    case F32_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.f32 == b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case NE_OPTION:
          t = (a.f32 != b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case LT_OPTION:
          t = (a.f32 < b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case LE_OPTION:
          t = (a.f32 <= b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case GT_OPTION:
          t = (a.f32 > b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case GE_OPTION:
          t = (a.f32 >= b.f32) && !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case EQU_OPTION:
          t = (a.f32 == b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case NEU_OPTION:
          t = (a.f32 != b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case LTU_OPTION:
          t = (a.f32 < b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case LEU_OPTION:
          t = (a.f32 <= b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case GTU_OPTION:
          t = (a.f32 > b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case GEU_OPTION:
          t = (a.f32 >= b.f32) || isNaN(a.f32) || isNaN(b.f32);
          break;
        case NUM_OPTION:
          t = !isNaN(a.f32) && !isNaN(b.f32);
          break;
        case NAN_OPTION:
          t = isNaN(a.f32) || isNaN(b.f32);
          break;
        default:
          assert(0);
      }
      break;
    case F64_TYPE:
    case FF64_TYPE:
      switch (cmpop) {
        case EQ_OPTION:
          t = (a.f64 == b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case NE_OPTION:
          t = (a.f64 != b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case LT_OPTION:
          t = (a.f64 < b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case LE_OPTION:
          t = (a.f64 <= b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case GT_OPTION:
          t = (a.f64 > b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case GE_OPTION:
          t = (a.f64 >= b.f64) && !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case EQU_OPTION:
          t = (a.f64 == b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case NEU_OPTION:
          t = (a.f64 != b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case LTU_OPTION:
          t = (a.f64 < b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case LEU_OPTION:
          t = (a.f64 <= b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case GTU_OPTION:
          t = (a.f64 > b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case GEU_OPTION:
          t = (a.f64 >= b.f64) || isNaN(a.f64) || isNaN(b.f64);
          break;
        case NUM_OPTION:
          t = !isNaN(a.f64) && !isNaN(b.f64);
          break;
        case NAN_OPTION:
          t = isNaN(a.f64) || isNaN(b.f64);
          break;
        default:
          assert(0);
      }
      break;
    default:
      assert(0);
      break;
  }

  return t;
}

/*
 * [한국어]
 * setp_impl - PTX `setp` 명령어 구현: 비교 결과를 predicate에 설정
 *
 * @pI: 실행 중인 PTX 명령어 (목적지 predicate, src1, src2, 비교 옵션 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `setp.cmp.type p, a, b` 명령어를 구현한다. p = (a cmp b).
 * CmpOp()으로 비교한 뒤 PTXPlus zero-flag 규칙에 따라 predicate 값을 반전하여 저장한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [setp_impl] → CmpOp() → set_operand_value()
 */

void setp_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b;

  int t = 0;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  assert(pI->get_num_operands() <  // [한국어] 4번째 boolean 연산 피연산자는 아직 미지원
         4);  // or need to deal with "c" operand / boolOp

  unsigned type = pI->get_type();
  unsigned cmpop = pI->get_cmpop();  // [한국어] 비교 연산 옵션
  a = thread->get_operand_value(src1, dst, type, thread, 1);
  b = thread->get_operand_value(src2, dst, type, thread, 1);

  t = CmpOp(type, a, b, cmpop);  // [한국어] 비교 연산 수행

  ptx_reg_t data;

  // the way ptxplus handles the zero flag, 1 = false and 0 = true
  data.pred =
      (t ==
       0);  // inverting predicate since ptxplus uses "1" for a set zero flag

  thread->set_operand_value(dst, data, PRED_TYPE, thread, pI);
}

/*
 * [한국어]
 * set_impl - PTX `set` 명령어 구현: 비교 결과를 일반 레지스터에 설정
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, 비교 옵션, .abs 수식어 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `set.cmp.type d, a, b` 명령어를 구현한다. d = (a cmp b) ? true_value : false_value.
 * .abs 수식어가 있으면 첫 번째 피연산자의 절댓값을 취한 뒤 비교한다.
 * 목적지가 float 타입이면 1.0f/0.0f, 정수 타입이면 0xFFFFFFFF/0을 저장한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [set_impl] → CmpOp() → set_operand_value()
 */

void set_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b;

  int t = 0;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  assert(pI->get_num_operands() <
         4);  // or need to deal with "c" operand / boolOp

  unsigned src_type = pI->get_type2();
  unsigned cmpop = pI->get_cmpop();

  a = thread->get_operand_value(src1, dst, src_type, thread, 1);
  b = thread->get_operand_value(src2, dst, src_type, thread, 1);

  // Take abs of first operand if needed
  if (pI->is_abs()) {  // [한국어] .abs 수식어: 첫 피연산자 절댓값
    switch (src_type) {
      case S16_TYPE:
        a.s16 = my_abs(a.s16);
        break;
      case S32_TYPE:
        a.s32 = my_abs(a.s32);
        break;
      case S64_TYPE:
        a.s64 = my_abs(a.s64);
        break;
      case U16_TYPE:
        a.u16 = a.u16;
        break;
      case U32_TYPE:
        a.u32 = my_abs(a.u32);
        break;
      case U64_TYPE:
        a.u64 = my_abs(a.u64);
        break;
      case F32_TYPE:
        a.f32 = my_abs(a.f32);
        break;
      case F64_TYPE:
      case FF64_TYPE:
        a.f64 = my_abs(a.f64);
        break;
      default:
        printf("Execution error: type mismatch with instruction\n");
        assert(0);
        break;
    }
  }

  t = CmpOp(src_type, a, b, cmpop);  // [한국어] 비교 연산 수행

  ptx_reg_t data;
  if (isFloat(pI->get_type())) {  // [한국어] 목적지가 float 타입인지 확인
    data.f32 = (t != 0) ? 1.0f : 0.0f;  // [한국어] float 결과: true=1.0f, false=0.0f
  } else {
    data.u32 = (t != 0) ? 0xFFFFFFFF : 0;  // [한국어] 정수 결과: true=0xFFFFFFFF, false=0
  }

  thread->set_operand_value(dst, data, pI->get_type(), thread, pI);
}

/*
 * [한국어]
 * shfl_impl - PTX `shfl` 명령어 구현: 워프 내 레인 간 레지스터 셔플
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1(데이터), src2(오프셋), src3(마스크), 모드 포함)
 * @core: SM(core_t) 객체 — 워프 내 스레드 정보 접근
 * @inst: warp_inst_t 참조 — 활성 스레드 마스크 제공
 *
 * PTX `shfl.mode.b32 d, a, b, c` 명령어를 구현한다.
 * 워프 내에서 한 레인의 레지스터 값을 다른 레인으로 복사한다.
 * 모드: UP/DOWN/BFLY/IDX. c의 마스크 필드로 대상 레인 범위를 제한한다.
 * 활성(active) 소스 레인이 아니면 경고를 출력하고 0을 반환한다.
 * warp_info의 done_threads 카운터를 사용하여 워프 단위로 완료를 추적한다.
 * 실행 컨텍스트: 기능 시뮬레이션 — 워프 내 레인 단위.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [shfl_impl] → get_operand_value() → set_operand_value()
 */

void shfl_impl(const ptx_instruction *pI, core_t *core, warp_inst_t inst) {
  unsigned i_type = pI->get_type();
  int tid;

  if (core->get_gpu()->is_functional_sim())  // [한국어] 기능/타이밍 시뮬레이션 모드 분기
    tid = inst.warp_id_func() * core->get_warp_size();  // [한국어] 기능 시뮬레이션 워프 ID 사용
  else
    tid = inst.warp_id() * core->get_warp_size();  // [한국어] 타이밍 시뮬레이션 워프 ID 사용

  ptx_thread_info *thread = core->get_thread_info()[tid];
  ptx_warp_info *warp_info = thread->m_warp_info;
  int lane = warp_info->get_done_threads();  // [한국어] 현재 처리 중인 레인 인덱스
  thread = core->get_thread_info()[tid + lane];  // [한국어] 현재 레인 스레드 선택

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  int bval = (thread->get_operand_value(src2, dst, i_type, thread, 1)).u32;
  int cval = (thread->get_operand_value(src3, dst, i_type, thread, 1)).u32;
  int mask = cval >> 8;
  bval &= 0x1F;  // [한국어] 오프셋 하위 5비트 추출
  cval &= 0x1F;  // [한국어] 마스크 하위 5비트 추출

  int maxLane = (lane & mask) | (cval & ~mask);  // [한국어] 최대 레인 계산
  int minLane = lane & mask;  // [한국어] 최소 레인 계산

  int src_idx;
  unsigned p;
  switch (pI->shfl_op()) {  // [한국어] shfl 모드 분기
    case UP_OPTION:  // [한국어] 위쪽 레인으로 이동
      src_idx = lane - bval;
      p = (src_idx >= maxLane);
      break;
    case DOWN_OPTION:  // [한국어] 아래쪽 레인으로 이동
      src_idx = lane + bval;
      p = (src_idx <= maxLane);
      break;
    case BFLY_OPTION:  // [한국어] XOR 기반 버터플라이 셔플
      src_idx = lane ^ bval;
      p = (src_idx <= maxLane);
      break;
    case IDX_OPTION:  // [한국어] 지정 인덱스 셔플
      src_idx = minLane | (bval & ~mask);
      p = (src_idx <= maxLane);
      break;
    default:
      printf("GPGPU-Sim PTX: ERROR: Invalid shfl option\n");
      assert(0);
      break;
  }
  // copy from own lane
  if (!p) src_idx = lane;  // [한국어] 범위 밖이면 자기 레인 사용

  // copy input from lane src_idx
  ptx_reg_t data;
  if (inst.active(src_idx)) {  // [한국어] 활성 소스 레인 여부 확인
    ptx_thread_info *source = core->get_thread_info()[tid + src_idx];  // [한국어] 소스 레인 스레드 선택
    data = source->get_operand_value(src1, dst, i_type, source, 1);
  } else {
    printf(
        "GPGPU-Sim PTX: WARNING: shfl input value unpredictable for inactive "
        "threads in a warp\n");
    data.u32 = 0;  // [한국어] 비활성 레인이면 0 사용
  }
  thread->set_operand_value(dst, data, i_type, thread, pI);

  /*
  TODO: deal with predicates appropriately using the following pseudocode:
  if (!isGuardPredicateTrue(src_idx)) {
          printf("GPGPU-Sim PTX: WARNING: shfl input value unpredictable for
  predicated-off threads in a warp\n");
  }
  if (dest predicate selected) data.pred = p;
  */

  // keep track of the number of threads that have executed in the warp
  warp_info->inc_done_threads();  // [한국어] 워프 완료 카운터 증가
  if (warp_info->get_done_threads() == inst.active_count()) {  // [한국어] 모든 활성 레인 완료 시 리셋
    warp_info->reset_done_threads();
  }
}

/*
 * [한국어]
 * shf_impl - PTX `shf` 명령어 구현: 64비트 연결 시프트 (funnel shift)
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1(하위), src2(상위), src3(시프트량), .left/.right, .clamp 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `shf.l|r.clamp.b32 d, a, b, c` 명령어를 구현한다.
 * src2(상위 32비트)와 src1(하위 32비트)를 연결한 64비트 값을 c만큼 좌/우 시프트하여
 * 32비트 결과를 반환한다. .clamp 수식어에 따라 시프트량을 0~31로 클램핑한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [shf_impl] → set_operand_value()
 */

void shf_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, c, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  // Only b32 is allowed  // [한국어] shf는 B32 타입만 허용
  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  if (i_type != B32_TYPE)  // [한국어] 타입 검증
    printf("Only the b32 data_type is allowed per the ISA\n");

  unsigned clamp_mode = pI->clamp_mode();  // [한국어] clamp 모드 확인
  unsigned n = c.u32 & 0x1f;  // [한국어] 시프트량 하위 5비트
  if (clamp_mode) {  // [한국어] clamp 모드 시 시프트량 제한
    if (c.u32 < 32)
      n = c;
    else
      n = 32;
  }
  if (pI->left_mode())  // [한국어] 좌/우 시프트 분기
    d.u32 = (b.u32 << n) | (a.u32 >> (32 - n));  // [한국어] 좌측 funnel shift
  else
    d.u32 = (b.u32 << (32 - n)) | (a.u32 >> n);  // [한국어] 우측 funnel shift

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

/*
 * [한국어]
 * shl_impl - PTX `shl` 명령어 구현: 논리/부호 없는 좌시프트
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2(시프트량), 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `shl.type d, a, b` 명령어를 구현한다. d = a << b.
 * 시프트량이 타입 비트 수 이상이면 결과를 0으로 한다.
 * B16/B32/B64 및 U16/U32/U64 타입을 지원한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [shl_impl] → set_operand_value()
 */

void shl_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case B16_TYPE:
    case U16_TYPE:
      if (b.u16 >= 16)
        d.u16 = 0;
      else
        d.u16 = (unsigned short)((a.u16 << b.u16) & 0xFFFF);
      break;
    case B32_TYPE:
    case U32_TYPE:
      if (b.u32 >= 32)
        d.u32 = 0;
      else
        d.u32 = (unsigned)((a.u32 << b.u32) & 0xFFFFFFFF);
      break;
    case B64_TYPE:
    case U64_TYPE:
      if (b.u32 >= 64)
        d.u64 = 0;
      else
        d.u64 = (a.u64 << b.u64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

/*
 * [한국어]
 * shr_impl - PTX `shr` 명령어 구현: 우시프트
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2(시프트량), 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `shr.type d, a, b` 명령어를 구현한다.
 * unsigned/boolean 타입은 논리 우시프트(0 채움), signed 타입은 산술 우시프트(부호 비트 채움)를 수행한다.
 * 시프트량이 타입 비트 수 이상이면 signed는 부호에 따라 0/-1, unsigned는 0을 반환한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [shr_impl] → set_operand_value()
 */

void shr_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, b, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);

  switch (i_type) {
    case U16_TYPE:
    case B16_TYPE:
      if (b.u16 < 16)
        d.u16 = (unsigned short)((a.u16 >> b.u16) & 0xFFFF);
      else
        d.u16 = 0;
      break;
    case U32_TYPE:
    case B32_TYPE:
      if (b.u32 < 32)
        d.u32 = (unsigned)((a.u32 >> b.u32) & 0xFFFFFFFF);
      else
        d.u32 = 0;
      break;
    case U64_TYPE:
    case B64_TYPE:
      if (b.u32 < 64)
        d.u64 = (a.u64 >> b.u64);
      else
        d.u64 = 0;
      break;
    case S16_TYPE:
      if (b.u16 < 16)
        d.s64 = (a.s16 >> b.s16);
      else {
        if (a.s16 < 0) {
          d.s64 = -1;
        } else {
          d.s64 = 0;
        }
      }
      break;
    case S32_TYPE:
      if (b.u32 < 32)
        d.s64 = (a.s32 >> b.s32);
      else {
        if (a.s32 < 0) {
          d.s64 = -1;
        } else {
          d.s64 = 0;
        }
      }
      break;
    case S64_TYPE:
      if (b.u64 < 64)
        d.s64 = (a.s64 >> b.u64);
      else {
        if (a.s64 < 0) {
          if (b.s32 < 0) {
            d.u64 = -1;
            d.s32 = 0;
          } else {
            d.s64 = -1;
          }
        } else {
          d.s64 = 0;
        }
      }
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

/*
 * [한국어]
 * sin_impl - PTX `sin` 명령어 구현: 사인 근사
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, 소스, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `sin.approx.f32 d, a` 명령어를 구현한다. d = sin(a).
 * SFU 연산으로, 타이밍 레이턴시는 -ptx_opcode_latency_sfu 옵션으로 설정된다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [sin_impl] → set_operand_value()
 */

void sin_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      d.f32 = sin(a.f32);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

/*
 * [한국어]
 * slct_impl - PTX `slct` 명령어 구현: 부호/비교 기반 선택
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, src3(조건), 타입1/타입2 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `slct.type d, a, b, c` 명령어를 구현한다. c >= 0이면 d=a, 아니면 d=b.
 * 조건(c)의 타입은 S32 또는 F32이며, 결과 타입은 다양한 16/32/64비트 타입을 지원한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [slct_impl] → set_operand_value()
 */

void slct_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();

  ptx_reg_t a, b, c, d;

  unsigned i_type = pI->get_type();
  unsigned c_type = pI->get_type2();
  bool t = false;
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  c = thread->get_operand_value(src3, dst, c_type, thread, 1);

  switch (c_type) {
    case S32_TYPE:
      t = c.s32 >= 0;
      break;
    case F32_TYPE:
      t = c.f32 >= 0;
      break;
    default:
      assert(0);
  }

  switch (i_type) {
    case B16_TYPE:
    case S16_TYPE:
    case U16_TYPE:
      d.u16 = t ? a.u16 : b.u16;
      break;
    case F32_TYPE:
    case B32_TYPE:
    case S32_TYPE:
    case U32_TYPE:
      d.u32 = t ? a.u32 : b.u32;
      break;
    case F64_TYPE:
    case FF64_TYPE:
    case B64_TYPE:
    case S64_TYPE:
    case U64_TYPE:
      d.u64 = t ? a.u64 : b.u64;
      break;
    default:
      assert(0);
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

/*
 * [한국어]
 * sqrt_impl - PTX `sqrt` 명령어 구현: 제곱근
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, 소스, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `sqrt.approx.{f32|f64} d, a` 명령어를 구현한다. d = sqrt(a).
 * 음수 입력 시 NaN을 반환한다. SFU 연산이며 레이턴시는 -ptx_opcode_latency_sfu 옵션으로 설정된다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [sqrt_impl] → set_operand_value()
 */

void sqrt_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t a, d;
  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();

  unsigned i_type = pI->get_type();
  a = thread->get_operand_value(src1, dst, i_type, thread, 1);

  switch (i_type) {
    case F32_TYPE:
      if (a.f32 < 0)
        d.f32 = nanf("");
      else
        d.f32 = sqrt(a.f32);
      break;
    case F64_TYPE:
    case FF64_TYPE:
      if (a.f64 < 0)
        d.f64 = nan("");
      else
        d.f64 = sqrt(a.f64);
      break;
    default:
      printf("Execution error: type mismatch with instruction\n");
      assert(0);
      break;
  }

  thread->set_operand_value(dst, d, i_type, thread, pI);
}

/*
 * [한국어]
 * sst_impl - PTX `sst` (sparse store) 명령어 구현: 희소 배열 저장
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTXPlus 확장 명령어로 보이는 sst를 구현한다.
 * 스레드들의 값을 sstarr 공간에 모은 뒤 0이 아닌 항목과 인덱스를 글로벌 메모리에 기록하고,
 * 남은 공간을 0으로 채운다. CTA 내 모든 스레드가 참여하는 bar_id 16 동기화를 사용한다.
 * 실행 컨텍스트: 기능 시뮬레이션 — CTA 단위 협력적 동작.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [sst_impl] → decode_space() → mem->write/read()
 */

void sst_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_instruction *cpI = const_cast<ptx_instruction *>(pI);  // constant
  const operand_info &dst = cpI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();
  const operand_info &src3 = pI->src3();
  unsigned type = pI->get_type();
  ptx_reg_t dst_data = thread->get_operand_value(dst, dst, type, thread, 1);
  ptx_reg_t src1_data = thread->get_operand_value(src1, src1, type, thread, 1);
  ptx_reg_t src2_data = thread->get_operand_value(src2, src1, type, thread, 1);
  ptx_reg_t src3_data = thread->get_operand_value(src3, src1, type, thread, 1);
  memory_space_t space = pI->get_space();
  memory_space *mem = NULL;
  addr_t addr =
      src2_data.u32 * 4;  // this assumes sstarr memory starts at address 0
  ptx_cta_info *cta_info = thread->m_cta_info;

  decode_space(space, thread, src1, mem, addr);

  size_t size;
  int t;
  type_info_key::type_decode(type, size, t);

  // store data in sstarr memory
  mem->write(addr, size / 8, &src3_data.s64, thread, pI);

  // sync threads
  cpI->set_bar_id(16);  // use 16 for sst because bar uses an int from 0-15

  thread->m_last_effective_address = addr;
  thread->m_last_memory_space = space;
  thread->m_last_dram_callback.function = bar_callback;
  thread->m_last_dram_callback.instruction = cpI;

  // the last thread that executes loads all of the data back from sstarr memory
  int NUM_THREADS = cta_info->num_threads();
  cta_info->inc_bar_threads();
  if (NUM_THREADS == cta_info->get_bar_threads()) {
    unsigned offset = 0;
    addr = 0;
    ptx_reg_t data;
    float sstarr_fdata[NUM_THREADS];
    signed long long sstarr_ldata[NUM_THREADS];
    // loop through all of the threads
    for (int tid = 0; tid < NUM_THREADS; tid++) {
      data.u64 = 0;
      mem->read(addr + (tid * 4), size / 8, &data.s64);
      sstarr_fdata[tid] = data.f32;
      sstarr_ldata[tid] = data.s64;
    }

    // squeeze the zeros out of the array and store data back into original
    // array
    mem = NULL;
    addr = src1_data.u32;
    space.set_type(global_space);
    decode_space(space, thread, src1, mem, addr);
    // store nonzero entries and indices
    for (int tid = 0; tid < NUM_THREADS; tid++) {
      if (sstarr_fdata[tid] != 0) {
        float ftid = (float)tid;
        mem->write(addr + (offset * 4), size / 8, &sstarr_ldata[tid], thread,
                   pI);
        mem->write(addr + ((NUM_THREADS + offset) * 4), size / 8, &ftid, thread,
                   pI);
        offset++;
      }
    }
    // store the number of nonzero elements in the array
    data = thread->get_operand_value(src1, dst, type, thread, 1);
    data.s64 += 4 * (offset - 1);
    thread->set_operand_value(dst, data, type, thread, pI);

    // fill the rest of the array with zeros (dst should always have a 0 in it)
    while (offset < NUM_THREADS) {
      mem->write(addr + (offset * 4), size / 8, &dst_data.s64, thread, pI);
      offset++;
    }

    cta_info->reset_bar_threads();
    thread->m_last_effective_address = addr + (NUM_THREADS - 1) * 4;
    thread->m_last_memory_space = space;
  }
}

/*
 * [한국어]
 * ssy_impl - PTX `ssy` 명령어 구현 (TODO)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `ssy`는 동기화/재합류 지점을 표시하는 명령어이다.
 * 현재 GPGPU-Sim에서는 no-op으로 처리되며 TODO 주석이 남아 있다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 */

void ssy_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // printf("Execution Warning: unimplemented ssy instruction is treated as a
  // nop\n");
  // TODO: add implementation
}

/*
 * [한국어]
 * st_impl - PTX `st` 명령어 구현: 메모리에 저장
 *
 * @pI: 실행 중인 PTX 명령어 (목적지(주소), src1(데이터), 메모리 공간, 타입, 벡터 수식어 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `st.{space}.{type} [a], b` 명령어를 구현한다.
 * decode_space()로 메모리 공간과 실제 HW 주소를 해석한 뒤 mem->write()로 데이터를 기록한다.
 * 벡터(V2/V3/V4) 저장 시 연속 주소에 각 요소를 기록한다.
 * m_last_effective_address와 m_last_memory_space를 갱신하여 타이밍 모델이 메모리 접근을 추적한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [st_impl] → decode_space() → mem->write()
 */

void st_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  const operand_info &dst = pI->dst();  // [한국어] 저장 대상 주소 오퍼랜드
  const operand_info &src1 = pI->src1();  // may be scalar or vector of regs  // [한국어] 저장할 데이터 오퍼랜드
  unsigned type = pI->get_type();
  ptx_reg_t addr_reg = thread->get_operand_value(dst, dst, type, thread, 1);
  ptx_reg_t data;
  memory_space_t space = pI->get_space();
  unsigned vector_spec = pI->get_vector();

  memory_space *mem = NULL;
  addr_t addr = addr_reg.u32;

  decode_space(space, thread, dst, mem, addr);  // [한국어] 메모리 공간/주소 해석

  size_t size;
  int t;
  type_info_key::type_decode(type, size, t);

  if (!vector_spec) {
    data = thread->get_operand_value(src1, dst, type, thread, 1);  // [한국어] 스칼라 데이터 읽기
    mem->write(addr, size / 8, &data.s64, thread, pI);  // [한국어] 메모리에 스칼라 기록
  } else {
    if (vector_spec == V2_TYPE) {
      ptx_reg_t *ptx_regs = new ptx_reg_t[2];
      thread->get_vector_operand_values(src1, ptx_regs, 2);  // [한국어] V2 벡터 데이터 읽기
      mem->write(addr, size / 8, &ptx_regs[0].s64, thread, pI);
      mem->write(addr + size / 8, size / 8, &ptx_regs[1].s64, thread, pI);
      delete[] ptx_regs;
    }
    if (vector_spec == V3_TYPE) {
      ptx_reg_t *ptx_regs = new ptx_reg_t[3];
      thread->get_vector_operand_values(src1, ptx_regs, 3);
      mem->write(addr, size / 8, &ptx_regs[0].s64, thread, pI);
      mem->write(addr + size / 8, size / 8, &ptx_regs[1].s64, thread, pI);
      mem->write(addr + 2 * size / 8, size / 8, &ptx_regs[2].s64, thread, pI);
      delete[] ptx_regs;
    }
    if (vector_spec == V4_TYPE) {
      ptx_reg_t *ptx_regs = new ptx_reg_t[4];
      thread->get_vector_operand_values(src1, ptx_regs, 4);
      mem->write(addr, size / 8, &ptx_regs[0].s64, thread, pI);
      mem->write(addr + size / 8, size / 8, &ptx_regs[1].s64, thread, pI);
      mem->write(addr + 2 * size / 8, size / 8, &ptx_regs[2].s64, thread, pI);
      mem->write(addr + 3 * size / 8, size / 8, &ptx_regs[3].s64, thread, pI);
      delete[] ptx_regs;
    }
  }
  thread->m_last_effective_address = addr;  // [한국어] 타이밍 모델용 효과 주소 기록
  thread->m_last_memory_space = space;  // [한국어] 타이밍 모델용 메모리 공간 기록
}

/*
 * [한국어]
 * sub_impl - PTX `sub` 명령어 구현: 뺄셈
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `sub.type d, a, b` 명령어를 구현한다. d = a - b.
 * 정수 타입은 캐리/오버플로우 플래그를 계산하여 CC 레지스터 업데이트에 사용한다.
 * 뺄셈은 2의 보수 덧셈으로 구현되며, 캐리 비트가 올바르게 설정되도록 상수(2^n)를 더한다.
 * 부동소수 타입은 단순히 float 값을 뺀다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [sub_impl] → set_operand_value(overflow, carry)
 */

void sub_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t data;
  int overflow = 0;  // [한국어] 오버플로우 플래그 초기화
  int carry = 0;  // [한국어] 캐리 플래그 초기화

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  ptx_reg_t src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  ptx_reg_t src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // performs addition. Sets carry and overflow if needed.
  // the constant is added in during subtraction so the carry bit is set
  // properly.
  switch (i_type) {
    case S8_TYPE:
      data.s64 = (src1_data.s64 & 0xFF) - (src2_data.s64 & 0xFF) + 0x100;  // [한국어] S8 2의 보수 뺄셈
      if (((src1_data.s64 & 0x80) - (src2_data.s64 & 0x80)) != 0) {
        overflow = ((src1_data.s64 & 0x80) - (data.s64 & 0x80)) == 0 ? 0 : 1;  // [한국어] 부호 오버플로우 검출
      }
      carry = (data.s32 & 0x100) >> 8;  // [한국어] S8 캐리 추출
      break;
    case S16_TYPE:
      data.s64 = (src1_data.s64 & 0xFFFF) - (src2_data.s64 & 0xFFFF) + 0x10000;
      if (((src1_data.s64 & 0x8000) - (src2_data.s64 & 0x8000)) != 0) {
        overflow =
            ((src1_data.s64 & 0x8000) - (data.s64 & 0x8000)) == 0 ? 0 : 1;
      }
      carry = (data.s32 & 0x10000) >> 16;
      break;
    case S32_TYPE:
      data.s64 = (src1_data.s64 & 0xFFFFFFFF) - (src2_data.s64 & 0xFFFFFFFF) +
                 0x100000000;
      if (((src1_data.s64 & 0x80000000) - (src2_data.s64 & 0x80000000)) != 0) {
        overflow = ((src1_data.s64 & 0x80000000) - (data.s64 & 0x80000000)) == 0
                       ? 0
                       : 1;
      }
      carry = ((data.u64) >> 32) & 0x0001;  // [한국어] S32 캐리 추출
      break;
    case S64_TYPE:
      data.s64 = src1_data.s64 - src2_data.s64;
      break;
    case B8_TYPE:
    case U8_TYPE:
      data.u64 = (src1_data.u64 & 0xFF) - (src2_data.u64 & 0xFF) + 0x100;
      carry = (data.u64 & 0x100) >> 8;
      break;
    case B16_TYPE:
    case U16_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFF) - (src2_data.u64 & 0xFFFF) + 0x10000;
      carry = (data.u64 & 0x10000) >> 16;
      break;
    case B32_TYPE:
    case U32_TYPE:
      data.u64 = (src1_data.u64 & 0xFFFFFFFF) - (src2_data.u64 & 0xFFFFFFFF) +
                 0x100000000;
      carry = (data.u64 & 0x100000000) >> 32;
      break;
    case B64_TYPE:
    case U64_TYPE:
      data.u64 = src1_data.u64 - src2_data.u64;
      break;
    case F16_TYPE:
      data.f16 = src1_data.f16 - src2_data.f16;
      break;  // assert(0); break;
    case F32_TYPE:
      data.f32 = src1_data.f32 - src2_data.f32;  // [한국어] F32 뺄셈
      break;
    case F64_TYPE:
    case FF64_TYPE:
      data.f64 = src1_data.f64 - src2_data.f64;
      break;
    default:
      assert(0);
      break;
  }

  thread->set_operand_value(dst, data, i_type, thread, pI, overflow, carry);  // [한국어] 결과 및 플래그 기록
}

/*
 * [한국어]
 * nop_impl - PTX `nop` 명령어 구현: 아무 동작 안 함
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `nop`은 no-operation이다. 파이프라인에서 한 슬롯을 소비하지만 기능적으로는 아무것도 하지 않는다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 */

void nop_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // Do nothing
}

/*
 * [한국어]
 * subc_impl - PTX `subc` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `subc`는 캐리를 고려한 뺄셈 명령어이다.
 * 현재 GPGPU-Sim에서는 구현되어 있지 않아 inst_not_implemented()를 호출한다.
 */

void subc_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * suld_impl - PTX `suld` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `suld`는 서피스 메모리 로드 명령어이다.
 * 현재 GPGPU-Sim에서는 구현되어 있지 않아 inst_not_implemented()를 호출한다.
 */

void suld_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * sured_impl - PTX `sured` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `sured`는 서피스 메모리 원자적 축소 명령어이다.
 * 현재 GPGPU-Sim에서는 구현되어 있지 않아 inst_not_implemented()를 호출한다.
 */

void sured_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * sust_impl - PTX `sust` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `sust`는 서피스 메모리 저장 명령어이다.
 * 현재 GPGPU-Sim에서는 구현되어 있지 않아 inst_not_implemented()를 호출한다.
 */

void sust_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * suq_impl - PTX `suq` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `suq`는 서피스 메모리 쿼리 명령어이다.
 * 현재 GPGPU-Sim에서는 구현되어 있지 않아 inst_not_implemented()를 호출한다.
 */

void suq_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

union intfloat {
  int a;
  float b;
};

/*
 * [한국어]
 * reduce_precision - float 가수부(mantissa) 정밀도를 줄이는 텍스처 헬퍼
 *
 * @x: 원본 float 값
 * @bits: 남길 가수부 하위 비트 수
 * @return: 정밀도가 감소된 float 값
 *
 * 텍스처 선형 보간(linear sampling)에서 사용되는 NVIDIA HW 특화 근사값을 에뮬레이션한다.
 * 지정된 비트 수만큼 가수부 하위 비트를 마스크하여 근사한다.
 * 실행 컨텍스트: tex_linf_sampling 낶부.
 */

float reduce_precision(float x, unsigned bits) {
  intfloat tmp;
  tmp.b = x;
  int v = tmp.a;
  int man = v & ((1 << 23) - 1);
  int mask = ((1 << bits) - 1) << (23 - bits);
  int nv = (v & ((-1) - ((1 << 23) - 1))) | (mask & man);
  tmp.a = nv;
  float result = tmp.b;
  return result;
}

/*
 * [한국어]
 * wrap - 텍스처 좌표 래핑(wrap) 주소 계산
 *
 * @x, @y: 텍셀 좌표
 * @mx, @my: 텍스처 너비/높이
 * @elem_size: 요소 크기(바이트)
 * @return: 1D 선형 배열 인덱스
 *
 * 좌표가 경계를 넘어가면 modulo 연산으로 반대편으로 돌아간다.
 * 실행 컨텍스트: tex_linf_sampling 낶부.
 */

unsigned wrap(unsigned x, unsigned y, unsigned mx, unsigned my,
              size_t elem_size) {
  unsigned nx = (mx + x) % mx;
  unsigned ny = (my + y) % my;
  return nx + mx * ny;
}

/*
 * [한국어]
 * clamp - 텍스처 좌표 클램핑(clamp) 주소 계산
 *
 * @x, @y: 텍셀 좌표
 * @mx, @my: 텍스처 너비/높이
 * @elem_size: 요소 크기(바이트)
 * @return: 1D 선형 배열 인덱스
 *
 * 좌표가 경계를 넘어가면 가장 가까운 경계 값으로 고정한다.
 * 실행 컨텍스트: tex_linf_sampling 낶부.
 */

unsigned clamp(unsigned x, unsigned y, unsigned mx, unsigned my,
               size_t elem_size) {
  unsigned nx = x;
  while (nx >= mx) nx -= elem_size;
  unsigned ny = (y >= my) ? my - 1 : y;
  return nx + mx * ny;
}

typedef unsigned (*texAddr_t)(unsigned x, unsigned y, unsigned mx, unsigned my,
                              size_t elem_size);

/*
 * [한국어]
 * tex_linf_sampling - 2x2 텍셀 선형 보간 샘플링
 *
 * @mem: 글로벌 메모리 공간
 * @tex_array_base: 텍스처 배열 기준 주소
 * @x, @y: 좌표
 * @width, @height: 텍스처 크기
 * @elem_size: 요소 크기(바이트)
 * @alpha, @beta: x/y 방향 보간 가중치
 * @b_lim: 경계 처리 함수(wrap/clamp)
 * @return: 보간된 float 샘플 값
 *
 * 2D 텍스처의 선형 필터링을 에뮬레이션한다. 인접한 4개 텍셀을 읽어
 * bilinear interpolation 공식으로 합성한다.
 * 실행 컨텍스트: tex_impl 낶부 (F32 linear 필터 모드).
 */

float tex_linf_sampling(memory_space *mem, unsigned tex_array_base, int x,
                        int y, unsigned int width, unsigned int height,
                        size_t elem_size, float alpha, float beta,
                        texAddr_t b_lim) {
  float Tij;
  float Ti1j;
  float Tij1;
  float Ti1j1;

  mem->read(tex_array_base + b_lim(x, y, width, height, elem_size), 4, &Tij);
  mem->read(tex_array_base + b_lim(x + elem_size, y, width, height, elem_size),
            4, &Ti1j);
  mem->read(tex_array_base + b_lim(x, y + 1, width, height, elem_size), 4,
            &Tij1);
  mem->read(
      tex_array_base + b_lim(x + elem_size, y + 1, width, height, elem_size), 4,
      &Ti1j1);

  float sample = (1 - alpha) * (1 - beta) * Tij + alpha * (1 - beta) * Ti1j +
                 (1 - alpha) * beta * Tij1 + alpha * beta * Ti1j1;

  return sample;
}

/*
 * [한국어]
 * textureNormalizeElementSigned - signed 정수 텍스처 요소를 [-1,1]로 정규화
 *
 * @element: 원본 정수 값
 * @bits: 채널 비트 수
 * @return: [-1.0, 1.0] 범위의 float
 *
 * cudaReadModeNormalizedFloat 모드에서 signed 정수 채널을 실수로 변환할 때 사용한다.
 * 실행 컨텍스트: textureNormalizeOutput 낶부.
 */

float textureNormalizeElementSigned(int element, int bits) {
  if (bits) {
    int maxN = (1 << bits) - 1;
    // removing upper bits
    element &= maxN;
    // normalizing the number to [-1.0,1.0]
    maxN >>= 1;
    float output = (float)element / maxN;
    if (output < -1.0f) output = -1.0f;
    return output;
  } else {
    return 0.0f;
  }
}

/*
 * [한국어]
 * textureNormalizeElementUnsigned - unsigned 정수 텍스처 요소를 [0,1]로 정규화
 *
 * @element: 원본 정수 값
 * @bits: 채널 비트 수
 * @return: [0.0, 1.0] 범위의 float
 *
 * cudaReadModeNormalizedFloat 모드에서 unsigned 정수 채널을 실수로 변환할 때 사용한다.
 * 실행 컨텍스트: textureNormalizeOutput 낶부.
 */

float textureNormalizeElementUnsigned(unsigned int element, int bits) {
  if (bits) {
    unsigned int maxN = (1 << bits) - 1;
    // removing upper bits and normalizing the number to [0.0,1.0]
    return (float)(element & maxN) / maxN;
  } else {
    return 0.0f;
  }
}

/*
 * [한국어]
 * textureNormalizeOutput - 텍스처 출력 4채널 정규화
 *
 * @desc: cudaArray 채널 포맷 설명자
 * @datax, @datay, @dataz, @dataw: in/out 텍스처 요소 값
 *
 * 채널 포맷 종류(signed/unsigned)에 따라 4개 채널을 모두 정규화한다.
 * cudaReadModeNormalizedFloat 모드에서 tex_impl 마지막에 호출된다.
 * 실행 컨텍스트: tex_impl 낶부.
 */

void textureNormalizeOutput(const struct cudaChannelFormatDesc &desc,
                            ptx_reg_t &datax, ptx_reg_t &datay,
                            ptx_reg_t &dataz, ptx_reg_t &dataw) {
  if (desc.f == cudaChannelFormatKindSigned) {
    datax.f32 = textureNormalizeElementSigned(datax.s32, desc.x);
    datay.f32 = textureNormalizeElementSigned(datay.s32, desc.y);
    dataz.f32 = textureNormalizeElementSigned(dataz.s32, desc.z);
    dataw.f32 = textureNormalizeElementSigned(dataw.s32, desc.w);
  } else if (desc.f == cudaChannelFormatKindUnsigned) {
    datax.f32 = textureNormalizeElementUnsigned(datax.u32, desc.x);
    datay.f32 = textureNormalizeElementUnsigned(datay.u32, desc.y);
    dataz.f32 = textureNormalizeElementUnsigned(dataz.u32, desc.z);
    dataw.f32 = textureNormalizeElementUnsigned(dataw.u32, desc.w);
  } else {
    assert(0 &&
           "Undefined texture read mode: cudaReadModeNormalizedFloat expect "
           "integer elements");
  }
}

/*
 * [한국어]
 * tex_impl - PTX `tex` 명령어 구현: 텍스처 샘플링
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, 텍스처 이름, 좌표, 차원, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `tex.{geom}.sampled_type d, sampler, coords` 명령어를 구현한다.
 * 1D/2D 텍스처에 대해 좌표 변환(normalized/un-normalized), 주소 모드(clamp/wrap),
 * 필터 모드(point/linear)를 처리하여 텍셀 데이터를 읽는다.
 * 마지막으로 m_last_effective_address와 m_last_memory_space를 기록하여
 * 타이밍 모델의 L1T/L2 텍스처 캐시 접근을 지원한다.
 * 텍스처 캐시 라인 크기는 gpgpusim.config의 -gpgpu_texcache_linesize 등으로 설정된다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [tex_impl] → tex_linf_sampling() → set_vector_operand_values()
 */

void tex_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
#if (CUDART_VERSION <= 1200)
  unsigned dimension = pI->dimension();
  const operand_info &dst =
      pI->dst();  // the registers to which fetched texel will be placed
  const operand_info &src1 = pI->src1();  // the name of the texture
  const operand_info &src2 =
      pI->src2();  // the vector registers containing coordinates of the texel
                   // to be fetched

  std::string texname = src1.name();
  // If indirect access, use register's value as address
  // to find the symbol
  if (src1.is_reg()) {
    ptx_reg_t src1_data =
        thread->get_operand_value(src1, dst, pI->get_type(), thread, 1);
    addr_t sym_addr = src1_data.u64;
    symbol *texRef = thread->get_symbol_table()->lookup_by_addr(sym_addr);
    assert(texRef != NULL);
    texname = texRef->name();
  }

  unsigned to_type = pI->get_type();
  unsigned c_type = pI->get_type2();
  fflush(stdout);
  ptx_reg_t data1, data2, data3, data4;
  if (!thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs)
    thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs = new ptx_reg_t[4];
  unsigned nelem = src2.get_vect_nelem();
  thread->get_vector_operand_values(
      src2, thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs,
      nelem);  // ptx_reg should be 4 entry vector type...coordinates into
               // texture
  /*
    For programs with many streams, textures can be bound and unbound
    asynchronously.  This means we need to use the kernel's "snapshot" of
    the state of the texture mappings when it was launched (so that we
    don't try to access the incorrect texture mapping if it's been updated,
    or that we don't access a mapping that has been unbound).
  */
  gpgpu_t *gpu = thread->get_gpu();
  kernel_info_t &k = thread->get_kernel();
  const struct textureReference *texref = gpu->get_texref(texname);
  const struct cudaArray *cuArray = k.get_texarray(texname);
  const struct textureInfo *texInfo = k.get_texinfo(texname);
  const struct textureReferenceAttr *texAttr = gpu->get_texattr(texname);

  // assume always 2D f32 input
  // access array with src2 coordinates
  memory_space *mem = thread->get_global_memory();
  float x_f32, y_f32;
  size_t size;
  int t;
  unsigned tex_array_base;
  unsigned int width = 0, height = 0;
  int x = 0;
  int y = 0;
  unsigned tex_array_index;
  float alpha = 0, beta = 0;

  type_info_key::type_decode(to_type, size, t);
  tex_array_base = cuArray->devPtr32;

  switch (dimension) {
    case GEOM_MODIFIER_1D:
      width = cuArray->width;
      height = cuArray->height;
      if (texref->normalized) {
        assert(c_type == F32_TYPE);
        x_f32 = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].f32;
        if (texref->addressMode[0] == cudaAddressModeClamp) {
          x_f32 = (x_f32 > 1.0) ? 1.0 : x_f32;
          x_f32 = (x_f32 < 0.0) ? 0.0 : x_f32;
        } else if (texref->addressMode[0] == cudaAddressModeWrap) {
          x_f32 = x_f32 - floor(x_f32);
        }

        if (texref->filterMode == cudaFilterModeLinear) {
          float xb = x_f32 * width - 0.5;
          alpha = xb - floor(xb);
          alpha = reduce_precision(alpha, 9);
          beta = 0.0;

          x = (int)floor(xb);
          y = 0;
        } else {
          x = (int)floor(x_f32 * width);
          y = 0;
        }
      } else {
        switch (c_type) {
          case S32_TYPE:
            x = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].s32;
            assert(texref->filterMode == cudaFilterModePoint);
            break;
          case F32_TYPE:
            x_f32 = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].f32;
            alpha = x_f32 -
                    floor(x_f32);  // offset into subtexel (for linear sampling)
            x = (int)x_f32;
            break;
          default:
            assert(0 && "Unsupported texture coordinate type.");
        }
        // handle texture fetch that exceeded boundaries
        if (texref->addressMode[0] == cudaAddressModeClamp) {
          x = (x > width - 1) ? (width - 1) : x;
          x = (x < 0) ? 0 : x;
        } else if (texref->addressMode[0] == cudaAddressModeWrap) {
          x = x % width;
        }
      }
      width *= (cuArray->desc.w + cuArray->desc.x + cuArray->desc.y +
                cuArray->desc.z) /
               8;
      x *= (cuArray->desc.w + cuArray->desc.x + cuArray->desc.y +
            cuArray->desc.z) /
           8;
      tex_array_index = tex_array_base + x;

      break;
    case GEOM_MODIFIER_2D:
      width = cuArray->width;
      height = cuArray->height;
      if (texref->normalized) {
        x_f32 = reduce_precision(
            thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].f32, 16);
        y_f32 = reduce_precision(
            thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[1].f32, 15);

        if (texref->addressMode[0]) {  // clamp
          if (x_f32 < 0) x_f32 = 0;
          if (x_f32 >= 1) x_f32 = 1 - 1 / x_f32;
        } else {  // wrap
          x_f32 = x_f32 - floor(x_f32);
        }
        if (texref->addressMode[1]) {  // clamp
          if (y_f32 < 0) y_f32 = 0;
          if (y_f32 >= 1) y_f32 = 1 - 1 / y_f32;
        } else {  // wrap
          y_f32 = y_f32 - floor(y_f32);
        }

        if (texref->filterMode == cudaFilterModeLinear) {
          float xb = x_f32 * width - 0.5;
          float yb = y_f32 * height - 0.5;
          alpha = xb - floor(xb);
          beta = yb - floor(yb);
          alpha = reduce_precision(alpha, 9);
          beta = reduce_precision(beta, 9);

          x = (int)floor(xb);
          y = (int)floor(yb);
        } else {
          x = (int)floor(x_f32 * width);
          y = (int)floor(y_f32 * height);
        }
      } else {
        x_f32 = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[0].f32;
        y_f32 = thread->get_gpu()->gpgpu_ctx->func_sim->ptx_tex_regs[1].f32;

        alpha = x_f32 - floor(x_f32);
        beta = y_f32 - floor(y_f32);

        x = (int)x_f32;
        y = (int)y_f32;
        if (texref->addressMode[0]) {  // clamp
          if (x < 0) x = 0;
          if (x >= (int)width) x = width - 1;
        } else {  // wrap
          x = x % width;
          if (x < 0) x *= -1;
        }
        if (texref->addressMode[1]) {  // clamp
          if (y < 0) y = 0;
          if (y >= (int)height) y = height - 1;
        } else {  // wrap
          y = y % height;
          if (y < 0) y *= -1;
        }
      }

      width *= (cuArray->desc.w + cuArray->desc.x + cuArray->desc.y +
                cuArray->desc.z) /
               8;
      x *= (cuArray->desc.w + cuArray->desc.x + cuArray->desc.y +
            cuArray->desc.z) /
           8;
      tex_array_index = tex_array_base + (x + width * y);
      break;
    default:
      assert(0);
      break;
  }
  switch (to_type) {
    case U8_TYPE:
    case U16_TYPE:
    case U32_TYPE:
    case B8_TYPE:
    case B16_TYPE:
    case B32_TYPE:
    case S8_TYPE:
    case S16_TYPE:
    case S32_TYPE: {
      unsigned long long elementOffset = 0;  // offset into the next element
      mem->read(tex_array_index, cuArray->desc.x / 8, &data1.u32);
      elementOffset += cuArray->desc.x / 8;
      if (cuArray->desc.y) {
        mem->read(tex_array_index + elementOffset, cuArray->desc.y / 8,
                  &data2.u32);
        elementOffset += cuArray->desc.y / 8;
        if (cuArray->desc.z) {
          mem->read(tex_array_index + elementOffset, cuArray->desc.z / 8,
                    &data3.u32);
          elementOffset += cuArray->desc.z / 8;
          if (cuArray->desc.w)
            mem->read(tex_array_index + elementOffset, cuArray->desc.w / 8,
                      &data4.u32);
        }
      }
      break;
    }
    case B64_TYPE:
    case U64_TYPE:
    case S64_TYPE:
      mem->read(tex_array_index, 8, &data1.u64);
      if (cuArray->desc.y) {
        mem->read(tex_array_index + 8, 8, &data2.u64);
        if (cuArray->desc.z) {
          mem->read(tex_array_index + 16, 8, &data3.u64);
          if (cuArray->desc.w) mem->read(tex_array_index + 24, 8, &data4.u64);
        }
      }
      break;
    case F16_TYPE:
      assert(0);
      break;
    case F32_TYPE: {
      if (texref->filterMode == cudaFilterModeLinear) {
        texAddr_t b_lim = wrap;
        if (texref->addressMode[0] == cudaAddressModeClamp) {
          b_lim = clamp;
        }
        size_t elem_size = (cuArray->desc.x + cuArray->desc.y +
                            cuArray->desc.z + cuArray->desc.w) /
                           8;
        size_t elem_ofst = 0;

        data1.f32 =
            tex_linf_sampling(mem, tex_array_base, x + elem_ofst, y, width,
                              height, elem_size, alpha, beta, b_lim);
        elem_ofst += cuArray->desc.x / 8;
        if (cuArray->desc.y) {
          data2.f32 =
              tex_linf_sampling(mem, tex_array_base, x + elem_ofst, y, width,
                                height, elem_size, alpha, beta, b_lim);
          elem_ofst += cuArray->desc.y / 8;
          if (cuArray->desc.z) {
            data3.f32 =
                tex_linf_sampling(mem, tex_array_base, x + elem_ofst, y, width,
                                  height, elem_size, alpha, beta, b_lim);
            elem_ofst += cuArray->desc.z / 8;
            if (cuArray->desc.w)
              data4.f32 = tex_linf_sampling(mem, tex_array_base, x + elem_ofst,
                                            y, width, height, elem_size, alpha,
                                            beta, b_lim);
          }
        }
      } else {
        mem->read(tex_array_index, cuArray->desc.x / 8, &data1.f32);
        if (cuArray->desc.y) {
          mem->read(tex_array_index + 4, cuArray->desc.y / 8, &data2.f32);
          if (cuArray->desc.z) {
            mem->read(tex_array_index + 8, cuArray->desc.z / 8, &data3.f32);
            if (cuArray->desc.w)
              mem->read(tex_array_index + 12, cuArray->desc.w / 8, &data4.f32);
          }
        }
      }
    } break;
    case F64_TYPE:
    case FF64_TYPE:
      mem->read(tex_array_index, 8, &data1.f64);
      if (cuArray->desc.y) {
        mem->read(tex_array_index + 8, 8, &data2.f64);
        if (cuArray->desc.z) {
          mem->read(tex_array_index + 16, 8, &data3.f64);
          if (cuArray->desc.w) mem->read(tex_array_index + 24, 8, &data4.f64);
        }
      }
      break;
    default:
      assert(0);
      break;
  }
  int x_block_coord, y_block_coord, memreqindex, blockoffset;

  switch (dimension) {
    case GEOM_MODIFIER_1D:
      thread->m_last_effective_address = tex_array_index;
      break;
    case GEOM_MODIFIER_2D:
      x_block_coord = x >> (texInfo->Tx_numbits + texInfo->texel_size_numbits);
      y_block_coord = y >> texInfo->Ty_numbits;

      memreqindex =
          ((y_block_coord * cuArray->width / texInfo->Tx) + x_block_coord) << 6;

      blockoffset = (x % (texInfo->Tx * texInfo->texel_size) +
                     (y % (texInfo->Ty)
                      << (texInfo->Tx_numbits + texInfo->texel_size_numbits)));
      memreqindex += blockoffset;
      thread->m_last_effective_address =
          tex_array_base + memreqindex;  // tex_array_index;
      break;
    default:
      assert(0);
  }
  thread->m_last_memory_space = tex_space;

  // normalize output into floating point numbers according to the texture read
  // mode
  if (texAttr->m_readmode == cudaReadModeNormalizedFloat) {
    textureNormalizeOutput(cuArray->desc, data1, data2, data3, data4);
  } else {
    assert(texAttr->m_readmode == cudaReadModeElementType);
  }

  thread->set_vector_operand_values(dst, data1, data2, data3, data4);
#endif
}

/*
 * [한국어]
 * txq_impl - PTX `txq` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `txq`는 텍스처/서피스 속성 쿼리 명령어이다.
 * 현재 GPGPU-Sim에서는 구현되어 있지 않아 inst_not_implemented()를 호출한다.
 */

void txq_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * trap_impl - PTX `trap` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `trap`은 디버거 트랩 명령어이다.
 * 현재 GPGPU-Sim에서는 구현되어 있지 않아 inst_not_implemented()를 호출한다.
 */

void trap_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * vabsdiff_impl - PTX `vabsdiff` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX 비디오 명령어 vabsdiff는 현재 구현되어 있지 않다.
 */

void vabsdiff_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * vadd_impl - PTX `vadd` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX 비디오 명령어 vadd는 현재 구현되어 있지 않다.
 */

void vadd_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * vmad_impl - PTX `vmad` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX 비디오 명령어 vmad는 현재 구현되어 있지 않다.
 */

void vmad_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

#define VMAX 0
#define VMIN 1

/*
 * [한국어]
 * vmax_impl - PTX `vmax` 명령어 구현: 비디오 최대값
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `vmax`는 비디오 메모리 연산용 최대값 명령어로,
 * video_mem_instruction()에 VMAX 연산 코드를 넘겨 처리한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 */

void vmax_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  video_mem_instruction(pI, thread, VMAX);
}

/*
 * [한국어]
 * vmin_impl - PTX `vmin` 명령어 구현: 비디오 최소값
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `vmin`은 비디오 메모리 연산용 최소값 명령어로,
 * video_mem_instruction()에 VMIN 연산 코드를 넘겨 처리한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 */

void vmin_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  video_mem_instruction(pI, thread, VMIN);
}

/*
 * [한국어]
 * vset_impl - PTX `vset` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX 비디오 명령어 vset은 현재 구현되어 있지 않다.
 */

void vset_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * vshl_impl - PTX `vshl` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX 비디오 명령어 vshl은 현재 구현되어 있지 않다.
 */

void vshl_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * vshr_impl - PTX `vshr` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX 비디오 명령어 vshr은 현재 구현되어 있지 않다.
 */

void vshr_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * vsub_impl - PTX `vsub` 명령어 구현 (미구현)
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX 비디오 명령어 vsub는 현재 구현되어 있지 않다.
 */

void vsub_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  inst_not_implemented(pI);
}

/*
 * [한국어]
 * vote_impl - PTX `vote` 명령어 구현: 워프 내 집계 투표
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, 소스 predicate, 투표 모드 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `vote.mode.pred d, a` / `vote.ballot.b32 d, a` 명령어를 구현한다.
 * 워프 내 모든 스레드의 predicate를 집계하여 any/all/uni/ballot 결과를 생성한다.
 * 마지막 스레드(last_tid)가 결과를 워프 전체에 기록한다.
 * PTXPlus zero-flag 규칙에 따라 predicate 값을 반전하여 평가한다.
 * 실행 컨텍스트: 기능 시뮬레이션 — 워프 단위 협력적 동작.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [vote_impl] → set_operand_value()
 */

void vote_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  static bool first_in_warp = true;  // [한국어] 워프 첫 실행 플래그
  static bool and_all;  // [한국어] 전원 true 집계 변수
  static bool or_all;  // [한국어] 한 명 이상 true 집계 변수
  static unsigned int ballot_result;  // [한국어] ballot 비트 마스크
  static std::list<ptx_thread_info *> threads_in_warp;
  static unsigned last_tid;

  if (first_in_warp) {  // [한국어] 워프 첫 스레드 초기화
    first_in_warp = false;
    threads_in_warp.clear();
    and_all = true;
    or_all = false;
    ballot_result = 0;
    int offset = 31;  // [한국어] 활성 스레드 최대 레인 탐색
    while ((offset >= 0) && !pI->active(offset)) offset--;
    assert(offset >= 0);
    last_tid =
        (thread->get_hw_tid() - (thread->get_hw_tid() % pI->warp_size())) +
        offset;
  }

  ptx_reg_t src1_data;
  const operand_info &src1 = pI->src1();
  src1_data = thread->get_operand_value(src1, pI->dst(), PRED_TYPE, thread, 1);

  // predicate value was changed so the lowest bit being set means the zero flag
  // is set. As a result, the value of src1_data.pred must be inverted to get
  // proper behavior
  bool pred_value = !(src1_data.pred & 0x0001);  // [한국어] predicate 반전
  bool invert = src1.is_neg_pred();  // [한국어] negate predicate 여부

  threads_in_warp.push_back(thread);
  and_all &= (invert ^ pred_value);  // [한국어] 전원 true 업데이트
  or_all |= (invert ^ pred_value);  // [한국어] any true 업데이트

  // vote.ballot
  if (invert ^ pred_value) {
    int lane_id = thread->get_hw_tid() % pI->warp_size();
    ballot_result |= (1 << lane_id);  // [한국어] ballot 마스크 비트 설정
  }

  if (thread->get_hw_tid() == last_tid) {  // [한국어] 마지막 스레드가 결과 분배
    if (pI->vote_mode() == ptx_instruction::vote_ballot) {
      ptx_reg_t data = ballot_result;
      for (std::list<ptx_thread_info *>::iterator t = threads_in_warp.begin();
           t != threads_in_warp.end(); ++t) {
        const operand_info &dst = pI->dst();
        (*t)->set_operand_value(dst, data, pI->get_type(), (*t), pI);
      }
    } else {
      bool pred_value = false;

      switch (pI->vote_mode()) {
        case ptx_instruction::vote_any:  // [한국어] any 모드
          pred_value = or_all;
          break;
        case ptx_instruction::vote_all:  // [한국어] all 모드
          pred_value = and_all;
          break;
        case ptx_instruction::vote_uni:  // [한국어] uni 모드
          pred_value = (or_all ^ and_all);
          break;
        default:
          abort();
      }
      ptx_reg_t data;
      data.pred = pred_value ? 0 : 1;  // the way ptxplus handles the zero flag,
                                       // 1 = false and 0 = true

      for (std::list<ptx_thread_info *>::iterator t = threads_in_warp.begin();
           t != threads_in_warp.end(); ++t) {
        const operand_info &dst = pI->dst();
        (*t)->set_operand_value(dst, data, PRED_TYPE, (*t), pI);
      }
    }
    first_in_warp = true;
  }
}

/*
 * [한국어]
 * activemask_impl - PTX `activemask` 명령어 구현: 현재 워프 활성 마스크 읽기
 *
 * @pI: 실행 중인 PTX 명령어
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `activemask.b32 d` 명령어를 구현한다.
 * 현재 워프에서 활성화된 스레드들의 비트 마스크를 U32 레지스터에 기록한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [activemask_impl] → set_operand_value()
 */

void activemask_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  active_mask_t l_activemask_bitset = pI->get_warp_active_mask();
  uint32_t l_activemask_uint =
      static_cast<uint32_t>(l_activemask_bitset.to_ulong());

  const operand_info &dst = pI->dst();
  thread->set_operand_value(dst, l_activemask_uint, U32_TYPE, thread, pI);
}

/*
 * [한국어]
 * xor_impl - PTX `xor` 명령어 구현: 비트/프레디케이트 XOR
 *
 * @pI: 실행 중인 PTX 명령어 (목적지, src1, src2, 타입 포함)
 * @thread: 현재 실행 중인 PTX 스레드
 *
 * PTX `xor.type d, a, b` 명령어를 구현한다. d = a ^ b.
 * predicate 타입은 PTXPlus zero-flag 규칙(1=false, 0=true)을 고려하여 처리한다.
 * 실행 컨텍스트: 기능 시뮬레이션 스레드.
 *
 * 호출 체인:
 *   ptx_thread_info::ptx_exec_inst() → [xor_impl] → set_operand_value()
 */

void xor_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  ptx_reg_t src1_data, src2_data, data;

  const operand_info &dst = pI->dst();
  const operand_info &src1 = pI->src1();
  const operand_info &src2 = pI->src2();

  unsigned i_type = pI->get_type();
  src1_data = thread->get_operand_value(src1, dst, i_type, thread, 1);
  src2_data = thread->get_operand_value(src2, dst, i_type, thread, 1);

  // the way ptxplus handles predicates: 1 = false and 0 = true
  if (i_type == PRED_TYPE)  // [한국어] predicate 타입 분기
    data.pred = ~(~(src1_data.pred) ^ ~(src2_data.pred));  // [한국어] PTXPlus 규칙으로 XOR
  else
    data.u64 = src1_data.u64 ^ src2_data.u64;  // [한국어] 비트 XOR

  thread->set_operand_value(dst, data, i_type, thread, pI);
}

void inst_not_implemented(const ptx_instruction *pI) {
  printf(
      "GPGPU-Sim PTX: ERROR (%s:%u) instruction \"%s\" not (yet) implemented\n",
      pI->source_file(), pI->source_line(), pI->get_opcode_cstr());
  abort();
}

ptx_reg_t srcOperandModifiers(ptx_reg_t opData, operand_info opInfo,
                              operand_info dstInfo, unsigned type,
                              ptx_thread_info *thread) {
  ptx_reg_t result;
  memory_space *mem = NULL;
  size_t size;
  int t;
  result.u64 = 0;

  // complete other cases for reading from memory, such as reading from other
  // const memory
  if (opInfo.get_addr_space() == global_space) {
    mem = thread->get_global_memory();
    type_info_key::type_decode(type, size, t);
    mem->read(opData.u32, size / 8, &result.u64);
    if (type == S16_TYPE || type == S32_TYPE)
      sign_extend(result, size, dstInfo);
  } else if (opInfo.get_addr_space() == shared_space) {
    mem = thread->m_shared_mem;
    type_info_key::type_decode(type, size, t);
    mem->read(opData.u32, size / 8, &result.u64);

    if (type == S16_TYPE || type == S32_TYPE)
      sign_extend(result, size, dstInfo);

  } else if (opInfo.get_addr_space() == const_space) {
    mem = thread->get_global_memory();
    type_info_key::type_decode(type, size, t);

    mem->read((opData.u32 + opInfo.get_const_mem_offset()), size / 8,
              &result.u64);

    if (type == S16_TYPE || type == S32_TYPE)
      sign_extend(result, size, dstInfo);
  } else {
    result = opData;
  }

  if (opInfo.get_operand_lohi() == 1) {
    result.u64 = result.u64 & 0xFFFF;
  } else if (opInfo.get_operand_lohi() == 2) {
    result.u64 = (result.u64 >> 16) & 0xFFFF;
  }

  if (opInfo.get_operand_neg() == true) {
    result.f32 = -result.f32;
  }

  return result;
}

void video_mem_instruction(const ptx_instruction *pI, ptx_thread_info *thread,
                           int op_code) {
  const operand_info &dst = pI->dst();    // d
  const operand_info &src1 = pI->src1();  // a
  const operand_info &src2 = pI->src2();  // b
  const operand_info &src3 = pI->src3();  // c

  const unsigned i_type = pI->get_type();

  std::list<int> scalar_type;
  std::list<int> options;

  ptx_reg_t a, b, ta, tb, c, data;

  a = thread->get_operand_value(src1, dst, i_type, thread, 1);
  b = thread->get_operand_value(src2, dst, i_type, thread, 1);
  c = thread->get_operand_value(src3, dst, i_type, thread, 1);

  // TODO: implement this
  // ta = partSelectSignExtend( a, atype );
  // tb = partSelectSignExtend( b, btype );
  ta = a;
  tb = b;

  options = pI->get_options();
  assert(options.size() == 1);

  auto option = options.begin();
  assert(*option == ATOMIC_MAX || *option == ATOMIC_MIN);

  switch (i_type) {
    case S32_TYPE: {
      // assert all operands are S32_TYPE:
      scalar_type = pI->get_scalar_type();
      for (std::list<int>::iterator scalar = scalar_type.begin();
           scalar != scalar_type.end(); scalar++) {
        assert(*scalar == S32_TYPE);
      }
      assert(scalar_type.size() == 3);
      scalar_type.clear();

      switch (op_code) {
        case VMAX:
          data.s32 = MY_MAX_I(ta.s32, tb.s32);
          break;
        case VMIN:
          data.s32 = MY_MIN_I(ta.s32, tb.s32);
          break;
        default:
          assert(0);
      }

      switch (*option) {
        case ATOMIC_MAX:
          data.s32 = MY_MAX_I(data.s32, c.s32);
          break;
        case ATOMIC_MIN:
          data.s32 = MY_MIN_I(data.s32, c.s32);
          break;
        default:
          assert(0);  // not yet implemented
      }
      break;
    }
    default:
      assert(0);  // not yet implemented
  }

  thread->set_operand_value(dst, data, i_type, thread, pI);

  return;
}
