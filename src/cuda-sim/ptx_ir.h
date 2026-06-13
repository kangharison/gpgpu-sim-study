// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, Wilson W.L. Fung,
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
 * [한국어 설명] PTX 중간 표현(IR) 자료구조 정의 (ptx_ir.h)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim의 기능 시뮬레이션(cuda-sim) 계층에서 사용하는 PTX(Parallel
 * Thread Execution) 중간 표현의 핵심 자료구조를 정의한다. PTX는 NVIDIA GPU의 가상
 * ISA(Instruction Set Architecture)로, CUDA 컴파일러(nvcc)가 CUDA C 코드를 SASS
 * (실제 GPU 기계어)로 컴파일하기 전에 생성하는 중간 표현이다. GPGPU-Sim은 이 PTX를
 * 파싱하고 함수/명령어 단위로 IR 객체를 구성한 뒤, 사이클-레벨 타이밍 모델과 연동하여
 * GPU 동작을 시뮬레이션한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름:
 *   CUDA 애플리케이션 바이너리
 *     → libcuda 인터셉트 (cuLaunchKernel 등 가로채기)
 *         → gpgpusim_entrypoint.cc (시뮬레이터 진입)
 *             → cuda-sim/ptx_parser.{cc,h} (PTX 소스 Lex/Yacc 파싱)
 *                 → [이 파일] ptx_ir.h — IR 자료구조 정의
 *                     → ptx_ir.cc — 메서드 구현
 *                     → instructions.cc — 명령어별 시맨틱 실행
 *                 → gpgpu-sim/shader.cc (타이밍 모델과 연동)
 * 이 파일은 기능 시뮬레이션(functional simulation) 단계에서 PTX 명령어를 메모리에
 * 표현하는 모든 클래스/구조체를 선언하며, 타이밍 시뮬레이션(gpgpu-sim/)이 이
 * 자료구조를 참조하여 warp_inst_t 레벨에서 사이클 단위 실행을 수행한다.
 *
 * === 타 모듈과의 연결 ===
 * - 의존(inbound): ptx_parser.y(Yacc 파서)가 파싱 중 이 클래스들을 생성하여
 *   function_info와 symbol_table에 명령어/심볼을 등록한다.
 * - 의존(outbound): instructions.cc가 ptx_instruction의 피연산자(operand_info)와
 *   타입 정보를 읽어 실제 GPU 연산 시맨틱을 구현한다.
 * - warp_inst_t 상속: ptx_instruction은 abstract_hardware_model.h의 warp_inst_t를
 *   상속하여, 타이밍 모델(shader.cc)에서 warp 단위 명령어 스케줄링에 직접 사용된다.
 * - gpgpu_context: 전역 시뮬레이터 컨텍스트(gpgpu_context*)를 통해 UID 발급, 전역
 *   PC-to-instruction 테이블(s_g_pc_to_insn) 접근 등이 이루어진다.
 * - 데이터 흐름: PTX 파서 → (symbol/operand_info/ptx_instruction 생성) →
 *   function_info::m_instructions 리스트 → ptx_assemble()로 배열화 →
 *   shader.cc의 warp 실행 루프에서 fetch/decode/execute.
 *
 * === 주요 함수/구조체 요약 ===
 * - type_info_key: PTX 타입 (.s32, .f64, shared 등)의 키 구조체; map 정렬용 비교자 포함
 * - symbol: PTX 심볼(변수, 레이블, 함수); 이름/타입/주소/레지스터번호 보유
 * - symbol_table: 스코프별 심볼 맵; 함수/변수/타입 등록 및 조회, 메모리 공간 할당기 포함
 * - operand_info: PTX 피연산자(레지스터/즉값/메모리/레이블 등); 명령어 필드로 사용
 * - basic_block_t: PTX 명령어 시퀀스의 기본 블록; dominaor/post-dominator 분석에 사용
 * - ptx_instruction: 하나의 PTX 명령어 전체(opcode, pred, operands, 옵션 등); warp_inst_t 상속
 * - param_info: 커널/함수 파라미터 정보(이름/타입/크기/값)
 * - function_info: 하나의 PTX 함수/커널 전체(명령어 목록, 심볼 테이블, CFG, 재수렴 분석)
 * - arg_buffer_t: 함수 호출 시 인자를 임시 저장하는 버퍼; 레지스터형/파라미터형 분기
 */

#ifndef ptx_ir_INCLUDED
#define ptx_ir_INCLUDED

/* [한국어] abstract_hardware_model.h: warp_inst_t, addr_t, memory_space_t 등
 * GPU 마이크로아키텍처 공통 추상 자료형 정의. ptx_instruction이 warp_inst_t를 상속. */
#include "../abstract_hardware_model.h"

#include <assert.h>   /* [한국어] 런타임 단언(assertion) 매크로 */
#include <cstdlib>    /* [한국어] malloc/calloc/free 등 표준 메모리 관리 */
#include <cstring>    /* [한국어] memcpy/snprintf 등 문자열/메모리 조작 */
#include <list>       /* [한국어] std::list — 명령어 리스트, 심볼 리스트 등 순서 있는 컨테이너 */
#include <map>        /* [한국어] std::map — 심볼 테이블(이름→symbol*) 구현 */
#include <string>     /* [한국어] std::string — 심볼 이름, 소스 파일명 등 문자열 저장 */
#include <vector>     /* [한국어] std::vector — 피연산자 배열, 기본 블록 벡터 등 */

//#include "ptx.tab.h"  /* [한국어] Yacc가 생성하는 토큰 정의 — ptx_ir.cc에서 직접 포함 */
/* [한국어] ptx_sim.h: ptx_thread_info, ptx_reg_t 등 PTX 시뮬레이션 자료형 정의 */
#include "ptx_sim.h"

/* [한국어] memory.h: memory_space 클래스 — shared/global/local/const 메모리 공간 추상화 */
#include "memory.h"

/* [한국어] 전역 시뮬레이터 컨텍스트 — UID 발급, 전역 PC-to-instruction 테이블 관리 등에 사용.
 * 전방 선언만 하고 실제 정의는 libcuda/gpgpu_context.h에 있음. */
class gpgpu_context;

/*
 * [한국어]
 * type_info_key - PTX 타입 정보를 표현하는 키 구조체
 *
 * PTX의 타입 시스템(.s32, .f64, .shared 등)을 내부적으로 표현하고,
 * symbol_table의 std::map<type_info_key, type_info*, ...>에서 정렬 키로 사용된다.
 * 각 타입은 메모리 공간(space), 스칼라 타입(scalar_type_spec), 벡터 폭,
 * 정렬, extern 여부, 배열 차원, 함수 주소 여부의 조합으로 유일하게 식별된다.
 *
 * 설정자: symbol_table::add_type() 또는 직접 생성자 호출 (ptx_parser.y에서 파싱 시 사용).
 * 읽는 자: symbol::is_shared(), is_global() 등 타입 판별 메서드들이 이 객체를 통해 공간 확인.
 * 동기화: 파싱은 단일 스레드 컨텍스트에서 수행되므로 별도 락 불필요.
 *
 * 호출 체인:
 *   ptx.y (파서 액션) → symbol_table::add_type() → [type_info_key 생성] → type_info 저장
 */
class type_info_key {
 public:
  /*
   * [한국어]
   * type_info_key() - 기본 생성자
   *
   * 초기화되지 않은 상태(m_init=false)의 키를 생성한다. 이후 set_is_func() 또는
   * 매개변수 생성자를 통해 초기화해야 한다. assert(m_init) 체크 전에 초기화 없이
   * 사용하면 런타임 에러가 발생한다.
   */
  type_info_key() {
    m_is_non_arch_reg = false; /* [한국어] 비아키텍처 레지스터 플래그 초기화 — "_" 레지스터 구분용 */
    m_init = false;            /* [한국어] 아직 초기화되지 않음 표시 */
  }
  /*
   * [한국어]
   * type_info_key(space, scalar, vector, align, extern, dim) - 완전 초기화 생성자
   *
   * @space_spec: PTX 메모리 공간 (reg_space, shared_space, global_space, local_space 등)
   * @scalar_type_spec: PTX 스칼라 타입 토큰 (S32_TYPE, F64_TYPE, PRED_TYPE 등)
   * @vector_spec: 벡터 폭 지정 (0=스칼라, V2_TYPE=2, V4_TYPE=4 등)
   * @alignment_spec: 정렬 요구사항 (바이트 단위; 0이면 기본 정렬)
   * @extern_spec: extern 선언 여부 (1이면 외부 모듈에서 정의됨)
   * @array_dim: 배열 차원 (0이면 단순 변수)
   *
   * 호출자: ptx.y 파서 액션에서 변수/파라미터 선언 시 직접 생성.
   */
  type_info_key(memory_space_t space_spec, int scalar_type_spec,
                int vector_spec, int alignment_spec, int extern_spec,
                int array_dim) {
    m_is_non_arch_reg = false;             /* [한국어] 비아키텍처 레지스터 여부 — 기본값 false */
    m_init = true;                         /* [한국어] 초기화 완료 표시 */
    m_space_spec = space_spec;             /* [한국어] 메모리 공간 설정 (reg_space, shared_space 등) */
    m_scalar_type_spec = scalar_type_spec; /* [한국어] PTX 스칼라 타입 (S32_TYPE, F32_TYPE 등) */
    m_vector_spec = vector_spec;           /* [한국어] 벡터 폭 (V2/V4/0=스칼라) */
    m_alignment_spec = alignment_spec;     /* [한국어] 정렬 요구사항 (바이트) */
    m_extern_spec = extern_spec;           /* [한국어] extern 선언 여부 */
    m_array_dim = array_dim;              /* [한국어] 배열 차원 크기 */
    m_is_function = 0;                    /* [한국어] 함수 주소 타입 아님 */
  }
  /*
   * [한국어]
   * set_is_func() - 함수 주소 타입으로 초기화
   *
   * 이 타입이 함수 포인터임을 나타낸다. 파서에서 함수 선언(`.func`, `.entry`)을
   * 처리할 때 해당 함수 심볼의 타입으로 사용된다. 기존 초기화가 없는 상태에서만
   * 호출해야 한다 (assert(!m_init)).
   *
   * 호출 체인:
   *   symbol_table::add_type(function_info*) → [type_info_key 생성] → set_is_func()
   */
  void set_is_func() {
    assert(!m_init);           /* [한국어] 이미 초기화된 키에 재초기화 시도 금지 */
    m_init = true;             /* [한국어] 초기화 완료 표시 */
    m_space_spec = undefined_space; /* [한국어] 함수 포인터는 특정 메모리 공간에 속하지 않음 */
    m_scalar_type_spec = 0;    /* [한국어] 스칼라 타입 없음 */
    m_vector_spec = 0;         /* [한국어] 벡터 폭 없음 */
    m_alignment_spec = 0;      /* [한국어] 정렬 요구사항 없음 */
    m_extern_spec = 0;         /* [한국어] extern 아님 */
    m_array_dim = 0;           /* [한국어] 배열 아님 */
    m_is_function = 1;         /* [한국어] 함수 주소 타입임을 표시 */
  }

  /* [한국어] set_array_dim() - 배열 차원 크기 설정. 배열 타입 생성 시 호출 */
  void set_array_dim(int array_dim) { m_array_dim = array_dim; }
  /* [한국어] get_array_dim() - 배열 차원 크기 반환. m_init 검사 후 반환 */
  int get_array_dim() const {
    assert(m_init);
    return m_array_dim;
  }
  /* [한국어] set_is_non_arch_reg() - 비아키텍처 레지스터 표시.
   * "_" 레지스터처럼 파서는 인식하지만 타이밍 시뮬레이터에 전달하지 않는 레지스터에 사용 */
  void set_is_non_arch_reg() { m_is_non_arch_reg = true; }

  /* [한국어] 타입 판별 메서드들 — 각 PTX 메모리 공간/타입 여부를 반환 */
  bool is_non_arch_reg() const { return m_is_non_arch_reg; } /* [한국어] 비아키텍처 레지스터("_")인지 확인 */
  bool is_reg() const { return m_space_spec == reg_space; }  /* [한국어] 레지스터 공간(.reg)인지 확인 */
  bool is_param_kernel() const { return m_space_spec == param_space_kernel; }    /* [한국어] 커널 파라미터 공간인지 확인 */
  bool is_param_local() const { return m_space_spec == param_space_local; }      /* [한국어] 로컬 파라미터 공간인지 확인 */
  bool is_param_unclassified() const {
    return m_space_spec == param_space_unclassified; /* [한국어] 미분류 파라미터 공간인지 확인 */
  }
  bool is_global() const { return m_space_spec == global_space; } /* [한국어] 글로벌 메모리 공간인지 확인 */
  bool is_local() const { return m_space_spec == local_space; }   /* [한국어] 로컬 메모리 공간인지 확인 */
  bool is_shared() const { return m_space_spec == shared_space; } /* [한국어] 공유 메모리 공간인지 확인 */
  bool is_const() const { return m_space_spec.get_type() == const_space; } /* [한국어] 상수 메모리 공간인지 확인 */
  bool is_tex() const { return m_space_spec == tex_space; }       /* [한국어] 텍스처 메모리 공간인지 확인 */
  bool is_func_addr() const { return m_is_function ? true : false; } /* [한국어] 함수 주소 타입인지 확인 */
  int scalar_type() const { return m_scalar_type_spec; }           /* [한국어] PTX 스칼라 타입 토큰값 반환 (S32_TYPE 등) */
  int get_alignment_spec() const { return m_alignment_spec; }      /* [한국어] 정렬 요구사항(바이트) 반환 */
  /* [한국어] type_decode() - PTX 타입 토큰을 크기(비트)와 기본타입 분류로 변환.
   * size: 비트 크기(예: S32→32), t: 1=정수부호, 0=정수무부호, -1=부동소수, 2=pred, 3=텍스처 */
  unsigned type_decode(size_t &size, int &t) const;
  static unsigned type_decode(int type, size_t &size, int &t);
  /* [한국어] get_memory_space() - 이 타입의 메모리 공간 열거값 반환 */
  memory_space_t get_memory_space() const { return m_space_spec; }

 private:
  bool m_init;
  /* [한국어] m_init: 이 키가 유효하게 초기화되었는지 여부.
   * 설정자: 생성자 또는 set_is_func() 에서 true로 설정.
   * 읽는 자: get_array_dim() 등 접근자에서 assert(m_init)로 사전 조건 검사.
   * 값 범위: false(미초기화), true(초기화 완료). */

  memory_space_t m_space_spec;
  /* [한국어] m_space_spec: PTX 메모리 공간 지정자 (reg_space, shared_space,
   * global_space, local_space, param_space_kernel, param_space_local, tex_space 등).
   * 설정자: 생성자에서 파서의 .space 토큰으로부터 설정됨.
   * 읽는 자: is_shared(), is_global() 등 판별 메서드들.
   * 값 범위: memory_space_t 열거형 — undefined_space부터 tex_space까지. */

  int m_scalar_type_spec;
  /* [한국어] m_scalar_type_spec: PTX 스칼라 데이터 타입 토큰 값.
   * 예: S8_TYPE, S16_TYPE, S32_TYPE, S64_TYPE, U8_TYPE, F32_TYPE, F64_TYPE, PRED_TYPE, B8_TYPE 등.
   * 설정자: 생성자에서 파서의 .type 토큰으로부터 설정됨.
   * 읽는 자: scalar_type() 접근자, type_decode()에서 비트 크기 계산에 사용.
   * 값 범위: ptx.tab.h에 정의된 타입 토큰 상수 (예: S32_TYPE=0x...) */

  int m_vector_spec;
  /* [한국어] m_vector_spec: PTX 벡터 타입 지정자 (V2_TYPE, V3_TYPE, V4_TYPE).
   * ld.v2/st.v4 등 벡터 load/store에서 몇 개 요소를 묶어 처리하는지 나타냄.
   * 설정자: 생성자에서 파서의 .v2/.v4 토큰으로부터 설정됨.
   * 값 범위: 0(스칼라), V2_TYPE, V3_TYPE, V4_TYPE. */

  int m_alignment_spec;
  /* [한국어] m_alignment_spec: 메모리 정렬 요구사항 (바이트).
   * .align N 지시어로 지정되며, 공유 메모리 배열 등의 정렬 경계를 결정함.
   * 값 범위: 0(기본 정렬), 또는 2의 거듭제곱 값(4, 8, 16 등). */

  int m_extern_spec;
  /* [한국어] m_extern_spec: extern 선언 여부 (1이면 extern).
   * 다른 PTX 모듈에서 정의되는 전역 변수/함수를 참조할 때 사용. */

  int m_array_dim;
  /* [한국어] m_array_dim: 배열 차원 크기 (원소 수).
   * .global .b32 arr[256] 같은 배열 선언에서 256에 해당.
   * 값 범위: 0(비배열), 양의 정수(배열 원소 수). */

  int m_is_function;
  /* [한국어] m_is_function: 함수 주소 타입 여부 (1이면 함수 포인터).
   * set_is_func()로 설정되며, symbol::get_pc()를 통해 function_info* 반환 가능.
   * 값 범위: 0(일반 타입), 1(함수 주소 타입). */

  bool m_is_non_arch_reg;
  /* [한국어] m_is_non_arch_reg: 비아키텍처 레지스터 여부.
   * "_" 레지스터처럼 파서는 인식하지만 타이밍 시뮬레이터에 전달하지 않는
   * 더미 레지스터를 식별하는 플래그. set_is_non_arch_reg()로 설정.
   * 읽는 자: operand_info::is_non_arch_reg()에서 타이밍 모델 전달 여부 결정. */

  friend struct type_info_key_compare; /* [한국어] std::map 정렬에 사용되는 비교 함수자 */
};

/* [한국어] symbol_table: 전방 선언 — type_info 내에서 포인터로만 참조됨 */
class symbol_table;

/*
 * [한국어]
 * type_info_key_compare - type_info_key를 std::map의 키로 사용하기 위한 비교 함수자
 *
 * symbol_table::m_types 맵의 Compare 인자로 사용되며, 두 type_info_key를 사전식으로
 * 비교한다. 비교 순서: space_spec → scalar_type_spec → vector_spec → alignment_spec
 * → extern_spec → array_dim → is_function. 이 순서는 타입의 주요 속성이 먼저 비교되도록
 * 설계된 것으로, 동일 타입은 동일 키로 매핑되어 중복 생성을 방지한다.
 *
 * 동기화: 파싱 단계는 단일 스레드이므로 별도 동기화 불필요.
 */
struct type_info_key_compare {
  /*
   * [한국어]
   * operator() - 두 type_info_key의 사전식 비교 (a < b)
   *
   * @a, @b: 비교할 두 타입 키 (모두 초기화되어야 함)
   * @return: a가 b보다 사전식으로 작으면 true; std::map의 strict weak ordering 요구 충족
   *
   * 각 필드를 순서대로 비교하여 최초로 다른 필드에서 결과를 반환한다.
   * 모든 필드가 같으면 false를 반환 (a == b).
   */
  bool operator()(const type_info_key &a, const type_info_key &b) const {
    assert(a.m_init && b.m_init);                          /* [한국어] 미초기화 키로 비교 시도 방지 */
    if (a.m_space_spec < b.m_space_spec) return true;      /* [한국어] 메모리 공간이 다르면 공간 기준 정렬 */
    if (a.m_scalar_type_spec < b.m_scalar_type_spec) return true; /* [한국어] 스칼라 타입 기준 정렬 */
    if (a.m_vector_spec < b.m_vector_spec) return true;    /* [한국어] 벡터 폭 기준 정렬 */
    if (a.m_alignment_spec < b.m_alignment_spec) return true; /* [한국어] 정렬 요구사항 기준 정렬 */
    if (a.m_extern_spec < b.m_extern_spec) return true;    /* [한국어] extern 여부 기준 정렬 */
    if (a.m_array_dim < b.m_array_dim) return true;        /* [한국어] 배열 차원 기준 정렬 */
    if (a.m_is_function < b.m_is_function) return true;    /* [한국어] 함수 주소 여부 기준 정렬 */

    return false; /* [한국어] 모든 필드가 같음 — a == b이므로 a < b는 false */
  }
};

/*
 * [한국어]
 * type_info - PTX 타입 정보 래퍼 클래스
 *
 * type_info_key를 보유하며, 심볼(symbol)이 참조하는 타입 객체의 실체이다.
 * symbol_table이 타입을 생성하고 소유하며, symbol은 const type_info* 포인터로 참조한다.
 * 타입이 속한 스코프(symbol_table*)도 보유하나 현재 m_scope는 생성 외에 직접 사용되지 않는다.
 *
 * 설정자: symbol_table::add_type() 에서 new로 생성.
 * 읽는 자: symbol::type()을 통해 반환받고, type()->get_key()로 타입 속성 판별.
 * 동기화: 파싱 단계에서만 생성되므로 별도 락 불필요.
 */
class type_info {
 public:
  /*
   * [한국어] 생성자 — scope와 type_info_key로 초기화.
   * @scope: 이 타입이 선언된 심볼 테이블 (현재는 저장만 하고 사용 안 함)
   * @t: 타입의 실제 내용을 담은 type_info_key
   */
  type_info(symbol_table *scope, type_info_key t) { m_type_info = t; }
  /* [한국어] get_key() - 내부 type_info_key 참조 반환 — 타입 속성 판별에 사용 */
  const type_info_key &get_key() const { return m_type_info; }

 private:
  symbol_table *m_scope;
  /* [한국어] m_scope: 이 타입이 선언된 심볼 테이블 스코프 포인터.
   * 현재 구현에서는 저장만 되고 직접 활용되지 않는다. (TODO 주석 있음)
   * 값 범위: 유효한 symbol_table* 또는 NULL (add_type(func) 경우 NULL 전달). */

  type_info_key m_type_info;
  /* [한국어] m_type_info: 이 타입의 실제 속성을 담은 키.
   * get_key()를 통해 외부에 const 참조로 제공됨.
   * 설정자: 생성자에서 초기화. 이후 변경 없음 (불변). */
};

/*
 * [한국어]
 * operand_type - PTX 피연산자의 종류를 나타내는 열거형
 *
 * PTX 명령어의 각 피연산자(operand_info)가 어떤 형태인지 구분한다.
 * 이 값은 operand_info::m_type에 저장되며, instructions.cc의 피연산자 읽기 로직에서
 * 어떤 union 필드를 사용할지 결정한다.
 *
 * 사용 예:
 *   add.s32 %r1, %r2, 5   → %r1, %r2는 reg_t; 5는 int_t
 *   ld.global.s32 %r1, [%r2+8]  → [%r2+8]은 address_t
 *   bra $L_end             → $L_end는 label_t
 */
enum operand_type {
  reg_t,          /* [한국어] 단일 레지스터 (예: %r1, %f2, %p0) */
  vector_t,       /* [한국어] 벡터 레지스터 묶음 (예: {%r0, %r1, %r2, %r3}) ld.v4 등에 사용 */
  builtin_t,      /* [한국어] CUDA 내장 변수 (예: %tid.x, %ctaid.y, %ntid.z 등 스레드/블록 ID) */
  address_t,      /* [한국어] 메모리 주소 (예: [%r1+4] 형식 — 베이스 레지스터 + 오프셋) */
  memory_t,       /* [한국어] 메모리 피연산자 (두 심볼로 표현; 주로 PTXPlus에서 사용) */
  float_op_t,     /* [한국어] 단정밀도 부동소수점 즉값 (float 리터럴) */
  double_op_t,    /* [한국어] 배정밀도 부동소수점 즉값 (double 리터럴) */
  int_t,          /* [한국어] 부호 있는 정수 즉값 */
  unsigned_t,     /* [한국어] 부호 없는 정수 즉값 또는 즉시 주소 */
  symbolic_t,     /* [한국어] 심볼릭 주소 (전역/공유/로컬 변수, param, tex 이름 등) */
  label_t,        /* [한국어] 분기 레이블 (bra 명령어의 대상 주소) */
  v_reg_t,        /* [한국어] 벡터 레지스터 단독 (현재 미사용 가능) */
  v_float_op_t,   /* [한국어] 벡터 float 즉값 배열 */
  v_double_op_t,  /* [한국어] 벡터 double 즉값 배열 */
  v_int_t,        /* [한국어] 벡터 정수 즉값 배열 */
  v_unsigned_t,   /* [한국어] 벡터 무부호 정수 즉값 배열 */
  undef_t         /* [한국어] 미정의 타입 — operand_info 기본값; 유효하지 않은 피연산자 */
};

/* [한국어] operand_info: 전방 선언 — symbol::add_initializer()의 인자 타입으로 사용 */
class operand_info;

/*
 * [한국어]
 * symbol - PTX 심볼(변수, 레이블, 함수)을 표현하는 클래스
 *
 * PTX 코드에서 선언되는 모든 이름(레지스터, 전역/공유/로컬 변수, 레이블, 함수 이름)을
 * 표현한다. 각 심볼은 이름, 타입, 선언 위치, 크기, 메모리 주소(또는 레지스터 번호)를
 * 보유한다. symbol_table이 이 객체들을 소유하며 이름으로 조회한다.
 *
 * 설정자: symbol_table::add_variable(), add_function()에서 new로 생성됨.
 * 읽는 자: operand_info(const symbol*)에서 심볼 참조, instructions.cc에서 레지스터 번호
 *          및 메모리 주소 접근.
 * 동기화: 파싱 단계(단일 스레드)에서 생성/설정되고, 시뮬레이션 단계에서는 읽기 전용.
 */
class symbol {
 public:
  /*
   * [한국어]
   * symbol() - 심볼 생성자
   *
   * @name: PTX 코드에서의 심볼 이름 (예: "%r1", "__shared_arr", "$L_end")
   * @type: 이 심볼의 타입 (type_info* — NULL이면 타입 없음; 레이블에서 발생 가능)
   * @location: 선언 위치 문자열 (파일명:줄번호 형식, 디버그 출력에 사용)
   * @size: 크기 (바이트 단위; 레지스터는 타입 크기, 배열은 전체 크기)
   * @ctx: 전역 GPGPU 컨텍스트 (UID 발급에 사용)
   * @return: (생성자이므로 반환 없음)
   *
   * type의 공간 속성(is_shared, is_const 등)을 읽어 멤버 플래그를 초기화한다.
   * 주소/레지스터 번호는 아직 유효하지 않은 상태(m_address_valid=false, m_reg_num_valid=false)로
   * 초기화되며, 이후 set_address() 또는 set_regno()로 설정된다.
   *
   * 호출 체인:
   *   symbol_table::add_variable() → new symbol(...) → [초기화]
   */
  symbol(const char *name, const type_info *type, const char *location,
         unsigned size, gpgpu_context *ctx) {
    gpgpu_ctx = ctx;                                /* [한국어] 전역 컨텍스트 참조 저장 — UID 발급에 사용 */
    m_uid = get_uid();                              /* [한국어] 전역 고유 ID 할당 (gpgpu_ctx->symbol_sm_next_uid++) */
    m_name = name;                                  /* [한국어] 심볼 이름 저장 */
    m_decl_location = location;                     /* [한국어] 선언 위치("file.ptx:123") 저장 — 디버깅용 */
    m_type = type;                                  /* [한국어] 타입 정보 포인터 저장 */
    m_size = size;                                  /* [한국어] 바이트 단위 크기 저장 */
    m_address_valid = false;                        /* [한국어] 아직 메모리 주소 미할당 */
    m_is_label = false;                             /* [한국어] 레이블이 아님으로 초기화 */
    m_is_shared = false;                            /* [한국어] 공유 메모리 변수 아님으로 초기화 */
    m_is_const = false;                             /* [한국어] 상수 메모리 변수 아님으로 초기화 */
    m_is_global = false;                            /* [한국어] 전역 메모리 변수 아님으로 초기화 */
    m_is_local = false;                             /* [한국어] 로컬 메모리 변수 아님으로 초기화 */
    m_is_param_local = false;                       /* [한국어] 로컬 파라미터 아님으로 초기화 */
    m_is_param_kernel = false;                      /* [한국어] 커널 파라미터 아님으로 초기화 */
    m_is_tex = false;                               /* [한국어] 텍스처 심볼 아님으로 초기화 */
    m_is_func_addr = false;                         /* [한국어] 함수 주소 아님으로 초기화 */
    m_reg_num_valid = false;                        /* [한국어] 레지스터 번호 미할당 */
    m_function = NULL;                              /* [한국어] 함수 심볼이 아닐 경우 NULL */
    m_reg_num = (unsigned)-1;                       /* [한국어] 유효하지 않은 레지스터 번호로 초기화 */
    m_arch_reg_num = (unsigned)-1;                  /* [한국어] 유효하지 않은 아키텍처 레지스터 번호로 초기화 */
    m_address = (unsigned)-1;                       /* [한국어] 유효하지 않은 주소로 초기화 */
    m_initializer.clear();                          /* [한국어] 초기화 값 리스트 비우기 */
    if (type) m_is_shared = type->get_key().is_shared();           /* [한국어] 공유 메모리 여부 타입에서 파악 */
    if (type) m_is_const = type->get_key().is_const();             /* [한국어] 상수 메모리 여부 타입에서 파악 */
    if (type) m_is_global = type->get_key().is_global();           /* [한국어] 전역 메모리 여부 타입에서 파악 */
    if (type) m_is_local = type->get_key().is_local();             /* [한국어] 로컬 메모리 여부 타입에서 파악 */
    if (type) m_is_param_local = type->get_key().is_param_local(); /* [한국어] 로컬 파라미터 여부 타입에서 파악 */
    if (type) m_is_param_kernel = type->get_key().is_param_kernel(); /* [한국어] 커널 파라미터 여부 타입에서 파악 */
    if (type) m_is_tex = type->get_key().is_tex();                 /* [한국어] 텍스처 여부 타입에서 파악 */
    if (type) m_is_func_addr = type->get_key().is_func_addr();     /* [한국어] 함수 주소 여부 타입에서 파악 */
  }
  /* [한국어] get_size_in_bytes() - 심볼의 바이트 단위 크기 반환 */
  unsigned get_size_in_bytes() const { return m_size; }
  /* [한국어] name() - 심볼의 PTX 이름 문자열 참조 반환 (예: "%r1", "__arr") */
  const std::string &name() const { return m_name; }
  /* [한국어] decl_location() - 선언 위치 문자열 반환 (예: "kernel.ptx:42") */
  const std::string &decl_location() const { return m_decl_location; }
  /* [한국어] type() - 심볼의 타입 정보 포인터 반환 — NULL일 경우 레이블 등 타입 없는 심볼 */
  const type_info *type() const { return m_type; }
  /* [한국어] has_valid_address() - 이 심볼에 유효한 메모리 주소가 할당되었는지 확인 */
  bool has_valid_address() const { return m_address_valid; }
  /*
   * [한국어]
   * get_address() - 심볼의 메모리 주소 반환
   *
   * 레지스터가 아니거나 레이블인 심볼의 메모리 주소를 반환한다.
   * 호출 전에 set_address() 또는 set_label_address()로 주소가 설정되어야 한다.
   * 레지스터 심볼에 대해 이 함수를 호출하면 assert 실패.
   *
   * @return: 심볼이 위치한 메모리 주소 (global/shared/local/param 공간의 오프셋)
   */
  addr_t get_address() const {
    assert(m_is_label ||
           !m_type->get_key().is_reg());  // todo : other assertions
    /* [한국어] 레이블이거나 레지스터가 아닌 경우에만 주소 접근 허용 */
    assert(m_address_valid); /* [한국어] 주소가 유효하게 설정된 경우에만 반환 */
    return m_address;
  }
  /* [한국어] get_pc() - 함수 심볼의 function_info* 반환 (non-func 심볼이면 NULL) */
  function_info *get_pc() const { return m_function; }
  /*
   * [한국어]
   * set_regno() - 레지스터 번호 설정
   *
   * @regno: 심볼 테이블 레지스터 번호 (symbol_table::next_reg_num()으로 할당)
   * @arch_regno: 아키텍처 레지스터 번호 (타이밍 모델에서 실제 레지스터 파일 인덱스)
   *
   * 파서가 .reg 공간의 심볼을 등록할 때 호출된다. 이후 reg_num(), arch_reg_num()으로 읽음.
   */
  void set_regno(unsigned regno, unsigned arch_regno) {
    m_reg_num_valid = true; /* [한국어] 레지스터 번호 유효 표시 */
    m_reg_num = regno;      /* [한국어] 심볼 테이블 레지스터 인덱스 저장 */
    m_arch_reg_num = arch_regno; /* [한국어] 아키텍처 레지스터 인덱스 저장 (scoreboard 등에서 사용) */
  }

  /*
   * [한국어]
   * set_address() - 심볼에 메모리 주소 할당
   *
   * @addr: 할당할 메모리 공간 내 오프셋 주소
   * 전역/공유/로컬/상수 변수 심볼에 대해 symbol_table이 주소 공간 할당기(alloc_*) 사용 후 설정.
   */
  void set_address(addr_t addr) {
    m_address_valid = true; /* [한국어] 주소 유효 표시 */
    m_address = addr;       /* [한국어] 메모리 주소 저장 */
  }
  /*
   * [한국어]
   * set_label_address() - 레이블 심볼의 PC 주소 설정
   *
   * @addr: 이 레이블이 가리키는 PTX 명령어의 PC 값 (m_instr_mem 인덱스)
   * 분기 대상 레이블을 어셈블 단계(ptx_assemble)에서 resolve할 때 사용.
   */
  void set_label_address(addr_t addr) {
    m_address_valid = true; /* [한국어] 주소 유효 표시 */
    m_address = addr;       /* [한국어] 레이블이 가리키는 PC 저장 */
    m_is_label = true;      /* [한국어] 레이블 심볼로 표시 */
  }
  /*
   * [한국어]
   * set_function() - 함수 심볼에 function_info* 연결
   *
   * @func: 이 심볼이 가리키는 함수의 function_info 객체
   * 파서가 함수 선언을 처리할 때 symbol_table::add_function()에서 호출.
   */
  void set_function(function_info *func) {
    m_function = func;    /* [한국어] 함수 정보 포인터 연결 */
    m_is_func_addr = true; /* [한국어] 함수 주소 심볼로 표시 */
  }

  /* [한국어] 심볼 속성 판별 메서드들 — 모두 m_is_* 멤버를 반환 */
  bool is_label() const { return m_is_label; }           /* [한국어] 분기 레이블인지 확인 */
  bool is_shared() const { return m_is_shared; }         /* [한국어] 공유 메모리 변수인지 확인 */
  bool is_sstarr() const { return m_is_sstarr; }         /* [한국어] sstarr(stack-stored array) 여부 확인 */
  bool is_const() const { return m_is_const; }           /* [한국어] 상수 메모리 변수인지 확인 */
  bool is_global() const { return m_is_global; }         /* [한국어] 전역 메모리 변수인지 확인 */
  bool is_local() const { return m_is_local; }           /* [한국어] 로컬 메모리 변수인지 확인 */
  bool is_param_local() const { return m_is_param_local; } /* [한국어] 로컬 파라미터인지 확인 */
  bool is_param_kernel() const { return m_is_param_kernel; } /* [한국어] 커널 파라미터인지 확인 */
  bool is_tex() const { return m_is_tex; }               /* [한국어] 텍스처 심볼인지 확인 */
  bool is_func_addr() const { return m_is_func_addr; }   /* [한국어] 함수 주소 심볼인지 확인 */
  /*
   * [한국어]
   * is_reg() - 이 심볼이 레지스터 공간(.reg)에 속하는지 확인
   *
   * @return: type이 NULL이면 false; 아니면 type->get_key().is_reg() 반환
   * 타입 없는 심볼(레이블 등)에 대해 안전하게 false를 반환.
   */
  bool is_reg() const {
    if (m_type == NULL) { /* [한국어] 타입 없는 심볼(예: 일부 레이블) — 레지스터 아님 */
      return false;
    }
    return m_type->get_key().is_reg(); /* [한국어] 타입의 메모리 공간이 reg_space인지 확인 */
  }
  /*
   * [한국어]
   * is_non_arch_reg() - 비아키텍처 레지스터("_")인지 확인
   *
   * @return: type이 NULL이면 false; 아니면 type->get_key().is_non_arch_reg() 반환
   * "_" 레지스터는 타이밍 시뮬레이터에 전달되지 않으므로 이 메서드로 걸러냄.
   */
  bool is_non_arch_reg() const {
    if (m_type == NULL) { /* [한국어] 타입 없는 심볼 — 비아키텍처 레지스터 아님 */
      return false;
    }
    return m_type->get_key().is_non_arch_reg(); /* [한국어] 비아키텍처 레지스터 여부 타입에서 확인 */
  }

  /* [한국어] add_initializer() - 전역/상수 변수의 초기화 값 리스트 설정 */
  void add_initializer(const std::list<operand_info> &init);
  /* [한국어] has_initializer() - 초기화 값이 있는지 확인 */
  bool has_initializer() const { return m_initializer.size() > 0; }
  /* [한국어] get_initializer() - 초기화 값 리스트 복사본 반환 */
  std::list<operand_info> get_initializer() const { return m_initializer; }
  /*
   * [한국어]
   * reg_num() - 심볼 테이블 레지스터 번호 반환
   *
   * @return: set_regno()로 설정된 심볼 테이블 인덱스
   * m_reg_num_valid가 true여야 호출 가능 (assert로 검사).
   * instructions.cc에서 레지스터 읽기/쓰기 시 사용.
   */
  unsigned reg_num() const {
    assert(m_reg_num_valid); /* [한국어] 레지스터 번호가 설정된 경우에만 반환 */
    return m_reg_num;
  }
  /*
   * [한국어]
   * arch_reg_num() - 아키텍처 레지스터 번호 반환
   *
   * @return: 타이밍 모델(scoreboard, register file)에서 사용하는 실제 레지스터 인덱스
   * shader.cc의 scoreboard가 이 번호로 레지스터 의존성을 추적함.
   */
  unsigned arch_reg_num() const {
    assert(m_reg_num_valid); /* [한국어] 레지스터 번호가 설정된 경우에만 반환 */
    return m_arch_reg_num;
  }
  /* [한국어] print_info() - 심볼 정보를 FILE*에 출력 (UID, 위치, 속성 플래그) */
  void print_info(FILE *fp) const;
  /* [한국어] uid() - 이 심볼의 전역 고유 ID 반환 (gpgpu_ctx->symbol_sm_next_uid에서 할당) */
  unsigned uid() const { return m_uid; }

 private:
  gpgpu_context *gpgpu_ctx;
  /* [한국어] gpgpu_ctx: 전역 시뮬레이터 컨텍스트 포인터.
   * get_uid()에서 gpgpu_ctx->symbol_sm_next_uid에 접근하기 위해 사용.
   * 파싱 스레드에서만 설정되며, 이후 읽기 전용. */

  unsigned get_uid();
  /* [한국어] get_uid() - 전역 고유 ID 발급. gpgpu_ctx->symbol_sm_next_uid를 원자적으로 증가.
   * 실제 구현: ptx_ir.cc의 symbol::get_uid() 참조. */

  unsigned m_uid;
  /* [한국어] m_uid: 이 심볼의 전역 고유 식별자.
   * 설정자: 생성자에서 get_uid() 호출로 발급.
   * 읽는 자: uid() 접근자, 디버그 출력(print_info).
   * 값 범위: 1 이상의 단조증가 정수. */

  const type_info *m_type;
  /* [한국어] m_type: 이 심볼의 타입 정보 포인터.
   * 설정자: 생성자 파라미터로 전달된 type_info* 저장 (NULL 가능).
   * 읽는 자: type(), is_reg(), is_shared() 등 판별 메서드들, instructions.cc.
   * 값 범위: 유효한 type_info* 또는 NULL (레이블/함수 심볼). */

  unsigned m_size;  // in bytes
  /* [한국어] m_size: 심볼의 크기 (바이트 단위).
   * 설정자: 생성자 파라미터 size에서 설정됨.
   * 읽는 자: get_size_in_bytes() 접근자; arg_buffer_t에서 파라미터 복사 크기 결정에 사용.
   * 값 범위: 레지스터 1-8바이트, 배열은 원소수×원소크기. */

  std::string m_name;
  /* [한국어] m_name: PTX 코드에서의 심볼 이름 문자열.
   * 예: "%r1"(레지스터), "__shared_data"(공유 변수), "$L_loop"(레이블), "vecAdd"(함수).
   * 설정자: 생성자에서 name 파라미터로 설정.
   * 읽는 자: name() 접근자; symbol_table::m_symbols 맵의 키로도 사용됨. */

  std::string m_decl_location;
  /* [한국어] m_decl_location: 심볼이 선언된 소스 파일 위치 ("file.ptx:42" 형식).
   * 설정자: 생성자에서 location 파라미터로 설정 (symbol_table::add_variable에서 snprintf로 생성).
   * 읽는 자: decl_location() 접근자, print_info() 디버그 출력.
   * 용도: 컴파일 오류/디버그 메시지에서 선언 위치 추적. */

  unsigned m_address;
  /* [한국어] m_address: 메모리 주소 또는 레이블의 PC 값.
   * 전역 변수: global_space 내 오프셋 (m_global_next에서 할당).
   * 공유 변수: shared_space 내 오프셋 (m_shared_next에서 할당).
   * 레이블: PTX 명령어의 PC 값 (m_instr_mem 인덱스).
   * 설정자: set_address() 또는 set_label_address()로 설정.
   * 읽는 자: get_address() 접근자; m_address_valid가 true인 경우에만 유효.
   * 값 범위: (unsigned)-1 (초기 미설정), 또는 유효한 주소/PC. */

  function_info *m_function;  // used for function symbols
  /* [한국어] m_function: 함수 심볼의 경우 해당 function_info 객체 포인터.
   * 비함수 심볼: NULL.
   * 설정자: set_function()에서 설정 (symbol_table::add_function()이 호출).
   * 읽는 자: get_pc()로 접근; CALL_OP 실행 시 target_func를 얻는 데 사용. */

  bool m_address_valid;
  /* [한국어] m_address_valid: m_address 필드가 유효한지 여부.
   * 설정자: set_address() 또는 set_label_address()에서 true로 설정.
   * 읽는 자: has_valid_address(), get_address()의 assert 조건.
   * 초기값: false (주소 미할당). */

  bool m_is_label;
  /* [한국어] m_is_label: 이 심볼이 분기 레이블인지 여부.
   * 설정자: set_label_address()에서 true로 설정됨.
   * 읽는 자: is_label() 접근자; ptx_instruction::is_label() 구현에 사용. */

  bool m_is_shared;
  /* [한국어] m_is_shared: shared_space에 속하는 심볼인지 여부 (공유 메모리 변수).
   * 설정자: 생성자에서 type->get_key().is_shared() 로 초기화.
   * 읽는 자: operand_info::is_shared()가 이 플래그를 통해 공유 메모리 접근 판별. */

  bool m_is_sstarr;
  /* [한국어] m_is_sstarr: stack-stored array(sstarr) 여부.
   * 공유 메모리 내에서 스택처럼 사용되는 배열을 표시하는 플래그.
   * 설정자: (생성자에서 초기화되지 않음 — 주의: 미초기화 버그 가능성 있음).
   * 읽는 자: is_sstarr() 접근자. */

  bool m_is_const;
  /* [한국어] m_is_const: 상수 메모리(.const)에 속하는 심볼인지 여부.
   * GPU에서 읽기 전용, 캐시 가속되는 상수 메모리 변수 식별.
   * 설정자: 생성자에서 type->get_key().is_const()로 초기화. */

  bool m_is_global;
  /* [한국어] m_is_global: 전역 메모리(.global)에 속하는 심볼인지 여부.
   * GPU 전역 메모리(DRAM을 통해 접근)에 저장되는 변수.
   * 설정자: 생성자에서 type->get_key().is_global()로 초기화. */

  bool m_is_local;
  /* [한국어] m_is_local: 로컬 메모리(.local)에 속하는 심볼인지 여부.
   * 스레드 개별 스택 영역 — 레지스터 spillover 또는 로컬 배열에 사용.
   * 설정자: 생성자에서 type->get_key().is_local()로 초기화. */

  bool m_is_param_local;
  /* [한국어] m_is_param_local: 로컬 파라미터 공간(.param, local)에 속하는 심볼인지 여부.
   * 디바이스 함수 내부에서 선언되는 .param 변수 (커널 파라미터와 구분).
   * 설정자: 생성자에서 type->get_key().is_param_local()로 초기화. */

  bool m_is_param_kernel;
  /* [한국어] m_is_param_kernel: 커널 파라미터 공간(.param, kernel)에 속하는 심볼인지 여부.
   * cuLaunchKernel()을 통해 전달되는 커널 인수 — PTX에서 .param 공간으로 접근.
   * 설정자: 생성자에서 type->get_key().is_param_kernel()로 초기화. */

  bool m_is_tex;
  /* [한국어] m_is_tex: 텍스처 메모리(.tex) 심볼인지 여부.
   * 텍스처 참조(texture reference) 심볼 식별.
   * 설정자: 생성자에서 type->get_key().is_tex()로 초기화. */

  bool m_is_func_addr;
  /* [한국어] m_is_func_addr: 함수 주소 심볼인지 여부.
   * set_function()으로 설정되며, CALL_OP에서 대상 함수를 찾을 때 사용됨.
   * 설정자: 생성자에서 type->get_key().is_func_addr()로 초기화, set_function()으로도 설정. */

  unsigned m_reg_num;
  /* [한국어] m_reg_num: 심볼 테이블 레지스터 번호 (PTX 레지스터 할당 순서 인덱스).
   * 설정자: set_regno()에서 symbol_table::next_reg_num()이 발급한 번호로 설정.
   * 읽는 자: reg_num() 접근자; instructions.cc에서 레지스터 파일 접근 시 사용.
   * 초기값: (unsigned)-1 (미설정). */

  unsigned m_arch_reg_num;
  /* [한국어] m_arch_reg_num: 아키텍처 레지스터 번호 (타이밍 모델의 물리 레지스터 인덱스).
   * 설정자: set_regno()에서 arch_regno 파라미터로 설정됨.
   * 읽는 자: arch_reg_num() 접근자; shader.cc의 scoreboard가 이 번호로 WAW/RAW 해저드 추적.
   * 초기값: (unsigned)-1 (미설정). */

  bool m_reg_num_valid;
  /* [한국어] m_reg_num_valid: m_reg_num과 m_arch_reg_num이 유효하게 설정되었는지 여부.
   * 설정자: set_regno()에서 true로 설정.
   * 읽는 자: reg_num() 및 arch_reg_num()의 assert 조건.
   * 초기값: false (레지스터 번호 미할당). */

  std::list<operand_info> m_initializer;
  /* [한국어] m_initializer: 전역/상수 변수의 초기화 값 리스트.
   * .global .s32 arr[] = {1, 2, 3} 같은 초기화 데이터를 operand_info 리스트로 저장.
   * 설정자: add_initializer()를 통해 파서가 설정.
   * 읽는 자: get_initializer() 접근자; function_info::finalize()에서 메모리 초기화에 사용.
   * 동기화: 파싱 단계에서만 설정되므로 락 불필요. */
};

/*
 * [한국어]
 * symbol_table - PTX 스코프별 심볼 테이블 클래스
 *
 * PTX 모듈/함수 단위의 심볼(변수, 레이블, 함수)을 이름으로 등록하고 조회하는 테이블이다.
 * 계층 구조(부모-자식)를 지원하여, 자식 테이블에서 조회 실패 시 부모 테이블로 위임한다.
 * 최상위 테이블은 모듈 전역 스코프, 자식 테이블은 각 함수의 로컬 스코프를 나타낸다.
 *
 * 또한 공유/전역/로컬/상수/텍스처 메모리 공간의 다음 빈 주소(next offset)를 추적하는
 * 주소 할당기(allocator) 역할을 겸한다. PTX 파서가 변수 선언을 처리할 때 이 테이블에
 * 심볼을 등록하고 해당 공간에서 적절한 오프셋을 할당한다.
 *
 * 설정자: ptx.y(파서), gpgpu_ptx_assemble()이 커널 파싱 중 변수/레이블/함수 추가.
 * 읽는 자: cuda-sim/ptx_sim.cc, instructions.cc가 시뮬레이션 중 심볼을 조회.
 * 동기화: 파싱은 단일 스레드, 시뮬레이션은 읽기 전용이므로 락 불필요.
 *
 * 호출 체인:
 *   ptx_parser(파싱) → add_variable/add_function → [m_symbols 맵 등록]
 *   instructions.cc(실행) → lookup() → [m_symbols 조회 → 부모로 재귀]
 */
class symbol_table {
 public:
  /* [한국어] 기본 생성자 — assert(0)으로 직접 사용 금지. 반드시 매개변수 버전 사용 */
  symbol_table();
  /*
   * [한국어]
   * symbol_table(scope_name, entry_point, parent, ctx) - 유효 생성자
   *
   * @scope_name: 이 테이블의 스코프 이름 (함수명 또는 모듈명)
   * @entry_point: 스코프 종류 (1=커널 진입점, 2=extern 함수, 3=CDP 명령어 그룹)
   * @parent: 부모 심볼 테이블 (모듈 전역 스코프이면 NULL)
   * @ctx: 전역 GPGPU 컨텍스트
   *
   * 부모가 있으면 부모의 shared/global 주소 카운터를 물려받는다.
   */
  symbol_table(const char *scope_name, unsigned entry_point,
               symbol_table *parent, gpgpu_context *ctx);
  /* [한국어] set_name() - 테이블의 스코프 이름 변경 (파서에서 함수명 확정 후 호출) */
  void set_name(const char *name);
  /* [한국어] get_ptx_version() - 이 스코프의 PTX 버전 반환 (부모가 있으면 부모로 재귀) */
  const ptx_version &get_ptx_version() const;
  /* [한국어] get_sm_target() - SM(sm_XX) 타겟 아키텍처 버전 반환 */
  unsigned get_sm_target() const;
  /* [한국어] set_ptx_version() - PTX 버전 설정 (.version 디렉티브 파싱 시 호출) */
  void set_ptx_version(float ver, unsigned ext);
  /* [한국어] set_sm_target() - SM 타겟 설정 (.target sm_75 등 파싱 시 호출) */
  void set_sm_target(const char *target, const char *ext, const char *ext2);
  /*
   * [한국어]
   * lookup() - 이름으로 심볼 조회 (재귀적으로 부모 테이블까지 탐색)
   *
   * @identifier: 조회할 심볼 이름 문자열
   * @return: 찾은 symbol* 또는 NULL (최상위 테이블에서도 없으면 NULL)
   *
   * 호출 체인:
   *   instructions.cc → symbol_table::lookup() → [m_symbols.find] → (없으면) m_parent->lookup()
   */
  symbol *lookup(const char *identifier);
  /*
   * [한국어]
   * lookup_by_addr() - 메모리 주소로 심볼 조회
   *
   * @addr: 조회할 메모리 주소
   * @return: 해당 주소를 가진 symbol* 또는 NULL
   * 레지스터가 아니고 주소가 유효한 심볼 중에서 일치하는 것을 반환한다.
   */
  symbol *lookup_by_addr(addr_t addr);
  /* [한국어] get_scope_name() - 이 테이블의 스코프 이름 반환 */
  std::string get_scope_name() const { return m_scope_name; }
  /*
   * [한국어]
   * add_variable() - 새 변수 심볼을 이 테이블에 등록
   *
   * @identifier: 변수 이름
   * @type: 변수 타입 (type_info*)
   * @size: 크기 (바이트)
   * @filename, @line: 선언 위치
   * @return: 생성된 symbol*
   *
   * 전역/상수 변수는 m_globals, m_consts 리스트에도 추가됨.
   */
  symbol *add_variable(const char *identifier, const type_info *type,
                       unsigned size, const char *filename, unsigned line);
  /*
   * [한국어]
   * add_function() - 함수 심볼을 이 테이블에 등록
   *
   * @func: 등록할 function_info*
   * @filename, @linenumber: 선언 위치
   * 이미 동일 이름 심볼이 있으면 등록하지 않고 반환 (중복 방지).
   */
  void add_function(function_info *func, const char *filename,
                    unsigned linenumber);
  /*
   * [한국어]
   * add_function_decl() - 함수 선언 등록 및 심볼 테이블 생성
   *
   * @name: 함수 이름
   * @entry_point: 커널 여부 (1=커널, 0=디바이스 함수, 2=extern)
   * @func_info: [out] 생성되거나 기존 function_info*
   * @sym_table: [out] 이 함수의 로컬 심볼 테이블*
   * @return: true이면 이미 선언된 함수 (prior_decl)
   *
   * 함수 최초 선언 시: function_info, symbol_table을 새로 생성하고 "_" 더미 레지스터 추가.
   * 재선언 시: 기존 객체 반환. register_ptx_function()도 호출.
   *
   * 호출 체인:
   *   ptx.y(함수 선언 파싱) → add_function_decl() → new function_info / new symbol_table
   */
  bool add_function_decl(const char *name, int entry_point,
                         function_info **func_info,
                         symbol_table **symbol_table);
  /*
   * [한국어]
   * lookup_function() - 함수 이름으로 function_info* 조회
   *
   * @name: 함수 이름
   * @return: 해당 함수의 function_info* (없으면 assert 실패)
   */
  function_info *lookup_function(std::string name);
  /*
   * [한국어]
   * add_type(space, scalar, vector, align, extern) - 새 타입 정보 생성 및 등록
   *
   * @return: 생성된 type_info* (m_types 맵에 저장되지 않고 heap에 할당만 함 — 주석 참조)
   * param_space_unclassified는 param_space_local로 변환된다.
   */
  type_info *add_type(memory_space_t space_spec, int scalar_type_spec,
                      int vector_spec, int alignment_spec, int extern_spec);
  /* [한국어] add_type(func) - 함수 주소 타입 생성 (is_function=1인 type_info 반환) */
  type_info *add_type(function_info *func);
  /*
   * [한국어]
   * get_array_type() - 기본 타입에 배열 차원을 추가한 타입 반환
   *
   * @base_type: 배열 원소 타입
   * @array_dim: 배열 원소 수
   * @return: 새로운 배열 타입 type_info*
   */
  type_info *get_array_type(type_info *base_type, unsigned array_dim);
  /* [한국어] set_label_address() - 레이블 심볼의 PC 주소 확정 (ptx_assemble 중 호출) */
  void set_label_address(const symbol *label, unsigned addr);
  /* [한국어] next_reg_num() - 새 레지스터 번호 발급 (m_reg_allocator 증가 후 반환) */
  unsigned next_reg_num() { return ++m_reg_allocator; }
  /* [한국어] 각 메모리 공간의 다음 빈 주소 반환 메서드들 */
  addr_t get_shared_next() { return m_shared_next; }  /* [한국어] 다음 공유 메모리 할당 주소 */
  addr_t get_sstarr_next() { return m_sstarr_next; }  /* [한국어] 다음 sstarr 할당 주소 */
  addr_t get_global_next() { return m_global_next; }  /* [한국어] 다음 전역 메모리 할당 주소 */
  addr_t get_local_next() { return m_local_next; }    /* [한국어] 다음 로컬 메모리 할당 주소 */
  addr_t get_tex_next() { return m_tex_next; }        /* [한국어] 다음 텍스처 메모리 할당 주소 */
  /* [한국어] 메모리 공간 할당 진행 메서드들 (num_bytes만큼 다음 주소 전진) */
  void alloc_shared(unsigned num_bytes) { m_shared_next += num_bytes; }  /* [한국어] 공유 메모리 할당 */
  void alloc_sstarr(unsigned num_bytes) { m_sstarr_next += num_bytes; }  /* [한국어] sstarr 할당 */
  void alloc_global(unsigned num_bytes) { m_global_next += num_bytes; }  /* [한국어] 전역 메모리 할당 */
  void alloc_local(unsigned num_bytes) { m_local_next += num_bytes; }    /* [한국어] 로컬 메모리 할당 */
  void alloc_tex(unsigned num_bytes) { m_tex_next += num_bytes; }        /* [한국어] 텍스처 메모리 할당 */

  typedef std::list<symbol *>::iterator iterator;
  /* [한국어] iterator: m_globals, m_consts 리스트 순회에 사용 */

  /* [한국어] 전역 변수 리스트 반복자 — function_info::finalize()에서 전역 메모리 초기화에 사용 */
  iterator global_iterator_begin() { return m_globals.begin(); }
  iterator global_iterator_end() { return m_globals.end(); }

  /* [한국어] 상수 변수 리스트 반복자 — 상수 메모리 초기화에 사용 */
  iterator const_iterator_begin() { return m_consts.begin(); }
  iterator const_iterator_end() { return m_consts.end(); }

  /* [한국어] dump() - 모든 심볼 정보를 stdout에 출력 (디버그용) */
  void dump();

  // Jin: handle instruction group for cdp
  /* [한국어] CDP(CUDA Dynamic Parallelism) 명령어 그룹 처리를 위한 메서드 */
  /* [한국어] start_inst_group() - 새 CDP 명령어 그룹 심볼 테이블 생성 및 활성화 */
  symbol_table *start_inst_group();
  /* [한국어] end_inst_group() - CDP 명령어 그룹 종료, 주소 카운터를 부모로 반환 */
  symbol_table *end_inst_group();

  // backward pointer
  /* [한국어] gpgpu_ctx: 전역 시뮬레이터 컨텍스트 포인터 (UID 발급, 전역 상태 접근) */
  class gpgpu_context *gpgpu_ctx;

 private:
  unsigned m_reg_allocator;
  /* [한국어] m_reg_allocator: 레지스터 번호 할당기 — next_reg_num()이 호출될 때마다 증가.
   * 각 함수 스코프별로 독립적으로 유지되며, 0부터 시작하여 레지스터 순서 번호를 발급.
   * 설정자: 생성자에서 0으로 초기화; next_reg_num()에서 전위 증가.
   * 읽는 자: symbol::m_reg_num으로 저장된 후 reg_num() 접근자에서 사용. */

  unsigned m_shared_next;
  /* [한국어] m_shared_next: 공유 메모리(.shared) 다음 빈 주소 오프셋.
   * 각 .shared 변수 선언 시 현재 값을 주소로 할당하고 크기만큼 전진.
   * 설정자: 생성자(0), alloc_shared(), CDP 그룹 동기화.
   * 읽는 자: add_variable() 이후 symbol::set_address()에 전달. */

  unsigned m_sstarr_next;
  /* [한국어] m_sstarr_next: sstarr(stack-stored array) 메모리 다음 빈 주소.
   * 공유 메모리 내 스택형 배열 영역을 위한 별도 할당기. */

  unsigned m_const_next;
  /* [한국어] m_const_next: 상수 메모리(.const) 다음 빈 주소 오프셋. */

  unsigned m_global_next;
  /* [한국어] m_global_next: 전역 메모리(.global) 다음 빈 주소 오프셋.
   * 초기값: 0x100 (예약 영역 이후부터 할당 시작; 생성자 참조).
   * 설정자: 생성자, alloc_global(), 부모 테이블 값 상속. */

  unsigned m_local_next;
  /* [한국어] m_local_next: 로컬 메모리(.local) 다음 빈 주소 오프셋.
   * 함수 스택 프레임 내 변수들의 오프셋 계산에 사용. */

  unsigned m_tex_next;
  /* [한국어] m_tex_next: 텍스처 메모리(.tex) 다음 빈 주소 오프셋. */

  symbol_table *m_parent;
  /* [한국어] m_parent: 부모 심볼 테이블 포인터.
   * 자식 테이블(함수 스코프)이 lookup() 실패 시 부모(모듈 스코프)로 위임.
   * 최상위 모듈 테이블은 NULL. */

  ptx_version m_ptx_version;
  /* [한국어] m_ptx_version: 이 모듈의 PTX 버전 정보 (.version 디렉티브).
   * 부모가 있으면 get_ptx_version()은 부모 버전을 반환 (버전은 모듈 전역). */

  std::string m_scope_name;
  /* [한국어] m_scope_name: 이 테이블의 스코프 이름 (함수명 또는 모듈명).
   * 디버그 출력(dump())과 CDP 그룹 이름 생성에 사용. */

  std::map<std::string, symbol *>
      m_symbols;  // map from name of register to pointers to the registers
  /* [한국어] m_symbols: 이름 → symbol* 매핑 — 이 스코프의 모든 심볼 저장.
   * 설정자: add_variable(), add_function()에서 새 심볼 등록.
   * 읽는 자: lookup(), lookup_by_addr()에서 조회.
   * 동기화: 파싱 단계 단일 스레드에서만 수정됨. */

  std::map<type_info_key, type_info *, type_info_key_compare> m_types;
  /* [한국어] m_types: 타입 키 → type_info* 매핑 — 타입 캐시.
   * 현재 코드에서는 get_array_type()의 TODO로 인해 실제로 사용되지 않음 (주석 참조).
   * 잠재적으로 타입 중복 생성 방지를 위한 캐시로 설계됨. */

  std::list<symbol *> m_globals;
  /* [한국어] m_globals: 전역 메모리(.global) 심볼 리스트.
   * add_variable()에서 전역 타입 심볼을 추가.
   * 읽는 자: global_iterator_begin/end()로 순회 — function_info::finalize()에서 초기화. */

  std::list<symbol *> m_consts;
  /* [한국어] m_consts: 상수 메모리(.const) 심볼 리스트.
   * add_variable()에서 상수 타입 심볼을 추가.
   * 읽는 자: const_iterator_begin/end()로 순회 — 상수 메모리 초기화에 사용. */

  std::map<std::string, function_info *> m_function_info_lookup;
  /* [한국어] m_function_info_lookup: 함수 이름 → function_info* 매핑.
   * 설정자: add_function_decl()에서 새 함수 등록.
   * 읽는 자: lookup_function()에서 함수 이름으로 function_info 조회. */

  std::map<std::string, symbol_table *> m_function_symtab_lookup;
  /* [한국어] m_function_symtab_lookup: 함수 이름 → 해당 함수의 심볼 테이블* 매핑.
   * 함수 선언 시 생성되는 함수 로컬 스코프 테이블을 저장.
   * 설정자: add_function_decl()에서 새 symbol_table 생성 후 등록. */

  // Jin: handle instruction group for cdp
  unsigned m_inst_group_id;
  /* [한국어] m_inst_group_id: CDP 명령어 그룹의 고유 ID 카운터.
   * start_inst_group()에서 그룹 이름 생성 시 사용, end_inst_group()에서 증가.
   * CDP(CUDA Dynamic Parallelism): GPU 커널에서 다른 커널을 동적으로 런치하는 기능. */

  std::map<std::string, symbol_table *> m_inst_group_symtab;
  /* [한국어] m_inst_group_symtab: CDP 명령어 그룹 이름 → 그룹별 심볼 테이블* 매핑.
   * start_inst_group()에서 새 그룹 테이블 생성 및 저장. */
};

/*
 * [한국어]
 * operand_info - PTX 명령어의 하나의 피연산자를 표현하는 클래스
 *
 * PTX 명령어의 소스/목적 피연산자를 표현한다. 하나의 피연산자는 레지스터, 즉값(리터럴),
 * 메모리 주소 표현식, 레이블, 벡터 레지스터 묶음, 내장 변수 등 다양한 형태일 수 있으며,
 * m_type(operand_type)으로 종류를 구분하고 union m_value에 실제 값을 저장한다.
 *
 * 설정자: ptx.y 파서가 명령어 피연산자를 파싱하면서 각 생성자를 호출하여 생성.
 *         ptx_instruction 생성자에서 m_operands 벡터에 복사됨.
 * 읽는 자: instructions.cc의 각 명령어 구현 함수가 dst(), src1(), src2() 등을 통해 접근.
 *          shader.cc에서 warp_inst_t 레벨로 변환 시 참조.
 * 동기화: 파싱 단계에서 생성 후 시뮬레이션 단계에서는 읽기 전용.
 *
 * 호출 체인:
 *   ptx.y(파서 액션) → operand_info 생성 → ptx_instruction::m_operands에 저장
 *   instructions.cc(실행) → ptx_instruction::src1() → [m_operands[1]] → operand_info 접근
 */
class operand_info {
 public:
  /*
   * [한국어]
   * operand_info(ctx) - 빈(무효) 피연산자 생성자
   *
   * @ctx: 전역 GPGPU 컨텍스트 (UID 발급용)
   * m_valid=false인 무효 피연산자 객체를 생성한다.
   * ptx_instruction의 m_return_var 초기화 등 "피연산자 없음" 상태를 나타낼 때 사용.
   */
  operand_info(gpgpu_context *ctx) {
    init(ctx);                         /* [한국어] 모든 필드를 기본값으로 초기화 */
    m_is_non_arch_reg = false;         /* [한국어] 비아키텍처 레지스터 아님 */
    m_addr_space = undefined_space;    /* [한국어] 메모리 공간 미지정 */
    m_operand_lohi = 0;                /* [한국어] lo/hi 분할 피연산자 아님 */
    m_double_operand_type = 0;         /* [한국어] 더블 타입 피연산자 아님 */
    m_operand_neg = false;             /* [한국어] 부호 반전 없음 */
    m_const_mem_offset = 0;            /* [한국어] 상수 메모리 오프셋 초기화 */
    m_uid = get_uid();                 /* [한국어] 고유 ID 발급 */
    m_valid = false;                   /* [한국어] 유효하지 않은 피연산자로 표시 */
    m_immediate_address = false;       /* [한국어] 즉시 주소 아님 */
    m_addr_offset = 0;                 /* [한국어] 주소 오프셋 0 */
    m_value.m_symbolic = NULL;         /* [한국어] 심볼 포인터 NULL 초기화 */
  }
  /*
   * [한국어]
   * operand_info(addr, ctx) - 심볼 기반 피연산자 생성자
   *
   * @addr: 이 피연산자가 참조하는 symbol* (레지스터, 변수, 레이블 등)
   * @ctx: 전역 GPGPU 컨텍스트
   *
   * addr의 속성(is_label, is_shared, is_reg 등)을 검사하여 m_type을 결정한다.
   * 레지스터는 reg_t, 그 외 대부분은 symbolic_t, 레이블은 label_t.
   * 파서에서 가장 자주 사용되는 생성자 — 레지스터명/변수명/레이블 파싱 시 호출.
   */
  operand_info(const symbol *addr, gpgpu_context *ctx) {
    init(ctx);                         /* [한국어] 모든 필드를 기본값으로 초기화 */
    m_is_non_arch_reg = false;         /* [한국어] 비아키텍처 레지스터 여부 초기화 */
    m_addr_space = undefined_space;    /* [한국어] 메모리 공간 초기화 */
    m_operand_lohi = 0;                /* [한국어] lo/hi 분할 초기화 */
    m_double_operand_type = 0;         /* [한국어] 더블 타입 초기화 */
    m_operand_neg = false;             /* [한국어] 부호 반전 초기화 */
    m_const_mem_offset = 0;            /* [한국어] 상수 메모리 오프셋 초기화 */
    m_uid = get_uid();                 /* [한국어] 고유 ID 발급 */
    m_valid = true;                    /* [한국어] 유효한 피연산자로 표시 */
    if (addr->is_label()) {            /* [한국어] 레이블 심볼이면 label_t로 분류 */
      m_type = label_t;
    } else if (addr->is_shared()) {    /* [한국어] 공유 메모리 변수이면 symbolic_t */
      m_type = symbolic_t;
    } else if (addr->is_const()) {     /* [한국어] 상수 메모리 변수이면 symbolic_t */
      m_type = symbolic_t;
    } else if (addr->is_global()) {    /* [한국어] 전역 메모리 변수이면 symbolic_t */
      m_type = symbolic_t;
    } else if (addr->is_local()) {     /* [한국어] 로컬 메모리 변수이면 symbolic_t */
      m_type = symbolic_t;
    } else if (addr->is_param_local()) { /* [한국어] 로컬 파라미터이면 symbolic_t */
      m_type = symbolic_t;
    } else if (addr->is_param_kernel()) { /* [한국어] 커널 파라미터이면 symbolic_t */
      m_type = symbolic_t;
    } else if (addr->is_tex()) {       /* [한국어] 텍스처 심볼이면 symbolic_t */
      m_type = symbolic_t;
    } else if (addr->is_func_addr()) { /* [한국어] 함수 주소이면 symbolic_t */
      m_type = symbolic_t;
    } else if (!addr->is_reg()) {      /* [한국어] 레지스터가 아닌 나머지도 symbolic_t */
      m_type = symbolic_t;
    } else {
      m_type = reg_t;                  /* [한국어] 레지스터 공간 심볼이면 reg_t */
    }

    m_is_non_arch_reg = addr->is_non_arch_reg(); /* [한국어] "_" 레지스터 여부 설정 */
    m_value.m_symbolic = addr;         /* [한국어] 심볼 포인터 저장 */
    m_addr_offset = 0;                 /* [한국어] 심볼 피연산자의 오프셋은 0 */
    m_vector = false;                  /* [한국어] 벡터 피연산자 아님 */
    m_neg_pred = false;                /* [한국어] 술어 부정 없음 */
    m_is_return_var = false;           /* [한국어] 반환 변수 아님 */
    m_immediate_address = false;       /* [한국어] 즉시 주소 아님 */
  }
  /*
   * [한국어]
   * operand_info(addr1, addr2, ctx) - 두 심볼로 구성된 메모리 피연산자 생성자
   *
   * @addr1, @addr2: 메모리 주소를 구성하는 두 심볼 (주로 PTXPlus 확장에서 사용)
   * m_type = memory_t로 설정하고 m_value.m_vector_symbolic[0..1]에 두 심볼을 저장.
   * 나머지 슬롯은 NULL로 채워 벡터 길이를 2로 명시.
   */
  operand_info(const symbol *addr1, const symbol *addr2, gpgpu_context *ctx) {
    init(ctx);                                          /* [한국어] 필드 기본값 초기화 */
    m_is_non_arch_reg = false;                          /* [한국어] 비아키텍처 레지스터 아님 */
    m_addr_space = undefined_space;                     /* [한국어] 메모리 공간 초기화 */
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = 0;
    m_uid = get_uid();                                  /* [한국어] 고유 ID 발급 */
    m_valid = true;                                     /* [한국어] 유효 피연산자 */
    m_type = memory_t;                                  /* [한국어] 메모리 피연산자 타입 */
    m_value.m_vector_symbolic = new const symbol *[8];  /* [한국어] 8개 슬롯 배열 동적 할당 */
    m_value.m_vector_symbolic[0] = addr1;               /* [한국어] 첫 번째 심볼 저장 */
    m_value.m_vector_symbolic[1] = addr2;               /* [한국어] 두 번째 심볼 저장 */
    m_value.m_vector_symbolic[2] = NULL;                /* [한국어] 나머지 슬롯 NULL — 길이 표시 */
    m_value.m_vector_symbolic[3] = NULL;
    m_value.m_vector_symbolic[4] = NULL;
    m_value.m_vector_symbolic[5] = NULL;
    m_value.m_vector_symbolic[6] = NULL;
    m_value.m_vector_symbolic[7] = NULL;
    m_addr_offset = 0;
    m_vector = false;                                   /* [한국어] 일반 메모리 피연산자이므로 벡터 아님 */
    m_neg_pred = false;
    m_is_return_var = false;
    m_immediate_address = false;
  }
  /*
   * [한국어]
   * operand_info(builtin_id, dim_mod, ctx) - 내장 변수 피연산자 생성자
   *
   * @builtin_id: 내장 변수 종류 (예: TIDX, TIDY, CTAIDX 등 파서 토큰)
   * @dim_mod: 차원 수정자 (예: .x=0, .y=1, .z=2)
   *
   * %tid.x, %ctaid.y 등 CUDA 내장 변수(threadIdx, blockIdx 등)를 표현.
   * m_type = builtin_t, m_value.m_int = builtin_id, m_addr_offset = dim_mod.
   * 실행 시 ptx_thread_info에서 실제 스레드/블록 ID 값으로 치환됨.
   */
  operand_info(int builtin_id, int dim_mod, gpgpu_context *ctx) {
    init(ctx);                         /* [한국어] 필드 기본값 초기화 */
    m_is_non_arch_reg = false;
    m_addr_space = undefined_space;
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = 0;
    m_uid = get_uid();                 /* [한국어] 고유 ID 발급 */
    m_valid = true;                    /* [한국어] 유효 피연산자 */
    m_vector = false;                  /* [한국어] 벡터 아님 */
    m_type = builtin_t;                /* [한국어] 내장 변수 타입 설정 */
    m_value.m_int = builtin_id;        /* [한국어] 내장 변수 ID 저장 (TIDX, CTAIDX 등) */
    m_addr_offset = dim_mod;           /* [한국어] 차원 수정자(.x/.y/.z) 오프셋으로 저장 */
    m_neg_pred = false;
    m_is_return_var = false;
    m_immediate_address = false;
  }
  /*
   * [한국어]
   * operand_info(addr, offset, ctx) - 심볼+오프셋 주소 피연산자 생성자
   *
   * @addr: 베이스 심볼 (공유/전역/로컬 변수 또는 레지스터)
   * @offset: 바이트 오프셋 (예: [%r1+8]에서 8)
   *
   * ld.global.s32 %r0, [addr+offset] 형식의 메모리 접근 피연산자를 표현.
   * m_type = address_t, m_value.m_symbolic = addr, m_addr_offset = offset.
   */
  operand_info(const symbol *addr, int offset, gpgpu_context *ctx) {
    init(ctx);                         /* [한국어] 필드 기본값 초기화 */
    m_is_non_arch_reg = false;
    m_addr_space = undefined_space;
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = 0;
    m_uid = get_uid();                 /* [한국어] 고유 ID 발급 */
    m_valid = true;                    /* [한국어] 유효 피연산자 */
    m_vector = false;
    m_type = address_t;                /* [한국어] 주소+오프셋 타입 */
    m_value.m_symbolic = addr;         /* [한국어] 베이스 심볼 저장 */
    m_addr_offset = offset;            /* [한국어] 바이트 오프셋 저장 */
    m_neg_pred = false;
    m_is_return_var = false;
    m_immediate_address = false;       /* [한국어] 즉시 주소 아님 — 심볼 기반 주소 */
  }
  /*
   * [한국어]
   * operand_info(unsigned x, ctx) — unsigned 즉시 주소 피연산자 생성자
   *
   * @x:   unsigned 즉시값 (주소 오프셋으로도 사용됨 — m_addr_offset = x)
   * @ctx: gpgpu_context 역방향 포인터
   * @return: 없음 (생성자)
   *
   * PTX 명령어에서 unsigned 리터럴이 즉시 주소(immediate address)로 사용될 때
   * 호출된다. 예: ld.global.u32 %r0, [0x1000] 형태에서 0x1000.
   * m_immediate_address = true 로 설정되는 것이 다른 리터럴 생성자와의 차이점.
   * m_addr_offset = x 도 함께 설정하여 주소 오프셋으로도 참조할 수 있게 한다.
   *
   * 호출 체인:
   *   PTX 파서 (ptx.y) → [이 생성자] → init()
   */
  operand_info(unsigned x, gpgpu_context *ctx) {
    init(ctx);                         // [한국어] 모든 필드를 안전한 기본값으로 초기화
    m_is_non_arch_reg = false;         // [한국어] 아키텍처 레지스터 여부: 일반 피연산자이므로 false
    m_addr_space = undefined_space;    // [한국어] 주소 공간: 즉시값이므로 미정의
    m_operand_lohi = 0;                // [한국어] lo/hi 분리 사용 없음
    m_double_operand_type = 0;         // [한국어] double 피연산자 특수 타입 없음
    m_operand_neg = false;             // [한국어] 부호 반전 없음
    m_const_mem_offset = 0;            // [한국어] 상수 메모리 오프셋 없음
    m_uid = get_uid();                 // [한국어] 전역 카운터에서 고유 ID 발급
    m_valid = true;                    // [한국어] 유효한 피연산자로 표시
    m_vector = false;                  // [한국어] 벡터 피연산자 아님
    m_type = unsigned_t;               // [한국어] 타입: unsigned 리터럴
    m_value.m_unsigned = x;            // [한국어] 리터럴 값 저장 (union의 unsigned 필드)
    m_addr_offset = x;                 // [한국어] 즉시 주소로도 사용 — 오프셋에 동일 값 저장
    m_neg_pred = false;                // [한국어] 부정 프레디케이트 없음
    m_is_return_var = false;           // [한국어] 반환 변수 아님
    m_immediate_address = true;        // [한국어] 즉시 주소 플래그 — 다른 리터럴 생성자와의 핵심 차이
  }
  /*
   * [한국어]
   * operand_info(int x, ctx) — signed 정수 리터럴 피연산자 생성자
   *
   * @x:   signed 정수 리터럴 값 (예: PTX에서 -5, 42 등)
   * @ctx: gpgpu_context 역방향 포인터
   * @return: 없음 (생성자)
   *
   * PTX 명령어에서 signed 정수 즉시값(immediate value)을 표현할 때 호출된다.
   * 예: add.s32 %r0, %r1, 42 에서 42.
   * m_immediate_address = false — 즉시 주소가 아니라 단순 정수 리터럴임.
   * m_addr_offset = 0 으로 설정하여 주소 오프셋으로는 사용하지 않음.
   *
   * 호출 체인:
   *   PTX 파서 (ptx.y) → [이 생성자] → init()
   */
  operand_info(int x, gpgpu_context *ctx) {
    init(ctx);                         // [한국어] 모든 필드를 안전한 기본값으로 초기화
    m_is_non_arch_reg = false;         // [한국어] 아키텍처 레지스터 여부: 일반 피연산자이므로 false
    m_addr_space = undefined_space;    // [한국어] 주소 공간: 즉시값이므로 미정의
    m_operand_lohi = 0;                // [한국어] lo/hi 분리 사용 없음
    m_double_operand_type = 0;         // [한국어] double 피연산자 특수 타입 없음
    m_operand_neg = false;             // [한국어] 부호 반전 없음
    m_const_mem_offset = 0;            // [한국어] 상수 메모리 오프셋 없음
    m_uid = get_uid();                 // [한국어] 전역 카운터에서 고유 ID 발급
    m_valid = true;                    // [한국어] 유효한 피연산자로 표시
    m_vector = false;                  // [한국어] 벡터 피연산자 아님
    m_type = int_t;                    // [한국어] 타입: signed 정수 리터럴
    m_value.m_int = x;                 // [한국어] signed 정수 리터럴 값 저장 (union의 int 필드)
    m_addr_offset = 0;                 // [한국어] 주소 오프셋 없음 — 단순 리터럴
    m_neg_pred = false;                // [한국어] 부정 프레디케이트 없음
    m_is_return_var = false;           // [한국어] 반환 변수 아님
    m_immediate_address = false;       // [한국어] 즉시 주소 아님 — 단순 signed 정수 리터럴
  }
  /*
   * [한국어]
   * operand_info(float x, ctx) — float 리터럴 피연산자 생성자
   *
   * @x:   float 즉시값 (예: PTX에서 0f3F800000 → 1.0f)
   * @ctx: gpgpu_context 역방향 포인터
   * @return: 없음 (생성자)
   *
   * PTX 명령어에서 32비트 부동소수점 리터럴을 표현할 때 호출된다.
   * 예: mul.f32 %f0, %f1, 0f3F800000(=1.0f).
   * m_type = float_op_t, m_value.m_float 에 값을 저장.
   * is_literal() 검사를 통과하므로 get_literal_value()로 ptx_reg_t로 변환 가능.
   *
   * 호출 체인:
   *   PTX 파서 (ptx.y) → [이 생성자] → init()
   */
  operand_info(float x, gpgpu_context *ctx) {
    init(ctx);                         // [한국어] 모든 필드를 안전한 기본값으로 초기화
    m_is_non_arch_reg = false;         // [한국어] 아키텍처 레지스터 여부: 일반 피연산자이므로 false
    m_addr_space = undefined_space;    // [한국어] 주소 공간: 즉시값이므로 미정의
    m_operand_lohi = 0;                // [한국어] lo/hi 분리 사용 없음
    m_double_operand_type = 0;         // [한국어] double 피연산자 특수 타입 없음
    m_operand_neg = false;             // [한국어] 부호 반전 없음
    m_const_mem_offset = 0;            // [한국어] 상수 메모리 오프셋 없음
    m_uid = get_uid();                 // [한국어] 전역 카운터에서 고유 ID 발급
    m_valid = true;                    // [한국어] 유효한 피연산자로 표시
    m_vector = false;                  // [한국어] 벡터 피연산자 아님
    m_type = float_op_t;               // [한국어] 타입: 32비트 float 리터럴
    m_value.m_float = x;               // [한국어] float 리터럴 값 저장 (union의 float 필드)
    m_addr_offset = 0;                 // [한국어] 주소 오프셋 없음
    m_neg_pred = false;                // [한국어] 부정 프레디케이트 없음
    m_is_return_var = false;           // [한국어] 반환 변수 아님
    m_immediate_address = false;       // [한국어] 즉시 주소 아님 — 단순 float 리터럴
  }
  /*
   * [한국어]
   * operand_info(double x, ctx) — double 리터럴 피연산자 생성자
   *
   * @x:   double 즉시값 (예: PTX에서 0d3FF0000000000000 → 1.0)
   * @ctx: gpgpu_context 역방향 포인터
   * @return: 없음 (생성자)
   *
   * PTX 명령어에서 64비트 배정도 부동소수점 리터럴을 표현할 때 호출된다.
   * 예: mul.f64 %fd0, %fd1, 0d3FF0000000000000(=1.0).
   * m_type = double_op_t, m_value.m_double 에 값을 저장.
   * is_literal()을 통과하며, get_literal_value()에서 result.f64 로 반환.
   *
   * 호출 체인:
   *   PTX 파서 (ptx.y) → [이 생성자] → init()
   */
  operand_info(double x, gpgpu_context *ctx) {
    init(ctx);                         // [한국어] 모든 필드를 안전한 기본값으로 초기화
    m_is_non_arch_reg = false;         // [한국어] 아키텍처 레지스터 여부: 일반 피연산자이므로 false
    m_addr_space = undefined_space;    // [한국어] 주소 공간: 즉시값이므로 미정의
    m_operand_lohi = 0;                // [한국어] lo/hi 분리 사용 없음
    m_double_operand_type = 0;         // [한국어] double 피연산자 특수 타입 없음
    m_operand_neg = false;             // [한국어] 부호 반전 없음
    m_const_mem_offset = 0;            // [한국어] 상수 메모리 오프셋 없음
    m_uid = get_uid();                 // [한국어] 전역 카운터에서 고유 ID 발급
    m_valid = true;                    // [한국어] 유효한 피연산자로 표시
    m_vector = false;                  // [한국어] 벡터 피연산자 아님
    m_type = double_op_t;              // [한국어] 타입: 64비트 double 리터럴
    m_value.m_double = x;              // [한국어] double 리터럴 값 저장 (union의 double 필드)
    m_addr_offset = 0;                 // [한국어] 주소 오프셋 없음
    m_neg_pred = false;                // [한국어] 부정 프레디케이트 없음
    m_is_return_var = false;           // [한국어] 반환 변수 아님
    m_immediate_address = false;       // [한국어] 즉시 주소 아님 — 단순 double 리터럴
  }
  /*
   * [한국어]
   * operand_info(s1,s2,s3,s4, ctx) — 4-레지스터 벡터 피연산자 생성자
   *
   * @s1~s4: 벡터의 각 요소에 해당하는 심볼 포인터 (예: {%f0,%f1,%f2,%f3})
   * @ctx:   gpgpu_context 역방향 포인터
   * @return: 없음 (생성자)
   *
   * PTX의 ld.v4/st.v4 명령어에서 4개의 레지스터를 묶어 벡터 피연산자로
   * 표현할 때 호출된다. 예: ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%r0].
   * 내부적으로 8요소 배열(m_vector_symbolic[8])을 heap에 할당하고
   * 인덱스 0-3에 s1-s4를 저장, 인덱스 4-7은 NULL로 채운다.
   * get_vect_nelem()은 NULL이 처음 나타나는 위치를 스캔하여 4를 반환.
   *
   * 호출 체인:
   *   PTX 파서 (ptx.y) → [이 생성자] → init()
   */
  operand_info(const symbol *s1, const symbol *s2, const symbol *s3,
               const symbol *s4, gpgpu_context *ctx) {
    init(ctx);                                      // [한국어] 필드 기본값 초기화
    m_is_non_arch_reg = false;                      // [한국어] 아키텍처 레지스터 여부: 일반 피연산자이므로 false
    m_addr_space = undefined_space;                 // [한국어] 벡터 피연산자는 주소 공간 별도 지정
    m_operand_lohi = 0;                             // [한국어] lo/hi 분리 없음
    m_double_operand_type = 0;                      // [한국어] double 특수 타입 없음
    m_operand_neg = false;                          // [한국어] 부호 반전 없음
    m_const_mem_offset = 0;                         // [한국어] 상수 메모리 오프셋 없음
    m_uid = get_uid();                              // [한국어] 전역 카운터에서 고유 ID 발급
    m_valid = true;                                 // [한국어] 유효한 피연산자로 표시
    m_vector = true;                                // [한국어] 벡터 피연산자임을 명시
    m_type = vector_t;                              // [한국어] 타입: 다중 레지스터 벡터
    m_value.m_vector_symbolic = new const symbol *[8]; // [한국어] 8요소 포인터 배열 heap 할당 (v4/v8 공용)
    m_value.m_vector_symbolic[0] = s1;              // [한국어] 벡터 첫째 요소 레지스터
    m_value.m_vector_symbolic[1] = s2;              // [한국어] 벡터 둘째 요소 레지스터
    m_value.m_vector_symbolic[2] = s3;              // [한국어] 벡터 셋째 요소 레지스터
    m_value.m_vector_symbolic[3] = s4;              // [한국어] 벡터 넷째 요소 레지스터
    m_value.m_vector_symbolic[4] = NULL;            // [한국어] 5번째 이후는 NULL — v4이므로 요소 없음
    m_value.m_vector_symbolic[5] = NULL;            // [한국어] 6번째: NULL sentinel
    m_value.m_vector_symbolic[6] = NULL;            // [한국어] 7번째: NULL sentinel
    m_value.m_vector_symbolic[7] = NULL;            // [한국어] 8번째: NULL sentinel
    m_addr_offset = 0;                              // [한국어] 주소 오프셋 없음
    m_neg_pred = false;                             // [한국어] 부정 프레디케이트 없음
    m_is_return_var = false;                        // [한국어] 반환 변수 아님
    m_immediate_address = false;                    // [한국어] 즉시 주소 아님
  }
  /*
   * [한국어]
   * operand_info(s1..s8, ctx) — 8-레지스터 벡터 피연산자 생성자
   *
   * @s1~s8: 벡터의 각 요소에 해당하는 심볼 포인터 (예: {%f0..%f7})
   * @ctx:   gpgpu_context 역방향 포인터
   * @return: 없음 (생성자)
   *
   * PTX의 ld.v8/st.v8 명령어에서 8개 레지스터를 묶어 벡터로 표현할 때 호출된다.
   * WMMA(Warp Matrix Multiply-Accumulate) 명령어의 타일 레지스터 피연산자로도
   * 사용된다. 4-요소 생성자와 달리 모든 인덱스 0-7에 유효한 심볼 포인터를 저장.
   * get_vect_nelem()은 NULL sentinel을 만나지 않으므로 8을 반환.
   *
   * 호출 체인:
   *   PTX 파서 (ptx.y) → [이 생성자] → init()
   */
  operand_info(const symbol *s1, const symbol *s2, const symbol *s3,
               const symbol *s4, const symbol *s5, const symbol *s6,
               const symbol *s7, const symbol *s8, gpgpu_context *ctx) {
    init(ctx);                                      // [한국어] 필드 기본값 초기화
    m_is_non_arch_reg = false;                      // [한국어] 아키텍처 레지스터 여부: 일반 피연산자이므로 false
    m_addr_space = undefined_space;                 // [한국어] 벡터 피연산자는 주소 공간 별도 지정
    m_operand_lohi = 0;                             // [한국어] lo/hi 분리 없음
    m_double_operand_type = 0;                      // [한국어] double 특수 타입 없음
    m_operand_neg = false;                          // [한국어] 부호 반전 없음
    m_const_mem_offset = 0;                         // [한국어] 상수 메모리 오프셋 없음
    m_uid = get_uid();                              // [한국어] 전역 카운터에서 고유 ID 발급
    m_valid = true;                                 // [한국어] 유효한 피연산자로 표시
    m_vector = true;                                // [한국어] 벡터 피연산자임을 명시
    m_type = vector_t;                              // [한국어] 타입: 다중 레지스터 벡터
    m_value.m_vector_symbolic = new const symbol *[8]; // [한국어] 8요소 포인터 배열 heap 할당
    m_value.m_vector_symbolic[0] = s1;              // [한국어] 벡터 1번째 요소 레지스터
    m_value.m_vector_symbolic[1] = s2;              // [한국어] 벡터 2번째 요소 레지스터
    m_value.m_vector_symbolic[2] = s3;              // [한국어] 벡터 3번째 요소 레지스터
    m_value.m_vector_symbolic[3] = s4;              // [한국어] 벡터 4번째 요소 레지스터
    m_value.m_vector_symbolic[4] = s5;              // [한국어] 벡터 5번째 요소 레지스터 (v8 전용)
    m_value.m_vector_symbolic[5] = s6;              // [한국어] 벡터 6번째 요소 레지스터 (v8 전용)
    m_value.m_vector_symbolic[6] = s7;              // [한국어] 벡터 7번째 요소 레지스터 (v8 전용)
    m_value.m_vector_symbolic[7] = s8;              // [한국어] 벡터 8번째 요소 레지스터 (v8 전용)
    m_addr_offset = 0;                              // [한국어] 주소 오프셋 없음
    m_neg_pred = false;                             // [한국어] 부정 프레디케이트 없음
    m_is_return_var = false;                        // [한국어] 반환 변수 아님
    m_immediate_address = false;                    // [한국어] 즉시 주소 아님
  }

  /*
   * [한국어]
   * init() — operand_info 필드 전체를 안전한 기본값으로 초기화하는 내부 헬퍼
   *
   * @ctx: gpgpu_context 역방향 포인터 — 전역 상태(unique ID 카운터 등)에 접근
   * @return: 없음 (void)
   *
   * 모든 public 생성자가 가장 먼저 호출하는 내부 초기화 함수.
   * 각 생성자가 개별 필드를 덮어쓰기 전에 union m_value 전체와 모든 플래그를
   * 안전한 초기 상태로 설정한다. 이렇게 하면 생성자 코드가 필요한 필드만
   * 명시적으로 설정해도 나머지 필드는 정의된 값을 가진다.
   * m_uid = (unsigned)-1 은 "아직 ID 없음" 센티넬; 각 생성자가 get_uid()로 덮어씀.
   *
   * 호출 체인:
   *   모든 operand_info 생성자 → [init()] → (이후 각 생성자가 필드 덮어씀)
   */
  void init(gpgpu_context *ctx) {
    gpgpu_ctx = ctx;                       // [한국어] gpgpu_context 역방향 포인터 저장
    m_uid = (unsigned)-1;                  // [한국어] "아직 ID 없음" 센티넬 — 생성자가 get_uid()로 갱신
    m_valid = false;                       // [한국어] 아직 유효하지 않음 — 생성자가 true로 설정
    m_vector = false;                      // [한국어] 벡터 아님 — 벡터 생성자가 true로 설정
    m_type = undef_t;                      // [한국어] 타입 미정의 — 생성자가 적절한 타입으로 덮어씀
    m_immediate_address = false;           // [한국어] 즉시 주소 아님 — unsigned 생성자만 true로 설정
    m_addr_space = undefined_space;        // [한국어] 주소 공간 미정의
    m_operand_lohi = 0;                    // [한국어] lo/hi 선택자 0 (기본: 전체 값 사용)
    m_double_operand_type = 0;             // [한국어] double 피연산자 특수 분류 없음
    m_operand_neg = false;                 // [한국어] 값 부호 반전 없음
    m_const_mem_offset = (unsigned)-1;     // [한국어] 상수 메모리 오프셋 센티넬 (미설정 표시)
    m_value.m_int = 0;                     // [한국어] union의 int 필드 초기화
    m_value.m_unsigned = (unsigned)-1;     // [한국어] union의 unsigned 필드 초기화 (센티넬 값)
    m_value.m_float = 0;                   // [한국어] union의 float 필드 초기화
    m_value.m_double = 0;                  // [한국어] union의 double 필드 초기화
    for (unsigned i = 0; i < 4; i++) {     // [한국어] 벡터 스칼라 배열 4요소 순회 초기화
      m_value.m_vint[i] = 0;              // [한국어] 벡터 int 요소[i] 초기화
      m_value.m_vunsigned[i] = 0;         // [한국어] 벡터 unsigned 요소[i] 초기화
      m_value.m_vfloat[i] = 0;            // [한국어] 벡터 float 요소[i] 초기화
      m_value.m_vdouble[i] = 0;           // [한국어] 벡터 double 요소[i] 초기화
    }
    m_value.m_symbolic = NULL;             // [한국어] 심볼 포인터 NULL 초기화
    m_value.m_vector_symbolic = NULL;      // [한국어] 벡터 심볼 포인터 배열 NULL 초기화
    m_addr_offset = 0;                     // [한국어] 주소 오프셋 0으로 초기화
    m_neg_pred = 0;                        // [한국어] 부정 프레디케이트 없음
    m_is_return_var = 0;                   // [한국어] 반환 변수 아님
    m_is_non_arch_reg = 0;                 // [한국어] 비아키텍처 레지스터 아님
  }
  /*
   * [한국어]
   * make_memory_operand() — 피연산자를 메모리 피연산자(memory_t)로 변환
   *
   * @return: 없음 (void)
   *
   * 심볼 기반 피연산자가 PTX 파서에서 대괄호([])로 감싸인 메모리 참조임을
   * 확인했을 때 호출된다. 예: ld.global [%addr] 파싱 시 %addr 피연산자에 호출.
   * m_type 을 memory_t 로 변경하여 이후 is_memory_operand() 검사가 true를 반환.
   *
   * 호출 체인:
   *   PTX 파서 (ptx.y) → [make_memory_operand()] → (피연산자 타입 변경)
   */
  void make_memory_operand() { m_type = memory_t; }
  /*
   * [한국어]
   * set_return() — 이 피연산자를 함수 반환 변수로 표시
   *
   * @return: 없음 (void)
   *
   * PTX 함수 호출(CALL 명령어) 파싱 시 반환 값을 받을 목적지 피연산자에 호출된다.
   * m_is_return_var = true 로 설정하여 func_addr() 등에서 반환 변수를 건너뛸 수 있게 함.
   *
   * 호출 체인:
   *   PTX 파서 (ptx.y) → [set_return()] → (m_is_return_var 플래그 설정)
   */
  void set_return() { m_is_return_var = true; }
  /*
   * [한국어]
   * set_immediate_addr() — 피연산자를 즉시 주소로 표시
   *
   * @return: 없음 (void)
   *
   * PTX에서 mov 또는 주소 계산에 쓰이는 즉시 주소 피연산자임을 표시.
   * 파서가 unsigned 리터럴을 즉시 주소로 사용한다고 판단할 때 호출.
   * unsigned 생성자는 자체적으로 m_immediate_address = true 로 설정하지만,
   * 이 메서드는 파서가 나중에 추가 지정할 수 있도록 공개된 인터페이스.
   *
   * 호출 체인:
   *   PTX 파서 (ptx.y) → [set_immediate_addr()]
   */
  void set_immediate_addr() { m_immediate_address = true; }
  /*
   * [한국어]
   * name() — 심볼 기반 피연산자의 이름(문자열) 반환
   *
   * @return: 심볼 테이블에서 이 피연산자에 연결된 심볼의 이름 (std::string 참조)
   *
   * symbolic_t, reg_t, address_t, memory_t, label_t 타입 피연산자에만 유효.
   * 리터럴(int_t, float_op_t 등) 또는 벡터(vector_t) 피연산자에서 호출하면
   * assert로 중단된다. 디버그 출력, 심볼 테이블 검색 등에서 사용.
   *
   * 호출 체인:
   *   instructions.cc 명령어 실행 → [name()] → symbol::name()
   */
  const std::string &name() const {
    assert(m_type == symbolic_t || m_type == reg_t || m_type == address_t ||
           m_type == memory_t || m_type == label_t); // [한국어] 심볼 기반 타입만 name() 호출 허용
    return m_value.m_symbolic->name(); // [한국어] 심볼 포인터를 통해 이름 문자열 반환
  }

  /*
   * [한국어]
   * get_vect_nelem() — 벡터 피연산자의 유효 요소 수 반환 (0~8)
   *
   * @return: 벡터에 포함된 유효 레지스터 요소 수 (0~8)
   *
   * m_vector_symbolic 배열에서 첫 번째 NULL 포인터의 위치를 선형 탐색하여
   * 요소 수를 결정한다. v4 생성자면 4, v8 생성자면 8을 반환.
   * is_vector() assertion 실패 시 중단되므로 반드시 is_vector() 확인 후 호출.
   *
   * 호출 체인:
   *   instructions.cc 벡터 ld/st 실행 → [get_vect_nelem()] → (스캔 후 반환)
   */
  unsigned get_vect_nelem() const {
    assert(is_vector());                            // [한국어] 벡터 피연산자 아니면 중단
    if (!m_value.m_vector_symbolic[0]) return 0;   // [한국어] 0번째가 NULL이면 빈 벡터
    if (!m_value.m_vector_symbolic[1]) return 1;   // [한국어] 1번째가 NULL이면 1요소
    if (!m_value.m_vector_symbolic[2]) return 2;   // [한국어] 2번째가 NULL이면 2요소
    if (!m_value.m_vector_symbolic[3]) return 3;   // [한국어] 3번째가 NULL이면 3요소
    if (!m_value.m_vector_symbolic[4]) return 4;   // [한국어] 4번째가 NULL이면 v4 (4요소)
    if (!m_value.m_vector_symbolic[5]) return 5;   // [한국어] 5번째가 NULL이면 5요소
    if (!m_value.m_vector_symbolic[6]) return 6;   // [한국어] 6번째가 NULL이면 6요소
    if (!m_value.m_vector_symbolic[7]) return 7;   // [한국어] 7번째가 NULL이면 7요소
    return 8;                                       // [한국어] 모두 유효하면 v8 (8요소)
  }

  /*
   * [한국어]
   * vec_symbol() — 벡터 피연산자의 idx번째 요소 심볼 반환
   *
   * @idx: 벡터 인덱스 (0-7)
   * @return: 해당 인덱스의 symbol 포인터 (NULL이면 assert 중단)
   *
   * 벡터 ld/st 명령어 실행 시 각 요소 레지스터에 순차적으로 접근하기 위해 사용.
   * idx >= 8 이거나 해당 인덱스의 심볼이 NULL이면 assert로 중단된다.
   *
   * 호출 체인:
   *   instructions.cc 벡터 명령어 실행 → [vec_symbol(idx)] → symbol 접근
   */
  const symbol *vec_symbol(int idx) const {
    assert(idx < 8);                               // [한국어] 인덱스 범위 검사: 0-7만 허용
    const symbol *result = m_value.m_vector_symbolic[idx]; // [한국어] 해당 인덱스의 심볼 포인터 추출
    assert(result != NULL);                        // [한국어] NULL이면 실제 요소 없음 — 잘못된 접근
    return result;                                 // [한국어] 유효한 심볼 포인터 반환
  }

  /*
   * [한국어]
   * vec_name1() — 벡터 첫째 요소 레지스터 이름 반환
   *
   * @return: 벡터 인덱스 0의 심볼 이름 (std::string 참조)
   *
   * 벡터 피연산자 디버그 출력 또는 심볼 테이블 조회 시 사용.
   * vector_t 타입이 아니면 assert 중단.
   *
   * 호출 체인:
   *   print_insn() / 디버그 → [vec_name1()] → symbol::name()
   */
  const std::string &vec_name1() const {
    assert(m_type == vector_t);                    // [한국어] 벡터 타입 검사
    return m_value.m_vector_symbolic[0]->name();  // [한국어] 0번째 요소 심볼 이름 반환
  }

  /*
   * [한국어]
   * vec_name2() — 벡터 둘째 요소 레지스터 이름 반환
   *
   * @return: 벡터 인덱스 1의 심볼 이름 (std::string 참조)
   *
   * vec_name1()과 동일한 역할, 두 번째 요소 대상.
   * vector_t 타입이 아니면 assert 중단.
   */
  const std::string &vec_name2() const {
    assert(m_type == vector_t);                    // [한국어] 벡터 타입 검사
    return m_value.m_vector_symbolic[1]->name();  // [한국어] 1번째 요소 심볼 이름 반환
  }

  /*
   * [한국어]
   * vec_name3() — 벡터 셋째 요소 레지스터 이름 반환
   *
   * @return: 벡터 인덱스 2의 심볼 이름 (std::string 참조)
   *
   * vec_name1()과 동일한 역할, 세 번째 요소 대상.
   * vector_t 타입이 아니면 assert 중단.
   */
  const std::string &vec_name3() const {
    assert(m_type == vector_t);                    // [한국어] 벡터 타입 검사
    return m_value.m_vector_symbolic[2]->name();  // [한국어] 2번째 요소 심볼 이름 반환
  }

  /*
   * [한국어]
   * vec_name4() — 벡터 넷째 요소 레지스터 이름 반환
   *
   * @return: 벡터 인덱스 3의 심볼 이름 (std::string 참조)
   *
   * vec_name1()과 동일한 역할, 네 번째 요소 대상.
   * v4 벡터 피연산자의 마지막 요소. vector_t 타입이 아니면 assert 중단.
   */
  const std::string &vec_name4() const {
    assert(m_type == vector_t);                    // [한국어] 벡터 타입 검사
    return m_value.m_vector_symbolic[3]->name();  // [한국어] 3번째 요소 심볼 이름 반환
  }

  /*
   * [한국어]
   * is_reg() — 이 피연산자가 레지스터인지 검사
   *
   * @return: true이면 레지스터 피연산자, false이면 비레지스터
   *
   * reg_t 타입이면 직접 true. symbolic_t 타입이면 심볼 테이블에서 타입 키를 조회하여
   * 해당 심볼이 레지스터로 선언됐는지 확인한다. PTX에서 %r0, %f0, %p0 등의 레지스터.
   * 명령어 실행 전 레지스터 파일 접근 여부 판단, 스코어보드 체크 등에서 사용.
   *
   * 호출 체인:
   *   instructions.cc 명령어 실행, scoreboard 체크 → [is_reg()] → symbol::type()::get_key()::is_reg()
   */
  bool is_reg() const {
    if (m_type == reg_t) {    // [한국어] reg_t 타입으로 명시적 생성된 경우 즉시 true
      return true;
    }
    if (m_type != symbolic_t) { // [한국어] symbolic_t가 아니면 레지스터가 아님
      return false;
    }
    return m_value.m_symbolic->type()->get_key().is_reg(); // [한국어] 심볼의 타입 키를 통해 레지스터 여부 조회
  }
  /*
   * [한국어]
   * is_param_local() — 로컬(디바이스 함수) 파라미터 공간 심볼인지 검사
   *
   * @return: true이면 .param 공간 로컬 파라미터, false이면 아님
   *
   * 디바이스 함수의 .param 공간 선언 변수를 구분할 때 사용.
   * 커널 파라미터(.param kernel)와 달리 디바이스 함수 호출 시 전달되는 파라미터.
   * arg_buffer_t 복사 로직에서 파라미터 종류 분기 시 사용.
   */
  bool is_param_local() const {
    if (m_type != symbolic_t) return false;                                    // [한국어] 심볼 타입 아니면 바로 false
    return m_value.m_symbolic->type()->get_key().is_param_local();             // [한국어] 심볼 타입 키에서 param_local 여부 조회
  }

  /*
   * [한국어]
   * is_param_kernel() — 커널 파라미터 공간 심볼인지 검사
   *
   * @return: true이면 .param 공간 커널 파라미터, false이면 아님
   *
   * cuLaunchKernel로 전달되는 커널 파라미터는 .param 공간에 저장된다.
   * finalize() 또는 param_to_shared()에서 파라미터 배치 시 커널 파라미터 구분.
   * is_param_local()과 함께 PTX .param 공간의 두 가지 용도를 구분.
   */
  bool is_param_kernel() const {
    if (m_type != symbolic_t) return false;                                    // [한국어] 심볼 타입 아니면 바로 false
    return m_value.m_symbolic->type()->get_key().is_param_kernel();            // [한국어] 심볼 타입 키에서 param_kernel 여부 조회
  }

  /*
   * [한국어]
   * is_vector() — 이 피연산자가 벡터 피연산자인지 검사
   *
   * @return: true이면 다중 레지스터 벡터 피연산자, false이면 스칼라
   *
   * m_vector 플래그만 확인하는 단순 접근자.
   * ld.v4/st.v4 등 벡터 명령어 실행 시 분기 판단에 사용.
   */
  bool is_vector() const {
    if (m_vector) return true; // [한국어] m_vector 플래그가 설정돼 있으면 벡터 피연산자
    return false;
  }
  /*
   * [한국어]
   * reg_num() — 스칼라 레지스터 심볼의 레지스터 번호 반환
   *
   * @return: 이 피연산자에 연결된 레지스터의 물리적 번호
   *
   * 레지스터 파일 접근 시 인덱스로 사용. is_reg()가 true인 경우에만 유효.
   * 스코어보드, operand collector에서 레지스터 번호를 키로 사용.
   */
  int reg_num() const { return m_value.m_symbolic->reg_num(); }
  /*
   * [한국어]
   * reg1_num()~reg8_num() — 벡터 피연산자의 각 요소 레지스터 번호 반환
   *
   * @return: 해당 벡터 요소 레지스터의 물리적 번호 (요소가 NULL이면 0 반환)
   *
   * ld.v4/st.v4/WMMA 명령어의 벡터 피연산자에서 각 요소 레지스터를 순차적으로
   * 접근하기 위해 사용. reg1_num()/reg2_num()은 NULL 검사 없이 접근하므로
   * 반드시 유효한 요소가 있어야 한다. reg3_num()~reg8_num()은 NULL이면 0을 반환.
   *
   * 호출 체인:
   *   instructions.cc 벡터 명령어 실행 → [regN_num()] → symbol::reg_num()
   */
  int reg1_num() const { return m_value.m_vector_symbolic[0]->reg_num(); } // [한국어] 벡터 요소 0번 레지스터 번호
  int reg2_num() const { return m_value.m_vector_symbolic[1]->reg_num(); } // [한국어] 벡터 요소 1번 레지스터 번호
  int reg3_num() const {
    return m_value.m_vector_symbolic[2]
               ? m_value.m_vector_symbolic[2]->reg_num() // [한국어] 요소 2번 심볼이 유효하면 레지스터 번호 반환
               : 0;                                       // [한국어] NULL이면 0 반환 (v2 벡터에서 3번째 요소 없음)
  }
  int reg4_num() const {
    return m_value.m_vector_symbolic[3]
               ? m_value.m_vector_symbolic[3]->reg_num() // [한국어] 요소 3번 심볼이 유효하면 레지스터 번호 반환
               : 0;                                       // [한국어] NULL이면 0 반환
  }
  int reg5_num() const {
    return m_value.m_vector_symbolic[4]
               ? m_value.m_vector_symbolic[4]->reg_num() // [한국어] 요소 4번 심볼이 유효하면 레지스터 번호 반환 (v8 전용)
               : 0;                                       // [한국어] NULL이면 0 반환 (v4에서 5번째 요소 없음)
  }
  int reg6_num() const {
    return m_value.m_vector_symbolic[5]
               ? m_value.m_vector_symbolic[5]->reg_num() // [한국어] 요소 5번 심볼이 유효하면 레지스터 번호 반환 (v8 전용)
               : 0;
  }
  int reg7_num() const {
    return m_value.m_vector_symbolic[6]
               ? m_value.m_vector_symbolic[6]->reg_num() // [한국어] 요소 6번 심볼이 유효하면 레지스터 번호 반환 (v8 전용)
               : 0;
  }
  int reg8_num() const {
    return m_value.m_vector_symbolic[7]
               ? m_value.m_vector_symbolic[7]->reg_num() // [한국어] 요소 7번 심볼이 유효하면 레지스터 번호 반환 (v8 전용)
               : 0;
  }
  /*
   * [한국어]
   * arch_reg_num() — 스칼라 심볼의 아키텍처 레지스터 번호 반환 (단일 인수)
   *
   * @return: 이 피연산자 심볼에 배정된 아키텍처(하드웨어) 레지스터 번호
   *
   * PTX 가상 레지스터는 심볼 테이블에서 레지스터 할당 시 아키텍처 레지스터 번호를
   * 부여받는다. warp_inst_t의 레지스터 벡터(in/out)에 기록될 때 이 번호를 사용.
   */
  int arch_reg_num() const { return m_value.m_symbolic->arch_reg_num(); }
  /*
   * [한국어]
   * arch_reg_num(n) — 벡터 피연산자의 n번째 요소 아키텍처 레지스터 번호 반환
   *
   * @n:     벡터 인덱스 (0-7)
   * @return: n번째 요소의 아키텍처 레지스터 번호, NULL 요소면 -1
   *
   * 벡터 ld/st 명령어에서 warp_inst_t 레지스터 벡터를 채울 때 호출됨.
   * 해당 인덱스의 심볼이 NULL이면 -1을 반환하여 유효하지 않음을 표시.
   */
  int arch_reg_num(unsigned n) const {
    return (m_value.m_vector_symbolic[n])
               ? m_value.m_vector_symbolic[n]->arch_reg_num() // [한국어] 유효한 심볼이면 아키텍처 레지스터 번호 반환
               : -1;                                          // [한국어] NULL 심볼이면 -1 반환 (유효하지 않음)
  }
  /*
   * [한국어]
   * is_label() — 이 피연산자가 레이블(branch target)인지 검사
   *
   * @return: true이면 레이블 피연산자 (BRA/CALL의 분기 목적지 이름)
   *
   * PTX bra 명령어의 대상 레이블을 표현할 때 m_type = label_t 로 설정됨.
   * CFG 구성 시 레이블을 기준으로 기본 블록 경계를 결정하는 데 사용.
   */
  bool is_label() const { return m_type == label_t; }
  /*
   * [한국어]
   * is_builtin() — 이 피연산자가 내장(builtin) 피연산자인지 검사
   *
   * @return: true이면 CUDA 내장 변수 (예: %tid.x, %ntid.x, %ctaid.x 등)
   *
   * PTX의 특수 레지스터(%tid, %ctaid, %ntid 등)는 builtin_t 타입으로 표현됨.
   * 명령어 실행 시 내장 변수는 현재 스레드 컨텍스트에서 동적으로 값을 조회.
   */
  bool is_builtin() const { return m_type == builtin_t; }

  // Memory operand used in ld / st instructions (ex. [__var1])
  /*
   * [한국어]
   * is_memory_operand() — 대괄호([]) 감싸인 메모리 참조 피연산자인지 검사
   *
   * @return: true이면 PTX 표준 메모리 피연산자 (예: [__var1], [%r0])
   *
   * make_memory_operand()로 m_type = memory_t 로 설정된 피연산자.
   * ld/st 명령어 실행 시 이 검사를 통해 메모리 주소를 계산해야 함을 판단.
   * is_memory_operand2()와 구분: 이 함수는 PTX 표준, 저건 PTXPlus 확장.
   */
  bool is_memory_operand() const { return m_type == memory_t; }

  // Memory operand with immediate access (ex. s[0x0004] or g[$r1+=0x0004])
  // This is used by the PTXPlus extension. The operand is assigned an address
  // space during parsing.
  /*
   * [한국어]
   * is_memory_operand2() — PTXPlus 확장의 즉시 주소 공간 메모리 피연산자인지 검사
   *
   * @return: true이면 PTXPlus 메모리 피연산자 (주소 공간이 명시된 경우)
   *
   * PTXPlus에서 s[0x0004]처럼 주소 공간(shared/global 등)이 명시된 메모리 접근.
   * 파서에서 set_addr_space()로 주소 공간이 설정된 경우 undefined_space가 아님.
   * has_memory_read()/has_memory_write() 에서 소스/목적지 피연산자 분류에 사용.
   */
  bool is_memory_operand2() const { return (m_addr_space != undefined_space); }

  /*
   * [한국어]
   * is_immediate_address() — 즉시 주소 피연산자인지 검사
   *
   * @return: true이면 즉시 주소 피연산자 (m_immediate_address 플래그)
   *
   * unsigned 생성자 또는 set_immediate_addr()으로 설정됨.
   * mov 명령어에서 레지스터에 즉시 주소를 로드할 때 이 피연산자 유형이 사용됨.
   */
  bool is_immediate_address() const { return m_immediate_address; }

  /*
   * [한국어]
   * is_literal() — 리터럴(즉시값) 피연산자인지 검사
   *
   * @return: true이면 int/float/double/unsigned 리터럴
   *
   * add/mul 등 산술 명령어의 즉시값 피연산자 여부를 판단.
   * get_literal_value()를 호출하기 전 반드시 이 검사를 통과해야 함.
   * vector_t, symbolic_t, memory_t 등은 리터럴이 아니므로 false 반환.
   */
  bool is_literal() const {
    return m_type == int_t || m_type == float_op_t || m_type == double_op_t ||
           m_type == unsigned_t; // [한국어] signed int, float, double, unsigned 리터럴 타입만 해당
  }
  /*
   * [한국어]
   * is_shared() — 공유 메모리(.shared) 심볼인지 검사
   *
   * @return: true이면 __shared__ 선언 변수를 참조하는 피연산자
   *
   * symbolic_t/address_t/memory_t 타입의 피연산자에서만 의미 있음.
   * 다른 타입(리터럴, 레지스터 등)이면 즉시 false 반환.
   * ld/st 명령어 실행 시 공유 메모리 시뮬레이션 경로로 분기하는 데 사용.
   */
  bool is_shared() const {
    if (!(m_type == symbolic_t || m_type == address_t || m_type == memory_t)) {
      return false; // [한국어] 심볼 기반 타입이 아니면 공유 메모리 심볼일 수 없음
    }
    return m_value.m_symbolic->is_shared(); // [한국어] 심볼 자체의 공유 메모리 여부 조회
  }
  /*
   * [한국어]
   * is_sstarr() — 공유 메모리 심볼이 정적 배열인지 검사
   *
   * @return: true이면 정적 크기의 __shared__ 배열
   *
   * 동적으로 크기가 결정되는 공유 메모리 배열과 정적 배열을 구분.
   * 시뮬레이터 내 공유 메모리 시뮬레이션 경로에서 배열 크기 처리 분기에 사용.
   */
  bool is_sstarr() const { return m_value.m_symbolic->is_sstarr(); }
  /*
   * [한국어]
   * is_const() — 상수 메모리(.const) 심볼인지 검사
   *
   * @return: true이면 __constant__ 선언 변수를 참조하는 피연산자
   *
   * 상수 메모리는 L1 캐시에 캐시되고 전체 워프에 브로드캐스트된다.
   * ld 명령어 실행 시 상수 캐시 경로로 분기하는 데 사용.
   */
  bool is_const() const { return m_value.m_symbolic->is_const(); }
  /*
   * [한국어]
   * is_global() — 전역 메모리(.global) 심볼인지 검사
   *
   * @return: true이면 전역 메모리 변수를 참조하는 피연산자
   *
   * 전역 메모리는 DRAM을 통해 접근되며 L2 캐시를 거친다.
   * ld/st 명령어 실행 시 L1 d-cache/L2/DRAM 경로로 분기하는 데 사용.
   */
  bool is_global() const { return m_value.m_symbolic->is_global(); }
  /*
   * [한국어]
   * is_local() — 로컬 메모리(.local) 심볼인지 검사
   *
   * @return: true이면 스레드별 로컬 메모리 변수를 참조하는 피연산자
   *
   * PTX .local 공간은 레지스터 스필 영역으로, DRAM에 매핑된다.
   * ld/st 명령어에서 로컬 메모리 접근으로 분기하는 데 사용.
   */
  bool is_local() const { return m_value.m_symbolic->is_local(); }
  /*
   * [한국어]
   * is_tex() — 텍스처 메모리(.tex) 심볼인지 검사
   *
   * @return: true이면 텍스처 메모리 변수를 참조하는 피연산자
   *
   * tex 명령어 실행 시 텍스처 캐시 경로로 분기하는 데 사용.
   */
  bool is_tex() const { return m_value.m_symbolic->is_tex(); }
  /*
   * [한국어]
   * is_return_var() — 함수 반환 변수 피연산자인지 검사
   *
   * @return: true이면 CALL 명령어의 반환 값을 받을 피연산자
   *
   * set_return()으로 표시된 피연산자. func_addr()에서 반환 변수를 건너뛰고
   * 함수 주소 피연산자를 찾을 때 이 검사를 사용한다.
   */
  bool is_return_var() const { return m_is_return_var; }

  /*
   * [한국어]
   * is_function_address() — 함수 주소 피연산자인지 검사
   *
   * @return: true이면 간접 함수 호출의 대상 함수 주소 피연산자
   *
   * CALL 명령어에서 함수 포인터를 통한 간접 호출 시 사용.
   * symbolic_t 타입의 심볼이 함수 주소로 선언된 경우 true를 반환.
   *
   * 호출 체인:
   *   instructions.cc CALL 명령어 실행 → [is_function_address()] → symbol::is_func_addr()
   */
  bool is_function_address() const {
    if (m_type != symbolic_t) { // [한국어] 심볼 타입이 아니면 함수 주소일 수 없음
      return false;
    }
    return m_value.m_symbolic->is_func_addr(); // [한국어] 심볼이 함수 주소로 선언됐는지 조회
  }

  /*
   * [한국어]
   * get_literal_value() — 리터럴 피연산자의 값을 ptx_reg_t 유니온으로 반환
   *
   * @return: ptx_reg_t 유니온 — 타입에 따라 s64(int), f32(float), f64(double),
   *          u32(unsigned) 필드에 값이 채워짐
   *
   * 명령어 실행 시 리터럴 피연산자의 실제 값을 읽기 위해 호출.
   * is_literal()이 true인 경우에만 유효; 다른 타입이면 assert로 중단.
   * 반환된 ptx_reg_t는 피연산자 타입에 맞는 필드만 유효하므로
   * 호출자가 타입을 알고 적절한 필드를 읽어야 한다.
   *
   * 호출 체인:
   *   instructions.cc 산술 명령어 실행 → [get_literal_value()] → ptx_reg_t 반환
   */
  ptx_reg_t get_literal_value() const {
    ptx_reg_t result;           // [한국어] 반환용 레지스터 유니온 선언
    switch (m_type) {           // [한국어] 피연산자 타입에 따라 적절한 유니온 필드에 값 저장
      case int_t:
        result.s64 = m_value.m_int;    // [한국어] signed 정수 → s64 필드에 저장
        break;
      case float_op_t:
        result.f32 = m_value.m_float;  // [한국어] float → f32 필드에 저장
        break;
      case double_op_t:
        result.f64 = m_value.m_double; // [한국어] double → f64 필드에 저장
        break;
      case unsigned_t:
        result.u32 = m_value.m_unsigned; // [한국어] unsigned → u32 필드에 저장
        break;
      default:
        assert(0); // [한국어] 리터럴이 아닌 타입 — 호출자 오류
        break;
    }
    return result; // [한국어] 타입에 맞는 필드가 채워진 ptx_reg_t 반환
  }
  /*
   * [한국어]
   * get_int() — signed 정수 리터럴의 raw int 값 반환
   *
   * @return: m_value.m_int (int 타입 리터럴의 raw 값)
   *
   * get_literal_value()가 ptx_reg_t를 반환하는 것과 달리
   * 단순히 int 값만 필요할 때 사용하는 편의 접근자.
   */
  int get_int() const { return m_value.m_int; }
  /*
   * [한국어]
   * get_addr_offset() — 주소 오프셋(바이트) 반환
   *
   * @return: m_addr_offset — [base+offset] 형태의 주소에서 byte 오프셋
   *
   * address_t 타입 피연산자(베이스+오프셋 주소)와 unsigned 즉시 주소에서
   * 오프셋 값을 읽을 때 사용. ld/st 주소 계산에서 호출됨.
   */
  int get_addr_offset() const { return m_addr_offset; }
  /*
   * [한국어]
   * get_symbol() — 심볼 기반 피연산자의 symbol 포인터 반환
   *
   * @return: m_value.m_symbolic — 이 피연산자와 연결된 symbol 객체 포인터
   *
   * 심볼 테이블에서 변수 정보(타입, 주소, 크기 등)를 조회할 때 사용.
   * symbolic_t/reg_t/address_t/memory_t 타입에서만 유효.
   */
  const symbol *get_symbol() const { return m_value.m_symbolic; }
  /*
   * [한국어]
   * set_type() — 피연산자 타입을 강제로 변경
   *
   * @type: 새로운 operand_type enum 값
   * @return: 없음 (void)
   *
   * 파서가 파싱 후 추가 컨텍스트를 통해 피연산자 타입을 보정할 때 사용.
   * make_memory_operand()도 내부적으로 m_type을 변경하는 유사한 패턴이나
   * 이 함수는 임의의 타입으로 변경 가능.
   */
  void set_type(enum operand_type type) { m_type = type; }
  /*
   * [한국어]
   * get_type() — 현재 피연산자 타입 반환
   *
   * @return: m_type — 현재 operand_type enum 값
   *
   * 명령어 실행 시 피연산자의 종류(레지스터/리터럴/메모리/심볼 등)를
   * 판단하기 위해 호출. is_reg(), is_literal() 등의 내부 구현에서도 사용.
   */
  enum operand_type get_type() const { return m_type; }
  /*
   * [한국어]
   * set_neg_pred() — 이 피연산자를 부정 프레디케이트(negated predicate)로 표시
   *
   * @return: 없음 (void)
   *
   * PTX에서 @!%p0 형태로 프레디케이트를 부정하여 사용할 때 설정.
   * m_valid 상태여야만 설정 가능 (assert 검사).
   * ptx_instruction의 명령어 실행 시 프레디케이트 평가 방향을 반전시키는 데 사용.
   */
  void set_neg_pred() {
    assert(m_valid); // [한국어] 유효한 피연산자에만 부정 프레디케이트 설정 가능
    m_neg_pred = true; // [한국어] 프레디케이트 부정 플래그 설정
  }
  /*
   * [한국어]
   * is_neg_pred() — 부정 프레디케이트 피연산자인지 검사
   *
   * @return: true이면 @!%p 형태의 부정 프레디케이트 피연산자
   *
   * 명령어 실행 시 프레디케이트 값을 반전하여 조건 평가에 사용.
   */
  bool is_neg_pred() const { return m_neg_pred; }
  /*
   * [한국어]
   * is_valid() — 이 피연산자가 유효하게 초기화됐는지 검사
   *
   * @return: true이면 유효한 피연산자 (생성자에서 m_valid = true 설정됨)
   *
   * 기본 생성자로 생성된 빈 피연산자(m_valid = false)와 실제 피연산자를 구분.
   * has_return() 등에서 반환 변수의 유효성 검사에 사용.
   */
  bool is_valid() const { return m_valid; }

  /*
   * [한국어]
   * set_addr_space() — 피연산자의 메모리 주소 공간 설정
   *
   * @set_value: _memory_space_t enum 값 (shared_space, global_space 등)
   * @return: 없음 (void)
   *
   * PTXPlus 파서에서 메모리 주소 공간이 명시된 피연산자를 파싱할 때 호출.
   * is_memory_operand2()가 이 값을 통해 PTXPlus 메모리 피연산자 여부를 판단.
   */
  void set_addr_space(enum _memory_space_t set_value) {
    m_addr_space = set_value; // [한국어] 주소 공간 저장
  }
  /*
   * [한국어]
   * get_addr_space() — 피연산자의 메모리 주소 공간 반환
   *
   * @return: 현재 설정된 _memory_space_t enum 값 (기본값: undefined_space)
   *
   * PTXPlus 메모리 피연산자 처리 시 주소 공간을 확인하는 데 사용.
   */
  enum _memory_space_t get_addr_space() const { return m_addr_space; }
  /*
   * [한국어]
   * set_operand_lohi() / get_operand_lohi() — lo/hi 피연산자 선택자 설정/조회
   *
   * @set_value: 0=전체, 1=하위(lo), 2=상위(hi) 중 하나
   * @return (get): 현재 lo/hi 선택자 값
   *
   * PTXPlus에서 64비트 피연산자의 하위 32비트(lo)나 상위 32비트(hi)만 접근할 때
   * 사용하는 분리 연산자. SASS 레벨 명령어 지원을 위한 확장.
   */
  void set_operand_lohi(int set_value) { m_operand_lohi = set_value; }
  int get_operand_lohi() const { return m_operand_lohi; }
  /*
   * [한국어]
   * set_double_operand_type() / get_double_operand_type()
   * — double 피연산자의 특수 분류 설정/조회
   *
   * @set_value: double 피연산자 분류 코드 (파서에서 정의한 값)
   * @return (get): 현재 double 피연산자 분류 코드
   *
   * PTXPlus에서 double 타입 피연산자의 특수 사용 방식(예: 분리 로드/저장)을
   * 구분하기 위한 보조 분류자.
   */
  void set_double_operand_type(int set_value) {
    m_double_operand_type = set_value; // [한국어] double 피연산자 분류 코드 저장
  }
  int get_double_operand_type() const { return m_double_operand_type; }
  /*
   * [한국어]
   * set_operand_neg() / get_operand_neg() — 피연산자 값 부호 반전 플래그 설정/조회
   *
   * @return (get): 현재 부호 반전 여부 (true이면 값에 -1 곱함)
   *
   * PTXPlus에서 -reg 형태의 부호 반전 피연산자를 표현.
   * 명령어 실행 시 피연산자 값을 읽은 후 이 플래그가 true이면 부호 반전 적용.
   */
  void set_operand_neg() { m_operand_neg = true; }
  bool get_operand_neg() const { return m_operand_neg; }
  /*
   * [한국어]
   * set_const_mem_offset() / get_const_mem_offset()
   * — 상수 메모리 오프셋 설정/조회
   *
   * @set_value: 상수 메모리 뱅크 내 바이트 오프셋 (addr_t 타입)
   * @return (get): 현재 저장된 상수 메모리 오프셋 (미설정 시 (unsigned)-1)
   *
   * PTX ld.const 명령어에서 상수 메모리의 특정 오프셋에 접근할 때 사용.
   * 시뮬레이터 내 상수 메모리 시뮬레이션 경로에서 이 오프셋을 참조.
   */
  void set_const_mem_offset(addr_t set_value) {
    m_const_mem_offset = set_value; // [한국어] 상수 메모리 오프셋 저장
  }
  addr_t get_const_mem_offset() const { return m_const_mem_offset; }
  /*
   * [한국어]
   * is_non_arch_reg() — 비아키텍처 레지스터인지 검사
   *
   * @return: true이면 아키텍처 레지스터 파일에 없는 가상 레지스터
   *
   * PTXPlus에서 아키텍처 레지스터가 아닌 특수 피연산자를 표현할 때 사용.
   * 아키텍처 레지스터 파일 접근이 아닌 별도 처리 경로로 분기하는 데 사용.
   */
  bool is_non_arch_reg() const { return m_is_non_arch_reg; }

 private:
  gpgpu_context *gpgpu_ctx;
  /* [한국어] gpgpu_context 역방향 포인터.
   * 설정자: init() — 모든 생성자 진입 시 첫 번째로 저장.
   * 읽는 자: get_uid() — 전역 피연산자 ID 카운터에 접근할 때.
   * 값 범위: NULL이 아닌 유효한 gpgpu_context 포인터.
   * 동기화: 단일 스레드(PTX 파서)에서만 operand_info를 생성하므로 별도 락 불필요. */

  unsigned m_uid;
  /* [한국어] 이 피연산자 인스턴스의 전역 고유 ID.
   * 설정자: 각 생성자에서 get_uid()를 호출하여 할당; init()에서 (unsigned)-1로 임시 초기화.
   * 읽는 자: 현재 직접 외부에 노출되는 getter 없음 — 디버그/비교 용도로 내부 사용.
   * 값 범위: 1부터 증가하는 양수; (unsigned)-1은 init() 직후 미초기화 센티넬.
   * 동기화: get_uid() 내부에서 원자적 카운터 증가 또는 단일 스레드 접근. */

  bool m_valid;
  /* [한국어] 이 피연산자가 유효하게 초기화됐는지 여부.
   * 설정자: init()에서 false로, 각 생성자의 본체에서 true로 설정.
   * 읽는 자: is_valid() — 기본 생성자로 만든 빈 피연산자 구분에 사용.
   *          has_return() — ptx_instruction에서 반환 변수 유효성 확인.
   * 값 범위: true(유효) / false(기본 생성자 또는 init() 직후 미초기화).
   * 동기화: 파서 단일 스레드에서만 쓰므로 별도 락 불필요. */

  bool m_vector;
  /* [한국어] 이 피연산자가 다중 레지스터 벡터 피연산자인지 여부.
   * 설정자: 4-레지스터/8-레지스터 벡터 생성자에서 true로 설정; 나머지는 false.
   * 읽는 자: is_vector() — 명령어 실행 시 벡터 처리 경로 분기.
   *          get_vect_nelem() — 실제 요소 수 계산 전 검사.
   * 값 범위: true(벡터) / false(스칼라 또는 리터럴 피연산자).
   * 동기화: 파서 단일 스레드에서만 쓰므로 별도 락 불필요. */

  enum operand_type m_type;
  /* [한국어] 이 피연산자의 종류를 나타내는 타입 열거값.
   * 설정자: init()에서 undef_t로, 각 생성자 본체에서 실제 타입으로 설정.
   *          make_memory_operand()로 memory_t로 변환, set_type()으로 강제 변경 가능.
   * 읽는 자: is_reg(), is_literal(), is_memory_operand(), is_label() 등 모든 타입 조회 함수.
   *          명령어 실행 함수(instructions.cc)에서 피연산자 접근 방식 결정.
   * 값 범위: undef_t, symbolic_t, reg_t, address_t, memory_t, label_t,
   *          int_t, float_op_t, double_op_t, unsigned_t, vector_t, builtin_t 등.
   * 동기화: 파서 단일 스레드에서만 쓰므로 별도 락 불필요. */

  bool m_immediate_address;
  /* [한국어] 이 피연산자가 즉시 주소(immediate address)인지 여부.
   * 설정자: unsigned 생성자에서 true로 설정; 다른 생성자에서는 false.
   *          set_immediate_addr()으로도 설정 가능.
   * 읽는 자: is_immediate_address() — 즉시 주소 피연산자 분기 판단.
   * 값 범위: true(즉시 주소) / false(심볼 기반 또는 리터럴 피연산자).
   * 동기화: 파서 단일 스레드에서만 쓰므로 별도 락 불필요. */

  enum _memory_space_t m_addr_space;
  /* [한국어] PTXPlus 메모리 피연산자의 주소 공간 분류.
   * 설정자: init()에서 undefined_space로, set_addr_space()로 실제 공간 설정.
   * 읽는 자: is_memory_operand2() — undefined_space가 아니면 PTXPlus 메모리 피연산자.
   *          get_addr_space() — PTXPlus 명령어 실행 시 주소 공간 확인.
   * 값 범위: undefined_space(기본), shared_space, global_space, local_space,
   *          param_space_local, param_space_kernel, const_space, tex_space.
   * 동기화: 파서 단일 스레드에서만 쓰므로 별도 락 불필요. */

  int m_operand_lohi;
  /* [한국어] PTXPlus 64비트 피연산자의 lo/hi 분리 선택자.
   * 설정자: init()에서 0으로, set_operand_lohi()로 1(lo) 또는 2(hi) 설정.
   * 읽는 자: get_operand_lohi() — PTXPlus 명령어 실행 시 lo/hi 선택 판단.
   * 값 범위: 0(전체), 1(하위 32비트), 2(상위 32비트).
   * 동기화: 파서 단일 스레드에서만 쓰므로 별도 락 불필요. */

  int m_double_operand_type;
  /* [한국어] PTXPlus double 피연산자의 특수 분류 코드.
   * 설정자: init()에서 0으로, set_double_operand_type()으로 설정.
   * 읽는 자: get_double_operand_type() — PTXPlus double 처리 분기 판단.
   * 값 범위: 0(일반) 또는 파서 정의 특수 코드.
   * 동기화: 파서 단일 스레드에서만 쓰므로 별도 락 불필요. */

  bool m_operand_neg;
  /* [한국어] PTXPlus 피연산자 값 부호 반전 플래그.
   * 설정자: init()에서 false로, set_operand_neg()로 true 설정.
   * 읽는 자: get_operand_neg() — 명령어 실행 시 값 부호 반전 적용 여부 판단.
   * 값 범위: true(부호 반전) / false(원래 값).
   * 동기화: 파서 단일 스레드에서만 쓰므로 별도 락 불필요. */

  addr_t m_const_mem_offset;
  /* [한국어] 상수 메모리(.const) 접근 시 뱅크 내 바이트 오프셋.
   * 설정자: init()에서 (unsigned)-1(미설정 센티넬)로, set_const_mem_offset()으로 실제 값 설정.
   * 읽는 자: get_const_mem_offset() — 상수 메모리 ld 명령어 실행 시 오프셋 계산.
   * 값 범위: 0 ~ 상수 메모리 뱅크 크기-1 (바이트); (unsigned)-1은 미설정 센티넬.
   * 동기화: 파서 단일 스레드에서만 쓰므로 별도 락 불필요. */

  union {
    int m_int;
    /* [한국어] signed 정수 리터럴 값 (int_t 타입 피연산자).
     * 설정자: int(x) 생성자에서 m_value.m_int = x 로 저장.
     * 읽는 자: get_int(), get_literal_value() (int_t 케이스).
     * 값 범위: INT_MIN ~ INT_MAX.
     * 동기화: union이므로 같은 시점에 m_int 외 다른 필드와 동시 접근 불가. */

    unsigned int m_unsigned;
    /* [한국어] unsigned 정수 리터럴 값 (unsigned_t 타입 피연산자).
     * 설정자: unsigned(x) 생성자에서 m_value.m_unsigned = x 로 저장.
     * 읽는 자: get_literal_value() (unsigned_t 케이스).
     * 값 범위: 0 ~ UINT_MAX.
     * 동기화: union이므로 같은 시점에 m_unsigned 외 다른 필드와 동시 접근 불가. */

    float m_float;
    /* [한국어] 32비트 float 리터럴 값 (float_op_t 타입 피연산자).
     * 설정자: float(x) 생성자에서 m_value.m_float = x 로 저장.
     * 읽는 자: get_literal_value() (float_op_t 케이스).
     * 값 범위: IEEE 754 단정도 부동소수점 범위.
     * 동기화: union이므로 같은 시점에 m_float 외 다른 필드와 동시 접근 불가. */

    double m_double;
    /* [한국어] 64비트 double 리터럴 값 (double_op_t 타입 피연산자).
     * 설정자: double(x) 생성자에서 m_value.m_double = x 로 저장.
     * 읽는 자: get_literal_value() (double_op_t 케이스).
     * 값 범위: IEEE 754 배정도 부동소수점 범위.
     * 동기화: union이므로 같은 시점에 m_double 외 다른 필드와 동시 접근 불가. */

    int m_vint[4];
    /* [한국어] 벡터 signed 정수 요소 배열 (현재 미사용, 확장용 예약).
     * 설정자: init()에서 0으로 초기화; 실제 벡터는 m_vector_symbolic 사용.
     * 읽는 자: 현재 직접 사용하는 코드 없음.
     * 동기화: union 필드이므로 m_vector_symbolic과 공존 불가. */

    unsigned int m_vunsigned[4];
    /* [한국어] 벡터 unsigned 정수 요소 배열 (현재 미사용, 확장용 예약).
     * 설정자: init()에서 0으로 초기화.
     * 읽는 자: 현재 직접 사용하는 코드 없음.
     * 동기화: union 필드이므로 다른 활성 필드와 동시 접근 불가. */

    float m_vfloat[4];
    /* [한국어] 벡터 float 요소 배열 (현재 미사용, 확장용 예약).
     * 설정자: init()에서 0으로 초기화.
     * 읽는 자: 현재 직접 사용하는 코드 없음.
     * 동기화: union 필드이므로 다른 활성 필드와 동시 접근 불가. */

    double m_vdouble[4];
    /* [한국어] 벡터 double 요소 배열 (현재 미사용, 확장용 예약).
     * 설정자: init()에서 0으로 초기화.
     * 읽는 자: 현재 직접 사용하는 코드 없음.
     * 동기화: union 필드이므로 다른 활성 필드와 동시 접근 불가. */

    const symbol *m_symbolic;
    /* [한국어] 심볼 기반 피연산자의 심볼 포인터 (symbolic_t/reg_t/address_t/memory_t/label_t).
     * 설정자: symbol 포인터 생성자들에서 m_value.m_symbolic = s 로 저장.
     * 읽는 자: name(), is_reg(), is_shared(), is_global() 등 심볼 속성 조회 함수.
     *          get_symbol(), reg_num(), arch_reg_num() (스칼라 버전).
     * 값 범위: 심볼 테이블에서 살아있는 symbol 객체 포인터 (NULL 불가).
     * 동기화: symbol 객체 자체는 파싱 완료 후 불변이므로 별도 락 불필요. */

    const symbol **m_vector_symbolic;
    /* [한국어] 벡터 피연산자의 심볼 포인터 배열 (heap 할당, 크기 항상 8).
     * 설정자: 4-요소/8-요소 벡터 생성자에서 new symbol*[8]로 할당 후 채움.
     * 읽는 자: get_vect_nelem() (NULL sentinel 스캔), vec_symbol(idx), vec_name1~4(),
     *          reg1_num()~reg8_num(), arch_reg_num(n).
     * 값 범위: 배열 포인터 (NULL이면 벡터 피연산자 아님); 배열 내 요소는
     *           유효 범위까지 symbol 포인터, 그 이후는 NULL sentinel.
     * 동기화: 생성 후 불변이므로 별도 락 불필요. (메모리 해제는 현재 미구현) */

  } m_value;

  int m_addr_offset;
  /* [한국어] 주소 기반 피연산자의 바이트 오프셋.
   * 설정자: address_t 생성자(베이스+오프셋)에서 offset 값 저장;
   *          unsigned 생성자에서는 즉시 주소 값(x) 자체를 저장;
   *          나머지 생성자에서는 0으로 초기화.
   * 읽는 자: get_addr_offset() — ld/st 주소 계산 시 베이스에 더할 오프셋.
   * 값 범위: 음수~양수 정수 (바이트 오프셋).
   * 동기화: 파서 단일 스레드에서만 쓰므로 별도 락 불필요. */

  bool m_neg_pred;
  /* [한국어] 부정 프레디케이트 플래그 (@!%p 형태).
   * 설정자: init()에서 false로(0으로 저장), set_neg_pred()로 true 설정.
   * 읽는 자: is_neg_pred() — 명령어 실행 시 프레디케이트 평가 방향 반전 여부 판단.
   * 값 범위: true(@!%p 부정 조건) / false(@%p 긍정 조건).
   * 동기화: 파서 단일 스레드에서만 쓰므로 별도 락 불필요. */

  bool m_is_return_var;
  /* [한국어] CALL 명령어의 반환 값 수신 피연산자 여부.
   * 설정자: init()에서 false(0)로, set_return()으로 true 설정.
   * 읽는 자: is_return_var() — ptx_instruction::func_addr()에서 함수 주소와 구분.
   * 값 범위: true(반환 값 피연산자) / false(일반 피연산자).
   * 동기화: 파서 단일 스레드에서만 쓰므로 별도 락 불필요. */

  bool m_is_non_arch_reg;
  /* [한국어] 아키텍처 레지스터 파일 외부의 가상 레지스터 여부.
   * 설정자: init()에서 false(0)로 초기화; PTXPlus 파서에서 특수 경우에 true 설정.
   * 읽는 자: is_non_arch_reg() — 아키텍처 레지스터 파일 접근 여부 분기.
   * 값 범위: true(비아키텍처 가상 레지스터) / false(표준 아키텍처 레지스터).
   * 동기화: 파서 단일 스레드에서만 쓰므로 별도 락 불필요. */

  /*
   * [한국어]
   * get_uid() — 전역 고유 ID 카운터에서 다음 ID를 발급
   *
   * @return: 새로 발급된 unsigned 고유 ID
   *
   * gpgpu_context에 저장된 전역 카운터를 증가시켜 각 operand_info 인스턴스에
   * 고유한 ID를 부여한다. 각 생성자에서 m_uid = get_uid()로 호출됨.
   * 구현은 ptx_ir.cc에 있으며, gpgpu_ctx를 통해 카운터에 접근.
   *
   * 호출 체인:
   *   각 operand_info 생성자 → [get_uid()] → gpgpu_context 카운터 증가
   */
  unsigned get_uid();
};

extern const char *g_opcode_string[];
/* [한국어] g_opcode_string — PTX 오퍼코드 정수값 → 문자열 매핑 전역 배열.
 * 선언: ptx_ir.cc (또는 관련 구현 파일)에서 정의.
 * 읽는 자: ptx_instruction::get_opcode_cstr() — 오퍼코드를 사람이 읽을 수 있는 이름으로 변환.
 * 인덱스: m_opcode 정수값 (음수 -1이면 레이블임). */

/*
 * [한국어]
 *
 * === 파일의 역할 (basic_block_t) ===
 * PTX 함수의 제어 흐름 그래프(CFG, Control Flow Graph)에서 기본 블록(Basic Block)을
 * 표현하는 구조체. 기본 블록은 한 번 진입하면 반드시 끝까지 순차 실행되는 명령어 시퀀스.
 *
 * === 전체 아키텍처에서의 위치 ===
 * function_info::create_basic_blocks() 가 PTX 명령어 리스트를 분석하여 이 구조체를
 * 생성한다. 이후 connect_basic_blocks() → find_dominators() → find_postdominators() →
 * find_ipostdominators() 순서로 CFG 분석이 진행되며, 최종적으로 SIMT 재합류 지점
 * (reconvergence point)을 결정하는 데 사용된다.
 *
 * === 타 모듈과의 연결 ===
 * - function_info::m_basic_blocks 벡터에 포인터로 저장됨
 * - ptx_instruction::assign_bb() 로 각 명령어에서 자신이 속한 기본 블록을 역참조
 * - dominator/postdominator 분석 결과는 SIMT stack(simt_stack)의 reconvergence 설정에 사용
 *
 * === 주요 함수/구조체 요약 ===
 * - basic_block_t(): 생성자 — ID, 시작/끝 명령어, 진입/종료 여부로 초기화
 * - dom(B):  이 블록이 B를 지배(dominate)하는지 — B.dominator_ids에 이 블록 ID 있으면 true
 * - pdom(B): 이 블록이 B를 후지배(postdominate)하는지 — B.postdominator_ids에 이 블록 ID 있으면 true
 */
struct basic_block_t {
  /*
   * [한국어]
   * basic_block_t() — 기본 블록 생성자
   *
   * @ID:    이 기본 블록의 고유 번호 (function_info::m_basic_blocks 벡터 인덱스)
   * @begin: 이 블록의 첫 번째 PTX 명령어 포인터
   * @end:   이 블록의 마지막 PTX 명령어 포인터 (inclusive)
   * @entry: true이면 함수의 진입(entry) 기본 블록
   * @ex:    true이면 함수의 종료(exit) 기본 블록 (ret 명령어 포함)
   *
   * function_info::create_basic_blocks()에서 각 기본 블록 경계를 결정한 후 호출.
   * immediatepostdominator_id/immediatedominator_id 는 -1로 초기화하여
   * find_ipostdominators()/find_idominators() 실행 전 미설정 상태를 표시.
   *
   * 호출 체인:
   *   function_info::create_basic_blocks() → [new basic_block_t(...)] → m_basic_blocks 저장
   */
  basic_block_t(unsigned ID, ptx_instruction *begin, ptx_instruction *end,
                bool entry, bool ex) {
    bb_id = ID;                          // [한국어] 기본 블록 고유 번호 저장
    ptx_begin = begin;                   // [한국어] 블록 시작 명령어 포인터 저장
    ptx_end = end;                       // [한국어] 블록 끝 명령어 포인터 저장 (inclusive)
    is_entry = entry;                    // [한국어] 함수 진입 블록 여부 설정
    is_exit = ex;                        // [한국어] 함수 종료 블록 여부 설정
    immediatepostdominator_id = -1;      // [한국어] 즉각 후지배자 미결정 센티넬 (-1)
    immediatedominator_id = -1;          // [한국어] 즉각 지배자 미결정 센티넬 (-1)
  }

  ptx_instruction *ptx_begin;
  /* [한국어] 이 기본 블록의 첫 번째 PTX 명령어 포인터.
   * 설정자: 생성자에서 create_basic_blocks()가 결정한 시작 명령어로 설정.
   * 읽는 자: print_basic_blocks(), CFG 순회, 명령어 시퀀스 처리.
   * 값 범위: 유효한 ptx_instruction 포인터 (NULL 불가).
   * 동기화: CFG 분석은 단일 스레드에서 진행하므로 별도 락 불필요. */

  ptx_instruction *ptx_end;
  /* [한국어] 이 기본 블록의 마지막 PTX 명령어 포인터 (inclusive).
   * 설정자: 생성자에서 create_basic_blocks()가 결정한 끝 명령어로 설정.
   * 읽는 자: print_basic_blocks(), connect_basic_blocks() (후속 블록 연결).
   * 값 범위: 유효한 ptx_instruction 포인터; ptx_begin과 같을 수 있음(단일 명령어 블록).
   * 동기화: CFG 분석은 단일 스레드에서 진행하므로 별도 락 불필요. */

  std::set<int>
      predecessor_ids;  // indices of other basic blocks in m_basic_blocks array
  /* [한국어] 이 기본 블록의 직전 블록들의 인덱스 집합 (m_basic_blocks 배열 기준).
   * 설정자: connect_basic_blocks() — 이전 블록의 successor에 이 블록이 추가될 때 역방향으로 설정.
   * 읽는 자: find_dominators()/find_postdominators() — CFG 역방향 순회에서 전임자 탐색.
   * 값 범위: 0 ~ m_basic_blocks.size()-1 범위의 정수 인덱스.
   * 동기화: CFG 분석 단일 스레드에서 순차 구축하므로 별도 락 불필요. */

  std::set<int> successor_ids;
  /* [한국어] 이 기본 블록 실행 후 도달 가능한 다음 블록들의 인덱스 집합.
   * 설정자: connect_basic_blocks() — 마지막 명령어(branch/bra 등)의 분기 대상을 분석하여 설정.
   * 읽는 자: find_dominators()/find_postdominators() — CFG 순방향 순회에서 후계자 탐색.
   * 값 범위: 0 ~ m_basic_blocks.size()-1 범위의 정수 인덱스.
   * 동기화: CFG 분석 단일 스레드에서 순차 구축하므로 별도 락 불필요. */

  std::set<int> postdominator_ids;
  /* [한국어] 이 기본 블록의 후지배자(post-dominator) 블록들의 인덱스 집합.
   * 설정자: find_postdominators() — Muchnick Fig 7.14 역방향 알고리즘으로 계산.
   * 읽는 자: pdom() — 특정 블록이 후지배 관계인지 확인;
   *           find_ipostdominators() — 즉각 후지배자 계산 시 집합 교차 연산.
   * 값 범위: 이 블록에서 종료 블록까지의 모든 경로에 포함되는 블록들의 인덱스.
   * 동기화: CFG 분석 단일 스레드에서 계산하므로 별도 락 불필요. */

  std::set<int> dominator_ids;
  /* [한국어] 이 기본 블록의 지배자(dominator) 블록들의 인덱스 집합.
   * 설정자: find_dominators() — Muchnick Fig 7.14 순방향 알고리즘으로 계산.
   * 읽는 자: dom() — 특정 블록이 지배 관계인지 확인;
   *           find_idominators() — 즉각 지배자 계산 시 집합 교차 연산.
   * 값 범위: 진입 블록부터 이 블록까지의 모든 경로에 포함되는 블록들의 인덱스.
   * 동기화: CFG 분석 단일 스레드에서 계산하므로 별도 락 불필요. */

  std::set<int> Tmp_ids;
  /* [한국어] 지배자/후지배자 알고리즘 내부에서 임시 교차 연산에 사용하는 집합.
   * 설정자: find_dominators() / find_postdominators() — 반복 고정점(fix-point) 계산 중 임시 저장.
   * 읽는 자: 알고리즘 수렴 후 dominator_ids/postdominator_ids 로 대입.
   * 값 범위: 알고리즘 각 반복 단계에서 변동.
   * 동기화: CFG 분석 단일 스레드에서 계산하므로 별도 락 불필요. */

  int immediatepostdominator_id;
  /* [한국어] 이 기본 블록의 즉각 후지배자(immediate post-dominator) 블록 인덱스.
   * 설정자: 생성자에서 -1(미결정)로; find_ipostdominators() — Muchnick Fig 7.15 알고리즘으로 설정.
   * 읽는 자: get_reconvergence_pairs() — SIMT 스택 재합류 지점 쌍(gpgpu_recon_t) 결정.
   *           분기 명령어(BRA)의 즉각 후지배자가 SIMT 재합류 target_pc.
   * 값 범위: 0 ~ m_basic_blocks.size()-1; -1이면 미결정.
   * 동기화: CFG 분석 단일 스레드에서 계산하므로 별도 락 불필요. */

  int immediatedominator_id;
  /* [한국어] 이 기본 블록의 즉각 지배자(immediate dominator) 블록 인덱스.
   * 설정자: 생성자에서 -1(미결정)로; find_idominators() — 즉각 지배자 계산.
   * 읽는 자: 지배자 트리 구성, 최적화 분석.
   * 값 범위: 0 ~ m_basic_blocks.size()-1; -1이면 미결정(진입 블록은 자신이 지배자).
   * 동기화: CFG 분석 단일 스레드에서 계산하므로 별도 락 불필요. */

  bool is_entry;
  /* [한국어] 이 기본 블록이 함수의 진입 기본 블록인지 여부.
   * 설정자: 생성자에서 create_basic_blocks()가 첫 번째 블록에 true로 설정.
   * 읽는 자: find_dominators() — 진입 블록의 지배자 집합 초기화(자기 자신만).
   * 값 범위: true(함수 진입 블록) / false(나머지 블록).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  bool is_exit;
  /* [한국어] 이 기본 블록이 함수의 종료 기본 블록인지 여부.
   * 설정자: 생성자에서 ret/exit 명령어를 포함하는 블록에 true로 설정.
   * 읽는 자: find_postdominators() — 종료 블록의 후지배자 집합 초기화(자기 자신만).
   * 값 범위: true(함수 종료 블록) / false(나머지 블록).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  unsigned bb_id;
  /* [한국어] 이 기본 블록의 고유 번호 (m_basic_blocks 배열에서의 인덱스).
   * 설정자: 생성자에서 ID 파라미터로 설정.
   * 읽는 자: dom()/pdom() — dominator_ids/postdominator_ids 집합에서 이 ID 검색.
   *           CFG 분석 알고리즘들에서 집합 연산 시 이 ID를 키로 사용.
   * 값 범위: 0 ~ m_basic_blocks.size()-1.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  // if this basic block dom B
  /*
   * [한국어]
   * dom() — 이 기본 블록이 B를 지배(dominate)하는지 검사
   *
   * @B: 대상 기본 블록 포인터
   * @return: true이면 이 블록이 B의 지배자 집합에 포함됨 (= 이 블록이 B를 지배)
   *
   * "블록 A가 블록 B를 지배한다" = 진입 블록부터 B까지의 모든 경로가 A를 통과함.
   * B의 dominator_ids 집합에 this->bb_id가 있으면 true를 반환.
   * find_ipostdominators()에서 즉각 후지배자 계산 시 불필요한 후보를 제거하는 데 사용.
   *
   * 호출 체인:
   *   function_info::find_ipostdominators() → [dom(B)] → set::find()
   */
  bool dom(const basic_block_t *B) {
    return (B->dominator_ids.find(this->bb_id) != B->dominator_ids.end()); // [한국어] B의 지배자 집합에 이 블록 ID가 있으면 지배 관계
  }

  // if this basic block pdom B
  /*
   * [한국어]
   * pdom() — 이 기본 블록이 B를 후지배(post-dominate)하는지 검사
   *
   * @B: 대상 기본 블록 포인터
   * @return: true이면 이 블록이 B의 후지배자 집합에 포함됨 (= 이 블록이 B를 후지배)
   *
   * "블록 A가 블록 B를 후지배한다" = B에서 종료 블록까지의 모든 경로가 A를 통과함.
   * SIMT 재합류 지점 결정 시 분기 명령어의 즉각 후지배자를 찾는 과정에서 사용.
   * B의 postdominator_ids 집합에 this->bb_id가 있으면 true를 반환.
   *
   * 호출 체인:
   *   function_info::find_ipostdominators() → [pdom(B)] → set::find()
   */
  bool pdom(const basic_block_t *B) {
    return (B->postdominator_ids.find(this->bb_id) !=
            B->postdominator_ids.end()); // [한국어] B의 후지배자 집합에 이 블록 ID가 있으면 후지배 관계
  }
};

/*
 * [한국어]
 *
 * === 파일의 역할 (gpgpu_recon_t) ===
 * GPGPU-Sim의 SIMT 스택(simt_stack) 재합류(reconvergence) 지점을 표현하는 구조체.
 * 분기 명령어(BRA)와 그 즉각 후지배자(immediate post-dominator) 명령어의 쌍으로,
 * 워프 내 분기 발산(divergence) 후 스레드들이 다시 모이는 지점을 정의한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * function_info::do_pdom() → get_reconvergence_pairs() 로 배열을 채우고,
 * ptx_sim_init_thread() 또는 simt_stack::update()에서 이 쌍을 SIMT 스택에 push.
 * source_pc의 분기 명령어 실행 시 target_pc를 재합류 주소로 SIMT 스택에 등록.
 *
 * === 타 모듈과의 연결 ===
 * - function_info::get_reconvergence_pairs() 가 이 배열을 채움
 * - shader.cc의 simt_stack — 재합류 PC를 스택에 push/pop하여 SIMT 실행 제어
 * - warp_inst_t 실행 경로 — source_inst, target_inst를 통해 명령어 직접 참조
 *
 * === 주요 함수/구조체 요약 ===
 * - source_pc:   분기 명령어의 PC (SIMT 스택에 source로 기록)
 * - target_pc:   분기의 즉각 후지배자 명령어의 PC (재합류 target)
 * - source_inst: source_pc에 해당하는 ptx_instruction 포인터
 * - target_inst: target_pc에 해당하는 ptx_instruction 포인터
 */
struct gpgpu_recon_t {
  address_type source_pc;
  /* [한국어] 분기 명령어(BRA)의 PC (프로그램 카운터).
   * 설정자: get_reconvergence_pairs() — basic_block_t의 마지막 명령어가 분기이면 해당 PC 저장.
   * 읽는 자: simt_stack::update() / ptx_sim_init_thread() — SIMT 스택에 재합류 쌍 등록.
   * 값 범위: 함수 내 유효한 PC 값 (m_start_PC 기준 오프셋).
   * 동기화: do_pdom() 완료 후 읽기 전용으로 사용하므로 별도 락 불필요. */

  address_type target_pc;
  /* [한국어] 분기의 즉각 후지배자 명령어의 PC — SIMT 재합류 목적지 주소.
   * 설정자: get_reconvergence_pairs() — source 블록의 immediatepostdominator_id 로부터 PC 계산.
   * 읽는 자: simt_stack::update() — 발산 분기 발생 시 이 PC를 재합류 주소로 스택에 push.
   * 값 범위: 함수 내 유효한 PC 값; source_pc 이후의 PC여야 함.
   * 동기화: do_pdom() 완료 후 읽기 전용으로 사용하므로 별도 락 불필요. */

  class ptx_instruction *source_inst;
  /* [한국어] source_pc에 해당하는 PTX 명령어 포인터 (분기 명령어).
   * 설정자: get_reconvergence_pairs() — m_instr_mem[source_pc - m_start_PC] 로 설정.
   * 읽는 자: SIMT 재합류 처리 — 분기 명령어의 타입/조건 등 추가 정보 접근.
   * 값 범위: 유효한 ptx_instruction 포인터 (BRA 또는 조건 분기 명령어).
   * 동기화: do_pdom() 완료 후 읽기 전용으로 사용하므로 별도 락 불필요. */

  class ptx_instruction *target_inst;
  /* [한국어] target_pc에 해당하는 PTX 명령어 포인터 (즉각 후지배자 명령어).
   * 설정자: get_reconvergence_pairs() — m_instr_mem[target_pc - m_start_PC] 로 설정.
   * 읽는 자: SIMT 재합류 처리 — 재합류 지점 명령어 정보 접근.
   * 값 범위: 유효한 ptx_instruction 포인터.
   * 동기화: do_pdom() 완료 후 읽기 전용으로 사용하므로 별도 락 불필요. */
};

/*
 * [한국어]
 *
 * === 파일의 역할 (ptx_instruction) ===
 * 하나의 PTX 명령어를 표현하는 클래스. warp_inst_t (abstract_hardware_model.h) 를 상속하여
 * PTX IR 노드인 동시에 타이밍 모델 명령어 역할을 겸한다. 파서가 생성하고, 기능 시뮬레이터
 * (instructions.cc)가 실행하며, 타이밍 모델(shader.cc)이 파이프라인 단계에 배치한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   PTX 파서(ptx.y) → ptx_instruction 생성 → function_info::add_inst()
 *   → ptx_assemble() → pre_decode() → set_opcode_and_latency()
 *   → shader.cc 파이프라인에서 warp_inst_t로 참조
 *   → instructions.cc execute()에서 피연산자 읽기/실행/쓰기
 *
 * === 타 모듈과의 연결 ===
 * - abstract_hardware_model.h: warp_inst_t 기반 클래스 — in/out 레지스터 벡터, 메모리 접근 정보
 * - function_info::m_instr_mem[]: PC 오프셋으로 인덱싱되는 명령어 배열에 저장
 * - basic_block_t: assign_bb()를 통해 CFG 기본 블록에 소속
 * - gpu-cache.cc: has_memory_read()/has_memory_write()로 메모리 접근 타입 판단
 *
 * === 주요 함수/구조체 요약 ===
 * - 생성자: 파서에서 파싱된 피연산자/옵션/타입 정보로 완전한 명령어 객체 구성
 * - pre_decode(): ptx_assemble() 후 레지스터 할당 및 warp_inst_t 기반 필드 초기화
 * - set_opcode_and_latency(): 오퍼코드에 따른 실행 레이턴시/파이프라인 유닛 설정
 * - dst()/src1()..src8(): 피연산자 접근자 — 명령어 실행 시 레지스터/메모리 값 읽기/쓰기
 * - has_memory_read()/has_memory_write(): 메모리 접근 타입 분류
 */
class ptx_instruction : public warp_inst_t {
 public:
  /*
   * [한국어]
   * ptx_instruction() — PTX 명령어 객체 생성자
   *
   * @opcode:       PTX 오퍼코드 정수 (g_opcode_string 인덱스; -1이면 레이블)
   * @pred:         프레디케이트 레지스터 심볼 (@%p 또는 @!%p; NULL이면 무조건 실행)
   * @neg_pred:     프레디케이트 부정 여부 (1이면 @!%p)
   * @pred_mod:     프레디케이트 수식자
   * @label:        레이블 심볼 (이 명령어가 레이블이면 non-NULL; 일반 명령어면 NULL)
   * @operands:     피연산자 리스트 (dst, src1, src2, ... 순서)
   * @return_var:   CALL 명령어의 반환 변수 피연산자
   * @options:      명령어 수식어 옵션 리스트 (예: .f32, .rn, .ca 등)
   * @wmma_options: WMMA(warp 행렬 곱) 전용 옵션 리스트
   * @scalar_type:  스칼라 타입 리스트 (예: S32, F32 등)
   * @space_spec:   메모리 공간 지정자 (예: global_space, shared_space)
   * @file:         소스 파일 이름 (디버그 정보)
   * @line:         소스 파일 행 번호 (디버그 정보)
   * @source:       원본 소스 텍스트 문자열
   * @config:       SM 코어 설정 (core_config) — set_opcode_and_latency()에 전달
   * @ctx:          gpgpu_context 역방향 포인터
   *
   * PTX 파서(ptx.y)가 각 명령어 규칙(rule)을 환원(reduce)할 때 호출.
   * 내부적으로 m_operands 벡터를 구성하고, set_opcode_and_latency()를 호출하여
   * warp_inst_t 기반의 실행 레이턴시/파이프라인 유닛 타입을 설정한다.
   * 구현은 ptx_ir.cc에 있으며 생성자에서 pre_decode()도 호출한다.
   *
   * 호출 체인:
   *   PTX 파서 (ptx.y) → [ptx_instruction(...)] → set_opcode_and_latency()
   */
  ptx_instruction(int opcode, const symbol *pred, int neg_pred, int pred_mod,
                  symbol *label, const std::list<operand_info> &operands,
                  const operand_info &return_var, const std::list<int> &options,
                  const std::list<int> &wmma_options,
                  const std::list<int> &scalar_type, memory_space_t space_spec,
                  const char *file, unsigned line, const char *source,
                  const core_config *config, gpgpu_context *ctx);

  /*
   * [한국어]
   * print_insn() — 이 명령어를 stdout에 출력 (디버그용)
   *
   * @return: 없음 (void)
   *
   * 디버그 또는 PTX 코드 덤프 시 명령어 텍스트를 표준 출력으로 인쇄.
   * 오퍼코드, 프레디케이트, 피연산자, 옵션 등을 사람이 읽을 수 있는 형태로 출력.
   *
   * 호출 체인:
   *   function_info::print_insn() → [print_insn()] (stdout 버전)
   */
  void print_insn() const;
  /*
   * [한국어]
   * print_insn(FILE *fp) — 이 명령어를 지정된 파일 스트림에 출력
   *
   * @fp: 출력 대상 FILE 포인터 (stderr, 로그 파일 등)
   * @return: 없음 (void) — virtual 함수
   *
   * function_info::print_insn(pc, fp)에서 PC별 명령어 출력 시 호출.
   * 파생 클래스가 재정의할 수 있도록 virtual로 선언.
   *
   * 호출 체인:
   *   function_info::print_insn(pc, fp) → [print_insn(fp)]
   */
  virtual void print_insn(FILE *fp) const;
  /*
   * [한국어]
   * to_string() — 명령어를 std::string으로 변환 (디버그/로깅용)
   *
   * @return: 이 명령어의 텍스트 표현 문자열
   *
   * print_insn()의 문자열 반환 버전.
   * 로그 메시지나 에러 출력에서 명령어 내용을 문자열로 포함할 때 사용.
   */
  std::string to_string() const;
  /*
   * [한국어]
   * inst_size() — 명령어 바이트 크기 반환
   *
   * @return: m_inst_size — 이 명령어의 바이트 크기 (보통 4 또는 8)
   *
   * ptx_assemble() 시 m_instr_mem 배열에서 다음 명령어 위치를 계산하거나
   * PC 증분 값을 결정할 때 사용.
   */
  unsigned inst_size() const { return m_inst_size; }
  /*
   * [한국어]
   * uid() — 명령어 고유 ID 반환
   *
   * @return: m_uid — 파서가 순차적으로 부여한 이 명령어의 전역 고유 번호
   *
   * 디버그/추적에서 명령어를 고유하게 식별하는 데 사용.
   */
  unsigned uid() const { return m_uid; }
  /*
   * [한국어]
   * get_opcode() — PTX 오퍼코드 정수 반환
   *
   * @return: m_opcode — PTX 오퍼코드 정수 (-1이면 레이블)
   *
   * instructions.cc의 명령어 dispatch 테이블에서 실행 함수를 선택하는 인덱스.
   * -1이면 레이블이므로 실행하지 않음.
   */
  int get_opcode() const { return m_opcode; }
  /*
   * [한국어]
   * get_opcode_cstr() — PTX 오퍼코드를 사람이 읽을 수 있는 문자열로 반환
   *
   * @return: g_opcode_string[m_opcode] 문자열; 레이블(-1)이면 "label"
   *
   * 디버그 출력, print_insn() 등에서 오퍼코드 이름을 출력할 때 사용.
   */
  const char *get_opcode_cstr() const {
    if (m_opcode != -1) {
      return g_opcode_string[m_opcode]; // [한국어] 오퍼코드 정수로 전역 문자열 테이블 조회
    } else {
      return "label"; // [한국어] m_opcode == -1 이면 레이블 (실제 명령어 아님)
    }
  }
  /*
   * [한국어]
   * source_file() — 소스 파일 이름 반환
   *
   * @return: m_source_file.c_str() — PTX 소스의 원본 파일 경로 문자열
   *
   * 디버그 정보 출력, 에러 메시지에서 소스 위치 표시에 사용.
   */
  const char *source_file() const { return m_source_file.c_str(); }
  /*
   * [한국어]
   * source_line() — 소스 파일 행 번호 반환
   *
   * @return: m_source_line — PTX 소스에서 이 명령어가 위치한 줄 번호
   *
   * 디버그 정보 출력, 에러 메시지에서 소스 위치 표시에 사용.
   */
  unsigned source_line() const { return m_source_line; }
  /*
   * [한국어]
   * get_num_operands() — 피연산자 개수 반환
   *
   * @return: m_operands.size() — 이 명령어의 총 피연산자 수 (dst + 모든 src)
   *
   * 명령어 피연산자를 순회하거나 특정 인덱스 피연산자에 접근하기 전 범위 확인.
   */
  unsigned get_num_operands() const { return m_operands.size(); }
  /*
   * [한국어]
   * has_pred() — 이 명령어에 프레디케이트가 있는지 검사
   *
   * @return: true이면 @%p 또는 @!%p 형태의 프레디케이트 가드 존재
   *
   * 명령어 실행 시 프레디케이트 레지스터 값을 확인하여 실행 여부 결정.
   * m_pred == NULL이면 무조건 실행.
   */
  bool has_pred() const { return m_pred != NULL; }
  /*
   * [한국어]
   * get_pred() — 프레디케이트 피연산자 반환 (operand_info 값 복사)
   *
   * @return: 프레디케이트 심볼에서 생성된 operand_info 객체
   *
   * 명령어 실행 시 프레디케이트 레지스터 값을 읽기 위해 호출.
   * has_pred()가 true일 때만 유효하게 호출할 수 있다.
   * 구현은 ptx_ir.cc에 있으며 m_pred 심볼로부터 operand_info를 생성하여 반환.
   */
  operand_info get_pred() const;
  /*
   * [한국어]
   * get_pred_neg() — 프레디케이트 부정 여부 반환
   *
   * @return: true이면 @!%p 형태 (프레디케이트가 false일 때 실행)
   *
   * 명령어 실행 시 프레디케이트를 반전하여 적용해야 하는지 판단.
   */
  bool get_pred_neg() const { return m_neg_pred; }
  /*
   * [한국어]
   * get_pred_mod() — 프레디케이트 수식자 반환
   *
   * @return: m_pred_mod — 프레디케이트 수식자 코드
   *
   * PTXPlus 확장에서 사용되는 프레디케이트 수식자 값.
   */
  int get_pred_mod() const { return m_pred_mod; }
  /*
   * [한국어]
   * get_source() — 원본 PTX 소스 텍스트 반환
   *
   * @return: m_source.c_str() — 이 명령어에 해당하는 원본 PTX 소스 줄 텍스트
   *
   * 디버그 출력, print_insn() 에서 소스 텍스트를 함께 표시할 때 사용.
   */
  const char *get_source() const { return m_source.c_str(); }

  /*
   * [한국어]
   * get_scalar_type() — 명령어의 스칼라 타입 목록 반환
   *
   * @return: m_scalar_type 리스트의 복사본 (PTX 타입 수식자 정수 코드)
   *
   * get_type()/get_type2()로 첫 번째/두 번째 타입에 개별 접근하는 것과 달리
   * 전체 타입 목록이 필요한 경우 사용.
   */
  const std::list<int> get_scalar_type() const { return m_scalar_type; }
  /*
   * [한국어]
   * get_options() — 명령어 수식어 옵션 목록 반환
   *
   * @return: m_options 리스트의 복사본 (PTX 수식어 정수 코드)
   *
   * 명령어 실행 시 특수 동작 제어 옵션(.rn, .ca, .cg 등)이 필요할 때 사용.
   */
  const std::list<int> get_options() const { return m_options; }

  typedef std::vector<operand_info>::const_iterator const_iterator;
  /* [한국어] const_iterator — m_operands 벡터의 상수 반복자 타입 별칭.
   * has_memory_read()/has_memory_write()에서 피연산자 순회에 사용. */

  /*
   * [한국어]
   * op_iter_begin() — 피연산자 벡터의 시작 반복자 반환
   *
   * @return: m_operands.begin() — 첫 번째 피연산자(dst)를 가리키는 const 반복자
   *
   * 모든 피연산자를 순회해야 할 때 사용 (예: has_memory_read/write 검사).
   */
  const_iterator op_iter_begin() const { return m_operands.begin(); }

  /*
   * [한국어]
   * op_iter_end() — 피연산자 벡터의 끝 반복자 반환
   *
   * @return: m_operands.end() — 마지막 피연산자 다음을 가리키는 const 반복자
   *
   * op_iter_begin()과 함께 피연산자 범위 순회 종료 조건으로 사용.
   */
  const_iterator op_iter_end() const { return m_operands.end(); }

  /*
   * [한국어]
   * dst() (const 버전) — 첫 번째 피연산자(목적지 레지스터) 반환
   *
   * @return: m_operands[0] — 명령어 결과를 쓸 목적지 피연산자 (const 참조)
   *
   * PTX 명령어에서 피연산자[0]은 항상 목적지(destination). 명령어 실행 후
   * 결과를 쓸 레지스터 또는 메모리 주소를 나타낸다. m_operands가 비어 있으면 assert 중단.
   *
   * 호출 체인:
   *   instructions.cc 명령어 실행 → [dst()] → operand_info 참조
   */
  const operand_info &dst() const {
    assert(!m_operands.empty()); // [한국어] 피연산자 없는 명령어에서 목적지 접근 방지
    return m_operands[0];        // [한국어] 첫 번째 피연산자가 항상 목적지
  }

  /*
   * [한국어]
   * func_addr() — CALL 명령어의 함수 주소 피연산자 반환
   *
   * @return: 함수 주소를 담은 operand_info const 참조
   *
   * CALL 명령어는 [반환변수,] 함수주소, [인수...] 형태의 피연산자 구조를 가진다.
   * 반환 변수가 첫 번째 피연산자인 경우 두 번째 피연산자가 함수 주소이고,
   * 반환 변수 없이 함수 주소가 첫 번째 피연산자인 경우도 있다.
   * is_return_var()로 첫 번째 피연산자가 반환 변수인지 확인 후 함수 주소 결정.
   *
   * 호출 체인:
   *   instructions.cc CALL 명령어 실행 → [func_addr()] → operand_info 참조
   */
  const operand_info &func_addr() const {
    assert(!m_operands.empty()); // [한국어] 피연산자 없는 명령어에서 함수 주소 접근 방지
    if (!m_operands[0].is_return_var()) {
      return m_operands[0]; // [한국어] 반환 변수 없음 — 첫 번째 피연산자가 함수 주소
    } else {
      assert(m_operands.size() >= 2); // [한국어] 반환 변수 있으면 최소 2개 피연산자 필요
      return m_operands[1]; // [한국어] 반환 변수 다음이 함수 주소
    }
  }

  /*
   * [한국어]
   * dst() (non-const 버전) — 목적지 피연산자 수정 가능 참조 반환
   *
   * @return: m_operands[0] — 수정 가능한 목적지 피연산자 참조
   *
   * 파서 또는 ptx_assemble() 단계에서 목적지 피연산자를 수정할 때 사용.
   * 실행 시에는 const 버전을 사용.
   */
  operand_info &dst() {
    assert(!m_operands.empty()); // [한국어] 피연산자 없는 명령어에서 목적지 접근 방지
    return m_operands[0];        // [한국어] 첫 번째 피연산자(목적지)를 수정 가능 참조로 반환
  }

  /*
   * [한국어]
   * src1()~src8() — 소스 피연산자 접근자 (인덱스 1~8)
   *
   * @return: m_operands[n] — n번째 피연산자 (const 참조)
   *          n=1: 첫 번째 소스, n=2: 두 번째 소스, ..., n=8: 여덟 번째 소스
   *
   * PTX 명령어에서 m_operands[0]은 dst, [1]부터가 소스 피연산자.
   * 각 함수는 접근 전에 피연산자 배열 크기를 assert로 검사하여
   * 존재하지 않는 소스에 접근하면 즉시 중단된다.
   * src3()~src8()은 다중 소스가 필요한 명령어(WMMA 등)에서 사용.
   *
   * 호출 체인:
   *   instructions.cc 명령어 실행 → [src1()..src8()] → operand_info 참조
   */
  const operand_info &src1() const {
    assert(m_operands.size() > 1); // [한국어] 소스 1이 없는 명령어에서 접근 방지
    return m_operands[1];          // [한국어] 인덱스 1 = 첫 번째 소스 피연산자
  }

  const operand_info &src2() const {
    assert(m_operands.size() > 2); // [한국어] 소스 2가 없는 명령어에서 접근 방지
    return m_operands[2];          // [한국어] 인덱스 2 = 두 번째 소스 피연산자
  }

  const operand_info &src3() const {
    assert(m_operands.size() > 3); // [한국어] 소스 3이 없는 명령어에서 접근 방지
    return m_operands[3];          // [한국어] 인덱스 3 = 세 번째 소스 피연산자
  }
  const operand_info &src4() const {
    assert(m_operands.size() > 4); // [한국어] 소스 4가 없는 명령어에서 접근 방지
    return m_operands[4];          // [한국어] 인덱스 4 = 네 번째 소스 피연산자
  }
  const operand_info &src5() const {
    assert(m_operands.size() > 5); // [한국어] 소스 5가 없는 명령어에서 접근 방지
    return m_operands[5];          // [한국어] 인덱스 5 = 다섯 번째 소스 피연산자
  }
  const operand_info &src6() const {
    assert(m_operands.size() > 6); // [한국어] 소스 6이 없는 명령어에서 접근 방지
    return m_operands[6];          // [한국어] 인덱스 6 = 여섯 번째 소스 피연산자
  }
  const operand_info &src7() const {
    assert(m_operands.size() > 7); // [한국어] 소스 7이 없는 명령어에서 접근 방지
    return m_operands[7];          // [한국어] 인덱스 7 = 일곱 번째 소스 피연산자
  }
  const operand_info &src8() const {
    assert(m_operands.size() > 8); // [한국어] 소스 8이 없는 명령어에서 접근 방지
    return m_operands[8];          // [한국어] 인덱스 8 = 여덟 번째 소스 피연산자
  }

  /*
   * [한국어]
   * operand_lookup() — 인덱스로 임의 피연산자 접근
   *
   * @n: 피연산자 인덱스 (0=dst, 1=src1, ...)
   * @return: m_operands[n] — n번째 피연산자 (const 참조)
   *
   * dst()/src1()~src8()의 일반화 버전. 피연산자 인덱스를 런타임에 결정해야 할 때
   * 또는 피연산자 배열을 루프로 순회할 때 사용.
   * n >= m_operands.size()이면 assert로 중단.
   */
  const operand_info &operand_lookup(unsigned n) const {
    assert(n < m_operands.size()); // [한국어] 유효 인덱스 범위 검사
    return m_operands[n];          // [한국어] n번째 피연산자 반환
  }
  /*
   * [한국어]
   * has_return() — 이 명령어(CALL)에 반환 변수가 있는지 검사
   *
   * @return: true이면 CALL 명령어의 반환 값을 받을 변수가 있음
   *
   * m_return_var.is_valid()로 반환 변수 피연산자 유효성 확인.
   * instructions.cc CALL 실행 시 반환 값 복사 여부를 결정.
   */
  bool has_return() const { return m_return_var.is_valid(); }

  /*
   * [한국어]
   * get_space() — 명령어의 메모리 주소 공간 지정자 반환
   *
   * @return: m_space_spec — memory_space_t enum (global, shared, local, const 등)
   *
   * ld/st/atom 명령어에서 접근할 메모리 공간을 명시.
   * 시뮬레이터에서 캐시 경로(공유 메모리, L1/L2/DRAM) 분기에 사용.
   */
  memory_space_t get_space() const { return m_space_spec; }
  /*
   * [한국어]
   * get_vector() — 명령어의 벡터 폭 지정자 반환
   *
   * @return: m_vector_spec — 벡터 폭 (1=스칼라, 2=v2, 4=v4 등)
   *
   * ld.v4/st.v4 등 벡터 메모리 명령어의 벡터 폭을 나타냄.
   * generate_mem_accesses()에서 벡터 폭에 따라 메모리 트랜잭션 수 결정.
   */
  unsigned get_vector() const { return m_vector_spec; }
  /*
   * [한국어]
   * get_atomic() — 원자 연산 종류 반환
   *
   * @return: m_atomic_spec — atom 명령어의 연산 종류 코드 (add, exch, cas 등)
   *
   * atom 명령어 실행 시 어떤 원자 연산을 수행할지 결정하는 데 사용.
   */
  unsigned get_atomic() const { return m_atomic_spec; }

  /*
   * [한국어]
   * get_wmma_type() — WMMA 명령어 타입 반환
   *
   * @return: m_wmma_type — wmma.load/wmma.store/wmma.mma 구분 코드
   *
   * Tensor Core(WMMA) 명령어 실행 시 동작 종류를 구분.
   */
  int get_wmma_type() const { return m_wmma_type; }
  /*
   * [한국어]
   * get_wmma_layout() — WMMA 행렬 배치(layout) 반환
   *
   * @index: 0=Matrix D (결과), 1=Matrix C (누적)
   * @return: m_wmma_layout[index] — row-major/col-major 구분 코드
   *
   * WMMA 명령어에서 행렬 A, B, C, D의 메모리 배치 순서(row/col major)를 결정.
   */
  int get_wmma_layout(int index) const {
    return m_wmma_layout[index];  // 0->Matrix D,1->Matrix C // [한국어] 0=행렬 D 배치, 1=행렬 C 배치 반환
  }
  /*
   * [한국어]
   * get_type() — 명령어의 첫 번째 스칼라 타입 반환
   *
   * @return: m_scalar_type.front() — 주 데이터 타입 코드 (예: S32_TYPE, F32_TYPE)
   *
   * 명령어 실행 시 피연산자 데이터 타입을 결정하는 데 사용.
   * m_scalar_type이 비어 있으면 assert 중단.
   */
  int get_type() const {
    assert(!m_scalar_type.empty()); // [한국어] 타입 정보 없는 명령어에서 접근 방지
    return m_scalar_type.front();   // [한국어] 리스트의 첫 번째(주 타입) 반환
  }

  /*
   * [한국어]
   * get_type2() — 명령어의 두 번째 스칼라 타입 반환
   *
   * @return: m_scalar_type.back() — 변환(cvt) 명령어의 소스 타입 코드
   *
   * cvt.f32.s32 처럼 소스와 목적지 타입이 다른 변환 명령어에서 두 번째 타입.
   * m_scalar_type에 정확히 2개의 타입이 있어야 하며, 아니면 assert 중단.
   */
  int get_type2() const {
    assert(m_scalar_type.size() == 2); // [한국어] 타입 변환 명령어만 두 번째 타입 보유
    return m_scalar_type.back();       // [한국어] 리스트의 마지막(소스 타입) 반환
  }

  /*
   * [한국어]
   * assign_bb() — 이 명령어를 특정 기본 블록에 배정
   *
   * @basic_block: 이 명령어가 속하는 basic_block_t 포인터
   * @return: 없음 (void)
   *
   * function_info::create_basic_blocks()에서 각 명령어가 어느 기본 블록에
   * 속하는지 기록할 때 호출. m_basic_block 역참조 포인터를 설정.
   * 이후 get_bb()로 역참조하여 CFG 분석 정보에 접근 가능.
   *
   * 호출 체인:
   *   function_info::create_basic_blocks() → [assign_bb(bb)] → m_basic_block 저장
   */
  void assign_bb(
      basic_block_t *basic_block)  // assign instruction to a basic block
  {
    m_basic_block = basic_block; // [한국어] 이 명령어가 속한 기본 블록 역참조 저장
  }
  /*
   * [한국어]
   * get_bb() — 이 명령어가 속한 기본 블록 포인터 반환
   *
   * @return: m_basic_block — assign_bb()로 설정된 기본 블록 포인터
   *
   * CFG 분석 결과에 접근하거나 명령어의 CFG 위치를 확인할 때 사용.
   */
  basic_block_t *get_bb() { return m_basic_block; }
  /*
   * [한국어]
   * set_m_instr_mem_index() — 명령어 메모리 배열 인덱스 저장
   *
   * @index: m_instr_mem 배열에서 이 명령어의 인덱스 (= PC - m_start_PC)
   * @return: 없음 (void)
   *
   * ptx_assemble()에서 명령어를 m_instr_mem 배열에 배치할 때 인덱스를 기록.
   * get_instruction(PC)의 역방향으로, 명령어로부터 인덱스를 역조회할 수 있게 함.
   */
  void set_m_instr_mem_index(unsigned index) { m_instr_mem_index = index; }
  /*
   * [한국어]
   * set_PC() — 이 명령어의 시뮬레이터 PC 설정
   *
   * @PC: 시뮬레이터 가상 PC 값 (m_start_PC + 인덱스 형태)
   * @return: 없음 (void)
   *
   * ptx_assemble()에서 각 명령어에 PC를 부여할 때 호출.
   * 이후 get_PC()로 조회하여 분기 주소 계산, SIMT 스택 등에서 사용.
   */
  void set_PC(addr_t PC) { m_PC = PC; }
  /*
   * [한국어]
   * get_PC() — 이 명령어의 시뮬레이터 PC 반환
   *
   * @return: m_PC — ptx_assemble()에서 부여된 가상 PC
   *
   * 분기 명령어 실행 시 목적지 PC 계산, SIMT 스택 reconvergence 지점 설정 등.
   */
  addr_t get_PC() const { return m_PC; }

  /*
   * [한국어]
   * get_m_instr_mem_index() — 명령어 메모리 배열 인덱스 반환
   *
   * @return: m_instr_mem_index — set_m_instr_mem_index()로 저장된 배열 인덱스
   *
   * 명령어에서 역방향으로 m_instr_mem 배열 위치를 참조할 때 사용.
   */
  unsigned get_m_instr_mem_index() { return m_instr_mem_index; }
  /*
   * [한국어]
   * get_cmpop() — 비교 연산 종류 반환
   *
   * @return: m_compare_op — setp/set 명령어의 비교 연산 코드 (eq, ne, lt, gt 등)
   *
   * setp(set predicate) 명령어 실행 시 어떤 비교 연산을 수행할지 결정.
   */
  unsigned get_cmpop() const { return m_compare_op; }
  /*
   * [한국어]
   * get_label() — 레이블 심볼 포인터 반환
   *
   * @return: m_label — 이 명령어가 레이블이면 해당 심볼 포인터; 아니면 NULL
   *
   * CFG 구성 시 레이블 위치를 식별하고 분기 목적지 해석에 사용.
   */
  const symbol *get_label() const { return m_label; }
  /*
   * [한국어]
   * is_label() — 이 명령어가 레이블인지 검사
   *
   * @return: true이면 레이블 (m_opcode == -1이고 m_label != NULL)
   *
   * 실제 실행 명령어와 레이블을 구분. 레이블은 명령어 시퀀스에 포함되지만
   * 실행되지 않고 분기 목적지 식별자로만 사용됨.
   */
  bool is_label() const {
    if (m_label) {
      assert(m_opcode == -1); // [한국어] 레이블이면 반드시 오퍼코드 없음 (-1) 확인
      return true;
    }
    return false;
  }
  /*
   * [한국어]
   * is_hi()/is_lo()/is_wide() — PTXPlus 64비트 분리 연산 수식자 검사
   *
   * @return: 각 수식자 플래그 값
   *
   * is_hi(): .hi 수식자 — 상위 32비트 결과 추출 (예: mul.hi.u32)
   * is_lo(): .lo 수식자 — 하위 32비트 결과 추출 (예: mul.lo.u32)
   * is_wide(): .wide 수식자 — 64비트 와이드 결과 생성 (예: mul.wide.s32)
   * PTXPlus/SASS 명령어에서 64비트 연산의 상위/하위/전체 결과 선택.
   */
  bool is_hi() const { return m_hi; }
  bool is_lo() const { return m_lo; }
  bool is_wide() const { return m_wide; }
  /*
   * [한국어]
   * is_uni() — 균등 분기(uniform branch) 여부 반환
   *
   * @return: true이면 .uni 수식자가 붙은 브랜치 — 워프 내 모든 스레드가 동일 경로
   *
   * 균등 분기는 발산(divergence)이 없으므로 SIMT 스택 push 없이 단순 점프.
   * 워프 스케줄러에서 균등 분기를 최적화 처리할 때 사용.
   */
  bool is_uni() const { return m_uni; }
  /*
   * [한국어]
   * is_exit() — 함수 종료 명령어(exit/ret)인지 검사
   *
   * @return: true이면 함수 실행을 종료하는 명령어
   *
   * PTX exit 명령어 또는 return 명령어. 스레드 실행 완료 처리에 사용.
   */
  bool is_exit() const { return m_exit; }
  /*
   * [한국어]
   * is_abs() — .abs 수식자 여부 반환
   *
   * @return: true이면 절대값 수식자 (.abs) 적용
   *
   * abs 명령어 또는 .abs 수식자가 붙은 명령어에서 절대값 연산 적용 여부.
   */
  bool is_abs() const { return m_abs; }
  /*
   * [한국어]
   * is_neg() — .neg 수식자 여부 반환
   *
   * @return: true이면 부호 반전 수식자 (.neg) 적용
   *
   * neg 명령어 또는 .neg 수식자에서 부호 반전 연산 적용 여부.
   */
  bool is_neg() const { return m_neg; }
  /*
   * [한국어]
   * is_to() — .to 수식자 여부 반환
   *
   * @return: true이면 .to 수식자 — tex 명령어에서 목적지 지정 수식자
   *
   * PTX tex 명령어에서 .to 옵션이 있으면 별도 처리 경로 사용.
   */
  bool is_to() const { return m_to_option; }
  /*
   * [한국어]
   * cache_option() — 캐시 동작 힌트 반환
   *
   * @return: m_cache_option — ld/st 명령어의 캐시 힌트 코드
   *          (예: .ca=캐시 all levels, .cg=캐시 global, .cs=streaming, .wb=writeback)
   *
   * L1/L2 캐시 정책 결정에 사용. 시뮬레이터의 캐시 모델에서 캐시 여부 판단.
   */
  unsigned cache_option() const { return m_cache_option; }
  /*
   * [한국어]
   * rounding_mode() — 부동소수점 반올림 모드 반환
   *
   * @return: m_rounding_mode — PTX 반올림 수식자 코드
   *          (예: .rn=가장 가까운 짝수, .rz=0 방향, .rm=음수 방향, .rp=양수 방향)
   *
   * FP 연산(add.f32, mul.f64 등) 실행 시 반올림 방향 제어.
   */
  unsigned rounding_mode() const { return m_rounding_mode; }
  /*
   * [한국어]
   * saturation_mode() — 포화(saturation) 모드 반환
   *
   * @return: m_saturation_mode — .sat 수식자 여부 코드
   *
   * .sat 수식자가 있으면 결과를 타입 범위로 클램핑(포화).
   * FP/정수 명령어에서 오버플로우 방지 처리.
   */
  unsigned saturation_mode() const { return m_saturation_mode; }
  /*
   * [한국어]
   * clamp_mode() — 텍스처 클램프 모드 반환
   *
   * @return: m_clamp_mode — tex 명령어의 경계 처리 코드 (clamp/wrap 등)
   *
   * 텍스처 좌표가 [0,1] 범위를 벗어날 때 경계 처리 방식 결정.
   */
  unsigned clamp_mode() const { return m_clamp_mode; }
  /*
   * [한국어]
   * left_mode() — 왼쪽 이동 모드 반환
   *
   * @return: m_left_mode — 시프트 방향 코드
   *
   * 비트 시프트 명령어(shl/shr)의 방향 지정.
   */
  unsigned left_mode() const { return m_left_mode; }
  /*
   * [한국어]
   * dimension() — 텍스처 기하 지정자 반환
   *
   * @return: m_geom_spec — tex 명령어의 텍스처 차원 코드 (1D, 2D, 3D)
   *
   * 텍스처 접근 시 몇 차원 텍스처인지 결정하는 데 사용.
   */
  unsigned dimension() const { return m_geom_spec; }
  /*
   * [한국어]
   * barrier_op() — 배리어 연산 종류 반환
   *
   * @return: m_barrier_op — bar 명령어의 동기화 타입 코드 (sync, arrive, red 등)
   *
   * PTX bar(barrier) 명령어의 동작 종류를 결정. 특히 bar.sync는
   * SM 내 워프 간 동기화 (__syncthreads())를 구현.
   */
  unsigned barrier_op() const { return m_barrier_op; }
  /*
   * [한국어]
   * shfl_op() — SHFL 셔플 연산 종류 반환
   *
   * @return: m_shfl_op — shfl 명령어의 셔플 타입 코드 (up/down/bfly/idx)
   *
   * PTX shfl 명령어(워프 내 레지스터 교환)의 방향/방식 결정.
   */
  unsigned shfl_op() const { return m_shfl_op; }
  /*
   * [한국어]
   * prmt_op() — PRMT 바이트 퍼뮤테이션 연산 종류 반환
   *
   * @return: m_prmt_op — prmt 명령어의 바이트 재배치 모드 코드
   *
   * PTX prmt 명령어(바이트 퍼뮤테이션)의 재배치 방식 결정.
   */
  unsigned prmt_op() const { return m_prmt_op; }
  /*
   * [한국어]
   * vote_mode_t — PTX VOTE 명령어의 워프 투표 연산 모드 열거형
   *
   * vote_any:    워프 내 하나 이상의 스레드가 true이면 결과 true (.any)
   * vote_all:    워프 내 모든 스레드가 true이어야 결과 true (.all)
   * vote_uni:    워프 내 모든 스레드가 동일 값이면 true (.uni)
   * vote_ballot: 워프 내 각 스레드의 프레디케이트 비트를 32비트 마스크로 수집 (.ballot)
   *
   * CUDA의 __any_sync()/__all_sync()/__ballot_sync()의 PTX 구현.
   * 워프 수준 집합 연산으로 분기 조건이나 데이터 마스크 생성에 사용.
   */
  enum vote_mode_t { vote_any, vote_all, vote_uni, vote_ballot };
  /*
   * [한국어]
   * vote_mode() — VOTE 명령어의 투표 모드 반환
   *
   * @return: m_vote_mode — vote_mode_t enum 값
   *
   * PTX vote 명령어 실행 시 어떤 워프 집합 연산을 수행할지 결정.
   */
  enum vote_mode_t vote_mode() const { return m_vote_mode; }

  /*
   * [한국어]
   * membar_level() — MEMBAR 명령어의 메모리 배리어 범위 반환
   *
   * @return: m_membar_level — 배리어 가시성 범위 코드
   *          (예: .cta=CTA 내, .gl=전체 GPU, .sys=시스템 전체)
   *
   * PTX membar 명령어(메모리 펜스)의 효력 범위 결정.
   * .cta이면 같은 CTA(블록) 내 스레드 간, .gl이면 GPU 전체 스레드 간,
   * .sys이면 CPU 포함 전체 시스템 간 메모리 순서 보장.
   */
  int membar_level() const { return m_membar_level; }

  /*
   * [한국어]
   * has_memory_read() — 이 명령어가 메모리 읽기 접근을 하는지 검사
   *
   * @return: true이면 메모리에서 데이터를 읽는 명령어
   *
   * 표준 PTX: LD_OP, LDU_OP, TEX_OP, MMA_LD_OP이면 즉시 true.
   * PTXPlus 확장: 소스 피연산자(n>0) 중 is_memory_operand2()인 것이 있으면 true.
   * shader.cc 파이프라인에서 이 명령어가 메모리 요청을 발생시키는지 판단하는 데 사용.
   *
   * 호출 체인:
   *   shader.cc 파이프라인 → [has_memory_read()] → 메모리 접근 여부 판단
   */
  bool has_memory_read() const {
    if (m_opcode == LD_OP || m_opcode == LDU_OP || m_opcode == TEX_OP ||
        m_opcode == MMA_LD_OP)                               // [한국어] PTX 표준 메모리 읽기 명령어
      return true;
    // Check PTXPlus operand type below
    // Source operands are memory operands
    ptx_instruction::const_iterator op = op_iter_begin();
    for (int n = 0; op != op_iter_end(); op++, n++) {  // process operands // [한국어] 모든 피연산자 순회
      if (n > 0 && op->is_memory_operand2())           // source operands only // [한국어] 소스 피연산자(n>0)가 PTXPlus 메모리 피연산자이면
        return true;                                   // [한국어] 메모리 읽기 있음
    }
    return false; // [한국어] 메모리 읽기 없음
  }
  /*
   * [한국어]
   * has_memory_write() — 이 명령어가 메모리 쓰기 접근을 하는지 검사
   *
   * @return: true이면 메모리에 데이터를 쓰는 명령어
   *
   * 표준 PTX: ST_OP, MMA_ST_OP이면 즉시 true.
   * PTXPlus 확장: 목적지 피연산자(n=0)가 is_memory_operand2()이면 true.
   * shader.cc 파이프라인에서 메모리 쓰기 요청 발생 여부 판단.
   *
   * 호출 체인:
   *   shader.cc 파이프라인 → [has_memory_write()] → 메모리 접근 여부 판단
   */
  bool has_memory_write() const {
    if (m_opcode == ST_OP || m_opcode == MMA_ST_OP) return true; // [한국어] PTX 표준 메모리 쓰기 명령어
    // Check PTXPlus operand type below
    // Destination operand is a memory operand
    ptx_instruction::const_iterator op = op_iter_begin();
    for (int n = 0; (op != op_iter_end() && n < 1);
         op++, n++) {                          // process operands // [한국어] 첫 번째 피연산자(목적지)만 검사
      if (n == 0 && op->is_memory_operand2())  // source operands only // [한국어] 목적지(n=0)가 PTXPlus 메모리 피연산자이면
        return true;                           // [한국어] 메모리 쓰기 있음
    }
    return false; // [한국어] 메모리 쓰기 없음
  }

 private:
  /*
   * [한국어]
   * set_opcode_and_latency() — 오퍼코드에 따른 실행 레이턴시 및 파이프라인 유닛 타입 설정
   *
   * @return: 없음 (void)
   *
   * 생성자 또는 pre_decode() 내에서 호출. m_opcode를 분석하여
   * warp_inst_t 기반의 실행 레이턴시(op_latency), 파이프라인 유닛 타입,
   * ALU/MEM/SFU 분류 등을 설정한다.
   * gpgpusim.config의 파이프라인 파라미터와 연동하여 타이밍 모델을 구성.
   *
   * 호출 체인:
   *   ptx_instruction 생성자 또는 pre_decode() → [set_opcode_and_latency()]
   */
  void set_opcode_and_latency();
  /*
   * [한국어]
   * set_bar_type() — BAR(barrier) 명령어 세부 타입 설정
   *
   * @return: 없음 (void)
   *
   * bar 명령어 파싱 후 m_barrier_op에 sync/arrive/red 등의 세부 타입 코드 설정.
   * 타이밍 모델에서 barrier 동기화 처리 경로 분기에 사용.
   */
  void set_bar_type();
  /*
   * [한국어]
   * set_fp_or_int_archop() — FP 또는 정수 아키텍처 연산 분류 설정
   *
   * @return: 없음 (void)
   *
   * 오퍼코드가 부동소수점 연산인지 정수 연산인지 분류하여
   * warp_inst_t의 아키텍처 연산 타입 필드를 설정.
   * AccelWattch 전력 모델에서 연산 단위별 카운터 업데이트에 사용.
   */
  void set_fp_or_int_archop();
  /*
   * [한국어]
   * set_mul_div_or_other_archop() — 곱셈/나눗셈 또는 기타 아키텍처 연산 분류 설정
   *
   * @return: 없음 (void)
   *
   * MUL/DIV/REM 계열 명령어와 그 외 명령어를 분류.
   * 타이밍 모델에서 SFU(Special Function Unit) 또는 INT/FP 실행 유닛 선택에 사용.
   */
  void set_mul_div_or_other_archop();

  basic_block_t *m_basic_block;
  /* [한국어] 이 명령어가 속한 기본 블록(CFG 노드) 역참조 포인터.
   * 설정자: assign_bb() — create_basic_blocks() 완료 후 각 명령어에 블록 배정.
   * 읽는 자: get_bb() — CFG 분석 결과(지배자/후지배자 등)에 접근.
   * 값 범위: 유효한 basic_block_t 포인터; create_basic_blocks() 전에는 초기화 안 됨.
   * 동기화: CFG 분석 단일 스레드에서만 설정하므로 별도 락 불필요. */

  unsigned m_uid;
  /* [한국어] 이 명령어의 전역 고유 ID (파서가 순차적으로 부여).
   * 설정자: 생성자에서 전역 ID 카운터(gpgpu_ctx)를 통해 할당.
   * 읽는 자: uid() — 디버그/추적에서 명령어 식별.
   * 값 범위: 1부터 단조 증가.
   * 동기화: 파서 단일 스레드에서만 생성하므로 별도 락 불필요. */

  addr_t m_PC;
  /* [한국어] 이 명령어에 ptx_assemble()이 부여한 시뮬레이터 가상 PC.
   * 설정자: set_PC() — ptx_assemble()에서 각 명령어에 순차적으로 PC 부여.
   * 읽는 자: get_PC() — 분기 목적지 계산, SIMT 스택 reconvergence 설정.
   * 값 범위: m_start_PC 이상의 정수 (function_info::m_start_PC 기준 오프셋).
   * 동기화: ptx_assemble() 완료 후 읽기 전용이므로 별도 락 불필요. */

  std::string m_source_file;
  /* [한국어] 이 명령어의 원본 PTX 소스 파일 경로 문자열.
   * 설정자: 생성자의 @file 파라미터로 설정.
   * 읽는 자: source_file() — 디버그 정보 출력.
   * 값 범위: PTX 파일 경로 문자열 (빈 문자열 가능).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  unsigned m_source_line;
  /* [한국어] 이 명령어가 위치한 PTX 소스 파일 행 번호.
   * 설정자: 생성자의 @line 파라미터로 설정.
   * 읽는 자: source_line() — 디버그 정보 출력.
   * 값 범위: 1 이상의 양수.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  std::string m_source;
  /* [한국어] 이 명령어의 원본 PTX 소스 텍스트 문자열.
   * 설정자: 생성자의 @source 파라미터로 설정.
   * 읽는 자: get_source() — 디버그/출력 시 소스 텍스트 표시.
   * 값 범위: 원본 PTX 텍스트 (예: "add.f32 %f0, %f1, %f2;").
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  const symbol *m_pred;
  /* [한국어] 이 명령어의 프레디케이트 레지스터 심볼 포인터.
   * 설정자: 생성자의 @pred 파라미터로 설정; 프레디케이트 없으면 NULL.
   * 읽는 자: has_pred(), get_pred() — 프레디케이트 가드 여부 및 값 조회.
   * 값 범위: 유효한 symbol 포인터 또는 NULL(무조건 실행).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  bool m_neg_pred;
  /* [한국어] 프레디케이트 부정(@!%p) 여부.
   * 설정자: 생성자의 @neg_pred 파라미터로 설정.
   * 읽는 자: get_pred_neg() — 프레디케이트 평가 방향 반전 여부 판단.
   * 값 범위: true(@!%p 부정 실행) / false(@%p 긍정 실행).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  int m_pred_mod;
  /* [한국어] 프레디케이트 수식자 코드 (PTXPlus 확장).
   * 설정자: 생성자의 @pred_mod 파라미터로 설정.
   * 읽는 자: get_pred_mod() — PTXPlus 프레디케이트 수식 처리.
   * 값 범위: 파서 정의 수식자 코드.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  int m_opcode;
  /* [한국어] PTX 오퍼코드 정수 코드.
   * 설정자: 생성자의 @opcode 파라미터로 설정; -1이면 레이블.
   * 읽는 자: get_opcode(), get_opcode_cstr(), is_label(), has_memory_read/write(),
   *           instructions.cc dispatch 테이블 인덱스.
   * 값 범위: -1(레이블) 또는 LD_OP/ST_OP/ADD_OP 등 ptx_sim.h 정의 정수.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  const symbol *m_label;
  /* [한국어] 레이블 명령어의 심볼 포인터 (레이블이 아니면 NULL).
   * 설정자: 생성자의 @label 파라미터로 설정.
   * 읽는 자: get_label(), is_label() — 분기 목적지 레이블 식별.
   * 값 범위: 유효한 symbol 포인터 또는 NULL.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  std::vector<operand_info> m_operands;
  /* [한국어] 명령어의 모든 피연산자 벡터 ([0]=dst, [1]=src1, [2]=src2, ...).
   * 설정자: 생성자에서 @operands 리스트를 벡터로 변환하여 저장.
   * 읽는 자: dst(), src1()~src8(), operand_lookup(), op_iter_begin/end(),
   *           get_num_operands(), has_memory_read/write().
   * 값 범위: 0개(레이블) ~ 9개(WMMA 등 다중 소스) 피연산자.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  operand_info m_return_var;
  /* [한국어] CALL 명령어의 반환 값 수신 변수 피연산자.
   * 설정자: 생성자의 @return_var 파라미터로 설정; 반환 없으면 기본(invalid) operand_info.
   * 읽는 자: has_return() — is_valid()로 유효성 확인; instructions.cc CALL 실행 시.
   * 값 범위: 유효한 operand_info(CALL with return) 또는 invalid(반환 없음).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  std::list<int> m_options;
  /* [한국어] PTX 명령어 수식어 옵션 목록 (정수 코드 리스트).
   * 설정자: 생성자의 @options 파라미터로 설정.
   * 읽는 자: get_options() — 명령어 실행 시 특수 동작 옵션 확인.
   * 값 범위: PTX 수식어 코드들 (.rn, .ca, .cg 등 파서 정의 정수).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  std::list<int> m_wmma_options;
  /* [한국어] WMMA(Tensor Core) 명령어 전용 옵션 목록.
   * 설정자: 생성자의 @wmma_options 파라미터로 설정.
   * 읽는 자: WMMA 명령어 실행 코드 — 행렬 타입/크기 등 WMMA 특수 옵션.
   * 값 범위: WMMA 연산 관련 파서 정의 코드.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  bool m_wide;
  /* [한국어] .wide 수식자 플래그 — 64비트 와이드 결과 생성 여부.
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: is_wide() — mul.wide/mad.wide 명령어 실행 시 결과 폭 결정.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  bool m_hi;
  /* [한국어] .hi 수식자 플래그 — 64비트 연산의 상위 32비트 결과 선택.
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: is_hi() — mul.hi 명령어 실행 시 상위 반만 목적지에 저장.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  bool m_lo;
  /* [한국어] .lo 수식자 플래그 — 64비트 연산의 하위 32비트 결과 선택.
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: is_lo() — mul.lo 명령어 실행 시 하위 반만 목적지에 저장.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  bool m_exit;
  /* [한국어] 함수 종료 명령어(exit/ret) 여부 플래그.
   * 설정자: 생성자에서 @opcode가 EXIT_OP 등이면 설정.
   * 읽는 자: is_exit() — 스레드 실행 완료 처리 분기.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  bool m_abs;
  /* [한국어] .abs 수식자 플래그 — 절대값 연산 적용 여부.
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: is_abs() — abs/neg 수식자가 붙은 명령어 실행 시 절대값 처리.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  bool m_neg;
  /* [한국어] .neg 수식자 플래그 — 부호 반전 연산 적용 여부.
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: is_neg() — neg 수식자가 붙은 명령어 실행 시 부호 반전 처리.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  bool m_uni;  // if branch instruction, this evaluates to true for uniform
               // branches (ie jumps)
  /* [한국어] .uni 수식자 플래그 — 균등 분기(모든 스레드가 같은 경로) 여부.
   * 설정자: 생성자에서 @options 분석 시 .uni 수식자를 발견하면 true.
   * 읽는 자: is_uni() — 타이밍 모델에서 SIMT 스택 push 없이 단순 점프 처리.
   * 값 범위: true(.uni 분기) / false(조건 분기 또는 발산 가능 분기).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  bool m_to_option;
  /* [한국어] .to 수식자 플래그 — tex 명령어의 목적지 명시 수식자 여부.
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: is_to() — tex 명령어 실행 시 목적지 처리 분기.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  unsigned m_cache_option;
  /* [한국어] ld/st 명령어의 캐시 힌트 코드.
   * 설정자: 생성자에서 @options 분석 시 .ca/.cg/.cs/.wb 등을 정수로 저장.
   * 읽는 자: cache_option() — 시뮬레이터 캐시 모델에서 캐시 정책 결정.
   * 값 범위: 파서 정의 캐시 힌트 코드 (0=기본값).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  int m_wmma_type;
  /* [한국어] WMMA 명령어 종류 코드 (load/store/mma 구분).
   * 설정자: 생성자에서 @wmma_options 분석 시 설정.
   * 읽는 자: get_wmma_type() — WMMA 명령어 실행 분기.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  int m_wmma_layout[2];
  /* [한국어] WMMA 행렬 A/B/C/D 메모리 배치 코드 배열.
   * [0]=행렬 D 배치, [1]=행렬 C 배치 (row-major=0 / col-major=1).
   * 설정자: 생성자에서 @wmma_options 분석 시 설정.
   * 읽는 자: get_wmma_layout(index) — WMMA 명령어 실행 시 행렬 배치 결정.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  int m_wmma_configuration;
  /* [한국어] WMMA 연산 구성 코드 (행렬 크기 m×n×k 등 설정).
   * 설정자: 생성자에서 @wmma_options 분석 시 설정.
   * 읽는 자: WMMA 명령어 실행 코드.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  unsigned m_rounding_mode;
  /* [한국어] FP 연산의 반올림 모드 코드 (.rn/.rz/.rm/.rp).
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: rounding_mode() — FP 명령어 실행 시 반올림 방향 설정.
   * 값 범위: 파서 정의 반올림 코드 (0=기본값).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  unsigned m_compare_op;
  /* [한국어] setp/set 명령어의 비교 연산 종류 코드 (eq/ne/lt/gt/le/ge 등).
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: get_cmpop() — setp 명령어 실행 시 비교 연산 선택.
   * 값 범위: 파서 정의 비교 연산 코드.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  unsigned m_saturation_mode;
  /* [한국어] 포화(clamp) 모드 코드 (.sat 수식자).
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: saturation_mode() — FP/정수 결과를 타입 범위로 포화 적용.
   * 값 범위: 0(포화 없음) / 비0(포화 적용).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  unsigned m_clamp_mode;
  /* [한국어] 텍스처 경계 처리 모드 코드 (clamp/wrap 등).
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: clamp_mode() — tex 명령어 실행 시 경계 처리 방식 결정.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  unsigned m_left_mode;
  /* [한국어] 비트 시프트 방향 모드 코드.
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: left_mode() — shl/shr 명령어의 방향 결정.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  unsigned m_barrier_op;
  /* [한국어] bar 명령어의 동기화 연산 종류 코드 (sync/arrive/red 등).
   * 설정자: 생성자에서 set_bar_type()으로 설정.
   * 읽는 자: barrier_op() — 타이밍 모델에서 배리어 동기화 처리 경로 분기.
   * 값 범위: 파서 정의 배리어 연산 코드.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  unsigned m_shfl_op;
  /* [한국어] shfl 명령어의 셔플 방향 코드 (up/down/bfly/idx).
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: shfl_op() — 워프 내 레지스터 교환 방식 결정.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  unsigned m_prmt_op;
  /* [한국어] prmt 명령어의 바이트 퍼뮤테이션 모드 코드.
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: prmt_op() — 바이트 재배치 방식 결정.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  std::list<int> m_scalar_type;
  /* [한국어] 명령어의 스칼라 타입 코드 목록 (최대 2개 — 소스, 목적지 타입).
   * 설정자: 생성자의 @scalar_type 파라미터로 설정.
   * 읽는 자: get_type()(첫 번째), get_type2()(두 번째), get_scalar_type()(전체).
   *           명령어 실행 시 데이터 타입 결정.
   * 값 범위: S8/S16/S32/S64/U8..U64/F16/F32/F64 등 파서 정의 타입 코드.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  memory_space_t m_space_spec;
  /* [한국어] ld/st/atom 명령어의 메모리 주소 공간 지정자.
   * 설정자: 생성자의 @space_spec 파라미터로 설정.
   * 읽는 자: get_space() — 메모리 접근 시 공간별 시뮬레이션 경로 분기.
   * 값 범위: undefined_space, global_space, shared_space, local_space, const_space, tex_space.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  int m_geom_spec;
  /* [한국어] tex 명령어의 텍스처 차원 지정자 코드 (1D/2D/3D).
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: dimension() — 텍스처 접근 차원 결정.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  int m_vector_spec;
  /* [한국어] ld/st 명령어의 벡터 폭 지정자 (1/2/4/8).
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: get_vector() — 벡터 메모리 트랜잭션 수 계산.
   * 값 범위: 1(스칼라), 2(v2), 4(v4), 8(v8).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  int m_atomic_spec;
  /* [한국어] atom 명령어의 원자 연산 종류 코드 (add/exch/cas/and/or/xor/max/min 등).
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: get_atomic() — atom 명령어 실행 시 원자 연산 선택.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  enum vote_mode_t m_vote_mode;
  /* [한국어] vote 명령어의 워프 투표 연산 모드.
   * 설정자: 생성자에서 @options 분석 시 vote_any/vote_all/vote_uni/vote_ballot 중 설정.
   * 읽는 자: vote_mode() — vote 명령어 실행 시 집합 연산 선택.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  int m_membar_level;
  /* [한국어] membar 명령어의 메모리 배리어 범위 코드 (.cta/.gl/.sys).
   * 설정자: 생성자에서 @options 분석 시 설정.
   * 읽는 자: membar_level() — 메모리 펜스 범위 결정 (CTA/전체 GPU/시스템).
   * 값 범위: 파서 정의 배리어 범위 코드 (예: 0=.cta, 1=.gl, 2=.sys).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  int m_instr_mem_index;  // index into m_instr_mem array
  /* [한국어] function_info::m_instr_mem 배열에서 이 명령어의 인덱스 (= PC - m_start_PC).
   * 설정자: set_m_instr_mem_index() — ptx_assemble()에서 각 명령어 배치 시 설정.
   * 읽는 자: get_m_instr_mem_index() — 명령어에서 역방향으로 배열 위치 조회.
   * 값 범위: 0 ~ m_instr_mem_size-1.
   * 동기화: ptx_assemble() 완료 후 읽기 전용이므로 별도 락 불필요. */

  unsigned m_inst_size;   // bytes
  /* [한국어] 이 명령어의 바이트 크기 (보통 4 또는 8).
   * 설정자: 생성자 또는 pre_decode()에서 오퍼코드에 따라 설정.
   * 읽는 자: inst_size() — ptx_assemble()에서 다음 명령어 PC 계산 시 사용.
   * 값 범위: 4(일반 PTX) 또는 8(일부 특수 명령어).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  /*
   * [한국어]
   * pre_decode() — ptx_assemble() 이후 warp_inst_t 기반 필드를 초기화하는 사후 디코드
   *
   * @return: 없음 (void) — virtual 함수
   *
   * ptx_assemble()이 각 명령어에 PC를 부여한 뒤 호출됨.
   * 피연산자를 분석하여 warp_inst_t의 입력/출력 레지스터 벡터(in, out)를 채우고,
   * 메모리 접근 타입과 파이프라인 단계를 설정한다.
   * 타이밍 모델(shader.cc)이 이 명령어를 파이프라인에 배치할 때 이 정보를 사용.
   *
   * 호출 체인:
   *   function_info::ptx_assemble() → [pre_decode()] → warp_inst_t 필드 초기화
   */
  virtual void pre_decode();
  friend class function_info;
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  /* [한국어] gpgpu_context 역방향 포인터.
   * 설정자: 생성자의 @ctx 파라미터로 설정.
   * 읽는 자: 전역 상태 접근 (ID 카운터, 설정 옵션 등).
   * 값 범위: NULL이 아닌 유효한 gpgpu_context 포인터.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */
};

/*
 * [한국어]
 *
 * === 파일의 역할 (param_info) ===
 * PTX 커널 또는 디바이스 함수의 하나의 파라미터 정보를 저장하는 클래스.
 * 파라미터의 이름, 타입, 크기, 포인터 여부, 메모리 공간, 그리고 실제 값(add_data로 설정)을 관리.
 *
 * === 전체 아키텍처에서의 위치 ===
 * function_info::add_param_name_type_size()가 파싱 시 param_info를 생성하여
 * m_ptx_kernel_param_info 맵에 저장. cuLaunchKernel 경로에서 add_param_data()가
 * 실제 파라미터 값을 add_data()로 채운다. finalize()에서 .param 메모리에 복사.
 *
 * === 타 모듈과의 연결 ===
 * - function_info::m_ptx_kernel_param_info: 인덱스 → param_info 맵
 * - gpgpu_ptx_sim_arg: 호스트에서 전달된 파라미터 버퍼 구조체
 * - memory_space: finalize()에서 .param 메모리 공간에 값을 복사
 *
 * === 주요 함수/구조체 요약 ===
 * - param_info(): 기본 생성자 — 유효하지 않은 빈 파라미터 표현
 * - param_info(name,type,size,...): 선언 정보로 파라미터 초기화
 * - add_data(v): cuLaunchKernel 경로에서 실제 파라미터 값 저장
 * - get_value(): 저장된 파라미터 값 반환 (finalize 시 .param 메모리에 복사)
 * - is_ptr_shared(): 공유 메모리 포인터 파라미터 여부 — param_to_shared() 분기
 */
class param_info {
 public:
  /*
   * [한국어]
   * param_info() — 기본 생성자 (빈/유효하지 않은 파라미터 표현)
   *
   * @return: 없음 (생성자)
   *
   * 빈 파라미터 슬롯을 만들 때 사용. m_valid = false로 설정하여
   * 이후 is_valid() 또는 get_*() 호출 시 assert로 실수를 잡아냄.
   * function_info::m_ptx_kernel_param_info 맵 초기화 시 기본값으로 생성됨.
   */
  param_info() {
    m_valid = false;        // [한국어] 아직 유효한 파라미터 정보 없음
    m_value_set = false;    // [한국어] 실제 파라미터 값 미설정
    m_size = 0;             // [한국어] 파라미터 크기 0 (미결정)
    m_is_ptr = false;       // [한국어] 포인터 파라미터 아님
  }
  /*
   * [한국어]
   * param_info(name, type, size, is_ptr, ptr_space) — 파라미터 선언 정보로 초기화
   *
   * @name:      PTX 파라미터 이름 (예: "__param_0")
   * @type:      파라미터 데이터 타입 코드 (정수 enum)
   * @size:      파라미터 크기 (바이트)
   * @is_ptr:    true이면 포인터 타입 파라미터
   * @ptr_space: 포인터가 가리키는 메모리 공간 (shared/global 등)
   * @return: 없음 (생성자)
   *
   * function_info::add_param_name_type_size()에서 PTX 파서가 파라미터 선언을
   * 파싱한 후 각 파라미터의 타입/크기 정보를 저장하기 위해 호출.
   * 이 단계에서는 값(value)은 아직 설정되지 않음 — add_data()가 나중에 호출됨.
   *
   * 호출 체인:
   *   function_info::add_param_name_type_size() → [param_info(name,type,...)]
   */
  param_info(std::string name, int type, size_t size, bool is_ptr,
             memory_space_t ptr_space) {
    m_valid = true;           // [한국어] 유효한 파라미터 정보로 표시
    m_value_set = false;      // [한국어] 값은 아직 미설정 (add_data()로 나중에 채움)
    m_name = name;            // [한국어] 파라미터 이름 저장
    m_type = type;            // [한국어] 데이터 타입 코드 저장
    m_size = size;            // [한국어] 파라미터 크기(바이트) 저장
    m_is_ptr = is_ptr;        // [한국어] 포인터 타입 여부 저장
    m_ptr_space = ptr_space;  // [한국어] 포인터가 가리키는 메모리 공간 저장
  }
  /*
   * [한국어]
   * add_data() — 파라미터의 실제 값 저장
   *
   * @v: 호스트에서 전달된 파라미터 값 구조체 (param_t)
   * @return: 없음 (void)
   *
   * cuLaunchKernel 경로에서 function_info::add_param_data()가 호출.
   * 이미 값이 설정됐는데 크기가 다르면 assert로 중단 (동시 커널 실행 충돌).
   * m_value_set = true로 설정하여 이후 get_value()를 유효하게 만든다.
   *
   * 호출 체인:
   *   cuLaunchKernel → function_info::add_param_data() → [add_data(v)]
   */
  void add_data(param_t v) {
    assert((!m_value_set) ||
           (m_value.size == v.size));  // if this fails concurrent kernel
                                       // launches might execute incorrectly
    // [한국어] 이미 값이 설정됐다면 크기가 동일해야 함 — 크기 불일치는 동시 커널 실행 버그 신호
    m_value_set = true;  // [한국어] 값 설정 완료 플래그
    m_value = v;         // [한국어] 실제 파라미터 값(param_t) 저장
  }
  /*
   * [한국어]
   * add_offset() — .param 메모리 내 이 파라미터의 바이트 오프셋 저장
   *
   * @offset: .param 메모리 프레임 내 이 파라미터의 시작 바이트 오프셋
   * @return: 없음 (void)
   *
   * finalize()에서 파라미터들을 .param 메모리에 순차적으로 배치할 때
   * 각 파라미터의 오프셋을 기록.
   */
  void add_offset(unsigned offset) { m_offset = offset; }
  /*
   * [한국어]
   * get_offset() — .param 메모리 내 이 파라미터의 바이트 오프셋 반환
   *
   * @return: m_offset — .param 메모리 프레임 내 오프셋 (바이트)
   *
   * finalize()에서 .param 메모리에 값을 복사할 때 목적지 오프셋으로 사용.
   * m_valid가 false이면 assert 중단.
   */
  unsigned get_offset() {
    assert(m_valid); // [한국어] 유효한 파라미터만 오프셋 조회 허용
    return m_offset; // [한국어] .param 메모리 내 이 파라미터의 바이트 오프셋 반환
  }
  /*
   * [한국어]
   * get_name() — 파라미터 이름 반환
   *
   * @return: m_name — PTX 심볼 테이블의 파라미터 이름 문자열
   *
   * 디버그 출력, list_param() 등에서 파라미터 이름 표시에 사용.
   * m_valid가 false이면 assert 중단.
   */
  std::string get_name() const {
    assert(m_valid); // [한국어] 유효한 파라미터만 이름 조회 허용
    return m_name;   // [한국어] 파라미터 이름 문자열 반환
  }
  /*
   * [한국어]
   * get_type() — 파라미터 데이터 타입 코드 반환
   *
   * @return: m_type — 파라미터의 PTX 타입 코드 (S32/F32/U64 등)
   *
   * finalize()에서 .param 메모리에 값을 복사할 때 타입에 맞는 크기 결정.
   * m_valid가 false이면 assert 중단.
   */
  int get_type() const {
    assert(m_valid); // [한국어] 유효한 파라미터만 타입 조회 허용
    return m_type;   // [한국어] 파라미터 데이터 타입 코드 반환
  }
  /*
   * [한국어]
   * get_value() — 저장된 실제 파라미터 값 반환
   *
   * @return: m_value — param_t 구조체 (호스트에서 전달된 파라미터 값)
   *
   * finalize()에서 .param 메모리에 파라미터 값을 복사할 때 호출.
   * add_data()가 먼저 호출돼야 하며, m_value_set이 false이면 assert 중단.
   */
  param_t get_value() const {
    assert(m_value_set); // [한국어] add_data()로 값이 설정된 경우에만 반환 허용
    return m_value;      // [한국어] 호스트에서 전달된 파라미터 값(param_t) 반환
  }
  /*
   * [한국어]
   * get_size() — 파라미터의 바이트 크기 반환
   *
   * @return: m_size — 파라미터 크기 (바이트)
   *
   * .param 메모리 레이아웃 계산, add_param_data()에서 복사 크기 결정에 사용.
   * m_valid가 false이면 assert 중단.
   */
  size_t get_size() const {
    assert(m_valid); // [한국어] 유효한 파라미터만 크기 조회 허용
    return m_size;   // [한국어] 파라미터 크기(바이트) 반환
  }
  /*
   * [한국어]
   * is_ptr_shared() — 이 파라미터가 공유 메모리 포인터인지 검사
   *
   * @return: true이면 __shared__ 공간을 가리키는 포인터 파라미터
   *
   * function_info::param_to_shared()에서 공유 메모리 포인터 파라미터를
   * 특별히 처리할 때 이 검사를 사용. m_is_ptr && m_ptr_space == shared_space 조건.
   * m_valid가 false이면 assert 중단.
   */
  bool is_ptr_shared() const {
    assert(m_valid); // [한국어] 유효한 파라미터만 검사 허용
    return (m_is_ptr and m_ptr_space == shared_space); // [한국어] 포인터이면서 공유 메모리 공간 가리키면 true
  }

 private:
  bool m_valid;
  /* [한국어] 이 param_info 객체가 유효하게 초기화됐는지 여부.
   * 설정자: 기본 생성자에서 false, 파라미터화 생성자에서 true.
   * 읽는 자: get_offset/name/type/size/is_ptr_shared() — 유효성 사전 검사.
   * 값 범위: true(유효) / false(기본 생성자로 만든 빈 슬롯).
   * 동기화: 단일 스레드(파서/커널 실행 스레드)에서만 쓰므로 별도 락 불필요. */

  std::string m_name;
  /* [한국어] PTX 심볼 테이블의 파라미터 이름 (예: "__param_0", "%A").
   * 설정자: 파라미터화 생성자에서 @name으로 설정.
   * 읽는 자: get_name() — 디버그/목록 출력.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  int m_type;
  /* [한국어] 파라미터의 PTX 데이터 타입 코드 (S32/F32/U64/B8 등).
   * 설정자: 파라미터화 생성자에서 @type으로 설정.
   * 읽는 자: get_type() — 타입에 맞는 크기 결정.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  size_t m_size;
  /* [한국어] 파라미터의 크기 (바이트).
   * 설정자: 파라미터화 생성자에서 @size로 설정.
   * 읽는 자: get_size() — .param 메모리 레이아웃 및 복사 크기.
   * 값 범위: 1~8 바이트 (스칼라) 또는 구조체 크기.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  bool m_value_set;
  /* [한국어] add_data()가 호출돼 실제 파라미터 값이 저장됐는지 여부.
   * 설정자: add_data()에서 m_value = v 저장 후 true로 설정.
   * 읽는 자: get_value()의 사전 조건 검사 — false이면 assert 중단.
   *           add_data()의 동시 실행 감지 (이미 설정됐으면 크기 일치 검사).
   * 값 범위: true(값 설정됨) / false(미설정).
   * 동기화: 단일 스레드에서만 호출되므로 별도 락 불필요; 다중 커널 실행 시 버그 감지 assert 있음. */

  param_t m_value;
  /* [한국어] 호스트에서 cuLaunchKernel로 전달된 파라미터 실제 값.
   * 설정자: add_data() — gpgpu_ptx_sim_arg에서 복사.
   * 읽는 자: get_value() — finalize()에서 .param 메모리에 복사 시 사용.
   * 값 범위: param_t 유니온 (크기에 따라 u8/u16/u32/u64 중 하나 유효).
   * 동기화: add_data() 호출 후 불변; 동시 호출 시 assert로 감지. */

  unsigned m_offset;
  /* [한국어] .param 메모리 프레임 내 이 파라미터의 바이트 오프셋.
   * 설정자: add_offset() — function_info::finalize()에서 레이아웃 계산 후 설정.
   * 읽는 자: get_offset() — 실제 .param 메모리 복사 시 목적지 주소 계산.
   * 값 범위: 0 이상 (이전 파라미터 크기+정렬의 누적 합).
   * 동기화: finalize() 단일 스레드에서 설정하므로 별도 락 불필요. */

  bool m_is_ptr;
  /* [한국어] 이 파라미터가 포인터 타입인지 여부.
   * 설정자: 파라미터화 생성자에서 @is_ptr으로 설정.
   * 읽는 자: is_ptr_shared() — 공유 메모리 포인터 여부 판단.
   * 값 범위: true(포인터) / false(값 타입).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  memory_space_t m_ptr_space;
  /* [한국어] 포인터 파라미터가 가리키는 메모리 공간 (m_is_ptr == true일 때만 유효).
   * 설정자: 파라미터화 생성자에서 @ptr_space로 설정.
   * 읽는 자: is_ptr_shared() — shared_space이면 param_to_shared() 특수 처리.
   * 값 범위: shared_space, global_space 등 memory_space_t enum 값.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */
};

/*
 * [한국어]
 *
 * === 파일의 역할 (function_info) ===
 * PTX 커널 또는 디바이스 함수 하나를 표현하는 핵심 클래스. 함수 이름, 파라미터,
 * PTX 명령어 목록(m_instructions), 명령어 메모리 배열(m_instr_mem), CFG(기본 블록,
 * 지배자/후지배자), SIMT 재합류 쌍(reconvergence pairs)을 모두 관리한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   PTX 파서(ptx.y) → function_info 생성 → add_inst() → ptx_assemble()
 *   → create_basic_blocks() → connect_basic_blocks() → find_dominators()
 *   → find_postdominators() → find_ipostdominators() → do_pdom()
 *   → get_reconvergence_pairs() → simt_stack 설정
 *   → finalize() → .param 메모리 복사
 *   → 커널 실행: get_instruction(PC) → ptx_thread_info 실행
 *
 * === 타 모듈과의 연결 ===
 * - symbol_table (m_symtab): 이 함수의 심볼 테이블 — 레지스터/변수/레이블 모두 보유
 * - ptx_instruction: m_instructions(리스트)와 m_instr_mem(PC 인덱스 배열)에 저장
 * - basic_block_t: m_basic_blocks 벡터로 CFG 노드 관리
 * - gpgpu_recon_t: get_reconvergence_pairs()로 SIMT 스택에 제공
 * - memory_space: finalize()에서 .param 메모리 접근
 *
 * === 주요 함수/구조체 요약 ===
 * - ptx_assemble():        명령어 목록 → m_instr_mem 배열 변환 및 PC 부여
 * - do_pdom():             CFG 분석 전체 파이프라인 (블록 생성 → 연결 → dom/pdom → ipdom)
 * - get_reconvergence_pairs(): SIMT 재합류 쌍 배열 반환
 * - finalize():            .param 메모리에 파라미터 값 복사
 * - get_instruction(PC):   PC로 ptx_instruction 포인터 조회
 * - add_param_data():      cuLaunchKernel 경로의 파라미터 값 전달
 */
class function_info {
 public:
  /*
   * [한국어]
   * function_info() — PTX 함수/커널 정보 객체 생성자
   *
   * @entry_point: true이면 .entry 커널 (호스트에서 직접 실행); false이면 .func 디바이스 함수
   * @ctx:         gpgpu_context 역방향 포인터
   *
   * PTX 파서(ptx.y)에서 .entry 또는 .func 선언을 파싱할 때 호출.
   * m_instructions, m_instr_mem, m_basic_blocks 등이 이후 단계에서 채워진다.
   * 구현은 ptx_ir.cc에 있으며 m_uid 발급, 기본 필드 초기화를 수행.
   *
   * 호출 체인:
   *   PTX 파서 (ptx.y) → [function_info(entry_point, ctx)] → (이후 add_inst, ptx_assemble 등 호출)
   */
  function_info(int entry_point, gpgpu_context *ctx);
  /*
   * [한국어]
   * get_ptx_version() — 이 함수의 PTX 버전 정보 반환
   *
   * @return: m_symtab->get_ptx_version() — 이 함수가 속한 모듈의 PTX 버전
   *
   * PTX .version 지시자에서 파싱된 버전. set_kernel_info()에서 버전 정수로 변환.
   */
  const ptx_version &get_ptx_version() const {
    return m_symtab->get_ptx_version(); // [한국어] 심볼 테이블에서 PTX 버전 정보 조회
  }
  /*
   * [한국어]
   * ~function_info() — 소멸자 (가상)
   *
   * 파생 클래스(trace_function_info 등)의 소멸자를 올바르게 호출하기 위해 virtual.
   * 현재 본체는 비어 있으나 파생 클래스에서 override 가능.
   */
  virtual ~function_info() {}
  /*
   * [한국어]
   * get_sm_target() — 이 함수를 컴파일한 대상 SM 아키텍처 버전 반환
   *
   * @return: m_symtab->get_sm_target() — .target 지시자에서 파싱된 SM 버전 정수
   *
   * set_kernel_info()에서 m_kernel_info.sm_target에 저장. 시뮬레이션 설정과
   * 컴파일 타겟 일치 여부 확인에 사용.
   */
  unsigned get_sm_target() const { return m_symtab->get_sm_target(); }
  /*
   * [한국어]
   * is_extern() — 이 함수가 외부 선언(extern)인지 검사
   *
   * @return: m_extern — true이면 .extern 선언된 외부 함수 (본체 없음)
   *
   * 외부 함수는 실행할 본체 명령어가 없으므로 ptx_assemble() 대상에서 제외.
   */
  bool is_extern() const { return m_extern; }
  /*
   * [한국어]
   * set_name() — 이 함수의 이름 설정
   *
   * @name: PTX 함수 이름 문자열 (예: "_Z9vectorAddPfS_S_i")
   * @return: 없음 (void)
   *
   * PTX 파서에서 .entry/.func 이름을 파싱한 후 호출.
   */
  void set_name(const char *name) { m_name = name; }
  /*
   * [한국어]
   * set_symtab() — 이 함수의 심볼 테이블 설정
   *
   * @symtab: 이 함수 전용 symbol_table 포인터
   * @return: 없음 (void)
   *
   * PTX 파서에서 함수 스코프 심볼 테이블을 생성한 후 이 함수에 연결.
   */
  void set_symtab(symbol_table *symtab) { m_symtab = symtab; }
  /*
   * [한국어]
   * get_name() — 이 함수의 이름 반환
   *
   * @return: m_name — PTX 함수 이름 문자열
   *
   * 함수 조회 테이블(g_sym_name_to_symbol_table 등)에서 키로 사용.
   */
  std::string get_name() const { return m_name; }
  /*
   * [한국어]
   * print_insn() — PC 위치의 명령어를 파일 스트림에 출력하고 다음 PC 반환
   *
   * @pc:  출력할 명령어의 PC 값
   * @fp:  출력 대상 FILE 포인터
   * @return: 다음 명령어의 PC 값 (= pc + 명령어 크기)
   *
   * 디버그 또는 PTX 코드 덤프 시 명령어를 행 단위로 출력.
   */
  unsigned print_insn(unsigned pc, FILE *fp) const;
  /*
   * [한국어]
   * get_insn_str() — PC 위치의 명령어를 문자열로 반환
   *
   * @pc:  대상 명령어의 PC 값
   * @return: 해당 PC의 명령어 텍스트 문자열
   *
   * 로그 메시지에 명령어 내용을 포함할 때 사용.
   */
  std::string get_insn_str(unsigned pc) const;
  /*
   * [한국어]
   * add_inst() — PTX 명령어 리스트를 이 함수에 추가
   *
   * @instructions: 파서가 생성한 ptx_instruction 포인터 리스트
   * @return: 없음 (void)
   *
   * PTX 파서에서 함수 본체 파싱 완료 후 호출. m_instructions 를 교체.
   * 이후 ptx_assemble()에서 m_instr_mem 배열로 변환됨.
   *
   * 호출 체인:
   *   PTX 파서 (ptx.y) → [add_inst(instructions)] → m_instructions 설정
   */
  void add_inst(const std::list<ptx_instruction *> &instructions) {
    m_instructions = instructions; // [한국어] 파서가 생성한 명령어 리스트를 이 함수에 저장
  }
  /*
   * [한국어]
   * find_next_real_instruction() — 레이블/nop을 건너뛰고 다음 실제 명령어를 찾는 반복자 반환
   *
   * @i: 현재 명령어 리스트 반복자
   * @return: 레이블이 아닌 다음 실제 명령어를 가리키는 반복자
   *
   * create_basic_blocks()에서 기본 블록 경계를 결정할 때 레이블을 건너뛰기 위해 사용.
   */
  std::list<ptx_instruction *>::iterator find_next_real_instruction(
      std::list<ptx_instruction *>::iterator i);
  /*
   * [한국어]
   * create_basic_blocks() — CFG 기본 블록 생성
   *
   * @return: 없음 (void)
   *
   * m_instructions 리스트를 선형 스캔하여 기본 블록 경계(분기 명령어 직후,
   * 레이블 직전)를 결정하고 basic_block_t 객체들을 생성하여 m_basic_blocks에 저장.
   * do_pdom() 파이프라인의 첫 번째 단계.
   *
   * 호출 체인:
   *   do_pdom() → [create_basic_blocks()] → connect_basic_blocks()
   */
  void create_basic_blocks();

  /*
   * [한국어]
   * print_basic_blocks() — 모든 기본 블록 정보 출력 (디버그)
   *
   * @return: 없음 (void)
   *
   * m_basic_blocks의 각 블록에 포함된 명령어 범위, ID, 진입/종료 여부를 출력.
   * do_pdom() 후 CFG 구성 검증 시 사용.
   */
  void print_basic_blocks();

  /*
   * [한국어]
   * print_basic_block_links() — 기본 블록 간 에지(predecessor/successor) 출력 (디버그)
   *
   * @return: 없음 (void)
   */
  void print_basic_block_links();
  /*
   * [한국어]
   * print_basic_block_dot() — CFG를 Graphviz DOT 형식으로 출력 (디버그)
   *
   * @return: 없음 (void)
   *
   * CFG를 시각화하기 위해 DOT 언어로 기본 블록 간 연결을 출력.
   * dot 명령으로 PDF/PNG로 변환하여 CFG 구조 시각적 확인 가능.
   */
  void print_basic_block_dot();

  /*
   * [한국어]
   * find_break_target() — break 명령어의 탈출 목적지 피연산자 탐색
   *
   * @p_break_insn: break 명령어 포인터
   * @return: break가 이동할 목적지 레이블의 operand_info 포인터
   *
   * PTX의 break 명령어를 처리하기 위해 현재 루프의 종료 레이블을 역추적.
   */
  operand_info *find_break_target(
      ptx_instruction *p_break_insn);  // find the target of a break instruction
  /*
   * [한국어]
   * connect_basic_blocks() — 기본 블록 간 predecessor/successor 에지 연결
   *
   * @return: 없음 (void)
   *
   * 각 기본 블록의 마지막 명령어(분기/순차 진행)를 분석하여
   * successor_ids와 predecessor_ids를 설정한다.
   * do_pdom() 파이프라인의 두 번째 단계.
   *
   * 호출 체인:
   *   do_pdom() → [connect_basic_blocks()] → find_dominators()
   */
  void connect_basic_blocks();  // iterate across m_basic_blocks of function,
                                // connecting basic blocks together
  /*
   * [한국어]
   * connect_break_targets() — break 명령어를 적절한 목적지 블록에 연결
   *
   * @return: true이면 break 명령어가 있어서 처리됨; false이면 없음
   *
   * break 명령어의 분기 목적지를 CFG에 정확히 연결하는 보정 단계.
   * connect_basic_blocks() 완료 후 호출되어 루프 탈출 에지를 추가.
   */
  bool
  connect_break_targets();  // connecting break instructions with proper targets

  // iterate across m_basic_blocks of function,
  // finding dominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.14
  /*
   * [한국어]
   * find_dominators() — 모든 기본 블록의 지배자 집합 계산
   *
   * @return: 없음 (void)
   *
   * Muchnick 교재 Fig 7.14의 순방향 데이터 플로우 알고리즘으로
   * 각 기본 블록의 dominator_ids 집합을 계산.
   * 초기화: 진입 블록은 자기 자신만; 나머지 블록은 전체 집합.
   * 반복 고정점(fix-point)에 도달할 때까지 predecessor들의 집합 교차.
   *
   * 호출 체인:
   *   do_pdom() → [find_dominators()] → find_idominators()
   */
  void find_dominators();
  /*
   * [한국어]
   * print_dominators() — 각 기본 블록의 지배자 집합 출력 (디버그)
   */
  void print_dominators();
  /*
   * [한국어]
   * find_idominators() — 모든 기본 블록의 즉각 지배자(idom) 계산
   *
   * @return: 없음 (void)
   *
   * Muchnick Fig 7.15 알고리즘으로 각 블록의 immediatedominator_id를 결정.
   * find_dominators() 완료 후 호출.
   */
  void find_idominators();
  /*
   * [한국어]
   * print_idominators() — 각 기본 블록의 즉각 지배자 출력 (디버그)
   */
  void print_idominators();

  // iterate across m_basic_blocks of function,
  // finding postdominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.14
  /*
   * [한국어]
   * find_postdominators() — 모든 기본 블록의 후지배자 집합 계산
   *
   * @return: 없음 (void)
   *
   * find_dominators()의 역방향 버전 — 종료 블록에서 역방향으로 후지배자 계산.
   * 각 블록의 postdominator_ids를 설정. SIMT 재합류 분석의 핵심 단계.
   *
   * 호출 체인:
   *   do_pdom() → [find_postdominators()] → find_ipostdominators()
   */
  void find_postdominators();
  /*
   * [한국어]
   * print_postdominators() — 각 기본 블록의 후지배자 집합 출력 (디버그)
   */
  void print_postdominators();

  // iterate across m_basic_blocks of function,
  // finding immediate postdominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.15
  /*
   * [한국어]
   * find_ipostdominators() — 모든 기본 블록의 즉각 후지배자(ipdom) 계산
   *
   * @return: 없음 (void)
   *
   * Muchnick Fig 7.15 알고리즘으로 각 블록의 immediatepostdominator_id를 결정.
   * 분기 명령어 블록의 ipdom이 SIMT 재합류 지점(target_pc)이 된다.
   * find_postdominators() 완료 후 호출.
   *
   * 호출 체인:
   *   do_pdom() → [find_ipostdominators()] → get_reconvergence_pairs()
   */
  void find_ipostdominators();
  /*
   * [한국어]
   * print_ipostdominators() — 각 기본 블록의 즉각 후지배자 출력 (디버그)
   */
  void print_ipostdominators();
  /*
   * [한국어]
   * do_pdom() — CFG 분석 전체 파이프라인 실행 (핵심 함수)
   *
   * @return: 없음 (void)
   *
   * SIMT 재합류 분석을 위한 전체 CFG 분석 시퀀스를 순서대로 호출:
   *   1. create_basic_blocks() — 기본 블록 생성
   *   2. connect_basic_blocks() — CFG 에지 연결
   *   3. connect_break_targets() — break 목적지 연결
   *   4. find_dominators() → find_idominators() — 지배자 분석
   *   5. find_postdominators() → find_ipostdominators() — 후지배자 분석
   *   6. pdom_done = true로 설정
   * 최초 커널 실행 전 ptx_sim_init_thread()에서 is_pdom_set() 확인 후 한 번만 호출.
   *
   * 호출 체인:
   *   ptx_sim_init_thread() → [do_pdom()] → get_reconvergence_pairs()
   */
  void do_pdom();  // function to call pdom analysis

  /*
   * [한국어]
   * get_num_reconvergence_pairs() — SIMT 재합류 쌍의 총 개수 반환
   *
   * @return: num_reconvergence_pairs — do_pdom() 분석으로 결정된 재합류 쌍 수
   *
   * 호출자가 get_reconvergence_pairs()를 위해 gpgpu_recon_t 배열 크기를 알 때 사용.
   */
  unsigned get_num_reconvergence_pairs();

  /*
   * [한국어]
   * get_reconvergence_pairs() — 모든 SIMT 재합류 쌍을 배열에 채움
   *
   * @recon_points: 사전 할당된 gpgpu_recon_t 배열 포인터 (크기: get_num_reconvergence_pairs())
   * @return: 없음 (void)
   *
   * do_pdom() 완료 후 호출. 각 분기 명령어와 그 ipdom을 (source_pc, target_pc) 쌍으로
   * recon_points 배열에 채운다. simt_stack 초기화 시 이 배열로 재합류 지점 등록.
   *
   * 호출 체인:
   *   ptx_sim_init_thread() → [get_reconvergence_pairs()] → simt_stack 설정
   */
  void get_reconvergence_pairs(gpgpu_recon_t *recon_points);

  /*
   * [한국어]
   * get_function_size() — 함수의 총 명령어 수 반환
   *
   * @return: m_instructions.size() — 파서가 파싱한 PTX 명령어 수 (레이블 포함)
   *
   * 함수 크기 통계, 루프 경계 추정 등에 사용.
   */
  unsigned get_function_size() { return m_instructions.size(); }

  /*
   * [한국어]
   * ptx_assemble() — PTX 명령어 리스트를 m_instr_mem 배열로 변환하고 PC 부여
   *
   * @return: 없음 (void)
   *
   * m_instructions 리스트를 순회하여 각 명령어에 순차적 PC(m_start_PC + offset)를
   * 부여하고 m_instr_mem[offset] = instruction 으로 배열에 저장.
   * 레이블은 labels 맵에 등록되고, 분기 목적지 해석에 사용됨.
   * 이후 pre_decode()를 각 명령어에 호출하여 warp_inst_t 필드를 초기화.
   * m_assembled = true로 설정하여 중복 어셈블 방지.
   *
   * 호출 체인:
   *   gpgpu_ptx_assemble() → [ptx_assemble()] → ptx_instruction::pre_decode()
   */
  void ptx_assemble();

  /*
   * [한국어]
   * ptx_get_inst_op() — 현재 스레드가 실행할 명령어의 오퍼코드 반환 (실행-주도 시뮬레이션용)
   *
   * @thread: 현재 실행 중인 ptx_thread_info 포인터
   * @return: 해당 스레드의 현재 PC에 있는 명령어의 오퍼코드 정수
   *
   * 실행 주도(execution-driven) 시뮬레이션에서 현재 실행할 명령어를 결정할 때 사용.
   */
  unsigned ptx_get_inst_op(ptx_thread_info *thread);
  /*
   * [한국어]
   * add_param() — 이름과 값 쌍으로 커널 파라미터 추가 (m_kernel_params 맵)
   *
   * @name:  파라미터 이름 문자열
   * @value: param_t 값 구조체
   * @return: 없음 (void)
   *
   * 구형 인터페이스 — 이름 키로 파라미터 값을 m_kernel_params 맵에 저장.
   * 신형 인터페이스는 add_param_data(argn, args) 사용.
   */
  void add_param(const char *name, struct param_t value) {
    m_kernel_params[name] = value; // [한국어] 이름 → 파라미터 값 맵에 저장
  }
  /*
   * [한국어]
   * add_param_name_type_size() — 파라미터 선언 정보를 m_ptx_kernel_param_info에 추가
   *
   * @index:  파라미터 인덱스 (0-based)
   * @name:   파라미터 이름
   * @type:   데이터 타입 코드
   * @size:   크기(바이트)
   * @ptr:    포인터 여부
   * @space:  포인터 공간
   * @return: 없음 (void)
   *
   * PTX 파서가 .param 선언을 파싱할 때 호출. m_ptx_kernel_param_info[index]에
   * param_info 객체를 생성하여 저장.
   *
   * 호출 체인:
   *   PTX 파서 (ptx.y) → [add_param_name_type_size(...)]
   */
  void add_param_name_type_size(unsigned index, std::string name, int type,
                                size_t size, bool ptr, memory_space_t space);
  /*
   * [한국어]
   * add_param_data() — 실제 커널 파라미터 값을 param_info에 저장
   *
   * @argn: 파라미터 인덱스 (0-based)
   * @args: 호스트에서 전달된 파라미터 버퍼 포인터 (gpgpu_ptx_sim_arg)
   * @return: 없음 (void)
   *
   * cuLaunchKernel 경로에서 각 파라미터 값을 m_ptx_kernel_param_info[argn]에
   * add_data()로 저장. 이후 finalize()에서 .param 메모리에 복사.
   *
   * 호출 체인:
   *   cuda_runtime_api.cc cuLaunchKernel → [add_param_data(argn, args)]
   */
  void add_param_data(unsigned argn, struct gpgpu_ptx_sim_arg *args);
  /*
   * [한국어]
   * add_return_var() — 이 함수의 반환 변수 심볼 설정
   *
   * @rv: 반환 변수의 symbol 포인터
   * @return: 없음 (void)
   *
   * PTX .func 선언에서 반환 변수를 파싱할 때 호출.
   */
  void add_return_var(const symbol *rv) { m_return_var_sym = rv; }
  /*
   * [한국어]
   * add_arg() — 이 함수의 인수(파라미터) 심볼을 m_args 벡터에 추가
   *
   * @arg: 파라미터 심볼 포인터 (NULL 불가)
   * @return: 없음 (void)
   *
   * PTX .func/.entry 파라미터 목록 파싱 시 호출. 디바이스 함수 호출 시
   * 인수 순서를 보존하여 get_arg(n)으로 n번째 파라미터 심볼 조회 가능.
   */
  void add_arg(const symbol *arg) {
    assert(arg != NULL);        // [한국어] NULL 파라미터 심볼 추가 방지
    m_args.push_back(arg);      // [한국어] 파라미터 심볼을 순서대로 벡터에 추가
  }
  /*
   * [한국어]
   * remove_args() — m_args 벡터 초기화 (모든 파라미터 심볼 제거)
   *
   * @return: 없음 (void)
   *
   * 함수 재사용 또는 재설정 시 파라미터 목록 초기화.
   */
  void remove_args() { m_args.clear(); }
  /*
   * [한국어]
   * num_args() — 이 함수의 파라미터 개수 반환
   *
   * @return: m_args.size() — 총 파라미터 수
   *
   * 함수 호출 시 제공된 인수 수와 선언된 파라미터 수 일치 검사.
   */
  unsigned num_args() const { return m_args.size(); }
  /*
   * [한국어]
   * get_args_aligned_size() — 모든 파라미터의 정렬을 고려한 총 크기 계산 및 반환
   *
   * @return: m_args_aligned_size — 파라미터 프레임의 정렬된 총 바이트 크기
   *
   * 디바이스 함수 호출 스택 프레임 크기 계산에 사용.
   * m_args_aligned_size가 -1이면 처음 호출 시 계산하여 캐시.
   */
  unsigned get_args_aligned_size();

  /*
   * [한국어]
   * get_arg() — n번째 파라미터 심볼 반환
   *
   * @n: 파라미터 인덱스 (0-based)
   * @return: m_args[n] — n번째 파라미터의 symbol 포인터
   *
   * 디바이스 함수 호출 시 실제 인수(arg_buffer_t)를 파라미터 심볼에 매핑.
   * n >= m_args.size()이면 assert 중단.
   */
  const symbol *get_arg(unsigned n) const {
    assert(n < m_args.size()); // [한국어] 유효 인덱스 범위 검사
    return m_args[n];          // [한국어] n번째 파라미터 심볼 반환
  }
  /*
   * [한국어]
   * has_return() — 이 함수가 반환 값을 가지는지 검사
   *
   * @return: true이면 .func 함수가 반환 변수를 가짐 (m_return_var_sym != NULL)
   *
   * CALL 명령어 실행 시 반환 값 복사 여부 결정.
   */
  bool has_return() const { return m_return_var_sym != NULL; }
  /*
   * [한국어]
   * get_return_var() — 반환 변수 심볼 포인터 반환
   *
   * @return: m_return_var_sym — 반환 변수 심볼 (없으면 NULL)
   *
   * CALL 명령어 실행 완료 후 반환 값을 복사할 때 목적지 심볼로 사용.
   */
  const symbol *get_return_var() const { return m_return_var_sym; }
  /*
   * [한국어]
   * get_instruction() — PC 값으로 ptx_instruction 포인터 조회
   *
   * @PC: 조회할 명령어의 시뮬레이터 PC
   * @return: 해당 PC의 ptx_instruction 포인터; 범위 초과 시 NULL
   *
   * 커널 실행 중 현재 PC의 명령어를 가져오는 핵심 인터페이스.
   * index = PC - m_start_PC 로 m_instr_mem 배열에 인덱싱.
   *
   * 호출 체인:
   *   ptx_thread_info::ptx_exec_inst() → [get_instruction(PC)] → ptx_instruction 실행
   */
  const ptx_instruction *get_instruction(unsigned PC) const {
    unsigned index = PC - m_start_PC;          // [한국어] PC를 m_instr_mem 배열 인덱스로 변환
    if (index < m_instr_mem_size) return m_instr_mem[index]; // [한국어] 유효 범위면 해당 명령어 포인터 반환
    return NULL;                               // [한국어] 범위 초과 시 NULL 반환 (함수 종료 경계)
  }
  /*
   * [한국어]
   * get_start_PC() — 이 함수의 시작 PC 반환
   *
   * @return: m_start_PC — ptx_assemble()에서 부여된 첫 번째 명령어의 PC
   *
   * 분기 목적지 PC를 배열 인덱스로 변환하거나 함수 범위 확인에 사용.
   */
  addr_t get_start_PC() const { return m_start_PC; }

  /*
   * [한국어]
   * finalize() — 커널 파라미터 값을 .param 메모리에 복사하는 최종화 함수
   *
   * @param_mem: .param 메모리 공간 포인터
   * @return: 없음 (void)
   *
   * cuLaunchKernel 후 커널 실행 직전 호출. m_ptx_kernel_param_info의 각 param_info에서
   * get_value()로 값을 읽어 .param 메모리의 해당 오프셋에 복사.
   * 이후 각 스레드가 ld.param으로 .param 메모리에서 파라미터를 읽음.
   *
   * 호출 체인:
   *   gpgpu_sim::launch() → [finalize(param_mem)] → .param 메모리 복사
   */
  void finalize(memory_space *param_mem);
  /*
   * [한국어]
   * param_to_shared() — 공유 메모리 포인터 파라미터를 공유 메모리로 초기화
   *
   * @shared_mem: 공유 메모리 공간 포인터
   * @symtab:     함수 심볼 테이블
   * @return: 없음 (void)
   *
   * is_ptr_shared()가 true인 파라미터를 공유 메모리 공간에 할당/초기화.
   */
  void param_to_shared(memory_space *shared_mem, symbol_table *symtab);
  /*
   * [한국어]
   * list_param() — 모든 파라미터 정보를 파일 스트림에 출력 (디버그)
   *
   * @fout: 출력 대상 FILE 포인터
   * @return: 없음 (void)
   */
  void list_param(FILE *fout) const;
  /*
   * [한국어]
   * ptx_jit_config() — JIT(Just-In-Time) 컴파일 설정 — 동적 malloc 포인터 크기 반영
   *
   * @mallocPtr_Size: 런타임에 결정된 malloc 포인터 → 크기 맵
   * @param_mem:      .param 메모리 공간
   * @gpu:            gpgpu_t 시뮬레이터 포인터
   * @gridDim:        커널 그리드 차원
   * @blockDim:       커널 블록 차원
   * @return: 없음 (void)
   *
   * Accel-Sim 트레이스 기반 시뮬레이션에서 동적 포인터 크기 정보를 반영하여
   * PTX 파라미터 설정을 조정. 기능 시뮬레이션에서 포인터 범위 추론에 사용.
   */
  void ptx_jit_config(std::map<unsigned long long, size_t> mallocPtr_Size,
                      memory_space *param_mem, gpgpu_t *gpu, dim3 gridDim,
                      dim3 blockDim);

  /*
   * [한국어]
   * get_kernel_info() — 커널 실행 정보 구조체 포인터 반환 (virtual)
   *
   * @return: &m_kernel_info — gpgpu_ptx_sim_info 구조체 (레지스터/공유 메모리/스레드 수 등)
   *
   * 타이밍 모델에서 커널 리소스 사용량을 조회할 때 사용.
   * maxthreads == maxnt_id 일치를 assert로 검증.
   */
  virtual const struct gpgpu_ptx_sim_info *get_kernel_info() const {
    assert(m_kernel_info.maxthreads == maxnt_id); // [한국어] ptxas 분석 maxthreads와 .maxntid 일치 검사
    return &m_kernel_info;                         // [한국어] 커널 실행 정보 구조체 포인터 반환
  }

  /*
   * [한국어]
   * set_kernel_info() — ptxas 파싱 결과(레지스터/공유 메모리 등)를 m_kernel_info에 저장 (virtual)
   *
   * @info: gpgpu_ptx_sim_info 구조체 (ptxas -v 출력에서 파싱된 정보)
   * @return: 없음 (void)
   *
   * ___.ptxinfo 파일에서 파싱된 레지스터 수/공유 메모리 크기 등을 저장.
   * ptx_version과 sm_target을 심볼 테이블에서 읽어 m_kernel_info에 반영.
   * maxthreads는 ptxas 결과보다 maxnt_id(PTX .maxntid 지시자)를 우선 적용.
   */
  virtual const void set_kernel_info(const struct gpgpu_ptx_sim_info &info) {
    m_kernel_info = info;                                          // [한국어] ptxas 분석 정보 복사
    m_kernel_info.ptx_version = 10 * get_ptx_version().ver();     // [한국어] PTX 버전을 정수로 변환 (예: 6.0 → 60)
    m_kernel_info.sm_target = get_ptx_version().target();         // [한국어] 대상 SM 버전 설정
    // THIS DEPENDS ON ptxas being called after the PTX is parsed.
    m_kernel_info.maxthreads = maxnt_id;                           // [한국어] .maxntid 값을 최대 스레드 수로 적용
  }
  /*
   * [한국어]
   * get_symtab() — 이 함수의 심볼 테이블 포인터 반환
   *
   * @return: m_symtab — 이 함수의 레지스터/변수/레이블 심볼 테이블
   *
   * 파라미터 초기화, 레지스터 할당, 레이블 해석 등에서 접근.
   */
  symbol_table *get_symtab() { return m_symtab; }

  /*
   * [한국어]
   * local_mem_framesize() — 이 함수의 로컬 메모리 스택 프레임 크기 반환
   *
   * @return: m_local_mem_framesize — 이 함수에 할당된 .local 공간 크기(바이트)
   *
   * 커널 실행 시 각 스레드에 로컬 메모리 프레임을 할당할 때 크기 참조.
   */
  unsigned local_mem_framesize() const { return m_local_mem_framesize; }
  /*
   * [한국어]
   * set_framesize() — 로컬 메모리 프레임 크기 설정
   *
   * @sz: 로컬 메모리 프레임 크기(바이트)
   * @return: 없음 (void)
   *
   * PTX 파서 또는 ptxas 결과에서 .local 공간 크기를 결정한 후 설정.
   */
  void set_framesize(unsigned sz) { m_local_mem_framesize = sz; }
  /*
   * [한국어]
   * is_entry_point() — 이 함수가 .entry 커널인지 검사
   *
   * @return: m_entry_point — true이면 호스트에서 직접 실행하는 커널; false이면 디바이스 함수
   *
   * ptx_assemble() 등에서 커널과 디바이스 함수를 구분하는 데 사용.
   */
  bool is_entry_point() const { return m_entry_point; }
  /*
   * [한국어]
   * is_pdom_set() / set_pdom() — pdom 분석 완료 플래그 조회/설정
   *
   * is_pdom_set(): true이면 do_pdom()이 이미 완료됨 — 중복 실행 방지
   * set_pdom():    do_pdom() 완료 후 pdom_done = true로 설정
   *
   * 최초 커널 실행 시 한 번만 CFG 분석을 수행하도록 보장.
   */
  bool is_pdom_set() const { return pdom_done; }  // return pdom flag
  void set_pdom() { pdom_done = true; }           // set pdom flag

  /*
   * [한국어]
   * add_config_param() — 파라미터 설정 정보(크기, 오프셋) 추가
   *
   * @size:      파라미터 크기(바이트)
   * @alignment: 파라미터 정렬 요구사항(바이트)
   * @return: 없음 (void)
   *
   * Accel-Sim 트레이스 기반 시뮬레이션에서 파라미터 레이아웃을 재구성할 때 사용.
   * 이전 파라미터의 끝에서 정렬 요구사항에 맞춰 다음 파라미터 오프셋 계산.
   *
   * 호출 체인:
   *   Accel-Sim 트레이스 파서 → [add_config_param(size, alignment)]
   */
  void add_config_param(size_t size, unsigned alignment) {
    unsigned offset = 0;                                 // [한국어] 첫 파라미터는 오프셋 0부터 시작
    if (m_param_configs.size() > 0) {                    // [한국어] 이전 파라미터가 있으면 다음 오프셋 계산
      unsigned offset_nom =
          m_param_configs.back().first + m_param_configs.back().second; // [한국어] 이전 파라미터 끝 = 이전 크기 + 이전 오프셋
      // ensure offset matches alignment requirements
      offset = offset_nom % alignment ? (offset_nom / alignment + 1) * alignment
                                      : offset_nom; // [한국어] 정렬 요구에 맞게 오프셋 올림(roundup)
    }
    m_param_configs.push_back(std::pair<size_t, unsigned>(size, offset)); // [한국어] (크기, 오프셋) 쌍을 벡터에 추가
  }

  /*
   * [한국어]
   * get_param_config() — n번째 파라미터의 (크기, 오프셋) 쌍 반환
   *
   * @param_num: 파라미터 인덱스 (0-based)
   * @return: pair<size_t, unsigned> — (파라미터 크기, .param 메모리 내 오프셋)
   *
   * Accel-Sim에서 파라미터 배치 정보를 조회할 때 사용.
   */
  std::pair<size_t, unsigned> get_param_config(unsigned param_num) const {
    return m_param_configs[param_num]; // [한국어] 인덱스로 파라미터 설정 쌍 반환
  }

  /*
   * [한국어]
   * set_maxnt_id() / get_maxnt_id() — PTX .maxntid 지시자의 최대 스레드 수 설정/조회
   *
   * set_maxnt_id(@maxthreads): PTX .maxntid 파싱 후 최대 스레드 수 설정
   * get_maxnt_id():            저장된 최대 스레드 수 반환
   *
   * set_kernel_info()에서 m_kernel_info.maxthreads에 반영.
   * 타이밍 모델이 블록당 최대 스레드 수를 알기 위해 사용.
   */
  void set_maxnt_id(unsigned maxthreads) { maxnt_id = maxthreads; }
  unsigned get_maxnt_id() { return maxnt_id; }
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  /* [한국어] gpgpu_context 역방향 포인터.
   * 설정자: 생성자 @ctx 파라미터로 설정.
   * 읽는 자: 전역 상태 접근 필요 시.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

 protected:
  // Registers/shmem/etc. used (from ptxas -v), loaded from ___.ptxinfo along
  // with ___.ptx
  struct gpgpu_ptx_sim_info m_kernel_info;
  /* [한국어] ptxas -v 출력에서 파싱된 커널 리소스 사용량 정보.
   * 설정자: set_kernel_info() — ___.ptxinfo 파일 파싱 후 저장.
   * 읽는 자: get_kernel_info() — 타이밍 모델에서 레지스터/공유 메모리 수 조회.
   * 주요 필드: num_regs(레지스터 수), smem(공유 메모리 크기), maxthreads(최대 스레드).
   * 동기화: set_kernel_info() 완료 후 읽기 전용이므로 별도 락 불필요. */

 private:
  unsigned maxnt_id;
  /* [한국어] PTX .maxntid 지시자에서 파싱된 최대 스레드 수.
   * 설정자: set_maxnt_id() — PTX 파서가 .maxntid 파싱 후 설정.
   * 읽는 자: get_maxnt_id(), set_kernel_info() — m_kernel_info.maxthreads에 반영.
   * 값 범위: 1 ~ SM의 최대 스레드 수 (예: 1024).
   * 동기화: 파싱 완료 후 불변이므로 별도 락 불필요. */

  unsigned m_uid;
  /* [한국어] 이 function_info 객체의 전역 고유 ID.
   * 설정자: 생성자에서 전역 카운터로 할당.
   * 읽는 자: 디버그/추적에서 함수 식별.
   * 동기화: 파싱 단일 스레드에서 생성하므로 별도 락 불필요. */

  unsigned m_local_mem_framesize;
  /* [한국어] 이 함수의 .local 메모리 스택 프레임 크기(바이트).
   * 설정자: set_framesize() — PTX 파서 또는 ptxas 결과에서 설정.
   * 읽는 자: local_mem_framesize() — 커널 실행 시 스레드별 로컬 메모리 할당.
   * 값 범위: 0(로컬 메모리 없음) ~ PTX .local 선언 총 크기.
   * 동기화: 파싱 완료 후 불변이므로 별도 락 불필요. */

  bool m_entry_point;
  /* [한국어] 이 함수가 .entry 커널인지 여부.
   * 설정자: 생성자의 @entry_point 파라미터로 설정.
   * 읽는 자: is_entry_point() — 커널/함수 구분.
   * 값 범위: true(.entry 커널) / false(.func 디바이스 함수).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  bool m_extern;
  /* [한국어] 이 함수가 .extern 외부 선언인지 여부.
   * 설정자: 생성자 또는 PTX 파서 설정.
   * 읽는 자: is_extern() — 본체 없는 외부 함수는 ptx_assemble() 제외.
   * 값 범위: true(외부 선언) / false(본체 있음).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  bool m_assembled;
  /* [한국어] ptx_assemble()이 완료됐는지 여부.
   * 설정자: ptx_assemble() 완료 시 true로 설정.
   * 읽는 자: ptx_assemble()의 중복 실행 방지 조건 검사.
   * 값 범위: true(어셈블 완료) / false(미완료).
   * 동기화: 단일 스레드에서 실행되므로 별도 락 불필요. */

  bool pdom_done;  // flag to check whether pdom is completed or not
  /* [한국어] do_pdom() CFG 분석이 완료됐는지 여부.
   * 설정자: set_pdom() — do_pdom() 완료 후 true로 설정.
   * 읽는 자: is_pdom_set() — 커널 최초 실행 시 중복 분석 방지.
   * 값 범위: true(완료) / false(미완료).
   * 동기화: 단일 스레드에서 실행되므로 별도 락 불필요. */

  std::string m_name;
  /* [한국어] PTX 함수/커널 이름 문자열 (예: "_Z9vectorAddPfS_S_i").
   * 설정자: set_name() — PTX 파서가 .entry/.func 이름 파싱 후 설정.
   * 읽는 자: get_name() — 함수 조회 테이블(g_sym_name_to_symbol_table)의 키로 사용.
   * 동기화: 파싱 완료 후 불변이므로 별도 락 불필요. */

  ptx_instruction **m_instr_mem;
  /* [한국어] PC 오프셋으로 인덱싱되는 PTX 명령어 포인터 배열.
   * 설정자: ptx_assemble() — m_instructions 리스트를 배열로 변환하며 heap 할당.
   *          set_m_instr_mem_index()로 각 명령어에 인덱스 기록.
   * 읽는 자: get_instruction(PC) — m_instr_mem[PC - m_start_PC] 로 조회.
   * 값 범위: 크기 m_instr_mem_size 배열; 레이블 위치는 NULL 또는 레이블 명령어.
   * 동기화: ptx_assemble() 완료 후 읽기 전용이므로 별도 락 불필요. */

  unsigned m_start_PC;
  /* [한국어] 이 함수의 첫 번째 명령어에 부여된 시뮬레이터 PC.
   * 설정자: ptx_assemble() — 전역 PC 카운터에서 할당.
   * 읽는 자: get_instruction(PC) (인덱스 계산), get_start_PC(), get_reconvergence_pairs().
   * 값 범위: 함수별로 유일한 PC 값.
   * 동기화: ptx_assemble() 완료 후 불변이므로 별도 락 불필요. */

  unsigned m_instr_mem_size;
  /* [한국어] m_instr_mem 배열의 크기 (마지막 명령어 오프셋 + 1).
   * 설정자: ptx_assemble() — 모든 명령어를 배열에 배치한 후 설정.
   * 읽는 자: get_instruction(PC) — 범위 초과 여부 검사.
   * 값 범위: 1 ~ 이 함수의 총 명령어 수.
   * 동기화: ptx_assemble() 완료 후 불변이므로 별도 락 불필요. */

  std::map<std::string, param_t> m_kernel_params;
  /* [한국어] 이름→파라미터 값 맵 (구형 add_param() 인터페이스용).
   * 설정자: add_param() — 이름 키로 파라미터 값 저장.
   * 읽는 자: 구형 파라미터 접근 코드.
   * 동기화: 커널 실행 준비 단계(단일 스레드)에서만 쓰므로 별도 락 불필요. */

  std::map<unsigned, param_info> m_ptx_kernel_param_info;
  /* [한국어] 인덱스→param_info 맵 (신형 인터페이스 — 타입/크기/값 통합 관리).
   * 설정자: add_param_name_type_size() — 파싱 시 param_info 생성;
   *          add_param_data() — cuLaunchKernel 시 실제 값 채움.
   * 읽는 자: finalize() — .param 메모리 복사 시 param_info 조회.
   * 동기화: 파싱~실행 준비 단계(단일 스레드)에서만 쓰므로 별도 락 불필요. */

  std::vector<std::pair<size_t, unsigned> > m_param_configs;
  /* [한국어] Accel-Sim 트레이스 기반 파라미터 레이아웃 설정 벡터 [(크기, 오프셋)].
   * 설정자: add_config_param() — 트레이스 파서에서 파라미터별 크기/오프셋 추가.
   * 읽는 자: get_param_config() — 파라미터 오프셋 조회.
   * 동기화: 설정 단계(단일 스레드)에서만 쓰므로 별도 락 불필요. */

  const symbol *m_return_var_sym;
  /* [한국어] .func 함수의 반환 변수 심볼 포인터.
   * 설정자: add_return_var() — PTX 파서가 반환 변수 파싱 후 설정.
   * 읽는 자: has_return(), get_return_var() — CALL 실행 시 반환 값 복사 여부/대상.
   * 값 범위: 유효한 symbol 포인터 또는 NULL(반환 없음).
   * 동기화: 파싱 완료 후 불변이므로 별도 락 불필요. */

  std::vector<const symbol *> m_args;
  /* [한국어] 이 함수의 파라미터 심볼들을 선언 순서대로 저장하는 벡터.
   * 설정자: add_arg() — PTX 파서가 파라미터 심볼 파싱 순서대로 추가.
   * 읽는 자: get_arg(n), num_args() — 함수 호출 시 인수-파라미터 매핑.
   * 동기화: 파싱 완료 후 불변이므로 별도 락 불필요. */

  std::list<ptx_instruction *> m_instructions;
  /* [한국어] 파서가 생성한 PTX 명령어 포인터 리스트 (add_inst()로 설정).
   * 설정자: add_inst() — 파서 완료 후 한 번에 리스트로 전달.
   * 읽는 자: ptx_assemble() — 배열로 변환; create_basic_blocks() — CFG 생성.
   * 동기화: ptx_assemble() 이후 m_instr_mem 배열이 주 참조가 되므로 별도 락 불필요. */

  std::vector<basic_block_t *> m_basic_blocks;
  /* [한국어] CFG의 기본 블록 포인터 벡터 (인덱스 = bb_id).
   * 설정자: create_basic_blocks() — basic_block_t를 heap에 생성하고 순서대로 저장.
   * 읽는 자: connect_basic_blocks(), find_dominators(), find_postdominators(),
   *           find_ipostdominators(), get_reconvergence_pairs().
   * 동기화: do_pdom() 완료 후 읽기 전용이므로 별도 락 불필요. */

  std::list<std::pair<unsigned, unsigned> > m_back_edges;
  /* [한국어] CFG의 뒤로 가는 에지(back edge) 목록 [(소스 블록 ID, 목적지 블록 ID)].
   * 설정자: connect_basic_blocks() — 루프를 감지할 때 뒤로 가는 에지 기록.
   * 읽는 자: 지배자 분석 — 루프 구조 파악.
   * 동기화: do_pdom() 완료 후 읽기 전용이므로 별도 락 불필요. */

  std::map<std::string, unsigned> labels;
  /* [한국어] 레이블 이름 → m_instr_mem 배열 인덱스 맵.
   * 설정자: ptx_assemble() — 레이블 명령어를 만날 때마다 이름과 인덱스를 등록.
   * 읽는 자: connect_basic_blocks() — 분기 목적지 레이블 이름으로 블록 인덱스 조회.
   * 동기화: ptx_assemble() 완료 후 읽기 전용이므로 별도 락 불필요. */

  unsigned num_reconvergence_pairs;
  /* [한국어] SIMT 재합류 쌍의 총 개수.
   * 설정자: get_num_reconvergence_pairs() — do_pdom() 후 계산하여 저장.
   * 읽는 자: get_reconvergence_pairs() — 재합류 쌍 배열 크기 결정.
   * 값 범위: 0 ~ 이 함수의 분기 명령어 수.
   * 동기화: do_pdom() 완료 후 읽기 전용이므로 별도 락 불필요. */

  // Registers/shmem/etc. used (from ptxas -v), loaded from ___.ptxinfo along
  // with ___.ptx
  // with ___.ptx

  symbol_table *m_symtab;
  /* [한국어] 이 함수의 심볼 테이블 포인터.
   * 설정자: set_symtab() — PTX 파서가 함수 스코프 심볼 테이블 생성 후 연결.
   * 읽는 자: get_symtab(), get_ptx_version(), get_sm_target() 등.
   *           레지스터/변수/레이블 심볼 접근의 진입점.
   * 동기화: 파싱 완료 후 불변이므로 별도 락 불필요. */

  // parameter size for device kernels
  int m_args_aligned_size;
  /* [한국어] 디바이스 함수 파라미터의 정렬을 고려한 총 크기(바이트).
   * 설정자: get_args_aligned_size() — 최초 호출 시 계산하여 캐시 (-1이면 미계산).
   * 읽는 자: get_args_aligned_size() — 디바이스 함수 호출 스택 프레임 크기.
   * 값 범위: -1(미계산) 또는 0 이상의 정렬된 크기.
   * 동기화: 단일 스레드에서만 계산하므로 별도 락 불필요. */

  addr_t m_n;  // offset in m_instr_mem (used in do_pdom)
  /* [한국어] ptx_assemble() 과정에서 사용하는 m_instr_mem 배열의 현재 오프셋 카운터.
   * 설정자: ptx_assemble() — 명령어를 배열에 배치할 때마다 증가.
   * 읽는 자: ptx_assemble() 내부 — 다음 명령어를 배치할 배열 위치 결정.
   * 값 범위: 0 ~ m_instr_mem_size-1.
   * 동기화: ptx_assemble() 단일 호출 내에서만 사용하므로 별도 락 불필요. */
};

/*
 * [한국어]
 *
 * === 파일의 역할 (arg_buffer_t) ===
 * 디바이스 함수 호출(CALL 명령어) 시 하나의 실제 인수(actual argument)를 임시로
 * 저장하는 버퍼 클래스. 소스 피연산자 값을 캡처하여 copy_buffer_list_into_frame()이
 * 호출될 때까지 보관한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 호출 체인:
 *   instructions.cc CALL 명령어 실행
 *   → copy_args_into_buffer_list() — 실제 인수들을 arg_buffer_t로 캡처하여 리스트에 저장
 *   → copy_buffer_list_into_frame() — 피호출 함수의 스택 프레임에 인수 값 복사
 *   → 피호출 함수 실행
 *
 * === 타 모듈과의 연결 ===
 * - ptx_thread_info: 레지스터 파일에서 소스 값 읽기, 프레임에 값 쓰기
 * - function_info: 파라미터 심볼(m_args)과 매핑하여 목적지 결정
 * - arg_buffer_list_t: std::list<arg_buffer_t> typedef로 인수 목록 관리
 *
 * === 주요 함수/구조체 요약 ===
 * - arg_buffer_t(dst, src_op, reg_value): 레지스터 소스 값을 캡처하는 생성자
 * - arg_buffer_t(dst, src_op, param_array, size): .param 메모리 소스 값을 캡처
 * - make_copy(): 깊은 복사 (m_param_value는 새 heap 블록 할당)
 * - get_reg()/get_param_buffer(): 캡처된 값 반환
 * - get_dst(): 이 인수의 목적지 파라미터 심볼 반환
 */
class arg_buffer_t {
 public:
  /*
   * [한국어]
   * arg_buffer_t(ctx) — 기본(빈) 인수 버퍼 생성자
   *
   * @ctx: gpgpu_context 포인터 (m_src_op 초기화에 전달)
   * @return: 없음 (생성자)
   *
   * 비어 있는 arg_buffer_t를 만들 때 사용. m_src_op는 ctx로 초기화.
   * m_is_reg/m_is_param = false 이므로 아직 유효한 인수 없음.
   */
  arg_buffer_t(gpgpu_context *ctx) : m_src_op(ctx) { // [한국어] m_src_op를 빈 operand_info(ctx)로 초기화
    m_is_reg = false;          // [한국어] 레지스터 소스 없음
    m_is_param = false;        // [한국어] .param 메모리 소스 없음
    m_param_value = NULL;      // [한국어] 파라미터 버퍼 포인터 NULL 초기화
    m_reg_value = ptx_reg_t(); // [한국어] 레지스터 값 0으로 초기화 (ptx_reg_t 기본 생성)
  }
  /*
   * [한국어]
   * arg_buffer_t(another, ctx) — 복사 생성자
   *
   * @another: 복사 원본 arg_buffer_t
   * @ctx:     gpgpu_context 포인터 (m_src_op 초기화에 전달)
   * @return: 없음 (생성자)
   *
   * make_copy()를 호출하여 깊은 복사 수행.
   * m_is_param이 true이면 m_param_value 버퍼를 새로 heap 할당하여 복사.
   */
  arg_buffer_t(const arg_buffer_t &another, gpgpu_context *ctx)
      : m_src_op(ctx) { // [한국어] m_src_op는 ctx로 임시 초기화 (make_copy에서 덮어씀)
    make_copy(another); // [한국어] 깊은 복사 수행
  }
  /*
   * [한국어]
   * make_copy() — 다른 arg_buffer_t의 깊은 복사 수행
   *
   * @another: 복사 원본 arg_buffer_t
   * @return: 없음 (void)
   *
   * 모든 필드를 복사하되, m_is_param이 true이면 m_param_value를 새 heap 블록에
   * 독립적으로 복사한다. 이렇게 해야 소유권이 분리되어 소멸자에서 이중 해제 방지.
   *
   * 호출 체인:
   *   복사 생성자 / operator=() → [make_copy(another)]
   */
  void make_copy(const arg_buffer_t &another) {
    m_dst = another.m_dst;             // [한국어] 목적지 심볼 포인터 복사
    m_src_op = another.m_src_op;       // [한국어] 소스 피연산자 복사
    m_is_reg = another.m_is_reg;       // [한국어] 레지스터 소스 플래그 복사
    m_is_param = another.m_is_param;   // [한국어] .param 소스 플래그 복사
    m_reg_value = another.m_reg_value; // [한국어] 레지스터 값 복사
    m_param_bytes = another.m_param_bytes; // [한국어] 파라미터 버퍼 크기 복사
    if (m_is_param) {                  // [한국어] .param 소스이면 버퍼도 깊은 복사 필요
      m_param_value = malloc(m_param_bytes); // [한국어] 새 버퍼 heap 할당
      memcpy(m_param_value, another.m_param_value, m_param_bytes); // [한국어] 원본 버퍼 내용 복사
    }
  }
  /*
   * [한국어]
   * operator=() — 대입 연산자
   *
   * @another: 대입 원본
   * @return: 없음 (void)
   *
   * make_copy()를 호출하여 깊은 복사 수행.
   * 기존 m_param_value 해제 없이 덮어씀에 주의 (이 클래스는 단순 사용 패턴).
   */
  void operator=(const arg_buffer_t &another) { make_copy(another); }
  /*
   * [한국어]
   * ~arg_buffer_t() — 소멸자
   *
   * m_is_param이 true이면 heap에 할당된 m_param_value 버퍼를 해제.
   * m_is_reg이면 m_reg_value는 스택 자동 해제이므로 추가 해제 불필요.
   */
  ~arg_buffer_t() {
    if (m_is_param) free(m_param_value); // [한국어] .param 버퍼만 명시적 해제 (레지스터 값은 자동 해제)
  }
  /*
   * [한국어]
   * arg_buffer_t(dst_sym, src_op, source_value) — 레지스터 소스 값 캡처 생성자
   *
   * @dst_sym:      목적지 파라미터 심볼 (피호출 함수의 파라미터)
   * @src_op:       소스 피연산자 (호출자 측의 레지스터 또는 .param)
   * @source_value: 캡처할 소스 레지스터의 ptx_reg_t 값
   * @return: 없음 (생성자)
   *
   * dst_sym이 레지스터이면 m_is_reg = true, m_reg_value = source_value.
   * dst_sym이 .param이면 m_is_param = true, source_value를 sizeof(ptx_reg_t) 크기
   * heap 버퍼에 복사하여 보관.
   *
   * 호출 체인:
   *   copy_arg_to_buffer() → [arg_buffer_t(dst, src_op, reg_value)]
   */
  arg_buffer_t(const symbol *dst_sym, const operand_info &src_op,
               ptx_reg_t source_value)
      : m_src_op(src_op) { // [한국어] 소스 피연산자 저장
    m_dst = dst_sym;               // [한국어] 목적지 파라미터 심볼 저장
    m_reg_value = ptx_reg_t();     // [한국어] 레지스터 값 기본 초기화 (덮어쓰기 전)
    if (dst_sym->is_reg()) {       // [한국어] 목적지가 레지스터이면
      m_is_reg = true;             // [한국어] 레지스터 소스 경로 선택
      m_is_param = false;          // [한국어] .param 경로 아님
      assert(src_op.is_reg());     // [한국어] 소스도 레지스터여야 함 — 타입 불일치 방지
      m_reg_value = source_value;  // [한국어] 소스 레지스터 값 직접 복사
    } else {                       // [한국어] 목적지가 .param 변수이면
      m_is_param = true;           // [한국어] .param 소스 경로 선택
      m_is_reg = false;            // [한국어] 레지스터 경로 아님
      m_param_value = calloc(sizeof(ptx_reg_t), 1); // [한국어] ptx_reg_t 크기 heap 버퍼 할당 및 0 초기화
      // new (m_param_value) ptx_reg_t(source_value);
      memcpy(m_param_value, &source_value, sizeof(ptx_reg_t)); // [한국어] 소스 값을 버퍼에 복사
      m_param_bytes = sizeof(ptx_reg_t);            // [한국어] 버퍼 크기 기록
    }
  }
  /*
   * [한국어]
   * arg_buffer_t(dst_sym, src_op, source_param_value_array, array_size)
   * — .param 메모리 소스 값 캡처 생성자
   *
   * @dst_sym:                 목적지 파라미터 심볼
   * @src_op:                  소스 피연산자 (.param 또는 레지스터)
   * @source_param_value_array: 소스 .param 메모리의 원시 바이트 배열 포인터
   * @array_size:              배열 크기(바이트)
   * @return: 없음 (생성자)
   *
   * dst_sym이 레지스터이면 array_size(1/2/4/8 바이트)에 따라 적절한
   * ptx_reg_t 유니온 필드에 값을 로드. dst_sym이 .param이면 array_size 크기
   * heap 버퍼를 할당하고 memcpy로 복사하여 보관.
   *
   * 호출 체인:
   *   copy_arg_to_buffer() → [arg_buffer_t(dst, src_op, param_array, size)]
   */
  arg_buffer_t(const symbol *dst_sym, const operand_info &src_op,
               void *source_param_value_array, unsigned array_size)
      : m_src_op(src_op) { // [한국어] 소스 피연산자 저장
    m_dst = dst_sym;               // [한국어] 목적지 파라미터 심볼 저장
    if (dst_sym->is_reg()) {       // [한국어] 목적지가 레지스터이면
      m_is_reg = true;             // [한국어] 레지스터 목적지 경로
      m_is_param = false;          // [한국어] .param 목적지 아님
      assert(src_op.is_param_local()); // [한국어] 소스가 로컬 .param이어야 함
      assert(dst_sym->get_size_in_bytes() == array_size); // [한국어] 소스 크기와 목적지 레지스터 크기 일치 검사
      switch (array_size) {        // [한국어] 크기에 따라 적절한 레지스터 유니온 필드에 로드
        case 1:
          m_reg_value.u8 = *(unsigned char *)source_param_value_array;   // [한국어] 1바이트 → u8 필드
          break;
        case 2:
          m_reg_value.u16 = *(unsigned short *)source_param_value_array; // [한국어] 2바이트 → u16 필드
          break;
        case 4:
          m_reg_value.u32 = *(unsigned int *)source_param_value_array;   // [한국어] 4바이트 → u32 필드
          break;
        case 8:
          m_reg_value.u64 = *(unsigned long long *)source_param_value_array; // [한국어] 8바이트 → u64 필드
          break;
        default:
          printf(
              "GPGPU-Sim PTX: ERROR ** source param size does not match known "
              "register sizes\n"); // [한국어] 알 수 없는 크기 — 에러 메시지 출력
          break;
      }
    } else {
      // param
      m_is_param = true;                    // [한국어] .param 목적지 경로
      m_is_reg = false;                     // [한국어] 레지스터 목적지 아님
      m_param_value = calloc(array_size, 1); // [한국어] array_size 바이트 heap 버퍼 할당 및 0 초기화
      m_param_bytes = array_size;            // [한국어] 버퍼 크기 기록
      memcpy(m_param_value, source_param_value_array, array_size); // [한국어] 소스 .param 값 복사
    }
  }

  /*
   * [한국어]
   * is_reg() — 이 버퍼가 레지스터 값을 담고 있는지 검사
   *
   * @return: true이면 m_reg_value에 유효한 레지스터 값이 있음
   *
   * copy_buffer_to_frame()에서 값을 어떻게 목적지에 복사할지 결정.
   */
  bool is_reg() const { return m_is_reg; }
  /*
   * [한국어]
   * get_reg() — 캡처된 레지스터 값 반환
   *
   * @return: m_reg_value — 캡처 시점의 소스 레지스터 값 (ptx_reg_t 복사)
   *
   * is_reg()가 true일 때만 유효. copy_buffer_to_frame()에서 목적지 레지스터에 쓸 때 사용.
   * m_is_reg가 false이면 assert 중단.
   */
  ptx_reg_t get_reg() const {
    assert(m_is_reg); // [한국어] 레지스터 값이 없으면 중단
    return m_reg_value; // [한국어] 캡처된 레지스터 값 반환
  }

  /*
   * [한국어]
   * get_param_buffer() — 캡처된 .param 메모리 버퍼 포인터 반환
   *
   * @return: m_param_value — heap에 할당된 파라미터 값 버퍼의 const void 포인터
   *
   * m_is_param이 true일 때만 유효. copy_buffer_to_frame()에서 목적지 .param에 쓸 때 사용.
   * m_is_param이 false이면 assert 중단.
   */
  const void *get_param_buffer() const {
    assert(m_is_param); // [한국어] .param 값이 없으면 중단
    return m_param_value; // [한국어] 캡처된 .param 버퍼 포인터 반환
  }
  /*
   * [한국어]
   * get_param_buffer_size() — 캡처된 .param 버퍼의 바이트 크기 반환
   *
   * @return: m_param_bytes — m_param_value 버퍼의 크기(바이트)
   *
   * copy_buffer_to_frame()에서 memcpy 크기로 사용.
   * m_is_param이 false이면 assert 중단.
   */
  size_t get_param_buffer_size() const {
    assert(m_is_param); // [한국어] .param 값이 없으면 중단
    return m_param_bytes; // [한국어] 버퍼 크기 반환
  }

  /*
   * [한국어]
   * get_dst() — 이 인수의 목적지 파라미터 심볼 반환
   *
   * @return: m_dst — 피호출 함수의 파라미터 심볼 포인터
   *
   * copy_buffer_to_frame()에서 값을 어느 파라미터 변수에 복사할지 결정.
   */
  const symbol *get_dst() const { return m_dst; }

 private:
  // destination of copy
  const symbol *m_dst;
  /* [한국어] 이 인수가 복사될 피호출 함수의 파라미터 심볼.
   * 설정자: 각 생성자의 @dst_sym 파라미터로 설정.
   * 읽는 자: get_dst() — copy_buffer_to_frame()에서 목적지 파라미터 결정.
   * 값 범위: 유효한 symbol 포인터 (NULL 불가).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  // source operand
  operand_info m_src_op;
  /* [한국어] 이 인수의 소스 피연산자 (호출자 측의 레지스터 또는 .param 피연산자).
   * 설정자: 각 생성자에서 @src_op으로 초기화.
   * 읽는 자: copy_buffer_to_frame() — 디버그/타입 확인용 소스 정보 접근.
   * 값 범위: 유효한 operand_info 객체 (레지스터 또는 .param 타입).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  // source information
  bool m_is_reg;
  /* [한국어] 소스가 레지스터 값이면 true — m_reg_value에 캡처된 값 보관.
   * 설정자: 각 생성자에서 목적지 타입에 따라 설정.
   * 읽는 자: is_reg(), get_reg() — copy_buffer_to_frame()에서 경로 분기.
   * 값 범위: true(레지스터) / false(.param 또는 빈 버퍼).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  bool m_is_param;
  /* [한국어] 소스가 .param 메모리 값이면 true — m_param_value에 heap 버퍼 보관.
   * 설정자: 각 생성자에서 목적지 타입에 따라 설정.
   * 읽는 자: get_param_buffer(), get_param_buffer_size() — 소멸자의 free() 조건.
   * 값 범위: true(.param 버퍼) / false(레지스터 또는 빈 버퍼).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  // source is register
  ptx_reg_t m_reg_value;
  /* [한국어] 레지스터 소스 경우(m_is_reg == true) 캡처된 ptx_reg_t 값.
   * 설정자: 레지스터 소스 생성자에서 source_value로 설정.
   * 읽는 자: get_reg() — copy_buffer_to_frame()에서 피호출 함수 레지스터에 쓸 때.
   * 값 범위: 유효한 ptx_reg_t 유니온 (타입에 따라 u8/u16/u32/u64/f32/f64 중 하나 유효).
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */

  // source is param
  void *m_param_value;
  /* [한국어] .param 소스 경우(m_is_param == true) heap 할당된 파라미터 값 버퍼.
   * 설정자: .param 소스 생성자에서 calloc/malloc 후 memcpy로 채움;
   *          make_copy()에서 새 버퍼 할당 후 복사.
   * 읽는 자: get_param_buffer() — copy_buffer_to_frame()에서 .param 메모리에 쓸 때.
   *           소멸자 — m_is_param이면 free() 호출.
   * 값 범위: array_size 또는 sizeof(ptx_reg_t) 바이트의 heap 버퍼.
   * 동기화: 단일 소유(unique ownership) — 복사 시 make_copy()가 새 버퍼 할당. */

  unsigned m_param_bytes;
  /* [한국어] m_param_value 버퍼의 바이트 크기.
   * 설정자: .param 소스 생성자에서 sizeof(ptx_reg_t) 또는 array_size로 설정.
   * 읽는 자: get_param_buffer_size() — memcpy 크기; make_copy() — 복사 크기.
   * 값 범위: 1~sizeof(ptx_reg_t) 또는 파라미터 배열 크기.
   * 동기화: 생성 후 불변이므로 별도 락 불필요. */
};

typedef std::list<arg_buffer_t> arg_buffer_list_t;
/* [한국어] arg_buffer_list_t — CALL 명령어 실행 시 실제 인수들을 담는 리스트 타입.
 * copy_args_into_buffer_list()가 채우고, copy_buffer_list_into_frame()이 소비.
 * 각 원소는 하나의 실제 인수(actual argument)를 캡처한 arg_buffer_t 객체. */

/*
 * [한국어]
 * copy_arg_to_buffer() — 단일 실제 인수를 arg_buffer_t로 캡처
 *
 * @thread:           현재 실행 중인 ptx_thread_info (소스 레지스터 값 읽기용)
 * @actual_param_op:  호출자 측 실제 인수 피연산자 (레지스터 또는 .param)
 * @formal_param:     피호출 함수의 대응 형식 파라미터 심볼
 * @return: 인수 값이 캡처된 arg_buffer_t 객체
 *
 * CALL 명령어 실행 시 copy_args_into_buffer_list()가 각 인수에 대해 호출.
 * 소스 피연산자 타입에 따라 레지스터 값(ptx_reg_t) 또는 .param 메모리 값을
 * arg_buffer_t에 저장하여 반환.
 *
 * 호출 체인:
 *   instructions.cc CALL 실행 → copy_args_into_buffer_list() → [copy_arg_to_buffer()]
 */
arg_buffer_t copy_arg_to_buffer(ptx_thread_info *thread,
                                operand_info actual_param_op,
                                const symbol *formal_param);
/*
 * [한국어]
 * copy_args_into_buffer_list() — CALL 명령어의 모든 실제 인수를 리스트로 캡처
 *
 * @pI:           CALL ptx_instruction 포인터
 * @thread:       현재 실행 중인 ptx_thread_info
 * @target_func:  피호출 function_info (형식 파라미터 목록 조회용)
 * @arg_values:   결과를 담을 arg_buffer_list_t 참조 (출력)
 * @return: 없음 (void)
 *
 * CALL 명령어의 피연산자와 target_func의 m_args를 순서대로 매핑하여
 * 각 인수를 copy_arg_to_buffer()로 캡처하고 arg_values에 추가.
 * 이후 copy_buffer_list_into_frame()이 피호출 함수의 스택 프레임에 복사.
 *
 * 호출 체인:
 *   instructions.cc CALL 실행 → [copy_args_into_buffer_list()] → copy_buffer_list_into_frame()
 */
void copy_args_into_buffer_list(const ptx_instruction *pI,
                                ptx_thread_info *thread,
                                const function_info *target_func,
                                arg_buffer_list_t &arg_values);
/*
 * [한국어]
 * copy_buffer_list_into_frame() — 캡처된 인수 목록을 피호출 함수의 스택 프레임에 복사
 *
 * @thread:     현재 실행 중인 ptx_thread_info (피호출 함수로 컨텍스트 전환됨)
 * @arg_values: copy_args_into_buffer_list()로 채워진 arg_buffer_list_t
 * @return: 없음 (void)
 *
 * 각 arg_buffer_t를 순회하며 copy_buffer_to_frame()을 호출하여
 * 레지스터 값 또는 .param 메모리 값을 피호출 함수의 레지스터/param 공간에 씀.
 *
 * 호출 체인:
 *   instructions.cc CALL 실행 → [copy_buffer_list_into_frame()] → copy_buffer_to_frame()
 */
void copy_buffer_list_into_frame(ptx_thread_info *thread,
                                 arg_buffer_list_t &arg_values);
/*
 * [한국어]
 * copy_buffer_to_frame() — 단일 arg_buffer_t를 피호출 함수의 스택 프레임에 복사
 *
 * @thread: 현재 실행 중인 ptx_thread_info (피호출 함수 컨텍스트)
 * @a:      복사할 단일 arg_buffer_t (레지스터 값 또는 .param 버퍼)
 * @return: 없음 (void)
 *
 * a.is_reg()이면 목적지 레지스터에 get_reg() 값을 씀.
 * a.is_param이면 목적지 .param 메모리에 get_param_buffer() 내용을 memcpy.
 *
 * 호출 체인:
 *   copy_buffer_list_into_frame() → [copy_buffer_to_frame(thread, a)]
 */
void copy_buffer_to_frame(ptx_thread_info *thread, const arg_buffer_t &a);

/*
 * [한국어]
 *
 * === 파일의 역할 (textureInfo) ===
 * CUDA 텍스처 메모리의 텍셀(texel) 레이아웃 정보를 담는 구조체.
 * GPU 텍스처 캐시 시뮬레이션에서 64B 캐시 블록 내 텍셀 배치와
 * 주소 계산에 필요한 타일링 인수 및 비트 수를 제공한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * cudaBindTexture()/cudaBindTexture2D() 경로에서 텍스처 참조 설정 시 생성.
 * gpu-cache.cc 또는 cuda-sim/instructions.cc의 TEX 명령어 실행에서 참조.
 *
 * === 타 모듈과의 연결 ===
 * - 텍스처 캐시(L1 texture cache): 텍셀 크기와 타일링 인수로 주소 계산
 * - cudaBindTexture 경로: 텍스처 채널 디스크립터로부터 texel_size 계산
 *
 * === 주요 함수/구조체 요약 ===
 * - texel_size: 텍셀 하나의 바이트 크기 (채널 디스크립터 총 비트 / 8)
 * - Tx, Ty: x/y 차원의 타일링 인수 (64B 블록 내 텍셀 수의 제곱근)
 * - Tx_numbits, Ty_numbits: log2(Tx), log2(Ty) — 비트 시프트 주소 계산용
 * - texel_size_numbits: log2(texel_size) — 텍셀 주소 → 바이트 주소 변환용
 */
struct textureInfo {
  unsigned int texel_size;  // size in bytes, e.g. (channelDesc.x+y+z+w)/8
  /* [한국어] 텍셀 하나의 바이트 크기.
   * 설정자: cudaBindTexture() 경로에서 채널 디스크립터 (x+y+z+w 비트 / 8)로 계산.
   * 읽는 자: TEX 명령어 실행 시 텍스처 주소 → 바이트 주소 변환.
   * 값 범위: 1/2/4/8/16 바이트 (CUDA 채널 포맷에 따라). */

  unsigned int Tx,
      Ty;  // tiling factor dimensions of layout of texels per 64B cache block
  /* [한국어] x(Tx)/y(Ty) 차원의 타일링 인수 — 64B 캐시 블록 내 각 차원의 텍셀 수.
   * 설정자: cudaBindTexture() 경로에서 텍셀 크기와 블록 크기로부터 계산.
   * 읽는 자: TEX 명령어 실행 시 2D 텍스처 주소의 차원별 타일 오프셋 계산.
   * 값 범위: 2의 거듭제곱 (1, 2, 4, 8 등 — 64B = Tx*Ty*texel_size 조건). */

  unsigned int Tx_numbits, Ty_numbits;  // log2(T)
  /* [한국어] log2(Tx), log2(Ty) — 타일링 인수의 비트 수.
   * 설정자: cudaBindTexture() 경로에서 Tx/Ty 계산 후 bitcount로 결정.
   * 읽는 자: TEX 명령어 주소 계산에서 비트 시프트 연산으로 곱셈 대체.
   * 값 범위: 0(Tx/Ty=1), 1(=2), 2(=4), 3(=8) 등. */

  unsigned int texel_size_numbits;      // log2(texel_size)
  /* [한국어] log2(texel_size) — 텍셀 크기의 비트 수.
   * 설정자: cudaBindTexture() 경로에서 texel_size 계산 후 bitcount로 결정.
   * 읽는 자: TEX 명령어에서 텍셀 인덱스 → 바이트 오프셋 변환 시 비트 시프트 사용.
   * 값 범위: 0(1B), 1(2B), 2(4B), 3(8B), 4(16B). */
};

extern std::map<std::string, symbol_table *> g_sym_name_to_symbol_table;
/* [한국어] g_sym_name_to_symbol_table — 함수/커널 이름 → 심볼 테이블 전역 맵.
 * 설정자: PTX 파싱 완료 시 각 function_info의 심볼 테이블을 이름으로 등록.
 * 읽는 자: gpgpu_ptx_assemble() — 커널 이름으로 function_info의 심볼 테이블 조회.
 *           CUDA 런타임 — 커널 이름으로 함수 정보를 찾을 때 사용.
 * 동기화: PTX 파싱 완료 후 읽기 전용이므로 별도 락 불필요. */

/*
 * [한국어]
 * gpgpu_ptx_assemble() — PTX 커널을 어셈블하여 시뮬레이터 명령어 메모리에 배치
 *
 * @kname: 어셈블할 커널 이름 문자열
 * @kinfo: 커널 정보 포인터 (gpgpu_ptx_sim_info 캐스팅)
 * @return: 없음 (void)
 *
 * g_sym_name_to_symbol_table에서 kname으로 function_info를 찾아
 * function_info::ptx_assemble()을 호출하여 m_instr_mem 배열을 구성.
 * 최초 커널 실행 전 한 번만 호출되며, 이후 get_instruction(PC)로 명령어 접근 가능.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint → [gpgpu_ptx_assemble(kname, kinfo)] → function_info::ptx_assemble()
 */
void gpgpu_ptx_assemble(std::string kname, void *kinfo);
#include "../option_parser.h"
/*
 * [한국어]
 * ptx_kernel_shmem_size() — PTX 커널의 공유 메모리 사용량 반환
 *
 * @kernel_impl: function_info 포인터 (void* 캐스팅)
 * @return: 이 커널이 사용하는 공유 메모리 크기(바이트)
 *
 * m_kernel_info.smem 값 반환. 타이밍 모델에서 SM의 공유 메모리 가용 여부 확인.
 */
unsigned ptx_kernel_shmem_size(void *kernel_impl);
/*
 * [한국어]
 * ptx_kernel_nregs() — PTX 커널의 레지스터 사용량 반환
 *
 * @kernel_impl: function_info 포인터 (void* 캐스팅)
 * @return: 이 커널이 사용하는 레지스터 수 (스레드당)
 *
 * m_kernel_info.num_regs 값 반환. 타이밍 모델에서 SM의 레지스터 파일 가용 여부 확인.
 */
unsigned ptx_kernel_nregs(void *kernel_impl);

#endif
