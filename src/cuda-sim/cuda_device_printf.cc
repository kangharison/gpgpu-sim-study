/*
 * [한국어 설명] GPU 디바이스 printf() 에뮬레이션 구현 (cuda_device_printf.cc)
 *
 * === 파일의 역할 ===
 * CUDA 커널 내 printf() 호출을 GPGPU-Sim 기능 시뮬레이션에서 에뮬레이션하는
 * 두 함수를 구현한다. my_cuda_printf()는 형식 문자열을 직접 파싱하여 stdout에 출력하고,
 * gpgpusim_cuda_vprintf()는 PTX 스레드의 파라미터 메모리에서 형식 문자열과 인자를
 * 읽어 my_cuda_printf에 전달하는 상위 래퍼이다.
 * 현재 구현은 %u, %f, %d 세 가지 형식자만 지원하며 그 외는 abort().
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: cuda-sim/instructions.cc (call 명령어 처리) →
 *   gpgpusim_cuda_vprintf (파라미터 메모리 읽기) → my_cuda_printf (형식 파싱 + 출력)
 * 기능 시뮬레이션 레이어에 속하며, 타이밍 모델과 무관하게 PTX 실행 시 즉시 호출.
 * 실행 컨텍스트: 호스트 CPU 단일 스레드.
 *
 * === 타 모듈과의 연결 ===
 * 의존: cuda_device_printf.h, ptx_ir.h (ptx_instruction, ptx_thread_info, function_info,
 *   operand_info, symbol, memory_space_t 등)
 * 의존받음: cuda-sim/instructions.cc (call 명령어 디스패치)
 * 데이터 흐름: PTX 파라미터 메모리 → 형식 문자열 주소 + 인자 주소 →
 *   각 메모리 공간에서 바이트 읽기 → stdout 출력
 *
 * === 주요 함수/구조체 요약 ===
 * decode_space(): 주소와 메모리 공간 타입 기반으로 실제 memory_space 포인터 결정 (외부 함수)
 * my_cuda_printf(): 간단한 형식 문자열 파서 — %u/%f/%d만 지원, 문자 단위 순회
 * gpgpusim_cuda_vprintf(): PTX 파라미터에서 형식 문자열과 인자를 읽어 my_cuda_printf 호출
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

#include "cuda_device_printf.h" /* [한국어] gpgpusim_cuda_vprintf 선언 */
#include "ptx_ir.h"             /* [한국어] ptx_instruction, ptx_thread_info, function_info, operand_info, symbol 등 */

/*
 * [한국어]
 * decode_space - 주소와 메모리 공간 타입으로 실제 memory_space 포인터 결정 (외부 함수)
 *
 * @space: 메모리 공간 종류 (generic_space, shared_space 등) — 업데이트됨
 * @thread: 현재 PTX 스레드 — shared/local 메모리 포인터 접근
 * @op: 피연산자 정보 — 메모리 공간 타입 힌트 제공
 * @mem: 결정된 memory_space 포인터 — 업데이트됨
 * @addr: 해당 공간 내 오프셋 주소 — 업데이트됨
 *
 * 이 함수는 instructions.cc에 구현되어 있으며, generic 주소를 실제 메모리 공간으로 변환.
 */
void decode_space(memory_space_t &space, ptx_thread_info *thread,
                  const operand_info &op, memory_space *&mem, addr_t &addr);

/*
 * [한국어]
 * my_cuda_printf - GPU printf 형식 문자열 파서 및 출력 함수
 *
 * @fmtstr: C-스타일 형식 문자열 (예: "value=%d\n")
 * @arg_list: 인자 값들이 순서대로 저장된 연속 메모리 버퍼
 *
 * 형식 문자열을 문자 단위로 순회하며 '%' 발견 시 형식자 파싱:
 *   - %u/%d: arg_list에서 unsigned long long을 읽어 출력
 *   - %f: arg_list에서 double을 읽어 출력
 *   - 그 외: 지원하지 않음 → abort()
 * 현재 구현은 단일 형식자 처리만 가능 (복합 형식 문자열 %05.2f 등 불가).
 * arg_offset: 현재 인자 인덱스 (8바이트 단위로 가정).
 * 실행 컨텍스트: 기능 시뮬레이션 (gpgpusim_cuda_vprintf에서 호출).
 *
 * 호출 체인:
 *   gpgpusim_cuda_vprintf → [이 함수] → fprintf(stdout)
 */
void my_cuda_printf(const char *fmtstr, const char *arg_list) {
  FILE *fp = stdout;           /* [한국어] printf 출력 대상 = stdout */
  unsigned i = 0, j = 0;      /* [한국어] i: 형식 문자열 인덱스, j: buf 내 형식자 조각 인덱스 */
  unsigned arg_offset = 0;    /* [한국어] 현재 인자 인덱스 — arg_list에서 8바이트 단위로 전진 */
  char buf[64];                /* [한국어] 현재 파싱 중인 형식자를 저장하는 임시 버퍼 (예: "%d") */
  bool in_fmt = false;         /* [한국어] 현재 '%' 형식자를 파싱 중인지 여부 */
  while (fmtstr[i]) {          /* [한국어] 형식 문자열 끝('\0')까지 문자 단위 순회 */
    char c = fmtstr[i++];      /* [한국어] 현재 문자 읽기 후 인덱스 전진 */
    if (!in_fmt) {             /* [한국어] 일반 문자 처리 모드 */
      if (c != '%') {          /* [한국어] 일반 문자 — 그대로 출력 */
        fprintf(fp, "%c", c);  /* [한국어] 형식자 없는 단순 문자 출력 */
      } else {                 /* [한국어] '%' 발견 — 형식자 파싱 시작 */
        in_fmt = true;         /* [한국어] 형식자 파싱 모드로 전환 */
        buf[0] = c;            /* [한국어] 형식자 시작 '%' 저장 */
        j = 1;                 /* [한국어] 버퍼 인덱스를 다음 위치로 */
      }
    } else {                   /* [한국어] 형식자 파싱 모드 — '%' 이후 형식 지정자 문자 처리 */
      if (!(c == 'u' || c == 'f' || c == 'd')) { /* [한국어] 지원하지 않는 형식자 — 제한된 구현 */
        printf(
            "GPGPU-Sim PTX: ERROR ** printf parsing support is limited to %%u, "
            "%%f, %%d at present"); /* [한국어] 지원 형식 제한 오류 메시지 */
        abort(); /* [한국어] 복구 불가 오류 — 지원 추가 필요 */
      }
      buf[j] = c;              /* [한국어] 형식 지정자 문자 저장 (예: 'd', 'f', 'u') */
      buf[j + 1] = 0;          /* [한국어] null 종료로 버퍼를 완전한 형식자 문자열로 만들기 */
      void *ptr = (void *)&arg_list[arg_offset]; /* [한국어] 현재 인자의 메모리 주소 계산 — arg_offset 바이트 오프셋 */
      // unsigned long long value = ((unsigned long long*)arg_list)[arg_offset];
      if (c == 'u' || c == 'd') {  /* [한국어] 정수 형식 (%u/%d) — unsigned long long으로 읽어 출력 */
        fprintf(fp, buf, *((unsigned long long *)ptr)); /* [한국어] 8바이트 정수 값을 buf 형식으로 출력 */
      } else if (c == 'f') {   /* [한국어] 부동소수점 형식 (%f) — double로 읽어 출력 */
        double tmp = *((double *)ptr); /* [한국어] 8바이트 double로 해석 */
        fprintf(fp, buf, tmp); /* [한국어] buf 형식으로 출력 */
      }
      arg_offset++;            /* [한국어] 다음 인자로 이동 (8바이트 단위 가정) */
      in_fmt = false;          /* [한국어] 형식자 파싱 완료 — 일반 문자 모드로 복귀 */
    }
  }
}

/*
 * [한국어]
 * gpgpusim_cuda_vprintf - CUDA printf() PTX call 명령어 에뮬레이션 핸들러
 *
 * @pI: 현재 실행 중인 PTX call 명령어 — 실제 파라미터 피연산자 조회 기반
 * @thread: printf를 호출한 PTX 스레드 — 로컬 메모리/param 메모리 접근
 * @target_func: vprintf 함수 정보 — has_return, num_args, get_arg, get_return_var
 *
 * CUDA vprintf 함수 시그니처: int vprintf(const char *fmt, void *valist).
 * PTX 파라미터에서 두 개 인자를 읽음:
 *   arg 0: 형식 문자열에 대한 포인터 (generic 주소) → decode_space로 실제 공간 결정 → 바이트 단위 읽기
 *   arg 1: 가변 인자 목록에 대한 포인터 → 로컬 메모리 프레임 크기만큼 읽기
 * 두 버퍼를 읽은 후 my_cuda_printf 호출 → 실제 출력 수행.
 * 메모리 읽기 후 malloc으로 할당한 버퍼는 출력 후 free.
 * 실행 컨텍스트: 기능 시뮬레이션 (호스트 CPU 단일 스레드).
 *
 * 호출 체인:
 *   instructions.cc (call 명령어 → printf 디스패치) → [이 함수]
 *     → decode_space, memory_space::read, my_cuda_printf
 */
void gpgpusim_cuda_vprintf(const ptx_instruction *pI, ptx_thread_info *thread,
                           const function_info *target_func) {
  char *fmtstr = NULL;    /* [한국어] 형식 문자열을 저장할 버퍼 포인터 (malloc으로 할당) */
  char *arg_list = NULL;  /* [한국어] 가변 인자 목록을 저장할 버퍼 포인터 (malloc으로 할당) */
  unsigned n_return = target_func->has_return(); /* [한국어] vprintf는 리턴값(int) 있음 — 피연산자 인덱스 계산용 */
  unsigned n_args = target_func->num_args();     /* [한국어] vprintf 인자 수 — 반드시 2개여야 함 */
  assert(n_args == 2); /* [한국어] fmt 포인터 + valist 포인터 = 정확히 2개 */
  for (unsigned arg = 0; arg < n_args; arg++) { /* [한국어] 2개 인자 순서대로 처리 */
    const operand_info &actual_param_op =
        pI->operand_lookup(n_return + 1 + arg); /* [한국어] PTX 명령어에서 arg번째 실제 파라미터 피연산자 조회 */
    const symbol *formal_param = target_func->get_arg(arg); /* [한국어] 함수 정의에서 arg번째 형식 파라미터 정보 */
    unsigned size = formal_param->get_size_in_bytes(); /* [한국어] 이 파라미터의 크기(바이트) */
    assert(formal_param->is_param_local());   /* [한국어] 파라미터는 로컬 파라미터 공간에 있어야 함 */
    assert(actual_param_op.is_param_local()); /* [한국어] 실제 피연산자도 로컬 파라미터여야 함 */
    addr_t from_addr = actual_param_op.get_symbol()->get_address(); /* [한국어] 이 파라미터의 로컬 메모리 내 주소 */
    unsigned long long buffer[1024];          /* [한국어] 파라미터 값을 읽을 임시 버퍼 */
    assert(size < 1024 * sizeof(unsigned long long)); /* [한국어] 버퍼 크기 초과 방지 */
    thread->m_local_mem->read(from_addr, size, buffer); /* [한국어] 스레드 로컬 메모리에서 파라미터 값 읽기 */
    addr_t addr =
        (addr_t)buffer[0];  // should be pointer to generic memory location
    /* [한국어] buffer[0]은 형식 문자열(또는 인자 목록)에 대한 generic 주소 포인터 */
    memory_space *mem = NULL;            /* [한국어] 실제 메모리 공간 포인터 — decode_space가 설정 */
    memory_space_t space = generic_space; /* [한국어] 초기 공간 타입 = generic (알 수 없음) */
    decode_space(space, thread, actual_param_op, mem,
                 addr);  // figure out which space
    /* [한국어] generic 주소를 실제 메모리 공간과 오프셋으로 변환 — shared/local/global 등 결정 */
    if (arg == 0) { /* [한국어] 첫 번째 인자 = 형식 문자열 포인터 */
      unsigned len = 0;
      char b = 0;
      do {  // figure out length
        /* [한국어] 형식 문자열 길이 계산 — null 종료까지 바이트 단위 읽기 */
        mem->read(addr + len, 1, &b); /* [한국어] addr+len 위치에서 1바이트 읽기 */
        len++;                        /* [한국어] 길이 카운터 증가 */
      } while (b);                    /* [한국어] null 바이트('\0') 읽으면 종료 */
      fmtstr = (char *)malloc(len + 64); /* [한국어] 형식 문자열 버퍼 할당 (+64: 안전 마진) */
      for (int i = 0; i < len; i++) mem->read(addr + i, 1, fmtstr + i); /* [한국어] 형식 문자열을 1바이트씩 읽어 fmtstr에 복사 */
      // mem->read(addr,len,fmtstr);
      /* [한국어] 한 번에 읽는 방식도 가능하나 메모리 공간 경계 처리 문제로 바이트 단위 사용 */
    } else { /* [한국어] 두 번째 인자 = 가변 인자 목록 포인터 */
      unsigned len = thread->get_finfo()->local_mem_framesize(); /* [한국어] 현재 스택 프레임 크기 = 인자 목록 전체 크기 */
      arg_list = (char *)malloc(len + 64); /* [한국어] 인자 목록 버퍼 할당 (+64: 안전 마진) */
      for (int i = 0; i < len; i++) mem->read(addr + i, 1, arg_list + i); /* [한국어] 인자 목록을 1바이트씩 읽어 arg_list에 복사 */
      // mem->read(addr,len,arg_list);
    }
  }
  my_cuda_printf(fmtstr, arg_list); /* [한국어] 형식 문자열과 인자 목록으로 실제 출력 수행 */
  free(fmtstr);   /* [한국어] 형식 문자열 버퍼 해제 */
  free(arg_list); /* [한국어] 인자 목록 버퍼 해제 */
}
