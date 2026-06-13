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

/*
 * [한국어 설명] PTX 인터랙티브 디버거 구현 (debug.cc)
 *
 * === 파일의 역할 ===
 * GPGPU-Sim 시뮬레이션 실행 중 사용자가 PTX 명령어 수준에서 대화형으로 디버깅할 수
 * 있는 인터랙티브 디버거를 구현한다. 브레이크포인트(파일:라인+스레드UID)와
 * 워치포인트(글로벌 메모리 주소 변경 감지)를 지원한다. 매 시뮬레이션 사이클마다
 * gpgpu_sim::gpgpu_debug()가 호출되어 중단 조건을 검사하고, 조건 충족 시 stdin에서
 * 사용자 명령을 읽는 대화형 루프로 진입한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: CUDA 앱 → libcuda → gpgpusim_entrypoint → gpu-sim.cc(cycle loop)
 *   → gpgpu_sim::gpgpu_debug() [이 파일] → PTX 스레드 상태 조회/출력
 * 타이밍 모델(gpgpu-sim/)에서 매 사이클 호출되지만, 실제 디버깅 대상은
 * 기능 시뮬레이션(cuda-sim/)의 PTX 스레드 상태이다.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 메인 루프 내 단일 스레드.
 *
 * === 타 모듈과의 연결 ===
 * 의존: debug.h (brk_pt), cuda-sim/ptx_ir.h (ptx_thread_info), cuda-sim/ptx_sim.h,
 *       gpgpu-sim/gpu-sim.h (gpgpu_sim 클래스, m_global_mem, dump_pipeline),
 *       gpgpu-sim/shader.h (SM 파이프라인 덤프)
 * 호출자: gpu-sim.cc의 gpgpu_sim::cycle() 또는 동등한 시뮬레이션 루프
 * 데이터 흐름: gpgpu_sim 객체의 m_global_mem(글로벌 메모리)을 직접 읽어 워치포인트
 *   값 변경 감지; ptx_thread_info로 스레드 위치/UID/PC 조회.
 *
 * === 주요 함수/구조체 요약 ===
 * gpgpu_sim::hit_watchpoint()  - 워치포인트 트리거 이벤트를 g_watchpoint_hits에 기록
 * gpgpu_sim::gpgpu_debug()     - 매 사이클 호출되는 인터랙티브 디버거 진입점
 * thread_at_brkpt()            - PTX 스레드가 브레이크포인트 조건을 충족하는지 확인
 */

#include "debug.h"           /* [한국어] brk_pt 클래스 정의 */
#include "cuda-sim/cuda-sim.h" /* [한국어] CUDA 기능 시뮬레이션 인터페이스 */
#include "cuda-sim/ptx_ir.h"   /* [한국어] ptx_thread_info, ptx_instruction 클래스 */
#include "cuda-sim/ptx_sim.h"  /* [한국어] PTX 시뮬레이션 유틸리티 */
#include "gpgpu-sim/gpu-sim.h" /* [한국어] gpgpu_sim 클래스 (m_global_mem, dump_pipeline 등) */
#include "gpgpu-sim/shader.h"  /* [한국어] SM 파이프라인 덤프 함수 */

#include <stdio.h>   /* [한국어] printf, fgets, stdin/stdout 입출력 */
#include <string.h>  /* [한국어] strtok, strcmp 문자열 처리 */
#include <map>       /* [한국어] std::map<unsigned, brk_pt> breakpoints 정적 변수용 */

/*
 * [한국어]
 * gpgpu_sim::hit_watchpoint() - 워치포인트 트리거 이벤트 기록
 *
 * @watchpoint_num: 트리거된 워치포인트 번호 (breakpoints 맵의 키값)
 * @thd: 워치포인트를 트리거한 PTX 스레드 포인터
 * @pI: 워치포인트 메모리 쓰기를 수행한 PTX 명령어 포인터
 *
 * cuda-sim/ 레이어에서 메모리 쓰기가 발생할 때 호출되어 g_watchpoint_hits 맵에
 * 이벤트를 기록한다. 이후 gpgpu_debug()가 이 맵을 확인하여 어떤 스레드/명령어가
 * 워치포인트를 트리거했는지 사용자에게 보고한다.
 * 실행 컨텍스트: cuda-sim/ 기능 시뮬레이션 실행 중 (메모리 쓰기 시점).
 *
 * 호출 체인:
 *   cuda-sim/(메모리 쓰기) → [hit_watchpoint()] → g_watchpoint_hits에 기록
 *   → gpgpu_debug()에서 g_watchpoint_hits 조회 → 사용자에게 출력
 */
void gpgpu_sim::hit_watchpoint(unsigned watchpoint_num, ptx_thread_info *thd,
                               const ptx_instruction *pI) {
  g_watchpoint_hits[watchpoint_num] = watchpoint_event(thd, pI);
  /* [한국어] 워치포인트 번호를 키로, 트리거한 스레드+명령어 쌍을 값으로 기록.
   * gpgpu_debug()가 다음 사이클에 이 이벤트를 감지하여 출력. */
}

/// interactive debugger

/*
 * [한국어]
 * gpgpu_sim::gpgpu_debug() - PTX 인터랙티브 디버거 메인 루프
 *
 * 매 시뮬레이션 사이클마다 gpu-sim.cc의 cycle() 루프에서 호출된다.
 * single_step=true이면 매 사이클마다 대화형 루프에 진입한다.
 * breakpoints 맵을 순회하여 워치포인트(메모리 변경 감지)와 브레이크포인트(파일:라인
 * 도달) 조건을 검사하고, 조건 충족 시 stdin에서 사용자 명령을 읽는 루프로 진입한다.
 * 지원 명령: dp(파이프라인 덤프), q(종료), b(브레이크포인트 설정), d(삭제),
 *            s(단일 스텝), c(연속 실행), w(워치포인트 설정), l(소스 목록), h(도움말).
 * 실행 컨텍스트: 시뮬레이션 메인 루프, 호스트 유저스페이스, 단일 스레드.
 *
 * 호출 체인:
 *   gpu-sim.cc::gpgpu_sim::cycle() → [gpgpu_debug()] → stdin 명령 처리
 *   → dump_pipeline() / m_global_mem->read() / ptx_thread_info 조회
 */
void gpgpu_sim::gpgpu_debug() {
  bool done = true; /* [한국어] 대화형 루프 탈출 조건: true이면 루프 건너뜀 */

  static bool single_step = true;
  /* [한국어] 단일-스텝 모드 플래그. true이면 매 사이클 대화형 루프 진입.
   * 's' 명령: 값 유지(다음 사이클도 중단). 'c' 명령: false로 설정(연속 실행). */
  static unsigned next_brkpt = 1;
  /* [한국어] 다음 브레이크포인트/워치포인트에 할당할 번호. 1부터 시작하여 증가. */
  static std::map<unsigned, brk_pt> breakpoints;
  /* [한국어] 등록된 모든 브레이크포인트/워치포인트의 맵. 키=번호, 값=brk_pt 객체.
   * 정적 변수로 여러 gpgpu_debug() 호출 간 상태 유지. */

  /// if single stepping, go to interactive debugger

  if (single_step) done = false;
  /* [한국어] single_step 모드이면 done=false로 설정해 아래 while 루프에 진입 */

  /// check if we've reached a breakpoint
  const ptx_thread_info *brk_thd = NULL;  /* [한국어] 브레이크포인트를 트리거한 스레드 포인터 */
  const ptx_instruction *brk_inst = NULL; /* [한국어] 브레이크포인트를 트리거한 PTX 명령어 포인터 */

  for (std::map<unsigned, brk_pt>::iterator i = breakpoints.begin();
       i != breakpoints.end(); i++) {
    /* [한국어] 모든 등록된 브레이크포인트/워치포인트를 순회하며 조건 검사 */
    unsigned num = i->first;   /* [한국어] 브레이크포인트 번호 */
    brk_pt &b = i->second;    /* [한국어] 해당 brk_pt 객체 참조 */
    if (b.is_watchpoint()) {
      /* [한국어] 워치포인트 모드: 메모리 값 변경 감지 */
      unsigned addr = b.get_addr(); /* [한국어] 감시 대상 글로벌 메모리 주소 */
      unsigned new_value;
      m_global_mem->read(addr, 4, &new_value);
      /* [한국어] 글로벌 메모리에서 4바이트 현재 값 읽기. DMA나 PTX 쓰기로 변경 가능. */
      if (new_value != b.get_value() ||
          g_watchpoint_hits.find(num) != g_watchpoint_hits.end()) {
        /* [한국어] 메모리 값이 변경됐거나 hit_watchpoint()가 이벤트를 기록한 경우 */
        printf(
            "GPGPU-Sim PTX DBG: watch point %u triggered (old value=%x, new "
            "value=%x)\n",
            num, b.get_value(), new_value);
        /* [한국어] 워치포인트 번호와 이전값/새값을 16진수로 출력 */
        std::map<unsigned, watchpoint_event>::iterator w =
            g_watchpoint_hits.find(num);
        /* [한국어] hit_watchpoint()가 기록한 스레드/명령어 이벤트 조회 */
        if (w == g_watchpoint_hits.end())
          printf("GPGPU-Sim PTX DBG: memory transfer modified value\n");
          /* [한국어] cuda-sim 쓰기가 아닌 DMA/메모리 전송에 의한 값 변경 */
        else {
          watchpoint_event wa = w->second; /* [한국어] 트리거 이벤트(스레드+명령어) 복사 */
          brk_thd = wa.thread();           /* [한국어] 트리거한 스레드 포인터 저장 */
          brk_inst = wa.inst();            /* [한국어] 트리거한 PTX 명령어 포인터 저장 */
          printf(
              "GPGPU-Sim PTX DBG: modified by thread uid=%u, sid=%u, "
              "hwtid=%u\n",
              brk_thd->get_uid(), brk_thd->get_hw_sid(), brk_thd->get_hw_tid());
          /* [한국어] 트리거 스레드의 UID(전역 고유ID), sid(SM번호), hw_tid(HW 스레드번호) 출력 */
          printf("GPGPU-Sim PTX DBG: ");
          brk_inst->print_insn(stdout); /* [한국어] 트리거한 PTX 명령어를 stdout에 출력 */
          printf("\n");
          g_watchpoint_hits.erase(w); /* [한국어] 처리한 이벤트를 맵에서 제거 */
        }
        b.set_value(new_value); /* [한국어] 기준값을 현재 값으로 갱신 (다음 변경 감지용) */
        done = false;           /* [한국어] 대화형 루프 진입 표시 */
      }
    } else {
      /*
     for( unsigned sid=0; sid < m_n_shader; sid++ ) {
        unsigned hw_thread_id = -1;
        abort();
        ptx_thread_info *thread =
     m_sc[sid]->get_functional_thread(hw_thread_id); if( thread_at_brkpt(thread,
     b) ) { done = false; printf("GPGPU-Sim PTX DBG: reached breakpoint %u at %s
     (sm=%u, hwtid=%u)\n", num, b.location().c_str(), sid, hw_thread_id );
           brk_thd = thread;
           brk_inst = brk_thd->get_inst();
           printf( "GPGPU-Sim PTX DBG: reached by thread uid=%u, sid=%u,
     hwtid=%u\n", brk_thd->get_uid(),brk_thd->get_hw_sid(),
     brk_thd->get_hw_tid() ); printf( "GPGPU-Sim PTX DBG: ");
           brk_inst->print_insn(stdout);
           printf( "\n" );
        }
     }
     */
      /* [한국어] 파일:라인 브레이크포인트 처리 코드. 현재 주석 처리됨(미구현).
       * get_functional_thread() API 변경으로 abort()가 추가되어 비활성 상태. */
    }
  }

  if (done) assert(g_watchpoint_hits.empty());
  /* [한국어] 중단 조건이 없으면 미처리 워치포인트 이벤트도 없어야 함 (불변식 검사) */

  /// enter interactive debugger loop

  while (!done) {
    /* [한국어] 브레이크포인트/워치포인트 조건 충족 또는 single_step 모드 시 진입 */
    printf("(ptx debugger) "); /* [한국어] 프롬프트 출력 */
    fflush(stdout);            /* [한국어] 버퍼 즉시 출력 (프롬프트가 보이도록) */

    char line[1024];           /* [한국어] stdin에서 읽은 명령 줄 버퍼 */
    char *ptr = fgets(line, 1024, stdin); /* [한국어] stdin에서 최대 1023자 한 줄 읽기 */

    char *tok = strtok(line, " \t\n"); /* [한국어] 첫 번째 토큰(명령어) 파싱 */
    if (!strcmp(tok, "dp")) {
      /* [한국어] 'dp <n>': SM 번호 n의 파이프라인 내용 덤프 */
      int shader_num = 0;
      tok = strtok(NULL, " \t\n"); /* [한국어] 다음 토큰: SM 번호 */
      sscanf(tok, "%d", &shader_num); /* [한국어] SM 번호를 정수로 파싱 */
      dump_pipeline((0x40 | 0x4 | 0x1), shader_num, 0);
      /* [한국어] 플래그 0x45: 파이프라인 스테이지 덤프 옵션 조합.
       * shader.cc의 dump_pipeline() 호출로 지정 SM의 파이프라인 상태 출력. */
      printf("\n");
      fflush(stdout);
    } else if (!strcmp(tok, "q") || !strcmp(tok, "quit")) {
      /* [한국어] 'q' 또는 'quit': 시뮬레이터 종료 확인 후 exit() */
      printf("\nreally quit GPGPU-Sim (y/n)?\n");
      ptr = fgets(line, 1024, stdin); /* [한국어] 확인 입력 읽기 */
      if (ptr == NULL) {
        printf("can't read input\n"); /* [한국어] stdin 읽기 실패 (EOF 등) */
        exit(0);
      }
      tok = strtok(line, " \t\n");
      if (!strcmp(tok, "y")) {
        exit(0); /* [한국어] 'y' 입력 시 프로세스 즉시 종료 */
      } else {
        printf("not quiting.\n"); /* [한국어] 다른 입력 시 계속 실행 */
      }
    } else if (!strcmp(tok, "b")) {
      /* [한국어] 'b <file>:<line> <thread_uid>': 파일:라인 브레이크포인트 설정 */
      tok = strtok(NULL, " \t\n");      /* [한국어] 다음 토큰: 파일:라인 문자열 */
      char brkpt[1024];
      sscanf(tok, "%s", brkpt);          /* [한국어] 파일:라인 문자열 읽기 */
      tok = strtok(NULL, " \t\n");       /* [한국어] 다음 토큰: 스레드 UID */
      unsigned uid;
      sscanf(tok, "%u", &uid);           /* [한국어] 스레드 UID를 unsigned 정수로 파싱 */
      breakpoints[next_brkpt++] = brk_pt(brkpt, uid);
      /* [한국어] 새 브레이크포인트를 맵에 등록하고 번호 증가 */
    } else if (!strcmp(tok, "d")) {
      /* [한국어] 'd <n>': 번호 n의 브레이크포인트 삭제 */
      tok = strtok(NULL, " \t\n");
      unsigned uid;
      sscanf(tok, "%u", &uid);           /* [한국어] 삭제할 브레이크포인트 번호 파싱 */
      breakpoints.erase(uid);            /* [한국어] 맵에서 해당 번호의 브레이크포인트 제거 */
    } else if (!strcmp(tok, "s")) {
      /* [한국어] 's': 단일 스텝 — 현재 명령 완료 후 다시 디버거 진입 (done=true) */
      done = true; /* [한국어] while 루프 탈출. single_step=true이므로 다음 사이클 재진입 */
    } else if (!strcmp(tok, "c")) {
      /* [한국어] 'c': 연속 실행 — single_step 해제 후 루프 탈출 */
      single_step = false; /* [한국어] 이후 브레이크포인트 조건 없으면 디버거 미진입 */
      done = true;         /* [한국어] while 루프 탈출 */
    } else if (!strcmp(tok, "w")) {
      /* [한국어] 'w <hex_addr>': 글로벌 메모리 주소에 워치포인트 설정 */
      tok = strtok(NULL, " \t\n");
      unsigned addr;
      sscanf(tok, "%x", &addr);           /* [한국어] 16진수 주소 파싱 */
      unsigned value;
      m_global_mem->read(addr, 4, &value); /* [한국어] 현재 메모리 값 읽기 (초기 기준값) */
      m_global_mem->set_watch(addr, next_brkpt);
      /* [한국어] 글로벌 메모리에 워치포인트 등록 — 쓰기 시 hit_watchpoint() 호출 트리거 */
      breakpoints[next_brkpt++] = brk_pt(addr, value);
      /* [한국어] 워치포인트를 brk_pt로 생성하여 breakpoints 맵에 등록 */
    } else if (!strcmp(tok, "l")) {
      /* [한국어] 'l': 현재 브레이크포인트 위치 주변 PTX 소스 코드 목록 출력 */
      if (brk_thd == NULL) {
        printf("no thread selected\n"); /* [한국어] 스레드가 선택되지 않은 경우 */
      } else {
        addr_t pc = brk_thd->get_pc();           /* [한국어] 현재 스레드 PC(프로그램 카운터) */
        addr_t start_pc = (pc < 5) ? 0 : (pc - 5); /* [한국어] 현재 PC 기준 앞 5줄부터 시작 */
        for (addr_t p = start_pc; p <= pc + 5; p++) {
          /* [한국어] 현재 PC 앞뒤 5줄 범위의 PTX 명령어 출력 */
          const ptx_instruction *i = brk_thd->get_inst(p); /* [한국어] PC p의 PTX 명령어 조회 */
          if (i) {
            if (p != pc)
              printf("    "); /* [한국어] 현재 실행 위치 아닌 줄은 4칸 들여쓰기 */
            else
              printf("==> "); /* [한국어] 현재 실행 위치 표시 */
            i->print_insn(stdout); /* [한국어] PTX 명령어 텍스트 출력 */
            printf("\n");
          }
        }
      }
    } else if (!strcmp(tok, "h")) {
      /* [한국어] 'h': 도움말 출력 — 사용 가능한 모든 디버거 명령 나열 */
      printf("commands:\n");
      printf("  q                           - quit GPGPU-Sim\n");
      printf("  b <file>:<line> <thead uid> - set breakpoint\n");
      printf("  w <global address>          - set watchpoint\n");
      printf("  del <n>                     - delete breakpoint\n");
      printf(
          "  s                           - single step one shader cycle (all "
          "cores)\n");
      printf(
          "  c                           - continue simulation without single "
          "stepping\n");
      printf(
          "  l                           - list PTX around current "
          "breakpoint\n");
      printf(
          "  dp <n>                      - display pipeline contents on SM "
          "<n>\n");
      printf("  h                           - print this message\n");
    } else {
      printf("\ncommand not understood.\n"); /* [한국어] 알 수 없는 명령어 오류 메시지 */
    }
    fflush(stdout); /* [한국어] 출력 버퍼 즉시 플러시 */
  }
}

/*
 * [한국어]
 * thread_at_brkpt() - PTX 스레드가 브레이크포인트 조건에 해당하는지 확인
 *
 * @thread: 검사할 PTX 스레드 포인터 (get_location(), get_uid() 제공)
 * @b: 비교할 brk_pt 브레이크포인트 객체
 * @return: true이면 스레드 위치가 브레이크포인트 조건과 일치 (중단 필요)
 *
 * thread의 현재 PTX 소스 위치(파일:라인)와 UID를 brk_pt::is_equal()에 전달해
 * 브레이크포인트 조건 충족 여부를 확인한다. 워치포인트 모드 brk_pt에는 false 반환.
 * 현재 gpgpu_debug()의 브레이크포인트 처리 코드가 주석 처리되어 실제 호출 안 됨.
 *
 * 호출 체인:
 *   gpgpu_sim::gpgpu_debug() → [thread_at_brkpt()] → brk_pt::is_equal()
 */
bool thread_at_brkpt(ptx_thread_info *thread, const class brk_pt &b) {
  return b.is_equal(thread->get_location(), thread->get_uid());
  /* [한국어] 스레드 현재 위치(파일:라인 문자열)와 UID를 is_equal()에 전달하여 비교 */
}
