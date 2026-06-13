// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda
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
 * [한국어 설명] GPGPU-Sim 통계 도구 구현 (stat-tool.cc)
 *
 * === 파일의 역할 ===
 * stat-tool.h에 선언된 모든 통계 로거 클래스와 전역 함수를 구현한다.
 * 크게 세 부분으로 구성된다:
 * (1) 스냅샷/스필 자동화 인프라: snap_shot_trigger 리스트, spill_log 리스트 관리.
 * (2) SM별 각종 통계 로거 구현: 워프 점유율, 메모리 접근, 레이턴시, 캐시 미스, CTA 수.
 * (3) thread_CFlocality, linear_histogram_logger 클래스 멤버 함수 구현.
 * 모든 로거는 전역 정적 변수로 관리되며, Create/Log/Snapshot/Print 네이밍 패턴을 따른다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: gpu-sim.cc::cycle() → try_snap_shot() → 각 로거의 snap_shot()
 *            shader.cc → shader_warp_occ_log/shader_cache_access_log 등
 *            시뮬레이션 종료 → spill_log_to_file(final=1) → 최종 통계 파일 출력
 *            visualizer.cc → cflog/shader_CTA_count visualizer 함수 호출
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 메인 루프 내.
 *
 * === 타 모듈과의 연결 ===
 * 의존: stat-tool.h (클래스/함수 선언), histogram.h, libcuda/gpgpu_context.h,
 *       zlib.h (gzFile), assert.h, stdio.h
 * 사용처: gpu-sim.cc (try_snap_shot, spill_log_to_file),
 *          shader.cc (shader_warp_occ_log, shader_cache_access_log 등)
 * 데이터 흐름: 각 Log 함수 → 해당 로거의 log() → m_curr_lin_hist 누적
 *   → try_snap_shot() → snap_shot() → m_lin_hist_archive 이동
 *   → spill_log_to_file() → 파일 출력 + 메모리 해제
 *
 * === 주요 함수/구조체 요약 ===
 * add_snap_shot_trigger()  - 새 로거를 전역 스냅샷 트리거 리스트에 등록
 * try_snap_shot()          - 현재 사이클이 스냅샷 주기이면 모든 로거 snap_shot() 호출
 * spill_log_to_file()      - 모든 로거의 아카이브를 파일로 내보냄
 * shader_mem_lat_log()     - log-sqrt2 스케일로 레이턴시 bin 계산 (비트 연산 최적화)
 * thread_CFlocality::snap_shot() - 현재 span을 아카이브에 저장하고 새 span 시작
 * linear_histogram_logger::spill() - 아카이브를 파일에 출력하고 메모리 해제
 */

#include "stat-tool.h"

#include <assert.h>   /* [한국어] 불변식 검사 */
#include <stdio.h>    /* [한국어] printf, fprintf 출력 */
#include <stdlib.h>   /* [한국어] abort() 등 */
#include <zlib.h>     /* [한국어] gzFile, gzprintf — AerialVision 압축 출력 */
#include <algorithm>  /* [한국어] std::fill */
#include <list>       /* [한국어] std::list — 아카이브, 트리거 리스트 */
#include <map>        /* [한국어] std::map */
#include <string>     /* [한국어] std::string */
#include <vector>     /* [한국어] std::vector */
#include "../../libcuda/gpgpu_context.h" /* [한국어] gpgpu_context::translate_pc_to_ptxlineno() */

////////////////////////////////////////////////////////////////////////////////

static unsigned long long min_snap_shot_interval = 0;
/* [한국어] 등록된 모든 snap_shot_trigger 중 최소 스냅샷 간격.
 * 최적화: 모든 간격이 최소 간격의 배수라고 가정 — 최소 간격 주기마다 try_snap_shot 실행.
 * 설정자: add_snap_shot_trigger(). 읽는 자: try_snap_shot(). */

static unsigned long long next_snap_shot_cycle = 0;
/* [한국어] 다음 스냅샷이 촬영될 사이클 번호.
 * 설정자: add_snap_shot_trigger(초기화), try_snap_shot()(스냅샷 후 next 갱신).
 * 읽는 자: try_snap_shot()에서 current_cycle과 비교. */

static std::list<snap_shot_trigger *> list_ss_trigger;
/* [한국어] 등록된 모든 snap_shot_trigger 포인터의 리스트.
 * 설정자: add_snap_shot_trigger(), remove_snap_shot_trigger().
 * 읽는 자: try_snap_shot()에서 순회하여 각 트리거의 snap_shot() 호출. */

/*
 * [한국어]
 * add_snap_shot_trigger() - 새 snap_shot_trigger를 전역 리스트에 등록
 *
 * @ss_trigger: 등록할 snap_shot_trigger 포인터
 *
 * 새 로거의 간격이 현재 최소 간격보다 작으면 min_snap_shot_interval 갱신.
 * 모든 간격이 서로 배수 관계라고 가정하는 빠른 최적화 방식 사용.
 * 실행 컨텍스트: 시뮬레이터 초기화 시, shader_warp_occ_create() 등에서 호출.
 */
void add_snap_shot_trigger(snap_shot_trigger *ss_trigger) {
  // quick optimization assuming that all snap shot intervals are perfect
  // multiples of each other
  if (min_snap_shot_interval == 0 ||
      min_snap_shot_interval > ss_trigger->get_interval()) {
    /* [한국어] 처음 등록이거나 더 작은 간격의 트리거이면 최소 간격 갱신 */
    min_snap_shot_interval = ss_trigger->get_interval();
    next_snap_shot_cycle =
        min_snap_shot_interval;  // assume that snap shots haven't started yet
    /* [한국어] 다음 스냅샷 주기를 최소 간격으로 설정 (시뮬레이션 아직 시작 안 했다고 가정) */
  }
  list_ss_trigger.push_back(ss_trigger); /* [한국어] 트리거 리스트 끝에 추가 */
}

/*
 * [한국어]
 * remove_snap_shot_trigger() - snap_shot_trigger를 전역 리스트에서 제거
 *
 * @ss_trigger: 제거할 snap_shot_trigger 포인터
 *
 * linear_histogram_logger 소멸자에서 this를 제거할 때 호출.
 */
void remove_snap_shot_trigger(snap_shot_trigger *ss_trigger) {
  list_ss_trigger.remove(ss_trigger); /* [한국어] 리스트에서 해당 포인터 값의 항목 제거 */
}

/*
 * [한국어]
 * try_snap_shot() - 현재 사이클이 스냅샷 주기이면 모든 트리거의 snap_shot() 호출
 *
 * @current_cycle: 현재 시뮬레이션 사이클 번호
 *
 * min_snap_shot_interval == 0이면 (등록된 트리거 없음) 즉시 반환.
 * current_cycle == next_snap_shot_cycle일 때만 모든 트리거 순회하여 snap_shot() 호출.
 * 이후 next_snap_shot_cycle을 min_snap_shot_interval만큼 증가.
 * 실행 컨텍스트: gpu-sim.cc::cycle() 루프에서 매 사이클 호출.
 *
 * 호출 체인:
 *   gpu-sim.cc::gpgpu_sim::cycle() → [try_snap_shot()] → 각 로거의 snap_shot()
 */
void try_snap_shot(unsigned long long current_cycle) {
  if (min_snap_shot_interval == 0) return; /* [한국어] 등록된 트리거 없으면 조기 반환 */
  if (current_cycle != next_snap_shot_cycle) return; /* [한국어] 스냅샷 주기 미도달 */

  std::list<snap_shot_trigger *>::iterator ss_trigger_iter =
      list_ss_trigger.begin();
  for (; ss_trigger_iter != list_ss_trigger.end(); ++ss_trigger_iter) {
    (*ss_trigger_iter)
        ->snap_shot(current_cycle);  // WF: should be try_snap_shot
    /* [한국어] 각 로거의 snap_shot() 직접 호출 (WF 코멘트: try_snap_shot이 맞을 것) */
  }
  next_snap_shot_cycle =
      current_cycle +
      min_snap_shot_interval;  // WF: stateful testing, maybe bad
  /* [한국어] 다음 스냅샷 주기를 현재 사이클 + 최소 간격으로 설정 */
}

////////////////////////////////////////////////////////////////////////////////

static unsigned long long spill_interval = 0;
/* [한국어] 로그 파일 내보내기(spill) 사이클 간격. 0이면 final 시에만 스필.
 * 설정자: set_spill_interval(). 읽는 자: spill_log_to_file(). */

static unsigned long long next_spill_cycle = 0;
/* [한국어] 다음 스필 수행 사이클. 설정자: set_spill_interval(), spill_log_to_file().
 * 읽는 자: spill_log_to_file()에서 current_cycle과 비교. */

static std::list<spill_log_interface *> list_spill_log;
/* [한국어] 등록된 모든 spill_log_interface 포인터의 리스트.
 * 설정자: add_spill_log(), remove_spill_log().
 * 읽는 자: spill_log_to_file()에서 순회하여 spill() 호출. */

/*
 * [한국어]
 * add_spill_log() - 새 spill_log_interface를 전역 리스트에 등록
 *
 * @spill_log: 등록할 spill_log_interface 포인터
 *
 * shader_warp_occ_create() 등에서 로거 생성 후 호출.
 */
void add_spill_log(spill_log_interface *spill_log) {
  list_spill_log.push_back(spill_log); /* [한국어] 스필 로그 리스트 끝에 추가 */
}

/*
 * [한국어]
 * remove_spill_log() - spill_log_interface를 전역 리스트에서 제거
 *
 * @spill_log: 제거할 spill_log_interface 포인터
 *
 * linear_histogram_logger 소멸자에서 호출.
 */
void remove_spill_log(spill_log_interface *spill_log) {
  list_spill_log.remove(spill_log); /* [한국어] 리스트에서 해당 포인터 제거 */
}

/*
 * [한국어]
 * set_spill_interval() - 로그 파일 내보내기 간격 설정
 *
 * @interval: 스필 사이클 간격. 0이면 자동 스필 비활성화(final 시에만 스필).
 */
void set_spill_interval(unsigned long long interval) {
  spill_interval = interval;          /* [한국어] 스필 간격 설정 */
  next_spill_cycle = spill_interval;  /* [한국어] 첫 스필 사이클 초기화 */
}

/*
 * [한국어]
 * spill_log_to_file() - 모든 로거의 아카이브를 파일로 내보냄
 *
 * @fout: 출력 대상 FILE*
 * @final: 0=정기 스필, 1=최종 출력 (현재 버퍼도 포함)
 * @current_cycle: 현재 시뮬레이션 사이클
 *
 * spill_interval==0이고 final이 아니면 즉시 반환 (자동 스필 비활성 상태).
 * current_cycle <= next_spill_cycle이면 스필 주기 미도달로 반환.
 * 조건 충족 시 모든 spill_log_interface의 spill()을 호출하고 fflush().
 * 실행 컨텍스트: 시뮬레이션 루프 또는 종료 시.
 *
 * 호출 체인:
 *   gpu-sim.cc::gpgpu_sim::cycle() 또는 종료 → [spill_log_to_file()] → 각 로거의 spill()
 */
void spill_log_to_file(FILE *fout, int final,
                       unsigned long long current_cycle) {
  if (!final && spill_interval == 0) return; /* [한국어] 자동 스필 비활성이고 최종도 아니면 반환 */
  if (!final && current_cycle <= next_spill_cycle) return; /* [한국어] 스필 주기 미도달 */

  fprintf(fout, "\n");  // ensure that the spill occurs at a new line
  /* [한국어] 스필 데이터가 새 줄에서 시작하도록 개행 출력 */
  std::list<spill_log_interface *>::iterator i_spill_log =
      list_spill_log.begin();
  for (; i_spill_log != list_spill_log.end(); ++i_spill_log) {
    (*i_spill_log)->spill(fout, final); /* [한국어] 각 로거의 아카이브를 파일로 출력 */
  }
  fflush(fout); /* [한국어] 출력 버퍼 즉시 플러시 (crash 대비 데이터 보전) */

  next_spill_cycle =
      current_cycle + spill_interval;  // WF: stateful testing, maybe bad
  /* [한국어] 다음 스필 주기를 현재 사이클 + spill_interval으로 설정 */
}

////////////////////////////////////////////////////////////////////////////////

static int n_thread_CFloggers = 0;
/* [한국어] 생성된 thread_CFlocality 로거의 수. create_thread_CFlogger()에서 설정. */

static thread_CFlocality **thread_CFlogger = NULL;
/* [한국어] thread_CFlocality 로거 포인터 배열. NULL이면 시각화 출력 없음.
 * 설정자: create_thread_CFlogger(). 읽는 자: cflog_* 함수들. */

/*
 * [한국어]
 * create_thread_CFlogger() - 스레드 제어 흐름 지역성 로거 배열 생성
 *
 * @ctx: gpgpu_context 포인터 (PC→PTX라인번호 변환용)
 * @n_loggers: 생성할 로거 수 (보통 SM 수)
 * @n_threads: SM당 최대 스레드 수
 * @start_pc: 초기 PC (m_thread_pc 배열 기본값)
 * @logging_interval: 스냅샷 간격. 0이면 스냅샷 없음 (visualizer 전용 모드).
 *
 * 기존 로거가 있으면 먼저 destroy_thread_CFlogger()로 삭제.
 * 각 로거를 "CFLog00", "CFLog01" 등의 이름으로 생성.
 * logging_interval != 0이면 snap_shot_trigger와 spill_log 리스트에 등록.
 */
void create_thread_CFlogger(gpgpu_context *ctx, int n_loggers, int n_threads,
                            address_type start_pc,
                            unsigned long long logging_interval) {
  destroy_thread_CFlogger(); /* [한국어] 기존 로거가 있으면 먼저 해제 */

  n_thread_CFloggers = n_loggers;        /* [한국어] 로거 수 저장 */
  thread_CFlogger = new thread_CFlocality *[n_loggers]; /* [한국어] 포인터 배열 할당 */

  std::string name_tpl("CFLog"); /* [한국어] 로거 이름 접두어 */
  char buffer[32];               /* [한국어] 로거 번호 문자열 임시 버퍼 */
  for (int i = 0; i < n_thread_CFloggers; i++) {
    snprintf(buffer, 32, "%02d", i); /* [한국어] 2자리 영숫자 번호 생성 (예: "00", "01") */
    thread_CFlogger[i] = new thread_CFlocality(
        ctx, name_tpl + buffer, logging_interval, n_threads, start_pc);
    /* [한국어] "CFLog00" 등의 이름으로 thread_CFlocality 객체 힙 생성 */
    if (logging_interval != 0) {
      /* [한국어] 스냅샷 간격이 설정된 경우 자동화 리스트에 등록 */
      add_snap_shot_trigger(thread_CFlogger[i]); /* [한국어] 스냅샷 자동화 등록 */
      add_spill_log(thread_CFlogger[i]);          /* [한국어] 스필 자동화 등록 */
    }
  }
}

/*
 * [한국어]
 * destroy_thread_CFlogger() - 스레드 CF 로거 배열 해제
 *
 * 각 로거를 snap_shot_trigger/spill_log 리스트에서 제거 후 delete.
 * thread_CFlogger 배열 자체도 delete[]. NULL 안전 처리.
 */
void destroy_thread_CFlogger() {
  if (thread_CFlogger != NULL) {
    for (int i = 0; i < n_thread_CFloggers; i++) {
      remove_snap_shot_trigger(thread_CFlogger[i]); /* [한국어] 스냅샷 트리거 리스트에서 제거 */
      remove_spill_log(thread_CFlogger[i]);          /* [한국어] 스필 로그 리스트에서 제거 */
      delete thread_CFlogger[i];                     /* [한국어] 로거 객체 해제 */
    }
    delete[] thread_CFlogger;  /* [한국어] 포인터 배열 해제 */
    thread_CFlogger = NULL;    /* [한국어] 댕글링 포인터 방지 */
  }
}

/*
 * [한국어]
 * cflog_update_thread_pc() - 특정 로거의 스레드 PC 업데이트
 *
 * @logger_id: 업데이트할 로거 번호 (SM 번호에 대응)
 * @thread_id: 업데이트할 스레드 ID
 * @pc: 새 PC 값
 *
 * thread_CFlogger == NULL이면 시각화 출력 없음 모드 — 즉시 반환.
 * thread_id < 0이면 미할당 스레드 — 무시.
 */
void cflog_update_thread_pc(int logger_id, int thread_id, address_type pc) {
  if (thread_CFlogger == NULL) return;  // this means no visualizer output
  /* [한국어] 로거가 없으면 시각화 미사용 모드 — 무시 */
  if (thread_id < 0) return; /* [한국어] 음수 스레드 ID는 미할당 스레드 — 무시 */
  thread_CFlogger[logger_id]->update_thread_pc(thread_id, pc);
  /* [한국어] 해당 로거에 스레드 PC 업데이트 위임 */
}

// deprecated
/*
 * [한국어]
 * cflog_snapshot() - 특정 로거의 스냅샷 강제 촬영 (deprecated)
 *
 * @logger_id: 스냅샷을 촬영할 로거 번호
 * @cycle: 스냅샷 사이클 번호
 *
 * 현재는 try_snap_shot()을 통한 자동 촬영으로 대체됨. deprecated.
 */
void cflog_snapshot(int logger_id, unsigned long long cycle) {
  thread_CFlogger[logger_id]->snap_shot(cycle); /* [한국어] 해당 로거의 스냅샷 강제 호출 */
}

/*
 * [한국어]
 * cflog_print() - 모든 CF 로거의 히스토그램을 파일에 출력
 *
 * @fout: 출력 FILE*
 *
 * thread_CFlogger == NULL이면 시각화 미사용 모드 — 즉시 반환.
 */
void cflog_print(FILE *fout) {
  if (thread_CFlogger == NULL) return;  // this means no visualizer output
  for (int i = 0; i < n_thread_CFloggers; i++) {
    thread_CFlogger[i]->print_histo(fout); /* [한국어] 각 로거의 히스토그램 출력 */
  }
}

/*
 * [한국어]
 * cflog_visualizer_print() - AerialVision용 CF 지역성 희소 히스토그램 FILE* 출력
 *
 * @fout: 출력 FILE*
 */
void cflog_visualizer_print(FILE *fout) {
  if (thread_CFlogger == NULL) return;  // this means no visualizer output
  for (int i = 0; i < n_thread_CFloggers; i++) {
    thread_CFlogger[i]->print_visualizer(fout); /* [한국어] AerialVision용 출력 위임 */
  }
}

/*
 * [한국어]
 * cflog_visualizer_gzprint() - AerialVision용 CF 지역성 희소 히스토그램 gzFile 출력
 *
 * @fout: 출력 gzFile (압축 파일 스트림)
 */
void cflog_visualizer_gzprint(gzFile fout) {
  if (thread_CFlogger == NULL) return;  // this means no visualizer output
  for (int i = 0; i < n_thread_CFloggers; i++) {
    thread_CFlogger[i]->print_visualizer(fout); /* [한국어] gzFile에 AerialVision 출력 위임 */
  }
}

////////////////////////////////////////////////////////////////////////////////

int insn_warp_occ_logger::s_ids = 0;
/* [한국어] insn_warp_occ_logger 클래스의 전역 ID 카운터 초기화 (0부터 시작) */

static std::vector<insn_warp_occ_logger> iwo_logger;
/* [한국어] SM별 명령어-워프점유율 로거 벡터. insn_warp_occ_create()에서 초기화. */

/*
 * [한국어]
 * insn_warp_occ_create() - 명령어별 워프 점유율 로거 배열 생성
 *
 * @n_loggers: 생성할 로거 수 (SM 수)
 * @simd_width: SIMD 폭 (워프 크기, 보통 32)
 *
 * 기존 로거를 비우고 n_loggers 개의 insn_warp_occ_logger를 생성.
 * 각 로거에 0부터 순번 ID를 부여.
 */
void insn_warp_occ_create(int n_loggers, int simd_width) {
  iwo_logger.clear(); /* [한국어] 기존 로거 제거 */
  iwo_logger.assign(n_loggers, insn_warp_occ_logger(simd_width));
  /* [한국어] n_loggers 개의 로거를 같은 simd_width로 생성하여 벡터에 할당 */
  for (unsigned i = 0; i < iwo_logger.size(); i++) {
    iwo_logger[i].set_id(i); /* [한국어] 각 로거에 순번 ID 부여 */
  }
}

/*
 * [한국어]
 * insn_warp_occ_log() - 특정 SM 로거에 PC-워프점유율 기록
 *
 * @logger_id: SM 번호 (로거 인덱스)
 * @pc: 실행 중인 명령어의 PTX PC
 * @warp_occ: 실행 시 활성 스레드 수 (1~simd_width)
 *
 * warp_occ <= 0이면 무효한 점유율로 무시.
 */
void insn_warp_occ_log(int logger_id, address_type pc, int warp_occ) {
  if (warp_occ <= 0) return; /* [한국어] 0 이하의 점유율은 유효하지 않음 — 무시 */
  iwo_logger[logger_id].log(pc, warp_occ); /* [한국어] 해당 SM 로거에 기록 위임 */
}

/*
 * [한국어]
 * insn_warp_occ_print() - 모든 SM의 명령어별 워프 점유율 히스토그램 출력
 *
 * @fout: 출력 FILE*
 */
void insn_warp_occ_print(FILE *fout) {
  for (unsigned i = 0; i < iwo_logger.size(); i++) {
    iwo_logger[i].print(fout); /* [한국어] 각 SM 로거의 히스토그램 출력 */
  }
}

////////////////////////////////////////////////////////////////////////////////

int linear_histogram_logger::s_ids = 0;
/* [한국어] linear_histogram_logger 클래스의 전역 ID 카운터 초기화 (0부터 시작) */

/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore active thread distribution (warp occ) logger
/////////////////////////////////////////////////////////////////////////////////////

static std::vector<linear_histogram_logger> s_warp_occ_logger;
/* [한국어] SM별 워프 점유율 분포 로거 벡터. shader_warp_occ_create()에서 초기화. */

/*
 * [한국어]
 * shader_warp_occ_create() - SM별 워프 점유율 분포 로거 배열 생성
 *
 * @n_loggers: SM 수 (로거 개수)
 * @simd_width: SIMD 폭 (워프 크기, 보통 32). simd_width+1 bins로 0~32 범위 수집.
 * @logging_interval: 스냅샷 간격 사이클 수
 *
 * simd_width+1 bins: 0개부터 simd_width개까지의 활성 스레드 수를 모두 포함.
 * 각 로거를 snap_shot_trigger와 spill_log 리스트에 등록하여 자동화.
 */
void shader_warp_occ_create(int n_loggers, int simd_width,
                            unsigned long long logging_interval) {
  // simd_width + 1 to include the case with full warp
  s_warp_occ_logger.assign(
      n_loggers,
      linear_histogram_logger(simd_width + 1, logging_interval, "ShdrWarpOcc"));
  /* [한국어] simd_width+1 bins의 로거 n_loggers 개 생성. 이름 "ShdrWarpOcc". */
  for (unsigned i = 0; i < s_warp_occ_logger.size(); i++) {
    s_warp_occ_logger[i].set_id(i);                    /* [한국어] SM 번호를 로거 ID로 설정 */
    add_snap_shot_trigger(&(s_warp_occ_logger[i]));    /* [한국어] 스냅샷 자동화 등록 */
    add_spill_log(&(s_warp_occ_logger[i]));            /* [한국어] 스필 자동화 등록 */
  }
}

/*
 * [한국어]
 * shader_warp_occ_log() - SM의 현재 워프 점유율 기록
 *
 * @logger_id: SM 번호
 * @warp_occ: 현재 워프 내 활성 스레드 수 (0~simd_width)
 */
void shader_warp_occ_log(int logger_id, int warp_occ) {
  s_warp_occ_logger[logger_id].log(warp_occ); /* [한국어] warp_occ를 bin 인덱스로 기록 */
}

/*
 * [한국어]
 * shader_warp_occ_snapshot() - SM 워프 점유율 스냅샷 강제 촬영
 *
 * @logger_id: SM 번호
 * @current_cycle: 스냅샷 사이클 번호
 */
void shader_warp_occ_snapshot(int logger_id, unsigned long long current_cycle) {
  s_warp_occ_logger[logger_id].snap_shot(current_cycle);
  /* [한국어] 해당 SM 로거의 현재 버퍼를 아카이브로 이동 */
}

/*
 * [한국어]
 * shader_warp_occ_print() - 모든 SM의 워프 점유율 히스토그램 출력
 *
 * @fout: 출력 FILE*
 */
void shader_warp_occ_print(FILE *fout) {
  for (unsigned i = 0; i < s_warp_occ_logger.size(); i++) {
    s_warp_occ_logger[i].print(fout); /* [한국어] 각 SM 로거의 히스토그램 출력 */
  }
}

/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore memory-access logger
/////////////////////////////////////////////////////////////////////////////////////

static int s_mem_acc_logger_n_dram = 0;
/* [한국어] DRAM 채널(메모리 파티션) 수. shader_mem_acc_log()에서 bin 인덱스 계산에 사용. */

static int s_mem_acc_logger_n_bank = 0;
/* [한국어] DRAM당 뱅크 수. shader_mem_acc_log()에서 bin 인덱스 계산에 사용. */

static std::vector<linear_histogram_logger> s_mem_acc_logger;
/* [한국어] SM별 메모리 접근 분포 로거 벡터. shader_mem_acc_create()에서 초기화. */

/*
 * [한국어]
 * shader_mem_acc_create() - SM별 메모리 접근 분포 로거 배열 생성
 *
 * @n_loggers: SM 수
 * @n_dram: DRAM 채널(메모리 파티션) 수
 * @n_bank: DRAM당 뱅크 수
 * @logging_interval: 스냅샷 간격 사이클 수
 *
 * 히스토그램 bin 수: 2 * n_dram * (n_bank+1)
 *   - 2배: 읽기(r)와 쓰기(w) 분리
 *   - n_dram: DRAM 채널별
 *   - (n_bank+1): n_bank+1 로 간격을 두어 데이터 정렬
 * bin 인덱스 = dram_id * n_bank + bank + write_offset (r: 0, w: n_bank*n_dram)
 */
void shader_mem_acc_create(int n_loggers, int n_dram, int n_bank,
                           unsigned long long logging_interval) {
  // (n_bank + 1) to space data out; 2x to separate read and write
  s_mem_acc_logger.assign(
      n_loggers, linear_histogram_logger(2 * n_dram * (n_bank + 1),
                                         logging_interval, "ShdrMemAcc"));
  /* [한국어] 2*n_dram*(n_bank+1) bins의 로거 생성. 읽기/쓰기 × DRAM채널 × 뱅크. */

  s_mem_acc_logger_n_dram = n_dram; /* [한국어] bin 인덱스 계산용 DRAM 수 저장 */
  s_mem_acc_logger_n_bank = n_bank; /* [한국어] bin 인덱스 계산용 뱅크 수 저장 */
  for (unsigned i = 0; i < s_mem_acc_logger.size(); i++) {
    s_mem_acc_logger[i].set_id(i);                 /* [한국어] SM 번호를 로거 ID로 설정 */
    add_snap_shot_trigger(&(s_mem_acc_logger[i])); /* [한국어] 스냅샷 자동화 등록 */
    add_spill_log(&(s_mem_acc_logger[i]));         /* [한국어] 스필 자동화 등록 */
  }
}

/*
 * [한국어]
 * shader_mem_acc_log() - SM의 메모리 접근(DRAM 채널/뱅크/읽쓰기) 기록
 *
 * @logger_id: SM 번호
 * @dram_id: 접근한 DRAM 채널(파티션) 번호
 * @bank: 접근한 DRAM 뱅크 번호
 * @rw: 'r'(읽기) 또는 'w'(쓰기)
 *
 * bin 인덱스 = dram_id * n_bank + bank + write_offset
 *   write_offset: 'r'=0, 'w'=(n_bank+1)*n_dram (쓰기를 읽기 bin 뒤에 배치)
 */
void shader_mem_acc_log(int logger_id, int dram_id, int bank, char rw) {
  if (s_mem_acc_logger_n_dram == 0) return; /* [한국어] 로거가 초기화되지 않은 경우 무시 */
  int write_offset = 0;                     /* [한국어] 읽기/쓰기 분리를 위한 bin 오프셋 */
  switch (rw) {
    case 'r':
      write_offset = 0; /* [한국어] 읽기: 오프셋 없음 (하위 절반 bins) */
      break;
    case 'w':
      write_offset = (s_mem_acc_logger_n_bank + 1) * s_mem_acc_logger_n_dram;
      /* [한국어] 쓰기: 읽기 영역을 건너뛰어 상위 절반 bins에 기록 */
      break;
    default:
      assert(0); /* [한국어] 'r'/'w' 외의 값은 설계 오류 */
      break;
  }
  s_mem_acc_logger[logger_id].log(dram_id * s_mem_acc_logger_n_bank + bank +
                                  write_offset);
  /* [한국어] dram_id*n_bank+bank+write_offset으로 bin 인덱스 계산 후 기록 */
}

/*
 * [한국어]
 * shader_mem_acc_snapshot() - SM 메모리 접근 스냅샷 강제 촬영
 */
void shader_mem_acc_snapshot(int logger_id, unsigned long long current_cycle) {
  s_mem_acc_logger[logger_id].snap_shot(current_cycle);
}

/*
 * [한국어]
 * shader_mem_acc_print() - 모든 SM의 메모리 접근 히스토그램 출력
 */
void shader_mem_acc_print(FILE *fout) {
  for (unsigned i = 0; i < s_mem_acc_logger.size(); i++) {
    s_mem_acc_logger[i].print(fout);
  }
}

/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore memory-latency logger
/////////////////////////////////////////////////////////////////////////////////////

static bool s_mem_lat_logger_used = false;
/* [한국어] 메모리 레이턴시 로거 사용 여부 플래그. shader_mem_lat_create() 호출 시 true. */

static int s_mem_lat_logger_nbins = 48;  // up to 2^24 = 16M
/* [한국어] 레이턴시 히스토그램 bin 수 = 48.
 * log-sqrt2 스케일로 2 bins/octave → 24 octave = 2^24 = 16M 사이클까지 표현 가능. */

static std::vector<linear_histogram_logger> s_mem_lat_logger;
/* [한국어] SM별 메모리 레이턴시 분포 로거 벡터. shader_mem_lat_create()에서 초기화. */

/*
 * [한국어]
 * shader_mem_lat_create() - SM별 메모리 레이턴시 분포 로거 배열 생성
 *
 * @n_loggers: SM 수
 * @logging_interval: 스냅샷 간격 사이클 수
 *
 * 48 bins의 log-sqrt2 스케일 레이턴시 히스토그램 로거 생성.
 * s_mem_lat_logger_used = true로 설정하여 log 함수 활성화.
 */
void shader_mem_lat_create(int n_loggers, unsigned long long logging_interval) {
  s_mem_lat_logger.assign(
      n_loggers, linear_histogram_logger(s_mem_lat_logger_nbins,
                                         logging_interval, "ShdrMemLat"));
  /* [한국어] 48 bins의 로거 n_loggers 개 생성. 이름 "ShdrMemLat". */

  for (unsigned i = 0; i < s_mem_lat_logger.size(); i++) {
    s_mem_lat_logger[i].set_id(i);                 /* [한국어] SM 번호를 로거 ID로 설정 */
    add_snap_shot_trigger(&(s_mem_lat_logger[i])); /* [한국어] 스냅샷 자동화 등록 */
    add_spill_log(&(s_mem_lat_logger[i]));         /* [한국어] 스필 자동화 등록 */
  }

  s_mem_lat_logger_used = true; /* [한국어] 로거 초기화 완료 — log 함수 활성화 */
}

/*
 * [한국어]
 * shader_mem_lat_log() - SM의 메모리 레이턴시를 log-sqrt2 스케일로 기록
 *
 * @logger_id: SM 번호
 * @latency: 메모리 접근 레이턴시 (사이클 수, 양의 정수)
 *
 * log-sqrt2(latency) 계산으로 48 bins에 분류.
 * 알고리즘: 비트 연산으로 floor(log2(latency))를 계산 → latency_bin = 2*bin.
 *           이후 (latency & (1 << (bin-1))) != 0이면 +1 (sqrt2 보정).
 *           결과: 각 octave(log2 구간)를 2 bins로 분할하여 더 세밀한 분류.
 * 실행 컨텍스트: 캐시 미스 완료 후 레이턴시 계산 시점에 호출.
 */
void shader_mem_lat_log(int logger_id, int latency) {
  if (s_mem_lat_logger_used == false) return; /* [한국어] 로거 미초기화 시 무시 */
  if (latency > (1 << (s_mem_lat_logger_nbins / 2)))
    assert(0);  // guard for out of bound bin
  /* [한국어] latency > 2^24이면 bin 범위 초과 — 설계 오류 */
  assert(latency > 0); /* [한국어] 0 이하 레이턴시는 유효하지 않음 */

  int latency_bin; /* [한국어] 최종 계산된 log-sqrt2 스케일 bin 인덱스 */

  int bin;  // LOG_2(latency)
  int v = latency;           /* [한국어] 비트 연산용 latency 복사본 */
  register unsigned int shift; /* [한국어] 현재 단계 비트 시프트 양 */

  /* [한국어] 비트 연산으로 floor(log2(latency)) 계산 (pow2_histogram::add2bin()과 동일 알고리즘) */
  bin = (v > 0xFFFF) << 4;  /* [한국어] 16비트 초과 여부 → 4비트 시프트 */
  v >>= bin;
  shift = (v > 0xFF) << 3;  /* [한국어] 8비트 초과 여부 → 3비트 시프트 */
  v >>= shift;
  bin |= shift;
  shift = (v > 0xF) << 2;   /* [한국어] 4비트 초과 여부 → 2비트 시프트 */
  v >>= shift;
  bin |= shift;
  shift = (v > 0x3) << 1;   /* [한국어] 2비트 초과 여부 → 1비트 시프트 */
  v >>= shift;
  bin |= shift;
  bin |= (v >> 1);           /* [한국어] 최하위 비트 기여 → bin = floor(log2(latency)) */
  latency_bin = 2 * bin;     /* [한국어] log-sqrt2 스케일의 하위 bin (log2 값의 2배) */
  if (bin > 0) {
    latency_bin += ((latency & (1 << (bin - 1))) != 0)
                       ? 1
                       : 0;  // approx. for LOG_sqrt2(latency)
    /* [한국어] (bin-1)번째 비트로 sqrt2 스케일 보정: latency가 2^bin의 절반 이상이면 +1.
     * 결과: 각 log2 구간을 2개 bins로 분할하여 sqrt2(~1.41) 간격의 스케일 달성. */
  }

  s_mem_lat_logger[logger_id].log(latency_bin); /* [한국어] 계산된 bin에 레이턴시 기록 */
}

/*
 * [한국어]
 * shader_mem_lat_snapshot() - SM 메모리 레이턴시 스냅샷 강제 촬영
 */
void shader_mem_lat_snapshot(int logger_id, unsigned long long current_cycle) {
  s_mem_lat_logger[logger_id].snap_shot(current_cycle);
}

/*
 * [한국어]
 * shader_mem_lat_print() - 모든 SM의 메모리 레이턴시 히스토그램 출력
 */
void shader_mem_lat_print(FILE *fout) {
  for (unsigned i = 0; i < s_mem_lat_logger.size(); i++) {
    s_mem_lat_logger[i].print(fout);
  }
}

/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore cache-miss logger
/////////////////////////////////////////////////////////////////////////////////////

static int s_cache_access_logger_n_types = 0;
/* [한국어] 캐시 타입 수 (NORMALS/TEXTURE/CONSTANT/INSTRUCTION = 4).
 * shader_cache_access_create()에서 설정. 0이면 로거 미초기화. */

static std::vector<linear_histogram_logger> s_cache_access_logger;
/* [한국어] SM별 캐시 접근/미스 분포 로거 벡터. */

int get_shader_normal_cache_id() { return NORMALS; }       /* [한국어] 일반 L1 캐시 타입 ID (0) */
int get_shader_texture_cache_id() { return TEXTURE; }     /* [한국어] 텍스처 캐시 타입 ID (1) */
int get_shader_constant_cache_id() { return CONSTANT; }   /* [한국어] 상수 캐시 타입 ID (2) */
int get_shader_instruction_cache_id() { return INSTRUCTION; } /* [한국어] 명령어 캐시 타입 ID (3) */

/*
 * [한국어]
 * shader_cache_access_create() - SM별 캐시 접근/미스 로거 배열 생성
 *
 * @n_loggers: SM 수
 * @n_types: 캐시 타입 수 (보통 4: NORMALS, TEXTURE, CONSTANT, INSTRUCTION)
 * @logging_interval: 스냅샷 간격 사이클 수
 *
 * n_types * 2 bins: 각 타입별 (접근 카운터, 미스 카운터) 2개씩.
 * bin 인덱스 = 2*type + miss (miss=0: 히트, miss=1: 미스).
 */
void shader_cache_access_create(int n_loggers, int n_types,
                                unsigned long long logging_interval) {
  // There are different type of cache (x2 for recording accesses and misses)
  s_cache_access_logger.assign(
      n_loggers,
      linear_histogram_logger(n_types * 2, logging_interval, "ShdrCacheMiss"));
  /* [한국어] n_types*2 bins의 로거 생성. 이름 "ShdrCacheMiss". */

  s_cache_access_logger_n_types = n_types; /* [한국어] 캐시 타입 수 저장 */
  for (unsigned i = 0; i < s_cache_access_logger.size(); i++) {
    s_cache_access_logger[i].set_id(i);                    /* [한국어] SM ID 설정 */
    add_snap_shot_trigger(&(s_cache_access_logger[i]));    /* [한국어] 스냅샷 자동화 등록 */
    add_spill_log(&(s_cache_access_logger[i]));            /* [한국어] 스필 자동화 등록 */
  }
}

/*
 * [한국어]
 * shader_cache_access_log() - SM의 캐시 접근/미스 기록
 *
 * @logger_id: SM 번호
 * @type: 캐시 타입 (NORMALS=0, TEXTURE=1, CONSTANT=2, INSTRUCTION=3)
 * @miss: 0=히트, 1=미스
 *
 * bin 인덱스 = 2*type + miss.
 * s_cache_access_logger_n_types==0이면 미초기화 — 무시.
 * logger_id<0이면 유효하지 않은 SM — 무시.
 */
void shader_cache_access_log(int logger_id, int type, int miss) {
  if (s_cache_access_logger_n_types == 0) return; /* [한국어] 로거 미초기화 */
  if (logger_id < 0) return;                       /* [한국어] 유효하지 않은 SM ID */
  assert(type == NORMALS || type == TEXTURE || type == CONSTANT ||
         type == INSTRUCTION); /* [한국어] 유효한 캐시 타입 확인 */
  assert(miss == 0 || miss == 1); /* [한국어] miss는 0(히트) 또는 1(미스)만 유효 */

  s_cache_access_logger[logger_id].log(2 * type + miss);
  /* [한국어] 2*type+miss로 bin 인덱스 계산: [NORMALS히트, NORMALS미스, TEXTURE히트, ...] */
}

/*
 * [한국어]
 * shader_cache_access_unlog() - SM의 캐시 접근/미스 기록 취소
 *
 * @logger_id: SM 번호, @type: 캐시 타입, @miss: 0=히트, 1=미스
 *
 * MSHR 병합 등으로 이미 기록된 이벤트를 취소할 때 사용 (카운터 감소).
 */
void shader_cache_access_unlog(int logger_id, int type, int miss) {
  if (s_cache_access_logger_n_types == 0) return;
  if (logger_id < 0) return;
  assert(type == NORMALS || type == TEXTURE || type == CONSTANT ||
         type == INSTRUCTION);
  assert(miss == 0 || miss == 1);

  s_cache_access_logger[logger_id].unlog(2 * type + miss);
  /* [한국어] 2*type+miss 인덱스의 카운터 감소 */
}

/*
 * [한국어]
 * shader_cache_access_print() - 모든 SM의 캐시 접근/미스 히스토그램 출력
 */
void shader_cache_access_print(FILE *fout) {
  for (unsigned i = 0; i < s_cache_access_logger.size(); i++) {
    s_cache_access_logger[i].print(fout);
  }
}

/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore CTA count logger (only make sense with
// gpgpu_spread_blocks_across_cores)
/////////////////////////////////////////////////////////////////////////////////////

static linear_histogram_logger *s_CTA_count_logger = NULL;
/* [한국어] SM당 현재 할당된 CTA(Cooperative Thread Array) 수 로거.
 * gpgpu_spread_blocks_across_cores 옵션 활성 시에만 의미 있음.
 * n_shaders bins: 각 bin이 하나의 SM에 대응. */

/*
 * [한국어]
 * shader_CTA_count_create() - SM별 CTA 수 로거 생성
 *
 * @n_shaders: SM 수 (bins 수로 사용 — 각 SM이 하나의 bin)
 * @logging_interval: 스냅샷 간격. 0이면 자동 스냅샷 없음.
 *
 * 단일 로거로 모든 SM의 CTA 수를 추적. reset_at_snap_shot=false로
 * 스냅샷 후에도 현재 CTA 수 유지 (누적 방식이 아닌 현재 상태 추적).
 * ID=-1: 특수 단일 로거임을 표시.
 */
void shader_CTA_count_create(int n_shaders,
                             unsigned long long logging_interval) {
  // only need one logger to track all the shaders
  if (s_CTA_count_logger != NULL) delete s_CTA_count_logger;
  /* [한국어] 기존 로거가 있으면 먼저 해제 */
  s_CTA_count_logger = new linear_histogram_logger(n_shaders, logging_interval,
                                                   "ShdrCTACount", false);
  /* [한국어] n_shaders bins, reset_at_snap_shot=false(현재 상태 유지)로 로거 생성 */

  s_CTA_count_logger->set_id(-1); /* [한국어] ID=-1: 단일 전역 로거 표식 */
  if (logging_interval != 0) {
    add_snap_shot_trigger(s_CTA_count_logger); /* [한국어] 스냅샷 자동화 등록 */
    add_spill_log(s_CTA_count_logger);          /* [한국어] 스필 자동화 등록 */
  }
}

/*
 * [한국어]
 * shader_CTA_count_log() - SM에 CTA 추가 기록
 *
 * @shader_id: CTA가 추가된 SM 번호 (bin 인덱스로 사용)
 * @nCTAadded: 추가된 CTA 수
 *
 * nCTAadded번 만큼 log(shader_id) 호출하여 SM bin 카운터 증가.
 */
void shader_CTA_count_log(int shader_id, int nCTAadded) {
  if (s_CTA_count_logger == NULL) return; /* [한국어] 로거 미초기화 — 무시 */

  for (int i = 0; i < nCTAadded; i++) {
    s_CTA_count_logger->log(shader_id); /* [한국어] shader_id bin 카운터 증가 */
  }
}

/*
 * [한국어]
 * shader_CTA_count_unlog() - SM에서 CTA 완료 기록 (카운터 감소)
 *
 * @shader_id: CTA가 완료된 SM 번호
 * @nCTAdone: 완료된 CTA 수
 */
void shader_CTA_count_unlog(int shader_id, int nCTAdone) {
  if (s_CTA_count_logger == NULL) return; /* [한국어] 로거 미초기화 — 무시 */

  for (int i = 0; i < nCTAdone; i++) {
    s_CTA_count_logger->unlog(shader_id); /* [한국어] shader_id bin 카운터 감소 */
  }
}

/*
 * [한국어]
 * shader_CTA_count_print() - CTA 수 히스토그램을 파일에 출력
 */
void shader_CTA_count_print(FILE *fout) {
  if (s_CTA_count_logger == NULL) return;
  s_CTA_count_logger->print(fout);
}

/*
 * [한국어]
 * shader_CTA_count_visualizer_print() - AerialVision용 CTA 분포를 FILE*에 출력
 */
void shader_CTA_count_visualizer_print(FILE *fout) {
  if (s_CTA_count_logger == NULL) return;
  s_CTA_count_logger->print_visualizer(fout);
}

/*
 * [한국어]
 * shader_CTA_count_visualizer_gzprint() - AerialVision용 CTA 분포를 gzFile에 출력
 */
void shader_CTA_count_visualizer_gzprint(gzFile fout) {
  if (s_CTA_count_logger == NULL) return;
  s_CTA_count_logger->print_visualizer(fout);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*
 * [한국어]
 * thread_insn_span::thread_insn_span() - 특정 사이클에서 시작하는 빈 span 생성
 *
 * @cycle: 이 span이 시작되는 시뮬레이션 사이클
 * @ctx: PC→PTX 라인번호 변환용 gpgpu_context 포인터
 *
 * tr1_hash_map_ismap == 1이면 std::map 기반, 아니면 해시맵(초기 bucket 32K) 사용.
 * 해시맵 초기 크기 32*1024는 SM당 예상되는 PC 수를 고려한 최적화.
 */
thread_insn_span::thread_insn_span(unsigned long long cycle, gpgpu_context *ctx)
    : m_cycle(cycle),
#if (tr1_hash_map_ismap == 1)
      m_insn_span_count()        /* [한국어] std::map 기반 span_count_map 빈 생성 */
#else
      m_insn_span_count(32 * 1024) /* [한국어] 해시맵 초기 bucket 32K로 충돌 최소화 */
#endif
{
  gpgpu_ctx = ctx; /* [한국어] PC→PTX 라인번호 변환용 컨텍스트 저장 */
}

thread_insn_span::~thread_insn_span() {} /* [한국어] trivial 소멸자 */

/*
 * [한국어]
 * thread_insn_span(복사 생성자) - other의 span 데이터를 새 ctx로 복사
 *
 * @other: 복사 원본 span
 * @ctx: 새 span의 gpgpu_context 포인터
 *
 * m_cycle과 m_insn_span_count를 other에서 복사. gpgpu_ctx는 새 ctx 사용.
 */
thread_insn_span::thread_insn_span(const thread_insn_span &other,
                                   gpgpu_context *ctx)
    : m_cycle(other.m_cycle), m_insn_span_count(other.m_insn_span_count) {
  gpgpu_ctx = ctx; /* [한국어] 복사된 span에 새 컨텍스트 포인터 설정 */
}

/*
 * [한국어]
 * thread_insn_span::operator=() - span 대입 연산자
 *
 * @other: 복사 원본 span
 * @return: *this
 *
 * 자기 대입 체크 후 m_insn_span_count와 m_cycle 복사.
 * 진단 목적으로 printf("thread_insn_span& operator=\n") 출력.
 */
thread_insn_span &thread_insn_span::operator=(const thread_insn_span &other) {
  printf("thread_insn_span& operator=\n"); /* [한국어] 대입 연산자 호출 진단 출력 */
  if (this != &other) { /* [한국어] 자기 대입 방지 */
    m_insn_span_count = other.m_insn_span_count; /* [한국어] span 카운트 맵 복사 */
    m_cycle = other.m_cycle;                     /* [한국어] 사이클 번호 복사 */
  }
  return *this;
}

/*
 * [한국어]
 * thread_insn_span::operator+=() - 두 span의 PC 카운터 합산
 *
 * @other: 합산할 span 객체
 * @return: *this (합산 결과)
 *
 * other의 각 PC에 대한 카운터를 현재 span에 누적.
 * 여러 스냅샷 구간을 집계할 때 사용.
 */
thread_insn_span &thread_insn_span::operator+=(const thread_insn_span &other) {
  span_count_map::const_iterator i_sc = other.m_insn_span_count.begin();
  for (; i_sc != other.m_insn_span_count.end(); ++i_sc) {
    m_insn_span_count[i_sc->first] += i_sc->second;
    /* [한국어] other의 각 PC 카운터를 현재 span에 누적 */
  }
  return *this;
}

/*
 * [한국어]
 * thread_insn_span::set_span() - 지정 PC를 span에 추가 (카운터 증가)
 *
 * @pc: 추가할 명령어 PC 주소
 *
 * ((int)pc) >= 0 조건: 미할당 스레드의 PC는 (address_type)(-1)로 설정됨.
 * 캐스팅하면 -1이 되므로 음수 검사로 미할당 스레드 필터링.
 */
void thread_insn_span::set_span(address_type pc) {
  if (((int)pc) >= 0) m_insn_span_count[pc] += 1;
  /* [한국어] 유효한 PC(>=0)만 기록. (address_type)(-1)은 미할당 스레드 표식 — 무시. */
}

/*
 * [한국어]
 * thread_insn_span::reset() - span 초기화
 *
 * @cycle: 새 span의 시작 사이클 번호
 *
 * m_cycle 갱신 후 m_insn_span_count 비움.
 * snap_shot() 후 새 인터벌 시작 시 호출.
 */
void thread_insn_span::reset(unsigned long long cycle) {
  m_cycle = cycle;              /* [한국어] 새 사이클 번호 설정 */
  m_insn_span_count.clear();   /* [한국어] 모든 PC 카운터 제거 */
}

/*
 * [한국어]
 * thread_insn_span::print_span() - PC 주소 목록을 파일에 출력 (진단용)
 *
 * @fout: 출력 FILE*
 *
 * 형식: "<cycle>: <pc0_hex> <pc1_hex> ..."
 */
void thread_insn_span::print_span(FILE *fout) const {
  fprintf(fout, "%d: ", (int)m_cycle); /* [한국어] 사이클 번호 출력 */
  span_count_map::const_iterator i_sc = m_insn_span_count.begin();
  for (; i_sc != m_insn_span_count.end(); ++i_sc) {
    fprintf(fout, "%llx ", i_sc->first); /* [한국어] 각 PC를 16진수로 출력 */
  }
  fprintf(fout, "\n");
}

/*
 * [한국어]
 * thread_insn_span::print_histo() - PC별 실행 횟수 히스토그램을 파일에 출력
 *
 * @fout: 출력 FILE*
 *
 * 형식: "<cycle>:<count0> <count1> ..."
 */
void thread_insn_span::print_histo(FILE *fout) const {
  fprintf(fout, "%d:", (int)m_cycle); /* [한국어] 사이클 번호 출력 */
  span_count_map::const_iterator i_sc = m_insn_span_count.begin();
  for (; i_sc != m_insn_span_count.end(); ++i_sc) {
    fprintf(fout, "%d ", i_sc->second); /* [한국어] 각 PC의 실행 횟수 출력 */
  }
  fprintf(fout, "\n");
}

/*
 * [한국어]
 * thread_insn_span::print_sparse_histo(FILE*) - PTX 라인번호 기반 희소 히스토그램을 FILE*에 출력
 *
 * @fout: 출력 FILE*
 *
 * AerialVision용 출력. PC를 PTX 소스 라인번호로 변환하여 출력.
 * 형식: "<ptx_line> <count> <ptx_line> <count> ..."
 * 항목이 없으면 "0 0 " 출력 (AerialVision이 빈 항목을 기대하지 않음).
 */
void thread_insn_span::print_sparse_histo(FILE *fout) const {
  int n_printed_entries = 0;
  span_count_map::const_iterator i_sc = m_insn_span_count.begin();
  for (; i_sc != m_insn_span_count.end(); ++i_sc) {
    unsigned ptx_lineno = gpgpu_ctx->translate_pc_to_ptxlineno(i_sc->first);
    /* [한국어] PC를 PTX 소스 라인번호로 변환 — AerialVision이 이 정보를 사용 */
    fprintf(fout, "%u %d ", ptx_lineno, i_sc->second);
    /* [한국어] "라인번호 실행횟수 " 형식으로 출력 */
    n_printed_entries++;
  }
  if (n_printed_entries == 0) {
    fprintf(fout, "0 0 "); /* [한국어] 실행된 명령어 없음을 나타내는 더미 항목 */
  }
  fprintf(fout, "\n");
}

/*
 * [한국어]
 * thread_insn_span::print_sparse_histo(gzFile) - PTX 라인번호 기반 희소 히스토그램을 gzFile에 출력
 *
 * @fout: 출력 gzFile (AerialVision용 압축 시각화 파일)
 *
 * FILE* 버전과 동일하지만 gzprintf를 사용하여 압축 스트림에 출력.
 */
void thread_insn_span::print_sparse_histo(gzFile fout) const {
  int n_printed_entries = 0;
  span_count_map::const_iterator i_sc = m_insn_span_count.begin();
  for (; i_sc != m_insn_span_count.end(); ++i_sc) {
    unsigned ptx_lineno = gpgpu_ctx->translate_pc_to_ptxlineno(i_sc->first);
    /* [한국어] PC를 PTX 소스 라인번호로 변환 */
    gzprintf(fout, "%u %d ", ptx_lineno, i_sc->second);
    /* [한국어] 압축 스트림에 "라인번호 실행횟수 " 출력 */
    n_printed_entries++;
  }
  if (n_printed_entries == 0) {
    gzprintf(fout, "0 0 "); /* [한국어] 빈 항목 더미 출력 */
  }
  gzprintf(fout, "\n");
}

////////////////////////////////////////////////////////////////////////////////

/*
 * [한국어]
 * thread_CFlocality::thread_CFlocality() - 스레드 CF 지역성 로거 생성자
 *
 * @ctx: gpgpu_context 포인터 (PC→PTX라인번호 변환용)
 * @name: 로거 이름 (예: "CFLog00")
 * @snap_shot_interval: 스냅샷 간격 사이클 수
 * @nthreads: SM당 최대 스레드 수 (m_thread_pc 배열 크기)
 * @start_pc: 초기 PC (m_thread_pc 기본값 — 실제로는 -1로 재설정)
 * @start_cycle: 시작 사이클 (기본 0)
 *
 * m_thread_pc를 -1로 초기화하여 미할당 스레드가 span에 영향 주지 않도록 함.
 * (address_type)(-1)은 set_span()에서 무시됨.
 */
thread_CFlocality::thread_CFlocality(gpgpu_context *ctx, std::string name,
                                     unsigned long long snap_shot_interval,
                                     int nthreads, address_type start_pc,
                                     unsigned long long start_cycle)
    : snap_shot_trigger(snap_shot_interval), /* [한국어] 스냅샷 트리거 기반 클래스 초기화 */
      m_name(name),                           /* [한국어] 로거 이름 초기화 */
      m_nthreads(nthreads),                   /* [한국어] 스레드 수 초기화 */
      m_thread_pc(nthreads, start_pc),        /* [한국어] 스레드 PC 배열 start_pc로 초기화 */
      m_cycle(start_cycle),                   /* [한국어] 시작 사이클 설정 */
      m_thd_span(start_cycle, ctx) {          /* [한국어] 현재 span 초기화 */
  std::fill(
      m_thread_pc.begin(), m_thread_pc.end(),
      -1);  // so that hw thread with no work assigned will not clobber results
  /* [한국어] 모든 스레드 PC를 -1로 설정. 미할당 스레드가 span에 영향 주지 않도록. */
}

thread_CFlocality::~thread_CFlocality() {} /* [한국어] trivial 소멸자 */

/*
 * [한국어]
 * thread_CFlocality::update_thread_pc() - 스레드 PC 업데이트 및 span에 기록
 *
 * @thread_id: 업데이트할 하드웨어 스레드 ID
 * @pc: 새 PC 값
 *
 * m_thread_pc[thread_id] = pc로 현재 PC 갱신.
 * m_thd_span.set_span(pc)으로 현재 span에 이 PC 실행 기록.
 */
void thread_CFlocality::update_thread_pc(int thread_id, address_type pc) {
  m_thread_pc[thread_id] = pc;  /* [한국어] 해당 스레드의 현재 PC 갱신 */
  m_thd_span.set_span(pc);       /* [한국어] 현재 span에 이 PC의 실행 기록 추가 */
}

/*
 * [한국어]
 * thread_CFlocality::snap_shot() - 현재 span을 아카이브에 저장하고 새 span 시작
 *
 * @current_cycle: 스냅샷 촬영 사이클 번호
 *
 * 1. 현재 m_thd_span을 m_thd_span_archive에 push_back
 * 2. m_thd_span을 current_cycle로 reset (카운터 비움)
 * 3. m_thread_pc[] 전체를 새 span의 초기 상태로 set_span
 *    (현재 PC에서 새 인터벌 시작)
 */
void thread_CFlocality::snap_shot(unsigned long long current_cycle) {
  m_thd_span_archive.push_back(m_thd_span); /* [한국어] 현재 span을 아카이브에 이동 */
  m_thd_span.reset(current_cycle);           /* [한국어] 새 인터벌 시작 — 카운터 초기화 */
  for (int i = 0; i < (int)m_thread_pc.size(); i++) {
    m_thd_span.set_span(m_thread_pc[i]);
    /* [한국어] 현재 모든 스레드의 PC를 새 span의 초기 상태로 설정 */
  }
}

/*
 * [한국어]
 * thread_CFlocality::spill() - 아카이브된 span 데이터를 파일에 출력하고 메모리 해제
 *
 * @fout: 출력 FILE*
 * @final: true이면 현재 진행 중인 m_thd_span도 출력
 *
 * 아카이브 리스트를 앞에서부터 순회하며 출력 후 erase.
 * final=true이면 현재 버퍼도 출력 (시뮬레이션 최종 결과).
 * 출력 형식: "<name>-<cycle>:<count0> <count1> ..."
 */
void thread_CFlocality::spill(FILE *fout, bool final) {
  std::list<thread_insn_span>::iterator lit = m_thd_span_archive.begin();
  for (; lit != m_thd_span_archive.end(); lit = m_thd_span_archive.erase(lit)) {
    /* [한국어] 아카이브 항목을 순회하며 출력 후 즉시 삭제 (메모리 해제) */
    fprintf(fout, "%s-", m_name.c_str()); /* [한국어] 로거 이름 접두어 출력 */
    lit->print_histo(fout);               /* [한국어] span 히스토그램 출력 */
  }
  assert(m_thd_span_archive.empty()); /* [한국어] 아카이브가 모두 비워졌는지 확인 */
  if (final) {
    fprintf(fout, "%s-", m_name.c_str()); /* [한국어] 현재 span 이름 출력 */
    m_thd_span.print_histo(fout);          /* [한국어] 현재 진행 중인 span 출력 */
  }
}

/*
 * [한국어]
 * thread_CFlocality::print_visualizer(FILE*) - AerialVision용 희소 히스토그램 출력
 *
 * @fout: 출력 FILE*
 *
 * AerialVision은 스냅샷 없이 현재 버퍼만 사용. 아카이브가 있으면 assert(0).
 * 출력 후 m_thd_span 리셋 및 현재 스레드 PC로 재초기화.
 */
void thread_CFlocality::print_visualizer(FILE *fout) {
  fprintf(fout, "%s: ", m_name.c_str()); /* [한국어] 로거 이름 출력 */
  if (m_thd_span_archive.empty()) {
    // visualizer do no require snap_shots
    m_thd_span.print_sparse_histo(fout); /* [한국어] 현재 span을 PTX 라인번호로 변환해 출력 */

    // clean the thread span
    m_thd_span.reset(0); /* [한국어] AerialVision 출력 후 span 초기화 (사이클=0) */
    for (int i = 0; i < (int)m_thread_pc.size(); i++)
      m_thd_span.set_span(m_thread_pc[i]);
    /* [한국어] 현재 스레드 PC들로 새 span 초기화 (다음 visualizer 출력 준비) */
  } else {
    assert(0);  // TODO: implement fall back so that visualizer can work with
                // snap shots
    /* [한국어] 아카이브가 있는 경우 visualizer 지원 미구현 — 향후 구현 필요 */
  }
}

/*
 * [한국어]
 * thread_CFlocality::print_visualizer(gzFile) - AerialVision용 희소 히스토그램 gzFile 출력
 *
 * @fout: 출력 gzFile
 *
 * FILE* 버전과 동일하지만 gzprintf/gzFile 사용.
 */
void thread_CFlocality::print_visualizer(gzFile fout) {
  gzprintf(fout, "%s: ", m_name.c_str()); /* [한국어] 로거 이름 출력 */
  if (m_thd_span_archive.empty()) {
    // visualizer do no require snap_shots
    m_thd_span.print_sparse_histo(fout); /* [한국어] 희소 히스토그램을 압축 스트림에 출력 */

    // clean the thread span
    m_thd_span.reset(0); /* [한국어] span 초기화 */
    for (int i = 0; i < (int)m_thread_pc.size(); i++) {
      m_thd_span.set_span(m_thread_pc[i]); /* [한국어] 현재 PC로 재초기화 */
    }
  } else {
    assert(0);  // TODO: implement fall back so that visualizer can work with
                // snap shots
  }
}

/*
 * [한국어]
 * thread_CFlocality::print_span() - 전체 아카이브의 PC 주소 목록 출력
 *
 * @fout: 출력 FILE*
 *
 * 아카이브 모든 span + 현재 span의 PC 주소 목록 출력.
 */
void thread_CFlocality::print_span(FILE *fout) const {
  std::list<thread_insn_span>::const_iterator lit = m_thd_span_archive.begin();
  for (; lit != m_thd_span_archive.end(); ++lit) {
    fprintf(fout, "%s-", m_name.c_str()); /* [한국어] 로거 이름 접두어 */
    lit->print_span(fout);                 /* [한국어] 아카이브 항목의 PC 주소 목록 출력 */
  }
  fprintf(fout, "%s-", m_name.c_str()); /* [한국어] 현재 span 이름 */
  m_thd_span.print_span(fout);           /* [한국어] 현재 span의 PC 주소 목록 출력 */
}

/*
 * [한국어]
 * thread_CFlocality::print_histo() - 전체 아카이브의 PC별 실행 횟수 히스토그램 출력
 *
 * @fout: 출력 FILE*
 *
 * 아카이브 모든 span + 현재 span의 히스토그램 출력.
 */
void thread_CFlocality::print_histo(FILE *fout) const {
  std::list<thread_insn_span>::const_iterator lit = m_thd_span_archive.begin();
  for (; lit != m_thd_span_archive.end(); ++lit) {
    fprintf(fout, "%s-", m_name.c_str()); /* [한국어] 로거 이름 접두어 */
    lit->print_histo(fout);                /* [한국어] 아카이브 항목 히스토그램 출력 */
  }
  fprintf(fout, "%s-", m_name.c_str()); /* [한국어] 현재 span 이름 */
  m_thd_span.print_histo(fout);          /* [한국어] 현재 span 히스토그램 출력 */
}

////////////////////////////////////////////////////////////////////////////////

/*
 * [한국어]
 * linear_histogram_logger::linear_histogram_logger() - 선형 히스토그램 로거 생성자
 *
 * @n_bins: 히스토그램 bin 수
 * @snap_shot_interval: 스냅샷 간격 사이클 수
 * @name: 로거 이름 (출력 접두어)
 * @reset_at_snap_shot: true=스냅샷 후 리셋, false=누적 유지 (기본 true)
 * @start_cycle: 시작 사이클 (기본 0)
 */
linear_histogram_logger::linear_histogram_logger(
    int n_bins, unsigned long long snap_shot_interval, const char *name,
    bool reset_at_snap_shot, unsigned long long start_cycle)
    : snap_shot_trigger(snap_shot_interval), /* [한국어] 스냅샷 트리거 초기화 */
      m_n_bins(n_bins),                       /* [한국어] bin 수 저장 */
      m_curr_lin_hist(m_n_bins, start_cycle), /* [한국어] 현재 스냅샷 버퍼 초기화 */
      m_lin_hist_archive(),                   /* [한국어] 빈 아카이브 리스트 */
      m_cycle(start_cycle),                   /* [한국어] 시작 사이클 */
      m_reset_at_snap_shot(reset_at_snap_shot), /* [한국어] 리셋 모드 설정 */
      m_name(name),                           /* [한국어] 이름 저장 */
      m_id(s_ids++) {}                        /* [한국어] 전역 ID 카운터에서 ID 할당 */

/*
 * [한국어]
 * linear_histogram_logger(복사 생성자) - 같은 설정의 빈 로거 생성
 *
 * @other: 복사 원본 로거
 *
 * other의 설정을 복사하되 m_curr_lin_hist는 0으로 초기화, m_lin_hist_archive는 비움.
 * vector의 assign()으로 많은 로거를 생성할 때 호출됨.
 */
linear_histogram_logger::linear_histogram_logger(
    const linear_histogram_logger &other)
    : snap_shot_trigger(other.get_interval()), /* [한국어] 스냅샷 간격 복사 */
      m_n_bins(other.m_n_bins),                 /* [한국어] bin 수 복사 */
      m_curr_lin_hist(m_n_bins, other.m_cycle), /* [한국어] 새 빈 히스토그램 스냅샷 (0으로 초기화) */
      m_lin_hist_archive(),                     /* [한국어] 빈 아카이브 (복사 안 함) */
      m_cycle(other.m_cycle),                   /* [한국어] 사이클 복사 */
      m_reset_at_snap_shot(other.m_reset_at_snap_shot), /* [한국어] 리셋 모드 복사 */
      m_name(other.m_name),                     /* [한국어] 이름 복사 */
      m_id(s_ids++) {}                          /* [한국어] 새 ID 할당 */

/*
 * [한국어]
 * linear_histogram_logger::~linear_histogram_logger() - 소멸자
 *
 * snap_shot_trigger와 spill_log 전역 리스트에서 this를 제거.
 * 소멸 시 자동으로 전역 리스트에서 제거됨 — 메모리 안전.
 */
linear_histogram_logger::~linear_histogram_logger() {
  remove_snap_shot_trigger(this); /* [한국어] 전역 스냅샷 트리거 리스트에서 제거 */
  remove_spill_log(this);         /* [한국어] 전역 스필 로그 리스트에서 제거 */
}

/*
 * [한국어]
 * linear_histogram_logger::snap_shot() - 현재 히스토그램을 아카이브에 저장
 *
 * @current_cycle: 스냅샷 촬영 사이클
 *
 * m_curr_lin_hist를 m_lin_hist_archive에 push_back(복사).
 * m_reset_at_snap_shot=true이면 reset() (카운터 0으로 초기화).
 * m_reset_at_snap_shot=false이면 set_cycle() (사이클만 갱신, 카운터 유지).
 */
void linear_histogram_logger::snap_shot(unsigned long long current_cycle) {
  m_lin_hist_archive.push_back(m_curr_lin_hist); /* [한국어] 현재 버퍼 아카이브로 복사 */
  if (m_reset_at_snap_shot) {
    m_curr_lin_hist.reset(current_cycle); /* [한국어] 새 인터벌 — 카운터 0으로 초기화 */
  } else {
    m_curr_lin_hist.set_cycle(current_cycle); /* [한국어] 사이클만 갱신, 카운터 유지 (누적 모드) */
  }
}

/*
 * [한국어]
 * linear_histogram_logger::spill() - 아카이브를 파일에 출력하고 메모리 해제
 *
 * @fout: 출력 FILE*, @final: true=현재 버퍼도 출력
 *
 * 아카이브를 앞에서부터 순회하며 출력 후 erase.
 * 형식: "<name><id_2digit>-<cycle> = <bin0> <bin1> ..."
 * m_id >= 0이면 2자리 ID, -1이면 0 출력 (단일 전역 로거용).
 */
void linear_histogram_logger::spill(FILE *fout, bool final) {
  std::list<linear_histogram_snapshot>::iterator iter =
      m_lin_hist_archive.begin();
  for (; iter != m_lin_hist_archive.end();
       iter = m_lin_hist_archive.erase(iter)) {
    /* [한국어] 아카이브 순회: 출력 후 즉시 삭제로 메모리 점진적 해제 */
    fprintf(fout, "%s%02d-", m_name.c_str(), (m_id >= 0) ? m_id : 0);
    /* [한국어] "이름ID-" 형식 접두어. ID=-1이면 00 출력. */
    iter->print(fout); /* [한국어] 스냅샷 히스토그램 출력 */
    fprintf(fout, "\n");
  }
  assert(m_lin_hist_archive.empty()); /* [한국어] 모든 아카이브가 비워졌는지 확인 */
  if (final) {
    fprintf(fout, "%s%02d-", m_name.c_str(), (m_id >= 0) ? m_id : 0);
    m_curr_lin_hist.print(fout); /* [한국어] 현재 진행 중인 버퍼 출력 */
    fprintf(fout, "\n");
  }
}

/*
 * [한국어]
 * linear_histogram_logger::print() - 전체 아카이브 + 현재 버퍼 출력
 *
 * @fout: 출력 FILE*
 *
 * spill()과 달리 아카이브를 삭제하지 않음 (const-like 동작).
 * 시뮬레이션 종료 후 최종 결과 출력에 사용.
 */
void linear_histogram_logger::print(FILE *fout) const {
  std::list<linear_histogram_snapshot>::const_iterator iter =
      m_lin_hist_archive.begin();
  for (; iter != m_lin_hist_archive.end(); ++iter) {
    fprintf(fout, "%s%02d-", m_name.c_str(), m_id); /* [한국어] 이름ID- 접두어 */
    iter->print(fout);  /* [한국어] 아카이브 스냅샷 출력 */
    fprintf(fout, "\n");
  }
  fprintf(fout, "%s%02d-", m_name.c_str(), m_id); /* [한국어] 현재 버퍼 이름ID- 접두어 */
  m_curr_lin_hist.print(fout);  /* [한국어] 현재 진행 중인 히스토그램 출력 */
  fprintf(fout, "\n");
}

/*
 * [한국어]
 * linear_histogram_logger::print_visualizer(FILE*) - AerialVision용 현재 버퍼 출력
 *
 * @fout: 출력 FILE*
 *
 * 아카이브 있으면 assert(0) — 현재 스냅샷 기반 visualizer 미지원.
 * 형식: "<name><id>: <bin0> <bin1> ..."
 * 출력 후 m_reset_at_snap_shot=true이면 m_curr_lin_hist 리셋.
 */
void linear_histogram_logger::print_visualizer(FILE *fout) {
  assert(m_lin_hist_archive.empty());  // don't support snapshot for now
  /* [한국어] 아카이브가 있는 경우 visualizer 출력 미지원 */
  fprintf(fout, "%s", m_name.c_str()); /* [한국어] 로거 이름 출력 */
  if (m_id >= 0) {
    fprintf(fout, "%02d: ", m_id); /* [한국어] 2자리 ID 출력 */
  } else {
    fprintf(fout, ": "); /* [한국어] ID=-1인 경우(단일 로거) 콜론만 출력 */
  }
  m_curr_lin_hist.print_visualizer(fout); /* [한국어] bin 카운터만 출력 (사이클 번호 제외) */
  fprintf(fout, "\n");
  if (m_reset_at_snap_shot) {
    m_curr_lin_hist.reset(0); /* [한국어] visualizer 출력 후 버퍼 초기화 (다음 인터벌 준비) */
  }
}

/*
 * [한국어]
 * linear_histogram_logger::print_visualizer(gzFile) - AerialVision용 현재 버퍼 gzFile 출력
 *
 * @fout: 출력 gzFile
 *
 * FILE* 버전과 동일하지만 gzprintf 사용.
 */
void linear_histogram_logger::print_visualizer(gzFile fout) {
  assert(m_lin_hist_archive.empty());  // don't support snapshot for now
  gzprintf(fout, "%s", m_name.c_str()); /* [한국어] 로거 이름 압축 스트림에 출력 */
  if (m_id >= 0) {
    gzprintf(fout, "%02d: ", m_id); /* [한국어] 2자리 ID 출력 */
  } else {
    gzprintf(fout, ": "); /* [한국어] 단일 로거의 경우 콜론만 출력 */
  }
  m_curr_lin_hist.print_visualizer(fout); /* [한국어] bin 카운터를 압축 스트림에 출력 */
  gzprintf(fout, "\n");
  if (m_reset_at_snap_shot) {
    m_curr_lin_hist.reset(0); /* [한국어] 출력 후 버퍼 초기화 */
  }
}
