// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
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
 * [한국어 설명] GPGPU-Sim 통계 도구 헤더 (stat-tool.h)
 *
 * === 파일의 역할 ===
 * GPGPU-Sim의 시간 기반 통계 수집 인프라를 정의한다. 크게 세 계층으로 구성된다:
 * (1) snap_shot_trigger: 일정 사이클 간격으로 스냅샷을 자동 촬영하는 추상 트리거
 * (2) spill_log_interface: 메모리 과다 사용 방지를 위해 로그를 파일로 내보내는 인터페이스
 * (3) 구체 로거 클래스들: thread_CFlocality(스레드 제어 흐름 지역성), insn_warp_occ_logger
 *     (명령어별 워프 점유율), linear_histogram_logger(범용 선형 히스토그램 로거)
 * 또한 SM별 워프 점유율, 메모리 접근, 레이턴시, 캐시 미스, CTA 수를 수집하는
 * 전역 C 스타일 함수 인터페이스를 제공한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름: gpu-sim.cc의 cycle() → try_snap_shot() → 각 로거의 snap_shot()
 *            → 인터벌마다 스냅샷 아카이브에 저장 → spill_log_to_file()로 파일 출력
 * AerialVision(시각화 도구)이 이 로거들의 print_visualizer() 출력을 소비.
 * 실행 컨텍스트: 호스트 유저스페이스, 시뮬레이션 메인 루프 내.
 *
 * === 타 모듈과의 연결 ===
 * 의존: abstract_hardware_model.h (address_type), tr1_hash_map.h (해시맵),
 *       histogram.h (linear_histogram), stdio.h (FILE*), zlib.h (gzFile)
 * 사용처: gpu-sim.cc (try_snap_shot, spill_log_to_file),
 *          shader.cc (shader_warp_occ_log, shader_cache_access_log 등),
 *          visualizer.cc (cflog_visualizer_gzprint, shader_CTA_count_visualizer_gzprint)
 * 데이터 흐름: 각 로거 → snap_shot_trigger 리스트 → try_snap_shot() → 스냅샷 아카이브
 *   → spill_log_to_file() → 통계 파일 출력
 *
 * === 주요 함수/구조체 요약 ===
 * snap_shot_trigger       - 스냅샷 자동화를 위한 추상 기반 클래스
 * spill_log_interface     - 로그 파일 내보내기 추상 인터페이스
 * thread_CFlocality       - 스레드 제어 흐름 지역성 로거
 * linear_histogram_logger - 선형 히스토그램 스냅샷 로거
 * try_snap_shot()         - 모든 snap_shot_trigger를 순회하여 스냅샷 촬영
 * spill_log_to_file()     - 모든 spill_log를 파일로 내보냄
 */

#ifndef STAT_TOOL_H
#define STAT_TOOL_H

#include "../abstract_hardware_model.h" /* [한국어] address_type, addr_t 등 하드웨어 추상 타입 */
#include "../tr1_hash_map.h"            /* [한국어] tr1_hash_map (std::unordered_map 호환) */
#include "histogram.h"                  /* [한국어] linear_histogram, pow2_histogram 클래스 */

#include <stdio.h> /* [한국어] FILE*, fprintf 등 파일 출력용 */
#include <zlib.h>  /* [한국어] gzFile, gzprintf — AerialVision 압축 출력용 */

class gpgpu_context; /* [한국어] GPGPU-Sim 전역 컨텍스트 클래스 전방 선언 */

/////////////////////////////////////////////////////////////////////////////////////
// logger snapshot trigger:
// - automate the snap_shot part of loggers to avoid modifying simulation loop
// everytime
//   a new time-dependent stat is added
/////////////////////////////////////////////////////////////////////////////////////

/*
 * [한국어]
 * snap_shot_trigger - 스냅샷 자동화를 위한 추상 기반 클래스
 *
 * 일정 사이클 간격(m_snap_shot_interval)마다 snap_shot()을 자동 호출하여
 * 시뮬레이션 루프를 수정하지 않고도 새 시간 기반 통계를 추가할 수 있다.
 * try_snap_shot()을 통해 사이클 조건을 검사하고, 조건 충족 시 순수 가상 함수
 * snap_shot()을 호출한다. 구체 로거 클래스는 이를 상속하여 snap_shot()을 구현한다.
 */
class snap_shot_trigger {
 public:
  /*
   * [한국어]
   * snap_shot_trigger() - 스냅샷 트리거 생성자
   *
   * @interval: 스냅샷 촬영 사이클 간격 (매 interval 사이클마다 snap_shot() 호출)
   */
  snap_shot_trigger(unsigned long long interval)
      : m_snap_shot_interval(interval) {}
  virtual ~snap_shot_trigger() {} /* [한국어] 가상 소멸자 — 서브클래스 소멸자 정확히 호출 */

  /*
   * [한국어]
   * try_snap_shot() - 현재 사이클이 스냅샷 주기에 해당하면 snap_shot() 호출
   *
   * @current_cycle: 현재 시뮬레이션 사이클 번호
   *
   * (current_cycle % m_snap_shot_interval == 0) && current_cycle != 0 조건 검사.
   * cycle=0에서는 촬영하지 않아 시뮬레이션 시작 직후 빈 스냅샷 방지.
   * stat-tool.cc의 전역 try_snap_shot() 함수가 이를 호출.
   */
  void try_snap_shot(unsigned long long current_cycle) {
    if ((current_cycle % m_snap_shot_interval == 0) && current_cycle != 0) {
      /* [한국어] 스냅샷 주기에 해당하고 사이클 0이 아닌 경우에만 촬영 */
      snap_shot(current_cycle);
    }
  }

  virtual void snap_shot(unsigned long long current_cycle) = 0;
  /* [한국어] 순수 가상 함수 — 서브클래스(thread_CFlocality, linear_histogram_logger)가 구현.
   * 현재까지 누적된 통계 데이터를 아카이브에 저장하고 현재 버퍼를 리셋. */

  /*
   * [한국어]
   * get_interval() - 스냅샷 촬영 간격 반환
   *
   * @return: m_snap_shot_interval — add_snap_shot_trigger()에서 최소 간격 계산에 사용
   */
  const unsigned long long &get_interval() const {
    return m_snap_shot_interval;
  }

 protected:
  unsigned long long m_snap_shot_interval;
  /* [한국어] 스냅샷 촬영 사이클 간격.
   * 설정자: 생성자. 읽는 자: try_snap_shot(), get_interval().
   * 값 범위: 양의 정수 (0이면 나눗셈 오류). */
};

/////////////////////////////////////////////////////////////////////////////////////
// spill log interface:
// - unified interface to spill log to file to avoid infinite memory usage for
// logging
/////////////////////////////////////////////////////////////////////////////////////

/*
 * [한국어]
 * spill_log_interface - 로그를 파일로 내보내는 통합 인터페이스
 *
 * 무한한 메모리 누적을 방지하기 위해 아카이브된 스냅샷 데이터를 파일에 쓰고
 * 메모리에서 제거하는 통일된 인터페이스를 정의한다.
 * spill_log_to_file() 전역 함수가 이 인터페이스를 구현하는 모든 객체에 spill()을 호출.
 * 구체 로거(thread_CFlocality, linear_histogram_logger)가 이를 상속.
 */
class spill_log_interface {
 public:
  spill_log_interface() {}
  virtual ~spill_log_interface() {} /* [한국어] 가상 소멸자 — 서브클래스 소멸자 보장 */

  /*
   * [한국어]
   * spill() - 아카이브된 로그 데이터를 파일로 내보내고 메모리에서 해제
   *
   * @fout: 출력 대상 FILE* (통계 로그 파일)
   * @final: true이면 현재 진행 중인 데이터도 함께 출력 (시뮬레이션 종료 시)
   *
   * 구현체는 아카이브 리스트를 순회하여 파일에 출력하고 메모리에서 제거.
   * final=false: 완료된 스냅샷만 출력 (현재 진행 중인 버퍼 제외).
   * final=true: 미완료 스냅샷도 출력 (시뮬레이션 최종 결과 포함).
   */
  virtual void spill(FILE *fout, bool final) = 0;
};

/////////////////////////////////////////////////////////////////////////////////////
// thread control-flow locality logger
/////////////////////////////////////////////////////////////////////////////////////

/*
 * [한국어]
 * thread_insn_span - 특정 사이클 구간에서 스레드들이 실행한 명령어 PC 집합 추적
 *
 * 각 스냅샷 구간에서 스레드들이 어떤 PC들을 실행했는지 추적하는 자료구조.
 * m_insn_span_count는 PC → 실행 횟수 매핑. set_span()으로 PC 추가,
 * print_sparse_histo()로 AerialVision용 희소 히스토그램 출력.
 * thread_CFlocality의 현재 버퍼(m_thd_span)와 아카이브(m_thd_span_archive)로 사용.
 */
class thread_insn_span {
 public:
  thread_insn_span(unsigned long long cycle, gpgpu_context *ctx);
  /* [한국어] 지정 사이클에서 시작하는 빈 span 생성. gpgpu_ctx는 PC→PTX라인번호 변환에 사용. */
  thread_insn_span(const thread_insn_span &other, gpgpu_context *ctx);
  /* [한국어] 복사 생성자 — other의 span_count_map과 cycle을 복사. */
  ~thread_insn_span(); /* [한국어] 소멸자 (trivial, 멤버가 스택/스마트 컨테이너) */

  thread_insn_span &operator=(const thread_insn_span &other);
  /* [한국어] 대입 연산자 — m_insn_span_count와 m_cycle 복사. */
  thread_insn_span &operator+=(const thread_insn_span &other);
  /* [한국어] span 누적 연산 — other의 각 PC 카운트를 현재 span에 합산. */
  /*
   * [한국어]
   * set_span() - 지정 PC를 span에 추가 (카운터 증가)
   *
   * @pc: 추가할 명령어 PC 주소
   *
   * m_insn_span_count[pc] += 1. pc < 0이면 무시 (미활성 하드웨어 스레드 필터링).
   */
  void set_span(address_type pc);
  void reset(unsigned long long cycle);
  /* [한국어] span 초기화 — m_cycle 갱신하고 m_insn_span_count 비움. */

  void print_span(FILE *fout) const;
  /* [한국어] 실행된 PC 주소 목록을 파일에 출력 (진단용). */
  void print_histo(FILE *fout) const;
  /* [한국어] PC별 실행 횟수를 히스토그램 형식으로 파일에 출력. */
  void print_sparse_histo(FILE *fout) const;
  /* [한국어] PTX 라인번호로 변환한 희소 히스토그램을 파일에 출력 (AerialVision용). */
  void print_sparse_histo(gzFile fout) const;
  /* [한국어] gzFile 압축 스트림에 희소 히스토그램 출력 (AerialVision gzip 파일용). */

 private:
  gpgpu_context *gpgpu_ctx;
  /* [한국어] PC→PTX 라인번호 변환용 GPGPU-Sim 전역 컨텍스트 포인터.
   * 설정자: 생성자. 읽는 자: print_sparse_histo()의 translate_pc_to_ptxlineno() 호출. */
  typedef tr1_hash_map<address_type, int> span_count_map;
  /* [한국어] PC 주소 → 실행 횟수 해시맵 타입 정의. address_type 키, int 카운터 값. */
  unsigned long long m_cycle;
  /* [한국어] 이 span이 시작된 시뮬레이션 사이클 번호.
   * 설정자: 생성자 및 reset(). 읽는 자: print_span/print_histo 출력 접두어. */
  span_count_map m_insn_span_count;
  /* [한국어] PC 주소별 실행 횟수 해시맵.
   * 설정자: set_span()이 해당 PC 카운터 증가. 읽는 자: print_*() 출력, operator+=.
   * 값 범위: 양의 정수 카운터. 동기화: 단일 SM/로거 스레드에서만 접근. */
};

/*
 * [한국어]
 * thread_CFlocality - 스레드 제어 흐름 지역성(Control-Flow Locality) 로거
 *
 * SM의 모든 하드웨어 스레드가 각 스냅샷 인터벌마다 어떤 PC들을 실행했는지 추적한다.
 * snap_shot_trigger와 spill_log_interface를 다중 상속하여 자동 스냅샷과
 * 파일 내보내기 기능을 제공한다. AerialVision 시각화 도구의 제어 흐름 지역성
 * 그래프 생성에 사용된다.
 */
class thread_CFlocality : public snap_shot_trigger, public spill_log_interface {
 public:
  thread_CFlocality(gpgpu_context *ctx, std::string name,
                    unsigned long long snap_shot_interval, int nthreads,
                    address_type start_pc, unsigned long long start_cycle = 0);
  /* [한국어] 생성자: nthreads 개 스레드의 PC 배열 초기화, snap_shot_trigger 설정. */
  ~thread_CFlocality(); /* [한국어] 소멸자 (trivial) */

  /*
   * [한국어]
   * update_thread_pc() - 특정 스레드의 현재 PC 업데이트
   *
   * @thread_id: 업데이트할 하드웨어 스레드 ID (0~nthreads-1)
   * @pc: 스레드가 현재 실행 중인 PTX PC
   *
   * m_thread_pc[thread_id] = pc로 갱신하고 m_thd_span.set_span(pc) 호출.
   * 시뮬레이션 중 매 스레드 명령어 실행 시 cflog_update_thread_pc()를 통해 호출.
   */
  void update_thread_pc(int thread_id, address_type pc);
  /*
   * [한국어]
   * snap_shot() - 현재 span을 아카이브에 저장하고 새 span 시작
   *
   * @current_cycle: 스냅샷 촬영 사이클 번호
   *
   * m_thd_span을 m_thd_span_archive에 push_back 후 reset().
   * 현재 m_thread_pc[] 전체를 새 span의 초기 상태로 설정.
   */
  void snap_shot(unsigned long long current_cycle);
  /*
   * [한국어]
   * spill() - 아카이브된 span 데이터를 파일에 출력하고 메모리 해제
   *
   * @fout: 출력 파일 스트림
   * @final: true이면 현재 진행 중인 m_thd_span도 출력
   */
  void spill(FILE *fout, bool final);

  void print_visualizer(FILE *fout);
  /* [한국어] AerialVision용 희소 히스토그램을 FILE*에 출력. */
  void print_visualizer(gzFile fout);
  /* [한국어] AerialVision용 희소 히스토그램을 gzFile 압축 파일에 출력. */
  void print_span(FILE *fout) const;
  /* [한국어] 전체 아카이브의 span(PC 목록)을 파일에 출력. */
  void print_histo(FILE *fout) const;
  /* [한국어] 전체 아카이브의 span 히스토그램을 파일에 출력. */

 private:
  std::string m_name;
  /* [한국어] 로거 이름 (예: "CFLog00"). 출력 시 접두어로 사용. */

  int m_nthreads;
  /* [한국어] 추적할 하드웨어 스레드 수. m_thread_pc 배열 크기.
   * 설정자: 생성자. 읽는 자: snap_shot()에서 전체 스레드 PC 초기화 순회. */
  std::vector<address_type> m_thread_pc;
  /* [한국어] 각 하드웨어 스레드의 현재 PC 배열 (인덱스 = 스레드 ID).
   * 설정자: update_thread_pc(). 읽는 자: snap_shot()에서 새 span 초기화.
   * 초기값: -1 (미할당 스레드는 set_span에서 무시). */

  unsigned long long m_cycle;
  /* [한국어] 현재 span의 시작 사이클. snap_shot() 시 갱신. */
  thread_insn_span m_thd_span;
  /* [한국어] 현재 스냅샷 인터벌의 활성 span 버퍼. snap_shot() 시 아카이브로 이동 후 리셋. */
  std::list<thread_insn_span> m_thd_span_archive;
  /* [한국어] 완료된 span 스냅샷의 아카이브 리스트. spill()에서 파일 출력 후 제거. */
};

/////////////////////////////////////////////////////////////////////////////////////
// per-insn active thread distribution (warp occ) logger
/////////////////////////////////////////////////////////////////////////////////////

/*
 * [한국어]
 * insn_warp_occ_logger - 명령어(PC)별 활성 스레드 수(워프 점유율) 분포 로거
 *
 * 각 PTX 명령어 PC마다 실행 시의 워프 내 활성 스레드 수 분포를 linear_histogram으로
 * 수집한다. m_insn_warp_occ[pc]는 PC pc에서의 워프 점유율 분포 히스토그램.
 * 벡터 크기는 PC가 증가할 때 2배씩 동적 확장된다.
 * 워프 분기(divergence)로 인한 활성 스레드 감소를 분석하는 데 사용.
 */
class insn_warp_occ_logger {
 public:
  /*
   * [한국어]
   * insn_warp_occ_logger() - 명령어별 워프 점유율 로거 생성자
   *
   * @simd_width: SIMD 폭 (워프 크기, 일반적으로 32). 히스토그램 bin 수로 사용.
   *
   * m_insn_warp_occ를 크기 1의 벡터로 초기화 (PC 0용). m_id는 전역 카운터 s_ids++.
   */
  insn_warp_occ_logger(int simd_width)
      : m_simd_width(simd_width),
        m_insn_warp_occ(1, linear_histogram(1, "", m_simd_width)),
        /* [한국어] PC 0 ~ 0 범위의 히스토그램 벡터 초기화. PC 추가 시 resize. */
        m_id(s_ids++) {} /* [한국어] 전역 로거 ID 자동 할당 */

  /*
   * [한국어]
   * insn_warp_occ_logger(복사 생성자) - 같은 크기의 히스토그램 벡터로 초기화
   *
   * other의 벡터 크기만 복사하고 히스토그램 내용은 새로 초기화 (카운터 0).
   * 이는 의도적 설계 — 새 로거는 빈 상태로 시작.
   */
  insn_warp_occ_logger(const insn_warp_occ_logger &other)
      : m_simd_width(other.m_simd_width),
        m_insn_warp_occ(other.m_insn_warp_occ.size(),
                        linear_histogram(1, "", m_simd_width)),
        /* [한국어] other와 같은 벡터 크기로 새 히스토그램 생성. 카운터는 0으로 초기화. */
        m_id(s_ids++) {} /* [한국어] 새 로거 ID 할당 */

  ~insn_warp_occ_logger() {} /* [한국어] trivial 소멸자 */

  /*
   * [한국어]
   * operator=() - 대입 연산자 (의도적으로 금지됨)
   *
   * assert(0)으로 대입 연산자 호출을 방지. 로거 대입은 설계 오류로 간주.
   */
  insn_warp_occ_logger &operator=(const insn_warp_occ_logger &p) {
    printf("insn_warp_occ_logger Operator= called: %02d \n", m_id);
    assert(0); /* [한국어] 대입 연산자는 지원하지 않음 — 호출 시 프로그램 중단 */
    return *this;
  }

  void set_id(int id) { m_id = id; } /* [한국어] 로거 ID 수동 설정 (배열 순번 부여 시 사용) */

  /*
   * [한국어]
   * log() - 특정 PC에서의 워프 점유율 기록
   *
   * @pc: 명령어의 PTX PC 주소 (배열 인덱스로 사용)
   * @warp_occ: 이 명령어 실행 시의 활성 스레드 수 (1~simd_width)
   *
   * pc >= 벡터 크기이면 벡터를 2*pc로 확장(동적 리사이즈).
   * m_insn_warp_occ[pc].add2bin(warp_occ - 1): 1-indexed를 0-indexed로 변환.
   */
  void log(address_type pc, int warp_occ) {
    if (pc >= m_insn_warp_occ.size())
      m_insn_warp_occ.resize(2 * pc, linear_histogram(1, "", m_simd_width));
    /* [한국어] PC가 현재 벡터 크기를 초과하면 2배로 확장 (amortized O(1)) */
    m_insn_warp_occ[pc].add2bin(warp_occ - 1);
    /* [한국어] 워프 점유율(1-based)을 0-based bin 인덱스로 변환하여 히스토그램에 추가 */
  }

  /*
   * [한국어]
   * print() - 모든 PC의 워프 점유율 분포를 파일에 출력
   *
   * @fout: 출력 대상 FILE*
   *
   * 형식: "InsnWarpOcc<id>-<pc> <bin0> <bin1> ... max=<max> avg=<avg>"
   * 시뮬레이션 종료 후 전체 워프 분기 분석 결과 출력에 사용.
   */
  void print(FILE *fout) const {
    for (unsigned i = 0; i < m_insn_warp_occ.size(); i++) {
      fprintf(fout, "InsnWarpOcc%02d-%d", m_id, i); /* [한국어] 로거ID-PC 형식 헤더 */
      m_insn_warp_occ[i].fprint(fout);               /* [한국어] 해당 PC의 히스토그램 출력 */
      fprintf(fout, "\n");
    }
  }

 private:
  int m_simd_width;
  /* [한국어] SIMD 폭 (워프 크기, 보통 32). 히스토그램 bin 수 및 최대값 범위.
   * 설정자: 생성자. 읽는 자: log()의 resize 시 새 히스토그램 생성. */
  std::vector<linear_histogram> m_insn_warp_occ;
  /* [한국어] PC 인덱스별 워프 점유율 분포 히스토그램 벡터.
   * 설정자: log()에서 필요 시 동적 확장. 읽는 자: print()에서 출력.
   * 값 범위: 크기는 최대 PC+1. 각 히스토그램은 0~simd_width-1 범위의 카운터. */
  int m_id;
  /* [한국어] 이 로거의 고유 ID. 출력 시 식별자로 사용.
   * 설정자: 생성자(s_ids++) 또는 set_id(). */
  static int s_ids;
  /* [한국어] 전역 로거 ID 카운터. 각 인스턴스 생성 시 자동 증가. */
};

/////////////////////////////////////////////////////////////////////////////////////
// generic linear histogram logger
/////////////////////////////////////////////////////////////////////////////////////

/*
 * [한국어]
 * linear_histogram_snapshot - 특정 사이클 시점의 선형 히스토그램 스냅샷
 *
 * n_bins 크기의 정수 벡터로 히스토그램 데이터를 저장. 각 위치(pos)에 addsample()로
 * 카운터를 증가시키고 subsample()로 감소시킬 수 있다 (unlog 지원).
 * linear_histogram_logger의 현재 버퍼(m_curr_lin_hist)와 아카이브로 사용.
 * AerialVision의 시각화에서 사이클별 히스토그램 데이터로 활용.
 */
class linear_histogram_snapshot {
 public:
  /*
   * [한국어]
   * linear_histogram_snapshot() - 지정 bin 수와 사이클로 빈 스냅샷 생성
   *
   * @n_bins: 히스토그램 bin 수 (카운터 배열 크기)
   * @cycle: 이 스냅샷의 시작 사이클 번호
   */
  linear_histogram_snapshot(int n_bins, unsigned long long cycle)
      : m_cycle(cycle), m_linear_histogram(n_bins, 0) {}
  /* [한국어] n_bins 크기의 정수 벡터를 0으로 초기화 */

  linear_histogram_snapshot(const linear_histogram_snapshot &other)
      : m_cycle(other.m_cycle), m_linear_histogram(other.m_linear_histogram) {}
  /* [한국어] 복사 생성자 — m_cycle과 히스토그램 벡터 전체 복사 */

  ~linear_histogram_snapshot() {} /* [한국어] trivial 소멸자 (std::vector 자동 해제) */

  /*
   * [한국어]
   * addsample() - 지정 위치의 카운터 증가
   *
   * @pos: 증가시킬 bin 인덱스 (0~n_bins-1)
   *
   * assert로 범위 검사 후 m_linear_histogram[pos] += 1.
   */
  void addsample(int pos) {
    assert((size_t)pos < m_linear_histogram.size()); /* [한국어] 범위 초과 방지 검사 */
    m_linear_histogram[pos] += 1; /* [한국어] 해당 bin 카운터 1 증가 */
  }

  /*
   * [한국어]
   * subsample() - 지정 위치의 카운터 감소 (unlog 지원)
   *
   * @pos: 감소시킬 bin 인덱스 (0~n_bins-1)
   *
   * shader_cache_access_unlog() 등에서 취소된 이벤트를 되돌릴 때 사용.
   */
  void subsample(int pos) {
    assert((size_t)pos < m_linear_histogram.size()); /* [한국어] 범위 초과 방지 검사 */
    m_linear_histogram[pos] -= 1; /* [한국어] 해당 bin 카운터 1 감소 */
  }

  /*
   * [한국어]
   * reset() - 스냅샷 초기화 (사이클 갱신 + 카운터 0으로 리셋)
   *
   * @cycle: 새 스냅샷의 시작 사이클 번호
   *
   * m_cycle 갱신 후 m_linear_histogram 전체를 0으로 assign.
   * snap_shot() 호출 후 현재 버퍼를 새 인터벌용으로 재사용 시 호출.
   */
  void reset(unsigned long long cycle) {
    m_cycle = cycle; /* [한국어] 새 사이클 번호 설정 */
    m_linear_histogram.assign(m_linear_histogram.size(), 0); /* [한국어] 모든 bin 0으로 초기화 */
  }

  void set_cycle(unsigned long long cycle) { m_cycle = cycle; }
  /* [한국어] 사이클 번호만 갱신 (reset_at_snap_shot=false 모드에서 사용) */

  /*
   * [한국어]
   * print() - 사이클 번호와 bin 카운터를 파일에 출력
   *
   * @fout: 출력 FILE*
   *
   * 형식: "<cycle> = <bin0> <bin1> ..."
   */
  void print(FILE *fout) const {
    fprintf(fout, "%d = ", (int)m_cycle); /* [한국어] 사이클 번호 출력 */
    for (unsigned int i = 0; i < m_linear_histogram.size(); i++) {
      fprintf(fout, "%d ", m_linear_histogram[i]); /* [한국어] 각 bin 카운터 출력 */
    }
  }

  /*
   * [한국어]
   * print_visualizer() - AerialVision용 bin 카운터만 출력 (사이클 번호 제외)
   *
   * @fout: 출력 FILE*
   */
  void print_visualizer(FILE *fout) const {
    for (unsigned int i = 0; i < m_linear_histogram.size(); i++) {
      fprintf(fout, "%d ", m_linear_histogram[i]); /* [한국어] 각 bin 카운터만 출력 */
    }
  }

  /*
   * [한국어]
   * print_visualizer() - AerialVision용 bin 카운터를 gzFile에 출력
   *
   * @fout: 출력 gzFile (zlib 압축 파일 스트림)
   */
  void print_visualizer(gzFile fout) const {
    for (unsigned int i = 0; i < m_linear_histogram.size(); i++) {
      gzprintf(fout, "%d ", m_linear_histogram[i]); /* [한국어] 압축 스트림에 bin 카운터 출력 */
    }
  }

 private:
  unsigned long long m_cycle;
  /* [한국어] 이 스냅샷의 시작 사이클 번호.
   * 설정자: 생성자, reset(), set_cycle(). 읽는 자: print() 출력 접두어. */
  std::vector<int> m_linear_histogram;
  /* [한국어] bin별 카운터 벡터 (크기 = n_bins).
   * 설정자: addsample() 증가, subsample() 감소, reset() 0으로 초기화.
   * 읽는 자: print(), print_visualizer(). */
};

/*
 * [한국어]
 * linear_histogram_logger - 주기적 스냅샷을 지원하는 선형 히스토그램 로거
 *
 * snap_shot_trigger와 spill_log_interface를 다중 상속하여 자동 스냅샷과
 * 파일 내보내기를 제공한다. 내부적으로 현재 버퍼(m_curr_lin_hist)와 아카이브
 * 리스트(m_lin_hist_archive)를 유지하며, snap_shot() 시 현재 버퍼를 아카이브로 이동.
 * SM별 워프 점유율, 메모리 접근 분포, 캐시 미스율, CTA 수 등 다양한 시간별 통계에 사용.
 */
class linear_histogram_logger : public snap_shot_trigger,
                                public spill_log_interface {
 public:
  /*
   * [한국어]
   * linear_histogram_logger() - 선형 히스토그램 로거 생성자
   *
   * @n_bins: 히스토그램 bin 수
   * @snap_shot_interval: 스냅샷 촬영 사이클 간격
   * @name: 로거 이름 (출력 접두어)
   * @reset_at_snap_shot: true이면 스냅샷 후 현재 버퍼 리셋, false이면 누적 유지
   * @start_cycle: 시작 사이클 (기본 0)
   */
  linear_histogram_logger(int n_bins, unsigned long long snap_shot_interval,
                          const char *name, bool reset_at_snap_shot = true,
                          unsigned long long start_cycle = 0);
  /*
   * [한국어]
   * linear_histogram_logger(복사 생성자) - 같은 설정으로 빈 로거 생성
   *
   * other의 설정(n_bins, interval, name, reset 플래그)을 복사하되
   * 히스토그램 데이터는 0으로 초기화. 벡터 assign() 시 복사 생성자 호출됨.
   */
  linear_histogram_logger(const linear_histogram_logger &other);

  ~linear_histogram_logger();
  /* [한국어] 소멸자: remove_snap_shot_trigger(this), remove_spill_log(this) 호출 */

  void set_id(int id) { m_id = id; }
  /* [한국어] 로거 ID 수동 설정 (배열 순번 부여 시 사용) */
  void log(int pos) { m_curr_lin_hist.addsample(pos); }
  /* [한국어] 현재 스냅샷 버퍼의 pos 위치 카운터 증가 */
  void unlog(int pos) { m_curr_lin_hist.subsample(pos); }
  /* [한국어] 현재 스냅샷 버퍼의 pos 위치 카운터 감소 (이벤트 취소용) */
  /*
   * [한국어]
   * snap_shot() - 현재 히스토그램 버퍼를 아카이브에 저장하고 리셋
   *
   * @current_cycle: 스냅샷 촬영 사이클
   *
   * m_curr_lin_hist를 m_lin_hist_archive에 push_back.
   * m_reset_at_snap_shot=true이면 reset(), false이면 set_cycle()만 호출.
   */
  void snap_shot(unsigned long long current_cycle);
  /*
   * [한국어]
   * spill() - 아카이브를 파일에 출력하고 메모리 해제
   *
   * @fout: 출력 FILE*, @final: true이면 현재 버퍼도 출력
   */
  void spill(FILE *fout, bool final);

  void print(FILE *fout) const;
  /* [한국어] 전체 아카이브 + 현재 버퍼를 파일에 출력. */
  void print_visualizer(FILE *fout);
  /* [한국어] AerialVision용 현재 버퍼 히스토그램을 FILE*에 출력. */
  void print_visualizer(gzFile fout);
  /* [한국어] AerialVision용 현재 버퍼 히스토그램을 gzFile에 출력. */

 private:
  int m_n_bins;
  /* [한국어] 히스토그램 bin 수. 설정자: 생성자. 읽는 자: snap_shot의 reset 시 사용. */
  linear_histogram_snapshot m_curr_lin_hist;
  /* [한국어] 현재 인터벌의 활성 히스토그램 버퍼.
   * 설정자: log()/unlog()로 누적. 읽는 자: snap_shot()에서 아카이브로 이동. */
  std::list<linear_histogram_snapshot> m_lin_hist_archive;
  /* [한국어] 완료된 스냅샷 아카이브 리스트. spill()에서 파일 출력 후 제거. */
  unsigned long long m_cycle;
  /* [한국어] 현재 버퍼의 시작 사이클. */
  bool m_reset_at_snap_shot;
  /* [한국어] 스냅샷 후 현재 버퍼를 리셋할지 여부.
   * true: 각 인터벌 독립 집계. false: 누적 통계 (shader_CTA_count_logger 등). */
  std::string m_name;
  /* [한국어] 로거 이름 (예: "ShdrWarpOcc"). 출력 시 "이름ID-" 형식 접두어. */
  int m_id;
  /* [한국어] 로거 고유 ID. 설정자: 생성자(s_ids++) 또는 set_id(). */
  static int s_ids;
  /* [한국어] 전역 로거 ID 카운터. */
};

enum cache_access_logger_types { NORMALS, TEXTURE, CONSTANT, INSTRUCTION };
/* [한국어] 캐시 접근 타입 열거형.
 * NORMALS=0: 일반 글로벌/로컬 L1 캐시, TEXTURE=1: 텍스처 캐시,
 * CONSTANT=2: 상수 캐시, INSTRUCTION=3: 명령어 캐시. */

/* [한국어] 전역 스냅샷/스필 관리 함수 */
void try_snap_shot(unsigned long long current_cycle);
/* [한국어] 등록된 모든 snap_shot_trigger를 순회하여 스냅샷 주기 도달 시 촬영.
 * gpu-sim.cc의 cycle() 루프에서 매 사이클 호출. */
void set_spill_interval(unsigned long long interval);
/* [한국어] 스필(로그 파일 내보내기) 간격을 설정. */
void spill_log_to_file(FILE *fout, int final, unsigned long long current_cycle);
/* [한국어] 등록된 모든 spill_log_interface에 spill() 호출. final=1이면 최종 출력. */

/* [한국어] 스레드 제어 흐름 지역성 로거 전역 함수 인터페이스 */
void create_thread_CFlogger(gpgpu_context *ctx, int n_loggers, int n_threads,
                            address_type start_pc,
                            unsigned long long logging_interval);
/* [한국어] n_loggers 개의 thread_CFlocality 로거를 생성하고 스냅샷/스필 리스트에 등록. */
void destroy_thread_CFlogger();
/* [한국어] 모든 스레드 CF 로거를 삭제하고 리스트에서 제거. */
void cflog_update_thread_pc(int logger_id, int thread_id, address_type pc);
/* [한국어] logger_id 번 로거에서 thread_id 스레드의 PC를 pc로 업데이트. */
void cflog_snapshot(int logger_id, unsigned long long cycle);
/* [한국어] deprecated: logger_id 번 로거의 스냅샷 강제 촬영. */
void cflog_print(FILE *fout);
/* [한국어] 모든 CF 로거의 히스토그램을 파일에 출력. */
void cflog_print_path_expression(FILE *fout);
/* [한국어] (미구현) 경로 표현식 출력. */
void cflog_visualizer_print(FILE *fout);
/* [한국어] AerialVision용 CF 로컬리티 희소 히스토그램을 FILE*에 출력. */
void cflog_visualizer_gzprint(gzFile fout);
/* [한국어] AerialVision용 CF 로컬리티 희소 히스토그램을 gzFile에 출력. */

/* [한국어] 명령어별 워프 점유율 로거 전역 함수 인터페이스 */
void insn_warp_occ_create(int n_loggers, int simd_width);
/* [한국어] n_loggers 개의 insn_warp_occ_logger 생성. */
void insn_warp_occ_log(int logger_id, address_type pc, int warp_occ);
/* [한국어] logger_id 로거에 PC-워프점유율 쌍 기록. warp_occ <= 0이면 무시. */
void insn_warp_occ_print(FILE *fout);
/* [한국어] 모든 명령어별 워프 점유율 히스토그램을 파일에 출력. */

/* [한국어] SM별 워프 점유율 로거 전역 함수 인터페이스 */
void shader_warp_occ_create(int n_loggers, int simd_width,
                            unsigned long long logging_interval);
/* [한국어] SM 수만큼 linear_histogram_logger 생성. simd_width+1 bins로 워프 점유율 분포 수집. */
void shader_warp_occ_log(int logger_id, int warp_occ);
/* [한국어] logger_id SM의 워프 점유율 기록. */
void shader_warp_occ_snapshot(int logger_id, unsigned long long current_cycle);
/* [한국어] logger_id SM의 워프 점유율 스냅샷 촬영. */
void shader_warp_occ_print(FILE *fout);
/* [한국어] 모든 SM 워프 점유율 히스토그램을 파일에 출력. */

/* [한국어] SM별 메모리 접근 로거 전역 함수 인터페이스 */
void shader_mem_acc_create(int n_loggers, int n_dram, int n_bank,
                           unsigned long long logging_interval);
/* [한국어] SM 수만큼 로거 생성. 2*n_dram*(n_bank+1) bins (읽기/쓰기 × DRAM × 뱅크). */
void shader_mem_acc_log(int logger_id, int dram_id, int bank, char rw);
/* [한국어] SM→DRAM 메모리 접근 기록. rw='r'(읽기) 또는 'w'(쓰기). */
void shader_mem_acc_snapshot(int logger_id, unsigned long long current_cycle);
/* [한국어] SM 메모리 접근 스냅샷 촬영. */
void shader_mem_acc_print(FILE *fout);
/* [한국어] 모든 SM 메모리 접근 히스토그램 출력. */

/* [한국어] SM별 메모리 레이턴시 로거 전역 함수 인터페이스 */
void shader_mem_lat_create(int n_loggers, unsigned long long logging_interval);
/* [한국어] SM 수만큼 로거 생성. 48 bins로 log-sqrt2 스케일 레이턴시 분포 수집 (최대 2^24). */
void shader_mem_lat_log(int logger_id, int latency);
/* [한국어] SM의 메모리 레이턴시(사이클) 기록. log-sqrt2 스케일로 bin 인덱스 계산. */
void shader_mem_lat_snapshot(int logger_id, unsigned long long current_cycle);
/* [한국어] SM 레이턴시 스냅샷 촬영. */
void shader_mem_lat_print(FILE *fout);
/* [한국어] 모든 SM 레이턴시 히스토그램 출력. */

/* [한국어] SM별 캐시 접근/미스 로거 전역 함수 인터페이스 */
int get_shader_normal_cache_id();    /* [한국어] NORMALS(0) 반환 */
int get_shader_texture_cache_id();   /* [한국어] TEXTURE(1) 반환 */
int get_shader_constant_cache_id();  /* [한국어] CONSTANT(2) 반환 */
int get_shader_instruction_cache_id(); /* [한국어] INSTRUCTION(3) 반환 */
void shader_cache_access_create(int n_loggers, int n_types,
                                unsigned long long logging_interval);
/* [한국어] SM 수만큼 로거 생성. n_types*2 bins (타입별 접근수+미스수). */
void shader_cache_access_log(int logger_id, int type, int miss);
/* [한국어] 캐시 접근 기록. type=캐시종류, miss=0(히트)/1(미스). */
void shader_cache_access_unlog(int logger_id, int type, int miss);
/* [한국어] 캐시 접근 취소 (카운터 감소). MSHR 병합 시 이중 계산 방지용. */
void shader_cache_access_print(FILE *fout);
/* [한국어] 모든 SM 캐시 접근/미스 히스토그램 출력. */

/* [한국어] SM별 CTA(Cooperative Thread Array) 수 로거 전역 함수 인터페이스 */
void shader_CTA_count_create(int n_shaders,
                             unsigned long long logging_interval);
/* [한국어] SM 수를 bins으로 하는 단일 로거 생성. gpgpu_spread_blocks_across_cores 옵션 시 유효. */
void shader_CTA_count_log(int shader_id, int nCTAadded);
/* [한국어] shader_id SM에 nCTAadded 개의 CTA가 추가되었음을 기록. */
void shader_CTA_count_unlog(int shader_id, int nCTAdone);
/* [한국어] shader_id SM에서 nCTAdone 개의 CTA가 완료되었음을 기록 (카운터 감소). */
void shader_CTA_count_resetnow();
/* [한국어] (현재 미구현) CTA 카운터 즉시 리셋. */
void shader_CTA_count_print(FILE *fout);
/* [한국어] CTA 수 히스토그램을 파일에 출력. */
void shader_CTA_count_visualizer_print(FILE *fout);
/* [한국어] AerialVision용 CTA 분포를 FILE*에 출력. */
void shader_CTA_count_visualizer_gzprint(gzFile fout);
/* [한국어] AerialVision용 CTA 분포를 gzFile에 출력. */

#endif /* CFLOGGER_H */
