/*
 * [한국어 설명] PTX 소스 라인 단위 통계 수집 구현 (ptx-stats.cc)
 *
 * === 파일의 역할 ===
 * ptx-stats.h에서 선언된 ptx_stats 클래스 메서드와 관련 자료구조를 구현한다.
 * PTX 소스 라인(파일명 + 줄 번호) 별로 실행 횟수, 파이프라인 지연, DRAM 트래픽,
 * 공유 메모리 뱅크 충돌, 비정렬 전역 메모리 접근, 인플라이트 명령어, 워프 분기 횟수를
 * 정적 해시맵(ptx_file_line_stats_tracker)에 누적하고, 시뮬레이션 종료 시 파일로 출력한다.
 * 또한 인플라이트 메모리 명령어 추적기(ptx_inflight_memory_insn_tracker)를 통해
 * 파이프라인 버블(exposed latency)을 원인 명령어에 귀속하는 메커니즘을 구현한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 통계 수집 흐름: cuda-sim/instructions.cc → ptx_file_line_stats_add_exec_count()
 *   gpgpu-sim/shader.cc → ptx_stats::ptx_file_line_stats_add_latency 등
 *   gpgpu-sim/gpu-sim.cc → ptx_stats::ptx_file_line_stats_write_file()
 * 기능 시뮬레이션과 타이밍 시뮬레이션 양쪽에서 호출되는 크로스-레이어 모듈.
 * 실행 컨텍스트: 호스트 CPU 단일 스레드.
 *
 * === 타 모듈과의 연결 ===
 * 의존: ptx-stats.h, option_parser.h, libcuda/gpgpu_context.h, ptx_ir.h, ptx_sim.h,
 *   tr1_hash_map.h (ptx_file_line_stats_map_t)
 * 의존받음: cuda-sim/instructions.cc, gpgpu-sim/shader.cc, gpgpu-sim/gpu-sim.cc
 * 공유 자료구조: ptx_file_line_stats_tracker (정적 전역 해시맵),
 *   inflight_mem_tracker (SM별 인플라이트 명령어 추적기 배열)
 *
 * === 주요 함수/구조체 요약 ===
 * ptx_file_line: PTX 소스 라인 식별자 (파일명 + 줄 번호)
 * ptx_file_line_stats: 단일 PTX 라인의 통계 집합체 (exec_count, latency 등 8개 필드)
 * ptx_inflight_memory_insn_tracker: SM별 인플라이트 메모리 명령어 카운트 맵 + 지연 귀속 메서드
 * ptx_file_line_stats_write_file(): 트래커 전체를 파일로 출력
 * ptx_file_line_stats_commit_exposed_latency(): 파이프라인 버블을 인플라이트 명령어에 귀속
 */
// Copyright (c) 2009-2011, Wilson W.L. Fung, Tor M. Aamodt
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

#include "ptx-stats.h"              /* [한국어] ptx_stats 클래스, 자유 함수 선언 */
#include <stdio.h>                   /* [한국어] fopen, fprintf, fflush, fclose — 통계 파일 출력 */
#include <map>                       /* [한국어] std::map — ptx_inflight_memory_insn_tracker의 명령어→카운트 맵 */
#include "../../libcuda/gpgpu_context.h" /* [한국어] gpgpu_context::pc_to_instruction() — PC로 ptx_instruction 조회 */
#include "../option_parser.h"        /* [한국어] option_parser_register — 통계 수집 옵션 등록 */
#include "../tr1_hash_map.h"         /* [한국어] tr1_hash_map — ptx_file_line_stats_tracker 맵 구현 */
#include "ptx_ir.h"                  /* [한국어] ptx_instruction::source_file/source_line() — PTX 소스 위치 조회 */
#include "ptx_sim.h"                 /* [한국어] ptx_thread_info (간접 의존 — pc_to_instruction을 통해 사용) */

/*
 * [한국어]
 * ptx_stats::ptx_file_line_stats_options - PTX 통계 수집 옵션 등록
 *
 * @opp: option_parser 핸들 — gpgpusim.config 파싱 컨텍스트
 *
 * -enable_ptx_file_line_stats(기본 1=활성화), -ptx_line_stats_filename(기본 "gpgpu_inst_stats.txt")
 * 두 가지 옵션을 option_parser에 등록한다.
 * 실행 컨텍스트: 시뮬레이터 옵션 파싱 단계.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint (옵션 등록) → [이 함수]
 */
void ptx_stats::ptx_file_line_stats_options(option_parser_t opp) {
  option_parser_register(
      opp, "-enable_ptx_file_line_stats", OPT_BOOL, &enable_ptx_file_line_stats,
      "Turn on PTX source line statistic profiling. (1 = On)", "1"); /* [한국어] 통계 수집 활성화 옵션 등록 (기본값 1=활성) */
  option_parser_register(
      opp, "-ptx_line_stats_filename", OPT_CSTR, &ptx_line_stats_filename,
      "Output file for PTX source line statistics.", "gpgpu_inst_stats.txt"); /* [한국어] 통계 출력 파일명 옵션 등록 */
}

// implementations

// defining a PTX source line = filename + line number
/* [한국어] PTX 소스 라인 식별자: 파일명(문자열) + 줄 번호의 쌍으로 정의 */

/*
 * [한국어]
 * ptx_file_line - PTX 소스 라인 고유 식별자 클래스
 *
 * ptx_file_line_stats_tracker 해시맵의 키로 사용. 파일명(std::string)과
 * 줄 번호(unsigned)의 쌍으로 PTX 소스 라인을 고유하게 식별한다.
 * operator< (std::map용), operator== (hash_map용), hash 함수가 구현되어 있어
 * 정렬된 맵(tr1_hash_map_ismap==1)과 해시맵 양쪽에서 사용 가능.
 * 실행 컨텍스트: 통계 수집 시 키로 생성됨 (호스트 CPU 단일 스레드).
 */
class ptx_file_line {
 public:
  /*
   * [한국어] 생성자 - 파일명과 줄 번호로 식별자 초기화
   *
   * @s: PTX 소스 파일명 (NULL이면 "NULL_NAME"으로 대체)
   * @l: PTX 소스 줄 번호
   */
  ptx_file_line(const char *s, int l) {
    if (s == NULL)  /* [한국어] 파일명이 NULL인 경우 — 알 수 없는 소스 위치 */
      st = "NULL_NAME"; /* [한국어] NULL 파일을 "NULL_NAME"으로 표현하여 키 충돌 방지 */
    else
      st = s;  /* [한국어] 파일명 문자열로 초기화 */
    line = l;  /* [한국어] 줄 번호 저장 */
  }

  /*
   * [한국어]
   * operator< - std::map 정렬용 비교 연산자
   *
   * 파일명으로 먼저 비교, 같으면 줄 번호로 비교. std::map<ptx_file_line, ...>의 키로 사용.
   */
  bool operator<(const ptx_file_line &other) const {
    if (st == other.st) { /* [한국어] 파일명이 같으면 줄 번호로 비교 */
      if (line < other.line)
        return true;  /* [한국어] 줄 번호가 작으면 이 키가 앞에 위치 */
      else
        return false;
    } else {
      return st < other.st; /* [한국어] 파일명 알파벳 순 비교 */
    }
  }

  /*
   * [한국어]
   * operator== - 해시맵 동등 비교 연산자
   *
   * 줄 번호와 파일명이 모두 같아야 동일한 PTX 라인으로 판단.
   */
  bool operator==(const ptx_file_line &other) const {
    return (line == other.line) && (st == other.st); /* [한국어] 줄 번호와 파일명 모두 일치해야 같은 라인 */
  }

  std::string st;  /* [한국어] PTX 소스 파일명.
                     * 설정자: 생성자에서 초기화.
                     * 읽는 자: operator<, operator==, 해시 함수, write_file().
                     * 값 범위: 유효한 파일명 또는 "NULL_NAME".
                     * 동기화: 불변 (생성 후 변경 없음). */
  unsigned line;   /* [한국어] PTX 소스 줄 번호.
                     * 설정자: 생성자에서 초기화.
                     * 읽는 자: operator<, operator==, 해시 함수, write_file().
                     * 값 범위: 1 이상의 양수.
                     * 동기화: 불변. */
};

// holds all statistics collected for a singe PTX source line
/* [한국어] 단일 PTX 소스 라인에 대해 수집된 모든 통계를 보관하는 클래스 */

/*
 * [한국어]
 * ptx_file_line_stats - 단일 PTX 소스 라인의 성능 통계 집합체
 *
 * ptx_file_line_stats_tracker 해시맵의 값(value)으로 사용되며,
 * 시뮬레이션 전체에 걸쳐 해당 PTX 라인의 누적 통계를 보관한다.
 * 생성자에서 모든 필드를 0으로 초기화하며, 각 add_* 함수가 누적 증가시킨다.
 * 실행 컨텍스트: 통계 수집 시 해시맵을 통해 접근 (호스트 CPU 단일 스레드).
 */
class ptx_file_line_stats {
 public:
  /*
   * [한국어] 생성자 - 모든 통계 카운터를 0으로 초기화
   */
  ptx_file_line_stats()
      : exec_count(0),
        latency(0),
        dram_traffic(0),
        smem_n_way_bank_conflict_total(0),
        smem_warp_count(0),
        gmem_n_access_total(0),
        gmem_warp_count(0),
        exposed_latency(0),
        warp_divergence(0) {}

  unsigned long exec_count;    /* [한국어] 이 PTX 라인을 실행한 스레드 누적 횟수 (스레드 단위, 워프 단위 아님).
                                 * 설정자: ptx_file_line_stats_add_exec_count()에서 +1.
                                 * 읽는 자: ptx_file_line_stats_write_file().
                                 * 값 범위: 0 ~ (전체 스레드 수 × 반복 횟수). */
  unsigned long long latency;  /* [한국어] 이 PTX 라인이 파이프라인에서 소비한 누적 사이클.
                                 * 설정자: ptx_file_line_stats_add_latency()에서 += latency.
                                 * 읽는 자: ptx_file_line_stats_write_file().
                                 * 값 범위: 0 이상. */
  unsigned long long dram_traffic; /* [한국어] 이 PTX 라인이 생성한 DRAM 요청 누적 수.
                                     * 설정자: ptx_file_line_stats_add_dram_traffic()에서 += dram_traffic.
                                     * 읽는 자: ptx_file_line_stats_write_file().
                                     * 값 범위: 0 이상. */
  unsigned long long
      smem_n_way_bank_conflict_total; // total number of banks accessed by this
                                       // instruction
  /* [한국어] 이 PTX 라인에서 발생한 공유 메모리 뱅크 충돌 누적 횟수 (n-way 합산).
   * 설정자: ptx_file_line_stats_add_smem_bank_conflict()에서 += n_way_bkconflict.
   * 읽는 자: ptx_file_line_stats_write_file().
   * 값 범위: 0 이상. 충돌 없으면 0. */
  unsigned long smem_warp_count; // number of warps accessing shared memory
  /* [한국어] 이 PTX 라인에서 공유 메모리에 접근한 워프 누적 수.
   * 설정자: ptx_file_line_stats_add_smem_bank_conflict()에서 +1.
   * 읽는 자: ptx_file_line_stats_write_file().
   * smem_n_way_bank_conflict_total / smem_warp_count = 평균 충돌 횟수 계산 가능. */
  unsigned long long gmem_n_access_total; // number of uncoalesced access in
                                           // total from this instruction
  /* [한국어] 이 PTX 라인에서 비정렬(uncoalesced)로 인해 생성된 전역 메모리 요청 누적 수.
   * 설정자: ptx_file_line_stats_add_uncoalesced_gmem()에서 += n_access.
   * 읽는 자: ptx_file_line_stats_write_file().
   * 값 범위: 0 이상. 정상 coalesced 접근은 카운트 안 됨. */
  unsigned long
      gmem_warp_count; // number of warps causing these uncoalesced access
  /* [한국어] 이 PTX 라인에서 비정렬 전역 메모리 접근을 일으킨 워프 누적 수.
   * 설정자: ptx_file_line_stats_add_uncoalesced_gmem()에서 +1.
   * 읽는 자: ptx_file_line_stats_write_file().
   * gmem_n_access_total / gmem_warp_count = 워프당 평균 추가 요청 수 계산 가능. */
  unsigned long long exposed_latency; // latency exposed as pipeline bubbles
                                       // (attributed to this instruction)
  /* [한국어] 이 PTX 라인이 파이프라인 버블(스톨)을 유발한 누적 사이클.
   * 설정자: ptx_inflight_memory_insn_tracker::attribute_exposed_latency()에서 누적.
   * 읽는 자: ptx_file_line_stats_write_file().
   * 메모리 레이턴시로 인해 다른 워프도 실행 못 한 사이클 수를 나타냄. */
  unsigned long long
      warp_divergence; // number of warp divergence occured at this instruction
  /* [한국어] 이 PTX 라인에서 워프 분기(divergence)가 발생한 누적 횟수.
   * 설정자: ptx_file_line_stats_add_warp_divergence()에서 += n_way_divergence.
   * 읽는 자: ptx_file_line_stats_write_file().
   * 값이 크면 이 라인이 분기 병목임을 의미. */
};

#if (tr1_hash_map_ismap == 1)
/* [한국어] std::map 기반: 해시 함수 불필요, operator< 사용 */
typedef tr1_hash_map<ptx_file_line, ptx_file_line_stats>
    ptx_file_line_stats_map_t; /* [한국어] ptx_file_line → ptx_file_line_stats 정렬 맵 타입 별칭 */
#else
/*
 * [한국어]
 * hash_ptx_file_line - ptx_file_line 해시 함수 객체
 *
 * unordered_map(해시맵)에서 ptx_file_line을 키로 사용하기 위한 해시 함수.
 * 줄 번호(unsigned)만 해싱 — 파일명이 다르더라도 같은 줄 번호면 같은 버킷에 배치됨.
 * 충돌 시 operator==로 구분.
 */
struct hash_ptx_file_line {
  std::size_t operator()(const ptx_file_line &pfline) const {
    std::hash<unsigned> hash_line; /* [한국어] std::hash<unsigned> — 줄 번호 해시 계산기 */
    return hash_line(pfline.line); /* [한국어] 줄 번호를 해시하여 버킷 결정 — 파일명은 operator==에서 구분 */
  }
};
typedef tr1_hash_map<ptx_file_line, ptx_file_line_stats, hash_ptx_file_line>
    ptx_file_line_stats_map_t; /* [한국어] ptx_file_line → ptx_file_line_stats 해시맵 타입 별칭 */
#endif

static ptx_file_line_stats_map_t ptx_file_line_stats_tracker; /* [한국어] PTX 소스 라인별 통계를 보관하는 전역 정적 해시맵.
                                                                 * 설정자: 각 add_* 함수에서 operator[]로 접근 및 누적.
                                                                 * 읽는 자: ptx_file_line_stats_write_file()에서 순회 출력.
                                                                 * 동기화: 단일 스레드 접근이므로 락 불필요.
                                                                 * 수명: 프로그램 시작부터 종료까지 유지 (static). */

/*
 * [한국어]
 * ptx_stats::ptx_file_line_stats_write_file - 수집된 통계를 파일로 출력
 *
 * enable_ptx_file_line_stats가 0이면 즉시 반환 (수집 비활성화 시 출력 생략).
 * ptx_file_line_stats_tracker의 모든 항목을 순회하며 ptx_line_stats_filename에 기록.
 * 출력 형식: "<파일명> <줄번호> : <exec_count> <latency> <dram_traffic>
 *   <smem_bk_conflicts> <smem_warp> <gmem_access_generated> <gmem_warp>
 *   <exposed_latency> <warp_divergence>"
 * 시뮬레이션 완료 후 gpu-sim.cc에서 호출됨.
 * 실행 컨텍스트: 시뮬레이션 종료 단계 (단일 스레드).
 *
 * 호출 체인:
 *   gpu-sim.cc (시뮬레이션 완료) → gpgpu_context::ptx_stats::ptx_file_line_stats_write_file
 */
// output statistics to a file
void ptx_stats::ptx_file_line_stats_write_file() {
  // check if stat collection is turned on
  if (enable_ptx_file_line_stats == 0) return; /* [한국어] 통계 수집 비활성화 시 출력 생략 */

  ptx_file_line_stats_map_t::iterator it; /* [한국어] 맵 순회 이터레이터 */
  FILE *pfile;

  pfile = fopen(ptx_line_stats_filename, "w"); /* [한국어] 통계 출력 파일 쓰기 모드로 열기 */
  fprintf(
      pfile,
      "kernel line : count latency dram_traffic smem_bk_conflicts smem_warp "
      "gmem_access_generated gmem_warp exposed_latency warp_divergence\n"); /* [한국어] 컬럼 헤더 출력 */
  for (it = ptx_file_line_stats_tracker.begin();
       it != ptx_file_line_stats_tracker.end(); it++) { /* [한국어] 모든 PTX 소스 라인 통계 순회 */
    fprintf(pfile, "%s %i : ", it->first.st.c_str(), it->first.line); /* [한국어] "<파일명> <줄번호> : " 키 출력 */
    fprintf(pfile, "%lu ", it->second.exec_count);                     /* [한국어] 실행 횟수 출력 */
    fprintf(pfile, "%llu ", it->second.latency);                       /* [한국어] 파이프라인 지연 사이클 출력 */
    fprintf(pfile, "%llu ", it->second.dram_traffic);                  /* [한국어] DRAM 요청 수 출력 */
    fprintf(pfile, "%llu ", it->second.smem_n_way_bank_conflict_total); /* [한국어] 공유 메모리 뱅크 충돌 합계 출력 */
    fprintf(pfile, "%lu ", it->second.smem_warp_count);                /* [한국어] 공유 메모리 접근 워프 수 출력 */
    fprintf(pfile, "%llu ", it->second.gmem_n_access_total);           /* [한국어] 비정렬 전역 메모리 요청 수 출력 */
    fprintf(pfile, "%lu ", it->second.gmem_warp_count);                /* [한국어] 비정렬 전역 메모리 워프 수 출력 */
    fprintf(pfile, "%llu ", it->second.exposed_latency);               /* [한국어] 노출된 파이프라인 버블 사이클 출력 */
    fprintf(pfile, "%llu ", it->second.warp_divergence);               /* [한국어] 워프 분기 횟수 출력 */
    fprintf(pfile, "\n"); /* [한국어] 라인 끝 줄바꿈 */
  }
  fflush(pfile); /* [한국어] 버퍼 강제 플러시 — 시뮬레이터 종료 전 파일 내용 보장 */
  fclose(pfile); /* [한국어] 파일 닫기 */
}

/*
 * [한국어]
 * ptx_file_line_stats_add_exec_count - PTX 명령어 실행 횟수 1 증가 (자유 함수)
 *
 * @pInsn: 실행된 PTX 명령어 — source_file/source_line으로 통계 키 결정
 *
 * 기능 시뮬레이션에서 PTX 명령어가 실행될 때마다 호출. 스레드 단위로 카운트
 * (워프가 아닌 개별 스레드 기준). ptx_file_line_stats_tracker에서 해당 라인의
 * exec_count를 1 증가.
 * 실행 컨텍스트: 기능 시뮬레이션 (instructions.cc execute() 중).
 *
 * 호출 체인:
 *   instructions.cc (각 PTX 명령어 execute) → [이 함수]
 */
// attribute one more execution count to this ptx instruction
// counting the number of threads (not warps) executing this instruction
void ptx_file_line_stats_add_exec_count(const ptx_instruction *pInsn) {
  ptx_file_line_stats_tracker[ptx_file_line(pInsn->source_file(),
                                            pInsn->source_line())]
      .exec_count += 1; /* [한국어] 해당 PTX 소스 라인의 실행 횟수 1 증가 (operator[]로 없으면 자동 생성) */
}

/*
 * [한국어]
 * ptx_stats::ptx_file_line_stats_add_latency - PC에 대응하는 PTX 라인에 파이프라인 지연 귀속
 *
 * @pc: 파이프라인 지연을 유발한 명령어의 PC — pc_to_instruction으로 PTX 라인 조회
 * @latency: 귀속할 사이클 수
 *
 * 타이밍 시뮬레이션에서 워프가 파이프라인에서 대기한 사이클 수를 해당 PTX 라인에 귀속.
 * pc_to_instruction이 NULL을 반환하는 경우(PC 매핑 없음) 귀속 생략.
 * 실행 컨텍스트: 타이밍 시뮬레이션 (shader.cc 파이프라인 사이클 처리 중).
 *
 * 호출 체인:
 *   gpgpu-sim/shader.cc → [이 함수]
 */
// attribute pipeline latency to this ptx instruction (specified by the pc)
// pipeline latency is the number of cycles a warp with this instruction spent
// in the pipeline
void ptx_stats::ptx_file_line_stats_add_latency(unsigned pc, unsigned latency) {
  const ptx_instruction *pInsn = gpgpu_ctx->pc_to_instruction(pc); /* [한국어] PC로 PTX 명령어 객체 조회 */

  if (pInsn != NULL) /* [한국어] PC에 대응하는 명령어가 있는 경우에만 귀속 */
    ptx_file_line_stats_tracker[ptx_file_line(pInsn->source_file(),
                                              pInsn->source_line())]
        .latency += latency; /* [한국어] 해당 PTX 라인의 누적 지연 사이클에 추가 */
}

/*
 * [한국어]
 * ptx_stats::ptx_file_line_stats_add_dram_traffic - PC에 대응하는 PTX 라인에 DRAM 트래픽 귀속
 *
 * @pc: DRAM 트래픽을 유발한 명령어의 PC
 * @dram_traffic: 귀속할 DRAM 요청 수
 *
 * 타이밍 시뮬레이션에서 DRAM 메모리 요청이 발생할 때 해당 PTX 라인에 요청 수 귀속.
 * 실행 컨텍스트: 타이밍 시뮬레이션 (DRAM 요청 처리 중).
 *
 * 호출 체인:
 *   gpgpu-sim/dram.cc → [이 함수]
 */
// attribute dram traffic to this ptx instruction (specified by the pc)
// dram traffic is counted in number of requests
void ptx_stats::ptx_file_line_stats_add_dram_traffic(unsigned pc,
                                                     unsigned dram_traffic) {
  const ptx_instruction *pInsn = gpgpu_ctx->pc_to_instruction(pc); /* [한국어] PC로 PTX 명령어 조회 */

  if (pInsn != NULL) /* [한국어] 유효한 명령어에만 귀속 */
    ptx_file_line_stats_tracker[ptx_file_line(pInsn->source_file(),
                                              pInsn->source_line())]
        .dram_traffic += dram_traffic; /* [한국어] 해당 PTX 라인의 누적 DRAM 요청 수에 추가 */
}

/*
 * [한국어]
 * ptx_stats::ptx_file_line_stats_add_smem_bank_conflict - 공유 메모리 뱅크 충돌 귀속
 *
 * @pc: 충돌을 유발한 명령어의 PC
 * @n_way_bkconflict: n-way 뱅크 충돌 횟수
 *
 * 공유 메모리 뱅크 충돌(shared memory bank conflict)이 발생할 때 해당 PTX 라인에 귀속.
 * smem_n_way_bank_conflict_total += n_way_bkconflict, smem_warp_count += 1.
 * 실행 컨텍스트: 타이밍 시뮬레이션 (operand collector / 공유 메모리 접근 처리 중).
 *
 * 호출 체인:
 *   gpgpu-sim/shader.cc (smem 뱅크 충돌 감지) → [이 함수]
 */
// attribute the number of shared memory access cycles to a ptx instruction
// counts both the number of warps doing shared memory access and the number of
// cycles involved
void ptx_stats::ptx_file_line_stats_add_smem_bank_conflict(
    unsigned pc, unsigned n_way_bkconflict) {
  const ptx_instruction *pInsn = gpgpu_ctx->pc_to_instruction(pc); /* [한국어] PC로 PTX 명령어 조회 */

  if (pInsn != NULL) {
    ptx_file_line_stats &line_stats = ptx_file_line_stats_tracker[ptx_file_line(
        pInsn->source_file(), pInsn->source_line())]; /* [한국어] 해당 PTX 라인의 통계 참조 (없으면 자동 생성) */
    line_stats.smem_n_way_bank_conflict_total += n_way_bkconflict; /* [한국어] n-way 충돌 합계 누적 */
    line_stats.smem_warp_count += 1; /* [한국어] 충돌한 워프 수 1 증가 */
  }
}

/*
 * [한국어]
 * ptx_stats::ptx_file_line_stats_add_uncoalesced_gmem - 비정렬 전역 메모리 접근 귀속
 *
 * @pc: 비정렬 접근을 유발한 명령어의 PC
 * @n_access: 비정렬로 인해 추가 생성된 메모리 요청 수
 *
 * 전역 메모리(global memory) 접근이 coalescing에 실패하여 추가 요청이 생성될 때 귀속.
 * gmem_n_access_total += n_access, gmem_warp_count += 1.
 * 실행 컨텍스트: 타이밍 시뮬레이션 (메모리 coalescing 처리 중).
 *
 * 호출 체인:
 *   gpgpu-sim/shader.cc (coalescing 유닛) → [이 함수]
 */
// attribute a non-coalesced mem access to a ptx instruction
// counts both the number of warps causing this and the number of memory
// requests generated
void ptx_stats::ptx_file_line_stats_add_uncoalesced_gmem(unsigned pc,
                                                         unsigned n_access) {
  const ptx_instruction *pInsn = gpgpu_ctx->pc_to_instruction(pc); /* [한국어] PC로 PTX 명령어 조회 */

  if (pInsn != NULL) {
    ptx_file_line_stats &line_stats = ptx_file_line_stats_tracker[ptx_file_line(
        pInsn->source_file(), pInsn->source_line())]; /* [한국어] 해당 PTX 라인의 통계 참조 */
    line_stats.gmem_n_access_total += n_access; /* [한국어] 비정렬로 인한 추가 요청 수 누적 */
    line_stats.gmem_warp_count += 1;            /* [한국어] 비정렬 접근 워프 수 1 증가 */
  }
}

// a class that tracks the inflight memory instructions of a shader core
// and attributes exposed latency to those instructions when signaled to do so
/* [한국어] SM(shader core) 내 인플라이트(처리 중) 메모리 명령어를 추적하고
 * 파이프라인 버블 발생 시 해당 명령어에 exposed latency를 귀속하는 클래스 */

/*
 * [한국어]
 * ptx_inflight_memory_insn_tracker - SM별 인플라이트 메모리 명령어 추적기
 *
 * 특정 SM에서 현재 처리 중(파이프라인에서 완료를 기다리는)인 메모리 명령어와
 * 그 참조 카운트를 std::map으로 관리한다. 파이프라인이 메모리 응답을 기다리며
 * 버블(빈 사이클)이 발생하면, 현재 인플라이트 상태인 모든 명령어에
 * 균등하게 exposed latency를 귀속한다.
 * 실행 컨텍스트: 타이밍 시뮬레이션 (shader.cc 사이클 처리 중).
 */
class ptx_inflight_memory_insn_tracker {
 public:
  typedef std::map<const ptx_instruction *, int> insn_count_map; /* [한국어] 명령어 포인터 → 인플라이트 카운트 맵 타입 별칭 */

  /*
   * [한국어]
   * add_count - 메모리 명령어를 인플라이트 목록에 추가 (또는 카운트 증가)
   *
   * @pInsn: 추가할 PTX 메모리 명령어
   * @count: 증가할 카운트 (기본 1) — 같은 명령어의 여러 워프가 동시에 인플라이트 가능
   */
  void add_count(const ptx_instruction *pInsn, int count = 1) {
    ptx_inflight_memory_insns[pInsn] += count; /* [한국어] 없으면 count로 초기화, 있으면 count만큼 증가 */
  }

  /*
   * [한국어]
   * sub_count - 완료된 메모리 명령어의 카운트 감소 (0 이하면 목록에서 제거)
   *
   * @pInsn: 완료된 PTX 메모리 명령어
   * @count: 감소할 카운트 (기본 1)
   *
   * assert로 pInsn이 맵에 있어야 함을 강제 — 등록되지 않은 명령어 완료는 버그.
   * 카운트가 0 이하가 되면 맵에서 완전히 제거.
   */
  void sub_count(const ptx_instruction *pInsn, int count = 1) {
    insn_count_map::iterator i_insncount;
    i_insncount = ptx_inflight_memory_insns.find(pInsn); /* [한국어] 인플라이트 맵에서 해당 명령어 탐색 */

    assert(i_insncount != ptx_inflight_memory_insns.end()); /* [한국어] 등록된 명령어여야 함 — 등록 없이 완료 불가 */

    i_insncount->second -= count; /* [한국어] 카운트 감소 */

    if (i_insncount->second <= 0) { /* [한국어] 카운트가 0 이하이면 완전히 완료 */
      ptx_inflight_memory_insns.erase(i_insncount); /* [한국어] 맵에서 제거 */
    }
  }

  /*
   * [한국어]
   * attribute_exposed_latency - 현재 인플라이트 명령어 모두에 exposed latency 귀속
   *
   * @count: 귀속할 버블 사이클 수 (기본 1)
   *
   * 파이프라인이 메모리 응답을 기다리며 빈 사이클이 발생했을 때 호출.
   * 현재 인플라이트 상태인 모든 명령어에 count 사이클씩 exposed_latency 증가.
   * 인플라이트 명령어가 없으면 귀속 생략.
   */
  void attribute_exposed_latency(int count = 1) {
    insn_count_map &exlat_insnmap = ptx_inflight_memory_insns; /* [한국어] 인플라이트 맵 참조 */
    insn_count_map::const_iterator i_exlatinsn;

    i_exlatinsn = exlat_insnmap.begin();
    for (; i_exlatinsn != exlat_insnmap.end(); ++i_exlatinsn) { /* [한국어] 모든 인플라이트 명령어 순회 */
      const ptx_instruction *pInsn = i_exlatinsn->first; /* [한국어] 현재 처리 중인 명령어 */
      ptx_file_line_stats &line_stats =
          ptx_file_line_stats_tracker[ptx_file_line(pInsn->source_file(),
                                                    pInsn->source_line())]; /* [한국어] 해당 PTX 라인 통계 참조 */
      line_stats.exposed_latency += count; /* [한국어] 파이프라인 버블 사이클을 이 명령어에 귀속 */
    }
  }

  insn_count_map ptx_inflight_memory_insns; /* [한국어] 현재 인플라이트 상태인 메모리 명령어 → 카운트 맵.
                                              * 설정자: add_count()에서 삽입/증가, sub_count()에서 감소/제거.
                                              * 읽는 자: attribute_exposed_latency()에서 순회.
                                              * 동기화: SM별 독립 추적기이므로 별도 락 불필요 (단일 스레드 시뮬레이션). */
};

static ptx_inflight_memory_insn_tracker *inflight_mem_tracker = NULL; /* [한국어] SM별 인플라이트 추적기 배열.
                                                                         * 설정자: ptx_file_line_stats_create_exposed_latency_tracker()에서 new[].
                                                                         * 읽는 자: add/sub_inflight_memory_insn, commit_exposed_latency.
                                                                         * 동기화: 초기화 후 SM별 독립 접근이므로 락 불필요.
                                                                         * 수명: 시뮬레이터 전체 수명 (해제 코드 없음). */

/*
 * [한국어]
 * ptx_file_line_stats_create_exposed_latency_tracker - SM별 인플라이트 추적기 배열 생성
 *
 * @n_shader_cores: 시뮬레이션할 SM 수 — 이 수만큼 추적기 객체 생성
 *
 * 시뮬레이터 초기화 시 한 번 호출. inflight_mem_tracker를 SM 수만큼 배열로 할당.
 * 이후 SC ID로 각 SM의 추적기에 접근.
 * 실행 컨텍스트: 시뮬레이터 초기화 단계.
 *
 * 호출 체인:
 *   gpu-sim.cc (시뮬레이터 초기화) → [이 함수]
 */
void ptx_file_line_stats_create_exposed_latency_tracker(int n_shader_cores) {
  inflight_mem_tracker = new ptx_inflight_memory_insn_tracker[n_shader_cores]; /* [한국어] SM 수만큼 추적기 배열 동적 할당 */
}

/*
 * [한국어]
 * ptx_stats::ptx_file_line_stats_add_inflight_memory_insn - 인플라이트 메모리 명령어 등록
 *
 * @sc_id: shader core(SM) ID
 * @pc: 인플라이트로 등록할 메모리 명령어의 PC
 *
 * 메모리 명령어가 L1 캐시 miss 등으로 파이프라인에서 대기 상태로 진입할 때 호출.
 * 실행 컨텍스트: 타이밍 시뮬레이션 (shader.cc 명령어 파이프라인 진입 시).
 *
 * 호출 체인:
 *   gpgpu-sim/shader.cc → [이 함수] → ptx_inflight_memory_insn_tracker::add_count
 */
// add an inflight memory instruction
void ptx_stats::ptx_file_line_stats_add_inflight_memory_insn(int sc_id,
                                                             unsigned pc) {
  const ptx_instruction *pInsn = gpgpu_ctx->pc_to_instruction(pc); /* [한국어] PC로 PTX 명령어 객체 조회 */

  inflight_mem_tracker[sc_id].add_count(pInsn); /* [한국어] 해당 SM의 인플라이트 추적기에 명령어 등록 */
}

/*
 * [한국어]
 * ptx_stats::ptx_file_line_stats_sub_inflight_memory_insn - 인플라이트 메모리 명령어 제거
 *
 * @sc_id: shader core(SM) ID
 * @pc: 완료되어 제거할 메모리 명령어의 PC
 *
 * 메모리 응답을 받아 명령어가 파이프라인 대기 상태를 벗어날 때 호출.
 * 실행 컨텍스트: 타이밍 시뮬레이션 (shader.cc 메모리 응답 처리 시).
 *
 * 호출 체인:
 *   gpgpu-sim/shader.cc → [이 함수] → ptx_inflight_memory_insn_tracker::sub_count
 */
// remove an inflight memory instruction
void ptx_stats::ptx_file_line_stats_sub_inflight_memory_insn(int sc_id,
                                                             unsigned pc) {
  const ptx_instruction *pInsn = gpgpu_ctx->pc_to_instruction(pc); /* [한국어] PC로 PTX 명령어 객체 조회 */

  inflight_mem_tracker[sc_id].sub_count(pInsn); /* [한국어] 해당 SM의 인플라이트 추적기에서 명령어 제거 */
}

/*
 * [한국어]
 * ptx_file_line_stats_commit_exposed_latency - 파이프라인 버블을 인플라이트 명령어에 귀속
 *
 * @sc_id: shader core(SM) ID
 * @exposed_latency: 이 SM에서 발생한 버블 사이클 수 (반드시 > 0)
 *
 * 파이프라인이 메모리 응답을 기다리며 비어있는 사이클이 발생했을 때 호출.
 * 해당 SM의 모든 인플라이트 메모리 명령어에 exposed_latency 사이클 귀속.
 * 실행 컨텍스트: 타이밍 시뮬레이션 (shader.cc 파이프라인 버블 감지 시).
 *
 * 호출 체인:
 *   gpgpu-sim/shader.cc (파이프라인 버블 발생) → [이 함수]
 *     → ptx_inflight_memory_insn_tracker::attribute_exposed_latency
 */
// attribute an empty cycle in the pipeline (exposed latency) to the ptx memory
// instructions in flight
void ptx_file_line_stats_commit_exposed_latency(int sc_id,
                                                int exposed_latency) {
  assert(exposed_latency > 0); /* [한국어] 0 이하의 버블 사이클은 논리 오류 — 방어적 검사 */
  inflight_mem_tracker[sc_id].attribute_exposed_latency(exposed_latency); /* [한국어] 해당 SM의 인플라이트 명령어에 버블 사이클 귀속 */
}

/*
 * [한국어]
 * ptx_stats::ptx_file_line_stats_add_warp_divergence - 워프 분기 횟수 귀속
 *
 * @pc: 분기가 발생한 PTX 명령어의 PC
 * @n_way_divergence: 분기 방향 수 (2이면 2-way, 값이 클수록 더 많이 갈라짐)
 *
 * 조건 분기 명령어(bra 등)에서 워프 내 스레드가 서로 다른 방향으로 갈라질 때 호출.
 * 워프 분기는 SIMT 효율 저하의 주요 원인이므로 이를 PTX 라인별로 추적.
 * 실행 컨텍스트: 타이밍 시뮬레이션 (shader.cc SIMT stack 처리 중).
 *
 * 호출 체인:
 *   gpgpu-sim/shader.cc (SIMT divergence 감지) → [이 함수]
 */
// attribute the number of warp divergence to a ptx instruction
void ptx_stats::ptx_file_line_stats_add_warp_divergence(
    unsigned pc, unsigned n_way_divergence) {
  const ptx_instruction *pInsn = gpgpu_ctx->pc_to_instruction(pc); /* [한국어] PC로 PTX 명령어 객체 조회 */

  ptx_file_line_stats &line_stats = ptx_file_line_stats_tracker[ptx_file_line(
      pInsn->source_file(), pInsn->source_line())]; /* [한국어] 해당 PTX 라인의 통계 참조 (없으면 자동 생성) */
  line_stats.warp_divergence += n_way_divergence; /* [한국어] 분기 방향 수 누적 */
}
