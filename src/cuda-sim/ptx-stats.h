/*
 * [한국어 설명] PTX 소스 라인 단위 실행 통계 인터페이스 (ptx-stats.h)
 *
 * === 파일의 역할 ===
 * 기능 시뮬레이션(cuda-sim)과 타이밍 시뮬레이션(gpgpu-sim) 양쪽에서 PTX 소스 라인별로
 * 실행 횟수, 파이프라인 지연, DRAM 트래픽, 공유 메모리 뱅크 충돌, 비정렬 전역 메모리 접근,
 * 워프 분기 횟수 등 상세 성능 통계를 수집하는 인터페이스를 정의한다.
 * 시뮬레이션 종료 후 ptx_line_stats_filename이 가리키는 파일에 PTX 소스 라인별 통계를 출력하여
 * 애플리케이션 병목 지점을 PTX 소스 수준에서 분석할 수 있게 한다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름 (통계 수집): instructions.cc (PTX 명령어 실행) → ptx_file_line_stats_add_exec_count()
 *   shader.cc (SM 파이프라인) → ptx_file_line_stats_add_latency/dram_traffic/smem 등
 * 실행 흐름 (출력): gpu-sim.cc 시뮬레이션 완료 → ptx_file_line_stats_write_file()
 * 실행 컨텍스트: 기능 시뮬레이션(exec_count) 및 타이밍 시뮬레이션(나머지) 호스트 CPU.
 *
 * === 타 모듈과의 연결 ===
 * 의존: option_parser.h (옵션 등록), libcuda/gpgpu_context.h (pc_to_instruction)
 * 의존받음: cuda-sim/instructions.cc (add_exec_count),
 *   gpgpu-sim/shader.cc (add_latency, add_smem_bank_conflict, add_uncoalesced_gmem,
 *     add_inflight_memory_insn/sub, add_warp_divergence),
 *   gpgpu-sim/gpu-sim.cc (add_dram_traffic, write_file)
 * 공유 자료구조: 내부 정적 ptx_file_line_stats_tracker (파일+줄번호 → 통계 맵)
 *
 * === 주요 함수/구조체 요약 ===
 * ptx_stats 클래스: 옵션 및 통계 수집 메서드 집합
 * ptx_file_line_stats_options(): -enable_ptx_file_line_stats, -ptx_line_stats_filename 등록
 * ptx_file_line_stats_write_file(): 수집된 통계를 파일로 출력
 * ptx_file_line_stats_add_exec_count(): 명령어 실행 횟수 증가 (기능 시뮬레이션용 자유 함수)
 * ptx_file_line_stats_commit_exposed_latency(): 파이프라인 버블 지연을 인플라이트 명령어에 귀속
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

#pragma once /* [한국어] 헤더 중복 포함 방지 — #ifndef 대신 #pragma once 사용 */

#include "../option_parser.h" /* [한국어] option_parser_t — ptx_file_line_stats_options에서 옵션 등록 */

#ifdef __cplusplus
// stat collection interface to cuda-sim
/* [한국어] C++ 환경(기능 시뮬레이션)에서만 사용하는 실행 횟수 수집 인터페이스 */
class ptx_instruction; /* [한국어] 전방 선언 — ptx_instruction::source_file/source_line() 접근 */
/*
 * [한국어]
 * ptx_file_line_stats_add_exec_count - PTX 명령어 실행 횟수 증가 (자유 함수)
 *
 * @pInsn: 실행된 PTX 명령어 포인터 — source_file/source_line으로 통계 키 결정
 *
 * 기능 시뮬레이션(instructions.cc)에서 PTX 명령어가 실행될 때마다 호출.
 * ptx_stats 멤버가 아닌 자유 함수로, C++ 링크에서만 사용 가능.
 * 내부적으로 ptx_file_line_stats_tracker의 exec_count를 1 증가.
 */
void ptx_file_line_stats_add_exec_count(const ptx_instruction* pInsn);
#endif

// stat collection interface to gpgpu-sim
/* [한국어] 타이밍 시뮬레이션(gpgpu-sim)에서 호출하는 인터페이스 — C/C++ 공용 */

/*
 * [한국어]
 * ptx_file_line_stats_create_exposed_latency_tracker - SM별 인플라이트 명령어 추적기 생성
 *
 * @n_shader_cores: 시뮬레이션할 SM(shader core) 수 — SM마다 독립 추적기 필요
 *
 * ptx_inflight_memory_insn_tracker 배열을 SM 수만큼 동적 할당.
 * 시뮬레이터 초기화 시 한 번 호출 — 이후 SM별로 인플라이트 메모리 명령어 관리.
 */
void ptx_file_line_stats_create_exposed_latency_tracker(int n_shader_cores);
/*
 * [한국어]
 * ptx_file_line_stats_commit_exposed_latency - 파이프라인 버블 지연을 인플라이트 명령어에 귀속
 *
 * @sc_id: shader core(SM) ID
 * @exposed_latency: 이 SM에서 발생한 파이프라인 버블 사이클 수
 *
 * 메모리 스톨로 파이프라인이 멈춘 사이클을 현재 인플라이트 메모리 명령어에 균등 귀속.
 * shader.cc에서 메모리 응답 대기 버블이 발생할 때마다 호출.
 */
void ptx_file_line_stats_commit_exposed_latency(int sc_id, int exposed_latency);

class gpgpu_context; /* [한국어] 전방 선언 — gpgpu_context::pc_to_instruction() 접근 */

/*
 * [한국어]
 * ptx_stats - PTX 소스 라인별 통계 수집 및 출력 클래스
 *
 * gpgpu_context의 멤버로 존재하며, 시뮬레이션 전체에서 PTX 소스 라인별 성능 지표를
 * 수집하고 시뮬레이션 종료 후 파일로 출력한다. 수집 항목: 실행 횟수, 파이프라인 지연,
 * DRAM 트래픽, 공유 메모리 뱅크 충돌, 비정렬 전역 메모리 접근, 인플라이트 명령어,
 * 워프 분기 횟수.
 * 실행 컨텍스트: 기능/타이밍 시뮬레이션 모두에서 호출됨 (호스트 CPU 단일 스레드).
 */
class ptx_stats {
 public:
  /*
   * [한국어] 생성자 — ptx_line_stats_filename과 gpgpu_ctx 초기화
   * @ctx: 상위 gpgpu_context 역방향 포인터
   */
  ptx_stats(gpgpu_context* ctx) {
    ptx_line_stats_filename = NULL; /* [한국어] 파일명은 option_parser에서 설정 전까지 NULL */
    gpgpu_ctx = ctx;                /* [한국어] pc_to_instruction() 접근을 위한 gpgpu_context 저장 */
  }
  char* ptx_line_stats_filename;   /* [한국어] 통계 출력 파일 경로 (-ptx_line_stats_filename 옵션).
                                     * 설정자: ptx_file_line_stats_options()에서 option_parser가 설정.
                                     * 읽는 자: ptx_file_line_stats_write_file()에서 fopen.
                                     * 기본값: "gpgpu_inst_stats.txt".
                                     * 동기화: 옵션 파싱 단계에서 설정 후 불변. */
  bool enable_ptx_file_line_stats; /* [한국어] 통계 수집 활성화 플래그 (-enable_ptx_file_line_stats).
                                     * 설정자: ptx_file_line_stats_options()에서 option_parser가 설정.
                                     * 읽는 자: 각 add_* 함수에서 수집 전 확인, write_file에서 출력 전 확인.
                                     * 기본값: 1(활성화).
                                     * 동기화: 옵션 파싱 후 불변. */
  gpgpu_context* gpgpu_ctx;        /* [한국어] 상위 gpgpu_context 역방향 포인터.
                                     * 설정자: 생성자에서 초기화.
                                     * 읽는 자: 각 add_* 함수에서 pc_to_instruction() 호출.
                                     * 동기화: 생성 후 불변. */
  // set options
  /*
   * [한국어]
   * ptx_file_line_stats_options - PTX 통계 관련 옵션 등록
   *
   * @opp: option_parser 핸들
   *
   * -enable_ptx_file_line_stats(기본 1), -ptx_line_stats_filename(기본 "gpgpu_inst_stats.txt") 등록.
   */
  void ptx_file_line_stats_options(option_parser_t opp);

  // output stats to a file
  /*
   * [한국어]
   * ptx_file_line_stats_write_file - 수집된 통계를 파일로 출력
   *
   * ptx_line_stats_filename에 지정된 파일을 열어 ptx_file_line_stats_tracker의
   * 모든 항목을 "파일명 줄번호 : 카운트들..." 형식으로 출력.
   * 시뮬레이션 완료 후 gpu-sim.cc에서 호출.
   */
  void ptx_file_line_stats_write_file();
  // stat collection interface to gpgpu-sim
  /*
   * [한국어]
   * ptx_file_line_stats_add_latency - PC에 대응하는 PTX 라인에 파이프라인 지연 귀속
   *
   * @pc: 프로그램 카운터 — pc_to_instruction으로 PTX 라인 조회
   * @latency: 귀속할 사이클 수
   */
  void ptx_file_line_stats_add_latency(unsigned pc, unsigned latency);
  /*
   * [한국어]
   * ptx_file_line_stats_add_dram_traffic - PC에 대응하는 PTX 라인에 DRAM 트래픽 귀속
   *
   * @pc: 프로그램 카운터
   * @dram_traffic: DRAM 요청 수
   */
  void ptx_file_line_stats_add_dram_traffic(unsigned pc, unsigned dram_traffic);
  /*
   * [한국어]
   * ptx_file_line_stats_add_smem_bank_conflict - 공유 메모리 뱅크 충돌 귀속
   *
   * @pc: 프로그램 카운터
   * @n_way_bkconflict: n-way 뱅크 충돌 횟수 (충돌한 뱅크 수)
   */
  void ptx_file_line_stats_add_smem_bank_conflict(unsigned pc,
                                                  unsigned n_way_bkconflict);
  /*
   * [한국어]
   * ptx_file_line_stats_add_uncoalesced_gmem - 비정렬 전역 메모리 접근 귀속
   *
   * @pc: 프로그램 카운터
   * @n_access: 비정렬로 인해 추가 생성된 메모리 요청 수
   */
  void ptx_file_line_stats_add_uncoalesced_gmem(unsigned pc, unsigned n_access);
  /*
   * [한국어]
   * ptx_file_line_stats_add_inflight_memory_insn - 인플라이트 메모리 명령어 등록
   *
   * @sc_id: shader core(SM) ID
   * @pc: 인플라이트 상태로 추가할 메모리 명령어의 PC
   */
  void ptx_file_line_stats_add_inflight_memory_insn(int sc_id, unsigned pc);
  /*
   * [한국어]
   * ptx_file_line_stats_sub_inflight_memory_insn - 인플라이트 메모리 명령어 제거
   *
   * @sc_id: shader core(SM) ID
   * @pc: 완료되어 제거할 메모리 명령어의 PC
   */
  void ptx_file_line_stats_sub_inflight_memory_insn(int sc_id, unsigned pc);
  /*
   * [한국어]
   * ptx_file_line_stats_add_warp_divergence - 워프 분기 발생 횟수 귀속
   *
   * @pc: 분기가 발생한 PTX 명령어의 PC
   * @n_way_divergence: 분기 방향 수 (2이면 2-way 분기)
   */
  void ptx_file_line_stats_add_warp_divergence(unsigned pc,
                                               unsigned n_way_divergence);
};
