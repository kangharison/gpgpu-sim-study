/*
 * [한국어 설명] GPGPU-Sim 하드웨어 추상 모델 구현 (abstract_hardware_model.cc)
 *
 * === 파일의 역할 ===
 * 이 파일은 GPGPU-Sim 타이밍 모델 전체의 기반이 되는 핵심 하드웨어 추상 객체들의
 * 구현을 담는다. warp 실행 명령어(warp_inst_t), SIMT 스택(simt_stack), 커널 실행
 * 정보(kernel_info_t), SM 추상 인터페이스(core_t), GPU 최상위 객체(gpgpu_t) 등
 * GPU 마이크로아키텍처를 모델링하는 데 필요한 핵심 구조체들의 메서드를 구현한다.
 * 특히 warp 메모리 접근 coalescing(32스레드 주소 병합), SIMT 분기/재합류 처리,
 * 체크포인트 저장·복구 기능이 이 파일에서 구현된다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 실행 흐름:
 *   CUDA 애플리케이션
 *     → libcuda 인터셉트 (libcuda/)
 *         → gpgpusim_entrypoint.cc (시뮬레이터 진입)
 *             → cuda-sim/ (PTX 기능 시뮬레이션)
 *             → [이 파일] (warp/커널/SIMT 추상 객체 메서드)
 *                 ↑ gpgpu-sim/shader.cc (타이밍 시뮬)이 이 객체들을 사용
 * 이 파일은 타이밍 모델(gpgpu-sim/)과 기능 모델(cuda-sim/) 사이의 다리 역할을
 * 하며, 두 모델이 공유하는 핵심 데이터 구조의 동작을 정의한다.
 * 실행 컨텍스트: 단일 시뮬레이션 스레드에서 호출됨 (cycle-by-cycle 루프).
 *
 * === 타 모듈과의 연결 ===
 * 의존하는 모듈:
 *   - cuda-sim/memory.h: memory_space_impl (전역/텍스처/공유 메모리 공간)
 *   - cuda-sim/ptx_ir.h: function_info (PTX 함수 엔트리), ptx_exec_inst()
 *   - cuda-sim/ptx-stats.h: ptx_file_line_stats_* (사이클/발산/지연 통계)
 *   - gpgpu-sim/gpu-sim.h: gpgpu_sim (최상위 시뮬레이터, cycle 카운터 제공)
 *   - option_parser.h: OptionParser (설정 파싱)
 *   - libcuda/gpgpu_context.h: gpgpu_context (전역 시뮬레이터 컨텍스트)
 * 이 파일에 의존하는 모듈:
 *   - gpgpu-sim/shader.cc: shader_core_ctx가 core_t를 상속, warp_inst_t 사용
 *   - gpgpu-sim/gpu-cache.cc: mem_access_t를 캐시 요청으로 소비
 *   - gpgpusim_entrypoint.cc: kernel_info_t를 통해 커널 실행 관리
 * 핵심 공유 자료구조:
 *   - warp_inst_t: 워프 단위 명령어 패킷 (active_mask + 스레드별 메모리 주소)
 *   - simt_stack: pdom(post-dominator) 기반 분기 처리 스택
 *   - kernel_info_t: 실행 중인 CUDA 커널의 그리드/블록/스레드 상태
 *
 * === 주요 함수/구조체 요약 ===
 * - warp_inst_t::generate_mem_accesses(): 32스레드 메모리 주소를 coalesce하여
 *   m_accessq에 mem_access_t 객체로 변환 (shared/global/local/tex/const 처리)
 * - warp_inst_t::memory_coalescing_arch(): Fermi/Kepler/Maxwell/Pascal/Volta
 *   아키텍처별 coalescing 규칙을 적용, subwarp 단위로 트랜잭션 생성
 * - simt_stack::update(): warp 분기 시 pdom 스택에 새 엔트리 push/pop 처리,
 *   CALL/RET/분기/재합류를 모두 통합 처리하는 핵심 함수
 * - kernel_info_t::kernel_info_t(): 커널 실행 시 그리드/블록 초기화,
 *   CDP(CUDA Dynamic Parallelism) parent/child 관계 설정 포함
 * - core_t::updateSIMTStack(): 명령어 실행 후 각 스레드의 next PC로
 *   SIMT 스택을 갱신하는 함수 (타이밍 모델 → 기능 모델 연결 지점)
 * - gpgpu_t::gpgpu_t(): GPU 최상위 객체 초기화, 전역/텍스처/서피스 메모리 공간 생성
 */

// Copyright (c) 2009-2021, Tor M. Aamodt, Inderpreet Singh, Timothy Rogers,
// Vijay Kandiah, Nikos Hardavellas, Mahmoud Khairy, Junrui Pan, Timothy G.
// Rogers The University of British Columbia, Northwestern University, Purdue
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

#include "abstract_hardware_model.h" /* [한국어] 이 파일이 구현하는 클래스/구조체 선언 (warp_inst_t, simt_stack, kernel_info_t, core_t, gpgpu_t 등) */
#include <sys/stat.h>   /* [한국어] stat(), mkdir() — 체크포인트 디렉토리 존재 확인 및 생성에 사용 */
#include <algorithm>    /* [한국어] std::find, std::max — child kernel 리스트 탐색, bank 접근 최댓값 계산에 사용 */
#include <iostream>     /* [한국어] std::cout/cerr — 디버그 출력 (직접 사용 빈도는 낮으나 포함) */
#include <sstream>      /* [한국어] std::stringstream — 체크포인트 파일에서 hex 문자열을 정수로 변환 시 사용 */
#include "../libcuda/gpgpu_context.h" /* [한국어] gpgpu_context — 전역 시뮬레이터 컨텍스트, uid 카운터·통계·device_runtime 접근 */
#include "cuda-sim/cuda-sim.h"        /* [한국어] ptx_fetch_inst() — PC로 PTX 명령어 객체를 가져오는 기능 시뮬레이션 인터페이스 */
#include "cuda-sim/memory.h"          /* [한국어] memory_space_impl<> — 전역/파라미터/텍스처 메모리 공간 구현체 */
#include "cuda-sim/ptx-stats.h"       /* [한국어] ptx_file_line_stats_* — 파일/라인별 사이클·발산·latency 통계 수집 */
#include "cuda-sim/ptx_ir.h"          /* [한국어] function_info — PTX 함수 엔트리, 파라미터 크기·이름 제공 */
#include "gpgpu-sim/gpu-sim.h"        /* [한국어] gpgpu_sim — 타이밍 시뮬레이터 최상위, gpu_sim_cycle 등 사이클 카운터 */
#include "gpgpusim_entrypoint.h"      /* [한국어] the_gpgpusim → g_stream_manager — CDP 스트림 등록/해제 */
#include "option_parser.h"            /* [한국어] option_parser_register() — gpgpusim.config 옵션 등록 */

/*
 * [한국어]
 * mem_access_t::init - 메모리 접근 요청 객체 초기화
 *
 * @ctx: 전역 시뮬레이터 컨텍스트 포인터. uid 발급 카운터(sm_next_access_uid)를
 *       보관하며, 이 함수는 그 카운터를 원자적으로 증가시켜 uid를 획득한다.
 * @return: 없음.
 *
 * generate_mem_accesses()나 memory_coalescing_arch*() 내에서 coalescing 완료 후
 * mem_access_t 객체를 m_accessq에 push_back하기 전에 호출된다.
 * uid는 시뮬레이터 전체에서 메모리 요청을 고유 식별하는 데 사용된다.
 * 실행 컨텍스트: 단일 시뮬레이션 스레드에서 호출되므로 별도 락 불필요.
 *
 * 호출 체인:
 *   mem_access_t 생성자 (abstract_hardware_model.h) → [이 함수]
 */
void mem_access_t::init(gpgpu_context *ctx) {
  gpgpu_ctx = ctx; /* [한국어] 전역 컨텍스트 포인터 저장 — uid 발급 및 통계 수집에 사용 */
  m_uid = ++(gpgpu_ctx->sm_next_access_uid); /* [한국어] 전역 uid 카운터를 1 증가시켜 이 메모리 요청의 고유 ID를 부여 */
  m_addr = 0;     /* [한국어] 대상 메모리 주소를 0으로 초기화 (이후 caller가 설정) */
  m_req_size = 0; /* [한국어] 요청 바이트 수를 0으로 초기화 (이후 caller가 캐시라인/세그먼트 크기로 설정) */
}

/*
 * [한국어]
 * warp_inst_t::issue - warp 명령어를 issue 단계에서 활성화
 *
 * @mask: 이 명령어가 실행될 스레드 레인의 활성 마스크 (32비트 bitset).
 *        warp scheduler가 SIMT 스택 top에서 가져온 active_mask를 전달한다.
 * @warp_id: SM 내 정적 warp 번호 (0 ~ num_warps-1).
 * @cycle: 현재 시뮬레이션 사이클 번호. issue_cycle 기록에 사용.
 * @dynamic_warp_id: 커널 실행 중 동적으로 부여된 warp 식별자 (통계·디버그 용).
 * @sch_id: 이 명령어를 발행한 warp scheduler의 인덱스.
 * @streamID: 이 명령어가 속한 CUDA 스트림 ID.
 * @return: 없음.
 *
 * warp scheduler가 ready 상태의 warp를 선택하고, 해당 명령어를 issue 큐에
 * 넣기 전에 호출한다. m_empty를 false로 설정하여 파이프라인 스테이지가
 * 이 슬롯을 "유효한 명령어"로 인식하게 한다.
 * initiation_interval을 cycles에 복사하여 functional unit의 점유 사이클을 설정.
 * 실행 컨텍스트: 타이밍 시뮬 사이클 루프 내 warp scheduler 단계에서 호출.
 *
 * 호출 체인:
 *   shader_core_ctx::issue_warp() → [이 함수]
 */
void warp_inst_t::issue(const active_mask_t &mask, unsigned warp_id,
                        unsigned long long cycle, int dynamic_warp_id,
                        int sch_id, unsigned long long streamID) {
  m_warp_active_mask = mask;      /* [한국어] 현재 issue되는 active_mask를 저장 — 실행 중 비활성 스레드 추적에 사용 */
  m_warp_issued_mask = mask;      /* [한국어] issue 시점의 마스크를 별도 보존 — 나중에 issued 마스크 참조 시 사용 */
  m_uid = ++(m_config->gpgpu_ctx->warp_inst_sm_next_uid); /* [한국어] 전역 warp 명령어 uid 카운터 증가 — 이 명령어 고유 식별 */
  m_streamID = streamID;          /* [한국어] 이 명령어가 속한 CUDA 스트림 ID 기록 */
  m_warp_id = warp_id;            /* [한국어] SM 내 정적 warp 번호 저장 */
  m_dynamic_warp_id = dynamic_warp_id; /* [한국어] 동적 warp 식별자 저장 (통계용) */
  issue_cycle = cycle;            /* [한국어] issue 사이클을 기록 — completed() 에서 latency 계산에 사용 */
  cycles = initiation_interval;   /* [한국어] functional unit 점유 사이클 초기화 (initiation_interval = 명령어 발행 간격) */
  m_cache_hit = false;            /* [한국어] 캐시 히트 여부 초기화 — L1D 접근 후 hit 여부에 따라 업데이트됨 */
  m_empty = false;                /* [한국어] 이 슬롯이 유효한 명령어임을 표시 — 파이프라인 스테이지가 비어있는지 검사 시 사용 */
  m_scheduler_id = sch_id;        /* [한국어] 발행한 warp scheduler 인덱스 저장 (다중 스케줄러 환경에서 통계 분리 목적) */
}

/*
 * [한국어]
 * checkpoint::checkpoint - 체크포인트 디렉토리 초기화
 *
 * @return: 없음.
 *
 * 시뮬레이터가 체크포인트 모드로 실행될 때 checkpoint 객체를 생성하면서
 * 호출된다. "checkpoint_files" 디렉토리가 없으면 새로 생성한다.
 * stat()으로 디렉토리 존재 여부를 먼저 확인하고, mkdir()로 생성한다.
 * 실행 컨텍스트: 시뮬레이터 초기화 단계 (단일 스레드).
 *
 * 호출 체인:
 *   gpgpusim_entrypoint / shader_core_ctx 초기화 → [이 함수]
 */
checkpoint::checkpoint() {
  struct stat st = {0}; /* [한국어] stat 구조체 초기화 — 파일/디렉토리 속성 조회에 사용 */

  if (stat("checkpoint_files", &st) == -1) { /* [한국어] "checkpoint_files" 디렉토리가 존재하지 않으면 (-1 반환) */
    mkdir("checkpoint_files", 0777); /* [한국어] 체크포인트 저장 디렉토리 생성 (모든 권한 허용) */
  }
}

/*
 * [한국어]
 * checkpoint::load_global_mem - 파일에서 전역 메모리 상태를 복원
 *
 * @temp_mem: 데이터를 복원할 메모리 공간 객체. memory_space 인터페이스를 통해
 *            특정 주소·오프셋에 데이터를 기록한다.
 * @f1name: 복원할 체크포인트 파일 경로. 텍스트 형식이며 store_global_mem()이
 *          저장한 포맷과 동일해야 한다.
 * @return: 없음. 파일 오픈 실패 시 assert로 중단.
 *
 * 체크포인트 파일을 한 줄씩 읽어 전역 메모리를 복원한다.
 * 파일 포맷:
 *   - 첫 토큰이 'g'/'s'/'l'로 시작하면 새로운 메모리 섹션 시작 (hex 주소가 뒤따름)
 *   - 그 외 라인은 4바이트 데이터 (hex 값) — 해당 index+offset에 기록
 * 실행 컨텍스트: 시뮬레이터 resume 초기화 단계.
 *
 * 호출 체인:
 *   shader_core_ctx::resume_kernel() → [이 함수]
 */
void checkpoint::load_global_mem(class memory_space *temp_mem, char *f1name) {
  FILE *fp2 = fopen(f1name, "r"); /* [한국어] 체크포인트 파일을 읽기 모드로 오픈 */
  assert(fp2 != NULL);            /* [한국어] 파일 오픈 실패 시 즉시 중단 */
  char line[128]; /* or other suitable maximum line size */
  unsigned int offset = 0; /* [한국어] 현재 섹션 내 바이트 오프셋 — 섹션 헤더 라인마다 0으로 초기화 */
  while (fgets(line, sizeof line, fp2) != NULL) /* read a line */
  {
    unsigned int index; /* [한국어] 현재 섹션의 기준 주소 (hex → 정수 변환된 값) */
    char *pch;          /* [한국어] strtok로 토큰화된 문자열 포인터 */
    pch = strtok(line, " "); /* [한국어] 라인의 첫 번째 토큰 추출 (공백 구분자) */
    if (pch[0] == 'g' || pch[0] == 's' || pch[0] == 'l') {
      /* [한국어] 섹션 헤더 라인: 'g'=global, 's'=shared, 'l'=local 메모리 기준 주소 */
      pch = strtok(NULL, " "); /* [한국어] 다음 토큰 = 섹션의 기준 주소 (hex 문자열) */

      std::stringstream ss; /* [한국어] hex 문자열을 정수로 변환하기 위한 stringstream */
      ss << std::hex << pch; /* [한국어] hex 형식으로 stringstream에 삽입 */
      ss >> index;           /* [한국어] 정수 index로 추출 (기준 주소) */

      offset = 0; /* [한국어] 새 섹션 시작 — 오프셋 리셋 */
    } else {
      /* [한국어] 데이터 라인: 4바이트 hex 값 */
      unsigned int data;    /* [한국어] 읽어온 4바이트 데이터 */
      std::stringstream ss; /* [한국어] hex 문자열 → 정수 변환용 */
      ss << std::hex << pch; /* [한국어] hex 형식으로 삽입 */
      ss >> data;            /* [한국어] 정수값으로 추출 */
      temp_mem->write_only(offset, index, 4, &data); /* [한국어] 기준주소(index)+오프셋 위치에 4바이트 기록 */
      offset = offset + 4; /* [한국어] 다음 4바이트로 오프셋 전진 */
    }
    // fputs ( line, stdout ); /* write the line */
  }
  fclose(fp2); /* [한국어] 파일 핸들 반환 */
}

/*
 * [한국어]
 * checkpoint::store_global_mem - 전역 메모리 상태를 파일로 저장
 *
 * @mem: 저장할 메모리 공간 객체. memory_space::print()를 통해 파일에 출력.
 * @fname: 저장 파일 경로.
 * @format: memory_space::print()에 전달되는 출력 형식 문자열.
 * @return: 없음. 파일 오픈 실패 시 assert로 중단.
 *
 * 체크포인트 시 전역 메모리 내용을 텍스트 형식으로 파일에 기록한다.
 * load_global_mem()과 쌍을 이루며, 동일한 format 인자로 저장/복원이 가능하다.
 * 실행 컨텍스트: 시뮬레이터 체크포인트 저장 단계.
 *
 * 호출 체인:
 *   shader_core_ctx::store_checkpoint() → [이 함수]
 */
void checkpoint::store_global_mem(class memory_space *mem, char *fname,
                                  char *format) {
  FILE *fp3 = fopen(fname, "w"); /* [한국어] 체크포인트 파일을 쓰기 모드로 오픈 */
  assert(fp3 != NULL);           /* [한국어] 파일 생성 실패 시 즉시 중단 */
  mem->print(format, fp3);       /* [한국어] 메모리 공간 전체를 지정 형식으로 파일에 출력 */
  fclose(fp3);                   /* [한국어] 파일 핸들 반환 */
}

/*
 * [한국어]
 * move_warp - 파이프라인 스테이지 간 warp_inst_t 포인터 이동 (스왑)
 *
 * @dst: 목적지 스테이지 슬롯 포인터의 참조. 반드시 empty() 상태여야 한다.
 * @src: 출처 스테이지 슬롯 포인터의 참조. 이동 후 clear()되어 빈 상태로 전환.
 * @return: 없음.
 *
 * 파이프라인 레지스터 간에 명령어를 전달할 때 사용하는 "swap 후 src 초기화" 패턴.
 * 새 객체를 생성하지 않고 포인터만 교환하므로 O(1) 이동이 가능하다.
 * dst가 비어 있어야 하는 이유: 파이프라인의 back-pressure를 체크하기 위함
 * (dst가 아직 사용 중이면 이동 불가 → structural hazard 발생 시 stall).
 * 이동 후 src에는 기존 dst 객체(빈 슬롯)가 들어가고, clear()로 명시적으로
 * 초기화된다 — 다음 사이클에 새 명령어를 받을 준비.
 * 실행 컨텍스트: 타이밍 시뮬 사이클 루프 내 파이프라인 advance 단계.
 *
 * 호출 체인:
 *   shader_core_ctx::issue() / execute() / writeback() → [이 함수]
 */
void move_warp(warp_inst_t *&dst, warp_inst_t *&src) {
  assert(dst->empty()); /* [한국어] dst가 비어있지 않으면 structural hazard — 이동 불가이므로 중단 */
  warp_inst_t *temp = dst; /* [한국어] dst 포인터를 임시 보관 (swap을 위해) */
  dst = src;               /* [한국어] src 명령어를 dst 슬롯으로 이동 */
  src = temp;              /* [한국어] 기존 dst(빈 슬롯)를 src 위치에 복원 */
  src->clear();            /* [한국어] src 슬롯을 명시적으로 초기화 — 다음 사이클 재사용 준비 */
}

/*
 * [한국어]
 * gpgpu_functional_sim_config::reg_options - 기능 시뮬레이션 설정 옵션 등록
 *
 * @opp: OptionParser 객체 포인터. 각 옵션의 이름·타입·대상 변수·설명·기본값을
 *       등록하며, 이후 gpgpusim.config 파일 파싱 시 해당 변수에 값이 채워진다.
 * @return: 없음.
 *
 * gpgpusim.config(또는 명령행 인자)에서 기능 시뮬레이션과 체크포인트 관련
 * 옵션들을 OptionParser에 등록한다. 등록 후 option_parser_cmdline()이 호출되면
 * 해당 멤버 변수들이 설정 파일 값으로 채워진다.
 * 주요 설정 그룹:
 *   1) PTX 추출 방법 (cuobjdump 사용 여부, PTXPlus 변환 여부)
 *   2) 체크포인트/resume 제어 (어느 커널·CTA에서 체크포인트 저장/복원)
 *   3) 디버그 명령어 덤프 (파일 출력, 특정 스레드 UID 지정)
 * 실행 컨텍스트: 시뮬레이터 시작 시 단 1회 호출.
 *
 * 호출 체인:
 *   gpgpusim_entrypoint.cc:GPGPUsim_Init() → [이 함수]
 */
void gpgpu_functional_sim_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_ptx_use_cuobjdump", OPT_BOOL,
                         &m_ptx_use_cuobjdump,
                         /* [한국어] cuobjdump로 바이너리에서 PTX/SASS를 추출할지 여부.
                          * CUDA 4.0 이상에서는 기본 활성화(1). 이를 통해 실제 GPU 바이너리의
                          * PTX를 추출해 기능 시뮬레이션에 사용한다. */
                         "Use cuobjdump to extract ptx and sass from binaries",
#if (CUDART_VERSION >= 4000)
                         "1"  /* [한국어] CUDA 4.0 이상: cuobjdump 사용 기본값 = 활성화 */
#else
                         "0"  /* [한국어] CUDA 4.0 미만: cuobjdump 미지원이므로 비활성화 */
#endif
  );
  option_parser_register(opp, "-gpgpu_experimental_lib_support", OPT_BOOL,
                         &m_experimental_lib_support,
                         /* [한국어] CUDA 라이브러리에서 코드 추출 시도 여부 (실험적, 현재 broken).
                          * cudaGetExportTable 미구현으로 사용 불가 상태. */
                         "Try to extract code from cuda libraries [Broken "
                         "because of unknown cudaGetExportTable]",
                         "0"); /* [한국어] 기본값: 비활성화 */
  option_parser_register(opp, "-checkpoint_option", OPT_INT32,
                         &checkpoint_option,
                         /* [한국어] 체크포인트 저장 모드: 0=체크포인트 없음, 1=저장 활성화.
                          * 체크포인트 저장 시 특정 커널·CTA 이후의 메모리/레지스터 상태를 파일로 덤프 */
                         " checkpointing flag (0 = no checkpoint)", "0");
  option_parser_register(
      opp, "-checkpoint_kernel", OPT_INT32, &checkpoint_kernel,
      /* [한국어] 몇 번째 커널 실행 중 체크포인트를 저장할지 지정 (1=첫 번째 커널) */
      " checkpointing during execution of which kernel (1- 1st kernel)", "1");
  option_parser_register(
      opp, "-checkpoint_CTA", OPT_INT32, &checkpoint_CTA,
      /* [한국어] 총 CTA 수 미만의 몇 번째 CTA 완료 후 체크포인트를 저장할지 지정 */
      " checkpointing after # of CTA (< less than total CTA)", "0");
  option_parser_register(opp, "-resume_option", OPT_INT32, &resume_option,
                         /* [한국어] 체크포인트에서 재개(resume) 여부: 0=재개 없음, 1=재개 활성화 */
                         " resume flag (0 = no resume)", "0");
  option_parser_register(opp, "-resume_kernel", OPT_INT32, &resume_kernel,
                         /* [한국어] 재개를 시작할 커널 번호 (1=첫 번째 커널부터 재개) */
                         " Resume from which kernel (1= 1st kernel)", "0");
  option_parser_register(opp, "-resume_CTA", OPT_INT32, &resume_CTA,
                         /* [한국어] 재개를 시작할 CTA 번호 */
                         " resume from which CTA ", "0");
  option_parser_register(opp, "-checkpoint_CTA_t", OPT_INT32, &checkpoint_CTA_t,
                         /* [한국어] 체크포인트 복원 후 완료로 처리할 CTA 수 임계값 */
                         " resume from which CTA ", "0");
  option_parser_register(opp, "-checkpoint_insn_Y", OPT_INT32,
                         &checkpoint_insn_Y,
                         /* [한국어] 체크포인트 복원 시 건너뛸 명령어 수 */
                         " resume from which CTA ", "0");

  option_parser_register(
      opp, "-gpgpu_ptx_convert_to_ptxplus", OPT_BOOL, &m_ptx_convert_to_ptxplus,
      /* [한국어] SASS(네이티브 ISA)를 PTXPlus로 변환하여 실행할지 여부.
       * cuobjdump_to_ptxplus 툴을 통해 SASS → PTXPlus 변환 후 시뮬레이션.
       * PTXPlus는 GPGPU-Sim의 확장 PTX ISA로 SASS에 더 가깝게 모델링됨 */
      "Convert SASS (native ISA) to ptxplus and run ptxplus", "0");
  option_parser_register(opp, "-gpgpu_ptx_force_max_capability", OPT_UINT32,
                         &m_ptx_force_max_capability,
                         /* [한국어] GPU compute capability 강제 지정 (예: 70 = Volta sm_70).
                          * 0이면 실제 기기 capability 사용 */
                         "Force maximum compute capability", "0");
  option_parser_register(
      opp, "-gpgpu_ptx_inst_debug_to_file", OPT_BOOL, &g_ptx_inst_debug_to_file,
      /* [한국어] 실행된 PTX 명령어의 디버그 정보를 파일로 출력할지 여부 */
      "Dump executed instructions' debug information to file", "0");
  option_parser_register(
      opp, "-gpgpu_ptx_inst_debug_file", OPT_CSTR, &g_ptx_inst_debug_file,
      /* [한국어] PTX 명령어 디버그 출력 파일명 (기본: inst_debug.txt) */
      "Executed instructions' debug output file", "inst_debug.txt");
  option_parser_register(opp, "-gpgpu_ptx_inst_debug_thread_uid", OPT_INT32,
                         &g_ptx_inst_debug_thread_uid,
                         /* [한국어] 디버그 출력을 수행할 특정 스레드 UID (기본: 1번 스레드) */
                         "Thread UID for executed instructions' debug output",
                         "1");
}

/*
 * [한국어]
 * gpgpu_functional_sim_config::ptx_set_tex_cache_linesize - 텍스처 캐시 라인 크기 설정
 *
 * @linesize: 텍스처 캐시의 캐시라인 크기 (바이트 단위). 타이밍 모델에서 결정된
 *            텍스처 캐시 설정을 기능 시뮬레이션 설정에 반영할 때 사용.
 * @return: 없음.
 *
 * 타이밍 모델(gpgpu-sim)의 텍스처 캐시 설정(gpgpu_cache_texl1_linesize)을
 * 기능 시뮬레이션 설정(m_texcache_linesize)에 동기화하기 위해 호출된다.
 * generate_mem_accesses()에서 tex_space 접근 시 cache_block_size를 결정할 때 사용.
 * 실행 컨텍스트: 시뮬레이터 초기화 단계.
 *
 * 호출 체인:
 *   gpgpu_sim 초기화 → [이 함수]
 */
void gpgpu_functional_sim_config::ptx_set_tex_cache_linesize(
    unsigned linesize) {
  m_texcache_linesize = linesize; /* [한국어] 텍스처 캐시 라인 크기를 기능 시뮬 설정에 반영 */
}

/*
 * [한국어]
 * gpgpu_t::gpgpu_t - GPU 최상위 시뮬레이터 객체 초기화
 *
 * @config: 기능 시뮬레이션 설정 객체 (ptx 추출 방법, 체크포인트 옵션 등).
 *          멤버 초기화 리스트에서 m_function_model_config에 복사됨.
 * @ctx: 전역 시뮬레이터 컨텍스트 포인터. uid 발급, 통계, device_runtime 접근에 사용.
 * @return: 없음. (생성자)
 *
 * GPU 시뮬레이터의 최상위 객체인 gpgpu_t(또는 그 서브클래스 gpgpu_sim)를 초기화.
 * 주요 초기화 내용:
 *   1) 전역/텍스처/서피스 메모리 공간 생성 (각 memory_space_impl<8192>, 64KB 페이지)
 *   2) 디바이스 힙 포인터(m_dev_malloc)를 GLOBAL_HEAP_START로 초기화
 *   3) config에서 체크포인트/resume 관련 설정값 복사
 *   4) 텍스처 매핑 테이블 초기화 (이름→텍스처 정보, 이름→CudaArray 등)
 *   5) PTX 명령어 디버그 파일 오픈 (설정된 경우)
 *   6) 시뮬레이션 사이클 카운터 초기화
 * gpgpu_sim(타이밍 시뮬)이 이 클래스를 상속하므로, 기능 시뮬에 필요한 공통 상태를
 * 이 생성자에서 설정한다.
 * 실행 컨텍스트: 시뮬레이터 시작 시 1회 호출.
 *
 * 호출 체인:
 *   gpgpu_sim::gpgpu_sim() (파생 클래스 생성자) → [이 함수 (기반 클래스 생성자)]
 */
gpgpu_t::gpgpu_t(const gpgpu_functional_sim_config &config, gpgpu_context *ctx)
    : m_function_model_config(config) { /* [한국어] 기능 시뮬 설정을 멤버에 복사 초기화 */
  gpgpu_ctx = ctx; /* [한국어] 전역 컨텍스트 포인터 저장 — uid 발급 및 device_runtime 접근 */
  m_global_mem = new memory_space_impl<8192>("global", 64 * 1024);
  /* [한국어] 전역 메모리 공간 생성 — 8192바이트 페이지 크기, 64KB 초기 용량.
   * CUDA 커널의 global 변수와 cudaMalloc 할당 공간이 여기에 매핑된다 */

  m_tex_mem = new memory_space_impl<8192>("tex", 64 * 1024);
  /* [한국어] 텍스처 메모리 공간 생성 — 텍스처 fetch 명령어가 접근하는 읽기 전용 메모리 */
  m_surf_mem = new memory_space_impl<8192>("surf", 64 * 1024);
  /* [한국어] 서피스(surface) 메모리 공간 생성 — CUDA surface 객체를 통한 읽기/쓰기 메모리 */

  m_dev_malloc = GLOBAL_HEAP_START;
  /* [한국어] 디바이스 힙 할당 포인터를 전역 힙 시작 주소로 초기화.
   * cudaMalloc() 호출 시 이 포인터에서 순차적으로 메모리를 할당 */
  checkpoint_option = m_function_model_config.get_checkpoint_option(); /* [한국어] 체크포인트 저장 모드 (0=없음, 1=활성화) */
  checkpoint_kernel = m_function_model_config.get_checkpoint_kernel(); /* [한국어] 체크포인트 저장 대상 커널 번호 */
  checkpoint_CTA = m_function_model_config.get_checkpoint_CTA();       /* [한국어] 체크포인트 저장 기준 CTA 수 */
  resume_option = m_function_model_config.get_resume_option();         /* [한국어] 체크포인트 재개 여부 */
  resume_kernel = m_function_model_config.get_resume_kernel();         /* [한국어] 재개 시작 커널 번호 */
  resume_CTA = m_function_model_config.get_resume_CTA();               /* [한국어] 재개 시작 CTA 번호 */
  checkpoint_CTA_t = m_function_model_config.get_checkpoint_CTA_t();   /* [한국어] 복원 후 완료 처리할 CTA 임계값 */
  checkpoint_insn_Y = m_function_model_config.get_checkpoint_insn_Y(); /* [한국어] 복원 후 건너뛸 명령어 수 */

  // initialize texture mappings to empty
  m_NameToTextureInfo.clear(); /* [한국어] 텍스처 이름 → textureInfo 매핑 테이블 초기화 */
  m_NameToCudaArray.clear();   /* [한국어] 텍스처 이름 → cudaArray 포인터 매핑 테이블 초기화 */
  m_TextureRefToName.clear();  /* [한국어] 텍스처 참조 변수 → 이름 역매핑 테이블 초기화 */
  m_NameToAttribute.clear();   /* [한국어] 텍스처 이름 → 속성(채널 수, 바이트 크기 등) 매핑 초기화 */

  if (m_function_model_config.get_ptx_inst_debug_to_file() != 0)
    /* [한국어] PTX 명령어 디버그 출력이 활성화된 경우 */
    ptx_inst_debug_file =
        fopen(m_function_model_config.get_ptx_inst_debug_file(), "w");
    /* [한국어] 설정된 파일명으로 디버그 출력 파일 오픈 (쓰기 모드) */

  gpu_sim_cycle = 0;     /* [한국어] 현재 커널 실행 사이클 카운터 초기화 */
  gpu_tot_sim_cycle = 0; /* [한국어] 누적 전체 시뮬레이션 사이클 카운터 초기화 */
}

/*
 * [한국어]
 * line_size_based_tag_func - 주소에서 캐시라인/세그먼트 정렬 기준 태그 주소 계산
 *
 * @address: 원본 메모리 주소 (스레드가 접근하는 실제 바이트 주소).
 * @line_size: 캐시라인 또는 세그먼트 크기 (바이트 단위, 반드시 2의 거듭제곱).
 * @return: address를 line_size 경계로 내림 정렬한 블록 기준 주소.
 *
 * 캐시 태그 함수로, 특정 주소가 속하는 캐시라인(또는 메모리 세그먼트)의
 * 시작 주소를 계산한다. 비트 AND 마스킹을 사용:
 *   결과 = address & ~(line_size - 1)
 *   예) address=0x105, line_size=128(0x80) → 0x105 & ~0x7F = 0x100
 * generate_mem_accesses()와 memory_coalescing_arch()에서 여러 스레드의 주소를
 * 동일 캐시라인으로 병합(coalesce)하기 위해 광범위하게 사용된다.
 * 실행 컨텍스트: 단일 시뮬레이션 스레드에서 인라인 수준으로 빈번히 호출.
 *
 * 호출 체인:
 *   warp_inst_t::generate_mem_accesses() → [이 함수]
 *   warp_inst_t::memory_coalescing_arch() → [이 함수]
 */
new_addr_type line_size_based_tag_func(new_addr_type address,
                                       new_addr_type line_size) {
  // gives the tag for an address based on a given line size
  return address & ~(line_size - 1); /* [한국어] line_size-1의 보수 마스크로 AND — 하위 비트를 0으로 클리어하여 블록 시작 주소 획득 */
}

/*
 * [한국어]
 * mem_access_type_str - 메모리 접근 타입 enum 값을 문자열로 변환
 *
 * @access_type: mem_access_type enum 값 (예: GLOBAL_ACC_R, LOCAL_ACC_W 등).
 *               MEM_ACCESS_TYPE_TUP_DEF 매크로로 정의된 전체 열거형 중 하나.
 * @return: 해당 enum 값의 문자열 이름 포인터 (정적 배열, 해제 불필요).
 *
 * X-macro 패턴을 사용하여 enum 값과 동일한 이름의 문자열 배열을 자동 생성한다.
 * MEM_ACCESS_TYPE_TUP_DEF 매크로가 MA_TUP_BEGIN/MA_TUP/MA_TUP_END를 순서대로
 * 호출하여 static 배열 access_type_str[]을 초기화한다.
 * 통계 출력 및 디버그 로그에서 접근 타입을 사람이 읽을 수 있는 형태로 표시할 때 사용.
 * 실행 컨텍스트: 시뮬레이션 통계 출력 시 호출.
 *
 * 호출 체인:
 *   gpu-sim.cc::print_stats() 또는 디버그 출력 → [이 함수]
 */
const char *mem_access_type_str(enum mem_access_type access_type) {
#define MA_TUP_BEGIN(X) static const char *access_type_str[] = { /* [한국어] X-macro: 정적 문자열 배열 선언 시작 */
#define MA_TUP(X) #X  /* [한국어] X-macro: enum 이름을 문자열 리터럴로 변환 (#X = stringification) */
#define MA_TUP_END(X) \
  }                   \
  ;  /* [한국어] X-macro: 배열 선언 닫기 및 세미콜론 */
  MEM_ACCESS_TYPE_TUP_DEF /* [한국어] 위에서 정의한 매크로로 access_type_str[] 배열 자동 생성 */
#undef MA_TUP_BEGIN /* [한국어] 매크로 재정의 오염 방지를 위해 사용 후 해제 */
#undef MA_TUP
#undef MA_TUP_END

  assert(access_type < NUM_MEM_ACCESS_TYPE); /* [한국어] 배열 범위 초과 접근 방지 */

  return access_type_str[access_type]; /* [한국어] enum 인덱스로 해당 문자열 반환 */
}

/*
 * [한국어]
 * warp_inst_t::clear_active - 특정 스레드들을 active_mask에서 비활성화
 *
 * @inactive: 비활성화할 스레드들의 마스크. 이 비트들이 현재 active_mask에
 *            모두 설정되어 있어야 한다 (assert로 검증).
 * @return: 없음.
 *
 * 메모리 접근이 완료되었거나 조기 종료된 스레드들을 active_mask에서 제거한다.
 * 비활성화 전에 해당 스레드들이 실제로 활성 상태였는지 assert로 검증:
 *   test = current_mask & inactive → test가 inactive와 같아야 유효
 * 이후 비트 NOT + AND로 해당 비트들을 클리어한다.
 * 실행 컨텍스트: 캐시 miss/hit 처리, 프리디케이션 적용 시 호출.
 *
 * 호출 체인:
 *   shader_core_ctx::memory_cycle() → [이 함수]
 */
void warp_inst_t::clear_active(const active_mask_t &inactive) {
  active_mask_t test = m_warp_active_mask; /* [한국어] 현재 active_mask를 임시 복사 — 검증용 */
  test &= inactive; /* [한국어] 비활성화할 스레드와 현재 활성 스레드의 교집합 계산 */
  assert(test == inactive);  // verify threads being disabled were active
  /* [한국어] 교집합이 inactive와 같아야 함 — inactive의 모든 스레드가 실제 활성 상태임을 보장 */
  m_warp_active_mask &= ~inactive; /* [한국어] inactive 마스크의 비트들을 active_mask에서 클리어 */
}

/*
 * [한국어]
 * warp_inst_t::set_not_active - 특정 레인 하나를 비활성화
 *
 * @lane_id: 비활성화할 스레드 레인 번호 (0~31).
 * @return: 없음.
 *
 * 단일 스레드 레인을 active_mask에서 제거하는 경량 버전.
 * 원자 연산 콜백 취소나 프리디케이션에서 개별 스레드를 끌 때 사용.
 * 실행 컨텍스트: 타이밍 시뮬 execute 단계.
 *
 * 호출 체인:
 *   warp_inst_t::set_active() / shader_core_ctx::execute() → [이 함수]
 */
void warp_inst_t::set_not_active(unsigned lane_id) {
  m_warp_active_mask.reset(lane_id); /* [한국어] 지정 레인 비트를 0으로 클리어 — 이 스레드는 이후 실행에서 제외 */
}

/*
 * [한국어]
 * warp_inst_t::set_active - active_mask를 새 마스크로 교체하고 원자 콜백 초기화
 *
 * @active: 새로 설정할 active_mask.
 *          SIMT 스택 top에서 가져온 active_mask를 적용할 때 사용.
 * @return: 없음.
 *
 * active_mask를 통째로 교체한다. 원자 명령어(m_isatomic=true)인 경우,
 * 비활성화된 레인의 dram_callback을 NULL로 초기화하여 do_atomic()이
 * 해당 레인의 콜백을 실행하지 않도록 한다. 원자 연산 콜백이 남아있으면
 * 비활성 스레드가 잘못된 메모리를 수정할 수 있으므로 반드시 클리어해야 함.
 * 실행 컨텍스트: core_t::getExecuteWarp()에서 SIMT 스택으로부터 마스크 적용 시.
 *
 * 호출 체인:
 *   core_t::getExecuteWarp() → [이 함수]
 */
void warp_inst_t::set_active(const active_mask_t &active) {
  m_warp_active_mask = active; /* [한국어] active_mask를 새 마스크로 완전 교체 */
  if (m_isatomic) { /* [한국어] 원자 명령어인 경우 비활성 레인의 콜백을 NULL로 초기화 */
    for (unsigned i = 0; i < m_config->warp_size; i++) { /* [한국어] 전체 32 레인 순회 */
      if (!m_warp_active_mask.test(i)) { /* [한국어] 비활성 레인인 경우 */
        m_per_scalar_thread[i].callback.function = NULL;
        /* [한국어] 원자 연산 콜백 함수 포인터 클리어 — 비활성 스레드의 원자 연산 방지 */
        m_per_scalar_thread[i].callback.instruction = NULL;
        /* [한국어] 관련 PTX 명령어 포인터 클리어 */
        m_per_scalar_thread[i].callback.thread = NULL;
        /* [한국어] 스레드 컨텍스트 포인터 클리어 */
      }
    }
  }
}

/*
 * [한국어]
 * warp_inst_t::do_atomic (오버로드 1) - 현재 active_mask로 원자 연산 실행
 *
 * @forceDo: true면 명령어가 empty 상태여도 원자 연산 강제 실행.
 *            false(기본)면 명령어가 유효할 때만 실행.
 * @return: 없음.
 *
 * 두 번째 오버로드(access_mask 버전)에 현재 active_mask를 전달하는 래퍼.
 * 실행 컨텍스트: 타이밍 시뮬의 메모리 완료 콜백 단계.
 *
 * 호출 체인:
 *   shader_core_ctx::memory_cycle() → [이 함수] → do_atomic(mask, forceDo)
 */
void warp_inst_t::do_atomic(bool forceDo) {
  do_atomic(m_warp_active_mask, forceDo); /* [한국어] 현재 active_mask를 인자로 전달하여 실제 원자 연산 실행 */
}

/*
 * [한국어]
 * warp_inst_t::do_atomic (오버로드 2) - 지정 마스크 스레드들의 원자 연산 콜백 실행
 *
 * @access_mask: 원자 연산을 실행할 스레드 레인의 마스크.
 *               L2 캐시에서 miss 처리 완료 후 DRAM 응답 시점에 전달.
 * @forceDo: true면 명령어가 empty 상태여도 강제 실행 (체크포인트 resume 용).
 * @return: 없음. should_do_atomic이 false면 즉시 반환.
 *
 * 원자 명령어(atomicAdd 등)의 read-modify-write 중 modify 단계를 실행한다.
 * 각 활성 스레드의 dram_callback(function, instruction, thread)을 호출하여
 * PTX 기능 시뮬레이션의 원자 연산을 완료시킨다.
 * should_do_atomic 플래그가 false면 (이미 실행했거나 조건 불충족 시) 건너뜀.
 * 실행 컨텍스트: DRAM 응답 처리 단계 (single sim thread).
 *
 * 호출 체인:
 *   mem_fetch 완료 콜백 → shader_core_ctx::writeback() → [이 함수]
 */
void warp_inst_t::do_atomic(const active_mask_t &access_mask, bool forceDo) {
  assert(m_isatomic && (!m_empty || forceDo)); /* [한국어] 이 함수는 원자 명령어에만 적용 — 빈 슬롯이면 forceDo가 필요 */
  if (!should_do_atomic) return; /* [한국어] 이미 원자 연산이 완료되었거나 조건 불충족이면 스킵 */
  for (unsigned i = 0; i < m_config->warp_size; i++) { /* [한국어] 전체 warp_size(32) 레인 순회 */
    if (access_mask.test(i)) { /* [한국어] 이 레인이 원자 연산 대상인지 확인 */
      dram_callback_t &cb = m_per_scalar_thread[i].callback; /* [한국어] 이 스레드의 원자 연산 콜백 구조체 참조 */
      if (cb.thread) cb.function(cb.instruction, cb.thread);
      /* [한국어] 콜백 스레드가 유효하면 원자 연산 함수 실행
       * cb.function: PTX 원자 명령어 실행 함수 포인터
       * cb.instruction: 해당 PTX 명령어 포인터
       * cb.thread: 이 스레드의 기능 시뮬 상태 (레지스터, 메모리 등) */
    }
  }
}

/*
 * [한국어]
 * warp_inst_t::broadcast_barrier_reduction - 배리어 리덕션 콜백 브로드캐스트 실행
 *
 * @access_mask: 배리어 리덕션을 수행할 스레드 레인 마스크.
 * @return: 없음.
 *
 * __syncthreads() 계열의 배리어 명령어에서 리덕션 연산(예: ballot, vote)을
 * 수행할 때 활성 스레드들의 콜백을 실행한다. do_atomic과 달리 should_do_atomic
 * 검사 없이 무조건 실행된다. 배리어 완료 후 결과를 브로드캐스트하는 맥락에서 사용.
 * 실행 컨텍스트: 타이밍 시뮬의 배리어 완료 처리 단계.
 *
 * 호출 체인:
 *   shader_core_ctx::barrier_reduction() → [이 함수]
 */
void warp_inst_t::broadcast_barrier_reduction(
    const active_mask_t &access_mask) {
  for (unsigned i = 0; i < m_config->warp_size; i++) { /* [한국어] 전체 warp_size 레인 순회 */
    if (access_mask.test(i)) { /* [한국어] 이 레인이 리덕션 대상인지 확인 */
      dram_callback_t &cb = m_per_scalar_thread[i].callback; /* [한국어] 이 스레드의 콜백 참조 */
      if (cb.thread) { /* [한국어] 유효한 스레드 컨텍스트가 있는 경우에만 실행 */
        cb.function(cb.instruction, cb.thread); /* [한국어] 배리어 리덕션 연산 콜백 실행 */
      }
    }
  }
}

/*
 * [한국어]
 * warp_inst_t::generate_mem_accesses - warp의 메모리 명령어를 실제 접근 요청으로 변환
 *
 * @return: 없음. 결과는 m_accessq에 mem_access_t 객체로 push_back됨.
 *
 * warp(32스레드)가 메모리 명령어를 실행할 때 각 스레드의 개별 주소를 분석하여
 * 캐시/DRAM으로 전달할 실제 메모리 요청 리스트를 생성한다. 이 과정이 coalescing임.
 * 주요 처리 단계:
 *   1) 조기 반환 조건 검사: empty/barrier/이미생성/비메모리 명령어/마스크0
 *   2) 메모리 공간 타입에 따라 access_type 결정 (CONST/TEX/GLOBAL/LOCAL/SHARED)
 *   3) 공간 타입별 요청 생성 전략:
 *      - shared/sstarr: bank conflict 계산 → cycles 필드에 conflict 수 반영
 *      - tex/const: 단순 cache_block_size 기반 병합
 *      - global/local: memory_coalescing_arch() 또는 memory_coalescing_arch_atomic()
 *   4) cache_block_size가 설정된 경우 블록 단위로 m_accessq에 push_back
 *   5) global space면 uncoalesced 접근 수 통계 수집
 * 실행 컨텍스트: 타이밍 시뮬의 LDST unit에서 명령어가 execute 단계에 진입 시.
 *   단일 시뮬 스레드에서 실행되므로 별도 동기화 불필요.
 *
 * 호출 체인:
 *   shader_core_ctx::ldst_unit::cycle() → [이 함수]
 *   → memory_coalescing_arch() / memory_coalescing_arch_atomic()
 *   → memory_coalescing_arch_reduce_and_send() → m_accessq.push_back()
 */
void warp_inst_t::generate_mem_accesses() {
  if (empty() || op == MEMORY_BARRIER_OP || m_mem_accesses_created) return;
  /* [한국어] 조기 반환: 빈 슬롯이거나, 메모리 배리어 명령어이거나, 이미 접근 요청이 생성된 경우 */
  if (!((op == LOAD_OP) || (op == TENSOR_CORE_LOAD_OP) || (op == STORE_OP) ||
        (op == TENSOR_CORE_STORE_OP)))
    return;
  /* [한국어] 메모리 load/store(Tensor Core 포함)가 아닌 명령어이면 조기 반환 */
  if (m_warp_active_mask.count() == 0) return;  // predicated off
  /* [한국어] 모든 스레드가 비활성(predicate off)이면 메모리 요청 불필요 */

  const size_t starting_queue_size = m_accessq.size();
  /* [한국어] uncoalesced 접근 통계 계산을 위해 이 함수 진입 전 큐 크기 기록 */

  assert(is_load() || is_store()); /* [한국어] 이 시점에서 반드시 load 또는 store여야 함 */

  // if((space.get_type() != tex_space) && (space.get_type() != const_space))
  assert(m_per_scalar_thread_valid);  // need address information per thread
  /* [한국어] 스레드별 메모리 주소(m_per_scalar_thread[i].memreqaddr)가
   * 기능 시뮬(ptx_exec_inst)에서 이미 채워져 있어야 함 */

  bool is_write = is_store(); /* [한국어] 쓰기 여부 결정 — access_type 선택 및 mem_access_t 생성에 사용 */

  mem_access_type access_type; /* [한국어] 메모리 접근 분류 (캐시/DRAM 경로 결정에 사용) */
  switch (space.get_type()) {
    case const_space:
    case param_space_kernel:
      access_type = CONST_ACC_R; /* [한국어] 상수/커널 파라미터 공간 — 읽기 전용, 상수 캐시 경로 */
      break;
    case tex_space:
      access_type = TEXTURE_ACC_R; /* [한국어] 텍스처 공간 — 읽기 전용, 텍스처 캐시 경로 */
      break;
    case global_space:
      access_type = is_write ? GLOBAL_ACC_W : GLOBAL_ACC_R; /* [한국어] 전역 메모리 — L1/L2/DRAM 경로, 읽기/쓰기 구분 */
      break;
    case local_space:
    case param_space_local:
      access_type = is_write ? LOCAL_ACC_W : LOCAL_ACC_R; /* [한국어] 로컬 메모리 — 스레드 전용 스택 공간, 전역 메모리에 매핑 */
      break;
    case shared_space:
      break; /* [한국어] 공유 메모리 — access_type 불필요, bank conflict만 계산 */
    case sstarr_space:
      break; /* [한국어] sstarr(shared static array) 공간 — shared와 동일 처리 */
    default:
      assert(0); /* [한국어] 알 수 없는 메모리 공간 타입 — 프로그래밍 오류 */
      break;
  }

  // Calculate memory accesses generated by this warp
  new_addr_type cache_block_size = 0;  // in bytes
  /* [한국어] tex/const/simple 공간에서 사용할 캐시라인 크기 (0이면 coalescing 함수가 직접 처리) */

  switch (space.get_type()) {
    case shared_space:
    case sstarr_space: {
      unsigned subwarp_size = m_config->warp_size / m_config->mem_warp_parts;
      /* [한국어] subwarp 크기 = 전체 warp_size / mem_warp_parts (예: 32/2=16).
       * GPGPU-Sim은 warp를 여러 파트로 나눠 shared memory 접근을 처리한다 */
      unsigned total_accesses = 0; /* [한국어] 전체 subwarp에 걸친 최대 bank 접근 수 누적 */
      for (unsigned subwarp = 0; subwarp < m_config->mem_warp_parts;
           subwarp++) { /* [한국어] 각 subwarp 파트별로 bank conflict 계산 */
        // data structures used per part warp
        std::map<unsigned, std::map<new_addr_type, unsigned> >
            bank_accs;  // bank -> word address -> access count
        /* [한국어] bank_accs: 뱅크 번호 → (워드 주소 → 접근 횟수) 중첩 맵
         * 각 뱅크에 어떤 워드에 몇 번 접근했는지 추적 */

        // step 1: compute accesses to words in banks
        for (unsigned thread = subwarp * subwarp_size;
             thread < (subwarp + 1) * subwarp_size; thread++) {
          /* [한국어] 이 subwarp에 속하는 스레드들을 순회 */
          if (!active(thread)) continue; /* [한국어] 비활성 스레드는 건너뜀 */
          new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
          /* [한국어] 이 스레드가 접근할 shared memory 주소 */
          // FIXME: deferred allocation of shared memory should not accumulate
          // across kernel launches assert( addr < m_config->gpgpu_shmem_size );
          unsigned bank = m_config->shmem_bank_func(addr);
          /* [한국어] 주소로부터 shared memory 뱅크 번호 계산 (일반적으로 addr/4 % num_banks) */
          new_addr_type word =
              line_size_based_tag_func(addr, m_config->WORD_SIZE);
          /* [한국어] 주소를 WORD_SIZE(4B) 단위로 정렬 — 같은 워드에 대한 접근을 묶기 위해 */
          bank_accs[bank][word]++; /* [한국어] 해당 뱅크의 해당 워드에 대한 접근 횟수 증가 */
        }

        if (m_config->shmem_limited_broadcast) {
          /* [한국어] 제한적 브로드캐스트 모드: 같은 워드에 여러 스레드가 접근하면 1번만 카운트 */
          // step 2: look for and select a broadcast bank/word if one occurs
          bool broadcast_detected = false;       /* [한국어] 브로드캐스트(같은 워드 다중 접근) 감지 여부 */
          new_addr_type broadcast_word = (new_addr_type)-1; /* [한국어] 브로드캐스트 워드 주소 (초기값: 무효) */
          unsigned broadcast_bank = (unsigned)-1;           /* [한국어] 브로드캐스트 뱅크 번호 (초기값: 무효) */
          std::map<unsigned, std::map<new_addr_type, unsigned> >::iterator b;
          for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
            /* [한국어] 모든 뱅크를 순회하며 브로드캐스트 발생 여부 탐색 */
            unsigned bank = b->first; /* [한국어] 현재 뱅크 번호 */
            std::map<new_addr_type, unsigned> &access_set = b->second; /* [한국어] 이 뱅크의 워드→접근수 맵 */
            std::map<new_addr_type, unsigned>::iterator w;
            for (w = access_set.begin(); w != access_set.end(); ++w) {
              if (w->second > 1) {
                /* [한국어] 같은 워드에 2회 이상 접근 = 브로드캐스트 발생 */
                // found a broadcast
                broadcast_detected = true;  /* [한국어] 브로드캐스트 감지 플래그 설정 */
                broadcast_bank = bank;       /* [한국어] 브로드캐스트가 발생한 뱅크 기록 */
                broadcast_word = w->first;   /* [한국어] 브로드캐스트 대상 워드 주소 기록 */
                break;
              }
            }
            if (broadcast_detected) break; /* [한국어] 첫 번째 브로드캐스트만 처리 후 외부 루프 탈출 */
          }

          // step 3: figure out max bank accesses performed, taking account of
          // broadcast case
          unsigned max_bank_accesses = 0; /* [한국어] 모든 뱅크 중 최대 실효 접근 수 */
          for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
            unsigned bank_accesses = 0; /* [한국어] 이 뱅크의 실효 접근 수 */
            std::map<new_addr_type, unsigned> &access_set = b->second;
            std::map<new_addr_type, unsigned>::iterator w;
            for (w = access_set.begin(); w != access_set.end(); ++w)
              bank_accesses += w->second; /* [한국어] 이 뱅크의 모든 워드 접근 수 합산 */
            if (broadcast_detected && broadcast_bank == b->first) {
              /* [한국어] 브로드캐스트 뱅크인 경우 — 브로드캐스트 접근은 1번만 카운트하므로 초과분 제거 */
              for (w = access_set.begin(); w != access_set.end(); ++w) {
                if (w->first == broadcast_word) {
                  unsigned n = w->second; /* [한국어] 브로드캐스트 워드의 실제 접근 횟수 */
                  assert(n > 1);  // or this wasn't a broadcast
                  assert(bank_accesses >= (n - 1));
                  bank_accesses -= (n - 1); /* [한국어] n개 접근 중 (n-1)개를 브로드캐스트로 처리 → 실효 접근 1로 감소 */
                  break;
                }
              }
            }
            if (bank_accesses > max_bank_accesses)
              max_bank_accesses = bank_accesses; /* [한국어] 이 subwarp에서 가장 많이 충돌한 뱅크의 접근 수 갱신 */
          }

          // step 4: accumulate
          total_accesses += max_bank_accesses; /* [한국어] 이 subwarp의 최대 뱅크 접근 수를 전체 누적에 더함 */
        } else {
          // step 2: look for the bank with the maximum number of access to
          // different words
          /* [한국어] 브로드캐스트 없는 모드: 가장 많은 고유 워드에 접근하는 뱅크의 접근 수가 사이클 결정 */
          unsigned max_bank_accesses = 0; /* [한국어] 최대 뱅크 접근 수 (고유 워드 기준) */
          std::map<unsigned, std::map<new_addr_type, unsigned> >::iterator b;
          for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
            max_bank_accesses =
                std::max(max_bank_accesses, (unsigned)b->second.size());
            /* [한국어] b->second.size() = 이 뱅크에 접근하는 고유 워드 수 = 뱅크 충돌 횟수 */
          }

          // step 3: accumulate
          total_accesses += max_bank_accesses; /* [한국어] 고유 워드 최대 뱅크 접근 수를 누적 */
        }
      }
      assert(total_accesses > 0 && total_accesses <= m_config->warp_size);
      /* [한국어] 유효 범위 검증: 최소 1, 최대 warp_size (32) 접근 */
      cycles = total_accesses;  // shared memory conflicts modeled as larger
                                // initiation interval
      /* [한국어] bank conflict 수를 cycles에 반영 — 충돌이 많을수록 이 명령어의 발행 간격이 길어짐 */
      m_config->gpgpu_ctx->stats->ptx_file_line_stats_add_smem_bank_conflict(
          pc, total_accesses);
      /* [한국어] PTX 파일·라인별 shared memory bank conflict 통계 기록 */
      break;
    }

    case tex_space:
      cache_block_size = m_config->gpgpu_cache_texl1_linesize;
      /* [한국어] 텍스처 공간: 텍스처 L1 캐시 라인 크기를 block 단위로 사용 */
      break;
    case const_space:
    case param_space_kernel:
      cache_block_size = m_config->gpgpu_cache_constl1_linesize;
      /* [한국어] 상수/커널 파라미터 공간: 상수 캐시 L1 라인 크기를 block 단위로 사용 */
      break;

    case global_space:
    case local_space:
    case param_space_local:
      if (m_config->gpgpu_coalesce_arch >= 13) {
        /* [한국어] compute capability 1.3 이상(Fermi, Kepler, Maxwell, Pascal, Volta)의 coalescing 처리 */
        if (isatomic())
          memory_coalescing_arch_atomic(is_write, access_type);
          /* [한국어] 원자 명령어: 각 스레드가 독립적인 트랜잭션이 필요 (byte 충돌 없어야 함) */
        else
          memory_coalescing_arch(is_write, access_type);
          /* [한국어] 일반 load/store: 동일 세그먼트 접근을 병합하여 최소 트랜잭션 생성 */
      } else
        abort(); /* [한국어] cc 1.3 미만 아키텍처는 미지원 */

      break;

    default:
      abort(); /* [한국어] 처리하지 않는 메모리 공간 타입 — 도달하면 프로그래밍 오류 */
  }

  if (cache_block_size) {
    /* [한국어] tex/const 공간처럼 단순 캐시 블록 단위 병합이 필요한 경우 */
    assert(m_accessq.empty()); /* [한국어] coalescing 함수가 이미 m_accessq를 채우지 않았음을 확인 */
    mem_access_byte_mask_t byte_mask; /* [한국어] 이 캐시 블록 내에서 접근되는 바이트의 비트 마스크 */
    std::map<new_addr_type, active_mask_t>
        accesses;  // block address -> set of thread offsets in warp
    /* [한국어] accesses: 캐시 블록 기준 주소 → 이 블록에 접근하는 스레드 집합 */
    std::map<new_addr_type, active_mask_t>::iterator a;
    for (unsigned thread = 0; thread < m_config->warp_size; thread++) {
      /* [한국어] 전체 warp의 스레드를 순회하며 각 스레드의 접근을 캐시 블록별로 분류 */
      if (!active(thread)) continue; /* [한국어] 비활성 스레드 건너뜀 */
      new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
      /* [한국어] 이 스레드가 접근할 메모리 주소 */
      new_addr_type block_address =
          line_size_based_tag_func(addr, cache_block_size);
      /* [한국어] 주소를 cache_block_size 단위로 내림 정렬 → 속하는 캐시 블록의 시작 주소 */
      accesses[block_address].set(thread); /* [한국어] 이 블록에 접근하는 스레드 집합에 추가 */
      unsigned idx = addr - block_address; /* [한국어] 블록 내 바이트 오프셋 */
      for (unsigned i = 0; i < data_size; i++) byte_mask.set(idx + i);
      /* [한국어] 이 스레드가 접근하는 바이트들을 byte_mask에 표시 (data_size 바이트) */
    }
    for (a = accesses.begin(); a != accesses.end(); ++a)
      m_accessq.push_back(mem_access_t(
          access_type, a->first, cache_block_size, is_write, a->second,
          byte_mask, mem_access_sector_mask_t(), m_config->gpgpu_ctx));
    /* [한국어] 각 캐시 블록별로 mem_access_t 객체를 생성하여 접근 큐에 추가
     * 인자: 접근타입, 블록주소, 블록크기, 쓰기여부, 활성스레드마스크, 바이트마스크, 섹터마스크, ctx */
  }

  if (space.get_type() == global_space) {
    m_config->gpgpu_ctx->stats->ptx_file_line_stats_add_uncoalesced_gmem(
        pc, m_accessq.size() - starting_queue_size);
    /* [한국어] global 공간 접근의 uncoalesced 요청 수 통계 기록
     * (이 함수에서 추가된 접근 수 = m_accessq.size() - starting_queue_size)
     * 이상적으로 완전히 coalesce되면 1개, 최악이면 warp_size(32)개의 요청이 생성됨 */
  }
  m_mem_accesses_created = true; /* [한국어] 접근 요청 생성 완료 표시 — 중복 생성 방지 */
}

/*
 * [한국어]
 * warp_inst_t::memory_coalescing_arch - 아키텍처별 메모리 coalescing 처리 (일반 load/store)
 *
 * @is_write: true면 store 명령어, false면 load 명령어.
 * @access_type: 메모리 접근 분류 (GLOBAL_ACC_R/W, LOCAL_ACC_R/W 등).
 * @return: 없음. 결과는 m_accessq에 mem_access_t로 추가됨.
 *
 * CUDA 아키텍처(Fermi/Kepler/Maxwell/Pascal/Volta)에 따른 global/local 메모리
 * coalescing 규칙을 구현한다. 32개 스레드의 서로 다른 주소들을 분석하여
 * 최소한의 메모리 트랜잭션으로 병합(coalesce)한다.
 *
 * 아키텍처별 세그먼트 크기 결정:
 *   - Fermi/Kepler (arch 20~38): L1은 normal(128B), L2는 sector(32B).
 *     gmem_skip_L1D=true 또는 CACHE_GLOBAL 명령이면 sector 모드
 *   - Maxwell/Pascal/Volta (arch 40+): L1/L2 모두 sector (32B)
 * data_size별 segment_size:
 *   - 1B: 32B 세그먼트
 *   - 2B: sector=32B, normal=64B
 *   - 4/8/16B: sector=32B, normal=128B
 *
 * 처리 단계:
 *   1) 각 subwarp(warp의 절반 또는 전체)의 스레드별 접근 주소를 같은 세그먼트로 묶음
 *   2) 로컬 메모리 4B 이상 접근은 4B 청크로 분할
 *   3) 세그먼트 경계를 걸치는 접근은 두 세그먼트에 각각 등록
 *   4) memory_coalescing_arch_reduce_and_send()로 세그먼트 크기 최적화 후 m_accessq에 추가
 *
 * 실행 컨텍스트: generate_mem_accesses()에서 호출 (단일 시뮬 스레드).
 *
 * 호출 체인:
 *   warp_inst_t::generate_mem_accesses() → [이 함수]
 *   → memory_coalescing_arch_reduce_and_send() → m_accessq.push_back()
 */
void warp_inst_t::memory_coalescing_arch(bool is_write,
                                         mem_access_type access_type) {
  // see the CUDA manual where it discusses coalescing rules before reading this
  unsigned segment_size = 0;        /* [한국어] 아키텍처/데이터 크기에 따른 메모리 세그먼트 크기 (바이트) */
  unsigned warp_parts = m_config->mem_warp_parts; /* [한국어] warp를 몇 파트로 나눌지 (일반적으로 1 또는 2) */
  bool sector_segment_size = false; /* [한국어] 세그먼트가 32B sector 크기인지 여부 (아키텍처에 따라 결정) */

  if (m_config->gpgpu_coalesce_arch >= 20 &&
      m_config->gpgpu_coalesce_arch < 39) {
    /* [한국어] Fermi(sm_20~21), Kepler(sm_30~37): L1은 normal(128B), L2는 sector(32B) */
    // Fermi and Kepler, L1 is normal and L2 is sector
    if (m_config->gmem_skip_L1D || cache_op == CACHE_GLOBAL)
      sector_segment_size = true;   /* [한국어] L1을 건너뛰거나 CACHE_GLOBAL 명령어면 L2 sector(32B) 모드 */
    else
      sector_segment_size = false;  /* [한국어] L1 캐시 경유 시 normal(128B) 세그먼트 모드 */
  } else if (m_config->gpgpu_coalesce_arch >= 40) {
    // Maxwell, Pascal and Volta, L1 and L2 are sectors
    // all requests should be 32 bytes
    sector_segment_size = true; /* [한국어] Maxwell(sm_50~53), Pascal(sm_60~62), Volta(sm_70+): 항상 sector(32B) */
  }

  switch (data_size) {
    case 1:
      segment_size = 32; /* [한국어] 1B 데이터: 최소 세그먼트인 32B 사용 */
      break;
    case 2:
      segment_size = sector_segment_size ? 32 : 64; /* [한국어] 2B: sector면 32B, normal이면 64B */
      break;
    case 4:
    case 8:
    case 16:
      segment_size = sector_segment_size ? 32 : 128; /* [한국어] 4/8/16B: sector면 32B, normal이면 128B */
      break;
  }
  unsigned subwarp_size = m_config->warp_size / warp_parts;
  /* [한국어] subwarp 크기: 전체 warp를 warp_parts로 나눈 크기 (예: 32/1=32 또는 32/2=16) */

  for (unsigned subwarp = 0; subwarp < warp_parts; subwarp++) {
    /* [한국어] 각 subwarp 파트별로 독립적으로 coalescing 처리 */
    std::map<new_addr_type, transaction_info> subwarp_transactions;
    /* [한국어] 이 subwarp의 세그먼트 주소 → 트랜잭션 정보 맵
     * 같은 세그먼트에 접근하는 스레드들이 하나의 transaction_info에 합쳐짐 */

    // step 1: find all transactions generated by this subwarp
    for (unsigned thread = subwarp * subwarp_size;
         thread < subwarp_size * (subwarp + 1); thread++) {
      /* [한국어] 이 subwarp에 속하는 스레드들 순회 */
      if (!active(thread)) continue; /* [한국어] 비활성 스레드 건너뜀 */

      unsigned data_size_coales = data_size; /* [한국어] coalescing에 사용할 실효 데이터 크기 */
      unsigned num_accesses = 1;             /* [한국어] 이 스레드의 접근 횟수 (로컬 메모리 분할 시 증가) */

      if (space.get_type() == local_space ||
          space.get_type() == param_space_local) {
        /* [한국어] 로컬 메모리 접근: 4B 이상이면 4B 청크로 분할 */
        // Local memory accesses >4B were split into 4B chunks
        if (data_size >= 4) {
          data_size_coales = 4;           /* [한국어] 4B 단위로 분할하여 coalescing */
          num_accesses = data_size / 4;   /* [한국어] 분할 횟수 = 데이터 크기 / 4B */
        }
        // Otherwise keep the same data_size for sub-4B access to local memory
        /* [한국어] 4B 미만 로컬 메모리 접근은 분할 없이 그대로 처리 */
      }

      assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD);
      /* [한국어] 명령어당 스레드별 최대 접근 수 초과 방지 */

      //            for(unsigned access=0; access<num_accesses; access++) {
      for (unsigned access = 0;
           (access < MAX_ACCESSES_PER_INSN_PER_THREAD) &&
           (m_per_scalar_thread[thread].memreqaddr[access] != 0);
           access++) {
        /* [한국어] 이 스레드의 각 접근 주소에 대해 세그먼트 분류 처리
         * memreqaddr[0]은 첫 번째 주소, 로컬 메모리 분할 시 [1],[2]...에 추가 주소 */
        new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[access];
        /* [한국어] 이 접근의 실제 메모리 주소 */
        new_addr_type block_address =
            line_size_based_tag_func(addr, segment_size);
        /* [한국어] 이 주소가 속하는 세그먼트의 시작 주소 */
        unsigned chunk =
            (addr & 127) / 32;  // which 32-byte chunk within in a 128-byte
                                // chunk does this thread access?
        /* [한국어] 128B 단위 내에서 이 주소가 속하는 32B 청크 번호 (0~3)
         * addr & 127 = 128B 경계 내 오프셋, /32 = 32B 청크 인덱스 */
        transaction_info &info = subwarp_transactions[block_address];
        /* [한국어] 이 세그먼트에 대한 트랜잭션 정보 참조 (없으면 새로 생성) */

        // can only write to one segment
        // it seems like in trace driven, a thread can write to more than one
        // segment assert(block_address ==
        // line_size_based_tag_func(addr+data_size_coales-1,segment_size));
        /* [한국어] 이상적으로는 하나의 세그먼트에만 쓰기 접근이 있어야 하지만,
         * trace-driven 시뮬에서는 세그먼트 경계를 걸치는 경우가 발생할 수 있어 assert 제거 */

        info.chunks.set(chunk); /* [한국어] 이 청크 비트 설정 (4비트 bitset, 청크 0~3) */
        info.active.set(thread); /* [한국어] 이 트랜잭션에 참여하는 스레드 마스크에 추가 */
        unsigned idx = (addr & 127); /* [한국어] 128B 경계 내 바이트 오프셋 */
        for (unsigned i = 0; i < data_size_coales; i++)
          if ((idx + i) < MAX_MEMORY_ACCESS_SIZE) info.bytes.set(idx + i);
        /* [한국어] 이 접근이 차지하는 바이트들을 byte 마스크에 표시
         * MAX_MEMORY_ACCESS_SIZE = 128B (한 트랜잭션의 최대 크기) */

        // it seems like in trace driven, a thread can write to more than one
        // segment handle this special case
        if (block_address != line_size_based_tag_func(
                                 addr + data_size_coales - 1, segment_size)) {
          /* [한국어] 접근이 세그먼트 경계를 걸치는 특수 케이스 처리 (trace-driven에서 발생 가능) */
          addr = addr + data_size_coales - 1; /* [한국어] 데이터의 마지막 바이트 주소로 이동 */
          new_addr_type block_address =
              line_size_based_tag_func(addr, segment_size);
          /* [한국어] 마지막 바이트가 속하는 (다른) 세그먼트 주소 */
          unsigned chunk = (addr & 127) / 32; /* [한국어] 마지막 바이트의 청크 번호 */
          transaction_info &info = subwarp_transactions[block_address];
          /* [한국어] 두 번째 세그먼트의 트랜잭션 정보 참조 */
          info.chunks.set(chunk); /* [한국어] 두 번째 세그먼트의 청크 비트 설정 */
          info.active.set(thread); /* [한국어] 이 스레드를 두 번째 트랜잭션에도 참여시킴 */
          unsigned idx = (addr & 127); /* [한국어] 두 번째 세그먼트 내 바이트 오프셋 */
          for (unsigned i = 0; i < data_size_coales; i++)
            if ((idx + i) < MAX_MEMORY_ACCESS_SIZE) info.bytes.set(idx + i);
          /* [한국어] 두 번째 세그먼트에서의 바이트 마스크 설정 */
        }
      }
    }

    // step 2: reduce each transaction size, if possible
    std::map<new_addr_type, transaction_info>::iterator t;
    for (t = subwarp_transactions.begin(); t != subwarp_transactions.end();
         t++) {
      /* [한국어] 이 subwarp의 모든 세그먼트 트랜잭션을 최적화하여 m_accessq에 추가 */
      new_addr_type addr = t->first;        /* [한국어] 세그먼트 기준 주소 */
      const transaction_info &info = t->second; /* [한국어] 이 세그먼트의 트랜잭션 정보 */

      memory_coalescing_arch_reduce_and_send(is_write, access_type, info, addr,
                                             segment_size);
      /* [한국어] 트랜잭션 크기를 가능한 만큼 줄인 후 m_accessq에 push_back */
    }
  }
}

/*
 * [한국어]
 * warp_inst_t::memory_coalescing_arch_atomic - 원자 명령어의 메모리 coalescing 처리
 *
 * @is_write: 원자 연산은 RMW이므로 일반적으로 쓰기 포함. access_type과 함께 전달.
 * @access_type: GLOBAL_ACC_R 또는 GLOBAL_ACC_W (원자는 전역 메모리만 허용).
 * @return: 없음. 결과는 m_accessq에 mem_access_t로 추가됨.
 *
 * 원자 명령어(atomicAdd, atomicCAS 등)의 메모리 접근 coalescing을 처리한다.
 * 일반 memory_coalescing_arch()와의 핵심 차이:
 *   - 원자 연산은 같은 세그먼트에 여러 스레드가 접근하더라도 byte가 겹치면 안 됨.
 *     각 스레드가 서로 다른 바이트를 원자적으로 수정해야 read-modify-write 의미론이 유지됨.
 *   - 따라서 하나의 세그먼트 주소에 대해 여러 transaction_info 객체가 생성될 수 있음.
 *     (일반 coalescing: map<addr, transaction_info> vs 원자: map<addr, list<transaction_info>>)
 *   - byte 충돌이 없는 기존 트랜잭션이 있으면 합류, 없으면 새 트랜잭션 생성.
 *
 * 아키텍처별 segment_size 결정 로직은 memory_coalescing_arch()와 동일.
 * 실행 컨텍스트: generate_mem_accesses()에서 isatomic()이 true일 때 호출.
 *
 * 호출 체인:
 *   warp_inst_t::generate_mem_accesses() → [이 함수]
 *   → memory_coalescing_arch_reduce_and_send() → m_accessq.push_back()
 */
void warp_inst_t::memory_coalescing_arch_atomic(bool is_write,
                                                mem_access_type access_type) {
  assert(space.get_type() ==
         global_space);  // Atomics allowed only for global memory
  /* [한국어] 원자 연산은 전역 메모리에만 허용 (shared memory 원자는 별도 처리) */

  // see the CUDA manual where it discusses coalescing rules before reading this
  unsigned segment_size = 0;                      /* [한국어] 세그먼트 크기 (바이트) */
  unsigned warp_parts = m_config->mem_warp_parts; /* [한국어] warp 파트 수 */
  bool sector_segment_size = false;               /* [한국어] sector(32B) 모드 여부 */

  if (m_config->gpgpu_coalesce_arch >= 20 &&
      m_config->gpgpu_coalesce_arch < 39) {
    // Fermi and Kepler, L1 is normal and L2 is sector
    if (m_config->gmem_skip_L1D || cache_op == CACHE_GLOBAL)
      sector_segment_size = true;  /* [한국어] L1 우회 또는 CACHE_GLOBAL: sector(32B) */
    else
      sector_segment_size = false; /* [한국어] L1 경유: normal(128B) */
  } else if (m_config->gpgpu_coalesce_arch >= 40) {
    // Maxwell, Pascal and Volta, L1 and L2 are sectors
    // all requests should be 32 bytes
    sector_segment_size = true; /* [한국어] Maxwell 이상: 항상 sector(32B) */
  }

  switch (data_size) {
    case 1:
      segment_size = 32; /* [한국어] 1B: 32B 세그먼트 */
      break;
    case 2:
      segment_size = sector_segment_size ? 32 : 64; /* [한국어] 2B: sector=32B, normal=64B */
      break;
    case 4:
    case 8:
    case 16:
      segment_size = sector_segment_size ? 32 : 128; /* [한국어] 4/8/16B: sector=32B, normal=128B */
      break;
  }
  unsigned subwarp_size = m_config->warp_size / warp_parts;
  /* [한국어] subwarp 크기 */

  for (unsigned subwarp = 0; subwarp < warp_parts; subwarp++) {
    /* [한국어] 각 subwarp 파트 처리 */
    std::map<new_addr_type, std::list<transaction_info> >
        subwarp_transactions;  // each block addr maps to a list of transactions
    /* [한국어] 원자 coalescing 특징: 같은 세그먼트 주소에 여러 트랜잭션이 생길 수 있음
     * (byte 충돌 없는 스레드들끼리 묶이고, 충돌 시 새 트랜잭션 생성) */

    // step 1: find all transactions generated by this subwarp
    for (unsigned thread = subwarp * subwarp_size;
         thread < subwarp_size * (subwarp + 1); thread++) {
      /* [한국어] 이 subwarp의 스레드들을 순회하며 각 스레드의 원자 접근 분류 */
      if (!active(thread)) continue; /* [한국어] 비활성 스레드 건너뜀 */

      new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
      /* [한국어] 이 스레드의 원자 연산 대상 주소 (원자는 1개 주소만 허용) */
      new_addr_type block_address =
          line_size_based_tag_func(addr, segment_size);
      /* [한국어] 이 주소가 속하는 세그먼트의 시작 주소 */
      unsigned chunk =
          (addr & 127) / 32;  // which 32-byte chunk within in a 128-byte chunk
                              // does this thread access?
      /* [한국어] 128B 단위 내 32B 청크 번호 (0~3) */

      // can only write to one segment
      assert(block_address ==
             line_size_based_tag_func(addr + data_size - 1, segment_size));
      /* [한국어] 원자 연산은 세그먼트 경계를 걸치면 안 됨 — 단일 세그먼트 내에서만 RMW 가능 */

      // Find a transaction that does not conflict with this thread's accesses
      bool new_transaction = true; /* [한국어] 새 트랜잭션이 필요한지 여부 */
      std::list<transaction_info>::iterator it;
      transaction_info *info; /* [한국어] 이 스레드가 참여할 트랜잭션 포인터 */
      for (it = subwarp_transactions[block_address].begin();
           it != subwarp_transactions[block_address].end(); it++) {
        /* [한국어] 이 세그먼트의 기존 트랜잭션 리스트를 순회하며 byte 충돌 없는 것을 탐색 */
        unsigned idx = (addr & 127); /* [한국어] 128B 내 바이트 오프셋 */
        if (not it->test_bytes(idx, idx + data_size - 1)) {
          /* [한국어] 이 스레드가 접근하는 byte 범위 [idx, idx+data_size-1]에 겹치는 byte가 없으면 합류 가능 */
          new_transaction = false; /* [한국어] 새 트랜잭션 불필요 */
          info = &(*it);           /* [한국어] 이 트랜잭션에 합류 */
          break;
        }
      }
      if (new_transaction) {
        // Need a new transaction
        subwarp_transactions[block_address].push_back(transaction_info());
        /* [한국어] byte 충돌로 인해 새 트랜잭션 생성 — 이 스레드만을 위한 별도 트랜잭션 */
        info = &subwarp_transactions[block_address].back();
        /* [한국어] 방금 추가한 새 트랜잭션의 포인터 획득 */
      }
      assert(info); /* [한국어] info가 유효한 포인터인지 확인 */

      info->chunks.set(chunk); /* [한국어] 청크 비트 설정 */
      info->active.set(thread); /* [한국어] 활성 스레드 마스크에 추가 */
      unsigned idx = (addr & 127); /* [한국어] 바이트 오프셋 */
      for (unsigned i = 0; i < data_size; i++) {
        assert(!info->bytes.test(idx + i));
        /* [한국어] byte 마스크에 이미 설정된 bit가 없어야 함 — 원자 연산 byte 겹침 방지 검증 */
        info->bytes.set(idx + i); /* [한국어] 이 스레드가 접근하는 byte 마스크 설정 */
      }
    }

    // step 2: reduce each transaction size, if possible
    std::map<new_addr_type, std::list<transaction_info> >::iterator t_list;
    for (t_list = subwarp_transactions.begin();
         t_list != subwarp_transactions.end(); t_list++) {
      // For each block addr
      new_addr_type addr = t_list->first; /* [한국어] 세그먼트 기준 주소 */
      const std::list<transaction_info> &transaction_list = t_list->second;
      /* [한국어] 이 세그먼트에 대한 트랜잭션 리스트 */

      std::list<transaction_info>::const_iterator t;
      for (t = transaction_list.begin(); t != transaction_list.end(); t++) {
        // For each transaction
        const transaction_info &info = *t; /* [한국어] 각 개별 트랜잭션 */
        memory_coalescing_arch_reduce_and_send(is_write, access_type, info,
                                               addr, segment_size);
        /* [한국어] 트랜잭션 크기 최적화 후 m_accessq에 추가 */
      }
    }
  }
}

/*
 * [한국어]
 * warp_inst_t::memory_coalescing_arch_reduce_and_send - 세그먼트 크기 최적화 후 접근 큐에 추가
 *
 * @is_write: 쓰기 여부.
 * @access_type: 메모리 접근 분류.
 * @info: 이 트랜잭션의 청크 비트맵·활성 스레드 마스크·바이트 마스크.
 * @addr: 세그먼트 시작 주소 (segment_size 정렬된 주소).
 * @segment_size: 초기 세그먼트 크기 (32/64/128 바이트).
 * @return: 없음. m_accessq에 최적화된 mem_access_t를 push_back.
 *
 * coalescing 후 결정된 트랜잭션의 실제 크기를 가능한 한 줄인다.
 * 128B 세그먼트에서 하위 또는 상위 64B만 접근된다면 64B로 축소.
 * 64B에서 하위 또는 상위 32B만 접근된다면 32B로 축소.
 * 이를 통해 메모리 대역폭 사용을 최소화한다.
 *
 * 청크(chunk) 비트맵(q[0..3]) 의미:
 *   q[0]: 0~31B (하위 1/4), q[1]: 32~63B (하위 2/4)
 *   q[2]: 64~95B (상위 1/4), q[3]: 96~127B (상위 2/4)
 * 압축 로직:
 *   128B: lower(q[0]||q[1]) vs upper(q[2]||q[3]) → 한쪽만이면 64B로 축소
 *   64B: 128B 내 위치(addr%128==0 vs 64)에 따라 q 비트를 h 비트로 매핑
 *   64B: lower(h[0]) vs upper(h[1]) → 한쪽만이면 32B로 축소 + addr 조정
 *
 * 실행 컨텍스트: memory_coalescing_arch() 및 memory_coalescing_arch_atomic()에서 호출.
 *
 * 호출 체인:
 *   memory_coalescing_arch() → [이 함수] → m_accessq.push_back()
 *   memory_coalescing_arch_atomic() → [이 함수] → m_accessq.push_back()
 */
void warp_inst_t::memory_coalescing_arch_reduce_and_send(
    bool is_write, mem_access_type access_type, const transaction_info &info,
    new_addr_type addr, unsigned segment_size) {
  assert((addr & (segment_size - 1)) == 0);
  /* [한국어] addr이 segment_size 경계로 정렬되어 있는지 검증 */

  const std::bitset<4> &q = info.chunks;
  /* [한국어] 청크 비트맵: 128B 세그먼트 내에서 어느 32B 청크가 사용되는지 표시 (bit 0~3) */
  assert(q.count() >= 1); /* [한국어] 최소 1개 청크는 사용되어야 함 */
  std::bitset<2> h;  // halves (used to check if 64 byte segment can be
                     // compressed into a single 32 byte segment)
  /* [한국어] h: 64B 단위 내에서 상/하위 32B 중 어느 쪽이 사용되는지 표시 */

  unsigned size = segment_size; /* [한국어] 초기 트랜잭션 크기 (segment_size에서 시작, 축소 가능) */
  if (segment_size == 128) {
    /* [한국어] 128B 세그먼트: 하위/상위 64B 중 한쪽만 접근 시 64B로 축소 */
    bool lower_half_used = q[0] || q[1]; /* [한국어] 하위 64B(청크 0,1) 사용 여부 */
    bool upper_half_used = q[2] || q[3]; /* [한국어] 상위 64B(청크 2,3) 사용 여부 */
    if (lower_half_used && !upper_half_used) {
      // only lower 64 bytes used
      size = 64; /* [한국어] 하위 64B만 사용 → 트랜잭션 크기를 64B로 축소 (addr 변경 없음) */
      if (q[0]) h.set(0); /* [한국어] 청크 0(0~31B)이 사용됨 → h의 하위 비트 설정 */
      if (q[1]) h.set(1); /* [한국어] 청크 1(32~63B)이 사용됨 → h의 상위 비트 설정 */
    } else if ((!lower_half_used) && upper_half_used) {
      // only upper 64 bytes used
      addr = addr + 64; /* [한국어] 상위 64B만 사용 → 트랜잭션 시작 주소를 +64 조정 */
      size = 64;        /* [한국어] 크기를 64B로 축소 */
      if (q[2]) h.set(0); /* [한국어] 청크 2(64~95B) → h의 하위 비트 (64B 블록 내 0~31B에 해당) */
      if (q[3]) h.set(1); /* [한국어] 청크 3(96~127B) → h의 상위 비트 (64B 블록 내 32~63B에 해당) */
    } else {
      assert(lower_half_used && upper_half_used);
      /* [한국어] 상하 64B 모두 사용 → 128B 전체 트랜잭션 필요, h 비트 설정 불필요 */
    }
  } else if (segment_size == 64) {
    // need to set halves
    /* [한국어] 64B 세그먼트: 128B 정렬 위치에 따라 q 비트를 h 비트로 매핑 */
    if ((addr % 128) == 0) {
      /* [한국어] 이 64B 세그먼트가 128B 블록의 하위 절반인 경우 */
      if (q[0]) h.set(0); /* [한국어] 청크 0(0~31B) → h 하위 */
      if (q[1]) h.set(1); /* [한국어] 청크 1(32~63B) → h 상위 */
    } else {
      assert((addr % 128) == 64);
      /* [한국어] 이 64B 세그먼트가 128B 블록의 상위 절반인 경우 */
      if (q[2]) h.set(0); /* [한국어] 청크 2(64~95B) → h 하위 */
      if (q[3]) h.set(1); /* [한국어] 청크 3(96~127B) → h 상위 */
    }
  }
  if (size == 64) {
    /* [한국어] 64B 트랜잭션에서 추가 압축 시도: 32B로 줄일 수 있는지 확인 */
    bool lower_half_used = h[0]; /* [한국어] 하위 32B(h[0]) 사용 여부 */
    bool upper_half_used = h[1]; /* [한국어] 상위 32B(h[1]) 사용 여부 */
    if (lower_half_used && !upper_half_used) {
      size = 32; /* [한국어] 하위 32B만 사용 → 32B로 추가 축소 (addr 변경 없음) */
    } else if ((!lower_half_used) && upper_half_used) {
      addr = addr + 32; /* [한국어] 상위 32B만 사용 → 시작 주소 +32 조정 */
      size = 32;        /* [한국어] 32B로 추가 축소 */
    } else {
      assert(lower_half_used && upper_half_used);
      /* [한국어] 상하 32B 모두 사용 → 64B 트랜잭션 유지 */
    }
  }
  m_accessq.push_back(mem_access_t(access_type, addr, size, is_write,
                                   info.active, info.bytes, info.chunks,
                                   m_config->gpgpu_ctx));
  /* [한국어] 최적화된 메모리 접근 요청을 접근 큐에 추가
   * 인자: 접근타입, 최종주소(축소 후), 최종크기(축소 후), 쓰기여부,
   *        활성스레드마스크, 바이트마스크, 청크비트맵, 컨텍스트 */
}

/*
 * [한국어]
 * warp_inst_t::completed - 명령어 완료 시 지연시간(latency) 통계 기록
 *
 * @cycle: 명령어가 완료된 현재 시뮬레이션 사이클 번호.
 * @return: 없음.
 *
 * 명령어가 파이프라인을 통과하여 완료될 때 호출된다.
 * issue_cycle에서 현재 cycle까지의 차이를 latency로 계산하고,
 * 활성 스레드 수(active_count())를 곱하여 가중 latency를 통계에 기록한다.
 * 이 통계는 PTX 파일·라인별로 누적되어 병목 명령어 분석에 활용된다.
 * assert로 underflow(cycle < issue_cycle) 감지: 사이클 역전 방지.
 * 실행 컨텍스트: 타이밍 시뮬의 writeback 단계.
 *
 * 호출 체인:
 *   shader_core_ctx::writeback() → [이 함수]
 */
void warp_inst_t::completed(unsigned long long cycle) const {
  unsigned long long latency = cycle - issue_cycle;
  /* [한국어] 이 명령어의 총 실행 지연시간 = 완료 사이클 - issue 사이클 */
  assert(latency <= cycle);  // underflow detection
  /* [한국어] latency > cycle이면 사이클 언더플로(issue_cycle > cycle) — 논리 오류 감지 */
  m_config->gpgpu_ctx->stats->ptx_file_line_stats_add_latency(
      pc, latency * active_count());
  /* [한국어] PTX 파일·라인별 가중 latency 통계 누적
   * 가중치 = 활성 스레드 수: 더 많은 스레드가 실행한 명령어가 더 큰 영향을 미침 */
}

/*
 * [한국어]
 * kernel_info_t::kernel_info_t (기본 생성자) - CUDA 커널 실행 정보 초기화
 *
 * @gridDim: CUDA 커널의 그리드 차원 (x, y, z 방향 CTA 수).
 *            cuLaunchKernel()의 gridDim 인자에서 전달됨.
 * @blockDim: CUDA 커널의 블록 차원 (x, y, z 방향 스레드 수).
 *             cuLaunchKernel()의 blockDim 인자에서 전달됨.
 * @entry: PTX 함수 엔트리 포인터. 커널 이름, 파라미터 크기, PTX IR 접근에 사용.
 * @streamID: 이 커널이 실행될 CUDA 스트림 ID.
 * @return: 없음. (생성자)
 *
 * 스트림 관리자(g_stream_manager)를 통해 CUDA 커널이 제출될 때 생성된다.
 * 주요 초기화 내용:
 *   1) 그리드/블록 차원 저장
 *   2) CTA 스케줄링 포인터(m_next_cta) 초기화 — (0,0,0)부터 시작
 *   3) 전역 커널 uid 발급 (kernel_info_m_next_uid 카운터 증가)
 *   4) 파라미터 메모리 공간 생성 (8KB 페이지, 64KB 초기 용량)
 *   5) CDP(CUDA Dynamic Parallelism) parent 포인터를 NULL로 초기화
 *   6) 커널 launch latency 계산 (g_kernel_launch_latency + 블록 수 × g_TB_launch_latency)
 * 실행 컨텍스트: 시뮬레이터의 stream_manager가 커널을 큐에 넣을 때 호출.
 *
 * 호출 체인:
 *   stream_manager::push() → [이 함수]
 */
kernel_info_t::kernel_info_t(dim3 gridDim, dim3 blockDim,
                             class function_info *entry,
                             unsigned long long streamID) {
  m_kernel_entry = entry;    /* [한국어] PTX 함수 엔트리 저장 — 커널 이름·파라미터 크기·PTX IR 접근 */
  m_grid_dim = gridDim;      /* [한국어] 그리드 차원 저장 (x,y,z 방향 CTA 수) */
  m_block_dim = blockDim;    /* [한국어] 블록 차원 저장 (x,y,z 방향 스레드 수) */
  m_next_cta.x = 0;          /* [한국어] 다음 스케줄할 CTA의 x 좌표 초기화 */
  m_next_cta.y = 0;          /* [한국어] 다음 스케줄할 CTA의 y 좌표 초기화 */
  m_next_cta.z = 0;          /* [한국어] 다음 스케줄할 CTA의 z 좌표 초기화 */
  m_next_tid = m_next_cta;   /* [한국어] 다음 스레드 ID 포인터도 (0,0,0)으로 초기화 */
  m_num_cores_running = 0;   /* [한국어] 현재 이 커널을 실행 중인 SM 수 초기화 */
  m_uid = (entry->gpgpu_ctx->kernel_info_m_next_uid)++;
  /* [한국어] 전역 커널 uid 카운터를 후위 증가하여 이 커널에 고유 ID 부여 */
  m_streamID = streamID;     /* [한국어] 이 커널이 실행될 CUDA 스트림 ID 저장 */
  m_param_mem = new memory_space_impl<8192>("param", 64 * 1024);
  /* [한국어] 커널 파라미터 전달용 메모리 공간 생성
   * 8192B 페이지 크기, 64KB 초기 용량 — 커널 인자들이 여기에 복사됨 */

  // Jin: parent and child kernel management for CDP
  m_parent_kernel = NULL;
  /* [한국어] CDP(CUDA Dynamic Parallelism) 부모 커널 포인터 초기화.
   * 이 커널이 다른 커널로부터 GPU 디바이스 코드에서 launch된 경우 set_parent()로 설정됨 */

  // Jin: launch latency management
  m_launch_latency = entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency;
  /* [한국어] 커널 launch 지연 시간 (사이클 단위). 커널이 실제 실행 시작까지 대기해야 하는 시간 */

  m_kernel_TB_latency =
      entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency +
      num_blocks() * entry->gpgpu_ctx->device_runtime->g_TB_launch_latency;
  /* [한국어] 전체 커널 TB(Thread Block) launch 지연시간 =
   * 커널 기본 launch 지연 + (총 블록 수 × 블록당 launch 지연)
   * CDP에서 자식 커널의 첫 CTA가 실행되기까지의 총 지연을 모델링 */

  cache_config_set = false;
  /* [한국어] 이 커널의 공유 메모리/L1 캐시 분할 설정 여부 초기화
   * true면 커널별 cache 설정이 적용됨 */
}

/*A snapshot of the texture mappings needs to be stored in the kernel's info as
kernels should use the texture bindings seen at the time of launch and textures
 can be bound/unbound asynchronously with respect to streams. */
/*
 * [한국어]
 * kernel_info_t::kernel_info_t (텍스처 스냅샷 생성자) - 텍스처 바인딩 스냅샷 포함 초기화
 *
 * @gridDim: 그리드 차원 (x,y,z 방향 CTA 수).
 * @blockDim: 블록 차원 (x,y,z 방향 스레드 수).
 * @entry: PTX 함수 엔트리 포인터.
 * @nameToCudaArray: launch 시점의 텍스처 이름 → cudaArray 포인터 스냅샷.
 *                   커널 실행 중 cudaBindTexture()가 변경되더라도 영향받지 않게 복사.
 * @nameToTextureInfo: launch 시점의 텍스처 이름 → 텍스처 속성(채널, 크기) 스냅샷.
 * @return: 없음. (생성자)
 *
 * 기본 생성자와 동일하게 초기화하되, 텍스처 바인딩 스냅샷을 추가로 저장한다.
 * CUDA 스트림 모델에서 텍스처는 비동기적으로 바인딩/언바인딩될 수 있으므로,
 * 커널 launch 시점의 텍스처 상태를 스냅샷으로 저장해야 커널 실행 중
 * 일관된 텍스처 접근이 가능하다 (영어 주석 참조).
 * streamID가 없는 버전 — stream manager의 다른 경로에서 사용됨.
 * 실행 컨텍스트: 텍스처를 사용하는 커널이 launch될 때.
 *
 * 호출 체인:
 *   stream_manager::push() (텍스처 버전) → [이 함수]
 */
kernel_info_t::kernel_info_t(
    dim3 gridDim, dim3 blockDim, class function_info *entry,
    std::map<std::string, const struct cudaArray *> nameToCudaArray,
    std::map<std::string, const struct textureInfo *> nameToTextureInfo) {
  m_kernel_entry = entry;    /* [한국어] PTX 함수 엔트리 저장 */
  m_grid_dim = gridDim;      /* [한국어] 그리드 차원 저장 */
  m_block_dim = blockDim;    /* [한국어] 블록 차원 저장 */
  m_next_cta.x = 0;          /* [한국어] 다음 CTA x 좌표 초기화 */
  m_next_cta.y = 0;          /* [한국어] 다음 CTA y 좌표 초기화 */
  m_next_cta.z = 0;          /* [한국어] 다음 CTA z 좌표 초기화 */
  m_next_tid = m_next_cta;   /* [한국어] 다음 스레드 ID 포인터 초기화 */
  m_num_cores_running = 0;   /* [한국어] 실행 중인 SM 수 초기화 */
  m_uid = (entry->gpgpu_ctx->kernel_info_m_next_uid)++;
  /* [한국어] 전역 커널 uid 발급 */
  m_param_mem = new memory_space_impl<8192>("param", 64 * 1024);
  /* [한국어] 커널 파라미터 메모리 공간 생성 */

  // Jin: parent and child kernel management for CDP
  m_parent_kernel = NULL;    /* [한국어] CDP 부모 커널 포인터 초기화 */

  // Jin: launch latency management
  m_launch_latency = entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency;
  /* [한국어] 커널 launch 지연시간 설정 */

  m_kernel_TB_latency =
      entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency +
      num_blocks() * entry->gpgpu_ctx->device_runtime->g_TB_launch_latency;
  /* [한국어] 전체 TB launch 지연시간 계산 */

  cache_config_set = false;  /* [한국어] 캐시 설정 미적용 초기화 */
  m_NameToCudaArray = nameToCudaArray;
  /* [한국어] launch 시점의 텍스처 이름 → cudaArray 스냅샷 저장
   * 복사본이므로 이후 cudaBindTexture 변경에 영향받지 않음 */
  m_NameToTextureInfo = nameToTextureInfo;
  /* [한국어] launch 시점의 텍스처 이름 → 텍스처 속성 스냅샷 저장 */
}

/*
 * [한국어]
 * kernel_info_t::~kernel_info_t - 커널 실행 정보 객체 소멸자
 *
 * @return: 없음.
 *
 * 커널 실행이 완전히 완료된 후 kernel_info_t 객체를 해제한다.
 * 파괴 순서:
 *   1) 활성 스레드가 없음을 검증 (m_active_threads.empty())
 *   2) 이 커널의 모든 CTA에 연결된 CUDA 스트림들을 해제
 *   3) 파라미터 메모리 공간 해제
 * 실행 컨텍스트: 커널 완료 후 stream_manager가 이 객체를 delete할 때.
 *
 * 호출 체인:
 *   stream_manager::register_finished_kernel() → [이 함수]
 */
kernel_info_t::~kernel_info_t() {
  assert(m_active_threads.empty()); /* [한국어] 아직 활성 스레드가 있으면 소멸 시점 오류 */
  destroy_cta_streams();            /* [한국어] 이 커널의 모든 CTA 스트림 해제 (CDP 스트림 포함) */
  delete m_param_mem;               /* [한국어] 커널 파라미터 메모리 공간 해제 */
}

/*
 * [한국어]
 * kernel_info_t::name - 커널 함수 이름 반환
 *
 * @return: PTX 함수 엔트리에서 가져온 커널 함수 이름 문자열.
 *
 * 통계 출력, 디버그 로그, 체크포인트 파일명 생성에 사용된다.
 */
std::string kernel_info_t::name() const { return m_kernel_entry->get_name(); }

// Jin: parent and child kernel management for CDP
/*
 * [한국어]
 * kernel_info_t::set_parent - CDP 부모-자식 커널 관계 설정
 *
 * @parent: 이 커널을 launch한 부모 커널의 포인터.
 * @parent_ctaid: 부모 커널에서 이 커널을 launch한 CTA의 ID (x,y,z).
 * @parent_tid: 부모 커널에서 이 커널을 launch한 스레드의 ID (x,y,z).
 * @return: 없음.
 *
 * CUDA Dynamic Parallelism(CDP)에서 GPU 디바이스 코드가 새 커널을 launch할 때
 * 부모-자식 관계를 양방향으로 설정한다.
 * 이 함수는 자식(this)의 m_parent_kernel을 설정하고,
 * parent->set_child(this)를 통해 부모의 m_child_kernels에도 추가.
 * 부모 커널은 모든 자식이 완료될 때까지 is_finished()가 false.
 * 실행 컨텍스트: CDP 커널 launch 시 device_runtime에서 호출.
 *
 * 호출 체인:
 *   device_runtime::launch_kernel() → [이 함수] → parent->set_child()
 */
void kernel_info_t::set_parent(kernel_info_t *parent, dim3 parent_ctaid,
                               dim3 parent_tid) {
  m_parent_kernel = parent;     /* [한국어] 부모 커널 포인터 저장 */
  m_parent_ctaid = parent_ctaid; /* [한국어] 부모 커널에서 이 커널을 launch한 CTA ID 저장 */
  m_parent_tid = parent_tid;    /* [한국어] 부모 커널에서 이 커널을 launch한 스레드 ID 저장 */
  parent->set_child(this);      /* [한국어] 부모 커널의 자식 목록에 자신(이 커널) 추가 */
}

/*
 * [한국어]
 * kernel_info_t::set_child - CDP 자식 커널을 부모의 목록에 추가
 *
 * @child: 추가할 자식 커널 포인터.
 * @return: 없음.
 *
 * set_parent()에서 내부적으로 호출된다. m_child_kernels 리스트에 추가.
 * 부모는 이 리스트가 빌 때까지 완료되지 않은 것으로 취급된다.
 */
void kernel_info_t::set_child(kernel_info_t *child) {
  m_child_kernels.push_back(child); /* [한국어] 자식 커널 포인터를 리스트 끝에 추가 */
}

/*
 * [한국어]
 * kernel_info_t::remove_child - CDP 자식 커널이 완료됐을 때 목록에서 제거
 *
 * @child: 제거할 자식 커널 포인터. 반드시 목록에 존재해야 한다.
 * @return: 없음. child가 목록에 없으면 assert로 중단.
 *
 * 자식 커널이 완료되면 notify_parent_finished()를 통해 부모에게 알리고,
 * 부모는 이 함수로 해당 자식을 m_child_kernels 리스트에서 제거한다.
 * 리스트가 비면 children_all_finished()가 true를 반환하여 부모도 완료 가능.
 *
 * 호출 체인:
 *   kernel_info_t::notify_parent_finished() → [이 함수]
 */
void kernel_info_t::remove_child(kernel_info_t *child) {
  assert(std::find(m_child_kernels.begin(), m_child_kernels.end(), child) !=
         m_child_kernels.end());
  /* [한국어] 제거하려는 자식이 실제로 목록에 있는지 검증 — 없으면 오류 */
  m_child_kernels.remove(child); /* [한국어] 리스트에서 자식 커널 포인터 제거 */
}

/*
 * [한국어]
 * kernel_info_t::is_finished - 커널(자신 + 모든 자식)이 완전히 완료되었는지 확인
 *
 * @return: true면 이 커널과 모든 자식 커널이 완료됨.
 *          false면 아직 실행 중.
 *
 * done()은 이 커널 자체의 모든 CTA가 완료됐는지 확인.
 * children_all_finished()는 m_child_kernels 리스트가 비었는지 확인.
 * 두 조건이 모두 true일 때만 커널이 완전히 끝난 것으로 판단.
 * 실행 컨텍스트: stream_manager의 완료 처리 루프에서 매 사이클 확인.
 */
bool kernel_info_t::is_finished() {
  if (done() && children_all_finished())
    return true; /* [한국어] 자신의 CTA도 완료되고 모든 자식 커널도 완료됨 */
  else
    return false; /* [한국어] 아직 실행 중인 CTA 또는 자식 커널이 남아있음 */
}

/*
 * [한국어]
 * kernel_info_t::children_all_finished - 모든 자식 커널 완료 여부 확인
 *
 * @return: true면 자식 커널이 없거나 모두 완료됨.
 *          false면 완료되지 않은 자식 커널이 남아있음.
 *
 * m_child_kernels 리스트가 빈 상태이면 모든 자식이 완료된 것.
 * 자식 커널이 완료될 때마다 remove_child()를 호출하여 리스트에서 제거되므로,
 * 리스트가 비면 모든 자식이 완료된 것이다.
 */
bool kernel_info_t::children_all_finished() {
  if (!m_child_kernels.empty()) return false; /* [한국어] 아직 자식 커널이 리스트에 있음 → 미완료 */

  return true; /* [한국어] 자식 리스트가 비어있음 → 모든 자식 완료 */
}

/*
 * [한국어]
 * kernel_info_t::notify_parent_finished - 자식 커널 완료 시 부모에게 통보
 *
 * @return: 없음.
 *
 * CDP 자식 커널이 완료되었을 때 호출된다. 처리 내용:
 *   1) 전역 파라미터 크기 카운터에서 이 커널의 파라미터 크기 감산 (256B 정렬)
 *   2) 부모 커널의 m_child_kernels 리스트에서 자신(이 커널)을 제거
 *   3) 부모 커널의 UID로 stream_manager에 완료 등록
 *      → 부모의 is_finished() 재확인 트리거
 * 부모 커널이 없으면 (m_parent_kernel == NULL) 아무 동작도 하지 않는다.
 * 실행 컨텍스트: 커널 실행 완료 시점.
 *
 * 호출 체인:
 *   shader_core_ctx::kernel_is_done() → [이 함수]
 *   → m_parent_kernel->remove_child()
 *   → stream_manager::register_finished_kernel()
 */
void kernel_info_t::notify_parent_finished() {
  if (m_parent_kernel) { /* [한국어] 부모 커널이 있는 경우(CDP 자식 커널)만 처리 */
    m_kernel_entry->gpgpu_ctx->device_runtime->g_total_param_size -=
        ((m_kernel_entry->get_args_aligned_size() + 255) / 256 * 256);
    /* [한국어] 전역 파라미터 크기 누적에서 이 커널의 파라미터 크기 제거.
     * (args_aligned_size + 255) / 256 * 256 = 256B 단위로 올림 정렬한 크기 */
    m_parent_kernel->remove_child(this);
    /* [한국어] 부모 커널의 자식 목록에서 이 커널 제거 */
    m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager
        ->register_finished_kernel(m_parent_kernel->get_uid());
    /* [한국어] 부모 커널 UID로 stream_manager에 완료 알림 등록
     * → stream_manager가 부모 커널의 is_finished()를 재확인 */
  }
}

/*
 * [한국어]
 * kernel_info_t::create_stream_cta - CDP CTA에 새 CUDA 스트림 생성
 *
 * @ctaid: 스트림을 생성할 CTA의 ID (x,y,z). 반드시 기본 스트림이 이미 존재해야 함.
 * @return: 새로 생성된 CUstream_st 포인터.
 *
 * CDP(CUDA Dynamic Parallelism)에서 GPU 디바이스 코드의 CTA가 cudaStreamCreate()를
 * 호출할 때 사용된다. 해당 CTA에 추가 스트림을 생성하여 m_cta_streams[ctaid]에 추가.
 * 기본 스트림(get_default_stream_cta로 생성된 것)이 먼저 있어야 한다.
 * stream_manager에도 등록하여 시뮬레이터 전역 스트림 추적에 포함시킨다.
 * 실행 컨텍스트: CDP 커널 실행 중 디바이스 코드가 스트림을 생성할 때.
 *
 * 호출 체인:
 *   device_runtime::cudaStreamCreate() → [이 함수]
 */
CUstream_st *kernel_info_t::create_stream_cta(dim3 ctaid) {
  assert(get_default_stream_cta(ctaid)); /* [한국어] 기본 스트림이 이미 존재하는지 확인 */
  CUstream_st *stream = new CUstream_st(); /* [한국어] 새 CUDA 스트림 객체 생성 */
  m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager->add_stream(stream);
  /* [한국어] 전역 stream_manager에 이 스트림 등록 */
  assert(m_cta_streams.find(ctaid) != m_cta_streams.end());
  /* [한국어] 이 CTA의 스트림 리스트가 존재하는지 확인 */
  assert(m_cta_streams[ctaid].size() >= 1);  // must have default stream
  /* [한국어] 최소 1개(기본 스트림)가 있어야 함 */
  m_cta_streams[ctaid].push_back(stream); /* [한국어] 이 CTA의 스트림 리스트에 새 스트림 추가 */

  return stream; /* [한국어] 새로 생성된 스트림 포인터 반환 */
}

/*
 * [한국어]
 * kernel_info_t::get_default_stream_cta - CTA의 기본 CUDA 스트림 반환 또는 생성
 *
 * @ctaid: 기본 스트림을 가져올 CTA의 ID (x,y,z).
 * @return: 해당 CTA의 기본 스트림 포인터.
 *          처음 호출 시 새 스트림을 생성하고 반환.
 *
 * CDP에서 CTA가 처음 스트림을 필요로 할 때 기본 스트림을 생성한다.
 * m_cta_streams 맵에 해당 CTA ID가 없으면 빈 리스트를 만들고 기본 스트림 생성.
 * 이미 있으면 리스트의 첫 번째 원소(기본 스트림)를 반환.
 * 실행 컨텍스트: CDP 커널에서 스트림이 처음 필요할 때.
 *
 * 호출 체인:
 *   create_stream_cta() → [이 함수] (기본 스트림 존재 확인)
 *   device_runtime → [이 함수] (기본 스트림 직접 획득)
 */
CUstream_st *kernel_info_t::get_default_stream_cta(dim3 ctaid) {
  if (m_cta_streams.find(ctaid) != m_cta_streams.end()) {
    /* [한국어] 이 CTA의 스트림 리스트가 이미 존재하는 경우 */
    assert(m_cta_streams[ctaid].size() >=
           1);  // already created, must have default stream
    /* [한국어] 최소 1개(기본 스트림)가 있어야 함 */
    return *(m_cta_streams[ctaid].begin()); /* [한국어] 리스트 첫 원소 = 기본 스트림 반환 */
  } else {
    /* [한국어] 이 CTA에 스트림이 없는 경우 — 처음 요청 시 기본 스트림 생성 */
    m_cta_streams[ctaid] = std::list<CUstream_st *>(); /* [한국어] 빈 스트림 리스트 초기화 */
    CUstream_st *stream = new CUstream_st();            /* [한국어] 기본 스트림 객체 생성 */
    m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager->add_stream(
        stream);
    /* [한국어] 전역 stream_manager에 기본 스트림 등록 */
    m_cta_streams[ctaid].push_back(stream); /* [한국어] 이 CTA의 리스트에 기본 스트림 추가 */
    return stream; /* [한국어] 새로 생성된 기본 스트림 반환 */
  }
}

/*
 * [한국어]
 * kernel_info_t::cta_has_stream - CTA가 특정 스트림을 가지고 있는지 확인
 *
 * @ctaid: 확인할 CTA의 ID (x,y,z).
 * @stream: 찾으려는 스트림 포인터.
 * @return: true면 해당 CTA가 이 스트림을 가지고 있음. false면 없음.
 *
 * CDP에서 스트림이 올바른 CTA에 속하는지 검증하는 데 사용.
 */
bool kernel_info_t::cta_has_stream(dim3 ctaid, CUstream_st *stream) {
  if (m_cta_streams.find(ctaid) == m_cta_streams.end()) return false;
  /* [한국어] 이 CTA에 스트림 리스트 자체가 없으면 false */

  std::list<CUstream_st *> &stream_list = m_cta_streams[ctaid];
  /* [한국어] 이 CTA의 스트림 리스트 참조 */
  if (std::find(stream_list.begin(), stream_list.end(), stream) ==
      stream_list.end())
    return false; /* [한국어] 스트림 리스트에서 찾지 못한 경우 */
  else
    return true; /* [한국어] 스트림이 리스트에 있음 */
}

/*
 * [한국어]
 * kernel_info_t::print_parent_info - CDP 부모 커널 정보 출력
 *
 * @return: 없음.
 *
 * 디버그 목적으로 이 커널의 부모 커널 정보를 stdout에 출력한다.
 * 부모 커널이 없으면 (최상위 커널) 아무것도 출력하지 않는다.
 * 출력 형식: "Parent <uid>: '<name>', Block (x,y,z), Thread (x,y,z)"
 */
void kernel_info_t::print_parent_info() {
  if (m_parent_kernel) { /* [한국어] 부모 커널이 있는 경우에만 출력 */
    printf("Parent %d: \'%s\', Block (%d, %d, %d), Thread (%d, %d, %d)\n",
           m_parent_kernel->get_uid(),    /* [한국어] 부모 커널 UID */
           m_parent_kernel->name().c_str(), /* [한국어] 부모 커널 이름 */
           m_parent_ctaid.x, m_parent_ctaid.y, m_parent_ctaid.z, /* [한국어] launch 위치 CTA ID */
           m_parent_tid.x, m_parent_tid.y, m_parent_tid.z); /* [한국어] launch 위치 스레드 ID */
  }
}

/*
 * [한국어]
 * kernel_info_t::destroy_cta_streams - 이 커널의 모든 CTA 스트림 해제
 *
 * @return: 없음.
 *
 * 소멸자(~kernel_info_t)에서 호출. m_cta_streams 맵에 있는 모든 CTA의
 * 모든 스트림을 stream_manager를 통해 해제하고 맵을 클리어한다.
 * 해제 전에 각 스트림의 stream_manager::destroy_stream()을 호출하여
 * 전역 스트림 추적에서도 제거한다.
 * 실행 컨텍스트: 소멸자에서 호출 (단일 시뮬 스레드).
 *
 * 호출 체인:
 *   ~kernel_info_t() → [이 함수]
 *   → stream_manager::destroy_stream()
 */
void kernel_info_t::destroy_cta_streams() {
  printf("Destroy streams for kernel %d: ", get_uid()); /* [한국어] 디버그 출력: 어느 커널의 스트림을 해제하는지 */
  size_t stream_size = 0; /* [한국어] 해제할 전체 스트림 수 집계 (디버그 출력용) */
  for (auto s = m_cta_streams.begin(); s != m_cta_streams.end(); s++) {
    /* [한국어] 각 CTA (s->first = ctaid, s->second = 스트림 리스트)를 순회 */
    stream_size += s->second.size(); /* [한국어] 이 CTA의 스트림 수 누적 */
    for (auto ss = s->second.begin(); ss != s->second.end(); ss++)
      m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager->destroy_stream(
          *ss);
    /* [한국어] 이 CTA의 각 스트림을 stream_manager를 통해 해제 */
    s->second.clear(); /* [한국어] 스트림 리스트 클리어 (포인터만 지움, 객체는 destroy_stream에서 해제) */
  }
  printf("size %lu\n", stream_size); /* [한국어] 해제된 스트림 총 수 출력 */
  m_cta_streams.clear(); /* [한국어] CTA → 스트림 맵 전체 클리어 */
}

/*
 * [한국어]
 * simt_stack::simt_stack - SIMT 스택 초기화
 *
 * @wid: 이 SIMT 스택이 속하는 warp 번호 (0 ~ num_warps-1).
 * @warpSize: warp 크기 (일반적으로 32스레드).
 * @gpu: 최상위 시뮬레이터 포인터. 사이클 카운터·통계 접근에 사용.
 * @return: 없음. (생성자)
 *
 * 각 warp마다 하나의 SIMT 스택이 존재한다. SIMT 스택은 warp의 분기(divergence)를
 * 추적하는 pdom(post-dominator) 기반 스택으로, SIMT 실행 의미론을 구현한다.
 * 생성 후 reset()을 호출하여 빈 스택으로 초기화한다.
 * 실행 컨텍스트: core_t::initilizeSIMTStack()에서 SM 초기화 시 호출.
 *
 * 호출 체인:
 *   core_t::initilizeSIMTStack() → [이 함수]
 */
simt_stack::simt_stack(unsigned wid, unsigned warpSize, class gpgpu_sim *gpu) {
  m_warp_id = wid;     /* [한국어] warp 번호 저장 — 디버그 출력 및 식별에 사용 */
  m_warp_size = warpSize; /* [한국어] warp 크기 저장 (일반적으로 32) */
  m_gpu = gpu;         /* [한국어] 최상위 시뮬레이터 포인터 저장 — 사이클 카운터 및 통계 접근 */
  reset();             /* [한국어] 스택을 빈 상태로 초기화 */
}

/*
 * [한국어]
 * simt_stack::reset - SIMT 스택 내용 전체 초기화
 *
 * @return: 없음.
 *
 * m_stack 벡터를 완전히 클리어한다.
 * 커널 launch 시 또는 resume 전에 호출하여 새로운 실행 시작을 위해 준비.
 */
void simt_stack::reset() { m_stack.clear(); } /* [한국어] 스택 벡터 전체 클리어 */

/*
 * [한국어]
 * simt_stack::launch - 커널 launch 시 SIMT 스택의 초기 진입점 설정
 *
 * @start_pc: 커널의 시작 PC (PTX 함수의 첫 명령어 주소).
 * @active_mask: 초기 활성 스레드 마스크. 일반적으로 모든 32비트가 1 (전체 활성).
 * @return: 없음.
 *
 * 새 커널이 launch될 때 SIMT 스택을 초기 상태로 설정한다.
 * 초기 entry를 생성하여 스택에 push:
 *   - m_pc = start_pc (커널 시작 주소)
 *   - m_calldepth = 1 (함수 호출 깊이 1 = 커널 진입)
 *   - m_active_mask = active_mask (전체 스레드 활성)
 *   - m_type = STACK_ENTRY_TYPE_NORMAL (일반 실행, CALL이 아님)
 * 실행 컨텍스트: SM이 새 warp를 launch할 때 호출.
 *
 * 호출 체인:
 *   shader_core_ctx::init_warps() → [이 함수]
 */
void simt_stack::launch(address_type start_pc, const simt_mask_t &active_mask) {
  reset(); /* [한국어] 이전 실행 상태 초기화 */
  simt_stack_entry new_stack_entry; /* [한국어] 초기 스택 엔트리 생성 */
  new_stack_entry.m_pc = start_pc;  /* [한국어] 커널 시작 PC 설정 */
  new_stack_entry.m_calldepth = 1;  /* [한국어] 호출 깊이 1 = 커널 최상위 레벨 */
  new_stack_entry.m_active_mask = active_mask; /* [한국어] 초기 활성 마스크 설정 */
  new_stack_entry.m_type = STACK_ENTRY_TYPE_NORMAL; /* [한국어] 일반 실행 타입 (CALL 엔트리가 아님) */
  m_stack.push_back(new_stack_entry); /* [한국어] 초기 엔트리를 스택에 push */
}

/*
 * [한국어]
 * simt_stack::resume - 체크포인트 파일에서 SIMT 스택 상태 복원
 *
 * @fname: SIMT 스택 체크포인트 파일 경로. print_checkpoint()가 저장한 형식.
 * @return: 없음. 파일 오픈 실패 시 assert로 중단.
 *
 * 체크포인트에서 시뮬레이션을 재개할 때 각 warp의 SIMT 스택 상태를 복원한다.
 * 파일 포맷 (print_checkpoint()의 출력 형식과 동일):
 *   각 라인: "<active_mask 비트 0~31 공백 구분> <PC> <calldepth> <recvg_pc> <branch_div_cycle> <type(0=NORMAL,1=CALL)> <warp_id> <warp_size>"
 * 각 라인이 스택의 하나의 엔트리에 해당하며, 파일 순서대로 m_stack에 push_back.
 * 실행 컨텍스트: 시뮬레이터 resume 초기화 단계.
 *
 * 호출 체인:
 *   shader_core_ctx::resume_kernel() → [이 함수]
 */
void simt_stack::resume(char *fname) {
  reset(); /* [한국어] 현재 스택 내용 초기화 */

  FILE *fp2 = fopen(fname, "r"); /* [한국어] 체크포인트 파일 읽기 모드 오픈 */
  assert(fp2 != NULL);           /* [한국어] 파일 오픈 실패 시 중단 */

  char line[200]; /* or other suitable maximum line size */

  while (fgets(line, sizeof line, fp2) != NULL) /* read a line */
  {
    simt_stack_entry new_stack_entry; /* [한국어] 복원할 스택 엔트리 */
    char *pch;
    pch = strtok(line, " "); /* [한국어] 첫 번째 토큰 (active_mask 첫 비트) */
    for (unsigned j = 0; j < m_warp_size; j++) {
      /* [한국어] warp_size(32)개의 active_mask 비트를 순서대로 복원 */
      if (pch[0] == '1')
        new_stack_entry.m_active_mask.set(j);   /* [한국어] '1'이면 해당 비트 활성화 */
      else
        new_stack_entry.m_active_mask.reset(j); /* [한국어] '0'이면 비활성화 */
      pch = strtok(NULL, " "); /* [한국어] 다음 토큰으로 이동 */
    }

    new_stack_entry.m_pc = atoi(pch); /* [한국어] PC 값 복원 */
    pch = strtok(NULL, " ");
    new_stack_entry.m_calldepth = atoi(pch); /* [한국어] 호출 깊이 복원 */
    pch = strtok(NULL, " ");
    new_stack_entry.m_recvg_pc = atoi(pch); /* [한국어] 재합류(reconvergence) PC 복원 */
    pch = strtok(NULL, " ");
    new_stack_entry.m_branch_div_cycle = atoi(pch); /* [한국어] 분기 발생 사이클 복원 */
    pch = strtok(NULL, " ");
    if (pch[0] == '0')
      new_stack_entry.m_type = STACK_ENTRY_TYPE_NORMAL; /* [한국어] 일반 실행 타입 */
    else
      new_stack_entry.m_type = STACK_ENTRY_TYPE_CALL;   /* [한국어] 함수 호출 타입 */
    m_stack.push_back(new_stack_entry); /* [한국어] 복원된 엔트리를 스택에 추가 */
  }
  fclose(fp2); /* [한국어] 파일 핸들 반환 */
}

/*
 * [한국어]
 * simt_stack::get_active_mask - 현재 실행할 스레드들의 활성 마스크 반환
 *
 * @return: 스택 top 엔트리의 active_mask 참조.
 *          32비트 bitset으로 각 비트가 해당 스레드 레인의 활성 여부를 나타냄.
 *
 * fetch 단계에서 이 warp를 실행할 스레드 집합을 결정하는 데 사용.
 * 스택 top이 현재 실행 중인 경로의 active_mask이다.
 * 실행 컨텍스트: warp scheduler의 fetch 단계.
 *
 * 호출 체인:
 *   core_t::getExecuteWarp() → [이 함수]
 */
const simt_mask_t &simt_stack::get_active_mask() const {
  assert(m_stack.size() > 0); /* [한국어] 스택이 비어있으면 논리 오류 */
  return m_stack.back().m_active_mask; /* [한국어] 스택 top의 active_mask 반환 (현재 실행 경로) */
}

/*
 * [한국어]
 * simt_stack::get_pdom_stack_top_info - 스택 top에서 PC와 재합류 PC 반환
 *
 * @pc: [출력] 스택 top의 현재 PC. 다음에 fetch할 명령어 주소.
 * @rpc: [출력] 스택 top의 재합류 PC(RPC). 분기 후 재합류할 주소.
 *       RPC가 (unsigned)-1이면 아직 분기 없음(재합류점 미설정).
 * @return: 없음.
 *
 * fetch 단계에서 이 warp가 다음에 실행할 PC와 SIMT 재합류 지점을 가져온다.
 * warp scheduler가 이 정보로 instruction fetch를 수행한다.
 * 실행 컨텍스트: warp scheduler의 fetch 단계.
 *
 * 호출 체인:
 *   shader_core_ctx::fetch() → [이 함수]
 *   core_t::getExecuteWarp() → [이 함수]
 */
void simt_stack::get_pdom_stack_top_info(unsigned *pc, unsigned *rpc) const {
  assert(m_stack.size() > 0); /* [한국어] 스택이 비어있으면 논리 오류 */
  *pc = m_stack.back().m_pc;       /* [한국어] 스택 top의 PC (다음 fetch 주소) */
  *rpc = m_stack.back().m_recvg_pc; /* [한국어] 스택 top의 재합류 PC */
}

/*
 * [한국어]
 * simt_stack::get_rp - 스택 top의 재합류 PC만 반환
 *
 * @return: 현재 실행 경로의 재합류 PC (reconvergence PC).
 *
 * update() 내에서 새 엔트리의 재합류 PC를 계산할 때 참조.
 * (unsigned)-1이면 재합류점이 아직 설정되지 않은 상태.
 */
unsigned simt_stack::get_rp() const {
  assert(m_stack.size() > 0); /* [한국어] 스택이 비어있으면 논리 오류 */
  return m_stack.back().m_recvg_pc; /* [한국어] 스택 top의 재합류 PC 반환 */
}

/*
 * [한국어]
 * simt_stack::print - SIMT 스택 상태를 사람이 읽기 쉬운 형식으로 출력
 *
 * @fout: 출력 파일 포인터 (stdout 또는 디버그 파일).
 * @return: 없음.
 *
 * 디버그 목적으로 현재 SIMT 스택의 모든 엔트리를 출력한다.
 * 출력 형식 (각 엔트리 1라인):
 *   "w<warp_id> <k> <active_mask 0/1 연속> pc: 0x<pc> rp: <rpc> tp: C/N cd: <calldepth> bd@<cycle> <PTX insn>"
 * 첫 번째 엔트리(k=0)에만 warp ID 출력, 나머지는 들여쓰기.
 * rp: ---- 는 재합류 PC가 없음(-1)을 의미.
 * tp: C = CALL, N = NORMAL 타입.
 * PTX 명령어는 ptx_print_insn()으로 해당 PC의 명령어를 출력.
 * 실행 컨텍스트: 디버그 출력 시 (시뮬레이션 도중 언제든 호출 가능).
 */
void simt_stack::print(FILE *fout) const {
  for (unsigned k = 0; k < m_stack.size(); k++) {
    /* [한국어] 스택의 모든 엔트리를 바닥(인덱스 0)부터 순서대로 출력 */
    simt_stack_entry stack_entry = m_stack[k]; /* [한국어] k번째 스택 엔트리 복사 */
    if (k == 0) {
      fprintf(fout, "w%02d %1u ", m_warp_id, k); /* [한국어] 첫 엔트리: warp ID와 엔트리 인덱스 출력 */
    } else {
      fprintf(fout, "    %1u ", k); /* [한국어] 이후 엔트리: warp ID 대신 공백으로 정렬 */
    }
    for (unsigned j = 0; j < m_warp_size; j++)
      fprintf(fout, "%c", (stack_entry.m_active_mask.test(j) ? '1' : '0'));
    /* [한국어] active_mask를 이진 문자열로 출력 (각 비트 = 해당 스레드 활성 여부) */
    fprintf(fout, " pc: 0x%03llx", stack_entry.m_pc); /* [한국어] PC를 16진수로 출력 */
    if (stack_entry.m_recvg_pc == (unsigned)-1) {
      /* [한국어] 재합류 PC가 설정되지 않은 경우 (분기 없는 최상위 레벨) */
      fprintf(fout, " rp: ---- tp: %s cd: %2u ",
              (stack_entry.m_type == STACK_ENTRY_TYPE_CALL ? "C" : "N"),
              /* [한국어] 타입: C=CALL(함수 호출), N=NORMAL(일반 분기) */
              stack_entry.m_calldepth); /* [한국어] 호출 깊이 */
    } else {
      /* [한국어] 재합류 PC가 있는 경우 (분기 진행 중) */
      fprintf(fout, " rp: %4llu tp: %s cd: %2u ", stack_entry.m_recvg_pc,
              /* [한국어] 재합류 PC 값 출력 */
              (stack_entry.m_type == STACK_ENTRY_TYPE_CALL ? "C" : "N"),
              stack_entry.m_calldepth);
    }
    if (stack_entry.m_branch_div_cycle != 0) {
      /* [한국어] 분기가 발생한 사이클이 기록되어 있으면 출력 */
      fprintf(fout, " bd@%6u ", (unsigned)stack_entry.m_branch_div_cycle);
      /* [한국어] bd@ = branch divergence at cycle 숫자 */
    } else {
      fprintf(fout, " "); /* [한국어] 분기 사이클 없음 — 공백으로 정렬 */
    }
    m_gpu->gpgpu_ctx->func_sim->ptx_print_insn(stack_entry.m_pc, fout);
    /* [한국어] 해당 PC의 PTX 명령어를 기능 시뮬레이터를 통해 출력 */
    fprintf(fout, "\n"); /* [한국어] 엔트리 출력 완료 후 줄바꿈 */
  }
}

/*
 * [한국어]
 * simt_stack::print_checkpoint - 체크포인트 저장용 SIMT 스택 직렬화 출력
 *
 * @fout: 체크포인트 파일 포인터.
 * @return: 없음.
 *
 * resume() 함수가 복원할 수 있는 형식으로 SIMT 스택 상태를 파일에 기록한다.
 * 각 라인(= 각 스택 엔트리) 형식:
 *   "<비트0> <비트1> ... <비트31> <pc> <calldepth> <recvg_pc> <branch_div_cycle> <type(0 or 1)> <warp_id> <warp_size>"
 * resume()의 파싱 코드와 완전히 대응되는 형식이어야 한다.
 * 실행 컨텍스트: 체크포인트 저장 단계.
 *
 * 호출 체인:
 *   shader_core_ctx::store_checkpoint() → [이 함수]
 */
void simt_stack::print_checkpoint(FILE *fout) const {
  for (unsigned k = 0; k < m_stack.size(); k++) {
    /* [한국어] 스택의 모든 엔트리를 순서대로 직렬화 */
    simt_stack_entry stack_entry = m_stack[k]; /* [한국어] k번째 엔트리 복사 */

    for (unsigned j = 0; j < m_warp_size; j++)
      fprintf(fout, "%c ", (stack_entry.m_active_mask.test(j) ? '1' : '0'));
    /* [한국어] active_mask 비트를 '1'/'0' 공백 구분으로 출력 */
    fprintf(fout, "%llu %d %llu %lld %d ", stack_entry.m_pc,
            stack_entry.m_calldepth, stack_entry.m_recvg_pc,
            stack_entry.m_branch_div_cycle, stack_entry.m_type);
    /* [한국어] PC, 호출깊이, 재합류PC, 분기사이클, 타입을 공백 구분으로 출력 */
    fprintf(fout, "%d %d\n", m_warp_id, m_warp_size);
    /* [한국어] warp ID와 warp 크기 출력 후 줄바꿈 */
  }
}

/*
 * [한국어]
 * simt_stack::update - warp 분기/재합류/CALL/RET 처리를 통한 SIMT 스택 갱신
 *
 * @thread_done: 이 명령어 실행 후 완료된(종료된) 스레드의 마스크.
 * @next_pc: 각 스레드가 다음에 실행할 PC 벡터 (크기 = warp_size).
 *            스레드가 완료됐으면 해당 위치는 무시됨.
 * @recvg_pc: 컴파일러가 계산한 이 분기의 post-dominator(재합류) PC.
 *             분기가 없으면 사용되지 않음.
 * @next_inst_op: 방금 실행된 명령어의 연산 종류 (CALL_OPS, RET_OPS, 기타).
 * @next_inst_size: 방금 실행된 명령어의 바이트 크기 (not-taken PC 계산에 사용).
 * @next_inst_pc: 방금 실행된 명령어의 PC (스택 top PC와 일치해야 함).
 * @return: 없음.
 *
 * GPGPU-Sim의 pdom(Post-dominator) SIMT 스택 갱신 알고리즘을 구현한다.
 * 매 명령어 실행 후 호출되어 다음 사이클에 실행할 스레드 집합을 결정한다.
 *
 * 처리 시나리오:
 *   [수렴 실행] 모든 스레드가 같은 next_pc → 스택 변화 없이 PC만 갱신
 *   [분기 발생] 스레드가 taken/not-taken 두 경로로 나뉨:
 *     1) 현재 top을 재합류 엔트리로 변환 (pc = recvg_pc)
 *     2) not-taken 경로를 위한 엔트리 push (먼저 처리)
 *     3) taken 경로를 위한 엔트리 push (나중에 처리)
 *   [재합류] next_pc가 top_recvg_pc와 같으면 해당 경로 엔트리 제거
 *   [CALL] STACK_ENTRY_TYPE_CALL 엔트리를 push
 *   [RET] STACK_ENTRY_TYPE_CALL 엔트리를 pop, 반환 후 재합류 확인
 *
 * 알고리즘 흐름:
 *   1단계: top_active_mask의 스레드들을 next_pc별로 그룹화
 *          (divergent_paths: next_pc → active_mask)
 *   2단계: not_taken_pc (= next_inst_pc + next_inst_size) 우선 처리
 *   3단계: 각 경로에 대해 CALL/RET 특수 처리 또는 분기/수렴 스택 조작
 *   4단계: 마지막 push된 더미 엔트리 pop
 *
 * 실행 컨텍스트: core_t::updateSIMTStack()을 통해 명령어 실행 후 호출.
 *   단일 시뮬 스레드에서 실행.
 *
 * 호출 체인:
 *   core_t::updateSIMTStack() → [이 함수]
 */
void simt_stack::update(simt_mask_t &thread_done, addr_vector_t &next_pc,
                        address_type recvg_pc, op_type next_inst_op,
                        unsigned next_inst_size, address_type next_inst_pc) {
  assert(m_stack.size() > 0); /* [한국어] 스택이 비어있으면 논리 오류 */

  assert(next_pc.size() == m_warp_size); /* [한국어] next_pc 벡터 크기가 warp_size와 일치해야 함 */

  simt_mask_t top_active_mask = m_stack.back().m_active_mask;
  /* [한국어] 스택 top의 active_mask 복사 — 이 명령어를 실행한 스레드들 */
  address_type top_recvg_pc = m_stack.back().m_recvg_pc;
  /* [한국어] 스택 top의 재합류 PC — 이 경로가 수렴해야 하는 지점 */
  address_type top_pc =
      m_stack.back().m_pc;  // the pc of the instruction just executed
  /* [한국어] 방금 실행된 명령어의 PC (스택 top PC) */
  stack_entry_type top_type = m_stack.back().m_type;
  /* [한국어] 스택 top 타입 (NORMAL 또는 CALL) */
  assert(top_pc == next_inst_pc); /* [한국어] 실행된 명령어가 스택 top PC와 일치해야 함 */
  assert(top_active_mask.any()); /* [한국어] 최소 1개 스레드는 활성 상태여야 함 */

  const address_type null_pc = -1; /* [한국어] 유효하지 않은 PC 값 (완료된 스레드 표시) */
  bool warp_diverged = false;      /* [한국어] 이 명령어에서 warp 분기가 발생했는지 여부 */
  address_type new_recvg_pc = null_pc; /* [한국어] 분기 발생 시 새로 설정할 재합류 PC */
  unsigned num_divergent_paths = 0;    /* [한국어] 분기 후 서로 다른 경로의 수 */

  std::map<address_type, simt_mask_t> divergent_paths;
  /* [한국어] next_pc별 스레드 그룹: PC → 해당 PC로 가는 스레드들의 마스크 */
  while (top_active_mask.any()) {
    /* [한국어] active_mask에 남은 스레드를 동일 next_pc끼리 그룹화 */
    // extract a group of threads with the same next PC among the active threads
    // in the warp
    address_type tmp_next_pc = null_pc; /* [한국어] 이번 루프에서 처리할 PC 값 (처음엔 무효) */
    simt_mask_t tmp_active_mask;         /* [한국어] 이 PC로 가는 스레드들의 마스크 */
    for (int i = m_warp_size - 1; i >= 0; i--) {
      /* [한국어] 역방향(높은 ID부터)으로 스레드 순회 — 그룹에서 하나씩 추출 */
      if (top_active_mask.test(i)) {  // is this thread active?
        if (thread_done.test(i)) {
          top_active_mask.reset(i);  // remove completed thread from active mask
          /* [한국어] 완료된 스레드는 active_mask에서 제거 — 더 이상 추적 불필요 */
        } else if (tmp_next_pc == null_pc) {
          /* [한국어] 이 그룹의 기준 PC가 아직 없으면 이 스레드의 next_pc를 기준으로 설정 */
          tmp_next_pc = next_pc[i];    /* [한국어] 이 그룹의 next PC로 설정 */
          tmp_active_mask.set(i);      /* [한국어] 이 스레드를 그룹에 추가 */
          top_active_mask.reset(i);    /* [한국어] 처리된 스레드를 working mask에서 제거 */
        } else if (tmp_next_pc == next_pc[i]) {
          /* [한국어] 이 스레드가 같은 next_pc로 가면 동일 그룹에 합류 */
          tmp_active_mask.set(i);   /* [한국어] 동일 경로 스레드를 그룹에 추가 */
          top_active_mask.reset(i); /* [한국어] 처리된 스레드 제거 */
        }
        /* [한국어] 다른 next_pc를 가진 스레드는 다음 while 루프 반복에서 처리됨 */
      }
    }

    if (tmp_next_pc == null_pc) {
      assert(!top_active_mask.any());  // all threads done
      /* [한국어] 남은 활성 스레드가 없고 유효한 PC 그룹도 없으면 모든 스레드 완료 */
      continue;
    }

    divergent_paths[tmp_next_pc] = tmp_active_mask; /* [한국어] 이 경로를 divergent_paths에 기록 */
    num_divergent_paths++; /* [한국어] 발견된 경로 수 증가 */
  }

  address_type not_taken_pc = next_inst_pc + next_inst_size;
  /* [한국어] not-taken(분기 미실행) 경로의 PC = 현재 명령어 주소 + 명령어 크기 (순차 실행) */
  assert(num_divergent_paths <= 2);
  /* [한국어] pdom 모델: 분기는 최대 2개 경로(taken/not-taken)만 가능 */
  for (unsigned i = 0; i < num_divergent_paths; i++) {
    /* [한국어] 각 분기 경로를 순서대로 처리: not-taken 경로를 먼저, taken 경로를 나중에 */
    address_type tmp_next_pc = null_pc; /* [한국어] 이번에 처리할 경로의 PC */
    simt_mask_t tmp_active_mask;
    tmp_active_mask.reset();
    if (divergent_paths.find(not_taken_pc) != divergent_paths.end()) {
      /* [한국어] not-taken 경로가 있으면 먼저 처리 (스택에 먼저 넣어야 나중에 팝됨) */
      assert(i == 0); /* [한국어] not-taken 경로는 반드시 첫 번째로 처리 */
      tmp_next_pc = not_taken_pc;
      tmp_active_mask = divergent_paths[tmp_next_pc];
      divergent_paths.erase(tmp_next_pc); /* [한국어] 처리된 경로를 맵에서 제거 */
    } else {
      /* [한국어] not-taken 경로가 없거나 이미 처리됨 → 남은 경로 처리 */
      std::map<address_type, simt_mask_t>::iterator it =
          divergent_paths.begin();
      tmp_next_pc = it->first;
      tmp_active_mask = divergent_paths[tmp_next_pc];
      divergent_paths.erase(tmp_next_pc); /* [한국어] 처리된 경로 제거 */
    }

    // HANDLE THE SPECIAL CASES FIRST
    if (next_inst_op == CALL_OPS) {
      // Since call is not a divergent instruction, all threads should have
      // executed a call instruction
      assert(num_divergent_paths == 1);
      /* [한국어] CALL 명령어는 분기가 아니므로 모든 스레드가 동일한 함수로 jump */

      simt_stack_entry new_stack_entry; /* [한국어] 함수 호출을 위한 새 스택 엔트리 */
      new_stack_entry.m_pc = tmp_next_pc;          /* [한국어] 호출된 함수의 시작 PC */
      new_stack_entry.m_active_mask = tmp_active_mask; /* [한국어] 호출 시 활성 스레드 마스크 */
      new_stack_entry.m_branch_div_cycle =
          m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
      /* [한국어] CALL이 발생한 현재 사이클 기록 */
      new_stack_entry.m_type = STACK_ENTRY_TYPE_CALL; /* [한국어] CALL 타입으로 설정 */
      m_stack.push_back(new_stack_entry); /* [한국어] CALL 엔트리를 스택에 push */
      return; /* [한국어] CALL은 분기 처리 불필요 — 즉시 반환 */
    } else if (next_inst_op == RET_OPS && top_type == STACK_ENTRY_TYPE_CALL) {
      // pop the CALL Entry
      assert(num_divergent_paths == 1);
      /* [한국어] RET 명령어이고 스택 top이 CALL 엔트리인 경우 — 함수 반환 처리 */
      m_stack.pop_back(); /* [한국어] CALL 스택 엔트리 제거 (함수에서 복귀) */

      assert(m_stack.size() > 0); /* [한국어] CALL 엔트리를 제거한 후에도 스택이 남아있어야 함 */
      m_stack.back().m_pc = tmp_next_pc;  // set the PC of the stack top entry
                                          // to return PC from  the call stack;
      /* [한국어] 반환 후 실행할 PC (return address)로 스택 top의 PC 갱신 */
      // Check if the New top of the stack is reconverging
      if (tmp_next_pc == m_stack.back().m_recvg_pc &&
          m_stack.back().m_type != STACK_ENTRY_TYPE_CALL) {
        /* [한국어] 반환 PC가 스택 top의 재합류 PC와 같으면 해당 분기도 재합류 완료 */
        assert(m_stack.back().m_type == STACK_ENTRY_TYPE_NORMAL);
        m_stack.pop_back(); /* [한국어] 재합류 완료된 NORMAL 엔트리도 제거 */
      }
      return; /* [한국어] RET 처리 완료 */
    }

    // discard the new entry if its PC matches with reconvergence PC
    // that automatically reconverges the entry
    // If the top stack entry is CALL, dont reconverge.
    if (tmp_next_pc == top_recvg_pc && (top_type != STACK_ENTRY_TYPE_CALL))
      continue;
    /* [한국어] 이 경로의 next_pc가 현재 재합류 PC와 같으면 이 경로는 이미 수렴됨
     * → 별도 스택 엔트리 불필요, 건너뜀. CALL 타입이면 재합류 처리 안 함 */

    // this new entry is not converging
    // if this entry does not include thread from the warp, divergence occurs
    if ((num_divergent_paths > 1) && !warp_diverged) {
      /* [한국어] 두 번째 경로가 있고 아직 분기 처리가 시작되지 않은 경우 → 분기 발생 */
      warp_diverged = true;
      // modify the existing top entry into a reconvergence entry in the pdom
      // stack
      /* [한국어] 현재 top 엔트리를 재합류 대기 엔트리로 변환 */
      new_recvg_pc = recvg_pc; /* [한국어] 컴파일러가 계산한 post-dominator PC를 재합류 PC로 설정 */
      if (new_recvg_pc != top_recvg_pc) {
        /* [한국어] 새 재합류 PC가 기존 top의 재합류 PC와 다른 경우 → 스택 수정 필요 */
        m_stack.back().m_pc = new_recvg_pc;
        /* [한국어] 현재 top의 PC를 재합류 PC로 변경 (이 엔트리가 재합류 대기 엔트리가 됨) */
        m_stack.back().m_branch_div_cycle =
            m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
        /* [한국어] 분기 발생 사이클 기록 */

        m_stack.push_back(simt_stack_entry()); /* [한국어] 분기 경로 처리를 위한 빈 엔트리 push */
      }
    }

    // discard the new entry if its PC matches with reconvergence PC
    if (warp_diverged && tmp_next_pc == new_recvg_pc) continue;
    /* [한국어] 이 경로의 next_pc가 새 재합류 PC와 같으면 이미 수렴됨 → 건너뜀 */

    // update the current top of pdom stack
    m_stack.back().m_pc = tmp_next_pc;          /* [한국어] 이 분기 경로의 PC를 스택 top에 설정 */
    m_stack.back().m_active_mask = tmp_active_mask; /* [한국어] 이 경로를 실행하는 스레드 마스크 설정 */
    if (warp_diverged) {
      /* [한국어] 분기 발생 시 새 경로 엔트리: calldepth=0(분기 생성), recvg_pc=새 재합류 PC */
      m_stack.back().m_calldepth = 0;           /* [한국어] 분기로 생성된 엔트리는 호출 깊이 0 */
      m_stack.back().m_recvg_pc = new_recvg_pc; /* [한국어] 이 경로의 재합류 PC = 새 재합류 PC */
    } else {
      /* [한국어] 분기 없는 수렴 실행: 기존 재합류 PC 유지 */
      m_stack.back().m_recvg_pc = top_recvg_pc; /* [한국어] 재합류 PC는 변경 없이 유지 */
    }

    m_stack.push_back(simt_stack_entry());
    /* [한국어] 다음 경로를 위한 빈 엔트리 push
     * 루프 종료 후 마지막에 pop_back()으로 제거됨 */
  }
  assert(m_stack.size() > 0); /* [한국어] 루프 종료 후 스택이 비어있으면 오류 */
  m_stack.pop_back();
  /* [한국어] 마지막 루프에서 push된 더미 빈 엔트리 제거
   * 알고리즘 구조상 경로 처리마다 빈 엔트리를 push하고 마지막에 하나를 pop */

  if (warp_diverged) {
    m_gpu->gpgpu_ctx->stats->ptx_file_line_stats_add_warp_divergence(top_pc, 1);
    /* [한국어] 분기 발생 시 PTX 파일·라인별 warp divergence 통계 기록 */
  }
}

/*
 * [한국어]
 * core_t::execute_warp_inst_t - warp의 활성 스레드들에 대해 PTX 명령어 기능 실행
 *
 * @inst: 실행할 warp 명령어 (active_mask, 스레드별 주소 등 포함).
 * @warpId: 이 명령어를 실행하는 warp 번호.
 *          (unsigned(-1)이면 inst.warp_id()에서 자동 획득)
 * @return: 없음.
 *
 * warp의 각 활성 스레드 레인에 대해 PTX 기능 시뮬레이터(ptx_exec_inst)를 실행.
 * 타이밍 시뮬과 기능 시뮬을 연결하는 핵심 연결점이다.
 * 기능 실행 후 checkExecutionStatusAndUpdate(virtual)를 호출하여 스레드 완료·
 * exception 처리 등을 서브클래스(shader_core_ctx)에서 처리하게 한다.
 * 실행 컨텍스트: 타이밍 시뮬의 execute 단계에서 각 warp 명령어 처리 시.
 *
 * 호출 체인:
 *   shader_core_ctx::execute() → [이 함수]
 *   → m_thread[tid]->ptx_exec_inst() (기능 시뮬)
 *   → checkExecutionStatusAndUpdate() (가상 함수, shader_core_ctx 구현)
 */
void core_t::execute_warp_inst_t(warp_inst_t &inst, unsigned warpId) {
  for (unsigned t = 0; t < m_warp_size; t++) {
    /* [한국어] warp의 모든 레인(0 ~ warp_size-1)을 순회 */
    if (inst.active(t)) {
      /* [한국어] 이 레인이 active_mask에서 활성화된 경우에만 기능 실행 */
      if (warpId == (unsigned(-1))) warpId = inst.warp_id();
      /* [한국어] warpId가 미지정(-1)이면 명령어 자체에서 warp ID 획득 */
      unsigned tid = m_warp_size * warpId + t;
      /* [한국어] 하드웨어 스레드 ID = warp 번호 × warp 크기 + 레인 번호 */
      m_thread[tid]->ptx_exec_inst(inst, t);
      /* [한국어] 이 스레드(레인 t)에 대해 PTX 명령어 기능 실행
       * inst: 공유되는 명령어 정보, t: 이 스레드 레인의 로컬 ID
       * 실행 결과: 레지스터 갱신, 메모리 주소 계산(memreqaddr 설정), 완료 상태 변경 */

      // virtual function
      checkExecutionStatusAndUpdate(inst, t, tid);
      /* [한국어] 스레드 실행 상태 확인 및 갱신 (가상 함수 — shader_core_ctx에서 구현)
       * 스레드 완료 여부, exception, barrier 등을 처리 */
    }
  }
}

/*
 * [한국어]
 * core_t::ptx_thread_done - 특정 하드웨어 스레드가 완료되었는지 확인
 *
 * @hw_thread_id: 확인할 하드웨어 스레드 ID (= warp_id × warp_size + lane_id).
 * @return: true면 스레드 완료(또는 초기화 안 됨). false면 실행 중.
 *
 * m_thread 배열에서 해당 스레드의 포인터가 NULL이거나 is_done()이 true면
 * 완료된 것으로 판단한다. updateSIMTStack()에서 thread_done 마스크 생성 시 사용.
 */
bool core_t::ptx_thread_done(unsigned hw_thread_id) const {
  return ((m_thread[hw_thread_id] == NULL) ||
          m_thread[hw_thread_id]->is_done());
  /* [한국어] NULL: 스레드가 아직 초기화 안 됨 또는 이미 해제됨.
   * is_done(): PTX 기능 시뮬에서 이 스레드가 실행 완료되었음을 표시 */
}

/*
 * [한국어]
 * core_t::updateSIMTStack - 명령어 실행 후 SIMT 스택 갱신
 *
 * @warpId: 갱신할 warp의 번호.
 * @inst: 방금 실행된 warp 명령어 (op, isize, pc, reconvergence_pc 포함).
 * @return: 없음.
 *
 * execute_warp_inst_t() 실행 후 각 스레드의 next PC와 완료 상태를 수집하여
 * SIMT 스택(simt_stack::update())에 전달한다. 이 함수가 타이밍 시뮬과
 * 기능 시뮬(m_thread 상태) 사이의 연결 다리 역할을 한다.
 *
 * 처리 내용:
 *   1) 각 레인의 완료 여부 확인 → thread_done 마스크 생성
 *   2) 완료되지 않은 레인의 next PC 수집 (m_thread[tid]->get_pc())
 *   3) RECONVERGE_RETURN_PC이면 실제 반환 PC를 get_return_pc()로 조회
 *   4) 수집한 정보로 simt_stack::update() 호출
 * 실행 컨텍스트: 타이밍 시뮬의 execute 단계.
 *
 * 호출 체인:
 *   shader_core_ctx::execute() → [이 함수] → m_simt_stack[warpId]->update()
 */
void core_t::updateSIMTStack(unsigned warpId, warp_inst_t *inst) {
  simt_mask_t thread_done; /* [한국어] 이 명령어 후 완료된 스레드들의 마스크 */
  addr_vector_t next_pc;   /* [한국어] 각 스레드의 다음 실행 PC 벡터 (크기 = warp_size) */
  unsigned wtid = warpId * m_warp_size; /* [한국어] 이 warp의 첫 번째 스레드의 하드웨어 스레드 ID */
  for (unsigned i = 0; i < m_warp_size; i++) {
    /* [한국어] warp의 각 레인(0 ~ warp_size-1)에 대해 상태 수집 */
    if (ptx_thread_done(wtid + i)) {
      /* [한국어] 이 스레드가 완료된 경우 */
      thread_done.set(i);                       /* [한국어] thread_done 마스크에 표시 */
      next_pc.push_back((address_type)-1);      /* [한국어] 완료된 스레드의 next PC는 무효값(-1) */
    } else {
      /* [한국어] 이 스레드가 아직 실행 중인 경우 */
      if (inst->reconvergence_pc == RECONVERGE_RETURN_PC)
        inst->reconvergence_pc = get_return_pc(m_thread[wtid + i]);
      /* [한국어] RECONVERGE_RETURN_PC 플래그: 재합류 PC가 함수 반환 주소와 같음을 의미.
       * 이 경우 해당 스레드의 실제 반환 PC를 기능 시뮬 상태에서 조회 */
      next_pc.push_back(m_thread[wtid + i]->get_pc());
      /* [한국어] 기능 시뮬에서 이 스레드가 다음에 실행할 PC 획득 */
    }
  }
  m_simt_stack[warpId]->update(thread_done, next_pc, inst->reconvergence_pc,
                               inst->op, inst->isize, inst->pc);
  /* [한국어] 수집된 정보로 SIMT 스택 갱신
   * thread_done: 완료된 스레드 마스크
   * next_pc: 각 스레드의 다음 PC
   * inst->reconvergence_pc: 컴파일러가 계산한 재합류 PC
   * inst->op: 명령어 타입 (CALL_OPS, RET_OPS 등)
   * inst->isize: 명령어 바이트 크기
   * inst->pc: 현재 명령어 PC */
}

//! Get the warp to be executed using the data taken form the SIMT stack
/*
 * [한국어]
 * core_t::getExecuteWarp - SIMT 스택 top에서 실행할 warp 명령어 생성
 *
 * @warpId: 가져올 warp의 번호.
 * @return: SIMT 스택 top의 PC에서 fetch된 warp_inst_t 객체 (값 복사).
 *          active_mask는 SIMT 스택 top의 마스크로 설정됨.
 *
 * fetch 단계에서 이 warp가 다음에 실행할 명령어를 PTX IR에서 가져오고,
 * SIMT 스택 top의 active_mask를 적용한다.
 * ptx_fetch_inst(pc)로 PTX IR에서 명령어를 가져와 값 복사한 후,
 * set_active()로 현재 실행해야 할 스레드 마스크를 설정한다.
 * 실행 컨텍스트: 타이밍 시뮬의 fetch 단계.
 *
 * 호출 체인:
 *   shader_core_ctx::fetch() → [이 함수]
 */
warp_inst_t core_t::getExecuteWarp(unsigned warpId) {
  unsigned pc, rpc; /* [한국어] SIMT 스택 top에서 가져올 PC와 재합류 PC */
  m_simt_stack[warpId]->get_pdom_stack_top_info(&pc, &rpc);
  /* [한국어] SIMT 스택 top에서 현재 PC와 재합류 PC 획득 */
  warp_inst_t wi = *(m_gpu->gpgpu_ctx->ptx_fetch_inst(pc));
  /* [한국어] PC에 해당하는 PTX 명령어를 기능 시뮬에서 가져와 값 복사
   * ptx_fetch_inst()는 PTX IR 캐시에서 해당 PC의 warp_inst_t 포인터 반환 */
  wi.set_active(m_simt_stack[warpId]->get_active_mask());
  /* [한국어] SIMT 스택 top의 active_mask를 이 명령어에 적용
   * 분기 상태에 따라 일부 스레드만 이 명령어를 실행하게 됨 */
  return wi; /* [한국어] active_mask가 설정된 warp 명령어 반환 (값 복사) */
}

/*
 * [한국어]
 * core_t::deleteSIMTStack - 모든 warp의 SIMT 스택 메모리 해제
 *
 * @return: 없음.
 *
 * SM이 소멸할 때 또는 커널 실행 완료 시 모든 warp의 SIMT 스택을 해제한다.
 * m_simt_stack 배열이 NULL이 아닌 경우에만 처리한다.
 * 실행 컨텍스트: shader_core_ctx 소멸 시.
 */
void core_t::deleteSIMTStack() {
  if (m_simt_stack) { /* [한국어] SIMT 스택 배열이 초기화되어 있는 경우에만 처리 */
    for (unsigned i = 0; i < m_warp_count; ++i) delete m_simt_stack[i];
    /* [한국어] 각 warp의 simt_stack 객체 해제 */
    delete[] m_simt_stack; /* [한국어] 포인터 배열 자체 해제 */
    m_simt_stack = NULL;   /* [한국어] 댕글링 포인터 방지를 위해 NULL로 초기화 */
  }
}

/*
 * [한국어]
 * core_t::initilizeSIMTStack - 이 SM의 모든 warp를 위한 SIMT 스택 배열 초기화
 *
 * @warp_count: 이 SM에서 동시에 실행 가능한 최대 warp 수.
 * @warp_size: warp 크기 (일반적으로 32).
 * @return: 없음.
 *
 * SM이 새 커널을 받아 초기화할 때 각 warp마다 simt_stack 객체를 생성한다.
 * m_simt_stack은 simt_stack 포인터 배열로, 인덱스가 warp 번호에 대응한다.
 * 실행 컨텍스트: shader_core_ctx::init_warps() 또는 커널 할당 시.
 *
 * 호출 체인:
 *   shader_core_ctx::init() → [이 함수]
 */
void core_t::initilizeSIMTStack(unsigned warp_count, unsigned warp_size) {
  m_simt_stack = new simt_stack *[warp_count]; /* [한국어] warp_count개의 simt_stack 포인터 배열 동적 할당 */
  for (unsigned i = 0; i < warp_count; ++i)
    m_simt_stack[i] = new simt_stack(i, warp_size, m_gpu);
  /* [한국어] 각 warp(0 ~ warp_count-1)에 대해 simt_stack 객체 생성
   * i = warp ID, warp_size = 32, m_gpu = 최상위 시뮬레이터 포인터 */
  m_warp_size = warp_size;   /* [한국어] warp 크기 저장 */
  m_warp_count = warp_count; /* [한국어] 총 warp 수 저장 */
}

/*
 * [한국어]
 * core_t::get_pdom_stack_top_info - 특정 warp의 SIMT 스택 top 정보 반환
 *
 * @warpId: 정보를 가져올 warp 번호.
 * @pc: [출력] 스택 top의 현재 PC.
 * @rpc: [출력] 스택 top의 재합류 PC.
 * @return: 없음.
 *
 * core_t 외부에서 특정 warp의 SIMT 스택 top 정보를 간편하게 조회하는 래퍼.
 * 직접 m_simt_stack[warpId]->get_pdom_stack_top_info()를 호출하는 것과 동일.
 */
void core_t::get_pdom_stack_top_info(unsigned warpId, unsigned *pc,
                                     unsigned *rpc) const {
  m_simt_stack[warpId]->get_pdom_stack_top_info(pc, rpc);
  /* [한국어] 해당 warp의 SIMT 스택에서 top PC와 재합류 PC 조회 */
}
